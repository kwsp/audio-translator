"""CLI entry point for audio-translator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from audio_translator.pipeline import (
    synthesize_translated_transcript,
    translate_audio_to_audio,
    translate_text_to_audio,
    translate_transcript_to_audio,
)


_TTS_BACKENDS = ["gemini", "edge"]
_STT_BACKENDS = ["gemini"]

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".webm", ".aac", ".mp4"}
_TEXT_EXTENSIONS = {".txt", ".md"}


def _detect_input_type(
    path: str,
) -> Literal["audio", "transcript", "translated_transcript", "text"]:
    """Infer the pipeline entry point from the input path."""
    if path.startswith("http://") or path.startswith("https://"):
        return "audio"
    ext = Path(path).suffix.lower()
    if ext in _AUDIO_EXTENSIONS:
        return "audio"
    if ext in _TEXT_EXTENSIONS:
        return "text"
    if ext == ".json":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc
        if "source_lang" in data:
            return "translated_transcript"
        if "lang" in data:
            return "transcript"
        raise ValueError(
            f"{path} does not look like a Transcript or TranslatedTranscript "
            "(expected a top-level 'lang' or 'source_lang' key)"
        )
    raise ValueError(
        f"Cannot determine input type for '{path}'. "
        f"Use a recognised extension ({', '.join(sorted(_AUDIO_EXTENSIONS | _TEXT_EXTENSIONS | {'.json'}))})"
        " or a URL, or pass text via --text / stdin."
    )


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="audio-translator",
        description="Translate spoken audio between languages using Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input auto-detection:\n"
            "  audio file / URL  →  full STT → translate → TTS pipeline\n"
            "  .txt / .md file   →  read as plain text, then translate → TTS\n"
            "  transcript JSON   →  skip STT, translate → TTS\n"
            "  translated JSON   →  skip STT + translation, TTS only\n"
            "  --text / stdin    →  inline or piped plain text"
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Audio file/URL, .txt file, transcript JSON, or translated-transcript JSON. "
            "The pipeline stage is chosen automatically by file type. "
            "Omit to read plain text from stdin or use --text."
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        metavar="PASSAGE",
        help="Inline plain-text passage to translate (skips STT).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Output directory for all pipeline artifacts "
            "(transcript.json, translated_transcript.json, audio.mp3). "
            "Defaults to a directory named after the input file."
        ),
    )
    parser.add_argument(
        "--source-lang",
        default="English",
        help="Source language for text/stdin input (default: English).",
    )
    parser.add_argument(
        "--target-lang",
        default="Mandarin Chinese",
        help='Target language (default: "Mandarin Chinese").',
    )
    parser.add_argument(
        "--voice-map",
        default=None,
        help='JSON string mapping speaker names to voice profiles, e.g. \'{"Speaker 1":"Kore"}\'.',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--tts-backend",
        default="edge",
        choices=_TTS_BACKENDS,
        help="TTS backend to use: 'edge' (default, free, no API key) or 'gemini' (higher-fidelity voices).",
    )
    parser.add_argument(
        "--stt-backend",
        default="gemini",
        choices=_STT_BACKENDS,
        help="STT backend to use: 'gemini' (default).",
    )

    args = parser.parse_args(argv)

    # Resolve text source: --text > stdin > positional input.
    if args.input is None and args.text is None:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read()
        else:
            parser.error(
                "provide a positional input file/URL, --text PASSAGE, or pipe text via stdin"
            )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    voice_map = None
    if args.voice_map:
        try:
            voice_map = json.loads(args.voice_map)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid --voice-map JSON: {exc}", file=sys.stderr)
            sys.exit(1)

    # Select TTS backend.
    tts = None
    if args.tts_backend == "gemini":
        from audio_translator.backends.gemini.tts import GeminiTTS  # noqa: PLC0415
        tts = GeminiTTS()
    else:  # edge (default)
        from audio_translator.backends.edge.tts import EdgeTTS  # noqa: PLC0415
        tts = EdgeTTS()

    # Select STT backend (currently only Gemini supported, but ready for expansion).
    stt = None
    if args.stt_backend == "gemini":
        from audio_translator.backends.gemini.stt import GeminiSTT  # noqa: PLC0415
        stt = GeminiSTT()

    try:
        if args.text:
            result = translate_text_to_audio(
                text=args.text,
                output_dir=args.output_dir,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                voice_map=voice_map,
                tts=tts,
            )
        else:
            input_type = _detect_input_type(args.input)
            if input_type == "audio":
                result = translate_audio_to_audio(
                    input=args.input,
                    output_dir=args.output_dir,
                    target_lang=args.target_lang,
                    voice_map=voice_map,
                    stt=stt,
                    tts=tts,
                )
            elif input_type == "text":
                result = translate_text_to_audio(
                    text=Path(args.input).read_text(encoding="utf-8"),
                    output_dir=args.output_dir or Path(args.input).stem,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    voice_map=voice_map,
                    tts=tts,
                )
            elif input_type == "transcript":
                result = translate_transcript_to_audio(
                    transcript=args.input,
                    output_dir=args.output_dir,
                    target_lang=args.target_lang,
                    voice_map=voice_map,
                    tts=tts,
                )
            else:  # translated_transcript
                result = synthesize_translated_transcript(
                    translated_transcript=args.input,
                    output_dir=args.output_dir,
                    voice_map=voice_map,
                    tts=tts,
                )

        print(f"\nOutputs written to: {result.output_dir}/")
        print(f"  transcript:            {result.transcript.name}")
        print(f"  translated transcript: {result.translated_transcript.name}")
        print(f"  audio:                 {result.audio.name}")
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
