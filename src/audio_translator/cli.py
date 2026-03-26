"""CLI entry point for audio-translator."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from audio_translator.pipeline import (
    synthesize_translated_transcript,
    translate_audio_to_audio,
    translate_text_to_audio,
    translate_transcript_to_audio,
)


_TTS_BACKENDS = ["gemini", "edge"]
_STT_BACKENDS = ["gemini"]


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="audio-translator",
        description="Translate spoken audio between languages using Gemini.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Audio file path, URL, or transcript JSON (with --transcript). Not required when --text is used.",
    )
    parser.add_argument(
        "--text",
        default=None,
        metavar="PASSAGE",
        help="Plain-text passage to translate. Wrapped into a single female speaker transcript; skips STT.",
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
        help="Source language (default: English).",
    )
    parser.add_argument(
        "--target-lang",
        default="Mandarin Chinese",
        help='Target language (default: "Mandarin Chinese").',
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Treat input as a Transcript JSON file (skip STT only).",
    )
    parser.add_argument(
        "--translated-transcript",
        action="store_true",
        dest="translated_transcript",
        help="Treat input as a TranslatedTranscript JSON (skip STT + translation, TTS only).",
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
        default="gemini",
        choices=_TTS_BACKENDS,
        help="TTS backend to use: 'gemini' (default) or 'edge' (free, no API key).",
    )
    parser.add_argument(
        "--stt-backend",
        default="gemini",
        choices=_STT_BACKENDS,
        help="STT backend to use: 'gemini' (default).",
    )

    args = parser.parse_args(argv)

    # Read piped stdin as text input when no other text/input source is given.
    if args.input is None and args.text is None:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read()
        else:
            parser.error("provide a positional input file/URL, --text PASSAGE, or pipe text via stdin")

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
    if args.tts_backend == "edge":
        from audio_translator.backends.edge.tts import EdgeTTS  # noqa: PLC0415
        tts = EdgeTTS()

    # Select STT backend (currently only Gemini supported, but ready for expansion).
    stt = None
    if args.stt_backend == "gemini":
        from audio_translator.backends.gemini.stt import GeminiSTT  # noqa: PLC0415
        stt = GeminiSTT()

    try:
        if args.translated_transcript:
            result = synthesize_translated_transcript(
                translated_transcript=args.input,
                output_dir=args.output_dir,
                voice_map=voice_map,
                tts=tts,
            )
        elif args.text:
            result = translate_text_to_audio(
                text=args.text,
                output_dir=args.output_dir,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                voice_map=voice_map,
                tts=tts,
            )
        elif args.transcript:
            result = translate_transcript_to_audio(
                transcript=args.input,
                output_dir=args.output_dir,
                target_lang=args.target_lang,
                voice_map=voice_map,
                tts=tts,
            )
        else:
            result = translate_audio_to_audio(
                input=args.input,
                output_dir=args.output_dir,
                target_lang=args.target_lang,
                voice_map=voice_map,
                stt=stt,
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
