"""CLI entry point for audio-translator."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from audio_translator.pipeline import translate_audio


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="audio-translator",
        description="Translate spoken audio between languages using Gemini.",
    )
    parser.add_argument(
        "input",
        help="Audio file path, URL, or transcript JSON (with --transcript).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.mp3",
        help="Output audio file path (default: output.mp3).",
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
        help="Treat input as a transcript JSON file (skip STT).",
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

    args = parser.parse_args(argv)

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

    try:
        out = translate_audio(
            input=args.input,
            output=args.output,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            voice_map=voice_map,
            skip_stt=args.transcript,
        )
        print(f"Output saved to: {out}")
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
