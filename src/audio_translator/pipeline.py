"""Pipeline orchestrator for the audio-translator.

Wires together STT → Translate → TTS stages with support for
skipping stages when appropriate (e.g. transcript already provided,
or already in the target language).

Output organisation
-------------------
All pipeline outputs are written to a single directory::

    <output_dir>/
      transcript.json            # STT output (Transcript)
      translated_transcript.json # Translation output (TranslatedTranscript)
      audio.mp3                  # Final audio

The directory defaults to a slug derived from the input path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from audio_translator.audio_utils import save_audio
from audio_translator.backends.base import STTBackend, TranslateBackend, TTSBackend
from audio_translator.backends.gemini.stt import GeminiSTT
from audio_translator.backends.gemini.translate import GeminiTranslate
from audio_translator.backends.gemini.tts import GeminiTTS
from audio_translator.models import Transcript, TranslatedSegment, TranslatedTranscript

logger = logging.getLogger(__name__)

_TRANSCRIPT_FILENAME = "transcript.json"
_TRANSLATED_FILENAME = "translated_transcript.json"
_AUDIO_FILENAME = "audio.mp3"


@dataclass
class PipelineResult:
    """Paths to all files written by the pipeline."""

    output_dir: Path
    transcript: Path
    translated_transcript: Path
    audio: Path


def _default_output_dir(input_path: str) -> Path:
    """Derive a sensible output directory name from the input path or URL."""
    if input_path.startswith("http://") or input_path.startswith("https://"):
        # Use last URL segment, strip query string
        slug = input_path.rstrip("/").split("/")[-1].split("?")[0]
        slug = re.sub(r"[^\w\-]", "_", slug)
    else:
        slug = Path(input_path).stem
    return Path(slug)


def _transcript_to_translated(transcript: Transcript) -> TranslatedTranscript:
    """Wrap a Transcript as a TranslatedTranscript when no translation is needed."""
    return TranslatedTranscript(
        source_lang=transcript.lang,
        target_lang=transcript.lang,
        speakers=transcript.speakers,
        segments=[
            TranslatedSegment(
                speaker=seg.speaker,
                timestamp=seg.timestamp,
                original_text=seg.text,
                translated_text=seg.text,
            )
            for seg in transcript.segments
        ],
    )


def translate_audio(
    input: str,
    output_dir: str | Path | None = None,
    source_lang: str = "English",
    target_lang: str = "Mandarin Chinese",
    voice_map: dict[str, str] | None = None,
    skip_stt: bool = False,
    skip_translate: bool = False,
    stt: STTBackend | None = None,
    translator: TranslateBackend | None = None,
    tts: TTSBackend | None = None,
) -> PipelineResult:
    """Run the full audio translation pipeline.

    Args:
        input: Audio file path, URL, transcript JSON (if skip_stt), or
            translated transcript JSON (if skip_translate).
        output_dir: Directory to write all outputs to. Defaults to a slug
            derived from the input filename.
        source_lang: Expected source language (hint for STT).
        target_lang: Target language for translation.
        voice_map: Optional speaker→voice name mapping.
        skip_stt: If True, treat ``input`` as a Transcript JSON file.
        skip_translate: If True, treat ``input`` as a TranslatedTranscript JSON
            and skip both STT and translation (TTS-only mode).
        stt: STT backend override (default: GeminiSTT).
        translator: Translation backend override (default: GeminiTranslate).
        tts: TTS backend override (default: GeminiTTS).

    Returns:
        PipelineResult with paths to all written files.
    """
    tts = tts or GeminiTTS()
    out_dir = Path(output_dir) if output_dir else _default_output_dir(input)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    transcript_path = out_dir / _TRANSCRIPT_FILENAME
    translated_path = out_dir / _TRANSLATED_FILENAME
    audio_path = out_dir / _AUDIO_FILENAME

    # --- TTS-only fast path: skip both STT and translation ---
    if skip_translate:
        logger.info("Loading translated transcript from %s", input)
        translated = TranslatedTranscript.model_validate_json(
            Path(input).read_text(encoding="utf-8")
        )
        logger.info(
            "Translated transcript: %d segments, %s → %s",
            len(translated.segments),
            translated.source_lang,
            translated.target_lang,
        )
        translated_path.write_text(
            translated.model_dump_json(indent=2), encoding="utf-8"
        )
        # --- Stage 3: TTS ---
        logger.info("Synthesizing speech (%d segments)", len(translated.segments))
        pcm_data = tts.synthesize(translated, voice_map)
        save_audio(audio_path, pcm_data)
        logger.info("Saved audio → %s", audio_path)
        return PipelineResult(
            output_dir=out_dir,
            transcript=translated_path,   # no raw transcript available
            translated_transcript=translated_path,
            audio=audio_path,
        )

    stt = stt or GeminiSTT()
    translator = translator or GeminiTranslate()

    # --- Stage 1: STT ---
    if skip_stt:
        logger.info("Loading transcript from %s", input)
        transcript = Transcript.model_validate_json(
            Path(input).read_text(encoding="utf-8")
        )
    else:
        logger.info("Transcribing audio: %s", input)
        transcript = stt.transcribe(input)

    logger.info(
        "Transcript: %d segments, language=%s", len(transcript.segments), transcript.lang
    )
    transcript_path.write_text(
        transcript.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Saved transcript → %s", transcript_path)

    # --- Stage 2: Translate (optional) ---
    if transcript.lang.lower().strip() == target_lang.lower().strip():
        logger.info("Transcript already in target language, skipping translation")
        translated = _transcript_to_translated(transcript)
    else:
        logger.info("Translating %s → %s", transcript.lang, target_lang)
        translated = translator.translate(transcript, target_lang)

    translated_path.write_text(
        translated.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Saved translated transcript → %s", translated_path)

    # --- Stage 3: TTS ---
    logger.info("Synthesizing speech (%d segments)", len(translated.segments))
    pcm_data = tts.synthesize(translated, voice_map)

    save_audio(audio_path, pcm_data)
    logger.info("Saved audio → %s", audio_path)

    return PipelineResult(
        output_dir=out_dir,
        transcript=transcript_path,
        translated_transcript=translated_path,
        audio=audio_path,
    )
