"""Pipeline orchestrator for the audio-translator.

Wires together STT → Translate → TTS stages with support for
skipping stages when appropriate (e.g. transcript already provided,
or already in the target language).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from audio_translator.audio_utils import save_audio
from audio_translator.backends.base import STTBackend, TranslateBackend, TTSBackend
from audio_translator.backends.gemini.stt import GeminiSTT
from audio_translator.backends.gemini.translate import GeminiTranslate
from audio_translator.backends.gemini.tts import GeminiTTS
from audio_translator.models import Transcript, TranslatedSegment, TranslatedTranscript

logger = logging.getLogger(__name__)


def _transcript_to_translated(
    transcript: Transcript,
) -> TranslatedTranscript:
    """Wrap a Transcript as a TranslatedTranscript when no translation is needed."""
    return TranslatedTranscript(
        source_lang=transcript.lang,
        target_lang=transcript.lang,
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
    output: str = "output.mp3",
    source_lang: str = "English",
    target_lang: str = "Mandarin Chinese",
    voice_map: dict[str, str] | None = None,
    skip_stt: bool = False,
    stt: STTBackend | None = None,
    translator: TranslateBackend | None = None,
    tts: TTSBackend | None = None,
) -> Path:
    """Run the full audio translation pipeline.

    Args:
        input: Audio file path, URL, or transcript JSON path (if skip_stt).
        output: Output audio file path.
        source_lang: Expected source language (hint for STT).
        target_lang: Target language for translation.
        voice_map: Optional speaker→voice name mapping.
        skip_stt: If True, treat ``input`` as a transcript JSON file.
        stt: STT backend override (default: GeminiSTT).
        translator: Translation backend override (default: GeminiTranslate).
        tts: TTS backend override (default: GeminiTTS).

    Returns:
        Path to the output audio file.
    """
    stt = stt or GeminiSTT()
    translator = translator or GeminiTranslate()
    tts = tts or GeminiTTS()

    # --- Stage 1: STT ---
    if skip_stt:
        logger.info("Loading transcript from %s", input)
        raw = Path(input).read_text(encoding="utf-8")
        transcript = Transcript.model_validate_json(raw)
    else:
        logger.info("Transcribing audio: %s", input)
        transcript = stt.transcribe(input)

    logger.info(
        "Transcript: %d segments, language=%s",
        len(transcript.segments),
        transcript.lang,
    )

    # --- Stage 2: Translate (optional) ---
    if transcript.lang.lower().strip() == target_lang.lower().strip():
        logger.info("Transcript already in target language, skipping translation")
        translated = _transcript_to_translated(transcript)
    else:
        logger.info("Translating %s → %s", transcript.lang, target_lang)
        translated = translator.translate(transcript, target_lang)

    # --- Stage 3: TTS ---
    logger.info("Synthesizing speech (%d segments)", len(translated.segments))
    pcm_data = tts.synthesize(translated, voice_map)

    # --- Save output ---
    out_path = save_audio(output, pcm_data)
    logger.info("Saved output to %s", out_path)
    return out_path
