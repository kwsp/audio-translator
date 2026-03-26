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
from audio_translator.models import Segment, Speaker, Transcript, TranslatedSegment, TranslatedTranscript

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


def text_to_transcript(
    text: str,
    lang: str = "English",
    speaker_name: str = "Speaker 1",
    speaker_gender: str = "female",
) -> Transcript:
    """Wrap a plain-text passage in a single-speaker, single-segment Transcript."""
    return Transcript(
        lang=lang,
        speakers=[Speaker(name=speaker_name, gender=speaker_gender)],  # type: ignore[arg-type]
        segments=[Segment(speaker=speaker_name, timestamp="00:00", text=text)],
    )


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


def _run_translate_and_synthesize(
    transcript: Transcript,
    out_dir: Path,
    target_lang: str,
    voice_map: dict[str, str] | None,
    translator: TranslateBackend,
    tts: TTSBackend,
) -> tuple[Path, Path, Path]:
    """Translate a Transcript and synthesize audio; write all artifacts.

    Returns:
        (transcript_path, translated_path, audio_path)
    """
    transcript_path = out_dir / _TRANSCRIPT_FILENAME
    translated_path = out_dir / _TRANSLATED_FILENAME
    audio_path = out_dir / _AUDIO_FILENAME

    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved transcript → %s", transcript_path)

    if transcript.lang.lower().strip() == target_lang.lower().strip():
        logger.info("Transcript already in target language, skipping translation")
        translated = _transcript_to_translated(transcript)
    else:
        logger.info("Translating %s → %s", transcript.lang, target_lang)
        translated = translator.translate(transcript, target_lang)

    translated_path.write_text(translated.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved translated transcript → %s", translated_path)

    logger.info("Synthesizing speech (%d segments)", len(translated.segments))
    pcm_data = tts.synthesize(translated, voice_map)
    save_audio(audio_path, pcm_data)
    logger.info("Saved audio → %s", audio_path)

    return transcript_path, translated_path, audio_path


def _run_synthesize(
    translated: TranslatedTranscript,
    out_dir: Path,
    voice_map: dict[str, str] | None,
    tts: TTSBackend,
) -> tuple[Path, Path]:
    """Synthesize audio from a TranslatedTranscript; write artifacts.

    Returns:
        (translated_path, audio_path)
    """
    translated_path = out_dir / _TRANSLATED_FILENAME
    audio_path = out_dir / _AUDIO_FILENAME

    translated_path.write_text(translated.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved translated transcript → %s", translated_path)

    logger.info("Synthesizing speech (%d segments)", len(translated.segments))
    pcm_data = tts.synthesize(translated, voice_map)
    save_audio(audio_path, pcm_data)
    logger.info("Saved audio → %s", audio_path)

    return translated_path, audio_path


# ---------------------------------------------------------------------------
# Public pipeline entry points
# ---------------------------------------------------------------------------


def translate_audio_to_audio(
    input: str,
    output_dir: str | Path | None = None,
    target_lang: str = "Mandarin Chinese",
    voice_map: dict[str, str] | None = None,
    stt: STTBackend | None = None,
    translator: TranslateBackend | None = None,
    tts: TTSBackend | None = None,
) -> PipelineResult:
    """Translate an audio file or URL to audio in the target language.

    Runs the full STT → Translate → TTS pipeline.

    Args:
        input: Local audio file path or URL.
        output_dir: Directory for all outputs. Defaults to a slug derived from
            the input filename/URL.
        target_lang: Target language for translation.
        voice_map: Optional speaker→voice name mapping for TTS.
        stt: STT backend override (default: GeminiSTT).
        translator: Translation backend override (default: GeminiTranslate).
        tts: TTS backend override (default: GeminiTTS).

    Returns:
        PipelineResult with paths to all written files.
    """
    stt = stt or GeminiSTT()
    translator = translator or GeminiTranslate()
    tts = tts or GeminiTTS()
    out_dir = Path(output_dir) if output_dir else _default_output_dir(input)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    logger.info("Transcribing audio: %s", input)
    transcript = stt.transcribe(input)
    logger.info(
        "Transcript: %d segments, language=%s", len(transcript.segments), transcript.lang
    )

    transcript_path, translated_path, audio_path = _run_translate_and_synthesize(
        transcript, out_dir, target_lang, voice_map, translator, tts
    )
    return PipelineResult(
        output_dir=out_dir,
        transcript=transcript_path,
        translated_transcript=translated_path,
        audio=audio_path,
    )


def translate_text_to_audio(
    text: str,
    output_dir: str | Path | None = None,
    source_lang: str = "English",
    target_lang: str = "Mandarin Chinese",
    voice_map: dict[str, str] | None = None,
    translator: TranslateBackend | None = None,
    tts: TTSBackend | None = None,
) -> PipelineResult:
    """Translate a plain-text passage to audio in the target language.

    Wraps the text into a single-speaker female transcript and runs
    Translate → TTS (STT is skipped).

    Args:
        text: Plain-text passage in the source language.
        output_dir: Directory for all outputs. Defaults to ``"text_input"``.
        source_lang: Language of the input text.
        target_lang: Target language for translation.
        voice_map: Optional speaker→voice name mapping for TTS.
        translator: Translation backend override (default: GeminiTranslate).
        tts: TTS backend override (default: GeminiTTS).

    Returns:
        PipelineResult with paths to all written files.
    """
    translator = translator or GeminiTranslate()
    tts = tts or GeminiTTS()
    out_dir = Path(output_dir) if output_dir else Path("text_input")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    logger.info("Building transcript from plain text (%d chars)", len(text))
    transcript = text_to_transcript(text, lang=source_lang)

    transcript_path, translated_path, audio_path = _run_translate_and_synthesize(
        transcript, out_dir, target_lang, voice_map, translator, tts
    )
    return PipelineResult(
        output_dir=out_dir,
        transcript=transcript_path,
        translated_transcript=translated_path,
        audio=audio_path,
    )


def translate_transcript_to_audio(
    transcript: Transcript | str,
    output_dir: str | Path | None = None,
    target_lang: str = "Mandarin Chinese",
    voice_map: dict[str, str] | None = None,
    translator: TranslateBackend | None = None,
    tts: TTSBackend | None = None,
) -> PipelineResult:
    """Translate a Transcript to audio in the target language.

    Runs Translate → TTS (STT is skipped).

    Args:
        transcript: A ``Transcript`` object or a path to a Transcript JSON file.
        output_dir: Directory for all outputs. Defaults to the stem of the JSON
            file path, or ``"transcript_output"`` when a Transcript object is
            passed directly.
        target_lang: Target language for translation.
        voice_map: Optional speaker→voice name mapping for TTS.
        translator: Translation backend override (default: GeminiTranslate).
        tts: TTS backend override (default: GeminiTTS).

    Returns:
        PipelineResult with paths to all written files.
    """
    translator = translator or GeminiTranslate()
    tts = tts or GeminiTTS()

    if isinstance(transcript, str):
        default_dir = Path(transcript).stem
        logger.info("Loading transcript from %s", transcript)
        loaded = Transcript.model_validate_json(
            Path(transcript).read_text(encoding="utf-8")
        )
    else:
        default_dir = "transcript_output"
        loaded = transcript

    logger.info(
        "Transcript: %d segments, language=%s", len(loaded.segments), loaded.lang
    )

    out_dir = Path(output_dir) if output_dir else Path(default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    transcript_path, translated_path, audio_path = _run_translate_and_synthesize(
        loaded, out_dir, target_lang, voice_map, translator, tts
    )
    return PipelineResult(
        output_dir=out_dir,
        transcript=transcript_path,
        translated_transcript=translated_path,
        audio=audio_path,
    )


def synthesize_translated_transcript(
    translated_transcript: TranslatedTranscript | str,
    output_dir: str | Path | None = None,
    voice_map: dict[str, str] | None = None,
    tts: TTSBackend | None = None,
) -> PipelineResult:
    """Synthesize audio from an already-translated transcript (TTS only).

    Skips both STT and translation.

    Args:
        translated_transcript: A ``TranslatedTranscript`` object or a path to a
            TranslatedTranscript JSON file.
        output_dir: Directory for all outputs. Defaults to the stem of the JSON
            file path, or ``"tts_output"`` when an object is passed directly.
        voice_map: Optional speaker→voice name mapping for TTS.
        tts: TTS backend override (default: GeminiTTS).

    Returns:
        PipelineResult with paths to all written files.
    """
    tts = tts or GeminiTTS()

    if isinstance(translated_transcript, str):
        default_dir = Path(translated_transcript).stem
        logger.info("Loading translated transcript from %s", translated_transcript)
        loaded = TranslatedTranscript.model_validate_json(
            Path(translated_transcript).read_text(encoding="utf-8")
        )
    else:
        default_dir = "tts_output"
        loaded = translated_transcript

    logger.info(
        "Translated transcript: %d segments, %s → %s",
        len(loaded.segments),
        loaded.source_lang,
        loaded.target_lang,
    )

    out_dir = Path(output_dir) if output_dir else Path(default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    translated_path, audio_path = _run_synthesize(loaded, out_dir, voice_map, tts)
    return PipelineResult(
        output_dir=out_dir,
        transcript=translated_path,  # no raw transcript available
        translated_transcript=translated_path,
        audio=audio_path,
    )
