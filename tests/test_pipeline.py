"""Unit tests for the pipeline orchestration logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from audio_translator.pipeline import (
    synthesize_translated_transcript,
    translate_audio_to_audio,
    translate_text_to_audio,
    translate_transcript_to_audio,
)
from audio_translator.models import Speaker, Transcript, TranslatedTranscript, Segment, TranslatedSegment


def _make_transcript(lang="English") -> Transcript:
    return Transcript(
        lang=lang,
        speakers=[Speaker(name="Speaker 1", gender="male")],
        segments=[Segment(speaker="Speaker 1", timestamp="00:00", text="Hello")],
    )


def _make_translated_transcript() -> TranslatedTranscript:
    return TranslatedTranscript(
        source_lang="English",
        target_lang="Mandarin Chinese",
        speakers=[Speaker(name="Speaker 1", gender="male")],
        segments=[
            TranslatedSegment(
                speaker="Speaker 1",
                timestamp="00:00",
                original_text="Hello",
                translated_text="你好",
            )
        ],
    )


def test_translate_audio_to_audio(tmp_path):
    stt_mock = MagicMock()
    trans_mock = MagicMock()
    tts_mock = MagicMock()

    stt_mock.transcribe.return_value = _make_transcript()
    trans_mock.translate.return_value = _make_translated_transcript()
    tts_mock.synthesize.return_value = b"\x00" * 4800

    out_dir = tmp_path / "test_run"
    result = translate_audio_to_audio(
        input="dummy.mp3",
        output_dir=out_dir,
        target_lang="Mandarin Chinese",
        stt=stt_mock,
        translator=trans_mock,
        tts=tts_mock,
    )

    assert result.output_dir.exists()
    assert result.transcript.exists()
    assert result.translated_transcript.exists()
    assert result.audio.exists()
    assert result.audio.suffix == ".mp3"

    stt_mock.transcribe.assert_called_with("dummy.mp3")
    trans_mock.translate.assert_called()
    tts_mock.synthesize.assert_called()


def test_translate_audio_to_audio_skips_translation_when_same_language(tmp_path):
    stt_mock = MagicMock()
    trans_mock = MagicMock()
    tts_mock = MagicMock()

    stt_mock.transcribe.return_value = _make_transcript(lang="Mandarin Chinese")
    tts_mock.synthesize.return_value = b"\x00" * 2400

    translate_audio_to_audio(
        input="dummy.mp3",
        output_dir=tmp_path / "test_skip",
        target_lang="Mandarin Chinese",
        stt=stt_mock,
        translator=trans_mock,
        tts=tts_mock,
    )

    assert not trans_mock.translate.called
    assert tts_mock.synthesize.called


def test_translate_text_to_audio(tmp_path):
    trans_mock = MagicMock()
    tts_mock = MagicMock()

    trans_mock.translate.return_value = _make_translated_transcript()
    tts_mock.synthesize.return_value = b"\x00" * 4800

    result = translate_text_to_audio(
        text="Hello world",
        output_dir=tmp_path / "text_out",
        source_lang="English",
        target_lang="Mandarin Chinese",
        translator=trans_mock,
        tts=tts_mock,
    )

    assert result.transcript.exists()
    assert result.translated_transcript.exists()
    assert result.audio.exists()
    trans_mock.translate.assert_called_once()
    tts_mock.synthesize.assert_called_once()


def test_translate_transcript_to_audio_from_file(tmp_path):
    trans_file = tmp_path / "trans.json"
    trans_file.write_text(_make_transcript().model_dump_json())

    trans_mock = MagicMock()
    tts_mock = MagicMock()
    trans_mock.translate.return_value = TranslatedTranscript(
        source_lang="English",
        target_lang="Spanish",
        speakers=_make_transcript().speakers,
        segments=[
            TranslatedSegment(
                speaker="Speaker 1",
                timestamp="00:00",
                original_text="Hello",
                translated_text="Hola",
            )
        ],
    )
    tts_mock.synthesize.return_value = b"\x00" * 2400

    result = translate_transcript_to_audio(
        transcript=str(trans_file),
        output_dir=tmp_path / "out",
        target_lang="Spanish",
        translator=trans_mock,
        tts=tts_mock,
    )

    trans_mock.translate.assert_called_once()
    assert result.audio.exists()


def test_translate_transcript_to_audio_from_object(tmp_path):
    trans_mock = MagicMock()
    tts_mock = MagicMock()
    trans_mock.translate.return_value = _make_translated_transcript()
    tts_mock.synthesize.return_value = b"\x00" * 2400

    result = translate_transcript_to_audio(
        transcript=_make_transcript(),
        output_dir=tmp_path / "out",
        target_lang="Mandarin Chinese",
        translator=trans_mock,
        tts=tts_mock,
    )

    trans_mock.translate.assert_called_once()
    assert result.audio.exists()


def test_synthesize_translated_transcript_from_file(tmp_path):
    tt_file = tmp_path / "tt.json"
    tt_file.write_text(_make_translated_transcript().model_dump_json())

    tts_mock = MagicMock()
    tts_mock.synthesize.return_value = b"\x00" * 2400

    result = synthesize_translated_transcript(
        translated_transcript=str(tt_file),
        output_dir=tmp_path / "out",
        tts=tts_mock,
    )

    tts_mock.synthesize.assert_called_once()
    assert result.audio.exists()


def test_synthesize_translated_transcript_from_object(tmp_path):
    tts_mock = MagicMock()
    tts_mock.synthesize.return_value = b"\x00" * 2400

    result = synthesize_translated_transcript(
        translated_transcript=_make_translated_transcript(),
        output_dir=tmp_path / "out",
        tts=tts_mock,
    )

    tts_mock.synthesize.assert_called_once()
    assert result.audio.exists()


def test_default_output_dir():
    from audio_translator.pipeline import _default_output_dir
    from pathlib import Path

    assert _default_output_dir("path/to/my_audio.mp3") == Path("my_audio")
    assert _default_output_dir("https://example.com/interview?v=123") == Path("interview")
    assert _default_output_dir("https://site.org/podcast/") == Path("podcast")
