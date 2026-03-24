"""Unit tests for the pipeline orchestration logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from audio_translator.pipeline import translate_audio
from audio_translator.models import Speaker, Transcript, TranslatedTranscript, Segment, TranslatedSegment


def test_translate_audio_pipeline(mocker, tmp_path):
    stt_mock = MagicMock()
    trans_mock = MagicMock()
    tts_mock = MagicMock()

    # Mock STT output
    stt_mock.transcribe.return_value = Transcript(
        lang="English",
        speakers=[Speaker(name="Speaker 1", gender="male")],
        segments=[Segment(speaker="Speaker 1", timestamp="00:00", text="Hello")],
    )

    # Mock Translation output
    trans_mock.translate.return_value = TranslatedTranscript(
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

    # Mock TTS output (raw PCM)
    tts_mock.synthesize.return_value = b"\x00" * 4800  # 0.2s of silence

    # Run the pipeline
    out_dir = tmp_path / "test_run"
    result = translate_audio(
        input="dummy.mp3",
        output_dir=out_dir,
        source_lang="English",
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

    # Verify that backends were correctly called.
    stt_mock.transcribe.assert_called_with("dummy.mp3")
    trans_mock.translate.assert_called()
    tts_mock.synthesize.assert_called()


def test_skip_translation_if_same_language(mocker, tmp_path):
    stt_mock = MagicMock()
    trans_mock = MagicMock()
    tts_mock = MagicMock()

    # Audio is already Mandarin Chinese.
    stt_mock.transcribe.return_value = Transcript(
        lang="Mandarin Chinese",
        speakers=[Speaker(name="Speaker 1", gender="female")],
        segments=[Segment(speaker="Speaker 1", timestamp="00:00", text="你好")],
    )
    tts_mock.synthesize.return_value = b"\x00" * 2400

    out_dir = tmp_path / "test_skip"
    translate_audio(
        input="dummy.mp3",
        output_dir=out_dir,
        source_lang="Chinese",
        target_lang="Mandarin Chinese",  # Should skip translation
        stt=stt_mock,
        translator=trans_mock,
        tts=tts_mock,
    )

    # Translate backend SHOULD NOT be called.
    assert not trans_mock.translate.called
    # But TTS SHOULD be called for the synthesis.
    assert tts_mock.synthesize.called


def test_default_output_dir():
    from audio_translator.pipeline import _default_output_dir
    from pathlib import Path

    # 1. Local path
    assert _default_output_dir("path/to/my_audio.mp3") == Path("my_audio")
    
    # 2. URL
    assert _default_output_dir("https://example.com/interview?v=123") == Path("interview")
    
    # 3. URL with trailing slash
    assert _default_output_dir("https://site.org/podcast/") == Path("podcast")


def test_pipeline_skip_stt(mocker, tmp_path):
    # Create a dummy transcript file
    trans = Transcript(
        lang="English",
        speakers=[Speaker(name="S1", gender="unknown")],
        segments=[Segment(speaker="S1", timestamp="0:00", text="Test")],
    )
    trans_file = tmp_path / "trans.json"
    trans_file.write_text(trans.model_dump_json())

    # Mock Translation and TTS backends
    trans_mock = MagicMock()
    trans_mock.translate.return_value = TranslatedTranscript(
        source_lang="English",
        target_lang="Spanish",
        speakers=trans.speakers,
        segments=[
            TranslatedSegment(
                speaker="S1",
                timestamp="0:00",
                original_text="Test",
                translated_text="Prueba",
            )
        ],
    )
    tts_mock = MagicMock()
    tts_mock.synthesize.return_value = b"\x00" * 2400

    # Run pipeline with skip_stt=True
    translate_audio(
        input=str(trans_file),
        skip_stt=True,
        target_lang="Spanish",
        stt=MagicMock(),  # Mandatory to avoid real GeminiSTT init
        translator=trans_mock,
        tts=tts_mock,
    )

    # Verify translator WAS called
    trans_mock.translate.assert_called_once()
