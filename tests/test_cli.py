"""Unit tests for the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
from audio_translator.cli import main, _detect_input_type
from audio_translator.models import Speaker, Transcript, TranslatedTranscript, Segment, TranslatedSegment


def _mock_result():
    return MagicMock(
        output_dir="sample/out",
        transcript=MagicMock(name="transcript.json"),
        translated_transcript=MagicMock(name="translated.json"),
        audio=MagicMock(name="audio.mp3"),
    )


# ---------------------------------------------------------------------------
# _detect_input_type unit tests
# ---------------------------------------------------------------------------

def test_detect_audio_extension():
    assert _detect_input_type("recording.mp3") == "audio"
    assert _detect_input_type("clip.wav") == "audio"
    assert _detect_input_type("talk.m4a") == "audio"


def test_detect_url():
    assert _detect_input_type("https://example.com/audio.mp3") == "audio"
    assert _detect_input_type("http://example.com/stream") == "audio"


def test_detect_text_extension():
    assert _detect_input_type("passage.txt") == "text"
    assert _detect_input_type("notes.md") == "text"


def test_detect_transcript_json(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(
        Transcript(
            lang="English",
            speakers=[Speaker(name="S1", gender="female")],
            segments=[Segment(speaker="S1", timestamp="00:00", text="Hi")],
        ).model_dump_json()
    )
    assert _detect_input_type(str(f)) == "transcript"


def test_detect_translated_transcript_json(tmp_path):
    f = tmp_path / "tt.json"
    f.write_text(
        TranslatedTranscript(
            source_lang="English",
            target_lang="Mandarin Chinese",
            segments=[
                TranslatedSegment(
                    speaker="S1", timestamp="00:00",
                    original_text="Hi", translated_text="你好"
                )
            ],
        ).model_dump_json()
    )
    assert _detect_input_type(str(f)) == "translated_transcript"


def test_detect_unknown_extension():
    with pytest.raises(ValueError, match="Cannot determine input type"):
        _detect_input_type("file.xyz")


def test_detect_ambiguous_json(tmp_path):
    f = tmp_path / "unknown.json"
    f.write_text('{"foo": "bar"}')
    with pytest.raises(ValueError, match="does not look like"):
        _detect_input_type(str(f))


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Translate spoken audio" in captured.out


@patch("audio_translator.cli.translate_audio_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_audio_file(mock_dotenv, mock_fn, capsys):
    mock_fn.return_value = _mock_result()

    main(["input.mp3", "-o", "custom_out", "--target-lang", "French"])

    mock_fn.assert_called_once_with(
        input="input.mp3",
        output_dir="custom_out",
        target_lang="French",
        voice_map=None,
        stt=ANY,
        tts=None,
    )
    captured = capsys.readouterr()
    assert "Outputs written to: sample/out/" in captured.out


@patch("audio_translator.cli.translate_text_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_text_flag(mock_dotenv, mock_fn, capsys):
    mock_fn.return_value = _mock_result()

    main(["--text", "Hello world", "--target-lang", "Spanish"])

    mock_fn.assert_called_once_with(
        text="Hello world",
        output_dir=None,
        source_lang="English",
        target_lang="Spanish",
        voice_map=None,
        tts=None,
    )


@patch("audio_translator.cli.translate_text_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_txt_file(mock_dotenv, mock_fn, tmp_path, capsys):
    txt = tmp_path / "passage.txt"
    txt.write_text("Hello from a file")
    mock_fn.return_value = _mock_result()

    main([str(txt), "--target-lang", "French"])

    mock_fn.assert_called_once_with(
        text="Hello from a file",
        output_dir="passage",
        source_lang="English",
        target_lang="French",
        voice_map=None,
        tts=None,
    )


@patch("audio_translator.cli.translate_transcript_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_transcript_json(mock_dotenv, mock_fn, tmp_path, capsys):
    f = tmp_path / "t.json"
    f.write_text(
        Transcript(
            lang="English",
            speakers=[Speaker(name="S1", gender="female")],
            segments=[Segment(speaker="S1", timestamp="00:00", text="Hi")],
        ).model_dump_json()
    )
    mock_fn.return_value = _mock_result()

    main([str(f), "--target-lang", "French"])

    mock_fn.assert_called_once_with(
        transcript=str(f),
        output_dir=None,
        target_lang="French",
        voice_map=None,
        tts=None,
    )


@patch("audio_translator.cli.synthesize_translated_transcript")
@patch("audio_translator.cli.load_dotenv")
def test_cli_translated_transcript_json(mock_dotenv, mock_fn, tmp_path, capsys):
    f = tmp_path / "tt.json"
    f.write_text(
        TranslatedTranscript(
            source_lang="English",
            target_lang="Mandarin Chinese",
            segments=[
                TranslatedSegment(
                    speaker="S1", timestamp="00:00",
                    original_text="Hi", translated_text="你好"
                )
            ],
        ).model_dump_json()
    )
    mock_fn.return_value = _mock_result()

    main([str(f)])

    mock_fn.assert_called_once_with(
        translated_transcript=str(f),
        output_dir=None,
        voice_map=None,
        tts=None,
    )


@patch("audio_translator.cli.translate_audio_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_voice_map(mock_dotenv, mock_fn, capsys):
    mock_fn.return_value = _mock_result()

    main(["in.mp3", "--voice-map", '{"Speaker 1": "Kore"}'])

    args = mock_fn.call_args[1]
    assert args["voice_map"] == {"Speaker 1": "Kore"}


@patch("audio_translator.cli.translate_audio_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_invalid_voice_map(mock_dotenv, mock_fn, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["in.mp3", "--voice-map", "not-a-json"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "invalid --voice-map JSON" in captured.err
