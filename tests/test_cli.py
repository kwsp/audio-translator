"""Unit tests for the CLI entry point."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest
from audio_translator.cli import main


def _mock_result():
    return MagicMock(
        output_dir="sample/out",
        transcript=MagicMock(name="transcript.json"),
        translated_transcript=MagicMock(name="translated.json"),
        audio=MagicMock(name="audio.mp3"),
    )


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Translate spoken audio" in captured.out


@patch("audio_translator.cli.translate_audio_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_basic_flow(mock_dotenv, mock_fn, capsys):
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
    mock_dotenv.assert_called_once()
    captured = capsys.readouterr()
    assert "Outputs written to: sample/out/" in captured.out


@patch("audio_translator.cli.translate_text_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_text_mode(mock_dotenv, mock_fn, capsys):
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


@patch("audio_translator.cli.translate_transcript_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_transcript_mode(mock_dotenv, mock_fn, capsys):
    mock_fn.return_value = _mock_result()

    main(["trans.json", "--transcript", "--target-lang", "French"])

    mock_fn.assert_called_once_with(
        transcript="trans.json",
        output_dir=None,
        target_lang="French",
        voice_map=None,
        tts=None,
    )


@patch("audio_translator.cli.synthesize_translated_transcript")
@patch("audio_translator.cli.load_dotenv")
def test_cli_translated_transcript_mode(mock_dotenv, mock_fn, capsys):
    mock_fn.return_value = _mock_result()

    main(["tt.json", "--translated-transcript"])

    mock_fn.assert_called_once_with(
        translated_transcript="tt.json",
        output_dir=None,
        voice_map=None,
        tts=None,
    )


@patch("audio_translator.cli.translate_audio_to_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_voice_map(mock_dotenv, mock_fn, capsys):
    mock_fn.return_value = _mock_result()

    main(["in.mp3", "--voice-map", '{"Speaker 1": "Kore"}'])

    mock_fn.assert_called_once()
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
