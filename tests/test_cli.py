"""Unit tests for the CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from audio_translator.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Translate spoken audio" in captured.out


@patch("audio_translator.cli.translate_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_basic_flow(mock_dotenv, mock_translate, capsys):
    mock_translate.return_value = MagicMock(
        output_dir="sample/out",
        transcript=MagicMock(name="transcript.json"),
        translated_transcript=MagicMock(name="translated.json"),
        audio=MagicMock(name="audio.mp3"),
    )

    main(["input.mp3", "-o", "custom_out", "--target-lang", "French"])

    mock_translate.assert_called_once_with(
        input="input.mp3",
        output_dir="custom_out",
        source_lang="English",
        target_lang="French",
        voice_map=None,
        skip_stt=False,
    )
    mock_dotenv.assert_called_once()
    captured = capsys.readouterr()
    assert "Outputs written to: sample/out/" in captured.out


@patch("audio_translator.cli.translate_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_voice_map(mock_dotenv, mock_translate, capsys):
    mock_translate.return_value = MagicMock(
        output_dir="out",
        transcript=MagicMock(name="t.json"),
        translated_transcript=MagicMock(name="tt.json"),
        audio=MagicMock(name="a.mp3"),
    )

    main(["in.mp3", "--voice-map", '{"Speaker 1": "Kore"}'])

    mock_translate.assert_called_once()
    args = mock_translate.call_args[1]
    assert args["voice_map"] == {"Speaker 1": "Kore"}


@patch("audio_translator.cli.translate_audio")
@patch("audio_translator.cli.load_dotenv")
def test_cli_invalid_voice_map(mock_dotenv, mock_translate, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["in.mp3", "--voice-map", "not-a-json"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "invalid --voice-map JSON" in captured.err
