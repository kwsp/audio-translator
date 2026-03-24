"""Unit tests for the audio format conversion utilities."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import pytest

from audio_translator.audio_utils import _CHANNELS, _SAMPLE_RATE, _SAMPLE_WIDTH, save_audio


def test_write_wav():
    """Verify that pcm_data is correctly written to a WAV file."""
    pcm_data = b"\x00" * 2400  # 0.1s of silence (24000 samples/s * 0.1s)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_audio(tmp_path, pcm_data, format="wav")
        # Read back WAV data
        with wave.open(str(tmp_path), "rb") as wf:
            assert wf.getnchannels() == _CHANNELS
            assert wf.getsampwidth() == _SAMPLE_WIDTH
            assert wf.getframerate() == _SAMPLE_RATE
            assert wf.getnframes() == len(pcm_data) // _SAMPLE_WIDTH
            read_pcm = wf.readframes(wf.getnframes())
            assert read_pcm == pcm_data
    finally:
        tmp_path.unlink()


def test_save_audio_mp3():
    """Verify that pcm_data is converted to MP3 using ffmpeg."""
    pcm_data = b"\x00" * 4800 # 0.2s of silence
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_audio(tmp_path, pcm_data, format="mp3")
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0
        # ffmpeg output check isn't easy, but we can check it's not empty and exists.
    finally:
        tmp_path.unlink()


def test_invalid_path():
    """Verify that save_audio handles invalid paths appropriately."""
    with pytest.raises(Exception):
        save_audio("/non/existent/path/out.mp3", b"data")
