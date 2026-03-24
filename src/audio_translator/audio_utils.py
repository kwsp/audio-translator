"""Audio format conversion utilities.

Converts raw PCM data (from Gemini TTS) to MP3 using the standard library
``wave`` module and ``ffmpeg`` subprocess. No dependency on ``pydub``.

Requires ``ffmpeg`` to be installed on the system.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

# Gemini TTS output format constants.
_SAMPLE_RATE = 24000
_SAMPLE_WIDTH = 2  # 16-bit
_CHANNELS = 1  # mono


def _write_wav(path: Path, pcm_data: bytes) -> None:
    """Write raw PCM bytes to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm_data)


def save_audio(
    path: str | Path,
    pcm_data: bytes,
    *,
    format: str = "mp3",
    bitrate: str = "192k",
) -> Path:
    """Save raw PCM audio bytes to a file (default MP3).

    For WAV output, writes directly. For other formats (MP3, etc.),
    writes a temporary WAV then converts via ``ffmpeg``.

    Args:
        path: Output file path.
        pcm_data: Raw PCM bytes (24kHz, 16-bit, mono).
        format: Output format (e.g. "mp3", "wav").
        bitrate: Audio bitrate for lossy formats.

    Returns:
        The resolved output path.
    """
    path = Path(path)

    if format == "wav":
        _write_wav(path, pcm_data)
        return path

    # For non-WAV formats, write temp WAV then convert with ffmpeg.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        _write_wav(tmp_wav, pcm_data)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_wav),
            "-b:a",
            bitrate,
            str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        tmp_wav.unlink(missing_ok=True)

    return path
