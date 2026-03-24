"""Edge-TTS backend for audio translation.

Uses Microsoft Edge's free neural TTS service via the edge-tts Python library.
No API key required - works entirely through the Edge TTS API.

Install the optional dependency:
    uv pip install -e ".[edge]"
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from audio_translator.audio_utils import _CHANNELS, _SAMPLE_RATE, _SAMPLE_WIDTH
from audio_translator.backends.base import TTSBackend
from audio_translator.models import Speaker, TranslatedTranscript

logger = logging.getLogger(__name__)

# Gender-keyed voice pools (two per gender for same-gender multi-speaker).
_DEFAULT_VOICES_BY_GENDER: dict[str, list[str]] = {
    "female":  ["en-US-AvaMultilingualNeural", "en-US-EmmaMultilingualNeural"],
    "male":    ["en-US-BrianMultilingualNeural", "en-US-AndrewMultilingualNeural"],
    "unknown": ["en-US-AvaMultilingualNeural"],
}
_DEFAULT_VOICE = "en-US-AvaMultilingualNeural"


def _build_gender_map(speakers: list[Speaker]) -> dict[str, str]:
    return {s.name: s.gender for s in speakers}


def _get_voice_map(
    speaker_names: list[str],
    gender_map: dict[str, str],
    user_map: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve speaker → short voice name, respecting user overrides and gender."""
    voice_map: dict[str, str] = {}
    gender_usage: dict[str, int] = {}

    for speaker in speaker_names:
        if user_map and speaker in user_map:
            voice_map[speaker] = user_map[speaker]
        else:
            gender = gender_map.get(speaker, "unknown")
            pool = _DEFAULT_VOICES_BY_GENDER.get(gender, _DEFAULT_VOICES_BY_GENDER["unknown"])
            idx = gender_usage.get(gender, 0)
            voice_map[speaker] = pool[idx % len(pool)]
            gender_usage[gender] = idx + 1

    return voice_map


async def _synthesize_segment(text: str, voice_full_name: str) -> bytes:
    """Call edge-tts to synthesize a single text segment; returns MP3 bytes."""
    try:
        import edge_tts  # noqa: PLC0415  (optional dependency)
    except ImportError as exc:
        raise ImportError(
            "edge-tts is not installed. Run: uv pip install -e '.[edge]'"
        ) from exc

    communicate = edge_tts.Communicate(text, voice_full_name)
    mp3_chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])

    return b"".join(mp3_chunks)


def _mp3_bytes_to_pcm(mp3_data: bytes) -> bytes:
    """Convert MP3 bytes to raw PCM (24kHz, 16-bit, mono) using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(mp3_data)
        tmp_mp3_path = Path(tmp_mp3.name)

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(tmp_mp3_path),
                "-f", "s16le",      # 16-bit signed little-endian PCM
                "-ar", str(_SAMPLE_RATE),
                "-ac", str(_CHANNELS),
                "pipe:1",           # output to stdout
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout
    finally:
        tmp_mp3_path.unlink(missing_ok=True)


class EdgeTTS(TTSBackend):
    """TTS backend using Microsoft Edge's free neural TTS service.

    Synthesizes each segment individually and stitches the PCM output.
    Supports gender-aware voice assignment with two voices per gender.

    Requires the ``edge-tts`` optional dependency:
        uv pip install -e ".[edge]"
    """

    def synthesize(
        self,
        transcript: TranslatedTranscript,
        voice_map: dict[str, str] | None = None,
    ) -> bytes:
        speakers = sorted({seg.speaker for seg in transcript.segments})
        gender_map = _build_gender_map(transcript.speakers)
        resolved_map = _get_voice_map(speakers, gender_map, voice_map)

        for speaker, voice in resolved_map.items():
            gender = gender_map.get(speaker, "unknown")
            logger.info("  %s (%s) → voice: %s", speaker, gender, voice)

        full_audio = b""
        for i, seg in enumerate(transcript.segments, 1):
            voice = resolved_map.get(seg.speaker, _DEFAULT_VOICE)

            logger.info(
                "Synthesizing segment %d/%d [%s, %s]",
                i,
                len(transcript.segments),
                seg.speaker,
                voice,
            )

            mp3_data = asyncio.run(_synthesize_segment(seg.translated_text, voice))
            full_audio += _mp3_bytes_to_pcm(mp3_data)

        return full_audio
