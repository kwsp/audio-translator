"""Unit tests for the Edge-TTS backend."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from audio_translator.backends.edge.tts import (
    EdgeTTS,
    _build_gender_map,
    _get_voice_map,
    _EDGE_VOICE_FULL_NAMES,
)
from audio_translator.models import Speaker, TranslatedSegment, TranslatedTranscript


# --- Voice mapping tests (no I/O needed) ------------------------------------

def test_edge_voice_mapping_gender():
    speakers = [
        Speaker(name="A", gender="female"),
        Speaker(name="B", gender="male"),
    ]
    gmap = _build_gender_map(speakers)
    vmap = _get_voice_map(["A", "B"], gmap, None)
    assert vmap["A"] == "AvaMultilingual"
    assert vmap["B"] == "BrianMultilingual"


def test_edge_same_gender_two_speakers():
    """Two speakers of the same gender should get distinct voices."""
    speakers = [
        Speaker(name="F1", gender="female"),
        Speaker(name="F2", gender="female"),
    ]
    gmap = _build_gender_map(speakers)
    vmap = _get_voice_map(["F1", "F2"], gmap, None)
    assert vmap["F1"] == "AvaMultilingual"
    assert vmap["F2"] == "EmmaMultilingual"


def test_edge_same_gender_two_male_speakers():
    speakers = [
        Speaker(name="M1", gender="male"),
        Speaker(name="M2", gender="male"),
    ]
    gmap = _build_gender_map(speakers)
    vmap = _get_voice_map(["M1", "M2"], gmap, None)
    assert vmap["M1"] == "BrianMultilingual"
    assert vmap["M2"] == "AndrewMultilingual"


def test_edge_voice_user_override():
    speakers = [Speaker(name="A", gender="female")]
    gmap = _build_gender_map(speakers)
    vmap = _get_voice_map(["A"], gmap, {"A": "BrianMultilingual"})
    assert vmap["A"] == "BrianMultilingual"


def test_edge_full_voice_names():
    """Verify the locale prefixes are present."""
    assert _EDGE_VOICE_FULL_NAMES["AvaMultilingual"] == "en-US-AvaMultilingualNeural"
    assert _EDGE_VOICE_FULL_NAMES["BrianMultilingual"] == "en-US-BrianMultilingualNeural"


# --- Synthesis tests (mocked) ------------------------------------------------

def _make_transcript() -> TranslatedTranscript:
    return TranslatedTranscript(
        source_lang="en",
        target_lang="zh",
        speakers=[
            Speaker(name="Speaker 1", gender="male"),
            Speaker(name="Speaker 2", gender="female"),
        ],
        segments=[
            TranslatedSegment(
                speaker="Speaker 1",
                timestamp="00:00",
                original_text="Hi",
                translated_text="你好",
            ),
            TranslatedSegment(
                speaker="Speaker 2",
                timestamp="00:05",
                original_text="Hello",
                translated_text="哈喽",
            ),
        ],
    )


@patch("audio_translator.backends.edge.tts._mp3_bytes_to_pcm", return_value=b"\x00" * 100)
@patch("audio_translator.backends.edge.tts._synthesize_segment")
def test_edge_tts_synthesize(mock_synth, mock_convert):
    """Verify EdgeTTS calls synthesis for each segment and stitches results."""
    mock_synth.return_value = b"MP3DATA"

    tts = EdgeTTS()
    result = tts.synthesize(_make_transcript())

    # One call per segment
    assert mock_synth.call_count == 2
    # PCM data is concatenated (100 bytes × 2 segments)
    assert len(result) == 200
    mock_convert.assert_called()


@patch("audio_translator.backends.edge.tts._mp3_bytes_to_pcm", return_value=b"\x00" * 100)
@patch("audio_translator.backends.edge.tts._synthesize_segment")
def test_edge_tts_correct_voices(mock_synth, mock_convert):
    """Verify male gets Brian and female gets Ava."""
    mock_synth.return_value = b"MP3"

    tts = EdgeTTS()
    tts.synthesize(_make_transcript())

    calls = mock_synth.call_args_list
    # Segment 1: Speaker 1 (male) → Brian
    assert "BrianMultilingual" in calls[0][0][1]
    # Segment 2: Speaker 2 (female) → Ava
    assert "AvaMultilingual" in calls[1][0][1]


@patch("audio_translator.backends.edge.tts._mp3_bytes_to_pcm")
@patch("audio_translator.backends.edge.tts._synthesize_segment")
def test_edge_tts_missing_dependency(mock_synth, mock_convert):
    """Verify useful error message when edge-tts is not installed."""
    mock_synth.side_effect = ImportError("edge-tts is not installed.")

    tts = EdgeTTS()
    with pytest.raises(ImportError, match="edge-tts"):
        tts.synthesize(_make_transcript())
