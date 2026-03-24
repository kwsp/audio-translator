"""Unit tests for the Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from audio_translator.models import (
    Segment,
    Speaker,
    Transcript,
    TranslatedSegment,
    TranslatedTranscript,
)


def test_speaker_model():
    s = Speaker(name="Speaker 1", gender="female")
    assert s.name == "Speaker 1"
    assert s.gender == "female"

    # Default value
    s2 = Speaker(name="Unknown")
    assert s2.gender == "unknown"

    # Invalid gender
    with pytest.raises(ValidationError):
        Speaker(name="Invalid", gender="robot")


def test_segment_model():
    seg = Segment(speaker="Speaker 1", timestamp="00:05", text="Hello world")
    assert seg.speaker == "Speaker 1"
    assert seg.timestamp == "00:05"
    assert seg.text == "Hello world"


def test_transcript_roundtrip():
    data = {
        "lang": "en",
        "speakers": [{"name": "Speaker 1", "gender": "male"}],
        "segments": [{"speaker": "Speaker 1", "timestamp": "0:00", "text": "Hi"}],
    }
    t = Transcript.model_validate(data)
    assert t.lang == "en"
    assert len(t.speakers) == 1
    assert t.speakers[0].gender == "male"
    assert len(t.segments) == 1

    # JSON roundtrip
    json_str = t.model_dump_json()
    t2 = Transcript.model_validate_json(json_str)
    assert t == t2


def test_translated_transcript_roundtrip():
    s1 = Speaker(name="S1", gender="female")
    seg = TranslatedSegment(
        speaker="S1",
        timestamp="00:10",
        original_text="Bonjour",
        translated_text="Hello",
    )
    tt = TranslatedTranscript(
        source_lang="fr",
        target_lang="en",
        speakers=[s1],
        segments=[seg],
    )
    assert tt.source_lang == "fr"
    assert tt.target_lang == "en"
    assert tt.speakers[0].name == "S1"
    assert tt.segments[0].translated_text == "Hello"

    json_str = tt.model_dump_json()
    tt2 = TranslatedTranscript.model_validate_json(json_str)
    assert tt == tt2
