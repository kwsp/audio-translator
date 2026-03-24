"""Pydantic data models for the audio-translator pipeline.

These models serve as the interchange format between pipeline stages and
are also used as Gemini structured-output schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Speaker(BaseModel):
    """Metadata for a single identified speaker."""

    name: str = Field(description="Speaker identifier, e.g. 'Speaker 1'")
    gender: Literal["male", "female", "unknown"] = Field(
        default="unknown",
        description="Detected gender of the speaker based on voice characteristics",
    )


class Segment(BaseModel):
    """A single transcribed speech segment."""

    speaker: str = Field(description="Speaker identifier matching a name in the speakers list")
    timestamp: str = Field(description="Segment timestamp, e.g. '01:23'")
    text: str = Field(description="Transcribed text in the transcript language")


class Transcript(BaseModel):
    """Diarized transcript produced by the STT stage."""

    lang: str = Field(description="Language of the transcribed text")
    speakers: list[Speaker] = Field(
        default_factory=list,
        description="List of identified speakers with gender metadata",
    )
    segments: list[Segment]


class TranslatedSegment(BaseModel):
    """A segment with both original and translated text."""

    speaker: str
    timestamp: str
    original_text: str
    translated_text: str


class TranslatedTranscript(BaseModel):
    """Transcript after the translation stage."""

    source_lang: str
    target_lang: str
    speakers: list[Speaker] = Field(default_factory=list)
    segments: list[TranslatedSegment]
