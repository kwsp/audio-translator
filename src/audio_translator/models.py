"""Pydantic data models for the audio-translator pipeline.

These models serve as the interchange format between pipeline stages and
are also used as Gemini structured-output schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A single transcribed speech segment."""

    speaker: str = Field(description="Speaker identifier, e.g. 'Speaker 1'")
    timestamp: str = Field(description="Segment timestamp, e.g. '01:23'")
    text: str = Field(description="Transcribed text in the transcript language")


class Transcript(BaseModel):
    """Diarized transcript produced by the STT stage."""

    lang: str = Field(description="Language of the transcribed text")
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
    segments: list[TranslatedSegment]
