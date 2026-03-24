"""Abstract base classes for pipeline backends.

Implement these interfaces to add new STT, translation, or TTS providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from audio_translator.models import Transcript, TranslatedTranscript


class STTBackend(ABC):
    """Speech-to-text backend with diarization support."""

    @abstractmethod
    def transcribe(self, input_path: str) -> Transcript:
        """Transcribe audio to a diarized transcript.

        Args:
            input_path: Local file path or URL to an audio source.

        Returns:
            A Transcript with speaker-labeled segments.
        """
        ...


class TranslateBackend(ABC):
    """Text translation backend."""

    @abstractmethod
    def translate(
        self, transcript: Transcript, target_lang: str
    ) -> TranslatedTranscript:
        """Translate transcript segments to the target language.

        Args:
            transcript: Source-language transcript.
            target_lang: Target language name (e.g. "Mandarin Chinese").

        Returns:
            A TranslatedTranscript with both original and translated text.
        """
        ...


class TTSBackend(ABC):
    """Text-to-speech backend."""

    @abstractmethod
    def synthesize(
        self,
        transcript: TranslatedTranscript,
        voice_map: dict[str, str] | None = None,
    ) -> bytes:
        """Synthesize translated transcript to audio.

        Args:
            transcript: Translated transcript with speaker labels.
            voice_map: Optional mapping of speaker name to voice profile name.

        Returns:
            Raw PCM audio bytes (24kHz, 16-bit, mono).
        """
        ...
