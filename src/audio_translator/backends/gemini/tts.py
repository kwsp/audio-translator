"""Gemini TTS backend using Gemini 2.5 Flash Preview TTS.

Supports single-speaker and multi-speaker (up to 2) synthesis.
Voice profiles are selected based on detected speaker gender from the Transcript.
Supports two distinct voices per gender for cases where both speakers share a gender.
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types

from audio_translator.backends.base import TTSBackend
from audio_translator.models import Speaker, TranslatedTranscript

logger = logging.getLogger(__name__)

# Gender-keyed default voices (ordered by preference).
# Two voices per gender handle the case where both speakers share a gender.
_DEFAULT_VOICES_BY_GENDER: dict[str, list[str]] = {
    "female":  ["Sulafat", "Aoede"],   # warm female, expressive female
    "male":    ["Charon",  "Puck"],    # calm resonant, clear natural
    "unknown": ["Kore"],               # neutral fallback
}


def _build_gender_map(speakers: list[Speaker]) -> dict[str, str]:
    """Build speaker_name → gender lookup from speaker metadata."""
    return {s.name: s.gender for s in speakers}


def _get_voice_map(
    speaker_names: list[str],
    gender_map: dict[str, str],
    user_map: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve speaker → voice name, respecting user overrides and gender.

    Speakers of the same gender each receive a distinct voice, cycling through
    the pool (e.g., two females → Sulafat, Aoede).
    """
    voice_map: dict[str, str] = {}
    gender_usage: dict[str, int] = {}  # count how many speakers per gender

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


def _build_tts_prompt(transcript: TranslatedTranscript) -> str:
    """Build the TTS input prompt from translated segments."""
    lines: list[str] = []
    for seg in transcript.segments:
        lines.append(f"{seg.speaker}: {seg.translated_text}")
    return "\n".join(lines)


class GeminiTTS(TTSBackend):
    """Text-to-speech using Gemini 2.5 Flash Preview TTS."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-preview-tts",
        client: genai.Client | None = None,
    ):
        self.model = model
        self.client = client or genai.Client()

    def synthesize(
        self,
        transcript: TranslatedTranscript,
        voice_map: dict[str, str] | None = None,
    ) -> bytes:
        speakers = sorted({seg.speaker for seg in transcript.segments})
        gender_map = _build_gender_map(transcript.speakers)
        resolved_map = _get_voice_map(speakers, gender_map, voice_map)
        prompt = _build_tts_prompt(transcript)

        for speaker, voice in resolved_map.items():
            gender = gender_map.get(speaker, "unknown")
            logger.info("  %s (%s) → voice: %s", speaker, gender, voice)

        if len(speakers) <= 1:
            speech_config = self._single_speaker_config(
                resolved_map.get(speakers[0], "Kore") if speakers else "Kore"
            )
        else:
            speech_config = self._multi_speaker_config(speakers[:2], resolved_map)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            ),
        )

        return response.candidates[0].content.parts[0].inline_data.data

    @staticmethod
    def _single_speaker_config(voice_name: str) -> types.SpeechConfig:
        return types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )

    @staticmethod
    def _multi_speaker_config(
        speakers: list[str], voice_map: dict[str, str]
    ) -> types.SpeechConfig:
        return types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker=speaker,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_map[speaker]
                            )
                        ),
                    )
                    for speaker in speakers
                ]
            )
        )
