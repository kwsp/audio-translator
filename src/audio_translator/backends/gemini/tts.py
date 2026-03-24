"""Gemini TTS backend using Gemini 2.5 Flash Preview TTS.

Supports single-speaker and multi-speaker (up to 2) synthesis
with configurable voice profiles.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from audio_translator.backends.base import TTSBackend
from audio_translator.models import TranslatedTranscript

_DEFAULT_VOICES = {
    "Speaker 1": "Kore",
    "Speaker 2": "Puck",
}


def _get_voice_map(
    speakers: list[str], user_map: dict[str, str] | None
) -> dict[str, str]:
    """Build a speaker→voice mapping, falling back to defaults."""
    voice_map: dict[str, str] = {}
    defaults = list(_DEFAULT_VOICES.values())
    for i, speaker in enumerate(speakers):
        if user_map and speaker in user_map:
            voice_map[speaker] = user_map[speaker]
        elif speaker in _DEFAULT_VOICES:
            voice_map[speaker] = _DEFAULT_VOICES[speaker]
        else:
            # Cycle through default voices for extra speakers.
            voice_map[speaker] = defaults[i % len(defaults)]
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
        resolved_map = _get_voice_map(speakers, voice_map)
        prompt = _build_tts_prompt(transcript)

        if len(speakers) <= 1:
            speech_config = self._single_speaker_config(
                resolved_map.get(speakers[0], "Kore") if speakers else "Kore"
            )
        else:
            speech_config = self._multi_speaker_config(
                speakers[:2], resolved_map
            )

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
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
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
