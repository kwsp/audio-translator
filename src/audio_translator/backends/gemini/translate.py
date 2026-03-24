"""Gemini translation backend using Gemini 2.5 Flash.

Translates a diarized transcript to a target language in a single API call
using structured output.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from audio_translator.backends.base import TranslateBackend
from audio_translator.models import Transcript, TranslatedTranscript


def _build_prompt(transcript: Transcript, target_lang: str) -> str:
    lines: list[str] = []
    for seg in transcript.segments:
        lines.append(f"[{seg.timestamp}] {seg.speaker}: {seg.text}")

    transcript_block = "\n".join(lines)
    return (
        f"Translate the following {transcript.lang} transcript to {target_lang}.\n"
        f"Preserve speaker labels, timestamps, and the original text.\n"
        f"Provide both the original_text and the translated_text for each segment.\n\n"
        f"{transcript_block}"
    )


class GeminiTranslate(TranslateBackend):
    """Translation using Gemini 2.5 Flash with structured output."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        client: genai.Client | None = None,
    ):
        self.model = model
        self.client = client or genai.Client()

    def translate(
        self, transcript: Transcript, target_lang: str
    ) -> TranslatedTranscript:
        prompt = _build_prompt(transcript, target_lang)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TranslatedTranscript,
            ),
        )

        return TranslatedTranscript.model_validate_json(response.text)
