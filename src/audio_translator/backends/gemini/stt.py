"""Gemini STT backend using Gemini 3 Flash Preview.

Supports audio file upload, inline audio data, and URL input
with diarization via structured output.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from google import genai
from google.genai import types

from audio_translator.backends.base import STTBackend
from audio_translator.models import Transcript  # noqa: F401 (Speaker imported via Transcript schema)

# Gemini Files API threshold (20 MB).
_UPLOAD_THRESHOLD = 20 * 1024 * 1024

_SUPPORTED_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

_TRANSCRIBE_PROMPT = """\
Transcribe the speech in this audio with diarization.

Requirements:
1. Identify each distinct speaker and label them as "Speaker 1", "Speaker 2", etc.
2. For each speaker, detect their gender ("male", "female", or "unknown") based on
   voice characteristics such as pitch and timbre.
3. Provide a timestamp (MM:SS format) for each segment.
4. Transcribe the spoken text accurately in the original language.
5. Detect the primary language of the audio and report it.

Output ONLY the JSON matching the provided schema.
"""


def _detect_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in _SUPPORTED_MIME:
        return _SUPPORTED_MIME[ext]
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


class GeminiSTT(STTBackend):
    """Speech-to-text using Gemini 3 Flash Preview with diarization."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        client: genai.Client | None = None,
    ):
        self.model = model
        self.client = client or genai.Client()

    def transcribe(self, input_path: str) -> Transcript:
        if _is_url(input_path):
            audio_part = types.Part(
                file_data=types.FileData(file_uri=input_path)
            )
        else:
            file_size = os.path.getsize(input_path)
            if file_size > _UPLOAD_THRESHOLD:
                uploaded = self.client.files.upload(file=input_path)
                audio_part = types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded.uri,
                        mime_type=uploaded.mime_type,
                    )
                )
            else:
                mime = _detect_mime(input_path)
                data = Path(input_path).read_bytes()
                audio_part = types.Part(
                    inline_data=types.Blob(mime_type=mime, data=data)
                )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    parts=[audio_part, types.Part(text=_TRANSCRIBE_PROMPT)]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Transcript,
            ),
        )

        return Transcript.model_validate_json(response.text)
