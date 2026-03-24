"""Unit tests for the Gemini backends with mocking."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from audio_translator.backends.gemini.stt import GeminiSTT
from audio_translator.backends.gemini.translate import GeminiTranslate
from audio_translator.backends.gemini.tts import GeminiTTS
from audio_translator.models import Speaker, Transcript, TranslatedSegment, TranslatedTranscript


def test_gemini_stt_backend(mocker, tmp_path):
    # Create a dummy audio file to avoid FileNotFoundError
    dummy_audio = tmp_path / "dummy.mp3"
    dummy_audio.write_bytes(b"fake_audio_data")

    mock_client = MagicMock()
    # Mocking the response texture from Gemini
    mock_response = MagicMock()
    mock_response.text = '{"lang":"English","speakers":[{"name":"Speaker 1","gender":"male"}],"segments":[{"speaker":"Speaker 1","timestamp":"00:00","text":"Hello"}]}'
    mock_client.models.generate_content.return_value = mock_response

    stt = GeminiSTT(client=mock_client)
    transcript = stt.transcribe(str(dummy_audio))

    assert isinstance(transcript, Transcript)
    assert transcript.lang == "English"
    assert transcript.segments[0].text == "Hello"
    assert transcript.speakers[0].gender == "male"
    assert mock_client.models.generate_content.called


def test_gemini_stt_malformed_json(mocker, tmp_path):
    dummy_audio = tmp_path / "dummy.mp3"
    dummy_audio.write_bytes(b"data")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = 'invalid json'
    mock_client.models.generate_content.return_value = mock_response

    stt = GeminiSTT(client=mock_client)
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        stt.transcribe(str(dummy_audio))


def test_gemini_stt_large_file_upload(mocker, tmp_path):
    dummy_audio = tmp_path / "large.mp3"
    dummy_audio.write_bytes(b"data")

    mock_client = MagicMock()
    # Mocking os.path.getsize to simulate a large file
    mocker.patch("os.path.getsize", return_value=30 * 1024 * 1024)
    # Mocking upload return value
    mock_file = MagicMock()
    mock_file.uri = "https://gemini/file/123"
    mock_file.mime_type = "audio/mp3"
    mock_client.files.upload.return_value = mock_file
    
    # Typical response
    mock_response = MagicMock()
    mock_response.text = '{"lang":"en","speakers":[],"segments":[]}'
    mock_client.models.generate_content.return_value = mock_response

    stt = GeminiSTT(client=mock_client)
    stt.transcribe(str(dummy_audio))

    # Verify upload was called
    mock_client.files.upload.assert_called_once()
    # Verify generate_content used the file URI
    call_args = mock_client.models.generate_content.call_args
    parts = call_args[1]["contents"][0].parts
    file_data = next(p.file_data for p in parts if p.file_data)
    assert file_data.file_uri == "https://gemini/file/123"


def test_gemini_translate_backend(mocker):
    mock_client = MagicMock()
    # Mocking translated output
    mock_response = MagicMock()
    mock_response.text = '{"source_lang":"English","target_lang":"Mandarin Chinese","segments":[{"speaker":"Speaker 1","timestamp":"00:00","original_text":"Hello","translated_text":"你好"}]}'
    mock_client.models.generate_content.return_value = mock_response

    translator = GeminiTranslate(client=mock_client)
    t = Transcript(
        lang="English",
        speakers=[Speaker(name="Speaker 1", gender="male")],
        segments=[{"speaker": "Speaker 1", "timestamp": "00:00", "text": "Hello"}],
    )
    result = translator.translate(t, "Mandarin Chinese")

    assert isinstance(result, TranslatedTranscript)
    assert result.target_lang == "Mandarin Chinese"
    assert result.segments[0].translated_text == "你好"
    # Ensure speaker metadata was copied over.
    assert result.speakers[0].gender == "male"
    assert mock_client.models.generate_content.called


def test_gemini_tts_backend(mocker):
    mock_client = MagicMock()
    # Mocking audio data
    mock_response = MagicMock()
    mock_response.candidates[0].content.parts[0].inline_data.data = b"PCM_DATA"
    mock_client.models.generate_content.return_value = mock_response

    tts = GeminiTTS(client=mock_client)
    tt = TranslatedTranscript(
        source_lang="en",
        target_lang="zh",
        speakers=[Speaker(name="Speaker 1", gender="male")],
        segments=[
            TranslatedSegment(
                speaker="Speaker 1",
                timestamp="00:00",
                original_text="Hi",
                translated_text="你好",
            )
        ],
    )
    pcm = tts.synthesize(tt)

    assert pcm == b"PCM_DATA"
    # Check if correct model and modalities were passed.
    config_call = mock_client.models.generate_content.call_args[1]["config"]
    assert "AUDIO" in config_call.response_modalities
    assert config_call.speech_config.voice_config is not None


def test_voice_mapping_logic():
    from audio_translator.backends.gemini.tts import _build_gender_map, _get_voice_map

    speakers = [
        Speaker(name="Man", gender="male"),
        Speaker(name="Woman", gender="female"),
        Speaker(name="Mystery", gender="unknown"),
    ]
    gmap = _build_gender_map(speakers)

    # 1. Test gender-based voice assignment
    vmap = _get_voice_map(["Man", "Woman", "Mystery"], gmap, None)
    assert vmap["Man"] == "Charon"
    assert vmap["Woman"] == "Sulafat"
    assert vmap["Mystery"] == "Kore"

    # 2. Test user override
    vmap_override = _get_voice_map(["Man"], gmap, {"Man": "Puck"})
    assert vmap_override["Man"] == "Puck"


def test_gemini_same_gender_two_speakers():
    """Two speakers of the same gender should each get a distinct voice."""
    from audio_translator.backends.gemini.tts import _build_gender_map, _get_voice_map

    speakers = [
        Speaker(name="F1", gender="female"),
        Speaker(name="F2", gender="female"),
    ]
    gmap = _build_gender_map(speakers)
    vmap = _get_voice_map(["F1", "F2"], gmap, None)
    assert vmap["F1"] == "Sulafat"
    assert vmap["F2"] == "Aoede"

    speakers_male = [
        Speaker(name="M1", gender="male"),
        Speaker(name="M2", gender="male"),
    ]
    gmap_male = _build_gender_map(speakers_male)
    vmap_male = _get_voice_map(["M1", "M2"], gmap_male, None)
    assert vmap_male["M1"] == "Charon"
    assert vmap_male["M2"] == "Puck"
