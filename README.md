# audio-translator

A Python CLI application and library that performs an end-to-end audio translation pipeline: **Speech-to-Text (STT)** → **Translation** → **Text-to-Speech (TTS)**.

Powered by Google's Gemini models for STT and translation, with Edge-TTS (free, no API key) as the default synthesis backend.

## Features

- **Diarization**: Automatically detects multiple speakers in an audio file.
- **Gender-Aware TTS**: Identifies the gender of each speaker and assigns an appropriate voice.
- **Auto-Detection**: Input type (audio, text, transcript, translated transcript) is detected automatically from the file extension or content.
- **Flexible Entry**: Start the pipeline at any stage — skip STT, skip translation, or run TTS only.
- **Backend Agnostic**: Stages are defined by abstract base classes (`STTBackend`, `TranslateBackend`, `TTSBackend`), making it easy to swap backends.

## Install (CLI)
```bash
uv tool install git+https://github.com/kwsp/audio-translator
```

## Prerequisites

1. **Python 3.11+**
2. **FFmpeg**: Required for audio format conversion.
   - macOS: `brew install ffmpeg`
   - Linux (Ubuntu): `sudo apt install ffmpeg`
3. **Google API Key** *(optional)*: Required only for Gemini STT/translation or `--tts-backend gemini`. Get a key from [Google AI Studio](https://aistudio.google.com/).

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd audio-translator
   ```

2. Install the package:
   ```bash
   uv pip install -e .
   ```

3. If using Gemini STT or translation, create a `.env` file with your API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Once installed, the `audio-translator` CLI command will be available.

The pipeline stage is chosen automatically from the input:

| Input | Pipeline |
|---|---|
| Audio file (`.mp3`, `.wav`, `.m4a`, …) or URL | STT → Translate → TTS |
| Text file (`.txt`, `.md`) | Translate → TTS |
| Transcript JSON (`{"lang": …}`) | Translate → TTS |
| Translated transcript JSON (`{"source_lang": …}`) | TTS only |
| `--text "…"` or piped stdin | Translate → TTS |

### Basic Audio Translation

Translate an audio file to Mandarin Chinese (default target):
```bash
audio-translator interview.mp3
```

Specify target language and output directory:
```bash
audio-translator interview.mp3 --target-lang "Spanish" -o output/
```

### Text Input

Translate a plain-text file:
```bash
audio-translator passage.txt --target-lang "French"
```

Inline text or piped stdin:
```bash
audio-translator --text "Hello world" --target-lang "Japanese"
echo "Hello world" | audio-translator --target-lang "Japanese"
```

### Resume from Intermediate Files

Skip STT using an existing transcript:
```bash
audio-translator transcript.json --target-lang "French"
```

Run TTS only from an already-translated transcript:
```bash
audio-translator translated_transcript.json
```

### Outputs

All artifacts are written to a single directory (defaulting to the input filename stem):
```text
output_dir/
├── transcript.json            # STT output with speaker diarization and gender
├── translated_transcript.json # Segment-by-segment translation
└── audio.mp3                  # Final synthesized multi-speaker audio
```

### Backend Selection

- **STT**: `--stt-backend [gemini]` (default: `gemini`)
- **TTS**: `--tts-backend [edge|gemini]` (default: `edge`)

```bash
# Use Gemini TTS for higher-fidelity voices (requires API key)
audio-translator interview.mp3 --tts-backend gemini
```

### Voice Mapping

Manually assign voices to specific speakers:
```bash
audio-translator podcast.mp3 --voice-map '{"Speaker 1": "Kore", "Speaker 2": "Aoede"}'
```

## Running Tests

```bash
uv pip install pytest pytest-mock
pytest tests/ -v
```
