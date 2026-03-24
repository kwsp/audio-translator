# Audio Translator

A Python CLI application and library that performs an end-to-end audio translation pipeline: **Speech-to-Text (STT)** → **Translation** → **Text-to-Speech (TTS)**.

Powered by Google's Gemini models, it supports diarization, speaker gender detection, and multi-speaker voice synthesis.

## Features

- **Diarization**: Automatically detects multiple speakers in an audio file.
- **Gender-Aware TTS**: Identifies the gender of each speaker and assigns an appropriate voice (e.g., `Charon` for male, `Sulafat` for female).
- **Timestamp Alignment**: Matches translated audio length to original speech segments for video synchronization using the `--align` flag.
- **Pipeline Flexibility**: 
  - Skip translation if the source and target languages match.
  - Provide a pre-transcribed JSON file to skip the STT stage.
- **Backend Agnostic**: Stages are defined by abstract base classes (`STTBackend`, `TranslateBackend`, `TTSBackend`), making it easy to swap Gemini out for local models in the future.

## Prerequisites

1. **Python 3.11+**
2. **FFmpeg**: Required for audio format conversion.
   - macOS: `brew install ffmpeg`
   - Linux (Ubuntu): `sudo apt install ffmpeg`
3. **Google API Key**: You need access to the Gemini API. Get a key from [Google AI Studio](https://aistudio.google.com/).

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd audio-translator
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the package using `uv` (or `pip`):
   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file in the root directory and add your API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Once installed, the `audio-translator` CLI command will be available.

### Basic Audio Translation
Translate an English audio file to Mandarin Chinese (default target):
```bash
audio-translator path/to/audio.mp3
```

Translate to a specific language and specify an output directory:
```bash
audio-translator interview.mp3 --target-lang "Spanish" -o output_spanish/
```

### Outputs
The pipeline writes all intermediate and final artifacts into a single directory (defaulting to the input filename stem):
```text
output_dir/
├── transcript.json            # Original STT output with speaker diarization and gender
├── translated_transcript.json # Segment-by-segment translation
└── audio.mp3                  # Final synthesized multi-speaker audio
```

### Advanced Usage

**Timestamp Alignment (for Video Matching)**
Inserts silence between speech segments to match the original audio's timestamps:
```bash
audio-translator video_audio.mp3 --align
```

**Override Voices**
Manually assign voices to specific speakers using a JSON string:
```bash
audio-translator podcast.mp3 --voice-map '{"Speaker 1": "Kore", "Speaker 2": "Aoede"}'
```

**Skip STT (Use existing transcript)**
If you already have a `transcript.json` and want to re-run translation/TTS:
```bash
audio-translator path/to/transcript.json --transcript --target-lang "French"
```

## Running Tests

The project includes a comprehensive `pytest` suite covering Pydantic models, audio utilities, pipeline orchestration, and CLI parsing.

1. Install test dependencies:
   ```bash
   uv pip install pytest pytest-mock
   ```

2. Run the tests:
   ```bash
   pytest tests/ -v
   ```
