---
name: audio-translator
description: AI-powered audio translation pipeline (STT → Translate → TTS) with speaker diarization and gender-aware voice synthesis.
---

# Audio Translator

Use this skill to translate spoken audio while preserving speaker identity and timing.

## Core Capabilities

- **Pipeline**: Automated STT (Gemini), Translation (Gemini), and TTS (Edge-TTS or Gemini).
- **Diarization**: Automatically detects and handles multiple speakers.
- **Gender-Aware**: Assigns gender-appropriate voices.
- **Flexible Entry**: Start from audio, plain text, a transcript JSON, or a translated transcript JSON — auto-detected from file type.

## Quick Start (CLI)

### Install
```bash
uv tool install git+https://github.com/kwsp/audio-translator
```

### Usage
```bash
# Full translation pipeline (audio → Mandarin Chinese by default)
audio-translator input.mp3

# Specify output directory and target language
audio-translator input.mp3 -o output/ --target-lang "Spanish"

# Use Gemini TTS for higher-fidelity voices (requires Google API key)
audio-translator input.mp3 --tts-backend gemini
```

## Input Auto-Detection

The pipeline stage is chosen automatically from the input:

| Input | Pipeline |
|---|---|
| Audio file (`.mp3`, `.wav`, `.m4a`, …) or URL | STT → Translate → TTS |
| Text file (`.txt`, `.md`) | Translate → TTS (STT skipped) |
| Transcript JSON (`{"lang": …}`) | Translate → TTS (STT skipped) |
| Translated transcript JSON (`{"source_lang": …}`) | TTS only |
| `--text "…"` or piped stdin | Translate → TTS (STT skipped) |

```bash
# Resume from an existing transcript (skip STT)
audio-translator transcript.json --target-lang "Spanish"

# TTS only from an already-translated transcript
audio-translator translated_transcript.json

# Inline text
audio-translator --text "Hello world" --target-lang "French"

# Piped stdin
echo "Hello world" | audio-translator --target-lang "French"
```

## Backend Selection

- **STT**: `--stt-backend [gemini]` (default: gemini)
- **TTS**: `--tts-backend [gemini|edge]` (default: edge)
  - `edge` is free and doesn't require a Google API key.
  - `gemini` offers higher-fidelity neural voices but requires a Google API key.

## Manual Voice Mapping

```bash
audio-translator input.mp3 --voice-map '{"Speaker 1": "Aoede", "Speaker 2": "Puck"}'
```

## Data Models

- **Transcript**: `lang`, `speakers` (name, gender), `segments` (speaker, timestamp, text).
- **TranslatedTranscript**: `source_lang`, `target_lang`, `speakers`, `segments` (speaker, timestamp, original_text, translated_text).

## Best Practices for Agents

1. **Verification**: Always check if `ffmpeg` is installed before running synthesis.
2. **Cost Optimization**: The default Edge-TTS backend is free. Use `--tts-backend gemini` only when higher-fidelity voices are required.
3. **Context**: Output directory contains all intermediate JSONs; use them to debug or resume failed runs.
