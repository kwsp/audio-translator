---
name: audio-translator
description: AI-powered audio translation pipeline (STT → Translate → TTS) with speaker diarization, gender-aware voice synthesis, and timestamp alignment.
---

# Audio Translator

Use this skill to translate spoken audio while preserving speaker identity and timing.

## Core Capabilities

- **Pipeline**: Automated STT (Gemini), Translation (Gemini), and TTS (Gemini or Edge-TTS).
- **Diarization**: Automatically detects and handles multiple speakers.
- **Gender-Aware**: Assigns gender-appropriate voices (e.g., male to `Charon`, female to `Sulafat`).
- **Timing**: Use `--align` to match translated audio duration to original speech segments.
- **Flexible Entry**: Start from audio, a raw transcript, or a translated transcript.

## Quick Start (CLI)

```bash
# Full translation to Mandarin (default)
audio-translator input.mp3 -o output/

# Use free Edge-TTS backend for synthesis
audio-translator input.mp3 --tts-backend edge

# Align audio length for video synchronization
audio-translator input.mp3 --align
```

## Backend Selection

- **STT**: `--stt-backend [gemini]` (default: gemini)
- **TTS**: `--tts-backend [gemini|edge]` (default: gemini)
  - `edge` is free and doesn't require a Google API key for the TTS stage.

## Advanced Workflows

### Resume from Transcript (Skip STT)
If `transcript.json` already exists:
```bash
audio-translator transcript.json --transcript --target-lang "Spanish"
```

### TTS-Only (Skip STT + Translate)
If `translated_transcript.json` already exists:
```bash
audio-translator translated.json --translated-transcript
```

### Manual Voice Mapping
```bash
audio-translator input.mp3 --voice-map '{"Speaker 1": "Aoede", "Speaker 2": "Puck"}'
```

## Data Models

- **Transcript**: `source_lang`, `speakers` (name, gender), `segments` (speaker, timestamp, text).
- **TranslatedTranscript**: Adds `target_lang` and `translated_text` to segments.

## Best Practices for Agents

1. **Verification**: Always check if `ffmpeg` is installed before running synthesis.
2. **Cost Optimization**: Use `--tts-backend edge` for large documents if high-fidelity Gemini voices aren't strictly required.
3. **Context**: Output directory contains all intermediate JSONs; use them to debug or resume failed runs.
