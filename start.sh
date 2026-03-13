#!/bin/bash
# WhisperStream Launcher
# Starts the real-time transcription + auto-typer

cd "$(dirname "$0")"

echo "🎙️ WhisperStream - Starting..."

# Activate uv environment and run
uv run python src/whisper_stream.py "$@"