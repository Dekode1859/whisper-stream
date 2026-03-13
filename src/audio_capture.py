#!/usr/bin/env python3
"""
WhisperStream - Wave 2: Audio Capture
Tests microphone input and buffering
"""

import sounddevice as sd
import numpy as np
from collections import deque
import time

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 0.5  # seconds per chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Audio buffer (stores last N chunks)
BUFFER_DURATION = 3.0  # Keep last 3 seconds of audio
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

audio_buffer = deque(maxlen=BUFFER_SIZE)


def audio_callback(indata, frames, time_info, status):
    """Called for each audio chunk from microphone."""
    if status:
        print(f"Audio status: {status}")
    
    # Convert to mono if stereo
    if len(indata.shape) > 1:
        indata = indata[:, 0]
    
    # Convert to float32 if needed
    if indata.dtype != np.float32:
        indata = indata.astype(np.float32)
    
    # Normalize to -1 to 1 range
    if indata.max() > 1.0:
        indata = indata / 32768.0
    
    # Add to buffer
    audio_buffer.extend(indata)


def list_devices():
    """List available audio input devices."""
    print("\n🎤 Available input devices:")
    devices = sd.query_devices()
    print(devices)
    print(f"\nDefault input: {sd.query_devices(kind='input')['name']}")


def test_mic(duration=5):
    """Test microphone input for specified duration."""
    print(f"\n🎙️ Testing mic for {duration} seconds...")
    print("Speak something! (Press Ctrl+C to stop early)\n")
    
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        device=None,  # Use default
        channels=1,
        dtype=np.float32,
        callback=audio_callback
    ):
        try:
            sd.sleep(duration * 1000)
        except KeyboardInterrupt:
            print("\nStopped.")
    
    print(f"\n✅ Captured {len(audio_buffer) / SAMPLE_RATE:.1f} seconds of audio")
    print(f"   Buffer size: {len(audio_buffer)} samples")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WhisperStream - Audio Capture Test")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--test", type=int, default=5, help="Test mic for N seconds")
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
    else:
        test_mic(args.test)


if __name__ == "__main__":
    main()