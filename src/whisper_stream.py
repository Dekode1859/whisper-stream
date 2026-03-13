#!/usr/bin/env python3
"""
WhisperStream - Real-time Transcription + Auto-Typer
Hotkey: Ctrl+Shift+Space to start/stop listening
"""

import sounddevice as sd
import numpy as np
from collections import deque
import time
import threading
import pyautogui
from pynput import keyboard

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds per audio chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Audio buffer
audio_buffer = deque(maxlen=int(SAMPLE_RATE * 3.0))

# State
is_listening = False
stream = None


def audio_callback(indata, frames, time_info, status):
    """Capture audio chunks while listening."""
    if status:
        print(f"Audio status: {status}")
    
    # Convert to mono float32
    if len(indata.shape) > 1:
        indata = indata[:, 0]
    
    if indata.dtype != np.float32:
        indata = indata.astype(np.float32)
    
    if indata.max() > 1.0:
        indata = indata / 32768.0
    
    audio_buffer.extend(indata)


def transcribe_buffer():
    """Transcribe the current audio buffer."""
    if len(audio_buffer) < SAMPLE_RATE * 0.5:  # Need at least 0.5s of audio
        return None
    
    # Get audio data
    audio_data = np.array(audio_buffer)
    
    # TODO: Replace with actual whisper.cpp call
    # For now, return placeholder
    # import whispercpp
    # model = whispercpp.Model.from_pretrained("base")
    # return model.transcribe(audio_data)
    
    # Placeholder: just return empty for testing
    return ""


def type_text(text):
    """Type text to active window."""
    if text:
        pyautogui.write(text, interval=0.02)


def toggle_listening():
    """Toggle listening state."""
    global is_listening, stream
    
    if is_listening:
        # Stop listening
        if stream:
            stream.stop()
            stream.close()
            stream = None
        is_listening = False
        print("\n🛑 Stopped listening")
    else:
        # Start listening
        audio_buffer.clear()
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype=np.float32,
            callback=audio_callback
        )
        stream.start()
        is_listening = True
        print("\n🎙️ Listening... (Ctrl+Shift+Space to stop)")


def on_activate():
    """Called when hotkey is pressed."""
    print("\n" + "=" * 40)
    toggle_listening()


def main_loop():
    """Main loop - process audio and type."""
    print("""
╔══════════════════════════════════════════╗
║     WhisperStream - Auto-Typer           ║
╠══════════════════════════════════════════╣
║  Press Ctrl+Shift+Space to start/stop   ║
║  Place cursor in text field to type      ║
║  Press Ctrl+C to exit                    ║
╚══════════════════════════════════════════╝
""")
    
    # Setup keyboard listener
    def for_canonical(f):
        return lambda k: f(listener.canonical(k))
    
    hotkey = {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.Key.space}
    current = set()
    
    def on_press(key):
        current.add(key)
        if current == hotkey:
            on_activate()
    
    def on_release(key):
        current.discard(key)
    
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()
    
    # Main processing loop
    try:
        while True:
            if is_listening and len(audio_buffer) >= SAMPLE_RATE * 0.5:
                # Transcribe
                text = transcribe_buffer()
                if text:
                    print(f"📝 Transcribed: {text[:50]}...")
                    type_text(text)
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        if stream:
            stream.stop()
            stream.close()
        listener.stop()


if __name__ == "__main__":
    main_loop()