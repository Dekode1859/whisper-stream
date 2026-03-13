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
import queue
import pyautogui
from pynput import keyboard

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds per audio chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Audio buffer - store raw chunks
audio_chunks = deque(maxlen=10)  # Keep last 10 chunks (~5 seconds)

# State
is_listening = False
stream = None
transcription_queue = queue.Queue()

# Whisper model (loaded once)
whisper_model = None
model_lock = threading.Lock()


def load_whisper_model():
    """Load faster-whisper model (thread-safe)."""
    global whisper_model
    with model_lock:
        if whisper_model is None:
            print("Loading Whisper model (base)... this may take a moment")
            from faster_whisper import WhisperModel
            # Use base model - downloads automatically on first run
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ Whisper model loaded")
    return whisper_model


def audio_callback(indata, frames, time_info, status):
    """Capture audio chunks while listening."""
    if status:
        print(f"Audio status: {status}")
    
    # Convert to mono float32
    if len(indata.shape) > 1:
        indata = indata[:, 0]
    
    if indata.dtype != np.float32:
        indata = indata.astype(np.float32)
    
    # Normalize
    if indata.max() > 1.0:
        indata = indata / 32768.0
    
    # Check if there's actual speech (simple VAD - volume threshold)
    rms = np.sqrt(np.mean(indata ** 2))
    if rms > 0.01:  # Threshold for speech
        audio_chunks.append(indata.copy())


def transcribe_thread():
    """Background thread for transcription."""
    while True:
        if not is_listening:
            time.sleep(0.1)
            continue
        
        if len(audio_chunks) < 2:  # Need at least 1 second
            time.sleep(0.1)
            continue
        
        # Combine all chunks
        audio_data = np.concatenate(list(audio_chunks))
        
        try:
            model = load_whisper_model()
            segments, info = model.transcribe(audio_data, beam_size=5, language="en")
            
            # Get transcribed text
            text = " ".join([seg.text for seg in segments]).strip()
            
            if text:
                transcription_queue.put(text)
        
        except Exception as e:
            print(f"Transcription error: {e}")
        
        time.sleep(0.5)  # Transcribe every 0.5 seconds


def type_text(text):
    """Type text to active window."""
    if text:
        print(f"📝 Typing: {text}")
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
        audio_chunks.clear()
        while not transcription_queue.empty():
            transcription_queue.get()
        
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
    print("=" * 40)
    toggle_listening()


def main_loop():
    """Main loop - process transcriptions and type."""
    print("""
╔══════════════════════════════════════════╗
║     WhisperStream - Auto-Typer           ║
╠══════════════════════════════════════════╣
║  Press Ctrl+Shift+Space to start/stop   ║
║  Place cursor in text field to type      ║
║  Press Ctrl+C to exit                    ║
╚══════════════════════════════════════════╝
""")
    
    # Start transcription thread
    t = threading.Thread(target=transcribe_thread, daemon=True)
    t.start()
    
    # Setup keyboard listener
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
    
    # Main typing loop
    try:
        while True:
            try:
                text = transcription_queue.get(timeout=0.1)
                type_text(text)
            except queue.Empty:
                pass
    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        if stream:
            stream.stop()
            stream.close()
        listener.stop()


if __name__ == "__main__":
    main_loop()