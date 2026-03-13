#!/usr/bin/env python3
"""
WhisperStream - Real-time Transcription + Auto-Typer
Hotkey: Ctrl+Shift+Space to start/stop listening

Strategy: Wait for brief silence, then type the stable transcription
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

# Silence detection
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence
SILENCE_DURATION = 1.0  # seconds of silence before typing

# Audio buffer and silence tracking
audio_chunks = deque(maxlen=10)
silence_start_time = None

# State
is_listening = False
stream = None
transcription_queue = queue.Queue()

# Track what's typed
typed_text = ""

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
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ Whisper model loaded")
    return whisper_model


def audio_callback(indata, frames, time_info, status):
    """Capture audio chunks while listening."""
    global silence_start_time
    
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
    
    # Check for speech/silence
    rms = np.sqrt(np.mean(indata ** 2))
    
    if rms > SILENCE_THRESHOLD:
        # Speech detected
        if silence_start_time is not None:
            silence_start_time = None  # Reset silence timer
        audio_chunks.append(indata.copy())
    else:
        # Silence
        if silence_start_time is None:
            silence_start_time = time.time()


def transcribe_and_wait():
    """Transcribe when there's silence, then type."""
    global typed_text
    
    while True:
        if not is_listening:
            time.sleep(0.1)
            continue
        
        # Check if we've had enough silence to type
        if silence_start_time and (time.time() - silence_start_time) >= SILENCE_DURATION:
            if len(audio_chunks) >= 2:
                # Combine all chunks
                audio_data = np.concatenate(list(audio_chunks))
                
                try:
                    model = load_whisper_model()
                    segments, info = model.transcribe(audio_data, beam_size=5, language="en")
                    
                    text = " ".join([seg.text for seg in segments]).strip()
                    
                    if text and text != typed_text:
                        # Find what to add
                        new_text = text[len(typed_text):].strip()
                        
                        if new_text and len(new_text) >= 2:
                            print(f"📝 Typing (after silence): {new_text}")
                            pyautogui.write(new_text, interval=0.02)
                            typed_text = text
                
                except Exception as e:
                    print(f"Transcription error: {e}")
                
                # Reset after typing
                audio_chunks.clear()
                silence_start_time = None
        
        time.sleep(0.1)


def toggle_listening():
    """Toggle listening state."""
    global is_listening, stream, typed_text, silence_start_time, audio_chunks
    
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
        silence_start_time = None
        typed_text = ""
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
        print("\n🎙️ Listening... (speak, then pause to type)")


def on_activate():
    """Called when hotkey is pressed."""
    print("=" * 40)
    toggle_listening()


def main_loop():
    """Main loop."""
    print("""
╔══════════════════════════════════════════╗
║     WhisperStream - Auto-Typer           ║
╠══════════════════════════════════════════╣
║  Press Ctrl+Shift+Space to start/stop   ║
║  Speak, then pause briefly to type       ║
║  Press Ctrl+C to exit                    ║
╚══════════════════════════════════════════╝
""")
    
    # Start transcription thread
    t = threading.Thread(target=transcribe_and_wait, daemon=True)
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
    
    try:
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        if stream:
            stream.stop()
            stream.close()
        listener.stop()


if __name__ == "__main__":
    main_loop()