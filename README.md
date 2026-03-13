# WhisperStream

Real-time audio transcription that types what you speak directly into the active window.

## Features

- 🎙️ Real-time transcription using Whisper.cpp
- ⌨️ Auto-types to cursor position
- 🔥 Hotkey trigger: `Ctrl+Shift+Space` to start/stop
- ⚡ Fast local inference (no cloud needed)

## Quick Start

```bash
# Make start.sh executable
chmod +x start.sh

# Run the app
./start.sh
```

## Usage

1. Run the script
2. Place your cursor in any text field (browser, editor, etc.)
3. Press `Ctrl+Shift+Space` to start listening
4. Speak — your words will be typed automatically
5. Press `Ctrl+Shift+Space` again to stop

## Requirements

- Python 3.10+
- Microphone
- Keyboard (for hotkey trigger)

## Architecture

```
Microphone → Audio Buffer → Whisper.cpp → pyautogui → Active Window
```

## License

MIT