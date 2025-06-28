# MCP Speech-to-Text Server

This project provides an MCP (Model Context Protocol) server for Speech-to-Text using OpenAI Whisper, FastMCP, and Voice Activity Detection (VAD). It allows MCP clients to send speech data for real-time transcription.

## Features
- Real-time speech-to-text transcription using Whisper
- Voice Activity Detection for efficient audio processing
- FastMCP integration for seamless communication with MCP clients

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/audiotools.git
cd audiotools
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python3 -m venv mcp_tool_env
source mcp_tool_env/bin/activate
```

### 3. Install FFmpeg (Required for Whisper)
**On macOS (using Homebrew):**
```bash
brew install ffmpeg
```
**On Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```
**On Windows:**
Download from [FFmpeg official site](https://ffmpeg.org/download.html) and add to your PATH.

### 4. Install Python Dependencies
```bash
pip install .
```

## Usage

### MCP Integration
Configure your MCP client to use the `mcp_speech_to_text_server.main` module as the server command, similar to:
```json
{
  "command": "path-to-python",
  "args": ["path-to-main.py"],
  "env": {},
  "transport": "stdio"
}
```

### Cursor Voice Control Rules
To enable voice control within Cursor, you can configure a project specific rule such as the one in voice_control_rules.mdc


## Microphone Test Utility (`mic_test.py`)
The `mic_test.py` script is a simple utility to test your microphone setup. It records a short audio snippet and plays it back. To use it, simply run:
```bash
python mic_test.py
```

## Requirements
- Python 3.8+
- `torch`
- `sounddevice`
- `numpy`
- `scipy`
- `openai-whisper`
- `fastmcp`
- `webrtcvad`
- `FFmpeg` (for Whisper)

## License
MIT

---

For questions or contributions, please open an issue or pull request on GitHub. 
