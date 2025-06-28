# MCP Speech-to-Text Server

This project provides an MCP (Model Context Protocol) server for Speech-to-Text using OpenAI Whisper, FastMCP, and Voice Activity Detection (VAD). It allows MCP clients to send speech data for real-time transcription.

Once this MCP server is added to a vide coding IDE like Cursor, the user can talk to the Cursor agent instead of typing.

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
To enable voice control within Cursor, you can configure a project specific rule such as the one in [voice_control_rules.mdc](https://github.com/CalvHobbes/vibe-coding-speech-to-text/blob/dca06f42a7f20f2831ad4894acc8a7bae707359b/.cursor/rules/voice_control_rules.mdc)


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

## My Observations
1) I'm using a local Whisper medium model on my Mac (Apple M2 Pro/16 GB), so it was a little slow for sure
2) It was very interesting to see what the Whisper model returned when I didn't speak - most times it returned "Thanks for watching", sometimes the equivalent of that in Chinese! Once, it returned "MBC 뉴스 김수근입니다", which is Korean for ""This is MBC News, Kim Soo-geun.". I get the model hallucinating to return "Thanks for watching", as that's apparently a very frequent phrase it saw in its training set, but the other two are rather intriguing. 
