# 🎙️ Local Voice Agent

A real-time voice agent running **entirely locally** (except the LLM), orchestrated by **Pipecat**.

## Stack

| Component | Technology | Runs |
|-----------|-----------|------|
| **Orchestration** | Pipecat | Local |
| **Transport** | SmallWebRTCTransport | Local (no API key) |
| **VAD** | Silero VAD | Local (CPU) |
| **Turn Detection** | Smart Turn v3 | Local (CPU, 12ms) |
| **STT** | Qwen3-ASR-0.6B | Local (Transformers) |
| **LLM** | Gemini 2.0 Flash | Cloud |
| **TTS** | PocketTTS | Local (CPU) |

## Setup

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- NVIDIA GPU recommended (for Qwen3-ASR), CPU works too
- Google Gemini API key

### Install & Run
```bash
# Install dependencies (auto-creates venv)
uv sync

# Configure API key
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run the agent
uv run python bot.py
```

Then open **http://localhost:7860** in your browser and click the mic to start talking.

## Architecture

```
Browser (WebRTC) → SmallWebRTCTransport → Silero VAD + Smart Turn v3
    → Qwen3-ASR STT → Gemini LLM → PocketTTS → WebRTC → Browser Speaker
```

## Notes
- **First run** downloads model weights (~1-2GB for Qwen3-ASR, ~400MB for PocketTTS)
- **PocketTTS** is English-only at the moment
- **Smart Turn v3** analyzes linguistic cues to detect natural turn endings, not just silence
- **Interruptions** are enabled — speak while the agent is responding to cut it off
