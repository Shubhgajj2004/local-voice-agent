# 🎙️ Local Voice Agent

Real-time local voice agent orchestrated by **Pipecat**.

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   Create `.env` with your `GOOGLE_API_KEY`.

3. **Run**:
   ```bash
   uv run python bot.py
   ```

4. **Connect**:
   Open **http://localhost:7860** in your browser.

## Features
- **Local STT**: Qwen3-ASR-0.6B
- **Local TTS**: PocketTTS (Low Latency)
- **Linguistic Turn Detection**: Smart Turn v3
- **Peer-to-Peer**: SmallWebRTC (No API key needed)
