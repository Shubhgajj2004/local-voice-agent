# 🎙️ Local Voice Agent

Real-time local voice agent orchestrated by **Pipecat**.

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   Create `.env` with your `GOOGLE_API_KEY`.

3. **Pre-download models (recommended)**:
   ```bash
   uv run python prepare_models.py
   ```

4. **Run**:
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

## 🧠 Using a Local LLM (Optional)

You can swap Google Gemini for a local OpenAI-compatible LLM (like Ollama, vLLM, or LlamaCpp) by modifying `bot.py`.

1.  **Open `bot.py`** and import the OpenAI service:
    ```python
    from pipecat.services.openai.llm import OpenAILLMService
    ```

2.  **Replace `GoogleLLMService`** with `OpenAILLMService`:
    ```python
    # ── LLM (Local OpenAI Compatible) ──
    llm = OpenAILLMService(
        model="llama3-8b",  # Your local model name
        api_key="dummy",    # Local servers usually ignore this
        base_url="http://localhost:8000/v1",  # Your local server URL
    )
    ```
