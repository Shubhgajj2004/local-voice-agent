"""
Local Voice Agent — Main Pipeline

Orchestrates a real-time voice agent using Pipecat:
  - Transport:  SmallWebRTCTransport (no API key, peer-to-peer)
  - VAD:        Silero VAD (built-in to Pipecat)
  - Turn:       Smart Turn v3 (ML-based end-of-speech detection)
  - STT:        Qwen3-ASR-0.6B (local, GPU/CPU)
  - LLM:        Google Gemini (cloud, streaming)
  - TTS:        PocketTTS (local, CPU, streaming)

Usage:
    python bot.py
    Then open http://localhost:7860 in your browser.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from services.qwen_stt import QwenSTTService
from services.pocket_tts_service import PocketTTSService

from aiohttp import web
import aiohttp_cors

load_dotenv()

# ─── Configuration ─────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

SYSTEM_PROMPT = """You are a helpful, friendly, and concise voice assistant. 
Keep your responses short and conversational — typically 1-3 sentences.
You are talking to a user through a microphone, so be natural and engaging.
Do not use markdown formatting, bullet points, or numbered lists in your responses.
Just speak naturally as you would in a conversation."""


# ─── Bot Entry Point (called by Pipecat runner) ───────
async def bot(webrtc_connection):
    """Create and run the voice agent pipeline.
    
    This function is called by Pipecat's development runner for each
    new WebRTC connection from the browser.
    """
    logger.info("🤖 New voice agent session started")

    # ── Transport (WebRTC — No API key needed) ──
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,  # PocketTTS native rate
        ),
    )

    # ── STT (Qwen3-ASR-0.6B — local) ──
    stt = QwenSTTService(
        model_name="Qwen/Qwen3-ASR-0.6B",
        language=None,  # Auto-detect language
    )

    # ── LLM (Gemini — cloud, streaming) ──
    llm = GoogleLLMService(
        api_key=GOOGLE_API_KEY,
        model="gemini-2.0-flash",
    )

    # ── TTS (PocketTTS — local, CPU, streaming) ──
    tts = PocketTTSService(
        voice="alba",
    )

    # ── Conversation Context ──
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    context = LLMContext(messages)

    # ── Smart Turn v3 + Silero VAD ──
    # Silero detects silence (200ms threshold), then Smart Turn ML model
    # decides if the user is truly done speaking or just pausing.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.2)
            ),
        ),
    )

    # ── Pipeline Assembly ──
    # Data flows: mic → STT → context → LLM → TTS → speaker
    pipeline = Pipeline(
        [
            transport.input(),      # Receive audio from browser (WebRTC)
            stt,                    # Qwen3-ASR-0.6B (local STT)
            user_aggregator,        # Smart Turn v3 + Silero VAD + context
            llm,                    # Gemini (cloud LLM, streaming)
            tts,                    # PocketTTS (local TTS, streaming)
            transport.output(),     # Send audio to browser (WebRTC)
            assistant_aggregator,   # Track assistant responses in context
        ]
    )

    # ── Run ──
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("🟢 Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("🔴 Client disconnected")
        await task.cancel()

    await runner.run(task)
    logger.info("🤖 Voice agent session ended")


# ─── Development Runner ────────────────────────────────
if __name__ == "__main__":
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    logger.info("=" * 60)
    logger.info("🎙️  Local Voice Agent")
    logger.info("   STT:  Qwen3-ASR-0.6B (local)")
    logger.info("   LLM:  Gemini 2.0 Flash (cloud)")
    logger.info("   TTS:  PocketTTS (local)")
    logger.info("   VAD:  Silero + Smart Turn v3")
    logger.info("=" * 60)

    async def main():
        # Setup the WebRTC connection handler
        connection = SmallWebRTCConnection()
        
        # Web Server Handlers
        async def handle_index(request):
            return web.FileResponse(os.path.join("static", "index.html"))

        async def handle_offer(request):
            data = await request.json()
            # Initialize with offer from browser
            await connection.initialize(sdp=data["sdp"], type=data["type"])
            # Get response answer
            answer = connection.get_answer()
            if not answer:
                return web.json_response({"error": "Failed to get answer"}, status=500)
            
            # Start the bot task asynchronously for this connection
            asyncio.create_task(bot(connection))
            
            return web.json_response(answer)

        # Setup Aiohttp App
        app = web.Application()
        
        # Enable CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })

        app.router.add_get("/", handle_index)
        app.router.add_post("/offer", handle_offer)
        
        for route in list(app.router.routes()):
            cors.add(route)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", 7860)
        
        logger.info("=" * 60)
        logger.info("🎙️  Local Voice Agent Started")
        logger.info("🚀 Server: http://localhost:7860")
        logger.info("=" * 60)
        
        await site.start()
        
        # Keep the server running
        while True:
            await asyncio.sleep(3600)

    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
