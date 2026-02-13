"""
Custom Pipecat TTS Service wrapping PocketTTS.

Uses the pocket-tts package for local, CPU-only text-to-speech.
~200ms latency to first audio chunk, streaming output via generate_audio_stream().
"""

import asyncio
import struct
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class PocketTTSService(TTSService):
    """Text-to-speech service using PocketTTS running locally on CPU.

    The model is loaded once at initialization. Audio is generated using
    the streaming API (generate_audio_stream) for low-latency chunked output.

    Args:
        voice: Voice name or path to WAV file. Built-in voices:
               alba, marius, javert, jean, fantine, cosette, eponine, azelma
        asr_sample_rate: Target sample rate to resample output to (for Pipecat transport).
                         PocketTTS outputs at its native rate, we resample if needed.
    """

    def __init__(
        self,
        *,
        voice: str = "alba",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._voice_name = voice
        self._model = None
        self._voice_state = None

    async def start(self, frame: Frame):
        await super().start(frame)
        logger.info(f"Loading PocketTTS model with voice: {self._voice_name}...")
        await asyncio.get_event_loop().run_in_executor(None, self._load_model)
        logger.info("PocketTTS model loaded successfully.")

    def _load_model(self):
        """Load PocketTTS model and voice state (blocking, runs in executor)."""
        from pocket_tts import TTSModel

        self._model = TTSModel.load_model()
        self._voice_state = self._model.get_state_for_audio_prompt(self._voice_name)

    @property
    def sample_rate(self) -> int:
        """Return PocketTTS native sample rate."""
        if self._model is not None:
            return self._model.sample_rate
        return 24000  # PocketTTS default until model is loaded

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using PocketTTS streaming API.

        Args:
            text: Text to synthesize.

        Yields:
            TTSStartedFrame, TTSAudioRawFrame chunks, TTSStoppedFrame.
        """
        if self._model is None or self._voice_state is None:
            yield ErrorFrame("PocketTTS model not loaded yet")
            return

        if not text or not text.strip():
            return

        logger.info(f"🔊 TTS: {text[:80]}{'...' if len(text) > 80 else ''}")

        yield TTSStartedFrame(context_id=context_id)

        try:
            # Try streaming first, fall back to non-streaming
            chunks_yielded = 0
            try:
                # Use streaming API for chunked output
                async for chunk_bytes in self._generate_stream(text):
                    if chunk_bytes:
                        yield TTSAudioRawFrame(
                            audio=chunk_bytes,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                        chunks_yielded += 1
            except AttributeError:
                # Fallback: generate_audio_stream may not exist in all versions
                logger.debug("Falling back to non-streaming TTS generation")
                audio_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._generate_audio, text
                )
                if audio_bytes:
                    # Split into chunks for smoother playback (~100ms chunks)
                    chunk_size = self.sample_rate * 2 // 10  # 100ms of 16-bit audio
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                        chunks_yielded += 1

            logger.debug(f"TTS generated {chunks_yielded} audio chunks")

        except Exception as e:
            logger.error(f"TTS error: {e}")
            yield ErrorFrame(f"TTS error: {str(e)}")

        yield TTSStoppedFrame(context_id=context_id)

    async def _generate_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Wrap PocketTTS streaming in an async generator."""
        loop = asyncio.get_event_loop()

        # Get the stream iterator in executor
        def get_stream():
            return self._model.generate_audio_stream(self._voice_state, text)

        stream = await loop.run_in_executor(None, get_stream)

        # Iterate over chunks
        def next_chunk(iterator):
            try:
                return next(iterator)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, next_chunk, stream)
            if chunk is None:
                break
            # Convert torch tensor to PCM bytes (16-bit signed int)
            audio_np = chunk.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            yield audio_int16.tobytes()

    def _generate_audio(self, text: str) -> bytes:
        """Non-streaming fallback: generate full audio at once (blocking)."""
        audio = self._model.generate_audio(self._voice_state, text)
        audio_np = audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logger.info("PocketTTS service stopped.")
