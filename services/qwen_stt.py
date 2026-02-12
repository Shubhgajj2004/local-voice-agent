"""
Custom Pipecat STT Service wrapping Qwen3-ASR-0.6B.

Uses the qwen-asr package with transformers backend for local speech-to-text.
The model runs on GPU (CUDA) for fast inference, or falls back to CPU.
"""

import asyncio
import io
import struct
from collections.abc import AsyncGenerator

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService


class QwenSTTService(STTService):
    """Speech-to-text service using Qwen3-ASR-0.6B running locally.

    The model is loaded once at initialization and reused for each utterance.
    Transcription happens per-utterance (after VAD/Smart Turn detects end-of-speech),
    which is the correct pattern for voice agent pipelines.

    Args:
        model_name: HuggingFace model ID. Default: "Qwen/Qwen3-ASR-0.6B"
        device: Device to run on. "cuda:0" for GPU, "cpu" for CPU.
        language: Force language (e.g. "English"), or None for auto-detect.
        dtype: Torch dtype. float16 for GPU, float32 for CPU.
    """

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        language: str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._model_name = model_name
        self._device = device
        self._language = language
        self._model = None

        # Auto-select dtype based on device
        if dtype is not None:
            self._dtype = dtype
        elif "cuda" in device:
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

    async def start(self, frame: Frame):
        await super().start(frame)
        # Load model in a thread to avoid blocking the event loop
        logger.info(f"Loading Qwen3-ASR model: {self._model_name} on {self._device}...")
        self._model = await asyncio.get_event_loop().run_in_executor(
            None, self._load_model
        )
        logger.info("Qwen3-ASR model loaded successfully.")

    def _load_model(self):
        """Load the Qwen3-ASR model (blocking, runs in executor)."""
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            self._model_name,
            dtype=self._dtype,
            device_map=self._device,
            max_new_tokens=256,
        )
        return model

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a complete utterance.

        Args:
            audio: Raw PCM audio bytes (16-bit, mono, 16kHz — Pipecat default).

        Yields:
            TranscriptionFrame with the transcribed text.
        """
        if self._model is None:
            yield ErrorFrame("Qwen STT model not loaded yet")
            return

        if len(audio) < 3200:  # Less than 100ms of audio at 16kHz
            return

        try:
            # Convert raw PCM bytes to numpy float32 array
            # Pipecat sends 16-bit signed PCM, mono, at sample_rate (default 16kHz)
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = 16000  # Pipecat default

            # Run transcription in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._transcribe,
                audio_np,
                sample_rate,
            )

            if result and result.strip():
                logger.info(f"📝 STT: {result}")
                yield TranscriptionFrame(
                    text=result,
                    user_id="user",
                    timestamp="",
                )

        except Exception as e:
            logger.error(f"STT error: {e}")
            yield ErrorFrame(f"STT error: {str(e)}")

    def _transcribe(self, audio_np: np.ndarray, sample_rate: int) -> str:
        """Run model.transcribe() (blocking, runs in executor)."""
        results = self._model.transcribe(
            audio=(audio_np, sample_rate),
            language=self._language,
        )
        if results and len(results) > 0:
            return results[0].text
        return ""

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logger.info("Qwen STT service stopped.")
