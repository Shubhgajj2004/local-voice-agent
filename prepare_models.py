"""Pre-download and warm local models for a smoother first-run.

Usage:
  uv run python prepare_models.py
  uv run python prepare_models.py --voice alba

This caches:
- Qwen/Qwen3-ASR-0.6B (STT)
- PocketTTS model + selected voice
"""

import argparse
import asyncio
from loguru import logger


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="alba", help="PocketTTS voice name")
    args = parser.parse_args()

    # Qwen3-ASR download
    logger.info("Downloading Qwen3-ASR model...")
    try:
        from qwen_asr import Qwen3ASRModel
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if "cuda" in device else torch.float32
        _ = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=dtype,
            device_map=device,
            max_new_tokens=256,
        )
        logger.info("Qwen3-ASR cached.")
    except Exception as e:
        logger.error(f"Failed to cache Qwen3-ASR: {e}")

    # PocketTTS download
    logger.info("Downloading PocketTTS model + voice...")
    try:
        from pocket_tts import TTSModel

        model = TTSModel.load_model()
        _ = model.get_state_for_audio_prompt(args.voice)
        logger.info(f"PocketTTS cached for voice: {args.voice}")
    except Exception as e:
        logger.error(f"Failed to cache PocketTTS: {e}")


if __name__ == "__main__":
    asyncio.run(main())
