"""Simple file-backed conversation memory for Pipecat.

Stores the last N messages (user/assistant) to disk and reloads on startup.
"""

import json
import os
from typing import List, Dict, Any

from loguru import logger
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


def load_history(path: str, max_messages: int) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[-max_messages:]
    except Exception as e:
        logger.warning(f"Failed to load history: {e}")
    return []


def save_history(path: str, messages: List[Dict[str, Any]], max_messages: int):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages[-max_messages:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save history: {e}")


class ContextStoreProcessor(FrameProcessor):
    def __init__(self, context, path: str, max_messages: int = 20):
        super().__init__()
        self._context = context
        self._path = path
        self._max_messages = max_messages

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Persist after each assistant response completes
        if isinstance(frame, LLMFullResponseEndFrame):
            # Skip system messages; persist only user/assistant turns
            msgs = [
                m for m in self._context.messages
                if m.get("role") in ("user", "assistant")
            ]
            save_history(self._path, msgs, self._max_messages)

        await self.push_frame(frame, direction)
