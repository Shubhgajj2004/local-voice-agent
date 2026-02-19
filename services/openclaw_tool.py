"""
OpenClaw tool bridge for Pipecat function calling.

Calls the local OpenClaw gateway via CLI and returns the assistant's text.
Designed to be non-blocking: it immediately speaks a short "working" message
while the tool runs in the background, then returns results to the LLM.
"""

import asyncio
import json
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams


async def openclaw_run(params: FunctionCallParams, task: str):
    """Run a task via OpenClaw.

    Args:
        task: The user request to execute with OpenClaw (emails, files, web, etc.).
    """
    if not task or not task.strip():
        await params.result_callback({"error": "Empty task"})
        return

    # Let the user know we're working (non-blocking UX).
    try:
        await params.llm.push_frame(TTSSpeakFrame("Working on that."))
    except Exception:
        pass

    # Run OpenClaw via CLI (async, non-blocking).
    cmd = [
        "openclaw",
        "agent",
        "--agent",
        "main",
        "--message",
        task,
        "--json",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="ignore").strip()
            logger.error(f"OpenClaw error: {err}")
            await params.result_callback({"error": err or "OpenClaw failed"})
            return

        raw = (stdout or b"").decode("utf-8", errors="ignore").strip()

        # Try JSON first; fallback to raw text.
        try:
            data = json.loads(raw)
        except Exception:
            data = {"text": raw}

        # Normalize to a simple text payload for the LLM.
        result_text = None
        if isinstance(data, dict):
            # Prefer the OpenClaw CLI JSON structure
            try:
                payloads = (
                    data.get("result", {})
                    .get("payloads", [])
                )
                if payloads and isinstance(payloads, list):
                    first = payloads[0] or {}
                    if isinstance(first, dict):
                        result_text = first.get("text")
            except Exception:
                pass

            # Fallbacks for other shapes
            result_text = result_text or (
                data.get("reply")
                or data.get("response")
                or data.get("text")
                or data.get("message")
            )

        await params.result_callback({"text": result_text or raw})

    except Exception as e:
        logger.exception("OpenClaw tool failed")
        await params.result_callback({"error": str(e)})
