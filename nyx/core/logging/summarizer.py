import os
import json
import asyncio
import datetime
import openai

DEFAULT_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-nano")

from nyx.core.memory.memory_manager import MemoryManager

LOG_DIR = "/mnt/nyx_logs"
SYSTEM_PROMPT = "Summarise for long-term memory (<=1 k tokens)."
TOKEN_CHARS = 4  # rough chars per token for chunking
CHUNK_TOKEN_LIMIT = 3000

async def _summarise(text: str) -> str:
    """
    Stream-summarise *text* with the Responses API.
    """
    client = get_openai_client()

    stream = await client.responses.create(
        model=DEFAULT_MODEL,
        instructions=SYSTEM_PROMPT,   # ← old “system” role
        input=text,                   # ← old “user” role
        temperature=0.3,
        max_tokens=250,
        stream=True,
    )

    summary = ""
    async for chunk in stream:                    # chunk: ResponseChunk
        if chunk.output_text:
            summary += chunk.output_text

    return summary.strip()



async def nightly_rollup(date: str) -> None:
    """Summarise logs for a given date and store them via MemoryManager."""
    path = os.path.join(LOG_DIR, f"{date}.jsonl")
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    buffer = ""
    char_limit = CHUNK_TOKEN_LIMIT * TOKEN_CHARS
    for line in lines:
        if len(buffer) + len(line) + 1 > char_limit:
            summary = await _summarise(buffer)
            await MemoryManager.add(summary, {"type": "daily_summary", "date": date})
            await asyncio.sleep(0)  # yield control
            buffer = ""
        buffer += line + "\n"

    if buffer:
        summary = await _summarise(buffer)
        await MemoryManager.add(summary, {"type": "daily_summary", "date": date})
        await asyncio.sleep(0)
