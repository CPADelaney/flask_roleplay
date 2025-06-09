import os
import json
import datetime
import asyncio
from reflection.reflection_agent import log_buffer, schedule_reflection

LOG_DIR = "/mnt/nyx_logs"

async def log_event(event: dict) -> None:
    """Append an event dict to today's log file asynchronously."""
    os.makedirs(LOG_DIR, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(LOG_DIR, f"{date}.jsonl")
    line = json.dumps(event)
    await asyncio.to_thread(_append_line, path, line)
    log_buffer.append(event)
    await schedule_reflection()


def _append_line(path: str, line: str) -> None:
    # newline="" avoids Windows-style CRLF line endings when running on Windows
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(line + "\n")
