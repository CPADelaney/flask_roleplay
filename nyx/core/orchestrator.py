from nyx.core.memory.memory_manager import MemoryManager
import asyncio
import datetime
import logging
from nyx.core.logging.summarizer import nightly_rollup
from reflection.reflection_agent import schedule_reflection

async def prepare_context(ctx: str, user_msg: str) -> str:
    """Prepend relevant memories to context.

    Parameters
    ----------
    ctx: str
        Existing context or system prompt.
    user_msg: str
        Latest user message used to fetch relevant memories.
    Returns
    -------
    str
        Augmented context including a KNOWLEDGE section and memory comments.
    """
    hits = await MemoryManager.fetch_relevant(user_msg, k=5)
    if not hits:
        return ctx

    bullet_lines = "\n".join(
        "- " + (h["text"][:300] + ("…" if len(h["text"]) > 300 else ""))
        for h in hits
    )
    knowledge = f"KNOWLEDGE:\n{bullet_lines}\n"
    comments = "".join(
        f"<!--MEM:{h.get('meta', {}).get('uid')},{h.get('score',0):.2f}-->"
        for h in hits
    )
    return f"{knowledge}{comments}\n{ctx}"

logger = logging.getLogger(__name__)
_background_task = None
_reflection_task = None


def start_background() -> None:
    """Start orchestrator background jobs."""
    global _background_task, _reflection_task
    if _background_task is None:
        _background_task = asyncio.create_task(_rollup_loop())
    if _reflection_task is None:
        _reflection_task = asyncio.create_task(_reflection_loop())


async def _rollup_loop() -> None:
    while True:
        now = datetime.datetime.now()
        run_at = now.replace(hour=3, minute=15, second=0, microsecond=0)
        if run_at <= now:
            run_at += datetime.timedelta(days=1)
        await asyncio.sleep((run_at - now).total_seconds())
        date = (run_at - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            await nightly_rollup(date)
        except Exception:
            logger.exception("Nightly rollup failed")


async def _reflection_loop() -> None:
    """Periodically check if reflection should run."""
    while True:
        await asyncio.sleep(60)
        try:
            await schedule_reflection()
        except Exception:
            logger.exception("Reflection scheduling failed")
