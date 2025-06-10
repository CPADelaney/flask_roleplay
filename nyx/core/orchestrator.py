"""Lightweight orchestrator glue layer for Nyx agents."""

import asyncio
import datetime
import logging

from nyx.core.memory.memory_manager import MemoryManager
from strategy.manager import StrategyManager
from nyx.core.logging import event_logger as EventLogger
from nyx.core.logging.summarizer import nightly_rollup
from nyx.core.reward import evaluator as RewardEvaluator

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
    if hits:
        bullet_lines = "\n".join(
            "- " + (h["text"][:300] + ("…" if len(h["text"]) > 300 else ""))
            for h in hits
        )
        knowledge = f"KNOWLEDGE:\n{bullet_lines}\n"
        comments = "".join(
            f"<!--MEM:{h.get('meta', {}).get('uid')},{h.get('score',0):.2f}-->"
            for h in hits
        )
        ctx = f"{knowledge}{comments}\n{ctx}"

    strategy_line = ""
    try:
        params = StrategyManager().current()
    except Exception:
        pass
    else:
        strategy_line = f"STRATEGY: {params.json()}\n"

    return f"{strategy_line}{ctx}"


async def log_and_score(event_type: str, payload: dict | None = None) -> float:
    """Log an event and return its reward score."""
    event = {"type": event_type}
    if payload:
        event["payload"] = payload
    await EventLogger.log_event(event)
    return RewardEvaluator.evaluate(event_type)

logger = logging.getLogger(__name__)
_background_task = None
_reflection_task = None
_practice_task = None


def start_background() -> None:
    """Start orchestrator background jobs."""
    global _background_task, _reflection_task, _practice_task
    if _background_task is None:
        _background_task = asyncio.create_task(_rollup_loop())
    if _reflection_task is None:
        _reflection_task = asyncio.create_task(_reflection_loop())
    if _practice_task is None:
        try:
            from nyx.core.practice.coding_practice import autoloop
        except ImportError:  # pragma: no cover - optional dependency
            pass
        else:
            _practice_task = asyncio.create_task(autoloop())


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
            from reflection import reflection_agent
            await reflection_agent.schedule_reflection()
        except Exception:
            logger.exception("Reflection scheduling failed")
