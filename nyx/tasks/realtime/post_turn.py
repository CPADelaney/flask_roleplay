"""Post-turn fan-out dispatcher."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from celery import shared_task, group

logger = logging.getLogger(__name__)


@shared_task(name="nyx.tasks.realtime.post_turn.dispatch", acks_late=True, autoretry_for=(), retry_backoff=False)
def dispatch(payload: Dict[str, Any] | None = None) -> str:
    """Fan out per-turn side effects to the appropriate queues."""

    payload = payload or {}
    side_effects: Dict[str, Dict[str, Any]] = payload.get("side_effects") or {}
    jobs: List[Any] = []

    try:
        from nyx.tasks.background import world_tasks, conflict_tasks, npc_tasks, lore_tasks
        from nyx.tasks.heavy import memory_tasks
    except Exception:  # pragma: no cover
        logger.exception("TurnPostProcessor could not import task modules")
        return "import-error"

    if side_effects.get("world"):
        jobs.append(world_tasks.apply_universal.s(side_effects["world"]))
    if side_effects.get("memory"):
        jobs.append(memory_tasks.add_and_embed.s(side_effects["memory"]))
    if side_effects.get("conflict"):
        jobs.append(conflict_tasks.process_events.s(side_effects["conflict"]))
    if side_effects.get("npc"):
        jobs.append(npc_tasks.run_adaptation_cycle.s(side_effects["npc"]))
    if side_effects.get("lore"):
        jobs.append(lore_tasks.precompute_scene_bundle.s(side_effects["lore"]))

    if not jobs:
        logger.debug("TurnPostProcessor no-op (turn_id=%s)", payload.get("turn_id"))
        return "no-op"

    result = group(jobs).apply_async()
    logger.debug("TurnPostProcessor dispatched group %s", result.id)
    return str(result.id)


__all__ = ["dispatch"]
