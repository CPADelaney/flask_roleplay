"""Slim aggregator SDK built on the unified cache service."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from context import unified_cache as _unified_cache
from context.context_performance import track_performance
from context.projection_helpers import parse_scene_projection_row
from db.read import read_scene_context

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ReadinessState:
    event: asyncio.Event


_READINESS: Optional[_ReadinessState] = None

context_cache = _unified_cache.context_cache


def _cache_key(user_id: int, conversation_id: int) -> str:
    return f"context:{user_id}:{conversation_id}"


async def init_singletons() -> None:
    """Initialise shared handles and signal readiness."""

    global _READINESS

    if _READINESS is None:
        _READINESS = _ReadinessState(event=asyncio.Event())

    if _READINESS.event.is_set():
        return

    try:
        await context_cache.start_background_cleanup()
    except Exception:  # pragma: no cover - defensive log
        logger.exception("Failed to start unified cache cleanup")

    _READINESS.event.set()
    logger.info("Aggregator SDK singletons marked ready.")


def is_context_ready() -> bool:
    """Return ``True`` once :func:`init_singletons` completes."""

    return bool(_READINESS and _READINESS.event.is_set())


async def wait_for_context_ready(timeout: Optional[float] = None) -> bool:
    """Await readiness with an optional timeout."""

    if _READINESS is None:
        return False

    try:
        await asyncio.wait_for(_READINESS.event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return False
    return True


@track_performance("get_aggregated_roleplay_context")
async def get_aggregated_roleplay_context(
    user_id: int,
    conversation_id: int,
    player_name: str,
) -> Dict[str, Any]:
    """Return cached roleplay context assembled from the scene projection."""

    if not is_context_ready():
        await init_singletons()

    if not await wait_for_context_ready():
        logger.warning("Aggregator context requested before readiness signal.")

    cache_key = _cache_key(user_id, conversation_id)

    async def _fetch() -> Dict[str, Any]:
        rows = await read_scene_context(user_id, conversation_id)
        if not rows:
            return {
                "currentRoleplay": {},
                "current_roleplay": {},
                "playerName": player_name,
                "playerStats": {},
                "npcsPresent": [],
                "activeEvents": [],
                "activeQuests": [],
            }

        projection = parse_scene_projection_row(rows[0])
        current_roleplay = projection.roleplay_dict()

        payload: Dict[str, Any] = {
            "currentRoleplay": current_roleplay,
            "current_roleplay": current_roleplay,
            "playerName": player_name,
            "playerStats": projection.player_stats_dict(),
            "npcsPresent": projection.npc_rows(),
            "activeEvents": projection.active_events(),
            "activeQuests": projection.active_quests(),
        }

        location = projection.current_location()
        if location:
            payload["currentLocation"] = location
            payload["current_location"] = location
            payload["location"] = location

        time_of_day = projection.time_of_day()
        if time_of_day:
            payload["timeOfDay"] = time_of_day

        return payload

    return await context_cache.get(cache_key, _fetch)
