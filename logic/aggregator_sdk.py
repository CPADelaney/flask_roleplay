"""Slim aggregator SDK built on the unified cache service."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from context import unified_cache as _unified_cache
from context.context_performance import track_performance
from context.context_service import (
    get_comprehensive_context as _service_get_comprehensive_context,
    get_context_service as _get_context_service,
)
from context.projection_helpers import parse_scene_projection_row
from db.read import read_scene_context
from openai_integration.conversations import (
    get_active_scene as get_openai_active_scene,
    get_latest_chatkit_thread,
    get_latest_conversation as get_latest_openai_conversation,
)
from story_templates.preset_manager import PresetStoryManager
if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from routes.story_routes import build_aggregator_text as _story_build_aggregator_text

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

        openai_payload: Dict[str, Any] = {}

        try:
            latest_conversation = await get_latest_openai_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to fetch latest OpenAI conversation")
        else:
            if latest_conversation:
                openai_payload["conversation"] = latest_conversation
                metadata = latest_conversation.get("metadata") or {}
                aggregator_metadata: Dict[str, Any] = {}
                queued_scene = metadata.get("queued_scene")
                if isinstance(queued_scene, dict):
                    scene_title = queued_scene.get("scene_title")
                    if scene_title:
                        aggregator_metadata["queuedScene"] = scene_title
                queued_closing = metadata.get("queued_scene_closing")
                if isinstance(queued_closing, dict):
                    closing_summary = queued_closing.get("scene_summary")
                    if closing_summary:
                        aggregator_metadata["queuedSceneClosing"] = closing_summary
                if aggregator_metadata:
                    payload["aggregatorMetadata"] = aggregator_metadata

        try:
            latest_thread = await get_latest_chatkit_thread(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to fetch latest ChatKit thread")
        else:
            if latest_thread:
                openai_payload["chatkit_thread"] = latest_thread

        try:
            active_scene = await get_openai_active_scene(
                conversation_id=conversation_id,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to fetch OpenAI active scene")
        else:
            if active_scene:
                openai_payload["active_scene"] = active_scene

        if openai_payload:
            payload["openai_integration"] = openai_payload

        return payload

    return await context_cache.get(cache_key, _fetch)


async def get_comprehensive_context(
    user_id: int,
    conversation_id: int,
    input_text: str = "",
    location: Optional[str] = None,
    context_budget: int = 4000,
    use_vector_search: Optional[bool] = None,
    use_delta: bool = True,
    source_version: Optional[int] = None,
    summary_level: Optional[int] = None,
) -> Dict[str, Any]:
    """Delegate to the context service comprehensive context builder."""

    return await _service_get_comprehensive_context(
        user_id=user_id,
        conversation_id=conversation_id,
        input_text=input_text,
        location=location,
        context_budget=context_budget,
        use_vector_search=use_vector_search,
        use_delta=use_delta,
        source_version=source_version,
        summary_level=summary_level,
    )


async def fallback_get_context(
    user_id: int,
    conversation_id: int,
    *,
    input_text: str = "",
    location: Optional[str] = None,
    context_budget: int = 4000,
    use_vector_search: Optional[bool] = None,
    use_delta: bool = True,
    include_memories: bool = True,
    include_npcs: bool = True,
    include_location: bool = True,
    include_quests: bool = True,
    source_version: Optional[int] = None,
    summary_level: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the basic context payload via the shared context service."""

    service = await _get_context_service(user_id, conversation_id)
    return await service.get_context(
        input_text=input_text,
        location=location,
        context_budget=context_budget,
        use_vector_search=use_vector_search,
        use_delta=use_delta,
        include_memories=include_memories,
        include_npcs=include_npcs,
        include_location=include_location,
        include_quests=include_quests,
        source_version=source_version,
        summary_level=summary_level,
    )


def build_aggregator_text(*args: Any, **kwargs: Any) -> str:
    """Lightweight wrapper forwarding to :mod:`routes.story_routes`."""

    from routes.story_routes import build_aggregator_text as _story_build_aggregator_text

    return _story_build_aggregator_text(*args, **kwargs)


__all__ = [
    "context_cache",
    "init_singletons",
    "is_context_ready",
    "wait_for_context_ready",
    "get_aggregated_roleplay_context",
    "get_comprehensive_context",
    "fallback_get_context",
    "build_aggregator_text",
    "PresetStoryManager",
    "get_latest_openai_conversation",
    "get_latest_chatkit_thread",
    "get_openai_active_scene",
]
