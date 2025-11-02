"""Conflict synthesizer background tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from infra.cache import cache_key
from nyx.tasks.base import NyxTask, app

from logic.conflict_system.conflict_synthesizer import LLM_ROUTE_TIMEOUT
from nyx.conflict.hotpath.route import (
    dispatch_scene_route_cache_update,
    get_scene_route_hash,
    get_scene_route_key_suffix,
    get_scene_route_versions,
)
from nyx.conflict.workers import llm_route_scene_subsystems
from monitoring.metrics import metrics
from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.nyx_agent.context import (
    build_canonical_snapshot_payload,
    fetch_canonical_snapshot,
    persist_canonical_snapshot,
)
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()
_ROUTE_CACHE_TTL = int(os.getenv("CONFLICT_ROUTE_CACHE_TTL", "300"))


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"conflict:{payload.get('conversation_id')}:{payload.get('turn_id')}"


def _routing_key(payload: Dict[str, Any]) -> str:
    conversation_id = payload.get("conversation_id")
    user_id = payload.get("user_id")
    scene_hash = payload.get("scene_hash") or ""
    scene_context = payload.get("scene_context")
    if not isinstance(scene_context, dict):
        scene_context = {}

    ids = (
        _coerce_ids(str(user_id), str(conversation_id))
        if user_id is not None and conversation_id is not None
        else None
    )
    versions: Dict[str, int] = {}
    conversation_part = str(conversation_id)
    if ids:
        versions = get_scene_route_versions(
            ids[0],
            ids[1],
            scene_context=scene_context,
            versions=payload.get("versions"),
        )
        conversation_part = str(ids[1])

    suffix = get_scene_route_key_suffix(versions)
    return cache_key("conflict-route", conversation_part, scene_hash, *suffix)


def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _coerce_ids(user_id: str, conversation_id: str) -> Optional[Tuple[int, int]]:
    try:
        return int(user_id), int(conversation_id)
    except (TypeError, ValueError):
        return None


def _hydrate_snapshot(user_id: str, conversation_id: str) -> Dict[str, Any]:
    snapshot = _SNAPSHOTS.get(user_id, conversation_id)
    if snapshot:
        return snapshot
    ids = _coerce_ids(user_id, conversation_id)
    if not ids:
        return snapshot
    canonical = _run_coro(fetch_canonical_snapshot(*ids))
    if canonical:
        hydrated = dict(canonical)
        _SNAPSHOTS.put(user_id, conversation_id, hydrated)
        return hydrated
    return snapshot


def _persist_snapshot(user_id: str, conversation_id: str, snapshot: Dict[str, Any]) -> None:
    ids = _coerce_ids(user_id, conversation_id)
    if not ids:
        return
    payload = build_canonical_snapshot_payload(snapshot)
    if not payload:
        return
    _run_coro(persist_canonical_snapshot(ids[0], ids[1], payload))


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.conflict_tasks.process_events",
    acks_late=True,
)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def process_events(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Process deferred conflict computations."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")

    snapshot = _hydrate_snapshot(user_id, conversation_id)
    history = snapshot.setdefault("conflict_history", [])
    history.append({"turn_id": turn_id, "payload": payload.get("payload", {})})
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)
    _persist_snapshot(user_id, conversation_id, snapshot)

    logger.debug("Processed conflict events for turn=%s conversation=%s", turn_id, conversation_id)
    return {"status": "queued", "turn_id": turn_id}


def _build_scene_router_prompt(scene_context: Dict[str, Any]) -> str:
    return (
        "Analyze this scene context and determine which conflict subsystems should be active:\n"
        f"{json.dumps(scene_context, indent=2, sort_keys=True)}\n\n"
        "Available subsystems must be returned as a JSON list of subsystem names."
    )


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.conflict_tasks.route_subsystems",
    acks_late=True,
)
@idempotent(key_fn=_routing_key)
def route_subsystems(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Run the orchestrator routing prompt and cache subsystem decisions."""

    if not payload:
        return None

    scene_context = payload.get("scene_context") or {}
    if not isinstance(scene_context, dict):
        scene_context = {}

    scene_hash = payload.get("scene_hash") or get_scene_route_hash(scene_context)

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    ids = _coerce_ids(user_id, conversation_id)
    if not ids:
        logger.warning("Unable to coerce ids for conflict routing: %s/%s", user_id, conversation_id)
        return {"status": "skipped", "reason": "invalid_ids"}

    try:
        from logic.conflict_system.conflict_synthesizer import get_synthesizer

        synthesizer = _run_coro(get_synthesizer(*ids))
    except Exception:
        logger.exception("Failed to load conflict synthesizer for routing")
        return {"status": "error", "reason": "synth_load_failed"}

    orchestrator = getattr(synthesizer, "_orchestrator", None)
    if orchestrator is None:
        logger.debug("No orchestrator available for routing; skipping cache warm")
        return {"status": "skipped", "reason": "no_orchestrator"}

    prompt = _build_scene_router_prompt(scene_context)

    try:
        subsystem_names = _run_coro(
            llm_route_scene_subsystems(
                synthesizer,
                prompt,
                timeout=LLM_ROUTE_TIMEOUT,
            )
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Conflict subsystem routing timed out after %ss for scene %s",
            LLM_ROUTE_TIMEOUT,
            scene_hash,
        )
        metrics().CONFLICT_ROUTER_TIMEOUTS.inc()
        synthesizer._performance_metrics["timeouts_count"] += 1
        return {"status": "timeout", "scene_hash": scene_hash}
    except ValueError:
        logger.exception("Conflict subsystem routing returned invalid payload")
        synthesizer._performance_metrics["failures_count"] += 1
        return {"status": "error", "scene_hash": scene_hash, "reason": "invalid_response"}
    except Exception:
        logger.exception("Conflict subsystem routing failed")
        synthesizer._performance_metrics["failures_count"] += 1
        return {"status": "error", "scene_hash": scene_hash}

    valid_names = list(subsystem_names)

    cache_ttl = int(payload.get("ttl") or getattr(synthesizer, "_cache_ttl", _ROUTE_CACHE_TTL))
    versions = get_scene_route_versions(
        ids[0],
        ids[1],
        scene_context=scene_context,
        versions=payload.get("versions"),
    )
    dispatch_scene_route_cache_update(
        user_id=ids[0],
        conversation_id=ids[1],
        scene_hash=scene_hash,
        subsystem_names=valid_names,
        cache_ttl=cache_ttl,
        scene_context=scene_context,
        versions=versions,
    )

    return {
        "status": "cached",
        "scene_hash": scene_hash,
        "subsystems": valid_names,
        "versions": versions,
    }


__all__ = ["process_events", "route_subsystems"]
