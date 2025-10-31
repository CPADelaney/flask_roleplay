"""Hot-path helpers for conflict subsystem routing."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence, Set

from infra.cache import cache_key, get_json, redis_lock, set_json
from monitoring.metrics import metrics
from nyx.common.outbox import DuplicateEventError, append_event, get_session_factory
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Cache bookkeeping
_CACHE_TYPE = "conflict_subsystem_routing"
_DEFAULT_CACHE_TTL = 300

_OUTBOX_SESSION_FACTORY: Optional[sessionmaker] = None


def _get_outbox_session_factory() -> Optional[sessionmaker]:
    global _OUTBOX_SESSION_FACTORY
    if _OUTBOX_SESSION_FACTORY is not None:
        return _OUTBOX_SESSION_FACTORY
    try:
        _OUTBOX_SESSION_FACTORY = get_session_factory()
    except Exception:  # pragma: no cover - configuration issues
        logger.exception("Failed to initialise outbox session factory for conflict routing")
        return None
    return _OUTBOX_SESSION_FACTORY


def _stringify(value: Any) -> Any:
    """Best-effort serializer for hashing scene context."""

    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    return repr(value)


def compute_scene_hash(scene_context: Optional[Dict[str, Any]]) -> str:
    """Compute a stable hash for a scene context payload."""

    normalized = json.dumps(
        _stringify(scene_context or {}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _scene_route_cache_key(user_id: int, conversation_id: int, scene_hash: str) -> str:
    return cache_key("conflict", user_id, conversation_id, "scene_route", scene_hash)


def _deserialize_subsystems(names: Sequence[str]) -> Set["SubsystemType"]:
    from logic.conflict_system.conflict_synthesizer import SubsystemType

    resolved: Set[SubsystemType] = set()
    for name in names:
        try:
            resolved.add(SubsystemType(name))
        except ValueError:
            logger.debug("Ignoring unknown subsystem returned by router: %s", name)
    return resolved


def store_scene_route(
    *,
    user_id: int,
    conversation_id: int,
    scene_hash: str,
    subsystem_names: Sequence[str],
    cache_ttl: int = _DEFAULT_CACHE_TTL,
) -> None:
    """Persist background routing results for hot-path reuse."""

    payload = {
        "subsystems": sorted(set(subsystem_names)),
        "routed_at": datetime.utcnow().isoformat(),
    }
    set_json(_scene_route_cache_key(user_id, conversation_id, scene_hash), payload, ex=cache_ttl)


def _queue_background_route(
    *,
    user_id: int,
    conversation_id: int,
    scene_context: Dict[str, Any],
    scene_hash: str,
    cache_ttl: int,
) -> None:
    lock_name = cache_key("conflict", user_id, conversation_id, "scene_route_lock", scene_hash)
    try:
        with redis_lock(lock_name, ttl=15):
            session_factory = _get_outbox_session_factory()
            if session_factory is None:
                return
            session = session_factory()
            payload = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "scene_context": scene_context,
                "scene_hash": scene_hash,
                "ttl": cache_ttl,
            }
            try:
                with session.begin():
                    append_event(
                        session,
                        topic="ConflictRouteRequested",
                        payload=payload,
                        dedupe_key=scene_hash,
                    )
            except DuplicateEventError:
                logger.debug("Route event already enqueued for scene hash %s", scene_hash)
            except Exception:  # pragma: no cover - database availability issues
                logger.exception("Failed to persist conflict route outbox event")
            finally:
                session.close()
    except RuntimeError:
        # Another worker already queued this scene routing job.
        logger.debug("Route task already queued for scene hash %s", scene_hash)
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Failed to enqueue subsystem routing task: %s", exc)


def route_scene_subsystems(
    scene_context: Dict[str, Any],
    *,
    user_id: int,
    conversation_id: int,
    fallback_subsystems: Iterable["SubsystemType"],
    orchestrator_available: bool,
    cache_ttl: int = _DEFAULT_CACHE_TTL,
    scene_hash: Optional[str] = None,
) -> Set["SubsystemType"]:
    """Return subsystems for scene processing via cache-first routing."""

    from logic.conflict_system.conflict_synthesizer import SubsystemType

    scene_hash = scene_hash or compute_scene_hash(scene_context)
    cache_key_value = _scene_route_cache_key(user_id, conversation_id, scene_hash)
    cached = get_json(cache_key_value) or {}
    names = cached.get("subsystems") if isinstance(cached, dict) else None

    if isinstance(names, list):
        metrics().CACHE_HIT_COUNT.labels(cache_type=_CACHE_TYPE).inc()
        metrics().CONFLICT_ROUTER_DECISIONS.labels(source="background").inc()
        return _deserialize_subsystems(names)

    metrics().CACHE_MISS_COUNT.labels(cache_type=_CACHE_TYPE).inc()

    if orchestrator_available:
        _queue_background_route(
            user_id=user_id,
            conversation_id=conversation_id,
            scene_context=scene_context,
            scene_hash=scene_hash,
            cache_ttl=cache_ttl,
        )

    fallback = {
        SubsystemType(subsystem) if isinstance(subsystem, str) else subsystem
        for subsystem in fallback_subsystems
    }
    metrics().CONFLICT_ROUTER_DECISIONS.labels(source="heuristic").inc()
    return fallback


__all__ = [
    "compute_scene_hash",
    "route_scene_subsystems",
    "store_scene_route",
]
