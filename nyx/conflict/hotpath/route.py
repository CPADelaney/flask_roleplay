"""Cache-first dispatch helpers for conflict subsystem routing."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

from infra.cache import cache_key, get_json, redis_lock, set_json
from monitoring.metrics import metrics
from nyx.common.outbox import DuplicateEventError, append_event, get_session_factory
from nyx.conversation.version_registry import version_registry
from nyx.telemetry.metrics import CACHE_HIT, CACHE_MISS
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

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
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    return repr(value)


def _compute_scene_hash(scene_context: Optional[Dict[str, Any]]) -> str:
    normalized = json.dumps(
        _stringify(scene_context or {}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _normalize_versions(versions: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    if not versions:
        return normalized
    for key in ("world", "conflict"):
        value = versions.get(key)  # type: ignore[arg-type]
        try:
            if value is not None:
                normalized[key] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _version_suffix_parts(versions: Optional[Mapping[str, Any]]) -> Sequence[str]:
    normalized = _normalize_versions(versions)
    suffix: list[str] = []
    world_version = normalized.get("world")
    if world_version is not None:
        suffix.append(f"w{world_version}")
    conflict_version = normalized.get("conflict")
    if conflict_version is not None:
        suffix.append(f"c{conflict_version}")
    return tuple(suffix)


def _scene_route_cache_key(
    user_id: int,
    conversation_id: int,
    scene_hash: str,
    *,
    versions: Optional[Mapping[str, Any]] = None,
) -> str:
    return cache_key(
        "conflict",
        user_id,
        conversation_id,
        "scene_route",
        scene_hash,
        *_version_suffix_parts(versions),
    )


def _scene_route_lock_key(
    user_id: int,
    conversation_id: int,
    scene_hash: str,
    *,
    versions: Optional[Mapping[str, Any]] = None,
) -> str:
    return cache_key(
        "conflict",
        user_id,
        conversation_id,
        "scene_route_lock",
        scene_hash,
        *_version_suffix_parts(versions),
    )


def _extract_conflict_id(scene_context: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(scene_context, dict):
        return None

    for key in ("conflict_id", "active_conflict_id"):
        value = scene_context.get(key)
        if isinstance(value, (str, int)):
            return str(value)

    conflict_block = scene_context.get("conflict") or scene_context.get("conflict_state")
    if isinstance(conflict_block, dict):
        for key in ("id", "conflict_id"):
            value = conflict_block.get(key)
            if isinstance(value, (str, int)):
                return str(value)

    conflicts = scene_context.get("conflicts")
    if isinstance(conflicts, dict):
        for key in ("active", "current", "id"):
            candidate = conflicts.get(key)
            if isinstance(candidate, (str, int)):
                return str(candidate)
            if isinstance(candidate, dict):
                nested = candidate.get("id") or candidate.get("conflict_id")
                if isinstance(nested, (str, int)):
                    return str(nested)
    elif isinstance(conflicts, list):
        for entry in conflicts:
            if not isinstance(entry, dict):
                continue
            nested = entry.get("id") or entry.get("conflict_id")
            if isinstance(nested, (str, int)):
                return str(nested)

    return None


def get_scene_route_hash_from_cache(scene_context: Optional[Dict[str, Any]]) -> str:
    """Return a stable hash representing the provided scene context."""

    return _compute_scene_hash(scene_context)


def get_scene_route_versions_from_cache(
    user_id: int,
    conversation_id: int,
    *,
    scene_context: Optional[Dict[str, Any]] = None,
    versions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    if versions:
        normalized = _normalize_versions(versions)
        if normalized:
            return normalized

    conflict_id = _extract_conflict_id(scene_context)
    try:
        counters = version_registry.get_counters(
            user_id,
            conversation_id,
            conflict_id=conflict_id,
        )
    except Exception:  # pragma: no cover - defensive guard
        logger.exception(
            "Failed to load version counters for conflict routing cache",
        )
        counters = {}

    normalized = _normalize_versions(counters)
    if "world" not in normalized:
        normalized["world"] = 0
    return normalized


def get_scene_route_suffix_from_cache(versions: Optional[Mapping[str, Any]]) -> Sequence[str]:
    return _version_suffix_parts(versions)


def get_scene_route_from_cache(
    *,
    user_id: int,
    conversation_id: int,
    scene_hash: str,
    versions: Mapping[str, Any],
) -> Optional[Set["SubsystemType"]]:
    cache_key_value = _scene_route_cache_key(
        user_id,
        conversation_id,
        scene_hash,
        versions=versions,
    )
    cached = get_json(cache_key_value) or {}
    names = cached.get("subsystems") if isinstance(cached, dict) else None

    if isinstance(names, list):
        metrics().CACHE_HIT_COUNT.labels(cache_type=_CACHE_TYPE).inc()
        CACHE_HIT.labels(section="conflict_scene_route").inc()
        metrics().CONFLICT_ROUTER_DECISIONS.labels(source="background").inc()
        return _deserialize_subsystems(names)

    metrics().CACHE_MISS_COUNT.labels(cache_type=_CACHE_TYPE).inc()
    CACHE_MISS.labels(section="conflict_scene_route").inc()
    return None


def enqueue_scene_route_refresh(
    *,
    user_id: int,
    conversation_id: int,
    scene_context: Dict[str, Any],
    scene_hash: str,
    cache_ttl: int,
    versions: Mapping[str, Any],
) -> None:
    lock_name = _scene_route_lock_key(
        user_id,
        conversation_id,
        scene_hash,
        versions=versions,
    )
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
                "versions": versions,
            }
            try:
                with session.begin():
                    append_event(
                        session,
                        topic="ConflictRouteRequested",
                        payload=payload,
                        dedupe_key=_scene_route_cache_key(
                            user_id,
                            conversation_id,
                            scene_hash,
                            versions=versions,
                        ),
                    )
            except DuplicateEventError:
                logger.debug("Route event already enqueued for scene hash %s", scene_hash)
            except Exception:  # pragma: no cover - database availability issues
                logger.exception("Failed to persist conflict route outbox event")
            finally:
                session.close()
    except RuntimeError:
        logger.debug("Route task already queued for scene hash %s", scene_hash)
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Failed to enqueue subsystem routing task: %s", exc)


def dispatch_store_scene_route(
    *,
    user_id: int,
    conversation_id: int,
    scene_hash: str,
    subsystem_names: Sequence[str],
    cache_ttl: int = _DEFAULT_CACHE_TTL,
    scene_context: Optional[Dict[str, Any]] = None,
    versions: Optional[Mapping[str, Any]] = None,
) -> None:
    resolved_versions = get_scene_route_versions_from_cache(
        user_id,
        conversation_id,
        scene_context=scene_context,
        versions=versions,
    )
    payload = {
        "subsystems": sorted(set(subsystem_names)),
        "routed_at": datetime.utcnow().isoformat(),
        "versions": resolved_versions,
    }
    set_json(
        _scene_route_cache_key(
            user_id,
            conversation_id,
            scene_hash,
            versions=resolved_versions,
        ),
        payload,
        ex=cache_ttl,
    )


def dispatch_scene_route(
    scene_context: Dict[str, Any],
    *,
    user_id: int,
    conversation_id: int,
    fallback_subsystems: Iterable["SubsystemType"],
    orchestrator_available: bool,
    cache_ttl: int = _DEFAULT_CACHE_TTL,
    scene_hash: Optional[str] = None,
) -> Set["SubsystemType"]:
    from logic.conflict_system.conflict_synthesizer import SubsystemType

    scene_hash = scene_hash or get_scene_route_hash_from_cache(scene_context)
    resolved_versions = get_scene_route_versions_from_cache(
        user_id,
        conversation_id,
        scene_context=scene_context,
    )

    cached = get_scene_route_from_cache(
        user_id=user_id,
        conversation_id=conversation_id,
        scene_hash=scene_hash,
        versions=resolved_versions,
    )
    if cached is not None:
        return cached

    if orchestrator_available:
        enqueue_scene_route_refresh(
            user_id=user_id,
            conversation_id=conversation_id,
            scene_context=scene_context,
            scene_hash=scene_hash,
            cache_ttl=cache_ttl,
            versions=resolved_versions,
        )

    fallback = {
        SubsystemType(subsystem) if isinstance(subsystem, str) else subsystem
        for subsystem in fallback_subsystems
    }
    metrics().CONFLICT_ROUTER_DECISIONS.labels(source="heuristic").inc()
    return fallback


def _deserialize_subsystems(names: Sequence[str]) -> Set["SubsystemType"]:
    from logic.conflict_system.conflict_synthesizer import SubsystemType

    resolved: Set[SubsystemType] = set()
    for name in names:
        try:
            resolved.add(SubsystemType(name))
        except ValueError:
            logger.debug("Ignoring unknown subsystem returned by router: %s", name)
    return resolved


__all__ = [
    "dispatch_scene_route",
    "dispatch_store_scene_route",
    "enqueue_scene_route_refresh",
    "get_scene_route_from_cache",
    "get_scene_route_hash_from_cache",
    "get_scene_route_suffix_from_cache",
    "get_scene_route_versions_from_cache",
]
