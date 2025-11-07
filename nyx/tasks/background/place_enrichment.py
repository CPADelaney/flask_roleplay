from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.location.fictional_resolver import resolve_fictional
from nyx.location.gemini_maps_adapter import resolve_location_with_gemini
from nyx.location.query import PlaceQuery
from nyx.location.types import Anchor, Place, ResolveResult
from nyx.tasks.base import NyxTask, app
from nyx.tasks.utils import run_coro

logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _deserialize_anchor(payload: Dict[str, Any]) -> Anchor:
    hints = payload.get("hints") if isinstance(payload.get("hints"), dict) else {}
    return Anchor(
        scope=payload.get("scope") or "real",
        focus=None,
        label=payload.get("label"),
        lat=_coerce_float(payload.get("lat")),
        lon=_coerce_float(payload.get("lon")),
        primary_city=payload.get("primary_city"),
        region=payload.get("region"),
        country=payload.get("country"),
        world_name=payload.get("world_name"),
        hints=hints,
    )


def _deserialize_query(payload: Dict[str, Any]) -> PlaceQuery:
    raw_text = payload.get("raw_text") or ""
    normalized = payload.get("normalized") or raw_text.lower()
    target = payload.get("target") or raw_text
    return PlaceQuery(
        raw_text=raw_text,
        normalized=normalized,
        is_travel=bool(payload.get("is_travel")),
        target=target,
        transport_hint=payload.get("transport_hint"),
    )


def _sanitize_meta(value: Any, *, depth: int = 0, max_depth: int = 5) -> Any:
    """Produce a JSON-serializable clone of meta payloads for Celery."""

    if depth >= max_depth:
        return None

    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 50:
                break
            sanitized[str(key)] = _sanitize_meta(item, depth=depth + 1, max_depth=max_depth)
        return sanitized

    if isinstance(value, (list, tuple, set)):
        sanitized_list = []
        for idx, item in enumerate(value):
            if idx >= 50:
                break
            sanitized_list.append(_sanitize_meta(item, depth=depth + 1, max_depth=max_depth))
        return sanitized_list

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.place_enrichment.enrich",
    queue="background",
    priority=7,
    acks_late=True,
)
def enrich(
    self, payload: Dict[str, Any], *, trace_id: str | None = None
) -> Optional[Dict[str, Any]]:
    """Run heavier Maps enrichment off the hot path."""

    _ = trace_id

    if not payload:
        return None

    query_payload = payload.get("query")
    anchor_payload = payload.get("anchor")

    if not isinstance(query_payload, dict) or not isinstance(anchor_payload, dict):
        logger.debug("[place_enrichment] Missing query or anchor payload")
        return None

    query = _deserialize_query(query_payload)
    anchor = _deserialize_anchor(anchor_payload)

    afc_cap = payload.get("afc_max_calls")
    afc_max_calls: Optional[int] = None
    if afc_cap is not None:
        try:
            afc_max_calls = max(1, int(afc_cap))
        except (TypeError, ValueError):
            logger.debug(
                "[place_enrichment] Invalid afc_max_calls payload=%r", afc_cap
            )

    logger.info(
        "[place_enrichment] Starting enrichment for '%s' (user=%s conversation=%s)",
        query.target or query.raw_text,
        payload.get("user_id"),
        payload.get("conversation_id"),
    )

    try:
        result = run_coro(
            resolve_location_with_gemini(
                query,
                anchor,
                afc_max_calls=afc_max_calls,
                include_events=True,
            )
        )
    except Exception:
        logger.exception(
            "[place_enrichment] Gemini enrichment failed for '%s'", 
            query.target or query.raw_text,
        )
        raise

    event_count = 0
    for op in result.operations or []:
        if op.get("op") == "events":
            items = op.get("items") or []
            event_count += len(items)

    logger.info(
        "[place_enrichment] Completed enrichment for '%s' candidates=%s events=%s",
        query.target or query.raw_text,
        len(result.candidates),
        event_count,
    )

    return {
        "status": result.status,
        "candidate_count": len(result.candidates),
        "event_count": event_count,
    }


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.place_enrichment.fictional_fallback",
    queue="background",
    priority=6,
    acks_late=True,
)
def fictional_fallback(
    self, payload: Dict[str, Any], *, trace_id: str | None = None
) -> Optional[Dict[str, Any]]:
    """Resolve fictional locations off-thread when real chains miss."""

    _ = trace_id

    if not payload:
        return None

    query_payload = payload.get("query")
    anchor_payload = payload.get("anchor")
    meta_payload = payload.get("meta")

    if not isinstance(query_payload, dict) or not isinstance(anchor_payload, dict):
        logger.debug("[place_enrichment] Missing query or anchor payload for fictional fallback")
        return None

    query = _deserialize_query(query_payload)
    anchor = _deserialize_anchor(anchor_payload)
    meta: Dict[str, Any] = dict(meta_payload) if isinstance(meta_payload, dict) else {}

    user_id = str(payload.get("user_id") or "")
    conversation_id = str(payload.get("conversation_id") or "")
    store = ConversationSnapshotStore()

    logger.info(
        "[place_enrichment] Starting fictional fallback for '%s' (user=%s conversation=%s)",
        query.target or query.raw_text,
        user_id,
        conversation_id,
    )

    try:
        result = run_coro(
            resolve_fictional(
                query,
                anchor,
                meta,
                store,
                user_id,
                conversation_id,
            )
        )
    except Exception:
        logger.exception(
            "[place_enrichment] Fictional fallback failed for '%s'",
            query.target or query.raw_text,
        )
        raise

    candidate_count = len(result.candidates or [])

    logger.info(
        "[place_enrichment] Completed fictional fallback for '%s' status=%s candidates=%s",
        query.target or query.raw_text,
        result.status,
        candidate_count,
    )

    return {
        "status": result.status,
        "candidate_count": candidate_count,
    }


def _serialize_query(query: PlaceQuery) -> Dict[str, Any]:
    return {
        "raw_text": query.raw_text,
        "normalized": query.normalized,
        "is_travel": query.is_travel,
        "target": query.target,
        "transport_hint": query.transport_hint,
    }


def _serialize_anchor(anchor: Anchor) -> Dict[str, Any]:
    return {
        "scope": anchor.scope,
        "label": anchor.label,
        "lat": anchor.lat,
        "lon": anchor.lon,
        "primary_city": anchor.primary_city,
        "region": anchor.region,
        "country": anchor.country,
        "world_name": anchor.world_name,
        "hints": dict(anchor.hints or {}),
    }


def _serialize_candidates(candidates: List[Place]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for place in candidates:
        meta = place.meta if isinstance(place.meta, dict) else {}
        out.append(
            {
                "name": place.name,
                "level": place.level,
                "place_id": meta.get("placeId"),
                "uri": meta.get("uri"),
            }
        )
    return out


def enqueue(
    *,
    user_id: str,
    conversation_id: str,
    query: PlaceQuery,
    anchor: Anchor,
    result: ResolveResult,
    afc_max_calls: Optional[int] = None,
) -> None:
    """Enqueue the enrichment task with contextual metadata."""

    candidate_places: List[Place] = [
        candidate.place for candidate in (result.candidates or [])
    ][:5]

    payload: Dict[str, Any] = {
        "user_id": str(user_id),
        "conversation_id": str(conversation_id),
        "query": _serialize_query(query),
        "anchor": _serialize_anchor(anchor),
        "result_status": getattr(result, "status", None),
        "candidates": _serialize_candidates(candidate_places),
    }
    if afc_max_calls is not None:
        try:
            payload["afc_max_calls"] = max(1, int(afc_max_calls))
        except (TypeError, ValueError):
            logger.debug(
                "[place_enrichment] Invalid afc_max_calls argument=%r", afc_max_calls
            )

    try:
        enrich.apply_async(kwargs={"payload": payload})
    except Exception:
        logger.warning(
            "[place_enrichment] Failed to enqueue enrichment task", exc_info=True
        )


def enqueue_fictional_fallback(
    *,
    user_id: str,
    conversation_id: str,
    query: PlaceQuery,
    anchor: Anchor,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Enqueue a background fictional resolver pass."""

    payload: Dict[str, Any] = {
        "user_id": str(user_id),
        "conversation_id": str(conversation_id),
        "query": _serialize_query(query),
        "anchor": _serialize_anchor(anchor),
    }

    sanitized_meta = _sanitize_meta(meta or {})
    if isinstance(sanitized_meta, dict) and sanitized_meta:
        payload["meta"] = sanitized_meta
    elif sanitized_meta not in (None, {}):
        payload["meta"] = {"value": sanitized_meta}

    try:
        fictional_fallback.apply_async(kwargs={"payload": payload})
    except Exception:
        logger.warning(
            "[place_enrichment] Failed to enqueue fictional fallback task", exc_info=True
        )


__all__ = ["enqueue", "enqueue_fictional_fallback", "enrich", "fictional_fallback"]
