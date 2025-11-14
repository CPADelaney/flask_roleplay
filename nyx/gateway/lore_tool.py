"""Async helpers that expose lore orchestrator operations for Nyx tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency guard
    import tiktoken
except ImportError:  # pragma: no cover - graceful fallback if tiktoken unavailable
    tiktoken = None  # type: ignore[assignment]

from lore.deep_cache import get_deep_lore_bundle, set_deep_lore_bundle
from lore.lore_orchestrator import get_lore_orchestrator

logger = logging.getLogger(__name__)


_TOKEN_LIMITS: Mapping[str, int] = {
    "minimal": 80,
    "short": 80,
    "brief": 80,
    "low": 80,
    "standard": 160,
    "medium": 160,
    "rich": 220,
    "comprehensive": 320,
    "verbose": 320,
    "default": 160,
    "normal": 160,
    "balanced": 220,
    "high": 320,
    "extended": 320,
    "full": 480,
    "detailed": 480,
}


async def _truncate_to_tokens(text: str, limit: int) -> str:
    """Trim ``text`` so it fits within ``limit`` tokens, preserving readability."""

    if limit <= 0:
        return ""
    if not text:
        return ""

    if tiktoken is not None:  # pragma: no branch - fast path when available
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - extremely defensive
            encoder = None
        if encoder is not None:
            tokens = encoder.encode(text)
            if len(tokens) <= limit:
                return text
            truncated = encoder.decode(tokens[:limit])
            if truncated and truncated[-1].isalnum():
                truncated = truncated.rstrip()
            if truncated and len(truncated) < len(text):
                truncated = truncated.rstrip() + "…"
            return truncated

    # Fallback: approximate tokens as whitespace-delimited words
    parts = text.split()
    if len(parts) <= limit:
        return text
    truncated = " ".join(parts[:limit]).rstrip()
    if truncated and len(truncated) < len(text):
        truncated += "…"
    return truncated


def _resolve_token_budget(detail_level: Optional[str]) -> int:
    if not detail_level:
        return _TOKEN_LIMITS["standard"]
    key = detail_level.strip().lower()
    return _TOKEN_LIMITS.get(key, _TOKEN_LIMITS["standard"])


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _fetch_lore_aspect(
    orchestrator: Any,
    aspect: str,
    payload: Mapping[str, Any],
    cache: MutableMapping[str, Any],
    *,
    user_id: Any,
    conversation_id: Any,
) -> Any:
    """Fetch an individual lore aspect using orchestrator helpers."""

    aspect_key = aspect.lower().strip()
    location_id = _coerce_int(payload.get("location_id"))
    npc_ids: Sequence[Any] = payload.get("npc_ids") or ()

    async def _get_location_context() -> Optional[Mapping[str, Any]]:
        if location_id is None:
            return None
        context = cache.get("location_context")
        if context is not None:
            return context
        cached_context = await get_deep_lore_bundle(user_id, conversation_id, location_id)
        if cached_context is not None:
            cache["location_context"] = cached_context
            return cached_context
        context = await orchestrator.da_get_comprehensive_location_context(location_id)
        if context is not None:
            cache["location_context"] = context
            await set_deep_lore_bundle(user_id, conversation_id, location_id, context)
        return context

    if aspect_key in {"location_history", "location_lore"}:
        if location_id is None:
            return None
        context = await _get_location_context()
        if not isinstance(context, Mapping):
            return context
        location_data = context.get("location") or {}
        lore_fields = {
            key[5:]: value
            for key, value in location_data.items()
            if key.startswith("lore_")
        }
        if lore_fields:
            return {
                "location_id": location_data.get("id") or location_data.get("location_id"),
                "name": location_data.get("name") or location_data.get("location_name"),
                "lore": lore_fields,
            }
        return location_data

    if aspect_key in {"location_flavor", "location_atmosphere"}:
        location_name = payload.get("location_name")
        if location_name is None and location_id is not None:
            context = await _get_location_context()
            loc = context.get("location") or {} if isinstance(context, Mapping) else {}
            location_name = loc.get("name") or loc.get("location_name")
        if not location_name:
            return None
        try:
            flavor_payload = await orchestrator.enhanced_scene_description(str(location_name))
        except AttributeError:
            flavor_payload = await orchestrator.generate_scene_description(str(location_name))
        except Exception:
            logger.exception("Failed to fetch location flavor for %s", location_name)
            return None
        if isinstance(flavor_payload, Mapping):
            # Prefer primary description field, fall back to textual serialization
            for key in ("description", "scene", "text", "content"):
                value = flavor_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return dict(flavor_payload)
        return flavor_payload

    if aspect_key in {"npc_backstory", "npc_histories"}:
        npc_map: Dict[str, Any] = {}
        coerced_ids = [_coerce_int(npc_id) for npc_id in npc_ids]
        filtered_ids = [npc_id for npc_id in coerced_ids if npc_id is not None]
        if not filtered_ids:
            return npc_map
        for npc_id in filtered_ids:
            try:
                details = await orchestrator.da_get_npc_details(npc_id=npc_id)
            except Exception:
                logger.exception("Failed to fetch NPC details for %s", npc_id)
                continue
            if not details:
                continue
            entry: Dict[str, Any] = {}
            if "backstory" in details:
                entry["backstory"] = details.get("backstory")
            if "lore_backstory" in details:
                entry.setdefault("backstory" if isinstance(details.get("lore_backstory"), str) else "lore_backstory", details.get("lore_backstory"))
            remaining = {k: v for k, v in details.items() if k not in {"backstory", "lore_backstory"}}
            if remaining:
                entry["metadata"] = remaining
            if entry:
                npc_map[str(npc_id)] = entry
        return npc_map

    if aspect_key in {"religious_context", "religion"}:
        if location_id is None:
            return []
        context = await _get_location_context()
        if not isinstance(context, Mapping):
            return context or []
        cultural = context.get("cultural_context") or {}
        religious_elements = cultural.get("religious_elements")
        if religious_elements is not None:
            return religious_elements
        return cultural

    if aspect_key in {"political_context", "politics"}:
        if location_id is None:
            return {}
        context = await _get_location_context()
        if not isinstance(context, Mapping):
            return context or {}
        return context.get("political_context") or {}

    logger.debug("Unknown lore aspect requested: %s", aspect_key)
    return None


async def handle_lore_operation(
    *,
    user_id: int,
    conversation_id: int,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Route lore operations through the orchestrator and normalise responses."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    aspects: Iterable[Any] = payload.get("aspects") or []
    aspects = [str(aspect) for aspect in aspects if aspect]
    if not aspects:
        return {}

    detail_level = payload.get("detail_level")
    token_budget = _resolve_token_budget(detail_level if isinstance(detail_level, str) else None)

    orchestrator = await get_lore_orchestrator(int(user_id), int(conversation_id))

    results: Dict[str, Any] = {}
    cache: Dict[str, Any] = {}

    for aspect in aspects:
        try:
            value = await _fetch_lore_aspect(
                orchestrator,
                aspect,
                payload,
                cache,
                user_id=user_id,
                conversation_id=conversation_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Lore aspect fetch failed", extra={"aspect": aspect})
            continue
        if value is None:
            continue
        if isinstance(value, str):
            results[aspect] = await _truncate_to_tokens(value, token_budget)
        else:
            results[aspect] = value

    return results


__all__ = ["handle_lore_operation"]
