# nyx/nyx_agent/orchestrator.py
"""Main orchestration and runtime functions for Nyx Agent SDK with enhanced reality enforcement"""

import asyncio
import dataclasses
import json
import logging
import math
import time
import uuid
from typing import Dict, List, Any, Optional, Iterable, TYPE_CHECKING, Callable, Mapping
from contextlib import asynccontextmanager

from agents import Agent, RunConfig, RunContextWrapper, ModelSettings
from db.connection import get_db_connection_context

# Import enhanced feasibility functions
from nyx.nyx_agent.feasibility import (
    detect_setting_type,
    assess_action_feasibility_fast,
)

from nyx.gateway.llm_gateway import execute, execute_stream, LLMRequest, LLMOperation
from nyx.telemetry.metrics import (
    REQUEST_LATENCY,
    TASK_FAILURES,
    record_queue_delay_from_context,
)
from nyx.telemetry.tracing import trace_step
from .config import Config

if TYPE_CHECKING:
    from nyx.nyx_agent_sdk import NyxSDKConfig
from . import context as context_module
from .context import NyxContext, PackedContext
from ._feasibility_helpers import (
    DeferPromptContext,
    build_defer_fallback_text,
    build_defer_prompt,
    coalesce_agent_output_text,
    extract_defer_details,
)
from .models import *
from .agents import (
    nyx_main_agent,
    nyx_defer_agent,
    reflection_agent,
    movement_transition_agent,
    DEFAULT_MODEL_SETTINGS,
)
from .assembly import assemble_nyx_response, resolve_scene_requests
from .tools import (
    update_relationship_state,
    generate_universal_updates_impl,
)
from .utils import (
    _did_call_tool,
    _extract_last_assistant_text,
    _js,
    sanitize_agent_tools_in_place,
    log_strict_hits,
    extract_runner_response,
)

# ---- optional punishment enforcer (refactored to accept meta) ---------------
try:
    # expects signature: enforce_all_rules_on_player(player_name, user_id, conversation_id, metadata)
    from logic.rule_enforcement import enforce_all_rules_on_player  # type: ignore
except Exception:  # pragma: no cover
    enforce_all_rules_on_player = None  # type: ignore

logger = logging.getLogger(__name__)

_LOCATION_RULE_PREFIXES = (
    "location_resolver:",
    "location:",
    "unavailable:location",
)

_LOCATION_RULES_EXACT = {
    "npc_absent",      # "No sign of X anywhere in this scene..."
    "item_absent",     # "No sign of that item / stash..."
}


DEFAULT_DEFER_RUN_TIMEOUT_SECONDS: float = getattr(
    Config, "DEFER_RUN_TIMEOUT_SECONDS", 45.0
)
DEFER_RUN_TIMEOUT_SECONDS: float = DEFAULT_DEFER_RUN_TIMEOUT_SECONDS

_MOVEMENT_FAST_PATH_CATEGORIES: set[str] = {"movement", "mundane_action"}
_MOVEMENT_FAST_PATH_DISTANCE_THRESHOLD_KM: float = 5.0

_MAIN_AGENT_RUN_LIMITS: Dict[str, Any] = {
    "max_turns": 6,
    "max_tool_calls": 4,
    "max_auto_messages": 3,
    "max_output_tokens": 1800,
    "default_total_timeout_budget": 4.0,
    "min_total_timeout_budget": 3.0,
    "max_total_timeout_budget": 12.0,
    "per_turn_timeout": 1.0,
    "per_step_timeout": 0.5,
}

_MOVEMENT_RUN_LIMITS: Dict[str, Any] = {
    "max_turns": 2,
    "max_tool_calls": 1,
    "max_auto_messages": 1,
    "max_output_tokens": 600,
    "default_total_timeout_budget": 3.0,
    "min_total_timeout_budget": 1.5,
    "max_total_timeout_budget": 8.0,
    "per_turn_timeout": 0.9,
    "per_step_timeout": 0.4,
}

_MOVEMENT_FAST_PATH_MODEL_SETTINGS = ModelSettings(
    strict_tools=False,
    max_tokens=600,
)


def _safe_time_budget(
    value: Optional[float], *, minimum: float = 0.25, maximum: Optional[float] = None
) -> Optional[float]:
    """Return a clamped positive time budget or ``None`` when invalid."""

    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not parsed or parsed <= 0:
        return None

    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _derive_run_limit_kwargs(
    limit_template: Dict[str, Any],
    *,
    time_budget: Optional[float] = None,
) -> Dict[str, Any]:
    """Merge static run limit defaults with the remaining time budget."""

    limits: Dict[str, Any] = {}
    for key in (
        "max_turns",
        "max_tool_calls",
        "max_auto_messages",
        "max_output_tokens",
    ):
        value = limit_template.get(key)
        if value is not None:
            limits[key] = value

    minimum_budget = float(limit_template.get("min_total_timeout_budget")) if limit_template.get("min_total_timeout_budget") else 0.25
    maximum_budget = (
        float(limit_template.get("max_total_timeout_budget"))
        if limit_template.get("max_total_timeout_budget")
        else None
    )
    default_budget = limit_template.get("default_total_timeout_budget")
    budget_source = time_budget if time_budget is not None else default_budget

    safe_budget = _safe_time_budget(
        budget_source,
        minimum=minimum_budget,
        maximum=maximum_budget,
    )
    per_turn_default = limit_template.get("per_turn_timeout")
    per_step_default = limit_template.get("per_step_timeout")

    if safe_budget is not None:
        limits["total_timeout_budget"] = round(safe_budget, 3)
        max_turns = max(1, int(limits.get("max_turns") or limit_template.get("max_turns") or 1))
        per_turn_cap = safe_budget / max_turns
        per_turn_value = per_turn_default if per_turn_default is not None else per_turn_cap
        per_turn_timeout = min(per_turn_value, per_turn_cap)
        per_step_value = per_step_default if per_step_default is not None else per_turn_timeout
        per_step_timeout = min(per_step_value, per_turn_timeout)
    else:
        per_turn_timeout = per_turn_default
        per_step_timeout = per_step_default if per_step_default is not None else per_turn_timeout

    if per_turn_timeout is not None:
        limits["per_turn_timeout"] = round(float(per_turn_timeout), 3)
    if per_step_timeout is not None:
        limits["per_step_timeout"] = round(float(per_step_timeout), 3)

    return limits


def _build_run_config_with_limits(
    *,
    base_kwargs: Dict[str, Any],
    limit_template: Dict[str, Any],
    time_budget: Optional[float],
    trace_id: str,
    log_label: str,
) -> tuple[RunConfig, Dict[str, Any]]:
    """Instantiate ``RunConfig`` with explicit limit kwargs and log them."""

    limit_kwargs = _derive_run_limit_kwargs(limit_template, time_budget=time_budget)
    run_config_kwargs = dict(base_kwargs)
    try:
        run_config = RunConfig(**{**run_config_kwargs, **limit_kwargs})
    except TypeError:
        run_config = RunConfig(**run_config_kwargs)
        for key, value in limit_kwargs.items():
            setattr(run_config, key, value)

    logger.info(
        f"[{trace_id}] [{log_label}] run_config_limits={_js(limit_kwargs)}",
    )
    return run_config, limit_kwargs


def _sorted_section_keys(section: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(section, dict):
        return []
    return sorted(section.keys())


def _locate_section_value(packed_context: "PackedContext", key: str) -> Any:
    for attr in ("canonical", "optional", "summarized"):
        section = getattr(packed_context, attr, None)
        if isinstance(section, dict) and key in section:
            return section[key]
    return None


def _count_npcs_from_value(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        items = value.get("npcs")
        if isinstance(items, list):
            return len(items)
        # Legacy dict format keyed by npc id/name
        return len(value) if value else 0
    if isinstance(value, list):
        return len(value)
    return None


def _count_memory_snippets(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        count_field = value.get("count")
        if isinstance(count_field, int):
            return max(count_field, 0)
        total = 0
        for item in value.values():
            if isinstance(item, list):
                total += len(item)
        return total if total else None
    if isinstance(value, list):
        return len(value)
    return None


def _extract_context_counts(packed_context: "PackedContext") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    npc_value = _locate_section_value(packed_context, "npcs")
    npc_count = _count_npcs_from_value(npc_value)
    if npc_count:
        counts["npcs"] = npc_count

    memory_value = _locate_section_value(packed_context, "memories")
    memory_count = _count_memory_snippets(memory_value)
    if memory_count:
        counts["memory_snippets"] = memory_count

    return counts


def _log_packed_context_details(packed_context: "PackedContext", trace_id: str) -> None:
    canonical_keys = _sorted_section_keys(getattr(packed_context, "canonical", None))
    optional_keys = _sorted_section_keys(getattr(packed_context, "optional", None))
    summarized_keys = _sorted_section_keys(getattr(packed_context, "summarized", None))
    metadata_keys = _sorted_section_keys(getattr(packed_context, "metadata", None))

    counts = _extract_context_counts(packed_context)

    try:
        packed_context.to_dict()
    except Exception:
        logger.debug(
            f"[{trace_id}] Failed to serialize packed context for logging",
            exc_info=True,
        )

    logger.info(
        "[%s] Packed context tokens=%s canonical=%s optional=%s summarized=%s metadata=%s counts=%s",
        trace_id,
        getattr(packed_context, "tokens_used", "unknown"),
        canonical_keys,
        optional_keys,
        summarized_keys,
        metadata_keys,
        counts or {},
    )


def _resolve_model_label(model: Any) -> str:
    if model is None:
        return "unknown"
    label = getattr(model, "model", None) or getattr(model, "name", None)
    if label:
        return str(label)
    return str(model)


def _is_soft_location_only_violation(fast_result: Dict[str, Any]) -> bool:
    """
    Return True if the fast feasibility result is a deny *only* because
    the target/place is missing or unresolved (location_resolver / npc_absent /
    item_absent), with no physics/magic/rule-based hard impossibilities.
    """

    if not isinstance(fast_result, dict):
        return False

    overall = fast_result.get("overall") or {}
    if overall.get("feasible") is not False:
        return False

    strategy = (overall.get("strategy") or "").lower()
    if strategy != "deny":
        return False

    per_intent = fast_result.get("per_intent") or []
    if not per_intent:
        # No detail → treat as hard deny
        return False

    any_violation = False

    for intent in per_intent:
        if intent.get("feasible") is True:
            # Allowed intent is fine
            continue

        violations = intent.get("violations") or []
        if not violations:
            # “Deny with no violations” is not soft
            return False

        for v in violations:
            rule = str(v.get("rule") or "").lower()
            if not rule:
                return False

            any_violation = True

            # Location-ish rules we treat as "soft"
            if rule in _LOCATION_RULES_EXACT:
                continue
            if any(rule.startswith(prefix) for prefix in _LOCATION_RULE_PREFIXES):
                continue

            # Anything else (established_impossibility, category:foo, scene:bar, etc.)
            # means this is NOT a pure location-only problem.
            return False

    return any_violation


def get_defer_run_timeout_seconds(config: Optional["NyxSDKConfig"] = None) -> float:
    """Return the defer agent timeout, preferring SDK configuration when provided."""

    if config is not None:
        timeout = getattr(config, "request_timeout_seconds", None)
        try:
            parsed = float(timeout)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0:
            return parsed
    return DEFER_RUN_TIMEOUT_SECONDS


async def _generate_defer_taunt(
    context: DeferPromptContext,
    trace_id: str,
    nyx_context: Optional[NyxContext] = None,
) -> Optional[str]:
    """Ask Nyx to craft a defer response; fall back to None on failure."""

    if nyx_defer_agent is None:
        return None

    prompt = build_defer_prompt(context)
    if not prompt.strip():
        return None

    run_kwargs = {"max_turns": 2}
    if RunContextWrapper is not None and nyx_context is not None:
        try:
            run_kwargs["context"] = RunContextWrapper(nyx_context)
        except Exception:
            logger.debug(f"[{trace_id}] Failed to wrap Nyx context for defer taunt", exc_info=True)

    try:
        # Use remaining turn budget if available; else fall back to default
        remaining = None
        try:
            import time as _t
            dl = (getattr(nyx_context, "current_context", {}) or {}).get("_deadline")
            if dl:
                remaining = max(0.25, dl - _t.monotonic())
        except Exception:
            remaining = None
        request = LLMRequest(
            prompt=prompt,
            agent=nyx_defer_agent,
            metadata={"operation": LLMOperation.ORCHESTRATION.value},
            runner_kwargs=run_kwargs or None,
        )
        result_wrapper = await asyncio.wait_for(
            _execute_llm(request),
            timeout=remaining if remaining is not None else get_defer_run_timeout_seconds(),
        )
        result = result_wrapper.raw
    except asyncio.TimeoutError:
        logger.debug(
            f"[{trace_id}] Nyx defer taunt generation timed out",
            exc_info=True,
        )
        return None
    except Exception:
        logger.debug(f"[{trace_id}] Nyx defer taunt generation failed", exc_info=True)
        return None

    return coalesce_agent_output_text(result)


def _is_meaningful(value: Any) -> bool:
    """Return True if the value carries information (non-empty/None)."""
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, set, dict)):
        return bool(value)
    return True


def _ensure_list(value: Any) -> List[Any]:
    """Coerce arbitrary metadata into a list for scene storage."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (set, tuple)):
        return list(value)
    return [value]


def _normalize_scene_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize legacy scene metadata keys onto canonical fields."""
    normalized: Dict[str, Any] = dict(context or {})

    def adopt(canonical: str, legacy_keys: Iterable[str], default_factory: Optional[Any] = None,
              coerce_list: bool = False) -> None:
        if not _is_meaningful(normalized.get(canonical)):
            for key in legacy_keys:
                if key in normalized and _is_meaningful(normalized.get(key)):
                    value = normalized[key]
                    normalized[canonical] = _ensure_list(value) if coerce_list else value
                    break

        if canonical not in normalized:
            if callable(default_factory):
                normalized[canonical] = default_factory()
            elif default_factory is not None:
                normalized[canonical] = default_factory
        elif coerce_list:
            normalized[canonical] = _ensure_list(normalized.get(canonical))

    adopt("current_location", ("location", "active_location", "scene_location"), default_factory=lambda: {})
    adopt(
        "present_npcs",
        ("npc_present", "npcs", "present_entities", "participants", "active_npcs"),
        default_factory=list,
        coerce_list=True,
    )
    adopt(
        "available_items",
        ("items", "inventory", "inventory_items", "scene_items"),
        default_factory=list,
        coerce_list=True,
    )
    adopt(
        "recent_interactions",
        ("recent_turns", "recent_dialogue", "recent_messages"),
        default_factory=list,
        coerce_list=True,
    )

    if isinstance(normalized.get("recent_interactions"), list):
        canonical_turns: List[Dict[str, Any]] = []
        for entry in normalized.get("recent_interactions", []):
            if not isinstance(entry, dict):
                continue
            sender = entry.get("sender")
            content = entry.get("content")
            if sender is None and content is None:
                continue
            turn: Dict[str, Any] = {}
            if sender is not None:
                turn["sender"] = sender
            if content is not None:
                turn["content"] = content
            canonical_turns.append(turn)
        normalized["recent_interactions"] = canonical_turns
        if not _is_meaningful(normalized.get("recent_turns")):
            normalized["recent_turns"] = canonical_turns

    if not _is_meaningful(normalized.get("present_entities")):
        normalized["present_entities"] = list(normalized.get("present_npcs", []))

    return normalized


async def _execute_llm(request: LLMRequest):
    """Execute an LLM request via the Nyx gateway."""

    return await execute(request)


def _preserve_hydrated_location(target: Dict[str, Any], location: Any) -> None:
    """Ensure hydrated location metadata survives context merges."""

    if not _is_meaningful(location):
        return

    for key in ("current_location", "location", "location_name", "location_id"):
        if not _is_meaningful(target.get(key)):
            target[key] = location


def _intents_are_movement_only(fast_result: Dict[str, Any]) -> bool:
    """Return True when every intent category is movement-centric."""

    per_intent = fast_result.get("per_intent")
    if not isinstance(per_intent, list) or not per_intent:
        return False

    for intent in per_intent:
        if not isinstance(intent, dict):
            return False

        categories = intent.get("categories") or []
        normalized = {
            str(cat).strip().lower()
            for cat in categories
            if isinstance(cat, str) and cat.strip()
        }

        if not normalized:
            return False

        if not normalized.issubset(_MOVEMENT_FAST_PATH_CATEGORIES):
            return False

    return True


def _context_payload_for_router(nyx_context: NyxContext) -> Dict[str, Any]:
    """Prepare a sanitized context payload for the location router."""

    payload: Dict[str, Any] = {}
    current = getattr(nyx_context, "current_context", {}) or {}

    for key, value in current.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        payload[key] = value

    return payload


def _extract_location_payload(resolve_result: Any) -> Dict[str, Any]:
    """Normalize a ResolveResult location payload for context storage."""

    if not resolve_result:
        return {}

    payload: Dict[str, Any] = {}
    location = getattr(resolve_result, "location", None)

    if location:
        if dataclasses.is_dataclass(location):
            payload = dataclasses.asdict(location)
        elif isinstance(location, dict):
            payload = dict(location)
        else:
            payload = {
                "id": getattr(location, "id", getattr(location, "location_id", None)),
                "location_id": getattr(location, "location_id", None),
                "name": getattr(location, "location_name", getattr(location, "name", None)),
                "location_name": getattr(location, "location_name", None),
                "location_name_lc": getattr(location, "location_name_lc", None),
                "location_type": getattr(location, "location_type", None),
                "city": getattr(location, "city", None),
                "region": getattr(location, "region", None),
                "country": getattr(location, "country", None),
                "description": getattr(location, "description", None),
                "external_place_id": getattr(location, "external_place_id", None),
                "lat": getattr(location, "lat", getattr(location, "latitude", None)),
                "lon": getattr(location, "lon", getattr(location, "longitude", getattr(location, "lng", None))),
            }
        name = payload.get("location_name") or payload.get("name")
        if name:
            payload.setdefault("name", name)
        return {k: v for k, v in payload.items() if v not in (None, "", [])}

    candidates = getattr(resolve_result, "candidates", None) or []
    if candidates:
        first = candidates[0]
        place = getattr(first, "place", None)
        if place:
            place_payload = {
                "name": getattr(place, "name", None),
                "key": getattr(place, "key", None),
                "level": getattr(place, "level", None),
            }
            address = getattr(place, "address", None)
            if address:
                place_payload["address"] = address
            meta = getattr(place, "meta", None)
            if meta:
                place_payload["meta"] = meta
            return {k: v for k, v in place_payload.items() if v not in (None, "", [])}

    return {}


def _compact_npcs(npcs: Any) -> List[Dict[str, Any]]:
    """Return at most two lightweight NPC descriptors."""

    compact: List[Dict[str, Any]] = []
    if not isinstance(npcs, list):
        return compact

    for npc in npcs:
        entry: Dict[str, Any] = {}
        if isinstance(npc, dict):
            for key in ("name", "nickname", "role", "description"):
                value = npc.get(key)
                if value:
                    entry[key] = value
            rel = npc.get("relationship")
            if isinstance(rel, dict):
                entry["relationship"] = {
                    k: rel[k]
                    for k in ("trust", "respect", "closeness")
                    if k in rel and rel[k] is not None
                }
        elif npc:
            entry["name"] = str(npc)

        if entry:
            compact.append(entry)

        if len(compact) >= 2:
            break

    return compact


def _extract_recent_turns(
    nyx_context: NyxContext,
    packed_context: Optional[PackedContext],
) -> List[Dict[str, Any]]:
    """Collect the last few turns from packed context or live context."""

    candidate_sequences: List[List[Any]] = []

    if packed_context is not None and hasattr(packed_context, "canonical"):
        canonical = getattr(packed_context, "canonical", {}) or {}
        for key in ("recent_interactions", "recent_turns"):
            sequence = canonical.get(key)
            if isinstance(sequence, list) and sequence:
                candidate_sequences.append(sequence)

    current_context = getattr(nyx_context, "current_context", {}) or {}
    for key in ("recent_interactions", "recent_turns"):
        sequence = current_context.get(key)
        if isinstance(sequence, list) and sequence:
            candidate_sequences.append(sequence)

    if not candidate_sequences:
        return []

    turns_source: List[Any] = []
    for sequence in candidate_sequences:
        if isinstance(sequence, list) and sequence:
            turns_source = sequence

    recent = turns_source[-3:]
    normalized: List[Dict[str, Any]] = []

    for turn in recent:
        if not isinstance(turn, dict):
            continue
        entry: Dict[str, Any] = {}
        sender = turn.get("sender") or turn.get("speaker")
        if sender:
            entry["sender"] = sender
        content = turn.get("content") or turn.get("text") or turn.get("message")
        if content:
            entry["content"] = content
        if entry:
            normalized.append(entry)

    return normalized


def _build_movement_scene_bundle(
    nyx_context: NyxContext,
    packed_context: Optional[PackedContext],
    location_payload: Dict[str, Any],
    movement_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose a compact scene summary for the movement fast path."""

    location_info: Dict[str, Any] = {}
    current_location = getattr(nyx_context, "current_context", {}).get("current_location")
    if isinstance(current_location, dict):
        location_info.update(current_location)
    if location_payload:
        location_info.update(location_payload)

    name = (
        location_info.get("name")
        or location_info.get("location_name")
        or location_info.get("location")
    )
    if name:
        location_info["name"] = name

    current_context = getattr(nyx_context, "current_context", {}) or {}
    recent_turns = _extract_recent_turns(nyx_context, packed_context)
    npcs = _compact_npcs(current_context.get("present_npcs"))

    npc_context = nyx_context.get_npc_context_for_response()
    conflict_context = nyx_context.get_conflict_context_for_response()
    lore_context = nyx_context.get_lore_context_for_response()

    snapshot = _extract_current_snapshot(nyx_context)
    canonical_location = _normalize_location_dict(
        current_context.get("current_location") or nyx_context.current_location
    )

    style_hints = {}
    for key in ("roleplay_config", "roleplay_style", "player_profile"):
        value = current_context.get(key)
        if value:
            style_hints[key] = value

    bundle: Dict[str, Any] = {
        "location": {k: v for k, v in location_info.items() if v not in (None, "")},
        "canonical_location": canonical_location,
        "npcs": npcs,
        "recent_turns": recent_turns,
        "npc_context": npc_context,
        "conflict_context": conflict_context,
        "lore_context": lore_context,
        "snapshot": snapshot,
        "style": style_hints,
    }

    if movement_meta:
        bundle["movement_meta"] = movement_meta

    return bundle


def _build_movement_transition_prompt(
    user_input: str,
    scene_bundle: Dict[str, Any],
    movement_meta: Dict[str, Any],
) -> str:
    """Create the lightweight transition prompt for the movement agent."""

    bundle_json = json.dumps(scene_bundle, ensure_ascii=False)

    origin = movement_meta.get("origin", {})
    destination = movement_meta.get("destination", {})
    approx_km = movement_meta.get("approx_distance_km")
    distance_line = (
        f"{approx_km:.1f} km" if isinstance(approx_km, (int, float)) else "unknown"
    )
    distance_class = movement_meta.get("distance_class") or "unknown"
    movement_kind = movement_meta.get("movement_kind") or "movement"
    same_place = movement_meta.get("destination_same_as_origin")
    travel_mode = movement_meta.get("travel_mode") or "unknown"
    travel_minutes = movement_meta.get("travel_duration_min")
    require_travel_choice = movement_meta.get("require_travel_choice", False)
    world_ctx = movement_meta.get("world", {})
    world_time = world_ctx.get("local_time") or "unknown"
    time_of_day = world_ctx.get("time_of_day") or "unknown"
    world_mood = world_ctx.get("world_mood") or world_ctx.get("mood") or "unknown"
    feasibility_meta = movement_meta.get("feasibility", {})
    soft_location_only = movement_meta.get("soft_location_only")
    teleport_allowed = feasibility_meta.get("teleport_allowed")
    has_currency = feasibility_meta.get("has_currency")

    def _format_location_block(label: str, payload: Mapping[str, Any]) -> str:
        lines = [f"{label}:"]
        for key in ("name", "city", "region", "country", "latitude", "longitude"):
            value = payload.get(key)
            if value not in (None, ""):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _summarize_context(label: str, payload: Mapping[str, Any], keys: Iterable[str]) -> str:
        if not isinstance(payload, Mapping) or not payload:
            return f"{label}: none"
        fragments = []
        for key in keys:
            value = payload.get(key)
            if value:
                fragments.append(f"{key}={value}")
        if not fragments:
            return f"{label}: none"
        return f"{label}: " + "; ".join(fragments)

    origin_block = _format_location_block("Origin", origin)
    destination_block = _format_location_block("Destination", destination)

    npc_context = scene_bundle.get("npc_context") or {}
    conflict_context = scene_bundle.get("conflict_context") or {}
    lore_context = scene_bundle.get("lore_context") or {}
    snapshot = scene_bundle.get("snapshot") or {}
    style_hints = scene_bundle.get("style") or {}
    canonical_location = scene_bundle.get("canonical_location") or {}

    npc_summary = _summarize_context(
        "NPCs",
        npc_context,
        ("scene_npcs", "npcs"),
    )
    conflict_summary = _summarize_context(
        "Conflict",
        conflict_context,
        ("active", "tensions"),
    )
    lore_summary = _summarize_context("Lore", lore_context, ("world", "nations"))
    style_summary = _summarize_context("Style", style_hints, style_hints.keys()) if style_hints else "Style: default"

    current_location_line = _summarize_context(
        "Canonical location",
        canonical_location,
        ("name", "city", "region", "country"),
    )

    target_line = "Target: already at destination; move within the location." if same_place else "Target: new location"

    travel_time_line = (
        f"{int(travel_minutes)} minutes" if isinstance(travel_minutes, (int, float)) else "unknown"
    )

    prompt_lines = [
        f"Player input: {user_input}",
        origin_block,
        destination_block,
        f"Approximate distance: {distance_line}",
        f"Movement scale: {distance_class}",
        f"Movement kind: {movement_kind} (same_place={same_place})",
        f"Inferred travel mode: {travel_mode}",
        f"Estimated travel time: {travel_time_line}",
        target_line,
        f"Current time: {world_time} ({time_of_day})",
        f"World mood: {world_mood}",
        current_location_line,
        npc_summary,
        conflict_summary,
        lore_summary,
        style_summary,
        "Snapshot summary: " + json.dumps(snapshot) if snapshot else "Snapshot summary: none",
        "Feasibility:",
        f"- soft_location_only: {soft_location_only}",
        f"- teleport_allowed: {teleport_allowed}",
        f"- has_currency: {has_currency}",
        "Instructions:",
        f"REQUIRE_TRAVEL_CHOICE: {require_travel_choice}",
        "- If REQUIRE_TRAVEL_CHOICE is true:",
        "  - Do NOT narrate arrival.",
        "  - Ask the player, in character, how they are traveling (e.g., by car, train, on foot).",
        "  - Do not assume a mode.",
        "- If REQUIRE_TRAVEL_CHOICE is false:",
        "  - Do NOT ask any travel-mode questions.",
        "  - Narrate the movement and arrival using travel_mode and travel_duration_min.",
        "- Keep the narration compact but flavorful (1–3 paragraphs, not a single generic sentence).",
        f"Scene bundle JSON: {bundle_json}",
    ]

    return "\n".join(prompt_lines)


async def _llm_fallback_movement_narration(movement_meta: Dict[str, Any]) -> str:
    location = movement_meta.get("destination") or movement_meta.get("origin") or {}
    loc_name = location.get("name") or location.get("location_name") or "your destination"
    world = movement_meta.get("world") or {}
    tod = world.get("time_of_day") or "timeless"
    mood = world.get("world_mood") or "neutral"

    prompt = (
        "Write a single short, vivid description of the player moving toward a place.\n\n"
        f"Destination name: {loc_name}\n"
        f"Time of day: {tod}\n"
        f"World mood: {mood}\n\n"
        "Keep it in second person, one paragraph, no inner monologue, no dialogue."
    )

    from logic.gpt_utils import call_gpt_text

    try:
        text = await call_gpt_text(prompt, model="gpt-5-nano")
        return text.strip() or f"You head toward {loc_name}."
    except Exception:
        return f"You head toward {loc_name}."


def _movement_requires_clarification(text: str) -> bool:
    """
    Heuristic to decide if the Movement Transition agent is asking the player
    to choose HOW they travel (e.g. “fly or drive?”), instead of treating travel
    as already resolved.
    """

    if not text:
        return False

    lowered = text.strip().lower()
    # If there is a question mark toward the end AND we see mode-choice language,
    # treat this as a clarification turn.
    if "?" in lowered[-200:]:
        mode_keywords = [
            "how do you get",
            "how are you getting",
            "fly", "flight", "plane",
            "drive", "road trip", "car",
            "train", "bus",
            "which do you choose",
            "would you rather",
        ]
        if any(kw in lowered for kw in mode_keywords):
            return True

    return False


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonicalize_location_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def _normalize_location_dict(location: Any) -> Dict[str, Any]:
    if not location:
        return {}

    if isinstance(location, str):
        raw = {"name": location, "location_name": location}
    elif dataclasses.is_dataclass(location):
        raw = dataclasses.asdict(location)
    elif isinstance(location, Mapping):
        raw = dict(location)
    else:
        raw = {}
        for attr in (
            "id",
            "location_id",
            "name",
            "location_name",
            "location",
            "city",
            "region",
            "country",
            "description",
            "lat",
            "lon",
            "latitude",
            "longitude",
            "g",
            "external_place_id",
            "location_type",
            "location_name_lc",
        ):
            value = getattr(location, attr, None)
            if value not in (None, ""):
                raw[attr] = value

    normalized: Dict[str, Any] = {}

    for key in (
        "id",
        "location_id",
        "name",
        "location_name",
        "location",
        "city",
        "region",
        "country",
        "description",
        "external_place_id",
        "location_type",
        "location_name_lc",
    ):
        value = raw.get(key)
        if value not in (None, ""):
            normalized[key] = value

    lat = raw.get("lat")
    if lat is None:
        lat = raw.get("latitude")
    if lat is None:
        lat = raw.get("g")
    lon = raw.get("lon")
    if lon is None:
        lon = raw.get("longitude")
    if lon is None:
        lon = raw.get("lng")

    lat_f = _coerce_float(lat)
    lon_f = _coerce_float(lon)
    if lat_f is not None:
        normalized["latitude"] = lat_f
    if lon_f is not None:
        normalized["longitude"] = lon_f

    return normalized


def _haversine_km(
    origin_lat: Optional[float],
    origin_lon: Optional[float],
    dest_lat: Optional[float],
    dest_lon: Optional[float],
) -> Optional[float]:
    if None in (origin_lat, origin_lon, dest_lat, dest_lon):
        return None

    lat1, lon1, lat2, lon2 = map(math.radians, [origin_lat, origin_lon, dest_lat, dest_lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c


def _classify_distance(
    distance_km: Optional[float],
    origin: Mapping[str, Any],
    destination: Mapping[str, Any],
) -> str:
    if isinstance(distance_km, (int, float)):
        if distance_km < 1:
            return "short"
        if distance_km < 20:
            return "city"
        return "long"

    origin_city = _canonicalize_location_token(origin.get("city"))
    destination_city = _canonicalize_location_token(destination.get("city"))
    if origin_city and destination_city and origin_city != destination_city:
        return "long"

    origin_region = _canonicalize_location_token(origin.get("region"))
    destination_region = _canonicalize_location_token(destination.get("region"))
    if origin_region and destination_region and origin_region != destination_region:
        return "city"

    origin_country = _canonicalize_location_token(origin.get("country"))
    destination_country = _canonicalize_location_token(destination.get("country"))
    if origin_country and destination_country and origin_country != destination_country:
        return "long"

    return "unknown"


_TRAVEL_SPEED_KMH: Dict[str, float] = {
    "walk": 4.0,
    "bike": 15.0,
    "car": 50.0,
    "bus": 35.0,
    "train": 90.0,
    "metro": 40.0,
    "air": 750.0,
    "teleport": 999_999.0,
}


def _extract_travel_mode_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()

    keyword_map = {
        "walk": ["walk", "on foot", "stroll", "wander"],
        "bike": ["bike", "bicycle", "cycle"],
        "car": ["drive", "car", "uber", "lyft", "taxi"],
        "bus": ["bus", "coach"],
        "train": ["train", "rail", "subway", "metro", "tube"],
        "air": ["fly", "plane", "flight", "airport"],
        "teleport": ["teleport", "portal", "instant transmission"],
    }

    for mode, keywords in keyword_map.items():
        if any(k in lowered for k in keywords):
            return mode
    return None


def _infer_travel_mode(user_text: str, distance_km: Optional[float], distance_class: str) -> Optional[str]:
    explicit = _extract_travel_mode_from_text(user_text)
    if explicit:
        return explicit

    if not isinstance(distance_km, (int, float)):
        return None

    if distance_km < 1:
        return "walk"
    if distance_km < 20:
        return "walk"
    if distance_km < 300:
        return "car"
    if distance_km < 1200:
        return "train"
    return "air"


def _estimate_travel_duration_minutes(
    mode: Optional[str],
    distance_km: Optional[float],
    *,
    movement_kind: str = "new_location",
) -> Optional[int]:
    if mode == "teleport":
        return 0

    if not isinstance(distance_km, (int, float)) or distance_km <= 0:
        if movement_kind == "within_location":
            return 5
        return None

    speed = _TRAVEL_SPEED_KMH.get(mode or "", 50.0)
    if speed <= 0:
        speed = 50.0

    hours = distance_km / speed
    minutes = max(1, int(hours * 60))

    if mode == "air":
        minutes += 90
    elif mode in ("car", "bus", "train", "metro"):
        minutes += 10

    return min(minutes, 24 * 60)


def _locations_look_local(origin: Mapping[str, Any], destination: Mapping[str, Any]) -> bool:
    origin_city = _canonicalize_location_token(origin.get("city"))
    destination_city = _canonicalize_location_token(destination.get("city"))
    if origin_city and destination_city and origin_city != destination_city:
        return False

    origin_region = _canonicalize_location_token(origin.get("region"))
    destination_region = _canonicalize_location_token(destination.get("region"))
    if origin_region and destination_region and origin_region != destination_region:
        return False

    origin_country = _canonicalize_location_token(origin.get("country"))
    destination_country = _canonicalize_location_token(destination.get("country"))
    if origin_country and destination_country and origin_country != destination_country:
        return False

    return True


def _locations_same_place(origin: Mapping[str, Any], destination: Mapping[str, Any]) -> bool:
    if not origin or not destination:
        return False

    origin_ext = origin.get("external_place_id") or origin.get("external_id")
    destination_ext = destination.get("external_place_id") or destination.get("external_id")
    if origin_ext and destination_ext and origin_ext == destination_ext:
        return True

    def _extract_name(payload: Mapping[str, Any]) -> str:
        for key in ("location_name_lc", "location_name", "name", "location"):
            value = payload.get(key)
            if value not in (None, ""):
                return _canonicalize_location_token(value)
        return ""

    origin_name = _extract_name(origin)
    destination_name = _extract_name(destination)

    if origin_name and destination_name and origin_name == destination_name:
        # Require that any overlapping geo descriptors do not conflict
        for key in ("city", "region", "country", "location_type"):
            origin_val = _canonicalize_location_token(origin.get(key))
            destination_val = _canonicalize_location_token(destination.get(key))
            if origin_val and destination_val and origin_val != destination_val:
                return False
        return True

    return False


def _extract_world_snapshot(nyx_context: NyxContext) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    world_state = getattr(nyx_context, "current_world_state", None)
    current_context = getattr(nyx_context, "current_context", {}) or {}

    time_source: Any = None
    mood_source: Any = None

    if isinstance(world_state, Mapping):
        time_source = world_state.get("current_time")
        mood_source = world_state.get("world_mood")
    elif world_state is not None:
        time_source = getattr(world_state, "current_time", None)
        mood_source = getattr(world_state, "world_mood", None)

    if time_source is None:
        time_source = current_context.get("current_time")

    if mood_source is None:
        mood_source = current_context.get("world_mood")

    def _extract_time_fields(source: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not source:
            return out
        if isinstance(source, Mapping):
            hour = source.get("hour")
            minute = source.get("minute")
            time_of_day = source.get("time_of_day") or source.get("phase")
        else:
            hour = getattr(source, "hour", None)
            minute = getattr(source, "minute", None)
            tod = getattr(source, "time_of_day", None)
            time_of_day = getattr(tod, "value", tod)

        if isinstance(time_of_day, Mapping):
            time_of_day = time_of_day.get("value") or time_of_day.get("name")

        if hour is not None and minute is not None:
            try:
                out["local_time"] = f"{int(hour):02d}:{int(minute):02d}"
            except (TypeError, ValueError):
                pass
        if time_of_day not in (None, ""):
            out["time_of_day"] = time_of_day
        return out

    snapshot.update(_extract_time_fields(time_source))

    if mood_source:
        if isinstance(mood_source, Mapping):
            mood_value = mood_source.get("value") or mood_source.get("state")
        else:
            mood_value = getattr(mood_source, "value", None) or str(mood_source)
        if mood_value not in (None, ""):
            snapshot["world_mood"] = mood_value

    return snapshot


def _advance_world_time(nyx_context: NyxContext, minutes: Optional[int]) -> None:
    if not isinstance(minutes, (int, float)) or minutes <= 0:
        return

    world_state = getattr(nyx_context, "current_world_state", None)
    ctx = getattr(nyx_context, "current_context", {}) or {}

    time_source: Any = None
    if isinstance(world_state, Mapping):
        time_source = world_state.get("current_time")

    if not isinstance(time_source, Mapping):
        time_source = ctx.get("current_time") or {}

    hour = int(time_source.get("hour", 12))
    minute = int(time_source.get("minute", 0))

    total = hour * 60 + minute + int(minutes)
    total %= 24 * 60

    new_hour = total // 60
    new_minute = total % 60

    time_source = dict(time_source)
    time_source["hour"] = new_hour
    time_source["minute"] = new_minute

    if 5 <= new_hour < 12:
        time_source["time_of_day"] = "morning"
    elif 12 <= new_hour < 17:
        time_source["time_of_day"] = "afternoon"
    elif 17 <= new_hour < 21:
        time_source["time_of_day"] = "evening"
    else:
        time_source["time_of_day"] = "night"

    if isinstance(world_state, dict):
        world_state["current_time"] = time_source
        nyx_context.current_world_state = world_state

    ctx["current_time"] = time_source
    nyx_context.current_context = ctx


def _apply_travel_time_to_world(nyx_context: NyxContext, movement_meta: Dict[str, Any]) -> None:
    minutes = movement_meta.get("travel_duration_min")
    try:
        _advance_world_time(nyx_context, minutes)
    except Exception:
        logger.debug("Failed to advance world time for movement", exc_info=True)


def _extract_current_snapshot(nyx_context: NyxContext) -> Dict[str, Any]:
    current_context = getattr(nyx_context, "current_context", {}) or {}
    snapshot_candidates = []

    for key in ("CurrentSnapshot", "current_snapshot", "currentSnapshot"):
        candidate = current_context.get(key)
        if isinstance(candidate, Mapping):
            snapshot_candidates.append(candidate)

    aggregator_data = current_context.get("aggregator_data")
    if isinstance(aggregator_data, Mapping):
        for nested_key in ("currentRoleplay", "current_roleplay"):
            nested = aggregator_data.get(nested_key)
            if not isinstance(nested, Mapping):
                continue
            candidate = (
                nested.get("CurrentSnapshot")
                or nested.get("currentSnapshot")
                or nested.get("current_snapshot")
            )
            if isinstance(candidate, Mapping):
                snapshot_candidates.append(candidate)

    snapshot: Dict[str, Any] = {}
    for candidate in snapshot_candidates:
        snapshot = dict(candidate)

    current_location = current_context.get("current_location") or nyx_context.current_location
    location_fallback = None
    if isinstance(current_location, Mapping):
        location_fallback = (
            current_location.get("location_name")
            or current_location.get("name")
            or current_location.get("location")
        )
    elif isinstance(current_location, str):
        location_fallback = current_location

    snapshot_location = snapshot.get("location_name") or snapshot.get("location")
    if (
        location_fallback
        and (snapshot_location is None or str(snapshot_location).strip().lower() in {"", "unknown", "n/a", "na"})
    ):
        snapshot["location_name"] = location_fallback

    return snapshot


def _extract_feasibility_caps(fast_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(fast_result, dict):
        return {}

    for key in ("capabilities", "caps"):
        candidate = fast_result.get(key)
        if isinstance(candidate, dict):
            return candidate

    setting_context = fast_result.get("setting_context")
    if isinstance(setting_context, dict):
        candidate = setting_context.get("capabilities")
        if isinstance(candidate, dict):
            return candidate

    return {}


def _build_movement_meta(
    origin_location: Any,
    destination_payload: Dict[str, Any],
    nyx_context: NyxContext,
    fast_result: Dict[str, Any],
) -> Dict[str, Any]:
    origin = _normalize_location_dict(origin_location)
    destination = _normalize_location_dict(destination_payload) or dict(origin)

    origin_lat = origin.get("latitude")
    origin_lon = origin.get("longitude")
    dest_lat = destination.get("latitude")
    dest_lon = destination.get("longitude")
    approx_distance = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)

    world_snapshot = _extract_world_snapshot(nyx_context)
    feasibility_caps = _extract_feasibility_caps(fast_result)
    soft_location_only = _is_soft_location_only_violation(fast_result)

    destination_same_as_origin = _locations_same_place(origin, destination)
    movement_kind = "within_location" if destination_same_as_origin else "new_location"

    ctx = getattr(nyx_context, "current_context", {}) or {}
    user_text = ctx.get("user_input") or getattr(nyx_context, "last_user_input", "") or ""
    distance_class = _classify_distance(approx_distance, origin, destination)

    travel_mode = _infer_travel_mode(str(user_text), approx_distance, distance_class)
    travel_minutes = _estimate_travel_duration_minutes(
        travel_mode,
        approx_distance,
        movement_kind=movement_kind,
    )

    require_travel_choice = (
        travel_mode is None
        and distance_class in {"city", "long"}
        and movement_kind != "within_location"
    )

    return {
        "origin": origin,
        "destination": destination,
        "approx_distance_km": approx_distance,
        "distance_class": distance_class,
        "world": world_snapshot,
        "feasibility": feasibility_caps,
        "soft_location_only": soft_location_only,
        "destination_same_as_origin": destination_same_as_origin,
        "movement_kind": movement_kind,
        "travel_mode": travel_mode,
        "travel_duration_min": travel_minutes,
        "require_travel_choice": require_travel_choice,
    }


def _movement_is_local(movement_meta: Dict[str, Any]) -> bool:
    distance = movement_meta.get("approx_distance_km")
    if isinstance(distance, (int, float)):
        return distance < _MOVEMENT_FAST_PATH_DISTANCE_THRESHOLD_KM

    origin = movement_meta.get("origin") or {}
    destination = movement_meta.get("destination") or {}
    return _locations_look_local(origin, destination)


async def _run_movement_transition_fast_path(
    nyx_context: NyxContext,
    packed_context: Optional[PackedContext],
    user_input: str,
    fast_result: Dict[str, Any],
    trace_id: str,
    start_time: float,
    *,
    movement_only: bool = False,
    time_left_fn: Optional[Callable[[], float]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute the streamlined movement transition pipeline."""

    logger.info(f"[{trace_id}] Movement-only intents detected; running transition fast path")

    nyx_context.current_context.setdefault("feasibility", fast_result)

    snapshot_store = getattr(context_module, "_SNAPSHOT_STORE", None)
    if snapshot_store is not None and not hasattr(snapshot_store, "get"):
        snapshot_store = None

    resolve_result: Any = None
    try:
        from nyx.location import router as location_router

        resolve_result = await location_router.resolve_place_or_travel(
            user_input,
            _context_payload_for_router(nyx_context),
            snapshot_store,
            str(nyx_context.user_id),
            str(nyx_context.conversation_id),
        )
    except Exception as exc:
        logger.warning(
            f"[{trace_id}] Movement router failed softly: {exc}",
            exc_info=True,
        )

    previous_location_id = nyx_context.current_context.get("location_id")
    existing_location = nyx_context.current_context.get("current_location")
    location_payload = _extract_location_payload(resolve_result)
    movement_meta = _build_movement_meta(
        existing_location,
        location_payload,
        nyx_context,
        fast_result,
    )

    destination_same = movement_meta.get("destination_same_as_origin")
    if destination_same:
        origin_name = movement_meta.get("origin", {}).get("name") or movement_meta.get("origin", {}).get("location_name")
        destination_name = movement_meta.get("destination", {}).get("name") or movement_meta.get("destination", {}).get("location_name")
        logger.info(
            f"[{trace_id}] [ROUTER] Same-place movement detected; treating as within-location "
            f"(origin={origin_name}, destination={destination_name})"
        )

    if not _movement_is_local(movement_meta):
        logger.info(
            f"[{trace_id}] Movement fast path skipped; distance classification={movement_meta.get('distance_class')} "
            f"approx_distance={movement_meta.get('approx_distance_km')}"
        )
        return None

    scene_bundle = _build_movement_scene_bundle(
        nyx_context,
        packed_context,
        location_payload,
        movement_meta=movement_meta,
    )

    prompt = _build_movement_transition_prompt(user_input, scene_bundle, movement_meta)
    agent_context = RunContextWrapper(nyx_context) if RunContextWrapper else nyx_context
    time_budget = None
    if callable(time_left_fn):
        try:
            time_budget = time_left_fn()
        except Exception:
            logger.debug(
                f"[{trace_id}] Failed to compute time budget for movement run",
                exc_info=True,
            )
    run_config, movement_limits = _build_run_config_with_limits(
        base_kwargs={
            "workflow_name": "Nyx Movement Transition",
            "model_settings": _MOVEMENT_FAST_PATH_MODEL_SETTINGS,
        },
        limit_template=_MOVEMENT_RUN_LIMITS,
        time_budget=time_budget,
        trace_id=trace_id,
        log_label="movement_fast_path",
    )

    try:
        agent_result = await run_agent_safely(
            movement_transition_agent,
            prompt,
            agent_context,
            run_config=run_config,
        )
    except Exception as exc:
        logger.warning(
            f"[{trace_id}] Movement transition agent failed softly: {exc}",
            exc_info=True,
        )
        agent_result = None

    narration = coalesce_agent_output_text(agent_result)
    if not narration:
        narration = await _llm_fallback_movement_narration(movement_meta)

    require_travel_choice = movement_meta.get("require_travel_choice", False)
    needs_clarification = bool(require_travel_choice)
    if not needs_clarification:
        needs_clarification = _movement_requires_clarification(narration)

    if movement_only:
        mode = "movement_only"
    else:
        mode = "movement_only" if needs_clarification else "movement_and_scene"

    location_transition = {
        "router_called": resolve_result is not None,
        "router_status": getattr(resolve_result, "status", None) if resolve_result else None,
        "router_choices": getattr(resolve_result, "choices", []),
        "router_operations": getattr(resolve_result, "operations", []),
        "router_metadata": getattr(resolve_result, "metadata", {}) if resolve_result else {},
        "location": scene_bundle.get("location", {}),
    }

    if needs_clarification:
        return {
            "success": True,
            "response": narration,
            "narration": narration,
            "mode": "movement_only",
            "metadata": {
                "movement_fast_path": True,
                "movement_mode": "movement_only",
                "movement_only_intent": movement_only,
                "needs_travel_mode_choice": True,
                "universal_updates": False,
                "feasibility": fast_result,
                "location_transition": location_transition,
                "scene_bundle": scene_bundle,
                "movement_run_limits": movement_limits,
                "movement_meta": movement_meta,
            },
            "trace_id": trace_id,
            "processing_time": time.time() - start_time,
        }

    if location_payload:
        merged = dict(existing_location) if isinstance(existing_location, dict) else {}
        merged.update(location_payload)
        name = (
            merged.get("name")
            or merged.get("location_name")
            or merged.get("location")
        )
        if name:
            merged["name"] = name
            nyx_context.current_context["location_name"] = name
        location_id = (
            merged.get("id")
            or merged.get("location_id")
        )
        if location_id is not None:
            nyx_context.current_context["location_id"] = location_id
        nyx_context.current_context["current_location"] = merged
        _preserve_hydrated_location(nyx_context.current_context, merged)

        if not destination_same:
            try:
                await nyx_context._refresh_location_from_context(previous_location_id)
            except Exception:
                logger.debug(
                    f"[{trace_id}] Movement fast path location refresh failed softly",
                    exc_info=True,
                )

    _apply_travel_time_to_world(nyx_context, movement_meta)

    return {
        "success": True,
        "response": narration,
        "narration": narration,
        "mode": mode,
        "metadata": {
            "movement_fast_path": True,
            "movement_mode": mode,
            "movement_only_intent": movement_only,
            "needs_travel_mode_choice": needs_clarification,
            "universal_updates": False,
            "feasibility": fast_result,
            "location_transition": location_transition,
            "scene_bundle": scene_bundle,
            "movement_run_limits": movement_limits,
            "movement_meta": movement_meta,
        },
        "trace_id": trace_id,
        "processing_time": time.time() - start_time,
    }

# ===== Logging Helper =====
@asynccontextmanager
async def _log_step(name: str, trace_id: str, **meta):
    """Async context manager for logging step execution"""
    t0 = time.time()
    logger.debug(f"[{trace_id}] ▶ START {name} meta={_js(meta)}")
    with trace_step(f"nyx.orchestrator.{name}", trace_id, **meta):
        try:
            yield
            dt = time.time() - t0
            logger.info(f"[{trace_id}] ✔ DONE  {name} in {dt:.3f}s")
        except Exception:
            dt = time.time() - t0
            logger.exception(f"[{trace_id}] ✖ FAIL  {name} after {dt:.3f}s meta={_js(meta)}")
            raise


# ===== Error Handling =====
async def run_agent_safely(
    agent: Agent,
    input_data: Any,
    context: Any,
    run_config: Optional[RunConfig] = None,
    fallback_response: Any = None
) -> Any:
    """Run agent with automatic fallback on strict schema errors"""
    try:
        # First attempt with the agent as-is
        base_runner_kwargs: Dict[str, Any] = {}
        if run_config is not None:
            base_runner_kwargs["run_config"] = run_config
        request = LLMRequest(
            prompt=input_data,
            agent=agent,
            context=context,
            metadata={"operation": LLMOperation.ORCHESTRATION.value},
            runner_kwargs=base_runner_kwargs,
        )
        result_wrapper = await _execute_llm(request)
        return result_wrapper.raw
    except Exception as e:
        error_msg = str(e).lower()
        if "additionalproperties" in error_msg or "strict schema" in error_msg:
            logger.warning(f"Strict schema error, attempting without structured output: {e}")

            # Create a simple text-only agent
            fallback_agent = Agent(
                name=f"{getattr(agent, 'name', 'Agent')} (Fallback)",
                instructions=getattr(agent, 'instructions', ''),
                model=getattr(agent, 'model', None),
                model_settings=DEFAULT_MODEL_SETTINGS,
            )


            try:
                fallback_runner_kwargs: Dict[str, Any] = {}
                if run_config is not None:
                    fallback_runner_kwargs["run_config"] = run_config
                fallback_request = LLMRequest(
                    prompt=input_data,
                    agent=fallback_agent,
                    context=context,
                    metadata={
                        "operation": LLMOperation.ORCHESTRATION.value,
                        "fallback": True,
                    },
                    runner_kwargs=fallback_runner_kwargs,
                )
                fallback_result = await _execute_llm(fallback_request)
                return fallback_result.raw
            except Exception as e2:
                logger.error(f"Fallback agent also failed: {e2}")
                if fallback_response is not None:
                    return fallback_response
                raise
        else:
            # Not a schema error, re-raise
            raise

async def run_agent_with_error_handling(
    agent: Agent,
    input_data: Any,
    context: NyxContext,
    output_type: Optional[type] = None,
    fallback_value: Any = None
) -> Any:
    """Legacy compatibility wrapper for running agents with error handling"""
    try:
        result = await run_agent_safely(
            agent,
            input_data,
            context,
            run_config=RunConfig(workflow_name=f"Nyx {getattr(agent, 'name', 'Agent')}"),
            fallback_response=fallback_value
        )
        if output_type:
            return result.final_output_as(output_type)
        return getattr(result, "final_output", None) or getattr(result, "output_text", None)
    except Exception as e:
        logger.error(f"Error running agent {getattr(agent, 'name', 'unknown')}: {e}")
        if fallback_value is not None:
            return fallback_value
        raise


async def decide_image_generation_standalone(ctx: NyxContext, scene_text: str) -> str:
    """Standalone image generation decision without tool context"""
    from nyx.nyx_agent.models import ImageGenerationDecision
    from nyx.nyx_agent.utils import _score_scene_text, _build_image_prompt
    
    # Ensure we have the actual NyxContext, not a wrapper
    if hasattr(ctx, 'context'):
        ctx = ctx.context
    
    score = _score_scene_text(scene_text)
    recent_images = ctx.current_context.get("recent_image_count", 0)
    threshold = 0.7 if recent_images > 3 else 0.6 if recent_images > 1 else 0.5

    should_generate = score > threshold
    image_prompt = _build_image_prompt(scene_text) if should_generate else None

    if should_generate:
        ctx.current_context["recent_image_count"] = recent_images + 1

    return ImageGenerationDecision(
        should_generate=should_generate,
        score=score,
        image_prompt=image_prompt,
        reasoning=f"Scene has visual impact score of {score:.2f} (threshold: {threshold:.2f})",
    ).model_dump_json()


# ===== Main Process Function with Enhanced Reality Enforcement =====
async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Process user input with non-blocking context initialization, background writes,
    and enhanced reality enforcement.
    """
    trace_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    monotonic = time.monotonic
    request_started = time.perf_counter()
    success = True
    failure_reason: Optional[str] = None
    record_queue_delay_from_context(context_data, queue="orchestrator")

    # Adopt a single absolute deadline from the SDK or create a reasonable one
    default_deadline_budget = max(8.0, min(get_defer_run_timeout_seconds(), 20.0))
    deadline = None
    if isinstance(context_data, dict):
        deadline = context_data.get("_deadline") or context_data.get("deadline")

    now = monotonic()
    try:
        deadline_value = float(deadline) if deadline is not None else None
    except (TypeError, ValueError):
        deadline_value = None

    if deadline_value is None:
        deadline = now + default_deadline_budget
    else:
        remaining = deadline_value - now
        if remaining < 4.0:
            logger.info(
                f"[{trace_id}] Provided deadline is too tight (remaining={remaining:.3f}s); "
                f"resetting to {default_deadline_budget:.1f}s budget",
            )
            deadline = now + default_deadline_budget
        else:
            deadline = deadline_value

    def time_left(floor: float = 0.5) -> float:
        remaining_budget = deadline - monotonic()
        if remaining_budget <= 0:
            return floor
        return remaining_budget

    nyx_context: Optional[NyxContext] = None
    packed_context: Optional[PackedContext] = None
    movement_prelude: Optional[str] = None
    enrichment_enabled = getattr(Config, "ENABLE_MOVEMENT_MAIN_AGENT_ENRICHMENT", True)

    logger.info(f"[{trace_id}] ========== PROCESS START ==========")
    logger.info(f"[{trace_id}] user_id={user_id} conversation_id={conversation_id}")
    logger.info(f"[{trace_id}] user_input={user_input[:200]}")

    fast: Optional[Dict[str, Any]] = None
    router_result: Optional[Dict[str, Any]] = None
    if isinstance(context_data, dict):
        feas_from_context = context_data.get("feasibility")
        if isinstance(feas_from_context, dict):
            fast = feas_from_context
        router_candidate = context_data.get("router_result")
        if isinstance(router_candidate, dict):
            router_result = router_candidate

    try:
        # ---- STEP 0: Mandatory fast feasibility (dynamic) ---------------------
        if fast is None:
            logger.info(f"[{trace_id}] Running mandatory fast feasibility check")
            try:
                from nyx.nyx_agent.feasibility import assess_action_feasibility_fast

                fast = await assess_action_feasibility_fast(user_id, conversation_id, user_input)
            except Exception as e:
                logger.error(f"[{trace_id}] Fast feasibility failed softly: {e}", exc_info=True)
        else:
            logger.info(
                f"[{trace_id}] Fast feasibility supplied via context; skipping duplicate check"
            )

        fast_overall: Dict[str, Any] = {}
        fast_strategy = ""
        fast_feasible_flag: Optional[bool] = None
        movement_fast_path = False
        movement_only_intents = False

        if isinstance(fast, dict):
            fast_overall = (fast or {}).get("overall", {}) or {}
            fast_strategy = (fast_overall.get("strategy") or "").lower()
            fast_feasible_flag = fast_overall.get("feasible")
            soft_location_only = _is_soft_location_only_violation(fast)
            fast_router = fast.get("router_result") if isinstance(fast.get("router_result"), dict) else None
            if fast_router is not None:
                router_result = fast_router

            logger.info(
                f"[{trace_id}] Fast feasibility: feasible={fast_feasible_flag} "
                f"strategy={fast_strategy} soft_location_only={soft_location_only}"
            )

            if fast_feasible_flag is True and fast_strategy == "allow":
                movement_only_intents = _intents_are_movement_only(fast)
                movement_fast_path = movement_only_intents
                logger.info(
                    f"[{trace_id}] Movement-only intents detected? {movement_only_intents}"
                )

            # Hard-block ONLY when it's a real impossibility, not a location-only miss
            if (
                fast_feasible_flag is False
                and fast_strategy == "deny"
                and not soft_location_only
            ):
                per = fast.get("per_intent") or []
                first = per[0] if per and isinstance(per[0], dict) else {}
                guidance = (
                    first.get("narrator_guidance")
                    or "That can't happen here. Try a grounded approach that fits the setting."
                )
                options = [{"text": o} for o in (first.get("suggested_alternatives") or [])]

                logger.warning(
                    f"[{trace_id}] ACTION BLOCKED (fast gate). Reason: {first.get('violations', [])}"
                )
                return {
                    "success": True,
                    "response": guidance,
                    "metadata": {
                        "choices": options[:4],
                        "universal_updates": False,
                        "feasibility": fast,
                        "action_blocked": True,
                        "block_reason": (first.get("violations") or [{}])[0].get(
                            "reason", "setting constraints"
                        ),
                        "reality_maintained": True,
                    },
                    "trace_id": trace_id,
                    "processing_time": time.time() - start_time,
                }

            # Soft location-only deny → let router + full pipeline handle it
            if (
                fast_feasible_flag is False
                and fast_strategy == "deny"
                and soft_location_only
            ):
                if not movement_fast_path:
                    movement_only_intents = _intents_are_movement_only(fast)
                    movement_fast_path = movement_only_intents
                if movement_fast_path:
                    logger.info(
                        f"[{trace_id}] Soft location-only violation with movement intents; enabling fast path."
                    )
                logger.info(
                    f"[{trace_id}] Fast feasibility DENY is soft location-only; "
                    "continuing to orchestrator + location resolver."
                )

        # ---- STEP 1: Context initialization -----------------------------------
        async with _log_step("context_init", trace_id):
            nyx_context = NyxContext(user_id, conversation_id)
            await nyx_context.initialize()

            base_context = _normalize_scene_context(context_data or {})
            base_context["_deadline"] = deadline
            base_context["user_input"] = user_input
            base_context["last_user_input"] = user_input
            nyx_context.last_user_input = user_input
            nyx_context.current_context = base_context
            _preserve_hydrated_location(nyx_context.current_context, nyx_context.current_location)

            packed_context = await nyx_context.build_context_for_input(
                user_input,
                base_context,
            )
            nyx_context.last_packed_context = packed_context
            if packed_context is not None and hasattr(packed_context, "to_dict"):
                try:
                    packed_keys = list(packed_context.to_dict().keys())
                    logger.debug(
                        f"[{trace_id}] packed_context keys={packed_keys}"
                    )
                except Exception:
                    logger.debug(
                        f"[{trace_id}] Failed to log packed_context keys", exc_info=True
                    )

        # ---- STEP 2: World state integration ----------------------------------
        async with _log_step("world_state", trace_id):
            # Await the 'world' subsystem before using it
            if nyx_context and await nyx_context.await_orchestrator("world"):
                orchestrator = getattr(nyx_context, "world_orchestrator", None)
                if orchestrator is not None:
                    cached_state = orchestrator.get_cached_state()
                    if cached_state is None:
                        cached_state = await orchestrator.get_world_state()
                    if cached_state is not None:
                        nyx_context.current_world_state = cached_state
                elif nyx_context.world_director and getattr(
                    nyx_context.world_director, "context", None
                ):
                    nyx_context.current_world_state = (
                        nyx_context.world_director.context.current_world_state
                    )

        if movement_fast_path and nyx_context is not None:
            nyx_context.current_context["feasibility"] = fast if isinstance(fast, dict) else {}
            fast_path_response = await _run_movement_transition_fast_path(
                nyx_context,
                packed_context,
                user_input,
                fast if isinstance(fast, dict) else {},
                trace_id,
                start_time,
                movement_only=movement_only_intents,
                time_left_fn=time_left,
            )
            if fast_path_response is not None:
                metadata = fast_path_response.get("metadata") or {}
                mode = fast_path_response.get("mode") or metadata.get("movement_mode")
                narration = (
                    fast_path_response.get("narration")
                    or fast_path_response.get("response")
                    or ""
                )
                needs_clarification = bool(metadata.get("needs_travel_mode_choice"))

                if needs_clarification:
                    logger.info(
                        f"[{trace_id}] Movement fast path produced clarifying question; returning without main agent."
                    )
                    return fast_path_response

                if mode == "movement_only":
                    if not enrichment_enabled or movement_only_intents:
                        logger.info(
                            f"[{trace_id}] Movement fast path satisfied turn; returning movement-only response."
                        )
                        return fast_path_response
                    movement_prelude = narration
                    logger.info(
                        f"[{trace_id}] Movement fast path produced prelude; main agent enrichment enabled, continuing."
                    )
                elif mode == "movement_and_scene":
                    movement_prelude = narration

            if movement_prelude:
                logger.info(f"[{trace_id}] Movement fast path produced arrival prelude; continuing with main agent.")
            else:
                logger.info(
                    f"[{trace_id}] Movement fast path declined; continuing with full orchestration",
                )

        # ---- STEP 3: Full feasibility (dynamic) --------------------------------
        logger.info(f"[{trace_id}] Running full feasibility assessment")
        feas: Optional[Dict[str, Any]] = None
        enhanced_input = user_input  # Initialize with original input

        assess_action_feasibility_fn = None
        record_impossibility_fn = None
        record_possibility_fn = None

        try:
            from nyx.nyx_agent.feasibility import (
                assess_action_feasibility as assess_action_feasibility_fn,
                record_impossibility as record_impossibility_fn,
                record_possibility as record_possibility_fn,
            )
        except ImportError:
            logger.warning(
                f"[{trace_id}] Full feasibility not available; proceeding without it."
            )
        except Exception as e:
            logger.warning(
                f"[{trace_id}] Full feasibility import failed softly: {e}", exc_info=True
            )

        fast_allows_action = fast_feasible_flag is True and fast_strategy == "allow"

        if assess_action_feasibility_fn:
            try:
                if fast_allows_action and isinstance(fast, dict):
                    feas = fast
                    nyx_context.current_context["feasibility"] = fast
                    logger.info(
                        f"[{trace_id}] Fast feasibility allowed action; skipping full feasibility assessment"
                    )
                else:
                    feas = await assess_action_feasibility_fn(
                        nyx_context,
                        user_input,
                        router_result=router_result,
                    )
                    nyx_context.current_context["feasibility"] = feas
                    logger.info(
                        f"[{trace_id}] Full feasibility: {feas.get('overall', {})}"
                    )
            except Exception as e:
                logger.warning(
                    f"[{trace_id}] Full feasibility failed softly: {e}", exc_info=True
                )

        if isinstance(feas, dict):
            overall = feas.get("overall", {}) or {}
            feasible_flag = overall.get("feasible")
            strategy = (overall.get("strategy") or "").lower()

            if feasible_flag is False and strategy == "deny":
                per = feas.get("per_intent") or []
                first = per[0] if per and isinstance(per[0], dict) else {}
                violations = first.get("violations", [])
                violation_text = (
                    violations[0]["reason"]
                    if violations
                    else "That violates the laws of this reality"
                )

                if record_impossibility_fn:
                    try:
                        await record_impossibility_fn(
                            nyx_context, user_input, violation_text
                        )
                    except Exception:
                        logger.debug(
                            f"[{trace_id}] record_impossibility failed softly",
                            exc_info=True,
                        )

                rejection_narrative = (
                    f"*{first.get('reality_response', 'Reality ripples and refuses.')}*\n\n"
                )
                rejection_narrative += first.get(
                    "narrator_guidance", "The world itself resists your attempt."
                )
                alternatives = first.get("suggested_alternatives", [])
                if alternatives:
                    rejection_narrative += (
                        f"\n\n*Perhaps you could {alternatives[0]} instead.*"
                    )
                choices = [
                    {
                        "text": alt,
                        "description": "A possible action within this reality",
                        "feasible": True,
                    }
                    for alt in alternatives[:4]
                ]

                return {
                    "success": True,
                    "response": rejection_narrative,
                    "metadata": {
                        "choices": choices,
                        "universal_updates": False,
                        "feasibility": feas,
                        "action_blocked": True,
                        "block_reason": violation_text,
                        "reality_maintained": True,
                    },
                    "trace_id": trace_id,
                    "processing_time": time.time() - start_time,
                }
            elif feasible_flag is False and strategy == "ask":
                constraints = feas.get("per_intent", [{}])[0].get("violations", [])
                constraint_text = (
                    "[REALITY CHECK: This action pushes boundaries. Consider: "
                    + ", ".join(v.get("reason", "") for v in constraints)
                    + ". Describe attempt with appropriate limitations.]"
                )
                enhanced_input = f"{constraint_text}\n\n{user_input}"
            elif feasible_flag is False and strategy == "defer":
                defer_context, extra_meta = extract_defer_details(feas)
                leads = extra_meta.get("leads", [])
                guidance = None
                if defer_context:
                    guidance = await _generate_defer_taunt(defer_context, trace_id, nyx_context)
                if not guidance:
                    guidance = (
                        build_defer_fallback_text(defer_context)
                        if defer_context
                        else "Oh, pet, slow down. Reality keeps its heel on you until you ground that attempt."
                    )

                logger.info(f"[{trace_id}] ACTION DEFERRED (full feasibility)")
                metadata = {
                    "choices": [{"text": lead} for lead in leads[:4]],
                    "universal_updates": False,
                    "feasibility": feas,
                    "action_blocked": True,
                    "action_deferred": True,
                    "reality_maintained": True,
                }
                metadata.update(extra_meta)
                return {
                    "success": True,
                    "response": guidance,
                    "metadata": metadata,
                    "trace_id": trace_id,
                    "processing_time": time.time() - start_time,
                }
            elif feasible_flag is True:
                if record_possibility_fn:
                    try:
                        intents = feas.get("per_intent", [])
                        if intents and (cats := intents[0].get("categories")):
                            await record_possibility_fn(
                                nyx_context, user_input, cats
                            )
                    except Exception:
                        logger.debug(
                            f"[{trace_id}] record_possibility failed softly",
                            exc_info=True,
                        )
                enhanced_input = (
                    "[REALITY CHECK: Action is feasible within universe laws.]\n\n"
                    f"{user_input}"
                )

        if movement_prelude:
            logger.info(f"[{trace_id}] Movement prelude injected into enhanced_input.")
            enhanced_input = f"{movement_prelude.strip()}\n\n{enhanced_input}"

        # ---- STEP 4: Tool sanitization ----------------------------------------
        async with _log_step("tool_sanitization", trace_id):
            sanitize_agent_tools_in_place(nyx_main_agent)
            log_strict_hits(nyx_main_agent)

        # ---- STEP 5: Run main agent with enhanced input ----------------------
        async with _log_step("agent_run", trace_id):
            runner_context = RunContextWrapper(nyx_context)
            safe_settings = ModelSettings(strict_tools=False)

            agent_model = getattr(nyx_main_agent, "model", None)
            model_label = _resolve_model_label(agent_model)
            tool_names: List[str] = []
            for tool in getattr(nyx_main_agent, "tools", None) or []:
                name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
                if not name and hasattr(tool, "func"):
                    name = getattr(tool.func, "__name__", None)
                tool_names.append(name or str(tool))

            context_stats: Dict[str, Any] = {}
            if packed_context is not None:
                context_stats = {
                    "token_budget": getattr(packed_context, "token_budget", None),
                    "tokens_used": getattr(packed_context, "tokens_used", None),
                    "canonical": len(getattr(packed_context, "canonical", {}) or {}),
                    "optional": len(getattr(packed_context, "optional", {}) or {}),
                    "summarized": len(getattr(packed_context, "summarized", {}) or {}),
                }

            llm_timeout = time_left()
            run_config, run_limits = _build_run_config_with_limits(
                base_kwargs={
                    "model_settings": safe_settings,
                    "workflow_name": "Nyx Main Orchestrator",
                },
                limit_template=_MAIN_AGENT_RUN_LIMITS,
                time_budget=llm_timeout,
                trace_id=trace_id,
                log_label="agent_run",
            )
            logger.info(
                f"[{trace_id}] [agent_run] starting model={model_label} tools={tool_names} "
                f"ctx_stats={context_stats} user_input={user_input[:80]!r} "
                f"enhanced_input={enhanced_input[:80]!r} timeout={llm_timeout:.2f}s"
            )

            request = LLMRequest(
                prompt=enhanced_input,
                agent=nyx_main_agent,
                context=runner_context,
                runner_kwargs={"run_config": run_config},
                metadata={
                    "trace_id": trace_id,
                    "operation": LLMOperation.ORCHESTRATION.value,
                    "stream": False,
                    "tags": ["nyx", "orchestrator", "main"],
                    "run_limits": run_limits,
                },
            )
            agent_start = time.monotonic()
            try:
                result = await _execute_llm(request)
            except Exception:
                elapsed = time.monotonic() - agent_start
                logger.exception(
                    f"[{trace_id}] [agent_run] failed elapsed={elapsed:.2f}s timeout={llm_timeout:.2f}s"
                )
                raise

            elapsed = time.monotonic() - agent_start
            logger.info(
                f"[{trace_id}] [agent_run] success elapsed={elapsed:.2f}s timeout={llm_timeout:.2f}s"
            )

            resp_stream = extract_runner_response(result.raw)

        post_run_narrative: str = ""
        defer_universal_updates = False
        defer_punishment = False

        # ---- STEP 6: Post-run enforcement (updates/image hooks + punishment) ---
        async with _log_step("post_run_enforcement", trace_id):
            post_run_narrative = _extract_last_assistant_text(resp_stream)

            if not _did_call_tool(resp_stream, "generate_universal_updates"):
                defer_universal_updates = True
                logger.info(
                    f"[{trace_id}] Deferring universal updates to background task"
                )

            if (
                time_left() >= 2.0
                and not _did_call_tool(resp_stream, "decide_image_generation")
                and post_run_narrative
                and len(post_run_narrative) > 20
            ):
                try:
                    image_result = await decide_image_generation_standalone(
                        nyx_context, post_run_narrative
                    )
                    resp_stream.append(
                        {
                            "type": "function_call_output",
                            "name": "decide_image_generation",
                            "output": image_result,
                        }
                    )
                except Exception as e:
                    logger.debug(
                        f"[{trace_id}] Post-run image decision failed softly: {e}"
                    )
            elif time_left() < 2.0:
                logger.info(
                    f"[{trace_id}] Skipping image decision due to low remaining budget"
                )

            if enforce_all_rules_on_player:
                defer_punishment = True
                logger.info(
                    f"[{trace_id}] Deferring punishment enforcement to background task"
                )
            else:
                logger.debug(
                    f"[{trace_id}] punishment module unavailable; skipping enforcement"
                )

        # ---- STEP 7: Response assembly (feasibility-aware) ---------------------
        async with _log_step("response_assembly", trace_id):
            resp_stream = await resolve_scene_requests(resp_stream, nyx_context)
            assembled = await assemble_nyx_response(
                agent_output=resp_stream,
                processing_metadata={
                    "feasibility": (feas or fast),
                    "punishment": nyx_context.current_context.get("punishment"),
                },
                user_input=user_input,
                conversation_id=str(conversation_id),
                nyx_context=nyx_context,
            )

            final_narrative = assembled.narrative or post_run_narrative

            out = {
                "success": True,
                "response": assembled.narrative,
                "metadata": {
                    "world": getattr(assembled, "world_state", {}),
                    "choices": getattr(assembled, "choices", []),
                    "emergent": getattr(assembled, "emergent_events", []),
                    "image": getattr(assembled, "image", None),
                    "telemetry": (assembled.metadata or {}).get("performance", {}),
                    "nyx_commentary": (assembled.metadata or {}).get("nyx_commentary"),
                    "universal_updates": (assembled.metadata or {}).get(
                        "universal_updates", False
                    ),
                    "reality_maintained": True,
                    "punishment": nyx_context.current_context.get("punishment"),
                },
                "trace_id": trace_id,
                "processing_time": time.time() - start_time,
            }

            async def perform_background_writes() -> None:
                try:
                    # Save the final context state (emotional, scene, etc.)
                    await _save_context_state(nyx_context)

                    if defer_universal_updates and final_narrative:
                        try:
                            update_result = await generate_universal_updates_impl(
                                nyx_context, final_narrative
                            )
                            tool_payload = {
                                "type": "function_call_output",
                                "name": "generate_universal_updates",
                                "output": json.dumps(
                                    {
                                        "success": getattr(update_result, "success", True),
                                        "updates_generated": getattr(
                                            update_result, "updates_generated", False
                                        ),
                                        "source": "background_post_run",
                                    }
                                ),
                            }
                            resp_stream.append(tool_payload)
                            nyx_context.current_context.setdefault(
                                "deferred_tool_outputs", []
                            ).append(tool_payload)
                        except Exception as e:
                            logger.debug(
                                f"[{trace_id}] Deferred universal updates failed softly: {e}"
                            )

                    if defer_punishment and enforce_all_rules_on_player:
                        try:
                            player_name = (
                                nyx_context.current_context.get("player_name") or "Chase"
                            )
                            punishment_meta = {
                                "scene_tags": nyx_context.current_context.get(
                                    "scene_tags", []
                                ),
                                "stimuli": nyx_context.current_context.get("stimuli", []),
                                "feasibility": (feas or fast),
                                "turn_index": nyx_context.current_context.get(
                                    "turn_index", 0
                                ),
                            }
                            punishment_result = await enforce_all_rules_on_player(
                                player_name=player_name,
                                user_id=user_id,
                                conversation_id=conversation_id,
                                metadata=punishment_meta,
                            )
                            punishment_payload = {
                                "type": "function_call_output",
                                "name": "enforce_punishments",
                                "output": json.dumps(punishment_result),
                            }
                            resp_stream.append(punishment_payload)
                            nyx_context.current_context.setdefault(
                                "deferred_tool_outputs", []
                            ).append(punishment_payload)
                            nyx_context.current_context["punishment"] = punishment_result
                        except Exception as e:
                            logger.debug(
                                f"[{trace_id}] Deferred punishment enforcement failed softly: {e}"
                            )

                    # Persist any universal updates (like location changes)
                    if nyx_context and nyx_context.context_broker:
                        bundle = getattr(
                            nyx_context.context_broker, "_last_bundle", None
                        )
                        if bundle:
                            pending_updates = (
                                await nyx_context.context_broker.collect_universal_updates(
                                    bundle
                                )
                            )
                            if pending_updates:
                                await nyx_context.context_broker.apply_updates_async(
                                    pending_updates
                                )

                    logger.info(f"[{trace_id}] ✔ Background writes completed.")
                except Exception as e:
                    logger.error(
                        f"[{trace_id}] ✖ Background write task failed: {e}",
                        exc_info=True,
                    )

            if nyx_context:
                task = asyncio.create_task(perform_background_writes())
                nyx_context._track_background_task(
                    task, task_name="post_turn_writes"
                )

            logger.info(f"[{trace_id}] ========== PROCESS COMPLETE (Response Sent) ==========")
            logger.info(f"[{trace_id}] Response length: {len(assembled.narrative or '')}")
            logger.info(f"[{trace_id}] Processing time: {out['processing_time']:.2f}s")

            return out

    except asyncio.CancelledError:
        success = False
        failure_reason = "cancelled"
        logger.info(f"[{trace_id}] ========== PROCESS CANCELLED ==========")
        raise
    except Exception as e:
        success = False
        failure_reason = type(e).__name__
        logger.error(f"[{trace_id}] ========== PROCESS FAILED ==========", exc_info=True)
        return {
            "success": False,
            "response": "I encountered an error processing your request. Please try again.",
            "error": str(e),
            "trace_id": trace_id,
            "processing_time": time.time() - start_time,
        }
    finally:
        duration = time.perf_counter() - request_started
        REQUEST_LATENCY.labels(component="orchestrator").observe(duration)
        if not success:
            TASK_FAILURES.labels(
                task="nyx_orchestrator", reason=failure_reason or "unknown"
            ).inc()


# ===== State Management =====
async def _save_context_state(ctx: NyxContext):
    """Save context state to database"""
    async with get_db_connection_context() as conn:
        try:
            normalized_context = _normalize_scene_context(getattr(ctx, "current_context", {}))
            if isinstance(getattr(ctx, "current_context", None), dict):
                ctx.current_context.clear()
                ctx.current_context.update(normalized_context)
            else:
                ctx.current_context = normalized_context

            # Get emotional state from current_context or provide default
            emotional_state = ctx.current_context.get('emotional_state', {
                'valence': 0.0,
                'arousal': 0.5,
                'dominance': 0.7
            })

            # Save emotional state
            await conn.execute("""
                INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
            """, ctx.user_id, ctx.conversation_id, json.dumps(emotional_state, ensure_ascii=False))
            
            # Save current scene state for future feasibility checks
            scene_state = {
                "location": ctx.current_context.get("current_location"),
                "items": ctx.current_context.get("available_items", []),
                "npcs": ctx.current_context.get("present_npcs", [])
            }
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'CurrentScene', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, ctx.user_id, ctx.conversation_id, json.dumps(scene_state))
            
            # Save scenario state if active
            if ctx.scenario_state and ctx.scenario_state.get("active") and ctx._tables_available.get("scenario_states", True):
                should_save_heartbeat = ctx.should_run_task("scenario_heartbeat")
                
                try:
                    if should_save_heartbeat:
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                        
                        ctx.record_task_run("scenario_heartbeat")
                    else:
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                            ON CONFLICT (user_id, conversation_id) 
                            DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                except Exception as e:
                    if "does not exist" in str(e) or "no such table" in str(e).lower():
                        ctx._tables_available["scenario_states"] = False
                        logger.warning("scenario_states table not available - skipping save")
                    else:
                        raise
            
            # Save learning metrics periodically
            if ctx.should_run_task("learning_save"):
                await conn.execute("""
                    INSERT INTO learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id, 
                json.dumps(ctx.learning_metrics, ensure_ascii=False), 
                json.dumps(dict(list(ctx.learned_patterns.items())[-Config.MAX_LEARNED_PATTERNS:]), ensure_ascii=False))
                
                ctx.record_task_run("learning_save")
            
            # Save performance metrics periodically
            if ctx.should_run_task("performance_save"):
                bounded_metrics = ctx.performance_metrics.copy()
                if "response_times" in bounded_metrics:
                    bounded_metrics["response_times"] = bounded_metrics["response_times"][-Config.MAX_RESPONSE_TIMES:]
                
                await conn.execute("""
                    INSERT INTO performance_metrics (user_id, conversation_id, metrics, error_log, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id,
                json.dumps(bounded_metrics, ensure_ascii=False),
                json.dumps(ctx.error_log[-Config.MAX_ERROR_LOG_ENTRIES:], ensure_ascii=False))
                
                ctx.record_task_run("performance_save")
                
        except Exception as e:
            logger.error(f"Error saving context state: {e}")

# ===== High-Level Operations =====
async def generate_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a reflection from Nyx on a specific topic"""
    try:
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()

        prompt = f"Create a reflection about: {topic}" if topic else \
                 "Create a reflection about the user based on your memories"

        result = await run_agent_safely(
            reflection_agent,
            prompt,
            context=nyx_context,
            run_config=RunConfig(workflow_name="Nyx Reflection"),
        )

        reflection = result.final_output_as(MemoryReflection)
        return {
            "reflection": reflection.reflection,
            "confidence": reflection.confidence,
            "topic": reflection.topic or topic,
        }
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        return {"reflection": "Unable to generate reflection at this time.", "confidence": 0.0, "topic": topic}

async def manage_scenario(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """DEPRECATED - Replace with emergent scenario management"""
    try:
        user_id = scenario_data.get("user_id")
        conversation_id = scenario_data.get("conversation_id")

        from story_agent.world_director_agent import CompleteWorldDirector

        director = CompleteWorldDirector(user_id, conversation_id)
        await director.initialize()

        next_moment = await director.generate_next_moment()

        return {
            "success": True,
            "emergent_scenario": next_moment.get("moment"),
            "world_state": next_moment.get("world_state"),
            "patterns": next_moment.get("patterns"),
            "linear_progression": None
        }
    except Exception as e:
        logger.error(f"Error managing scenario: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def manage_relationships(interaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Manage and update relationships between entities."""
    nyx_context = None
    
    try:
        user_id = interaction_data.get("user_id")
        conversation_id = interaction_data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise ValueError("interaction_data must include user_id and conversation_id")
        
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        participants = interaction_data.get("participants", [])
        relationship_updates = {}
        
        for i, p1 in enumerate(participants):
            for p2 in participants[i+1:]:
                from .models import kvlist_to_dict
                p1_dict = kvlist_to_dict(p1) if not isinstance(p1, dict) else p1
                p2_dict = kvlist_to_dict(p2) if not isinstance(p2, dict) else p2
                
                entity_key = "_".join(sorted([str(p1_dict.get('id', p1)), str(p2_dict.get('id', p2))]))
                
                trust_change = 0.1 if interaction_data.get("outcome") == "success" else -0.05
                bond_change = 0.05 if interaction_data.get("emotional_impact", {}).get("positive", 0) > 0 else 0
                power_change = 0.0
                 
                if interaction_data.get("interaction_type") == "training":
                    power_change = 0.05
                elif interaction_data.get("interaction_type") == "conflict":
                    power_change = -0.05
                
                result = await update_relationship_state(
                    RunContextWrapper(context=nyx_context),
                    UpdateRelationshipStateInput(
                        entity_id=entity_key,
                        trust_change=trust_change,
                        power_change=power_change,
                        bond_change=bond_change
                    )
                )
                
                relationship_updates[entity_key] = json.loads(result)
        
        logger.warning("interaction_history table not found in schema - skipping interaction storage")
        
        for pair, updates in relationship_updates.items():
            await nyx_context.learn_from_interaction(
                action=f"relationship_{interaction_data.get('interaction_type', 'general')}",
                outcome=interaction_data.get("outcome", "unknown"),
                success=updates.get("changes", {}).get("trust", 0) > 0
            )
        
        return {
            "success": True,
            "relationship_updates": relationship_updates,
            "analysis": {
                "total_relationships_updated": len(relationship_updates),
                "interaction_type": interaction_data.get("interaction_type"),
                "outcome": interaction_data.get("outcome"),
                "stored_in_history": False
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing relationships: {e}")
        if nyx_context:
            nyx_context.log_error(e, interaction_data)
        return {
            "success": False,
            "error": str(e)
        }

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store user and Nyx messages in database"""
    async with get_db_connection_context() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "user", user_input
        )
        
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "Nyx", nyx_response
        )
