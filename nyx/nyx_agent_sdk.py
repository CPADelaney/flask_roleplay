# nyx/nyx_agent_sdk.py
"""
Nyx Agent SDK — durable 'final stop' over the modern nyx.nyx_agent stack.

Capabilities:
- Primary path: nyx.nyx_agent.orchestrator.process_user_input
- Resilience: fallback to direct agent run if orchestrator fails
- Bidirectional moderation: optional pre-input and post-output guardrails
- Streaming: best-effort chunking for responsive UIs
- Cache warmup: pre-load scene bundles for faster first responses
- Hooks: telemetry + background task systems (no-op if not installed)
- Extras: per-conversation concurrency guard, short idempotency cache,
          timeouts + single retry, optional response filtering & post hooks
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple

# ── Core modern orchestrator
from .nyx_agent.orchestrator import (
    process_user_input as _orchestrator_process,
    _preserve_hydrated_location,
)
from .nyx_agent import orchestrator as _orchestrator
from .nyx_agent.context import (
    NyxContext,
    SceneScope,
    build_canonical_snapshot_payload,
    persist_canonical_snapshot,
)
from .nyx_agent._feasibility_helpers import (
    DeferPromptContext,
    build_defer_fallback_text,
    build_defer_prompt,
    coalesce_agent_output_text,
    extract_defer_details,
)

from nyx.nyx_agent.models import NyxResponse
from nyx.conversation.snapshot_store import ConversationSnapshotStore

try:
    from nyx.conversation import ConversationStore
except Exception:  # pragma: no cover - store is optional in some runtimes
    ConversationStore = None  # type: ignore
from nyx.core.side_effects import (
    ConflictEvent,
    LoreHint,
    MemoryEvent,
    NPCStimulus,
    SideEffect,
    WorldDelta,
    group_side_effects,
)
from nyx.gateway.llm_gateway import execute, execute_stream, LLMRequest, LLMOperation
from nyx.telemetry.metrics import (
    TASK_FAILURES,
    TTFB_SECONDS,
    record_queue_delay_from_context,
)
from nyx.telemetry.tracing import trace_step
from nyx.config import flags

try:
    import openai
except Exception:  # pragma: no cover
    openai = None  # type: ignore


def _agents_trace(workflow_name: str):
    """Return an OpenAI Agents trace context if available."""

    if openai is None:
        return nullcontext()

    try:
        return openai.agents.trace(workflow_name)
    except Exception:
        return nullcontext()


try:  # pragma: no cover - Celery is optional in some environments
    from nyx.tasks.realtime.post_turn import dispatch as post_turn_dispatch
except Exception:  # pragma: no cover
    post_turn_dispatch = None  # type: ignore

# Fallback agent runtime (loaded lazily; optional in some environments)
try:
    from agents import RunConfig, ModelSettings, RunContextWrapper
    from .nyx_agent.agents import nyx_main_agent, nyx_defer_agent, DEFAULT_MODEL_SETTINGS
except Exception:  # pragma: no cover
    RunConfig = None
    ModelSettings = None
    RunContextWrapper = None
    nyx_main_agent = None
    nyx_defer_agent = None
    DEFAULT_MODEL_SETTINGS = None

# Core feasibility + canon tracking helpers (best-effort import)
try:
    from .nyx_agent.feasibility import (
        assess_action_feasibility,
        record_impossibility,
        record_possibility,
    )
except Exception:  # pragma: no cover
    assess_action_feasibility = None  # type: ignore
    record_impossibility = None  # type: ignore
    record_possibility = None  # type: ignore

# Guardrails (optional)
try:
    from .nyx_agent.guardrails import content_moderation_guardrail
except Exception:  # pragma: no cover
    content_moderation_guardrail = None

# Response filter (optional; tone/policy scrubbing)
try:
    from nyx.response_filter import ResponseFilter  # preferred (matches context.py)
except Exception:  # pragma: no cover
    try:
        from .response_filter import ResponseFilter  # local fallback
    except Exception:  # pragma: no cover
        ResponseFilter = None  # type: ignore

# Optional telemetry / background queue (auto-noop if absent)
try:
    from .utils.performance import log_performance_metrics as _log_perf
except Exception:  # pragma: no cover
    _log_perf = None

try:
    from .utils.background import enqueue_task as _enqueue_task
except Exception:  # pragma: no cover
    _enqueue_task = None

logger = logging.getLogger(__name__)

# ---- Location normalization helpers (gateway-level) -------------------------

_LOCATION_KEYS = (
    "CurrentLocation", "current_location", "currentLocation", "location", "Location",
)
_LOCATION_ID_KEYS = (
    "CurrentLocationId", "current_location_id", "currentLocationId", "location_id", "LocationId",
)

def _slugify_location(value: str) -> str:
    if not isinstance(value, str):
        return "unknown"
    s = value.strip().lower()
    out, prev_dash = [], False
    for ch in s:
        if ch.isalnum():
            out.append(ch); prev_dash = False
        else:
            if not prev_dash:
                out.append("-"); prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "unknown"

def _extract_location_from_mapping(mapping: Dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
    """Read display + id from a loose dict with many legacy key options."""
    if not isinstance(mapping, dict):
        return None, None

    display = None
    for k in _LOCATION_KEYS:
        v = mapping.get(k)
        if isinstance(v, str) and v.strip():
            display = v.strip()
            break
        if k == "location" and isinstance(v, dict):
            name = v.get("name") or v.get("label") or v.get("slug") or v.get("location")
            if isinstance(name, str) and name.strip():
                display = name.strip()
                break

    loc_id = None
    for k in _LOCATION_ID_KEYS:
        v = mapping.get(k)
        try:
            if v is not None:
                loc_id = int(v)
                if loc_id <= 0:
                    loc_id = None
        except (TypeError, ValueError):
            pass
        if loc_id is not None:
            break

    if loc_id is None and isinstance(mapping.get("location"), dict):
        v = mapping["location"].get("id") or mapping["location"].get("location_id") or mapping["location"].get("pk")
        try:
            if v is not None:
                loc_id = int(v)
                if loc_id <= 0:
                    loc_id = None
        except (TypeError, ValueError):
            loc_id = None

    return display, loc_id

def _normalize_location_meta_inplace(meta: Dict[str, Any]) -> None:
    """
    Ensure metadata has both a canonical `locationInfo` and a mirrored `scene_scope`.
    Does not overwrite existing scene_scope fields if they are already set.
    """
    if not isinstance(meta, dict):
        return

    # 1) Try canonical first
    li = meta.get("locationInfo") if isinstance(meta.get("locationInfo"), dict) else None
    display = None
    loc_id = None
    slug = None

    if li:
        display = li.get("display") or li.get("name") or li.get("label")
        if isinstance(display, str) and display.strip():
            display = display.strip()
        else:
            display = None
        loc_id = li.get("id")
        try:
            if loc_id is not None:
                loc_id = int(loc_id)
                if loc_id <= 0:
                    loc_id = None
        except (TypeError, ValueError):
            loc_id = None
        slug = (li.get("slug") or (display and _slugify_location(display)) or "unknown")

    # 2) Fallback to roleplay-style + legacy keys
    if not display or slug is None or loc_id is None:
        rp = meta.get("currentRoleplay") or meta.get("current_roleplay") or {}
        if not isinstance(rp, dict):
            rp = {}
        rp_display, rp_id = _extract_location_from_mapping(rp)
        fb_display, fb_id = _extract_location_from_mapping(meta)  # last-resort loose read

        display = display or rp_display or fb_display or "Unknown"
        if loc_id is None:
            loc_id = rp_id if rp_id is not None else fb_id
        slug = slug or _slugify_location(display)

    # 3) Stamp locationInfo if missing/incomplete
    li_obj = meta.setdefault("locationInfo", {})
    if "display" not in li_obj:
        li_obj["display"] = display
    if li_obj.get("id") is None and loc_id is not None:
        li_obj["id"] = loc_id
    if not li_obj.get("slug"):
        li_obj["slug"] = slug
    li_obj.setdefault("source", "gateway")
    li_obj.setdefault("updated_at", datetime.utcnow().isoformat())

    # 4) Mirror onto scene_scope (don’t clobber explicit values)
    ss = meta.setdefault("scene_scope", {})
    if not ss.get("location_name"):
        ss["location_name"] = display
    if ss.get("location_id") is None and loc_id is not None:
        ss["location_id"] = loc_id
    if not ss.get("location_slug"):
        ss["location_slug"] = slug


async def _execute_llm(request: LLMRequest):
    """Execute an LLM request via the Nyx gateway."""

    return await execute(request)


_LEGACY_SIDE_EFFECT_TASKS: Dict[str, Dict[str, Any]] = {
    "world": {"task": "nyx.tasks.background.world_tasks.apply_universal", "queue": "background"},
    "memory": {"task": "nyx.tasks.heavy.memory_tasks.add_and_embed", "queue": "heavy"},
    "conflict": {"task": "nyx.tasks.background.conflict_tasks.process_events", "queue": "background"},
    "npc": {"task": "nyx.tasks.background.npc_tasks.run_adaptation_cycle", "queue": "background"},
    "lore": {"task": "nyx.tasks.background.lore_tasks.precompute_scene_bundle", "queue": "background"},
}


async def _invalidate_context_cache_safe(user_id: str | int, conversation_id: str | int) -> None:
    """Invalidate unified context cache for this (user, conversation)."""

    try:
        from logic.aggregator_sdk import context_cache  # late import to avoid cycles
    except Exception:
        return

    if not context_cache:
        return

    key_prefixes = [
        f"context:{int(user_id)}:{int(conversation_id)}",
        f"agg:{int(user_id)}:{int(conversation_id)}:",
    ]

    try:
        delete_fn = getattr(context_cache, "delete", None)
        if callable(delete_fn):
            for prefix in key_prefixes:
                result = delete_fn(key_prefix=prefix)
                if asyncio.iscoroutine(result):
                    await result
            return

        invalidate_many_fn = getattr(context_cache, "invalidate_many", None)
        if callable(invalidate_many_fn):
            result = invalidate_many_fn(key_prefixes)
            if asyncio.iscoroutine(result):
                await result
            return

        legacy_fn = getattr(context_cache, "invalidate", None)
        if callable(legacy_fn):  # pragma: no cover - compatibility shim
            for prefix in key_prefixes:
                result = legacy_fn(prefix)
                if asyncio.iscoroutine(result):
                    await result
            return
    except Exception:
        logger.debug("Context cache invalidation failed softly", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# Config & response models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NyxSDKConfig:
    """Runtime knobs for the SDK adapter."""
    # Moderation
    pre_moderate_input: bool = False
    post_moderate_output: bool = False
    redact_on_moderation_block: bool = True

    # Timeouts & retry
    request_timeout_seconds: float = 120.0
    safety_margin_seconds: float = 1.0      # leave a little time for cleanup/logging
    min_step_seconds: float = 0.25          # never schedule sub-250ms timeouts
    retry_on_failure: bool = True
    retry_delay_seconds: float = 0.75
    timeout_profiles: Dict[str, float] = field(default_factory=dict)

    # Concurrency & caching
    rate_limit_per_conversation: bool = True
    result_cache_ttl_seconds: int = 10  # idempotency window

    # Streaming emulation
    streaming_chunk_size: int = 320  # characters per chunk

    # Telemetry & filtering
    enable_telemetry: bool = False
    enable_response_filter: bool = True


@dataclass
class NyxResponse:
    """Compatibility envelope for app layers that expect a simple object."""
    narrative: str
    choices: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    trace_id: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    # convenience fields pulled forward if present
    image: Optional[Dict[str, Any]] = None
    universal_updates: bool = False
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "choices": self.choices,
            "metadata": self.metadata,
            "world_state": self.world_state,
            "success": self.success,
            "trace_id": self.trace_id,
            "processing_time": self.processing_time,
            "error": self.error,
            "image": self.image,
            "universal_updates": self.universal_updates,
            "telemetry": self.telemetry,
        }

    @classmethod
    def from_orchestrator(cls, result: Dict[str, Any]) -> "NyxResponse":
        meta = result.get("metadata", {}) or {}
        image = meta.get("image")
        world = meta.get("world") or {}
        return cls(
            narrative=result.get("response", "") or "",
            choices=meta.get("choices", []) or [],
            metadata=meta,
            world_state=world,
            success=bool(result.get("success", False)),
            trace_id=result.get("trace_id"),
            processing_time=result.get("processing_time"),
            error=result.get("error"),
            image=image if isinstance(image, dict) else None,
            universal_updates=bool(meta.get("universal_updates", False)),
            telemetry=meta.get("telemetry", {}) or {},
        )


# Post-assembly hook signature
PostHook = Callable[[NyxResponse], Awaitable[NyxResponse]]


# ──────────────────────────────────────────────────────────────────────────────
# SDK – primary adapter with resilience, streaming, warmup, and hooks
# ──────────────────────────────────────────────────────────────────────────────

class NyxAgentSDK:
    """
    High-level entry that:
      1) tries the modern orchestrator,
      2) falls back to a direct agent run if something breaks,
      3) adds moderation, streaming, telemetry, warmup, and background hooks,
      4) guards concurrency and dedups duplicate sends briefly.
    """

    def __init__(self, config: Optional[NyxSDKConfig] = None):
        self.config = config or NyxSDKConfig()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._result_cache: Dict[str, Tuple[float, NyxResponse]] = {}
        self._post_hooks: List[PostHook] = []
        # Store the ResponseFilter class instead of an instance
        self._filter_class = ResponseFilter if (ResponseFilter and self.config.enable_response_filter) else None
        # optional per-conversation context kept only when explicitly warmed
        self._warm_contexts: Dict[str, NyxContext] = {}
        self._snapshot_store = ConversationSnapshotStore() if flags.versioned_cache_enabled() else None
        self._conversation_store = ConversationStore() if ConversationStore else None
        self._legacy_snapshots: Dict[Tuple[str, str], Dict[str, Any]] = {}
        raw_profiles = getattr(self.config, "timeout_profiles", {}) or {}
        self._timeout_profiles: Dict[str, float] = {}
        for key, value in raw_profiles.items():
            try:
                budget = float(value)
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid timeout profile %r=%r", key, value)
                continue
            normalized_key = str(key).strip().lower()
            if not normalized_key:
                continue
            self._timeout_profiles[normalized_key] = budget

    async def _generate_defer_narrative(
        self,
        context: DeferPromptContext,
        trace_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """Ask Nyx to craft a defer narrative; return None if the call fails."""

        if nyx_defer_agent is None:
            return None

        prompt = build_defer_prompt(context)
        if not prompt.strip():
            return None

        # Respect overall deadline if provided
        effective_timeout = timeout
        if effective_timeout is None:
            try:
                effective_timeout = self._resolve_timeout_budget({}) / 3.0
            except Exception:
                effective_timeout = 10.0
        try:
            with _agents_trace("Nyx SDK - Defer Narrative"):
                request = LLMRequest(
                    prompt=prompt,
                    agent=nyx_defer_agent,
                    metadata={"operation": LLMOperation.ORCHESTRATION.value},
                    runner_kwargs={"max_turns": 2},
                )
                result_wrapper = await asyncio.wait_for(
                    _execute_llm(request),
                    timeout=max(0.25, effective_timeout),
                )
                result = result_wrapper.raw
        except asyncio.TimeoutError:
            logger.debug(
                f"[SDK-{trace_id}] Nyx defer narrative generation timed out",
                exc_info=True,
            )
            return None
        except Exception:
            logger.debug(
                f"[SDK-{trace_id}] Nyx defer narrative generation failed",
                exc_info=True,
            )
            return None

        return coalesce_agent_output_text(result)

    # ── lifecycle -------------------------------------------------------------

    async def initialize_agent(self) -> None:
        """Kept for historical symmetry; orchestrator builds its own agents."""
        return None

    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Free rate-limit lock, cached results, warmed context, and addiction gate state for a conversation."""
        self._locks.pop(conversation_id, None)
        self._warm_contexts.pop(conversation_id, None)
        
        # Clear result cache for this conversation
        keys = [k for k in self._result_cache if k.startswith(f"{conversation_id}|")]
        for k in keys:
            self._result_cache.pop(k, None)
        
        # Purge addiction gate state if available
        try:
            from logic.addiction_system_sdk import purge_gate_state
            # We need user_id which might not be available here
            # If you track user_id per conversation, use it:
            # user_id = self._get_user_id_for_conversation(conversation_id)
            # purge_gate_state(int(user_id), int(conversation_id))
            
            # Alternative: expose a version that just needs conversation_id
            # For now, we can iterate through all keys looking for matches
            from logic.addiction_system_sdk import _GATE_STATE, _GATE_LOCKS, _GATE_TOUCH
            
            # Find and remove all entries for this conversation
            conv_id = int(conversation_id) if str(conversation_id).isdigit() else conversation_id
            keys_to_remove = []
            for key in _GATE_STATE.keys():
                if key[1] == conv_id:  # key is (user_id, conversation_id)
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                _GATE_STATE.pop(key, None)
                _GATE_LOCKS.pop(key, None)
                _GATE_TOUCH.pop(key, None)
                
            logger.debug(f"Purged {len(keys_to_remove)} addiction gate entries for conversation {conversation_id}")
        except ImportError:
            pass  # Addiction system not available
        except Exception as e:
            logger.debug(f"Failed to purge addiction gate state: {e}")

    async def warmup_cache(self, conversation_id: str, location: str) -> None:
        """
        Pre-warm LRU/Redis scene-bundle caches for a given location (faster first response).
        """
        ctx = NyxContext(user_id=0, conversation_id=int(conversation_id))
        await ctx.initialize()
        scope = SceneScope(location_id=location)
        try:
            await ctx.context_broker.load_or_fetch_bundle(scene_scope=scope)
            self._warm_contexts[conversation_id] = ctx
            logger.info("Warmup complete for conversation=%s location=%s", conversation_id, location)
        except Exception:
            logger.debug("warmup_cache failed (non-fatal)", exc_info=True)

    # ── hooks -----------------------------------------------------------------

    def register_post_hook(self, hook: PostHook) -> None:
        """Allow other Nyx modules to enrich/transform the final response."""
        self._post_hooks.append(hook)

    def _resolve_timeout_budget(self, meta: Dict[str, Any]) -> float:
        """Resolve the timeout budget for a request based on metadata and profiles."""
        min_step = float(getattr(self.config, "min_step_seconds", 0.25))
        budget = float(getattr(self.config, "request_timeout_seconds", 120.0))

        override = meta.get("timeout_override_seconds") if isinstance(meta, dict) else None
        if override is not None:
            try:
                return max(min_step, float(override))
            except (TypeError, ValueError):
                logger.debug("Invalid timeout_override_seconds=%r; falling back to default", override)

        profile = meta.get("timeout_profile") if isinstance(meta, dict) else None
        if isinstance(profile, str):
            profile_key = profile.strip().lower()
            if profile_key:
                profile_budget = self._timeout_profiles.get(profile_key)
                if profile_budget is not None:
                    return max(min_step, float(profile_budget))

        return max(min_step, budget)

    # ── main entrypoints ------------------------------------------------------

    async def process_user_input(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NyxResponse:
        import time, uuid, re
        t0 = time.time()
        trace_id = uuid.uuid4().hex[:8]
        meta = dict(metadata or {})

        logger.info(f"[SDK-{trace_id}] Processing input: {message[:120]}")
        # Compute a single absolute deadline and pass it downstream.
        timeout_budget = self._resolve_timeout_budget(meta)
        deadline = time.monotonic() + timeout_budget
        meta["_deadline"] = deadline
        if meta.get("timeout_profile"):
            logger.debug(
                "[SDK-%s] Timeout profile %r resolved to %.2fs",
                trace_id,
                meta.get("timeout_profile"),
                timeout_budget,
            )
        safety_margin = float(getattr(self.config, "safety_margin_seconds", 1.0))
        min_step = float(getattr(self.config, "min_step_seconds", 0.25))

        def _time_left() -> float:
            return max(min_step, deadline - time.monotonic())

        # --- rate limiting (unchanged) ---
        lock = None
        if getattr(self.config, "rate_limit_per_conversation", False):
            lock = self._locks.setdefault(conversation_id, asyncio.Lock())
            await lock.acquire()

        # cache key set early
        cache_key = self._cache_key(conversation_id, user_id, message, meta)

        # Normalize location meta early (so orchestrator receives a consistent scene_scope)
        _normalize_location_meta_inplace(meta)

        try:
            # --- cache check ---
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(f"[SDK-{trace_id}] Cache hit")
                return cached

            # --- (1) Optional pre-moderation ---
            if getattr(self.config, "pre_moderate_input", False) and content_moderation_guardrail:
                try:
                    verdict = await content_moderation_guardrail(message)
                    meta["moderation_pre"] = verdict
                    if verdict and verdict.get("blocked"):
                        safe_text = verdict.get("safe_text") or "Your message couldn't be processed."
                        resp = NyxResponse(
                            narrative=safe_text,
                            metadata={"blocked_by": "pre_moderation", "verdict": verdict},
                            success=True,
                            trace_id=trace_id,
                            processing_time=time.time() - t0,
                        )
                        self._write_cache(cache_key, resp)
                        return resp
                except Exception:
                    logger.debug(
                        f"[SDK-{trace_id}] pre-moderation failed softly",
                        exc_info=True,
                    )

            # --- (2) Fast feasibility gate (now softened for movement/location) ---
            feas = None
            try:
                from nyx.nyx_agent.feasibility import assess_action_feasibility_fast

                feas = await assess_action_feasibility_fast(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    text=message,
                )
                meta["feasibility"] = feas
                router_blob = feas.get("router_result") if isinstance(feas, dict) else None
                if isinstance(router_blob, dict):
                    meta["router_result"] = router_blob
            except ImportError as e:
                logger.error(f"[SDK-{trace_id}] Feasibility module not found: {e}")
            except Exception as e:
                logger.error(
                    f"[SDK-{trace_id}] Fast feasibility failed softly: {e}",
                    exc_info=True,
                )

            if isinstance(feas, dict):
                overall = feas.get("overall", {}) or {}
                feasible_flag = overall.get("feasible")
                strategy = (overall.get("strategy") or "").lower()

                # ---------- NEW: robust, future-proof softening ----------
                def _is_soft_violation(rule: str) -> bool:
                    """
                    Treat purely *absence / resolver* style violations as soft:
                    - npc_absent, item_absent, anything containing 'absent'
                    - location_resolver:* (real/fictional resolver couldn't ground it yet)
                    These should NOT hard-block; router + orchestrator can often fix them.
                    """
                    r = str(rule or "").lower()
                    if not r:
                        return False
                    if r.startswith("location_resolver:"):
                        return True
                    if "absent" in r:
                        return True
                    if r in {"npc_absent", "item_absent"}:
                        return True
                    return False

                def _is_soft_location_only(feas_payload: Dict[str, Any]) -> bool:
                    """
                    Returns True when ALL violating intents are:
                      - movement / mundane_action (or uncategorized),
                      - and ONLY have soft 'absent / location_resolver:*' style violations.
                    Those should be treated as advisory, not as pre-orchestrator hard blocks.
                    """
                    per = feas_payload.get("per_intent") or []
                    if not per:
                        return False

                    saw_soft = False

                    for intent in per:
                        if not isinstance(intent, dict):
                            continue

                        cats = set(intent.get("categories") or [])
                        violations = intent.get("violations") or []
                        rules = {
                            str(v.get("rule") or "").lower()
                            for v in violations
                            if isinstance(v, dict)
                        }
                        if not rules:
                            continue

                        # parse_error or explicit "established_impossibility" remain hard
                        if "parse_error" in rules or "established_impossibility" in rules:
                            return False

                        all_soft = all(_is_soft_violation(r) for r in rules)
                        movement_like = (
                            "movement" in cats
                            or "mundane_action" in cats
                            or not cats
                        )

                        if all_soft and movement_like:
                            saw_soft = True
                        else:
                            # Mixed or non-movement + hard violations → treat as hard
                            return False

                    return saw_soft

                soft_location_only = _is_soft_location_only(feas)

                # Only hard-block when:
                #   - overall says it's not feasible, AND
                #   - strategy is deny/defer/ask, AND
                #   - it's NOT just a soft movement/location issue
                if (
                    feasible_flag is False
                    and strategy in {"deny", "defer", "ask"}
                    and not soft_location_only
                ):
                    per = feas.get("per_intent") or []
                    first = per[0] if per and isinstance(per[0], dict) else {}

                    if strategy == "defer":
                        # Defer: enqueue background resolution and give user a narratively flavored "working on it"
                        defer_context, extra_meta = extract_defer_details(feas)
                        leads = extra_meta.get("leads", [])
                        guidance = None
                        if defer_context:
                            guidance = await self._generate_defer_narrative(
                                defer_context, trace_id, timeout=_time_left()
                            )
                        if not guidance:
                            guidance = build_defer_fallback_text(
                                defer_context
                            ) if defer_context else (
                                "Reality is processing that attempt in the background."
                            )
                        alternatives = leads

                    elif strategy == "ask":
                        guidance = first.get("narrator_guidance") or (
                            "I need a bit more detail to ground that."
                        )
                        alternatives = (
                            first.get("suggested_alternatives")
                            or feas.get("choices")
                            or []
                        )
                        extra_meta = {
                            "violations": first.get("violations") or []
                        }

                    else:  # "deny" and not soft_location_only
                        guidance = first.get("narrator_guidance") or (
                            "That can't happen here."
                        )
                        alternatives = first.get("suggested_alternatives") or []
                        extra_meta = {
                            "violations": first.get("violations") or []
                        }

                    alt_list = (
                        list(alternatives)
                        if isinstance(alternatives, (list, tuple))
                        else []
                    )

                    metadata_out = {
                        "action_blocked": strategy != "ask",
                        "feasibility": feas,
                        "block_stage": "pre_orchestrator",
                        "strategy": strategy,
                    }
                    if strategy == "defer":
                        metadata_out["action_deferred"] = True
                    metadata_out.update(extra_meta)

                    resp = NyxResponse(
                        narrative=guidance,
                        choices=[{"text": alt} for alt in alt_list[:4]],
                        metadata=metadata_out,
                        success=True,
                        trace_id=trace_id,
                        processing_time=time.time() - t0,
                    )
                    self._write_cache(cache_key, resp)
                    return resp

                logger.info(
                    f"[SDK-{trace_id}] Feasibility: feasible={feasible_flag} "
                    f"strategy={strategy} soft_location_only={soft_location_only}"
                )

            # --- turn_index, scene_tags, stimuli (unchanged) ---
            if not hasattr(self, "_turn_indices"):
                self._turn_indices: Dict[Tuple[str, str], int] = {}
            key = (str(user_id), str(conversation_id))
            turn_index = self._turn_indices.get(key, -1) + 1
            self._turn_indices[key] = turn_index
            meta["turn_index"] = turn_index

            if "scene_tags" not in meta:
                try:
                    if hasattr(self, "scene_manager"):
                        current_scene = self.scene_manager.get_current_scene(
                            conversation_id
                        )
                        if current_scene:
                            tags = list(current_scene.get("tags", []))
                            if tags:
                                meta["scene_tags"] = tags
                                meta["scene"] = {"tags": tags}
                except Exception:
                    logger.debug(
                        f"[SDK-{trace_id}] scene tag lookup failed softly",
                        exc_info=True,
                    )

            def _compile_stimuli_regex() -> Optional[re.Pattern]:
                tokens = None
                try:
                    from logic.addiction_system_sdk import (
                        AddictionTriggerConfig,
                    )  # type: ignore

                    cfg = AddictionTriggerConfig()
                    vocab = set()
                    for vs in cfg.stimuli_affinity.values():
                        vocab |= set(vs or [])
                    tokens = sorted(vocab)
                except Exception:
                    tokens = [
                        "feet",
                        "toes",
                        "ankle",
                        "barefoot",
                        "sandals",
                        "flipflops",
                        "heels",
                        "perfume",
                        "musk",
                        "sweat",
                        "locker",
                        "gym",
                        "laundry",
                        "socks",
                        "ankle_socks",
                        "knee_highs",
                        "thigh_highs",
                        "stockings",
                        "hips",
                        "ass",
                        "shorts",
                        "tight_skirt",
                        "snicker",
                        "laugh",
                        "eye_roll",
                        "dismissive",
                        "order",
                        "command",
                        "kneel",
                        "obedience",
                        "perspiration",
                        "moist",
                    ]
                escaped = []
                for tkn in tokens:
                    if "_" in tkn:
                        escaped.append(
                            r"\b" + re.escape(tkn).replace(r"\_", r"[-_ ]") + r"\b"
                        )
                    else:
                        escaped.append(r"\b" + re.escape(tkn) + r"\b")
                return re.compile(r"(?:%s)" % "|".join(escaped), flags=re.IGNORECASE)

            def _extract_stimuli(text: str) -> List[str]:
                rx = getattr(self, "_stimuli_rx", None)
                if rx is None:
                    rx = _compile_stimuli_regex()
                    setattr(self, "_stimuli_rx", rx)
                if not rx:
                    return []
                return sorted(
                    {
                        m.group(0).lower()
                        .replace("-", "_")
                        .replace(" ", "_")
                        for m in rx.finditer(text or "")
                    }
                )

            msg_stimuli = _extract_stimuli(message)
            meta_stimuli = {
                s.lower() for s in meta.get("stimuli", []) if isinstance(s, str)
            }
            combined_stimuli = sorted(set(msg_stimuli) | meta_stimuli)
            if combined_stimuli:
                meta["stimuli"] = combined_stimuli

            # --- orchestrator call ---
            result = await self._call_orchestrator_with_timeout(
                message=message,
                conversation_id=conversation_id,
                user_id=user_id,
                meta=meta,
                deadline=deadline,
            )
            resp = NyxResponse.from_orchestrator(result)
            resp.processing_time = resp.processing_time or (time.time() - t0)
            resp.trace_id = resp.trace_id or trace_id

            # Normalize location on the way out
            try:
                _normalize_location_meta_inplace(resp.metadata)
            except Exception:
                logger.debug(
                    f"[SDK-{trace_id}] output location normalization failed softly",
                    exc_info=True,
                )

            # glean stimuli from generated narrative
            try:
                if resp and resp.narrative:
                    out_stimuli = _extract_stimuli(resp.narrative)
                    if out_stimuli:
                        resp.metadata.setdefault("observed_stimuli", [])
                        resp.metadata["observed_stimuli"] = sorted(
                            {
                                *resp.metadata["observed_stimuli"],
                                *out_stimuli,
                            }
                        )
            except Exception:
                logger.debug(
                    f"[SDK-{trace_id}] output stimuli glean failed softly",
                    exc_info=True,
                )

            # --- post moderation ---
            if getattr(self.config, "post_moderate_output", False) and content_moderation_guardrail:
                try:
                    verdict = await content_moderation_guardrail(
                        resp.narrative, is_output=True
                    )
                    resp.metadata["moderation_post"] = verdict
                    if verdict and verdict.get("blocked"):
                        if getattr(self.config, "redact_on_moderation_block", False):
                            resp.narrative = (
                                verdict.get("safe_text") or "Content redacted."
                            )
                            resp.metadata["moderated"] = True
                        else:
                            resp = self._blocked_response(
                                "Content cannot be displayed.",
                                trace_id,
                                t0,
                                stage="post",
                                details=verdict,
                            )
                except Exception:
                    logger.debug(
                        "post-moderation failed (non-fatal)",
                        exc_info=True,
                    )

            # --- response filter (optional) ---
            if getattr(self, "_filter_class", None) and resp.narrative:
                try:
                    filter_instance = self._filter_class(
                        user_id=int(user_id),
                        conversation_id=int(conversation_id),
                    )
                    filtered = filter_instance.filter_text(resp.narrative)
                    if filtered != resp.narrative:
                        resp.narrative = filtered
                        resp.metadata.setdefault("filters", {})[
                            "response_filter"
                        ] = True
                except Exception:
                    logger.debug("ResponseFilter failed softly", exc_info=True)

            # --- post hooks ---
            if getattr(self, "_post_hooks", None):
                for hook in self._post_hooks:
                    try:
                        resp = await hook(resp)
                    except Exception:
                        logger.debug("post_hook failed softly", exc_info=True)

            # --- telemetry/maintenance/fanout ---
            await self._maybe_log_perf(resp)
            if _time_left() > safety_margin:
                await self._maybe_enqueue_maintenance(resp, conversation_id)
                await self._fanout_post_turn(
                    resp, user_id, conversation_id, trace_id
                )

            # Invalidate unified/aggregator cache if we actually touched world state
            try:
                world_changed = bool(resp.world_state) or bool(
                    resp.metadata.get("universal_updates")
                ) or bool(resp.metadata.get("canon_event_applied")) or bool(
                    resp.metadata.get("action_deferred")
                )
                if world_changed:
                    await _invalidate_context_cache_safe(user_id, conversation_id)
                    self._clear_result_cache_for_conversation(conversation_id)
            except Exception:
                logger.debug(
                    f"[SDK-{trace_id}] cache invalidation failed softly",
                    exc_info=True,
                )

            if resp.success:
                await self._append_conversation_turns_safe(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    user_message=message,
                    nyx_response=resp,
                    metadata=meta,
                    trace_id=trace_id,
                )

            self._write_cache(cache_key, resp)
            return resp

        except Exception as e:
            logger.error(
                "orchestrator path failed; attempting fallback. err=%s",
                e,
                exc_info=True,
            )
            try:
                resp = await self._fallback_direct_run(
                    message=message,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata=meta,
                    trace_id=trace_id,
                    t0=t0,
                )

                # post moderation/filter/hooks as above
                if getattr(self.config, "post_moderate_output", False) and content_moderation_guardrail:
                    try:
                        verdict = await content_moderation_guardrail(
                            resp.narrative, is_output=True
                        )
                        resp.metadata["moderation_post"] = verdict
                        if verdict and verdict.get("blocked"):
                            if getattr(self.config, "redact_on_moderation_block", False):
                                resp.narrative = (
                                    verdict.get("safe_text") or "Content redacted."
                                )
                                resp.metadata["moderated"] = True
                            else:
                                resp = self._blocked_response(
                                    "Content cannot be displayed.",
                                    trace_id,
                                    t0,
                                    stage="post",
                                    details=verdict,
                                )
                    except Exception:
                        logger.debug(
                            "post-moderation failed (non-fatal)",
                            exc_info=True,
                        )

                if getattr(self, "_filter_class", None) and resp.narrative:
                    try:
                        filter_instance = self._filter_class(
                            user_id=int(user_id),
                            conversation_id=int(conversation_id),
                        )
                        filtered = filter_instance.filter_text(resp.narrative)
                        if filtered != resp.narrative:
                            resp.narrative = filtered
                            resp.metadata.setdefault("filters", {})[
                                "response_filter"
                            ] = True
                    except Exception:
                        logger.debug(
                            "ResponseFilter failed softly", exc_info=True
                        )

                if getattr(self, "_post_hooks", None):
                    for hook in self._post_hooks:
                        try:
                            resp = await hook(resp)
                        except Exception:
                            logger.debug(
                                "post_hook failed softly", exc_info=True
                            )

                await self._maybe_log_perf(resp)
                if _time_left() > safety_margin:
                    await self._maybe_enqueue_maintenance(resp, conversation_id)
                    await self._fanout_post_turn(
                        resp, user_id, conversation_id, trace_id
                    )

                try:
                    _normalize_location_meta_inplace(resp.metadata)
                    world_changed = bool(resp.world_state) or bool(
                        resp.metadata.get("universal_updates")
                    )
                    if world_changed:
                        await _invalidate_context_cache_safe(
                            user_id, conversation_id
                        )
                        self._clear_result_cache_for_conversation(conversation_id)
                except Exception:
                    logger.debug(
                        f"[SDK-{trace_id}] fallback post steps failed softly",
                        exc_info=True,
                    )

                if resp.success:
                    await self._append_conversation_turns_safe(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        user_message=message,
                        nyx_response=resp,
                        metadata=meta,
                        trace_id=trace_id,
                    )

                self._write_cache(cache_key, resp)
                return resp

            except Exception:
                logger.exception(
                    f"[SDK-{trace_id}] Fallback path failed", exc_info=True
                )
                raise

        except Exception as e:
            logger.error(f"[SDK-{trace_id}] Process failed", exc_info=True)
            TASK_FAILURES.labels(task="nyx_sdk", reason=type(e).__name__).inc()
            return NyxResponse(
                narrative=f"System error: {e}",
                metadata={"error": "sdk_process_failure", "details": str(e)},
                success=False,
                trace_id=trace_id,
                processing_time=time.time() - t0,
                error=str(e),
            )
        finally:
            if lock and lock.locked():
                lock.release()




    async def stream_user_input(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Best-effort streaming. If future runtimes expose true token streaming,
        integrate here. For now, emulate by chunking final text.
        Yields event dicts: {"type": "...", ...}
        """
        t0 = time.time()
        yield {"type": "start", "conversation_id": conversation_id, "user_id": user_id}

        resp = await self.process_user_input(message, conversation_id, user_id, metadata)
        first_chunk_delay = time.time() - t0
        TTFB_SECONDS.labels(channel="sdk_stream").observe(first_chunk_delay)
        text = resp.narrative or ""
        chunk = max(64, self.config.streaming_chunk_size)

        i = 0
        while i < len(text):
            yield {"type": "token", "text": text[i : i + chunk]}
            i += chunk

        yield {
            "type": "end",
            "success": resp.success,
            "trace_id": resp.trace_id,
            "processing_time": resp.processing_time or (time.time() - t0),
            "metadata": resp.metadata,
        }

    # ── internals -------------------------------------------------------------

    async def _append_conversation_turns_safe(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: Optional[str],
        nyx_response: Optional[NyxResponse],
        metadata: Optional[Dict[str, Any]],
        trace_id: str,
    ) -> None:
        store = getattr(self, "_conversation_store", None)
        if not store or nyx_response is None:
            return

        try:
            meta = metadata or {}

            def _pick_name(keys: Tuple[str, ...], default: str) -> str:
                for key in keys:
                    value = meta.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                return default

            player_name = _pick_name(
                ("player_name", "user_display_name", "user_name", "player"), "Player"
            )
            nyx_name = _pick_name(("nyx_display_name", "assistant_name"), "Nyx")

            if user_message and str(user_message).strip():
                await store.append_turn(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    turn={"sender": player_name, "content": str(user_message)},
                )

            narrative = nyx_response.narrative if nyx_response.narrative is not None else ""
            if narrative:
                await store.append_turn(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    turn={
                        "sender": nyx_name,
                        "content": narrative,
                        "metadata": nyx_response.metadata,
                    },
                )
        except Exception:
            logger.debug(
                "[SDK-%s] ConversationStore append failed softly", trace_id, exc_info=True
            )

    async def _call_orchestrator_with_timeout(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        meta: Dict[str, Any],
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Call orchestrator without an outer watchdog to avoid mid-TLS cancellations.
        Pass an absolute deadline in metadata and let the orchestrator budget the turn.
        Still retry once if there's meaningful time left.
        """
        if deadline is None:
            deadline = meta.get("_deadline") or (time.monotonic() + self._resolve_timeout_budget(meta))
        def _left() -> float:
            return max(float(getattr(self.config, "min_step_seconds", 0.25)), deadline - time.monotonic())
        record_queue_delay_from_context(meta, queue="sdk")
        try:
            with trace_step(
                "nyx_sdk.orchestrator_call",
                meta.get("trace_id"),
                conversation_id=str(conversation_id),
                user_id=str(user_id),
                attempt="primary",
            ):
                return await _orchestrator_process(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    user_input=message,
                    context_data=meta,
                )
        except Exception as first_error:
            logger.warning("orchestrator call failed: %s", first_error)
            if not self.config.retry_on_failure:
                raise
            # Retry only if we still have budget
            delay = min(self.config.retry_delay_seconds, max(0.0, _left() - 1.0))
            if delay <= 0.0:
                raise
            await asyncio.sleep(delay)
            logger.info("retrying orchestrator once…")
            with trace_step(
                "nyx_sdk.orchestrator_call",
                meta.get("trace_id"),
                conversation_id=str(conversation_id),
                user_id=str(user_id),
                attempt="retry",
            ):
                return await _orchestrator_process(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    user_input=message,
                    context_data=meta,
                )

    def _is_soft_location_only_violation(feas: Dict[str, Any]) -> bool:
        """
        Return True when all violating intents are 'soft' absence / location issues
        (e.g. npc_absent, item_absent, location_resolver:*) on movement / mundane actions.
        Those should not cause a hard pre-block; we let the router / resolver handle them.
        """
        if not isinstance(feas, dict):
            return False
    
        per = feas.get("per_intent") or []
        if not per:
            return False
    
        saw_soft = False
    
        for intent in per:
            if not isinstance(intent, dict):
                continue
    
            cats = set(intent.get("categories") or [])
            violations = intent.get("violations") or []
    
            rules = {
                str(v.get("rule") or "").lower()
                for v in violations
                if isinstance(v, dict)
            }
            if not rules:
                continue
    
            # Hard reasons that must remain hard
            if "parse_error" in rules or "established_impossibility" in rules:
                return False
    
            all_soft = True
            for r in rules:
                if not r:
                    continue
                # location resolver asks / denies
                if r.startswith("location_resolver:"):
                    continue
                # “no NPC / item by that name here”
                if r in {"npc_absent", "item_absent"}:
                    continue
                if "absent" in r:
                    continue
                # anything else is not soft
                all_soft = False
                break
    
            movement_like = (
                "movement" in cats
                or "mundane_action" in cats
                or not cats  # parser gave us nothing, still likely a move
            )
    
            if all_soft and movement_like:
                saw_soft = True
            else:
                # Mixed hard + soft or wrong categories → treat as real deny
                return False
    
        return saw_soft


    async def _fallback_direct_run(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        metadata: Dict[str, Any],
        trace_id: str,
        t0: float,
    ) -> NyxResponse:
        """
        Minimal reproduction of the orchestrator flow using the public agent APIs.
        Only invoked if the primary path fails and the runtime is available.
        This version mirrors the soft-vs-hard feasibility logic from process_user_input:
        it will NOT hard-deny purely "soft" absence/location violations (npc_absent,
        item_absent, location_resolver:*, etc.), but will still deny true world-rule
        or established-impossibility violations.
        """
        if not (RunConfig and RunContextWrapper and nyx_main_agent):
            logger.error("[SDK-%s] Fallback runtime unavailable", trace_id)
            raise RuntimeError("Fallback agent runtime is unavailable in this environment.")

        logger.info("[SDK-%s] Entering fallback agent run", trace_id)

        ctx = NyxContext(user_id=int(user_id), conversation_id=int(conversation_id))
        await ctx.initialize()
        ctx.current_context = (metadata or {}).copy()
        ctx.current_context["user_input"] = message
        _preserve_hydrated_location(ctx.current_context, ctx.current_location)

        import time as _time_mod
        deadline = metadata.get("_deadline") or (
            _time_mod.monotonic() + self._resolve_timeout_budget(metadata or {})
        )
        min_step = float(getattr(self.config, "min_step_seconds", 0.25))
        safety_margin = float(getattr(self.config, "safety_margin_seconds", 1.0))

        def _left() -> float:
            return max(min_step, deadline - _time_mod.monotonic())

        # ----------------- FAST FEASIBILITY (fallback) -----------------
        feasibility = None
        if assess_action_feasibility:
            try:
                feasibility = await assess_action_feasibility(ctx, message)
                ctx.current_context["feasibility"] = feasibility
                logger.info(
                    "[SDK-%s] fallback feasibility overall=%s",
                    trace_id,
                    (feasibility or {}).get("overall", {}),
                )
            except Exception as e:
                logger.error(
                    "[SDK-%s] fallback feasibility check failed softly: %s",
                    trace_id,
                    e,
                    exc_info=True,
                )

        def _is_soft_violation(rule: str) -> bool:
            """
            Same definition as in process_user_input:
            treat "absence" / resolver-only issues as soft.
            """
            r = str(rule or "").lower()
            if not r:
                return False
            if r.startswith("location_resolver:"):
                return True
            if "absent" in r:
                return True
            if r in {"npc_absent", "item_absent"}:
                return True
            return False

        def _is_soft_location_only(feas: Dict[str, Any]) -> bool:
            """
            True if every violating intent is a movement/mundane action whose only
            violations are soft (npc_absent/item_absent/location_resolver:*).
            Those should not cause a hard pre-block; we let the agent/orchestrator
            try to resolve or narratively handle them.
            """
            per = feas.get("per_intent") or []
            if not per:
                return False

            saw_soft = False
            for intent in per:
                if not isinstance(intent, dict):
                    continue
                cats = set(intent.get("categories") or [])
                violations = intent.get("violations") or []
                rules = {
                    str(v.get("rule") or "").lower()
                    for v in violations
                    if isinstance(v, dict)
                }
                if not rules:
                    continue

                # hard reasons always stay hard
                if "parse_error" in rules or "established_impossibility" in rules:
                    return False

                all_soft = all(_is_soft_violation(r) for r in rules)
                movement_like = (
                    "movement" in cats
                    or "mundane_action" in cats
                    or not cats
                )

                if all_soft and movement_like:
                    saw_keep = True
                    saw_soft = True
                else:
                    return False

            return saw_soft

        enhanced_input = message

        if isinstance(feasibility, dict):
            overall = feasibility.get("overall", {}) or {}
            feasible_flag = overall.get("feasible")
            strategy = (overall.get("strategy") or "").lower()

            soft_location_only = _is_soft_location_only(feasibility)

            # HARD DENY ONLY if not soft-location-only
            if feasible_flag is False and strategy == "deny" and not soft_location_only:
                per = feasibility.get("per_intent") or []
                first = per[0] if per and isinstance(per[0], dict) else {}
                violations = first.get("violations", []) or []
                if any(
                    (isinstance(v, dict) and v.get("rule") == "established_impossibility")
                    for v in violations
                ):
                    violation_text = (
                        violations[-1].get("reason")
                        if violations and isinstance(violations[-1], dict)
                        else "That contradicts something already established."
                    )
                else:
                    violation_text = first.get("narrator_guidance") or "That action can’t occur in this world."

                if record_impossibility:
                    try:
                        await record_impossibility(ctx, message, violation_text)
                    except Exception:
                        logger.debug(
                            "[SDK-%s] record_impossibility failed softly", trace_id, exc_info=True
                        )

                logger.warning(
                    "[SDK-%s] Fallback DENY enforced. reason=%s", trace_id, violation_text
                )

                alternatives = first.get("suggested_alternatives") or []
                rejection_narrative = (
                    f"{violation_text}\n\n"
                    "Try something that fits what’s already true about this world, "
                    "or ask the Narrator for a hint."
                )

                choice_payload = [
                    {
                        "text": alt,
                        "description": "A possible action within this reality",
                        "feasible": True,
                    }
                    for alt in alternatives[:4]
                ]

                meta = {
                    "world": {},
                    "image": {},
                    "telemetry": {},
                    "universal_updates": False,
                    "path": "fallback_feasibility",
                    "trace_id": trace_id,
                    "feasibility": feasibility,
                    "action_blocked": True,
                    "block_reason": violation_text,
                    "reality_maintained": True,
                    "violations": violations,
                }

                return NyxResponse(
                    narrative=rejection_narrative,
                    choices=choice_payload,
                    metadata=meta,
                    world_state={},
                    success=True,
                    trace_id=trace_id,
                    processing_time=(_time_mod.time() - t0),
                )

            if feasible_flag is False and strategy == "ask" and not soft_location_only:
                per = feasibility.get("overall") or {}
                guidance = per.get("narrator_guidance") or "I need a bit more detail to ground that."
                alternatives = (
                    per.get("suggested_alternatives")
                    or feasibility.get("choices")
                    or []
                )
                logger.info("[SDK-%s] Fallback prompting for clarification", trace_id)
                resp = NyxResponse(
                    narrative=guidance,
                    choices=[{"text": alt} for alt in alternatives[:4]],
                    metadata={
                        "feasibility": feasibility,
                        "action_blocked": True,
                        "block_stage": "fallback_pre_orchestrator",
                        "strategy": "ask",
                    },
                    success=True,
                    trace_id=trace_id,
                    processing_time=(_time_mod.time() - t0),
                )
                return resp

            if feasible_flag is True:
                if record_possibility:
                    try:
                        intents = feasibility.get("per_intent") or []
                        if intents:
                            cats = intents[0].get("categories", [])
                            if cats:
                                await record_possibility(ctx, message, cats)
                    except Exception:
                        logger.debug(
                            "[SDK-%s] record_possibility failed softly", trace_id, exc_info=True
                        )

                enhanced_input = (
                    "[REALITY CHECK: Action is feasible within universe laws.]\n\n"
                    f"{message}"
                )
                logger.info("[SDK-%s] Fallback proceeding with feasible action", trace_id)

        # ----------------- RUN MAIN AGENT (fallback) -----------------
        safe_settings = ModelSettings(strict_tools=False)
        run_config = RunConfig(model_settings=safe_settings, workflow_name="Nyx Fallback")
        runner_context = RunContextWrapper(ctx)

        try:
            with _agents_trace("Nyx SDK - Fallback Run"):
                request = LLMRequest(
                    prompt=enhanced_input,
                    agent=nyx_main_agent,
                    metadata={
                        "operation": LLMOperation.ORCHESTRATION.value,
                        "path": "sdk_fallback",
                    },
                    runner_kwargs={
                        "context": runner_context,
                        "run_config": run_config,
                    },
                )
                result_wrapper = await asyncio.wait_for(
                    _execute_llm(request),
                    timeout=max(0.5, _left() - safety_margin),
                )
                result = result_wrapper.raw
        except Exception:
            logger.exception("[SDK-%s] Fallback Runner invocation failed", trace_id)
            raise

        # Extract narrative
        def _extract_text(obj: Any) -> str:
            for attr in ("final_output", "output_text", "output", "text"):
                val = getattr(obj, attr, None)
                if isinstance(val, str) and val.strip():
                    return val
            for attr in ("messages", "history", "events"):
                seq = getattr(obj, attr, None) or []
                for ev in seq:
                    content = ev.get("content") if isinstance(ev, dict) else None
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in {"output_text", "text"}:
                                txt = c.get("text")
                                if isinstance(txt, str) and txt.strip():
                                    return txt
            return str(obj) if obj is not None else ""

        narrative = _extract_text(result) or ""

        meta = {
            "world": {},
            "image": {},
            "telemetry": {},
            "universal_updates": False,
            "path": "fallback",
            "trace_id": trace_id,
        }
        if feasibility is not None:
            meta["feasibility"] = feasibility
            router_blob = feasibility.get("router_result") if isinstance(feasibility, dict) else None
            if isinstance(router_blob, dict):
                meta["router_result"] = router_blob

        response = NyxResponse(
            narrative=narrative,
            choices=[],
            metadata=meta,
            world_state={},
            success=True,
            trace_id=trace_id,
            processing_time=(_time_mod.time() - t0),
        )
        logger.info(
            "[SDK-%s] Fallback response generated (narrative_len=%d)",
            trace_id,
            len(narrative),
        )
        if response.success:
            await self._append_conversation_turns_safe(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=message,
                nyx_response=response,
                metadata=metadata,
                trace_id=trace_id,
            )
        return response


    def _clear_result_cache_for_conversation(self, conversation_id: str) -> None:
        """Remove all cached results for a given conversation id."""
        keys = [k for k in list(self._result_cache.keys()) if k.startswith(f"{conversation_id}|")]
        for k in keys:
            self._result_cache.pop(k, None)

    async def _maybe_log_perf(self, resp: NyxResponse) -> None:
        if not (self.config.enable_telemetry and _log_perf):
            return
        try:
            await _log_perf(
                {
                    "total_time": resp.processing_time,
                    "timestamp": time.time(),
                    "bundle_sections": None,  # orchestrator owns granular logs
                    "success": resp.success,
                }
            )
        except Exception:
            logger.debug("telemetry logging failed (non-fatal)", exc_info=True)

    async def _maybe_enqueue_maintenance(self, resp: NyxResponse, conversation_id: str) -> None:
        """Lightweight background nudge; orchestrator already does the heavy lifting."""
        if not _enqueue_task:
            return
        try:
            await _enqueue_task(
                task_name="world.update_universal",
                params={"response": resp.to_dict(), "conversation_id": str(conversation_id)},
                priority="low",
                delay_seconds=1,
            )
        except Exception:
            logger.debug("enqueue_task failed (non-fatal)", exc_info=True)

    async def _build_side_effect_payload(
        self,
        resp: NyxResponse,
        user_id: str,
        conversation_id: str,
        trace_id: str,
        user_id_int: Optional[int] = None,
        conversation_id_int: Optional[int] = None,
    ) -> tuple[str, Dict[str, Dict[str, Any]]]:
        metadata = resp.metadata or {}
        user_key = str(user_id)
        conversation_key = str(conversation_id)
        snapshot_key = (user_key, conversation_key)

        # Normalize once more for safety
        try:
            _normalize_location_meta_inplace(metadata)
        except Exception:
            logger.debug(f"[SDK-{trace_id}] side-effect location normalization failed softly", exc_info=True)

        turn_id = metadata.get("turn_id") or f"{trace_id}-{int(time.time() * 1000)}"
        metadata["turn_id"] = turn_id

        if flags.versioned_cache_enabled() and self._snapshot_store is not None:
            snapshot = self._snapshot_store.get(user_key, conversation_key)
        else:
            snapshot = dict(self._legacy_snapshots.get(snapshot_key, {}))

        if not isinstance(snapshot, dict):
            snapshot = {}

        try:
            current_world_version = int(snapshot.get("world_version", 0) or 0)
        except (TypeError, ValueError):
            current_world_version = 0

        # Prefer scene_scope; fill it from locationInfo if missing
        scene_scope = metadata.get("scene_scope") or {}
        if isinstance(scene_scope, str):
            try:
                import json
                scene_scope = json.loads(scene_scope)
            except Exception:
                scene_scope = {}

        li = metadata.get("locationInfo") if isinstance(metadata.get("locationInfo"), dict) else None
        if li:
            scene_scope.setdefault("location_name", li.get("display") or li.get("name"))
            if scene_scope.get("location_id") is None and li.get("id") is not None:
                try:
                    scene_scope["location_id"] = int(li["id"])
                except (TypeError, ValueError):
                    pass
            scene_scope.setdefault("location_slug", li.get("slug"))

        participants: List[str] = []
        npc_ids = scene_scope.get("npc_ids") if isinstance(scene_scope, dict) else None
        if isinstance(npc_ids, list):
            participants = [str(n) for n in npc_ids]
        elif isinstance(npc_ids, set):
            participants = [str(n) for n in sorted(npc_ids)]

        context_stats = metadata.get("context_stats") if isinstance(metadata, dict) else None
        if not participants and isinstance(context_stats, dict):
            candidate = context_stats.get("npc_names") or context_stats.get("active_npcs")
            if isinstance(candidate, list):
                participants = [str(n) for n in candidate]

        nation_ids = scene_scope.get("nation_ids") if isinstance(scene_scope, dict) else None
        region_id = None
        if isinstance(nation_ids, list) and nation_ids:
            region_id = nation_ids[0]
        elif isinstance(nation_ids, set) and nation_ids:
            region_id = next(iter(nation_ids))

        # Prefer a concrete location id; else none
        scene_id = None
        if isinstance(scene_scope, dict):
            scene_id = scene_scope.get("location_id") or scene_scope.get("scene_id")

        world_state = resp.world_state or {}
        next_world_version = current_world_version + 1 if world_state else current_world_version

        events: List[SideEffect] = []

        conflict_meta = (
            metadata.get("conflict_event")
            or metadata.get("conflict")
            or metadata.get("active_conflict")
        )
        conflict_id = None
        conflict_active = snapshot.get("conflict_active", False)
        if isinstance(conflict_meta, dict) and conflict_meta:
            conflict_id = conflict_meta.get("conflict_id") or conflict_meta.get("id")
            conflict_active = bool(
                conflict_meta.get("active")
                or conflict_meta.get("is_active")
                or conflict_meta.get("ongoing")
            )

        if flags.domain_events_enabled():
            if resp.narrative:
                events.append(
                    MemoryEvent(
                        turn_id=turn_id,
                        user_id=user_key,
                        conversation_id=conversation_key,
                        text=resp.narrative,
                        refs={
                            "scene_id": scene_id,
                            "region_id": region_id,
                            "trace_id": trace_id,
                        },
                    )
                )

            if world_state:
                events.append(
                    WorldDelta(
                        turn_id=turn_id,
                        user_id=user_key,
                        conversation_id=conversation_key,
                        deltas=world_state,
                        incoming_world_version=next_world_version,
                        metadata={"trace_id": trace_id},
                    )
                )

            if (
                flags.conflict_fsm_enabled()
                and isinstance(conflict_meta, dict)
                and conflict_meta
            ):
                events.append(
                    ConflictEvent(
                        turn_id=turn_id,
                        user_id=user_key,
                        conversation_id=conversation_key,
                        conflict_id=str(conflict_id) if conflict_id is not None else None,
                        payload=conflict_meta,
                    )
                )

            if participants:
                events.append(
                    NPCStimulus(
                        turn_id=turn_id,
                        user_id=user_key,
                        conversation_id=conversation_key,
                        npcs=participants,
                        payload={"trace_id": trace_id},
                    )
                )

            if scene_id is not None or region_id is not None:
                hint_payload: Dict[str, Any] = {"trace_id": trace_id}
                if isinstance(scene_scope, dict):
                    scope_nation_ids = scene_scope.get("nation_ids")
                    normalized_nation_ids: List[int] = []
                    if isinstance(scope_nation_ids, (list, tuple, set)):
                        for raw in scope_nation_ids:
                            try:
                                normalized = int(raw)
                            except (TypeError, ValueError):
                                continue
                            normalized_nation_ids.append(normalized)
                    elif isinstance(scope_nation_ids, int):
                        normalized_nation_ids.append(scope_nation_ids)
                    if normalized_nation_ids:
                        hint_payload["nation_ids"] = sorted(set(normalized_nation_ids))[:5]
                events.append(
                    LoreHint(
                        turn_id=turn_id,
                        user_id=user_key,
                        conversation_id=conversation_key,
                        scene_id=str(scene_id) if scene_id is not None else None,
                        region_id=str(region_id) if region_id is not None else None,
                        payload=hint_payload,
                    )
                )

        # Build updated snapshot – prefer locationInfo-backed data
        updated_snapshot = dict(snapshot)
        if isinstance(scene_scope, dict):
            if scene_scope.get("location_name"):
                updated_snapshot["location_name"] = scene_scope.get("location_name")
            if scene_scope.get("location_id") is not None:
                updated_snapshot["scene_id"] = str(scene_scope.get("location_id"))
            if scene_scope.get("time_window") is not None:
                updated_snapshot["time_window"] = scene_scope.get("time_window")
            if scene_scope.get("location_slug"):
                updated_snapshot["location_slug"] = scene_scope.get("location_slug")
        if region_id is not None:
            updated_snapshot["region_id"] = str(region_id)
        if participants:
            updated_snapshot["participants"] = participants
        updated_snapshot["world_version"] = next_world_version
        if conflict_id is not None:
            updated_snapshot["conflict_id"] = str(conflict_id)
        if conflict_meta:
            updated_snapshot["conflict_active"] = conflict_active
        updated_snapshot["updated_at"] = datetime.utcnow().isoformat()

        if flags.versioned_cache_enabled() and self._snapshot_store is not None:
            self._snapshot_store.put(user_key, conversation_key, updated_snapshot)
        else:
            self._legacy_snapshots[snapshot_key] = dict(updated_snapshot)

        if (
            flags.versioned_cache_enabled()
            and self._snapshot_store is not None
            and user_id_int is not None
            and conversation_id_int is not None
        ):
            canonical_payload = build_canonical_snapshot_payload(updated_snapshot)
            if canonical_payload:
                await persist_canonical_snapshot(user_id_int, conversation_id_int, canonical_payload)

        grouped = group_side_effects(events)
        return turn_id, grouped

    async def _fanout_post_turn(
        self,
        resp: NyxResponse,
        user_id: str,
        conversation_id: str,
        trace_id: str,
    ) -> None:
        use_outbox = flags.outbox_enabled() and post_turn_dispatch is not None
        user_id_text = str(user_id)
        conversation_id_text = str(conversation_id)
        user_id_int: Optional[int]
        conversation_id_int: Optional[int]
        try:
            user_id_int = int(user_id_text)
            conversation_id_int = int(conversation_id_text)
        except (TypeError, ValueError):
            logger.warning(
                "[SDK-%s] Skipping canonical snapshot persistence due to non-integer identifiers user_id=%s conversation_id=%s",
                trace_id,
                user_id,
                conversation_id,
            )
            user_id_int = None
            conversation_id_int = None
        try:
            turn_id, grouped = await self._build_side_effect_payload(
                resp,
                user_id_text,
                conversation_id_text,
                trace_id,
                user_id_int,
                conversation_id_int,
            )
        except Exception:
            logger.exception("[SDK-%s] Failed to build side-effect payload", trace_id)
            return
        if not grouped:
            return

        payload = {
            "user_id": user_id_text,
            "conversation_id": conversation_id_text,
            "turn_id": turn_id,
            "trace_id": trace_id,
            "side_effects": grouped,
        }

        start = time.perf_counter()

        if use_outbox:
            def _enqueue() -> None:
                try:
                    post_turn_dispatch.apply_async(kwargs={"payload": payload}, queue="realtime", priority=0)
                except Exception:  # pragma: no cover
                    logger.exception("[SDK-%s] Failed to enqueue TurnPostProcessor", trace_id)

            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _enqueue)
            except RuntimeError:
                _enqueue()
        else:
            self._dispatch_side_effects_legacy(payload, trace_id)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        resp.metadata.setdefault("telemetry", {}).setdefault("post_turn", {})[
            "enqueue_ms"
        ] = round(elapsed_ms, 2)
        resp.metadata.setdefault("post_turn", {})["turn_id"] = turn_id

    def _blocked_response(self, safe_text: str, trace_id: str, t0: float, stage: str, details: Optional[Dict[str, Any]]) -> NyxResponse:
        return NyxResponse(
            narrative=safe_text,
            metadata={"moderation_blocked": True, "stage": stage, "details": details or {}},
            success=False,
            trace_id=trace_id,
            processing_time=(time.time() - t0),
        )

    def _dispatch_side_effects_legacy(self, payload: Dict[str, Any], trace_id: str) -> None:
        side_effects = payload.get("side_effects") or {}
        if not side_effects:
            return

        try:
            from nyx.tasks.base import app as celery_app
        except Exception:  # pragma: no cover - legacy path best effort
            logger.warning("[SDK-%s] Celery app unavailable; skipping legacy fanout", trace_id)
            return

        for key, effect_payload in side_effects.items():
            if not effect_payload:
                continue
            config = _LEGACY_SIDE_EFFECT_TASKS.get(key)
            if not config:
                continue
            options: Dict[str, Any] = {}
            queue = config.get("queue")
            if queue:
                options.setdefault("queue", queue)
                options.setdefault("routing_key", queue)
            try:
                celery_app.send_task(config["task"], kwargs={"payload": effect_payload}, **options)
            except Exception:
                logger.exception(
                    "[SDK-%s] Failed to send %s side-effect via legacy path",
                    trace_id,
                    key,
                )

    def _error_response(self, error_message: str, trace_id: str, t0: float) -> NyxResponse:
        return NyxResponse(
            narrative="*Nyx's form flickers* Something interfered with our connection…",
            metadata={"error": error_message},
            success=False,
            trace_id=trace_id,
            processing_time=(time.time() - t0),
            error=error_message,
        )

    def _cache_key(self, conversation_id: str, user_id: str, message: str, meta: Dict[str, Any]) -> str:
        # Ignore volatile metadata keys starting with underscore
        stable_meta = {k: v for k, v in (meta or {}).items() if not str(k).startswith("_")}
        h = hashlib.sha256()
        h.update(message.encode("utf-8"))
        h.update(str(sorted(stable_meta.items())).encode("utf-8"))
        return f"{conversation_id}|{user_id}|{h.hexdigest()[:16]}"

    def _read_cache(self, key: str) -> Optional[NyxResponse]:
        ttl = self.config.result_cache_ttl_seconds
        if ttl <= 0:
            return None
        entry = self._result_cache.get(key)
        if not entry:
            return None
        ts, resp = entry
        if (time.time() - ts) <= ttl:
            return resp
        self._result_cache.pop(key, None)
        return None

    def _write_cache(self, key: str, resp: NyxResponse) -> None:
        ttl = self.config.result_cache_ttl_seconds
        if ttl <= 0:
            return
        self._result_cache[key] = (time.time(), resp)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy-compatible runner
# ──────────────────────────────────────────────────────────────────────────────

class NyxAgentRunner:
    """Legacy interface wrapper retaining the historical `.run()` surface."""

    def __init__(self, config: Optional[NyxSDKConfig] = None):
        self.sdk = NyxAgentSDK(config)

    async def run(
        self,
        user_input: str,
        conversation_id: str,
        user_id: str,
        **kwargs,
    ) -> NyxResponse:
        return await self.sdk.process_user_input(
            message=user_input,
            conversation_id=conversation_id,
            user_id=user_id,
            metadata=kwargs or {},
        )

    async def initialize(self) -> None:
        await self.sdk.initialize_agent()


__all__ = [
    "NyxSDKConfig",
    "NyxAgentSDK",
    "NyxAgentRunner",
    "NyxResponse",
    "process_user_input",
]

# ──────────────────────────────────────────────────────────────────────────────
# Back-compat top-level function (what routes.story_routes expects)
# ──────────────────────────────────────────────────────────────────────────────
async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Thin wrapper so legacy imports like
        from nyx.nyx_agent_sdk import process_user_input
    continue to work. Returns an orchestrator-shaped dict.
    """
    sdk = NyxAgentSDK()
    resp = await sdk.process_user_input(
        message=user_input,
        conversation_id=str(conversation_id),
        user_id=str(user_id),
        metadata=context_data or {},
    )
    return {
        "response": resp.narrative,
        "success": resp.success,
        "metadata": resp.metadata,
        "trace_id": resp.trace_id,
        "processing_time": resp.processing_time,
        "error": resp.error,
    }
