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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple

# ── Core modern orchestrator
from .nyx_agent.orchestrator import process_user_input as _orchestrator_process
from .nyx_agent.context import NyxContext, SceneScope

from nyx.nyx_agent.models import NyxResponse

# Fallback agent runtime (loaded lazily; optional in some environments)
try:
    from agents import Runner, RunConfig, ModelSettings, RunContextWrapper
    from .nyx_agent.agents import nyx_main_agent, DEFAULT_MODEL_SETTINGS
except Exception:  # pragma: no cover
    Runner = None
    RunConfig = None
    ModelSettings = None
    RunContextWrapper = None
    nyx_main_agent = None
    DEFAULT_MODEL_SETTINGS = None

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
    request_timeout_seconds: float = 45.0
    retry_on_failure: bool = True
    retry_delay_seconds: float = 0.75

    # Concurrency & caching
    rate_limit_per_conversation: bool = True
    result_cache_ttl_seconds: int = 10  # idempotency window

    # Streaming emulation
    streaming_chunk_size: int = 320  # characters per chunk

    # Telemetry & filtering
    enable_telemetry: bool = True
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

    # ── main entrypoints ------------------------------------------------------

    async def process_user_input(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NyxResponse:
        """Process with MANDATORY feasibility checks (dynamic & robust) + addiction meta (turn_index, scene_tags, stimuli)."""
        import time, uuid, re
        t0 = time.time()
        trace_id = uuid.uuid4().hex[:8]
        meta = dict(metadata or {})
    
        logger.info(f"[SDK-{trace_id}] Processing input: {message[:120]}")
    
        # --- rate limiting (unchanged) ---
        lock = None
        if getattr(self.config, "rate_limit_per_conversation", False):
            lock = self._locks.setdefault(conversation_id, asyncio.Lock())
            await lock.acquire()
    
        # --- small helpers (local, side-effect free) ---
        def _lower(s: Optional[str]) -> str:
            return (s or "").lower()
    
        def _compile_stimuli_regex() -> Optional[re.Pattern]:
            """Union of known stimuli; prefer shared vocab from addiction_system_sdk, else fallback local set."""
            tokens = None
            try:
                # Prefer the canonical mapping from the addiction SDK
                from logic.addiction_system_sdk import AddictionTriggerConfig  # type: ignore
                cfg = AddictionTriggerConfig()
                vocab = set()
                for vs in cfg.stimuli_affinity.values():
                    vocab |= set(vs or [])
                tokens = sorted(vocab)
            except Exception:
                # Minimal fallback; keep it tiny and safe to change
                tokens = [
                    "feet","toes","ankle","barefoot","sandals","flipflops","heels",
                    "perfume","musk","sweat","locker","gym","laundry","socks",
                    "ankle_socks","knee_highs","thigh_highs","stockings",
                    "hips","ass","shorts","tight_skirt",
                    "snicker","laugh","eye_roll","dismissive",
                    "order","command","kneel","obedience",
                    "perspiration","moist"
                ]
            # turn underscores into a pattern that also matches spaces/dashes
            escaped = []
            for t in tokens:
                if "_" in t:
                    escaped.append(r"\b" + re.escape(t).replace(r"\_", r"[-_ ]") + r"\b")
                else:
                    escaped.append(r"\b" + re.escape(t) + r"\b")
            return re.compile(r"(?:%s)" % "|".join(escaped), flags=re.IGNORECASE)
    
        def _extract_stimuli(text: str) -> List[str]:
            rx = getattr(self, "_stimuli_rx", None)
            if rx is None:
                rx = _compile_stimuli_regex()
                setattr(self, "_stimuli_rx", rx)
            if not rx:
                return []
            return sorted({m.group(0).lower().replace("-", "_").replace(" ", "_") for m in rx.finditer(text or "")})
    
        # --- cache key set EARLY so all early returns use the same key ---
        cache_key = self._cache_key(conversation_id, user_id, message, meta)
    
        try:
            # --- cache check ---
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(f"[SDK-{trace_id}] Cache hit")
                return cached
    
            # --- 1) Optional pre-moderation ---
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
                    logger.debug(f"[SDK-{trace_id}] pre-moderation failed softly", exc_info=True)
    
            # --- 2) MANDATORY fast feasibility gate ---
            feas = None
            try:
                from nyx.nyx_agent.feasibility import assess_action_feasibility_fast
                feas = await assess_action_feasibility_fast(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    text=message
                )
                meta["feasibility"] = feas
            except ImportError as e:
                logger.error(f"[SDK-{trace_id}] Feasibility module not found: {e}")
            except Exception as e:
                logger.error(f"[SDK-{trace_id}] Fast feasibility failed softly: {e}", exc_info=True)
    
            if isinstance(feas, dict):
                overall = feas.get("overall", {})
                feasible_flag = overall.get("feasible")
                strategy = (overall.get("strategy") or "").lower()
    
                # Hard block at SDK if 'deny'
                if feasible_flag is False and strategy == "deny":
                    per = feas.get("per_intent") or []
                    first = per[0] if per and isinstance(per[0], dict) else {}
                    guidance = first.get("narrator_guidance") or "That can't happen here."
                    alternatives = first.get("suggested_alternatives") or []
                    violations = first.get("violations") or []
    
                    resp = NyxResponse(
                        narrative=guidance,
                        choices=[{"text": alt} for alt in alternatives[:4]],
                        metadata={
                            "action_blocked": True,
                            "feasibility": feas,
                            "block_stage": "pre_orchestrator",
                            "violations": violations,
                            "strategy": "deny",
                        },
                        success=True,
                        trace_id=trace_id,
                        processing_time=time.time() - t0
                    )
                    self._write_cache(cache_key, resp)
                    return resp
    
                logger.info(f"[SDK-{trace_id}] Feasibility: feasible={feasible_flag} strategy={strategy}")
    
            # --- 2.5) Inject turn_index, scene tags, and stimuli for addiction gating ---
            # Turn indexing (persistent per (user, conversation))
            if not hasattr(self, "_turn_indices"):
                self._turn_indices: Dict[Tuple[str, str], int] = {}
            key = (str(user_id), str(conversation_id))
            turn_index = self._turn_indices.get(key, -1) + 1
            self._turn_indices[key] = turn_index
            meta["turn_index"] = turn_index
    
            # Scene tags (optional; keep your manager wiring if present)
            if "scene_tags" not in meta:
                try:
                    if hasattr(self, "scene_manager"):
                        current_scene = self.scene_manager.get_current_scene(conversation_id)
                        if current_scene:
                            tags = list(current_scene.get("tags", []))
                            if tags:
                                meta["scene_tags"] = tags
                                meta["scene"] = {"tags": tags}
                except Exception:
                    logger.debug(f"[SDK-{trace_id}] scene tag lookup failed softly", exc_info=True)
    
            # Stimuli from the current user message (plus any incoming metadata hint)
            msg_stimuli = _extract_stimuli(message)
            meta_stimuli = set(_lower(s) for s in meta.get("stimuli", []) if isinstance(s, str))
            combined_stimuli = sorted(set(msg_stimuli) | meta_stimuli)
            if combined_stimuli:
                meta["stimuli"] = combined_stimuli
    
            # --- 3) Orchestrator call with enriched meta ---
            result = await self._call_orchestrator_with_timeout(
                message=message,
                conversation_id=conversation_id,
                user_id=user_id,
                meta=meta,
            )
            resp = NyxResponse.from_orchestrator(result)
            resp.processing_time = resp.processing_time or (time.time() - t0)
            resp.trace_id = resp.trace_id or trace_id
    
            # Optionally: glean stimuli from orchestrator narrative to help next turn (cheap & safe)
            try:
                if resp and resp.narrative:
                    out_stimuli = _extract_stimuli(resp.narrative)
                    if out_stimuli:
                        # Store a short-term hint; up to you if you want to persist elsewhere
                        meta.setdefault("observed_stimuli", [])
                        meta["observed_stimuli"] = sorted({*meta["observed_stimuli"], *out_stimuli})
            except Exception:
                logger.debug(f"[SDK-{trace_id}] output stimuli glean failed softly", exc_info=True)
    
            # --- 4) Optional post-moderation ---
            if getattr(self.config, "post_moderate_output", False) and content_moderation_guardrail:
                try:
                    verdict = await content_moderation_guardrail(resp.narrative, is_output=True)
                    resp.metadata["moderation_post"] = verdict
                    if verdict and verdict.get("blocked"):
                        if getattr(self.config, "redact_on_moderation_block", False):
                            resp.narrative = verdict.get("safe_text") or "Content redacted."
                            resp.metadata["moderated"] = True
                        else:
                            resp = self._blocked_response("Content cannot be displayed.", trace_id, t0, stage="post", details=verdict)
                except Exception:
                    logger.debug("post-moderation failed (non-fatal)", exc_info=True)
    
            # --- 5) Optional response filter ---
            if getattr(self, "_filter_class", None) and resp.narrative:
                try:
                    filter_instance = self._filter_class(
                        user_id=int(user_id),
                        conversation_id=int(conversation_id),
                    )
                    filtered = filter_instance.filter_text(resp.narrative)
                    if filtered != resp.narrative:
                        resp.narrative = filtered
                        resp.metadata.setdefault("filters", {})["response_filter"] = True
                except Exception:
                    logger.debug("ResponseFilter failed softly", exc_info=True)
    
            # --- 6) Optional post hooks ---
            if getattr(self, "_post_hooks", None):
                for hook in self._post_hooks:
                    try:
                        resp = await hook(resp)
                    except Exception:
                        logger.debug("post_hook failed softly", exc_info=True)
    
            # --- 7) Telemetry & background maintenance ---
            await self._maybe_log_perf(resp)
            await self._maybe_enqueue_maintenance(resp, conversation_id)
    
            # --- 8) Cache & return ---
            self._write_cache(cache_key, resp)
            return resp
    
        except Exception as e:
            logger.error("orchestrator path failed; attempting fallback. err=%s", e, exc_info=True)
            try:
                resp = await self._fallback_direct_run(
                    message=message,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata=meta,
                    trace_id=trace_id,
                    t0=t0,
                )
    
                # Post moderation/filter/hooks even on fallback
                if getattr(self.config, "post_moderate_output", False) and content_moderation_guardrail:
                    try:
                        verdict = await content_moderation_guardrail(resp.narrative, is_output=True)
                        resp.metadata["moderation_post"] = verdict
                        if verdict and verdict.get("blocked"):
                            if getattr(self.config, "redact_on_moderation_block", False):
                                resp.narrative = verdict.get("safe_text") or "Content redacted."
                                resp.metadata["moderated"] = True
                            else:
                                resp = self._blocked_response("Content cannot be displayed.", trace_id, t0, stage="post", details=verdict)
                    except Exception:
                        logger.debug("post-moderation failed (non-fatal)", exc_info=True)
    
                if getattr(self, "_filter_class", None) and resp.narrative:
                    try:
                        filter_instance = self._filter_class(
                            user_id=int(user_id),
                            conversation_id=int(conversation_id),
                        )
                        filtered = filter_instance.filter_text(resp.narrative)
                        if filtered != resp.narrative:
                            resp.narrative = filtered
                            resp.metadata.setdefault("filters", {})["response_filter"] = True
                    except Exception:
                        logger.debug("ResponseFilter failed softly", exc_info=True)
    
                if getattr(self, "_post_hooks", None):
                    for hook in self._post_hooks:
                        try:
                            resp = await hook(resp)
                        except Exception:
                            logger.debug("post_hook failed softly", exc_info=True)
    
                await self._maybe_log_perf(resp)
                await self._maybe_enqueue_maintenance(resp, conversation_id)
    
                self._write_cache(cache_key, resp)
                return resp
    
            except Exception as e:
                logger.error(f"[SDK-{trace_id}] Process failed", exc_info=True)
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

    async def _call_orchestrator_with_timeout(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call orchestrator with timeout and (optional) single retry."""
        try:
            return await asyncio.wait_for(
                _orchestrator_process(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    user_input=message,
                    context_data=meta,
                ),
                timeout=self.config.request_timeout_seconds,
            )
        except Exception as first_error:
            logger.warning("orchestrator call failed: %s", first_error)
            if not self.config.retry_on_failure:
                raise
            await asyncio.sleep(self.config.retry_delay_seconds)
            logger.info("retrying orchestrator once…")
            return await asyncio.wait_for(
                _orchestrator_process(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    user_input=message,
                    context_data=meta,
                ),
                timeout=self.config.request_timeout_seconds,
            )

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
        Minimal reproduction of the orchestrator flow using public agent APIs.
        Only invoked if the primary path fails and the runtime is available.
        We avoid importing the class-based assembler; instead we safely coalesce
        the runner history into a single narrative.
        """
        if not (Runner and RunConfig and RunContextWrapper and nyx_main_agent):
            raise RuntimeError("Fallback agent runtime is unavailable in this environment.")

        ctx = NyxContext(user_id=int(user_id), conversation_id=int(conversation_id))
        await ctx.initialize()
        ctx.current_context = (metadata or {}).copy()
        ctx.current_context["user_input"] = message

        safe_settings = ModelSettings(strict_tools=False, response_format=None)
        run_config = RunConfig(model_settings=safe_settings, workflow_name="Nyx Fallback")
        runner_context = RunContextWrapper(ctx)

        result = await Runner.run(nyx_main_agent, message, context=runner_context, run_config=run_config)

        # 1) Extract best-effort text from the run
        def _extract_text(obj: Any) -> str:
            for attr in ("final_output", "output_text", "output", "text"):
                val = getattr(obj, attr, None)
                if isinstance(val, str) and val.strip():
                    return val
            # Try messages/history if present
            for attr in ("messages", "history", "events"):
                seq = getattr(obj, attr, None) or []
                # look for assistant content blocks
                for ev in seq:
                    content = ev.get("content") if isinstance(ev, dict) else None
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in {"output_text", "text"}:
                                if isinstance(c.get("text"), str):
                                    return c["text"]
            # final fallback
            return str(obj)

        narrative = _extract_text(result) or ""

        # 2) Minimal image/world placeholders to keep API shape
        meta = {
            "world": {},
            "image": {},
            "telemetry": {},
            "universal_updates": False,
            "path": "fallback",
            "trace_id": trace_id,
        }

        return NyxResponse(
            narrative=narrative,
            choices=[],
            metadata=meta,
            world_state={},
            success=True,
            trace_id=trace_id,
            processing_time=(time.time() - t0),
        )

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

    def _blocked_response(self, safe_text: str, trace_id: str, t0: float, stage: str, details: Optional[Dict[str, Any]]) -> NyxResponse:
        return NyxResponse(
            narrative=safe_text,
            metadata={"moderation_blocked": True, "stage": stage, "details": details or {}},
            success=False,
            trace_id=trace_id,
            processing_time=(time.time() - t0),
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
