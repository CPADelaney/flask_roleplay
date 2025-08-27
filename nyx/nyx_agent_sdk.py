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
from .nyx_agent.assembly import assemble_nyx_response, resolve_scene_requests  # fallback assembler

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
    from .response_filter import ResponseFilter
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
        self._filter = ResponseFilter() if (ResponseFilter and self.config.enable_response_filter) else None
        # optional per-conversation context kept only when explicitly warmed
        self._warm_contexts: Dict[str, NyxContext] = {}

    # ── lifecycle -------------------------------------------------------------

    async def initialize_agent(self) -> None:
        """Kept for historical symmetry; orchestrator builds its own agents."""
        return None

    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Free rate-limit lock, cached results, and any warmed context for a conversation."""
        self._locks.pop(conversation_id, None)
        self._warm_contexts.pop(conversation_id, None)
        keys = [k for k in self._result_cache if k.startswith(f"{conversation_id}|")]
        for k in keys:
            self._result_cache.pop(k, None)

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
        """
        Final-stop pipeline:
        1) (opt) pre-moderation on input
        2) per-conversation rate-limit + idempotency cache
        3) orchestrator call with timeout (+ single retry)
        4) normalize to NyxResponse
        5) (opt) post-moderation on output
        6) (opt) response filter
        7) (opt) post hooks
        8) (opt) telemetry + background tasks
        """
        t0 = time.time()
        trace_id = uuid.uuid4().hex[:8]
        meta = metadata or {}

        # rate-limit per conversation (optional)
        lock = None
        if self.config.rate_limit_per_conversation:
            lock = self._locks.setdefault(conversation_id, asyncio.Lock())
            await lock.acquire()  # release in finally

        # idempotency cache (short horizon)
        cache_key = self._cache_key(conversation_id, user_id, message, meta)
        cached = self._read_cache(cache_key)
        if cached:
            cached.processing_time = cached.processing_time or (time.time() - t0)
            if lock and lock.locked():
                lock.release()
            return cached

        try:
            # 1) PRE moderation (optional)
            if self.config.pre_moderate_input and content_moderation_guardrail:
                try:
                    verdict = await content_moderation_guardrail(message)
                    if verdict and verdict.get("blocked"):
                        safe_text = verdict.get("safe_text") or "Your message couldn't be processed."
                        resp = self._blocked_response(safe_text, trace_id, t0, stage="pre", details=verdict)
                        self._write_cache(cache_key, resp)
                        return resp
                    else:
                        meta = {**meta, "moderation_pre": verdict}
                except Exception:
                    logger.debug("pre-moderation failed (non-fatal)", exc_info=True)

            # 2) Primary path with timeout + retry
            result = await self._call_orchestrator_with_timeout(
                message=message,
                conversation_id=conversation_id,
                user_id=user_id,
                meta=meta,
            )
            resp = NyxResponse.from_orchestrator(result)
            resp.processing_time = resp.processing_time or (time.time() - t0)
            resp.trace_id = resp.trace_id or trace_id

            # 3) POST moderation (optional)
            if self.config.post_moderate_output and content_moderation_guardrail:
                try:
                    verdict = await content_moderation_guardrail(resp.narrative, is_output=True)
                    resp.metadata["moderation_post"] = verdict
                    if verdict and verdict.get("blocked"):
                        if self.config.redact_on_moderation_block:
                            resp.narrative = verdict.get("safe_text") or "Content redacted."
                            resp.metadata["moderated"] = True
                        else:
                            resp = self._blocked_response("Content cannot be displayed.", trace_id, t0, stage="post", details=verdict)
                except Exception:
                    logger.debug("post-moderation failed (non-fatal)", exc_info=True)

            # 4) Response filter (optional)
            if self._filter and resp.narrative:
                try:
                    filtered = self._filter.filter_text(resp.narrative)
                    if filtered != resp.narrative:
                        resp.narrative = filtered
                        resp.metadata.setdefault("filters", {})["response_filter"] = True
                except Exception:
                    logger.debug("ResponseFilter failed softly", exc_info=True)

            # 5) Post hooks (optional)
            if self._post_hooks:
                for hook in self._post_hooks:
                    try:
                        resp = await hook(resp)
                    except Exception:
                        logger.debug("post_hook failed softly", exc_info=True)

            # 6) Telemetry + background queue (optional)
            await self._maybe_log_perf(resp)
            await self._maybe_enqueue_maintenance(resp, conversation_id)

            # 7) Cache and return
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
                if self.config.post_moderate_output and content_moderation_guardrail:
                    try:
                        verdict = await content_moderation_guardrail(resp.narrative, is_output=True)
                        resp.metadata["moderation_post"] = verdict
                        if verdict and verdict.get("blocked"):
                            if self.config.redact_on_moderation_block:
                                resp.narrative = verdict.get("safe_text") or "Content redacted."
                                resp.metadata["moderated"] = True
                            else:
                                resp = self._blocked_response("Content cannot be displayed.", trace_id, t0, stage="post", details=verdict)
                    except Exception:
                        logger.debug("post-moderation failed (non-fatal)", exc_info=True)

                if self._filter and resp.narrative:
                    try:
                        filtered = self._filter.filter_text(resp.narrative)
                        if filtered != resp.narrative:
                            resp.narrative = filtered
                            resp.metadata.setdefault("filters", {})["response_filter"] = True
                    except Exception:
                        logger.debug("ResponseFilter failed softly", exc_info=True)

                if self._post_hooks:
                    for hook in self._post_hooks:
                        try:
                            resp = await hook(resp)
                        except Exception:
                            logger.debug("post_hook failed softly", exc_info=True)

                await self._maybe_log_perf(resp)
                await self._maybe_enqueue_maintenance(resp, conversation_id)

                self._write_cache(cache_key, resp)
                return resp

            except Exception as e2:
                logger.error("fallback path also failed", exc_info=True)
                resp = self._error_response(str(e2), trace_id, t0)
                self._write_cache(cache_key, resp)
                return resp

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

        history = []
        for attr in ("messages", "history", "events"):
            if hasattr(result, attr):
                history = getattr(result, attr) or []
                if history:
                    break

        if not history:
            text = (
                getattr(result, "final_output", None)
                or getattr(result, "output", None)
                or getattr(result, "text", None)
                or str(result)
            )
            history = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]

        history = await resolve_scene_requests(history, ctx)
        assembled = assemble_nyx_response(history)

        return NyxResponse(
            narrative=assembled.get("narrative", "") or "",
            choices=assembled.get("choices", []) or [],
            metadata={
                "world": assembled.get("world", {}) or {},
                "emergent": assembled.get("emergent", {}) or {},
                "image": assembled.get("image", {}) or {},
                "telemetry": assembled.get("telemetry", {}) or {},
                "nyx_commentary": assembled.get("nyx_commentary"),
                "universal_updates": assembled.get("universal_updates", False),
                "path": "fallback",
                "trace_id": trace_id,
            },
            world_state=assembled.get("world", {}) or {},
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
            narrative="*Nyx’s form flickers* Something interfered with our connection…",
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
]
