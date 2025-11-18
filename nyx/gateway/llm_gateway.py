"""Gateway helpers for orchestrating Nyx LLM agent executions.

The helpers in this module expose a high-level interface around
``agents.Runner`` that is safe to use from non-core Nyx packages.  The goal is
to encapsulate retry/backoff behaviour, consistent logging, metadata
collection, and dynamic agent resolution without importing Nyx core modules.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import random
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from enum import Enum
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Optional, Protocol

import httpx

from nyx.telemetry.metrics import LLM_TOKENS_IN, LLM_TOKENS_OUT, REQUEST_LATENCY
from nyx.telemetry.tracing import trace_step

logger = logging.getLogger(__name__)

class SupportsAgent(Protocol):
    """Protocol capturing the minimal surface of an agent object."""

    name: str
    model: Any


AgentSpec = SupportsAgent | Callable[[], SupportsAgent] | str


@dataclass(slots=True)
class LLMRequest:
    """Describe a request that should be executed via an agent.

    Attributes:
        prompt: The user prompt to send to ``Runner``.
        agent: Primary agent specification.  Accepts either an agent instance,
            a callable returning an agent, or a string (``module:attr`` or a
            known short name) which will be resolved dynamically.
        context: Optional context forwarded to ``Runner.run``.
        metadata: Arbitrary metadata that will be merged into the final result
            object and logged alongside execution events.
        runner_kwargs: Keyword arguments forwarded directly to
            ``Runner.run``.  This allows the caller to opt into additional SDK
            behaviour (e.g. ``run_config`` or ``hooks``) without the gateway
            needing to know about them.
        fallback_agent: Optional secondary agent specification used when the
            primary agent fails to resolve or execution errors even after
            retrying.
        max_attempts: Number of retry attempts per agent before failing over to
            a fallback or raising the last error.
        backoff_initial: Base sleep duration in seconds for retries.
        backoff_multiplier: Exponential multiplier applied after each attempt.
        backoff_jitter: Optional random jitter range (seconds) added to the
            computed backoff value to prevent thundering herds.
        retryable_exceptions: Tuple of exception classes that should trigger a
            retry.  Defaults to ``Exception`` meaning all errors are retryable
            unless the caller provides a narrower tuple.
    """

    prompt: Any
    agent: AgentSpec
    context: Any | None = None
    metadata: Mapping[str, Any] | None = None
    runner_kwargs: Mapping[str, Any] | None = None
    model_override: str | None = None
    fallback_agent: AgentSpec | None = None
    max_attempts: int = 3
    backoff_initial: float = 0.5
    backoff_multiplier: float = 2.0
    backoff_jitter: float = 0.2
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)

    def with_agent(self, agent: AgentSpec) -> "LLMRequest":
        """Return a new request with the supplied agent."""

        return LLMRequest(
            prompt=self.prompt,
            agent=agent,
            context=self.context,
            metadata=self.metadata,
            runner_kwargs=self.runner_kwargs,
            model_override=self.model_override,
            fallback_agent=self.fallback_agent,
            max_attempts=self.max_attempts,
            backoff_initial=self.backoff_initial,
            backoff_multiplier=self.backoff_multiplier,
            backoff_jitter=self.backoff_jitter,
            retryable_exceptions=self.retryable_exceptions,
        )


@dataclass(slots=True)
class LLMResult:
    """Normalised response returned by :func:`execute`."""

    text: str
    raw: Any
    agent_name: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    used_fallback: bool = False
    duration: float | None = None

    def __bool__(self) -> bool:  # pragma: no cover - legacy compat helper
        return bool(self.text or self.raw)


_runner: type | None = None
_run_config_cls: type | None = None

_PROMPT_PREVIEW_MAX_CHARS = 256
_SENSITIVE_PROMPT_FLAGS = (
    "redact_prompt",
    "sensitive_prompt",
    "contains_sensitive_data",
    "sensitive",
)


class LLMOperation(str, Enum):
    """Enumerate high-level operations routed through the LLM gateway."""

    ORCHESTRATION = "orchestration"
    DEFER_RESPONSE = "defer_response"


def _lazy_import_runner() -> type:
    """Import ``Runner`` lazily to avoid touching Nyx core packages."""

    global _runner
    if _runner is None:
        from agents import Runner  # type: ignore

        _runner = Runner
    return _runner


def _lazy_import_run_config() -> type:
    """Import ``RunConfig`` lazily to avoid heavy agent SDK imports."""

    global _run_config_cls
    if _run_config_cls is None:
        from agents import RunConfig  # type: ignore

        _run_config_cls = RunConfig
    return _run_config_cls


def _iter_candidate_agents(request: LLMRequest) -> Iterable[tuple[AgentSpec, bool]]:
    primary = request.agent
    yield primary, False
    if request.fallback_agent is not None:
        yield request.fallback_agent, True


def _resolve_agent(agent: AgentSpec) -> SupportsAgent:
    if callable(agent) and not isinstance(agent, str):
        resolved = agent()
        if resolved is None:
            raise ValueError("Agent factory returned None")
        return resolved
    if isinstance(agent, str):
        resolved = _resolve_agent_by_name(agent)
        if resolved is None:
            raise LookupError(f"Could not resolve agent '{agent}'")
        return resolved
    return agent


def _resolve_agent_by_name(name: str) -> SupportsAgent | None:
    module_name: str | None = None
    attr_name = name
    if ":" in name:
        module_name, attr_name = name.split(":", 1)
    elif "." in name:
        module_name, attr_name = name.rsplit(".", 1)

    candidates: list[tuple[str, str]] = []
    if module_name:
        candidates.append((module_name, attr_name))
    else:
        candidates.extend(
            (
                "nyx.nyx_agent.agents",
                attr_name,
            ),
            (
                "nyx.nyx_agent.agent_factory",
                attr_name,
            ),
            (
                "nyx.nyx_agent",
                attr_name,
            ),
        )

    for mod_name, attribute in candidates:
        try:
            module = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(module, attribute):
            resolved = getattr(module, attribute)
            if callable(resolved) and not isinstance(resolved, type):
                try:
                    candidate = resolved()
                except TypeError:
                    # Non-nullary callable; skip
                    continue
                if candidate is None:
                    continue
                return candidate
            return resolved
    return None


def _extract_text(result: Any) -> str:
    if result is None:
        return ""
    for attr in ("final_output", "output", "text", "content"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, str):
                return value
            if isinstance(value, Sequence) and value and isinstance(value[0], str):
                return value[0]
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping) and "text" in result:
        text_val = result.get("text")
        if isinstance(text_val, str):
            return text_val
    if isinstance(result, SimpleNamespace) and hasattr(result, "final_output"):
        value = getattr(result, "final_output")
        if isinstance(value, str):
            return value
    return ""


def _extract_metadata(result: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for attr in ("usage", "response_id", "id", "model", "raw"):
        if hasattr(result, attr):
            meta[attr] = getattr(result, attr)
    if isinstance(result, Mapping):
        meta.update({k: result[k] for k in ("usage", "response_id") if k in result})
    return meta


def _stringify_prompt(prompt: Any) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    try:
        return str(prompt)
    except Exception:
        return "<unprintable prompt>"


def _build_prompt_preview(
    prompt: Any, metadata: Mapping[str, Any] | None
) -> tuple[str, int]:
    prompt_text = _stringify_prompt(prompt)
    prompt_length = len(prompt_text)
    meta = metadata or {}
    if not prompt_text:
        return "", prompt_length
    if meta.get("log_prompt_preview") is False:
        return "<suppressed>", prompt_length
    if any(bool(meta.get(flag)) for flag in _SENSITIVE_PROMPT_FLAGS):
        return "<redacted>", prompt_length
    sanitized = prompt_text.replace("\n", "\\n").strip()
    if len(sanitized) > _PROMPT_PREVIEW_MAX_CHARS:
        sanitized = sanitized[: _PROMPT_PREVIEW_MAX_CHARS - 1] + "…"
    return sanitized, prompt_length


def _coerce_count(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summarize_agents_run(raw_result: Any) -> dict[str, int]:
    summary: dict[str, int] = {}
    containers: list[Any] = [raw_result]
    for attr in ("raw", "metadata", "summary", "stats", "run", "details"):
        if hasattr(raw_result, attr):
            candidate = getattr(raw_result, attr)
            if candidate is not None:
                containers.append(candidate)
    if isinstance(raw_result, Mapping):
        for key in ("summary", "stats", "run", "details"):
            candidate = raw_result.get(key)
            if candidate is not None:
                containers.append(candidate)

    def _lookup(keys: tuple[str, ...]) -> Optional[int]:
        for container in containers:
            if container is None:
                continue
            for key in keys:
                value = None
                if isinstance(container, Mapping) and key in container:
                    value = container.get(key)
                elif hasattr(container, key):
                    value = getattr(container, key)
                if value is None:
                    continue
                count = _coerce_count(value)
                if count is not None:
                    return count
        return None

    turn_count = _lookup(("turn_count", "turns", "turns_taken", "iterations"))
    if turn_count is not None:
        summary["turn_count"] = turn_count

    tool_count = _lookup(
        (
            "tool_calls",
            "tool_invocations",
            "tool_uses",
            "tools_called",
            "function_calls",
        )
    )
    if tool_count is not None:
        summary["tool_call_count"] = tool_count

    return summary


def _count_guardrail_invocations(metadata: Mapping[str, Any] | None) -> int:
    if not metadata:
        return 0
    guardrail_keys = (
        "guardrail",
        "guardrails",
        "guardrail_hits",
        "guardrail_events",
        "input_guardrails",
        "output_guardrails",
        "moderation_pre",
        "moderation_post",
    )

    def _count_value(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, Mapping):
            return len(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return len(value)
        if isinstance(value, SimpleNamespace):
            return len(vars(value)) or 1
        return 1

    total = 0
    for key in guardrail_keys:
        if key in metadata:
            total += _count_value(metadata.get(key))
    return total


async def _run_once(
    agent: SupportsAgent,
    request: LLMRequest,
    *,
    runner_cls: type,
    attempt: int,
    use_fallback: bool,
) -> LLMResult:
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _lookup_usage(container: Any, *keys: str) -> Optional[int]:
        if container is None:
            return None
        for key in keys:
            candidate = None
            if isinstance(container, Mapping) and key in container:
                candidate = container.get(key)
            elif hasattr(container, key):
                candidate = getattr(container, key)
            if candidate is not None:
                parsed = _safe_int(candidate)
                if parsed is not None:
                    return parsed
        return None

    start = time.perf_counter()
    runner_kwargs = dict(request.runner_kwargs or {})
    if request.model_override:
        run_config_cls = _lazy_import_run_config()
        run_config = runner_kwargs.get("run_config")
        if run_config is None:
            runner_kwargs["run_config"] = run_config_cls(model=request.model_override)
        elif isinstance(run_config, run_config_cls):
            runner_kwargs["run_config"] = replace(run_config, model=request.model_override)
        elif hasattr(run_config, "__dataclass_fields__"):
            try:
                runner_kwargs["run_config"] = replace(run_config, model=request.model_override)
            except TypeError:
                runner_kwargs["run_config"] = run_config_cls(model=request.model_override)
        else:
            try:
                runner_kwargs["run_config"] = replace(run_config, model=request.model_override)  # type: ignore[arg-type]
            except Exception:
                runner_kwargs["run_config"] = run_config_cls(model=request.model_override)
    runner_context = runner_kwargs.pop("context", None)
    context = request.context if request.context is not None else runner_context
    if runner_context is not None and request.context is not None:
        logger.warning(
            "nyx.gateway.llm: runner_kwargs contained 'context'; using request.context and dropping the duplicate."
        )
    metadata = dict(request.metadata or {})
    operation = str(metadata.get("operation") or "unknown")
    trace_id = metadata.get("trace_id")
    subsystem = str(
        metadata.get("subsystem")
        or metadata.get("source_subsystem")
        or "unknown"
    )
    prompt_preview, prompt_length = _build_prompt_preview(request.prompt, metadata)

    model_name: str | None = None
    run_config = runner_kwargs.get("run_config")
    if run_config is not None:
        model_name = getattr(run_config, "model", None)
    if model_name is None:
        model_name = request.model_override or metadata.get("model") or getattr(agent, "model", None)
    model_label = str(model_name) if model_name else "unknown"

    log_payload = {
        "agent": getattr(agent, "name", str(agent)),
        "attempt": attempt,
        "fallback": use_fallback,
        "metadata_keys": sorted(str(k) for k in metadata.keys()),
        "operation": operation,
        "subsystem": subsystem,
        "model": model_label,
        "trace_id": trace_id,
        "prompt_preview": prompt_preview,
        "prompt_length": prompt_length,
    }
    logger.info("nyx.gateway.llm.execute.start", extra=log_payload)

    with trace_step(
        "nyx.gateway.llm.run",
        trace_id,
        agent=getattr(agent, "name", str(agent)),
        attempt=attempt,
        fallback=use_fallback,
        operation=operation,
    ):
        raw_result = await runner_cls.run(
            agent,
            request.prompt,
            context=context,
            **runner_kwargs,
        )

    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(component="llm_gateway").observe(duration)

    payload_metadata = {**metadata, **_extract_metadata(raw_result)}
    text = _extract_text(raw_result)

    usage_source = getattr(raw_result, "usage", None)
    if usage_source is None and isinstance(raw_result, Mapping):
        usage_source = raw_result.get("usage")
    if usage_source is None:
        usage_source = metadata.get("usage")
    prompt_tokens = _lookup_usage(usage_source, "prompt_tokens", "input_tokens")
    completion_tokens = _lookup_usage(usage_source, "completion_tokens", "output_tokens")
    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    if prompt_tokens is not None:
        LLM_TOKENS_IN.labels(operation=operation).inc(prompt_tokens)
    if completion_tokens is not None:
        LLM_TOKENS_OUT.labels(operation=operation).inc(completion_tokens)

    guardrail_invocations = _count_guardrail_invocations(payload_metadata)
    run_summary = _summarize_agents_run(raw_result)

    result = LLMResult(
        text=text,
        raw=raw_result,
        agent_name=getattr(agent, "name", None),
        metadata=payload_metadata,
        attempts=attempt,
        used_fallback=use_fallback,
        duration=duration,
    )
    success_payload = {**log_payload, "duration": duration, "has_text": bool(text), **run_summary}
    if prompt_tokens is not None:
        success_payload["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        success_payload["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        success_payload["total_tokens"] = total_tokens
    if guardrail_invocations:
        success_payload["guardrail_invocations"] = guardrail_invocations
    logger.info("nyx.gateway.llm.execute.success", extra=success_payload)
    return result


async def execute(request: LLMRequest) -> LLMResult:
    """Execute the supplied request, returning an :class:`LLMResult`."""

    runner_cls = _lazy_import_runner()
    last_error: BaseException | None = None
    request_metadata = dict(request.metadata or {})
    request_operation = str(request_metadata.get("operation") or "unknown")
    request_subsystem = str(
        request_metadata.get("subsystem")
        or request_metadata.get("source_subsystem")
        or "unknown"
    )
    request_model_label = str(
        request.model_override
        or request_metadata.get("model")
        or request_metadata.get("preferred_model")
        or "unknown"
    )

    for spec, is_fallback in _iter_candidate_agents(request):
        try:
            agent = _resolve_agent(spec)
        except Exception as exc:
            logger.exception(
                "nyx.gateway.llm.resolve_failed",
                extra={"agent_spec": str(spec), "fallback": is_fallback},
            )
            last_error = exc
            continue

        for attempt in range(1, max(request.max_attempts, 1) + 1):
            attempt_start = time.perf_counter()
            try:
                return await _run_once(
                    agent,
                    request,
                    runner_cls=runner_cls,
                    attempt=attempt,
                    use_fallback=is_fallback,
                )
            except request.retryable_exceptions as exc:  # type: ignore[misc]
                last_error = exc
                elapsed = time.perf_counter() - attempt_start
                attempt_payload = {
                    "agent": getattr(agent, "name", str(agent)),
                    "attempt": attempt,
                    "fallback": is_fallback,
                    "operation": request_operation,
                    "subsystem": request_subsystem,
                    "model": request_model_label,
                    "duration": elapsed,
                    "error_type": type(exc).__name__,
                    "error_str": str(exc)[:300],
                }
                if attempt >= request.max_attempts:
                    logger.exception(
                        "nyx.gateway.llm.execute.failed",
                        extra=attempt_payload,
                    )
                    break
                delay = request.backoff_initial * (request.backoff_multiplier ** (attempt - 1))
                if request.backoff_jitter:
                    delay += random.uniform(0, request.backoff_jitter)
                logger.warning(
                    "nyx.gateway.llm.execute.retry",
                    extra={**attempt_payload, "sleep": round(delay, 3)},
                )
                await asyncio.sleep(delay)
            except Exception as exc:  # pragma: no cover - safeguard for unexpected errors
                last_error = exc
                elapsed = time.perf_counter() - attempt_start
                logger.exception(
                    "nyx.gateway.llm.execute.unexpected_failure",
                    extra={
                        "agent": getattr(agent, "name", str(agent)),
                        "attempt": attempt,
                        "fallback": is_fallback,
                        "operation": request_operation,
                        "subsystem": request_subsystem,
                        "model": request_model_label,
                        "duration": elapsed,
                        "error_type": type(exc).__name__,
                        "error_str": str(exc)[:300],
                    },
                )
                break
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM execution failed without an explicit exception")


async def execute_stream(request: LLMRequest) -> AsyncIterator[LLMResult]:
    """Streaming variant of :func:`execute` that yields a single result.

    ``agents.Runner`` does not expose a native streaming API in this project,
    so the function simply delegates to :func:`execute` and yields the final
    response.  The async generator signature keeps the call-site compatible
    with future streaming implementations.
    """

    result = await execute(request)
    yield result
