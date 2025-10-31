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
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol

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


def _lazy_import_runner() -> type:
    """Import ``Runner`` lazily to avoid touching Nyx core packages."""

    global _runner
    if _runner is None:
        from agents import Runner  # type: ignore

        _runner = Runner
    return _runner


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


async def _run_once(
    agent: SupportsAgent,
    request: LLMRequest,
    *,
    runner_cls: type,
    attempt: int,
    use_fallback: bool,
) -> LLMResult:
    start = time.perf_counter()
    runner_kwargs = dict(request.runner_kwargs or {})
    context = request.context
    metadata = dict(request.metadata or {})

    log_payload = {
        "agent": getattr(agent, "name", str(agent)),
        "attempt": attempt,
        "fallback": use_fallback,
        "metadata_keys": sorted(str(k) for k in metadata.keys()),
    }
    logger.info("nyx.gateway.llm.execute.start", extra=log_payload)

    raw_result = await runner_cls.run(
        agent,
        request.prompt,
        context=context,
        **runner_kwargs,
    )

    duration = time.perf_counter() - start
    payload_metadata = {**metadata, **_extract_metadata(raw_result)}
    text = _extract_text(raw_result)
    result = LLMResult(
        text=text,
        raw=raw_result,
        agent_name=getattr(agent, "name", None),
        metadata=payload_metadata,
        attempts=attempt,
        used_fallback=use_fallback,
        duration=duration,
    )
    logger.info(
        "nyx.gateway.llm.execute.success",
        extra={**log_payload, "duration": duration, "has_text": bool(text)},
    )
    return result


async def execute(request: LLMRequest) -> LLMResult:
    """Execute the supplied request, returning an :class:`LLMResult`."""

    runner_cls = _lazy_import_runner()
    last_error: BaseException | None = None

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
                if attempt >= request.max_attempts:
                    logger.exception(
                        "nyx.gateway.llm.execute.failed",
                        extra={
                            "agent": getattr(agent, "name", str(agent)),
                            "attempt": attempt,
                            "fallback": is_fallback,
                        },
                    )
                    break
                delay = request.backoff_initial * (request.backoff_multiplier ** (attempt - 1))
                if request.backoff_jitter:
                    delay += random.uniform(0, request.backoff_jitter)
                logger.warning(
                    "nyx.gateway.llm.execute.retry",
                    extra={
                        "agent": getattr(agent, "name", str(agent)),
                        "attempt": attempt,
                        "fallback": is_fallback,
                        "sleep": round(delay, 3),
                    },
                )
                await asyncio.sleep(delay)
            except Exception as exc:  # pragma: no cover - safeguard for unexpected errors
                last_error = exc
                logger.exception(
                    "nyx.gateway.llm.execute.unexpected_failure",
                    extra={
                        "agent": getattr(agent, "name", str(agent)),
                        "attempt": attempt,
                        "fallback": is_fallback,
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
