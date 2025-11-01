"""Prometheus metrics helpers for Nyx hot-path observability."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised in environments without prometheus_client
    from prometheus_client import Counter, Histogram  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback

    class _NoopMetric:
        """Graceful fallback collector when prometheus_client is unavailable."""

        def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
            return self

        def observe(self, value: float) -> None:
            return None

        def inc(self, amount: float = 1.0) -> None:
            return None

        @contextmanager
        def time(self):
            yield None

    def Counter(*args: Any, **kwargs: Any):  # type: ignore
        return _NoopMetric()

    def Histogram(*args: Any, **kwargs: Any):  # type: ignore
        return _NoopMetric()

REQUEST_LATENCY = Histogram(
    "nyx_request_latency_seconds",
    "Latency for critical Nyx request paths",
    labelnames=("component",),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)

TTFB_SECONDS = Histogram(
    "nyx_ttfb_seconds",
    "Time-to-first-byte (or chunk) for streaming responses",
    labelnames=("channel",),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)

QUEUE_DELAY_SECONDS = Histogram(
    "nyx_queue_delay_seconds",
    "Observed delay between enqueue time and worker start",
    labelnames=("queue",),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

LLM_TOKENS_IN = Counter(
    "nyx_llm_tokens_in_total",
    "Prompt tokens consumed by LLM operations",
    ["operation"],
)

LLM_TOKENS_OUT = Counter(
    "nyx_llm_tokens_out_total",
    "Completion tokens produced by LLM operations",
    ["operation"],
)

CACHE_HIT = Counter(
    "nyx_cache_hit_total",
    "Cache hits for Nyx hot-path caches",
    ["section"],
)

CACHE_MISS = Counter(
    "nyx_cache_miss_total",
    "Cache misses for Nyx hot-path caches",
    ["section"],
)

TASK_FAILURES = Counter(
    "nyx_task_failures_total",
    "Task or orchestrator failures by reason",
    ["task", "reason"],
)

_QUEUE_CANDIDATE_KEYS: Sequence[str] = (
    "enqueued_at",
    "enqueuedAt",
    "queued_at",
    "queue_enqueued_at",
    "_enqueued_at",
)

_NESTED_CONTEXT_KEYS: Sequence[str] = (
    "headers",
    "meta",
    "_meta",
    "options",
    "delivery_info",
)


def _coerce_timestamp(value: Any) -> Optional[float]:
    """Return a UNIX timestamp for known enqueued_at value formats."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            normalised = text.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalised).timestamp()
            except ValueError:
                logger.debug("Unable to parse enqueued_at timestamp: %s", text)
                return None
    if isinstance(value, Mapping):
        for key in ("ts", "timestamp", "enqueued_at", "enqueuedAt"):
            if key in value:
                return _coerce_timestamp(value[key])
    return None


def _extract_enqueued_value(context: Mapping[str, Any]) -> Optional[Any]:
    for key in _QUEUE_CANDIDATE_KEYS:
        if key in context:
            return context[key]
    for nested_key in _NESTED_CONTEXT_KEYS:
        nested = context.get(nested_key)
        if isinstance(nested, Mapping):
            nested_value = _extract_enqueued_value(nested)
            if nested_value is not None:
                return nested_value
    return None


def record_queue_delay(value: Any, *, queue: str = "default", now: Optional[float] = None) -> Optional[float]:
    """Record queue delay for a raw enqueued_at value."""

    timestamp = _coerce_timestamp(value)
    if timestamp is None:
        return None
    current = now if now is not None else time.time()
    delay = max(0.0, float(current) - float(timestamp))
    QUEUE_DELAY_SECONDS.labels(queue=queue).observe(delay)
    return delay


def record_queue_delay_from_context(
    context: Optional[Mapping[str, Any]],
    *,
    queue: str = "default",
    now: Optional[float] = None,
) -> Optional[float]:
    """Inspect a context mapping (e.g. Celery headers) and record queue delay."""

    if not isinstance(context, Mapping):
        return None
    value = _extract_enqueued_value(context)
    if value is None:
        return None
    return record_queue_delay(value, queue=queue, now=now)


__all__ = [
    "REQUEST_LATENCY",
    "TTFB_SECONDS",
    "QUEUE_DELAY_SECONDS",
    "LLM_TOKENS_IN",
    "LLM_TOKENS_OUT",
    "CACHE_HIT",
    "CACHE_MISS",
    "TASK_FAILURES",
    "record_queue_delay",
    "record_queue_delay_from_context",
]
