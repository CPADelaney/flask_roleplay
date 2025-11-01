"""Structured tracing helpers for Nyx hot-path instrumentation."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def _clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``None`` values to keep logs compact."""

    return {key: value for key, value in payload.items() if value is not None}


class TraceSpan:
    """Context manager that emits structured start/stop records."""

    def __init__(self, name: str, trace_id: Optional[str] = None, **fields: Any) -> None:
        self.name = name
        self.trace_id = trace_id
        self.fields = fields
        self._start_wall = time.time()
        self._start_perf = time.perf_counter()

    def _log(self, event: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "event": event,
            "step": self.name,
            "trace_id": self.trace_id,
            "ts": time.time(),
            **self.fields,
            **extra,
        }
        logger.info(json.dumps(_clean_payload(payload), sort_keys=True))

    def mark(self, event: str, **extra: Any) -> None:
        """Emit an intermediate event inside the span."""

        self._log(event, **extra)

    def __enter__(self) -> "TraceSpan":
        self._log("start", started_at=self._start_wall)
        return self

    def __exit__(self, exc_type, exc: Any, _tb: Any) -> bool:
        duration = time.perf_counter() - self._start_perf
        status = "error" if exc_type else "ok"
        extra: Dict[str, Any] = {"duration": duration, "status": status}
        if exc is not None:
            extra["error"] = getattr(exc, "__class__", type(exc)).__name__
        self._log("end", **extra)
        # Do not suppress exceptions
        return False


@contextmanager
def trace_step(name: str, trace_id: Optional[str] = None, **fields: Any) -> Iterator[TraceSpan]:
    """Create a :class:`TraceSpan` context manager for ad-hoc tracing."""

    span = TraceSpan(name, trace_id=trace_id, **fields)
    try:
        yield span.__enter__()
    except Exception as exc:
        span.__exit__(exc.__class__, exc, exc.__traceback__)
        raise
    else:
        span.__exit__(None, None, None)
