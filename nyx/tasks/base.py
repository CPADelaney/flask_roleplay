"""Celery task utilities for Nyx."""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Mapping, MutableMapping

from celery import Task


TRACE_ID_HEADER = "nyx-trace-id"
_TRACE_ID_CTX: ContextVar[str | None] = ContextVar("nyx_trace_id", default=None)

def current_trace_id() -> str | None:
    """Return the trace identifier for the current task context."""

    return _TRACE_ID_CTX.get()


class NyxTask(Task):
    """Base task with trace propagation and structured logging."""

    abstract = True
    autoretry_for = (TimeoutError, ConnectionError)
    retry_kwargs = {"max_retries": 3, "countdown": 2}

    def apply_async(self, args: Any | None = None, kwargs: Dict[str, Any] | None = None, **options: Any):
        kwargs = dict(kwargs or {})
        headers: MutableMapping[str, Any] = options.setdefault("headers", {})

        trace_id = kwargs.get("trace_id") or headers.get(TRACE_ID_HEADER) or current_trace_id()
        if trace_id is None:
            trace_id = uuid.uuid4().hex

        headers.setdefault(TRACE_ID_HEADER, trace_id)
        kwargs.setdefault("trace_id", trace_id)

        return super().apply_async(args=args, kwargs=kwargs, **options)

    def __call__(self, *args: Any, **kwargs: Any):
        mutable_kwargs = dict(kwargs)

        trace_id = self._resolve_trace_id(mutable_kwargs)
        token = _TRACE_ID_CTX.set(trace_id)

        logger = logging.getLogger(self.name or __name__)
        start = time.perf_counter()

        logger.info("[%s] ▶ start", trace_id)
        try:
            result = super().__call__(*args, **mutable_kwargs)
        except Exception:
            duration = time.perf_counter() - start
            logger.exception("[%s] ✖ fail in %.3fs", trace_id, duration)
            raise
        else:
            duration = time.perf_counter() - start
            logger.info("[%s] ✔ done in %.3fs", trace_id, duration)
            return result
        finally:
            _TRACE_ID_CTX.reset(token)

    # Celery stores headers on the request; typing is loose.
    def _resolve_trace_id(self, kwargs: Dict[str, Any]) -> str:
        trace_id = kwargs.pop("trace_id", None)

        request = getattr(self, "request", None)
        if request is not None:
            headers: Mapping[str, Any] | None = getattr(request, "headers", None)
            if headers:
                trace_id = trace_id or headers.get(TRACE_ID_HEADER)

            if trace_id is None:
                request_kwargs: Mapping[str, Any] | None = getattr(request, "kwargs", None)
                if request_kwargs:
                    trace_id = request_kwargs.get("trace_id")

        if trace_id is None:
            trace_id = current_trace_id()

        if trace_id is None:
            trace_id = uuid.uuid4().hex

        return trace_id


# Re-export the shared Celery app for convenience.
from .celery_app import app  # noqa: E402  # isort:skip


__all__ = ["NyxTask", "current_trace_id", "TRACE_ID_HEADER", "app"]
