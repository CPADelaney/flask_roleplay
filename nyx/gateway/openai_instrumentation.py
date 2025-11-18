"""Instrumentation helpers for the OpenAI async client."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger("nyx.openai")

try:  # pragma: no cover - optional dependency
    from openai import AsyncClient
except ImportError:  # pragma: no cover - optional dependency
    AsyncClient = None  # type: ignore[assignment]


def _build_payload(opts: Any, *, elapsed: float, stream: bool, error: BaseException | None) -> dict[str, Any]:
    payload = {
        "method": getattr(opts, "method", None),
        "path": getattr(opts, "path", None),
        "elapsed": elapsed,
        "stream": bool(stream),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_str"] = str(error)[:300]
    return payload


def _install() -> None:
    if AsyncClient is None:  # pragma: no cover - dependency missing
        return

    original_request: Callable[..., Any] | None = getattr(AsyncClient, "request", None)
    if original_request is None:
        return
    if getattr(original_request, "__nyx_instrumented__", False):
        return

    async def timed_request(self, cast_to, opts, *, stream: bool = False, stream_cls=None):  # type: ignore[override]
        start = time.perf_counter()
        try:
            response = await original_request(  # type: ignore[misc]
                self,
                cast_to,
                opts,
                stream=stream,
                stream_cls=stream_cls,
            )
        except Exception as exc:  # pragma: no cover - logging side effect
            elapsed = time.perf_counter() - start
            logger.error(
                "nyx.openai.request_failed",
                extra=_build_payload(opts, elapsed=elapsed, stream=stream, error=exc),
            )
            raise

        elapsed = time.perf_counter() - start
        logger.info(
            "nyx.openai.request",
            extra=_build_payload(opts, elapsed=elapsed, stream=stream, error=None),
        )
        return response

    timed_request.__nyx_instrumented__ = True  # type: ignore[attr-defined]
    AsyncClient.request = timed_request  # type: ignore[assignment]


_install()

__all__ = []
