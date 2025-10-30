"""Common utilities for background tasks."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def with_retry(func: Callable) -> Callable:
    """Decorator to add retry logic to Celery tasks.

    This decorator works with Celery's bind=True pattern to provide
    automatic retry on failure with exponential backoff.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as exc:
            # Exponential backoff: 10s, 30s, 90s
            retry_countdown = 10 * (3 ** self.request.retries)
            logger.warning(
                f"Task {self.name} failed (attempt {self.request.retries + 1}): {exc}. "
                f"Retrying in {retry_countdown}s"
            )
            raise self.retry(exc=exc, countdown=retry_countdown)

    return wrapper


def run_coro(coro) -> Any:
    """Run an async coroutine in a new event loop (blocking).

    Use this to call async functions from Celery tasks (which are synchronous).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
