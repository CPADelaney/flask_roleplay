"""Best-effort idempotency helpers for Celery tasks."""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Callable

try:  # pragma: no cover - optional dependency during tests
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore

_LOCK = threading.Lock()
_IN_MEMORY_KEYS: set[str] = set()


def _get_client():
    if redis is None:
        return None
    url = os.getenv("NYX_IDEMPOTENCY_REDIS", os.getenv("REDIS_URL", "redis://localhost:6379/1"))
    try:
        client = redis.Redis.from_url(url)
        client.ping()
        return client
    except Exception:
        return None


_REDIS_CLIENT = _get_client()


def idempotent(key_fn: Callable[..., str], ttl_sec: int = 3600) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that prevents duplicate task execution for a window."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = key_fn(*args, **kwargs)
            if not key:
                return func(*args, **kwargs)

            client = _REDIS_CLIENT
            if client is not None:
                try:
                    if not client.setnx(key, "1"):
                        return None
                    client.expire(key, ttl_sec)
                    return func(*args, **kwargs)
                except Exception:
                    pass

            with _LOCK:
                if key in _IN_MEMORY_KEYS:
                    return None
                _IN_MEMORY_KEYS.add(key)
            try:
                return func(*args, **kwargs)
            finally:
                pass

        return wrapper

    return decorator


def clear_cache() -> None:
    """Clear the in-memory idempotency cache (test helper)."""

    with _LOCK:
        _IN_MEMORY_KEYS.clear()


__all__ = ["idempotent", "clear_cache"]
