"""Utilities for tracking lore cache versions."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from db.connection import get_db_connection_context

try:  # pragma: no cover - redis optional in tests
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


logger = logging.getLogger(__name__)

_VERSION_KEY_ENV = "NYX_LORE_VERSION_KEY"
_DEFAULT_VERSION_KEY = "nyx:lore:version"
_LOCAL_VERSION = 0
_LOCAL_LOCK = threading.Lock()
_redis_client: Optional["redis.Redis"] = None


def _get_redis_client() -> Optional["redis.Redis"]:
    """Return a shared Redis client if Redis is available."""

    global _redis_client
    if redis is None:
        return None
    if _redis_client is not None:
        return _redis_client

    redis_url = (
        os.getenv("NYX_LORE_VERSION_REDIS")
        or os.getenv("NYX_SNAPSHOT_REDIS")
        or os.getenv("REDIS_URL")
    )
    if not redis_url:
        return None

    try:
        client = redis.Redis.from_url(redis_url)
        client.ping()
    except Exception:  # pragma: no cover - best effort cache bump
        logger.debug("Failed to initialize Redis client for lore versioning", exc_info=True)
        return None

    _redis_client = client
    return _redis_client


def _version_key() -> str:
    return os.getenv(_VERSION_KEY_ENV, _DEFAULT_VERSION_KEY)


def bump_lore_version() -> int:
    """Advance the lore version counter.

    Returns the incremented counter value. Failures are logged at warning level
    without raising exceptions so lore mutations continue.
    """

    global _LOCAL_VERSION, _redis_client

    client = _get_redis_client()
    if client is not None:
        try:
            new_value = int(client.incr(_version_key()))
        except Exception as exc:  # pragma: no cover - network errors best effort
            logger.warning("Failed to bump lore version via Redis: %s", exc, exc_info=True)
            _redis_client = None
        else:
            with _LOCAL_LOCK:
                if new_value > _LOCAL_VERSION:
                    _LOCAL_VERSION = new_value
            return new_value

    with _LOCAL_LOCK:
        _LOCAL_VERSION += 1
        return _LOCAL_VERSION


def _should_bump_from_sql(sql: str) -> bool:
    """Heuristically determine if a SQL statement mutates lore state."""

    if not sql:
        return False

    command = sql.lstrip().split(None, 1)
    if not command:
        return False

    keyword = command[0].upper()
    if keyword in {"SELECT", "SHOW", "EXPLAIN"}:
        return False

    if keyword == "WITH":
        upper_sql = sql.upper()
        return any(token in upper_sql for token in ("INSERT", "UPDATE", "DELETE", "MERGE"))

    return keyword in {"INSERT", "UPDATE", "DELETE", "MERGE"}


def _wrap_execution(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an asyncpg execute-style callable to bump lore version on writes."""

    async def wrapper(sql: str, *args, **kwargs):
        result = await fn(sql, *args, **kwargs)
        if _should_bump_from_sql(sql):
            bump_lore_version()
        return result

    return wrapper


@asynccontextmanager
async def get_lore_db_connection_context(*args, **kwargs) -> AsyncIterator[Any]:
    """Yield a DB connection that bumps lore version for mutating statements."""

    async with get_db_connection_context(*args, **kwargs) as conn:
        original_execute = getattr(conn, "execute", None)
        original_executemany = getattr(conn, "executemany", None)
        original_fetch = getattr(conn, "fetch", None)
        original_fetchrow = getattr(conn, "fetchrow", None)
        original_fetchval = getattr(conn, "fetchval", None)

        if original_execute is not None:
            conn.execute = _wrap_execution(original_execute)  # type: ignore[attr-defined]
        if original_executemany is not None:
            conn.executemany = _wrap_execution(original_executemany)  # type: ignore[attr-defined]
        if original_fetch is not None:
            conn.fetch = _wrap_execution(original_fetch)  # type: ignore[attr-defined]
        if original_fetchrow is not None:
            conn.fetchrow = _wrap_execution(original_fetchrow)  # type: ignore[attr-defined]
        if original_fetchval is not None:
            conn.fetchval = _wrap_execution(original_fetchval)  # type: ignore[attr-defined]

        try:
            yield conn
        finally:
            if original_execute is not None:
                conn.execute = original_execute  # type: ignore[attr-defined]
            if original_executemany is not None:
                conn.executemany = original_executemany  # type: ignore[attr-defined]
            if original_fetch is not None:
                conn.fetch = original_fetch  # type: ignore[attr-defined]
            if original_fetchrow is not None:
                conn.fetchrow = original_fetchrow  # type: ignore[attr-defined]
            if original_fetchval is not None:
                conn.fetchval = original_fetchval  # type: ignore[attr-defined]


__all__ = ["bump_lore_version", "get_lore_db_connection_context"]
