"""Tests for the Celery coroutine runner helper."""

from __future__ import annotations

import asyncio
import importlib

from nyx.tasks.utils import run_coro


def _clear_worker_event_loop(connection_module) -> None:
    """Remove any persisted worker event loop to start from a clean slate."""

    thread_local = connection_module._state.thread_local  # type: ignore[attr-defined]
    if hasattr(thread_local, "event_loop"):
        loop = thread_local.event_loop
        if loop and not loop.is_closed():
            loop.close()
        del thread_local.event_loop

    connection_module._state.reset_locks()  # type: ignore[attr-defined]


def test_run_coro_reuses_worker_event_loop():
    """Back-to-back calls should reuse the worker loop instead of closing it."""

    connection = importlib.import_module("db.connection")
    _clear_worker_event_loop(connection)

    seen_loops: list[asyncio.AbstractEventLoop] = []

    async def fake_db_operation(label: str) -> str:
        loop = asyncio.get_running_loop()
        seen_loops.append(loop)
        await asyncio.sleep(0)
        return label

    try:
        first = run_coro(fake_db_operation("first"))
        second = run_coro(fake_db_operation("second"))

        assert first == "first"
        assert second == "second"
        assert len(seen_loops) == 2

        worker_loop = seen_loops[0]
        assert seen_loops[1] is worker_loop, "Expected persistent worker event loop"
        assert not worker_loop.is_closed()
        assert connection._state.thread_local.event_loop is worker_loop  # type: ignore[attr-defined]
    finally:
        if seen_loops:
            loop = seen_loops[0]
            if not loop.is_closed():
                loop.close()
        setattr(connection._state.thread_local, "event_loop", None)  # type: ignore[attr-defined]
        connection._state.reset_locks()  # type: ignore[attr-defined]
