import asyncio
import logging

import pytest

from db import connection as db_connection


class _DummyConnection:
    def __init__(self):
        self._terminated = False
        self._closed = False

    def is_closed(self):
        return self._closed

    def terminate(self):
        self._terminated = True
        self._closed = True


class _DummyPool:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._closed = False
        self._terminated = False
        self._size = 1
        self._acquired = False
        self.acquire_calls = 0
        self.release_timeouts = []
        self.closed_calls = 0
        self._last_connection = None

    def get_size(self):
        return self._size

    def get_idle_size(self):
        return self._size - int(self._acquired)

    async def acquire(self):
        self._acquired = True
        self.acquire_calls += 1
        self._last_connection = _DummyConnection()
        return self._last_connection

    async def release(self, conn, timeout=None):
        self._acquired = False
        self.release_timeouts.append(timeout)

    async def close(self):
        self.closed_calls += 1
        self._closed = True

    def terminate(self):
        self._terminated = True
        self._closed = True


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_pool_from_dead_loop_is_terminated(monkeypatch, caplog, anyio_backend):
    """Ensure pools owned by closed loops are terminated without asyncpg loop errors."""

    caplog.set_level(logging.WARNING)

    isolated_state = db_connection.GlobalPoolState()
    monkeypatch.setattr(db_connection, "_state", isolated_state)

    old_loop = asyncio.new_event_loop()
    try:
        old_pool = _DummyPool(old_loop)
        isolated_state.pool = old_pool
        isolated_state.pool_loop = old_loop
        isolated_state.pool_state = db_connection.PoolState.HEALTHY
        isolated_state.metrics.pool_state = db_connection.PoolState.HEALTHY

        old_loop.close()

        current_loop = asyncio.get_running_loop()
        new_pool = _DummyPool(current_loop)

        async def fake_initialize_connection_pool(app=None, force_new=False):
            isolated_state.pool = new_pool
            isolated_state.pool_loop = current_loop
            isolated_state.pool_state = db_connection.PoolState.HEALTHY
            isolated_state.metrics.pool_state = db_connection.PoolState.HEALTHY
            return True

        monkeypatch.setattr(
            db_connection,
            "initialize_connection_pool",
            fake_initialize_connection_pool,
        )

        async with db_connection.get_db_connection_context(timeout=0.1) as conn:
            assert conn.raw_connection is new_pool._last_connection

    finally:
        if not old_loop.is_closed():
            old_loop.close()

    assert old_pool._terminated is True
    assert old_pool.closed_calls == 0
    assert isolated_state.pool is new_pool
    assert new_pool.acquire_calls == 1
    assert new_pool.release_timeouts, "Expected connection release to be invoked"
    assert not any(
        "Non-thread-safe operation" in record.getMessage()
        for record in caplog.records
    )
