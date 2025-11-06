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


class _ClosableConnection:
    def __init__(self):
        self.closed = False
        self.close_calls = 0

    async def close(self):
        self.closed = True
        self.close_calls += 1


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


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_close_existing_pool_handles_terminate_runtime_error(monkeypatch, anyio_backend):
    """close_existing_pool treats RuntimeError("Event loop is closed") from terminate as graceful."""

    isolated_state = db_connection.GlobalPoolState()
    monkeypatch.setattr(db_connection, "_state", isolated_state)

    class _TerminateRuntimePool:
        def __init__(self, loop: asyncio.AbstractEventLoop):
            self._loop = loop
            self._closed = False
            self.close_calls = 0
            self.terminate_calls = 0

        def get_size(self):
            return 1

        def get_idle_size(self):
            return 1

        async def close(self):
            self.close_calls += 1
            raise RuntimeError("Event loop is closed")

        def terminate(self):
            self.terminate_calls += 1
            raise RuntimeError("Event loop is closed")

    loop = asyncio.get_running_loop()
    failing_pool = _TerminateRuntimePool(loop)
    isolated_state.pool = failing_pool
    isolated_state.pool_loop = loop
    isolated_state.pool_state = db_connection.PoolState.HEALTHY
    isolated_state.metrics.pool_state = db_connection.PoolState.HEALTHY

    await db_connection.close_existing_pool()

    assert isolated_state.pool is None
    assert failing_pool.terminate_calls == 1
    assert failing_pool._closed is True


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize(
    "exception_type",
    [asyncio.CancelledError, asyncio.TimeoutError],
)
async def test_register_vector_retry_closes_connection_on_cancel_or_timeout(
    monkeypatch, exception_type, anyio_backend
):
    """Registration failures from cancellation or timeout should close the connection."""

    conn = _ClosableConnection()

    async def failing_register(connection):
        raise exception_type()

    monkeypatch.setattr(
        db_connection.pgvector_asyncpg,
        "register_vector",
        failing_register,
    )

    with pytest.raises(exception_type):
        await db_connection._register_vector_with_retry(
            conn,
            setup_timeout=0.05,
            max_retries=3,
            initial_retry_delay=0.01,
        )

    assert conn.closed is True
    assert conn.close_calls == 1
