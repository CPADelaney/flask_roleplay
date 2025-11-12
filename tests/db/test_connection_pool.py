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


class _DeferredPool:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self.expire_calls = 0
        self.acquire_calls = 0

    async def acquire(self):
        self.acquire_calls += 1
        return _DeferredConnection(self)

    async def expire_connections(self):
        self.expire_calls += 1


class _PoolHolder:
    def __init__(self, pool):
        self._pool = pool


class _DeferredConnection:
    def __init__(self, pool):
        self._closed = False
        self._terminated = False
        self._vector_registered = False
        self._holder = _PoolHolder(pool)

    async def close(self):
        self._closed = True

    def terminate(self):
        self._terminated = True

    def is_closed(self):
        return self._closed


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_resilient_wrapper_tracks_pending_tasks(monkeypatch, anyio_backend):
    """Ensure resilient wrapper registers and cleans up pending operations during cancellation."""

    isolated_state = db_connection.GlobalPoolState()
    monkeypatch.setattr(db_connection, "_state", isolated_state)

    class _BlockingPool:
        async def acquire(self):
            raise AssertionError("acquire should not be called in this test")

        async def release(self, conn, timeout=None):
            return None

    class _BlockingConnection:
        def __init__(self):
            self.started = asyncio.Event()
            self.block_event = asyncio.Event()
            self.cancelled = False

        async def fetch(self, *args, **kwargs):
            self.started.set()
            try:
                await self.block_event.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    pool = _BlockingPool()
    raw_conn = _BlockingConnection()
    conn_id = id(raw_conn)
    ops_lock = isolated_state.get_connection_ops_lock()

    async with ops_lock:
        isolated_state.connection_pending_ops[conn_id] = []

    wrapped = db_connection._ResilientConnectionWrapper(
        pool,
        raw_conn,
        conn_id,
        ops_lock,
    )

    wait_task = asyncio.create_task(
        asyncio.wait_for(wrapped.fetch("SELECT 1"), timeout=0.1)
    )

    await raw_conn.started.wait()

    async with ops_lock:
        pending_ops = list(isolated_state.connection_pending_ops[conn_id])

    assert len(pending_ops) == 1
    tracked_task = pending_ops[0]
    assert isinstance(tracked_task, asyncio.Task)
    assert not tracked_task.done()

    with pytest.raises(asyncio.TimeoutError):
        await wait_task

    assert raw_conn.cancelled is True

    async with ops_lock:
        assert isolated_state.connection_pending_ops.get(conn_id) == []


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_pool_from_dead_loop_is_terminated(monkeypatch, caplog, anyio_backend):
    """Ensure pools owned by closed loops are terminated without asyncpg loop errors."""

    caplog.set_level(logging.WARNING)

    async def _fake_register(conn):
        return None

    monkeypatch.setattr(
        db_connection.pgvector_asyncpg,
        "register_vector",
        _fake_register,
    )

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
            assert isinstance(conn.raw_connection, _DummyConnection)

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
async def test_setup_connection_registers_by_default(monkeypatch, anyio_backend):
    """setup_connection registers pgvector when DB_REGISTER_VECTOR is unset."""

    monkeypatch.delenv("DB_REGISTER_VECTOR", raising=False)

    registered_connections = []

    async def fake_register(conn):
        registered_connections.append(conn)

    monkeypatch.setattr(
        db_connection.pgvector_asyncpg,
        "register_vector",
        fake_register,
    )

    test_conn = _DummyConnection()

    await db_connection.setup_connection(test_conn)

    assert registered_connections == [test_conn]

    another_conn = _DummyConnection()

    with db_connection.skip_vector_registration():
        await db_connection.setup_connection(another_conn)

    assert registered_connections == [test_conn]


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
async def test_deferred_registration_failure_expires_connection(monkeypatch, anyio_backend):
    isolated_state = db_connection.GlobalPoolState()
    monkeypatch.setattr(db_connection, "_state", isolated_state)

    monkeypatch.setenv("DB_REGISTER_VECTOR", "1")

    loop = asyncio.get_running_loop()
    pool = _DeferredPool(loop)
    isolated_state.pool = pool
    isolated_state.pool_loop = loop
    isolated_state.pool_state = db_connection.PoolState.HEALTHY
    isolated_state.metrics.pool_state = db_connection.PoolState.HEALTHY

    register_calls = {"count": 0}

    async def fake_register(
        conn,
        *,
        setup_timeout: float,
        max_retries: int,
        initial_retry_delay: float,
    ) -> None:
        register_calls["count"] += 1
        if register_calls["count"] == 1:
            conn._terminated = True
            await conn.close()
            await conn._holder._pool.expire_connections()
            raise asyncio.TimeoutError
        conn._vector_registered = True

    monkeypatch.setattr(db_connection, "_register_vector_with_retry", fake_register)

    first_conn = await pool.acquire()

    with pytest.raises(asyncio.TimeoutError):
        await db_connection.setup_connection(first_conn)

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert first_conn._closed is True
    assert first_conn._terminated is True
    assert pool.expire_calls == 1

    replacement_conn = await pool.acquire()
    await db_connection.setup_connection(replacement_conn)

    assert register_calls["count"] == 2
    assert replacement_conn._vector_registered is True
    assert replacement_conn._closed is False
    assert replacement_conn._terminated is False
