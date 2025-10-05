import asyncio

import pytest

from db import connection as db_connection


@pytest.mark.asyncio
async def test_track_operation_respects_shutdown(monkeypatch):
    """Ensure Celery shutdown path rejects new operations cleanly."""

    # Start from a clean slate for shutdown tracking.
    monkeypatch.setattr(db_connection, "_SHUTTING_DOWN", False)
    monkeypatch.setattr(db_connection, "_SHUTTING_DOWN_PIDS", set())
    monkeypatch.setattr(db_connection, "_pending_operations", set())
    monkeypatch.setattr(db_connection, "_pending_operations_lock", None)

    async def sample_operation():
        await asyncio.sleep(0)
        return "ok"

    # Normal operation should succeed.
    result = await db_connection.track_operation(sample_operation())
    assert result == "ok"

    # Simulate Celery shutdown and ensure new work is rejected.
    db_connection.mark_shutting_down()
    assert db_connection.is_shutting_down() is True

    blocked_coro = sample_operation()
    with pytest.raises(ConnectionError):
        try:
            await db_connection.track_operation(blocked_coro)
        finally:
            try:
                blocked_coro.close()
            except RuntimeError:
                # Coroutine was already consumed by track_operation
                pass

    # No pending work should remain and shutdown wait should be a no-op.
    await db_connection.wait_for_pending_operations()


@pytest.mark.asyncio
async def test_close_connection_pool_falls_back_to_terminate(monkeypatch):
    """Close gracefully unless asyncpg raises AttributeError."""

    class DummyPool:
        def __init__(self):
            self._closed = False
            self.terminate_called = False

        async def close(self):
            raise AttributeError("'connection' object has no attribute 'is_closed'")

        def terminate(self):
            self.terminate_called = True
            self._closed = True

    dummy_pool = DummyPool()

    monkeypatch.setattr(db_connection, "DB_POOL", dummy_pool)
    monkeypatch.setattr(db_connection, "DB_POOL_LOOP", None)

    # Should not raise even though close() fails; terminate is used instead.
    await db_connection.close_connection_pool()

    assert dummy_pool.terminate_called is True
    # close_connection_pool should have cleared the module globals
    assert db_connection.DB_POOL is None
