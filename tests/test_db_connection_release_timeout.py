import asyncio

import pytest

from db import connection as db_connection


class _FakeConnection:
    def __init__(self):
        self._con = object()
        self._terminated = False
        self._closed = False

    def is_closed(self):
        return self._closed

    def terminate(self):
        self._terminated = True
        self._closed = True


class _FakePool:
    def __init__(self, exc):
        self._closed = False
        self._exc = exc
        self.released_with_timeout = None
        self._conn = _FakeConnection()

    async def acquire(self):
        return self._conn

    async def release(self, conn, timeout):
        self.released_with_timeout = timeout
        raise self._exc


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_type", [asyncio.TimeoutError, asyncio.CancelledError])
async def test_connection_release_timeout_triggers_terminate(monkeypatch, exc_type):
    fake_pool = _FakePool(exc_type())

    # Ensure global pool state does not interfere with the test.
    loop = db_connection.get_or_create_event_loop()
    monkeypatch.setattr(db_connection, "DB_POOL", fake_pool)
    monkeypatch.setattr(db_connection, "DB_POOL_LOOP", loop)

    # Use a deterministic release timeout for easier assertions.
    monkeypatch.setenv("DB_COMMAND_TIMEOUT", "1")
    monkeypatch.setenv("DB_RELEASE_TIMEOUT", "0.5")

    async with db_connection.get_db_connection_context(timeout=0.1) as conn:
        # Acquire succeeds and yields our fake connection.
        assert isinstance(conn, _FakeConnection)

    # Release should have been attempted with the configured timeout.
    assert fake_pool.released_with_timeout == pytest.approx(0.5)

    # The connection should have been terminated after the release timeout.
    assert conn._terminated is True
