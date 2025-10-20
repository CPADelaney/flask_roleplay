import sys
from pathlib import Path

import pytest
import asyncpg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        self.release_called = False
        self._conn = _FakeConnection()

    async def acquire(self):
        return self._conn

    async def release(self, conn, timeout):
        self.release_called = True
        raise self._exc


@pytest.mark.asyncio
async def test_connection_missing_from_pool_is_terminated(monkeypatch):
    fake_pool = _FakePool(asyncpg.exceptions.ConnectionDoesNotExistError("missing"))

    loop = db_connection.get_or_create_event_loop()
    monkeypatch.setattr(db_connection, "DB_POOL", fake_pool)
    monkeypatch.setattr(db_connection, "DB_POOL_LOOP", loop)

    monkeypatch.setenv("DB_COMMAND_TIMEOUT", "1")
    monkeypatch.setenv("DB_RELEASE_TIMEOUT", "0.5")

    async with db_connection.get_db_connection_context(timeout=0.1) as conn_wrapper:
        raw_conn = conn_wrapper.raw_connection
        assert isinstance(raw_conn, _FakeConnection)

    assert fake_pool.release_called is True
    assert raw_conn._terminated is True


@pytest.mark.asyncio
async def test_attribute_error_during_release_is_terminated(monkeypatch):
    fake_pool = _FakePool(AttributeError("_holder missing"))

    loop = db_connection.get_or_create_event_loop()
    monkeypatch.setattr(db_connection, "DB_POOL", fake_pool)
    monkeypatch.setattr(db_connection, "DB_POOL_LOOP", loop)

    monkeypatch.setenv("DB_COMMAND_TIMEOUT", "1")
    monkeypatch.setenv("DB_RELEASE_TIMEOUT", "0.5")

    async with db_connection.get_db_connection_context(timeout=0.1) as conn_wrapper:
        raw_conn = conn_wrapper.raw_connection
        assert isinstance(raw_conn, _FakeConnection)

    assert fake_pool.release_called is True
    assert raw_conn._terminated is True
