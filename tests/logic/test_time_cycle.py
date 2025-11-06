import asyncio
from contextlib import asynccontextmanager, contextmanager

from logic import time_cycle


def test_get_current_time_batches_query(monkeypatch):
    fetch_calls = []

    class DummyConnection:
        async def fetch(self, query, *args):
            fetch_calls.append((query, args))
            return [
                {"key": "CurrentYear", "value": "42"},
                {"key": "TimeOfDay", "value": "Evening"},
            ]

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    @contextmanager
    def fake_skip_vector_registration():
        yield None

    monkeypatch.setattr(time_cycle, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(time_cycle, "skip_vector_registration", fake_skip_vector_registration)

    result = asyncio.run(time_cycle.get_current_time(1, 2))

    assert result == (42, 1, 1, "Evening")
    assert len(fetch_calls) == 1
    normalized_query = " ".join(fetch_calls[0][0].split())
    assert "SELECT key, value FROM CurrentRoleplay" in normalized_query
    assert "key = ANY" in normalized_query


def test_get_current_time_defaults(monkeypatch):
    class DummyConnection:
        async def fetch(self, query, *args):
            return []

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    @contextmanager
    def fake_skip_vector_registration():
        yield None

    monkeypatch.setattr(time_cycle, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(time_cycle, "skip_vector_registration", fake_skip_vector_registration)

    result = asyncio.run(time_cycle.get_current_time(5, 6))

    assert result == (1, 1, 1, "Morning")
