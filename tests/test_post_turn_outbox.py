from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
import importlib.util
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

fake_celery_app = types.SimpleNamespace(send_task=lambda *args, **kwargs: None)
sys.modules["celery_config"] = types.SimpleNamespace(celery_app=fake_celery_app)

_db_pkg = types.ModuleType("db")
_db_connection = types.ModuleType("db.connection")
_db_connection.get_db_connection_context = lambda *args, **kwargs: None  # patched in tests
_db_connection.run_async_in_worker_loop = lambda coro: coro
sys.modules["db"] = _db_pkg
sys.modules["db.connection"] = _db_connection
_db_pkg.connection = _db_connection

spec = importlib.util.spec_from_file_location(
    "post_turn_under_test", ROOT / "nyx" / "tasks" / "realtime" / "post_turn.py"
)
post_turn = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["post_turn_under_test"] = post_turn
spec.loader.exec_module(post_turn)


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeConn:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def transaction(self):
        return FakeTransaction()

    async def fetchrow(self, query: str, current_time: datetime):
        for row in self.rows:
            if row.get("dead_lettered"):
                continue
            if row.get("next_run_at") and row["next_run_at"] > current_time:
                continue
            return {
                "id": row["id"],
                "payload": row["payload"],
                "attempts": row.get("attempts", 0),
            }
        return None

    async def execute(self, query: str, *params):
        normalized = query.strip()
        if normalized.startswith("DELETE"):
            row_id = params[0]
            self.rows[:] = [row for row in self.rows if row["id"] != row_id]
        elif normalized.startswith("UPDATE"):
            row_id, attempts, last_error, next_run_at, dead_lettered = params
            for row in self.rows:
                if row["id"] == row_id:
                    row["attempts"] = attempts
                    row["last_error"] = last_error
                    row["next_run_at"] = next_run_at
                    row["dead_lettered"] = dead_lettered
                    break


class FakeConnContext:
    def __init__(self, conn: FakeConn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_outbox_success_deletes_row(monkeypatch):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "payload": {"task_name": "task.name", "kwargs": {"payload": {"foo": "bar"}}, "queue": "background"},
            "attempts": 0,
            "last_error": None,
            "next_run_at": now,
            "dead_lettered": False,
        }
    ]
    conn = FakeConn(rows)

    def fake_get_conn():
        return FakeConnContext(conn)

    monkeypatch.setattr(post_turn, "get_db_connection_context", fake_get_conn)
    monkeypatch.setattr(post_turn, "_utcnow", lambda: now)

    sent: List[Dict[str, Any]] = []
    monkeypatch.setattr(post_turn, "_enqueue_task", lambda payload: sent.append(payload))

    processed = await post_turn._drain_outbox(limit=5)

    assert processed == 1
    assert not rows
    assert sent == [
        {"task_name": "task.name", "kwargs": {"payload": {"foo": "bar"}}, "queue": "background"}
    ]


@pytest.mark.asyncio
async def test_outbox_retry_backoff(monkeypatch):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "payload": {"task_name": "task.name", "kwargs": {"payload": {}}},
            "attempts": 0,
            "last_error": None,
            "next_run_at": now,
            "dead_lettered": False,
        }
    ]
    conn = FakeConn(rows)

    def fake_get_conn():
        return FakeConnContext(conn)

    monkeypatch.setattr(post_turn, "get_db_connection_context", fake_get_conn)
    monkeypatch.setattr(post_turn, "_utcnow", lambda: now)

    def raise_error(_payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(post_turn, "_enqueue_task", raise_error)

    processed = await post_turn._drain_outbox(limit=1)

    assert processed == 0
    assert rows[0]["attempts"] == 1
    assert rows[0]["dead_lettered"] is False
    assert rows[0]["last_error"] == "RuntimeError: boom"
    assert rows[0]["next_run_at"] == now + timedelta(seconds=post_turn._INITIAL_BACKOFF_SECONDS)


@pytest.mark.asyncio
async def test_outbox_dead_letters_after_max_attempts(monkeypatch):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "payload": {"task_name": "task.name", "kwargs": {"payload": {}}},
            "attempts": post_turn._MAX_ATTEMPTS - 1,
            "last_error": None,
            "next_run_at": now,
            "dead_lettered": False,
        }
    ]
    conn = FakeConn(rows)

    def fake_get_conn():
        return FakeConnContext(conn)

    monkeypatch.setattr(post_turn, "get_db_connection_context", fake_get_conn)
    monkeypatch.setattr(post_turn, "_utcnow", lambda: now)

    def raise_error(_payload):
        raise RuntimeError("kaput")

    monkeypatch.setattr(post_turn, "_enqueue_task", raise_error)

    processed = await post_turn._drain_outbox(limit=1)

    assert processed == 0
    assert rows[0]["attempts"] == post_turn._MAX_ATTEMPTS
    assert rows[0]["dead_lettered"] is True
    assert rows[0]["last_error"] == "RuntimeError: kaput"
    assert rows[0]["next_run_at"] == now
