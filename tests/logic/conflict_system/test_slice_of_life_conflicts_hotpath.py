import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from logic.conflict_system.slice_of_life_conflicts import SliceOfLifeConflictType
from logic.conflict_system.slice_of_life_conflicts_hotpath import get_detected_tensions


class _DummyConn:
    def __init__(self, row: Dict[str, Any]):
        self._row = row

    async def fetchrow(self, *_args, **_kwargs):
        return dict(self._row)


class _DummyContext:
    def __init__(self, row: Dict[str, Any]):
        self._row = row

    async def __aenter__(self):
        return _DummyConn(self._row)

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_get_detected_tensions_skips_refresh_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_row = {
        "status": "ready",
        "payload": [
            {"type": "subtle_rivalry", "intensity": "tension", "description": "Quiet friction"}
        ],
        "updated_at": datetime.utcnow() - timedelta(hours=1),
    }

    queued: List[Dict[str, Any]] = []

    monkeypatch.setattr(
        "logic.conflict_system.slice_of_life_conflicts_hotpath.get_db_connection_context",
        lambda: _DummyContext(stale_row),
    )
    monkeypatch.setattr(
        "logic.conflict_system.slice_of_life_conflicts_hotpath._queue_task",
        lambda *args, **kwargs: queued.append({"args": args, "kwargs": kwargs}),
    )

    tensions = asyncio.run(get_detected_tensions(1, 2))

    assert tensions
    assert tensions[0]["type"] == SliceOfLifeConflictType.SUBTLE_RIVALRY
    assert queued == []


def test_get_detected_tensions_queues_when_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_row = {
        "status": "ready",
        "payload": [],
        "updated_at": datetime.utcnow() - timedelta(hours=1),
    }

    queued: List[Dict[str, Any]] = []

    monkeypatch.setattr(
        "logic.conflict_system.slice_of_life_conflicts_hotpath.get_db_connection_context",
        lambda: _DummyContext(stale_row),
    )
    monkeypatch.setattr(
        "logic.conflict_system.slice_of_life_conflicts_hotpath._queue_task",
        lambda *args, **kwargs: queued.append({"args": args, "kwargs": kwargs}),
    )

    asyncio.run(get_detected_tensions(3, 4, eager=True))

    assert queued, "Eager reads should queue a refresh task"
    queued_args = queued[0]["args"]
    assert queued_args[0] == "refresh_tension_cache"
    assert queued_args[1]["user_id"] == 3
    assert queued_args[1]["conversation_id"] == 4
