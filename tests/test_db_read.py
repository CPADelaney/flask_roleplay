import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DB_DSN", "postgresql://user:pass@localhost/testdb")

from db import read as db_read


def test_read_scene_context_with_limit(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_fetch(query: str, params: List[Any]) -> List[Dict[str, Any]]:
        captured["query"] = query
        captured["params"] = params
        return [{"user_id": 1, "conversation_id": 2, "scene_context": {}}]

    monkeypatch.setattr(db_read, "_fetch", fake_fetch)

    rows = asyncio.run(db_read.read_scene_context(1, 2, limit=5))

    assert rows and rows[0]["user_id"] == 1
    assert "LIMIT $3" in captured["query"]
    assert captured["params"] == [1, 2, 5]


def test_read_scene_context_without_limit(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_fetch(query: str, params: List[Any]) -> List[Dict[str, Any]]:
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(db_read, "_fetch", fake_fetch)

    asyncio.run(db_read.read_scene_context(1, 2, limit=None))

    assert "LIMIT" not in captured["query"].upper()
    assert captured["params"] == [1, 2]


def test_read_entity_cards_with_embedding(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_fetch(query: str, params: List[Any]) -> List[Dict[str, Any]]:
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(db_read, "_fetch", fake_fetch)

    embedding = (0.1, 0.2, 0.3)
    asyncio.run(
        db_read.read_entity_cards(
            7,
            11,
            embedding=embedding,
            query_text="garden",
            entity_types=["npc", "memory"],
            limit=250,
        )
    )

    assert "vector_score DESC" in captured["query"]
    assert captured["params"][0] == list(embedding)
    assert captured["params"][-1] == 200  # limit is clamped


def test_read_entity_cards_text_only_defaults(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_fetch(query: str, params: List[Any]) -> List[Dict[str, Any]]:
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(db_read, "_fetch", fake_fetch)

    asyncio.run(db_read.read_entity_cards(3, 5, query_text="rose"))

    assert captured["params"] == [3, 5, "rose", ["npc", "location", "memory"], 10]
    assert "text_score DESC" in captured["query"]


def test_read_recent_chunks_clamps_limit(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_fetch(query: str, params: List[Any]) -> List[Dict[str, Any]]:
        captured["query"] = query
        captured["params"] = params
        return []

    monkeypatch.setattr(db_read, "_fetch", fake_fetch)

    asyncio.run(db_read.read_recent_chunks(9, 13, limit=500))

    assert captured["params"] == [9, 13, 200]
    assert "LIMIT $3" in captured["query"]


def test_read_rows_dispatch(monkeypatch):
    calls: Dict[str, Any] = {}

    async def fake_scene(*args, **kwargs):  # pragma: no cover - patched branch
        calls["scene"] = (args, kwargs)
        return [{"scene_context": {}}]

    async def fake_cards(*args, **kwargs):
        calls["cards"] = (args, kwargs)
        return []

    async def fake_chunks(*args, **kwargs):
        calls["chunks"] = (args, kwargs)
        return []

    monkeypatch.setattr(db_read, "read_scene_context", fake_scene)
    monkeypatch.setattr(db_read, "read_entity_cards", fake_cards)
    monkeypatch.setattr(db_read, "read_recent_chunks", fake_chunks)

    asyncio.run(db_read.read_rows("scene_context", user_id=1, conversation_id=2))
    asyncio.run(
        db_read.read_rows(
            "entity_cards",
            user_id=3,
            conversation_id=4,
            limit=99,
            entity_types=["npc"],
            query_text="test",
        )
    )
    asyncio.run(db_read.read_rows("recent_chunks", user_id=5, conversation_id=6))

    assert "scene" in calls and calls["scene"][0] == (1, 2, None)
    assert "cards" in calls
    assert calls["cards"][0][:2] == (3, 4)
    assert calls["chunks"][0][:2] == (5, 6)

    with pytest.raises(ValueError):
        asyncio.run(db_read.read_rows("not-a-view", user_id=1, conversation_id=2))

