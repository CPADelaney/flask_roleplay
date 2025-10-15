import os
import sys
from contextlib import asynccontextmanager

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nyx.nyx_agent import feasibility


class _FakeConnection:
    def __init__(self):
        self.locations = []

    async def execute(self, query, *args):
        if "INSERT INTO Locations" in query:
            user_id, conversation_id, location_name = args[:3]
            existing = next(
                (
                    row
                    for row in self.locations
                    if row["user_id"] == user_id
                    and row["conversation_id"] == conversation_id
                    and row["location_name"] == location_name
                ),
                None,
            )
            if not existing:
                self.locations.append(
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "location_name": location_name,
                    }
                )
        return "OK"

    async def fetch(self, query, *args):
        if "SELECT location_name FROM Locations" in query:
            user_id, conversation_id = args
            return [
                {"location_name": row["location_name"]}
                for row in self.locations
                if row["user_id"] == user_id and row["conversation_id"] == conversation_id
            ]
        return []

    async def fetchrow(self, *args, **kwargs):
        return None

    async def fetchval(self, *args, **kwargs):
        return None


class _DummyNyxContext:
    def __init__(self, user_id: int, conversation_id: int) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id


@pytest.mark.asyncio
async def test_minted_location_persisted_and_reused(monkeypatch):
    fake_conn = _FakeConnection()

    @asynccontextmanager
    async def fake_db_context(*_args, **_kwargs):
        yield fake_conn

    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    minted_call_count = 0
    minted_name = "Shadow Citadel"
    normalized = "shadow citadel"

    async def fake_resolver(*_args, **_kwargs):
        nonlocal minted_call_count
        minted_call_count += 1
        return {
            "decision": "allow",
            "reason": "minted",
            "score": 0.9,
            "kind": "fictional",
            "token": minted_name,
            "normalized": normalized,
            "branch": "fictional",
            "minted": True,
        }

    monkeypatch.setattr(feasibility, "_resolve_location_candidate", fake_resolver)

    intent = {
        "raw_text": "Travel to the Shadow Citadel.",
        "destination": [minted_name],
        "categories": ["movement"],
    }
    text_l = intent["raw_text"].lower()

    setting_context = {
        "known_location_names": [],
        "world_model": {"allow_fictional_locations": True},
    }

    unresolved = await feasibility._find_unresolved_location_targets(
        intent,
        text_l,
        set(),
        set(),
        set(),
        set(),
        set(),
        setting_context,
        user_id=1,
        conversation_id=2,
    )

    assert unresolved == []
    assert normalized in setting_context["known_location_names"]
    assert any(row["location_name"] == normalized for row in fake_conn.locations)

    context_bundle = await feasibility._load_comprehensive_context(_DummyNyxContext(1, 2))
    assert normalized in context_bundle["known_location_names"]

    known_tokens = feasibility._build_known_location_tokens(
        set(),
        set(),
        context_bundle["known_location_names"],
        {},
    )

    second_context = {"known_location_names": context_bundle["known_location_names"]}
    second_unresolved = await feasibility._find_unresolved_location_targets(
        intent,
        text_l,
        set(),
        set(),
        known_tokens,
        set(),
        set(),
        second_context,
        user_id=1,
        conversation_id=2,
    )

    assert second_unresolved == []
    assert minted_call_count == 1
    assert sum(1 for row in fake_conn.locations if row["location_name"] == normalized) == 1
