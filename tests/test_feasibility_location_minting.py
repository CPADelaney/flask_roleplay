import os
import sys
from contextlib import asynccontextmanager

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nyx.location.types import Location as LocationModel
from nyx.nyx_agent import feasibility


class _FakeConnection:
    def __init__(self):
        self.locations = []

    def upsert_location(self, user_id: int, conversation_id: int, location_name: str) -> None:
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
    stored_candidates = []

    async def fake_get_or_create_location(
        conn,
        *,
        user_id,
        conversation_id,
        candidate,
        **_kwargs,
    ):
        stored_candidates.append(candidate)
        normalized_name = feasibility._normalize_location_phrase(candidate.place.name)
        conn.upsert_location(int(user_id), int(conversation_id), normalized_name)
        return LocationModel(
            user_id=user_id,
            conversation_id=conversation_id,
            location_name=normalized_name,
            location_type=candidate.place.level,
            city=candidate.place.address.get("city"),
            region=candidate.place.address.get("region"),
            country=candidate.place.address.get("country"),
        )

    monkeypatch.setattr(feasibility, "get_or_create_location", fake_get_or_create_location)

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
    assert minted_name in setting_context["known_location_names"]
    assert normalized in setting_context["known_location_names"]
    assert any(row["location_name"] == normalized for row in fake_conn.locations)

    resolver_cache = setting_context["location_resolver_cache"]
    assert normalized in resolver_cache
    minted_decision = resolver_cache[normalized]
    assert isinstance(minted_decision.get("location"), LocationModel)
    assert minted_decision["location"].location_name == normalized
    assert minted_decision.get("candidate").place.name == minted_name

    assert stored_candidates

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
    assert len(stored_candidates) == 1
    assert sum(1 for row in fake_conn.locations if row["location_name"] == normalized) == 1
