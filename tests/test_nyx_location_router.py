import asyncio
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.location import router as location_router
from nyx.location.types import ResolveResult, STATUS_NOT_FOUND


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_resolve_place_or_travel_uses_disneyland_shortcut(monkeypatch):
    store = ConversationSnapshotStore()
    user_id = "42"
    conversation_id = "7"
    store.put(
        user_id,
        conversation_id,
        {
            "location_name": location_router.DISNEYLAND_PARK_SHORTCUT["name"],
            "place_id": location_router.DISNEYLAND_PARK_SHORTCUT["place_id"],
        },
    )

    meta = {
        "world": {
            "type": "real",
            "primary_city": location_router.DISNEYLAND_PARK_SHORTCUT["city"],
            "region": location_router.DISNEYLAND_PARK_SHORTCUT["region"],
            "country": location_router.DISNEYLAND_PARK_SHORTCUT["country"],
        },
        "locationInfo": {
            "geo": {
                "lat": location_router.DISNEYLAND_PARK_SHORTCUT["lat"],
                "lon": location_router.DISNEYLAND_PARK_SHORTCUT["lon"],
                "city": location_router.DISNEYLAND_PARK_SHORTCUT["city"],
                "region": location_router.DISNEYLAND_PARK_SHORTCUT["region"],
                "country": location_router.DISNEYLAND_PARK_SHORTCUT["country"],
            }
        },
        "currentRoleplay": {
            "CurrentLocation": {
                "name": location_router.DISNEYLAND_PARK_SHORTCUT["name"],
                "city": location_router.DISNEYLAND_PARK_SHORTCUT["city"],
                "region": location_router.DISNEYLAND_PARK_SHORTCUT["region"],
                "country": location_router.DISNEYLAND_PARK_SHORTCUT["country"],
                "place_id": location_router.DISNEYLAND_PARK_SHORTCUT["place_id"],
            }
        },
    }

    gemini_called = False

    async def fake_resolve_location_with_gemini(*_args, **_kwargs):
        nonlocal gemini_called
        gemini_called = True
        return ResolveResult(status=STATUS_NOT_FOUND)

    class DummyContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_get_or_create_location(*_args, **_kwargs):
        class DummyLocation:
            id = 101
            location_name = location_router.DISNEYLAND_PARK_SHORTCUT["name"]
            location_type = "venue"

        return DummyLocation()

    async def fake_track_player_movement(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        location_router,
        "resolve_location_with_gemini",
        fake_resolve_location_with_gemini,
    )
    monkeypatch.setattr(
        location_router,
        "get_db_connection_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        location_router,
        "get_or_create_location",
        fake_get_or_create_location,
    )
    monkeypatch.setattr(
        location_router,
        "_track_player_movement",
        fake_track_player_movement,
    )

    result = await location_router.resolve_place_or_travel(
        "I go to Disneyland Park",
        meta,
        store,
        user_id,
        conversation_id,
    )

    await asyncio.sleep(0)

    assert result.status == "exact"
    assert result.candidates
    top_candidate = result.candidates[0]
    assert top_candidate.place.key == location_router.DISNEYLAND_PARK_SHORTCUT["place_id"]
    assert top_candidate.place.address["city"] == location_router.DISNEYLAND_PARK_SHORTCUT["city"]
    assert result.metadata["router"]["disneyland_shortcut"] is True
    assert any(op.get("op") == "poi.navigate" for op in result.operations)
    assert gemini_called is False
