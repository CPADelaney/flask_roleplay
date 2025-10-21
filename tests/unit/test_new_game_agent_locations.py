import asyncio
import json
import logging
import os
import sys
import types
from contextlib import asynccontextmanager

import asyncpg

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: object()
dummy_models.Pooling = lambda *args, **kwargs: object()


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def encode(self, texts, **kwargs):
        return [[0.0] * self._dim for _ in texts]

    def get_sentence_embedding_dimension(self):
        return self._dim


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models

import new_game_agent
from nyx.location import hierarchy as location_hierarchy
from nyx.location.types import Location, DEFAULT_REALM


def test_create_preset_locations_uses_factory(monkeypatch, caplog):
    agent = new_game_agent.NewGameAgent.__new__(new_game_agent.NewGameAgent)

    calls = []

    @asynccontextmanager
    async def fake_connection_context():
        yield object()

    async def slow_find_or_create(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return "should_not_happen"

    async def fast_orchestrator(user_id, conversation_id):
        return object()

    async def fake_get_or_create_location(
        conn,
        *,
        user_id: int,
        conversation_id: int,
        candidate,
        scope: str,
        **kwargs,
    ) -> Location:
        calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "candidate": candidate,
                "scope": scope,
            }
        )
        display_name = candidate.place.meta.get("display_name", candidate.place.name)
        normalized = location_hierarchy._normalize_location_name(display_name)
        return Location(user_id=user_id, conversation_id=conversation_id, location_name=normalized)

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_connection_context)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", slow_find_or_create)
    monkeypatch.setattr(new_game_agent.canon, "get_canon_memory_orchestrator", fast_orchestrator)
    monkeypatch.setattr(new_game_agent, "get_or_create_location", fake_get_or_create_location)
    monkeypatch.setattr(new_game_agent, "PRESET_LOCATION_CANON_TIMEOUT", 0.05, raising=False)

    ctx = types.SimpleNamespace(context={"user_id": 7, "conversation_id": 11})
    preset_payload = {
        "name": "Bootstrap",
        "required_locations": [
            {
                "name": "Quick Plaza",
                "description": "Central hub",
                "type": "public",
                "building": "Quick Plaza Pavilion",
                "district": "Central Commons",
                "city": "Evergreen City",
                "region": "Washington",
                "country": "USA",
                "rooms": {"gazebo": "Primary rendezvous point"},
            },
            {
                "name": "Schedule Hall",
                "description": "Detailed coordination center",
                "type": "administrative",
                "open_hours": {"Mon": "09:00-17:00"},
                "building": "Coordination Annex",
                "room": "Operations Desk",
                "district": "Central Commons",
                "city": "Evergreen City",
                "region": "Washington",
                "country": "USA",
            },
        ],
    }

    caplog.set_level(logging.INFO)

    async def run_flow():
        return await new_game_agent.NewGameAgent._create_preset_locations(agent, ctx, preset_payload)

    locations = asyncio.run(run_flow())

    assert locations == ["quick plaza", "quick plaza :: gazebo", "schedule hall"]
    assert len(calls) == len(locations)
    assert all(call["scope"] == "real" for call in calls)

    first_meta = calls[0]["candidate"].place.meta
    room_meta = calls[1]["candidate"].place.meta
    schedule_meta = calls[2]["candidate"].place.meta

    assert first_meta["description"] == "Central hub"
    assert first_meta["building"] == "Quick Plaza Pavilion"
    assert first_meta["district"] == "Central Commons"
    assert first_meta["rooms"]["gazebo"] == "Primary rendezvous point"
    assert first_meta["scope"] == "real"
    assert first_meta["is_fictional"] is False
    assert first_meta["planet"] == "Earth"
    assert first_meta["galaxy"] == "Milky Way"
    assert first_meta["realm"] == DEFAULT_REALM
    assert room_meta["scope"] == "real"
    assert room_meta["is_fictional"] is False
    assert room_meta["parent_location"] == "Quick Plaza"
    assert room_meta["room"] == "Gazebo"

    assert schedule_meta["open_hours"] == {"Mon": "09:00-17:00"}
    assert schedule_meta["room"] == "Operations Desk"
    assert schedule_meta["city"] == "Evergreen City"
    assert schedule_meta["scope"] == "real"
    assert schedule_meta["is_fictional"] is False

    fallback_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "preset_location_bootstrap_lightweight_path"
    ]
    assert fallback_logs, "expected lightweight fallback log"


def test_create_preset_locations_serializes_open_hours(monkeypatch):
    agent = new_game_agent.NewGameAgent.__new__(new_game_agent.NewGameAgent)

    observed_meta = []

    @asynccontextmanager
    async def fake_connection_context():
        yield object()

    async def slow_find_or_create(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return "should_not_happen"

    async def fast_orchestrator(user_id, conversation_id):
        return object()

    async def fake_get_or_create_location(
        conn,
        *,
        user_id: int,
        conversation_id: int,
        candidate,
        scope: str,
        **kwargs,
    ) -> Location:
        observed_meta.append(candidate.place.meta)
        normalized = location_hierarchy._normalize_location_name(candidate.place.name)
        return Location(user_id=user_id, conversation_id=conversation_id, location_name=normalized)

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_connection_context)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", slow_find_or_create)
    monkeypatch.setattr(new_game_agent.canon, "get_canon_memory_orchestrator", fast_orchestrator)
    monkeypatch.setattr(new_game_agent, "get_or_create_location", fake_get_or_create_location)
    monkeypatch.setattr(new_game_agent, "PRESET_LOCATION_CANON_TIMEOUT", 0.05, raising=False)

    ctx = types.SimpleNamespace(context={"user_id": 3, "conversation_id": 14})
    preset_payload = {
        "name": "Bootstrap",
        "required_locations": [
            {
                "name": "Merge Plaza",
                "description": "Refreshed description",
                "type": "plaza",
                "open_hours": json.dumps({"Mon": "08:00-18:00"}),
            }
        ],
    }

    async def run_flow():
        return await new_game_agent.NewGameAgent._create_preset_locations(agent, ctx, preset_payload)

    locations = asyncio.run(run_flow())

    assert locations == ["merge plaza"]
    assert observed_meta
    meta = observed_meta[0]
    assert isinstance(meta.get("open_hours"), dict)
    assert meta["open_hours"] == {"Mon": "08:00-18:00"}


def test_create_preset_locations_handles_nested_open_hours(monkeypatch):
    agent = new_game_agent.NewGameAgent.__new__(new_game_agent.NewGameAgent)

    class FakeConnection:
        def __init__(self):
            self.insert_args = None

        async def fetchrow(self, query, *args):
            if "INSERT INTO Locations" in query:
                self.insert_args = args

                json_field_indexes = {
                    "open_hours": 19,
                    "notable_features": 25,
                    "hidden_aspects": 26,
                    "access_restrictions": 27,
                    "local_customs": 28,
                }
                for index in json_field_indexes.values():
                    value = args[index]
                    if isinstance(value, (dict, list)):
                        raise asyncpg.DataError("JSON value must be serialized before binding")

                return {
                    "id": 42,
                    "user_id": args[0],
                    "conversation_id": args[1],
                    "location_name": args[2],
                }

            return None

        async def execute(self, *_args, **_kwargs):
            return "OK"

    fake_conn = FakeConnection()

    @asynccontextmanager
    async def fake_connection_context():
        yield fake_conn

    async def slow_orchestrator(*_args, **_kwargs):
        await asyncio.sleep(0.2)

    async def forbidden_find_or_create(*_args, **_kwargs):
        raise AssertionError("Canon path should not be invoked in fallback")

    async def fake_assign_hierarchy(*_args, **_kwargs):
        return {"chain": [], "leaf": {"id": 1}, "world_name": "Fictional World"}

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_connection_context)
    monkeypatch.setattr(new_game_agent.canon, "get_canon_memory_orchestrator", slow_orchestrator)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", forbidden_find_or_create)
    monkeypatch.setattr(new_game_agent, "PRESET_LOCATION_CANON_TIMEOUT", 0.05, raising=False)
    monkeypatch.setattr(location_hierarchy, "assign_hierarchy", fake_assign_hierarchy)

    ctx = types.SimpleNamespace(context={"user_id": 9, "conversation_id": 27})
    nested_open_hours = {
        "weekday": {
            "Mon": {"start": "08:00", "end": "17:00"},
            "Tue": {"start": "08:00", "end": "17:00"},
        }
    }
    preset_payload = {
        "name": "Starlight",
        "required_locations": [
            {
                "name": "Aurora Observatory",
                "description": "A serene observation deck with expansive views.",
                "type": "observatory",
                "open_hours": nested_open_hours,
            }
        ],
    }

    async def run_flow():
        return await new_game_agent.NewGameAgent._create_preset_locations(agent, ctx, preset_payload)

    locations = asyncio.run(run_flow())

    assert locations == ["aurora observatory"]
    assert fake_conn.insert_args is not None

    serialized_open_hours = fake_conn.insert_args[19]
    assert isinstance(serialized_open_hours, str)
    assert json.loads(serialized_open_hours) == nested_open_hours

    for index in (25, 26, 27, 28):
        value = fake_conn.insert_args[index]
        if value is not None:
            assert isinstance(value, str)


def test_queen_of_thorns_locations_use_earth_defaults(monkeypatch):
    agent = new_game_agent.NewGameAgent.__new__(new_game_agent.NewGameAgent)

    class FakeConnection:
        def __init__(self):
            self.insert_args = None
            self.insert_history = []

        async def fetchrow(self, query, *args):
            if "INSERT INTO Locations" in query:
                self.insert_args = args
                self.insert_history.append(args)
                return {
                    "id": 7,
                    "user_id": args[0],
                    "conversation_id": args[1],
                    "location_name": args[2],
                    "scope": "real",
                    "planet": args[13],
                    "galaxy": args[14],
                    "realm": args[15],
                    "is_fictional": args[18],
                }
            return None

        async def execute(self, *_args, **_kwargs):
            return "OK"

    fake_conn = FakeConnection()

    @asynccontextmanager
    async def fake_connection_context():
        yield fake_conn

    async def slow_orchestrator(*_args, **_kwargs):
        await asyncio.sleep(0.2)

    async def forbidden_find_or_create(*_args, **_kwargs):
        raise AssertionError("Canon path should not be invoked in fallback")

    async def fake_assign_hierarchy(*_args, **_kwargs):
        return {"chain": [], "leaf": {"id": 1}, "world_name": "Earth"}

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_connection_context)
    monkeypatch.setattr(new_game_agent.canon, "get_canon_memory_orchestrator", slow_orchestrator)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", forbidden_find_or_create)
    monkeypatch.setattr(new_game_agent, "PRESET_LOCATION_CANON_TIMEOUT", 0.05, raising=False)
    monkeypatch.setattr(location_hierarchy, "assign_hierarchy", fake_assign_hierarchy)

    ctx = types.SimpleNamespace(context={"user_id": 5, "conversation_id": 19})

    preset_payload = {
        "name": "Queen of Thorns",
        "required_locations": [
            {
                "name": "Velvet Sanctum",
                "description": "An underground temple hidden beneath San Francisco",
                "type": "nightclub_dungeon",
                "building": "Velvet Sanctum Subterranean Complex",
                "areas": {"main_stage": "Where the Queen holds court"},
                "city": "San Francisco",
                "region": "California",
                "country": "USA",
            }
        ],
    }

    async def run_flow():
        return await new_game_agent.NewGameAgent._create_preset_locations(agent, ctx, preset_payload)

    locations = asyncio.run(run_flow())

    assert locations == ["velvet sanctum", "velvet sanctum :: main stage"]
    assert fake_conn.insert_history

    base_insert = fake_conn.insert_history[0]

    planet_arg = base_insert[13]
    galaxy_arg = base_insert[14]
    realm_arg = base_insert[15]
    is_fictional_arg = base_insert[18]

    assert planet_arg == "Earth"
    assert galaxy_arg == "Milky Way"
    assert realm_arg == DEFAULT_REALM
    assert is_fictional_arg is False
