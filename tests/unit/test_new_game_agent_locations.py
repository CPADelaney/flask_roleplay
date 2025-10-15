import asyncio
import importlib
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from time import perf_counter
import types

import asyncpg
import pytest

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


class StubConnection:
    def __init__(self):
        self.insert_calls = []

    async def fetchrow(self, query, *args):
        normalized_query = " ".join(query.split())
        if normalized_query.startswith("INSERT INTO Locations"):
            self.insert_calls.append((normalized_query, args))
            return {"location_name": args[2]}
        raise AssertionError(f"Unexpected fetchrow: {normalized_query}")

    async def fetch(self, *args, **kwargs):
        raise AssertionError("fetch should not be called in fallback path")

    async def fetchval(self, *args, **kwargs):
        raise AssertionError("fetchval should not be called in fallback path")

    async def execute(self, *args, **kwargs):
        raise AssertionError("execute should not be called in fallback path")


class MigrationAwareStubConnection:
    """Stub connection that simulates migration state transitions."""

    def __init__(self):
        self.insert_calls = []
        self.executed = []
        self.constraint_exists = False
        self.constraint_name = None
        self.index_exists = False
        self.index_name = None

    async def fetch(self, query, *args):
        normalized_query = " ".join(query.split())
        if (
            "FROM Locations" in normalized_query
            and "HAVING COUNT(*) > 1" in normalized_query
        ):
            return []
        return []

    async def fetchrow(self, query, *args):
        normalized_query = " ".join(query.split())
        if "FROM pg_constraint" in normalized_query and "con.contype = 'u'" in normalized_query:
            if self.constraint_exists:
                return {"constraint_name": self.constraint_name, "is_valid": True}
            return None
        if "FROM pg_index" in normalized_query and "idx.indisunique" in normalized_query:
            if self.index_exists:
                return {"index_name": self.index_name, "is_valid": True}
            return None
        if normalized_query.startswith("INSERT INTO Locations"):
            if not self.constraint_exists:
                raise asyncpg.exceptions.UndefinedObjectError(
                    "constraint idx_locations_user_conversation_name does not exist"
                )
            self.insert_calls.append((normalized_query, args))
            return {"location_name": args[2]}
        return None

    async def fetchval(self, query, *args):
        if "string_agg(format('%I'" in query:
            quoted_parts = []
            for part in args[0]:
                escaped = part.replace('"', '""')
                quoted_parts.append(f'"{escaped}"')
            return ".".join(quoted_parts)
        return None

    async def execute(self, query, *args):
        normalized_query = " ".join(query.split())
        self.executed.append(normalized_query)

        if "ALTER TABLE" in normalized_query and "ADD CONSTRAINT" in normalized_query:
            self.constraint_exists = True
            self.constraint_name = "idx_locations_user_conversation_name"
            self.index_exists = True
            self.index_name = "idx_locations_user_conversation_name"
            return None
        if "ALTER TABLE" in normalized_query and "RENAME CONSTRAINT" in normalized_query:
            self.constraint_exists = True
            self.constraint_name = "idx_locations_user_conversation_name"
            return None
        if "ALTER TABLE" in normalized_query and "VALIDATE CONSTRAINT" in normalized_query:
            return None
        if "ALTER INDEX" in normalized_query and "RENAME TO" in normalized_query:
            self.index_exists = True
            self.index_name = "idx_locations_user_conversation_name"
            return None
        if "DROP INDEX" in normalized_query and "idx_locations_user_conversation_name" in normalized_query:
            self.index_exists = False
            return None
        return None


class ManualUpsertStubConnection:
    """Stub connection that mimics manual upsert behaviour when the constraint is missing."""

    def __init__(self):
        self.locations = {}
        self.insert_attempts = 0
        self.manual_inserts = 0
        self.update_calls = 0
        self.index_dropped = False
        self.executed = []

    def seed_location(self, user_id, conversation_id, name, description, location_type, open_hours):
        self.locations[(user_id, conversation_id, name)] = {
            "description": description,
            "location_type": location_type,
            "open_hours": open_hours,
        }

    async def fetch(self, *args, **kwargs):
        raise AssertionError("fetch should not be called in manual upsert stub")

    async def fetchval(self, *args, **kwargs):
        raise AssertionError("fetchval should not be called in manual upsert stub")

    async def fetchrow(self, query, *args):
        normalized_query = " ".join(query.split())

        if normalized_query.startswith("INSERT INTO Locations") and "ON CONFLICT" in normalized_query:
            self.insert_attempts += 1
            raise asyncpg.UndefinedObjectError(
                "constraint idx_locations_user_conversation_name does not exist"
            )

        if normalized_query.startswith("SELECT location_name"):
            key = (args[0], args[1], args[2])
            current = self.locations.get(key)
            if current is None:
                return None
            return {"location_name": key[2], **current}

        if normalized_query.startswith("UPDATE Locations"):
            key = (args[0], args[1], args[2])
            description = args[3]
            location_type = args[4]
            open_hours = args[5]
            existing = self.locations.get(key, {})
            if open_hours is None:
                open_hours = existing.get("open_hours")
            self.locations[key] = {
                "description": description,
                "location_type": location_type,
                "open_hours": open_hours,
            }
            self.update_calls += 1
            return {"location_name": key[2], **self.locations[key]}

        if normalized_query.startswith("INSERT INTO Locations"):
            key = (args[0], args[1], args[2])
            self.locations[key] = {
                "description": args[3],
                "location_type": args[4],
                "open_hours": args[5],
            }
            self.manual_inserts += 1
            return {"location_name": key[2], **self.locations[key]}

        raise AssertionError(f"Unexpected fetchrow: {normalized_query}")

    async def execute(self, query, *args):
        normalized_query = " ".join(query.split())
        self.executed.append(normalized_query)
        if normalized_query.startswith("DROP INDEX"):
            self.index_dropped = True
        return None


def test_create_preset_locations_uses_fallback(monkeypatch, caplog):
    agent = new_game_agent.NewGameAgent.__new__(new_game_agent.NewGameAgent)

    stub_conn = StubConnection()

    @asynccontextmanager
    async def fake_connection_context():
        yield stub_conn

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_connection_context)

    async def slow_find_or_create(*args, **kwargs):
        await asyncio.sleep(0.2)
        return "should_not_be_reached"

    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", slow_find_or_create)

    async def fast_orchestrator(user_id, conversation_id):
        return object()

    monkeypatch.setattr(new_game_agent.canon, "get_canon_memory_orchestrator", fast_orchestrator)

    monkeypatch.setattr(new_game_agent, "PRESET_LOCATION_CANON_TIMEOUT", 0.05, raising=False)

    ctx = types.SimpleNamespace(context={"user_id": 42, "conversation_id": 99})
    preset_payload = {
        "name": "Bootstrap",
        "required_locations": [
            {"name": "Quick Plaza", "description": "Central hub", "type": "public"},
            {
                "name": "Schedule Hall",
                "description": "Detailed coordination center",
                "type": "administrative",
                "open_hours": {"Mon": "09:00-17:00"},
            },
        ],
    }

    caplog.set_level(logging.INFO)

    async def run_creation():
        start = perf_counter()
        locations = await new_game_agent.NewGameAgent._create_preset_locations(agent, ctx, preset_payload)
        elapsed = perf_counter() - start
        return locations, elapsed

    locations, elapsed = asyncio.run(run_creation())

    assert locations == ["Quick Plaza", "Schedule Hall"]
    assert elapsed < 1.0, "fallback should avoid long waits"
    assert stub_conn.insert_calls, "fallback path should write to Locations"
    assert len(stub_conn.insert_calls) == 2

    fallback_logs = [record for record in caplog.records if record.getMessage() == "preset_location_bootstrap_lightweight_path"]
    assert fallback_logs, "should log lightweight path usage"
    assert getattr(fallback_logs[0], "location_name", None) == "Quick Plaza"
    assert getattr(fallback_logs[0], "reason", "") in {"timeout", "lightweight_mode_active"}

    _, first_insert_args = stub_conn.insert_calls[0]
    _, second_insert_args = stub_conn.insert_calls[1]
    assert first_insert_args[2] == "Quick Plaza"
    assert second_insert_args[2] == "Schedule Hall"
    assert second_insert_args[5] == json.dumps({"Mon": "09:00-17:00"})


def test_locations_migration_allows_on_constraint_fallback(monkeypatch):
    agent = new_game_agent.NewGameAgent.__new__(new_game_agent.NewGameAgent)

    migration_module = importlib.import_module(
        "db.migrations.002_locations_unique_constraint_guardrail"
    )

    stub_conn = MigrationAwareStubConnection()

    @asynccontextmanager
    async def fake_connection_context():
        yield stub_conn

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_connection_context)

    async def fast_orchestrator(user_id, conversation_id):
        return object()

    async def slow_find_or_create(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return "should_not_be_reached"

    monkeypatch.setattr(new_game_agent.canon, "get_canon_memory_orchestrator", fast_orchestrator)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", slow_find_or_create)
    monkeypatch.setattr(new_game_agent, "PRESET_LOCATION_CANON_TIMEOUT", 0.05, raising=False)

    ctx = types.SimpleNamespace(context={"user_id": 7, "conversation_id": 11})
    preset_payload = {
        "name": "Bootstrap",
        "required_locations": [{"name": "Fallback Plaza", "description": "Plaza"}],
    }

    async def run_flow():
        await migration_module.upgrade(stub_conn)
        assert stub_conn.constraint_exists, "Migration should create the named constraint"

        locations = await new_game_agent.NewGameAgent._create_preset_locations(
            agent, ctx, preset_payload
        )
        return locations

    locations = asyncio.run(run_flow())

    assert locations == ["Fallback Plaza"]
    assert stub_conn.insert_calls
    normalized_query, _ = stub_conn.insert_calls[0]
    assert "ON CONFLICT (user_id, conversation_id, location_name)" in normalized_query
    assert any("ADD CONSTRAINT" in stmt for stmt in stub_conn.executed)


def test_manual_upsert_when_unique_constraint_missing(monkeypatch, caplog):
    agent = new_game_agent.NewGameAgent.__new__(new_game_agent.NewGameAgent)

    stub_conn = ManualUpsertStubConnection()
    user_id = 3
    conversation_id = 14
    location_name = "Merge Plaza"
    original_hours = json.dumps({"Mon": "08:00-18:00"})
    stub_conn.seed_location(
        user_id,
        conversation_id,
        location_name,
        "Legacy description",
        "market",
        original_hours,
    )

    @asynccontextmanager
    async def fake_connection_context():
        yield stub_conn

    async def slow_find_or_create(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return "should_not_be_reached"

    async def fast_orchestrator(user_id, conversation_id):
        return object()

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_connection_context)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", slow_find_or_create)
    monkeypatch.setattr(new_game_agent.canon, "get_canon_memory_orchestrator", fast_orchestrator)
    monkeypatch.setattr(new_game_agent, "PRESET_LOCATION_CANON_TIMEOUT", 0.05, raising=False)

    caplog.set_level(logging.WARNING)

    ctx = types.SimpleNamespace(context={"user_id": user_id, "conversation_id": conversation_id})
    preset_payload = {
        "name": "Bootstrap",
        "required_locations": [
            {
                "name": location_name,
                "description": "Refreshed description",
                "type": "plaza",
                "schedule": None,
                # Intentionally omit open_hours to ensure the coalesce path is used.
            }
        ],
    }

    async def run_flow():
        await stub_conn.execute("DROP INDEX IF EXISTS idx_locations_user_conversation_name")
        return await new_game_agent.NewGameAgent._create_preset_locations(agent, ctx, preset_payload)

    locations = asyncio.run(run_flow())

    assert locations == [location_name]
    assert stub_conn.insert_attempts == 1
    assert stub_conn.update_calls == 1
    assert stub_conn.manual_inserts == 0
    assert stub_conn.index_dropped

    stored = stub_conn.locations[(user_id, conversation_id, location_name)]
    assert stored["description"] == "Refreshed description"
    assert stored["location_type"] == "plaza"
    assert stored["open_hours"] == original_hours

    warnings = [
        record
        for record in caplog.records
        if "run the Locations uniqueness migration" in record.getMessage()
    ]
    assert warnings, "Expected warning about missing Locations uniqueness migration"
