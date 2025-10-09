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
                raise asyncpg.exceptions.InvalidColumnReferenceError(
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
    assert "ON CONSTRAINT idx_locations_user_conversation_name" in normalized_query
    assert any("ADD CONSTRAINT" in stmt for stmt in stub_conn.executed)
