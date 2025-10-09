import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from time import perf_counter
import types

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
            {"name": "Quick Plaza", "description": "Central hub", "type": "public"}
        ],
    }

    caplog.set_level(logging.INFO)

    async def run_creation():
        start = perf_counter()
        locations = await new_game_agent.NewGameAgent._create_preset_locations(agent, ctx, preset_payload)
        elapsed = perf_counter() - start
        return locations, elapsed

    locations, elapsed = asyncio.run(run_creation())

    assert locations == ["Quick Plaza"]
    assert elapsed < 1.0, "fallback should avoid long waits"
    assert stub_conn.insert_calls, "fallback path should write to Locations"

    fallback_logs = [record for record in caplog.records if record.getMessage() == "preset_location_bootstrap_lightweight_path"]
    assert fallback_logs, "should log lightweight path usage"
    assert getattr(fallback_logs[0], "location_name", None) == "Quick Plaza"
    assert getattr(fallback_logs[0], "reason", "") in {"timeout", "lightweight_mode_active"}
