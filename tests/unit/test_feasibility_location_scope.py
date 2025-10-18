import importlib
import os
import sys
import types
import typing
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import typing_extensions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

typing.TypedDict = typing_extensions.TypedDict

dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: None
dummy_models.Pooling = lambda *args, **kwargs: None


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 3


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from nyx.location.types import DEFAULT_REALM
from nyx.nyx_agent.context import NyxContext

feasibility = importlib.import_module("nyx.nyx_agent.feasibility")


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeConnection:
    def __init__(self, *, user_record, other_record):
        self._records = {
            (user_record["user_id"], user_record["conversation_id"], user_record["location_name"]): user_record,
            (other_record["user_id"], other_record["conversation_id"], other_record["location_name"]): other_record,
        }
        self.location_queries = []
        self.known_location_queries = []

    async def fetch(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT key, value FROM CurrentRoleplay"):
            return [{"key": "CurrentLocation", "value": "Shared Base"}]
        if normalized.startswith("SELECT rule_name, condition, effect FROM GameRules"):
            return []
        if normalized.startswith("SELECT item_name, equipped FROM PlayerInventory"):
            return []
        if normalized.startswith("SELECT location_name FROM Locations"):
            assert "user_id=$1" in normalized and "conversation_id=$2" in normalized
            expected_user_id, expected_conversation_id = args
            self.known_location_queries.append(args)
            assert expected_user_id == 101
            assert expected_conversation_id == 202
            return [
                {"location_name": "Shared Base"},
            ]
        if normalized.startswith("SELECT npc_name FROM NPCStats"):
            return []
        if normalized.startswith("SELECT content FROM messages"):
            return []
        return []

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT * FROM Locations"):
            assert "user_id=$1" in normalized and "conversation_id=$2" in normalized
            assert "location_name=$3" in normalized
            self.location_queries.append(args)
            return self._records.get((args[0], args[1], args[2]))
        if normalized.startswith("SELECT value FROM CurrentRoleplay"):
            return None
        if normalized.startswith("SELECT * FROM PlayerStats"):
            return None
        return None

    async def fetchval(self, *_args, **_kwargs):
        return None


@pytest.mark.anyio
async def test_load_context_scopes_location_by_user(monkeypatch):
    user_location = {
        "id": 1,
        "user_id": 101,
        "conversation_id": 202,
        "location_name": "Shared Base",
        "description": "User specific description",
        "location_type": "bunker",
        "planet": "Mars",
        "galaxy": "Milky Way",
        "realm": DEFAULT_REALM,
    }
    other_user_location = {
        "id": 2,
        "user_id": 999,
        "conversation_id": 202,
        "location_name": "Shared Base",
        "description": "Other user description",
        "location_type": "bunker",
        "planet": "Venus",
        "galaxy": "Milky Way",
        "realm": DEFAULT_REALM,
    }

    connection = FakeConnection(user_record=user_location, other_record=other_user_location)

    @asynccontextmanager
    async def fake_db_context():
        yield connection

    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    ctx = NyxContext(user_id=101, conversation_id=202)
    context_bundle = await feasibility._load_comprehensive_context(ctx)

    assert connection.location_queries == [(101, 202, "Shared Base")]
    assert connection.known_location_queries == [(101, 202)]
    location_obj = context_bundle["location_object"]
    assert location_obj is not None
    assert location_obj.user_id == 101
    assert location_obj.description == "User specific description"
    assert location_obj.planet == "Mars"
    assert context_bundle["known_location_names"] == ["Shared Base"]
