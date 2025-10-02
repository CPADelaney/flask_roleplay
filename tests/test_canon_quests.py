import json
import os
import sys
import types
from contextlib import asynccontextmanager

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def _install_stub_modules():
    db_pkg = types.ModuleType("db")
    db_pkg.__path__ = []  # type: ignore[attr-defined]

    db_connection = types.ModuleType("db.connection")

    class _DBConn:
        async def fetchval(self, *args, **kwargs):
            return None

    @asynccontextmanager
    async def _db_context():
        yield _DBConn()

    async def _track_operation(coro):
        return await coro

    db_connection.get_db_connection_context = _db_context  # type: ignore[attr-defined]
    db_connection.is_shutting_down = lambda: False  # type: ignore[attr-defined]
    db_connection.track_operation = _track_operation  # type: ignore[attr-defined]

    db_pkg.connection = db_connection  # type: ignore[attr-defined]

    nyx_pkg = types.ModuleType("nyx")
    nyx_pkg.__path__ = []  # type: ignore[attr-defined]

    nyx_governance = types.ModuleType("nyx.nyx_governance")

    class _AgentType:
        def __getattr__(self, name):
            return name.lower()

    nyx_governance.AgentType = _AgentType()  # type: ignore[attr-defined]

    nyx_helpers = types.ModuleType("nyx.governance_helpers")

    def _with_governance(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    nyx_helpers.with_governance = _with_governance  # type: ignore[attr-defined]

    embedding_pkg = types.ModuleType("embedding")
    embedding_pkg.__path__ = []  # type: ignore[attr-defined]

    embedding_vector_store = types.ModuleType("embedding.vector_store")

    async def _generate_embedding(text):
        return [0.0, 0.0, 0.0]

    embedding_vector_store.generate_embedding = _generate_embedding  # type: ignore[attr-defined]

    agents_module = types.ModuleType("agents")

    class _Runner:
        pass

    agents_module.Runner = _Runner  # type: ignore[attr-defined]

    memory_pkg = types.ModuleType("memory")
    memory_pkg.__path__ = []  # type: ignore[attr-defined]

    memory_orchestrator = types.ModuleType("memory.memory_orchestrator")

    async def _get_memory_orchestrator(user_id, conversation_id):
        class _Orchestrator:
            async def ensure_canon_synced(self):
                return None

            async def store_memory(self, *args, **kwargs):
                return None

        return _Orchestrator()

    memory_orchestrator.get_memory_orchestrator = _get_memory_orchestrator  # type: ignore[attr-defined]
    memory_orchestrator.EntityType = type("EntityType", (), {"LORE": "lore"})  # type: ignore[attr-defined]

    repo_root = os.path.dirname(os.path.dirname(__file__))

    lore_pkg = types.ModuleType("lore")
    lore_pkg.__path__ = [os.path.join(repo_root, "lore")]  # type: ignore[attr-defined]

    lore_core_pkg = types.ModuleType("lore.core")
    lore_core_pkg.__path__ = [os.path.join(repo_root, "lore", "core")]  # type: ignore[attr-defined]

    lore_pkg.core = lore_core_pkg  # type: ignore[attr-defined]

    lore_core_validation = types.ModuleType("lore.core.validation")

    class _CanonValidationAgent:
        pass

    lore_core_validation.CanonValidationAgent = _CanonValidationAgent  # type: ignore[attr-defined]

    sys.modules.setdefault("db", db_pkg)
    sys.modules.setdefault("db.connection", db_connection)
    sys.modules.setdefault("nyx", nyx_pkg)
    sys.modules.setdefault("nyx.nyx_governance", nyx_governance)
    sys.modules.setdefault("nyx.governance_helpers", nyx_helpers)
    sys.modules.setdefault("embedding", embedding_pkg)
    sys.modules.setdefault("embedding.vector_store", embedding_vector_store)
    sys.modules.setdefault("agents", agents_module)
    sys.modules.setdefault("memory", memory_pkg)
    sys.modules.setdefault("memory.memory_orchestrator", memory_orchestrator)
    sys.modules.setdefault("lore", lore_pkg)
    sys.modules.setdefault("lore.core", lore_core_pkg)
    sys.modules.setdefault("lore.core.validation", lore_core_validation)


_install_stub_modules()

from lore.core.canon import find_or_create_quest
from lore.core.context import CanonicalContext


class QuestConnectionStub:
    def __init__(self, existing_quest_id=None, inserted_id=101):
        self.existing_quest_id = existing_quest_id
        self.inserted_id = inserted_id
        self.fetchrow_calls = []
        self.fetchval_calls = []
        self.fetch_calls = []

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())
        self.fetchrow_calls.append((normalized, args))

        if normalized.startswith("SELECT quest_id FROM Quests"):
            if self.existing_quest_id is None:
                return None
            return {"quest_id": self.existing_quest_id}

        raise AssertionError(f"Unexpected fetchrow: {normalized}")

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        self.fetchval_calls.append((normalized, args))

        if normalized.startswith("INSERT INTO Quests"):
            return self.inserted_id
        if normalized.startswith("INSERT INTO CanonicalEvents"):
            return 1

        raise AssertionError(f"Unexpected fetchval: {normalized}")

    async def fetch(self, query, *args):
        normalized = " ".join(query.split())
        self.fetch_calls.append((normalized, args))

        if normalized.startswith("SELECT id FROM CanonicalEvents"):
            return []

        raise AssertionError(f"Unexpected fetch: {normalized}")


@pytest.mark.asyncio
async def test_find_or_create_quest_serializes_default_reward():
    ctx = CanonicalContext(user_id=1, conversation_id=2)
    conn = QuestConnectionStub()

    quest_id = await find_or_create_quest(ctx, conn, "Test Quest")

    assert quest_id == conn.inserted_id
    assert conn.fetchval_calls, "Quest insertion should have been attempted"

    inserted_reward = conn.fetchval_calls[0][1][-1]
    assert isinstance(inserted_reward, str)
    assert inserted_reward == json.dumps([])
    assert json.loads(inserted_reward) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reward",
    [
        ["gold", "sword"],
        {"xp": 100, "items": ["amulet"]},
        "mysterious artifact",
    ],
)
async def test_find_or_create_quest_serializes_provided_reward(reward):
    ctx = CanonicalContext(user_id=3, conversation_id=4)
    conn = QuestConnectionStub(inserted_id=202)

    quest_id = await find_or_create_quest(ctx, conn, "Rewarded Quest", reward=reward)

    assert quest_id == conn.inserted_id
    inserted_reward = conn.fetchval_calls[0][1][-1]

    assert isinstance(inserted_reward, str)
    assert inserted_reward == json.dumps(reward)
    assert json.loads(inserted_reward) == reward
