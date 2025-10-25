import os
import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("OPENAI_API_KEY", "test-key")


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, *args, **kwargs):
        return []


class _DummyTransformer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _DummyPooling:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = _DummyTransformer
dummy_models.Pooling = _DummyPooling

dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = _DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules.setdefault("sentence_transformers", dummy_sentence_transformers)
sys.modules.setdefault("sentence_transformers.models", dummy_models)


class _DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, *args, **kwargs):
        return []


class _DummyAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, *args, **kwargs):
        return types.SimpleNamespace(last_hidden_state=[])


dummy_transformers = types.ModuleType("transformers")
dummy_transformers.AutoTokenizer = _DummyAutoTokenizer
dummy_transformers.AutoModel = _DummyAutoModel

sys.modules.setdefault("transformers", dummy_transformers)

nyx_tasks_stub = types.ModuleType("nyx.tasks")
nyx_tasks_stub.__path__ = [str(Path(__file__).resolve().parents[2] / "nyx" / "tasks")]
sys.modules.setdefault("nyx.tasks", nyx_tasks_stub)


async def _dummy_add(*args, **kwargs):
    return "stub"


async def _dummy_query(*args, **kwargs):
    return []


dummy_vector_store = types.ModuleType("nyx.core.memory.vector_store")
dummy_vector_store.add = _dummy_add
dummy_vector_store.query = _dummy_query
dummy_vector_store.VECTOR_DIM = 384

sys.modules.setdefault("nyx.core.memory.vector_store", dummy_vector_store)


class _DummyUpdaterContext:
    def __init__(self, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def initialize(self):
        return None


async def _dummy_apply_universal(ctx, user_id, conversation_id, updates, conn):
    return {"success": True, "updates_applied": 0}


def _dummy_convert_updates(updates):
    return dict(updates or {})


dummy_universal_updater = types.ModuleType("logic.universal_updater_agent")
dummy_universal_updater.UniversalUpdaterContext = _DummyUpdaterContext
dummy_universal_updater.apply_universal_updates_async = _dummy_apply_universal
dummy_universal_updater.convert_updates_for_database = _dummy_convert_updates

sys.modules.setdefault("logic.universal_updater_agent", dummy_universal_updater)


async def _dummy_fetch_snapshot(*args, **kwargs):
    return {}


async def _dummy_persist_snapshot(*args, **kwargs):
    return None


def _dummy_build_snapshot(snapshot):
    return dict(snapshot or {})


dummy_nyx_context = types.ModuleType("nyx.nyx_agent.context")
dummy_nyx_context.fetch_canonical_snapshot = _dummy_fetch_snapshot
dummy_nyx_context.persist_canonical_snapshot = _dummy_persist_snapshot
dummy_nyx_context.build_canonical_snapshot_payload = _dummy_build_snapshot

sys.modules.setdefault("nyx.nyx_agent.context", dummy_nyx_context)

import importlib

from nyx.conversation.snapshot_store import ConversationSnapshotStore

world_tasks = importlib.import_module("nyx.tasks.background.world_tasks")
npc_tasks = importlib.import_module("nyx.tasks.background.npc_tasks")


@pytest.fixture
def world_task_setup(monkeypatch):
    store = ConversationSnapshotStore(namespace="test:world")
    monkeypatch.setattr(world_tasks, "_SNAPSHOTS", store)
    monkeypatch.setattr(world_tasks, "_persist_snapshot", lambda *args, **kwargs: None)

    user_id = "1"
    conversation_id = "2"
    store.put(user_id, conversation_id, {"world_version": 0})

    calls = {}

    class DummyContext:
        def __init__(self, uid, cid):
            calls["context_args"] = (uid, cid)
            self.user_id = uid
            self.conversation_id = cid

        async def initialize(self):
            calls["context_initialized"] = True

    monkeypatch.setattr(world_tasks, "UniversalUpdaterContext", DummyContext)

    def fake_convert(updates):
        calls["converted"] = dict(updates)
        return dict(updates)

    monkeypatch.setattr(world_tasks, "convert_updates_for_database", fake_convert)

    class DummyConnCtx:
        async def __aenter__(self):
            calls["conn_enter"] = True
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            calls["conn_exit"] = True

    monkeypatch.setattr(world_tasks, "get_db_connection_context", lambda: DummyConnCtx())

    return store, calls


def test_apply_universal_invokes_universal_updater(monkeypatch, world_task_setup):
    store, calls = world_task_setup

    async def fake_apply(ctx, user_id, conversation_id, updates, conn):
        calls["apply_args"] = (user_id, conversation_id, updates)
        return {"success": True, "updates_applied": 1, "details": {"roleplay_updates": updates.get("roleplay_updates", {})}}

    monkeypatch.setattr(world_tasks, "apply_universal_updates_async", fake_apply)

    payload = {
        "conversation_id": "2",
        "user_id": "1",
        "turn_id": 42,
        "incoming_world_version": 3,
        "deltas": {"roleplay_updates": {"CurrentLocation": "Arcology Plaza"}},
    }

    result = world_tasks.apply_universal(payload)

    assert result["status"] == "applied"
    assert result["version"] == 3
    assert result["result"]["updates_applied"] == 1
    assert calls["apply_args"][2]["roleplay_updates"]["CurrentLocation"] == "Arcology Plaza"

    snapshot = store.get("1", "2")
    assert snapshot["world_version"] == 3
    assert snapshot["pending_world_deltas"][0]["turn_id"] == 42
    assert snapshot["last_world_apply"]["result"]["updates_applied"] == 1


def test_apply_universal_propagates_failure(monkeypatch, world_task_setup):
    store, calls = world_task_setup

    async def failing_apply(ctx, user_id, conversation_id, updates, conn):
        calls["apply_args"] = (user_id, conversation_id, updates)
        return {"success": False, "error": "canon rejected"}

    monkeypatch.setattr(world_tasks, "apply_universal_updates_async", failing_apply)

    payload = {
        "conversation_id": "2",
        "user_id": "1",
        "turn_id": 43,
        "incoming_world_version": 4,
        "deltas": {"roleplay_updates": {"CurrentLocation": "Lower Deck"}},
    }

    with pytest.raises(RuntimeError):
        world_tasks.apply_universal(payload)

    snapshot = store.get("1", "2")
    assert snapshot.get("world_version", 0) == 0
    assert "pending_world_deltas" not in snapshot


@pytest.fixture
def npc_task_setup(monkeypatch):
    store = ConversationSnapshotStore(namespace="test:npc")
    monkeypatch.setattr(npc_tasks, "_SNAPSHOTS", store)
    monkeypatch.setattr(npc_tasks, "_persist_snapshot", lambda *args, **kwargs: None)

    user_id = "10"
    conversation_id = "20"
    store.put(user_id, conversation_id, {})

    state = {
        "event_result": {"event_processed": True},
        "cycle_result": {"cycle_completed": True, "npc_adaptations": {1: {"memory_learning": {"intensity_change": 2}}}},
    }

    class DummyManager:
        def __init__(self, uid, cid):
            state["init_args"] = (uid, cid)

        async def initialize(self):
            state["initialized"] = True

        async def process_event_for_learning(self, event_text, event_type, npc_ids, player_response=None):
            state["event_args"] = {
                "event_text": event_text,
                "event_type": event_type,
                "npc_ids": list(npc_ids),
                "player_response": player_response,
            }
            return state["event_result"]

        async def run_regular_adaptation_cycle(self, npc_ids):
            state["cycle_ids"] = list(npc_ids)
            return state["cycle_result"]

    monkeypatch.setattr(npc_tasks, "NPCLearningManager", DummyManager)

    return store, state


def test_run_adaptation_cycle_invokes_learning_manager(monkeypatch, npc_task_setup):
    store, state = npc_task_setup

    payload = {
        "conversation_id": "20",
        "user_id": "10",
        "turn_id": 5,
        "npcs": [1],
        "payload": {"event_text": "Player taunted the guard", "event_type": "taunt"},
    }

    result = npc_tasks.run_adaptation_cycle(payload)

    assert result["status"] == "applied"
    assert state["cycle_ids"] == [1]
    assert state["event_args"]["event_text"] == "Player taunted the guard"

    snapshot = store.get("10", "20")
    assert snapshot["npc_events"][0]["result"]["npc_adaptations"][1]["memory_learning"]["intensity_change"] == 2


def test_run_adaptation_cycle_raises_on_failure(monkeypatch, npc_task_setup):
    store, state = npc_task_setup
    state["cycle_result"] = {"cycle_completed": False, "error": "learning stalled"}

    payload = {
        "conversation_id": "20",
        "user_id": "10",
        "turn_id": 6,
        "npcs": [1],
        "payload": {"event_text": "Player ignored orders", "event_type": "defiance"},
    }

    with pytest.raises(RuntimeError):
        npc_tasks.run_adaptation_cycle(payload)

    snapshot = store.get("10", "20")
    assert snapshot == {}
