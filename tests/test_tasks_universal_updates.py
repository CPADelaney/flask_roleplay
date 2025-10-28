"""Regression tests for Celery universal update tasks."""

import asyncio
import os
import pathlib
import sys
import types
from contextlib import contextmanager

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def encode(self, texts, **kwargs):
        return [[0.0] * self._dim for _ in texts]

    def get_sentence_embedding_dimension(self):
        return self._dim


class DummyTransformer:
    def __init__(self, *args, **kwargs):
        pass


class DummyPooling:
    def __init__(self, *args, **kwargs):
        pass


dummy_models_v2 = types.ModuleType("sentence_transformers.models")
dummy_models_v2.Transformer = DummyTransformer
dummy_models_v2.Pooling = DummyPooling

dummy_sentence_transformers_v2 = types.ModuleType("sentence_transformers")
dummy_sentence_transformers_v2.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers_v2.models = dummy_models_v2

sys.modules.setdefault("sentence_transformers", dummy_sentence_transformers_v2)
sys.modules.setdefault("sentence_transformers.models", dummy_models_v2)


class DummyOpenAIChat:
    class Completions:
        @staticmethod
        def create(*args, **kwargs):
            message = types.SimpleNamespace(content="stub response")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    completions = Completions()


class DummyOpenAIClient:
    chat = DummyOpenAIChat()


def _fake_openai_client():
    return DummyOpenAIClient()


async def _fake_chatgpt_response(*args, **kwargs):
    return {"response": "stub"}


dummy_chatgpt = types.ModuleType("logic.chatgpt_integration")
dummy_chatgpt.get_chatgpt_response = _fake_chatgpt_response
dummy_chatgpt.get_openai_client = _fake_openai_client

logic_module = types.ModuleType("logic")
logic_module.chatgpt_integration = dummy_chatgpt


@contextmanager
def _trace(*args, **kwargs):
    yield


@contextmanager
def _custom_span(*args, **kwargs):
    yield


class _RunContextWrapper:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


dummy_agents = types.ModuleType("agents")
dummy_agents.trace = _trace
dummy_agents.custom_span = _custom_span
dummy_agents.RunContextWrapper = _RunContextWrapper

dummy_agents_tracing = types.ModuleType("agents.tracing")
dummy_agents_tracing.get_current_trace = lambda: None


class _NewGameAgent:
    async def process_preset_game_direct(self, *args, **kwargs):
        return {}

    async def process_new_game(self, *args, **kwargs):
        return {}


dummy_new_game_agent = types.ModuleType("new_game_agent")
dummy_new_game_agent.NewGameAgent = _NewGameAgent


class _NPCLearningManager:
    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        return None

    async def run_regular_adaptation_cycle(self, *args, **kwargs):
        return None


dummy_npcs_learning = types.ModuleType("npcs.npc_learning_adaptation")
dummy_npcs_learning.NPCLearningManager = _NPCLearningManager

dummy_npcs = types.ModuleType("npcs")
dummy_npcs.npc_learning_adaptation = dummy_npcs_learning


async def _run_maintenance_through_nyx(*args, **kwargs):
    return None


dummy_memory_nyx = types.ModuleType("memory.memory_nyx_integration")
dummy_memory_nyx.run_maintenance_through_nyx = _run_maintenance_through_nyx


class _NyxBrain:
    async def initialize(self):
        return None


class _CheckpointingPlannerAgent:
    async def plan(self, *args, **kwargs):
        return None


dummy_nyx_core_brain_base = types.ModuleType("nyx.core.brain.base")
dummy_nyx_core_brain_base.NyxBrain = _NyxBrain

dummy_nyx_core_brain_checkpoint = types.ModuleType("nyx.core.brain.checkpointing_agent")
dummy_nyx_core_brain_checkpoint.CheckpointingPlannerAgent = _CheckpointingPlannerAgent

dummy_nyx_core_brain = types.ModuleType("nyx.core.brain")
dummy_nyx_core_brain.base = dummy_nyx_core_brain_base
dummy_nyx_core_brain.checkpointing_agent = dummy_nyx_core_brain_checkpoint

dummy_nyx_core = types.ModuleType("nyx.core")
dummy_nyx_core.brain = dummy_nyx_core_brain


class _NyxSDKConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NyxAgentSDK:
    def __init__(self, config):
        self.config = config

    async def initialize_agent(self):
        return None


dummy_nyx_sdk = types.ModuleType("nyx.nyx_agent_sdk")
dummy_nyx_sdk.NyxSDKConfig = _NyxSDKConfig
dummy_nyx_sdk.NyxAgentSDK = _NyxAgentSDK

dummy_nyx = types.ModuleType("nyx")
dummy_nyx.core = dummy_nyx_core
dummy_nyx.nyx_agent_sdk = dummy_nyx_sdk

sys.modules.setdefault("agents", dummy_agents)
sys.modules.setdefault("agents.tracing", dummy_agents_tracing)
sys.modules.setdefault("logic", logic_module)
sys.modules.setdefault("logic.chatgpt_integration", dummy_chatgpt)
sys.modules.setdefault("new_game_agent", dummy_new_game_agent)
sys.modules.setdefault("npcs", dummy_npcs)
sys.modules.setdefault("npcs.npc_learning_adaptation", dummy_npcs_learning)
sys.modules.setdefault("memory.memory_nyx_integration", dummy_memory_nyx)
sys.modules.setdefault("nyx", dummy_nyx)
sys.modules.setdefault("nyx.core", dummy_nyx_core)
sys.modules.setdefault("nyx.core.brain", dummy_nyx_core_brain)
sys.modules.setdefault("nyx.core.brain.base", dummy_nyx_core_brain_base)
sys.modules.setdefault("nyx.core.brain.checkpointing_agent", dummy_nyx_core_brain_checkpoint)
sys.modules.setdefault("nyx.nyx_agent_sdk", dummy_nyx_sdk)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import tasks


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _run_sync(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_world_update_universal_task_applies_updates(monkeypatch):
    calls: dict[str, object] = {}

    class DummyContext:
        def __init__(self, user_id, conversation_id):
            calls["ctx_args"] = (user_id, conversation_id)

        async def initialize(self):
            calls["ctx_initialized"] = True

    async def fake_apply(ctx, user_id, conversation_id, updates, conn):
        calls["apply_args"] = (ctx, user_id, conversation_id, updates, conn)
        return {"success": True, "updates_applied": 1}

    dummy_module = types.ModuleType("logic.universal_updater_agent")
    dummy_module.UniversalUpdaterContext = DummyContext
    dummy_module.apply_universal_updates_async = fake_apply
    monkeypatch.setitem(sys.modules, "logic.universal_updater_agent", dummy_module)

    async def fake_get_user(conv_id: int) -> int:
        calls["resolved_user"] = conv_id + 100
        return conv_id + 100

    monkeypatch.setattr(tasks, "_get_user_id_for_conversation", fake_get_user)

    class DummyConnCtx:
        async def __aenter__(self):
            calls["conn_enter"] = True
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            calls["conn_exit"] = True

    monkeypatch.setattr(tasks, "get_db_connection_context", lambda: DummyConnCtx())
    monkeypatch.setattr(tasks, "run_async_in_worker_loop", _run_sync)

    response = {"roleplay_updates": {"CurrentLocation": "Orbital Cafe"}}

    result = tasks.world_update_universal_task("5", response)

    assert result == {"ok": True, "conversation_id": 5}
    assert calls["resolved_user"] == 105
    assert calls["ctx_args"] == (105, 5)
    assert calls["ctx_initialized"] is True
    apply_ctx, user_id, conv_id, updates, conn = calls["apply_args"]
    assert isinstance(apply_ctx, DummyContext)
    assert user_id == 105
    assert conv_id == 5
    assert updates is response
    assert conn is not None
    assert calls["conn_enter"] is True
    assert calls["conn_exit"] is True

