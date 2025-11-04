from contextlib import contextmanager
import os
import sys
import types
from unittest.mock import Mock

import pytest


@contextmanager
def _noop_context(*args, **kwargs):
    yield


class _RunContextWrapper:
    def __init__(self, context=None):
        self.context = context or {}


dummy_agents = types.ModuleType("agents")
dummy_agents.trace = _noop_context
dummy_agents.custom_span = _noop_context
dummy_agents.RunContextWrapper = _RunContextWrapper
sys.modules.setdefault("agents", dummy_agents)

dummy_agents_tracing = types.ModuleType("agents.tracing")
dummy_agents_tracing.get_current_trace = lambda: None
sys.modules.setdefault("agents.tracing", dummy_agents_tracing)


def _fake_chatgpt_response(*args, **kwargs):
    return {"response": "ok"}


class _FakeOpenAIClient:
    class _Chat:
        class _Completions:
            @staticmethod
            def create(*args, **kwargs):
                message = types.SimpleNamespace(content="stub")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        completions = _Completions()

    chat = _Chat()


def _fake_openai_client():
    return _FakeOpenAIClient()


dummy_logic = types.ModuleType("logic")
dummy_chatgpt = types.ModuleType("logic.chatgpt_integration")
dummy_chatgpt.get_chatgpt_response = _fake_chatgpt_response
dummy_chatgpt.get_openai_client = _fake_openai_client
dummy_logic.chatgpt_integration = dummy_chatgpt
sys.modules.setdefault("logic", dummy_logic)
sys.modules.setdefault("logic.chatgpt_integration", dummy_chatgpt)


class _NewGameAgent:
    async def process_preset_game_direct(self, *args, **kwargs):
        return {}

    async def process_new_game(self, *args, **kwargs):
        return {}


dummy_new_game_agent = types.ModuleType("new_game_agent")
dummy_new_game_agent.NewGameAgent = _NewGameAgent
sys.modules.setdefault("new_game_agent", dummy_new_game_agent)


class _NPCLearningManager:
    def __init__(self, *args, **kwargs):
        pass


dummy_npcs = types.ModuleType("npcs")
dummy_npcs_learning = types.ModuleType("npcs.npc_learning_adaptation")
dummy_npcs_learning.NPCLearningManager = _NPCLearningManager
sys.modules.setdefault("npcs", dummy_npcs)
sys.modules.setdefault("npcs.npc_learning_adaptation", dummy_npcs_learning)


def _run_maintenance_through_nyx(*args, **kwargs):
    return None


dummy_memory = types.ModuleType("memory")
dummy_memory_nyx = types.ModuleType("memory.memory_nyx_integration")
dummy_memory_nyx.run_maintenance_through_nyx = _run_maintenance_through_nyx
dummy_memory.memory_nyx_integration = dummy_memory_nyx
sys.modules.setdefault("memory", dummy_memory)
sys.modules.setdefault("memory.memory_nyx_integration", dummy_memory_nyx)


class _RegionalCultureSystem:
    async def analyze_cultural_conflict(self, *args, **kwargs):
        return {}


dummy_lore = types.ModuleType("lore")
dummy_lore_systems = types.ModuleType("lore.systems")
dummy_lore_regional = types.ModuleType("lore.systems.regional_culture")
dummy_lore_regional.RegionalCultureSystem = _RegionalCultureSystem
dummy_lore_systems.regional_culture = dummy_lore_regional
dummy_lore.systems = dummy_lore_systems
sys.modules.setdefault("lore", dummy_lore)
sys.modules.setdefault("lore.systems", dummy_lore_systems)
sys.modules.setdefault("lore.systems.regional_culture", dummy_lore_regional)


class _NyxBrain:
    initialized = True

    @classmethod
    async def get_instance(cls, *args, **kwargs):
        return cls()

    async def restore_entity_from_distributed_checkpoints(self):
        return True

    async def gather_checkpoint_state(self, *args, **kwargs):
        return {}

    async def save_planned_checkpoint(self, *args, **kwargs):
        return None


dummy_nyx = types.ModuleType("nyx")
dummy_nyx_core = types.ModuleType("nyx.core")
dummy_nyx_core_brain = types.ModuleType("nyx.core.brain")
dummy_nyx_core_brain_base = types.ModuleType("nyx.core.brain.base")
dummy_nyx_core_brain_base.NyxBrain = _NyxBrain
dummy_nyx_core_brain_checkpoint = types.ModuleType("nyx.core.brain.checkpointing_agent")


class _CheckpointingPlannerAgent:
    async def recommend_checkpoint(self, *args, **kwargs):
        return {}


dummy_nyx_core_brain_checkpoint.CheckpointingPlannerAgent = _CheckpointingPlannerAgent
dummy_nyx.core = dummy_nyx_core
dummy_nyx_core.brain = dummy_nyx_core_brain
dummy_nyx_core_brain.base = dummy_nyx_core_brain_base
dummy_nyx_core_brain.checkpointing_agent = dummy_nyx_core_brain_checkpoint
sys.modules.setdefault("nyx", dummy_nyx)
sys.modules.setdefault("nyx.core", dummy_nyx_core)
sys.modules.setdefault("nyx.core.brain", dummy_nyx_core_brain)
sys.modules.setdefault("nyx.core.brain.base", dummy_nyx_core_brain_base)
sys.modules.setdefault("nyx.core.brain.checkpointing_agent", dummy_nyx_core_brain_checkpoint)


class _NyxContext:
    pass


class _NyxSDKConfig:
    def __init__(self, *args, **kwargs):
        pass


class _NyxAgentSDK:
    def __init__(self, config):
        self.config = config

    async def initialize_agent(self):
        return None


dummy_nyx_agent_context = types.ModuleType("nyx.nyx_agent.context")
dummy_nyx_agent_context.NyxContext = _NyxContext
dummy_nyx_sdk = types.ModuleType("nyx.nyx_agent_sdk")
dummy_nyx_sdk.NyxSDKConfig = _NyxSDKConfig
dummy_nyx_sdk.NyxAgentSDK = _NyxAgentSDK
sys.modules.setdefault("nyx.nyx_agent", types.ModuleType("nyx.nyx_agent"))
sys.modules.setdefault("nyx.nyx_agent.context", dummy_nyx_agent_context)
sys.modules.setdefault("nyx.nyx_agent_sdk", dummy_nyx_sdk)


class _DummyCeleryApp:
    def task(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


dummy_celery_app_module = types.ModuleType("nyx.tasks.celery_app")
dummy_celery_app_module.app = _DummyCeleryApp()
sys.modules.setdefault("nyx.tasks", types.ModuleType("nyx.tasks"))
sys.modules.setdefault("nyx.tasks.celery_app", dummy_celery_app_module)


os.environ.setdefault("OPENAI_API_KEY", "test-key")

import tasks


class DummyTask:
    def __init__(self):
        self.retry = Mock(side_effect=AssertionError("retry should not be called"))


@pytest.mark.unit
def test_update_scene_conflict_context_caches_result(monkeypatch):
    fake_client = Mock()
    monkeypatch.setattr(tasks, "get_redis_client", Mock(return_value=fake_client))
    monkeypatch.setattr(tasks.asyncio, "run", Mock(return_value={"conflicts": ["c1"]}))

    cache_key = "conflict:scene:123"

    dummy_task = DummyTask()
    monkeypatch.setattr(tasks.update_scene_conflict_context, "retry", dummy_task.retry)

    tasks.update_scene_conflict_context.run(
        1,
        42,
        {"scene": "info"},
        cache_key,
    )

    fake_client.set.assert_called_once()
    args, kwargs = fake_client.set.call_args
    assert args[0] == cache_key
    assert "ex" in kwargs and kwargs["ex"] == 600
