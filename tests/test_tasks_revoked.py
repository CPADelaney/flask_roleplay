import asyncio
from contextlib import asynccontextmanager, contextmanager
import pathlib
import sys
import os
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def encode(self, texts, **kwargs):
        return [[0.0] * self._dim for _ in texts]

    def get_sentence_embedding_dimension(self):
        return self._dim


class DummyTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def get_word_embedding_dimension(self):
        return self._dim


class DummyPooling:
    def __init__(self, dim, pooling_mode="mean"):
        self.dim = dim
        self.pooling_mode = pooling_mode


dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: DummyTransformer(*args, **kwargs)
dummy_models.Pooling = lambda *args, **kwargs: DummyPooling(*args, **kwargs)

dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models


dummy_agents = types.ModuleType("agents")


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


dummy_agents.trace = _trace
dummy_agents.custom_span = _custom_span
dummy_agents.RunContextWrapper = _RunContextWrapper

dummy_agents_tracing = types.ModuleType("agents.tracing")
dummy_agents_tracing.get_current_trace = lambda: None


logic_module = types.ModuleType("logic")
dummy_chatgpt = types.ModuleType("logic.chatgpt_integration")


async def _fake_chatgpt_response(*args, **kwargs):
    return {"response": "stub"}


class _FakeChat:
    class Completions:
        @staticmethod
        def create(*args, **kwargs):
            message = types.SimpleNamespace(content="stub response")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    completions = Completions()


class _FakeOpenAIClient:
    chat = _FakeChat()


def _fake_openai_client():
    return _FakeOpenAIClient()


dummy_chatgpt.get_chatgpt_response = _fake_chatgpt_response
dummy_chatgpt.get_openai_client = _fake_openai_client
logic_module.chatgpt_integration = dummy_chatgpt


dummy_new_game_agent = types.ModuleType("new_game_agent")


class _NewGameAgent:
    async def process_preset_game_direct(self, *args, **kwargs):
        return {}

    async def process_new_game(self, *args, **kwargs):
        return {}


dummy_new_game_agent.NewGameAgent = _NewGameAgent


dummy_npcs = types.ModuleType("npcs")
dummy_npcs_learning = types.ModuleType("npcs.npc_learning_adaptation")


class _NPCLearningManager:
    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        return None

    async def run_regular_adaptation_cycle(self, *args, **kwargs):
        return None


dummy_npcs_learning.NPCLearningManager = _NPCLearningManager
dummy_npcs.npc_learning_adaptation = dummy_npcs_learning


dummy_memory = types.ModuleType("memory")
dummy_memory_nyx = types.ModuleType("memory.memory_nyx_integration")


async def _run_maintenance_through_nyx(*args, **kwargs):
    return None


dummy_memory_nyx.run_maintenance_through_nyx = _run_maintenance_through_nyx
dummy_memory.memory_nyx_integration = dummy_memory_nyx


dummy_nyx = types.ModuleType("nyx")
dummy_nyx_core = types.ModuleType("nyx.core")
dummy_nyx_core_brain = types.ModuleType("nyx.core.brain")
dummy_nyx_core_brain_base = types.ModuleType("nyx.core.brain.base")
dummy_nyx_core_brain_checkpoint = types.ModuleType("nyx.core.brain.checkpointing_agent")


class _NyxBrain:
    @classmethod
    async def get_instance(cls, *args, **kwargs):
        return cls()

    async def ensure_initialized(self):
        return None


dummy_nyx_core_brain_base.NyxBrain = _NyxBrain


class _CheckpointingPlannerAgent:
    async def plan(self, *args, **kwargs):
        return None


dummy_nyx_core_brain_checkpoint.CheckpointingPlannerAgent = _CheckpointingPlannerAgent
dummy_nyx_core_brain.base = dummy_nyx_core_brain_base
dummy_nyx_core_brain.checkpointing_agent = dummy_nyx_core_brain_checkpoint
dummy_nyx_core.brain = dummy_nyx_core_brain
dummy_nyx.core = dummy_nyx_core


dummy_nyx_sdk = types.ModuleType("nyx.nyx_agent_sdk")


class _NyxSDKConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NyxAgentSDK:
    def __init__(self, config):
        self.config = config

    async def initialize_agent(self):
        return None


dummy_nyx_sdk.NyxSDKConfig = _NyxSDKConfig
dummy_nyx_sdk.NyxAgentSDK = _NyxAgentSDK
dummy_nyx.nyx_agent_sdk = dummy_nyx_sdk


sys.modules["agents"] = dummy_agents
sys.modules["agents.tracing"] = dummy_agents_tracing
sys.modules["logic"] = logic_module
sys.modules["logic.chatgpt_integration"] = dummy_chatgpt
sys.modules["new_game_agent"] = dummy_new_game_agent
sys.modules["npcs"] = dummy_npcs
sys.modules["npcs.npc_learning_adaptation"] = dummy_npcs_learning
sys.modules["memory"] = dummy_memory
sys.modules["memory.memory_nyx_integration"] = dummy_memory_nyx
sys.modules["nyx"] = dummy_nyx
sys.modules["nyx.core"] = dummy_nyx_core
sys.modules["nyx.core.brain"] = dummy_nyx_core_brain
sys.modules["nyx.core.brain.base"] = dummy_nyx_core_brain_base
sys.modules["nyx.core.brain.checkpointing_agent"] = dummy_nyx_core_brain_checkpoint
sys.modules["nyx.nyx_agent_sdk"] = dummy_nyx_sdk


import tasks


def _run_coroutine(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_process_new_game_task_revoked_marks_conversation_failed(monkeypatch):
    statuses = {"value": "processing"}
    executed = {}

    class DummyConnection:
        async def execute(self, query, *args):
            executed["query"] = query
            executed["args"] = args
            if "UPDATE conversations" in query:
                statuses["value"] = "failed"
            return None

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    run_calls = []

    monkeypatch.setattr(tasks, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(
        tasks,
        "run_async_in_worker_loop",
        lambda coro: run_calls.append(coro) or _run_coroutine(coro),
    )

    request = type(
        "DummyRequest",
        (),
        {
            "args": (42, {"conversation_id": 101}),
            "kwargs": {},
            "name": tasks.process_new_game_task.name,
            "task": tasks.process_new_game_task.name,
        },
    )()

    assert statuses["value"] == "processing"

    tasks._handle_process_new_game_task_revoked(
        sender=tasks.process_new_game_task,
        request=request,
        expired=True,
    )

    assert statuses["value"] == "failed"
    assert run_calls, "run_async_in_worker_loop should execute DB update"
    assert "UPDATE conversations" in executed["query"]
    assert executed["args"][0] == 101
    assert executed["args"][1] == 42
