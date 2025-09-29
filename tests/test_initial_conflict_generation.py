import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


class DummyConnection:
    def __init__(self, fetchval_result=None):
        self.fetchval_result = fetchval_result
        self.execute_calls = []

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))

    async def fetchval(self, *args, **kwargs):
        return self.fetchval_result


class DummyContextManager:
    def __init__(self, connection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyRunContext:
    def __init__(self, context):
        self.context = context
        self.user_id = None
        self.conversation_id = None


class DummyConflictIntegration:
    def __init__(self, response):
        self._response = response

    async def initialize(self):
        return None

    async def generate_conflict(self, *args, **kwargs):
        return self._response

    @classmethod
    async def get_instance(cls, *args, **kwargs):
        return cls(
            {
                "success": True,
                "raw_result": {"conflict_name": "Raw Result Title"},
                "conflict_details": {"name": "Detailed Title"},
            }
        )


@pytest.fixture
def tasks_module(monkeypatch):
    # Stub modules that perform heavy initialization
    stub_sentence_transformers = types.ModuleType("sentence_transformers")

    stub_agents = types.ModuleType("agents")
    stub_agents.trace = lambda *args, **kwargs: (lambda func: func)
    stub_agents.custom_span = lambda *args, **kwargs: (lambda func: func)
    stub_agents.RunContextWrapper = DummyRunContext
    stub_agents.tracing = types.SimpleNamespace(get_current_trace=lambda: None)
    monkeypatch.setitem(sys.modules, "agents", stub_agents)
    monkeypatch.setitem(sys.modules, "agents.tracing", stub_agents.tracing)

    stub_chatgpt = types.SimpleNamespace(
        get_chatgpt_response=lambda *args, **kwargs: {},
        get_openai_client=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "logic.chatgpt_integration", stub_chatgpt)

    stub_new_game = types.SimpleNamespace(NewGameAgent=object)
    monkeypatch.setitem(sys.modules, "new_game_agent", stub_new_game)

    stub_npc_learning = types.SimpleNamespace(NPCLearningManager=object)
    monkeypatch.setitem(
        sys.modules, "npcs.npc_learning_adaptation", stub_npc_learning
    )

    stub_memory_integration = types.SimpleNamespace(
        run_maintenance_through_nyx=lambda *args, **kwargs: None
    )
    monkeypatch.setitem(
        sys.modules, "memory.memory_nyx_integration", stub_memory_integration
    )

    stub_nyx_brain = types.SimpleNamespace(NyxBrain=object)
    monkeypatch.setitem(sys.modules, "nyx.core.brain.base", stub_nyx_brain)

    stub_checkpoint_agent = types.SimpleNamespace(CheckpointingPlannerAgent=object)
    monkeypatch.setitem(
        sys.modules,
        "nyx.core.brain.checkpointing_agent",
        stub_checkpoint_agent,
    )

    class _DummySDK:
        def __init__(self, *args, **kwargs):
            pass

        async def initialize_agent(self):
            return None

    class _DummySDKConfig:
        def __init__(self, *args, **kwargs):
            pass

    stub_nyx_sdk = types.SimpleNamespace(NyxAgentSDK=_DummySDK, NyxSDKConfig=_DummySDKConfig)
    monkeypatch.setitem(sys.modules, "nyx.nyx_agent_sdk", stub_nyx_sdk)

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            self._dim = 1536

        def encode(self, *args, **kwargs):
            return []

        def get_sentence_embedding_dimension(self):
            return self._dim

    class _DummyTransformer:
        def __init__(self, *args, **kwargs):
            self._dim = 1536

        def get_word_embedding_dimension(self):
            return self._dim

    class _DummyPooling:
        def __init__(self, *args, **kwargs):
            pass

    stub_sentence_transformers.SentenceTransformer = _DummySentenceTransformer
    stub_sentence_transformers.models = types.SimpleNamespace(
        Transformer=_DummyTransformer,
        Pooling=_DummyPooling,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub_sentence_transformers)

    class _DummyFaissIndex:
        def __init__(self, *args, **kwargs):
            self.ntotal = 0

        def add_with_ids(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def search(self, *args, **kwargs):
            return [], []

        def remove_ids(self, *args, **kwargs):
            pass

    class _DummyFaissModule(types.ModuleType):
        def __init__(self):
            super().__init__("faiss")
            self.IndexIDMap = _DummyFaissIndex
            self.IndexFlatIP = _DummyFaissIndex

        def read_index(self, *args, **kwargs):
            return _DummyFaissIndex()

        def write_index(self, *args, **kwargs):
            return None

    stub_faiss = _DummyFaissModule()
    monkeypatch.setitem(sys.modules, "faiss", stub_faiss)
    monkeypatch.setitem(sys.modules, "faiss.swigfaiss", types.ModuleType("faiss.swigfaiss"))
    monkeypatch.setitem(sys.modules, "faiss.swigfaiss_avx2", types.ModuleType("faiss.swigfaiss_avx2"))
    monkeypatch.setitem(sys.modules, "faiss.swigfaiss_avx512", types.ModuleType("faiss.swigfaiss_avx512"))

    stub_conflict_module = types.SimpleNamespace(ConflictSystemIntegration=DummyConflictIntegration)
    monkeypatch.setitem(
        sys.modules,
        "logic.conflict_system.conflict_integration",
        stub_conflict_module,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    module = importlib.import_module("tasks")
    return module


def test_generate_initial_conflict_prefers_raw_conflict_name(tasks_module, monkeypatch):
    monkeypatch.setattr(tasks_module, "RunContextWrapper", DummyRunContext)

    def _run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(tasks_module, "run_async_in_worker_loop", _run)

    first_conn = DummyConnection()
    second_conn = DummyConnection(fetchval_result=3)
    third_conn = DummyConnection()
    connections = [first_conn, second_conn, third_conn]

    def fake_get_db_connection_context():
        if not connections:
            raise AssertionError("No more connections available")
        return DummyContextManager(connections.pop(0))

    monkeypatch.setattr(tasks_module, "get_db_connection_context", fake_get_db_connection_context)

    result = tasks_module.generate_initial_conflict_task(user_id=1, conversation_id=42)

    assert result["initial_conflict"] == "Raw Result Title"

    summary_calls = [
        call for call in third_conn.execute_calls if "InitialConflictSummary" in call[0]
    ]
    assert summary_calls, "Expected InitialConflictSummary write"
    _, args = summary_calls[0]
    assert args[-1] == "Raw Result Title"

    assert not connections


def test_generate_initial_conflict_falls_back_to_db_title(tasks_module, monkeypatch):
    monkeypatch.setattr(tasks_module, "RunContextWrapper", DummyRunContext)

    def _run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(tasks_module, "run_async_in_worker_loop", _run)

    fallback_response = {
        "success": True,
        "raw_result": {},
        "conflict_details": {},
        "conflict_id": 99,
    }

    class _FallbackConflictIntegration:
        def __init__(self, response):
            self._response = response

        async def initialize(self):
            return None

        async def generate_conflict(self, *args, **kwargs):
            return self._response

        @classmethod
        async def get_instance(cls, *args, **kwargs):
            return cls(fallback_response)

    monkeypatch.setattr(
        sys.modules["logic.conflict_system.conflict_integration"],
        "ConflictSystemIntegration",
        _FallbackConflictIntegration,
    )

    first_conn = DummyConnection()
    second_conn = DummyConnection(fetchval_result=3)
    third_conn = DummyConnection(fetchval_result="Database Canonical Title")
    fourth_conn = DummyConnection()
    connections = [first_conn, second_conn, third_conn, fourth_conn]

    def fake_get_db_connection_context():
        if not connections:
            raise AssertionError("No more connections available")
        return DummyContextManager(connections.pop(0))

    monkeypatch.setattr(tasks_module, "get_db_connection_context", fake_get_db_connection_context)

    result = tasks_module.generate_initial_conflict_task(user_id=1, conversation_id=42)

    assert result["initial_conflict"] == "Database Canonical Title"

    summary_calls = [
        call for call in fourth_conn.execute_calls if "InitialConflictSummary" in call[0]
    ]
    assert summary_calls, "Expected InitialConflictSummary write"
    _, args = summary_calls[0]
    assert args[-1] == "Database Canonical Title"

    assert not connections


def test_generate_initial_conflict_uses_db_name_when_conflict_names_missing(
    tasks_module, monkeypatch
):
    monkeypatch.setattr(tasks_module, "RunContextWrapper", DummyRunContext)

    def _run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(tasks_module, "run_async_in_worker_loop", _run)

    fallback_response = {
        "success": True,
        "raw_result": {},
        "conflict_details": {},
        "conflict_id": 123,
    }

    class _FallbackConflictIntegration:
        def __init__(self, response):
            self._response = response

        async def initialize(self):
            return None

        async def generate_conflict(self, *args, **kwargs):
            return self._response

        @classmethod
        async def get_instance(cls, *args, **kwargs):
            return cls(fallback_response)

    monkeypatch.setattr(
        sys.modules["logic.conflict_system.conflict_integration"],
        "ConflictSystemIntegration",
        _FallbackConflictIntegration,
    )

    first_conn = DummyConnection()
    second_conn = DummyConnection(fetchval_result=3)
    third_conn = DummyConnection(fetchval_result="   Stored   Conflict   Name   ")
    fourth_conn = DummyConnection()
    connections = [first_conn, second_conn, third_conn, fourth_conn]

    def fake_get_db_connection_context():
        if not connections:
            raise AssertionError("No more connections available")
        return DummyContextManager(connections.pop(0))

    monkeypatch.setattr(
        tasks_module, "get_db_connection_context", fake_get_db_connection_context
    )

    result = tasks_module.generate_initial_conflict_task(user_id=1, conversation_id=77)

    assert result["initial_conflict"] == "Stored Conflict Name"

    summary_calls = [
        call for call in fourth_conn.execute_calls if "InitialConflictSummary" in call[0]
    ]
    assert summary_calls, "Expected InitialConflictSummary write"
    _, args = summary_calls[0]
    assert args[-1] == "Stored Conflict Name"

    assert not connections


def test_generate_initial_conflict_resolves_template_subsystem_conflict_id(
    tasks_module, monkeypatch
):
    monkeypatch.setattr(tasks_module, "RunContextWrapper", DummyRunContext)

    def _run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(tasks_module, "run_async_in_worker_loop", _run)

    template_response = {
        "success": True,
        "raw_result": {
            "status": "created",
            "conflict_type": "major",
            "conflict_id": 555,
            "subsystem_responses": {
                "template": {
                    "template_used": 42,
                    "generated_conflict": 555,
                    "conflict_id": 555,
                    "narrative_hooks": ["Hook one", "Hook two"],
                }
            },
        },
        "conflict_details": {},
        "conflict_id": 555,
    }

    class _TemplateConflictIntegration:
        def __init__(self, response):
            self._response = response

        async def initialize(self):
            return None

        async def generate_conflict(self, *args, **kwargs):
            return self._response

        @classmethod
        async def get_instance(cls, *args, **kwargs):
            return cls(template_response)

    monkeypatch.setattr(
        sys.modules["logic.conflict_system.conflict_integration"],
        "ConflictSystemIntegration",
        _TemplateConflictIntegration,
    )

    first_conn = DummyConnection()
    second_conn = DummyConnection(fetchval_result=3)
    third_conn = DummyConnection(fetchval_result="Template Stored Name")
    fourth_conn = DummyConnection()
    connections = [first_conn, second_conn, third_conn, fourth_conn]

    def fake_get_db_connection_context():
        if not connections:
            raise AssertionError("No more connections available")
        return DummyContextManager(connections.pop(0))

    monkeypatch.setattr(
        tasks_module, "get_db_connection_context", fake_get_db_connection_context
    )

    result = tasks_module.generate_initial_conflict_task(user_id=5, conversation_id=99)

    assert result["initial_conflict"] == "Template Stored Name"

    summary_calls = [
        call for call in fourth_conn.execute_calls if "InitialConflictSummary" in call[0]
    ]
    assert summary_calls, "Expected InitialConflictSummary write"
    _, args = summary_calls[0]
    assert args[-1] == "Template Stored Name"

    assert not connections
