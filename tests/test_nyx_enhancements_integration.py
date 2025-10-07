import asyncio
import json
import os
import sys
import types
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import typing_extensions

typing.TypedDict = typing_extensions.TypedDict

stub_sentence_transformers = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def encode(self, texts, **kwargs):
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 3


def _noop_model_factory(*_: Any, **__: Any) -> None:
    return None


def _noop_pooling_factory(*_: Any, **__: Any) -> None:
    return None


stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
stub_sentence_transformers.models = types.ModuleType("sentence_transformers.models")
stub_sentence_transformers.models.Transformer = _noop_model_factory
stub_sentence_transformers.models.Pooling = _noop_pooling_factory

sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)
sys.modules.setdefault("sentence_transformers.models", stub_sentence_transformers.models)
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

nyx_module = types.ModuleType("nyx")
nyx_agent_sdk_module = types.ModuleType("nyx.nyx_agent_sdk")


class _StubNyxResponse:
    def __init__(self, narrative: str = "", metadata: Dict[str, Any] | None = None) -> None:
        self.narrative = narrative
        self.metadata = metadata or {}
        self.image = None
        self.world_state = None
        self.choices = None
        self.error = None
        self.trace_id = "stub-trace"
        self.telemetry = {}


class _StubNyxAgentSDK:
    async def warmup_cache(self, *_: Any, **__: Any) -> None:
        return None

    async def process_user_input(self, *_: Any, **__: Any) -> _StubNyxResponse:
        return _StubNyxResponse()


class _StubNyxConfig:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass


async def _stub_process_user_input(*_: Any, **__: Any) -> _StubNyxResponse:
    return _StubNyxResponse()


nyx_agent_sdk_module.NyxAgentSDK = _StubNyxAgentSDK
nyx_agent_sdk_module.NyxSDKConfig = _StubNyxConfig
nyx_agent_sdk_module.NyxResponse = _StubNyxResponse
nyx_agent_sdk_module.process_user_input = _stub_process_user_input

nyx_context_module = types.ModuleType("nyx.nyx_agent.context")


class _StubNyxContext:
    @staticmethod
    def _normalize_location_value(value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                value = value.decode("utf-8", errors="ignore")

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                parsed = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                parsed = candidate

            if isinstance(parsed, str):
                parsed = parsed.strip()
                return parsed or None
            if isinstance(parsed, dict):
                for key in ("name", "location", "location_name", "id", "scene_id"):
                    token = parsed.get(key)
                    if isinstance(token, str) and token.strip():
                        return token.strip()
                    if isinstance(token, (int, float)):
                        token_str = str(token).strip()
                        if token_str:
                            return token_str
                return None
            if isinstance(parsed, (int, float)):
                token = str(parsed).strip()
                return token or None
            return None

        if isinstance(value, (int, float)):
            token = str(value).strip()
            return token or None

        if isinstance(value, dict):
            for key in ("name", "location", "location_name", "id", "scene_id"):
                token = value.get(key)
                if isinstance(token, str) and token.strip():
                    return token.strip()
                if isinstance(token, (int, float)):
                    token_str = str(token).strip()
                    if token_str:
                        return token_str

        return None


nyx_context_module.NyxContext = _StubNyxContext
nyx_context_module.ContextBroker = object
nyx_context_module.SceneScope = object

nyx_orchestrator_module = types.ModuleType("nyx.nyx_agent.orchestrator")


async def _stub_async_return(*_: Any, **__: Any) -> Dict[str, Any]:
    return {}


async def _stub_store_messages(*_: Any, **__: Any) -> None:
    return None


nyx_orchestrator_module.generate_reflection = _stub_async_return
nyx_orchestrator_module.manage_relationships = _stub_async_return
nyx_orchestrator_module.manage_scenario = _stub_async_return
nyx_orchestrator_module.store_messages = _stub_store_messages

nyx_agent_pkg = types.ModuleType("nyx.nyx_agent")
nyx_agent_pkg.context = nyx_context_module
nyx_agent_pkg.orchestrator = nyx_orchestrator_module

memory_orchestrator_module = types.ModuleType("memory.memory_orchestrator")


class _StubMemoryOrchestrator:
    async def consolidate_memories(self, *args: Any, **kwargs: Any) -> None:
        return None


class _StubEntityType:
    PLAYER = "player"


async def _stub_get_memory_orchestrator(*_: Any, **__: Any) -> _StubMemoryOrchestrator:
    return _StubMemoryOrchestrator()


memory_orchestrator_module.MemoryOrchestrator = _StubMemoryOrchestrator
memory_orchestrator_module.EntityType = _StubEntityType
memory_orchestrator_module.get_memory_orchestrator = _stub_get_memory_orchestrator

conflict_synth_module = types.ModuleType("logic.conflict_system.conflict_synthesizer")


class _StubConflictSynthesizer:
    async def calculate_tensions(self) -> list[Any]:
        return []


def _stub_get_synthesizer(*_: Any, **__: Any) -> _StubConflictSynthesizer:
    return _StubConflictSynthesizer()


conflict_synth_module.get_synthesizer = _stub_get_synthesizer
conflict_synth_module.ConflictSynthesizer = _StubConflictSynthesizer

conflict_background_module = types.ModuleType("logic.conflict_system.background_processor")
conflict_background_module.get_conflict_scheduler = lambda *_, **__: None

nyx_user_model_module = types.ModuleType("nyx.user_model_sdk")
nyx_user_model_module.UserModelManager = object

npc_orchestrator_module = types.ModuleType("npcs.npc_orchestrator")
npc_orchestrator_module.NPCOrchestrator = object

nyx_module.nyx_agent_sdk = nyx_agent_sdk_module
nyx_module.nyx_agent = nyx_agent_pkg
nyx_module.user_model_sdk = nyx_user_model_module

sys.modules.setdefault("nyx", nyx_module)
sys.modules.setdefault("nyx.nyx_agent_sdk", nyx_agent_sdk_module)
sys.modules.setdefault("nyx.nyx_agent", nyx_agent_pkg)
sys.modules.setdefault("nyx.nyx_agent.context", nyx_context_module)
sys.modules.setdefault("nyx.nyx_agent.orchestrator", nyx_orchestrator_module)
sys.modules.setdefault("nyx.user_model_sdk", nyx_user_model_module)
sys.modules.setdefault("memory.memory_orchestrator", memory_orchestrator_module)
sys.modules.setdefault("logic.conflict_system.conflict_synthesizer", conflict_synth_module)
sys.modules.setdefault("logic.conflict_system.background_processor", conflict_background_module)
sys.modules.setdefault("npcs.npc_orchestrator", npc_orchestrator_module)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from logic import nyx_enhancements_integration as integration


class _StubConnection:
    def __init__(self, *, location_payload: Any, npc_payload: str | None = None) -> None:
        self._location_payload = location_payload
        self._npc_payload = npc_payload

    async def fetchrow(self, query: str, *args: Any) -> Dict[str, Any] | None:
        if "CurrentRoleplay" not in query:
            raise AssertionError(f"Unexpected query: {query}")

        if "'CurrentLocation'" in query:
            return {"value": self._location_payload}

        if "'introduced_npcs'" in query and self._npc_payload is not None:
            return {"value": self._npc_payload}

        return None


class _StubConnectionContext:
    def __init__(self, connection: _StubConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _StubConnection:
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _RecordingSDK:
    def __init__(self) -> None:
        self.received_metadata: Dict[str, Any] | None = None

    async def warmup_cache(self, *_: Any, **__: Any) -> None:
        return None

    async def process_user_input(self, **kwargs: Any):
        self.received_metadata = kwargs.get("metadata")
        return SimpleNamespace(
            narrative="", metadata={}, image=None, world_state=None, choices=None
        )


def test_enhanced_background_chat_task_normalizes_current_location(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_location = json.dumps({"name": "Velvet Sanctum", "id": "scene-019"})
    stub_connection = _StubConnection(
        location_payload=raw_location, npc_payload=json.dumps([{"npc_id": "Ava"}])
    )

    monkeypatch.setattr(
        integration,
        "get_db_connection_context",
        lambda: _StubConnectionContext(stub_connection),
    )

    async def _recent_turns_stub(*_: Any, **__: Any) -> list[Dict[str, Any]]:
        return []

    monkeypatch.setattr(integration, "fetch_recent_turns", _recent_turns_stub)

    async def _noop(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(integration, "store_messages", _noop)
    monkeypatch.setattr(integration, "store_performance_metrics", _noop)
    monkeypatch.setattr(integration, "handle_image_generation_through_sdk", _noop)
    monkeypatch.setattr(integration, "update_narrative_arcs_from_sdk_response", _noop)

    socket_stub = SimpleNamespace(emit=_noop)
    monkeypatch.setattr(integration, "current_app", SimpleNamespace(socketio=socket_stub))

    recording_sdk = _RecordingSDK()
    monkeypatch.setattr(integration, "get_sdk", lambda: recording_sdk)

    asyncio.run(
        integration.enhanced_background_chat_task(
            conversation_id=17,
            user_input="Describe the surroundings.",
            user_id=23,
        )
    )

    assert recording_sdk.received_metadata is not None
    assert recording_sdk.received_metadata.get("location") == "Velvet Sanctum"
