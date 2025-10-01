import inspect
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from quart import Quart

sys.path.append(str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Provide lightweight stubs for heavy optional dependencies and Nyx packages
# that aren't required for these route tests.
stub_sentence_transformers = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:  # pragma: no cover - simple compatibility shim
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def encode(self, *_: Any, **__: Any) -> list[Any]:
        return []


stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

nyx_module = types.ModuleType("nyx")
nyx_agent_sdk_module = types.ModuleType("nyx.nyx_agent_sdk")


class _StubNyxResponse:
    def __init__(self, narrative: str = "", success: bool = True, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.narrative = narrative
        self.success = success
        self.metadata = metadata or {}
        self.image = None
        self.world_state = None
        self.choices = None
        self.error = None
        self.trace_id = "stub-trace"
        self.telemetry = {}


class _StubNyxAgentSDK:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def warmup_cache(self, *_: Any, **__: Any) -> None:
        return None

    async def process_user_input(self, *_: Any, **__: Any) -> _StubNyxResponse:
        return _StubNyxResponse()

    async def stream_user_input(self, *_: Any, **__: Any):  # pragma: no cover - unused helper
        yield {}

    async def cleanup_conversation(self, *_: Any, **__: Any) -> None:  # pragma: no cover - unused helper
        return None


class _StubNyxConfig:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass


nyx_agent_sdk_module.NyxAgentSDK = _StubNyxAgentSDK
nyx_agent_sdk_module.NyxSDKConfig = _StubNyxConfig
nyx_agent_sdk_module.NyxResponse = _StubNyxResponse

nyx_agent_pkg = types.ModuleType("nyx.nyx_agent")
nyx_orchestrator_module = types.ModuleType("nyx.nyx_agent.orchestrator")
nyx_context_module = types.ModuleType("nyx.nyx_agent.context")
nyx_models_module = types.ModuleType("nyx.nyx_agent.models")


async def _stub_async_return(*_: Any, **__: Any) -> Dict[str, Any]:
    return {}


nyx_orchestrator_module.generate_reflection = _stub_async_return
nyx_orchestrator_module.manage_relationships = _stub_async_return
nyx_orchestrator_module.manage_scenario = _stub_async_return


class _StubNyxContext:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.memory_orchestrator = None

    async def initialize(self) -> None:
        return None


nyx_context_module.NyxContext = _StubNyxContext


class _StubModel:
    pass


nyx_models_module.MemoryReflection = _StubModel
nyx_models_module.RelationshipUpdate = _StubModel
nyx_models_module.ScenarioDecision = _StubModel

nyx_scene_manager_module = types.ModuleType("nyx.scene_manager_sdk")
nyx_governance_module = types.ModuleType("nyx.nyx_governance")
nyx_governance_helpers_module = types.ModuleType("nyx.governance_helpers")
nyx_constants_module = types.ModuleType("nyx.constants")
nyx_directive_handler_module = types.ModuleType("nyx.directive_handler")
lore_module = types.ModuleType("lore")
lore_core_module = types.ModuleType("lore.core")
lore_canon_module = types.ModuleType("lore.core.canon")
lore_system_module = types.ModuleType("lore.core.lore_system")
memory_module = types.ModuleType("memory")
memory_orchestrator_module = types.ModuleType("memory.memory_orchestrator")
memory_integration_module = types.ModuleType("memory.memory_integration")
memory_service_module = types.ModuleType("memory.memory_service")
memory_nyx_module = types.ModuleType("memory.memory_nyx_integration")
memory_wrapper_module = types.ModuleType("memory.wrapper")


async def _stub_update_narrative_arcs_for_interaction(*_: Any, **__: Any) -> None:
    return None


nyx_scene_manager_module.update_narrative_arcs_for_interaction = _stub_update_narrative_arcs_for_interaction


class _StubAgentType:  # pragma: no cover - sentinel enum replacement
    PLAYER = "player"
    NPC = "npc"
    UNIVERSAL_UPDATER = "universal_updater"
    MEMORY = "memory"
    GOVERNANCE = "governance"


nyx_governance_module.AgentType = _StubAgentType


def _stub_with_governance(*_: Any, **__: Any):  # pragma: no cover - decorator shim
    def decorator(func):
        return func

    return decorator


nyx_governance_helpers_module.with_governance = _stub_with_governance
nyx_governance_helpers_module.with_governance_permission = _stub_with_governance
nyx_governance_helpers_module.with_action_reporting = _stub_with_governance
nyx_directive_handler_module.DirectiveHandler = object


class _StubDirectiveType:  # pragma: no cover - sentinel enums
    DIRECTIVE = "directive"


class _StubDirectivePriority:
    HIGH = "high"
    NORMAL = "normal"


nyx_constants_module.AgentType = _StubAgentType
nyx_constants_module.DirectiveType = _StubDirectiveType
nyx_constants_module.DirectivePriority = _StubDirectivePriority
nyx_governance_module.DirectiveType = _StubDirectiveType
nyx_governance_module.DirectivePriority = _StubDirectivePriority


def _stub_get_memory_orchestrator(*_: Any, **__: Any) -> None:
    return None


class _StubEntityType:  # pragma: no cover - sentinel enum replacement
    PLAYER = "player"


memory_orchestrator_module.get_memory_orchestrator = _stub_get_memory_orchestrator
memory_orchestrator_module.EntityType = _StubEntityType


def _stub_memory_service(*_: Any, **__: Any) -> None:
    return None


memory_integration_module.get_memory_service = _stub_memory_service
memory_nyx_module.MemoryNyxBridge = object
memory_service_module.MemoryEmbeddingService = object
memory_wrapper_module.MemorySystem = object
lore_system_module.LoreSystem = object
lore_core_module.canon = lore_canon_module
lore_module.core = lore_core_module

nyx_module.nyx_agent_sdk = nyx_agent_sdk_module
nyx_module.nyx_agent = nyx_agent_pkg
nyx_module.scene_manager_sdk = nyx_scene_manager_module
nyx_module.nyx_governance = nyx_governance_module
nyx_module.governance_helpers = nyx_governance_helpers_module
nyx_module.constants = nyx_constants_module
nyx_module.directive_handler = nyx_directive_handler_module
memory_module.memory_orchestrator = memory_orchestrator_module
memory_module.memory_integration = memory_integration_module
memory_module.memory_service = memory_service_module
memory_module.memory_nyx_integration = memory_nyx_module
memory_module.wrapper = memory_wrapper_module
lore_module.core = lore_core_module
lore_core_module.lore_system = lore_system_module
nyx_agent_pkg.orchestrator = nyx_orchestrator_module
nyx_agent_pkg.context = nyx_context_module
nyx_agent_pkg.models = nyx_models_module

sys.modules.setdefault("nyx", nyx_module)
sys.modules.setdefault("nyx.nyx_agent_sdk", nyx_agent_sdk_module)
sys.modules.setdefault("nyx.nyx_agent", nyx_agent_pkg)
sys.modules.setdefault("nyx.nyx_agent.orchestrator", nyx_orchestrator_module)
sys.modules.setdefault("nyx.nyx_agent.context", nyx_context_module)
sys.modules.setdefault("nyx.nyx_agent.models", nyx_models_module)
sys.modules.setdefault("nyx.scene_manager_sdk", nyx_scene_manager_module)
sys.modules.setdefault("nyx.nyx_governance", nyx_governance_module)
sys.modules.setdefault("nyx.governance_helpers", nyx_governance_helpers_module)
sys.modules.setdefault("nyx.constants", nyx_constants_module)
sys.modules.setdefault("nyx.directive_handler", nyx_directive_handler_module)
sys.modules.setdefault("lore", lore_module)
sys.modules.setdefault("lore.core", lore_core_module)
sys.modules.setdefault("lore.core.canon", lore_canon_module)
sys.modules.setdefault("lore.core.lore_system", lore_system_module)
sys.modules.setdefault("memory", memory_module)
sys.modules.setdefault("memory.memory_orchestrator", memory_orchestrator_module)
sys.modules.setdefault("memory.memory_integration", memory_integration_module)
sys.modules.setdefault("memory.memory_service", memory_service_module)
sys.modules.setdefault("memory.memory_nyx_integration", memory_nyx_module)
sys.modules.setdefault("memory.wrapper", memory_wrapper_module)

from routes import nyx_agent_routes_sdk as nyx_routes
from routes.nyx_agent_routes_sdk import nyx_agent_bp


@dataclass
class _DummySDKResponse:
    narrative: str = "Generated narrative"
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    image: Optional[Dict[str, Any]] = None
    world_state: Optional[Dict[str, Any]] = None
    choices: Optional[Any] = None
    error: Optional[str] = None
    trace_id: Optional[str] = "trace-123"
    telemetry: Dict[str, Any] = field(default_factory=lambda: {"latency_ms": 12})


class _DummySDK:
    def __init__(self, response: _DummySDKResponse) -> None:
        self._response = response

    async def warmup_cache(self, *_: Any, **__: Any) -> None:
        return None

    async def process_user_input(self, *_: Any, **__: Any) -> _DummySDKResponse:
        return self._response


@pytest.mark.asyncio
async def test_nyx_response_rule_enforcement_returns_serializable_result(monkeypatch: pytest.MonkeyPatch) -> None:
    app = Quart(__name__)
    app.secret_key = "testing-secret"
    app.register_blueprint(nyx_agent_bp)

    expected_rule_result = {
        "tier": "soft",
        "triggered": [
            {"condition": "obedience > 50", "effect": "Minor reprimand", "outcome": {"message": "warn"}}
        ],
        "telemetry": {"violations": 1, "severity": 2},
    }

    dummy_sdk = _DummySDK(_DummySDKResponse())

    monkeypatch.setattr(nyx_routes, "get_sdk", lambda: dummy_sdk)

    async def fake_time_advancement(*_: Any, **__: Any) -> Dict[str, Any]:
        return {"time_advanced": False, "would_advance": False, "periods": 0, "confirm_needed": False}

    async def fake_relationship_events(*_: Any, **__: Any) -> Dict[str, Any]:
        return {"events": [], "interaction_results": None}

    async def fake_rule_enforcement(*_: Any, **__: Any) -> Dict[str, Any]:
        return expected_rule_result

    async def fake_addiction_effects(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(nyx_routes, "process_time_advancement", fake_time_advancement)
    monkeypatch.setattr(nyx_routes, "process_relationship_events", fake_relationship_events)
    monkeypatch.setattr(nyx_routes, "enforce_all_rules_on_player", fake_rule_enforcement)
    monkeypatch.setattr(nyx_routes, "check_addiction_effects", fake_addiction_effects)

    test_client = app.test_client()

    async with test_client.session_transaction() as test_session:
        test_session["user_id"] = 7

    payload = {
        "user_input": "Hello Nyx",
        "conversation_id": 21,
        "player_name": "Chase",
    }

    response = await test_client.post("/nyx_response", json=payload)
    assert response.status_code == 200

    data = await response.get_json()
    assert isinstance(data, dict)
    assert data["rule_effects"] == expected_rule_result
    assert not inspect.iscoroutine(data["rule_effects"])
    assert data["rule_effects"]["tier"] == "soft"
