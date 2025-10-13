import asyncio
import os
import sys
import types
import importlib
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

stub_sentence_transformers = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:  # pragma: no cover - compatibility shim
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def encode(self, *_: object, **__: object) -> list[object]:
        return []


stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nyx_module = importlib.import_module("nyx")
nyx_agent_root = Path(__file__).resolve().parents[1] / "nyx" / "nyx_agent"

nyx_agent_pkg = types.ModuleType("nyx.nyx_agent")
nyx_agent_pkg.__path__ = [str(nyx_agent_root)]
sys.modules["nyx.nyx_agent"] = nyx_agent_pkg
setattr(nyx_module, "nyx_agent", nyx_agent_pkg)

nyx_context_module = types.ModuleType("nyx.nyx_agent.context")


class _StubNyxContext:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    async def initialize(self) -> None:
        return None


class _StubSceneScope:
    def __init__(self, *_: object, **__: object) -> None:
        pass


nyx_context_module.NyxContext = _StubNyxContext
nyx_context_module.SceneScope = _StubSceneScope
nyx_context_module.build_canonical_snapshot_payload = lambda *_, **__: {}
nyx_context_module.persist_canonical_snapshot = lambda *_, **__: None
nyx_context_module.PackedContext = object
sys.modules["nyx.nyx_agent.context"] = nyx_context_module

nyx_agents_module = types.ModuleType("nyx.nyx_agent.agents")
nyx_agents_module.nyx_main_agent = object()
nyx_agents_module.nyx_defer_agent = object()
nyx_agents_module.reflection_agent = object()
nyx_agents_module.DEFAULT_MODEL_SETTINGS = {}
sys.modules["nyx.nyx_agent.agents"] = nyx_agents_module

nyx_assembly_module = types.ModuleType("nyx.nyx_agent.assembly")


async def _stub_async(*_: object, **__: object) -> dict:
    return {}


nyx_assembly_module.assemble_nyx_response = _stub_async
nyx_assembly_module.resolve_scene_requests = _stub_async
sys.modules["nyx.nyx_agent.assembly"] = nyx_assembly_module

nyx_tools_module = types.ModuleType("nyx.nyx_agent.tools")
nyx_tools_module.update_relationship_state = _stub_async
nyx_tools_module.generate_universal_updates_impl = _stub_async
sys.modules["nyx.nyx_agent.tools"] = nyx_tools_module

nyx_utils_module = types.ModuleType("nyx.nyx_agent.utils")
nyx_utils_module._did_call_tool = lambda *_, **__: False
nyx_utils_module._extract_last_assistant_text = lambda *_, **__: ""
nyx_utils_module._js = lambda value: value
nyx_utils_module.sanitize_agent_tools_in_place = lambda *_, **__: None
nyx_utils_module.log_strict_hits = lambda *_, **__: None
nyx_utils_module.extract_runner_response = lambda *_, **__: {}
sys.modules["nyx.nyx_agent.utils"] = nyx_utils_module

nyx_models_module = types.ModuleType("nyx.nyx_agent.models")


class _StubNyxResponse:
    def __init__(self, narrative: str = "", **_: object) -> None:
        self.narrative = narrative


nyx_models_module.NyxResponse = _StubNyxResponse
sys.modules["nyx.nyx_agent.models"] = nyx_models_module

nyx_feas_module = types.ModuleType("nyx.nyx_agent.feasibility")


async def _stub_feasibility(*_: object, **__: object) -> dict:
    return {}


nyx_feas_module.assess_action_feasibility = _stub_feasibility
nyx_feas_module.record_impossibility = lambda *_, **__: None
nyx_feas_module.record_possibility = lambda *_, **__: None
nyx_feas_module.detect_setting_type = lambda *_, **__: ""
nyx_feas_module.assess_action_feasibility_fast = _stub_feasibility
sys.modules["nyx.nyx_agent.feasibility"] = nyx_feas_module

snapshot_store_module = types.ModuleType("nyx.conversation.snapshot_store")


class _StubSnapshotStore:
    def __init__(self, *_: object, **__: object) -> None:
        pass


snapshot_store_module.ConversationSnapshotStore = _StubSnapshotStore
sys.modules["nyx.conversation.snapshot_store"] = snapshot_store_module

side_effects_module = types.ModuleType("nyx.core.side_effects")


class _StubSideEffect:
    pass


side_effects_module.SideEffect = _StubSideEffect
side_effects_module.ConflictEvent = _StubSideEffect
side_effects_module.LoreHint = _StubSideEffect
side_effects_module.MemoryEvent = _StubSideEffect
side_effects_module.NPCStimulus = _StubSideEffect
side_effects_module.WorldDelta = _StubSideEffect
side_effects_module.group_side_effects = lambda effects: {"effects": list(effects or [])}
sys.modules["nyx.core.side_effects"] = side_effects_module

post_turn_module = types.ModuleType("nyx.tasks.realtime.post_turn")


async def _stub_dispatch(*_: object, **__: object) -> None:
    return None


post_turn_module.dispatch = _stub_dispatch
sys.modules.setdefault("nyx.tasks", types.ModuleType("nyx.tasks"))
sys.modules.setdefault("nyx.tasks.realtime", types.ModuleType("nyx.tasks.realtime"))
sys.modules["nyx.tasks.realtime.post_turn"] = post_turn_module

agents_module = types.ModuleType("agents")


class _StubRunContextWrapper:
    def __init__(self, *_: object, **__: object) -> None:
        pass


class _StubRunner:
    async def run(self, *_: object, **__: object) -> dict:
        return {}


class _StubRunConfig:
    pass


class _StubModelSettings:
    pass


class _StubAgent:
    pass


agents_module.Runner = _StubRunner()
agents_module.RunConfig = _StubRunConfig
agents_module.ModelSettings = _StubModelSettings
agents_module.RunContextWrapper = _StubRunContextWrapper
agents_module.Agent = _StubAgent
sys.modules["agents"] = agents_module

response_filter_module = types.ModuleType("nyx.response_filter")


class _StubResponseFilter:
    def __init__(self, *_: object, **__: object) -> None:
        pass


response_filter_module.ResponseFilter = _StubResponseFilter
sys.modules["nyx.response_filter"] = response_filter_module

nyx_task_integration_module = types.ModuleType("nyx.nyx_task_integration")


class _StubNyxTaskIntegration:
    pass


nyx_task_integration_module.NyxTaskIntegration = _StubNyxTaskIntegration
sys.modules["nyx.nyx_task_integration"] = nyx_task_integration_module

user_model_module = types.ModuleType("nyx.user_model_sdk")


class _StubUserModelManager:
    async def initialize(self) -> None:
        return None


user_model_module.UserModelManager = _StubUserModelManager
sys.modules["nyx.user_model_sdk"] = user_model_module

aggregator_module = types.ModuleType("logic.aggregator_sdk")
aggregator_module.get_comprehensive_context = _stub_async
sys.modules.setdefault("logic", types.ModuleType("logic"))
sys.modules["logic.aggregator_sdk"] = aggregator_module

conflict_synth_module = types.ModuleType("logic.conflict_system.conflict_synthesizer")
conflict_synth_module.get_synthesizer = lambda *_, **__: None
sys.modules.setdefault("logic.conflict_system", types.ModuleType("logic.conflict_system"))
sys.modules["logic.conflict_system.conflict_synthesizer"] = conflict_synth_module

conflict_scheduler_module = types.ModuleType("logic.conflict_system.background_processor")
conflict_scheduler_module.get_conflict_scheduler = lambda *_, **__: None
sys.modules["logic.conflict_system.background_processor"] = conflict_scheduler_module

db_connection_module = types.ModuleType("db.connection")


@asynccontextmanager
async def _stub_db_connection(*_: object, **__: object):
    yield None


db_connection_module.get_db_connection_context = _stub_db_connection
sys.modules["db.connection"] = db_connection_module

_load_module("nyx.nyx_agent.config", nyx_agent_root / "config.py")
_load_module("nyx.nyx_agent._feasibility_helpers", nyx_agent_root / "_feasibility_helpers.py")
_load_module("nyx.nyx_agent.orchestrator", nyx_agent_root / "orchestrator.py")
sdk_module = _load_module(
    "nyx.nyx_agent_sdk", Path(__file__).resolve().parents[1] / "nyx" / "nyx_agent_sdk.py"
)


def test_defer_narrative_respects_configured_timeout(monkeypatch):
    """Calls just beyond the legacy timeout should still succeed with the new ceiling."""

    async def _run() -> None:
        sdk = sdk_module.NyxAgentSDK(
            sdk_module.NyxSDKConfig(request_timeout_seconds=0.1)
        )

        context = types.SimpleNamespace(
            narrator_guidance="Hold that fantasy until reality catches up.",
            leads=["gather supplies"],
            violations=[{"rule": "missing_item", "reason": "no supplies present"}],
            persona_prefix="Sweet thing,",
            reason_phrases=["you're empty-handed"],
        )

        async def fake_run(*_args, **_kwargs):
            await asyncio.sleep(0.06)
            return {"final_output": "Deferred guidance."}

        runner_stub = types.SimpleNamespace(run=fake_run)
        defer_agent = object()

        monkeypatch.setattr(sdk_module, "Runner", runner_stub, raising=False)
        monkeypatch.setattr(sdk_module, "nyx_defer_agent", defer_agent, raising=False)
        monkeypatch.setattr(
            sdk_module._orchestrator, "Runner", runner_stub, raising=False
        )
        monkeypatch.setattr(
            sdk_module._orchestrator, "nyx_defer_agent", defer_agent, raising=False
        )
        monkeypatch.setattr(
            sdk_module._orchestrator, "DEFER_RUN_TIMEOUT_SECONDS", 0.05, raising=False
        )

        loop = asyncio.get_running_loop()
        start = loop.time()
        result = await sdk._generate_defer_narrative(context, "test-trace")
        elapsed = loop.time() - start

        assert result == "Deferred guidance."
        assert elapsed >= 0.06
        assert elapsed < 0.5

    asyncio.run(_run())
