import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_STUBBED_MODULES = [
    "openai_integration.conversations",
    "nyx.integrate",
    "nyx.nyx_governance",
    "lore.core.lore_system",
    "lore.core.canon",
    "lore.core",
    "agents",
]

_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
for module_name in sorted(_STUBBED_MODULES, key=len, reverse=True):
    sys.modules.pop(module_name, None)

sys.modules["lore.core"] = types.ModuleType("lore.core")
stub_canon_module = types.ModuleType("lore.core.canon")
sys.modules["lore.core.canon"] = stub_canon_module
setattr(sys.modules["lore.core"], "canon", stub_canon_module)

stub_lore_module = types.ModuleType("lore.core.lore_system")


class _StubLoreSystem:
    @classmethod
    async def get_instance(cls, user_id, conversation_id):
        return object()


setattr(stub_lore_module, "LoreSystem", _StubLoreSystem)
sys.modules["lore.core.lore_system"] = stub_lore_module

agents_stub = types.ModuleType("agents")


class _StubAgent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __class_getitem__(cls, item):
        return cls


class _StubAgentOutputSchema:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubModelSettings:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubRunner:
    @staticmethod
    async def run(*args, **kwargs):
        raise NotImplementedError


def _stub_function_tool(func):
    return func


class _StubRunContextWrapperModule:
    def __init__(self, context):
        self.context = context


class _StubGuardrailFunctionOutput:
    pass


class _StubInputGuardrail:
    def __init__(self, guardrail_function=None):
        self.guardrail_function = guardrail_function


@contextmanager
def _stub_trace(*args, **kwargs):
    yield


def _stub_handoff(*args, **kwargs):
    return None


agents_stub.Agent = _StubAgent
agents_stub.AgentOutputSchema = _StubAgentOutputSchema
agents_stub.ModelSettings = _StubModelSettings
agents_stub.Runner = _StubRunner()
agents_stub.function_tool = _stub_function_tool
agents_stub.RunContextWrapper = _StubRunContextWrapperModule
agents_stub.GuardrailFunctionOutput = _StubGuardrailFunctionOutput
agents_stub.InputGuardrail = _StubInputGuardrail
agents_stub.trace = _stub_trace
agents_stub.handoff = _stub_handoff
sys.modules["agents"] = agents_stub

nyx_governance_stub = types.ModuleType("nyx.nyx_governance")


class _StubNyxUnifiedGovernor:
    async def check_action_permission(self, **kwargs):
        return {"approved": True}

    async def process_agent_action_report(self, **kwargs):
        return None


nyx_governance_stub.NyxUnifiedGovernor = _StubNyxUnifiedGovernor
nyx_governance_stub.AgentType = types.SimpleNamespace(UNIVERSAL_UPDATER="universal")
nyx_governance_stub.DirectiveType = types.SimpleNamespace(ACTION="action")
nyx_governance_stub.DirectivePriority = types.SimpleNamespace(MEDIUM="medium")
sys.modules["nyx.nyx_governance"] = nyx_governance_stub

nyx_integrate_stub = types.ModuleType("nyx.integrate")


async def _stub_get_central_governance(user_id, conversation_id):
    return _StubNyxUnifiedGovernor()


nyx_integrate_stub.get_central_governance = _stub_get_central_governance
sys.modules["nyx.integrate"] = nyx_integrate_stub

conversations_stub = types.ModuleType("openai_integration.conversations")


async def _stub_ensure_scene_seal_item(*args, **kwargs):
    return None


def _stub_extract_scene_seal_from_updates(payload):
    return {}


conversations_stub.ensure_scene_seal_item = _stub_ensure_scene_seal_item
conversations_stub.extract_scene_seal_from_updates = _stub_extract_scene_seal_from_updates
sys.modules["openai_integration.conversations"] = conversations_stub

from logic import universal_updater_agent as updater


class DummyGovernor:
    async def check_action_permission(self, **kwargs):
        return {"approved": True}

    async def process_agent_action_report(self, **kwargs):
        return None


@contextmanager
def _noop_trace(*args, **kwargs):
    yield


class _StubRunContextWrapper:
    def __init__(self, context):
        self.context = context


# Use pytest-anyio to avoid requiring pytest-asyncio in the test environment.
@pytest.mark.anyio("asyncio")
async def test_process_universal_update_handles_missing_narrative(monkeypatch):
    dummy_governor = DummyGovernor()

    async def fake_get_central_governance(user_id, conversation_id):
        return dummy_governor

    async def fake_lore_get_instance(cls, user_id, conversation_id):
        return object()

    async def fake_runner_run(agent, prompt, context):
        class DummyResult:
            def final_output_as(self, schema):
                # Deliberately omit the narrative field
                return schema(user_id=1, conversation_id=2)

        return DummyResult()

    captured = {}

    async def fake_apply_universal_updates(ctx, update_json):
        captured["payload"] = json.loads(update_json)
        return updater.ApplyUpdatesResult(success=True, updates_applied=0)

    monkeypatch.setattr(updater, "get_central_governance", fake_get_central_governance)
    monkeypatch.setattr(updater.LoreSystem, "get_instance", classmethod(fake_lore_get_instance))
    monkeypatch.setattr(updater.Runner, "run", staticmethod(fake_runner_run))
    monkeypatch.setattr(updater, "trace", _noop_trace)
    monkeypatch.setattr(updater, "RunContextWrapper", _StubRunContextWrapper)
    monkeypatch.setattr(updater, "_apply_universal_updates_impl", fake_apply_universal_updates)

    result = await updater.process_universal_update(1, 2, "", context=None)

    assert result["success"] is True
    assert captured["payload"]["narrative"] == updater.DEFAULT_NARRATIVE_FALLBACK
@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
def _restore_stubbed_modules():
    yield
    for name, module in _ORIGINAL_MODULES.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

