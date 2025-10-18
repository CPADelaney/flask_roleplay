import os
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class _StubConnection:
    def __init__(self):
        self.setting_rows = [
            {"key": "SettingType", "value": "modern_realistic"},
            {"key": "CurrentLocation", "value": "Shared Plaza"},
        ]
        self.location_rows = [
            {
                "id": 1,
                "user_id": 7,
                "conversation_id": 888,
                "location_name": "Shared Plaza",
            },
            {
                "id": 2,
                "user_id": 42,
                "conversation_id": 888,
                "location_name": "Shared Plaza",
            },
        ]
        self.last_location_args = None

    async def fetch(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT key, value FROM CurrentRoleplay"):
            return list(self.setting_rows)
        if normalized.startswith("SELECT rule_name, condition, effect FROM GameRules"):
            return []
        if normalized.startswith("SELECT item_name, equipped FROM PlayerInventory"):
            return []
        if normalized.startswith("SELECT location_name FROM Locations"):
            return [
                {"location_name": row["location_name"]}
                for row in self.location_rows
            ]
        if normalized.startswith("SELECT npc_name FROM NPCStats"):
            return []
        if normalized.startswith("SELECT content FROM messages"):
            return []
        raise AssertionError(f"Unexpected fetch query: {normalized}")

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT * FROM Locations"):
            self.last_location_args = args
            user_id, conversation_id, location_name = args
            for row in self.location_rows:
                if (
                    row["user_id"] == user_id
                    and row["conversation_id"] == conversation_id
                    and row["location_name"] == location_name
                ):
                    return row
            return None
        if normalized.startswith("SELECT * FROM PlayerStats"):
            return {"user_id": args[0], "conversation_id": args[1]}
        if normalized.startswith("SELECT value FROM CurrentRoleplay"):
            return None
        raise AssertionError(f"Unexpected fetchrow query: {normalized}")

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT value FROM CurrentRoleplay"):
            return None
        raise AssertionError(f"Unexpected fetchval query: {normalized}")


class _StubConnectionContext:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_load_context_filters_locations_by_user(monkeypatch):
    import importlib.util
    import types
    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np

    class _StubTransformer:
        def __init__(self, *args, **kwargs):
            self._dim = 1

        def get_word_embedding_dimension(self):
            return self._dim

    class _StubPooling:
        def __init__(self, *args, **kwargs):
            pass

    class _StubSentenceTransformer:
        def __init__(self, modules=None, *args, **kwargs):
            self._dim = 1
            if modules:
                for module in modules:
                    getter = getattr(module, "get_word_embedding_dimension", None)
                    if getter:
                        self._dim = getter()

        def encode(self, texts, **_kwargs):
            return np.zeros((len(texts), self._dim), dtype=float)

        def get_sentence_embedding_dimension(self):
            return self._dim

    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = _StubSentenceTransformer
    sentence_transformers_stub.models = types.SimpleNamespace(
        Transformer=_StubTransformer,
        Pooling=_StubPooling,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_stub)

    agents_stub = types.ModuleType("agents")

    class _StubAgent:
        def __init__(self, *_args, **_kwargs):
            pass

    class _StubRunner:
        @staticmethod
        async def run(*_args, **_kwargs):
            return types.SimpleNamespace(final_output="{}")

    agents_stub.Agent = _StubAgent
    agents_stub.Runner = _StubRunner
    monkeypatch.setitem(sys.modules, "agents", agents_stub)

    logic_pkg = types.ModuleType("logic")
    logic_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "logic", logic_pkg)

    action_parser_stub = types.ModuleType("logic.action_parser")
    action_parser_stub.parse_action_intents = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "logic.action_parser", action_parser_stub)
    logic_pkg.action_parser = action_parser_stub

    aggregator_stub = types.ModuleType("logic.aggregator_sdk")
    aggregator_stub.fallback_get_context = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "logic.aggregator_sdk", aggregator_stub)
    logic_pkg.aggregator_sdk = aggregator_stub

    nyx_pkg = types.ModuleType("nyx")
    nyx_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "nyx", nyx_pkg)

    nyx_agent_pkg = types.ModuleType("nyx.nyx_agent")
    nyx_agent_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "nyx.nyx_agent", nyx_agent_pkg)
    nyx_pkg.nyx_agent = nyx_agent_pkg

    @dataclass
    class NyxContext:
        user_id: int
        conversation_id: int

    context_stub = types.ModuleType("nyx.nyx_agent.context")
    context_stub.NyxContext = NyxContext
    monkeypatch.setitem(sys.modules, "nyx.nyx_agent.context", context_stub)
    nyx_agent_pkg.context = context_stub

    nyx_feas_pkg = types.ModuleType("nyx.feas")
    nyx_feas_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "nyx.feas", nyx_feas_pkg)
    nyx_pkg.feas = nyx_feas_pkg

    feas_actions_pkg = types.ModuleType("nyx.feas.actions")
    feas_actions_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "nyx.feas.actions", feas_actions_pkg)
    nyx_feas_pkg.actions = feas_actions_pkg

    mundane_stub = types.ModuleType("nyx.feas.actions.mundane")
    mundane_stub.evaluate_mundane = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "nyx.feas.actions.mundane", mundane_stub)
    feas_actions_pkg.mundane = mundane_stub

    feas_archetypes_pkg = types.ModuleType("nyx.feas.archetypes")
    feas_archetypes_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "nyx.feas.archetypes", feas_archetypes_pkg)
    nyx_feas_pkg.archetypes = feas_archetypes_pkg

    modern_stub = types.ModuleType("nyx.feas.archetypes.modern_baseline")
    modern_stub.ModernBaseline = type("ModernBaseline", (), {})
    monkeypatch.setitem(sys.modules, "nyx.feas.archetypes.modern_baseline", modern_stub)
    feas_archetypes_pkg.modern_baseline = modern_stub

    roman_stub = types.ModuleType("nyx.feas.archetypes.roman_empire")
    roman_stub.RomanEmpire = type("RomanEmpire", (), {})
    monkeypatch.setitem(sys.modules, "nyx.feas.archetypes.roman_empire", roman_stub)
    feas_archetypes_pkg.roman_empire = roman_stub

    underwater_stub = types.ModuleType("nyx.feas.archetypes.underwater_scifi")
    underwater_stub.UnderwaterSciFi = type("UnderwaterSciFi", (), {})
    monkeypatch.setitem(sys.modules, "nyx.feas.archetypes.underwater_scifi", underwater_stub)
    feas_archetypes_pkg.underwater_scifi = underwater_stub

    capabilities_stub = types.ModuleType("nyx.feas.capabilities")
    capabilities_stub.merge_caps = lambda base, extra: {**(base or {}), **(extra or {})}
    monkeypatch.setitem(sys.modules, "nyx.feas.capabilities", capabilities_stub)
    nyx_feas_pkg.capabilities = capabilities_stub

    context_feas_stub = types.ModuleType("nyx.feas.context")
    context_feas_stub.build_affordance_index = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "nyx.feas.context", context_feas_stub)
    nyx_feas_pkg.context = context_feas_stub

    nyx_geo_pkg = types.ModuleType("nyx.geo")
    nyx_geo_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "nyx.geo", nyx_geo_pkg)
    nyx_pkg.geo = nyx_geo_pkg

    toponym_stub = types.ModuleType("nyx.geo.toponym")
    toponym_stub.plausibility_score = lambda *_args, **_kwargs: 1.0
    monkeypatch.setitem(sys.modules, "nyx.geo.toponym", toponym_stub)
    nyx_geo_pkg.toponym = toponym_stub

    nyx_location_pkg = types.ModuleType("nyx.location")
    nyx_location_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "nyx.location", nyx_location_pkg)
    nyx_pkg.location = nyx_location_pkg

    location_config_stub = types.ModuleType("nyx.location.config")

    @dataclass
    class LocationSettings:  # type: ignore
        pass

    location_config_stub.LocationSettings = LocationSettings
    monkeypatch.setitem(sys.modules, "nyx.location.config", location_config_stub)
    nyx_location_pkg.config = location_config_stub

    location_hierarchy_stub = types.ModuleType("nyx.location.hierarchy")

    async def _noop_assign(*_args, **_kwargs):
        return {}

    async def _noop_get_or_create(*_args, **_kwargs):
        return {}

    location_hierarchy_stub.assign_hierarchy = _noop_assign
    location_hierarchy_stub.get_or_create_location = _noop_get_or_create
    monkeypatch.setitem(sys.modules, "nyx.location.hierarchy", location_hierarchy_stub)
    nyx_location_pkg.hierarchy = location_hierarchy_stub

    location_policies_stub = types.ModuleType("nyx.location.policies")
    location_policies_stub.resolver_policy_for_context = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "nyx.location.policies", location_policies_stub)
    nyx_location_pkg.policies = location_policies_stub

    location_types_stub = types.ModuleType("nyx.location.types")

    @dataclass
    class Location:
        user_id: int
        conversation_id: int
        location_name: str
        id: Optional[int] = None

    location_types_stub.Location = Location
    location_types_stub.Anchor = object
    location_types_stub.Candidate = object
    location_types_stub.Place = object
    location_types_stub.ResolveResult = object
    location_types_stub.Scope = object
    location_types_stub.STATUS_ASK = "ask"
    location_types_stub.STATUS_EXACT = "exact"
    location_types_stub.STATUS_MULTIPLE = "multiple"
    location_types_stub.STATUS_TRAVEL_PLAN = "travel"
    monkeypatch.setitem(sys.modules, "nyx.location.types", location_types_stub)
    nyx_location_pkg.types = location_types_stub

    spec = importlib.util.spec_from_file_location(
        "feasibility_under_test",
        Path(__file__).resolve().parents[2] / "nyx" / "nyx_agent" / "feasibility.py",
    )
    feasibility = importlib.util.module_from_spec(spec)
    sys.modules["feasibility_under_test"] = feasibility
    assert spec and spec.loader
    spec.loader.exec_module(feasibility)

    stub_conn = _StubConnection()

    monkeypatch.setattr(
        feasibility,
        "get_db_connection_context",
        lambda: _StubConnectionContext(stub_conn),
    )

    nyx_ctx = NyxContext(user_id=42, conversation_id=888)

    context = await feasibility._load_comprehensive_context(nyx_ctx)

    location_object = context.get("location_object")
    assert location_object is not None
    assert location_object.user_id == 42
    assert location_object.conversation_id == 888
    assert location_object.location_name == "Shared Plaza"

    assert stub_conn.last_location_args == (42, 888, "Shared Plaza")
