import asyncio
import importlib
import json
import os
import sys
import types
import typing
from contextlib import asynccontextmanager

import pytest
import typing_extensions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

typing.TypedDict = typing_extensions.TypedDict

dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: None
dummy_models.Pooling = lambda *args, **kwargs: None


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 3


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

orchestrator = importlib.import_module("nyx.nyx_agent.orchestrator")
feasibility = importlib.import_module("nyx.nyx_agent.feasibility")


class FakeConnection:
    def __init__(self):
        self.current_roleplay: dict[str, str] = {}
        self.locations: list[str] = []

    async def execute(self, query, *args):
        if "CurrentRoleplay" in query and "CurrentScene" in query:
            self.current_roleplay["CurrentScene"] = args[2]
        elif "CurrentRoleplay" in query and "CurrentLocation" in query:
            self.current_roleplay["CurrentLocation"] = args[2]
        elif "NyxAgentState" in query:
            self.current_roleplay["NyxAgentState"] = args[2]
        return None

    async def fetch(self, query, *args):
        if "FROM CurrentRoleplay" in query and "key = ANY" in query:
            keys = args[2]
            return [
                {"key": key, "value": self.current_roleplay.get(key)}
                for key in keys
                if key in self.current_roleplay
            ]
        if "SELECT location_name FROM Locations" in query:
            return [
                {"location_name": name}
                for name in self.locations
            ]
        if "FROM GameRules" in query:
            return []
        return []

    async def fetchval(self, *args, **kwargs):
        return None

    async def fetchrow(self, query, *args):
        if "FROM CurrentRoleplay" in query and "key='CurrentScene'" in query:
            value = self.current_roleplay.get("CurrentScene")
            if value is None:
                return None
            return {"value": value}
        return None


@pytest.mark.parametrize("action_text", ["Talk to Mallory about the treasure."])
def test_fast_feasibility_blocks_absent_entities(monkeypatch, action_text):
    fake_conn = FakeConnection()
    fake_conn.current_roleplay.update(
        {
            "SettingCapabilities": json.dumps({"technology": "modern"}),
            "SettingType": "modern_realistic",
            "SettingKind": "modern_realistic",
            "RealityContext": "normal",
            "PhysicsModel": "realistic",
        }
    )

    @asynccontextmanager
    async def fake_db_context():
        yield fake_conn

    monkeypatch.setattr(orchestrator, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    async def fake_parse_action_intents(text: str):
        return [
            {
                "raw_text": text,
                "categories": ["dialogue"],
                "direct_object": ["Mallory"],
            }
        ]

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)

    class DummyCtx:
        def __init__(self):
            self.current_context = {
                "location": {"name": "Atrium"},
                "npc_present": ["Ava"],
                "items": ["Silver Key"],
            }
            self.user_id = 1
            self.conversation_id = 2
            self.scenario_state = {}
            self._tables_available = {"scenario_states": True}
            self.learning_metrics = {}
            self.learned_patterns = {}
            self.performance_metrics = {}
            self.error_log = []

        def should_run_task(self, _name: str) -> bool:
            return False

        def record_task_run(self, _name: str) -> None:
            return None

    async def _run():
        ctx = DummyCtx()
        await orchestrator._save_context_state(ctx)
        stored_scene = json.loads(fake_conn.current_roleplay.get("CurrentScene", "{}"))
        assert stored_scene["npcs"] == ["Ava"]
        assert stored_scene["items"] == ["Silver Key"]

        result = await feasibility.assess_action_feasibility_fast(
            ctx.user_id,
            ctx.conversation_id,
            action_text,
        )
        return stored_scene, result

    stored_scene, result = asyncio.run(_run())

    assert stored_scene["location"]["name"] == "Atrium"
    overall = result.get("overall", {})
    per_intent = (result.get("per_intent") or [])[0]
    assert overall.get("feasible") is False
    assert overall.get("strategy") == "deny"
    assert per_intent.get("strategy") == "deny"
    violation_blob = json.dumps(per_intent.get("violations", []))
    assert "mallory" in violation_blob.lower()


@pytest.mark.parametrize("action_text", ["Inspect the location for clues."])
def test_fast_feasibility_allows_location_reference(monkeypatch, action_text):
    fake_conn = FakeConnection()
    fake_conn.current_roleplay.update(
        {
            "SettingCapabilities": json.dumps({"technology": "modern"}),
            "SettingType": "modern_realistic",
            "SettingKind": "modern_realistic",
            "RealityContext": "normal",
            "PhysicsModel": "realistic",
            "CurrentLocation": "Atrium",
            "CurrentScene": json.dumps({"location": {"name": "Atrium"}}),
        }
    )
    fake_conn.locations = ["Atrium"]

    @asynccontextmanager
    async def fake_db_context():
        yield fake_conn

    monkeypatch.setattr(orchestrator, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    async def fake_parse_action_intents(text: str):
        return [
            {
                "raw_text": text,
                "categories": ["investigation"],
                "direct_object": ["location"],
            }
        ]

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)

    async def _run():
        return await feasibility.assess_action_feasibility_fast(1, 2, action_text)

    result = asyncio.run(_run())

    overall = result.get("overall", {})
    per_intent = (result.get("per_intent") or [])[0]
    assert overall.get("feasible") is True
    assert overall.get("strategy") == "allow"
    assert per_intent.get("feasible") is True
    assert per_intent.get("strategy") == "allow"


@pytest.mark.parametrize("action_text", ["Go to the hidden moon base."])
def test_fast_feasibility_blocks_unknown_location(monkeypatch, action_text):
    fake_conn = FakeConnection()
    fake_conn.current_roleplay.update(
        {
            "SettingCapabilities": json.dumps({"technology": "modern"}),
            "SettingType": "modern_realistic",
            "SettingKind": "modern_realistic",
            "RealityContext": "normal",
            "PhysicsModel": "realistic",
            "CurrentLocation": "Atrium",
            "CurrentScene": json.dumps({"location": {"name": "Atrium"}}),
        }
    )
    fake_conn.locations = ["Atrium"]

    @asynccontextmanager
    async def fake_db_context():
        yield fake_conn

    monkeypatch.setattr(orchestrator, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    async def fake_parse_action_intents(text: str):
        return [
            {
                "raw_text": text,
                "categories": ["movement"],
                "direct_object": ["Hidden Moon Base"],
            }
        ]

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)

    async def _run():
        return await feasibility.assess_action_feasibility_fast(1, 2, action_text)

    result = asyncio.run(_run())

    overall = result.get("overall", {})
    per_intent = (result.get("per_intent") or [])[0]
    assert overall.get("feasible") is False
    assert overall.get("strategy") == "deny"
    assert per_intent.get("strategy") == "deny"
    violation_blob = json.dumps(per_intent.get("violations", []))
    assert "location_absent" in violation_blob
    assert "hidden moon base" in violation_blob.lower()


@pytest.mark.parametrize("action_text", ["Go to Pier 39."])
def test_fast_feasibility_accepts_real_world_toponyms(monkeypatch, action_text):
    fake_conn = FakeConnection()
    fake_conn.current_roleplay.update(
        {
            "SettingCapabilities": json.dumps({"technology": "modern"}),
            "SettingType": "modern_realistic",
            "SettingKind": "modern_realistic",
            "RealityContext": "normal",
            "PhysicsModel": "realistic",
            "CurrentLocation": "Atrium",
            "CurrentScene": json.dumps({"location": {"name": "Atrium"}}),
        }
    )
    fake_conn.locations = ["Atrium"]

    @asynccontextmanager
    async def fake_db_context():
        yield fake_conn

    monkeypatch.setattr(orchestrator, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    async def fake_parse_action_intents(text: str):
        return [
            {
                "raw_text": text,
                "categories": ["movement"],
                "destination": ["Pier 39"],
            }
        ]

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)

    async def _run():
        return await feasibility.assess_action_feasibility_fast(1, 2, action_text)

    result = asyncio.run(_run())

    overall = result.get("overall", {})
    per_intent = (result.get("per_intent") or [])[0]
    assert overall.get("feasible") is True
    assert overall.get("strategy") == "allow"
    assert per_intent.get("feasible") is True
    assert per_intent.get("strategy") == "allow"
    violation_blob = json.dumps(per_intent.get("violations", []))
    assert "location_absent" not in violation_blob


@pytest.mark.parametrize("action_text", ["Head to a bar."])
def test_fast_feasibility_accepts_generic_venue_requests(monkeypatch, action_text):
    fake_conn = FakeConnection()
    fake_conn.current_roleplay.update(
        {
            "SettingCapabilities": json.dumps({"technology": "modern"}),
            "SettingType": "modern_realistic",
            "SettingKind": "modern_realistic",
            "RealityContext": "normal",
            "PhysicsModel": "realistic",
            "CurrentLocation": "Atrium",
            "CurrentScene": json.dumps({
                "location": {"name": "Atrium"},
                "location_features": ["bar", "atrium"]
            }),
        }
    )
    fake_conn.locations = ["Atrium"]

    @asynccontextmanager
    async def fake_db_context():
        yield fake_conn

    monkeypatch.setattr(orchestrator, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    async def fake_parse_action_intents(text: str):
        return [
            {
                "raw_text": text,
                "categories": ["movement"],
                "destination": ["a bar"],
            }
        ]

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)

    async def _run():
        return await feasibility.assess_action_feasibility_fast(1, 2, action_text)

    result = asyncio.run(_run())

    overall = result.get("overall", {})
    per_intent = (result.get("per_intent") or [])[0]
    assert overall.get("feasible") is True
    assert overall.get("strategy") == "allow"
    assert per_intent.get("feasible") is True
    assert per_intent.get("strategy") == "allow"
    violation_blob = json.dumps(per_intent.get("violations", []))
    assert "location_absent" not in violation_blob


def test_hydrated_location_survives_normalization_roundtrip():
    hydrated = "Atrium"
    raw_context = {
        "current_location": {},
        "location": "",
        "location_name": None,
        "location_id": None,
    }

    base_context = orchestrator._normalize_scene_context(raw_context)
    orchestrator._preserve_hydrated_location(base_context, hydrated)
    renormalized = orchestrator._normalize_scene_context(base_context)

    assert renormalized["current_location"] == hydrated
    assert renormalized["location"] == hydrated
    assert renormalized["location_name"] == hydrated
    assert renormalized["location_id"] == hydrated
