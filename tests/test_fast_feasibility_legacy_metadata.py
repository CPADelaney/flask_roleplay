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
        self.location_records: dict[str, dict] = {}

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

    async def fetchval(self, query, *args):
        if "FROM CurrentRoleplay" in query:
            if "key='CurrentScene'" in query:
                return self.current_roleplay.get("CurrentScene")
            if "key='CurrentLocation'" in query:
                return self.current_roleplay.get("CurrentLocation")
            if "key='CurrentTime'" in query:
                return self.current_roleplay.get("CurrentTime")
        return None

    async def fetchrow(self, query, *args):
        if "FROM CurrentRoleplay" in query and "key='CurrentScene'" in query:
            value = self.current_roleplay.get("CurrentScene")
            if value is None:
                return None
            return {"value": value}
        if "FROM Locations" in query:
            location_name = args[2] if len(args) > 2 else None
            return self.location_records.get(location_name)
        return None


@pytest.fixture(autouse=True)
def fake_plausibility_scores(monkeypatch):
    scores: dict[str, float] = {
        "pier 39": 0.92,
        "hidden moon base": 0.02,
        "harbor in topeka": 0.05,
    }

    async def _fake_plausibility(name: str, *_args, **_kwargs) -> float:
        return scores.get(str(name).lower(), 0.0)

    monkeypatch.setattr(feasibility, "plausibility_score", _fake_plausibility)
    return scores


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
    router_payload = result.get("router_result") or {}
    assert router_payload.get("intents"), "router_result should include parsed intents"
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


def test_apply_travel_plan_metadata_adds_consequences():
    travel_operations = [
        {
            "op": "travel.plan",
            "legs": [
                {
                    "kind": "flight",
                    "dest_label": "Paris Charles de Gaulle",
                    "estimate_min": 600,
                },
                {
                    "kind": "train",
                    "dest_label": "Paris Central",
                    "estimate_min": 45,
                },
            ],
        }
    ]

    payload: dict[str, typing.Any] = {}
    feasibility._apply_travel_plan_metadata(payload, travel_operations)

    assert payload["resolver_travel_plan"] == travel_operations
    consequences = payload.get("consequences") or []
    assert len(consequences) == 2
    first, second = consequences
    assert first["method"] == "flight"
    assert first["destination"] == "Paris Charles de Gaulle"
    assert first["duration_minutes"] == pytest.approx(600.0)
    assert second["method"] == "train"
    assert second["destination"] == "Paris Central"
    assert second["duration_minutes"] == pytest.approx(45.0)


def test_apply_travel_plan_metadata_preserves_existing_consequences():
    travel_operations = [
        {
            "op": "travel.plan",
            "legs": [
                {
                    "kind": "shuttle",
                    "dest_label": "Orbital Platform",
                    "estimate_min": 30,
                }
            ],
        }
    ]

    payload: dict[str, typing.Any] = {"consequences": [{"method": "existing"}]}
    feasibility._apply_travel_plan_metadata(payload, travel_operations)

    assert payload["resolver_travel_plan"] == travel_operations
    consequences = payload.get("consequences") or []
    assert consequences[0] == {"method": "existing"}
    assert consequences[1]["method"] == "shuttle"
    assert consequences[1]["destination"] == "Orbital Platform"
    assert consequences[1]["duration_minutes"] == pytest.approx(30.0)


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
    assert overall.get("strategy") == "ask"
    assert overall.get("soft_location_only") is True
    assert per_intent.get("strategy") == "ask"
    violation_blob = json.dumps(per_intent.get("violations", []))
    assert "location_resolver:ask" in violation_blob
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


def test_load_current_scene_normalizes_dict_location(monkeypatch):
    fake_conn = FakeConnection()
    fake_conn.current_roleplay.update(
        {
            "CurrentScene": json.dumps({"location": {"name": "Atrium"}}),
        }
    )
    fake_conn.location_records["Atrium"] = {
        "notable_features": ["glass ceiling"],
        "hidden_aspects": ["secret door"],
        "description": "A bright atrium",
    }

    @asynccontextmanager
    async def fake_db_context():
        yield fake_conn

    monkeypatch.setattr(orchestrator, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    class DummyNyxCtx:
        user_id = 1
        conversation_id = 2

    scene = asyncio.run(feasibility._load_current_scene(DummyNyxCtx()))

    assert scene["location_features"] == ["glass ceiling"]
    assert scene["location_description"] == "A bright atrium"


def test_text_marker_requires_location_vocab_hit():
    intent = {
        "categories": [],
        "direct_object": ["password"],
    }
    no_move_tokens = {"password"}
    assert (
        feasibility._intent_requests_location_move(
            intent,
            "enter the password.",
            no_move_tokens,
        )
        is False
    )

    location_intent = {
        "categories": [],
        "direct_object": ["tavern"],
    }
    location_tokens = {"the tavern"}
    assert (
        feasibility._intent_requests_location_move(
            location_intent,
            "enter the tavern.",
            location_tokens,
        )
        is True
    )


@pytest.mark.parametrize(
    "original, normalized",
    [
        ("harbor in Topeka", "harbor in topeka"),
        ("the harbor", "the harbor"),
    ],
)
def test_real_world_toponym_keywords_require_supporting_evidence(original, normalized):
    modern_context = {"setting_kind": "modern_realistic"}

    assert (
        feasibility._looks_like_real_world_toponym(original, normalized, modern_context)
        is False
    )


@pytest.mark.parametrize("action_text", ["Find a harbor in Topeka."])
def test_fast_feasibility_denies_implausible_harbor(monkeypatch, action_text, fake_plausibility_scores):
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
            "WorldModel": json.dumps({"branch": "modern_realistic"}),
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
                "destination": ["harbor in Topeka"],
            }
        ]

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)

    async def _run():
        return await feasibility.assess_action_feasibility_fast(1, 2, action_text)

    result = asyncio.run(_run())

    overall = result.get("overall", {})
    per_intent = (result.get("per_intent") or [])[0]
    assert overall.get("feasible") is False
    assert overall.get("strategy") == "ask"
    assert overall.get("soft_location_only") is True
    assert per_intent.get("strategy") == "ask"
    violation_blob = json.dumps(per_intent.get("violations", []))
    assert "location_resolver:ask" in violation_blob
    assert "harbor in topeka" in violation_blob.lower()


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
