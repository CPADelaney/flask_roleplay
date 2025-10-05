import asyncio
import json
import os
import sys
import types
import typing

import pytest
import typing_extensions

typing.TypedDict = typing_extensions.TypedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: None
dummy_models.Pooling = lambda *args, **kwargs: None


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):  # pragma: no cover - defensive stub
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):  # pragma: no cover - defensive stub
        return 3


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from nyx.nyx_agent import feasibility
from nyx.nyx_agent._feasibility_helpers import (
    DeferPromptContext,
    build_defer_fallback_text,
    build_defer_prompt,
    coalesce_agent_output_text,
    extract_defer_details,
)
from nyx.nyx_agent_sdk import NyxAgentSDK


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_extract_defer_details_returns_context_and_leads():
    feasibility_payload = {
        "overall": {"feasible": False, "strategy": "defer"},
        "per_intent": [
            {
                "narrator_guidance": "You need to locate the key first.",
                "leads": ["Search the study", "Ask the caretaker"],
                "violations": [{"reason": "you haven't located the key yet"}],
            }
        ],
    }

    context, extra = extract_defer_details(feasibility_payload)

    assert context is not None
    assert context.persona_prefix in ("Oh, pet,", "Mmm, kitten,", "Sweet thing,")
    assert "locate the key" in context.narrator_guidance.lower()
    assert context.leads == ["Search the study", "Ask the caretaker"]
    assert extra["leads"] == context.leads
    assert extra["violations"] == [{"reason": "you haven't located the key yet"}]

    fallback = build_defer_fallback_text(context)
    assert any(keyword in fallback.lower() for keyword in ("pet", "kitten", "sweet thing"))
    assert "locate the key" in fallback.lower()


def test_coalesce_agent_output_prefers_final_output():
    agent_result = types.SimpleNamespace(
        final_output="  Oh, pet, bring me the key first.  ",
        messages=[{"content": "fallback"}],
    )

    assert (
        coalesce_agent_output_text(agent_result)
        == "Oh, pet, bring me the key first."
    )

    dict_result = {
        "final_output": "  Sweet thing, patience.  ",
        "messages": [{"content": "not used"}],
    }

    assert coalesce_agent_output_text(dict_result) == "Sweet thing, patience."


def test_build_defer_prompt_includes_disbelief_instruction_for_absent_targets():
    context = DeferPromptContext(
        narrator_guidance="There's no bull anywhere in sight—what fantasy are you chasing?",
        leads=["Talk to the bartender"],
        violations=[
            {
                "rule": "npc_absent",
                "reason": "No sign of bull anywhere in this scene—I'm genuinely baffled about who you're trying to engage.",
            }
        ],
        persona_prefix="Oh, pet,",
        reason_phrases=[
            "No sign of bull anywhere in this scene—I'm genuinely baffled about who you're trying to engage."
        ],
    )

    prompt = build_defer_prompt(context)

    lower_prompt = prompt.lower()
    assert "exasperated disbelief" in lower_prompt
    assert "imaginary" in lower_prompt


def test_extract_defer_details_empty_for_non_defer():
    feasibility_payload = {
        "overall": {"feasible": True, "strategy": "allow"},
        "per_intent": [],
    }

    context, extra = extract_defer_details(feasibility_payload)

    assert context is None
    assert extra == {}


def test_assess_action_feasibility_defers_for_missing_mundane_prereqs(monkeypatch):
    async def fake_parse_action_intents(_):
        return [
            {
                "categories": ["mundane_action"],
                "instruments": ["rope"],
                "direct_object": ["shopkeeper"],
            }
        ]

    async def fake_load_context(_):
        return {
            "caps_loaded": True,
            "capabilities": {"can_trade": True},
            "available_items": ["rope"],
            "present_entities": [],
            "location": {"name": "Quiet plaza"},
            "hard_rules": [],
            "established_impossibilities": [],
        }

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)
    monkeypatch.setattr(feasibility, "_load_comprehensive_context", fake_load_context)

    class DummyCtx:
        pass

    async def _run():
        result = await feasibility.assess_action_feasibility(DummyCtx(), "tie up the shopkeeper")

        assert result["overall"] == {"feasible": False, "strategy": "defer"}
        assert result["per_intent"], "expected per-intent results"
        intent_result = result["per_intent"][0]
        assert intent_result["strategy"] == "defer"
        assert intent_result["feasible"] is False
        assert intent_result["violations"][0]["rule"] == "missing_prereq"
        assert "shopkeeper" in intent_result["violations"][0]["reason"].lower()

    asyncio.run(_run())


@pytest.mark.anyio
async def test_fast_feasibility_defer_on_missing_scene_entities(monkeypatch):
    async def fake_parse_action_intents(_text):
        return [
            {
                "categories": ["social"],
                "direct_object": ["captain mira"],
                "instruments": ["plasma lance"],
            }
        ]

    scene_payload = {
        "npcs": ["dockhand"],
        "items": ["wrench"],
        "location_features": ["cargo crates"],
        "time_phase": "night",
    }

    class DummyConn:
        async def fetch(self, query, *args):
            query_str = str(query)
            if "CurrentRoleplay" in query_str:
                return [
                    {"key": "SettingKind", "value": "modern_realistic"},
                    {"key": "CurrentScene", "value": json.dumps(scene_payload)},
                    {"key": "CurrentLocation", "value": "Hangar Bay"},
                    {"key": "SettingCapabilities", "value": json.dumps({"technology": "modern"})},
                    {"key": "EstablishedImpossibilities", "value": json.dumps([])},
                ]
            if "GameRules" in query_str:
                return []
            return []

    class DummyContext:
        async def __aenter__(self):
            return DummyConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)
    monkeypatch.setattr(feasibility, "get_db_connection_context", lambda: DummyContext())

    result = await feasibility.assess_action_feasibility_fast(
        user_id=7,
        conversation_id=42,
        text="I call out to Captain Mira with the plasma lance",
    )

    assert result["overall"] == {"feasible": False, "strategy": "defer"}
    assert result["per_intent"], "expected per-intent details"
    intent_result = result["per_intent"][0]
    assert intent_result["strategy"] == "defer"
    assert intent_result["leads"]
    assert any(v["rule"] == "npc_absent" for v in intent_result["violations"])
    assert any("dockhand" in lead.lower() for lead in intent_result["leads"])

    monkeypatch.setattr("nyx.nyx_agent_sdk.content_moderation_guardrail", None, raising=False)

    async def fail_orchestrator_call(*_args, **_kwargs):
        raise AssertionError("orchestrator should not be invoked on defer")

    monkeypatch.setattr(
        NyxAgentSDK,
        "_call_orchestrator_with_timeout",
        fail_orchestrator_call,
        raising=False,
    )

    class DummyRunner:
        called = False
        last_prompt = None

        @staticmethod
        async def run(_agent, prompt, **_kwargs):
            DummyRunner.called = True
            DummyRunner.last_prompt = prompt
            return types.SimpleNamespace(
                final_output="  Mmm, kitten, find the dockhand before you posture.  ",
                messages=[{"content": "fallback"}],
            )

    monkeypatch.setattr("nyx.nyx_agent_sdk.Runner", DummyRunner, raising=False)
    monkeypatch.setattr("nyx.nyx_agent_sdk.nyx_main_agent", object(), raising=False)

    sdk = NyxAgentSDK()
    response = await sdk.process_user_input(
        message="I call out to Captain Mira with the plasma lance",
        conversation_id="42",
        user_id="7",
        metadata={},
    )

    assert response.metadata.get("action_deferred") is True
    assert response.metadata.get("strategy") == "defer"
    assert DummyRunner.called is True
    assert "dockhand" in (DummyRunner.last_prompt or "").lower()
    assert response.narrative == "Mmm, kitten, find the dockhand before you posture."


@pytest.mark.anyio
async def test_fast_feasibility_ignores_inherent_instruments(monkeypatch):
    async def fake_parse_action_intents(_text):
        return [
            {
                "categories": ["mundane_action"],
                "direct_object": ["dockhand"],
                "instruments": ["hands", "mouth", "grappling hook"],
            }
        ]

    scene_payload = {
        "npcs": ["dockhand"],
        "items": ["rope"],
        "location_features": ["cargo crates"],
        "time_phase": "night",
    }

    class DummyConn:
        async def fetch(self, query, *args):
            query_str = str(query)
            if "CurrentRoleplay" in query_str:
                return [
                    {"key": "SettingKind", "value": "modern_realistic"},
                    {"key": "CurrentScene", "value": json.dumps(scene_payload)},
                    {"key": "CurrentLocation", "value": "Hangar Bay"},
                    {"key": "SettingCapabilities", "value": json.dumps({"technology": "modern"})},
                    {"key": "EstablishedImpossibilities", "value": json.dumps([])},
                ]
            if "GameRules" in query_str:
                return []
            return []

    class DummyContext:
        async def __aenter__(self):
            return DummyConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)
    monkeypatch.setattr(feasibility, "get_db_connection_context", lambda: DummyContext())

    result = await feasibility.assess_action_feasibility_fast(
        user_id=9,
        conversation_id=99,
        text="I lash out with my hands, teeth, and a grappling hook",
    )

    per_intent = result["per_intent"][0]
    assert per_intent["strategy"] == "defer"
    item_violation = next(v for v in per_intent["violations"] if v["rule"] == "item_absent")
    reason = item_violation["reason"].lower()
    assert "grappling hook" in reason
    assert "hand" not in reason
    assert "mouth" not in reason


@pytest.mark.anyio
async def test_fast_feasibility_missing_targets_uses_sharper_language(monkeypatch):
    async def fake_parse_action_intents(_text):
        return [
            {
                "categories": ["mundane_action"],
                "direct_object": ["bull"],
                "instruments": ["fruit roll-up"],
            }
        ]

    scene_payload = {
        "npcs": ["bartender"],
        "items": ["bar stool"],
        "location_features": ["sticky floor"],
        "time_phase": "dawn",
    }

    class DummyConn:
        async def fetch(self, query, *args):
            query_str = str(query)
            if "CurrentRoleplay" in query_str:
                return [
                    {"key": "SettingKind", "value": "modern_realistic"},
                    {"key": "CurrentScene", "value": json.dumps(scene_payload)},
                    {"key": "CurrentLocation", "value": "Tavern"},
                    {"key": "SettingCapabilities", "value": json.dumps({"technology": "modern"})},
                    {"key": "EstablishedImpossibilities", "value": json.dumps([])},
                ]
            if "GameRules" in query_str:
                return []
            return []

    class DummyContext:
        async def __aenter__(self):
            return DummyConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)
    monkeypatch.setattr(feasibility, "get_db_connection_context", lambda: DummyContext())

    result = await feasibility.assess_action_feasibility_fast(
        user_id=12,
        conversation_id=88,
        text="I stare down the bull and whip out a fruit roll-up",
    )

    per_intent = result["per_intent"][0]
    guidance = per_intent["narrator_guidance"].lower()
    assert "what fantasy are you chasing" in guidance
    assert "bull" in guidance
    assert "fruit roll-up" in guidance

    target_violation = next(v for v in per_intent["violations"] if v["rule"] == "npc_absent")
    item_violation = next(v for v in per_intent["violations"] if v["rule"] == "item_absent")

    assert "no sign of bull" in target_violation["reason"].lower()
    assert "baffled" in target_violation["reason"].lower()
    assert "no sign of fruit roll-up" in item_violation["reason"].lower()
    assert "baffled" in item_violation["reason"].lower()


@pytest.mark.anyio
async def test_fast_feasibility_allows_location_reference_tokens(monkeypatch):
    async def fake_parse_action_intents(_text):
        return [
            {
                "categories": ["question"],
                "direct_object": ["current_location"],
                "instruments": [],
            }
        ]

    scene_payload = {
        "npcs": ["bartender"],
        "items": ["bar stool"],
        "location_features": ["oak bar", "framed map"],
        "time_phase": "day",
    }

    class DummyConn:
        async def fetch(self, query, *args):
            query_str = str(query)
            if "CurrentRoleplay" in query_str:
                return [
                    {"key": "SettingKind", "value": "modern_realistic"},
                    {"key": "CurrentScene", "value": json.dumps(scene_payload)},
                    {"key": "CurrentLocation", "value": "Tavern"},
                    {"key": "SettingCapabilities", "value": json.dumps({"technology": "modern"})},
                    {"key": "EstablishedImpossibilities", "value": json.dumps([])},
                ]
            if "GameRules" in query_str:
                return []
            return []

    class DummyContext:
        async def __aenter__(self):
            return DummyConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)
    monkeypatch.setattr(feasibility, "get_db_connection_context", lambda: DummyContext())

    result = await feasibility.assess_action_feasibility_fast(
        user_id=21,
        conversation_id=84,
        text="Where am I right now?",
    )

    assert result["overall"] == {"feasible": True, "strategy": "allow"}
    per_intent = result["per_intent"][0]
    assert per_intent["strategy"] == "allow"
    assert not any(v.get("rule") == "npc_absent" for v in per_intent.get("violations", []))


@pytest.mark.anyio
async def test_fast_feasibility_allows_search_for_mundane_items(monkeypatch):
    async def fake_parse_action_intents(_text):
        return [
            {
                "categories": ["investigation"],
                "direct_object": [],
                "instruments": ["coin"],
            }
        ]

    scene_payload = {
        "npcs": ["merchant"],
        "items": ["crate"],
        "location_features": ["bustling stalls", "flagstone floor"],
        "time_phase": "day",
    }

    class DummyConn:
        async def fetch(self, query, *args):
            query_str = str(query)
            if "CurrentRoleplay" in query_str:
                return [
                    {"key": "SettingKind", "value": "modern_realistic"},
                    {"key": "CurrentScene", "value": json.dumps(scene_payload)},
                    {"key": "CurrentLocation", "value": "Bazaar"},
                    {"key": "SettingCapabilities", "value": json.dumps({"technology": "modern"})},
                    {"key": "EstablishedImpossibilities", "value": json.dumps([])},
                ]
            if "GameRules" in query_str:
                return []
            return []

    class DummyContext:
        async def __aenter__(self):
            return DummyConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)
    monkeypatch.setattr(feasibility, "get_db_connection_context", lambda: DummyContext())

    result = await feasibility.assess_action_feasibility_fast(
        user_id=22,
        conversation_id=85,
        text="I look around the stalls for a loose coin.",
    )

    assert result["overall"] == {"feasible": True, "strategy": "allow"}
    per_intent = result["per_intent"][0]
    assert per_intent["strategy"] == "allow"
    assert not any(v.get("rule") == "item_absent" for v in per_intent.get("violations", []))


@pytest.mark.anyio
async def test_fast_feasibility_allows_search_for_rocks(monkeypatch):
    async def fake_parse_action_intents(_text):
        return [
            {
                "categories": ["investigation"],
                "direct_object": [],
                "instruments": ["rock"],
            }
        ]

    scene_payload = {
        "npcs": ["ranger"],
        "items": ["satchel"],
        "location_features": ["winding trail", "loose gravel"],
        "time_phase": "dusk",
    }

    class DummyConn:
        async def fetch(self, query, *args):
            query_str = str(query)
            if "CurrentRoleplay" in query_str:
                return [
                    {"key": "SettingKind", "value": "modern_realistic"},
                    {"key": "CurrentScene", "value": json.dumps(scene_payload)},
                    {"key": "CurrentLocation", "value": "Forest Path"},
                    {"key": "SettingCapabilities", "value": json.dumps({"technology": "modern"})},
                    {"key": "EstablishedImpossibilities", "value": json.dumps([])},
                ]
            if "GameRules" in query_str:
                return []
            return []

    class DummyContext:
        async def __aenter__(self):
            return DummyConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(feasibility, "parse_action_intents", fake_parse_action_intents)
    monkeypatch.setattr(feasibility, "get_db_connection_context", lambda: DummyContext())

    result = await feasibility.assess_action_feasibility_fast(
        user_id=23,
        conversation_id=86,
        text="I scour the trail for a smooth rock I can toss.",
    )

    assert result["overall"] == {"feasible": True, "strategy": "allow"}
    per_intent = result["per_intent"][0]
    assert per_intent["strategy"] == "allow"
    assert not any(v.get("rule") == "item_absent" for v in per_intent.get("violations", []))


@pytest.mark.anyio
async def test_sdk_defer_response_includes_reason_and_tone(monkeypatch):
    async def fake_assess_action_feasibility_fast(**_kwargs):
        return {
            "overall": {"feasible": False, "strategy": "defer"},
            "per_intent": [
                {
                    "narrator_guidance": "You need to locate the key first",
                    "leads": ["search the study"],
                    "violations": [{"reason": "you haven't located the key yet"}],
                }
            ],
        }

    monkeypatch.setattr(
        "nyx.nyx_agent.feasibility.assess_action_feasibility_fast",
        fake_assess_action_feasibility_fast,
        raising=False,
    )
    monkeypatch.setattr(
        "nyx.nyx_agent_sdk.assess_action_feasibility_fast",
        fake_assess_action_feasibility_fast,
        raising=False,
    )
    monkeypatch.setattr("nyx.nyx_agent_sdk.content_moderation_guardrail", None, raising=False)

    async def fail_orchestrator_call(*_args, **_kwargs):
        raise AssertionError("orchestrator should not be called when defer is handled pre-orchestration")

    monkeypatch.setattr(
        NyxAgentSDK,
        "_call_orchestrator_with_timeout",
        fail_orchestrator_call,
        raising=False,
    )

    class DummyRunner:
        called = False
        last_prompt = None

        @staticmethod
        async def run(_agent, prompt, **_kwargs):
            DummyRunner.called = True
            DummyRunner.last_prompt = prompt
            return types.SimpleNamespace(
                final_output="  Oh, pet, you haven't located the key yet. Bring me the key first.  ",
                messages=[{"content": "fallback"}],
            )

    monkeypatch.setattr("nyx.nyx_agent_sdk.Runner", DummyRunner, raising=False)
    monkeypatch.setattr("nyx.nyx_agent_sdk.nyx_main_agent", object(), raising=False)

    sdk = NyxAgentSDK()
    response = await sdk.process_user_input(
        message="I shove past the lock and barge inside",
        conversation_id="1",
        user_id="1",
        metadata={},
    )

    assert response.metadata.get("action_deferred") is True
    assert response.metadata["violations"][0]["reason"] == "you haven't located the key yet"
    assert DummyRunner.called is True
    assert "you haven't located the key yet" in DummyRunner.last_prompt
    assert response.narrative == "Oh, pet, you haven't located the key yet. Bring me the key first."
    violation_reason = response.metadata["violations"][0]["reason"].lower()
    assert violation_reason in response.narrative.lower()
