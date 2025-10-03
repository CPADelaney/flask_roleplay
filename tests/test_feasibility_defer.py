import asyncio
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
from nyx.nyx_agent._feasibility_helpers import extract_defer_details


def test_extract_defer_details_returns_guidance_and_leads():
    feasibility_payload = {
        "overall": {"feasible": False, "strategy": "defer"},
        "per_intent": [
            {
                "narrator_guidance": "You need to locate the key first.",
                "leads": ["Search the study", "Ask the caretaker"],
                "violations": [{"reason": "missing_prerequisite"}],
            }
        ],
    }

    guidance, leads, extra = extract_defer_details(feasibility_payload)

    assert guidance == "You need to locate the key first."
    assert leads == ["Search the study", "Ask the caretaker"]
    assert extra["leads"] == leads
    assert extra["violations"] == [{"reason": "missing_prerequisite"}]


def test_extract_defer_details_empty_for_non_defer():
    feasibility_payload = {
        "overall": {"feasible": True, "strategy": "allow"},
        "per_intent": [],
    }

    guidance, leads, extra = extract_defer_details(feasibility_payload)

    assert guidance == ""
    assert leads == []
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
