import asyncio
import os
import sys
import types
import typing
import typing_extensions

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: None
dummy_models.Pooling = lambda *args, **kwargs: None


class _StubSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **_kwargs):
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 3


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

typing.TypedDict = typing_extensions.TypedDict

from nyx.location.types import Location
from nyx.nyx_agent import feasibility


def _baseline_setting_context() -> dict:
    return {
        "kind": "modern_realistic",
        "type": "modern_realistic",
        "reality_context": "normal",
        "setting_era": "contemporary",
        "technology_level": "modern",
        "location": {"name": "Atrium"},
        "location_features": [],
        "known_location_names": [],
        "scene": {},
        "world_model": {
            "branch": "modern_realistic",
            "allow_fictional_locations": False,
            "resolver": {
                "allow_threshold": feasibility.LOCATION_RESOLVER_ALLOW_THRESHOLD,
                "ask_threshold": feasibility.LOCATION_RESOLVER_ASK_THRESHOLD,
                "fictional_policy": "deny",
            },
            "raw": {"branch": "modern_realistic"},
        },
    }


def test_resolver_allows_known_toponym(monkeypatch):
    captured = {}

    async def _fake_score(name: str, *_args, **_kwargs) -> float:
        captured["near"] = _kwargs.get("near")
        return 0.95 if str(name).lower() == "pier 39" else 0.0

    monkeypatch.setattr(feasibility, "plausibility_score", _fake_score)

    intent = {
        "raw_text": "Go to Pier 39.",
        "categories": ["movement"],
        "destination": ["Pier 39"],
    }
    text_l = intent["raw_text"].lower()
    setting_context = _baseline_setting_context()

    unresolved = asyncio.run(
        feasibility._find_unresolved_location_targets(
            intent,
            text_l,
            set(),
            set(),
            set(),
            set(),
            set(),
            setting_context,
        )
    )

    assert unresolved == []
    cache = setting_context.get("location_resolver_cache", {})
    verdict = cache.get("pier 39")
    assert verdict
    assert verdict.get("decision") == "allow"
    assert captured.get("near") == "Atrium"


def test_resolver_denies_implausible_request(monkeypatch):
    captured = {}

    async def _fake_score(name: str, *_args, **_kwargs) -> float:
        captured["near"] = _kwargs.get("near")
        lowered = str(name).lower()
        if lowered == "pier 39":
            return 0.95
        if lowered == "harbor in topeka":
            return 0.05
        return 0.0

    monkeypatch.setattr(feasibility, "plausibility_score", _fake_score)

    intent = {
        "raw_text": "Find a harbor in Topeka.",
        "categories": ["movement"],
        "destination": ["harbor in Topeka"],
    }
    text_l = intent["raw_text"].lower()
    setting_context = _baseline_setting_context()

    unresolved = asyncio.run(
        feasibility._find_unresolved_location_targets(
            intent,
            text_l,
            set(),
            set(),
            set(),
            set(),
            set(),
            setting_context,
        )
    )

    assert unresolved
    cache = setting_context.get("location_resolver_cache", {})
    verdict = cache.get("harbor in topeka") or cache.get("harbor in topeka".lower())
    assert verdict
    assert verdict.get("decision") == "deny"
    assert captured.get("near") == "Atrium"
    assert "resolver" in verdict.get("reason", "").lower()


def test_resolver_prefers_location_hierarchy_anchor(monkeypatch):
    captured = {}

    async def _fake_score(name: str, *_args, **_kwargs) -> float:
        captured["near"] = _kwargs.get("near")
        return 0.95 if str(name).lower() == "pier 39" else 0.0

    monkeypatch.setattr(feasibility, "plausibility_score", _fake_score)

    intent = {
        "raw_text": "Go to Pier 39.",
        "categories": ["movement"],
        "destination": ["Pier 39"],
    }
    text_l = intent["raw_text"].lower()
    setting_context = _baseline_setting_context()
    location_obj = Location(
        user_id=1,
        conversation_id=1,
        location_name="Pier 39",
        district="North Beach",
        city="San Francisco",
        region="California",
        country="USA",
        is_fictional=False,
    )
    setting_context["_location_object"] = location_obj
    setting_context["location_object"] = location_obj.to_dict()

    unresolved = asyncio.run(
        feasibility._find_unresolved_location_targets(
            intent,
            text_l,
            set(),
            set(),
            set(),
            set(),
            set(),
            setting_context,
        )
    )

    assert unresolved == []
    assert captured.get("near") == "North Beach, San Francisco, California, USA"
