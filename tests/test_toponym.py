import asyncio

import pytest

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


def test_world_model_override_thresholds():
    metadata = {
        "branch": "modern_realistic",
        "allow_fictional_locations": False,
        "resolver": {
            "allow_threshold": 0.88,
            "ask_threshold": 0.66,
            "fictional_policy": "ask",
        },
    }
    base_context = {
        "kind": "modern_realistic",
        "type": "modern_realistic",
        "reality_context": "normal",
    }

    normalized = feasibility._normalize_world_model_metadata(metadata, base_context)

    resolver_cfg = normalized.get("resolver", {})
    assert resolver_cfg.get("allow_threshold") == 0.88
    assert resolver_cfg.get("ask_threshold") == 0.66
    assert resolver_cfg.get("fictional_policy") == "ask"
