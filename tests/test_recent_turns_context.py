import os
import sys
import types
from pathlib import Path
import typing

from typing_extensions import TypedDict as _CompatTypedDict

import asyncio

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
typing.TypedDict = _CompatTypedDict  # type: ignore[attr-defined]


class _StubSentenceTransformer:  # pragma: no cover - simple stub
    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode(self, *args, **kwargs):
        return []


stub_sentence_transformers = types.ModuleType("sentence_transformers")
stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)

stub_conflict_synthesizer = types.ModuleType("logic.conflict_system.conflict_synthesizer")
stub_conflict_synthesizer.get_synthesizer = lambda *args, **kwargs: None
stub_conflict_synthesizer.ConflictSynthesizer = object
stub_conflict_synthesizer.ConflictContext = object
stub_conflict_synthesizer.SubsystemType = object
stub_conflict_synthesizer.EventType = object
stub_conflict_synthesizer.SystemEvent = object

stub_conflict_background = types.ModuleType("logic.conflict_system.background_processor")
stub_conflict_background.get_conflict_scheduler = lambda *args, **kwargs: None

stub_conflict_pkg = types.ModuleType("logic.conflict_system")
stub_conflict_pkg.conflict_synthesizer = stub_conflict_synthesizer
stub_conflict_pkg.background_processor = stub_conflict_background

sys.modules.setdefault("logic.conflict_system", stub_conflict_pkg)
sys.modules["logic.conflict_system.conflict_synthesizer"] = stub_conflict_synthesizer
sys.modules["logic.conflict_system.background_processor"] = stub_conflict_background

from nyx.nyx_agent.context import (
    BundleSection,
    ContextBundle,
    NyxContext,
    SceneScope,
)


class _StubBroker:
    def __init__(self, bundle: ContextBundle) -> None:
        self._bundle = bundle

    async def compute_scene_scope(self, user_input, current_context):
        return self._bundle.scene_scope

    async def load_or_fetch_bundle(self, scene_scope):
        return self._bundle

    def log_metrics_line(self, scene_key, packed_context):  # pragma: no cover - noop
        return None


def test_recent_turns_propagated_into_bundle_metadata():
    scene_scope = SceneScope()
    bundle = ContextBundle(
        scene_scope=scene_scope,
        npcs=BundleSection(data={}, canonical=True),
        memories=BundleSection(data={}, canonical=True),
        lore=BundleSection(data={}, canonical=True),
        conflicts=BundleSection(data={}, canonical=True),
        world=BundleSection(data={}, canonical=True),
        narrative=BundleSection(data={}, canonical=True),
        metadata={},
    )
    broker = _StubBroker(bundle)

    context = NyxContext(user_id=1, conversation_id=2, context_broker=broker)
    context.current_context["recent_turns"] = [
        {"sender": "User", "content": "Hello"},
        {"sender": "Nyx", "content": "Welcome"},
    ]

    asyncio.run(context.build_context_for_input("How are things?", {}))

    assert bundle.metadata["recent_interactions"] == [
        {"sender": "User", "content": "Hello"},
        {"sender": "Nyx", "content": "Welcome"},
    ]
    assert bundle.narrative.data.get("recent") == [
        {"sender": "User", "content": "Hello"},
        {"sender": "Nyx", "content": "Welcome"},
    ]
    assert context.current_context["recent_turns"] == [
        {"sender": "User", "content": "Hello"},
        {"sender": "Nyx", "content": "Welcome"},
    ]


def test_lore_hints_recommend_tool_when_priority_high():
    scene_scope = SceneScope()
    bundle = ContextBundle(
        scene_scope=scene_scope,
        npcs=BundleSection(data={}, canonical=True),
        memories=BundleSection(data={}, canonical=True),
        lore=BundleSection(data={}, canonical=True),
        conflicts=BundleSection(data={}, canonical=True),
        world=BundleSection(data={}, canonical=True),
        narrative=BundleSection(data={}, canonical=True),
        metadata={},
    )
    broker = _StubBroker(bundle)

    context = NyxContext(user_id=3, conversation_id=4, context_broker=broker)

    user_input = "Tell me about the history of this temple?"
    asyncio.run(context.build_context_for_input(user_input, {}))

    hints = bundle.metadata.get("hints", {})
    assert hints.get("lore_priority", 0) >= 0.75
    assert hints.get("lore_tool_recommended") is True
    assert "location_history" in hints.get("suggested_aspects", [])
    assert "religious_context" in hints.get("suggested_aspects", [])
    assert context.current_context.get("hints") == hints
