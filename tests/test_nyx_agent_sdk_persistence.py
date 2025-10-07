import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

stub_sentence_transformers = ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def encode(self, *_: Any, **__: Any) -> List[float]:
        return [0.0]


stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
stub_sentence_transformers.util = SimpleNamespace(cos_sim=lambda *_, **__: 0.0)

sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)


async def _noop_async(*_: Any, **__: Any) -> None:
    return None


class _StubNyxContext:
    def __init__(self, user_id: int, conversation_id: int) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context_broker = SimpleNamespace(load_or_fetch_bundle=_noop_async)
        self.current_context: Dict[str, Any] = {}
        self.current_location: Dict[str, Any] = {}

    async def initialize(self) -> None:
        return None


class _StubSceneScope(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def _stub_build_canonical_snapshot_payload(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    payload: Dict[str, Any] = {}
    for key in (
        "scene_id",
        "location_name",
        "region_id",
        "world_version",
        "conflict_id",
        "conflict_active",
        "time_window",
        "updated_at",
    ):
        if key not in snapshot:
            continue
        value = snapshot.get(key)
        if value is None and key not in {"world_version", "conflict_active"}:
            continue
        payload[key] = value
    participants = snapshot.get("participants")
    if isinstance(participants, (list, tuple, set)):
        payload["participants"] = [str(p) for p in participants]
    elif participants is not None:
        payload["participants"] = [str(participants)]
    return payload


async def _stub_persist_canonical_snapshot(*_: Any, **__: Any) -> None:
    return None


stub_context = ModuleType("nyx.nyx_agent.context")
stub_context.NyxContext = _StubNyxContext
stub_context.SceneScope = _StubSceneScope
stub_context.build_canonical_snapshot_payload = _stub_build_canonical_snapshot_payload
stub_context.persist_canonical_snapshot = _stub_persist_canonical_snapshot
stub_context.fetch_canonical_snapshot = _noop_async

sys.modules.setdefault("nyx.nyx_agent.context", stub_context)


stub_orchestrator = ModuleType("nyx.nyx_agent.orchestrator")


async def _stub_orchestrator_process(*_: Any, **__: Any) -> Dict[str, Any]:
    return {"response": "", "success": True, "metadata": {}}


def _stub_preserve_hydrated_location(*_: Any, **__: Any) -> None:
    return None


stub_orchestrator.process_user_input = _stub_orchestrator_process
stub_orchestrator._preserve_hydrated_location = _stub_preserve_hydrated_location

sys.modules.setdefault("nyx.nyx_agent.orchestrator", stub_orchestrator)


stub_models = ModuleType("nyx.nyx_agent.models")


@dataclass
class _StubModelsNyxResponse:
    narrative: str = ""


stub_models.NyxResponse = _StubModelsNyxResponse

sys.modules.setdefault("nyx.nyx_agent.models", stub_models)


stub_helpers = ModuleType("nyx.nyx_agent._feasibility_helpers")


class DeferPromptContext:  # type: ignore
    pass


def build_defer_fallback_text(*_: Any, **__: Any) -> str:
    return ""


def build_defer_prompt(*_: Any, **__: Any) -> str:
    return ""


def coalesce_agent_output_text(*_: Any, **__: Any) -> str:
    return ""


def extract_defer_details(*_: Any, **__: Any) -> Dict[str, Any]:
    return {}


stub_helpers.DeferPromptContext = DeferPromptContext
stub_helpers.build_defer_fallback_text = build_defer_fallback_text
stub_helpers.build_defer_prompt = build_defer_prompt
stub_helpers.coalesce_agent_output_text = coalesce_agent_output_text
stub_helpers.extract_defer_details = extract_defer_details

sys.modules.setdefault("nyx.nyx_agent._feasibility_helpers", stub_helpers)

import pytest

from nyx.nyx_agent_sdk import NyxAgentSDK, NyxResponse


class _DummySnapshotStore:
    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        return dict(self._data.get((user_id, conversation_id), {}))

    def put(self, user_id: str, conversation_id: str, snapshot: Dict[str, Any]) -> None:
        self._data[(user_id, conversation_id)] = dict(snapshot)


class _DummyDispatch:
    def __init__(self) -> None:
        self.calls = []

    def apply_async(self, *, kwargs: Dict[str, Any], queue: str, priority: int) -> None:
        self.calls.append({"kwargs": kwargs, "queue": queue, "priority": priority})


@pytest.mark.asyncio
async def test_fanout_persists_canonical_snapshot_with_integer_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = NyxAgentSDK()
    sdk._snapshot_store = _DummySnapshotStore()

    recorded: Dict[str, Any] = {}

    async def _record_persist(user_id: int, conversation_id: int, payload: Dict[str, Any]) -> None:
        recorded["args"] = (user_id, conversation_id, payload)

    dummy_dispatch = _DummyDispatch()

    monkeypatch.setattr("nyx.nyx_agent_sdk.persist_canonical_snapshot", _record_persist)
    monkeypatch.setattr("nyx.nyx_agent_sdk.post_turn_dispatch", dummy_dispatch)

    response = NyxResponse(narrative="hello", metadata={}, world_state={"delta": {"foo": "bar"}})

    await sdk._fanout_post_turn(response, "101", "202", trace_id="trace")

    assert recorded["args"][0] == 101
    assert recorded["args"][1] == 202
    assert recorded["args"][2]  # payload is not empty
    assert dummy_dispatch.calls


@pytest.mark.asyncio
async def test_fanout_skips_persistence_for_non_integer_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = NyxAgentSDK()
    sdk._snapshot_store = _DummySnapshotStore()

    called = False

    async def _record_persist(*_: Any, **__: Any) -> None:
        nonlocal called
        called = True

    dummy_dispatch = _DummyDispatch()

    monkeypatch.setattr("nyx.nyx_agent_sdk.persist_canonical_snapshot", _record_persist)
    monkeypatch.setattr("nyx.nyx_agent_sdk.post_turn_dispatch", dummy_dispatch)

    response = NyxResponse(narrative="hello", metadata={}, world_state={"delta": {"foo": "bar"}})

    await sdk._fanout_post_turn(response, "user-1", "conversation-A", trace_id="trace")

    assert called is False
    assert dummy_dispatch.calls
