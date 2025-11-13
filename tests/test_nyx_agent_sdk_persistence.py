import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import asyncio
import time
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


class _RecorderConversationStore:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def append_turn(
        self,
        *,
        user_id: Any,
        conversation_id: Any,
        turn: Dict[str, Any],
    ) -> None:
        self.calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "turn": dict(turn),
            }
        )


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


class _StubBaseModel:
    pass


stub_models.NyxResponse = _StubModelsNyxResponse
stub_models.BaseModel = _StubBaseModel

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

stub_package = ModuleType("nyx.nyx_agent")
stub_package.__path__ = []  # type: ignore[attr-defined]
stub_package.orchestrator = stub_orchestrator
stub_package.context = stub_context
stub_package.models = stub_models
stub_package._feasibility_helpers = stub_helpers

sys.modules.setdefault("nyx.nyx_agent", stub_package)

import pytest

from nyx.nyx_agent_sdk import NyxAgentSDK, NyxResponse, NyxSDKConfig


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


def test_fanout_persists_canonical_snapshot_with_integer_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
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

    asyncio.run(_run())


def test_fanout_skips_persistence_for_non_integer_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
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

    asyncio.run(_run())


def test_process_user_input_appends_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        sdk = NyxAgentSDK(
            NyxSDKConfig(
                rate_limit_per_conversation=False,
                enable_response_filter=False,
                enable_telemetry=False,
            )
        )
        recorder = _RecorderConversationStore()
        sdk._conversation_store = recorder

        async def _fake_orchestrator(*_: Any, **__: Any) -> Dict[str, Any]:
            return {
                "response": "Narrative from Nyx",
                "success": True,
                "metadata": {"world": {}, "telemetry": {}},
            }

        async def _fake_async(*_: Any, **__: Any) -> None:
            return None

        monkeypatch.setattr(sdk, "_call_orchestrator_with_timeout", _fake_orchestrator)
        monkeypatch.setattr(sdk, "_fanout_post_turn", _fake_async)
        monkeypatch.setattr(sdk, "_maybe_enqueue_maintenance", _fake_async)
        monkeypatch.setattr(sdk, "_maybe_log_perf", _fake_async)
        monkeypatch.setattr(
            "nyx.nyx_agent_sdk._invalidate_context_cache_safe", _fake_async, raising=False
        )

        response = await sdk.process_user_input(
            "Hello Nyx",
            conversation_id="42",
            user_id="7",
            metadata={"player_name": "Hero", "nyx_display_name": "Narrator"},
        )

        assert response.narrative == "Narrative from Nyx"
        assert len(recorder.calls) == 2
        first, second = recorder.calls
        assert first["turn"]["sender"] == "Hero"
        assert first["turn"]["content"] == "Hello Nyx"
        assert second["turn"]["sender"] == "Narrator"
        assert second["turn"]["content"] == "Narrative from Nyx"
        assert second["turn"].get("metadata") == response.metadata

    asyncio.run(_run())


def test_fallback_run_appends_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        sdk = NyxAgentSDK(
            NyxSDKConfig(
                rate_limit_per_conversation=False,
                enable_response_filter=False,
                enable_telemetry=False,
            )
        )
        recorder = _RecorderConversationStore()
        sdk._conversation_store = recorder

        class _StubModelSettings:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class _StubRunConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class _StubRunContextWrapper:
            def __init__(self, ctx: Any) -> None:
                self.ctx = ctx

        class _Wrapper:
            def __init__(self, text: str) -> None:
                self.raw = SimpleNamespace(final_output=text)

        async def _fake_execute(*_: Any, **__: Any) -> _Wrapper:
            return _Wrapper("Fallback narrative")

        monkeypatch.setattr("nyx.nyx_agent_sdk.ModelSettings", _StubModelSettings, raising=False)
        monkeypatch.setattr("nyx.nyx_agent_sdk.RunConfig", _StubRunConfig, raising=False)
        monkeypatch.setattr(
            "nyx.nyx_agent_sdk.RunContextWrapper", _StubRunContextWrapper, raising=False
        )
        monkeypatch.setattr("nyx.nyx_agent_sdk.nyx_main_agent", object(), raising=False)
        monkeypatch.setattr("nyx.nyx_agent_sdk._execute_llm", _fake_execute, raising=False)
        monkeypatch.setattr("nyx.nyx_agent_sdk.assess_action_feasibility", None, raising=False)

        metadata = {
            "_deadline": time.monotonic() + 5,
            "player_name": "Hero",
            "nyx_display_name": "Narrator",
        }

        response = await sdk._fallback_direct_run(
            message="Hello Nyx",
            conversation_id="42",
            user_id="7",
            metadata=metadata,
            trace_id="trace",
            t0=0.0,
        )

        assert response.narrative == "Fallback narrative"
        assert len(recorder.calls) == 2
        first, second = recorder.calls
        assert first["turn"]["sender"] == "Hero"
        assert first["turn"]["content"] == "Hello Nyx"
        assert second["turn"]["sender"] == "Narrator"
        assert second["turn"]["content"] == "Fallback narrative"

    asyncio.run(_run())
