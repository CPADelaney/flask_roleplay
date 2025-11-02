import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List

import pytest
import typing
from typing_extensions import TypedDict

os.environ.setdefault("OPENAI_API_KEY", "test-key")
typing.TypedDict = TypedDict  # Backport for pydantic v2 on Python 3.11

from nyx.conflict.workers import llm_route_scene_subsystems
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMOperation, LLMRequest
from nyx.nyx_agent import orchestrator as nyx_orchestrator
from nyx.nyx_agent_sdk import NyxAgentSDK, NyxResponse, NyxSDKConfig
from story_agent import story_orchestrator
from nyx.tasks.background import canon_tasks, conflict_tasks


class AttrDict(dict):
    """Dictionary that exposes keys as attributes for compatibility helpers."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - helper
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


class GatewayRecorder:
    """Collect calls made through the gateway during tests."""

    def __init__(self) -> None:
        self.calls: List[SimpleNamespace] = []

    def record(self, kind: str, request: LLMRequest, result: llm_gateway.LLMResult) -> None:
        self.calls.append(SimpleNamespace(kind=kind, request=request, result=result))

    def for_stage(self, stage: str) -> List[SimpleNamespace]:
        return [call for call in self.calls if (call.request.metadata or {}).get("stage") == stage]


@pytest.fixture
def gateway_stub(monkeypatch: pytest.MonkeyPatch) -> GatewayRecorder:
    recorder = GatewayRecorder()

    async def fake_execute(request: LLMRequest) -> llm_gateway.LLMResult:
        metadata = dict(request.metadata or {})
        stage = metadata.get("stage") or metadata.get("operation") or "default"
        payload = AttrDict(
            {
                "stage": stage,
                "prompt": request.prompt,
                "metadata": metadata,
                "data": [f"{stage}_subsystem"],
                "narrative": f"{stage}:{request.prompt}",
            }
        )
        result = llm_gateway.LLMResult(
            text=f"{stage}:{request.prompt}",
            raw=payload,
            agent_name=getattr(request.agent, "name", None),
            metadata=metadata,
        )
        recorder.record("execute", request, result)
        return result

    async def fake_execute_stream(request: LLMRequest):  # pragma: no cover - not exercised yet
        metadata = dict(request.metadata or {})
        stage = metadata.get("stage") or metadata.get("operation") or "default"
        payload = AttrDict(
            {
                "stage": stage,
                "prompt": request.prompt,
                "metadata": metadata,
                "chunks": [f"{stage}:{request.prompt}"],
            }
        )
        result = llm_gateway.LLMResult(
            text=f"{stage}:{request.prompt}",
            raw=payload,
            agent_name=getattr(request.agent, "name", None),
            metadata=metadata,
        )
        recorder.record("execute_stream", request, result)
        yield result

    monkeypatch.setattr(llm_gateway, "execute", fake_execute)
    monkeypatch.setattr(llm_gateway, "execute_stream", fake_execute_stream)
    monkeypatch.setattr(nyx_orchestrator, "execute", fake_execute)
    monkeypatch.setattr(story_orchestrator, "execute", fake_execute)
    monkeypatch.setattr("nyx.conflict.workers.compute.execute", fake_execute)
    monkeypatch.setattr(canon_tasks, "execute", fake_execute)

    import agents

    async def forbid_runner(cls, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - ensures no calls
        raise AssertionError("Runner.run should not be invoked directly")

    monkeypatch.setattr(agents.Runner, "run", classmethod(forbid_runner))
    return recorder


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class EntryPointCase:
    name: str
    runner: Callable[[pytest.MonkeyPatch, GatewayRecorder], Awaitable[Dict[str, Any]]]
    checker: Callable[[Dict[str, Any], GatewayRecorder], None]


async def _run_nyx_orchestrator(_: pytest.MonkeyPatch, __: GatewayRecorder) -> Dict[str, Any]:
    agent = SimpleNamespace(name="nyx-main", instructions="", model=object())
    context = SimpleNamespace()
    result = await nyx_orchestrator.run_agent_safely(agent, "Perform action", context)
    return {"result": result}


def _check_nyx_orchestrator(data: Dict[str, Any], recorder: GatewayRecorder) -> None:
    call = recorder.calls[-1]
    assert call.kind == "execute"
    assert data["result"] is call.result.raw
    assert call.request.prompt == "Perform action"


async def _run_sdk_stream(monkeypatch: pytest.MonkeyPatch, _: GatewayRecorder) -> Dict[str, Any]:
    config = NyxSDKConfig(
        pre_moderate_input=False,
        post_moderate_output=False,
        rate_limit_per_conversation=False,
        retry_on_failure=False,
        enable_response_filter=False,
        streaming_chunk_size=5,
    )
    sdk = NyxAgentSDK(config=config)

    async def fake_process(self, message: str, conversation_id: str, user_id: str, metadata: Dict[str, Any] | None = None) -> NyxResponse:
        request = LLMRequest(
            prompt=message,
            agent=SimpleNamespace(name="sdk-agent", model=object()),
            metadata={
                "operation": LLMOperation.ORCHESTRATION.value,
                "stage": "sdk_stream",
            },
        )
        result = await llm_gateway.execute(request)
        raw = result.raw
        return NyxResponse(
            narrative=raw["narrative"],
            metadata={"gateway_raw": raw},
            success=True,
            trace_id="stream-trace",
            processing_time=0.5,
        )

    monkeypatch.setattr(NyxAgentSDK, "process_user_input", fake_process, raising=False)

    events: List[Dict[str, Any]] = []
    async for event in sdk.stream_user_input("hello world", "123", "456"):
        events.append(event)
    return {"events": events}


def _check_sdk_stream(data: Dict[str, Any], recorder: GatewayRecorder) -> None:
    events = data["events"]
    assert events[0]["type"] == "start"
    token_events = [event for event in events if event["type"] == "token"]
    assert token_events, "expected at least one token event"
    combined = "".join(event["text"] for event in token_events)
    stage_calls = recorder.for_stage("sdk_stream")
    assert len(stage_calls) == 1
    raw = stage_calls[0].result.raw
    assert combined == raw["narrative"]
    end_event = events[-1]
    assert end_event["type"] == "end"
    assert end_event["metadata"]["gateway_raw"] is raw


async def _run_story_orchestrator(monkeypatch: pytest.MonkeyPatch, _: GatewayRecorder) -> Dict[str, Any]:
    stub_agent = SimpleNamespace(name="daily-agent", model=object(), instructions="daily task")
    stub_module = ModuleType("story_agent.daily_task_generator")
    stub_module.DailyTaskGenerator = stub_agent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "story_agent.daily_task_generator", stub_module)
    packet = SimpleNamespace(daily_task=None)
    await story_orchestrator.StoryOrchestrator._attach_daily_task(SimpleNamespace(), packet)
    return {"packet": packet}


def _check_story_orchestrator(data: Dict[str, Any], recorder: GatewayRecorder) -> None:
    calls = recorder.for_stage("daily_task")
    assert len(calls) == 1
    raw = calls[0].result.raw
    assert data["packet"].daily_task == raw["data"]


async def _run_conflict_orchestrator(_: pytest.MonkeyPatch, __: GatewayRecorder) -> Dict[str, Any]:
    synthesizer = SimpleNamespace(
        _orchestrator=SimpleNamespace(name="conflict-route", model=object()),
        synthesize_scene_router_prompt=lambda scene_context: "route this scene",
    )
    subsystems = await llm_route_scene_subsystems(
        synthesizer,
        {"scene": "test"},
        timeout=1.0,
    )
    return {"subsystems": subsystems}


def _check_conflict_orchestrator(data: Dict[str, Any], recorder: GatewayRecorder) -> None:
    calls = recorder.for_stage("conflict_route")
    assert len(calls) == 1
    assert data["subsystems"] == ["conflict_route_subsystem"]


async def _run_canon_lore(monkeypatch: pytest.MonkeyPatch, _: GatewayRecorder) -> Dict[str, Any]:
    class DummySubsystem:
        def __init__(self) -> None:
            self.precedent_analyzer = SimpleNamespace(name="precedent", model=object())
            self.calls: List[Dict[str, Any]] = []

        async def update_compliance_suggestions(
            self,
            cache_id: int,
            suggestions: List[str],
            status: str,
            error: str | None = None,
        ) -> None:
            self.calls.append(
                {
                    "cache_id": cache_id,
                    "suggestions": suggestions,
                    "status": status,
                    "error": error,
                }
            )

    subsystem = DummySubsystem()

    @asynccontextmanager
    async def fake_db_context():
        class DummyConnection:
            async def fetch(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
                return []

        yield DummyConnection()

    monkeypatch.setattr(canon_tasks, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(
        canon_tasks,
        "extract_runner_response",
        lambda response: json.dumps({"suggestions": ["respect canon"]}),
    )

    await canon_tasks._generate_compliance_suggestions(
        subsystem,
        cache_id=99,
        conflict_type="duel",
        conflict_context={"intensity": "high"},
        matching_event_ids=[],
    )
    return {"subsystem": subsystem}


def _check_canon_lore(data: Dict[str, Any], recorder: GatewayRecorder) -> None:
    calls = recorder.for_stage("canon_compliance")
    assert len(calls) == 1
    subsystem = data["subsystem"]
    assert subsystem.calls == [
        {
            "cache_id": 99,
            "suggestions": ["respect canon"],
            "status": "ready",
            "error": None,
        }
    ]


ENTRY_CASES = [
    EntryPointCase("nyx_orchestrator", _run_nyx_orchestrator, _check_nyx_orchestrator),
    EntryPointCase("nyx_sdk_stream", _run_sdk_stream, _check_sdk_stream),
    EntryPointCase("story_orchestrator", _run_story_orchestrator, _check_story_orchestrator),
    EntryPointCase("conflict_orchestrator", _run_conflict_orchestrator, _check_conflict_orchestrator),
    EntryPointCase("canon_lore", _run_canon_lore, _check_canon_lore),
]


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("case", ENTRY_CASES, ids=lambda case: case.name)
async def test_gateway_entrypoints_use_llm_gateway(case: EntryPointCase, monkeypatch: pytest.MonkeyPatch, gateway_stub: GatewayRecorder) -> None:
    data = await case.runner(monkeypatch, gateway_stub)
    case.checker(data, gateway_stub)
    assert any(call.kind == "execute" for call in gateway_stub.calls), "LLM gateway was not invoked"
