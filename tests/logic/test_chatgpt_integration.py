"""Unit tests for chatgpt_integration Nyx normalization helpers."""

from __future__ import annotations

import os
import sys
import asyncio
import types
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

# Ensure API key is available before importing the integration module which
# constructs the OpenAI client manager at import time.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SENTENCE_TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Provide lightweight Nyx stubs so importing chatgpt_integration does not pull
# the full Nyx runtime (which loads HuggingFace models).
nyx_stub = types.ModuleType("nyx")
nyx_stub.__path__ = []
nyx_core_stub = types.ModuleType("nyx.core")
nyx_core_stub.__path__ = []
nyx_orchestrator_stub = types.ModuleType("nyx.core.orchestrator")
nyx_agent_sdk_stub = types.ModuleType("nyx.nyx_agent_sdk")
universal_updater_stub = types.ModuleType("logic.universal_updater_agent")
nyx_conversation_stub = types.ModuleType("nyx.conversation")
nyx_conversation_store_stub = types.ModuleType("nyx.conversation.store")
nyx_gateway_stub = types.ModuleType("nyx.gateway")
nyx_gateway_responses_stub = types.ModuleType("nyx.gateway.openai_responses")


class _StubConversationStore:
    async def fetch_recent_turns(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def append_turn(self, *args: Any, **kwargs: Any) -> None:
        return None


nyx_conversation_store_stub.ConversationStore = _StubConversationStore
nyx_conversation_stub.store = nyx_conversation_store_stub


async def _stub_run_response_for_conversation(**_: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output=[],
        output_text="",
        usage=types.SimpleNamespace(input_tokens=0, output_tokens=0),
    )


nyx_gateway_responses_stub.run_response_for_conversation = _stub_run_response_for_conversation
nyx_gateway_stub.openai_responses = nyx_gateway_responses_stub

setattr(nyx_orchestrator_stub, "prepare_context", None)
setattr(nyx_agent_sdk_stub, "process_user_input", None)
async def _default_process_universal_update(**kwargs):
    return {}
setattr(universal_updater_stub, "process_universal_update", _default_process_universal_update)

sys.modules.setdefault("nyx", nyx_stub)
sys.modules.setdefault("nyx.core", nyx_core_stub)
sys.modules.setdefault("nyx.core.orchestrator", nyx_orchestrator_stub)
sys.modules.setdefault("nyx.nyx_agent_sdk", nyx_agent_sdk_stub)
sys.modules.setdefault("nyx.conversation", nyx_conversation_stub)
sys.modules.setdefault("nyx.conversation.store", nyx_conversation_store_stub)
sys.modules.setdefault("nyx.gateway", nyx_gateway_stub)
sys.modules.setdefault("nyx.gateway.openai_responses", nyx_gateway_responses_stub)
sys.modules.setdefault("logic.universal_updater_agent", universal_updater_stub)

setattr(nyx_stub, "core", nyx_core_stub)
setattr(nyx_core_stub, "orchestrator", nyx_orchestrator_stub)
setattr(nyx_stub, "nyx_agent_sdk", nyx_agent_sdk_stub)
setattr(nyx_stub, "conversation", nyx_conversation_stub)
setattr(nyx_stub, "gateway", nyx_gateway_stub)

from logic import chatgpt_integration as chatgpt  # noqa: E402


@asynccontextmanager
async def _fake_db_context():
    class _FakeConn:
        async def fetchrow(self, *args, **kwargs):
            return {"user_id": 99}

        async def fetchval(self, *args, **kwargs):
            return None

    yield _FakeConn()


async def _run_nyx_string_response_test(monkeypatch):
    """Nyx string responses should normalize to function-call payloads."""

    narrative_text = "Nyx string narrative"
    process_calls: dict[str, dict] = {}

    monkeypatch.setattr(chatgpt, "get_db_connection_context", _fake_db_context)

    async def _fake_check_preset_story(conversation_id: int):
        return None

    monkeypatch.setattr(chatgpt, "check_preset_story", _fake_check_preset_story)

    async def _fake_process_universal_update(**kwargs):
        process_calls["kwargs"] = kwargs
        return {"success": True}

    universal_updater_stub.process_universal_update = _fake_process_universal_update

    async def _fake_nyx_process_input(**kwargs):
        return {
            "success": True,
            "response": narrative_text,
            "performance": {"tokens_used": 7},
        }

    chatgpt.nyx_process_input = _fake_nyx_process_input

    result = await chatgpt.get_chatgpt_response(
        conversation_id=1,
        aggregator_text="context",
        user_input="hello",
        reflection_enabled=False,
        use_nyx_integration=True,
    )

    assert result["type"] == "function_call"
    assert result["function_name"] == "apply_universal_update"
    assert result["function_args"]["narrative"] == narrative_text
    assert result["function_args"]["roleplay_updates"] == []
    assert result["tokens_used"] == 7

    assert "kwargs" in process_calls
    assert process_calls["kwargs"]["narrative"] == narrative_text


def test_get_chatgpt_response_nyx_string_narrative(monkeypatch):
    asyncio.run(_run_nyx_string_response_test(monkeypatch))


async def _run_force_json_response_headroom_test(monkeypatch):
    """Forcing JSON responses should expand the output token budget."""

    captured: dict[str, Any] = {}
    fake_client = object()

    monkeypatch.setattr(chatgpt, "get_db_connection_context", _fake_db_context)
    async def _fake_check_preset_story(conversation_id):
        return None

    monkeypatch.setattr(chatgpt, "check_preset_story", _fake_check_preset_story)

    monkeypatch.setattr(
        chatgpt.OpenAIClientManager,
        "async_client",
        property(lambda self: fake_client),
    )

    class _StubStore:
        def __init__(self) -> None:
            self.append_calls: list[dict[str, Any]] = []

        async def fetch_recent_turns(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            captured["history_args"] = {"args": args, "kwargs": kwargs}
            return []

        async def append_turn(self, *args: Any, **kwargs: Any) -> None:
            self.append_calls.append({"args": args, "kwargs": kwargs})

    store = _StubStore()
    monkeypatch.setattr(chatgpt, "_conversation_store", store)

    async def _fake_run_response_for_conversation(**kwargs: Any):
        captured["run_kwargs"] = kwargs
        return types.SimpleNamespace(
            output=[],
            output_text="{}",
            usage=types.SimpleNamespace(input_tokens=100, output_tokens=50),
        )

    monkeypatch.setattr(chatgpt, "run_response_for_conversation", _fake_run_response_for_conversation)

    result = await chatgpt.get_chatgpt_response(
        conversation_id=1,
        aggregator_text="context",
        user_input="hello",
        reflection_enabled=False,
        use_nyx_integration=False,
        force_json_response=True,
    )

    overrides = captured["run_kwargs"]["request_overrides"]
    assert overrides["max_output_tokens"] == chatgpt._compute_forced_json_max_tokens("gpt-5-nano", 2048)
    assert "tool_choice" not in overrides
    assert captured["run_kwargs"]["model"] == "gpt-5-nano"
    assert captured["run_kwargs"]["latest_user_input"] == "hello"
    assert captured["run_kwargs"]["context_prompt"] == "context"
    expected_budget = chatgpt._compute_forced_json_max_tokens("gpt-5-nano", 2048)
    assert overrides["max_output_tokens"] == expected_budget
    assert result["type"] == "input_text"
    assert len(store.append_calls) == 2


def test_force_json_response_headroom(monkeypatch):
    asyncio.run(_run_force_json_response_headroom_test(monkeypatch))


def test_convert_response_to_array_format_preserves_pre_normalized_lists():
    roleplay_updates = [{"key": "CurrentYear", "value": 2088}]
    chase_schedule = [
        {
            "key": "Monday",
            "value": {
                "Morning": "Train",
                "Afternoon": "Work",
                "Evening": "Patrol",
                "Night": "Rest",
            },
        }
    ]
    stat_updates = [{"key": "Courage", "value": 5}]
    npc_schedule = [{"key": "Tuesday", "value": {"Morning": "Plan"}}]
    npc_schedule_updates = [{"key": "Wednesday", "value": {"Morning": "Scout"}}]

    response = {
        "narrative": "Pre-normalized payload",
        "universal_updates": {
            "roleplay_updates": roleplay_updates,
            "ChaseSchedule": chase_schedule,
            "character_stat_updates": {
                "player_name": "Chase",
                "stats": stat_updates,
            },
            "npc_creations": [
                {
                    "npc_id": 1,
                    "schedule": npc_schedule,
                    "schedule_updates": npc_schedule_updates,
                }
            ],
        },
    }

    function_args = chatgpt.convert_response_to_array_format(response)

    assert function_args["roleplay_updates"] is roleplay_updates
    assert function_args["ChaseSchedule"] is chase_schedule
    assert (
        function_args["character_stat_updates"]["stats"] is stat_updates
    )
    assert function_args["npc_creations"][0]["schedule"] is npc_schedule
    assert (
        function_args["npc_creations"][0]["schedule_updates"] is npc_schedule_updates
    )
