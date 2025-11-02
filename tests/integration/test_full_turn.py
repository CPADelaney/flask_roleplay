import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import pytest


async def _exercise_full_turn(monkeypatch):
    # Enable all rollout flags
    flag_names = [
        "NYX_FLAG_LLM_GATEWAY",
        "NYX_FLAG_OUTBOX",
        "NYX_FLAG_VERSIONED_CACHE",
        "NYX_FLAG_CONFLICT_FSM",
        "NYX_FLAG_DOMAIN_EVENTS",
        "NYX_FLAG_OUTPUT_EVALS",
    ]
    for name in flag_names:
        monkeypatch.setenv(name, "on")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import nyx.config.flags as flags

    importlib.reload(flags)

    # Ensure TypedDict imports resolve on Python < 3.12
    import typing
    from typing_extensions import TypedDict as _CompatTypedDict

    monkeypatch.setattr(typing, "TypedDict", _CompatTypedDict, raising=False)

    # Provide a stub orchestrator module before importing the SDK
    orchestrator_module = types.ModuleType("nyx.nyx_agent.orchestrator")
    default_payload = {
        "success": True,
        "response": "A gentle breeze whispers through the atrium.",
        "metadata": {
            "world": {"weather": "breeze"},
            "turn_id": "turn-001",
            "scene_scope": {"location_id": 7, "npc_ids": [101], "nation_ids": [55]},
            "locationInfo": {"id": 7, "display": "Atrium", "slug": "atrium"},
            "context_stats": {"npc_names": ["Aurelian"]},
            "conflict_event": {"conflict_id": "conf-xyz", "active": True},
        },
        "trace_id": "trace-test",
        "processing_time": 0.05,
    }

    async def fake_process_user_input(user_id, conversation_id, user_input, context_data=None):
        return dict(default_payload)

    def fake_preserve_hydrated_location(target, location):
        if isinstance(target, dict) and location is not None:
            target.setdefault("current_location", location)

    orchestrator_module.process_user_input = fake_process_user_input
    orchestrator_module._preserve_hydrated_location = fake_preserve_hydrated_location
    orchestrator_module.generate_reflection = fake_process_user_input
    orchestrator_module.manage_scenario = fake_process_user_input
    orchestrator_module.manage_relationships = fake_process_user_input
    orchestrator_module.store_messages = fake_process_user_input
    orchestrator_module.run_agent_safely = fake_process_user_input
    orchestrator_module.run_agent_with_error_handling = fake_process_user_input
    orchestrator_module.decide_image_generation_standalone = fake_process_user_input
    sys.modules["nyx.nyx_agent.orchestrator"] = orchestrator_module

    import nyx.nyx_agent_sdk as sdk

    importlib.reload(sdk)

    # Stub snapshot store backend (avoid Redis) and persistence
    monkeypatch.setattr(sdk.ConversationSnapshotStore, "_build_client", lambda self: None)

    async def fake_persist(*_args, **_kwargs):
        return None

    monkeypatch.setattr(sdk, "persist_canonical_snapshot", fake_persist)
    monkeypatch.setattr(
        sdk,
        "build_canonical_snapshot_payload",
        lambda snapshot: {"snapshot": dict(snapshot)},
    )

    # Prevent optional background hooks from running
    monkeypatch.setattr(sdk, "_enqueue_task", None)
    monkeypatch.setattr(sdk, "_log_perf", None)

    # Stub Runner.run to satisfy contract even if unused
    try:
        import agents
    except ImportError:  # pragma: no cover - agent SDK not installed in some test envs
        agents = None
    else:
        async def fake_runner_run(*_args, **_kwargs):
            return SimpleNamespace(final_output="stubbed")

        monkeypatch.setattr(agents.Runner, "run", fake_runner_run)

    recorded_payloads = []
    processed_effects = []
    legacy_task_calls = []
    memory_written = []
    fsm_state = {}

    def handle_effects(side_effects):
        for key, payload in (side_effects or {}).items():
            if not payload:
                continue
            if key == "memory":
                memory_written.append(payload.get("text"))
            elif key == "conflict":
                fsm_state[payload.get("conflict_id")] = "INTEGRATED"
            processed_effects.append((key, payload))

    class FakeDispatch:
        def apply_async(self, args=None, kwargs=None, **_opts):
            payload = dict(kwargs or {}).get("payload") or {}
            recorded_payloads.append(payload)
            handle_effects(payload.get("side_effects"))

    fake_dispatch = FakeDispatch()
    monkeypatch.setattr(sdk, "post_turn_dispatch", fake_dispatch)

    # Patch Celery app send_task for legacy path
    import nyx.tasks.base as tasks_base

    reverse_map = {cfg["task"]: key for key, cfg in sdk._LEGACY_SIDE_EFFECT_TASKS.items()}

    def fake_send_task(task_name, kwargs=None, **options):
        payload = (kwargs or {}).get("payload")
        effect_key = reverse_map.get(task_name)
        if effect_key:
            handle_effects({effect_key: payload})
        legacy_task_calls.append((task_name, payload, options))

    monkeypatch.setattr(tasks_base.app, "send_task", fake_send_task)

    sdk_instance = sdk.NyxAgentSDK()
    response = await sdk_instance.process_user_input(
        message="hello",
        conversation_id="123",
        user_id="42",
        metadata={},
    )

    assert response.success
    assert "breeze" in response.narrative
    assert recorded_payloads, "Outbox payloads should be recorded when flag is on"
    assert memory_written and any("breeze" in text for text in memory_written)
    assert fsm_state.get("conf-xyz") == "INTEGRATED"
    assert processed_effects, "Side effects should be processed when flags are enabled"

    # Legacy path when outbox flag disabled
    monkeypatch.setenv("NYX_FLAG_OUTBOX", "off")
    recorded_payloads.clear()
    processed_effects.clear()
    legacy_task_calls.clear()
    memory_written.clear()
    fsm_state.clear()

    response_legacy = await sdk_instance.process_user_input(
        message="hello again",
        conversation_id="123",
        user_id="42",
        metadata={},
    )

    assert response_legacy.success
    assert not recorded_payloads, "Outbox dispatcher should not be used when flag is off"
    assert legacy_task_calls, "Legacy send_task path should process side effects"
    assert memory_written, "Memory side effects should still be processed"
    assert fsm_state.get("conf-xyz") == "INTEGRATED"


def test_full_user_turn_with_flags(monkeypatch):
    asyncio.run(_exercise_full_turn(monkeypatch))
