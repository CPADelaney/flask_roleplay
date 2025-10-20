"""Unit tests for chatgpt_integration Nyx normalization helpers."""

from __future__ import annotations

import os
import sys
import asyncio
import types
from pathlib import Path
from contextlib import asynccontextmanager

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

setattr(nyx_orchestrator_stub, "prepare_context", None)
setattr(nyx_agent_sdk_stub, "process_user_input", None)
async def _default_process_universal_update(**kwargs):
    return {}
setattr(universal_updater_stub, "process_universal_update", _default_process_universal_update)

sys.modules.setdefault("nyx", nyx_stub)
sys.modules.setdefault("nyx.core", nyx_core_stub)
sys.modules.setdefault("nyx.core.orchestrator", nyx_orchestrator_stub)
sys.modules.setdefault("nyx.nyx_agent_sdk", nyx_agent_sdk_stub)
sys.modules.setdefault("logic.universal_updater_agent", universal_updater_stub)

setattr(nyx_stub, "core", nyx_core_stub)
setattr(nyx_core_stub, "orchestrator", nyx_orchestrator_stub)
setattr(nyx_stub, "nyx_agent_sdk", nyx_agent_sdk_stub)

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
