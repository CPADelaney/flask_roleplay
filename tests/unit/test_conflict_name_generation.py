import asyncio
import importlib
import sys

from types import ModuleType

import pytest


@pytest.fixture
def conflict_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module_name = "logic.conflict_system.conflict_integration"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_generate_conflict_name_uses_llm_response(monkeypatch, conflict_module):
    async def fake_completion(*_, **__):
        return "Embers of Defiance"

    fallback_called = {"value": False}

    def fake_fallback(self, conflict_type, context):
        fallback_called["value"] = True
        return "Fallback Name"

    fake_chatgpt_module = ModuleType("logic.chatgpt_integration")
    fake_chatgpt_module.generate_text_completion = fake_completion
    monkeypatch.setitem(sys.modules, "logic.chatgpt_integration", fake_chatgpt_module)
    monkeypatch.setattr(conflict_module.ConflictSystemIntegration, "_generate_conflict_name_fallback", fake_fallback)

    integration = conflict_module.ConflictSystemIntegration.__new__(conflict_module.ConflictSystemIntegration)

    name = asyncio.run(
        conflict_module.ConflictSystemIntegration._generate_conflict_name(
            integration,
            "major",
            {"intensity": "medium", "description": "A simmering rivalry"},
        )
    )

    assert name == "Embers of Defiance"
    assert not fallback_called["value"]
    assert name not in {"Power Struggle", "Rising Tensions", "Clash of Wills"}
