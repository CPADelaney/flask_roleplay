import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_generate_text_completion_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    async def _prepare_context(system_prompt: str, user_prompt: str) -> str:
        return system_prompt

    nyx_module = ModuleType("nyx")
    core_module = ModuleType("nyx.core")
    orchestrator_module = ModuleType("nyx.core.orchestrator")
    orchestrator_module.prepare_context = _prepare_context
    nyx_module.core = core_module
    core_module.orchestrator = orchestrator_module

    monkeypatch.setitem(sys.modules, "nyx", nyx_module)
    monkeypatch.setitem(sys.modules, "nyx.core", core_module)
    monkeypatch.setitem(sys.modules, "nyx.core.orchestrator", orchestrator_module)
    module_name = "logic.chatgpt_integration"
    if module_name in sys.modules:
        chatgpt_integration = importlib.reload(sys.modules[module_name])
    else:
        chatgpt_integration = importlib.import_module(module_name)

    class DummyResponsesClient:
        def __init__(self, text: str):
            self.text = text

        async def create(self, **kwargs):
            chunk = SimpleNamespace(text=self.text, type="output_text", annotations=[])
            message = SimpleNamespace(content=[chunk], role="assistant")
            return SimpleNamespace(output_text="", output=[message])

    dummy_client = SimpleNamespace(responses=DummyResponsesClient("Successful fallback"))
    chatgpt_integration._client_manager._async_client = dummy_client
    monkeypatch.setattr(chatgpt_integration, "PREPARE_CONTEXT_AVAILABLE", False)

    result = asyncio.run(
        chatgpt_integration.generate_text_completion(
            system_prompt="system",
            user_prompt="user",
        )
    )

    assert result == "Successful fallback"
