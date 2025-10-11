import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_chatgpt_module(monkeypatch):
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
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_generate_text_completion_fallback(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    class DummyResponsesClient:
        def __init__(self, text: str):
            self.text = text
            self.last_input = None
            self.last_instructions = None

        async def create(self, **kwargs):
            self.last_input = kwargs.get("input")
            self.last_instructions = kwargs.get("instructions")
            text_value = SimpleNamespace(value=self.text)
            chunk = SimpleNamespace(
                text=text_value,
                type="output_text",
                annotations=[],
            )
            content_collection = SimpleNamespace(data=[chunk])
            message = SimpleNamespace(content=content_collection, role="assistant")
            output_collection = SimpleNamespace(data=[message])
            return SimpleNamespace(output_text="", output=output_collection)

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
    assert dummy_client.responses.last_instructions == "system"
    assert dummy_client.responses.last_input == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "user"}],
        }
    ]


def test_generate_text_completion_retries_when_empty(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    class DummyResponsesClient:
        def __init__(self):
            self.calls = 0
            self.last_input = None
            self.last_instructions = None

        async def create(self, **kwargs):
            self.calls += 1
            self.last_input = kwargs.get("input")
            self.last_instructions = kwargs.get("instructions")
            if self.calls == 1:
                return SimpleNamespace(output_text="   ", output=SimpleNamespace(data=[]))

            text_value = SimpleNamespace(value="Second attempt")
            chunk = SimpleNamespace(
                text=text_value,
                type="output_text",
                annotations=[],
            )
            content_collection = SimpleNamespace(data=[chunk])
            message = SimpleNamespace(content=content_collection, role="assistant")
            output_collection = SimpleNamespace(data=[message])
            return SimpleNamespace(output_text="", output=output_collection)

    dummy_client = SimpleNamespace(responses=DummyResponsesClient())
    chatgpt_integration._client_manager._async_client = dummy_client
    monkeypatch.setattr(chatgpt_integration, "PREPARE_CONTEXT_AVAILABLE", False)

    result = asyncio.run(
        chatgpt_integration.generate_text_completion(
            system_prompt="system",
            user_prompt="user",
        )
    )

    assert result == "Second attempt"
    assert dummy_client.responses.calls == 2
    assert dummy_client.responses.last_instructions == "system"
    assert dummy_client.responses.last_input == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "user"}],
        }
    ]


def test_generate_text_completion_raises_on_double_empty(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    class DummyResponsesClient:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):  # noqa: ARG002 - interface parity
            self.calls += 1
            empty_output = SimpleNamespace(data=[])
            return SimpleNamespace(output_text="", output=empty_output)

    dummy_client = SimpleNamespace(responses=DummyResponsesClient())
    chatgpt_integration._client_manager._async_client = dummy_client
    monkeypatch.setattr(chatgpt_integration, "PREPARE_CONTEXT_AVAILABLE", False)

    with pytest.raises(chatgpt_integration.EmptyLLMOutputError):
        asyncio.run(
            chatgpt_integration.generate_text_completion(
                system_prompt="system",
                user_prompt="user",
            )
        )


def test_generate_text_completion_expands_token_limit(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    class DummyResponsesClient:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                incomplete_details = SimpleNamespace(reason="max_output_tokens")
                return SimpleNamespace(
                    status="incomplete",
                    incomplete_details=incomplete_details,
                    output_text="",
                    output=SimpleNamespace(data=[]),
                )

            text_value = SimpleNamespace(value="Expanded response")
            chunk = SimpleNamespace(
                text=text_value,
                type="output_text",
                annotations=[],
            )
            content_collection = SimpleNamespace(data=[chunk])
            message = SimpleNamespace(content=content_collection, role="assistant")
            output_collection = SimpleNamespace(data=[message])
            return SimpleNamespace(
                status="completed",
                incomplete_details=None,
                output_text="",
                output=output_collection,
            )

    dummy_client = SimpleNamespace(responses=DummyResponsesClient())
    chatgpt_integration._client_manager._async_client = dummy_client
    monkeypatch.setattr(chatgpt_integration, "PREPARE_CONTEXT_AVAILABLE", False)

    result = asyncio.run(
        chatgpt_integration.generate_text_completion(
            system_prompt="system",
            user_prompt="user",
            max_tokens=500,
            model="gpt-4o",
        )
    )

    assert result == "Expanded response"
    assert len(dummy_client.responses.calls) == 2
    first_call, second_call = dummy_client.responses.calls
    assert first_call["max_output_tokens"] == 500
    expected_second = min(
        500 + max(500, 512),
        chatgpt_integration.MODEL_TOKEN_LIMITS["gpt-4o"],
    )
    assert second_call["max_output_tokens"] == expected_second


def test_generate_text_completion_handles_multiple_incompletes(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    class DummyResponsesClient:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) < 3:
                incomplete_details = SimpleNamespace(reason="max_output_tokens")
                return SimpleNamespace(
                    status="incomplete",
                    incomplete_details=incomplete_details,
                    output_text="",
                    output=SimpleNamespace(data=[]),
                )

            text_value = SimpleNamespace(value="Final after expansion")
            chunk = SimpleNamespace(
                text=text_value,
                type="output_text",
                annotations=[],
            )
            content_collection = SimpleNamespace(data=[chunk])
            message = SimpleNamespace(content=content_collection, role="assistant")
            output_collection = SimpleNamespace(data=[message])
            return SimpleNamespace(
                status="completed",
                incomplete_details=None,
                output_text="",
                output=output_collection,
            )

    dummy_client = SimpleNamespace(responses=DummyResponsesClient())
    chatgpt_integration._client_manager._async_client = dummy_client
    monkeypatch.setattr(chatgpt_integration, "PREPARE_CONTEXT_AVAILABLE", False)

    result = asyncio.run(
        chatgpt_integration.generate_text_completion(
            system_prompt="system",
            user_prompt="user",
            max_tokens=256,
            model="gpt-4o",
        )
    )

    assert result == "Final after expansion"
    assert len(dummy_client.responses.calls) == 3
    first_call, second_call, third_call = dummy_client.responses.calls
    assert first_call["max_output_tokens"] == 256
    assert second_call["max_output_tokens"] > first_call["max_output_tokens"]
    assert third_call["max_output_tokens"] > second_call["max_output_tokens"]


def test_generate_text_completion_raises_when_model_limit_reached(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    class DummyResponsesClient:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):  # noqa: ARG002 - interface parity
            self.calls.append(kwargs)
            incomplete_details = SimpleNamespace(reason="max_output_tokens")
            return SimpleNamespace(
                status="incomplete",
                incomplete_details=incomplete_details,
                output_text="",
                output=SimpleNamespace(data=[]),
            )

    dummy_client = SimpleNamespace(responses=DummyResponsesClient())
    chatgpt_integration._client_manager._async_client = dummy_client
    monkeypatch.setattr(chatgpt_integration, "PREPARE_CONTEXT_AVAILABLE", False)
    monkeypatch.setitem(chatgpt_integration.MODEL_TOKEN_LIMITS, "gpt-5-nano", 600)

    with pytest.raises(chatgpt_integration.EmptyLLMOutputError) as excinfo:
        asyncio.run(
            chatgpt_integration.generate_text_completion(
                system_prompt="system",
                user_prompt="user",
                max_tokens=500,
                model="gpt-5-nano",
            )
        )

    assert len(dummy_client.responses.calls) == 2
    first_call, second_call = dummy_client.responses.calls
    assert first_call["max_output_tokens"] == 500
    assert second_call["max_output_tokens"] == 600
    assert excinfo.value.diagnostics["max_output_tokens"] == 600
    assert excinfo.value.diagnostics["model_limit"] == 600


def test_extract_output_text_surfaces_refusal(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    refusal_chunk = SimpleNamespace(type="refusal", refusal="I can't help with that.")
    message = SimpleNamespace(content=[refusal_chunk])
    output_collection = SimpleNamespace(data=[message])
    response = SimpleNamespace(output_text="", output=output_collection)

    text, is_refusal = chatgpt_integration._extract_output_text(response)

    assert text == "I can't help with that."
    assert is_refusal is True


def test_extract_output_text_reads_model_dump(monkeypatch):
    chatgpt_integration = _load_chatgpt_module(monkeypatch)

    class DummyResponse:
        output_text = ""
        output = None

        @staticmethod
        def model_dump():
            return {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Payload text",
                            }
                        ]
                    }
                ]
            }

    text, is_refusal = chatgpt_integration._extract_output_text(DummyResponse())

    assert text == "Payload text"
    assert is_refusal is False
