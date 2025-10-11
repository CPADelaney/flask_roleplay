from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatkit_server.streaming import (
    extract_response_text,
    extract_thread_metadata,
    format_messages_for_chatkit,
    stream_chatkit_tokens,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class DummyServer:
    def __init__(self, events):
        self._events = list(events)

    async def respond(self, **_: object):  # pragma: no cover - behaviour exercised via stream_chatkit_tokens
        for event in self._events:
            yield event


class DummySio:
    def __init__(self):
        self.events = []

    async def emit(self, event_name, data, room=None):
        self.events.append((event_name, data, room))


@pytest.mark.anyio("asyncio")
async def test_stream_chatkit_tokens_orders_events_before_completion():
    events = [
        {"type": "response.output_text.delta", "delta": {"text": "Hello "}},
        {"type": "response.output_text.delta", "delta": {"text": "world"}},
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "Hello world"},
                        ]
                    }
                ],
                "thread_id": "thread-1",
                "run_id": "run-1",
                "assistant_id": "assistant-1",
                "status": "completed",
                "id": "resp-1",
                "model": "gpt-test",
                "conversation": {"id": "conv-1"},
            },
        },
    ]
    server = DummyServer(events)
    sio = DummySio()
    tokens = []

    async def on_delta(token: str) -> None:
        tokens.append(token)
        await sio.emit("new_token", {"token": token})

    full_text, final_payload = await stream_chatkit_tokens(
        server,
        model="gpt-test",
        input_data=[],
        metadata=None,
        on_delta=on_delta,
    )

    await sio.emit("done", {"full_text": full_text})

    assert tokens == ["Hello ", "world"]
    assert full_text == "Hello world"
    assert [event for event, *_ in sio.events] == ["new_token", "new_token", "done"]

    metadata = extract_thread_metadata(final_payload)
    assert metadata["thread_id"] == "thread-1"
    assert metadata["run_id"] == "run-1"
    assert metadata["assistant_id"] == "assistant-1"
    assert metadata["conversation_id"] == "conv-1"

    assert extract_response_text(final_payload) == "Hello world"


def test_format_messages_for_chatkit_wraps_text():
    messages = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "World"},
    ]

    formatted = format_messages_for_chatkit(messages)

    assert formatted == [
        {"role": "system", "content": [{"type": "input_text", "text": "Hello"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "World"}]},
    ]
