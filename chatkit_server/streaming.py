"""Helpers for streaming ChatKit responses to Socket.IO clients."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

DeltaCallback = Callable[[str], Awaitable[None]]


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        # Convert lightweight objects into dictionaries for easier access
        data: Dict[str, Any] = {}
        for attr in dir(value):
            if attr.startswith("_"):
                continue
            try:
                attr_value = getattr(value, attr)
            except AttributeError:
                continue
            if callable(attr_value):
                continue
            data[attr] = attr_value
        return data
    return None


def extract_delta_text(delta: Any) -> str:
    """Best-effort extraction of text from a ChatKit delta payload."""

    if delta is None:
        return ""

    if isinstance(delta, str):
        return delta

    mapping = _as_mapping(delta)
    if mapping:
        for key in ("text", "value", "content"):
            text_value = mapping.get(key)
            if isinstance(text_value, str):
                return text_value
            if isinstance(text_value, Sequence) and not isinstance(text_value, (str, bytes)):
                parts = [part for part in text_value if isinstance(part, str)]
                if parts:
                    return "".join(parts)
        # Some SDKs nest the text under delta["delta"]["text"]
        nested = mapping.get("delta")
        if nested:
            nested_text = extract_delta_text(nested)
            if nested_text:
                return nested_text

    if hasattr(delta, "text"):
        text_attr = getattr(delta, "text")
        if isinstance(text_attr, str):
            return text_attr

    if hasattr(delta, "value"):
        value_attr = getattr(delta, "value")
        if isinstance(value_attr, str):
            return value_attr

    return str(delta)


def extract_response_text(response: Any) -> str:
    """Extract the aggregated text from a ChatKit final response payload."""

    mapping = _as_mapping(response)
    if not mapping:
        return str(response) if response is not None else ""

    text_parts: List[str] = []
    outputs = mapping.get("output") or mapping.get("outputs") or []

    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        for item in outputs:
            item_map = _as_mapping(item)
            if not item_map:
                continue
            contents = item_map.get("content")
            if isinstance(contents, Sequence) and not isinstance(contents, (str, bytes)):
                for part in contents:
                    part_map = _as_mapping(part)
                    if not part_map:
                        continue
                    if part_map.get("type") in {"output_text", "text"} and isinstance(part_map.get("text"), str):
                        text_parts.append(part_map["text"])
            elif isinstance(contents, str):
                text_parts.append(contents)

    if not text_parts:
        fallback = mapping.get("text") or mapping.get("content") or mapping.get("response")
        if isinstance(fallback, str):
            return fallback

    return "".join(text_parts)


def extract_thread_metadata(response: Any) -> Dict[str, Optional[Any]]:
    """Return key identifiers from the ChatKit final response payload."""

    mapping = _as_mapping(response) or {}

    thread_info = mapping.get("thread")
    thread_map = _as_mapping(thread_info) if thread_info else None

    thread_id = mapping.get("thread_id")
    if not thread_id and thread_map:
        thread_id = thread_map.get("id")

    run_info = mapping.get("run")
    run_map = _as_mapping(run_info) if run_info else None

    run_id = mapping.get("run_id")
    if not run_id and run_map:
        run_id = run_map.get("id")

    assistant_id = mapping.get("assistant_id")
    if not assistant_id and run_map:
        assistant_id = run_map.get("assistant_id")

    conversation_info = mapping.get("conversation")
    conversation_map = _as_mapping(conversation_info) if conversation_info else None

    conversation_id = mapping.get("conversation_id")
    if not conversation_id and conversation_map:
        conversation_id = conversation_map.get("id")

    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "assistant_id": assistant_id,
        "status": mapping.get("status"),
        "response_id": mapping.get("id") or mapping.get("response_id"),
        "model": mapping.get("model"),
        "conversation_id": conversation_id,
    }


def format_messages_for_chatkit(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Convert legacy chat messages into Responses API input format."""

    formatted: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, str):
            formatted.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
        elif isinstance(content, Sequence):
            # Assume already formatted content list
            formatted.append({"role": role, "content": list(content)})
        else:
            formatted.append({
                "role": role,
                "content": [{"type": "text", "text": str(content)}],
            })

    return formatted


async def stream_chatkit_tokens(
    server,
    *,
    model: str,
    input_data: Any,
    metadata: Optional[Mapping[str, Any]] = None,
    on_delta: DeltaCallback,
) -> tuple[str, Optional[Any]]:
    """Stream ChatKit output via the provided callback and return the final payload."""

    tokens: List[str] = []
    final_payload: Optional[Any] = None

    async for event in server.respond(model=model, input=input_data, metadata=metadata):
        event_type = _get_attr_or_key(event, "type")

        if event_type == "response.output_text.delta":
            delta = _get_attr_or_key(event, "delta")
            text = extract_delta_text(delta)
            if text:
                tokens.append(text)
                await on_delta(text)
        elif event_type == "response.completed":
            final_payload = _get_attr_or_key(event, "response")
        elif event_type == "response.error":
            error = _get_attr_or_key(event, "error")
            raise RuntimeError(f"ChatKit streaming error: {error}")

    return "".join(tokens), final_payload
