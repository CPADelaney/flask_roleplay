"""Helpers for invoking the OpenAI Responses API with Nyx conversations."""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, TYPE_CHECKING

from nyx.conversation.store import ConversationStore

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from openai import AsyncOpenAI as _AsyncOpenAI
else:  # pragma: no cover - runtime fallback for typing
    _AsyncOpenAI = Any  # type: ignore[assignment]

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency setup
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency setup
    AsyncOpenAI = None  # type: ignore[assignment]

ROLE_MAP: Mapping[str, str] = {
    "user": "user",
    "player": "user",
    "assistant": "assistant",
    "nyx": "assistant",
    "narrator": "assistant",
    "gm": "assistant",
    "system": "system",
}


_STORE = ConversationStore()
_CLIENT: Optional[_AsyncOpenAI]
if AsyncOpenAI is None:  # pragma: no cover - optional dependency setup
    _CLIENT = None
else:
    try:
        _CLIENT = AsyncOpenAI()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to instantiate AsyncOpenAI client")
        _CLIENT = None


def _ensure_client(client: Optional[_AsyncOpenAI]) -> _AsyncOpenAI:
    if client is not None:
        return client
    if _CLIENT is None:
        raise RuntimeError("AsyncOpenAI client is unavailable")
    return _CLIENT


def _normalize_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _resolve_role(sender: Optional[str]) -> str:
    if not sender:
        return "user"
    key = sender.lower()
    return ROLE_MAP.get(key, "user")


def _build_responses_input(
    *,
    turns: Sequence[Mapping[str, Any]],
    system_prompt: Optional[str] = None,
) -> List[MutableMapping[str, Any]]:
    """Build the stateless Responses API input payload from stored turns."""

    messages: List[MutableMapping[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": _normalize_text(system_prompt),
                    }
                ],
            }
        )

    for turn in turns:
        sender = turn.get("sender")
        content = turn.get("content")
        if content in (None, ""):
            continue
        messages.append(
            {
                "role": _resolve_role(str(sender) if sender is not None else None),
                "content": [
                    {
                        "type": "text",
                        "text": _normalize_text(content),
                    }
                ],
            }
        )

    return messages


async def run_response_for_conversation(
    *,
    user_id: int,
    conversation_id: int,
    model: str,
    metadata: Optional[Mapping[str, Any]] = None,
    system_prompt: Optional[str] = None,
    context_prompt: Optional[str] = None,
    latest_user_input: Optional[str] = None,
    latest_user_role: str = "user",
    history_limit: int = 8,
    client: Optional[_AsyncOpenAI] = None,
    store: Optional[ConversationStore] = None,
    request_overrides: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Run the Responses API for a persisted conversation."""

    if not model:
        raise ValueError("model must be provided")

    openai_client = _ensure_client(client)
    conversation_store = store or _STORE

    turns: Sequence[Mapping[str, Any]] = await conversation_store.fetch_recent_turns(
        user_id=user_id, conversation_id=conversation_id, limit=history_limit
    )
    responses_input = _build_responses_input(
        turns=turns, system_prompt=system_prompt
    )

    if context_prompt:
        context_message: MutableMapping[str, Any] = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": _normalize_text(context_prompt),
                }
            ],
        }
        insert_index = 0
        if responses_input and responses_input[0].get("role") == "system":
            insert_index = 1
        responses_input.insert(insert_index, context_message)

    if latest_user_input:
        responses_input.append(
            {
                "role": _resolve_role(latest_user_role),
                "content": [
                    {
                        "type": "text",
                        "text": _normalize_text(latest_user_input),
                    }
                ],
            }
        )

    request: MutableMapping[str, Any] = {
        "model": model,
        "input": responses_input,
    }
    if request_overrides:
        request.update(dict(request_overrides))
    if metadata:
        sanitized = {
            str(key): str(value)
            for key, value in metadata.items()
            if value is not None
        }
        existing = request.get("metadata")
        if isinstance(existing, Mapping):
            merged = {str(k): str(v) for k, v in existing.items() if v is not None}
            merged.update(sanitized)
            request["metadata"] = merged
        else:
            request["metadata"] = sanitized

    try:
        response = await openai_client.responses.create(**request)
    except Exception:
        logger.exception(
            "Failed to run responses for user_id=%s conversation_id=%s model=%s",
            user_id,
            conversation_id,
            model,
        )
        raise

    return response


__all__ = ["ROLE_MAP", "_build_responses_input", "run_response_for_conversation"]
