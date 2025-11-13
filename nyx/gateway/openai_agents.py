"""Helpers for interacting with OpenAI's Agents API."""

from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Optional, TYPE_CHECKING

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


def _build_metadata(
    *,
    user_id: int,
    conversation_id: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> MutableMapping[str, str]:
    payload: MutableMapping[str, str] = {
        "user_id": str(user_id),
        "conversation_id": str(conversation_id),
    }
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            try:
                payload[str(key)] = str(value)
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "Skipping non-stringable metadata key=%r value=%r", key, value
                )
    return payload


async def create_thread_for_conversation(
    *,
    user_id: int,
    conversation_id: int,
    metadata: Optional[Mapping[str, Any]] = None,
    client: Optional[_AsyncOpenAI] = None,
) -> str:
    """Create a remote OpenAI thread for the given conversation."""

    openai_client = _ensure_client(client)
    request_metadata = _build_metadata(
        user_id=user_id, conversation_id=conversation_id, metadata=metadata
    )

    try:
        thread = await openai_client.beta.threads.create(metadata=request_metadata)
    except Exception:
        logger.exception(
            "Failed to create thread for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        raise

    thread_id = getattr(thread, "id", None)
    if not thread_id and isinstance(thread, Mapping):
        thread_id = thread.get("id")
    if not thread_id:
        raise RuntimeError("OpenAI thread response missing id")
    return str(thread_id)


async def run_agent_for_conversation(
    *,
    user_id: int,
    conversation_id: int,
    assistant_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
    system_prompt: Optional[str] = None,
    client: Optional[_AsyncOpenAI] = None,
    store: Optional[ConversationStore] = None,
) -> Any:
    """Run an OpenAI agent tied to a persisted conversation thread."""

    if not assistant_id:
        raise ValueError("assistant_id must be provided")

    openai_client = _ensure_client(client)
    conversation_store = store or _STORE
    binding = await conversation_store.get_or_create_thread_id(
        user_id=user_id, conversation_id=conversation_id
    )

    run_kwargs: dict[str, Any] = {"assistant_id": assistant_id}

    request_metadata = _build_metadata(
        user_id=user_id, conversation_id=conversation_id, metadata=metadata
    )
    if request_metadata:
        run_kwargs["metadata"] = request_metadata
    if system_prompt:
        run_kwargs["instructions"] = system_prompt

    try:
        run = await openai_client.beta.threads.runs.create(
            thread_id=binding.remote_id, **run_kwargs
        )
    except Exception:
        logger.exception(
            "Failed to run agent for user_id=%s conversation_id=%s assistant_id=%s",
            user_id,
            conversation_id,
            assistant_id,
        )
        raise

    return run


__all__ = ["create_thread_for_conversation", "run_agent_for_conversation"]
