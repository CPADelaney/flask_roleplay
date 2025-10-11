"""Lightweight server wrapper around the OpenAI Chat Completions API.

This module intentionally keeps the surface area small so that tests can
exercise the streaming logic in isolation.  The real project wires the
``RoleplayChatServer`` into Socket.IO handlers that expect an async iterator
producing OpenAI "Responses" streaming events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Optional


class RoleplayChatServer:
    """Facade around an OpenAI "Responses" client for roleplay chats."""

    def __init__(self, client) -> None:
        self._client = client

    async def respond(
        self,
        *,
        model: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream model output back to callers.

        ``openai.responses.create`` returns an awaitable ``AsyncStream`` when
        ``stream=True`` is provided.  The previous implementation forgot to
        ``await`` that coroutine which meant ``async for`` was iterating over
        the coroutine object instead of the resolved stream.  The behaviour
        differed between OpenAI client versions and could raise ``TypeError``
        when the object was not directly iterable.  Awaiting the coroutine (or
        using the context-manager helper as we do here) ensures we always
        receive the actual stream before iterating.
        """

        conversation_ref = None
        if metadata is not None:
            conversation_ref = (
                metadata.get("openai_conversation_id")
                or metadata.get("conversation")
            )

        stream_kwargs = dict(kwargs)
        if conversation_ref and "conversation" not in stream_kwargs:
            stream_kwargs["conversation"] = conversation_ref

        async with self._client.responses.stream(  # type: ignore[call-arg]
            model=model,
            input=input,
            metadata=metadata,
            **stream_kwargs,
        ) as stream:
            async for event in stream:
                yield event
