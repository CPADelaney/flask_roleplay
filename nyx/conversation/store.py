"""Async helpers for storing and retrieving OpenAI Agent conversation turns.

The store centralises access to the ``openai_threads`` mapping table as well as
basic message persistence helpers used by Nyx orchestration flows.

Typical usage::

    store = ConversationStore()

    async def sync_turn():
        binding = await store.get_or_create_thread_id(user_id=1, conversation_id=42)
        await store.append_turn(
            user_id=binding.user_id,
            conversation_id=binding.conversation_id,
            turn={"sender": "user", "content": "Hello Nyx!"},
        )
        history = await store.fetch_recent_turns(
            user_id=binding.user_id,
            conversation_id=binding.conversation_id,
            limit=4,
        )
        # ``history`` now contains the last four normalized conversation turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from db.connection import get_db_connection_context
from utils.conversation_history import _normalize_turn

logger = logging.getLogger(__name__)

_CHANNEL_OPENAI = "openai"


@dataclass(frozen=True)
class ThreadBinding:
    """Represents the mapping between a local conversation and a remote thread."""

    user_id: int
    conversation_id: int
    channel: str
    remote_id: str
    created_at: datetime

    @classmethod
    def from_mapping(cls, entry: Mapping[str, Any]) -> "ThreadBinding":
        """Build a :class:`ThreadBinding` from a database record."""

        return cls(
            user_id=int(entry["user_id"]),
            conversation_id=int(entry["conversation_id"]),
            channel=str(entry["channel"]),
            remote_id=str(entry["remote_id"]),
            created_at=entry["created_at"],
        )


class ConversationStore:
    """Persist conversation state for Nyx's OpenAI Agent integrations."""

    def __init__(self, *, channel: str = _CHANNEL_OPENAI) -> None:
        self._channel = channel

    async def get_or_create_thread_id(
        self, *, user_id: int, conversation_id: int
    ) -> ThreadBinding:
        """Return the remote thread binding, creating one when necessary."""

        try:
            user_id_int = int(user_id)
            conversation_id_int = int(conversation_id)
        except (TypeError, ValueError):
            raise ValueError("user_id and conversation_id must be coercible to int")

        binding = await self._fetch_binding(
            user_id=user_id_int, conversation_id=conversation_id_int
        )
        if binding is not None:
            return binding

        remote_id = await self._create_remote_thread(
            user_id=user_id_int, conversation_id=conversation_id_int
        )
        return await self._upsert_binding(
            user_id=user_id_int,
            conversation_id=conversation_id_int,
            remote_id=remote_id,
        )

    async def append_turn(
        self,
        *,
        user_id: int,
        conversation_id: int,
        turn: Mapping[str, Any],
    ) -> None:
        """Persist a single conversation turn for later retrieval."""

        try:
            conversation_id_int = int(conversation_id)
        except (TypeError, ValueError):
            logger.warning(
                "append_turn received invalid conversation_id: user_id=%s conversation_id=%r",
                user_id,
                conversation_id,
            )
            return

        normalized = _normalize_turn(turn)
        sender = normalized.get("sender")
        content = normalized.get("content")
        if not sender or content is None:
            logger.debug(
                "append_turn skipping turn without sender/content: user_id=%s conversation_id=%s turn=%r",
                user_id,
                conversation_id_int,
                turn,
            )
            return

        query = """
            INSERT INTO messages (conversation_id, sender, content)
            VALUES ($1, $2, $3)
        """

        try:
            async with get_db_connection_context() as conn:
                await conn.execute(query, conversation_id_int, sender, content)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to append turn: user_id=%s conversation_id=%s",  # pragma: no cover
                user_id,
                conversation_id_int,
            )

    async def fetch_recent_turns(
        self,
        *,
        user_id: int,
        conversation_id: int,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent turns for the given conversation."""

        if limit <= 0:
            return []

        try:
            user_id_int = int(user_id)
            conversation_id_int = int(conversation_id)
        except (TypeError, ValueError):
            logger.warning(
                "fetch_recent_turns received invalid identifiers: user_id=%r conversation_id=%r",
                user_id,
                conversation_id,
            )
            return []

        query = """
            SELECT m.sender, m.content
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = $1
              AND m.conversation_id = $2
            ORDER BY m.created_at DESC
            LIMIT $3
        """

        rows: Sequence[Mapping[str, Any]] = []
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, user_id_int, conversation_id_int, limit)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to fetch recent turns: user_id=%s conversation_id=%s",  # pragma: no cover
                user_id_int,
                conversation_id_int,
            )
            return []

        ordered: Sequence[Mapping[str, Any]] = list(reversed(rows))
        turns: List[Dict[str, Any]] = []
        for entry in ordered:
            if not isinstance(entry, Mapping):
                continue
            normalized = _normalize_turn(entry)
            if normalized:
                turns.append(normalized)
        return turns

    async def _fetch_binding(
        self, *, user_id: int, conversation_id: int
    ) -> Optional[ThreadBinding]:
        query = """
            SELECT user_id, conversation_id, channel, remote_id, created_at
            FROM openai_threads
            WHERE user_id = $1 AND conversation_id = $2 AND channel = $3
        """

        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(query, user_id, conversation_id, self._channel)
        if row:
            return ThreadBinding.from_mapping(row)
        return None

    async def _upsert_binding(
        self,
        *,
        user_id: int,
        conversation_id: int,
        remote_id: str,
    ) -> ThreadBinding:
        query = """
            INSERT INTO openai_threads (user_id, conversation_id, channel, remote_id)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, conversation_id, channel) DO UPDATE
            SET remote_id = EXCLUDED.remote_id
            RETURNING user_id, conversation_id, channel, remote_id, created_at
        """

        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                query,
                user_id,
                conversation_id,
                self._channel,
                remote_id,
            )

        if row:
            return ThreadBinding.from_mapping(row)

        # Fallback: attempt to fetch again so callers always receive a binding.
        binding = await self._fetch_binding(
            user_id=user_id, conversation_id=conversation_id
        )
        if binding is None:  # pragma: no cover - unexpected path
            raise RuntimeError("Failed to persist OpenAI thread binding")
        return binding

    async def _create_remote_thread(self, *, user_id: int, conversation_id: int) -> str:
        try:
            from nyx.gateway import openai_agents
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenAI Agents gateway is unavailable") from exc

        remote_id = await openai_agents.create_thread_for_conversation(
            user_id=user_id, conversation_id=conversation_id
        )
        return str(remote_id)


__all__ = ["ConversationStore", "ThreadBinding"]
