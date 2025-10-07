"""Helper utilities for retrieving recent conversation history."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


Number = Union[int, str]


def _coerce_int(value: Number, *, name: str) -> int:
    """Safely coerce ``value`` to ``int`` for SQL parameters."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("Failed to coerce %s=%r to int for recent turn fetch", name, value)
        raise


def _normalize_turn(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a turn payload limited to ``sender``/``content`` keys."""

    sender = entry.get("sender")
    content = entry.get("content")
    normalized: Dict[str, Any] = {}
    if sender is not None:
        normalized["sender"] = sender
    if content is not None:
        normalized["content"] = content
    return normalized


async def fetch_recent_turns(
    user_id: Number,
    conversation_id: Number,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Fetch the latest ``limit`` conversation turns for Nyx context assembly.

    The turns are returned in chronological order and contain only the
    ``sender`` and ``content`` fields. Any missing or malformed rows are
    filtered out to protect downstream prompt assembly.
    """

    if limit <= 0:
        return []

    try:
        user_id_int = _coerce_int(user_id, name="user_id")
        conversation_id_int = _coerce_int(conversation_id, name="conversation_id")
    except Exception:
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

    try:
        async with get_db_connection_context() as conn:
            rows: Sequence[Mapping[str, Any]] = await conn.fetch(
                query,
                user_id_int,
                conversation_id_int,
                limit,
            )
    except Exception:
        logger.exception(
            "Failed to fetch recent turns for user_id=%s conversation_id=%s",
            user_id_int,
            conversation_id_int,
        )
        return []

    # Reverse once to chronological order (oldest → newest)
    ordered: Iterable[Mapping[str, Any]] = reversed(rows)

    turns: List[Dict[str, Any]] = []
    for row in ordered:
        if not isinstance(row, Mapping):
            continue
        normalized = _normalize_turn(row)
        if normalized:
            turns.append(normalized)

    return turns


__all__ = ["fetch_recent_turns"]

