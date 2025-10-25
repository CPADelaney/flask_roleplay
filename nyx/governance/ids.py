"""Utilities for consistent agent identifier formatting and normalization."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple


def _stringify_kind(kind: str | Enum) -> str:
    """Return a normalized string representation for an agent kind."""
    if isinstance(kind, Enum):
        kind_value = kind.value
    else:
        kind_value = str(kind)
    kind_str = kind_value.strip()
    if not kind_str:
        raise ValueError("Agent kind must be a non-empty string")
    return kind_str


def format_agent_id(kind: str | Enum, conversation_id: int | str) -> str:
    """Create a deterministic agent identifier for a conversation-scoped agent."""
    conversation_str = str(conversation_id).strip()
    if not conversation_str:
        raise ValueError("Conversation id must be provided")
    return f"{_stringify_kind(kind)}_{conversation_str}"


def parse_agent_id(agent_id: str) -> Optional[Tuple[str, str]]:
    """Attempt to split an agent identifier into (kind, suffix).

    The suffix usually corresponds to a conversation id. Legacy identifiers may
    use different separators; we therefore inspect multiple delimiter options.
    """
    for separator in ("_", ":", "-"):
        if separator in agent_id:
            prefix, suffix = agent_id.rsplit(separator, 1)
            prefix = prefix.strip()
            suffix = suffix.strip()
            if prefix and suffix:
                return prefix, suffix
    return None


def legacy_agent_ids(kind: str | Enum, conversation_id: int | str) -> Tuple[str, ...]:
    """Return legacy identifier variants for backfilling persisted records."""
    normalized = format_agent_id(kind, conversation_id)
    kind_str = _stringify_kind(kind)
    conversation_str = str(conversation_id).strip()

    candidates = {
        kind_str,
        f"{kind_str}:{conversation_str}",
        f"{kind_str}-{conversation_str}",
    }
    candidates.discard(normalized)
    return tuple(candidate for candidate in candidates if candidate)
