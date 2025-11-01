"""In-process registry for tracking user memory version numbers."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Dict, Tuple, Optional

logger = logging.getLogger("memory_version_registry")

# Internal state guarded by a process-wide lock. The key is a tuple of
# (user_id, conversation_id) so we can differentiate between concurrent
# conversations if callers provide that context. For callers that only pass
# the user identifier we normalize to ``(user_id, None)``.
_memory_versions: Dict[Tuple[int, Optional[int]], int] = {}
_registry_lock = Lock()


def _normalize_key(user_id: int, conversation_id: Optional[int] = None) -> Tuple[int, Optional[int]]:
    if user_id is None:
        raise ValueError("user_id is required to bump the memory version")
    return user_id, conversation_id


def get_memory_version(user_id: int, conversation_id: Optional[int] = None) -> int:
    """Return the current memory version for the given scope."""
    key = _normalize_key(user_id, conversation_id)
    with _registry_lock:
        return _memory_versions.get(key, 0)


def bump_memory_version(user_id: int, conversation_id: Optional[int] = None) -> int:
    """Increment and return the memory version for the given scope."""
    key = _normalize_key(user_id, conversation_id)
    with _registry_lock:
        next_version = _memory_versions.get(key, 0) + 1
        _memory_versions[key] = next_version
    logger.debug(
        "Bumped memory version", extra={"user_id": user_id, "conversation_id": conversation_id, "version": next_version}
    )
    return next_version


__all__ = ["bump_memory_version", "get_memory_version"]
