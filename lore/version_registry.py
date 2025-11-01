"""Lore version registry utilities.

Provides a lightweight in-process registry that tracks the current
lore schema/version for cache coordination across subsystems.
"""
from __future__ import annotations

import re
import threading
from typing import Optional

_LOCK = threading.Lock()
_LORE_VERSION: int = 1
_SUFFIX_PATTERN = re.compile(r":l\d+$")


def get_lore_version() -> int:
    """Return the current lore version value."""
    return _LORE_VERSION


def set_lore_version(version: int) -> int:
    """Set the lore version explicitly (primarily for tests)."""
    global _LORE_VERSION
    with _LOCK:
        _LORE_VERSION = int(version)
        return _LORE_VERSION


def bump_lore_version() -> int:
    """Increment the lore version and return the new value."""
    global _LORE_VERSION
    with _LOCK:
        _LORE_VERSION += 1
        return _LORE_VERSION


def with_lore_version_suffix(base_key: Optional[str]) -> str:
    """Append (or replace) the lore-version suffix on the provided cache key."""
    key = str(base_key or "")
    key = _SUFFIX_PATTERN.sub("", key)
    return f"{key}:l{get_lore_version()}"


__all__ = [
    "get_lore_version",
    "set_lore_version",
    "bump_lore_version",
    "with_lore_version_suffix",
]
