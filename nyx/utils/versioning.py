"""Simple optimistic concurrency helpers."""

from __future__ import annotations


def reject_if_stale(current: int, incoming: int) -> bool:
    """Return True if the incoming version is fresh enough to apply."""

    try:
        current_val = int(current)
    except (TypeError, ValueError):
        current_val = 0
    try:
        incoming_val = int(incoming)
    except (TypeError, ValueError):
        incoming_val = 0
    return incoming_val >= current_val


__all__ = ["reject_if_stale"]
