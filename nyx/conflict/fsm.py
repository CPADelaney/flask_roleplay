"""State machine utilities for conflict resolution."""

from __future__ import annotations

from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from nyx.conflict.models import ConflictResolution


class Status(str, Enum):
    DRAFT = "DRAFT"
    EVAL = "EVAL"
    CANON = "CANON"
    INTEGRATING = "INTEGRATING"
    INTEGRATED = "INTEGRATED"
    FAILED = "FAILED"


_ALLOWED_TRANSITIONS: dict[Status, set[Status]] = {
    Status.DRAFT: {Status.EVAL, Status.FAILED},
    Status.EVAL: {Status.CANON, Status.FAILED},
    Status.CANON: {Status.INTEGRATING, Status.FAILED},
    Status.INTEGRATING: {Status.INTEGRATED, Status.FAILED},
    Status.INTEGRATED: set(),
    Status.FAILED: set(),
}

_MUTABLE_FIELDS = {"draft_text", "eval_score", "eval_notes", "integrated_changes"}


def transition(session: Session, cr: ConflictResolution, to: Status, **kwargs: Any) -> ConflictResolution:
    """Transition the conflict resolution to a new state, validating invariants."""

    current = Status(cr.status)
    target = Status(to)

    if target != current:
        allowed = _ALLOWED_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise ValueError(f"Cannot transition from {current.value} to {target.value}")
        cr.status = target.value

    for key in kwargs:
        if key not in _MUTABLE_FIELDS:
            raise ValueError(f"Unsupported field update for conflict resolution: {key}")

    for field, value in kwargs.items():
        setattr(cr, field, value)

    session.flush()
    session.refresh(cr)
    return cr


__all__ = ["Status", "transition"]
