"""Signal definitions for the conflict system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from nyx.nyx_agent.context import SceneScope
else:  # pragma: no cover - runtime fallback to avoid circular import
    SceneScope = Any


class ConflictSignalType(Enum):
    """Enumerates the types of signals that can be emitted in the conflict system."""

    SCENE_ENTERED = "scene_entered"
    TIME_TICK = "time_tick"
    PLAYER_ACTION = "player_action"
    FACT_BECAME_PUBLIC = "fact_became_public"
    RELATIONSHIP_CHANGE = "relationship_change"
    CONFLICT_CREATED = "conflict_created"
    CONFLICT_UPDATED = "conflict_updated"
    CONFLICT_RESOLVED = "conflict_resolved"


@dataclass(slots=True)
class ConflictSignal:
    """Standard payload for conflict-related signals."""

    type: ConflictSignalType
    user_id: int
    conversation_id: int
    scene_scope: SceneScope | None = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


__all__ = ["ConflictSignalType", "ConflictSignal"]
