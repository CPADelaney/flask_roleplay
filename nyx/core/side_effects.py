"""Typed side-effect events emitted by the synchronous response path."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


class WorldDelta(BaseModel):
    """World-state mutations captured during the sync turn."""

    turn_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    deltas: Dict[str, Any] = Field(default_factory=dict)
    incoming_world_version: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryEvent(BaseModel):
    """Memory payload awaiting durable storage and embedding."""

    turn_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    text: str
    refs: Dict[str, Any] = Field(default_factory=dict)


class ConflictEvent(BaseModel):
    """Conflict synthesizer inputs produced during the turn."""

    turn_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    conflict_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class NPCStimulus(BaseModel):
    """NPC adaptation stimuli for post-turn processing."""

    turn_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    npcs: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)


class LoreHint(BaseModel):
    """Scene or regional lore bundle hints to precompute."""

    turn_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    scene_id: Optional[str] = None
    region_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


SideEffect = WorldDelta | MemoryEvent | ConflictEvent | NPCStimulus | LoreHint


def group_side_effects(events: Sequence[SideEffect]) -> Dict[str, Dict[str, Any]]:
    """Group side effects by type for transport to Celery."""

    grouped: Dict[str, Dict[str, Any]] = {}
    for event in events:
        payload = event.model_dump(exclude_none=True)
        if isinstance(event, WorldDelta):
            grouped["world"] = payload
        elif isinstance(event, MemoryEvent):
            grouped["memory"] = payload
        elif isinstance(event, ConflictEvent):
            grouped["conflict"] = payload
        elif isinstance(event, NPCStimulus):
            grouped["npc"] = payload
        elif isinstance(event, LoreHint):
            grouped["lore"] = payload
    return grouped


__all__ = [
    "WorldDelta",
    "MemoryEvent",
    "ConflictEvent",
    "NPCStimulus",
    "LoreHint",
    "SideEffect",
    "group_side_effects",
]
