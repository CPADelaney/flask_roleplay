"""Typed canonical delta schema for universal updates."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from typing_extensions import Annotated, Literal


class DeltaBuildError(ValueError):
    """Raised when a legacy payload cannot be transformed into a delta."""


class _OperationBase(BaseModel):
    """Common base class for typed operations."""

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
    }

    type: str

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("operation type must be a non-empty string")
        if not re.fullmatch(r"[a-z]+\.[a-z_]+", value):
            raise ValueError(f"invalid operation type '{value}'")
        return value


class NPCMoveOperation(_OperationBase):
    """Move an NPC to a new canonical location."""

    type: Literal["npc.move"] = "npc.move"
    npc_id: int = Field(..., ge=1)
    location_slug: Optional[str] = Field(default=None, min_length=1)
    location_id: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _ensure_location(self) -> "NPCMoveOperation":
        if self.location_slug is None and self.location_id is None:
            raise ValueError("npc.move requires either location_slug or location_id")
        return self


class RelationshipBumpOperation(_OperationBase):
    """Increment a relationship score between two entities."""

    type: Literal["relationship.bump"] = "relationship.bump"
    source_type: str = Field(..., min_length=1)
    source_id: int = Field(..., ge=1)
    target_type: str = Field(..., min_length=1)
    target_id: int = Field(..., ge=1)
    delta: int
    context: Optional[str] = None

    @field_validator("delta")
    @classmethod
    def _validate_delta(cls, value: int) -> int:
        if value == 0:
            raise ValueError("relationship.bump delta cannot be zero")
        return value


class NarrativeAppendOperation(_OperationBase):
    """Append narrative text to the canon timeline."""

    type: Literal["narrative.append"] = "narrative.append"
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("narrative.append text cannot be empty")
        return cleaned


DeltaOperation = Annotated[
    Union[NPCMoveOperation, RelationshipBumpOperation, NarrativeAppendOperation],
    Field(discriminator="type"),
]


class UniversalDelta(BaseModel):
    """Validated payload sent to ``canon.apply_event``."""

    model_config = {"extra": "forbid"}

    user_id: int = Field(..., ge=1)
    conversation_id: int = Field(..., ge=1)
    request_id: UUID = Field(default_factory=uuid4)
    operations: List[DeltaOperation]
    actor: str = Field(default="universal_updater", min_length=1)
    version: int = Field(default=1, ge=1)

    @field_validator("operations")
    @classmethod
    def _ensure_operations(cls, value: List[DeltaOperation]) -> List[DeltaOperation]:
        if not value:
            raise ValueError("at least one operation must be present")
        return value

    @property
    def operation_count(self) -> int:
        return len(self.operations)


def _coerce_request_id(request_id: Optional[Union[str, UUID]]) -> Optional[UUID]:
    if request_id is None:
        return None
    if isinstance(request_id, UUID):
        return request_id
    try:
        return UUID(str(request_id))
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise DeltaBuildError("invalid request_id") from exc


def build_delta_from_legacy_payload(
    *,
    user_id: int,
    conversation_id: int,
    payload: Dict[str, Any],
    request_id: Optional[Union[str, UUID]] = None,
) -> UniversalDelta:
    """Construct a :class:`UniversalDelta` from legacy updater structures."""

    if not isinstance(payload, dict):
        raise DeltaBuildError("updates payload must be a dictionary")

    operations: List[DeltaOperation] = []

    narrative_text = payload.get("narrative")
    if isinstance(narrative_text, str) and narrative_text.strip():
        operations.append(NarrativeAppendOperation(text=narrative_text))

    for raw_update in payload.get("npc_updates", []) or []:
        if not isinstance(raw_update, dict):
            raise DeltaBuildError("npc_updates entries must be dictionaries")
        npc_id = raw_update.get("npc_id")
        if not npc_id:
            raise DeltaBuildError("npc_updates entries must include npc_id")
        location_slug = raw_update.get("current_location")
        location_id = raw_update.get("location_id")
        if location_slug or location_id:
            operations.append(
                NPCMoveOperation(
                    npc_id=int(npc_id),
                    location_slug=location_slug,
                    location_id=location_id,
                )
            )

    for link in payload.get("social_links", []) or []:
        if not isinstance(link, dict):
            raise DeltaBuildError("social_links entries must be dictionaries")
        delta = link.get("level_change")
        if delta in (None, 0):
            continue
        operations.append(
            RelationshipBumpOperation(
                source_type=str(link.get("entity1_type", "")).strip() or "npc",
                source_id=int(link.get("entity1_id")),
                target_type=str(link.get("entity2_type", "")).strip() or "npc",
                target_id=int(link.get("entity2_id")),
                delta=int(delta),
                context=link.get("group_context"),
            )
        )

    if not operations:
        raise DeltaBuildError("no canonical operations could be extracted from updates")

    coerced_request_id = _coerce_request_id(request_id or payload.get("request_id"))

    try:
        return UniversalDelta(
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=coerced_request_id or uuid4(),
            operations=operations,
        )
    except ValidationError as exc:  # pragma: no cover - bubbled up for clarity
        raise DeltaBuildError(str(exc)) from exc
