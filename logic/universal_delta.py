"""Typed canonical delta schema for universal updates."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union
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


class PlayerMoveOperation(_OperationBase):
    """Record a player character moving to a new location."""

    type: Literal["player.move"] = "player.move"
    player_id: Optional[int] = Field(default=None, ge=1)
    location_slug: Optional[str] = Field(default=None, min_length=1)
    location_id: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _ensure_location(self) -> "PlayerMoveOperation":
        if self.location_slug is None and self.location_id is None:
            raise ValueError("player.move requires either location_slug or location_id")
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
    Union[
        NPCMoveOperation,
        PlayerMoveOperation,
        RelationshipBumpOperation,
        NarrativeAppendOperation,
    ],
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


_LOCATION_KEYS = (
    "CurrentLocation",
    "current_location",
    "currentLocation",
    "location",
    "Location",
)
_LOCATION_ID_KEYS = (
    "CurrentLocationId",
    "current_location_id",
    "currentLocationId",
    "location_id",
    "LocationId",
)


def _clean_location_slug(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, Mapping):
        for key in ("slug", "name", "label", "location"):
            candidate = value.get(key)
            cleaned = _clean_location_slug(candidate)
            if cleaned:
                return cleaned
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _coerce_location_id(value: Any) -> Optional[int]:
    if isinstance(value, Mapping):
        for key in ("id", "location_id", "pk"):
            coerced = _coerce_location_id(value.get(key))
            if coerced is not None:
                return coerced
        return None
    try:
        if value is None:
            return None
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _extract_location_from_mapping(mapping: Mapping[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    location_slug: Optional[str] = None
    location_id: Optional[int] = None

    for key in _LOCATION_KEYS:
        if location_slug:
            break
        if key in mapping:
            location_slug = _clean_location_slug(mapping.get(key))

    for key in _LOCATION_ID_KEYS:
        if location_id is not None:
            break
        if key in mapping:
            location_id = _coerce_location_id(mapping.get(key))

    # Nested composite value
    composite = mapping.get("location") if "location" in mapping else None
    if composite is not None:
        if location_slug is None:
            location_slug = _clean_location_slug(composite)
        if location_id is None:
            location_id = _coerce_location_id(composite)

    return location_slug, location_id


def _extract_location_from_current_scene(value: Any) -> Tuple[Optional[str], Optional[int]]:
    """Derive location details from a ``CurrentScene`` payload."""

    if value is None:
        return None, None

    scene_mapping: Optional[Mapping[str, Any]] = None

    if isinstance(value, Mapping):
        scene_mapping = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return _clean_location_slug(stripped), _coerce_location_id(stripped)
        if isinstance(parsed, Mapping):
            scene_mapping = parsed
        else:
            return _clean_location_slug(parsed), _coerce_location_id(parsed)
    else:
        return _clean_location_slug(value), _coerce_location_id(value)

    slug: Optional[str] = None
    location_id: Optional[int] = None

    location_payload = scene_mapping.get("location") if scene_mapping else None
    if location_payload is not None:
        slug = _clean_location_slug(location_payload)
        location_id = _coerce_location_id(location_payload)

    if slug is None or location_id is None:
        extra_slug, extra_id = _extract_location_from_mapping(scene_mapping or {})
        if slug is None:
            slug = extra_slug
        if location_id is None:
            location_id = extra_id

    return slug, location_id


def _extract_player_location(payload: Mapping[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    location_slug: Optional[str] = None
    location_id: Optional[int] = None

    roleplay_updates = payload.get("roleplay_updates")
    if isinstance(roleplay_updates, Mapping):
        slug, loc_id = _extract_location_from_mapping(roleplay_updates)
        location_slug = location_slug or slug
        location_id = location_id or loc_id
        if not any(key in roleplay_updates for key in _LOCATION_KEYS):
            for scene_key in ("CurrentScene", "currentScene"):
                if scene_key in roleplay_updates:
                    scene_slug, scene_id = _extract_location_from_current_scene(
                        roleplay_updates.get(scene_key)
                    )
                    if location_slug is None and scene_slug:
                        location_slug = scene_slug
                    if location_id is None and scene_id is not None:
                        location_id = scene_id
    elif isinstance(roleplay_updates, Sequence):
        flattened: Dict[str, Any] = {}
        for item in roleplay_updates:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key") or item.get("name") or item.get("field")
            if key is None:
                continue
            flattened[str(key)] = item.get("value")
        if flattened:
            slug, loc_id = _extract_location_from_mapping(flattened)
            location_slug = location_slug or slug
            location_id = location_id or loc_id
            if not any(key in flattened for key in _LOCATION_KEYS):
                for scene_key in ("CurrentScene", "currentScene"):
                    if scene_key in flattened:
                        scene_slug, scene_id = _extract_location_from_current_scene(
                            flattened.get(scene_key)
                        )
                        if location_slug is None and scene_slug:
                            location_slug = scene_slug
                        if location_id is None and scene_id is not None:
                            location_id = scene_id

    if location_slug is None or location_id is None:
        slug, loc_id = _extract_location_from_mapping(payload)
        if location_slug is None:
            location_slug = slug
        if location_id is None:
            location_id = loc_id
        if not any(key in payload for key in _LOCATION_KEYS):
            for scene_key in ("CurrentScene", "currentScene"):
                if scene_key in payload:
                    scene_slug, scene_id = _extract_location_from_current_scene(
                        payload.get(scene_key)
                    )
                    if location_slug is None and scene_slug:
                        location_slug = scene_slug
                    if location_id is None and scene_id is not None:
                        location_id = scene_id

    return location_slug, location_id


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

    def _relationship_delta_from_entry(entry: Mapping[str, Any]) -> Optional[int]:
        level_change = entry.get("level_change")
        if level_change is not None:
            try:
                coerced = int(round(float(level_change)))
            except (TypeError, ValueError) as exc:
                raise DeltaBuildError("relationship level_change must be numeric") from exc
            if coerced != 0:
                return coerced

        dimension_changes = entry.get("dimension_changes")
        if isinstance(dimension_changes, Mapping) and dimension_changes:
            total = 0.0
            seen_numeric = False
            for value in dimension_changes.values():
                try:
                    total += float(value)
                except (TypeError, ValueError):
                    continue
                else:
                    seen_numeric = True
            if seen_numeric:
                coerced = int(round(total))
                if coerced != 0:
                    return coerced
        return None

    def _append_relationship_operation(entry: Mapping[str, Any]) -> None:
        delta = _relationship_delta_from_entry(entry)
        if delta is None:
            return

        operations.append(
            RelationshipBumpOperation(
                source_type=str(entry.get("entity1_type", "")).strip() or "npc",
                source_id=int(entry.get("entity1_id")),
                target_type=str(entry.get("entity2_type", "")).strip() or "npc",
                target_id=int(entry.get("entity2_id")),
                delta=delta,
                context=entry.get("group_context"),
            )
        )

    for key in ("social_links", "relationship_updates"):
        entries = payload.get(key)
        if not entries:
            continue
        for link in entries or []:
            if not isinstance(link, Mapping):
                raise DeltaBuildError(f"{key} entries must be dictionaries")
            required_fields = (
                link.get("entity1_id"),
                link.get("entity2_id"),
            )
            if any(field in (None, "") for field in required_fields):
                raise DeltaBuildError(f"{key} entries must include entity identifiers")
            _append_relationship_operation(link)

    location_slug, location_id = _extract_player_location(payload)
    if location_slug or location_id:
        operations.append(
            PlayerMoveOperation(
                player_id=user_id,
                location_slug=location_slug,
                location_id=location_id,
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
