# nyx/location/types.py
from __future__ import annotations

import os
import re
import string
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple, cast

DEFAULT_REALM = os.getenv("NYX_DEFAULT_REALM", "Prime Material")

Scope = Literal["real", "fictional"]

_DELIMITER_PATTERN = re.compile(r"\b(within|in)\b", re.IGNORECASE)
_PUNCTUATION_STRIP = string.whitespace + string.punctuation + "“”‘’–—"


def _clean_segment(value: str) -> str:
    return value.strip(_PUNCTUATION_STRIP)


def parse_location_chain(raw: str) -> Tuple[str, List[str]]:
    """Split a location label into a leaf name and parent chain.

    The input is trimmed of leading/trailing punctuation, then repeatedly
    partitioned on the last case-insensitive occurrence of "in"/"within". Each
    extracted segment becomes a parent, collected from nearest to farthest
    ancestor with duplicates removed case-insensitively. The remainder becomes
    the leaf display name. Strings without the delimiter are returned unchanged
    with an empty parent list.
    """

    if not isinstance(raw, str):
        raw = str(raw)

    working = _clean_segment(raw)
    if not working:
        return "", []

    parents: List[str] = []
    while True:
        matches = list(_DELIMITER_PATTERN.finditer(working))
        if not matches:
            break
        last_match = matches[-1]
        left = working[: last_match.start()]
        right = working[last_match.end() :]
        parent_candidate = _clean_segment(right)
        if not parent_candidate:
            break
        parents.append(parent_candidate)
        working = _clean_segment(left)
        if not working:
            break

    leaf = _clean_segment(working) if working else ""
    parents.reverse()

    deduped_parents: List[str] = []
    seen: set[str] = set()
    for parent in parents:
        key = parent.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped_parents.append(parent)

    if not leaf:
        leaf = _clean_segment(raw)

    return leaf, deduped_parents

PlaceLevel = Literal[
    "world",
    "country",
    "region",
    "state",
    "city",
    "district",
    "neighborhood",
    "venue",
    "virtual",
    "route",
    "unknown",
]


@dataclass
class Place:
    """Represents a concrete or fictional location entity."""

    name: str
    level: PlaceLevel
    key: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    address: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.meta, dict):
            self.meta = dict(self.meta or {})  # type: ignore[arg-type]

        source_label = self.meta.get("display_name") or self.name
        leaf, parent_chain = parse_location_chain(source_label or "")

        self.name = leaf
        self.meta["display_name"] = leaf

        if parent_chain:
            self.meta.setdefault("parents", parent_chain)


@dataclass
class PlaceEdge:
    """Relationship between two places in the location graph."""

    source: str
    target: str
    kind: Literal["contains", "near", "route", "alias", "anchors"]
    distance_km: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anchor:
    """Scope-aware anchor describing the player's current context."""

    scope: Scope
    focus: Optional[Place] = None
    label: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    primary_city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    world_name: Optional[str] = None
    hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    """A ranked candidate place with supporting metadata."""

    place: Place
    confidence: float = 0.0
    distance_km: Optional[float] = None
    edges: List[PlaceEdge] = field(default_factory=list)
    rationale: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


ResolveStatus = Literal[
    "exact",
    "nearby",
    "multiple",
    "ambiguous",
    "not_found",
    "travel_plan",
    "ask",
]

STATUS_EXACT: ResolveStatus = "exact"
STATUS_NEARBY: ResolveStatus = "nearby"
STATUS_MULTIPLE: ResolveStatus = "multiple"
STATUS_AMBIGUOUS: ResolveStatus = "ambiguous"
STATUS_NOT_FOUND: ResolveStatus = "not_found"
STATUS_TRAVEL_PLAN: ResolveStatus = "travel_plan"
STATUS_ASK: ResolveStatus = "ask"


@dataclass
class ResolveResult:
    """Result container for real and fictional location resolution."""

    status: ResolveStatus
    message: Optional[str] = None
    choices: List[str] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    anchor: Optional[Anchor] = None
    scope: Optional[Scope] = None
    errors: List[str] = field(default_factory=list)
    location: Optional["Location"] = None

    @property
    def canonical_ops(self) -> List[Dict[str, Any]]:
        """Backward compatible view of operations for legacy callers."""

        return self.operations

    @canonical_ops.setter
    def canonical_ops(self, value: Optional[List[Dict[str, Any]]]) -> None:
        self.operations = list(value or [])


@dataclass
class Location:
    """Represents a persisted location row with full world metadata."""

    user_id: int
    conversation_id: int
    location_name: str
    id: Optional[int] = None
    description: Optional[str] = None
    location_type: Optional[str] = None
    parent_location: Optional[str] = None
    cultural_significance: Optional[str] = "moderate"
    economic_importance: Optional[str] = "moderate"
    strategic_value: Optional[int] = 5
    population_density: Optional[str] = "moderate"
    notable_features: List[Any] = field(default_factory=list)
    hidden_aspects: List[Any] = field(default_factory=list)
    access_restrictions: List[Any] = field(default_factory=list)
    local_customs: List[Any] = field(default_factory=list)
    room: Optional[str] = None
    building: Optional[str] = None
    district: Optional[str] = None
    district_type: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    planet: str = "Earth"
    galaxy: str = "Milky Way"
    realm: Optional[str] = DEFAULT_REALM
    scope: Optional[Scope] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    is_fictional: bool = False
    open_hours: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    controlling_faction: Optional[str] = None
    external_place_id: Optional[str] = None

    @property
    def location_id(self) -> Optional[int]:
        """Backward compatible alias for the primary key column."""

        return self.id

    @location_id.setter
    def location_id(self, value: Optional[int]) -> None:
        self.id = value

    def __post_init__(self) -> None:
        try:
            self.user_id = int(self.user_id)
        except (TypeError, ValueError):
            raise ValueError("user_id must be coercible to int")
        try:
            self.conversation_id = int(self.conversation_id)
        except (TypeError, ValueError):
            raise ValueError("conversation_id must be coercible to int")
        self.location_name = self.location_name.strip()
        self.description = self._clean_optional_str(self.description)
        self.location_type = self._clean_optional_str(self.location_type)
        self.parent_location = self._clean_optional_str(self.parent_location)
        self.cultural_significance = self._clean_optional_str(self.cultural_significance)
        self.economic_importance = self._clean_optional_str(self.economic_importance)
        self.population_density = self._clean_optional_str(self.population_density)
        self.room = self._clean_optional_str(self.room)
        self.building = self._clean_optional_str(self.building)
        self.district = self._clean_optional_str(self.district)
        self.district_type = self._clean_optional_str(self.district_type)
        self.city = self._clean_optional_str(self.city)
        self.region = self._clean_optional_str(self.region)
        self.country = self._clean_optional_str(self.country)
        self.planet = self._clean_required_str(self.planet, default="Earth")
        self.galaxy = self._clean_required_str(self.galaxy, default="Milky Way")
        realm_value = self._clean_optional_str(self.realm)
        self.realm = realm_value or DEFAULT_REALM
        self.lat = self._coerce_float(self.lat)
        self.lon = self._coerce_float(self.lon)
        self.is_fictional = bool(self.is_fictional)
        try:
            resolved_scope = self._resolve_scope(self.scope, self.is_fictional)
        except ValueError as exc:
            raise ValueError("scope must be either 'real' or 'fictional'") from exc
        self.scope = resolved_scope
        self.is_fictional = self.scope == "fictional"
        self.notable_features = self._coerce_list(self.notable_features)
        self.hidden_aspects = self._coerce_list(self.hidden_aspects)
        self.access_restrictions = self._coerce_list(self.access_restrictions)
        self.local_customs = self._coerce_list(self.local_customs)
        self.open_hours = self._coerce_dict(self.open_hours)
        self.embedding = self._coerce_float_list(self.embedding)
        self.controlling_faction = self._clean_optional_str(self.controlling_faction)
        self.external_place_id = self._clean_optional_str(self.external_place_id)
        if self.strategic_value is not None:
            try:
                self.strategic_value = int(self.strategic_value)
            except (TypeError, ValueError):
                self.strategic_value = None

    @staticmethod
    def _clean_optional_str(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return str(value).strip() or None

    @staticmethod
    def _clean_required_str(value: Optional[str], *, default: str) -> str:
        cleaned = Location._clean_optional_str(value)
        return cleaned or default

    @staticmethod
    def _coerce_float(value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_list(value: Optional[Iterable[Any]]) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, (set, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _coerce_dict(value: Optional[Any]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return dict(value)
        return None

    @staticmethod
    def _coerce_float_list(value: Optional[Any]) -> Optional[List[float]]:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return [float(v) for v in value]
        if isinstance(value, memoryview):
            try:
                data = value.tolist()  # type: ignore[attr-defined]
            except AttributeError:
                data = list(value)
            return [float(item) for item in data]
        if isinstance(value, Iterable) and not isinstance(value, (str, dict)):
            out: List[float] = []
            for item in value:
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    continue
            return out or None
        return None

    @staticmethod
    def _resolve_scope(scope_value: Any, is_fictional_value: Any) -> Scope:
        default_scope: Scope = "fictional" if bool(is_fictional_value) else "real"
        if scope_value is None:
            return default_scope
        if isinstance(scope_value, str):
            normalized = scope_value.strip().lower()
            if not normalized:
                return default_scope
            if normalized in {"true", "false"}:
                return "fictional" if normalized == "true" else "real"
            if normalized in {"real", "fictional"}:
                return cast(Scope, normalized)
        if isinstance(scope_value, bool):
            return "fictional" if scope_value else "real"
        if isinstance(scope_value, (int, float)):
            if scope_value == 0:
                return "real"
            if scope_value == 1:
                return "fictional"
        raise ValueError(f"Invalid scope value: {scope_value!r}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], **overrides: Any) -> "Location":
        data: Dict[str, Any] = dict(mapping)
        data.update(overrides)
        if "id" not in data and "location_id" in data:
            data["id"] = data["location_id"]
        if "scope" in data:
            data["scope"] = cls._resolve_scope(data.get("scope"), data.get("is_fictional"))
        else:
            data["scope"] = cls._resolve_scope(None, data.get("is_fictional"))
        init_kwargs: Dict[str, Any] = {}
        for field_def in fields(cls):
            if field_def.name in data:
                init_kwargs[field_def.name] = data[field_def.name]
            elif field_def.default is MISSING and field_def.default_factory is MISSING:
                raise ValueError(f"Missing required field '{field_def.name}' for Location")
        return cls(**init_kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_record(cls, record: Mapping[str, Any], **overrides: Any) -> "Location":
        return cls.from_mapping(record, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "location_name": self.location_name,
            "description": self.description,
            "location_type": self.location_type,
            "parent_location": self.parent_location,
            "cultural_significance": self.cultural_significance,
            "economic_importance": self.economic_importance,
            "strategic_value": self.strategic_value,
            "population_density": self.population_density,
            "notable_features": list(self.notable_features),
            "hidden_aspects": list(self.hidden_aspects),
            "access_restrictions": list(self.access_restrictions),
            "local_customs": list(self.local_customs),
            "room": self.room,
            "building": self.building,
            "district": self.district,
            "district_type": self.district_type,
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "planet": self.planet,
            "galaxy": self.galaxy,
            "realm": self.realm,
            "scope": self.scope,
            "lat": self.lat,
            "lon": self.lon,
            "is_fictional": self.is_fictional,
            "open_hours": dict(self.open_hours) if isinstance(self.open_hours, dict) else None,
            "embedding": list(self.embedding) if self.embedding else None,
            "controlling_faction": self.controlling_faction,
            "external_place_id": self.external_place_id,
        }


__all__ = [
    "Anchor",
    "DEFAULT_REALM",
    "Candidate",
    "Location",
    "parse_location_chain",
    "Place",
    "PlaceEdge",
    "PlaceLevel",
    "ResolveResult",
    "ResolveStatus",
    "Scope",
    "STATUS_EXACT",
    "STATUS_NEARBY",
    "STATUS_MULTIPLE",
    "STATUS_AMBIGUOUS",
    "STATUS_NOT_FOUND",
    "STATUS_TRAVEL_PLAN",
    "STATUS_ASK",
]

