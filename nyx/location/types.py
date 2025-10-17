# nyx/location/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Scope = Literal["real", "fictional", "hybrid"]

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
    anchor: Optional[Anchor] = None
    scope: Optional[Scope] = None
    errors: List[str] = field(default_factory=list)

    @property
    def canonical_ops(self) -> List[Dict[str, Any]]:
        """Backward compatible view of operations for legacy callers."""

        return self.operations

    @canonical_ops.setter
    def canonical_ops(self, value: Optional[List[Dict[str, Any]]]) -> None:
        self.operations = list(value or [])


__all__ = [
    "Anchor",
    "Candidate",
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
