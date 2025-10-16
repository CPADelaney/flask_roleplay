# nyx/location/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class SettingKind(str, Enum):
    REAL = "real"
    FICTIONAL = "fictional"
    HYBRID = "hybrid"

@dataclass
class SettingProfile:
    kind: SettingKind
    primary_city: Optional[str] = None           # e.g., "San Francisco"
    region: Optional[str] = None                 # e.g., "CA"
    country: Optional[str] = None                # e.g., "USA"
    lat: Optional[float] = None
    lon: Optional[float] = None
    label: Optional[str] = None                  # "SoMa, San Francisco"
    world_name: Optional[str] = None             # for fictional worlds

@dataclass
class PlaceQuery:
    raw_text: str
    normalized: str
    is_travel: bool = False                      # e.g., "fly to Tokyo"
    target: Optional[str] = None                 # the thing to find
    transport_hint: Optional[str] = None         # "walk", "drive", "fly"

@dataclass
class PlaceCandidate:
    name: str
    lat: float
    lon: float
    address: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None               # "landmark", "restaurant", "airport", "festival"
    confidence: float = 0.0
    notes: Optional[str] = None

class ResolutionStatus(str, Enum):
    EXACT = "exact"
    NEARBY = "nearby"
    MULTIPLE = "multiple"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    TRAVEL_PLAN = "travel_plan"

@dataclass
class TravelLeg:
    kind: str                                    # "local", "transit", "flight"
    origin_label: str
    dest_label: str
    origin: Optional[Tuple[float, float]] = None
    dest: Optional[Tuple[float, float]] = None
    estimate_min: Optional[int] = None
    carrier: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class TravelPlan:
    legs: List[TravelLeg] = field(default_factory=list)
    arrival_setting: Optional[SettingProfile] = None

@dataclass
class ResolutionResult:
    status: ResolutionStatus
    message: Optional[str] = None                # guidance/"ask" prompt text
    choices: List[str] = field(default_factory=list)  # clarifying options
    candidates: List[PlaceCandidate] = field(default_factory=list)
    travel: Optional[TravelPlan] = None
    canonical_ops: List[Dict[str, Any]] = field(default_factory=list)  # ops for universal_updater
    anchor_used: Optional[str] = None            # e.g., "SoMa, San Francisco"
