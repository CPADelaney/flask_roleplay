"""
Unified Schemas for Lore System

This module provides all data schemas and validation for the lore system.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class LoreType(str, Enum):
    """Types of lore content"""
    WORLD = "world"
    NPC = "npc"
    LOCATION = "location"
    EVENT = "event"
    CULTURE = "culture"
    RELIGION = "religion"
    FACTION = "faction"
    CONFLICT = "conflict"
    ARTIFACT = "artifact"
    QUEST = "quest"

class RelationType(str, Enum):
    """Types of relationships between entities"""
    ALLY = "ally"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    FAMILY = "family"
    FRIEND = "friend"
    RIVAL = "rival"
    MENTOR = "mentor"
    STUDENT = "student"
    BUSINESS = "business"
    ROMANTIC = "romantic"

class LocationType(str, Enum):
    """Types of locations"""
    CITY = "city"
    TOWN = "town"
    VILLAGE = "village"
    DUNGEON = "dungeon"
    WILDERNESS = "wilderness"
    LANDMARK = "landmark"
    BUILDING = "building"
    REGION = "region"
    REALM = "realm"

class ConflictType(str, Enum):
    """Types of conflicts"""
    WAR = "war"
    SKIRMISH = "skirmish"
    POLITICAL = "political"
    ECONOMIC = "economic"
    SOCIAL = "social"
    RELIGIOUS = "religious"
    PERSONAL = "personal"

class ConflictStatus(str, Enum):
    """Status of conflicts"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    DORMANT = "dormant"
    ESCALATING = "escalating"
    DEESCALATING = "deescalating"

class BaseEntity(BaseModel):
    """Base model for all lore entities"""
    id: int
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        orm_mode = True

class World(BaseEntity):
    """World/Setting model"""
    settings: Dict[str, Any]
    rules: Dict[str, Any]
    themes: List[str]
    cultures: List[int] = Field(default_factory=list)  # Culture IDs
    religions: List[int] = Field(default_factory=list)  # Religion IDs
    factions: List[int] = Field(default_factory=list)  # Faction IDs

class NPC(BaseEntity):
    """NPC model"""
    faction_id: Optional[int]
    location_id: Optional[int]
    traits: List[str]
    relationships: Dict[int, Dict[str, Any]]  # NPC ID -> Relationship details
    knowledge: Dict[str, Any]
    status: str = "active"

class Location(BaseEntity):
    """Location model"""
    region_id: Optional[int]
    type: LocationType
    coordinates: Optional[Dict[str, float]]
    properties: Dict[str, Any]
    climate: Optional[str]
    terrain: Optional[str]
    resources: List[str] = Field(default_factory=list)
    points_of_interest: List[int] = Field(default_factory=list)  # POI IDs

class PointOfInterest(BaseEntity):
    """Point of Interest model"""
    location_id: int
    type: str
    significance: str

class Culture(BaseEntity):
    """Culture model"""
    traditions: List[str]
    values: List[str]
    social_structure: Dict[str, Any]
    beliefs: List[str]

class Religion(BaseEntity):
    """Religion model"""
    deities: List[Dict[str, Any]]
    practices: List[str]
    holy_sites: List[int]  # Location IDs
    beliefs: List[str]

class Faction(BaseEntity):
    """Faction model"""
    alignment: str
    influence: float = Field(ge=0, le=1)
    relationships: Dict[int, Dict[str, Any]]  # Faction ID -> Relationship details
    resources: Dict[str, Any]
    goals: List[str]

class Conflict(BaseEntity):
    """Conflict model"""
    type: ConflictType
    status: ConflictStatus
    start_date: datetime
    end_date: Optional[datetime]
    participants: List[Dict[str, Any]]
    causes: List[str]
    effects: List[str]

class HistoricalEvent(BaseEntity):
    """Historical Event model"""
    date: datetime
    significance: str
    participants: List[Dict[str, Any]]
    causes: List[str]
    effects: List[str]
    related_events: List[int] = Field(default_factory=list)  # Event IDs

class Artifact(BaseEntity):
    """Artifact model"""
    type: str
    properties: Dict[str, Any]
    location_id: Optional[int]
    owner_id: Optional[int]  # NPC or Faction ID
    history: List[Dict[str, Any]]

class Quest(BaseEntity):
    """Quest model"""
    type: str
    status: str
    objectives: List[Dict[str, Any]]
    rewards: List[Dict[str, Any]]
    prerequisites: List[Dict[str, Any]]
    related_npcs: List[int]  # NPC IDs
    related_locations: List[int]  # Location IDs

# Request/Response Models
class QueryRequest(BaseModel):
    """Model for lore query requests"""
    type: LoreType
    params: Dict[str, Any] = Field(default_factory=dict)

class GenerationRequest(BaseModel):
    """Model for lore generation requests"""
    type: LoreType
    parameters: Dict[str, Any]

class IntegrationRequest(BaseModel):
    """Model for lore integration requests"""
    type: LoreType
    content: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str
    type: str
    details: Optional[Dict[str, Any]]
    status_code: int = 500

# Validation Functions
def validate_coordinates(coords: Dict[str, float]) -> bool:
    """Validate location coordinates"""
    required = {'latitude', 'longitude'}
    if not all(k in coords for k in required):
        return False
    return -90 <= coords['latitude'] <= 90 and -180 <= coords['longitude'] <= 180

def validate_relationship_strength(strength: float) -> bool:
    """Validate relationship strength"""
    return 0 <= strength <= 1

def validate_date_range(start: datetime, end: Optional[datetime]) -> bool:
    """Validate a date range"""
    if end is None:
        return True
    return start <= end 