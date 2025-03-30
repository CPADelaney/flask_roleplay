# nyx/core/spatial/spatial_schemas.py

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

class SpatialCoordinate(BaseModel):
    """Represents a 2D or 3D coordinate"""
    x: float
    y: float
    z: Optional[float] = None
    reference_frame: str = "global"
    uncertainty: float = 0.0  # Uncertainty radius in appropriate units

class SpatialObservation(BaseModel):
    """Represents a spatial observation input"""
    observation_type: str  # object, region, relative_position, etc.
    content: Dict[str, Any]  # Flexible content based on observation type
    confidence: float = 1.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    observer_position: Optional[SpatialCoordinate] = None
    observer_orientation: Optional[Dict[str, float]] = None  # heading, pitch, roll
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SpatialQuery(BaseModel):
    """Query parameters for spatial information retrieval"""
    query_type: str  # nearest, contains, path, region_info, etc.
    parameters: Dict[str, Any]
    reference_position: Optional[SpatialCoordinate] = None
    map_id: Optional[str] = None

class NavigationRequest(BaseModel):
    """Request for navigation assistance"""
    start_location: Union[str, Dict[str, float]]  # Location name or coordinates
    destination: str  # Destination name
    preferences: Dict[str, Any] = Field(default_factory=dict)  # Preferences like "avoid_stairs", "prefer_landmarks"
    map_id: Optional[str] = None
    include_alternatives: bool = False

class NavigationResult(BaseModel):
    """Result of a navigation request"""
    success: bool
    message: str
    directions: List[str]
    distance: Optional[float] = None
    estimated_time: Optional[float] = None
    route_id: Optional[str] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)

class SpatialDescription(BaseModel):
    """Description of a spatial area"""
    description: str
    objects: List[Dict[str, Any]] = Field(default_factory=list)
    regions: List[Dict[str, Any]] = Field(default_factory=list)
    landmarks: List[Dict[str, Any]] = Field(default_factory=list)

class EnvironmentChange(BaseModel):
    """Represents a change in the environment"""
    change_type: str  # addition, removal, movement, etc.
    object_id: Optional[str] = None
    region_id: Optional[str] = None
    old_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    description: Optional[str] = None

class SpatialMemoryReference(BaseModel):
    """Reference to a spatial memory"""
    memory_id: str
    relevance: float
    location_id: Optional[str] = None  # Object or region ID
    coordinates: Optional[SpatialCoordinate] = None
    timestamp: str
    description: str

class LandmarkInfo(BaseModel):
    """Information about a landmark"""
    landmark_id: str
    name: str
    landmark_type: str
    salience: float = 1.0  # How salient/memorable this landmark is
    description: Optional[str] = None
    coordinates: SpatialCoordinate
    visibility_radius: Optional[float] = None  # How far away it can be seen
    properties: Dict[str, Any] = Field(default_factory=dict)

class SpatialEvent(BaseModel):
    """An event that happened at a specific location"""
    event_id: str
    event_type: str
    location_id: Optional[str] = None
    coordinates: Optional[SpatialCoordinate] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    description: str
    participants: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    memory_id: Optional[str] = None  # Associated memory ID if available

class SpatialRelationship(BaseModel):
    """Relationship between spatial entities"""
    source_id: str
    target_id: str
    relationship_type: str  # contains, adjacent, above, below, etc.
    strength: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)

class PathfindingRequest(BaseModel):
    """Request for pathfinding between locations"""
    start_id: str
    end_id: str
    constraints: Dict[str, Any] = Field(default_factory=dict)  # e.g., "avoid_regions": [...]
    preference: str = "shortest"  # shortest, safest, most_landmarks, etc.
    map_id: Optional[str] = None

class CognitiveMapSummary(BaseModel):
    """Summary of a cognitive map"""
    map_id: str
    name: str
    description: Optional[str] = None
    object_count: int
    region_count: int
    route_count: int
    landmark_count: int
    completeness: float
    accuracy: float
    creation_date: str
    last_updated: str
