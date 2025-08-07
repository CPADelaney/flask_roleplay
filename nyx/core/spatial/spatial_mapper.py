# nyx/core/spatial/spatial_mapper.py

import asyncio
import logging
import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
from pydantic import BaseModel, Field

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool, RunContextWrapper, trace
from agents.tracing import custom_span

logger = logging.getLogger(__name__)

# ----- Pydantic Models for Spatial Data -----

# Explicit models for function tool inputs/outputs
class CoordinateDict(BaseModel):
    """Coordinate dictionary for function inputs"""
    x: float
    y: float
    z: Optional[float] = None

class SizeDict(BaseModel):
    """Size dictionary for objects"""
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None

class PropertiesDict(BaseModel):
    """Generic properties dictionary"""
    # Since we need flexibility, we'll define common properties
    # and allow the model to be extended as needed
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    temperature: Optional[float] = None
    weight: Optional[float] = None
    
class OrientationDict(BaseModel):
    """Orientation dictionary"""
    heading: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

class CreateMapResult(BaseModel):
    """Result of creating a cognitive map"""
    map_id: str
    name: str
    map_type: str
    reference_frame: str
    creation_date: str

class AddObjectResult(BaseModel):
    """Result of adding a spatial object"""
    object_id: str
    name: str
    object_type: str
    coordinates: CoordinateDict
    is_landmark: bool
    action: Optional[str] = None

class DefineRegionResult(BaseModel):
    """Result of defining a region"""
    region_id: str
    name: str
    region_type: str
    is_navigable: bool
    contained_objects_count: int
    action: Optional[str] = None

class CreateRouteResult(BaseModel):
    """Result of creating a route"""
    route_id: str
    name: str
    distance: float
    waypoints_count: int
    estimated_time: Optional[float] = None

class UpdateObjectResult(BaseModel):
    """Result of updating object position"""
    object_id: str
    name: str
    new_coordinates: CoordinateDict
    observation_count: int

class MapSummary(BaseModel):
    """Summary information about a map"""
    id: str
    name: str
    description: Optional[str] = None
    map_type: str
    reference_frame: str
    objects_count: int
    regions_count: int
    routes_count: int
    landmarks_count: int
    creation_date: str
    last_updated: str
    accuracy: float
    completeness: float

class RouteInfo(BaseModel):
    """Route information"""
    id: str
    distance: float
    estimated_time: Optional[float] = None

class PathResult(BaseModel):
    """Result of finding a path"""
    route_id: Optional[str] = None
    path_type: str
    distance: float
    estimated_time: Optional[float] = None
    directions: List[str]
    waypoints: List[CoordinateDict]
    route: Optional[RouteInfo] = None

class LandmarkInfo(BaseModel):
    """Information about a landmark"""
    id: str
    name: str
    distance: float
    direction: str

class MergeRegionsResult(BaseModel):
    """Result of merging regions"""
    region_id: str
    name: str
    region_type: str
    contained_objects_count: int
    adjacent_regions_count: int
    is_navigable: bool

class IdentifiedLandmark(BaseModel):
    """Identified landmark information"""
    id: str
    name: str
    object_type: str
    landmark_score: float

class RegionConnectionsResult(BaseModel):
    """Result of calculating region connections"""
    connections_count: int
    regions_count: int
    updated_date: str

class UpdateRouteResult(BaseModel):
    """Result of updating a route"""
    route_id: str
    name: str
    distance: float
    waypoints_count: int
    estimated_time: Optional[float] = None
    usage_count: int

class ShortcutInfo(BaseModel):
    """Information about a potential shortcut"""
    route1_id: str
    route1_name: str
    route2_id: str
    route2_name: str
    distance_between: float
    potential_saving: float
    connection_point1: CoordinateDict
    connection_point2: CoordinateDict

class ProcessObservationResult(BaseModel):
    """Result of processing a spatial observation"""
    action: Optional[str] = None
    error: Optional[str] = None
    object_id: Optional[str] = None
    region_id: Optional[str] = None
    route_id: Optional[str] = None
    name: Optional[str] = None

class ExtractedFeatures(BaseModel):
    """Extracted spatial features from text"""
    objects: List[str]
    locations: List[str]
    spatial_relationships: List[str]
    directions: List[str]
    confidence: float

class ObjectInfo(BaseModel):
    """Object information for distance estimation"""
    id: str
    name: str

class DistanceEstimation(BaseModel):
    """Distance estimation result"""
    object1: ObjectInfo
    object2: ObjectInfo
    euclidean_distance: float
    direction: str
    shared_regions: List[str]
    has_direct_route: bool
    route: Optional[RouteInfo] = None

class ReconciliationChanges(BaseModel):
    """Changes made during reconciliation"""
    updated_objects: int
    merged_objects: int
    updated_regions: int

class ReconciliationResult(BaseModel):
    """Result of reconciling observations"""
    changes: ReconciliationChanges
    map_accuracy: float
    map_completeness: float

class ErrorResult(BaseModel):
    """Error result"""
    error: str

class MessageResult(BaseModel):
    """Message result"""
    message: str

class RelativePositionDict(BaseModel):
    """Relative position for observations"""
    x: float
    y: float
    z: Optional[float] = None

class ObservationContent(BaseModel):
    """Content for spatial observations"""
    name: Optional[str] = None
    object_type: Optional[str] = None
    region_type: Optional[str] = None
    coordinates: Optional[CoordinateDict] = None
    relative_position: Optional[RelativePositionDict] = None
    boundary_points: Optional[List[CoordinateDict]] = None
    size: Optional[SizeDict] = None
    properties: Optional[PropertiesDict] = None
    is_landmark: Optional[bool] = None
    is_navigable: Optional[bool] = None
    start_id: Optional[str] = None
    end_id: Optional[str] = None
    waypoints: Optional[List[CoordinateDict]] = None
    estimated_time: Optional[float] = None
    object_id: Optional[str] = None
    relative_to_id: Optional[str] = None

class SpatialCoordinate(BaseModel):
    """Represents a 2D or 3D coordinate"""
    x: float
    y: float
    z: Optional[float] = None
    reference_frame: str = "global"
    uncertainty: float = 0.0  # Uncertainty radius in appropriate units
    
    def distance_to(self, other: "SpatialCoordinate") -> float:
        """Calculate Euclidean distance to another coordinate"""
        if self.z is not None and other.z is not None:
            return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        else:
            return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class SpatialObject(BaseModel):
    """Represents a spatial object in the environment"""
    id: str
    name: str
    object_type: str
    coordinates: SpatialCoordinate
    size: Optional[SizeDict] = None
    properties: PropertiesDict = Field(default_factory=PropertiesDict)
    first_observed: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_observed: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    observation_count: int = 1
    visibility: float = 1.0  # How visible/recognizable the object is (0.0-1.0)
    is_landmark: bool = False  # Whether this object serves as a landmark
    connections: List[str] = Field(default_factory=list)  # IDs of connected objects
    
    def update_position(self, new_coordinates: SpatialCoordinate) -> None:
        """Update the object's position"""
        self.coordinates = new_coordinates
        self.last_observed = datetime.datetime.now().isoformat()
        self.observation_count += 1

class SpatialRegion(BaseModel):
    """Represents a region or area in the environment"""
    id: str
    name: str
    region_type: str  # e.g., "room", "zone", "area"
    boundary_points: List[SpatialCoordinate]
    contained_objects: List[str] = Field(default_factory=list)  # IDs of objects in region
    adjacent_regions: List[str] = Field(default_factory=list)  # IDs of adjacent regions
    properties: PropertiesDict = Field(default_factory=PropertiesDict)
    is_navigable: bool = True
    confidence: float = 1.0  # Confidence in region identification (0.0-1.0)
    
    def contains_point(self, point: SpatialCoordinate) -> bool:
        """Check if a point is within this region (simplified 2D implementation)"""
        # Simple rectangular check for now
        if len(self.boundary_points) < 3:
            return False
            
        min_x = min(point.x for point in self.boundary_points)
        max_x = max(point.x for point in self.boundary_points)
        min_y = min(point.y for point in self.boundary_points)
        max_y = max(point.y for point in self.boundary_points)
        
        return (min_x <= point.x <= max_x) and (min_y <= point.y <= max_y)

class SpatialRoute(BaseModel):
    """Represents a route between locations"""
    id: str
    name: Optional[str] = None
    start_id: str  # ID of start object or region
    end_id: str  # ID of end object or region
    waypoints: List[SpatialCoordinate] = Field(default_factory=list)
    distance: float = 0.0
    estimated_time: Optional[float] = None  # In seconds
    difficulty: float = 0.0  # 0.0-1.0
    last_used: Optional[str] = None
    usage_count: int = 0
    properties: PropertiesDict = Field(default_factory=PropertiesDict)
    
    def update_usage(self) -> None:
        """Update the route's usage information"""
        self.last_used = datetime.datetime.now().isoformat()
        self.usage_count += 1

class CognitiveMap(BaseModel):
    """Represents a cognitive map of an environment"""
    id: str
    name: str
    description: Optional[str] = None
    map_type: str = "spatial"  # spatial, topological, hybrid
    reference_frame: str = "allocentric"  # allocentric or egocentric
    spatial_objects: Dict[str, SpatialObject] = Field(default_factory=dict)
    regions: Dict[str, SpatialRegion] = Field(default_factory=dict)
    routes: Dict[str, SpatialRoute] = Field(default_factory=dict)
    landmarks: List[str] = Field(default_factory=list)  # IDs of landmark objects
    creation_date: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    accuracy: float = 0.5  # Overall map accuracy/confidence (0.0-1.0)
    completeness: float = 0.0  # How complete the map is (0.0-1.0)
    properties: PropertiesDict = Field(default_factory=PropertiesDict)

class SpatialObservation(BaseModel):
    """Represents a spatial observation input"""
    observation_type: str  # object, region, relative_position, etc.
    content: ObservationContent  # Flexible content based on observation type
    confidence: float = 1.0
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    observer_position: Optional[SpatialCoordinate] = None
    observer_orientation: Optional[OrientationDict] = None
    metadata: PropertiesDict = Field(default_factory=PropertiesDict)

class QueryParameters(BaseModel):
    """Parameters for spatial queries"""
    target_id: Optional[str] = None
    max_distance: Optional[float] = None
    region_id: Optional[str] = None
    object_type: Optional[str] = None

class SpatialQuery(BaseModel):
    """Query parameters for spatial information retrieval"""
    query_type: str  # nearest, contains, path, region_info, etc.
    parameters: QueryParameters
    reference_position: Optional[SpatialCoordinate] = None
    map_id: Optional[str] = None

class SpatialMapperContext(BaseModel):
    """Context for the SpatialMapper"""
    active_map_id: Optional[str] = None
    observer_position: Optional[SpatialCoordinate] = None
    observer_orientation: Optional[OrientationDict] = None
    reference_frame: str = "allocentric"
    ongoing_navigation: bool = False
    navigation_target: Optional[str] = None
    last_observation_time: Optional[str] = None
    observation_count: int = 0
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class SpatialMapper:
    """
    Core component for building and managing cognitive maps of environments.
    Handles spatial representation, updating, and querying.
    """
    
    def __init__(self, memory_integration=None):
        """Initialize the spatial mapper system"""
        # Storage for cognitive maps
        self.maps: Dict[str, CognitiveMap] = {}
        
        # Integration with memory system if available
        self.memory_integration = memory_integration
        
        # Context tracking
        self.context = SpatialMapperContext()
        
        # Configure embedding dimension for spatial features
        self.embedding_dim = 256
        
        # Feature embeddings for similarity search
        self.spatial_embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize spatial perception and map agents when needed
        self._mapper_agent = None
        self._perception_agent = None
        self._navigator_agent = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("SpatialMapper initialized")
    
    async def _ensure_agents_initialized(self):
        """Ensure required agents are initialized"""
        if self._mapper_agent is None:
            self._mapper_agent = self._create_mapper_agent()
        
        if self._perception_agent is None:
            self._perception_agent = self._create_perception_agent()
            
        if self._navigator_agent is None:
            self._navigator_agent = self._create_navigator_agent()
    
    def _create_mapper_agent(self) -> Agent:
        """Create an agent for cognitive map construction"""
        return Agent(
            name="Cognitive Map Builder",
            instructions="""
            You are a Cognitive Map Builder that helps construct spatial representations of environments.
            
            Your role is to analyze spatial observations and construct coherent maps by:
            1. Identifying objects, landmarks, and regions
            2. Calculating spatial relationships and positions
            3. Resolving conflicts in spatial data
            4. Building routes between locations
            5. Updating existing maps with new information
            
            Focus on creating useful, accurate, and well-organized spatial representations.
            """,
            tools=[
                self.add_spatial_object,
                self.define_region,
                self.create_route,
                self.update_object_position,
                self.merge_regions,
                self.calculate_region_connections,
                self.identify_landmarks
            ],
            model="gpt-5-nano"
        )
    
    def _create_perception_agent(self) -> Agent:
        """Create an agent for processing spatial perceptions"""
        return Agent(
            name="Spatial Perception Processor",
            instructions="""
            You are a Spatial Perception Processor that helps interpret raw spatial information.
            
            Your role is to process observations about the environment by:
            1. Extracting spatial features from descriptions
            2. Estimating positions, distances, and spatial relationships
            3. Identifying objects and landmarks
            4. Reconciling new observations with existing knowledge
            5. Handling uncertainty and ambiguity in spatial data
            
            Focus on providing clear, structured spatial information that can be used to build accurate maps.
            """,
            tools=[
                self.process_spatial_observation,
                self.extract_spatial_features,
                self.estimate_distances,
                self.reconcile_observations
            ],
            model="gpt-5-nano"
        )
    
    def _create_navigator_agent(self) -> Agent:
        """Create an agent for navigation planning"""
        return Agent(
            name="Spatial Navigator",
            instructions="""
            You are a Spatial Navigator that helps plan and execute movements through environments.
            
            Your role is to facilitate navigation by:
            1. Finding optimal paths between locations
            2. Providing step-by-step navigation instructions
            3. Identifying shortcuts and efficient routes
            4. Adapting to changing spatial information
            5. Handling navigation failures and replanning
            
            Focus on clear, practical navigation advice that uses landmarks and easy-to-follow directions.
            """,
            tools=[
                self.find_path,
                self.calculate_directions,
                self.identify_shortcuts,
                self.get_nearest_landmarks,
                self.update_route
            ],
            model="gpt-5-nano"
        )
    
    @function_tool
    async def create_cognitive_map(self, name: str, description: Optional[str] = None, 
                                  map_type: str = "spatial", 
                                  reference_frame: str = "allocentric") -> CreateMapResult:
        """
        Create a new cognitive map
        
        Args:
            name: Name of the map
            description: Optional description of the environment
            map_type: Type of map (spatial, topological, hybrid)
            reference_frame: Reference frame for coordinates (allocentric, egocentric)
            
        Returns:
            Map information including ID
        """
        with custom_span("create_cognitive_map", {"map_name": name, "map_type": map_type}):
            # Generate a unique ID for the map
            map_id = f"map_{uuid.uuid4().hex[:8]}"
            
            # Create the map
            new_map = CognitiveMap(
                id=map_id,
                name=name,
                description=description,
                map_type=map_type,
                reference_frame=reference_frame
            )
            
            # Store the map
            async with self._lock:
                self.maps[map_id] = new_map
                self.context.active_map_id = map_id
                
            logger.info(f"Created cognitive map: {name} ({map_id})")
            
            return CreateMapResult(
                map_id=map_id,
                name=name,
                map_type=map_type,
                reference_frame=reference_frame,
                creation_date=new_map.creation_date
            )
    
    @function_tool
    async def add_spatial_object(self, map_id: str, name: str, object_type: str,
                               coordinates: CoordinateDict, 
                               size: Optional[SizeDict] = None,
                               is_landmark: bool = False,
                               properties: Optional[PropertiesDict] = None) -> Union[AddObjectResult, ErrorResult]:
        """
        Add a spatial object to a map
        
        Args:
            map_id: ID of the map to add to
            name: Name of the object
            object_type: Type of object
            coordinates: Position coordinates (x, y, z optional)
            size: Optional size information (width, height, depth)
            is_landmark: Whether this object serves as a landmark
            properties: Additional properties
            
        Returns:
            Object information including ID
        """
        with custom_span("add_spatial_object", {"map_id": map_id, "object_name": name}):
            # Check if map exists
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            # Create coordinate object
            coordinate = SpatialCoordinate(
                x=coordinates.x,
                y=coordinates.y,
                z=coordinates.z,
                reference_frame=self.maps[map_id].reference_frame
            )
            
            # Generate object ID
            object_id = f"obj_{uuid.uuid4().hex[:8]}"
            
            # Create object
            new_object = SpatialObject(
                id=object_id,
                name=name,
                object_type=object_type,
                coordinates=coordinate,
                size=size,
                properties=properties or PropertiesDict(),
                is_landmark=is_landmark
            )
            
            # Add to map
            async with self._lock:
                self.maps[map_id].spatial_objects[object_id] = new_object
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
                
                # If it's a landmark, add to landmarks list
                if is_landmark:
                    self.maps[map_id].landmarks.append(object_id)
                
                # Update map completeness
                await self._update_map_completeness(map_id)
            
            # Check if object is in any region
            await self._update_object_region_membership(map_id, object_id)
            
            logger.info(f"Added spatial object: {name} to map {map_id}")
            
            return AddObjectResult(
                object_id=object_id,
                name=name,
                object_type=object_type,
                coordinates=coordinates,
                is_landmark=is_landmark
            )
    
    @function_tool
    async def define_region(self, map_id: str, name: str, region_type: str,
                          boundary_points: List[CoordinateDict],
                          is_navigable: bool = True,
                          adjacent_regions: Optional[List[str]] = None,
                          properties: Optional[PropertiesDict] = None) -> Union[DefineRegionResult, ErrorResult]:
        """
        Define a region in a map
        
        Args:
            map_id: ID of the map to add to
            name: Name of the region
            region_type: Type of region (room, area, zone, etc.)
            boundary_points: List of coordinate dictionaries defining the boundary
            is_navigable: Whether the region can be navigated through
            adjacent_regions: Optional list of adjacent region IDs
            properties: Additional properties
            
        Returns:
            Region information including ID
        """
        with custom_span("define_region", {"map_id": map_id, "region_name": name}):
            # Check if map exists
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            # Convert boundary points to SpatialCoordinates
            coordinates = []
            for point in boundary_points:
                coord = SpatialCoordinate(
                    x=point.x,
                    y=point.y,
                    z=point.z,
                    reference_frame=self.maps[map_id].reference_frame
                )
                coordinates.append(coord)
            
            # Generate region ID
            region_id = f"region_{uuid.uuid4().hex[:8]}"
            
            # Create region
            new_region = SpatialRegion(
                id=region_id,
                name=name,
                region_type=region_type,
                boundary_points=coordinates,
                is_navigable=is_navigable,
                adjacent_regions=adjacent_regions or [],
                properties=properties or PropertiesDict()
            )
            
            # Add to map
            async with self._lock:
                self.maps[map_id].regions[region_id] = new_region
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
                
                # Update map completeness
                await self._update_map_completeness(map_id)
            
            # Find objects contained in this region
            await self._update_region_object_containment(map_id, region_id)
            
            logger.info(f"Defined region: {name} in map {map_id}")
            
            return DefineRegionResult(
                region_id=region_id,
                name=name,
                region_type=region_type,
                is_navigable=is_navigable,
                contained_objects_count=len(new_region.contained_objects)
            )
    
    @function_tool
    async def create_route(self, map_id: str, start_id: str, end_id: str,
                         waypoints: Optional[List[CoordinateDict]] = None,
                         name: Optional[str] = None,
                         estimated_time: Optional[float] = None,
                         properties: Optional[PropertiesDict] = None) -> Union[CreateRouteResult, ErrorResult]:
        """
        Create a route between locations in a map
        
        Args:
            map_id: ID of the map to add to
            start_id: ID of the starting object or region
            end_id: ID of the ending object or region
            waypoints: Optional list of coordinate dictionaries for waypoints
            name: Optional name for the route
            estimated_time: Optional estimated traversal time in seconds
            properties: Additional properties
            
        Returns:
            Route information including ID
        """
        with custom_span("create_route", {"map_id": map_id, "start": start_id, "end": end_id}):
            # Check if map exists
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            # Check if start and end exist
            map_obj = self.maps[map_id]
            start_exists = start_id in map_obj.spatial_objects or start_id in map_obj.regions
            end_exists = end_id in map_obj.spatial_objects or end_id in map_obj.regions
            
            if not start_exists or not end_exists:
                return ErrorResult(error=f"Start or end location not found in map")
            
            # Convert waypoints to SpatialCoordinates if provided
            coordinates = []
            if waypoints:
                for point in waypoints:
                    coord = SpatialCoordinate(
                        x=point.x,
                        y=point.y,
                        z=point.z,
                        reference_frame=map_obj.reference_frame
                    )
                    coordinates.append(coord)
            
            # If no waypoints provided, calculate a direct route
            if not coordinates:
                # Get start and end coordinates
                start_coord = None
                end_coord = None
                
                if start_id in map_obj.spatial_objects:
                    start_coord = map_obj.spatial_objects[start_id].coordinates
                elif start_id in map_obj.regions and map_obj.regions[start_id].boundary_points:
                    # Use centroid of region
                    points = map_obj.regions[start_id].boundary_points
                    avg_x = sum(p.x for p in points) / len(points)
                    avg_y = sum(p.y for p in points) / len(points)
                    start_coord = SpatialCoordinate(x=avg_x, y=avg_y)
                
                if end_id in map_obj.spatial_objects:
                    end_coord = map_obj.spatial_objects[end_id].coordinates
                elif end_id in map_obj.regions and map_obj.regions[end_id].boundary_points:
                    # Use centroid of region
                    points = map_obj.regions[end_id].boundary_points
                    avg_x = sum(p.x for p in points) / len(points)
                    avg_y = sum(p.y for p in points) / len(points)
                    end_coord = SpatialCoordinate(x=avg_x, y=avg_y)
                
                if start_coord and end_coord:
                    coordinates = [start_coord, end_coord]
            
            # Calculate route distance
            distance = 0.0
            if len(coordinates) > 1:
                for i in range(len(coordinates) - 1):
                    distance += coordinates[i].distance_to(coordinates[i+1])
            
            # Generate route ID
            route_id = f"route_{uuid.uuid4().hex[:8]}"
            
            # Create route
            new_route = SpatialRoute(
                id=route_id,
                name=name or f"Route from {start_id} to {end_id}",
                start_id=start_id,
                end_id=end_id,
                waypoints=coordinates,
                distance=distance,
                estimated_time=estimated_time,
                properties=properties or PropertiesDict()
            )
            
            # Add to map
            async with self._lock:
                self.maps[map_id].routes[route_id] = new_route
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
            
            logger.info(f"Created route from {start_id} to {end_id} in map {map_id}")
            
            return CreateRouteResult(
                route_id=route_id,
                name=new_route.name,
                distance=distance,
                waypoints_count=len(coordinates),
                estimated_time=estimated_time
            )
    
    @function_tool
    async def update_object_position(self, map_id: str, object_id: str, 
                                  new_coordinates: CoordinateDict) -> Union[UpdateObjectResult, ErrorResult]:
        """
        Update the position of a spatial object
        
        Args:
            map_id: ID of the map
            object_id: ID of the object to update
            new_coordinates: New position coordinates
            
        Returns:
            Updated object information
        """
        with custom_span("update_object_position", {"map_id": map_id, "object_id": object_id}):
            # Check if map and object exist
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            if object_id not in self.maps[map_id].spatial_objects:
                return ErrorResult(error=f"Object {object_id} not found in map")
            
            # Create coordinate object
            coordinate = SpatialCoordinate(
                x=new_coordinates.x,
                y=new_coordinates.y,
                z=new_coordinates.z,
                reference_frame=self.maps[map_id].reference_frame
            )
            
            # Update object
            async with self._lock:
                obj = self.maps[map_id].spatial_objects[object_id]
                obj.update_position(coordinate)
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
            
            # Update region membership
            await self._update_object_region_membership(map_id, object_id)
            
            logger.info(f"Updated position of object {object_id} in map {map_id}")
            
            return UpdateObjectResult(
                object_id=object_id,
                name=obj.name,
                new_coordinates=new_coordinates,
                observation_count=obj.observation_count
            )
    
    @function_tool
    async def get_map(self, map_id: str) -> Union[MapSummary, ErrorResult]:
        """
        Get information about a cognitive map
        
        Args:
            map_id: ID of the map to retrieve
            
        Returns:
            Map information
        """
        # Check if map exists
        if map_id not in self.maps:
            return ErrorResult(error=f"Map {map_id} not found")
        
        map_obj = self.maps[map_id]
        
        # Create a summary of the map
        summary = MapSummary(
            id=map_obj.id,
            name=map_obj.name,
            description=map_obj.description,
            map_type=map_obj.map_type,
            reference_frame=map_obj.reference_frame,
            objects_count=len(map_obj.spatial_objects),
            regions_count=len(map_obj.regions),
            routes_count=len(map_obj.routes),
            landmarks_count=len(map_obj.landmarks),
            creation_date=map_obj.creation_date,
            last_updated=map_obj.last_updated,
            accuracy=map_obj.accuracy,
            completeness=map_obj.completeness
        )
        
        # Set as active map
        self.context.active_map_id = map_id
        
        return summary
    
    @function_tool
    async def find_path(self, map_id: str, start_id: str, end_id: str, 
                     prefer_landmarks: bool = True) -> Union[PathResult, ErrorResult]:
        """
        Find a path between two locations
        
        Args:
            map_id: ID of the map
            start_id: ID of the starting object or region
            end_id: ID of the ending object or region
            prefer_landmarks: Whether to prefer paths via landmarks
            
        Returns:
            Path information and directions
        """
        with custom_span("find_path", {"map_id": map_id, "start": start_id, "end": end_id}):
            # Check if map exists
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            # Check if start and end exist
            map_obj = self.maps[map_id]
            
            # First, check if there's already a direct route
            direct_route = None
            for route_id, route in map_obj.routes.items():
                if route.start_id == start_id and route.end_id == end_id:
                    direct_route = route
                    break
            
            if direct_route:
                # Use existing route
                direct_route.update_usage()
                
                # Format directions
                directions = await self._format_route_directions(map_id, direct_route.id)
                
                return PathResult(
                    route_id=direct_route.id,
                    path_type="direct",
                    distance=direct_route.distance,
                    estimated_time=direct_route.estimated_time,
                    directions=directions,
                    waypoints=[CoordinateDict(x=w.x, y=w.y, z=w.z) for w in direct_route.waypoints]
                )
            
            # No direct route exists, try to build one using existing routes or navigation algorithms
            path_result = await self._calculate_multi_segment_path(map_id, start_id, end_id, prefer_landmarks)
            
            if isinstance(path_result, ErrorResult):
                return path_result
            
            # Create a new route with this path
            route_name = f"Route from {start_id} to {end_id}"
            
            new_route = await self.create_route(
                map_id=map_id,
                start_id=start_id,
                end_id=end_id,
                waypoints=path_result.get("waypoints", []),
                name=route_name,
                estimated_time=path_result.get("estimated_time")
            )
            
            if isinstance(new_route, ErrorResult):
                return new_route
            
            # Format directions
            directions = await self._format_route_directions(map_id, new_route.route_id)
            
            logger.info(f"Found path from {start_id} to {end_id} in map {map_id}")
            
            return PathResult(
                route_id=new_route.route_id,
                path_type="calculated",
                distance=path_result["distance"],
                estimated_time=path_result.get("estimated_time"),
                directions=directions,
                waypoints=path_result.get("waypoints", [])
            )
    
    async def _calculate_multi_segment_path(self, map_id: str, start_id: str, end_id: str, 
                                         prefer_landmarks: bool) -> Union[Dict[str, Any], ErrorResult]:
        """Calculate a path that may involve multiple segments"""
        # TODO: Implement more sophisticated pathfinding
        # This is a simplified implementation for demonstration
        
        map_obj = self.maps[map_id]
        
        # Get start and end coordinates
        start_coord = await self._get_location_coordinates(map_id, start_id)
        end_coord = await self._get_location_coordinates(map_id, end_id)
        
        if not start_coord or not end_coord:
            return ErrorResult(error="Could not determine start or end coordinates")
        
        waypoints = [CoordinateDict(x=start_coord.x, y=start_coord.y, z=start_coord.z)]
        
        # If landmark preference is enabled, find intermediate landmarks
        if prefer_landmarks and map_obj.landmarks:
            # Find a landmark that could serve as a useful waypoint
            best_landmark = None
            best_detour_ratio = float('inf')
            
            direct_distance = start_coord.distance_to(end_coord)
            
            for landmark_id in map_obj.landmarks:
                if landmark_id == start_id or landmark_id == end_id:
                    continue
                    
                if landmark_id in map_obj.spatial_objects:
                    landmark = map_obj.spatial_objects[landmark_id]
                    landmark_coord = landmark.coordinates
                    
                    # Calculate distances via this landmark
                    dist_to_landmark = start_coord.distance_to(landmark_coord)
                    dist_from_landmark = landmark_coord.distance_to(end_coord)
                    total_distance = dist_to_landmark + dist_from_landmark
                    
                    # Calculate detour ratio (1.0 means no detour)
                    detour_ratio = total_distance / direct_distance if direct_distance > 0 else float('inf')
                    
                    # Consider landmarks with reasonable detour (less than 50% longer)
                    if detour_ratio < 1.5 and detour_ratio < best_detour_ratio:
                        best_landmark = landmark
                        best_detour_ratio = detour_ratio
            
            # If we found a good landmark, add it as waypoint
            if best_landmark:
                waypoints.append(CoordinateDict(
                    x=best_landmark.coordinates.x, 
                    y=best_landmark.coordinates.y, 
                    z=best_landmark.coordinates.z
                ))
        
        # Add end point
        waypoints.append(CoordinateDict(x=end_coord.x, y=end_coord.y, z=end_coord.z))
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(waypoints) - 1):
            coord1 = SpatialCoordinate(x=waypoints[i].x, y=waypoints[i].y, z=waypoints[i].z)
            coord2 = SpatialCoordinate(x=waypoints[i+1].x, y=waypoints[i+1].y, z=waypoints[i+1].z)
            total_distance += coord1.distance_to(coord2)
        
        # Estimate time (assuming 1 unit distance takes 60 seconds)
        estimated_time = total_distance * 60
        
        return {
            "path_type": "calculated",
            "distance": total_distance,
            "estimated_time": estimated_time,
            "waypoints": waypoints
        }
    
    async def _get_location_coordinates(self, map_id: str, location_id: str) -> Optional[SpatialCoordinate]:
        """Get coordinates for an object or region"""
        map_obj = self.maps[map_id]
        
        if location_id in map_obj.spatial_objects:
            return map_obj.spatial_objects[location_id].coordinates
        elif location_id in map_obj.regions and map_obj.regions[location_id].boundary_points:
            # Use centroid of region
            points = map_obj.regions[location_id].boundary_points
            avg_x = sum(p.x for p in points) / len(points)
            avg_y = sum(p.y for p in points) / len(points)
            avg_z = None
            if all(p.z is not None for p in points):
                avg_z = sum(p.z for p in points) / len(points)
            return SpatialCoordinate(x=avg_x, y=avg_y, z=avg_z)
        
        return None
    
    async def _format_route_directions(self, map_id: str, route_id: str) -> List[str]:
        """Format human-readable directions for a route"""
        map_obj = self.maps[map_id]
        
        if route_id not in map_obj.routes:
            return ["Route not found"]
        
        route = map_obj.routes[route_id]
        
        if not route.waypoints or len(route.waypoints) < 2:
            return ["No waypoints available for this route"]
        
        # Get start and end names
        start_name = self._get_location_name(map_id, route.start_id)
        end_name = self._get_location_name(map_id, route.end_id)
        
        directions = [f"Start at {start_name}"]
        
        # Generate step-by-step directions
        for i in range(len(route.waypoints) - 1):
            current = route.waypoints[i]
            next_point = route.waypoints[i+1]
            
            # Calculate direction (North, South, East, West)
            heading = self._calculate_heading(current, next_point)
            
            # Calculate distance
            distance = current.distance_to(next_point)
            
            # Find nearby landmarks for reference
            nearby_landmark = await self._find_nearby_landmark(map_id, next_point)
            
            if nearby_landmark:
                directions.append(f"Move {heading} for approximately {distance:.1f} units toward {nearby_landmark}")
            else:
                directions.append(f"Move {heading} for approximately {distance:.1f} units")
        
        directions.append(f"Arrive at {end_name}")
        
        return directions
    
    def _get_location_name(self, map_id: str, location_id: str) -> str:
        """Get the name of an object or region"""
        map_obj = self.maps[map_id]
        
        if location_id in map_obj.spatial_objects:
            return map_obj.spatial_objects[location_id].name
        elif location_id in map_obj.regions:
            return map_obj.regions[location_id].name
        
        return f"Location {location_id}"
    
    def _calculate_heading(self, start: SpatialCoordinate, end: SpatialCoordinate) -> str:
        """Calculate cardinal direction between points"""
        dx = end.x - start.x
        dy = end.y - start.y
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Convert to cardinal direction
        if -22.5 <= angle < 22.5:
            return "east"
        elif 22.5 <= angle < 67.5:
            return "northeast"
        elif 67.5 <= angle < 112.5:
            return "north"
        elif 112.5 <= angle < 157.5:
            return "northwest"
        elif 157.5 <= angle or angle < -157.5:
            return "west"
        elif -157.5 <= angle < -112.5:
            return "southwest"
        elif -112.5 <= angle < -67.5:
            return "south"
        else:  # -67.5 <= angle < -22.5
            return "southeast"
    
    async def _find_nearby_landmark(self, map_id: str, position: SpatialCoordinate, 
                               max_distance: float = 5.0) -> Optional[str]:
        """Find a nearby landmark to use as a reference"""
        map_obj = self.maps[map_id]
        
        closest_landmark = None
        closest_distance = float('inf')
        
        for landmark_id in map_obj.landmarks:
            if landmark_id in map_obj.spatial_objects:
                landmark = map_obj.spatial_objects[landmark_id]
                distance = position.distance_to(landmark.coordinates)
                
                if distance < max_distance and distance < closest_distance:
                    closest_landmark = landmark
                    closest_distance = distance
        
        if closest_landmark:
            return closest_landmark.name
        
        return None
    
    @function_tool
    async def calculate_directions(self, map_id: str, start_id: str, end_id: str) -> Union[List[str], List[ErrorResult]]:
        """
        Calculate human-readable directions between two locations
        
        Args:
            map_id: ID of the map
            start_id: ID of the starting object or region
            end_id: ID of the ending object or region
            
        Returns:
            List of direction steps
        """
        # Find or create a path
        path_result = await self.find_path(map_id, start_id, end_id)
        
        if isinstance(path_result, ErrorResult):
            return [path_result]
        
        return path_result.directions
    
    @function_tool
    async def get_nearest_landmarks(self, map_id: str, location_id: str, 
                                 max_count: int = 3) -> Union[List[LandmarkInfo], List[ErrorResult]]:
        """
        Find the nearest landmarks to a location
        
        Args:
            map_id: ID of the map
            location_id: ID of the location (object or region)
            max_count: Maximum number of landmarks to return
            
        Returns:
            List of nearby landmarks with distances
        """
        with custom_span("get_nearest_landmarks", {"map_id": map_id, "location": location_id}):
            # Check if map exists
            if map_id not in self.maps:
                return [ErrorResult(error=f"Map {map_id} not found")]
            
            # Get coordinates of the location
            location_coord = await self._get_location_coordinates(map_id, location_id)
            
            if not location_coord:
                return [ErrorResult(error=f"Location {location_id} not found or has no coordinates")]
            
            map_obj = self.maps[map_id]
            landmarks = []
            
            # Calculate distances to all landmarks
            for landmark_id in map_obj.landmarks:
                if landmark_id in map_obj.spatial_objects:
                    landmark = map_obj.spatial_objects[landmark_id]
                    distance = location_coord.distance_to(landmark.coordinates)
                    
                    landmarks.append(LandmarkInfo(
                        id=landmark_id,
                        name=landmark.name,
                        distance=distance,
                        direction=self._calculate_heading(location_coord, landmark.coordinates)
                    ))
            
            # Sort by distance and return top N
            landmarks.sort(key=lambda x: x.distance)
            return landmarks[:max_count]
    
    @function_tool
    async def merge_regions(self, map_id: str, region_ids: List[str], 
                          new_name: str, new_type: Optional[str] = None) -> Union[MergeRegionsResult, ErrorResult]:
        """
        Merge multiple regions into a single larger region using proper polygon union.
        
        Args:
            map_id: ID of the map
            region_ids: List of region IDs to merge
            new_name: Name for the merged region
            new_type: Optional type for the merged region
            
        Returns:
            Information about the new merged region
        """
        with custom_span("merge_regions", {"map_id": map_id, "count": len(region_ids)}):
            # Validate inputs
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            map_obj = self.maps[map_id]
            
            if len(region_ids) < 2:
                return ErrorResult(error="Need at least two regions to merge")
            
            # Validate all regions exist and collect their data
            regions_to_merge = []
            for region_id in region_ids:
                if region_id not in map_obj.regions:
                    return ErrorResult(error=f"Region {region_id} not found in map")
                regions_to_merge.append(map_obj.regions[region_id])
            
            # Validate regions are valid polygons
            for region in regions_to_merge:
                if len(region.boundary_points) < 3:
                    return ErrorResult(error=f"Region {region.id} has insufficient boundary points")
            
            try:
                # Compute the union of all regions
                merged_boundary = await self._compute_regions_union(regions_to_merge)
                
                if not merged_boundary or len(merged_boundary) < 3:
                    return ErrorResult(error="Failed to compute valid merged boundary")
                
                # Merge properties intelligently
                combined_properties = await self._merge_region_properties(regions_to_merge)
                
                # Determine navigability (AND operation - all must be navigable)
                is_navigable = all(region.is_navigable for region in regions_to_merge)
                
                # Collect all contained objects (union with deduplication)
                contained_objects = []
                seen_objects = set()
                for region in regions_to_merge:
                    for obj_id in region.contained_objects:
                        if obj_id not in seen_objects:
                            contained_objects.append(obj_id)
                            seen_objects.add(obj_id)
                
                # Find adjacent regions (excluding the ones being merged)
                merged_region_ids_set = set(region_ids)
                adjacent_regions = []
                seen_adjacent = set()
                
                for region in regions_to_merge:
                    for adj_id in region.adjacent_regions:
                        if adj_id not in merged_region_ids_set and adj_id not in seen_adjacent:
                            # Verify the adjacent region still exists
                            if adj_id in map_obj.regions:
                                adjacent_regions.append(adj_id)
                                seen_adjacent.add(adj_id)
                
                # Generate unique region ID
                merged_region_id = f"region_{uuid.uuid4().hex[:8]}"
                
                # Determine region type
                if new_type is None:
                    # Use the most common type among merged regions
                    type_counts = {}
                    for region in regions_to_merge:
                        type_counts[region.region_type] = type_counts.get(region.region_type, 0) + 1
                    new_type = max(type_counts, key=type_counts.get)
                
                # Calculate confidence as weighted average based on region areas
                total_area = 0.0
                weighted_confidence = 0.0
                for region in regions_to_merge:
                    area = self._calculate_polygon_area(region.boundary_points)
                    total_area += area
                    weighted_confidence += region.confidence * area
                
                merged_confidence = weighted_confidence / total_area if total_area > 0 else 0.5
                
                # Create the merged region
                merged_region = SpatialRegion(
                    id=merged_region_id,
                    name=new_name,
                    region_type=new_type,
                    boundary_points=merged_boundary,
                    is_navigable=is_navigable,
                    contained_objects=contained_objects,
                    adjacent_regions=adjacent_regions,
                    properties=combined_properties,
                    confidence=merged_confidence
                )
                
                # Validate the merged region
                if not self._is_valid_region(merged_region):
                    return ErrorResult(error="Merged region failed validation")
                
                # Update the map atomically
                async with self._lock:
                    # Add the new merged region
                    map_obj.regions[merged_region_id] = merged_region
                    
                    # Remove the old regions
                    for region_id in region_ids:
                        del map_obj.regions[region_id]
                    
                    # Update references in adjacent regions
                    for adj_region_id in adjacent_regions:
                        if adj_region_id in map_obj.regions:
                            adj_region = map_obj.regions[adj_region_id]
                            # Remove references to old regions
                            adj_region.adjacent_regions = [
                                r for r in adj_region.adjacent_regions 
                                if r not in merged_region_ids_set
                            ]
                            # Add reference to new merged region
                            if merged_region_id not in adj_region.adjacent_regions:
                                adj_region.adjacent_regions.append(merged_region_id)
                    
                    # Update routes that referenced the old regions
                    await self._update_routes_after_region_merge(
                        map_obj, region_ids, merged_region_id
                    )
                    
                    # Update objects to ensure they know they're in the merged region
                    for obj_id in contained_objects:
                        if obj_id in map_obj.spatial_objects:
                            # This is already handled by contained_objects list
                            pass
                    
                    # Update map metadata
                    map_obj.last_updated = datetime.datetime.now().isoformat()
                    
                    # Recalculate map completeness
                    await self._update_map_completeness(map_id)
                
                logger.info(f"Successfully merged {len(region_ids)} regions into '{new_name}' in map {map_id}")
                
                return MergeRegionsResult(
                    region_id=merged_region_id,
                    name=new_name,
                    region_type=merged_region.region_type,
                    contained_objects_count=len(merged_region.contained_objects),
                    adjacent_regions_count=len(merged_region.adjacent_regions),
                    is_navigable=merged_region.is_navigable
                )
                
            except Exception as e:
                logger.error(f"Error merging regions: {str(e)}")
                return ErrorResult(error=f"Failed to merge regions: {str(e)}")
    
    async def _compute_regions_union(self, regions: List[SpatialRegion]) -> List[SpatialCoordinate]:
        """
        Compute the union of multiple regions using a robust polygon union algorithm.
        Returns the boundary points of the merged region.
        """
        if not regions:
            return []
        
        if len(regions) == 1:
            return regions[0].boundary_points.copy()
        
        # Convert regions to a format suitable for union operation
        polygons = []
        for region in regions:
            polygon = self._boundary_points_to_polygon(region.boundary_points)
            if polygon:
                polygons.append(polygon)
        
        if not polygons:
            return []
        
        # Compute union iteratively for robustness
        result_polygon = polygons[0]
        
        for i in range(1, len(polygons)):
            result_polygon = self._polygon_union(result_polygon, polygons[i])
            if not result_polygon:
                # If union fails, fall back to convex hull
                all_points = []
                for region in regions:
                    all_points.extend(region.boundary_points)
                return self._compute_convex_hull(all_points)
        
        # Convert back to boundary points
        return self._polygon_to_boundary_points(result_polygon)
    
    def _boundary_points_to_polygon(self, points: List[SpatialCoordinate]) -> Optional[List[Tuple[float, float]]]:
        """Convert boundary points to polygon representation for geometric operations"""
        if len(points) < 3:
            return None
        
        # Ensure the polygon is properly oriented (counter-clockwise)
        polygon = [(p.x, p.y) for p in points]
        
        # Check if polygon is clockwise and reverse if needed
        if self._is_clockwise(polygon):
            polygon.reverse()
        
        return polygon
    
    def _polygon_to_boundary_points(self, polygon: List[Tuple[float, float]]) -> List[SpatialCoordinate]:
        """Convert polygon representation back to boundary points"""
        return [
            SpatialCoordinate(x=x, y=y, reference_frame="global")
            for x, y in polygon
        ]
    
    def _is_clockwise(self, polygon: List[Tuple[float, float]]) -> bool:
        """Check if a polygon is oriented clockwise using the shoelace formula"""
        if len(polygon) < 3:
            return False
        
        # Calculate the signed area
        area = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += (polygon[j][0] - polygon[i][0]) * (polygon[j][1] + polygon[i][1])
        
        return area > 0
    
    def _polygon_union(self, poly1: List[Tuple[float, float]], 
                      poly2: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """
        Compute the union of two polygons using the Sutherland-Hodgman algorithm
        extended for union operations.
        """
        # First check if one polygon contains the other
        if self._polygon_contains_polygon(poly1, poly2):
            return poly1.copy()
        if self._polygon_contains_polygon(poly2, poly1):
            return poly2.copy()
        
        # Find all intersection points
        intersections = self._find_all_polygon_intersections(poly1, poly2)
        
        # If no intersections, polygons are disjoint
        if not intersections:
            # Check if they're truly disjoint or one is inside the other
            if self._point_in_polygon(poly1[0], poly2):
                return poly2.copy()
            elif self._point_in_polygon(poly2[0], poly1):
                return poly1.copy()
            else:
                # Disjoint polygons - return convex hull of both
                return self._compute_convex_hull_of_points(poly1 + poly2)
        
        # Build the union using a modified Weiler-Atherton algorithm
        return self._weiler_atherton_union(poly1, poly2, intersections)
    
    def _polygon_contains_polygon(self, outer: List[Tuple[float, float]], 
                                inner: List[Tuple[float, float]]) -> bool:
        """Check if outer polygon completely contains inner polygon"""
        # All vertices of inner must be inside outer
        for vertex in inner:
            if not self._point_in_polygon(vertex, outer):
                return False
        
        # Also check that no edges intersect (to handle cases where inner
        # vertices are inside but edges cross)
        for i in range(len(inner)):
            j = (i + 1) % len(inner)
            for k in range(len(outer)):
                l = (k + 1) % len(outer)
                if self._line_segments_intersect_2d(
                    inner[i], inner[j], outer[k], outer[l]
                ):
                    return False
        
        return True
    
    def _point_in_polygon(self, point: Tuple[float, float], 
                         polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside a polygon using the ray casting algorithm.
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def _find_all_polygon_intersections(self, poly1: List[Tuple[float, float]], 
                                      poly2: List[Tuple[float, float]]) -> List[Dict]:
        """Find all intersection points between two polygons"""
        intersections = []
        
        for i in range(len(poly1)):
            j = (i + 1) % len(poly1)
            edge1 = (poly1[i], poly1[j])
            
            for k in range(len(poly2)):
                l = (k + 1) % len(poly2)
                edge2 = (poly2[k], poly2[l])
                
                intersection = self._line_segment_intersection_2d(
                    edge1[0], edge1[1], edge2[0], edge2[1]
                )
                
                if intersection:
                    intersections.append({
                        'point': intersection,
                        'edge1_index': i,
                        'edge2_index': k,
                        'param1': self._get_edge_parameter(edge1[0], edge1[1], intersection),
                        'param2': self._get_edge_parameter(edge2[0], edge2[1], intersection)
                    })
        
        return intersections
    
    def _line_segments_intersect_2d(self, p1: Tuple[float, float], p2: Tuple[float, float],
                                   p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two 2D line segments intersect"""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _line_segment_intersection_2d(self, p1: Tuple[float, float], p2: Tuple[float, float],
                                    p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find the intersection point of two 2D line segments, if it exists"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection exists
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def _get_edge_parameter(self, start: Tuple[float, float], end: Tuple[float, float], 
                           point: Tuple[float, float]) -> float:
        """Get the parameter t for where point lies on the edge from start to end"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if abs(dx) > abs(dy):
            return (point[0] - start[0]) / dx if abs(dx) > 1e-10 else 0
        else:
            return (point[1] - start[1]) / dy if abs(dy) > 1e-10 else 0
    
    def _weiler_atherton_union(self, poly1: List[Tuple[float, float]], 
                             poly2: List[Tuple[float, float]], 
                             intersections: List[Dict]) -> List[Tuple[float, float]]:
        """
        Compute polygon union using the Weiler-Atherton algorithm.
        This is a complex algorithm that handles all cases including holes.
        """
        # Build vertex lists with intersections inserted
        vertices1 = self._build_vertex_list(poly1, intersections, 1)
        vertices2 = self._build_vertex_list(poly2, intersections, 2)
        
        # Mark entry/exit points
        self._mark_entry_exit_points(vertices1, vertices2, poly1, poly2)
        
        # Trace the union boundary
        result = self._trace_union_boundary(vertices1, vertices2)
        
        if not result:
            # Fallback to convex hull if tracing fails
            return self._compute_convex_hull_of_points(poly1 + poly2)
        
        return result
    
    def _build_vertex_list(self, polygon: List[Tuple[float, float]], 
                          intersections: List[Dict], 
                          polygon_index: int) -> List[Dict]:
        """Build a vertex list with intersections inserted in order"""
        vertices = []
        
        for i in range(len(polygon)):
            # Add original vertex
            vertices.append({
                'point': polygon[i],
                'is_intersection': False,
                'original_index': i
            })
            
            # Add any intersections on the edge from i to i+1
            edge_intersections = []
            for inter in intersections:
                if polygon_index == 1 and inter['edge1_index'] == i:
                    edge_intersections.append(inter)
                elif polygon_index == 2 and inter['edge2_index'] == i:
                    edge_intersections.append(inter)
            
            # Sort intersections by parameter along edge
            key = 'param1' if polygon_index == 1 else 'param2'
            edge_intersections.sort(key=lambda x: x[key])
            
            # Add sorted intersections
            for inter in edge_intersections:
                vertices.append({
                    'point': inter['point'],
                    'is_intersection': True,
                    'intersection_data': inter
                })
        
        return vertices
    
    def _mark_entry_exit_points(self, vertices1: List[Dict], vertices2: List[Dict],
                              poly1: List[Tuple[float, float]], 
                              poly2: List[Tuple[float, float]]) -> None:
        """Mark intersection points as entry or exit points"""
        for v in vertices1:
            if v['is_intersection']:
                # Check the direction of the edge after this intersection
                next_idx = (vertices1.index(v) + 1) % len(vertices1)
                next_point = vertices1[next_idx]['point']
                
                # A point is an entry if moving from outside to inside the other polygon
                mid_point = (
                    (v['point'][0] + next_point[0]) / 2,
                    (v['point'][1] + next_point[1]) / 2
                )
                
                v['is_entry'] = self._point_in_polygon(mid_point, poly2)
        
        # Do the same for vertices2
        for v in vertices2:
            if v['is_intersection']:
                next_idx = (vertices2.index(v) + 1) % len(vertices2)
                next_point = vertices2[next_idx]['point']
                
                mid_point = (
                    (v['point'][0] + next_point[0]) / 2,
                    (v['point'][1] + next_point[1]) / 2
                )
                
                v['is_entry'] = self._point_in_polygon(mid_point, poly1)
    
    def _trace_union_boundary(self, vertices1: List[Dict], 
                            vertices2: List[Dict]) -> List[Tuple[float, float]]:
        """Trace the boundary of the union polygon"""
        result = []
        visited = set()
        
        # Start from a vertex that's definitely on the union boundary
        start_vertex = None
        for v in vertices1:
            if not v['is_intersection']:
                # Check if this vertex is outside the other polygon
                if not self._point_in_polygon(v['point'], [u['point'] for u in vertices2]):
                    start_vertex = v
                    break
        
        if not start_vertex:
            # All vertices of poly1 are inside poly2
            return [v['point'] for v in vertices2]
        
        # Trace the boundary
        current_list = vertices1
        current_idx = vertices1.index(start_vertex)
        other_list = vertices2
        
        max_iterations = len(vertices1) + len(vertices2) + len(visited)
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            current = current_list[current_idx]
            if current['point'] in visited:
                break
                
            result.append(current['point'])
            visited.add(current['point'])
            
            # Move to next vertex
            current_idx = (current_idx + 1) % len(current_list)
            next_vertex = current_list[current_idx]
            
            # If we hit an entry intersection, switch to the other polygon
            if next_vertex.get('is_intersection') and next_vertex.get('is_entry'):
                # Find corresponding intersection in other list
                for i, v in enumerate(other_list):
                    if v.get('is_intersection') and v['point'] == next_vertex['point']:
                        current_list, other_list = other_list, current_list
                        current_idx = i
                        break
        
        return result if len(result) >= 3 else []
    
    def _compute_convex_hull(self, points: List[SpatialCoordinate]) -> List[SpatialCoordinate]:
        """Compute convex hull of points using Graham's scan algorithm"""
        if len(points) < 3:
            return points
        
        # Convert to tuples for easier processing
        point_tuples = [(p.x, p.y) for p in points]
        hull_tuples = self._compute_convex_hull_of_points(point_tuples)
        
        # Convert back to SpatialCoordinate
        return [
            SpatialCoordinate(x=x, y=y, reference_frame=points[0].reference_frame)
            for x, y in hull_tuples
        ]
    
    def _compute_convex_hull_of_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Compute convex hull using Graham's scan algorithm"""
        if len(points) < 3:
            return points
        
        # Remove duplicates
        unique_points = list(set(points))
        if len(unique_points) < 3:
            return unique_points
        
        # Find the point with lowest y-coordinate (and leftmost if tie)
        start = min(unique_points, key=lambda p: (p[1], p[0]))
        
        # Sort points by polar angle with respect to start point
        def polar_angle(p):
            dx = p[0] - start[0]
            dy = p[1] - start[1]
            return np.arctan2(dy, dx), dx*dx + dy*dy  # angle and distance squared
        
        sorted_points = sorted(unique_points, key=polar_angle)
        
        # Build the hull
        hull = []
        
        for p in sorted_points:
            # Remove points that make clockwise turn
            while len(hull) > 1 and self._cross_product_2d(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        
        return hull
    
    def _cross_product_2d(self, o: Tuple[float, float], a: Tuple[float, float], 
                         b: Tuple[float, float]) -> float:
        """2D cross product of vectors OA and OB"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    async def _merge_region_properties(self, regions: List[SpatialRegion]) -> PropertiesDict:
        """Intelligently merge properties from multiple regions"""
        merged_props = PropertiesDict()
        
        # Collect all descriptions and categories
        descriptions = []
        categories = set()
        colors = []
        materials = set()
        
        total_weight = 0.0
        weighted_temp = 0.0
        
        for region in regions:
            props = region.properties
            
            if props.description:
                descriptions.append(props.description)
            
            if props.category:
                categories.add(props.category)
            
            if props.color:
                colors.append(props.color)
            
            if props.material:
                materials.add(props.material)
            
            if props.temperature is not None:
                # Weight by region size
                area = self._calculate_polygon_area(region.boundary_points)
                weighted_temp += props.temperature * area
                total_weight += area
            
            if props.weight is not None:
                # For weight, sum them up
                if merged_props.weight is None:
                    merged_props.weight = 0.0
                merged_props.weight += props.weight
        
        # Merge descriptions
        if descriptions:
            if len(descriptions) == 1:
                merged_props.description = descriptions[0]
            else:
                merged_props.description = f"Merged region combining: {'; '.join(descriptions)}"
        
        # Merge categories
        if categories:
            if len(categories) == 1:
                merged_props.category = list(categories)[0]
            else:
                merged_props.category = f"Mixed ({', '.join(sorted(categories))})"
        
        # For color, use the most common one
        if colors:
            from collections import Counter
            color_counts = Counter(colors)
            merged_props.color = color_counts.most_common(1)[0][0]
        
        # For materials, list all unique ones
        if materials:
            if len(materials) == 1:
                merged_props.material = list(materials)[0]
            else:
                merged_props.material = f"Multiple: {', '.join(sorted(materials))}"
        
        # Calculate weighted average temperature
        if total_weight > 0:
            merged_props.temperature = weighted_temp / total_weight
        
        # Status defaults to "merged"
        merged_props.status = "merged"
        
        return merged_props
    
    def _calculate_polygon_area(self, points: List[SpatialCoordinate]) -> float:
        """Calculate the area of a polygon using the shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        
        return abs(area) / 2.0
    
    def _is_valid_region(self, region: SpatialRegion) -> bool:
        """Validate that a region is geometrically valid"""
        if len(region.boundary_points) < 3:
            return False
        
        # Check for self-intersections
        edges = self._get_edges(region.boundary_points)
        for i in range(len(edges)):
            for j in range(i + 2, len(edges)):
                # Skip adjacent edges
                if j == len(edges) - 1 and i == 0:
                    continue
                    
                edge1 = edges[i]
                edge2 = edges[j]
                
                if self._line_segments_intersect(
                    edge1[0], edge1[1], edge2[0], edge2[1]
                ):
                    return False
        
        # Check that area is positive
        area = self._calculate_polygon_area(region.boundary_points)
        if area <= 1e-10:
            return False
        
        return True
    
    async def _update_routes_after_region_merge(self, map_obj: CognitiveMap, 
                                              old_region_ids: List[str], 
                                              new_region_id: str) -> None:
        """Update routes that referenced the old regions to reference the new merged region"""
        for route_id, route in map_obj.routes.items():
            updated = False
            
            if route.start_id in old_region_ids:
                route.start_id = new_region_id
                updated = True
            
            if route.end_id in old_region_ids:
                route.end_id = new_region_id
                updated = True
            
            if updated:
                # Update route name if it was auto-generated
                if route.name and " to " in route.name:
                    start_name = self._get_location_name(map_obj.id, route.start_id)
                    end_name = self._get_location_name(map_obj.id, route.end_id)
                    route.name = f"Route from {start_name} to {end_name}"
    
    @function_tool
    async def identify_landmarks(self, map_id: str, max_count: int = 5) -> Union[List[IdentifiedLandmark], List[ErrorResult]]:
        """
        Identify important landmarks in a map
        
        Args:
            map_id: ID of the map
            max_count: Maximum number of landmarks to identify
            
        Returns:
            List of identified landmarks
        """
        with custom_span("identify_landmarks", {"map_id": map_id}):
            # Check if map exists
            if map_id not in self.maps:
                return [ErrorResult(error=f"Map {map_id} not found")]
            
            map_obj = self.maps[map_id]
            
            # Scoring algorithm for landmark potential
            landmark_scores = {}
            
            for obj_id, obj in map_obj.spatial_objects.items():
                if obj.is_landmark:
                    continue  # Skip objects already marked as landmarks
                
                # Initial score based on object properties
                score = 0.0
                
                # Score based on visibility
                score += obj.visibility * 2.0
                
                # Score based on observation count
                score += min(obj.observation_count / 3.0, 1.0)
                
                # Score based on connections (more connected = better landmark)
                score += min(len(obj.connections) / 3.0, 1.0)
                
                # Score based on size if available
                if obj.size:
                    size_values = []
                    if obj.size.width:
                        size_values.append(obj.size.width)
                    if obj.size.height:
                        size_values.append(obj.size.height)
                    if obj.size.depth:
                        size_values.append(obj.size.depth)
                    
                    if size_values:
                        avg_size = sum(size_values) / len(size_values)
                        score += min(avg_size / 5.0, 1.0)  # Larger objects make better landmarks
                
                # Score based on being at region intersections
                containing_regions = []
                for region_id, region in map_obj.regions.items():
                    if obj_id in region.contained_objects:
                        containing_regions.append(region_id)
                
                if len(containing_regions) > 1:
                    score += 1.0  # Being at intersection of regions is good for a landmark
                
                # Store score
                landmark_scores[obj_id] = score
            
            # Sort by score
            sorted_landmarks = sorted(landmark_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Take top N and mark as landmarks
            new_landmarks = []
            for obj_id, score in sorted_landmarks[:max_count]:
                obj = map_obj.spatial_objects[obj_id]
                
                # Update object
                async with self._lock:
                    obj.is_landmark = True
                    if obj_id not in map_obj.landmarks:
                        map_obj.landmarks.append(obj_id)
                
                # Add to result
                new_landmarks.append(IdentifiedLandmark(
                    id=obj_id,
                    name=obj.name,
                    object_type=obj.object_type,
                    landmark_score=score
                ))
            
            logger.info(f"Identified {len(new_landmarks)} new landmarks in map {map_id}")
            
            return new_landmarks
    
    @function_tool
    async def calculate_region_connections(self, map_id: str) -> Union[RegionConnectionsResult, ErrorResult]:
        """
        Calculate connections between regions
        
        Args:
            map_id: ID of the map
            
        Returns:
            Information about region connections
        """
        with custom_span("calculate_region_connections", {"map_id": map_id}):
            # Check if map exists
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            map_obj = self.maps[map_id]
            
            # Reset all adjacency information
            for region_id in map_obj.regions:
                map_obj.regions[region_id].adjacent_regions = []
            
            # Check each pair of regions for adjacency
            regions = list(map_obj.regions.values())
            connections_count = 0
            
            for i in range(len(regions)):
                for j in range(i+1, len(regions)):
                    region1 = regions[i]
                    region2 = regions[j]
                    
                    # Check for adjacency
                    is_adjacent = await self._are_regions_adjacent(region1, region2)
                    
                    if is_adjacent:
                        region1.adjacent_regions.append(region2.id)
                        region2.adjacent_regions.append(region1.id)
                        connections_count += 1
            
            # Update map
            async with self._lock:
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
            
            logger.info(f"Calculated {connections_count} region connections in map {map_id}")
            
            return RegionConnectionsResult(
                connections_count=connections_count,
                regions_count=len(regions),
                updated_date=self.maps[map_id].last_updated
            )
    
    async def _are_regions_adjacent(self, region1: SpatialRegion, region2: SpatialRegion) -> bool:
        """
        Determine if two regions are adjacent using robust computational geometry.
        Two regions are considered adjacent if they share at least one edge or vertex.
        """
        # Configuration for numerical comparisons
        EPSILON = 1e-10  # For floating point comparisons
        VERTEX_THRESHOLD = 0.01  # Maximum distance to consider vertices shared
        EDGE_THRESHOLD = 0.1  # Maximum distance to consider edges adjacent
        MIN_OVERLAP_RATIO = 0.1  # Minimum overlap ratio for shared edges
        
        # Early exit if regions don't have enough points
        if len(region1.boundary_points) < 3 or len(region2.boundary_points) < 3:
            return False
        
        # Compute bounding boxes for quick rejection test
        bbox1 = self._compute_bounding_box(region1.boundary_points)
        bbox2 = self._compute_bounding_box(region2.boundary_points)
        
        # Add small buffer for adjacency check
        buffer = max(VERTEX_THRESHOLD, EDGE_THRESHOLD) * 2
        
        # If bounding boxes don't overlap (with buffer), regions can't be adjacent
        if not self._bounding_boxes_overlap(bbox1, bbox2, buffer):
            return False
        
        # Check for shared vertices
        shared_vertices = self._find_shared_vertices(
            region1.boundary_points, 
            region2.boundary_points, 
            VERTEX_THRESHOLD
        )
        
        if len(shared_vertices) > 0:
            # If regions share at least one vertex, check if they share an edge
            # or if they only touch at corners
            shared_edges = self._find_shared_edges(
                region1.boundary_points,
                region2.boundary_points,
                EDGE_THRESHOLD,
                MIN_OVERLAP_RATIO,
                EPSILON
            )
            
            if len(shared_edges) > 0:
                return True
            
            # Check if shared vertices form a valid adjacency (not just corner touch)
            # For corner-only touch, we need exactly one shared vertex
            if len(shared_vertices) == 1:
                # Check if this is a T-junction or proper corner adjacency
                return self._is_valid_corner_adjacency(
                    region1.boundary_points,
                    region2.boundary_points,
                    shared_vertices[0],
                    EDGE_THRESHOLD
                )
        
        # Check for edge adjacency without shared vertices (parallel edges)
        edges1 = self._get_edges(region1.boundary_points)
        edges2 = self._get_edges(region2.boundary_points)
        
        for edge1 in edges1:
            for edge2 in edges2:
                if self._edges_are_adjacent(edge1, edge2, EDGE_THRESHOLD, MIN_OVERLAP_RATIO, EPSILON):
                    return True
        
        return False
    
    def _compute_bounding_box(self, points: List[SpatialCoordinate]) -> Dict[str, float]:
        """Compute axis-aligned bounding box for a set of points"""
        if not points:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
        
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        
        # Handle 3D case
        if all(p.z is not None for p in points):
            min_z = min(p.z for p in points)
            max_z = max(p.z for p in points)
        else:
            min_z = max_z = 0
        
        return {
            "min_x": min_x, "max_x": max_x,
            "min_y": min_y, "max_y": max_y,
            "min_z": min_z, "max_z": max_z
        }
    
    def _bounding_boxes_overlap(self, bbox1: Dict[str, float], bbox2: Dict[str, float], buffer: float) -> bool:
        """Check if two bounding boxes overlap with given buffer"""
        return not (
            bbox1["max_x"] + buffer < bbox2["min_x"] or
            bbox2["max_x"] + buffer < bbox1["min_x"] or
            bbox1["max_y"] + buffer < bbox2["min_y"] or
            bbox2["max_y"] + buffer < bbox1["min_y"]
        )
    
    def _find_shared_vertices(self, points1: List[SpatialCoordinate], 
                             points2: List[SpatialCoordinate], 
                             threshold: float) -> List[Tuple[int, int]]:
        """Find vertices that are shared between two polygons"""
        shared = []
        
        for i, p1 in enumerate(points1):
            for j, p2 in enumerate(points2):
                if p1.distance_to(p2) < threshold:
                    shared.append((i, j))
        
        return shared
    
    def _get_edges(self, points: List[SpatialCoordinate]) -> List[Tuple[SpatialCoordinate, SpatialCoordinate]]:
        """Get all edges of a polygon"""
        edges = []
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            edges.append((points[i], points[j]))
        
        return edges
    
    def _find_shared_edges(self, points1: List[SpatialCoordinate], 
                          points2: List[SpatialCoordinate],
                          edge_threshold: float,
                          min_overlap_ratio: float,
                          epsilon: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find edges that are shared between two polygons"""
        shared_edges = []
        edges1 = self._get_edges(points1)
        edges2 = self._get_edges(points2)
        
        for i, edge1 in enumerate(edges1):
            edge1_start_idx = i
            edge1_end_idx = (i + 1) % len(points1)
            
            for j, edge2 in enumerate(edges2):
                edge2_start_idx = j
                edge2_end_idx = (j + 1) % len(points2)
                
                if self._edges_are_collinear_and_overlapping(
                    edge1, edge2, edge_threshold, min_overlap_ratio, epsilon
                ):
                    shared_edges.append((
                        (edge1_start_idx, edge1_end_idx),
                        (edge2_start_idx, edge2_end_idx)
                    ))
        
        return shared_edges
    
    def _edges_are_adjacent(self, edge1: Tuple[SpatialCoordinate, SpatialCoordinate],
                           edge2: Tuple[SpatialCoordinate, SpatialCoordinate],
                           threshold: float, min_overlap_ratio: float, epsilon: float) -> bool:
        """Check if two edges are adjacent (close, parallel, and overlapping)"""
        # First check if edges are parallel
        if not self._edges_are_parallel(edge1, edge2, epsilon):
            return False
        
        # Check if edges are close enough
        if not self._edges_are_close(edge1, edge2, threshold):
            return False
        
        # Check if edges overlap sufficiently
        return self._edges_overlap_sufficiently(edge1, edge2, min_overlap_ratio, epsilon)
    
    def _edges_are_parallel(self, edge1: Tuple[SpatialCoordinate, SpatialCoordinate],
                           edge2: Tuple[SpatialCoordinate, SpatialCoordinate],
                           epsilon: float) -> bool:
        """Check if two edges are parallel within epsilon tolerance"""
        p1, p2 = edge1
        p3, p4 = edge2
        
        # Calculate direction vectors
        dir1 = self._normalize_vector((p2.x - p1.x, p2.y - p1.y))
        dir2 = self._normalize_vector((p4.x - p3.x, p4.y - p3.y))
        
        if dir1 is None or dir2 is None:
            return False
        
        # Check if directions are parallel (dot product close to ±1)
        dot_product = abs(dir1[0] * dir2[0] + dir1[1] * dir2[1])
        return abs(dot_product - 1.0) < epsilon
    
    def _edges_are_collinear_and_overlapping(self, edge1: Tuple[SpatialCoordinate, SpatialCoordinate],
                                            edge2: Tuple[SpatialCoordinate, SpatialCoordinate],
                                            threshold: float, min_overlap_ratio: float,
                                            epsilon: float) -> bool:
        """Check if two edges are collinear and overlapping"""
        # Check if edges are collinear
        if not self._edges_are_collinear(edge1, edge2, threshold, epsilon):
            return False
        
        # Check overlap
        return self._edges_overlap_sufficiently(edge1, edge2, min_overlap_ratio, epsilon)
    
    def _edges_are_collinear(self, edge1: Tuple[SpatialCoordinate, SpatialCoordinate],
                            edge2: Tuple[SpatialCoordinate, SpatialCoordinate],
                            threshold: float, epsilon: float) -> bool:
        """Check if two edges are collinear (on the same line)"""
        p1, p2 = edge1
        p3, p4 = edge2
        
        # Check if all four points are collinear
        # Use the cross product method: if points are collinear, cross product is zero
        
        # Check if p3 is on the line defined by p1-p2
        dist1 = self._point_to_line_distance(p3, p1, p2)
        if dist1 > threshold:
            return False
        
        # Check if p4 is on the line defined by p1-p2
        dist2 = self._point_to_line_distance(p4, p1, p2)
        if dist2 > threshold:
            return False
        
        return True
    
    def _edges_are_close(self, edge1: Tuple[SpatialCoordinate, SpatialCoordinate],
                        edge2: Tuple[SpatialCoordinate, SpatialCoordinate],
                        threshold: float) -> bool:
        """Check if two edges are within threshold distance"""
        p1, p2 = edge1
        p3, p4 = edge2
        
        # Check minimum distance between the line segments
        min_dist = self._line_segment_distance(p1, p2, p3, p4)
        return min_dist < threshold
    
    def _edges_overlap_sufficiently(self, edge1: Tuple[SpatialCoordinate, SpatialCoordinate],
                                   edge2: Tuple[SpatialCoordinate, SpatialCoordinate],
                                   min_overlap_ratio: float, epsilon: float) -> bool:
        """Check if two edges have sufficient overlap"""
        p1, p2 = edge1
        p3, p4 = edge2
        
        # Project edges onto a common axis
        # Use the longer edge as the reference axis
        edge1_len = p1.distance_to(p2)
        edge2_len = p3.distance_to(p4)
        
        if edge1_len < epsilon and edge2_len < epsilon:
            # Both edges are essentially points
            return False
        
        # Use the longer edge as reference
        if edge1_len >= edge2_len:
            ref_start, ref_end = p1, p2
            other_start, other_end = p3, p4
            ref_len = edge1_len
        else:
            ref_start, ref_end = p3, p4
            other_start, other_end = p1, p2
            ref_len = edge2_len
        
        # Project other edge onto reference edge
        proj_start = self._project_point_onto_line_segment(other_start, ref_start, ref_end)
        proj_end = self._project_point_onto_line_segment(other_end, ref_start, ref_end)
        
        # Calculate overlap length
        overlap_length = self._calculate_overlap_length(
            ref_start, ref_end, proj_start, proj_end
        )
        
        # Check if overlap is sufficient
        min_edge_len = min(edge1_len, edge2_len)
        return overlap_length > min_overlap_ratio * min_edge_len
    
    def _normalize_vector(self, vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Normalize a 2D vector"""
        length = np.sqrt(vec[0]**2 + vec[1]**2)
        if length < 1e-10:
            return None
        return (vec[0] / length, vec[1] / length)
    
    def _point_to_line_distance(self, point: SpatialCoordinate,
                              line_start: SpatialCoordinate,
                              line_end: SpatialCoordinate) -> float:
        """Calculate perpendicular distance from point to infinite line"""
        # Using the formula: |ax + by + c| / sqrt(a² + b²)
        # where the line is defined by two points
        
        x0, y0 = point.x, point.y
        x1, y1 = line_start.x, line_start.y
        x2, y2 = line_end.x, line_end.y
        
        # Handle the case where line_start == line_end
        if abs(x2 - x1) < 1e-10 and abs(y2 - y1) < 1e-10:
            return point.distance_to(line_start)
        
        # Calculate the perpendicular distance
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        return numerator / denominator
    
    def _line_segment_distance(self, p1: SpatialCoordinate, p2: SpatialCoordinate,
                             p3: SpatialCoordinate, p4: SpatialCoordinate) -> float:
        """Calculate minimum distance between two line segments"""
        # Check all possible closest points
        distances = [
            self._point_to_line_segment_distance(p1, p3, p4),
            self._point_to_line_segment_distance(p2, p3, p4),
            self._point_to_line_segment_distance(p3, p1, p2),
            self._point_to_line_segment_distance(p4, p1, p2)
        ]
        
        # Also check if segments intersect
        if self._line_segments_intersect(p1, p2, p3, p4):
            return 0.0
        
        return min(distances)
    
    def _point_to_line_segment_distance(self, point: SpatialCoordinate,
                                       seg_start: SpatialCoordinate,
                                       seg_end: SpatialCoordinate) -> float:
        """Calculate minimum distance from point to line segment"""
        # Vector from seg_start to seg_end
        seg_vec_x = seg_end.x - seg_start.x
        seg_vec_y = seg_end.y - seg_start.y
        
        # Vector from seg_start to point
        point_vec_x = point.x - seg_start.x
        point_vec_y = point.y - seg_start.y
        
        # Calculate the parameter t for the projection
        seg_len_sq = seg_vec_x**2 + seg_vec_y**2
        
        if seg_len_sq < 1e-10:
            # Segment is essentially a point
            return point.distance_to(seg_start)
        
        # t represents where the projection falls on the line segment
        # t = 0 means seg_start, t = 1 means seg_end
        t = (point_vec_x * seg_vec_x + point_vec_y * seg_vec_y) / seg_len_sq
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
        
        # Find the closest point on the segment
        closest_x = seg_start.x + t * seg_vec_x
        closest_y = seg_start.y + t * seg_vec_y
        
        # If dealing with 3D coordinates
        if point.z is not None and seg_start.z is not None and seg_end.z is not None:
            seg_vec_z = seg_end.z - seg_start.z
            point_vec_z = point.z - seg_start.z
            
            # Recalculate t for 3D
            seg_len_sq += seg_vec_z**2
            if seg_len_sq > 1e-10:
                t = (point_vec_x * seg_vec_x + point_vec_y * seg_vec_y + 
                     point_vec_z * seg_vec_z) / seg_len_sq
                t = max(0.0, min(1.0, t))
            
            closest_z = seg_start.z + t * seg_vec_z
            return np.sqrt((point.x - closest_x)**2 + (point.y - closest_y)**2 + 
                          (point.z - closest_z)**2)
        
        return np.sqrt((point.x - closest_x)**2 + (point.y - closest_y)**2)
    
    def _line_segments_intersect(self, p1: SpatialCoordinate, p2: SpatialCoordinate,
                               p3: SpatialCoordinate, p4: SpatialCoordinate) -> bool:
        """Check if two line segments intersect using the orientation method"""
        def orientation(p: SpatialCoordinate, q: SpatialCoordinate, 
                       r: SpatialCoordinate) -> int:
            """Find orientation of ordered triplet (p, q, r)
            Returns:
            0 if p, q and r are colinear
            1 if Clockwise
            2 if Counterclockwise
            """
            val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else 2
        
        def on_segment(p: SpatialCoordinate, q: SpatialCoordinate, 
                      r: SpatialCoordinate) -> bool:
            """Check if point q lies on segment pr (given they are colinear)"""
            return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                    q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case: segments intersect if they have different orientations
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases for colinear points
        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True
        
        return False
    
    def _project_point_onto_line_segment(self, point: SpatialCoordinate,
                                       seg_start: SpatialCoordinate,
                                       seg_end: SpatialCoordinate) -> float:
        """Project a point onto a line segment and return the parameter t"""
        seg_vec_x = seg_end.x - seg_start.x
        seg_vec_y = seg_end.y - seg_start.y
        
        seg_len_sq = seg_vec_x**2 + seg_vec_y**2
        
        if seg_len_sq < 1e-10:
            return 0.0
        
        point_vec_x = point.x - seg_start.x
        point_vec_y = point.y - seg_start.y
        
        t = (point_vec_x * seg_vec_x + point_vec_y * seg_vec_y) / seg_len_sq
        return max(0.0, min(1.0, t))
    
    def _calculate_overlap_length(self, ref_start: SpatialCoordinate,
                                ref_end: SpatialCoordinate,
                                t1: float, t2: float) -> float:
        """Calculate the overlap length given projection parameters"""
        # Ensure t1 <= t2
        if t1 > t2:
            t1, t2 = t2, t1
        
        # Clamp to [0, 1] range
        t1 = max(0.0, t1)
        t2 = min(1.0, t2)
        
        if t2 <= t1:
            return 0.0
        
        # Calculate the actual overlap length
        ref_len = ref_start.distance_to(ref_end)
        return (t2 - t1) * ref_len
    
    def _is_valid_corner_adjacency(self, points1: List[SpatialCoordinate],
                                  points2: List[SpatialCoordinate],
                                  shared_vertex: Tuple[int, int],
                                  edge_threshold: float) -> bool:
        """Check if a shared vertex represents valid corner adjacency"""
        idx1, idx2 = shared_vertex
        n1 = len(points1)
        n2 = len(points2)
        
        # Get the edges connected to the shared vertex in both polygons
        prev1 = (idx1 - 1) % n1
        next1 = (idx1 + 1) % n1
        prev2 = (idx2 - 1) % n2
        next2 = (idx2 + 1) % n2
        
        # Check if this is a T-junction (one edge from polygon1 touches a vertex of polygon2)
        # This happens when an edge from one polygon is very close to the shared vertex
        edge1_prev = (points1[prev1], points1[idx1])
        edge1_next = (points1[idx1], points1[next1])
        edge2_prev = (points2[prev2], points2[idx2])
        edge2_next = (points2[idx2], points2[next2])
        
        # Check if any edge from polygon2 passes very close to the shared vertex
        # without being connected to it (T-junction case)
        for i in range(n2):
            if i == idx2 or i == prev2:
                continue
            j = (i + 1) % n2
            if j == idx2:
                continue
                
            edge = (points2[i], points2[j])
            dist = self._point_to_line_segment_distance(points1[idx1], edge[0], edge[1])
            if dist < edge_threshold:
                return True
        
        # Check the reverse case
        for i in range(n1):
            if i == idx1 or i == prev1:
                continue
            j = (i + 1) % n1
            if j == idx1:
                continue
                
            edge = (points1[i], points1[j])
            dist = self._point_to_line_segment_distance(points2[idx2], edge[0], edge[1])
            if dist < edge_threshold:
                return True
        
        # For simple corner touch, we typically don't consider it adjacency
        # unless it's part of a more complex configuration
        return False
    
    async def _update_object_region_membership(self, map_id: str, object_id: str) -> None:
        """Update which regions contain an object"""
        if map_id not in self.maps or object_id not in self.maps[map_id].spatial_objects:
            return
        
        map_obj = self.maps[map_id]
        obj = map_obj.spatial_objects[object_id]
        
        # Remove object from all regions
        for region_id, region in map_obj.regions.items():
            if object_id in region.contained_objects:
                region.contained_objects.remove(object_id)
        
        # Check each region to see if it contains the object
        for region_id, region in map_obj.regions.items():
            if region.contains_point(obj.coordinates):
                region.contained_objects.append(object_id)
    
    async def _update_region_object_containment(self, map_id: str, region_id: str) -> None:
        """Find which objects are contained in a region"""
        if map_id not in self.maps or region_id not in self.maps[map_id].regions:
            return
        
        map_obj = self.maps[map_id]
        region = map_obj.regions[region_id]
        
        # Check each object to see if it's in this region
        for object_id, obj in map_obj.spatial_objects.items():
            if region.contains_point(obj.coordinates):
                if object_id not in region.contained_objects:
                    region.contained_objects.append(object_id)
    
    async def _update_map_completeness(self, map_id: str) -> None:
        """Update the completeness metric of a map"""
        if map_id not in self.maps:
            return
        
        map_obj = self.maps[map_id]
        
        # Simple completeness heuristic based on number of elements
        obj_count = len(map_obj.spatial_objects)
        region_count = len(map_obj.regions)
        route_count = len(map_obj.routes)
        landmark_count = len(map_obj.landmarks)
        
        # Calculate completeness score (between 0 and 1)
        # Different weights for different element types
        completeness = min(1.0, (
            (obj_count * 0.3) + 
            (region_count * 0.3) + 
            (route_count * 0.2) + 
            (landmark_count * 0.2)
        ) / 10.0)  # Normalized by expected count of 10 elements
        
        map_obj.completeness = completeness
    
    @function_tool
    async def update_route(self, map_id: str, route_id: str, 
                        waypoints: Optional[List[CoordinateDict]] = None,
                        estimated_time: Optional[float] = None) -> Union[UpdateRouteResult, ErrorResult]:
        """
        Update an existing route
        
        Args:
            map_id: ID of the map
            route_id: ID of the route to update
            waypoints: Optional new waypoints
            estimated_time: Optional new estimated time
            
        Returns:
            Updated route information
        """
        with custom_span("update_route", {"map_id": map_id, "route_id": route_id}):
            # Check if map and route exist
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            if route_id not in self.maps[map_id].routes:
                return ErrorResult(error=f"Route {route_id} not found in map")
            
            map_obj = self.maps[map_id]
            route = map_obj.routes[route_id]
            
            # Update waypoints if provided
            if waypoints:
                coordinates = []
                for point in waypoints:
                    coord = SpatialCoordinate(
                        x=point.x,
                        y=point.y,
                        z=point.z,
                        reference_frame=map_obj.reference_frame
                    )
                    coordinates.append(coord)
                
                route.waypoints = coordinates
                
                # Recalculate distance
                distance = 0.0
                if len(coordinates) > 1:
                    for i in range(len(coordinates) - 1):
                        distance += coordinates[i].distance_to(coordinates[i+1])
                
                route.distance = distance
            
            # Update estimated time if provided
            if estimated_time is not None:
                route.estimated_time = estimated_time
            
            # Update route usage
            route.update_usage()
            
            # Update map
            async with self._lock:
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
            
            logger.info(f"Updated route {route_id} in map {map_id}")
            
            return UpdateRouteResult(
                route_id=route_id,
                name=route.name,
                distance=route.distance,
                waypoints_count=len(route.waypoints),
                estimated_time=route.estimated_time,
                usage_count=route.usage_count
            )
    
    @function_tool
    async def identify_shortcuts(self, map_id: str, max_count: int = 3) -> Union[List[ShortcutInfo], List[ErrorResult], List[MessageResult]]:
        """
        Identify potential shortcuts between locations
        
        Args:
            map_id: ID of the map
            max_count: Maximum number of shortcuts to find
            
        Returns:
            List of potential shortcuts
        """
        with custom_span("identify_shortcuts", {"map_id": map_id}):
            # Check if map exists
            if map_id not in self.maps:
                return [ErrorResult(error=f"Map {map_id} not found")]
            
            map_obj = self.maps[map_id]
            
            # Need at least a few routes to find shortcuts
            if len(map_obj.routes) < 3:
                return [MessageResult(message="Not enough routes to identify shortcuts")]
            
            shortcuts = []
            
            # Find pairs of routes that could be connected
            routes = list(map_obj.routes.values())
            
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    route1 = routes[i]
                    route2 = routes[j]
                    
                    # Check if routes share an endpoint
                    if (route1.start_id == route2.start_id or 
                        route1.start_id == route2.end_id or 
                        route1.end_id == route2.start_id or 
                        route1.end_id == route2.end_id):
                        continue  # Skip routes that already connect
                    
                    # Check for closest points between routes
                    min_distance = float('inf')
                    closest_point1 = None
                    closest_point2 = None
                    
                    for wp1 in route1.waypoints:
                        for wp2 in route2.waypoints:
                            distance = wp1.distance_to(wp2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_point1 = wp1
                                closest_point2 = wp2
                    
                    # Only consider shortcuts if the routes are reasonably close
                    if min_distance < 10.0 and closest_point1 and closest_point2:
                        # Calculate potential improvement
                        route1_name = self._get_location_name(map_id, route1.start_id) + " to " + self._get_location_name(map_id, route1.end_id)
                        route2_name = self._get_location_name(map_id, route2.start_id) + " to " + self._get_location_name(map_id, route2.end_id)
                        
                        shortcuts.append(ShortcutInfo(
                            route1_id=route1.id,
                            route1_name=route1_name,
                            route2_id=route2.id,
                            route2_name=route2_name,
                            distance_between=min_distance,
                            potential_saving=min(route1.distance, route2.distance) * 0.5,
                            connection_point1=CoordinateDict(x=closest_point1.x, y=closest_point1.y, z=closest_point1.z),
                            connection_point2=CoordinateDict(x=closest_point2.x, y=closest_point2.y, z=closest_point2.z)
                        ))
            
            # Sort by potential savings and return top results
            shortcuts.sort(key=lambda x: x.potential_saving, reverse=True)
            return shortcuts[:max_count]
    
    @function_tool
    async def process_spatial_observation(self, observation: SpatialObservation) -> ProcessObservationResult:
        """
        Process a spatial observation to update maps
        
        Args:
            observation: Spatial observation object
            
        Returns:
            Processing results
        """
        with custom_span("process_spatial_observation", {"type": observation.observation_type}):
            # Process different types of observations
            if not self.context.active_map_id:
                return ProcessObservationResult(error="No active map set")
            
            map_id = self.context.active_map_id
            
            # Update observer position
            if observation.observer_position:
                self.context.observer_position = observation.observer_position
            
            if observation.observer_orientation:
                self.context.observer_orientation = observation.observer_orientation
                
            self.context.last_observation_time = observation.timestamp
            self.context.observation_count += 1
            
            # Process based on observation type
            if observation.observation_type == "object":
                # Process object observation
                result = await self._process_object_observation(map_id, observation)
            elif observation.observation_type == "region":
                # Process region observation
                result = await self._process_region_observation(map_id, observation)
            elif observation.observation_type == "route":
                # Process route observation
                result = await self._process_route_observation(map_id, observation)
            elif observation.observation_type == "relative_position":
                # Process relative position observation
                result = await self._process_relative_position(map_id, observation)
            else:
                result = ProcessObservationResult(error=f"Unknown observation type: {observation.observation_type}")
            
            # Store in memory if memory integration is available
            if self.memory_integration and hasattr(self.memory_integration, 'add_memory'):
                try:
                    memory_text = f"Observed {observation.observation_type}: {observation.content}"
                    await self.memory_integration.add_memory(
                        memory_text=memory_text,
                        memory_type="observation",
                        memory_scope="spatial",
                        tags=["spatial", observation.observation_type],
                        metadata={
                            "timestamp": observation.timestamp,
                            "spatial_observation": observation.dict()
                        }
                    )
                except Exception as e:
                    logger.error(f"Error storing spatial observation in memory: {e}")
            
            return result
    
    async def _process_object_observation(self, map_id: str, observation: SpatialObservation) -> ProcessObservationResult:
        """Process an object observation"""
        content = observation.content
        
        # Check if required fields are present
        if not content.name or not content.object_type:
            return ProcessObservationResult(error="Object observation missing required fields")
        
        # Extract coordinates
        if content.coordinates:
            coordinates = content.coordinates
        elif observation.observer_position and content.relative_position:
            # Calculate absolute position from relative position
            rel_pos = content.relative_position
            observer_pos = observation.observer_position
            
            coordinates = CoordinateDict(
                x=observer_pos.x + rel_pos.x,
                y=observer_pos.y + rel_pos.y,
                z=observer_pos.z + rel_pos.z if observer_pos.z is not None and rel_pos.z is not None else None
            )
        else:
            return ProcessObservationResult(error="Object observation missing position information")
        
        # Check if this is an update to an existing object
        existing_object_id = None
        
        # First, check by name if provided
        if content.name:
            for obj_id, obj in self.maps[map_id].spatial_objects.items():
                if obj.name == content.name:
                    existing_object_id = obj_id
                    break
        
        # If name not found, try checking by position
        if not existing_object_id and coordinates:
            coord = SpatialCoordinate(x=coordinates.x, y=coordinates.y, z=coordinates.z)
            closest_distance = float('inf')
            closest_id = None
            
            for obj_id, obj in self.maps[map_id].spatial_objects.items():
                if obj.object_type == content.object_type:  # Only match same type
                    distance = coord.distance_to(obj.coordinates)
                    if distance < closest_distance and distance < 2.0:  # Max 2 units distance for match
                        closest_distance = distance
                        closest_id = obj_id
            
            if closest_id:
                existing_object_id = closest_id
        
        # Process size information if provided
        size = content.size
        
        # Process properties
        properties = content.properties or PropertiesDict()
        
        # Process is_landmark
        is_landmark = content.is_landmark or False
        
        # Update existing object or create new one
        if existing_object_id:
            result = await self.update_object_position(
                map_id=map_id,
                object_id=existing_object_id,
                new_coordinates=coordinates
            )
            
            if isinstance(result, ErrorResult):
                return ProcessObservationResult(error=result.error)
            
            # Update other properties if needed
            obj = self.maps[map_id].spatial_objects[existing_object_id]
            
            if size:
                obj.size = size
            
            # Update properties
            if properties.description:
                obj.properties.description = properties.description
            if properties.category:
                obj.properties.category = properties.category
            # ... update other properties as needed
                
            if is_landmark and not obj.is_landmark:
                obj.is_landmark = True
                if existing_object_id not in self.maps[map_id].landmarks:
                    self.maps[map_id].landmarks.append(existing_object_id)
            
            return ProcessObservationResult(
                action="updated",
                object_id=existing_object_id,
                name=obj.name
            )
        else:
            # Create new object
            result = await self.add_spatial_object(
                map_id=map_id,
                name=content.name,
                object_type=content.object_type,
                coordinates=coordinates,
                size=size,
                is_landmark=is_landmark,
                properties=properties
            )
            
            if isinstance(result, ErrorResult):
                return ProcessObservationResult(error=result.error)
            
            return ProcessObservationResult(
                action="created",
                object_id=result.object_id,
                name=result.name
            )
    
    async def _process_region_observation(self, map_id: str, observation: SpatialObservation) -> ProcessObservationResult:
        """Process a region observation"""
        content = observation.content
        
        # Check if required fields are present
        if not content.name or not content.region_type:
            return ProcessObservationResult(error="Region observation missing required fields")
        
        # Check if boundary points are provided
        if not content.boundary_points:
            return ProcessObservationResult(error="Region observation missing boundary points")
        
        # Check if this is an update to an existing region
        existing_region_id = None
        
        # First, check by name if provided
        if content.name:
            for region_id, region in self.maps[map_id].regions.items():
                if region.name == content.name:
                    existing_region_id = region_id
                    break
        
        # Process properties
        properties = content.properties or PropertiesDict()
        
        # Process navigability
        is_navigable = content.is_navigable if content.is_navigable is not None else True
        
        # Update existing region or create new one
        if existing_region_id:
            # For now, we'll just update the region properties
            region = self.maps[map_id].regions[existing_region_id]
            
            # Update properties
            if properties.description:
                region.properties.description = properties.description
            if properties.category:
                region.properties.category = properties.category
            # ... update other properties as needed
                
            region.is_navigable = is_navigable
            
            # Update map
            async with self._lock:
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
            
            return ProcessObservationResult(
                region_id=existing_region_id,
                name=region.name,
                action="updated"
            )
        else:
            # Create new region
            result = await self.define_region(
                map_id=map_id,
                name=content.name,
                region_type=content.region_type,
                boundary_points=content.boundary_points,
                is_navigable=is_navigable,
                properties=properties
            )
            
            if isinstance(result, ErrorResult):
                return ProcessObservationResult(error=result.error)
            
            return ProcessObservationResult(
                action="created",
                region_id=result.region_id,
                name=result.name
            )
    
    async def _process_route_observation(self, map_id: str, observation: SpatialObservation) -> ProcessObservationResult:
        """Process a route observation"""
        content = observation.content
        
        # Check if required fields are present
        if not content.start_id or not content.end_id:
            return ProcessObservationResult(error="Route observation missing required fields")
        
        # Get waypoints if provided
        waypoints = content.waypoints
        
        # Get estimated time if provided
        estimated_time = content.estimated_time
        
        # Check if this is an update to an existing route
        existing_route_id = None
        
        # Check for an existing route between these points
        for route_id, route in self.maps[map_id].routes.items():
            if route.start_id == content.start_id and route.end_id == content.end_id:
                existing_route_id = route_id
                break
        
        # Process properties
        properties = content.properties or PropertiesDict()
        
        # Update existing route or create new one
        if existing_route_id:
            # Update the route
            result = await self.update_route(
                map_id=map_id,
                route_id=existing_route_id,
                waypoints=waypoints,
                estimated_time=estimated_time
            )
            
            if isinstance(result, ErrorResult):
                return ProcessObservationResult(error=result.error)
            
            # Update properties
            route = self.maps[map_id].routes[existing_route_id]
            if properties.description:
                route.properties.description = properties.description
            # ... update other properties as needed
            
            return ProcessObservationResult(
                action="updated",
                route_id=existing_route_id,
                name=route.name
            )
        else:
            # Create new route
            result = await self.create_route(
                map_id=map_id,
                start_id=content.start_id,
                end_id=content.end_id,
                waypoints=waypoints,
                name=content.name,
                estimated_time=estimated_time,
                properties=properties
            )
            
            if isinstance(result, ErrorResult):
                return ProcessObservationResult(error=result.error)
            
            return ProcessObservationResult(
                action="created",
                route_id=result.route_id,
                name=result.name
            )
    
    async def _process_relative_position(self, map_id: str, observation: SpatialObservation) -> ProcessObservationResult:
        """Process a relative position observation"""
        content = observation.content
        
        # Check if required fields are present
        if not content.object_id or not content.relative_to_id or not content.relative_position:
            return ProcessObservationResult(error="Relative position observation missing required fields")
        
        # Check if objects exist
        if content.object_id not in self.maps[map_id].spatial_objects:
            return ProcessObservationResult(error=f"Object {content.object_id} not found")
        
        reference_obj_id = content.relative_to_id
        if reference_obj_id not in self.maps[map_id].spatial_objects:
            return ProcessObservationResult(error=f"Reference object {reference_obj_id} not found")
        
        # Get reference object position
        reference_obj = self.maps[map_id].spatial_objects[reference_obj_id]
        reference_pos = reference_obj.coordinates
        
        # Calculate absolute position
        rel_pos = content.relative_position
        new_coordinates = CoordinateDict(
            x=reference_pos.x + rel_pos.x,
            y=reference_pos.y + rel_pos.y,
            z=reference_pos.z + rel_pos.z if reference_pos.z is not None and rel_pos.z is not None else None
        )
        
        # Update object position
        result = await self.update_object_position(
            map_id=map_id,
            object_id=content.object_id,
            new_coordinates=new_coordinates
        )
        
        if isinstance(result, ErrorResult):
            return ProcessObservationResult(error=result.error)
        
        return ProcessObservationResult(
            action="updated_relative",
            object_id=content.object_id,
            name=result.name
        )
    
    @function_tool
    async def extract_spatial_features(self, text: str) -> ExtractedFeatures:
        """
        Extract spatial features from natural language text
        
        Args:
            text: Text describing a spatial observation
            
        Returns:
            Extracted spatial features
        """
        with custom_span("extract_spatial_features"):
            # Ensure perception agent is initialized
            await self._ensure_agents_initialized()
            
            # Call the perception agent to process the text
            with trace(workflow_name="extract_spatial_features"):
                prompt = f"Extract spatial information from this text: {text}"
                
                result = await Runner.run(
                    self._perception_agent,
                    prompt
                )
            
            # If we have a result, return it
            if hasattr(result, 'final_output') and result.final_output:
                return result.final_output
            
            # Fallback simple extraction
            features = ExtractedFeatures(
                objects=[],
                locations=[],
                spatial_relationships=[],
                directions=[],
                confidence=0.5
            )
            
            # Extract objects (look for nouns)
            objects = ["table", "chair", "door", "window", "wall", "room"]
            for obj in objects:
                if obj in text.lower():
                    features.objects.append(obj)
            
            # Extract locations (look for location prepositions)
            location_patterns = ["in the", "at the", "on the", "near the"]
            for pattern in location_patterns:
                if pattern in text.lower():
                    idx = text.lower().find(pattern)
                    if idx >= 0:
                        # Try to extract the location phrase
                        end_idx = text.find(".", idx)
                        if end_idx < 0:
                            end_idx = len(text)
                        location_phrase = text[idx:end_idx].strip()
                        features.locations.append(location_phrase)
            
            # Extract spatial relationships (look for spatial prepositions)
            spatial_preps = ["above", "below", "under", "behind", "in front of", "beside", "between"]
            for prep in spatial_preps:
                if prep in text.lower():
                    idx = text.lower().find(prep)
                    if idx >= 0:
                        # Try to extract the relationship phrase
                        start_idx = max(0, idx - 20)
                        end_idx = min(len(text), idx + 20)
                        rel_phrase = text[start_idx:end_idx].strip()
                        features.spatial_relationships.append(rel_phrase)
            
            # Extract directions (look for cardinal directions and movement verbs)
            directions = ["north", "south", "east", "west", "up", "down", "left", "right"]
            movement_verbs = ["move", "walk", "go", "turn"]
            
            for direction in directions:
                if direction in text.lower():
                    features.directions.append(direction)
            
            for verb in movement_verbs:
                if verb in text.lower():
                    idx = text.lower().find(verb)
                    if idx >= 0:
                        # Try to extract the direction phrase
                        end_idx = text.find(".", idx)
                        if end_idx < 0:
                            end_idx = len(text)
                        direction_phrase = text[idx:end_idx].strip()
                        features.directions.append(direction_phrase)
            
            return features
    
    @function_tool
    async def estimate_distances(self, object1_id: str, object2_id: str, 
                              map_id: Optional[str] = None) -> Union[DistanceEstimation, ErrorResult]:
        """
        Estimate distance between two objects
        
        Args:
            object1_id: ID of the first object
            object2_id: ID of the second object
            map_id: Optional map ID (uses active map if not provided)
            
        Returns:
            Distance information
        """
        with custom_span("estimate_distances", {"object1": object1_id, "object2": object2_id}):
            # Use active map if not provided
            if not map_id:
                if not self.context.active_map_id:
                    return ErrorResult(error="No active map set")
                map_id = self.context.active_map_id
            
            # Check if map exists
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            map_obj = self.maps[map_id]
            
            # Check if objects exist
            if object1_id not in map_obj.spatial_objects:
                return ErrorResult(error=f"Object {object1_id} not found")
            
            if object2_id not in map_obj.spatial_objects:
                return ErrorResult(error=f"Object {object2_id} not found")
            
            # Get object coordinates
            obj1 = map_obj.spatial_objects[object1_id]
            obj2 = map_obj.spatial_objects[object2_id]
            
            # Calculate Euclidean distance
            distance = obj1.coordinates.distance_to(obj2.coordinates)
            
            # Calculate direction
            heading = self._calculate_heading(obj1.coordinates, obj2.coordinates)
            
            # Check if objects are in the same region
            shared_regions = []
            
            for region_id, region in map_obj.regions.items():
                if object1_id in region.contained_objects and object2_id in region.contained_objects:
                    shared_regions.append(region.name)
            
            # Check if there's a direct route between them
            direct_route = None
            route_id = None
            
            for r_id, route in map_obj.routes.items():
                if (route.start_id == object1_id and route.end_id == object2_id) or \
                   (route.start_id == object2_id and route.end_id == object1_id):
                    direct_route = route
                    route_id = r_id
                    break
            
            result = DistanceEstimation(
                object1=ObjectInfo(
                    id=object1_id,
                    name=obj1.name
                ),
                object2=ObjectInfo(
                    id=object2_id,
                    name=obj2.name
                ),
                euclidean_distance=distance,
                direction=heading,
                shared_regions=shared_regions,
                has_direct_route=direct_route is not None
            )
            
            if direct_route:
                result.route = RouteInfo(
                    id=route_id,
                    distance=direct_route.distance,
                    estimated_time=direct_route.estimated_time
                )
            
            return result
    
    @function_tool
    async def reconcile_observations(self, map_id: str) -> Union[ReconciliationResult, ErrorResult]:
        """
        Reconcile conflicting observations and improve map consistency
        
        Args:
            map_id: ID of the map to reconcile
            
        Returns:
            Reconciliation results
        """
        with custom_span("reconcile_observations", {"map_id": map_id}):
            # Check if map exists
            if map_id not in self.maps:
                return ErrorResult(error=f"Map {map_id} not found")
            
            map_obj = self.maps[map_id]
            
            # Track changes
            changes = ReconciliationChanges(
                updated_objects=0,
                merged_objects=0,
                updated_regions=0
            )
            
            # Find and merge duplicate objects
            objects = list(map_obj.spatial_objects.values())
            object_ids = list(map_obj.spatial_objects.keys())
            
            # Group similar objects
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    obj1 = objects[i]
                    obj2 = objects[j]
                    
                    # Skip if either object has been processed already
                    if object_ids[i] not in map_obj.spatial_objects or object_ids[j] not in map_obj.spatial_objects:
                        continue
                    
                    # Check if objects are likely duplicates
                    if obj1.name == obj2.name and obj1.object_type == obj2.object_type:
                        distance = obj1.coordinates.distance_to(obj2.coordinates)
                        
                        # If they're close enough, merge them
                        if distance < 2.0:
                            # Keep the more observed one and update its position to the average
                            if obj1.observation_count >= obj2.observation_count:
                                keeper = obj1
                                remove_id = object_ids[j]
                            else:
                                keeper = obj2
                                remove_id = object_ids[i]
                            
                            # Average the coordinates
                            avg_x = (obj1.coordinates.x + obj2.coordinates.x) / 2
                            avg_y = (obj1.coordinates.y + obj2.coordinates.y) / 2
                            avg_z = None
                            if obj1.coordinates.z is not None and obj2.coordinates.z is not None:
                                avg_z = (obj1.coordinates.z + obj2.coordinates.z) / 2
                            
                            new_coordinates = CoordinateDict(
                                x=avg_x,
                                y=avg_y,
                                z=avg_z
                            )
                            
                            # Update position of keeper
                            await self.update_object_position(
                                map_id=map_id,
                                object_id=keeper.id,
                                new_coordinates=new_coordinates
                            )
                            
                            # Merge properties
                            removed_obj = map_obj.spatial_objects[remove_id]
                            # Merge properties
                            if removed_obj.properties.description and not keeper.properties.description:
                                keeper.properties.description = removed_obj.properties.description
                            if removed_obj.properties.category and not keeper.properties.category:
                                keeper.properties.category = removed_obj.properties.category
                            # ... merge other properties as needed
                            
                            # Combine visibility
                            keeper.visibility = max(keeper.visibility, removed_obj.visibility)
                            
                            # Update observation count
                            keeper.observation_count += removed_obj.observation_count
                            
                            # Transfer connections
                            for conn in removed_obj.connections:
                                if conn not in keeper.connections:
                                    keeper.connections.append(conn)
                            
                            # Remove the duplicate
                            del map_obj.spatial_objects[remove_id]
                            
                            # Update routes that referenced the removed object
                            for route_id, route in map_obj.routes.items():
                                if route.start_id == remove_id:
                                    route.start_id = keeper.id
                                if route.end_id == remove_id:
                                    route.end_id = keeper.id
                            
                            # Update region containment
                            for region_id, region in map_obj.regions.items():
                                if remove_id in region.contained_objects:
                                    region.contained_objects.remove(remove_id)
                                    if keeper.id not in region.contained_objects:
                                        region.contained_objects.append(keeper.id)
                            
                            # Update landmarks list
                            if remove_id in map_obj.landmarks:
                                map_obj.landmarks.remove(remove_id)
                                if keeper.id not in map_obj.landmarks and keeper.is_landmark:
                                    map_obj.landmarks.append(keeper.id)
                            
                            changes.merged_objects += 1
            
            # Update map
            async with self._lock:
                map_obj.last_updated = datetime.datetime.now().isoformat()
            
            # Update map accuracy based on reconciliation
            accuracy_improvement = min(0.1, changes.merged_objects * 0.02)
            map_obj.accuracy = min(1.0, map_obj.accuracy + accuracy_improvement)
            
            return ReconciliationResult(
                changes=changes,
                map_accuracy=map_obj.accuracy,
                map_completeness=map_obj.completeness
            )
