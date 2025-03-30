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
    size: Optional[Dict[str, float]] = None  # width, height, depth
    properties: Dict[str, Any] = Field(default_factory=dict)
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
    properties: Dict[str, Any] = Field(default_factory=dict)
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
    properties: Dict[str, Any] = Field(default_factory=dict)
    
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
    properties: Dict[str, Any] = Field(default_factory=dict)

class SpatialObservation(BaseModel):
    """Represents a spatial observation input"""
    observation_type: str  # object, region, relative_position, etc.
    content: Dict[str, Any]  # Flexible content based on observation type
    confidence: float = 1.0
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    observer_position: Optional[SpatialCoordinate] = None
    observer_orientation: Optional[Dict[str, float]] = None  # heading, pitch, roll
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SpatialQuery(BaseModel):
    """Query parameters for spatial information retrieval"""
    query_type: str  # nearest, contains, path, region_info, etc.
    parameters: Dict[str, Any]
    reference_position: Optional[SpatialCoordinate] = None
    map_id: Optional[str] = None

class SpatialMapperContext(BaseModel):
    """Context for the SpatialMapper"""
    active_map_id: Optional[str] = None
    observer_position: Optional[SpatialCoordinate] = None
    observer_orientation: Optional[Dict[str, float]] = None
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
                function_tool(self.add_spatial_object),
                function_tool(self.define_region),
                function_tool(self.create_route),
                function_tool(self.update_object_position),
                function_tool(self.merge_regions),
                function_tool(self.calculate_region_connections),
                function_tool(self.identify_landmarks)
            ]
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
                function_tool(self.process_spatial_observation),
                function_tool(self.extract_spatial_features),
                function_tool(self.estimate_distances),
                function_tool(self.reconcile_observations)
            ]
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
                function_tool(self.find_path),
                function_tool(self.calculate_directions),
                function_tool(self.identify_shortcuts),
                function_tool(self.get_nearest_landmarks),
                function_tool(self.update_route)
            ]
        )
    
    @function_tool
    async def create_cognitive_map(self, name: str, description: Optional[str] = None, 
                                  map_type: str = "spatial", 
                                  reference_frame: str = "allocentric") -> Dict[str, Any]:
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
            
            return {
                "map_id": map_id,
                "name": name,
                "map_type": map_type,
                "reference_frame": reference_frame,
                "creation_date": new_map.creation_date
            }
    
    @function_tool
    async def add_spatial_object(self, map_id: str, name: str, object_type: str,
                               coordinates: Dict[str, float], 
                               size: Optional[Dict[str, float]] = None,
                               is_landmark: bool = False,
                               properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
            # Create coordinate object
            coordinate = SpatialCoordinate(
                x=coordinates.get("x", 0.0),
                y=coordinates.get("y", 0.0),
                z=coordinates.get("z") if "z" in coordinates else None,
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
                properties=properties or {},
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
            
            return {
                "object_id": object_id,
                "name": name,
                "object_type": object_type,
                "coordinates": coordinate.dict(),
                "is_landmark": is_landmark
            }
    
    @function_tool
    async def define_region(self, map_id: str, name: str, region_type: str,
                          boundary_points: List[Dict[str, float]],
                          is_navigable: bool = True,
                          adjacent_regions: Optional[List[str]] = None,
                          properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
            # Convert boundary points to SpatialCoordinates
            coordinates = []
            for point in boundary_points:
                coord = SpatialCoordinate(
                    x=point.get("x", 0.0),
                    y=point.get("y", 0.0),
                    z=point.get("z") if "z" in point else None,
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
                properties=properties or {}
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
            
            return {
                "region_id": region_id,
                "name": name,
                "region_type": region_type,
                "is_navigable": is_navigable,
                "contained_objects_count": len(new_region.contained_objects)
            }
    
    @function_tool
    async def create_route(self, map_id: str, start_id: str, end_id: str,
                         waypoints: Optional[List[Dict[str, float]]] = None,
                         name: Optional[str] = None,
                         estimated_time: Optional[float] = None,
                         properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
            # Check if start and end exist
            map_obj = self.maps[map_id]
            start_exists = start_id in map_obj.spatial_objects or start_id in map_obj.regions
            end_exists = end_id in map_obj.spatial_objects or end_id in map_obj.regions
            
            if not start_exists or not end_exists:
                return {"error": f"Start or end location not found in map"}
            
            # Convert waypoints to SpatialCoordinates if provided
            coordinates = []
            if waypoints:
                for point in waypoints:
                    coord = SpatialCoordinate(
                        x=point.get("x", 0.0),
                        y=point.get("y", 0.0),
                        z=point.get("z") if "z" in point else None,
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
                properties=properties or {}
            )
            
            # Add to map
            async with self._lock:
                self.maps[map_id].routes[route_id] = new_route
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
            
            logger.info(f"Created route from {start_id} to {end_id} in map {map_id}")
            
            return {
                "route_id": route_id,
                "name": new_route.name,
                "distance": distance,
                "waypoints_count": len(coordinates),
                "estimated_time": estimated_time
            }
    
    @function_tool
    async def update_object_position(self, map_id: str, object_id: str, 
                                  new_coordinates: Dict[str, float]) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
            if object_id not in self.maps[map_id].spatial_objects:
                return {"error": f"Object {object_id} not found in map"}
            
            # Create coordinate object
            coordinate = SpatialCoordinate(
                x=new_coordinates.get("x", 0.0),
                y=new_coordinates.get("y", 0.0),
                z=new_coordinates.get("z") if "z" in new_coordinates else None,
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
            
            return {
                "object_id": object_id,
                "name": obj.name,
                "new_coordinates": coordinate.dict(),
                "observation_count": obj.observation_count
            }
    
    @function_tool
    async def get_map(self, map_id: str) -> Dict[str, Any]:
        """
        Get information about a cognitive map
        
        Args:
            map_id: ID of the map to retrieve
            
        Returns:
            Map information
        """
        # Check if map exists
        if map_id not in self.maps:
            return {"error": f"Map {map_id} not found"}
        
        map_obj = self.maps[map_id]
        
        # Create a summary of the map
        summary = {
            "id": map_obj.id,
            "name": map_obj.name,
            "description": map_obj.description,
            "map_type": map_obj.map_type,
            "reference_frame": map_obj.reference_frame,
            "objects_count": len(map_obj.spatial_objects),
            "regions_count": len(map_obj.regions),
            "routes_count": len(map_obj.routes),
            "landmarks_count": len(map_obj.landmarks),
            "creation_date": map_obj.creation_date,
            "last_updated": map_obj.last_updated,
            "accuracy": map_obj.accuracy,
            "completeness": map_obj.completeness
        }
        
        # Set as active map
        self.context.active_map_id = map_id
        
        return summary
    
    @function_tool
    async def find_path(self, map_id: str, start_id: str, end_id: str, 
                     prefer_landmarks: bool = True) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
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
                
                return {
                    "route_id": direct_route.id,
                    "path_type": "direct",
                    "distance": direct_route.distance,
                    "estimated_time": direct_route.estimated_time,
                    "directions": directions,
                    "waypoints": [w.dict() for w in direct_route.waypoints]
                }
            
            # No direct route exists, try to build one using existing routes or navigation algorithms
            path_result = await self._calculate_multi_segment_path(map_id, start_id, end_id, prefer_landmarks)
            
            if "error" in path_result:
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
            
            # Format directions
            directions = await self._format_route_directions(map_id, new_route["route_id"])
            
            path_result["directions"] = directions
            
            logger.info(f"Found path from {start_id} to {end_id} in map {map_id}")
            
            return path_result
    
    async def _calculate_multi_segment_path(self, map_id: str, start_id: str, end_id: str, 
                                         prefer_landmarks: bool) -> Dict[str, Any]:
        """Calculate a path that may involve multiple segments"""
        # TODO: Implement more sophisticated pathfinding
        # This is a simplified implementation for demonstration
        
        map_obj = self.maps[map_id]
        
        # Get start and end coordinates
        start_coord = await self._get_location_coordinates(map_id, start_id)
        end_coord = await self._get_location_coordinates(map_id, end_id)
        
        if not start_coord or not end_coord:
            return {"error": "Could not determine start or end coordinates"}
        
        waypoints = [start_coord.dict()]
        
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
                waypoints.append(best_landmark.coordinates.dict())
        
        # Add end point
        waypoints.append(end_coord.dict())
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(waypoints) - 1):
            coord1 = SpatialCoordinate(**waypoints[i])
            coord2 = SpatialCoordinate(**waypoints[i+1])
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
    async def calculate_directions(self, map_id: str, start_id: str, end_id: str) -> List[str]:
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
        
        if "error" in path_result:
            return [f"Error: {path_result['error']}"]
        
        return path_result.get("directions", ["No directions available"])
    
    @function_tool
    async def get_nearest_landmarks(self, map_id: str, location_id: str, 
                                 max_count: int = 3) -> List[Dict[str, Any]]:
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
                return [{"error": f"Map {map_id} not found"}]
            
            # Get coordinates of the location
            location_coord = await self._get_location_coordinates(map_id, location_id)
            
            if not location_coord:
                return [{"error": f"Location {location_id} not found or has no coordinates"}]
            
            map_obj = self.maps[map_id]
            landmarks = []
            
            # Calculate distances to all landmarks
            for landmark_id in map_obj.landmarks:
                if landmark_id in map_obj.spatial_objects:
                    landmark = map_obj.spatial_objects[landmark_id]
                    distance = location_coord.distance_to(landmark.coordinates)
                    
                    landmarks.append({
                        "id": landmark_id,
                        "name": landmark.name,
                        "distance": distance,
                        "direction": self._calculate_heading(location_coord, landmark.coordinates)
                    })
            
            # Sort by distance and return top N
            landmarks.sort(key=lambda x: x["distance"])
            return landmarks[:max_count]
    
    @function_tool
    async def merge_regions(self, map_id: str, region_ids: List[str], 
                          new_name: str, new_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Merge multiple regions into a single larger region
        
        Args:
            map_id: ID of the map
            region_ids: List of region IDs to merge
            new_name: Name for the merged region
            new_type: Optional type for the merged region
            
        Returns:
            Information about the new merged region
        """
        with custom_span("merge_regions", {"map_id": map_id, "count": len(region_ids)}):
            # Check if map exists
            if map_id not in self.maps:
                return {"error": f"Map {map_id} not found"}
            
            map_obj = self.maps[map_id]
            
            # Check if all regions exist
            for region_id in region_ids:
                if region_id not in map_obj.regions:
                    return {"error": f"Region {region_id} not found in map"}
            
            if len(region_ids) < 2:
                return {"error": "Need at least two regions to merge"}
            
            # Collect all boundary points from the regions
            all_points = []
            for region_id in region_ids:
                all_points.extend(map_obj.regions[region_id].boundary_points)
            
            # Simple merging approach: find convex hull or bounding box
            # For simplicity, we'll just use the extremes to create a bounding box
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
            
            # Create new boundary points
            new_boundary = [
                SpatialCoordinate(x=min_x, y=min_y),
                SpatialCoordinate(x=max_x, y=min_y),
                SpatialCoordinate(x=max_x, y=max_y),
                SpatialCoordinate(x=min_x, y=max_y)
            ]
            
            # Get combined properties
            combined_properties = {}
            for region_id in region_ids:
                combined_properties.update(map_obj.regions[region_id].properties)
            
            # Determine navigability (if any region is non-navigable, the merged region is too)
            is_navigable = all(map_obj.regions[region_id].is_navigable for region_id in region_ids)
            
            # Get all contained objects
            contained_objects = []
            for region_id in region_ids:
                contained_objects.extend(map_obj.regions[region_id].contained_objects)
            
            # Get adjacent regions (excluding the ones being merged)
            adjacent_regions = []
            for region_id in region_ids:
                adjacent_regions.extend([r for r in map_obj.regions[region_id].adjacent_regions 
                                         if r not in region_ids])
            
            # Generate region ID
            merged_region_id = f"region_{uuid.uuid4().hex[:8]}"
            
            # Create merged region
            merged_region = SpatialRegion(
                id=merged_region_id,
                name=new_name,
                region_type=new_type or map_obj.regions[region_ids[0]].region_type,
                boundary_points=new_boundary,
                is_navigable=is_navigable,
                contained_objects=list(set(contained_objects)),  # Remove duplicates
                adjacent_regions=list(set(adjacent_regions)),    # Remove duplicates
                properties=combined_properties
            )
            
            # Add to map
            async with self._lock:
                self.maps[map_id].regions[merged_region_id] = merged_region
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
                
                # Update adjacent regions to point to the merged region
                for region_id in merged_region.adjacent_regions:
                    if region_id in map_obj.regions:
                        # Remove references to merged regions
                        map_obj.regions[region_id].adjacent_regions = [
                            r for r in map_obj.regions[region_id].adjacent_regions 
                            if r not in region_ids
                        ]
                        # Add reference to new merged region
                        map_obj.regions[region_id].adjacent_regions.append(merged_region_id)
            
            logger.info(f"Merged {len(region_ids)} regions into {new_name} in map {map_id}")
            
            return {
                "region_id": merged_region_id,
                "name": new_name,
                "region_type": merged_region.region_type,
                "contained_objects_count": len(merged_region.contained_objects),
                "adjacent_regions_count": len(merged_region.adjacent_regions),
                "is_navigable": merged_region.is_navigable
            }
    
    @function_tool
    async def identify_landmarks(self, map_id: str, max_count: int = 5) -> List[Dict[str, Any]]:
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
                return [{"error": f"Map {map_id} not found"}]
            
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
                    size_values = list(obj.size.values())
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
                new_landmarks.append({
                    "id": obj_id,
                    "name": obj.name,
                    "object_type": obj.object_type,
                    "landmark_score": score
                })
            
            logger.info(f"Identified {len(new_landmarks)} new landmarks in map {map_id}")
            
            return new_landmarks
    
    @function_tool
    async def calculate_region_connections(self, map_id: str) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
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
            
            return {
                "connections_count": connections_count,
                "regions_count": len(regions),
                "updated_date": self.maps[map_id].last_updated
            }
    
    async def _are_regions_adjacent(self, region1: SpatialRegion, region2: SpatialRegion) -> bool:
        """Determine if two regions are adjacent"""
        # This is a simplified implementation
        # In a real system, you would check if the polygons share a boundary
        
        # Check if any boundary point of region1 is close to any boundary point of region2
        threshold = 1.0  # Maximum distance to consider regions adjacent
        
        for point1 in region1.boundary_points:
            for point2 in region2.boundary_points:
                if point1.distance_to(point2) < threshold:
                    return True
        
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
                        waypoints: Optional[List[Dict[str, float]]] = None,
                        estimated_time: Optional[float] = None) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
            if route_id not in self.maps[map_id].routes:
                return {"error": f"Route {route_id} not found in map"}
            
            map_obj = self.maps[map_id]
            route = map_obj.routes[route_id]
            
            # Update waypoints if provided
            if waypoints:
                coordinates = []
                for point in waypoints:
                    coord = SpatialCoordinate(
                        x=point.get("x", 0.0),
                        y=point.get("y", 0.0),
                        z=point.get("z") if "z" in point else None,
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
            
            return {
                "route_id": route_id,
                "name": route.name,
                "distance": route.distance,
                "waypoints_count": len(route.waypoints),
                "estimated_time": route.estimated_time,
                "usage_count": route.usage_count
            }
    
    @function_tool
    async def identify_shortcuts(self, map_id: str, max_count: int = 3) -> List[Dict[str, Any]]:
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
                return [{"error": f"Map {map_id} not found"}]
            
            map_obj = self.maps[map_id]
            
            # Need at least a few routes to find shortcuts
            if len(map_obj.routes) < 3:
                return [{"message": "Not enough routes to identify shortcuts"}]
            
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
                        
                        shortcuts.append({
                            "route1_id": route1.id,
                            "route1_name": route1_name,
                            "route2_id": route2.id,
                            "route2_name": route2_name,
                            "distance_between": min_distance,
                            "potential_saving": min(route1.distance, route2.distance) * 0.5,
                            "connection_point1": closest_point1.dict(),
                            "connection_point2": closest_point2.dict()
                        })
            
            # Sort by potential savings and return top results
            shortcuts.sort(key=lambda x: x["potential_saving"], reverse=True)
            return shortcuts[:max_count]
    
    @function_tool
    async def process_spatial_observation(self, observation: SpatialObservation) -> Dict[str, Any]:
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
                return {"error": "No active map set"}
            
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
                result = {"error": f"Unknown observation type: {observation.observation_type}"}
            
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
    
    async def _process_object_observation(self, map_id: str, observation: SpatialObservation) -> Dict[str, Any]:
        """Process an object observation"""
        content = observation.content
        
        # Check if required fields are present
        if "name" not in content or "object_type" not in content:
            return {"error": "Object observation missing required fields"}
        
        # Extract coordinates
        if "coordinates" in content:
            coordinates = content["coordinates"]
        elif observation.observer_position and "relative_position" in content:
            # Calculate absolute position from relative position
            rel_pos = content["relative_position"]
            observer_pos = observation.observer_position
            
            coordinates = {
                "x": observer_pos.x + rel_pos.get("x", 0),
                "y": observer_pos.y + rel_pos.get("y", 0)
            }
            
            if observer_pos.z is not None and "z" in rel_pos:
                coordinates["z"] = observer_pos.z + rel_pos["z"]
        else:
            return {"error": "Object observation missing position information"}
        
        # Check if this is an update to an existing object
        existing_object_id = None
        
        # First, check by name if provided
        if "name" in content:
            for obj_id, obj in self.maps[map_id].spatial_objects.items():
                if obj.name == content["name"]:
                    existing_object_id = obj_id
                    break
        
        # If name not found, try checking by position
        if not existing_object_id and "coordinates" in content:
            coord = SpatialCoordinate(**coordinates)
            closest_distance = float('inf')
            closest_id = None
            
            for obj_id, obj in self.maps[map_id].spatial_objects.items():
                if obj.object_type == content["object_type"]:  # Only match same type
                    distance = coord.distance_to(obj.coordinates)
                    if distance < closest_distance and distance < 2.0:  # Max 2 units distance for match
                        closest_distance = distance
                        closest_id = obj_id
            
            if closest_id:
                existing_object_id = closest_id
        
        # Process size information if provided
        size = content.get("size")
        
        # Process properties
        properties = content.get("properties", {})
        
        # Process is_landmark
        is_landmark = content.get("is_landmark", False)
        
        # Update existing object or create new one
        if existing_object_id:
            result = await self.update_object_position(
                map_id=map_id,
                object_id=existing_object_id,
                new_coordinates=coordinates
            )
            
            # Update other properties if needed
            obj = self.maps[map_id].spatial_objects[existing_object_id]
            
            if size:
                obj.size = size
            
            for key, value in properties.items():
                obj.properties[key] = value
                
            if is_landmark and not obj.is_landmark:
                obj.is_landmark = True
                if existing_object_id not in self.maps[map_id].landmarks:
                    self.maps[map_id].landmarks.append(existing_object_id)
            
            result["action"] = "updated"
            return result
        else:
            # Create new object
            result = await self.add_spatial_object(
                map_id=map_id,
                name=content["name"],
                object_type=content["object_type"],
                coordinates=coordinates,
                size=size,
                is_landmark=is_landmark,
                properties=properties
            )
            
            result["action"] = "created"
            return result
    
    async def _process_region_observation(self, map_id: str, observation: SpatialObservation) -> Dict[str, Any]:
        """Process a region observation"""
        content = observation.content
        
        # Check if required fields are present
        if "name" not in content or "region_type" not in content:
            return {"error": "Region observation missing required fields"}
        
        # Check if boundary points are provided
        if "boundary_points" not in content:
            return {"error": "Region observation missing boundary points"}
        
        # Check if this is an update to an existing region
        existing_region_id = None
        
        # First, check by name if provided
        if "name" in content:
            for region_id, region in self.maps[map_id].regions.items():
                if region.name == content["name"]:
                    existing_region_id = region_id
                    break
        
        # Process properties
        properties = content.get("properties", {})
        
        # Process navigability
        is_navigable = content.get("is_navigable", True)
        
        # Update existing region or create new one
        if existing_region_id:
            # For now, we'll just update the region properties
            region = self.maps[map_id].regions[existing_region_id]
            
            # Update properties
            for key, value in properties.items():
                region.properties[key] = value
                
            region.is_navigable = is_navigable
            
            # Update map
            async with self._lock:
                self.maps[map_id].last_updated = datetime.datetime.now().isoformat()
            
            return {
                "region_id": existing_region_id,
                "name": region.name,
                "action": "updated"
            }
        else:
            # Create new region
            result = await self.define_region(
                map_id=map_id,
                name=content["name"],
                region_type=content["region_type"],
                boundary_points=content["boundary_points"],
                is_navigable=is_navigable,
                properties=properties
            )
            
            result["action"] = "created"
            return result
    
    async def _process_route_observation(self, map_id: str, observation: SpatialObservation) -> Dict[str, Any]:
        """Process a route observation"""
        content = observation.content
        
        # Check if required fields are present
        if "start_id" not in content or "end_id" not in content:
            return {"error": "Route observation missing required fields"}
        
        # Get waypoints if provided
        waypoints = content.get("waypoints")
        
        # Get estimated time if provided
        estimated_time = content.get("estimated_time")
        
        # Check if this is an update to an existing route
        existing_route_id = None
        
        # Check for an existing route between these points
        for route_id, route in self.maps[map_id].routes.items():
            if route.start_id == content["start_id"] and route.end_id == content["end_id"]:
                existing_route_id = route_id
                break
        
        # Process properties
        properties = content.get("properties", {})
        
        # Update existing route or create new one
        if existing_route_id:
            # Update the route
            result = await self.update_route(
                map_id=map_id,
                route_id=existing_route_id,
                waypoints=waypoints,
                estimated_time=estimated_time
            )
            
            # Update properties
            route = self.maps[map_id].routes[existing_route_id]
            for key, value in properties.items():
                route.properties[key] = value
            
            result["action"] = "updated"
            return result
        else:
            # Create new route
            result = await self.create_route(
                map_id=map_id,
                start_id=content["start_id"],
                end_id=content["end_id"],
                waypoints=waypoints,
                name=content.get("name"),
                estimated_time=estimated_time,
                properties=properties
            )
            
            result["action"] = "created"
            return result
    
    async def _process_relative_position(self, map_id: str, observation: SpatialObservation) -> Dict[str, Any]:
        """Process a relative position observation"""
        content = observation.content
        
        # Check if required fields are present
        if "object_id" not in content or "relative_to_id" not in content or "relative_position" not in content:
            return {"error": "Relative position observation missing required fields"}
        
        # Check if objects exist
        if content["object_id"] not in self.maps[map_id].spatial_objects:
            return {"error": f"Object {content['object_id']} not found"}
        
        reference_obj_id = content["relative_to_id"]
        if reference_obj_id not in self.maps[map_id].spatial_objects:
            return {"error": f"Reference object {reference_obj_id} not found"}
        
        # Get reference object position
        reference_obj = self.maps[map_id].spatial_objects[reference_obj_id]
        reference_pos = reference_obj.coordinates
        
        # Calculate absolute position
        rel_pos = content["relative_position"]
        new_coordinates = {
            "x": reference_pos.x + rel_pos.get("x", 0),
            "y": reference_pos.y + rel_pos.get("y", 0)
        }
        
        if reference_pos.z is not None and "z" in rel_pos:
            new_coordinates["z"] = reference_pos.z + rel_pos["z"]
        
        # Update object position
        result = await self.update_object_position(
            map_id=map_id,
            object_id=content["object_id"],
            new_coordinates=new_coordinates
        )
        
        result["action"] = "updated_relative"
        return result
    
    @function_tool
    async def extract_spatial_features(self, text: str) -> Dict[str, Any]:
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
            features = {
                "objects": [],
                "locations": [],
                "spatial_relationships": [],
                "directions": [],
                "confidence": 0.5
            }
            
            # Extract objects (look for nouns)
            objects = ["table", "chair", "door", "window", "wall", "room"]
            for obj in objects:
                if obj in text.lower():
                    features["objects"].append(obj)
            
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
                        features["locations"].append(location_phrase)
            
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
                        features["spatial_relationships"].append(rel_phrase)
            
            # Extract directions (look for cardinal directions and movement verbs)
            directions = ["north", "south", "east", "west", "up", "down", "left", "right"]
            movement_verbs = ["move", "walk", "go", "turn"]
            
            for direction in directions:
                if direction in text.lower():
                    features["directions"].append(direction)
            
            for verb in movement_verbs:
                if verb in text.lower():
                    idx = text.lower().find(verb)
                    if idx >= 0:
                        # Try to extract the direction phrase
                        end_idx = text.find(".", idx)
                        if end_idx < 0:
                            end_idx = len(text)
                        direction_phrase = text[idx:end_idx].strip()
                        features["directions"].append(direction_phrase)
            
            return features
    
    @function_tool
    async def estimate_distances(self, object1_id: str, object2_id: str, 
                              map_id: Optional[str] = None) -> Dict[str, Any]:
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
                    return {"error": "No active map set"}
                map_id = self.context.active_map_id
            
            # Check if map exists
            if map_id not in self.maps:
                return {"error": f"Map {map_id} not found"}
            
            map_obj = self.maps[map_id]
            
            # Check if objects exist
            if object1_id not in map_obj.spatial_objects:
                return {"error": f"Object {object1_id} not found"}
            
            if object2_id not in map_obj.spatial_objects:
                return {"error": f"Object {object2_id} not found"}
            
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
            
            result = {
                "object1": {
                    "id": object1_id,
                    "name": obj1.name
                },
                "object2": {
                    "id": object2_id,
                    "name": obj2.name
                },
                "euclidean_distance": distance,
                "direction": heading,
                "shared_regions": shared_regions,
                "has_direct_route": direct_route is not None
            }
            
            if direct_route:
                result["route"] = {
                    "id": route_id,
                    "distance": direct_route.distance,
                    "estimated_time": direct_route.estimated_time
                }
            
            return result
    
    @function_tool
    async def reconcile_observations(self, map_id: str) -> Dict[str, Any]:
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
                return {"error": f"Map {map_id} not found"}
            
            map_obj = self.maps[map_id]
            
            # Track changes
            changes = {
                "updated_objects": 0,
                "merged_objects": 0,
                "updated_regions": 0
            }
            
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
                            
                            new_coordinates = {
                                "x": avg_x,
                                "y": avg_y
                            }
                            
                            if avg_z is not None:
                                new_coordinates["z"] = avg_z
                            
                            # Update position of keeper
                            await self.update_object_position(
                                map_id=map_id,
                                object_id=keeper.id,
                                new_coordinates=new_coordinates
                            )
                            
                            # Merge properties
                            for key, value in map_obj.spatial_objects[remove_id].properties.items():
                                if key not in keeper.properties:
                                    keeper.properties[key] = value
                            
                            # Combine visibility
                            keeper.visibility = max(keeper.visibility, map_obj.spatial_objects[remove_id].visibility)
                            
                            # Update observation count
                            keeper.observation_count += map_obj.spatial_objects[remove_id].observation_count
                            
                            # Transfer connections
                            for conn in map_obj.spatial_objects[remove_id].connections:
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
                            
                            changes["merged_objects"] += 1
            
            # Update map
            async with self._lock:
                map_obj.last_updated = datetime.datetime.now().isoformat()
            
            # Update map accuracy based on reconciliation
            accuracy_improvement = min(0.1, changes["merged_objects"] * 0.02)
            map_obj.accuracy = min(1.0, map_obj.accuracy + accuracy_improvement)
            
            return {
                "changes": changes,
                "map_accuracy": map_obj.accuracy,
                "map_completeness": map_obj.completeness
            }
