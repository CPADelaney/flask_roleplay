# nyx/core/spatial/spatial_memory.py

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime

from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, RunContextWrapper
from agents.tracing import custom_span

from nyx.core.spatial.spatial_schemas import (
    SpatialCoordinate, SpatialEvent, SpatialMemoryReference, 
    LandmarkInfo, EnvironmentChange
)

logger = logging.getLogger(__name__)

# Pydantic models for function tool inputs/outputs
class CoordinatesInput(BaseModel):
    """Coordinates input model"""
    x: float
    y: float
    z: Optional[float] = None

class PropertiesInput(BaseModel):
    """Generic properties model"""
    significance: Optional[int] = None
    change_type: Optional[str] = None
    map_id: Optional[str] = None
    # Allow any additional fields by not specifying them

class StateInput(BaseModel):
    """State information model"""
    state_data: Optional[str] = None  # JSON string representation of state

class StoreMemoryResult(BaseModel):
    """Result of storing a spatial memory"""
    memory_id: str
    success: bool
    location_id: Optional[str] = None
    coordinates: Optional[CoordinatesInput] = None
    map_id: Optional[str] = None
    error: Optional[str] = None

class MemoryInfo(BaseModel):
    """Information about a retrieved memory"""
    memory_id: str
    memory_text: Optional[str] = None
    description: Optional[str] = None
    memory_type: Optional[str] = None
    significance: Optional[int] = None
    timestamp: Optional[str] = None
    location_id: Optional[str] = None
    distance: Optional[float] = None
    coordinates: Optional[CoordinatesInput] = None

class EventResult(BaseModel):
    """Result of recording a spatial event"""
    event_id: str
    memory_id: Optional[str] = None
    event_type: str
    location_id: Optional[str] = None
    coordinates: Optional[CoordinatesInput] = None
    timestamp: str

class LandmarkResult(BaseModel):
    """Result of registering an episodic landmark"""
    landmark_id: str
    name: str
    landmark_type: str
    salience: float
    map_id: str
    error: Optional[str] = None

class ChangeResult(BaseModel):
    """Result of recording an environment change"""
    change_type: str
    timestamp: str
    object_id: Optional[str] = None
    region_id: Optional[str] = None
    map_id: str
    error: Optional[str] = None

class LandmarkInfo(BaseModel):
    """Information about an episodic landmark"""
    landmark_id: str
    name: str
    landmark_type: str
    salience: float
    description: str
    coordinates: CoordinatesInput
    visibility_radius: Optional[float] = None

class ChangeInfo(BaseModel):
    """Information about an environment change"""
    change_type: str
    timestamp: str
    object_id: Optional[str] = None
    region_id: Optional[str] = None
    description: str
    old_state: Optional[StateInput] = None
    new_state: Optional[StateInput] = None

class SpatialMemoryIntegration:
    """
    Integrates spatial cognition with the memory system.
    Manages spatial memories, episodic events at locations, and retrieval based on spatial context.
    """
    
    def __init__(self, spatial_mapper, memory_core=None):
        """
        Initialize with spatial mapper and memory core references
        
        Args:
            spatial_mapper: Reference to the SpatialMapper
            memory_core: Optional reference to MemoryCore
        """
        self.spatial_mapper = spatial_mapper
        self.memory_core = memory_core
        
        # Register with the spatial mapper
        if spatial_mapper:
            spatial_mapper.memory_integration = self
        
        # Local storage for spatial memory references
        self.spatial_memories: Dict[str, SpatialMemoryReference] = {}
        
        # Index from locations to memories
        self.location_memory_index: Dict[str, Set[str]] = {}  # location_id -> set of memory_ids
        
        # Index from coordinates to memories (grid-based for efficient lookup)
        self.coordinate_grid: Dict[Tuple[int, int], Set[str]] = {}  # (grid_x, grid_y) -> set of memory_ids
        self.grid_cell_size = 5.0  # Size of each grid cell for spatial indexing
        
        # Events that occurred at locations
        self.spatial_events: Dict[str, SpatialEvent] = {}
        
        # Salient landmarks with episodic significance
        self.episodic_landmarks: Dict[str, LandmarkInfo] = {}
        
        # Environment changes
        self.environment_changes: List[EnvironmentChange] = []
        
        # Memory agent for processing and retrieving spatial memories
        self._spatial_memory_agent = self._create_spatial_memory_agent()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("SpatialMemoryIntegration initialized")
    
    def _create_spatial_memory_agent(self) -> Agent:
        """Create an agent for spatial memory processing"""
        return Agent(
            name="Spatial Memory Agent",
            instructions="""
            You are a Spatial Memory Agent that connects spatial information with episodic memories.
            
            Your role is to:
            1. Create meaningful memories linked to locations
            2. Retrieve memories based on spatial context
            3. Identify important landmarks with episodic significance
            4. Track changes in environments over time
            5. Help build cognitive maps with personal significance
            
            Focus on making spatial memories vivid, useful, and emotionally salient.
            Connect locations to the experiences and events that happened there.
            """,
            tools=[
                self.store_spatial_memory,
                self.retrieve_memories_at_location,
                self.retrieve_memories_near_coordinates,
                self.record_spatial_event,
                self.register_episodic_landmark,
                self.record_environment_change,
                self.get_episodic_landmarks,
                self.get_environment_changes
            ],
            model="gpt-5-nano"
        )
    
    @function_tool
    async def store_spatial_memory(self, 
                                memory_text: str,
                                location_id: Optional[str] = None,
                                coordinates: Optional[CoordinatesInput] = None,
                                map_id: Optional[str] = None,
                                tags: List[str] = None,
                                memory_type: str = "observation",
                                significance: int = 5) -> StoreMemoryResult:
        """
        Store a memory linked to a spatial location
        
        Args:
            memory_text: Text content of the memory
            location_id: Optional ID of associated object or region
            coordinates: Optional coordinates where memory occurred
            map_id: Optional map ID (uses active map if not provided)
            tags: Optional tags for the memory
            memory_type: Type of memory (observation, reflection, experience)
            significance: Importance level (1-10)
            
        Returns:
            Information about the stored memory
        """
        with custom_span("store_spatial_memory"):
            # Set default tags
            if tags is None:
                tags = ["spatial"]
            else:
                # Ensure "spatial" tag is included
                if "spatial" not in tags:
                    tags.append("spatial")
            
            # Use active map if not provided
            if not map_id and self.spatial_mapper:
                if not self.spatial_mapper.context.active_map_id:
                    return StoreMemoryResult(
                        memory_id="",
                        success=False,
                        error="No active map set and no map_id provided"
                    )
                map_id = self.spatial_mapper.context.active_map_id
            
            # Prepare metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "spatial": True,
                "map_id": map_id
            }
            
            # Add location information
            if location_id:
                metadata["location_id"] = location_id
                
                # Add location name if available
                if self.spatial_mapper and map_id in self.spatial_mapper.maps:
                    map_obj = self.spatial_mapper.maps[map_id]
                    
                    if location_id in map_obj.spatial_objects:
                        metadata["location_name"] = map_obj.spatial_objects[location_id].name
                    elif location_id in map_obj.regions:
                        metadata["location_name"] = map_obj.regions[location_id].name
            
            # Add coordinates
            if coordinates:
                metadata["coordinates"] = coordinates.model_dump()
            
            # Store in memory core if available
            memory_id = None
            if self.memory_core and hasattr(self.memory_core, 'add_memory'):
                try:
                    memory_id = await self.memory_core.add_memory(
                        memory_text=memory_text,
                        memory_type=memory_type,
                        memory_scope="spatial",
                        significance=significance,
                        tags=tags,
                        metadata=metadata
                    )
                    logger.info(f"Stored spatial memory with ID: {memory_id}")
                except Exception as e:
                    logger.error(f"Error storing in memory core: {e}")
                    memory_id = f"local_{datetime.now().timestamp()}"
            else:
                # Generate a local ID if memory core not available
                memory_id = f"local_{datetime.now().timestamp()}"
            
            # Create memory reference
            memory_ref = SpatialMemoryReference(
                memory_id=memory_id,
                relevance=1.0,
                location_id=location_id,
                coordinates=SpatialCoordinate(**coordinates.model_dump()) if coordinates else None,
                timestamp=metadata["timestamp"],
                description=memory_text[:100]  # Store a brief description
            )
            
            # Store in local indices
            async with self._lock:
                # Store reference
                self.spatial_memories[memory_id] = memory_ref
                
                # Update location index
                if location_id:
                    if location_id not in self.location_memory_index:
                        self.location_memory_index[location_id] = set()
                    self.location_memory_index[location_id].add(memory_id)
                
                # Update coordinate grid
                if coordinates:
                    grid_x = int(coordinates.x / self.grid_cell_size)
                    grid_y = int(coordinates.y / self.grid_cell_size)
                    grid_key = (grid_x, grid_y)
                    
                    if grid_key not in self.coordinate_grid:
                        self.coordinate_grid[grid_key] = set()
                    self.coordinate_grid[grid_key].add(memory_id)
            
            return StoreMemoryResult(
                memory_id=memory_id,
                success=True,
                location_id=location_id,
                coordinates=coordinates,
                map_id=map_id
            )
    
    @function_tool
    async def retrieve_memories_at_location(self, 
                                         location_id: str,
                                         limit: int = 5,
                                         min_significance: int = 0,
                                         map_id: Optional[str] = None) -> List[MemoryInfo]:
        """
        Retrieve memories associated with a location
        
        Args:
            location_id: ID of the location (object or region)
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold
            map_id: Optional map ID (uses active map if not provided)
            
        Returns:
            List of memories associated with the location
        """
        with custom_span("retrieve_memories_at_location", {"location_id": location_id}):
            # Check if location is in index
            if location_id not in self.location_memory_index:
                return []
            
            memory_ids = self.location_memory_index[location_id]
            results = []
            
            # If memory core is available, retrieve from it
            if self.memory_core and hasattr(self.memory_core, 'get_memory'):
                for memory_id in memory_ids:
                    try:
                        memory = await self.memory_core.get_memory(memory_id)
                        
                        # Apply significance filter
                        if memory and memory.get("significance", 0) >= min_significance:
                            # Format memory for return
                            results.append(MemoryInfo(
                                memory_id=memory_id,
                                memory_text=memory.get("memory_text", ""),
                                memory_type=memory.get("memory_type", "observation"),
                                significance=memory.get("significance", 5),
                                timestamp=memory.get("metadata", {}).get("timestamp", ""),
                                location_id=location_id
                            ))
                    except Exception as e:
                        logger.error(f"Error retrieving memory {memory_id}: {e}")
            else:
                # Use local references if memory core not available
                for memory_id in memory_ids:
                    if memory_id in self.spatial_memories:
                        memory_ref = self.spatial_memories[memory_id]
                        results.append(MemoryInfo(
                            memory_id=memory_id,
                            description=memory_ref.description,
                            timestamp=memory_ref.timestamp,
                            location_id=location_id
                        ))
            
            # Sort by timestamp (most recent first) and limit results
            results.sort(key=lambda x: x.timestamp or "", reverse=True)
            return results[:limit]
    
    @function_tool
    async def retrieve_memories_near_coordinates(self, 
                                              coordinates: CoordinatesInput,
                                              radius: float = 10.0,
                                              limit: int = 5,
                                              min_significance: int = 0) -> List[MemoryInfo]:
        """
        Retrieve memories near specified coordinates
        
        Args:
            coordinates: Center coordinates to search around
            radius: Search radius around coordinates
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of memories near the coordinates
        """
        with custom_span("retrieve_memories_near_coordinates"):
            results = []
            coord = SpatialCoordinate(**coordinates.model_dump())
            
            # Calculate grid cell range to search
            min_grid_x = int((coordinates.x - radius) / self.grid_cell_size)
            max_grid_x = int((coordinates.x + radius) / self.grid_cell_size)
            min_grid_y = int((coordinates.y - radius) / self.grid_cell_size)
            max_grid_y = int((coordinates.y + radius) / self.grid_cell_size)
            
            # Collect memory IDs from all relevant grid cells
            candidate_memory_ids = set()
            for grid_x in range(min_grid_x, max_grid_x + 1):
                for grid_y in range(min_grid_y, max_grid_y + 1):
                    grid_key = (grid_x, grid_y)
                    if grid_key in self.coordinate_grid:
                        candidate_memory_ids.update(self.coordinate_grid[grid_key])
            
            # If memory core is available, retrieve from it
            if self.memory_core and hasattr(self.memory_core, 'get_memory'):
                for memory_id in candidate_memory_ids:
                    try:
                        memory = await self.memory_core.get_memory(memory_id)
                        
                        if not memory:
                            continue
                            
                        # Check if within radius
                        mem_coords = memory.get("metadata", {}).get("coordinates")
                        if mem_coords:
                            mem_coord = SpatialCoordinate(**mem_coords)
                            distance = coord.distance_to(mem_coord) if hasattr(coord, "distance_to") else self._calculate_distance(coord, mem_coord)
                            
                            # Skip if outside radius
                            if distance > radius:
                                continue
                            
                            # Apply significance filter
                            if memory.get("significance", 0) >= min_significance:
                                # Format memory for return
                                results.append(MemoryInfo(
                                    memory_id=memory_id,
                                    memory_text=memory.get("memory_text", ""),
                                    memory_type=memory.get("memory_type", "observation"),
                                    significance=memory.get("significance", 5),
                                    timestamp=memory.get("metadata", {}).get("timestamp", ""),
                                    distance=distance,
                                    coordinates=CoordinatesInput(**mem_coords)
                                ))
                    except Exception as e:
                        logger.error(f"Error retrieving memory {memory_id}: {e}")
            else:
                # Use local references if memory core not available
                for memory_id in candidate_memory_ids:
                    if memory_id in self.spatial_memories:
                        memory_ref = self.spatial_memories[memory_id]
                        
                        if memory_ref.coordinates:
                            # Check if within radius
                            distance = coord.distance_to(memory_ref.coordinates) if hasattr(coord, "distance_to") else self._calculate_distance(coord, memory_ref.coordinates)
                            
                            if distance <= radius:
                                results.append(MemoryInfo(
                                    memory_id=memory_id,
                                    description=memory_ref.description,
                                    timestamp=memory_ref.timestamp,
                                    distance=distance,
                                    coordinates=CoordinatesInput(**memory_ref.coordinates.dict())
                                ))
            
            # Sort by distance and limit results
            results.sort(key=lambda x: x.distance or float('inf'))
            return results[:limit]
    
    def _calculate_distance(self, coord1: SpatialCoordinate, coord2: SpatialCoordinate) -> float:
        """Calculate Euclidean distance between coordinates"""
        dx = coord1.x - coord2.x
        dy = coord1.y - coord2.y
        
        distance = (dx**2 + dy**2)**0.5
        
        # Use z-coordinate if available in both
        if coord1.z is not None and coord2.z is not None:
            dz = coord1.z - coord2.z
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
        return distance
    
    @function_tool
    async def record_spatial_event(self,
                                event_type: str,
                                description: str,
                                location_id: Optional[str] = None,
                                coordinates: Optional[CoordinatesInput] = None,
                                participants: List[str] = None,
                                properties: Optional[PropertiesInput] = None,
                                store_as_memory: bool = True) -> EventResult:
        """
        Record an event that occurred at a specific location
        
        Args:
            event_type: Type of event
            description: Description of the event
            location_id: Optional ID of the location (object or region)
            coordinates: Optional coordinates where the event occurred
            participants: Optional list of participants
            properties: Optional additional properties
            store_as_memory: Whether to also store as a spatial memory
            
        Returns:
            Information about the recorded event
        """
        with custom_span("record_spatial_event", {"event_type": event_type}):
            # Generate event ID
            event_id = f"event_{datetime.now().timestamp()}"
            
            # Create event object
            event = SpatialEvent(
                event_id=event_id,
                event_type=event_type,
                location_id=location_id,
                coordinates=SpatialCoordinate(**coordinates.model_dump()) if coordinates else None,
                timestamp=datetime.now().isoformat(),
                description=description,
                participants=participants or [],
                properties=properties.model_dump() if properties else {}
            )
            
            # Store as memory if requested
            memory_id = None
            if store_as_memory:
                memory_result = await self.store_spatial_memory(
                    memory_text=f"Event: {description}",
                    location_id=location_id,
                    coordinates=coordinates,
                    tags=["spatial", "event", event_type],
                    memory_type="experience",
                    significance=properties.significance if properties and properties.significance else 6
                )
                
                memory_id = memory_result.memory_id
                event.memory_id = memory_id
            
            # Store event
            async with self._lock:
                self.spatial_events[event_id] = event
            
            return EventResult(
                event_id=event_id,
                memory_id=memory_id,
                event_type=event_type,
                location_id=location_id,
                coordinates=coordinates,
                timestamp=event.timestamp
            )
    
    @function_tool
    async def register_episodic_landmark(self,
                                      landmark_id: str,
                                      map_id: Optional[str] = None,
                                      description: Optional[str] = None,
                                      salience: float = 1.0,
                                      visibility_radius: Optional[float] = None,
                                      properties: Optional[PropertiesInput] = None) -> LandmarkResult:
        """
        Register a landmark with episodic significance
        
        Args:
            landmark_id: ID of the landmark object
            map_id: Optional map ID (uses active map if not provided)
            description: Optional description of the landmark's significance
            salience: How salient/memorable this landmark is (0.0-1.0)
            visibility_radius: Optional radius where landmark is visible
            properties: Optional additional properties
            
        Returns:
            Information about the registered landmark
        """
        with custom_span("register_episodic_landmark", {"landmark_id": landmark_id}):
            # Use active map if not provided
            if not map_id and self.spatial_mapper:
                if not self.spatial_mapper.context.active_map_id:
                    return LandmarkResult(
                        landmark_id=landmark_id,
                        name="",
                        landmark_type="",
                        salience=0.0,
                        map_id="",
                        error="No active map set and no map_id provided"
                    )
                map_id = self.spatial_mapper.context.active_map_id
            
            # Check if landmark exists
            if not self.spatial_mapper or map_id not in self.spatial_mapper.maps:
                return LandmarkResult(
                    landmark_id=landmark_id,
                    name="",
                    landmark_type="",
                    salience=0.0,
                    map_id=map_id or "",
                    error=f"Map {map_id} not found"
                )
                
            map_obj = self.spatial_mapper.maps[map_id]
            
            if landmark_id not in map_obj.spatial_objects:
                return LandmarkResult(
                    landmark_id=landmark_id,
                    name="",
                    landmark_type="",
                    salience=0.0,
                    map_id=map_id,
                    error=f"Landmark {landmark_id} not found in map {map_id}"
                )
            
            # Get landmark object
            landmark_obj = map_obj.spatial_objects[landmark_id]
            
            # Make sure it's marked as a landmark
            if not landmark_obj.is_landmark:
                landmark_obj.is_landmark = True
                if landmark_id not in map_obj.landmarks:
                    map_obj.landmarks.append(landmark_id)
            
            # Create landmark info
            landmark_info = LandmarkInfo(
                landmark_id=landmark_id,
                name=landmark_obj.name,
                landmark_type=landmark_obj.object_type,
                salience=salience,
                description=description or f"Landmark: {landmark_obj.name}",
                coordinates=CoordinatesInput(**landmark_obj.coordinates.dict()),
                visibility_radius=visibility_radius
            )
            
            # Store landmark info
            async with self._lock:
                self.episodic_landmarks[landmark_id] = landmark_info
            
            return LandmarkResult(
                landmark_id=landmark_id,
                name=landmark_obj.name,
                landmark_type=landmark_obj.object_type,
                salience=salience,
                map_id=map_id
            )
    
    @function_tool
    async def record_environment_change(self,
                                     change_type: str,
                                     description: str,
                                     object_id: Optional[str] = None,
                                     region_id: Optional[str] = None,
                                     old_state: Optional[StateInput] = None,
                                     new_state: Optional[StateInput] = None,
                                     map_id: Optional[str] = None) -> ChangeResult:
        """
        Record a change in the environment
        
        Args:
            change_type: Type of change (addition, removal, movement, etc.)
            description: Description of the change
            object_id: Optional ID of affected object
            region_id: Optional ID of affected region
            old_state: Optional previous state
            new_state: Optional new state
            map_id: Optional map ID (uses active map if not provided)
            
        Returns:
            Information about the recorded change
        """
        with custom_span("record_environment_change", {"change_type": change_type}):
            # Use active map if not provided
            if not map_id and self.spatial_mapper:
                if not self.spatial_mapper.context.active_map_id:
                    return ChangeResult(
                        change_type=change_type,
                        timestamp=datetime.now().isoformat(),
                        map_id="",
                        error="No active map set and no map_id provided"
                    )
                map_id = self.spatial_mapper.context.active_map_id
            
            # Create change object
            change = EnvironmentChange(
                change_type=change_type,
                object_id=object_id,
                region_id=region_id,
                old_state=old_state.model_dump() if old_state else None,
                new_state=new_state.model_dump() if new_state else None,
                timestamp=datetime.now().isoformat(),
                description=description
            )
            
            # Store change
            async with self._lock:
                self.environment_changes.append(change)
            
            # Also record as a spatial event
            await self.record_spatial_event(
                event_type=f"environment_{change_type}",
                description=description,
                location_id=object_id or region_id,
                properties=PropertiesInput(
                    change_type=change_type,
                    map_id=map_id
                ),
                store_as_memory=True
            )
            
            return ChangeResult(
                change_type=change_type,
                timestamp=change.timestamp,
                object_id=object_id,
                region_id=region_id,
                map_id=map_id
            )
    
    @function_tool
    async def get_episodic_landmarks(self, 
                                  map_id: Optional[str] = None, 
                                  min_salience: float = 0.0) -> List[LandmarkInfo]:
        """
        Get episodic landmarks
        
        Args:
            map_id: Optional map ID to filter landmarks
            min_salience: Minimum salience threshold
            
        Returns:
            List of episodic landmarks
        """
        results = []
        
        for landmark_id, landmark in self.episodic_landmarks.items():
            # Apply salience filter
            if landmark.salience < min_salience:
                continue
                
            # Apply map filter if specified
            if map_id and self.spatial_mapper:
                if map_id in self.spatial_mapper.maps:
                    map_obj = self.spatial_mapper.maps[map_id]
                    
                    # Skip if landmark not in this map
                    if landmark_id not in map_obj.spatial_objects:
                        continue
            
            # Add to results
            results.append(landmark)
        
        # Sort by salience (highest first)
        results.sort(key=lambda x: x.salience, reverse=True)
        return results
    
    @function_tool
    async def get_environment_changes(self, 
                                   map_id: Optional[str] = None,
                                   object_id: Optional[str] = None,
                                   region_id: Optional[str] = None,
                                   change_types: Optional[List[str]] = None,
                                   limit: int = 10) -> List[ChangeInfo]:
        """
        Get environment changes
        
        Args:
            map_id: Optional map ID to filter changes
            object_id: Optional object ID to filter changes
            region_id: Optional region ID to filter changes
            change_types: Optional list of change types to filter
            limit: Maximum number of changes to return
            
        Returns:
            List of environment changes
        """
        results = []
        
        for change in self.environment_changes:
            # Apply object filter
            if object_id and change.object_id != object_id:
                continue
                
            # Apply region filter
            if region_id and change.region_id != region_id:
                continue
                
            # Apply change type filter
            if change_types and change.change_type not in change_types:
                continue
            
            # Add to results
            results.append(ChangeInfo(
                change_type=change.change_type,
                timestamp=change.timestamp,
                object_id=change.object_id,
                region_id=change.region_id,
                description=change.description,
                old_state=StateInput(state_data=str(change.old_state)) if change.old_state else None,
                new_state=StateInput(state_data=str(change.new_state)) if change.new_state else None
            ))
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]
