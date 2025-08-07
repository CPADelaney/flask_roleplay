# nyx/core/spatial/navigator_agent.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel

from agents import Agent, Runner, function_tool, handoff, trace, ModelSettings, RunContextWrapper

from nyx.core.spatial.spatial_mapper import SpatialMapper
from nyx.core.spatial.spatial_schemas import (
    SpatialObservation, NavigationRequest, SpatialQuery, 
    NavigationResult, SpatialDescription
)

logger = logging.getLogger(__name__)

# Create explicit Pydantic models for Dict types used in function tools
class PositionDict(BaseModel):
    """Position coordinates"""
    x: float
    y: float
    z: Optional[float] = None

class ProcessingResult(BaseModel):
    """Result from processing spatial description"""
    message: Optional[str] = None
    error: Optional[str] = None
    map_id: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    final_output: Optional[Any] = None

class SpatialNavigatorAgent:
    """
    Agent responsible for navigating in space and providing spatial guidance.
    Integrates with the SpatialMapper to understand and traverse environments.
    """
    
    def __init__(self, spatial_mapper: SpatialMapper):
        """Initialize with a spatial mapper"""
        self.spatial_mapper = spatial_mapper
        self.navigator_agent = self._create_navigator_agent()
        self.mapper_agent = self._create_map_builder_agent()
        self.triage_agent = self._create_triage_agent()
        
    def _create_triage_agent(self) -> Agent:
        """Create the main triage agent for spatial cognition"""
        return Agent(
            name="Spatial Cognition Agent",
            instructions="""
            You are the Spatial Cognition Agent that helps understand and navigate through environments.
            
            You help create and maintain cognitive maps, which are mental representations of spaces.
            You provide navigation assistance, spatial reasoning, and help understand environments.
            
            For observation processing, mapping, or spatial understanding tasks, hand off to the Map Builder agent.
            For navigation, wayfinding, or route planning tasks, hand off to the Spatial Navigator agent.
            
            Respond helpfully to spatial queries and try to understand the spatial needs behind user requests.
            """,
            handoffs=[
                handoff(self.navigator_agent, 
                        tool_description_override="For navigation, route finding, and direction giving tasks"),
                handoff(self.mapper_agent, 
                        tool_description_override="For map building, observation processing, and spatial environment understanding")
            ],
            model="gpt-5-nano"
        )
    
    def _create_navigator_agent(self) -> Agent:
        """Create the spatial navigation agent"""
        return Agent(
            name="Spatial Navigator",
            instructions="""
            You are a Spatial Navigator agent that helps users navigate through environments.
            
            Your capabilities include:
            1. Finding paths between locations
            2. Giving step-by-step directions
            3. Identifying shortcuts and landmarks
            4. Creating and updating routes
            5. Helping users understand spatial relationships
            
            Provide clear, concise navigation instructions using landmarks when possible.
            Explain routes in terms of observable features and easy-to-follow directions.
            """,
            tools=[
                self.spatial_mapper.find_path,
                self.spatial_mapper.calculate_directions,
                self.spatial_mapper.identify_shortcuts,
                self.spatial_mapper.get_nearest_landmarks,
                self.spatial_mapper.update_route,
                self.navigate_to_location,
                self.describe_surroundings,
                self.find_path_to_landmark
            ],
            model_settings=ModelSettings(
                temperature=0.2  # Lower temperature for more predictable navigation
            ),
            model="gpt-5-nano"
        )
    
    def _create_map_builder_agent(self) -> Agent:
        """Create the cognitive map builder agent"""
        return Agent(
            name="Map Builder",
            instructions="""
            You are a Map Builder agent that helps create and update cognitive maps of environments.
            
            Your capabilities include:
            1. Processing spatial observations
            2. Creating and updating cognitive maps
            3. Identifying landmarks and regions
            4. Reconciling different observations
            5. Extracting spatial information from descriptions
            
            Help users build and maintain accurate mental models of spaces by organizing
            spatial information into coherent cognitive maps. Pay attention to landmarks,
            boundaries, and spatial relationships.
            """,
            tools=[
                self.spatial_mapper.create_cognitive_map,
                self.spatial_mapper.add_spatial_object,
                self.spatial_mapper.define_region,
                self.spatial_mapper.process_spatial_observation,
                self.spatial_mapper.extract_spatial_features,
                self.spatial_mapper.get_map,
                self.spatial_mapper.identify_landmarks,
                self.spatial_mapper.reconcile_observations,
                self.process_spatial_description
            ],
            model_settings=ModelSettings(
                temperature=0.3
            ),
            model="gpt-5-nano"
        )
    
    @function_tool
    async def navigate_to_location(self, 
                                location_name: str, 
                                current_position: Optional[PositionDict] = None,
                                map_id: Optional[str] = None) -> NavigationResult:
        """
        Navigate to a named location
        
        Args:
            location_name: Name of the location to navigate to
            current_position: Optional current position coordinates
            map_id: Optional map ID (uses active map if not provided)
            
        Returns:
            Navigation instructions and path information
        """
        # Use active map if not provided
        if not map_id:
            if not self.spatial_mapper.context.active_map_id:
                return NavigationResult(
                    success=False,
                    message="No active map set",
                    directions=[]
                )
            map_id = self.spatial_mapper.context.active_map_id
        
        # Find the location by name
        target_id = None
        start_id = None
        
        # Look through objects and regions for the target name
        for obj_id, obj in self.spatial_mapper.maps[map_id].spatial_objects.items():
            if obj.name.lower() == location_name.lower():
                target_id = obj_id
                break
                
        if not target_id:
            # Look through regions
            for region_id, region in self.spatial_mapper.maps[map_id].regions.items():
                if region.name.lower() == location_name.lower():
                    target_id = region_id
                    break
        
        if not target_id:
            return NavigationResult(
                success=False,
                message=f"Could not find a location named '{location_name}'",
                directions=[]
            )
        
        # If current position provided, create a temporary starting point
        if current_position:
            # Create a temporary object for the current position
            start_id = await self._create_temporary_position(map_id, current_position.model_dump())
        else:
            # Use observer position if available
            if self.spatial_mapper.context.observer_position:
                pos = self.spatial_mapper.context.observer_position
                start_id = await self._create_temporary_position(
                    map_id, {"x": pos.x, "y": pos.y, "z": pos.z if pos.z is not None else None}
                )
            else:
                return NavigationResult(
                    success=False,
                    message="No current position provided or available",
                    directions=[]
                )
        
        # Find path
        path_result = await self.spatial_mapper.find_path(
            map_id=map_id,
            start_id=start_id,
            end_id=target_id,
            prefer_landmarks=True
        )
        
        # Clean up temporary position if created
        if current_position or self.spatial_mapper.context.observer_position:
            # Remove temporary object
            await self._cleanup_temporary_position(map_id, start_id)
        
        if "error" in path_result:
            return NavigationResult(
                success=False,
                message=f"Navigation error: {path_result['error']}",
                directions=[]
            )
        
        return NavigationResult(
            success=True,
            message=f"Found route to {location_name}",
            directions=path_result.get("directions", []),
            distance=path_result.get("distance", 0),
            estimated_time=path_result.get("estimated_time"),
            route_id=path_result.get("route_id")
        )

    async def get_current_location(self) -> Optional[Dict[str, Any]]:
        """
        Get the current location of the observer
        
        Returns:
            Current location information or None if not available
        """
        # Check if we have an observer position in the spatial mapper context
        if self.spatial_mapper and self.spatial_mapper.context.observer_position:
            pos = self.spatial_mapper.context.observer_position
            
            # Get the active map ID
            map_id = self.spatial_mapper.context.active_map_id
            
            # Find which region the observer is in (if any)
            current_region = None
            if map_id and map_id in self.spatial_mapper.maps:
                map_obj = self.spatial_mapper.maps[map_id]
                for region_id, region in map_obj.regions.items():
                    if region.contains_point(pos):
                        current_region = {
                            "id": region_id,
                            "name": region.name,
                            "type": region.region_type
                        }
                        break
            
            return {
                "coordinates": {
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z
                },
                "map_id": map_id,
                "region": current_region,
                "reference_frame": pos.reference_frame
            }
        
        return None
    
    async def _create_temporary_position(self, map_id: str, position: Dict[str, float]) -> str:
        """Create a temporary position marker"""
        result = await self.spatial_mapper.add_spatial_object(
            map_id=map_id,
            name="Current Position",
            object_type="position_marker",
            coordinates=position,
            properties={"temporary": True}
        )
        
        return result["object_id"]
    
    async def _cleanup_temporary_position(self, map_id: str, object_id: str) -> None:
        """Remove a temporary position marker"""
        # Check if object exists and is a temporary marker
        if (map_id in self.spatial_mapper.maps and 
            object_id in self.spatial_mapper.maps[map_id].spatial_objects):
            
            obj = self.spatial_mapper.maps[map_id].spatial_objects[object_id]
            if (obj.object_type == "position_marker" and 
                obj.properties.get("temporary", False)):
                
                # Delete the object
                if hasattr(self.spatial_mapper, "delete_spatial_object"):
                    await self.spatial_mapper.delete_spatial_object(map_id, object_id)
                else:
                    # Manual deletion if method not available
                    del self.spatial_mapper.maps[map_id].spatial_objects[object_id]
    
    @function_tool
    async def describe_surroundings(self, 
                                 position: Optional[PositionDict] = None,
                                 radius: float = 10.0,
                                 map_id: Optional[str] = None) -> SpatialDescription:
        """
        Describe the surroundings at a position
        
        Args:
            position: Optional position coordinates (uses current position if not provided)
            radius: Radius to consider for surroundings
            map_id: Optional map ID (uses active map if not provided)
            
        Returns:
            Description of the surroundings
        """
        # Use active map if not provided
        if not map_id:
            if not self.spatial_mapper.context.active_map_id:
                return SpatialDescription(
                    description="No active map set",
                    objects=[],
                    regions=[],
                    landmarks=[]
                )
            map_id = self.spatial_mapper.context.active_map_id
        
        # Get position to use
        use_position = None
        
        if position:
            use_position = position.model_dump()
        elif self.spatial_mapper.context.observer_position:
            pos = self.spatial_mapper.context.observer_position
            use_position = {"x": pos.x, "y": pos.y, "z": pos.z if pos.z is not None else None}
        
        if not use_position:
            return SpatialDescription(
                description="No position provided or available",
                objects=[],
                regions=[],
                landmarks=[]
            )
        
        map_obj = self.spatial_mapper.maps[map_id]
        
        # Find nearby objects
        nearby_objects = []
        current_regions = []
        nearby_landmarks = []
        
        for obj_id, obj in map_obj.spatial_objects.items():
            # Calculate distance
            dx = obj.coordinates.x - use_position["x"]
            dy = obj.coordinates.y - use_position["y"]
            dz = 0
            if "z" in use_position and use_position["z"] is not None and obj.coordinates.z is not None:
                dz = obj.coordinates.z - use_position["z"]
            
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            if distance <= radius:
                direction = self.spatial_mapper._calculate_heading(
                    self.spatial_mapper.SpatialCoordinate(**use_position),
                    obj.coordinates
                )
                
                nearby_objects.append({
                    "id": obj_id,
                    "name": obj.name,
                    "object_type": obj.object_type,
                    "distance": distance,
                    "direction": direction,
                    "is_landmark": obj.is_landmark
                })
                
                if obj.is_landmark:
                    nearby_landmarks.append({
                        "id": obj_id,
                        "name": obj.name,
                        "distance": distance,
                        "direction": direction
                    })
        
        # Find regions containing the position
        pos_coord = self.spatial_mapper.SpatialCoordinate(**use_position)
        
        for region_id, region in map_obj.regions.items():
            if region.contains_point(pos_coord):
                current_regions.append({
                    "id": region_id,
                    "name": region.name,
                    "region_type": region.region_type,
                    "is_navigable": region.is_navigable
                })
        
        # Sort objects by distance
        nearby_objects.sort(key=lambda x: x["distance"])
        nearby_landmarks.sort(key=lambda x: x["distance"])
        
        # Generate description
        location_desc = "an unknown area"
        if current_regions:
            location_desc = f"the {current_regions[0]['name']}"
            if len(current_regions) > 1:
                location_desc += f" (also in {', '.join([r['name'] for r in current_regions[1:]])}"
        
        objects_desc = ""
        if nearby_objects:
            objects_desc = f"There are {len(nearby_objects)} objects nearby, "
            if len(nearby_objects) <= 5:
                obj_names = [f"{obj['name']} to the {obj['direction']}" for obj in nearby_objects]
                objects_desc += f"including {', '.join(obj_names[:-1])} and {obj_names[-1]}." if len(obj_names) > 1 else f"including {obj_names[0]}."
            else:
                obj_names = [f"{obj['name']} to the {obj['direction']}" for obj in nearby_objects[:5]]
                objects_desc += f"including {', '.join(obj_names[:-1])} and {obj_names[-1]} (plus {len(nearby_objects) - 5} more)."
        
        landmarks_desc = ""
        if nearby_landmarks:
            landmarks_desc = f"Notable landmarks include "
            landmark_names = [f"{lm['name']} ({lm['distance']:.1f} units {lm['direction']})" for lm in nearby_landmarks[:3]]
            landmarks_desc += f"{', '.join(landmark_names[:-1])} and {landmark_names[-1]}." if len(landmark_names) > 1 else f"{landmark_names[0]}."
        
        full_description = f"You are in {location_desc}. {objects_desc} {landmarks_desc}"
        
        return SpatialDescription(
            description=full_description,
            objects=nearby_objects,
            regions=current_regions,
            landmarks=nearby_landmarks
        )
    
    @function_tool
    async def find_path_to_landmark(self, 
                                 landmark_type: str,
                                 current_position: Optional[PositionDict] = None,
                                 map_id: Optional[str] = None) -> NavigationResult:
        """
        Find a path to the nearest landmark of a given type
        
        Args:
            landmark_type: Type of landmark to navigate to
            current_position: Optional current position coordinates
            map_id: Optional map ID (uses active map if not provided)
            
        Returns:
            Navigation instructions and path information
        """
        # Use active map if not provided
        if not map_id:
            if not self.spatial_mapper.context.active_map_id:
                return NavigationResult(
                    success=False,
                    message="No active map set",
                    directions=[]
                )
            map_id = self.spatial_mapper.context.active_map_id
        
        # Get position to use
        use_position = None
        
        if current_position:
            use_position = current_position.model_dump()
        elif self.spatial_mapper.context.observer_position:
            pos = self.spatial_mapper.context.observer_position
            use_position = {"x": pos.x, "y": pos.y, "z": pos.z if pos.z is not None else None}
        
        if not use_position:
            return NavigationResult(
                success=False,
                message="No position provided or available",
                directions=[]
            )
        
        # Create a temporary position
        start_id = await self._create_temporary_position(map_id, use_position)
        
        # Find the nearest landmark of the given type
        map_obj = self.spatial_mapper.maps[map_id]
        nearest_landmark = None
        nearest_distance = float('inf')
        
        for obj_id in map_obj.landmarks:
            if obj_id in map_obj.spatial_objects:
                obj = map_obj.spatial_objects[obj_id]
                
                # Check if the landmark matches the type
                if obj.object_type.lower() == landmark_type.lower() or landmark_type.lower() in obj.properties.get("types", []):
                    # Calculate distance
                    distance = obj.coordinates.distance_to(
                        self.spatial_mapper.SpatialCoordinate(**use_position)
                    )
                    
                    if distance < nearest_distance:
                        nearest_landmark = obj
                        nearest_distance = distance
        
        if not nearest_landmark:
            # Clean up temporary position
            await self._cleanup_temporary_position(map_id, start_id)
            
            return NavigationResult(
                success=False,
                message=f"Could not find a landmark of type '{landmark_type}'",
                directions=[]
            )
        
        # Find path to the landmark
        path_result = await self.spatial_mapper.find_path(
            map_id=map_id,
            start_id=start_id,
            end_id=nearest_landmark.id,
            prefer_landmarks=True
        )
        
        # Clean up temporary position
        await self._cleanup_temporary_position(map_id, start_id)
        
        if "error" in path_result:
            return NavigationResult(
                success=False,
                message=f"Navigation error: {path_result['error']}",
                directions=[]
            )
        
        return NavigationResult(
            success=True,
            message=f"Found route to {nearest_landmark.name} ({landmark_type})",
            directions=path_result.get("directions", []),
            distance=path_result.get("distance", 0),
            estimated_time=path_result.get("estimated_time"),
            route_id=path_result.get("route_id")
        )
    
    @function_tool
    async def process_spatial_description(self, 
                                       description: str,
                                       map_id: Optional[str] = None) -> ProcessingResult:
        """
        Process a natural language description of a space
        
        Args:
            description: Text description of the environment
            map_id: Optional map ID (uses active map if not provided)
            
        Returns:
            Processing results
        """
        # Use active map if not provided
        if not map_id:
            if not self.spatial_mapper.context.active_map_id:
                # Create a new map for this description
                new_map = await self.spatial_mapper.create_cognitive_map(
                    name="Environment from description",
                    description="Map created from text description"
                )
                map_id = new_map["map_id"]
            else:
                map_id = self.spatial_mapper.context.active_map_id
        
        # First, extract spatial features
        features = await self.spatial_mapper.extract_spatial_features(description)
        
        if not features or "error" in features:
            return ProcessingResult(error="Failed to extract spatial features from description")
        
        # Process the features to update the map
        with trace(workflow_name="process_spatial_description"):
            # Create a detailed prompt for the map builder agent
            prompt = f"""
            Process this spatial description to update the cognitive map (ID: {map_id}):
            
            DESCRIPTION:
            {description}
            
            EXTRACTED FEATURES:
            {features}
            
            Process this description to:
            1. Identify objects, regions, and landmarks
            2. Update the cognitive map with this information
            3. Establish spatial relationships between elements
            """
            
            result = await Runner.run(
                self.mapper_agent,
                prompt
            )
            
            if hasattr(result, 'final_output'):
                return ProcessingResult(
                    message="Description processed successfully",
                    map_id=map_id,
                    features=features,
                    final_output=result.final_output
                )
            else:
                return ProcessingResult(
                    message="Description processed but no structured output available",
                    map_id=map_id,
                    features=features
                )
    
    async def process_request(self, request: str) -> str:
        """Process a user request through the triage agent"""
        with trace(workflow_name="spatial_cognition_request"):
            result = await Runner.run(
                self.triage_agent,
                request
            )
            
            return result.final_output
