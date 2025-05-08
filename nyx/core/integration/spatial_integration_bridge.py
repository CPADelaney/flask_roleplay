# nyx/core/integration/

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import Agent, Runner, function_tool, handoff, trace, ModelSettings, RunContextWrapper, trace_method
from agents.tracing import custom_span

# Import spatial components
from nyx.core.spatial.spatial_mapper import SpatialMapper, SpatialMapperContext
from nyx.core.spatial.navigator_agent import SpatialNavigatorAgent
from nyx.core.spatial.spatial_memory import SpatialMemoryIntegration
from nyx.core.spatial.map_visualization import MapVisualization
from nyx.core.spatial.spatial_schemas import (
    SpatialObservation, NavigationRequest, SpatialQuery, 
    NavigationResult, SpatialDescription
)

logger = logging.getLogger(__name__)

class SpatialIntegrationBridge:
    """
    Integration bridge for spatial cognition and reasoning.
    
    Connects spatial mapping, navigation, memory, and visualization
    with other cognitive systems including memory, attention, and decision-making.
    """
    
    def __init__(self, 
                nyx_brain=None,
                spatial_mapper=None,
                memory_orchestrator=None,
                attention_system=None):
        """Initialize the spatial integration bridge."""
        self.brain = nyx_brain
        
        # Core spatial components
        self.spatial_mapper = spatial_mapper or SpatialMapper()
        self.navigator_agent = None
        self.spatial_memory = None
        self.map_visualization = MapVisualization()
        
        # Connected systems
        self.memory_orchestrator = memory_orchestrator
        self.attention_system = attention_system
        
        # Integration state tracking
        self._subscribed = False
        self.active_maps = {}
        self.recent_observations = []
        
        # Create agents
        self.spatial_triage_agent = None
        
        logger.info("SpatialIntegrationBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Initialize navigator agent
            self.navigator_agent = SpatialNavigatorAgent(self.spatial_mapper)
            
            # Initialize spatial memory integration
            self.spatial_memory = SpatialMemoryIntegration(
                self.spatial_mapper, 
                self.memory_orchestrator
            )
            
            # Initialize agents
            self.spatial_triage_agent = self._create_spatial_triage_agent()
            
            # Subscribe to relevant events
            if not self._subscribed:
                self._subscribed = True
            
            logger.info("SpatialIntegrationBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing SpatialIntegrationBridge: {e}")
            return False
    
    def _create_spatial_triage_agent(self) -> Agent:
        """Create the main triage agent for spatial queries"""
        return Agent(
            name="Spatial Cognition Agent",
            instructions="""
            You are the Spatial Cognition Agent that helps understand and navigate through environments.
            
            You help create and maintain cognitive maps, which are mental representations of spaces.
            You provide navigation assistance, spatial reasoning, and help understand environments.
            
            For direct navigation, wayfinding, or route planning tasks, hand off to the Spatial Navigator agent.
            For map building, environment understanding, or spatial memory tasks, hand off to the Map Builder agent.
            
            Respond helpfully to spatial queries and try to understand the spatial needs behind user requests.
            """,
            handoffs=[
                handoff(self.navigator_agent.navigator_agent, 
                        tool_description="For navigation, route finding, and direction giving tasks"),
                handoff(self.navigator_agent.mapper_agent, 
                        tool_description="For map building, observation processing, and spatial environment understanding")
            ],
            tools=[
                function_tool(self.process_spatial_observation),
                function_tool(self.describe_current_surroundings),
                function_tool(self.generate_map_visualization),
                function_tool(self.retrieve_spatial_memories)
            ]
        )
    
    @function_tool
    async def process_spatial_observation(self, 
                                      observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a spatial observation and update cognitive maps.
        
        Args:
            observation: Spatial observation data
            
        Returns:
            Processing results
        """
        # Convert to proper spatial observation format
        if not isinstance(observation, SpatialObservation):
            # Convert dict to SpatialObservation
            if "observation_type" not in observation:
                observation["observation_type"] = "object"  # Default
            
            spatial_obs = SpatialObservation(**observation)
        else:
            spatial_obs = observation
        
        # Process through spatial mapper
        result = await self.spatial_mapper.process_spatial_observation(spatial_obs)
        
        # Keep track of recent observations
        self.recent_observations.append({
            "timestamp": spatial_obs.timestamp,
            "type": spatial_obs.observation_type,
            "result": result
        })
        if len(self.recent_observations) > 20:
            self.recent_observations = self.recent_observations[-20:]
        
        # If memory integration is available, store observation in spatial memory
        if self.spatial_memory:
            # Format observation for memory
            memory_text = f"Observed {spatial_obs.observation_type}: {spatial_obs.content}"
            
            # Get location information if available
            location_id = None
            coordinates = None
            
            if "object_id" in result:
                location_id = result["object_id"]
            elif "region_id" in result:
                location_id = result["region_id"]
            
            if spatial_obs.observer_position:
                coordinates = spatial_obs.observer_position.dict()
            
            # Store in spatial memory
            await self.spatial_memory.store_spatial_memory(
                memory_text=memory_text,
                location_id=location_id,
                coordinates=coordinates,
                memory_type="observation",
                significance=int(spatial_obs.confidence * 10)  # Scale 0-1 to 0-10
            )
        
        return result

    @function_tool
    async def navigate_to_location(self, 
                               location: str,
                               current_position: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate navigation instructions to a named location.
        
        Args:
            location: Name of the target location
            current_position: Optional current coordinates
            
        Returns:
            Navigation instructions
        """
        # Delegate to navigator agent
        result = await self.navigator_agent.navigate_to_location(
            location_name=location,
            current_position=current_position
        )
        
        # If successful and attention system is available, focus attention on navigation
        if result.success and self.attention_system and hasattr(self.attention_system, 'focus_attention'):
            await self.attention_system.focus_attention(
                target=f"navigating to {location}",
                target_type="spatial_navigation",
                attention_level=0.8,  # High attention for navigation
                source="spatial_integration_bridge"
            )
        
        return {
            "success": result.success,
            "directions": result.directions,
            "distance": result.distance,
            "estimated_time": result.estimated_time,
            "location": location
        }

    @function_tool
    async def describe_current_surroundings(self) -> Dict[str, Any]:
        """
        Describe the current surroundings based on spatial knowledge.
        
        Returns:
            Description of surroundings
        """
        # Get description of surroundings
        description = await self.navigator_agent.describe_surroundings()
        
        # If attention system is available, note what's around
        if self.attention_system and hasattr(self.attention_system, 'focus_attention'):
            # Focus attention on most notable landmark if any
            if description.landmarks and len(description.landmarks) > 0:
                landmark = description.landmarks[0]
                await self.attention_system.focus_attention(
                    target=f"landmark: {landmark['name']}",
                    target_type="spatial_landmark",
                    attention_level=0.7,
                    source="spatial_integration_bridge"
                )
        
        # Format response
        return {
            "description": description.description,
            "objects_count": len(description.objects),
            "landmarks_count": len(description.landmarks),
            "regions": [r["name"] for r in description.regions]
        }

    @function_tool
    async def retrieve_spatial_memories(self, 
                                    location_id: Optional[str] = None,
                                    coordinates: Optional[Dict[str, float]] = None,
                                    radius: float = 10.0,
                                    limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories associated with a location or area.
        
        Args:
            location_id: Optional ID of a specific location
            coordinates: Optional coordinates to search around
            radius: Search radius around coordinates
            limit: Maximum number of memories to return
            
        Returns:
            List of spatial memories
        """
        if not self.spatial_memory:
            return []
        
        if location_id:
            # Retrieve memories for specific location
            return await self.spatial_memory.retrieve_memories_at_location(
                location_id=location_id,
                limit=limit
            )
        elif coordinates:
            # Retrieve memories near coordinates
            return await self.spatial_memory.retrieve_memories_near_coordinates(
                coordinates=coordinates,
                radius=radius,
                limit=limit
            )
        
        return []

    @function_tool
    async def generate_map_visualization(self, 
                                     map_id: Optional[str] = None,
                                     highlight_landmarks: List[str] = None,
                                     highlight_route: Optional[str] = None,
                                     format: str = "svg") -> Dict[str, Any]:
        """
        Generate a visualization of a cognitive map.
        
        Args:
            map_id: Optional ID of the map (uses active map if not provided)
            highlight_landmarks: Optional list of landmark IDs to highlight
            highlight_route: Optional route ID to highlight
            format: Visualization format (svg, ascii, data)
            
        Returns:
            Map visualization
        """
        # Use active map if not provided
        if not map_id:
            if not self.spatial_mapper.context.active_map_id:
                return {"error": "No active map set"}
            map_id = self.spatial_mapper.context.active_map_id
        
        # Check if map exists
        if map_id not in self.spatial_mapper.maps:
            return {"error": f"Map {map_id} not found"}
        
        map_obj = self.spatial_mapper.maps[map_id]
        
        # Generate visualization
        if format.lower() == "svg":
            # Generate SVG visualization
            svg = self.map_visualization.generate_svg(
                cognitive_map=map_obj,
                highlight_landmark_ids=highlight_landmarks,
                highlight_route_id=highlight_route
            )
            return {
                "map_id": map_id,
                "format": "svg",
                "visualization": svg
            }
        
        elif format.lower() == "ascii":
            # Generate ASCII visualization
            ascii_map = self.map_visualization.generate_ascii_map(
                cognitive_map=map_obj
            )
            return {
                "map_id": map_id,
                "format": "ascii",
                "visualization": ascii_map
            }
        
        elif format.lower() == "data":
            # Generate data representation
            map_data = self.map_visualization.generate_map_data(
                cognitive_map=map_obj
            )
            return {
                "map_id": map_id,
                "format": "data",
                "visualization": map_data
            }
        
        else:
            return {"error": f"Unsupported format: {format}"}

    async def _handle_sensory_input(self, event) -> None:
        """
        Handle sensory input events that may contain spatial information.
        """
        # Extract relevant data
        modality = event.data.get("modality")
        content = event.data.get("content")
        
        # Process visual input (images) as they may contain spatial information
        if modality == "visual" or modality == "image":
            # Extract spatial information from visual input
            if self.spatial_mapper and hasattr(self.spatial_mapper, 'extract_spatial_features'):
                features = await self.spatial_mapper.extract_spatial_features(
                    str(content)  # Convert content to string for processing
                )
                
                if features and "objects" in features:
                    # Process detected objects as spatial observations
                    for obj in features.get("objects", []):
                        # Create observation for each object
                        observation = {
                            "observation_type": "object",
                            "content": {
                                "name": obj,
                                "object_type": obj,
                                # Estimate position if available
                                "coordinates": features.get("coordinates", {}).get(obj, {"x": 0, "y": 0})
                            }
                        }
                        
                        # Process the observation
                        asyncio.create_task(self.process_spatial_observation(observation))

    async def _handle_user_interaction(self, event) -> None:
        """
        Handle user interactions that may reference locations or navigation.
        """
        # Extract data
        content = event.data.get("content", "")
        
        # Simple detection of navigation-related queries
        navigation_keywords = ["where", "go to", "navigate", "find", "location", "map"]
        is_navigation_query = any(keyword in content.lower() for keyword in navigation_keywords)
        
        if is_navigation_query and self.navigator_agent:
            # This seems to be a navigation query, but we'll need NLP to extract the target
            # For now, this is a placeholder for more sophisticated processing
            if "go to " in content.lower():
                target = content.lower().split("go to ")[1].strip()
                asyncio.create_task(self.navigate_to_location(target))

    async def _handle_attention_change(self, event) -> None:
        """
        Handle attention focus changes that may affect spatial awareness.
        """
        # Extract focus data
        focus = event.data.get("focus_target", "")
        focus_type = event.data.get("focus_type", "")
        
        # If attention is on a spatial target, update spatial awareness
        if "object" in focus_type or "location" in focus_type:
            # Try to find this object in current spatial maps
            if self.spatial_mapper and self.spatial_mapper.context.active_map_id:
                map_id = self.spatial_mapper.context.active_map_id
                map_obj = self.spatial_mapper.maps.get(map_id)
                
                if map_obj:
                    # Search for object by name
                    for obj_id, obj in map_obj.spatial_objects.items():
                        if obj.name.lower() in focus.lower():
                            # Found the object, describe surroundings around it
                            if self.navigator_agent:
                                asyncio.create_task(
                                    self.navigator_agent.describe_surroundings(
                                        position=obj.coordinates.dict()
                                    )
                                )
                                break

    async def _handle_memory_retrieved(self, event) -> None:
        """
        Handle memory retrieval events that might have spatial context.
        """
        # Extract memory data
        memory = event.data.get("memory")
        
        if not memory:
            return
            
        # Check if this memory has spatial metadata
        metadata = memory.get("metadata", {})
        
        if "spatial" in metadata and metadata["spatial"]:
            # This is a spatial memory, check for location information
            if "location_id" in metadata and self.spatial_mapper:
                location_id = metadata["location_id"]
                
                # Get current map id
                map_id = self.spatial_mapper.context.active_map_id
                
                # Update visualization to highlight this location
                # This would be implemented with a visualization component
                
                # Also retrieve other memories at this location
                if self.spatial_memory:
                    other_memories = await self.spatial_memory.retrieve_memories_at_location(
                        location_id=location_id,
                        limit=3  # Just a few related memories
                    )
                    
                    # These could be used to enhance the context

    async def _handle_location_changed(self, event) -> None:
        """
        Handle location change events.
        """
        # Extract location data
        new_location = event.data.get("location")
        coordinates = event.data.get("coordinates")
        
        if not new_location and not coordinates:
            return
            
        # Update observer position in spatial mapper
        if self.spatial_mapper and coordinates:
            from nyx.core.spatial.spatial_schemas import SpatialCoordinate
            self.spatial_mapper.context.observer_position = SpatialCoordinate(**coordinates)
            
            # Describe new surroundings
            asyncio.create_task(self.describe_current_surroundings())
            
            # Retrieve memories at this location
            if self.spatial_memory:
                asyncio.create_task(
                    self.retrieve_spatial_memories(coordinates=coordinates)
                )

    async def process_spatial_request(self, request: str) -> str:
        """
        Process a user request through the spatial triage agent
        
        Args:
            request: The user's spatial request or query
            
        Returns:
            Response from the spatial cognition system
        """
        with trace(workflow_name="spatial_request"):
            result = await Runner.run(
                self.spatial_triage_agent,
                request
            )
            
            return result.final_output

    @trace_method(level=TraceLevel.INFO, group_id="SpatialIntegration")
    async def integrate_with_planning(self, 
                                   goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate spatial awareness with planning and goal achievement.
        
        Args:
            goal: Goal to achieve that may involve spatial navigation
            
        Returns:
            Spatial context for goal planning
        """
        spatial_context = {
            "has_spatial_component": False
        }
        
        # Check if goal has spatial components
        spatial_keywords = ["go to", "find", "locate", "navigate", "move to", "place"]
        goal_desc = goal.get("description", "").lower()
        
        has_spatial_component = any(keyword in goal_desc for keyword in spatial_keywords)
        
        if has_spatial_component:
            spatial_context["has_spatial_component"] = True
            
            # Extract target location (simplified, you'd use NLP for this)
            target_location = None
            for keyword in spatial_keywords:
                if keyword in goal_desc:
                    parts = goal_desc.split(keyword)
                    if len(parts) > 1:
                        target_location = parts[1].strip().split()[0]
                        break
            
            if target_location:
                # Get navigation plan
                nav_plan = await self.navigate_to_location(target_location)
                spatial_context["navigation_plan"] = nav_plan
                
                # Add path steps to goal if needed
                if nav_plan.get("success", False) and "directions" in nav_plan:
                    spatial_context["path_steps"] = nav_plan["directions"]
        
        return spatial_context

# Function to create the bridge
def create_spatial_integration_bridge(nyx_brain=None):
    """Create a spatial integration bridge for the given brain."""
    return SpatialIntegrationBridge(
        nyx_brain=nyx_brain,
        spatial_mapper=None,  # Will be initialized in the bridge
        memory_orchestrator=nyx_brain.memory_orchestrator if hasattr(nyx_brain, "memory_orchestrator") else None,
        attention_system=nyx_brain.attention_system if hasattr(nyx_brain, "attention_system") else None
    )
