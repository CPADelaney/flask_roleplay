# nyx/core/a2a/context_aware_spatial_mapper.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareSpatialMapper(ContextAwareModule):
    """
    Enhanced SpatialMapper with full context distribution capabilities
    """
    
    def __init__(self, original_spatial_mapper):
        super().__init__("spatial_mapper")
        self.original_mapper = original_spatial_mapper
        self.context_subscriptions = [
            "movement_detected", "location_update", "navigation_request",
            "memory_spatial_reference", "goal_spatial_requirement",
            "emotional_spatial_preference", "attention_spatial_focus"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize spatial processing for this context"""
        logger.debug(f"SpatialMapper received context for user: {context.user_id}")
        
        # Analyze input for spatial content
        spatial_analysis = await self._analyze_spatial_content(context.user_input)
        
        # Get current spatial state
        current_map = None
        if self.original_mapper.context.active_map_id:
            current_map = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
        
        # Send initial spatial context
        await self.send_context_update(
            update_type="spatial_context_available",
            data={
                "active_map_id": self.original_mapper.context.active_map_id,
                "observer_position": self.original_mapper.context.observer_position.dict() 
                    if self.original_mapper.context.observer_position else None,
                "spatial_analysis": spatial_analysis,
                "map_summary": self._get_map_summary(current_map) if current_map else None,
                "navigation_active": self.original_mapper.context.ongoing_navigation
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect spatial cognition"""
        
        if update.update_type == "movement_detected":
            # Update observer position based on movement
            movement_data = update.data
            new_position = movement_data.get("new_position")
            if new_position:
                await self._update_observer_position(new_position)
        
        elif update.update_type == "memory_spatial_reference":
            # Memory module found spatial references
            memory_data = update.data
            locations = memory_data.get("referenced_locations", [])
            for location in locations:
                await self._process_memory_location_reference(location)
        
        elif update.update_type == "goal_spatial_requirement":
            # Goal requires navigation to location
            goal_data = update.data
            destination = goal_data.get("destination")
            if destination:
                await self._prepare_navigation_for_goal(destination, goal_data)
        
        elif update.update_type == "emotional_spatial_preference":
            # Emotional state affects spatial preferences
            emotional_data = update.data
            await self._adjust_spatial_preferences(emotional_data)
        
        elif update.update_type == "attention_spatial_focus":
            # Attention controller directing spatial focus
            attention_data = update.data
            focus_location = attention_data.get("focus_location")
            if focus_location:
                await self._focus_on_location(focus_location)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full spatial context awareness"""
        # Extract spatial features from input
        spatial_features = await self.original_mapper.extract_spatial_features(context.user_input)
        
        # Get cross-module messages for spatial context
        messages = await self.get_cross_module_messages()
        
        # Process based on spatial content
        result = {
            "spatial_features_extracted": spatial_features,
            "spatial_processing_complete": False
        }
        
        # Check if we need to create or update maps
        if spatial_features and "objects" in spatial_features:
            for obj in spatial_features["objects"]:
                await self._process_spatial_object_with_context(obj, context, messages)
            result["spatial_processing_complete"] = True
        
        # Check for navigation requests
        if self._contains_navigation_request(context.user_input):
            nav_result = await self._process_navigation_request(context, messages)
            result["navigation_result"] = nav_result
        
        # Send processing results
        await self.send_context_update(
            update_type="spatial_processing_complete",
            data={
                "features_extracted": len(spatial_features.get("objects", [])),
                "navigation_requested": "navigation_result" in result,
                "maps_updated": result["spatial_processing_complete"]
            }
        )
        
        return result
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze spatial context and relationships"""
        # Get current spatial state
        analysis = {
            "spatial_coherence": await self._analyze_spatial_coherence(context),
            "navigation_possibilities": await self._analyze_navigation_options(context),
            "spatial_memory_integration": await self._analyze_spatial_memories(context),
            "landmark_salience": await self._analyze_landmark_salience(context)
        }
        
        # Check cross-module spatial references
        messages = await self.get_cross_module_messages()
        cross_module_spatial = self._extract_cross_module_spatial_references(messages)
        analysis["cross_module_spatial_references"] = cross_module_spatial
        
        return analysis
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize spatial components for response generation"""
        messages = await self.get_cross_module_messages()
        
        synthesis = {
            "spatial_context_summary": await self._generate_spatial_summary(context),
            "navigation_suggestions": await self._generate_navigation_suggestions(context, messages),
            "spatial_memory_cues": await self._generate_spatial_memory_cues(context),
            "landmark_references": await self._suggest_landmark_references(context)
        }
        
        # If navigation is active, provide detailed guidance
        if self.original_mapper.context.ongoing_navigation:
            synthesis["navigation_guidance"] = await self._generate_navigation_guidance(context)
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_spatial_content(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for spatial content"""
        input_lower = user_input.lower()
        
        analysis = {
            "contains_location_reference": any(word in input_lower for word in 
                ["where", "location", "place", "room", "area", "here", "there"]),
            "contains_movement_language": any(word in input_lower for word in 
                ["go", "move", "walk", "navigate", "travel", "head", "proceed"]),
            "contains_spatial_query": any(word in input_lower for word in 
                ["how far", "distance", "near", "close", "adjacent", "next to"]),
            "contains_direction_reference": any(word in input_lower for word in 
                ["north", "south", "east", "west", "left", "right", "up", "down"]),
            "spatial_confidence": 0.0
        }
        
        # Calculate confidence
        spatial_indicators = sum(1 for v in analysis.values() if v and isinstance(v, bool))
        analysis["spatial_confidence"] = min(1.0, spatial_indicators * 0.25)
        
        return analysis
    
    def _get_map_summary(self, cognitive_map) -> Dict[str, Any]:
        """Get a summary of a cognitive map"""
        if not cognitive_map:
            return {}
        
        return {
            "map_id": cognitive_map.id,
            "map_name": cognitive_map.name,
            "object_count": len(cognitive_map.spatial_objects),
            "region_count": len(cognitive_map.regions),
            "route_count": len(cognitive_map.routes),
            "landmark_count": len(cognitive_map.landmarks),
            "completeness": cognitive_map.completeness,
            "last_updated": cognitive_map.last_updated
        }
    
    async def _update_observer_position(self, new_position: Dict[str, Any]):
        """Update the observer's position"""
        if hasattr(self.original_mapper, 'SpatialCoordinate'):
            coord = self.original_mapper.SpatialCoordinate(**new_position)
            self.original_mapper.context.observer_position = coord
            self.original_mapper.context.last_observation_time = datetime.now().isoformat()
    
    async def _process_memory_location_reference(self, location_ref: Dict[str, Any]):
        """Process a location reference from memory"""
        location_id = location_ref.get("location_id")
        memory_id = location_ref.get("memory_id")
        
        if location_id and self.original_mapper.context.active_map_id:
            map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
            if map_obj and location_id in map_obj.spatial_objects:
                # Update object's memory associations
                obj = map_obj.spatial_objects[location_id]
                if "memory_associations" not in obj.properties:
                    obj.properties["memory_associations"] = []
                if memory_id not in obj.properties["memory_associations"]:
                    obj.properties["memory_associations"].append(memory_id)
    
    async def _prepare_navigation_for_goal(self, destination: str, goal_data: Dict[str, Any]):
        """Prepare navigation for a goal requirement"""
        self.original_mapper.context.ongoing_navigation = True
        self.original_mapper.context.navigation_target = destination
        
        # Send navigation preparation update
        await self.send_context_update(
            update_type="navigation_prepared",
            data={
                "destination": destination,
                "goal_id": goal_data.get("goal_id"),
                "urgency": goal_data.get("urgency", 0.5)
            },
            priority=ContextPriority.HIGH
        )
    
    async def _adjust_spatial_preferences(self, emotional_data: Dict[str, Any]):
        """Adjust spatial preferences based on emotional state"""
        # This could affect route planning preferences
        emotional_state = emotional_data.get("emotional_state", {})
        
        # Example: anxiety might prefer familiar routes
        if emotional_state.get("Anxiety", 0) > 0.6:
            # Prefer routes with more landmarks
            if hasattr(self.original_mapper, 'route_preferences'):
                self.original_mapper.route_preferences["prefer_landmarks"] = True
    
    async def _focus_on_location(self, focus_location: str):
        """Focus spatial attention on a specific location"""
        # This could highlight certain areas or objects in the map
        if self.original_mapper.context.active_map_id:
            map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
            if map_obj:
                # Mark location as focus point
                for obj_id, obj in map_obj.spatial_objects.items():
                    if obj.name.lower() == focus_location.lower():
                        obj.properties["attention_focus"] = True
                        obj.properties["focus_timestamp"] = datetime.now().isoformat()
    
    def _contains_navigation_request(self, user_input: str) -> bool:
        """Check if input contains a navigation request"""
        nav_keywords = ["navigate", "go to", "find", "where is", "how do i get", 
                       "take me", "show me the way", "directions to"]
        input_lower = user_input.lower()
        return any(keyword in input_lower for keyword in nav_keywords)
    
    async def _process_spatial_object_with_context(self, obj_name: str, context: SharedContext, messages: Dict):
        """Process a spatial object with full context"""
        # Enhanced object processing considering context
        if self.original_mapper.context.active_map_id:
            # Check if object relates to any goals
            goal_related = False
            for module_name, module_messages in messages.items():
                if module_name == "goal_manager":
                    for msg in module_messages:
                        if msg.get("type") == "goal_context_available":
                            active_goals = msg.get("data", {}).get("active_goals", [])
                            for goal in active_goals:
                                if obj_name in goal.get("description", "").lower():
                                    goal_related = True
                                    break
            
            # Add object with enhanced properties
            result = await self.original_mapper.add_spatial_object(
                map_id=self.original_mapper.context.active_map_id,
                name=obj_name,
                object_type="object",  # Could be enhanced with classification
                coordinates={"x": 0, "y": 0},  # Would need actual coordinates
                properties={
                    "discovered_from_input": True,
                    "goal_related": goal_related,
                    "context_confidence": context.session_context.get("confidence", 0.5)
                }
            )
    
    async def _process_navigation_request(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Process a navigation request with context"""
        # Extract destination from input
        destination = self._extract_destination(context.user_input)
        
        if not destination:
            return {"error": "Could not determine destination"}
        
        # Check emotional context for navigation preferences
        prefer_safe = False
        if context.emotional_state:
            anxiety_level = context.emotional_state.get("Anxiety", 0)
            if anxiety_level > 0.5:
                prefer_safe = True
        
        # Prepare navigation
        nav_result = {
            "destination": destination,
            "preferences": {
                "prefer_landmarks": True,
                "prefer_safe": prefer_safe
            }
        }
        
        return nav_result
    
    def _extract_destination(self, user_input: str) -> Optional[str]:
        """Extract destination from user input"""
        # Simple extraction - would be enhanced with NLP
        prep_phrases = ["to the", "to", "find", "where is", "navigate to"]
        input_lower = user_input.lower()
        
        for phrase in prep_phrases:
            if phrase in input_lower:
                idx = input_lower.find(phrase) + len(phrase)
                # Extract next word(s) as destination
                remaining = user_input[idx:].strip()
                words = remaining.split()
                if words:
                    return words[0]  # Simple version - take first word
        
        return None
    
    async def _analyze_spatial_coherence(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze spatial coherence of current context"""
        coherence_score = 1.0
        issues = []
        
        if self.original_mapper.context.active_map_id:
            map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
            if map_obj:
                # Check map completeness
                if map_obj.completeness < 0.3:
                    coherence_score -= 0.3
                    issues.append("incomplete_map")
                
                # Check for disconnected regions
                if len(map_obj.regions) > 1:
                    connected_regions = sum(1 for r in map_obj.regions.values() 
                                          if len(r.adjacent_regions) > 0)
                    if connected_regions < len(map_obj.regions):
                        coherence_score -= 0.2
                        issues.append("disconnected_regions")
        
        return {
            "coherence_score": max(0, coherence_score),
            "issues": issues,
            "recommendations": self._get_coherence_recommendations(issues)
        }
    
    def _get_coherence_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations for improving spatial coherence"""
        recommendations = []
        
        if "incomplete_map" in issues:
            recommendations.append("Explore more areas to complete the map")
        if "disconnected_regions" in issues:
            recommendations.append("Find connections between isolated regions")
        
        return recommendations
    
    async def _analyze_navigation_options(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Analyze available navigation options"""
        options = []
        
        if self.original_mapper.context.active_map_id:
            map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
            if map_obj and self.original_mapper.context.observer_position:
                # Find nearby navigable locations
                current_pos = self.original_mapper.context.observer_position
                
                for obj_id, obj in map_obj.spatial_objects.items():
                    if obj.is_landmark:
                        distance = current_pos.distance_to(obj.coordinates) if hasattr(current_pos, 'distance_to') else 0
                        options.append({
                            "destination": obj.name,
                            "destination_id": obj_id,
                            "distance": distance,
                            "is_landmark": True
                        })
        
        # Sort by distance
        options.sort(key=lambda x: x["distance"])
        return options[:5]  # Top 5 options
    
    async def _analyze_spatial_memories(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze spatial memory integration"""
        if hasattr(self.original_mapper, 'memory_integration') and self.original_mapper.memory_integration:
            # Get memories at current location
            if self.original_mapper.context.observer_position:
                memories = await self.original_mapper.memory_integration.retrieve_memories_near_coordinates(
                    coordinates=self.original_mapper.context.observer_position.dict(),
                    radius=5.0
                )
                
                return {
                    "nearby_memories": len(memories),
                    "memory_density": len(memories) / 5.0,  # Memories per unit area
                    "has_episodic_significance": len(memories) > 0
                }
        
        return {"nearby_memories": 0, "memory_density": 0, "has_episodic_significance": False}
    
    async def _analyze_landmark_salience(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Analyze salience of landmarks"""
        salient_landmarks = []
        
        if self.original_mapper.context.active_map_id:
            map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
            if map_obj:
                for landmark_id in map_obj.landmarks:
                    if landmark_id in map_obj.spatial_objects:
                        landmark = map_obj.spatial_objects[landmark_id]
                        
                        # Calculate salience based on various factors
                        salience = 0.5  # Base salience
                        
                        # Observation count increases salience
                        salience += min(0.3, landmark.observation_count * 0.05)
                        
                        # Memory associations increase salience
                        memory_count = len(landmark.properties.get("memory_associations", []))
                        salience += min(0.2, memory_count * 0.1)
                        
                        salient_landmarks.append({
                            "landmark_id": landmark_id,
                            "name": landmark.name,
                            "salience": salience,
                            "observation_count": landmark.observation_count
                        })
        
        # Sort by salience
        salient_landmarks.sort(key=lambda x: x["salience"], reverse=True)
        return salient_landmarks
    
    def _extract_cross_module_spatial_references(self, messages: Dict) -> List[Dict[str, Any]]:
        """Extract spatial references from other modules"""
        spatial_refs = []
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                # Look for spatial content in messages
                if "location" in msg.get("data", {}):
                    spatial_refs.append({
                        "source_module": module_name,
                        "location": msg["data"]["location"],
                        "context": msg.get("type", "unknown")
                    })
        
        return spatial_refs
    
    async def _generate_spatial_summary(self, context: SharedContext) -> str:
        """Generate a summary of current spatial context"""
        if not self.original_mapper.context.active_map_id:
            return "No spatial map currently active"
        
        map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
        if not map_obj:
            return "Map data unavailable"
        
        summary_parts = []
        summary_parts.append(f"Currently in map: {map_obj.name}")
        
        if self.original_mapper.context.observer_position:
            summary_parts.append("Position tracked")
        
        if map_obj.landmarks:
            summary_parts.append(f"{len(map_obj.landmarks)} landmarks identified")
        
        if self.original_mapper.context.ongoing_navigation:
            summary_parts.append(f"Navigating to: {self.original_mapper.context.navigation_target}")
        
        return ". ".join(summary_parts)
    
    async def _generate_navigation_suggestions(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate navigation suggestions based on context"""
        suggestions = []
        
        # Check for goal-related navigation needs
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg.get("type") == "goal_spatial_requirement":
                        destination = msg.get("data", {}).get("destination")
                        if destination:
                            suggestions.append(f"Navigate to {destination} for goal completion")
        
        # Add exploration suggestions if map is incomplete
        if self.original_mapper.context.active_map_id:
            map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
            if map_obj and map_obj.completeness < 0.5:
                suggestions.append("Explore unexplored areas to complete your map")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    async def _generate_spatial_memory_cues(self, context: SharedContext) -> List[str]:
        """Generate spatial memory cues for response"""
        cues = []
        
        if hasattr(self.original_mapper, 'memory_integration') and self.original_mapper.memory_integration:
            if self.original_mapper.context.observer_position:
                memories = await self.original_mapper.memory_integration.retrieve_memories_near_coordinates(
                    coordinates=self.original_mapper.context.observer_position.dict(),
                    radius=10.0,
                    limit=3
                )
                
                for memory in memories:
                    cues.append(f"Near here: {memory.get('description', 'A memory')}")
        
        return cues
    
    async def _suggest_landmark_references(self, context: SharedContext) -> List[str]:
        """Suggest landmarks to reference in response"""
        references = []
        
        if self.original_mapper.context.active_map_id:
            map_obj = self.original_mapper.maps.get(self.original_mapper.context.active_map_id)
            if map_obj and self.original_mapper.context.observer_position:
                # Get nearest landmarks
                result = await self.original_mapper.get_nearest_landmarks(
                    map_id=self.original_mapper.context.active_map_id,
                    location_id="current_position",  # Would need proper implementation
                    max_count=3
                )
                
                for landmark in result:
                    if not isinstance(landmark, dict) or "error" not in landmark:
                        references.append(f"{landmark.get('name', 'landmark')} is {landmark.get('direction', 'nearby')}")
        
        return references
    
    async def _generate_navigation_guidance(self, context: SharedContext) -> Dict[str, Any]:
        """Generate detailed navigation guidance"""
        if not self.original_mapper.context.navigation_target:
            return {"status": "no_active_navigation"}
        
        guidance = {
            "destination": self.original_mapper.context.navigation_target,
            "current_position": self.original_mapper.context.observer_position.dict() 
                if self.original_mapper.context.observer_position else None,
            "next_steps": ["Continue toward destination"],
            "landmarks_ahead": []
        }
        
        # Would include more detailed pathfinding here
        return guidance
    
    # Delegate all other methods to the original mapper
    def __getattr__(self, name):
        """Delegate any missing methods to the original mapper"""
        return getattr(self.original_mapper, name)
