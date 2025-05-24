# nyx/core/a2a/context_aware_navigator_agent.py

import logging
from typing import Dict, List, Any, Optional

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareSpatialNavigatorAgent(ContextAwareModule):
    """
    Enhanced SpatialNavigatorAgent with context distribution capabilities
    """
    
    def __init__(self, original_navigator):
        super().__init__("spatial_navigator")
        self.original_navigator = original_navigator
        self.context_subscriptions = [
            "navigation_request", "obstacle_detected", "route_blocked",
            "goal_destination_set", "emotional_navigation_preference",
            "memory_navigation_hint"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize navigation processing for this context"""
        logger.debug(f"SpatialNavigator received context for navigation")
        
        # Check if input contains navigation request
        nav_intent = self._detect_navigation_intent(context.user_input)
        
        # Send initial navigation context
        await self.send_context_update(
            update_type="navigation_context_initialized",
            data={
                "navigation_intent_detected": nav_intent is not None,
                "intent_type": nav_intent,
                "navigator_ready": True,
                "spatial_mapper_available": self.original_navigator.spatial_mapper is not None
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle navigation-relevant updates from other modules"""
        
        if update.update_type == "goal_destination_set":
            # Goal manager set a destination
            goal_data = update.data
            destination = goal_data.get("destination")
            if destination:
                await self._prepare_goal_navigation(destination, goal_data)
        
        elif update.update_type == "obstacle_detected":
            # Obstacle detected on current route
            obstacle_data = update.data
            await self._handle_obstacle(obstacle_data)
        
        elif update.update_type == "emotional_navigation_preference":
            # Emotional state affects navigation preferences
            emotional_data = update.data
            await self._adjust_navigation_style(emotional_data)
        
        elif update.update_type == "memory_navigation_hint":
            # Memory provides navigation hints
            memory_data = update.data
            await self._incorporate_memory_hints(memory_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process navigation requests with context awareness"""
        # Detect navigation intent
        nav_intent = self._detect_navigation_intent(context.user_input)
        
        if not nav_intent:
            return {"navigation_processing": False}
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Extract destination and preferences
        nav_params = await self._extract_navigation_parameters(context, messages)
        
        # Process navigation based on intent
        result = {}
        
        if nav_intent == "navigate_to":
            result = await self._process_navigation_request(nav_params, context, messages)
        elif nav_intent == "find_path":
            result = await self._process_pathfinding_request(nav_params, context, messages)
        elif nav_intent == "describe_location":
            result = await self._process_location_description(nav_params, context, messages)
        
        # Send navigation result update
        if result.get("success"):
            await self.send_context_update(
                update_type="navigation_processed",
                data={
                    "destination": nav_params.get("destination"),
                    "route_found": result.get("route_id") is not None,
                    "distance": result.get("distance")
                }
            )
        
        return result
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze navigation context and options"""
        messages = await self.get_cross_module_messages()
        
        analysis = {
            "navigation_feasibility": await self._analyze_navigation_feasibility(context, messages),
            "route_options": await self._analyze_route_options(context),
            "contextual_preferences": await self._analyze_contextual_preferences(messages),
            "navigation_risks": await self._analyze_navigation_risks(context)
        }
        
        return analysis
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize navigation guidance for response"""
        messages = await self.get_cross_module_messages()
        
        synthesis = {
            "navigation_recommendations": await self._synthesize_recommendations(context, messages),
            "contextual_directions": await self._synthesize_contextual_directions(context, messages),
            "landmark_guidance": await self._synthesize_landmark_guidance(context),
            "safety_considerations": await self._synthesize_safety_considerations(context, messages)
        }
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    def _detect_navigation_intent(self, user_input: str) -> Optional[str]:
        """Detect navigation intent from user input"""
        input_lower = user_input.lower()
        
        navigate_patterns = ["go to", "navigate to", "take me to", "find my way to", "head to"]
        path_patterns = ["how do i get to", "path to", "route to", "directions to"]
        location_patterns = ["where am i", "describe this place", "what's around me"]
        
        for pattern in navigate_patterns:
            if pattern in input_lower:
                return "navigate_to"
        
        for pattern in path_patterns:
            if pattern in input_lower:
                return "find_path"
        
        for pattern in location_patterns:
            if pattern in input_lower:
                return "describe_location"
        
        return None
    
    async def _prepare_goal_navigation(self, destination: str, goal_data: Dict[str, Any]):
        """Prepare navigation for a goal-driven destination"""
        # Set navigation context
        nav_context = {
            "destination": destination,
            "purpose": "goal_completion",
            "urgency": goal_data.get("urgency", 0.5),
            "goal_id": goal_data.get("goal_id")
        }
        
        # Store navigation context
        if hasattr(self.original_navigator, "_navigation_context"):
            self.original_navigator._navigation_context = nav_context
    
    async def _handle_obstacle(self, obstacle_data: Dict[str, Any]):
        """Handle obstacle detection during navigation"""
        # Would trigger replanning
        logger.info(f"Obstacle detected: {obstacle_data.get('description', 'Unknown obstacle')}")
        
        # Send update about replanning
        await self.send_context_update(
            update_type="navigation_replanning",
            data={
                "reason": "obstacle_detected",
                "obstacle": obstacle_data
            },
            priority=ContextPriority.HIGH
        )
    
    async def _adjust_navigation_style(self, emotional_data: Dict[str, Any]):
        """Adjust navigation style based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        
        # Anxiety prefers familiar routes
        if emotional_state.get("Anxiety", 0) > 0.6:
            if hasattr(self.original_navigator, "_navigation_preferences"):
                self.original_navigator._navigation_preferences["prefer_familiar"] = True
                self.original_navigator._navigation_preferences["avoid_crowds"] = True
        
        # Excitement might prefer scenic routes
        if emotional_state.get("Excitement", 0) > 0.7:
            if hasattr(self.original_navigator, "_navigation_preferences"):
                self.original_navigator._navigation_preferences["prefer_scenic"] = True
    
    async def _incorporate_memory_hints(self, memory_data: Dict[str, Any]):
        """Incorporate navigation hints from memory"""
        hints = memory_data.get("navigation_hints", [])
        
        for hint in hints:
            # Would use hints to influence route selection
            logger.debug(f"Navigation hint from memory: {hint}")
    
    async def _extract_navigation_parameters(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Extract navigation parameters from context"""
        params = {
            "destination": None,
            "preferences": {},
            "constraints": []
        }
        
        # Extract destination from input
        input_lower = context.user_input.lower()
        
        # Simple destination extraction
        nav_phrases = ["go to", "navigate to", "take me to", "find", "get to"]
        for phrase in nav_phrases:
            if phrase in input_lower:
                idx = input_lower.find(phrase) + len(phrase)
                remaining = context.user_input[idx:].strip()
                if remaining:
                    # Take the rest as destination
                    params["destination"] = remaining.split('.')[0].strip()
                    break
        
        # Check for emotional preferences
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg.get("type") == "emotional_state_update":
                        emotional_state = msg.get("data", {}).get("emotional_state", {})
                        if emotional_state.get("Anxiety", 0) > 0.5:
                            params["preferences"]["prefer_landmarks"] = True
                            params["preferences"]["avoid_complex"] = True
        
        return params
    
    async def _process_navigation_request(self, nav_params: Dict[str, Any], 
                                        context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Process a navigation request with context"""
        destination = nav_params.get("destination")
        
        if not destination:
            return {"success": False, "error": "No destination specified"}
        
        # Use the navigator's navigate_to_location function
        result = await self.original_navigator.navigate_to_location(
            location_name=destination,
            current_position=None,  # Will use current position
            map_id=None  # Will use active map
        )
        
        # Enhance result with context
        if result.success:
            # Add contextual information
            result_dict = result.dict()
            result_dict["contextual_notes"] = []
            
            # Add memory-based notes
            for module_name, module_messages in messages.items():
                if module_name == "memory_core":
                    for msg in module_messages:
                        if msg.get("type") == "memory_context_available":
                            result_dict["contextual_notes"].append("Previous experiences at destination available")
            
            return result_dict
        else:
            return {"success": False, "error": result.message}
    
    async def _process_pathfinding_request(self, nav_params: Dict[str, Any],
                                         context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Process a pathfinding request"""
        destination = nav_params.get("destination")
        
        if not destination:
            return {"success": False, "error": "No destination specified"}
        
        # Find path using spatial mapper
        if self.original_navigator.spatial_mapper:
            # Would implement pathfinding logic
            return {
                "success": True,
                "message": f"Pathfinding to {destination}",
                "path_analysis": "Path computation would occur here"
            }
        
        return {"success": False, "error": "Spatial mapper not available"}
    
    async def _process_location_description(self, nav_params: Dict[str, Any],
                                          context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Process a location description request"""
        # Use the navigator's describe_surroundings function
        result = await self.original_navigator.describe_surroundings(
            position=None,  # Will use current position
            radius=10.0,
            map_id=None  # Will use active map
        )
        
        # Enhance with context
        description_dict = result.dict()
        
        # Add memory context
        for module_name, module_messages in messages.items():
            if module_name == "spatial_memory":
                for msg in module_messages:
                    if msg.get("type") == "location_memories_available":
                        memory_count = msg.get("data", {}).get("memory_count", 0)
                        if memory_count > 0:
                            description_dict["memory_note"] = f"This location has {memory_count} associated memories"
        
        return {
            "success": True,
            "description": description_dict
        }
    
    async def _analyze_navigation_feasibility(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze feasibility of navigation"""
        feasibility = {
            "can_navigate": True,
            "confidence": 1.0,
            "limiting_factors": []
        }
        
        # Check if spatial mapper is available
        if not self.original_navigator.spatial_mapper:
            feasibility["can_navigate"] = False
            feasibility["confidence"] = 0.0
            feasibility["limiting_factors"].append("No spatial map available")
        
        # Check emotional factors
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg.get("type") == "emotional_state_update":
                        emotional_state = msg.get("data", {}).get("emotional_state", {})
                        if emotional_state.get("Fear", 0) > 0.8:
                            feasibility["confidence"] *= 0.5
                            feasibility["limiting_factors"].append("High fear affecting navigation")
        
        return feasibility
    
    async def _analyze_route_options(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Analyze available route options"""
        options = []
        
        # Would analyze available routes from spatial mapper
        if self.original_navigator.spatial_mapper and self.original_navigator.spatial_mapper.context.active_map_id:
            map_id = self.original_navigator.spatial_mapper.context.active_map_id
            map_obj = self.original_navigator.spatial_mapper.maps.get(map_id)
            
            if map_obj:
                for route_id, route in map_obj.routes.items():
                    options.append({
                        "route_id": route_id,
                        "distance": route.distance,
                        "estimated_time": route.estimated_time,
                        "usage_count": route.usage_count
                    })
        
        return options[:5]  # Top 5 options
    
    async def _analyze_contextual_preferences(self, messages: Dict) -> Dict[str, Any]:
        """Analyze contextual navigation preferences"""
        preferences = {
            "speed_priority": 0.5,
            "safety_priority": 0.5,
            "landmark_preference": 0.5,
            "exploration_tendency": 0.5
        }
        
        # Adjust based on module contexts
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg.get("type") == "goal_context_available":
                        # Urgent goals increase speed priority
                        goal_priorities = msg.get("data", {}).get("goal_priorities", {})
                        max_priority = max(goal_priorities.values()) if goal_priorities else 0
                        preferences["speed_priority"] = min(1.0, 0.5 + max_priority * 0.5)
        
        return preferences
    
    async def _analyze_navigation_risks(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Analyze potential navigation risks"""
        risks = []
        
        # Check for incomplete map
        if self.original_navigator.spatial_mapper:
            if self.original_navigator.spatial_mapper.context.active_map_id:
                map_id = self.original_navigator.spatial_mapper.context.active_map_id
                map_obj = self.original_navigator.spatial_mapper.maps.get(map_id)
                
                if map_obj and map_obj.completeness < 0.5:
                    risks.append({
                        "risk_type": "incomplete_map",
                        "severity": "medium",
                        "description": "Map is less than 50% complete"
                    })
        
        return risks
    
    async def _synthesize_recommendations(self, context: SharedContext, messages: Dict) -> List[str]:
        """Synthesize navigation recommendations"""
        recommendations = []
        
        # Basic recommendations
        if self.original_navigator.spatial_mapper and self.original_navigator.spatial_mapper.context.ongoing_navigation:
            recommendations.append("Continue following current navigation guidance")
        
        # Context-based recommendations
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg.get("type") == "goal_spatial_requirement":
                        destination = msg.get("data", {}).get("destination")
                        if destination:
                            recommendations.append(f"Navigate to {destination} to complete goal")
        
        return recommendations[:3]
    
    async def _synthesize_contextual_directions(self, context: SharedContext, messages: Dict) -> List[str]:
        """Synthesize context-aware directions"""
        directions = []
        
        # Would generate directions based on current navigation state
        if self.original_navigator.spatial_mapper and self.original_navigator.spatial_mapper.context.navigation_target:
            target = self.original_navigator.spatial_mapper.context.navigation_target
            directions.append(f"Proceeding toward {target}")
        
        return directions
    
    async def _synthesize_landmark_guidance(self, context: SharedContext) -> List[str]:
        """Synthesize landmark-based guidance"""
        guidance = []
        
        # Would extract landmark guidance from current route
        if self.original_navigator.spatial_mapper:
            # Get nearest landmarks
            mapper = self.original_navigator.spatial_mapper
            if mapper.context.active_map_id and mapper.context.observer_position:
                # Simple landmark reference
                guidance.append("Use nearby landmarks for orientation")
        
        return guidance
    
    async def _synthesize_safety_considerations(self, context: SharedContext, messages: Dict) -> List[str]:
        """Synthesize safety considerations for navigation"""
        safety_notes = []
        
        # Check emotional state for safety concerns
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg.get("type") == "emotional_state_update":
                        emotional_state = msg.get("data", {}).get("emotional_state", {})
                        if emotional_state.get("Anxiety", 0) > 0.7:
                            safety_notes.append("Take familiar routes when possible")
        
        return safety_notes
    
    # Delegate all other methods to the original navigator
    def __getattr__(self, name):
        """Delegate any missing methods to the original navigator"""
        return getattr(self.original_navigator, name)
