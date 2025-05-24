# nyx/core/a2a/context_aware_spatial_memory.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareSpatialMemoryIntegration(ContextAwareModule):
    """
    Enhanced SpatialMemoryIntegration with context distribution capabilities
    """
    
    def __init__(self, original_spatial_memory):
        super().__init__("spatial_memory")
        self.original_memory = original_spatial_memory
        self.context_subscriptions = [
            "location_change", "memory_retrieval_request", "emotional_memory_trigger",
            "landmark_identified", "spatial_event_occurred", "goal_location_association"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize spatial memory processing for this context"""
        logger.debug(f"SpatialMemory received context for user: {context.user_id}")
        
        # Check for spatial references in input
        spatial_refs = await self._extract_spatial_references(context.user_input)
        
        # Send initial spatial memory context
        await self.send_context_update(
            update_type="spatial_memory_context",
            data={
                "spatial_references_found": len(spatial_refs) > 0,
                "spatial_refs": spatial_refs,
                "active_location": await self._get_current_location_context(),
                "recent_spatial_events": await self._get_recent_spatial_events()
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that affect spatial memory"""
        
        if update.update_type == "location_change":
            # Store memory of location change
            location_data = update.data
            await self._record_location_change(location_data)
            
            # Check for memories at new location
            memories = await self._retrieve_location_memories(location_data.get("new_location"))
            if memories:
                await self.send_context_update(
                    update_type="location_memories_available",
                    data={
                        "location": location_data.get("new_location"),
                        "memories": memories,
                        "memory_count": len(memories)
                    }
                )
        
        elif update.update_type == "emotional_state_update":
            # Link emotional state to current location
            emotional_data = update.data
            await self._link_emotion_to_location(emotional_data)
        
        elif update.update_type == "landmark_identified":
            # Register landmark with episodic significance
            landmark_data = update.data
            await self._register_episodic_landmark(landmark_data)
        
        elif update.update_type == "goal_completion":
            # Record goal completion at location
            goal_data = update.data
            await self._record_goal_completion_location(goal_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for spatial memory operations"""
        # Check if input requests spatial memories
        if self._requests_spatial_memories(context.user_input):
            memories = await self._retrieve_relevant_spatial_memories(context)
            
            await self.send_context_update(
                update_type="spatial_memories_retrieved",
                data={
                    "memory_count": len(memories),
                    "memories": memories
                }
            )
            
            return {
                "spatial_memories_retrieved": True,
                "memory_count": len(memories),
                "memories": memories
            }
        
        # Store current interaction as spatial memory if location is known
        if self.original_memory.spatial_mapper and self.original_memory.spatial_mapper.context.observer_position:
            memory_id = await self._store_interaction_memory(context)
            return {
                "spatial_memory_stored": True,
                "memory_id": memory_id
            }
        
        return {"spatial_memory_processing": "minimal"}
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze spatial memory patterns"""
        analysis = {
            "spatial_memory_density": await self._analyze_memory_density(),
            "significant_locations": await self._identify_significant_locations(),
            "spatial_emotional_associations": await self._analyze_spatial_emotions(),
            "navigation_patterns": await self._analyze_navigation_patterns()
        }
        
        return analysis
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize spatial memory insights for response"""
        messages = await self.get_cross_module_messages()
        
        synthesis = {
            "location_based_insights": await self._generate_location_insights(context),
            "spatial_memory_prompts": await self._generate_memory_prompts(context),
            "landmark_memories": await self._synthesize_landmark_memories(context),
            "spatial_narrative": await self._create_spatial_narrative(context, messages)
        }
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _extract_spatial_references(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract spatial references from user input"""
        input_lower = user_input.lower()
        spatial_refs = []
        
        # Location keywords
        location_keywords = ["at", "in", "near", "by", "from", "to", "where", "place", "location"]
        
        for keyword in location_keywords:
            if keyword in input_lower:
                # Simple extraction - would be enhanced with NLP
                idx = input_lower.find(keyword)
                context_start = max(0, idx - 20)
                context_end = min(len(user_input), idx + 30)
                
                spatial_refs.append({
                    "keyword": keyword,
                    "context": user_input[context_start:context_end],
                    "position": idx
                })
        
        return spatial_refs
    
    async def _get_current_location_context(self) -> Optional[Dict[str, Any]]:
        """Get current location context"""
        if not self.original_memory.spatial_mapper:
            return None
        
        mapper = self.original_memory.spatial_mapper
        if not mapper.context.observer_position:
            return None
        
        return {
            "position": mapper.context.observer_position.dict(),
            "active_map": mapper.context.active_map_id,
            "navigation_active": mapper.context.ongoing_navigation
        }
    
    async def _get_recent_spatial_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent spatial events"""
        if not self.original_memory.spatial_events:
            return []
        
        # Get most recent events
        events = sorted(
            self.original_memory.spatial_events.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )[:limit]
        
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "location_id": e.location_id,
                "timestamp": e.timestamp,
                "description": e.description[:100]
            }
            for e in events
        ]
    
    async def _record_location_change(self, location_data: Dict[str, Any]):
        """Record a location change as a spatial memory"""
        old_location = location_data.get("old_location")
        new_location = location_data.get("new_location")
        
        memory_text = f"Moved from {old_location} to {new_location}"
        
        await self.original_memory.store_spatial_memory(
            memory_text=memory_text,
            location_id=new_location,
            tags=["movement", "location_change"],
            memory_type="observation",
            significance=3
        )
    
    async def _retrieve_location_memories(self, location_id: str) -> List[Dict[str, Any]]:
        """Retrieve memories for a specific location"""
        if not location_id:
            return []
        
        return await self.original_memory.retrieve_memories_at_location(
            location_id=location_id,
            limit=5
        )
    
    async def _link_emotion_to_location(self, emotional_data: Dict[str, Any]):
        """Link emotional state to current location"""
        if not self.original_memory.spatial_mapper:
            return
        
        mapper = self.original_memory.spatial_mapper
        if not mapper.context.observer_position:
            return
        
        # Store emotional context at location
        dominant_emotion = emotional_data.get("dominant_emotion")
        if dominant_emotion:
            emotion_name, strength = dominant_emotion
            
            memory_text = f"Felt {emotion_name} (strength: {strength:.2f}) at this location"
            
            await self.original_memory.store_spatial_memory(
                memory_text=memory_text,
                coordinates=mapper.context.observer_position.dict(),
                tags=["emotion", emotion_name.lower()],
                memory_type="experience",
                significance=int(5 + strength * 3)  # Higher significance for stronger emotions
            )
    
    async def _register_episodic_landmark(self, landmark_data: Dict[str, Any]):
        """Register a landmark with episodic significance"""
        landmark_id = landmark_data.get("landmark_id")
        significance_reason = landmark_data.get("significance", "Notable landmark")
        
        if landmark_id:
            await self.original_memory.register_episodic_landmark(
                landmark_id=landmark_id,
                description=significance_reason,
                salience=landmark_data.get("salience", 0.7),
                properties={"discovery_context": landmark_data}
            )
    
    async def _record_goal_completion_location(self, goal_data: Dict[str, Any]):
        """Record goal completion at current location"""
        if not self.original_memory.spatial_mapper:
            return
        
        mapper = self.original_memory.spatial_mapper
        if not mapper.context.observer_position:
            return
        
        goal_description = goal_data.get("goal_description", "Goal completed")
        
        # Create spatial event for goal completion
        await self.original_memory.record_spatial_event(
            event_type="goal_completion",
            description=f"Completed goal: {goal_description}",
            coordinates=mapper.context.observer_position.dict(),
            properties={
                "goal_id": goal_data.get("goal_id"),
                "goal_type": goal_data.get("goal_type")
            },
            store_as_memory=True
        )
    
    def _requests_spatial_memories(self, user_input: str) -> bool:
        """Check if input requests spatial memories"""
        memory_keywords = ["remember", "recall", "what happened", "memory", "memories", 
                          "been here", "this place", "last time"]
        spatial_keywords = ["here", "there", "place", "location", "room", "area"]
        
        input_lower = user_input.lower()
        
        has_memory_keyword = any(keyword in input_lower for keyword in memory_keywords)
        has_spatial_keyword = any(keyword in input_lower for keyword in spatial_keywords)
        
        return has_memory_keyword and has_spatial_keyword
    
    async def _retrieve_relevant_spatial_memories(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Retrieve spatial memories relevant to current context"""
        memories = []
        
        # Get memories at current location
        if self.original_memory.spatial_mapper and self.original_memory.spatial_mapper.context.observer_position:
            position = self.original_memory.spatial_mapper.context.observer_position
            
            memories = await self.original_memory.retrieve_memories_near_coordinates(
                coordinates=position.dict(),
                radius=15.0,
                limit=10
            )
        
        return memories
    
    async def _store_interaction_memory(self, context: SharedContext) -> Optional[str]:
        """Store current interaction as spatial memory"""
        if not self.original_memory.spatial_mapper:
            return None
        
        mapper = self.original_memory.spatial_mapper
        if not mapper.context.observer_position:
            return None
        
        # Create memory text from interaction
        memory_text = f"Interaction: {context.user_input[:200]}"
        
        result = await self.original_memory.store_spatial_memory(
            memory_text=memory_text,
            coordinates=mapper.context.observer_position.dict(),
            tags=["interaction", "conversation"],
            memory_type="observation",
            significance=4
        )
        
        return result.get("memory_id")
    
    async def _analyze_memory_density(self) -> Dict[str, Any]:
        """Analyze spatial memory density"""
        if not self.original_memory.coordinate_grid:
            return {"density": 0, "hotspots": []}
        
        # Calculate memory density per grid cell
        densities = []
        for grid_key, memory_ids in self.original_memory.coordinate_grid.items():
            densities.append({
                "grid": grid_key,
                "memory_count": len(memory_ids),
                "density": len(memory_ids) / (self.original_memory.grid_cell_size ** 2)
            })
        
        # Sort by density
        densities.sort(key=lambda x: x["density"], reverse=True)
        
        return {
            "average_density": sum(d["density"] for d in densities) / len(densities) if densities else 0,
            "hotspots": densities[:3],  # Top 3 memory hotspots
            "total_grid_cells": len(self.original_memory.coordinate_grid)
        }
    
    async def _identify_significant_locations(self) -> List[Dict[str, Any]]:
        """Identify locations with significant memories"""
        significant = []
        
        # Analyze location memory index
        for location_id, memory_ids in self.original_memory.location_memory_index.items():
            if len(memory_ids) >= 3:  # Threshold for significance
                significant.append({
                    "location_id": location_id,
                    "memory_count": len(memory_ids),
                    "significance_score": min(1.0, len(memory_ids) / 10.0)
                })
        
        # Sort by significance
        significant.sort(key=lambda x: x["significance_score"], reverse=True)
        return significant[:5]
    
    async def _analyze_spatial_emotions(self) -> Dict[str, Any]:
        """Analyze emotional associations with locations"""
        emotional_locations = {}
        
        # Would analyze stored memories for emotional content
        # This is a simplified version
        return {
            "emotional_hotspots": [],
            "dominant_spatial_emotion": "neutral",
            "emotional_variance": 0.0
        }
    
    async def _analyze_navigation_patterns(self) -> Dict[str, Any]:
        """Analyze navigation patterns from spatial memories"""
        # Would analyze movement patterns from stored events
        return {
            "frequently_visited": [],
            "navigation_loops": 0,
            "exploration_tendency": 0.5
        }
    
    async def _generate_location_insights(self, context: SharedContext) -> List[str]:
        """Generate insights about current location"""
        insights = []
        
        location_context = await self._get_current_location_context()
        if location_context:
            # Get memories at current location
            if self.original_memory.spatial_mapper:
                memories = await self._retrieve_location_memories("current")
                if memories:
                    insights.append(f"This location has {len(memories)} associated memories")
        
        return insights
    
    async def _generate_memory_prompts(self, context: SharedContext) -> List[str]:
        """Generate prompts based on spatial memories"""
        prompts = []
        
        # Get recent spatial events
        recent_events = await self._get_recent_spatial_events(3)
        for event in recent_events:
            prompts.append(f"Remember: {event['description']}")
        
        return prompts
    
    async def _synthesize_landmark_memories(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Synthesize memories associated with landmarks"""
        landmark_memories = []
        
        for landmark_id, landmark_info in self.original_memory.episodic_landmarks.items():
            # Get memories at landmark
            memories = await self.original_memory.retrieve_memories_at_location(
                location_id=landmark_id,
                limit=3
            )
            
            if memories:
                landmark_memories.append({
                    "landmark": landmark_info.name,
                    "memory_count": len(memories),
                    "salience": landmark_info.salience
                })
        
        return landmark_memories
    
    async def _create_spatial_narrative(self, context: SharedContext, messages: Dict) -> str:
        """Create a spatial narrative from memories and context"""
        narrative_parts = []
        
        # Add current location context
        location_context = await self._get_current_location_context()
        if location_context:
            narrative_parts.append("Currently present in the mapped environment")
        
        # Add significant locations
        significant = await self._identify_significant_locations()
        if significant:
            narrative_parts.append(f"{len(significant)} locations hold significant memories")
        
        # Add emotional context
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg.get("type") == "emotional_state_update":
                        narrative_parts.append("Emotional state influences spatial perception")
                        break
        
        return ". ".join(narrative_parts) if narrative_parts else "Spatial context is minimal"
    
    # Delegate all other methods to the original memory integration
    def __getattr__(self, name):
        """Delegate any missing methods to the original memory integration"""
        return getattr(self.original_memory, name)
