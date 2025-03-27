# nyx/core/integration/dominance_memory_reflection_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class DominanceMemoryReflectionBridge:
    """
    Integrates dominance expressions with memory and reflection systems.
    Ensures dominance interactions are properly stored, reflected upon,
    and learned from, leading to improved dominance capabilities over time.
    """
    
    def __init__(self, 
                dominance_system=None,
                memory_orchestrator=None,
                reflection_engine=None,
                relationship_manager=None):
        """Initialize the bridge."""
        self.dominance_system = dominance_system
        self.memory_orchestrator = memory_orchestrator
        self.reflection_engine = reflection_engine
        self.relationship_manager = relationship_manager
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Memory and reflection configuration
        self.min_significance_threshold = 4  # Minimum significance for memory storage (1-10)
        self.reflection_threshold = 0.7  # Threshold for triggering reflection
        self.min_memories_for_reflection = 3  # Minimum memories needed for reflection
        
        # Track dominance memories by user
        self.dominance_memories_by_user = {}  # user_id -> list of memory IDs
        
        # Integration event subscriptions
        self._subscribed = False
        
        logger.info("DominanceMemoryReflectionBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("dominance_action", self._handle_dominance_action)
                self.event_bus.subscribe("dominance_outcome", self._handle_dominance_outcome)
                self._subscribed = True
            
            logger.info("DominanceMemoryReflectionBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing DominanceMemoryReflectionBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceMemoryReflection")
    async def store_dominance_memory(self, 
                                  user_id: str,
                                  memory_text: str,
                                  significance: int,
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a dominance-related memory.
        
        Args:
            user_id: User ID
            memory_text: Memory text
            significance: Memory significance (1-10)
            metadata: Additional metadata
            
        Returns:
            Memory storage result
        """
        if not self.memory_orchestrator:
            return {"status": "error", "message": "Memory orchestrator not available"}
        
        try:
            # Set dominance tags
            tags = ["dominance", "user_interaction"]
            
            # Add additional tags based on metadata
            if metadata.get("dominance_type"):
                tags.append(metadata["dominance_type"])
                
            if metadata.get("outcome") == "success":
                tags.append("success")
            elif metadata.get("outcome") == "failure":
                tags.append("failure")
                
            # Add intensity tag
            intensity = metadata.get("intensity", 0.5)
            if intensity < 0.3:
                tags.append("low_intensity")
            elif intensity < 0.7:
                tags.append("medium_intensity")
            else:
                tags.append("high_intensity")
            
            # Add emotional context to metadata
            if "emotional_context" not in metadata:
                # Default emotional context
                metadata["emotional_context"] = {
                    "primary_emotion": "dominance" if metadata.get("outcome") == "success" else "frustration",
                    "valence": 0.7 if metadata.get("outcome") == "success" else -0.3,
                    "arousal": 0.6 if metadata.get("intensity", 0.5) > 0.5 else 0.4
                }
            
            # Add user_id to metadata
            metadata["user_id"] = user_id
            
            # Store memory
            memory_id = await self.memory_orchestrator.add_memory(
                memory_text=memory_text,
                memory_type="experience",
                significance=significance,
                tags=tags,
                metadata=metadata
            )
            
            # Track memory by user
            if user_id not in self.dominance_memories_by_user:
                self.dominance_memories_by_user[user_id] = []
                
            self.dominance_memories_by_user[user_id].append(memory_id)
            
            # Trigger reflection if enough memories
            user_memories = self.dominance_memories_by_user.get(user_id, [])
            if len(user_memories) >= self.min_memories_for_reflection:
                asyncio.create_task(self._trigger_dominance_reflection(user_id))
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "significance": significance,
                "tags": tags
            }
            
        except Exception as e:
            logger.error(f"Error storing dominance memory: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceMemoryReflection")
    async def _trigger_dominance_reflection(self, user_id: str) -> Dict[str, Any]:
        """
        Trigger a reflection on dominance interactions with a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Reflection result
        """
        if not self.reflection_engine or not self.memory_orchestrator:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Get dominance memories for this user
            memory_ids = self.dominance_memories_by_user.get(user_id, [])
            if len(memory_ids) < self.min_memories_for_reflection:
                return {"status": "error", "message": "Not enough memories for reflection"}
                
            # Retrieve memories
            memories = []
            for memory_id in memory_ids[-10:]:  # Use last 10 at most
                memory = await self.memory_orchestrator.get_memory(memory_id)
                if memory:
                    memories.append(memory)
            
            if not memories:
                return {"status": "error", "message": "Failed to retrieve memories"}
                
            # Prepare neurochemical context
            neurochemical_state = {
                "nyxamine": 0.7,  # High dopamine for dominance
                "seranix": 0.5,
                "oxynixin": 0.3,
                "cortanyx": 0.2,
                "adrenyx": 0.6
            }
            
            # Generate reflection
            reflection_text, confidence = await self.reflection_engine.generate_reflection(
                memories=memories,
                topic=f"dominance_interactions_with_{user_id}",
                neurochemical_state=neurochemical_state
            )
            
            if not reflection_text:
                return {"status": "error", "message": "Failed to generate reflection"}
                
            # Store reflection as memory
            reflection_metadata = {
                "user_id": user_id,
                "reflection_type": "dominance",
                "source_memory_ids": memory_ids[-10:],
                "neurochemical_state": neurochemical_state
            }
            
            reflection_memory_id = await self.memory_orchestrator.add_memory(
                memory_text=reflection_text,
                memory_type="reflection",
                significance=8,  # High significance for dominance reflections
                tags=["dominance", "reflection", user_id],
                metadata=reflection_metadata
            )
            
            # Update relationship with reflection insights
            if self.relationship_manager:
                # Extract key insights from reflection
                # This would ideally use an LLM to extract structured insights
                # For now, we use a simplified approach
                
                # Check for success/failure mentions
                success_indicators = ["success", "successful", "well", "effective", "positive"]
                failure_indicators = ["fail", "unsuccessful", "negative", "issue", "problem"]
                
                success_count = sum(1 for indicator in success_indicators if indicator in reflection_text.lower())
                failure_count = sum(1 for indicator in failure_indicators if indicator in reflection_text.lower())
                
                positive_reflection = success_count > failure_count
                
                # Update relationship if clear positive/negative reflection
                if positive_reflection or failure_count > success_count:
                    relationship_update = {
                        "interaction_type": "dominance_reflection",
                        "summary": f"Reflection on dominance interactions with {user_id}",
                        "emotional_context": {
                            "primary_emotion": "satisfaction" if positive_reflection else "analytical",
                            "valence": 0.6 if positive_reflection else 0.0,
                            "arousal": 0.4
                        }
                    }
                    
                    await self.relationship_manager.update_relationship_on_interaction(
                        user_id, relationship_update
                    )
            
            return {
                "status": "success",
                "reflection_text": reflection_text,
                "confidence": confidence,
                "reflection_memory_id": reflection_memory_id,
                "memories_used": len(memories),
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error triggering dominance reflection: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceMemoryReflection")
    async def get_dominance_patterns(self, user_id: str) -> Dict[str, Any]:
        """
        Extract patterns from dominance interactions with a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Extracted patterns
        """
        if not self.memory_orchestrator or not self.reflection_engine:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Get dominance memories for this user
            memory_ids = self.dominance_memories_by_user.get(user_id, [])
            if not memory_ids:
                return {"status": "error", "message": "No dominance memories found for this user"}
                
            # Retrieve memories
            memories = []
            for memory_id in memory_ids:
                memory = await self.memory_orchestrator.get_memory(memory_id)
                if memory:
                    memories.append(memory)
            
            if not memories:
                return {"status": "error", "message": "Failed to retrieve memories"}
                
            # Create abstraction using reflection engine
            abstraction_text, abstraction_data = await self.reflection_engine.create_abstraction(
                memories=memories,
                pattern_type="dominance_behavior"
            )
            
            if not abstraction_text:
                return {"status": "error", "message": "Failed to create abstraction"}
                
            # Return patterns
            return {
                "status": "success",
                "patterns": abstraction_text,
                "pattern_data": abstraction_data,
                "memories_analyzed": len(memories),
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error extracting dominance patterns: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_dominance_action(self, event: Event) -> None:
        """
        Handle dominance action events from the event bus.
        
        Args:
            event: Dominance action event
        """
        try:
            # Extract event data
            action = event.data.get("action")
            user_id = event.data.get("user_id")
            intensity = event.data.get("intensity", 0.5)
            
            if not action or not user_id:
                return
            
            # Store memory if action is significant
            significance = int(5 + (intensity * 5))  # Scale intensity to significance (1-10)
            
            if significance >= self.min_significance_threshold:
                # Create memory text
                memory_text = f"Executed dominance action '{action}' on user {user_id} with intensity {intensity:.2f}"
                
                # Create metadata
                metadata = {
                    "action_type": action,
                    "intensity": intensity,
                    "dominance_type": "command" if "command" in action.lower() else (
                        "escalation" if "escalate" in action.lower() else "general"
                    ),
                    "timestamp": event.timestamp.isoformat()
                }
                
                # Store memory
                asyncio.create_task(self.store_dominance_memory(
                    user_id=user_id,
                    memory_text=memory_text,
                    significance=significance,
                    metadata=metadata
                ))
            
        except Exception as e:
            logger.error(f"Error handling dominance action event: {e}")
    
    async def _handle_dominance_outcome(self, event: Event) -> None:
        """
        Handle dominance outcome events from the event bus.
        
        Args:
            event: Dominance outcome event
        """
        try:
            # Extract event data
            action = event.data.get("action")
            user_id = event.data.get("user_id")
            intensity = event.data.get("intensity", 0.5)
            outcome = event.data.get("outcome")
            user_response = event.data.get("user_response", "")
            
            if not action or not user_id or not outcome:
                return
            
            # Calculate significance based on intensity and outcome
            significance = int(5 + (intensity * 5))  # Base significance
            
            # Adjust significance for very successful or failed outcomes
            if outcome == "success" and intensity > 0.7:
                significance += 1
            elif outcome == "failure" and intensity > 0.5:
                significance += 2  # Failed high-intensity actions are particularly noteworthy
            
            # Ensure significance is within bounds
            significance = max(1, min(10, significance))
            
            # Create memory text
            memory_text = f"Dominance action '{action}' on user {user_id} with intensity {intensity:.2f} resulted in {outcome}."
            if user_response:
                memory_text += f" User response: {user_response}"
            
            # Create metadata
            metadata = {
                "action_type": action,
                "intensity": intensity,
                "outcome": outcome,
                "user_response": user_response,
                "dominance_type": "command" if "command" in action.lower() else (
                    "escalation" if "escalate" in action.lower() else "general"
                ),
                "timestamp": event.timestamp.isoformat()
            }
            
            # Store memory
            asyncio.create_task(self.store_dominance_memory(
                user_id=user_id,
                memory_text=memory_text,
                significance=significance,
                metadata=metadata
            ))
            
            # Trigger reflection immediately for high-significance outcomes
            if (outcome == "success" and intensity > 0.8) or (outcome == "failure" and intensity > 0.5):
                if user_id in self.dominance_memories_by_user and len(self.dominance_memories_by_user[user_id]) >= self.min_memories_for_reflection:
                    asyncio.create_task(self._trigger_dominance_reflection(user_id))
            
        except Exception as e:
            logger.error(f"Error handling dominance outcome event: {e}")

# Function to create the bridge
def create_dominance_memory_reflection_bridge(nyx_brain):
    """Create a dominance-memory-reflection bridge for the given brain."""
    return DominanceMemoryReflectionBridge(
        dominance_system=nyx_brain.dominance_system if hasattr(nyx_brain, "dominance_system") else None,
        memory_orchestrator=nyx_brain.memory_orchestrator if hasattr(nyx_brain, "memory_orchestrator") else None,
        reflection_engine=nyx_brain.reflection_engine if hasattr(nyx_brain, "reflection_engine") else None,
        relationship_manager=nyx_brain.relationship_manager if hasattr(nyx_brain, "relationship_manager") else None
    )
