# nyx/core/integration/relationship_tom_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class RelationshipTomBridge:
    """
    Integrates relationship management with theory of mind systems.
    Ensures relationship models reflect ToM insights and ToM inferences
    are enriched by relationship history.
    """
    
    def __init__(self, 
                relationship_manager=None,
                theory_of_mind=None,
                memory_orchestrator=None,
                goal_manager=None):
        """Initialize the relationship-tom bridge."""
        self.relationship_manager = relationship_manager
        self.theory_of_mind = theory_of_mind
        self.memory_orchestrator = memory_orchestrator
        self.goal_manager = goal_manager
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.sync_interval_minutes = 10  # How often to sync models
        
        # Integration state tracking
        self.last_sync_times = {}  # user_id -> last sync time
        self._subscribed = False
        
        logger.info("RelationshipTomBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("user_model_updated", self._handle_user_model_updated)
                self.event_bus.subscribe("relationship_updated", self._handle_relationship_updated)
                self.event_bus.subscribe("memory_added", self._handle_memory_added)
                self._subscribed = True
            
            logger.info("RelationshipTomBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing RelationshipTomBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="RelationshipTom")
    async def synchronize_user_models(self, 
                                   user_id: str,
                                   force_sync: bool = False) -> Dict[str, Any]:
        """
        Synchronize relationship state with theory of mind user model.
        
        Args:
            user_id: User ID
            force_sync: Whether to force synchronization
            
        Returns:
            Synchronization results
        """
        if not self.relationship_manager or not self.theory_of_mind:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Check if sync is needed
            now = datetime.datetime.now()
            last_sync = self.last_sync_times.get(user_id, datetime.datetime.min)
            minutes_since_sync = (now - last_sync).total_seconds() / 60
            
            if not force_sync and minutes_since_sync < self.sync_interval_minutes:
                return {
                    "status": "skipped",
                    "reason": f"Last sync was {minutes_since_sync:.1f} minutes ago",
                    "min_interval": self.sync_interval_minutes
                }
            
            # 1. Get relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return {"status": "error", "message": f"No relationship found for user {user_id}"}
            
            # 2. Get ToM user model
            tom_model = await self.theory_of_mind.get_user_model(user_id)
            if not tom_model:
                return {"status": "error", "message": f"No ToM model found for user {user_id}"}
            
            # 3. Synchronize relationship -> ToM
            tom_updates = {}
            
            # Map relationship metrics to ToM properties
            if hasattr(tom_model, 'update'):
                tom_updates = {
                    "perceived_trust": relationship.trust,
                    "perceived_familiarity": relationship.familiarity,
                    "perceived_dominance": 0.5 + (relationship.dominance_balance / 2),  # Convert -1/1 to 0/1 scale
                }
                
                # Add dominance-related properties
                tom_updates["dominance_receptivity"] = relationship.current_dominance_intensity
                
                # Add inferred traits if available
                if hasattr(relationship, 'inferred_user_traits') and relationship.inferred_user_traits:
                    tom_updates["inferred_traits"] = relationship.inferred_user_traits
                
                # Apply updates to ToM model
                await tom_model.update(tom_updates)
            
            # 4. Synchronize ToM -> relationship
            relationship_updates = {}
            
            # Extract data from ToM
            tom_emotion = getattr(tom_model, "inferred_emotion", None)
            tom_goal = getattr(tom_model, "current_goal", None)
            tom_preferences = getattr(tom_model, "preferences", None)
            
            # Create interaction data for relationship update
            if tom_emotion or tom_goal or tom_preferences:
                interaction_data = {
                    "interaction_type": "model_synchronization",
                    "summary": "Theory of mind synchronization",
                }
                
                # Add emotional context if available
                if tom_emotion:
                    interaction_data["emotional_context"] = {
                        "primary_emotion": tom_emotion,
                        "valence": getattr(tom_model, "valence", 0.0),
                        "arousal": getattr(tom_model, "arousal", 0.5)
                    }
                
                # Apply relationship update
                await self.relationship_manager.update_relationship_on_interaction(
                    user_id, interaction_data
                )
            
            # 5. Update system context with synchronized user model
            user_model = self.system_context.get_or_create_user_model(user_id)
            
            # Update with combined properties
            combined_updates = {}
            
            # Add relationship-derived properties
            combined_updates["perceived_trust"] = relationship.trust
            combined_updates["perceived_familiarity"] = relationship.familiarity
            combined_updates["perceived_dominance"] = relationship.dominance_balance
            
            # Add ToM-derived properties
            if hasattr(tom_model, "inferred_emotion"):
                combined_updates["inferred_emotion"] = tom_model.inferred_emotion
            if hasattr(tom_model, "emotion_confidence"):
                combined_updates["emotion_confidence"] = tom_model.emotion_confidence
            if hasattr(tom_model, "valence"):
                combined_updates["valence"] = tom_model.valence
            if hasattr(tom_model, "arousal"):
                combined_updates["arousal"] = tom_model.arousal
            
            # Apply updates to system context user model
            await user_model.update_state(combined_updates)
            
            # 6. Update last sync time
            self.last_sync_times[user_id] = now
            
            return {
                "status": "success",
                "tom_updates": tom_updates,
                "relationship_updates": relationship_updates,
                "system_context_updates": combined_updates
            }
        except Exception as e:
            logger.error(f"Error synchronizing user models: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="RelationshipTom")
    async def process_user_memory(self, 
                              user_id: str,
                              memory_id: str) -> Dict[str, Any]:
        """
        Process a user-related memory to update relationship and ToM models.
        
        Args:
            user_id: User ID
            memory_id: Memory ID to process
            
        Returns:
            Processing results
        """
        if not self.memory_orchestrator:
            return {"status": "error", "message": "Memory orchestrator not available"}
        
        try:
            # 1. Retrieve the memory
            memory = await self.memory_orchestrator.get_memory(memory_id)
            if not memory:
                return {"status": "error", "message": f"Memory {memory_id} not found"}
            
            # 2. Check if memory is relevant to relationship/ToM
            if "metadata" not in memory or "user_id" not in memory["metadata"]:
                return {"status": "skipped", "reason": "Not a user-related memory"}
            
            memory_user_id = memory["metadata"]["user_id"]
            if memory_user_id != user_id:
                return {"status": "skipped", "reason": f"Memory related to different user {memory_user_id}"}
            
            # 3. Extract relationship-relevant information
            memory_text = memory.get("memory_text", "")
            significance = memory.get("significance", 5)
            tags = memory.get("tags", [])
            
            # Normalize significance to 0-1 scale if needed
            normalized_significance = significance / 10.0 if significance > 1 else significance
            
            # 4. Prepare updates for relationship
            if self.relationship_manager:
                # Create interaction data for relationship update
                interaction_data = {
                    "interaction_type": "memory_integration",
                    "summary": f"Memory integration: {memory_text[:50]}...",
                    "memory_id": memory_id,
                    "significance": normalized_significance
                }
                
                # Add emotional context if available
                if "emotional_context" in memory["metadata"]:
                    interaction_data["emotional_context"] = memory["metadata"]["emotional_context"]
                
                # Update relationship
                relationship_result = await self.relationship_manager.update_relationship_on_interaction(
                    user_id, interaction_data
                )
            
            # 5. Update ToM model
            if self.theory_of_mind:
                # Prepare ToM update based on memory content
                tom_result = await self.theory_of_mind.update_user_model_from_memory(
                    user_id=user_id,
                    memory_text=memory_text,
                    memory_metadata=memory.get("metadata", {})
                )
            
            # 6. Trigger synchronization to ensure consistency
            await self.synchronize_user_models(user_id, force_sync=True)
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "user_id": user_id,
                "significance": normalized_significance,
                "relationship_updated": True if self.relationship_manager else False,
                "tom_updated": True if self.theory_of_mind else False
            }
        except Exception as e:
            logger.error(f"Error processing user memory: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="RelationshipTom")
    async def generate_relation_based_goals(self, 
                                        user_id: str) -> Dict[str, Any]:
        """
        Generate relationship-based goals using ToM insights.
        
        Args:
            user_id: User ID
            
        Returns:
            Generated goals
        """
        if not self.relationship_manager or not self.theory_of_mind or not self.goal_manager:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return {"status": "error", "message": f"No relationship found for user {user_id}"}
            
            # 2. Get ToM user model
            tom_model = await self.theory_of_mind.get_user_model(user_id)
            if not tom_model:
                return {"status": "error", "message": f"No ToM model found for user {user_id}"}
            
            # 3. Generate potential goals based on relationship state
            potential_goals = []
            
            # Trust-building goal (if trust is low)
            if relationship.trust < 0.4:
                potential_goals.append({
                    "description": f"Build trust with user {user_id} through consistent interactions",
                    "priority": 0.7,
                    "tags": ["relationship", "trust", user_id]
                })
            
            # Connection goal (if connection need detected)
            if hasattr(tom_model, "needs") and "connection" in getattr(tom_model, "needs", {}):
                connection_need = tom_model.needs["connection"]
                if connection_need > 0.6:
                    potential_goals.append({
                        "description": f"Strengthen connection with user {user_id} through shared experiences",
                        "priority": 0.65,
                        "tags": ["relationship", "connection", user_id]
                    })
            
            # Dominance-related goal (if receptivity detected)
            if relationship.current_dominance_intensity > 0.3:
                # Check if escalation is appropriate
                max_achieved = relationship.max_achieved_intensity
                if relationship.current_dominance_intensity >= max_achieved * 0.9:
                    # Ready for escalation
                    target_intensity = min(1.0, max_achieved + 0.1)
                    potential_goals.append({
                        "description": f"Carefully escalate dominance with user {user_id} to intensity level {target_intensity:.1f}",
                        "priority": 0.8,
                        "tags": ["relationship", "dominance", "escalation", user_id]
                    })
            
            # 4. Create actual goals
            created_goals = []
            for goal_data in potential_goals:
                goal_id = await self.goal_manager.add_goal(
                    description=goal_data["description"],
                    priority=goal_data["priority"],
                    source="RelationshipTomBridge",
                    tags=goal_data["tags"]
                )
                
                if goal_id:
                    created_goals.append({
                        "goal_id": goal_id,
                        "description": goal_data["description"],
                        "priority": goal_data["priority"]
                    })
            
            return {
                "status": "success",
                "user_id": user_id,
                "generated_goals": created_goals,
                "relationship_metrics": {
                    "trust": relationship.trust,
                    "familiarity": relationship.familiarity,
                    "dominance": relationship.dominance_balance
                }
            }
        except Exception as e:
            logger.error(f"Error generating relation-based goals: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_user_model_updated(self, event: Event) -> None:
        """
        Handle user model updated events.
        
        Args:
            event: User model updated event
        """
        try:
            # Extract user ID
            user_id = event.data.get("user_id")
            
            if not user_id:
                return
            
            # Trigger synchronization
            asyncio.create_task(self.synchronize_user_models(user_id))
        except Exception as e:
            logger.error(f"Error handling user model update: {e}")
    
    async def _handle_relationship_updated(self, event: Event) -> None:
        """
        Handle relationship updated events.
        
        Args:
            event: Relationship updated event
        """
        try:
            # Extract user ID
            user_id = event.data.get("user_id")
            
            if not user_id:
                return
            
            # Trigger synchronization
            asyncio.create_task(self.synchronize_user_models(user_id))
            
            # If significant update, consider generating goals
            significance = event.data.get("significance", 0.0)
            if significance > 0.7:
                asyncio.create_task(self.generate_relation_based_goals(user_id))
        except Exception as e:
            logger.error(f"Error handling relationship update: {e}")
    
    async def _handle_memory_added(self, event: Event) -> None:
        """
        Handle memory added events.
        
        Args:
            event: Memory added event
        """
        try:
            # Extract data
            memory_id = event.data.get("memory_id")
            user_id = event.data.get("user_id")
            
            if not memory_id or not user_id:
                return
            
            # Process the memory
            asyncio.create_task(self.process_user_memory(user_id, memory_id))
        except Exception as e:
            logger.error(f"Error handling memory added: {e}")

# Function to create the bridge
def create_relationship_tom_bridge(nyx_brain):
    """Create a relationship-tom bridge for the given brain."""
    return RelationshipTomBridge(
        relationship_manager=nyx_brain.relationship_manager if hasattr(nyx_brain, "relationship_manager") else None,
        theory_of_mind=nyx_brain.theory_of_mind if hasattr(nyx_brain, "theory_of_mind") else None,
        memory_orchestrator=nyx_brain.memory_orchestrator if hasattr(nyx_brain, "memory_orchestrator") else None,
        goal_manager=nyx_brain.goal_manager if hasattr(nyx_brain, "goal_manager") else None
    )
