# nyx/core/integration/autonomous_cognitive_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

# Import the core modules to integrate
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.reflexive_system import ReflexiveSystem
from nyx.core.relationship_manager import RelationshipManager
from nyx.core.temporal_perception import TemporalPerceptionSystem

logger = logging.getLogger(__name__)

class AutonomousCognitiveBridge:
    """
    Integration bridge connecting autonomous cognitive systems.
    
    This bridge integrates the reflection engine, reflexive system, relationship manager,
    and temporal perception system with the broader Nyx architecture. It enables:
    
    1. Cross-module access to cognitive functions
    2. Event-based communication between cognitive modules and other systems
    3. Coordinated cognitive processing across reflection, reflexes, and relationships
    4. Time-aware cognitive processes through temporal perception integration
    5. Unified cognitive state monitoring and management
    """
    
    def __init__(self, 
                nyx_brain=None,
                reflection_engine=None,
                reflexive_system=None,
                relationship_manager=None,
                temporal_perception=None):
        """Initialize the autonomous cognitive bridge."""
        self.brain = nyx_brain
        
        # Store module references
        self.reflection_engine = reflection_engine
        self.reflexive_system = reflexive_system
        self.relationship_manager = relationship_manager
        self.temporal_perception = temporal_perception
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Track related core systems
        self.memory_orchestrator = None
        self.emotional_core = None
        self.theory_of_mind = None
        self.knowledge_core = None
        self.action_selector = None
        
        # Integration state tracking
        self._subscribed = False
        self._initialized = False
        
        # Cognitive session tracking
        self.active_sessions = {}  # user_id -> session data
        
        # Recent cognitive activities
        self.recent_reflections = []
        self.recent_reflexes = []
        self.recent_relationship_updates = []
        
        # Performance metrics
        self.cognitive_metrics = {
            "reflection_request_count": 0,
            "reflection_success_rate": 1.0,
            "reflexive_request_count": 0,
            "reflexive_response_time_ms": 0,
            "relationship_update_count": 0,
            "avg_temporal_processing_ms": 0
        }
        
        logger.info("AutonomousCognitiveBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Set up references to core systems if not provided
            if not self.reflection_engine and hasattr(self.brain, "reflection_engine"):
                self.reflection_engine = self.brain.reflection_engine
            
            if not self.reflexive_system and hasattr(self.brain, "reflexive_system"):
                self.reflexive_system = self.brain.reflexive_system
            
            if not self.relationship_manager and hasattr(self.brain, "relationship_manager"):
                self.relationship_manager = self.brain.relationship_manager
                
            if not self.temporal_perception and hasattr(self.brain, "temporal_perception"):
                self.temporal_perception = self.brain.temporal_perception
            
            # Get related core systems
            if hasattr(self.brain, "memory_orchestrator"):
                self.memory_orchestrator = self.brain.memory_orchestrator
                
            if hasattr(self.brain, "emotional_core"):
                self.emotional_core = self.brain.emotional_core
                
            if hasattr(self.brain, "theory_of_mind"):
                self.theory_of_mind = self.brain.theory_of_mind
                
            if hasattr(self.brain, "knowledge_core"):
                self.knowledge_core = self.brain.knowledge_core
                
            if hasattr(self.brain, "action_selector"):
                self.action_selector = self.brain.action_selector
            
            # Pass necessary references to core systems if missing
            if self.reflection_engine and not self.reflection_engine.emotional_core and self.emotional_core:
                self.reflection_engine.emotional_core = self.emotional_core
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("request_reflection", self._handle_reflection_request)
                self.event_bus.subscribe("process_stimulus", self._handle_stimulus)
                self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
                self.event_bus.subscribe("user_feedback", self._handle_user_feedback)
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("goal_updated", self._handle_goal_updated)
                self.event_bus.subscribe("memory_added", self._handle_memory_added)
                self._subscribed = True
            
            # Initialize temporal perception for active users
            if self.temporal_perception:
                for user_id in self.active_sessions.keys():
                    # Initialize temporal perception if not already done
                    if user_id not in getattr(self.temporal_perception, "user_systems", {}):
                        await self._initialize_temporal_perception(user_id)
            
            self._initialized = True
            logger.info("AutonomousCognitiveBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing AutonomousCognitiveBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="AutonomousCognitive")
    async def generate_reflection(self, 
                              memories: List[Dict[str, Any]], 
                              topic: Optional[str] = None,
                              emotional_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a reflection using the reflection engine.
        
        Args:
            memories: List of memories to reflect upon
            topic: Optional topic for the reflection
            emotional_state: Optional emotional state to consider
            
        Returns:
            Generated reflection with metadata
        """
        if not self.reflection_engine:
            return {"status": "error", "message": "Reflection engine not available"}
        
        self.cognitive_metrics["reflection_request_count"] += 1
        start_time = datetime.datetime.now()
        
        try:
            # Get current emotional state if not provided
            if not emotional_state and self.emotional_core:
                if hasattr(self.emotional_core, "get_formatted_emotional_state"):
                    emotional_state = self.emotional_core.get_formatted_emotional_state()
                elif hasattr(self.emotional_core, "get_emotional_state"):
                    emotional_state = self.emotional_core.get_emotional_state()
            
            # Create a neurochemical state if possible
            neurochemical_state = None
            if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state"):
                # Use actual neurochemical state if available
                neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
            elif hasattr(self.emotional_core, "neurochemicals"):
                neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
            
            # Generate the reflection
            reflection_text, confidence = await self.reflection_engine.generate_reflection(
                memories=memories,
                topic=topic,
                neurochemical_state=neurochemical_state
            )
            
            # Calculate processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Track success metrics
            if reflection_text and confidence > 0.3:
                # Count as success
                prev_count = self.cognitive_metrics["reflection_request_count"] - 1
                prev_rate = self.cognitive_metrics["reflection_success_rate"]
                new_rate = (prev_count * prev_rate + 1) / self.cognitive_metrics["reflection_request_count"]
                self.cognitive_metrics["reflection_success_rate"] = new_rate
            else:
                # Count as failure
                prev_count = self.cognitive_metrics["reflection_request_count"] - 1
                prev_rate = self.cognitive_metrics["reflection_success_rate"]
                new_rate = (prev_count * prev_rate) / self.cognitive_metrics["reflection_request_count"]
                self.cognitive_metrics["reflection_success_rate"] = new_rate
            
            # Cache reflection
            reflection_data = {
                "text": reflection_text,
                "confidence": confidence,
                "topic": topic,
                "timestamp": datetime.datetime.now().isoformat(),
                "memory_count": len(memories),
                "processing_time": processing_time
            }
            
            self.recent_reflections.append(reflection_data)
            if len(self.recent_reflections) > 10:
                self.recent_reflections.pop(0)
            
            # Fire event for reflection created
            event = Event(
                event_type="reflection_created",
                source="autonomous_cognitive_bridge",
                data={
                    "reflection_text": reflection_text,
                    "confidence": confidence,
                    "topic": topic,
                    "memory_count": len(memories)
                }
            )
            await self.event_bus.publish(event)
            
            return {
                "status": "success",
                "reflection": reflection_text,
                "confidence": confidence,
                "processing_time": processing_time,
                "topic": topic,
                "memory_count": len(memories)
            }
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            
            # Update success metrics for failure
            prev_count = self.cognitive_metrics["reflection_request_count"] - 1
            prev_rate = self.cognitive_metrics["reflection_success_rate"]
            new_rate = (prev_count * prev_rate) / self.cognitive_metrics["reflection_request_count"]
            self.cognitive_metrics["reflection_success_rate"] = new_rate
            
            return {
                "status": "error",
                "message": str(e),
                "reflection": "I was unable to form a clear reflection at this time."
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="AutonomousCognitive")
    async def process_stimulus_fast(self, 
                               stimulus: Dict[str, Any],
                               domain: str = None,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a stimulus through the reflexive system for rapid response.
        
        Args:
            stimulus: The stimulus data
            domain: Optional domain to limit reflex patterns
            context: Additional context information
            
        Returns:
            Rapid reflexive response
        """
        if not self.reflexive_system:
            return {"status": "error", "message": "Reflexive system not available"}
        
        self.cognitive_metrics["reflexive_request_count"] += 1
        start_time = datetime.datetime.now()
        
        try:
            # Check if this should use reflexes or deliberate thinking
            should_use_reflex, confidence = self.reflexive_system.decision_system.should_use_reflex(
                stimulus, context=context, domain=domain
            )
            
            if not should_use_reflex:
                return {
                    "status": "delegated",
                    "reason": "deliberate_thinking_required",
                    "confidence": confidence
                }
            
            # Process through reflexive system
            result = await self.reflexive_system.process_stimulus_fast(
                stimulus=stimulus,
                domain=domain,
                context=context
            )
            
            # Calculate response time
            response_time_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            # Update metrics
            total_responses = self.cognitive_metrics["reflexive_request_count"]
            prev_avg = self.cognitive_metrics["reflexive_response_time_ms"]
            
            if total_responses == 1:
                self.cognitive_metrics["reflexive_response_time_ms"] = response_time_ms
            else:
                new_avg = ((total_responses - 1) * prev_avg + response_time_ms) / total_responses
                self.cognitive_metrics["reflexive_response_time_ms"] = new_avg
            
            # Cache result if successful
            if result.get("success", False):
                reflex_data = {
                    "pattern_name": result.get("pattern_name", "unknown"),
                    "reaction_time_ms": result.get("reaction_time_ms", response_time_ms),
                    "domain": domain,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "success": True
                }
                
                self.recent_reflexes.append(reflex_data)
                if len(self.recent_reflexes) > 10:
                    self.recent_reflexes.pop(0)
            
            # Add response timing to result
            result["total_response_time_ms"] = response_time_ms
            
            # Fire event for reflexive response
            event = Event(
                event_type="reflexive_response",
                source="autonomous_cognitive_bridge",
                data={
                    "stimulus_type": "fast",
                    "domain": domain,
                    "response_time_ms": response_time_ms,
                    "success": result.get("success", False),
                    "pattern_used": result.get("pattern_name")
                }
            )
            await self.event_bus.publish(event)
            
            return result
        except Exception as e:
            logger.error(f"Error processing stimulus through reflexive system: {e}")
            response_time_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "status": "error",
                "message": str(e),
                "reaction_time_ms": response_time_ms
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="AutonomousCognitive")
    async def update_relationship(self, 
                              user_id: str,
                              interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update relationship state based on an interaction.
        
        Args:
            user_id: User ID
            interaction_data: Data about the interaction
            
        Returns:
            Updated relationship state
        """
        if not self.relationship_manager:
            return {"status": "error", "message": "Relationship manager not available"}
        
        self.cognitive_metrics["relationship_update_count"] += 1
        
        try:
            # Process temporal perception first if available
            if self.temporal_perception:
                temporal_system = await self._get_user_temporal_system(user_id)
                if temporal_system:
                    temporal_result = await temporal_system.on_interaction_start()
                    
                    # Add temporal data to interaction context
                    if "context" not in interaction_data:
                        interaction_data["context"] = {}
                    
                    interaction_data["context"]["temporal"] = {
                        "time_since_last": temporal_result.get("time_since_last_interaction", 0),
                        "time_category": temporal_result.get("time_category", "unknown"),
                        "session_duration": temporal_result.get("session_duration", 0)
                    }
                    
                    # Add time expressions if available
                    if "time_expression" in temporal_result.get("perception_state", {}):
                        expression = temporal_result["perception_state"]["time_expression"]
                        if "emotional_context" not in interaction_data:
                            interaction_data["emotional_context"] = {}
                        
                        interaction_data["emotional_context"]["temporal_expression"] = expression
            
            # Update the relationship
            result = await self.relationship_manager.update_relationship_on_interaction(
                user_id=user_id,
                interaction_data=interaction_data
            )
            
            # Get current relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            
            # Cache update
            update_data = {
                "user_id": user_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "trust_impact": result.get("trust_impact", 0),
                "intimacy_impact": result.get("intimacy_impact", 0),
                "current_trust": result.get("trust", relationship.trust if relationship else 0)
            }
            
            self.recent_relationship_updates.append(update_data)
            if len(self.recent_relationship_updates) > 10:
                self.recent_relationship_updates.pop(0)
            
            # Ensure user is in active sessions
            if user_id not in self.active_sessions:
                self.active_sessions[user_id] = {
                    "first_interaction": datetime.datetime.now().isoformat(),
                    "last_interaction": datetime.datetime.now().isoformat(),
                    "interaction_count": 1
                }
            else:
                self.active_sessions[user_id]["last_interaction"] = datetime.datetime.now().isoformat()
                self.active_sessions[user_id]["interaction_count"] = \
                    self.active_sessions[user_id].get("interaction_count", 0) + 1
            
            # If ToM is available, update ToM with relationship insights
            if self.theory_of_mind and hasattr(self.theory_of_mind, "update_user_model"):
                # Extract relevant relationship data for ToM
                tom_update = {
                    "trust_level": relationship.trust if relationship else 0,
                    "familiarity": relationship.familiarity if relationship else 0,
                    "relationship_update": {
                        "trust_impact": result.get("trust_impact", 0),
                        "intimacy_impact": result.get("intimacy_impact", 0),
                        "dominance_impact": result.get("dominance_impact", 0)
                    }
                }
                
                # If we have inferred traits, include them
                if relationship and hasattr(relationship, "inferred_user_traits"):
                    tom_update["inferred_traits"] = relationship.inferred_user_traits
                
                # Update ToM
                await self.theory_of_mind.update_user_model(user_id, tom_update)
            
            # Fire event for relationship update
            event = Event(
                event_type="relationship_updated",
                source="autonomous_cognitive_bridge",
                data={
                    "user_id": user_id,
                    "trust": relationship.trust if relationship else 0,
                    "familiarity": relationship.familiarity if relationship else 0,
                    "intimacy": relationship.intimacy if relationship else 0,
                    "trust_impact": result.get("trust_impact", 0)
                }
            )
            await self.event_bus.publish(event)
            
            return result
        except Exception as e:
            logger.error(f"Error updating relationship for user {user_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "user_id": user_id
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="AutonomousCognitive")
    async def process_user_interaction(self, 
                                   user_id: str,
                                   content: str,
                                   emotional_context: Optional[Dict[str, Any]] = None,
                                   interaction_type: str = "conversation") -> Dict[str, Any]:
        """
        Process a user interaction through all cognitive systems.
        
        Args:
            user_id: User ID
            content: Interaction content
            emotional_context: Optional emotional context
            interaction_type: Type of interaction
            
        Returns:
            Cognitive processing results
        """
        results = {
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "relationship_updated": False,
            "reflexive_processing": False,
            "reflection_generated": False,
            "temporal_processing": False
        }
        
        try:
            # 1. Process through temporal perception if available
            if self.temporal_perception:
                temporal_system = await self._get_user_temporal_system(user_id)
                if temporal_system:
                    temporal_result = await temporal_system.on_interaction_start()
                    results["temporal_processing"] = True
                    results["temporal_result"] = {
                        "time_category": temporal_result.get("time_category"),
                        "time_since_last": temporal_result.get("time_since_last_interaction")
                    }
            
            # 2. Check if this should be processed reflexively
            if self.reflexive_system:
                # Create a stimulus from the interaction
                stimulus = {
                    "type": "user_input",
                    "content": content,
                    "user_id": user_id,
                    "interaction_type": interaction_type
                }
                
                # Add emotional context if available
                if emotional_context:
                    stimulus["emotional_context"] = emotional_context
                
                # Check if should process reflexively
                should_use_reflex, confidence = self.reflexive_system.decision_system.should_use_reflex(
                    stimulus, context={"user_id": user_id}, domain="conversation"
                )
                
                if should_use_reflex and confidence > 0.7:
                    # Process reflexively
                    reflex_result = await self.process_stimulus_fast(
                        stimulus=stimulus,
                        domain="conversation",
                        context={"user_id": user_id}
                    )
                    
                    results["reflexive_processing"] = True
                    results["reflexive_result"] = {
                        "success": reflex_result.get("success", False),
                        "reaction_time_ms": reflex_result.get("reaction_time_ms", 0),
                        "pattern_used": reflex_result.get("pattern_name")
                    }
            
            # 3. Update relationship
            interaction_data = {
                "interaction_type": interaction_type,
                "content": content,
                "emotional_context": emotional_context or {},
                "summary": f"User {user_id} {interaction_type}: {content[:30]}..."
            }
            
            relationship_result = await self.update_relationship(user_id, interaction_data)
            
            results["relationship_updated"] = relationship_result.get("status") == "success"
            results["relationship_result"] = {
                "trust": relationship_result.get("trust", 0),
                "trust_impact": relationship_result.get("trust_impact", 0)
            }
            
            # 4. Generate a reflection if appropriate
            # Only do this occasionally or for significant interactions
            should_reflect = False
            
            # Check if we should generate a reflection based on interaction count
            if user_id in self.active_sessions:
                interaction_count = self.active_sessions[user_id].get("interaction_count", 0)
                # Generate reflection every 5 interactions or if something significant happened
                if (interaction_count % 5 == 0 or 
                    abs(relationship_result.get("trust_impact", 0)) > 0.1):
                    should_reflect = True
            
            if should_reflect and self.reflection_engine and self.memory_orchestrator:
                # Get recent memories for this user
                try:
                    memories = await self.memory_orchestrator.retrieve_memories(
                        query=f"user:{user_id}",
                        limit=5,
                        recency_bias=0.7
                    )
                    
                    if memories and len(memories) >= 2:
                        # Generate reflection
                        reflection_result = await self.generate_reflection(
                            memories=memories,
                            topic=f"interaction with {user_id}",
                            emotional_state=emotional_context
                        )
                        
                        results["reflection_generated"] = reflection_result.get("status") == "success"
                        results["reflection"] = reflection_result.get("reflection")
                except Exception as e:
                    logger.error(f"Error generating reflection for user {user_id}: {e}")
            
            # Track this interaction in active sessions
            if user_id not in self.active_sessions:
                self.active_sessions[user_id] = {
                    "first_interaction": datetime.datetime.now().isoformat(),
                    "last_interaction": datetime.datetime.now().isoformat(),
                    "interaction_count": 1
                }
            else:
                self.active_sessions[user_id]["last_interaction"] = datetime.datetime.now().isoformat()
                self.active_sessions[user_id]["interaction_count"] = \
                    self.active_sessions[user_id].get("interaction_count", 0) + 1
            
            # Complete temporal processing
            if self.temporal_perception:
                temporal_system = await self._get_user_temporal_system(user_id)
                if temporal_system:
                    await temporal_system.on_interaction_end()
            
            return results
        except Exception as e:
            logger.error(f"Error processing user interaction for user {user_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "user_id": user_id
            }
    
    async def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get metrics about cognitive systems."""
        metrics = self.cognitive_metrics.copy()
        
        # Add reflexive system metrics
        if self.reflexive_system:
            try:
                reflex_stats = await self.reflexive_system.get_reflexive_stats()
                metrics["reflexive_patterns"] = reflex_stats.get("total_patterns", 0)
                metrics["reflexive_avg_reaction_time_ms"] = reflex_stats.get("overall_avg_reaction_time_ms", 0)
                metrics["reflexive_active_status"] = reflex_stats.get("active_status", False)
            except Exception as e:
                logger.error(f"Error getting reflexive stats: {e}")
        
        # Add reflection metrics
        metrics["reflection_recent_count"] = len(self.recent_reflections)
        
        # Add relationship metrics
        metrics["relationship_active_users"] = len(self.active_sessions)
        metrics["relationship_updates"] = len(self.recent_relationship_updates)
        
        # Add temporal perception metrics if available
        if self.temporal_perception:
            has_temporal_users = hasattr(self.temporal_perception, "user_systems")
            metrics["temporal_perception_users"] = len(getattr(self.temporal_perception, "user_systems", {})) \
                if has_temporal_users else 0
        
        return metrics
    
    async def get_relationship_summary(self, user_id: str) -> str:
        """Get a summary of the relationship with a user."""
        if not self.relationship_manager:
            return "Relationship data not available."
        
        try:
            return await self.relationship_manager.get_relationship_summary(user_id)
        except Exception as e:
            logger.error(f"Error getting relationship summary: {e}")
            return f"Unable to retrieve relationship summary: {str(e)}"
    
    async def _get_user_temporal_system(self, user_id: str) -> Optional[TemporalPerceptionSystem]:
        """Get or create a temporal system for a user."""
        if not self.temporal_perception:
            return None
        
        # Check if this temporal perception system stores per-user systems
        has_user_systems = hasattr(self.temporal_perception, "user_systems")
        
        if has_user_systems:
            user_systems = self.temporal_perception.user_systems
            
            # Create system if it doesn't exist
            if user_id not in user_systems:
                return await self._initialize_temporal_perception(user_id)
            
            return user_systems[user_id]
        else:
            # Assume the temporal perception itself is the system
            return self.temporal_perception
    
    async def _initialize_temporal_perception(self, user_id: str) -> Optional[TemporalPerceptionSystem]:
        """Initialize temporal perception for a user."""
        if not self.temporal_perception:
            return None
        
        try:
            # Check if this class creates per-user systems
            if hasattr(self.temporal_perception, "create_user_system"):
                # Get first interaction time if available
                first_interaction = None
                if user_id in self.active_sessions:
                    first_interaction = self.active_sessions[user_id].get("first_interaction")
                
                # Create and initialize the system
                user_system = await self.temporal_perception.create_user_system(
                    user_id=user_id,
                    first_interaction=first_interaction,
                    brain_context=self.brain
                )
                
                return user_system
            elif hasattr(self.temporal_perception, "initialize"):
                # Initialize the system itself
                first_interaction = None
                if user_id in self.active_sessions:
                    first_interaction = self.active_sessions[user_id].get("first_interaction")
                
                await self.temporal_perception.initialize(
                    brain_context=self.brain,
                    first_interaction_timestamp=first_interaction
                )
                
                return self.temporal_perception
                
            return None
        except Exception as e:
            logger.error(f"Error initializing temporal perception for user {user_id}: {e}")
            return None
    
    async def _handle_reflection_request(self, event: Event) -> None:
        """
        Handle reflection request events.
        
        Args:
            event: Reflection request event
        """
        try:
            # Extract data
            memories = event.data.get("memories", [])
            topic = event.data.get("topic")
            request_id = event.data.get("request_id")
            
            if not memories:
                logger.warning("Reflection request received with no memories")
                return
            
            # Generate reflection
            result = await self.generate_reflection(memories, topic)
            
            # Send response event
            response = Event(
                event_type="reflection_response",
                source="autonomous_cognitive_bridge",
                data={
                    "reflection": result.get("reflection"),
                    "confidence": result.get("confidence", 0),
                    "request_id": request_id,
                    "status": result.get("status", "error"),
                    "topic": topic
                }
            )
            await self.event_bus.publish(response)
        except Exception as e:
            logger.error(f"Error handling reflection request: {e}")
    
    async def _handle_stimulus(self, event: Event) -> None:
        """
        Handle stimulus processing events.
        
        Args:
            event: Stimulus event
        """
        try:
            # Extract data
            stimulus = event.data.get("stimulus")
            domain = event.data.get("domain")
            context = event.data.get("context")
            request_id = event.data.get("request_id")
            
            if not stimulus:
                logger.warning("Stimulus processing request received with no stimulus data")
                return
            
            # Process stimulus
            result = await self.process_stimulus_fast(stimulus, domain, context)
            
            # Send response event
            response = Event(
                event_type="stimulus_response",
                source="autonomous_cognitive_bridge",
                data={
                    "result": result,
                    "request_id": request_id,
                    "status": result.get("status", "error"),
                    "reaction_time_ms": result.get("reaction_time_ms", 0)
                }
            )
            await self.event_bus.publish(response)
        except Exception as e:
            logger.error(f"Error handling stimulus: {e}")
    
    async def _handle_user_interaction(self, event: Event) -> None:
        """
        Handle user interaction events.
        
        Args:
            event: User interaction event
        """
        try:
            # Extract data
            user_id = event.data.get("user_id")
            content = event.data.get("content")
            input_type = event.data.get("input_type", "conversation")
            emotional_analysis = event.data.get("emotional_analysis")
            
            if not user_id or not content:
                return
            
            # Process the interaction
            asyncio.create_task(
                self.process_user_interaction(
                    user_id=user_id,
                    content=content,
                    emotional_context=emotional_analysis,
                    interaction_type=input_type
                )
            )
        except Exception as e:
            logger.error(f"Error handling user interaction: {e}")
    
    async def _handle_user_feedback(self, event: Event) -> None:
        """
        Handle user feedback events.
        
        Args:
            event: User feedback event
        """
        try:
            # Extract data
            user_id = event.data.get("user_id")
            feedback_type = event.data.get("type")
            rating = event.data.get("rating")
            
            if not user_id or not feedback_type:
                return
            
            # Update relationship with feedback
            interaction_data = {
                "interaction_type": "feedback",
                "user_feedback": {
                    "type": feedback_type,
                    "rating": rating
                },
                "summary": f"User {user_id} provided {feedback_type} feedback with rating {rating}"
            }
            
            asyncio.create_task(self.update_relationship(user_id, interaction_data))
        except Exception as e:
            logger.error(f"Error handling user feedback: {e}")
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional state change events.
        
        Args:
            event: Emotional state change event
        """
        # Used mainly for future expansions
        pass
    
    async def _handle_goal_updated(self, event: Event) -> None:
        """
        Handle goal updated events.
        
        Args:
            event: Goal updated event
        """
        try:
            # Extract data
            goal_id = event.data.get("goal_id")
            status = event.data.get("status")
            metadata = event.data.get("metadata", {})
            
            # Check if this is a user-related goal
            user_id = metadata.get("user_id")
            if not user_id:
                return
            
            # Update relationship with goal outcome
            if status in ["completed", "failed"]:
                interaction_data = {
                    "interaction_type": "goal_outcome",
                    "goal_outcome": {
                        "goal_id": goal_id,
                        "status": status,
                        "priority": metadata.get("priority", 0.5)
                    },
                    "summary": f"Goal {goal_id} for user {user_id} {status}"
                }
                
                asyncio.create_task(self.update_relationship(user_id, interaction_data))
        except Exception as e:
            logger.error(f"Error handling goal updated: {e}")
    
    async def _handle_memory_added(self, event: Event) -> None:
        """
        Handle memory added events.
        
        Args:
            event: Memory added event
        """
        try:
            # Extract memory data
            memory_id = event.data.get("memory_id")
            user_id = event.data.get("user_id")
            significance = event.data.get("significance", 0)
            
            # Skip if not user-related memory or not significant
            if not user_id or significance < 7:
                return
            
            # Update relationship with key memory
            interaction_data = {
                "interaction_type": "key_memory",
                "memory_id": memory_id,
                "significance": significance,
                "summary": f"Significant memory {memory_id} created for user {user_id}"
            }
            
            asyncio.create_task(self.update_relationship(user_id, interaction_data))
        except Exception as e:
            logger.error(f"Error handling memory added: {e}")

# Function to create the bridge
def create_autonomous_cognitive_bridge(nyx_brain):
    """Create an autonomous cognitive bridge for the given brain."""
    return AutonomousCognitiveBridge(nyx_brain=nyx_brain)
