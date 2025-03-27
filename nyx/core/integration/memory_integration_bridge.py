# nyx/core/integration/memory_integration_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class MemoryIntegrationBridge:
    """
    Central integration bridge for the memory system.
    
    Provides unified access to memory operations across modules and ensures
    that memory events are properly propagated throughout the system.
    
    Key functions:
    1. Manages memory lifecycle events (creation, retrieval, update)
    2. Coordinates memory maintenance across integration boundaries
    3. Provides cross-module memory access with appropriate context
    4. Enables experience retrieval for other modules
    """
    
    def __init__(self, 
                memory_core=None,
                emotional_core=None,
                identity_evolution=None,
                knowledge_core=None,
                attention_system=None):
        """Initialize the memory integration bridge."""
        self.memory_core = memory_core
        self.emotional_core = emotional_core
        self.identity_evolution = identity_evolution
        self.knowledge_core = knowledge_core
        self.attention_system = attention_system
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.memory_significance_threshold = 0.6  # Threshold for broadcasting memory events
        self.memory_maintenance_interval = 3600  # Seconds between maintenance cycles
        self.attention_boost_factor = 0.3  # How much attention boosts memory significance
        
        # Tracking variables
        self.pending_maintenance = False
        self.last_maintenance = datetime.datetime.now()
        self._subscribed = False
        
        # Memory event cache
        self.recent_memory_events = []  # List of recent memory events for analysis
        self.max_event_history = 100
        
        logger.info("MemoryIntegrationBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("memory_added", self._handle_memory_added)
                self.event_bus.subscribe("memory_retrieved", self._handle_memory_retrieved)
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("attention_focus_changed", self._handle_attention_change)
                self._subscribed = True
            
            # Schedule initial maintenance
            asyncio.create_task(self._schedule_maintenance())
            
            logger.info("MemoryIntegrationBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing MemoryIntegrationBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="MemoryIntegration")
    async def add_memory_with_context(self, 
                                   memory_text: str,
                                   memory_type: str = "observation",
                                   memory_scope: str = "game",
                                   significance: int = 5,
                                   tags: List[str] = None,
                                   metadata: Dict[str, Any] = None,
                                   emotional_context: Dict[str, Any] = None,
                                   attention_level: float = None) -> Dict[str, Any]:
        """
        Add a memory with integrated system context.
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory
            memory_scope: Scope of memory
            significance: Importance level (1-10)
            tags: Optional tags for categorization
            metadata: Additional metadata
            emotional_context: Emotional context override
            attention_level: Attention level override
            
        Returns:
            Memory creation result with context
        """
        if not self.memory_core:
            return {"status": "error", "message": "Memory core not available"}
        
        try:
            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}
                
            # Add timestamp if not provided
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.datetime.now().isoformat()
            
            # Incorporate emotional context if available
            if emotional_context is None and self.emotional_core:
                try:
                    # Get current emotional state
                    if hasattr(self.emotional_core, 'get_emotional_state_matrix'):
                        emotional_state = await self.emotional_core.get_emotional_state_matrix()
                        
                        # Format emotional context
                        emotional_context = {
                            "primary_emotion": emotional_state.get("primary_emotion", {}).get("name", "neutral") 
                                if isinstance(emotional_state.get("primary_emotion"), dict) 
                                else emotional_state.get("primary_emotion", "neutral"),
                            "primary_intensity": emotional_state.get("primary_emotion", {}).get("intensity", 0.5)
                                if isinstance(emotional_state.get("primary_emotion"), dict)
                                else 0.5,
                            "valence": emotional_state.get("valence", 0.0),
                            "arousal": emotional_state.get("arousal", 0.5)
                        }
                        
                except Exception as e:
                    logger.warning(f"Error retrieving emotional context: {e}")
            
            # Add emotional context to metadata
            if emotional_context:
                metadata["emotional_context"] = emotional_context
            
            # Adjust significance based on attention
            adjusted_significance = significance
            
            # Get current attention focus if not provided
            if attention_level is None and self.attention_system:
                try:
                    attention_state = await self.attention_system.get_current_focus()
                    if attention_state and "focus" in attention_state:
                        attention_level = attention_state["focus"].get("attention_level", 0.5)
                except Exception as e:
                    logger.warning(f"Error retrieving attention state: {e}")
            
            # Boost significance based on attention
            if attention_level is not None:
                attention_boost = int(attention_level * self.attention_boost_factor * 10)
                adjusted_significance = min(10, significance + attention_boost)
            
            # Add memory through memory core
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type=memory_type,
                memory_scope=memory_scope,
                significance=adjusted_significance,
                tags=tags or [],
                metadata=metadata
            )
            
            # Broadcast memory_added event if significant enough
            if adjusted_significance >= self.memory_significance_threshold * 10:
                # Only broadcast for significant memories
                memory = await self.memory_core.get_memory(memory_id)
                
                if memory:
                    event = Event(
                        event_type="memory_added",
                        source="memory_integration_bridge",
                        data={
                            "memory_id": memory_id,
                            "memory_text": memory_text,
                            "memory_type": memory_type,
                            "significance": adjusted_significance,
                            "tags": tags or []
                        }
                    )
                    await self.event_bus.publish(event)
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "original_significance": significance,
                "adjusted_significance": adjusted_significance,
                "emotional_context": emotional_context,
                "attention_level": attention_level
            }
        except Exception as e:
            logger.error(f"Error adding memory with context: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="MemoryIntegration")
    async def retrieve_memories_with_context(self, 
                                         query: str,
                                         memory_types: List[str] = None,
                                         limit: int = 5,
                                         min_significance: int = 3,
                                         current_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve memories with integrated system context.
        
        Args:
            query: Search query
            memory_types: Types of memories to retrieve
            limit: Maximum number of memories to return
            min_significance: Minimum significance level
            current_context: Optional additional context
            
        Returns:
            Memory retrieval results with context
        """
        if not self.memory_core:
            return {"status": "error", "message": "Memory core not available"}
        
        try:
            # Build context from current system state
            retrieval_context = current_context or {}
            
            # Add emotional context if available
            if "emotional_state" not in retrieval_context and self.emotional_core:
                try:
                    # Get current emotional state
                    if hasattr(self.emotional_core, 'get_emotional_state_matrix'):
                        emotional_state = await self.emotional_core.get_emotional_state_matrix()
                        retrieval_context["emotional_state"] = emotional_state
                except Exception as e:
                    logger.warning(f"Error retrieving emotional context for memory retrieval: {e}")
            
            # Add identity context if available
            if "identity_profile" not in retrieval_context and self.identity_evolution:
                try:
                    # Get identity profile
                    identity_profile = await self.identity_evolution.get_identity_profile()
                    retrieval_context["identity_profile"] = identity_profile
                except Exception as e:
                    logger.warning(f"Error retrieving identity context for memory retrieval: {e}")
            
            # Get entities from current context
            entities = []
            if "entities" in retrieval_context:
                entities = retrieval_context["entities"]
            
            # Retrieve memories
            memories = await self.memory_core.retrieve_memories(
                query=query,
                memory_types=memory_types,
                limit=limit,
                min_significance=min_significance,
                include_archived=False,
                entities=entities,
                emotional_state=retrieval_context.get("emotional_state")
            )
            
            # Add integration metadata and track retrievals
            enhanced_memories = []
            memory_ids = []
            
            for memory in memories:
                memory_id = memory.get("id")
                memory_ids.append(memory_id)
                
                # Enhance with integration data
                enhanced_memory = memory.copy()
                enhanced_memory["retrieved_with_context"] = True
                enhanced_memory["retrieval_query"] = query
                
                enhanced_memories.append(enhanced_memory)
                
                # Broadcast memory_retrieved event
                event = Event(
                    event_type="memory_retrieved",
                    source="memory_integration_bridge",
                    data={
                        "memory_id": memory_id,
                        "memory": memory,
                        "query": query
                    }
                )
                await self.event_bus.publish(event)
            
            return {
                "status": "success",
                "memories": enhanced_memories,
                "query": query,
                "context_used": retrieval_context,
                "count": len(enhanced_memories)
            }
        except Exception as e:
            logger.error(f"Error retrieving memories with context: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="MemoryIntegration")
    async def create_integrated_reflection(self, 
                                       topic: str,
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a reflection with integrated emotional and identity context.
        
        Args:
            topic: Topic to reflect on
            context: Optional additional context
            
        Returns:
            Reflection creation result
        """
        if not self.memory_core:
            return {"status": "error", "message": "Memory core not available"}
        
        try:
            # Get reflection data from memory core
            reflection_result = await self.memory_core.create_reflection_from_memories(topic)
            
            if "reflection_id" not in reflection_result or not reflection_result["reflection_id"]:
                return reflection_result
            
            # Enhance reflection with identity context if available
            if self.identity_evolution:
                try:
                    # Get key identity traits
                    identity_profile = await self.identity_evolution.get_identity_profile()
                    
                    if identity_profile and "traits" in identity_profile:
                        # Update reflection with identity context
                        reflection_id = reflection_result["reflection_id"]
                        memory = await self.memory_core.get_memory(reflection_id)
                        
                        if memory:
                            metadata = memory.get("metadata", {})
                            metadata["identity_context"] = {
                                "relevant_traits": list(identity_profile["traits"].keys())[:5],
                                "integration_timestamp": datetime.datetime.now().isoformat()
                            }
                            
                            await self.memory_core.update_memory(
                                reflection_id, {"metadata": metadata}
                            )
                            
                            # Update reflection result with identity info
                            reflection_result["identity_integrated"] = True
                except Exception as e:
                    logger.warning(f"Error integrating identity with reflection: {e}")
            
            # Broadcast reflection creation event
            event = Event(
                event_type="reflection_created",
                source="memory_integration_bridge",
                data={
                    "reflection_id": reflection_result.get("reflection_id"),
                    "topic": topic,
                    "confidence": reflection_result.get("confidence", 0.5)
                }
            )
            await self.event_bus.publish(event)
            
            return reflection_result
        except Exception as e:
            logger.error(f"Error creating integrated reflection: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="MemoryIntegration")
    async def get_experiences_for_context(self, 
                                      current_context: Dict[str, Any],
                                      limit: int = 2) -> Dict[str, Any]:
        """
        Get relevant experiences for the current interaction context.
        
        Args:
            current_context: Current interaction context
            limit: Maximum number of experiences to return
            
        Returns:
            Relevant experiences with conversational formatting
        """
        if not self.memory_core:
            return {"status": "error", "message": "Memory core not available"}
        
        try:
            # Extract context information
            query = current_context.get("query", "")
            scenario_type = current_context.get("scenario_type", "")
            emotional_state = current_context.get("emotional_state")
            entities = current_context.get("entities", [])
            
            # Get relevant experiences
            experiences = await self.memory_core.retrieve_relevant_experiences(
                query=query,
                scenario_type=scenario_type,
                emotional_state=emotional_state,
                entities=entities,
                limit=limit,
                min_relevance=0.6
            )
            
            if not experiences:
                return {
                    "status": "no_experiences",
                    "message": "No relevant experiences found"
                }
            
            # Format experiences for conversation
            formatted_experiences = []
            for exp in experiences:
                recall_result = await self.memory_core.generate_conversational_recall(
                    experience=exp,
                    context=current_context
                )
                
                formatted_experiences.append({
                    "experience_id": exp.get("id"),
                    "recall_text": recall_result.get("recall_text"),
                    "tone": recall_result.get("tone", "standard"),
                    "confidence": recall_result.get("confidence", 0.5),
                    "relevance": exp.get("relevance_score", 0.5)
                })
            
            return {
                "status": "success",
                "experiences": formatted_experiences,
                "count": len(formatted_experiences),
                "query": query
            }
        except Exception as e:
            logger.error(f"Error getting experiences for context: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="MemoryIntegration")
    async def run_integrated_maintenance(self) -> Dict[str, Any]:
        """
        Run memory maintenance with cross-module integration.
        
        Returns:
            Maintenance results
        """
        if not self.memory_core:
            return {"status": "error", "message": "Memory core not available"}
        
        try:
            self.last_maintenance = datetime.datetime.now()
            self.pending_maintenance = False
            
            # Run basic memory maintenance
            maintenance_result = await self.memory_core.run_maintenance()
            
            # Check for schema creation opportunities
            if self.knowledge_core:
                try:
                    # Look for patterns that could become knowledge
                    schemas = await self.memory_core.detect_schema_from_memories()
                    
                    if schemas and "schema_id" in schemas:
                        # Integrate schema with knowledge if possible
                        if hasattr(self.knowledge_core, "add_pattern_from_schema"):
                            await self.knowledge_core.add_pattern_from_schema(schemas)
                except Exception as e:
                    logger.warning(f"Error integrating schemas with knowledge: {e}")
            
            # Check for identity-relevant memories
            if self.identity_evolution:
                try:
                    # Check for potential memory crystallization
                    memory_ids = await self._identify_identity_relevant_memories()
                    
                    # Crystallize important memories
                    for memory_id in memory_ids:
                        await self.memory_core.crystallize_memory(
                            memory_id=memory_id,
                            reason="identity_relevance",
                            importance_data={"identity_relevance": True}
                        )
                except Exception as e:
                    logger.warning(f"Error processing identity-relevant memories: {e}")
            
            # Schedule next maintenance
            asyncio.create_task(self._schedule_maintenance())
            
            return {
                "status": "success",
                "maintenance_results": maintenance_result,
                "next_maintenance": (datetime.datetime.now() + 
                                    datetime.timedelta(seconds=self.memory_maintenance_interval)).isoformat()
            }
        except Exception as e:
            logger.error(f"Error running integrated maintenance: {e}")
            self.pending_maintenance = False  # Reset flag on error
            return {"status": "error", "message": str(e)}
    
    async def _identify_identity_relevant_memories(self) -> List[str]:
        """Identify memories that are relevant to identity."""
        # Get recent memories
        recent_memories = await self.memory_core.retrieve_memories(
            query="identity OR personality OR self",
            limit=10,
            memory_types=["observation", "experience", "reflection"]
        )
        
        # Filter for identity relevance
        identity_relevant = []
        for memory in recent_memories:
            # Check if memory has identity relevance
            tags = memory.get("tags", [])
            if any(tag in tags for tag in ["identity", "self", "personality", "formative"]):
                identity_relevant.append(memory["id"])
                continue
                
            # Check memory text for identity relevance
            memory_text = memory.get("memory_text", "").lower()
            if any(phrase in memory_text for phrase in [
                "i am", "my personality", "defines me", "my nature", "who i am"
            ]):
                identity_relevant.append(memory["id"])
        
        return identity_relevant
    
    async def _schedule_maintenance(self) -> None:
        """Schedule memory maintenance to run periodically."""
        # Only schedule if not already pending
        if self.pending_maintenance:
            return
            
        # Calculate time until next maintenance
        now = datetime.datetime.now()
        seconds_since_last = (now - self.last_maintenance).total_seconds()
        
        if seconds_since_last < self.memory_maintenance_interval:
            # Wait until interval has passed
            wait_seconds = self.memory_maintenance_interval - seconds_since_last
            
            # Mark as pending and schedule
            self.pending_maintenance = True
            await asyncio.sleep(wait_seconds)
            
            # Run maintenance if still pending (not canceled)
            if self.pending_maintenance:
                await self.run_integrated_maintenance()
        else:
            # Run immediately
            await self.run_integrated_maintenance()
    
    async def _handle_memory_added(self, event: Event) -> None:
        """
        Handle memory added events.
        
        Args:
            event: Memory added event
        """
        try:
            # Extract event data
            memory_id = event.data.get("memory_id")
            significance = event.data.get("significance", 5)
            
            if not memory_id:
                return
            
            # Track the event
            self.recent_memory_events.append({
                "event_type": "memory_added",
                "memory_id": memory_id,
                "significance": significance,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source
            })
            
            # Trim event history if needed
            if len(self.recent_memory_events) > self.max_event_history:
                self.recent_memory_events = self.recent_memory_events[-self.max_event_history:]
            
            # Check for identity integration for significant memories
            if self.identity_evolution and significance >= 8:
                # Create a task to assess memory importance
                asyncio.create_task(self._assess_memory_for_identity(memory_id))
        except Exception as e:
            logger.error(f"Error handling memory added event: {e}")
    
    async def _assess_memory_for_identity(self, memory_id: str) -> None:
        """Assess if a memory should be integrated with identity."""
        try:
            # Get the memory
            memory = await self.memory_core.get_memory(memory_id)
            
            if not memory:
                return
                
            # Check if this memory might be identity-relevant
            importance = await self.memory_core.assess_memory_importance(memory_id)
            
            if importance.get("important", False):
                # Memory is important - update identity if possible
                if hasattr(self.identity_evolution, "process_memory_for_identity"):
                    await self.identity_evolution.process_memory_for_identity(memory)
                
                # Crystallize the memory
                await self.memory_core.crystallize_memory(
                    memory_id=memory_id,
                    reason="identity_importance",
                    importance_data=importance
                )
        except Exception as e:
            logger.error(f"Error assessing memory for identity: {e}")
    
    async def _handle_memory_retrieved(self, event: Event) -> None:
        """
        Handle memory retrieved events.
        
        Args:
            event: Memory retrieved event
        """
        try:
            # Extract event data
            memory_id = event.data.get("memory_id")
            
            if not memory_id:
                return
            
            # Track the event
            self.recent_memory_events.append({
                "event_type": "memory_retrieved",
                "memory_id": memory_id,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source
            })
            
            # Trim event history if needed
            if len(self.recent_memory_events) > self.max_event_history:
                self.recent_memory_events = self.recent_memory_events[-self.max_event_history:]
            
            # Check if this is a frequently retrieved memory
            retrieval_count = 0
            for mevent in self.recent_memory_events:
                if (mevent["event_type"] == "memory_retrieved" and 
                    mevent["memory_id"] == memory_id):
                    retrieval_count += 1
            
            # If memory is frequently retrieved, consider crystallization
            if retrieval_count >= 3:
                asyncio.create_task(self._check_for_crystallization(memory_id))
        except Exception as e:
            logger.error(f"Error handling memory retrieved event: {e}")
    
    async def _check_for_crystallization(self, memory_id: str) -> None:
        """Check if a memory should be crystallized due to frequent retrieval."""
        try:
            # Check crystallization criteria
            should_crystallize = await self.memory_core.check_for_crystallization(memory_id)
            
            if should_crystallize:
                # Crystallize the memory
                await self.memory_core.crystallize_memory(
                    memory_id=memory_id,
                    reason="frequent_retrieval"
                )
        except Exception as e:
            logger.error(f"Error checking for crystallization: {e}")
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional change events.
        
        Args:
            event: Emotional change event
        """
        # This could trigger memory retrievals based on emotion
        # or affect memory formation priorities
        pass
    
    async def _handle_attention_change(self, event: Event) -> None:
        """
        Handle attention change events.
        
        Args:
            event: Attention change event
        """
        # This could prioritize memory formation for attended content
        pass

# Factory function to create the bridge
def create_memory_integration_bridge(nyx_brain):
    """Create a memory integration bridge for the given brain."""
    return MemoryIntegrationBridge(
        memory_core=nyx_brain.memory_core if hasattr(nyx_brain, "memory_core") else None,
        emotional_core=nyx_brain.emotional_core if hasattr(nyx_brain, "emotional_core") else None,
        identity_evolution=nyx_brain.identity_evolution if hasattr(nyx_brain, "identity_evolution") else None,
        knowledge_core=nyx_brain.knowledge_core if hasattr(nyx_brain, "knowledge_core") else None,
        attention_system=nyx_brain.attention_system if hasattr(nyx_brain, "attention_system") else None
    )
