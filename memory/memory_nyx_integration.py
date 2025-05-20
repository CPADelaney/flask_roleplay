# memory/memory_nyx_integration.py

"""
Integration module between the memory agent and Nyx's central governance system.
This allows memory operations to be governed, tracked, and coordinated by Nyx.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from memory.core import Memory

from agents import trace, Runner

from memory.memory_agent_wrapper import MemoryAgentWrapper

from memory.memory_agent_sdk import (
    create_memory_agent, 
    MemorySystemContext, 
    MemoryInput, 
    MemoryQueryInput, 
    BeliefInput, 
    BeliefsQueryInput,
    MaintenanceInput,
    AnalysisInput
)

from memory.wrapper import MemorySystem
from utils.caching import get_cache, set_cache, delete_cache

from nyx.constants import AgentType, DirectiveType, DirectivePriority

logger = logging.getLogger(__name__)

class MemoryNyxBridge:
    """
    Bridge between the memory agent and Nyx's governance system.
    
    This class ensures that:
    1. Memory operations are governed by Nyx's permissions system
    2. Memory directives from Nyx are passed to the memory agent
    3. Memory operations are tracked and reported to Nyx
    4. NPC agents can access memory functionality through Nyx
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the bridge between memory agent and Nyx.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        from nyx.nyx_governance import NyxUnifiedGovernor
        self.governor = NyxUnifiedGovernor(user_id, conversation_id)
        self.memory_context = MemorySystemContext(user_id, conversation_id)
        self.memory_agent = None
        self.memory_system = None
        self.state_tracker = {
            "last_operation": None,
            "operation_count": 0,
            "error_count": 0,
            "last_error": None,
            "active_transactions": 0
        }
        self.error_history = []
        self.transaction_stack = []

    def _track_error(self, operation: str, error: str):
        """Track errors for analysis and recovery."""
        self.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "error": error
        })
        self.state_tracker["error_count"] += 1
        self.state_tracker["last_error"] = error

    def _track_state_change(self, operation: str, details: Dict[str, Any]):
        """Track state changes for consistency."""
        self.state_tracker["last_operation"] = operation
        self.state_tracker["operation_count"] += 1
        self.state_tracker.update(details)

    async def initialize(self):
        """Initialize the bridge with proper error handling."""
        try:
            # Create the base memory agent using the OpenAI Agents SDK
            base_agent = create_memory_agent(self.user_id, self.conversation_id)
            
            # Wrap it with our compatibility layer
            self.memory_agent = MemoryAgentWrapper(base_agent, self.memory_context)
            
            # Initialize memory system
            self.memory_system = MemorySystem.get_instance(
                self.user_id, self.conversation_id
            )
            self.memory_context.memory_system = self.memory_system
            
            # Register the memory agent with Nyx's governance
            await self.governor.register_agent(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_instance=self.memory_agent,
                agent_id="memory_manager"
            )
            
            # Issue a general directive for memory management
            await self.governor.issue_directive(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                directive_type=DirectiveType.ACTION,
                directive_data={
                    "instruction": "Maintain entity memories and ensure proper consolidation.",
                    "scope": "global"
                },
                priority=DirectivePriority.MEDIUM,
                duration_minutes=24*60  # 24 hours
            )
            
            self._track_state_change("initialization", {"status": "success"})
            logger.info("Memory agent registered with Nyx governance system")
            return self
            
        except Exception as e:
            self._track_error("initialization", str(e))
            logger.error(f"Failed to initialize MemoryNyxBridge: {str(e)}")
            raise

    # In the _process_significant_memory method
    async def _process_significant_memory(self, memory: Memory) -> Dict[str, Any]:
        """Process a significant memory with proper error handling and state tracking."""
        try:
            # Start transaction
            self.transaction_stack.append({
                "type": "memory_processing",
                "memory_id": memory.id,
                "start_time": datetime.now()
            })
            
            # Using the new connection pattern directly here
            from db.connection import get_db_connection_context
            
            async with get_db_connection_context() as conn:
                # Validate memory
                if not self._validate_memory(memory):
                    raise ValueError("Invalid memory format")
                
                # Process memory with emotional context
                emotional_context = await self._get_emotional_context(memory)
                
                # Update memory with emotional context
                memory.emotional_intensity = emotional_context.get("intensity", 0)
                memory.metadata["emotional_context"] = emotional_context
                
                # Propagate memory to related systems
                propagation_results = await self._propagate_memory(memory, conn)
                
                # Track state change
                self._track_state_change("memory_processing", {
                    "memory_id": memory.id,
                    "emotional_intensity": memory.emotional_intensity,
                    "propagation_success": bool(propagation_results)
                })
                
                return {
                    "memory": memory.to_dict(),
                    "emotional_context": emotional_context,
                    "propagation_results": propagation_results
                }
                
        except Exception as e:
            self._track_error("memory_processing", str(e))
            await self._rollback_transaction()
            logger.error(f"Error processing significant memory: {str(e)}")
            raise
            
        finally:
            # Clean up transaction
            if self.transaction_stack:
                self.transaction_stack.pop()

    def _validate_memory(self, memory: Memory) -> bool:
        """Validate memory format and content."""
        if not memory or not memory.text:
            return False
            
        # Check required fields
        required_fields = ["text", "memory_type", "significance"]
        if not all(hasattr(memory, field) for field in required_fields):
            return False
            
        # Validate memory type
        if memory.memory_type not in [t.value for t in MemoryType]:
            return False
            
        # Validate significance
        if not 1 <= memory.significance <= 5:
            return False
            
        return True

    async def _get_emotional_context(self, memory: Memory) -> Dict[str, Any]:
        """Get emotional context for memory with caching."""
        try:
            # Use cache if available
            cache_key = f"emotional_context:{memory.id}"
            cached_context = await get_cache(cache_key)
            if cached_context:
                return cached_context
            
            # Analyze emotional content
            emotional_context = await self.emotional_manager.analyze_emotional_content(memory.text)
            
            # Cache results
            await set_cache(cache_key, emotional_context, ttl=3600)  # 1 hour
            
            return emotional_context
            
        except Exception as e:
            self._track_error("emotional_context", str(e))
            logger.error(f"Error getting emotional context: {str(e)}")
            return {"intensity": 0, "primary_emotion": "neutral"}

    async def _propagate_memory(self, memory: Memory) -> Dict[str, Any]:
        """Propagate memory to related systems with proper error handling."""
        propagation_results = {}
        
        try:
            # Propagate to emotional system
            emotional_result = await self.emotional_manager.process_memory(memory)
            propagation_results["emotional"] = emotional_result
            
            # Propagate to semantic system
            semantic_result = await self.semantic_manager.process_memory(memory)
            propagation_results["semantic"] = semantic_result
            
            # Propagate to reconsolidation system
            reconsolidation_result = await self.reconsolidation_manager.process_memory(memory)
            propagation_results["reconsolidation"] = reconsolidation_result
            
            return propagation_results
            
        except Exception as e:
            self._track_error("memory_propagation", str(e))
            logger.error(f"Error propagating memory: {str(e)}")
            return {}

    async def _rollback_transaction(self):
        """Rollback the current transaction."""
        if not self.transaction_stack:
            return
            
        current_transaction = self.transaction_stack[-1]
        try:
            # Rollback memory changes
            if current_transaction["type"] == "memory_processing":
                await self._rollback_memory_changes(current_transaction["memory_id"])
            
            # Remove from stack
            self.transaction_stack.pop()
            
        except Exception as e:
            logger.error(f"Error during rollback: {str(e)}")

    async def _rollback_memory_changes(self, memory_id: int):
        """Rollback changes to a specific memory."""
        try:
            from db.connection import get_db_connection_context
            
            # Restore memory from backup if available
            backup_key = f"memory_backup:{memory_id}"
            backup = await get_cache(backup_key)
            
            if backup:
                async with get_db_connection_context() as conn:
                    await self.memory_system.restore_memory(memory_id, backup, conn=conn)
            
            # Clear related caches
            await delete_cache(f"emotional_context:{memory_id}")
            await delete_cache(f"semantic_context:{memory_id}")
            
        except Exception as e:
            logger.error(f"Error rolling back memory changes: {str(e)}")

    async def remember(
        self, 
        entity_type: str, 
        entity_id: int, 
        memory_text: str,
        importance: str = "medium", 
        emotional: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a memory with Nyx governance.
        
        Args:
            entity_type: Type of entity ("player", "npc", etc.)
            entity_id: ID of the entity
            memory_text: The memory text to record
            importance: Importance level ("trivial", "low", "medium", "high", "critical")
            emotional: Whether to analyze emotional content
            tags: Optional tags for the memory
        """
        # Check permission with governance system
        action_details = {
            "operation": "remember",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "memory_text": memory_text,
            "importance": importance
        }
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action_type="memory_operation",
            action_details=action_details
        )
        
        if not permission["approved"]:
            logger.warning(f"Memory creation not approved: {permission.get('reasoning', 'Unknown reason')}")
            return {"error": permission.get("reasoning", "Memory creation not permitted by Nyx")}
        
        # If approved, proceed with memory creation
        with trace(workflow_name="Memory Operation - Remember"):
            memory_input = MemoryInput(
                entity_type=entity_type,
                entity_id=entity_id,
                memory_text=memory_text,
                importance=importance,
                emotional=emotional,
                tags=tags
            )
            
            result = await self.memory_agent.remember(
                self.memory_context,
                entity_type=memory_input.entity_type,
                entity_id=memory_input.entity_id,
                memory_text=memory_input.memory_text,
                importance=memory_input.importance,
                emotional=memory_input.emotional,
                tags=memory_input.tags
            )
            
            # Report action back to Nyx
            await self.governor.process_agent_action_report(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                action={
                    "type": "memory_operation",
                    "operation": "remember",
                    "description": f"Created memory for {entity_type} {entity_id}: {memory_text[:50]}..."
                },
                result=result
            )
            
            return result
    
    async def recall(
        self,
        entity_type: str,
        entity_id: int,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Recall memories for an entity with Nyx governance.
        
        Args:
            entity_type: Type of entity ("player", "npc", etc.)
            entity_id: ID of the entity
            query: Optional search query
            context: Current context that might influence recall
            limit: Maximum number of memories to return
        """
        # Check permission with governance system
        action_details = {
            "operation": "recall",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "query": query,
            "limit": limit
        }
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action_type="memory_operation",
            action_details=action_details
        )
        
        if not permission["approved"]:
            logger.warning(f"Memory recall not approved: {permission.get('reasoning', 'Unknown reason')}")
            return {"error": permission.get("reasoning", "Memory recall not permitted by Nyx")}
        
        # If approved, proceed with memory recall
        with trace(workflow_name="Memory Operation - Recall"):
            memory_query = MemoryQueryInput(
                entity_type=entity_type,
                entity_id=entity_id,
                query=query,
                context=context,
                limit=limit
            )
            
        result = await self.memory_agent.recall(
            self.memory_context,
            entity_type=memory_query.entity_type,
            entity_id=memory_query.entity_id,
            query=memory_query.query,
            context_text=memory_query.context,  # Changed to context_text to match the wrapper's parameter name
            limit=memory_query.limit
        )
            
        # Report action back to Nyx
        await self.governor.process_agent_action_report(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action={
                "type": "memory_operation",
                "operation": "recall",
                "description": f"Recalled memories for {entity_type} {entity_id}" +
                             (f" with query: {query}" if query else "")
            },
            result={"memory_count": len(result.get("memories", []))}
        )
        
        return result
    
    async def create_belief(
        self,
        entity_type: str,
        entity_id: int,
        belief_text: str,
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Create a belief for an entity with Nyx governance.
        
        Args:
            entity_type: Type of entity ("player", "npc", etc.)
            entity_id: ID of the entity
            belief_text: The belief statement
            confidence: Confidence in this belief (0.0-1.0)
        """
        # Check permission with governance system
        action_details = {
            "operation": "create_belief",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "belief_text": belief_text,
            "confidence": confidence
        }
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action_type="memory_operation",
            action_details=action_details
        )
        
        if not permission["approved"]:
            logger.warning(f"Belief creation not approved: {permission.get('reasoning', 'Unknown reason')}")
            return {"error": permission.get("reasoning", "Belief creation not permitted by Nyx")}
        
        # If approved, proceed with belief creation
        with trace(workflow_name="Memory Operation - Create Belief"):
            belief_input = BeliefInput(
                entity_type=entity_type,
                entity_id=entity_id,
                belief_text=belief_text,
                confidence=confidence
            )
            
            result = await self.memory_agent.create_belief(
                self.memory_context,
                entity_type=belief_input.entity_type,
                entity_id=belief_input.entity_id,
                belief_text=belief_input.belief_text,
                confidence=belief_input.confidence
            )
            
            # Report action back to Nyx
            await self.governor.process_agent_action_report(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                action={
                    "type": "memory_operation",
                    "operation": "create_belief",
                    "description": f"Created belief for {entity_type} {entity_id}: {belief_text[:50]}..."
                },
                result=result
            )
            
            return result
    
    async def get_beliefs(
        self,
        entity_type: str,
        entity_id: int,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get beliefs for an entity with Nyx governance.
        
        Args:
            entity_type: Type of entity ("player", "npc", etc.)
            entity_id: ID of the entity
            topic: Optional topic filter
        """
        # Check permission with governance system
        action_details = {
            "operation": "get_beliefs",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "topic": topic
        }
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action_type="memory_operation",
            action_details=action_details
        )
        
        if not permission["approved"]:
            logger.warning(f"Belief retrieval not approved: {permission.get('reasoning', 'Unknown reason')}")
            return []
        
        # If approved, proceed with belief retrieval
        with trace(workflow_name="Memory Operation - Get Beliefs"):
            beliefs_query = BeliefsQueryInput(
                entity_type=entity_type,
                entity_id=entity_id,
                topic=topic
            )
            
            result = await self.memory_agent.get_beliefs(
                self.memory_context,
                entity_type=beliefs_query.entity_type,
                entity_id=beliefs_query.entity_id,
                topic=beliefs_query.topic
            )
            
            # Report action back to Nyx
            await self.governor.process_agent_action_report(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                action={
                    "type": "memory_operation",
                    "operation": "get_beliefs",
                    "description": f"Retrieved beliefs for {entity_type} {entity_id}" +
                                 (f" with topic: {topic}" if topic else "")
                },
                result={"belief_count": len(result)}
            )
            
            return result
    
    async def run_maintenance(
        self,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Run memory maintenance with Nyx governance.
        
        Args:
            entity_type: Type of entity ("player", "npc", etc.)
            entity_id: ID of the entity
        """
        # Check permission with governance system
        action_details = {
            "operation": "run_maintenance",
            "entity_type": entity_type,
            "entity_id": entity_id
        }
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action_type="memory_operation",
            action_details=action_details
        )
        
        if not permission["approved"]:
            logger.warning(f"Memory maintenance not approved: {permission.get('reasoning', 'Unknown reason')}")
            return {"error": permission.get("reasoning", "Memory maintenance not permitted by Nyx")}
        
        # If approved, proceed with memory maintenance
        with trace(workflow_name="Memory Operation - Run Maintenance"):
            maintenance_input = MaintenanceInput(
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            result = await self.memory_agent.run_maintenance(
                self.memory_context,
                entity_type=maintenance_input.entity_type,
                entity_id=maintenance_input.entity_id
            )
            
            # Report action back to Nyx
            await self.governor.process_agent_action_report(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                action={
                    "type": "memory_operation",
                    "operation": "run_maintenance",
                    "description": f"Ran memory maintenance for {entity_type} {entity_id}"
                },
                result=result
            )
            
            return result
    
    async def analyze_memories(
        self,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Analyze memories with Nyx governance.
        
        Args:
            entity_type: Type of entity ("player", "npc", etc.)
            entity_id: ID of the entity
        """
        # Check permission with governance system
        action_details = {
            "operation": "analyze_memories",
            "entity_type": entity_type,
            "entity_id": entity_id
        }
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action_type="memory_operation",
            action_details=action_details
        )
        
        if not permission["approved"]:
            logger.warning(f"Memory analysis not approved: {permission.get('reasoning', 'Unknown reason')}")
            return {"error": permission.get("reasoning", "Memory analysis not permitted by Nyx")}
        
        # If approved, proceed with memory analysis
        with trace(workflow_name="Memory Operation - Analyze Memories"):
            analysis_input = AnalysisInput(
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            result = await self.memory_agent.analyze_memories(
                self.memory_context,
                entity_type=analysis_input.entity_type,
                entity_id=analysis_input.entity_id
            )
            
            # Report action back to Nyx
            await self.governor.process_agent_action_report(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                action={
                    "type": "memory_operation",
                    "operation": "analyze_memories",
                    "description": f"Analyzed memories for {entity_type} {entity_id}"
                },
                result=result
            )
            
            return result
    
    async def generate_schemas(
        self,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Generate memory schemas with Nyx governance.
        
        Args:
            entity_type: Type of entity ("player", "npc", etc.)
            entity_id: ID of the entity
        """
        # Check permission with governance system
        action_details = {
            "operation": "generate_schemas",
            "entity_type": entity_type,
            "entity_id": entity_id
        }
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            action_type="memory_operation",
            action_details=action_details
        )
        
        if not permission["approved"]:
            logger.warning(f"Schema generation not approved: {permission.get('reasoning', 'Unknown reason')}")
            return {"error": permission.get("reasoning", "Schema generation not permitted by Nyx")}
        
        # If approved, proceed with schema generation
        with trace(workflow_name="Memory Operation - Generate Schemas"):
            result = await self.memory_agent.generate_schemas(
                self.memory_context,
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            # Report action back to Nyx
            await self.governor.process_agent_action_report(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                action={
                    "type": "memory_operation",
                    "operation": "generate_schemas",
                    "description": f"Generated memory schemas for {entity_type} {entity_id}"
                },
                result=result
            )
            
            return result
    
    async def process_memory_directive(
        self, 
        directive_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a directive from Nyx for the memory system.
        
        Args:
            directive_data: The directive data
            
        Returns:
            Results of processing the directive
        """
        operation = directive_data.get("operation")
        if not operation:
            return {"error": "No operation specified in directive"}
        
        entity_type = directive_data.get("entity_type", "npc")
        entity_id = directive_data.get("entity_id", 0)
        
        # Process the directive based on the operation
        if operation == "remember":
            result = await self.remember(
                entity_type=entity_type,
                entity_id=entity_id,
                memory_text=directive_data.get("memory_text", ""),
                importance=directive_data.get("importance", "medium"),
                emotional=directive_data.get("emotional", True),
                tags=directive_data.get("tags")
            )
        elif operation == "recall":
            result = await self.recall(
                entity_type=entity_type,
                entity_id=entity_id,
                query=directive_data.get("query"),
                context=directive_data.get("context"),
                limit=directive_data.get("limit", 5)
            )
        elif operation == "create_belief":
            result = await self.create_belief(
                entity_type=entity_type,
                entity_id=entity_id,
                belief_text=directive_data.get("belief_text", ""),
                confidence=directive_data.get("confidence", 0.7)
            )
        elif operation == "get_beliefs":
            result = await self.get_beliefs(
                entity_type=entity_type,
                entity_id=entity_id,
                topic=directive_data.get("topic")
            )
        elif operation == "run_maintenance":
            result = await self.run_maintenance(
                entity_type=entity_type,
                entity_id=entity_id
            )
        elif operation == "analyze_memories":
            result = await self.analyze_memories(
                entity_type=entity_type,
                entity_id=entity_id
            )
        elif operation == "generate_schemas":
            result = await self.generate_schemas(
                entity_type=entity_type,
                entity_id=entity_id
            )
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return {
            "directive_processed": True,
            "operation": operation,
            "result": result
        }

# Helper function to get the memory-nyx bridge
async def get_memory_nyx_bridge(user_id: int, conversation_id: int) -> MemoryNyxBridge:
    """
    Get (or create) the memory-nyx bridge for a user/conversation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        The memory-nyx bridge
    """
    # Use a cache to avoid recreating the bridge unnecessarily
    cache_key = f"memory_nyx_bridge:{user_id}:{conversation_id}"
    
    # Check if it's already in global dict
    if not hasattr(get_memory_nyx_bridge, "cache"):
        get_memory_nyx_bridge.cache = {}
    
    if cache_key in get_memory_nyx_bridge.cache:
        return get_memory_nyx_bridge.cache[cache_key]
    
    # Create new bridge
    bridge = MemoryNyxBridge(user_id, conversation_id)
    await bridge.initialize()
    
    # Cache it
    get_memory_nyx_bridge.cache[cache_key] = bridge
    
    return bridge
    
# Convenience functions for memory operations through Nyx
async def remember_through_nyx(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: int,
    memory_text: str,
    importance: str = "medium",
    emotional: bool = True,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a memory through Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        entity_type: Type of entity ("player", "npc", etc.)
        entity_id: ID of the entity
        memory_text: The memory text to record
        importance: Importance level ("trivial", "low", "medium", "high", "critical")
        emotional: Whether to analyze emotional content
        tags: Optional tags for the memory
    """
    bridge = await get_memory_nyx_bridge(user_id, conversation_id)
    return await bridge.remember(
        entity_type=entity_type,
        entity_id=entity_id,
        memory_text=memory_text,
        importance=importance,
        emotional=emotional,
        tags=tags
    )

async def recall_through_nyx(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: int,
    query: Optional[str] = None,
    context: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Recall memories through Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        entity_type: Type of entity ("player", "npc", etc.)
        entity_id: ID of the entity
        query: Optional search query
        context: Current context that might influence recall
        limit: Maximum number of memories to return
    """
    bridge = await get_memory_nyx_bridge(user_id, conversation_id)
    return await bridge.recall(
        entity_type=entity_type,
        entity_id=entity_id,
        query=query,
        context=context,
        limit=limit
    )

async def create_belief_through_nyx(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: int,
    belief_text: str,
    confidence: float = 0.7
) -> Dict[str, Any]:
    """
    Create a belief through Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        entity_type: Type of entity ("player", "npc", etc.)
        entity_id: ID of the entity
        belief_text: The belief statement
        confidence: Confidence in this belief (0.0-1.0)
    """
    bridge = await get_memory_nyx_bridge(user_id, conversation_id)
    return await bridge.create_belief(
        entity_type=entity_type,
        entity_id=entity_id,
        belief_text=belief_text,
        confidence=confidence
    )

async def run_maintenance_through_nyx(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: int
) -> Dict[str, Any]:
    """
    Run memory maintenance through Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        entity_type: Type of entity ("player", "npc", etc.)
        entity_id: ID of the entity
    """
    bridge = await get_memory_nyx_bridge(user_id, conversation_id)
    return await bridge.run_maintenance(
        entity_type=entity_type,
        entity_id=entity_id
    )
