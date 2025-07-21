# memory/memory_integration.py
"""
Memory Integration Module

This module provides:
1. Helper functions to integrate the memory retrieval system with Celery tasks
2. Integration layer between the memory system and Nyx's central governance framework
"""

import os
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import wraps
from datetime import datetime

# Import memory components
from memory.memory_service import MemoryEmbeddingService
from memory.memory_retriever import MemoryRetrieverAgent
from memory.memory_nyx_integration import MemoryNyxBridge
from memory.core import MemoryType, MemorySignificance

# Import database connection
from db.connection import get_db_connection_context

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# PART 1: Memory Service Helper Functions
# ============================================================================

# Global registry to avoid recreating services
_memory_services = {}
_memory_retrievers = {}

async def get_memory_service(
    user_id: int,
    conversation_id: int,
    vector_store_type: str = "chroma",  # or "faiss" or "qdrant"
    embedding_model: str = "local",     # or "openai"
    config: Optional[Dict[str, Any]] = None
) -> MemoryEmbeddingService:
    """
    Get or create a memory service instance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        MemoryEmbeddingService instance
    """
    global _memory_services
    
    key = f"{user_id}:{conversation_id}:{vector_store_type}:{embedding_model}"
    
    if key not in _memory_services:
        service = MemoryEmbeddingService(
            user_id=user_id,
            conversation_id=conversation_id,
            vector_store_type=vector_store_type,
            embedding_model=embedding_model,
            config=config
        )
        await service.initialize()
        _memory_services[key] = service
    
    return _memory_services[key]

async def get_memory_retriever(
    user_id: int,
    conversation_id: int,
    llm_type: str = "openai",  # or "huggingface"
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> MemoryRetrieverAgent:
    """
    Get or create a memory retriever agent.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        llm_type: Type of LLM to use
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        MemoryRetrieverAgent instance
    """
    global _memory_retrievers
    
    key = f"{user_id}:{conversation_id}:{llm_type}:{vector_store_type}:{embedding_model}"
    
    if key not in _memory_retrievers:
        # Get or create memory service
        memory_service = await get_memory_service(
            user_id=user_id,
            conversation_id=conversation_id,
            vector_store_type=vector_store_type,
            embedding_model=embedding_model,
            config=config
        )
        
        # Create retriever
        retriever = MemoryRetrieverAgent(
            user_id=user_id,
            conversation_id=conversation_id,
            llm_type=llm_type,
            memory_service=memory_service,
            config=config
        )
        await retriever.initialize()
        _memory_retrievers[key] = retriever
    
    return _memory_retrievers[key]

async def cleanup_memory_services():
    """Close all memory services."""
    global _memory_services
    
    close_tasks = []
    for key, service in list(_memory_services.items()):
        close_tasks.append(asyncio.create_task(service.close()))
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    _memory_services.clear()

async def cleanup_memory_retrievers():
    """Close all memory retrievers."""
    global _memory_retrievers
    
    close_tasks = []
    for key, retriever in list(_memory_retrievers.items()):
        close_tasks.append(asyncio.create_task(retriever.close()))
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    _memory_retrievers.clear()

async def add_memory_from_message(
    user_id: int,
    conversation_id: int,
    message_text: str,
    entity_type: str = "memory",
    metadata: Optional[Dict[str, Any]] = None,
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a memory from a message.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        message_text: Message text
        entity_type: Entity type
        metadata: Optional metadata
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        Memory ID
    """
    # Initialize metadata if not provided
    metadata = metadata or {}
    
    # Get memory service
    memory_service = await get_memory_service(
        user_id=user_id,
        conversation_id=conversation_id,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        config=config
    )
    
    # Add memory
    memory_id = await memory_service.add_memory(
        text=message_text,
        metadata=metadata,
        entity_type=entity_type
    )
    
    return memory_id

async def retrieve_relevant_memories(
    user_id: int,
    conversation_id: int,
    query_text: str,
    entity_types: Optional[List[str]] = None,
    top_k: int = 5,
    threshold: float = 0.7,
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve memories relevant to a query.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        entity_types: List of entity types to search
        top_k: Number of results to return
        threshold: Relevance threshold
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        List of relevant memories
    """
    # Get memory service
    memory_service = await get_memory_service(
        user_id=user_id,
        conversation_id=conversation_id,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        config=config
    )
    
    # Default to searching all entity types if none specified
    if not entity_types:
        entity_types = ["memory", "npc", "location", "narrative"]
    
    # Collect memories from all entity types
    all_memories = []
    
    for entity_type in entity_types:
        try:
            # Search for memories of this entity type
            memories = await memory_service.search_memories(
                query_text=query_text,
                entity_type=entity_type,
                top_k=top_k,
                fetch_content=True
            )
            
            # Filter by threshold
            memories = [m for m in memories if m["relevance"] >= threshold]
            
            all_memories.extend(memories)
            
        except Exception as e:
            logger.error(f"Error retrieving {entity_type} memories: {e}")
    
    # Sort by relevance
    all_memories.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Limit to top_k overall
    return all_memories[:top_k]

async def analyze_with_memory(
    user_id: int,
    conversation_id: int,
    query_text: str,
    entity_types: Optional[List[str]] = None,
    top_k: int = 5,
    threshold: float = 0.7,
    llm_type: str = "openai",
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze a query with relevant memories.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        entity_types: List of entity types to search
        top_k: Number of results to return
        threshold: Relevance threshold
        llm_type: Type of LLM to use
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        Dictionary with analysis results
    """
    # Get memory retriever
    retriever = await get_memory_retriever(
        user_id=user_id,
        conversation_id=conversation_id,
        llm_type=llm_type,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        config=config
    )
    
    # Retrieve and analyze memories
    result = await retriever.retrieve_and_analyze(
        query=query_text,
        entity_types=entity_types,
        top_k=top_k,
        threshold=threshold
    )
    
    return result

async def enrich_context_with_memories(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enrich a context dictionary with relevant memories.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User input text
        context: Context dictionary to enrich
        
    Returns:
        Enriched context dictionary
    """
    try:
        # Get memory retriever
        retriever = await get_memory_retriever(
            user_id=user_id,
            conversation_id=conversation_id,
            llm_type="openai",  # Use "huggingface" if preferred
            vector_store_type="chroma",  # Or "faiss" or "qdrant"
            embedding_model="local"  # Or "openai"
        )
        
        # Retrieve and analyze memories
        memory_result = await retriever.retrieve_and_analyze(
            query=user_input,
            entity_types=["memory", "npc", "location", "narrative"],
            top_k=5,
            threshold=0.7
        )
        
        # Add to context
        if memory_result["found_memories"]:
            # Get original context memory structure or create it
            if "memories" not in context:
                context["memories"] = []
            
            # Add retrieved memories
            context["memories"].extend(memory_result["memories"])
            
            # Add analysis
            if "memory_analysis" not in context:
                context["memory_analysis"] = {}
            
            context["memory_analysis"] = {
                "primary_theme": memory_result["analysis"].primary_theme,
                "insights": [insight.dict() for insight in memory_result["analysis"].insights],
                "suggested_response": memory_result["analysis"].suggested_response
            }
        
        return context
    
    except Exception as e:
        logger.error(f"Error enriching context with memories: {e}")
        return context  # Return original context if error occurs

# Celery task wrapper
def memory_celery_task(func):
    """Decorator to handle async memory tasks in Celery."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Run the async function in the asyncio event loop
        result = asyncio.run(func(*args, **kwargs))
        return result
    
    return wrapper

# Example Celery task function
@memory_celery_task
async def process_memory_task(user_id, conversation_id, message_text, entity_type="memory"):
    """
    Celery task to process a memory.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        message_text: Message text
        entity_type: Entity type
        
    Returns:
        Dictionary with task result
    """
    try:
        # Add memory to vector store
        memory_id = await add_memory_from_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message_text,
            entity_type=entity_type
        )
        
        return {
            "success": True,
            "memory_id": memory_id,
            "message": f"Successfully processed memory for user {user_id}, conversation {conversation_id}"
        }
    
    except Exception as e:
        logger.error(f"Error processing memory task: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# PART 2: Memory Nyx Governance Integration
# ============================================================================

class MemoryIntegration(MemoryNyxBridge):
    """
    Integration between Memory System and Nyx Governance.
    
    Extends MemoryNyxBridge to provide complete compatibility with
    the expected interface in nyx/integrate.py.
    """
    
    async def get_state(self) -> Dict[str, Any]:
        """
        Get current memory system state for state reconciliation.
        
        Returns:
            Dictionary with memory system state including memory counts,
            recent operations, and system health
        """
        if not self.initialized:
            await self.initialize()
        
        # Get memory system stats
        memory_counts = {}
        entity_types = ["npc", "player", "nyx", "location", "item"]
        
        try:
            # Get memory counts by entity type
            for entity_type in entity_types:
                memory_counts[entity_type] = await self._get_memory_count(entity_type)
            
            # Get recent maintenance data
            maintenance_data = await self._get_maintenance_schedule()
            
            # Get recent memory telemetry
            telemetry = await self._get_recent_telemetry()
            
            # Get memory evolution stats
            evolution_stats = await self._get_memory_evolution_stats()
            
            # Format the state
            return {
                "memory_counts": memory_counts,
                "maintenance_schedule": maintenance_data,
                "telemetry": {
                    "recent_operations": [t for t in telemetry if t.get("success", False)][-5:],
                    "recent_errors": [t for t in telemetry if not t.get("success", False)][-5:],
                    "error_count": len([t for t in telemetry if not t.get("success", False)]),
                    "operation_count": len(telemetry)
                },
                "evolution": evolution_stats,
                "last_operation": self.state_tracker.get("last_operation"),
                "last_operation_time": self.state_tracker.get("last_operation_time"),
                "status": "healthy" if self.state_tracker.get("error_count", 0) < 5 else "degraded"
            }
            
        except Exception as e:
            logger.error(f"Error getting memory state: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def _get_memory_count(self, entity_type: str) -> int:
        """Get memory count for an entity type using the unified_memories table."""
        try:
            async with get_db_connection_context() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM unified_memories
                    WHERE user_id = $1 AND conversation_id = $2 AND entity_type = $3
                    AND status != 'deleted'
                """, self.user_id, self.conversation_id, entity_type)
                
                return count or 0
        except Exception as e:
            logger.error(f"Error getting memory count for {entity_type}: {e}")
            return 0
    
    async def _get_maintenance_schedule(self) -> List[Dict[str, Any]]:
        """Get maintenance schedule data from the database."""
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT entity_type, entity_id, maintenance_schedule, 
                           next_maintenance_date, last_maintenance_date
                    FROM MemoryMaintenanceSchedule
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY next_maintenance_date ASC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting maintenance schedule: {e}")
            return []
    
    async def _get_recent_telemetry(self) -> List[Dict[str, Any]]:
        """Get recent memory telemetry data."""
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT operation, success, duration, data_size, error, timestamp
                    FROM memory_telemetry
                    ORDER BY timestamp DESC
                    LIMIT 20
                """)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting memory telemetry: {e}")
            return []
    
    async def _get_memory_evolution_stats(self) -> Dict[str, Any]:
        """Get memory evolution statistics."""
        try:
            async with get_db_connection_context() as conn:
                # Get recent context evolutions
                evolutions = await conn.fetch("""
                    SELECT evolution_id, context_shift, timestamp
                    FROM ContextEvolution
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY timestamp DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
                # Get memory evolution stats
                memory_shifts = await conn.fetch("""
                    SELECT COUNT(*) as count, AVG(relevance_change) as avg_change
                    FROM MemoryContextEvolution mce
                    JOIN ContextEvolution ce ON mce.evolution_id = ce.evolution_id
                    WHERE ce.user_id = $1 AND ce.conversation_id = $2
                    GROUP BY mce.evolution_id
                    ORDER BY ce.timestamp DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
                return {
                    "recent_evolutions": [dict(row) for row in evolutions],
                    "memory_shifts": [dict(row) for row in memory_shifts]
                }
        except Exception as e:
            logger.error(f"Error getting memory evolution stats: {e}")
            return {"recent_evolutions": [], "memory_shifts": []}
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get health metrics for memory system.
        
        Returns:
            Dictionary with health metrics including error rate,
            operation counts, and system status
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get database metrics
            async with get_db_connection_context() as conn:
                # Get error rate from telemetry
                error_rate_row = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_ops,
                        COUNT(*) FILTER (WHERE NOT success) as error_count
                    FROM memory_telemetry
                    WHERE timestamp > NOW() - interval '1 day'
                """)
                
                total_ops = error_rate_row['total_ops'] if error_rate_row else 0
                error_count = error_rate_row['error_count'] if error_rate_row else 0
                
                # Get average operation duration
                avg_duration_row = await conn.fetchrow("""
                    SELECT AVG(duration) as avg_duration
                    FROM memory_telemetry
                    WHERE timestamp > NOW() - interval '1 hour'
                """)
                
                avg_duration = avg_duration_row['avg_duration'] if avg_duration_row else 0
                
                # Get memory counts
                memory_count_row = await conn.fetchrow("""
                    SELECT COUNT(*) as memory_count
                    FROM unified_memories
                    WHERE user_id = $1 AND conversation_id = $2
                    AND status != 'deleted'
                """, self.user_id, self.conversation_id)
                
                memory_count = memory_count_row['memory_count'] if memory_count_row else 0
                
                # Get consolidated memory ratio
                consolidated_row = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) FILTER (WHERE is_consolidated) as consolidated_count,
                        COUNT(*) as total_count
                    FROM unified_memories
                    WHERE user_id = $1 AND conversation_id = $2
                    AND status != 'deleted'
                """, self.user_id, self.conversation_id)
                
                consolidated_count = consolidated_row['consolidated_count'] if consolidated_row else 0
                total_count = consolidated_row['total_count'] if consolidated_row else 0
                consolidated_ratio = consolidated_count / total_count if total_count > 0 else 0
            
            # Calculate error rate
            error_rate = error_count / max(1, total_ops)
            
            return {
                "status": "healthy" if error_rate < 0.1 else "degraded",
                "error_rate": error_rate,
                "operation_count": total_ops,
                "error_count": error_count,
                "avg_response_time": avg_duration,
                "memory_count": memory_count,
                "consolidated_ratio": consolidated_ratio,
                "active_transactions": self.state_tracker.get("active_transactions", 0),
                "last_operation_time": self.state_tracker.get("last_operation_time")
            }
            
        except Exception as e:
            logger.error(f"Error getting memory health metrics: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _track_state_change(self, operation: str, details: Dict[str, Any]):
        """
        Track state changes for metrics and monitoring.
        Extends the base implementation to add timestamp.
        """
        # Call parent implementation first
        super()._track_state_change(operation, details)
        
        # Add timestamp for tracking
        self.state_tracker["last_operation_time"] = datetime.now().isoformat()
        
        # Keep track of recent operations
        if "recent_operations" not in self.state_tracker:
            self.state_tracker["recent_operations"] = []
            
        operation_record = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        
        self.state_tracker["recent_operations"].append(operation_record)
        if len(self.state_tracker["recent_operations"]) > 20:
            self.state_tracker["recent_operations"] = self.state_tracker["recent_operations"][-20:]
    
    async def update_entity_schemas(self, entity_type: str, entity_id: int) -> Dict[str, Any]:
        """
        Update schemas for an entity based on their memories.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Schema update results
        """
        try:
            # Generate schemas for the entity using the method in MemoryNyxBridge
            result = await self.generate_schemas(
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            # Track state change
            self._track_state_change("update_entity_schemas", {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "schema_count": len(result.get("schemas", []))
            })
            
            return result
            
        except Exception as e:
            self._track_error("update_entity_schemas", str(e))
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def update_maintenance_schedule(self, entity_type: str, entity_id: int, 
                                         schedule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the maintenance schedule for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            schedule: Maintenance schedule configuration
            
        Returns:
            Update result
        """
        try:
            async with get_db_connection_context() as conn:
                # Calculate next maintenance date based on schedule
                if "frequency_days" in schedule:
                    next_date = f"NOW() + interval '{schedule['frequency_days']} days'"
                else:
                    next_date = "NOW() + interval '7 days'"  # Default to weekly
                
                # Update or insert maintenance schedule
                await conn.execute("""
                    INSERT INTO MemoryMaintenanceSchedule 
                    (user_id, conversation_id, entity_type, entity_id, 
                     maintenance_schedule, next_maintenance_date)
                    VALUES ($1, $2, $3, $4, $5, {})
                    ON CONFLICT (user_id, conversation_id, entity_type, entity_id)
                    DO UPDATE SET 
                        maintenance_schedule = $5,
                        next_maintenance_date = {}
                """.format(next_date, next_date), 
                    self.user_id, self.conversation_id, entity_type, entity_id, 
                    json.dumps(schedule))
                
                # Track state change
                self._track_state_change("update_maintenance_schedule", {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "schedule": schedule
                })
                
                return {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "schedule_updated": True,
                    "next_maintenance": next_date
                }
                
        except Exception as e:
            self._track_error("update_maintenance_schedule", str(e))
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def record_memory_evolution(self, context_data: Dict[str, Any], 
                                     changes: Dict[str, Any], 
                                     affected_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Record a memory evolution event and update memory relevance.
        
        Args:
            context_data: Current context data
            changes: Changes in the context
            affected_memories: Memories affected by the evolution
            
        Returns:
            Evolution record result
        """
        try:
            # Calculate context shift based on changes
            context_shift = self._calculate_context_shift(changes)
            
            async with get_db_connection_context() as conn:
                # Insert evolution record
                evolution_id = await conn.fetchval("""
                    INSERT INTO ContextEvolution
                    (user_id, conversation_id, context_data, changes, context_shift)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING evolution_id
                """, self.user_id, self.conversation_id, 
                     json.dumps(context_data), json.dumps(changes), context_shift)
                
                # Record memory relevance changes
                for memory in affected_memories:
                    memory_id = memory.get("id")
                    relevance_change = memory.get("relevance_change", 0.0)
                    
                    if memory_id:
                        await conn.execute("""
                            INSERT INTO MemoryContextEvolution
                            (memory_id, evolution_id, relevance_change)
                            VALUES ($1, $2, $3)
                        """, memory_id, evolution_id, relevance_change)
                        
                        # Update memory relevance
                        await conn.execute("""
                            UPDATE unified_memories
                            SET relevance_score = relevance_score + $1,
                                last_context_update = NOW()
                            WHERE id = $2
                        """, relevance_change, memory_id)
            
            # Track state change
            self._track_state_change("record_memory_evolution", {
                "evolution_id": evolution_id,
                "context_shift": context_shift,
                "affected_memories": len(affected_memories)
            })
            
            return {
                "evolution_id": evolution_id,
                "context_shift": context_shift,
                "affected_memories": len(affected_memories),
                "status": "success"
            }
            
        except Exception as e:
            self._track_error("record_memory_evolution", str(e))
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _calculate_context_shift(self, changes: Dict[str, Any]) -> float:
        """Calculate context shift magnitude based on changes."""
        shift_value = 0.0
        
        # Check themes changed
        if "themes" in changes:
            added = len(changes["themes"].get("added", []))
            removed = len(changes["themes"].get("removed", []))
            shift_value += (added + removed) * 0.1
        
        # Check characters changed
        if "characters" in changes:
            added = len(changes["characters"].get("added", []))
            removed = len(changes["characters"].get("removed", []))
            shift_value += (added + removed) * 0.2
        
        # Check locations changed
        if "locations" in changes:
            added = len(changes["locations"].get("added", []))
            removed = len(changes["locations"].get("removed", []))
            shift_value += (added + removed) * 0.15
        
        # Check relationships changed
        if "relationships" in changes:
            added = len(changes["relationships"].get("added", []))
            removed = len(changes["relationships"].get("removed", []))
            modified = len(changes["relationships"].get("modified", []))
            shift_value += (added + removed + modified) * 0.25
        
        return min(shift_value, 1.0)  # Cap at 1.0
