# context/vector_integration.py

"""
Integration with vector database for semantic context retrieval.

This module provides specialized integrations with the RPGEntityManager
from paste.txt to enable semantic search capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import math

# Import configuration
from context_config import get_config

logger = logging.getLogger(__name__)

class VectorContextManager:
    """
    Manager for vector database integration with context optimization.
    
    This class wraps the RPGEntityManager from paste.txt to provide
    semantic search capabilities for context optimization.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the vector context manager.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.entity_manager = None
        self.initialized = False
        
        # Get configuration
        self.config = get_config()
        self.vector_db_config = self.config.get_vector_db_config()
        self.enabled = self.config.get("vector_db", "enabled", True)
    
    async def initialize(self):
        """Initialize the vector database connection"""
        if self.initialized or not self.enabled:
            return
        
        try:
            # Import the RPGEntityManager from paste.txt
            from paste import RPGEntityManager
            
            # Create entity manager
            self.entity_manager = RPGEntityManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                vector_db_config=self.vector_db_config
            )
            
            # Initialize the entity manager
            await self.entity_manager.initialize()
            
            self.initialized = True
            logger.info(f"Initialized vector context manager for user {self.user_id}, conversation {self.conversation_id}")
        except ImportError:
            logger.warning("Failed to import RPGEntityManager, vector search disabled")
            self.enabled = False
        except Exception as e:
            logger.error(f"Error initializing vector context manager: {e}")
            self.enabled = False
    
    async def close(self):
        """Close the vector database connection"""
        if self.entity_manager:
            try:
                await self.entity_manager.close()
                self.entity_manager = None
                self.initialized = False
                logger.info(f"Closed vector context manager for user {self.user_id}, conversation {self.conversation_id}")
            except Exception as e:
                logger.error(f"Error closing vector context manager: {e}")
    
    async def get_relevant_npcs(
        self, 
        query_text: str, 
        current_location: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get NPCs relevant to the query text.
        
        Args:
            query_text: Query text
            current_location: Optional current location for filtering
            limit: Maximum number of NPCs to return
            
        Returns:
            List of relevant NPCs
        """
        if not self.enabled:
            return []
        
        await self.initialize()
        
        if not self.entity_manager:
            return []
        
        try:
            # Get NPCs through vector search
            npc_entities = await self.entity_manager.get_relevant_entities(
                query_text=query_text,
                top_k=limit,
                entity_types=["npc"]
            )
            
            # Extract NPC data from entities
            npcs = []
            for entity in npc_entities:
                metadata = entity.get("metadata", {})
                if metadata.get("entity_type") == "npc":
                    # Add relevance score from vector search
                    metadata["relevance_score"] = entity.get("score", 0.5)
                    npcs.append(metadata)
            
            # Apply location filter if provided
            if current_location and npcs:
                # Boost NPCs in the current location
                for npc in npcs:
                    npc_location = npc.get("current_location")
                    if npc_location and npc_location.lower() == current_location.lower():
                        # Boost relevance score
                        npc["relevance_score"] = min(1.0, npc.get("relevance_score", 0.5) + 0.3)
            
            return npcs
        except Exception as e:
            logger.error(f"Error getting relevant NPCs: {e}")
            return []
    
    async def get_relevant_memories(
        self, 
        query_text: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the query text.
        
        Args:
            query_text: Query text
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        if not self.enabled:
            return []
        
        await self.initialize()
        
        if not self.entity_manager:
            return []
        
        try:
            # Get memories through vector search
            memory_entities = await self.entity_manager.get_relevant_entities(
                query_text=query_text,
                top_k=limit,
                entity_types=["memory"]
            )
            
            # Extract memory data from entities
            memories = []
            for entity in memory_entities:
                metadata = entity.get("metadata", {})
                if metadata.get("entity_type") == "memory":
                    # Add relevance score from vector search
                    metadata["relevance_score"] = entity.get("score", 0.5)
                    memories.append(metadata)
            
            return memories
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def get_relevant_locations(
        self, 
        query_text: str, 
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get locations relevant to the query text.
        
        Args:
            query_text: Query text
            limit: Maximum number of locations to return
            
        Returns:
            List of relevant locations
        """
        if not self.enabled:
            return []
        
        await self.initialize()
        
        if not self.entity_manager:
            return []
        
        try:
            # Get locations through vector search
            location_entities = await self.entity_manager.get_relevant_entities(
                query_text=query_text,
                top_k=limit,
                entity_types=["location"]
            )
            
            # Extract location data from entities
            locations = []
            for entity in location_entities:
                metadata = entity.get("metadata", {})
                if metadata.get("entity_type") == "location":
                    # Add relevance score from vector search
                    metadata["relevance_score"] = entity.get("score", 0.5)
                    locations.append(metadata)
            
            return locations
        except Exception as e:
            logger.error(f"Error getting relevant locations: {e}")
            return []
    
    async def get_context_for_input(
        self, 
        query_text: str, 
        current_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get context for the input text using vector search.
        
        Args:
            query_text: Query text
            current_location: Optional current location
            
        Returns:
            Dictionary with context from vector search
        """
        if not self.enabled:
            return {}
        
        await self.initialize()
        
        if not self.entity_manager:
            return {}
        
        try:
            # Use RPGEntityManager's context function directly
            context = await self.entity_manager.get_context_for_input(
                query_text, current_location
            )
            
            # Add source information
            context["source"] = "vector_search"
            
            return context
        except Exception as e:
            logger.error(f"Error getting context for input: {e}")
            return {}
    
    async def add_memory_embeddings(
        self, 
        memory_id: str, 
        content: str, 
        memory_type: str, 
        importance: float, 
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a memory to the vector database.
        
        Args:
            memory_id: Memory ID
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            tags: Optional list of tags
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        await self.initialize()
        
        if not self.entity_manager:
            return False
        
        try:
            # Add memory using the entity manager
            result = await self.entity_manager.add_memory(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or []
            )
            
            return result
        except Exception as e:
            logger.error(f"Error adding memory embeddings: {e}")
            return False
    
    async def search_with_hybrid_ranking(
        self, 
        query_text: str, 
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
        temporal_boost: bool = True,
        recency_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search entities with hybrid ranking combining vector similarity and temporal relevance.
        
        Args:
            query_text: Query text
            entity_types: List of entity types to search for
            limit: Maximum number of results to return
            temporal_boost: Whether to boost recency
            recency_weight: Weight of recency in ranking
            
        Returns:
            List of ranked search results
        """
        if not self.enabled:
            return []
        
        await self.initialize()
        
        if not self.entity_manager:
            return []
        
        if entity_types is None:
            entity_types = ["npc", "location", "memory", "narrative"]
        
        try:
            # Get raw results from vector search
            raw_results = await self.entity_manager.get_relevant_entities(
                query_text=query_text,
                top_k=limit * 2,  # Get more results than needed for reranking
                entity_types=entity_types
            )
            
            # Apply hybrid ranking
            ranked_results = self._apply_hybrid_ranking(
                raw_results, temporal_boost, recency_weight
            )
            
            # Return top results after reranking
            return ranked_results[:limit]
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []
    
    def _apply_hybrid_ranking(
        self, 
        results: List[Dict[str, Any]], 
        temporal_boost: bool, 
        recency_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid ranking to search results.
        
        Args:
            results: Raw search results
            temporal_boost: Whether to boost recency
            recency_weight: Weight of recency in ranking
            
        Returns:
            Reranked results
        """
        # If temporal boost is disabled, return results as is
        if not temporal_boost:
            return results
        
        from datetime import datetime
        now = datetime.now()
        
        # Calculate hybrid scores
        for result in results:
            # Get base vector similarity score
            base_score = result.get("score", 0.5)
            
            # Calculate temporal score if timestamp available
            temporal_score = 0.5  # Default middle value
            metadata = result.get("metadata", {})
            
            if "timestamp" in metadata:
                try:
                    # Parse timestamp
                    timestamp_str = metadata["timestamp"]
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        
                        # Calculate days difference
                        days_diff = (now - timestamp).days
                        
                        # Newer items get higher scores
                        # Using exponential decay: score = exp(-0.1 * days)
                        temporal_score = math.exp(-0.1 * max(0, days_diff))
                except:
                    pass
            
            # Combine scores with weighting
            # vector_weight + recency_weight should equal 1.0
            vector_weight = 1.0 - recency_weight
            hybrid_score = (vector_weight * base_score) + (recency_weight * temporal_score)
            
            # Update the result with the hybrid score
            result["hybrid_score"] = hybrid_score
        
        # Sort by hybrid score
        results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        return results

# -------------------------------------------------------------------------------
# Singleton Manager
# -------------------------------------------------------------------------------

_manager_instances = {}

async def get_vector_manager(user_id: int, conversation_id: int) -> VectorContextManager:
    """
    Get or create a vector context manager instance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        VectorContextManager instance
    """
    global _manager_instances
    
    key = f"{user_id}:{conversation_id}"
    
    if key not in _manager_instances:
        # Create new instance
        manager = VectorContextManager(user_id, conversation_id)
        await manager.initialize()
        _manager_instances[key] = manager
    
    return _manager_instances[key]

# -------------------------------------------------------------------------------
# Main API Functions
# -------------------------------------------------------------------------------

async def get_vector_enhanced_context(
    user_id: int, 
    conversation_id: int, 
    query_text: str, 
    current_location: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get enhanced context using vector search capabilities.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        current_location: Optional current location
        limit: Maximum number of items per category
        
    Returns:
        Vector-enhanced context
    """
    # Get vector manager
    manager = await get_vector_manager(user_id, conversation_id)
    
    # Skip if vector search is disabled
    if not manager.enabled:
        return {"vector_search_enabled": False}
    
    # Get components in parallel
    npcs_task = asyncio.create_task(
        manager.get_relevant_npcs(query_text, current_location, limit)
    )
    
    memories_task = asyncio.create_task(
        manager.get_relevant_memories(query_text, limit)
    )
    
    locations_task = asyncio.create_task(
        manager.get_relevant_locations(query_text, limit)
    )
    
    # Wait for all tasks to complete
    npcs, memories, locations = await asyncio.gather(
        npcs_task, memories_task, locations_task
    )
    
    # Assemble context
    context = {
        "vector_search_enabled": True,
        "npcs": npcs,
        "memories": memories,
        "locations": locations,
        "query_text": query_text,
        "current_location": current_location
    }
    
    return context

async def store_memory_with_embedding(
    user_id: int,
    conversation_id: int,
    memory_id: str,
    content: str,
    memory_type: str = "observation",
    importance: float = 0.5,
    tags: Optional[List[str]] = None
) -> bool:
    """
    Store a memory in the database and add it to the vector database.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        memory_id: Memory ID
        content: Memory content
        memory_type: Type of memory
        importance: Importance score (0-1)
        tags: Optional list of tags
        
    Returns:
        True if successful, False otherwise
    """
    # Get vector manager
    manager = await get_vector_manager(user_id, conversation_id)
    
    # Add to vector database if enabled
    if manager.enabled:
        return await manager.add_memory_embeddings(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags
        )
    
    return False

async def cleanup_vector_managers():
    """Close all vector managers to release resources"""
    global _manager_instances
    
    for key, manager in list(_manager_instances.items()):
        try:
            await manager.close()
        except Exception as e:
            logger.error(f"Error closing vector manager {key}: {e}")
    
    _manager_instances.clear()
