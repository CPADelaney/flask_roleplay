# context/vector_service.py

import asyncio
import logging
import time
import hashlib
import math
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from context.unified_cache import context_cache
from context.context_config import get_config

logger = logging.getLogger(__name__)

class VectorService:
    """
    Unified vector service with simplified API and optimized performance.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the vector service"""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.entity_manager = None
        self.initialized = False
        self.embedding_cache = {}
        self.batch_queue = asyncio.Queue()
        self.batch_task = None
        self.enabled = self.config.get("vector_db", "enabled", True) and \
                       self.config.is_enabled("use_vector_search")
    
    async def initialize(self):
        """Initialize the vector service"""
        if self.initialized or not self.enabled:
            return
            
        try:
            # Create entity manager
            vector_db_config = self.config.get_vector_db_config()
            
            # Import here to avoid circular imports
            from context.optimized_db import RPGEntityManager, create_vector_database
            
            # Create database
            vector_db = create_vector_database(vector_db_config)
            await vector_db.initialize()
            
            # Create entity manager
            self.entity_manager = RPGEntityManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                vector_db_config=vector_db_config
            )
            
            # Initialize entity manager
            await self.entity_manager.initialize()
            
            # Start batch processor
            self.batch_task = asyncio.create_task(self._process_batch_queue())
            
            self.initialized = True
            logger.info(f"Initialized vector service for user {self.user_id}, conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Error initializing vector service: {e}")
            self.enabled = False
    
    async def close(self):
        """Close the vector service"""
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        if self.entity_manager:
            await self.entity_manager.close()
        
        self.initialized = False
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text with caching"""
        if not self.enabled:
            # Return zeroed vector if not enabled
            return [0.0] * 384
            
        # Hash text for cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"embed:{text_hash}"
        
        # Check local memory cache (fastest)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Function to generate embedding
        async def generate_embedding():
            # Use entity manager's embedding service if available
            if self.entity_manager and hasattr(self.entity_manager, 'embedding_service') and self.entity_manager.embedding_service:
                return await self.entity_manager.embedding_service.get_embedding(text)
            
            # Fallback to random embedding for testing
            vec = list(np.random.normal(0, 1, 384))
            return vec / np.linalg.norm(vec)
        
        # Get from cache or generate
        embedding = await context_cache.get(
            cache_key,
            generate_embedding,
            cache_level=3,  # Long-term cache
            importance=0.3,
            ttl_override=86400  # 24 hours
        )
        
        # Store in local cache
        self.embedding_cache[cache_key] = embedding
        
        # Keep local cache manageable
        if len(self.embedding_cache) > 1000:
            # Remove 20% of entries
            keys_to_remove = list(self.embedding_cache.keys())[:200]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        return embedding
    
    async def _process_batch_queue(self):
        """Process batched vector operations"""
        while True:
            try:
                # Process batches up to 10 items or wait 100ms max
                batch = []
                
                try:
                    # Get first item
                    first_item = await asyncio.wait_for(self.batch_queue.get(), timeout=0.1)
                    batch.append(first_item)
                    
                    # Get more items if available (no waiting)
                    while len(batch) < 10 and not self.batch_queue.empty():
                        batch.append(self.batch_queue.get_nowait())
                except asyncio.TimeoutError:
                    # No items available within timeout
                    await asyncio.sleep(0.01)  # Avoid tight loop
                    continue
                
                if not batch:
                    continue
                
                # Process each batch item
                for item in batch:
                    try:
                        operation = item["operation"]
                        
                        if operation == "search":
                            result = await self._perform_search(item["data"])
                        elif operation == "add_memory":
                            result = await self._add_memory(item["data"])
                        elif operation == "add_entity":
                            result = await self._add_entity(item["data"])
                        else:
                            result = {"error": f"Unknown operation: {operation}"}
                        
                        # Set result
                        item["future"].set_result(result)
                    except Exception as e:
                        # Set exception
                        item["future"].set_exception(e)
                    finally:
                        self.batch_queue.task_done()
            
            except asyncio.CancelledError:
                # Task was cancelled
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)  # Avoid tight error loop
    
    async def _perform_search(self, data):
        """Perform a vector search operation"""
        if not self.entity_manager:
            await self.initialize()
            if not self.entity_manager:
                return []
        
        query_text = data.get("query_text", "")
        entity_types = data.get("entity_types", ["npc", "location", "memory"])
        top_k = data.get("top_k", 5)
        
        # Get embedding for query
        query_embedding = await self.get_embedding(query_text)
        
        # Perform search
        results = await self.entity_manager.get_relevant_entities(
            query_text="",  # Not used when embedding is provided
            top_k=top_k,
            entity_types=entity_types,
            query_embedding=query_embedding
        )
        
        # Apply hybrid ranking if requested
        if data.get("hybrid_ranking", False):
            results = self._apply_hybrid_ranking(
                results,
                recency_weight=data.get("recency_weight", 0.3)
            )
        
        return results
    
    def _apply_hybrid_ranking(self, results, recency_weight=0.3):
        """Apply hybrid ranking with vector similarity and recency"""
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
    
    async def _add_memory(self, data):
        """Add a memory to the vector database"""
        if not self.entity_manager:
            await self.initialize()
            if not self.entity_manager:
                return False
        
        memory_id = data.get("memory_id", "")
        content = data.get("content", "")
        memory_type = data.get("memory_type", "observation")
        importance = data.get("importance", 0.5)
        tags = data.get("tags", [])
        
        # Get embedding for content
        embedding = await self.get_embedding(content)
        
        # Add to entity manager
        return await self.entity_manager.add_memory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            embedding=embedding
        )
    
    async def _add_entity(self, data):
        """Add an entity to the vector database"""
        if not self.entity_manager:
            await self.initialize()
            if not self.entity_manager:
                return False
        
        entity_type = data.get("entity_type", "")
        entity_id = data.get("entity_id", "")
        content = data.get("content", "")
        
        # Create embedding text based on entity type
        if entity_type == "npc":
            embed_text = f"NPC: {data.get('name', '')}. {data.get('description', '')}"
        elif entity_type == "location":
            embed_text = f"Location: {data.get('name', '')}. {data.get('description', '')}"
        else:
            embed_text = content
        
        # Get embedding
        embedding = await self.get_embedding(embed_text)
        
        # Add to entity manager using generic method
        return await self.entity_manager.add_entity(
            entity_type=entity_type,
            entity_id=entity_id,
            content=content,
            embedding=embedding,
            **{k: v for k, v in data.items() if k not in ["entity_type", "entity_id", "content"]}
        )
    
    async def search_entities(
        self,
        query_text: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 5,
        hybrid_ranking: bool = True,
        recency_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by query text
        
        Args:
            query_text: Search query text
            entity_types: Types of entities to search
            top_k: Maximum number of results
            hybrid_ranking: Whether to apply hybrid ranking
            recency_weight: Weight of recency in ranking
            
        Returns:
            List of matching entities
        """
        if not self.enabled:
            return []
        
        # Create future for result
        future = asyncio.get_event_loop().create_future()
        
        # Add to batch queue
        await self.batch_queue.put({
            "operation": "search",
            "data": {
                "query_text": query_text,
                "entity_types": entity_types or ["npc", "location", "memory"],
                "top_k": top_k,
                "hybrid_ranking": hybrid_ranking,
                "recency_weight": recency_weight
            },
            "future": future
        })
        
        # Wait for result
        try:
            return await future
        except Exception as e:
            logger.error(f"Error in search_entities: {e}")
            return []
    
    async def add_memory(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "observation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a memory to the vector database
        
        Args:
            memory_id: Memory ID
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            tags: Optional tags
            
        Returns:
            Success indicator
        """
        if not self.enabled:
            return False
            
        # Create future for result
        future = asyncio.get_event_loop().create_future()
        
        # Add to batch queue
        await self.batch_queue.put({
            "operation": "add_memory",
            "data": {
                "memory_id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "tags": tags or []
            },
            "future": future
        })
        
        # Wait for result
        try:
            return await future
        except Exception as e:
            logger.error(f"Error in add_memory: {e}")
            return False
    
    async def get_context_for_input(
        self,
        input_text: str,
        current_location: Optional[str] = None,
        max_items: int = 10
    ) -> Dict[str, Any]:
        """
        Get relevant context for input text
        
        Args:
            input_text: Input text
            current_location: Current location
            max_items: Maximum items per category
            
        Returns:
            Context dictionary
        """
        if not self.enabled:
            return {}
            
        # Use cache
        cache_key = f"vector_context:{self.user_id}:{self.conversation_id}:{input_text}:{current_location}"
        
        async def fetch_context():
            # Combine input with location for better context
            query = input_text
            if current_location:
                query += f" Location: {current_location}"
            
            # Search for entities
            results = await self.search_entities(
                query_text=query,
                entity_types=["npc", "location", "memory", "narrative"],
                top_k=max_items
            )
            
            # Organize by entity type
            context = {
                "npcs": [],
                "locations": [],
                "memories": [],
                "narratives": []
            }
            
            for result in results:
                metadata = result.get("metadata", {})
                entity_type = metadata.get("entity_type")
                
                if entity_type == "npc":
                    context["npcs"].append({
                        "npc_id": metadata.get("npc_id"),
                        "npc_name": metadata.get("npc_name"),
                        "description": metadata.get("description"),
                        "personality": metadata.get("personality"),
                        "location": metadata.get("location"),
                        "relevance": result.get("score", 0.5)
                    })
                elif entity_type == "location":
                    context["locations"].append({
                        "location_id": metadata.get("location_id"),
                        "location_name": metadata.get("location_name"),
                        "description": metadata.get("description"),
                        "connected_locations": metadata.get("connected_locations", []),
                        "relevance": result.get("score", 0.5)
                    })
                elif entity_type == "memory":
                    context["memories"].append({
                        "memory_id": metadata.get("memory_id"),
                        "content": metadata.get("content"),
                        "memory_type": metadata.get("memory_type"),
                        "importance": metadata.get("importance", 0.5),
                        "relevance": result.get("score", 0.5)
                    })
                elif entity_type == "narrative":
                    context["narratives"].append({
                        "narrative_id": metadata.get("narrative_id"),
                        "content": metadata.get("content"),
                        "narrative_type": metadata.get("narrative_type"),
                        "importance": metadata.get("importance", 0.5),
                        "relevance": result.get("score", 0.5)
                    })
            
            return context
        
        # Get from cache or fetch
        importance = min(0.7, 0.3 + (len(input_text) / 100))  # Longer queries = more important
        return await context_cache.get(
            cache_key, 
            fetch_context, 
            cache_level=1,  # L1 cache
            importance=importance,
            ttl_override=30  # 30 seconds
        )


# Global registry
_vector_services = {}

async def get_vector_service(user_id: int, conversation_id: int) -> VectorService:
    """Get or create a vector service instance"""
    global _vector_services
    
    key = f"{user_id}:{conversation_id}"
    
    if key not in _vector_services:
        service = VectorService(user_id, conversation_id)
        _vector_services[key] = service
    
    return _vector_services[key]

async def cleanup_vector_services():
    """Close all vector services"""
    global _vector_services
    
    close_tasks = []
    for key, service in list(_vector_services.items()):
        close_tasks.append(asyncio.create_task(service.close()))
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    _vector_services.clear()

async def get_vector_enhanced_context(
    user_id: int,
    conversation_id: int,
    query_text: str,
    current_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get context enhanced with vector search
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        current_location: Optional current location
        
    Returns:
        Vector-enhanced context
    """
    # Get vector service
    service = await get_vector_service(user_id, conversation_id)
    
    # Get context
    context = await service.get_context_for_input(
        input_text=query_text,
        current_location=current_location
    )
    
    return {
        "vector_search_enabled": service.enabled,
        **context
    }
