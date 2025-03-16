# context/vector_service.py

"""
Optimized Vector Service that integrates with the unified cache system
and provides coordinated vector search capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from context.unified_cache import context_cache
from context.context_config import get_config
from context.optimized_db import RPGEntityManager, create_vector_database

logger = logging.getLogger(__name__)

# Singleton instance pattern
_vector_managers = {}

class OptimizedVectorService:
    """
    Vector service with caching, batching, and performance optimization
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.entity_manager = None
        self.initialized = False
        self.embedding_cache = {}
        self.config = get_config()
        self.batch_queue = asyncio.Queue()
        self.batch_task = None
        self.vector_db_config = self.config.get_vector_db_config()
        self.enabled = self.config.get("vector_db", "enabled", True) and \
                      self.config.is_enabled("use_vector_search")
    
    async def initialize(self):
        """Initialize the vector service"""
        if self.initialized or not self.enabled:
            return

        try:
            # Create entity manager (lightweight operation)
            self.entity_manager = await self._create_entity_manager()
            
            # Start batch processing task
            self.batch_task = asyncio.create_task(self._process_batch_queue())
            
            self.initialized = True
            logger.info(f"Initialized vector service for user {self.user_id}, conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Error initializing vector service: {e}")
            self.enabled = False
    
    async def _create_entity_manager(self):
        """Create and initialize the entity manager with caching"""
        # Cache key for entity manager
        cache_key = f"entity_manager:{self.user_id}:{self.conversation_id}"
        
        # Function to create entity manager if not in cache
        async def create_manager():
            # Create vector database
            vector_db = create_vector_database(self.vector_db_config)
            
            # Initialize it
            await vector_db.initialize()
            
            # Create entity manager
            entity_manager = RPGEntityManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                vector_db_config=self.vector_db_config,
                embedding_service=None  # We'll handle embeddings separately
            )
            
            # Initialize it
            await entity_manager.initialize()
            
            return entity_manager
        
        # Get from cache or create new with 15 minute TTL (level 2 cache)
        return await context_cache.get(
            cache_key, 
            create_manager, 
            cache_level=2, 
            ttl_override=900
        )
    
    async def close(self):
        """Close the vector service"""
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
            self.batch_task = None
            
        if self.entity_manager:
            try:
                await self.entity_manager.close()
                self.entity_manager = None
                self.initialized = False
                logger.info(f"Closed vector service for user {self.user_id}, conversation {self.conversation_id}")
            except Exception as e:
                logger.error(f"Error closing vector service: {e}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text with caching
        """
        # Check if in local memory cache first (fastest)
        cache_key = f"embed:{hash(text)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Function to generate embedding
        async def generate_embedding():
            # Use entity_manager's embedding service if available
            if hasattr(self.entity_manager, 'embedding_service') and self.entity_manager.embedding_service:
                return await self.entity_manager.embedding_service.get_embedding(text)
            else:
                # Fallback to random embedding for testing
                vec = list(np.random.normal(0, 1, 384))
                return vec / np.linalg.norm(vec)
        
        # Get from cache or generate new (cache for 1 day in L3)
        embedding = await context_cache.get(
            cache_key, 
            generate_embedding, 
            cache_level=3, 
            ttl_override=86400,  # 1 day TTL for embeddings
            compress=True  # Compress embeddings to save space
        )
        
        # Store in local memory cache too
        self.embedding_cache[cache_key] = embedding
        
        # Keep local cache manageable
        if len(self.embedding_cache) > 1000:
            # Remove random 20% of items when cache gets too big
            keys_to_remove = np.random.choice(
                list(self.embedding_cache.keys()), 
                size=200, 
                replace=False
            )
            for key in keys_to_remove:
                self.embedding_cache.pop(key, None)
        
        return embedding
    
    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts with batching
        """
        results = []
        futures = []
        
        # Create a task for each text
        for text in texts:
            future = asyncio.create_task(self.get_embedding(text))
            futures.append(future)
        
        # Gather all results
        results = await asyncio.gather(*futures)
        return results
    
    async def add_to_batch_queue(self, operation: str, data: Dict[str, Any]) -> str:
        """
        Add an operation to the batch queue
        """
        if not self.enabled:
            return "vector_search_disabled"
            
        # Generate a unique ID for this request
        request_id = f"{operation}_{hash(str(data))}_{time.time()}"
        
        # Create a future that will be resolved when the operation completes
        future = asyncio.Future()
        
        # Add to queue
        await self.batch_queue.put({
            "id": request_id,
            "operation": operation,
            "data": data,
            "future": future
        })
        
        return request_id, future
    
    async def _process_batch_queue(self):
        """
        Process the batch queue in the background
        """
        while True:
            try:
                # Get the next batch (up to 10 items or wait max 100ms)
                batch = []
                try:
                    # Get first item (with timeout)
                    batch.append(await asyncio.wait_for(self.batch_queue.get(), 0.1))
                    
                    # Get more items if available (no waiting)
                    while len(batch) < 10 and not self.batch_queue.empty():
                        batch.append(self.batch_queue.get_nowait())
                except asyncio.TimeoutError:
                    # No items in queue within timeout
                    if not batch:
                        await asyncio.sleep(0.1)  # Avoid tight loop
                        continue
                        
                # Skip if no batch items
                if not batch:
                    continue
                    
                # Group operations
                operations = {}
                for item in batch:
                    op = item["operation"]
                    if op not in operations:
                        operations[op] = []
                    operations[op].append(item)
                
                # Process each operation type
                for op, items in operations.items():
                    if op == "search_entities":
                        await self._process_batch_search(items)
                    elif op == "add_entity":
                        await self._process_batch_add(items)
                    else:
                        # Handle unknown operations individually
                        for item in items:
                            try:
                                result = await self._process_single_operation(item)
                                item["future"].set_result(result)
                            except Exception as e:
                                item["future"].set_exception(e)
                            finally:
                                self.batch_queue.task_done()
                
            except asyncio.CancelledError:
                # Task was cancelled
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(1)  # Avoid tight error loop
    
    async def _process_batch_search(self, items):
        """
        Process a batch of search operations
        """
        try:
            # Initialize entity manager if needed
            await self.initialize()
            
            if not self.entity_manager:
                # Set all futures with error
                for item in items:
                    item["future"].set_result([])
                    self.batch_queue.task_done()
                return
            
            # Group by entity type for more efficient searching
            by_entity_type = {}
            for item in items:
                entity_types = item["data"].get("entity_types", ["npc", "location", "memory"])
                entity_types_key = "-".join(sorted(entity_types))
                
                if entity_types_key not in by_entity_type:
                    by_entity_type[entity_types_key] = []
                by_entity_type[entity_types_key].append(item)
            
            # Process each group
            for entity_types_key, group_items in by_entity_type.items():
                entity_types = entity_types_key.split("-")
                
                # Generate embeddings for all queries at once
                query_texts = [item["data"]["query_text"] for item in group_items]
                embeddings = await self.get_batch_embeddings(query_texts)
                
                # Search for each embedding
                for i, item in enumerate(group_items):
                    query_embedding = embeddings[i]
                    top_k = item["data"].get("top_k", 5)
                    
                    # Do the search
                    results = await self.entity_manager.get_relevant_entities(
                        query_text="",  # Not used when we provide embedding directly
                        top_k=top_k,
                        entity_types=entity_types,
                        query_embedding=query_embedding  # Use precomputed embedding
                    )
                    
                    # Set result
                    item["future"].set_result(results)
                    self.batch_queue.task_done()
        
        except Exception as e:
            # Set exception for all futures
            for item in items:
                item["future"].set_exception(e)
                self.batch_queue.task_done()
    
    async def _process_batch_add(self, items):
        """
        Process a batch of add operations
        """
        try:
            # Initialize entity manager if needed
            await self.initialize()
            
            if not self.entity_manager:
                # Set all futures with error
                for item in items:
                    item["future"].set_result(False)
                    self.batch_queue.task_done()
                return
            
            # Group by entity type for more efficient processing
            by_entity_type = {}
            for item in items:
                entity_type = item["data"].get("entity_type", "unknown")
                
                if entity_type not in by_entity_type:
                    by_entity_type[entity_type] = []
                by_entity_type[entity_type].append(item)
            
            # Process each entity type
            for entity_type, group_items in by_entity_type.items():
                if entity_type == "memory":
                    # Get all content for batch embedding
                    contents = [item["data"]["content"] for item in group_items]
                    embeddings = await self.get_batch_embeddings(contents)
                    
                    # Add each memory with its embedding
                    for i, item in enumerate(group_items):
                        data = item["data"]
                        
                        # Use pre-computed embedding
                        result = await self.entity_manager.add_memory(
                            memory_id=data["memory_id"],
                            content=data["content"],
                            memory_type=data.get("memory_type", "observation"),
                            importance=data.get("importance", 0.5),
                            tags=data.get("tags", []),
                            embedding=embeddings[i]  # Use precomputed embedding
                        )
                        
                        # Set result
                        item["future"].set_result(result)
                        self.batch_queue.task_done()
                
                elif entity_type == "npc":
                    # Handle NPCs
                    # Similar pattern for other entity types...
                    for item in group_items:
                        # Process individually for now
                        result = await self._process_single_operation(item)
                        item["future"].set_result(result)
                        self.batch_queue.task_done()
                
                else:
                    # Handle other entity types individually
                    for item in group_items:
                        result = await self._process_single_operation(item)
                        item["future"].set_result(result)
                        self.batch_queue.task_done()
        
        except Exception as e:
            # Set exception for all futures
            for item in items:
                item["future"].set_exception(e)
                self.batch_queue.task_done()
    
    async def _process_single_operation(self, item):
        """
        Process a single operation
        """
        # Initialize entity manager if needed
        await self.initialize()
        
        if not self.entity_manager:
            return None
            
        operation = item["operation"]
        data = item["data"]
        
        # Dispatch to appropriate method
        if operation == "search_entities":
            query_text = data.get("query_text", "")
            top_k = data.get("top_k", 5)
            entity_types = data.get("entity_types", ["npc", "location", "memory"])
            
            # Get embedding for query
            query_embedding = await self.get_embedding(query_text)
            
            # Do the search
            return await self.entity_manager.get_relevant_entities(
                query_text="",  # Not used when we provide embedding directly
                top_k=top_k,
                entity_types=entity_types,
                query_embedding=query_embedding
            )
        
        elif operation == "add_entity":
            entity_type = data.get("entity_type")
            
            if entity_type == "memory":
                # Add memory
                content = data.get("content", "")
                memory_id = data.get("memory_id", "")
                memory_type = data.get("memory_type", "observation")
                importance = data.get("importance", 0.5)
                tags = data.get("tags", [])
                
                # Get embedding
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
            else:
                # Handle other entity types
                # (implementation for other types would go here)
                logger.warning(f"Unsupported entity type: {entity_type}")
                return False
        
        else:
            logger.warning(f"Unknown operation: {operation}")
            return None
    
    async def search_entities(
        self,
        query_text: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by query text (API method)
        """
        if not self.enabled:
            return []
            
        # Use the batch queue
        request_id, future = await self.add_to_batch_queue(
            "search_entities",
            {
                "query_text": query_text,
                "entity_types": entity_types or ["npc", "location", "memory"],
                "top_k": top_k
            }
        )
        
        # Wait for the result
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
        Add a memory to the vector database (API method)
        """
        if not self.enabled:
            return False
            
        # Use the batch queue
        request_id, future = await self.add_to_batch_queue(
            "add_entity",
            {
                "entity_type": "memory",
                "memory_id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "tags": tags or []
            }
        )
        
        # Wait for the result
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
        Get relevant context for input text with caching
        """
        if not self.enabled:
            return {}
            
        # Cache key based on input and location
        cache_key = f"context:{self.user_id}:{self.conversation_id}:{input_text}:{current_location}"
        
        async def fetch_context():
            # Initialize if needed
            await self.initialize()
            
            if not self.entity_manager:
                return {}
                
            # Combine input with location for better context
            query = input_text
            if current_location:
                query += f" Location: {current_location}"
            
            # Get relevant entities through search
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
        
        # Get from cache or fetch (30 second TTL, importance based on query length)
        importance = min(1.0, len(input_text) / 200)  # Longer queries = more important
        return await context_cache.get(
            cache_key, 
            fetch_context, 
            cache_level=1,  # L1 cache for short term
            importance=importance,
            ttl_override=30  # Short TTL for context
        )

# Global function to get or create vector service
async def get_vector_service(user_id: int, conversation_id: int) -> OptimizedVectorService:
    """
    Get or create a vector service instance
    """
    global _vector_managers
    
    key = f"{user_id}:{conversation_id}"
    
    if key not in _vector_managers:
        # Create new instance
        service = OptimizedVectorService(user_id, conversation_id)
        # Don't await initialize here - do it lazily when needed
        _vector_managers[key] = service
    
    return _vector_managers[key]

# Cleanup function
async def cleanup_vector_services():
    """Close all vector services"""
    global _vector_managers
    
    close_tasks = []
    for key, service in list(_vector_managers.items()):
        close_tasks.append(asyncio.create_task(service.close()))
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    _vector_managers.clear()
