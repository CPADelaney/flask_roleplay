# context/vector_service.py

import asyncio
import logging
import time
import hashlib
import math
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Agent SDK imports
from agents import Agent, function_tool, RunContextWrapper, trace, RunConfig
from pydantic import BaseModel, Field

from context.unified_cache import context_cache
from context.context_config import get_config

from context.models import (
    EntityMetadata, NPCMetadata, LocationMetadata,
    MemoryMetadata
)

logger = logging.getLogger(__name__)


# --- Pydantic Models for Vector Service ---

class VectorSearchResultItem(BaseModel):
    """Individual search result item"""
    id: str
    score: float
    entity_type: str
    metadata: Union[NPCMetadata, LocationMetadata, MemoryMetadata, EntityMetadata]
    
    class Config:
        extra = "forbid"


class VectorContextResult(BaseModel):
    """Result for context retrieval"""
    npcs: List['NPCContextItem'] = []
    locations: List['LocationContextItem'] = []
    memories: List['MemoryContextItem'] = []
    narratives: List['NarrativeContextItem'] = []
    
    class Config:
        extra = "forbid"


class NPCContextItem(BaseModel):
    """NPC context item"""
    npc_id: str
    npc_name: str
    description: Optional[str] = None
    personality: Optional[str] = None
    location: Optional[str] = None
    relevance: float
    
    class Config:
        extra = "forbid"


class LocationContextItem(BaseModel):
    """Location context item"""
    location_id: str
    location_name: str
    description: Optional[str] = None
    connected_locations: List[str] = []
    relevance: float
    
    class Config:
        extra = "forbid"


class MemoryContextItem(BaseModel):
    """Memory context item"""
    memory_id: str
    content: str
    memory_type: str
    importance: float
    relevance: float
    
    class Config:
        extra = "forbid"


class NarrativeContextItem(BaseModel):
    """Narrative context item"""
    narrative_id: str
    content: str
    narrative_type: str
    importance: float
    relevance: float
    
    class Config:
        extra = "forbid"


class VectorSearchRequest(BaseModel):
    """Request model for vector search"""
    query_text: str
    entity_types: Optional[List[str]] = None
    top_k: int = 5
    hybrid_ranking: bool = True
    recency_weight: float = 0.3


class VectorSearchResult(BaseModel):
    """Result model for vector search"""
    query: str
    results: List[VectorSearchResultItem] = []
    total_found: int = 0
    entity_types_searched: List[str] = []
    search_time_ms: float = 0.0


# Update EntityVectorRequest
class EntityVectorRequest(BaseModel):
    """Request model for adding an entity vector"""
    entity_type: str
    entity_id: str
    content: str
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Union[NPCMetadata, LocationMetadata, EntityMetadata]] = None


class MemoryVectorRequest(BaseModel):
    """Request model for adding a memory vector"""
    memory_id: str
    content: str
    memory_type: str = "observation"
    importance: float = 0.5
    tags: Optional[List[str]] = None

class VectorService:
    """
    Unified vector service with a simplified API and optimized performance.
    Refactored to use the OpenAI Agents SDK. 
    **Internal** methods do not require RunContextWrapper.
    """

    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the vector service"""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = None  # Will be initialized asynchronously
        self.entity_manager = None
        self.initialized = False
        self.embedding_cache = {}
        self.batch_queue = asyncio.Queue()
        self.batch_task = None
        self.enabled = False  # Will be set during initialization
    
    async def initialize(self):
        """Initialize the vector service (async)"""
        if self.initialized:
            return
        try:
            # Get configuration
            self.config = await get_config()
            
            # Check if vector search is enabled
            self.enabled = (
                self.config.get("vector_db", "enabled", True)
                and self.config.is_enabled("use_vector_search")
            )
            if not self.enabled:
                logger.info(
                    f"Vector service disabled for user {self.user_id}, "
                    f"conversation {self.conversation_id}"
                )
                return
            
            # Import here to avoid circular imports
            from context.optimized_db import RPGEntityManager, create_vector_database
            
            # Create database
            vector_db_config = self.config.get_vector_db_config()
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
            logger.info(
                f"Initialized vector service for user {self.user_id}, "
                f"conversation {self.conversation_id}"
            )
        except Exception as e:
            logger.error(f"Error initializing vector service: {e}")
            self.enabled = False
    
    async def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        if not self.initialized:
            await self.initialize()
        return self.initialized and self.enabled
    
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
        logger.info(
            f"Closed vector service for user {self.user_id}, "
            f"conversation {self.conversation_id}"
        )
    
    # ---------------------------------------------------------------------
    #           INTERNAL (PRIVATE) METHODS (no @function_tool, have `self`)
    # ---------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Internal method to get an embedding for text with caching.
        No run_context here; keeps function_tool away from instance methods.
        """
        with trace(workflow_name="get_embedding"):
            if not self.enabled:
                # Return a zeroed vector if not enabled
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
                if (
                    self.entity_manager
                    and hasattr(self.entity_manager, "embedding_service")
                    and self.entity_manager.embedding_service
                ):
                    return await self.entity_manager.embedding_service.get_embedding(text)
                
                # Fallback to random embedding for testing
                vec = list(np.random.normal(0, 1, 384))
                norm = np.linalg.norm(vec)
                if norm == 0:
                    return vec
                return list(vec / norm)
            
            # Use unified cache to fetch or generate
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
        """Process batched vector operations in the background"""
        while True:
            try:
                batch = []
                
                try:
                    # Grab the first item with a short wait
                    first_item = await asyncio.wait_for(self.batch_queue.get(), timeout=0.1)
                    batch.append(first_item)
                    
                    # Grab more items if immediately available (no wait)
                    while len(batch) < 10 and not self.batch_queue.empty():
                        batch.append(self.batch_queue.get_nowait())
                except asyncio.TimeoutError:
                    # Nothing arrived in 0.1s
                    await asyncio.sleep(0.01)
                    continue
                
                if not batch:
                    continue
                
                # Process each item in the batch
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
                        
                        # Complete future
                        item["future"].set_result(result)
                    except Exception as e:
                        item["future"].set_exception(e)
                    finally:
                        self.batch_queue.task_done()
            
            except asyncio.CancelledError:
                logger.debug("Batch queue processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)  # avoid tight error loop
    
    async def _perform_search(self, data) -> List[VectorSearchResultItem]:
        """Perform a vector search operation."""
        if not self.entity_manager:
            await self.initialize()
            if not self.entity_manager:
                return []
        
        query_text = data.get("query_text", "")
        entity_types = data.get("entity_types", ["npc", "location", "memory"])
        top_k = data.get("top_k", 5)
        
        # Get embedding for query
        query_embedding = await self._get_embedding(query_text)
        
        # Perform search
        raw_results = await self.entity_manager.search_entities(
            query_text="",  # Not used when embedding is provided
            top_k=top_k,
            entity_types=entity_types,
            query_embedding=query_embedding
        )
        
        # Convert to typed results
        typed_results = []
        for result in raw_results:
            metadata_dict = result.get("metadata", {})
            entity_type = metadata_dict.get("entity_type", "")
            
            # Create appropriate metadata object
            if entity_type == "npc":
                metadata = NPCMetadata(**metadata_dict)
            elif entity_type == "location":
                metadata = LocationMetadata(**metadata_dict)
            elif entity_type == "memory":
                metadata = MemoryMetadata(**metadata_dict)
            else:
                metadata = EntityMetadata(**metadata_dict)
            
            typed_results.append(VectorSearchResultItem(
                id=result.get("id", ""),
                score=result.get("score", 0.0),
                entity_type=entity_type,
                metadata=metadata
            ))
        
        # Apply hybrid ranking if requested
        if data.get("hybrid_ranking", False):
            typed_results = self._apply_hybrid_ranking(
                typed_results,
                recency_weight=data.get("recency_weight", 0.3)
            )
        
        return typed_results
    
    def _apply_hybrid_ranking(self, results: List[VectorSearchResultItem], recency_weight=0.3) -> List[VectorSearchResultItem]:
        """Apply a hybrid ranking with vector similarity and recency."""
        from datetime import datetime
        now = datetime.now()
        
        # Calculate hybrid scores
        for result in results:
            base_score = result.score
            
            # Calculate temporal score if timestamp available
            temporal_score = 0.5
            if result.metadata.timestamp:
                try:
                    timestamp = datetime.fromisoformat(result.metadata.timestamp.replace("Z", "+00:00"))
                    days_diff = (now - timestamp).days
                    # Newer items get higher scores
                    temporal_score = math.exp(-0.1 * max(0, days_diff))
                except:
                    pass
            
            vector_weight = 1.0 - recency_weight
            hybrid_score = (vector_weight * base_score) + (recency_weight * temporal_score)
            # Store hybrid score in metadata for sorting
            result.score = hybrid_score
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    async def _add_memory(self, data):
        """Add a memory to the vector database."""
        if not self.entity_manager:
            await self.initialize()
            if not self.entity_manager:
                return False
        
        memory_id = data.get("memory_id", "")
        content = data.get("content", "")
        memory_type = data.get("memory_type", "observation")
        importance = data.get("importance", 0.5)
        tags = data.get("tags", [])
        
        # Get embedding
        embedding = await self._get_embedding(content)
        
        return await self.entity_manager.add_memory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            embedding=embedding
        )
    
    async def _add_entity(self, data):
        """Add an entity to the vector database."""
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
        embedding = await self._get_embedding(embed_text)
        
        return await self.entity_manager.add_entity(
            entity_type=entity_type,
            entity_id=entity_id,
            content=content,
            embedding=embedding,
            **{
                k: v
                for k, v in data.items()
                if k not in ["entity_type", "entity_id", "content"]
            }
        )
    
    async def _get_context_for_input(
        self,
        input_text: str,
        current_location: Optional[str] = None,
        max_items: int = 10
    ) -> VectorContextResult:
        """
        Internal method to get relevant context for input text.
        """
        if not await self.is_initialized():
            return VectorContextResult()
        
        cache_key = (
            f"vector_context:{self.user_id}:{self.conversation_id}:"
            f"{input_text}:{current_location}"
        )
        
        async def fetch_context():
            query = input_text
            if current_location:
                query += f" Location: {current_location}"
            
            # Build search request
            search_request = VectorSearchRequest(
                query_text=query,
                entity_types=["npc", "location", "memory", "narrative"],
                top_k=max_items
            )
            
            # Perform search
            start_time = time.time()
            future = asyncio.get_event_loop().create_future()
            await self.batch_queue.put({
                "operation": "search",
                "data": {
                    "query_text": search_request.query_text,
                    "entity_types": search_request.entity_types,
                    "top_k": search_request.top_k,
                    "hybrid_ranking": search_request.hybrid_ranking,
                    "recency_weight": search_request.recency_weight,
                },
                "future": future
            })
            
            try:
                results = await future
            except Exception as e:
                logger.error(f"Error performing search in get_context_for_input: {e}")
                return VectorContextResult()
            
            context = VectorContextResult()
            
            for result in results:
                if isinstance(result, VectorSearchResultItem):
                    metadata = result.metadata
                    entity_type = result.entity_type
                else:
                    # Handle raw dict results from entity manager
                    metadata = result.get("metadata", {})
                    entity_type = metadata.get("entity_type")
                
                if entity_type == "npc":
                    context.npcs.append(NPCContextItem(
                        npc_id=metadata.get("npc_id") or metadata.get("entity_id", ""),
                        npc_name=metadata.get("npc_name") or metadata.get("name", ""),
                        description=metadata.get("description"),
                        personality=metadata.get("personality"),
                        location=metadata.get("location"),
                        relevance=result.get("score", 0.5) if isinstance(result, dict) else result.score
                    ))
                elif entity_type == "location":
                    context.locations.append(LocationContextItem(
                        location_id=metadata.get("location_id") or metadata.get("entity_id", ""),
                        location_name=metadata.get("location_name") or metadata.get("name", ""),
                        description=metadata.get("description"),
                        connected_locations=metadata.get("connected_locations", []),
                        relevance=result.get("score", 0.5) if isinstance(result, dict) else result.score
                    ))
                elif entity_type == "memory":
                    context.memories.append(MemoryContextItem(
                        memory_id=metadata.get("memory_id") or metadata.get("entity_id", ""),
                        content=metadata.get("content", ""),
                        memory_type=metadata.get("memory_type", ""),
                        importance=metadata.get("importance", 0.5),
                        relevance=result.get("score", 0.5) if isinstance(result, dict) else result.score
                    ))
                elif entity_type == "narrative":
                    context.narratives.append(NarrativeContextItem(
                        narrative_id=metadata.get("narrative_id") or metadata.get("entity_id", ""),
                        content=metadata.get("content", ""),
                        narrative_type=metadata.get("narrative_type", ""),
                        importance=metadata.get("importance", 0.5),
                        relevance=result.get("score", 0.5) if isinstance(result, dict) else result.score
                    ))
            
            return context
        
        importance = min(0.7, 0.3 + (len(input_text) / 100))
        return await context_cache.get(
            cache_key,
            fetch_context,
            cache_level=1,
            importance=importance,
            ttl_override=30
        )


# ---------------------------------------------------------------------
#            STANDALONE TOOL FUNCTIONS (run_context first param)
# ---------------------------------------------------------------------

# Global registry for VectorService instances
_vector_services: Dict[str, VectorService] = {}

async def get_vector_service(user_id: int, conversation_id: int) -> VectorService:
    """Get or create a vector service instance."""
    global _vector_services
    key = f"{user_id}:{conversation_id}"
    if key not in _vector_services:
        service = VectorService(user_id, conversation_id)
        await service.initialize()
        _vector_services[key] = service
    return _vector_services[key]

async def cleanup_vector_services():
    """Close all vector services."""
    global _vector_services
    close_tasks = []
    for key, service in list(_vector_services.items()):
        close_tasks.append(asyncio.create_task(service.close()))
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    _vector_services.clear()


# ---------------------------------------------------------------------
#  "function_tool" Decorated Methods with ctx FIRST, calling private methods
# ---------------------------------------------------------------------

@function_tool
async def get_embedding_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    text: str
) -> List[float]:
    """
    Tool: Get embedding vector for text (with caching).
    
    The first param is ctx (RunContextWrapper), 
    so we comply with the requirement that run_context is the first param.
    """
    service = await get_vector_service(user_id, conversation_id)
    return await service._get_embedding(text)

@function_tool
async def search_entities_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    request: VectorSearchRequest
) -> VectorSearchResult:
    """
    Tool: Search for entities by query text.
    """
    service = await get_vector_service(user_id, conversation_id)
    
    if not await service.is_initialized():
        return VectorSearchResult(
            query=request.query_text,
            results=[],
            total_found=0,
            entity_types_searched=request.entity_types or [],
            search_time_ms=0.0
        )
    
    start_time = time.time()
    
    future = asyncio.get_event_loop().create_future()
    await service.batch_queue.put({
        "operation": "search",
        "data": {
            "query_text": request.query_text,
            "entity_types": request.entity_types or ["npc", "location", "memory"],
            "top_k": request.top_k,
            "hybrid_ranking": request.hybrid_ranking,
            "recency_weight": request.recency_weight,
        },
        "future": future
    })
    
    try:
        results = await future
        elapsed_ms = (time.time() - start_time) * 1000
        
        return VectorSearchResult(
            query=request.query_text,
            results=results,
            total_found=len(results),
            entity_types_searched=request.entity_types or ["npc", "location", "memory"],
            search_time_ms=elapsed_ms
        )
    except Exception as e:
        logger.error(f"Error in search_entities_tool: {e}")
        return VectorSearchResult(
            query=request.query_text,
            results=[],
            total_found=0,
            entity_types_searched=request.entity_types or [],
            search_time_ms=(time.time() - start_time) * 1000,
        )

@function_tool
async def add_vector_memory_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    request: MemoryVectorRequest
) -> bool:
    """
    Tool: Add a memory to the vector database.
    """
    service = await get_vector_service(user_id, conversation_id)
    if not await service.is_initialized():
        return False
    
    future = asyncio.get_event_loop().create_future()
    await service.batch_queue.put({
        "operation": "add_memory",
        "data": {
            "memory_id": request.memory_id,
            "content": request.content,
            "memory_type": request.memory_type,
            "importance": request.importance,
            "tags": request.tags or []
        },
        "future": future
    })
    
    try:
        return await future
    except Exception as e:
        logger.error(f"Error in add_vector_memory_tool: {e}")
        return False

@function_tool
async def add_entity_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    request: EntityVectorRequest
) -> bool:
    """
    Tool: Add an entity to the vector database.
    """
    service = await get_vector_service(user_id, conversation_id)
    if not await service.is_initialized():
        return False
    
    future = asyncio.get_event_loop().create_future()
    data = {
        "entity_type": request.entity_type,
        "entity_id": request.entity_id,
        "content": request.content,
    }
    if request.name:
        data["name"] = request.name
    if request.description:
        data["description"] = request.description
    if request.metadata:
        # Convert Pydantic model to dict
        data.update(request.metadata.dict(exclude_none=True))
    
    await service.batch_queue.put({
        "operation": "add_entity",
        "data": data,
        "future": future
    })
    
    try:
        return await future
    except Exception as e:
        logger.error(f"Error in add_entity_tool: {e}")
        return False

@function_tool
async def get_context_for_input_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    input_text: str,
    current_location: Optional[str] = None,
    max_items: int = 10
) -> VectorContextResult:
    """
    Tool: Get relevant context for input text, possibly including location.
    """
    service = await get_vector_service(user_id, conversation_id)
    return await service._get_context_for_input(
        input_text=input_text,
        current_location=current_location,
        max_items=max_items
    )

# ---------------------------------------------------------------------
#        Create the Vector Agent (register tool functions only)
# ---------------------------------------------------------------------

def create_vector_agent() -> Agent:
    """Create a vector search agent using the OpenAI Agents SDK."""
    agent = Agent(
        name="Vector Search",
        instructions="""
        You are a vector search agent specialized in semantic search and retrieval.
        Your tasks include:
        
        1. Searching for relevant entities based on semantic similarity
        2. Adding vectors for memories and entities
        3. Retrieving context based on input text
        
        When handling vector operations, prioritize semantic relevance and recency.
        """,
        # Register only the top-level tool functions here:
        tools=[
            get_embedding_tool,
            search_entities_tool,
            add_vector_memory_tool,
            add_entity_tool,
            get_context_for_input_tool,
        ],
    )
    return agent

def get_vector_agent() -> Agent:
    """Get or create a vector agent instance."""
    return create_vector_agent()
