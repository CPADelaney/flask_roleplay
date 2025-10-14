# context/vector_service.py

import asyncio
import logging
import time
import hashlib
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Agent SDK imports
from agents import Agent, function_tool, RunContextWrapper, trace, RunConfig
from pydantic import BaseModel, Field

from context.unified_cache import context_cache
from context.context_config import get_config
from db.read import read_entity_cards, read_recent_chunks

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
    card: Optional[Dict[str, Any]] = None
    
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

        self.collections = {
            "npc": "npc_embeddings",
            "location": "location_embeddings", 
            "memory": "memory_embeddings",
            "narrative": "narrative_embeddings"
        }
    
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
    
    async def add_memory(self, memory_id: str, content: str, memory_type: str = "observation", 
                         importance: float = 0.5, tags: Optional[List[str]] = None) -> bool:
        """
        Public method to add a memory to the vector database.
        
        Args:
            memory_id: Unique identifier for the memory
            content: Memory content text
            memory_type: Type of memory
            importance: Importance score (0-1)
            tags: Optional list of tags
            
        Returns:
            bool: Success status
        """
        if not await self.is_initialized():
            return False
        
        future = asyncio.get_event_loop().create_future()
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
        
        try:
            return await future
        except Exception as e:
            logger.error(f"Error in add_memory: {e}")
            return False
    
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

    async def get_context_for_input(
        self,
        input_text: str,
        current_location: Optional[str] = None,
        max_items: int = 10
    ) -> Dict[str, Any]:
        """
        Public method to get relevant context for input text.
        
        Args:
            input_text: Current input text
            current_location: Optional current location
            max_items: Maximum number of context items
            
        Returns:
            Dict with relevant context from vector database
        """
        # Call the internal method and convert result to dict
        result = await self._get_context_for_input(
            input_text=input_text,
            current_location=current_location,
            max_items=max_items
        )
        
        # Convert VectorContextResult to dict for compatibility
        if hasattr(result, 'dict'):
            return result.dict()
        elif hasattr(result, 'model_dump'):
            return result.model_dump()
        else:
            # If it's already a dict or has a compatible structure
            return {
                "npcs": [item.dict() if hasattr(item, 'dict') else item for item in result.npcs],
                "locations": [item.dict() if hasattr(item, 'dict') else item for item in result.locations],
                "memories": [item.dict() if hasattr(item, 'dict') else item for item in result.memories],
                "narratives": [item.dict() if hasattr(item, 'dict') else item for item in result.narratives]
            }
    
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
                return [0.0] * 1536
            
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
                vec = list(np.random.normal(0, 1, 1536))
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
                            result = await self.add_entity(item["data"])
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

    @staticmethod
    def _sanitize_score(value: Optional[float]) -> float:
        """Clamp scores into [0, 1] and guard against None/NaN."""
        if value is None:
            return 0.0
        try:
            if math.isnan(value):  # type: ignore[arg-type]
                return 0.0
        except TypeError:
            pass
        return float(max(0.0, min(1.0, value)))

    def _compute_recency_boost(
        self,
        updated_at: Optional[datetime],
        decay: timedelta = timedelta(days=3)
    ) -> float:
        """Compute a recency boost (0-1) favouring fresher rows."""
        if not updated_at:
            return 0.0

        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_seconds = max((now - updated_at).total_seconds(), 0.0)
        decay_seconds = max(decay.total_seconds(), 1.0)
        return math.exp(-age_seconds / decay_seconds)

    def _compute_hybrid_score(
        self,
        vector_score: Optional[float],
        text_score: Optional[float],
        updated_at: Optional[datetime],
        decay: timedelta = timedelta(days=3)
    ) -> float:
        """Combine vector, text, and recency scores with fixed weights."""
        vector_component = self._sanitize_score(vector_score)
        text_component = self._sanitize_score(text_score)
        recency_component = self._compute_recency_boost(updated_at, decay)
        return (0.6 * vector_component) + (0.3 * text_component) + (0.1 * recency_component)

    def _score_entity_rows(
        self,
        rows: List[Dict[str, Any]],
        decay: timedelta = timedelta(days=3)
    ) -> List[Dict[str, Any]]:
        """Attach hybrid scores to raw rows and sort descending."""
        scored_rows: List[Dict[str, Any]] = []
        for row in rows:
            updated_at = row.get("updated_at")
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at)
                except ValueError:
                    updated_at = None
            score = self._compute_hybrid_score(
                row.get("vector_score"),
                row.get("text_score"),
                updated_at,
                decay,
            )
            enriched = dict(row)
            enriched["score"] = score
            enriched["updated_at"] = updated_at
            scored_rows.append(enriched)

        scored_rows.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored_rows

    async def _query_entity_cards(
        self,
        query_embedding: List[float],
        query_text: str,
        entity_types: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fetch entity cards scored via pgvector + text search."""

        if not entity_types:
            entity_types = ["npc", "location", "memory"]

        limit = max(top_k * 4, top_k)
        ts_query = query_text.strip()

        embedding_array = list(query_embedding)

        try:
            rows = await read_entity_cards(
                self.user_id,
                self.conversation_id,
                embedding=embedding_array,
                query_text=ts_query,
                entity_types=entity_types,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - safety for missing DB
            logger.debug("Hybrid entity card query failed: %s", exc)
            return []

        scored = self._score_entity_rows([dict(row) for row in rows])
        return scored[:top_k]

    async def _query_recent_chunks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent episodic chunks for the conversation."""
        try:
            rows = await read_recent_chunks(
                self.user_id,
                self.conversation_id,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - safety for missing DB
            logger.debug("Recent chunk query failed: %s", exc)
            return []

        return [dict(row) for row in rows]

    async def get_entity_cards(
        self,
        query_text: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Public helper returning hybrid-ranked entity cards."""

        if entity_types is None:
            entity_types = ["npc", "location", "memory"]

        query_embedding = await self._get_embedding(query_text)
        return await self._query_entity_cards(query_embedding, query_text, entity_types, top_k)

    async def fetch_recent_chunks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Public helper returning recent episodic chunks."""

        return await self._query_recent_chunks(limit)

    async def _perform_search(self, data) -> List[VectorSearchResultItem]:
        """Perform a vector search operation."""
        query_text = data.get("query_text", "")
        entity_types = data.get("entity_types", ["npc", "location", "memory"])
        top_k = data.get("top_k", 5)

        # Get embedding for query
        query_embedding = await self._get_embedding(query_text)

        hybrid_rows = await self._query_entity_cards(
            query_embedding=query_embedding,
            query_text=query_text,
            entity_types=entity_types,
            top_k=top_k,
        )

        raw_results: List[Dict[str, Any]] = []

        if hybrid_rows:
            raw_results = hybrid_rows
        elif self.entity_manager:
            # Fallback to legacy vector service if available
            raw_results = await self.entity_manager.search_entities(
                query_text="",
                top_k=top_k,
                entity_types=entity_types,
                query_embedding=query_embedding
            )
        else:
            return []

        # Convert to typed results
        typed_results = []
        for result in raw_results:
            metadata_source: Dict[str, Any] = {}
            card = None
            entity_type: str = ""

            if isinstance(result, dict):
                card = result.get("card")
                entity_type = result.get("entity_type", "")
                metadata_source = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
                if not entity_type and metadata_source:
                    entity_type = metadata_source.get("entity_type", "")
            else:
                try:
                    metadata_source = result.get("metadata", {})  # type: ignore[assignment]
                except AttributeError:
                    metadata_source = {}
                entity_type = metadata_source.get("entity_type", "") if isinstance(metadata_source, dict) else ""

            # FIX: Filter metadata_dict to only include fields relevant to each metadata type
            if entity_type == "npc":
                card_data = card or metadata_source.get("card", {}) if isinstance(metadata_source, dict) else {}
                personality_traits = card_data.get("personality_traits") if isinstance(card_data, dict) else None
                relationships = card_data.get("relationships") if isinstance(card_data, dict) else None
                metadata = NPCMetadata(
                    personality=", ".join(personality_traits) if isinstance(personality_traits, list) else personality_traits,
                    location=card_data.get("location") if isinstance(card_data, dict) else None,
                    tags=relationships if isinstance(relationships, list) else None,
                )
            elif entity_type == "location":
                card_data = card or metadata_source.get("card", {}) if isinstance(metadata_source, dict) else {}
                metadata = LocationMetadata(
                    location_type=card_data.get("location_type") if isinstance(card_data, dict) else None,
                    attributes=card_data.get("notable_features") if isinstance(card_data, dict) else None,
                )
            elif entity_type == "memory":
                card_data = card or metadata_source.get("card", {}) if isinstance(metadata_source, dict) else {}
                metadata = MemoryMetadata(
                    context_type=card_data.get("memory_type") if isinstance(card_data, dict) else None,
                    tags=card_data.get("tags") if isinstance(card_data, dict) else None,
                )
            else:
                metadata = EntityMetadata()

            if isinstance(result, dict):
                score = result.get("score", result.get("vector_score", 0.0))
                entity_id = result.get("entity_id", "")
                card_payload = result.get("card")
            else:
                score = result.get("score", 0.0)
                entity_id = result.get("id", "")
                card_payload = metadata_source.get("card") if isinstance(metadata_source, dict) else None

            typed_results.append(VectorSearchResultItem(
                id=str(entity_id),
                score=float(score or 0.0),
                entity_type=entity_type,
                metadata=metadata,
                card=card_payload
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
    
    async def add_entity(
        self,
        entity_type: str,          # "npc", "location", "memory", "narrative"
        entity_id: str,            # Unique identifier for this entity
        content: str,              # Text content for embedding generation
        embedding: Optional[List[float]] = None,  # Pre-computed embedding (optional)
        **metadata                 # Any additional metadata as keyword arguments
    ) -> bool:
        """
        Add an entity to the vector database (generic method used by all entity types)
        
        This is the core method that handles:
        1. Validation of entity type
        2. Embedding generation (if not provided)
        3. Metadata enrichment
        4. Storage in the appropriate vector collection
        
        Args:
            entity_type: Type of entity (must be in self.collections)
            entity_id: Unique ID for the entity
            content: Text content for the entity (used for embedding if not provided)
            embedding: Optional pre-computed embedding vector (1536 dimensions)
            **metadata: Additional metadata for the entity (varies by entity type)
            
        Returns:
            bool: True if successfully added, False otherwise
            
        Example:
            await add_entity(
                entity_type="memory",
                entity_id="mem_123",
                content="The player discovered a hidden treasure",
                importance=0.8,
                tags=["treasure", "discovery"],
                location="Dark Cave"
            )
        """
        # Step 1: Validate entity type
        if entity_type not in self.collections:
            logger.error(f"Unsupported entity type: {entity_type}")
            return False
        
        # Step 2: Get the collection name for this entity type
        collection_name = self.collections[entity_type]
        # e.g., "memory" -> "memory_embeddings"
        
        # Step 3: Generate embedding if not provided
        if embedding is None:
            if self.embedding_service:
                try:
                    # Use the embedding service if available
                    embedding = await self.embedding_service.get_embedding(content)
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    # Fallback: Generate random 1536-dimensional embedding
                    vec = list(np.random.normal(0, 1, 1536))
                    embedding = vec / np.linalg.norm(vec)
            else:
                # No embedding service, use random embedding
                vec = list(np.random.normal(0, 1, 1536))
                embedding = vec / np.linalg.norm(vec)
        
        # Step 4: Build complete metadata
        # Always include these core fields
        full_metadata = {
            "user_id": self.user_id,           # From class instance
            "conversation_id": self.conversation_id,  # From class instance
            "entity_type": entity_type,        # Type of entity
            "entity_id": entity_id,            # Unique ID
            "content": content,                # Original text content
            **metadata                         # All additional metadata passed in
        }
        
        # Step 5: Store in vector database
        # The ID format is "{entity_type}_{entity_id}" for uniqueness
        return await self.vector_db.insert_vectors(
            collection_name=collection_name,
            ids=[f"{entity_type}_{entity_id}"],
            vectors=[embedding],
            metadata=[full_metadata]
        )
    
    async def search_entities(
        self,
        query_text: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Public method to search for entities by query text.
        
        Args:
            query_text: Query text for search
            entity_types: List of entity types to search (e.g., ["memory", "npc"])
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of search results with metadata
        """
        if not await self.is_initialized():
            return []
        
        # Get embedding for query
        query_embedding = await self._get_embedding(query_text)
        
        # Use entity manager to perform the search
        if self.entity_manager:
            return await self.entity_manager.search_entities(
                query_text=query_text,
                query_embedding=query_embedding,
                entity_types=entity_types,
                top_k=top_k,
                filter_dict=filter_dict
            )
        
        return []
    
    async def add_entity(
        self,
        entity_type: str,
        entity_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        **metadata
    ) -> bool:
        """
        Public method to add an entity to the vector database.
        
        Args:
            entity_type: Type of entity (npc, location, memory, narrative)
            entity_id: Unique ID for the entity
            content: Text content for the entity
            embedding: Optional pre-computed embedding
            **metadata: Additional metadata for the entity
            
        Returns:
            bool: Success status
        """
        if not await self.is_initialized():
            return False
        
        future = asyncio.get_event_loop().create_future()
        data = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "content": content,
            "embedding": embedding,
            **metadata
        }
        
        await self.batch_queue.put({
            "operation": "add_entity",
            "data": data,
            "future": future
        })
        
        try:
            return await future
        except Exception as e:
            logger.error(f"Error in add_entity: {e}")
            return False
    
    async def add_npc(
        self,
        npc_id: str,
        npc_name: str,
        description: str = "",
        personality: str = "",
        location: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        **extra_metadata
    ) -> bool:
        """
        Public method to add an NPC to the vector database.
        
        Args:
            npc_id: Unique identifier for the NPC
            npc_name: Name of the NPC
            description: Physical description
            personality: Personality traits
            location: Current location
            embedding: Optional pre-computed embedding
            **extra_metadata: Additional metadata fields
            
        Returns:
            bool: Success status
        """
        if not await self.is_initialized():
            return False
        
        # Create content for embedding if not provided
        content = f"NPC: {npc_name}. {description} {personality}"
        
        return await self.add_entity(
            entity_type="npc",
            entity_id=npc_id,
            content=content,
            embedding=embedding,
            npc_name=npc_name,
            description=description,
            personality=personality,
            location=location,
            **extra_metadata
        )
    
    async def add_location(
        self,
        location_id: str,
        location_name: str,
        description: str = "",
        location_type: Optional[str] = None,
        connected_locations: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        **extra_metadata
    ) -> bool:
        """
        Public method to add a location to the vector database.
        
        Args:
            location_id: Unique identifier for the location
            location_name: Name of the location
            description: Location description
            location_type: Type of location (city, dungeon, etc.)
            connected_locations: List of connected location IDs
            embedding: Optional pre-computed embedding
            **extra_metadata: Additional metadata fields
            
        Returns:
            bool: Success status
        """
        if not await self.is_initialized():
            return False
        
        # Create content for embedding if not provided
        content = f"Location: {location_name}. {description}"
        
        return await self.add_entity(
            entity_type="location",
            entity_id=location_id,
            content=content,
            embedding=embedding,
            location_name=location_name,
            description=description,
            location_type=location_type,
            connected_locations=connected_locations or [],
            **extra_metadata
        )
    
    async def add_narrative(
        self,
        narrative_id: str,
        content: str,
        narrative_type: str = "story",
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        **extra_metadata
    ) -> bool:
        """
        Public method to add a narrative element to the vector database.
        
        Args:
            narrative_id: Unique identifier for the narrative
            content: Narrative content text
            narrative_type: Type of narrative (story, quest, etc.)
            importance: Importance score (0-1)
            embedding: Optional pre-computed embedding
            **extra_metadata: Additional metadata fields
            
        Returns:
            bool: Success status
        """
        if not await self.is_initialized():
            return False
        
        return await self.add_entity(
            entity_type="narrative",
            entity_id=narrative_id,
            content=content,
            embedding=embedding,
            narrative_type=narrative_type,
            importance=importance,
            **extra_metadata
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
                    metadata_model = result.metadata
                    entity_type = result.entity_type
                    card_payload = result.card or {}
                    if hasattr(metadata_model, "model_dump"):
                        metadata = metadata_model.model_dump()
                    elif hasattr(metadata_model, "dict"):
                        metadata = metadata_model.dict()
                    else:
                        metadata = dict(metadata_model) if isinstance(metadata_model, dict) else {}
                else:
                    # Handle raw dict results from entity manager
                    metadata = result.get("metadata", {})
                    entity_type = metadata.get("entity_type")
                    card_payload = result.get("card", {})

                if entity_type == "npc":
                    card_data = card_payload if isinstance(card_payload, dict) else {}
                    personality = metadata.get("personality") or \
                        (", ".join(card_data.get("personality_traits", []))
                         if isinstance(card_data.get("personality_traits"), list) else card_data.get("personality"))
                    context.npcs.append(NPCContextItem(
                        npc_id=str(card_data.get("npc_id") or metadata.get("npc_id") or metadata.get("entity_id", "")),
                        npc_name=card_data.get("npc_name") or metadata.get("npc_name") or metadata.get("name", ""),
                        description=card_data.get("description") or metadata.get("description"),
                        personality=personality,
                        location=card_data.get("location") or metadata.get("location"),
                        relevance=result.get("score", 0.5) if isinstance(result, dict) else result.score
                    ))
                elif entity_type == "location":
                    card_data = card_payload if isinstance(card_payload, dict) else {}
                    context.locations.append(LocationContextItem(
                        location_id=str(card_data.get("location_id") or metadata.get("location_id") or metadata.get("entity_id", "")),
                        location_name=card_data.get("location_name") or metadata.get("location_name") or metadata.get("name", ""),
                        description=card_data.get("description") or metadata.get("description"),
                        connected_locations=card_data.get("notable_features")
                        if isinstance(card_data.get("notable_features"), list)
                        else metadata.get("connected_locations", []),
                        relevance=result.get("score", 0.5) if isinstance(result, dict) else result.score
                    ))
                elif entity_type == "memory":
                    card_data = card_payload if isinstance(card_payload, dict) else {}
                    context.memories.append(MemoryContextItem(
                        memory_id=str(card_data.get("memory_id") or metadata.get("memory_id") or metadata.get("entity_id", "")),
                        content=card_data.get("content") or metadata.get("content", ""),
                        memory_type=card_data.get("memory_type") or metadata.get("memory_type", ""),
                        importance=float(card_data.get("importance") or metadata.get("importance", 0.5)),
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
