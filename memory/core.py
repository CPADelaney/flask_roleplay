# memory/core.py

import time
import logging
import asyncpg
import asyncio
import json
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from functools import wraps
from dataclasses import dataclass, field

# Local embedding model (alternative to OpenAI)
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

# For OpenAI fallback
import openai
from openai import AsyncOpenAI

# Configure logging
logger = logging.getLogger("memory_system")

# Telemetry setup
from .telemetry import MemoryTelemetry

# Database configuration
from config import DB_CONFIG, EMBEDDING_CONFIG

# Constants
EMBEDDING_DIMENSION = 1536  # OpenAI Ada-002 dimensionality
EMBEDDING_MODEL = EMBEDDING_CONFIG.get("model", "text-embedding-ada-002")
EMBEDDING_BATCH_SIZE = EMBEDDING_CONFIG.get("batch_size", 5)
MEMORY_CACHE_TTL = 300  # 5 minutes

class MemoryType(str, Enum):
    """Types of memories in the system."""
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    CONSOLIDATED = "consolidated"
    PROCEDURAL = "procedural"
    
class MemoryStatus(str, Enum):
    """Status of a memory in its lifecycle."""
    ACTIVE = "active"
    SUMMARIZED = "summarized"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MemorySignificance(int, Enum):
    """Significance levels for memories."""
    TRIVIAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class Memory:
    """Data class representing a memory."""
    id: Optional[int] = None
    text: str = ""
    memory_type: MemoryType = MemoryType.OBSERVATION
    significance: int = MemorySignificance.MEDIUM
    emotional_intensity: int = 0
    status: MemoryStatus = MemoryStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    last_recalled: Optional[datetime] = None
    times_recalled: int = 0
    is_consolidated: bool = False
    
    @property
    def age_days(self) -> float:
        """Calculate memory age in days."""
        if not self.timestamp:
            return 0
        return (datetime.now() - self.timestamp).total_seconds() / 86400
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        result = {
            "text": self.text,
            "memory_type": self.memory_type.value if isinstance(self.memory_type, Enum) else self.memory_type,
            "significance": self.significance,
            "emotional_intensity": self.emotional_intensity,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "tags": self.tags,
            "metadata": self.metadata,
            "times_recalled": self.times_recalled,
            "is_consolidated": self.is_consolidated,
        }
        
        if self.id is not None:
            result["id"] = self.id
            
        if self.timestamp:
            result["timestamp"] = self.timestamp.isoformat()
            
        if self.last_recalled:
            result["last_recalled"] = self.last_recalled.isoformat()
            
        if self.embedding:
            result["has_embedding"] = True
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary."""
        memory = cls(
            id=data.get("id"),
            text=data.get("text", ""),
            memory_type=data.get("memory_type", MemoryType.OBSERVATION),
            significance=data.get("significance", MemorySignificance.MEDIUM),
            emotional_intensity=data.get("emotional_intensity", 0),
            status=data.get("status", MemoryStatus.ACTIVE),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            times_recalled=data.get("times_recalled", 0),
            is_consolidated=data.get("is_consolidated", False)
        )
        
        # Process timestamps
        if "timestamp" in data:
            if isinstance(data["timestamp"], str):
                memory.timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            else:
                memory.timestamp = data["timestamp"]
                
        if "last_recalled" in data:
            if isinstance(data["last_recalled"], str):
                memory.last_recalled = datetime.fromisoformat(data["last_recalled"].replace("Z", "+00:00"))
            else:
                memory.last_recalled = data["last_recalled"]
                
        return memory
    
    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'Memory':
        """Create memory from database row."""
        memory = cls(
            id=row.get("id"),
            text=row.get("memory_text", ""),
            memory_type=row.get("memory_type", MemoryType.OBSERVATION),
            significance=row.get("significance", MemorySignificance.MEDIUM),
            emotional_intensity=row.get("emotional_intensity", 0),
            status=MemoryStatus(row.get("status", "active")),
            tags=row.get("tags", []),
            embedding=row.get("embedding"),
            times_recalled=row.get("times_recalled", 0),
            is_consolidated=row.get("is_consolidated", False)
        )
        
        # Parse metadata if it exists
        if "metadata" in row and row["metadata"]:
            memory.metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
        
        # Process timestamps
        memory.timestamp = row.get("timestamp", datetime.now())
        memory.last_recalled = row.get("last_recalled")
                
        return memory


class MemoryCache:
    """
    Memory cache implementation with time-based expiration.
    Thread-safe for use in async code.
    """
    def __init__(self, ttl_seconds: int = MEMORY_CACHE_TTL):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            # Check expiration
            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp > self._ttl:
                # Expired - remove it
                del self._cache[key]
                del self._timestamps[key]
                return None
                
            return self._cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set item in cache with current timestamp."""
        async with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    async def delete(self, key: str) -> None:
        """Delete item from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()


class EmbeddingProvider:
    """
    Abstract base class for embedding providers.
    Allows swapping between different embedding models.
    """
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        raise NotImplementedError("Subclasses must implement get_embedding")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        raise NotImplementedError("Subclasses must implement get_embeddings")


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider using SentenceTransformers library."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise ImportError("SentenceTransformers is not installed. Install with: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding using SentenceTransformers (runs in thread pool)."""
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, lambda: self.model.encode(text))
        return embedding.tolist()
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (runs in thread pool)."""
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, lambda: self.model.encode(texts))
        return embeddings.tolist()


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider using OpenAI API."""
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self._last_call = 0
        self._min_interval = 0.1  # 100ms between API calls
    
    async def _rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_call = time.time()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API."""
        await self._rate_limit()
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            # Return zero vector as fallback
            return [0.0] * EMBEDDING_DIMENSION
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        await self._rate_limit()
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            # Return zero vectors as fallback
            return [[0.0] * EMBEDDING_DIMENSION for _ in texts]


class FallbackEmbedding(EmbeddingProvider):
    """
    Embedding provider with prioritized fallback.
    Tries local embedding first, falls back to OpenAI if needed.
    """
    def __init__(self):
        self.providers = []
        
        # Try to initialize local provider first
        try:
            self.providers.append(SentenceTransformerEmbedding())
            logger.info("Using SentenceTransformers for embeddings")
        except (ImportError, Exception) as e:
            logger.warning(f"SentenceTransformers unavailable: {e}")
        
        # Add OpenAI provider as fallback
        try:
            self.providers.append(OpenAIEmbedding())
            logger.info("Added OpenAI fallback for embeddings")
        except Exception as e:
            logger.warning(f"OpenAI embedding unavailable: {e}")
            
        if not self.providers:
            # Last resort fallback
            logger.error("No embedding providers available! Using zero vectors.")
            
    async def get_embedding(self, text: str) -> List[float]:
        """Try each provider in sequence until one works."""
        for provider in self.providers:
            try:
                return await provider.get_embedding(text)
            except Exception as e:
                logger.warning(f"Embedding provider {provider.__class__.__name__} failed: {e}")
                
        # If all providers fail, return zero vector
        logger.error("All embedding providers failed")
        return [0.0] * EMBEDDING_DIMENSION
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Try each provider in sequence until one works."""
        for provider in self.providers:
            try:
                return await provider.get_embeddings(texts)
            except Exception as e:
                logger.warning(f"Batch embedding provider {provider.__class__.__name__} failed: {e}")
                
        # If all providers fail, return zero vectors
        logger.error("All batch embedding providers failed")
        return [[0.0] * EMBEDDING_DIMENSION for _ in texts]


class DBConnectionManager:
    """
    Manages database connections with a connection pool.
    """
    _pool = None
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get or create the connection pool."""
        async with cls._lock:
            if cls._pool is None:
                try:
                    cls._pool = await asyncpg.create_pool(
                        dsn=DB_CONFIG["dsn"],
                        min_size=DB_CONFIG.get("min_connections", 5),
                        max_size=DB_CONFIG.get("max_connections", 20),
                        command_timeout=DB_CONFIG.get("command_timeout", 60),
                        statement_cache_size=DB_CONFIG.get("statement_cache_size", 0),  # Disabled by default to prevent issues
                        max_inactive_connection_lifetime=DB_CONFIG.get("max_inactive_connection_lifetime", 300)
                    )
                    logger.info("Database connection pool created")
                except Exception as e:
                    logger.critical(f"Failed to create database connection pool: {e}")
                    raise
            return cls._pool
    
    @classmethod
    async def close_pool(cls) -> None:
        """Close the connection pool."""
        async with cls._lock:
            if cls._pool is not None:
                await cls._pool.close()
                cls._pool = None
                logger.info("Database connection pool closed")
                
    @classmethod
    async def acquire(cls) -> asyncpg.Connection:
        """Acquire a connection from the pool."""
        pool = await cls.get_pool()
        return await pool.acquire()
    
    @classmethod
    async def release(cls, connection: asyncpg.Connection) -> None:
        """Release a connection back to the pool."""
        pool = await cls.get_pool()
        await pool.release(connection)


def with_transaction(func):
    """
    Decorator to handle transaction management.
    Ensures proper error handling and transaction rollback.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Get connection
        conn = kwargs.pop('conn', None)
        external_conn = conn is not None
        
        if not external_conn:
            conn = await DBConnectionManager.acquire()
        
        # Start transaction tracking
        start_time = time.time()
        
        try:
            # Start transaction
            if not external_conn:
                tx = conn.transaction()
                await tx.start()
            
            # Call the actual function
            result = await func(self, *args, conn=conn, **kwargs)
            
            # Commit transaction
            if not external_conn:
                await tx.commit()
            
            # Record telemetry
            elapsed = time.time() - start_time
            await MemoryTelemetry.record(
                operation=func.__name__,
                success=True,
                duration=elapsed,
                data_size=1  # Default value, replace with actual size if available
            )
            
            return result
            
        except Exception as e:
            # Rollback transaction on error
            if not external_conn:
                try:
                    await tx.rollback()
                except Exception as rollback_e:
                    logger.error(f"Error during transaction rollback: {rollback_e}")
            
            # Record error telemetry
            elapsed = time.time() - start_time
            await MemoryTelemetry.record(
                operation=func.__name__,
                success=False,
                duration=elapsed,
                error=str(e)
            )
            
            # Log error
            logger.error(f"Transaction error in {func.__name__}: {e}")
            raise
            
        finally:
            # Release connection back to pool
            if not external_conn:
                await DBConnectionManager.release(conn)
    
    return wrapper


class UnifiedMemoryManager:
    """
    Unified memory management system.
    Provides consistent interface for all memory operations.
    """
    def __init__(self, 
                 entity_type: str, 
                 entity_id: int, 
                 user_id: int, 
                 conversation_id: int):
        """
        Initialize the memory manager.
        
        Args:
            entity_type: Type of entity (npc, player, nyx)
            entity_id: ID of the entity
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Cache for memory operations to reduce database load
        self.cache = MemoryCache(ttl_seconds=MEMORY_CACHE_TTL)
        
        # Embedding provider with fallback options
        self.embedding_provider = FallbackEmbedding()
    
    @with_transaction
    async def add_memory(self, 
                        memory: Union[Memory, str],
                        memory_type: MemoryType = MemoryType.OBSERVATION,
                        significance: int = MemorySignificance.MEDIUM,
                        emotional_intensity: int = 0,
                        tags: List[str] = None,
                        metadata: Dict[str, Any] = None,
                        embedding: List[float] = None,
                        conn: Optional[asyncpg.Connection] = None) -> int:
        """
        Add a new memory.
        
        Args:
            memory: Memory object or text string
            memory_type: Type of memory
            significance: Importance of memory (1-5)
            emotional_intensity: Emotional impact (0-100)
            tags: List of tags for categorization
            metadata: Additional structured data
            embedding: Pre-computed embedding vector
            conn: Optional existing database connection
            
        Returns:
            ID of created memory
        """
        # If memory is a string, convert to Memory object
        if isinstance(memory, str):
            memory = Memory(
                text=memory,
                memory_type=memory_type,
                significance=significance,
                emotional_intensity=emotional_intensity,
                tags=tags or [],
                metadata=metadata or {},
                timestamp=datetime.now()
            )
        
        # Generate embedding if not provided
        if not memory.embedding and not embedding:
            embedding = await self.embedding_provider.get_embedding(memory.text)
        
        # Set timestamp if not present
        if not memory.timestamp:
            memory.timestamp = datetime.now()
        
        # Insert memory into database
        memory_id = await conn.fetchval("""
            INSERT INTO unified_memories (
                entity_type, entity_id, user_id, conversation_id,
                memory_text, memory_type, significance, emotional_intensity,
                tags, embedding, metadata, timestamp, times_recalled,
                status, is_consolidated
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            RETURNING id
        """, 
            self.entity_type, self.entity_id, self.user_id, self.conversation_id,
            memory.text, memory.memory_type, memory.significance, memory.emotional_intensity,
            memory.tags, embedding or memory.embedding, json.dumps(memory.metadata),
            memory.timestamp, memory.times_recalled,
            memory.status.value if isinstance(memory.status, Enum) else memory.status,
            memory.is_consolidated
        )
        
        # Invalidate relevant cache entries
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        
        # For significant memories, process special handling
        if memory.significance >= MemorySignificance.HIGH:
            await self._process_significant_memory(memory, memory_id, conn)
            
        # Return memory ID
        return memory_id
    
    async def _process_significant_memory(self, 
                                         memory: Memory,
                                         memory_id: int,
                                         conn: asyncpg.Connection) -> None:
        """
        Process a significant memory with special handling.
        - Create semantic abstraction 
        - Propagate to connected entities
        - Update narrative arcs
        """
        # 1. Create semantic abstraction for high significance memories
        if memory.significance >= MemorySignificance.HIGH:
            await self._create_semantic_memory(memory, memory_id, conn)
            
        # 2. Propagate memory to connected entities
        if memory.significance >= MemorySignificance.MEDIUM:
            await self._propagate_memory(memory, conn)
    
    async def _create_semantic_memory(self, 
                                     memory: Memory,
                                     source_id: int,
                                     conn: asyncpg.Connection) -> Optional[int]:
        """
        Create a semantic abstraction from an episodic/observation memory.
        """
        # Simple abstraction for demonstration
        # In production, you might use an LLM to generate this
        semantic_text = f"Abstract understanding: {memory.text}"
        
        # Create semantic memory
        semantic_memory = Memory(
            text=semantic_text,
            memory_type=MemoryType.SEMANTIC,
            significance=max(MemorySignificance.MEDIUM, memory.significance - 1),
            emotional_intensity=max(0, memory.emotional_intensity - 20),
            tags=memory.tags + ["semantic", "abstraction"],
            metadata={"source_memory_id": source_id},
            timestamp=datetime.now()
        )
        
        # Generate embedding
        embedding = await self.embedding_provider.get_embedding(semantic_text)
        
        # Insert semantic memory
        semantic_id = await conn.fetchval("""
            INSERT INTO unified_memories (
                entity_type, entity_id, user_id, conversation_id,
                memory_text, memory_type, significance, emotional_intensity,
                tags, embedding, metadata, timestamp, times_recalled,
                status, is_consolidated
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            RETURNING id
        """, 
            self.entity_type, self.entity_id, self.user_id, self.conversation_id,
            semantic_memory.text, semantic_memory.memory_type.value, semantic_memory.significance, 
            semantic_memory.emotional_intensity, semantic_memory.tags, embedding,
            json.dumps(semantic_memory.metadata), semantic_memory.timestamp, 0,
            MemoryStatus.ACTIVE.value, False
        )
        
        return semantic_id
        
    async def _propagate_memory(self, 
                               memory: Memory, 
                               conn: asyncpg.Connection) -> None:
        """
        Propagate memory to connected entities based on relationships.
        """
        # Find connected entities through relationships
        rows = await conn.fetch("""
            SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
            FROM social_links
            WHERE user_id = $1 AND conversation_id = $2
            AND ((entity1_type = $3 AND entity1_id = $4) OR (entity2_type = $3 AND entity2_id = $4))
            AND link_level >= 30
        """, self.user_id, self.conversation_id, self.entity_type, self.entity_id)
        
        if not rows:
            return
            
        # For each connected entity, create a modified memory
        for row in rows:
            # Determine target entity
            target_type, target_id = (row['entity2_type'], row['entity2_id']) \
                if (row['entity1_type'] == self.entity_type and row['entity1_id'] == self.entity_id) \
                else (row['entity1_type'], row['entity1_id'])
            
            # Skip if target is the original entity
            if target_type == self.entity_type and target_id == self.entity_id:
                continue
                
            # Create propagated memory
            derived_text = f"Heard that {memory.text}"
            
            propagated_memory = Memory(
                text=derived_text,
                memory_type=MemoryType.OBSERVATION,
                significance=max(MemorySignificance.TRIVIAL, memory.significance - 2),
                emotional_intensity=max(0, memory.emotional_intensity - 30),
                tags=memory.tags + ["secondhand", "propagated"],
                metadata={
                    "source_entity_type": self.entity_type,
                    "source_entity_id": self.entity_id,
                    "source_memory_text": memory.text
                },
                timestamp=datetime.now()
            )
            
            # Create new memory manager for target entity
            target_manager = UnifiedMemoryManager(
                entity_type=target_type,
                entity_id=target_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            # Add memory to target entity
            try:
                await target_manager.add_memory(propagated_memory, conn=conn)
            except Exception as e:
                logger.error(f"Error propagating memory to {target_type}:{target_id}: {e}")
    
    @with_transaction
    async def retrieve_memories(self,
                              query: str = None,
                              memory_types: List[str] = None,
                              tags: List[str] = None,
                              min_significance: int = 0,
                              limit: int = 10,
                              exclude_ids: List[int] = None,
                              context: Dict[str, Any] = None,
                              use_embedding: bool = True,
                              conn: Optional[asyncpg.Connection] = None) -> List[Memory]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Query string for relevance matching
            memory_types: Types of memories to retrieve
            tags: Tags to filter by
            min_significance: Minimum significance threshold
            limit: Maximum number of memories to return
            exclude_ids: Memory IDs to exclude
            context: Additional context for relevance calculation
            use_embedding: Whether to use embedding similarity
            conn: Optional existing database connection
            
        Returns:
            List of Memory objects
        """
        memory_types = memory_types or [t.value for t in MemoryType]
        exclude_ids = exclude_ids or []
        context = context or {}
        
        # Try cache first if no specific filters
        if not query and not tags and not exclude_ids and min_significance == 0:
            cache_key = f"memories_{self.entity_type}_{self.entity_id}"
            cached = await self.cache.get(cache_key)
            if cached:
                return cached
        
        # Base query without embedding similarity
        if use_embedding and query:
            # Generate embedding for query
            query_embedding = await self.embedding_provider.get_embedding(query)
            
            # Query with embedding similarity
            rows = await conn.fetch("""
                SELECT id, memory_text, memory_type, significance, emotional_intensity,
                       tags, metadata, timestamp, times_recalled, last_recalled,
                       status, is_consolidated,
                       embedding <-> $1 AS similarity
                FROM unified_memories
                WHERE entity_type = $2
                AND entity_id = $3
                AND user_id = $4
                AND conversation_id = $5
                AND memory_type = ANY($6)
                AND significance >= $7
                AND id != ALL($8)
                AND status != 'deleted'
                ORDER BY similarity
                LIMIT $9
            """, 
                query_embedding, self.entity_type, self.entity_id, 
                self.user_id, self.conversation_id, memory_types,
                min_significance, exclude_ids, limit
            )
        else:
            # Query without embedding similarity
            rows = await conn.fetch("""
                SELECT id, memory_text, memory_type, significance, emotional_intensity,
                       tags, metadata, timestamp, times_recalled, last_recalled,
                       status, is_consolidated
                FROM unified_memories
                WHERE entity_type = $1
                AND entity_id = $2
                AND user_id = $3
                AND conversation_id = $4
                AND memory_type = ANY($5)
                AND significance >= $6
                AND id != ALL($7)
                AND status != 'deleted'
                ORDER BY significance DESC, timestamp DESC
                LIMIT $8
            """, 
                self.entity_type, self.entity_id, 
                self.user_id, self.conversation_id, memory_types,
                min_significance, exclude_ids, limit
            )
            
        # Convert rows to Memory objects
        memories = [Memory.from_db_row(dict(row)) for row in rows]
        
        # Apply tag filtering if specified
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
            
        # Apply context-based relevance calculation if available
        if context and memories:
            memories = await self._calculate_contextual_relevance(memories, context)
        
        # Update memory access stats in batch
        if memories:
            memory_ids = [m.id for m in memories]
            await self._update_memory_access_batch(memory_ids, conn)
            
        # Store in cache if it's a general query
        if not query and not tags and not exclude_ids and min_significance == 0:
            cache_key = f"memories_{self.entity_type}_{self.entity_id}"
            await self.cache.set(cache_key, memories)
            
        return memories
    
    async def _calculate_contextual_relevance(self, 
                                            memories: List[Memory],
                                            context: Dict[str, Any]) -> List[Memory]:
        """
        Calculate contextual relevance scores for memories.
        Sorts memories by relevance to current context.
        """
        # Extract context elements
        location = context.get("location", "")
        time_of_day = context.get("time_of_day", "")
        
        # Score each memory
        for memory in memories:
            relevance_score = 0
            
            # Base score is significance
            relevance_score += memory.significance * 10
            
            # Boost for recency
            if memory.timestamp:
                days_old = (datetime.now() - memory.timestamp).total_seconds() / 86400
                recency_boost = max(0, 10 - days_old) 
                relevance_score += recency_boost
                
            # Boost for frequent recall
            relevance_score += min(20, memory.times_recalled * 2)
            
            # Boost for emotional intensity
            relevance_score += memory.emotional_intensity / 5
            
            # Boost for location match
            if location and location in memory.text:
                relevance_score += 15
                
            # Boost for time match
            if time_of_day and time_of_day in memory.text:
                relevance_score += 10
                
            # Store score in memory object
            memory.metadata["relevance_score"] = relevance_score
            
        # Sort by relevance score
        memories.sort(key=lambda m: m.metadata.get("relevance_score", 0), reverse=True)
        
        return memories
    
    async def _update_memory_access_batch(self, 
                                        memory_ids: List[int],
                                        conn: asyncpg.Connection) -> None:
        """
        Update access stats for multiple memories in a single batch operation.
        """
        if not memory_ids:
            return
            
        # Update times_recalled and last_recalled in batch
        await conn.execute("""
            UPDATE unified_memories
            SET times_recalled = times_recalled + 1,
                last_recalled = CURRENT_TIMESTAMP
            WHERE id = ANY($1)
        """, memory_ids)
    
    @with_transaction
    async def consolidate_memories(self, 
                                  age_threshold_days: int = 7,
                                  min_count: int = 3,
                                  min_similarity: float = 0.8,
                                  conn: Optional[asyncpg.Connection] = None) -> List[int]:
        """
        Consolidate similar memories into higher-level patterns.
        
        Args:
            age_threshold_days: Minimum age for memories to consider
            min_count: Minimum number of memories needed for consolidation
            min_similarity: Minimum similarity threshold (0.0-1.0)
            conn: Optional existing database connection
            
        Returns:
            List of IDs of newly created consolidated memories
        """
        # Get memories for potential consolidation
        age_cutoff = datetime.now() - timedelta(days=age_threshold_days)
        
        rows = await conn.fetch("""
            SELECT id, memory_text, memory_type, significance, emotional_intensity,
                   tags, embedding, metadata, timestamp
            FROM unified_memories
            WHERE entity_type = $1
            AND entity_id = $2
            AND user_id = $3
            AND conversation_id = $4
            AND timestamp < $5
            AND is_consolidated = FALSE
            AND status = 'active'
            AND memory_type != 'consolidated'
            ORDER BY timestamp
        """, 
            self.entity_type, self.entity_id, 
            self.user_id, self.conversation_id,
            age_cutoff
        )
        
        if len(rows) < min_count:
            return []
            
        # Convert to Memory objects
        memories = [Memory.from_db_row(dict(row)) for row in rows]
        
        # Find clusters of similar memories
        clusters = await self._cluster_memories(memories, min_similarity, min_count)
        
        # Create consolidated memories from clusters
        consolidated_ids = []
        
        for cluster in clusters:
            if len(cluster) < min_count:
                continue
                
            # Get all memory IDs in this cluster
            memory_ids = [m.id for m in cluster]
            memory_texts = [m.text for m in cluster]
            
            # Collect all tags
            all_tags = set()
            for memory in cluster:
                all_tags.update(memory.tags)
            
            # Create consolidated memory
            consolidated_text = f"Consolidated pattern from {len(cluster)} memories: {'; '.join(memory_texts[:3])}..."
            
            consolidated_memory = Memory(
                text=consolidated_text,
                memory_type=MemoryType.CONSOLIDATED,
                significance=max(m.significance for m in cluster),
                emotional_intensity=int(sum(m.emotional_intensity for m in cluster) / len(cluster)),
                tags=list(all_tags) + ["consolidated"],
                metadata={"source_memory_ids": memory_ids},
                timestamp=datetime.now()
            )
            
            # Generate embedding
            embedding = await self.embedding_provider.get_embedding(consolidated_text)
            
            # Insert consolidated memory
            consolidated_id = await conn.fetchval("""
                INSERT INTO unified_memories (
                    entity_type, entity_id, user_id, conversation_id,
                    memory_text, memory_type, significance, emotional_intensity,
                    tags, embedding, metadata, timestamp, times_recalled,
                    status, is_consolidated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                RETURNING id
            """, 
                self.entity_type, self.entity_id, self.user_id, self.conversation_id,
                consolidated_memory.text, consolidated_memory.memory_type.value, 
                consolidated_memory.significance, consolidated_memory.emotional_intensity,
                consolidated_memory.tags, embedding, json.dumps(consolidated_memory.metadata),
                consolidated_memory.timestamp, 0, MemoryStatus.ACTIVE.value, False
            )
            
            consolidated_ids.append(consolidated_id)
            
            # Mark original memories as consolidated (but don't delete them)
            await conn.execute("""
                UPDATE unified_memories
                SET is_consolidated = TRUE
                WHERE id = ANY($1)
            """, memory_ids)
            
        # Invalidate cache
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        
        return consolidated_ids
    
    async def _cluster_memories(self, 
                              memories: List[Memory],
                              similarity_threshold: float,
                              min_cluster_size: int) -> List[List[Memory]]:
        """
        Cluster memories based on embedding similarity.
        Uses a greedy approach for simplicity.
        """
        clusters = []
        unprocessed = memories.copy()
        
        while unprocessed:
            # Take first memory as seed
            seed = unprocessed.pop(0)
            current_cluster = [seed]
            
            i = 0
            while i < len(unprocessed):
                memory = unprocessed[i]
                
                # Calculate similarity (dot product)
                if seed.embedding and memory.embedding:
                    similarity = np.dot(seed.embedding, memory.embedding)
                    
                    if similarity >= similarity_threshold:
                        current_cluster.append(memory)
                        unprocessed.pop(i)
                    else:
                        i += 1
                else:
                    i += 1
            
            # Only keep clusters meeting minimum size
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
        
        return clusters
    
    @with_transaction
    async def apply_memory_decay(self, 
                               age_threshold_days: int = 30,
                               recall_threshold: int = 3,
                               decay_rate: float = 0.2,
                               conn: Optional[asyncpg.Connection] = None) -> int:
        """
        Apply decay to old memories to simulate forgetting.
        
        Args:
            age_threshold_days: Minimum age for memories to consider
            recall_threshold: Minimum recalls to protect from decay
            decay_rate: Rate of significance decay (0.0-1.0)
            conn: Optional existing database connection
            
        Returns:
            Number of affected memories
        """
        # Get memories for potential decay
        age_cutoff = datetime.now() - timedelta(days=age_threshold_days)
        
        rows = await conn.fetch("""
            SELECT id, significance, times_recalled, 
                   EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400 AS days_old
            FROM unified_memories
            WHERE entity_type = $1
            AND entity_id = $2
            AND user_id = $3
            AND conversation_id = $4
            AND timestamp < $5
            AND status = 'active'
            AND significance > 1
        """, 
            self.entity_type, self.entity_id, 
            self.user_id, self.conversation_id,
            age_cutoff
        )
        
        if not rows:
            return 0
            
        # Process each memory
        affected_count = 0
        archive_ids = []
        
        for row in rows:
            memory_id = row['id']
            significance = row['significance']
            times_recalled = row['times_recalled']
            days_old = row['days_old']
            
            # Memories recalled frequently decay slower
            recall_factor = min(1.0, times_recalled / recall_threshold)
            age_factor = min(1.0, days_old / 100)  # Max effect at 100 days
            
            # Calculate decay amount
            decay_amount = decay_rate * age_factor * (1 - recall_factor)
            
            # Apply decay with floor of 1
            new_significance = max(1, significance - decay_amount)
            
            # Update database
            if new_significance < significance:
                await conn.execute("""
                    UPDATE unified_memories
                    SET significance = $1
                    WHERE id = $2
                """, new_significance, memory_id)
                
                affected_count += 1
                
                # If decayed to minimum, consider archiving
                if new_significance == 1 and times_recalled <= 1:
                    archive_ids.append(memory_id)
        
        # Archive memories that have decayed to minimum significance
        if archive_ids:
            await conn.execute("""
                UPDATE unified_memories
                SET status = 'archived'
                WHERE id = ANY($1)
            """, archive_ids)
            
        # Invalidate cache
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        
        return affected_count
    
    @with_transaction
    async def reconsolidate_memory(self,
                                  memory_id: int,
                                  alteration_strength: float = 0.1,
                                  conn: Optional[asyncpg.Connection] = None) -> bool:
        """
        Slightly alter a memory to simulate reconsolidation effects.
        
        Args:
            memory_id: ID of memory to reconsolidate
            alteration_strength: Strength of alterations (0.0-1.0)
            conn: Optional existing database connection
            
        Returns:
            True if successful, False otherwise
        """
        # Get the memory
        row = await conn.fetchrow("""
            SELECT memory_text, metadata, memory_type, significance
            FROM unified_memories
            WHERE id = $1
            AND entity_type = $2
            AND entity_id = $3
            AND user_id = $4
            AND conversation_id = $5
        """, 
            memory_id, self.entity_type, self.entity_id, 
            self.user_id, self.conversation_id
        )
        
        if not row:
            return False
            
        memory_text = row['memory_text']
        memory_type = row['memory_type']
        significance = row['significance']
        
        # Parse metadata
        metadata = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'] or '{}')
        
        # Skip reconsolidation for semantic or consolidated memories
        if memory_type in (MemoryType.SEMANTIC.value, MemoryType.CONSOLIDATED.value):
            return False
            
        # Skip reconsolidation for high significance memories (more stable)
        if significance >= MemorySignificance.HIGH:
            return False
            
        # Get original form if available
        original_form = metadata.get("original_form", memory_text)
        
        # Store current version in reconsolidation history
        if "reconsolidation_history" not in metadata:
            metadata["reconsolidation_history"] = []
            
        metadata["reconsolidation_history"].append({
            "previous_text": memory_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 3 versions to avoid metadata bloat
        if len(metadata["reconsolidation_history"]) > 3:
            metadata["reconsolidation_history"] = metadata["reconsolidation_history"][-3:]
            
        # Store original form if not already present
        if "original_form" not in metadata:
            metadata["original_form"] = original_form
        
        # Apply simple word-level alterations
        altered_text = await self._alter_memory_text(memory_text, alteration_strength)
        
        # Generate new embedding
        embedding = await self.embedding_provider.get_embedding(altered_text)
        
        # Update the memory
        await conn.execute("""
            UPDATE unified_memories
            SET memory_text = $1,
                metadata = $2,
                embedding = $3
            WHERE id = $4
        """, altered_text, json.dumps(metadata), embedding, memory_id)
        
        # Invalidate cache
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        
        return True
    
    async def _alter_memory_text(self, text: str, alteration_strength: float) -> str:
        """
        Apply subtle alterations to memory text to simulate reconsolidation.
        Uses simple word-level transformations.
        
        In production, consider using an LLM for more nuanced alterations.
        """
        words = text.split()
        altered_words = words.copy()
        
        for i in range(len(words)):
            # Skip very short words
            if len(words[i]) <= 3:
                continue
                
            # Random chance based on alteration strength
            if random.random() < alteration_strength:
                # Possible alterations (simple approach for demo)
                alterations = [
                    # Intensifier
                    lambda w: f"very {w}" if len(w) > 3 else w,
                    # Softener
                    lambda w: f"somewhat {w}" if len(w) > 3 else w,
                    # Change adjective form
                    lambda w: f"{w}ly" if not w.endswith("ly") and len(w) > 4 else w,
                    # No change (identity function)
                    lambda w: w
                ]
                
                # Choose a random alteration
                alteration = random.choice(alterations)
                altered_words[i] = alteration(words[i])
        
        return " ".join(altered_words)
    
    @with_transaction
    async def archive_old_memories(self,
                                 age_threshold_days: int = 60,
                                 significance_threshold: int = 2,
                                 recall_threshold: int = 2,
                                 conn: Optional[asyncpg.Connection] = None) -> int:
        """
        Archive old, low-significance memories.
        
        Args:
            age_threshold_days: Age threshold for archiving
            significance_threshold: Maximum significance for archiving
            recall_threshold: Maximum recalls for archiving
            conn: Optional existing database connection
            
        Returns:
            Number of archived memories
        """
        # Get memories for potential archiving
        age_cutoff = datetime.now() - timedelta(days=age_threshold_days)
        
        result = await conn.execute("""
            UPDATE unified_memories
            SET status = 'archived'
            WHERE entity_type = $1
            AND entity_id = $2
            AND user_id = $3
            AND conversation_id = $4
            AND timestamp < $5
            AND status = 'active'
            AND significance <= $6
            AND times_recalled <= $7
        """, 
            self.entity_type, self.entity_id, 
            self.user_id, self.conversation_id,
            age_cutoff, significance_threshold, recall_threshold
        )
        
        # Invalidate cache
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        
        # Extract number of affected rows
        affected = int(result.split(" ")[-1]) if " " in result else 0
        
        return affected
    
    @with_transaction
    async def batch_add_memories(self,
                               memories: List[Union[Memory, str]],
                               embed_batch_size: int = EMBEDDING_BATCH_SIZE,
                               conn: Optional[asyncpg.Connection] = None) -> List[int]:
        """
        Add multiple memories in a batch operation.
        
        Args:
            memories: List of Memory objects or strings
            embed_batch_size: Number of embeddings to generate in each batch
            conn: Optional existing database connection
            
        Returns:
            List of created memory IDs
        """
        if not memories:
            return []
            
        # Convert string memories to Memory objects
        memory_objects = []
        for memory in memories:
            if isinstance(memory, str):
                memory_objects.append(Memory(
                    text=memory,
                    memory_type=MemoryType.OBSERVATION,
                    significance=MemorySignificance.MEDIUM,
                    emotional_intensity=0,
                    tags=[],
                    metadata={},
                    timestamp=datetime.now()
                ))
            else:
                memory_objects.append(memory)
        
        # Generate embeddings in batches
        memory_batches = [memory_objects[i:i+embed_batch_size] 
                        for i in range(0, len(memory_objects), embed_batch_size)]
        
        memory_ids = []
        
        for batch in memory_batches:
            # Generate embeddings for batch
            texts = [memory.text for memory in batch]
            embeddings = await self.embedding_provider.get_embeddings(texts)
            
            # Insert memories in batch
            for i, memory in enumerate(batch):
                # Insert memory
                memory_id = await conn.fetchval("""
                    INSERT INTO unified_memories (
                        entity_type, entity_id, user_id, conversation_id,
                        memory_text, memory_type, significance, emotional_intensity,
                        tags, embedding, metadata, timestamp, times_recalled,
                        status, is_consolidated
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    RETURNING id
                """, 
                    self.entity_type, self.entity_id, self.user_id, self.conversation_id,
                    memory.text, 
                    memory.memory_type.value if isinstance(memory.memory_type, Enum) else memory.memory_type, 
                    memory.significance, memory.emotional_intensity,
                    memory.tags, embeddings[i], json.dumps(memory.metadata),
                    memory.timestamp or datetime.now(), memory.times_recalled,
                    memory.status.value if isinstance(memory.status, Enum) else memory.status,
                    memory.is_consolidated
                )
                
                memory_ids.append(memory_id)
        
        # Invalidate cache
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        
        return memory_ids
    
    @with_transaction
    async def perform_maintenance(self,
                                conn: Optional[asyncpg.Connection] = None) -> Dict[str, int]:
        """
        Perform all maintenance operations in one batch.
        
        Args:
            conn: Optional existing database connection
            
        Returns:
            Dictionary with maintenance statistics
        """
        # Step 1: Apply memory decay
        decayed = await self.apply_memory_decay(conn=conn)
        
        # Step 2: Consolidate similar memories
        consolidated = await self.consolidate_memories(conn=conn)
        
        # Step 3: Archive old memories
        archived = await self.archive_old_memories(conn=conn)
        
        return {
            "memories_decayed": decayed,
            "memories_consolidated": len(consolidated),
            "memories_archived": archived
        }
    
    @staticmethod
    @with_transaction
    async def create_tables(conn: Optional[asyncpg.Connection] = None) -> None:
        """
        Create necessary database tables if they don't exist.
        """
        # Create unified_memories table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS unified_memories (
                id SERIAL PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                memory_text TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'observation',
                significance INTEGER NOT NULL DEFAULT 3,
                emotional_intensity INTEGER NOT NULL DEFAULT 0,
                tags TEXT[] DEFAULT '{}',
                embedding VECTOR(1536),
                metadata JSONB,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                times_recalled INTEGER NOT NULL DEFAULT 0,
                last_recalled TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'active',
                is_consolidated BOOLEAN NOT NULL DEFAULT FALSE
            );
            
            -- Create indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_unified_memories_entity ON unified_memories(entity_type, entity_id);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_user_conv ON unified_memories(user_id, conversation_id);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_timestamp ON unified_memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_memory_type ON unified_memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_status ON unified_memories(status);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_significance ON unified_memories(significance);
            
            -- Create vector index for embedding similarity search
            CREATE INDEX IF NOT EXISTS idx_unified_memories_embedding_hnsw ON unified_memories USING hnsw (embedding vector_cosine_ops);
        """)
        
        # Create telemetry table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_telemetry (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                operation TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                duration FLOAT NOT NULL,
                data_size INTEGER,
                error TEXT,
                metadata JSONB
            );
            
            CREATE INDEX IF NOT EXISTS idx_memory_telemetry_timestamp ON memory_telemetry(timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_telemetry_operation ON memory_telemetry(operation);
            CREATE INDEX IF NOT EXISTS idx_memory_telemetry_success ON memory_telemetry(success);
        """)
        
        logger.info("Memory system tables created successfully")
