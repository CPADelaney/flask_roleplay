# memory/core.py

import time
import logging
import asyncpg
import asyncio
import json
import random
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from functools import wraps
from dataclasses import dataclass, field
from collections import OrderedDict

# Attempt to load SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

# For OpenAI fallback
import openai
from openai import AsyncOpenAI

logger = logging.getLogger("memory_system")

# Telemetry setup
# (Assuming there's a .telemetry module with MemoryTelemetry)
from .telemetry import MemoryTelemetry

# Database configuration
from .config import DB_CONFIG, EMBEDDING_CONFIG

EMBEDDING_DIMENSION = 1536  # OpenAI Ada-002 dimensionality
EMBEDDING_MODEL = EMBEDDING_CONFIG.get("model", "text-embedding-ada-002")
EMBEDDING_BATCH_SIZE = EMBEDDING_CONFIG.get("batch_size", 5)
MEMORY_CACHE_TTL = 300  # 5 minutes

class MemoryType(str, Enum):
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    CONSOLIDATED = "consolidated"
    PROCEDURAL = "procedural"

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUMMARIZED = "summarized"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MemorySignificance(int, Enum):
    TRIVIAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class Memory:
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
    # Added to simulate "fidelity" or accuracy, which can degrade over time like human memory
    fidelity: float = 1.0

    @property
    def age_days(self) -> float:
        if not self.timestamp:
            return 0
        return (datetime.now() - self.timestamp).total_seconds() / 86400

    def to_dict(self) -> Dict[str, Any]:
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
            "fidelity": self.fidelity,
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
            is_consolidated=data.get("is_consolidated", False),
            fidelity=data.get("fidelity", 1.0)
        )

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
        memory.timestamp = row.get("timestamp", datetime.now())
        memory.last_recalled = row.get("last_recalled")

        if "metadata" in row and row["metadata"]:
            memory.metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])

        # Initialize fidelity if present in metadata or default
        memory.fidelity = memory.metadata.get("fidelity", 1.0)
        return memory

class MemoryCache:
    def __init__(self, ttl_seconds: int = MEMORY_CACHE_TTL):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            return self._cache[key]

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()

    async def delete(self, key: str) -> None:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()

class EmbeddingProvider:
    async def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement get_embedding")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement get_embeddings")

class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise ImportError("SentenceTransformers is not installed.")
        self.model = SentenceTransformer(model_name)

    async def get_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, lambda: self.model.encode(text))
        return embedding.tolist()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, lambda: self.model.encode(texts))
        return embeddings.tolist()

class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self._last_call = 0
        self._min_interval = 0.1

    async def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    async def get_embedding(self, text: str) -> List[float]:
        await self._rate_limit()
        try:
            response = await self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return [0.0] * EMBEDDING_DIMENSION

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        await self._rate_limit()
        try:
            response = await self.client.embeddings.create(model=self.model, input=texts)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            return [[0.0] * EMBEDDING_DIMENSION for _ in texts]

class FallbackEmbedding(EmbeddingProvider):
    def __init__(self):
        self.providers = []
        
        # Check if we should use OpenAI first based on config
        if EMBEDDING_MODEL == "text-embedding-ada-002" or os.getenv("FORCE_OPENAI_EMBEDDINGS"):
            try:
                self.providers.append(OpenAIEmbedding())
                logger.info("Using OpenAI for embeddings (1536 dimensions)")
            except Exception as e:
                logger.warning(f"OpenAI embedding unavailable: {e}")
        
        # Only fall back to SentenceTransformers if OpenAI isn't available
        if not self.providers:
            try:
                self.providers.append(SentenceTransformerEmbedding())
                logger.info("Using SentenceTransformers for embeddings (1536 dimensions)")
                logger.warning("WARNING: Dimension mismatch - database expects 1536!")
            except (ImportError, Exception) as e:
                logger.warning(f"SentenceTransformers unavailable: {e}")

        if not self.providers:
            logger.error("No embedding providers available! Using zero vectors.")

    async def get_embedding(self, text: str) -> List[float]:
        for provider in self.providers:
            try:
                return await provider.get_embedding(text)
            except Exception as e:
                logger.warning(f"Embedding provider {provider.__class__.__name__} failed: {e}")
        logger.error("All embedding providers failed")
        return [0.0] * EMBEDDING_DIMENSION

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        for provider in self.providers:
            try:
                return await provider.get_embeddings(texts)
            except Exception as e:
                logger.warning(f"Batch embedding provider {provider.__class__.__name__} failed: {e}")
        logger.error("All batch embedding providers failed")
        return [[0.0] * EMBEDDING_DIMENSION for _ in texts]

class DBConnectionManager:
    """
    Legacy connection manager for backward compatibility.
    Now uses the new connection pattern internally.
    """
    _pool = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        logger.warning("Using deprecated get_pool method, consider migrating to get_db_connection_context")
        if cls._pool is None:
            from db.connection import get_db_connection_pool
            cls._pool = await get_db_connection_pool()
        return cls._pool

    @classmethod
    async def close_pool(cls) -> None:
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("Database connection pool closed")

    @classmethod
    async def acquire(cls) -> asyncpg.Connection:
        logger.warning("Using deprecated acquire method, consider migrating to get_db_connection_context")
        from db.connection import get_db_connection_context
        # This creates a new connection each time, but it's for backward compatibility
        async with get_db_connection_context() as conn:
            return conn

    @classmethod
    async def release(cls, connection: asyncpg.Connection) -> None:
        # This is a no-op now since the connection context will handle closing
        pass
        
def with_transaction(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        conn = kwargs.pop('conn', None)
        external_conn = conn is not None
        if not external_conn:
            from db.connection import get_db_connection_context
            async with get_db_connection_context() as conn:
                tx = conn.transaction()
                start_time = time.time()
                
                try:
                    await tx.start()
                    result = await func(self, *args, conn=conn, **kwargs)
                    await tx.commit()
                    
                    elapsed = time.time() - start_time
                    await MemoryTelemetry.record(
                        self.user_id,
                        self.conversation_id,
                        operation=func.__name__,
                        success=True,
                        duration=elapsed,
                        data_size=1
                    )
                    return result
                except Exception as e:
                    await tx.rollback()
                    
                    elapsed = time.time() - start_time
                    await MemoryTelemetry.record(
                        self.user_id,
                        self.conversation_id,
                        operation=func.__name__,
                        success=False,
                        duration=elapsed,
                        error=str(e)
                    )
                    logger.error(f"Transaction error in {func.__name__}: {e}")
                    raise
        else:
            # If a connection was provided, just use it
            start_time = time.time()
            try:
                result = await func(self, *args, conn=conn, **kwargs)
                
                elapsed = time.time() - start_time
                await MemoryTelemetry.record(
                    self.user_id,
                    self.conversation_id,
                    operation=func.__name__,
                    success=True,
                    duration=elapsed,
                    data_size=1
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                await MemoryTelemetry.record(
                    self.user_id,
                    self.conversation_id,
                    operation=func.__name__,
                    success=False,
                    duration=elapsed,
                    error=str(e)
                )
                logger.error(f"Transaction error in {func.__name__}: {e}")
                raise
    return wrapper

class UnifiedMemoryManager:
    def __init__(
        self,
        entity_type: str,
        entity_id: int,
        user_id: int,
        conversation_id: int
    ):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.cache = MemoryCache(ttl_seconds=MEMORY_CACHE_TTL)
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

        if not memory.embedding and not embedding:
            embedding = await self.embedding_provider.get_embedding(memory.text)

        if not memory.timestamp:
            memory.timestamp = datetime.now()

        # Store fidelity in metadata so it can be loaded from DB row
        if "fidelity" not in memory.metadata:
            memory.metadata["fidelity"] = memory.fidelity

        memory_id = await conn.fetchval(
            """
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
            json.dumps(memory.tags), embedding or memory.embedding, json.dumps(memory.metadata),
            memory.timestamp, memory.times_recalled,
            memory.status.value if isinstance(memory.status, Enum) else memory.status,
            memory.is_consolidated
        )

        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")

        if memory.significance >= MemorySignificance.HIGH:
            await self._process_significant_memory(memory, memory_id, conn)
        return memory_id

    async def _process_significant_memory(self,
                                         memory: Memory,
                                         memory_id: int,
                                         conn: asyncpg.Connection) -> None:
        if memory.significance >= MemorySignificance.HIGH:
            await self._create_semantic_memory(memory, memory_id, conn)
        if memory.significance >= MemorySignificance.MEDIUM:
            await self._propagate_memory(memory, conn)

    async def _create_semantic_memory(self,
                                      memory: Memory,
                                      source_id: int,
                                      conn: asyncpg.Connection) -> Optional[int]:
        semantic_text = f"Abstract understanding: {memory.text}"
        semantic_memory = Memory(
            text=semantic_text,
            memory_type=MemoryType.SEMANTIC,
            significance=max(MemorySignificance.MEDIUM, memory.significance - 1),
            emotional_intensity=max(0, memory.emotional_intensity - 20),
            tags=memory.tags + ["semantic", "abstraction"],
            metadata={"source_memory_id": source_id},
            timestamp=datetime.now()
        )
        embedding = await self.embedding_provider.get_embedding(semantic_text)
        semantic_id = await conn.fetchval(
            """
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
            semantic_memory.emotional_intensity, json.dumps(semantic_memory.tags), embedding,
            json.dumps(semantic_memory.metadata), semantic_memory.timestamp, 0,
            MemoryStatus.ACTIVE.value, False
        )
        return semantic_id

    async def _propagate_memory(self,
                                memory: Memory,
                                conn: asyncpg.Connection) -> None:
        rows = await conn.fetch(
            """
            SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
            FROM SocialLinks 
            WHERE user_id = $1 AND conversation_id = $2
            AND ((entity1_type = $3 AND entity1_id = $4) OR (entity2_type = $3 AND entity2_id = $4))
            AND link_level >= 30
            """,
            self.user_id, self.conversation_id, self.entity_type, self.entity_id
        )
        if not rows:
            return
        for row in rows:
            if (row['entity1_type'] == self.entity_type and row['entity1_id'] == self.entity_id):
                target_type, target_id = (row['entity2_type'], row['entity2_id'])
            else:
                target_type, target_id = (row['entity1_type'], row['entity1_id'])
    
            if target_type == self.entity_type and target_id == self.entity_id:
                continue
    
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
            target_manager = UnifiedMemoryManager(
                entity_type=target_type,
                entity_id=target_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
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
        memory_types = memory_types or [t.value for t in MemoryType]
        exclude_ids = exclude_ids or []
        context = context or {}

        # Try cache
        if not query and not tags and not exclude_ids and min_significance == 0:
            cache_key = f"memories_{self.entity_type}_{self.entity_id}"
            cached = await self.cache.get(cache_key)
            if cached:
                # Slight optional fidelity drift when retrieving from cache
                for cm in cached:
                    if random.random() > cm.fidelity:
                        cm.text = await self._apply_fidelity_distortion(cm.text, cm.fidelity)
                return cached

        if use_embedding and query:
            query_embedding = await self.embedding_provider.get_embedding(query)
            rows = await conn.fetch(
                """
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
            rows = await conn.fetch(
                """
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

        memories = [Memory.from_db_row(dict(row)) for row in rows]
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]

        if context and memories:
            memories = await self._calculate_contextual_relevance(memories, context)

        if memories:
            memory_ids = [m.id for m in memories]
            await self._update_memory_access_batch(memory_ids, conn)

        if not query and not tags and not exclude_ids and min_significance == 0:
            cache_key = f"memories_{self.entity_type}_{self.entity_id}"
            await self.cache.set(cache_key, memories)

        return memories

    async def _apply_fidelity_distortion(self, text: str, fidelity: float) -> str:
        """
        Slightly distorts text if fidelity is < 1.0.
        The lower the fidelity, the higher chance of random word alterations.
        """
        words = text.split()
        for i in range(len(words)):
            if len(words[i]) > 3 and random.random() > fidelity:
                # Replace or alter the word
                words[i] = self._random_word_distortion(words[i])
        return " ".join(words)

    def _random_word_distortion(self, word: str) -> str:
        alterations = [
            lambda w: f"{w}?" if not w.endswith("?") else w,
            lambda w: w[::-1] if len(w) > 4 else w,
            lambda w: w.upper() if random.random() < 0.5 else w.lower()
        ]
        choice = random.choice(alterations)
        return choice(word)

    async def _calculate_contextual_relevance(self,
                                              memories: List[Memory],
                                              context: Dict[str, Any]) -> List[Memory]:
        location = context.get("location", "")
        time_of_day = context.get("time_of_day", "")

        for memory in memories:
            relevance_score = memory.significance * 10
            if memory.timestamp:
                days_old = (datetime.now() - memory.timestamp).total_seconds() / 86400
                recency_boost = max(0, 10 - days_old)
                relevance_score += recency_boost
            relevance_score += min(20, memory.times_recalled * 2)
            relevance_score += memory.emotional_intensity / 5
            if location and location in memory.text:
                relevance_score += 15
            if time_of_day and time_of_day in memory.text:
                relevance_score += 10
            memory.metadata["relevance_score"] = relevance_score
        memories.sort(key=lambda m: m.metadata.get("relevance_score", 0), reverse=True)
        return memories

    async def _update_memory_access_batch(self,
                                          memory_ids: List[int],
                                          conn: asyncpg.Connection) -> None:
        if not memory_ids:
            return
        await conn.execute(
            """
            UPDATE unified_memories
            SET times_recalled = times_recalled + 1,
                last_recalled = CURRENT_TIMESTAMP
            WHERE id = ANY($1)
            """,
            memory_ids
        )

    @with_transaction
    async def consolidate_memories(self,
                                   age_threshold_days: int = 7,
                                   min_count: int = 3,
                                   min_similarity: float = 0.8,
                                   conn: Optional[asyncpg.Connection] = None) -> List[int]:
        age_cutoff = datetime.now() - timedelta(days=age_threshold_days)
        rows = await conn.fetch(
            """
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

        memories = [Memory.from_db_row(dict(row)) for row in rows]
        clusters = await self._cluster_memories(memories, min_similarity, min_count)
        consolidated_ids = []

        for cluster in clusters:
            if len(cluster) < min_count:
                continue
            memory_ids = [m.id for m in cluster]
            memory_texts = [m.text for m in cluster]
            all_tags = set()
            for memory in cluster:
                all_tags.update(memory.tags)
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
            embedding = await self.embedding_provider.get_embedding(consolidated_text)
            consolidated_id = await conn.fetchval(
                """
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
                json.dumps(consolidated_memory.tags), embedding, json.dumps(consolidated_memory.metadata),
                consolidated_memory.timestamp, 0, MemoryStatus.ACTIVE.value, False
            )
            consolidated_ids.append(consolidated_id)
            await conn.execute(
                """
                UPDATE unified_memories
                SET is_consolidated = TRUE
                WHERE id = ANY($1)
                """,
                memory_ids
            )

        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        return consolidated_ids

    async def _cluster_memories(self,
                                memories: List[Memory],
                                similarity_threshold: float,
                                min_cluster_size: int) -> List[List[Memory]]:
        clusters = []
        unprocessed = memories.copy()
        while unprocessed:
            seed = unprocessed.pop(0)
            current_cluster = [seed]
            i = 0
            while i < len(unprocessed):
                memory = unprocessed[i]
                if seed.embedding and memory.embedding:
                    similarity = float(np.dot(seed.embedding, memory.embedding))
                    if similarity >= similarity_threshold:
                        current_cluster.append(memory)
                        unprocessed.pop(i)
                    else:
                        i += 1
                else:
                    i += 1
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
        return clusters

    @with_transaction
    async def apply_memory_decay(self,
                                 age_threshold_days: int = 30,
                                 recall_threshold: int = 3,
                                 decay_rate: float = 0.2,
                                 conn: Optional[asyncpg.Connection] = None) -> int:
        age_cutoff = datetime.now() - timedelta(days=age_threshold_days)
        rows = await conn.fetch(
            """
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
        affected_count = 0
        archive_ids = []
        for row in rows:
            memory_id = row['id']
            significance = row['significance']
            times_recalled = row['times_recalled']
            days_old = row['days_old']
            recall_factor = min(1.0, times_recalled / recall_threshold)
            age_factor = min(1.0, days_old / 100)
            decay_amount = decay_rate * age_factor * (1 - recall_factor)
            new_significance = max(1, significance - decay_amount)

            if new_significance < significance:
                await conn.execute(
                    """
                    UPDATE unified_memories
                    SET significance = $1
                    WHERE id = $2
                    """,
                    new_significance, memory_id
                )
                affected_count += 1
                if new_significance == 1 and times_recalled <= 1:
                    archive_ids.append(memory_id)

        if archive_ids:
            await conn.execute(
                """
                UPDATE unified_memories
                SET status = 'archived'
                WHERE id = ANY($1)
                """,
                archive_ids
            )
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        return affected_count

    @with_transaction
    async def reconsolidate_memory(self,
                                   memory_id: int,
                                   alteration_strength: float = 0.1,
                                   conn: Optional[asyncpg.Connection] = None) -> bool:
        row = await conn.fetchrow(
            """
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
        metadata = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'] or '{}')
        if memory_type in (MemoryType.SEMANTIC.value, MemoryType.CONSOLIDATED.value):
            return False
        if significance >= MemorySignificance.HIGH:
            return False
        original_form = metadata.get("original_form", memory_text)
        if "reconsolidation_history" not in metadata:
            metadata["reconsolidation_history"] = []
        metadata["reconsolidation_history"].append({
            "previous_text": memory_text,
            "timestamp": datetime.now().isoformat()
        })
        if len(metadata["reconsolidation_history"]) > 3:
            metadata["reconsolidation_history"] = metadata["reconsolidation_history"][-3:]
        if "original_form" not in metadata:
            metadata["original_form"] = original_form

        altered_text = await self._alter_memory_text(memory_text, alteration_strength)
        embedding = await self.embedding_provider.get_embedding(altered_text)
        metadata["fidelity"] = max(0.0, metadata.get("fidelity", 1.0) - (alteration_strength * 0.1))

        await conn.execute(
            """
            UPDATE unified_memories
            SET memory_text = $1,
                metadata = $2,
                embedding = $3
            WHERE id = $4
            """,
            altered_text, json.dumps(metadata), embedding, memory_id
        )
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        return True

    async def _alter_memory_text(self, text: str, alteration_strength: float) -> str:
        words = text.split()
        for i in range(len(words)):
            if len(words[i]) <= 3:
                continue
            if random.random() < alteration_strength:
                alterations = [
                    lambda w: f"very {w}" if len(w) > 3 else w,
                    lambda w: f"somewhat {w}" if len(w) > 3 else w,
                    lambda w: f"{w}ly" if not w.endswith("ly") and len(w) > 4 else w,
                    lambda w: w
                ]
                alteration = random.choice(alterations)
                words[i] = alteration(words[i])
        return " ".join(words)

    @with_transaction
    async def archive_old_memories(self,
                                   age_threshold_days: int = 60,
                                   significance_threshold: int = 2,
                                   recall_threshold: int = 2,
                                   conn: Optional[asyncpg.Connection] = None) -> int:
        age_cutoff = datetime.now() - timedelta(days=age_threshold_days)
        result = await conn.execute(
            """
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
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        affected = int(result.split(" ")[-1]) if " " in result else 0
        return affected

    @with_transaction
    async def batch_add_memories(self,
                                 memories: List[Union[Memory, str]],
                                 embed_batch_size: int = EMBEDDING_BATCH_SIZE,
                                 conn: Optional[asyncpg.Connection] = None) -> List[int]:
        if not memories:
            return []
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

        memory_batches = [memory_objects[i:i+embed_batch_size]
                          for i in range(0, len(memory_objects), embed_batch_size)]
        memory_ids = []
        for batch in memory_batches:
            texts = [memory.text for memory in batch]
            embeddings = await self.embedding_provider.get_embeddings(texts)
            for i, memory in enumerate(batch):
                if "fidelity" not in memory.metadata:
                    memory.metadata["fidelity"] = memory.fidelity
                memory_id = await conn.fetchval(
                    """
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
                    json.dumps(memory.tags), embeddings[i], json.dumps(memory.metadata),
                    memory.timestamp or datetime.now(), memory.times_recalled,
                    memory.status.value if isinstance(memory.status, Enum) else memory.status,
                    memory.is_consolidated
                )
                memory_ids.append(memory_id)
        await self.cache.delete(f"memories_{self.entity_type}_{self.entity_id}")
        return memory_ids

    @with_transaction
    async def perform_maintenance(self,
                                  conn: Optional[asyncpg.Connection] = None) -> Dict[str, int]:
        decayed = await self.apply_memory_decay(conn=conn)
        consolidated = await self.consolidate_memories(conn=conn)
        archived = await self.archive_old_memories(conn=conn)
        return {
            "memories_decayed": decayed,
            "memories_consolidated": len(consolidated),
            "memories_archived": archived
        }

    @classmethod
    @with_transaction
    async def create_tables(cls, conn: Optional[asyncpg.Connection] = None) -> None:
        """
        Static method: Initializes all required tables for unified memory.
        Must be called as UnifiedMemoryManager.create_tables(conn), NOT as an instance method.
        """
        await conn.execute(
            """
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
            CREATE INDEX IF NOT EXISTS idx_unified_memories_entity ON unified_memories(entity_type, entity_id);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_user_conv ON unified_memories(user_id, conversation_id);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_timestamp ON unified_memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_memory_type ON unified_memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_status ON unified_memories(status);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_significance ON unified_memories(significance);
            CREATE INDEX IF NOT EXISTS idx_unified_memories_embedding_hnsw ON unified_memories USING hnsw (embedding vector_cosine_ops);
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_telemetry (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                operation TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                duration FLOAT NOT NULL,
                data_size INTEGER,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_telemetry_timestamp ON memory_telemetry(timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_telemetry_operation ON memory_telemetry(operation);
            CREATE INDEX IF NOT EXISTS idx_memory_telemetry_success ON memory_telemetry(success);
            """
        )
        logger.info("Memory system tables created successfully")

# ---------------------------------------------------------------------
# Below is an additional "enhanced" style memory core for demonstration.
# We rename classes to avoid conflicts with the above system.
# ---------------------------------------------------------------------

class AlternateMemoryEntry:
    def __init__(self, content, entity_id=None, created_at=None, embedding=None, is_summary=False):
        self.id = None
        self.content = content
        self.entity_id = entity_id
        self.created_at = created_at or datetime.utcnow()
        self.last_accessed = self.created_at
        self.embedding = embedding
        self.is_summary = is_summary
        self.importance = 1.0
        self.active = True
        self.source_ids = []

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "entity_id": self.entity_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "importance": self.importance,
            "is_summary": self.is_summary,
            "active": self.active,
            "source_ids": list(self.source_ids),
        }

class AlternateMemoryCore:
    def __init__(self, embedder=None, summarizer=None, decay_rate=0.1, decay_interval=86400, cache_size=100):
        self.embedder = embedder
        self.summarizer = summarizer
        self.memories = []
        self.memory_index = {}
        self.entity_index = {}
        self.query_cache = OrderedDict()
        self.cache_size = cache_size
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval

    def add_memory(self, content, entity_id=None, timestamp=None, immediate_embed=False, linked_entities=None):
        entry = AlternateMemoryEntry(content, entity_id=entity_id, created_at=timestamp, is_summary=False)
        if immediate_embed and self.embedder:
            entry.embedding = self.embedder.generate_embedding(content)
        entry.id = len(self.memories) + 1
        self.memories.append(entry)
        self.memory_index[entry.id] = entry
        if entity_id is not None:
            self.entity_index.setdefault(entity_id, []).append(entry.id)
        self.query_cache.clear()
        if linked_entities:
            for linked_id in linked_entities:
                if linked_id is None or linked_id == entity_id:
                    continue
                adapted_content = self._adapt_content_for_entity(content, source_entity=entity_id, target_entity=linked_id)
                self.add_memory(adapted_content, entity_id=linked_id, timestamp=timestamp,
                                immediate_embed=immediate_embed, linked_entities=None)
        return entry.id

    def _adapt_content_for_entity(self, content, source_entity, target_entity):
        return content

    def retrieve_memories(self, query, top_k=5, entity_id=None):
        now = datetime.utcnow()
        cache_key = (query, entity_id)
        if cache_key in self.query_cache:
            result_ids = self.query_cache[cache_key]
            results = [self.memory_index[mid] for mid in result_ids if mid in self.memory_index]
            for entry in results:
                entry.last_accessed = now
                entry.importance = min(1.0, entry.importance + 0.1)
            return results
        query_embedding = None
        if self.embedder:
            query_embedding = self.embedder.generate_embedding(query)
        scored_entries = []
        for entry in self.memories:
            if not entry.active:
                continue
            if entity_id is not None:
                if entry.entity_id is not None and entry.entity_id != entity_id:
                    continue
            if self.embedder:
                if entry.embedding is None:
                    entry.embedding = self.embedder.generate_embedding(entry.content)
                sim = 0.0
                if query_embedding is not None:
                    q_vec = np.array(query_embedding)
                    m_vec = np.array(entry.embedding)
                    dot = float(np.dot(q_vec, m_vec))
                    sim = dot / ((np.linalg.norm(q_vec) * np.linalg.norm(m_vec)) + 1e-8)
            else:
                q_words = set(query.lower().split())
                e_words = set(entry.content.lower().split())
                sim = float(len(q_words & e_words))
            score = sim * entry.importance
            if score > 0:
                scored_entries.append((score, entry.id))
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        top_ids = [eid for (_score, eid) in scored_entries[:top_k]]
        self.query_cache[cache_key] = top_ids
        if len(self.query_cache) > self.cache_size:
            self.query_cache.popitem(last=False)
        results = []
        for mid in top_ids:
            if mid in self.memory_index:
                entry = self.memory_index[mid]
                entry.last_accessed = now
                entry.importance = min(1.0, entry.importance + 0.1)
                results.append(entry)
        return results

    def consolidate_memory(self):
        now = datetime.utcnow()
        groups = {}
        for entry in self.memories:
            if not entry.active:
                continue
            group_key = entry.entity_id if entry.entity_id is not None else "_global"
            groups.setdefault(group_key, []).append(entry)
        for key, entries in groups.items():
            if len(entries) < 2:
                continue
            entries.sort(key=lambda e: e.created_at)
            oldest = entries[0]
            age_days = (now - oldest.created_at).days
            if age_days > 30 or len(entries) > 50:
                num_to_summarize = max(5, len(entries) // 2)
                to_summarize = entries[:num_to_summarize]
                if self.summarizer:
                    texts = [e.content for e in to_summarize]
                    summary_text = self.summarizer.summarize(texts)
                else:
                    summary_text = " | ".join(e.content for e in to_summarize)
                summary_entry = AlternateMemoryEntry(content=summary_text,
                                                     entity_id=(None if key == "_global" else key),
                                                     created_at=now,
                                                     is_summary=True)
                summary_entry.id = len(self.memories) + 1
                self.memories.append(summary_entry)
                self.memory_index[summary_entry.id] = summary_entry
                if summary_entry.entity_id is not None:
                    self.entity_index.setdefault(summary_entry.entity_id, []).append(summary_entry.id)
                for e in to_summarize:
                    e.active = False
                    summary_entry.source_ids.append(e.id)
                self.query_cache.clear()
                if self.embedder:
                    summary_entry.embedding = self.embedder.generate_embedding(summary_text)

    def apply_decay(self):
        now = datetime.utcnow()
        for entry in self.memories:
            if not entry.active:
                continue
            elapsed = (now - entry.last_accessed).total_seconds()
            if elapsed >= self.decay_interval:
                intervals = elapsed // self.decay_interval
                decay_amount = self.decay_rate * intervals
                entry.importance = max(0.0, entry.importance - decay_amount)
                entry.last_accessed = now
                if entry.importance <= 0.1:
                    entry.active = False
