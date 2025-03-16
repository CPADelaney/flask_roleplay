# context/memory_manager.py

"""
Integrated Memory Manager that combines memory consolidation, prioritization,
retrieval, and storage for efficient long-term game management.
"""

import asyncio
import logging
import json
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import hashlib

from context.unified_cache import context_cache
from context.context_config import get_config
from context.vector_service import get_vector_service

logger = logging.getLogger(__name__)

class MemoryImportance:
    """Utility class for scoring memory importance"""
    
    @staticmethod
    def score_memory(
        memory_content: str,
        memory_type: str,
        recency: float,
        access_count: int
    ) -> float:
        """
        Score a memory's importance based on various factors
        
        Args:
            memory_content: The text content
            memory_type: Type of memory (observation, event, etc.)
            recency: How recent (0-1, 1 being very recent)
            access_count: How many times this has been accessed
            
        Returns:
            Importance score from 0-1
        """
        # Base score by type
        type_scores = {
            "observation": 0.3,
            "event": 0.5,
            "scene": 0.4,
            "decision": 0.6,
            "character_development": 0.7,
            "game_mechanic": 0.5,
            "quest": 0.8
        }
        type_score = type_scores.get(memory_type, 0.4)
        
        # Content-based score
        content_score = min(1.0, len(memory_content) / 500) * 0.2
        
        # Text-based importance markers
        important_terms = [
            "vital", "important", "critical", "key", "crucial", 
            "significant", "essential", "fundamental", "pivotal"
        ]
        term_score = 0.0
        for term in important_terms:
            if term in memory_content.lower():
                term_score += 0.1
        term_score = min(0.3, term_score)
        
        # Recency factor
        recency_score = recency * 0.2
        
        # Access count factor (with diminishing returns)
        access_score = min(0.3, 0.05 * math.log(1 + access_count))
        
        # Combine scores
        total_score = type_score + content_score + term_score + recency_score + access_score
        
        # Normalize to 0-1
        return min(1.0, total_score)

class Memory:
    """
    Enhanced representation of a memory with metadata
    """
    
    def __init__(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "observation",
        created_at: Optional[datetime] = None,
        importance: float = 0.5,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        tags: List[str] = None,
        vector_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Dict[str, Any] = None
    ):
        self.memory_id = memory_id
        self.content = content
        self.memory_type = memory_type
        self.created_at = created_at or datetime.now()
        self.importance = importance
        self.access_count = access_count
        self.last_accessed = last_accessed or self.created_at
        self.tags = tags or []
        self.vector_id = vector_id
        self.embedding = embedding
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }
        
        # Only include embedding if available
        if self.embedding:
            result["embedding"] = self.embedding
        
        if self.vector_id:
            result["vector_id"] = self.vector_id
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary"""
        return cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            importance=data["importance"],
            access_count=data["access_count"],
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if isinstance(data["last_accessed"], str) else data["last_accessed"],
            tags=data.get("tags", []),
            vector_id=data.get("vector_id"),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {})
        )
    
    def access(self) -> None:
        """Record an access to this memory"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_recency(self) -> float:
        """
        Get recency score (0-1)
        1.0 = very recent, 0.0 = very old
        """
        now = datetime.now()
        age = (now - self.created_at).total_seconds()
        
        # Scale recency - exponential decay over 30 days
        decay_factor = 2592000  # 30 days in seconds
        recency = math.exp(-age / decay_factor)
        
        return recency
    
    def calculate_importance(self) -> float:
        """Calculate and update the importance score"""
        recency = self.get_recency()
        
        # Score memory based on multiple factors
        score = MemoryImportance.score_memory(
            self.content,
            self.memory_type,
            recency,
            self.access_count
        )
        
        # Update importance
        self.importance = score
        
        return score

class MemoryManager:
    """
    Integrated memory manager that combines various memory features
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.memories = {}  # In-memory cache of recent memories
        self.consolidation_task = None
        self.maintenance_interval = self.config.get(
            "memory_consolidation", "consolidation_interval_hours", 24
        ) * 3600  # Convert to seconds
        self.last_maintenance = time.time()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the memory manager"""
        if self.is_initialized:
            return
            
        # Load important memories
        await self._load_important_memories()
        
        # Schedule background consolidation if enabled
        if self.config.get("memory_consolidation", "enabled", True):
            self.consolidation_task = asyncio.create_task(self._run_maintenance_loop())
        
        self.is_initialized = True
        logger.info(f"Initialized memory manager for user {self.user_id}, conversation {self.conversation_id}")
    
    async def _run_maintenance_loop(self):
        """Run the maintenance loop in the background"""
        try:
            while True:
                # Sleep until next maintenance interval
                await asyncio.sleep(self.maintenance_interval)
                
                # Run maintenance
                try:
                    await self.run_maintenance()
                except Exception as e:
                    logger.error(f"Error during memory maintenance: {e}")
        except asyncio.CancelledError:
            # Task cancelled
            logger.info(f"Memory maintenance task cancelled for user {self.user_id}")
    
    async def close(self):
        """Close the memory manager"""
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass
            self.consolidation_task = None
        
        self.is_initialized = False
        logger.info(f"Closed memory manager for user {self.user_id}, conversation {self.conversation_id}")
    
    async def _load_important_memories(self):
        """Load important memories into local cache"""
        # Use cache key for important memories
        cache_key = f"important_memories:{self.user_id}:{self.conversation_id}"
        
        async def fetch_important_memories():
            try:
                # Get database connection
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                try:
                    # Get important memories (high importance or recently accessed)
                    rows = await conn.fetch("""
                        SELECT id, entry_type, entry_text, entry_metadata, 
                               created_at, importance, access_count, last_accessed, tags
                        FROM PlayerJournal
                        WHERE user_id = $1 AND conversation_id = $2
                          AND (importance >= 0.7 OR last_accessed > NOW() - INTERVAL '7 days')
                        ORDER BY importance DESC, last_accessed DESC
                        LIMIT 100
                    """, self.user_id, self.conversation_id)
                    
                    # Convert to Memory objects
                    memories = {}
                    for row in rows:
                        # Parse metadata JSON
                        metadata = {}
                        if row["entry_metadata"]:
                            try:
                                metadata = json.loads(row["entry_metadata"])
                            except:
                                pass
                        
                        # Parse tags
                        tags = []
                        if row["tags"]:
                            try:
                                tags = json.loads(row["tags"])
                            except:
                                pass
                        
                        memory = Memory(
                            memory_id=str(row["id"]),
                            content=row["entry_text"],
                            memory_type=row["entry_type"],
                            created_at=row["created_at"],
                            importance=row["importance"],
                            access_count=row["access_count"],
                            last_accessed=row["last_accessed"],
                            tags=tags,
                            metadata=metadata
                        )
                        
                        memories[str(row["id"])] = memory
                    
                    return memories
                
                finally:
                    await conn.close()
            
            except Exception as e:
                logger.error(f"Error loading important memories: {e}")
                return {}
        
        # Get from cache or fetch (1 hour TTL, medium importance)
        memories = await context_cache.get(
            cache_key, 
            fetch_important_memories, 
            cache_level=2,  # L2 cache
            importance=0.6,
            ttl_override=3600  # 1 hour
        )
        
        # Store in local cache
        self.memories = memories
        
        return len(memories)
    
    async def add_memory(
        self,
        content: str,
        memory_type: str = "observation",
        importance: Optional[float] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add a new memory with vector storage integration
        
        Args:
            content: Memory content text
            memory_type: Type of memory
            importance: Optional importance override (auto-calculated if None)
            tags: Optional tags
            metadata: Optional additional metadata
            
        Returns:
            Memory ID
        """
        # Generate unique memory ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_base = f"{self.user_id}:{self.conversation_id}:{content}:{timestamp}"
        memory_id = f"{timestamp}-{hashlib.md5(hash_base.encode()).hexdigest()[:8]}"
        
        # Create memory object
        memory = Memory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Auto-calculate importance if not provided
        if importance is None:
            importance = memory.calculate_importance()
        memory.importance = importance
        
        # Store in database
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Insert into database
                result = await conn.fetchrow("""
                    INSERT INTO PlayerJournal(
                        user_id, conversation_id, entry_type, entry_text, 
                        entry_metadata, importance, access_count, last_accessed, tags
                    )
                    VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, 
                    self.user_id, 
                    self.conversation_id,
                    memory_type,
                    content,
                    json.dumps(metadata) if metadata else None,
                    importance,
                    0,  # access_count
                    datetime.now(),  # last_accessed
                    json.dumps(tags) if tags else None
                )
                
                # Get the database ID
                db_id = result["id"]
                memory.memory_id = str(db_id)
                
                # If the memory is important, store in local cache
                if importance >= 0.7:
                    self.memories[memory.memory_id] = memory
                
                # Store vector embedding if vector search is enabled
                if self.config.is_enabled("use_vector_search"):
                    # Get vector service
                    vector_service = await get_vector_service(self.user_id, self.conversation_id)
                    
                    # Add to vector database
                    await vector_service.add_memory(
                        memory_id=memory.memory_id,
                        content=content,
                        memory_type=memory_type,
                        importance=importance,
                        tags=tags
                    )
                
                return memory.memory_id
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return ""
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory object or None if not found
        """
        # Check local cache first
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access()
            return memory
        
        # Use cache for other memories
        cache_key = f"memory:{self.user_id}:{self.conversation_id}:{memory_id}"
        
        async def fetch_memory():
            try:
                # Get database connection
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                try:
                    # Get memory from database
                    row = await conn.fetchrow("""
                        SELECT id, entry_type, entry_text, entry_metadata, 
                               created_at, importance, access_count, last_accessed, tags
                        FROM PlayerJournal
                        WHERE user_id = $1 AND conversation_id = $2 AND id = $3
                    """, self.user_id, self.conversation_id, int(memory_id))
                    
                    if not row:
                        return None
                    
                    # Parse metadata JSON
                    metadata = {}
                    if row["entry_metadata"]:
                        try:
                            metadata = json.loads(row["entry_metadata"])
                        except:
                            pass
                    
                    # Parse tags
                    tags = []
                    if row["tags"]:
                        try:
                            tags = json.loads(row["tags"])
                        except:
                            pass
                    
                    # Create Memory object
                    memory = Memory(
                        memory_id=str(row["id"]),
                        content=row["entry_text"],
                        memory_type=row["entry_type"],
                        created_at=row["created_at"],
                        importance=row["importance"],
                        access_count=row["access_count"],
                        last_accessed=row["last_accessed"],
                        tags=tags,
                        metadata=metadata
                    )
                    
                    # Update access count in database
                    await conn.execute("""
                        UPDATE PlayerJournal
                        SET access_count = access_count + 1, last_accessed = NOW()
                        WHERE id = $1
                    """, int(memory_id))
                    
                    # If important, store in local cache
                    if memory.importance >= 0.5:
                        self.memories[memory.memory_id] = memory
                    
                    return memory
                
                finally:
                    await conn.close()
            
            except Exception as e:
                logger.error(f"Error getting memory: {e}")
                return None
        
        # Get from cache or fetch (5 minute TTL, medium importance)
        memory = await context_cache.get(
            cache_key, 
            fetch_memory, 
            cache_level=1,  # L1 cache
            importance=0.5,
            ttl_override=300  # 5 minutes
        )
        
        if memory:
            memory.access()
            
        return memory
    
    async def search_memories(
        self,
        query_text: str,
        memory_types: List[str] = None,
        tags: List[str] = None,
        limit: int = 5,
        use_vector: bool = True
    ) -> List[Memory]:
        """
        Search memories by query text with vector search option
        
        Args:
            query_text: Search query text
            memory_types: Optional list of memory types to filter
            tags: Optional list of tags to filter
            limit: Maximum results to return
            use_vector: Whether to use vector search
            
        Returns:
            List of matching Memory objects
        """
        cache_key = f"memory_search:{self.user_id}:{self.conversation_id}:{query_text}:{memory_types}:{tags}:{limit}:{use_vector}"
        
        async def perform_search():
            results = []
            
            # Use vector search if enabled
            if use_vector and self.config.is_enabled("use_vector_search"):
                # Get vector service
                vector_service = await get_vector_service(self.user_id, self.conversation_id)
                
                # Search for memories
                vector_results = await vector_service.search_entities(
                    query_text=query_text,
                    entity_types=["memory"],
                    top_k=limit
                )
                
                # Convert results to Memory objects
                for result in vector_results:
                    metadata = result.get("metadata", {})
                    if metadata.get("entity_type") == "memory":
                        memory_id = metadata.get("memory_id")
                        if memory_id:
                            # Get full memory
                            memory = await self.get_memory(memory_id)
                            if memory:
                                results.append(memory)
            
            # If we didn't get enough results or vector search is disabled,
            # fall back to database search
            if len(results) < limit:
                db_limit = limit - len(results)
                db_results = await self._search_memories_in_db(
                    query_text, memory_types, tags, db_limit
                )
                
                # Add unique results
                existing_ids = {m.memory_id for m in results}
                for memory in db_results:
                    if memory.memory_id not in existing_ids:
                        results.append(memory)
                        existing_ids.add(memory.memory_id)
            
            # Record access for retrieved memories
            for memory in results:
                memory.access()
            
            return results
        
        # Get from cache or search (30 second TTL, importance based on query)
        importance = min(0.8, 0.3 + (len(query_text) / 100))
        return await context_cache.get(
            cache_key, 
            perform_search, 
            cache_level=1,  # L1 cache
            importance=importance,
            ttl_override=30  # 30 seconds
        )
    
    async def _search_memories_in_db(
        self,
        query_text: str,
        memory_types: List[str] = None,
        tags: List[str] = None,
        limit: int = 5
    ) -> List[Memory]:
        """
        Search memories in database using text search
        
        Args:
            query_text: Search query text
            memory_types: Optional list of memory types to filter
            tags: Optional list of tags to filter
            limit: Maximum results to return
            
        Returns:
            List of matching Memory objects
        """
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Prepare query
                query = """
                    SELECT id, entry_type, entry_text, entry_metadata, 
                           created_at, importance, access_count, last_accessed, tags
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                """
                params = [self.user_id, self.conversation_id]
                
                # Add text search condition
                if query_text:
                    query += f" AND entry_text ILIKE ${len(params) + 1}"
                    params.append(f"%{query_text}%")
                
                # Add memory type filter
                if memory_types:
                    placeholders = ', '.join(f'${i}' for i in range(len(params) + 1, len(params) + len(memory_types) + 1))
                    query += f" AND entry_type IN ({placeholders})"
                    params.extend(memory_types)
                
                # Add ordering and limit
                query += " ORDER BY importance DESC, last_accessed DESC LIMIT $" + str(len(params) + 1)
                params.append(limit)
                
                # Execute query
                rows = await conn.fetch(query, *params)
                
                # Process results
                results = []
                for row in rows:
                    # Parse metadata JSON
                    metadata = {}
                    if row["entry_metadata"]:
                        try:
                            metadata = json.loads(row["entry_metadata"])
                        except:
                            pass
                    
                    # Parse tags
                    row_tags = []
                    if row["tags"]:
                        try:
                            row_tags = json.loads(row["tags"])
                        except:
                            pass
                    
                    # Apply tag filter if needed
                    if tags and not any(tag in row_tags for tag in tags):
                        continue
                    
                    # Create Memory object
                    memory = Memory(
                        memory_id=str(row["id"]),
                        content=row["entry_text"],
                        memory_type=row["entry_type"],
                        created_at=row["created_at"],
                        importance=row["importance"],
                        access_count=row["access_count"],
                        last_accessed=row["last_accessed"],
                        tags=row_tags,
                        metadata=metadata
                    )
                    
                    results.append(memory)
                
                return results
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error searching memories in DB: {e}")
            return []
    
    async def get_recent_memories(
        self,
        days: int = 3,
        memory_types: List[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Get recent memories
        
        Args:
            days: Number of days to look back
            memory_types: Optional list of memory types to filter
            limit: Maximum results to return
            
        Returns:
            List of recent Memory objects
        """
        cache_key = f"recent_memories:{self.user_id}:{self.conversation_id}:{days}:{memory_types}:{limit}"
        
        async def fetch_recent_memories():
            try:
                # Get database connection
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                try:
                    # Prepare query
                    query = """
                        SELECT id, entry_type, entry_text, entry_metadata, 
                               created_at, importance, access_count, last_accessed, tags
                        FROM PlayerJournal
                        WHERE user_id = $1 AND conversation_id = $2
                          AND created_at > NOW() - INTERVAL '1 day' * $3
                    """
                    params = [self.user_id, self.conversation_id, days]
                    
                    # Add memory type filter
                    if memory_types:
                        placeholders = ', '.join(f'${i}' for i in range(len(params) + 1, len(params) + len(memory_types) + 1))
                        query += f" AND entry_type IN ({placeholders})"
                        params.extend(memory_types)
                    
                    # Add ordering and limit
                    query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
                    params.append(limit)
                    
                    # Execute query
                    rows = await conn.fetch(query, *params)
                    
                    # Process results
                    results = []
                    for row in rows:
                        # Parse metadata JSON
                        metadata = {}
                        if row["entry_metadata"]:
                            try:
                                metadata = json.loads(row["entry_metadata"])
                            except:
                                pass
                        
                        # Parse tags
                        tags = []
                        if row["tags"]:
                            try:
                                tags = json.loads(row["tags"])
                            except:
                                pass
                        
                        # Create Memory object
                        memory = Memory(
                            memory_id=str(row["id"]),
                            content=row["entry_text"],
                            memory_type=row["entry_type"],
                            created_at=row["created_at"],
                            importance=row["importance"],
                            access_count=row["access_count"],
                            last_accessed=row["last_accessed"],
                            tags=tags,
                            metadata=metadata
                        )
                        
                        results.append(memory)
                    
                    return results
                
                finally:
                    await conn.close()
            
            except Exception as e:
                logger.error(f"Error getting recent memories: {e}")
                return []
        
        # Get from cache or fetch (2 minute TTL, medium importance)
        return await context_cache.get(
            cache_key, 
            fetch_recent_memories, 
            cache_level=1,  # L1 cache
            importance=0.4,
            ttl_override=120  # 2 minutes
        )
    
    async def consolidate_memories(
        self,
        days_threshold: int = 7,
        min_memories: int = 5
    ) -> Dict[str, Any]:
        """
        Consolidate old memories into summaries
        
        Args:
            days_threshold: Age in days for consolidation
            min_memories: Minimum number of memories to consolidate
            
        Returns:
            Dictionary with consolidation results
        """
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Get old memories
                cutoff_date = datetime.now() - timedelta(days=days_threshold)
                
                rows = await conn.fetch("""
                    SELECT id, entry_type, entry_text, created_at, importance
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                      AND created_at < $3
                      AND consolidated = FALSE
                    ORDER BY entry_type, created_at
                """, self.user_id, self.conversation_id, cutoff_date)
                
                # Group by memory type
                memories_by_type = {}
                for row in rows:
                    memory_type = row["entry_type"]
                    if memory_type not in memories_by_type:
                        memories_by_type[memory_type] = []
                    
                    memories_by_type[memory_type].append({
                        "id": row["id"],
                        "content": row["entry_text"],
                        "created_at": row["created_at"],
                        "importance": row["importance"]
                    })
                
                # Process each group
                consolidation_results = {}
                total_consolidated = 0
                
                for memory_type, memories in memories_by_type.items():
                    # Skip if not enough memories
                    if len(memories) < min_memories:
                        continue
                    
                    # Sort by created_at
                    memories.sort(key=lambda m: m["created_at"])
                    
                    # Create consolidation batches (by week)
                    batches = []
                    current_batch = []
                    current_week = memories[0]["created_at"].isocalendar()[1]  # Week number
                    
                    for memory in memories:
                        memory_week = memory["created_at"].isocalendar()[1]
                        
                        if memory_week != current_week or len(current_batch) >= 20:
                            # Start a new batch
                            if current_batch:
                                batches.append(current_batch)
                            current_batch = [memory]
                            current_week = memory_week
                        else:
                            # Add to current batch
                            current_batch.append(memory)
                    
                    # Add last batch
                    if current_batch:
                        batches.append(current_batch)
                    
                    # Process each batch
                    for batch in batches:
                        # Generate summary ID
                        batch_start = batch[0]["created_at"].strftime("%Y-%m-%d")
                        batch_end = batch[-1]["created_at"].strftime("%Y-%m-%d")
                        summary_id = f"summary-{memory_type}-{batch_start}-to-{batch_end}"
                        
                        # Generate summary content
                        memory_contents = [m["content"] for m in batch]
                        summary_text = await self._generate_memory_summary(memory_contents, memory_type)
                        
                        # Calculate importance (average of source memories)
                        avg_importance = sum(m["importance"] for m in batch) / len(batch)
                        
                        # Create tags
                        tags = ["summary", memory_type, f"period:{batch_start}-to-{batch_end}"]
                        
                        # Create metadata
                        metadata = {
                            "summary_type": "time_period",
                            "source_memory_ids": [m["id"] for m in batch],
                            "start_date": batch_start,
                            "end_date": batch_end,
                            "memory_count": len(batch)
                        }
                        
                        # Add the summary memory
                        await self.add_memory(
                            content=summary_text,
                            memory_type=f"summary_{memory_type}",
                            importance=avg_importance,
                            tags=tags,
                            metadata=metadata
                        )
                        
                        # Mark original memories as consolidated
                        memory_ids = [m["id"] for m in batch]
                        placeholders = ', '.join(str(id) for id in memory_ids)
                        
                        await conn.execute(f"""
                            UPDATE PlayerJournal
                            SET consolidated = TRUE
                            WHERE id IN ({placeholders})
                        """)
                        
                        # Record results
                        if memory_type not in consolidation_results:
                            consolidation_results[memory_type] = []
                        
                        consolidation_results[memory_type].append({
                            "summary_id": summary_id,
                            "memory_count": len(batch),
                            "period": f"{batch_start} to {batch_end}"
                        })
                        
                        total_consolidated += len(batch)
                
                return {
                    "consolidated": total_consolidated > 0,
                    "total_memories_consolidated": total_consolidated,
                    "summaries_created": sum(len(summaries) for summaries in consolidation_results.values()),
                    "results_by_type": consolidation_results
                }
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return {
                "consolidated": False,
                "error": str(e)
            }
    
    async def _generate_memory_summary(
        self,
        memory_contents: List[str],
        memory_type: str
    ) -> str:
        """
        Generate a summary from multiple memories
        
        Args:
            memory_contents: List of memory content strings
            memory_type: Type of memories
            
        Returns:
            Summary text
        """
        # In a production system, you would use an LLM here
        # For this example, we'll create a simple concatenation
        combined = "\n".join(memory_contents)
        
        # Truncate if too long
        if len(combined) > 500:
            combined = combined[:497] + "..."
        
        return f"Summary of {len(memory_contents)} {memory_type} memories: {combined}"
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run all maintenance tasks
        
        Returns:
            Dictionary with maintenance results
        """
        self.last_maintenance = time.time()
        
        results = {
            "memory_consolidation": None,
            "vector_maintenance": False,
            "embedding_verification": False
        }
        
        # 1. Memory consolidation
        days_threshold = self.config.get("memory_consolidation", "days_threshold", 7)
        min_memories = self.config.get("memory_consolidation", "min_memories_to_consolidate", 5)
        
        consolidation_result = await self.consolidate_memories(
            days_threshold=days_threshold,
            min_memories=min_memories
        )
        
        results["memory_consolidation"] = consolidation_result
        
        # 2. Vector maintenance (verify embeddings exist for important memories)
        if self.config.is_enabled("use_vector_search"):
            # This would verify and fix any memories missing from vector DB
            results["vector_maintenance"] = True
        
        # 3. Update importance scores for memories
        # This would recalculate importance scores for memories that
        # have changed in access patterns
        
        # 4. Clean up temporary or very old, unimportant memories
        # This would remove or archive memories that are no longer needed
        
        return results

# Global function to get or create memory manager
async def get_memory_manager(user_id: int, conversation_id: int) -> MemoryManager:
    """
    Get or create a memory manager instance
    """
    # Use the cache to avoid creating multiple instances
    cache_key = f"memory_manager:{user_id}:{conversation_id}"
    
    async def create_manager():
        manager = MemoryManager(user_id, conversation_id)
        await manager.initialize()
        return manager
    
    # Get from cache or create new with 10 minute TTL (level 2 cache)
    return await context_cache.get(
        cache_key, 
        create_manager, 
        cache_level=2, 
        ttl_override=600
    )

# Cleanup function
async def cleanup_memory_managers():
    """Close all memory managers"""
    # Get all memory manager keys from cache
    memory_manager_keys = [
        key for key in context_cache.l1_cache.keys() 
        if key.startswith("memory_manager:")
    ]
    memory_manager_keys.extend([
        key for key in context_cache.l2_cache.keys() 
        if key.startswith("memory_manager:")
    ])
    
    # Close each manager
    for key in set(memory_manager_keys):
        manager = context_cache.l1_cache.get(key) or context_cache.l2_cache.get(key)
        if manager:
            await manager.close()
    
    # Clear from cache
    context_cache.invalidate("memory_manager:")
