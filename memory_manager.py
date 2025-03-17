# context/memory_manager.py

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

class Memory:
    """Unified memory representation with metadata"""
    
    def __init__(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "observation",
        created_at: Optional[datetime] = None,
        importance: float = 0.5,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.memory_id = memory_id
        self.content = content
        self.memory_type = memory_type
        self.created_at = created_at or datetime.now()
        self.importance = importance
        self.access_count = access_count
        self.last_accessed = last_accessed or self.created_at
        self.tags = tags or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary"""
        timestamp = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        last_accessed = datetime.fromisoformat(data["last_accessed"]) if isinstance(data["last_accessed"], str) else data["last_accessed"]
        
        memory = cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            created_at=timestamp,
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        return memory
    
    def access(self) -> None:
        """Record an access to this memory"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_importance(self) -> float:
        """Calculate and update the importance score"""
        # Base score by type
        type_scores = {
            "observation": 0.3,
            "event": 0.5,
            "scene": 0.4,
            "decision": 0.6,
            "character_development": 0.7,
            "game_mechanic": 0.5,
            "quest": 0.8,
            "summary": 0.7
        }
        type_score = type_scores.get(self.memory_type, 0.4)
        
        # Recency factor (newer = more important)
        age_days = (datetime.now() - self.created_at).days
        recency = 1.0 if age_days < 1 else max(0.0, 1.0 - (age_days / 30))
        
        # Access score (frequently accessed = more important)
        access_score = min(0.3, 0.05 * math.log(1 + self.access_count))
        
        # Calculate final score
        self.importance = min(1.0, (type_score * 0.5) + (recency * 0.3) + (access_score * 0.2))
        
        return self.importance


class MemoryManager:
    """Integrated memory manager for storage, retrieval, and consolidation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.memories = {}  # In-memory cache of recent/important memories
        self.memory_index = {}  # Indices for efficient lookup
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the memory manager"""
        if self.is_initialized:
            return
            
        # Load important memories
        await self._load_important_memories()
        
        self.is_initialized = True
        logger.info(f"Initialized memory manager for user {self.user_id}, conversation {self.conversation_id}")
    
    async def close(self):
        """Close the memory manager"""
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
                        LIMIT 50
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
        
        # Build indices
        self._build_memory_indices()
        
        return len(memories)
    
    def _build_memory_indices(self):
        """Build memory indices for efficient lookup"""
        # Reset indices
        self.memory_index = {
            "type": {},
            "tag": {},
            "timestamp": []
        }
        
        # Build type index
        for memory_id, memory in self.memories.items():
            # Type index
            memory_type = memory.memory_type
            if memory_type not in self.memory_index["type"]:
                self.memory_index["type"][memory_type] = set()
            self.memory_index["type"][memory_type].add(memory_id)
            
            # Tag index
            for tag in memory.tags:
                if tag not in self.memory_index["tag"]:
                    self.memory_index["tag"][tag] = set()
                self.memory_index["tag"][tag].add(memory_id)
            
            # Timestamp index (sorted by created_at)
            self.memory_index["timestamp"].append((memory.created_at, memory_id))
        
        # Sort timestamp index
        self.memory_index["timestamp"].sort(reverse=True)
    
    async def add_memory(
        self,
        content: str,
        memory_type: str = "observation",
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_vector: bool = True
    ) -> str:
        """Add a new memory with vector storage integration"""
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
                if importance >= 0.5:
                    self.memories[memory.memory_id] = memory
                    self._build_memory_indices()
                
                # Store vector embedding if vector search is enabled
                if store_vector and self.config.is_enabled("use_vector_search"):
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
        """Get a memory by ID"""
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
                        self._build_memory_indices()
                    
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
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5,
        use_vector: bool = True
    ) -> List[Memory]:
        """Search memories by query text with vector search option"""
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
        importance = min(0.7, 0.3 + (len(query_text) / 100))
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
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Memory]:
        """Search memories in database using text search"""
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
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Get recent memories"""
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
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance tasks"""
        # Determine if consolidation is needed based on configuration
        should_consolidate = self.config.get("memory_consolidation", "enabled", True)
        
        if not should_consolidate:
            return {
                "consolidated": False,
                "reason": "Memory consolidation disabled in config"
            }
            
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Count old, unconsolidated memories
                days_threshold = self.config.get("memory_consolidation", "days_threshold", 7)
                min_memories = self.config.get("memory_consolidation", "min_memories_to_consolidate", 5)
                
                cutoff_date = datetime.now() - timedelta(days=days_threshold)
                
                count_row = await conn.fetchrow("""
                    SELECT COUNT(*) as count
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                      AND created_at < $3
                      AND consolidated = FALSE
                """, self.user_id, self.conversation_id, cutoff_date)
                
                if not count_row or count_row["count"] < min_memories:
                    return {
                        "consolidated": False,
                        "reason": f"Not enough old memories to consolidate: {count_row['count'] if count_row else 0} < {min_memories}"
                    }
                
                # At this point, we would perform memory consolidation
                # Simplified version just marks memories as consolidated
                await conn.execute("""
                    UPDATE PlayerJournal
                    SET consolidated = TRUE
                    WHERE user_id = $1 AND conversation_id = $2
                      AND created_at < $3
                      AND consolidated = FALSE
                """, self.user_id, self.conversation_id, cutoff_date)
                
                return {
                    "consolidated": True,
                    "count": count_row["count"],
                    "threshold_days": days_threshold
                }
            
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Error during memory maintenance: {e}")
            return {
                "consolidated": False,
                "error": str(e)
            }


# Global manager registry
_memory_managers = {}

async def get_memory_manager(user_id: int, conversation_id: int) -> MemoryManager:
    """Get or create a memory manager instance"""
    key = f"{user_id}:{conversation_id}"
    
    if key not in _memory_managers:
        manager = MemoryManager(user_id, conversation_id)
        await manager.initialize()
        _memory_managers[key] = manager
    
    return _memory_managers[key]

async def cleanup_memory_managers():
    """Close all memory managers"""
    global _memory_managers
    
    # Close each manager
    for manager in _memory_managers.values():
        await manager.close()
    
    # Clear registry
    _memory_managers.clear()
