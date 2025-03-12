# nyx/nyx_memory_system.py

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncpg
import numpy as np

from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity

logger = logging.getLogger(__name__)

class NyxMemorySystem:
    """
    Enhanced memory system for Nyx that persists across sessions and games.
    This extends beyond the NPCMemoryManager with:
    - Global memories (across all users)
    - User-specific memories (persistent across their games)
    - Game-specific memories (for the current session)
    - Memory reconsolidation and semantic extraction
    """

    def __init__(self, user_id: int = None, conversation_id: int = None):
        """
        Initialize the memory system.
        
        Args:
            user_id: Current user ID (optional for global memories)
            conversation_id: Current conversation/game ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Cache structures
        self._memory_cache = {}
        self._cache_ttl = 300  # 5 minutes in seconds
        self._cache_timestamp = datetime.now()
    
    async def add_memory(
        self, 
        memory_text: str,
        memory_type: str = "observation",
        memory_scope: str = "game",  # "global", "user", or "game"
        significance: int = 5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Add a new memory to the appropriate scope.
        
        Args:
            memory_text: The content of the memory
            memory_type: Type of memory (observation, reflection, abstraction)
            memory_scope: Scope of memory (global, user, or game)
            significance: Importance of memory (1-10)
            tags: List of tags for categorization and retrieval
            metadata: Additional structured data to store
            
        Returns:
            ID of the created memory
        """
        tags = tags or []
        metadata = metadata or {}
        
        # Add timestamp to metadata
        metadata["created_at"] = datetime.now().isoformat()
        
        # Generate embedding for semantic search
        embedding = await generate_embedding(memory_text)
        
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                # Table selection based on scope
                table_name = ""
                query_params = []
                
                if memory_scope == "global":
                    table_name = "NyxGlobalMemories"
                    query = """
                        INSERT INTO NyxGlobalMemories 
                        (memory_text, memory_type, significance, embedding, tags, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING id
                    """
                    query_params = [memory_text, memory_type, significance, embedding, tags, json.dumps(metadata)]
                    
                elif memory_scope == "user":
                    if not self.user_id:
                        raise ValueError("user_id required for user-scoped memories")
                    
                    table_name = "NyxUserMemories"
                    query = """
                        INSERT INTO NyxUserMemories
                        (user_id, memory_text, memory_type, significance, embedding, tags, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                    """
                    query_params = [self.user_id, memory_text, memory_type, significance, embedding, tags, json.dumps(metadata)]
                    
                else:  # game scope
                    if not self.user_id or not self.conversation_id:
                        raise ValueError("user_id and conversation_id required for game-scoped memories")
                        
                    table_name = "NyxMemories"
                    query = """
                        INSERT INTO NyxMemories
                        (user_id, conversation_id, memory_text, memory_type, significance, embedding, tags, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        RETURNING id
                    """
                    query_params = [self.user_id, self.conversation_id, memory_text, memory_type, significance, embedding, tags, json.dumps(metadata)]
                
                # Execute query
                memory_id = await conn.fetchval(query, *query_params)
                
                # For significant memories, create a semantic abstraction
                if significance >= 7 and memory_type == "observation":
                    await self._create_semantic_abstraction(memory_text, memory_id, memory_scope, tags, conn)
                
                # Clear relevant cache
                self._invalidate_cache()
                
                return memory_id

    async def retrieve_npc_memories(self, query, npc_ids=None):
        """Retrieve memories from specific NPCs"""
        coordinator = NPCAgentCoordinator(self.user_id, self.conversation_id)
        npc_ids = await coordinator.load_agents(npc_ids)
        
        all_memories = []
        for npc_id in npc_ids:
            if npc_id in coordinator.active_agents:
                agent = coordinator.active_agents[npc_id]
                memory_result = await agent.memory_manager.retrieve_memories(query)
                for memory in memory_result.get("memories", []):
                    memory["npc_id"] = npc_id
                    all_memories.append(memory)
        
        return all_memories
    
    async def retrieve_memories(
        self,
        query: str,
        scopes: List[str] = None,
        memory_types: List[str] = None,
        limit: int = 10,
        min_significance: int = 3,
        include_metadata: bool = True,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from specified scopes.
        
        Args:
            query: Search query
            scopes: Memory scopes to search ["global", "user", "game"]
            memory_types: Types of memories to include
            limit: Maximum number of memories to return
            min_significance: Minimum significance level
            include_metadata: Whether to include metadata in results
            context: Optional context information for relevance scoring
            
        Returns:
            List of relevant memories
        """
        scopes = scopes or ["game", "user", "global"]
        memory_types = memory_types or ["observation", "reflection", "abstraction"]
        context = context or {}
        
        # Check cache
        cache_key = f"retrieve_{query}_{'-'.join(scopes)}_{'-'.join(memory_types)}_{limit}_{min_significance}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Generate embedding for query
        query_embedding = await generate_embedding(query)
        
        all_memories = []
        
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                for scope in scopes:
                    # Build query based on scope
                    if scope == "global":
                        query_sql = """
                            SELECT id, memory_text, memory_type, significance, tags, metadata,
                                   embedding <-> $1 AS distance
                            FROM NyxGlobalMemories
                            WHERE memory_type = ANY($2)
                              AND significance >= $3
                            ORDER BY distance
                            LIMIT $4
                        """
                        rows = await conn.fetch(query_sql, query_embedding, memory_types, min_significance, limit)
                        
                    elif scope == "user" and self.user_id:
                        query_sql = """
                            SELECT id, memory_text, memory_type, significance, tags, metadata,
                                   embedding <-> $1 AS distance
                            FROM NyxUserMemories
                            WHERE user_id = $2
                              AND memory_type = ANY($3)
                              AND significance >= $4
                            ORDER BY distance
                            LIMIT $5
                        """
                        rows = await conn.fetch(query_sql, query_embedding, self.user_id, memory_types, min_significance, limit)
                        
                    elif scope == "game" and self.user_id and self.conversation_id:
                        query_sql = """
                            SELECT id, memory_text, memory_type, significance, tags, metadata,
                                   embedding <-> $1 AS distance
                            FROM NyxMemories
                            WHERE user_id = $2
                              AND conversation_id = $3
                              AND memory_type = ANY($4)
                              AND significance >= $5
                              AND is_archived = FALSE
                            ORDER BY distance
                            LIMIT $6
                        """
                        rows = await conn.fetch(query_sql, query_embedding, self.user_id, self.conversation_id, memory_types, min_significance, limit)
                    else:
                        continue
                    
                    # Process results
                    for row in rows:
                        memory = dict(row)
                        memory["scope"] = scope
                        
                        # Calculate relevance score (lower distance = higher relevance)
                        memory["relevance"] = 1.0 / (1.0 + memory.pop("distance"))
                        
                        # Parse metadata
                        if memory.get("metadata") and include_metadata:
                            memory["metadata"] = json.loads(memory["metadata"])
                        elif not include_metadata:
                            memory.pop("metadata", None)
                            
                        # Convert embedding to list if needed
                        if "embedding" in memory:
                            memory.pop("embedding")
                            
                        all_memories.append(memory)
                
                # Sort by relevance and limit results
                all_memories.sort(key=lambda x: x["relevance"], reverse=True)
                result = all_memories[:limit]
                
                # Update retrieval stats for these memories
                memory_ids_by_scope = {scope: [] for scope in scopes}
                for memory in result:
                    memory_scope = memory["scope"]
                    memory_ids_by_scope[memory_scope].append(memory["id"])
                
                # Update retrieval statistics
                for scope, ids in memory_ids_by_scope.items():
                    if not ids:
                        continue
                        
                    if scope == "global":
                        await conn.execute("""
                            UPDATE NyxGlobalMemories
                            SET times_recalled = times_recalled + 1,
                                last_recalled = CURRENT_TIMESTAMP
                            WHERE id = ANY($1)
                        """, ids)
                    elif scope == "user":
                        await conn.execute("""
                            UPDATE NyxUserMemories
                            SET times_recalled = times_recalled + 1,
                                last_recalled = CURRENT_TIMESTAMP
                            WHERE id = ANY($1) AND user_id = $2
                        """, ids, self.user_id)
                    elif scope == "game":
                        await conn.execute("""
                            UPDATE NyxMemories
                            SET times_recalled = times_recalled + 1,
                                last_recalled = CURRENT_TIMESTAMP
                            WHERE id = ANY($1) AND user_id = $2 AND conversation_id = $3
                        """, ids, self.user_id, self.conversation_id)
                
                # Store in cache
                self._add_to_cache(cache_key, result)
                
                return result
    
    async def _create_semantic_abstraction(
        self, 
        memory_text: str, 
        source_id: int, 
        memory_scope: str,
        tags: List[str],
        conn
    ):
        """Create a higher-level semantic abstraction from a memory."""
        from nyx.llm_integration import create_semantic_abstraction
        
        try:
            # Use LLM to create abstraction
            abstraction = await create_semantic_abstraction(memory_text)
            
            # Generate embedding for the abstraction
            embedding = await generate_embedding(abstraction)
            
            # Store in same scope as original but with type 'abstraction'
            if memory_scope == "global":
                query = """
                    INSERT INTO NyxGlobalMemories 
                    (memory_text, memory_type, significance, embedding, tags, metadata)
                    VALUES ($1, 'abstraction', $2, $3, $4, $5)
                    RETURNING id
                """
                params = [
                    abstraction, 
                    6,  # Slightly higher significance for abstractions
                    embedding, 
                    tags + ["abstraction", "semantic"],
                    json.dumps({"source_memory_id": source_id})
                ]
                
            elif memory_scope == "user":
                query = """
                    INSERT INTO NyxUserMemories
                    (user_id, memory_text, memory_type, significance, embedding, tags, metadata)
                    VALUES ($1, $2, 'abstraction', $3, $4, $5, $6)
                    RETURNING id
                """
                params = [
                    self.user_id,
                    abstraction, 
                    6,
                    embedding, 
                    tags + ["abstraction", "semantic"],
                    json.dumps({"source_memory_id": source_id})
                ]
                
            else:  # game scope 
                query = """
                    INSERT INTO NyxMemories
                    (user_id, conversation_id, memory_text, memory_type, significance, embedding, tags, metadata)
                    VALUES ($1, $2, $3, 'abstraction', $4, $5, $6, $7)
                    RETURNING id
                """
                params = [
                    self.user_id,
                    self.conversation_id,
                    abstraction, 
                    6,
                    embedding, 
                    tags + ["abstraction", "semantic"],
                    json.dumps({"source_memory_id": source_id})
                ]
            
            await conn.execute(query, *params)
            
        except Exception as e:
            logger.error(f"Error creating semantic abstraction: {e}")
    
    async def generate_reflection(self, topic: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a reflective thought about a topic or the player based on memories.
        
        Args:
            topic: Optional topic to reflect on
            context: Optional context information
            
        Returns:
            Dictionary with reflection text and metadata
        """
        from nyx.llm_integration import generate_reflection
        
        # Get relevant memories
        query = topic if topic else "player behavior personality preferences"
        scopes = ["game", "user", "global"]
        memories = await self.retrieve_memories(
            query=query,
            scopes=scopes,
            memory_types=["observation", "abstraction", "reflection"],
            limit=10,
            min_significance=4,
            context=context
        )
        
        # Extract memory texts
        memory_texts = [m["memory_text"] for m in memories]
        
        # Generate reflection using LLM
        reflection_text = await generate_reflection(memory_texts, topic, context)
        
        # Calculate confidence based on memory relevance and quantity
        avg_relevance = sum(m["relevance"] for m in memories) / len(memories) if memories else 0
        memory_count_factor = min(1.0, len(memories) / 10)
        confidence = avg_relevance * 0.7 + memory_count_factor * 0.3
        
        # Store this reflection
        reflection_id = await self.add_memory(
            memory_text=reflection_text,
            memory_type="reflection",
            memory_scope="game" if self.conversation_id else "user",
            significance=5,
            tags=["reflection", topic] if topic else ["reflection", "player_model"],
            metadata={
                "topic": topic,
                "confidence": confidence,
                "source_memory_ids": [m["id"] for m in memories]
            }
        )
        
        return {
            "reflection": reflection_text,
            "confidence": confidence,
            "memory_count": len(memories),
            "id": reflection_id
        }
    
    async def run_maintenance(self, scope: str = "all"):
        """
        Run maintenance tasks on memories - consolidation, decay, etc.
        
        Args:
            scope: Scope to maintain ("global", "user", "game", or "all")
        """
        # Implementations for:
        # 1. Memory decay - reduce significance of old, unretrieved memories
        # 2. Consolidation - combine similar memories
        # 3. Archive - move very old, low-significance memories to archive
        pass
    
    def _get_from_cache(self, key: str) -> Any:
        """Get a value from cache if valid."""
        if key not in self._memory_cache:
            return None
            
        # Check if cache is expired
        if (datetime.now() - self._cache_timestamp).total_seconds() > self._cache_ttl:
            return None
            
        return self._memory_cache.get(key)
    
    def _add_to_cache(self, key: str, value: Any):
        """Add a value to the cache."""
        self._memory_cache[key] = value
        
    def _invalidate_cache(self):
        """Invalidate the cache."""
        self._memory_cache = {}
        self._cache_timestamp = datetime.now()

# Database initialization
async def initialize_nyx_memory_tables():
    """Create the necessary database tables for Nyx's memory system."""
    async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
        async with pool.acquire() as conn:
            # Global memories (shared across all users/games)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS NyxGlobalMemories (
                    id SERIAL PRIMARY KEY,
                    memory_text TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    significance INTEGER NOT NULL,
                    embedding VECTOR(1536),
                    tags TEXT[] DEFAULT '{}',
                    times_recalled INTEGER DEFAULT 0,
                    last_recalled TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # User-specific memories (persist across games)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS NyxUserMemories (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    significance INTEGER NOT NULL,
                    embedding VECTOR(1536),
                    tags TEXT[] DEFAULT '{}',
                    times_recalled INTEGER DEFAULT 0,
                    last_recalled TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Game-specific memories (the existing NyxMemories table)
            # Ensure it has all needed fields
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS NyxMemories (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    significance INTEGER NOT NULL,
                    embedding VECTOR(1536),
                    tags TEXT[] DEFAULT '{}',
                    times_recalled INTEGER DEFAULT 0,
                    last_recalled TIMESTAMP,
                    is_archived BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for efficient retrieval
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_nyx_global_memories_embedding ON NyxGlobalMemories USING ivfflat (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_nyx_user_memories_embedding ON NyxUserMemories USING ivfflat (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_nyx_memories_embedding ON NyxMemories USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_nyx_user_memories_user_id ON NyxUserMemories(user_id);
                CREATE INDEX IF NOT EXISTS idx_nyx_memories_user_conv ON NyxMemories(user_id, conversation_id);
            """)
