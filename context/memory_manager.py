# context/memory_manager.py

import asyncio
import logging
import json
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import hashlib
from collections import defaultdict

# Agent SDK imports
from agents import Agent, function_tool, RunContextWrapper, trace
from pydantic import BaseModel, Field

from context.unified_cache import context_cache
from context.context_config import get_config
from context.vector_service import get_vector_service, VectorService

# Updated import for async connection management
from db.connection import get_db_connection_context

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


# Pydantic models for Agent SDK integration
class MemoryModel(BaseModel):
    """Pydantic model for memory data"""
    memory_id: str
    content: str
    memory_type: str = "observation"
    created_at: str
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    
    @classmethod
    def from_memory(cls, memory: Memory) -> 'MemoryModel':
        """Convert Memory object to MemoryModel"""
        return cls(
            memory_id=memory.memory_id,
            content=memory.content,
            memory_type=memory.memory_type,
            created_at=memory.created_at.isoformat(),
            importance=memory.importance,
            access_count=memory.access_count,
            last_accessed=memory.last_accessed.isoformat(),
            tags=memory.tags,
            metadata=memory.metadata
        )


class MemorySearchRequest(BaseModel):
    """Request model for memory search"""
    query_text: str
    memory_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    limit: int = 5
    use_vector: bool = True


class MemorySearchResult(BaseModel):
    """Result model for memory search"""
    memories: List[MemoryModel] = []
    query: str
    total_found: int
    vector_search_used: bool
    search_time_ms: float


class MemoryAddRequest(BaseModel):
    """Request model for adding a memory"""
    content: str
    memory_type: str = "observation"
    importance: Optional[float] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    store_vector: bool = True


class MemoryAddResult(BaseModel):
    """Result model for adding a memory"""
    memory_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class MemoryConsolidationRules(BaseModel):
    """Rules for memory consolidation"""
    time_window_days: int = 7
    min_importance: float = 0.4
    group_by: str = "type"
    max_memories_per_group: int = 5
    min_memories_to_summarize: int = 3


class MemoryConsolidationResult(BaseModel):
    """Result of memory consolidation"""
    processed_count: int = 0
    consolidated_count: int = 0
    summary_count: int = 0
    success: bool = False
    error: Optional[str] = None


class MemoryManager:
    """Integrated memory manager for storage, retrieval, and consolidation"""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = None  # Will be properly initialized in async initialize()
        self.memories: Dict[str, Memory] = {}  # In-memory cache
        self.is_initialized = False
        self.vector_service: Optional[VectorService] = None
        
        # Lock for shared resource access
        self._init_lock = asyncio.Lock()

        # Indices for efficient lookup
        self.memory_indices: Dict[str, defaultdict] = {
            "by_type": defaultdict(list),
            "by_importance": defaultdict(list),
            "by_recency": defaultdict(list),
            "by_tag": defaultdict(set)
        }
        
        # Configuration Defaults (can be updated by Nyx)
        self.type_scores = {
            "event": 1.0, "relationship": 0.8, "knowledge": 0.6, "emotion": 0.4,
            "observation": 0.6, "scene": 0.7, "decision": 0.9, "character_development": 0.9,
            "game_mechanic": 0.7, "quest": 1.0, "summary": 0.8
        }
        self.importance_weights = {"recency": 0.4, "access_frequency": 0.3, "type": 0.3}
        self.recency_weights = {"hour": 1.0, "day": 0.8, "week": 0.6, "month": 0.4}
        self.include_rules: Set[str] = set()
        self.exclude_rules: Set[str] = set()
        self.importance_threshold: float = 0.5

        # Nyx directive handling state
        self.nyx_directives = {}
        self.nyx_overrides = {}
        self.nyx_prohibitions = {}

        # Nyx governance integration state
        self.governance = None
        self.directive_handler = None
        
        logger.debug(f"MemoryManager instance created for {user_id}:{conversation_id}")
    
    async def initialize(self):
        """Initialize the memory manager asynchronously."""
        # Use a lock to prevent multiple concurrent initializations
        async with self._init_lock:
            if self.is_initialized:
                logger.debug(f"MemoryManager for {self.user_id}:{self.conversation_id} already initialized.")
                return
                
            logger.info(f"Initializing MemoryManager for {self.user_id}:{self.conversation_id}...")
            try:
                # Get config using the async method
                self.config = await get_config()
                
                # Initialize Vector Service if needed
                if self.config.is_enabled("use_vector_search"):
                    self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
                    logger.info("Vector service initialized for MemoryManager.")

                # Initialize Nyx integration
                await self._initialize_nyx_integration(self.user_id, self.conversation_id)

                # Load important memories from DB into cache
                await self._load_important_memories()

                # Build memory indices from the loaded cache
                self._build_memory_indices()

                self.is_initialized = True
                logger.info(f"MemoryManager initialized successfully for {self.user_id}:{self.conversation_id}.")
            except Exception as e:
                logger.exception(f"Error initializing memory manager for {self.user_id}:{self.conversation_id}: {e}")
                raise

    async def _initialize_nyx_integration(self, user_id: int, conversation_id: int):
        """Initialize Nyx governance integration asynchronously."""
        try:
            from nyx.integrate import get_central_governance
            from nyx.directive_handler import DirectiveHandler
            from nyx.nyx_governance import AgentType
            
            # Get governance system
            self.governance = await get_central_governance(user_id, conversation_id)
            
            # Initialize directive handler
            self.directive_handler = DirectiveHandler(
                user_id,
                conversation_id,
                AgentType.MEMORY_MANAGER,
                "memory_manager"
            )
            
            # Register handlers
            self.directive_handler.register_handler("action", self._handle_action_directive)
            self.directive_handler.register_handler("override", self._handle_override_directive)
            self.directive_handler.register_handler("prohibition", self._handle_prohibition_directive)
            
            logger.info("Initialized Nyx integration for memory manager")
        except Exception as e:
            logger.error(f"Error initializing Nyx integration: {e}")

    @function_tool
    async def search_memories(
        self, 
        ctx: RunContextWrapper, 
        request: MemorySearchRequest
    ) -> MemorySearchResult:
        """
        Search memories by query text with vector search option
        
        Args:
            request: Search request with query text, filters and options
        
        Returns:
            List of matching memories with metadata
        """
        with trace(workflow_name="search_memories"):
            start_time = time.monotonic()
            
            # Ensure initialization
            if not self.is_initialized:
                await self.initialize()
            
            # Determine if vector search is enabled and available
            effective_use_vector = (request.use_vector and 
                                    self.config.is_enabled("use_vector_search") and 
                                    self.vector_service is not None)
            
            # Create a hash for cache key
            cache_key_params = f"{request.query_text}:{request.memory_types}:{request.tags}:{request.limit}:{effective_use_vector}"
            cache_key = f"memory_search:{self.user_id}:{self.conversation_id}:{hashlib.md5(cache_key_params.encode()).hexdigest()}"
            ttl_override = 30  # 30 seconds

            async def perform_search():
                results_map: Dict[str, Memory] = {}  # Use dict to handle unique IDs easily

                # 1. Vector Search (if enabled)
                if effective_use_vector:
                    logger.debug(f"Performing vector search for: '{request.query_text}'")
                    try:
                        vector_results = await self.vector_service.search_entities(
                            query_text=request.query_text,
                            entity_types=["memory"],
                            top_k=request.limit
                        )

                        memory_ids_from_vector = []
                        for result in vector_results:
                             metadata = result.get("metadata", {})
                             if metadata.get("entity_type") == "memory":
                                 memory_id = metadata.get("memory_id")
                                 if memory_id:
                                     memory_ids_from_vector.append(memory_id)

                        # Batch fetch full memories for vector results
                        for mem_id in memory_ids_from_vector:
                             memory = await self.get_memory(mem_id)
                             if memory:
                                 # Apply post-fetch filtering if vector filters weren't perfect
                                 if request.memory_types and memory.memory_type not in request.memory_types: 
                                    continue
                                 if request.tags and not any(tag in memory.tags for tag in request.tags): 
                                    continue
                                 results_map[memory.memory_id] = memory

                    except Exception as vec_err:
                        logger.error(f"Vector search failed: {vec_err}", exc_info=True)
                        # Continue to DB search as fallback

                # 2. Database Search (if needed or vector disabled)
                needed = request.limit - len(results_map)
                if needed > 0 or not effective_use_vector:
                    logger.debug(f"Performing DB search for: '{request.query_text}' (needed: {needed})")
                    try:
                        db_results = await self._search_memories_in_db(
                            request.query_text, request.memory_types, request.tags, request.limit
                        )
                        # Add unique results from DB up to the overall limit
                        for memory in db_results:
                            if memory.memory_id not in results_map and len(results_map) < request.limit:
                                results_map[memory.memory_id] = memory
                    except Exception as db_search_err:
                         logger.error(f"DB search failed: {db_search_err}", exc_info=True)

                # 3. Final sorting and limit
                final_results = list(results_map.values())
                # Re-sort if combining results, e.g., by importance or recency
                final_results.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)

                return final_results[:request.limit]  # Apply final limit

            # Get from cache or search
            search_importance = min(0.7, 0.3 + (len(request.query_text) / 100.0))
            results = await context_cache.get(
                cache_key,
                perform_search,
                cache_level=1,  # L1 cache
                importance=search_importance,
                ttl_override=ttl_override
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            
            # Convert Memory objects to MemoryModel
            memory_models = [MemoryModel.from_memory(memory) for memory in results]
            
            return MemorySearchResult(
                memories=memory_models,
                query=request.query_text,
                total_found=len(results),
                vector_search_used=effective_use_vector,
                search_time_ms=duration_ms
            )

    @function_tool
    async def add_memory(
        self,
        ctx: RunContextWrapper,
        request: MemoryAddRequest
    ) -> MemoryAddResult:
        """
        Add a new memory with vector storage integration
        
        Args:
            request: Memory data to add
            
        Returns:
            Result with success status and memory ID
        """
        with trace(workflow_name="add_memory"):
            # Generate a placeholder memory ID for the object initially
            placeholder_id = f"temp_{datetime.now().timestamp()}_{hashlib.md5(request.content.encode()).hexdigest()[:6]}"

            # Create memory object
            memory = Memory(
                memory_id=placeholder_id,
                content=request.content,
                memory_type=request.memory_type,
                tags=request.tags or [],
                metadata=request.metadata or {}
            )

            # Calculate importance if not provided
            calculated_importance = memory.calculate_importance()
            final_importance = request.importance if request.importance is not None else calculated_importance
            memory.importance = final_importance

            # Serialize metadata and tags for DB
            metadata_json = None
            if request.metadata:
                try: 
                    metadata_json = json.dumps(request.metadata)
                except TypeError: 
                    logger.error("Failed to serialize metadata", exc_info=True)
                    metadata_json = "{}"
                    
            tags_json = None
            if request.tags:
                try: 
                    tags_json = json.dumps(request.tags)
                except TypeError: 
                    logger.error("Failed to serialize tags", exc_info=True)
                    tags_json = "[]"

            # Store in database
            db_id: Optional[int] = None
            try:
                async with get_db_connection_context() as conn:
                    # Use a transaction for atomicity
                    async with conn.transaction():
                        # Insert into database
                        insert_query = """
                            INSERT INTO PlayerJournal(
                                user_id, conversation_id, entry_type, entry_text,
                                entry_metadata, importance, access_count, last_accessed, created_at, tags, consolidated
                            )
                            VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, FALSE)
                            RETURNING id
                        """
                        db_id = await conn.fetchval(
                            insert_query,
                            self.user_id, self.conversation_id,
                            request.memory_type, request.content, metadata_json,
                            final_importance, 0,  # access_count
                            memory.created_at, memory.created_at,  # last_accessed = created_at initially
                            tags_json
                        )

                if db_id is None:
                    logger.error("Failed to add memory to DB (fetchval returned None).")
                    return MemoryAddResult(
                        success=False,
                        error="Database error - could not create memory"
                    )

                # Update memory object with the real DB ID (as string)
                memory.memory_id = str(db_id)
                logger.info(f"Added memory {memory.memory_id} (type: {request.memory_type}, importance: {final_importance:.2f})")

                # Add to local cache if important enough
                cache_importance_threshold = 0.6
                if self.config:
                    cache_importance_threshold = self.config.get("memory_cache_importance_threshold", 0.6)
                    
                if final_importance >= cache_importance_threshold:
                    self.memories[memory.memory_id] = memory
                    self._build_memory_indices()  # Rebuild indices after adding to cache
                    logger.debug(f"Memory {memory.memory_id} added to local cache.")

                # Store vector embedding if enabled
                if request.store_vector and self.config and self.config.is_enabled("use_vector_search") and self.vector_service:
                    try:
                        await self.vector_service.add_memory(
                            memory_id=memory.memory_id,
                            content=request.content,
                            memory_type=request.memory_type,
                            importance=final_importance,
                            tags=request.tags
                        )
                        logger.debug(f"Vector added for memory {memory.memory_id}")
                    except Exception as vec_err:
                        logger.error(f"Failed to add vector for memory {memory.memory_id}: {vec_err}", exc_info=True)

                return MemoryAddResult(
                    memory_id=memory.memory_id,
                    success=True
                )

            except Exception as e:
                logger.error(f"Unexpected error adding memory: {e}", exc_info=True)
                return MemoryAddResult(
                    success=False,
                    error=str(e)
                )

    @function_tool
    async def get_memory(
        self, 
        ctx: RunContextWrapper, 
        memory_id: str
    ) -> Optional[MemoryModel]:
        """
        Get a memory by ID, checking local cache first, then DB.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object if found, otherwise None
        """
        with trace(workflow_name="get_memory"):
            if not memory_id or not memory_id.isdigit():
                 logger.warning(f"Attempted to get memory with invalid ID: {memory_id}")
                 return None

            # 1. Check local instance cache
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.access()  # Update local object access time/count
                logger.debug(f"Memory {memory_id} found in local cache.")
                return MemoryModel.from_memory(memory)

            # 2. Check distributed cache (via context_cache)
            cache_key = f"memory:{self.user_id}:{self.conversation_id}:{memory_id}"
            ttl_override = 300  # 5 minutes

            async def fetch_memory_from_db():
                logger.debug(f"Fetching memory {memory_id} from DB for {self.user_id}:{self.conversation_id}")
                query = """
                    SELECT id, entry_type, entry_text, entry_metadata,
                           created_at, importance, access_count, last_accessed, tags, consolidated
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2 AND id = $3
                """
                try:
                    async with get_db_connection_context() as conn:
                        async with conn.transaction():
                            # Fetch memory
                            row = await conn.fetchrow(query, self.user_id, self.conversation_id, int(memory_id))
                            if not row:
                                logger.warning(f"Memory {memory_id} not found in DB.")
                                return None

                            memory = self._parse_journal_row(row)

                            # Update access count/time in DB *before* returning
                            update_query = """
                                UPDATE PlayerJournal
                                SET access_count = access_count + 1, last_accessed = NOW()
                                WHERE id = $1
                                RETURNING access_count, last_accessed
                            """
                            update_result = await conn.fetchrow(update_query, int(memory_id))
                            if update_result:
                                memory.access_count = update_result['access_count']
                                memory.last_accessed = update_result['last_accessed']
                                logger.debug(f"Updated access count for memory {memory_id} in DB.")

                            # Add to local cache if important enough
                            cache_importance_threshold = 0.6
                            if self.config:
                                cache_importance_threshold = self.config.get("memory_cache_importance_threshold", 0.6)
                                
                            if memory.importance >= cache_importance_threshold:
                                 self.memories[memory.memory_id] = memory
                                 logger.debug(f"Memory {memory_id} added to local cache after DB fetch.")

                            return memory

                except Exception as e:
                    logger.error(f"Error getting memory {memory_id}: {e}", exc_info=True)
                    return None

            # Get from context_cache or fetch from DB
            cached_data = await context_cache.get(
                cache_key,
                fetch_memory_from_db,
                cache_level=1,  # L1 cache
                importance=0.5,  # Fixed importance for cache item itself
                ttl_override=ttl_override
            )

            # Handle potential dict from cache
            if isinstance(cached_data, dict):
                memory_obj = Memory.from_dict(cached_data)
                if memory_obj and memory_id not in self.memories:
                     cache_importance_threshold = 0.6
                     if self.config:
                        cache_importance_threshold = self.config.get("memory_cache_importance_threshold", 0.6)
                        
                     if memory_obj.importance >= cache_importance_threshold:
                          self.memories[memory_id] = memory_obj
                          logger.debug(f"Memory {memory_id} added to local cache after cache fetch.")
                return MemoryModel.from_memory(memory_obj)
            elif isinstance(cached_data, Memory):
                if memory_id not in self.memories:
                     cache_importance_threshold = 0.6
                     if self.config:
                        cache_importance_threshold = self.config.get("memory_cache_importance_threshold", 0.6)
                        
                     if cached_data.importance >= cache_importance_threshold:
                           self.memories[memory_id] = cached_data
                           logger.debug(f"Memory {memory_id} added to local cache after cache fetch.")
                return MemoryModel.from_memory(cached_data)
            else:
                return None

    @function_tool
    async def get_recent_memories(
        self,
        ctx: RunContextWrapper,
        days: int = 3,
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[MemoryModel]:
        """
        Get recent memories asynchronously.
        
        Args:
            days: Number of days to look back
            memory_types: Types of memories to include
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memories
        """
        with trace(workflow_name="get_recent_memories"):
            cache_key = f"recent_memories:{self.user_id}:{self.conversation_id}:{days}:{memory_types}:{limit}"
            ttl_override = 120  # 2 minutes

            async def fetch_recent_memories():
                logger.debug(f"Fetching recent memories (last {days} days) from DB for {self.user_id}:{self.conversation_id}")
                query = """
                    SELECT id, entry_type, entry_text, entry_metadata,
                           created_at, importance, access_count, last_accessed, tags, consolidated
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                      AND created_at > NOW() - ($3 * INTERVAL '1 day')
                """
                params: List[Any] = [self.user_id, self.conversation_id, days]
                conditions: List[str] = []

                # Memory type filter
                if memory_types:
                    type_placeholders = []
                    for mem_type in memory_types:
                        param_index = len(params) + 1
                        type_placeholders.append(f"${param_index}")
                        params.append(mem_type)
                    conditions.append(f"entry_type IN ({', '.join(type_placeholders)})")

                # Combine conditions
                if conditions:
                    query += " AND " + " AND ".join(conditions)

                # Add ordering and limit
                param_index = len(params) + 1
                query += f" ORDER BY created_at DESC LIMIT ${param_index}"
                params.append(limit)

                memories = []
                try:
                    async with get_db_connection_context() as conn:
                        rows = await conn.fetch(query, *params)
                        for row_data in rows:
                            memories.append(self._parse_journal_row(row_data))
                    logger.info(f"Fetched {len(memories)} recent memories from DB for {self.user_id}:{self.conversation_id}")
                    return memories
                except Exception as e:
                    logger.error(f"Error fetching recent memories: {e}", exc_info=True)
                    return []

            # Get from cache or fetch
            results = await context_cache.get(
                cache_key,
                fetch_recent_memories,
                cache_level=1,  # L1 cache
                importance=0.4,
                ttl_override=ttl_override
            )
            
            # Convert to MemoryModel objects
            if isinstance(results, list):
                return [MemoryModel.from_memory(mem) for mem in results]
            return []

    @function_tool
    async def get_memories_by_npc(
        self,
        ctx: RunContextWrapper,
        npc_id: str,
        limit: int = 5
    ) -> List[MemoryModel]:
        """
        Get memories related to a specific NPC.
        
        Args:
            npc_id: ID of the NPC
            limit: Maximum number of memories to return
            
        Returns:
            List of memories related to the NPC
        """
        with trace(workflow_name="get_memories_by_npc"):
            cache_key = f"npc_memories:{self.user_id}:{self.conversation_id}:{npc_id}:{limit}"
            ttl_override = 180  # 3 minutes

            async def fetch_npc_memories():
                logger.debug(f"Fetching memories for NPC {npc_id}")
                query = """
                    SELECT pj.id, pj.entry_type, pj.entry_text, pj.entry_metadata,
                           pj.created_at, pj.importance, pj.access_count, pj.last_accessed, pj.tags
                    FROM PlayerJournal pj
                    WHERE pj.user_id = $1 AND pj.conversation_id = $2
                      AND (
                        pj.entry_metadata::jsonb ? 'npc_id' AND pj.entry_metadata::jsonb->>'npc_id' = $3
                        OR
                        pj.entry_text LIKE '%' || (SELECT npc_name FROM NPCStats WHERE npc_id = $3 LIMIT 1) || '%'
                      )
                    ORDER BY pj.importance DESC, pj.created_at DESC
                    LIMIT $4
                """
                
                memories = []
                try:
                    async with get_db_connection_context() as conn:
                        rows = await conn.fetch(query, self.user_id, self.conversation_id, npc_id, limit)
                        for row_data in rows:
                            memories.append(self._parse_journal_row(row_data))
                    logger.info(f"Fetched {len(memories)} memories for NPC {npc_id}")
                    return memories
                except Exception as e:
                    logger.error(f"Error fetching memories for NPC {npc_id}: {e}", exc_info=True)
                    return []

            # Get from cache or fetch
            results = await context_cache.get(
                cache_key,
                fetch_npc_memories,
                cache_level=1,  # L1 cache
                importance=0.6,
                ttl_override=ttl_override
            )
            
            # Convert to MemoryModel objects
            if isinstance(results, list):
                return [MemoryModel.from_memory(mem) for mem in results]
            return []

    @function_tool
    async def consolidate_memories(
        self,
        ctx: RunContextWrapper,
        rules: MemoryConsolidationRules
    ) -> MemoryConsolidationResult:
        """
        Consolidate memories based on rules.
        
        Args:
            rules: Consolidation rules
            
        Returns:
            Result of the consolidation operation
        """
        with trace(workflow_name="consolidate_memories"):
            logger.info(f"Starting memory consolidation with rules: {rules.dict()}")
            try:
                memories = await self._get_memories_to_consolidate(rules)
                if not memories:
                    logger.info("No memories found matching consolidation criteria.")
                    return MemoryConsolidationResult(
                        processed_count=0,
                        success=True
                    )

                grouped_memories = self._group_memories(memories, rules)
                if not grouped_memories:
                     logger.info("No groups formed for consolidation.")
                     return MemoryConsolidationResult(
                         processed_count=len(memories),
                         consolidated_count=0,
                         success=True
                     )

                summaries = await self._generate_memory_summaries(grouped_memories, rules)
                if not summaries:
                     logger.info("No summaries generated for consolidation.")
                     return MemoryConsolidationResult(
                         processed_count=len(memories),
                         consolidated_count=0,
                         success=True
                     )

                await self._store_consolidated_memories(summaries)

                logger.info(f"Consolidated {len(memories)} memories into {len(summaries)} summaries.")
                return MemoryConsolidationResult(
                    processed_count=len(memories),
                    consolidated_count=len(memories),
                    summary_count=len(summaries),
                    success=True
                )
            except Exception as e:
                logger.error(f"Error consolidating memories: {e}", exc_info=True)
                return MemoryConsolidationResult(
                    success=False,
                    error=str(e)
                )

    @function_tool
    async def run_maintenance(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Run maintenance tasks like memory consolidation asynchronously.
        
        Returns:
            Dictionary with maintenance results
        """
        with trace(workflow_name="run_maintenance"):
            logger.info(f"Running memory maintenance for {self.user_id}:{self.conversation_id}...")
            
            if self.config is None:
                self.config = await get_config()
                
            should_consolidate = self.config.get("memory_consolidation", "enabled", False)

            if not should_consolidate:
                logger.info("Memory consolidation disabled in config.")
                return {"consolidated": False, "reason": "Memory consolidation disabled"}

            days_threshold = self.config.get("memory_consolidation", "days_threshold", 7)
            min_memories = self.config.get("memory_consolidation", "min_memories_to_consolidate", 20)
            cutoff_date = datetime.now() - timedelta(days=days_threshold)

            consolidation_rules = MemoryConsolidationRules(
                time_window_days=days_threshold,
                min_importance=0.0,  # Consolidate potentially low importance old memories
                group_by="type",  # Or configure this
                max_memories_per_group=5,
                min_memories_to_summarize=3,
            )

            try:
                 # Check count *before* running full consolidation for efficiency
                 async with get_db_connection_context() as conn:
                     count_query = """
                         SELECT COUNT(*) as count
                         FROM PlayerJournal
                         WHERE user_id = $1 AND conversation_id = $2
                           AND created_at < $3
                           AND consolidated = FALSE
                     """
                     count_result = await conn.fetchval(count_query, self.user_id, self.conversation_id, cutoff_date)
                     count = count_result or 0

                 if count < min_memories:
                     logger.info(f"Skipping consolidation: Found {count} unconsolidated memories older than {days_threshold} days (minimum {min_memories}).")
                     return {"consolidated": False, "reason": f"Not enough old memories: {count} < {min_memories}"}

                 logger.info(f"Found {count} potential memories for consolidation. Proceeding...")
                 result = await self.consolidate_memories(ctx, consolidation_rules)
                 
                 return {
                     "consolidated": result.success,
                     "checked_count": count,
                     "consolidated_count": result.consolidated_count,
                     "summary_count": result.summary_count,
                     "threshold_days": days_threshold
                 }

            except Exception as e:
                logger.error(f"Error during memory maintenance: {e}", exc_info=True)
                return {"consolidated": False, "error": str(e)}

    async def _search_memories_in_db(
        self,
        query_text: str,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Memory]:
        """Search memories in database using text search, type, and tag filters."""
        memories = []
        if limit <= 0: 
            return memories

        try:
            async with get_db_connection_context() as conn:
                base_query = """
                    SELECT id, entry_type, entry_text, entry_metadata,
                           created_at, importance, access_count, last_accessed, tags, consolidated
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                """
                params: List[Any] = [self.user_id, self.conversation_id]
                conditions: List[str] = []

                # Text search
                if query_text:
                    # Basic ILIKE search
                    param_index = len(params) + 1
                    conditions.append(f"entry_text ILIKE ${param_index}")
                    params.append(f"%{query_text}%")

                # Memory type filter
                if memory_types:
                    type_placeholders = []
                    for mem_type in memory_types:
                        param_index = len(params) + 1
                        type_placeholders.append(f"${param_index}")
                        params.append(mem_type)
                    conditions.append(f"entry_type IN ({', '.join(type_placeholders)})")

                # Tag filter
                if tags:
                    # Assuming tags column is JSONB array of strings
                    param_index = len(params) + 1
                    conditions.append(f"tags @> ${param_index}::jsonb")
                    params.append(json.dumps(tags))

                # Combine conditions
                if conditions:
                    base_query += " AND " + " AND ".join(conditions)

                # Add ordering and limit
                param_index = len(params) + 1
                base_query += f" ORDER BY importance DESC, last_accessed DESC LIMIT ${param_index}"
                params.append(limit)

                # Execute query
                rows = await conn.fetch(base_query, *params)

                # Process results
                for row_data in rows:
                    memories.append(self._parse_journal_row(row_data))

            logger.debug(f"DB search found {len(memories)} memories matching criteria.")
            return memories

        except Exception as e:
            logger.error(f"Error searching memories in DB: {e}", exc_info=True)
            return []

    async def _get_memories_to_consolidate(self, rules: MemoryConsolidationRules) -> List[Memory]:
        """Get memories that should be consolidated based on rules."""
        time_window_days = rules.time_window_days
        min_importance = rules.min_importance
        max_memories = 500  # Hard limit for memory consolidation

        threshold = datetime.now() - timedelta(days=time_window_days)

        query = """
            SELECT id, entry_type, entry_text, entry_metadata,
                   created_at, importance, access_count, last_accessed, tags
            FROM PlayerJournal
            WHERE user_id = $1 AND conversation_id = $2
            AND created_at >= $3
            AND importance >= $4
            AND consolidated = FALSE
            ORDER BY created_at ASC
            LIMIT $5
        """
        memories = []
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    query,
                    self.user_id, self.conversation_id,
                    threshold, min_importance, max_memories
                )
                for row_data in rows:
                     memories.append(self._parse_journal_row(row_data))
            logger.debug(f"Fetched {len(memories)} potential memories for consolidation.")
            return memories
        except Exception as e:
            logger.error(f"Error getting memories to consolidate: {e}", exc_info=True)
            return []

    def _group_memories(self, memories: List[Memory], 
                       rules: MemoryConsolidationRules) -> Dict[str, List[Memory]]:
        """Group memories based on rules."""
        try:
            grouped = defaultdict(list)
            
            # Get grouping key from rules
            group_by = rules.group_by
            
            # Group memories
            for memory in memories:
                if group_by == "type":
                    key = memory.memory_type
                elif group_by == "importance":
                    key = self._get_importance_bracket(memory.importance)
                elif group_by == "recency":
                    key = self._get_recency_bracket(memory.created_at)
                else:
                    key = getattr(memory, group_by, "unknown")
                
                grouped[key].append(memory)
            
            return grouped
        except Exception as e:
            logger.error(f"Error grouping memories: {e}")
            return {}

    def _get_importance_bracket(self, importance: float) -> str:
        """Get importance bracket for grouping."""
        if importance >= 0.8:
            return "critical"
        elif importance >= 0.6:
            return "high"
        elif importance >= 0.4:
            return "medium"
        else:
            return "low"

    def _get_recency_bracket(self, timestamp: datetime) -> str:
        """Get recency bracket for grouping."""
        age = (datetime.now() - timestamp).total_seconds()
        
        if age < 3600:  # 1 hour
            return "recent"
        elif age < 86400:  # 24 hours
            return "today"
        elif age < 604800:  # 1 week
            return "week"
        else:
            return "old"

    async def _generate_memory_summaries(self, grouped_memories: Dict[str, List[Memory]], 
                                       rules: MemoryConsolidationRules) -> List[Memory]:
        """Generate summaries for groups of memories."""
        try:
            summaries = []
            
            for key, memories in grouped_memories.items():
                # Sort memories by importance and recency
                sorted_memories = sorted(
                    memories,
                    key=lambda m: (m.importance, m.created_at),
                    reverse=True
                )
                
                # Get top memories
                top_memories = sorted_memories[:rules.max_memories_per_group]
                
                # Generate summary
                summary_id = f"summary_{key}_{datetime.now().timestamp()}"
                summary = Memory(
                    memory_id=summary_id,
                    content=self._generate_summary_content(top_memories),
                    memory_type="summary",
                    importance=self._calculate_group_importance(top_memories),
                    created_at=datetime.now(),
                    metadata={
                        "group_key": key,
                        "memory_count": len(memories),
                        "time_span": {
                            "start": min(m.created_at for m in memories).isoformat(),
                            "end": max(m.created_at for m in memories).isoformat()
                        }
                    }
                )
                
                summaries.append(summary)
            
            return summaries
        except Exception as e:
            logger.error(f"Error generating memory summaries: {e}")
            return []

    def _generate_summary_content(self, memories: List[Memory]) -> str:
        """Generate content for a memory summary."""
        try:
            # Get key points from memories
            key_points = []
            for memory in memories:
                if memory.memory_type == "event":
                    key_points.append(f"- {memory.content}")
                elif memory.memory_type == "relationship":
                    key_points.append(f"* {memory.content}")
                else:
                    key_points.append(f"• {memory.content}")
            
            # Generate summary
            summary = "Summary of related memories:\n"
            summary += "\n".join(key_points)
            
            return summary
        except Exception as e:
            logger.error(f"Error generating summary content: {e}")
            return "Error generating summary"

    def _calculate_group_importance(self, memories: List[Memory]) -> float:
        """Calculate importance for a group of memories."""
        try:
            if not memories:
                return 0.0
            
            # Calculate weighted average importance
            total_weight = 0.0
            weighted_sum = 0.0
            
            for memory in memories:
                # Weight by recency
                age = (datetime.now() - memory.created_at).total_seconds()
                weight = 1.0 / (1.0 + age / 86400)  # Decay over 24 hours
                
                weighted_sum += memory.importance * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating group importance: {e}")
            return 0.0

    async def _store_consolidated_memories(self, summaries: List[Memory]) -> None:
        """Store consolidated memory summaries in PlayerJournal."""
        if not summaries:
            return

        insert_query = """
            INSERT INTO PlayerJournal(
                user_id, conversation_id, entry_type, entry_text,
                entry_metadata, importance, access_count, last_accessed, created_at, tags,
                consolidated
            )
            VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, TRUE)
        """

        values_list = []
        for summary in summaries:
            metadata_json = None
            try:
                metadata_json = json.dumps(summary.metadata)
            except TypeError as json_err:
                 logger.error(f"Failed to serialize summary metadata for {summary.memory_id}: {json_err}")
                 metadata_json = json.dumps({"error": "serialization failed"})

            tags_json = None
            try:
                tags_json = json.dumps(summary.tags)
            except TypeError as json_err:
                 logger.error(f"Failed to serialize summary tags for {summary.memory_id}: {json_err}")
                 tags_json = json.dumps(["error"])

            values_list.append((
                self.user_id, self.conversation_id,
                summary.memory_type, summary.content, metadata_json,
                summary.importance, summary.access_count, summary.last_accessed, summary.created_at,
                tags_json
            ))

        if not values_list:
             return

        try:
            async with get_db_connection_context() as conn:
                # Use execute_many pattern with asyncpg
                await conn.executemany(insert_query, values_list)
            logger.info(f"Stored {len(summaries)} consolidated memory summaries in PlayerJournal.")
        except Exception as e:
            logger.error(f"Error storing consolidated memories: {e}", exc_info=True)

    async def _load_important_memories(self):
        """Load important memories into local cache using async context."""
        cache_key = f"important_memories:{self.user_id}:{self.conversation_id}"
        ttl_override = 3600  # 1 hour

        async def fetch_important_memories():
            logger.debug(f"Fetching important memories from DB for {self.user_id}:{self.conversation_id}")
            query = """
                SELECT id, entry_type, entry_text, entry_metadata,
                       created_at, importance, access_count, last_accessed, tags, consolidated
                FROM PlayerJournal
                WHERE user_id = $1 AND conversation_id = $2
                  AND (importance >= $3 OR last_accessed > NOW() - INTERVAL '7 days')
                ORDER BY importance DESC, last_accessed DESC
                LIMIT $4
            """
            # Use importance threshold from config + a bit lower for recency check
            importance_thresh = 0.6
            if self.config:
                importance_thresh = self.config.get("memory_cache_importance_threshold", 0.6)
            limit = 100
            if self.config:
                limit = self.config.get("memory_cache_limit", 100)

            memories_dict = {}
            try:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(query, self.user_id, self.conversation_id, importance_thresh, limit)
                    for row_data in rows:
                        memory = self._parse_journal_row(row_data)
                        memories_dict[memory.memory_id] = memory
                logger.info(f"Loaded {len(memories_dict)} important memories from DB for {self.user_id}:{self.conversation_id}")
                return memories_dict
            except Exception as e:
                logger.error(f"Error loading important memories: {e}", exc_info=True)
                return {}

        # Get from cache or fetch
        memories_result = await context_cache.get(
            cache_key,
            fetch_important_memories,
            cache_level=2,  # L2 cache
            importance=0.6,  # Fixed importance for this cache item itself
            ttl_override=ttl_override
        )

        # Store fetched memories in the instance's cache
        self.memories = memories_result if isinstance(memories_result, dict) else {}
        logger.debug(f"Memory cache populated with {len(self.memories)} memories for {self.user_id}:{self.conversation_id}")

    def _build_memory_indices(self):
        """Build memory indices from the self.memories cache"""
        logger.debug(f"Building memory indices from {len(self.memories)} cached memories...")
        # Reset indices
        self.memory_indices = {
            "by_type": defaultdict(list),
            "by_importance": defaultdict(list),
            "by_recency": [],
            "by_tag": defaultdict(set)
        }

        # Temporary list for recency sorting
        recency_list = []

        for memory_id, memory in self.memories.items():
            # Type index
            self.memory_indices["by_type"][memory.memory_type].append(memory_id)

            # Importance index
            bracket = self._get_importance_bracket(memory.importance)
            self.memory_indices["by_importance"][bracket].append(memory_id)

            # Tag index
            for tag in memory.tags:
                self.memory_indices["by_tag"][tag].add(memory_id)

            # Recency list
            recency_list.append((memory.created_at, memory_id))

        # Sort recency list (newest first) and store
        recency_list.sort(key=lambda x: x[0], reverse=True)
        self.memory_indices["by_recency"] = recency_list

        logger.debug("Memory indices built.")

    def _parse_journal_row(self, row) -> Memory:
        """Helper to parse a PlayerJournal row into a Memory object."""
        metadata = {}
        if row["entry_metadata"]:
            try: 
                metadata = json.loads(row["entry_metadata"])
            except (json.JSONDecodeError, TypeError): 
                pass
                
        tags = []
        if row["tags"]:
            try: 
                tags = json.loads(row["tags"])
            except (json.JSONDecodeError, TypeError): 
                pass
                
        if not isinstance(tags, list): 
            tags = []

        return Memory(
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

    async def _handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx."""
        instruction = directive.get("instruction", "")
        logging.info(f"[MemoryManager] Processing action directive: {instruction}")
        
        if "consolidate_memories" in instruction.lower():
            # Apply memory consolidation
            params = directive.get("parameters", {})
            consolidation_rules = MemoryConsolidationRules(**params.get("consolidation_rules", {}))
            
            # Consolidate memories
            result = await self.consolidate_memories(None, consolidation_rules)
            
            return {"result": "memories_consolidated", "success": result.success}
            
        elif "prioritize_memories" in instruction.lower():
            # Apply memory prioritization
            params = directive.get("parameters", {})
            priority_rules = params.get("priority_rules", {})
            
            # Update priority rules
            self._update_priority_rules(priority_rules)
            
            # Re-prioritize memories
            await self._prioritize_memories()
            
            return {"result": "memories_prioritized"}
            
        elif "filter_memories" in instruction.lower():
            # Apply memory filtering
            params = directive.get("parameters", {})
            filter_rules = params.get("filter_rules", {})
            
            # Update filter rules
            self._update_filter_rules(filter_rules)
            
            # Apply filters
            await self._apply_memory_filters()
            
            return {"result": "memories_filtered"}
        
        return {"result": "action_not_recognized"}

    async def _handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx."""
        logging.info(f"[MemoryManager] Processing override directive")
        
        # Extract override details
        override_action = directive.get("override_action", {})
        applies_to = directive.get("applies_to", [])
        
        # Store override
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_overrides[directive_id] = {
                "action": override_action,
                "applies_to": applies_to
            }
        
        return {"result": "override_stored"}

    async def _handle_prohibition_directive(self, directive: dict) -> dict:
        """Handle a prohibition directive from Nyx."""
        logging.info(f"[MemoryManager] Processing prohibition directive")
        
        # Extract prohibition details
        prohibited_actions = directive.get("prohibited_actions", [])
        reason = directive.get("reason", "No reason provided")
        
        # Store prohibition
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_prohibitions[directive_id] = {
                "prohibited_actions": prohibited_actions,
                "reason": reason
            }
        
        return {"result": "prohibition_stored"}

    def _update_priority_rules(self, rules: Dict[str, Any]) -> None:
        """Update priority rules based on Nyx directive."""
        try:
            # Update type scores
            if "type_scores" in rules:
                self.type_scores.update(rules["type_scores"])
            
            # Update importance weights
            if "importance_weights" in rules:
                self.importance_weights.update(rules["importance_weights"])
            
            # Update recency weights
            if "recency_weights" in rules:
                self.recency_weights.update(rules["recency_weights"])
            
            logger.info("Updated priority rules from Nyx directive")
        except Exception as e:
            logger.error(f"Error updating priority rules: {e}")

    def _update_filter_rules(self, rules: Dict[str, Any]) -> None:
        """Update filter rules based on Nyx directive."""
        try:
            # Update inclusion rules
            if "include_only" in rules:
                self.include_rules = set(rules["include_only"])
            
            # Update exclusion rules
            if "exclude" in rules:
                self.exclude_rules = set(rules["exclude"])
            
            # Update importance threshold
            if "importance_threshold" in rules:
                self.importance_threshold = rules["importance_threshold"]
            
            logger.info("Updated filter rules from Nyx directive")
        except Exception as e:
            logger.error(f"Error updating filter rules: {e}")

    async def _prioritize_memories(self) -> None:
        """Re-prioritize memories based on current rules."""
        logger.info("Starting memory re-prioritization...")
        try:
            memories = await self._get_all_memories()
            if not memories:
                 logger.info("No memories found to re-prioritize.")
                 return

            updated_memories = []
            for memory in memories:
                new_importance = self._calculate_memory_importance(memory)
                if abs(new_importance - memory.importance) > 0.01:
                    memory.importance = new_importance
                    updated_memories.append(memory)

            if updated_memories:
                await self._update_memory_importance(updated_memories)
                # Update local cache as well
                for mem in updated_memories:
                     if mem.memory_id in self.memories:
                          self.memories[mem.memory_id].importance = mem.importance
                self._build_memory_indices()

            logger.info(f"Re-prioritized memories. Updated importance for {len(updated_memories)} memories.")
        except Exception as e:
            logger.error(f"Error prioritizing memories: {e}", exc_info=True)

    async def _get_all_memories(self) -> List[Memory]:
        """Retrieve all memories from the database."""
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT id, entry_type, entry_text, entry_metadata,
                           created_at, importance, access_count, last_accessed, tags
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 1000
                """
                
                rows = await conn.fetch(query, self.user_id, self.conversation_id)
                
                memories = []
                for row in rows:
                    memories.append(self._parse_journal_row(row))
                
                return memories
        except Exception as e:
            logger.error(f"Error retrieving all memories: {e}")
            return []

    def _calculate_memory_importance(self, memory: Memory) -> float:
        """Calculate importance score for a memory."""
        try:
            importance = 0.0
            
            # Type-based importance
            type_score = self.type_scores.get(memory.memory_type, 0.5)
            importance += type_score * self.importance_weights["type"]
            
            # Recency-based importance
            age = (datetime.now() - memory.created_at).total_seconds()
            recency_score = 1.0 / (1.0 + age / 86400)  # Decay over 24 hours
            importance += recency_score * self.importance_weights["recency"]
            
            # Access frequency importance
            access_score = min(1.0, memory.access_count / 10)  # Cap at 10 accesses
            importance += access_score * self.importance_weights["access_frequency"]
            
            return min(1.0, importance)  # Cap at 1.0
        except Exception as e:
            logger.error(f"Error calculating memory importance: {e}")
            return 0.5

    async def _update_memory_importance(self, memories: List[Memory]) -> None:
        """Update importance scores for memories in database."""
        if not memories:
            return

        update_query = """
            UPDATE PlayerJournal
            SET importance = $1
            WHERE user_id = $2 AND conversation_id = $3 AND id = $4
        """
        
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    for memory in memories:
                        if not memory.memory_id.isdigit():
                            continue
                        
                        await conn.execute(
                            update_query,
                            memory.importance,
                            self.user_id,
                            self.conversation_id,
                            int(memory.memory_id)
                        )
                        
            logger.info(f"Updated importance for {len(memories)} memories in DB.")
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}", exc_info=True)

    async def _apply_memory_filters(self) -> None:
        """Apply current filter rules to memories (operates on in-memory cache)."""
        logger.info("Applying memory filters to in-memory cache...")
        try:
            # We assume filters apply to the *cached* memories for performance.
            filtered_memories_cache = {}
            for memory_id, memory in self.memories.items():
                if self._should_include_memory(memory):
                    filtered_memories_cache[memory_id] = memory

            self.memories = filtered_memories_cache
            self._build_memory_indices()

            logger.info(f"Applied memory filters. Cache size now {len(self.memories)}.")
        except Exception as e:
            logger.error(f"Error applying memory filters: {e}", exc_info=True)

    def _should_include_memory(self, memory: Memory) -> bool:
        """Check if memory should be included based on filter rules."""
        try:
            # Check inclusion rules
            if self.include_rules and memory.memory_type not in self.include_rules:
                return False
            
            # Check exclusion rules
            if self.exclude_rules and memory.memory_type in self.exclude_rules:
                return False
            
            # Check importance threshold
            if memory.importance < self.importance_threshold:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking memory inclusion: {e}")
            return False

    async def close(self):
         """Perform cleanup if necessary."""
         logger.info(f"Closing MemoryManager for {self.user_id}:{self.conversation_id}.")
         
         # Cancel any pending tasks or background processing
         if self.vector_service:
             try:
                 await self.vector_service.close()
             except Exception as e:
                 logger.error(f"Error closing vector service: {e}")
         
         # Clear in-memory caches
         self.memories.clear()
         self.memory_indices = {
             "by_type": defaultdict(list),
             "by_importance": defaultdict(list),
             "by_recency": [],
             "by_tag": defaultdict(set)
         }
         
         self.is_initialized = False


def create_memory_agent() -> Agent:
    """Create a memory agent using the OpenAI Agents SDK"""
    # This is a placeholder function - in real usage, you'd initialize with actual user_id and conversation_id
    memory_manager = MemoryManager(user_id=0, conversation_id=0)
    
    # Define the agent
    agent = Agent(
        name="Memory Manager",
        instructions="""
        You are a memory manager agent specialized in storing, retrieving, and consolidating memories.
        Your tasks include:
        
        1. Searching for relevant memories based on queries
        2. Adding new memories to the system
        3. Retrieving recent or important memories
        4. Consolidating old memories into summaries
        5. Running maintenance on the memory system
        
        When handling memory operations, prioritize important and relevant information.
        """,
        tools=[
            memory_manager.search_memories,
            memory_manager.add_memory,
            memory_manager.get_memory,
            memory_manager.get_recent_memories,
            memory_manager.get_memories_by_npc,
            memory_manager.consolidate_memories,
            memory_manager.run_maintenance,
        ],
    )
    
    return agent


# --- Global Manager Registry and Factory ---
_memory_managers: Dict[str, MemoryManager] = {}
_manager_lock = asyncio.Lock()  # Lock for managing the registry creation

async def get_memory_manager(user_id: int, conversation_id: int) -> MemoryManager:
    """Get or create a memory manager instance asynchronously."""
    key = f"{user_id}:{conversation_id}"

    # Check if exists first without lock for performance
    manager = _memory_managers.get(key)
    if manager and manager.is_initialized:
        return manager

    # If not found or not initialized, acquire lock to create/initialize
    async with _manager_lock:
        # Double-check if it was created while waiting for the lock
        manager = _memory_managers.get(key)
        if manager and manager.is_initialized:
            return manager

        # Create and initialize the manager
        logger.info(f"Creating new MemoryManager instance for {key}")
        manager = MemoryManager(user_id, conversation_id)
        try:
            await manager.initialize()
            _memory_managers[key] = manager
            return manager
        except Exception as e:
            # Initialization failed, don't store the broken manager
            logger.critical(f"Failed to initialize MemoryManager for {key}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize MemoryManager for {key}") from e


async def cleanup_memory_managers():
    """Close all registered memory managers."""
    global _memory_managers
    logger.info(f"Cleaning up {len(_memory_managers)} memory managers...")
    async with _manager_lock:  # Ensure no creation happens during cleanup
        managers_to_close = list(_memory_managers.values())
        _memory_managers.clear()  # Clear registry immediately

    # Close managers outside the lock
    for manager in managers_to_close:
        try:
            await manager.close()
        except Exception as e:
            logger.error(f"Error closing manager for {manager.user_id}:{manager.conversation_id}: {e}", exc_info=True)
    logger.info("Memory managers cleanup complete.")

def get_memory_agent() -> Agent:
    """Get the memory agent"""
    return create_memory_agent()
