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

from context.models import (
    MemoryMetadata, TimeSpanMetadata,
    MemorySearchRequest as MemorySearchRequestModel,
    MemoryAddRequest as MemoryAddRequestModel
)

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
        metadata: Optional[MemoryMetadata] = None
    ):
        self.memory_id = memory_id
        self.content = content
        self.memory_type = memory_type
        self.created_at = created_at or datetime.now()
        self.importance = importance
        self.access_count = access_count
        self.last_accessed = last_accessed or self.created_at
        self.tags = tags or []
        self.metadata = metadata or MemoryMetadata()
    
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
            "metadata": self.metadata.dict() if self.metadata else {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary"""
        timestamp = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        last_accessed = datetime.fromisoformat(data["last_accessed"]) if isinstance(data["last_accessed"], str) else data["last_accessed"]
        
        # Parse metadata
        metadata_dict = data.get("metadata", {})
        metadata = MemoryMetadata(**metadata_dict) if metadata_dict else MemoryMetadata()
        
        memory = cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            created_at=timestamp,
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
            tags=data.get("tags", []),
            metadata=metadata
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
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    
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
            metadata=memory.metadata if memory.metadata else MemoryMetadata()
        )


# Use the imported models from context.models
MemorySearchRequest = MemorySearchRequestModel
MemoryAddRequest = MemoryAddRequestModel


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
    """
    Integrated memory manager for storage, retrieval, and consolidation.

    NOTE: No more @function_tool on these instance methods. We'll create
    standalone tool functions at the bottom that call these `_xyz` methods.
    """

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
            "by_recency": [],
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
        with trace(workflow_name="memory_manager_init"):
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

    # ---------------------------------------------------------------------
    #   INTERNAL (PRIVATE) METHODS: previously had @function_tool, now removed
    # ---------------------------------------------------------------------

    async def _search_memories(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Internal method: search memories by query text with vector search option."""
        with trace(workflow_name="search_memories"):
            start_time = time.monotonic()
            
            if not self.is_initialized:
                await self.initialize()
            
            # Vector usage check
            effective_use_vector = (
                request.use_vector
                and self.config.is_enabled("use_vector_search")
                and self.vector_service is not None
            )
            
            # Cache key
            cache_key_params = f"{request.query_text}:{request.memory_types}:{request.tags}:{request.limit}:{effective_use_vector}"
            cache_key = f"memory_search:{self.user_id}:{self.conversation_id}:{hashlib.md5(cache_key_params.encode()).hexdigest()}"
            ttl_override = 30
            
            async def perform_search():
                results_map: Dict[str, Memory] = {}

                # 1. Vector Search
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
                                mem_id = metadata.get("memory_id")
                                if mem_id:
                                    memory_ids_from_vector.append(mem_id)
                        
                        # Fetch full memories for these IDs
                        for mem_id in memory_ids_from_vector:
                            memory_obj = await self._get_memory(mem_id)
                            if memory_obj:
                                # Check filters
                                if request.memory_types and memory_obj.memory_type not in request.memory_types:
                                    continue
                                if request.tags and not any(tag in memory_obj.tags for tag in request.tags):
                                    continue
                                results_map[memory_obj.memory_id] = memory_obj
                    except Exception as vec_err:
                        logger.error(f"Vector search failed: {vec_err}", exc_info=True)
                
                # 2. Database / fallback search
                needed = request.limit - len(results_map)
                if needed > 0 or not effective_use_vector:
                    logger.debug(f"Performing DB search for: '{request.query_text}' (need {needed})")
                    try:
                        db_results = await self._search_memories_in_db(
                            query_text=request.query_text,
                            memory_types=request.memory_types,
                            tags=request.tags,
                            limit=request.limit
                        )
                        for mem in db_results:
                            if mem.memory_id not in results_map and len(results_map) < request.limit:
                                results_map[mem.memory_id] = mem
                    except Exception as db_search_err:
                        logger.error(f"DB search failed: {db_search_err}", exc_info=True)
                
                final_results = list(results_map.values())
                # Sort final by importance and recency
                final_results.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
                return final_results[: request.limit]
            
            # get from cache or do actual search
            search_importance = min(0.7, 0.3 + (len(request.query_text) / 100.0))
            results = await context_cache.get(
                cache_key,
                perform_search,
                cache_level=1,
                importance=search_importance,
                ttl_override=ttl_override
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            memory_models = [MemoryModel.from_memory(m) for m in results]
            return MemorySearchResult(
                memories=memory_models,
                query=request.query_text,
                total_found=len(results),
                vector_search_used=effective_use_vector,
                search_time_ms=duration_ms
            )

    async def _add_memory(self, request: MemoryAddRequest) -> MemoryAddResult:
        """Internal method: add a new memory with vector storage integration."""
        with trace(workflow_name="add_memory"):
            placeholder_id = f"temp_{datetime.now().timestamp()}_{hashlib.md5(request.content.encode()).hexdigest()[:6]}"
            
            memory = Memory(
                memory_id=placeholder_id,
                content=request.content,
                memory_type=request.memory_type,
                tags=request.tags or [],
                metadata=request.metadata or MemoryMetadata()
            )
            
            calculated_importance = memory.calculate_importance()
            final_importance = request.importance if request.importance is not None else calculated_importance
            memory.importance = final_importance
    
            # Serialize metadata properly
            metadata_json = None
            if request.metadata:
                try:
                    metadata_json = json.dumps(request.metadata.dict())
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
    
            db_id: Optional[int] = None
            try:
                async with get_db_connection_context() as conn:
                    # Check if conversation exists first
                    conversation_exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = $1 AND user_id = $2)",
                        self.conversation_id, self.user_id
                    )
                    
                    if not conversation_exists:
                        logger.warning(f"Cannot add memory: Conversation {self.conversation_id} does not exist for user {self.user_id}")
                        return MemoryAddResult(
                            success=False, 
                            error=f"Conversation does not exist"
                        )
                    
                    # Continue with memory insertion if conversation exists
                    async with conn.transaction():
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
                            request.memory_type, request.content,
                            metadata_json, final_importance, 0,  # access_count
                            memory.created_at, memory.created_at,  # last_accessed = created_at
                            tags_json
                        )
                
                # Rest of the method remains unchanged
                if db_id is None:
                    logger.error("Failed to add memory to DB (fetchval returned None).")
                    return MemoryAddResult(success=False, error="Database error - could not create memory")
                
                memory.memory_id = str(db_id)
                logger.info(f"Added memory {memory.memory_id} (type: {request.memory_type}, importance: {final_importance:.2f})")
                
                # Add to local cache if above threshold
                cache_importance_threshold = 0.6
                if self.config:
                    cache_importance_threshold = self.config.get("memory_cache_importance_threshold", 0.6)
                if final_importance >= cache_importance_threshold:
                    self.memories[memory.memory_id] = memory
                    self._build_memory_indices()
                    logger.debug(f"Memory {memory.memory_id} added to local cache.")
                
                # Vector embedding if enabled
                if request.store_vector and self.config and self.config.is_enabled("use_vector_search") and self.vector_service:
                    try:
                        await self.vector_service.add_memory(
                            memory_id=memory.memory_id,
                            content=memory.content,
                            memory_type=memory.memory_type,
                            importance=final_importance,
                            tags=request.tags
                        )
                        logger.debug(f"Vector added for memory {memory.memory_id}")
                    except Exception as vec_err:
                        logger.error(f"Failed to add vector for memory {memory.memory_id}: {vec_err}", exc_info=True)
                
                return MemoryAddResult(memory_id=memory.memory_id, success=True)
            except Exception as e:
                logger.error(f"Unexpected error adding memory: {e}", exc_info=True)
                return MemoryAddResult(success=False, error=str(e))

    async def _get_memory(self, memory_id: str) -> Optional[Memory]:
        """Internal method: get a Memory object by ID (no run_context)."""
        if not memory_id or not memory_id.isdigit():
            logger.warning(f"Attempted to get memory with invalid ID: {memory_id}")
            return None
        
        # 1. Check local cache
        if memory_id in self.memories:
            mem = self.memories[memory_id]
            mem.access()
            logger.debug(f"Memory {memory_id} found in local cache.")
            return mem
        
        # 2. Check distributed cache
        cache_key = f"memory:{self.user_id}:{self.conversation_id}:{memory_id}"
        ttl_override = 300
        
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
                        row = await conn.fetchrow(query, self.user_id, self.conversation_id, int(memory_id))
                        if not row:
                            logger.warning(f"Memory {memory_id} not found in DB.")
                            return None
                        mem = self._parse_journal_row(row)
                        
                        # update DB access count/time
                        update_query = """
                            UPDATE PlayerJournal
                            SET access_count = access_count + 1, last_accessed = NOW()
                            WHERE id = $1
                            RETURNING access_count, last_accessed
                        """
                        update_result = await conn.fetchrow(update_query, int(memory_id))
                        if update_result:
                            mem.access_count = update_result["access_count"]
                            mem.last_accessed = update_result["last_accessed"]
                            logger.debug(f"Updated access count for memory {memory_id} in DB.")
                        
                        # maybe add to local cache
                        cache_importance_threshold = 0.6
                        if self.config:
                            cache_importance_threshold = self.config.get("memory_cache_importance_threshold", 0.6)
                        if mem.importance >= cache_importance_threshold:
                            self.memories[mem.memory_id] = mem
                            logger.debug(f"Memory {memory_id} added to local cache after DB fetch.")
                        
                        return mem
            except Exception as e:
                logger.error(f"Error getting memory {memory_id}: {e}", exc_info=True)
                return None
        
        cached_data = await context_cache.get(
            cache_key,
            fetch_memory_from_db,
            cache_level=1,
            importance=0.5,
            ttl_override=ttl_override
        )
        
        if isinstance(cached_data, Memory):
            return cached_data
        elif isinstance(cached_data, dict):
            mem_obj = Memory.from_dict(cached_data)
            if mem_obj and memory_id not in self.memories:
                cache_importance_threshold = 0.6
                if self.config:
                    cache_importance_threshold = self.config.get("memory_cache_importance_threshold", 0.6)
                if mem_obj.importance >= cache_importance_threshold:
                    self.memories[memory_id] = mem_obj
                    logger.debug(f"Memory {memory_id} added to local cache after cache fetch.")
            return mem_obj
        
        return None

    async def _get_recent_memories(
        self,
        days: int = 3,
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Internal method: get recent memory objects, no run_context."""
        cache_key = f"recent_memories:{self.user_id}:{self.conversation_id}:{days}:{memory_types}:{limit}"
        ttl_override = 120
        
        async def fetch_recent_memories():
            logger.debug(f"Fetching recent memories (last {days} days)")
            query = """
                SELECT id, entry_type, entry_text, entry_metadata,
                       created_at, importance, access_count, last_accessed, tags, consolidated
                FROM PlayerJournal
                WHERE user_id = $1 AND conversation_id = $2
                  AND created_at > NOW() - ($3 * INTERVAL '1 day')
            """
            params: List[Any] = [self.user_id, self.conversation_id, days]
            conditions: List[str] = []
            
            if memory_types:
                type_placeholders = []
                for mem_type in memory_types:
                    param_index = len(params) + 1
                    type_placeholders.append(f"${param_index}")
                    params.append(mem_type)
                conditions.append(f"entry_type IN ({', '.join(type_placeholders)})")
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            param_index = len(params) + 1
            query += f" ORDER BY created_at DESC LIMIT ${param_index}"
            params.append(limit)
            
            mems = []
            try:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(query, *params)
                    for row_data in rows:
                        mems.append(self._parse_journal_row(row_data))
                logger.info(f"Fetched {len(mems)} recent memories from DB.")
                return mems
            except Exception as e:
                logger.error(f"Error fetching recent memories: {e}", exc_info=True)
                return []
        
        results = await context_cache.get(
            cache_key,
            fetch_recent_memories,
            cache_level=1,
            importance=0.4,
            ttl_override=ttl_override
        )
        return results if isinstance(results, list) else []

    async def _get_memories_by_npc(self, npc_id: str, limit: int = 5) -> List[Memory]:
        """Internal method: get memory objects related to a specific NPC."""
        cache_key = f"npc_memories:{self.user_id}:{self.conversation_id}:{npc_id}:{limit}"
        ttl_override = 180
        
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
            mems = []
            try:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(query, self.user_id, self.conversation_id, npc_id, limit)
                    for row_data in rows:
                        mems.append(self._parse_journal_row(row_data))
                logger.info(f"Fetched {len(mems)} memories for NPC {npc_id}")
                return mems
            except Exception as e:
                logger.error(f"Error fetching memories for NPC {npc_id}: {e}", exc_info=True)
                return []
        
        results = await context_cache.get(
            cache_key,
            fetch_npc_memories,
            cache_level=1,
            importance=0.6,
            ttl_override=ttl_override
        )
        return results if isinstance(results, list) else []

    async def _consolidate_memories(self, rules: MemoryConsolidationRules) -> MemoryConsolidationResult:
        """Internal method: consolidate memories based on rules."""
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
                return MemoryConsolidationResult(success=False, error=str(e))

    async def _run_maintenance(self) -> Dict[str, Any]:
        """Internal method: run maintenance tasks like memory consolidation asynchronously."""
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
                min_importance=0.0,
                group_by="type",
                max_memories_per_group=5,
                min_memories_to_summarize=3,
            )
            
            try:
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
                    logger.info(f"Skipping consolidation: Found {count} old memories, need {min_memories}.")
                    return {"consolidated": False, "reason": f"Not enough old memories: {count} < {min_memories}"}
                
                logger.info(f"Found {count} potential memories for consolidation. Proceeding...")
                result = await self._consolidate_memories(consolidation_rules)
                
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
        """Search memories in database (text search, type, tag filters)."""
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
                
                # text search
                if query_text:
                    param_index = len(params) + 1
                    conditions.append(f"entry_text ILIKE ${param_index}")
                    params.append(f"%{query_text}%")
                
                # memory type filter
                if memory_types:
                    type_placeholders = []
                    for mem_type in memory_types:
                        param_index = len(params) + 1
                        type_placeholders.append(f"${param_index}")
                        params.append(mem_type)
                    conditions.append(f"entry_type IN ({', '.join(type_placeholders)})")
                
                # tag filter
                if tags:
                    param_index = len(params) + 1
                    conditions.append(f"tags @> ${param_index}::jsonb")
                    params.append(json.dumps(tags))
                
                if conditions:
                    base_query += " AND " + " AND ".join(conditions)
                
                param_index = len(params) + 1
                base_query += f" ORDER BY importance DESC, last_accessed DESC LIMIT ${param_index}"
                params.append(limit)
                
                rows = await conn.fetch(base_query, *params)
                for row_data in rows:
                    memories.append(self._parse_journal_row(row_data))
            
            logger.debug(f"DB search found {len(memories)} memories.")
            return memories
        except Exception as e:
            logger.error(f"Error searching memories in DB: {e}", exc_info=True)
            return []

    async def _get_memories_to_consolidate(self, rules: MemoryConsolidationRules) -> List[Memory]:
        """Get memories that should be consolidated based on rules."""
        time_window_days = rules.time_window_days
        min_importance = rules.min_importance
        max_memories = 500
        
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
        mems = []
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    query,
                    self.user_id, self.conversation_id,
                    threshold, min_importance, max_memories
                )
                for row_data in rows:
                    mems.append(self._parse_journal_row(row_data))
            logger.debug(f"Fetched {len(mems)} potential memories for consolidation.")
            return mems
        except Exception as e:
            logger.error(f"Error getting memories to consolidate: {e}", exc_info=True)
            return []

    def _group_memories(self, memories: List[Memory], rules: MemoryConsolidationRules) -> Dict[str, List[Memory]]:
        """Group memories based on rules."""
        grouped = defaultdict(list)
        group_by = rules.group_by
        
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

    def _get_importance_bracket(self, importance: float) -> str:
        if importance >= 0.8:
            return "critical"
        elif importance >= 0.6:
            return "high"
        elif importance >= 0.4:
            return "medium"
        else:
            return "low"

    def _get_recency_bracket(self, timestamp: datetime) -> str:
        age = (datetime.now() - timestamp).total_seconds()
        if age < 3600:
            return "recent"
        elif age < 86400:
            return "today"
        elif age < 604800:
            return "week"
        else:
            return "old"

    async def _generate_memory_summaries(
        self,
        grouped_memories: Dict[str, List[Memory]],
        rules: MemoryConsolidationRules
    ) -> List[Memory]:
        """Generate summaries for memory groups."""
        summaries = []
        for key, group_list in grouped_memories.items():
            sorted_list = sorted(
                group_list,
                key=lambda m: (m.importance, m.created_at),
                reverse=True
            )
            top_memories = sorted_list[: rules.max_memories_per_group]
            
            summary_id = f"summary_{key}_{datetime.now().timestamp()}"
            content = self._generate_summary_content(top_memories)
            importance = self._calculate_group_importance(top_memories)
            
            # Create metadata with TimeSpanMetadata
            time_span = TimeSpanMetadata(
                start=min(m.created_at for m in group_list).isoformat(),
                end=max(m.created_at for m in group_list).isoformat()
            )
            
            metadata = MemoryMetadata(
                group_key=key,
                memory_count=len(group_list),
                time_span=time_span
            )
            
            summary = Memory(
                memory_id=summary_id,
                content=content,
                memory_type="summary",
                importance=importance,
                created_at=datetime.now(),
                metadata=metadata
            )
            summaries.append(summary)
        return summaries

    def _generate_summary_content(self, memories: List[Memory]) -> str:
        try:
            key_points = []
            for mem in memories:
                if mem.memory_type == "event":
                    key_points.append(f"- {mem.content}")
                elif mem.memory_type == "relationship":
                    key_points.append(f"* {mem.content}")
                else:
                    key_points.append(f"• {mem.content}")
            summary = "Summary of related memories:\n" + "\n".join(key_points)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary content: {e}")
            return "Error generating summary"

    def _calculate_group_importance(self, memories: List[Memory]) -> float:
        if not memories:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for mem in memories:
            age = (datetime.now() - mem.created_at).total_seconds()
            weight = 1.0 / (1.0 + age / 86400)
            weighted_sum += mem.importance * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def _store_consolidated_memories(self, summaries: List[Memory]) -> None:
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
            try:
                metadata_json = json.dumps(summary.metadata)
            except TypeError:
                metadata_json = json.dumps({"error": "serialization failed"})
            try:
                tags_json = json.dumps(summary.tags)
            except TypeError:
                tags_json = json.dumps(["error"])
            
            values_list.append((
                self.user_id, self.conversation_id,
                summary.memory_type, summary.content,
                metadata_json, summary.importance,
                summary.access_count, summary.last_accessed,
                summary.created_at, tags_json
            ))
        
        if not values_list:
            return
        try:
            async with get_db_connection_context() as conn:
                await conn.executemany(insert_query, values_list)
            logger.info(f"Stored {len(summaries)} consolidated memory summaries.")
        except Exception as e:
            logger.error(f"Error storing consolidated memories: {e}", exc_info=True)

    async def _load_important_memories(self):
        """Load important memories into local cache using async context."""
        cache_key = f"important_memories:{self.user_id}:{self.conversation_id}"
        ttl_override = 3600
        
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
            importance_thresh = 0.6
            if self.config:
                importance_thresh = self.config.get("memory_cache_importance_threshold", 0.6)
            limit = 100
            if self.config:
                limit = self.config.get("memory_cache_limit", 100)
            
            mem_dict = {}
            try:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(query, self.user_id, self.conversation_id, importance_thresh, limit)
                    for row_data in rows:
                        mem_obj = self._parse_journal_row(row_data)
                        mem_dict[mem_obj.memory_id] = mem_obj
                logger.info(f"Loaded {len(mem_dict)} important memories from DB.")
                return mem_dict
            except Exception as e:
                logger.error(f"Error loading important memories: {e}", exc_info=True)
                return {}
        
        memories_result = await context_cache.get(
            cache_key,
            fetch_important_memories,
            cache_level=2,
            importance=0.6,
            ttl_override=ttl_override
        )
        if isinstance(memories_result, dict):
            self.memories = memories_result
        else:
            self.memories = {}
        logger.debug(f"Memory cache populated with {len(self.memories)} memories.")

    def _build_memory_indices(self):
        """Build memory indices from the self.memories cache."""
        logger.debug(f"Building memory indices from {len(self.memories)} cached memories...")
        self.memory_indices = {
            "by_type": defaultdict(list),
            "by_importance": defaultdict(list),
            "by_recency": [],
            "by_tag": defaultdict(set)
        }
        recency_list = []
        for mem_id, mem in self.memories.items():
            self.memory_indices["by_type"][mem.memory_type].append(mem_id)
            bracket = self._get_importance_bracket(mem.importance)
            self.memory_indices["by_importance"][bracket].append(mem_id)
            for tag in mem.tags:
                self.memory_indices["by_tag"][tag].add(mem_id)
            recency_list.append((mem.created_at, mem_id))
        recency_list.sort(key=lambda x: x[0], reverse=True)
        self.memory_indices["by_recency"] = recency_list
        logger.debug("Memory indices built.")

    def _parse_journal_row(self, row) -> Memory:
        metadata = None
        if row["entry_metadata"]:
            try:
                metadata_dict = json.loads(row["entry_metadata"])
                metadata = MemoryMetadata(**metadata_dict)
            except (json.JSONDecodeError, TypeError, ValidationError):
                metadata = MemoryMetadata()
        
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
        instruction = directive.get("instruction", "")
        logging.info(f"[MemoryManager] Processing action directive: {instruction}")
        
        if "consolidate_memories" in instruction.lower():
            params = directive.get("parameters", {})
            consolidation_rules = MemoryConsolidationRules(**params.get("consolidation_rules", {}))
            result = await self._consolidate_memories(consolidation_rules)
            return {"result": "memories_consolidated", "success": result.success}
        elif "prioritize_memories" in instruction.lower():
            params = directive.get("parameters", {})
            priority_rules = params.get("priority_rules", {})
            self._update_priority_rules(priority_rules)
            await self._prioritize_memories()
            return {"result": "memories_prioritized"}
        elif "filter_memories" in instruction.lower():
            params = directive.get("parameters", {})
            filter_rules = params.get("filter_rules", {})
            self._update_filter_rules(filter_rules)
            await self._apply_memory_filters()
            return {"result": "memories_filtered"}
        
        return {"result": "action_not_recognized"}

    async def _handle_override_directive(self, directive: dict) -> dict:
        logging.info(f"[MemoryManager] Processing override directive")
        override_action = directive.get("override_action", {})
        applies_to = directive.get("applies_to", [])
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_overrides[directive_id] = {
                "action": override_action,
                "applies_to": applies_to
            }
        return {"result": "override_stored"}

    async def _handle_prohibition_directive(self, directive: dict) -> dict:
        logging.info(f"[MemoryManager] Processing prohibition directive")
        prohibited_actions = directive.get("prohibited_actions", [])
        reason = directive.get("reason", "No reason provided")
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_prohibitions[directive_id] = {
                "prohibited_actions": prohibited_actions,
                "reason": reason
            }
        return {"result": "prohibition_stored"}

    def _update_priority_rules(self, rules: Dict[str, Any]) -> None:
        try:
            if "type_scores" in rules:
                self.type_scores.update(rules["type_scores"])
            if "importance_weights" in rules:
                self.importance_weights.update(rules["importance_weights"])
            if "recency_weights" in rules:
                self.recency_weights.update(rules["recency_weights"])
            logger.info("Updated priority rules from Nyx directive")
        except Exception as e:
            logger.error(f"Error updating priority rules: {e}")

    async def _prioritize_memories(self) -> None:
        logger.info("Starting memory re-prioritization...")
        try:
            all_mems = await self._get_all_memories()
            if not all_mems:
                logger.info("No memories found to re-prioritize.")
                return
            updated_mems = []
            for mem in all_mems:
                new_imp = self._calculate_memory_importance(mem)
                if abs(new_imp - mem.importance) > 0.01:
                    mem.importance = new_imp
                    updated_mems.append(mem)
            if updated_mems:
                await self._update_memory_importance(updated_mems)
                for m in updated_mems:
                    if m.memory_id in self.memories:
                        self.memories[m.memory_id].importance = m.importance
                self._build_memory_indices()
            logger.info(f"Re-prioritized {len(updated_mems)} memories.")
        except Exception as e:
            logger.error(f"Error prioritizing memories: {e}", exc_info=True)

    async def _get_all_memories(self) -> List[Memory]:
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
                return [self._parse_journal_row(r) for r in rows]
        except Exception as e:
            logger.error(f"Error retrieving all memories: {e}")
            return []

    def _calculate_memory_importance(self, memory: Memory) -> float:
        try:
            importance = 0.0
            type_score = self.type_scores.get(memory.memory_type, 0.5)
            importance += type_score * self.importance_weights["type"]
            
            age = (datetime.now() - memory.created_at).total_seconds()
            recency_score = 1.0 / (1.0 + age / 86400)
            importance += recency_score * self.importance_weights["recency"]
            
            access_score = min(1.0, memory.access_count / 10)
            importance += access_score * self.importance_weights["access_frequency"]
            
            return min(1.0, importance)
        except Exception as e:
            logger.error(f"Error calculating memory importance: {e}")
            return 0.5

    async def _update_memory_importance(self, memories: List[Memory]) -> None:
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
                    for mem in memories:
                        if not mem.memory_id.isdigit():
                            continue
                        await conn.execute(
                            update_query,
                            mem.importance,
                            self.user_id,
                            self.conversation_id,
                            int(mem.memory_id)
                        )
            logger.info(f"Updated importance for {len(memories)} memories in DB.")
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}", exc_info=True)

    async def _apply_memory_filters(self) -> None:
        logger.info("Applying memory filters to in-memory cache...")
        try:
            filtered_cache = {}
            for mem_id, mem in self.memories.items():
                if self._should_include_memory(mem):
                    filtered_cache[mem_id] = mem
            self.memories = filtered_cache
            self._build_memory_indices()
            logger.info(f"Applied memory filters. Cache size now {len(self.memories)}.")
        except Exception as e:
            logger.error(f"Error applying memory filters: {e}", exc_info=True)

    def _should_include_memory(self, memory: Memory) -> bool:
        try:
            if self.include_rules and memory.memory_type not in self.include_rules:
                return False
            if self.exclude_rules and memory.memory_type in self.exclude_rules:
                return False
            if memory.importance < self.importance_threshold:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking memory inclusion: {e}")
            return False

    async def close(self):
        """Perform cleanup if necessary."""
        logger.info(f"Closing MemoryManager for {self.user_id}:{self.conversation_id}.")
        if self.vector_service:
            try:
                await self.vector_service.close()
            except Exception as e:
                logger.error(f"Error closing vector service: {e}")
        self.memories.clear()
        self.memory_indices = {
            "by_type": defaultdict(list),
            "by_importance": defaultdict(list),
            "by_recency": [],
            "by_tag": defaultdict(set)
        }
        self.is_initialized = False

# ---------------------------------------------------------------------
#  STANDALONE TOOL FUNCTIONS (with run_context FIRST) that call the _ methods
# ---------------------------------------------------------------------

_memory_managers: Dict[str, MemoryManager] = {}
_manager_lock = asyncio.Lock()


async def get_memory_manager(user_id: int, conversation_id: int) -> MemoryManager:
    """Get or create a memory manager instance asynchronously."""
    key = f"{user_id}:{conversation_id}"
    mgr = _memory_managers.get(key)
    if mgr and mgr.is_initialized:
        return mgr
    
    async with _manager_lock:
        mgr = _memory_managers.get(key)
        if mgr and mgr.is_initialized:
            return mgr
        logger.info(f"Creating new MemoryManager instance for {key}")
        mgr = MemoryManager(user_id, conversation_id)
        try:
            await mgr.initialize()
            _memory_managers[key] = mgr
            return mgr
        except Exception as e:
            logger.critical(f"Failed to initialize MemoryManager for {key}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize MemoryManager for {key}") from e


async def cleanup_memory_managers():
    """Close all registered memory managers."""
    global _memory_managers
    logger.info(f"Cleaning up {len(_memory_managers)} memory managers...")
    async with _manager_lock:
        managers_to_close = list(_memory_managers.values())
        _memory_managers.clear()

    for m in managers_to_close:
        try:
            await m.close()
        except Exception as e:
            logger.error(f"Error closing manager for {m.user_id}:{m.conversation_id}: {e}", exc_info=True)
    logger.info("Memory managers cleanup complete.")

# ---------------------------------------------------------------------
#    Agent Tools: Each @function_tool has ctx first, calls internal methods
# ---------------------------------------------------------------------

@function_tool
async def search_memories_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    request: MemorySearchRequest
) -> MemorySearchResult:
    mgr = await get_memory_manager(user_id, conversation_id)
    return await mgr._search_memories(request)

@function_tool
async def add_memory_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    request: MemoryAddRequest
) -> MemoryAddResult:
    mgr = await get_memory_manager(user_id, conversation_id)
    return await mgr._add_memory(request)

@function_tool
async def get_memory_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    memory_id: str
) -> Optional[MemoryModel]:
    mgr = await get_memory_manager(user_id, conversation_id)
    mem = await mgr._get_memory(memory_id)
    return MemoryModel.from_memory(mem) if mem else None

@function_tool
async def get_recent_memories_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    days: int = 3,
    memory_types: Optional[List[str]] = None,
    limit: int = 10
) -> List[MemoryModel]:
    mgr = await get_memory_manager(user_id, conversation_id)
    mems = await mgr._get_recent_memories(days, memory_types, limit)
    return [MemoryModel.from_memory(m) for m in mems]

@function_tool
async def get_memories_by_npc_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    npc_id: str,
    limit: int = 5
) -> List[MemoryModel]:
    mgr = await get_memory_manager(user_id, conversation_id)
    mems = await mgr._get_memories_by_npc(npc_id, limit)
    return [MemoryModel.from_memory(m) for m in mems]

@function_tool
async def consolidate_memories_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    rules: MemoryConsolidationRules
) -> MemoryConsolidationResult:
    mgr = await get_memory_manager(user_id, conversation_id)
    return await mgr._consolidate_memories(rules)

@function_tool
async def run_maintenance_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    mgr = await get_memory_manager(user_id, conversation_id)
    return await mgr._run_maintenance()

# ---------------------------------------------------------------------
#   Create Agent with the standalone tool functions
# ---------------------------------------------------------------------

def create_memory_agent() -> Agent:
    """Create a memory agent using the OpenAI Agents SDK."""
    # We no longer register the instance methods as tools;
    # we register our new top-level @function_tool methods:
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
            search_memories_tool,
            add_memory_tool,
            get_memory_tool,
            get_recent_memories_tool,
            get_memories_by_npc_tool,
            consolidate_memories_tool,
            run_maintenance_tool,
        ],
    )
    return agent

def get_memory_agent() -> Agent:
    """Get the memory agent"""
    return create_memory_agent()
