# memory/memory_orchestrator.py

"""
Memory Orchestrator with Scene Bundle Optimization
Complete refactor for optimized context assembly pipeline
"""

import logging
import asyncio
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Lazy imports for heavy dependencies
_memory_core = None
_UnifiedMemoryManager = None
_Memory = None
_MemoryType = None
_MemoryStatus = None
_MemorySignificance = None
_MemoryCache = None
_game_time_helper_module = None

logger = logging.getLogger(__name__)


# ============================================================================
# Scene Bundle Data Classes
# ============================================================================

@dataclass
class SceneScope:
    """Defines the scope of a scene for context filtering"""
    location_id: Optional[str] = None
    npc_ids: Set[int] = field(default_factory=set)
    topics: Set[str] = field(default_factory=set)
    lore_tags: Set[str] = field(default_factory=set)
    time_window_hours: int = 24  # Recent memories window
    
    def to_cache_key(self) -> str:
        """Generate stable cache key for this scope"""
        # Sort sets for consistent hashing
        key_parts = [
            f"loc:{self.location_id or 'none'}",
            f"npcs:{','.join(map(str, sorted(self.npc_ids)))}",
            f"topics:{','.join(sorted(self.topics))}",
            f"lore:{','.join(sorted(self.lore_tags))}",
            f"window:{self.time_window_hours}"
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()


@dataclass
class MemoryBundle:
    """Scene-scoped memory bundle for efficient context assembly"""
    # Canonical memory data
    canon_memories: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scene-relevant memories  
    scene_memories: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cross-entity patterns
    linked_memories: List[Dict[str, Any]] = field(default_factory=list)
    emergent_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    scope: Optional[SceneScope] = None
    last_changed_at: float = field(default_factory=time.time)
    bundle_size_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "canon": self.canon_memories,
            "scene": self.scene_memories,
            "linked": self.linked_memories,
            "patterns": self.emergent_patterns,
            "last_changed": self.last_changed_at,
            "token_size": self.bundle_size_tokens
        }


# ============================================================================
# Lazy Import Functions
# ============================================================================

def _lazy_import_memory_core():
    """Lazy import for memory core components."""
    global _memory_core, _UnifiedMemoryManager, _Memory, _MemoryType, _MemoryStatus, _MemorySignificance, _MemoryCache
    
    if _memory_core is None:
        import memory.core as _memory_core
        _UnifiedMemoryManager = _memory_core.UnifiedMemoryManager
        _Memory = _memory_core.Memory
        _MemoryType = _memory_core.MemoryType
        _MemoryStatus = _memory_core.MemoryStatus
        _MemorySignificance = _memory_core.MemorySignificance
        _MemoryCache = _memory_core.MemoryCache
    
    return _UnifiedMemoryManager, _Memory, _MemoryType, _MemoryStatus, _MemorySignificance, _MemoryCache


def _lazy_import_game_time_helper():
    """Lazy import for game time helper functions."""
    global _game_time_helper_module
    if _game_time_helper_module is None:
        from logic import game_time_helper as _game_time_helper_module
    return _game_time_helper_module



    
# ============================================================================
# Entity Types
# ============================================================================

class EntityType(str, Enum):
    """Supported entity types in the memory system."""
    PLAYER = "player"
    NPC = "npc"
    NYX = "nyx"  # The DM/narrator
    LOCATION = "location"
    ITEM = "item"
    LORE = "lore"
    CONFLICT = "conflict"
    CONTEXT = "context"
    MEMORY = "memory"  # Generic memory type
    NARRATIVE = "narrative"


# ============================================================================
# Memory Orchestrator
# ============================================================================

class MemoryOrchestrator:
    """
    Central orchestrator for ALL memory operations with scene bundle optimization.
    
    This class serves as the single access point for:
    - Core memory operations (store, retrieve, update, delete)
    - Scene-scoped memory bundles for efficient context assembly
    - Specialized memory managers (NPC, Player, Nyx, etc.)
    - Advanced features (schemas, emotional, masks, flashbacks, etc.)
    - Memory agents and LLM-based operations
    - Vector store and embedding operations
    - Integration with Nyx governance
    - Telemetry and maintenance
    
    Performance optimizations:
    - Scene-scoped bundle caching
    - Parallel memory retrieval
    - Delta-based updates
    - Background maintenance tasks
    """
    
    _instances = {}  # Singleton pattern per user/conversation
    _instances_lock = asyncio.Lock()  # Protect against rare races in get_instance
    _bundle_cache = {}  # Scene bundle cache across conversations
    _bundle_ttl = 300  # 5 minute TTL for bundles
    _bundle_max_size = 512  # Max bundles in cache (soft LRU)
    
    # Class-wide bundle bookkeeping (keeps class-level cache coherent)
    _bundle_cached_at: Dict[str, float] = {}  # cache_key -> last access time (for LRU)
    _bundle_created_at: Dict[str, float] = {}  # cache_key -> creation time (for TTL)
    _bundle_index: Dict[str, Set[str]] = defaultdict(set)  # "kind:id" -> set(cache_keys)
    _bundle_locks: Dict[str, asyncio.Lock] = {}  # cache_key -> lock
    
    # Semaphore to limit concurrent vector searches
    _vector_search_semaphore = asyncio.Semaphore(10)

    @classmethod
    def _sem(cls) -> asyncio.Semaphore:
        if getattr(cls, "_vector_search_semaphore", None) is None:
            cls._vector_search_semaphore = asyncio.Semaphore(10)
        return cls._vector_search_semaphore
    
    @classmethod
    def _get_lock(cls, key: str) -> asyncio.Lock:
        """Get or create a lock for the given cache key (race-proof)."""
        lock = cls._bundle_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            existing = cls._bundle_locks.setdefault(key, lock)
            lock = existing  # Ensure we use the canonical one
        return lock

    @classmethod
    def _evict_bundle_key(cls, cache_key: str) -> None:
        entry = cls._bundle_cache.get(cache_key) or {}
        ents = entry.get('ents', set())
        for ent in ents:
            s = cls._bundle_index.get(ent)
            if s:
                s.discard(cache_key)
                if not s:
                    cls._bundle_index.pop(ent, None)
        cls._bundle_cache.pop(cache_key, None)
        cls._bundle_cached_at.pop(cache_key, None)
        cls._bundle_created_at.pop(cache_key, None)
        cls._bundle_locks.pop(cache_key, None)
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'MemoryOrchestrator':
        """
        Get or create an orchestrator instance for the given user/conversation.
        Thread-safe against concurrent initialization.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            MemoryOrchestrator instance
        """
        key = (user_id, conversation_id)
        
        # Fast path - check without lock first
        if key in cls._instances:
            return cls._instances[key]
        
        # Slow path - acquire lock for initialization
        async with cls._instances_lock:
            # Double-check after acquiring lock
            if key not in cls._instances:
                instance = cls(user_id, conversation_id)
                await instance.initialize()
                cls._instances[key] = instance
            return cls._instances[key]
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the orchestrator."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core components (lazily initialized)
        self.unified_manager = None
        self.memory_managers = {}
        self.embedding_service = None
        self.cache = None
        
        # Agent wrappers
        self.memory_agent_wrapper = None
        self.retriever_agent = None
        
        # Specialized managers
        self.nyx_memory_manager = None
        self.emotional_manager = None
        self.schema_manager = None
        self.semantic_manager = None
        self.flashback_manager = None
        self.mask_manager = None
        self.interference_manager = None
        
        # State
        self.initialized = False
        self.last_sync = datetime.now()
        self.last_maintenance = datetime.now()
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    async def initialize(self):
        """Initialize all memory subsystems."""
        if self.initialized:
            return
        
        try:
            # Lazy import core components
            UnifiedMemoryManager, _, _, _, _, _ = _lazy_import_memory_core()
            
            # Initialize unified manager
            self.unified_manager = UnifiedMemoryManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            await self.unified_manager.initialize()
            
            # Initialize cache
            from memory.cache import MemoryCache
            self.cache = MemoryCache(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            # Initialize embedding service
            from memory.embedding_service import MemoryEmbeddingService
            self.embedding_service = MemoryEmbeddingService(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            # Initialize agent wrappers
            from memory.memory_agent import MemoryAgentWrapper
            self.memory_agent_wrapper = MemoryAgentWrapper(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            from memory.retriever_agent import RetrieverAgent
            self.retriever_agent = RetrieverAgent(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            # Initialize specialized managers
            from memory.nyx_memory import NyxMemoryManager
            self.nyx_memory_manager = NyxMemoryManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                unified_manager=self.unified_manager
            )
            
            from memory.emotional_memory import EmotionalMemoryManager
            self.emotional_manager = EmotionalMemoryManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            from memory.schema_manager import SchemaManager
            self.schema_manager = SchemaManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            from memory.semantic_manager import SemanticManager
            self.semantic_manager = SemanticManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            from memory.flashback_manager import FlashbackManager
            self.flashback_manager = FlashbackManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            from memory.mask_manager import MaskManager
            self.mask_manager = MaskManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            from memory.interference_manager import InterferenceManager
            self.interference_manager = InterferenceManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            self.initialized = True
            logger.info(f"Memory orchestrator initialized for user {self.user_id}, conversation {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory orchestrator: {e}")
            raise
    
    # ========================================================================
    # Scene Bundle Operations (NEW - Performance Optimization)
    # ========================================================================
    
    @staticmethod
    def _as_mem_list(res: Any) -> List[Dict[str, Any]]:
        if not res:
            return []
        if isinstance(res, list):
            return res
        if isinstance(res, dict):
            for key in ("memories", "results", "items"):
                val = res.get(key)
                if isinstance(val, list):
                    return val
            return []
        return []
    
    def _index_bundle(self, cache_key: str, scope: SceneScope) -> Set[str]:
        """Index a bundle cache entry for efficient invalidation (class-wide)."""
        cls = self.__class__
        # Build the entity set
        ents = {f"player:{self.user_id}"}
        if scope.location_id:
            ents.add(f"location:{scope.location_id}")
        for nid in scope.npc_ids:
            ents.add(f"npc:{nid}")
        for t in scope.topics:
            ents.add(f"topic:{t}")
        for lt in scope.lore_tags:
            ents.add(f"lore:{lt}")
        # Index each entity
        for ent in ents:
            cls._bundle_index[ent].add(cache_key)
        # Store timestamps
        now = time.monotonic()
        cls._bundle_created_at[cache_key] = now  # For TTL
        cls._bundle_cached_at[cache_key] = now   # For LRU
        # Return the ents set for storage
        return ents
    
    async def get_scene_bundle(self, scope: SceneScope, token_budget: int = 5000) -> Dict[str, Any]:
        """
        Get a scene-scoped memory bundle for efficient context assembly.
        
        This is the primary method for the new context assembly pipeline.
        Returns only memories relevant to the current scene scope.
        
        Args:
            scope: Scene scope defining what memories to include
            token_budget: Maximum token budget for the bundle (default 5000)
            
        Returns:
            Dictionary with canon memories, scene memories, linked memories, patterns
        """
        if not self.initialized:
            await self.initialize()
        
        cls = self.__class__
        cache_key = f"{self.user_id}:{self.conversation_id}:{scope.to_cache_key()}"
        
        # Fast path with separate TTL and LRU tracking
        cached = cls._bundle_cache.get(cache_key)
        if cached:
            created_at = cls._bundle_created_at.get(cache_key, 0.0)
            # Check TTL based on creation time
            if time.monotonic() - created_at < cls._bundle_ttl:
                cls._bundle_cached_at[cache_key] = time.monotonic()  # Touch for LRU
                logger.debug(f"Bundle cache hit for {cache_key[:50]}...")
                return cached['bundle']
            else:
                logger.debug(f"Bundle cache expired for {cache_key[:50]}...")
        
        logger.debug(f"Bundle cache miss for {cache_key[:50]}...")
        
        # Prevent duplicate builds across instances
        async with cls._get_lock(cache_key):
            cached = cls._bundle_cache.get(cache_key)
            if cached:
                created_at = cls._bundle_created_at.get(cache_key, 0.0)
                if time.monotonic() - created_at < cls._bundle_ttl:
                    cls._bundle_cached_at[cache_key] = time.monotonic()  # Touch for LRU
                    return cached['bundle']
            
            bundle = await self._build_scene_bundle(scope, token_budget)
            payload = bundle.to_dict()
            
            # Index and get the entity set
            ents = self._index_bundle(cache_key, scope)
            
            # Cache with entity set for proper cleanup (class-wide)
            cls._bundle_cache[cache_key] = {
                'bundle': payload,
                'timestamp': time.time(),
                'scope': scope,
                'ents': ents,  # Store exact keys used for reverse index
            }
            self._cleanup_bundle_cache()
            return payload
    
    async def get_scene_delta(self, scope: SceneScope, since_timestamp: float) -> Dict[str, Any]:
        """
        Get only memories that changed since the given timestamp.
        """
        if not self.initialized:
            await self.initialize()
    
        # Guard against invalid timestamps
        if not since_timestamp or since_timestamp <= 0:
            since_timestamp = time.time() - 86400  # 24h fallback
    
        # Clamp to reasonable recent window (max 7 days ago)
        max_age = time.time() - (7 * 86400)
        if since_timestamp < max_age:
            since_timestamp = max_age
    
        since_dt = datetime.fromtimestamp(since_timestamp)
        iso_since = since_dt.isoformat()
    
        tasks = []
    
        # Player delta
        tasks.append(self.retrieve_memories(
            entity_type=EntityType.PLAYER.value,
            entity_id=self.user_id,
            filters={'modified_after': iso_since, 'location': scope.location_id} if scope.location_id else {'modified_after': iso_since},
            limit=30,
        ))
    
        # NPC deltas
        for npc_id in scope.npc_ids:
            tasks.append(self.retrieve_memories(
                entity_type=EntityType.NPC.value,
                entity_id=npc_id,
                filters={'modified_after': iso_since},
                limit=20,
            ))
    
        # Location/topic/lore deltas via vector store (if supported by filters)
        vs_filters = {'modified_after': iso_since}
        if scope.location_id:
            tasks.append(self.search_vector_store(
                query=f"location:{scope.location_id}",
                filter_dict={**vs_filters, 'location': scope.location_id},
                top_k=15
            ))
        for topic in scope.topics:
            tasks.append(self.search_vector_store(
                query=topic, filter_dict={**vs_filters, 'tags': topic}, top_k=10
            ))
        for lt in scope.lore_tags:
            tasks.append(self.search_vector_store(
                query=lt, filter_dict={**vs_filters, 'lore_tags': lt}, top_k=10
            ))
    
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
        # Flatten + normalize
        flat: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception) or r is None:
                continue
            flat.extend(self._as_mem_list(r) if not (isinstance(r, dict) and 'memories' in r)
                        else r['memories'])
    
        # Dedup by memory_id
        seen = set()
        unique = []
        for m in flat:
            mid = m.get('memory_id') or m.get('id')
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(m)
    
        # Split canon vs scene
        canon = [m for m in unique if m.get('is_canon') or m.get('canonical')]
        scene = [m for m in unique if not (m.get('is_canon') or m.get('canonical'))]
    
        bundle = MemoryBundle(
            canon_memories=canon,
            scene_memories=scene,
            linked_memories=[],
            emergent_patterns=[],
            scope=scope,
            last_changed_at=time.time(),
        )
        bundle.bundle_size_tokens = self._estimate_bundle_tokens(bundle)
        return bundle.to_dict()
    
    async def _build_scene_bundle(self, scope: SceneScope, token_budget: int = 5000) -> MemoryBundle:
        """
        Build a complete scene bundle from scratch.
    
        Uses parallel fetching for efficiency with limits on concurrent searches.
        """
        # Cap pathological scopes
        MAX_NPCS = 20
        MAX_TOPICS = 10
        MAX_LORE_TAGS = 10
    
        # Parallel fetch all memory types
        tasks = []
    
        # Get player memories
        tasks.append(self._get_player_memories_for_scope(scope))
    
        # Get NPC memories for all NPCs in scope (capped)
        npc_ids = list(scope.npc_ids)[:MAX_NPCS]
        if len(scope.npc_ids) > MAX_NPCS:
            logger.warning(f"Capped NPCs from {len(scope.npc_ids)} to {MAX_NPCS}")
        for npc_id in npc_ids:
            tasks.append(self._get_npc_memories_for_scope(npc_id, scope))
    
        # Get location memories if location specified
        if scope.location_id:
            tasks.append(self._get_location_memories_for_scope(scope))
    
        # Get topic-related memories (capped)
        topics = list(scope.topics)[:MAX_TOPICS]
        if len(scope.topics) > MAX_TOPICS:
            logger.warning(f"Capped topics from {len(scope.topics)} to {MAX_TOPICS}")
        if topics:
            limited_scope = SceneScope(
                location_id=scope.location_id,
                npc_ids=scope.npc_ids,
                topics=set(topics),
                lore_tags=scope.lore_tags,
                time_window_hours=scope.time_window_hours
            )
            tasks.append(self._get_topic_memories_for_scope(limited_scope))
    
        # Get lore-tagged memories (capped)
        lore_tags = list(scope.lore_tags)[:MAX_LORE_TAGS]
        if len(scope.lore_tags) > MAX_LORE_TAGS:
            logger.warning(f"Capped lore tags from {len(scope.lore_tags)} to {MAX_LORE_TAGS}")
        if lore_tags:
            limited_scope = SceneScope(
                location_id=scope.location_id,
                npc_ids=scope.npc_ids,
                topics=scope.topics,
                lore_tags=set(lore_tags),
                time_window_hours=scope.time_window_hours
            )
            tasks.append(self._get_lore_memories_for_scope(limited_scope))
    
        # Execute all fetches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
        # Process results safely
        all_memories: List[Dict[str, Any]] = []
        canon_memories: List[Dict[str, Any]] = []
    
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching memories for scope: {result}")
                continue
            all_memories.extend(self._as_mem_list(result))
    
        # Separate canon from regular memories
        for memory in all_memories:
            if memory.get('is_canon') or memory.get('canonical'):
                canon_memories.append(memory)
    
        # Find linked memories and patterns (limited computation)
        linked_memories = await self._find_linked_memories(all_memories, scope, limit=10)
        patterns = await self._find_emergent_patterns(all_memories, scope, limit=5)
    
        # Build bundle
        bundle = MemoryBundle(
            canon_memories=canon_memories[:20],   # Limit canon memories
            scene_memories=all_memories[:50],     # Limit scene memories
            linked_memories=linked_memories,
            emergent_patterns=patterns,
            scope=scope,
            last_changed_at=time.time()
        )
    
        # Trim to token budget and finalize token size
        bundle = self._trim_to_token_budget(bundle, token_budget)
        bundle.bundle_size_tokens = self._estimate_bundle_tokens(bundle)
    
        return bundle
    
    async def _get_player_memories_for_scope(self, scope: SceneScope) -> List[Dict[str, Any]]:
        """Get player memories relevant to the scene scope."""
        filters = {}
        
        # Add time window filter
        if scope.time_window_hours:
            cutoff = datetime.now() - timedelta(hours=scope.time_window_hours)
            filters['created_after'] = cutoff.isoformat()
        
        # Add location filter if specified
        if scope.location_id:
            filters['location'] = scope.location_id
        
        # Get memories
        memories = await self.retrieve_memories(
            entity_type=EntityType.PLAYER.value,
            entity_id=self.user_id,
            filters=filters,
            limit=30
        )
        
        return memories.get('memories', [])
    
    async def _get_npc_memories_for_scope(self, npc_id: int, scope: SceneScope) -> List[Dict[str, Any]]:
        """Get NPC memories relevant to the scene scope."""
        filters = {}
        
        # Add time window filter
        if scope.time_window_hours:
            cutoff = datetime.now() - timedelta(hours=scope.time_window_hours)
            filters['created_after'] = cutoff.isoformat()
        
        # Get memories
        memories = await self.retrieve_memories(
            entity_type=EntityType.NPC.value,
            entity_id=npc_id,
            filters=filters,
            limit=20
        )
        
        return memories.get('memories', [])
    
    async def _get_location_memories_for_scope(self, scope: SceneScope) -> List[Dict[str, Any]]:
        """Get location-specific memories."""
        res = await self.search_vector_store(
            query=f"location:{scope.location_id}",
            filter_dict={'location': scope.location_id},
            top_k=15
        )
        return self._as_mem_list(res)
    
    async def _get_topic_memories_for_scope(self, scope: SceneScope) -> List[Dict[str, Any]]:
        """Get memories related to specified topics."""
        all_memories = []
        for topic in scope.topics:
            res = await self.search_vector_store(
                query=topic, filter_dict={'tags': topic}, top_k=10
            )
            all_memories.extend(self._as_mem_list(res))
    
        # Deduplicate by memory_id
        seen = set()
        unique = []
        for mem in all_memories:
            mid = mem.get('memory_id') or mem.get('id')
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(mem)
        return unique
        
    async def _get_lore_memories_for_scope(self, scope: SceneScope) -> List[Dict[str, Any]]:
        """Get memories tagged with lore elements."""
        all_memories = []
        for lore_tag in scope.lore_tags:
            res = await self.search_vector_store(
                query=lore_tag, filter_dict={'lore_tags': lore_tag}, top_k=10
            )
            all_memories.extend(self._as_mem_list(res))
    
        # Deduplicate
        seen = set()
        unique = []
        for mem in all_memories:
            mid = mem.get('memory_id') or mem.get('id')
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(mem)
        return unique
    
    async def _find_linked_memories(self, memories: List[Dict], scope: SceneScope, limit: int = 10) -> List[Dict]:
        """Find memories linked to the current set through relationships."""
        if not memories:
            return []
        
        # Extract entities mentioned in memories
        mentioned_entities = set()
        valid_entity_types = {t.value for t in EntityType}
        
        for mem in memories:
            # Look for entity references in metadata
            metadata = mem.get('metadata', {})
            if 'entities' in metadata:
                for entity in metadata['entities']:
                    # Normalize entity type
                    etype = str(entity.get('type', '')).lower()
                    
                    # Robust ID parsing
                    raw = entity.get('id')
                    if raw is not None:
                        try:
                            # Try to parse as integer
                            eid = int(str(raw))
                        except (TypeError, ValueError):
                            # Keep as-is for UUIDs, strings, etc.
                            eid = raw
                        
                        # Only add valid entity types
                        if etype in valid_entity_types:
                            mentioned_entities.add((etype, eid))
        
        # Find memories involving these entities
        linked = []
        for entity_type, entity_id in list(mentioned_entities)[:5]:  # Limit lookups
            entity_memories = await self.retrieve_memories(
                entity_type=entity_type,
                entity_id=entity_id,
                limit=3
            )
            linked.extend(entity_memories.get('memories', []))
        
        return linked[:limit]
    
    async def _find_emergent_patterns(self, memories: List[Dict], scope: SceneScope, limit: int = 5) -> List[Dict]:
        """Find emergent patterns in the memory set."""
        if not memories or len(memories) < 3:
            return []
        
        # Quick pattern analysis
        patterns = []
        
        # Find recurring themes
        theme_counts = defaultdict(int)
        emotion_counts = defaultdict(int)
        
        for mem in memories:
            # Count themes
            for tag in mem.get('tags', []):
                theme_counts[tag] += 1
            
            # Count emotions
            emotional_state = mem.get('emotional_state', {})
            for emotion, intensity in emotional_state.items():
                if intensity > 0.5:
                    emotion_counts[emotion] += 1
        
        # Build pattern objects
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            if count >= 2:
                patterns.append({
                    'type': 'recurring_theme',
                    'theme': theme,
                    'occurrences': count,
                    'strength': min(1.0, count / len(memories))
                })
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:2]:
            if count >= 2:
                patterns.append({
                    'type': 'emotional_pattern',
                    'emotion': emotion,
                    'occurrences': count,
                    'strength': min(1.0, count / len(memories))
                })
        
        return patterns[:limit]
    
    def _estimate_bundle_tokens(self, bundle: MemoryBundle) -> int:
        """Estimate token count using a lenient serializer."""
        def _dump(x):  # lenient JSON (datetimes, enums, etc.)
            return json.dumps(x, ensure_ascii=False, default=str)
        
        total_chars = 0
        for section in (bundle.canon_memories, bundle.scene_memories, 
                       bundle.linked_memories, bundle.emergent_patterns):
            for item in section:
                try:
                    total_chars += len(_dump(item))
                except Exception:
                    # Fallback for truly problematic objects
                    total_chars += len(str(item))
        return total_chars // 4
    
    def _trim_to_token_budget(self, bundle: MemoryBundle, token_budget: int = 5000) -> MemoryBundle:
        """
        Trim bundle to fit within token budget.
        Priority: canon > patterns > scene > linked
        """
        # Start with everything
        result = MemoryBundle(
            canon_memories=bundle.canon_memories[:],
            scene_memories=bundle.scene_memories[:],
            linked_memories=bundle.linked_memories[:],
            emergent_patterns=bundle.emergent_patterns[:],
            scope=bundle.scope,
            last_changed_at=bundle.last_changed_at
        )
        
        # Check if we're already under budget
        current_tokens = self._estimate_bundle_tokens(result)
        if current_tokens <= token_budget:
            return result
        
        # Progressive trimming (least important first)
        # 1. Drop linked memories first
        if result.linked_memories and current_tokens > token_budget:
            result.linked_memories = []
            current_tokens = self._estimate_bundle_tokens(result)
        
        # 2. Reduce scene memories
        if current_tokens > token_budget:
            # Keep reducing scene memories by half until under budget
            while len(result.scene_memories) > 5 and current_tokens > token_budget:
                result.scene_memories = result.scene_memories[:len(result.scene_memories)//2]
                current_tokens = self._estimate_bundle_tokens(result)
        
        # 3. Reduce patterns
        if current_tokens > token_budget and len(result.emergent_patterns) > 2:
            result.emergent_patterns = result.emergent_patterns[:2]
            current_tokens = self._estimate_bundle_tokens(result)
        
        # 4. As last resort, trim canon (but keep at least 5)
        if current_tokens > token_budget and len(result.canon_memories) > 5:
            result.canon_memories = result.canon_memories[:5]
        
        result.bundle_size_tokens = self._estimate_bundle_tokens(result)
        return result
    
    def _cleanup_bundle_cache(self):
        """Remove expired entries from class-wide bundle cache and enforce size limit."""
        cls = self.__class__
        now = time.monotonic()
        
        # First pass: remove expired entries (based on creation time)
        expired = [k for k in list(cls._bundle_cache.keys())
                   if now - cls._bundle_created_at.get(k, 0.0) > cls._bundle_ttl]
        
        for k in expired:
            entry = cls._bundle_cache.get(k) or {}
            # Use the stored entity set for proper cleanup
            ents = entry.get('ents', set())
            for ent in ents:
                if ent in cls._bundle_index:
                    cls._bundle_index[ent].discard(k)
                    if not cls._bundle_index[ent]:
                        del cls._bundle_index[ent]
            cls._bundle_cache.pop(k, None)
            cls._bundle_cached_at.pop(k, None)
            cls._bundle_created_at.pop(k, None)
            cls._bundle_locks.pop(k, None)
        
        # Second pass: enforce size limit (true LRU eviction based on last access)
        if len(cls._bundle_cache) > cls._bundle_max_size:
            # Sort by last access time (oldest first)
            sorted_keys = sorted(
                cls._bundle_cache.keys(),
                key=lambda k: cls._bundle_cached_at.get(k, 0.0)
            )
            # Remove oldest entries until under limit with bounded hysteresis
            over = len(cls._bundle_cache) - cls._bundle_max_size
            num_to_remove = min(over + 50, len(sorted_keys))  # Bounded hysteresis
            to_remove = sorted_keys[:num_to_remove]
            
            for k in to_remove:
                entry = cls._bundle_cache.get(k) or {}
                ents = entry.get('ents', set())
                for ent in ents:
                    if ent in cls._bundle_index:
                        cls._bundle_index[ent].discard(k)
                        if not cls._bundle_index[ent]:
                            del cls._bundle_index[ent]
                cls._bundle_cache.pop(k, None)
                cls._bundle_cached_at.pop(k, None)
                cls._bundle_created_at.pop(k, None)
                cls._bundle_locks.pop(k, None)
            
            if len(to_remove) > 0:
                logger.info(f"True LRU evicted {len(to_remove)} bundle cache entries")
    
    # ========================================================================
    # Core Memory Operations (Existing)
    # ========================================================================
    
    async def store_memory(
        self,
        entity_type: str,
        entity_id: int,
        content: str,
        memory_type: Optional[str] = None,
        significance: float = 0.5,
        emotional_intensity: float = 0.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_canon: bool = False
    ) -> Dict[str, Any]:
        """
        Store a new memory.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            content: Memory content
            memory_type: Type of memory
            significance: Importance (0-1)
            emotional_intensity: Emotional weight (0-1)
            tags: Memory tags
            metadata: Additional metadata
            is_canon: Whether this is canonical information
            
        Returns:
            Dict with memory_id and success status
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get or create entity manager
            manager = await self._get_entity_manager(entity_type, entity_id)
            
            # Ensure metadata exists and apply canon flags before persistence
            md = dict(metadata or {})
            if is_canon:
                md['is_canon'] = True
                md['canonical'] = True
            
            # Store memory
            memory = await manager.store_memory(
                content=content,
                memory_type=memory_type,
                significance=significance,
                emotional_intensity=emotional_intensity,
                tags=tags,
                metadata=md
            )
            
            # Invalidate affected bundles via both entity and metadata
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            self._invalidate_by_metadata(md, tags)
            
            return {
                "memory_id": getattr(memory, "memory_id", None) or getattr(memory, "id", None),
                "success": True,
                "entity_type": entity_type,
                "entity_id": entity_id
            }
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {"error": str(e), "success": False}
    
    async def retrieve_memories(
        self,
        entity_type: str,
        entity_id: int,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve memories for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            query: Search query
            memory_type: Filter by type
            tags: Filter by tags
            limit: Maximum results
            filters: Additional filters
            include_analysis: Include memory analysis
            
        Returns:
            Dict with memories and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Build a stable cache key with user/conversation in prefix
            prefix = f"mem:{self.user_id}:{self.conversation_id}:"
            key_payload = {
                "et": entity_type,
                "eid": entity_id,
                "q": query,
                "mt": memory_type,
                "tags": sorted(tags or []),
                "limit": limit,
                "filters": filters or {},
                "ia": bool(include_analysis),  # Include analysis flag
            }
            cache_key = prefix + hashlib.md5(
                json.dumps(key_payload, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            cached = await self.cache.get(cache_key)
            if cached:
                logger.debug(f"Retrieval cache hit for entity {entity_type}:{entity_id}")
                return cached
            
            logger.debug(f"Retrieval cache miss for entity {entity_type}:{entity_id}")
            
            # Get entity manager
            manager = await self._get_entity_manager(entity_type, entity_id)
            
            # Retrieve memories
            if query:
                memories = await self.search_vector_store(
                    query=query,
                    entity_type=entity_type,
                    top_k=limit,
                    filter_dict=filters
                )
                mem_list = self._as_mem_list(memories)
            else:
                # Prefer a query-capable API if available
                mem_list = None
                if hasattr(manager, "query_memories"):
                    res = await manager.query_memories(
                        filters=filters or {},
                        limit=limit,
                        memory_type=memory_type,
                        tags=tags
                    )
                    mem_list = res if isinstance(res, list) else res.get('memories', [])
                else:
                    res = await manager.get_recent_memories(limit=limit)
                    mem_list = res if isinstance(res, list) else res.get('memories', [])
            
            # Format response
            result = {
                "memories": mem_list or [],
                "count": len(mem_list or []),
                "entity_type": entity_type,
                "entity_id": entity_id
            }
            
            # Add analysis if requested
            if include_analysis and result.get("memories"):
                result["analysis"] = await self.analyze_memory_set(
                    memories=result["memories"],
                    entity_type=entity_type,
                    entity_id=entity_id
                )
            
            # Cache result
            await self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {"error": str(e), "memories": [], "count": 0}
    
    async def update_memory(
        self,
        entity_type: str,
        entity_id: int,
        memory_id: int,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        if not self.initialized:
            await self.initialize()
        
        try:
            manager = await self._get_entity_manager(entity_type, entity_id)
            updated = await manager.update_memory(memory_id, updates)
            
            # Invalidate caches for both entity and metadata
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            self._invalidate_by_metadata(updates.get('metadata'), updates.get('tags'))
            
            return {"success": True, "memory_id": memory_id}
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return {"error": str(e), "success": False}
    
    async def delete_memory(
        self,
        entity_type: str,
        entity_id: int,
        memory_id: int
    ) -> Dict[str, Any]:
        """Delete a memory."""
        if not self.initialized:
            await self.initialize()
        
        try:
            manager = await self._get_entity_manager(entity_type, entity_id)
            
            # Try to get the memory metadata before deletion for cache invalidation
            prev = None
            if hasattr(manager, "get_memory"):
                try:
                    prev = await manager.get_memory(memory_id)
                except Exception:
                    prev = None
            
            # Delete the memory
            await manager.delete_memory(memory_id)
            
            # Invalidate caches
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            if prev:
                self._invalidate_by_metadata(prev.get('metadata'), prev.get('tags'))
            
            return {"success": True, "memory_id": memory_id}
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"error": str(e), "success": False}
    
    # ========================================================================
    # Entity Manager Operations
    # ========================================================================
    
    async def _get_entity_manager(self, entity_type: str, entity_id: int):
        key = f"{entity_type}_{entity_id}"
        if key not in self.memory_managers:
            if entity_type == EntityType.NPC.value:
                from memory.npc_memory import NPCMemoryManager
                manager = NPCMemoryManager(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    npc_id=entity_id
                )
                await manager.initialize()
            elif entity_type == EntityType.PLAYER.value:
                from memory.player_memory import PlayerMemoryManager
                manager = PlayerMemoryManager(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
                await manager.initialize()
            else:
                manager = self.unified_manager  # already initialized in initialize()
            self.memory_managers[key] = manager
        return self.memory_managers[key]
    
    async def _invalidate_caches_for_entity(self, entity_type: str, entity_id: int):
        """
        Invalidate bundle + retrieval caches related to a specific entity.
        """
        cls = self.__class__
    
        # 1) Evict scene bundles via reverse index (class-wide)
        ent_key = f"{entity_type.lower()}:{entity_id}"
        affected = list(cls._bundle_index.get(ent_key, set()))
        for cache_key in affected:
            cls._evict_bundle_key(cache_key)
        cls._bundle_index.pop(ent_key, None)
    
        # 2) Evict retrieval caches (narrowed to this user/conversation)
        prefix = f"mem:{self.user_id}:{self.conversation_id}:"
        await self.cache.clear_pattern(prefix + "*")
    
    def _invalidate_by_metadata(self, metadata: Optional[Dict[str, Any]], tags: Optional[List[str]] = None):
        """
        Invalidate caches based on metadata fields (location, topics, lore tags).
        """
        cls = self.__class__
        if not metadata and not tags:
            return
    
        ent_keys = set()
    
        # Extract entity keys from metadata
        if metadata:
            if metadata.get('location'):
                ent_keys.add(f"location:{metadata['location']}")
            for lt in (metadata.get('lore_tags') or []):
                ent_keys.add(f"lore:{lt}")
    
        # Extract from tags
        for t in (tags or []):
            ent_keys.add(f"topic:{t}")
    
        # Invalidate each affected entity key
        for ent in ent_keys:
            affected = list(cls._bundle_index.get(ent, set()))
            for cache_key in affected:
                cls._evict_bundle_key(cache_key)
            cls._bundle_index.pop(ent, None)
    
    # ========================================================================
    # Analysis Operations
    # ========================================================================
    
    async def analyze_memory_set(
        self,
        memories: List[Any],
        entity_type: Optional[str] = None,
        entity_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze a set of memories for patterns and insights."""
        if not memories:
            return {"message": "No memories to analyze"}
        
        # Handle both Memory objects and dicts
        analysis = {
            "total_count": len(memories),
            "average_significance": 0,
            "average_emotional_intensity": 0,
            "common_tags": {},
            "memory_types": {},
            "dominant_emotions": []
        }
        
        emotion_totals = {}
        
        for memory in memories:
            # Handle dict format
            if isinstance(memory, dict):
                significance = memory.get("significance", 0)
                emotional_intensity = memory.get("emotional_intensity", 0)
                tags = memory.get("tags", [])
                memory_type = memory.get("type", "unknown")
                emotional_state = memory.get("emotional_state", {})
            else:
                # Handle Memory object
                significance = getattr(memory, "significance", 0)
                emotional_intensity = getattr(memory, "emotional_intensity", 0)
                tags = getattr(memory, "tags", [])
                memory_type = getattr(memory, "memory_type", "unknown")
                emotional_state = getattr(memory, "emotional_state", {})
                if hasattr(memory_type, 'value'):
                    memory_type = memory_type.value
            
            analysis["average_significance"] += significance
            analysis["average_emotional_intensity"] += emotional_intensity
            
            # Count tags
            for tag in tags:
                analysis["common_tags"][tag] = analysis["common_tags"].get(tag, 0) + 1
            
            # Count memory types
            analysis["memory_types"][str(memory_type)] = analysis["memory_types"].get(str(memory_type), 0) + 1
            
            # Track emotions
            for emotion, intensity in emotional_state.items():
                if emotion not in emotion_totals:
                    emotion_totals[emotion] = 0
                emotion_totals[emotion] += intensity
        
        if len(memories) > 0:
            analysis["average_significance"] /= len(memories)
            analysis["average_emotional_intensity"] /= len(memories)
            
            # Get top 3 dominant emotions
            if emotion_totals:
                sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
                analysis["dominant_emotions"] = [
                    {"emotion": em, "total_intensity": intensity} 
                    for em, intensity in sorted_emotions[:3]
                ]
        
        return analysis
    
    # ========================================================================
    # Context Enrichment Operations
    # ========================================================================
    
    async def enrich_context(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich context with relevant memories.
        
        This is kept for backward compatibility but internally uses scene bundles.
        """
        # Create scope from context
        scope = SceneScope(
            location_id=context.get('current_location'),
            npc_ids=set(context.get('active_npcs', [])),
            topics=set(context.get('topics', [])),
            lore_tags=set(context.get('lore_tags', []))
        )
        
        # Get scene bundle
        bundle = await self.get_scene_bundle(scope)
        
        # Merge bundle into context
        context['memories'] = {
            'canon': bundle.get('canon', []),
            'scene': bundle.get('scene', []),
            'linked': bundle.get('linked', []),
            'patterns': bundle.get('patterns', [])
        }
        
        return context
    
    # ========================================================================
    # Vector Store Operations
    # ========================================================================
    
    async def add_to_vector_store(
        self,
        text: str,
        metadata: Dict[str, Any],
        entity_type: str = "memory"
    ) -> str:
        """Add content directly to the vector store."""
        if not self.initialized:
            await self.initialize()
        
        return await self.embedding_service.add_memory(
            text=text,
            metadata=metadata,
            entity_type=entity_type
        )
    
    async def search_vector_store(
        self,
        query: str,
        entity_type: Optional[str] = None,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the vector store directly with concurrency limiting."""
        if not self.initialized:
            await self.initialize()
    
        async with self.__class__._sem():
            return await self.embedding_service.search_memories(
                query_text=query,
                entity_type=entity_type,
                top_k=top_k,
                filter_dict=filter_dict,
                fetch_content=True
            )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for text."""
        if not self.initialized:
            await self.initialize()
        
        return await self.embedding_service.generate_embedding(text)
    
    # ========================================================================
    # Agent-Based Operations
    # ========================================================================
    
    async def agent_remember(
        self,
        entity_type: str,
        entity_id: int,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Store a memory using the memory agent wrapper."""
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_agent_wrapper.remember(
            run_context=None,
            entity_type=entity_type,
            entity_id=entity_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags
        )
    
    async def agent_recall(
        self,
        entity_type: str,
        entity_id: int,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Recall memories using the memory agent wrapper."""
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_agent_wrapper.recall(
            run_context=None,
            entity_type=entity_type,
            entity_id=entity_id,
            query=query,
            context=context,
            limit=limit
        )
    
    # ========================================================================
    # Maintenance Operations (Move to Background)
    # ========================================================================
    
    async def run_maintenance(self, background: bool = True) -> Dict[str, Any]:
        """
        Run memory maintenance tasks.
        
        Args:
            background: If True, enqueue to background worker
            
        Returns:
            Status of maintenance tasks
        """
        if background:
            # Enqueue to background worker
            from celery import current_app
            task = current_app.send_task(
                'memory.tasks.run_memory_maintenance',
                args=[self.user_id, self.conversation_id]
            )
            return {"status": "queued", "task_id": task.id}
        
        # Run inline (for backward compatibility)
        results = {}
        
        # Only run if enough time has passed
        if (datetime.now() - self.last_maintenance).seconds < 300:  # 5 minutes
            return {"status": "skipped", "reason": "Too soon since last maintenance"}
        
        # Run consolidation
        results['consolidation'] = await self._run_consolidation()
        
        # Run decay
        results['decay'] = await self._run_decay()
        
        # Run pattern analysis
        results['patterns'] = await self._run_pattern_analysis()
        
        self.last_maintenance = datetime.now()
        
        return results
    
    async def _run_consolidation(self) -> Dict[str, Any]:
        """Run memory consolidation for all entities."""
        results = {}
        entities = await self._get_active_entities()
        
        for entity_type, entity_id in entities:
            manager = await self._get_entity_manager(entity_type, entity_id)
            if hasattr(manager, 'consolidate_memories'):
                consolidated = await manager.consolidate_memories()
                results[f"{entity_type}_{entity_id}"] = len(consolidated)
        
        return results
    
    async def _run_decay(self) -> Dict[str, Any]:
        """Run memory decay for all entities."""
        results = {}
        entities = await self._get_active_entities()
        
        for entity_type, entity_id in entities:
            manager = await self._get_entity_manager(entity_type, entity_id)
            if hasattr(manager, 'apply_memory_decay'):
                decayed = await manager.apply_memory_decay()
                results[f"{entity_type}_{entity_id}"] = decayed
        
        return results
    
    async def _run_pattern_analysis(self) -> Dict[str, Any]:
        """Run pattern analysis across all memories."""
        # This is expensive - definitely move to background
        patterns = await self.analyze_cross_entity_patterns()
        return {"patterns_found": len(patterns.get('patterns', []))}
    
    async def _get_active_entities(self) -> List[Tuple[str, int]]:
        """Get list of active entities with memories."""
        entities = [(EntityType.PLAYER.value, self.user_id)]
        
        # Add active NPCs from context
        # This would be populated from the NyxContext
        # For now, return just the player
        
        return entities
    
    # ========================================================================
    # Cross-Entity Pattern Analysis
    # ========================================================================
    
    async def analyze_cross_entity_patterns(
        self,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze patterns across multiple entities."""
        try:
            entities = await self._get_active_entities()
            
            # Collect memories from all entities
            all_memories = []
            for entity_type, entity_id in entities:
                memories = await self.retrieve_memories(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    limit=20
                )
                all_memories.extend(memories.get('memories', []))
            
            # Find patterns
            patterns = await self._find_emergent_patterns(all_memories, SceneScope(), limit=10)
            
            return {
                "patterns": patterns,
                "entity_count": len(entities),
                "memory_count": len(all_memories),
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-entity patterns: {e}")
            return {"error": str(e), "patterns": []}
    
    # ========================================================================
    # Specialized Memory Operations
    # ========================================================================
    
    async def create_flashback(
        self,
        entity_type: str,
        entity_id: int,
        trigger: str
    ) -> Dict[str, Any]:
        """Create a flashback memory sequence."""
        if not self.initialized:
            await self.initialize()
        
        return await self.flashback_manager.create_flashback(
            entity_type=entity_type,
            entity_id=entity_id,
            trigger=trigger
        )
    
    async def apply_mask(
        self,
        entity_type: str,
        entity_id: int,
        memory_id: int,
        mask_type: str = "partial"
    ) -> Dict[str, Any]:
        """Apply a mask to a memory."""
        if not self.initialized:
            await self.initialize()
        
        return await self.mask_manager.apply_mask(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_id=memory_id,
            mask_type=mask_type
        )
    
    async def create_false_memory(
        self,
        entity_type: str,
        entity_id: int,
        false_text: str,
        base_memory_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Create a false memory (for story purposes)."""
        if not self.initialized:
            await self.initialize()
        
        return await self.interference_manager.create_false_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            false_memory_text=false_text,
            related_true_memory_ids=base_memory_ids
        )
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    async def close(self):
        """Clean up resources."""
        if self.embedding_service:
            await self.embedding_service.close()
        if self.retriever_agent:
            await self.retriever_agent.close()
        
        # Clear per-conversation retrieval cache
        await self.cache.clear()
        
        # Remove only this conversation's bundles from class-wide cache
        cls = self.__class__
        prefix = f"{self.user_id}:{self.conversation_id}:"
        to_delete = [k for k in list(cls._bundle_cache.keys()) if k.startswith(prefix)]
        for k in to_delete:
            cls._evict_bundle_key(k)
        
        # Remove this instance from the singleton registry
        try:
            del cls._instances[(self.user_id, self.conversation_id)]
        except KeyError:
            pass
        
        self.initialized = False


# ============================================================================
# Convenience Functions
# ============================================================================

async def get_memory_orchestrator(user_id: int, conversation_id: int) -> MemoryOrchestrator:
    """
    Get or create a Memory Orchestrator instance.
    
    This is the primary entry point for all memory operations in the system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        MemoryOrchestrator instance
    """
    return await MemoryOrchestrator.get_instance(user_id, conversation_id)
