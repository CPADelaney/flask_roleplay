# memory/memory_orchestrator.py
"""
Memory Orchestrator with Scene Bundle Optimization
Complete refactor for optimized context assembly pipeline
"""
from __future__ import annotations

import logging
import asyncio
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Iterable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# New/optional integrations (safe fallbacks)
try:
    from memory.masks import ProgressiveRevealManager
except Exception:
    ProgressiveRevealManager = None

try:
    from memory.memory_agent_sdk import create_memory_agent, MemorySystemContext
    from memory.memory_agent_wrapper import MemoryAgentWrapper as V2MemoryAgentWrapper
except Exception:
    create_memory_agent = None
    MemorySystemContext = None
    V2MemoryAgentWrapper = None

try:
    from memory.memory_nyx_integration import get_memory_nyx_bridge
except Exception:
    get_memory_nyx_bridge = None

try:
    from memory.memory_config import get_memory_config
except Exception:
    get_memory_config = None

# Optional/new components with safe fallbacks
try:
    from memory.memory_service import MemoryEmbeddingService as LC_MemoryEmbeddingService
except Exception:
    LC_MemoryEmbeddingService = None

try:
    from memory.memory_retriever import MemoryRetrieverAgent as LC_MemoryRetrieverAgent
except Exception:
    LC_MemoryRetrieverAgent = None

try:
    from memory.reconsolidation import ReconsolidationManager
except Exception:
    ReconsolidationManager = None

try:
    from memory.schemas import MemorySchemaManager as NewMemorySchemaManager
except Exception:
    NewMemorySchemaManager = None

try:
    from memory.semantic import SemanticMemoryManager as NewSemanticMemoryManager
except Exception:
    NewSemanticMemoryManager = None

try:
    from memory.telemetry import MemoryTelemetry
except Exception:
    MemoryTelemetry = None

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

MEMORY_CACHE_TTL = 300  # 5 minutes default
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
        self.integrated_system = None

        # New managers/integrations
        self.reconsolidation_manager = None

        # New integrations
        self.progressive_reveal_manager = None  # masks.py
        self.nyx_bridge = None                  # governance bridge
        self.memory_config = None               # memory_config.py
        
        # State
        self.initialized = False
        self.last_sync = datetime.now()
        self.last_maintenance = datetime.now()
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    async def initialize(self):
        """Initialize all memory subsystems (prefers NEW stack, falls back to LEGACY)."""
        if self.initialized:
            return

        try:
            # --- Core / cache (LEGACY; keep) ---
            UnifiedMemoryManager, _, _, _, _, _ = _lazy_import_memory_core()

            # Retrieval/cache for per-entity calls
            from memory.core import MemoryCache
            self.cache = MemoryCache(ttl_seconds=MEMORY_CACHE_TTL)

            # --- Configuration (NEW if available) ---
            if self.memory_config is None:
                try:
                    if get_memory_config:
                        self.memory_config = get_memory_config()
                    else:
                        from memory.memory_config import get_memory_config as _get_cfg
                        self.memory_config = _get_cfg()
                except Exception:
                    self.memory_config = None

            # --- Embedding service (NEW preferred: memory.memory_service) ---
            # Fallback: legacy memory.embedding_service
            try:
                if LC_MemoryEmbeddingService:
                    vs_type = (self.memory_config or {}).get("vector_store", {}).get("type", "chroma")
                    emb_type = (self.memory_config or {}).get("embedding", {}).get("type", "local")
                    self.embedding_service = LC_MemoryEmbeddingService(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id,
                        vector_store_type=vs_type,
                        embedding_model=emb_type,
                        config=(self.memory_config or {})
                    )
                    await self.embedding_service.initialize()
                else:
                    from memory.embedding_service import MemoryEmbeddingService as OldEmbeddingService  # LEGACY
                    self.embedding_service = OldEmbeddingService(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
            except Exception as e:
                logger.error(f"Failed to initialize embedding service: {e}")
                raise

            # --- Agent wrapper (NEW preferred: SDK + v2.1 wrapper) ---
            # Fallback: legacy memory.memory_agent.MemoryAgentWrapper
            try:
                if create_memory_agent and V2MemoryAgentWrapper and MemorySystemContext:
                    base_agent = create_memory_agent(self.user_id, self.conversation_id)
                    agent_context = MemorySystemContext(self.user_id, self.conversation_id)
                    self.memory_agent_wrapper = V2MemoryAgentWrapper(base_agent, agent_context)
                else:
                    from memory.memory_agent import MemoryAgentWrapper as LegacyAgentWrapper  # LEGACY
                    self.memory_agent_wrapper = LegacyAgentWrapper(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
            except Exception as e:
                logger.warning(f"Agent wrapper init failed: {e}")
                self.memory_agent_wrapper = None

            # --- Retriever (NEW preferred: LangChain MemoryRetrieverAgent) ---
            # Fallback: legacy memory.retriever_agent.RetrieverAgent
            try:
                if LC_MemoryRetrieverAgent:
                    llm_type = (self.memory_config or {}).get("llm", {}).get("type", "openai")
                    # Adapt config keys expected by the retriever
                    retriever_cfg = dict(self.memory_config or {})
                    llm_cfg = (self.memory_config or {}).get("llm", {})
                    retriever_cfg["openai_model_name"] = llm_cfg.get("openai_model", "gpt-5-nano")
                    retriever_cfg["hf_model_name"] = llm_cfg.get("huggingface_model", "mistralai/Mistral-7B-Instruct-v0.2")
                    retriever_cfg["temperature"] = llm_cfg.get("temperature", 0.0)
                    self.retriever_agent = LC_MemoryRetrieverAgent(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id,
                        llm_type=llm_type,
                        memory_service=self.embedding_service,
                        config=retriever_cfg
                    )
                    await self.retriever_agent.initialize()
                else:
                    from memory.retriever_agent import RetrieverAgent as LegacyRetriever  # LEGACY
                    self.retriever_agent = LegacyRetriever(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
            except Exception as e:
                logger.warning(f"Retriever init failed: {e}")
                self.retriever_agent = None

            # --- Specialized managers (mixed NEW/LEGACY) ---

            # Nyx memory (LEGACY manager; keep as-is)
            from memory.memory_nyx_integration import NyxMemoryManager
            self.nyx_memory_manager = NyxMemoryManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                unified_manager=self.unified_manager  # may be None; matches existing signature
            )

            # Emotions (LEGACY manager; keep as-is)
            from memory.emotional import EmotionalMemoryManager
            self.emotional_manager = EmotionalMemoryManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )

            # Schema manager (NEW preferred)
            try:
                if NewMemorySchemaManager:
                    self.schema_manager = NewMemorySchemaManager(self.user_id, self.conversation_id)  # NEW
                else:
                    raise ImportError("New schema manager not available")
            except Exception:
                from memory.schema_manager import SchemaManager  # LEGACY
                self.schema_manager = SchemaManager(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )

            # Semantic manager (NEW preferred)
            try:
                if NewSemanticMemoryManager:
                    self.semantic_manager = NewSemanticMemoryManager(self.user_id, self.conversation_id)  # NEW
                else:
                    raise ImportError("New semantic manager not available")
            except Exception:
                from memory.semantic_manager import SemanticManager  # LEGACY
                self.semantic_manager = SemanticManager(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )

            # Flashbacks (LEGACY)
            from memory.flashbacks import FlashbackManager
            self.flashback_manager = FlashbackManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )

            # Mask manager (LEGACY; basic masking)
            from memory.mask import MaskManager
            self.mask_manager = MaskManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )

            # Progressive Reveal (NEW masks system; additive)
            try:
                if ProgressiveRevealManager:
                    self.progressive_reveal_manager = ProgressiveRevealManager(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
            except Exception:
                self.progressive_reveal_manager = None

            # Interference (LEGACY)
            from memory.interference import MemoryInterferenceManager
            self.interference_manager = MemoryInterferenceManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )

            # Reconsolidation (NEW; optional)
            try:
                if ReconsolidationManager:
                    self.reconsolidation_manager = ReconsolidationManager(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
            except Exception:
                self.reconsolidation_manager = None

            # Integrated high-level system (LEGACY/primary orchestrator integration)
            from memory.integrated import IntegratedMemorySystem
            self.integrated_system = IntegratedMemorySystem(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )

            # Nyx governance bridge (NEW; optional)
            try:
                if get_memory_nyx_bridge:
                    self.nyx_bridge = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
            except Exception:
                self.nyx_bridge = None

            self.initialized = True
            self.last_sync = datetime.now()
            self.last_maintenance = datetime.now()
            logger.info(f"Memory orchestrator initialized for user {self.user_id}, conversation {self.conversation_id}")

        except Exception as e:
            logger.error(f"Failed to initialize memory orchestrator: {e}")
            raise
    
    # ========================================================================
    # Scene Bundle Operations (NEW - Performance Optimization)
    # ========================================================================

    async def reconsolidate_memory(self, entity_type: str, entity_id: int, memory_id: int,
                                   emotional_context: Optional[Dict[str, Any]] = None,
                                   recall_context: Optional[str] = None,
                                   alteration_strength: float = 0.1) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        if not self.reconsolidation_manager:
            return {"error": "Reconsolidation not available"}
        t0 = time.time()
        try:
            res = await self.reconsolidation_manager.reconsolidate_memory(
                memory_id=memory_id,
                entity_type=entity_type,
                entity_id=entity_id,
                emotional_context=emotional_context,
                recall_context=recall_context,
                alteration_strength=alteration_strength
            )
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            await self._telemetry("reconsolidate_memory", started_at=t0, success=("error" not in res),
                                  metadata={"et": entity_type, "eid": entity_id})
            return res
        except Exception as e:
            await self._telemetry("reconsolidate_memory", started_at=t0, success=False, error=str(e))
            return {"error": str(e)}

    async def run_reconsolidation_sweep(self, entity_type: str, entity_id: int, max_memories: int = 5) -> List[int]:
        if not self.initialized:
            await self.initialize()
        if not self.reconsolidation_manager:
            return []
        ids = await self.reconsolidation_manager.check_memories_for_reconsolidation(
            entity_type=entity_type, entity_id=entity_id, max_memories=max_memories
        )
        if ids:
            await self._invalidate_caches_for_entity(entity_type, entity_id)
        return ids
    
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
    
    def _coerce_scope(self, scope: Any) -> SceneScope:
        if isinstance(scope, SceneScope):
            return scope
        # graceful adapter for NyxContext.SceneScope or dicts
        def _get(obj, attr, default=None):
            return getattr(obj, attr, default) if not isinstance(obj, dict) else obj.get(attr, default)
        return SceneScope(
            location_id=_get(scope, "location_id") or _get(scope, "location"),
            npc_ids=set(_get(scope, "npc_ids", []) or []),
            topics=set(_get(scope, "topics", []) or []),
            lore_tags=set(_get(scope, "lore_tags", []) or []),
            time_window_hours=_get(scope, "time_window", _get(scope, "time_window_hours", 24)) or 24,
        )
    
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
    
    async def get_scene_bundle(self, scope: Any, token_budget: int = 5000) -> Dict[str, Any]:
        """
        Get a scene-scoped memory bundle for efficient context assembly (telemetry-instrumented).
        """
        t0 = time.time()
        out_payload: Dict[str, Any] = {}
        try:
            if not self.initialized:
                await self.initialize()
    
            scope = self._coerce_scope(scope)
    
            cls = self.__class__
            cache_key = f"{self.user_id}:{self.conversation_id}:{scope.to_cache_key()}"
    
            # Fast path with separate TTL and LRU tracking
            cached = cls._bundle_cache.get(cache_key)
            if cached:
                created_at = cls._bundle_created_at.get(cache_key, 0.0)
                if time.monotonic() - created_at < cls._bundle_ttl:
                    cls._bundle_cached_at[cache_key] = time.monotonic()  # Touch for LRU
                    logger.debug(f"Bundle cache hit for {cache_key[:50]}...")
                    out_payload = cached["bundle"]
                else:
                    logger.debug(f"Bundle cache expired for {cache_key[:50]}...")
    
            if not out_payload:
                logger.debug(f"Bundle cache miss for {cache_key[:50]}...")
    
                # Prevent duplicate builds across instances
                async with cls._get_lock(cache_key):
                    cached = cls._bundle_cache.get(cache_key)
                    if cached:
                        created_at = cls._bundle_created_at.get(cache_key, 0.0)
                        if time.monotonic() - created_at < cls._bundle_ttl:
                            cls._bundle_cached_at[cache_key] = time.monotonic()  # Touch for LRU
                            out_payload = cached["bundle"]
    
                    if not out_payload:
                        bundle = await self._build_scene_bundle(scope, token_budget)
                        payload = bundle.to_dict()
    
                        # Index and get the entity set
                        ents = self._index_bundle(cache_key, scope)
    
                        # Cache with entity set for proper cleanup (class-wide)
                        cls._bundle_cache[cache_key] = {
                            "bundle": payload,
                            "timestamp": time.time(),
                            "scope": scope,
                            "ents": ents,
                        }
                        self._cleanup_bundle_cache()
                        out_payload = payload
    
            # Telemetry (success)
            try:
                ds = (len(out_payload.get("scene", [])) + len(out_payload.get("canon", []))) if out_payload else 0
                await self._telemetry(
                    "get_scene_bundle",
                    started_at=t0,
                    success=True,
                    data_size=ds,
                    metadata={"loc": scope.location_id, "npcs": list(scope.npc_ids)}
                )
            except Exception:
                pass
    
            return out_payload
    
        except Exception as e:
            # Telemetry (failure)
            try:
                await self._telemetry(
                    "get_scene_bundle",
                    started_at=t0,
                    success=False,
                    error=str(e)
                )
            except Exception:
                pass
            raise
    
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
    
        # Emotion-aware augmentation (mood congruent recall)
        try:
            bundle = await self._augment_bundle_with_emotion(bundle)
        except Exception as e:
            logger.debug(f"Emotion augmentation skipped: {e}")
    
        # Mask integrity overlay for NPCs in scope (if available)
        if self.progressive_reveal_manager and scope.npc_ids:
            try:
                # Fetch masks in parallel
                mask_tasks = [self.get_npc_mask(npc_id) for npc_id in scope.npc_ids]
                mask_results = await asyncio.gather(*mask_tasks, return_exceptions=True)
                for mr in mask_results:
                    if isinstance(mr, Exception) or not isinstance(mr, dict):
                        continue
                    if mr.get("npc_id") and "integrity" in mr:
                        bundle.emergent_patterns.append({
                            "type": "mask_integrity",
                            "npc_id": mr["npc_id"],
                            "npc_name": mr.get("npc_name"),
                            "integrity": mr.get("integrity"),
                            "strength": max(0.0, min(1.0, (100 - float(mr.get("integrity", 0))) / 100.0)),
                            "traits_hint": list((mr.get("hidden_traits") or {}).keys())[:2],
                        })
            except Exception as e:
                logger.debug(f"Mask overlay skipped: {e}")
    
        # Trim to token budget and finalize token size
        bundle = self._trim_to_token_budget(bundle, token_budget)
        bundle.bundle_size_tokens = self._estimate_bundle_tokens(bundle)
    
        return bundle

    async def _telemetry(self, operation: str, *, success: bool = True, started_at: float | None = None,
                         data_size: int | None = None, error: str | None = None, metadata: Dict[str, Any] | None = None):
        if not MemoryTelemetry:
            return
        try:
            dur = 0.0
            if started_at is not None:
                dur = max(0.0, time.time() - started_at)
            await MemoryTelemetry.record(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                operation=operation,
                success=success,
                duration=dur,
                data_size=data_size,
                error=error,
                metadata=metadata or {}
            )
        except Exception:
            pass

    async def _augment_bundle_with_emotion(self, bundle: MemoryBundle) -> MemoryBundle:
        """Append mood-congruent memories for player and NPCs into scene_memories (deduped)."""
        scope = bundle.scope or SceneScope()
        seen_ids = set()
        def _add_unique(mem_dicts: List[Dict[str, Any]]):
            for m in mem_dicts:
                mid = m.get('id') or m.get('memory_id')
                if mid and mid in seen_ids:
                    continue
                if mid:
                    seen_ids.add(mid)
                bundle.scene_memories.append(m)
        # seed seen with existing
        for s in bundle.scene_memories:
            mid = s.get('id') or s.get('memory_id')
            if mid:
                seen_ids.add(mid)

        # Player
        try:
            state = await self.emotional_manager.get_entity_emotional_state(
                entity_type=EntityType.PLAYER.value, entity_id=self.user_id
            )
            mood = {
                "primary_emotion": (state.get("current_emotion") or {}).get("primary", "neutral"),
                "intensity": (state.get("current_emotion") or {}).get("intensity", 0.3),
                "valence": (state.get("current_emotion") or {}).get("valence", 0.0),
                "arousal": (state.get("current_emotion") or {}).get("arousal", 0.0),
            }
            congruent = await self.emotional_manager.retrieve_mood_congruent_memories(
                entity_type=EntityType.PLAYER.value, entity_id=self.user_id, current_mood=mood, limit=3
            )
            # adapt shape
            mems = [{"id": x["id"], "text": x["text"], "tags": ["mood_congruent","player"]} for x in (congruent or [])]
            _add_unique(mems)
        except Exception:
            pass

        # NPCs
        for npc_id in (scope.npc_ids or []):
            try:
                state = await self.emotional_manager.get_entity_emotional_state(
                    entity_type=EntityType.NPC.value, entity_id=npc_id
                )
                mood = {
                    "primary_emotion": (state.get("current_emotion") or {}).get("primary", "neutral"),
                    "intensity": (state.get("current_emotion") or {}).get("intensity", 0.3),
                    "valence": (state.get("current_emotion") or {}).get("valence", 0.0),
                    "arousal": (state.get("current_emotion") or {}).get("arousal", 0.0),
                }
                congruent = await self.emotional_manager.retrieve_mood_congruent_memories(
                    entity_type=EntityType.NPC.value, entity_id=npc_id, current_mood=mood, limit=2
                )
                mems = [{"id": x["id"], "text": x["text"], "tags": ["mood_congruent","npc"]} for x in (congruent or [])]
                _add_unique(mems)
            except Exception:
                continue
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
        """Get location-specific memories with a hard guard to avoid no-op vector calls."""
        # Guard: no location → no search (saves a semaphore slot & avoids empty queries)
        loc = getattr(scope, "location_id", None)
        if not loc:
            return []
    
        q = f"location:{loc}".strip()
        if not q or q == "location:":
            return []
    
        res = await self.search_vector_store(
            query=q,
            filter_dict={'location': loc},
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
            for tag in (mem.get('tags') or []):
                theme_counts[tag] += 1

            # Count emotions: support both flat 'emotional_state' and metadata.emotions
            emotional_state = mem.get('emotional_state') or {}
            if not emotional_state:
                md = mem.get('metadata') or {}
                emo = md.get('emotions') or {}
                # try primary first
                primary = (emo.get('primary') or {})
                pname = primary.get('name')
                pinten = float(primary.get('intensity', 0.0) or 0.0)
                if pname:
                    emotional_state[pname] = pinten
                # include secondaries
                for k, v in (emo.get('secondary') or {}).items():
                    try:
                        emotional_state[k] = max(emotional_state.get(k, 0.0), float(v or 0.0))
                    except Exception:
                        pass
            for emotion, intensity in emotional_state.items():
                try:
                    if float(intensity) > 0.5:
                        emotion_counts[emotion] += 1
                except Exception:
                    continue
        
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
        Trim bundle to fit within token budget deterministically.
        Priority: canon > patterns > scene > linked
        Selection within each group is deterministic (score-based), which improves cache reuse.
    
        Scoring rules:
          - Memories: sort by (significance desc, created_at desc)
          - Patterns: sort by (strength desc, occurrences desc)
        """
    
        def _parse_ts(v: Any) -> float:
            """Try to normalize a created_at/created/timestamp value into a float seconds epoch."""
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    # best effort; tolerate 'Z' and offsets
                    from datetime import datetime
                    v2 = v.replace('Z', '+00:00')  # naive ISO fix
                    return datetime.fromisoformat(v2).timestamp()
                except Exception:
                    return 0.0
            return 0.0
    
        def _mem_score(mem: Dict[str, Any]) -> Tuple[float, float]:
            sig = float(mem.get("significance", 0.0) or 0.0)
            ts = _parse_ts(mem.get("created_at") or mem.get("created") or mem.get("timestamp"))
            return (sig, ts)
    
        def _pat_score(p: Dict[str, Any]) -> Tuple[float, float]:
            return (float(p.get("strength", 0.0) or 0.0), float(p.get("occurrences", 0) or 0))
    
        # Start from copies so we never mutate the incoming bundle in-place
        result = MemoryBundle(
            canon_memories=list(bundle.canon_memories or []),
            scene_memories=list(bundle.scene_memories or []),
            linked_memories=list(bundle.linked_memories or []),
            emergent_patterns=list(bundle.emergent_patterns or []),
            scope=bundle.scope,
            last_changed_at=bundle.last_changed_at
        )
    
        # Deterministic ordering inside each section
        result.canon_memories.sort(key=_mem_score, reverse=True)
        result.scene_memories.sort(key=_mem_score, reverse=True)
        result.linked_memories.sort(key=_mem_score, reverse=True)
        result.emergent_patterns.sort(key=_pat_score, reverse=True)
    
        # Fast path
        current_tokens = self._estimate_bundle_tokens(result)
        if current_tokens <= token_budget:
            result.bundle_size_tokens = current_tokens
            return result
    
        # 1) Drop linked memories first (lowest priority)
        if result.linked_memories and current_tokens > token_budget:
            result.linked_memories = []
            current_tokens = self._estimate_bundle_tokens(result)
            if current_tokens <= token_budget:
                result.bundle_size_tokens = current_tokens
                return result
    
        # 2) Deterministically trim scene memories by taking top-K
        if result.scene_memories and current_tokens > token_budget:
            # Binary shrink to preserve the most relevant items deterministically
            lo, hi = 5, len(result.scene_memories)  # keep at least 5 if possible
            # If already under 5, hi==lo in which case we'll fall through
            best = result.scene_memories[:]
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = result.scene_memories[:mid]
                tmp = MemoryBundle(
                    canon_memories=result.canon_memories,
                    scene_memories=candidate,
                    linked_memories=result.linked_memories,
                    emergent_patterns=result.emergent_patterns,
                    scope=result.scope,
                    last_changed_at=result.last_changed_at
                )
                tok = self._estimate_bundle_tokens(tmp)
                if tok <= token_budget:
                    best = candidate
                    lo = mid + 1
                else:
                    hi = mid - 1
            result.scene_memories = best
            current_tokens = self._estimate_bundle_tokens(result)
    
        # 3) Trim patterns to the strongest few if still over budget
        if result.emergent_patterns and current_tokens > token_budget:
            # keep the top-2 strongest patterns deterministically
            result.emergent_patterns = result.emergent_patterns[:2]
            current_tokens = self._estimate_bundle_tokens(result)
    
        # 4) Last resort: trim canon, but keep the top-5 most significant/most recent
        if result.canon_memories and current_tokens > token_budget and len(result.canon_memories) > 5:
            result.canon_memories = result.canon_memories[:5]
            current_tokens = self._estimate_bundle_tokens(result)
    
        result.bundle_size_tokens = current_tokens
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
        Store a new memory (telemetry-instrumented).
        """
        t0 = time.time()
    
        if not self.initialized:
            await self.initialize()
    
        try:
            manager = await self._get_entity_manager(entity_type, entity_id)
    
            md: Dict[str, Any] = dict(metadata or {})
            if is_canon:
                md["is_canon"] = True
                md["canonical"] = True
    
            memory = await manager.store_memory(
                content=content,
                memory_type=memory_type,
                significance=significance,
                emotional_intensity=emotional_intensity,
                tags=tags,
                metadata=md
            )
    
            # Invalidate caches using both entity and metadata signals
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            self._invalidate_by_metadata(md, tags)
    
            out = {
                "memory_id": getattr(memory, "memory_id", None) or getattr(memory, "id", None),
                "success": True,
                "entity_type": entity_type,
                "entity_id": entity_id
            }
    
            # Telemetry (success)
            try:
                await self._telemetry(
                    "store_memory",
                    started_at=t0,
                    success=True,
                    metadata={"et": entity_type, "eid": entity_id, "tags": (tags or [])}
                )
            except Exception:
                pass
    
            return out
    
        except Exception as e:
            # Telemetry (failure)
            try:
                await self._telemetry(
                    "store_memory",
                    started_at=t0,
                    success=False,
                    error=str(e),
                    metadata={"et": entity_type, "eid": entity_id, "tags": (tags or [])}
                )
            except Exception:
                pass
    
            logger.error(f"Error storing memory: {e}")
            return {"error": str(e), "success": False}

    @staticmethod
    def _as_et_str(et: Union[str, Enum]) -> str:
        try:
            # handle memory_orchestrator.EntityType and any Enum
            return et.value.lower() if isinstance(et, Enum) else str(et).lower()
        except Exception:
            return str(et).lower()
    
    async def retrieve_memories(
        self,
        entity_type: Union[str, Enum],
        entity_id: Union[int, str],
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_analysis: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        t0 = time.time()
    
        if not self.initialized:
            await self.initialize()
    
        # Accept old param names (e.g., use_llm_analysis)
        if not include_analysis and isinstance(kwargs.get("use_llm_analysis"), bool):
            include_analysis = kwargs["use_llm_analysis"]
    
        entity_type = self._as_et_str(entity_type)
    
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
                "ia": bool(include_analysis),
            }
            cache_key = prefix + hashlib.md5(
                json.dumps(key_payload, sort_keys=True, default=str).encode()
            ).hexdigest()
    
            cached = await self.cache.get(cache_key)
            if cached:
                logger.debug(f"Retrieval cache hit for entity {entity_type}:{entity_id}")
                result = cached
            else:
                logger.debug(f"Retrieval cache miss for entity {entity_type}:{entity_id}")
    
                # Get entity manager
                manager = await self._get_entity_manager(entity_type, entity_id)
    
                # Retrieve memories with robust fallbacks
                mem_list = []
                if query:
                    vs = await self.search_vector_store(
                        query=query,
                        entity_type=entity_type,
                        top_k=limit,
                        filter_dict=filters
                    )
                    mem_list = self._as_mem_list(vs)
                else:
                    if hasattr(manager, "retrieve_memories"):
                        raw = await manager.retrieve_memories(
                            query=None,
                            tags=tags,
                            limit=max(limit, 20)
                        )
                        mem_list = [m.to_dict() if hasattr(m, "to_dict") else m for m in raw]
                        ca = (filters or {}).get("created_after")
                        if ca:
                            try:
                                from datetime import datetime as _dt
                                cutoff = _dt.fromisoformat(str(ca).replace("Z", "+00:00"))
                                def _p_ts(x):
                                    s = x.get("timestamp") or x.get("created_at") or x.get("created")
                                    if not s: return None
                                    return _dt.fromisoformat(str(s).replace("Z", "+00:00"))
                                mem_list = [m for m in mem_list if (_p_ts(m) and _p_ts(m) >= cutoff)]
                            except Exception:
                                pass
                        if memory_type:
                            mt = str(memory_type).lower()
                            mem_list = [m for m in mem_list if str(m.get("memory_type","")).lower() == mt]
                        mem_list = mem_list[:limit]
                    elif hasattr(manager, "query_memories"):
                        res = await manager.query_memories(
                            filters=filters or {},
                            limit=limit,
                            memory_type=memory_type,
                            tags=tags
                        )
                        mem_list = res if isinstance(res, list) else res.get("memories", [])
                    elif hasattr(manager, "get_recent_memories"):
                        res = await manager.get_recent_memories(limit=limit)
                        mem_list = res if isinstance(res, list) else res.get("memories", [])
    
                result = {
                    "memories": mem_list or [],
                    "count": len(mem_list or []),
                    "entity_type": entity_type,
                    "entity_id": entity_id
                }
    
                if include_analysis and result.get("memories"):
                    result["analysis"] = await self.analyze_memory_set(
                        memories=result["memories"],
                        entity_type=entity_type,
                        entity_id=entity_id
                    )
    
                await self.cache.set(cache_key, result)
    
            # Telemetry (success)
            try:
                await self._telemetry(
                    "retrieve_memories",
                    started_at=t0,
                    success=True,
                    data_size=result.get("count", 0),
                    metadata={"et": entity_type, "eid": entity_id, "limit": limit}
                )
            except Exception:
                pass
    
            return result
    
        except Exception as e:
            # Telemetry (failure)
            try:
                await self._telemetry(
                    "retrieve_memories",
                    started_at=t0,
                    success=False,
                    error=str(e),
                    metadata={"et": entity_type, "eid": entity_id}
                )
            except Exception:
                pass
            logger.error(f"Error retrieving memories: {e}")
            return {"error": str(e), "memories": [], "count": 0}

    async def add_event_memory(
        self,
        npc_id: int,
        text: str,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        significance: int = 3
    ) -> Dict[str, Any]:
        """
        Store an NPC event as a memory via the integrated system.
        This replaces direct JSONB appends in NPCStats.memory.
        """
        if not self.initialized:
            await self.initialize()
    
        memory_kwargs = {
            "significance": significance,
            "tags": (tags or []) + ["npc_event"],
            "metadata": {"source": "logic.memory_logic", **(metadata or {})},
        }
        res = await self.integrated_add_memory(
            entity_type=EntityType.NPC.value,
            entity_id=npc_id,
            memory_text=text,
            memory_kwargs=memory_kwargs,
        )
        return res
    
    async def propagate_shared_memories(
        self,
        source_npc_id: int,
        source_npc_name: str,
        memories: Iterable[str],
    ) -> int:
        """
        Propagate 'secondhand' memories from one NPC to linked NPCs (trust/affection ≥ 50).
        Centralized version of logic/memory_logic.propagate_shared_memories.
        """
        if not self.initialized:
            await self.initialize()
    
        # Lazy import to avoid top-level dependency
        try:
            from db.connection import get_db_connection_context
        except Exception:
            logger.error("DB connection context not available for propagation")
            return 0
    
        propagated = 0
        try:
            async with get_db_connection_context() as conn:
                links = await conn.fetch("""
                    SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, dynamics
                      FROM SocialLinks
                     WHERE user_id=$1 AND conversation_id=$2
                       AND ((entity1_type='npc' AND entity1_id=$3) OR (entity2_type='npc' AND entity2_id=$3))
                """, self.user_id, self.conversation_id, int(source_npc_id))
    
            strong_links = []
            for link in links:
                dyn = link["dynamics"]
                if isinstance(dyn, str):
                    try:
                        dyn = json.loads(dyn)
                    except Exception:
                        dyn = {}
                if (dyn or {}).get("trust", 0) >= 50 or (dyn or {}).get("affection", 0) >= 50:
                    strong_links.append(link)
    
            tasks: List[asyncio.Task] = []
            for link in strong_links:
                e1t, e1i = link["entity1_type"], str(link["entity1_id"])
                e2t, e2i = link["entity2_type"], str(link["entity2_id"])
                target_t, target_i = (e2t, e2i) if (e1t == "npc" and e1i == str(source_npc_id)) else (e1t, e1i)
    
                if target_t != "npc" or target_i == str(source_npc_id):
                    continue
    
                for mem_text in memories:
                    secondhand = f"I heard that {source_npc_name}: \"{mem_text}\""
                    tasks.append(asyncio.create_task(
                        self.integrated_add_memory(
                            entity_type=EntityType.NPC.value,
                            entity_id=int(target_i) if target_i.isdigit() else target_i,
                            memory_text=secondhand,
                            memory_kwargs={
                                "significance": 1,  # low
                                "tags": ["propagated", "secondhand", f"from_npc:{source_npc_id}"],
                                "metadata": {"source": "propagation", "origin_npc_id": source_npc_id}
                            }
                        )
                    ))
    
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, dict) and not r.get("error"):
                        propagated += 1
    
        except Exception as e:
            logger.error(f"Propagation failed: {e}", exc_info=True)
    
        return propagated
    
    async def get_mask_perception_difficulty(self, npc_id: int) -> Dict[str, Any]:
        """
        Thin wrapper to expose ProgressiveRevealManager.get_perception_difficulty via orchestrator.
        """
        if not self.initialized:
            await self.initialize()
        if not self.progressive_reveal_manager:
            return {"error": "ProgressiveRevealManager not available"}
        try:
            return await self.progressive_reveal_manager.get_perception_difficulty(npc_id)
        except Exception as e:
            logger.error(f"get_mask_perception_difficulty failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def update_memory(
        self,
        entity_type: str,
        entity_id: int,
        memory_id: int,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing memory (telemetry-instrumented)."""
        t0 = time.time()
    
        if not self.initialized:
            await self.initialize()
    
        try:
            manager = await self._get_entity_manager(entity_type, entity_id)
            updated = await manager.update_memory(memory_id, updates)
    
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            self._invalidate_by_metadata(updates.get("metadata"), updates.get("tags"))
    
            out = {"success": True, "memory_id": memory_id}
    
            # Telemetry (success)
            try:
                await self._telemetry(
                    "update_memory",
                    started_at=t0,
                    success=True,
                    metadata={"et": entity_type, "eid": entity_id, "mid": memory_id}
                )
            except Exception:
                pass
    
            return out
    
        except Exception as e:
            # Telemetry (failure)
            try:
                await self._telemetry(
                    "update_memory",
                    started_at=t0,
                    success=False,
                    error=str(e),
                    metadata={"et": entity_type, "eid": entity_id, "mid": memory_id}
                )
            except Exception:
                pass
    
            logger.error(f"Error updating memory: {e}")
            return {"error": str(e), "success": False}
    
    async def delete_memory(
        self,
        entity_type: str,
        entity_id: int,
        memory_id: int
    ) -> Dict[str, Any]:
        """Delete a memory (telemetry-instrumented)."""
        t0 = time.time()
    
        if not self.initialized:
            await self.initialize()
    
        try:
            manager = await self._get_entity_manager(entity_type, entity_id)
    
            prev = None
            if hasattr(manager, "get_memory"):
                try:
                    prev = await manager.get_memory(memory_id)
                except Exception:
                    prev = None
    
            await manager.delete_memory(memory_id)
    
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            if prev:
                self._invalidate_by_metadata(prev.get("metadata"), prev.get("tags"))
    
            out = {"success": True, "memory_id": memory_id}
    
            # Telemetry (success)
            try:
                await self._telemetry(
                    "delete_memory",
                    started_at=t0,
                    success=True,
                    metadata={"et": entity_type, "eid": entity_id, "mid": memory_id}
                )
            except Exception:
                pass
    
            return out
    
        except Exception as e:
            # Telemetry (failure)
            try:
                await self._telemetry(
                    "delete_memory",
                    started_at=t0,
                    success=False,
                    error=str(e),
                    metadata={"et": entity_type, "eid": entity_id, "mid": memory_id}
                )
            except Exception:
                pass
    
            logger.error(f"Error deleting memory: {e}")
            return {"error": str(e), "success": False}
    
    # ========================================================================
    # Entity Manager Operations
    # ========================================================================
    
    async def _get_entity_manager(self, entity_type: str, entity_id: int):
            key = f"{entity_type}_{entity_id}"
            if key not in self.memory_managers:
                if entity_type == EntityType.NPC.value:
                    from memory.managers import NPCMemoryManager
                    manager = NPCMemoryManager(
                        npc_id=entity_id,
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
                elif entity_type == EntityType.PLAYER.value:
                    # Player manager in managers.py requires player_name; source from DB or default.
                    from memory.managers import PlayerMemoryManager
                    player_name = "Chase"
                    try:
                        from db.connection import get_db_connection_context
                        async with get_db_connection_context() as conn:
                            row = await conn.fetchrow(
                                "SELECT player_name FROM PlayerStats WHERE user_id=$1 AND conversation_id=$2 LIMIT 1",
                                self.user_id, self.conversation_id
                            )
                            if row and row.get("player_name"):
                                player_name = row["player_name"]
                    except Exception:
                        pass
                    manager = PlayerMemoryManager(
                        player_name=player_name,
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
                else:
                    # Fallback: core UnifiedMemoryManager (per-entity instance)
                    from memory.core import UnifiedMemoryManager
                    manager = UnifiedMemoryManager(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
                self.memory_managers[key] = manager
            return self.memory_managers[key]

    async def ensure_npc_mask(self, npc_id: int, overwrite: bool = False) -> Dict[str, Any]:
        """
        Ensure an NPC mask exists; initialize if necessary.
        Supports both new-style (no user_id args) and legacy (requires user_id, conversation_id).
        """
        if not self.initialized:
            await self.initialize()
        if not self.progressive_reveal_manager:
            return {"error": "ProgressiveRevealManager not available"}
        try:
            # New-style interface
            if hasattr(self.progressive_reveal_manager, "initialize_npc_mask"):
                try:
                    return await self.progressive_reveal_manager.initialize_npc_mask(
                        npc_id=npc_id, overwrite=overwrite
                    )
                except TypeError:
                    # Legacy interface expects explicit IDs
                    return await self.progressive_reveal_manager.initialize_npc_mask(
                        self.user_id, self.conversation_id, npc_id, overwrite
                    )
            return {"error": "initialize_npc_mask not supported"}
        except Exception as e:
            logger.error(f"ensure_npc_mask failed: {e}", exc_info=True)
            return {"error": str(e)}

    # -------- Integrated system convenience wrappers --------
    async def integrated_add_memory(
        self,
        entity_type: str,
        entity_id: int,
        memory_text: str,
        memory_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
    
        t0 = time.time()
        try:
            res = await self.integrated_system.add_memory(
                entity_type, entity_id, memory_text, memory_kwargs or {}
            )
    
            # Invalidate caches to reflect new content
            await self._invalidate_caches_for_entity(entity_type, entity_id)
            self._invalidate_by_metadata(
                (memory_kwargs or {}).get("metadata"),
                (memory_kwargs or {}).get("tags")
            )
    
            try:
                await self._telemetry("integrated_add_memory", started_at=t0, success=True,
                                      metadata={"et": entity_type, "eid": entity_id})
            except Exception:
                pass
    
            return res
    
        except Exception as e:
            try:
                await self._telemetry("integrated_add_memory", started_at=t0, success=False,
                                      error=str(e), metadata={"et": entity_type, "eid": entity_id})
            except Exception:
                pass
            raise

    async def integrated_retrieve_memories(self, entity_type: str, entity_id: int, query: Optional[str] = None, current_context: Optional[Dict[str, Any]] = None, retrieval_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        return await self.integrated_system.retrieve_memories(entity_type, entity_id, query=query, current_context=current_context or {}, retrieval_kwargs=retrieval_kwargs or {})

    async def run_integrated_maintenance(self, entity_type: str, entity_id: int, options: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        res = await self.integrated_system.run_memory_maintenance(entity_type, entity_id, maintenance_options=options or {})
        # Broad invalidation after maintenance
        await self._invalidate_caches_for_entity(entity_type, entity_id)
        return res
    
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
        await self.cache.clear()
    
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
        t0 = time.time()
        try:
            # Create scope from context
            scope = SceneScope(
                location_id=context.get('current_location'),
                npc_ids=set(context.get('active_npcs', [])),
                topics=set(context.get('topics', [])),
                lore_tags=set(context.get('lore_tags', []))
            )
    
            # Get scene bundle
            bundle = await self.get_scene_bundle(scope)
    
            # Trauma triggers → add to linked if any
            try:
                trig_player = await self.emotional_manager.process_traumatic_triggers(
                    entity_type=EntityType.PLAYER.value, entity_id=self.user_id, text=user_input
                )
                linked_extra = []
                if trig_player.get("triggered"):
                    for tm in trig_player.get("triggered_memories", []):
                        linked_extra.append({
                            "id": tm.get("id"),
                            "text": tm.get("text"),
                            "tags": ["trauma_trigger","player"]
                        })
                for npc_id in context.get('active_npcs', []):
                    trig_npc = await self.emotional_manager.process_traumatic_triggers(
                        entity_type=EntityType.NPC.value, entity_id=npc_id, text=user_input
                    )
                    if trig_npc.get("triggered"):
                        for tm in trig_npc.get("triggered_memories", []):
                            linked_extra.append({
                                "id": tm.get("id"),
                                "text": tm.get("text"),
                                "tags": ["trauma_trigger","npc"]
                            })
                if linked_extra:
                    bundle['linked'] = (bundle.get('linked') or []) + linked_extra
            except Exception:
                pass
    
            # Optional: combine with retriever analysis for extra insights
            try:
                from memory.memory_integration import enrich_context_with_memories as _enrich_legacy
                context = await _enrich_legacy(self.user_id, self.conversation_id, user_input, context)
            except Exception:
                pass
    
            # Optional: add retriever's synthesis/analysis (LangChain agent)
            try:
                if self.retriever_agent and hasattr(self.retriever_agent, "retrieve_and_analyze"):
                    rr = await self.retriever_agent.retrieve_and_analyze(
                        query=user_input,
                        entity_types=["memory", "npc", "location", "narrative"],
                        top_k=5,
                        threshold=(self.memory_config or {}).get("vector_store", {}).get("similarity_threshold", 0.7)
                    )
                    if rr and rr.get("found_memories"):
                        context.setdefault("memory_analysis", {})
                        ma = rr.get("analysis")
                        if ma and hasattr(ma, "dict"):
                            ma = ma.dict()
                        context["memory_analysis"].update({
                            "primary_theme": (ma or {}).get("primary_theme"),
                            "insights": (ma or {}).get("insights", []),
                            "suggested_response": (ma or {}).get("suggested_response")
                        })
            except Exception:
                pass
    
            # Merge bundle into context
            context['memories'] = {
                'canon': bundle.get('canon', []),
                'scene': bundle.get('scene', []),
                'linked': bundle.get('linked', []),
                'patterns': bundle.get('patterns', [])
            }
    
            try:
                await self._telemetry("enrich_context", started_at=t0, success=True,
                                      metadata={"loc": context.get("current_location"),
                                                "npc_count": len(context.get("active_npcs", []))})
            except Exception:
                pass
    
            return context
    
        except Exception as e:
            try:
                await self._telemetry("enrich_context", started_at=t0, success=False, error=str(e))
            except Exception:
                pass
            raise

    async def get_scene_brief(self, scope) -> Dict[str, Any]:
        brief: Dict[str, Any] = {"anchors": {}, "signals": {}, "links": {}}
        try:
            topics: List[str] = list(getattr(scope, "topics", []) or [])
            lore_tags: List[str] = list(getattr(scope, "lore_tags", []) or [])
            if topics:
                brief["anchors"]["topics"] = topics[:5]
            if lore_tags:
                brief["anchors"]["lore_tags"] = lore_tags[:5]

            # optional tiny mood hint; soft-fail if not available
            try:
                emo = await self.emotional_manager.get_entity_emotional_state(  # type: ignore[attr-defined]
                    entity_type="player", entity_id=self.user_id
                )
                mood = (emo or {}).get("current_emotion", {}).get("primary")
                if mood:
                    brief["signals"]["dominant_mood"] = str(mood)
            except Exception:
                pass
        except Exception:
            pass
        return brief
    
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
        """Search the vector store directly with concurrency limiting (telemetry-instrumented)."""
        t0 = time.time()
    
        if not self.initialized:
            await self.initialize()
    
        try:
            async with self.__class__._sem():
                res = await self.embedding_service.search_memories(
                    query_text=query,
                    entity_type=entity_type,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    fetch_content=True
                )
    
            # Telemetry (success)
            try:
                await self._telemetry(
                    "search_vector_store",
                    started_at=t0,
                    success=True,
                    data_size=len(res or []),
                    metadata={"query": query, "top_k": top_k, "et": entity_type}
                )
            except Exception:
                pass
    
            return res
    
        except Exception as e:
            # Telemetry (failure)
            try:
                await self._telemetry(
                    "search_vector_store",
                    started_at=t0,
                    success=False,
                    error=str(e),
                    metadata={"query": query, "top_k": top_k}
                )
            except Exception:
                pass
    
            raise
    
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
        if not self.initialized:
            await self.initialize()
    
        t0 = time.time()
        try:
            # Prefer Nyx governance bridge if present
            if self.nyx_bridge:
                try:
                    res = await self.nyx_bridge.remember(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        memory_text=memory_text,
                        importance=importance,
                        emotional=emotional,
                        tags=tags
                    )
                    # Invalidate caches to reflect new content
                    await self._invalidate_caches_for_entity(entity_type, entity_id)
                    self._invalidate_by_metadata(None, tags or [])
                    try:
                        await self._telemetry("agent_remember", started_at=t0, success=True,
                                              metadata={"et": entity_type, "eid": entity_id})
                    except Exception:
                        pass
                    return res
                except Exception as e:
                    logger.warning(f"Nyx-bridge remember failed, fallback to agent wrapper: {e}")
    
            if not self.memory_agent_wrapper:
                raise RuntimeError("No memory agent wrapper available")
    
            out = await self.memory_agent_wrapper.remember(
                run_context=None,
                entity_type=entity_type,
                entity_id=entity_id,
                memory_text=memory_text,
                importance=importance,
                emotional=emotional,
                tags=tags
            )
            # Invalidate for wrapper path too
            try:
                await self._invalidate_caches_for_entity(entity_type, entity_id)
                self._invalidate_by_metadata(None, tags or [])
            except Exception:
                pass
    
            try:
                await self._telemetry("agent_remember", started_at=t0, success=True,
                                      metadata={"et": entity_type, "eid": entity_id})
            except Exception:
                pass
            return out
    
        except Exception as e:
            try:
                await self._telemetry("agent_remember", started_at=t0, success=False, error=str(e),
                                      metadata={"et": entity_type, "eid": entity_id})
            except Exception:
                pass
            raise

    async def agent_recall(
        self,
        entity_type: str,
        entity_id: int,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
    
        t0 = time.time()
        try:
            # Prefer Nyx governance bridge if present
            if self.nyx_bridge:
                try:
                    out = await self.nyx_bridge.recall(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        query=query,
                        context=context,
                        limit=limit
                    )
                    try:
                        await self._telemetry("agent_recall", started_at=t0, success=True,
                                              metadata={"et": entity_type, "eid": entity_id, "limit": limit})
                    except Exception:
                        pass
                    return out
                except Exception as e:
                    logger.warning(f"Nyx-bridge recall failed, fallback to agent wrapper: {e}")
    
            if not self.memory_agent_wrapper:
                raise RuntimeError("No memory agent wrapper available")
    
            out = await self.memory_agent_wrapper.recall(
                run_context=None,
                entity_type=entity_type,
                entity_id=entity_id,
                query=query,
                context=context,
                limit=limit
            )
            try:
                await self._telemetry("agent_recall", started_at=t0, success=True,
                                      metadata={"et": entity_type, "eid": entity_id, "limit": limit})
            except Exception:
                pass
            return out
    
        except Exception as e:
            try:
                await self._telemetry("agent_recall", started_at=t0, success=False, error=str(e),
                                      metadata={"et": entity_type, "eid": entity_id})
            except Exception:
                pass
            raise

    async def get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        if not self.progressive_reveal_manager:
            return {"error": "ProgressiveRevealManager not available"}
        return await self.progressive_reveal_manager.get_npc_mask(npc_id)

    async def generate_mask_slippage(
        self,
        npc_id: int,
        trigger: Optional[str] = None,
        severity: Optional[int] = None,
        reveal_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        if not self.progressive_reveal_manager:
            return {"error": "ProgressiveRevealManager not available"}

        res = await self.progressive_reveal_manager.generate_mask_slippage(
            npc_id=npc_id, trigger=trigger, severity=severity, reveal_type=reveal_type
        )

        # Invalidate NPC + tags (mask_slippage, trait) to refresh scene bundles
        try:
            await self._invalidate_caches_for_entity(EntityType.NPC.value, npc_id)
            tags = ["mask_slippage"]
            if isinstance(res, dict) and res.get("trait_revealed"):
                tags.append(res["trait_revealed"])
            self._invalidate_by_metadata(None, tags)
        except Exception:
            pass

        return res

    async def check_automated_mask_reveals(self) -> List[Dict[str, Any]]:
        if not self.initialized:
            await self.initialize()
        if not self.progressive_reveal_manager:
            return []
    
        try:
            # New-style: bound to user/conversation on init – no args
            try:
                reveals = await self.progressive_reveal_manager.check_for_automated_reveals()
            except TypeError:
                # Legacy: method expects user_id, conversation_id
                reveals = await self.progressive_reveal_manager.check_for_automated_reveals(
                    self.user_id, self.conversation_id
                )
        except Exception:
            reveals = []
    
        # Invalidate caches for affected NPCs
        affected_npcs = {r.get("npc_id") for r in reveals if r.get("npc_id")}
        for nid in affected_npcs:
            try:
                await self._invalidate_caches_for_entity(EntityType.NPC.value, nid)
            except Exception:
                pass
        return reveals
    
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

        # Progressive mask auto-reveals (optional)
        try:
            if self.progressive_reveal_manager:
                auto_reveals = await self.check_automated_mask_reveals()
                results['mask_auto_reveals'] = len(auto_reveals)
        except Exception as e:
            logger.debug(f"Automated mask reveals skipped: {e}")

        # Reconsolidation sweep (low-intensity)
        try:
            if self.reconsolidation_manager:
                sweep_ids = await self.run_reconsolidation_sweep(EntityType.PLAYER.value, self.user_id, max_memories=3)
                results['reconsolidation'] = {"reconsolidated_count": len(sweep_ids)}
        except Exception as e:
            results['reconsolidation'] = {"error": str(e)}
        
        self.last_maintenance = datetime.now()
        
        return results

    async def create_belief(
        self,
        entity_type: str,
        entity_id: int,
        belief_text: str,
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Create a belief for an entity (prefers new semantic manager; falls back to integrated).
        """
        if not self.initialized:
            await self.initialize()
    
        # Prefer new semantic manager if available
        if self.semantic_manager and hasattr(self.semantic_manager, "create_belief"):
            try:
                return await self.semantic_manager.create_belief(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    belief_text=belief_text,
                    confidence=confidence
                )
            except Exception as e:
                logger.warning(f"semantic_manager.create_belief failed, falling back to integrated: {e}")
    
        # Fallback: integrated system’s semantic manager
        try:
            return await self.integrated_system.semantic_manager.create_belief(
                entity_type=entity_type,
                entity_id=entity_id,
                belief_text=belief_text,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"create_belief failed: {e}")
            return {"error": str(e)}
    
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

    async def remember_emotional(
        self,
        entity_type: str,
        entity_id: int,
        memory_text: str,
        primary_emotion: str,
        emotion_intensity: float,
        secondary_emotions: Optional[Dict[str, float]] = None,
        significance: int = 3,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Store a memory with emotional data and invalidate affected caches."""
        res = await self.emotional_manager.add_emotional_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_text=memory_text,
            primary_emotion=primary_emotion,
            emotion_intensity=emotion_intensity,
            secondary_emotions=secondary_emotions,
            significance=significance,
            tags=tags or []
        )
        # Invalidate caches for the entity
        await self._invalidate_caches_for_entity(entity_type, entity_id)
        return res
    
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
        if self.embedding_service:
            await self.embedding_service.close()
        if self.retriever_agent and hasattr(self.retriever_agent, "close"):
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

async def recall(
    user_id: int,
    conversation_id: int,
    entity_type: Union[str, Enum],
    entity_id: Union[int, str],
    query: Optional[str] = None,
    context: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Module-level recall that grabs the orchestrator and calls its agent_recall."""
    orchestrator = await get_memory_orchestrator(user_id, conversation_id)
    return await orchestrator.agent_recall(
        entity_type=entity_type,
        entity_id=entity_id,
        query=query,
        context=context,
        limit=limit,
    )


async def remember(
    user_id: int,
    conversation_id: int,
    entity_type: Union[str, Enum],
    entity_id: Union[int, str],
    memory_text: str,
    importance: str = "medium",
    emotional: bool = True,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Module-level remember that grabs the orchestrator and calls its agent_remember."""
    orchestrator = await get_memory_orchestrator(user_id, conversation_id)
    return await orchestrator.agent_remember(
        entity_type=entity_type,
        entity_id=entity_id,
        memory_text=memory_text,
        importance=importance,
        emotional=emotional,
        tags=tags,
    )

__all__ = [
    "MemoryOrchestrator",
    "get_memory_orchestrator",
    "recall",
    "remember",
]
