# lore/lore_orchestrator.py
"""
Lore Orchestrator - ENHANCED with Scene Bundle Support
=====================================================
Master orchestrator for all lore operations across the Nyx game.
Acts as the single entry point for external systems to interact with lore.

FULLY INTEGRATED: Includes education, geopolitical, local lore, politics, religion, 
and world lore managers with all their specialized functionality.

NEW: Scene-scoped bundle methods for optimized context assembly.
"""
from __future__ import annotations

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set, AsyncGenerator, Protocol, Iterable
from datetime import datetime, timedelta
import json
from enum import Enum
from dataclasses import dataclass, field
import os
import asyncpg
import hashlib
import time
import inspect
import re
from agents import InputGuardrailTripwireTriggered

# Core imports
from db.connection import get_db_connection_context

# Nyx governance
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission

# Agents SDK imports for new managers
from agents import Agent, function_tool, Runner, ModelSettings, trace, handoff
from agents.run import RunConfig
from agents.run_context import RunContextWrapper
from pydantic import BaseModel, Field

from nyx.scene_keys import generate_scene_cache_key 

# Import the specialized manager input/output models
from lore.managers.education import (
    EducationalSystem, KnowledgeTradition, TeachingContent,
    KnowledgeExchangeResult, StreamingPhaseUpdate
)
from lore.managers.geopolitical import (
    GeographicRegion, PoliticalEntity, BorderDispute, ConflictSimulation,
    EconomicTradeSimulation, ClimateGeographyEffect, CovertOperation
)
from lore.managers.local_lore import (
    LocationDataInput, MythCreationInput, HistoryCreationInput, LandmarkCreationInput,
    UrbanMyth, LocalHistory, Landmark, NarrativeEvolution, MythTransmissionResult,
    NarrativeConnection, ConsistencyCheckResult, TouristDevelopment, TraditionDynamics,
    LegendVariant, LocationLoreResult, LoreEvolutionResult,
    NarrativeStyle, EvolutionType, ConnectionType
)

# Import models from politics module
from lore.managers.politics import (
    DiplomaticNegotiationResult, MediaCoverageItem, DiplomaticNegotiation,
    FactionAgentProxy
)

# Import models from religion module  
from lore.managers.religion import (
    DeityParams, PantheonParams, ReligiousPracticeParams, HolySiteParams,
    ReligiousTextParams, ReligiousOrderParams, ReligiousConflictParams,
    NationReligionDistribution, CompleteRitual, SectarianPosition
)

# Cache + Canon integration
from lore.core.cache import GLOBAL_LORE_CACHE
from lore.core.canon import (
    initialize_canon_memory_integration,
    log_canonical_event as canon_log_canonical_event,
    find_or_create_npc,
    find_or_create_nation,
    find_or_create_location,
    find_or_create_faction,
    find_or_create_historical_event,
    find_or_create_urban_myth,
    find_or_create_landmark,
    find_or_create_event,
    find_or_create_quest,
    sync_entity_to_memory,
    ensure_embedding_columns,
)

logger = logging.getLogger(__name__)

# Database connection
DB_DSN = os.getenv("DB_DSN")

# Singleton instance storage
_ORCHESTRATOR_INSTANCES: Dict[Tuple[int, int], "LoreOrchestrator"] = {}

def _as_canonical_ctx(self, ctx: Any = None):
    Ctx = self._get_canonical_context_class()
    if ctx and hasattr(ctx, 'context'):
        return ctx
    if isinstance(ctx, dict):
        return Ctx.from_dict(ctx)
    return Ctx(self.user_id, self.conversation_id)

# ===== CANONICAL LORE BUNDLE SCHEMA CONTRACT =====
# All orchestrator-facing data should conform to these shapes.
#
# Nation (politics):
#   { id: int, name: str, government: str, culture: dict|str, matriarchy_level?: int }
#
# Conflict (international or local):
#   {
#       id: int, type: str, description: str,
#       stakeholders: [{type: 'nation'|'faction'|'npc'|'location'|..., id: int, name?: str}], 
#       intensity: float (0..1), phase: str, resolution_status: 'ongoing'|'resolved'|...
#   }
#
# Religion:
#   { id: int, name: str, deities?: [str], beliefs?: str, influence?: float }
#
# Pantheon:
#   { id: int, name: str, description?: str, matriarchal_elements?: any }
#
# Deity:
#   { id: int, name: str, domains?: [str], description?: str }
#
# ReligiousPractice:
#   { id: int, name: str, practice_type?: str, description?: str, purpose?: str }
#
# HolySite:
#   { id: int, name: str, location_id?: int, description?: str }
#
# ReligiousOrder:
#   { id: int, name: str, doctrine?: str, hierarchy?: any }
#
# NationReligionDistribution:
#   {
#     id: int, nation_id: int, state_religion: bool,
#     primary_pantheon_id?: int,
#     pantheon_distribution?: { [pantheon_id: str]: number },
#     religiosity_level?: int, religious_tolerance?: int,
#     religious_laws?: dict, religious_holidays?: [str],
#     religious_conflicts?: [str], religious_minorities?: [str]
#   }
#
# Location:
#   { id: int, name: str, description?: str, nation_id?: int, ... }
#
# Myth:
#   { id: int, title: str, description?: str, origin?: str, belief_level?: float, has_variants?: bool }
#
# World Lore (free-form, but keys inside bundles should be simple scalars/arrays/dicts)
# ================================================

class SceneScope(Protocol):
    """Protocol for scene scope objects."""
    location_id: Optional[int]
    npc_ids: Set[int]
    lore_tags: Set[str]
    topics: Set[str]
    conflict_ids: Set[int]
    nation_ids: Set[int]
    link_hints: Dict[str, Any]
    
    def to_key(self) -> str:
        """Generate a cache key for this scope."""
        ...


@dataclass
class OrchestratorConfig:
    """Configuration for the Lore Orchestrator"""
    enable_matriarchal_theme: bool = True
    enable_governance: bool = True
    enable_metrics: bool = True
    enable_validation: bool = True
    enable_cache: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    bundle_cache_max_size: int = 100  # Separate limit for scene bundles
    max_parallel_operations: int = 10
    auto_initialize: bool = True
    resource_limits: Dict[str, Any] = None
    redis_url: Optional[str] = None
    max_size_mb: float = 100
    bundle_ttl: float = 60.0  # TTL for scene bundles
    subfetch_timeout: float = 1.5  # Timeout for individual fetch operations
    

@dataclass
class SceneBundleData:
    """Scene-specific lore bundle data"""
    location: Dict[str, Any] = field(default_factory=dict)
    world: Dict[str, Any] = field(default_factory=dict)
    canonical_rules: List[Union[str, Dict[str, str]]] = field(default_factory=list)
    nations: List[Dict[str, Any]] = field(default_factory=list)
    religions: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    myths: List[Dict[str, Any]] = field(default_factory=list)
    languages: List[Dict[str, Any]] = field(default_factory=list)
    cultural_norms: List[Dict[str, Any]] = field(default_factory=list)
    etiquette: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'location': self.location,
            'world': self.world,
            'canonical_rules': self.canonical_rules,
            'nations': self.nations,
            'religions': self.religions,
            'conflicts': self.conflicts,
            'myths': self.myths,
            'languages': self.languages,
            'cultural_norms': self.cultural_norms,
            'etiquette': self.etiquette,
        }


class LoreOrchestrator:
    """
    Master orchestrator that provides unified access to all lore functionality.
    Acts as the single entry point for external systems to interact with lore.
    
    FULLY INTEGRATED with:
    - Educational system management
    - Geopolitical system management  
    - Local lore and urban myth management
    - Politics and diplomacy management
    - Religion and faith systems management
    - World lore and resource management
    - Canon system for canonical state management
    - Cache system for performance optimization
    - Registry system for manager coordination
    - Validation system for data integrity
    
    NEW: Scene bundle support for optimized context assembly
    """
    
    @classmethod
    def get_instance(cls, user_id: int, conversation_id: int, config: Optional[OrchestratorConfig] = None) -> "LoreOrchestrator":
        """Get or create a singleton instance for the given user/conversation."""
        key = (user_id, conversation_id)
        
        if key not in _ORCHESTRATOR_INSTANCES:
            _ORCHESTRATOR_INSTANCES[key] = cls(user_id, conversation_id, config)
        elif config:
            # Update config if provided
            _ORCHESTRATOR_INSTANCES[key].config = config
            
        return _ORCHESTRATOR_INSTANCES[key]
    
    def __init__(self, user_id: int, conversation_id: int, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the Lore Orchestrator.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            config: Optional configuration
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = config or OrchestratorConfig()
        self.initialized = False
        
        # Core components (lazy loaded)
        self._lore_system = None
        self._canon_module = None
        self._cache_system = None
        self._registry_system = None
        self._canon_validation = None
        self._canonical_context_class = None

        self.nyx_context = None
        self.context_broker = None
        
        # Manager instances (lazy loaded)
        self._education_manager = None
        self._religion_manager = None
        self._local_lore_manager = None
        self._geopolitical_manager = None
        self._politics_manager = None
        self._regional_culture_system = None
        self._world_lore_manager = None
        
        # Integration components (lazy loaded)
        self._npc_integration = None
        self._conflict_integration = None
        self._context_enhancer = None
        self._lore_generator = None
        self._master_coordinator = None
        self._content_validator = None
        self._relationship_mapper = None
        self._unified_trace_system = None
        
        # Data access components (lazy loaded)
        self._npc_data_access = None
        self._location_data_access = None
        self._faction_data_access = None
        self._knowledge_access = None
        
        # Metrics
        self.metrics = {
            "operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bundle_hits": 0,
            "bundle_misses": 0,
            "db_roundtrips": 0,
            "errors": 0,
            "last_operation": None
        }
        
        # Scene bundle cache
        self._scene_bundle_cache = {}
        self._bundle_last_changed = {}  # Wall clock time for external reference
        self._bundle_cached_at = {}     # Monotonic time for TTL checks  
        self._bundle_ttl = config.bundle_ttl if config else 60.0
        self._cache_lock = asyncio.Lock()  # Prevent cache race conditions
        
        # Change tracking for delta updates
        self._element_snapshots = {}  # Track last known state of elements
        self._change_log = {}  # Track changes per scope
        self._change_tracking_enabled = True
        
        logger.info(f"LoreOrchestrator created for user {user_id}, conversation {conversation_id}")
    
    # ===== INITIALIZATION =====
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self.initialized:
            return
    
        logger.info(f"Initializing LoreOrchestrator for user {self.user_id}")
    
        try:
            # Initialize core systems
            await self._get_lore_system()
            await self._get_canon_module()
    
            if self.config.enable_cache:
                await self._get_cache_system()
    
            # Prepare the registry
            await self._get_registry_system()
    
            # Initialize database if needed
            await self._ensure_database_setup()
    
            # NEW: bootstrap canon ↔ memory integration (best-effort)
            try:
                await initialize_canon_memory_integration(self.user_id, self.conversation_id)
            except Exception as e:
                logger.debug(f"Canon-memory integration init skipped: {e}")
    
            self.initialized = True
            logger.info("LoreOrchestrator initialization complete")
    
        except Exception as e:
            logger.error(f"Failed to initialize LoreOrchestrator: {e}", exc_info=True)
            raise
    
    async def _ensure_database_setup(self) -> None:
        """Ensure database tables and indexes exist."""
        try:
            async with get_db_connection_context() as conn:
                # Check if Nations table exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'nations'
                    )
                """)
                
                if not exists:
                    logger.info("Creating lore database tables...")
                    # This would trigger the creation of all necessary tables
                    # through the canon module's initialization
                    canon = await self._get_canon_module()
                    if hasattr(canon, 'initialize_database'):
                        await canon.initialize_database(conn)
                
                # Create change tracking table if it doesn't exist
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS LoreChangeLog (
                        change_id SERIAL PRIMARY KEY,
                        conversation_id INTEGER NOT NULL,
                        element_type VARCHAR(50) NOT NULL,
                        element_id INTEGER NOT NULL,
                        operation VARCHAR(20) NOT NULL,
                        changed_fields JSONB,
                        old_value JSONB,
                        new_value JSONB,
                        scope_keys TEXT[],
                        timestamp TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create indexes for efficient querying
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_changes_conversation_time 
                    ON LoreChangeLog(conversation_id, timestamp DESC)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_changes_scope 
                    ON LoreChangeLog USING GIN(scope_keys)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_changes_element 
                    ON LoreChangeLog(element_type, element_id)
                """)
                    
        except Exception as e:
            logger.warning(f"Database setup check failed: {e}")
            # Continue anyway - tables may be created on demand

    def attach_nyx_context(self, nyx_context: Any = None, context_broker: Any = None) -> None:
        """
        Optional: call this once after constructing to enable cross-system invalidation.
        Either pass nyx_context (that has .context_broker) or pass context_broker directly.
        """
        self.nyx_context = nyx_context
        self.context_broker = context_broker or getattr(nyx_context, "context_broker", None)

    
    def _create_mock_context(self, **attributes) -> Any:
        """
        Create a mock context object for operations that require governance context.
        
        This utility method creates lightweight mock objects that satisfy
        the requirements of @with_governance decorated functions without
        requiring full governance context imports.
        
        Args:
            **attributes: Key-value pairs to set as object attributes
            
        Returns:
            Mock object with specified attributes
        """
        # Default context structure many governance decorators expect
        if 'context' not in attributes:
            attributes['context'] = {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            }
        
        # Create and return mock object with attributes
        return type('MockContext', (object,), attributes)()
    
    # ===== SCENE BUNDLE METHODS (NEW) =====
    
    async def get_scene_bundle(self, scope: Any) -> Dict[str, Any]:
        """
        Get scene-scoped lore bundle with canonical data prioritized.
        Includes matriarchal framework context when enabled.
        """
        start_time = time.perf_counter()
    
        # Generate cache key from scope
        cache_key = self._generate_scene_cache_key(scope)
    
        # Check cache first with lock for race safety
        async with self._cache_lock:
            cached = self._get_cached_bundle(cache_key)
            if cached and not self._is_bundle_stale(cached):
                logger.debug(f"Scene bundle cache hit for key {cache_key[:8]}")
                self.metrics['bundle_hits'] = self.metrics.get('bundle_hits', 0) + 1
                return cached
    
        self.metrics['bundle_misses'] = self.metrics.get('bundle_misses', 0) + 1
    
        # Build fresh bundle in parallel
        bundle_data = SceneBundleData()
        canonical = False
    
        # Prepare parallel tasks with semaphore
        sem = asyncio.Semaphore(self.config.max_parallel_operations)
    
        async def _fetch_with_semaphore(task_coro):
            """Execute task with semaphore limit."""
            async with sem:
                return await task_coro
    
        # Use configurable timeout
        timeout = self.config.subfetch_timeout if hasattr(self.config, 'subfetch_timeout') else 1.5
    
        async def _with_timeout(coro, label):
            """Wrap task with timeout to prevent tail latency."""
            try:
                return await asyncio.wait_for(coro, timeout)
            except asyncio.TimeoutError:
                logger.warning(f"{label} fetch timed out after {timeout}s")
                return None
    
        # Build task list based on scope
        tasks = []
    
        loc_ref = getattr(scope, 'location_id', None)
        if isinstance(loc_ref, int):
            tasks.append(('location', self._fetch_location_lore_for_bundle(loc_ref)))
            tasks.append(('religions', self._fetch_religions_for_location(loc_ref)))
            tasks.append(('myths', self._fetch_myths_for_location(loc_ref)))
        elif isinstance(loc_ref, str) and loc_ref:
            # Name-only: fetch the location context, skip id-dependent tables
            tasks.append(('location', self.get_location_context(loc_ref)))
    
        if hasattr(scope, 'lore_tags') and scope.lore_tags:
            tasks.append(('world', self._fetch_world_lore_for_bundle(list(scope.lore_tags)[:10])))

        if hasattr(scope, 'conflict_ids') and scope.conflict_ids:
            cid_list = list(scope.conflict_ids)[:5]
            tasks.append(('conflicts', self._fetch_conflicts_for_bundle(cid_list)))
    
        if hasattr(scope, 'nation_ids') and scope.nation_ids:
            nid_list = list(scope.nation_ids)[:5]
            tasks.append(('nations', self._fetch_nations_for_bundle(nid_list)))
            tasks.append(('culture', self._fetch_cultural_data_for_bundle(nid_list)))
    
        # NEW: Matriarchal framework tasks (optional)
        need_mpf = bool(self.config.enable_matriarchal_theme) and (
            (hasattr(scope, 'nation_ids') and scope.nation_ids) or
            (getattr(scope, 'location_id', None) is not None) or
            (hasattr(scope, 'lore_tags') and scope.lore_tags)
        )
        # In get_scene_bundle(), the matriarchal framework tasks section:
        if need_mpf:
            tasks.append(('mpf_core', self.mpf_generate_core_principles()))
            tasks.append(('mpf_expressions', self.mpf_generate_power_expressions(limit=3)))
            # You could also add:
            tasks.append(('mpf_constraints', self.mpf_generate_hierarchical_constraints()))
    
        # Execute all tasks in parallel with timeout protection
        try:
            # Create wrapped tasks with semaphore and timeout
            wrapped_tasks = [
                (label, asyncio.create_task(
                    _with_timeout(_fetch_with_semaphore(coro), label)
                ))
                for label, coro in tasks
            ]
    
            # Wait for all to complete
            results = []
            for label, task in wrapped_tasks:
                try:
                    result = await task
                    results.append((label, result))
                except Exception as e:
                    logger.warning(f"Failed to fetch {label}: {e}")
                    results.append((label, None))
    
            # Process results
            for label, data in results:
                if data is None:
                    continue
    
                if label == 'location':
                    bundle_data.location = data
                    if 'canonical_rules' in data:
                        bundle_data.canonical_rules.extend([
                            {"text": rule, "source": "location"}
                            for rule in data['canonical_rules']
                        ])
                        canonical = True
    
                elif label == 'world':
                    bundle_data.world = data
    
                elif label == 'nations':
                    bundle_data.nations = data
                    canonical = True  # Nations are canonical
    
                elif label == 'culture':
                    bundle_data.languages = data.get('languages', [])
                    bundle_data.cultural_norms = data.get('norms', [])
                    bundle_data.etiquette = data.get('etiquette', [])
    
                # NEW: Matriarchal framework results
                elif label == 'mpf_core':
                    # Ensure world dict exists and attach principles
                    bundle_data.world = bundle_data.world or {}
                    bundle_data.world["matriarchal_principles"] = data
                    canonical = True
    
                elif label == 'mpf_expressions':
                    # Ensure world dict exists and attach expressions
                    bundle_data.world = bundle_data.world or {}
                    bundle_data.world["power_expressions"] = data
    
                elif label == 'religions':
                    bundle_data.religions = data[:3]  # Top 3 religions
    
                elif label == 'conflicts':
                    bundle_data.conflicts = data
    
                elif label == 'myths':
                    bundle_data.myths = data[:3]  # Top 3 myths
    
            # Add canonical consistency rules (after main fetches)
            if self.config.enable_validation:
                try:
                    canonical_rules = await _with_timeout(
                        _fetch_with_semaphore(self._get_canonical_rules_for_scope(scope)),
                        'canonical_rules'
                    )
                    if canonical_rules:
                        bundle_data.canonical_rules.extend([
                            {"text": rule, "source": "validation"}
                            for rule in canonical_rules
                        ])
                        canonical = True
                except Exception as e:
                    logger.debug(f"Could not fetch canonical rules: {e}")
    
        except Exception as e:
            logger.error(f"Error building scene bundle: {e}")
            # Return minimal bundle on error
    
        # Calculate build time
        build_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['build_ms'] = self.metrics.get('build_ms', [])
        if isinstance(self.metrics['build_ms'], list):
            self.metrics['build_ms'].append(build_ms)
            # Keep only last 100 measurements
            if len(self.metrics['build_ms']) > 100:
                self.metrics['build_ms'] = self.metrics['build_ms'][-100:]
    
        # Build result with consistent structure and anchors
        result = {
            'section': 'lore',  # Consistent with other bundles
            'anchors': {
                'location_id': getattr(scope, 'location_id', None),
                'nation_ids': sorted(list(getattr(scope, 'nation_ids', set())))[:5],
                'conflict_ids': sorted(list(getattr(scope, 'conflict_ids', set())))[:5],
            },
            'data': bundle_data.to_dict(),
            'canonical': canonical,  # Changed from 'canon'
            'last_changed_at': time.time(),
            'version': f"lore_{cache_key[:8]}_{int(time.time())}",
            'build_ms': build_ms
        }
    
        # Optional: apply matriarchal lens before caching
        if self.config.enable_matriarchal_theme:
            try:
                result = await self.apply_matriarchal_lens_to_bundle(result)
            except Exception as e:
                logger.debug(f"Matriarchal lens application skipped due to error: {e}")
    
        # Cache the bundle with lock for race safety
        async with self._cache_lock:
            self._cache_bundle(cache_key, result)
    
        return result

    async def quick_setup_world(self, world_description: str) -> Dict[str, Any]:
        """
        Quick setup for a new world with core systems seeded.
        Note: nations are generated via GeopoliticalSystemManager tool function.
        """
        results = {
            'world': None,
            'nations': [],
            'religions': [],
            'education': [],
            'cultures': [],
            'status': 'initializing'
        }
        try:
            await self.initialize()
            ctx = self._create_mock_context()
            
            # Base world
            world = await self.generate_complete_world(ctx, world_description)
            results['world'] = world
            
            # Nations (use class static tool to avoid instance-binding pitfalls)
            from agents.run_context import RunContextWrapper
            run_ctx = RunContextWrapper(context={'user_id': self.user_id, 'conversation_id': self.conversation_id})
            from lore.managers.geopolitical import GeopoliticalSystemManager
            nations = await GeopoliticalSystemManager.generate_world_nations(run_ctx, count=5)
            results['nations'] = nations or []
            
            # Faith system
            religion_mgr = await self._get_religion_manager()
            faith_system = await religion_mgr.generate_complete_faith_system(run_ctx)
            results['religions'] = faith_system
            
            # Educational systems
            education_mgr = await self._get_education_manager()
            edu_systems = await education_mgr.generate_educational_systems(run_ctx)
            results['education'] = edu_systems
            
            # Initialize cultures for some nations
            for nation in (nations or [])[:3]:
                try:
                    culture_data = await self.initialize_nation_culture(run_ctx, nation['id'], language_count=2)
                    results['cultures'].append({'nation_id': nation['id'], 'culture': culture_data})
                except Exception:
                    continue
            
            results['status'] = 'complete'
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logger.error(f"quick_setup_world failed: {e}", exc_info=True)
        return results

    async def perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check across all systems."""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'systems': {},
            'issues': [],
            'recommendations': []
        }
        
        systems_to_check = [
            ('world_lore', self._world_lore_manager),
            ('education', self._education_manager),
            ('religion', self._religion_manager),
            ('politics', self._politics_manager),
            ('geopolitical', self._geopolitical_manager),
            ('local_lore', self._local_lore_manager),
            ('regional_culture', self._regional_culture_system),
        ]
        
        for name, sys_attr in systems_to_check:
            try:
                if sys_attr is None:
                    health_report['systems'][name] = {'status': 'not_loaded', 'healthy': True}
                    continue
                system = sys_attr
                sys_health = {'status': 'loaded', 'healthy': True}
                
                if hasattr(system, 'initialized'):
                    sys_health['initialized'] = bool(system.initialized)
                    if not system.initialized:
                        sys_health['healthy'] = False
                        health_report['issues'].append(f"{name} not initialized")
                
                # DB ping
                if hasattr(system, 'get_connection_pool'):
                    try:
                        pool = await system.get_connection_pool()
                        async with pool.acquire() as conn:
                            await conn.fetchval("SELECT 1")
                        sys_health['db_connection'] = 'ok'
                    except Exception as e:
                        sys_health['db_connection'] = 'failed'
                        sys_health['healthy'] = False
                        health_report['issues'].append(f"{name} DB ping failed: {e}")
                
                # Cache
                if hasattr(system, '_cache_stats'):
                    stats = getattr(system, '_cache_stats', {})
                    hits = float(stats.get('hits', 0))
                    misses = float(stats.get('misses', 0))
                    hit_rate = (hits / (hits + misses)) if (hits + misses) > 0 else 0.0
                    sys_health['cache_hit_rate'] = hit_rate
                    if hit_rate < 0.30:
                        health_report['recommendations'].append(f"Low cache hit rate for {name}: {hit_rate:.2%}")
                
                health_report['systems'][name] = sys_health
                if not sys_health['healthy'] and health_report['overall_status'] != 'unhealthy':
                    health_report['overall_status'] = 'degraded'
            
            except Exception as e:
                health_report['systems'][name] = {'status': 'error', 'healthy': False, 'error': str(e)}
                health_report['overall_status'] = 'unhealthy'
        
        # Canon consistency check
        try:
            canonical = await self.check_canonical_consistency()
            health_report['canonical_consistency'] = canonical
            if not canonical.get('is_consistent', True):
                health_report['overall_status'] = 'degraded'
                health_report['issues'].extend(canonical.get('violations', []))
        except Exception as e:
            health_report['canonical_consistency'] = {'error': str(e)}
            health_report['overall_status'] = 'degraded'
        
        return health_report
    
    async def get_scene_bundle_enhanced(self, scope: Any) -> Dict[str, Any]:
        """
        Enhanced scene bundle that integrates multiple subsystems on top of base bundle.
        Non-fatal on missing subsystem methods.
        """
        bundle = await self.get_scene_bundle(scope)  # keep your original
        
        # Add education (best-effort, generic search by tag/location name)
        try:
            if getattr(scope, 'location_id', None):
                # No direct API exists; best-effort: fetch top systems (if any) or leave empty
                bundle['data']['education'] = bundle['data'].get('education', [])
        except Exception as e:
            logger.debug(f"Education enrichment skipped: {e}")
        
        # Add geopolitical border disputes (best-effort general sample)
        try:
            if getattr(scope, 'nation_ids', None):
                # If you later add a real API, swap this block to call it
                from db.connection import get_db_connection_context
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch("""
                        SELECT id, region1_id, region2_id, dispute_type, severity, status
                        FROM BorderDisputes
                        ORDER BY severity DESC
                        LIMIT 5
                    """)
                if rows:
                    bundle['data'].setdefault('geopolitical', {})
                    bundle['data']['geopolitical']['sample_disputes'] = [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"Geopolitical enrichment skipped: {e}")
        
        # Add world events/relationships per tags/location
        try:
            world_mgr = await self._get_world_lore_manager()
            # Events by tags
            if getattr(scope, 'lore_tags', None):
                for tag in list(scope.lore_tags)[:3]:
                    events = await world_mgr.get_world_events(f"tag_{tag}")
                    if events:
                        bundle['data'].setdefault('world_events', [])
                        bundle['data']['world_events'].extend(events[:2])
            # Relationships by location
            loc_id = getattr(scope, 'location_id', None)
            if loc_id:
                rel = await world_mgr.get_world_relationships(f"location_{loc_id}")
                if rel:
                    bundle['data']['world_relationships'] = rel
        except Exception as e:
            logger.debug(f"World lore enrichment skipped: {e}")
        
        return bundle
    
    async def propagate_change_across_systems(
        self, 
        change_type: str,
        entity_type: str,
        entity_id: int,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propagate a canonical change across relevant systems and caches.
        Uses LoreSystem.propose_and_enact_change for durable updates.
        """
        from lore.core.lore_system import LoreSystem
        results = {
            'primary_change': {'entity_type': entity_type, 'entity_id': entity_id, 'changes': changes},
            'cascaded_changes': [],
            'affected_systems': []
        }
        lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
        ctx = self._create_mock_context()
        
        # Apply canonical change per entity type
        try:
            if entity_type == 'nation':
                await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="Nations",
                    entity_identifier={"id": entity_id},
                    updates=changes,
                    reason=f"{change_type} propagated via orchestrator"
                )
                results['affected_systems'].append('politics')
                results['affected_systems'].append('geopolitical')
                results['affected_systems'].append('regional_culture')
            
            elif entity_type == 'location':
                await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="Locations",
                    entity_identifier={"id": entity_id},
                    updates=changes,
                    reason=f"{change_type} propagated via orchestrator"
                )
                results['affected_systems'].append('local_lore')
                results['affected_systems'].append('world_lore')
            
            elif entity_type == 'religion':
                # Choose best table for your domain model; ReligiousConflicts/Religion/NationReligion etc.
                # Here, best-effort example for Religions
                await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="Religions",
                    entity_identifier={"id": entity_id},
                    updates=changes,
                    reason=f"{change_type} propagated via orchestrator"
                )
                results['affected_systems'].append('religion')
                results['affected_systems'].append('education')
            
            elif entity_type == 'conflict':
                await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NationalConflicts",
                    entity_identifier={"id": entity_id},
                    updates=changes,
                    reason=f"{change_type} propagated via orchestrator"
                )
                results['affected_systems'].append('politics')
                results['affected_systems'].append('geopolitical')
            
        except Exception as e:
            logger.error(f"Propagation apply failed for {entity_type}:{entity_id}: {e}")
        
        # Track + invalidate
        await self.record_lore_change(entity_type, entity_id, 'update', new_data=changes)
        scope_keys = self._get_affected_scope_keys(entity_type, entity_id, changes)
        await self._invalidate_scope_keys(scope_keys)
        
        # Invalidate world data best-effort (broad)
        try:
            world_mgr = await self._get_world_lore_manager()
            await world_mgr.invalidate_world_data(None, recursive=True)
            results['affected_systems'].append('world_lore')
        except Exception:
            pass
        
        return results

    async def create_unified_context(
        self, 
        location_id: Optional[int] = None,
        nation_ids: Optional[List[int]] = None,
        include_cultural: bool = True,
        include_religious: bool = True,
        include_political: bool = True
    ) -> Dict[str, Any]:
        """Create a unified context combining data from all subsystems."""
        ctx = self._create_mock_context()
        
        context = {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'timestamp': datetime.now().isoformat(),
        }
        
        # World foundation
        try:
            world_mgr = await self._get_world_lore_manager()
            context['world'] = await world_mgr.get_world_data('main')
        except Exception as e:
            context['world'] = None
            logger.debug(f"Unified context: world data unavailable: {e}")
        
        # Location-specific
        if location_id:
            try:
                context['location'] = await self._fetch_location_lore_for_bundle(location_id)
            except Exception as e:
                logger.debug(f"Unified context: location bundle fetch failed: {e}")
                context['location'] = {}
            
            # Local lore (structured)
            try:
                local_mgr = await self._get_local_lore_manager()
                from agents.run_context import RunContextWrapper
                run_ctx = RunContextWrapper(context={'user_id': self.user_id, 'conversation_id': self.conversation_id})
                context['local_lore'] = await local_mgr.get_location_lore(run_ctx, location_id)
            except Exception as e:
                logger.debug(f"Unified context: local lore fetch failed: {e}")
                context['local_lore'] = None
        
        # Nation-related
        if nation_ids:
            try:
                context['nations'] = await self._fetch_nations_for_bundle(nation_ids)
            except Exception as e:
                logger.debug(f"Unified context: nations fetch failed: {e}")
                context['nations'] = []
            
            if include_cultural:
                try:
                    context['cultural'] = await self._fetch_cultural_data_for_bundle(nation_ids)
                except Exception as e:
                    logger.debug(f"Unified context: cultural fetch failed: {e}")
                    context['cultural'] = {}
            
            if include_political:
                try:
                    politics_mgr = await self._get_politics_manager()
                    from agents.run_context import RunContextWrapper
                    run_ctx = RunContextWrapper(context={'user_id': self.user_id, 'conversation_id': self.conversation_id})
                    context['political_relations'] = []
                    for nid in nation_ids[:3]:
                        try:
                            # Reuse existing comprehensive method
                            rel = await politics_mgr.get_nation_politics(run_ctx, nid)
                            context['political_relations'].append(rel)
                        except Exception:
                            continue
                except Exception as e:
                    logger.debug(f"Unified context: political relations fetch failed: {e}")
                    context['political_relations'] = []
        
        # Religious (best-effort)
        if include_religious and location_id:
            try:
                context['religions'] = await self._fetch_religions_for_location(location_id)
            except Exception as e:
                logger.debug(f"Unified context: religion fetch failed: {e}")
                context['religions'] = []
        
        return context

    async def get_scene_brief(self, scope) -> Dict[str, Any]:
        brief: Dict[str, Any] = {"anchors": {}, "signals": {}, "links": {}}
        try:
            brief["anchors"]["location_id"] = getattr(scope, "location_id", None)

            # canonical rules (best-effort & time-boxed by caller as well)
            try:
                rules: List[str] = await asyncio.wait_for(self._get_canonical_rules_for_scope(scope), timeout=0.35)
                brief["signals"]["canonical_rules"] = list(rules or [])[:3]
            except Exception:
                pass

            # quick nation from location
            loc_id = getattr(scope, "location_id", None)
            if isinstance(loc_id, int):
                data = await self.get_location_context(loc_id)
                nid = (data or {}).get("nation_id")
                if nid is not None:
                    try:
                        brief["anchors"]["nation_ids"] = [int(nid)]
                    except Exception:
                        pass

            # surface lore tags if scope already has them
            lore_tags = list(getattr(scope, "lore_tags", []) or [])
            if lore_tags:
                brief["anchors"]["lore_tags"] = lore_tags[:5]
        except Exception:
            pass
        return brief

    async def apply_matriarchal_lens_to_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally rewrite parts of the bundle to reflect matriarchal themes more strongly.
        This is intentionally conservative: only the most visible textual fields are touched.
        """
        if not self.config.enable_matriarchal_theme:
            return bundle
    
        try:
            mpf = await self._get_matriarchal_power_framework()
        except Exception as e:
            logger.debug(f"Could not initialize matriarchal framework: {e}")
            return bundle
    
        # Carefully transform just the most visible textual fields
        foundation: Dict[str, Any] = {}
    
        # Location description
        try:
            loc_desc = (
                bundle.get("data", {})
                      .get("location", {})
                      .get("description")
            )
            if loc_desc and isinstance(loc_desc, str) and loc_desc.strip():
                foundation["location_description"] = loc_desc
        except Exception:
            pass
    
        # Optionally, add other small, self-contained text fields here in future
    
        if not foundation:
            return bundle  # Nothing to transform
    
        try:
            transformed = await mpf.apply_power_lens(foundation)
        except Exception as e:
            logger.debug(f"Matriarchal lens application error: {e}")
            return bundle
    
        # Apply back to bundle
        try:
            if "location_description" in transformed:
                bundle["data"]["location"]["description"] = transformed["location_description"]
        except Exception:
            pass
    
        return bundle

    async def _track_lore_dynamics_change(
        self,
        change_type: str,
        change_data: Dict[str, Any]
    ) -> None:
        """Track changes from the LoreDynamicsSystem."""
        # Map dynamics change types to standard element types
        element_type_map = {
            'myth_evolution': 'myth',
            'culture_development': 'cultural_element',
            'geopolitical_shift': 'faction',
            'reputation_change': 'notable_figure',
            'emergent_event': 'event'
        }
        
        element_type = element_type_map.get(change_type, 'world_lore')
        
        # Extract relevant IDs and scope keys
        element_id = change_data.get('id', 0)
        scope_keys = self._get_affected_scope_keys(
            element_type, element_id, change_data
        )
        
        await self._track_element_change(
            element_type,
            element_id,
            change_data,
            scope_keys
        )
    
    async def _track_cultural_change(
        self,
        nation_id: int,
        change_type: str,
        change_data: Dict[str, Any]
    ) -> None:
        """Track changes from the RegionalCultureSystem."""
        # Map cultural change types
        element_type_map = {
            'language': 'language',
            'dialect': 'dialect',
            'norm': 'cultural_norm',
            'etiquette': 'etiquette',
            'diffusion': 'cultural_exchange'
        }
        
        element_type = element_type_map.get(change_type, 'cultural_element')
        
        # Add nation context to change data
        change_data['nation_id'] = nation_id
        
        # Generate scope keys that include the nation
        scope_keys = self._get_affected_scope_keys(
            element_type,
            change_data.get('id', 0),
            change_data
        )
        
        await self._track_element_change(
            element_type,
            change_data.get('id', 0),
            change_data,
            scope_keys
        )

    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated systems."""
        status = {
            "orchestrator": {
                "initialized": self.initialized,
                "metrics": self.metrics,
                "cache_status": {
                    "scene_bundles": len(self._scene_bundle_cache),
                    "change_log_entries": sum(len(log) for log in self._change_log.values()),
                    "element_snapshots": len(self._element_snapshots),
                    "bundle_last_changed": len(self._bundle_last_changed),
                },
            },
            "systems": {},
            "cache_analytics": {},
            "resource_limits": {},
        }
    
        # LoreCache analytics (optional)
        try:
            analytics = await GLOBAL_LORE_CACHE.get_cache_analytics()
            status["cache_analytics"] = {
                "hit_rate": analytics.get("hit_rate", 0),
                "size_bytes": analytics.get("size_bytes", 0),
                "keys_count": analytics.get("keys_count", 0),
                "avg_access_time_ms": analytics.get("avg_access_time_ms", 0),
            }
        except Exception as e:
            logger.debug(f"Could not get cache analytics: {e}")
            status["cache_analytics"] = {"error": str(e)}
    
        # Check all system attributes
        attr_names = [
            # Core systems
            ("lore_system", "_lore_system"),
            ("canon_module", "_canon_module"),
            ("cache_system", "_cache_system"),
            ("registry_system", "_registry_system"),
            ("canon_validation", "_canon_validation"),
            
            # Manager systems
            ("education", "_education_manager"),
            ("religion", "_religion_manager"),
            ("local_lore", "_local_lore_manager"),
            ("geopolitical", "_geopolitical_manager"),
            ("politics", "_politics_manager"),
            ("world_lore", "_world_lore_manager"),
            
            # Specialized systems
            ("lore_dynamics", "_lore_dynamics_system"),
            ("regional_culture", "_regional_culture_system"),
            ("matriarchal_framework", "_mpf"),  # Fixed: using actual framework
            
            # Integration components
            ("npc_integration", "_npc_integration"),
            ("conflict_integration", "_conflict_integration"),
            ("context_enhancer", "_context_enhancer"),
            ("lore_generator", "_lore_generator"),
            ("dynamic_generator", "_dynamic_generator"),
            
            # Support systems
            ("master_coordinator", "_master_coordinator"),
            ("content_validator", "_content_validator"),
            ("relationship_mapper", "_relationship_mapper"),
            ("unified_trace", "_unified_trace_system"),
            
            # Data access layers
            ("npc_data_access", "_npc_data_access"),
            ("location_data_access", "_location_data_access"),
            ("faction_data_access", "_faction_data_access"),
            ("knowledge_access", "_knowledge_access"),
            
            # Additional systems (if they exist)
            ("lore_update_system", "_lore_update_system"),
            ("national_conflict_system", "_national_conflict_system"),
            ("religious_distribution_system", "_religious_distribution_system"),
        ]
    
        for name, attr in attr_names:
            if hasattr(self, attr):
                system = getattr(self, attr)
                if system is not None:
                    system_status = {
                        "loaded": True,
                        "type": type(system).__name__,
                    }
                    
                    # Check initialization status
                    if hasattr(system, "initialized"):
                        system_status["initialized"] = system.initialized
                    elif hasattr(system, "is_initialized") and callable(system.is_initialized):
                        system_status["initialized"] = system.is_initialized()
                    else:
                        system_status["initialized"] = True  # Assume initialized if no flag
                    
                    # Check governance status
                    if hasattr(system, "governor"):
                        system_status["has_governance"] = system.governor is not None
                    
                    # Check for health check capability
                    if hasattr(system, "health_check"):
                        system_status["health_check_available"] = True
                    
                    # Get metrics if available
                    if hasattr(system, "metrics"):
                        system_status["metrics"] = getattr(system, "metrics", {})
                    
                    status["systems"][name] = system_status
                else:
                    status["systems"][name] = {"loaded": False, "reason": "Attribute exists but is None"}
            else:
                status["systems"][name] = {"loaded": False, "reason": "Attribute not found"}
    
        # Calculate overall health
        total_systems = len(attr_names)
        loaded_systems = sum(1 for s in status["systems"].values() if s.get("loaded", False))
        initialized_systems = sum(1 for s in status["systems"].values() 
                                 if s.get("loaded", False) and s.get("initialized", False))
        
        status["summary"] = {
            "total_systems": total_systems,
            "loaded_systems": loaded_systems,
            "initialized_systems": initialized_systems,
            "health_percentage": (initialized_systems / total_systems * 100) if total_systems > 0 else 0,
            "status": "healthy" if initialized_systems >= total_systems * 0.8 else 
                      "degraded" if initialized_systems >= total_systems * 0.5 else "unhealthy"
        }
    
        # Check resource limits if configured
        if self.config.resource_limits:
            status["resource_limits"] = {
                "max_parallel_operations": self.config.max_parallel_operations,
                "cache_ttl": self.config.cache_ttl,
                "cache_max_size": self.config.cache_max_size,
                "bundle_cache_max_size": self.config.bundle_cache_max_size,
                "bundle_ttl": self.config.bundle_ttl,
            }
            
            # Check if we're near limits
            if len(self._scene_bundle_cache) > self.config.bundle_cache_max_size * 0.9:
                status["warnings"] = status.get("warnings", [])
                status["warnings"].append(f"Bundle cache near limit: {len(self._scene_bundle_cache)}/{self.config.bundle_cache_max_size}")
    
        # Add timestamp
        status["checked_at"] = datetime.now().isoformat()
        
        # Add operational metrics
        status["operational_metrics"] = {
            "total_operations": self.metrics.get("operations", 0),
            "cache_hits": self.metrics.get("cache_hits", 0),
            "cache_misses": self.metrics.get("cache_misses", 0),
            "bundle_hits": self.metrics.get("bundle_hits", 0),
            "bundle_misses": self.metrics.get("bundle_misses", 0),
            "db_roundtrips": self.metrics.get("db_roundtrips", 0),
            "errors": self.metrics.get("errors", 0),
            "last_operation": self.metrics.get("last_operation"),
        }
        
        # Calculate cache hit rate
        total_cache_ops = status["operational_metrics"]["cache_hits"] + status["operational_metrics"]["cache_misses"]
        if total_cache_ops > 0:
            status["operational_metrics"]["cache_hit_rate"] = (
                status["operational_metrics"]["cache_hits"] / total_cache_ops * 100
            )
        
        total_bundle_ops = status["operational_metrics"]["bundle_hits"] + status["operational_metrics"]["bundle_misses"]
        if total_bundle_ops > 0:
            status["operational_metrics"]["bundle_hit_rate"] = (
                status["operational_metrics"]["bundle_hits"] / total_bundle_ops * 100
            )
    
        return status

    async def culture_simulate_exchange(
        self, ctx,
        nation1_id: int,
        nation2_id: int,
        exchange_type: str = "full",
        years: int = 50
    ) -> Dict[str, Any]:
        """
        Simulate comprehensive cultural exchange between nations.
        
        Args:
            ctx: Context object
            nation1_id: First nation ID
            nation2_id: Second nation ID
            exchange_type: Type of exchange (full, language, customs, cuisine, etc.)
            years: Years to simulate
            
        Returns:
            Exchange results and impacts
        """
        rcs = await self._get_regional_culture_system()
        
        if exchange_type == "full":
            # Full cultural diffusion
            result = await rcs.simulate_cultural_diffusion(ctx, nation1_id, nation2_id, years)
        else:
            # Specific exchange type
            result = await rcs._apply_specific_diffusion(nation1_id, nation2_id, exchange_type, years)
        
        # Track the exchange
        if self._change_tracking_enabled:
            await self._track_cultural_change(
                nation1_id,
                'diffusion',
                {
                    'type': exchange_type,
                    'partner_nation': nation2_id,
                    'years': years,
                    'results': result
                }
            )
            await self._track_cultural_change(
                nation2_id,
                'diffusion',
                {
                    'type': exchange_type,
                    'partner_nation': nation1_id,
                    'years': years,
                    'results': result
                }
            )
        
        # Invalidate scene caches for both nations
        nation1_keys = self._get_affected_scope_keys('nation', nation1_id, {'nation_id': nation1_id})
        nation2_keys = self._get_affected_scope_keys('nation', nation2_id, {'nation_id': nation2_id})
        await self._invalidate_scope_keys(nation1_keys + nation2_keys)
        
        return result
    
    async def culture_get_comprehensive_summary(
        self, ctx,
        nation_id: int,
        include_comparisons: bool = False,
        comparison_nations: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Get a comprehensive cultural summary with optional comparisons.
        
        Args:
            ctx: Context object
            nation_id: Nation to summarize
            include_comparisons: Whether to include comparisons
            comparison_nations: Nations to compare with
            
        Returns:
            Comprehensive cultural summary
        """
        rcs = await self._get_regional_culture_system()
        
        # Get base cultural data
        culture = await rcs.get_nation_culture(ctx, nation_id)
        
        # Generate narrative summary
        summary = await rcs.summarize_culture(nation_id, format_type="detailed")
        
        result = {
            'nation_id': nation_id,
            'culture_data': culture,
            'narrative_summary': summary
        }
        
        # Add comparisons if requested
        if include_comparisons and comparison_nations:
            comparisons = []
            for other_nation_id in comparison_nations[:3]:  # Limit comparisons
                conflict_analysis = await rcs.detect_cultural_conflicts(
                    nation_id, other_nation_id
                )
                comparisons.append({
                    'nation_id': other_nation_id,
                    'conflicts': conflict_analysis
                })
            result['comparisons'] = comparisons
        
        return result
        
    async def _fetch_cultural_data_for_bundle(self, nation_ids: List[int]) -> Dict[str, Any]:
        """Enhanced cultural data fetching for scene bundles."""
        try:
            data = {'languages': [], 'norms': [], 'etiquette': [], 'customs': [], 'dialects': [], 'exchanges': []}
    
            async with get_db_connection_context() as conn:
                # Your original efficient queries - keep these!
                # Languages linked to nation_ids (primary or minority)
                langs = await conn.fetch("""
                    SELECT l.id, l.name, l.language_family, l.writing_system,
                           l.difficulty, l.relation_to_power, l.formality_levels,
                           l.common_phrases, l.dialects,
                           array_agg(DISTINCT ld.region_id) FILTER (WHERE ld.id IS NOT NULL) as dialect_regions,
                           array_agg(DISTINCT ld.name) FILTER (WHERE ld.id IS NOT NULL) as dialect_names
                    FROM Languages l
                    LEFT JOIN LanguageDialects ld ON l.id = ld.language_id
                    WHERE EXISTS (
                        SELECT 1 FROM unnest($1::int[]) AS nid
                        WHERE nid = ANY(l.primary_regions) OR nid = ANY(l.minority_regions)
                    )
                    GROUP BY l.id
                    LIMIT 10
                """, nation_ids[:10])
    
                for r in langs:
                    lang_data = {
                        'id': r['id'],
                        'name': r['name'],
                        'family': r['language_family'],
                        'writing': r['writing_system'],
                        'difficulty': r['difficulty'],
                        'power_relation': r['relation_to_power'],
                        'formality_levels': r['formality_levels'],
                        'has_dialects': bool(r['dialect_names'])
                    }
                    
                    # Add common phrases sample (limit size)
                    if r['common_phrases']:
                        phrases = json.loads(r['common_phrases']) if isinstance(r['common_phrases'], str) else r['common_phrases']
                        lang_data['sample_phrases'] = dict(list(phrases.items())[:3])
                    
                    data['languages'].append(lang_data)
    
                # Cultural norms for those nations - enhanced query
                norms = await conn.fetch("""
                    SELECT cn.id, cn.category, cn.description, cn.taboo_level, cn.gender_specific,
                           cn.female_variation, cn.male_variation, cn.consequence,
                           cn.regional_variations,
                           COALESCE(n.name, n.nation_name) AS nation_name,
                           n.matriarchy_level
                    FROM CulturalNorms cn
                    JOIN Nations n ON cn.nation_id = COALESCE(n.id, n.nation_id)
                    WHERE cn.nation_id = ANY($1::int[])
                    ORDER BY cn.taboo_level DESC, cn.category
                    LIMIT 15
                """, nation_ids[:5])
    
                for r in norms:
                    norm_data = {
                        'id': r['id'],
                        'category': r['category'],
                        'description': (r['description'] or '')[:150],
                        'taboo_level': r['taboo_level'],
                        'gender_specific': r['gender_specific'],
                        'nation': r['nation_name'],
                        'matriarchy_level': r['matriarchy_level']
                    }
                    
                    # Include gender variations if they exist
                    if r['gender_specific']:
                        norm_data['female_variation'] = (r['female_variation'] or '')[:100]
                        norm_data['male_variation'] = (r['male_variation'] or '')[:100]
                    
                    # Include consequence for high taboo items
                    if r['taboo_level'] >= 7:
                        norm_data['consequence'] = (r['consequence'] or 'Social sanctions')
                        
                    data['norms'].append(norm_data)
    
                # Etiquette for those nations - enhanced
                etqs = await conn.fetch("""
                    SELECT e.id, e.context, e.greeting_ritual, e.power_display, 
                           e.gender_distinctions, e.title_system, e.gift_giving,
                           e.dining_etiquette, e.taboos,
                           COALESCE(n.name, n.nation_name) AS nation_name
                    FROM Etiquette e
                    JOIN Nations n ON e.nation_id = COALESCE(n.id, n.nation_id)
                    WHERE e.nation_id = ANY($1::int[])
                    ORDER BY 
                        CASE e.context 
                            WHEN 'court' THEN 1
                            WHEN 'diplomatic' THEN 2
                            WHEN 'public' THEN 3
                            ELSE 4
                        END
                    LIMIT 10
                """, nation_ids[:5])
    
                for r in etqs:
                    data['etiquette'].append({
                        'id': r['id'],
                        'context': r['context'],
                        'nation': r['nation_name'],
                        'greeting': (r['greeting_ritual'] or '')[:100],
                        'power_display': (r['power_display'] or '')[:100],
                        'gender_rules': r['gender_distinctions'],
                        'titles': (r['title_system'] or '')[:50],
                        'gifts': (r['gift_giving'] or '')[:50],
                        'dining': (r['dining_etiquette'] or '')[:50],
                        'major_taboos': r['taboos'][:3] if r['taboos'] else []
                    })
    
                # NEW: Social customs
                customs = await conn.fetch("""
                    SELECT sc.id, sc.name, sc.description, sc.context,
                           sc.formality_level, sc.gender_rules,
                           sc.violation_consequences,
                           n.name as origin_nation
                    FROM SocialCustoms sc
                    JOIN Nations n ON sc.nation_origin = n.id
                    WHERE sc.nation_origin = ANY($1::int[])
                       OR $1::int[] && sc.adopted_by
                    LIMIT 10
                """, nation_ids[:5])
    
                for r in customs:
                    data['customs'].append({
                        'id': r['id'],
                        'name': r['name'],
                        'description': (r['description'] or '')[:100],
                        'context': r['context'],
                        'formality': r['formality_level'],
                        'origin': r['origin_nation']
                    })
    
                # NEW: Active dialects in the region
                dialects = await conn.fetch("""
                    SELECT ld.id, ld.name, ld.region_id, ld.prestige_level,
                           ld.social_context, l.name as parent_language,
                           n.name as region_name
                    FROM LanguageDialects ld
                    JOIN Languages l ON ld.language_id = l.id
                    JOIN Nations n ON ld.region_id = n.id
                    WHERE ld.region_id = ANY($1::int[])
                    ORDER BY ld.prestige_level DESC
                    LIMIT 5
                """, nation_ids[:3])
    
                for r in dialects:
                    data['dialects'].append({
                        'id': r['id'],
                        'name': r['name'],
                        'parent_language': r['parent_language'],
                        'region': r['region_name'],
                        'prestige': r['prestige_level'],
                        'context': r['social_context']
                    })
    
                # NEW: Recent cultural exchanges
                if len(nation_ids) >= 2:
                    exchanges = await conn.fetch("""
                        SELECT ce.exchange_type, ce.impact_level,
                               ce.cultural_resistance, ce.exchange_details,
                               n1.name as nation1_name, n2.name as nation2_name
                        FROM CulturalExchanges ce
                        JOIN Nations n1 ON ce.nation1_id = n1.id
                        JOIN Nations n2 ON ce.nation2_id = n2.id
                        WHERE (ce.nation1_id = ANY($1::int[]) AND ce.nation2_id = ANY($1::int[]))
                           OR (ce.nation2_id = ANY($1::int[]) AND ce.nation1_id = ANY($1::int[]))
                        ORDER BY ce.timestamp DESC
                        LIMIT 5
                    """, nation_ids[:5])
    
                    for r in exchanges:
                        exchange_summary = {
                            'type': r['exchange_type'],
                            'between': [r['nation1_name'], r['nation2_name']],
                            'impact': r['impact_level'],
                            'resistance': r['cultural_resistance']
                        }
                        
                        # Extract key elements from exchange details
                        if r['exchange_details']:
                            details = json.loads(r['exchange_details']) if isinstance(r['exchange_details'], str) else r['exchange_details']
                            if 'vocabulary' in details:
                                exchange_summary['vocabulary_exchanged'] = len(details['vocabulary'])
                            if 'customs' in details:
                                exchange_summary['customs_shared'] = len(details.get('customs', []))
                        
                        data['exchanges'].append(exchange_summary)
    
                # Cultural conflict detection (if multiple nations)
                if len(nation_ids) >= 2:
                    # Quick conflict check using direct SQL
                    conflict_check = await conn.fetchrow("""
                        WITH norm_conflicts AS (
                            SELECT 
                                cn1.category,
                                cn1.description as norm1,
                                cn2.description as norm2,
                                ABS(cn1.taboo_level - cn2.taboo_level) as taboo_diff,
                                cn1.nation_id as nation1_id,
                                cn2.nation_id as nation2_id
                            FROM CulturalNorms cn1
                            CROSS JOIN CulturalNorms cn2
                            WHERE cn1.nation_id = $1 
                              AND cn2.nation_id = $2
                              AND cn1.category = cn2.category
                              AND ABS(cn1.taboo_level - cn2.taboo_level) > 5
                        )
                        SELECT COUNT(*) as conflict_count,
                               MAX(taboo_diff) as max_difference
                        FROM norm_conflicts
                    """, nation_ids[0], nation_ids[1])
                    
                    if conflict_check and conflict_check['conflict_count'] > 0:
                        data['conflict_indicators'] = {
                            'norm_conflicts': conflict_check['conflict_count'],
                            'max_taboo_difference': conflict_check['max_difference'],
                            'risk_level': 'high' if conflict_check['max_difference'] > 7 else 'medium'
                        }
    
            # Get RegionalCultureSystem for advanced analysis only if needed
            if len(nation_ids) >= 2 and not data.get('conflict_indicators'):
                try:
                    rcs = await self._get_regional_culture_system()
                    conflict_analysis = await rcs.detect_cultural_conflicts(
                        nation_ids[0], nation_ids[1]
                    )
                    if conflict_analysis and "error" not in conflict_analysis:
                        data['detailed_conflicts'] = conflict_analysis.get('potential_conflicts', [])[:3]
                except Exception as e:
                    logger.debug(f"Could not perform detailed conflict analysis: {e}")
    
            return data
            
        except Exception as e:
            logger.debug(f"Could not fetch cultural data: {e}")
            return {'languages': [], 'norms': [], 'etiquette': [], 'customs': [], 
                    'dialects': [], 'exchanges': []}
    
    async def get_scene_delta(self, scope: Any, since_ts: float) -> Dict[str, Any]:
        """
        Get only the changes to lore since a given timestamp.
        
        Args:
            scope: SceneScope for filtering
            since_ts: Timestamp to get changes since
            
        Returns:
            Dictionary with:
                - section: 'lore' (for consistent merging)
                - anchors: IDs of core entities in the bundle
                - data: Changed lore data
                - canonical: Boolean indicating if contains canonical changes
                - last_changed_at: Timestamp of last change
                - version: Version string
        """
        cache_key = self._generate_scene_cache_key(scope)
        
        # First check if we have any tracked changes
        changes = await self._get_changes_since(cache_key, since_ts)
        
        if not changes:
            # Check if we have a cached timestamp for this scope
            if cache_key in self._bundle_last_changed:
                last_changed = self._bundle_last_changed[cache_key]
                if last_changed <= since_ts:
                    # No changes since requested time
                    self.metrics['bundle_hits'] = self.metrics.get('bundle_hits', 0) + 1
                    return {
                        'section': 'lore',
                        'anchors': {
                            'location_id': getattr(scope, 'location_id', None),
                            'nation_ids': sorted(list(getattr(scope, 'nation_ids', set())))[:5],
                            'conflict_ids': sorted(list(getattr(scope, 'conflict_ids', set())))[:5],
                        },
                        'data': {},
                        'canonical': False,
                        'last_changed_at': last_changed,
                        'version': f"lore_delta_{cache_key[:8]}_{int(last_changed)}"
                    }
            
            # No change tracking available - return full bundle
            self.metrics['bundle_misses'] = self.metrics.get('bundle_misses', 0) + 1
            return await self.get_scene_bundle(scope)
        
        self.metrics['bundle_hits'] = self.metrics.get('bundle_hits', 0) + 1
        
        # Build delta bundle from changes
        delta_data = SceneBundleData()
        canonical_changed = False
        
        for change in changes:
            element_type = change['element_type']
            element_id = change['element_id']
            operation = change['operation']
            new_value = change.get('new_value', {})
            changed_fields = change.get('changed_fields', [])
            
            # Process changes by element type
            if element_type == 'location':
                if operation != 'delete':
                    delta_data.location.update(new_value)
                    if 'canonical_rules' in changed_fields:
                        canonical_changed = True
                        rules = new_value.get('canonical_rules', [])
                        delta_data.canonical_rules.extend([
                            {"text": rule, "source": "location"}
                            for rule in rules
                        ])
                        
            elif element_type == 'nation':
                # Find and update or add nation
                nation_found = False
                for nation in delta_data.nations:
                    if nation.get('id') == element_id:
                        nation.update(new_value)
                        nation['_operation'] = operation
                        nation_found = True
                        break
                
                if not nation_found and operation != 'delete':
                    nation_entry = {
                        'id': element_id,
                        '_operation': operation,
                        **new_value
                    }
                    delta_data.nations.append(nation_entry)
                canonical_changed = True  # Nations are canonical
                
            elif element_type == 'religion':
                religion_found = False
                for religion in delta_data.religions:
                    if religion.get('id') == element_id:
                        religion.update(new_value)
                        religion['_operation'] = operation
                        religion_found = True
                        break
                
                if not religion_found and operation != 'delete':
                    religion_entry = {
                        'id': element_id,
                        '_operation': operation,
                        **new_value
                    }
                    delta_data.religions.append(religion_entry)
                    
            elif element_type == 'myth':
                myth_found = False
                for myth in delta_data.myths:
                    if myth.get('id') == element_id:
                        myth.update(new_value)
                        myth['_operation'] = operation
                        myth_found = True
                        break
                
                if not myth_found and operation != 'delete':
                    myth_entry = {
                        'id': element_id,
                        '_operation': operation,
                        **new_value
                    }
                    delta_data.myths.append(myth_entry)
                    
            elif element_type == 'conflict':
                conflict_found = False
                for conflict in delta_data.conflicts:
                    if conflict.get('id') == element_id:
                        conflict.update(new_value)
                        conflict['_operation'] = operation
                        conflict_found = True
                        break
                
                if not conflict_found and operation != 'delete':
                    conflict_entry = {
                        'id': element_id,
                        '_operation': operation,
                        **new_value
                    }
                    delta_data.conflicts.append(conflict_entry)
            
            elif element_type == 'world_lore':
                # Update world lore section
                delta_data.world.update(new_value)
                if 'canonical' in new_value:
                    canonical_changed = True
        
        # Get latest timestamp from changes
        latest_timestamp = max(
            change.get('timestamp', since_ts) for change in changes
        )
        
        # Build result with consistent structure
        result = {
            'section': 'lore',  # Consistent with other bundles
            'anchors': {
                'location_id': getattr(scope, 'location_id', None),
                'nation_ids': sorted(list(getattr(scope, 'nation_ids', set())))[:5],
                'conflict_ids': sorted(list(getattr(scope, 'conflict_ids', set())))[:5],
            },
            'data': delta_data.to_dict(),
            'canonical': canonical_changed,  # Changed from 'canon'
            'last_changed_at': latest_timestamp,
            'version': f"lore_delta_{cache_key[:8]}_{int(latest_timestamp)}",
            'is_delta': True,
            'change_summary': {
                'total_changes': len(changes),
                'creates': len([c for c in changes if c['operation'] == 'create']),
                'updates': len([c for c in changes if c['operation'] == 'update']),
                'deletes': len([c for c in changes if c['operation'] == 'delete']),
                'element_types': list(set(c['element_type'] for c in changes))
            }
        }
        
        # Update our tracking
        self._bundle_last_changed[cache_key] = latest_timestamp
        
        return result
    
    # ===== CHANGE TRACKING METHODS =====
    
    async def _track_element_change(
        self, 
        element_type: str, 
        element_id: int, 
        new_data: Dict[str, Any],
        scope_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Track changes to a lore element.
        
        Args:
            element_type: Type of element (location, nation, myth, etc.)
            element_id: ID of the element
            new_data: New data for the element
            scope_keys: Optional list of scope keys this change affects
            
        Returns:
            True if changes were detected and recorded
        """
        if not self._change_tracking_enabled:
            return False
        
        element_key = f"{element_type}:{element_id}"
        old_data = self._element_snapshots.get(element_key)
        
        # Detect what changed
        operation = 'create' if old_data is None else 'update'
        changed_fields = []
        
        if old_data:
            # Compare fields
            all_fields = set(old_data.keys()) | set(new_data.keys())
            for field in all_fields:
                old_val = old_data.get(field)
                new_val = new_data.get(field)
                if old_val != new_val:
                    changed_fields.append(field)
        else:
            # New element
            changed_fields = list(new_data.keys())
        
        if not changed_fields and operation == 'update':
            # No actual changes
            return False
        
        # Record the change
        change_record = {
            'element_type': element_type,
            'element_id': element_id,
            'operation': operation,
            'changed_fields': changed_fields,
            'old_value': {k: old_data.get(k) for k in changed_fields if old_data} if old_data else None,
            'new_value': {k: new_data.get(k) for k in changed_fields if k in new_data},
            'timestamp': time.time()
        }
        
        # Store in database
        await self._persist_change(change_record, scope_keys)
        
        # Update snapshot
        self._element_snapshots[element_key] = new_data.copy()
        
        # Update in-memory change log for affected scopes
        if scope_keys:
            for scope_key in scope_keys:
                if scope_key not in self._change_log:
                    self._change_log[scope_key] = []
                self._change_log[scope_key].append(change_record)
                
                # Limit in-memory log size
                if len(self._change_log[scope_key]) > 100:
                    self._change_log[scope_key] = self._change_log[scope_key][-50:]
        
        return True
    
    async def _persist_change(
        self, 
        change_record: Dict[str, Any], 
        scope_keys: Optional[List[str]] = None
    ) -> None:
        """Persist a change record to the database."""
        try:
            async with get_db_connection_context() as conn:
                # Pass Python dicts directly - asyncpg handles JSONB conversion
                await conn.execute("""
                    INSERT INTO LoreChangeLog (
                        conversation_id, element_type, element_id,
                        operation, changed_fields, old_value, new_value,
                        scope_keys, timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    self.conversation_id,
                    change_record['element_type'],
                    change_record['element_id'],
                    change_record['operation'],
                    change_record['changed_fields'],  # Direct list - asyncpg handles JSONB
                    change_record.get('old_value'),   # Direct dict - asyncpg handles JSONB
                    change_record.get('new_value'),   # Direct dict - asyncpg handles JSONB
                    scope_keys or [],
                    datetime.fromtimestamp(change_record['timestamp'])
                )
        except Exception as e:
            logger.warning(f"Failed to persist change record: {e}")

    async def _get_pool(self):
        """Get database connection pool."""
        if not hasattr(self, '_pool'):
            self._pool = await asyncpg.create_pool(dsn=DB_DSN)
        return self._pool

    async def get_location_context(self, location_ref: Union[int, str]) -> Dict[str, Any]:
        try:
            async with get_db_connection_context() as conn:
                if isinstance(location_ref, int):
                    row = await conn.fetchrow("""
                        SELECT 
                            COALESCE(id, location_id) AS id,
                            COALESCE(location_name, name) AS name,
                            description, nation_id, governance, culture, population,
                            canonical_rules, tags
                        FROM Locations
                        WHERE COALESCE(id, location_id) = $1
                        LIMIT 1
                    """, location_ref)
                else:
                    row = await conn.fetchrow("""
                        SELECT 
                            COALESCE(id, location_id) AS id,
                            COALESCE(location_name, name) AS name,
                            description, nation_id, governance, culture, population,
                            canonical_rules, tags
                        FROM Locations
                        WHERE COALESCE(location_name, name) = $1
                        LIMIT 1
                    """, location_ref)
    
                if not row:
                    return {}
    
                data = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': (row['description'] or '')[:200],
                    'nation_id': row['nation_id'],
                    'governance': row['governance'] or {},
                    'culture': row['culture'] or {},
                    'population': row['population'] or 0,
                    'canonical_rules': row['canonical_rules'] or [],
                    'tags': row['tags'] or []
                }
    
                landmarks = await conn.fetch("""
                    SELECT landmark_id AS id, name, significance
                    FROM Landmarks
                    WHERE location_id = $1
                    ORDER BY significance DESC
                    LIMIT 3
                """, row['id'])
                if landmarks:
                    data['landmarks'] = [
                        {'id': lm['id'], 'name': lm['name'], 'significance': lm['significance']}
                        for lm in landmarks
                    ]
    
                events = await conn.fetch("""
                    SELECT event_name AS name, description, event_date
                    FROM Events
                    WHERE location = $1
                    ORDER BY event_date DESC
                    LIMIT 2
                """, row['name'])
                if events:
                    data['recent_events'] = [
                        {
                            'name': e['name'],
                            'description': (e['description'] or '')[:100],
                            'date': e['event_date'].isoformat() if e['event_date'] else None
                        }
                        for e in events
                    ]
    
                return data
    
        except Exception as e:
            logger.debug(f"get_location_context failed for '{location_ref}': {e}")
            return {}


    async def dynamics_execute_plan_step(self, ctx, plan_id: str, step_index: int) -> Dict[str, Any]:
        from lore.systems.dynamics import MultiStepPlanner
        dynamics = await self._get_lore_dynamics_system()
        planner = MultiStepPlanner(dynamics)
        context = {"user_id": self.user_id, "conversation_id": self.conversation_id}
        result = await planner.execute_plan_step(plan_id, step_index, context)
        if self._change_tracking_enabled and result.get('applied_changes'):
            for change in result['applied_changes']:
                await self.record_lore_change(change.get('element_type', 'world_lore'),
                                              change.get('element_id', 0), 'update', new_data=change)
        return result

    async def dynamics_create_evolution_plan_noctx(self, initial_prompt: str) -> Dict[str, Any]:
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        return await self.dynamics_create_evolution_plan(ctx, initial_prompt)
    
    async def dynamics_execute_plan_step_noctx(self, plan_id: str, step_index: int) -> Dict[str, Any]:
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        return await self.dynamics_execute_plan_step(ctx, plan_id, step_index)
    
    async def dynamics_evaluate_narrative_noctx(self, narrative_element: Dict[str, Any], element_type: str) -> Dict[str, Any]:
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        return await self.dynamics_evaluate_narrative(ctx, narrative_element, element_type)
    
    async def dynamics_evolve_narrative_element(
        self, ctx,
        element_type: str,
        initial_element: Optional[Dict[str, Any]] = None,
        generations: int = 3
    ) -> Dict[str, Any]:
        """
        Evolve a narrative element through multiple generations.
        
        Args:
            ctx: Context object
            element_type: Type of element to evolve
            initial_element: Optional seed element
            generations: Number of evolution generations
            
        Returns:
            Evolved narrative element
        """
        from lore.systems.dynamics import NarrativeEvolutionSystem
        dynamics = await self._get_lore_dynamics_system()
        evolution_system = NarrativeEvolutionSystem(dynamics)
        
        context = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "world_state": await dynamics._fetch_world_state()
        }
        
        result = await evolution_system.evolve_narrative_element(
            element_type, context, initial_element, generations
        )
        
        # Record the evolution
        if self._change_tracking_enabled and not result.get('error'):
            await self.record_lore_change(
                element_type,
                result.get('id', 0),
                'evolve',
                new_data=result
            )
        
        return result
    
    async def dynamics_stream_world_changes_with_ctx(
        self,
        ctx,
        event_data: Dict[str, Any],
        affected_elements: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Wrapper that accepts ctx and forwards to the canonical streamer."""
        async for chunk in self.dynamics_stream_world_changes(event_data, affected_elements):
            yield chunk

    
    async def analyze_setting_and_generate_orgs(self) -> Dict[str, Any]:
        from lore.setting_analyzer import SettingAnalyzer
        analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
        await analyzer.initialize_governance()
        ctx = self._create_mock_context()
        return await analyzer.generate_organizations(ctx)
    
    async def _get_changes_since(
        self, 
        scope_key: str, 
        since_ts: float
    ) -> List[Dict[str, Any]]:
        """
        Get all changes for a scope since a timestamp.
        
        Args:
            scope_key: The scope key to filter by
            since_ts: Timestamp to get changes since
            
        Returns:
            List of change records
        """
        changes = []
        
        # Check in-memory log first
        if scope_key in self._change_log:
            changes = [
                c for c in self._change_log[scope_key]
                if c['timestamp'] > since_ts
            ]
            if changes:
                return changes
        
        # Fall back to database
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT element_type, element_id, operation,
                           changed_fields, old_value, new_value,
                           EXTRACT(EPOCH FROM timestamp) as timestamp
                    FROM LoreChangeLog
                    WHERE conversation_id = $1
                      AND $2 = ANY(scope_keys)
                      AND timestamp > $3
                    ORDER BY timestamp ASC
                    LIMIT 100
                """,
                    self.conversation_id,
                    scope_key,
                    datetime.fromtimestamp(since_ts)
                )
                
                for row in rows:
                    changes.append({
                        'element_type': row['element_type'],
                        'element_id': row['element_id'],
                        'operation': row['operation'],
                        'changed_fields': row['changed_fields'] or [],  # asyncpg returns Python list/dict
                        'old_value': row['old_value'],  # asyncpg returns Python dict
                        'new_value': row['new_value'],  # asyncpg returns Python dict
                        'timestamp': row['timestamp']
                    })
                
                # Cache in memory for next time
                if changes and scope_key not in self._change_log:
                    self._change_log[scope_key] = changes[-20:]  # Keep last 20
                    
        except Exception as e:
            logger.warning(f"Failed to load changes from database: {e}")
        
        return changes

    async def on_element_updated(self, element_type: str, element_id: int, element_data: Dict[str, Any]):
        keys = self._get_affected_scope_keys(element_type, element_id, element_data or {})
        await self._invalidate_scope_keys(keys)

    async def _get_matriarchal_system(self):
        if not hasattr(self, '_matriarchal_system'):
            # IMPORTANT: pick your canonical class path and stick to it
            # If your canonical implementation is lore.main.MatriarchalLoreSystem, change import accordingly.
            from lore.matriarchal_lore_system import MatriarchalLoreSystem
            self._matriarchal_system = MatriarchalLoreSystem(self.user_id, self.conversation_id)
            await self._matriarchal_system.ensure_initialized() if hasattr(self._matriarchal_system, 'ensure_initialized') else None
            # Register with governance (if available on the class)
            try:
                if hasattr(self._matriarchal_system, 'register_with_governance'):
                    await self._matriarchal_system.register_with_governance()
            except Exception as e:
                logger.debug(f"Matriarchal system governance registration failed: {e}")
            logger.info("Matriarchal lore system initialized")
        return self._matriarchal_system
    
    def _get_affected_scope_keys(
        self, 
        element_type: str, 
        element_id: int,
        element_data: Dict[str, Any]
    ) -> List[str]:
        """
        Determine which scene-scope cache keys are affected by a lore change.
    
        Heuristics:
          - Always generate a "direct" scope for the changed entity (e.g., loc, nation, conflict).
          - Also generate combined scopes from any linkage fields present in element_data:
            location_id(s), nation_id(s), conflict_id(s), lore_tags/tags, topics, stakeholders, etc.
          - No I/O here: this is intentionally synchronous and best-effort.
        """
        keys: Set[str] = set()
    
        def _mk_scope(
            location_id: Optional[int] = None,
            nation_ids: Optional[Iterable[int]] = None,
            conflict_ids: Optional[Iterable[int]] = None,
            lore_tags: Optional[Iterable[str]] = None,
            topics: Optional[Iterable[str]] = None,
            npc_ids: Optional[Iterable[int]] = None,
        ) -> None:
            scope = type("Scope", (), {
                "location_id": location_id,
                "npc_ids": set(int(x) for x in (npc_ids or []) if x is not None),
                "lore_tags": set(str(x) for x in (lore_tags or []) if x),
                "topics": set(str(x).lower() for x in (topics or []) if x),
                "conflict_ids": set(int(x) for x in (conflict_ids or []) if x is not None),
                "nation_ids": set(int(x) for x in (nation_ids or []) if x is not None),
                "link_hints": {},
            })()
            keys.add(self._generate_scene_cache_key(scope))
    
        # Convenience extractors (robust to naming variance)
        def _ints(value) -> List[int]:
            if value is None:
                return []
            if isinstance(value, (set, tuple, list)):
                out = []
                for v in value:
                    try:
                        out.append(int(v))
                    except Exception:
                        pass
                return out
            try:
                return [int(value)]
            except Exception:
                return []
    
        def _strs(value) -> List[str]:
            if value is None:
                return []
            if isinstance(value, (set, tuple, list)):
                return [str(v) for v in value if v is not None]
            return [str(value)]
    
        # Generic link material found in element_data
        loc_id = element_data.get("location_id")
        loc_ids = set(_ints(element_data.get("location_ids"))) | ({int(loc_id)} if loc_id is not None else set())
    
        nation_ids = set(_ints(
            element_data.get("nation_ids") or
            element_data.get("nations") or
            element_data.get("nation_id")
        ))
    
        conflict_ids = set(_ints(
            element_data.get("conflict_ids") or
            element_data.get("conflict_id")
        ))
    
        tags = set(_strs(
            element_data.get("lore_tags") or
            element_data.get("tags")
        ))
    
        topics = set(_strs(
            element_data.get("topics") or
            element_data.get("topic")
        ))
    
        # Stakeholders may carry nation/location/npc hints (various shapes)
        stakeholders = element_data.get("stakeholders") or []
        for s in stakeholders:
            if isinstance(s, dict):
                stype = str(s.get("type", "")).lower()
                sid = s.get("id") or s.get("nation_id") or s.get("npc_id") or s.get("location_id")
                try:
                    sid_int = int(sid) if sid is not None else None
                except Exception:
                    sid_int = None
    
                if stype in {"nation", "state", "faction"} and sid_int is not None:
                    nation_ids.add(sid_int)
                elif stype in {"location", "city", "region"} and sid_int is not None:
                    loc_ids.add(sid_int)
                elif stype in {"npc", "character", "person"} and sid_int is not None:
                    # we don't anchor scopes on NPCs in the lore orchestrator, but keep to combine if needed
                    pass
            elif isinstance(s, str):
                # Accept formats like "nation:12", "location:7", "npc:55"
                m = re.match(r"(nation|state|faction|location|city|region|npc|character|person):(\d+)", s.lower())
                if m:
                    stype, sid = m.groups()
                    sid = int(sid)
                    if stype in {"nation", "state", "faction"}:
                        nation_ids.add(sid)
                    elif stype in {"location", "city", "region"}:
                        loc_ids.add(sid)
                    # npc ignored for now (see note above)
    
        # Element-type specific anchors + sensible combos
        et = element_type.lower()
    
        if et == "location":
            _mk_scope(location_id=element_id)
            if nation_ids:
                _mk_scope(location_id=element_id, nation_ids=nation_ids)
                _mk_scope(nation_ids=nation_ids)
            if tags:
                _mk_scope(location_id=element_id, lore_tags=tags)
                _mk_scope(lore_tags=tags)
            if topics:
                _mk_scope(location_id=element_id, topics=topics)
    
        elif et == "nation":
            _mk_scope(nation_ids=[element_id])
            if loc_ids:
                for lid in list(loc_ids)[:10]:
                    _mk_scope(location_id=lid, nation_ids=[element_id])
            if conflict_ids:
                for cid in list(conflict_ids)[:10]:
                    _mk_scope(conflict_ids=[cid])
                    _mk_scope(nation_ids=[element_id], conflict_ids=[cid])
            if tags:
                _mk_scope(lore_tags=tags)
    
        elif et == "religion":
            # Prefer scopes where this religion clearly manifests
            if nation_ids:
                for nid in list(nation_ids)[:5]:
                    _mk_scope(nation_ids=[nid])
            if loc_ids:
                for lid in list(loc_ids)[:8]:
                    _mk_scope(location_id=lid)
            if tags:
                _mk_scope(lore_tags=tags)
    
        elif et == "myth":
            # Myths usually attach to a location; also propagate tags
            if loc_ids:
                for lid in list(loc_ids)[:8]:
                    _mk_scope(location_id=lid)
                    if tags:
                        _mk_scope(location_id=lid, lore_tags=tags)
            if tags:
                _mk_scope(lore_tags=tags)
    
        elif et == "conflict":
            _mk_scope(conflict_ids=[element_id])
            if nation_ids:
                for nid in list(nation_ids)[:5]:
                    _mk_scope(nation_ids=[nid], conflict_ids=[element_id])
            if loc_ids:
                for lid in list(loc_ids)[:5]:
                    _mk_scope(location_id=lid, conflict_ids=[element_id])
            if tags:
                _mk_scope(lore_tags=tags)
    
        elif et in {"world_lore", "world", "global"}:
            if tags:
                _mk_scope(lore_tags=tags)
            if topics:
                _mk_scope(topics=topics)
    
        elif et in {"event", "historical_event"}:
            if loc_ids:
                for lid in list(loc_ids)[:8]:
                    _mk_scope(location_id=lid)
            if nation_ids:
                for nid in list(nation_ids)[:5]:
                    _mk_scope(nation_ids=[nid])
            if conflict_ids:
                for cid in list(conflict_ids)[:10]:
                    _mk_scope(conflict_ids=[cid])
    
        else:
            # Fallback: infer from whatever link hints we do have
            if loc_ids:
                for lid in list(loc_ids)[:8]:
                    _mk_scope(location_id=lid)
            if nation_ids:
                for nid in list(nation_ids)[:5]:
                    _mk_scope(nation_ids=[nid])
            if conflict_ids:
                for cid in list(conflict_ids)[:10]:
                    _mk_scope(conflict_ids=[cid])
            if tags:
                _mk_scope(lore_tags=tags)
            if topics:
                _mk_scope(topics=topics)
    
        # Always generate one ultra-generic anchor if nothing was inferred
        if not keys:
            _mk_scope()
    
        return list(keys)

    async def _invalidate_scope_keys(self, keys: List[str]) -> None:
        """
        Try to invalidate scene-scope caches across systems via the ContextBroker.
        No-ops (but logs) if no broker is attached.
        """
        if not keys:
            return
        broker = self.context_broker or getattr(self.nyx_context, "context_broker", None)
        if not broker:
            logger.debug("No context broker attached; skipping cross-system invalidation")
            return
        try:
            maybe = broker.invalidate_by_scope_keys(keys)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception as e:
            logger.warning(f"Scope invalidation failed: {e}")
    
    # ===== SCENE BUNDLE HELPER METHODS =====
    
    def _generate_scene_cache_key(self, scope: Any) -> str:
        """
        Stable, shared scene key:
          - If scope.to_key() exists, use it (already md5 hex from SceneScope).
          - Else use nyx.scene_keys.generate_scene_cache_key(scope).
          - Else fallback to deterministic local hash.
        """
        to_key = getattr(scope, "to_key", None)
        if callable(to_key):
            try:
                return to_key()
            except Exception:
                pass
    
        # Shared generator (keeps keys identical across Lore/NPC/Context)
        try:
            return generate_scene_cache_key(scope)
        except Exception:
            pass
    
        # Last-resort fallback (kept from your original logic)
        key_parts: List[str] = []
        loc_id = getattr(scope, "location_id", None)
        if loc_id is not None:
            key_parts.append(f"loc_{loc_id}")
        else:
            loc_name = getattr(scope, "location_name", None)
            if loc_name:
                key_parts.append(f"locname_{str(loc_name).lower()}")
    
        def _push_list(label: str, values: Any, head: int, coerce=str) -> None:
            if not values:
                return
            try:
                lst = list(values)
            except TypeError:
                return
            lst = sorted(lst)
            head_vals = [coerce(v) for v in lst[:head]]
            key_parts.append(f"{label}_{'_'.join(head_vals)}+n={len(lst)}")
    
        _push_list("npcs", getattr(scope, "npc_ids", []), head=5, coerce=lambda v: str(int(v)))
        _push_list("tags", getattr(scope, "lore_tags", []), head=5, coerce=str)
        _push_list("topics", getattr(scope, "topics", []), head=3, coerce=lambda v: str(v).lower())
        _push_list("nations", getattr(scope, "nation_ids", []), head=3, coerce=lambda v: str(int(v)))
        _push_list("conflicts", getattr(scope, "conflict_ids", []), head=3, coerce=lambda v: str(int(v)))
    
        if not key_parts:
            key_parts.append("empty")
    
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode("utf-8")).hexdigest()

    # ===== CACHE ANALYTICS / OPTIMIZATION =====
    
    async def cache_get_analytics(self) -> Dict[str, Any]:
        """Expose cache analytics (LoreCache)."""
        return await GLOBAL_LORE_CACHE.get_cache_analytics()
    
    async def cache_optimize(self) -> Dict[str, Any]:
        """Run the cache optimization agent (LoreCache)."""
        return await GLOBAL_LORE_CACHE.optimize_cache()
    
    async def cache_warm_predictive(self, warm_strategy: str = "high_miss") -> Dict[str, Any]:
        """Pre-warm the cache based on recent miss patterns."""
        return await GLOBAL_LORE_CACHE.warm_predictive_cache(warm_strategy=warm_strategy)
    
    def _get_cached_bundle(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get a cached bundle if it exists and is fresh (no memoization)."""
        bundle = self._scene_bundle_cache.get(cache_key)
        if bundle and self._is_bundle_stale(bundle):
            # Hard-evict stale entries so future calls rebuild
            self._scene_bundle_cache.pop(cache_key, None)
            self._bundle_last_changed.pop(cache_key, None)
            self._bundle_cached_at.pop(cache_key, None)
            return None
        return bundle

    # ===== CANON WRAPPERS =====
    
    async def canon_log_event(self, ctx, event_description: str, tags: Optional[List[str]] = None, significance: int = 5) -> int:
        async with get_db_connection_context() as conn:
            return await canon_log_canonical_event(ctx, conn, event_description, tags or [], significance)
    
    async def canon_find_or_create_npc(self, ctx, npc_name: str, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_npc(ctx, conn, npc_name, **kwargs)
    
    async def canon_find_or_create_nation(self, ctx, nation_name: str, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_nation(ctx, conn, nation_name, **kwargs)
    
    async def canon_find_or_create_location(self, ctx, location_name: str, **kwargs) -> str:
        async with get_db_connection_context() as conn:
            return await find_or_create_location(ctx, conn, location_name, **kwargs)
    
    async def canon_find_or_create_faction(self, ctx, faction_name: str, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_faction(ctx, conn, faction_name, **kwargs)
    
    async def canon_find_or_create_historical_event(self, ctx, event_name: str, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_historical_event(ctx, conn, event_name, **kwargs)
    
    async def canon_find_or_create_myth(self, ctx, name: str, description: str, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_urban_myth(ctx, conn, name=name, description=description, **kwargs)
    
    async def canon_find_or_create_landmark(self, ctx, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_landmark(ctx, conn, **kwargs)
    
    async def canon_find_or_create_event(self, ctx, event_name: str, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_event(ctx, conn, event_name, **kwargs)
    
    async def canon_find_or_create_quest(self, ctx, quest_name: str, **kwargs) -> int:
        async with get_db_connection_context() as conn:
            return await find_or_create_quest(ctx, conn, quest_name, **kwargs)
    
    async def canon_sync_entity_to_memory(self, ctx, entity_type: str, entity_id: int, force: bool = False) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            return await sync_entity_to_memory(ctx, conn, entity_type, entity_id, force=force)
    
    async def canon_ensure_embedding_columns(self) -> None:
        async with get_db_connection_context() as conn:
            await ensure_embedding_columns(conn)
    
    def _is_bundle_stale(self, bundle: Dict[str, Any]) -> bool:
        """Check if a bundle is stale based on TTL using monotonic clock."""
        # Use cache key from bundle for TTL tracking
        cache_key = bundle.get('_cache_key')
        if cache_key is None:
            return True
        
        cached_at = self._bundle_cached_at.get(cache_key)
        if cached_at is None:
            return True
        
        age = time.monotonic() - cached_at
        return age > self._bundle_ttl
    
    def _cache_bundle(self, cache_key: str, bundle: Dict[str, Any]) -> None:
        """Cache a bundle and track its change time."""
        # Store cache key in bundle for TTL tracking
        bundle['_cache_key'] = cache_key
        
        self._scene_bundle_cache[cache_key] = bundle
        self._bundle_last_changed[cache_key] = bundle['last_changed_at']
        self._bundle_cached_at[cache_key] = time.monotonic()
        
        # Limit cache size using separate limit for bundles
        max_size = self.config.bundle_cache_max_size if hasattr(self.config, 'bundle_cache_max_size') else 100
        if len(self._scene_bundle_cache) > max_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self._scene_bundle_cache.keys(),
                key=lambda k: self._scene_bundle_cache[k].get('last_changed_at', 0)
            )
            # Remove oldest 25%
            for key in sorted_keys[:len(self._scene_bundle_cache) // 4]:
                self._scene_bundle_cache.pop(key, None)
                self._bundle_last_changed.pop(key, None)
                self._bundle_cached_at.pop(key, None)
    
    async def _fetch_location_lore_for_bundle(self, location_id: int) -> Dict[str, Any]:
        """Fetch location-specific lore in canonical shape (id/name)."""
        try:
            location_lore = {}
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
    
                # Normalize columns to canonical id/name
                row = await conn.fetchrow("""
                    SELECT 
                        COALESCE(id, location_id) AS id,
                        COALESCE(location_name, name) AS name,
                        description,
                        nation_id,
                        governance,
                        culture,
                        population,
                        canonical_rules,
                        tags
                    FROM Locations
                    WHERE COALESCE(id, location_id) = $1
                    LIMIT 1
                """, location_id)
    
                if row:
                    location_lore = {
                        'id': row['id'],
                        'name': row['name'],
                        'description': (row['description'] or '')[:200],
                        'nation_id': row['nation_id'],
                        'governance': row['governance'] or {},
                        'culture': row['culture'] or {},
                        'population': row['population'] or 0,
                        'canonical_rules': row['canonical_rules'] or [],
                        'tags': row['tags'] or []
                    }
    
                    # Change tracking
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('location', row['id'], location_lore)
                        await self._track_element_change('location', row['id'], location_lore, scope_keys)
    
                    # Landmarks (limit 3)
                    lms = await conn.fetch("""
                        SELECT landmark_id AS id, name, significance
                        FROM Landmarks
                        WHERE location_id = $1
                        ORDER BY significance DESC
                        LIMIT 3
                    """, row['id'])
                    if lms:
                        location_lore['landmarks'] = [
                            {'id': lm['id'], 'name': lm['name'], 'significance': lm['significance']}
                            for lm in lms
                        ]
    
                    # Recent events by location name
                    evs = await conn.fetch("""
                        SELECT event_name AS name, description, event_date
                        FROM Events
                        WHERE location = $1
                        ORDER BY event_date DESC
                        LIMIT 2
                    """, row['name'])
                    if evs:
                        location_lore['recent_events'] = [
                            {
                                'name': e['name'],
                                'description': (e['description'] or '')[:100],
                                'date': e['event_date'].isoformat() if e['event_date'] else None
                            }
                            for e in evs
                        ]
    
            return location_lore
    
        except Exception as e:
            logger.debug(f"Could not fetch location lore: {e}")
            return {}
    
    async def _fetch_world_lore_for_bundle(self, tags: List[str]) -> Dict[str, Any]:
        """Fetch world lore based on tags."""
        try:
            # Use get_tagged_lore method
            tagged_lore = await self.get_tagged_lore(tags=tags)
            
            # Add any world-level facts from cache
            if self.config.enable_cache and self._cache_system:
                for tag in tags[:3]:
                    cached = await self._cache_system.get(f"world_lore_{tag}")
                    if cached:
                        tagged_lore[tag] = cached
            
            return tagged_lore
        except Exception as e:
            logger.debug(f"Could not fetch world lore: {e}")
            return {}
    
    async def _fetch_nations_for_bundle(self, nation_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetch nation data for bundle using schema-normalized batch query.
        Ensures culture is always populated via COALESCE(culture, lore_context).
        """
        try:
            nations: List[Dict[str, Any]] = []
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
    
                rows = await conn.fetch("""
                    SELECT
                        COALESCE(id, nation_id)     AS id,
                        COALESCE(name, nation_name) AS name,
                        COALESCE(government_type, government) AS government_type,
                        COALESCE(culture, lore_context) AS culture
                    FROM Nations
                    WHERE COALESCE(id, nation_id) = ANY($1::int[])
                """, nation_ids[:5])
    
                for r in rows:
                    nation_data = {
                        'id': r['id'],
                        'name': r['name'],
                        'government': r['government_type'] or 'unknown',
                        'culture': r['culture'] or {}
                    }
                    nations.append(nation_data)
    
                    # Change tracking
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('nation', r['id'], nation_data)
                        await self._track_element_change('nation', r['id'], nation_data, scope_keys)
    
            # Attach brief culture summary (non-blocking best-effort)
            try:
                for n in nations[:2]:  # only summarize first two to keep it light
                    try:
                        summary = await asyncio.wait_for(self.rc_summarize_culture(n['id'], format_type="brief"), timeout=1.2)
                        n['culture_summary'] = summary
                    except Exception:
                        pass
            except Exception:
                pass
            
            return nations
        except Exception as e:
            logger.debug(f"Could not fetch nations: {e}")
            return []
        
    async def _fetch_religions_for_location(self, location_id: int) -> List[Dict[str, Any]]:
        try:
            religions = []
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
    
                query_try = """
                    SELECT DISTINCT 
                        COALESCE(r.id, r.religion_id) AS id,
                        COALESCE(r.religion_name, r.name) AS name,
                        r.deity_names,
                        r.core_beliefs,
                        r.sacred_texts,
                        nrd.influence_level
                    FROM Religions r
                    LEFT JOIN NationReligionDistribution nrd ON COALESCE(r.id, r.religion_id) = nrd.religion_id
                    LEFT JOIN Locations l ON l.nation_id = nrd.nation_id
                    WHERE COALESCE(l.id, l.location_id) = $1 AND COALESCE(nrd.influence_level, 0) > 0.1
                    ORDER BY nrd.influence_level DESC
                    LIMIT 5
                """
    
                rows = []
                try:
                    rows = await conn.fetch(query_try, location_id)
                except Exception:
                    # Fallback: derive a pseudo-influence from NationReligion.religiosity_level
                    rows = await conn.fetch("""
                        SELECT DISTINCT 
                            COALESCE(r.id, r.religion_id) AS id,
                            COALESCE(r.religion_name, r.name) AS name,
                            r.deity_names,
                            r.core_beliefs,
                            r.sacred_texts,
                            nr.religiosity_level AS influence_level
                        FROM Religions r
                        JOIN NationReligion nr ON TRUE
                        JOIN Locations l ON l.nation_id = nr.nation_id
                        WHERE COALESCE(l.id, l.location_id) = $1
                        ORDER BY COALESCE(nr.religiosity_level, 0) DESC
                        LIMIT 5
                    """, location_id)
    
                for row in rows:
                    religion_data = {
                        'id': row['id'],
                        'name': row['name'],
                        'deities': (row['deity_names'] or [])[:3],
                        'beliefs': (row['core_beliefs'] or '')[:100],
                        'influence': row.get('influence_level') or 0.5
                    }
                    religions.append(religion_data)
    
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('religion', row['id'], religion_data)
                        await self._track_element_change('religion', row['id'], religion_data, scope_keys)
    
            return religions
        except Exception as e:
            logger.debug(f"Could not fetch religions: {e}")
            return []
    
    async def _fetch_conflicts_for_bundle(self, conflict_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetch conflicts in canonical shape:
        { id, type, description, stakeholders, intensity, phase, resolution_status }
        Supports both Conflicts and NationalConflicts.
        """
        conflicts: List[Dict[str, Any]] = []
    
        async def _from_conflicts_table(conn):
            rows = await conn.fetch("""
                SELECT 
                    COALESCE(id, conflict_id) AS id,
                    conflict_type,
                    description,
                    stakeholders,
                    COALESCE(intensity, NULL) AS intensity,
                    COALESCE(phase, NULL) AS phase,
                    COALESCE(resolution_status, NULL) AS resolution_status
                FROM Conflicts
                WHERE COALESCE(id, conflict_id) = ANY($1::int[])
            """, conflict_ids[:5])
            out = []
            for r in rows:
                out.append({
                    'id': r['id'],
                    'type': r['conflict_type'],
                    'description': (r['description'] or '')[:150],
                    'stakeholders': (r['stakeholders'] or [])[:3],
                    'intensity': r['intensity'] if r['intensity'] is not None else 0.5,
                    'phase': r['phase'] or 'active',
                    'resolution_status': r['resolution_status'] or 'ongoing'
                })
            return out
    
        async def _from_national_conflicts(conn):
            rows = await conn.fetch("""
                SELECT 
                    id,
                    COALESCE(conflict_type,'standard') AS conflict_type,
                    description,
                    COALESCE(severity,5) AS severity,
                    COALESCE(status,'active') AS status,
                    COALESCE(involved_nations, ARRAY[]::int[]) AS involved_nations
                FROM NationalConflicts
                WHERE id = ANY($1::int[])
            """, conflict_ids[:5])
            out = []
            for r in rows:
                out.append({
                    'id': r['id'],
                    'type': r['conflict_type'],
                    'description': (r['description'] or '')[:150],
                    # Synthesize stakeholders from involved_nations
                    'stakeholders': [{'type': 'nation', 'id': nid} for nid in (r['involved_nations'] or [])][:3],
                    # Map severity -> intensity 0..1
                    'intensity': max(0.0, min(1.0, float(r['severity']) / 10.0)),
                    # Map status -> phase
                    'phase': r['status'],
                    # We don't have resolution_status here; infer from status
                    'resolution_status': 'resolved' if (r['status'] or '').lower() == 'resolved' else 'ongoing'
                })
            return out
    
        try:
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
                try:
                    primary = await _from_conflicts_table(conn)
                    conflicts.extend(primary)
                except Exception as e:
                    logger.debug(f"Primary Conflicts fetch failed, will try NationalConflicts: {e}")
    
                # If we didn't get enough or none, try NationalConflicts too
                if len(conflicts) < len(conflict_ids):
                    secondary = await _from_national_conflicts(conn)
                    # Avoid duplicates if ids overlap
                    seen = {c['id'] for c in conflicts}
                    conflicts.extend([c for c in secondary if c['id'] not in seen])
    
                # Change tracking on all
                if self._change_tracking_enabled:
                    for c in conflicts:
                        scope_keys = self._get_affected_scope_keys('conflict', c['id'], c)
                        await self._track_element_change('conflict', c['id'], c, scope_keys)
    
        except Exception as e:
            logger.debug(f"Could not fetch conflicts: {e}")
    
        return conflicts
    
    async def _fetch_myths_for_location(self, location_id: int) -> List[Dict[str, Any]]:
        """Fetch urban myths for a location with canonical fields (id/title/...)."""
        try:
            myths = []
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
                result = await conn.fetch("""
                    SELECT 
                        COALESCE(um.id, um.myth_id) AS id,
                        um.title,
                        um.description,
                        COALESCE(um.origin_period, 'unknown') AS origin_period,
                        COALESCE(um.belief_level, 0.3) AS belief_level,
                        um.variants
                    FROM UrbanMyths um
                    WHERE um.location_id = $1
                    ORDER BY COALESCE(um.belief_level, 0) DESC
                    LIMIT 5
                """, location_id)
    
                for row in result:
                    myth_data = {
                        'id': row['id'],
                        'title': row['title'],
                        'description': (row['description'] or '')[:200],
                        'origin': row['origin_period'],
                        'belief_level': row['belief_level'],
                        'has_variants': bool(row['variants'])
                    }
                    myths.append(myth_data)
    
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('myth', row['id'], myth_data)
                        await self._track_element_change('myth', row['id'], myth_data, scope_keys)
    
            return myths
        except Exception as e:
            logger.debug(f"Could not fetch myths: {e}")
            return []
        
    async def _get_canonical_rules_for_scope(self, scope: Any) -> List[str]:
        """Get canonical consistency rules relevant to the scope."""
        try:
            rules = []
            
            # Get canon module and validation (ensure initialized)
            canon = await self._get_canon_module()
            validator = await self._get_canon_validation()
            
            # Add location-specific canonical rules
            if hasattr(scope, 'location_id') and scope.location_id:
                async with get_db_connection_context() as conn:
                    # Get location governance rules
                    location_rules = await conn.fetchval("""
                        SELECT canonical_rules
                        FROM Locations
                        WHERE COALESCE(id, location_id) = $1
                    """, scope.location_id)
                    
                    if location_rules:
                        if isinstance(location_rules, list):
                            rules.extend(location_rules[:3])
                        elif isinstance(location_rules, str):
                            rules.append(location_rules)
            
            # Add nation-specific rules if nations in scope
            if hasattr(scope, 'nation_ids') and scope.nation_ids:
                async with get_db_connection_context() as conn:
                    for nation_id in list(scope.nation_ids)[:2]:
                        nation_rules = await conn.fetchval("""
                            SELECT governance_rules 
                            FROM Nations 
                            WHERE COALESCE(id, nation_id) = $1
                        """, nation_id)
                        
                        if nation_rules:
                            if isinstance(nation_rules, dict):
                                # Extract key rules from governance structure
                                if 'laws' in nation_rules:
                                    rules.append(f"Nation {nation_id}: {nation_rules['laws'][:100]}")
            
            # Add global canonical rules from validator
            if hasattr(validator, 'get_global_rules'):
                global_rules = await validator.get_global_rules()
                rules.extend(global_rules[:2])
            else:
                # Default canonical rules
                rules.extend([
                    "Maintain timeline consistency",
                    "Respect established character relationships",
                    "Honor location governance structures"
                ])
            
            # Deduplicate and limit
            seen = set()
            unique_rules = []
            for rule in rules:
                if rule not in seen:
                    seen.add(rule)
                    unique_rules.append(rule)
            
            return unique_rules[:5]  # Limit to top 5 rules
        except Exception as e:
            logger.debug(f"Could not fetch canonical rules: {e}")
            return []
    
    async def get_tagged_lore(self, tags: List[str]) -> Dict[str, Any]:
        """
        Get lore elements by tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            Dictionary of tagged lore elements
        """
        try:
            tagged_lore = {}
            
            async with get_db_connection_context() as conn:
                # Search across multiple lore tables for tagged content
                for tag in tags[:10]:  # Limit to 10 tags
                    tag_lower = tag.lower()
                    lore_items = []
                    
                    # Search in Nations
                    nations = await conn.fetch("""
                        SELECT nation_id, nation_name, culture, lore_context
                        FROM Nations
                        WHERE LOWER(culture::text) LIKE $1 
                           OR LOWER(lore_context::text) LIKE $1
                        LIMIT 2
                    """, f'%{tag_lower}%')
                    
                    for nation in nations:
                        lore_items.append({
                            'type': 'nation',
                            'id': nation['nation_id'],
                            'name': nation['nation_name'],
                            'relevance': 0.8
                        })
                    
                    # Search in Religions
                    religions = await conn.fetch("""
                        SELECT religion_id, religion_name, core_beliefs
                        FROM Religions
                        WHERE LOWER(core_beliefs::text) LIKE $1
                           OR LOWER(religion_name) LIKE $1
                        LIMIT 2
                    """, f'%{tag_lower}%')
                    
                    for religion in religions:
                        lore_items.append({
                            'type': 'religion',
                            'id': religion['religion_id'],
                            'name': religion['religion_name'],
                            'relevance': 0.7
                        })
                    
                    # Search in historical events (if table exists)
                    try:
                        events = await conn.fetch("""
                            SELECT event_id, event_name, description
                            FROM Events
                            WHERE LOWER(description::text) LIKE $1
                               OR LOWER(event_name) LIKE $1
                            LIMIT 2
                        """, f'%{tag_lower}%')
                        
                        for event in events:
                            lore_items.append({
                                'type': 'event',
                                'id': event['event_id'],
                                'name': event['event_name'],
                                'relevance': 0.6
                            })
                    except:
                        pass  # Events table might not exist
                    
                    if lore_items:
                        tagged_lore[tag] = lore_items
            
            # Add cached lore if available
            if self.config.enable_cache and self._cache_system:
                for tag in tags:
                    cache_key = f"lore_tag_{tag}"
                    cached = await self._cache_system.get(cache_key)
                    if cached and tag not in tagged_lore:
                        tagged_lore[tag] = cached
            
            return tagged_lore
            
        except Exception as e:
            logger.error(f"Error fetching tagged lore: {e}")
            return {}
    
    async def check_canonical_consistency(self) -> Dict[str, Any]:
        """
        Check canonical consistency across all lore systems.
    
        Returns:
            Dictionary with consistency check results and rules
        """
        try:
            result = {
                'is_consistent': True,
                'rules': [],
                'violations': [],
                'warnings': []
            }
    
            # Ensure modules/agents are initialized
            await self._get_canon_module()
            validator = await self._get_canon_validation()
    
            if validator:
                async with get_db_connection_context() as conn:
                    # Timeline consistency: prefer event_date for Events
                    timeline_check = await conn.fetchval("""
                        SELECT COUNT(*)
                        FROM Events
                        WHERE conversation_id = $1
                          AND event_date > CURRENT_TIMESTAMP
                    """, self.conversation_id)
    
                    if timeline_check and timeline_check > 0:
                        result['warnings'].append(f"Found {timeline_check} future-dated events")
                        result['is_consistent'] = False
    
                    # NPC relationship symmetry
                    npc_check = await conn.fetch("""
                        SELECT sl1.entity1_id, sl1.entity2_id, sl1.link_type
                        FROM SocialLinks sl1
                        JOIN SocialLinks sl2
                          ON sl1.entity1_id = sl2.entity2_id
                         AND sl1.entity2_id = sl2.entity1_id
                        WHERE sl1.conversation_id = $1
                          AND sl1.link_type != sl2.link_type
                        LIMIT 5
                    """, self.conversation_id)
    
                    if npc_check:
                        for row in npc_check:
                            result['warnings'].append(
                                f"Asymmetric relationship between NPCs {row['entity1_id']} and {row['entity2_id']}"
                            )
    
                    # Collect canonical rules from existing data
                    rules_query = await conn.fetch("""
                        SELECT DISTINCT canonical_rule 
                        FROM (
                            SELECT unnest(canonical_rules) as canonical_rule 
                            FROM Locations 
                            WHERE conversation_id = $1
                            UNION
                            SELECT governance_rules::text as canonical_rule 
                            FROM Nations 
                            WHERE conversation_id = $1
                        ) rules
                        WHERE canonical_rule IS NOT NULL
                        LIMIT 10
                    """, self.conversation_id)
    
                    for row in rules_query:
                        if row['canonical_rule']:
                            result['rules'].append(row['canonical_rule'])
    
            # Default rules if none were sourced
            if not result['rules']:
                result['rules'] = [
                    "Maintain temporal consistency across all events",
                    "Preserve established character relationships and personalities",
                    "Respect location governance and cultural norms",
                    "Ensure magic/technology consistency with world rules",
                    "Honor faction alliances and conflicts"
                ]
    
            # Overall consistency decision
            if result['violations'] or len(result['warnings']) > 5:
                result['is_consistent'] = False
    
            # Summary
            result['summary'] = {
                'total_rules': len(result['rules']),
                'violations': len(result['violations']),
                'warnings': len(result['warnings']),
                'checked_at': datetime.now().isoformat()
            }
    
            return result
    
        except Exception as e:
            logger.error(f"Error checking canonical consistency: {e}")
            return {
                'is_consistent': True,  # Assume consistent on error
                'rules': [
                    "Maintain temporal consistency",
                    "Preserve character relationships",
                    "Respect world rules"
                ],
                'error': str(e)
            }
        
    # ===== COMPONENT GETTERS (LAZY INITIALIZATION) =====
    
    async def _get_canon_module(self):
        """
        Get or initialize the canon module.
        CRITICAL: This is lazy loaded as many other modules depend on it.
        """
        if not self._canon_module:
            # Import here to avoid circular dependencies
            from lore.core import canon
            self._canon_module = canon
            logger.info("Canon module loaded")
        return self._canon_module
    
    async def _get_cache_system(self):
        """Get or initialize the cache system."""
        if not self._cache_system:
            from lore.core.cache import LoreCache
            self._cache_system = LoreCache(
                max_size=self.config.cache_max_size,
                ttl=self.config.cache_ttl
            )
            logger.info("Cache system initialized")
        return self._cache_system
    
    async def _get_registry_system(self):
        """Get or initialize the manager registry."""
        if not self._registry_system:
            from lore.core.registry import ManagerRegistry
            self._registry_system = ManagerRegistry(self.user_id, self.conversation_id)
            logger.info("Registry system initialized")
        return self._registry_system
    
    async def _get_canon_validation(self):
        """Get or initialize the canon validation agent."""
        if not self._canon_validation:
            from lore.core.validation import CanonValidationAgent
            self._canon_validation = CanonValidationAgent()
            logger.info("Canon validation agent initialized")
        return self._canon_validation
    
    def _get_canonical_context_class(self):
        """Get the CanonicalContext class (lazy loaded)."""
        if not self._canonical_context_class:
            from lore.core.context import CanonicalContext
            self._canonical_context_class = CanonicalContext
            logger.info("CanonicalContext class loaded")
        return self._canonical_context_class
    
    async def _get_education_manager(self):
        """Get or initialize the education manager."""
        if not self._education_manager:
            from lore.managers.education import EducationalSystemManager
            self._education_manager = EducationalSystemManager(self.user_id, self.conversation_id)
            await self._education_manager.ensure_initialized()
            logger.info("Education manager initialized")
        return self._education_manager
    
    async def _get_religion_manager(self):
        """Get or initialize the religion manager."""
        if not self._religion_manager:
            from lore.managers.religion import ReligionManager
            self._religion_manager = ReligionManager(self.user_id, self.conversation_id)
            await self._religion_manager.ensure_initialized()
            logger.info("Religion manager initialized")
        return self._religion_manager
    
    async def _get_local_lore_manager(self):
        """Get or initialize the local lore manager."""
        if not self._local_lore_manager:
            from lore.managers.local_lore import LocalLoreManager
            self._local_lore_manager = LocalLoreManager(self.user_id, self.conversation_id)
            await self._local_lore_manager.ensure_initialized()
            logger.info("Local lore manager initialized")
        return self._local_lore_manager
    
    async def _get_geopolitical_manager(self):
        """Get or initialize the geopolitical manager."""
        if not self._geopolitical_manager:
            from lore.managers.geopolitical import GeopoliticalSystemManager
            self._geopolitical_manager = GeopoliticalSystemManager(self.user_id, self.conversation_id)
            await self._geopolitical_manager.ensure_initialized()
            logger.info("Geopolitical manager initialized")
        return self._geopolitical_manager
    
    async def _get_politics_manager(self):
        """Get or initialize the politics manager."""
        if not self._politics_manager:
            from lore.managers.politics import WorldPoliticsManager
            self._politics_manager = WorldPoliticsManager(self.user_id, self.conversation_id)
            await self._politics_manager.ensure_initialized()
            logger.info("Politics manager initialized")
        return self._politics_manager
    
    async def _get_lore_system(self):
        """Get or initialize the core lore system."""
        if not self._lore_system:
            from lore.core.lore_system import LoreSystem
            self._lore_system = LoreSystem(self.user_id, self.conversation_id)
            await self._lore_system.initialize()
            logger.info("Core lore system initialized")
        return self._lore_system
    
    async def _get_npc_integration(self):
        if not self._npc_integration:
            from lore.integration import NPCLoreIntegration
            self._npc_integration = NPCLoreIntegration(self.user_id, self.conversation_id)
            # Inject governance if available
            try:
                governor = await get_central_governance(self.user_id, self.conversation_id)
                self._npc_integration.set_governor(governor)
            except Exception:
                logger.debug("Could not attach governor to NPCLoreIntegration")
            await self._npc_integration.initialize()
            logger.info("NPC integration initialized")
        return self._npc_integration
    
    async def _get_context_enhancer(self):
        if not self._context_enhancer:
            from lore.integration import ContextEnhancer
            self._context_enhancer = ContextEnhancer(self.user_id, self.conversation_id)
            try:
                governor = await get_central_governance(self.user_id, self.conversation_id)
                self._context_enhancer.set_governor(governor)
            except Exception:
                logger.debug("Could not attach governor to ContextEnhancer")
            await self._context_enhancer.initialize()
            logger.info("Context enhancer initialized")
        return self._context_enhancer
    
    async def _get_world_lore_manager(self):
        if not self._world_lore_manager:
            from lore.managers.world_lore_manager import WorldLoreManager
            self._world_lore_manager = WorldLoreManager(self.user_id, self.conversation_id)
            await self._world_lore_manager.ensure_initialized()
            logger.info("World lore manager initialized")
        return self._world_lore_manager
    
    async def _get_lore_dynamics_system(self):
        """Get or initialize the lore dynamics system."""
        if not hasattr(self, '_lore_dynamics_system'):
            from lore.systems.dynamics import LoreDynamicsSystem
            self._lore_dynamics_system = LoreDynamicsSystem(self.user_id, self.conversation_id)
            await self._lore_dynamics_system.ensure_initialized()
            # Attach governance if available and register
            try:
                governor = await get_central_governance(self.user_id, self.conversation_id)
                self._lore_dynamics_system.governor = governor
                await self._lore_dynamics_system.register_with_governance()
            except Exception as e:
                logger.debug(f"Could not attach/register governance to LoreDynamicsSystem: {e}")
            logger.info("Lore dynamics system initialized")
        return self._lore_dynamics_system

    async def dynamics_evolve_lore_with_event(self, ctx, event_description: str) -> Dict[str, Any]:
        ds = await self._get_lore_dynamics_system()
        result = await ds.evolve_lore_with_event(ctx, event_description)
    
        # Best-effort change tracking so scene deltas work
        try:
            # Updates (existing elements)
            for upd in result.get('updates', []):
                etype = str(upd.get('lore_type', 'world_lore')).lower()
                eid = upd.get('lore_id')
                if eid is None:
                    continue
                # Normalize type names to our canonical delta types
                if etype in ('locations', 'locationlore', 'location'):
                    etype = 'location'
                elif etype in ('historicalevents', 'worldlore', 'culturalelements', 'geographicregions'):
                    etype = 'world_lore'
                elif etype in ('factions',):
                    etype = 'faction'
                elif etype in ('urbanmyths', 'localhistories', 'landmarks', 'notablefigures'):
                    # optional: track each as their own; otherwise 'world_lore'
                    etype = 'myth' if etype == 'urbanmyths' else etype
    
                new_data = {
                    'name': upd.get('name'),
                    'description': upd.get('new_description'),
                    'update_reason': upd.get('update_reason'),
                    'impact_level': upd.get('impact_level'),
                }
                await self.record_lore_change(etype, int(eid), 'update', new_data=new_data)
        except Exception as e:
            logger.debug(f"dynamics_evolve_lore_with_event: change tracking (updates) failed: {e}")
    
        try:
            # New elements created
            for ne in result.get('new_elements', []):
                etype = str(ne.get('lore_type', 'world_lore')).lower()
                # ID may not be returned by generation; skip if missing
                eid = ne.get('id') or ne.get('lore_id')
                if not eid:
                    continue
                if etype in ('factions',):
                    etype = 'faction'
                elif etype in ('culturalelements', 'worldlore', 'geographicregions'):
                    etype = 'world_lore'
                await self.record_lore_change(etype, int(eid), 'create', new_data=ne)
        except Exception as e:
            logger.debug(f"dynamics_evolve_lore_with_event: change tracking (new elements) failed: {e}")
    
        return result

    async def dynamics_stream_world_changes(
        self,
        event_data: Dict[str, Any],
        affected_elements: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream progressive updates about world changes."""
        from lore.systems.dynamics import WorldStateStreamer
        ds = await self._get_lore_dynamics_system()
        streamer = WorldStateStreamer(ds)
        async for chunk in streamer.stream_world_changes(event_data, affected_elements):
            yield chunk
    
    async def dynamics_stream_evolution_scenario(self, initial_state: Optional[Dict[str, Any]] = None, years: int = 10):
        from lore.systems.dynamics import WorldStateStreamer
        ds = await self._get_lore_dynamics_system()
        streamer = WorldStateStreamer(ds)
        async for year_chunk in streamer.stream_evolution_scenario(initial_state or {}, years):
            yield year_chunk
    
    async def dynamics_create_evolution_plan(self, ctx, initial_prompt: str) -> Dict[str, Any]:
        from lore.systems.dynamics import MultiStepPlanner
        dynamics = await self._get_lore_dynamics_system()
        planner = MultiStepPlanner(dynamics)
        context = {"user_id": self.user_id, "conversation_id": self.conversation_id}
        plan = await planner.create_evolution_plan(initial_prompt, context)
        if self._change_tracking_enabled:
            await self.record_lore_change('narrative_plan', plan.get('id', 0), 'create',
                                          new_data={'prompt': initial_prompt, 'steps': len(plan.get('steps', []))})
        return plan
    
    async def dynamics_evaluate_narrative(self, ctx, narrative_element: Dict[str, Any], element_type: str) -> Dict[str, Any]:
        from lore.systems.dynamics import NarrativeEvaluator
        dynamics = await self._get_lore_dynamics_system()
        evaluator = NarrativeEvaluator(dynamics)
        return await evaluator.evaluate_narrative(narrative_element, element_type)

    async def dynamics_generate_emergent_event(self, ctx) -> Dict[str, Any]:
        ds = await self._get_lore_dynamics_system()
        data = await ds.generate_emergent_event(ctx)
    
        # Optionally record the top-level event as world_lore
        try:
            if data and 'description' in data:
                await self.record_lore_change('world_lore', 0, 'create', new_data={'event': data.get('event_name'), 'description': data['description']})
        except Exception:
            pass
        return data
    
    async def dynamics_mature_lore_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        ds = await self._get_lore_dynamics_system()
        ctx = self._create_mock_context()
        try:
            return await ds.mature_lore_over_time(ctx, days_passed)
        except TypeError:
            return await ds.mature_lore_over_time(days_passed)
    
    async def _get_lore_update_system(self):
        """Get or initialize the lore update system."""
        if not hasattr(self, '_lore_update_system'):
            from lore.matriarchal_lore_system import LoreUpdateSystem
            self._lore_update_system = LoreUpdateSystem(self.user_id, self.conversation_id)
            logger.info("Lore update system initialized")
        return self._lore_update_system
    
    async def _get_dynamic_generator(self):
        """Get or initialize the dynamic lore generator."""
        if not hasattr(self, '_dynamic_generator'):
            from lore.lore_generator import DynamicLoreGenerator
            self._dynamic_generator = DynamicLoreGenerator.get_instance(
                self.user_id, 
                self.conversation_id
            )
            await self._dynamic_generator.initialize()
            logger.info("Dynamic lore generator initialized")
        return self._dynamic_generator

    # agentic flows (governed)
    async def agent_create_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        from agents.run_context import RunContextWrapper
        from lore.lore_agents import create_complete_lore_with_governance
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        return await create_complete_lore_with_governance(ctx, environment_desc)
    
    async def agent_generate_scene_description(self, location_name: str, lore_context: Dict[str, Any], npc_ids: List[int] = None):
        from agents.run_context import RunContextWrapper
        from lore.lore_agents import generate_scene_description_with_lore_and_governance
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        return await generate_scene_description_with_lore_and_governance(ctx, location_name, lore_context, npc_ids or [])
    
    # consolidated dynamics/generator (governed)
    async def generator_generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        gen = await self._get_dynamic_generator()
        return await gen.generate_complete_lore(environment_desc)
    
    async def generator_initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        gen = await self._get_dynamic_generator()
        return await gen.initialize_world_lore(environment_desc)

    async def get_resource_stats(self) -> Dict[str, Any]:
        from lore.resource_manager import resource_manager
        return await resource_manager.get_resource_stats()
    
    async def optimize_resources_now(self) -> Dict[str, Any]:
        from lore.resource_manager import resource_manager
        return await resource_manager.optimize_resources()
    
    async def cleanup_resources_now(self) -> Dict[str, Any]:
        from lore.resource_manager import resource_manager
        return await resource_manager.cleanup_resources()

    async def apply_dialect(self, text: str, dialect_id: int, intensity: str = 'medium', npc_id: Optional[int] = None) -> str:
        integ = await self._get_npc_integration()
        return await integ.apply_dialect_to_text(text, dialect_id, intensity, npc_id)
    
    async def enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        enhancer = await self._get_context_enhancer()
        return await enhancer.enhance_context(context)
    
    async def enhanced_scene_description(self, location_name: str) -> Dict[str, Any]:
        enhancer = await self._get_context_enhancer()
        return await enhancer.generate_scene_description(location_name)
    
    async def get_conflict_lore(self, conflict_id: int) -> List[Dict[str, Any]]:
        ci = await self._get_conflict_integration()
        return await ci.get_conflict_lore(conflict_id)
    
    async def get_faction_conflicts(self, faction_id: int) -> List[Dict[str, Any]]:
        ci = await self._get_conflict_integration()
        return await ci.get_faction_conflicts(faction_id)
    
    async def generate_faction_conflict(self, faction_a_id: int, faction_b_id: int) -> Dict[str, Any]:
        ci = await self._get_conflict_integration()
        return await ci.generate_faction_conflict(faction_a_id, faction_b_id)
    
    async def _get_conflict_integration(self):
        if not self._conflict_integration:
            from lore.integration import ConflictIntegration
            self._conflict_integration = ConflictIntegration(self.user_id, self.conversation_id)
            try:
                governor = await get_central_governance(self.user_id, self.conversation_id)
                self._conflict_integration.set_governor(governor)
            except Exception:
                logger.debug("Could not attach governor to ConflictIntegration")
            await self._conflict_integration.initialize()
            logger.info("Conflict integration initialized")
        return self._conflict_integration

    async def da_get_npc_details(self, npc_id: int = None, npc_name: str = None) -> Dict[str, Any]:
        da = await self._get_npc_data_access()
        return await da.get_npc_details(npc_id=npc_id, npc_name=npc_name)
    
    async def da_get_npc_relationships(self, npc_id: int) -> List[Dict[str, Any]]:
        da = await self._get_npc_data_access()
        return await da.get_npc_relationships(npc_id)
    
    async def da_get_npc_personality(self, npc_id: int) -> Dict[str, Any]:
        da = await self._get_npc_data_access()
        return await da.get_npc_personality(npc_id)
    
    async def da_get_location_with_lore(self, location_id: int) -> Dict[str, Any]:
        da = await self._get_location_data_access()
        return await da.get_location_with_lore(location_id)
    
    async def da_get_comprehensive_location_context(self, location_id: int) -> Dict[str, Any]:
        da = await self._get_location_data_access()
        return await da.get_comprehensive_location_context(location_id)
    
    async def da_get_faction_details(self, faction_id: int) -> Dict[str, Any]:
        da = await self._get_faction_data_access()
        return await da.get_faction_details(faction_id)
    
    async def da_get_entity_knowledge(self, entity_type: str, entity_id: int) -> List[Dict[str, Any]]:
        da = await self._get_knowledge_access()
        return await da.get_entity_knowledge(entity_type, entity_id)
    
    async def da_get_relevant_lore(self, query: str, min_relevance: float = 0.6,
                                   limit: int = 5, lore_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        da = await self._get_knowledge_access()
        return await da.get_relevant_lore(query, min_relevance=min_relevance, limit=limit, lore_types=lore_types)
    
    async def da_generate_available_lore_for_context(self, query_text: str, entity_type: str, entity_id: int, limit: int = 5):
        da = await self._get_knowledge_access()
        return await da.generate_available_lore_for_context(query_text, entity_type, entity_id, limit)
    
    async def _get_master_coordinator(self):
        """Get or initialize the master coordinator."""
        if not self._master_coordinator:
            from lore.coordination import MasterCoordinator
            self._master_coordinator = MasterCoordinator(self.user_id, self.conversation_id)
            logger.info("Master coordinator initialized")
        return self._master_coordinator
    
    async def _get_content_validator(self):
        """Get or initialize the content validator."""
        if not self._content_validator:
            from lore.validation import ContentValidator
            self._content_validator = ContentValidator()
            logger.info("Content validator initialized")
        return self._content_validator
    
    async def _get_relationship_mapper(self):
        """Get or initialize the relationship mapper."""
        if not self._relationship_mapper:
            from lore.mapping import RelationshipMapper
            self._relationship_mapper = RelationshipMapper()
            logger.info("Relationship mapper initialized")
        return self._relationship_mapper
    
    async def _get_unified_trace_system(self):
        """Get or initialize the unified trace system."""
        if not self._unified_trace_system:
            from lore.tracing import UnifiedTraceSystem
            self._unified_trace_system = UnifiedTraceSystem(self.user_id, self.conversation_id)
            logger.info("Unified trace system initialized")
        return self._unified_trace_system
    
    async def _get_npc_data_access(self):
        if not self._npc_data_access:
            from lore.data_access import NPCDataAccess
            self._npc_data_access = NPCDataAccess(self.user_id, self.conversation_id)
            logger.info("NPC data access initialized")
        return self._npc_data_access
    
    async def _get_location_data_access(self):
        if not self._location_data_access:
            from lore.data_access import LocationDataAccess
            self._location_data_access = LocationDataAccess(self.user_id, self.conversation_id)
            logger.info("Location data access initialized")
        return self._location_data_access
    
    async def _get_faction_data_access(self):
        if not self._faction_data_access:
            from lore.data_access import FactionDataAccess
            self._faction_data_access = FactionDataAccess(self.user_id, self.conversation_id)
            logger.info("Faction data access initialized")
        return self._faction_data_access
    
    async def _get_knowledge_access(self):
        if not self._knowledge_access:
            from lore.data_access import LoreKnowledgeAccess
            self._knowledge_access = LoreKnowledgeAccess(self.user_id, self.conversation_id)
            logger.info("Knowledge access initialized")
        return self._knowledge_access
        
    async def _get_lore_generator(self):
        """Get or initialize lore generator."""
        if not self._lore_generator:
            from lore.lore_generator import DynamicLoreGenerator
            self._lore_generator = DynamicLoreGenerator.get_instance(
                self.user_id,
                self.conversation_id
            )
            logger.info("Lore generator initialized")
        return self._lore_generator
        
    async def _get_regional_culture_system(self):
        if not self._regional_culture_system:
            from lore.systems.regional_culture import RegionalCultureSystem
            self._regional_culture_system = RegionalCultureSystem(self.user_id, self.conversation_id)
            await self._regional_culture_system.ensure_initialized()
            logger.info("Regional culture system initialized")
        return self._regional_culture_system
    
    async def _get_national_conflict_system(self):
        """Get or initialize national conflict system."""
        if not hasattr(self, '_national_conflict_system'):
            from lore.matriarchal_lore_system import NationalConflictSystem
            self._national_conflict_system = NationalConflictSystem(self.user_id, self.conversation_id)
            logger.info("National conflict system initialized")
        return self._national_conflict_system
    
    async def _get_religious_distribution_system(self):
        """Get or initialize religious distribution system."""
        if not hasattr(self, '_religious_distribution_system'):
            from lore.matriarchal_lore_system import ReligiousDistributionSystem
            self._religious_distribution_system = ReligiousDistributionSystem(self.user_id, self.conversation_id)
            logger.info("Religious distribution system initialized")
        return self._religious_distribution_system
    
    async def _get_matriarchal_power_framework(self):
        """Get or initialize the matriarchal power structure framework."""
        if not hasattr(self, "_mpf"):
            from lore.frameworks.matriarchal import MatriarchalPowerStructureFramework
            self._mpf = MatriarchalPowerStructureFramework(self.user_id, self.conversation_id)
            
            # Initialize if it has the method
            if hasattr(self._mpf, 'ensure_initialized'):
                await self._mpf.ensure_initialized()
            
            # Register with governance if available
            try:
                governor = await get_central_governance(self.user_id, self.conversation_id)
                self._mpf.governor = governor
                if hasattr(self._mpf, "register_with_governance"):
                    await self._mpf.register_with_governance()
            except Exception as e:
                logger.debug(f"Could not attach/register governance to Matriarchal framework: {e}")
            
            logger.info("MatriarchalPowerStructureFramework initialized")
        return self._mpf

    # ===== MATRIARCHAL FRAMEWORK WRAPPERS =====
    
    async def mpf_generate_core_principles(self) -> Dict[str, Any]:
        mpf = await self._get_matriarchal_power_framework()
        out = await mpf.generate_core_principles()
        try:
            return out.dict()
        except Exception:
            # If it’s already a dict
            return out
    
    async def mpf_generate_hierarchical_constraints(self) -> Dict[str, Any]:
        mpf = await self._get_matriarchal_power_framework()
        out = await mpf.generate_hierarchical_constraints()
        try:
            return out.dict()
        except Exception:
            return out
    
    async def mpf_generate_power_expressions(self, limit: int = 5) -> List[Dict[str, Any]]:
        mpf = await self._get_matriarchal_power_framework()
        out = await mpf.generate_power_expressions()
        if hasattr(out, "__iter__"):
            try:
                return [e.dict() if hasattr(e, "dict") else e for e in out][:limit]
            except Exception:
                return list(out)[:limit]
        return []
    
    async def mpf_apply_power_lens(self, foundation_data: Dict[str, Any]) -> Dict[str, Any]:
        mpf = await self._get_matriarchal_power_framework()
        return await mpf.apply_power_lens(foundation_data)
    
    async def mpf_develop_narrative_through_dialogue(self, narrative_theme: str, initial_scene: str) -> AsyncGenerator[str, None]:
        mpf = await self._get_matriarchal_power_framework()
        async for chunk in mpf.develop_narrative_through_dialogue(narrative_theme, initial_scene):
            yield chunk
    
    # ===== CORE LORE OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world",
        action_description="Generating complete world lore",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def generate_complete_world(self, ctx, environment_desc: str, use_matriarchal_theme: Optional[bool] = None) -> Dict[str, Any]:
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
    
        use_matriarchal = use_matriarchal_theme if use_matriarchal_theme is not None else self.config.enable_matriarchal_theme
    
        # Generate base world using LoreSystem
        lore_system = await self._get_lore_system()
        result = await lore_system.generate_complete_lore(environment_desc)
        
        # If matriarchal theme is enabled, enhance with the framework
        if use_matriarchal:
            mpf = await self._get_matriarchal_power_framework()
            
            # Apply matriarchal lens to the generated world
            result = await mpf.apply_power_lens(result)
            
            # Add matriarchal-specific elements
            result['matriarchal_principles'] = await mpf.generate_core_principles()
            result['power_expressions'] = await mpf.generate_power_expressions()
            result['hierarchical_constraints'] = await mpf.generate_hierarchical_constraints()
    
        self.metrics["operations"] += 1
        self.metrics["last_operation"] = "generate_world"
        return result
            
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_world",
        action_description="Evolving world with narrative event",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def evolve_world_with_event(self, ctx, event_description: str, affected_location_id: Optional[int] = None) -> Dict[str, Any]:
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
    
        # If matriarchal theme is enabled, use the specialized system
        if self.config.enable_matriarchal_theme:
            ms = await self._get_matriarchal_system()
            return await ms.handle_narrative_event(ctx, event_description, affected_location_id=affected_location_id, player_data=None)
    
        # Use LoreDynamicsSystem (with ctx!) and its change tracking wrapper
        result = await self.dynamics_evolve_lore_with_event(ctx, event_description)
    
        # If specific location affected, update local lore as a secondary step
        if affected_location_id:
            local_mgr = await self._get_local_lore_manager()
            location_update = await local_mgr.evolve_location_lore(ctx, affected_location_id, event_description)
            result['location_update'] = location_update
    
            # Record location change (best-effort)
            if self._change_tracking_enabled and location_update:
                await self.record_lore_change(
                    'location',
                    affected_location_id,
                    'update',
                    new_data={'event': event_description, 'updates': location_update}
                )
    
        return result
    
    # ===== NPC LORE OPERATIONS =====
    
    async def get_npc_data(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get NPC data for a list of NPC IDs.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to their data
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import get_npc_data
        return await get_npc_data(npc_ids)
    
    async def determine_relevant_lore_for_npc(self, npc_id: int, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Determine which lore elements are relevant to a specific NPC.
        
        Args:
            npc_id: The ID of the NPC
            context: Optional context dictionary
            
        Returns:
            Dict containing relevant lore elements and their relevance scores
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import determine_relevant_lore
        return await determine_relevant_lore(npc_id, context)
    
    async def integrate_npc_lore(
        self, 
        npc_id: int, 
        relevant_lore: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Integrate relevant lore with an NPC, updating their knowledge and beliefs.
        
        Args:
            npc_id: The ID of the NPC
            relevant_lore: Dictionary of relevant lore elements
            context: Optional context dictionary
            
        Returns:
            Dict containing integration results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import integrate_npc_lore
        return await integrate_npc_lore(npc_id, relevant_lore, context)

    # ===== REGIONAL CULTURE OPERATIONS (wrappers) =====
    
    async def rc_generate_languages(self, ctx, count: int = 5) -> List[Dict[str, Any]]:
        rcs = await self._get_regional_culture_system()
        return await rcs.generate_languages(ctx, count)
    
    async def rc_generate_cultural_norms(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        rcs = await self._get_regional_culture_system()
        return await rcs.generate_cultural_norms(ctx, nation_id)
    
    async def rc_generate_etiquette(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        rcs = await self._get_regional_culture_system()
        return await rcs.generate_etiquette(ctx, nation_id)
    
    async def rc_simulate_cultural_diffusion(self, ctx, nation1_id: int, nation2_id: int, years: int = 50) -> Dict[str, Any]:
        rcs = await self._get_regional_culture_system()
        return await rcs.simulate_cultural_diffusion(ctx, nation1_id, nation2_id, years)
    
    async def rc_evolve_dialect(self, ctx, language_id: int, region_id: int, years: int = 100) -> Dict[str, Any]:
        rcs = await self._get_regional_culture_system()
        return await rcs.evolve_dialect(ctx, language_id, region_id, years)
    
    async def rc_get_nation_culture(self, ctx, nation_id: int) -> Dict[str, Any]:
        rcs = await self._get_regional_culture_system()
        return await rcs.get_nation_culture(ctx, nation_id)
    
    async def rc_summarize_culture(self, nation_id: int, format_type: str = "brief") -> str:
        rcs = await self._get_regional_culture_system()
        return await rcs.summarize_culture(nation_id=nation_id, format_type=format_type)
    
    async def rc_detect_cultural_conflicts(self, nation_id1: int, nation_id2: int) -> Dict[str, Any]:
        rcs = await self._get_regional_culture_system()
        return await rcs.detect_cultural_conflicts(nation_id1=nation_id1, nation_id2=nation_id2)
    
    async def rc_get_all_languages(self) -> List[Dict[str, Any]]:
        rcs = await self._get_regional_culture_system()
        return await rcs.get_all_languages()
    
    async def rc_get_language_details(self, language_id: int) -> Dict[str, Any]:
        rcs = await self._get_regional_culture_system()
        return await rcs.get_language_details(language_id)
    
    async def rc_compare_etiquette(self, nation_id1: int, nation_id2: int, context: str) -> Dict[str, Any]:
        rcs = await self._get_regional_culture_system()
        return await rcs.compare_etiquette(nation_id1, nation_id2, context)
    
    async def rc_generate_diplomatic_protocol(self, nation_id1: int, nation_id2: int) -> Dict[str, Any]:
        rcs = await self._get_regional_culture_system()
        return await rcs.generate_diplomatic_protocol(nation_id1, nation_id2)
    
    async def initialize_npc_lore_knowledge(
        self,
        npc_id: int,
        cultural_background: str,
        faction_affiliations: List[str]
    ) -> Dict[str, Any]:
        """
        Initialize NPC lore knowledge based on background.
        
        Args:
            npc_id: ID of the NPC
            cultural_background: NPC's cultural background
            faction_affiliations: List of faction names
            
        Returns:
            Integration results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_npc_integration()
        ctx = self._create_mock_context(npc_id=npc_id)
        return await integration.initialize_npc_lore_knowledge(
            ctx,
            npc_id,
            cultural_background,
            faction_affiliations
        )
    
    async def process_npc_lore_interaction(self, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Process a lore-related interaction between player and NPC.
        
        Args:
            npc_id: ID of the NPC
            player_input: Player's input/question
            
        Returns:
            Interaction results and NPC response
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_npc_integration()
        ctx = self._create_mock_context(npc_id=npc_id)
        return await integration.process_npc_lore_interaction(ctx, npc_id, player_input)
    
    # ===== LOCATION OPERATIONS =====
    
    async def generate_scene_description(self, location_name: str) -> Dict[str, Any]:
        """
        Generate an atmospheric scene description with integrated lore.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Scene description with lore elements
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import generate_scene_description
        location_data = await self.get_location_context(location_name)
        lore_context = await self._get_lore_system()
        
        return await generate_scene_description(
            location_data,
            lore_context,
            {}
        )
    
    # ===== LOCAL LORE OPERATIONS =====
    
    async def create_urban_myth(self, ctx, params: MythCreationInput) -> int:
        """
        Create an urban myth for a location.
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import create_urban_myth
        from agents import RunContextWrapper
    
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
    
        myth_id = await create_urban_myth(run_ctx, params)
    
        # Record the change
        if myth_id and self._change_tracking_enabled:
            myth_data = {
                'title': getattr(params, 'title', None),
                'description': getattr(params, 'description', None),
                'location_id': getattr(params, 'location_id', None),
                'belief_level': getattr(params, 'belief_level', 0.5)
            }
            await self.record_lore_change('myth', myth_id, 'create', new_data=myth_data)
    
        return myth_id
    
    async def add_local_history(self, ctx, params: HistoryCreationInput) -> int:
        """
        Add local history to a location.
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import add_local_history
        from agents import RunContextWrapper
    
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
    
        return await add_local_history(run_ctx, params)
    
    async def add_landmark(self, ctx, params: LandmarkCreationInput) -> int:
        """
        Add a landmark to a location.
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import add_landmark
        from agents import RunContextWrapper
    
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
    
        return await add_landmark(run_ctx, params)
    
    async def evolve_myth(
        self, ctx,
        myth_id: int,
        evolution_type: EvolutionType,
        causal_factors: Optional[List[str]] = None
    ) -> NarrativeEvolution:
        """
        Evolve an urban myth.
        
        Args:
            ctx: Context object
            myth_id: ID of the myth
            evolution_type: Type of evolution
            causal_factors: Causal factors
            
        Returns:
            NarrativeEvolution results
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import evolve_myth
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await evolve_myth(run_ctx, myth_id, evolution_type, causal_factors)
    
    async def get_location_lore(self, ctx, location_id: int) -> LocationLoreResult:
        """
        Get all lore for a location.
        
        Args:
            ctx: Context object
            location_id: Location ID
            
        Returns:
            LocationLoreResult
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import get_location_lore
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await get_location_lore(run_ctx, location_id)
    
    # ===== EDUCATION SYSTEM OPERATIONS =====
    
    async def generate_educational_systems(self, ctx, count: int = 3) -> List[EducationalSystem]:
        """
        Generate educational systems for the world.
        
        Args:
            ctx: Context object
            count: Number of systems to generate
            
        Returns:
            List of EducationalSystem objects
        """
        manager = await self._get_education_manager()
        from lore.managers.education import generate_educational_systems
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await generate_educational_systems(run_ctx, count)
    
    async def generate_knowledge_traditions(self, ctx, system_id: int, count: int = 5) -> List[KnowledgeTradition]:
        """
        Generate knowledge traditions for an educational system.
        
        Args:
            ctx: Context object
            system_id: Educational system ID
            count: Number of traditions to generate
            
        Returns:
            List of KnowledgeTradition objects
        """
        manager = await self._get_education_manager()
        from lore.managers.education import generate_knowledge_traditions
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await generate_knowledge_traditions(run_ctx, system_id, count)
    
    # ===== RELIGION OPERATIONS =====
    
    async def add_deity(self, ctx, params: DeityParams) -> int:
        manager = await self._get_religion_manager()
        return await manager.add_deity(ctx, params)
        
    async def add_pantheon(self, ctx, params: PantheonParams) -> int:
        manager = await self._get_religion_manager()
        return await manager.add_pantheon(ctx, params)
    
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        manager = await self._get_religion_manager()
        raw = await manager.generate_complete_faith_system(ctx)
        return self._deep_norm_faith_system(raw or {})
    
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        manager = await self._get_religion_manager()
        raw = await manager.distribute_religions(ctx)
        return [self._norm_distribution(d or {}) for d in (raw or [])]
    
    async def generate_ritual(self, ctx, pantheon_id: int, deity_id: Optional[int] = None,
                              purpose: str = "blessing", formality_level: int = 5) -> Dict[str, Any]:
        manager = await self._get_religion_manager()
        ritual = await manager.generate_ritual(ctx, pantheon_id, deity_id, purpose, formality_level)
        # Normalize embedded references if present
        if isinstance(ritual, dict):
            if 'pantheon' in ritual and isinstance(ritual['pantheon'], dict):
                ritual['pantheon'] = self._norm_pantheon(ritual['pantheon'])
            if 'deity' in ritual and isinstance(ritual['deity'], dict):
                ritual['deity'] = self._norm_deity(ritual['deity'])
            if 'practice' in ritual and isinstance(ritual['practice'], dict):
                ritual['practice'] = self._norm_practice(ritual['practice'])
        return ritual
    
    # ===== POLITICS OPERATIONS =====
    
    async def add_nation(
        self, ctx,
        name: str,
        government_type: str,
        description: str,
        relative_power: int = 5,
        matriarchy_level: int = 50,
        **kwargs
    ) -> int:
        """Add a nation to the world."""
        manager = await self._get_politics_manager()
        return await manager.add_nation(
            ctx, name, government_type, description, relative_power,
            matriarchy_level, **kwargs
        )
    
    async def get_all_nations(self, ctx) -> List[Dict[str, Any]]:
        manager = await self._get_politics_manager()
        raw = await manager.get_all_nations(ctx)
        return [self._norm_nation(n or {}) for n in (raw or [])]
    
    async def generate_initial_conflicts(self, ctx, count: int = 3) -> List[Dict[str, Any]]:
        manager = await self._get_politics_manager()
        raw = await manager.generate_initial_conflicts(ctx, count)
        return [self._norm_conflict(c or {}) for c in (raw or [])]
    
    async def simulate_diplomatic_negotiation(self, ctx, nation1_id: int, nation2_id: int, issue: str) -> Dict[str, Any]:
        manager = await self._get_politics_manager()
        res = await manager.simulate_diplomatic_negotiation(ctx, nation1_id, nation2_id, issue)
        # Keep the manager’s structure but normalize embedded nations/conflicts if present
        if isinstance(res, dict):
            if 'nations' in res and isinstance(res['nations'], list):
                res['nations'] = [self._norm_nation(n) for n in res['nations']]
            if 'conflict' in res and isinstance(res['conflict'], dict):
                res['conflict'] = self._norm_conflict(res['conflict'])
        return res
        
    # ===== GEOPOLITICAL OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="geopolitical_operation",
        action_description="Performing geopolitical operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def add_geographic_region(
        self, ctx,
        name: str,
        region_type: str,
        description: str,
        **kwargs
    ) -> int:
        """
        Add a geographic region.
        
        Args:
            ctx: Context object
            name: Region name
            region_type: Type of region
            description: Description
            **kwargs: Additional parameters (climate, resources, governing_faction, etc.)
            
        Returns:
            Region ID
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await GeopoliticalSystemManager.add_geographic_region(
            run_ctx, name, region_type, description, **kwargs
        )
    
    # ===== LORE DYNAMICS OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """
        Generate a random emergent event in the world.
        
        Returns:
            Generated event data with lore updates
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        dynamics = await self._get_lore_dynamics_system()
        return await dynamics.generate_emergent_event(ctx)
    
    async def mature_lore_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """
        Natural evolution of lore over time.
        
        Args:
            days_passed: Number of days to simulate
            
        Returns:
            Maturation summary with changes
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        dynamics = await self._get_lore_dynamics_system()
        ctx = self._create_mock_context()
        return await dynamics.mature_lore_over_time(days_passed)
    
    # ===== CANON OPERATIONS =====
    
    async def ensure_canonical_consistency(self, ctx, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure content is canonically consistent.
        
        Args:
            ctx: Context object
            content: Content to validate
            
        Returns:
            Validation results with any corrections
        """
        canon = await self._get_canon_module()
        return await canon.ensure_canonical_consistency(ctx, content)
    
    async def log_canonical_event(
        self, ctx,
        event_description: str,
        tags: List[str] = None,
        significance: int = 3
    ) -> int:
        """
        Log a canonical event.
        
        Args:
            ctx: Context object
            event_description: Description of the event
            tags: Event tags
            significance: Significance level (1-10)
            
        Returns:
            Event ID
        """
        canon = await self._get_canon_module()
        async with get_db_connection_context() as conn:
            return await canon.log_canonical_event(
                ctx, conn, event_description, tags, significance
            )
            
    async def initialize_nation_culture(
        self,
        ctx,
        nation_id: int,
        language_count: int = 2,
        generate_norms: bool = True,
        generate_etiquette: bool = True
    ) -> Dict[str, Any]:
        """Initialize cultural system for a nation via RegionalCultureSystem."""
        results: Dict[str, Any] = {}
    
        if language_count > 0:
            results['languages'] = await self.rc_generate_languages(ctx, language_count)
    
        if generate_norms:
            results['cultural_norms'] = await self.rc_generate_cultural_norms(ctx, nation_id)
    
        if generate_etiquette:
            results['etiquette'] = await self.rc_generate_etiquette(ctx, nation_id)
    
        await self.record_lore_change('nation_culture', nation_id, 'initialize', new_data=results)
        return results
    
    async def evolve_world_culture(
        self,
        ctx,
        years: int = 100,
        include_diffusion: bool = True,
        include_dialects: bool = True
    ) -> Dict[str, Any]:
        """Run culture evolution across nations: diffusion and dialects."""
        results = {'diffusions': [], 'dialects': [], 'summary': {}}
    
        async with get_db_connection_context() as conn:
            relations = await conn.fetch("""
                SELECT DISTINCT nation1_id, nation2_id
                FROM InternationalRelations
                WHERE relationship_quality >= 5
                LIMIT 5
            """)
    
            if include_diffusion and relations:
                for rel in relations[:3]:
                    d = await self.rc_simulate_cultural_diffusion(ctx, rel['nation1_id'], rel['nation2_id'], years)
                    results['diffusions'].append(d)
    
            if include_dialects:
                langs = await conn.fetch("""
                    SELECT id, primary_regions
                    FROM Languages
                    WHERE array_length(primary_regions, 1) > 0
                    LIMIT 3
                """)
                for l in langs:
                    region_ids = l['primary_regions'] or []
                    if region_ids:
                        evo = await self.rc_evolve_dialect(ctx, l['id'], region_ids[0], years)
                        results['dialects'].append(evo)
    
        results['summary'] = {
            'years_simulated': years,
            'diffusions_created': len(results['diffusions']),
            'dialects_evolved': len(results['dialects']),
        }
        return results
    
    # ===== PUBLIC CHANGE TRACKING API =====
    
    async def record_lore_change(
        self,
        element_type: str,
        element_id: int,
        operation: str,
        new_data: Optional[Dict[str, Any]] = None,
        old_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Public API to record a lore change.
        
        Args:
            element_type: Type of element (location, nation, myth, etc.)
            element_id: ID of the element
            operation: Operation type (create, update, delete)
            new_data: New data (for create/update)
            old_data: Old data (for update/delete)
            
        Returns:
            True if change was recorded successfully
        """
        # Normalize common cultural shorthand to stable types
        cultural_map = {
            'language': 'language',
            'languages': 'language',
            'cultural_norm': 'cultural_norm',
            'cultural_norms': 'cultural_norm',
            'etiquette': 'etiquette',
            'dialect': 'dialect',
            'language_dialect': 'dialect',
            'cultural_element': 'cultural_element',
        }
        et_lower = (element_type or '').lower()
        element_type = cultural_map.get(et_lower, element_type)

        if not self._change_tracking_enabled:
            return False
        
        # Determine scope keys affected
        scope_keys = self._get_affected_scope_keys(
            element_type, 
            element_id, 
            new_data or old_data or {}
        )
        
        # Build change record
        change_record = {
            'element_type': element_type,
            'element_id': element_id,
            'operation': operation,
            'changed_fields': [],
            'old_value': old_data,
            'new_value': new_data,
            'timestamp': time.time()
        }
        
        # Determine changed fields
        if old_data and new_data:
            all_fields = set(old_data.keys()) | set(new_data.keys())
            change_record['changed_fields'] = [
                field for field in all_fields
                if old_data.get(field) != new_data.get(field)
            ]
        elif new_data:
            change_record['changed_fields'] = list(new_data.keys())
        elif old_data:
            change_record['changed_fields'] = list(old_data.keys())
        
        # Persist the change
        await self._persist_change(change_record, scope_keys)
        
        # Update in-memory tracking
        element_key = f"{element_type}:{element_id}"
        if operation == 'delete':
            self._element_snapshots.pop(element_key, None)
        elif new_data:
            self._element_snapshots[element_key] = new_data.copy()
        
        # Update change logs for affected scopes
        for scope_key in scope_keys:
            if scope_key not in self._change_log:
                self._change_log[scope_key] = []
            self._change_log[scope_key].append(change_record)
            
            # Update last changed timestamp
            self._bundle_last_changed[scope_key] = change_record['timestamp']
            
            # Invalidate cached bundle
            self._scene_bundle_cache.pop(scope_key, None)
            # Clear monotonic timestamp by cache key
            self._bundle_cached_at.pop(scope_key, None)

        await self._invalidate_scope_keys(scope_keys)
        
        return True
    
    async def clear_change_history(
        self,
        before_ts: Optional[float] = None,
        scope_key: Optional[str] = None
    ) -> int:
        """
        Clear old change history to save space.
        
        Args:
            before_ts: Clear changes before this timestamp
            scope_key: Optional specific scope to clear
            
        Returns:
            Number of records cleared
        """
        count = 0
        
        try:
            async with get_db_connection_context() as conn:
                if scope_key:
                    # Clear for specific scope
                    result = await conn.execute("""
                        DELETE FROM LoreChangeLog
                        WHERE conversation_id = $1
                          AND $2 = ANY(scope_keys)
                          AND ($3 IS NULL OR timestamp < $3)
                    """,
                        self.conversation_id,
                        scope_key,
                        datetime.fromtimestamp(before_ts) if before_ts else None
                    )
                else:
                    # Clear all old changes
                    if not before_ts:
                        before_ts = time.time() - (7 * 24 * 3600)  # Default: 7 days old
                    
                    result = await conn.execute("""
                        DELETE FROM LoreChangeLog
                        WHERE conversation_id = $1
                          AND timestamp < $2
                    """,
                        self.conversation_id,
                        datetime.fromtimestamp(before_ts)
                    )
                
                count = int(result.split()[-1]) if result else 0
                
        except Exception as e:
            logger.warning(f"Failed to clear change history: {e}")
        
        # Clear in-memory logs if requested
        if scope_key and scope_key in self._change_log:
            if before_ts:
                self._change_log[scope_key] = [
                    c for c in self._change_log[scope_key]
                    if c['timestamp'] >= before_ts
                ]
            else:
                del self._change_log[scope_key]
        
        return count
    
    def enable_change_tracking(self, enabled: bool = True) -> None:
        """Enable or disable change tracking."""
        self._change_tracking_enabled = enabled
        logger.info(f"Change tracking {'enabled' if enabled else 'disabled'}")
    
    async def get_change_history(
        self,
        element_type: Optional[str] = None,
        element_id: Optional[int] = None,
        since_ts: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get change history for debugging/auditing.
    
        Args:
            element_type: Optional filter by element type
            element_id: Optional filter by element ID
            since_ts: Optional filter by timestamp
            limit: Maximum number of records to return
    
        Returns:
            List of change records
        """
        changes = []
    
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT element_type, element_id, operation,
                           changed_fields, old_value, new_value,
                           scope_keys, EXTRACT(EPOCH FROM timestamp) as timestamp
                    FROM LoreChangeLog
                    WHERE conversation_id = $1
                """
                params = [self.conversation_id]
    
                if element_type:
                    query += f" AND element_type = ${len(params) + 1}"
                    params.append(element_type)
    
                if element_id is not None:
                    query += f" AND element_id = ${len(params) + 1}"
                    params.append(element_id)
    
                if since_ts:
                    query += f" AND timestamp > ${len(params) + 1}"
                    params.append(datetime.fromtimestamp(since_ts))
    
                query += f" ORDER BY timestamp DESC LIMIT ${len(params) + 1}"
                params.append(limit)
    
                rows = await conn.fetch(query, *params)
    
                for row in rows:
                    changes.append({
                        'element_type': row['element_type'],
                        'element_id': row['element_id'],
                        'operation': row['operation'],
                        'changed_fields': row['changed_fields'] or [],
                        'old_value': row['old_value'],
                        'new_value': row['new_value'],
                        'scope_keys': row['scope_keys'],
                        'timestamp': row['timestamp']
                    })
    
        except Exception as e:
            logger.warning(f"Failed to get change history: {e}")
    
        return changes

    # ===== NORMALIZATION HELPERS =====
    
    def _coalesce_id(self, obj: Dict[str, Any], *candidates: str) -> Optional[int]:
        for c in candidates:
            if c in obj and obj[c] is not None:
                try:
                    return int(obj[c])
                except Exception:
                    pass
        return None
    
    def _coalesce_name(self, obj: Dict[str, Any], *candidates: str) -> Optional[str]:
        for c in candidates:
            v = obj.get(c)
            if isinstance(v, str) and v.strip():
                return v
        return None
    
    def _norm_nation(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id', 'nation_id'),
            'name': self._coalesce_name(raw, 'name', 'nation_name'),
            'government': raw.get('government_type') or raw.get('government') or 'unknown',
            'culture': raw.get('culture') or raw.get('lore_context') or {},
            'matriarchy_level': raw.get('matriarchy_level')
        }
    
    def _norm_conflict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        # Used for both Conflicts and NationalConflicts
        cid = self._coalesce_id(raw, 'id', 'conflict_id')
        ctype = raw.get('type') or raw.get('conflict_type') or 'standard'
        desc = raw.get('description') or ''
        intensity = raw.get('intensity')
        if intensity is None and 'severity' in raw:
            try:
                intensity = max(0.0, min(1.0, float(raw['severity']) / 10.0))
            except Exception:
                intensity = 0.5
        intensity = intensity if isinstance(intensity, (float, int)) else 0.5
        phase = raw.get('phase') or raw.get('status') or 'active'
        res = raw.get('resolution_status')
        if not res:
            res = 'resolved' if str(phase).lower() == 'resolved' else 'ongoing'
    
        # Stakeholders synthesis
        stakeholders = raw.get('stakeholders')
        if not stakeholders:
            nations = raw.get('involved_nations') or []
            stakeholders = [{'type': 'nation', 'id': int(n)} for n in nations if n is not None]
    
        return {
            'id': cid,
            'type': ctype,
            'description': desc[:150],
            'stakeholders': stakeholders[:3] if isinstance(stakeholders, list) else [],
            'intensity': float(intensity),
            'phase': phase,
            'resolution_status': res
        }
    
    def _norm_religion(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id', 'religion_id'),
            'name': self._coalesce_name(raw, 'name', 'religion_name'),
            'deities': raw.get('deity_names') or raw.get('deities') or [],
            'beliefs': (raw.get('core_beliefs') or raw.get('beliefs') or '')[:200],
            'influence': raw.get('influence') or raw.get('influence_level') or 0.0
        }
    
    def _norm_pantheon(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id', 'pantheon_id'),
            'name': self._coalesce_name(raw, 'name', 'pantheon_name'),
            'description': raw.get('description'),
            'matriarchal_elements': raw.get('matriarchal_elements')
        }
    
    def _norm_deity(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id', 'deity_id'),
            'name': self._coalesce_name(raw, 'name', 'deity_name'),
            'domains': raw.get('domains') or [],
            'description': raw.get('description')
        }
    
    def _norm_practice(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id', 'practice_id'),
            'name': self._coalesce_name(raw, 'name', 'practice_name'),
            'practice_type': raw.get('practice_type'),
            'description': raw.get('description'),
            'purpose': raw.get('purpose')
        }
    
    def _norm_holy_site(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id', 'holy_site_id'),
            'name': self._coalesce_name(raw, 'name', 'site_name'),
            'location_id': self._coalesce_id(raw, 'location_id'),
            'description': raw.get('description')
        }
    
    def _norm_religious_order(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id', 'order_id'),
            'name': self._coalesce_name(raw, 'name', 'order_name'),
            'doctrine': raw.get('doctrine'),
            'hierarchy': raw.get('hierarchy')
        }
    
    def _norm_distribution(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': self._coalesce_id(raw, 'id'),
            'nation_id': self._coalesce_id(raw, 'nation_id'),
            'state_religion': bool(raw.get('state_religion', False)),
            'primary_pantheon_id': self._coalesce_id(raw, 'primary_pantheon_id'),
            'pantheon_distribution': raw.get('pantheon_distribution') or {},
            'religiosity_level': raw.get('religiosity_level'),
            'religious_tolerance': raw.get('religious_tolerance'),
            'religious_leadership': raw.get('religious_leadership'),
            'religious_laws': raw.get('religious_laws') or {},
            'religious_holidays': raw.get('religious_holidays') or [],
            'religious_conflicts': raw.get('religious_conflicts') or [],
            'religious_minorities': raw.get('religious_minorities') or []
        }
    
    def _norm_list(self, items: Optional[List[Dict[str, Any]]], norm_fn) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        out = []
        for it in items:
            try:
                out.append(norm_fn(it or {}))
            except Exception:
                # best-effort: skip malformed entries
                continue
        return out
    
    def _deep_norm_faith_system(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a full faith system payload returned by ReligionManager."""
        if not isinstance(data, dict):
            return {}
        out = dict(data)
    
        if 'pantheons' in out:
            out['pantheons'] = self._norm_list(out['pantheons'], self._norm_pantheon)
        if 'deities' in out:
            out['deities'] = self._norm_list(out['deities'], self._norm_deity)
        if 'practices' in out:
            out['practices'] = self._norm_list(out['practices'], self._norm_practice)
        if 'holy_sites' in out:
            out['holy_sites'] = self._norm_list(out['holy_sites'], self._norm_holy_site)
        if 'orders' in out:
            out['orders'] = self._norm_list(out['orders'], self._norm_religious_order)
        if 'religious_conflicts' in out:
            out['religious_conflicts'] = self._norm_list(out['religious_conflicts'], self._norm_conflict)
        return out

# ===== CONVENIENCE FUNCTIONS =====

async def get_lore_orchestrator(user_id: int, conversation_id: int, config: Optional[OrchestratorConfig] = None) -> LoreOrchestrator:
    """
    Get a lore orchestrator instance, initializing if needed.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        config: Optional configuration
        
    Returns:
        Initialized LoreOrchestrator instance
    """
    orchestrator = LoreOrchestrator.get_instance(user_id, conversation_id, config)
    if not orchestrator.initialized and orchestrator.config.auto_initialize:
        await orchestrator.initialize()
    return orchestrator


async def generate_world(user_id: int, conversation_id: int, environment_desc: str, **kwargs) -> Dict[str, Any]:
    """
    Quick function to generate a complete world.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        environment_desc: Environment description
        **kwargs: Additional options
        
    Returns:
        Complete world lore
    """
    orchestrator = await get_lore_orchestrator(user_id, conversation_id)
    ctx = orchestrator._create_mock_context()
    return await orchestrator.generate_complete_world(ctx, environment_desc, **kwargs)


async def evolve_world(user_id: int, conversation_id: int, event_description: str, **kwargs) -> Dict[str, Any]:
    """
    Quick function to evolve the world with an event.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        event_description: Event description
        **kwargs: Additional options
        
    Returns:
        Evolution results
    """
    orchestrator = await get_lore_orchestrator(user_id, conversation_id)
    ctx = orchestrator._create_mock_context()
    return await orchestrator.evolve_world_with_event(ctx, event_description, **kwargs)

# ===== MODULE INITIALIZATION =====

def setup_lore_orchestrator():
    """Setup function for module initialization."""
    logging.basicConfig(level=logging.INFO)
    logger.info("FULLY INTEGRATED Lore Orchestrator loaded with all modules: education, geopolitical, local lore, politics, religion, and world lore management")
    logger.info("ENHANCED with scene bundle support for optimized context assembly")


# Run setup on module import
setup_lore_orchestrator()
