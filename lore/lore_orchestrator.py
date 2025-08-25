"""
Lore Orchestrator - ENHANCED with Scene Bundle Support
=====================================================
Master orchestrator for all lore operations across the Nyx game.
Acts as the single entry point for external systems to interact with lore.

FULLY INTEGRATED: Includes education, geopolitical, local lore, politics, religion, 
and world lore managers with all their specialized functionality.

NEW: Scene-scoped bundle methods for optimized context assembly.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set, AsyncGenerator, Protocol
from datetime import datetime, timedelta
import json
from enum import Enum
from dataclasses import dataclass, field
import os
import asyncpg
import hashlib
import time

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

logger = logging.getLogger(__name__)

# Database connection
DB_DSN = os.getenv("DB_DSN")

# Singleton instance storage
_ORCHESTRATOR_INSTANCES: Dict[Tuple[int, int], "LoreOrchestrator"] = {}


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
    canonical_rules: List[Union[str, Dict[str, str]]] = field(default_factory=list)  # Can be strings or {text, source} dicts
    nations: List[Dict[str, Any]] = field(default_factory=list)
    religions: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    myths: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'location': self.location,
            'world': self.world,
            'canonical_rules': self.canonical_rules,
            'nations': self.nations,
            'religions': self.religions,
            'conflicts': self.conflicts,
            'myths': self.myths
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
        
        Args:
            scope: SceneScope containing location_id, npc_ids, topics, lore_tags, etc.
            
        Returns:
            Dictionary with:
                - section: 'lore' (for consistent merging)
                - anchors: IDs of core entities in the bundle
                - data: Scene-relevant lore data (SceneBundleData)
                - canonical: Boolean indicating if contains canonical data
                - last_changed_at: Timestamp of last change
                - version: Version string for cache validation
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
        
        if hasattr(scope, 'location_id') and scope.location_id:
            tasks.append(('location', self._fetch_location_lore_for_bundle(scope.location_id)))
            tasks.append(('religions', self._fetch_religions_for_location(scope.location_id)))
            tasks.append(('myths', self._fetch_myths_for_location(scope.location_id)))
        
        if hasattr(scope, 'lore_tags') and scope.lore_tags:
            tasks.append(('world', self._fetch_world_lore_for_bundle(list(scope.lore_tags)[:10])))
        
        if hasattr(scope, 'nation_ids') and scope.nation_ids:
            tasks.append(('nations', self._fetch_nations_for_bundle(list(scope.nation_ids)[:5])))
        
        if hasattr(scope, 'conflict_ids') and scope.conflict_ids:
            tasks.append(('conflicts', self._fetch_conflicts_for_bundle(list(scope.conflict_ids)[:5])))
        
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
        
        # Cache the bundle with lock for race safety
        async with self._cache_lock:
            self._cache_bundle(cache_key, result)
        
        return result
    
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

    async def get_location_context(self, location_name: str) -> Dict[str, Any]:
        """
        Retrieve a lightweight location context by name.
        Used by legacy call sites that expect a dict with key fields.
        """
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT location_id, location_name, description,
                           nation_id, governance, culture, population,
                           canonical_rules, tags
                    FROM Locations
                    WHERE location_name = $1
                    LIMIT 1
                """, location_name)
    
                if not row:
                    return {}
    
                data = {
                    'id': row['location_id'],
                    'name': row['location_name'],
                    'description': (row['description'] or '')[:200],
                    'nation_id': row['nation_id'],
                    'governance': row['governance'] or {},
                    'culture': row['culture'] or {},
                    'population': row['population'] or 0,
                    'canonical_rules': row['canonical_rules'] or [],
                    'tags': row['tags'] or []
                }
    
                # Optional enrichments to mirror bundle fetches (best-effort)
                landmarks = await conn.fetch("""
                    SELECT landmark_id, name, significance
                    FROM Landmarks
                    WHERE location_id = $1
                    ORDER BY significance DESC
                    LIMIT 3
                """, row['location_id'])
                if landmarks:
                    data['landmarks'] = [
                        {'id': lm['landmark_id'], 'name': lm['name'], 'significance': lm['significance']}
                        for lm in landmarks
                    ]
    
                events = await conn.fetch("""
                    SELECT event_name, description, event_date
                    FROM Events
                    WHERE location = $1
                    ORDER BY event_date DESC
                    LIMIT 2
                """, row['location_name'])
                if events:
                    data['recent_events'] = [
                        {
                            'name': e['event_name'],
                            'description': (e['description'] or '')[:100],
                            'date': e['event_date'].isoformat() if e['event_date'] else None
                        }
                        for e in events
                    ]
    
                return data
    
        except Exception as e:
            logger.debug(f"get_location_context failed for '{location_name}': {e}")
            return {}
        
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
    
    def _get_affected_scope_keys(
        self, 
        element_type: str, 
        element_id: int,
        element_data: Dict[str, Any]
    ) -> List[str]:
        """
        Determine which scope keys are affected by a change.
        
        Args:
            element_type: Type of element
            element_id: ID of element
            element_data: Element data
            
        Returns:
            List of affected scope keys
        """
        scope_keys = []
        
        # Generate scope keys based on element type
        if element_type == 'location':
            # This affects scopes that include this location
            scope = type('Scope', (), {
                'location_id': element_id,
                'npc_ids': set(),
                'lore_tags': set(),
                'topics': set(),
                'conflict_ids': set(),
                'nation_ids': set(),
                'link_hints': {}
            })()
            scope_keys.append(self._generate_scene_cache_key(scope))
            
        elif element_type == 'nation':
            # Affects scopes that include this nation
            scope = type('Scope', (), {
                'location_id': None,
                'npc_ids': set(),
                'lore_tags': set(),
                'topics': set(),
                'conflict_ids': set(),
                'nation_ids': {element_id},
                'link_hints': {}
            })()
            scope_keys.append(self._generate_scene_cache_key(scope))
            
        # Add more sophisticated scope detection based on relationships
        # For example, if a myth changes, find all locations that reference it
        
        return scope_keys
    
    # ===== SCENE BUNDLE HELPER METHODS =====
    
    def _generate_scene_cache_key(self, scope: Any) -> str:
        to_key = getattr(scope, "to_key", None)
        if callable(to_key):
            return hashlib.md5(to_key().encode()).hexdigest()
    
        key_parts = []
        
        # Only add location if not None
        if getattr(scope, 'location_id', None) is not None:
            key_parts.append(f"loc_{scope.location_id}")
        
        if hasattr(scope, 'npc_ids') and scope.npc_ids:
            s = sorted(scope.npc_ids)
            npc_str = "_".join(str(nid) for nid in s[:5])
            # Add count to avoid collisions from truncation
            key_parts.append(f"npcs_{npc_str}+n={len(s)}")
        
        if hasattr(scope, 'lore_tags') and scope.lore_tags:
            s = sorted(scope.lore_tags)
            tag_str = "_".join(s[:5])
            # Add count to avoid collisions
            key_parts.append(f"tags_{tag_str}+n={len(s)}")
        
        if hasattr(scope, 'topics') and scope.topics:
            s = sorted(scope.topics)
            topic_str = "_".join(s[:3])
            # Add count to avoid collisions
            key_parts.append(f"topics_{topic_str}+n={len(s)}")
        
        if hasattr(scope, 'nation_ids') and scope.nation_ids:
            s = sorted(scope.nation_ids)
            nation_str = "_".join(str(nid) for nid in s[:3])
            key_parts.append(f"nations_{nation_str}+n={len(s)}")
        
        if hasattr(scope, 'conflict_ids') and scope.conflict_ids:
            s = sorted(scope.conflict_ids)
            conflict_str = "_".join(str(cid) for cid in s[:3])
            key_parts.append(f"conflicts_{conflict_str}+n={len(s)}")
        
        if not key_parts:
            key_parts.append("empty")
    
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
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
        """Fetch location-specific lore for a bundle."""
        try:
            location_lore = {}
            
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
                
                # Get location data
                location = await conn.fetchrow("""
                    SELECT location_id, location_name, description, 
                           nation_id, governance, culture, population,
                           canonical_rules, tags
                    FROM Locations
                    WHERE location_id = $1
                """, location_id)
                
                if location:
                    location_lore = {
                        'id': location['location_id'],
                        'name': location['location_name'],
                        'description': (location['description'] or '')[:200],
                        'nation_id': location['nation_id'],
                        'governance': location['governance'] or {},
                        'culture': location['culture'] or {},
                        'population': location['population'] or 0,
                        'canonical_rules': location['canonical_rules'] or [],
                        'tags': location['tags'] or []
                    }
                    
                    # Track changes if enabled
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('location', location_id, location_lore)
                        await self._track_element_change('location', location_id, location_lore, scope_keys)
                    
                    # Get associated landmarks
                    landmarks = await conn.fetch("""
                        SELECT landmark_id, name, significance
                        FROM Landmarks
                        WHERE location_id = $1
                        ORDER BY significance DESC
                        LIMIT 3
                    """, location_id)
                    
                    if landmarks:
                        location_lore['landmarks'] = [
                            {
                                'id': lm['landmark_id'],
                                'name': lm['name'],
                                'significance': lm['significance']
                            }
                            for lm in landmarks
                        ]
                    
                    # Get recent historical events
                    events = await conn.fetch("""
                        SELECT event_name, description, event_date
                        FROM Events
                        WHERE location = $1
                        ORDER BY event_date DESC
                        LIMIT 2
                    """, location['location_name'])
                    
                    if events:
                        location_lore['recent_events'] = [
                            {
                                'name': e['event_name'],
                                'description': (e['description'] or '')[:100],
                                'date': e['event_date'].isoformat() if e['event_date'] else None
                            }
                            for e in events
                        ]
            
            return location_lore
            
        except Exception as e:
            logger.debug(f"Could not fetch location lore: {e}")
            # Fallback to simpler approach
            try:
                # Try the existing method as fallback
                location_name = f"location_{location_id}"
                context = await self.get_location_context(location_name)
                return {
                    'id': location_id,
                    'description': context.get('description', '')[:200],
                    'governance': context.get('governance', {}),
                    'culture': context.get('culture', {}),
                    'canonical_rules': context.get('canonical_rules', [])
                }
            except:
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
        """Fetch nation data for bundle using batch query."""
        try:
            nations = []
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
                
                # Batch query using ANY()
                rows = await conn.fetch("""
                    SELECT nation_id, nation_name, government_type, 
                           culture, lore_context
                    FROM Nations
                    WHERE nation_id = ANY($1::int[])
                """, nation_ids[:5])
                
                for nation in rows:
                    nation_data = {
                        'id': nation['nation_id'],
                        'name': nation['nation_name'],
                        'government': nation['government_type'],
                        'culture': nation['culture']
                    }
                    nations.append(nation_data)
                    
                    # Track changes
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('nation', nation['nation_id'], nation_data)
                        await self._track_element_change('nation', nation['nation_id'], nation_data, scope_keys)
            
            return nations
        except Exception as e:
            logger.debug(f"Could not fetch nations: {e}")
            return []
    
    async def _fetch_religions_for_location(self, location_id: int) -> List[Dict[str, Any]]:
        """Fetch religions active in a location."""
        try:
            religions = []
            
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
                
                # Query religions active in this location
                result = await conn.fetch("""
                    SELECT DISTINCT r.religion_id, r.religion_name, 
                           r.deity_names, r.core_beliefs, r.sacred_texts,
                           nrd.influence_level
                    FROM Religions r
                    LEFT JOIN NationReligionDistribution nrd ON r.religion_id = nrd.religion_id
                    LEFT JOIN Locations l ON l.nation_id = nrd.nation_id
                    WHERE l.location_id = $1 AND nrd.influence_level > 0.1
                    ORDER BY nrd.influence_level DESC
                    LIMIT 5
                """, location_id)
                
                for row in result:
                    religion_data = {
                        'id': row['religion_id'],
                        'name': row['religion_name'],
                        'deities': (row['deity_names'] or [])[:3],  # Guard against None
                        'beliefs': (row['core_beliefs'] or '')[:100],
                        'influence': row['influence_level'] or 0.5
                    }
                    religions.append(religion_data)
                    
                    # Track changes
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('religion', row['religion_id'], religion_data)
                        await self._track_element_change('religion', row['religion_id'], religion_data, scope_keys)
            
            return religions
        except Exception as e:
            logger.debug(f"Could not fetch religions: {e}")
            return []
    
    async def _fetch_conflicts_for_bundle(self, conflict_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch conflict data for bundle using batch query."""
        try:
            conflicts = []
            
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
                
                # Batch query using ANY()
                rows = await conn.fetch("""
                    SELECT conflict_id, conflict_type, description,
                           stakeholders, intensity, phase, resolution_status
                    FROM Conflicts
                    WHERE conflict_id = ANY($1::int[])
                """, conflict_ids[:5])
                
                for conflict in rows:
                    conflict_data = {
                        'id': conflict['conflict_id'],
                        'type': conflict['conflict_type'],
                        'description': (conflict['description'] or '')[:150],
                        'stakeholders': (conflict['stakeholders'] or [])[:3],
                        'intensity': conflict['intensity'] or 0.5,
                        'phase': conflict['phase'] or 'active',
                        'resolution_status': conflict['resolution_status'] or 'ongoing'
                    }
                    conflicts.append(conflict_data)
                    
                    # Track changes
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('conflict', conflict['conflict_id'], conflict_data)
                        await self._track_element_change('conflict', conflict['conflict_id'], conflict_data, scope_keys)
            
            return conflicts
        except Exception as e:
            logger.debug(f"Could not fetch conflicts: {e}")
            return []
    
    async def _fetch_myths_for_location(self, location_id: int) -> List[Dict[str, Any]]:
        """Fetch urban myths for a location."""
        try:
            myths = []
            
            async with get_db_connection_context() as conn:
                self.metrics['db_roundtrips'] = self.metrics.get('db_roundtrips', 0) + 1
                
                # Query urban myths for this location
                result = await conn.fetch("""
                    SELECT um.myth_id, um.title, um.description, 
                           um.origin_period, um.belief_level, um.variants
                    FROM UrbanMyths um
                    WHERE um.location_id = $1
                    ORDER BY um.belief_level DESC
                    LIMIT 5
                """, location_id)
                
                for row in result:
                    myth_data = {
                        'id': row['myth_id'],
                        'title': row['title'],
                        'description': (row['description'] or '')[:200],
                        'origin': row['origin_period'] or 'unknown',
                        'belief_level': row['belief_level'] or 0.3,
                        'has_variants': bool(row['variants'])
                    }
                    myths.append(myth_data)
                    
                    # Track changes
                    if self._change_tracking_enabled:
                        scope_keys = self._get_affected_scope_keys('myth', row['myth_id'], myth_data)
                        await self._track_element_change('myth', row['myth_id'], myth_data, scope_keys)
            
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
                        WHERE location_id = $1
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
                            WHERE nation_id = $1
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
        """Get or initialize NPC integration."""
        if not self._npc_integration:
            from lore.integration import NPCLoreIntegration
            self._npc_integration = NPCLoreIntegration()
            logger.info("NPC integration initialized")
        return self._npc_integration
    
    async def _get_context_enhancer(self):
        """Get or initialize context enhancer."""
        if not self._context_enhancer:
            from lore.integration import ContextEnhancer
            self._context_enhancer = ContextEnhancer()
            logger.info("Context enhancer initialized")
        return self._context_enhancer
    
    async def _get_world_lore_manager(self):
        """Get or initialize the world lore manager."""
        if not self._world_lore_manager:
            from lore.managers.world_lore import WorldLoreManager
            self._world_lore_manager = WorldLoreManager(self.user_id, self.conversation_id)
            await self._world_lore_manager.ensure_initialized()
            logger.info("World lore manager initialized")
        return self._world_lore_manager
    
    async def _get_lore_dynamics_system(self):
        """Get or initialize the lore dynamics system."""
        if not hasattr(self, '_lore_dynamics_system'):
            from lore.systems.dynamics import LoreDynamicsSystem
            self._lore_dynamics_system = LoreDynamicsSystem(self.user_id, self.conversation_id)
            logger.info("Lore dynamics system initialized")
        return self._lore_dynamics_system
    
    async def _get_lore_update_system(self):
        """Get or initialize the lore update system."""
        if not hasattr(self, '_lore_update_system'):
            from lore.systems.lore_update import LoreUpdateSystem
            self._lore_update_system = LoreUpdateSystem(self.user_id, self.conversation_id)
            logger.info("Lore update system initialized")
        return self._lore_update_system
    
    async def _get_matriarchal_system(self):
        """Get or initialize the matriarchal lore system."""
        if not hasattr(self, '_matriarchal_system'):
            from lore.matriarchal_lore_system import MatriarchalLoreSystem
            self._matriarchal_system = MatriarchalLoreSystem(self.user_id, self.conversation_id)
            await self._matriarchal_system.initialize()
            logger.info("Matriarchal lore system initialized")
        return self._matriarchal_system
    
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
    
    async def _get_conflict_integration(self):
        """Get or initialize conflict integration."""
        if not self._conflict_integration:
            from lore.integration import ConflictIntegration
            self._conflict_integration = ConflictIntegration()
            logger.info("Conflict integration initialized")
        return self._conflict_integration
    
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
        """Get or initialize NPC data access."""
        if not self._npc_data_access:
            from lore.data_access import NPCDataAccess
            self._npc_data_access = NPCDataAccess()
            logger.info("NPC data access initialized")
        return self._npc_data_access
    
    async def _get_location_data_access(self):
        """Get or initialize location data access."""
        if not self._location_data_access:
            from lore.data_access import LocationDataAccess
            self._location_data_access = LocationDataAccess()
            logger.info("Location data access initialized")
        return self._location_data_access
    
    async def _get_faction_data_access(self):
        """Get or initialize faction data access."""
        if not self._faction_data_access:
            from lore.data_access import FactionDataAccess
            self._faction_data_access = FactionDataAccess()
            logger.info("Faction data access initialized")
        return self._faction_data_access
    
    async def _get_knowledge_access(self):
        """Get or initialize knowledge access."""
        if not self._knowledge_access:
            from lore.data_access import LoreKnowledgeAccess
            self._knowledge_access = LoreKnowledgeAccess()
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
        """Get or initialize regional culture system."""
        if not self._regional_culture_system:
            from lore.systems.regional_culture import RegionalCultureSystem
            self._regional_culture_system = RegionalCultureSystem(self.user_id, self.conversation_id)
            await self._regional_culture_system.initialize_tables()
            logger.info("Regional culture system initialized")
        return self._regional_culture_system
    
    async def _get_national_conflict_system(self):
        """Get or initialize national conflict system."""
        if not hasattr(self, '_national_conflict_system'):
            from lore.systems.national_conflict import NationalConflictSystem
            self._national_conflict_system = NationalConflictSystem(self.user_id, self.conversation_id)
            logger.info("National conflict system initialized")
        return self._national_conflict_system
    
    async def _get_religious_distribution_system(self):
        """Get or initialize religious distribution system."""
        if not hasattr(self, '_religious_distribution_system'):
            from lore.systems.religious_distribution import ReligiousDistributionSystem
            self._religious_distribution_system = ReligiousDistributionSystem(self.user_id, self.conversation_id)
            logger.info("Religious distribution system initialized")
        return self._religious_distribution_system
    
    async def _get_matriarchal_power_framework(self):
        """Get or initialize matriarchal power framework."""
        if not hasattr(self, '_matriarchal_power_framework'):
            from lore.frameworks.matriarchal_power import MatriarchalPowerFramework
            self._matriarchal_power_framework = MatriarchalPowerFramework(self.user_id, self.conversation_id)
            logger.info("Matriarchal power framework initialized")
        return self._matriarchal_power_framework
    
    # ===== CORE LORE OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world",
        action_description="Generating complete world lore",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def generate_complete_world(self, ctx, environment_desc: str, use_matriarchal_theme: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate a complete world with all lore components.
        
        Args:
            ctx: Context object
            environment_desc: Description of the environment/setting
            use_matriarchal_theme: Override config setting for matriarchal theme
            
        Returns:
            Complete world lore package
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        # Determine theme
        use_matriarchal = use_matriarchal_theme if use_matriarchal_theme is not None else self.config.enable_matriarchal_theme
        
        # Get the lore system
        lore_system = await self._get_lore_system()
        
        # Generate the world
        result = await lore_system.generate_complete_world(
            ctx,
            environment_desc,
            use_matriarchal_theme=use_matriarchal
        )
        
        # Update metrics
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
        """
        Evolve the world based on a narrative event.
        
        Args:
            ctx: Context object
            event_description: Description of the event
            affected_location_id: Optional specific location affected
            
        Returns:
            Evolution results and updates
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        # Get lore dynamics system
        dynamics = await self._get_lore_dynamics_system()
        
        # Evolve the world
        result = await dynamics.evolve_lore_with_event(event_description)
        
        # If specific location affected, update local lore
        if affected_location_id:
            local_mgr = await self._get_local_lore_manager()
            location_update = await local_mgr.evolve_location_lore(
                ctx, affected_location_id, event_description
            )
            result['location_update'] = location_update
            
            # Record location change
            if self._change_tracking_enabled and location_update:
                await self.record_lore_change(
                    'location', 
                    affected_location_id, 
                    'update',
                    new_data={'event': event_description, 'updates': location_update}
                )
        
        # Record world-level changes from the evolution
        if self._change_tracking_enabled and 'updates' in result:
            for update in result.get('updates', []):
                if 'element_type' in update and 'element_id' in update:
                    await self.record_lore_change(
                        update['element_type'],
                        update['element_id'],
                        update.get('operation', 'update'),
                        new_data=update.get('new_data'),
                        old_data=update.get('old_data')
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
        """Add a deity to the world."""
        manager = await self._get_religion_manager()
        return await manager.add_deity(ctx, params)
    
    async def add_pantheon(self, ctx, params: PantheonParams) -> int:
        """Add a pantheon to the world."""
        manager = await self._get_religion_manager()
        return await manager.add_pantheon(ctx, params)
    
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """Generate a complete faith system with all components."""
        manager = await self._get_religion_manager()
        return await manager.generate_complete_faith_system(ctx)
    
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        """Distribute religions across nations."""
        manager = await self._get_religion_manager()
        return await manager.distribute_religions(ctx)
    
    async def generate_ritual(
        self, ctx,
        pantheon_id: int,
        deity_id: Optional[int] = None,
        purpose: str = "blessing",
        formality_level: int = 5
    ) -> Dict[str, Any]:
        """Generate a detailed religious ritual."""
        manager = await self._get_religion_manager()
        return await manager.generate_ritual(ctx, pantheon_id, deity_id, purpose, formality_level)
    
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
        """Get all nations in the world."""
        manager = await self._get_politics_manager()
        return await manager.get_all_nations(ctx)
    
    async def generate_initial_conflicts(self, ctx, count: int = 3) -> List[Dict[str, Any]]:
        """Generate initial conflicts between nations."""
        manager = await self._get_politics_manager()
        return await manager.generate_initial_conflicts(ctx, count)
    
    async def simulate_diplomatic_negotiation(
        self, ctx, 
        nation1_id: int, 
        nation2_id: int, 
        issue: str
    ) -> Dict[str, Any]:
        """Simulate diplomatic negotiations between two nations."""
        manager = await self._get_politics_manager()
        return await manager.simulate_diplomatic_negotiation(ctx, nation1_id, nation2_id, issue)
    
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
