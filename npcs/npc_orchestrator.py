# npcs/npc_orchestrator.py

"""
NPC System Orchestrator with Scene Bundle Optimization
Complete implementation with calendar, time, and relationship system integrations
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import random

# Import core subsystems
from npcs.npc_agent_system import NPCAgentSystem
from npcs.npc_coordinator import NPCAgentCoordinator
from npcs.npc_memory import NPCMemoryManager
from npcs.npc_perception import EnvironmentPerception
from npcs.npc_relationship import NPCRelationshipManager

from nyx.scene_keys import generate_scene_cache_key

# Import preset NPC handler
from npcs.preset_npc_handler import PresetNPCHandler

# Import belief systems
from npcs.belief_system_integration import NPCBeliefSystemIntegration

# Import lore systems
from npcs.lore_context_manager import LoreContextManager

# Import behavior and decision systems
from npcs.npc_behavior import BehaviorEvolution
from npcs.npc_decisions import NPCDecisionEngine

# Import creation and template systems
from npcs.new_npc_creation import NPCCreationHandler

# Import database utilities
from db.connection import get_db_connection_context

# Import canon system for logging canonical events
from lore.core.canon import log_canonical_event

logger = logging.getLogger(__name__)
NPCS_SECTION_SUFFIX = "|npcs"


@dataclass(slots=True)
class NPCSnapshot:
    """Comprehensive snapshot of an NPC state (memory-optimized with slots)"""
    npc_id: int
    name: str
    role: str
    status: str
    location: str
    
    # Canonical information
    canonical_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Personality & traits
    personality_traits: List[str] = field(default_factory=list)
    dominance: int = 50
    cruelty: int = 50
    
    # Relationships
    trust: int = 0
    respect: int = 0
    closeness: int = 0
    intensity: int = 0
    
    # Dynamic state
    mask_integrity: int = 100
    emotional_state: Dict[str, Any] = field(default_factory=dict)
    recent_memories: List[Dict[str, Any]] = field(default_factory=list)
    
    # Special mechanics
    scheming_level: int = 0
    betrayal_planning: bool = False
    special_mechanics: Dict[str, Any] = field(default_factory=dict)


class NPCStatus(Enum):
    """NPC activity status"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    AWAY = "away"
    SCHEMING = "scheming"
    OBSERVING = "observing"


class NPCOrchestrator:
    """
    Main orchestrator for NPC system with scene bundle optimization.
    Manages all NPC subsystems and provides unified interface for context assembly.
    """
    
    def __init__(self, user_id: str, conversation_id: str, config: Optional[Dict[str, Any]] = None):
        # Convert to int for consistency with SDK
        self.user_id = int(user_id) if isinstance(user_id, str) else user_id
        self.conversation_id = int(conversation_id) if isinstance(conversation_id, str) else conversation_id
    
        # Configuration
        self.config = config or {}
        
        # Core subsystems - use normalized IDs
        self._memory_system = NPCMemoryManager(self.user_id, self.conversation_id)
        self._relationship_manager = NPCRelationshipManager(self.user_id, self.conversation_id)
        self._perception_system = EnvironmentPerception(self.user_id, self.conversation_id)
        self._belief_system = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)
        self._lore_manager = LoreContextManager(self.user_id, self.conversation_id)
        self._behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
        self._decision_engine = NPCDecisionEngine(self.user_id, self.conversation_id)
        self._creation_handler = NPCCreationHandler(self.user_id, self.conversation_id)
        self._preset_handler = PresetNPCHandler(self.user_id, self.conversation_id)
        
        # Agent systems - use normalized IDs
        self._agent_system = NPCAgentSystem(self.user_id, self.conversation_id)
        self._agent_coordinator = NPCAgentCoordinator(self.user_id, self.conversation_id)
        
        # Nyx integration
        self._nyx_bridge = None
        self._npc_bridge = None
        
        # Lazy-loaded systems
        self._calendar_system = None
        self._integrated_system = None
        self._dynamic_relationship_manager = None
        
        # Caching for performance (configurable TTLs)
        self._snapshot_cache: Dict[int, Tuple[NPCSnapshot, datetime]] = {}
        self._snapshot_ttl = timedelta(seconds=float(self.config.get('snapshot_ttl', 30)))
        self._bundle_cache: Dict[str, Tuple[Dict, float]] = {}
        self._bundle_ttl = float(self.config.get('bundle_ttl', 60.0))  # seconds
        
        # In-flight deduplication for snapshots
        self._inflight: Dict[int, asyncio.Task] = {}
        
        # DB connection semaphore for parallel requests
        self._db_semaphore = asyncio.Semaphore(self.config.get('max_db_connections', 6))
        
        # Scene state tracking
        self._scene_state_cache: Dict[int, Dict[str, Any]] = {}
        self._last_update_times: Dict[int, float] = {}
        
        # Active NPCs tracking
        self._active_npcs: Set[int] = set()
        self._location_index: Dict[str, Set[int]] = defaultdict(set)
        self._npc_status: Dict[int, NPCStatus] = {}
        
        # Canon configuration (now actually read from config)
        self.enable_canon = bool(self.config.get('enable_canon', True))
        self.auto_canonize = bool(self.config.get('auto_canonize', True))
        self.canon_cache: Dict[str, List[Dict]] = {}
        self._bundle_index: Dict[int, Set[str]] = defaultdict(set)
        
        # Performance metrics
        self.metrics = {
            'bundle_fetches': 0,
            'cache_hits': 0,
            'heavy_hits': 0,  # Heavy snapshots served from cache
            'light_hits': 0,  # Light snapshots served from cache
            'heavy_misses': 0,  # Heavy requests that missed cache
            'light_misses': 0,  # Light requests that missed cache
            'delta_updates': 0,
            'avg_bundle_time': 0.0
        }
        
        logger.info(f"NPCOrchestrator initialized for user {user_id}, conversation {conversation_id}")
    
    async def initialize(self):
        """Initialize all subsystems"""
        try:
            await self._memory_system.initialize()
            await self._relationship_manager.initialize()
            await self._perception_system.initialize()
            await self._belief_system.initialize()
            await self._lore_manager.initialize()
            
            # Ensure calendar tables exist
            if self._calendar_system:
                await self._calendar_system['ensure_tables'](self.user_id, self.conversation_id)
            
            # Load active NPCs
            await self._load_active_npcs()
            
            logger.info("NPC Orchestrator fully initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NPC Orchestrator: {e}")
            raise
    
    async def get_all_npcs(self, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all NPCs for this user/conversation. Required by ContextBroker.
        
        Args:
            location: Optional location to filter NPCs by
        """
        async with get_db_connection_context() as conn:
            if location:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND current_location = $3
                """, self.user_id, self.conversation_id, location)
            else:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
        return [{"id": r["npc_id"], "name": r["npc_name"]} for r in rows]
    
    # ==================== SCENE BUNDLE METHODS ====================
    from nyx.scene_keys import generate_scene_cache_key
    
    SECTION_SUFFIX = "|npcs"  # pipe avoids any collision with md5 hex

    async def get_scene_bundle(self, scope: 'SceneScope') -> Dict[str, Any]:
        start_time = time.time()
        scene_key = generate_scene_cache_key(scope)
        cache_key = f"{scene_key}{self.SECTION_SUFFIX}"
    
        # Fast path: serve from cache if fresh
        if cache_key in self._bundle_cache:
            cached_bundle, cached_time = self._bundle_cache[cache_key]
            if time.time() - cached_time < self._bundle_ttl:
                self.metrics['cache_hits'] += 1
                return cached_bundle
    
        bundle = {
            'section': 'npcs',
            'data': {'npcs': [], 'active_in_scene': []},
            'canonical': False,
            'last_changed_at': time.time(),
            'priority': 8,
            'version': f"npc_bundle_{time.time()}",
        }
    
        canonical_count = 0
    
        # Limit for perf
        npc_ids = list(getattr(scope, "npc_ids", []))[:10]
        bundle['data']['active_in_scene'] = npc_ids
    
        # Parallel light snapshots (DB concurrency guarded inside)
        snapshots = await asyncio.gather(
            *[self.get_npc_snapshot(n, force_refresh=False, light=True) for n in npc_ids],
            return_exceptions=True,
        )
    
        snapshot_map: Dict[int, NPCSnapshot] = {}
        link_hints = getattr(scope, "link_hints", {}) or {}
        topics = getattr(scope, "topics", set()) or set()
    
        for npc_id, res in zip(npc_ids, snapshots):
            if isinstance(res, Exception):
                logger.warning(f"Failed to get NPC {npc_id}: {res}")
                continue
    
            snapshot = res
            snapshot_map[npc_id] = snapshot
    
            npc_entry = {
                'id': npc_id,
                'name': snapshot.name,
                'role': snapshot.role,
                'canonical': bool(snapshot.canonical_events),
            }
    
            if snapshot.canonical_events:
                canonical_count += 1
                npc_entry['canonical_events'] = snapshot.canonical_events[:2]
                bundle['canonical'] = True
    
            npc_entry['core_traits'] = {
                'personality': snapshot.personality_traits[:3] if snapshot.personality_traits else [],
                'dominance': snapshot.dominance,
                'cruelty': snapshot.cruelty,
            }
    
            npc_entry['state'] = {
                'status': snapshot.status,
                'location': snapshot.location,
                'mask_integrity': snapshot.mask_integrity,
                'current_intent': snapshot.emotional_state.get('intent') if snapshot.emotional_state else None,
            }
    
            npc_entry['relationship'] = {
                'trust': snapshot.trust,
                'respect': snapshot.respect,
                'closeness': snapshot.closeness,
                'intensity': snapshot.intensity,
            }
    
            if topics:
                relevant_topics = await self._check_npc_topic_relevance(npc_id, topics)
                if relevant_topics:
                    npc_entry['relevant_topics'] = list(relevant_topics)
    
            related_npcs = link_hints.get("related_npcs")
            if related_npcs:
                relationships = await self._get_npc_relationships(npc_id, related_npcs)
                if relationships:
                    npc_entry['relationships'] = relationships
    
            bundle['data']['npcs'].append(npc_entry)
    
            # Update location index and scene-state cache
            old_location = self._scene_state_cache.get(npc_id, {}).get('location')
            if old_location and old_location != snapshot.location:
                loc_set = self._location_index.get(old_location)
                if loc_set:
                    loc_set.discard(npc_id)
                    if not loc_set:
                        self._location_index.pop(old_location, None)
            if snapshot.location:
                self._location_index[snapshot.location].add(npc_id)
    
            self._scene_state_cache[npc_id] = {
                'status': snapshot.status,
                'location': snapshot.location,
                'trust': snapshot.trust,
                'respect': snapshot.respect,
                'closeness': snapshot.closeness,
                'emotional_state': snapshot.emotional_state,
            }
    
        bundle['data']['canonical_count'] = canonical_count
        bundle['data']['scene_dynamics'] = await self._get_scene_dynamics(npc_ids, snapshot_map)
    
        if len(npc_ids) > 1:
            bundle['data']['group_dynamics'] = await self._get_group_dynamics(npc_ids, snapshot_map)
    
        # Cache bundle and wire reverse index so we can invalidate on NPC changes
        self._bundle_cache[cache_key] = (bundle, time.time())
        index = self._bundle_index  # local alias
        for nid in npc_ids:
            s = index.get(nid)
            if s is None:
                index[nid] = {cache_key}
            else:
                s.add(cache_key)
    
        # Prune expired entries (also cleans reverse index)
        self._prune_bundle_cache()
    
        # Update metrics
        elapsed = time.time() - start_time
        self.metrics['bundle_fetches'] += 1
        self.metrics['avg_bundle_time'] = (
            (self.metrics['avg_bundle_time'] * (self.metrics['bundle_fetches'] - 1) + elapsed)
            / self.metrics['bundle_fetches']
        )
    
        return bundle
    
    def _notify_npc_changed(self, npc_id: int):
        self._last_update_times[npc_id] = time.time()
    
        keys = self._bundle_index.pop(npc_id, set())
        if not keys:
            return
    
        for ck in keys:
            self._bundle_cache.pop(ck, None)
    
        # Optional: after mass invalidation, trim any empty sets left behind
        for nid, s in list(self._bundle_index.items()):
            if not s:
                self._bundle_index.pop(nid, None)
    
    # helper: invalidate NPC bundles for a *scene* key (called by ContextBroker)
    def invalidate_npc_bundles_for_scene_key(self, scene_key: str):
        key = f"{scene_key}{self.SECTION_SUFFIX}"
    
        # Drop the cached bundle
        self._bundle_cache.pop(key, None)
    
        # Remove this key from every NPC’s reverse index set
        for nid, keys in list(self._bundle_index.items()):
            if key in keys:
                keys.discard(key)
                if not keys:
                    self._bundle_index.pop(nid, None)
    
    def _prune_bundle_cache(self):
        """Remove expired entries from bundle cache and clean reverse index."""
        now = time.time()
        expired = [k for k, (_, ts) in self._bundle_cache.items() if now - ts > self._bundle_ttl]
        if not expired:
            return
    
        for k in expired:
            self._bundle_cache.pop(k, None)
    
        # Scrub expired keys from reverse index
        expired_set = set(expired)
        for nid, keys in list(self._bundle_index.items()):
            keys.difference_update(expired_set)
            if not keys:
                self._bundle_index.pop(nid, None)
    
    async def get_scene_delta(self, scope: 'SceneScope', since_ts: float) -> Dict[str, Any]:
        """
        Get only changes since timestamp for NPCs in scope.
        Efficient delta updates for cached scenes.
        """
        delta = {
            'data': {'changes': []},
            'canonical': False,
            'last_changed_at': time.time(),
            'priority': 8
        }
        
        changes_found = False
        
        # Helper to convert timestamps to epoch
        def _to_epoch(ts):
            if isinstance(ts, (int, float)):
                return float(ts)
            try:
                s = str(ts)
                if s.endswith('Z'):
                    s = s[:-1] + '+00:00'
                return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0
        
        for npc_id in list(scope.npc_ids)[:10]:
            # Check if NPC has changed since timestamp
            if not await self._has_npc_changed(npc_id, since_ts):
                continue
            
            changes_found = True
            
            try:
                # Get fresh snapshot (semaphore is inside _build_npc_snapshot)
                snapshot = await self.get_npc_snapshot(npc_id, force_refresh=True, light=True)
                
                # Build delta entry (only changed fields)
                change = {
                    'id': npc_id,
                    'name': snapshot.name,
                    'changed_at': time.time()
                }
                
                # Track what changed
                change_types = []
                
                # Check for new canonical events
                if snapshot.canonical_events:
                    recent_events = [
                        e for e in snapshot.canonical_events 
                        if _to_epoch(e.get('timestamp', 0)) > since_ts
                    ]
                    if recent_events:
                        change['new_canonical_events'] = recent_events[:2]
                        delta['canonical'] = True
                        change_types.append('canonical')
                
                # Check for state changes
                old_state = self._scene_state_cache.get(npc_id, {})
                
                if snapshot.status != old_state.get('status'):
                    change['status'] = snapshot.status
                    change_types.append('status')
                
                if snapshot.location != old_state.get('location'):
                    change['location'] = snapshot.location
                    change_types.append('location')
                    
                    # Update location index with pruning
                    old_location = old_state.get('location')
                    if old_location:
                        loc_set = self._location_index.get(old_location)
                        if loc_set:
                            loc_set.discard(npc_id)
                            if not loc_set:  # Prune empty location buckets
                                self._location_index.pop(old_location, None)
                    if snapshot.location:
                        self._location_index[snapshot.location].add(npc_id)
                
                # Relationship changes (threshold for significance)
                rel_changes = {}
                if abs(snapshot.trust - old_state.get('trust', 0)) > 5:
                    rel_changes['trust'] = snapshot.trust
                if abs(snapshot.respect - old_state.get('respect', 0)) > 5:
                    rel_changes['respect'] = snapshot.respect
                if abs(snapshot.closeness - old_state.get('closeness', 0)) > 5:
                    rel_changes['closeness'] = snapshot.closeness
                
                if rel_changes:
                    change['relationship_changes'] = rel_changes
                    change_types.append('relationship')
                
                # Emotional state changes
                old_intent = old_state.get('emotional_state', {}).get('intent')
                new_intent = snapshot.emotional_state.get('intent')
                if new_intent and new_intent != old_intent:
                    change['new_intent'] = new_intent
                    change_types.append('intent')
                
                change['change_types'] = change_types
                delta['data']['changes'].append(change)
                
                # Update state cache
                self._scene_state_cache[npc_id] = {
                    'status': snapshot.status,
                    'location': snapshot.location,
                    'trust': snapshot.trust,
                    'respect': snapshot.respect,
                    'closeness': snapshot.closeness,
                    'emotional_state': snapshot.emotional_state
                }
                
            except Exception as e:
                logger.warning(f"Failed to get delta for NPC {npc_id}: {e}")
                continue
        
        if not changes_found:
            delta['data']['no_changes'] = True
        
        self.metrics['delta_updates'] += 1
        return delta
    
    # ==================== SNAPSHOT METHODS ====================
    
    def _is_heavy_snapshot(self, snapshot: NPCSnapshot) -> bool:
        """Check if a snapshot has heavy fields populated"""
        return bool(snapshot.recent_memories) or bool(snapshot.emotional_state)
    
    async def get_npc_snapshot(self, npc_id: int, force_refresh: bool = False, light: bool = True) -> NPCSnapshot:
        """
        Get comprehensive NPC snapshot with caching and in-flight deduplication.
        
        Args:
            npc_id: ID of the NPC
            force_refresh: Skip cache and force fresh fetch
            light: Skip heavy operations like memory recall for performance
        """
        # Check cache first
        if not force_refresh and npc_id in self._snapshot_cache:
            snapshot, timestamp = self._snapshot_cache[npc_id]
            if datetime.now() - timestamp < self._snapshot_ttl:
                # If caller wants light, cached heavy or light is fine
                # If caller wants heavy, ensure cached is heavy
                if light or self._is_heavy_snapshot(snapshot):
                    if light:
                        self.metrics['light_hits'] += 1
                    else:
                        self.metrics['heavy_hits'] += 1
                    return snapshot
                else:
                    self.metrics['heavy_misses'] += 1
        
        # Check if already fetching this NPC
        if npc_id in self._inflight:
            snapshot = await self._inflight[npc_id]
            # Same heavy/light check after awaiting in-flight
            if light or self._is_heavy_snapshot(snapshot):
                return snapshot
            # Need heavy but got light, must refetch
            self.metrics['heavy_misses'] += 1
        
        # Count a miss for cold/expired cache
        if light:
            self.metrics['light_misses'] += 1
        elif npc_id not in self._inflight:
            # Only count heavy miss if not already counted from in-flight mismatch
            self.metrics['heavy_misses'] += 1
        
        # Build fresh snapshot with in-flight tracking
        async def _build():
            try:
                snapshot = await self._build_npc_snapshot(npc_id, light=light)
                
                # Always prefer caching heavy over light
                existing = self._snapshot_cache.get(npc_id)
                if not existing or self._is_heavy_snapshot(snapshot) or not self._is_heavy_snapshot(existing[0]):
                    self._snapshot_cache[npc_id] = (snapshot, datetime.now())
                
                self._last_update_times[npc_id] = time.time()
                return snapshot
            finally:
                self._inflight.pop(npc_id, None)
        
        task = asyncio.create_task(_build())
        self._inflight[npc_id] = task
        return await task
    
    async def _build_npc_snapshot(self, npc_id: int, light: bool = True) -> NPCSnapshot:
        """Build a comprehensive snapshot of an NPC including canonical events."""
        # Centralize DB concurrency limits
        async with self._db_semaphore:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT npc_name, current_location, dominance, cruelty, 
                           closeness, trust, respect, intensity, mask_integrity,
                           personality_traits, schedule, scheming_level, betrayal_planning,
                           special_mechanics, status, role
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if not row:
                    raise ValueError(f"NPC {npc_id} not found")
                
                # Convert Record to dict for safe access
                row_map = dict(row)
                
                # Get canonical events if canon enabled
                canonical_events = []
                if self.enable_canon:
                    # Check cache first
                    cache_key = f"canon_{npc_id}"
                    if cache_key in self.canon_cache:
                        canonical_events = self.canon_cache[cache_key]
                    else:
                        events = await conn.fetch("""
                            SELECT event_text, tags, significance, timestamp
                            FROM CanonicalEvents
                            WHERE user_id = $1 AND conversation_id = $2
                            AND (event_text ILIKE '%' || $3 || '%' OR tags @> ARRAY[$3]::text[])
                            ORDER BY significance DESC, timestamp DESC
                            LIMIT 5
                        """, self.user_id, self.conversation_id, row_map["npc_name"])
                        
                        canonical_events = [dict(e) for e in events]
                        self.canon_cache[cache_key] = canonical_events
        
        # Get emotional state (skip if light mode)
        emotional_state = {} if light else await self._memory_system.get_npc_emotion(npc_id)
        
        # Get recent memories (skip if light mode)
        recent_memories = []
        if not light:
            memory_result = await self._memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="",
                limit=5
            )
            recent_memories = memory_result.get("memories", [])
        
        # Normalize personality traits
        traits = row_map.get("personality_traits", [])
        if isinstance(traits, str):
            try:
                traits = json.loads(traits)
            except Exception:
                traits = [traits] if traits else []
        
        # Normalize special_mechanics if stored as JSON text
        special = row_map.get("special_mechanics", {})
        if isinstance(special, str):
            try:
                special = json.loads(special)
            except Exception:
                special = {}
        
        # Build snapshot
        snapshot = NPCSnapshot(
            npc_id=npc_id,
            name=row_map["npc_name"],
            role=row_map.get("role", "NPC"),
            status=row_map.get("status", "active"),
            location=row_map["current_location"],
            canonical_events=canonical_events,
            personality_traits=traits,
            dominance=row_map.get("dominance") or 50,
            cruelty=row_map.get("cruelty") or 50,
            trust=row_map.get("trust") or 0,
            respect=row_map.get("respect") or 0,
            closeness=row_map.get("closeness") or 0,
            intensity=row_map.get("intensity") or 0,
            mask_integrity=row_map.get("mask_integrity") or 100,
            emotional_state=emotional_state,
            recent_memories=recent_memories,
            scheming_level=row_map.get("scheming_level", 0),
            betrayal_planning=row_map.get("betrayal_planning", False),
            special_mechanics=special
        )
        
        return snapshot
    
    # ==================== CALENDAR INTEGRATION ====================
    
    async def _get_calendar_system(self):
        """Lazy load calendar system."""
        if self._calendar_system is None:
            from logic.calendar import (
                ensure_calendar_tables,
                add_calendar_event,
                get_calendar_events,
                get_events_for_display,
                check_current_events,
                mark_event_completed,
                mark_event_missed,
                auto_process_missed_events
            )
            self._calendar_system = {
                'ensure_tables': ensure_calendar_tables,
                'add_event': add_calendar_event,
                'get_events': get_calendar_events,
                'get_display': get_events_for_display,
                'check_current': check_current_events,
                'mark_completed': mark_event_completed,
                'mark_missed': mark_event_missed,
                'auto_process': auto_process_missed_events
            }
            await self._calendar_system['ensure_tables'](self.user_id, self.conversation_id)
        return self._calendar_system
    
    async def schedule_npc_event(
        self,
        npc_id: int,
        event_name: str,
        event_type: str,
        year: int,
        month: int,
        day: int,
        time_of_day: str,
        location: Optional[str] = None,
        duration: int = 1,
        priority: int = 2,
        requirements: Optional[Dict[str, Any]] = None,
        rewards: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Schedule a calendar event for an NPC with canon integration."""
        calendar = await self._get_calendar_system()
        
        # Get NPC data for context
        snapshot = await self.get_npc_snapshot(npc_id)
        
        # Use NPC's current location if not specified
        if not location:
            location = snapshot.location
        
        # Add NPCs as involved parties
        involved_npcs = [{"npc_id": npc_id, "role": "primary"}]
        
        # Create the event
        result = await calendar['add_event'](
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            event_name=f"{snapshot.name}: {event_name}",
            event_type=event_type,
            year=year,
            month=month,
            day=day,
            time_of_day=time_of_day,
            description=f"Event involving {snapshot.name}",
            location=location,
            duration=duration,
            priority=priority,
            involved_npcs=involved_npcs,
            requirements=requirements,
            rewards=rewards,
            is_recurring=False
        )
        
        # Canonize if significant
        if self.enable_canon and self.auto_canonize and priority >= 3:
            async with get_db_connection_context() as conn:
                await log_canonical_event(
                    conn,
                    self.user_id,
                    self.conversation_id,
                    f"Scheduled event for {snapshot.name}: {event_name} on {year}-{month:02d}-{day:02d}",
                    tags=["npc", "calendar", "event", event_type],
                    significance=priority + 2
                )
        
        # Add memory for NPC about the scheduled event
        await self._memory_system.store(
            entity_type="npc",
            entity_id=npc_id,
            content=f"I have {event_name} scheduled for {time_of_day} on day {day}",
            memory_type="scheduling",
            significance=priority
        )
        
        self._notify_npc_changed(npc_id)
        return result
    
    async def check_npc_current_events(self, npc_id: int) -> List[Dict[str, Any]]:
        """Check what events an NPC should be participating in right now."""
        calendar = await self._get_calendar_system()
        
        # Get current game time
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT key, value FROM CurrentRoleplay
                WHERE key IN ('Year','Month','Day','TimeOfDay')
                  AND user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        
        time_vals = {r['key']: r['value'] for r in rows} if rows else {}
        if not time_vals:
            return []
        
        current_events = await calendar['check_current'](
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            year=int(time_vals.get('Year', 1)),
            month=int(time_vals.get('Month', 1)),
            day=int(time_vals.get('Day', 1)),
            time_of_day=time_vals.get('TimeOfDay', 'Morning')
        )
        
        # Filter for this NPC
        npc_events = []
        for event in current_events:
            if any(n.get('npc_id') == npc_id for n in event.get('involved_npcs', [])):
                npc_events.append(event)
                
                # Update NPC status if they have an event
                if event.get('event_priority', 0) >= 4:
                    self._npc_status[npc_id] = NPCStatus.BUSY
        
        return npc_events
    
    async def process_calendar_events_for_all_npcs(self) -> Dict[str, Any]:
        """Process current calendar events for all NPCs."""
        results = {
            "processed_events": [],
            "missed_events": [],
            "npc_statuses": {}
        }
        
        calendar = await self._get_calendar_system()
        
        # Fetch real in-game time once
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT key, value FROM CurrentRoleplay
                    WHERE key IN ('Year','Month','Day','TimeOfDay')
                      AND user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
            time_map = {r["key"]: r["value"] for r in rows} if rows else {}
        except Exception as e:
            logger.exception(f"Failed to read CurrentRoleplay time: {e}")
            time_map = {}
        
        year = int(time_map.get("Year", 1))
        month = int(time_map.get("Month", 1))
        day = int(time_map.get("Day", 1))
        time_of_day = time_map.get("TimeOfDay", "Morning")
        
        # Auto-process missed events using real game time
        try:
            await calendar["auto_process"](
                self.user_id,
                self.conversation_id,
                year, month, day, time_of_day
            )
        except Exception as e:
            logger.exception(f"Auto-process missed events failed: {e}")
        
        # Check current events for each NPC
        for npc_id in list(self._active_npcs):
            try:
                events = await self.check_npc_current_events(npc_id)
            except Exception as e:
                logger.exception(f"check_npc_current_events failed for NPC {npc_id}: {e}")
                continue
            
            if events:
                results["npc_statuses"][npc_id] = "has_events"
                # after mark_completed / mark_missed
                self._notify_npc_changed(npc_id)
                for event in events:
                    try:
                        if event.get("can_participate"):
                            await calendar["mark_completed"](
                                self.user_id,
                                self.conversation_id,
                                event["event_id"],
                                {"npc_id": npc_id}
                            )
                            results["processed_events"].append({
                                "npc_id": npc_id,
                                "event_id": event.get("event_id"),
                                "event_name": event.get("event_name"),
                                "time_of_day": event.get("time_of_day"),
                            })
                        else:
                            await calendar["mark_missed"](
                                self.user_id,
                                self.conversation_id,
                                event["event_id"]
                            )
                            results["missed_events"].append({
                                "npc_id": npc_id,
                                "event_id": event.get("event_id"),
                                "event_name": event.get("event_name"),
                                "time_of_day": event.get("time_of_day"),
                                "reason": event.get("missing_requirements", []),
                            })
                    except Exception as e:
                        logger.exception(
                            f"Failed to process calendar event {event.get('event_id')} for NPC {npc_id}: {e}"
                        )
        
        return results
    
    # ==================== RELATIONSHIP INTEGRATION ====================
    
    async def _get_integrated_npc_system(self):
        """Lazy load the fully integrated NPC system."""
        if self._integrated_system is None:
            from logic.fully_integrated_npc_system import IntegratedNPCSystem
            self._integrated_system = IntegratedNPCSystem(
                self.user_id,
                self.conversation_id
            )
            await self._integrated_system.initialize()
        return self._integrated_system
    
    async def _get_dynamic_relationship_manager(self):
        """Lazy load the dynamic relationship manager."""
        if self._dynamic_relationship_manager is None:
            from logic.dynamic_relationships import OptimizedRelationshipManager
            self._dynamic_relationship_manager = OptimizedRelationshipManager(
                self.user_id,
                self.conversation_id
            )
        return self._dynamic_relationship_manager
    
    async def get_relationship_dynamics(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int
    ) -> Dict[str, Any]:
        """Get dynamic relationship state between entities."""
        manager = await self._get_dynamic_relationship_manager()
        
        state = await manager.get_relationship_state(
            entity1_type, entity1_id,
            entity2_type, entity2_id
        )
        
        return {
            'dimensions': {
                'trust': float(state.dimensions.trust),
                'affection': float(state.dimensions.affection),
                'respect': float(state.dimensions.respect),
                'familiarity': float(state.dimensions.familiarity),
                'tension': float(state.dimensions.tension)
            },
            'patterns': list(state.history.active_patterns),
            'archetypes': list(state.active_archetypes),
            'momentum': {
                'magnitude': float(state.momentum.get_magnitude()),
                'direction': state.momentum.get_direction()
            },
            'contexts': asdict(state.contexts)
        }
    
    async def process_relationship_interaction(
        self,
        npc_id: int,
        interaction_type: str,
        context: str = "casual",
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """Process an interaction between player and NPC."""
        manager = await self._get_dynamic_relationship_manager()
        
        # Build interaction data
        interaction = {
            'type': interaction_type,
            'context': context,
            'intensity': intensity,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process with player as entity1, NPC as entity2
        result = await manager.process_interaction(
            entity1_type="player",
            entity1_id=1,  # Assuming player_id is 1
            entity2_type="npc",
            entity2_id=npc_id,
            interaction=interaction
        )
        
        # Invalidate NPC snapshot cache since relationship changed
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        self._notify_npc_changed(npc_id)
        
        return result
    
    # ==================== COMPREHENSIVE SCENE GENERATION ====================
    
    async def generate_comprehensive_scene(
        self,
        location: Optional[str] = None,
        include_player: bool = True,
        time_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive scene with all systems integrated."""
        
        # Get NPCs at location
        if location:
            npcs_at_location = await self.get_npcs_at_location(location)
            npc_ids = [npc.npc_id for npc in npcs_at_location]
        else:
            npc_ids = list(self._active_npcs)[:5]  # Limit to 5 for performance
        
        scene = {
            "location": location,
            "npcs": {},
            "calendar_events": [],
            "relationships": [],
            "group_dynamics": {},
            "ongoing_narratives": [],
            "environmental_context": {},
            "canonical_constraints": []
        }
        
        # Build snapshot map to avoid duplicate fetches (use light mode)
        snapshot_map = {}
        async def fetch_snapshot_safe(npc_id):
            # _build_npc_snapshot already enforces DB concurrency
            return await self.get_npc_snapshot(npc_id, light=True)
        
        snapshot_tasks = [fetch_snapshot_safe(npc_id) for npc_id in npc_ids]
        snapshots = await asyncio.gather(*snapshot_tasks, return_exceptions=True)
        
        for npc_id, snapshot_result in zip(npc_ids, snapshots):
            if not isinstance(snapshot_result, Exception):
                snapshot_map[npc_id] = snapshot_result
                scene["npcs"][npc_id] = asdict(snapshot_result)
        
        # Check calendar events
        calendar_results = await self.process_calendar_events_for_all_npcs()
        scene["calendar_events"] = calendar_results.get("processed_events", [])
        
        # Get relationships if player is included
        if include_player:
            for npc_id in npc_ids:
                rel_dynamics = await self.get_relationship_dynamics(
                    "player", 1, "npc", npc_id
                )
                scene["relationships"].append({
                    "npc_id": npc_id,
                    "dynamics": rel_dynamics
                })
        
        # Get group dynamics with snapshot map
        scene["group_dynamics"] = await self._get_group_dynamics(npc_ids, snapshot_map)
        
        # Get time context
        if not time_context:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT key, value FROM CurrentRoleplay
                    WHERE key IN ('Year','Month','Day','TimeOfDay','Season')
                    AND user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
            time_context = {r['key']: r['value'] for r in rows} if rows else {}
        
        scene["environmental_context"]["time"] = time_context
        
        # Get canonical constraints for the scene
        if self.enable_canon:
            scene["canonical_constraints"] = await self._get_canonical_constraints(location, npc_ids)
        
        return scene
    
    # ==================== HELPER METHODS ====================
    
    async def _has_npc_changed(self, npc_id: int, since_ts: float) -> bool:
        """Check if NPC has changed since timestamp"""
        last_update = self._last_update_times.get(npc_id, 0)
        return last_update > since_ts
    
    async def _check_npc_topic_relevance(self, npc_id: int, topics: Set[str]) -> Set[str]:
        """Check which topics are relevant to an NPC"""
        relevant = set()
        
        # Get NPC's interests and specialties from DB or cache
        snapshot = await self.get_npc_snapshot(npc_id)
        
        # Simple keyword matching for now
        npc_text = f"{snapshot.name} {snapshot.role} {' '.join(snapshot.personality_traits)}"
        npc_text_lower = npc_text.lower()
        
        for topic in topics:
            if topic.lower() in npc_text_lower:
                relevant.add(topic)
        
        # Check memories for topic relevance
        if snapshot.recent_memories:
            memory_text = ' '.join([m.get('content', '') for m in snapshot.recent_memories])
            memory_text_lower = memory_text.lower()
            for topic in topics:
                if topic.lower() in memory_text_lower:
                    relevant.add(topic)
        
        return relevant
    
    async def _get_npc_relationships(self, npc_id: int, related_npc_ids: List[int]) -> List[Dict]:
        """Get relationships between NPCs"""
        relationships = []
        manager = await self._get_dynamic_relationship_manager()
        
        for other_id in related_npc_ids:
            if other_id == npc_id:
                continue
            
            try:
                state = await manager.get_relationship_state(
                    "npc", npc_id, "npc", other_id
                )
                relationships.append({
                    'with_npc': other_id,
                    'trust': float(state.dimensions.trust),
                    'affection': float(state.dimensions.affection),
                    'tension': float(state.dimensions.tension),
                    'patterns': list(state.history.active_patterns)[:2]
                })
            except Exception:
                # Relationship might not exist
                pass
        
        return relationships
    
    async def _get_scene_dynamics(self, npc_ids: List[int], snapshot_map: Optional[Dict[int, NPCSnapshot]] = None) -> Dict[str, Any]:
        """Get dynamics for NPCs in a scene"""
        dynamics = {
            'tension_level': 0,
            'dominant_mood': 'neutral',
            'active_conflicts': [],
            'alliances': []
        }
        
        if not npc_ids:
            return dynamics
        
        # Calculate aggregate tension
        tensions = []
        for npc_id in npc_ids:
            # Use provided snapshot or fetch if needed (light mode)
            if snapshot_map and npc_id in snapshot_map:
                snapshot = snapshot_map[npc_id]
            else:
                snapshot = await self.get_npc_snapshot(npc_id, light=True)
            
            if snapshot.scheming_level > 0:
                tensions.append(snapshot.scheming_level)
            if snapshot.betrayal_planning:
                tensions.append(50)  # Betrayal adds tension
        
        if tensions:
            dynamics['tension_level'] = sum(tensions) // len(tensions)
        
        # Determine mood based on NPCs present
        moods = []
        for npc_id in npc_ids:
            if snapshot_map and npc_id in snapshot_map:
                snapshot = snapshot_map[npc_id]
            else:
                snapshot = await self.get_npc_snapshot(npc_id, light=True)
            
            if snapshot.emotional_state.get('mood'):
                moods.append(snapshot.emotional_state['mood'])
        
        if moods:
            # Simple majority mood
            from collections import Counter
            mood_counts = Counter(moods)
            dynamics['dominant_mood'] = mood_counts.most_common(1)[0][0]
        
        return dynamics
    
    async def _get_group_dynamics(self, npc_ids: List[int], snapshot_map: Optional[Dict[int, NPCSnapshot]] = None) -> Dict[str, Any]:
        """Get group dynamics for multiple NPCs"""
        dynamics = {
            'power_structure': [],
            'group_cohesion': 0,
            'potential_conflicts': []
        }
        
        if len(npc_ids) < 2:
            return dynamics
        
        # Determine power hierarchy
        power_scores = []
        for npc_id in npc_ids:
            # Use provided snapshot or fetch if needed (light mode)
            if snapshot_map and npc_id in snapshot_map:
                snapshot = snapshot_map[npc_id]
            else:
                snapshot = await self.get_npc_snapshot(npc_id, light=True)
            
            power = snapshot.dominance + snapshot.respect
            power_scores.append((npc_id, snapshot.name, power))
        
        power_scores.sort(key=lambda x: x[2], reverse=True)
        dynamics['power_structure'] = [
            {'id': p[0], 'name': p[1], 'power': p[2]} 
            for p in power_scores[:3]
        ]
        
        # Calculate cohesion using dynamic relationships
        manager = await self._get_dynamic_relationship_manager()
        total_affection = 0
        total_tension = 0
        pair_count = 0
        
        for i, npc_id in enumerate(npc_ids):
            for other_id in npc_ids[i+1:]:
                try:
                    state = await manager.get_relationship_state(
                        "npc", npc_id, "npc", other_id
                    )
                    total_affection += state.dimensions.affection
                    total_tension += state.dimensions.tension
                    pair_count += 1
                except Exception:
                    pass
        
        if pair_count > 0:
            cohesion = (total_affection - total_tension) / pair_count
            dynamics['group_cohesion'] = int(round(cohesion))
        
        return dynamics
    
    async def _get_canonical_constraints(
        self,
        location: Optional[str],
        npc_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Get canonical constraints for a scene."""
        constraints = []
        
        async with get_db_connection_context() as conn:
            # Get location-based canonical events
            if location:
                location_events = await conn.fetch("""
                    SELECT event_text, tags, significance
                    FROM CanonicalEvents
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (event_text ILIKE '%' || $3 || '%' OR tags @> ARRAY[$3]::text[])
                    ORDER BY significance DESC
                    LIMIT 3
                """, self.user_id, self.conversation_id, location)
                
                for event in location_events:
                    constraints.append({
                        'type': 'location',
                        'text': event['event_text'],
                        'tags': event['tags'],
                        'significance': event['significance']
                    })
            
            # Get NPC-based canonical constraints
            for npc_id in npc_ids[:3]:  # Limit to top 3 NPCs
                snapshot = await self.get_npc_snapshot(npc_id, light=True)
                if snapshot.canonical_events:
                    for event in snapshot.canonical_events[:1]:  # Top event per NPC
                        constraints.append({
                            'type': 'npc',
                            'npc_id': npc_id,
                            'npc_name': snapshot.name,
                            'text': event.get('event_text', ''),
                            'significance': event.get('significance', 5)
                        })
        
        return constraints
    
    # ==================== EXISTING METHODS (PRESERVED) ====================
    
    async def _load_active_npcs(self):
        """Load active NPCs from database"""
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND status IN ('active', 'idle', 'observing')
            """, self.user_id, self.conversation_id)
            
            for row in rows:
                self._active_npcs.add(row['npc_id'])
                if row['current_location']:
                    self._location_index[row['current_location']].add(row['npc_id'])
    
    async def get_npcs_at_location(self, location: str) -> List[NPCSnapshot]:
        """Get all NPCs at a specific location"""
        npc_ids = set(self._location_index.get(location, set()))
        
        # Fallback to DB if index not yet warmed for this location
        if not npc_ids:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                      AND current_location = $3
                      AND status IN ('active','idle','observing')
                """, self.user_id, self.conversation_id, location)
            npc_ids = {r["npc_id"] for r in rows}
            if npc_ids:
                self._location_index[location] = set(npc_ids)
        
        snapshots = []
        for npc_id in npc_ids:
            try:
                # Use light mode for performance
                snapshot = await self.get_npc_snapshot(npc_id, light=True)
                snapshots.append(snapshot)
            except Exception as e:
                logger.warning(f"Failed to get snapshot for NPC {npc_id}: {e}")
        
        return snapshots
    
    async def handle_player_action(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle player action and generate NPC responses"""
        # Use bridge system if available
        if self._npc_bridge:
            return await self._npc_bridge.handle_player_action(action, context)
        
        # Otherwise use agent system
        return await self._agent_system.process_player_action(action, context)
    
    async def progress_npc_narrative(
        self,
        npc_id: int,
        corruption_change: int = 0,
        dependency_change: int = 0,
        realization_change: int = 0
    ) -> Dict[str, Any]:
        """Progress an NPC's narrative arc"""
        async with get_db_connection_context() as conn:
            # Update narrative progress
            await conn.execute("""
                UPDATE NPCStats
                SET corruption = LEAST(100, GREATEST(0, corruption + $1)),
                    dependency = LEAST(100, GREATEST(0, dependency + $2)),
                    realization = LEAST(100, GREATEST(0, realization + $3))
                WHERE npc_id = $4 AND user_id = $5 AND conversation_id = $6
            """, corruption_change, dependency_change, realization_change,
                npc_id, self.user_id, self.conversation_id)
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        self._notify_npc_changed(npc_id)
        
        return {
            'npc_id': npc_id,
            'corruption_change': corruption_change,
            'dependency_change': dependency_change,
            'realization_change': realization_change
        }
    
    async def check_for_npc_revelation(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """Check if NPC has a revelation about Nyx's nature"""
        snapshot = await self.get_npc_snapshot(npc_id)
        
        # Revelation logic based on mask integrity and other factors
        if snapshot.mask_integrity < 30:
            revelation_chance = (100 - snapshot.mask_integrity) / 100
            if random.random() < revelation_chance:
                return {
                    'npc_id': npc_id,
                    'revelation_type': 'nyx_nature',
                    'intensity': 100 - snapshot.mask_integrity
                }
        
        return None
    
    async def sync(self, force: bool = False):
        """Sync orchestrator state"""
        # Clear caches if forced
        if force:
            self._snapshot_cache.clear()
            self._bundle_cache.clear()
            self.canon_cache.clear()
            
        # Reload active NPCs
        await self._load_active_npcs()
    
    async def cleanup(self):
        """Cleanup orchestrator resources"""
        self._snapshot_cache.clear()
        self._bundle_cache.clear()
        self.canon_cache.clear()
        self._scene_state_cache.clear()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'cache_size': len(self._snapshot_cache),
            'bundle_cache_size': len(self._bundle_cache),
            'active_npcs': len(self._active_npcs),
            'cache_effectiveness': {
                'heavy_hit_rate': self.metrics['heavy_hits'] / max(1, self.metrics['heavy_hits'] + self.metrics['heavy_misses']),
                'light_hit_rate': self.metrics['light_hits'] / max(1, self.metrics['light_hits'] + self.metrics['light_misses'])
            }
        }
    
    # ==================== API EXPANSION METHODS ====================
    
    async def expand_npc(self, npc_id: int, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Expand NPC details for on-demand retrieval.
        
        Args:
            npc_id: ID of the NPC to expand
            fields: Optional list of specific fields to include
        
        Returns:
            Dictionary with expanded NPC data
        """
        snapshot = await self.get_npc_snapshot(npc_id, force_refresh=False, light=False)
        doc = asdict(snapshot)
        
        if fields:
            doc = {k: doc[k] for k in fields if k in doc}
        
        return doc
    
    async def expand_npc_canon(self, npc_id: int, limit: int = 3) -> Dict[str, Any]:
        """
        Lightweight expansion for just canonical events.
        
        Args:
            npc_id: ID of the NPC
            limit: Maximum number of canonical events to return
        
        Returns:
            Dictionary with NPC ID, name, and canonical events
        """
        snapshot = await self.get_npc_snapshot(npc_id, light=True)
        return {
            "id": npc_id,
            "name": snapshot.name,
            "canonical_events": snapshot.canonical_events[:limit]
        }
    
    async def expand_relationships(self, npc_ids: List[int]) -> Dict[int, Any]:
        """
        Expand relationship details between NPCs.
        
        Args:
            npc_ids: List of NPC IDs to get relationships for
        
        Returns:
            Dictionary mapping NPC IDs to their relationships
        """
        manager = await self._get_dynamic_relationship_manager()
        relationships = {}
        
        for i, npc_a in enumerate(npc_ids):
            for npc_b in npc_ids[i+1:]:
                try:
                    state = await manager.get_relationship_state(
                        "npc", npc_a, "npc", npc_b
                    )
                    relationships.setdefault(npc_a, {})[npc_b] = {
                        "trust": float(state.dimensions.trust),
                        "affection": float(state.dimensions.affection),
                        "tension": float(state.dimensions.tension),
                        "patterns": list(state.history.active_patterns)[:2]
                    }
                    # Add reverse relationship
                    relationships.setdefault(npc_b, {})[npc_a] = {
                        "trust": float(state.dimensions.trust),
                        "affection": float(state.dimensions.affection),
                        "tension": float(state.dimensions.tension),
                        "patterns": list(state.history.active_patterns)[:2]
                    }
                except Exception:
                    pass
        
        return relationships
