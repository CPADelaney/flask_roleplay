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
from types import SimpleNamespace

# Import core subsystems
from npcs.npc_agent_system import NPCAgentSystem
from npcs.npc_coordinator import NPCAgentCoordinator
from npcs.npc_memory import NPCMemoryManager
from npcs.npc_perception import EnvironmentPerception, PerceptionContext
from npcs.npc_learning_adaptation import NPCLearningManager

from nyx.scene_keys import generate_scene_cache_key

from npcs.npc_handler import NPCHandler, NPCInteractionProposal

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

    # ---- Lore (small, optional) ----
    lore_summary: List[Dict[str, Any]] = field(default_factory=list)  # [{"type","id","name","knowledge_level"}]
    
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

        # Conflict synthesizer integration (production)
        self.enable_conflicts = bool(self.config.get("enable_conflict_system", True))
        self._conflict_synth = None  # ConflictSynthesizer
        self._conflict_cache: Dict[int, Tuple[Dict[str, Any], float]] = {}
        self._conflict_cache_ttl = float(self.config.get("conflict_cache_ttl_s", 30.0))
        
        # Core subsystems - use normalized IDs
        self._perception: Dict[int, EnvironmentPerception] = {}
        self._belief_system = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)
        self._lore_manager = LoreContextManager(self.user_id, self.conversation_id)
        self._behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
        self._creation_handler = NPCCreationHandler(self.user_id, self.conversation_id)
        self._preset_handler = PresetNPCHandler

        self.enable_learning_adaptation = bool(self.config.get("enable_learning_adaptation", True))
        self._learning_manager = None

        # IntegratedNPCSystem routing toggles
        self._integrated_routing = {
            "use_for_conversation": bool(self.config.get("use_integrated_for_conversation", False)),
            "use_for_group": bool(self.config.get("use_integrated_for_group", False)),
            "use_for_time_advance": bool(self.config.get("use_integrated_for_time", True)),
            "use_for_scheduled_activities": bool(self.config.get("use_integrated_for_scheduled", True)),
            "use_for_relationship_ops": bool(self.config.get("use_integrated_for_relationships", True)),
        }

        self._decision_engines: Dict[int, NPCDecisionEngine] = {}
        self._decision_engine_timestamps: Dict[int, float] = {}
        self._decision_engine_ttl = float(self.config.get("decision_engine_ttl_s", 600.0))  # 10m default

        # Belief consolidation throttling
        self._last_belief_consolidation: Dict[int, float] = {}
        self._consolidation_cooldown = float(self.config.get('belief_consolidation_cooldown_s', 180.0))  # 3 minutes default

        # Agent systems - use normalized IDs
        self._agent_system = NPCAgentSystem(self.user_id, self.conversation_id)
        self._agent_coordinator = NPCAgentCoordinator(self.user_id, self.conversation_id)
        
        # Nyx integration
        self._nyx_bridge = None
        self._npc_bridge = None

        self._include_lore_in_scene_bundle = bool(self.config.get('include_lore_in_scene_bundle', False))
        
        # Lazy-loaded systems
        self._calendar_system = None
        self._integrated_system = None
        self._dynamic_relationship_manager = None

        self._responder = NPCHandler(self.user_id, self.conversation_id)
        
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

        # per-NPC memory managers
        self._mem_mgrs: Dict[int, NPCMemoryManager] = {}
        self._active_status_values = {"active", "idle", "observing"}

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

        # Behavior eval throttling
        self._last_behavior_eval: Dict[int, float] = {}
        self._behavior_eval_cooldown = float(self.config.get('behavior_eval_cooldown_s', 180.0))  # 3m default

        # Behavior metrics
        self.metrics.update({
            'behavior_evals': 0,
            'behavior_applies': 0,
            'behavior_skipped_cooldown': 0,
        })

        self.metrics.update({
            'group_decisions': 0,
            'group_player_actions': 0,
        })
        
        logger.info(f"NPCOrchestrator initialized for user {user_id}, conversation {conversation_id}")

    def _is_active_npc(self, npc_id: int) -> bool:
        return npc_id in self._active_npcs

    def _get_mem_mgr(self, npc_id: int, cache_if_active: bool = True) -> NPCMemoryManager:
        # Return cached if present
        mgr = self._mem_mgrs.get(npc_id)
        if mgr:
            return mgr
    
        # Create a new manager; enable reporting only if active
        is_active = self._is_active_npc(npc_id)
        mgr = NPCMemoryManager(
            npc_id,
            self.user_id,
            self.conversation_id,
            enable_reporting=is_active
        )
    
        # Cache only if active and allowed
        if cache_if_active and is_active:
            self._mem_mgrs[npc_id] = mgr
    
        return mgr

    def _prune_mem_mgr_cache(self) -> None:
        # Remove cached managers for NPCs that are no longer active
        for nid in list(self._mem_mgrs.keys()):
            if nid not in self._active_npcs:
                self._mem_mgrs.pop(nid, None)
    
    def _update_active_trackers(self, npc_id: int, status: str) -> None:
        """Keep _active_npcs aligned to observed status transitions and prune caches on inactivity."""
        s = (status or "").lower()
        if s in self._active_status_values:
            if npc_id not in self._active_npcs:
                self._active_npcs.add(npc_id)
        else:
            # Became inactive -> drop from active set and caches
            self._active_npcs.discard(npc_id)
            self._mem_mgrs.pop(npc_id, None)
            # Also drop per-NPC perception cache
            try:
                self._perception.pop(npc_id, None)
            except Exception:
                pass

    async def initialize(self):
            """Initialize all subsystems"""
            try:
                from logic.narrative_events import initialize_player_stats as _init_stats
                await _init_stats(self.user_id, self.conversation_id)
            except Exception as e:
                logger.debug(f"[Stats] player stats init skipped: {e}")
                
            try:
                # Perception is per-NPC now; nothing to initialize globally
                await self._belief_system.initialize()
                await self._lore_manager.initialize()
                
                # Ensure calendar tables if already loaded; otherwise, they'll be ensured on first use
                if self._calendar_system:
                    try:
                        await self._calendar_system['ensure_tables'](self.user_id, self.conversation_id)
                    except Exception as ce:
                        logger.debug(f"[Calendar] ensure_tables failed during init: {ce}")
            except Exception as e:
                logger.error(f"Failed to initialize belief/lore systems: {e}")
    
            try:
                from logic.addiction_system_sdk import register_with_governance
                await register_with_governance(self.user_id, self.conversation_id)
            except Exception as e:
                logger.debug(f"[Addictions] governance registration skipped: {e}")
                
            try:
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

    async def _get_learning_manager(self) -> NPCLearningManager:
        if self._learning_manager is None:
            self._learning_manager = NPCLearningManager(self.user_id, self.conversation_id)
            await self._learning_manager.initialize()
        return self._learning_manager

    async def run_learning_adaptation_cycle(self, npc_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        if not self.enable_learning_adaptation:
            return {"learning_disabled": True}
        try:
            mgr = await self._get_learning_manager()
            ids = list(npc_ids or list(self._active_npcs))[:20]
            res = await mgr.run_regular_adaptation_cycle(ids)
    
            changed: List[int] = []
            for k, payload in (res.get("npc_adaptations") or {}).items():
                try:
                    nid = int(k)
                except Exception:
                    nid = k
                mem = (payload or {}).get("memory_learning") or {}
                rel = (payload or {}).get("relationship_adaptation") or {}
                if (mem.get("intensity_change", 0) != 0) or rel.get("adapted"):
                    changed.append(nid)
    
            for nid in changed:
                self.invalidate_npc_state(nid, reason="learning_cycle")
    
            if changed:
                try:
                    await self._maybe_evaluate_behavior(changed, reason="post_learning_cycle", use_user_model=False)
                except Exception as be:
                    logger.debug(f"[Learning] behavior eval failed post cycle: {be}")
    
            return res
        except Exception as e:
            logger.exception(f"run_learning_adaptation_cycle failed: {e}")
            return {"error": str(e)}

    async def _learning_record_player_interaction(
        self,
        npc_id: int,
        interaction_type: str,
        interaction_details: Dict[str, Any],
        player_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.enable_learning_adaptation:
            return {"learning_disabled": True}
    
        try:
            mgr = await self._get_learning_manager()
            result = await mgr.process_event_for_learning(
                event_text=interaction_details.get("summary") or interaction_type,
                event_type=interaction_type,
                npc_ids=[npc_id],
                player_response=player_response
            )
    
            npc_res = result.get("npc_learning", {}).get(npc_id) or {}
            # If LoreSystem changed stats (e.g., intensity), invalidate caches and optionally run behavior eval
            changed = bool(npc_res.get("stats_updated")) or bool(
                (npc_res.get("adaptation_results") or {}).get("intensity_change", 0)
            )
            if changed:
                self.invalidate_npc_state(npc_id, reason="learning_adaptation")
                try:
                    await self._maybe_evaluate_behavior([npc_id], reason="post_learning_adaptation", use_user_model=False)
                except Exception as be:
                    logger.debug(f"[Learning] behavior eval failed post adaptation for NPC {npc_id}: {be}")
            return npc_res
        except Exception as e:
            logger.debug(f"[Learning] record interaction failed for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    # ==================== SCENE BUNDLE METHODS ====================
    async def get_scene_bundle(self, scope: 'SceneScope') -> Dict[str, Any]:
        start_time = time.time()
        scene_key = generate_scene_cache_key(scope)
        cache_key = f"{scene_key}{NPCS_SECTION_SUFFIX}"
    
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

            # Conflict summary (if any, very small)
            try:
                conf = await self._get_conflict_state_for_npc(npc_id)
                if conf:
                    npc_entry['conflict'] = {
                        'id': conf.get('conflict_id'),
                        'type': conf.get('type'),
                        'active': conf.get('active', True),
                        'severity': conf.get('intensity'),  # 0..100
                        'phase': conf.get('phase')
                    }
            except Exception as e:
                logger.debug(f"[Conflict] scene bundle conflict fetch failed for NPC {npc_id}: {e}")

            if self._include_lore_in_scene_bundle:
                try:
                    lore_ctx = await self._lore_manager.get_lore_context(npc_id, "profile")
                    knowledge = (lore_ctx or {}).get("knowledge") or []
                    if knowledge:
                        npc_entry['lore'] = [
                            {"name": k.get("name"), "lvl": k.get("knowledge_level"), "id": k.get("lore_id"), "type": k.get("lore_type")}
                            for k in knowledge[:2]
                        ]
                except Exception as e:
                    logger.debug(f"[Lore] scene-bundle fetch failed for NPC {npc_id}: {e}")
    
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
                'conflict': {
                    'id': conf.get('conflict_id') if conf else None,
                    'active': bool(conf.get('active')) if conf else False,
                    'severity': conf.get('intensity') if conf else None
                }
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
    
    async def _mem_orch(self):
        """Get the MemoryOrchestrator for this user/conversation."""
        from memory.memory_orchestrator import get_memory_orchestrator, EntityType
        return await get_memory_orchestrator(self.user_id, self.conversation_id)

    async def _get_decision_engine(self, npc_id: int) -> NPCDecisionEngine:
        now = time.time()
        engine = self._decision_engines.get(npc_id)
        ts = self._decision_engine_timestamps.get(npc_id, 0.0)
        if engine and (now - ts) < self._decision_engine_ttl:
            return engine
    
        # Create or refresh
        engine = await NPCDecisionEngine.create(npc_id, self.user_id, self.conversation_id)
        self._decision_engines[npc_id] = engine
        self._decision_engine_timestamps[npc_id] = now
        return engine

    async def decide_for_npc(
        self,
        npc_id: int,
        perception_override: Optional[Dict[str, Any]] = None,
        available_actions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        engine = await self._get_decision_engine(npc_id)
    
        # Build a minimal perception dict (DecisionEngine will validate/accept dict)
        # Keep this light; the engine tools will fetch deeper bits as needed.
        if perception_override:
            perception = perception_override
        else:
            # Pull what we already have cheaply (avoid heavy memory calls here)
            snapshot = await self.get_npc_snapshot(npc_id, light=True)
            try:
                per = self._get_perception(npc_id)
                env = await per.fetch_environment_data(PerceptionContext(location=snapshot.location))
            except Exception:
                env = {}
            # Very small relationship seed for player; engine tools will fetch fuller state if needed
            rel = {"player": {"link_level": max(snapshot.trust, snapshot.closeness)}}
            # Time context (light)
            time_ctx = {}
            try:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch("""
                        SELECT key, value FROM CurrentRoleplay
                        WHERE key IN ('TimeOfDay') AND user_id=$1 AND conversation_id=$2
                    """, self.user_id, self.conversation_id)
                time_ctx = {r["key"]: r["value"] for r in rows} if rows else {}
            except Exception:
                pass
    
            perception = {
                "environment": {
                    "location": snapshot.location,
                    "entities_present": [
                        {"type": "npc", "id": other_id}
                        for other_id in self._location_index.get(snapshot.location, set())
                        if other_id != npc_id
                    ]
                } | (env or {}),
                "relevant_memories": [],              # Keep empty; engine tools can fetch
                "relationships": rel,
                "emotional_state": snapshot.emotional_state or {},
                "mask": {"integrity": snapshot.mask_integrity, "presented_traits": {}, "hidden_traits": {}},
                "beliefs": [],                        # Engine tools can fetch by itself if needed
                "time_context": {"time_of_day": time_ctx.get("TimeOfDay", "Morning")},
                "narrative_context": {}
            }

            # Conflict-aware narrative context
            try:
                conf = await self._get_conflict_state_for_npc(npc_id)
            except Exception:
                conf = None

            perception["narrative_context"] = {
                "conflict": {
                    "active": bool(conf.get("active")) if conf else False,
                    "severity": conf.get("intensity") if conf else None,
                    "phase": conf.get("phase") if conf else None,
                } if conf else {}
            }
    
        decision = await engine.decide(perception=perception, available_actions=available_actions)
    
        # Optional: log beliefs + invalidate caches
        try:
            from types import SimpleNamespace
            ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
            await self._belief_system.process_event_for_beliefs(
                ctx=ctx_obj,
                event_text=f"NPC chose action: {decision.get('type', 'unknown')} - {decision.get('description','')}",
                event_type="npc_decision",
                npc_ids=[npc_id],
                factuality=1.0
            )
        except Exception as e:
            logger.debug(f"[Beliefs] decision hook failed for NPC {npc_id}: {e}")
    
        # Mark changed for scene deltas
        self._notify_npc_changed(npc_id)
        return decision or {}

    def _invalidate_conflict_cache(self, npc_id: Optional[int] = None) -> None:
        if npc_id is None:
            self._conflict_cache.clear()
            return
        try:
            self._conflict_cache.pop(int(npc_id), None)
        except Exception:
            pass

    def invalidate_npc_state(self, npc_id: int, reason: str = "") -> None:
        """
        Invalidate a single NPC's cached state and mark deltas.
        Canonical single version; replace duplicates with this one.
        """
        try:
            self._snapshot_cache.pop(npc_id, None)
        except Exception:
            pass
        # Also invalidate per-NPC conflict cache entry
        try:
            self._invalidate_conflict_cache(npc_id)
        except Exception:
            pass
        # Mark changed for delta feeds and clear bundle associations
        self._notify_npc_changed(npc_id)
        if reason:
            logger.debug(f"[Orchestrator] Invalidated NPC {npc_id} ({reason})")

    async def _get_conflict_state_for_npc(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """Return a tiny summary: {conflict_id, type, intensity, active} for one NPC, cached."""
        # Cache fast path
        entry = self._conflict_cache.get(npc_id)
        if entry and (time.time() - entry[1] < self._conflict_cache_ttl):
            return entry[0]

        synth = await self._get_conflict_synth()
        if not synth:
            return None

        try:
            scope = self._ConflictScope(location_id=None, npc_ids=[npc_id])
            bundle = await synth.get_scene_bundle(scope)
            # Bundle shape: bundle["active"] holds list of conflicts (see synthesizer code)
            for c in (bundle.get("active") or []):
                stks = c.get("stakeholders") or []
                if npc_id in stks:
                    # Map intensity [0..1] to a simple severity [0..100]
                    intensity = float(c.get("intensity", 0.0) or 0.0)
                    state = {
                        "conflict_id": c.get("id"),
                        "type": c.get("type"),
                        "intensity": max(0, min(100, int(round(intensity * 100)))),
                        "active": True,
                        "phase": c.get("phase"),
                        "canonical": bool(c.get("canonical")),
                    }
                    self._conflict_cache[npc_id] = (state, time.time())
                    return state
        except Exception as e:
            logger.debug(f"[Conflict] per-NPC conflict fetch failed for {npc_id}: {e}")

        # No conflict matched
        self._conflict_cache[npc_id] = (None, time.time())
        return None
    
    def _get_perception(self, npc_id: int) -> EnvironmentPerception:
        p = self._perception.get(npc_id)
        if p is None:
            p = EnvironmentPerception(npc_id, self.user_id, self.conversation_id)
            self._perception[npc_id] = p
        return p
    
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
        key = f"{scene_key}{NPCS_SECTION_SUFFIX}"
    
        # Drop the cached bundle
        self._bundle_cache.pop(key, None)
    
        # Remove this key from every NPC’s reverse index set
        for nid, keys in list(self._bundle_index.items()):
            if key in keys:
                keys.discard(key)
                if not keys:
                    self._bundle_index.pop(nid, None)

    async def route_player_conversation_beliefs(
        self,
        npc_ids: List[int],
        text: str,
        topic: Optional[str] = "general",
        credibility: float = 0.7
    ) -> None:
        """Push a player utterance into beliefs and consolidate (throttled)."""
        if not npc_ids or not text:
            return
        from types import SimpleNamespace
        for nid in npc_ids[:10]:
            ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=nid)
            try:
                await self._belief_system.process_conversation_for_beliefs(
                    ctx=ctx_obj,
                    conversation_text=text,
                    speaker_id="player",
                    listener_id=nid,
                    topic=topic or "general",
                    credibility=float(credibility),
                )
            except Exception as e:
                logger.warning(f"[Beliefs] routing failed for NPC {nid}: {e}")

        await self._maybe_consolidate_beliefs(
            npc_ids,
            topic_filter=topic or "general",
            reason="npc_handler"
        )

    async def synthesize_player_interaction(
        self,
        npc_id: int,
        interaction_type: str,
        player_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Orchestrated conversational pipeline:
        - Gather context from orchestrator subsystems
        - Delegate response generation to NPCHandler (no side effects)
        - Apply side effects here (relationships, stats via LoreSystem, memory, beliefs)
        - Invalidate caches and return a clean payload for final synthesis
        """
        from memory.memory_orchestrator import get_memory_orchestrator, EntityType
        # 1) Gather preloaded context
        snap = await self.get_npc_snapshot(npc_id, light=True)
    
        # Relationship dims for preloading
        try:
            rel = await self.get_relationship_dynamics("player", self.user_id, "npc", npc_id)
            rel_pre = {
                "trust": rel["dimensions"]["trust"],
                "respect": rel["dimensions"]["respect"],
                "affection": rel["dimensions"]["affection"],
                "patterns": rel.get("patterns", []),
                "archetypes": rel.get("archetypes", [])
            }
        except Exception:
            rel_pre = {"trust": 0, "respect": 0, "affection": 0, "patterns": [], "archetypes": []}
    
        # Memories (small) via MemoryOrchestrator
        try:
            orch = await self._mem_orch()
            rm = await orch.retrieve_memories(
                entity_type=EntityType.NPC.value,
                entity_id=npc_id,
                limit=5
            )
            memories_pre = rm.get("memories", [])
        except Exception:
            memories_pre = []
    
        preloaded = {
            "npc_details": {
                "npc_id": snap.npc_id,
                "npc_name": snap.name,
                "dominance": snap.dominance,
                "cruelty": snap.cruelty,
                "closeness": snap.closeness,
                "trust": snap.trust,
                "respect": snap.respect,
                "intensity": snap.intensity,
                "personality_traits": snap.personality_traits,
                "current_location": snap.location,
            },
            "memories": memories_pre,
            "relationship": rel_pre
        }
    
        # 2) Generate a proposal (no side effects inside handler)
        proposal: NPCInteractionProposal = await self._responder.generate_interaction_proposal(
            npc_id=npc_id,
            interaction_type=interaction_type,
            player_input=player_input,
            context=context or {},
            preloaded=preloaded
        )
    
        # 3) Apply side effects centrally
    
        # 3a) Relationship update
        if proposal.proposed_relationship_interaction:
            try:
                await self.process_relationship_interaction(
                    npc_id=npc_id,
                    interaction_type=proposal.proposed_relationship_interaction,
                    context="conversation",
                    intensity=1.0
                )
            except Exception as e:
                logger.debug(f"[Orchestrator] relationship apply failed: {e}")
    
        # 3b) Stat changes via LoreSystem (treat as deltas)
        if proposal.proposed_stat_changes:
            try:
                from lore.core.lore_system import LoreSystem
                from agents import RunContextWrapper
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                ctx = RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                    'npc_id': npc_id
                })
    
                # Load current values and compute clamped updates
                async with get_db_connection_context() as conn:
                    current = {}
                    for stat, delta in proposal.proposed_stat_changes.items():
                        row = await conn.fetchrow(
                            f"SELECT {stat} FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3",
                            self.user_id, self.conversation_id, npc_id
                        )
                        if row and stat in row:
                            current[stat] = row[stat]
    
                updates = {}
                for stat, delta in proposal.proposed_stat_changes.items():
                    if stat in current:
                        new_val = max(0, min(100, int(current[stat]) + int(delta)))
                        if new_val != current[stat]:
                            updates[stat] = new_val
    
                if updates:
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="NPCStats",
                        entity_identifier={"npc_id": npc_id},
                        updates=updates,
                        reason="Player interaction (orchestrator-applied)"
                    )
            except Exception as e:
                logger.debug(f"[Orchestrator] stat apply failed: {e}")
    
        # 3c) Memory add via MemoryOrchestrator
        if proposal.memory_note:
            try:
                await orch.integrated_add_memory(
                    entity_type=EntityType.NPC.value,
                    entity_id=npc_id,
                    memory_text=proposal.memory_note,
                    memory_kwargs={
                        "significance": 3,  # medium
                        "tags": proposal.memory_tags or ["interaction"],
                        "metadata": {"source": "npc_orchestrator"}
                    }
                )
            except Exception as e:
                logger.debug(f"[Orchestrator] memory add failed: {e}")
    
        # 3d) Beliefs (conversation)
        try:
            await self.route_player_conversation_beliefs(
                npc_ids=[npc_id],
                text=player_input,
                topic=context.get("topic", "general"),
                credibility=float(context.get("credibility", 0.7))
            )
        except Exception as e:
            logger.debug(f"[Orchestrator] belief route failed: {e}")
    
        # Notify conflict synthesizer of this conversation scene (non-blocking)
        try:
            synth = await self._get_conflict_synth()
            if synth:
                scene_ctx = {
                    "scene_id": int(time.time() * 1000) % 2_147_483_647,
                    "scene_type": interaction_type or "conversation",
                    "characters_present": [npc_id],
                    "location_id": None,
                    "dialogue": [{"speaker": "player", "text": player_input[:300]}] if player_input else [],
                    "timestamp": datetime.utcnow().isoformat()
                }
                asyncio.create_task(synth.process_scene(scene_ctx))
        except Exception as e:
            logger.debug(f"[Conflict] conversation notify failed for NPC {npc_id}: {e}")
    
        # 3e) Invalidate caches
        self.invalidate_npc_state(npc_id, reason="player_interaction")
    
        # 3f) Learning & adaptation (intensity etc.) after conversation
        try:
            player_response = None
            if isinstance(proposal.meta, dict):
                player_response = {
                    "compliance_level": proposal.meta.get("player_compliance", 0),
                    "emotional_response": proposal.meta.get("player_emotion", "neutral"),
                }
    
            interaction_details = {
                "summary": player_input[:200],
                "topic": context.get("topic", "general"),
                "duration": context.get("duration", 0),
            }
    
            await self._learning_record_player_interaction(
                npc_id=npc_id,
                interaction_type=interaction_type if interaction_type else "conversation",
                interaction_details=interaction_details,
                player_response=player_response,
            )
        except Exception as e:
            logger.debug(f"[Learning] interaction hook failed for NPC {npc_id}: {e}")
    
        # 3g) Optional: emergent addiction analysis from this scene (throttled externally if needed)
        try:
            text_for_analysis = " ".join(filter(None, [
                player_input or "",
                (proposal.response or {}).get("text") if isinstance(proposal.response, dict) else ""
            ]))
            if text_for_analysis:
                add_pack = await self.analyze_emergent_addictions(text_for_analysis, npcs=[{"npc_id": npc_id}])
                if isinstance(add_pack, dict) and add_pack.get("update_results"):
                    # Attach a tiny summary for the final synthesizer if desired
                    pass
        except Exception as e:
            logger.debug(f"[Addictions] emergent analysis failed: {e}")
    
        # 4) Return payload for the final response synthesizer
        return {
            "npc_id": npc_id,
            "npc_name": snap.name,
            "response": proposal.response,
            "applied_stat_changes": proposal.proposed_stat_changes,
            "meta": proposal.meta
        }
    
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

                # Conflict change detection (id/active/severity)
                try:
                    conf = await self._get_conflict_state_for_npc(npc_id)
                    old_conf = old_state.get('conflict', {}) if old_state else {}
                    new_id = conf.get('conflict_id') if conf else None
                    if new_id != old_conf.get('id'):
                        change['conflict_id'] = new_id
                        change_types.append('conflict_id')
                    new_active = bool(conf.get('active')) if conf else False
                    if new_active != bool(old_conf.get('active', False)):
                        change['conflict_active'] = new_active
                        change_types.append('conflict_active')
                    new_sev = conf.get('intensity') if conf else None
                    if new_sev != old_conf.get('severity'):
                        change['conflict_severity'] = new_sev
                        change_types.append('conflict_severity')
                except Exception:
                    pass

                # Update state cache (also persist conflict)
                self._scene_state_cache[npc_id] = {
                    'status': snapshot.status,
                    'location': snapshot.location,
                    'trust': snapshot.trust,
                    'respect': snapshot.respect,
                    'closeness': snapshot.closeness,
                    'emotional_state': snapshot.emotional_state,
                    'conflict': {
                        'id': (conf.get('conflict_id') if conf else None),
                        'active': bool(conf.get('active')) if conf else False,
                        'severity': (conf.get('intensity') if conf else None)
                    }
                }
                
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

    async def coordinate_group_interaction(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any],
        available_actions: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Run the NPCAgentCoordinator to make a group decision, then
        feed salient results into beliefs and refresh caches.
        """
        try:
            out = await self._agent_coordinator.make_group_decisions(
                npc_ids=npc_ids,
                shared_context=shared_context,
                available_actions=available_actions
            )
            self.metrics['group_decisions'] += 1

            # Belief/event logging for each NPC
            if out and (out.get("group_actions") or out.get("individual_actions")):
                from types import SimpleNamespace
                for nid in npc_ids[:10]:
                    try:
                        ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=nid)
                        await self._belief_system.process_event_for_beliefs(
                            ctx=ctx_obj,
                            event_text="Group decision executed",
                            event_type="group_decision",
                            npc_ids=[nid],
                            factuality=1.0
                        )
                        # mark changed for delta feeds
                        self._notify_npc_changed(nid)
                    except Exception as be:
                        logger.warning(f"[Beliefs] group-decision hook failed for NPC {nid}: {be}")

            return out or {}
        except Exception as e:
            logger.exception(f"group interaction coordination failed: {e}")
            return {"error": str(e)}

    async def apply_relationship_drift(self) -> Dict[str, Any]:
        try:
            mgr = await self._get_dynamic_relationship_manager()
            await mgr.apply_daily_drift()
            # Flush any pending relationship writes
            await mgr._flush_updates()
            # Mark active NPCs as changed for deltas (lightweight)
            for nid in list(self._active_npcs):
                self._notify_npc_changed(nid)
            return {"success": True}
        except Exception as e:
            logger.debug(f"[Relationships] apply_relationship_drift failed: {e}")
            return {"success": False, "error": str(e)}

    async def handle_group_player_action(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Delegate a player action that impacts multiple NPCs to the NPCAgentCoordinator.
        Includes belief routing + throttled consolidation.
        """
        try:
            res = await self._agent_coordinator.handle_player_action(
                player_action=action,
                context=context,
                npc_ids=npc_ids
            )
            self.metrics['group_player_actions'] += 1

            # Belief integration (reuse your existing text if present)
            text = action.get("description", "") or ""
            try:
                if res and res.get("npc_responses"):
                    affected = npc_ids or [r.get("npc_id") for r in res["npc_responses"] if isinstance(r, dict) and r.get("npc_id")]
                    affected = [n for n in affected if n is not None][:10]
                else:
                    affected = npc_ids or []

                if text and affected:
                    from types import SimpleNamespace
                    for nid in affected:
                        ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=nid)
                        await self._belief_system.process_conversation_for_beliefs(
                            ctx=ctx_obj,
                            conversation_text=text,
                            speaker_id="player",
                            listener_id=nid,
                            topic=context.get("topic", "group"),
                            credibility=float(context.get("credibility", 0.7))
                        )
                        self._notify_npc_changed(nid)
                        

                    await self._maybe_consolidate_beliefs(
                        affected,
                        topic_filter=context.get("topic", "group"),
                        reason="post_group_player_action"
                    )
            except Exception as be:
                logger.warning(f"[Beliefs] post-group-action belief routing failed: {be}")

            # Notify conflict synthesizer about this scene/action (non-blocking)
            try:
                synth = await self._get_conflict_synth()
                if synth:
                    scene_ctx = {
                        "scene_id": int(time.time() * 1000) % 2_147_483_647,
                        "scene_type": action.get("type") or "group_action",
                        "characters_present": (affected or npc_ids or [])[:10],
                        "location_id": None,  # unknown id; pass None
                        "dialogue": [{"speaker": "player", "text": action.get("description", "")}] if action.get("description") else [],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    asyncio.create_task(synth.process_scene(scene_ctx))
            except Exception as e:
                logger.debug(f"[Conflict] scene notify failed: {e}")

            # Learning adaptation for affected NPCs (if any)
            try:
                if text and affected and self.enable_learning_adaptation:
                    mgr = await self._get_learning_manager()
                    await mgr.process_event_for_learning(
                        event_text=text,
                        event_type=action.get("type") or action.get("interaction_type") or "group_action",
                        npc_ids=affected,
                        player_response={
                            "compliance_level": context.get("player_compliance", 0),
                            "emotional_response": context.get("player_emotion", "neutral"),
                        } if isinstance(context, dict) else None,
                    )
                    for nid in affected:
                        self.invalidate_npc_state(nid, reason="group_learning")
                    await self._maybe_evaluate_behavior(affected, reason="post_group_learning", use_user_model=False)
            except Exception as le:
                logger.debug(f"[Learning] group learning hook failed: {le}")

            return res or {"npc_responses": []}
        except Exception as e:
            logger.exception(f"group player action failed: {e}")
            return {"error": str(e), "npc_responses": []}


    async def evaluate_and_apply_npc_behavior(
        self,
        npc_id: int,
        use_user_model: bool = True
    ) -> Dict[str, Any]:
        """
        Runs BehaviorEvolution for one NPC, applies changes via LoreSystem,
        notifies caches, and feeds significant changes into Beliefs.
        """
        try:
            # Evaluate (optionally with user preference model)
            if use_user_model:
                eval_result = await self._behavior_evolution.evaluate_npc_scheming_with_user_model(npc_id)
            else:
                eval_result = await self._behavior_evolution.evaluate_npc_scheming(npc_id)

            if "error" in eval_result:
                return eval_result

            # Apply to DB via LoreSystem governance
            applied = await self._behavior_evolution.apply_scheming_adjustments(npc_id, eval_result)
            eval_result["applied"] = bool(applied)
            self.metrics['behavior_evals'] += 1
            if applied:
                self.metrics['behavior_applies'] += 1

                # Invalidate snapshot + bundles
                if npc_id in self._snapshot_cache:
                    self._snapshot_cache.pop(npc_id, None)
                self._notify_npc_changed(npc_id)

                # Belief hooks for salient flags
                try:
                    significant_bits = []
                    if eval_result.get("betrayal_planning"):
                        significant_bits.append("betrayal_planning")
                    if eval_result.get("targeting_player"):
                        significant_bits.append("targeting_player")
                    if (eval_result.get("scheme_level", 0) >= 5):
                        significant_bits.append("high_scheming")

                    if significant_bits:
                        from types import SimpleNamespace
                        ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                        await self._belief_system.process_event_for_beliefs(
                            ctx=ctx_obj,
                            event_text=f"Behavior evolved: {', '.join(significant_bits)}",
                            event_type="behavior_evolution",
                            npc_ids=[npc_id],
                            factuality=1.0
                        )
                        # Throttled consolidation specific to behavior
                        await self._maybe_consolidate_beliefs([npc_id], topic_filter="behavior_evolution", reason="post_behavior_eval")
                except Exception as be:
                    logger.warning(f"[Beliefs] behavior-evolution belief hook failed for NPC {npc_id}: {be}")

            return eval_result
        except Exception as e:
            logger.exception(f"Behavior evolution failed for NPC {npc_id}: {e}")
            return {"error": str(e)}

    async def _maybe_evaluate_behavior(
        self,
        npc_ids: List[int],
        reason: str = "",
        use_user_model: bool = False
    ) -> None:
        """
        Throttle & run behavior evolution for a set of NPCs, apply adjustments,
        invalidate caches, and mark deltas.
        """
        now = time.time()
        to_run: List[int] = []
        for nid in npc_ids:
            last = self._last_behavior_eval.get(nid, 0.0)
            if now - last >= self._behavior_eval_cooldown:
                to_run.append(nid)
                self._last_behavior_eval[nid] = now
    
        if not to_run:
            return
    
        try:
            self.metrics['behavior_evals'] += len(to_run)
            for nid in to_run:
                try:
                    if use_user_model:
                        adjustments = await self._behavior_evolution.evaluate_npc_scheming_with_user_model(nid)
                    else:
                        adjustments = await self._behavior_evolution.evaluate_npc_scheming(nid)
    
                    if isinstance(adjustments, dict) and "error" not in adjustments:
                        applied = await self._behavior_evolution.apply_scheming_adjustments(nid, adjustments)
                        if applied:
                            # FIXED METRIC KEY
                            self.metrics['behavior_applies'] += 1
                            # Invalidate snapshot cache & mark change for delta feeds
                            try:
                                self._snapshot_cache.pop(nid, None)
                            except Exception:
                                pass
                            self._notify_npc_changed(nid)
                except Exception as e:
                    logger.warning(f"[Behavior] evolution failed for NPC {nid}: {e}")
    
            if reason:
                logger.debug(f"[Behavior] Evaluated for {to_run} ({reason})")
        except Exception as e:
            logger.warning(f"[Behavior] batch evaluation failed ({reason}): {e}")

    
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
        from memory.memory_orchestrator import get_memory_orchestrator, EntityType
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
    
                self._update_active_trackers(npc_id, row_map.get("status", "active"))
                
                # Get canonical events if canon enabled
                canonical_events = []
                if self.enable_canon:
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
    
        # Heavy fields (optional)
        emotional_state = {}
        recent_memories = []
        mask_integrity = row_map.get("mask_integrity") or 100
    
        if not light:
            # Emotion via MemoryOrchestrator (preferred)
            try:
                orch = await self._mem_orch()
                emo = await orch.emotional_manager.get_entity_emotional_state(
                    entity_type=EntityType.NPC.value,
                    entity_id=npc_id
                )
                if isinstance(emo, dict) and emo:
                    emotional_state = emo
            except Exception:
                emotional_state = {}
    
            # Mask integrity via MemoryOrchestrator (overrides DB)
            try:
                orch = await self._mem_orch()
                mask = await orch.get_npc_mask(npc_id)
                if isinstance(mask, dict) and "integrity" in mask:
                    mask_integrity = int(mask.get("integrity", mask_integrity))
            except Exception:
                pass
    
            # Memory recall via MemoryOrchestrator (light)
            try:
                orch = await self._mem_orch()
                rm = await orch.retrieve_memories(
                    entity_type=EntityType.NPC.value,
                    entity_id=npc_id,
                    limit=5
                )
                recent_memories = rm.get("memories", [])
            except Exception:
                recent_memories = []
    
        # Lore summary (heavy only; keep tiny)
        lore_summary = []
        if not light:
            try:
                lore_ctx = await self._lore_manager.get_lore_context(npc_id, "profile")
                knowledge = (lore_ctx or {}).get("knowledge") or []
                for k in knowledge[:3]:
                    lore_summary.append({
                        "lore_type": k.get("lore_type"),
                        "lore_id": k.get("lore_id"),
                        "name": k.get("name"),
                        "knowledge_level": k.get("knowledge_level"),
                    })
            except Exception as e:
                logger.debug(f"[Lore] summary fetch failed for NPC {npc_id}: {e}")
    
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
            npc_id=int(npc_id),
            name=row_map["npc_name"],
            role=row_map.get("role", "NPC"),
            status=row_map.get("status", "active"),
            location=row_map["current_location"],
            canonical_events=canonical_events,
            personality_traits=traits,
            lore_summary=lore_summary,
            dominance=row_map.get("dominance") or 50,
            cruelty=row_map.get("cruelty") or 50,
            trust=row_map.get("trust") or 0,
            respect=row_map.get("respect") or 0,
            closeness=row_map.get("closeness") or 0,
            intensity=row_map.get("intensity") or 0,
            mask_integrity=mask_integrity,
            emotional_state=emotional_state,
            recent_memories=recent_memories,
            scheming_level=row_map.get("scheming_level", 0),
            betrayal_planning=row_map.get("betrayal_planning", False),
            special_mechanics=special
        )
        
        return snapshot

    async def update_player_addiction(
        self,
        addiction_type: str,
        npc_id: Optional[int] = None,
        progression_multiplier: float = 1.0,
        player_name: str = "Chase"
    ) -> Dict[str, Any]:
        """
        Progress/regress a specific addiction; returns change summary.
        """
        try:
            from logic.addiction_system_sdk import process_addiction_update
            res = await process_addiction_update(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                player_name=player_name,
                addiction_type=addiction_type,
                progression_multiplier=float(progression_multiplier),
                target_npc_id=int(npc_id) if npc_id is not None else None
            )
            # Optional belief hook on progression/regression
            try:
                upd = res.get("update", {})
                if upd and ("progressed" in upd or "regressed" in upd):
                    changed = []
                    if upd.get("progressed"):
                        changed.append("progressed")
                    if upd.get("regressed"):
                        changed.append("regressed")
                    if changed:
                        from types import SimpleNamespace
                        ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                        await self._belief_system.process_event_for_beliefs(
                            ctx=ctx_obj,
                            event_text=f"Addiction {addiction_type} {', '.join(changed)} to level {upd.get('new_level', 0)}",
                            event_type="addiction_update",
                            npc_ids=[npc_id] if npc_id else [],
                            factuality=1.0
                        )
            except Exception:
                pass
            return res
        except Exception as e:
            logger.error(f"update_player_addiction failed: {e}")
            return {"error": str(e)}
    
    async def get_player_addiction_status(self, player_name: str = "Chase") -> Dict[str, Any]:
        """
        Snapshot of current addictions with labels; safe for UI.
        """
        try:
            from logic.addiction_system_sdk import get_addiction_status
            return await get_addiction_status(self.user_id, self.conversation_id, player_name)
        except Exception as e:
            logger.error(f"get_player_addiction_status failed: {e}")
            return {"has_addictions": False, "error": str(e)}
    
    async def apply_addiction_effects_from_status(self, status: Dict[str, Any], player_name: str = "Chase") -> Dict[str, Any]:
        """
        Expand an addiction status object into narrative effects.
        """
        try:
            from logic.addiction_system_sdk import process_addiction_effects
            return await process_addiction_effects(self.user_id, self.conversation_id, player_name, status or {})
        except Exception as e:
            logger.error(f"apply_addiction_effects_from_status failed: {e}")
            return {"effects": [], "has_effects": False, "error": str(e)}
    
    async def analyze_emergent_addictions(self, recent_text: str, npcs: Optional[List[dict]] = None, player_name: str = "Chase") -> Dict[str, Any]:
        """
        Analyze recent narrative/events for emergent addictions and apply updates.
        """
        try:
            from logic.addiction_emergence import analyze_and_apply_emergent_addictions
            return await analyze_and_apply_emergent_addictions(
                self.user_id, self.conversation_id, player_name, recent_text or "", npcs=npcs
            )
        except Exception as e:
            logger.error(f"analyze_emergent_addictions failed: {e}")
            return {"error": str(e)}
    
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

    async def get_scene_brief(self, scope) -> Dict[str, Any]:
        brief: Dict[str, Any] = {"anchors": {}, "signals": {}, "links": {}}
        try:
            npc_ids: List[int] = []
            for nid in list(getattr(scope, "npc_ids", []) or [])[:10]:
                try:
                    npc_ids.append(int(nid))
                except Exception:
                    continue

            if npc_ids:
                brief["anchors"]["npc_ids"] = npc_ids

            # related_npcs heuristic: if multiple in scene, surface them
            if len(npc_ids) > 1:
                brief["links"]["related_npcs"] = npc_ids[:10]

            # minimal relationship hints for player → npc
            rel_map: Dict[str, Dict[str, int]] = {}
            for nid in npc_ids[:5]:
                try:
                    rel = await self.get_relationship_dynamics("player", self.user_id, "npc", int(nid))
                    dims = (rel or {}).get("dimensions", {}) or {}
                    rel_map[str(nid)] = {
                        "player_trust": int(dims.get("trust", 0)),
                        "closeness": int(dims.get("intimacy", 0)),
                    }
                except Exception:
                    continue
            if rel_map:
                brief["links"]["relations_by_npc"] = rel_map
        except Exception:
            pass
        return brief
    
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
        from memory.memory_orchestrator import get_memory_orchestrator, EntityType
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
        
        # Add memory for NPC about the scheduled event (via MemoryOrchestrator)
        try:
            orch = await self._mem_orch()
            await orch.integrated_add_memory(
                entity_type=EntityType.NPC.value,
                entity_id=npc_id,
                memory_text=f"I have {event_name} scheduled for {time_of_day} on day {day}",
                memory_kwargs={
                    "significance": int(priority),
                    "tags": ["calendar", "scheduled"],
                    "metadata": {"source": "npc_orchestrator"}
                }
            )
        except Exception as e:
            logger.debug(f"[Orchestrator] schedule memory add failed for NPC {npc_id}: {e}")
        
        self._notify_npc_changed(npc_id)
        return result

    async def decide_for_scope(self, scope: 'SceneScope') -> Dict[int, Dict[str, Any]]:
        npc_ids = list(getattr(scope, "npc_ids", []))[:10]
        out: Dict[int, Dict[str, Any]] = {}
        for nid in npc_ids:
            try:
                out[nid] = await self.decide_for_npc(nid)
            except Exception as e:
                logger.debug(f"decide_for_scope failed for NPC {nid}: {e}")
        return out
    
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

    async def update_beliefs_on_knowledge_discovery(
        self,
        npc_id: int,
        knowledge_type: str,
        knowledge_id: int
    ) -> Dict[str, Any]:
        """Forward knowledge updates into the belief system."""
        try:
            return await self._belief_system.update_beliefs_on_knowledge_discovery(
                npc_id=int(npc_id),
                knowledge_type=knowledge_type,
                knowledge_id=int(knowledge_id),
            )
        except Exception as e:
            logger.exception("update_beliefs_on_knowledge_discovery failed: %s", e)
            return {"beliefs_updated": False, "error": str(e)}

    async def report_world_event_for_beliefs(
        self,
        text: str,
        event_type: str,
        npc_ids: List[int],
        factuality: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Convenience bridge into the belief system from any other subsystem.
        Example:
            await orchestrator.report_world_event_for_beliefs(
                "A riot broke out in the market", "riot", [12, 15, 22], 0.9
            )
        """
        try:
            return await self._belief_system.process_event_for_beliefs(
                None,  # ctx
                event_text=text,
                event_type=event_type,
                npc_ids=[int(n) for n in npc_ids],
                factuality=float(factuality),
            )
        except Exception as e:
            logger.exception("report_world_event_for_beliefs failed: %s", e)
            return {"event_processed": False, "error": str(e)}

    async def consolidate_beliefs_for_npcs(self, npc_ids: List[int], topic_filter: Optional[str] = None):
        """
        Ask each NPC to form a narrative from their recent memories (meso layer),
        which indirectly influences their macro/worldview stance over time.
        """
        from types import SimpleNamespace
        for npc_id in npc_ids:
            try:
                ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                await self._belief_system.form_narrative_from_recent_events(
                    ctx=ctx_obj,
                    npc_id=npc_id,
                    max_events=5,
                    topic_filter=topic_filter
                )
            except Exception as e:
                logger.warning(f"[Beliefs] consolidate failed for NPC {npc_id}: {e}")

    async def _maybe_consolidate_beliefs(
        self,
        npc_ids: List[int],
        topic_filter: Optional[str] = None,
        reason: str = ""
    ):
        """Throttle and run belief consolidation for a set of NPCs."""
        now = time.time()
        to_run = []
        for nid in npc_ids:
            last = self._last_belief_consolidation.get(nid, 0.0)
            if now - last >= self._consolidation_cooldown:
                to_run.append(nid)
                self._last_belief_consolidation[nid] = now
    
        if not to_run:
            return
    
        try:
            await self.consolidate_beliefs_for_npcs(to_run, topic_filter=topic_filter)
            if reason:
                logger.debug(f"[Beliefs] Consolidated for {to_run} ({reason})")
        except Exception as e:
            logger.warning(f"[Beliefs] Consolidation failed ({reason}): {e}")

    
    async def process_calendar_events_for_all_npcs(self) -> Dict[str, Any]:
        """Process current calendar events for all NPCs and feed them into belief formation."""
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
            await calendar["auto_process"](self.user_id, self.conversation_id, year, month, day, time_of_day)
        except Exception as e:
            logger.exception(f"Auto-process missed events failed: {e}")
    
        touched_npcs: Set[int] = set()
    
        # Check current events for each NPC
        for npc_id in list(self._active_npcs):
            try:
                events = await self.check_npc_current_events(npc_id)
            except Exception as e:
                logger.exception(f"check_npc_current_events failed for NPC {npc_id}: {e}")
                continue
    
            if not events:
                continue
    
            results["npc_statuses"][npc_id] = "has_events"
            self._notify_npc_changed(npc_id)
            touched_npcs.add(npc_id)
    
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
                        # Feed into beliefs: happened event
                        try:
                            ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                            event_text = event.get("event_name") or f"NPC {npc_id} participated in an event"
                            await self._belief_system.process_event_for_beliefs(
                                ctx=ctx_obj,
                                event_text=event_text,
                                event_type=event.get("event_type", "calendar_event"),
                                npc_ids=[npc_id],
                                factuality=1.0
                            )
                        except Exception as be:
                            logger.warning(f"[Beliefs] calendar belief hook failed for NPC {npc_id}: {be}")
    
                    else:
                        await calendar["mark_missed"](self.user_id, self.conversation_id, event["event_id"])
                        results["missed_events"].append({
                            "npc_id": npc_id,
                            "event_id": event.get("event_id"),
                            "event_name": event.get("event_name"),
                            "time_of_day": event.get("time_of_day"),
                            "reason": event.get("missing_requirements", []),
                        })
                        # Feed into beliefs: missed event
                        try:
                            ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                            event_text = f"Missed: {event.get('event_name', 'scheduled event')}"
                            await self._belief_system.process_event_for_beliefs(
                                ctx=ctx_obj,
                                event_text=event_text,
                                event_type="missed_calendar_event",
                                npc_ids=[npc_id],
                                factuality=0.9
                            )
                        except Exception as be:
                            logger.warning(f"[Beliefs] missed-calendar belief hook failed for NPC {npc_id}: {be}")
    
                except Exception as e:
                    logger.exception(f"Failed to process calendar event {event.get('event_id')} for NPC {npc_id}: {e}")
    
        # Throttled consolidation after an event wave
        if touched_npcs:
            await self._maybe_consolidate_beliefs(
                list(touched_npcs),
                topic_filter="calendar_event",
                reason="post_calendar_processing"
            )

        # Behavior evolution after calendar events (throttled)
        if touched_npcs:
            try:
                await self._maybe_evaluate_behavior(list(touched_npcs), reason="post_calendar_processing", use_user_model=False)
            except Exception as e:
                logger.warning(f"[Behavior] post_calendar evolution failed: {e}")

        # Optional: run learning cycle after calendar processing
        try:
            if touched_npcs and self.enable_learning_adaptation:
                await self.run_learning_adaptation_cycle(list(touched_npcs))
        except Exception as e:
            logger.debug(f"[Learning] post-calendar cycle failed: {e}")

        try:
            await self.apply_relationship_drift()
        except Exception as e:
            logger.debug(f"[Relationships] drift pass failed: {e}")

        # Advance conflicts after calendar processing (day tick if day changed, else scene sync)
        try:
            synth = await self._get_conflict_synth()
            if synth:
                # If you detect day transition outside this method, call handle_day_transition(new_day) there.
                # Here we can push a lightweight scene sync for touched NPCs.
                scope = self._ConflictScope(npc_ids=list(touched_npcs)[:10])
                # fetch bundle once (warms caches and updates internal states)
                _ = await synth.get_scene_bundle(scope)
        except Exception as e:
            logger.debug(f"[Conflict] post-calendar scene sync failed: {e}")
    
        return results

    async def _get_npc_bridge(self):
        """Lazy-load the NPCSystemBridge and configure mode."""
        if self._npc_bridge is None:
            try:
                from logic.npc_agent_bridge import NPCSystemBridge
                self._npc_bridge = NPCSystemBridge(self.user_id, self.conversation_id)
                # allow config switch to new system path inside the bridge
                self._npc_bridge.use_new_system = bool(self.config.get("npc_bridge_use_new_system", False))
            except Exception as e:
                logger.debug(f"[Bridge] NPCSystemBridge unavailable: {e}")
                self._npc_bridge = False
        return self._npc_bridge or None
    async def get_conflict_metrics(self) -> Dict[str, Any]:
        synth = await self._get_conflict_synth()
        if not synth:
            return {"enabled": False}
        try:
            return await synth.get_performance_metrics()
        except Exception as e:
            logger.debug(f"[Conflict] metrics fetch failed: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def evaluate_behavior_for_scope(self, scope: 'SceneScope', use_user_model: bool = True) -> None:
        npc_ids = list(getattr(scope, "npc_ids", []))[:10]
        if not npc_ids:
            return None
        await self._maybe_evaluate_behavior(npc_ids, reason="explicit_scope_eval", use_user_model=use_user_model)
        return None

    async def generate_scheming_opportunity(self, npc_id: int, trigger_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Thin bridge to BehaviorEvolution.generate_scheming_opportunity plus cache/Beliefs."""
        try:
            opp = await self._behavior_evolution.generate_scheming_opportunity(npc_id, trigger_event)
            if opp:
                # Mark changed for delta feeds
                self._notify_npc_changed(npc_id)
                # Optional: belief event
                try:
                    from types import SimpleNamespace
                    ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                    await self._belief_system.process_event_for_beliefs(
                        ctx=ctx_obj,
                        event_text=f"Opportunity spotted: {opp.get('type','unknown')}",
                        event_type="scheming_opportunity",
                        npc_ids=[npc_id],
                        factuality=0.9
                    )
                except Exception:
                    pass
            return opp
        except Exception as e:
            logger.warning(f"[Behavior] opportunity generation failed for NPC {npc_id}: {e}")
            return None

    
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

    async def get_player_stats(self, scope: str = "visible", player_name: str = "Chase") -> Dict[str, Any]:
        """
        Get player stats through stats_logic.
        scope: 'visible' | 'hidden' | 'all'
        """
        try:
            from logic.stats_logic import (
                get_player_visible_stats,
                get_player_hidden_stats,
                get_all_player_stats,
            )
            if scope == "all":
                return await get_all_player_stats(self.user_id, self.conversation_id, player_name)
            if scope == "hidden":
                return await get_player_hidden_stats(self.user_id, self.conversation_id, player_name)
            return await get_player_visible_stats(self.user_id, self.conversation_id, player_name)
        except Exception as e:
            logger.error(f"get_player_stats failed: {e}")
            return {}
    
    async def apply_player_stat_changes(self, changes: Dict[str, int], reason: str = "", player_name: str = "Chase") -> Dict[str, Any]:
        """
        Bulk apply changes to player stats via stats_logic.apply_stat_changes
        """
        try:
            from logic.stats_logic import apply_stat_changes
            return await apply_stat_changes(self.user_id, self.conversation_id, player_name, changes, reason or "orchestrator")
        except Exception as e:
            logger.error(f"apply_player_stat_changes failed: {e}")
            return {"success": False, "error": str(e)}

    async def _route_stats_world_activity(
        self,
        activity_name: str,
        intensity: float = 1.0,
        hours: int = 0,
        npc_id: Optional[int] = None,
        location: Optional[str] = None,
        forced: bool = False,
        world_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use stats_logic.process_world_activity to apply player stat effects based on activity.
        Returns a compact payload and triggers belief/narrative hooks for thresholds/combos.
        """
        if not activity_name:
            return {"skipped": True}
        try:
            from logic.stats_logic import process_world_activity
            res = await process_world_activity(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                activity_name=activity_name,
                player_name="Chase",
                world_context=world_context or {},
                intensity=float(intensity),
                hours=int(hours or 0),
                npc_id=int(npc_id) if npc_id is not None else None,
                location=location,
                is_forced=bool(forced)
            )
    
            # Belief hooks for threshold/combo triggers (if present)
            try:
                trig = []
                for t in (res.get("thresholds_triggered") or []):
                    name = f"{t.get('stat')}:{t.get('threshold')}"
                    trig.append(name)
                combos = res.get("combinations_active") or []
                if trig or combos:
                    from types import SimpleNamespace
                    ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                    text_bits = []
                    if trig:
                        text_bits.append(f"Thresholds: {', '.join(trig)}")
                    if combos:
                        text_bits.append(f"Combinations: {', '.join(combos)}")
                    if text_bits:
                        await self._belief_system.process_event_for_beliefs(
                            ctx=ctx_obj,
                            event_text="; ".join(text_bits)[:250],
                            event_type="player_stat_shift",
                            npc_ids=[npc_id] if npc_id else [],
                            factuality=1.0
                        )
            except Exception as be:
                logger.debug(f"[Beliefs] stat thresholds hook failed: {be}")
    
            return res or {"success": False}
        except Exception as e:
            logger.debug(f"[Stats] world activity routing failed: {e}")
            return {"success": False, "error": str(e)}
    
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
            entity1_type, entity1_id, entity2_type, entity2_id
        )
        return {
            'dimensions': {
                'trust': float(state.dimensions.trust),
                'affection': float(state.dimensions.affection),
                'respect': float(state.dimensions.respect),
                'familiarity': float(state.dimensions.fascination),  # or map as needed
                'volatility': float(state.dimensions.volatility),
                'unresolved_conflict': float(state.dimensions.unresolved_conflict),
                'intimacy': float(state.dimensions.intimacy),
                'dependence': float(state.dimensions.dependence),
                'frequency': float(state.dimensions.frequency),
                'influence': float(state.dimensions.influence),
            },
            'patterns': list(state.history.active_patterns),
            'archetypes': list(state.active_archetypes),
            'momentum': {
                'magnitude': float(state.momentum.get_magnitude())
            },
            'contexts': asdict(state.contexts)
        }
    async def poll_relationship_events(self, max_events: int = 25) -> List[Dict[str, Any]]:
        try:
            from logic.dynamic_relationships import event_generator
            return await event_generator.drain_events(max_events=max_events)
        except Exception as e:
            logger.debug(f"[Relationships] poll failed: {e}")
            return []

    async def apply_relationship_crossroads_choice(self, crossroads: Dict[str, Any], choice_index: int) -> Dict[str, Any]:
        """
        Accept a UI crossroads payload (dict), convert to CrossroadsEvent, and apply.
        """
        try:
            from logic.fully_integrated_npc_system import CrossroadsEvent
            integ = await self._get_integrated_npc_system()
    
            # Convert dict -> CrossroadsEvent (relationship_state can be None)
            ev = CrossroadsEvent(
                entity1_type=crossroads.get("entity1_type", "npc"),
                entity1_id=int(crossroads.get("entity1_id")),
                entity2_type=crossroads.get("entity2_type", "player"),
                entity2_id=int(crossroads.get("entity2_id")),
                relationship_state=None,  # Not required by the applier
                event_type=crossroads.get("event_type", "relationship_crossroads"),
                description=crossroads.get("description", ""),
                options=crossroads.get("options", []),
                expires_in=int(crossroads.get("expires_in", 3))
            )
    
            return await integ.apply_crossroads_choice(ev, choice_index)
        except Exception as e:
            return {"error": str(e)}

    async def run_scheduled_activities_if_enabled(self) -> Optional[Dict[str, Any]]:
        if not self._integrated_routing.get("use_for_scheduled_activities", True):
            return None
        try:
            integ = await self._get_integrated_npc_system()
            res = await integ.process_npc_scheduled_activities()
            # Invalidate changed NPCs for scene deltas
            for nid in {r.get("npc_id") for r in res.get("npc_responses", []) if isinstance(r, dict)}:
                if nid:
                    self._notify_npc_changed(int(nid))
            return res
        except Exception as e:
            logger.debug(f"[Integrated] scheduled activities failed: {e}")
            return {"error": str(e)}

    async def _route_time_advance_via_integrated(self, player_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use IntegratedNPCSystem to detect activity type, advance time if appropriate,
        process scheduled activities, calendar events, and drain relationship events.
        """
        try:
            integ = await self._get_integrated_npc_system()
            pa = await integ.process_player_activity(player_input, context or {})
            out = {"integrated_player_activity": pa}
    
            if pa.get("activity_type"):
                should_advance = pa.get("time_advanced") or pa.get("would_advance")
                if should_advance:
                    adv = await integ.advance_time_with_activity(pa["activity_type"])
                    out["time_advance_result"] = adv
    
                    if adv.get("time_advanced"):
                        # Scheduled activities
                        try:
                            sched = await integ.process_npc_scheduled_activities()
                            out["scheduled_activities"] = sched
                        except Exception as e:
                            out["scheduled_activities_error"] = str(e)
    
                        # Calendar processing
                        try:
                            cal = await self.process_calendar_events_for_all_npcs()
                            out["calendar_results"] = cal
                        except Exception as e:
                            out["calendar_error"] = str(e)
    
                        # Relationship events
                        try:
                            from logic.dynamic_relationships import event_generator
                            rel_events = await event_generator.drain_events(max_events=25)
                            out["relationship_events"] = rel_events
                        except Exception as e:
                            out["relationship_events_error"] = str(e)

                        # Stats: apply player effects for this activity (if available)
                        try:
                            activity = pa.get("activity_type")
                            npc_hint = (pa.get("npc_id") or pa.get("target_npc_id"))
                            hours_adv = 0
                            try:
                                hours_adv = int((adv or {}).get("hours_advanced") or 0)
                            except Exception:
                                hours_adv = 0
                        
                            if activity:
                                stats_pack = await self._route_stats_world_activity(
                                    activity_name=str(activity),
                                    intensity=float(pa.get("intensity", 1.0)),
                                    hours=hours_adv,
                                    npc_id=int(npc_hint) if npc_hint is not None else None,
                                    location=(pa.get("location") or None),
                                    forced=bool(pa.get("forced", False)),
                                    world_context={"source": "integrated_time_route"}
                                )
                                if isinstance(stats_pack, dict):
                                    out.setdefault("systems", {}).setdefault("stats", {})["activity"] = stats_pack
                        except Exception as se:
                            out.setdefault("systems", {}).setdefault("stats", {})["error"] = str(se)

                        try:
                            add_status = await self.get_player_addiction_status()
                            out.setdefault("systems", {}).setdefault("addictions", {})["status"] = add_status
                            if add_status.get("has_addictions"):
                                add_fx = await self.apply_addiction_effects_from_status(add_status)
                                out["systems"]["addictions"]["effects"] = add_fx
                        except Exception as ae:
                            out.setdefault("systems", {}).setdefault("addictions", {})["error"] = str(ae)

                        # Narrative checks after time advance (optional but recommended)
                        try:
                            narrative_pack = {}
                            personal = await self.check_and_log_personal_revelations()
                            moment = await self.check_and_log_narrative_moments()
                            if personal:
                                narrative_pack["personal_revelation"] = personal
                            if moment:
                                narrative_pack["narrative_moment"] = moment
                            if narrative_pack:
                                out["narrative"] = narrative_pack
                        except Exception as ne:
                            out["narrative_error"] = str(ne)
    
                        # Invalidate active NPCs for deltas
                        for nid in list(self._active_npcs):
                            self._notify_npc_changed(nid)
    
            return out
        except Exception as e:
            return {"error": f"time routing via integrated system failed: {e}"}
        
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
            entity1_id=self.user_id,  # FIX: use real player id, not hard-coded 1
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
                    "player", self.user_id, "npc", npc_id
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

    # --- LORE INTEGRATION BRIDGE ----------------------------------------------
    
    async def get_npc_lore_context(self, npc_id: int, context_type: str = "profile") -> Dict[str, Any]:
        """Thin wrapper to retrieve lore context with centralized error handling."""
        try:
            return await self._lore_manager.get_lore_context(npc_id, context_type)
        except Exception as e:
            logger.warning(f"[Lore] get_lore_context failed for NPC {npc_id}: {e}")
            return {}
    
    def invalidate_lore_cache_for(self, lore_id: Union[int, str]):
        """Invalidate lore context cache entries tied to a given lore_change id."""
        try:
            self._lore_manager.context_cache.invalidate(f"lore_change_{lore_id}")
        except Exception as e:
            logger.debug(f"[Lore] cache invalidate failed for id={lore_id}: {e}")
    
    async def handle_lore_change(
        self,
        lore_change: Dict[str, Any],
        source_npc_id: int,
        affected_npcs: List[int]
    ) -> Dict[str, Any]:
        """
        Orchestrated lore change handling:
        - Run impact + propagation via LoreContextManager
        - Canonize significant changes (optional)
        - Feed event into belief system + throttled consolidation
        - Invalidate NPC and scene caches touched by these NPCs
        """
        # 1) Run through the manager
        result = await self._lore_manager.handle_lore_change(lore_change, source_npc_id, affected_npcs)
    
        # 2) Canon (optional but usually desirable for major changes)
        try:
            if self.enable_canon and self.auto_canonize:
                significance = 7 if lore_change.get("is_major_revelation") else 4
                async with get_db_connection_context() as conn:
                    await log_canonical_event(
                        conn,
                        self.user_id,
                        self.conversation_id,
                        f"Lore change: {lore_change.get('name','(unnamed)')} ({lore_change.get('id')})",
                        tags=["lore", "change", lore_change.get("type", "general")],
                        significance=significance
                    )
        except Exception as e:
            logger.debug(f"[Lore] canon logging failed: {e}")
    
        # 3) Belief hooks + consolidation
        try:
            for nid in affected_npcs:
                ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=nid)
                await self._belief_system.process_event_for_beliefs(
                    ctx=ctx_obj,
                    event_text=f"Lore changed: {lore_change.get('name','(unnamed)')}",
                    event_type="lore_change",
                    npc_ids=[nid],
                    factuality=1.0
                )
            await self._maybe_consolidate_beliefs(affected_npcs, topic_filter="lore_change", reason="post_lore_change")
        except Exception as e:
            logger.warning(f"[Lore] belief routing after lore change failed: {e}")
    
        # 4) Invalidate NPC + scene caches
        for nid in affected_npcs:
            self._notify_npc_changed(nid)
    
        # 5) Lore cache invalidation for this change id (if present)
        if "id" in lore_change:
            self.invalidate_lore_cache_for(lore_change["id"])
    
        return result

    # ==================== PRESET NPC INTEGRATION ====================

    async def create_or_update_preset_npc(
        self,
        npc_data: Dict[str, Any],
        story_context: Dict[str, Any],
        ctx_obj: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Orchestrator wrapper for Preset NPC creation/update.
        - Calls PresetNPCHandler.create_detailed_npc
        - Refreshes snapshot/state/indexes
        - Invalidates caches and marks deltas
        - Routes a belief event + optional canon log
        """
        # Build a ctx with .context expected by PresetNPCHandler (if not provided)
        ctx = ctx_obj or SimpleNamespace(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # 1) Create or update via preset handler
        try:
            npc_id = await PresetNPCHandler.create_detailed_npc(ctx, npc_data, story_context)
        except Exception as e:
            logger.exception(f"[PresetNPC] create_detailed_npc failed: {e}")
            return {"error": str(e)}

        # 2) Invalidate snapshot cache and mark change
        try:
            self._snapshot_cache.pop(npc_id, None)
        except Exception:
            pass
        self._notify_npc_changed(npc_id)

        # 3) Refresh snapshot + index this NPC (active set, location index, scene-state cache)
        try:
            snap = await self.get_npc_snapshot(npc_id, force_refresh=True, light=True)

            # Active tracking (reuse existing helpers/sets)
            self._update_active_trackers(npc_id, snap.status)

            # Location index maintenance
            if snap.location:
                self._location_index[snap.location].add(npc_id)

            # Scene-state cache
            self._scene_state_cache[npc_id] = {
                "status": snap.status,
                "location": snap.location,
                "trust": snap.trust,
                "respect": snap.respect,
                "closeness": snap.closeness,
                "emotional_state": snap.emotional_state or {},
            }

        except Exception as e:
            logger.debug(f"[PresetNPC] post-create snapshot/index refresh failed for NPC {npc_id}: {e}")

        # 4) Optional: belief + canon hooks
        try:
            # Belief event about initialization
            try:
                from types import SimpleNamespace
                bctx = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                await self._belief_system.process_event_for_beliefs(
                    ctx=bctx,
                    event_text=f"Preset NPC initialized: {npc_data.get('name', '(unnamed)')}",
                    event_type="preset_npc_initialized",
                    npc_ids=[npc_id],
                    factuality=1.0
                )
            except Exception as be:
                logger.debug(f"[PresetNPC] belief hook failed for NPC {npc_id}: {be}")

            # Throttled consolidation
            try:
                await self._maybe_consolidate_beliefs([npc_id], topic_filter="preset_init", reason="preset_init")
            except Exception as ce:
                logger.debug(f"[PresetNPC] consolidation failed for NPC {npc_id}: {ce}")

            # Canon event (respect config flags)
            if self.enable_canon and self.auto_canonize:
                async with get_db_connection_context() as conn:
                    text = f"Preset NPC initialized: {npc_data.get('name', '(unnamed)')}"
                    await log_canonical_event(
                        conn,
                        self.user_id,
                        self.conversation_id,
                        text,
                        tags=["npc", "preset", "init"],
                        significance=6
                    )
        except Exception as e:
            logger.debug(f"[PresetNPC] belief/canon wrapper failed for NPC {npc_id}: {e}")

        try:
            await self._maybe_evaluate_behavior([npc_id], reason="post_preset_init", use_user_model=False)
        except Exception as e:
            logger.debug(f"[PresetNPC] post-init behavior eval failed for NPC {npc_id}: {e}")

        # 5) Return a simple payload for callers
        # Try to include light context for convenience
        try:
            snap = await self.get_npc_snapshot(npc_id, light=True)
            return {
                "npc_id": npc_id,
                "name": snap.name,
                "location": snap.location,
                "status": snap.status,
                "traits": snap.personality_traits[:5],
            }
        except Exception:
            return {"npc_id": npc_id, "name": npc_data.get("name")}

    async def create_or_update_preset_npcs(
        self,
        npcs: List[Dict[str, Any]],
        story_context: Dict[str, Any],
        ctx_obj: Optional[Any] = None,
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        Bulk variant: integrate multiple preset NPCs.
        Returns a summary + per-NPC results.
        """
        results = []
        errors = 0

        for data in npcs:
            try:
                res = await self.create_or_update_preset_npc(
                    npc_data=data,
                    story_context=story_context,
                    ctx_obj=ctx_obj
                )
                if "error" in res:
                    errors += 1
                    if fail_fast:
                        return {"error": res["error"], "results": results}
                results.append(res)
            except Exception as e:
                errors += 1
                results.append({"error": str(e), "npc_name": data.get("name")})
                if fail_fast:
                    return {"error": str(e), "results": results}

        # After a wave of preset NPC updates, refresh active sets (safety)
        try:
            await self._load_active_npcs()
        except Exception as e:
            logger.debug(f"[PresetNPC] active set refresh failed: {e}")

        return {"results": results, "errors": errors}
    
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
            memory_text = ' '.join([m.get('text', '') for m in snapshot.recent_memories])
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
        dynamics = {
            'tension_level': 0,
            'dominant_mood': 'neutral',
            'active_conflicts': [],
            'alliances': []
        }
        if not npc_ids:
            return dynamics

        # Start with conflict synthesizer context (world_tension, conflicts)
        tensions = []
        try:
            synth = await self._get_conflict_synth()
            if synth:
                scope = self._ConflictScope(npc_ids=list(npc_ids))
                cb = await synth.get_scene_bundle(scope)
                # world_tension is a float [0..1] in bundle; map to [0..100] contribution
                wt = float(cb.get("world_tension") or 0.0)
                if wt > 0:
                    tensions.append(int(round(wt * 100)))
                # Include a few conflicts for UI
                for c in (cb.get("active") or [])[:3]:
                    dynamics['active_conflicts'].append({
                        'id': c.get('id'),
                        'type': c.get('type'),
                        'intensity': c.get('intensity')
                    })
        except Exception as e:
            logger.debug(f"[Conflict] scene dynamics fetch failed: {e}")

        # Existing scheming/betrayal signals
        for npc_id in npc_ids:
            snapshot = snapshot_map.get(npc_id) if snapshot_map else await self.get_npc_snapshot(npc_id, light=True)
            if snapshot.scheming_level > 0:
                tensions.append(snapshot.scheming_level)
            if snapshot.betrayal_planning:
                tensions.append(50)

        if tensions:
            dynamics['tension_level'] = sum(tensions) // max(1, len(tensions))

        # Mood (unchanged)
        moods = []
        for npc_id in npc_ids:
            snapshot = snapshot_map.get(npc_id) if snapshot_map else await self.get_npc_snapshot(npc_id, light=True)
            if snapshot.emotional_state.get('mood'):
                moods.append(snapshot.emotional_state['mood'])
        if moods:
            from collections import Counter
            dynamics['dominant_mood'] = Counter(moods).most_common(1)[0][0]

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
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND status IN ('active', 'idle', 'observing')
            """, self.user_id, self.conversation_id)
    
            self._active_npcs.clear()
            self._location_index.clear()
    
            for row in rows:
                self._active_npcs.add(row['npc_id'])
                if row['current_location']:
                    self._location_index[row['current_location']].add(row['npc_id'])
    
        # NEW: drop cached memory-managers for NPCs no longer active
        self._prune_mem_mgr_cache()
        
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
        """
        Handle player action and generate NPC responses. Preferred flow:
        - Group fast-path -> NPCAgentCoordinator
        - Conversational single-NPC -> orchestrated pipeline (NPCHandler proposal + orchestrator side effects)
        - Default -> NPCSystemBridge (fallback to NPCAgentSystem)
    
        Also:
        - Notifies ConflictSynthesizer of action context (non-blocking) on the default path
        - Preserves belief routing + behavior throttling for conversational actions (legacy path)
        - Optionally routes time advance + scheduled activities via IntegratedNPCSystem
        """
        result: Dict[str, Any] = {}
        path = "default"
    
        # Try group fast-path first
        try:
            explicit_group = action.get("target") == "group" or action.get("scope") == "group"
            mentioned_ids_raw = action.get("npc_ids") or []
            mentioned_ids: List[int] = []
            if isinstance(mentioned_ids_raw, list):
                for x in mentioned_ids_raw:
                    try:
                        mentioned_ids.append(int(x))
                    except Exception:
                        continue
                mentioned_ids = list({*mentioned_ids})
    
            if explicit_group or (isinstance(mentioned_ids, list) and len(mentioned_ids) > 1):
                group_res = await self.handle_group_player_action(action, context, mentioned_ids)
                result = group_res or {"npc_responses": []}
                path = "group"
    
                # Optional: Integrated time routing
                if self._integrated_routing.get("use_for_time_advance", True):
                    try:
                        time_pack = await self._route_time_advance_via_integrated(action.get("description", "") or "", context or {})
                        if isinstance(time_pack, dict):
                            result.setdefault("systems", {})["time"] = time_pack
                            rel_evs = time_pack.get("relationship_events") or []
                            if rel_evs:
                                result.setdefault("events", []).extend([{"type": "relationship", "data": e} for e in rel_evs])
                    except Exception as e:
                        logger.debug(f"[IntegratedTime] routing failed (group): {e}")
    
                return result
        except Exception:
            # fall through to single-NPC/conversational/default flows
            pass
    
        # Orchestrated conversational pipeline for a single NPC (preferred path)
        try:
            action_type = (action.get("type") or "").lower()
            is_conversation = action_type in {
                "talk", "say", "speak", "ask", "persuade", "deceive",
                "intimidate", "flirt", "confess", "negotiate"
            }
            if is_conversation:
                # Resolve a single target NPC
                target_ids: List[int] = []
                npc_ids_field = action.get("npc_ids") or []
                if isinstance(npc_ids_field, list) and npc_ids_field:
                    try:
                        target_ids = [int(npc_ids_field[0])]
                    except Exception:
                        target_ids = []
    
                if not target_ids:
                    try:
                        target_ids = await self._agent_system.determine_affected_npcs(action, context)
                    except Exception:
                        target_ids = []
    
                npc_id = (target_ids or list(self._active_npcs)[:1])[0] if (target_ids or self._active_npcs) else None
    
                if npc_id is not None:
                    out = await self.synthesize_player_interaction(
                        npc_id=npc_id,
                        interaction_type=(action.get("interaction_type") or action.get("tone") or "friendly"),
                        player_input=action.get("description", ""),
                        context=context or {}
                    )
                    result = out or {}
                    path = "conversation"
    
                    # Optional: Integrated time routing
                    if self._integrated_routing.get("use_for_time_advance", True):
                        try:
                            time_pack = await self._route_time_advance_via_integrated(action.get("description", "") or "", context or {})
                            if isinstance(time_pack, dict):
                                result.setdefault("systems", {})["time"] = time_pack
                                rel_evs = time_pack.get("relationship_events") or []
                                if rel_evs:
                                    result.setdefault("events", []).extend([{"type": "relationship", "data": e} for e in rel_evs])
                        except Exception as e:
                            logger.debug(f"[IntegratedTime] routing failed (conversation): {e}")
    
                    return result
        except Exception as e:
            logger.debug(f"[Orchestrator] conversational pipeline fallback: {e}")
    
        # Default path: prefer NPCSystemBridge, fall back to NPCAgentSystem
        try:
            bridge = await self._get_npc_bridge()
            if bridge:
                result = await bridge.handle_player_action(action, context)
            else:
                result = await self._agent_system.process_player_action(action, context)
        except Exception as e:
            logger.debug(f"[Bridge] handle_player_action failed, using legacy path: {e}")
            result = await self._agent_system.process_player_action(action, context)
    
        # Optional: Integrated time routing (default path)
        if self._integrated_routing.get("use_for_time_advance", True):
            try:
                time_pack = await self._route_time_advance_via_integrated(action.get("description", "") or "", context or {})
                if isinstance(time_pack, dict):
                    result.setdefault("systems", {})["time"] = time_pack
                    rel_evs = time_pack.get("relationship_events") or []
                    if rel_evs:
                        result.setdefault("events", []).extend([{"type": "relationship", "data": e} for e in rel_evs])
            except Exception as e:
                logger.debug(f"[IntegratedTime] routing failed (default): {e}")
    
        # Notify ConflictSynthesizer about this action as a scene (non-blocking),
        # so conflict subsystems can detect/update conflicts based on current context.
        try:
            synth = await self._get_conflict_synth()
            if synth:
                # Derive affected NPCs when possible
                affected_ids: List[int] = []
                if isinstance(result, dict) and result.get("npc_responses"):
                    for r in result["npc_responses"]:
                        if isinstance(r, dict):
                            rid = r.get("npc_id")
                            if rid is not None:
                                try:
                                    affected_ids.append(int(rid))
                                except Exception:
                                    pass
                if not affected_ids:
                    try:
                        affected_ids = await self._agent_system.determine_affected_npcs(action, context)
                    except Exception:
                        affected_ids = []
    
                scene_ctx = {
                    "scene_id": int(time.time() * 1000) % 2_147_483_647,
                    "scene_type": action.get("type") or "action",
                    "characters_present": affected_ids[:10],
                    "location_id": None,  # Unknown numeric id in this context
                    "dialogue": [{"speaker": "player", "text": action.get("description", "")}] if action.get("description") else [],
                    "timestamp": datetime.utcnow().isoformat()
                }
                asyncio.create_task(synth.process_scene(scene_ctx))
        except Exception as e:
            logger.debug(f"[Conflict] bridge/default scene notify failed: {e}")
    
        # Belief integration for conversational actions (legacy path)
        affected_ids: List[int] = []
        try:
            action_type = (action.get("type") or "").lower()
            is_conversation = action_type in {
                "talk", "say", "speak", "ask", "persuade", "deceive",
                "intimidate", "flirt", "confess", "negotiate"
            }
            text = action.get("description", "") or ""
    
            if is_conversation and text and path == "default":
                # Resolve affected NPCs
                try:
                    affected_ids = await self._agent_system.determine_affected_npcs(action, context)
                except Exception as e:
                    logger.warning(f"[Beliefs] could not resolve affected NPCs; fallback to active set: {e}")
                    affected_ids = list(self._active_npcs)[:3]
    
                if affected_ids:
                    from types import SimpleNamespace
                    for npc_id in affected_ids:
                        ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                        try:
                            await self._belief_system.process_conversation_for_beliefs(
                                ctx=ctx_obj,
                                conversation_text=text,
                                speaker_id="player",
                                listener_id=npc_id,
                                topic=context.get("topic", "general"),
                                credibility=float(context.get("credibility", 0.7))
                            )
                        except Exception as e:
                            logger.warning(f"[Beliefs] conversation belief hook failed for NPC {npc_id}: {e}")
    
                    # Throttled consolidation right after conversation bursts
                    await self._maybe_consolidate_beliefs(
                        affected_ids,
                        topic_filter=context.get("topic"),
                        reason="post_conversation"
                    )
    
        except Exception as e:
            logger.exception(f"[Beliefs] post-action belief routing failed: {e}")
    
        # Behavior evolution right after conversations (throttled, legacy path)
        try:
            if affected_ids and path == "default":
                await self._maybe_evaluate_behavior(affected_ids, reason="post_conversation", use_user_model=True)
        except Exception as e:
            logger.warning(f"[Behavior] post_conversation evolution failed: {e}")

        # Optional: direct stats activity routing for explicit actions (non-time-advance)
        try:
            act_name = (action.get("activity") or action.get("type") or "").lower()
            # Only run if not already covered by integrated time routing
            if act_name in {"eat", "meal", "sleep", "rest", "train", "exercise", "service_task", "wait"}:
                stats_out = await self._route_stats_world_activity(
                    activity_name=act_name,
                    intensity=float(action.get("intensity", 1.0)),
                    hours=int(action.get("hours", 0) or 0),
                    npc_id=(action.get("npc_id") if isinstance(action.get("npc_id"), int) else None),
                    location=action.get("location"),
                    forced=bool(action.get("is_forced", False)),
                    world_context={"source": "handle_player_action"}
                )
                if isinstance(stats_out, dict):
                    result.setdefault("systems", {}).setdefault("stats", {})["activity"] = stats_out
        except Exception as se:
            logger.debug(f"[Stats] direct activity routing failed: {se}")
        
        return result

    async def detect_npc_deception(self, npc_id: int, deception_type: str = "hidden_motive", player_name: str = "Chase") -> Dict[str, Any]:
        """
        Bridge to stats_logic.detect_deception for empathy-based insight checks.
        """
        try:
            from logic.stats_logic import detect_deception
            res = await detect_deception(self.user_id, self.conversation_id, player_name, int(npc_id), deception_type)
            # Optional belief hook on success
            if res.get("success") and res.get("insight"):
                try:
                    from types import SimpleNamespace
                    ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                    await self._belief_system.process_event_for_beliefs(
                        ctx=ctx_obj,
                        event_text=f"Insight: {res['insight']}"[:250],
                        event_type="insight_check",
                        npc_ids=[npc_id],
                        factuality=1.0
                    )
                except Exception:
                    pass
            return res
        except Exception as e:
            logger.error(f"detect_npc_deception failed: {e}")
            return {"success": False, "error": str(e)}

    async def progress_npc_narrative(
        self,
        npc_id: int,
        corruption_change: int = 0,
        dependency_change: int = 0,
        realization_change: int = 0,
        force_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delegate narrative progression to logic.npc_narrative_progression,
        then handle cache invalidation, optional belief routing, and event surfacing.
        """
        try:
            from logic.npc_narrative_progression import progress_npc_narrative_stage
            res = await progress_npc_narrative_stage(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                npc_id=int(npc_id),
                corruption_change=int(corruption_change),
                dependency_change=int(dependency_change),
                realization_change=int(realization_change),
                force_stage=force_stage
            )
    
            # Invalidate snapshot and mark deltas
            self.invalidate_npc_state(npc_id, reason="narrative_progression")
    
            # Optional: surface relationship events generated by the interaction inside the module
            try:
                from logic.dynamic_relationships import event_generator
                rel_events = await event_generator.drain_events(max_events=25)
                if isinstance(res, dict):
                    if rel_events:
                        res["relationship_events"] = rel_events
            except Exception:
                pass
    
            # Optional: belief hook about narrative change (if changed)
            try:
                if isinstance(res, dict) and res.get("stage_changed"):
                    from types import SimpleNamespace
                    ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                    await self._belief_system.process_event_for_beliefs(
                        ctx=ctx_obj,
                        event_text=f"Narrative progressed to {res.get('new_stage')}",
                        event_type="narrative_progression",
                        npc_ids=[npc_id],
                        factuality=1.0
                    )
                    await self._maybe_consolidate_beliefs([npc_id], topic_filter="narrative_progression", reason="narrative_progression")
            except Exception:
                pass
    
            return res
        except Exception as e:
            logger.error(f"progress_npc_narrative failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_narrative_relationship_overview(self) -> Dict[str, Any]:
        """
        Return a stage-centric overview across all NPC relationships for this user/conversation.
        """
        try:
            from logic.narrative_events import get_relationship_overview
            return await get_relationship_overview(self.user_id, self.conversation_id)
        except Exception as e:
            logger.error(f"get_narrative_relationship_overview failed: {e}")
            return {"error": str(e), "total_relationships": 0, "by_stage": {}, "relationships": []}

    async def generate_player_inner_monologue(self, topic: Optional[str] = None, log_to_journal: bool = False) -> Dict[str, Any]:
        """
        Generate a brief inner monologue for the player. Optionally log to PlayerJournal.
        """
        try:
            from logic.narrative_events import generate_inner_monologue
            text = await generate_inner_monologue(self.user_id, self.conversation_id, topic)
            payload = {"text": text}
    
            if log_to_journal and text:
                try:
                    async with get_db_connection_context() as conn:
                        journal_id = await conn.fetchval("""
                            INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                            VALUES ($1, $2, 'inner_monologue', $3, CURRENT_TIMESTAMP)
                            RETURNING id
                        """, self.user_id, self.conversation_id, text)
                    payload["journal_id"] = journal_id
                except Exception as je:
                    logger.debug(f"[Monologue] journal log failed: {je}")
    
            return payload
        except Exception as e:
            logger.error(f"generate_player_inner_monologue failed: {e}")
            return {"text": "What's happening to me?", "error": str(e)}

    async def check_and_log_personal_revelations(self) -> Optional[Dict[str, Any]]:
        """
        Check for a personal revelation across relationships, log it, hook beliefs, and return the result.
        """
        try:
            from logic.narrative_events import check_for_personal_revelations
            res = await check_for_personal_revelations(self.user_id, self.conversation_id)
            if not res:
                return None
    
            # Optional: belief hook (personal revelation about the player's state)
            try:
                from types import SimpleNamespace
                ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=res.get("npc_id"))
                await self._belief_system.process_event_for_beliefs(
                    ctx=ctx_obj,
                    event_text=res.get("inner_monologue", "")[:200] or "Personal revelation occurred",
                    event_type="personal_revelation",
                    npc_ids=[res.get("npc_id")] if res.get("npc_id") else [],
                    factuality=1.0
                )
                if res.get("npc_id"):
                    # Mark NPC changed for scene deltas
                    self._notify_npc_changed(int(res["npc_id"]))
            except Exception as be:
                logger.debug(f"[Beliefs] personal revelation hook failed: {be}")
    
            return res
        except Exception as e:
            logger.error(f"check_and_log_personal_revelations failed: {e}")
            return None
    
    async def add_dream_sequence_entry(self) -> Optional[Dict[str, Any]]:
        """
        Generate and journal a dream sequence influenced by relationship stages.
        """
        try:
            from logic.narrative_events import add_dream_sequence
            res = await add_dream_sequence(self.user_id, self.conversation_id)
            return res
        except Exception as e:
            logger.error(f"add_dream_sequence_entry failed: {e}")
            return None

    async def initialize_player_stats_if_needed(self) -> None:
        """
        Ensure PlayerStats exist; safe to call at session start.
        """
        try:
            from logic.narrative_events import initialize_player_stats
            await initialize_player_stats(self.user_id, self.conversation_id)
        except Exception as e:
            logger.debug(f"initialize_player_stats_if_needed failed: {e}")

    async def analyze_narrative_tone(self, narrative_text: str) -> Dict[str, Any]:
        """
        Run a tone and thematic analysis for a given narrative text.
        """
        try:
            from logic.narrative_events import analyze_narrative_tone as _analyze
            return await _analyze(narrative_text)
        except Exception as e:
            logger.error(f"analyze_narrative_tone failed: {e}")
            return {"dominant_tone": "error", "power_dynamics": "error", "implied_stages": {}, "manipulation_techniques": [], "intensity_level": 0, "error": str(e)}

    async def add_moment_of_clarity_entry(self, realization_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Add a moment of clarity to the journal based on overall relationship stage distribution.
        """
        try:
            from logic.narrative_events import add_moment_of_clarity
            res = await add_moment_of_clarity(self.user_id, self.conversation_id, realization_text)
            return res
        except Exception as e:
            logger.error(f"add_moment_of_clarity_entry failed: {e}")
            return None

    async def check_and_log_narrative_moments(self) -> Optional[Dict[str, Any]]:
        """
        Check for a narrative moment driven by the stage mix and write it to the journal. Adds belief hooks.
        """
        try:
            from logic.narrative_events import check_for_narrative_moments
            res = await check_for_narrative_moments(self.user_id, self.conversation_id)
            if not res:
                return None
    
            # Optional: belief hook (world/narrative-level)
            try:
                from types import SimpleNamespace
                ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=None)
                await self._belief_system.process_event_for_beliefs(
                    ctx=ctx_obj,
                    event_text=res.get("scene_text", "")[:200] or "Narrative moment occurred",
                    event_type="narrative_moment",
                    npc_ids=[],
                    factuality=1.0
                )
            except Exception as be:
                logger.debug(f"[Beliefs] narrative moment hook failed: {be}")
    
            return res
        except Exception as e:
            logger.error(f"check_and_log_narrative_moments failed: {e}")
            return None

    async def get_npc_narrative_stage(self, npc_id: int) -> Dict[str, Any]:
        """
        Thin wrapper to fetch the current narrative stage via the narrative module.
        """
        try:
            from logic.npc_narrative_progression import get_npc_narrative_stage as _get_stage
            stage = await _get_stage(self.user_id, self.conversation_id, int(npc_id))
            return {
                "npc_id": int(npc_id),
                "stage": {
                    "name": stage.name,
                    "description": stage.description,
                    "requirements": {
                        "required_corruption": stage.required_corruption,
                        "required_dependency": stage.required_dependency,
                        "required_realization": stage.required_realization
                    }
                }
            }
        except Exception as e:
            logger.error(f"get_npc_narrative_stage failed: {e}")
            return {"npc_id": int(npc_id), "error": str(e)}
    
    async def check_for_npc_revelation(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """
        Delegate revelation checks to logic.npc_narrative_progression and
        integrate with beliefs/caches.
        """
        try:
            from logic.npc_narrative_progression import check_for_npc_revelation as _check_revelation
            res = await _check_revelation(self.user_id, self.conversation_id, int(npc_id))
            if not res:
                return None
    
            # Invalidate caches since narrative/relationship context likely changed
            self.invalidate_npc_state(npc_id, reason="npc_revelation")
    
            # Optional: belief event for the revelation
            try:
                from types import SimpleNamespace
                ctx_obj = SimpleNamespace(user_id=self.user_id, conversation_id=self.conversation_id, npc_id=npc_id)
                await self._belief_system.process_event_for_beliefs(
                    ctx=ctx_obj,
                    event_text=f"Revelation: {res.get('revelation_text','(text)')[:200]}",
                    event_type="npc_revelation",
                    npc_ids=[npc_id],
                    factuality=1.0
                )
                await self._maybe_consolidate_beliefs([npc_id], topic_filter="npc_revelation", reason="npc_revelation")
            except Exception:
                pass
    
            return res
        except Exception as e:
            logger.error(f"check_for_npc_revelation failed: {e}")
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

    async def _get_conflict_synth(self):
        """Lazy-load the ConflictSynthesizer."""
        if not self.enable_conflicts:
            return None
        if self._conflict_synth is None:
            try:
                from logic.conflict_system.conflict_synthesizer import get_synthesizer
                self._conflict_synth = await get_synthesizer(self.user_id, self.conversation_id)
            except Exception as e:
                logger.debug(f"[Conflict] synthesizer unavailable: {e}")
                self._conflict_synth = False
        return self._conflict_synth or None

    class _ConflictScope:
        """Minimal scope object compatible with ConflictSynthesizer.get_scene_bundle"""
        __slots__ = ("location_id", "npc_ids", "topics", "lore_tags", "conflict_ids", "time_window_hours", "link_hints")

        def __init__(self, location_id=None, npc_ids=None, topics=None, lore_tags=None, conflict_ids=None, time_window_hours=24, link_hints=None):
            self.location_id = location_id
            self.npc_ids = list(npc_ids or [])
            self.topics = list(topics or [])
            self.lore_tags = list(lore_tags or [])
            self.conflict_ids = list(conflict_ids or [])
            self.time_window_hours = time_window_hours
            self.link_hints = link_hints or {}

        def to_cache_key(self) -> str:
            import json, hashlib
            key_str = json.dumps({
                "loc": self.location_id,
                "npcs": sorted(self.npc_ids),
                "topics": sorted(self.topics),
                "lore": sorted(self.lore_tags),
                "window": self.time_window_hours,
            }, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
