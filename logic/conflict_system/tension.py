# logic/conflict_system/tension.py
"""
Dynamic Tension System with LLM-generated content.
Refactored for performance to work as a non-blocking subsystem under the Conflict Synthesizer.

REFACTORED FOR PERFORMANCE:
- Expensive LLM calls for tension manifestation are offloaded to a background Celery task.
- Scene-specific tension "bundles" (manifestations, cues) are cached in Redis for fast retrieval.
- The main event loop is non-blocking; it returns cached data or a minimal fallback immediately.
- A Redis lock prevents redundant background generation tasks.
- Core tension updates are fast, numerical operations; narrative emerges from the cached bundles.
"""

import logging
import json
import asyncio
import os
import hashlib
import dataclasses
import redis.asyncio as redis # POLISH: Use the async version of the redis client
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import weakref

from agents import Agent, Runner, function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from logic.conflict_system.dynamic_conflict_template import extract_runner_response

# Import your Celery app instance here
# from tasks import celery_app

logger = logging.getLogger(__name__)

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client: Optional[redis.Redis] = None

async def get_redis_client() -> redis.Redis:
    """Lazy-loads and returns a singleton async Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

CACHE_KEY_TEMPLATE = "tension_bundle:{user_id}:{conv_id}:{scene_hash}"
CACHE_TTL_SECONDS = 300  # 5 minutes

GENERATION_LOCK_KEY_TEMPLATE = "lock:tension_gen:{user_id}:{conv_id}:{scene_hash}"
GENERATION_LOCK_TIMEOUT_SECONDS = 120  # 2 minutes

# Lazy orchestrator types (avoid circular import)
def _orch():
    from logic.conflict_system.conflict_synthesizer import (
        SubsystemType, EventType, SystemEvent, SubsystemResponse
    )
    return SubsystemType, EventType, SystemEvent, SubsystemResponse


# ===============================================================================
# TENSION TYPES AND STRUCTURES
# ===============================================================================

class TensionObservation(TypedDict):
    type: str
    level: float
    source: str
    note: str

class AnalyzeSceneTensionsResponse(TypedDict):
    tension_score: float
    should_generate_conflict: bool
    primary_dynamic: str
    observations: List[TensionObservation]
    error: str

class ModifyTensionResponse(TypedDict):
    success: bool
    tension_type: str
    applied_change: float
    new_level: float
    clamped: bool
    reason: str
    side_effects: List[str]
    error: str

class TensionCategory(TypedDict):
    name: str
    level: float

class TensionReportResponse(TypedDict):
    total_categories: int
    categories: List[TensionCategory]
    overall_score: float
    hotspots: List[str]
    last_updated_iso: str
    error: str


class TensionType(Enum):
    POWER = "power"
    SOCIAL = "social"
    SEXUAL = "sexual"
    EMOTIONAL = "emotional"
    ADDICTION = "addiction"
    VITAL = "vital"
    ECONOMIC = "economic"
    IDEOLOGICAL = "ideological"
    TERRITORIAL = "territorial"

class TensionLevel(Enum):
    ABSENT = 0.0
    SUBTLE = 0.2
    NOTICEABLE = 0.4
    PALPABLE = 0.6
    INTENSE = 0.8
    BREAKING = 1.0

@dataclass
class TensionSource:
    source_type: str
    source_id: Any
    contribution: float
    description: str

@dataclass
class TensionManifestation:
    tension_type: TensionType
    level: float
    physical_cues: List[str]
    dialogue_modifications: List[str]
    environmental_changes: List[str]
    player_sensations: List[str]


# ===============================================================================
# TENSION SUBSYSTEM (Refactored for Performance)
# ===============================================================================

class TensionSubsystem:
    """
    Manages tension dynamics as a non-blocking, cache-first subsystem.
    Uses scene-specific bundle caching for expensive LLM-generated content.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._synthesizer = None  # weakref set in initialize
        
        # Lightweight tension state (fast numerical updates)
        self._current_tensions: Dict[TensionType, float] = {}
        
        # LLM agents (lazy-loaded)
        self._tension_analyzer = None
        self._manifestation_generator = None
        self._escalation_narrator = None
    
    # ----- Subsystem interface -----
    
    @property
    def subsystem_type(self):
        SubsystemType, _, _, _ = _orch()
        return SubsystemType.TENSION
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'calculate_tensions',
            'build_tension',
            'resolve_tension',
            'generate_manifestation',
            'check_breaking_point',
            'analyze_tension_sources',
            'create_tension_narrative'
        }
    
    @property
    def dependencies(self) -> Set:
        SubsystemType, _, _, _ = _orch()
        return {SubsystemType.STAKEHOLDER, SubsystemType.FLOW, SubsystemType.SOCIAL}
    
    @property
    def event_subscriptions(self) -> Set:
        _, EventType, _, _ = _orch()
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.STAKEHOLDER_ACTION,
            EventType.PHASE_TRANSITION,
            EventType.PLAYER_CHOICE,
            EventType.NPC_REACTION,
            EventType.STATE_SYNC,
            EventType.TENSION_CHANGED,
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        self._synthesizer = weakref.ref(synthesizer)
        await self._load_tension_state()  # Fast initial load of numbers
        return True
    
    # ----- Core Event Handler (Refactored for non-blocking operation) -----
    
    async def handle_event(self, event):
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        try:
            et = event.event_type
            
            # Fast numerical updates (these are already performant)
            if et == EventType.CONFLICT_CREATED:
                return await self._on_conflict_created(event)
            if et == EventType.STAKEHOLDER_ACTION:
                return await self._on_stakeholder_action(event)
            if et == EventType.PHASE_TRANSITION:
                return await self._on_phase_transition(event)
            if et == EventType.PLAYER_CHOICE:
                return await self._on_player_choice(event)
            if et == EventType.NPC_REACTION:
                return await self._on_npc_reaction(event)
            if et == EventType.TENSION_CHANGED:
                return await self._on_tension_changed(event)
            if et == EventType.HEALTH_CHECK:
                return await self._on_health_check(event)
            
            # THIS IS THE CRITICAL, REFACTORED PATH
            # Scene processing now uses cached bundles
            if et == EventType.STATE_SYNC:
                return await self._on_state_sync_non_blocking(event)
            
            # Default response
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={},
                side_effects=[]
            )
        except Exception as e:
            logger.error(f"Tension system error handling event: {e}", exc_info=True)
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )
    
    # ===== THE CORE PERFORMANCE FIX =====
    
    async def _on_state_sync_non_blocking(self, event):
        """
        Handles scene processing by serving a cached bundle or triggering a background update.
        """
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        scene_context = payload.get('scene_context') or payload
        
        scene_hash = self._hash_scene_context(scene_context)
        cache_key = CACHE_KEY_TEMPLATE.format(
            user_id=self.user_id, conv_id=self.conversation_id, scene_hash=scene_hash
        )
        
        # 1. Fast Path: Check Redis Cache (using async client)
        try:
            redis_client = await get_redis_client()
            cached_bundle = await redis_client.get(cache_key)
            if cached_bundle:
                logger.debug(f"Tension bundle cache HIT for scene: {scene_hash}")
                return SubsystemResponse(
                    subsystem=self.subsystem_type, event_id=event.event_id,
                    success=True, data=json.loads(cached_bundle)
                )
        except Exception as e:
            logger.warning(f"Redis cache check for tension bundle failed: {e}")
        
        # 2. Cache Miss: Trigger background generation
        logger.debug(f"Tension bundle cache MISS for scene: {scene_hash}. Triggering background job.")
        lock_key = GENERATION_LOCK_KEY_TEMPLATE.format(
            user_id=self.user_id, conv_id=self.conversation_id, scene_hash=scene_hash
        )
        
        try:
            redis_client = await get_redis_client()
            lock_acquired = await redis_client.set(lock_key, "1", ex=GENERATION_LOCK_TIMEOUT_SECONDS, nx=True)
            
            if lock_acquired:
                try:
                    # from tasks import update_tension_bundle_cache
                    # update_tension_bundle_cache.delay(self.user_id, self.conversation_id, scene_context)
                    logger.info(f"Dispatched background tension bundle generation for scene: {scene_hash}")
                except ImportError:
                    logger.warning("Celery task not available, running generation inline as a fallback.")
                    asyncio.create_task(self.perform_bundle_generation_and_cache(scene_context))
            else:
                logger.debug("Tension bundle generation is already in progress (lock held).")
        except Exception as e:
            logger.error(f"Failed to acquire lock or dispatch task for tension bundle: {e}")

        # 3. Immediately return a minimal, non-LLM fallback response
        dominant_type, dominant_level = self._get_dominant_tension()
        fallback_manifestation = self._create_fallback_manifestation(dominant_type, dominant_level)
        
        fallback_bundle = {
            'manifestation': dataclasses.asdict(fallback_manifestation),
            'breaking_point': None,
            'current_tensions': {t.value: v for t, v in self._current_tensions.items()},
            'status': 'generation_in_progress',
        }
        
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id,
            success=True, data=fallback_bundle
        )

    
    # ===== Background Generation Logic (Called by Celery) =====
    
    async def perform_bundle_generation_and_cache(self, scene_context: Dict[str, Any]):
        """
        The core logic for the background worker. Generates the full tension bundle
        with LLM calls and caches it in Redis.
        
        This method is called by the Celery task - it's expensive but non-blocking
        because it runs in a background worker.
        """
        scene_hash = self._hash_scene_context(scene_context)
        logger.info(f"Starting background tension bundle generation for scene: {scene_hash}")
        
        try:
            # 1. Generate the expensive manifestation content (LLM call)
            manifestation = await self._generate_tension_manifestation_llm(scene_context)
            
            # 2. Check for breaking points (potential LLM call)
            breaking_point = await self.check_tension_breaking_point()
            
            # 3. Assemble the complete bundle
            bundle = {
                'manifestation': dataclasses.asdict(manifestation),
                'breaking_point': breaking_point,
                'current_tensions': {t.value: v for t, v in self._current_tensions.items()},
                'status': 'completed',
                'generated_at': datetime.now().isoformat()
            }
            
            # 4. Cache the result in Redis
            cache_key = CACHE_KEY_TEMPLATE.format(
                user_id=self.user_id,
                conv_id=self.conversation_id,
                scene_hash=scene_hash
            )
            
            redis_client.set(cache_key, json.dumps(bundle), ex=CACHE_TTL_SECONDS)
            logger.info(f"Successfully cached tension bundle for scene: {scene_hash}")
            
        except Exception as e:
            logger.error(f"Failed to generate tension bundle: {e}", exc_info=True)
        finally:
            # POLISH: Always release the lock, even on failure
            scene_hash = self._hash_scene_context(scene_context)
            lock_key = GENERATION_LOCK_KEY_TEMPLATE.format(
                user_id=self.user_id, conv_id=self.conversation_id, scene_hash=scene_hash
            )
            try:
                redis_client = await get_redis_client()
                await redis_client.delete(lock_key)
            except Exception as lock_e:
                logger.error(f"Failed to release tension generation lock: {lock_e}")

    
    async def _generate_tension_manifestation_llm(self, scene_context: Dict[str, Any]) -> TensionManifestation:
        """
        The actual slow, LLM-powered manifestation generation.
        This is called only in background workers, never in the main request path.
        """
        dominant_type, dominant_level = self._get_dominant_tension()
        
        if dominant_level < 0.1:
            return self._create_no_tension_manifestation()
        
        prompt = f"""
        Generate tension manifestations for this scene:

        Dominant Tension: {dominant_type.value} ({dominant_level:.2f})
        Scene Context: {json.dumps(scene_context, indent=2, default=str)}
        
        All current tensions:
        {json.dumps({t.value: v for t, v in self._current_tensions.items()}, indent=2)}

        Return JSON with arrays:
        {{
            "physical_cues": ["visible body language", "facial expressions", "posture changes"],
            "dialogue_modifications": ["tone changes", "word choices", "pauses"],
            "environmental_changes": ["atmosphere shifts", "ambient details", "setting cues"],
            "player_sensations": ["what the player character feels", "visceral sensations"]
        }}
        """
        
        try:
            response = await Runner.run(self.manifestation_generator, prompt)
            result = json.loads(extract_runner_response(response) or '{}')
            
            return TensionManifestation(
                tension_type=dominant_type,
                level=float(dominant_level),
                physical_cues=list(result.get('physical_cues', [])),
                dialogue_modifications=list(result.get('dialogue_modifications', [])),
                environmental_changes=list(result.get('environmental_changes', [])),
                player_sensations=list(result.get('player_sensations', []))
            )
        except Exception as e:
            logger.error(f"Failed to generate manifestation via LLM: {e}")
            return self._create_fallback_manifestation(dominant_type, float(dominant_level))
    
    # ===== Fast, Numerical Event Handlers =====
    
    async def _on_conflict_created(self, event):
        """Fast numerical tension initialization"""
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        conflict_type = payload.get('conflict_type', '')
        context = payload.get('context', {}) or {}
        
        initial_tensions = await self._determine_initial_tensions(conflict_type, context)
        side_effects = []
        
        for tension_type, level in initial_tensions.items():
            self._current_tensions[tension_type] = float(level)
            if level > 0.3:
                side_effects.append(SystemEvent(
                    event_id=f"tension_{tension_type.value}_{event.event_id}",
                    event_type=EventType.TENSION_CHANGED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'tension_type': tension_type.value,
                        'level': float(level),
                        'source': 'conflict_creation'
                    },
                    priority=5
                ))
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_initialized': {t.value: l for t, l in initial_tensions.items()},
                'dominant_tension': max(initial_tensions.items(), key=lambda x: x[1])[0].value
                    if initial_tensions else None
            },
            side_effects=side_effects
        )
    
    async def _on_stakeholder_action(self, event):
        """Fast numerical tension adjustment"""
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        action_type = str(payload.get('action_type', ''))
        intensity = float(payload.get('intensity', 0.5) or 0.5)
        
        tension_changes = self._map_action_to_tension_changes(action_type, intensity)
        side_effects = []
        
        for ttype, change in tension_changes.items():
            old_level = float(self._current_tensions.get(ttype, 0.0))
            new_level = max(0.0, min(1.0, old_level + float(change)))
            self._current_tensions[ttype] = new_level
            
            if new_level >= TensionLevel.BREAKING.value and old_level < TensionLevel.BREAKING.value:
                side_effects.append(SystemEvent(
                    event_id=f"breaking_{ttype.value}_{event.event_id}",
                    event_type=EventType.EDGE_CASE_DETECTED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'edge_case': 'tension_breaking_point',
                        'tension_type': ttype.value,
                        'level': new_level,
                        'requires_immediate_resolution': True
                    },
                    priority=1
                ))
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_modified': {t.value: c for t, c in tension_changes.items()},
                'current_tensions': {t.value: v for t, v in self._current_tensions.items()}
            },
            side_effects=side_effects
        )
    
    async def _on_phase_transition(self, event):
        """Fast numerical adjustment for phase changes"""
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        from_phase = payload.get('from_phase', '')
        to_phase = payload.get('to_phase', payload.get('phase', ''))
        
        adjustments = self._calculate_phase_tension_adjustments(from_phase, to_phase)
        for t, adj in adjustments.items():
            current = float(self._current_tensions.get(t, 0.0))
            self._current_tensions[t] = max(0.0, min(1.0, current + float(adj)))
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_adjusted': {t.value: a for t, a in adjustments.items()},
                'phase_impact': 'tensions_shifted'
            },
            side_effects=[]
        )
    
    async def _on_player_choice(self, event):
        """Fast numerical adjustment for player choices"""
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        choice_type = str(payload.get('choice_type', ''))
        
        impact = self._calculate_choice_tension_impact(choice_type)
        for t, change in impact.items():
            current = float(self._current_tensions.get(t, 0.0))
            self._current_tensions[t] = max(0.0, min(1.0, current + float(change)))
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tension_changes': {t.value: c for t, c in impact.items()},
                'player_impact': 'acknowledged'
            },
            side_effects=[]
        )
    
    async def _on_npc_reaction(self, event):
        """Fast emotional tension adjustment"""
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        emotional_state = (payload.get('emotional_state') or '').lower()
        
        if emotional_state in ['angry', 'distressed', 'fearful']:
            current = float(self._current_tensions.get(TensionType.EMOTIONAL, 0.0))
            self._current_tensions[TensionType.EMOTIONAL] = min(1.0, current + 0.1)
            await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'tension_adjusted': True},
            side_effects=[]
        )
    
    async def _on_tension_changed(self, event):
        """Handle direct tension adjustments (from modify_tension tool)"""
        _, EventType, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        ttype = str(payload.get('tension_type', 'emotional')).lower()
        change = float(payload.get('change', 0.0))
        reason = str(payload.get('reason', ''))
        
        # Resolve enum robustly
        try:
            t_enum = TensionType[ttype.upper()]
        except Exception:
            t_enum = TensionType.EMOTIONAL
        
        old = float(self._current_tensions.get(t_enum, 0.0))
        new = old + change
        clamped = False
        
        if new < 0.0:
            new = 0.0
            clamped = True
        if new > 1.0:
            new = 1.0
            clamped = True
        
        self._current_tensions[t_enum] = float(new)
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'success': True,
                'tension_type': t_enum.value,
                'applied_change': float(change),
                'new_level': float(new),
                'clamped': bool(clamped),
                'side_effects': [
                    f"tension_{t_enum.value}_set_{new:.2f}",
                    f"reason:{reason}"
                ]
            },
            side_effects=[]
        )
    
    async def _on_health_check(self, event):
        """Lightweight health check"""
        _, _, _, SubsystemResponse = _orch()
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=await self.health_check(),
            side_effects=[]
        )
    
    # ===== Helper Methods =====
    
    def _hash_scene_context(self, scene_context: Dict[str, Any]) -> str:
        """
        Creates a stable hash for a scene's context to use as a cache key.
        This ensures that similar scenes get the same cached manifestation.
        """
        # Normalize lists and sort keys to ensure consistent hashing
        normalized = {
            'loc': scene_context.get('location_id') or scene_context.get('location'),
            'npcs': sorted(list(set(scene_context.get('npcs', []) or scene_context.get('present_npcs', []))))
        }
        key_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_dominant_tension(self) -> Tuple[TensionType, float]:
        """Quick, non-blocking way to get the current dominant tension"""
        if not self._current_tensions:
            return (TensionType.EMOTIONAL, 0.0)
        dominant_type, dominant_level = max(self._current_tensions.items(), key=lambda x: x[1])
        return dominant_type, float(dominant_level)
    
    async def _load_tension_state(self):
        """Load tension state from database (fast numerical data)"""
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT tension_type, level
                    FROM TensionLevels
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
            
            for r in rows or []:
                try:
                    self._current_tensions[TensionType(str(r['tension_type']))] = float(r['level'])
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to load tension state: {e}")
    
    async def _save_tension_state(self):
        """Save tension state to database (fast numerical data)"""
        try:
            async with get_db_connection_context() as conn:
                for ttype, level in self._current_tensions.items():
                    await conn.execute("""
                        INSERT INTO TensionLevels 
                        (user_id, conversation_id, tension_type, level)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (user_id, conversation_id, tension_type)
                        DO UPDATE SET level = EXCLUDED.level, updated_at = NOW()
                    """, self.user_id, self.conversation_id, ttype.value, float(level))
        except Exception as e:
            logger.error(f"Failed to save tension state: {e}")
    
    async def _determine_initial_tensions(
        self,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Dict[TensionType, float]:
        """Determine initial tensions for a conflict"""
        patterns = {
            'power': {TensionType.POWER: 0.6, TensionType.SOCIAL: 0.3},
            'social': {TensionType.SOCIAL: 0.7, TensionType.EMOTIONAL: 0.4},
            'romantic': {TensionType.SEXUAL: 0.5, TensionType.EMOTIONAL: 0.5},
            'economic': {TensionType.ECONOMIC: 0.8, TensionType.POWER: 0.3},
            'ideological': {TensionType.IDEOLOGICAL: 0.7, TensionType.SOCIAL: 0.4}
        }
        
        for key, pattern in patterns.items():
            if key in (conflict_type or '').lower():
                return pattern
        
        return {TensionType.EMOTIONAL: 0.4, TensionType.SOCIAL: 0.3}
    
    def _map_action_to_tension_changes(
        self,
        action_type: str,
        intensity: float
    ) -> Dict[TensionType, float]:
        """Map stakeholder actions to tension changes"""
        action = (action_type or '').lower()
        changes: Dict[TensionType, float] = {}
        
        if 'aggressive' in action:
            changes[TensionType.POWER] = 0.2 * intensity
            changes[TensionType.EMOTIONAL] = 0.1 * intensity
        elif 'diplomatic' in action:
            changes[TensionType.SOCIAL] = -0.1 * intensity
            changes[TensionType.POWER] = -0.05 * intensity
        elif 'manipulative' in action:
            changes[TensionType.SOCIAL] = 0.15 * intensity
            changes[TensionType.EMOTIONAL] = 0.1 * intensity
        
        return changes
    
    def _calculate_phase_tension_adjustments(
        self,
        from_phase: str,
        to_phase: str
    ) -> Dict[TensionType, float]:
        """Calculate tension adjustments for phase transitions"""
        adjustments: Dict[TensionType, float] = {}
        
        if to_phase == 'climax':
            for t in TensionType:
                adjustments[t] = 0.2
        elif to_phase == 'resolution':
            for t in TensionType:
                adjustments[t] = -0.3
        elif from_phase == 'emerging' and to_phase == 'rising':
            adjustments[TensionType.EMOTIONAL] = 0.1
            adjustments[TensionType.SOCIAL] = 0.1
        
        return adjustments
    
    def _calculate_choice_tension_impact(self, choice_type: str) -> Dict[TensionType, float]:
        """Calculate tension impact of player choices"""
        choice = (choice_type or '').lower()
        impacts: Dict[TensionType, float] = {}
        
        if 'submit' in choice:
            impacts[TensionType.POWER] = -0.1
            impacts[TensionType.EMOTIONAL] = 0.05
        elif 'resist' in choice:
            impacts[TensionType.POWER] = 0.15
            impacts[TensionType.SOCIAL] = 0.1
        elif 'negotiate' in choice:
            impacts[TensionType.SOCIAL] = -0.05
        
        return impacts
    
    def _create_fallback_manifestation(
        self,
        tension_type: TensionType,
        level: float
    ) -> TensionManifestation:
        """Create fallback manifestation when LLM not available"""
        return TensionManifestation(
            tension_type=tension_type,
            level=level,
            physical_cues=[f"Subtle {tension_type.value} tension visible in body language"],
            dialogue_modifications=["Careful word choices", "Measured tone"],
            environmental_changes=["Charged atmosphere", "Palpable energy"],
            player_sensations=["Underlying tension", "Awareness of dynamics"]
        )
    
    def _create_no_tension_manifestation(self) -> TensionManifestation:
        """Create manifestation for no tension"""
        return TensionManifestation(
            tension_type=TensionType.EMOTIONAL,
            level=0.0,
            physical_cues=["Relaxed postures", "Natural movements"],
            dialogue_modifications=["Natural speech", "Easy conversation"],
            environmental_changes=["Comfortable atmosphere", "Relaxed setting"],
            player_sensations=["Sense of ease", "No particular tension"]
        )
    
    async def check_tension_breaking_point(self) -> Optional[Dict[str, Any]]:
        """Check if any tension has reached breaking point and generate a dynamic trigger."""
        breaking = {t: l for t, l in self._current_tensions.items() if l >= TensionLevel.BREAKING.value}
        if not breaking:
            return None
        
        breaking_type = max(breaking.items(), key=lambda x: x[1])[0]
        
        prompt = f"""
        A tension has reached a breaking point:

        Breaking Tension: {breaking_type.value}
        Level: {breaking[breaking_type]:.2f}

        Describe the trigger event and its immediate consequences in a dramatic way.
        Provide 2-3 tough choices for the player.

        Return JSON:
        {{ "trigger": "...", "consequences": ["...", "..."], "choices": ["...", "..."] }}
        """
        try:
            response = await Runner.run(self.escalation_narrator, prompt)
            result = json.loads(extract_runner_response(response) or '{}')
            return {
                'breaking_tension': breaking_type.value,
                'trigger': result.get('trigger', 'The tension snaps'),
                'consequences': result.get('consequences', ['Things cannot continue as they were.']),
                'player_choices': result.get('choices', ['React.', 'Freeze.'])
            }
        except Exception as e:
            logger.error(f"Failed to generate breaking point via LLM: {e}")
            return {
                'breaking_tension': breaking_type.value,
                'trigger': 'The tension reaches a breaking point',
                'consequences': ['Things cannot continue as they were']
            }

    # POLISH: Add the explicit health_check method for interface consistency
    async def health_check(self) -> Dict[str, Any]:
        """Provides a lightweight health status of the tension subsystem."""
        total_tension = sum(self._current_tensions.values())
        return {
            'healthy': total_tension < 5.0, # Arbitrary threshold for "healthy"
            'active_tensions': len([t for t, v in self._current_tensions.items() if v > 0.1]),
            'total_tension_score': total_tension,
            'critical_tensions': [t.value for t, v in self._current_tensions.items() if v > 0.8],
            'status': 'operational'
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get tension data for a specific conflict"""
        return {
            'tensions': {t.value: v for t, v in self._current_tensions.items()},
            'dominant_tension': max(self._current_tensions.items(), key=lambda x: x[1])[0].value
                if self._current_tensions else None,
            'total_tension': sum(self._current_tensions.values())
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of tension system"""
        return {
            'current_tensions': {t.value: v for t, v in self._current_tensions.items()},
            'breaking_points': [t.value for t, v in self._current_tensions.items() if v >= 0.9]
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if tension system is relevant to scene"""
        if any(v > 0.1 for v in self._current_tensions.values()):
            return True
        activity = (scene_context or {}).get('activity', '').lower()
        return any(word in activity for word in ['argument', 'confrontation', 'negotiation', 'intimate'])
    
    # ===== LLM Agents =====
    
    @property
    def tension_analyzer(self) -> Agent:
        if self._tension_analyzer is None:
            self._tension_analyzer = Agent(
                name="Tension Analyzer",
                instructions="Analyze various sources to determine tension levels and types.",
                model="gpt-5-nano",
            )
        return self._tension_analyzer
    
    @property
    def manifestation_generator(self) -> Agent:
        if self._manifestation_generator is None:
            self._manifestation_generator = Agent(
                name="Tension Manifestation Generator",
                instructions="""
                Generate specific, sensory manifestations of tension in scenes.
                Focus on:
                - Physical cues that players can observe
                - Dialogue modifications that reflect tension
                - Environmental changes that enhance atmosphere
                - Player sensations that create immersion
                
                Make manifestations subtle and realistic.
                """,
                model="gpt-5-nano",
            )
        return self._manifestation_generator
    
    @property
    def escalation_narrator(self) -> Agent:
        if self._escalation_narrator is None:
            self._escalation_narrator = Agent(
                name="Tension Escalation Narrator",
                instructions="Narrate how tensions build, peak, and release.",
                model="gpt-5-nano",
            )
        return self._escalation_narrator


# ===============================================================================
# ASYNC CELERY TASK (to be defined in your tasks.py)
# ===============================================================================

# @celery_app.task(name="tasks.update_tension_bundle_cache")
# def update_tension_bundle_cache(user_id: int, conversation_id: int, scene_context: Dict[str, Any]):
#     """Celery task to generate and cache the tension bundle in the background."""
#     from logic.conflict_system.conflict_synthesizer import get_synthesizer, SubsystemType
# 
#     async def do_generation():
#         synthesizer = await get_synthesizer(user_id, conversation_id)
#         tension_system = synthesizer._subsystems.get(SubsystemType.TENSION)
#         if tension_system:
#             await tension_system.perform_bundle_generation_and_cache(scene_context)
# 
#     asyncio.run(do_generation())


# ===============================================================================
# PUBLIC API (Now much faster and more resilient)
# ===============================================================================

@function_tool
async def analyze_scene_tensions(
    ctx: RunContextWrapper,
    scene_description: str,
    npcs_present: List[int],
    current_activity: str
) -> AnalyzeSceneTensionsResponse:
    """
    Analyze scene tensions using cached bundle when available.
    This tool is now much faster because it uses cached manifestations.
    """
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    SubsystemType, EventType, SystemEvent, _ = _orch()
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    event = SystemEvent(
        event_id=f"tension_analyze_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={
            'scene_context': {
                'scene_description': scene_description,
                'npcs': npcs_present,
                'present_npcs': npcs_present,
                'activity': current_activity,
                'location': scene_description[:100]  # Use part of description as location
            }
        },
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )
    
    responses = await synthesizer.emit_event(event) or []
    data = {}
    
    for r in responses:
        if r.subsystem == SubsystemType.TENSION:
            data = r.data or {}
            break
    
    # Extract tensions from bundle
    current_tensions = data.get('current_tensions', {})
    score = max(0.0, min(1.0, sum(current_tensions.values()) / max(1, len(current_tensions))))
    primary = max(current_tensions.items(), key=lambda x: x[1])[0] if current_tensions else 'none'
    
    observations: List[TensionObservation] = []
    for t, lvl in current_tensions.items():
        level = float(max(0.0, min(1.0, lvl)))
        observations.append({
            'type': str(t),
            'level': level,
            'source': 'tension_system',
            'note': f"Current {t} tension level"
        })
    
    return {
        'tension_score': float(score),
        'should_generate_conflict': bool(score > 0.6),
        'primary_dynamic': primary,
        'observations': observations,
        'error': "" if data else "No response from tension system",
    }


@function_tool
async def modify_tension(
    ctx: RunContextWrapper,
    tension_type: str,
    change: float,
    reason: str
) -> ModifyTensionResponse:
    """
    Modify tension levels. This remains fast as it's a numerical operation.
    """
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    SubsystemType, EventType, SystemEvent, _ = _orch()
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    event = SystemEvent(
        event_id=f"manual_tension_{tension_type}",
        event_type=EventType.TENSION_CHANGED,
        source_subsystem=SubsystemType.TENSION,
        payload={
            'tension_type': tension_type,
            'change': float(change),
            'reason': reason
        },
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )
    
    responses = await synthesizer.emit_event(event) or []
    data = responses[0].data if responses else {}
    
    new_level = float(max(0.0, min(1.0, data.get('new_level', 0.0))))
    side_effects = data.get('side_effects', [])
    
    if not isinstance(side_effects, list):
        side_effects = [str(side_effects)]
    
    return {
        'success': bool(data.get('success', bool(responses))),
        'tension_type': str(data.get('tension_type', tension_type)),
        'applied_change': float(data.get('applied_change', change)),
        'new_level': new_level,
        'clamped': bool(data.get('clamped', False)),
        'reason': str(reason),
        'side_effects': [str(s) for s in side_effects[:20]],
        'error': "" if responses else "No response from tension system",
    }


@function_tool
async def get_tension_report(
    ctx: RunContextWrapper
) -> TensionReportResponse:
    """
    Get comprehensive tension report. Fast because it reads numerical state.
    """
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    SubsystemType, EventType, SystemEvent, _ = _orch()
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Get current state (fast, no LLM)
    tension_system = synthesizer._subsystems.get(SubsystemType.TENSION)
    if not tension_system:
        return {
            'total_categories': 0,
            'categories': [],
            'overall_score': 0.0,
            'hotspots': [],
            'last_updated_iso': datetime.now().isoformat(),
            'error': "Tension system not available"
        }
    
    state = await tension_system.get_state()
    current_tensions = state.get('current_tensions', {})
    
    categories: List[TensionCategory] = []
    for name, lvl in current_tensions.items():
        val = float(max(0.0, min(1.0, lvl)))
        categories.append({'name': str(name), 'level': val})
    
    overall = float(max(0.0, min(1.0, sum(current_tensions.values()) / max(1, len(current_tensions)))))
    hotspots = [name for name, lvl in current_tensions.items() if lvl > 0.7]
    
    return {
        'total_categories': len(categories),
        'categories': categories,
        'overall_score': overall,
        'hotspots': hotspots,
        'last_updated_iso': datetime.now().isoformat(),
        'error': ""
    }
