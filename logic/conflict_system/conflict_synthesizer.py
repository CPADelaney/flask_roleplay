# logic/conflict_system/conflict_synthesizer.py
"""
Conflict Synthesizer - THE Central Orchestration System

This module is the single orchestrator for all conflict operations.
All conflict subsystems register with and communicate through this synthesizer.
No other module should attempt to orchestrate - they only handle their specific domain.

PRODUCTION-READY VERSION with:
- Scene-scoped bundles for fast context assembly
- Parallel subsystem processing with proper cleanup
- Smart caching with TTL and thread-safe access
- Performance metrics tracking with P95
- Delta updates for incremental refresh
- Canonical conflict prioritization
- All critical bugs fixed and safety improvements applied
"""

import logging
import json
import asyncio
import time
import uuid
import hashlib
import os
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING, Tuple
from pydantic import BaseModel, Field, ConfigDict  
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, OrderedDict
import weakref

from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from agents import function_tool, RunContextWrapper, Runner

if TYPE_CHECKING:
    from nyx.nyx_agent.context import SceneScope

from logic.conflict_system.background_processor import (
    get_conflict_scheduler,
    BackgroundConflictProcessor
)
from logic.conflict_system.integration_hooks import ConflictEventHooks

logger = logging.getLogger(__name__)

# Configuration from environment
PARALLEL_TIMEOUT = float(os.getenv('CONFLICT_PARALLEL_TIMEOUT', '2.0'))
MAX_PARALLEL_TASKS = int(os.getenv('CONFLICT_MAX_PARALLEL', '6'))
BUNDLE_TTL_SECONDS = float(os.getenv('CONFLICT_BUNDLE_TTL', '30.0'))
MAX_EVENT_QUEUE = int(os.getenv('CONFLICT_EVENT_Q_MAX', '1000'))
MAX_EVENT_HISTORY = int(os.getenv('CONFLICT_HISTORY_MAX', '1000'))
MAX_BUNDLE_CACHE = int(os.getenv('CONFLICT_BUNDLE_CACHE_MAX', '256'))
LLM_ROUTE_TIMEOUT = float(os.getenv('CONFLICT_LLM_TIMEOUT', '5.0'))
MAX_SIDE_EFFECTS_PER_EVENT = int(os.getenv('CONFLICT_MAX_SIDE_EFFECTS', '100'))

# Time source usage:
# - time.perf_counter() for durations (monotonic, unaffected by clock adjustments)
# - time.time() for timestamps (wall clock, needed for cross-process cache TTLs)

# ===============================================================================
# ORCHESTRATION TYPES
# ===============================================================================

class ConflictContext(BaseModel):
    """Context data for conflict operations."""
    model_config = ConfigDict(extra="ignore")

    # Scene-related fields
    scene_type: Optional[str] = None
    scene_description: Optional[str] = None
    activity: Optional[str] = None
    activity_type: Optional[str] = None
    location: Optional[str] = None
    location_id: Optional[int] = None
    timestamp: Optional[str] = None  # ISO-8601 preferred

    # Participant fields (various names used across modules)
    participants: Optional[List[int]] = None
    present_npcs: Optional[List[int]] = None
    npcs: Optional[List[int]] = None
    character_ids: Optional[List[int]] = None
    stakeholders: Optional[List[int]] = None

    # Multi-party characteristics
    is_multiparty: Optional[bool] = False
    party_count: Optional[int] = Field(None, ge=2)  # Number of distinct parties
    multiparty_dynamics: Optional[Dict[str, Any]] = None  # Alliance potential, etc.
    faction_data: Optional[List[Dict[str, Any]]] = None  # Info about each faction

    # Conflict-specific fields
    conflict_type: Optional[str] = None
    intensity: Optional[str] = None  # e.g., "tension", "friction"
    intensity_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = None
    phase: Optional[str] = None

    # Context and history
    recent_events: Optional[List[str]] = None
    evidence: Optional[List[str]] = None
    tension_source: Optional[str] = None

    # Template and generation fields
    use_template: Optional[bool] = None
    template_id: Optional[int] = None
    generation_data: Optional[Dict[str, Any]] = None
    hooks: Optional[List[str]] = None
    complexity: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Integration mode and processing flags
    integration_mode: Optional[str] = None
    integrating_conflicts: Optional[bool] = None
    boost_engagement: Optional[bool] = None

    # Flexible metadata for additional fields
    metadata: Optional[Dict[str, Any]] = None


class ConflictCreationResponse(BaseModel):
    """Response model for conflict creation."""
    model_config = ConfigDict(extra="ignore")

    conflict_id: int = Field(..., ge=0)
    conflict_type: Optional[str] = None
    conflict_name: Optional[str] = None
    status: str
    message: Optional[str] = None
    created_at: Optional[str] = None  # ISO-8601

    # Phase and flow information
    initial_phase: Optional[str] = None
    pacing_style: Optional[str] = None

    # Stakeholder information
    stakeholders_created: Optional[int] = None
    stakeholder_ids: Optional[List[int]] = None

    # Template information
    template_used: Optional[int] = None
    generated_conflict: Optional[int] = None
    narrative_hooks: Optional[List[str]] = None

    # Additional details
    conflict_details: Optional[Dict[str, Any]] = None
    subsystem_responses: Optional[Dict[str, Any]] = None


class SceneContext(BaseModel):
    """Context for scene processing."""
    model_config = ConfigDict(extra="ignore")

    # Identification
    scene_id: int
    scene_type: str

    # Participants & place
    characters_present: List[int]
    location_id: int

    # Optional timing & linkage
    timestamp: Optional[str] = None  # ISO-8601
    previous_scene_id: Optional[int] = None

    # Optional content
    action_sequence: Optional[List[str]] = None
    dialogue: Optional[List[Dict[str, str]]] = None


class SceneProcessingResponse(BaseModel):
    """Response from scene processing."""
    model_config = ConfigDict(extra="ignore")

    scene_id: int
    processed: bool = True
    
    # Conflict information
    conflicts_active: Optional[List[int]] = None
    conflicts_detected: Optional[List[str]] = None
    
    # Subsystem data
    subsystem_data: Optional[Dict[str, Any]] = None


class SubsystemType(Enum):
    TENSION = "tension"
    STAKEHOLDER = "stakeholder"
    PHASE = "phase"
    FLOW = "flow"
    SOCIAL = "social"
    LEVERAGE = "leverage"
    BACKGROUND = "background"
    VICTORY = "victory"
    CANON = "canon"
    TEMPLATE = "template"
    EDGE_HANDLER = "edge_handler"
    SLICE_OF_LIFE = "slice_of_life"
    DETECTION = "detection"
    RESOLUTION = "resolution"
    ORCHESTRATOR = "orchestrator"  # <-- add this

class EventType(Enum):
    CONFLICT_CREATED = "conflict_created"
    CONFLICT_UPDATED = "conflict_updated"
    CONFLICT_RESOLVED = "conflict_resolved"
    TENSION_CHANGED = "tension_changed"
    PHASE_TRANSITION = "phase_transition"
    INTENSITY_CHANGED = "intensity_changed"
    STAKEHOLDER_ACTION = "stakeholder_action"
    PLAYER_CHOICE = "player_choice"
    NPC_REACTION = "npc_reaction"
    EDGE_CASE_DETECTED = "edge_case_detected"
    CANON_ESTABLISHED = "canon_established"
    TEMPLATE_GENERATED = "template_generated"
    HEALTH_CHECK = "health_check"
    STATE_SYNC = "state_sync"
    DAY_TRANSITION = "day_transition"  # ADD THIS
    SCENE_ENTER = "scene_enter"  # ADD THIS
    CONFLICT_SIGNIFICANT_CHANGE = "conflict_significant_change"  # ADD THIS


@dataclass
class SystemEvent:
    event_id: str
    event_type: EventType
    source_subsystem: SubsystemType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    target_subsystems: Optional[Set[SubsystemType]] = None
    requires_response: bool = False
    priority: int = 5  # 1-10, 1 is highest priority


@dataclass
class SubsystemResponse:
    subsystem: SubsystemType
    event_id: str
    success: bool
    data: Dict[str, Any]
    side_effects: List[SystemEvent] = field(default_factory=list)


# ===============================================================================
# SUBSYSTEM INTERFACE
# ===============================================================================

class ConflictSubsystem:
    """Base interface all subsystems must implement"""
    
    @property
    def subsystem_type(self) -> SubsystemType:
        """Return the subsystem type"""
        raise NotImplementedError
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        raise NotImplementedError
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        """Return other subsystems this depends on"""
        return set()
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        """Return events this subsystem wants to receive"""
        return set()
    
    async def initialize(self, synthesizer: 'ConflictSynthesizer') -> bool:
        """Initialize the subsystem with synthesizer reference"""
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Handle an event from the synthesizer"""
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        return {'healthy': True}


# ===============================================================================
# THE MASTER CONFLICT SYNTHESIZER
# ===============================================================================

class ConflictSynthesizer:
    """
    THE central orchestrator for all conflict subsystems.
    All conflict operations flow through this synthesizer.
    
    PRODUCTION-READY with scene bundles, caching, and parallel processing.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._subsystems: Dict[SubsystemType, Any] = {}
        
        # ADD: Background processor reference
        self.scheduler = get_conflict_scheduler()
        self.processor = self.scheduler.get_processor(user_id, conversation_id)
        
        # ADD: Cache for scene context
        self._scene_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Event system with bounded queue
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_EVENT_QUEUE)
        self._event_handlers: Dict[EventType, List[SubsystemType]] = defaultdict(list)
        self._event_history: List[SystemEvent] = []
        
        # State management
        self._conflict_states: Dict[int, Dict[str, Any]] = {}
        self._global_metrics = {
            'total_conflicts': 0,
            'active_conflicts': 0,
            'resolved_conflicts': 0,
            'system_health': 1.0,
            'complexity_score': 0.0
        }
        
        # Processing control
        self._processing = False
        self._shutdown = False
        
        # Orchestration agents (lazy loaded)
        self._orchestrator = None
        self._state_manager = None
        self._event_router = None
        self._health_monitor = None
        
        # Bundle cache for scene-scoped conflict data (bounded with OrderedDict for LRU)
        self._bundle_cache: OrderedDict[str, Tuple[Dict, float]] = OrderedDict()
        self._bundle_ttl = BUNDLE_TTL_SECONDS
        self._cache_hits = 0
        self._cache_misses = 0
        self._bundle_lock = asyncio.Lock()  # Thread-safe cache access
        
        # Performance tracking with detailed metrics
        self._performance_metrics = {
            'bundle_fetch_times': [],
            'parallel_process_times': [],
            'cache_operations': 0,
            'events_processed': 0,
            'timeouts_count': 0,
            'failures_count': 0,
            'subsystem_timeouts': defaultdict(int),  # Track per-subsystem timeouts
            'subsystem_failures': defaultdict(int)   # Track per-subsystem failures
        }
        
        # Parallel processing control
        self._parallel_semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
        self._parallel_timeout = PARALLEL_TIMEOUT
        
        
        # Background task handles for clean shutdown
        self._bg_tasks: Dict[str, asyncio.Task] = {}

        self._last_scene = None
    
    # ========== Subsystem Registration ==========
    
    async def register_subsystem(self, subsystem: ConflictSubsystem) -> bool:
        try:
            if not await subsystem.initialize(self):
                logger.error(f"Failed to initialize {subsystem.subsystem_type}")
                return False
    
            self._subsystems[subsystem.subsystem_type] = subsystem   # <-- unified
            for event_type in subsystem.event_subscriptions:
                self._event_handlers[event_type].append(subsystem.subsystem_type)
            logger.info(f"Registered subsystem: {subsystem.subsystem_type}")
            return True
        except Exception:
            logger.exception(f"Error registering subsystem {subsystem.subsystem_type}")
            return False
        
    async def initialize_all_subsystems(self):
        """Initialize all default subsystems and start background workers (idempotent)."""
        try:
            from logic.conflict_system.tension import TensionSubsystem
            from logic.conflict_system.autonomous_stakeholder_actions import StakeholderAutonomySystem
            from logic.conflict_system.conflict_flow import ConflictFlowSubsystem
            from logic.conflict_system.social_circle import SocialDynamicsSubsystem
            from logic.conflict_system.leverage import LeverageSubsystem
            from logic.conflict_system.background_grand_conflicts import BackgroundConflictSubsystem
            from logic.conflict_system.conflict_canon import ConflictCanonSubsystem
            from logic.conflict_system.dynamic_conflict_template import TemplateGeneratorSubsystem
            from logic.conflict_system.edge_cases import ConflictEdgeCaseSubsystem
            from logic.conflict_system.slice_of_life_conflicts import SliceOfLifeConflictSubsystem
            from logic.conflict_system.enhanced_conflict_integration import EnhancedIntegrationSubsystem
            from logic.conflict_system.conflict_victory import ConflictVictorySubsystem
    
            subsystems = [
                TensionSubsystem(self.user_id, self.conversation_id),
                StakeholderAutonomySystem(self.user_id, self.conversation_id),
                PhaseSubsystem(self.user_id, self.conversation_id),
                ConflictFlowSubsystem(self.user_id, self.conversation_id),
                SocialDynamicsSubsystem(self.user_id, self.conversation_id),
                LeverageSubsystem(self.user_id, self.conversation_id),
                BackgroundConflictSubsystem(self.user_id, self.conversation_id),
                ConflictVictorySubsystem(self.user_id, self.conversation_id),
                ConflictCanonSubsystem(self.user_id, self.conversation_id),
                TemplateGeneratorSubsystem(self.user_id, self.conversation_id),
                ConflictEdgeCaseSubsystem(self.user_id, self.conversation_id),
                SliceOfLifeConflictSubsystem(self.user_id, self.conversation_id),
                EnhancedIntegrationSubsystem(self.user_id, self.conversation_id),
            ]
    
            # (Re)register only if not already present
            for s in subsystems:
                if s.subsystem_type not in self._subsystems:
                    await self.register_subsystem(s)
    
            # Start background workers exactly once
            if 'events' not in self._bg_tasks or self._bg_tasks['events'].done():
                self._bg_tasks['events'] = asyncio.create_task(self._process_events())
            if 'cleanup' not in self._bg_tasks or self._bg_tasks['cleanup'].done():
                self._bg_tasks['cleanup'] = asyncio.create_task(self._periodic_cache_cleanup())
    
            logger.info("Conflict subsystems initialized")
        except Exception:
            logger.exception("Failed to initialize subsystems")
            raise

    
    # ========== Event System ==========
    
    async def emit_event(self, event: SystemEvent) -> Optional[List[SubsystemResponse]]:
        """Emit an event to relevant subsystems with sane backpressure and side-effect limits."""
        try:
            # Bounded history
            self._event_history.append(event)
            if len(self._event_history) > MAX_EVENT_HISTORY:
                self._event_history = self._event_history[-MAX_EVENT_HISTORY:]
    
            if event.requires_response:
                responses = await self._process_event_parallel(event)
                if responses is not None:
                    self._performance_metrics['events_processed'] += 1
    
                # Best-effort side effects
                if responses:
                    side_count = 0
                    for r in responses:
                        for se in r.side_effects:
                            if side_count >= MAX_SIDE_EFFECTS_PER_EVENT:
                                logger.warning(f"Side-effect cap reached for {event.event_id}")
                                break
                            try:
                                self._event_queue.put_nowait(se)
                                side_count += 1
                            except asyncio.QueueFull:
                                logger.warning(f"Event queue full, dropping side effect {se.event_id}")
                                self._performance_metrics['failures_count'] += 1
                    if side_count:
                        logger.debug(f"Enqueued {side_count} side effects for {event.event_id}")
                return responses
    
            # Async fire-and-forget
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Event queue full, dropping event {event.event_id}")
                self._performance_metrics['failures_count'] += 1
                return None
            return None
    
        except Exception:
            logger.exception(f"Failed to emit event {event.event_id}")
            self._performance_metrics['failures_count'] += 1
            return None

    async def conflict_context_for_scene(self, scene_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get conflict context optimized for current scene. Main entry for NyxContext.
        """
        # Stable, safe cache key (handles lists/dicts)
        try:
            key_str = json.dumps(scene_info, sort_keys=True, default=str)
        except Exception:
            key_str = str(scene_info)
        cache_key = f"scene_{hashlib.md5(key_str.encode()).hexdigest()}"
        if cache_key in self._scene_cache:
            cached = self._scene_cache[cache_key]
            if datetime.utcnow().timestamp() - cached['timestamp'] < self._cache_ttl:
                return cached['data']

        # Background fast-path: scene-relevant updates (ambient effects, immediate NPC updates enqueued)
        ambient_from_processor = []
        try:
            immediate = await self.processor.process_scene_relevant_updates(scene_info)
            ambient_from_processor = immediate.get('ambient_effects') or []
        except Exception as e:
            logger.debug(f"Scene-relevant background updates failed: {e}")
    
        event = SystemEvent(
            event_id=f"scene_{datetime.now().timestamp()}",
            event_type=EventType.SCENE_ENTER,
            source_subsystem=SubsystemType.ORCHESTRATOR,
            payload={'scene_context': scene_info},
            target_subsystems={SubsystemType.BACKGROUND},
        )
    
        responses = await self.emit_event(event)
    
        context = {
            'conflicts': [],
            'tensions': {},
            'opportunities': [],
            'ambient_effects': [],
            'world_tension': 0.0,
        }
    
        if responses:
            for response in responses:
                if response.success and response.data:
                    if response.subsystem == SubsystemType.BACKGROUND:
                        scene_ctx = response.data.get('scene_context', {}) or {}
                        context['conflicts'].extend(scene_ctx.get('active_conflicts', []) or [])
                        # Accept either key and merge processor ambient
                        ambient = scene_ctx.get('ambient_effects') or scene_ctx.get('ambient_atmosphere') or []
                        context['ambient_effects'].extend(ambient or [])
                        context['ambient_effects'].extend(ambient_from_processor or [])
                        wt = scene_ctx.get('world_tension')
                        if isinstance(wt, (int, float)):
                            context['world_tension'] = float(wt)
    
        self._scene_cache[cache_key] = {
            'timestamp': datetime.utcnow().timestamp(),
            'data': context,
        }
        return context
    
    
    async def handle_day_transition(self, new_day: int) -> Dict[str, Any]:
        """
        Handle game day transitions with background processing.
        """
        logger.info(f"Processing day transition to day {new_day}")
    
        result = await ConflictEventHooks.on_game_day_transition(
            self.user_id, self.conversation_id, new_day
        )
        
        # Kick off processor daily updates so background conflicts tick and queue items (news, etc.)
        try:
            daily = await self.processor.process_daily_updates()
            # Best-effort: drain a few high-priority items now
            if self.processor._processing_queue:
                await self.processor.process_queued_items(max_items=5)
            logger.debug(f"Background daily updates: {daily}")
        except Exception as e:
            logger.debug(f"Daily background update skipped/failed: {e}")

    
        event = SystemEvent(
            event_id=f"day_{new_day}_{datetime.now().timestamp()}",
            event_type=EventType.DAY_TRANSITION,
            source_subsystem=SubsystemType.ORCHESTRATOR,
            payload={'new_day': new_day, 'processing_result': result},
            # no 'broadcast' field in SystemEvent; rely on subscriptions
            requires_response=False,
            priority=7,
        )
    
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, day transition event dropped")
    
        async with self._bundle_lock:
            self._bundle_cache.clear()
            logger.info(f"Cleared bundle cache for day {new_day}")
    
        return result
    
    
    async def process_background_queue(self, max_items: int = 5) -> List[Dict[str, Any]]:
        """Process items from the background conflict queue."""
        items = await self.processor.process_queued_items(max_items=max_items)
        # Best-effort: emit simple events for generated news
        try:
            for it in items or []:
                if it.get('type') == 'news' and it.get('result', {}).get('generated'):
                    ev = SystemEvent(
                        event_id=f"bg_news_{it.get('conflict_id')}_{time.time()}",
                        event_type=EventType.STATE_SYNC,
                        source_subsystem=SubsystemType.ORCHESTRATOR,
                        payload={'news': it['result'].get('news'), 'conflict_id': it.get('conflict_id')},
                        requires_response=False,
                        priority=9
                    )
                    try:
                        self._event_queue.put_nowait(ev)
                    except asyncio.QueueFull:
                        pass
        except Exception:
            logger.debug("Failed to enqueue background news events (non-fatal)")
        return items
    
    
    async def should_generate_content(self, content_type: str, entity_id: int) -> tuple[bool, str]:
        """
        Async: check if we should generate new content (respects limits).
        """
        if content_type == 'news':
            return await self.processor.should_generate_news(entity_id)
        return True, "allowed"
    
    
    async def _handle_scene_transition(self, old_scene, new_scene):
        """
        Internal method to handle scene transitions.
        """
        if old_scene == new_scene:
            return
    
        context = await ConflictEventHooks.on_scene_transition(
            self.user_id, self.conversation_id, old_scene, new_scene
        )
    
        event = SystemEvent(
            event_id=f"scene_transition_{datetime.now().timestamp()}",
            event_type=EventType.SCENE_ENTER,
            source_subsystem=SubsystemType.ORCHESTRATOR,
            payload={'old_scene': old_scene, 'new_scene': new_scene, 'context': context},
            target_subsystems={SubsystemType.BACKGROUND},
            requires_response=False,
            priority=6,
        )
    
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    
    async def _process_events(self):
        """Background event processing loop with timeout protection"""
        self._processing = True
        
        while not self._shutdown:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                # Use parallel processing with timeouts for background events too
                responses = await self._process_event_parallel(event)
                
                # Track successful event processing
                self._performance_metrics['events_processed'] += 1
                
                # Process side effects (bounded by queue size and cap)
                if responses:
                    side_count = 0
                    for response in responses:
                        if side_count >= MAX_SIDE_EFFECTS_PER_EVENT:
                            break
                        for side_effect in response.side_effects:
                            if side_count >= MAX_SIDE_EFFECTS_PER_EVENT:
                                logger.warning(f"Capping side effects at {MAX_SIDE_EFFECTS_PER_EVENT} for background event")
                                break
                            try:
                                self._event_queue.put_nowait(side_effect)
                                side_count += 1
                            except asyncio.QueueFull:
                                logger.warning(f"Event queue full, dropping side effect {side_effect.event_id}")
                                self._performance_metrics['failures_count'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception:
                logger.exception("Error processing event")
                self._performance_metrics['failures_count'] += 1
        
        self._processing = False
    
    async def _get_subsystem_bundle_with_timeout(
        self,
        subsystem: ConflictSubsystem,
        scope: 'SceneScope',
        subsystem_type: SubsystemType
    ) -> Dict[str, Any]:
        """Call subsystem.get_scene_bundle(scope) with orchestrator timeout & accounting."""
        if not hasattr(subsystem, "get_scene_bundle"):
            return {}
        try:
            result = await asyncio.wait_for(
                subsystem.get_scene_bundle(scope),
                timeout=self._parallel_timeout
            )
            return result if isinstance(result, dict) else {}
        except asyncio.TimeoutError:
            logger.warning(f"{subsystem_type.value}.get_scene_bundle timed out")
            self._performance_metrics['timeouts_count'] += 1
            self._performance_metrics['subsystem_timeouts'][subsystem_type.value] += 1
            return {}
        except Exception as e:
            logger.debug(f"{subsystem_type.value} bundle fetch failed: {e}")
            self._performance_metrics['failures_count'] += 1
            self._performance_metrics['subsystem_failures'][subsystem_type.value] += 1
            return {}
    
  
    async def _process_event_parallel(self, event: SystemEvent) -> List[SubsystemResponse]:
        """Process event in parallel across subsystems with detailed tracking."""
        start_time = time.perf_counter()
    
        # Handlers subscribed to this event
        handler_subsystems = list(self._event_handlers.get(event.event_type, []))
    
        # Targeted subsystems (if any)
        if event.target_subsystems:
            handler_subsystems.extend(list(event.target_subsystems))
    
        # Create tasks
        task_map: Dict[asyncio.Task, SubsystemType] = {}
        for subsystem_type in set(handler_subsystems):
            subsystem = self._subsystems.get(subsystem_type)
            if not subsystem:
                continue
            task = asyncio.create_task(self._handle_with_semaphore(subsystem, event))
            task_map[task] = subsystem_type
    
        if not task_map:
            return []
    
        done, pending = await asyncio.wait(task_map.keys(), timeout=self._parallel_timeout)
    
        # Cancel pending with accounting
        for task in pending:
            subsystem_type = task_map[task]
            logger.warning(f"Subsystem {subsystem_type.value} timed out on event {event.event_id}")
            self._performance_metrics['timeouts_count'] += 1
            self._performance_metrics['subsystem_timeouts'][subsystem_type.value] += 1
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
        # Collect results
        responses: List[SubsystemResponse] = []
        for task in done:
            subsystem_type = task_map[task]
            try:
                result = task.result()
                if isinstance(result, SubsystemResponse):
                    responses.append(result)
                    if not result.success:
                        self._performance_metrics['failures_count'] += 1
                        self._performance_metrics['subsystem_failures'][subsystem_type.value] += 1
                else:
                    logger.error(f"Subsystem {subsystem_type.value} returned non-Response on {event.event_id}")
                    self._performance_metrics['failures_count'] += 1
                    self._performance_metrics['subsystem_failures'][subsystem_type.value] += 1
            except Exception:
                logger.exception(f"Subsystem {subsystem_type.value} crashed on {event.event_id}")
                self._performance_metrics['failures_count'] += 1
                self._performance_metrics['subsystem_failures'][subsystem_type.value] += 1
    
        # Perf tracking
        elapsed = time.perf_counter() - start_time
        self._performance_metrics['parallel_process_times'].append(elapsed)
        if len(self._performance_metrics['parallel_process_times']) > 100:
            self._performance_metrics['parallel_process_times'].pop(0)
    
        if pending:
            logger.info(f"Parallel processing: {len(done)} completed, {len(pending)} timed out in {elapsed:.3f}s")
    
        return responses
        
    async def _handle_with_semaphore(self, subsystem: ConflictSubsystem, 
                                    event: SystemEvent) -> SubsystemResponse:
        """Handle event with semaphore control for parallel limiting"""
        async with self._parallel_semaphore:
            try:
                return await subsystem.handle_event(event)
            except Exception as e:
                logger.error(f"Error in {subsystem.subsystem_type} handling: {e}")
                return SubsystemResponse(
                    subsystem=subsystem.subsystem_type,
                    event_id=event.event_id,
                    success=False,
                    data={'error': str(e)}
                )
    
    # ========== Conflict Operations ==========
    
    async def create_conflict(
        self,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new conflict with all subsystems participating"""
        
        # Generate a stable operation id so subsystems can reconcile deferred work
        operation_id = f"create_{conflict_type}_{uuid.uuid4()}"
        
        # Determine which subsystems should be active (fixed async call)
        active_subsystems = await self._determine_required_subsystems(conflict_type, context)
        
        # Create conflict event
        event = SystemEvent(
            event_id=operation_id,
            event_type=EventType.CONFLICT_CREATED,
            source_subsystem=SubsystemType.DETECTION,
            payload={
                'conflict_type': conflict_type,
                'context': context
            },
            target_subsystems=active_subsystems,
            requires_response=True,
            priority=1
        )
        
        # Get responses from all subsystems (uses parallel processing)
        responses = await self.emit_event(event)
        
        # Handle case where emit_event returns None on failure
        if responses is None:
            logger.warning(f"Failed to emit conflict creation event for {conflict_type}")
            return {'status': 'failed', 'message': 'Event emission failed'}
        
        # Aggregate responses into conflict creation result
        result = self._aggregate_conflict_creation(responses)
        
        # If we have a real conflict_id, emit a follow-up STATE_SYNC
        conflict_id = result.get('conflict_id')
        if conflict_id is not None:
            # Track state in-memory with timestamp
            self._conflict_states[conflict_id] = {
                **result,
                'last_updated': time.time()
            }
            
            await self.emit_event(SystemEvent(
                event_id=f"conflict_ready_{operation_id}",
                event_type=EventType.STATE_SYNC,
                source_subsystem=SubsystemType.SLICE_OF_LIFE,
                payload={
                    'operation_id': operation_id,
                    'conflict_id': conflict_id,
                    'context': context,
                },
                requires_response=False,
                priority=4
            ))
            
            # Update metrics
            self._global_metrics['total_conflicts'] += 1
            self._global_metrics['active_conflicts'] += 1
            
            # Invalidate relevant caches
            await self._invalidate_caches_for_conflict(conflict_id)
        
        return result
    
    async def update_conflict(
        self,
        conflict_id: int,
        update_type: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a conflict with coordinated subsystem responses"""
        
        # Create update event
        event = SystemEvent(
            event_id=f"update_{conflict_id}_{datetime.now().timestamp()}",
            event_type=EventType.CONFLICT_UPDATED,
            source_subsystem=SubsystemType.FLOW,
            payload={
                'conflict_id': conflict_id,
                'update_type': update_type,
                'data': update_data
            },
            requires_response=True,
            priority=3
        )
        
        # Process update (uses parallel processing)
        responses = await self.emit_event(event)
        
        # Handle case where emit_event returns None on failure
        if responses is None:
            logger.warning(f"Failed to emit conflict update event for conflict {conflict_id}")
            return {'status': 'failed', 'message': 'Event emission failed'}
        
        # Aggregate responses
        result = self._aggregate_update_responses(responses)
        
        # Update state with timestamp
        if conflict_id not in self._conflict_states:
            self._conflict_states[conflict_id] = {}
        self._conflict_states[conflict_id].update(result)
        self._conflict_states[conflict_id]['last_updated'] = time.time()
        
        # Invalidate caches
        await self._invalidate_caches_for_conflict(conflict_id)
        
        return result
    
    async def resolve_conflict(
        self,
        conflict_id: int,
        resolution_type: str,
        resolution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve a conflict with all subsystems participating"""
        
        # Create resolution event
        event = SystemEvent(
            event_id=f"resolve_{conflict_id}_{datetime.now().timestamp()}",
            event_type=EventType.CONFLICT_RESOLVED,
            source_subsystem=SubsystemType.RESOLUTION,
            payload={
                'conflict_id': conflict_id,
                'resolution_type': resolution_type,
                'context': resolution_context
            },
            requires_response=True,
            priority=2
        )
        
        # Process resolution (uses parallel processing)
        responses = await self.emit_event(event)
        
        # Handle case where emit_event returns None on failure
        if responses is None:
            logger.warning(f"Failed to emit conflict resolution event for conflict {conflict_id}")
            return {'resolved': False, 'message': 'Event emission failed'}
        
        # Aggregate responses
        result = self._aggregate_resolution_responses(responses)
        
        # Update metrics (clamp active_conflicts to non-negative)
        self._global_metrics['active_conflicts'] = max(0, self._global_metrics['active_conflicts'] - 1)
        self._global_metrics['resolved_conflicts'] += 1
        
        # Mark conflict as resolved with timestamp
        if conflict_id in self._conflict_states:
            self._conflict_states[conflict_id]['resolved'] = True
            self._conflict_states[conflict_id]['resolution'] = result
            self._conflict_states[conflict_id]['last_updated'] = time.time()
        
        # Invalidate caches
        await self._invalidate_caches_for_conflict(conflict_id)
        
        return result
    
    async def process_scene(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a scene with parallel subsystem handling and transition tracking.
        """
        # Ensure first-call safety
        if not hasattr(self, "_last_scene"):
            self._last_scene = None
    
        # Convert to a lightweight scope signature for transition detection
        scope_sig = {
            'location': scene_context.get('location'),
            'npcs': scene_context.get('npcs', []) or [],
            'topics': scene_context.get('topics', []) or []
        }
    
        # Async transition notification if scope changed
        if self._last_scene != scope_sig:
            asyncio.create_task(self._handle_scene_transition(self._last_scene, scope_sig))
            self._last_scene = scope_sig
    
        # Determine which subsystems should be active (LLM-routed w/ fallback)
        active = await self._determine_active_subsystems(scene_context)
    
        # Build and emit routing event
        event = SystemEvent(
            event_id=f"scene_{datetime.now().timestamp()}",
            event_type=EventType.STATE_SYNC,
            source_subsystem=SubsystemType.SLICE_OF_LIFE,
            payload={"scene_context": scene_context},
            target_subsystems=active,
            requires_response=True,
            priority=5,
        )
    
        responses = await self.emit_event(event)
        if responses is None:
            logger.warning("Failed to emit scene processing event")
            return {
                'scene_id': scene_context.get('scene_id', 0),
                'processed': False,
                'error': 'Event emission failed'
            }
    
        return self._synthesize_scene_result(responses, scene_context)

    
    # ==================== SCENE BUNDLE METHODS ====================
    
    async def get_scene_bundle(self, scope: 'SceneScope') -> Dict[str, Any]:
        """
        Scene-scoped conflict bundle with stable cache key, normalized shape, and safe merges.
        Keeps your background fast-path and parallel subsystem fetches.
        """
        # First-call safety for transitions
        if not hasattr(self, "_last_scene"):
            self._last_scene = None
    
        # Stable cache key (prefer scope.to_cache_key; else canonical JSON)
        try:
            scene_key = scope.to_cache_key()
        except Exception:
            key_str = json.dumps({
                "loc": getattr(scope, "location_id", None),
                "npcs": sorted(list(getattr(scope, "npc_ids", []) or [])),
                "topics": sorted(list(getattr(scope, "topics", []) or [])),
                "lore": sorted(list(getattr(scope, "lore_tags", []) or [])),
                "window": getattr(scope, "time_window_hours", 24),
            }, sort_keys=True, default=str)
            scene_key = hashlib.md5(key_str.encode()).hexdigest()
    
        cache_key = f"scene:{self.user_id}:{self.conversation_id}:{scene_key}:conflicts"
    
        # Cache fast path + LRU touch
        async with self._bundle_lock:
            entry = self._bundle_cache.get(cache_key)
            if entry:
                bundle, ts = entry
                if time.time() - ts < self._bundle_ttl:
                    self._cache_hits += 1
                    if self._last_scene != scope:
                        asyncio.create_task(self._handle_scene_transition(self._last_scene, scope))
                        self._last_scene = scope
                    self._bundle_cache.move_to_end(cache_key)
                    return bundle
    
        self._cache_misses += 1
        start = time.time()
    
        # Base bundle
        bundle = {
            "conflicts": [],
            "active_tensions": {},
            "opportunities": [],
            "ambient_effects": [],
            "world_tension": 0.0,
            "canon": {},
            "timestamp": time.time(),
            "last_changed_at": 0.0,
        }
    
        # Background fast-path (guarded)
        background_subsystem = self._subsystems.get(SubsystemType.BACKGROUND)
        if background_subsystem and hasattr(background_subsystem, "manager"):
            try:
                scene_context = {
                    "location": getattr(scope, "location_id", None),
                    "npcs": getattr(scope, "npc_ids", []) or [],
                    "conversation_topics": getattr(scope, "topics", []) or [],
                }
                # Hard guard: skip call if everything is empty
                if scene_context["location"] or scene_context["npcs"] or scene_context["conversation_topics"]:
                    bg_context = await background_subsystem.manager.get_scene_context(scene_context)
                    bundle["conflicts"].extend(bg_context.get("active_conflicts", []) or [])
                    bundle["ambient_effects"].extend(bg_context.get("ambient_atmosphere", []) or [])
                    bundle["world_tension"] = float(bg_context.get("world_tension", 0.0) or 0.0)
                    bundle["last_changed_at"] = max(
                        bundle["last_changed_at"],
                        float(bg_context.get("last_changed_at", 0.0) or 0.0),
                    )
            except Exception as e:
                logger.debug(f"Background scene context failed (non-fatal): {e}")
    
        # Enrich from in-memory conflict states (cheap, scoped)
        try:
            conflicts_dict = await self._get_scene_relevant_conflicts(scope)
        except Exception as e:
            conflicts_dict = {}
            logger.debug(f"_get_scene_relevant_conflicts failed: {e}")
    
        active_from_state, latest_change = [], 0.0
        for cid, state in (conflicts_dict or {}).items():
            level = self._extract(state,
                                  ("intensity_level",),
                                  ("subsystem_responses", "tension", "level"),
                                  default=0.5) or 0.5
            stk = self._extract(state,
                                ("stakeholder_ids",),
                                ("subsystem_responses", "stakeholder", "ids"),
                                default=[]) or []
            active_from_state.append({
                "id": cid,
                "type": state.get("conflict_type"),
                "intensity": max(0.0, min(1.0, float(level))),
                "stakeholders": stk,
                "phase": state.get("phase") or (state.get("subsystem_responses", {})
                                                .get("phase", {}).get("phase")),
                "canonical": bool(self._is_canonical_conflict(state)),
            })
            latest_change = max(latest_change, float(state.get("last_updated", 0.0) or 0.0))
        if active_from_state:
            seen, merged = set(), []
            for item in (bundle["conflicts"] + active_from_state):
                cid = item.get("id")
                if cid is not None and cid in seen:
                    continue
                if cid is not None:
                    seen.add(cid)
                merged.append(item)
            bundle["conflicts"] = merged
            bundle["last_changed_at"] = max(bundle["last_changed_at"], latest_change)
    
        # Quick tensions / opportunities
        npc_ids = getattr(scope, "npc_ids", []) or []
        if npc_ids and not bundle["active_tensions"]:
            try:
                bundle["active_tensions"] = await self._get_npc_tensions(npc_ids) or {}
            except Exception as e:
                logger.debug(f"_get_npc_tensions failed: {e}")
    
        if not bundle["opportunities"]:
            try:
                bundle["opportunities"] = self._find_conflict_opportunities(scope, conflicts_dict)[:5]
            except Exception:
                bundle["opportunities"] = []
    
        # Parallel subsystem mini-bundles (non-background)
        tasks = []
        for subsystem_type, subsystem in self._subsystems.items():
            if subsystem_type == SubsystemType.BACKGROUND:
                continue
            if hasattr(subsystem, "get_scene_bundle"):
                tasks.append(self._get_subsystem_bundle_with_timeout(subsystem, scope, subsystem_type))
    
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict):
                    if "conflicts" in result:
                        bundle["conflicts"].extend(result.get("conflicts") or [])
                    if "active" in result:
                        bundle["conflicts"].extend(result.get("active") or [])
                    if "active_tensions" in result:
                        bundle["active_tensions"].update(result.get("active_tensions") or {})
                    if "tensions" in result:
                        bundle["active_tensions"].update(result.get("tensions") or {})
                    if "opportunities" in result:
                        bundle["opportunities"].extend(result.get("opportunities") or [])
                    if "ambient_effects" in result:
                        bundle["ambient_effects"].extend(result.get("ambient_effects") or [])
                    if "world_tension" in result:
                        try:
                            bundle["world_tension"] = max(bundle["world_tension"], float(result["world_tension"]))
                        except Exception:
                            pass
                    if "last_changed_at" in result:
                        try:
                            bundle["last_changed_at"] = max(bundle["last_changed_at"], float(result["last_changed_at"]))
                        except Exception:
                            pass
                # exceptions are ignored due to return_exceptions=True
    
        # Normalize shape expected by NyxContext
        bundle["active"] = bundle.get("conflicts", [])
        bundle["tensions"] = bundle.get("active_tensions", {})
        bundle["opportunities"] = (bundle.get("opportunities") or [])[:5]
    
        # Cache store + LRU cap
        async with self._bundle_lock:
            self._bundle_cache[cache_key] = (bundle, time.time())
            self._bundle_cache.move_to_end(cache_key)
            max_cache = int(os.getenv("CONFLICT_BUNDLE_CACHE_MAX", "256"))
            while len(self._bundle_cache) > max_cache:
                self._bundle_cache.popitem(last=False)
    
        # Perf metric
        self._performance_metrics["bundle_fetch_times"].append(time.time() - start)
        if len(self._performance_metrics["bundle_fetch_times"]) > 200:
            self._performance_metrics["bundle_fetch_times"].pop(0)
    
        # Transition hook
        if self._last_scene != scope:
            asyncio.create_task(self._handle_scene_transition(self._last_scene, scope))
            self._last_scene = scope
    
        return bundle

    
    async def get_scene_delta(self, scope: 'SceneScope', since_ts: float) -> Dict[str, Any]:
        """
        Get only conflicts that changed since timestamp.
        Scope-aware fast path for incremental updates.
        """
        relevant = await self._get_scene_relevant_conflicts(scope)
    
        latest_change = max(
            (float(s.get('last_updated', 0) or 0) for s in relevant.values()),
            default=0.0
        )
    
        if latest_change <= (since_ts or 0.0):
            return {
                'section': 'conflicts',
                'data': {},
                'canonical': False,
                'last_changed_at': latest_change,
                'priority': 1,
                'version': f"conflict_bundle_{time.time()}",
                'is_delta': True
            }
    
        bundle = await self.get_scene_bundle(scope)
        bundle['is_delta'] = True
        return bundle
    
    async def get_conflict_state(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Get state of a specific conflict (existing method for compatibility)"""
        return self._conflict_states.get(conflict_id)
    
    # ==================== HELPER METHODS ====================
    
    def _generate_cache_key(self, scope: 'SceneScope') -> str:
        """Generate stable cache key from scope (synchronous)"""
        # Try multiple methods to get a key
        key_fn = getattr(scope, "to_cache_key", None) or getattr(scope, "to_key", None)
        
        if callable(key_fn):
            scene_key = key_fn()
        else:
            # Fallback: create stable key from scope attributes
            key_str = json.dumps({
                "loc": getattr(scope, "location_id", None),
                "npcs": sorted(getattr(scope, "npc_ids", []) or []),
                "topics": sorted(getattr(scope, "topics", []) or []),
                "lore": sorted(getattr(scope, "lore_tags", []) or []),
                "window": getattr(scope, "time_window_hours", 24)
            }, sort_keys=True, default=str)
            scene_key = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{self.conversation_id}:{scene_key}:conflicts"
    
    def _extract(self, state: Dict[str, Any], *paths: Tuple[str, ...], default=None) -> Any:
        """Resilient field extraction from nested dicts with proper type hints"""
        for path in paths:
            cur = state
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok:
                return cur
        return default
    
    async def _get_scene_relevant_conflicts(self, scope: 'SceneScope') -> Dict[int, Dict]:
        """Get conflicts relevant to the scene scope with resilient field extraction"""
        # Early bail if scope is empty (with proper guards)
        npc_ids = getattr(scope, "npc_ids", []) or []
        location_id = getattr(scope, "location_id", None)
        topics = getattr(scope, "topics", []) or []
        
        if not npc_ids and not location_id and not topics:
            return {}
        
        relevant = {}
        
        # Snapshot conflict states to avoid RuntimeError if modified during iteration
        conflict_states_snapshot = list(self._conflict_states.items())
        
        # Filter by participants
        for conflict_id, state in conflict_states_snapshot:
            # Skip resolved conflicts
            if state.get('resolved'):
                continue
            
            # Extract stakeholder IDs from various possible locations
            stakeholder_ids = self._extract(
                state,
                ('stakeholder_ids',),
                ('subsystem_responses', 'stakeholder', 'ids'),
                default=[]
            )
            
            # Check if any scene NPCs are stakeholders
            if npc_ids and any(npc_id in stakeholder_ids for npc_id in npc_ids):
                relevant[conflict_id] = state
                continue
            
            # Extract location from various possible locations
            state_loc = self._extract(
                state,
                ('location_id',),
                ('subsystem_responses', 'canon', 'location_id'),
                default=None
            )
            
            # Check if conflict is at scene location
            if location_id and state_loc == location_id:
                relevant[conflict_id] = state
                continue
            
            # Extract tags from various possible locations
            conflict_tags = set(self._extract(
                state,
                ('tags',),
                ('subsystem_responses', 'canon', 'tags'),
                default=[]
            ) or [])
            
            # Check if conflict matches scene topics
            if topics and conflict_tags.intersection(topics):
                relevant[conflict_id] = state
        
        return relevant
    
    def _is_canonical_conflict(self, conflict_data: Dict) -> bool:
        """Determine if a conflict is canonical (high priority)"""
        # Canonical = main story conflicts, high stakes, or critical phase
        return any([
            conflict_data.get('is_main_story'),
            conflict_data.get('intensity_level', 0) > 0.7,
            conflict_data.get('phase') in ['climax', 'resolution'],
            conflict_data.get('canonical_event')
        ])
    
    async def _get_npc_tensions(self, npc_ids: List[int]) -> Dict[str, float]:
        """Get tension levels between NPCs (lightweight) with value clamping"""
        tensions = {}
        
        # Use existing tension subsystem if available
        if SubsystemType.TENSION in self._subsystems:
            tension_subsystem = self._subsystems[SubsystemType.TENSION]
            if hasattr(tension_subsystem, 'get_npc_tensions'):
                try:
                    raw_tensions = await tension_subsystem.get_npc_tensions(npc_ids)
                    # Format as "npc1_vs_npc2": level with clamping
                    for (npc1, npc2), level in raw_tensions.items():
                        if npc1 in npc_ids and npc2 in npc_ids:
                            key = f"{min(npc1, npc2)}_vs_{max(npc1, npc2)}"
                            # Clamp between 0.0 and 1.0 for safety
                            tensions[key] = max(0.0, min(1.0, float(level)))
                except Exception as e:
                    logger.error(f"Failed to get NPC tensions: {e}")
        
        return tensions
    
    async def _check_slice_active(self, scope: 'SceneScope') -> bool:
        """Check if slice-of-life mode is active for this scene"""
        slice_subsystem = self._subsystems.get(SubsystemType.SLICE_OF_LIFE)
        location_id = getattr(scope, "location_id", None)
        
        if slice_subsystem and location_id and hasattr(slice_subsystem, 'is_active_for_scene'):
            try:
                return await slice_subsystem.is_active_for_scene(location_id)
            except Exception:
                return False
        return False
    
    def _find_conflict_opportunities(self, scope: 'SceneScope', 
                                    conflicts: Dict[int, Dict]) -> List[Dict]:
        """Find emergent conflict opportunities based on scene context"""
        opportunities = []
        
        # Look for faction conflicts if multiple factions present
        factions = set()
        for conflict in conflicts.values():
            if 'faction_ids' in conflict:
                factions.update(conflict['faction_ids'])
        
        if len(factions) > 1:
            opportunities.append({
                'type': 'faction_tension',
                'description': 'Multiple factions present',
                'factions': list(factions)[:3]
            })
        
        # Look for relationship conflicts (with proper guard)
        npc_ids = getattr(scope, "npc_ids", []) or []
        if len(npc_ids) > 2:
            opportunities.append({
                'type': 'social_dynamics',
                'description': 'Multiple NPCs with history',
                'npcs': list(npc_ids)[:5]
            })
        
        # Check for location-based opportunities (with proper guard)
        location_id = getattr(scope, "location_id", None)
        topics = getattr(scope, "topics", None)
        if location_id and topics and 'tense_location' in topics:
            opportunities.append({
                'type': 'location_tension',
                'description': 'Historically tense location',
                'location': location_id
            })
        
        return opportunities[:3]  # Cap opportunities
    
    async def _invalidate_caches_for_conflict(self, conflict_id: int):
        """Invalidate all bundle cache entries after conflict change"""
        async with self._bundle_lock:
            if self._bundle_cache:
                cleared = len(self._bundle_cache)
                self._bundle_cache.clear()
                logger.debug(f"Invalidated {cleared} conflict bundle cache entries after conflict {conflict_id} change")
    
    async def _periodic_cache_cleanup(self):
        """Background task to clean expired bundle cache entries"""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = time.time()
                async with self._bundle_lock:
                    # Find expired entries
                    expired_keys = []
                    for key, (_, cached_time) in self._bundle_cache.items():
                        if current_time - cached_time > self._bundle_ttl:
                            expired_keys.append(key)
                    
                    # Remove expired entries
                    for key in expired_keys:
                        del self._bundle_cache[key]
                    
                    # Also enforce max size limit
                    while len(self._bundle_cache) > MAX_BUNDLE_CACHE:
                        oldest_key = next(iter(self._bundle_cache))
                        del self._bundle_cache[oldest_key]
                
                if expired_keys:
                    logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                    
            except Exception:
                logger.exception("Cache cleanup error")
    
    # ========== Performance Metrics ==========
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        cache_hit_rate = (
            self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        )
        
        # Calculate p95 times with better quantile behavior
        def calc_p95(times: List[float]) -> float:
            if not times:
                return 0.0
            sorted_times = sorted(times)
            idx = max(0, int(len(sorted_times) * 0.95) - 1)
            return sorted_times[idx]
        
        # Count pending background tasks
        pending_bg_tasks = sum(1 for t in self._bg_tasks.values() if not t.done())
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'bundle_cache_size': len(self._bundle_cache),
            'active_conflicts': self._global_metrics['active_conflicts'],
            'total_conflicts': self._global_metrics['total_conflicts'],
            'resolved_conflicts': self._global_metrics['resolved_conflicts'],
            'subsystems_active': len(self._subsystems),
            'event_queue_size': self._event_queue.qsize(),
            'bundle_fetch_p95': calc_p95(self._performance_metrics['bundle_fetch_times']),
            'parallel_process_p95': calc_p95(self._performance_metrics['parallel_process_times']),
            'cache_operations': self._performance_metrics['cache_operations'],
            'events_processed': self._performance_metrics['events_processed'],
            'pending_bg_tasks': pending_bg_tasks,
            'timeouts_count': self._performance_metrics['timeouts_count'],
            'failures_count': self._performance_metrics['failures_count'],
            'subsystem_timeouts': dict(self._performance_metrics['subsystem_timeouts']),
            'subsystem_failures': dict(self._performance_metrics['subsystem_failures'])
        }
    
    def reset_metrics(self):
        """Reset performance metrics for load testing"""
        self._cache_hits = 0
        self._cache_misses = 0
        self._performance_metrics = {
            'bundle_fetch_times': [],
            'parallel_process_times': [],
            'cache_operations': 0,
            'events_processed': 0,
            'timeouts_count': 0,
            'failures_count': 0,
            'subsystem_timeouts': defaultdict(int),
            'subsystem_failures': defaultdict(int)
        }
        logger.info("Performance metrics reset")
    
    # ========== Helper Methods ==========
    
    async def _determine_subsystems_for_operation(
        self,
        operation: str,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Set[SubsystemType]:
        """Determine which subsystems should handle an operation"""
        
        subsystems = set()
        
        # Always include edge handler for safety
        subsystems.add(SubsystemType.EDGE_HANDLER)
        
        # Based on actual conflict type
        if 'slice' in conflict_type.lower():
            subsystems.add(SubsystemType.SLICE_OF_LIFE)
        if 'social' in conflict_type.lower():
            subsystems.add(SubsystemType.SOCIAL)
        if 'background' in conflict_type.lower():
            subsystems.add(SubsystemType.BACKGROUND)
        if 'power' in conflict_type.lower() or 'political' in conflict_type.lower():
            subsystems.add(SubsystemType.LEVERAGE)
        
        # Check context for additional clues
        if context.get('is_multiparty'):
            subsystems.add(SubsystemType.SOCIAL)
            subsystems.add(SubsystemType.LEVERAGE)
        
        # Add core subsystems for all operations
        if operation in ['create', 'update']:
            subsystems.update([
                SubsystemType.TENSION,
                SubsystemType.STAKEHOLDER,
                SubsystemType.PHASE,
                SubsystemType.FLOW
            ])
        
        return subsystems
    
    async def _determine_required_subsystems(
        self,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Set[SubsystemType]:
        """Determine required subsystems for a conflict type (async safe)"""
        return await self._determine_subsystems_for_operation('create', conflict_type, context)
    
    async def _determine_active_subsystems(
        self,
        scene_context: Dict[str, Any]
    ) -> Set[SubsystemType]:
        """Determine active subsystems for scene processing with LLM timeout"""
        
        # Use LLM if orchestrator available
        if self._orchestrator:
            prompt = f"""
            Analyze this scene context and determine which conflict subsystems should be active:
            {json.dumps(scene_context, indent=2)}
            
            Available subsystems: {[s.value for s in SubsystemType]}
            
            Return a JSON list of subsystem names that should process this scene.
            """
            
            try:
                # Add timeout to LLM call
                response = await asyncio.wait_for(
                    Runner.run(self._orchestrator, prompt),
                    timeout=LLM_ROUTE_TIMEOUT
                )
                response_text = extract_runner_response(response)
                subsystem_names = json.loads(response_text)
                return {SubsystemType(name) for name in subsystem_names if name in SubsystemType._value2member_map_}
            except asyncio.TimeoutError:
                logger.warning(f"LLM routing timed out after {LLM_ROUTE_TIMEOUT}s, using fallback")
                self._performance_metrics['timeouts_count'] += 1
            except Exception:
                logger.exception("Failed to parse LLM response")
                self._performance_metrics['failures_count'] += 1
        
        # Fallback to default set
        return {
        SubsystemType.TENSION,
        SubsystemType.FLOW,
        SubsystemType.STAKEHOLDER,
        SubsystemType.SLICE_OF_LIFE,
        SubsystemType.EDGE_HANDLER,
        SubsystemType.LEVERAGE, # add this
        }
    
    def _aggregate_conflict_creation(self, responses: List[SubsystemResponse]) -> Dict[str, Any]:
        """Aggregate subsystem responses for conflict creation"""
        # Guard against empty responses
        if not responses:
            return {'status': 'created', 'subsystem_responses': {}}
        
        result = {
            'status': 'created',
            'subsystem_responses': {}
        }
        
        for response in responses:
            # Merge response data
            result['subsystem_responses'][response.subsystem.value] = response.data
            
            # Extract key fields
            if 'conflict_id' in response.data:
                result['conflict_id'] = response.data['conflict_id']
            if 'conflict_name' in response.data:
                result['conflict_name'] = response.data['conflict_name']
            if 'initial_phase' in response.data:
                result['initial_phase'] = response.data['initial_phase']
        
        return result
    
    def _aggregate_update_responses(self, responses: List[SubsystemResponse]) -> Dict[str, Any]:
        """Aggregate subsystem responses for conflict update"""
        # Guard against empty responses
        if not responses:
            return {'subsystem_updates': {}}
        
        result = {'subsystem_updates': {}}
        
        for response in responses:
            result['subsystem_updates'][response.subsystem.value] = response.data
            
            # Merge phase transitions
            if 'phase' in response.data:
                result['phase'] = response.data['phase']
        
        return result
    
    def _aggregate_resolution_responses(self, responses: List[SubsystemResponse]) -> Dict[str, Any]:
        """Aggregate subsystem responses for conflict resolution"""
        # Guard against empty responses
        if not responses:
            return {'resolved': True, 'resolution_details': {}}
        
        result = {
            'resolved': True,
            'resolution_details': {}
        }
        
        for response in responses:
            result['resolution_details'][response.subsystem.value] = response.data
        
        return result
    
    def _synthesize_scene_result(self, responses: List[SubsystemResponse], 
                                scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize scene processing results from subsystem responses"""
        result = {
            'scene_id': scene_context.get('scene_id', 0),
            'processed': True,
            'conflicts_active': [],
            'conflicts_detected': [],
            'subsystem_data': {}
        }
        
        # Guard against empty responses
        if not responses:
            return result
        
        for response in responses:
            # Aggregate subsystem data
            result['subsystem_data'][response.subsystem.value] = response.data
            
            # Extract active conflicts
            if 'active_conflicts' in response.data:
                result['conflicts_active'].extend(response.data['active_conflicts'])
            
            # Extract detected conflicts
            if 'detected_conflicts' in response.data:
                result['conflicts_detected'].extend(response.data['detected_conflicts'])
        
        return result
    
    # ========== Cleanup ==========
    
    async def shutdown(self):
        """Gracefully shutdown the synthesizer with proper task cleanup"""
        self._shutdown = True
        
        # Cancel all background tasks
        for name, task in self._bg_tasks.items():
            logger.info(f"Cancelling background task: {name}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(f"Error cancelling task {name}")
        
        # Wait for event processing to stop
        await asyncio.sleep(0.5)
        
        # Cleanup subsystems
        for subsystem in self._subsystems.values():
            if hasattr(subsystem, 'cleanup'):
                try:
                    await subsystem.cleanup()
                except Exception:
                    logger.exception(f"Error cleaning up subsystem {subsystem.subsystem_type}")
        
        # Clear caches
        async with self._bundle_lock:
            self._bundle_cache.clear()
        
        logger.info("Conflict synthesizer shutdown complete")


# ===============================================================================
# PUBLIC API
# ===============================================================================

# Global synthesizer instance cache (per user/conversation)
_synthesizers: Dict[tuple[int, int], ConflictSynthesizer] = {}

async def get_synthesizer(user_id: int, conversation_id: int) -> ConflictSynthesizer:
    key = (user_id, conversation_id)
    synth = _synthesizers.get(key)
    if not synth:
        synth = ConflictSynthesizer(user_id, conversation_id)
        await synth.initialize_all_subsystems()   # correct method name
        _synthesizers[key] = synth
        # optional day-transition hook
        try:
            from logic.time_cycle import register_day_transition_handler
            register_day_transition_handler(user_id, conversation_id, synth.handle_day_transition)
            logger.info("Registered synthesizer for day transitions")
        except ImportError:
            pass
    return synth


async def release_synthesizer(user_id: int, conversation_id: int):
    """Release and cleanup a synthesizer instance"""
    key = (user_id, conversation_id)
    synthesizer = _synthesizers.pop(key, None)
    if synthesizer:
        await synthesizer.shutdown()
        logger.info(f"Released synthesizer for user {user_id}, conversation {conversation_id}")


@function_tool
async def orchestrate_conflict_creation(
    ctx: RunContextWrapper,
    conflict_type: str,
    context_json: str,  # JSON string instead of object
) -> str:               # return JSON string
    """Create a conflict. Accepts JSON string, returns JSON string."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

    # Parse input JSON
    try:
        context: Dict[str, Any] = json.loads(context_json) if context_json else {}
    except Exception:
        context = {}

    result = await synthesizer.create_conflict(conflict_type, context)

    response_data = {
        'conflict_id': result.get('conflict_id', 0),
        'status': result.get('status', 'created'),
        'conflict_type': result.get('conflict_type', conflict_type),
        'conflict_name': result.get('conflict_name'),
        'message': result.get('message'),
        'created_at': result.get('created_at'),
        'initial_phase': result.get('initial_phase'),
        'pacing_style': result.get('pacing_style'),
        'stakeholders_created': result.get('stakeholders_created'),
        'stakeholder_ids': result.get('stakeholder_ids'),
        'template_used': result.get('template_used'),
        'generated_conflict': result.get('generated_conflict'),
        'narrative_hooks': result.get('narrative_hooks'),
        'conflict_details': result.get('conflict_details'),
        'subsystem_responses': result.get('subsystem_responses'),
    }
    payload = {k: v for k, v in response_data.items() if v is not None}
    return json.dumps(payload, ensure_ascii=False)


@function_tool
async def orchestrate_scene_processing(
    ctx: RunContextWrapper,
    scene_context_json: str,  # JSON string
) -> str:                     # JSON string
    """Process a scene. Accepts JSON string, returns JSON string."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

    try:
        scene_context: Dict[str, Any] = json.loads(scene_context_json) if scene_context_json else {}
    except Exception:
        scene_context = {}

    result = await synthesizer.process_scene(scene_context)

    response_data = {
        'scene_id': result.get('scene_id'),
        'processed': result.get('processed', True),
        'conflicts_active': result.get('conflicts_active'),
        'conflicts_detected': result.get('conflicts_detected'),
        'manifestations': result.get('manifestations'),
        'events_triggered': result.get('events_triggered'),
        'tensions_detected': result.get('tensions_detected'),
        'slice_of_life_active': result.get('slice_of_life_active'),
        'next_scene_suggestions': result.get('next_scene_suggestions'),
        'player_choices': result.get('player_choices'),
        'choices': result.get('choices'),
        'npc_behaviors': result.get('npc_behaviors'),
        'npc_reactions': result.get('npc_reactions'),
        'atmospheric_elements': result.get('atmospheric_elements'),
        'atmosphere': result.get('atmosphere'),
        'environmental_cues': result.get('environmental_cues'),
        'state_changes': result.get('state_changes'),
        'subsystem_data': result.get('subsystem_data'),
        'experience_quality': result.get('experience_quality'),
        'recommended_mode_change': result.get('recommended_mode_change'),
    }
    payload = {k: v for k, v in response_data.items() if v is not None}
    return json.dumps(payload, ensure_ascii=False)


@function_tool
async def orchestrate_conflict_resolution(
    ctx: RunContextWrapper,
    conflict_id: int,
    resolution_type: str,
    context_json: str,  # JSON string
) -> str:               # JSON string
    """Resolve a conflict. Accepts JSON string, returns JSON string."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

    try:
        context: Dict[str, Any] = json.loads(context_json) if context_json else {}
    except Exception:
        context = {}

    result = await synthesizer.resolve_conflict(conflict_id, resolution_type, context)

    response_data = {
        'conflict_id': conflict_id,
        'resolved': result.get('resolved', True),
        'resolution_type': resolution_type,
        'resolution_details': result.get('resolution_details'),
    }
    return json.dumps(response_data, ensure_ascii=False)


@function_tool
async def get_conflict_bundle(
    ctx: RunContextWrapper,
    scope_json: str,  # JSON string with scope fields
) -> str:             # JSON string
    """Get scene-scoped conflict bundle. Accepts JSON string, returns JSON string."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    try:
        scope_dict: Dict[str, Any] = json.loads(scope_json) if scope_json else {}
    except Exception:
        scope_dict = {}
    
    # Create a minimal scope object
    class MinimalScope:
        def __init__(self, data: Dict):
            self.location_id = data.get('location_id')
            self.npc_ids = data.get('npc_ids', [])
            self.topics = data.get('topics', [])
            self.lore_tags = data.get('lore_tags', [])
            self.conflict_ids = data.get('conflict_ids', [])
            self.time_window_hours = data.get('time_window_hours', 24)
            self.link_hints = data.get('link_hints', {})
        
        def to_cache_key(self) -> str:
            key_str = json.dumps({
                "loc": self.location_id,
                "npcs": sorted(self.npc_ids or []),
                "topics": sorted(self.topics or []),
                "lore": sorted(self.lore_tags or []),
                "window": self.time_window_hours
            }, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    scope = MinimalScope(scope_dict)
    bundle = await synthesizer.get_scene_bundle(scope)
    
    return json.dumps(bundle, ensure_ascii=False, default=str)


@function_tool
async def get_conflict_delta(
    ctx: RunContextWrapper,
    scope_json: str,  # JSON string with scope fields
    since_ts: float,   # Timestamp to check changes since
) -> str:             # JSON string
    """Get conflicts that changed since timestamp. Accepts JSON string, returns JSON string."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Validate and normalize since_ts
    try:
        since_ts = float(since_ts) if since_ts and since_ts > 0 else 0.0
    except (TypeError, ValueError):
        since_ts = 0.0
    
    try:
        scope_dict: Dict[str, Any] = json.loads(scope_json) if scope_json else {}
    except Exception:
        scope_dict = {}
    
    # Create a minimal scope object
    class MinimalScope:
        def __init__(self, data: Dict):
            self.location_id = data.get('location_id')
            self.npc_ids = data.get('npc_ids', [])
            self.topics = data.get('topics', [])
            self.lore_tags = data.get('lore_tags', [])
            self.conflict_ids = data.get('conflict_ids', [])
            self.time_window_hours = data.get('time_window_hours', 24)
            self.link_hints = data.get('link_hints', {})
        
        def to_cache_key(self) -> str:
            key_str = json.dumps({
                "loc": self.location_id,
                "npcs": sorted(self.npc_ids or []),
                "topics": sorted(self.topics or []),
                "lore": sorted(self.lore_tags or []),
                "window": self.time_window_hours
            }, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    scope = MinimalScope(scope_dict)
    delta = await synthesizer.get_scene_delta(scope, since_ts)
    
    return json.dumps(delta, ensure_ascii=False, default=str)


@function_tool
async def get_conflict_metrics(
    ctx: RunContextWrapper
) -> str:  # JSON string
    """Get performance metrics for the conflict synthesizer. Returns JSON string."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    metrics = await synthesizer.get_performance_metrics()
    
    # Add computed rates if we have data
    if metrics.get('events_processed', 0) > 0:
        events = metrics['events_processed']
        metrics['failure_rate'] = metrics.get('failures_count', 0) / events
        metrics['timeout_rate'] = metrics.get('timeouts_count', 0) / events
    
    return json.dumps(metrics, ensure_ascii=False, default=str)
