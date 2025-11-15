# logic/conflict_system/background_grand_conflicts.py
"""
COMPLETE Refactored Background Grand Conflicts System
Preserves ALL original functionality while optimizing for background processing.
"""

import logging
import json
import random
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

import asyncpg

from agents import Agent, function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from logic.conflict_system.background_processor import (
    get_conflict_scheduler,
    BackgroundConflictProcessor,
    ProcessingPriority,
    ConflictContentLimits,
    get_high_intensity_threshold,
    resolve_intensity_value,
)
from logic.conflict_system.background_grand_conflicts_hotpath import (
    fetch_conflict_event,
    fetch_conflict_news,
    fetch_conflict_opportunities,
    fetch_conflict_ripples,
    fetch_generated_conflict,
    queue_conflict_advance,
    queue_conflict_generation,
    queue_conflict_news,
    queue_conflict_opportunity_check,
    queue_conflict_ripples,
)

logger = logging.getLogger(__name__)

# ===============================================================================
# BACKGROUND CONFLICT TYPES
# ===============================================================================

class GrandConflictType(Enum):
    """Large-scale conflicts that happen in the background"""
    POLITICAL_SUCCESSION = "political_succession"
    ECONOMIC_CRISIS = "economic_crisis"
    FACTION_WAR = "faction_war"
    CULTURAL_REVOLUTION = "cultural_revolution"
    RESOURCE_SHORTAGE = "resource_shortage"
    TERRITORIAL_DISPUTE = "territorial_dispute"
    IDEOLOGICAL_SCHISM = "ideological_schism"
    TRADE_WAR = "trade_war"


class BackgroundIntensity(Enum):
    """How prominently the conflict features in daily life"""
    DISTANT_RUMOR = "distant_rumor"
    OCCASIONAL_NEWS = "occasional_news"
    REGULAR_TOPIC = "regular_topic"
    AMBIENT_TENSION = "ambient_tension"
    VISIBLE_EFFECTS = "visible_effects"


@dataclass
class BackgroundConflict:
    """A grand conflict happening in the background"""
    conflict_id: int
    conflict_type: GrandConflictType
    name: str
    description: str
    intensity: BackgroundIntensity
    progress: float
    factions: List[str]
    current_state: str
    recent_developments: List[str]
    impact_on_daily_life: List[str]
    player_awareness_level: float
    last_news_generation: Optional[int] = None
    news_count: int = 0


@dataclass
class WorldEvent:
    """An event in a background conflict"""
    conflict_id: int
    event_type: str
    description: str
    faction_impacts: Dict[str, float]
    creates_opportunity: bool
    opportunity_window: Optional[int]


class InitializeBackgroundWorldResponse(TypedDict):
    world_initialized: bool
    message: str

class DailyBackgroundFlavorResponse(TypedDict):
    world_tension: float
    background_news: List[str]
    ambient_effects: List[str]
    overheard: str
    optional_hook: str


INTENSITY_TO_FLOAT = {
    BackgroundIntensity.DISTANT_RUMOR: 0.2,
    BackgroundIntensity.OCCASIONAL_NEWS: 0.4,
    BackgroundIntensity.REGULAR_TOPIC: 0.6,
    BackgroundIntensity.AMBIENT_TENSION: 0.8,
    BackgroundIntensity.VISIBLE_EFFECTS: 1.0,
}

PROGRESS_MILESTONES = [
    (25.0, "emerging"),
    (50.0, "surging"),
    (75.0, "critical"),
    (95.0, "climax"),
]


async def _get_current_game_day(user_id: int, conversation_id: int) -> int:
    """Return the current in-game day stored in CurrentRoleplay.

    The helper queries the database directly for the ``CurrentDay`` entry so callers
    always receive a plain integer. When the lookup fails because of a database
    error, the function logs the failure and falls back to day ``1``. Missing or
    non-integer values also return ``1`` without additional logging to avoid noise.
    """

    try:
        async with get_db_connection_context() as conn:
            raw_value = await conn.fetchval(
                """
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentDay'
                """,
                user_id,
                conversation_id,
            )
    except (asyncio.TimeoutError, asyncpg.PostgresError) as exc:
        logger.warning(
            "Could not retrieve current game day from database, defaulting to 1. Error: %s",
            exc,
        )
        return 1

    try:
        return int(raw_value) if raw_value is not None else 1
    except (TypeError, ValueError):
        return 1


def adjust_intensity_value(current: float, change: str) -> float:
    """Adjust intensity value based on change direction"""
    if change == "increase":
        return min(1.0, current + 0.2)
    elif change == "decrease":
        return max(0.2, current - 0.2)
    return current


# ===============================================================================
# BACKGROUND CONFLICT ORCHESTRATOR
# ===============================================================================

class BackgroundConflictOrchestrator:
    """Orchestrates background conflict generation and evolution with LLM"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._conflict_generator = None
        self._evolution_agent = None
        self.processor = get_conflict_scheduler().get_processor(user_id, conversation_id)
        self.limits = ConflictContentLimits()

    def _payload_to_conflict(self, payload: Dict[str, Any]) -> BackgroundConflict:
        """Convert cached payload into a BackgroundConflict instance."""
        conflict_type_raw = payload.get("conflict_type", GrandConflictType.POLITICAL_SUCCESSION.value)
        try:
            conflict_type = GrandConflictType(conflict_type_raw)
        except ValueError:
            try:
                conflict_type = GrandConflictType[conflict_type_raw.upper()]
            except KeyError:
                conflict_type = GrandConflictType.POLITICAL_SUCCESSION

        intensity_raw = payload.get("intensity", BackgroundIntensity.DISTANT_RUMOR.value)
        try:
            intensity = BackgroundIntensity[intensity_raw.upper()]
        except KeyError:
            try:
                intensity = BackgroundIntensity(intensity_raw)
            except ValueError:
                intensity = BackgroundIntensity.DISTANT_RUMOR

        return BackgroundConflict(
            conflict_id=int(payload.get("conflict_id", 0)),
            conflict_type=conflict_type,
            name=payload.get("name", "Background Conflict"),
            description=payload.get("description", ""),
            intensity=intensity,
            progress=float(payload.get("progress", 0.0)),
            factions=list(payload.get("factions", [])),
            current_state=payload.get("current_state", ""),
            recent_developments=list(payload.get("recent_developments", [])),
            impact_on_daily_life=list(payload.get("impact_on_daily_life", [])),
            player_awareness_level=float(payload.get("player_awareness_level", 0.1)),
            last_news_generation=payload.get("last_news_generation"),
            news_count=int(payload.get("news_count", 0)),
        )

    def _payload_to_event(self, payload: Dict[str, Any]) -> WorldEvent:
        """Convert cached payload into a WorldEvent instance."""
        return WorldEvent(
            conflict_id=int(payload.get("conflict_id", 0)),
            event_type=payload.get("event_type", "development"),
            description=payload.get("description", ""),
            faction_impacts=payload.get("faction_impacts", {}),
            creates_opportunity=bool(payload.get("creates_opportunity", False)),
            opportunity_window=(
                int(payload.get("opportunity_window"))
                if payload.get("opportunity_window") is not None
                else None
            ),
        )

    @property
    def conflict_generator(self) -> Agent:
        """Lazy load conflict generator agent"""
        if self._conflict_generator is None:
            self._conflict_generator = Agent(
                name="Background Conflict Generator",
                instructions="""
                Generate grand-scale conflicts that happen in the background.
                
                Create conflicts that:
                - Feel massive in scope
                - Have multiple factions
                - Evolve over time
                - Impact daily life indirectly
                - Create atmospheric tension
                - Could involve the player optionally
                
                Make them feel real and consequential without dominating gameplay.
                """,
                model="gpt-5-nano",
            )
        return self._conflict_generator
    
    @property
    def evolution_agent(self) -> Agent:
        """Lazy load evolution agent"""
        if self._evolution_agent is None:
            self._evolution_agent = Agent(
                name="Conflict Evolution Agent",
                instructions="""
                Advance background conflicts based on their current state.
                
                Create developments that:
                - Feel like natural progressions
                - Reflect faction dynamics
                - Create ripple effects
                - Change intensity appropriately
                - Generate opportunities occasionally
                - Move toward resolution eventually
                
                Keep progression realistic and engaging.
                """,
                model="gpt-5-nano",
            )
        return self._evolution_agent
    
    async def generate_background_conflict(self) -> Optional[BackgroundConflict]:
        """Generate a new background conflict"""
        # First check if we need a new conflict
        async with get_db_connection_context() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM backgroundconflicts
                WHERE user_id = $1 AND conversation_id = $2 AND status = 'active'
                """,
                self.user_id,
                self.conversation_id,
            )

            if count >= 5:  # Max 5 active conflicts
                logger.info("Max active conflicts reached, skipping generation")
                return None

        cached = fetch_generated_conflict(self.user_id, self.conversation_id)
        if cached:
            return self._payload_to_conflict(cached)

        queue_conflict_generation(self.user_id, self.conversation_id)

        return None
    
    async def advance_background_conflict(self, conflict: BackgroundConflict) -> Optional[WorldEvent]:
        """Advance a background conflict's state"""
        cached_event = fetch_conflict_event(conflict.conflict_id)
        if cached_event:
            if cached_event.get("news_worthy"):
                queue_conflict_news(
                    self.user_id,
                    self.conversation_id,
                    conflict,
                    metadata={"reason": "significant_event"},
                )
            return self._payload_to_event(cached_event)

        # Check if we should actually advance (not every time)
        should_advance = random.random() < 0.3  # 30% chance
        if not should_advance:
            return None

        queue_conflict_advance(self.user_id, self.conversation_id, conflict)
        return None
    
# ===============================================================================
# BACKGROUND NEWS GENERATOR
# ===============================================================================

class BackgroundNewsGenerator:
    """Generates dynamic news and rumors with LLM - with limits"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._news_generator = None
        self.processor = get_conflict_scheduler().get_processor(user_id, conversation_id)
    
    @property
    def news_generator(self) -> Agent:
        if self._news_generator is None:
            self._news_generator = Agent(
                name="News Article Generator",
                instructions="""
                Generate news articles about background conflicts.
                
                Vary between:
                - Official announcements (formal, careful)
                - Independent reporting (balanced, investigative)
                - Tabloid coverage (sensational, dramatic)
                - Underground news (subversive, revealing)
                
                Match tone to source. Include bias and spin.
                """,
                model="gpt-5-nano",
            )
        return self._news_generator

    async def generate_news_item(
        self,
        conflict: BackgroundConflict,
        news_type: str = "random",
    ) -> Optional[Dict[str, Any]]:
        """Generate dynamic news about a conflict - respecting limits"""

        # Check if we should generate news
        should_gen, reason = await self.processor.should_generate_news(conflict.conflict_id)
        if not should_gen:
            logger.info(f"Skipping news generation for conflict {conflict.conflict_id}: {reason}")
            cached = fetch_conflict_news(conflict.conflict_id)
            if cached:
                return cached
            return None

        cached = fetch_conflict_news(conflict.conflict_id)
        if cached:
            return cached

        news_payload_type = None if news_type == "random" else news_type
        queue_conflict_news(
            self.user_id,
            self.conversation_id,
            conflict,
            news_type=news_payload_type,
            metadata={"reason": reason},
        )

        return None


# ===============================================================================
# BACKGROUND CONFLICT RIPPLES
# ===============================================================================

class BackgroundConflictRipples:
    """Manages how background conflicts affect daily life"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._ripple_generator = None
        self._opportunity_creator = None
        self.processor = get_conflict_scheduler().get_processor(user_id, conversation_id)
    
    @property
    def ripple_generator(self) -> Agent:
        if self._ripple_generator is None:
            self._ripple_generator = Agent(
                name="Ripple Effect Generator",
                instructions="""
                Generate subtle effects of grand conflicts on daily life.
                
                Create ripples that:
                - Affect atmosphere and mood
                - Change NPC behaviors subtly
                - Alter available resources
                - Create background tension
                - Suggest larger forces
                
                Keep effects indirect but noticeable to observant players.
                """,
                model="gpt-5-nano",
            )
        return self._ripple_generator
    
    @property
    def opportunity_creator(self) -> Agent:
        if self._opportunity_creator is None:
            self._opportunity_creator = Agent(
                name="Opportunity Creator",
                instructions="""
                Create optional opportunities from background conflicts.
                
                Generate opportunities that:
                - Are completely optional
                - Offer interesting choices
                - Connect to larger events
                - Have multiple approaches
                - Create memorable moments
                
                Players should feel these are bonuses, not obligations.
                """,
                model="gpt-5-nano",
            )
        return self._opportunity_creator

    async def generate_daily_ripples(
        self,
        active_conflicts: List[BackgroundConflict]
    ) -> Dict[str, Any]:
        """Generate today's ripple effects - limited to avoid spam"""

        # Only generate ripples occasionally, not every turn
        if random.random() > 0.3:  # 30% chance
            cached = fetch_conflict_ripples(self.user_id, self.conversation_id)
            if cached:
                return {"ripples": cached, "opportunities": []}
            return {"ripples": {}, "opportunities": []}

        if not active_conflicts:
            return {"ripples": {}, "opportunities": []}

        # Pick most intense conflict for ripples
        most_intense = max(active_conflicts, key=lambda c: INTENSITY_TO_FLOAT[c.intensity])
        cached = fetch_conflict_ripples(self.user_id, self.conversation_id)
        if cached:
            return {"ripples": cached, "opportunities": []}

        queue_conflict_ripples(self.user_id, self.conversation_id, most_intense)

        return {"ripples": {}, "opportunities": []}
    
    async def check_for_opportunities(
        self,
        active_conflicts: List[BackgroundConflict],
        player_skills: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if any conflicts create opportunities - rarely"""

        cached = fetch_conflict_opportunities(self.user_id, self.conversation_id)
        if cached:
            return cached

        # Very low chance of opportunities to avoid spam
        if random.random() > 0.1:  # 10% chance
            return []

        queue_conflict_opportunity_check(
            self.user_id,
            self.conversation_id,
            active_conflicts,
            player_skills=player_skills,
        )

        return []
    
    def _player_could_engage(self, player_skills: Dict, conflict: BackgroundConflict) -> bool:
        """Check if player has skills to engage with conflict"""
        # Simple check - can be expanded
        return len(player_skills) > 0 and conflict.player_awareness_level > 0.3


# ===============================================================================
# BACKGROUND CONFLICT SUBSYSTEM
# ===============================================================================

class BackgroundConflictSubsystem:
    """
    Implements the ConflictSubsystem interface with optimization.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Components
        self.orchestrator = BackgroundConflictOrchestrator(user_id, conversation_id)
        self.news_generator = BackgroundNewsGenerator(user_id, conversation_id)
        self.ripple_manager = BackgroundConflictRipples(user_id, conversation_id)
        self.processor = get_conflict_scheduler().get_processor(user_id, conversation_id)
        
        # Reference to synthesizer (set during initialization)
        self.synthesizer = None

        # Provide manager interface expected by ConflictSynthesizer fast-path
        self.manager = self
        
        # Caches
        self._context_cache = {}
        self._cache_ttl = 300  # 5 minutes

    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.BACKGROUND
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        return {
            'world_events',
            'grand_conflicts',
            'news_generation',
            'ripple_effects',
            'conversation_topics',
            'ambient_atmosphere'
        }
    
    @property
    def dependencies(self) -> Set:
        """Return other subsystems this depends on"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return {SubsystemType.CANON}  # Background events can become canonical
    
    @property
    def event_subscriptions(self) -> Set:
        """Return events this subsystem wants to receive"""
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.HEALTH_CHECK,
            EventType.STATE_SYNC,
            EventType.CONFLICT_CREATED,
            EventType.PHASE_TRANSITION,
            EventType.SCENE_ENTER,
            EventType.DAY_TRANSITION
         }

    
    async def get_scene_context(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a cached bundle for fast scene assembly without generating new LLM text."""

        current_day = await _get_current_game_day(self.user_id, self.conversation_id)
        bundle_key = self._scene_bundle_cache_key(current_day)
        now_ts = datetime.utcnow().timestamp()

        cached_bundle = self._context_cache.get(bundle_key)
        if cached_bundle and now_ts - cached_bundle['timestamp'] < self._cache_ttl:
            return dict(cached_bundle['data'])

        bundle = await self._build_scene_context_bundle(current_day=current_day)
        self._context_cache[bundle_key] = {'timestamp': now_ts, 'data': bundle}
        return dict(bundle)
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        
        # Initialize background world if needed
        active_conflicts = await self._get_active_background_conflicts()
        if len(active_conflicts) < 2:
            # Queue initial generation for background
            self.processor._processing_queue.append({
                'type': 'generate_initial_conflicts',
                'priority': ProcessingPriority.HIGH
            })
        
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""
        from logic.conflict_system.conflict_synthesizer import EventType, SubsystemResponse, SystemEvent
    
        try:
            if event.event_type == EventType.HEALTH_CHECK:
                health = await self.health_check()
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=health,
                    side_effects=[]
                )
    
            elif event.event_type == EventType.STATE_SYNC:
                # Don't generate new content, just return cached/existing
                daily_update = await self.daily_background_update(generate_new=False)
    
                # Only create news events if we have actual news
                side_effects: List[SystemEvent] = []
                if daily_update.get("news"):
                    for news in daily_update.get("news", [])[:1]:  # Max 1 news event
                        side_effects.append(
                            SystemEvent(
                                event_id=f"news_{event.event_id}",
                                event_type=EventType.STATE_SYNC,
                                source_subsystem=self.subsystem_type,
                                payload={"news": news},
                                priority=8,  # Low priority
                            )
                        )
    
                active_conflict_ids = daily_update.get('active_conflict_ids', []) or []
                active_conflicts = daily_update.get('active_conflicts', []) or []

                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        'background_update': daily_update,
                        'news_count': len(daily_update.get("news", [])),
                        'active_conflicts': active_conflicts,
                        'active_conflict_ids': active_conflict_ids,
                        'active_conflict_count': len(active_conflict_ids),
                    },
                    side_effects=side_effects
                )
    
            elif event.event_type == EventType.CONFLICT_CREATED:
                # Handle new conflict creation (only if flagged as background)
                payload = event.payload or {}
                if payload.get('is_background', False):
                    conflict = await self.orchestrator.generate_background_conflict()
                    if conflict:
                        return SubsystemResponse(
                            subsystem=self.subsystem_type,
                            event_id=event.event_id,
                            success=True,
                            data={'conflict': conflict.__dict__},
                            side_effects=[]
                        )
                    else:
                        return SubsystemResponse(
                            subsystem=self.subsystem_type,
                            event_id=event.event_id,
                            success=False,
                            data={'error': 'Max active conflicts reached or generation skipped'},
                            side_effects=[]
                        )
                # If not a background conflict, we don't handle it here
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=False,
                    data={'error': 'Event not relevant for background subsystem'},
                    side_effects=[]
                )
    
            elif event.event_type == EventType.PHASE_TRANSITION:
                # Handle conflict phase transitions
                payload = event.payload or {}
                conflict_id = payload.get('conflict_id')
                if conflict_id:
                    conflicts = await self._get_active_background_conflicts()
                    for conf_data in conflicts:
                        if conf_data['id'] == conflict_id:
                            conflict = self._db_to_background_conflict(conf_data)
                            adv_event = await self.orchestrator.advance_background_conflict(conflict)
                            if adv_event:
                                return SubsystemResponse(
                                    subsystem=self.subsystem_type,
                                    event_id=event.event_id,
                                    success=True,
                                    data={'event': adv_event.__dict__},
                                    side_effects=[]
                                )
                    # Conflict not found or no advancement occurred
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=False,
                        data={'error': 'Conflict not found or no advancement occurred'},
                        side_effects=[]
                    )
    
            elif event.event_type == EventType.SCENE_ENTER:
                scene_ctx = (event.payload or {}).get('scene_context', {}) or {}
                relevant = await self.is_relevant_to_scene(scene_ctx)
                # Light scene context; do not force generation
                bg = await self.get_scene_context(scene_ctx) if relevant else {
                    'active_conflicts': [],
                    'ambient_atmosphere': [],
                    'world_tension': 0.0,
                    'last_changed_at': datetime.utcnow().timestamp()
                }
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'scene_context': bg},
                    side_effects=[]
                )
    
            elif event.event_type == EventType.DAY_TRANSITION:
                payload, side_effects = await self._on_day_transition(event)
                try:
                    asyncio.create_task(self.processor.process_queued_items(max_items=5))
                except Exception:
                    logger.debug("Background processor tick failed to schedule", exc_info=True)
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=payload,
                    side_effects=side_effects,
                )
    
            # Default: not handled
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'Event not handled'},
                side_effects=[]
            )
    
        except Exception as e:
            logger.error(f"Error handling event {event.event_type}: {e}", exc_info=True)
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check subsystem health"""
        try:
            active = await self._get_active_background_conflicts()
            active_conflicts = [
                self._db_to_background_conflict(conflict_data) for conflict_data in active
            ]
            summaries = [self._summarize_conflict(conflict) for conflict in active_conflicts]
            return {
                'status': 'healthy',
                'active_conflicts': summaries,
                'active_conflict_count': len(summaries),
                'queue_size': len(self.processor._processing_queue),
                'cache_size': len(self._context_cache)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'active_conflicts': [],
                'active_conflict_count': 0
            }
    
    async def daily_background_update(self, generate_new: bool = False) -> Dict[str, Any]:
        """Return the latest background bundle without triggering new generation."""

        current_day = await _get_current_game_day(self.user_id, self.conversation_id)
        cache_key = self._daily_update_cache_key(current_day)
        now_ts = datetime.utcnow().timestamp()

        cached = self._context_cache.get(cache_key)
        if cached and not generate_new and now_ts - cached['timestamp'] < self._cache_ttl:
            return dict(cached['data'])

        bundle_key = self._scene_bundle_cache_key(current_day)
        bundle_entry = self._context_cache.get(bundle_key)
        if bundle_entry and not generate_new and now_ts - bundle_entry['timestamp'] < self._cache_ttl:
            bundle = dict(bundle_entry['data'])
        else:
            bundle = await self._build_scene_context_bundle(current_day=current_day)
            self._context_cache[bundle_key] = {'timestamp': now_ts, 'data': bundle}

        daily_bundle = self._daily_bundle_from_scene(bundle)
        self._context_cache[cache_key] = {'timestamp': now_ts, 'data': daily_bundle}
        return dict(daily_bundle)

    async def _on_day_transition(self, event) -> Tuple[Dict[str, Any], List[Any]]:
        """Advance background conflicts and refresh cached bundles for a new day."""

        from logic.conflict_system.conflict_synthesizer import EventType, SystemEvent

        payload = event.payload or {}
        tick_payload = payload.get('tick') if isinstance(payload, dict) else None
        if isinstance(tick_payload, dict):
            payload = tick_payload

        day_value = payload.get('new_day') or payload.get('day')
        month_value = payload.get('month')
        year_value = payload.get('year')

        current_day: Optional[int] = None
        if isinstance(day_value, dict):
            try:
                current_day = int(day_value.get('day'))
            except (TypeError, ValueError):
                current_day = None
            month_value = month_value or day_value.get('month')
            year_value = year_value or day_value.get('year')
        elif hasattr(day_value, 'day'):
            try:
                current_day = int(day_value.day)
                month_value = month_value or getattr(day_value, 'month', None)
                year_value = year_value or getattr(day_value, 'year', None)
            except Exception:
                current_day = None
        elif day_value is not None:
            try:
                current_day = int(day_value)
            except (TypeError, ValueError):
                current_day = None

        if current_day is None:
            current_day = await _get_current_game_day(self.user_id, self.conversation_id)

        conflicts_data = await self._get_active_background_conflicts()
        if not conflicts_data:
            bundle = await self._build_scene_context_bundle(current_day=current_day)
            now_ts = datetime.utcnow().timestamp()
            daily_bundle = self._daily_bundle_from_scene(bundle)
            self._context_cache[self._scene_bundle_cache_key(current_day)] = {
                'timestamp': now_ts,
                'data': bundle,
            }
            self._context_cache[self._daily_update_cache_key(current_day)] = {
                'timestamp': now_ts,
                'data': daily_bundle,
            }
            payload_data = {
                'day': current_day,
                'daily_bundle': daily_bundle,
                'world_tension': bundle.get('world_tension', 0.0),
                'threshold_crossings': [],
                'scheduled_news': [],
                'scheduled_ripples': [],
            }
            if month_value is not None:
                payload_data['month'] = month_value
            if year_value is not None:
                payload_data['year'] = year_value
            return payload_data, []

        high_threshold = get_high_intensity_threshold()
        updated_records: List[Dict[str, Any]] = []
        threshold_crossings: List[int] = []
        milestone_events: List[Dict[str, Any]] = []
        news_queued: List[int] = []
        ripple_scheduled: List[int] = []

        for record in conflicts_data:
            conflict = self._db_to_background_conflict(record)
            metadata = json.loads(record.get('metadata', '{}') or '{}')

            previous_progress = float(record.get('progress') or metadata.get('last_progress', conflict.progress))
            previous_intensity_value = resolve_intensity_value(record.get('intensity'))

            progress_delta = self._compute_progress_delta(previous_intensity_value, metadata)
            new_progress = max(0.0, min(100.0, previous_progress + progress_delta))

            conflict.progress = new_progress
            new_intensity_enum = self._intensity_from_progress(new_progress)
            conflict.intensity = new_intensity_enum
            new_intensity_value = float(INTENSITY_TO_FLOAT[new_intensity_enum])

            crossed_high = previous_intensity_value < high_threshold <= new_intensity_value
            milestone = self._detect_progress_milestone(previous_progress, new_progress)

            metadata.update({
                'last_progress': new_progress,
                'last_update_day': current_day,
                'last_intensity_value': new_intensity_value,
                'last_intensity_label': new_intensity_enum.value,
            })

            updated_records.append({
                'conflict': conflict,
                'metadata': metadata,
                'new_intensity_value': new_intensity_value,
                'crossed_high': crossed_high,
                'milestone': milestone,
            })

            if crossed_high:
                threshold_crossings.append(conflict.conflict_id)

            if milestone:
                milestone_events.append({
                    'conflict_id': conflict.conflict_id,
                    'conflict': conflict.name,
                    'milestone': milestone,
                    'progress': round(new_progress, 2),
                    'intensity': new_intensity_enum.value,
                })

        for record in updated_records:
            conflict = record['conflict']
            metadata = record['metadata']

            if (record['crossed_high'] or record['milestone'] in {'critical', 'climax'}) and metadata.get('last_threshold_analysis_day') != current_day:
                analysis = await self._run_threshold_analysis(
                    conflict,
                    metadata,
                    current_day=current_day,
                )
                if analysis:
                    metadata['threshold_analysis'] = analysis
                    metadata['last_threshold_analysis_day'] = current_day

            if record['crossed_high']:
                self.processor.queue_ripple_generation(
                    conflict,
                    priority=ProcessingPriority.HIGH,
                    reason='intensity_threshold',
                )
                metadata['last_ripple_day'] = current_day
                ripple_scheduled.append(conflict.conflict_id)

            if (record['crossed_high'] or record['milestone'] in {'surging', 'critical', 'climax'}) and metadata.get('last_news_day') != current_day:
                try:
                    should_news, reason = await self.processor.should_generate_news(conflict.conflict_id)
                except Exception:
                    should_news, reason = (False, None)

                if should_news:
                    self.processor.queue_news_generation(
                        conflict.conflict_id,
                        priority=ProcessingPriority.HIGH if record['crossed_high'] else ProcessingPriority.NORMAL,
                        reason=reason or 'threshold_crossed',
                    )
                    metadata['last_news_day'] = current_day
                    news_queued.append(conflict.conflict_id)

        update_rows = [
            (
                record['conflict'].progress,
                record['new_intensity_value'],
                json.dumps(record['metadata']),
                record['conflict'].conflict_id,
                self.user_id,
                self.conversation_id,
            )
            for record in updated_records
        ]

        if update_rows:
            async with get_db_connection_context() as conn:
                await conn.executemany(
                    """
                    UPDATE BackgroundConflicts
                    SET progress = $1,
                        intensity = $2,
                        metadata = COALESCE($3::jsonb, '{}'::jsonb),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $4 AND user_id = $5 AND conversation_id = $6
                    """,
                    update_rows,
                )

        updated_conflicts = [record['conflict'] for record in updated_records]
        bundle = await self._build_scene_context_bundle(
            current_day=current_day,
            conflicts=updated_conflicts,
        )
        daily_bundle = self._daily_bundle_from_scene(bundle)
        daily_bundle['events_today'] = milestone_events
        daily_bundle['news'] = list(daily_bundle.get('news', []))
        daily_bundle['ripple_effects'] = bundle.get('ripple_effects', {'ripples': {}})
        daily_bundle.setdefault('optional_opportunities', [])

        now_ts = datetime.utcnow().timestamp()
        self._context_cache[self._scene_bundle_cache_key(current_day)] = {
            'timestamp': now_ts,
            'data': bundle,
        }
        self._context_cache[self._daily_update_cache_key(current_day)] = {
            'timestamp': now_ts,
            'data': daily_bundle,
        }

        ripple_key = self._ripple_cache_key(current_day)
        if ripple_key not in self._context_cache:
            self._context_cache[ripple_key] = {
                'timestamp': now_ts,
                'data': {'ripples': {}},
            }

        payload_data = {
            'day': current_day,
            'daily_bundle': daily_bundle,
            'world_tension': bundle.get('world_tension', 0.0),
            'threshold_crossings': threshold_crossings,
            'scheduled_news': news_queued,
            'scheduled_ripples': ripple_scheduled,
            'milestones': milestone_events,
        }
        if month_value is not None:
            payload_data['month'] = month_value
        if year_value is not None:
            payload_data['year'] = year_value

        side_effects: List[SystemEvent] = []
        for milestone in milestone_events:
            side_effects.append(
                SystemEvent(
                    event_id=f"background_conflict_{milestone['conflict_id']}_{current_day}",
                    event_type=EventType.CONFLICT_UPDATED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'conflict_id': milestone['conflict_id'],
                        'milestone': milestone['milestone'],
                        'progress': milestone['progress'],
                        'intensity': milestone['intensity'],
                    },
                    priority=6,
                )
            )

        return payload_data, side_effects
    
    async def get_conversation_topics(self) -> List[str]:
        """Get background conflict topics for NPC conversations"""
        current_day = await _get_current_game_day(self.user_id, self.conversation_id)
        async with get_db_connection_context() as conn:
            recent_news = await conn.fetch(
                """
                SELECT headline FROM backgroundnews
                WHERE user_id = $1 AND conversation_id = $2
                  AND game_day > $3 - 7
                ORDER BY game_day DESC, id DESC
                LIMIT 5
                """,
                self.user_id,
                self.conversation_id,
                current_day
            )
        
        return [f"Did you hear about {news['headline']}?" for news in recent_news]
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        # Background is always somewhat relevant for atmosphere
        location_raw = scene_context.get('location', '')
        location = str(location_raw or '')
    
        public_locations = ['market', 'tavern', 'plaza', 'court']
        if any(loc in location.lower() for loc in public_locations):
            return True
    
        conflicts = await self._get_active_background_conflicts()
        threshold = get_high_intensity_threshold()
        for conflict in conflicts:
            if resolve_intensity_value(conflict.get('intensity')) >= threshold:
                return True
    
        return random.random() < 0.3

    def _scene_bundle_cache_key(self, day: int) -> str:
        return f"scene_bundle_{int(day)}"

    def _daily_update_cache_key(self, day: int) -> str:
        return f"daily_update_{int(day)}"

    def _ripple_cache_key(self, day: int) -> str:
        return f"daily_ripples_{int(day)}"

    async def _build_scene_context_bundle(
        self,
        *,
        current_day: int,
        conflicts: Optional[List[BackgroundConflict]] = None,
    ) -> Dict[str, Any]:
        if conflicts is None:
            conflicts_data = await self._get_active_background_conflicts()
            conflicts = [self._db_to_background_conflict(data) for data in conflicts_data]

        summaries = [self._summarize_conflict(conflict) for conflict in conflicts]
        conflict_ids = [conflict.conflict_id for conflict in conflicts]

        news_items = await self._fetch_recent_news(limit=5)
        ripple_effects = self._get_cached_ripples(current_day)

        ambient = ripple_effects.get('ripples', {}).get('ambient_mood') or []
        if not isinstance(ambient, list):
            ambient = [str(ambient)] if ambient else []
        if not ambient:
            for conflict in conflicts:
                for impact in conflict.impact_on_daily_life[:2]:
                    if not impact:
                        continue
                    ambient.append(str(impact))
                    if len(ambient) >= 5:
                        break
                if len(ambient) >= 5:
                    break

        tension_values: List[float] = []
        for conflict in conflicts:
            try:
                tension_values.append(float(INTENSITY_TO_FLOAT[conflict.intensity]))
            except KeyError:
                tension_values.append(0.4)

        world_tension = 0.0
        if tension_values:
            world_tension = round(sum(tension_values) / len(tension_values), 3)

        bundle = {
            'background_conflicts': summaries,
            'conflict_ids': conflict_ids,
            'ambient_atmosphere': ambient[:5],
            'world_tension': world_tension,
            'news': news_items,
            'ripple_effects': ripple_effects,
            'last_changed_at': datetime.utcnow().timestamp(),
            'last_updated_day': current_day,
        }
        return bundle

    def _daily_bundle_from_scene(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'active_conflicts': list(bundle.get('background_conflicts', [])),
            'active_conflict_ids': list(bundle.get('conflict_ids', [])),
            'events_today': [],
            'news': list(bundle.get('news', [])),
            'ripple_effects': dict(bundle.get('ripple_effects', {'ripples': {}})),
            'optional_opportunities': [],
            'world_tension': float(bundle.get('world_tension', 0.0)),
        }

    def _compute_progress_delta(self, intensity_value: float, metadata: Dict[str, Any]) -> float:
        momentum = metadata.get('momentum', metadata.get('momentum_score', 0.5))
        volatility = metadata.get('volatility', 0.3)
        trend = str(metadata.get('trend', metadata.get('direction', 'steady'))).lower()

        try:
            momentum = float(momentum)
        except (TypeError, ValueError):
            momentum = 0.5
        momentum = max(0.0, min(1.0, momentum))

        try:
            volatility = float(volatility)
        except (TypeError, ValueError):
            volatility = 0.3
        volatility = max(0.0, min(1.0, volatility))

        base = max(0.4, intensity_value * 10.0)
        base *= 0.8 + momentum * 0.5
        base += volatility * 2.0

        if trend in {'escalating', 'rising', 'surging', 'boiling'}:
            base *= 1.2
        elif trend in {'cooling', 'declining', 'stabilizing', 'subsiding'}:
            base *= 0.75

        return max(0.5, min(base, 12.0))

    def _intensity_from_progress(self, progress: float) -> BackgroundIntensity:
        normalized = max(0.0, min(progress, 100.0)) / 100.0
        if normalized >= 0.85:
            return BackgroundIntensity.VISIBLE_EFFECTS
        if normalized >= 0.65:
            return BackgroundIntensity.AMBIENT_TENSION
        if normalized >= 0.45:
            return BackgroundIntensity.REGULAR_TOPIC
        if normalized >= 0.25:
            return BackgroundIntensity.OCCASIONAL_NEWS
        return BackgroundIntensity.DISTANT_RUMOR

    def _detect_progress_milestone(self, previous: float, current: float) -> Optional[str]:
        for threshold, label in PROGRESS_MILESTONES:
            if previous < threshold <= current:
                return label
        return None

    async def _run_threshold_analysis(
        self,
        conflict: BackgroundConflict,
        metadata: Dict[str, Any],
        *,
        current_day: int,
    ) -> Optional[Dict[str, Any]]:
        try:
            from lore.utils.llm_gateway import build_llm_request
            from nyx.gateway import llm_gateway
            from nyx.config import WARMUP_MODEL
        except Exception:
            logger.debug("Threshold analysis skipped due to missing dependencies", exc_info=True)
            return None

        context_payload = {
            'conflict': {
                'name': conflict.name,
                'type': conflict.conflict_type.value,
                'progress': round(conflict.progress, 2),
                'intensity': conflict.intensity.value,
                'current_state': conflict.current_state,
                'factions': conflict.factions[:3],
            },
            'metadata': {
                'momentum': metadata.get('momentum'),
                'volatility': metadata.get('volatility'),
                'trend': metadata.get('trend'),
                'last_analysis': metadata.get('threshold_analysis'),
            },
            'day': current_day,
        }

        prompt = (
            "Analyze the background conflict state and return JSON with keys "
            "`risk_tier`, `recommended_ripple`, and `escalation_outlook`."
        )

        try:
            request = build_llm_request(
                self.orchestrator.evolution_agent,
                prompt,
                context=context_payload,
                model_override=WARMUP_MODEL,
                run_config={'response_format': 'json_object'},
            )
            result = await llm_gateway.execute(request)
            raw_text = (result.text or '').strip()
            if not raw_text:
                return None
            return json.loads(raw_text)
        except Exception:
            logger.debug("Background conflict threshold analysis failed", exc_info=True)
            return None

    async def _fetch_recent_news(self, *, limit: int = 5) -> List[Dict[str, Any]]:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                """
                SELECT headline, source, game_day
                FROM backgroundnews
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY game_day DESC, id DESC
                LIMIT $3
                """,
                self.user_id,
                self.conversation_id,
                limit,
            )
        return [dict(row) for row in rows]

    def _get_cached_ripples(self, current_day: int) -> Dict[str, Any]:
        ripple_key = self._ripple_cache_key(current_day)
        cached = self._context_cache.get(ripple_key)
        if cached and datetime.utcnow().timestamp() - cached['timestamp'] < 3600:
            return dict(cached['data'])
        return {'ripples': {}}
    
    async def _get_active_background_conflicts(self) -> List[Dict[str, Any]]:
        """Get active background conflicts from database, including recent developments."""
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch(
                """
                WITH RankedDevelopments AS (
                    SELECT
                        conflict_id,
                        development,
                        ROW_NUMBER() OVER(PARTITION BY conflict_id ORDER BY game_day DESC, id DESC) as rn
                    FROM backgrounddevelopments
                    WHERE conflict_id IN (
                        SELECT id FROM backgroundconflicts
                        WHERE user_id = $1 AND conversation_id = $2 AND status = 'active'
                    )
                )
                SELECT
                    bc.*,
                    COALESCE(
                        (SELECT jsonb_agg(rd.development ORDER BY rd.rn DESC)
                         FROM RankedDevelopments rd
                         WHERE rd.conflict_id = bc.id AND rd.rn <= 3),
                        '[]'::jsonb
                    ) as recent_developments
                FROM backgroundconflicts bc
                WHERE bc.user_id = $1 AND bc.conversation_id = $2
                AND bc.status = 'active'
                ORDER BY bc.updated_at DESC
                """,
                self.user_id,
                self.conversation_id
            )
        return [dict(c) for c in conflicts]
    
    def _db_to_background_conflict(self, data: Dict[str, Any]) -> BackgroundConflict:
        """Convert database row to BackgroundConflict object"""
        metadata = json.loads(data.get('metadata', '{}'))
        
        # <<< FIX IS HERE: Robustly handle intensity value >>>
        intensity_value = data.get('intensity')
        intensity_enum = BackgroundIntensity.DISTANT_RUMOR  # Default value

        if isinstance(intensity_value, str):
            try:
                intensity_enum = BackgroundIntensity[intensity_value.upper()]
            except KeyError:
                logger.warning(f"Invalid intensity string '{intensity_value}' in DB, defaulting to DISTANT_RUMOR.")
        
        elif isinstance(intensity_value, (int, float)):
            # Convert float back to the closest enum
            if intensity_value <= 0.2:
                intensity_enum = BackgroundIntensity.DISTANT_RUMOR
            elif intensity_value <= 0.4:
                intensity_enum = BackgroundIntensity.OCCASIONAL_NEWS
            elif intensity_value <= 0.6:
                intensity_enum = BackgroundIntensity.REGULAR_TOPIC
            elif intensity_value <= 0.8:
                intensity_enum = BackgroundIntensity.AMBIENT_TENSION
            else:
                intensity_enum = BackgroundIntensity.VISIBLE_EFFECTS
        
        return BackgroundConflict(
            conflict_id=data['id'],
            conflict_type=GrandConflictType[data['conflict_type'].upper()],
            name=data['name'],
            description=data['description'],
            intensity=intensity_enum, # Use the resolved enum value
            progress=float(data.get('progress', 0.0)),
            factions=json.loads(data['factions']),
            current_state=data['current_state'],
            recent_developments=data.get('recent_developments', []),
            impact_on_daily_life=metadata.get('daily_life_impacts', []),
            player_awareness_level=float(data.get('awareness', 0.1)),
            last_news_generation=data.get('last_news_generation'),
            news_count=data.get('news_count', 0)
        )

    def _summarize_conflict(self, conflict: BackgroundConflict) -> Dict[str, Any]:
        """Create a lightweight summary for downstream consumers."""
        try:
            intensity_level = float(INTENSITY_TO_FLOAT[conflict.intensity])
        except KeyError:
            intensity_level = 0.4

        return {
            'id': conflict.conflict_id,
            'type': 'background',
            'name': conflict.name,
            'intensity': conflict.intensity.value,
            'intensity_level': intensity_level,
            'progress': conflict.progress,
        }


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def initialize_background_world(
    ctx: RunContextWrapper
) -> InitializeBackgroundWorldResponse:
    """Initialize the background world with grand conflicts"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get synthesizer
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # The subsystem will handle initialization
    from logic.conflict_system.conflict_synthesizer import SubsystemType
    subsystem = getattr(synthesizer, "_subsystems", {}).get(SubsystemType.BACKGROUND)
    
    if subsystem:
        # Queue initial conflict generation if needed
        await subsystem.initialize(synthesizer)
        
        return {
            'world_initialized': True,
            'message': 'Background world initialized with optimized processing',
        }
    
    return {
        'world_initialized': False,
        'message': 'Failed to initialize background world'
    }


@function_tool
async def get_daily_background_flavor(
    ctx: RunContextWrapper
) -> DailyBackgroundFlavorResponse:
    """Get today's background world flavor for atmospheric text"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    event = SystemEvent(
        event_id=f"flavor_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={'request': 'daily_flavor'},
        target_subsystems={SubsystemType.BACKGROUND},
        requires_response=True,
    )
    
    responses = await synthesizer.emit_event(event)
    
    if responses:
        for response in responses:
            if response.subsystem == SubsystemType.BACKGROUND:
                update = response.data.get('background_update', {}) or {}
                
                world_tension = float(update.get('world_tension', 0.0))
                
                news_items = update.get('news', []) or []
                news_headlines = [
                    item.get('headline', str(item)) 
                    for item in news_items if item
                ]

                ripple_effects = update.get('ripple_effects', {}) or {}
                ripples = ripple_effects.get('ripples', {}) or {}
                ambient_effects = ripples.get('ambient_mood', []) or []
                if not isinstance(ambient_effects, list):
                    ambient_effects = [str(ambient_effects)] if ambient_effects else []

                overheard_snippets = ripples.get('overheard_snippets', []) or []
                overheard = overheard_snippets[0] if overheard_snippets else ""
                
                opportunities = update.get('optional_opportunities', []) or []
                optional_hook = ""
                if opportunities:
                    optional_hook = opportunities[0].get('title', '')
                
                return {
                    'world_tension': world_tension,
                    'background_news': news_headlines,
                    'ambient_effects': ambient_effects,
                    'overheard': overheard,
                    'optional_hook': optional_hook
                }
    
    # Fallback
    return {
        'world_tension': 0.0,
        'background_news': [],
        'ambient_effects': [],
        'overheard': "",
        'optional_hook': ""
    }
