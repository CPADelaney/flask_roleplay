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

from logic.conflict_system.dynamic_conflict_template import extract_runner_response

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context
from logic.conflict_system.background_processor import (
    get_conflict_scheduler,
    BackgroundConflictProcessor,
    ProcessingPriority,
    ConflictContentLimits,
    get_high_intensity_threshold,
    resolve_intensity_value,
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
                self.user_id, self.conversation_id
            )
            
            if count >= 5:  # Max 5 active conflicts
                logger.info("Max active conflicts reached, skipping generation")
                return None
        
        conflict_type = random.choice(list(GrandConflictType))
        
        prompt = f"""
        Generate a {conflict_type.value} background conflict.
        
        Consider:
        - World setting and established lore
        - Current game state
        - Player's position in the world
        - Other active conflicts
        
        Create something that feels:
        - Important but distant
        - Complex with multiple sides
        - Slowly evolving
        - Atmospheric
        
        Return JSON:
        {{
            "name": "Conflict name",
            "description": "2-3 sentence overview",
            "factions": ["Faction 1", "Faction 2", "Faction 3"],
            "initial_state": "Current situation",
            "initial_development": "What just happened to start/escalate this",
            "daily_life_impacts": [
                "How it affects common people",
                "What changes in daily routines",
                "Ambient signs of the conflict"
            ],
            "potential_hooks": ["Optional player involvement 1", "Optional involvement 2"],
            "initial_intensity": "distant_rumor|occasional_news|regular_topic",
            "estimated_duration": 10-50 (game days)
        }}
        """
        
        response = await Runner.run(self.conflict_generator, prompt)
        data = json.loads(extract_runner_response(response))

        # Convert the string intensity (e.g., "distant_rumor") from the LLM into a float
        intensity_float = resolve_intensity_value(data.get("initial_intensity"))
        
        # Store in database
        async with get_db_connection_context() as conn:
            conflict_id = await conn.fetchval(
                """
                INSERT INTO backgroundconflicts
                (user_id, conversation_id, conflict_type, name, description,
                 factions, current_state, intensity, progress, status,
                 metadata, news_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
                """,
                self.user_id,
                self.conversation_id,
                conflict_type.value,
                data["name"],
                data["description"],
                json.dumps(data["factions"]),
                data["initial_state"],
                # OLD, INCORRECT LINE: data.get("initial_intensity", "distant_rumor"),
                intensity_float,  # <-- USE THE CONVERTED FLOAT HERE
                0.0,  # Initial progress
                'active',
                json.dumps({
                    "potential_hooks": data.get("potential_hooks", []),
                    "estimated_duration": data.get("estimated_duration", 30),
                    "created_at": datetime.utcnow().isoformat(),
                    "daily_life_impacts": data.get("daily_life_impacts", [])
                }),
                0  # news_count
            )
            
            # Store initial development
            await conn.execute(
                """
                INSERT INTO backgrounddevelopments
                (conflict_id, development, game_day)
                VALUES ($1, $2, $3)
                """,
                conflict_id,
                data["initial_development"],
                await _get_current_game_day(self.user_id, self.conversation_id)
            )
        
        conflict = BackgroundConflict(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            name=data["name"],
            description=data["description"],
            intensity=BackgroundIntensity[data.get("initial_intensity", "distant_rumor").upper()],
            progress=0.0,
            factions=data["factions"],
            current_state=data["initial_state"],
            recent_developments=[data["initial_development"]],
            impact_on_daily_life=data.get("daily_life_impacts", []),
            player_awareness_level=0.1,
            news_count=0
        )
        
        # Queue initial news generation if appropriate
        self.processor._processing_queue.append({
            'type': 'generate_news',
            'conflict_id': conflict_id,
            'priority': ProcessingPriority.NORMAL,
            'reason': 'initial_news'
        })
        
        return conflict
    
    async def advance_background_conflict(self, conflict: BackgroundConflict) -> Optional[WorldEvent]:
        """Advance a background conflict's state"""
        # Check if we should actually advance (not every time)
        should_advance = random.random() < 0.3  # 30% chance
        if not should_advance:
            return None
            
        prompt = f"""
        Advance this background conflict:
        
        Conflict: {conflict.name}
        Type: {conflict.conflict_type.value}
        Current State: {conflict.current_state}
        Progress: {conflict.progress}%
        Intensity: {conflict.intensity.value}
        Factions: {json.dumps(conflict.factions)}
        Recent: {conflict.recent_developments[-1] if conflict.recent_developments else 'Beginning'}
        
        Generate the next development:
        - Natural progression from current state
        - Reflects faction dynamics
        - Appropriate to intensity level
        - Moves story forward
        
        Return JSON:
        {{
            "event_type": "battle|negotiation|escalation|development|revelation",
            "description": "What happened (2-3 sentences)",
            "new_state": "Updated conflict state",
            "progress_change": -10 to +10,
            "intensity_change": "increase|maintain|decrease",
            "faction_impacts": {{"Faction": impact_score}},
            "creates_opportunity": true/false,
            "opportunity_description": "Optional: what player could do",
            "ripple_effects": ["Effect 1", "Effect 2"],
            "news_worthy": true/false
        }}
        """
        
        response = await Runner.run(self.evolution_agent, prompt)
        data = json.loads(extract_runner_response(response))
        
        # Update conflict in database
        new_intensity = self._calculate_new_intensity(
            conflict.intensity,
            data.get("intensity_change", "maintain")
        )
        
        new_progress = max(0, min(100, conflict.progress + data.get("progress_change", 0)))
        
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE backgroundconflicts
                SET current_state = $1,
                    progress = $2,
                    intensity = $3,
                    updated_at = CURRENT_TIMESTAMP,
                    last_significant_change = CASE 
                        WHEN $4 THEN $5
                        ELSE last_significant_change
                    END
                WHERE id = $6
                """,
                data.get("new_state", conflict.current_state),
                new_progress,
                new_intensity,
                data.get("news_worthy", False),
                json.dumps({
                    'timestamp': datetime.utcnow().isoformat(),
                    'magnitude': abs(data.get("progress_change", 0)) / 10.0,
                    'description': data.get("description", "")
                }),
                conflict.conflict_id
            )
            
            # Store development
            await conn.execute(
                """
                INSERT INTO backgrounddevelopments
                (conflict_id, development, game_day)
                VALUES ($1, $2, $3)
                """,
                conflict.conflict_id,
                data["description"],
                await _get_current_game_day(self.user_id, self.conversation_id)
            )
            
            # Handle opportunity creation
            if data.get("creates_opportunity"):
                await conn.execute(
                    """
                    INSERT INTO conflictopportunities
                    (conflict_id, description, expires_on, status,
                     user_id, conversation_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    conflict.conflict_id,
                    data.get("opportunity_description", "An opportunity arises"),
                    (await _get_current_game_day(self.user_id, self.conversation_id)) + 7,
                    'available',
                    self.user_id,
                    self.conversation_id
                )
        
        # Queue news generation if newsworthy and allowed
        if data.get("news_worthy"):
            should_gen, reason = await self.processor.should_generate_news(conflict.conflict_id)
            if should_gen:
                self.processor._processing_queue.append({
                    'type': 'generate_news',
                    'conflict_id': conflict.conflict_id,
                    'priority': ProcessingPriority.HIGH,
                    'reason': 'significant_event',
                    'event_data': data
                })
        
        return WorldEvent(
            conflict_id=conflict.conflict_id,
            event_type=data.get("event_type", "development"),
            description=data["description"],
            faction_impacts=data.get("faction_impacts", {}),
            creates_opportunity=data.get("creates_opportunity", False),
            opportunity_window=7 if data.get("creates_opportunity") else None
        )
    
    def _calculate_new_intensity(self, current: BackgroundIntensity, change: str) -> str:
        """Calculate new intensity level"""
        intensities = list(BackgroundIntensity)
        current_idx = intensities.index(current)
        
        if change == "increase" and current_idx < len(intensities) - 1:
            return intensities[current_idx + 1].value
        elif change == "decrease" and current_idx > 0:
            return intensities[current_idx - 1].value
        return current.value


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
            return None
        
        if news_type == "random":
            news_type = random.choice(["official", "independent", "tabloid", "rumor"])
        
        prompt = f"""
        Generate {news_type} news about this conflict:
        
        Conflict: {conflict.name}
        Current State: {conflict.current_state}
        Recent Development: {conflict.recent_developments[-1] if conflict.recent_developments else 'Initial stages'}
        Factions: {json.dumps(conflict.factions)}
        
        Create news that:
        - Matches {news_type} style perfectly
        - Feels authentic to source
        - Includes appropriate bias
        - Creates atmosphere
        - Could influence opinions
        
        Return JSON:
        {{
            "headline": "Attention-grabbing headline",
            "source": "News source name",
            "content": "2-3 paragraph article/rumor",
            "reliability": 0.0 to 1.0,
            "bias": "faction or perspective bias",
            "spin": "how truth is distorted",
            "public_reaction": "How people might react",
            "conversation_starter": "How NPCs might discuss this",
            "hidden_truth": "What's really happening"
        }}
        """
        
        response = await Runner.run(self.news_generator, prompt)
        news_data = json.loads(extract_runner_response(response))
        
        current_day = await _get_current_game_day(self.user_id, self.conversation_id)
        
        # Store in DB
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO backgroundnews
                (user_id, conversation_id, conflict_id, headline,
                 source, content, reliability, game_day)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                self.user_id,
                self.conversation_id,
                conflict.conflict_id,
                news_data["headline"],
                news_data["source"],
                news_data["content"],
                news_data["reliability"],
                current_day
            )
            
            # Update conflict news tracking
            await conn.execute(
                """
                UPDATE backgroundconflicts
                SET news_count = news_count + 1,
                    last_news_generation = $1
                WHERE id = $2
                """,
                current_day,
                conflict.conflict_id
            )
        
        return news_data


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
            return {"ripples": {}, "opportunities": []}
        
        if not active_conflicts:
            return {"ripples": {}, "opportunities": []}
        
        # Pick most intense conflict for ripples
        most_intense = max(active_conflicts, key=lambda c: INTENSITY_TO_FLOAT[c.intensity])
        
        prompt = f"""
        Generate subtle ripple effects from this conflict:
        
        Conflict: {most_intense.name}
        Intensity: {most_intense.intensity.value}
        Current State: {most_intense.current_state}
        
        Create effects that:
        - Match the intensity level
        - Feel atmospheric not intrusive
        - Could be noticed or ignored
        - Add texture to the world
        
        Return JSON:
        {{
            "ambient_mood": ["Mood descriptor 1", "Mood descriptor 2"],
            "npc_behaviors": ["Behavioral change 1", "Behavioral change 2"],
            "resource_changes": {{"Resource": "change description"}},
            "overheard_snippets": ["Snippet 1", "Snippet 2"],
            "environmental_details": ["Detail 1", "Detail 2"]
        }}
        """
        
        response = await Runner.run(self.ripple_generator, prompt)
        ripple_data = json.loads(extract_runner_response(response))
        
        # Store significant ripples for later use
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO conflictripples
                (conflict_id, ripple_data, game_day, user_id, conversation_id)
                VALUES ($1, $2, $3, $4, $5)
                """,
                most_intense.conflict_id,
                json.dumps(ripple_data),
                await _get_current_game_day(self.user_id, self.conversation_id),
                self.user_id,
                self.conversation_id
            )
        
        return {"ripples": ripple_data, "opportunities": []}
    
    async def check_for_opportunities(
        self,
        active_conflicts: List[BackgroundConflict],
        player_skills: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if any conflicts create opportunities - rarely"""
        
        # Very low chance of opportunities to avoid spam
        if random.random() > 0.1:  # 10% chance
            return []
        
        opportunities = []
        
        for conflict in active_conflicts:
            # Only high-intensity conflicts create opportunities
            if conflict.intensity not in [BackgroundIntensity.AMBIENT_TENSION, 
                                         BackgroundIntensity.VISIBLE_EFFECTS]:
                continue
            
            # Check if player has relevant skills
            if self._player_could_engage(player_skills, conflict):
                prompt = f"""
                Create an optional opportunity from this conflict:
                
                Conflict: {conflict.name}
                State: {conflict.current_state}
                Player Skills: {json.dumps(player_skills)}
                
                Generate opportunity that:
                - Is completely optional
                - Has multiple approaches
                - Offers meaningful choice
                - Could be ignored without penalty
                
                Return JSON:
                {{
                    "title": "Opportunity title",
                    "description": "What's available",
                    "approaches": ["Approach 1", "Approach 2"],
                    "potential_rewards": ["Reward 1", "Reward 2"],
                    "potential_risks": ["Risk 1", "Risk 2"],
                    "time_sensitive": true/false,
                    "skill_requirements": {{"Skill": level}}
                }}
                """
                
                response = await Runner.run(self.opportunity_creator, prompt)
                opp_data = json.loads(extract_runner_response(response))
                opp_data['conflict_id'] = conflict.conflict_id
                opportunities.append(opp_data)
                
                # Only one opportunity per check
                break
        
        return opportunities
    
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
        """
        Fast-path for ConflictSynthesizer.get_scene_bundle().
        Returns:
          {
            'active_conflicts': [ { 'id': int, 'type': 'background', 'intensity': float, 'name': str } ],
            'ambient_atmosphere': [str, ...],
            'world_tension': float,
            'last_changed_at': float
          }
        """
        update = await self.daily_background_update(generate_new=False)
        # Build ambient list from ripples
        ripples = ((update.get('ripple_effects') or {}).get('ripples') or {})
        ambient = ripples.get('ambient_mood') or []
        if not isinstance(ambient, list):
            ambient = [str(ambient)]
        
        # Represent active conflicts lightly for the bundle
        active_conflicts_list = []
        conflicts = await self._get_active_background_conflicts()
        for c in conflicts[:5]:
            try:
                intensity_enum = BackgroundIntensity[c['intensity'].upper()]
                intensity_level = float(INTENSITY_TO_FLOAT[intensity_enum])
            except Exception:
                intensity_level = 0.4
            active_conflicts_list.append({
                'id': c['id'],
                'type': 'background',
                'name': c.get('name', 'Background Conflict'),
                'intensity': intensity_level
            })
        
        return {
            'active_conflicts': active_conflicts_list,
            'ambient_atmosphere': ambient,
            'world_tension': float(update.get('world_tension') or 0.0),
            'last_changed_at': datetime.utcnow().timestamp()
        }
    
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
                # Advance background world; permit generation
                await self.daily_background_update(generate_new=True)
                # Optionally let processor tick asynchronously (kept lightweight)
                try:
                    asyncio.create_task(self.processor.process_queued_items(max_items=5))
                except Exception:
                    pass
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'day_transition': True},
                    side_effects=[]
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
        """Daily update of all background conflicts - optionally generate new content"""
        
        current_day = await _get_current_game_day(self.user_id, self.conversation_id)
        cache_key = f"daily_update_{current_day}"
        if cache_key in self._context_cache and not generate_new:
            cached = self._context_cache[cache_key]
            if datetime.utcnow().timestamp() - cached['timestamp'] < self._cache_ttl:
                return cached['data']
        
        # Get active background conflicts
        conflicts_data = await self._get_active_background_conflicts()
        
        active_conflicts = []
        active_conflict_summaries = []
        for conflict_data in conflicts_data:
            conflict = self._db_to_background_conflict(conflict_data)
            active_conflicts.append(conflict)
            active_conflict_summaries.append(self._summarize_conflict(conflict))

        active_conflict_ids = [conflict.conflict_id for conflict in active_conflicts]

        # Only generate new conflicts if needed and allowed
        if generate_new and len(active_conflicts) < 3:
            new_conflict = await self.orchestrator.generate_background_conflict()
            if new_conflict:
                active_conflicts.append(new_conflict)
                active_conflict_summaries.append(self._summarize_conflict(new_conflict))
                active_conflict_ids.append(new_conflict.conflict_id)
        
        # Process conflict advances if generating new content
        events = []
        if generate_new:
            for conflict in active_conflicts:
                if random.random() < 0.3:  # 30% chance each day
                    event = await self.orchestrator.advance_background_conflict(conflict)
                    if event:
                        events.append({
                            'conflict': conflict.name,
                            'event': event.description
                        })
        
        # Get existing news (don't generate new unless explicitly in background)
        news_items = []
        async with get_db_connection_context() as conn:
            recent_news = await conn.fetch(
                """
                SELECT headline, source FROM backgroundnews
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY game_day DESC
                LIMIT 3
                """,
                self.user_id, self.conversation_id
            )
            news_items = [dict(n) for n in recent_news]
        
        # Get cached ripple effects (normalized object with 'ripples' key)
        ripples_obj = {'ripples': {}}
        if active_conflicts:
            # Use cached ripples if available
            ripple_key = f"daily_ripples_{current_day}"
            if ripple_key in self._context_cache:
                cached_ripple = self._context_cache[ripple_key]
                if datetime.utcnow().timestamp() - cached_ripple['timestamp'] < 3600:
                    ripples_obj = cached_ripple['data']
            elif generate_new: # Only generate new if not in cache and permitted
                ripple_result = await self.ripple_manager.generate_daily_ripples(active_conflicts)
                ripples = ripple_result.get('ripples', {})
                ripples_obj = {'ripples': ripples}
                self._context_cache[ripple_key] = {
                    'timestamp': datetime.utcnow().timestamp(),
                    'data': ripples_obj
                }
        
        # Check for opportunities (rarely)
        opportunities = []
        if generate_new and random.random() < 0.1:  # 10% chance
            opportunities = await self.ripple_manager.check_for_opportunities(
                active_conflicts, {}  # Would pass actual player skills
            )
        
        result = {
            'active_conflicts': active_conflict_summaries,
            'active_conflict_ids': active_conflict_ids,
            'events_today': events,
            'news': news_items,
            'ripple_effects': ripples_obj,
            'optional_opportunities': opportunities,
            'world_tension': sum(c.progress for c in active_conflicts) / (len(active_conflicts) * 100) if active_conflicts else 0
        }
        
        # Cache the result
        self._context_cache[cache_key] = {
            'timestamp': datetime.utcnow().timestamp(),
            'data': result
        }
        
        return result
    
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
        return BackgroundConflict(
            conflict_id=data['id'],
            conflict_type=GrandConflictType[data['conflict_type'].upper()],
            name=data['name'],
            description=data['description'],
            intensity=BackgroundIntensity[data['intensity'].upper()],
            progress=float(data.get('progress', 0.0)),
            factions=json.loads(data['factions']),
            current_state=data['current_state'],
            recent_developments=data.get('recent_developments', []), # This is now a list from the DB
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
