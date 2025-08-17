# logic/conflict_system/background_grand_conflicts.py
"""
Background Grand Conflicts System with LLM-generated content
Integrated with ConflictSynthesizer as the central orchestrator
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from logic.conflict_system.dynamic_conflict_template import extract_runner_response

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context
from logic.time_cycle import get_current_game_day

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
    background_news: List[str]          # always a list (possibly empty)
    ambient_effects: List[str]          # always a list (possibly empty)
    overheard: str                      # empty string if none
    optional_hook: str                  # empty string if none




# ===============================================================================
# BACKGROUND CONFLICT SUBSYSTEM (Integrated with Synthesizer)
# ===============================================================================

class BackgroundConflictSubsystem:
    """
    Background conflict subsystem that integrates with ConflictSynthesizer.
    Implements the ConflictSubsystem interface.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Components
        self.orchestrator = BackgroundConflictOrchestrator(user_id, conversation_id)
        self.news_generator = BackgroundNewsGenerator(user_id, conversation_id)
        self.ripple_manager = BackgroundConflictRipples(user_id, conversation_id)
        
        # Reference to synthesizer (set during initialization)
        self.synthesizer = None
    
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
            EventType.PHASE_TRANSITION
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        
        # Initialize background world if needed
        active_conflicts = await self._get_active_background_conflicts()
        if len(active_conflicts) < 2:
            # Generate initial background conflicts
            await self.orchestrator.generate_background_conflict()
        
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
                )
    
            elif event.event_type == EventType.STATE_SYNC:
                # Generate daily background update
                daily_update = await self.daily_background_update()
    
                # Generate side effects (news items as events)
                side_effects = []
                for news in daily_update.get("news", [])[:1]:
                    side_effects.append(
                        SystemEvent(
                            event_id=f"news_{event.event_id}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=self.subsystem_type,
                            payload={"news": news},
                            priority=8,  # Low priority
                        )
                    )
    
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        "background_update": daily_update,
                        "world_tension": daily_update.get("world_tension", 0),
                        "manifestation": daily_update.get("ripple_effects", {}),
                    },
                    side_effects=side_effects,
                )
    
            elif event.event_type == EventType.CONFLICT_CREATED:
                # High intensity conflicts might affect background
                if event.payload.get("intensity") == "confrontation":
                    await self._adjust_background_tensions(0.1)
    
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={"background_adjusted": True},
                )
    
            elif event.event_type == EventType.PHASE_TRANSITION:
                # Background conflicts might advance when main conflicts transition
                if random.random() < 0.3:  # 30% chance
                    conflict = await self._get_random_background_conflict()
                    if conflict:
                        world_evt = await self.orchestrator.advance_background_conflict(conflict)
                        return SubsystemResponse(
                            subsystem=self.subsystem_type,
                            event_id=event.event_id,  # keep original SystemEvent id
                            success=True,
                            data={"background_advanced": True, "event": world_evt.description},
                        )
    
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={},
            )
    
        except Exception as e:
            logger.error(f"Background subsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=getattr(event, "event_id", "unknown"),
                success=False,
                data={"error": str(e)},
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        active_conflicts = await self._get_active_background_conflicts()
        
        health_status = {
            'healthy': len(active_conflicts) < 10,
            'active_background_conflicts': len(active_conflicts),
            'issue': None
        }
        
        if len(active_conflicts) >= 10:
            health_status['issue'] = 'Too many background conflicts'
        elif len(active_conflicts) == 0:
            health_status['issue'] = 'No background conflicts active'
        
        return health_status
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get background-related data for a specific conflict"""
        # Get conversation topics related to this conflict
        topics = await self.get_conversation_topics()
        
        # Get any ripple effects
        active_conflicts = await self._get_active_background_conflicts()
        ripples = await self.ripple_manager.generate_daily_ripples(
            [self._db_to_background_conflict(c) for c in active_conflicts]
        )
        
        return {
            'conversation_topics': topics[:3],
            'ripple_effects': ripples.get('ripples', {})
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of background system"""
        update = await self.daily_background_update()
        return {
            'world_tension': update.get('world_tension', 0),
            'active_background_conflicts': update.get('active_conflicts', 0),
            'recent_events': update.get('events_today', []),
            'optional_opportunities': update.get('optional_opportunities', [])
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if background system is relevant to scene"""
        # Background is always somewhat relevant for atmosphere
        # But more relevant in certain contexts
        location = scene_context.get('location', '')
        
        # More relevant in public spaces where news spreads
        public_locations = ['market', 'tavern', 'plaza', 'court']
        if any(loc in location.lower() for loc in public_locations):
            return True
        
        # Always at least minimally relevant
        return random.random() < 0.3
    
    # ========== Daily Update System ==========
    
    async def daily_background_update(self) -> Dict[str, Any]:
        """Daily update of all background conflicts"""
        
        # Get active background conflicts
        conflicts_data = await self._get_active_background_conflicts()
        
        active_conflicts = []
        for conflict_data in conflicts_data:
            active_conflicts.append(self._db_to_background_conflict(conflict_data))
        
        # Generate new conflicts if needed
        if len(active_conflicts) < 3:
            new_conflict = await self.orchestrator.generate_background_conflict()
            active_conflicts.append(new_conflict)
        
        # Advance conflicts probabilistically
        events = []
        for conflict in active_conflicts:
            if random.random() < 0.3:  # 30% chance each day
                event = await self.orchestrator.advance_background_conflict(conflict)
                events.append({
                    'conflict': conflict.name,
                    'event': event.description
                })
        
        # Generate news items
        news_items = []
        for conflict in active_conflicts:
            if random.random() < 0.5:  # 50% chance of news
                news = await self.news_generator.generate_news_item(conflict)
                news_items.append(news)
        
        # Generate ripple effects
        ripples = await self.ripple_manager.generate_daily_ripples(active_conflicts)
        
        # Check for opportunities
        opportunities = await self.ripple_manager.check_for_opportunities(
            active_conflicts, {}  # Would pass actual player skills
        )
        
        return {
            'active_conflicts': len(active_conflicts),
            'events_today': events,
            'news': news_items[:3],  # Limit to 3 news items
            'ripple_effects': ripples,
            'optional_opportunities': opportunities,
            'world_tension': sum(c.progress for c in active_conflicts) / (len(active_conflicts) * 100) if active_conflicts else 0
        }
    
    async def get_conversation_topics(self) -> List[str]:
        """Get background conflict topics for NPC conversations"""
        async with get_db_connection_context() as conn:
            recent_news = await conn.fetch(
                """
                SELECT headline, content FROM BackgroundNews
                WHERE user_id = $1 AND conversation_id = $2
                  AND game_day > (
                    SELECT CAST(value AS INTEGER) FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentDay'
                  ) - 7
                ORDER BY game_day DESC
                LIMIT 10
                """,
                self.user_id,
                self.conversation_id,
            )
    
        topics: List[str] = []
        for news in recent_news:
            headline = news["headline"]
            content = news["content"] or ""
            topics.append(f"Did you hear about {headline}?")
            topics.append(content[:100] + "...")
        return topics

    
    # ========== Helper Methods ==========
    
    async def _get_active_background_conflicts(self):
        """Get active background conflicts from database"""
        async with get_db_connection_context() as conn:
            return await conn.fetch("""
                SELECT * FROM BackgroundConflicts
                WHERE user_id = $1 AND conversation_id = $2 AND is_active = true
            """, self.user_id, self.conversation_id)
    
    async def _get_random_background_conflict(self) -> Optional[BackgroundConflict]:
        """Get a random active background conflict"""
        conflicts = await self._get_active_background_conflicts()
        if conflicts:
            conflict_data = random.choice(conflicts)
            return self._db_to_background_conflict(conflict_data)
        return None
    
    def _db_to_background_conflict(self, db_row) -> BackgroundConflict:
        """Convert database row to BackgroundConflict object safely"""
        row = dict(db_row)  # asyncpg.Record -> dict
    
        # factions may be absent or stored elsewhere; normalize to list
        raw_factions = row.get("factions")
        factions: List[str]
        if isinstance(raw_factions, list):
            factions = raw_factions
        elif isinstance(raw_factions, (str, bytes)):
            try:
                parsed = json.loads(raw_factions)
                factions = parsed if isinstance(parsed, list) else []
            except Exception:
                factions = []
        else:
            factions = []
    
        return BackgroundConflict(
            conflict_id=row["conflict_id"],
            conflict_type=GrandConflictType(row["conflict_type"]),
            name=row["name"],
            description=row["description"],
            intensity=BackgroundIntensity(row["intensity"]),
            progress=float(row.get("progress", 0.0)),
            factions=factions,
            current_state=row.get("current_state", ""),
            recent_developments=[],
            impact_on_daily_life=[],
            player_awareness_level=float(row.get("player_awareness", 0.1)),
        )
    
    async def _adjust_background_tensions(self, amount: float):
        """Adjust all background conflict tensions"""
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE BackgroundConflicts
                SET progress = LEAST(100, progress + $1)
                WHERE user_id = $2 AND conversation_id = $3 AND is_active = true
            """, amount * 100, self.user_id, self.conversation_id)


# ===============================================================================
# ENHANCED BACKGROUND CONFLICT ORCHESTRATOR
# ===============================================================================

class BackgroundConflictOrchestrator:
    """Manages grand conflicts with dynamic LLM generation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded LLM agents
        self._world_event_agent = None
        self._news_agent = None
        self._ripple_agent = None
        self._faction_agent = None
        self._development_agent = None
    
    # ========== Lazy-loaded Agent Properties ==========
    
    @property
    def world_event_agent(self) -> Agent:
        """Agent for generating background world events"""
        if self._world_event_agent is None:
            self._world_event_agent = Agent(
                name="World Event Generator",
                instructions="""
                Generate grand conflict events that happen in the background.
                
                Create events that:
                - Feel like distant but important world events
                - Not directly involve the player
                - Create rich atmosphere and context
                - Provide natural conversation topics
                - Suggest the world is alive beyond daily life
                
                Focus on political intrigue, economic shifts, cultural movements,
                and resource conflicts. Keep events remote but consequential.
                """,
                model="gpt-5-nano",
            )
        return self._world_event_agent
    
    @property
    def development_agent(self) -> Agent:
        """Agent for progressing conflicts"""
        if self._development_agent is None:
            self._development_agent = Agent(
                name="Conflict Development Generator",
                instructions="""
                Generate natural progressions for background conflicts.
                
                Create developments that:
                - Feel organic and consequential
                - Build on previous events
                - Introduce unexpected twists
                - Maintain narrative coherence
                - Create ripple effects
                
                Balance predictable progression with surprises.
                """,
                model="gpt-5-nano",
            )
        return self._development_agent
    
    @property
    def faction_agent(self) -> Agent:
        """Agent for generating faction dynamics"""
        if self._faction_agent is None:
            self._faction_agent = Agent(
                name="Faction Dynamics Generator",
                instructions="""
                Generate complex faction behaviors and interactions.
                
                Create:
                - Faction motivations and goals
                - Alliance formations and betrayals
                - Strategic moves and counter-moves
                - Internal faction politics
                - Public versus private agendas
                
                Make factions feel like real political entities with depth.
                """,
                model="gpt-5-nano",
            )
        return self._faction_agent
    
    # ========== Dynamic Generation Methods ==========
    
    async def generate_background_conflict(
        self, conflict_type: Optional[GrandConflictType] = None
    ) -> BackgroundConflict:
        """Generate a new background conflict with LLM"""
        if not conflict_type:
            conflict_type = random.choice(list(GrandConflictType))
    
        prompt = f"""
        Generate a {conflict_type.value} conflict for background worldbuilding.
    
        Context: This is happening in the wider world, not directly affecting
        the player's daily life but creating atmosphere and context.
    
        Return JSON:
        {{
            "name": "Compelling headline-style name",
            "description": "2-3 sentence dramatic overview",
            "factions": ["3-5 named factions with clear stakes"],
            "initial_state": "Current tense situation",
            "potential_developments": ["5 possible future events"],
            "daily_life_impacts": ["3 subtle effects on everyday life"],
            "conversation_hooks": ["3 ways NPCs might mention this"],
            "faction_dynamics": {{
                "alliances": ["current alliances"],
                "tensions": ["current tensions"],
                "wild_cards": ["unpredictable elements"]
            }}
        }}
        """
    
        response = await Runner.run(self.world_event_agent, prompt)
        data = json.loads(extract_runner_response(response))
    
        # Create database entry
        async with get_db_connection_context() as conn:
            conflict_id = await conn.fetchval(
                """
                INSERT INTO BackgroundConflicts
                (user_id, conversation_id, conflict_type, name, description,
                 intensity, progress, is_active, current_state)
                VALUES ($1, $2, $3, $4, $5, $6, 0, true, $7)
                RETURNING conflict_id
                """,
                self.user_id,
                self.conversation_id,
                conflict_type.value,
                data["name"],
                data["description"],
                BackgroundIntensity.DISTANT_RUMOR.value,
                data["initial_state"],
            )
    
            # Store factions (if present)
            for faction in (data.get("factions") or []):
                await conn.execute(
                    """
                    INSERT INTO BackgroundConflictFactions
                    (conflict_id, faction_name, power_level, stance)
                    VALUES ($1, $2, $3, 'neutral')
                    """,
                    conflict_id,
                    faction,
                    random.uniform(0.3, 0.7),
                )
    
        return BackgroundConflict(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            name=data["name"],
            description=data["description"],
            intensity=BackgroundIntensity.DISTANT_RUMOR,
            progress=0.0,
            factions=list(data.get("factions") or []),
            current_state=data["initial_state"],
            recent_developments=[],
            impact_on_daily_life=list(data.get("daily_life_impacts") or []),
            player_awareness_level=0.1,
        )
    
    async def advance_background_conflict(self, conflict: BackgroundConflict) -> WorldEvent:
        """Advance conflict with dynamic development"""
        prompt = f"""
        Advance this background conflict naturally:
    
        Conflict: {conflict.name}
        Type: {conflict.conflict_type.value}
        Current State: {conflict.current_state}
        Progress: {conflict.progress}%
        Factions: {json.dumps(conflict.factions)}
        Recent: {json.dumps(conflict.recent_developments[-3:] if conflict.recent_developments else [])}
    
        Generate the next development that:
        - Builds on current state
        - Feels consequential
        - Could affect daily life subtly
        - Introduces complexity
    
        Return JSON:
        {{
            "event_type": "battle/negotiation/revelation/escalation/twist",
            "description": "Dramatic 2-3 sentence event",
            "faction_impacts": {{"faction": impact_value}},
            "new_state": "Updated conflict state",
            "progress_change": -10 to +20,
            "intensity_change": "increase/decrease/maintain",
            "creates_opportunity": true/false,
            "opportunity_description": "Optional player involvement",
            "ripple_effects": ["3 subtle effects on the world"],
            "npc_reactions": ["3 ways NPCs might react"]
        }}
        """
    
        response = await Runner.run(self.development_agent, prompt)
        data = json.loads(extract_runner_response(response))
    
        # Update DB
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE BackgroundConflicts
                SET current_state = $1,
                    progress = progress + $2,
                    intensity = $3
                WHERE conflict_id = $4
                """,
                data["new_state"],
                data["progress_change"],
                self._calculate_new_intensity(conflict.intensity, data["intensity_change"]),
                conflict.conflict_id,
            )
    
        return WorldEvent(
            conflict_id=conflict.conflict_id,
            event_type=data["event_type"],
            description=data["description"],
            faction_impacts=data.get("faction_impacts", {}),
            creates_opportunity=bool(data.get("creates_opportunity")),
            opportunity_window=7 if data.get("creates_opportunity") else None,
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
    """Generates dynamic news and rumors with LLM"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._news_generator = None
    
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
    ) -> Dict[str, Any]:
        """Generate dynamic news about a conflict"""
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
    
        # Store in DB
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO BackgroundNews
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
                await get_current_game_day(self.user_id, self.conversation_id, use_names=False),
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
        """Generate today's ripple effects"""
        
        if not active_conflicts:
            return {"ripples": {}}
        
        conflicts_summary = [
            {
                "name": c.name,
                "intensity": c.intensity.value,
                "current_state": c.current_state
            }
            for c in active_conflicts
        ]
        
        prompt = f"""
        Generate daily ripple effects from these background conflicts:
        
        Conflicts: {json.dumps(conflicts_summary)}
        
        Create subtle effects that:
        - Change daily atmosphere
        - Affect NPC moods
        - Create overheard snippets
        - Alter minor details
        - Build tension
        
        Return JSON:
        {{
            "ambient_mood": "overall atmosphere today",
            "npc_mood_modifier": "how NPCs are affected",
            "overheard_snippets": ["5 things player might overhear"],
            "visual_cues": ["3 subtle environmental changes"],
            "price_changes": {{"item": percentage_change}},
            "crowd_behaviors": ["2 subtle crowd behaviors"],
            "ambient_sounds": ["3 background sounds suggesting tension"]
        }}
        """
        
        response = await Runner.run(self.ripple_generator, prompt)
        return {"ripples": json.loads(response.output)}
    
    async def check_for_opportunities(
        self,
        active_conflicts: List[BackgroundConflict],
        player_skills: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check if any conflicts create player opportunities"""
        
        opportunities = []
        
        for conflict in active_conflicts:
            if conflict.intensity.value in ['regular_topic', 'ambient_tension', 'visible_effects']:
                if random.random() < 0.2:  # 20% chance
                    opportunity = await self._generate_opportunity(conflict, player_skills)
                    if opportunity:
                        opportunities.append(opportunity)
        
        return opportunities
    
    async def _generate_opportunity(
        self,
        conflict: BackgroundConflict,
        player_skills: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Generate a specific opportunity"""
        
        prompt = f"""
        Generate an optional opportunity from this conflict:
        
        Conflict: {conflict.name}
        Type: {conflict.conflict_type.value}
        Current State: {conflict.current_state}
        Player Skills: {json.dumps(player_skills)}
        
        Create an opportunity that:
        - Is completely optional
        - Relates to the conflict tangentially
        - Offers interesting rewards
        - Has multiple approaches
        - Creates a memorable moment
        
        Return JSON:
        {{
            "title": "Intriguing opportunity name",
            "description": "What the opportunity is",
            "hook": "How player discovers it",
            "approaches": [
                {{
                    "method": "approach type",
                    "requirement": "skill or resource needed",
                    "risk": "low/medium/high",
                    "reward": "what player gains"
                }}
            ],
            "window": "how long available (in days)",
            "connection": "how it relates to the conflict",
            "consequences": "potential long-term effects"
        }}
        """
        
        response = await Runner.run(self.opportunity_creator, prompt)
        return json.loads(response.output)


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

    # Get synthesizer (keeps side effects consistent even if this does nothing yet)
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    _ = await get_synthesizer(user_id, conversation_id)

    return {
        'world_initialized': True,
        'message': 'Background world initialized through synthesizer',
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

                # Defensive extraction with strict types for the tool response
                world_tension = float(update.get('world_tension', 0.0))
                news = update.get('news') or []
                if not isinstance(news, list):
                    news = [str(news)]

                ripples = ((update.get('ripple_effects') or {}).get('ripples') or {})
                ambient = ripples.get('ambient_mood') or []
                if not isinstance(ambient, list):
                    ambient = [str(ambient)]

                overheard_snips = ripples.get('overheard_snippets') or []
                if not isinstance(overheard_snips, list):
                    overheard_snips = [str(overheard_snips)]
                overheard = str(overheard_snips[0]) if overheard_snips else ""

                optional_ops = update.get('optional_opportunities') or []
                if not isinstance(optional_ops, list):
                    optional_ops = [str(optional_ops)]
                optional_hook = str(optional_ops[0]) if optional_ops else ""

                return {
                    'world_tension': world_tension,
                    'background_news': [str(x) for x in news[:1]],   # 0 or 1 item
                    'ambient_effects': [str(x) for x in ambient],
                    'overheard': overheard,
                    'optional_hook': optional_hook,
                }

    # Fallback with strict, non-nullable shapes
    return {
        'world_tension': 0.0,
        'background_news': [],
        'ambient_effects': [],
        'overheard': "",
        'optional_hook': "",
    }
