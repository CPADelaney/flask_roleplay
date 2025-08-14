# logic/conflict_system/background_grand_conflicts.py
"""
Background Grand Conflicts System with LLM-generated content
Maintains large-scale conflicts as ambient worldbuilding elements
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

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
    def news_agent(self) -> Agent:
        """Agent for generating news and rumors"""
        if self._news_agent is None:
            self._news_agent = Agent(
                name="News and Rumor Generator",
                instructions="""
                Generate news snippets, rumors, and gossip about background conflicts.
                
                Create varied content:
                - Official news (mostly accurate)
                - Rumors (partially true)
                - Gossip (embellished)
                - Propaganda (biased)
                - Conspiracy theories (mostly false)
                
                Vary tone and reliability. Keep atmospheric, not actionable.
                """,
                model="gpt-5-nano",
            )
        return self._news_agent
    
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
    def ripple_agent(self) -> Agent:
        """Agent for generating ripple effects"""
        if self._ripple_agent is None:
            self._ripple_agent = Agent(
                name="Ripple Effect Generator",
                instructions="""
                Generate subtle ripple effects from grand conflicts.
                
                Create effects that:
                - Subtly affect daily life
                - Change ambient atmosphere
                - Influence NPC behaviors
                - Create conversation topics
                - Suggest larger forces at work
                
                Keep effects indirect but noticeable.
                """,
                model="gpt-5-nano",
            )
        return self._ripple_agent
    
    # ========== Dynamic Generation Methods ==========
    
    async def generate_background_conflict(
        self, 
        conflict_type: Optional[GrandConflictType] = None
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
        data = json.loads(response.output)
        
        # Create database entry
        async with get_db_connection_context() as conn:
            conflict_id = await conn.fetchval("""
                INSERT INTO BackgroundConflicts
                (user_id, conversation_id, conflict_type, name, description,
                 intensity, progress, is_active, current_state)
                VALUES ($1, $2, $3, $4, $5, $6, 0, true, $7)
                RETURNING conflict_id
            """, self.user_id, self.conversation_id, conflict_type.value,
            data['name'], data['description'], 
            BackgroundIntensity.DISTANT_RUMOR.value,
            data['initial_state'])
            
            # Store factions
            for faction in data['factions']:
                await conn.execute("""
                    INSERT INTO BackgroundConflictFactions
                    (conflict_id, faction_name, power_level, stance)
                    VALUES ($1, $2, $3, 'neutral')
                """, conflict_id, faction, random.uniform(0.3, 0.7))
        
        return BackgroundConflict(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            name=data['name'],
            description=data['description'],
            intensity=BackgroundIntensity.DISTANT_RUMOR,
            progress=0.0,
            factions=data['factions'],
            current_state=data['initial_state'],
            recent_developments=[],
            impact_on_daily_life=data['daily_life_impacts'],
            player_awareness_level=0.1
        )
    
    async def advance_background_conflict(
        self,
        conflict: BackgroundConflict
    ) -> WorldEvent:
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
        data = json.loads(response.output)
        
        # Update database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE BackgroundConflicts
                SET current_state = $1,
                    progress = progress + $2,
                    intensity = $3
                WHERE conflict_id = $4
            """, data['new_state'], data['progress_change'],
            self._calculate_new_intensity(conflict.intensity, data['intensity_change']),
            conflict.conflict_id)
        
        return WorldEvent(
            conflict_id=conflict.conflict_id,
            event_type=data['event_type'],
            description=data['description'],
            faction_impacts=data['faction_impacts'],
            creates_opportunity=data['creates_opportunity'],
            opportunity_window=7 if data['creates_opportunity'] else None
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
    
    async def generate_faction_move(
        self,
        conflict: BackgroundConflict,
        faction_name: str
    ) -> Dict[str, Any]:
        """Generate a strategic move by a faction"""
        
        prompt = f"""
        Generate a strategic move for this faction:
        
        Faction: {faction_name}
        Conflict: {conflict.name}
        Current State: {conflict.current_state}
        Other Factions: {json.dumps([f for f in conflict.factions if f != faction_name])}
        
        Create a move that:
        - Advances faction interests
        - Creates new dynamics
        - Could backfire
        - Has hidden motives
        
        Return JSON:
        {{
            "move_type": "alliance/betrayal/maneuver/revelation/gambit",
            "public_action": "What everyone sees",
            "hidden_motive": "Real intention",
            "targets": ["affected factions"],
            "success_probability": 0.0 to 1.0,
            "potential_backfire": "How this could go wrong",
            "rumors_generated": ["3 rumors this creates"]
        }}
        """
        
        response = await Runner.run(self.faction_agent, prompt)
        return json.loads(response.output)


# ===============================================================================
# BACKGROUND NEWS GENERATOR
# ===============================================================================

class BackgroundNewsGenerator:
    """Generates dynamic news and rumors with LLM"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._news_generator = None
        self._rumor_mill = None
        self._propaganda_writer = None
    
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
    
    @property
    def rumor_mill(self) -> Agent:
        if self._rumor_mill is None:
            self._rumor_mill = Agent(
                name="Rumor Mill",
                instructions="""
                Generate rumors and gossip about world events.
                
                Create rumors that are:
                - Based on partial truths
                - Embellished or distorted
                - Revealing hidden dynamics
                - Creating social tension
                - Sometimes completely false
                
                Make them feel like organic gossip.
                """,
                model="gpt-5-nano",
            )
        return self._rumor_mill
    
    @property
    def propaganda_writer(self) -> Agent:
        if self._propaganda_writer is None:
            self._propaganda_writer = Agent(
                name="Propaganda Writer",
                instructions="""
                Generate propaganda and biased messaging.
                
                Create content that:
                - Serves faction interests
                - Distorts truth cleverly
                - Appeals to emotions
                - Creates division
                - Seems reasonable on surface
                
                Make propaganda feel authentic to its source.
                """,
                model="gpt-5-nano",
            )
        return self._propaganda_writer
    
    async def generate_news_item(
        self,
        conflict: BackgroundConflict,
        news_type: str = "random"
    ) -> Dict[str, Any]:
        """Generate dynamic news about a conflict"""
        
        if news_type == "random":
            news_type = random.choice(["official", "independent", "tabloid", "rumor", "propaganda"])
        
        # Select appropriate agent
        if news_type == "rumor":
            agent = self.rumor_mill
        elif news_type == "propaganda":
            agent = self.propaganda_writer
        else:
            agent = self.news_generator
        
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
        
        response = await Runner.run(agent, prompt)
        news_data = json.loads(response.output)
        
        # Store in database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO BackgroundNews
                (user_id, conversation_id, conflict_id, headline, 
                 source, content, reliability, game_day)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, self.user_id, self.conversation_id, conflict.conflict_id,
            news_data['headline'], news_data['source'], 
            news_data['content'], news_data['reliability'],
            await get_current_game_day(self.user_id, self.conversation_id))
        
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
# BACKGROUND CONFLICT MANAGER
# ===============================================================================

class BackgroundConflictManager:
    """Main manager for all background conflict systems"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.orchestrator = BackgroundConflictOrchestrator(user_id, conversation_id)
        self.news_generator = BackgroundNewsGenerator(user_id, conversation_id)
        self.ripple_manager = BackgroundConflictRipples(user_id, conversation_id)
    
    async def daily_background_update(self) -> Dict[str, Any]:
        """Daily update of all background conflicts"""
        
        # Get active background conflicts
        async with get_db_connection_context() as conn:
            conflicts_data = await conn.fetch("""
                SELECT * FROM BackgroundConflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
        
        active_conflicts = []
        for conflict_data in conflicts_data:
            active_conflicts.append(BackgroundConflict(
                conflict_id=conflict_data['conflict_id'],
                conflict_type=GrandConflictType(conflict_data['conflict_type']),
                name=conflict_data['name'],
                description=conflict_data['description'],
                intensity=BackgroundIntensity(conflict_data['intensity']),
                progress=conflict_data['progress'],
                factions=json.loads(conflict_data.get('factions', '[]')),
                current_state=conflict_data['current_state'],
                recent_developments=[],
                impact_on_daily_life=[],
                player_awareness_level=conflict_data.get('player_awareness', 0.1)
            ))
        
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
            # Get recent news
            recent_news = await conn.fetch("""
                SELECT headline, content FROM BackgroundNews
                WHERE user_id = $1 AND conversation_id = $2
                AND game_day > (SELECT current_day FROM game_calendar 
                               WHERE user_id = $3 AND conversation_id = $4) - 7
                ORDER BY game_day DESC
                LIMIT 10
            """, self.user_id, self.conversation_id, self.user_id, self.conversation_id)
        
        topics = []
        for news in recent_news:
            topics.append(f"Did you hear about {news['headline']}?")
            topics.append(news['content'][:100] + "...")
        
        return topics


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def initialize_background_world(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Initialize the background world with grand conflicts"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = BackgroundConflictManager(user_id, conversation_id)
    
    # Generate initial set of background conflicts
    conflict_types = [
        GrandConflictType.POLITICAL_SUCCESSION,
        GrandConflictType.ECONOMIC_CRISIS,
        GrandConflictType.FACTION_WAR
    ]
    
    conflicts = []
    for conflict_type in conflict_types:
        conflict = await manager.orchestrator.generate_background_conflict(conflict_type)
        conflicts.append({
            'name': conflict.name,
            'type': conflict.conflict_type.value,
            'description': conflict.description
        })
    
    return {
        'world_initialized': True,
        'background_conflicts': conflicts,
        'message': 'The world beyond your daily life is full of distant dramas'
    }


@function_tool
async def get_daily_background_flavor(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Get today's background world flavor for atmospheric text"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = BackgroundConflictManager(user_id, conversation_id)
    
    # Get daily update
    daily_update = await manager.daily_background_update()
    
    # Format for use in scenes
    flavor = {
        'world_tension': daily_update['world_tension'],
        'background_news': daily_update['news'][:1] if daily_update['news'] else None,
        'ambient_effects': daily_update['ripple_effects'].get('ripples', {}).get('ambient_changes', []),
        'overheard': random.choice(
            daily_update['ripple_effects'].get('ripples', {}).get('overheard_snippets', [''])
        ) if daily_update['ripple_effects'].get('ripples', {}).get('overheard_snippets') else None,
        'optional_hook': daily_update['optional_opportunities'][0] if daily_update['optional_opportunities'] else None
    }
    
    return flavor
