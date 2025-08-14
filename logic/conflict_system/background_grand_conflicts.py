# logic/conflict_system/background_grand_conflicts.py
"""
Background Grand Conflicts System
Maintains large-scale conflicts as ambient worldbuilding elements
These run passively and create atmosphere without requiring player involvement
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
    DISTANT_RUMOR = "distant_rumor"  # Barely mentioned
    OCCASIONAL_NEWS = "occasional_news"  # Comes up in conversation
    REGULAR_TOPIC = "regular_topic"  # Frequently discussed
    AMBIENT_TENSION = "ambient_tension"  # Affects mood
    VISIBLE_EFFECTS = "visible_effects"  # Minor daily impacts


@dataclass
class BackgroundConflict:
    """A grand conflict happening in the background"""
    conflict_id: int
    conflict_type: GrandConflictType
    name: str
    description: str
    intensity: BackgroundIntensity
    progress: float  # 0-100
    factions: List[str]  # Named factions involved
    current_state: str  # LLM-generated current status
    recent_developments: List[str]
    impact_on_daily_life: List[str]
    player_awareness_level: float  # 0-1, how much player knows


@dataclass
class WorldEvent:
    """An event in a background conflict"""
    conflict_id: int
    event_type: str
    description: str
    faction_impacts: Dict[str, float]
    creates_opportunity: bool
    opportunity_window: Optional[int]  # Days before opportunity expires


# ===============================================================================
# BACKGROUND CONFLICT ORCHESTRATOR
# ===============================================================================

class BackgroundConflictOrchestrator:
    """Manages grand conflicts as background elements"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.world_event_agent = self._create_world_event_agent()
        self.news_agent = self._create_news_agent()
        self.ripple_agent = self._create_ripple_agent()
        
    def _create_world_event_agent(self) -> Agent:
        """Agent for generating background world events"""
        return Agent(
            name="World Event Generator",
            instructions="""
            Generate grand conflict events that happen in the background of a slice-of-life game.
            
            These events should:
            - Feel like distant but important world events
            - Not directly involve the player
            - Create atmosphere and context
            - Occasionally provide conversation topics
            - Suggest the world is alive beyond the player's daily life
            
            Focus on:
            - Political maneuvering in distant capitals
            - Economic shifts affecting regions
            - Cultural movements gaining momentum
            - Resource conflicts between factions
            
            Keep events feeling remote but consequential.
            The player is living their daily life while history happens around them.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
    
    def _create_news_agent(self) -> Agent:
        """Agent for generating news and rumors about conflicts"""
        return Agent(
            name="News and Rumor Generator",
            instructions="""
            Generate news snippets, rumors, and gossip about background conflicts.
            
            Create content that would naturally come up in:
            - Overheard conversations
            - News broadcasts/papers
            - Social media equivalents
            - Casual gossip
            - NPC commentary
            
            Vary the reliability and bias of information:
            - Official news (mostly accurate)
            - Rumors (partially true)
            - Gossip (embellished)
            - Propaganda (biased)
            - Conspiracy theories (mostly false)
            
            Keep it atmospheric, not actionable.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
    
    def _create_ripple_agent(self) -> Agent:
        """Agent for determining how grand conflicts affect daily life"""
        return Agent(
            name="Ripple Effect Generator",
            instructions="""
            Determine subtle ways background conflicts affect daily life.
            
            Generate minor impacts like:
            - Certain goods becoming expensive/scarce
            - NPCs mentioning worried relatives
            - Changes in social atmosphere
            - Shifted NPC priorities or moods
            - New conversation topics
            - Subtle changes in routines
            
            Effects should be:
            - Indirect and atmospheric
            - Not requiring player action
            - Adding texture to daily life
            - Occasionally creating opportunities
            
            Never force player involvement.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def generate_background_conflict(
        self, 
        conflict_type: Optional[GrandConflictType] = None
    ) -> BackgroundConflict:
        """Generate a new background conflict"""
        
        if not conflict_type:
            conflict_type = random.choice(list(GrandConflictType))
        
        prompt = f"""
        Generate a {conflict_type.value} conflict for background worldbuilding.
        
        This should feel like a major world event that:
        - Involves powerful factions/entities
        - Has complex political/economic implications
        - Would be on the news
        - Doesn't directly involve a regular person's daily life
        
        Return JSON:
        {{
            "name": "Conflict name (like a news headline)",
            "description": "2-3 sentence overview",
            "factions": ["Faction A", "Faction B", "Faction C"],
            "initial_state": "Current situation",
            "potential_developments": [
                "Possible future event 1",
                "Possible future event 2"
            ],
            "daily_life_impacts": [
                "Subtle effect on everyday life"
            ],
            "conversation_hooks": [
                "How NPCs might mention this"
            ]
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
        """Advance a background conflict with a new event"""
        
        prompt = f"""
        Current conflict: {conflict.name}
        Type: {conflict.conflict_type.value}
        Current state: {conflict.current_state}
        Progress: {conflict.progress}%
        Factions: {json.dumps(conflict.factions)}
        
        Generate the next development in this conflict.
        
        Return JSON:
        {{
            "event_type": "battle/negotiation/revelation/escalation/development",
            "description": "What happened (2-3 sentences)",
            "faction_impacts": {{"faction_name": impact_value}},
            "new_state": "Updated conflict state",
            "progress_change": -10 to +20,
            "intensity_change": "increase/decrease/maintain",
            "creates_opportunity": true/false,
            "opportunity_description": "Optional player opportunity"
        }}
        
        Keep it feeling distant but important.
        """
        
        response = await Runner.run(self.world_event_agent, prompt)
        event_data = json.loads(response.output)
        
        # Create world event
        event = WorldEvent(
            conflict_id=conflict.conflict_id,
            event_type=event_data['event_type'],
            description=event_data['description'],
            faction_impacts=event_data['faction_impacts'],
            creates_opportunity=event_data['creates_opportunity'],
            opportunity_window=7 if event_data['creates_opportunity'] else None
        )
        
        # Update conflict state
        new_progress = min(100, max(0, conflict.progress + event_data['progress_change']))
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE BackgroundConflicts
                SET progress = $1, current_state = $2, last_event_at = CURRENT_TIMESTAMP
                WHERE conflict_id = $3
            """, new_progress, event_data['new_state'], conflict.conflict_id)
            
            # Record event
            await conn.execute("""
                INSERT INTO BackgroundConflictEvents
                (conflict_id, event_type, description, impact_data, game_day)
                VALUES ($1, $2, $3, $4, $5)
            """, conflict.conflict_id, event.event_type, event.description,
            json.dumps(event_data), await get_current_game_day(self.user_id, self.conversation_id))
        
        # Update intensity if needed
        if event_data['intensity_change'] == 'increase':
            await self._increase_intensity(conflict)
        elif event_data['intensity_change'] == 'decrease':
            await self._decrease_intensity(conflict)
        
        return event
    
    async def _increase_intensity(self, conflict: BackgroundConflict):
        """Increase how prominent the conflict is"""
        intensity_order = [
            BackgroundIntensity.DISTANT_RUMOR,
            BackgroundIntensity.OCCASIONAL_NEWS,
            BackgroundIntensity.REGULAR_TOPIC,
            BackgroundIntensity.AMBIENT_TENSION,
            BackgroundIntensity.VISIBLE_EFFECTS
        ]
        
        current_index = intensity_order.index(conflict.intensity)
        if current_index < len(intensity_order) - 1:
            new_intensity = intensity_order[current_index + 1]
            
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE BackgroundConflicts
                    SET intensity = $1
                    WHERE conflict_id = $2
                """, new_intensity.value, conflict.conflict_id)
    
    async def _decrease_intensity(self, conflict: BackgroundConflict):
        """Decrease how prominent the conflict is"""
        intensity_order = [
            BackgroundIntensity.DISTANT_RUMOR,
            BackgroundIntensity.OCCASIONAL_NEWS,
            BackgroundIntensity.REGULAR_TOPIC,
            BackgroundIntensity.AMBIENT_TENSION,
            BackgroundIntensity.VISIBLE_EFFECTS
        ]
        
        current_index = intensity_order.index(conflict.intensity)
        if current_index > 0:
            new_intensity = intensity_order[current_index - 1]
            
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE BackgroundConflicts
                    SET intensity = $1
                    WHERE conflict_id = $2
                """, new_intensity.value, conflict.conflict_id)


# ===============================================================================
# NEWS AND RUMOR GENERATION
# ===============================================================================

class BackgroundNewsGenerator:
    """Generates news, rumors, and gossip about background conflicts"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.news_agent = Agent(
            name="News Generator",
            instructions="""
            Generate varied news content about background world events.
            
            Content types:
            - Official news: Formal, mostly accurate
            - Rumors: Informal, partially true
            - Gossip: Personal, embellished
            - Propaganda: Biased toward a faction
            - Conspiracy: Wild speculation
            
            Match tone to content type.
            Keep it atmospheric, not a call to action.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.85)
        )
    
    async def generate_news_item(
        self,
        conflict: BackgroundConflict,
        news_type: str = "random"
    ) -> Dict[str, Any]:
        """Generate a news item about a background conflict"""
        
        if news_type == "random":
            news_type = random.choice([
                "official", "rumor", "gossip", "propaganda", "conspiracy"
            ])
        
        prompt = f"""
        Generate {news_type} news about: {conflict.name}
        Current state: {conflict.current_state}
        Recent developments: {json.dumps(conflict.recent_developments[-3:])}
        
        Return JSON:
        {{
            "headline": "Attention-grabbing title",
            "content": "1-2 sentence news item",
            "source": "Who's saying this",
            "reliability": 0.0-1.0,
            "npc_commentary": "How an NPC might mention this",
            "mood_impact": "How this affects social atmosphere"
        }}
        
        Match the tone to the news type:
        - Official: Formal and measured
        - Rumor: "I heard that..."
        - Gossip: Personal and dramatic
        - Propaganda: Obviously biased
        - Conspiracy: Wild and speculative
        """
        
        response = await Runner.run(self.news_agent, prompt)
        news_data = json.loads(response.output)
        
        # Store in database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO BackgroundNews
                (conflict_id, news_type, headline, content, source, 
                 reliability, game_day, user_id, conversation_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, conflict.conflict_id, news_type, news_data['headline'],
            news_data['content'], news_data['source'], news_data['reliability'],
            await get_current_game_day(self.user_id, self.conversation_id),
            self.user_id, self.conversation_id)
        
        return news_data
    
    async def generate_npc_discussion(
        self,
        conflict: BackgroundConflict,
        npc_personalities: List[Dict]
    ) -> List[str]:
        """Generate how NPCs discuss background conflicts"""
        
        prompt = f"""
        Background conflict: {conflict.name}
        Intensity: {conflict.intensity.value}
        NPCs present: {json.dumps(npc_personalities)}
        
        Generate casual mentions of this conflict in conversation.
        
        Return JSON array of NPC comments:
        [
            {{
                "npc_index": 0,
                "comment": "Casual mention of the conflict",
                "emotion": "worried/dismissive/interested/bored",
                "follows_up": true/false
            }}
        ]
        
        Keep it natural - these are background topics, not main conversation.
        People mention world events casually while living their lives.
        """
        
        response = await Runner.run(self.news_agent, prompt)
        discussions = json.loads(response.output)
        
        return [d['comment'] for d in discussions]


# ===============================================================================
# RIPPLE EFFECTS ON DAILY LIFE
# ===============================================================================

class BackgroundConflictRipples:
    """Manages how background conflicts create ripples in daily life"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.ripple_agent = Agent(
            name="Ripple Effect Generator",
            instructions="""
            Generate subtle effects of distant conflicts on daily life.
            
            Focus on:
            - Minor inconveniences or changes
            - Mood and atmosphere shifts
            - Conversation topics
            - NPC preoccupations
            - Small economic effects
            
            Never create effects that demand player action.
            Keep it atmospheric and textural.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def generate_daily_ripples(
        self,
        active_conflicts: List[BackgroundConflict]
    ) -> Dict[str, Any]:
        """Generate how background conflicts affect today"""
        
        # Only process conflicts with sufficient intensity
        relevant_conflicts = [
            c for c in active_conflicts
            if c.intensity not in [
                BackgroundIntensity.DISTANT_RUMOR,
                BackgroundIntensity.OCCASIONAL_NEWS
            ]
        ]
        
        if not relevant_conflicts:
            return {'ripples': [], 'mood_modifier': 0}
        
        prompt = f"""
        Active background conflicts affecting the world:
        {json.dumps([{
            'name': c.name,
            'type': c.conflict_type.value,
            'intensity': c.intensity.value
        } for c in relevant_conflicts])}
        
        Generate subtle daily life effects.
        
        Return JSON:
        {{
            "item_availability": [
                {{"item": "what's affected", "reason": "vague connection to conflict"}}
            ],
            "npc_mood_shifts": [
                {{"mood_change": "worried/distracted/energized", "reason": "vague worry"}}
            ],
            "ambient_changes": [
                "Description of subtle atmosphere change"
            ],
            "overheard_snippets": [
                "Fragment of overheard conversation about conflicts"
            ],
            "price_changes": [
                {{"category": "goods type", "change": 1.1, "reason": "supply chain"}}
            ],
            "optional_opportunities": [
                {{"description": "Something player could do if they want",
                  "reward_type": "information/connection/small_benefit"}}
            ]
        }}
        
        Keep all effects minor and atmospheric.
        """
        
        response = await Runner.run(self.ripple_agent, prompt)
        ripples = json.loads(response.output)
        
        # Calculate overall mood modifier
        mood_modifier = len(relevant_conflicts) * 0.05  # Small tension increase
        
        return {
            'ripples': ripples,
            'mood_modifier': mood_modifier,
            'affected_by_conflicts': [c.name for c in relevant_conflicts]
        }
    
    async def check_for_opportunities(
        self,
        active_conflicts: List[BackgroundConflict],
        player_skills: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check if any background conflicts create optional opportunities"""
        
        opportunities = []
        
        for conflict in active_conflicts:
            # Only conflicts with certain intensity create opportunities
            if conflict.intensity not in [
                BackgroundIntensity.REGULAR_TOPIC,
                BackgroundIntensity.AMBIENT_TENSION,
                BackgroundIntensity.VISIBLE_EFFECTS
            ]:
                continue
            
            # Check recent events for opportunities
            async with get_db_connection_context() as conn:
                recent_events = await conn.fetch("""
                    SELECT * FROM BackgroundConflictEvents
                    WHERE conflict_id = $1
                    AND game_day > (SELECT current_day FROM game_calendar 
                                   WHERE user_id = $2 AND conversation_id = $3) - 3
                    ORDER BY game_day DESC
                    LIMIT 3
                """, conflict.conflict_id, self.user_id, self.conversation_id)
            
            for event in recent_events:
                event_data = json.loads(event['impact_data'])
                if event_data.get('creates_opportunity'):
                    opportunities.append({
                        'conflict_name': conflict.name,
                        'opportunity': event_data.get('opportunity_description', 
                                                     'A chance to learn more'),
                        'window': 'A few days',
                        'optional': True,
                        'rewards': ['information', 'connections', 'minor_benefits']
                    })
        
        return opportunities


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
        if len(active_conflicts) < 3:  # Maintain 3-5 background conflicts
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
            'world_tension': sum(c.progress for c in active_conflicts) / (len(active_conflicts) * 100)
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
            topics.append(news['content'])
        
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


@function_tool
async def generate_npc_worldly_comment(
    ctx: RunContextWrapper,
    npc_id: int,
    context: str = "casual"
) -> str:
    """Generate an NPC comment about background world events"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = BackgroundConflictManager(user_id, conversation_id)
    
    # Get conversation topics
    topics = await manager.get_conversation_topics()
    
    if not topics:
        return ""
    
    # Get NPC personality for appropriate comment style
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, personality_traits FROM NPCStats
            WHERE npc_id = $1
        """, npc_id)
    
    personality = json.loads(npc['personality_traits']) if npc else {}
    
    # Generate contextual comment
    agent = Agent(
        name="NPC Commenter",
        instructions=f"""
        Generate a casual comment about world events.
        NPC personality: {json.dumps(personality)}
        Context: {context}
        
        Make it feel natural, like someone mentioning the news in passing.
        """,
        model="gpt-5-nano",
        model_settings=ModelSettings(temperature=0.8)
    )
    
    prompt = f"""
    Topic options: {random.choice(topics)}
    
    Generate a single casual comment (1 sentence) that this NPC might make.
    Keep it conversational and match their personality.
    """
    
    response = await Runner.run(agent, prompt)
    return response.output


# ===============================================================================
# PLAYER ENGAGEMENT OPTIONS
# ===============================================================================

@function_tool
async def check_background_engagement_options(
    ctx: RunContextWrapper
) -> List[Dict[str, Any]]:
    """Check what background conflicts the player could engage with if they wanted"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    options = []
    
    async with get_db_connection_context() as conn:
        # Get high-intensity conflicts
        conflicts = await conn.fetch("""
            SELECT * FROM BackgroundConflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND is_active = true
            AND intensity IN ('ambient_tension', 'visible_effects')
        """, user_id, conversation_id)
        
        for conflict in conflicts:
            options.append({
                'conflict_name': conflict['name'],
                'engagement_type': 'optional_investigation',
                'description': f"You could look into {conflict['name']} if you're curious",
                'commitment_level': 'minimal',
                'potential_rewards': ['knowledge', 'connections'],
                'completely_optional': True
            })
    
    return options
