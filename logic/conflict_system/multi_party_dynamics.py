# logic/conflict_system/multi_party_dynamics.py
"""
Multi-Party Dynamics System with LLM-generated content
Integrated with ConflictSynthesizer as the central orchestrator
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# MULTI-PARTY STRUCTURES
# ===============================================================================

class FactionRole(Enum):
    """Roles factions can play in conflicts"""
    AGGRESSOR = "aggressor"
    DEFENDER = "defender"
    MEDIATOR = "mediator"
    OPPORTUNIST = "opportunist"
    NEUTRAL = "neutral"
    WILDCARD = "wildcard"
    KINGMAKER = "kingmaker"  # Can tip the balance

class AllianceType(Enum):
    """Types of alliances between parties"""
    FORMAL = "formal"  # Open alliance
    SECRET = "secret"  # Hidden cooperation
    TEMPORARY = "temporary"  # Issue-specific
    DEFENSIVE = "defensive"  # Only if attacked
    OPPORTUNISTIC = "opportunistic"  # Based on advantage

class BetrayalType(Enum):
    """Types of betrayal"""
    DEFECTION = "defection"  # Switch sides
    SABOTAGE = "sabotage"  # Undermine from within
    INFORMATION_LEAK = "information_leak"  # Share secrets
    ABANDONMENT = "abandonment"  # Leave at crucial moment
    DOUBLE_CROSS = "double_cross"  # Play multiple sides

@dataclass
class Faction:
    """A party in multi-party conflict"""
    faction_id: int
    name: str
    members: List[int]  # NPC/Player IDs
    resources: Dict[str, float]  # Various resources
    goals: List[str]
    strengths: List[str]
    weaknesses: List[str]
    current_stance: FactionRole
    reputation: float  # 0-1, affects alliance potential

@dataclass
class MultiPartyConflict:
    """A conflict involving multiple parties"""
    conflict_id: int
    name: str
    factions: Dict[int, Faction]  # faction_id -> Faction
    alliances: List['Alliance']
    active_negotiations: List['Negotiation']
    power_balance: Dict[int, float]  # faction_id -> power level
    escalation_level: float  # 0-1, how intense
    potential_outcomes: List[str]

@dataclass
class Alliance:
    """An alliance between factions"""
    alliance_id: int
    alliance_type: AllianceType
    members: Set[int]  # faction_ids
    terms: Dict[str, Any]
    strength: float  # 0-1, how strong the bond
    secret: bool
    expiration_condition: Optional[str]

@dataclass
class Negotiation:
    """An ongoing negotiation"""
    negotiation_id: int
    participants: List[int]  # faction_ids
    topic: str
    offers: Dict[int, str]  # faction_id -> offer
    leverage_in_play: Dict[int, List[str]]
    deadline: Optional[datetime]
    mediator: Optional[int]  # faction_id of mediator


# ===============================================================================
# MULTI-PARTY SUBSYSTEM (Integrated with Synthesizer)
# ===============================================================================

class MultiPartyConflictSubsystem:
    """
    Multi-party subsystem that integrates with ConflictSynthesizer.
    Manages complex multi-faction conflicts, alliances, and betrayals.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._faction_strategist = None
        self._alliance_broker = None
        self._betrayal_orchestrator = None
        self._negotiation_mediator = None
        self._outcome_predictor = None
        
        # Reference to synthesizer
        self.synthesizer = None
    
    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.MULTIPARTY
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        return {
            'faction_management',
            'alliance_formation',
            'betrayal_orchestration',
            'negotiation_mediation',
            'outcome_prediction',
            'power_balance_tracking'
        }
    
    @property
    def dependencies(self) -> Set:
        """Return other subsystems this depends on"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return {
            SubsystemType.STAKEHOLDER,  # Need stakeholder system
            SubsystemType.LEVERAGE,  # Use leverage in negotiations
            SubsystemType.SOCIAL  # Social dynamics affect alliances
        }
    
    @property
    def event_subscriptions(self) -> Set:
        """Return events this subsystem wants to receive"""
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.CONFLICT_CREATED,
            EventType.STAKEHOLDER_ACTION,
            EventType.PHASE_TRANSITION,
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        return True
        
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
        
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                # Check context for multiparty flag instead of conflict type
                context = event.payload.get('context', {})
                conflict_type = event.payload.get('conflict_type')
                
                # Check if this should be a multi-party conflict based on context
                if context.get('is_multiparty') or 'faction' in conflict_type:
                    # Initialize as multi-party
                    conflict_id = event.payload.get('conflict_id')
                    factions = await self._initialize_factions(conflict_id)
                    
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={
                            'multi_party_initialized': True,
                            'faction_count': len(factions)
                        }
                    )
                    
            elif event.event_type == EventType.STAKEHOLDER_ACTION:
                # Process faction actions
                stakeholder_id = event.payload.get('stakeholder_id')
                action_type = event.payload.get('action_type')
                
                # Check if stakeholder is part of a faction
                faction = await self._get_stakeholder_faction(stakeholder_id)
                
                if faction:
                    # Process as faction action
                    result = await self._process_faction_action(
                        faction, action_type, event.payload
                    )
                    
                    side_effects = []
                    
                    # Check for alliance or betrayal triggers
                    if result.get('triggers_alliance'):
                        side_effects.append(SystemEvent(
                            event_id=f"alliance_{event.event_id}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=self.subsystem_type,
                            payload={'alliance_formed': result['alliance']},
                            priority=5
                        ))
                    
                    if result.get('triggers_betrayal'):
                        side_effects.append(SystemEvent(
                            event_id=f"betrayal_{event.event_id}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=self.subsystem_type,
                            payload={'betrayal_occurred': result['betrayal']},
                            priority=5
                        ))
                    
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data=result,
                        side_effects=side_effects
                    )
            
            elif event.event_type == EventType.PHASE_TRANSITION:
                # Handle phase transitions for multi-party conflicts
                return await self._handle_phase_transition(event)
            
            elif event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check()
                )
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={}
            )
            
        except Exception as e:
            logger.error(f"Multi-party subsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        async with get_db_connection_context() as conn:
            faction_count = await conn.fetchval("""
                SELECT COUNT(*) FROM factions
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            alliance_count = await conn.fetchval("""
                SELECT COUNT(*) FROM alliances
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
            
            negotiation_count = await conn.fetchval("""
                SELECT COUNT(*) FROM negotiations
                WHERE user_id = $1 AND conversation_id = $2
                AND status = 'active'
            """, self.user_id, self.conversation_id)
        
        is_healthy = faction_count < 20 and negotiation_count < 10
        
        return {
            'healthy': is_healthy,
            'active_factions': faction_count,
            'active_alliances': alliance_count,
            'active_negotiations': negotiation_count,
            'issue': 'Too many factions' if faction_count >= 20 else None
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get multi-party data for a specific conflict"""
        async with get_db_connection_context() as conn:
            # Get factions involved in this conflict
            factions = await conn.fetch("""
                SELECT f.* FROM factions f
                JOIN faction_conflicts fc ON f.faction_id = fc.faction_id
                WHERE fc.conflict_id = $1
            """, conflict_id)
            
            # Get alliances
            alliances = await conn.fetch("""
                SELECT * FROM alliances
                WHERE faction_id_1 IN (
                    SELECT faction_id FROM faction_conflicts WHERE conflict_id = $1
                )
                AND is_active = true
            """, conflict_id)
        
        return {
            'faction_count': len(factions),
            'factions': [
                {
                    'id': f['faction_id'],
                    'name': f['name'],
                    'stance': f['stance'],
                    'reputation': f['reputation']
                }
                for f in factions
            ],
            'alliance_count': len(alliances)
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of multi-party system"""
        async with get_db_connection_context() as conn:
            # Get power balance
            top_factions = await conn.fetch("""
                SELECT faction_id, name, reputation
                FROM factions
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY reputation DESC
                LIMIT 3
            """, self.user_id, self.conversation_id)
        
        return {
            'dominant_factions': [
                {'name': f['name'], 'reputation': f['reputation']}
                for f in top_factions
            ],
            'power_balance': await self._calculate_power_balance()
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if multi-party system is relevant to scene"""
        # Relevant when multiple NPCs from different factions are present
        present_npcs = scene_context.get('present_npcs', [])
        
        if len(present_npcs) < 2:
            return False
        
        # Check if NPCs belong to different factions
        async with get_db_connection_context() as conn:
            faction_diversity = await conn.fetchval("""
                SELECT COUNT(DISTINCT f.faction_id)
                FROM factions f
                JOIN faction_members fm ON f.faction_id = fm.faction_id
                WHERE fm.member_id = ANY($1)
            """, present_npcs)
        
        return faction_diversity > 1
    
    # ========== Agent Properties ==========
    
    @property
    def faction_strategist(self) -> Agent:
        """Agent for faction strategy generation"""
        if self._faction_strategist is None:
            self._faction_strategist = Agent(
                name="Faction Strategist",
                instructions="""
                Generate realistic faction strategies in multi-party conflicts.
                
                Consider:
                - Each faction's goals and resources
                - Power dynamics and balance
                - Historical relationships
                - Risk vs reward calculations
                - Long-term vs short-term gains
                
                Create strategies that are:
                - Politically savvy
                - Resource-aware
                - Contextually appropriate
                - Interesting for gameplay
                
                Balance realpolitik with dramatic potential.
                """,
                model="gpt-5-nano",
            )
        return self._faction_strategist
    
    @property
    def alliance_broker(self) -> Agent:
        """Agent for managing alliances"""
        if self._alliance_broker is None:
            self._alliance_broker = Agent(
                name="Alliance Broker",
                instructions="""
                Facilitate alliance formation and management.
                
                Consider:
                - Mutual benefits and shared threats
                - Trust levels and past betrayals
                - Resource complementarity
                - Power balance impacts
                - Secret vs public alliances
                
                Generate alliances that:
                - Make strategic sense
                - Have clear terms
                - Include exit clauses
                - Create interesting dynamics
                
                Think like a diplomatic strategist.
                """,
                model="gpt-5-nano",
            )
        return self._alliance_broker
    
    @property
    def betrayal_orchestrator(self) -> Agent:
        """Agent for managing betrayals"""
        if self._betrayal_orchestrator is None:
            self._betrayal_orchestrator = Agent(
                name="Betrayal Orchestrator",
                instructions="""
                Orchestrate realistic betrayals in multi-party conflicts.
                
                Consider:
                - Motivations for betrayal
                - Timing and opportunity
                - Risk of discovery
                - Consequences and backlash
                - Cover stories and justifications
                
                Create betrayals that are:
                - Motivated by clear gains
                - Dramatically satisfying
                - Not overdone or random
                - Consequential
                
                Balance shock value with believability.
                """,
                model="gpt-5-nano",
            )
        return self._betrayal_orchestrator
    
    @property
    def negotiation_mediator(self) -> Agent:
        """Agent for managing negotiations"""
        if self._negotiation_mediator is None:
            self._negotiation_mediator = Agent(
                name="Negotiation Mediator",
                instructions="""
                Facilitate complex multi-party negotiations.
                
                Consider:
                - Each party's bottom line
                - BATNA (best alternative to negotiated agreement)
                - Leverage and pressure points
                - Face-saving measures
                - Win-win possibilities
                
                Generate negotiations that:
                - Feel realistic
                - Have genuine stakes
                - Allow for creative solutions
                - Build or destroy trust
                
                Think like a skilled diplomat.
                """,
                model="gpt-5-nano",
            )
        return self._negotiation_mediator
    
    @property
    def outcome_predictor(self) -> Agent:
        """Agent for predicting conflict outcomes"""
        if self._outcome_predictor is None:
            self._outcome_predictor = Agent(
                name="Outcome Predictor",
                instructions="""
                Predict likely outcomes of multi-party conflicts.
                
                Consider:
                - Current power balance
                - Alliance strengths
                - Resource availability
                - Escalation trajectory
                - Historical precedents
                
                Generate predictions that:
                - Account for multiple variables
                - Include unexpected possibilities
                - Consider domino effects
                - Remain plausible
                
                Think like a strategic analyst.
                """,
                model="gpt-5-nano",
            )
        return self._outcome_predictor
    
    # ========== Multi-Party Management Methods ==========
    
    async def _initialize_factions(self, conflict_id: int) -> List[Faction]:
        """Initialize factions for a multi-party conflict"""
        
        # Get stakeholders
        async with get_db_connection_context() as conn:
            stakeholders = await conn.fetch("""
                SELECT * FROM conflict_stakeholders WHERE conflict_id = $1
            """, conflict_id)
        
        # Group into factions or create individual factions
        factions = []
        for stakeholder in stakeholders[:3]:  # Limit to 3 initial factions
            faction = await self.create_faction(
                f"Faction_{stakeholder['stakeholder_id']}",
                [stakeholder['stakeholder_id']],
                ["Achieve dominance", "Protect interests"]
            )
            factions.append(faction)
            
            # Link faction to conflict
            await conn.execute("""
                INSERT INTO faction_conflicts (faction_id, conflict_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, faction.faction_id, conflict_id)
        
        return factions
    
    async def create_faction(
        self,
        name: str,
        initial_members: List[int],
        goals: List[str]
    ) -> Faction:
        """Create a new faction with LLM-generated characteristics"""
        
        prompt = f"""
        Create a faction profile:
        
        Name: {name}
        Initial Members: {len(initial_members)} members
        Primary Goals: {json.dumps(goals)}
        
        Generate:
        1. 3-4 key strengths
        2. 2-3 vulnerabilities/weaknesses
        3. Initial resources (political: 0-1, economic: 0-1, military: 0-1, information: 0-1)
        4. Starting stance (aggressor/defender/mediator/opportunist/neutral)
        5. Reputation score (0-1)
        6. Unique characteristics
        
        Make the faction feel distinct and interesting.
        Format as JSON.
        """
        
        response = await self.faction_strategist.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store faction
            async with get_db_connection_context() as conn:
                faction_id = await conn.fetchval("""
                    INSERT INTO factions
                    (user_id, conversation_id, name, members, goals, resources, stance, reputation)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING faction_id
                """, self.user_id, self.conversation_id,
                name, json.dumps(initial_members), json.dumps(goals),
                json.dumps(result.get('resources', {})),
                result.get('stance', 'neutral'),
                result.get('reputation', 0.5))
                
                # Store faction members
                for member_id in initial_members:
                    await conn.execute("""
                        INSERT INTO faction_members (faction_id, member_id)
                        VALUES ($1, $2)
                        ON CONFLICT DO NOTHING
                    """, faction_id, member_id)
            
            return Faction(
                faction_id=faction_id,
                name=name,
                members=initial_members,
                resources=result.get('resources', {}),
                goals=goals,
                strengths=result.get('strengths', []),
                weaknesses=result.get('weaknesses', []),
                current_stance=FactionRole[result.get('stance', 'NEUTRAL').upper()],
                reputation=result.get('reputation', 0.5)
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to create faction: {e}")
            return self._create_fallback_faction(name, initial_members, goals)
    
    async def _get_stakeholder_faction(self, stakeholder_id: int) -> Optional[Faction]:
        """Get the faction a stakeholder belongs to"""
        async with get_db_connection_context() as conn:
            faction_data = await conn.fetchrow("""
                SELECT f.* FROM factions f
                JOIN faction_members fm ON f.faction_id = fm.faction_id
                WHERE fm.member_id = $1
            """, stakeholder_id)
        
        if faction_data:
            return Faction(
                faction_id=faction_data['faction_id'],
                name=faction_data['name'],
                members=json.loads(faction_data['members']),
                resources=json.loads(faction_data['resources']),
                goals=json.loads(faction_data['goals']),
                strengths=[],
                weaknesses=[],
                current_stance=FactionRole[faction_data['stance'].upper()],
                reputation=faction_data['reputation']
            )
        return None
    
    async def _process_faction_action(
        self,
        faction: Faction,
        action_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a faction's action"""
        
        result = {
            'faction_id': faction.faction_id,
            'action_processed': True
        }
        
        # Check for alliance opportunities
        if action_type in ['negotiate', 'ally']:
            alliance_opportunity = await self._check_alliance_opportunity(faction)
            if alliance_opportunity:
                result['triggers_alliance'] = True
                result['alliance'] = alliance_opportunity
        
        # Check for betrayal opportunities
        if action_type in ['attack', 'betray'] and faction.current_stance == FactionRole.OPPORTUNIST:
            betrayal_opportunity = await self._check_betrayal_opportunity(faction)
            if betrayal_opportunity:
                result['triggers_betrayal'] = True
                result['betrayal'] = betrayal_opportunity
        
        return result
    
    async def _check_alliance_opportunity(self, faction: Faction) -> Optional[Dict[str, Any]]:
        """Check if faction should form an alliance"""
        if random.random() < 0.3:  # 30% chance
            # Find potential ally
            async with get_db_connection_context() as conn:
                potential_ally = await conn.fetchrow("""
                    SELECT * FROM factions
                    WHERE user_id = $1 AND conversation_id = $2
                    AND faction_id != $3
                    AND ABS(reputation - $4) < 0.3
                    ORDER BY RANDOM()
                    LIMIT 1
                """, self.user_id, self.conversation_id, faction.faction_id, faction.reputation)
            
            if potential_ally:
                return {
                    'proposer': faction.faction_id,
                    'target': potential_ally['faction_id'],
                    'type': 'temporary'
                }
        return None
    
    async def _check_betrayal_opportunity(self, faction: Faction) -> Optional[Dict[str, Any]]:
        """Check if faction should betray an ally"""
        # Check if faction has alliances
        async with get_db_connection_context() as conn:
            alliance = await conn.fetchrow("""
                SELECT * FROM alliances
                WHERE (faction_id_1 = $1 OR faction_id_2 = $1)
                AND is_active = true
                ORDER BY created_at ASC
                LIMIT 1
            """, faction.faction_id)
        
        if alliance and random.random() < 0.2:  # 20% chance if has alliance
            return {
                'betrayer_id': faction.faction_id,
                'alliance_id': alliance['alliance_id'],
                'type': 'defection'
            }
        return None
    
    async def _check_for_betrayals(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Check for potential betrayals during climax"""
        betrayals = []
        
        # Get factions in conflict
        async with get_db_connection_context() as conn:
            factions = await conn.fetch("""
                SELECT f.* FROM factions f
                JOIN faction_conflicts fc ON f.faction_id = fc.faction_id
                WHERE fc.conflict_id = $1
                AND f.stance = 'opportunist'
            """, conflict_id)
        
        for faction_data in factions:
            if random.random() < 0.4:  # 40% chance for opportunists
                faction = Faction(
                    faction_id=faction_data['faction_id'],
                    name=faction_data['name'],
                    members=json.loads(faction_data['members']),
                    resources=json.loads(faction_data['resources']),
                    goals=json.loads(faction_data['goals']),
                    strengths=[],
                    weaknesses=[],
                    current_stance=FactionRole.OPPORTUNIST,
                    reputation=faction_data['reputation']
                )
                
                betrayal = await self._check_betrayal_opportunity(faction)
                if betrayal:
                    betrayals.append(betrayal)
        
        return betrayals
    
    async def _calculate_power_balance(self) -> Dict[str, float]:
        """Calculate current power balance"""
        async with get_db_connection_context() as conn:
            factions = await conn.fetch("""
                SELECT faction_id, name, reputation, resources
                FROM factions
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        
        power_balance = {}
        for faction in factions:
            resources = json.loads(faction['resources'])
            # Calculate power as combination of reputation and resources
            power = faction['reputation'] * 0.5 + sum(resources.values()) / len(resources) * 0.5
            power_balance[faction['name']] = power
        
        return power_balance
    
    def _create_fallback_faction(
        self,
        name: str,
        members: List[int],
        goals: List[str]
    ) -> Faction:
        """Create fallback faction if LLM fails"""
        
        return Faction(
            faction_id=0,
            name=name,
            members=members,
            resources={'political': 0.5, 'economic': 0.5},
            goals=goals,
            strengths=['Determination'],
            weaknesses=['Limited resources'],
            current_stance=FactionRole.NEUTRAL,
            reputation=0.5
        )


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def create_multi_party_conflict(
    ctx: RunContextWrapper,
    conflict_name: str,
    initial_factions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a new multi-party conflict through synthesizer"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Use synthesizer to create conflict
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Create conflict with multi-party type
    conflict_result = await synthesizer.create_conflict(
        'multi_faction_war',
        {
            'name': conflict_name,
            'factions': initial_factions
        }
    )
    
    return conflict_result


@function_tool
async def faction_take_action(
    ctx: RunContextWrapper,
    faction_id: int,
    conflict_id: int
) -> Dict[str, Any]:
    """Have a faction take its turn through synthesizer"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Use synthesizer to emit stakeholder action
    from logic.conflict_system.conflict_synthesizer import get_synthesizer, SystemEvent, EventType, SubsystemType
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    event = SystemEvent(
        event_id=f"faction_action_{faction_id}",
        event_type=EventType.STAKEHOLDER_ACTION,
        source_subsystem=SubsystemType.MULTIPARTY,
        payload={
            'stakeholder_id': faction_id,
            'conflict_id': conflict_id,
            'action_type': 'faction_turn'
        },
        requires_response=True
    )
    
    responses = await synthesizer.emit_event(event)
    
    for response in responses:
        if response.subsystem == SubsystemType.MULTIPARTY:
            return response.data
    
    return {'error': 'No response from multi-party system'}


@function_tool
async def negotiate_between_factions(
    ctx: RunContextWrapper,
    topic: str,
    participating_factions: List[int],
    initial_position: str
) -> Dict[str, Any]:
    """Start a negotiation between multiple factions"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get synthesizer and multi-party subsystem
    from logic.conflict_system.conflict_synthesizer import get_synthesizer, SubsystemType
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    multiparty_subsystem = synthesizer._subsystems.get(SubsystemType.MULTIPARTY)
    if not multiparty_subsystem:
        return {'error': 'Multi-party subsystem not available'}
    
    # Create negotiation through subsystem
    from logic.conflict_system.multi_party_dynamics import Negotiation
    
    # Store negotiation
    async with get_db_connection_context() as conn:
        negotiation_id = await conn.fetchval("""
            INSERT INTO negotiations
            (user_id, conversation_id, topic, participants, status)
            VALUES ($1, $2, $3, $4, 'active')
            RETURNING negotiation_id
        """, user_id, conversation_id, topic, json.dumps(participating_factions))
    
    negotiation = Negotiation(
        negotiation_id=negotiation_id,
        participants=participating_factions,
        topic=topic,
        offers={p: initial_position for p in participating_factions},
        leverage_in_play={},
        deadline=None,
        mediator=None
    )
    
    return {
        'negotiation_id': negotiation.negotiation_id,
        'topic': negotiation.topic,
        'participants': negotiation.participants,
        'status': 'initiated'
    }
