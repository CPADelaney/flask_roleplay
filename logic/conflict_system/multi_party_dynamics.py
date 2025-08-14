# logic/conflict_system/multi_party_dynamics.py
"""
Multi-Party Dynamics System with LLM-generated content.
Manages complex conflicts involving multiple factions, alliances, and betrayals.
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
# MULTI-PARTY MANAGER WITH LLM
# ===============================================================================

class MultiPartyDynamicsManager:
    """
    Manages complex multi-party conflicts using LLM for dynamic generation.
    Handles faction interactions, alliances, betrayals, and negotiations.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._faction_strategist = None
        self._alliance_broker = None
        self._betrayal_orchestrator = None
        self._negotiation_mediator = None
        self._outcome_predictor = None
    
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
    
    # ========== Faction Management ==========
    
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
    
    async def determine_faction_action(
        self,
        faction: Faction,
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine what action a faction should take"""
        
        prompt = f"""
        Determine faction action:
        
        Faction: {faction.name}
        Current Stance: {faction.current_stance.value}
        Resources: {json.dumps(faction.resources)}
        Goals: {json.dumps(faction.goals)}
        
        Conflict State:
        {json.dumps(conflict_state, indent=2)}
        
        Determine:
        1. Primary action (attack/defend/negotiate/ally/betray/wait)
        2. Target (if applicable)
        3. Resources to commit (0-1 for each type)
        4. Rationale
        5. Success probability (0-1)
        6. Risks
        
        Make the decision strategically sound.
        Format as JSON.
        """
        
        response = await self.faction_strategist.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            return {
                'faction_id': faction.faction_id,
                'action': result.get('action', 'wait'),
                'target': result.get('target'),
                'resources_committed': result.get('resources', {}),
                'rationale': result.get('rationale', 'Strategic decision'),
                'success_probability': result.get('success_probability', 0.5),
                'risks': result.get('risks', [])
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to determine faction action: {e}")
            return {'faction_id': faction.faction_id, 'action': 'wait'}
    
    # ========== Alliance System ==========
    
    async def propose_alliance(
        self,
        proposer_id: int,
        target_id: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate an alliance proposal"""
        
        # Get faction details
        proposer = await self._get_faction_details(proposer_id)
        target = await self._get_faction_details(target_id)
        
        prompt = f"""
        Generate alliance proposal:
        
        Proposer: {proposer.get('name', 'Unknown')}
        Target: {target.get('name', 'Unknown')}
        Context: {json.dumps(context, indent=2)}
        
        Create:
        1. Alliance type (formal/secret/temporary/defensive/opportunistic)
        2. Specific terms (what each party provides/gains)
        3. Duration/expiration condition
        4. Enforcement mechanisms
        5. Exit clauses
        6. Why target should accept
        
        Make it mutually beneficial but realistic.
        Format as JSON.
        """
        
        response = await self.alliance_broker.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store proposal
            async with get_db_connection_context() as conn:
                proposal_id = await conn.fetchval("""
                    INSERT INTO alliance_proposals
                    (user_id, conversation_id, proposer_id, target_id, 
                     alliance_type, terms, status)
                    VALUES ($1, $2, $3, $4, $5, $6, 'pending')
                    RETURNING proposal_id
                """, self.user_id, self.conversation_id,
                proposer_id, target_id,
                result.get('type', 'temporary'),
                json.dumps(result.get('terms', {})))
            
            return {
                'proposal_id': proposal_id,
                'type': result.get('type'),
                'terms': result.get('terms'),
                'expiration': result.get('expiration'),
                'enforcement': result.get('enforcement'),
                'exit_clauses': result.get('exit_clauses', []),
                'pitch': result.get('pitch', 'Mutual benefit')
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to propose alliance: {e}")
            return {'error': 'Failed to generate proposal'}
    
    async def evaluate_alliance_proposal(
        self,
        faction_id: int,
        proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate whether to accept an alliance proposal"""
        
        faction = await self._get_faction_details(faction_id)
        
        prompt = f"""
        Evaluate alliance proposal:
        
        Evaluating Faction: {faction.get('name')}
        Goals: {faction.get('goals')}
        Current Alliances: {faction.get('alliances', [])}
        
        Proposal:
        {json.dumps(proposal, indent=2)}
        
        Determine:
        1. Should accept? (yes/no/counter)
        2. Benefits vs costs analysis
        3. Trust level in proposer (0-1)
        4. Conditions or modifications needed
        5. Long-term implications
        
        Be strategically minded.
        Format as JSON.
        """
        
        response = await self.alliance_broker.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            return {
                'decision': result.get('accept', 'no'),
                'analysis': result.get('analysis'),
                'trust_level': result.get('trust', 0.5),
                'conditions': result.get('conditions', []),
                'implications': result.get('implications', [])
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to evaluate proposal: {e}")
            return {'decision': 'no', 'reason': 'Evaluation failed'}
    
    # ========== Betrayal System ==========
    
    async def plan_betrayal(
        self,
        betrayer_id: int,
        target_alliance_id: int,
        motivation: str
    ) -> Dict[str, Any]:
        """Plan a betrayal with timing and method"""
        
        betrayer = await self._get_faction_details(betrayer_id)
        alliance = await self._get_alliance_details(target_alliance_id)
        
        prompt = f"""
        Plan a betrayal:
        
        Betrayer: {betrayer.get('name')}
        Current Alliance: {json.dumps(alliance, indent=2)}
        Motivation: {motivation}
        
        Generate:
        1. Betrayal type (defection/sabotage/leak/abandonment/double-cross)
        2. Optimal timing (immediate/specific trigger/gradual)
        3. Method of execution
        4. Cover story/justification
        5. Expected gains
        6. Risk of discovery (0-1)
        7. Potential backlash
        
        Make it cunning but not cartoonishly evil.
        Format as JSON.
        """
        
        response = await self.betrayal_orchestrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            return {
                'betrayal_type': result.get('type'),
                'timing': result.get('timing'),
                'method': result.get('method'),
                'cover_story': result.get('cover_story'),
                'expected_gains': result.get('gains'),
                'discovery_risk': result.get('discovery_risk', 0.5),
                'potential_backlash': result.get('backlash', [])
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to plan betrayal: {e}")
            return {'error': 'Failed to plan betrayal'}
    
    async def execute_betrayal(
        self,
        betrayal_plan: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a betrayal and generate consequences"""
        
        prompt = f"""
        Execute betrayal:
        
        Plan: {json.dumps(betrayal_plan, indent=2)}
        Context: {json.dumps(execution_context, indent=2)}
        
        Determine:
        1. Success level (0-1)
        2. Immediate consequences
        3. Who discovers the betrayal
        4. Faction reactions
        5. Power balance shift
        6. Narrative description (2-3 sentences)
        
        Make consequences proportional and realistic.
        Format as JSON.
        """
        
        response = await self.betrayal_orchestrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store betrayal
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO betrayals
                    (user_id, conversation_id, betrayer_id, betrayal_type, 
                     success_level, consequences)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, self.user_id, self.conversation_id,
                execution_context.get('betrayer_id'),
                betrayal_plan.get('betrayal_type'),
                result.get('success', 0.5),
                json.dumps(result.get('consequences', {})))
            
            return {
                'success': result.get('success', 0.5),
                'immediate_consequences': result.get('consequences'),
                'discovered_by': result.get('discovered_by', []),
                'faction_reactions': result.get('reactions', {}),
                'power_shift': result.get('power_shift', {}),
                'narrative': result.get('narrative', 'The betrayal unfolds')
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to execute betrayal: {e}")
            return {'success': 0.0, 'narrative': 'The betrayal fails'}
    
    # ========== Negotiation System ==========
    
    async def initiate_negotiation(
        self,
        initiator_id: int,
        participants: List[int],
        topic: str,
        initial_position: str
    ) -> Negotiation:
        """Start a multi-party negotiation"""
        
        # Get participant details
        participant_details = await self._get_multiple_faction_details(participants)
        
        prompt = f"""
        Set up negotiation:
        
        Initiator: Faction {initiator_id}
        Participants: {json.dumps(participant_details, indent=2)}
        Topic: {topic}
        Initial Position: {initial_position}
        
        Generate:
        1. Each party's opening position
        2. Their hidden bottom lines
        3. Leverage they might use
        4. Potential compromise points
        5. Deal breakers for each party
        
        Make positions realistic and negotiable.
        Format as JSON.
        """
        
        response = await self.negotiation_mediator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store negotiation
            async with get_db_connection_context() as conn:
                negotiation_id = await conn.fetchval("""
                    INSERT INTO negotiations
                    (user_id, conversation_id, topic, participants, status)
                    VALUES ($1, $2, $3, $4, 'active')
                    RETURNING negotiation_id
                """, self.user_id, self.conversation_id,
                topic, json.dumps(participants))
            
            return Negotiation(
                negotiation_id=negotiation_id,
                participants=participants,
                topic=topic,
                offers={p: result.get('positions', {}).get(str(p), '') for p in participants},
                leverage_in_play=result.get('leverage', {}),
                deadline=None,
                mediator=None
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to initiate negotiation: {e}")
            return self._create_fallback_negotiation(participants, topic)
    
    async def advance_negotiation(
        self,
        negotiation: Negotiation,
        new_offer: Dict[int, str]
    ) -> Dict[str, Any]:
        """Advance negotiation with new offers and counter-offers"""
        
        prompt = f"""
        Advance negotiation:
        
        Topic: {negotiation.topic}
        Current Offers: {json.dumps(negotiation.offers, indent=2)}
        New Offer: {json.dumps(new_offer, indent=2)}
        
        Generate:
        1. Reactions from each party
        2. Counter-offers
        3. Movement toward agreement (0-1)
        4. Sticking points remaining
        5. Breakthrough potential
        
        Show realistic give-and-take.
        Format as JSON.
        """
        
        response = await self.negotiation_mediator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Update negotiation
            negotiation.offers.update(new_offer)
            for party_id, counter in result.get('counter_offers', {}).items():
                negotiation.offers[int(party_id)] = counter
            
            return {
                'reactions': result.get('reactions', {}),
                'counter_offers': result.get('counter_offers', {}),
                'progress': result.get('progress', 0.5),
                'sticking_points': result.get('sticking_points', []),
                'breakthrough_possible': result.get('breakthrough', False)
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to advance negotiation: {e}")
            return {'progress': 0.0}
    
    # ========== Outcome Prediction ==========
    
    async def predict_conflict_outcomes(
        self,
        conflict: MultiPartyConflict
    ) -> List[Dict[str, Any]]:
        """Predict possible outcomes of the conflict"""
        
        # Prepare conflict summary
        conflict_summary = {
            'factions': {f_id: self._faction_to_dict(f) for f_id, f in conflict.factions.items()},
            'alliances': [self._alliance_to_dict(a) for a in conflict.alliances],
            'power_balance': conflict.power_balance,
            'escalation': conflict.escalation_level
        }
        
        prompt = f"""
        Predict conflict outcomes:
        
        Conflict State:
        {json.dumps(conflict_summary, indent=2)}
        
        Generate 3-4 possible outcomes:
        For each outcome provide:
        1. Description (what happens)
        2. Probability (0-1)
        3. Winners and losers
        4. Long-term consequences
        5. Trigger conditions
        
        Include both expected and surprising possibilities.
        Format as JSON array.
        """
        
        response = await self.outcome_predictor.run(prompt)
        
        try:
            outcomes = json.loads(response.content)
            
            return [
                {
                    'description': o.get('description'),
                    'probability': o.get('probability', 0.25),
                    'winners': o.get('winners', []),
                    'losers': o.get('losers', []),
                    'consequences': o.get('consequences', []),
                    'triggers': o.get('triggers', [])
                }
                for o in outcomes
            ]
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to predict outcomes: {e}")
            return []
    
    # ========== Helper Methods ==========
    
    async def _get_faction_details(self, faction_id: int) -> Dict:
        """Get faction details"""
        
        async with get_db_connection_context() as conn:
            faction = await conn.fetchrow("""
                SELECT * FROM factions WHERE faction_id = $1
            """, faction_id)
        
        return dict(faction) if faction else {}
    
    async def _get_multiple_faction_details(self, faction_ids: List[int]) -> List[Dict]:
        """Get details for multiple factions"""
        
        details = []
        for f_id in faction_ids:
            details.append(await self._get_faction_details(f_id))
        return details
    
    async def _get_alliance_details(self, alliance_id: int) -> Dict:
        """Get alliance details"""
        
        async with get_db_connection_context() as conn:
            alliance = await conn.fetchrow("""
                SELECT * FROM alliances WHERE alliance_id = $1
            """, alliance_id)
        
        return dict(alliance) if alliance else {}
    
    def _faction_to_dict(self, faction: Faction) -> Dict:
        """Convert faction to dict"""
        
        return {
            'name': faction.name,
            'members': faction.members,
            'resources': faction.resources,
            'stance': faction.current_stance.value,
            'reputation': faction.reputation
        }
    
    def _alliance_to_dict(self, alliance: Alliance) -> Dict:
        """Convert alliance to dict"""
        
        return {
            'type': alliance.alliance_type.value,
            'members': list(alliance.members),
            'strength': alliance.strength,
            'secret': alliance.secret
        }
    
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
    
    def _create_fallback_negotiation(
        self,
        participants: List[int],
        topic: str
    ) -> Negotiation:
        """Create fallback negotiation if LLM fails"""
        
        return Negotiation(
            negotiation_id=0,
            participants=participants,
            topic=topic,
            offers={},
            leverage_in_play={},
            deadline=None,
            mediator=None
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
    """Create a new multi-party conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = MultiPartyDynamicsManager(user_id, conversation_id)
    
    # Create factions
    factions = {}
    for f_data in initial_factions:
        faction = await manager.create_faction(
            f_data['name'],
            f_data.get('members', []),
            f_data.get('goals', [])
        )
        factions[faction.faction_id] = faction
    
    # Store conflict
    async with get_db_connection_context() as conn:
        conflict_id = await conn.fetchval("""
            INSERT INTO multi_party_conflicts
            (user_id, conversation_id, name, faction_ids, escalation_level)
            VALUES ($1, $2, $3, $4, 0.3)
            RETURNING conflict_id
        """, user_id, conversation_id,
        conflict_name, json.dumps(list(factions.keys())))
    
    return {
        'conflict_id': conflict_id,
        'name': conflict_name,
        'factions': {
            f_id: {
                'name': f.name,
                'stance': f.current_stance.value,
                'reputation': f.reputation
            }
            for f_id, f in factions.items()
        }
    }

@function_tool
async def faction_take_action(
    ctx: RunContextWrapper,
    faction_id: int,
    conflict_id: int
) -> Dict[str, Any]:
    """Have a faction take its turn"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = MultiPartyDynamicsManager(user_id, conversation_id)
    
    # Get faction and conflict state
    faction_data = await manager._get_faction_details(faction_id)
    
    # Create faction object
    faction = Faction(
        faction_id=faction_id,
        name=faction_data['name'],
        members=json.loads(faction_data.get('members', '[]')),
        resources=json.loads(faction_data.get('resources', '{}')),
        goals=json.loads(faction_data.get('goals', '[]')),
        strengths=[],
        weaknesses=[],
        current_stance=FactionRole[faction_data.get('stance', 'NEUTRAL').upper()],
        reputation=faction_data.get('reputation', 0.5)
    )
    
    # Get conflict state
    conflict_state = {'conflict_id': conflict_id}  # Would be more detailed
    
    action = await manager.determine_faction_action(faction, conflict_state)
    return action

@function_tool
async def propose_faction_alliance(
    ctx: RunContextWrapper,
    proposer_faction: int,
    target_faction: int,
    reason: str
) -> Dict[str, Any]:
    """Propose an alliance between factions"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = MultiPartyDynamicsManager(user_id, conversation_id)
    
    context = {'reason': reason}
    proposal = await manager.propose_alliance(proposer_faction, target_faction, context)
    
    # Auto-evaluate for NPC factions
    if target_faction != user_id:  # If target is NPC faction
        evaluation = await manager.evaluate_alliance_proposal(target_faction, proposal)
        proposal['target_response'] = evaluation
    
    return proposal

@function_tool
async def betray_alliance(
    ctx: RunContextWrapper,
    betrayer_faction: int,
    alliance_id: int,
    motivation: str
) -> Dict[str, Any]:
    """Plan and potentially execute a betrayal"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = MultiPartyDynamicsManager(user_id, conversation_id)
    
    # Plan betrayal
    plan = await manager.plan_betrayal(betrayer_faction, alliance_id, motivation)
    
    # Optionally auto-execute based on timing
    if plan.get('timing') == 'immediate':
        execution = await manager.execute_betrayal(
            plan,
            {'betrayer_id': betrayer_faction}
        )
        plan['execution'] = execution
    
    return plan

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
    
    manager = MultiPartyDynamicsManager(user_id, conversation_id)
    
    # Determine initiator (first in list)
    initiator = participating_factions[0] if participating_factions else user_id
    
    negotiation = await manager.initiate_negotiation(
        initiator,
        participating_factions,
        topic,
        initial_position
    )
    
    return {
        'negotiation_id': negotiation.negotiation_id,
        'topic': negotiation.topic,
        'participants': negotiation.participants,
        'opening_positions': negotiation.offers
    }
