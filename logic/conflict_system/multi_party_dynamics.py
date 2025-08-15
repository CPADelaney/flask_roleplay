# logic/conflict_system/multi_party_dynamics.py
"""
Multi-Party Dynamics System with LLM-generated content and Relationship Integration
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
from logic.fully_integrated_npc_system import IntegratedNPCSystem
from logic.relationship_integration import RelationshipIntegration

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
    betrayals: List['Betrayal']
    power_balance: 'PowerBalance'
    escalation_level: float  # 0-1
    phase: str  # opening, rising, climax, falling, resolution

@dataclass
class Alliance:
    """Alliance between factions"""
    alliance_id: int
    faction_1: int
    faction_2: int
    alliance_type: AllianceType
    strength: float  # 0-1
    terms: List[str]
    secret: bool
    created_at: datetime
    relationship_based: bool = False  # NEW: whether based on character relationships

@dataclass
class Betrayal:
    """A betrayal in multi-party conflict"""
    betrayal_id: int
    betrayer_id: int  # Faction or NPC ID
    victim_id: int
    betrayal_type: BetrayalType
    impact: float  # 0-1
    revealed: bool
    consequences: List[str]
    relationship_factors: List[str] = field(default_factory=list)  # NEW

@dataclass
class Negotiation:
    """Ongoing negotiation between parties"""
    negotiation_id: int
    participants: List[int]  # Faction IDs
    topic: str
    offers: Dict[int, Any]  # faction_id -> offer
    leverage_in_play: Dict[int, str]  # faction_id -> leverage description
    deadline: Optional[datetime]
    mediator: Optional[int]
    relationship_strength: float = 0.0  # NEW: based on participant relationships

@dataclass 
class PowerBalance:
    """Current power distribution"""
    dominant_faction: Optional[int]
    power_distribution: Dict[int, float]  # faction_id -> power (0-1)
    contested_resources: List[str]
    kingmakers: List[int]  # Factions that can tip balance

# ===============================================================================
# MULTI-PARTY DYNAMICS SUBSYSTEM WITH RELATIONSHIP INTEGRATION
# ===============================================================================

class MultiPartyConflictSubsystem:
    """
    Manages complex multi-faction conflicts with relationship awareness.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize relationship systems
        self.npc_system = IntegratedNPCSystem(user_id, conversation_id)
        self.relationship_integration = RelationshipIntegration(user_id, conversation_id)
        
        # Lazy-loaded agents
        self._faction_strategist = None
        self._alliance_broker = None
        self._betrayal_orchestrator = None
        self._negotiation_mediator = None
        self._outcome_predictor = None
        
        # Reference to synthesizer
        self.synthesizer = None
        
        # Cache for relationship data
        self._relationship_cache = {}
        self._cache_timestamp = None
    
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
            'power_balance_tracking',
            'relationship_based_dynamics'  # NEW
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
    
    async def _get_relationship_factors(self, entity1_id: int, entity2_id: int, 
                                       entity1_type: str = "npc", 
                                       entity2_type: str = "npc") -> Dict[str, Any]:
        """Get relationship factors between two entities."""
        try:
            # Check cache first (5 minute TTL)
            cache_key = f"{entity1_id}:{entity2_id}"
            if cache_key in self._relationship_cache:
                if self._cache_timestamp and (datetime.now() - self._cache_timestamp).seconds < 300:
                    return self._relationship_cache[cache_key]
            
            # Get full relationship data
            relationship = await self.relationship_integration.get_relationship(
                entity1_type, entity1_id,
                entity2_type, entity2_id
            )
            
            if not relationship:
                # Create new relationship if it doesn't exist
                relationship = await self.relationship_integration.create_relationship(
                    entity1_type, entity1_id,
                    entity2_type, entity2_id,
                    relationship_type="faction_member"
                )
            
            # Extract key factors for conflict decisions
            factors = {
                'exists': relationship is not None,
                'trust': relationship.get('trust', 0),
                'closeness': relationship.get('closeness', 0),
                'tension': relationship.get('tension', 0),
                'power_balance': relationship.get('power_balance', 0),
                'loyalty': relationship.get('loyalty', 50),
                'history': relationship.get('interaction_history', []),
                'archetype': relationship.get('archetype', 'neutral')
            }
            
            # Cache the result
            self._relationship_cache[cache_key] = factors
            self._cache_timestamp = datetime.now()
            
            return factors
            
        except Exception as e:
            logger.error(f"Error fetching relationship factors: {e}")
            return {
                'exists': False,
                'trust': 0,
                'closeness': 0,
                'tension': 0,
                'power_balance': 0,
                'loyalty': 50
            }
    
    async def _evaluate_alliance_potential(self, faction1_id: int, faction2_id: int,
                                          faction1_members: List[int], 
                                          faction2_members: List[int]) -> float:
        """Evaluate alliance potential based on member relationships."""
        total_score = 0.0
        relationship_count = 0
        
        # Check relationships between faction members
        for member1 in faction1_members:
            for member2 in faction2_members:
                factors = await self._get_relationship_factors(member1, member2)
                
                if factors['exists']:
                    relationship_count += 1
                    
                    # Calculate alliance score based on relationship
                    score = 0.0
                    
                    # High trust increases alliance potential
                    score += factors['trust'] * 0.3
                    
                    # High closeness increases alliance potential
                    score += factors['closeness'] * 0.2
                    
                    # Low tension increases alliance potential
                    score += (100 - factors['tension']) * 0.2
                    
                    # Loyalty affects alliance stability
                    score += factors['loyalty'] * 0.15
                    
                    # Power balance affects alliance type
                    if abs(factors['power_balance']) < 30:
                        # Balanced power = stable alliance
                        score += 15
                    
                    # Archetype influences
                    if factors['archetype'] in ['allies', 'mentor_student', 'partners']:
                        score += 20
                    elif factors['archetype'] in ['rivals', 'enemies', 'predator_prey']:
                        score -= 30
                    
                    total_score += score
        
        # Average the scores
        if relationship_count > 0:
            return min(100, max(0, total_score / relationship_count))
        
        # No existing relationships = neutral potential
        return 50.0
    
    async def _evaluate_betrayal_likelihood(self, betrayer_id: int, target_faction_id: int,
                                           betrayer_faction_members: List[int],
                                           target_faction_members: List[int]) -> Dict[str, Any]:
        """Evaluate likelihood and type of betrayal based on relationships."""
        betrayal_score = 0.0
        betrayal_factors = []
        suggested_type = None
        
        # Check betrayer's relationships with current faction
        internal_loyalty = 0.0
        for member in betrayer_faction_members:
            if member != betrayer_id:
                factors = await self._get_relationship_factors(betrayer_id, member)
                internal_loyalty += factors['trust'] * 0.5 + factors['loyalty'] * 0.5
        
        if betrayer_faction_members:
            internal_loyalty /= len(betrayer_faction_members)
        
        # Check relationships with target faction
        external_attraction = 0.0
        for member in target_faction_members:
            factors = await self._get_relationship_factors(betrayer_id, member)
            external_attraction += factors['closeness'] * 0.3 + factors['trust'] * 0.2
        
        if target_faction_members:
            external_attraction /= len(target_faction_members)
        
        # Calculate betrayal likelihood
        if internal_loyalty < 30:
            betrayal_score += 30
            betrayal_factors.append("Low loyalty to current faction")
            
        if external_attraction > 70:
            betrayal_score += 25
            betrayal_factors.append("Strong ties to target faction")
            
        if internal_loyalty < external_attraction:
            betrayal_score += 20
            betrayal_factors.append("Stronger external relationships")
        
        # Determine betrayal type based on relationships
        if betrayal_score > 50:
            if internal_loyalty < 20:
                suggested_type = BetrayalType.DEFECTION
            elif external_attraction > 80:
                suggested_type = BetrayalType.DOUBLE_CROSS
            else:
                suggested_type = BetrayalType.INFORMATION_LEAK
        
        return {
            'likelihood': min(100, betrayal_score),
            'factors': betrayal_factors,
            'suggested_type': suggested_type,
            'internal_loyalty': internal_loyalty,
            'external_attraction': external_attraction
        }
    
    async def handle_event(self, event) -> Any:
        """Handle events with relationship awareness."""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
        
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                # Check if this should be a multi-party conflict
                conflict_type = event.payload.get('conflict_type')
                if 'faction' in conflict_type or 'multi' in conflict_type:
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
                # Process faction actions with relationship context
                stakeholder_id = event.payload.get('stakeholder_id')
                action_type = event.payload.get('action_type')
                
                # Get stakeholder's faction
                faction = await self._get_stakeholder_faction(stakeholder_id)
                
                if faction:
                    # Process with relationship awareness
                    result = await self._process_faction_action(
                        faction, action_type, event.payload
                    )
                    
                    # Create side effects based on relationships
                    side_effects = []
                    
                    if result.get('triggers_alliance'):
                        side_effects.append(SystemEvent(
                            event_id=f"alliance_{event.event_id}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=self.subsystem_type,
                            payload={
                                'alliance_formed': True,
                                'alliance_score': result.get('alliance_score', 0),
                                'relationship_based': True
                            },
                            priority=5
                        ))
                    
                    if result.get('triggers_betrayal'):
                        side_effects.append(SystemEvent(
                            event_id=f"betrayal_{event.event_id}",
                            event_type=EventType.CONFLICT_UPDATE,
                            source_subsystem=self.subsystem_type,
                            payload={
                                'betrayal_occurred': True,
                                'betrayal_type': result.get('suggested_type', BetrayalType.DEFECTION).value,
                                'betrayal_factors': result.get('factors', [])
                            },
                            priority=8
                        ))
                    
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data=result,
                        side_effects=side_effects
                    )
            
            elif event.event_type == EventType.PHASE_TRANSITION:
                # Update faction dynamics based on phase
                conflict_id = event.payload.get('conflict_id')
                new_phase = event.payload.get('phase')
                
                if new_phase == 'climax':
                    # Time for betrayals and power plays based on relationships
                    betrayals = await self._check_for_betrayals(conflict_id)
                    
                    side_effects = []
                    for betrayal in betrayals:
                        side_effects.append(SystemEvent(
                            event_id=f"betrayal_{event.event_id}_{betrayal['betrayer_id']}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=self.subsystem_type,
                            payload=betrayal,
                            priority=2
                        ))
                    
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'betrayals_triggered': len(betrayals)},
                        side_effects=side_effects
                    )
            
            # Default response
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'processed': True}
            )
            
        except Exception as e:
            logger.error(f"Error handling event in multiparty subsystem: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def _initialize_factions(self, conflict_id: int) -> List[Faction]:
        """Initialize factions for a new conflict"""
        # Implementation would create factions based on conflict type
        return []
    
    async def _get_stakeholder_faction(self, stakeholder_id: int) -> Optional[Faction]:
        """Get faction for a stakeholder"""
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
    
    async def _process_faction_action(self, faction: Faction, 
                                     action_type: str, 
                                     payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process faction action with relationship awareness."""
        result = {
            'action_processed': True,
            'action_type': action_type,
            'faction_id': faction.faction_id
        }
        
        # Get faction members
        faction_members = faction.members
        
        if action_type == 'propose_alliance':
            target_faction_id = payload.get('target_faction_id')
            target_faction = await self._get_faction_by_id(target_faction_id)
            
            if target_faction:
                # Evaluate based on relationships
                alliance_score = await self._evaluate_alliance_potential(
                    faction.faction_id, target_faction_id,
                    faction_members, target_faction.members
                )
                
                result['alliance_score'] = alliance_score
                result['triggers_alliance'] = alliance_score > 65
                
                if result['triggers_alliance']:
                    # Strengthen relationships between allied factions
                    for member1 in faction_members:
                        for member2 in target_faction.members:
                            await self.relationship_integration.update_relationship(
                                "npc", member1, "npc", member2,
                                {"trust": 10, "closeness": 5}
                            )
            
        elif action_type == 'consider_betrayal':
            betrayer_id = payload.get('potential_betrayer_id')
            target_faction_id = payload.get('target_faction_id')
            target_faction = await self._get_faction_by_id(target_faction_id)
            
            if target_faction:
                betrayal_eval = await self._evaluate_betrayal_likelihood(
                    betrayer_id, target_faction_id,
                    faction_members, target_faction.members
                )
                
                result.update(betrayal_eval)
                result['triggers_betrayal'] = betrayal_eval['likelihood'] > 70
                
                if result['triggers_betrayal']:
                    # Update relationships to reflect betrayal
                    for member in faction_members:
                        if member != betrayer_id:
                            await self.relationship_integration.update_relationship(
                                "npc", betrayer_id, "npc", member,
                                {"trust": -30, "tension": 20}
                            )
        
        elif action_type == 'faction_negotiation':
            # Consider relationships in negotiation positions
            negotiation_strength = await self._calculate_negotiation_strength(faction)
            result['negotiation_strength'] = negotiation_strength
            result['leverage_from_relationships'] = negotiation_strength > 60
        
        return result
    
    async def _get_faction_by_id(self, faction_id: int) -> Optional[Faction]:
        """Get faction by ID"""
        async with get_db_connection_context() as conn:
            faction_data = await conn.fetchrow("""
                SELECT * FROM factions
                WHERE faction_id = $1
            """, faction_id)
        
        if faction_data:
            return Faction(
                faction_id=faction_data['faction_id'],
                name=faction_data['name'],
                members=json.loads(faction_data['members']),
                resources=json.loads(faction_data['resources']),
                goals=json.loads(faction_data['goals']),
                strengths=json.loads(faction_data.get('strengths', '[]')),
                weaknesses=json.loads(faction_data.get('weaknesses', '[]')),
                current_stance=FactionRole[faction_data['stance'].upper()],
                reputation=faction_data['reputation']
            )
        return None
    
    async def _calculate_negotiation_strength(self, faction: Faction) -> float:
        """Calculate negotiation strength based on relationships"""
        strength = 0.0
        
        for member in faction.members:
            # Check member's relationships with other factions
            async with get_db_connection_context() as conn:
                relationships = await conn.fetch("""
                    SELECT AVG(trust) as avg_trust, AVG(power_balance) as avg_power
                    FROM relationship_states
                    WHERE (entity1_id = $1 AND entity1_type = 'npc')
                    OR (entity2_id = $1 AND entity2_type = 'npc')
                """, member)
                
                if relationships:
                    for row in relationships:
                        if row['avg_trust']:
                            strength += row['avg_trust'] * 0.3
                        if row['avg_power']:
                            strength += abs(row['avg_power']) * 0.2
        
        if faction.members:
            strength /= len(faction.members)
        
        return min(100, strength)
    
    async def _check_for_betrayals(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Check for potential betrayals based on relationships during climax"""
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
            
            # Evaluate betrayal opportunities based on relationships
            for member in faction.members:
                # Find potential target factions
                target_factions = await conn.fetch("""
                    SELECT f.* FROM factions f
                    JOIN faction_conflicts fc ON f.faction_id = fc.faction_id
                    WHERE fc.conflict_id = $1
                    AND f.faction_id != $2
                """, conflict_id, faction.faction_id)
                
                for target_data in target_factions:
                    target_faction = await self._get_faction_by_id(target_data['faction_id'])
                    if target_faction:
                        betrayal_eval = await self._evaluate_betrayal_likelihood(
                            member, target_faction.faction_id,
                            faction.members, target_faction.members
                        )
                        
                        if betrayal_eval['likelihood'] > 70:
                            betrayals.append({
                                'betrayer_id': member,
                                'source_faction': faction.faction_id,
                                'target_faction': target_faction.faction_id,
                                'type': betrayal_eval['suggested_type'].value,
                                'factors': betrayal_eval['factors']
                            })
                            break  # One betrayal per member
        
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
            # Power based on resources and reputation
            power = faction['reputation'] * 0.5 + sum(resources.values()) * 0.5
            power_balance[faction['name']] = power
        
        return power_balance
    
    async def get_conflict_state(self, conflict_id: int) -> Dict[str, Any]:
        """Get current state of multi-party conflict"""
        async with get_db_connection_context() as conn:
            # Get factions
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
                Consider faction resources, goals, and relationships.
                Balance aggression with diplomacy.
                Create compelling power dynamics.
                """,
                tools=[],
                model=ModelSettings(model="gpt-4o", temperature=0.8)
            )
        return self._faction_strategist
    
    @property
    def alliance_broker(self) -> Agent:
        """Agent for alliance negotiations"""
        if self._alliance_broker is None:
            self._alliance_broker = Agent(
                name="Alliance Broker",
                instructions="""
                Broker alliances between factions based on:
                - Shared interests and enemies
                - Power balance considerations
                - Trust and past relationships
                - Resource complementarity
                Create realistic alliance terms and conditions.
                """,
                tools=[],
                model=ModelSettings(model="gpt-4o", temperature=0.7)
            )
        return self._alliance_broker
    
    @property
    def betrayal_orchestrator(self) -> Agent:
        """Agent for betrayal scenarios"""
        if self._betrayal_orchestrator is None:
            self._betrayal_orchestrator = Agent(
                name="Betrayal Orchestrator",
                instructions="""
                Orchestrate dramatic betrayals based on:
                - Character motivations and relationships
                - Opportune timing
                - Maximum narrative impact
                - Realistic consequences
                Balance shock value with believability.
                """,
                tools=[],
                model=ModelSettings(model="gpt-4o", temperature=0.9)
            )
        return self._betrayal_orchestrator
    
    @property
    def negotiation_mediator(self) -> Agent:
        """Agent for multi-party negotiations"""
        if self._negotiation_mediator is None:
            self._negotiation_mediator = Agent(
                name="Negotiation Mediator",
                instructions="""
                Mediate complex negotiations between multiple parties.
                Consider each faction's leverage and goals.
                Create realistic compromises and deals.
                Account for relationship dynamics.
                """,
                tools=[],
                model=ModelSettings(model="gpt-4o", temperature=0.6)
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
                Consider power dynamics, alliances, and wild cards.
                Generate multiple possible scenarios.
                Account for relationship-based factors.
                """,
                tools=[],
                model=ModelSettings(model="gpt-4o", temperature=0.5)
            )
        return self._outcome_predictor

# ===============================================================================
# FUNCTION TOOLS WITH RELATIONSHIP INTEGRATION
# ===============================================================================

@function_tool
async def initialize_multi_faction_conflict(
    ctx: RunContextWrapper,
    conflict_name: str,
    initial_factions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Initialize a multi-faction conflict through synthesizer with relationship awareness"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get synthesizer
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Initialize relationship integration for faction creation
    rel_integration = RelationshipIntegration(user_id, conversation_id)
    
    # Enhance faction data with relationship information
    for faction in initial_factions:
        if 'members' in faction:
            # Calculate internal cohesion based on member relationships
            cohesion = 0.0
            member_count = 0
            
            for i, member1 in enumerate(faction['members']):
                for member2 in faction['members'][i+1:]:
                    try:
                        rel = await rel_integration.get_relationship(
                            "npc", member1, "npc", member2
                        )
                        if rel:
                            cohesion += rel.get('trust', 0) + rel.get('closeness', 0)
                            member_count += 1
                    except:
                        pass
            
            if member_count > 0:
                faction['internal_cohesion'] = cohesion / (member_count * 2)  # Normalize
            else:
                faction['internal_cohesion'] = 50.0
    
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
    """Have a faction take its turn through synthesizer with relationship awareness"""
    
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
            'action_type': 'faction_turn',
            'consider_relationships': True  # NEW flag
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
    """Start a negotiation between multiple factions with relationship-based leverage"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get synthesizer and multi-party subsystem
    from logic.conflict_system.conflict_synthesizer import get_synthesizer, SubsystemType
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    multiparty_subsystem = synthesizer._subsystems.get(SubsystemType.MULTIPARTY)
    if not multiparty_subsystem:
        return {'error': 'Multi-party subsystem not available'}
    
    # Calculate relationship-based negotiation strength for each faction
    rel_integration = RelationshipIntegration(user_id, conversation_id)
    faction_strengths = {}
    
    for faction_id in participating_factions:
        # Get faction members
        async with get_db_connection_context() as conn:
            members = await conn.fetch("""
                SELECT member_id FROM faction_members
                WHERE faction_id = $1
            """, faction_id)
        
        # Calculate average relationship quality with other factions
        total_strength = 0.0
        relationship_count = 0
        
        for member_row in members:
            member_id = member_row['member_id']
            # Check relationships with members of other factions
            for other_faction_id in participating_factions:
                if other_faction_id != faction_id:
                    other_members = await conn.fetch("""
                        SELECT member_id FROM faction_members
                        WHERE faction_id = $1
                    """, other_faction_id)
                    
                    for other_row in other_members:
                        try:
                            rel = await rel_integration.get_relationship(
                                "npc", member_id, "npc", other_row['member_id']
                            )
                            if rel:
                                # Power balance and trust affect negotiation
                                strength = rel.get('power_balance', 0) * 0.3 + rel.get('trust', 0) * 0.2
                                total_strength += strength
                                relationship_count += 1
                        except:
                            pass
        
        if relationship_count > 0:
            faction_strengths[faction_id] = total_strength / relationship_count
        else:
            faction_strengths[faction_id] = 50.0
    
    # Store negotiation with relationship data
    async with get_db_connection_context() as conn:
        negotiation_id = await conn.fetchval("""
            INSERT INTO negotiations
            (user_id, conversation_id, topic, participants, status, metadata)
            VALUES ($1, $2, $3, $4, 'active', $5)
            RETURNING negotiation_id
        """, user_id, conversation_id, topic, json.dumps(participating_factions),
            json.dumps({'faction_strengths': faction_strengths}))
    
    negotiation = Negotiation(
        negotiation_id=negotiation_id,
        participants=participating_factions,
        topic=topic,
        offers={p: initial_position for p in participating_factions},
        leverage_in_play={},
        deadline=None,
        mediator=None,
        relationship_strength=max(faction_strengths.values()) if faction_strengths else 50.0
    )
    
    return {
        'negotiation_id': negotiation.negotiation_id,
        'topic': negotiation.topic,
        'participants': negotiation.participants,
        'status': 'initiated',
        'faction_strengths': faction_strengths,
        'relationship_factors_considered': True
    }
