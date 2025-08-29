# logic/conflict_system/multi_party_dynamics.py
"""
Multi-Party Dynamics System with LLM-generated content and Relationship Integration
Integrated with ConflictSynthesizer as a registered subsystem (circular-safe).
"""

import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple, TypedDict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from logic.fully_integrated_npc_system import IntegratedNPCSystem
from logic.relationship_integration import RelationshipIntegration

logger = logging.getLogger(__name__)

# Lazy orchestrator types (avoid circular imports at module load time)
def _orch():
    from logic.conflict_system.conflict_synthesizer import (
        SubsystemType, EventType, SystemEvent, SubsystemResponse
    )
    return SubsystemType, EventType, SystemEvent, SubsystemResponse


# ===============================================================================
# MULTI-PARTY STRUCTURES
# ===============================================================================

class InitializeMultiFactionConflictResponse(TypedDict):
    conflict_id: int
    conflict_name: str
    status: str
    factions_registered: int
    message: str
    error: str

class FactionTakeActionResponse(TypedDict):
    performed: bool
    action_type: str
    stakeholder_id: int
    effects: List[str]
    notes: str
    error: str

class FactionStrengthItem(TypedDict):
    faction_id: int
    strength: float

class NegotiateBetweenFactionsResponse(TypedDict):
    negotiation_id: int
    topic: str
    participants: List[int]
    status: str
    faction_strengths: List[FactionStrengthItem]
    relationship_factors_considered: bool
    error: str


class FactionRole(Enum):
    AGGRESSOR = "aggressor"
    DEFENDER = "defender"
    MEDIATOR = "mediator"
    OPPORTUNIST = "opportunist"
    NEUTRAL = "neutral"
    WILDCARD = "wildcard"
    KINGMAKER = "kingmaker"

class AllianceType(Enum):
    FORMAL = "formal"
    SECRET = "secret"
    TEMPORARY = "temporary"
    DEFENSIVE = "defensive"
    OPPORTUNISTIC = "opportunistic"

class BetrayalType(Enum):
    DEFECTION = "defection"
    SABOTAGE = "sabotage"
    INFORMATION_LEAK = "information_leak"
    ABANDONMENT = "abandonment"
    DOUBLE_CROSS = "double_cross"

@dataclass
class Faction:
    faction_id: int
    name: str
    members: List[int]            # NPC/Player IDs
    resources: Dict[str, float]
    goals: List[str]
    strengths: List[str]
    weaknesses: List[str]
    current_stance: FactionRole
    reputation: float             # 0-1 (or 0-100 in DB; normalize as needed)

@dataclass
class MultiPartyConflict:
    conflict_id: int
    name: str
    factions: Dict[int, Faction]
    alliances: List['Alliance']
    active_negotiations: List['Negotiation']
    betrayals: List['Betrayal']
    power_balance: 'PowerBalance'
    escalation_level: float       # 0-1
    phase: str

@dataclass
class Alliance:
    alliance_id: int
    faction_1: int
    faction_2: int
    alliance_type: AllianceType
    strength: float
    terms: List[str]
    secret: bool
    created_at: datetime
    relationship_based: bool = False

@dataclass
class Betrayal:
    betrayal_id: int
    betrayer_id: int
    victim_id: int
    betrayal_type: BetrayalType
    impact: float
    revealed: bool
    consequences: List[str]
    relationship_factors: List[str] = field(default_factory=list)

@dataclass
class Negotiation:
    negotiation_id: int
    participants: List[int]
    topic: str
    offers: Dict[int, Any]
    leverage_in_play: Dict[int, str]
    deadline: Optional[datetime]
    mediator: Optional[int]
    relationship_strength: float = 0.0

@dataclass 
class PowerBalance:
    dominant_faction: Optional[int]
    power_distribution: Dict[int, float]
    contested_resources: List[str]
    kingmakers: List[int]


# ===============================================================================
# MULTI-PARTY DYNAMICS SUBSYSTEM (duck-typed; orchestrator-friendly)
# ===============================================================================

class MultiPartyConflictSubsystem:
    """
    Manages complex multi-faction conflicts with relationship awareness.
    Registered with ConflictSynthesizer (requires SubsystemType.MULTIPARTY).
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        self.npc_system = IntegratedNPCSystem(user_id, conversation_id)
        self.relationship_integration = RelationshipIntegration(user_id, conversation_id)
        
        self._faction_strategist = None
        self._alliance_broker = None
        self._betrayal_orchestrator = None
        self._negotiation_mediator = None
        self._outcome_predictor = None
        
        self.synthesizer = None  # weakref set in initialize
        
        self._relationship_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Optional[datetime] = None
    
    # ----- Subsystem interface -----
    
    @property
    def subsystem_type(self):
        SubsystemType, _, _, _ = _orch()
        return SubsystemType.MULTIPARTY
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'faction_management',
            'alliance_formation',
            'betrayal_orchestration',
            'negotiation_mediation',
            'outcome_prediction',
            'power_balance_tracking',
            'relationship_based_dynamics'
        }
    
    @property
    def dependencies(self) -> Set:
        SubsystemType, _, _, _ = _orch()
        return {SubsystemType.STAKEHOLDER, SubsystemType.LEVERAGE, SubsystemType.SOCIAL}
    
    @property
    def event_subscriptions(self) -> Set:
        _, EventType, _, _ = _orch()
        return {
            EventType.CONFLICT_CREATED,
            EventType.STAKEHOLDER_ACTION,
            EventType.PHASE_TRANSITION,
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    # Optional: lightweight scene bundle for orchestrator merges
    async def get_scene_bundle(self, scope) -> Dict[str, Any]:
        try:
            # If multiple factions likely present, surface an opportunity
            npc_ids = list(getattr(scope, "npc_ids", []) or [])
            ambient, opportunities = [], []
            if npc_ids and len(npc_ids) >= 3:
                ambient.append("factions_take_notice")
                opportunities.append({
                    'type': 'faction_tension',
                    'description': 'Multiple faction-aligned NPCs in proximity'
                })
            return {
                'ambient_effects': ambient,
                'opportunities': opportunities,
                'last_changed_at': datetime.now().timestamp()
            }
        except Exception:
            return {}
    
    # ----- Event handling -----
    
    async def handle_event(self, event) -> Any:
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                conflict_type = (event.payload or {}).get('conflict_type', '') or ''
                ctx = (event.payload or {}).get('context', {}) or {}
                # Detect multiparty by type or context hints
                is_multi = any(k in conflict_type.lower() for k in ['faction', 'multi'])
                is_multi = is_multi or bool(ctx.get('factions'))
                if is_multi:
                    # Initialize factions if provided in context
                    factions_seed = ctx.get('factions') or []
                    initialized = await self._initialize_factions(event.payload.get('conflict_id'), factions_seed)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'multi_party_initialized': True, 'faction_count': len(initialized)},
                        side_effects=[]
                    )
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'multi_party_skipped': True},
                    side_effects=[]
                )
            
            if event.event_type == EventType.STAKEHOLDER_ACTION:
                payload = event.payload or {}
                stakeholder_id = payload.get('stakeholder_id')
                action_type = (payload.get('action_type') or '').lower()
                
                faction = await self._get_stakeholder_faction(stakeholder_id)
                if not faction:
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'processed': False, 'reason': 'no_faction'},
                        side_effects=[]
                    )
                
                # Normalize 'faction_turn' into a simple composite action
                if action_type == 'faction_turn':
                    # Decide a light-weight action based on relationships
                    result = await self._process_faction_turn(faction, payload)
                else:
                    result = await self._process_faction_action(faction, action_type, payload)
                
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
                        event_type=EventType.CONFLICT_UPDATED,  # fixed name
                        source_subsystem=self.subsystem_type,
                        payload={
                            'betrayal_occurred': True,
                            'betrayal_type': (result.get('suggested_type') or BetrayalType.DEFECTION).value,
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
            
            if event.event_type == EventType.PHASE_TRANSITION:
                conflict_id = (event.payload or {}).get('conflict_id')
                new_phase = (event.payload or {}).get('phase') or (event.payload or {}).get('new_phase')
                side_effects = []
                
                if new_phase == 'climax':
                    betrayals = await self._check_for_betrayals(conflict_id)
                    SubsystemType, EventType, SystemEvent, _ = _orch()
                    for b in betrayals:
                        side_effects.append(SystemEvent(
                            event_id=f"betrayal_{event.event_id}_{b['betrayer_id']}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=self.subsystem_type,
                            payload=b,
                            priority=2
                        ))
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'betrayals_triggered': len(betrayals)},
                        side_effects=side_effects
                    )
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'phase_processed': new_phase},
                    side_effects=[]
                )
            
            if event.event_type == EventType.HEALTH_CHECK:
                # Lightweight internal health summary
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'healthy': True, 'relationship_cache_keys': len(self._relationship_cache)},
                    side_effects=[]
                )
            
            # Default pass-through
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'processed': True},
                side_effects=[]
            )
        
        except Exception as e:
            logger.error(f"Error handling event in multiparty subsystem: {e}")
            SubsystemType, _, _, SubsystemResponse = _orch()
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )
    
    # ----- Relationship helpers -----
    
    async def _get_relationship_factors(self, entity1_id: int, entity2_id: int,
                                       entity1_type: str = "npc",
                                       entity2_type: str = "npc") -> Dict[str, Any]:
        try:
            cache_key = f"{entity1_id}:{entity2_id}"
            if cache_key in self._relationship_cache and self._cache_timestamp:
                if (datetime.now() - self._cache_timestamp).seconds < 300:
                    return self._relationship_cache[cache_key]
            
            relationship = await self.relationship_integration.get_relationship(
                entity1_type, entity1_id, entity2_type, entity2_id
            )
            if not relationship:
                relationship = await self.relationship_integration.create_relationship(
                    entity1_type, entity1_id, entity2_type, entity2_id,
                    relationship_type="faction_member"
                )
            factors = {
                'exists': relationship is not None,
                'trust': float(relationship.get('trust', 0)),
                'closeness': float(relationship.get('closeness', 0)),
                'tension': float(relationship.get('tension', 0)),
                'power_balance': float(relationship.get('power_balance', 0)),
                'loyalty': float(relationship.get('loyalty', 50)),
                'history': relationship.get('interaction_history', []),
                'archetype': relationship.get('archetype', 'neutral')
            }
            self._relationship_cache[cache_key] = factors
            self._cache_timestamp = datetime.now()
            return factors
        except Exception as e:
            logger.error(f"Error fetching relationship factors: {e}")
            return {'exists': False, 'trust': 0, 'closeness': 0, 'tension': 0, 'power_balance': 0, 'loyalty': 50}
    
    async def _evaluate_alliance_potential(self, faction1_id: int, faction2_id: int,
                                          faction1_members: List[int],
                                          faction2_members: List[int]) -> float:
        total_score = 0.0
        relationship_count = 0
        for m1 in faction1_members:
            for m2 in faction2_members:
                f = await self._get_relationship_factors(m1, m2)
                if f['exists']:
                    relationship_count += 1
                    score = 0.0
                    score += f['trust'] * 0.3
                    score += f['closeness'] * 0.2
                    score += (100 - f['tension']) * 0.2
                    score += f['loyalty'] * 0.15
                    if abs(f['power_balance']) < 30:
                        score += 15
                    if f['archetype'] in ['allies', 'mentor_student', 'partners']:
                        score += 20
                    elif f['archetype'] in ['rivals', 'enemies', 'predator_prey']:
                        score -= 30
                    total_score += score
        if relationship_count > 0:
            return float(min(100, max(0, total_score / relationship_count)))
        return 50.0
    
    async def _evaluate_betrayal_likelihood(self, betrayer_id: int, target_faction_id: int,
                                           betrayer_members: List[int],
                                           target_members: List[int]) -> Dict[str, Any]:
        betrayal_score = 0.0
        betrayal_factors: List[str] = []
        suggested_type = None
        
        # Internal loyalty
        internal_loyalty = 0.0
        for member in betrayer_members:
            if member != betrayer_id:
                f = await self._get_relationship_factors(betrayer_id, member)
                internal_loyalty += f['trust'] * 0.5 + f['loyalty'] * 0.5
        if betrayer_members:
            internal_loyalty /= len(betrayer_members)
        
        # External attraction
        external_attraction = 0.0
        for member in target_members:
            f = await self._get_relationship_factors(betrayer_id, member)
            external_attraction += f['closeness'] * 0.3 + f['trust'] * 0.2
        if target_members:
            external_attraction /= len(target_members)
        
        if internal_loyalty < 30:
            betrayal_score += 30
            betrayal_factors.append("Low loyalty to current faction")
        if external_attraction > 70:
            betrayal_score += 25
            betrayal_factors.append("Strong ties to target faction")
        if internal_loyalty < external_attraction:
            betrayal_score += 20
            betrayal_factors.append("Stronger external relationships")
        
        if betrayal_score > 50:
            if internal_loyalty < 20:
                suggested_type = BetrayalType.DEFECTION
            elif external_attraction > 80:
                suggested_type = BetrayalType.DOUBLE_CROSS
            else:
                suggested_type = BetrayalType.INFORMATION_LEAK
        
        return {
            'likelihood': float(min(100, betrayal_score)),
            'factors': betrayal_factors,
            'suggested_type': suggested_type,
            'internal_loyalty': float(internal_loyalty),
            'external_attraction': float(external_attraction)
        }
    
    # ----- Core behavior -----
    
    async def _initialize_factions(self, conflict_id: Optional[int], factions_seed: List[Dict[str, Any]]) -> List[Faction]:
        """Initialize factions for a new conflict (if provided in context)."""
        initialized: List[Faction] = []
        try:
            if not factions_seed:
                return []
            async with get_db_connection_context() as conn:
                for f in factions_seed[:6]:
                    name = str(f.get('name', 'faction'))
                    members = list(f.get('members', []) or [])
                    resources = dict(f.get('resources', {}) or {})
                    stance = str(f.get('stance', 'neutral')).upper()
                    rep = float(f.get('reputation', 50.0))
                    fid = await conn.fetchval("""
                        INSERT INTO factions (name, members, resources, stance, reputation, user_id, conversation_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING faction_id
                    """, name, json.dumps(members), json.dumps(resources), stance.lower(), rep, self.user_id, self.conversation_id)
                    initialized.append(Faction(
                        faction_id=int(fid),
                        name=name,
                        members=members,
                        resources=resources,
                        goals=list(f.get('goals', []) or []),
                        strengths=list(f.get('strengths', []) or []),
                        weaknesses=list(f.get('weaknesses', []) or []),
                        current_stance=FactionRole[stance] if stance in FactionRole.__members__ else FactionRole.NEUTRAL,
                        reputation=rep
                    ))
                    if conflict_id:
                        await conn.execute("""
                            INSERT INTO faction_conflicts (faction_id, conflict_id)
                            VALUES ($1, $2)
                            ON CONFLICT DO NOTHING
                        """, int(fid), int(conflict_id))
        except Exception as e:
            logger.warning(f"Faction initialization failed: {e}")
        return initialized
    
    async def _get_stakeholder_faction(self, stakeholder_id: int) -> Optional[Faction]:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT f.*
                FROM factions f
                JOIN faction_members fm ON f.faction_id = fm.faction_id
                WHERE fm.member_id = $1
            """, int(stakeholder_id))
        if not row:
            return None
        try:
            return Faction(
                faction_id=int(row['faction_id']),
                name=row['name'],
                members=list(json.loads(row['members'])),
                resources=dict(json.loads(row['resources'])),
                goals=list(json.loads(row['goals'])) if row.get('goals') else [],
                strengths=list(json.loads(row['strengths'])) if row.get('strengths') else [],
                weaknesses=list(json.loads(row['weaknesses'])) if row.get('weaknesses') else [],
                current_stance=FactionRole[str(row['stance']).upper()],
                reputation=float(row['reputation'])
            )
        except Exception:
            # Fallback if JSON cols missing
            return Faction(
                faction_id=int(row['faction_id']),
                name=row['name'],
                members=[],
                resources={},
                goals=[],
                strengths=[],
                weaknesses=[],
                current_stance=FactionRole.NEUTRAL,
                reputation=float(row.get('reputation', 50.0))
            )
    
    async def _process_faction_turn(self, faction: Faction, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simple composite action when a faction takes its turn."""
        # Evaluate current negotiation strength as a proxy action
        strength = await self._calculate_negotiation_strength(faction)
        result = {
            'action_processed': True,
            'action_type': 'faction_turn',
            'faction_id': faction.faction_id,
            'negotiation_strength': float(strength),
            'effects': []
        }
        if strength > 65:
            result['effects'].append('secured_minor_concessions')
        elif strength < 35:
            result['effects'].append('lost_face_in_negotiations')
        return result
    
    async def _process_faction_action(self, faction: Faction, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = {'action_processed': True, 'action_type': action_type, 'faction_id': faction.faction_id}
        
        if action_type == 'propose_alliance':
            target_faction_id = payload.get('target_faction_id')
            target = await self._get_faction_by_id(target_faction_id)
            if target:
                score = await self._evaluate_alliance_potential(faction.faction_id, target_faction_id,
                                                                faction.members, target.members)
                result['alliance_score'] = score
                result['triggers_alliance'] = score > 65
                if result['triggers_alliance']:
                    # Nudge relationships positively
                    for m1 in faction.members:
                        for m2 in target.members:
                            await self.relationship_integration.update_relationship(
                                "npc", m1, "npc", m2, {"trust": 10, "closeness": 5}
                            )
        
        elif action_type == 'consider_betrayal':
            betrayer_id = payload.get('potential_betrayer_id')
            target_faction_id = payload.get('target_faction_id')
            target = await self._get_faction_by_id(target_faction_id)
            if target:
                eval_ = await self._evaluate_betrayal_likelihood(betrayer_id, target_faction_id,
                                                                 faction.members, target.members)
                result.update(eval_)
                result['triggers_betrayal'] = eval_['likelihood'] > 70
                if result['triggers_betrayal']:
                    for member in faction.members:
                        if member != betrayer_id:
                            await self.relationship_integration.update_relationship(
                                "npc", betrayer_id, "npc", member,
                                {"trust": -30, "tension": 20}
                            )
        
        elif action_type == 'faction_negotiation':
            strength = await self._calculate_negotiation_strength(faction)
            result['negotiation_strength'] = strength
            result['leverage_from_relationships'] = strength > 60
        
        return result
    
    async def _get_faction_by_id(self, faction_id: int) -> Optional[Faction]:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM factions WHERE faction_id = $1
            """, int(faction_id))
        if not row:
            return None
        return Faction(
            faction_id=int(row['faction_id']),
            name=row['name'],
            members=list(json.loads(row['members'])),
            resources=dict(json.loads(row['resources'])),
            goals=list(json.loads(row.get('goals', '[]'))),
            strengths=list(json.loads(row.get('strengths', '[]'))),
            weaknesses=list(json.loads(row.get('weaknesses', '[]'))),
            current_stance=FactionRole[str(row['stance']).upper()],
            reputation=float(row['reputation'])
        )
    
    async def _calculate_negotiation_strength(self, faction: Faction) -> float:
        strength = 0.0
        # Aggregate from relationships table(s)
        for member in faction.members:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT AVG(trust) as avg_trust, AVG(power_balance) as avg_power
                    FROM relationship_states
                    WHERE (entity1_id = $1 AND entity1_type = 'npc')
                       OR (entity2_id = $1 AND entity2_type = 'npc')
                """, int(member))
            for row in rows or []:
                if row['avg_trust'] is not None:
                    strength += float(row['avg_trust']) * 0.3
                if row['avg_power'] is not None:
                    strength += abs(float(row['avg_power'])) * 0.2
        if faction.members:
            strength /= len(faction.members)
        return float(min(100, max(0, strength)))
    
    async def _check_for_betrayals(self, conflict_id: Optional[int]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not conflict_id:
            return results
        
        # Fetch opportunist factions participating in this conflict
        async with get_db_connection_context() as conn:
            factions = await conn.fetch("""
                SELECT f.*
                FROM factions f
                JOIN faction_conflicts fc ON f.faction_id = fc.faction_id
                WHERE fc.conflict_id = $1 AND f.stance = 'opportunist'
            """, int(conflict_id))
        
        for frow in factions or []:
            faction = Faction(
                faction_id=int(frow['faction_id']),
                name=frow['name'],
                members=list(json.loads(frow['members'])),
                resources=dict(json.loads(frow['resources'])),
                goals=list(json.loads(frow.get('goals', '[]'))),
                strengths=[],
                weaknesses=[],
                current_stance=FactionRole.OPPORTUNIST,
                reputation=float(frow['reputation'])
            )
            # For each member, find a target faction and evaluate
            for member in faction.members:
                async with get_db_connection_context() as conn:
                    target_rows = await conn.fetch("""
                        SELECT f.*
                        FROM factions f
                        JOIN faction_conflicts fc ON f.faction_id = fc.faction_id
                        WHERE fc.conflict_id = $1 AND f.faction_id != $2
                    """, int(conflict_id), int(faction.faction_id))
                for trow in target_rows or []:
                    target = await self._get_faction_by_id(int(trow['faction_id']))
                    if not target:
                        continue
                    eval_ = await self._evaluate_betrayal_likelihood(
                        int(member), int(target.faction_id),
                        faction.members, target.members
                    )
                    if eval_['likelihood'] > 70:
                        results.append({
                            'betrayer_id': int(member),
                            'source_faction': int(faction.faction_id),
                            'target_faction': int(target.faction_id),
                            'type': (eval_.get('suggested_type') or BetrayalType.DEFECTION).value,
                            'factors': eval_['factors']
                        })
                        break  # one betrayal per member
        return results
    
    async def _calculate_power_balance(self) -> Dict[str, float]:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT faction_id, name, reputation, resources
                FROM factions
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        pb: Dict[str, float] = {}
        for row in rows or []:
            try:
                resources = json.loads(row['resources']) if row.get('resources') else {}
                # Simple heuristic; adjust if reputation scale is 0-100
                rep = float(row.get('reputation', 50.0))
                power = rep * 0.5 + sum(float(v) for v in resources.values()) * 0.5
                pb[row['name']] = float(power)
            except Exception:
                continue
        return pb
    
    # ----- Subsystem status endpoints -----
    
    async def get_conflict_state(self, conflict_id: int) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            factions = await conn.fetch("""
                SELECT f.*
                FROM factions f
                JOIN faction_conflicts fc ON f.faction_id = fc.faction_id
                WHERE fc.conflict_id = $1
            """, int(conflict_id))
            alliances = await conn.fetch("""
                SELECT *
                FROM alliances
                WHERE is_active = true
                  AND (faction_id_1 IN (SELECT faction_id FROM faction_conflicts WHERE conflict_id = $1)
                       OR faction_id_2 IN (SELECT faction_id FROM faction_conflicts WHERE conflict_id = $1))
            """, int(conflict_id))
        return {
            'faction_count': len(factions or []),
            'factions': [
                {
                    'id': int(f['faction_id']),
                    'name': f['name'],
                    'stance': f['stance'],
                    'reputation': float(f['reputation'])
                } for f in (factions or [])
            ],
            'alliance_count': len(alliances or [])
        }
    
    async def get_state(self) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            top = await conn.fetch("""
                SELECT faction_id, name, reputation
                FROM factions
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY reputation DESC
                LIMIT 3
            """, self.user_id, self.conversation_id)
        return {
            'dominant_factions': [
                {'name': r['name'], 'reputation': float(r['reputation'])} for r in (top or [])
            ],
            'power_balance': await self._calculate_power_balance()
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        present_npcs = list((scene_context or {}).get('present_npcs', []))
        if len(present_npcs) < 2:
            return False
        async with get_db_connection_context() as conn:
            faction_diversity = await conn.fetchval("""
                SELECT COUNT(DISTINCT f.faction_id)
                FROM factions f
                JOIN faction_members fm ON f.faction_id = fm.faction_id
                WHERE fm.member_id = ANY($1)
            """, present_npcs)
        return bool(faction_diversity and faction_diversity > 1)
    
    # ----- Agent properties (lazy) -----
    
    @property
    def faction_strategist(self) -> Agent:
        if self._faction_strategist is None:
            self._faction_strategist = Agent(
                name="Faction Strategist",
                instructions="Generate realistic faction strategies.",
                tools=[],
                model=ModelSettings(model="gpt-5-nano")
            )
        return self._faction_strategist
    
    @property
    def alliance_broker(self) -> Agent:
        if self._alliance_broker is None:
            self._alliance_broker = Agent(
                name="Alliance Broker",
                instructions="Broker alliances based on power and relationships.",
                tools=[],
                model=ModelSettings(model="gpt-5-nano")
            )
        return self._alliance_broker
    
    @property
    def betrayal_orchestrator(self) -> Agent:
        if self._betrayal_orchestrator is None:
            self._betrayal_orchestrator = Agent(
                name="Betrayal Orchestrator",
                instructions="Orchestrate betrayals grounded in relationships.",
                tools=[],
                model=ModelSettings(model="gpt-5-nano")
            )
        return self._betrayal_orchestrator
    
    @property
    def negotiation_mediator(self) -> Agent:
        if self._negotiation_mediator is None:
            self._negotiation_mediator = Agent(
                name="Negotiation Mediator",
                instructions="Mediate complex negotiations between multiple parties.",
                tools=[],
                model=ModelSettings(model="gpt-5-nano")
            )
        return self._negotiation_mediator
    
    @property
    def outcome_predictor(self) -> Agent:
        if self._outcome_predictor is None:
            self._outcome_predictor = Agent(
                name="Outcome Predictor",
                instructions="Predict outcomes of multi-party conflicts.",
                tools=[],
                model=ModelSettings(model="gpt-5-nano")
            )
        return self._outcome_predictor


# ===============================================================================
# FUNCTION TOOLS (orchestrator-friendly)
# ===============================================================================

@function_tool
async def initialize_multi_faction_conflict(
    ctx: RunContextWrapper,
    conflict_name: str,
    initial_factions_json: str,
) -> InitializeMultiFactionConflictResponse:
    """Initialize a multi-faction conflict via synthesizer (strict schema)."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    try:
        seeds = json.loads(initial_factions_json) if initial_factions_json else []
        if not isinstance(seeds, list):
            seeds = []
    except Exception:
        seeds = []
    
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Optional cohesion via relationships
    try:
        from logic.relationships.integration import RelationshipIntegration as RelAlt
        rel_integration = RelAlt(user_id, conversation_id)
    except Exception:
        rel_integration = None
    
    if rel_integration:
        for f in seeds:
            members = list((f or {}).get('members', []) or [])
            if not members:
                f['internal_cohesion'] = 50.0
                continue
            cohesion_sum, pair_count = 0.0, 0
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    try:
                        rel = await rel_integration.get_relationship("npc", members[i], "npc", members[j])
                    except Exception:
                        rel = None
                    if rel:
                        cohesion_sum += float(rel.get('trust', 0.0)) + float(rel.get('closeness', 0.0))
                        pair_count += 1
            f['internal_cohesion'] = float((cohesion_sum / (pair_count * 2.0)) if pair_count else 50.0)
    
    raw = await synthesizer.create_conflict(
        'multi_faction_war',
        {'name': conflict_name, 'factions': seeds}
    ) or {}
    
    return {
        'conflict_id': int(raw.get('conflict_id', 0)),
        'conflict_name': str(raw.get('conflict_name', conflict_name)),
        'status': str(raw.get('status', 'created')),
        'factions_registered': int(len(seeds)),
        'message': str(raw.get('message', 'initialized')),
        'error': "",
    }


@function_tool
async def faction_take_action(
    ctx: RunContextWrapper,
    faction_id: int,
    conflict_id: int
) -> FactionTakeActionResponse:
    """Have a faction take its turn via synthesizer (strict schema)."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    SubsystemType, EventType, SystemEvent, _ = _orch()
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    event = SystemEvent(
        event_id=f"faction_action_{faction_id}",
        event_type=EventType.STAKEHOLDER_ACTION,
        source_subsystem=SubsystemType.MULTIPARTY,
        payload={
            'stakeholder_id': int(faction_id),
            'conflict_id': int(conflict_id),
            'action_type': 'faction_turn',
            'consider_relationships': True,
        },
        target_subsystems={SubsystemType.MULTIPARTY},
        requires_response=True
    )
    responses = await synthesizer.emit_event(event) or []
    data = {}
    for r in responses:
        if r.subsystem == SubsystemType.MULTIPARTY:
            data = r.data or {}
            break
    effects = data.get('effects', [])
    if not isinstance(effects, list):
        effects = [str(effects)]
    return {
        'performed': bool(data.get('action_processed', True)),
        'action_type': str(data.get('action_type', 'faction_turn')),
        'stakeholder_id': int(faction_id),
        'effects': [str(e) for e in effects],
        'notes': str(data.get('notes', '')),
        'error': "",
    }


@function_tool
async def negotiate_between_factions(
    ctx: RunContextWrapper,
    topic: str,
    participating_factions: List[int],
    initial_position: str
) -> NegotiateBetweenFactionsResponse:
    """Start a negotiation with relationship-based strengths (strict schema)."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    try:
        from logic.relationships.integration import RelationshipIntegration as RelAlt
    except Exception:
        RelAlt = None
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Prefetch faction members
    faction_members: Dict[int, List[int]] = {}
    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT faction_id, member_id
            FROM faction_members
            WHERE faction_id = ANY($1::int[])
        """, participating_factions)
    for row in rows or []:
        fid = int(row['faction_id'])
        faction_members.setdefault(fid, []).append(int(row['member_id']))
    
    strengths_map: Dict[int, float] = {}
    if RelAlt:
        rel_integration = RelAlt(user_id, conversation_id)
        for fid in participating_factions:
            members = faction_members.get(fid, [])
            if not members:
                strengths_map[fid] = 50.0
                continue
            total, count = 0.0, 0
            for other_fid in participating_factions:
                if other_fid == fid:
                    continue
                others = faction_members.get(other_fid, [])
                for m1 in members:
                    for m2 in others:
                        try:
                            rel = await rel_integration.get_relationship("npc", m1, "npc", m2)
                        except Exception:
                            rel = None
                        if rel:
                            total += float(rel.get('power_balance', 0.0)) * 0.3 + float(rel.get('trust', 0.0)) * 0.2
                            count += 1
            strengths_map[fid] = float((total / count) if count else 50.0)
    else:
        for fid in participating_factions:
            strengths_map[fid] = 50.0
    
    # Persist negotiation
    async with get_db_connection_context() as conn:
        negotiation_id = await conn.fetchval("""
            INSERT INTO negotiations
                (user_id, conversation_id, topic, participants, status, metadata)
            VALUES ($1, $2, $3, $4, 'active', $5)
            RETURNING negotiation_id
        """, user_id, conversation_id, topic,
           json.dumps([int(fid) for fid in participating_factions]),
           json.dumps({'faction_strengths': strengths_map, 'initial_position': initial_position}))
    
    strengths_list: List[FactionStrengthItem] = [
        {'faction_id': int(fid), 'strength': float(val)} for fid, val in strengths_map.items()
    ]
    return {
        'negotiation_id': int(negotiation_id or 0),
        'topic': str(topic),
        'participants': [int(x) for x in participating_factions],
        'status': 'initiated',
        'faction_strengths': strengths_list,
        'relationship_factors_considered': True,
        'error': "",
    }
