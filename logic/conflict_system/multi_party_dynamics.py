# logic/conflict_system/multi_party_dynamics.py
"""
Multi-party conflict dynamics for slice-of-life RPG
Supports competing NPC interests, dynamic alliances, and emergent coalitions
All content dynamically generated via LLM agents
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# MULTI-PARTY CONFLICT STRUCTURES
# ===============================================================================

class StakeholderGoalType(Enum):
    """Types of goals NPCs can have in conflicts"""
    EXCLUSIVE_CONTROL = "exclusive_control"  # Only one can win
    SHARED_BENEFIT = "shared_benefit"  # Multiple can benefit
    PREVENT_OUTCOME = "prevent_outcome"  # Stop someone else
    MAINTAIN_STATUS = "maintain_status"  # Keep things as they are
    SHIFT_DYNAMICS = "shift_dynamics"  # Change power balance


@dataclass
class StakeholderPosition:
    """An NPC's position in a multi-party conflict"""
    npc_id: int
    desired_outcome: str  # LLM-generated description
    goal_type: StakeholderGoalType
    priority: float  # How important this is to them (0-1)
    flexibility: float  # Willingness to compromise (0-1)
    red_lines: List[str]  # LLM-generated unacceptable outcomes
    ideal_allies: List[int]  # NPCs they'd prefer to work with
    unacceptable_allies: List[int]  # NPCs they won't work with


@dataclass
class DynamicAlliance:
    """Temporary alliance between NPCs in a conflict"""
    member_ids: Set[int]
    shared_goal: str  # LLM-generated common objective
    against_ids: Set[int]  # NPCs they're opposing
    stability: float  # How stable the alliance is (0-1)
    formation_reason: str  # LLM-generated reason
    potential_fractures: List[str]  # LLM-generated weak points


class MultiPartyConflictOrchestrator:
    """Orchestrates complex multi-party conflicts with competing interests"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.alliance_agent = self._create_alliance_agent()
        self.negotiation_agent = self._create_negotiation_agent()
        self.outcome_agent = self._create_outcome_agent()
        
    def _create_alliance_agent(self) -> Agent:
        """Agent for generating dynamic alliances"""
        return Agent(
            name="Alliance Dynamics Generator",
            instructions="""
            Generate realistic alliance dynamics for multi-party conflicts in a slice-of-life setting.
            
            Consider:
            - NPC personalities and relationships
            - Shared vs competing interests
            - Power dynamics and dependencies
            - Historical patterns and grudges
            - Practical benefits of cooperation
            
            Alliances should be:
            - Fluid and conditional
            - Based on mutual benefit or shared opposition
            - Vulnerable to changing circumstances
            - Influenced by personality compatibility
            
            Focus on subtle social dynamics, not dramatic betrayals.
            Generate unique, contextual alliance reasoning.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
    
    def _create_negotiation_agent(self) -> Agent:
        """Agent for generating negotiation dynamics"""
        return Agent(
            name="Negotiation Dynamics Generator",
            instructions="""
            Generate complex negotiation dynamics between multiple parties.
            
            Consider each NPC's:
            - Core desires and flexibility
            - Relationship with other stakeholders
            - Power position and leverage
            - Personality-driven negotiation style
            - Hidden agendas and true priorities
            
            Generate:
            - Compromise proposals that feel natural
            - Coalition possibilities
            - Deal-breakers for each party
            - Creative solutions that satisfy multiple parties
            - Reasons why certain alliances form or fail
            
            Keep negotiations subtle and embedded in daily life.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    def _create_outcome_agent(self) -> Agent:
        """Agent for determining conflict outcomes"""
        return Agent(
            name="Multi-Party Outcome Generator",
            instructions="""
            Generate nuanced outcomes for multi-party conflicts where:
            - Victories are rarely absolute
            - Compromises create new dynamics
            - Alliances leave lasting effects
            - Power balances shift subtly
            
            Consider:
            - Who got what they wanted (fully/partially/not at all)
            - How alliances affected outcomes
            - New tensions created by resolutions
            - Changes to social dynamics
            - Future implications
            
            Generate outcomes that feel like natural consequences 
            of social maneuvering, not game mechanics.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.75)
        )
    
    async def generate_stakeholder_positions(
        self,
        conflict_id: int,
        conflict_context: str,
        participating_npcs: List[int]
    ) -> Dict[int, StakeholderPosition]:
        """Generate unique positions for each NPC in the conflict"""
        
        positions = {}
        
        async with get_db_connection_context() as conn:
            # Get NPC data for context
            npcs_data = []
            for npc_id in participating_npcs:
                npc = await conn.fetchrow("""
                    SELECT npc_id, npc_name, personality_traits, 
                           dominance, ambition, relationships
                    FROM NPCStats
                    WHERE npc_id = $1
                """, npc_id)
                
                if npc:
                    # Get relationships with other stakeholders
                    relationships = await conn.fetch("""
                        SELECT entity2_id, dimension, current_value
                        FROM relationship_dimensions
                        WHERE entity1_id = $1 
                        AND entity2_id = ANY($2)
                        AND dimension IN ('trust', 'rivalry', 'dependency')
                    """, npc_id, participating_npcs)
                    
                    npcs_data.append({
                        'id': npc_id,
                        'name': npc['npc_name'],
                        'personality': json.loads(npc['personality_traits'] or '{}'),
                        'dominance': npc['dominance'],
                        'ambition': npc['ambition'],
                        'relationships': [dict(r) for r in relationships]
                    })
        
        # Generate positions for each NPC using LLM
        prompt = f"""
        Conflict context: {conflict_context}
        NPCs involved: {json.dumps(npcs_data)}
        
        For each NPC, generate their position in this conflict.
        Consider their personality, relationships, and power level.
        
        Return JSON array with positions for each NPC:
        [{{
            "npc_id": id,
            "desired_outcome": "what they specifically want",
            "goal_type": "exclusive_control/shared_benefit/prevent_outcome/maintain_status/shift_dynamics",
            "priority": 0.0-1.0,
            "flexibility": 0.0-1.0,
            "red_lines": ["unacceptable outcome 1", "unacceptable outcome 2"],
            "ideal_allies": [npc_ids they'd work with],
            "unacceptable_allies": [npc_ids they won't work with],
            "reasoning": "why they want this based on personality/relationships"
        }}]
        
        Make each position unique and driven by the NPC's characteristics.
        Create natural conflicts of interest between some NPCs.
        Some should have mutually exclusive goals.
        """
        
        response = await Runner.run(self.alliance_agent, prompt)
        positions_data = json.loads(response.output)
        
        for pos_data in positions_data:
            positions[pos_data['npc_id']] = StakeholderPosition(
                npc_id=pos_data['npc_id'],
                desired_outcome=pos_data['desired_outcome'],
                goal_type=StakeholderGoalType[pos_data['goal_type'].upper()],
                priority=pos_data['priority'],
                flexibility=pos_data['flexibility'],
                red_lines=pos_data['red_lines'],
                ideal_allies=pos_data.get('ideal_allies', []),
                unacceptable_allies=pos_data.get('unacceptable_allies', [])
            )
        
        return positions
    
    async def detect_alliance_opportunities(
        self,
        positions: Dict[int, StakeholderPosition]
    ) -> List[DynamicAlliance]:
        """Detect potential alliances based on stakeholder positions"""
        
        # Prepare data for LLM analysis
        positions_data = []
        for npc_id, pos in positions.items():
            positions_data.append({
                'npc_id': npc_id,
                'desired_outcome': pos.desired_outcome,
                'goal_type': pos.goal_type.value,
                'flexibility': pos.flexibility,
                'ideal_allies': pos.ideal_allies,
                'unacceptable_allies': pos.unacceptable_allies
            })
        
        prompt = f"""
        Analyze these stakeholder positions for alliance opportunities:
        {json.dumps(positions_data)}
        
        Identify natural alliances that would form based on:
        - Shared interests or compatible goals
        - Opposition to a common threat
        - Practical benefits of cooperation
        - Personality compatibility
        - Desire to prevent specific outcomes
        
        Return JSON array of potential alliances:
        [{{
            "members": [npc_ids],
            "shared_goal": "what unites them",
            "against": [npc_ids they oppose],
            "stability": 0.0-1.0,
            "formation_reason": "detailed reason for alliance",
            "potential_fractures": ["what could break this alliance"]
        }}]
        
        Consider both strong alliances and temporary conveniences.
        Some NPCs might be in multiple potential alliances.
        """
        
        response = await Runner.run(self.alliance_agent, prompt)
        alliances_data = json.loads(response.output)
        
        alliances = []
        for alliance_data in alliances_data:
            alliances.append(DynamicAlliance(
                member_ids=set(alliance_data['members']),
                shared_goal=alliance_data['shared_goal'],
                against_ids=set(alliance_data.get('against', [])),
                stability=alliance_data['stability'],
                formation_reason=alliance_data['formation_reason'],
                potential_fractures=alliance_data['potential_fractures']
            ))
        
        return alliances
    
    async def negotiate_compromise(
        self,
        positions: Dict[int, StakeholderPosition],
        alliances: List[DynamicAlliance]
    ) -> Dict[str, Any]:
        """Generate compromise negotiations between parties"""
        
        # Prepare negotiation context
        negotiation_context = {
            'positions': [
                {
                    'npc_id': npc_id,
                    'desired': pos.desired_outcome,
                    'flexibility': pos.flexibility,
                    'red_lines': pos.red_lines
                }
                for npc_id, pos in positions.items()
            ],
            'alliances': [
                {
                    'members': list(alliance.member_ids),
                    'goal': alliance.shared_goal,
                    'stability': alliance.stability
                }
                for alliance in alliances
            ]
        }
        
        prompt = f"""
        Generate negotiation dynamics for this multi-party conflict:
        {json.dumps(negotiation_context)}
        
        Create realistic negotiation proposals where:
        - Some parties seek mutual benefit
        - Others form coalitions to exclude specific parties
        - Compromises reflect personality and relationships
        - Power dynamics influence negotiation leverage
        
        Return JSON with:
        {{
            "initial_proposals": [
                {{
                    "proposer": npc_id,
                    "proposal": "specific compromise offered",
                    "benefits_for": {{npc_id: "what they get"}},
                    "excludes": [npc_ids who get nothing]
                }}
            ],
            "counter_proposals": [
                {{
                    "proposer": npc_id,
                    "counter_to": proposal_index,
                    "modification": "how they'd change it",
                    "reasoning": "why this is better"
                }}
            ],
            "coalition_deals": [
                {{
                    "coalition": [npc_ids],
                    "deal": "what they agree to",
                    "against": [npc_ids they're excluding],
                    "enforcement": "how they ensure compliance"
                }}
            ],
            "negotiation_breakdown": {{
                "likely_outcome": "most probable resolution",
                "winners": [npc_ids who get most],
                "compromisers": [npc_ids who get some],
                "losers": [npc_ids who get least]
            }}
        }}
        """
        
        response = await Runner.run(self.negotiation_agent, prompt)
        return json.loads(response.output)
    
    async def evolve_alliances(
        self,
        conflict_id: int,
        current_alliances: List[DynamicAlliance],
        recent_events: List[str]
    ) -> Tuple[List[DynamicAlliance], List[str]]:
        """Evolve alliances based on events"""
        
        prompt = f"""
        Current alliances: {json.dumps([
            {
                'members': list(a.member_ids),
                'goal': a.shared_goal,
                'stability': a.stability,
                'fractures': a.potential_fractures
            }
            for a in current_alliances
        ])}
        
        Recent events: {json.dumps(recent_events)}
        
        Determine how alliances evolve based on events.
        Consider:
        - Trust broken or reinforced
        - Goals achieved or thwarted  
        - New information revealed
        - Power dynamics shifted
        - External pressures
        
        Return JSON:
        {{
            "alliance_changes": [
                {{
                    "type": "strengthen/weaken/break/form",
                    "affected_members": [npc_ids],
                    "reason": "what caused this change",
                    "new_stability": 0.0-1.0
                }}
            ],
            "new_alliances": [
                {{
                    "members": [npc_ids],
                    "goal": "new shared objective",
                    "formed_because": "triggering event"
                }}
            ],
            "narrative_moments": [
                "Description of alliance shift moment"
            ]
        }}
        """
        
        response = await Runner.run(self.alliance_agent, prompt)
        evolution_data = json.loads(response.output)
        
        # Apply changes to alliances
        updated_alliances = []
        narrative_moments = evolution_data.get('narrative_moments', [])
        
        for alliance in current_alliances:
            modified = False
            for change in evolution_data['alliance_changes']:
                if set(change['affected_members']) == alliance.member_ids:
                    if change['type'] == 'break':
                        modified = True
                        break
                    elif change['type'] in ['strengthen', 'weaken']:
                        alliance.stability = change['new_stability']
                        modified = True
            
            if not modified or alliance.stability > 0.1:
                updated_alliances.append(alliance)
        
        # Add new alliances
        for new_alliance_data in evolution_data.get('new_alliances', []):
            updated_alliances.append(DynamicAlliance(
                member_ids=set(new_alliance_data['members']),
                shared_goal=new_alliance_data['goal'],
                against_ids=set(),
                stability=0.6,
                formation_reason=new_alliance_data['formed_because'],
                potential_fractures=[]
            ))
        
        return updated_alliances, narrative_moments


# ===============================================================================
# COMPETITION MANIFESTATION IN DAILY LIFE
# ===============================================================================

class DailyCompetitionManager:
    """Manages how multi-party competition manifests in daily activities"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.competition_agent = self._create_competition_agent()
        
    def _create_competition_agent(self) -> Agent:
        """Agent for generating subtle competition in daily life"""
        return Agent(
            name="Daily Competition Generator",
            instructions="""
            Generate subtle competitive dynamics in everyday activities.
            
            Focus on:
            - Indirect competition through daily choices
            - Social maneuvering in routine interactions
            - Subtle alliance signals in group settings
            - Competition disguised as helpfulness
            - Passive-aggressive cooperation
            
            Avoid:
            - Direct confrontation
            - Explicit declarations of opposition
            - Dramatic power plays
            
            Competition should feel like natural social friction.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def manifest_competition_in_activity(
        self,
        activity_type: str,
        competing_npcs: List[int],
        their_positions: Dict[int, StakeholderPosition],
        current_alliances: List[DynamicAlliance]
    ) -> Dict[str, Any]:
        """Generate how competition manifests in a specific activity"""
        
        # Get NPC details
        async with get_db_connection_context() as conn:
            npcs = []
            for npc_id in competing_npcs:
                npc = await conn.fetchrow("""
                    SELECT npc_name, dominance, personality_traits
                    FROM NPCStats WHERE npc_id = $1
                """, npc_id)
                if npc:
                    npcs.append({
                        'id': npc_id,
                        'name': npc['npc_name'],
                        'dominance': npc['dominance'],
                        'position': their_positions[npc_id].desired_outcome
                    })
        
        # Identify alliance dynamics
        alliance_map = {}
        for alliance in current_alliances:
            for member_id in alliance.member_ids:
                if member_id in competing_npcs:
                    alliance_map[member_id] = [
                        m for m in alliance.member_ids 
                        if m in competing_npcs and m != member_id
                    ]
        
        prompt = f"""
        Activity: {activity_type}
        Competing NPCs: {json.dumps(npcs)}
        Current alliances: {json.dumps(alliance_map)}
        
        Generate subtle competition during this {activity_type}.
        
        Return JSON:
        {{
            "scene_dynamics": "Overall competitive atmosphere",
            "npc_behaviors": {{
                npc_id: {{
                    "action": "What they do",
                    "subtext": "Hidden meaning",
                    "targets": [who they're competing with],
                    "alliance_signals": [subtle support for allies]
                }}
            }},
            "micro_conflicts": [
                {{
                    "between": [npc_ids],
                    "over": "what they're competing about",
                    "manifestation": "how it shows in the activity"
                }}
            ],
            "player_opportunities": [
                {{
                    "choice": "what player can do",
                    "affects": [npc_ids affected],
                    "tilts_toward": [npc_ids who benefit]
                }}
            ]
        }}
        
        Keep all competition subtle and embedded in the activity.
        """
        
        response = await Runner.run(self.competition_agent, prompt)
        return json.loads(response.output)


# ===============================================================================
# OUTCOME DETERMINATION
# ===============================================================================

class MultiPartyOutcomeResolver:
    """Resolves multi-party conflicts with nuanced outcomes"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.outcome_agent = Agent(
            name="Multi-Party Outcome Resolver",
            instructions="""
            Generate nuanced outcomes for multi-party conflicts where:
            - Multiple parties can partially win
            - Compromises create new dynamics
            - Some parties can team up to exclude others
            - Pyrrhic victories are possible
            - New conflicts emerge from resolutions
            
            Consider personality, relationships, and power dynamics.
            Generate unique, contextual outcomes.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.75)
        )
    
    async def resolve_multi_party_conflict(
        self,
        conflict_id: int,
        positions: Dict[int, StakeholderPosition],
        final_alliances: List[DynamicAlliance],
        accumulated_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine the outcome of a multi-party conflict"""
        
        # Analyze accumulated patterns
        npc_action_counts = {}
        alliance_strength = {}
        
        for event in accumulated_events:
            for npc_id in event.get('participating_npcs', []):
                npc_action_counts[npc_id] = npc_action_counts.get(npc_id, 0) + 1
        
        for alliance in final_alliances:
            for member_id in alliance.member_ids:
                alliance_strength[member_id] = alliance.stability
        
        prompt = f"""
        Resolve this multi-party conflict based on accumulated events.
        
        Initial positions: {json.dumps({
            npc_id: {
                'desired': pos.desired_outcome,
                'flexibility': pos.flexibility,
                'priority': pos.priority
            }
            for npc_id, pos in positions.items()
        })}
        
        Final alliances: {json.dumps([{
            'members': list(a.member_ids),
            'goal': a.shared_goal,
            'stability': a.stability
        } for a in final_alliances])}
        
        Activity levels: {json.dumps(npc_action_counts)}
        
        Generate a nuanced outcome where:
        - Strong alliances might achieve shared goals
        - Flexible parties might get partial wins
        - Excluded parties might get nothing or create problems
        - Some original goals might be impossible now
        
        Return JSON:
        {{
            "primary_outcome": "What actually happens",
            "winners": [
                {{
                    "npc_id": id,
                    "achievement": "what they got",
                    "satisfaction": 0.0-1.0
                }}
            ],
            "partial_winners": [
                {{
                    "npc_id": id, 
                    "achievement": "partial success",
                    "satisfaction": 0.0-1.0
                }}
            ],
            "losers": [
                {{
                    "npc_id": id,
                    "loss": "what they didn't get",
                    "resentment": 0.0-1.0
                }}
            ],
            "new_dynamics": [
                {{
                    "between": [npc_ids],
                    "change": "how their relationship changed",
                    "lasting_effect": "ongoing impact"
                }}
            ],
            "seeds_of_future_conflict": [
                "Unresolved tension or new problem created"
            ]
        }}
        """
        
        response = await Runner.run(self.outcome_agent, prompt)
        outcome_data = json.loads(response.output)
        
        # Record outcome in database
        await self._record_outcome(conflict_id, outcome_data)
        
        return outcome_data
    
    async def _record_outcome(self, conflict_id: int, outcome_data: Dict):
        """Record the multi-party outcome in database"""
        
        async with get_db_connection_context() as conn:
            # Update conflict record
            await conn.execute("""
                UPDATE Conflicts
                SET 
                    outcome = $1,
                    resolution_description = $2,
                    is_active = false,
                    resolved_at = CURRENT_TIMESTAMP
                WHERE conflict_id = $3
            """, 'multi_party_resolution', 
            outcome_data['primary_outcome'], 
            conflict_id)
            
            # Record individual stakeholder outcomes
            for winner in outcome_data.get('winners', []):
                await conn.execute("""
                    INSERT INTO stakeholder_outcomes
                    (conflict_id, npc_id, outcome_type, achievement, satisfaction)
                    VALUES ($1, $2, 'win', $3, $4)
                """, conflict_id, winner['npc_id'], 
                winner['achievement'], winner['satisfaction'])
            
            for partial in outcome_data.get('partial_winners', []):
                await conn.execute("""
                    INSERT INTO stakeholder_outcomes
                    (conflict_id, npc_id, outcome_type, achievement, satisfaction)
                    VALUES ($1, $2, 'partial', $3, $4)
                """, conflict_id, partial['npc_id'],
                partial['achievement'], partial['satisfaction'])
            
            for loser in outcome_data.get('losers', []):
                await conn.execute("""
                    INSERT INTO stakeholder_outcomes
                    (conflict_id, npc_id, outcome_type, achievement, satisfaction)
                    VALUES ($1, $2, 'loss', $3, $4)
                """, conflict_id, loser['npc_id'],
                loser['loss'], -loser['resentment'])


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def generate_multi_party_conflict(
    ctx: RunContextWrapper,
    participating_npcs: List[int],
    conflict_seed: str  # e.g., "control over household decisions", "social standing at work"
) -> Dict[str, Any]:
    """
    Generate a multi-party conflict with competing NPC interests
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    orchestrator = MultiPartyConflictOrchestrator(user_id, conversation_id)
    
    # Create conflict record
    async with get_db_connection_context() as conn:
        conflict_id = await conn.fetchval("""
            INSERT INTO Conflicts
            (user_id, conversation_id, conflict_type, conflict_name,
             description, intensity, phase, is_active, progress)
            VALUES ($1, $2, 'multi_party_competition', $3, $4, 
                    'subtext', 'emerging', true, 0)
            RETURNING conflict_id
        """, user_id, conversation_id,
        f"Competition: {conflict_seed[:50]}",
        f"Multiple parties maneuvering over {conflict_seed}")
    
    # Generate stakeholder positions
    positions = await orchestrator.generate_stakeholder_positions(
        conflict_id, conflict_seed, participating_npcs
    )
    
    # Detect potential alliances
    alliances = await orchestrator.detect_alliance_opportunities(positions)
    
    # Generate initial negotiations
    negotiations = await orchestrator.negotiate_compromise(positions, alliances)
    
    # Store stakeholder data
    async with get_db_connection_context() as conn:
        for npc_id, position in positions.items():
            await conn.execute("""
                INSERT INTO conflict_stakeholders
                (conflict_id, npc_id, desired_outcome, faction, 
                 involvement_level, private_goal)
                VALUES ($1, $2, $3, $4, 'primary', $5)
            """, conflict_id, npc_id, position.desired_outcome,
            position.goal_type.value, json.dumps({
                'priority': position.priority,
                'flexibility': position.flexibility,
                'red_lines': position.red_lines
            }))
    
    return {
        'conflict_id': conflict_id,
        'seed': conflict_seed,
        'stakeholder_positions': {
            npc_id: {
                'desired': pos.desired_outcome,
                'goal_type': pos.goal_type.value,
                'flexibility': pos.flexibility
            }
            for npc_id, pos in positions.items()
        },
        'initial_alliances': [
            {
                'members': list(a.member_ids),
                'goal': a.shared_goal,
                'against': list(a.against_ids)
            }
            for a in alliances
        ],
        'negotiation_landscape': negotiations
    }


@function_tool
async def process_multi_party_dynamics(
    ctx: RunContextWrapper,
    conflict_id: int,
    current_activity: str,
    present_npcs: List[int]
) -> Dict[str, Any]:
    """
    Process multi-party dynamics during an activity
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get conflict data
    async with get_db_connection_context() as conn:
        stakeholders = await conn.fetch("""
            SELECT * FROM conflict_stakeholders
            WHERE conflict_id = $1
        """, conflict_id)
        
        conflict = await conn.fetchrow("""
            SELECT * FROM Conflicts
            WHERE conflict_id = $1
        """, conflict_id)
    
    # Only process if multiple stakeholders present
    present_stakeholders = [
        s for s in stakeholders 
        if s['npc_id'] in present_npcs
    ]
    
    if len(present_stakeholders) < 2:
        return {'no_competition': True}
    
    # Reconstruct positions
    positions = {}
    for stakeholder in present_stakeholders:
        private_goal = json.loads(stakeholder.get('private_goal', '{}'))
        positions[stakeholder['npc_id']] = StakeholderPosition(
            npc_id=stakeholder['npc_id'],
            desired_outcome=stakeholder['desired_outcome'],
            goal_type=StakeholderGoalType(stakeholder['faction']),
            priority=private_goal.get('priority', 0.5),
            flexibility=private_goal.get('flexibility', 0.5),
            red_lines=private_goal.get('red_lines', []),
            ideal_allies=[],
            unacceptable_allies=[]
        )
    
    # Get current alliances
    orchestrator = MultiPartyConflictOrchestrator(user_id, conversation_id)
    alliances = await orchestrator.detect_alliance_opportunities(positions)
    
    # Generate competition in current activity
    competition_manager = DailyCompetitionManager(user_id, conversation_id)
    competition = await competition_manager.manifest_competition_in_activity(
        current_activity,
        [s['npc_id'] for s in present_stakeholders],
        positions,
        alliances
    )
    
    return competition
