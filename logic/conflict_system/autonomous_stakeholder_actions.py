# logic/conflict_system/autonomous_stakeholder_actions.py
"""
Autonomous NPC stakeholder actions adapted for slice-of-life conflicts
NPCs take subtle actions during daily routines rather than dramatic "turns"
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# AUTONOMOUS ACTION TYPES
# ===============================================================================

class AutonomousActionType(Enum):
    """Types of autonomous actions NPCs take in conflicts"""
    ESTABLISH_PRECEDENT = "establish_precedent"  # Create new routine
    REINFORCE_PATTERN = "reinforce_pattern"  # Strengthen existing dynamic
    SUBTLE_REBELLION = "subtle_rebellion"  # Quiet resistance
    INDIRECT_PRESSURE = "indirect_pressure"  # Influence through others
    STRATEGIC_GIFT = "strategic_gift"  # Gift with implications
    INFORMATION_GATHERING = "information_gathering"  # Learn player patterns
    ALLIANCE_BUILDING = "alliance_building"  # Strengthen bonds with others
    BOUNDARY_TESTING = "boundary_testing"  # Push limits gently
    WITHDRAWAL = "withdrawal"  # Strategic distance


@dataclass
class AutonomousAction:
    """An autonomous action taken by an NPC in a conflict"""
    npc_id: int
    action_type: AutonomousActionType
    description: str
    target: str  # "player", "routine", "other_npc", "environment"
    subtlety_level: float  # 0-1, how hidden the action is
    impact: Dict[str, float]  # Effect on various dynamics
    requires_player_presence: bool
    can_be_discovered_later: bool


# ===============================================================================
# CONFLICT MEMORY ADAPTATION
# ===============================================================================

class SliceOfLifeConflictMemory:
    """Tracks patterns and precedents in daily conflicts"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
    async def record_micro_event(
        self,
        conflict_id: int,
        event_type: str,
        description: str,
        participants: List[int],
        outcome: Optional[str] = None
    ):
        """Record small conflict-related events"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO conflict_pattern_memory
                (conflict_id, event_type, description, participants,
                 outcome, significance, game_day, user_id, conversation_id)
                VALUES ($1, $2, $3, $4, $5, $6, 
                        (SELECT current_day FROM game_calendar 
                         WHERE user_id = $7 AND conversation_id = $8),
                        $7, $8)
            """, conflict_id, event_type, description, json.dumps(participants),
            outcome, 0.1,  # Low significance for individual events
            self.user_id, self.conversation_id)
    
    async def detect_patterns(
        self,
        conflict_id: int,
        lookback_days: int = 7
    ) -> List[Dict[str, Any]]:
        """Detect patterns in accumulated micro-events"""
        
        async with get_db_connection_context() as conn:
            events = await conn.fetch("""
                SELECT * FROM conflict_pattern_memory
                WHERE conflict_id = $1
                AND game_day > (SELECT current_day - $2 FROM game_calendar
                               WHERE user_id = $3 AND conversation_id = $4)
                ORDER BY game_day DESC
            """, conflict_id, lookback_days, self.user_id, self.conversation_id)
        
        patterns = []
        
        # Count event types
        event_counts = {}
        for event in events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Detect patterns
        for event_type, count in event_counts.items():
            if count >= 3:  # Pattern threshold
                patterns.append({
                    'pattern_type': event_type,
                    'frequency': count,
                    'strength': min(1.0, count / 10),
                    'description': f"Repeated {event_type} ({count} times)"
                })
        
        return patterns
    
    async def get_precedents(
        self,
        conflict_id: int,
        situation_type: str
    ) -> List[Dict[str, Any]]:
        """Get precedents for similar situations"""
        
        async with get_db_connection_context() as conn:
            precedents = await conn.fetch("""
                SELECT * FROM conflict_pattern_memory
                WHERE conflict_id = $1
                AND event_type = $2
                AND outcome IS NOT NULL
                ORDER BY game_day DESC
                LIMIT 5
            """, conflict_id, situation_type)
        
        return [
            {
                'description': p['description'],
                'outcome': p['outcome'],
                'days_ago': p.get('days_ago', 0),
                'can_reference': True
            }
            for p in precedents
        ]


# ===============================================================================
# AUTONOMOUS STAKEHOLDER MANAGER
# ===============================================================================

class AutonomousStakeholderManager:
    """Manages autonomous NPC actions in slice-of-life conflicts"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.action_agent = self._create_action_agent()
        self.memory = SliceOfLifeConflictMemory(user_id, conversation_id)
        
    def _create_action_agent(self) -> Agent:
        """Create agent for generating autonomous NPC actions"""
        return Agent(
            name="Autonomous Action Generator",
            instructions="""
            Generate subtle autonomous actions NPCs take in daily conflicts.
            
            Actions should be:
            - Indirect and deniable
            - Embedded in normal routine
            - Building toward long-term goals
            - Respectful of established dynamics
            - Never confrontational
            
            Focus on:
            - Establishing helpful precedents
            - Creating fait accomplis
            - Building social pressure
            - Shifting dynamics gradually
            
            Examples:
            - Rearranging shared spaces while player is out
            - Starting new routines without asking
            - Mentioning preferences to other NPCs
            - Buying items that shape future decisions
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def generate_autonomous_actions(
        self,
        conflict_id: int,
        stakeholder_npcs: List[int],
        current_phase: str,
        player_location: str
    ) -> List[AutonomousAction]:
        """Generate autonomous actions for NPC stakeholders"""
        
        actions = []
        
        for npc_id in stakeholder_npcs:
            # Get NPC data and conflict position
            npc_data = await self._get_npc_context(npc_id, conflict_id)
            
            # Check if NPC can act
            if not await self._can_npc_act(npc_id, player_location):
                continue
            
            # Generate appropriate action
            action = await self._generate_npc_action(
                npc_id, npc_data, conflict_id, current_phase
            )
            
            if action:
                actions.append(action)
                
                # Execute the action
                await self._execute_autonomous_action(action, conflict_id)
        
        return actions
    
    async def _get_npc_context(
        self,
        npc_id: int,
        conflict_id: int
    ) -> Dict[str, Any]:
        """Get NPC's context for decision making"""
        
        async with get_db_connection_context() as conn:
            # Get NPC stats
            npc = await conn.fetchrow("""
                SELECT * FROM NPCStats WHERE npc_id = $1
            """, npc_id)
            
            # Get conflict position
            position = await conn.fetchrow("""
                SELECT * FROM conflict_stakeholders
                WHERE conflict_id = $1 AND npc_id = $2
            """, conflict_id, npc_id)
            
            # Get recent patterns
            patterns = await self.memory.detect_patterns(conflict_id)
        
        return {
            'npc_id': npc_id,
            'personality': json.loads(npc['personality_traits'] or '{}'),
            'dominance': npc['dominance'],
            'desired_outcome': position['desired_outcome'],
            'recent_patterns': patterns
        }
    
    async def _can_npc_act(
        self,
        npc_id: int,
        player_location: str
    ) -> bool:
        """Check if NPC can take autonomous action"""
        
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT current_location, is_available FROM NPCStats
                WHERE npc_id = $1
            """, npc_id)
        
        # NPC can act if they're available and either:
        # 1. In same location as player (direct action)
        # 2. In different location (indirect action)
        return npc['is_available']
    
    async def _generate_npc_action(
        self,
        npc_id: int,
        npc_data: Dict,
        conflict_id: int,
        current_phase: str
    ) -> Optional[AutonomousAction]:
        """Generate an appropriate autonomous action"""
        
        prompt = f"""
        NPC context: {json.dumps(npc_data)}
        Conflict phase: {current_phase}
        
        Generate a subtle autonomous action this NPC would take.
        
        Return JSON:
        {{
            "action_type": "establish_precedent/reinforce_pattern/subtle_rebellion/etc",
            "description": "What the NPC does (1-2 sentences)",
            "target": "player/routine/other_npc/environment",
            "subtlety_level": 0.0-1.0,
            "impact": {{
                "control": -0.1 to 0.1,
                "precedent": 0.0 to 0.2,
                "tension": -0.1 to 0.1
            }},
            "requires_player_presence": true/false,
            "can_be_discovered_later": true/false,
            "reasoning": "Why NPC chose this action"
        }}
        
        Keep actions subtle and deniable.
        """
        
        response = await Runner.run(self.action_agent, prompt)
        action_data = json.loads(response.output)
        
        return AutonomousAction(
            npc_id=npc_id,
            action_type=AutonomousActionType[action_data['action_type'].upper()],
            description=action_data['description'],
            target=action_data['target'],
            subtlety_level=action_data['subtlety_level'],
            impact=action_data['impact'],
            requires_player_presence=action_data['requires_player_presence'],
            can_be_discovered_later=action_data['can_be_discovered_later']
        )
    
    async def _execute_autonomous_action(
        self,
        action: AutonomousAction,
        conflict_id: int
    ):
        """Execute an autonomous action and record its effects"""
        
        # Record the action
        await self.memory.record_micro_event(
            conflict_id=conflict_id,
            event_type=action.action_type.value,
            description=action.description,
            participants=[action.npc_id],
            outcome=None  # Outcome determined later
        )
        
        # Apply impacts
        async with get_db_connection_context() as conn:
            # Update conflict dynamics
            for dimension, change in action.impact.items():
                await conn.execute("""
                    INSERT INTO conflict_dynamics
                    (conflict_id, dimension, current_value, last_change)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (conflict_id, dimension)
                    DO UPDATE SET 
                        current_value = conflict_dynamics.current_value + $4,
                        last_change = $4
                """, conflict_id, dimension, change, change)
        
        # Store for potential discovery
        if action.can_be_discovered_later:
            await self._store_discoverable_action(action, conflict_id)
    
    async def _store_discoverable_action(
        self,
        action: AutonomousAction,
        conflict_id: int
    ):
        """Store action for potential later discovery"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO discoverable_actions
                (conflict_id, npc_id, action_description, 
                 subtlety_level, discovered, game_day)
                VALUES ($1, $2, $3, $4, false,
                        (SELECT current_day FROM game_calendar 
                         WHERE user_id = $5 AND conversation_id = $6))
            """, conflict_id, action.npc_id, action.description,
            action.subtlety_level, self.user_id, self.conversation_id)


# ===============================================================================
# CONSEQUENCE AND REWARD ADAPTATION
# ===============================================================================

@dataclass
class SliceOfLifeConsequence:
    """Consequences in slice-of-life conflicts"""
    type: str  # "pattern_established", "privilege_gained", "boundary_set"
    description: str
    affects_routine: bool
    social_impact: Dict[str, float]
    lasting_duration: Optional[int]  # Days, or None for permanent


class ConsequenceGenerator:
    """Generates appropriate consequences for resolved conflicts"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.consequence_agent = Agent(
            name="Consequence Generator",
            instructions="""
            Generate realistic consequences for slice-of-life conflicts.
            
            Consequences should be:
            - Natural results of established patterns
            - Affecting daily routines and dynamics
            - Subtle but meaningful
            - Creating new status quos
            
            Focus on:
            - Who makes decisions going forward
            - Changes in daily routines
            - Shifted relationship dynamics
            - New unspoken rules
            - Social standing adjustments
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def generate_consequences(
        self,
        conflict_id: int,
        resolution_type: str,
        winner_npcs: List[int],
        loser_npcs: List[int]
    ) -> List[SliceOfLifeConsequence]:
        """Generate consequences based on conflict resolution"""
        
        prompt = f"""
        Conflict resolved via: {resolution_type}
        Winners: {winner_npcs}
        Losers: {loser_npcs}
        
        Generate realistic daily-life consequences.
        
        Return JSON array:
        [{{
            "type": "pattern_established/privilege_gained/boundary_set/routine_changed",
            "description": "Specific consequence",
            "affects_routine": true/false,
            "social_impact": {{
                "respect": -0.1 to 0.1,
                "autonomy": -0.1 to 0.1,
                "social_standing": -0.1 to 0.1
            }},
            "lasting_duration": days or null for permanent,
            "example_manifestation": "How this shows up in daily life"
        }}]
        """
        
        response = await Runner.run(self.consequence_agent, prompt)
        consequences_data = json.loads(response.output)
        
        consequences = []
        for cons_data in consequences_data:
            consequences.append(SliceOfLifeConsequence(
                type=cons_data['type'],
                description=cons_data['description'],
                affects_routine=cons_data['affects_routine'],
                social_impact=cons_data['social_impact'],
                lasting_duration=cons_data['lasting_duration']
            ))
        
        # Apply consequences
        await self._apply_consequences(conflict_id, consequences)
        
        return consequences
    
    async def _apply_consequences(
        self,
        conflict_id: int,
        consequences: List[SliceOfLifeConsequence]
    ):
        """Apply consequences to game state"""
        
        async with get_db_connection_context() as conn:
            for consequence in consequences:
                # Record consequence
                await conn.execute("""
                    INSERT INTO conflict_consequences
                    (conflict_id, consequence_type, description,
                     affects_routine, social_impact, lasting_until)
                    VALUES ($1, $2, $3, $4, $5,
                            CASE WHEN $6 IS NOT NULL 
                            THEN (SELECT current_day + $6 FROM game_calendar
                                  WHERE user_id = $7 AND conversation_id = $8)
                            ELSE NULL END)
                """, conflict_id, consequence.type, consequence.description,
                consequence.affects_routine, json.dumps(consequence.social_impact),
                consequence.lasting_duration, self.user_id, self.conversation_id)


# ===============================================================================
# SECRET AND HIDDEN DYNAMICS
# ===============================================================================

class HiddenDynamicsManager:
    """Manages hidden aspects of slice-of-life conflicts"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.secret_agent = Agent(
            name="Hidden Dynamics Generator",
            instructions="""
            Generate hidden dynamics in daily conflicts.
            
            Focus on:
            - Unspoken desires about routines
            - Hidden resentments from patterns
            - Secret alliances between NPCs
            - Private goals for domestic control
            - Undisclosed preferences
            
            Keep secrets realistic and grounded in daily life.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
    
    async def generate_hidden_dynamics(
        self,
        conflict_id: int,
        npc_id: int
    ) -> Dict[str, Any]:
        """Generate hidden dynamics for an NPC in conflict"""
        
        # Get NPC context
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT * FROM NPCStats WHERE npc_id = $1
            """, npc_id)
            
            position = await conn.fetchrow("""
                SELECT * FROM conflict_stakeholders
                WHERE conflict_id = $1 AND npc_id = $2
            """, conflict_id, npc_id)
        
        prompt = f"""
        NPC personality: {json.dumps(json.loads(npc['personality_traits'] or '{}'))}
        Public position: {position['desired_outcome']}
        
        Generate hidden dynamics.
        
        Return JSON:
        {{
            "true_desire": "What they really want",
            "hidden_resentment": "What they're upset about",
            "secret_preference": "Unexpressed preference",
            "private_goal": "Long-term aim",
            "will_reveal_if": "Condition for revelation",
            "signs_if_observant": ["Subtle clue 1", "Subtle clue 2"]
        }}
        """
        
        response = await Runner.run(self.secret_agent, prompt)
        return json.loads(response.output)


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def process_autonomous_stakeholder_actions(
    ctx: RunContextWrapper,
    conflict_id: int,
    time_phase: str
) -> List[Dict[str, Any]]:
    """Process autonomous NPC actions for a conflict during a time phase"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = AutonomousStakeholderManager(user_id, conversation_id)
    
    # Get stakeholder NPCs
    async with get_db_connection_context() as conn:
        stakeholders = await conn.fetch("""
            SELECT npc_id FROM conflict_stakeholders
            WHERE conflict_id = $1
        """, conflict_id)
        
        player_location = await conn.fetchval("""
            SELECT current_location FROM player_state
            WHERE user_id = $1 AND conversation_id = $2
        """, user_id, conversation_id)
    
    npc_ids = [s['npc_id'] for s in stakeholders]
    
    # Generate autonomous actions
    actions = await manager.generate_autonomous_actions(
        conflict_id, npc_ids, time_phase, player_location
    )
    
    # Format for display
    return [
        {
            'npc_id': action.npc_id,
            'action': action.description,
            'noticeable': action.subtlety_level < 0.5,
            'requires_presence': action.requires_player_presence
        }
        for action in actions
    ]


@function_tool
async def discover_past_actions(
    ctx: RunContextWrapper,
    days_back: int = 3
) -> List[Dict[str, Any]]:
    """Discover autonomous actions NPCs took while player was away"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    discoveries = []
    
    async with get_db_connection_context() as conn:
        actions = await conn.fetch("""
            SELECT * FROM discoverable_actions
            WHERE user_id = $1 AND conversation_id = $2
            AND discovered = false
            AND game_day > (SELECT current_day - $3 FROM game_calendar
                           WHERE user_id = $1 AND conversation_id = $2)
            AND random() < (1.0 - subtlety_level)  -- Chance based on subtlety
        """, user_id, conversation_id, days_back)
        
        for action in actions:
            discoveries.append({
                'npc_id': action['npc_id'],
                'action': action['action_description'],
                'when': f"{action['game_day']} days ago",
                'player_response_options': [
                    'Accept the change',
                    'Quietly undo it',
                    'Address it directly',
                    'Ignore but remember'
                ]
            })
            
            # Mark as discovered
            await conn.execute("""
                UPDATE discoverable_actions
                SET discovered = true
                WHERE action_id = $1
            """, action['action_id'])
    
    return discoveries
