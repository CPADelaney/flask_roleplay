# logic/conflict_system/autonomous_stakeholder_actions.py
"""
Autonomous Stakeholder Actions System with LLM-generated decisions.
Manages NPC decision-making, reactions, and autonomous actions in conflicts.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# STAKEHOLDER STRUCTURES
# ===============================================================================

class StakeholderRole(Enum):
    """Roles stakeholders can take in conflicts"""
    INSTIGATOR = "instigator"
    DEFENDER = "defender"
    MEDIATOR = "mediator"
    OPPORTUNIST = "opportunist"
    VICTIM = "victim"
    BYSTANDER = "bystander"
    ESCALATOR = "escalator"
    PEACEMAKER = "peacemaker"

class ActionType(Enum):
    """Types of actions stakeholders can take"""
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    DIPLOMATIC = "diplomatic"
    MANIPULATIVE = "manipulative"
    SUPPORTIVE = "supportive"
    EVASIVE = "evasive"
    OBSERVANT = "observant"
    STRATEGIC = "strategic"

class DecisionStyle(Enum):
    """How stakeholders make decisions"""
    EMOTIONAL = "emotional"  # Driven by feelings
    RATIONAL = "rational"  # Logic-based
    INSTINCTIVE = "instinctive"  # Gut reactions
    CALCULATING = "calculating"  # Long-term planning
    REACTIVE = "reactive"  # Response to immediate stimuli
    PRINCIPLED = "principled"  # Based on values

@dataclass
class Stakeholder:
    """An NPC stakeholder in a conflict"""
    stakeholder_id: int
    npc_id: int
    name: str
    personality_traits: List[str]
    current_role: StakeholderRole
    decision_style: DecisionStyle
    goals: List[str]
    resources: Dict[str, float]
    relationships: Dict[int, float]  # Other stakeholder_id -> relationship value
    stress_level: float  # 0-1, affects decision quality
    commitment_level: float  # 0-1, how invested in conflict

@dataclass
class StakeholderAction:
    """An action taken by a stakeholder"""
    action_id: int
    stakeholder_id: int
    action_type: ActionType
    description: str
    target: Optional[int]  # Target stakeholder_id if applicable
    resources_used: Dict[str, float]
    success_probability: float
    consequences: Dict[str, Any]
    timestamp: datetime

@dataclass
class StakeholderReaction:
    """A reaction to another stakeholder's action"""
    reaction_id: int
    stakeholder_id: int
    triggering_action_id: int
    reaction_type: str  # "counter", "support", "ignore", "escalate", "de-escalate"
    description: str
    emotional_response: str
    relationship_impact: Dict[int, float]

@dataclass
class StakeholderStrategy:
    """A long-term strategy for a stakeholder"""
    strategy_id: int
    stakeholder_id: int
    strategy_name: str
    objectives: List[str]
    tactics: List[str]
    success_conditions: List[str]
    abandon_conditions: List[str]
    time_horizon: str  # "immediate", "short-term", "long-term"

# ===============================================================================
# AUTONOMOUS STAKEHOLDER MANAGER WITH LLM
# ===============================================================================

class StakeholderAutonomySystem:
    """
    Manages autonomous NPC actions using LLM for intelligent decision-making.
    Creates realistic, personality-driven stakeholder behaviors.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.manager = AutonomousStakeholderManager(user_id, conversation_id)
        self._decision_maker = None
        self._reaction_generator = None
        self._strategy_planner = None
        self._personality_analyzer = None
    
    @property
    def decision_maker(self) -> Agent:
        """Agent for making stakeholder decisions"""
        if self._decision_maker is None:
            self._decision_maker = Agent(
                name="Stakeholder Decision Maker",
                instructions="""
                Make decisions for NPC stakeholders in conflicts.
                
                Consider:
                - Personality traits and values
                - Current emotional state and stress
                - Available resources and constraints
                - Relationships and alliances
                - Long-term goals vs immediate needs
                - Risk tolerance and decision style
                
                Generate decisions that are:
                - True to character personality
                - Contextually appropriate
                - Strategically reasonable
                - Dramatically interesting
                - Varied based on individual traits
                
                Think like each unique character would think.
                """,
                model="gpt-5-nano",
            )
        return self._decision_maker
    
    @property
    def reaction_generator(self) -> Agent:
        """Agent for generating stakeholder reactions"""
        if self._reaction_generator is None:
            self._reaction_generator = Agent(
                name="Stakeholder Reaction Generator",
                instructions="""
                Generate realistic reactions to events and actions.
                
                Consider:
                - Personality and emotional state
                - Relationship with action initiator
                - Personal stakes in the conflict
                - Past experiences and patterns
                - Cultural and social norms
                
                Create reactions that:
                - Feel authentic to the character
                - Show emotional depth
                - Advance the conflict naturally
                - Vary based on context
                - Include both immediate and considered responses
                
                Balance emotional authenticity with strategic thinking.
                """,
                model="gpt-5-nano",
            )
        return self._reaction_generator
    
    @property
    def strategy_planner(self) -> Agent:
        """Agent for planning stakeholder strategies"""
        if self._strategy_planner is None:
            self._strategy_planner = Agent(
                name="Stakeholder Strategy Planner",
                instructions="""
                Develop long-term strategies for stakeholders.
                
                Consider:
                - Character's intelligence and foresight
                - Available resources and allies
                - Conflict dynamics and power balance
                - Personal values and red lines
                - Risk vs reward calculations
                
                Create strategies that:
                - Match character capabilities
                - Have clear objectives
                - Include contingency plans
                - Adapt to changing situations
                - Feel realistic, not omniscient
                
                Think like a character planning within their limitations.
                """,
                model="gpt-5-nano",
            )
        return self._strategy_planner
    
    @property
    def personality_analyzer(self) -> Agent:
        """Agent for analyzing personality influence on actions"""
        if self._personality_analyzer is None:
            self._personality_analyzer = Agent(
                name="Personality Analyzer",
                instructions="""
                Analyze how personality traits affect decisions.
                
                Consider:
                - Core personality traits
                - Stress responses and coping mechanisms
                - Decision-making patterns
                - Emotional regulation abilities
                - Social tendencies
                
                Provide insights on:
                - Likely behavioral patterns
                - Stress breaking points
                - Relationship dynamics
                - Decision biases
                - Character growth potential
                
                Create psychologically consistent characters.
                """,
                model="gpt-5-nano",
            )
        return self._personality_analyzer
    
    # ========== Stakeholder Creation ==========
    
    async def create_stakeholder(
        self,
        npc_id: int,
        conflict_id: int,
        initial_role: Optional[str] = None
    ) -> Stakeholder:
        """Create a stakeholder with personality-driven characteristics"""
        
        # Get NPC details
        npc_details = await self._get_npc_details(npc_id)
        
        prompt = f"""
        Create stakeholder profile:
        
        NPC: {npc_details.get('name', 'Unknown')}
        Personality: {npc_details.get('personality_traits', 'Unknown')}
        Conflict Context: Conflict #{conflict_id}
        Suggested Role: {initial_role or 'determine based on personality'}
        
        Generate:
        1. Stakeholder role (instigator/defender/mediator/opportunist/victim/bystander/escalator/peacemaker)
        2. Decision style (emotional/rational/instinctive/calculating/reactive/principled)
        3. 3-4 specific goals in this conflict
        4. Initial stress level (0-1)
        5. Commitment level (0-1)
        6. Key resources they bring
        
        Match everything to personality.
        Format as JSON.
        """
        
        response = await self.personality_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store stakeholder
            async with get_db_connection_context() as conn:
                stakeholder_id = await conn.fetchval("""
                    INSERT INTO stakeholders
                    (user_id, conversation_id, npc_id, conflict_id,
                     role, decision_style, stress_level, commitment_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING stakeholder_id
                """, self.user_id, self.conversation_id, npc_id, conflict_id,
                result.get('role', 'bystander'),
                result.get('decision_style', 'reactive'),
                result.get('stress_level', 0.3),
                result.get('commitment_level', 0.5))
            
            return Stakeholder(
                stakeholder_id=stakeholder_id,
                npc_id=npc_id,
                name=npc_details.get('name', 'Unknown'),
                personality_traits=json.loads(npc_details.get('personality_traits', '[]')),
                current_role=StakeholderRole[result.get('role', 'BYSTANDER').upper()],
                decision_style=DecisionStyle[result.get('decision_style', 'REACTIVE').upper()],
                goals=result.get('goals', []),
                resources=result.get('resources', {}),
                relationships={},
                stress_level=result.get('stress_level', 0.3),
                commitment_level=result.get('commitment_level', 0.5)
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to create stakeholder: {e}")
            return self._create_fallback_stakeholder(npc_id, npc_details)
    
    # ========== Decision Making ==========
    
    async def make_autonomous_decision(
        self,
        stakeholder: Stakeholder,
        conflict_state: Dict[str, Any],
        available_options: Optional[List[str]] = None
    ) -> StakeholderAction:
        """Make an autonomous decision for a stakeholder"""
        
        prompt = f"""
        Make decision for stakeholder:
        
        Character: {stakeholder.name}
        Personality: {stakeholder.personality_traits}
        Role: {stakeholder.current_role.value}
        Decision Style: {stakeholder.decision_style.value}
        Goals: {json.dumps(stakeholder.goals)}
        Stress Level: {stakeholder.stress_level}
        
        Conflict State: {json.dumps(conflict_state, indent=2)}
        Available Options: {json.dumps(available_options) if available_options else 'Generate appropriate action'}
        
        Decide:
        1. Action type (aggressive/defensive/diplomatic/manipulative/supportive/evasive/observant/strategic)
        2. Specific action description
        3. Target (if applicable)
        4. Resources to commit
        5. Success probability (0-1, be realistic)
        6. Rationale (why this action, in character)
        
        Make decision true to personality and stress level.
        Format as JSON.
        """
        
        response = await self.decision_maker.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store action
            async with get_db_connection_context() as conn:
                action_id = await conn.fetchval("""
                    INSERT INTO stakeholder_actions
                    (user_id, conversation_id, stakeholder_id, action_type,
                     description, success_probability)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING action_id
                """, self.user_id, self.conversation_id,
                stakeholder.stakeholder_id,
                result.get('action_type', 'observant'),
                result.get('description', 'Observes the situation'),
                result.get('success_probability', 0.5))
            
            return StakeholderAction(
                action_id=action_id,
                stakeholder_id=stakeholder.stakeholder_id,
                action_type=ActionType[result.get('action_type', 'OBSERVANT').upper()],
                description=result.get('description', 'Takes action'),
                target=result.get('target'),
                resources_used=result.get('resources', {}),
                success_probability=result.get('success_probability', 0.5),
                consequences=result.get('consequences', {}),
                timestamp=datetime.now()
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to make decision: {e}")
            return self._create_fallback_action(stakeholder)
    
    # ========== Reaction System ==========

    async def process_event(self, conflict_id: int, event: Dict[str, Any]):
        """Process events for stakeholder reactions"""
        # Delegate to existing manager methods
        return await self.manager.process_stakeholder_event(conflict_id, event)
    
    async def generate_reaction(
        self,
        stakeholder: Stakeholder,
        triggering_action: StakeholderAction,
        action_context: Dict[str, Any]
    ) -> StakeholderReaction:
        """Generate a reaction to another stakeholder's action"""
        
        prompt = f"""
        Generate reaction:
        
        Reacting Character: {stakeholder.name}
        Personality: {stakeholder.personality_traits}
        Current Stress: {stakeholder.stress_level}
        Relationship with Actor: {stakeholder.relationships.get(triggering_action.stakeholder_id, 0.5)}
        
        Triggering Action: {triggering_action.description}
        Action Type: {triggering_action.action_type.value}
        Context: {json.dumps(action_context, indent=2)}
        
        Generate:
        1. Reaction type (counter/support/ignore/escalate/de-escalate)
        2. Specific reaction description
        3. Emotional response (how they feel)
        4. Relationship impact (-1 to 1)
        5. Stress impact on self (-1 to 1)
        6. Follow-up intentions
        
        Make reaction authentic to character.
        Format as JSON.
        """
        
        response = await self.reaction_generator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store reaction
            async with get_db_connection_context() as conn:
                reaction_id = await conn.fetchval("""
                    INSERT INTO stakeholder_reactions
                    (user_id, conversation_id, stakeholder_id, triggering_action_id,
                     reaction_type, description, emotional_response)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING reaction_id
                """, self.user_id, self.conversation_id,
                stakeholder.stakeholder_id, triggering_action.action_id,
                result.get('reaction_type', 'observe'),
                result.get('description', 'Reacts to the action'),
                result.get('emotional_response', 'neutral'))
            
            # Update stakeholder stress
            stakeholder.stress_level = max(0, min(1, 
                stakeholder.stress_level + result.get('stress_impact', 0)))
            
            return StakeholderReaction(
                reaction_id=reaction_id,
                stakeholder_id=stakeholder.stakeholder_id,
                triggering_action_id=triggering_action.action_id,
                reaction_type=result.get('reaction_type', 'observe'),
                description=result.get('description', 'Reacts'),
                emotional_response=result.get('emotional_response', 'neutral'),
                relationship_impact={
                    triggering_action.stakeholder_id: result.get('relationship_impact', 0)
                }
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to generate reaction: {e}")
            return self._create_fallback_reaction(stakeholder, triggering_action)
    
    # ========== Strategy Planning ==========
    
    async def develop_strategy(
        self,
        stakeholder: Stakeholder,
        conflict_analysis: Dict[str, Any]
    ) -> StakeholderStrategy:
        """Develop a long-term strategy for a stakeholder"""
        
        prompt = f"""
        Develop strategy for stakeholder:
        
        Character: {stakeholder.name}
        Personality: {stakeholder.personality_traits}
        Decision Style: {stakeholder.decision_style.value}
        Goals: {json.dumps(stakeholder.goals)}
        Resources: {json.dumps(stakeholder.resources)}
        
        Conflict Analysis: {json.dumps(conflict_analysis, indent=2)}
        
        Create strategy:
        1. Strategy name (brief, descriptive)
        2. 3-4 key objectives
        3. 4-5 specific tactics
        4. Success conditions (when to consider it successful)
        5. Abandon conditions (when to change strategy)
        6. Time horizon (immediate/short-term/long-term)
        
        Match strategy to character's capabilities and style.
        Format as JSON.
        """
        
        response = await self.strategy_planner.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store strategy
            async with get_db_connection_context() as conn:
                strategy_id = await conn.fetchval("""
                    INSERT INTO stakeholder_strategies
                    (user_id, conversation_id, stakeholder_id, strategy_name,
                     objectives, tactics, time_horizon)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING strategy_id
                """, self.user_id, self.conversation_id,
                stakeholder.stakeholder_id,
                result.get('name', 'Default Strategy'),
                json.dumps(result.get('objectives', [])),
                json.dumps(result.get('tactics', [])),
                result.get('time_horizon', 'short-term'))
            
            return StakeholderStrategy(
                strategy_id=strategy_id,
                stakeholder_id=stakeholder.stakeholder_id,
                strategy_name=result.get('name', 'Default Strategy'),
                objectives=result.get('objectives', []),
                tactics=result.get('tactics', []),
                success_conditions=result.get('success_conditions', []),
                abandon_conditions=result.get('abandon_conditions', []),
                time_horizon=result.get('time_horizon', 'short-term')
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to develop strategy: {e}")
            return self._create_fallback_strategy(stakeholder)
    
    # ========== Stress and Adaptation ==========
    
    async def update_stakeholder_stress(
        self,
        stakeholder: Stakeholder,
        stressor: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update stakeholder stress and potential breaking point"""
        
        prompt = f"""
        Analyze stress impact:
        
        Character: {stakeholder.name}
        Personality: {stakeholder.personality_traits}
        Current Stress: {stakeholder.stress_level}
        
        Stressor: {json.dumps(stressor, indent=2)}
        
        Determine:
        1. Stress change (-1 to 1)
        2. Coping mechanism used
        3. Behavioral changes
        4. Breaking point reached? (yes/no)
        5. If breaking: What happens?
        
        Consider personality's stress resilience.
        Format as JSON.
        """
        
        response = await self.personality_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Update stress
            old_stress = stakeholder.stress_level
            stakeholder.stress_level = max(0, min(1, 
                stakeholder.stress_level + result.get('stress_change', 0)))
            
            # Handle breaking point
            breaking_point_action = None
            if result.get('breaking_point'):
                breaking_point_action = await self._handle_breaking_point(
                    stakeholder,
                    result.get('breaking_action', 'withdraws from conflict')
                )
            
            return {
                'old_stress': old_stress,
                'new_stress': stakeholder.stress_level,
                'coping_mechanism': result.get('coping', 'endures'),
                'behavioral_changes': result.get('changes', []),
                'breaking_point': result.get('breaking_point', False),
                'breaking_action': breaking_point_action
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to update stress: {e}")
            return {'new_stress': stakeholder.stress_level}
    
    async def adapt_stakeholder_role(
        self,
        stakeholder: Stakeholder,
        changing_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt stakeholder role based on changing conditions"""
        
        prompt = f"""
        Evaluate role adaptation:
        
        Character: {stakeholder.name}
        Current Role: {stakeholder.current_role.value}
        Personality: {stakeholder.personality_traits}
        Stress: {stakeholder.stress_level}
        
        Changing Conditions: {json.dumps(changing_conditions, indent=2)}
        
        Determine:
        1. Should change role? (yes/no)
        2. If yes, new role
        3. Reason for change
        4. How the change manifests
        5. Impact on behavior
        
        Make changes feel organic to character development.
        Format as JSON.
        """
        
        response = await self.personality_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            if result.get('change_role'):
                old_role = stakeholder.current_role
                stakeholder.current_role = StakeholderRole[
                    result.get('new_role', 'BYSTANDER').upper()
                ]
                
                # Store role change
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE stakeholders
                        SET role = $1
                        WHERE stakeholder_id = $2
                    """, stakeholder.current_role.value, stakeholder.stakeholder_id)
                
                return {
                    'role_changed': True,
                    'old_role': old_role.value,
                    'new_role': stakeholder.current_role.value,
                    'reason': result.get('reason', 'Circumstances changed'),
                    'manifestation': result.get('manifestation', 'Behavior shifts'),
                    'behavioral_impact': result.get('impact', [])
                }
            
            return {'role_changed': False}
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to adapt role: {e}")
            return {'role_changed': False}
    
    # ========== Helper Methods ==========
    
    async def _get_npc_details(self, npc_id: int) -> Dict:
        """Get NPC details from database"""
        
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT * FROM NPCs WHERE npc_id = $1
            """, npc_id)
        
        return dict(npc) if npc else {}
    
    async def _handle_breaking_point(
        self,
        stakeholder: Stakeholder,
        breaking_action: str
    ) -> Dict[str, Any]:
        """Handle a stakeholder reaching their breaking point"""
        
        # Create dramatic action
        action = StakeholderAction(
            action_id=0,
            stakeholder_id=stakeholder.stakeholder_id,
            action_type=ActionType.EMOTIONAL,
            description=breaking_action,
            target=None,
            resources_used={},
            success_probability=1.0,
            consequences={'type': 'breaking_point'},
            timestamp=datetime.now()
        )
        
        # Update stakeholder state
        stakeholder.commitment_level = 0.0
        stakeholder.current_role = StakeholderRole.BYSTANDER
        
        return {
            'action': breaking_action,
            'stakeholder_withdraws': True,
            'stress_relief': 0.5
        }
    
    def _create_fallback_stakeholder(
        self,
        npc_id: int,
        npc_details: Dict
    ) -> Stakeholder:
        """Create fallback stakeholder if LLM fails"""
        
        return Stakeholder(
            stakeholder_id=0,
            npc_id=npc_id,
            name=npc_details.get('name', 'Unknown'),
            personality_traits=[],
            current_role=StakeholderRole.BYSTANDER,
            decision_style=DecisionStyle.REACTIVE,
            goals=[],
            resources={},
            relationships={},
            stress_level=0.3,
            commitment_level=0.5
        )
    
    def _create_fallback_action(self, stakeholder: Stakeholder) -> StakeholderAction:
        """Create fallback action if LLM fails"""
        
        return StakeholderAction(
            action_id=0,
            stakeholder_id=stakeholder.stakeholder_id,
            action_type=ActionType.OBSERVANT,
            description="Observes the situation",
            target=None,
            resources_used={},
            success_probability=0.7,
            consequences={},
            timestamp=datetime.now()
        )
    
    def _create_fallback_reaction(
        self,
        stakeholder: Stakeholder,
        action: StakeholderAction
    ) -> StakeholderReaction:
        """Create fallback reaction if LLM fails"""
        
        return StakeholderReaction(
            reaction_id=0,
            stakeholder_id=stakeholder.stakeholder_id,
            triggering_action_id=action.action_id,
            reaction_type="observe",
            description="Notices the action",
            emotional_response="neutral",
            relationship_impact={}
        )
    
    def _create_fallback_strategy(self, stakeholder: Stakeholder) -> StakeholderStrategy:
        """Create fallback strategy if LLM fails"""
        
        return StakeholderStrategy(
            strategy_id=0,
            stakeholder_id=stakeholder.stakeholder_id,
            strategy_name="Wait and See",
            objectives=["Survive", "Minimize losses"],
            tactics=["Observe", "React carefully"],
            success_conditions=["Conflict ends"],
            abandon_conditions=["Direct threat"],
            time_horizon="short-term"
        )

# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def create_conflict_stakeholder(
    ctx: RunContextWrapper,
    npc_id: int,
    conflict_id: int,
    suggested_role: Optional[str] = None
) -> Dict[str, Any]:
    """Create a stakeholder for a conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = AutonomousStakeholderManager(user_id, conversation_id)
    
    stakeholder = await manager.create_stakeholder(npc_id, conflict_id, suggested_role)
    
    return {
        'stakeholder_id': stakeholder.stakeholder_id,
        'npc_id': stakeholder.npc_id,
        'name': stakeholder.name,
        'role': stakeholder.current_role.value,
        'decision_style': stakeholder.decision_style.value,
        'goals': stakeholder.goals,
        'stress_level': stakeholder.stress_level,
        'commitment_level': stakeholder.commitment_level
    }

@function_tool
async def stakeholder_take_action(
    ctx: RunContextWrapper,
    stakeholder_id: int,
    conflict_state: Dict[str, Any],
    options: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Have a stakeholder take an autonomous action"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = AutonomousStakeholderManager(user_id, conversation_id)
    
    # Get stakeholder details
    async with get_db_connection_context() as conn:
        stakeholder_data = await conn.fetchrow("""
            SELECT s.*, n.name, n.personality_traits
            FROM stakeholders s
            JOIN NPCs n ON s.npc_id = n.npc_id
            WHERE s.stakeholder_id = $1
        """, stakeholder_id)
    
    if not stakeholder_data:
        return {'error': 'Stakeholder not found'}
    
    # Create stakeholder object
    stakeholder = Stakeholder(
        stakeholder_id=stakeholder_id,
        npc_id=stakeholder_data['npc_id'],
        name=stakeholder_data['name'],
        personality_traits=json.loads(stakeholder_data.get('personality_traits', '[]')),
        current_role=StakeholderRole[stakeholder_data['role'].upper()],
        decision_style=DecisionStyle[stakeholder_data['decision_style'].upper()],
        goals=[],
        resources={},
        relationships={},
        stress_level=stakeholder_data['stress_level'],
        commitment_level=stakeholder_data['commitment_level']
    )
    
    action = await manager.make_autonomous_decision(stakeholder, conflict_state, options)
    
    return {
        'action_id': action.action_id,
        'action_type': action.action_type.value,
        'description': action.description,
        'target': action.target,
        'success_probability': action.success_probability,
        'resources_used': action.resources_used
    }

@function_tool
async def generate_stakeholder_reaction(
    ctx: RunContextWrapper,
    stakeholder_id: int,
    triggering_action_id: int,
    action_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate a stakeholder's reaction to an action"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = AutonomousStakeholderManager(user_id, conversation_id)
    
    # Get stakeholder and action details
    async with get_db_connection_context() as conn:
        stakeholder_data = await conn.fetchrow("""
            SELECT s.*, n.name, n.personality_traits
            FROM stakeholders s
            JOIN NPCs n ON s.npc_id = n.npc_id
            WHERE s.stakeholder_id = $1
        """, stakeholder_id)
        
        action_data = await conn.fetchrow("""
            SELECT * FROM stakeholder_actions
            WHERE action_id = $1
        """, triggering_action_id)
    
    if not stakeholder_data or not action_data:
        return {'error': 'Data not found'}
    
    # Create objects
    stakeholder = Stakeholder(
        stakeholder_id=stakeholder_id,
        npc_id=stakeholder_data['npc_id'],
        name=stakeholder_data['name'],
        personality_traits=json.loads(stakeholder_data.get('personality_traits', '[]')),
        current_role=StakeholderRole[stakeholder_data['role'].upper()],
        decision_style=DecisionStyle[stakeholder_data['decision_style'].upper()],
        goals=[],
        resources={},
        relationships={},
        stress_level=stakeholder_data['stress_level'],
        commitment_level=stakeholder_data['commitment_level']
    )
    
    action = StakeholderAction(
        action_id=triggering_action_id,
        stakeholder_id=action_data['stakeholder_id'],
        action_type=ActionType[action_data['action_type'].upper()],
        description=action_data['description'],
        target=action_data.get('target'),
        resources_used={},
        success_probability=action_data['success_probability'],
        consequences={},
        timestamp=action_data['created_at']
    )
    
    reaction = await manager.generate_reaction(stakeholder, action, action_context or {})
    
    return {
        'reaction_id': reaction.reaction_id,
        'reaction_type': reaction.reaction_type,
        'description': reaction.description,
        'emotional_response': reaction.emotional_response,
        'relationship_impact': reaction.relationship_impact
    }

@function_tool
async def develop_stakeholder_strategy(
    ctx: RunContextWrapper,
    stakeholder_id: int,
    conflict_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Develop a strategy for a stakeholder"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = AutonomousStakeholderManager(user_id, conversation_id)
    
    # Get stakeholder details
    async with get_db_connection_context() as conn:
        stakeholder_data = await conn.fetchrow("""
            SELECT s.*, n.name, n.personality_traits
            FROM stakeholders s
            JOIN NPCs n ON s.npc_id = n.npc_id
            WHERE s.stakeholder_id = $1
        """, stakeholder_id)
    
    if not stakeholder_data:
        return {'error': 'Stakeholder not found'}
    
    stakeholder = Stakeholder(
        stakeholder_id=stakeholder_id,
        npc_id=stakeholder_data['npc_id'],
        name=stakeholder_data['name'],
        personality_traits=json.loads(stakeholder_data.get('personality_traits', '[]')),
        current_role=StakeholderRole[stakeholder_data['role'].upper()],
        decision_style=DecisionStyle[stakeholder_data['decision_style'].upper()],
        goals=[],
        resources={},
        relationships={},
        stress_level=stakeholder_data['stress_level'],
        commitment_level=stakeholder_data['commitment_level']
    )
    
    strategy = await manager.develop_strategy(stakeholder, conflict_analysis)
    
    return {
        'strategy_id': strategy.strategy_id,
        'strategy_name': strategy.strategy_name,
        'objectives': strategy.objectives,
        'tactics': strategy.tactics,
        'success_conditions': strategy.success_conditions,
        'abandon_conditions': strategy.abandon_conditions,
        'time_horizon': strategy.time_horizon
    }

@function_tool
async def update_stakeholder_stress(
    ctx: RunContextWrapper,
    stakeholder_id: int,
    stressor: Dict[str, Any]
) -> Dict[str, Any]:
    """Update a stakeholder's stress level"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = AutonomousStakeholderManager(user_id, conversation_id)
    
    # Get stakeholder
    async with get_db_connection_context() as conn:
        stakeholder_data = await conn.fetchrow("""
            SELECT s.*, n.name, n.personality_traits
            FROM stakeholders s
            JOIN NPCs n ON s.npc_id = n.npc_id
            WHERE s.stakeholder_id = $1
        """, stakeholder_id)
    
    if not stakeholder_data:
        return {'error': 'Stakeholder not found'}
    
    stakeholder = Stakeholder(
        stakeholder_id=stakeholder_id,
        npc_id=stakeholder_data['npc_id'],
        name=stakeholder_data['name'],
        personality_traits=json.loads(stakeholder_data.get('personality_traits', '[]')),
        current_role=StakeholderRole[stakeholder_data['role'].upper()],
        decision_style=DecisionStyle[stakeholder_data['decision_style'].upper()],
        goals=[],
        resources={},
        relationships={},
        stress_level=stakeholder_data['stress_level'],
        commitment_level=stakeholder_data['commitment_level']
    )
    
    result = await manager.update_stakeholder_stress(stakeholder, stressor)
    
    # Update database
    async with get_db_connection_context() as conn:
        await conn.execute("""
            UPDATE stakeholders
            SET stress_level = $1
            WHERE stakeholder_id = $2
        """, stakeholder.stress_level, stakeholder_id)
    
    return result
