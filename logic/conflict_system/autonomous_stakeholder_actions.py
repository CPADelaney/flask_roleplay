# logic/conflict_system/autonomous_stakeholder_actions.py
"""
Autonomous Stakeholder Actions System with LLM-generated decisions.
Refactored to work as a subsystem under the Conflict Synthesizer orchestrator.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import weakref

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

# Import orchestrator interfaces
from logic.conflict_system.conflict_synthesizer import (
    ConflictSubsystem,
    SubsystemType,
    EventType,
    SystemEvent,
    SubsystemResponse
)

logger = logging.getLogger(__name__)

# ===============================================================================
# STAKEHOLDER STRUCTURES (Preserved)
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
    EMOTIONAL = "emotional"
    RATIONAL = "rational"
    INSTINCTIVE = "instinctive"
    CALCULATING = "calculating"
    REACTIVE = "reactive"
    PRINCIPLED = "principled"

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
    relationships: Dict[int, float]
    stress_level: float
    commitment_level: float

@dataclass
class StakeholderAction:
    """An action taken by a stakeholder"""
    action_id: int
    stakeholder_id: int
    action_type: ActionType
    description: str
    target: Optional[int]
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
    reaction_type: str
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
    time_horizon: str

# ===============================================================================
# REFACTORED STAKEHOLDER AUTONOMY SYSTEM
# ===============================================================================

class StakeholderAutonomySystem(ConflictSubsystem):
    """
    Manages autonomous NPC actions as a subsystem under the orchestrator.
    Now implements ConflictSubsystem interface for proper integration.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._synthesizer = None  # Will be set by orchestrator
        
        # Stakeholder state
        self._active_stakeholders: Dict[int, Stakeholder] = {}
        self._pending_actions: List[StakeholderAction] = []
        self._strategy_cache: Dict[int, StakeholderStrategy] = {}
        
        # LLM agents (preserved)
        self._decision_maker = None
        self._reaction_generator = None
        self._strategy_planner = None
        self._personality_analyzer = None
    
    # ========== ConflictSubsystem Interface Implementation ==========
    
    @property
    def subsystem_type(self) -> SubsystemType:
        """Identify as stakeholder subsystem"""
        return SubsystemType.STAKEHOLDER
    
    @property
    def capabilities(self) -> Set[str]:
        """Capabilities this subsystem provides"""
        return {
            'create_stakeholder',
            'autonomous_decision',
            'generate_reaction',
            'develop_strategy',
            'update_stress',
            'adapt_role',
            'process_breaking_point',
            'manage_relationships'
        }
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        """Other subsystems we depend on"""
        return {
            SubsystemType.TENSION,  # Stress affects tension
            SubsystemType.SOCIAL,  # For relationship dynamics
            SubsystemType.FLOW,  # For timing decisions
        }
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        """Events we want to receive from orchestrator"""
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.PHASE_TRANSITION,
            EventType.PLAYER_CHOICE,
            EventType.TENSION_CHANGED,
            EventType.STATE_SYNC,
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize with synthesizer reference"""
        self._synthesizer = weakref.ref(synthesizer)
        
        # Load active stakeholders from DB
        await self._load_active_stakeholders()
        
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Handle events from the orchestrator"""
        
        try:
            # Route to appropriate handler
            handlers = {
                EventType.CONFLICT_CREATED: self._on_conflict_created,
                EventType.CONFLICT_UPDATED: self._on_conflict_updated,
                EventType.PHASE_TRANSITION: self._on_phase_transition,
                EventType.PLAYER_CHOICE: self._on_player_choice,
                EventType.TENSION_CHANGED: self._on_tension_changed,
                EventType.STATE_SYNC: self._on_state_sync,
                EventType.HEALTH_CHECK: self._on_health_check
            }
            
            handler = handlers.get(event.event_type)
            if handler:
                return await handler(event)
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'status': 'no_action_taken'}
            )
            
        except Exception as e:
            logger.error(f"Stakeholder system error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status"""
        
        # Check for stressed stakeholders
        stressed = [s for s in self._active_stakeholders.values() if s.stress_level > 0.8]
        
        return {
            'healthy': len(stressed) < len(self._active_stakeholders) / 2,
            'active_stakeholders': len(self._active_stakeholders),
            'stressed_stakeholders': len(stressed),
            'pending_actions': len(self._pending_actions),
            'status': 'operational'
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get stakeholder data for a specific conflict"""
        
        # Get stakeholders for this conflict
        conflict_stakeholders = []
        for stakeholder in self._active_stakeholders.values():
            # Check if stakeholder is in this conflict
            # (Would check database for conflict association)
            conflict_stakeholders.append({
                'id': stakeholder.stakeholder_id,
                'name': stakeholder.name,
                'role': stakeholder.current_role.value,
                'stress': stakeholder.stress_level,
                'commitment': stakeholder.commitment_level
            })
        
        return {
            'stakeholders': conflict_stakeholders,
            'total_stakeholders': len(conflict_stakeholders)
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current subsystem state"""
        
        return {
            'active_stakeholders': len(self._active_stakeholders),
            'stressed_count': len([s for s in self._active_stakeholders.values() if s.stress_level > 0.7]),
            'pending_actions': len(self._pending_actions),
            'active_strategies': len(self._strategy_cache)
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if stakeholder system should process this scene"""
        
        # Relevant if NPCs are present
        npcs = scene_context.get('npcs', [])
        
        # Check if any NPCs are stakeholders
        for npc_id in npcs:
            if any(s.npc_id == npc_id for s in self._active_stakeholders.values()):
                return True
        
        return False
    
    # ========== Event Handlers ==========
    
    async def _on_conflict_created(self, event: SystemEvent) -> SubsystemResponse:
        """Handle new conflict creation - create stakeholders"""
        
        conflict_id = event.payload.get('conflict_id')
        conflict_type = event.payload.get('conflict_type')
        context = event.payload.get('context', {})
        
        # Get NPCs involved
        npcs = context.get('npcs', [])
        
        # Create stakeholders for NPCs
        created_stakeholders = []
        side_effects = []
        
        for npc_id in npcs[:5]:  # Limit stakeholders
            stakeholder = await self.create_stakeholder(
                npc_id,
                conflict_id,
                self._determine_initial_role(conflict_type)
            )
            
            if stakeholder:
                self._active_stakeholders[stakeholder.stakeholder_id] = stakeholder
                created_stakeholders.append(stakeholder)
                
                # Notify about stakeholder creation
                side_effects.append(SystemEvent(
                    event_id=f"stakeholder_created_{stakeholder.stakeholder_id}",
                    event_type=EventType.STAKEHOLDER_ACTION,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'stakeholder_id': stakeholder.stakeholder_id,
                        'action_type': 'joined_conflict',
                        'role': stakeholder.current_role.value
                    }
                ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'stakeholders_created': len(created_stakeholders),
                'stakeholder_ids': [s.stakeholder_id for s in created_stakeholders]
            },
            side_effects=side_effects
        )
    
    async def _on_conflict_updated(self, event: SystemEvent) -> SubsystemResponse:
        """Handle conflict updates - stakeholders may take actions"""
        
        conflict_id = event.payload.get('conflict_id')
        update_type = event.payload.get('update_type')
        
        # Determine which stakeholders should act
        acting_stakeholders = self._select_acting_stakeholders(update_type)
        
        # Generate actions
        actions = []
        side_effects = []
        
        for stakeholder in acting_stakeholders:
            action = await self.make_autonomous_decision(
                stakeholder,
                event.payload
            )
            
            if action:
                actions.append(action)
                self._pending_actions.append(action)
                
                # Notify about stakeholder action
                side_effects.append(SystemEvent(
                    event_id=f"action_{action.action_id}",
                    event_type=EventType.STAKEHOLDER_ACTION,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'stakeholder_id': stakeholder.stakeholder_id,
                        'action_type': action.action_type.value,
                        'target_id': action.target,
                        'intensity': action.success_probability
                    }
                ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'actions_taken': len(actions),
                'action_types': [a.action_type.value for a in actions]
            },
            side_effects=side_effects
        )
    
    async def _on_phase_transition(self, event: SystemEvent) -> SubsystemResponse:
        """Handle phase transitions - stakeholders adapt"""
        
        from_phase = event.payload.get('from_phase')
        to_phase = event.payload.get('to_phase')
        
        # Adapt stakeholder roles based on phase
        adaptations = []
        
        for stakeholder in self._active_stakeholders.values():
            if self._should_adapt_role(stakeholder, to_phase):
                adaptation = await self.adapt_stakeholder_role(
                    stakeholder,
                    {'phase': to_phase}
                )
                if adaptation.get('role_changed'):
                    adaptations.append(adaptation)
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'adaptations': len(adaptations),
                'phase_response': 'stakeholders_adapted'
            }
        )
    
    async def _on_player_choice(self, event: SystemEvent) -> SubsystemResponse:
        """Handle player choices - stakeholders react"""
        
        choice_type = event.payload.get('choice_type')
        target_npc = event.payload.get('target_npc')
        
        # Generate reactions from relevant stakeholders
        reactions = []
        side_effects = []
        
        for stakeholder in self._active_stakeholders.values():
            if self._should_react_to_choice(stakeholder, choice_type, target_npc):
                # Create mock action for reaction
                triggering_action = StakeholderAction(
                    action_id=0,
                    stakeholder_id=0,  # Player
                    action_type=ActionType.STRATEGIC,
                    description=f"Player choice: {choice_type}",
                    target=target_npc,
                    resources_used={},
                    success_probability=1.0,
                    consequences={},
                    timestamp=datetime.now()
                )
                
                reaction = await self.generate_reaction(
                    stakeholder,
                    triggering_action,
                    event.payload
                )
                
                reactions.append(reaction)
                
                # Update stress based on reaction
                if reaction.emotional_response in ['angry', 'fearful', 'frustrated']:
                    stakeholder.stress_level = min(1.0, stakeholder.stress_level + 0.1)
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'reactions_generated': len(reactions),
                'emotional_responses': [r.emotional_response for r in reactions]
            },
            side_effects=side_effects
        )
    
    async def _on_tension_changed(self, event: SystemEvent) -> SubsystemResponse:
        """Handle tension changes - affects stakeholder stress"""
        
        tension_type = event.payload.get('tension_type')
        level = event.payload.get('level', 0)
        
        # High tension increases stakeholder stress
        if level > 0.7:
            for stakeholder in self._active_stakeholders.values():
                stakeholder.stress_level = min(1.0, stakeholder.stress_level + 0.05)
                
                # Check for breaking points
                if stakeholder.stress_level >= 0.9:
                    await self._handle_breaking_point(
                        stakeholder,
                        "Overwhelmed by tension"
                    )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'stress_updated': True}
        )
    
    async def _on_state_sync(self, event: SystemEvent) -> SubsystemResponse:
        """Sync state with scene processing"""
        
        scene_context = event.payload
        npcs_present = scene_context.get('npcs', [])
        
        # Determine NPC behaviors based on stakeholder state
        npc_behaviors = {}
        
        for npc_id in npcs_present:
            stakeholder = self._find_stakeholder_by_npc(npc_id)
            if stakeholder:
                behavior = self._determine_scene_behavior(stakeholder)
                npc_behaviors[npc_id] = behavior
        
        # Check for autonomous actions
        autonomous_actions = []
        for stakeholder in self._active_stakeholders.values():
            if stakeholder.npc_id in npcs_present:
                if self._should_take_autonomous_action(stakeholder, scene_context):
                    action = await self.make_autonomous_decision(
                        stakeholder,
                        scene_context
                    )
                    if action:
                        autonomous_actions.append(action)
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'npc_behaviors': npc_behaviors,
                'autonomous_actions': [
                    {
                        'stakeholder': a.stakeholder_id,
                        'action': a.description,
                        'type': a.action_type.value
                    }
                    for a in autonomous_actions
                ]
            }
        )
    
    async def _on_health_check(self, event: SystemEvent) -> SubsystemResponse:
        """Respond to health check"""
        
        health = await self.health_check()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=health
        )
    
    # ========== Core Stakeholder Management (Modified for Orchestrator) ==========
    
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
        1. Stakeholder role
        2. Decision style
        3. 3-4 specific goals
        4. Initial stress level (0-1)
        5. Commitment level (0-1)
        
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
            
            stakeholder = Stakeholder(
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
            
            # Notify orchestrator of new stakeholder
            if self._synthesizer and self._synthesizer():
                await self._synthesizer().emit_event(SystemEvent(
                    event_id=f"stakeholder_{stakeholder_id}",
                    event_type=EventType.STAKEHOLDER_ACTION,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'action_type': 'stakeholder_created',
                        'stakeholder_id': stakeholder_id,
                        'role': stakeholder.current_role.value
                    }
                ))
            
            return stakeholder
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to create stakeholder: {e}")
            return self._create_fallback_stakeholder(npc_id, npc_details)
    
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
        Stress Level: {stakeholder.stress_level}
        
        Conflict State: {json.dumps(conflict_state, indent=2)}
        
        Decide:
        1. Action type
        2. Specific action description
        3. Target (if applicable)
        4. Success probability (0-1)
        
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
        
        Triggering Action: {triggering_action.description}
        Action Type: {triggering_action.action_type.value}
        
        Generate:
        1. Reaction type (counter/support/ignore/escalate/de-escalate)
        2. Specific reaction description
        3. Emotional response
        
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
        Stress: {stakeholder.stress_level}
        
        Changing Conditions: {json.dumps(changing_conditions, indent=2)}
        
        Determine:
        1. Should change role? (yes/no)
        2. If yes, new role
        3. Reason for change
        
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
                    'reason': result.get('reason', 'Circumstances changed')
                }
            
            return {'role_changed': False}
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to adapt role: {e}")
            return {'role_changed': False}
    
    # ========== Helper Methods ==========
    
    async def _load_active_stakeholders(self):
        """Load active stakeholders from database"""
        
        async with get_db_connection_context() as conn:
            stakeholders = await conn.fetch("""
                SELECT s.*, n.name, n.personality_traits
                FROM stakeholders s
                JOIN NPCs n ON s.npc_id = n.npc_id
                WHERE s.user_id = $1 AND s.conversation_id = $2
                AND EXISTS (
                    SELECT 1 FROM Conflicts c
                    WHERE c.conflict_id = s.conflict_id
                    AND c.is_active = true
                )
            """, self.user_id, self.conversation_id)
        
        for s in stakeholders:
            stakeholder = Stakeholder(
                stakeholder_id=s['stakeholder_id'],
                npc_id=s['npc_id'],
                name=s['name'],
                personality_traits=json.loads(s.get('personality_traits', '[]')),
                current_role=StakeholderRole[s['role'].upper()],
                decision_style=DecisionStyle[s['decision_style'].upper()],
                goals=[],
                resources={},
                relationships={},
                stress_level=s['stress_level'],
                commitment_level=s['commitment_level']
            )
            self._active_stakeholders[stakeholder.stakeholder_id] = stakeholder
    
    def _determine_initial_role(self, conflict_type: str) -> str:
        """Determine initial role based on conflict type"""
        
        if 'power' in conflict_type.lower():
            return random.choice(['instigator', 'defender', 'opportunist'])
        elif 'social' in conflict_type.lower():
            return random.choice(['mediator', 'bystander', 'escalator'])
        else:
            return 'bystander'
    
    def _select_acting_stakeholders(self, update_type: str) -> List[Stakeholder]:
        """Select which stakeholders should act"""
        
        acting = []
        
        for stakeholder in self._active_stakeholders.values():
            # Active roles always act
            if stakeholder.current_role in [
                StakeholderRole.INSTIGATOR,
                StakeholderRole.ESCALATOR,
                StakeholderRole.MEDIATOR
            ]:
                acting.append(stakeholder)
            # Others act probabilistically
            elif random.random() < stakeholder.commitment_level:
                acting.append(stakeholder)
        
        return acting[:3]  # Limit to prevent overwhelming
    
    def _should_adapt_role(self, stakeholder: Stakeholder, phase: str) -> bool:
        """Check if stakeholder should adapt role"""
        
        # Adapt if stressed
        if stakeholder.stress_level > 0.8:
            return True
        
        # Adapt based on phase
        if phase == 'climax' and stakeholder.current_role == StakeholderRole.BYSTANDER:
            return True
        
        if phase == 'resolution' and stakeholder.current_role == StakeholderRole.ESCALATOR:
            return True
        
        return False
    
    def _should_react_to_choice(
        self,
        stakeholder: Stakeholder,
        choice_type: str,
        target_npc: Optional[int]
    ) -> bool:
        """Check if stakeholder should react to player choice"""
        
        # Always react if targeted
        if target_npc == stakeholder.npc_id:
            return True
        
        # React based on role and commitment
        if stakeholder.current_role in [
            StakeholderRole.MEDIATOR,
            StakeholderRole.INSTIGATOR
        ]:
            return True
        
        return random.random() < stakeholder.commitment_level
    
    def _find_stakeholder_by_npc(self, npc_id: int) -> Optional[Stakeholder]:
        """Find stakeholder by NPC ID"""
        
        for stakeholder in self._active_stakeholders.values():
            if stakeholder.npc_id == npc_id:
                return stakeholder
        return None
    
    def _determine_scene_behavior(self, stakeholder: Stakeholder) -> str:
        """Determine NPC behavior in scene based on stakeholder state"""
        
        if stakeholder.stress_level > 0.8:
            return "agitated"
        elif stakeholder.current_role == StakeholderRole.MEDIATOR:
            return "conciliatory"
        elif stakeholder.current_role == StakeholderRole.INSTIGATOR:
            return "provocative"
        elif stakeholder.current_role == StakeholderRole.BYSTANDER:
            return "observant"
        else:
            return "engaged"
    
    def _should_take_autonomous_action(
        self,
        stakeholder: Stakeholder,
        scene_context: Dict[str, Any]
    ) -> bool:
        """Check if stakeholder should take autonomous action"""
        
        # High stress increases action likelihood
        if stakeholder.stress_level > 0.7:
            return random.random() < 0.5
        
        # Active roles more likely to act
        if stakeholder.current_role in [
            StakeholderRole.INSTIGATOR,
            StakeholderRole.ESCALATOR
        ]:
            return random.random() < 0.4
        
        return random.random() < 0.2
    
    async def _handle_breaking_point(
        self,
        stakeholder: Stakeholder,
        breaking_action: str
    ) -> Dict[str, Any]:
        """Handle a stakeholder reaching their breaking point"""
        
        # Update stakeholder state
        stakeholder.commitment_level = 0.0
        stakeholder.current_role = StakeholderRole.BYSTANDER
        
        # Notify orchestrator
        if self._synthesizer and self._synthesizer():
            await self._synthesizer().emit_event(SystemEvent(
                event_id=f"breaking_{stakeholder.stakeholder_id}",
                event_type=EventType.EDGE_CASE_DETECTED,
                source_subsystem=self.subsystem_type,
                payload={
                    'edge_case': 'stakeholder_breaking_point',
                    'stakeholder_id': stakeholder.stakeholder_id,
                    'action': breaking_action
                },
                priority=2
            ))
        
        return {
            'action': breaking_action,
            'stakeholder_withdraws': True,
            'stress_relief': 0.5
        }
    
    async def _get_npc_details(self, npc_id: int) -> Dict:
        """Get NPC details from database"""
        
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT * FROM NPCs WHERE npc_id = $1
            """, npc_id)
        
        return dict(npc) if npc else {}
    
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
    
    # ========== Process Event (Required by parent class) ==========
    
    async def process_event(self, conflict_id: int, event: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - now routes through handle_event"""
        
        # Convert to SystemEvent
        system_event = SystemEvent(
            event_id=f"legacy_{conflict_id}",
            event_type=EventType.CONFLICT_UPDATED,
            source_subsystem=SubsystemType.STAKEHOLDER,
            payload={
                'conflict_id': conflict_id,
                **event
            }
        )
        
        response = await self.handle_event(system_event)
        return response.data
    
    # ========== LLM Agent Properties (Preserved) ==========
    
    @property
    def decision_maker(self) -> Agent:
        if self._decision_maker is None:
            self._decision_maker = Agent(
                name="Stakeholder Decision Maker",
                instructions="""
                Make decisions for NPC stakeholders in conflicts.
                Consider personality, stress, resources, and goals.
                Generate decisions that are true to character.
                """,
                model="gpt-5-nano",
            )
        return self._decision_maker
    
    @property
    def reaction_generator(self) -> Agent:
        if self._reaction_generator is None:
            self._reaction_generator = Agent(
                name="Stakeholder Reaction Generator",
                instructions="""
                Generate realistic reactions to events and actions.
                Consider personality, relationships, and emotional state.
                """,
                model="gpt-5-nano",
            )
        return self._reaction_generator
    
    @property
    def strategy_planner(self) -> Agent:
        if self._strategy_planner is None:
            self._strategy_planner = Agent(
                name="Stakeholder Strategy Planner",
                instructions="""
                Develop long-term strategies for stakeholders.
                Match strategies to character capabilities and style.
                """,
                model="gpt-5-nano",
            )
        return self._strategy_planner
    
    @property
    def personality_analyzer(self) -> Agent:
        if self._personality_analyzer is None:
            self._personality_analyzer = Agent(
                name="Personality Analyzer",
                instructions="""
                Analyze how personality traits affect decisions.
                Create psychologically consistent characters.
                """,
                model="gpt-5-nano",
            )
        return self._personality_analyzer


# ===============================================================================
# PUBLIC API - Now Routes Through Orchestrator
# ===============================================================================

@function_tool
async def create_conflict_stakeholder(
    ctx: RunContextWrapper,
    npc_id: int,
    conflict_id: int,
    suggested_role: Optional[str] = None
) -> Dict[str, Any]:
    """Create a stakeholder for a conflict - routes through orchestrator"""
    
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Emit stakeholder creation event
    event = SystemEvent(
        event_id=f"create_stakeholder_{npc_id}",
        event_type=EventType.STAKEHOLDER_ACTION,
        source_subsystem=SubsystemType.STAKEHOLDER,
        payload={
            'action_type': 'create_stakeholder',
            'npc_id': npc_id,
            'conflict_id': conflict_id,
            'suggested_role': suggested_role
        },
        requires_response=True
    )
    
    responses = await synthesizer.emit_event(event)
    
    if responses:
        return responses[0].data
    return {'error': 'Failed to create stakeholder'}

@function_tool
async def stakeholder_take_action(
    ctx: RunContextWrapper,
    stakeholder_id: int,
    conflict_state: Dict[str, Any],
    options: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Have a stakeholder take an autonomous action - routes through orchestrator"""
    
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Emit action event
    event = SystemEvent(
        event_id=f"stakeholder_action_{stakeholder_id}",
        event_type=EventType.STAKEHOLDER_ACTION,
        source_subsystem=SubsystemType.STAKEHOLDER,
        payload={
            'stakeholder_id': stakeholder_id,
            'conflict_state': conflict_state,
            'options': options
        },
        requires_response=True
    )
    
    responses = await synthesizer.emit_event(event)
    
    if responses:
        return responses[0].data
    return {'error': 'Failed to execute action'}
