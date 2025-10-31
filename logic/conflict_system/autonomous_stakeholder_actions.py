# logic/conflict_system/autonomous_stakeholder_actions.py
"""
Autonomous Stakeholder Actions System with LLM-generated decisions.
Refactored to use OpenAI Responses API (gpt-5-nano) and to integrate as a
ConflictSubsystem under the Conflict Synthesizer orchestrator.

- Uses AsyncOpenAI client; no temperature/max_tokens specified.
- Subscribes to STAKEHOLDER_ACTION and provides a concrete handler.
- Uniform JSON extraction for robustness (handles fenced or messy output).
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import weakref

from db.connection import get_db_connection_context
from agents import function_tool, RunContextWrapper  # kept for public API tools

# Orchestrator interfaces
from logic.conflict_system.conflict_synthesizer import (
    ConflictSubsystem,
    SubsystemType,
    EventType,
    SystemEvent,
    SubsystemResponse,
)

logger = logging.getLogger(__name__)

# ===============================================================================
# OpenAI Responses API Helper
# ===============================================================================

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
    pending_task_id: Optional[str] = None

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
    task_id: Optional[str] = None

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
    task_id: Optional[str] = None

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
# STAKEHOLDER AUTONOMY SUBSYSTEM
# ===============================================================================

class StakeholderAutonomySystem(ConflictSubsystem):
    """
    Manages autonomous NPC actions as a subsystem under the orchestrator.
    Uses OpenAI Responses API (gpt-5-nano) for decision/reaction/role logic.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._synthesizer = None  # weakref set by orchestrator
        
        # Stakeholder state
        self._active_stakeholders: Dict[int, Stakeholder] = {}
        self._pending_actions: List[StakeholderAction] = []
        self._strategy_cache: Dict[int, StakeholderStrategy] = {}
        self._pending_creations: Dict[str, Dict[str, Any]] = {}  # operation_id -> {npcs, conflict_type}

    # ========== ConflictSubsystem Interface ==========
    
    @property
    def subsystem_type(self) -> SubsystemType:
        return SubsystemType.STAKEHOLDER
    
    @property
    def capabilities(self) -> Set[str]:
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
        return {
            SubsystemType.TENSION,
            SubsystemType.SOCIAL,
            SubsystemType.FLOW,
        }
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.PHASE_TRANSITION,
            EventType.PLAYER_CHOICE,
            EventType.TENSION_CHANGED,
            EventType.STATE_SYNC,
            EventType.HEALTH_CHECK,
            EventType.STAKEHOLDER_ACTION,  # important for function tools
        }
    
    async def initialize(self, synthesizer) -> bool:
        self._synthesizer = weakref.ref(synthesizer)
        await self._load_active_stakeholders()
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Route events to appropriate handlers with safe fallbacks."""
        try:
            handlers = {
                EventType.CONFLICT_CREATED: self._on_conflict_created,
                EventType.CONFLICT_UPDATED: self._on_conflict_updated,
                EventType.PHASE_TRANSITION: self._on_phase_transition,
                EventType.PLAYER_CHOICE: self._on_player_choice,
                EventType.TENSION_CHANGED: self._on_tension_changed,
                EventType.STATE_SYNC: self._on_state_sync,
                EventType.HEALTH_CHECK: self._on_health_check,
                EventType.STAKEHOLDER_ACTION: self._on_stakeholder_action,
            }
            handler = handlers.get(event.event_type)
            if handler:
                return await handler(event)
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'status': 'no_action_taken'},
                side_effects=[],
            )
        except Exception as e:
            logger.error(f"Stakeholder system error on {event.event_id}: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[],
            )

    # ========== Health & State ==========
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            stressed = [s for s in self._active_stakeholders.values() if s.stress_level > 0.8]
            return {
                'healthy': len(stressed) < max(1, len(self._active_stakeholders)) / 2,
                'active_stakeholders': len(self._active_stakeholders),
                'stressed_stakeholders': len(stressed),
                'pending_actions': len(self._pending_actions),
                'status': 'operational'
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    async def get_state(self) -> Dict[str, Any]:
        return {
            'active_stakeholders': len(self._active_stakeholders),
            'stressed_count': len([s for s in self._active_stakeholders.values() if s.stress_level > 0.7]),
            'pending_actions': len(self._pending_actions),
            'active_strategies': len(self._strategy_cache)
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Return a light snapshot for a conflict (DB association can be extended)."""
        conflict_stakeholders = []
        for s in self._active_stakeholders.values():
            # TODO: filter by conflict_id when mapping is available
            conflict_stakeholders.append({
                'id': s.stakeholder_id,
                'name': s.name,
                'role': s.current_role.value,
                'stress': s.stress_level,
                'commitment': s.commitment_level
            })
        return {'stakeholders': conflict_stakeholders, 'total_stakeholders': len(conflict_stakeholders)}
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        npcs = scene_context.get('npcs', []) or []
        for npc_id in npcs:
            if any(s.npc_id == npc_id for s in self._active_stakeholders.values()):
                return True
        return False

    # ========== Event Handlers ==========
    
    async def _on_conflict_created(self, event: SystemEvent) -> SubsystemResponse:
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')
        conflict_type = payload.get('conflict_type')
        context = payload.get('context', {}) or {}
        npcs = context.get('npcs') or context.get('present_npcs') or context.get('participants') or []

        # If we don't have a conflict_id yet (common on create), defer creation
        if not conflict_id:
            self._pending_creations[event.event_id] = {'npcs': npcs, 'conflict_type': conflict_type}
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'stakeholders_created': 0, 'deferred': True, 'operation_id': event.event_id},
                side_effects=[]
            )
        
        created = await self._create_stakeholders_for_npcs(npcs, conflict_id, conflict_type)
        side_effects = [
            SystemEvent(
                event_id=f"stakeholder_created_{s.stakeholder_id}",
                event_type=EventType.STAKEHOLDER_ACTION,
                source_subsystem=self.subsystem_type,
                payload={'stakeholder_id': s.stakeholder_id, 'action_type': 'joined_conflict', 'role': s.current_role.value},
            )
            for s in created
        ]
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'stakeholders_created': len(created), 'stakeholder_ids': [s.stakeholder_id for s in created]},
            side_effects=side_effects
        )
    
    async def _on_conflict_updated(self, event: SystemEvent) -> SubsystemResponse:
        payload = event.payload or {}
        update_type = payload.get('update_type')

        acting_stakeholders = self._select_acting_stakeholders(update_type or "")

        # HOT PATH: Dispatch background action generation instead of blocking
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            should_dispatch_action_generation,
            dispatch_action_generation,
        )

        dispatched = 0
        for s in acting_stakeholders:
            if should_dispatch_action_generation(s, payload):
                dispatch_action_generation(s, payload)
                dispatched += 1

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'actions_dispatched': dispatched,
                'message': f'Dispatched {dispatched} background action generations'
            },
            side_effects=[]
        )
    
    async def _on_phase_transition(self, event: SystemEvent) -> SubsystemResponse:
        payload = event.payload or {}
        to_phase = payload.get('to_phase')
        
        adaptations = []
        for s in self._active_stakeholders.values():
            if self._should_adapt_role(s, to_phase or ""):
                adaptation = await self.adapt_stakeholder_role(s, {'phase': to_phase})
                if adaptation.get('role_changed'):
                    adaptations.append(adaptation)
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'adaptations': len(adaptations), 'phase_response': 'stakeholders_adapted'},
            side_effects=[]
        )
    
    async def _on_player_choice(self, event: SystemEvent) -> SubsystemResponse:
        payload = event.payload or {}
        choice_type = payload.get('choice_type')
        target_npc = payload.get('target_npc')

        # HOT PATH: Dispatch background reaction generation instead of blocking
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import dispatch_reaction_generation

        dispatched = 0
        for s in self._active_stakeholders.values():
            if self._should_react_to_choice(s, choice_type, target_npc):
                dispatch_reaction_generation(s, payload, event.event_id)
                dispatched += 1
                # Update stress lightly (still fast, rule-based)
                if choice_type in ['aggressive', 'confrontational', 'escalating']:
                    s.stress_level = min(1.0, s.stress_level + 0.1)

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'reactions_dispatched': dispatched,
                'message': f'Dispatched {dispatched} background reaction generations'
            },
            side_effects=[]
        )
    
    async def _on_tension_changed(self, event: SystemEvent) -> SubsystemResponse:
        payload = event.payload or {}
        level = float(payload.get('level', 0.0) or 0.0)
        
        if level > 0.7:
            for s in self._active_stakeholders.values():
                s.stress_level = min(1.0, s.stress_level + 0.05)
                if s.stress_level >= 0.9:
                    await self._handle_breaking_point(s, "Overwhelmed by tension")
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'stress_updated': True},
            side_effects=[]
        )
    
    async def _on_state_sync(self, event: SystemEvent) -> SubsystemResponse:
        raw = event.payload or {}
        scene_context = raw.get('scene_context', raw) or {}

        # Finalize deferred stakeholder creation if conflict_id now known
        created_after_defer = 0
        op_id = raw.get('operation_id')
        cid = raw.get('conflict_id')
        if op_id and cid and op_id in self._pending_creations:
            pending = self._pending_creations.pop(op_id)
            created = await self._create_stakeholders_for_npcs(
                npcs=pending.get('npcs') or [],
                conflict_id=cid,
                conflict_type=pending.get('conflict_type'),
            )
            created_after_defer = len(created)

        # HOT PATH: Use fast helper functions
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            fetch_ready_actions_for_scene,
            determine_scene_behavior,
            dispatch_action_generation,
        )

        npcs_present = scene_context.get('npcs', []) or []

        # Fast rule-based behavior hints (no LLM)
        npc_behaviors: Dict[int, str] = {}
        for npc_id in npcs_present:
            s = self._find_stakeholder_by_npc(npc_id)
            if s:
                npc_behaviors[npc_id] = determine_scene_behavior(s, scene_context)

        # Fetch ready actions from DB (precomputed by workers)
        ready_actions = await fetch_ready_actions_for_scene(scene_context, limit=10)

        # Dispatch background generation for stakeholders that need fresh actions
        dispatched = 0
        for s in self._active_stakeholders.values():
            if s.npc_id in npcs_present and self._should_take_autonomous_action(s, scene_context):
                # Check if we have a recent action; if not, dispatch generation
                has_recent = any(
                    a.get("stakeholder_id") == s.stakeholder_id for a in ready_actions
                )
                if not has_recent:
                    dispatch_action_generation(s, scene_context)
                    dispatched += 1

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'npc_behaviors': npc_behaviors,
                'ready_actions_count': len(ready_actions),
                'actions_dispatched': dispatched,
                'stakeholders_created_after_defer': created_after_defer,
            },
            side_effects=[]
        )
    
    async def _on_health_check(self, event: SystemEvent) -> SubsystemResponse:
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=await self.health_check(),
            side_effects=[]
        )
    
    async def _on_stakeholder_action(self, event: SystemEvent) -> SubsystemResponse:
        payload = event.payload or {}
        action_type = payload.get('action_type')

        # Create a stakeholder on demand
        if action_type == 'create_stakeholder':
            npc_id = payload.get('npc_id')
            conflict_id = payload.get('conflict_id')
            suggested_role = payload.get('suggested_role')
            if not (npc_id and conflict_id):
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=False,
                    data={'error': 'npc_id and conflict_id required'},
                    side_effects=[]
                )
            stakeholder = await self.create_stakeholder(npc_id, conflict_id, suggested_role)
            ok = stakeholder is not None
            if ok:
                self._active_stakeholders[stakeholder.stakeholder_id] = stakeholder
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=ok,
                data={'stakeholder_created': ok, 'stakeholder_id': getattr(stakeholder, 'stakeholder_id', 0)},
                side_effects=[]
            )

        # Execute an autonomous action for a stakeholder
        stakeholder_id = payload.get('stakeholder_id')
        if stakeholder_id is None:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'stakeholder_id required'},
                side_effects=[]
            )
        s = self._active_stakeholders.get(stakeholder_id)
        if not s:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'stakeholder not found'},
                side_effects=[]
            )

        # HOT PATH: Dispatch background action generation
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import dispatch_action_generation

        conflict_state = payload.get('conflict_state') or {}
        dispatch_action_generation(s, conflict_state)

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'message': 'Action generation dispatched to background',
                'stakeholder_id': stakeholder_id,
            },
            side_effects=[]
        )

    # ========== Core Stakeholder Management ==========
    
    async def create_stakeholder(
        self,
        npc_id: int,
        conflict_id: int,
        initial_role: Optional[str] = None
    ) -> Optional[Stakeholder]:
        """Create a stakeholder record quickly and defer LLM enrichment to background tasks."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            dispatch_stakeholder_profile_enrichment,
        )

        npc_details = await self._get_npc_details(npc_id)

        # Heuristic defaults for hot-path creation
        role_value = (initial_role or 'bystander').lower()
        if role_value not in StakeholderRole._value2member_map_:
            role_value = 'bystander'

        decision_defaults = {
            'instigator': 'reactive',
            'defender': 'rational',
            'mediator': 'principled',
            'opportunist': 'calculating',
            'victim': 'emotional',
            'peacemaker': 'principled',
            'escalator': 'reactive',
        }
        decision_style = decision_defaults.get(role_value, 'reactive')
        stress_default = 0.45 if role_value in {'instigator', 'escalator'} else 0.3
        commitment_default = 0.6 if role_value not in {'bystander', 'victim'} else 0.4

        try:
            async with get_db_connection_context() as conn:
                stakeholder_id = await conn.fetchval(
                    """
                    INSERT INTO stakeholders
                    (user_id, conversation_id, npc_id, conflict_id,
                     role, decision_style, stress_level, commitment_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING stakeholder_id
                    """,
                    self.user_id,
                    self.conversation_id,
                    npc_id,
                    conflict_id,
                    role_value,
                    decision_style,
                    stress_default,
                    commitment_default,
                )

            traits_raw = npc_details.get('personality_traits', [])
            traits = json.loads(traits_raw) if isinstance(traits_raw, str) else (traits_raw or [])

            stakeholder = Stakeholder(
                stakeholder_id=stakeholder_id,
                npc_id=npc_id,
                name=npc_details.get('name', 'Unknown'),
                personality_traits=traits,
                current_role=StakeholderRole[role_value.upper()],
                decision_style=DecisionStyle[decision_style.upper()],
                goals=[],
                resources={},
                relationships={},
                stress_level=stress_default,
                commitment_level=commitment_default,
            )

            enrichment_task = dispatch_stakeholder_profile_enrichment({
                'stakeholder_id': stakeholder_id,
                'npc_id': npc_id,
                'conflict_id': conflict_id,
                'initial_role': role_value,
            })
            stakeholder.pending_task_id = enrichment_task

            if self._synthesizer and self._synthesizer():
                await self._synthesizer().emit_event(SystemEvent(
                    event_id=f"stakeholder_{stakeholder_id}",
                    event_type=EventType.STAKEHOLDER_ACTION,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'action_type': 'stakeholder_created',
                        'stakeholder_id': stakeholder_id,
                        'role': stakeholder.current_role.value,
                        'task_id': enrichment_task,
                    }
                ))

            return stakeholder
        except Exception as e:
            logger.warning(f"Failed to create stakeholder for npc {npc_id}: {e}")
            return None
    
    async def make_autonomous_decision(
        self,
        stakeholder: Stakeholder,
        conflict_state: Dict[str, Any],
        available_options: Optional[List[str]] = None
    ) -> Optional[StakeholderAction]:
        """Return cached decision payload or dispatch background generation."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            compute_decision_context_hash,
            dispatch_action_generation,
            fetch_planned_item,
            mark_action_consumed,
        )

        context_hash = compute_decision_context_hash(
            stakeholder.stakeholder_id,
            conflict_state or {},
            available_options,
        )

        planned = await fetch_planned_item(
            stakeholder_id=stakeholder.stakeholder_id,
            kind='action',
            context_hash=context_hash,
        )

        if planned:
            planned_id, payload = planned
            await mark_action_consumed(planned_id)
            action = StakeholderAction(
                action_id=payload.get('planned_action_id', planned_id),
                stakeholder_id=stakeholder.stakeholder_id,
                action_type=ActionType[(payload.get('action_type', 'observant') or 'observant').upper()],
                description=payload.get('description', 'Takes action'),
                target=payload.get('target'),
                resources_used=payload.get('resources', {}),
                success_probability=float(payload.get('success_probability', 0.5) or 0.5),
                consequences=payload.get('consequences', {}),
                timestamp=datetime.now(),
            )
            return action

        scene_context = {}
        if isinstance(conflict_state, dict):
            scene_context = conflict_state.get('scene_context') or {
                'scene_id': conflict_state.get('scene_id'),
                'conflict_id': conflict_state.get('conflict_id'),
                'stakeholder_ids': [stakeholder.stakeholder_id],
                'phase': conflict_state.get('phase'),
            }

        task_id = dispatch_action_generation(
            stakeholder,
            scene_context or {'stakeholder_ids': [stakeholder.stakeholder_id]},
            conflict_state=conflict_state,
            available_options=available_options,
            context_hash=context_hash,
        )

        # Heuristic fallback action
        fallback_type = ActionType.OBSERVANT
        if stakeholder.current_role in {StakeholderRole.INSTIGATOR, StakeholderRole.ESCALATOR}:
            fallback_type = ActionType.AGGRESSIVE
        elif stakeholder.current_role == StakeholderRole.DEFENDER:
            fallback_type = ActionType.DEFENSIVE
        elif stakeholder.current_role == StakeholderRole.MEDIATOR:
            fallback_type = ActionType.DIPLOMATIC

        return StakeholderAction(
            action_id=0,
            stakeholder_id=stakeholder.stakeholder_id,
            action_type=fallback_type,
            description="Acts using heuristic fallback",
            target=None,
            resources_used={},
            success_probability=0.55,
            consequences={},
            timestamp=datetime.now(),
            task_id=task_id,
        )

    async def generate_reaction(
        self,
        stakeholder: Stakeholder,
        triggering_action: StakeholderAction,
        action_context: Dict[str, Any]
    ) -> StakeholderReaction:
        """Return cached reaction or dispatch background generation."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            dispatch_reaction_generation,
            fetch_planned_item,
            mark_action_consumed,
        )

        event_id = action_context.get('event_id') or f"action_{triggering_action.action_id}"

        planned = await fetch_planned_item(
            stakeholder_id=stakeholder.stakeholder_id,
            kind='reaction',
            context_hash=event_id,
        )

        if planned:
            planned_id, payload = planned
            await mark_action_consumed(planned_id)
            stakeholder.stress_level = max(
                0.0,
                min(1.0, stakeholder.stress_level + float(payload.get('stress_impact', 0.0) or 0.0)),
            )
            return StakeholderReaction(
                reaction_id=payload.get('planned_action_id', planned_id),
                stakeholder_id=stakeholder.stakeholder_id,
                triggering_action_id=triggering_action.action_id,
                reaction_type=payload.get('reaction_type', 'observe'),
                description=payload.get('description', 'Reacts'),
                emotional_response=payload.get('emotional_response', 'neutral'),
                relationship_impact={
                    triggering_action.stakeholder_id: float(payload.get('relationship_impact', 0.0) or 0.0)
                },
            )

        task_id = dispatch_reaction_generation(
            stakeholder,
            action_context,
            event_id,
            triggering_action={
                'description': triggering_action.description,
                'action_type': triggering_action.action_type.value,
            },
        )

        # Heuristic fallback reaction
        fallback_description = "Keeps emotions in check"
        reaction_type = "observe"
        if triggering_action.action_type in {ActionType.AGGRESSIVE, ActionType.MANIPULATIVE}:
            reaction_type = "counter" if stakeholder.current_role != StakeholderRole.BYSTANDER else "ignore"
            fallback_description = "Responds cautiously"
        elif triggering_action.action_type == ActionType.DIPLOMATIC:
            reaction_type = "support"
            fallback_description = "Offers support"

        return StakeholderReaction(
            reaction_id=0,
            stakeholder_id=stakeholder.stakeholder_id,
            triggering_action_id=triggering_action.action_id,
            reaction_type=reaction_type,
            description=fallback_description,
            emotional_response="neutral",
            relationship_impact={triggering_action.stakeholder_id: 0.0},
            task_id=task_id,
        )
    
    async def adapt_stakeholder_role(
        self,
        stakeholder: Stakeholder,
        changing_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply heuristic role adaptation and dispatch background evaluation."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            compute_decision_context_hash,
            dispatch_role_adaptation,
            fetch_planned_item,
            mark_action_consumed,
        )

        context_hash = compute_decision_context_hash(
            stakeholder.stakeholder_id,
            changing_conditions or {},
        )

        planned = await fetch_planned_item(
            stakeholder_id=stakeholder.stakeholder_id,
            kind='role_adaptation',
            context_hash=context_hash,
        )
        if planned:
            planned_id, payload = planned
            await mark_action_consumed(planned_id)
            if payload.get('change_role'):
                old_role = stakeholder.current_role
                stakeholder.current_role = StakeholderRole[payload.get('new_role', old_role.value).upper()]
                async with get_db_connection_context() as conn:
                    await conn.execute(
                        "UPDATE stakeholders SET role = $1 WHERE stakeholder_id = $2",
                        stakeholder.current_role.value,
                        stakeholder.stakeholder_id,
                    )
                return {
                    'role_changed': True,
                    'old_role': old_role.value,
                    'new_role': stakeholder.current_role.value,
                    'reason': payload.get('reason', 'Background evaluation'),
                }
            return {'role_changed': False, 'reason': 'No change recommended'}

        # Heuristic fallback: adjust role based on stress and phase
        phase = changing_conditions.get('phase') if isinstance(changing_conditions, dict) else None
        old_role = stakeholder.current_role
        role_changed = False
        reason = 'Heuristic decision'

        if stakeholder.stress_level > 0.85 and stakeholder.current_role == StakeholderRole.BYSTANDER:
            stakeholder.current_role = StakeholderRole.VICTIM
            role_changed = True
        elif phase == 'resolution' and stakeholder.current_role == StakeholderRole.ESCALATOR:
            stakeholder.current_role = StakeholderRole.PEACEMAKER
            role_changed = True
        elif phase == 'climax' and stakeholder.current_role == StakeholderRole.BYSTANDER and stakeholder.commitment_level > 0.6:
            stakeholder.current_role = StakeholderRole.DEFENDER
            role_changed = True
        else:
            reason = 'No heuristic change'

        if role_changed:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    "UPDATE stakeholders SET role = $1 WHERE stakeholder_id = $2",
                    stakeholder.current_role.value,
                    stakeholder.stakeholder_id,
                )

        task_id = dispatch_role_adaptation({
            'stakeholder_id': stakeholder.stakeholder_id,
            'stakeholder_snapshot': {
                'name': stakeholder.name,
                'role': stakeholder.current_role.value,
                'stress_level': stakeholder.stress_level,
            },
            'changing_conditions': changing_conditions,
            'context_hash': context_hash,
        })
        stakeholder.pending_task_id = task_id or stakeholder.pending_task_id

        return {
            'role_changed': role_changed,
            'old_role': old_role.value if role_changed else old_role.value,
            'new_role': stakeholder.current_role.value,
            'reason': reason,
            'task_id': task_id,
        }

    # ========== Helper Methods ==========
    
    async def _load_active_stakeholders(self):
        """Load active stakeholders from database for this user/session."""
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                """
                SELECT s.*, n.name, n.personality_traits
                FROM stakeholders s
                JOIN NPCs n ON s.npc_id = n.npc_id
                WHERE s.user_id = $1 AND s.conversation_id = $2
                  AND EXISTS (
                    SELECT 1 FROM Conflicts c
                    WHERE c.conflict_id = s.conflict_id
                      AND c.is_active = true
                  )
                """,
                self.user_id, self.conversation_id
            )
        for r in rows:
            try:
                traits = r.get('personality_traits')
                traits_list = json.loads(traits) if isinstance(traits, str) else (traits or [])
                sh = Stakeholder(
                    stakeholder_id=r['stakeholder_id'],
                    npc_id=r['npc_id'],
                    name=r['name'],
                    personality_traits=traits_list,
                    current_role=StakeholderRole[r['role'].upper()],
                    decision_style=DecisionStyle[r['decision_style'].upper()],
                    goals=[],
                    resources={},
                    relationships={},
                    stress_level=float(r['stress_level']),
                    commitment_level=float(r['commitment_level'])
                )
                self._active_stakeholders[sh.stakeholder_id] = sh
            except Exception as e:
                logger.debug(f"Skipping stakeholder row due to error: {e}")
    
    def _determine_initial_role(self, conflict_type: str) -> str:
        ct = (conflict_type or '').lower()
        if 'power' in ct:
            return random.choice(['instigator', 'defender', 'opportunist'])
        if 'social' in ct:
            return random.choice(['mediator', 'bystander', 'escalator'])
        return 'bystander'
    
    async def _create_stakeholders_for_npcs(
        self,
        npcs: List[int],
        conflict_id: int,
        conflict_type: Optional[str]
    ) -> List[Stakeholder]:
        created: List[Stakeholder] = []
        for npc_id in (npcs or [])[:5]:  # safety cap
            sh = await self.create_stakeholder(
                npc_id=npc_id,
                conflict_id=conflict_id,
                initial_role=self._determine_initial_role(conflict_type or "")
            )
            if sh:
                self._active_stakeholders[sh.stakeholder_id] = sh
                created.append(sh)
        return created
    
    def _select_acting_stakeholders(self, update_type: str) -> List[Stakeholder]:
        acting: List[Stakeholder] = []
        for s in self._active_stakeholders.values():
            if s.current_role in [StakeholderRole.INSTIGATOR, StakeholderRole.ESCALATOR, StakeholderRole.MEDIATOR]:
                acting.append(s)
            elif random.random() < s.commitment_level:
                acting.append(s)
        return acting[:3]
    
    def _should_adapt_role(self, stakeholder: Stakeholder, phase: str) -> bool:
        if stakeholder.stress_level > 0.8:
            return True
        if phase == 'climax' and stakeholder.current_role == StakeholderRole.BYSTANDER:
            return True
        if phase == 'resolution' and stakeholder.current_role == StakeholderRole.ESCALATOR:
            return True
        return False
    
    def _should_react_to_choice(
        self,
        stakeholder: Stakeholder,
        choice_type: Optional[str],
        target_npc: Optional[int]
    ) -> bool:
        if target_npc == stakeholder.npc_id:
            return True
        if stakeholder.current_role in [StakeholderRole.MEDIATOR, StakeholderRole.INSTIGATOR]:
            return True
        return random.random() < stakeholder.commitment_level
    
    def _find_stakeholder_by_npc(self, npc_id: int) -> Optional[Stakeholder]:
        for s in self._active_stakeholders.values():
            if s.npc_id == npc_id:
                return s
        return None
    
    def _determine_scene_behavior(self, stakeholder: Stakeholder) -> str:
        if stakeholder.stress_level > 0.8:
            return "agitated"
        if stakeholder.current_role == StakeholderRole.MEDIATOR:
            return "conciliatory"
        if stakeholder.current_role == StakeholderRole.INSTIGATOR:
            return "provocative"
        if stakeholder.current_role == StakeholderRole.BYSTANDER:
            return "observant"
        return "engaged"
    
    def _should_take_autonomous_action(self, stakeholder: Stakeholder, scene_context: Dict[str, Any]) -> bool:
        if stakeholder.stress_level > 0.7:
            return random.random() < 0.5
        if stakeholder.current_role in [StakeholderRole.INSTIGATOR, StakeholderRole.ESCALATOR]:
            return random.random() < 0.4
        return random.random() < 0.2
    
    async def _handle_breaking_point(self, stakeholder: Stakeholder, breaking_action: str) -> Dict[str, Any]:
        stakeholder.commitment_level = 0.0
        stakeholder.current_role = StakeholderRole.BYSTANDER
        if self._synthesizer and self._synthesizer():
            await self._synthesizer().emit_event(SystemEvent(
                event_id=f"breaking_{stakeholder.stakeholder_id}",
                event_type=EventType.EDGE_CASE_DETECTED,
                source_subsystem=self.subsystem_type,
                payload={'edge_case': 'stakeholder_breaking_point', 'stakeholder_id': stakeholder.stakeholder_id, 'action': breaking_action},
                priority=2
            ))
        return {'action': breaking_action, 'stakeholder_withdraws': True, 'stress_relief': 0.5}
    
    async def _get_npc_details(self, npc_id: int) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("SELECT * FROM NPCs WHERE npc_id = $1", npc_id)
        return dict(row) if row else {}

    # ========== Legacy compatibility ==========
    
    async def process_event(self, conflict_id: int, event: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy adapter: wraps into a SystemEvent and routes through handle_event."""
        system_event = SystemEvent(
            event_id=f"legacy_{conflict_id}",
            event_type=EventType.CONFLICT_UPDATED,
            source_subsystem=SubsystemType.STAKEHOLDER,
            payload={'conflict_id': conflict_id, **event}
        )
        response = await self.handle_event(system_event)
        return response.data

# ===============================================================================
# PUBLIC API - Routes Through Orchestrator
# ===============================================================================

@function_tool
async def create_conflict_stakeholder(
    ctx: RunContextWrapper,
    npc_id: int,
    conflict_id: int,
    suggested_role: Optional[str] = None
) -> str:
    """Create a stakeholder for a conflict via orchestrator."""
    from logic.conflict_system.conflict_synthesizer import get_synthesizer

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

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
    payload = responses[0].data if responses else {'error': 'Failed to create stakeholder'}
    return json.dumps(payload, ensure_ascii=False)

@function_tool
async def stakeholder_take_action(
    ctx: RunContextWrapper,
    stakeholder_id: int,
    conflict_state_json: str,
    options_json: Optional[str] = None
) -> str:
    """Have a stakeholder take an autonomous action via orchestrator."""
    from logic.conflict_system.conflict_synthesizer import get_synthesizer

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

    try:
        conflict_state: Dict[str, Any] = json.loads(conflict_state_json) if conflict_state_json else {}
    except Exception:
        conflict_state = {}

    options: Optional[List[str]] = None
    if options_json:
        try:
            parsed = json.loads(options_json)
            if isinstance(parsed, list):
                options = parsed
        except Exception:
            options = None

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
    payload = responses[0].data if responses else {'error': 'Failed to execute action'}
    return json.dumps(payload, ensure_ascii=False)
