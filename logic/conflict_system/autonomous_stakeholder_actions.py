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
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import weakref

from openai import AsyncOpenAI

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

_client: Optional[AsyncOpenAI] = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client

_json_block_re = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")

def _extract_json(text: str) -> str:
    """
    Extract the first JSON object/array from a possibly noisy LLM response.
    """
    if not text:
        return "{}"
    # Direct JSON?
    text = text.strip()
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text
    # Try code fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        if inner:
            return inner
    # Fallback: first JSON-ish block
    m = _json_block_re.search(text)
    if m:
        return m.group(0)
    return "{}"

async def llm_json(prompt: str) -> Dict[str, Any]:
    """
    Call OpenAI Responses API and parse JSON output robustly, without temperature/max tokens.
    """
    try:
        client = _get_client()
        resp = await client.responses.create(
            model="gpt-5-nano",
            input=prompt,
        )
        # output_text is the convenience field on new SDKs
        text = getattr(resp, "output_text", None)
        if not text:
            # Fallback: stitch from structured output if needed
            try:
                parts = []
                for item in getattr(resp, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                        t = getattr(c, "text", None)
                        if t:
                            parts.append(t)
                text = "\n".join(parts).strip()
            except Exception:
                text = ""
        payload = _extract_json(text)
        return json.loads(payload)
    except Exception as e:
        logger.warning(f"llm_json failed: {e}")
        return {}

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
        actions: List[StakeholderAction] = []
        side_effects: List[SystemEvent] = []
        
        for s in acting_stakeholders:
            action = await self.make_autonomous_decision(s, payload)
            if action:
                actions.append(action)
                self._pending_actions.append(action)
                side_effects.append(SystemEvent(
                    event_id=f"action_{action.action_id}",
                    event_type=EventType.STAKEHOLDER_ACTION,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'stakeholder_id': s.stakeholder_id,
                        'action_type': action.action_type.value,
                        'target_id': action.target,
                        'intensity': action.success_probability
                    }
                ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'actions_taken': len(actions), 'action_types': [a.action_type.value for a in actions]},
            side_effects=side_effects
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
        
        reactions: List[StakeholderReaction] = []
        for s in self._active_stakeholders.values():
            if self._should_react_to_choice(s, choice_type, target_npc):
                triggering_action = StakeholderAction(
                    action_id=0,
                    stakeholder_id=0,
                    action_type=ActionType.STRATEGIC,
                    description=f"Player choice: {choice_type}",
                    target=target_npc,
                    resources_used={},
                    success_probability=1.0,
                    consequences={},
                    timestamp=datetime.now()
                )
                reaction = await self.generate_reaction(s, triggering_action, payload)
                reactions.append(reaction)
                # Update stress lightly based on emotion
                if reaction.emotional_response in ['angry', 'fearful', 'frustrated']:
                    s.stress_level = min(1.0, s.stress_level + 0.1)
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'reactions_generated': len(reactions), 'emotional_responses': [r.emotional_response for r in reactions]},
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
        
        npcs_present = scene_context.get('npcs', []) or []
        npc_behaviors: Dict[int, str] = {}
        for npc_id in npcs_present:
            s = self._find_stakeholder_by_npc(npc_id)
            if s:
                npc_behaviors[npc_id] = self._determine_scene_behavior(s)
        
        autonomous_actions = []
        for s in self._active_stakeholders.values():
            if s.npc_id in npcs_present and self._should_take_autonomous_action(s, scene_context):
                action = await self.make_autonomous_decision(s, scene_context)
                if action:
                    autonomous_actions.append(action)
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'npc_behaviors': npc_behaviors,
                'autonomous_actions': [
                    {'stakeholder': a.stakeholder_id, 'action': a.description, 'type': a.action_type.value}
                    for a in autonomous_actions
                ],
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
        conflict_state = payload.get('conflict_state') or {}
        options = payload.get('options')  # optional list[str]
        action = await self.make_autonomous_decision(s, conflict_state, options)
        if not action:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'no action generated'},
                side_effects=[]
            )
        self._pending_actions.append(action)
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'action_id': action.action_id,
                'action_type': action.action_type.value,
                'description': action.description,
                'success_probability': action.success_probability,
                'target': action.target,
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
        """Create a stakeholder with personality-driven characteristics via LLM."""
        npc_details = await self._get_npc_details(npc_id)
        
        prompt = f"""
Create stakeholder profile as JSON:
NPC: {npc_details.get('name', 'Unknown')}
Personality: {npc_details.get('personality_traits', 'Unknown')}
Conflict Context: Conflict #{conflict_id}
Suggested Role: {initial_role or 'determine based on personality'}

Return JSON:
{{
  "role": "bystander|instigator|defender|mediator|opportunist|victim|escalator|peacemaker",
  "decision_style": "reactive|rational|emotional|instinctive|calculating|principled",
  "goals": ["..."],
  "resources": {{"influence": 0.0, "wealth": 0.0}},
  "stress_level": 0.0,
  "commitment_level": 0.0
}}
"""
        result = await llm_json(prompt)
        try:
            # Store stakeholder
            async with get_db_connection_context() as conn:
                stakeholder_id = await conn.fetchval(
                    """
                    INSERT INTO stakeholders
                    (user_id, conversation_id, npc_id, conflict_id,
                     role, decision_style, stress_level, commitment_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING stakeholder_id
                    """,
                    self.user_id, self.conversation_id, npc_id, conflict_id,
                    result.get('role', 'bystander'),
                    result.get('decision_style', 'reactive'),
                    float(result.get('stress_level', 0.3) or 0.3),
                    float(result.get('commitment_level', 0.5) or 0.5)
                )
            
            stakeholder = Stakeholder(
                stakeholder_id=stakeholder_id,
                npc_id=npc_id,
                name=npc_details.get('name', 'Unknown'),
                personality_traits=json.loads(npc_details.get('personality_traits', '[]')) if isinstance(npc_details.get('personality_traits'), str) else (npc_details.get('personality_traits') or []),
                current_role=StakeholderRole[(result.get('role', 'bystander') or 'bystander').upper()],
                decision_style=DecisionStyle[(result.get('decision_style', 'reactive') or 'reactive').upper()],
                goals=result.get('goals', []),
                resources=result.get('resources', {}),
                relationships={},
                stress_level=float(result.get('stress_level', 0.3) or 0.3),
                commitment_level=float(result.get('commitment_level', 0.5) or 0.5)
            )
            
            # Notify orchestrator of new stakeholder (best effort)
            if self._synthesizer and self._synthesizer():
                await self._synthesizer().emit_event(SystemEvent(
                    event_id=f"stakeholder_{stakeholder_id}",
                    event_type=EventType.STAKEHOLDER_ACTION,
                    source_subsystem=self.subsystem_type,
                    payload={'action_type': 'stakeholder_created', 'stakeholder_id': stakeholder_id, 'role': stakeholder.current_role.value}
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
        """Make an autonomous decision for a stakeholder via LLM."""
        prompt = f"""
Make decision as JSON:
Character: {stakeholder.name}
Personality: {stakeholder.personality_traits}
Role: {stakeholder.current_role.value}
Decision Style: {stakeholder.decision_style.value}
Stress Level: {stakeholder.stress_level}

Conflict State: {json.dumps(conflict_state, indent=2)}
Available Options: {json.dumps(available_options) if available_options else "default"}

Return JSON:
{{
  "action_type": "observant|aggressive|defensive|diplomatic|manipulative|supportive|evasive|strategic",
  "description": "What they do",
  "target": 123,
  "resources": {{}},
  "success_probability": 0.0,
  "consequences": {{}}
}}
"""
        result = await llm_json(prompt)
        try:
            async with get_db_connection_context() as conn:
                action_id = await conn.fetchval(
                    """
                    INSERT INTO stakeholder_actions
                    (user_id, conversation_id, stakeholder_id, action_type,
                     description, success_probability)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING action_id
                    """,
                    self.user_id, self.conversation_id,
                    stakeholder.stakeholder_id,
                    (result.get('action_type') or 'observant'),
                    result.get('description', 'Observes the situation'),
                    float(result.get('success_probability', 0.5) or 0.5)
                )
            return StakeholderAction(
                action_id=action_id,
                stakeholder_id=stakeholder.stakeholder_id,
                action_type=ActionType[(result.get('action_type', 'observant') or 'observant').upper()],
                description=result.get('description', 'Takes action'),
                target=result.get('target'),
                resources_used=result.get('resources', {}),
                success_probability=float(result.get('success_probability', 0.5) or 0.5),
                consequences=result.get('consequences', {}),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Decision generation failed for stakeholder {stakeholder.stakeholder_id}: {e}")
            # Fallback conservative action
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
    
    async def generate_reaction(
        self,
        stakeholder: Stakeholder,
        triggering_action: StakeholderAction,
        action_context: Dict[str, Any]
    ) -> StakeholderReaction:
        """Generate a reaction to another stakeholder's action via LLM."""
        prompt = f"""
Generate reaction as JSON:
Reacting Character: {stakeholder.name}
Personality: {stakeholder.personality_traits}
Current Stress: {stakeholder.stress_level}

Triggering Action: {triggering_action.description} ({triggering_action.action_type.value})

Return JSON:
{{
  "reaction_type": "counter|support|ignore|escalate|de-escalate",
  "description": "What they do",
  "emotional_response": "neutral|angry|fearful|surprised|relieved",
  "relationship_impact": 0.0,
  "stress_impact": 0.0
}}
"""
        result = await llm_json(prompt)
        try:
            async with get_db_connection_context() as conn:
                reaction_id = await conn.fetchval(
                    """
                    INSERT INTO stakeholder_reactions
                    (user_id, conversation_id, stakeholder_id, triggering_action_id,
                     reaction_type, description, emotional_response)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING reaction_id
                    """,
                    self.user_id, self.conversation_id,
                    stakeholder.stakeholder_id, triggering_action.action_id,
                    result.get('reaction_type', 'observe'),
                    result.get('description', 'Reacts to the action'),
                    result.get('emotional_response', 'neutral')
                )
            # Update stakeholder stress (clamped)
            stakeholder.stress_level = max(0.0, min(1.0, stakeholder.stress_level + float(result.get('stress_impact', 0.0) or 0.0)))
            return StakeholderReaction(
                reaction_id=reaction_id,
                stakeholder_id=stakeholder.stakeholder_id,
                triggering_action_id=triggering_action.action_id,
                reaction_type=result.get('reaction_type', 'observe'),
                description=result.get('description', 'Reacts'),
                emotional_response=result.get('emotional_response', 'neutral'),
                relationship_impact={triggering_action.stakeholder_id: float(result.get('relationship_impact', 0.0) or 0.0)}
            )
        except Exception as e:
            logger.warning(f"Reaction generation failed for stakeholder {stakeholder.stakeholder_id}: {e}")
            return StakeholderReaction(
                reaction_id=0,
                stakeholder_id=stakeholder.stakeholder_id,
                triggering_action_id=triggering_action.action_id,
                reaction_type="observe",
                description="Notices the action",
                emotional_response="neutral",
                relationship_impact={}
            )
    
    async def adapt_stakeholder_role(
        self,
        stakeholder: Stakeholder,
        changing_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt stakeholder role based on changing conditions via LLM."""
        prompt = f"""
Evaluate role adaptation as JSON:
Character: {stakeholder.name}
Current Role: {stakeholder.current_role.value}
Stress: {stakeholder.stress_level}

Changing Conditions: {json.dumps(changing_conditions, indent=2)}

Return JSON:
{{
  "change_role": true/false,
  "new_role": "bystander|instigator|defender|mediator|opportunist|victim|escalator|peacemaker",
  "reason": "..."
}}
"""
        result = await llm_json(prompt)
        try:
            change = bool(result.get('change_role', False))
            if change:
                new_role = (result.get('new_role') or 'bystander').upper()
                old_role = stakeholder.current_role
                stakeholder.current_role = StakeholderRole[new_role]
                async with get_db_connection_context() as conn:
                    await conn.execute(
                        "UPDATE stakeholders SET role = $1 WHERE stakeholder_id = $2",
                        stakeholder.current_role.value, stakeholder.stakeholder_id
                    )
                return {'role_changed': True, 'old_role': old_role.value, 'new_role': stakeholder.current_role.value, 'reason': result.get('reason', 'Circumstances changed')}
            return {'role_changed': False}
        except Exception as e:
            logger.warning(f"Role adaptation failed for stakeholder {stakeholder.stakeholder_id}: {e}")
            return {'role_changed': False}

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
