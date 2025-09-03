# logic/conflict_system/conflict_flow.py
"""
Conflict Flow System with LLM-generated pacing and transitions.
Refactored to work as a ConflictSubsystem with the synthesizer.
"""

import logging
import json
import random
import weakref
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from agents import Agent, function_tool, RunContextWrapper, Runner
from db.connection import get_db_connection_context
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # Only for type hints — does not execute at runtime
    from logic.conflict_system.conflict_synthesizer import ConflictSynthesizer
    from logic.conflict_system.conflict_synthesizer import ConflictSubsystem as ConflictSubsystemBase
else:
    # Runtime shim to avoid circular import during class creation
    class ConflictSubsystemBase:  # no methods needed; duck-typed by synthesizer
        pass

def _orch_types():
    from logic.conflict_system.conflict_synthesizer import (
        SubsystemType, EventType, SystemEvent, SubsystemResponse
    )
    return SubsystemType, EventType, SystemEvent, SubsystemResponse
    
from logic.conflict_system.dynamic_conflict_template import extract_runner_response

logger = logging.getLogger(__name__)

# ===============================================================================
# FLOW STRUCTURES
# ===============================================================================

class ConflictPhase(Enum):
    """Phases of conflict progression"""
    SEEDS = "seeds"
    EMERGING = "emerging"
    RISING = "rising"
    CONFRONTATION = "confrontation"
    CLIMAX = "climax"
    FALLING = "falling"
    RESOLUTION = "resolution"
    AFTERMATH = "aftermath"

class PacingStyle(Enum):
    """Different pacing styles for conflicts"""
    SLOW_BURN = "slow_burn"
    RAPID_ESCALATION = "rapid_escalation"
    WAVES = "waves"
    STEADY = "steady"
    ERRATIC = "erratic"

class TransitionType(Enum):
    """Types of phase transitions"""
    NATURAL = "natural"
    TRIGGERED = "triggered"
    FORCED = "forced"
    STALLED = "stalled"
    REVERSED = "reversed"

@dataclass
class ConflictFlow:
    """The flow state of a conflict"""
    conflict_id: int
    current_phase: ConflictPhase
    pacing_style: PacingStyle
    intensity: float
    momentum: float
    phase_progress: float
    transitions_history: List['PhaseTransition']
    dramatic_beats: List['DramaticBeat']
    next_transition_conditions: List[str]

@dataclass
class PhaseTransition:
    """A transition between conflict phases"""
    from_phase: ConflictPhase
    to_phase: ConflictPhase
    transition_type: TransitionType
    trigger: str
    narrative: str
    timestamp: datetime

@dataclass
class DramaticBeat:
    """A significant moment in conflict flow"""
    beat_type: str
    description: str
    impact_on_flow: float
    characters_involved: List[int]
    timestamp: datetime

@dataclass
class FlowModifier:
    """Something that affects conflict flow"""
    modifier_type: str
    source: str
    effect_on_intensity: float
    effect_on_momentum: float
    duration: Optional[timedelta]

# ===============================================================================
# CONFLICT FLOW SUBSYSTEM
# ===============================================================================

class ConflictFlowSubsystem(ConflictSubsystemBase):
    """
    Manages conflict pacing and flow as a subsystem of the synthesizer.
    Controls dramatic rhythm and ensures engaging progression.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None  # weakref set by synthesizer
        
        # Flow states cache
        self._flow_states: Dict[int, ConflictFlow] = {}
        # Pending initializations keyed by operation_id (event_id of CONFLICT_CREATED)
        self._pending_init: Dict[str, Dict[str, Any]] = {}
        
        # Lazy-loaded LLM agents
        self._pacing_director = None
        self._transition_narrator = None
        self._beat_generator = None
        self._flow_analyzer = None
    
    # ========== ConflictSubsystem Interface ==========
    
    @property
    def subsystem_type(self):
        SubsystemType, _, _, _ = _orch_types()
        return SubsystemType.FLOW
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'pacing_control',
            'phase_management',
            'dramatic_beats',
            'flow_analysis',
            'transition_narration',
            'momentum_tracking'
        }
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        return set()  # Flow is foundational
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        # Include HEALTH_CHECK and STATE_SYNC for orchestration integration
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.TENSION_CHANGED,
            EventType.PLAYER_CHOICE,
            EventType.STAKEHOLDER_ACTION,
            EventType.HEALTH_CHECK,
            EventType.STATE_SYNC,
        }
    
    async def initialize(self, synthesizer: 'ConflictSynthesizer') -> bool:
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event):
        """Handle events from synthesizer with robust routing and deferrals (circular-safe)."""
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch_types()
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                return await self._handle_conflict_created(event)
            if event.event_type == EventType.CONFLICT_UPDATED:
                return await self._handle_conflict_updated(event)
            if event.event_type == EventType.TENSION_CHANGED:
                return await self._handle_tension_changed(event)
            if event.event_type == EventType.PLAYER_CHOICE:
                return await self._handle_player_choice(event)
            if event.event_type == EventType.STAKEHOLDER_ACTION:
                return await self._handle_stakeholder_action(event)
            if event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check(),
                    side_effects=[]
                )
            if event.event_type == EventType.STATE_SYNC:
                # Finalize any deferred initializations if conflict_id now known
                return await self._handle_state_sync(event)
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'handled': False},
                side_effects=[]
            )
        except Exception as e:
            logger.error(f"FlowSubsystem error on {event.event_id}: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of flow subsystem"""
        try:
            active_flows = len(self._flow_states)
            stalled_flows = sum(
                1 for f in self._flow_states.values()
                if (f.phase_progress or 0.0) > 0.9 and (f.momentum or 0.0) < 0.1
            )
            
            return {
                'healthy': stalled_flows < max(1, active_flows) / 2,
                'active_flows': active_flows,
                'stalled_flows': stalled_flows,
                'average_momentum': (
                    sum((f.momentum or 0.0) for f in self._flow_states.values()) / max(1, active_flows)
                ),
                'pending_initializations': len(self._pending_init),
            }
        except Exception as e:
            return {'healthy': False, 'issue': str(e)}
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get flow-specific conflict data"""
        flow = self._flow_states.get(conflict_id) or await self._load_flow_state(conflict_id)
        if flow:
            return {
                'subsystem': 'flow',
                'current_phase': flow.current_phase.value,
                'pacing_style': flow.pacing_style.value,
                'intensity': float(flow.intensity),
                'momentum': float(flow.momentum),
                'phase_progress': float(flow.phase_progress),
                'beat_count': len(flow.dramatic_beats)
            }
        return {'subsystem': 'flow', 'no_flow_data': True}
    
    async def get_state(self) -> Dict[str, Any]:
        return {
            'active_flows': len(self._flow_states),
            'phase_distribution': self._get_phase_distribution(),
            'average_intensity': self._get_average_intensity()
        }
    
    # ========== Event Handlers ==========
    
    async def _handle_conflict_created(self, event: SystemEvent) -> SubsystemResponse:
        """Initialize flow for new conflict, deferring if id is not present."""
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')
        conflict_type = payload.get('conflict_type') or 'unknown'
        context = payload.get('context', {}) or {}
        
        # If no id yet (common in creation), defer and wait for STATE_SYNC with operation id
        if not conflict_id:
            self._pending_init[event.event_id] = {
                'conflict_type': conflict_type,
                'context': context,
            }
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'flow_initialized': False, 'deferred': True, 'operation_id': event.event_id},
                side_effects=[]
            )
        
        flow = await self.initialize_conflict_flow(conflict_id, conflict_type, context)
        self._flow_states[conflict_id] = flow
        
        side_effects: List[SystemEvent] = []
        if flow.pacing_style == PacingStyle.RAPID_ESCALATION:
            beat = await self.generate_dramatic_beat(flow, context)
            if beat:
                side_effects.append(SystemEvent(
                    event_id=f"beat_{conflict_id}_{datetime.now().timestamp()}",
                    event_type=EventType.INTENSITY_CHANGED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'conflict_id': conflict_id,
                        'beat': beat.description,
                        'new_intensity': flow.intensity
                    },
                    priority=6
                ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'flow_initialized': True,
                'initial_phase': flow.current_phase.value,
                'pacing_style': flow.pacing_style.value
            },
            side_effects=side_effects
        )
    
    async def _handle_state_sync(self, event: SystemEvent) -> SubsystemResponse:
        """Finalize deferred initialization when we receive conflict_id via STATE_SYNC."""
        payload = event.payload or {}
        op_id = payload.get('operation_id')
        conflict_id = payload.get('conflict_id')
        if op_id and conflict_id and op_id in self._pending_init:
            pending = self._pending_init.pop(op_id)
            flow = await self.initialize_conflict_flow(
                conflict_id,
                pending.get('conflict_type', 'unknown'),
                pending.get('context', {}) or {}
            )
            self._flow_states[conflict_id] = flow
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={
                    'flow_initialized': True,
                    'initial_phase': flow.current_phase.value,
                    'pacing_style': flow.pacing_style.value,
                    'finalized_from_operation': op_id
                },
                side_effects=[]
            )
        # Nothing to do
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'status': 'no_action_taken'},
            side_effects=[]
        )
    
    async def _handle_conflict_updated(self, event: SystemEvent) -> SubsystemResponse:
        """Update flow based on conflict changes"""
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')
        
        if not conflict_id:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'Missing conflict_id'},
                side_effects=[]
            )
        
        flow = self._flow_states.get(conflict_id) or await self._load_flow_state(conflict_id)
        if not flow:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'Flow not found'},
                side_effects=[]
            )
        self._flow_states[conflict_id] = flow
        
        result = await self.update_conflict_flow(flow, payload)
        side_effects: List[SystemEvent] = []
        
        if result.get('transition'):
            transition: PhaseTransition = result['transition']
            side_effects.append(SystemEvent(
                event_id=f"transition_{conflict_id}_{datetime.now().timestamp()}",
                event_type=EventType.PHASE_TRANSITION,
                source_subsystem=self.subsystem_type,
                payload={
                    'conflict_id': conflict_id,
                    'from_phase': transition.from_phase.value,
                    'to_phase': transition.to_phase.value,
                    'narrative': transition.narrative
                },
                priority=4
            ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=result,
            side_effects=side_effects
        )
    
    async def _handle_tension_changed(self, event: SystemEvent) -> SubsystemResponse:
        """Handle tension changes affecting flow"""
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')
        new_tension = float(payload.get('new_tension', 0.5) or 0.5)
        
        flow = self._flow_states.get(conflict_id)
        if flow:
            old_momentum = flow.momentum
            flow.momentum = max(-1.0, min(1.0, flow.momentum + (new_tension - 0.5) * 0.3))
            
            if abs(flow.momentum - old_momentum) > 0.3:
                beat = await self.generate_dramatic_beat(flow, payload)
                if beat:
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'momentum_shift': flow.momentum - old_momentum},
                        side_effects=[SystemEvent(
                            event_id=f"tension_beat_{conflict_id}_{datetime.now().timestamp()}",
                            event_type=EventType.INTENSITY_CHANGED,
                            source_subsystem=self.subsystem_type,
                            payload={'conflict_id': conflict_id, 'beat': beat.description, 'new_intensity': flow.intensity},
                            priority=6
                        )]
                    )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'tension_acknowledged': True},
            side_effects=[]
        )
    
    async def _handle_player_choice(self, event: SystemEvent) -> SubsystemResponse:
        """Handle player choices affecting flow"""
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')
        choice_impact = payload.get('impact', {}) or {}
        
        flow = self._flow_states.get(conflict_id)
        if flow:
            if choice_impact.get('escalate'):
                flow.momentum = min(1.0, flow.momentum + 0.2)
                flow.phase_progress += 0.15
            elif choice_impact.get('de_escalate'):
                flow.momentum = max(-0.5, flow.momentum - 0.2)
            flow.phase_progress = max(0.0, min(1.5, flow.phase_progress))
            
            if flow.phase_progress >= 1.0:
                transition = await self._handle_phase_transition(flow, payload)
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'phase_transition': transition.to_phase.value},
                    side_effects=[SystemEvent(
                        event_id=f"transition_{conflict_id}_{datetime.now().timestamp()}",
                        event_type=EventType.PHASE_TRANSITION,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'conflict_id': conflict_id,
                            'from_phase': transition.from_phase.value,
                            'to_phase': transition.to_phase.value,
                            'narrative': transition.narrative
                        },
                        priority=5
                    )]
                )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'choice_processed': True},
            side_effects=[]
        )
    
    async def _handle_stakeholder_action(self, event: SystemEvent) -> SubsystemResponse:
        """Handle stakeholder actions affecting flow."""
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')
        action_type = (payload.get('action_type') or '').lower()
        
        # If conflict_id missing, try to resolve via stakeholder_id -> stakeholders table
        if not conflict_id and payload.get('stakeholder_id'):
            conflict_id = await self._conflict_id_for_stakeholder(int(payload['stakeholder_id']))
        
        if conflict_id is None:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'no_flow': True, 'reason': 'unknown_conflict'},
                side_effects=[]
            )
        
        flow = self._flow_states.get(conflict_id) or await self._load_flow_state(conflict_id)
        if not flow:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'no_flow': True},
                side_effects=[]
            )
        
        # Different actions have different flow impacts
        if action_type in ('aggressive', 'escalate', 'escalation'):
            flow.intensity = min(1.0, flow.intensity + 0.1)
            flow.momentum = min(1.0, flow.momentum + 0.15)
        elif action_type in ('withdraw', 'evasive', 'de_escalate', 'deescalate'):
            flow.momentum = max(-1.0, flow.momentum - 0.3)
        elif action_type in ('supportive', 'diplomatic', 'mediating', 'mediate'):
            flow.momentum = max(-1.0, min(1.0, flow.momentum + 0.05))
        
        await self._save_flow_state(flow)
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'flow_adjusted': True, 'new_momentum': flow.momentum, 'new_intensity': flow.intensity},
            side_effects=[]
        )
    
    # ========== Agent Properties ==========
    
    @property
    def pacing_director(self) -> Agent:
        if self._pacing_director is None:
            self._pacing_director = Agent(
                name="Pacing Director",
                instructions="""
                Direct the pacing and rhythm of conflicts.
                Keep pacing natural, avoid rushing or dragging.
                """,
                model="gpt-5-nano",
            )
        return self._pacing_director
    
    @property
    def transition_narrator(self) -> Agent:
        if self._transition_narrator is None:
            self._transition_narrator = Agent(
                name="Transition Narrator",
                instructions="""
                Narrate transitions between conflict phases with clear, earned shifts.
                """,
                model="gpt-5-nano",
            )
        return self._transition_narrator
    
    @property
    def beat_generator(self) -> Agent:
        if self._beat_generator is None:
            self._beat_generator = Agent(
                name="Dramatic Beat Generator",
                instructions="""
                Generate dramatic beats that advance conflict meaningfully.
                """,
                model="gpt-5-nano",
            )
        return self._beat_generator
    
    @property
    def flow_analyzer(self) -> Agent:
        if self._flow_analyzer is None:
            self._flow_analyzer = Agent(
                name="Flow Analyzer",
                instructions="""
                Analyze flow effectiveness and advise on intensity, momentum, and transitions.
                """,
                model="gpt-5-nano",
            )
        return self._flow_analyzer
    
    # ========== Core Flow Methods ==========
    
    async def initialize_conflict_flow(
        self,
        conflict_id: int,
        conflict_type: str,
        initial_context: Dict[str, Any]
    ) -> ConflictFlow:
        """Initialize flow for a new conflict"""
        prompt = f"""
Initialize conflict flow:

Conflict Type: {conflict_type}
Context: {json.dumps(initial_context, indent=2)}

Return JSON:
{{
  "phase": "seeds|emerging|rising",
  "pacing": "slow_burn|rapid_escalation|waves|steady|erratic",
  "intensity": 0.0,
  "momentum": 0.0,
  "conditions": ["..."]
}}
"""
        response = await Runner.run(self.pacing_director, prompt)
        try:
            result = json.loads(extract_runner_response(response))
        except Exception:
            result = {}
        
        phase = (result.get('phase') or 'emerging').upper()
        pacing = (result.get('pacing') or 'steady').upper()
        intensity = float(result.get('intensity', 0.3) or 0.3)
        momentum = float(result.get('momentum', 0.2) or 0.2)
        conditions = result.get('conditions', []) or []
        
        # Persist
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO conflict_flows
                    (user_id, conversation_id, conflict_id, current_phase, pacing_style, intensity, momentum, phase_progress)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 0.0)
                ON CONFLICT (conflict_id) DO UPDATE
                SET current_phase = EXCLUDED.current_phase,
                    pacing_style = EXCLUDED.pacing_style,
                    intensity = EXCLUDED.intensity,
                    momentum = EXCLUDED.momentum
            """, self.user_id, self.conversation_id, conflict_id,
               phase.lower(), pacing.lower(), intensity, momentum)
        
        return ConflictFlow(
            conflict_id=conflict_id,
            current_phase=ConflictPhase[phase],
            pacing_style=PacingStyle[pacing],
            intensity=max(0.0, min(1.0, intensity)),
            momentum=max(-1.0, min(1.0, momentum)),
            phase_progress=0.0,
            transitions_history=[],
            dramatic_beats=[],
            next_transition_conditions=conditions
        )
    
    async def update_conflict_flow(
        self,
        flow: ConflictFlow,
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update flow based on event data."""
        prompt = f"""
Update conflict flow:

Current State:
- Phase: {flow.current_phase.value}
- Intensity: {flow.intensity}
- Momentum: {flow.momentum}
- Progress: {flow.phase_progress}

Event: {json.dumps(event, indent=2)}

Return JSON:
{{
  "intensity": 0.0,
  "momentum": 0.0,
  "progress_change": 0.0,
  "should_transition": true/false,
  "narrative_impact": "..."
}}
"""
        response = await Runner.run(self.flow_analyzer, prompt)
        try:
            result = json.loads(extract_runner_response(response))
        except Exception:
            result = {}
        
        old_intensity = flow.intensity
        flow.intensity = max(0.0, min(1.0, float(result.get('intensity', flow.intensity) or flow.intensity)))
        flow.momentum = max(-1.0, min(1.0, float(result.get('momentum', flow.momentum) or flow.momentum)))
        flow.phase_progress = max(0.0, flow.phase_progress + float(result.get('progress_change', 0.1) or 0.1))
        
        transition = None
        if bool(result.get('should_transition')) or flow.phase_progress >= 1.0:
            transition = await self._handle_phase_transition(flow, event)
        
        await self._save_flow_state(flow)
        
        return {
            'intensity_change': flow.intensity - old_intensity,
            'new_momentum': flow.momentum,
            'phase_progress': flow.phase_progress,
            'transition': transition,
            'narrative_impact': result.get('narrative_impact', 'The conflict evolves')
        }
    
    async def generate_dramatic_beat(
        self,
        flow: ConflictFlow,
        context: Dict[str, Any]
    ) -> Optional[DramaticBeat]:
        """Generate a dramatic beat for the conflict"""
        prompt = f"""
Generate dramatic beat:

Current Phase: {flow.current_phase.value}
Intensity: {flow.intensity}
Momentum: {flow.momentum}
Context: {json.dumps(context, indent=2)}

Return JSON:
{{
  "type": "revelation|betrayal|escalation|reconciliation|twist|moment",
  "description": "...",
  "impact": 0.0,
  "characters": [1,2,3]
}}
"""
        response = await Runner.run(self.beat_generator, prompt)
        try:
            result = json.loads(extract_runner_response(response))
        except Exception:
            result = {}
        
        beat = DramaticBeat(
            beat_type=result.get('type', 'moment'),
            description=result.get('description', 'A significant moment occurs'),
            impact_on_flow=float(result.get('impact', 0.2) or 0.2),
            characters_involved=result.get('characters', []) or [],
            timestamp=datetime.now()
        )
        # Apply beat impact
        flow.momentum = max(-1.0, min(1.0, flow.momentum + beat.impact_on_flow * 0.5))
        flow.intensity = max(0.0, min(1.0, flow.intensity + abs(beat.impact_on_flow) * 0.3))
        flow.dramatic_beats.append(beat)
        
        await self._store_dramatic_beat(flow.conflict_id, beat)
        await self._save_flow_state(flow)
        return beat
    
    # ========== Helper Methods ==========
    
    async def _handle_phase_transition(
        self,
        flow: ConflictFlow,
        trigger_event: Dict[str, Any]
    ) -> PhaseTransition:
        """Handle transition between phases"""
        next_phase = await self._determine_next_phase(flow, trigger_event)
        
        prompt = f"""
Narrate phase transition:

From: {flow.current_phase.value}
To: {next_phase.value}
Trigger: {json.dumps(trigger_event, indent=2)}
Current Intensity: {flow.intensity}

Return JSON:
{{
  "type": "natural|triggered|forced|stalled|reversed",
  "narrative": "2-3 sentences"
}}
"""
        response = await Runner.run(self.transition_narrator, prompt)
        try:
            result = json.loads(extract_runner_response(response))
        except Exception:
            result = {}
        
        transition = PhaseTransition(
            from_phase=flow.current_phase,
            to_phase=next_phase,
            transition_type=TransitionType[(result.get('type', 'natural') or 'natural').upper()],
            trigger=str(trigger_event),
            narrative=result.get('narrative', 'The conflict shifts'),
            timestamp=datetime.now()
        )
        
        # Update flow
        flow.current_phase = next_phase
        flow.phase_progress = 0.0
        flow.transitions_history.append(transition)
        
        await self._store_transition(flow.conflict_id, transition)
        await self._save_flow_state(flow)
        return transition
    
    async def _determine_next_phase(
        self,
        flow: ConflictFlow,
        trigger: Dict[str, Any]
    ) -> ConflictPhase:
        """Determine the next phase in progression"""
        natural_progression = {
            ConflictPhase.SEEDS: ConflictPhase.EMERGING,
            ConflictPhase.EMERGING: ConflictPhase.RISING,
            ConflictPhase.RISING: ConflictPhase.CONFRONTATION,
            ConflictPhase.CONFRONTATION: ConflictPhase.CLIMAX,
            ConflictPhase.CLIMAX: ConflictPhase.FALLING,
            ConflictPhase.FALLING: ConflictPhase.RESOLUTION,
            ConflictPhase.RESOLUTION: ConflictPhase.AFTERMATH,
            ConflictPhase.AFTERMATH: ConflictPhase.AFTERMATH
        }
        if flow.momentum < -0.5:  # Reversal cases
            reverse_progression = {
                ConflictPhase.RISING: ConflictPhase.EMERGING,
                ConflictPhase.CONFRONTATION: ConflictPhase.RISING,
                ConflictPhase.CLIMAX: ConflictPhase.CONFRONTATION
            }
            return reverse_progression.get(flow.current_phase, natural_progression[flow.current_phase])
        return natural_progression[flow.current_phase]
    
    async def _load_flow_state(self, conflict_id: int) -> Optional[ConflictFlow]:
        """Load flow state from database"""
        async with get_db_connection_context() as conn:
            flow_data = await conn.fetchrow("""
                SELECT * FROM conflict_flows
                WHERE user_id = $1 AND conversation_id = $2 AND conflict_id = $3
            """, self.user_id, self.conversation_id, conflict_id)
        
        if flow_data:
            return ConflictFlow(
                conflict_id=conflict_id,
                current_phase=ConflictPhase[flow_data['current_phase'].upper()],
                pacing_style=PacingStyle[flow_data['pacing_style'].upper()],
                intensity=float(flow_data['intensity']),
                momentum=float(flow_data['momentum']),
                phase_progress=float(flow_data.get('phase_progress', 0.5) or 0.5),
                transitions_history=[],
                dramatic_beats=[],
                next_transition_conditions=[]
            )
        return None
    
    async def _save_flow_state(self, flow: ConflictFlow):
        """Save flow state to database"""
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE conflict_flows
                SET current_phase = $1,
                    intensity = $2,
                    momentum = $3,
                    phase_progress = $4
                WHERE user_id = $5 AND conversation_id = $6 AND conflict_id = $7
            """, flow.current_phase.value, float(flow.intensity), float(flow.momentum),
               float(flow.phase_progress), self.user_id, self.conversation_id, flow.conflict_id)
    
    async def _store_transition(self, conflict_id: int, transition: PhaseTransition):
        """Store phase transition in database"""
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO phase_transitions
                    (user_id, conversation_id, conflict_id, from_phase, to_phase,
                     transition_type, trigger, narrative, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
            """, self.user_id, self.conversation_id, conflict_id,
               transition.from_phase.value, transition.to_phase.value,
               transition.transition_type.value, transition.trigger, transition.narrative)
    
    async def _store_dramatic_beat(self, conflict_id: int, beat: DramaticBeat):
        """Store dramatic beat in database"""
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO dramatic_beats
                    (user_id, conversation_id, conflict_id, beat_type,
                     description, impact, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            """, self.user_id, self.conversation_id, conflict_id,
               beat.beat_type, beat.description, float(beat.impact_on_flow))
    
    def _get_phase_distribution(self) -> Dict[str, int]:
        """Get distribution of conflicts across phases"""
        distribution: Dict[str, int] = {}
        for flow in self._flow_states.values():
            phase = flow.current_phase.value
            distribution[phase] = distribution.get(phase, 0) + 1
        return distribution
    
    def _get_average_intensity(self) -> float:
        """Get average intensity across all flows"""
        if not self._flow_states:
            return 0.0
        return sum((f.intensity or 0.0) for f in self._flow_states.values()) / max(1, len(self._flow_states))
    
    def _create_default_flow(self, conflict_id: int) -> ConflictFlow:
        """Create default flow if LLM fails"""
        return ConflictFlow(
            conflict_id=conflict_id,
            current_phase=ConflictPhase.EMERGING,
            pacing_style=PacingStyle.STEADY,
            intensity=0.3,
            momentum=0.2,
            phase_progress=0.0,
            transitions_history=[],
            dramatic_beats=[],
            next_transition_conditions=[]
        )
    
    def _create_fallback_transition(
        self,
        from_phase: ConflictPhase,
        to_phase: ConflictPhase
    ) -> PhaseTransition:
        """Create fallback transition if LLM fails"""
        return PhaseTransition(
            from_phase=from_phase,
            to_phase=to_phase,
            transition_type=TransitionType.NATURAL,
            trigger="Natural progression",
            narrative="The conflict evolves to its next stage",
            timestamp=datetime.now()
        )
    
    def _create_fallback_beat(self) -> DramaticBeat:
        """Create fallback beat if LLM fails"""
        return DramaticBeat(
            beat_type="moment",
            description="A significant moment in the conflict",
            impact_on_flow=0.2,
            characters_involved=[],
            timestamp=datetime.now()
        )
    
    async def _conflict_id_for_stakeholder(self, stakeholder_id: int) -> Optional[int]:
        """Resolve conflict_id from a stakeholder id via DB."""
        async with get_db_connection_context() as conn:
            cid = await conn.fetchval("""
                SELECT conflict_id FROM stakeholders
                WHERE user_id = $1 AND conversation_id = $2 AND stakeholder_id = $3
            """, self.user_id, self.conversation_id, stakeholder_id)
        return int(cid) if cid is not None else None
