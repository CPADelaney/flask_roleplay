# logic/conflict_system/conflict_flow.py
"""
Conflict Flow System with LLM-generated pacing and transitions.
Refactored to work as a ConflictSubsystem with the synthesizer.
"""

import logging
import json
import random
import weakref
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from logic.conflict_system.conflict_synthesizer import (
    ConflictSubsystem, SubsystemType, EventType,
    SystemEvent, SubsystemResponse
)

logger = logging.getLogger(__name__)

# ===============================================================================
# FLOW STRUCTURES (Preserved from original)
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

class ConflictFlowSubsystem(ConflictSubsystem):
    """
    Manages conflict pacing and flow as a subsystem of the synthesizer.
    Controls dramatic rhythm and ensures engaging progression.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None  # Will be set by synthesizer
        
        # Flow states cache
        self._flow_states: Dict[int, ConflictFlow] = {}
        
        # Lazy-loaded LLM agents
        self._pacing_director = None
        self._transition_narrator = None
        self._beat_generator = None
        self._flow_analyzer = None
    
    # ========== ConflictSubsystem Interface Implementation ==========
    
    @property
    def subsystem_type(self) -> SubsystemType:
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
        return set()  # Flow is foundational, no dependencies
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.TENSION_CHANGED,
            EventType.PLAYER_CHOICE,
            EventType.STAKEHOLDER_ACTION
        }
    
    async def initialize(self, synthesizer: 'ConflictSynthesizer') -> bool:
        """Initialize with synthesizer reference"""
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Handle events from synthesizer"""
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                return await self._handle_conflict_created(event)
            elif event.event_type == EventType.CONFLICT_UPDATED:
                return await self._handle_conflict_updated(event)
            elif event.event_type == EventType.TENSION_CHANGED:
                return await self._handle_tension_changed(event)
            elif event.event_type == EventType.PLAYER_CHOICE:
                return await self._handle_player_choice(event)
            elif event.event_type == EventType.STAKEHOLDER_ACTION:
                return await self._handle_stakeholder_action(event)
            else:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'handled': False}
                )
        except Exception as e:
            logger.error(f"FlowSubsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of flow subsystem"""
        try:
            active_flows = len(self._flow_states)
            stalled_flows = sum(1 for f in self._flow_states.values() 
                              if f.phase_progress > 0.9 and f.momentum < 0.1)
            
            return {
                'healthy': stalled_flows < active_flows / 2,
                'active_flows': active_flows,
                'stalled_flows': stalled_flows,
                'average_momentum': sum(f.momentum for f in self._flow_states.values()) / max(1, active_flows)
            }
        except Exception as e:
            return {'healthy': False, 'issue': str(e)}
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get flow-specific conflict data"""
        flow = self._flow_states.get(conflict_id)
        if not flow:
            flow = await self._load_flow_state(conflict_id)
        
        if flow:
            return {
                'subsystem': 'flow',
                'current_phase': flow.current_phase.value,
                'pacing_style': flow.pacing_style.value,
                'intensity': flow.intensity,
                'momentum': flow.momentum,
                'phase_progress': flow.phase_progress,
                'beat_count': len(flow.dramatic_beats)
            }
        return {'subsystem': 'flow', 'no_flow_data': True}
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of flow subsystem"""
        return {
            'active_flows': len(self._flow_states),
            'phase_distribution': self._get_phase_distribution(),
            'average_intensity': self._get_average_intensity()
        }
    
    # ========== Event Handlers ==========
    
    async def _handle_conflict_created(self, event: SystemEvent) -> SubsystemResponse:
        """Initialize flow for new conflict"""
        conflict_id = event.payload.get('conflict_id')
        conflict_type = event.payload.get('conflict_type')
        context = event.payload.get('context', {})
        
        if not conflict_id:
            # If no ID yet, request one
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'No conflict_id provided'}
            )
        
        # Initialize flow
        flow = await self.initialize_conflict_flow(conflict_id, conflict_type, context)
        self._flow_states[conflict_id] = flow
        
        # Check for initial dramatic beat
        side_effects = []
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
    
    async def _handle_conflict_updated(self, event: SystemEvent) -> SubsystemResponse:
        """Update flow based on conflict changes"""
        conflict_id = event.payload.get('conflict_id')
        
        flow = self._flow_states.get(conflict_id)
        if not flow:
            flow = await self._load_flow_state(conflict_id)
            if not flow:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=False,
                    data={'error': 'Flow not found'}
                )
            self._flow_states[conflict_id] = flow
        
        # Update flow
        result = await self.update_conflict_flow(flow, event.payload)
        
        side_effects = []
        
        # Check for phase transition
        if result.get('transition'):
            transition = result['transition']
            side_effects.append(SystemEvent(
                event_id=f"transition_{conflict_id}",
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
        conflict_id = event.payload.get('conflict_id')
        new_tension = event.payload.get('new_tension', 0.5)
        
        flow = self._flow_states.get(conflict_id)
        if flow:
            # Adjust momentum based on tension change
            old_momentum = flow.momentum
            flow.momentum = max(-1, min(1, flow.momentum + (new_tension - 0.5) * 0.3))
            
            # Check if this triggers a beat
            if abs(flow.momentum - old_momentum) > 0.3:
                beat = await self.generate_dramatic_beat(flow, event.payload)
                if beat:
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'momentum_shift': flow.momentum - old_momentum},
                        side_effects=[SystemEvent(
                            event_id=f"tension_beat_{conflict_id}",
                            event_type=EventType.INTENSITY_CHANGED,
                            source_subsystem=self.subsystem_type,
                            payload={'conflict_id': conflict_id, 'beat': beat.description}
                        )]
                    )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'tension_acknowledged': True}
        )
    
    async def _handle_player_choice(self, event: SystemEvent) -> SubsystemResponse:
        """Handle player choices affecting flow"""
        conflict_id = event.payload.get('conflict_id')
        choice_impact = event.payload.get('impact', {})
        
        flow = self._flow_states.get(conflict_id)
        if flow:
            # Player choices can accelerate or decelerate flow
            if choice_impact.get('escalate'):
                flow.momentum = min(1, flow.momentum + 0.2)
                flow.phase_progress += 0.15
            elif choice_impact.get('de_escalate'):
                flow.momentum = max(-0.5, flow.momentum - 0.2)
            
            # Check for phase transition
            if flow.phase_progress >= 1.0:
                transition = await self._handle_phase_transition(flow, event.payload)
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'phase_transition': transition.to_phase.value}
                )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'choice_processed': True}
        )
    
    async def _handle_stakeholder_action(self, event: SystemEvent) -> SubsystemResponse:
        """Handle stakeholder actions affecting flow"""
        conflict_id = event.payload.get('conflict_id')
        action_type = event.payload.get('action_type')
        
        flow = self._flow_states.get(conflict_id)
        if flow:
            # Different actions have different flow impacts
            if action_type == 'escalate':
                flow.intensity = min(1, flow.intensity + 0.1)
                flow.momentum = min(1, flow.momentum + 0.15)
            elif action_type == 'withdraw':
                flow.momentum = max(-1, flow.momentum - 0.3)
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'flow_adjusted': True, 'new_momentum': flow.momentum}
            )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'no_flow': True}
        )
    
    # ========== Agent Properties (Preserved from original) ==========
    
    @property
    def pacing_director(self) -> Agent:
        """Agent for managing conflict pacing"""
        if self._pacing_director is None:
            self._pacing_director = Agent(
                name="Pacing Director",
                instructions="""
                Direct the pacing and rhythm of conflicts.
                
                Consider:
                - Current dramatic tension
                - Player engagement and fatigue
                - Natural story rhythms
                - Build-up and release patterns
                - Slice-of-life vs dramatic moments
                
                Create pacing that:
                - Feels natural and engaging
                - Avoids both rushing and dragging
                - Builds anticipation
                - Provides breathing room
                - Maintains interest
                
                Think like a film editor managing tension.
                """,
                model="gpt-5-nano",
            )
        return self._pacing_director
    
    @property
    def transition_narrator(self) -> Agent:
        """Agent for narrating phase transitions"""
        if self._transition_narrator is None:
            self._transition_narrator = Agent(
                name="Transition Narrator",
                instructions="""
                Narrate transitions between conflict phases.
                
                Create transitions that:
                - Feel earned and natural
                - Show clear change
                - Maintain continuity
                - Build on what came before
                - Set up what's next
                
                Focus on:
                - Emotional shifts
                - Power dynamic changes
                - Environmental cues
                - Character reactions
                - Subtle turning points
                
                Make transitions smooth but noticeable.
                """,
                model="gpt-5-nano",
            )
        return self._transition_narrator
    
    @property
    def beat_generator(self) -> Agent:
        """Agent for generating dramatic beats"""
        if self._beat_generator is None:
            self._beat_generator = Agent(
                name="Dramatic Beat Generator",
                instructions="""
                Generate dramatic beats within conflicts.
                
                Create beats that:
                - Advance the conflict meaningfully
                - Reveal character or information
                - Shift dynamics
                - Create memorable moments
                - Feel organic to the situation
                
                Types of beats:
                - Revelations and discoveries
                - Betrayals and alliances
                - Escalations and de-escalations
                - Character moments
                - Environmental changes
                
                Balance impact with believability.
                """,
                model="gpt-5-nano",
            )
        return self._beat_generator
    
    @property
    def flow_analyzer(self) -> Agent:
        """Agent for analyzing conflict flow"""
        if self._flow_analyzer is None:
            self._flow_analyzer = Agent(
                name="Flow Analyzer",
                instructions="""
                Analyze the flow and rhythm of conflicts.
                
                Assess:
                - Pacing effectiveness
                - Dramatic momentum
                - Player engagement indicators
                - Natural transition points
                - Flow problems or stagnation
                
                Provide insights on:
                - When to accelerate or slow down
                - Optimal transition timing
                - Missing dramatic elements
                - Flow improvement opportunities
                
                Think like a story analyst.
                """,
                model="gpt-5-nano",
            )
        return self._flow_analyzer
    
    # ========== Core Flow Methods (Modified from original) ==========
    
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
        
        Determine:
        1. Starting phase (seeds/emerging/rising)
        2. Pacing style (slow_burn/rapid_escalation/waves/steady/erratic)
        3. Initial intensity (0.0-1.0)
        4. Initial momentum (-1.0 to 1.0)
        5. Conditions for first transition
        6. Estimated total duration
        
        Match pacing to conflict type and context.
        Format as JSON.
        """
        
        response = await self.pacing_director.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store flow state
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO conflict_flows
                    (user_id, conversation_id, conflict_id, current_phase, 
                     pacing_style, intensity, momentum)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (conflict_id) DO UPDATE
                    SET current_phase = $4, pacing_style = $5, 
                        intensity = $6, momentum = $7
                """, self.user_id, self.conversation_id, conflict_id,
                result.get('phase', 'emerging'),
                result.get('pacing', 'steady'),
                result.get('intensity', 0.3),
                result.get('momentum', 0.2))
            
            return ConflictFlow(
                conflict_id=conflict_id,
                current_phase=ConflictPhase[result.get('phase', 'EMERGING').upper()],
                pacing_style=PacingStyle[result.get('pacing', 'STEADY').upper()],
                intensity=result.get('intensity', 0.3),
                momentum=result.get('momentum', 0.2),
                phase_progress=0.0,
                transitions_history=[],
                dramatic_beats=[],
                next_transition_conditions=result.get('conditions', [])
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to initialize flow: {e}")
            return self._create_default_flow(conflict_id)
    
    async def update_conflict_flow(
        self,
        flow: ConflictFlow,
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update flow based on events"""
        
        prompt = f"""
        Update conflict flow:
        
        Current State:
        - Phase: {flow.current_phase.value}
        - Intensity: {flow.intensity}
        - Momentum: {flow.momentum}
        - Progress: {flow.phase_progress}
        
        Event: {json.dumps(event, indent=2)}
        
        Determine:
        1. New intensity (0-1)
        2. New momentum (-1 to 1)
        3. Phase progress change
        4. Should transition? (yes/no)
        5. Flow modifiers to apply
        6. Narrative impact
        
        Keep changes proportional to event significance.
        Format as JSON.
        """
        
        response = await self.flow_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Update flow values
            old_intensity = flow.intensity
            flow.intensity = max(0, min(1, result.get('intensity', flow.intensity)))
            flow.momentum = max(-1, min(1, result.get('momentum', flow.momentum)))
            flow.phase_progress += result.get('progress_change', 0.1)
            
            # Check for transition
            transition = None
            if result.get('should_transition') or flow.phase_progress >= 1.0:
                transition = await self._handle_phase_transition(flow, event)
            
            # Update database
            await self._save_flow_state(flow)
            
            return {
                'intensity_change': flow.intensity - old_intensity,
                'new_momentum': flow.momentum,
                'phase_progress': flow.phase_progress,
                'transition': transition,
                'narrative_impact': result.get('narrative_impact', 'The conflict evolves')
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to update flow: {e}")
            return {'intensity_change': 0, 'new_momentum': flow.momentum}
    
    async def generate_dramatic_beat(
        self,
        flow: ConflictFlow,
        context: Dict[str, Any]
    ) -> DramaticBeat:
        """Generate a dramatic beat for the conflict"""
        
        prompt = f"""
        Generate dramatic beat:
        
        Current Phase: {flow.current_phase.value}
        Intensity: {flow.intensity}
        Momentum: {flow.momentum}
        Context: {json.dumps(context, indent=2)}
        
        Create a dramatic moment:
        1. Beat type (revelation/betrayal/escalation/reconciliation/twist)
        2. Specific description (what happens)
        3. Impact on flow (-1 to 1)
        4. Characters involved
        5. Long-term significance
        
        Make it memorable but appropriate to current phase.
        Format as JSON.
        """
        
        response = await self.beat_generator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            beat = DramaticBeat(
                beat_type=result.get('type', 'moment'),
                description=result.get('description', 'A significant moment occurs'),
                impact_on_flow=float(result.get('impact', 0.2)),
                characters_involved=result.get('characters', []),
                timestamp=datetime.now()
            )
            
            # Apply beat impact
            flow.momentum += beat.impact_on_flow * 0.5
            flow.intensity += abs(beat.impact_on_flow) * 0.3
            flow.dramatic_beats.append(beat)
            
            # Store beat
            await self._store_dramatic_beat(flow.conflict_id, beat)
            
            return beat
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to generate beat: {e}")
            return self._create_fallback_beat()
    
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
        
        Create:
        1. Transition type (natural/triggered/forced/stalled/reversed)
        2. Narrative description (2-3 sentences)
        3. Immediate effects
        4. What changes for characters
        5. Environmental/mood shifts
        
        Make the transition feel significant but smooth.
        Format as JSON.
        """
        
        response = await self.transition_narrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            transition = PhaseTransition(
                from_phase=flow.current_phase,
                to_phase=next_phase,
                transition_type=TransitionType[result.get('type', 'NATURAL').upper()],
                trigger=str(trigger_event),
                narrative=result.get('narrative', 'The conflict shifts'),
                timestamp=datetime.now()
            )
            
            # Update flow
            flow.current_phase = next_phase
            flow.phase_progress = 0.0
            flow.transitions_history.append(transition)
            
            # Store transition
            await self._store_transition(flow.conflict_id, transition)
            
            return transition
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to handle transition: {e}")
            return self._create_fallback_transition(flow.current_phase, next_phase)
    
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
        
        # Check for special conditions
        if flow.momentum < -0.5:  # Reversing
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
                WHERE conflict_id = $1
            """, conflict_id)
        
        if flow_data:
            return ConflictFlow(
                conflict_id=conflict_id,
                current_phase=ConflictPhase[flow_data['current_phase'].upper()],
                pacing_style=PacingStyle[flow_data['pacing_style'].upper()],
                intensity=flow_data['intensity'],
                momentum=flow_data['momentum'],
                phase_progress=flow_data.get('phase_progress', 0.5),
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
                SET current_phase = $1, intensity = $2, 
                    momentum = $3, phase_progress = $4
                WHERE conflict_id = $5
            """, flow.current_phase.value, flow.intensity,
            flow.momentum, flow.phase_progress, flow.conflict_id)
    
    async def _store_transition(self, conflict_id: int, transition: PhaseTransition):
        """Store phase transition in database"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO phase_transitions
                (user_id, conversation_id, conflict_id, from_phase, to_phase,
                 transition_type, trigger, narrative)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, self.user_id, self.conversation_id, conflict_id,
            transition.from_phase.value, transition.to_phase.value,
            transition.transition_type.value, transition.trigger,
            transition.narrative)
    
    async def _store_dramatic_beat(self, conflict_id: int, beat: DramaticBeat):
        """Store dramatic beat in database"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO dramatic_beats
                (user_id, conversation_id, conflict_id, beat_type,
                 description, impact)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, self.user_id, self.conversation_id, conflict_id,
            beat.beat_type, beat.description, beat.impact_on_flow)
    
    def _get_phase_distribution(self) -> Dict[str, int]:
        """Get distribution of conflicts across phases"""
        distribution = {}
        for flow in self._flow_states.values():
            phase = flow.current_phase.value
            distribution[phase] = distribution.get(phase, 0) + 1
        return distribution
    
    def _get_average_intensity(self) -> float:
        """Get average intensity across all flows"""
        if not self._flow_states:
            return 0.0
        return sum(f.intensity for f in self._flow_states.values()) / len(self._flow_states)
    
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
