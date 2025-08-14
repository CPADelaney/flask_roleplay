# logic/conflict_system/conflict_flow.py
"""
Conflict Flow System with LLM-generated pacing and transitions.
Manages the rhythm, pacing, and dramatic flow of conflicts.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# FLOW STRUCTURES
# ===============================================================================

class ConflictPhase(Enum):
    """Phases of conflict progression"""
    SEEDS = "seeds"  # Conflict seeds being planted
    EMERGING = "emerging"  # Starting to surface
    RISING = "rising"  # Building tension
    CONFRONTATION = "confrontation"  # Direct conflict
    CLIMAX = "climax"  # Peak intensity
    FALLING = "falling"  # De-escalation
    RESOLUTION = "resolution"  # Settling
    AFTERMATH = "aftermath"  # Long-term effects

class PacingStyle(Enum):
    """Different pacing styles for conflicts"""
    SLOW_BURN = "slow_burn"  # Gradual build over time
    RAPID_ESCALATION = "rapid_escalation"  # Quick to conflict
    WAVES = "waves"  # Cycles of tension and release
    STEADY = "steady"  # Consistent pressure
    ERRATIC = "erratic"  # Unpredictable changes

class TransitionType(Enum):
    """Types of phase transitions"""
    NATURAL = "natural"  # Organic progression
    TRIGGERED = "triggered"  # Event-based
    FORCED = "forced"  # External intervention
    STALLED = "stalled"  # Progress blocked
    REVERSED = "reversed"  # Moving backward

@dataclass
class ConflictFlow:
    """The flow state of a conflict"""
    conflict_id: int
    current_phase: ConflictPhase
    pacing_style: PacingStyle
    intensity: float  # 0-1, current intensity
    momentum: float  # -1 to 1, direction of change
    phase_progress: float  # 0-1, progress in current phase
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
    beat_type: str  # "revelation", "betrayal", "reconciliation", etc.
    description: str
    impact_on_flow: float  # How much it affects momentum
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
# CONFLICT FLOW MANAGER WITH LLM
# ===============================================================================

class ConflictFlowManager:
    """
    Manages conflict pacing and flow using LLM for dynamic generation.
    Controls dramatic rhythm and ensures engaging progression.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._pacing_director = None
        self._transition_narrator = None
        self._beat_generator = None
        self._flow_analyzer = None
    
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
    
    # ========== Flow State Management ==========
    
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
    
    # ========== Phase Transitions ==========
    
    async def _handle_phase_transition(
        self,
        flow: ConflictFlow,
        trigger_event: Dict[str, Any]
    ) -> PhaseTransition:
        """Handle transition between phases"""
        
        # Determine next phase
        next_phase = await self._determine_next_phase(flow, trigger_event)
        
        # Generate transition narrative
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
        
        # Natural progression
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
    
    # ========== Dramatic Beats ==========
    
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
    
    async def check_for_beat_opportunity(
        self,
        flow: ConflictFlow,
        recent_events: List[Dict[str, Any]]
    ) -> bool:
        """Check if it's time for a dramatic beat"""
        
        prompt = f"""
        Analyze beat opportunity:
        
        Current Phase: {flow.current_phase.value}
        Intensity: {flow.intensity}
        Last Beat: {len(flow.dramatic_beats)} beats ago
        Recent Events: {json.dumps(recent_events[-3:] if recent_events else [], indent=2)}
        
        Should we generate a dramatic beat now?
        Consider:
        - Pacing (not too many beats)
        - Build-up (has tension accumulated?)
        - Timing (is it dramatically appropriate?)
        
        Answer: yes/no with reasoning
        Format as JSON: {{"generate_beat": true/false, "reason": "..."}}
        """
        
        response = await self.flow_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            return result.get('generate_beat', False)
        except json.JSONDecodeError:
            # Default: check simple conditions
            return len(flow.dramatic_beats) < 3 and flow.intensity > 0.6
    
    # ========== Pacing Control ==========
    
    async def adjust_pacing(
        self,
        flow: ConflictFlow,
        target_feel: str  # "accelerate", "maintain", "decelerate"
    ) -> Dict[str, Any]:
        """Adjust conflict pacing"""
        
        prompt = f"""
        Adjust conflict pacing:
        
        Current Pacing: {flow.pacing_style.value}
        Current Phase: {flow.current_phase.value}
        Target Feel: {target_feel}
        
        Generate:
        1. Pacing adjustments to make
        2. Events to introduce
        3. Narrative techniques to use
        4. Expected player experience
        
        Keep adjustments subtle and natural.
        Format as JSON.
        """
        
        response = await self.pacing_director.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Apply adjustments
            if target_feel == "accelerate":
                flow.momentum = min(1.0, flow.momentum + 0.3)
                flow.phase_progress += 0.2
            elif target_feel == "decelerate":
                flow.momentum = max(-0.5, flow.momentum - 0.3)
                flow.phase_progress *= 0.8
            
            return {
                'adjustments': result.get('adjustments', []),
                'suggested_events': result.get('events', []),
                'techniques': result.get('techniques', []),
                'expected_experience': result.get('experience', 'Pacing adjusted')
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to adjust pacing: {e}")
            return {'adjustments': [], 'suggested_events': []}
    
    async def analyze_flow_health(
        self,
        flow: ConflictFlow,
        player_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze if conflict flow is healthy"""
        
        prompt = f"""
        Analyze conflict flow health:
        
        Current State:
        - Phase: {flow.current_phase.value}
        - Intensity: {flow.intensity}
        - Momentum: {flow.momentum}
        - Phase Progress: {flow.phase_progress}
        - Beats Generated: {len(flow.dramatic_beats)}
        
        Recent Player Actions: {json.dumps(player_actions[-5:] if player_actions else [], indent=2)}
        
        Assess:
        1. Is pacing appropriate? (rating 0-1)
        2. Player engagement level (0-1)
        3. Risk of stagnation (0-1)
        4. Need for intervention (yes/no)
        5. Recommendations
        
        Be analytical and constructive.
        Format as JSON.
        """
        
        response = await self.flow_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            return {
                'health_score': result.get('pacing_rating', 0.7),
                'engagement': result.get('engagement', 0.5),
                'stagnation_risk': result.get('stagnation_risk', 0.2),
                'needs_intervention': result.get('intervention', False),
                'recommendations': result.get('recommendations', [])
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to analyze flow health: {e}")
            return {'health_score': 0.5, 'recommendations': []}
    
    # ========== Helper Methods ==========
    
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

# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def initialize_conflict_flow(
    ctx: RunContextWrapper,
    conflict_id: int,
    conflict_type: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Initialize flow for a new conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = ConflictFlowManager(user_id, conversation_id)
    
    flow = await manager.initialize_conflict_flow(
        conflict_id,
        conflict_type,
        context or {}
    )
    
    return {
        'conflict_id': conflict_id,
        'initial_phase': flow.current_phase.value,
        'pacing_style': flow.pacing_style.value,
        'intensity': flow.intensity,
        'momentum': flow.momentum,
        'next_conditions': flow.next_transition_conditions
    }

@function_tool
async def progress_conflict_flow(
    ctx: RunContextWrapper,
    conflict_id: int,
    event: Dict[str, Any]
) -> Dict[str, Any]:
    """Progress conflict flow based on an event"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = ConflictFlowManager(user_id, conversation_id)
    
    # Get current flow
    async with get_db_connection_context() as conn:
        flow_data = await conn.fetchrow("""
            SELECT * FROM conflict_flows
            WHERE conflict_id = $1
        """, conflict_id)
    
    if not flow_data:
        return {'error': 'Flow not found'}
    
    # Create flow object
    flow = ConflictFlow(
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
    
    # Update flow
    result = await manager.update_conflict_flow(flow, event)
    
    # Check for beat opportunity
    if await manager.check_for_beat_opportunity(flow, [event]):
        beat = await manager.generate_dramatic_beat(flow, event)
        result['dramatic_beat'] = {
            'type': beat.beat_type,
            'description': beat.description,
            'impact': beat.impact_on_flow
        }
    
    # Update database
    async with get_db_connection_context() as conn:
        await conn.execute("""
            UPDATE conflict_flows
            SET current_phase = $1, intensity = $2, momentum = $3,
                phase_progress = $4
            WHERE conflict_id = $5
        """, flow.current_phase.value, flow.intensity, flow.momentum,
        flow.phase_progress, conflict_id)
    
    return result

@function_tool
async def generate_conflict_beat(
    ctx: RunContextWrapper,
    conflict_id: int,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate a dramatic beat for a conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = ConflictFlowManager(user_id, conversation_id)
    
    # Get flow
    async with get_db_connection_context() as conn:
        flow_data = await conn.fetchrow("""
            SELECT * FROM conflict_flows WHERE conflict_id = $1
        """, conflict_id)
    
    if not flow_data:
        return {'error': 'Flow not found'}
    
    flow = ConflictFlow(
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
    
    beat = await manager.generate_dramatic_beat(flow, context or {})
    
    return {
        'beat_type': beat.beat_type,
        'description': beat.description,
        'impact': beat.impact_on_flow,
        'new_intensity': flow.intensity,
        'new_momentum': flow.momentum
    }

@function_tool
async def adjust_conflict_pacing(
    ctx: RunContextWrapper,
    conflict_id: int,
    target_feel: str  # "accelerate", "maintain", "decelerate"
) -> Dict[str, Any]:
    """Adjust the pacing of a conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = ConflictFlowManager(user_id, conversation_id)
    
    # Get flow
    async with get_db_connection_context() as conn:
        flow_data = await conn.fetchrow("""
            SELECT * FROM conflict_flows WHERE conflict_id = $1
        """, conflict_id)
    
    if not flow_data:
        return {'error': 'Flow not found'}
    
    flow = ConflictFlow(
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
    
    result = await manager.adjust_pacing(flow, target_feel)
    
    # Update database
    async with get_db_connection_context() as conn:
        await conn.execute("""
            UPDATE conflict_flows
            SET momentum = $1, phase_progress = $2
            WHERE conflict_id = $3
        """, flow.momentum, flow.phase_progress, conflict_id)
    
    return result

@function_tool
async def analyze_conflict_flow_health(
    ctx: RunContextWrapper,
    conflict_id: int,
    recent_player_actions: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Analyze the health of conflict flow"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = ConflictFlowManager(user_id, conversation_id)
    
    # Get flow
    async with get_db_connection_context() as conn:
        flow_data = await conn.fetchrow("""
            SELECT * FROM conflict_flows WHERE conflict_id = $1
        """, conflict_id)
    
    if not flow_data:
        return {'error': 'Flow not found'}
    
    flow = ConflictFlow(
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
    
    return await manager.analyze_flow_health(flow, recent_player_actions or [])
