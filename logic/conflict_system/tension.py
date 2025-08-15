# logic/conflict_system/tension.py
"""
Dynamic Tension System with LLM-generated content.
Refactored to work as a subsystem under the Conflict Synthesizer orchestrator.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
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
# TENSION TYPES AND STRUCTURES (Preserved)
# ===============================================================================

class TensionObservation(TypedDict):
    type: str
    level: float
    source: str
    note: str

class AnalyzeSceneTensionsResponse(TypedDict):
    tension_score: float
    should_generate_conflict: bool
    primary_dynamic: str
    observations: List[TensionObservation]
    error: str

class ModifyTensionResponse(TypedDict):
    success: bool
    tension_type: str
    applied_change: float
    new_level: float
    clamped: bool
    reason: str
    side_effects: List[str]
    error: str

class TensionCategory(TypedDict):
    name: str
    level: float

class TensionReportResponse(TypedDict):
    total_categories: int
    categories: List[TensionCategory]
    overall_score: float
    hotspots: List[str]
    last_updated_iso: str
    error: str



class TensionType(Enum):
    """Different types of tension in the game"""
    POWER = "power"
    SOCIAL = "social"
    SEXUAL = "sexual"
    EMOTIONAL = "emotional"
    ADDICTION = "addiction"
    VITAL = "vital"
    ECONOMIC = "economic"
    IDEOLOGICAL = "ideological"
    TERRITORIAL = "territorial"

class TensionLevel(Enum):
    """Levels of tension intensity"""
    ABSENT = 0.0
    SUBTLE = 0.2
    NOTICEABLE = 0.4
    PALPABLE = 0.6
    INTENSE = 0.8
    BREAKING = 1.0

@dataclass
class TensionSource:
    """A source contributing to tension"""
    source_type: str
    source_id: Any
    contribution: float
    description: str

@dataclass
class TensionManifestation:
    """How tension manifests in a scene"""
    tension_type: TensionType
    level: float
    physical_cues: List[str]
    dialogue_modifications: List[str]
    environmental_changes: List[str]
    player_sensations: List[str]

# ===============================================================================
# REFACTORED TENSION SYSTEM
# ===============================================================================

class TensionSystem(ConflictSubsystem):
    """
    Manages tension dynamics as a subsystem under the orchestrator.
    Now implements ConflictSubsystem interface for proper integration.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._synthesizer = None  # Will be set by orchestrator
        
        # Tension state
        self._current_tensions: Dict[TensionType, float] = {}
        self._tension_sources: List[TensionSource] = []
        
        # LLM agents (unchanged)
        self._tension_analyzer = None
        self._manifestation_generator = None
        self._escalation_narrator = None
    
    # ========== ConflictSubsystem Interface Implementation ==========
    
    @property
    def subsystem_type(self) -> SubsystemType:
        """Identify as tension subsystem"""
        return SubsystemType.TENSION
    
    @property
    def capabilities(self) -> Set[str]:
        """Capabilities this subsystem provides"""
        return {
            'calculate_tensions',
            'build_tension',
            'resolve_tension',
            'generate_manifestation',
            'check_breaking_point',
            'analyze_tension_sources',
            'create_tension_narrative'
        }
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        """Other subsystems we depend on"""
        return {
            SubsystemType.STAKEHOLDER,  # For NPC emotional states
            SubsystemType.FLOW,  # For conflict phase context
            SubsystemType.SOCIAL,  # For social tensions
        }
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        """Events we want to receive from orchestrator"""
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.STAKEHOLDER_ACTION,
            EventType.PHASE_TRANSITION,
            EventType.PLAYER_CHOICE,
            EventType.NPC_REACTION,
            EventType.STATE_SYNC,
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize with synthesizer reference"""
        self._synthesizer = weakref.ref(synthesizer)
        
        # Load current tension state from DB
        await self._load_tension_state()
        
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Handle events from the orchestrator"""
        
        try:
            # Route to appropriate handler
            handlers = {
                EventType.CONFLICT_CREATED: self._on_conflict_created,
                EventType.STAKEHOLDER_ACTION: self._on_stakeholder_action,
                EventType.PHASE_TRANSITION: self._on_phase_transition,
                EventType.PLAYER_CHOICE: self._on_player_choice,
                EventType.NPC_REACTION: self._on_npc_reaction,
                EventType.STATE_SYNC: self._on_state_sync,
                EventType.HEALTH_CHECK: self._on_health_check
            }
            
            handler = handlers.get(event.event_type)
            if handler:
                return await handler(event)
            
            # Default response for unhandled events
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'status': 'no_action_taken'}
            )
            
        except Exception as e:
            logger.error(f"Tension system error handling event: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status"""
        
        total_tension = sum(self._current_tensions.values())
        
        return {
            'healthy': total_tension < 5.0,  # Too much tension is unhealthy
            'active_tensions': len([t for t, v in self._current_tensions.items() if v > 0.1]),
            'total_tension': total_tension,
            'critical_tensions': [t.value for t, v in self._current_tensions.items() if v > 0.8],
            'status': 'operational'
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get tension data for a specific conflict"""
        
        # Return current tensions relevant to this conflict
        return {
            'tensions': {t.value: v for t, v in self._current_tensions.items()},
            'dominant_tension': max(self._current_tensions.items(), key=lambda x: x[1])[0].value if self._current_tensions else None,
            'total_tension': sum(self._current_tensions.values())
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current subsystem state"""
        
        return {
            'current_tensions': {t.value: v for t, v in self._current_tensions.items()},
            'active_sources': len(self._tension_sources),
            'breaking_points': [t.value for t, v in self._current_tensions.items() if v >= 0.9]
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if tension system should process this scene"""
        
        # Always relevant if there are active tensions
        if any(v > 0.1 for v in self._current_tensions.values()):
            return True
        
        # Check for tension-inducing elements in scene
        activity = scene_context.get('activity', '').lower()
        tension_activities = ['argument', 'confrontation', 'negotiation', 'intimate']
        
        return any(word in activity for word in tension_activities)
    
    # ========== Event Handlers ==========
    
    async def _on_conflict_created(self, event: SystemEvent) -> SubsystemResponse:
        """Handle new conflict creation"""
        
        conflict_type = event.payload.get('conflict_type')
        context = event.payload.get('context', {})
        
        # Determine initial tensions
        initial_tensions = await self._determine_initial_tensions(conflict_type, context)
        
        # Apply tensions
        side_effects = []
        for tension_type, level in initial_tensions.items():
            self._current_tensions[tension_type] = level
            
            # Notify orchestrator of significant tensions
            if level > 0.3:
                side_effects.append(SystemEvent(
                    event_id=f"tension_{tension_type.value}_{event.event_id}",
                    event_type=EventType.TENSION_CHANGED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'tension_type': tension_type.value,
                        'level': level,
                        'source': 'conflict_creation'
                    },
                    priority=5
                ))
        
        # Store in database
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_initialized': {t.value: l for t, l in initial_tensions.items()},
                'dominant_tension': max(initial_tensions.items(), key=lambda x: x[1])[0].value if initial_tensions else None
            },
            side_effects=side_effects
        )
    
    async def _on_stakeholder_action(self, event: SystemEvent) -> SubsystemResponse:
        """Handle stakeholder actions affecting tension"""
        
        action_type = event.payload.get('action_type')
        intensity = event.payload.get('intensity', 0.5)
        
        # Map action to tension changes
        tension_changes = self._map_action_to_tension_changes(action_type, intensity)
        
        # Apply changes
        side_effects = []
        for tension_type, change in tension_changes.items():
            old_level = self._current_tensions.get(tension_type, 0)
            new_level = max(0, min(1.0, old_level + change))
            self._current_tensions[tension_type] = new_level
            
            # Check for breaking point
            if new_level >= TensionLevel.BREAKING.value and old_level < TensionLevel.BREAKING.value:
                side_effects.append(SystemEvent(
                    event_id=f"breaking_{tension_type.value}_{event.event_id}",
                    event_type=EventType.EDGE_CASE_DETECTED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'edge_case': 'tension_breaking_point',
                        'tension_type': tension_type.value,
                        'level': new_level,
                        'requires_immediate_resolution': True
                    },
                    priority=1
                ))
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_modified': {t.value: c for t, c in tension_changes.items()},
                'current_tensions': {t.value: v for t, v in self._current_tensions.items()}
            },
            side_effects=side_effects
        )
    
    async def _on_phase_transition(self, event: SystemEvent) -> SubsystemResponse:
        """Handle conflict phase transitions"""
        
        from_phase = event.payload.get('from_phase')
        to_phase = event.payload.get('to_phase')
        
        # Adjust tensions based on phase
        adjustments = self._calculate_phase_tension_adjustments(from_phase, to_phase)
        
        for tension_type, adjustment in adjustments.items():
            current = self._current_tensions.get(tension_type, 0)
            self._current_tensions[tension_type] = max(0, min(1.0, current + adjustment))
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_adjusted': {t.value: a for t, a in adjustments.items()},
                'phase_impact': 'tensions_shifted'
            }
        )
    
    async def _on_player_choice(self, event: SystemEvent) -> SubsystemResponse:
        """Handle player choices affecting tension"""
        
        choice_id = event.payload.get('choice_id')
        choice_type = event.payload.get('choice_type')
        
        # Generate tension narrative for the choice
        narrative = await self._generate_choice_tension_narrative(choice_type)
        
        # Determine tension impact
        impact = self._calculate_choice_tension_impact(choice_type)
        
        # Apply impact
        for tension_type, change in impact.items():
            current = self._current_tensions.get(tension_type, 0)
            self._current_tensions[tension_type] = max(0, min(1.0, current + change))
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tension_narrative': narrative,
                'tension_changes': {t.value: c for t, c in impact.items()},
                'player_impact': 'acknowledged'
            }
        )
    
    async def _on_npc_reaction(self, event: SystemEvent) -> SubsystemResponse:
        """Handle NPC reactions affecting tension"""
        
        npc_id = event.payload.get('npc_id')
        reaction_type = event.payload.get('reaction_type')
        emotional_state = event.payload.get('emotional_state')
        
        # NPCs in distress increase emotional tension
        if emotional_state in ['angry', 'distressed', 'fearful']:
            self._current_tensions[TensionType.EMOTIONAL] = min(
                1.0,
                self._current_tensions.get(TensionType.EMOTIONAL, 0) + 0.1
            )
        
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'tension_adjusted': True}
        )
    
    async def _on_state_sync(self, event: SystemEvent) -> SubsystemResponse:
        """Sync state with scene processing"""
        
        scene_context = event.payload
        
        # Generate tension manifestation for the scene
        manifestation = await self.generate_tension_manifestation(scene_context)
        
        # Check for breaking points
        breaking_point = await self.check_tension_breaking_point()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'manifestation': {
                    'type': manifestation.tension_type.value,
                    'level': manifestation.level,
                    'physical_cues': manifestation.physical_cues,
                    'dialogue_mods': manifestation.dialogue_modifications,
                    'environment': manifestation.environmental_changes,
                    'sensations': manifestation.player_sensations
                },
                'breaking_point': breaking_point,
                'current_tensions': {t.value: v for t, v in self._current_tensions.items()}
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
    
    # ========== Core Tension Management (Modified for Orchestrator) ==========
    
    async def calculate_current_tensions(self) -> Dict[TensionType, float]:
        """Calculate current tension levels from all sources"""
        
        # This is now primarily called internally or by orchestrator
        tensions = {t: 0.0 for t in TensionType}
        
        # Get tension sources
        sources = await self._gather_tension_sources()
        
        # Use LLM to analyze
        prompt = f"""
        Analyze these tension sources and calculate overall tension levels:
        
        Sources:
        {json.dumps([self._source_to_dict(s) for s in sources], indent=2)}
        
        For each tension type, calculate a level from 0.0 to 1.0.
        Format as JSON: {{"power": 0.X, "social": 0.X, ...}}
        """
        
        response = await self.tension_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            for tension_type in TensionType:
                if tension_type.value in result:
                    tensions[tension_type] = min(1.0, max(0.0, float(result[tension_type.value])))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse tension levels: {e}")
            # Fallback calculation
            for source in sources:
                tensions[self._map_source_to_tension_type(source)] += source.contribution * 0.2
        
        self._current_tensions = tensions
        await self._save_tension_state()
        
        # Notify orchestrator of significant changes
        if self._synthesizer and self._synthesizer():
            for t, level in tensions.items():
                if level > 0.7:
                    await self._synthesizer().emit_event(SystemEvent(
                        event_id=f"high_tension_{t.value}",
                        event_type=EventType.TENSION_CHANGED,
                        source_subsystem=self.subsystem_type,
                        payload={'tension_type': t.value, 'level': level}
                    ))
        
        return tensions
    
    async def build_tension(
        self,
        tension_type: TensionType,
        amount: float,
        source: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build tension with contextual narrative"""
        
        current_level = self._current_tensions.get(tension_type, 0)
        new_level = min(1.0, current_level + amount)
        self._current_tensions[tension_type] = new_level
        
        # Generate narrative
        prompt = f"""
        Narrate tension building:
        
        Type: {tension_type.value}
        Current Level: {current_level:.2f} → {new_level:.2f}
        Source: {source}
        
        Create a brief description (1-2 sentences) and 2-3 subtle cues.
        Format as JSON.
        """
        
        response = await self.escalation_narrator.run(prompt)
        
        try:
            narrative = json.loads(response.content)
        except json.JSONDecodeError:
            narrative = {
                'description': f"The {tension_type.value} tension grows",
                'cues': ['A subtle shift in atmosphere']
            }
        
        await self._save_tension_state()
        
        # Notify orchestrator if threshold crossed
        threshold = self._check_threshold_crossed(current_level, new_level)
        if threshold and self._synthesizer and self._synthesizer():
            await self._synthesizer().emit_event(SystemEvent(
                event_id=f"threshold_{tension_type.value}_{threshold}",
                event_type=EventType.TENSION_CHANGED,
                source_subsystem=self.subsystem_type,
                payload={
                    'tension_type': tension_type.value,
                    'threshold': threshold,
                    'new_level': new_level
                }
            ))
        
        return {
            'tension_type': tension_type.value,
            'old_level': current_level,
            'new_level': new_level,
            'narrative': narrative,
            'threshold_crossed': threshold
        }
    
    async def resolve_tension(
        self,
        tension_type: TensionType,
        amount: float,
        resolution_type: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Resolve tension with contextual narrative"""
        
        current_level = self._current_tensions.get(tension_type, 0)
        new_level = max(0.0, current_level - amount)
        self._current_tensions[tension_type] = new_level
        
        # Generate resolution narrative
        prompt = f"""
        Narrate tension resolution:
        
        Type: {tension_type.value}
        Current Level: {current_level:.2f} → {new_level:.2f}
        Resolution Type: {resolution_type}
        
        Create a release description and aftermath mood.
        Format as JSON.
        """
        
        response = await self.escalation_narrator.run(prompt)
        
        try:
            narrative = json.loads(response.content)
        except json.JSONDecodeError:
            narrative = {
                'release': f"The {tension_type.value} tension eases",
                'aftermath': 'calm'
            }
        
        await self._save_tension_state()
        
        return {
            'tension_type': tension_type.value,
            'old_level': current_level,
            'new_level': new_level,
            'narrative': narrative,
            'fully_resolved': new_level < 0.1
        }
    
    async def generate_tension_manifestation(
        self,
        scene_context: Dict[str, Any]
    ) -> TensionManifestation:
        """Generate how current tensions manifest in a scene"""
        
        # Find dominant tension
        if not self._current_tensions:
            return self._create_no_tension_manifestation()
        
        dominant_type, dominant_level = max(
            self._current_tensions.items(),
            key=lambda x: x[1]
        )
        
        if dominant_level < 0.1:
            return self._create_no_tension_manifestation()
        
        # Generate manifestation with LLM
        prompt = f"""
        Generate tension manifestations:
        
        Dominant Tension: {dominant_type.value} ({dominant_level:.2f})
        Scene: {json.dumps(scene_context, indent=2)}
        
        Create specific sensory details:
        - 3-4 physical cues
        - 2-3 dialogue modifications
        - 2-3 environmental changes
        - 2-3 player sensations
        
        Format as JSON with arrays.
        """
        
        response = await self.manifestation_generator.run(prompt)
        
        try:
            result = json.loads(response.content)
            return TensionManifestation(
                tension_type=dominant_type,
                level=dominant_level,
                physical_cues=result.get('physical_cues', []),
                dialogue_modifications=result.get('dialogue_modifications', []),
                environmental_changes=result.get('environmental_changes', []),
                player_sensations=result.get('player_sensations', [])
            )
        except json.JSONDecodeError:
            return self._create_fallback_manifestation(dominant_type, dominant_level)
    
    async def check_tension_breaking_point(self) -> Optional[Dict[str, Any]]:
        """Check if any tension has reached breaking point"""
        
        breaking_tensions = {
            t: level for t, level in self._current_tensions.items()
            if level >= TensionLevel.BREAKING.value
        }
        
        if not breaking_tensions:
            return None
        
        breaking_type = max(breaking_tensions.items(), key=lambda x: x[1])[0]
        
        prompt = f"""
        A tension has reached breaking point:
        
        Breaking Tension: {breaking_type.value}
        Level: {breaking_tensions[breaking_type]:.2f}
        
        Generate the triggering moment and consequences.
        Format as JSON.
        """
        
        response = await self.escalation_narrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            return {
                'breaking_tension': breaking_type.value,
                'trigger': result.get('trigger', 'The tension snaps'),
                'consequences': result.get('consequences', []),
                'player_choices': result.get('choices', [])
            }
        except json.JSONDecodeError:
            return {
                'breaking_tension': breaking_type.value,
                'trigger': 'The tension reaches a breaking point',
                'consequences': ['Things cannot continue as they were']
            }
    
    # ========== Helper Methods (Modified) ==========
    
    async def _load_tension_state(self):
        """Load tension state from database"""
        
        async with get_db_connection_context() as conn:
            tensions = await conn.fetch("""
                SELECT tension_type, level FROM TensionLevels
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        
        for tension in tensions:
            try:
                self._current_tensions[TensionType(tension['tension_type'])] = tension['level']
            except ValueError:
                pass
    
    async def _save_tension_state(self):
        """Save tension state to database"""
        
        async with get_db_connection_context() as conn:
            for tension_type, level in self._current_tensions.items():
                await conn.execute("""
                    INSERT INTO TensionLevels (user_id, conversation_id, tension_type, level)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, conversation_id, tension_type)
                    DO UPDATE SET level = $4, updated_at = NOW()
                """, self.user_id, self.conversation_id, tension_type.value, level)
    
    async def _determine_initial_tensions(
        self,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Dict[TensionType, float]:
        """Determine initial tensions for a conflict"""
        
        # Map conflict types to tension patterns
        patterns = {
            'power': {TensionType.POWER: 0.6, TensionType.SOCIAL: 0.3},
            'social': {TensionType.SOCIAL: 0.7, TensionType.EMOTIONAL: 0.4},
            'romantic': {TensionType.SEXUAL: 0.5, TensionType.EMOTIONAL: 0.5},
            'economic': {TensionType.ECONOMIC: 0.8, TensionType.POWER: 0.3},
            'ideological': {TensionType.IDEOLOGICAL: 0.7, TensionType.SOCIAL: 0.4}
        }
        
        # Find matching pattern
        for key, pattern in patterns.items():
            if key in conflict_type.lower():
                return pattern
        
        # Default pattern
        return {TensionType.EMOTIONAL: 0.4, TensionType.SOCIAL: 0.3}
    
    def _map_action_to_tension_changes(
        self,
        action_type: str,
        intensity: float
    ) -> Dict[TensionType, float]:
        """Map stakeholder actions to tension changes"""
        
        changes = {}
        
        if 'aggressive' in action_type.lower():
            changes[TensionType.POWER] = 0.2 * intensity
            changes[TensionType.EMOTIONAL] = 0.1 * intensity
        elif 'diplomatic' in action_type.lower():
            changes[TensionType.SOCIAL] = -0.1 * intensity
            changes[TensionType.POWER] = -0.05 * intensity
        elif 'manipulative' in action_type.lower():
            changes[TensionType.SOCIAL] = 0.15 * intensity
            changes[TensionType.EMOTIONAL] = 0.1 * intensity
        
        return changes
    
    def _calculate_phase_tension_adjustments(
        self,
        from_phase: str,
        to_phase: str
    ) -> Dict[TensionType, float]:
        """Calculate tension adjustments for phase transitions"""
        
        adjustments = {}
        
        if to_phase == 'climax':
            # Intensify all tensions
            for t in TensionType:
                adjustments[t] = 0.2
        elif to_phase == 'resolution':
            # Reduce most tensions
            for t in TensionType:
                adjustments[t] = -0.3
        elif from_phase == 'emerging' and to_phase == 'rising':
            # Gradual increase
            adjustments[TensionType.EMOTIONAL] = 0.1
            adjustments[TensionType.SOCIAL] = 0.1
        
        return adjustments
    
    async def _generate_choice_tension_narrative(self, choice_type: str) -> str:
        """Generate narrative for player choice impact on tension"""
        
        prompt = f"""
        Generate a brief narrative for how this choice affects tension:
        Choice Type: {choice_type}
        
        One sentence showing the subtle tension shift.
        """
        
        response = await self.escalation_narrator.run(prompt)
        return response.content.strip()
    
    def _calculate_choice_tension_impact(self, choice_type: str) -> Dict[TensionType, float]:
        """Calculate how player choice affects tensions"""
        
        impacts = {}
        
        if 'submit' in choice_type.lower():
            impacts[TensionType.POWER] = -0.1
            impacts[TensionType.EMOTIONAL] = 0.05
        elif 'resist' in choice_type.lower():
            impacts[TensionType.POWER] = 0.15
            impacts[TensionType.SOCIAL] = 0.1
        elif 'negotiate' in choice_type.lower():
            impacts[TensionType.SOCIAL] = -0.05
        
        return impacts
    
    # ========== Preserved Helper Methods ==========
    
    async def _gather_tension_sources(self) -> List[TensionSource]:
        """Gather all current sources of tension"""
        
        sources = []
        
        async with get_db_connection_context() as conn:
            # Get conflict tensions
            conflicts = await conn.fetch("""
                SELECT conflict_id, conflict_type, intensity, progress
                FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
            
            for conflict in conflicts:
                sources.append(TensionSource(
                    source_type="conflict",
                    source_id=conflict['conflict_id'],
                    contribution=conflict['progress'] / 100 * 0.5,
                    description=f"{conflict['conflict_type']} conflict"
                ))
        
        return sources
    
    def _source_to_dict(self, source: TensionSource) -> Dict:
        """Convert TensionSource to dict for JSON"""
        return {
            'type': source.source_type,
            'id': str(source.source_id),
            'contribution': source.contribution,
            'description': source.description
        }
    
    def _map_source_to_tension_type(self, source: TensionSource) -> TensionType:
        """Map a source to its primary tension type"""
        
        mappings = {
            'conflict': TensionType.POWER,
            'npc': TensionType.EMOTIONAL,
            'environment': TensionType.VITAL,
            'activity': TensionType.SOCIAL
        }
        return mappings.get(source.source_type, TensionType.EMOTIONAL)
    
    def _check_threshold_crossed(self, old_level: float, new_level: float) -> Optional[str]:
        """Check if a significant threshold was crossed"""
        
        thresholds = [
            (0.2, "subtle"),
            (0.4, "noticeable"),
            (0.6, "palpable"),
            (0.8, "intense"),
            (1.0, "breaking")
        ]
        
        for threshold, name in thresholds:
            if old_level < threshold <= new_level:
                return f"entered_{name}"
            elif new_level < threshold <= old_level:
                return f"left_{name}"
        
        return None
    
    def _create_no_tension_manifestation(self) -> TensionManifestation:
        """Create manifestation for no/minimal tension"""
        
        return TensionManifestation(
            tension_type=TensionType.EMOTIONAL,
            level=0.0,
            physical_cues=["Relaxed postures"],
            dialogue_modifications=["Natural speech"],
            environmental_changes=["Comfortable atmosphere"],
            player_sensations=["A sense of ease"]
        )
    
    def _create_fallback_manifestation(
        self,
        tension_type: TensionType,
        level: float
    ) -> TensionManifestation:
        """Create fallback manifestation if LLM fails"""
        
        return TensionManifestation(
            tension_type=tension_type,
            level=level,
            physical_cues=[f"Subtle {tension_type.value} tension"],
            dialogue_modifications=["Careful words"],
            environmental_changes=["Charged atmosphere"],
            player_sensations=["Underlying tension"]
        )
    
    # ========== LLM Agent Properties (Preserved) ==========
    
    @property
    def tension_analyzer(self) -> Agent:
        """Agent for analyzing tension sources and levels"""
        if self._tension_analyzer is None:
            self._tension_analyzer = Agent(
                name="Tension Analyzer",
                instructions="""
                Analyze various sources to determine tension levels and types.
                
                Consider relationship dynamics, conflicts, environmental factors,
                and player choices. Identify subtle tensions that build gradually.
                """,
                model="gpt-5-nano",
            )
        return self._tension_analyzer
    
    @property
    def manifestation_generator(self) -> Agent:
        """Agent for generating tension manifestations"""
        if self._manifestation_generator is None:
            self._manifestation_generator = Agent(
                name="Tension Manifestation Generator",
                instructions="""
                Generate specific, sensory manifestations of tension.
                
                Create physical cues, dialogue modifications, environmental changes,
                and player sensations. Make manifestations subtle and contextual.
                """,
                model="gpt-5-nano",
            )
        return self._manifestation_generator
    
    @property
    def escalation_narrator(self) -> Agent:
        """Agent for narrating tension escalation"""
        if self._escalation_narrator is None:
            self._escalation_narrator = Agent(
                name="Tension Escalation Narrator",
                instructions="""
                Narrate how tensions build, peak, and release.
                
                Focus on gradual accumulation, tipping points, and atmospheric descriptions.
                Use sensory details and emotional weight.
                """,
                model="gpt-5-nano",
            )
        return self._escalation_narrator


# ===============================================================================
# PUBLIC API - Now Routes Through Orchestrator
# ===============================================================================

@function_tool
async def analyze_scene_tensions(
    ctx: RunContextWrapper,
    scene_description: str,
    npcs_present: List[int],
    current_activity: str
) -> AnalyzeSceneTensionsResponse:
    """Analyze tensions in current scene - routes through orchestrator (strict schema)."""

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    synthesizer = await get_synthesizer(user_id, conversation_id)

    # Ask the TENSION subsystem directly for an analysis
    event = SystemEvent(
        event_id=f"tension_analyze_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={
            'scene_description': scene_description,
            'npcs': npcs_present,          # for subsystems expecting 'npcs'
            'present_npcs': npcs_present,  # for subsystems expecting 'present_npcs'
            'activity': current_activity,
            'request': 'tension_analysis'
        },
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )

    responses = await synthesizer.emit_event(event) or []
    data = {}
    for r in responses:
        if r.subsystem == SubsystemType.TENSION:
            data = r.data or {}
            break

    # Normalize
    score = float(data.get('tension_score', 0.0))
    score = 0.0 if score < 0 else (1.0 if score > 1 else score)

    raw_obs = data.get('tensions', []) or []
    observations: List[TensionObservation] = []
    for t in raw_obs[:20]:  # keep it bounded
        level = float(t.get('level', 0.0))
        level = 0.0 if level < 0 else (1.0 if level > 1 else level)
        observations.append({
            'type': str(t.get('type', 'ambient')),
            'level': level,
            'source': str(t.get('source', 'unknown')),
            'note': str(t.get('notes', t.get('note', ''))),
        })

    return {
        'tension_score': score,
        'should_generate_conflict': bool(data.get('should_generate_conflict', False)),
        'primary_dynamic': str(data.get('primary_dynamic', 'none')),
        'observations': observations,
        'error': "" if data else "No response from tension system",
    }


@function_tool
async def modify_tension(
    ctx: RunContextWrapper,
    tension_type: str,
    change: float,
    reason: str
) -> ModifyTensionResponse:
    """Modify a specific tension level - routes through orchestrator (strict schema)."""

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    synthesizer = await get_synthesizer(user_id, conversation_id)

    event = SystemEvent(
        event_id=f"manual_tension_{tension_type}",
        event_type=EventType.TENSION_CHANGED,
        source_subsystem=SubsystemType.TENSION,
        payload={
            'tension_type': tension_type,
            'change': float(change),
            'reason': reason
        },
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )

    responses = await synthesizer.emit_event(event) or []
    data = responses[0].data if responses else {}

    new_level = float(data.get('new_level', 0.0))
    clamped = bool(data.get('clamped', (new_level < 0 or new_level > 1)))
    if new_level < 0: new_level = 0.0
    if new_level > 1: new_level = 1.0

    side_effects = data.get('side_effects', [])
    if not isinstance(side_effects, list):
        side_effects = [str(side_effects)]

    return {
        'success': bool(data.get('success', bool(responses))),
        'tension_type': str(tension_type),
        'applied_change': float(data.get('applied_change', change)),
        'new_level': new_level,
        'clamped': clamped,
        'reason': str(reason),
        'side_effects': [str(s) for s in side_effects[:20]],
        'error': "" if responses else "No response from tension system",
    }


@function_tool
async def get_tension_report(
    ctx: RunContextWrapper
) -> TensionReportResponse:
    """Get comprehensive tension report - routes through orchestrator (strict schema)."""

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    synthesizer = await get_synthesizer(user_id, conversation_id)

    # Ask TENSION subsystem for a structured report
    event = SystemEvent(
        event_id=f"tension_report_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={'request': 'tension_report'},
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )

    responses = await synthesizer.emit_event(event) or []
    data = {}
    for r in responses:
        if r.subsystem == SubsystemType.TENSION:
            data = r.data or {}
            break

    # Normalize
    categories_raw = data.get('categories', data.get('tensions', {}))
    categories: List[TensionCategory] = []
    if isinstance(categories_raw, dict):
        # map object -> list of {name, level}
        for name, lvl in list(categories_raw.items())[:50]:
            val = float(lvl)
            if val < 0: val = 0.0
            if val > 1: val = 1.0
            categories.append({'name': str(name), 'level': val})
    elif isinstance(categories_raw, list):
        for item in categories_raw[:50]:
            name = str(item.get('name', item.get('type', 'unknown')))
            val = float(item.get('level', 0.0))
            if val < 0: val = 0.0
            if val > 1: val = 1.0
            categories.append({'name': name, 'level': val})

    overall = float(data.get('overall_score', data.get('tension_score', 0.0)))
    if overall < 0: overall = 0.0
    if overall > 1: overall = 1.0

    hotspots_raw = data.get('hotspots', [])
    if not isinstance(hotspots_raw, list):
        hotspots_raw = [str(hotspots_raw)]
    hotspots = [str(h) for h in hotspots_raw[:20]]

    last_updated = data.get('last_updated') or datetime.now().isoformat()

    return {
        'total_categories': len(categories),
        'categories': categories,
        'overall_score': overall,
        'hotspots': hotspots,
        'last_updated_iso': str(last_updated),
        'error': "" if data else "No response from tension system",
    }
