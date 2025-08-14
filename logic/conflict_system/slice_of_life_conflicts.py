# logic/conflict_system/slice_of_life_conflicts.py
"""
Slice-of-life conflict system with LLM-generated dynamic content.
Refactored to work as a ConflictSubsystem with the synthesizer.
"""

import logging
import json
import random
import weakref
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from logic.conflict_system.conflict_synthesizer import (
    ConflictSubsystem, SubsystemType, EventType, 
    SystemEvent, SubsystemResponse
)

logger = logging.getLogger(__name__)

# ===============================================================================
# ENUMS (Preserved from original)
# ===============================================================================

class SliceOfLifeConflictType(Enum):
    """Types of subtle daily conflicts"""
    PERMISSION_PATTERNS = "permission_patterns"
    ROUTINE_DOMINANCE = "routine_dominance"
    BOUNDARY_EROSION = "boundary_erosion"
    SOCIAL_PECKING_ORDER = "social_pecking_order"
    FRIENDSHIP_BOUNDARIES = "friendship_boundaries"
    ROLE_EXPECTATIONS = "role_expectations"
    FINANCIAL_CONTROL = "financial_control"
    SUBTLE_RIVALRY = "subtle_rivalry"
    PREFERENCE_SUBMISSION = "preference_submission"
    CARE_DEPENDENCY = "care_dependency"
    INDEPENDENCE_STRUGGLE = "independence_struggle"
    PASSIVE_AGGRESSION = "passive_aggression"
    EMOTIONAL_LABOR = "emotional_labor"
    DECISION_FATIGUE = "decision_fatigue"
    MASK_SLIPPAGE = "mask_slippage"
    DOMESTIC_HIERARCHY = "domestic_hierarchy"
    GROOMING_PATTERNS = "grooming_patterns"
    GASLIGHTING_GENTLE = "gaslighting_gentle"
    SOCIAL_ISOLATION = "social_isolation"
    CONDITIONING_RESISTANCE = "conditioning_resistance"

class ConflictIntensity(Enum):
    """How overtly the conflict manifests"""
    SUBTEXT = "subtext"
    TENSION = "tension"
    PASSIVE = "passive"
    DIRECT = "direct"
    CONFRONTATION = "confrontation"

class ResolutionApproach(Enum):
    """How conflicts resolve in slice-of-life"""
    GRADUAL_ACCEPTANCE = "gradual_acceptance"
    SUBTLE_RESISTANCE = "subtle_resistance"
    NEGOTIATED_COMPROMISE = "negotiated_compromise"
    ESTABLISHED_PATTERN = "established_pattern"
    THIRD_PARTY_INFLUENCE = "third_party_influence"
    TIME_EROSION = "time_erosion"

@dataclass
class SliceOfLifeStake:
    """What's at stake in mundane conflicts"""
    stake_type: str
    description: str
    daily_impact: str
    relationship_impact: str
    accumulation_factor: float

@dataclass
class DailyConflictEvent:
    """A conflict moment embedded in daily routine"""
    activity_type: str
    conflict_manifestation: str
    choice_presented: bool
    accumulation_impact: float
    npc_reactions: Dict[int, str]

# ===============================================================================
# SLICE OF LIFE CONFLICT SUBSYSTEM
# ===============================================================================

class SliceOfLifeConflictSubsystem(ConflictSubsystem):
    """
    Slice-of-life conflict subsystem integrated with synthesizer.
    Combines all slice-of-life components into one subsystem.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None  # Will be set by synthesizer
        
        # Components
        self.detector = EmergentConflictDetector(user_id, conversation_id)
        self.manager = SliceOfLifeConflictManager(user_id, conversation_id)
        self.resolver = PatternBasedResolution(user_id, conversation_id)
        self.daily_integration = ConflictDailyIntegration(user_id, conversation_id)
    
    # ========== ConflictSubsystem Interface Implementation ==========
    
    @property
    def subsystem_type(self) -> SubsystemType:
        return SubsystemType.SLICE_OF_LIFE
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'detect_emerging_tensions',
            'embed_in_daily_activities',
            'pattern_based_resolution',
            'generate_slice_of_life_conflicts',
            'subtle_conflict_manifestation'
        }
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        return {SubsystemType.FLOW, SubsystemType.SOCIAL}
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        return {
            EventType.STATE_SYNC,
            EventType.PLAYER_CHOICE,
            EventType.PHASE_TRANSITION,
            EventType.CONFLICT_CREATED
        }
    
    async def initialize(self, synthesizer: 'ConflictSynthesizer') -> bool:
        """Initialize with synthesizer reference"""
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Handle events from synthesizer"""
        try:
            if event.event_type == EventType.STATE_SYNC:
                return await self._handle_state_sync(event)
            elif event.event_type == EventType.PLAYER_CHOICE:
                return await self._handle_player_choice(event)
            elif event.event_type == EventType.PHASE_TRANSITION:
                return await self._handle_phase_transition(event)
            elif event.event_type == EventType.CONFLICT_CREATED:
                return await self._handle_conflict_creation(event)
            else:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'handled': False}
                )
        except Exception as e:
            logger.error(f"SliceOfLife error handling event: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of subsystem"""
        try:
            tensions = await self.detector.detect_brewing_tensions()
            return {
                'healthy': True,
                'active_tensions': len(tensions),
                'components': {
                    'detector': 'operational',
                    'manager': 'operational',
                    'resolver': 'operational',
                    'daily_integration': 'operational'
                }
            }
        except Exception as e:
            return {'healthy': False, 'issue': str(e)}
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get slice-of-life specific conflict data"""
        resolution = await self.resolver.check_resolution_by_pattern(conflict_id)
        return {
            'subsystem': 'slice_of_life',
            'pattern_resolution_available': resolution is not None,
            'resolution_details': resolution
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        tensions = await self.detector.detect_brewing_tensions()
        return {
            'emerging_tensions': len(tensions),
            'tension_types': [t.get('type', 'unknown') for t in tensions],
            'daily_integration_active': True
        }
    
    # ========== Event Handlers ==========
    
    async def _handle_state_sync(self, event: SystemEvent) -> SubsystemResponse:
        """Handle scene state synchronization"""
        scene_context = event.payload
        
        # Detect emerging tensions
        tensions = await self.detector.detect_brewing_tensions()
        
        # Check for conflicts in current activity
        activity = scene_context.get('activity', 'daily_routine')
        present_npcs = scene_context.get('present_npcs', [])
        
        manifestations = []
        side_effects = []
        
        if tensions:
            # Create new conflict if tension is high enough
            for tension in tensions[:1]:  # Limit to one new conflict
                if tension.get('tension_level', 0) > 0.6:
                    side_effects.append(SystemEvent(
                        event_id=f"create_slice_{datetime.now().timestamp()}",
                        event_type=EventType.CONFLICT_CREATED,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'conflict_type': tension['type'].value,
                            'context': {
                                'description': tension['description'],
                                'intensity': tension['intensity'].value,
                                'evidence': tension.get('evidence', [])
                            }
                        },
                        priority=5
                    ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_detected': len(tensions),
                'manifestations': manifestations,
                'slice_of_life_active': len(manifestations) > 0
            },
            side_effects=side_effects
        )
    
    async def _handle_player_choice(self, event: SystemEvent) -> SubsystemResponse:
        """Handle player choices"""
        conflict_id = event.payload.get('conflict_id')
        
        if not conflict_id:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'no_conflict': True}
            )
        
        # Check for pattern-based resolution
        resolution = await self.resolver.check_resolution_by_pattern(conflict_id)
        
        side_effects = []
        if resolution:
            side_effects.append(SystemEvent(
                event_id=f"resolve_{conflict_id}",
                event_type=EventType.CONFLICT_RESOLVED,
                source_subsystem=self.subsystem_type,
                payload={
                    'conflict_id': conflict_id,
                    'resolution': resolution,
                    'type': 'pattern_based'
                }
            ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'choice_processed': True, 'resolution': resolution},
            side_effects=side_effects
        )
    
    async def _handle_phase_transition(self, event: SystemEvent) -> SubsystemResponse:
        """Handle phase transitions"""
        new_phase = event.payload.get('new_phase')
        
        data = {
            'integration_adjustment': 'reducing' if new_phase in ['resolution', 'aftermath'] else 'maintaining',
            'phase_acknowledged': new_phase
        }
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=data
        )
    
    async def _handle_conflict_creation(self, event: SystemEvent) -> SubsystemResponse:
        """Handle new conflict creation"""
        conflict_type = event.payload.get('conflict_type', '')
        
        # Check if this is a slice-of-life type
        is_slice_of_life = conflict_type in [t.value for t in SliceOfLifeConflictType]
        
        if is_slice_of_life:
            context = event.payload.get('context', {})
            conflict_id = event.payload.get('conflict_id')
            
            if conflict_id:
                event_result = await self.manager.embed_conflict_in_activity(
                    conflict_id,
                    context.get('activity', 'conversation'),
                    context.get('npcs', [])
                )
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        'conflict_type': 'slice_of_life',
                        'initial_manifestation': event_result.conflict_manifestation,
                        'daily_integration': True
                    }
                )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'not_slice_of_life': True}
        )

# ===============================================================================
# ORIGINAL COMPONENTS (Preserved with minor modifications)
# ===============================================================================

class EmergentConflictDetector:
    """Detects conflicts emerging from daily interactions using LLM analysis"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._pattern_analyzer = None
        
    @property
    def pattern_analyzer(self) -> Agent:
        """Lazy-load pattern analysis agent"""
        if self._pattern_analyzer is None:
            self._pattern_analyzer = Agent(
                name="Pattern Analyzer",
                instructions="""
                Analyze memory patterns and relationship dynamics for emerging slice-of-life conflicts.
                
                Look for:
                - Recurring permission-seeking behaviors
                - Subtle boundary violations
                - Power imbalances in daily routines
                - Emotional labor disparities
                - Control patterns disguised as care
                - Gradual isolation tactics
                - Decision fatigue patterns
                
                Generate nuanced, contextual conflict descriptions that feel organic to daily life.
                Focus on subtle tensions rather than dramatic confrontations.
                """,
                model="gpt-5-nano",
            )
        return self._pattern_analyzer
    
    async def detect_brewing_tensions(self) -> List[Dict[str, Any]]:
        """Analyze recent interactions for emerging conflicts with LLM"""
        
        async with get_db_connection_context() as conn:
            # Get recent memory patterns
            memory_patterns = await conn.fetch("""
                SELECT 
                    entity_id,
                    entity_type,
                    memory_text,
                    emotional_valence,
                    tags
                FROM enhanced_memories
                WHERE user_id = $1 
                AND conversation_id = $2
                AND created_at > NOW() - INTERVAL '3 days'
                ORDER BY created_at DESC
                LIMIT 100
            """, self.user_id, self.conversation_id)
            
            # Get relationship dynamics
            relationships = await conn.fetch("""
                SELECT 
                    entity1_id,
                    entity2_id,
                    dimension,
                    current_value,
                    recent_delta
                FROM relationship_dimensions
                WHERE user_id = $1 
                AND conversation_id = $2
                AND dimension IN ('dominance', 'control', 'dependency', 'resistance')
            """, self.user_id, self.conversation_id)
        
        if not memory_patterns and not relationships:
            return []
        
        # Use LLM to analyze patterns
        conflicts = await self._analyze_patterns_with_llm(memory_patterns, relationships)
        
        return conflicts
    
    async def _analyze_patterns_with_llm(
        self, 
        memories: List, 
        relationships: List
    ) -> List[Dict]:
        """Use LLM to detect emerging tensions from patterns"""
        
        # Prepare context for LLM
        memory_summary = self._summarize_memories(memories[:20])
        relationship_summary = self._summarize_relationships(relationships)
        
        prompt = f"""
        Analyze these recent patterns for emerging slice-of-life conflicts:
        
        Recent Memories:
        {memory_summary}
        
        Relationship Dynamics:
        {relationship_summary}
        
        Identify 1-3 brewing tensions that could become subtle daily conflicts.
        For each, provide:
        1. Conflict type (permission_patterns, boundary_erosion, etc.)
        2. Intensity level (subtext, tension, passive)
        3. A specific, contextual description
        4. Evidence from the patterns
        
        Format as JSON array.
        """
        
        response = await self.pattern_analyzer.run(prompt)
        
        try:
            tensions = json.loads(response.content)
            
            # Map to internal format
            conflicts = []
            for tension in tensions:
                conflicts.append({
                    'type': SliceOfLifeConflictType[tension.get('conflict_type', 'SUBTLE_RIVALRY').upper()],
                    'intensity': ConflictIntensity[tension.get('intensity', 'TENSION').upper()],
                    'description': tension.get('description', 'A subtle tension emerges'),
                    'evidence': tension.get('evidence', []),
                    'tension_level': random.uniform(0.3, 0.7)
                })
            
            return conflicts
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return []
    
    def _summarize_memories(self, memories: List) -> str:
        """Create a summary of memories for LLM context"""
        summary = []
        for m in memories[:10]:
            summary.append(f"- {m['memory_text']} (emotion: {m.get('emotional_valence', 'neutral')})")
        return "\n".join(summary) if summary else "No recent significant memories"
    
    def _summarize_relationships(self, relationships: List) -> str:
        """Create a summary of relationship dynamics"""
        summary = []
        for r in relationships[:5]:
            summary.append(
                f"- {r['dimension']}: {r['current_value']:.2f} "
                f"(recent change: {r.get('recent_delta', 0):.2f})"
            )
        return "\n".join(summary) if summary else "No significant relationship dynamics"


class SliceOfLifeConflictManager:
    """Manages conflicts through daily activities with LLM generation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._conflict_agent = None
        
    @property
    def conflict_agent(self) -> Agent:
        """Lazy-load conflict generation agent"""
        if self._conflict_agent is None:
            self._conflict_agent = Agent(
                name="Slice of Life Conflict Director",
                instructions="""
                Generate subtle conflicts embedded in daily routines.
                
                Focus on:
                - Mundane moments where power dynamics emerge
                - Small choices that reveal larger patterns
                - Passive-aggressive behaviors disguised as normal interaction
                - The accumulation of tiny surrenders
                - Conflicts that feel realistic and relatable
                
                Avoid dramatic confrontations. Keep everything subtle and slice-of-life.
                Make NPCs feel like real people with complex motivations.
                """,
                model="gpt-5-nano",
            )
        return self._conflict_agent
    
    async def embed_conflict_in_activity(
        self,
        conflict_id: int,
        activity_type: str,
        participating_npcs: List[int]
    ) -> DailyConflictEvent:
        """Generate how a conflict manifests in a daily activity"""
        
        # Get conflict details
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            # Get NPC details for context
            npc_details = []
            for npc_id in participating_npcs[:3]:
                npc = await conn.fetchrow("""
                    SELECT name, personality_traits FROM NPCs WHERE npc_id = $1
                """, npc_id)
                if npc:
                    npc_details.append(f"{npc['name']} ({npc.get('personality_traits', 'unknown')})")
        
        # Generate manifestation with LLM
        prompt = f"""
        Generate how this conflict manifests during {activity_type}:
        
        Conflict Type: {conflict['conflict_type']}
        Intensity: {conflict.get('intensity', 'tension')}
        Phase: {conflict.get('phase', 'active')}
        NPCs Present: {', '.join(npc_details)}
        
        Create:
        1. A specific manifestation (1-2 sentences)
        2. Whether player gets an explicit choice (true/false)
        3. Impact on conflict progression (0.0-1.0)
        4. Brief NPC reactions (one line each)
        
        Keep it subtle and realistic to daily life.
        Format as JSON.
        """
        
        response = await self.conflict_agent.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Build NPC reactions dict
            npc_reactions = {}
            if 'npc_reactions' in result:
                for i, npc_id in enumerate(participating_npcs[:len(result['npc_reactions'])]):
                    npc_reactions[npc_id] = result['npc_reactions'][i]
            
            return DailyConflictEvent(
                activity_type=activity_type,
                conflict_manifestation=result.get('manifestation', 'A subtle tension colors the moment'),
                choice_presented=result.get('choice_presented', False),
                accumulation_impact=float(result.get('impact', 0.1)),
                npc_reactions=npc_reactions
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse conflict event: {e}")
            # Return fallback
            return DailyConflictEvent(
                activity_type=activity_type,
                conflict_manifestation="The underlying tension affects the interaction",
                choice_presented=False,
                accumulation_impact=0.1,
                npc_reactions={}
            )


class PatternBasedResolution:
    """Resolves conflicts based on accumulated patterns using LLM"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._resolution_agent = None
    
    @property
    def resolution_agent(self) -> Agent:
        """Lazy-load resolution agent"""
        if self._resolution_agent is None:
            self._resolution_agent = Agent(
                name="Conflict Resolution Analyst",
                instructions="""
                Analyze conflict patterns to determine natural resolutions.
                
                Consider:
                - How small daily choices accumulate
                - Whether patterns are becoming entrenched
                - If resistance is increasing or decreasing
                - Natural resolution through routine establishment
                - The role of exhaustion and decision fatigue
                
                Generate resolutions that feel organic and earned, not dramatic.
                Focus on patterns solidifying or slowly dissolving.
                """,
                model="gpt-5-nano",
            )
        return self._resolution_agent
    
    async def check_resolution_by_pattern(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Check if conflict should resolve based on patterns"""
        
        async with get_db_connection_context() as conn:
            # Get conflict history
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            # Get related memories
            memories = await conn.fetch("""
                SELECT memory_text, emotional_valence, created_at
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND tags @> ARRAY[$3::text]
                ORDER BY created_at DESC
                LIMIT 20
            """, self.user_id, self.conversation_id, f"conflict_{conflict_id}")
        
        if not memories or conflict['progress'] < 50:
            return None
        
        # Use LLM to analyze resolution potential
        prompt = f"""
        Analyze if this conflict should naturally resolve:
        
        Conflict: {conflict['conflict_type']} (Progress: {conflict['progress']}%)
        Phase: {conflict['phase']}
        Recent Pattern:
        {self._format_memory_pattern(memories[:10])}
        
        Determine:
        1. Should it resolve? (yes/no)
        2. Resolution type (gradual_acceptance, subtle_resistance, etc.)
        3. Brief description of how it resolves
        4. New established patterns or routines
        
        Format as JSON. Only resolve if patterns clearly indicate it.
        """
        
        response = await self.resolution_agent.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            if result.get('should_resolve', False):
                return {
                    'resolution_type': ResolutionApproach[result.get('type', 'TIME_EROSION').upper()],
                    'description': result.get('description', 'The conflict fades into routine'),
                    'new_patterns': result.get('new_patterns', []),
                    'final_state': 'resolved'
                }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse resolution: {e}")
        
        return None
    
    def _format_memory_pattern(self, memories: List) -> str:
        """Format memories for LLM analysis"""
        pattern = []
        for m in memories:
            pattern.append(f"- {m['memory_text']}")
        return "\n".join(pattern)


class ConflictDailyIntegration:
    """Integrates conflicts with daily routines using LLM"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._integration_agent = None
    
    @property
    def integration_agent(self) -> Agent:
        """Lazy-load integration agent"""
        if self._integration_agent is None:
            self._integration_agent = Agent(
                name="Daily Conflict Integrator",
                instructions="""
                Weave conflicts naturally into daily activities.
                
                Make conflicts feel like:
                - Natural extensions of routine interactions
                - Subtle undercurrents rather than focal points
                - Realistic relationship dynamics
                - Accumulations of small moments
                
                Generate specific details that make scenes feel lived-in and real.
                """,
                model="gpt-5-nano",
            )
        return self._integration_agent
    
    async def get_conflicts_for_time_of_day(self, time_of_day: str) -> List[Dict]:
        """Get conflicts appropriate for current time"""
        
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.*, array_agg(cs.npc_id) as stakeholder_npcs
                FROM Conflicts c
                LEFT JOIN conflict_stakeholders cs ON c.conflict_id = cs.conflict_id
                WHERE c.user_id = $1 AND c.conversation_id = $2
                AND c.is_active = true
                GROUP BY c.conflict_id
            """, self.user_id, self.conversation_id)
        
        # Filter conflicts appropriate for time
        appropriate = []
        for conflict in conflicts:
            if await self._is_appropriate_for_time(conflict, time_of_day):
                appropriate.append(dict(conflict))
        
        return appropriate
    
    async def _is_appropriate_for_time(self, conflict: Dict, time_of_day: str) -> bool:
        """Determine if conflict fits the time of day"""
        
        # Use LLM for intelligent filtering
        prompt = f"""
        Is this conflict appropriate for {time_of_day}?
        
        Conflict Type: {conflict['conflict_type']}
        Intensity: {conflict.get('intensity', 'tension')}
        
        Consider typical daily patterns and when such tensions would naturally surface.
        Answer: yes or no
        """
        
        response = await self.integration_agent.run(prompt)
        return 'yes' in response.content.lower()
