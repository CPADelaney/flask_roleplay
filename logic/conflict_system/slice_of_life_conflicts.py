# logic/conflict_system/slice_of_life_conflicts.py
"""
Slice-of-life conflict system with LLM-generated dynamic content.
Refactored to work as a ConflictSubsystem with the synthesizer (circular-safe).
"""

import logging
import json
import random
import weakref
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from agents import Agent, Runner
from db.connection import get_db_connection_context
from logic.conflict_system.dynamic_conflict_template import extract_runner_response

logger = logging.getLogger(__name__)

# ===============================================================================
# Lazy orchestrator type access (avoid circular imports at module load)
# ===============================================================================

def _orch():
    from logic.conflict_system.conflict_synthesizer import (
        SubsystemType, EventType, SystemEvent, SubsystemResponse
    )
    return SubsystemType, EventType, SystemEvent, SubsystemResponse


# ===============================================================================
# ENUMS
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
# SLICE OF LIFE CONFLICT SUBSYSTEM (duck-typed, circular-safe)
# ===============================================================================

class SliceOfLifeConflictSubsystem:
    """
    Slice-of-life conflict subsystem integrated with synthesizer.
    Combines all slice-of-life components into one subsystem.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None  # weakref set in initialize
        
        # Components
        self.detector = EmergentConflictDetector(user_id, conversation_id)
        self.manager = SliceOfLifeConflictManager(user_id, conversation_id)
        self.resolver = PatternBasedResolution(user_id, conversation_id)
        self.daily_integration = ConflictDailyIntegration(user_id, conversation_id)
    
    # ========== Subsystem Interface ==========
    
    @property
    def subsystem_type(self):
        SubsystemType, _, _, _ = _orch()
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
    def dependencies(self) -> Set:
        SubsystemType, _, _, _ = _orch()
        return {SubsystemType.FLOW, SubsystemType.SOCIAL}
    
    @property
    def event_subscriptions(self) -> Set:
        _, EventType, _, _ = _orch()
        return {
            EventType.STATE_SYNC,
            EventType.PLAYER_CHOICE,
            EventType.PHASE_TRANSITION,
            EventType.CONFLICT_CREATED
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize with synthesizer reference"""
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event):
        """Handle events from synthesizer"""
        _, EventType, _, SubsystemResponse = _orch()
        try:
            if event.event_type == EventType.STATE_SYNC:
                return await self._handle_state_sync(event)
            if event.event_type == EventType.PLAYER_CHOICE:
                return await self._handle_player_choice(event)
            if event.event_type == EventType.PHASE_TRANSITION:
                return await self._handle_phase_transition(event)
            if event.event_type == EventType.CONFLICT_CREATED:
                return await self._handle_conflict_creation(event)
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'handled': False},
                side_effects=[]
            )
        except Exception as e:
            logger.error(f"SliceOfLife error handling event: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
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
    
    # Optional: provide a small scene bundle to orchestrator for parallel merges
    async def get_scene_bundle(self, scope) -> Dict[str, Any]:
        """
        Cheap bundle: surface subtle ambient effects and opportunities.
        """
        try:
            tensions = await self.detector.detect_brewing_tensions()
            ambient = []
            for t in tensions[:3]:
                label = str(getattr(t.get('intensity'), 'value', t.get('intensity', 'tension')))
                ambient.append(f"slicing_{label}")
            opportunities = []
            for t in tensions[:2]:
                opportunities.append({
                    'type': 'slice_opportunity',
                    'description': str(t.get('description', 'subtle pattern')),
                })
            return {
                'ambient_effects': ambient,
                'opportunities': opportunities,
                'last_changed_at': datetime.now().timestamp(),
            }
        except Exception as e:
            logger.debug(f"slice_of_life get_scene_bundle failed: {e}")
            return {}
    
    # ========== Event Handlers ==========
    
    async def _handle_state_sync(self, event):
        """Handle scene state synchronization"""
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        scene_context = payload.get('scene_context') or payload
        
        # Detect emerging tensions (DB + LLM)
        tensions = await self.detector.detect_brewing_tensions()
        
        activity = scene_context.get('activity', scene_context.get('scene_type', 'daily_routine'))
        present_npcs = scene_context.get('present_npcs') or scene_context.get('npcs') or []
        
        manifestations: List[str] = []
        side_effects = []
        
        # Opportunistically propose a new slice conflict if one looks strong
        for tension in tensions[:1]:
            try:
                level = float(tension.get('tension_level', 0.0) or 0.0)
                if level > 0.6:
                    ctype = getattr(tension.get('type'), 'value', str(tension.get('type', 'permission_patterns')))
                    intensity = getattr(tension.get('intensity'), 'value', str(tension.get('intensity', 'tension')))
                    side_effects.append(SystemEvent(
                        event_id=f"create_slice_{datetime.now().timestamp()}",
                        event_type=EventType.CONFLICT_CREATED,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'conflict_type': ctype,
                            'context': {
                                'activity': activity,
                                'npcs': present_npcs,
                                'description': tension.get('description'),
                                'intensity': intensity,
                                'evidence': tension.get('evidence', [])
                            }
                        },
                        priority=5
                    ))
            except Exception:
                continue
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'tensions_detected': len(tensions),
                'manifestations': manifestations,
                'slice_of_life_active': bool(manifestations)
            },
            side_effects=side_effects
        )
    
    async def _handle_player_choice(self, event):
        """Handle player choices"""
        _, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')
        
        if not conflict_id:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'no_conflict': True},
                side_effects=[]
            )
        
        # Check for pattern-based resolution
        resolution = await self.resolver.check_resolution_by_pattern(int(conflict_id))
        
        side_effects = []
        if resolution:
            side_effects.append(SystemEvent(
                event_id=f"resolve_{conflict_id}",
                event_type=EventType.CONFLICT_RESOLVED,
                source_subsystem=self.subsystem_type,
                payload={
                    'conflict_id': int(conflict_id),
                    'resolution_type': 'pattern_based',
                    'context': {'resolution': resolution}
                },
                priority=4
            ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'choice_processed': True, 'resolution': resolution},
            side_effects=side_effects
        )
    
    async def _handle_phase_transition(self, event):
        """Handle phase transitions"""
        _, _, _, SubsystemResponse = _orch()
        new_phase = (event.payload or {}).get('new_phase')
        data = {
            'integration_adjustment': 'reducing' if new_phase in ['resolution', 'aftermath'] else 'maintaining',
            'phase_acknowledged': new_phase
        }
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=data,
            side_effects=[]
        )
    
    async def _handle_conflict_creation(self, event):
        """Handle new conflict creation (post-create embedding if applicable)"""
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        conflict_type = str(payload.get('conflict_type', ''))
        
        is_slice_of_life = conflict_type in [t.value for t in SliceOfLifeConflictType]
        if not is_slice_of_life:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'not_slice_of_life': True},
                side_effects=[]
            )
        
        context = payload.get('context', {}) or {}
        conflict_id = payload.get('conflict_id')  # May be absent during initial create event
        if conflict_id:
            event_result = await self.manager.embed_conflict_in_activity(
                int(conflict_id),
                context.get('activity', 'conversation'),
                list(context.get('npcs', []) or [])
            )
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={
                    'conflict_type': 'slice_of_life',
                    'initial_manifestation': event_result.conflict_manifestation,
                    'daily_integration': True
                },
                side_effects=[]
            )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'pending_conflict_embedding': True},
            side_effects=[]
        )


# ===============================================================================
# ORIGINAL COMPONENTS (schema-safe + LLM parsing hardened)
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
                """,
                model="gpt-5-nano",
            )
        return self._pattern_analyzer
    
    async def detect_brewing_tensions(self) -> List[Dict[str, Any]]:
        """Analyze recent interactions for emerging conflicts with LLM"""
        async with get_db_connection_context() as conn:
            memory_patterns = await conn.fetch("""
                SELECT entity_id, entity_type, memory_text, emotional_valence, tags
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                  AND created_at > NOW() - INTERVAL '3 days'
                ORDER BY created_at DESC
                LIMIT 100
            """, self.user_id, self.conversation_id)
            
            relationships = await conn.fetch("""
                SELECT entity1_id, entity2_id, dimension, current_value, recent_delta
                FROM relationship_dimensions
                WHERE user_id = $1 AND conversation_id = $2
                  AND dimension IN ('dominance', 'control', 'dependency', 'resistance')
            """, self.user_id, self.conversation_id)
        
        if not memory_patterns and not relationships:
            return []
        
        return await self._analyze_patterns_with_llm(memory_patterns, relationships)
    
    async def _analyze_patterns_with_llm(
        self, 
        memories: List, 
        relationships: List
    ) -> List[Dict]:
        """Use LLM to detect emerging tensions from patterns"""
        memory_summary = self._summarize_memories(memories[:20])
        relationship_summary = self._summarize_relationships(relationships)
        
        prompt = f"""
Analyze recent patterns for emerging slice-of-life conflicts:

Recent Memories:
{memory_summary}

Relationship Dynamics:
{relationship_summary}

Return a JSON array (1-3 items). Each item:
{{
  "conflict_type": "permission_patterns|boundary_erosion|subtle_rivalry|...",
  "intensity": "subtext|tension|passive",
  "description": "specific, contextual",
  "evidence": ["..."]
}}
"""
        response = await Runner.run(self.pattern_analyzer, prompt)
        try:
            parsed = extract_runner_response(response)
            tensions = json.loads(parsed)
            conflicts = []
            for t in tensions:
                try:
                    ctype = SliceOfLifeConflictType[str(t.get('conflict_type', 'subtle_rivalry')).upper()]
                except Exception:
                    ctype = SliceOfLifeConflictType.SUBTLE_RIVALRY
                try:
                    intensity = ConflictIntensity[str(t.get('intensity', 'tension')).upper()]
                except Exception:
                    intensity = ConflictIntensity.TENSION
                conflicts.append({
                    'type': ctype,
                    'intensity': intensity,
                    'description': t.get('description', 'A subtle tension emerges'),
                    'evidence': t.get('evidence', []),
                    'tension_level': random.uniform(0.3, 0.7)
                })
            return conflicts
        except Exception as e:
            logger.warning(f"Failed to parse LLM tension analysis: {e}")
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
                f"- {r['dimension']}: {float(r['current_value']):.2f} "
                f"(recent change: {float(r.get('recent_delta', 0) or 0):.2f})"
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
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE id = $1
            """, int(conflict_id))
            
            npc_details = []
            for npc_id in participating_npcs[:3]:
                npc = await conn.fetchrow("""
                    SELECT name, personality_traits FROM NPCs WHERE id = $1
                """, int(npc_id))
                if npc:
                    name = npc['name']
                    traits = npc.get('personality_traits', 'unknown') if isinstance(npc, dict) else npc['personality_traits']
                    npc_details.append(f"{name} ({traits})")
        
        prompt = f"""
Generate how this conflict manifests during {activity_type}:

Conflict Type: {conflict['conflict_type']}
Intensity: {conflict.get('intensity', 'tension')}
Phase: {conflict.get('phase', 'active')}
NPCs Present: {', '.join(npc_details)}

Return JSON:
{{
  "manifestation": "1-2 sentences",
  "choice_presented": false,
  "impact": 0.1,
  "npc_reactions": ["npc 1 line", "npc 2 line"]
}}
"""
        response = await Runner.run(self.conflict_agent, prompt)
        try:
            parsed = extract_runner_response(response)
            result = json.loads(parsed)
            npc_reactions: Dict[int, str] = {}
            if isinstance(result.get('npc_reactions'), list):
                for i, npc_id in enumerate(participating_npcs[:len(result['npc_reactions'])]):
                    npc_reactions[int(npc_id)] = str(result['npc_reactions'][i])
            return DailyConflictEvent(
                activity_type=activity_type,
                conflict_manifestation=str(result.get('manifestation', 'A subtle tension colors the moment')),
                choice_presented=bool(result.get('choice_presented', False)),
                accumulation_impact=float(result.get('impact', 0.1)),
                npc_reactions=npc_reactions
            )
        except Exception as e:
            logger.warning(f"Failed to parse conflict manifestation: {e}")
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
                """,
                model="gpt-5-nano",
            )
        return self._resolution_agent
    
    async def check_resolution_by_pattern(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Check if conflict should resolve based on patterns"""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE id = $1
            """, int(conflict_id))
            memories = await conn.fetch("""
                SELECT memory_text, emotional_valence, created_at
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags @> ARRAY[$3::text]
                ORDER BY created_at DESC
                LIMIT 20
            """, self.user_id, self.conversation_id, f"conflict_{int(conflict_id)}")
        
        if not memories or (conflict and int(conflict.get('progress', conflict.get('progress', 0))) < 50):
            return None
        
        prompt = f"""
Analyze if this conflict should naturally resolve:

Conflict: {conflict['conflict_type']} (Progress: {conflict['progress']}%)
Phase: {conflict.get('phase', 'active')}
Recent Pattern:
{self._format_memory_pattern(memories[:10])}

Return JSON:
{{
  "should_resolve": false,
  "type": "time_erosion|subtle_resistance|...",
  "description": "how it resolves",
  "new_patterns": []
}}
"""
        response = await Runner.run(self.resolution_agent, prompt)
        try:
            parsed = extract_runner_response(response)
            result = json.loads(parsed)
            if bool(result.get('should_resolve', False)):
                try:
                    rtype = ResolutionApproach[str(result.get('type', 'time_erosion')).upper()]
                except Exception:
                    rtype = ResolutionApproach.TIME_EROSION
                return {
                    'resolution_type': rtype,
                    'description': str(result.get('description', 'The conflict fades into routine')),
                    'new_patterns': list(result.get('new_patterns', [])),
                    'final_state': 'resolved'
                }
        except Exception as e:
            logger.warning(f"Failed to parse resolution: {e}")
        return None
    
    def _format_memory_pattern(self, memories: List) -> str:
        """Format memories for LLM analysis"""
        return "\n".join(f"- {m['memory_text']}" for m in memories)


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
                """,
                model="gpt-5-nano",
            )
        return self._integration_agent
    
    async def get_conflicts_for_time_of_day(self, time_of_day: str) -> List[Dict]:
        """Get conflicts appropriate for current time"""
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.*, COALESCE(array_agg(s.npc_id) FILTER (WHERE s.npc_id IS NOT NULL), '{}') AS stakeholder_npcs
                FROM Conflicts c
                LEFT JOIN stakeholders s
                  ON s.conflict_id = c.id
                 AND s.user_id = $1
                 AND s.conversation_id = $2
                WHERE c.user_id = $1 AND c.conversation_id = $2
                  AND c.is_active = true
                GROUP BY c.id
            """, self.user_id, self.conversation_id)
        
        appropriate = []
        for c in conflicts:
            if await self._is_appropriate_for_time(dict(c), time_of_day):
                appropriate.append(dict(c))
        return appropriate
    
    async def _is_appropriate_for_time(self, conflict: Dict, time_of_day: str) -> bool:
        """Determine if conflict fits the time of day"""
        prompt = f"""
Is this conflict appropriate for {time_of_day}?

Conflict Type: {conflict.get('conflict_type')}
Intensity: {conflict.get('intensity', 'tension')}

Answer just yes or no.
"""
        response = await Runner.run(self.integration_agent, prompt)
        try:
            text = extract_runner_response(response).strip().lower()
            return 'yes' in text and 'no' not in text[:5]
        except Exception:
            return False
