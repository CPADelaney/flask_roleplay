# logic/conflict_system/slice_of_life_conflicts.py
"""
Slice-of-life conflict system with LLM-generated dynamic content.
Refactored to work as a ConflictSubsystem with the synthesizer (circular-safe).
"""

import logging
import weakref
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from db.connection import get_db_connection_context

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

    async def detect_brewing_tensions(self) -> List[Dict[str, Any]]:
        """Analyze recent interactions for emerging conflicts using cache-first helper."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            get_detected_tensions,
        )

        return await get_detected_tensions(self.user_id, self.conversation_id)

    async def collect_tension_inputs(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch memory and relationship slices for downstream processing."""
        async with get_db_connection_context() as conn:
            memory_rows = await conn.fetch(
                """
                SELECT entity_id, entity_type, memory_text, emotional_valence, tags
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                  AND created_at > NOW() - INTERVAL '3 days'
                ORDER BY created_at DESC
                LIMIT 100
                """,
                self.user_id,
                self.conversation_id,
            )

            relationship_rows = await conn.fetch(
                """
                SELECT entity1_id, entity2_id, dimension, current_value, recent_delta
                FROM relationship_dimensions
                WHERE user_id = $1 AND conversation_id = $2
                  AND dimension IN ('dominance', 'control', 'dependency', 'resistance')
                """,
                self.user_id,
                self.conversation_id,
            )
        memories = [dict(row) for row in memory_rows]
        relationships = [dict(row) for row in relationship_rows]
        return memories, relationships

    async def _detect_tensions(self) -> List[Dict[str, Any]]:
        """Slow-path implementation that now returns heuristic tensions."""
        memories, relationships = await self.collect_tension_inputs()

        if not memories and not relationships:
            return []

        tensions = await self._analyze_patterns_with_llm(memories, relationships)

        sanitized: List[Dict[str, Any]] = []
        for tension in tensions:
            sanitized.append(
                {
                    "type": getattr(tension.get("type"), "value", tension.get("type", "subtle_rivalry")),
                    "intensity": getattr(
                        tension.get("intensity"), "value", tension.get("intensity", "tension")
                    ),
                    "description": tension.get("description", "A subtle tension emerges"),
                    "evidence": list(tension.get("evidence", [])),
                    "tension_level": float(tension.get("tension_level", 0.5)),
                }
            )
        return sanitized

    async def _analyze_patterns_with_llm(
        self,
        memories: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Heuristic approximation of tension analysis used as a fallback."""

        tensions: List[Dict[str, Any]] = []

        relationship_map = {
            "dominance": SliceOfLifeConflictType.ROUTINE_DOMINANCE,
            "control": SliceOfLifeConflictType.FINANCIAL_CONTROL,
            "dependency": SliceOfLifeConflictType.CARE_DEPENDENCY,
            "resistance": SliceOfLifeConflictType.INDEPENDENCE_STRUGGLE,
        }

        for rel in relationships[:5]:
            dimension = str(rel.get("dimension", "")).lower()
            current_value = float(rel.get("current_value") or 0.0)
            if abs(current_value) < 0.25:
                continue

            ctype = relationship_map.get(dimension, SliceOfLifeConflictType.SUBTLE_RIVALRY)
            magnitude = min(max(abs(current_value), 0.1), 1.0)
            if magnitude > 0.75:
                intensity = ConflictIntensity.CONFRONTATION
            elif magnitude > 0.55:
                intensity = ConflictIntensity.DIRECT
            elif magnitude > 0.4:
                intensity = ConflictIntensity.PASSIVE
            else:
                intensity = ConflictIntensity.TENSION

            description = (
                f"{dimension.title()} dynamics are trending toward {('imbalance' if current_value > 0 else 'withdrawal')}"
                if dimension
                else "Subtle relationship pressure is forming"
            )

            evidence = []
            if relationships and dimension:
                evidence.append(
                    f"{dimension.title()} score shifted by {float(rel.get('recent_delta') or 0.0):+.2f}"
                )
            if memories:
                evidence.append(memories[0].get("memory_text", "Recent interaction felt strained."))

            tensions.append(
                {
                    "type": ctype,
                    "intensity": intensity,
                    "description": description,
                    "evidence": evidence,
                    "tension_level": round(magnitude, 2),
                }
            )

        if not tensions and memories:
            sample = memories[0]
            sentiment = str(sample.get("emotional_valence", "neutral")).lower()
            intensity = ConflictIntensity.SUBTEXT if sentiment == "positive" else ConflictIntensity.TENSION
            tensions.append(
                {
                    "type": SliceOfLifeConflictType.SUBTLE_RIVALRY,
                    "intensity": intensity,
                    "description": sample.get(
                        "memory_text", "A low-grade disagreement is brewing in daily routines."
                    ),
                    "evidence": [sample.get("memory_text", "")],
                    "tension_level": 0.35,
                }
            )

        return tensions[:3]
    
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

    async def embed_conflict_in_activity(
        self,
        conflict_id: int,
        activity_type: str,
        participating_npcs: List[int]
    ) -> DailyConflictEvent:
        """Cache-first helper that returns a daily conflict event."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            get_activity_manifestation,
        )

        return await get_activity_manifestation(
            self.user_id,
            self.conversation_id,
            conflict_id,
            activity_type,
            participating_npcs,
        )

    async def collect_activity_context(
        self,
        conflict_id: int,
        participating_npcs: List[int],
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Load conflict record and NPC descriptors for downstream use."""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow(
                """SELECT * FROM Conflicts WHERE id = $1""",
                int(conflict_id),
            )

            npc_details = []
            for npc_id in participating_npcs[:3]:
                npc = await conn.fetchrow(
                    """SELECT name, personality_traits FROM NPCs WHERE id = $1""",
                    int(npc_id),
                )
                if npc:
                    name = npc["name"]
                    traits = (
                        npc.get("personality_traits", "unknown")
                        if isinstance(npc, dict)
                        else npc["personality_traits"]
                    )
                    npc_details.append(f"{name} ({traits})")
        return dict(conflict) if conflict else None, npc_details

    async def _generate_conflict_manifestation(
        self,
        conflict: Optional[Dict[str, Any]],
        activity_type: str,
        participating_npcs: List[int],
        npc_descriptors: Optional[List[str]] = None,
    ) -> DailyConflictEvent:
        """Heuristic manifestation generator used when LLM support is unavailable."""

        npc_descriptors = npc_descriptors or []
        intensity = str((conflict or {}).get("intensity", "tension")).lower()
        phase = str((conflict or {}).get("phase", "active")).lower()

        if intensity in {"confrontation", "direct"}:
            manifestation = (
                f"{activity_type.title()} is interrupted by a pointed exchange"
                f" between the involved parties."
            )
            impact = 0.25
            choice = True
        elif phase == "cooldown":
            manifestation = (
                f"Residual tension lingers during {activity_type}, but everyone keeps things polite."
            )
            impact = 0.05
            choice = False
        else:
            npc_clause = (
                f" involving {', '.join(npc_descriptors[:2])}" if npc_descriptors else ""
            )
            manifestation = (
                f"{activity_type.title()} carries an undercurrent of hesitation{npc_clause}."
            )
            impact = 0.12
            choice = False

        npc_reactions: Dict[int, str] = {}
        for idx, npc_id in enumerate(participating_npcs[: len(npc_descriptors or [])]):
            npc_reactions[int(npc_id)] = (
                f"{npc_descriptors[idx].split('(')[0].strip()} keeps their distance."
            )

        return DailyConflictEvent(
            activity_type=activity_type,
            conflict_manifestation=manifestation,
            choice_presented=choice,
            accumulation_impact=impact,
            npc_reactions=npc_reactions,
        )

    async def _embed_conflict_in_activity(
        self,
        conflict_id: int,
        activity_type: str,
        participating_npcs: List[int],
    ) -> DailyConflictEvent:
        """Heuristic-only embedding retained for compatibility with cache warmers."""

        conflict, npc_details = await self.collect_activity_context(
            conflict_id, participating_npcs
        )
        return await self._generate_conflict_manifestation(
            conflict, activity_type, participating_npcs, npc_details
        )


class PatternBasedResolution:
    """Resolves conflicts based on accumulated patterns using LLM"""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def check_resolution_by_pattern(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Cache-first helper that returns pattern-based resolution if available."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            get_resolution_recommendation,
        )

        return await get_resolution_recommendation(
            self.user_id, self.conversation_id, conflict_id
        )

    async def collect_resolution_inputs(
        self, conflict_id: int
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Gather conflict record and tagged memories for evaluation."""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow(
                """SELECT * FROM Conflicts WHERE id = $1""",
                int(conflict_id),
            )
            memories = await conn.fetch(
                """
                SELECT memory_text, emotional_valence, created_at
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags @> ARRAY[$3::text]
                ORDER BY created_at DESC
                LIMIT 20
                """,
                self.user_id,
                self.conversation_id,
                f"conflict_{int(conflict_id)}",
            )
        return dict(conflict) if conflict else None, [dict(row) for row in memories]

    async def _evaluate_resolution(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Heuristic evaluation that approximates the previous LLM decision."""

        conflict, memories = await self.collect_resolution_inputs(conflict_id)

        if not conflict or not memories:
            return None

        progress = float(conflict.get("progress", 0) or 0)
        phase = str(conflict.get("phase", "active")).lower()
        positive_memories = [
            m for m in memories if str(m.get("emotional_valence", "")).lower() in {"positive", "hopeful"}
        ]

        if progress >= 85 or phase in {"cooldown", "dormant"}:
            resolution_type = ResolutionApproach.TIME_EROSION.value
        elif progress >= 65 and positive_memories:
            resolution_type = ResolutionApproach.NEGOTIATED_COMPROMISE.value
        elif len(memories) >= 5 and all(
            str(m.get("emotional_valence", "")).lower() == "negative" for m in memories[:3]
        ):
            resolution_type = ResolutionApproach.SUBTLE_RESISTANCE.value
        else:
            return None

        description = "Conflict momentum suggests a natural winding down."
        if resolution_type == ResolutionApproach.NEGOTIATED_COMPROMISE.value:
            description = "Recent interactions show parties testing small compromises."
        elif resolution_type == ResolutionApproach.SUBTLE_RESISTANCE.value:
            description = "Persistence of negative beats indicates resistance rather than escalation."

        return {
            "resolution_type": resolution_type,
            "description": description,
            "new_patterns": [m.get("memory_text", "") for m in positive_memories[:2]],
            "final_state": "resolved",
        }
    
    def _format_memory_pattern(self, memories: List) -> str:
        """Format memories for LLM analysis"""
        return "\n".join(f"- {m['memory_text']}" for m in memories)


class ConflictDailyIntegration:
    """Integrates conflicts with daily routines using LLM"""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

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
        """Cache-first helper for time-of-day suitability."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            is_conflict_appropriate_for_time,
        )

        return await is_conflict_appropriate_for_time(
            self.user_id, self.conversation_id, conflict, time_of_day
        )

    async def _evaluate_time_appropriateness(
        self, conflict: Dict, time_of_day: str
    ) -> bool:
        """Heuristic classifier for time-of-day suitability."""

        if not conflict:
            return False

        intensity = str(conflict.get("intensity", "tension")).lower()
        phase = str(conflict.get("phase", "active")).lower()
        time_key = time_of_day.lower()

        quiet_hours = {"sleep", "rest", "late_night", "midnight"}
        routine_hours = {"morning", "commute", "work", "afternoon"}

        if intensity in {"confrontation", "direct"} and time_key in quiet_hours:
            return False
        if phase == "dormant" and time_key not in routine_hours:
            return False
        if intensity == "subtext":
            return True

        return time_key not in {"sleep", "rest"}
