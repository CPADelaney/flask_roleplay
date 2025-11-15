# logic/conflict_system/conflict_victory.py
"""
Victory Conditions System with LLM-generated dynamic content
Integrated with ConflictSynthesizer as the central orchestrator
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Set, TypedDict, Sequence, Mapping
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

import nyx.gateway.llm_gateway as llm_gateway
from agents import Agent, function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from logic.conflict_system import conflict_victory_hotpath
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from logic.conflict_system.conflict_synthesizer import EventType, SubsystemResponse, SystemEvent
from nyx.config import INTERACTIVE_MODEL, WARMUP_MODEL
from nyx.gateway.llm_gateway import LLMRequest

logger = logging.getLogger(__name__)

# ===============================================================================
# VICTORY TYPES
# ===============================================================================

class VictoryType(Enum):
    """Types of conflict victories"""
    DOMINANCE = "dominance"
    SUBMISSION = "submission"
    COMPROMISE = "compromise"
    PYRRHIC = "pyrrhic"
    TACTICAL = "tactical"
    MORAL = "moral"
    ESCAPE = "escape"
    TRANSFORMATION = "transformation"
    STALEMATE = "stalemate"
    NARRATIVE = "narrative"


@dataclass
class VictoryCondition:
    """Defines what constitutes victory in a conflict"""
    condition_id: int
    conflict_id: int
    stakeholder_id: int
    victory_type: VictoryType
    description: str
    requirements: Dict[str, Any]
    progress: float
    is_achieved: bool
    achievement_impact: Dict[str, float]
    task_metadata: Dict[str, Any]


class AchievementDTO(TypedDict, total=False):
    id: int
    name: str
    description: str
    points: float

class PartialVictoryDTO(TypedDict, total=False):
    stakeholder_id: int
    description: str
    progress: float

class CheckVictoryResponse(TypedDict):
    victory_achieved: bool
    achievements: List[AchievementDTO]
    partial_victories: List[PartialVictoryDTO]
    epilogue: str
    conflict_continues: bool
    error: str

class VictoryPathDTO(TypedDict):
    stakeholder: int
    type: str
    description: str
    requirements: List[str]
    current_progress: float

class VictoryPathsResponse(TypedDict):
    conflict_id: int
    victory_paths: List[VictoryPathDTO]
    error: str


# ===============================================================================
# VICTORY SUBSYSTEM (Integrated with Synthesizer)
# ===============================================================================

class ConflictVictorySubsystem:
    """
    Victory subsystem that integrates with ConflictSynthesizer.
    Manages victory conditions and resolutions.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._victory_generator = None
        self._achievement_narrator = None
        self._consequence_calculator = None
        self._epilogue_writer = None
        
        # Reference to synthesizer (weakref set in initialize)
        self.synthesizer = None
    
    @property
    def subsystem_type(self):
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.VICTORY
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'victory_conditions',
            'achievement_tracking',
            'consequence_calculation',
            'epilogue_generation',
            'partial_victories',
            'narrative_closure'
        }
    
    @property
    def dependencies(self) -> Set:
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return {
            SubsystemType.STAKEHOLDER,  # Need stakeholder info
            SubsystemType.FLOW,         # Need conflict flow state
            SubsystemType.CANON         # Victories can become canonical
        }
    
    @property
    def event_subscriptions(self) -> Set:
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.PHASE_TRANSITION,
            EventType.CONFLICT_RESOLVED,
            EventType.HEALTH_CHECK,
        }
    
    async def initialize(self, synthesizer) -> bool:
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""

        handlers = {
            EventType.CONFLICT_CREATED: self._on_conflict_created,
            EventType.CONFLICT_UPDATED: self._on_conflict_updated,
            EventType.PHASE_TRANSITION: self._on_phase_transition,
            EventType.CONFLICT_RESOLVED: self._on_conflict_resolved,
            EventType.HEALTH_CHECK: self._on_health_check,
        }

        handler = handlers.get(event.event_type)

        try:
            if handler:
                return await handler(event)
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'ignored': True},
                side_effects=[],
            )
        except Exception as exc:
            logger.error("Victory subsystem error while handling %s: %s", event.event_type, exc, exc_info=True)
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(exc)},
                side_effects=[],
            )

    async def _on_conflict_created(self, event) -> SubsystemResponse:
        conflict_id = int(event.payload.get('conflict_id') or 0)
        if conflict_id <= 0:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'missing_conflict_id'},
                side_effects=[],
            )

        conditions = await self._generate_initial_conditions(conflict_id)
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'conflict_id': conflict_id,
                'victory_conditions_created': len(conditions),
                'condition_types': [c.victory_type.value for c in conditions],
            },
            side_effects=[],
        )

    async def _on_conflict_updated(self, event) -> SubsystemResponse:
        conflict_id = int(event.payload.get('conflict_id') or 0)
        current_state = (event.payload or {}).get('state') or {}
        if conflict_id <= 0:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'missing_conflict_id'},
                side_effects=[],
            )

        achievements = await self.check_victory_conditions(conflict_id, current_state)
        side_effects: List[SystemEvent] = []
        if achievements:
            partial = await self.evaluate_partial_victories(conflict_id)
            side_effects.append(
                self._build_resolution_event(
                    conflict_id,
                    achievements,
                    partial,
                    event.event_id,
                )
            )

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'conflict_id': conflict_id,
                'victories_achieved': len(achievements),
                'achievements': achievements,
            },
            side_effects=side_effects,
        )

    async def _on_phase_transition(self, event) -> SubsystemResponse:
        payload = event.payload or {}
        conflict_id = int(payload.get('conflict_id') or 0)
        new_phase = (payload.get('to_phase') or payload.get('phase') or '').lower()
        side_effects: List[SystemEvent] = []
        data: Dict[str, Any] = {
            'conflict_id': conflict_id,
            'phase': new_phase,
        }

        if conflict_id > 0:
            if new_phase == 'climax':
                await self._accelerate_victory_progress(conflict_id)
            achievements = await self.check_victory_conditions(conflict_id, payload.get('state'))
            if achievements:
                partial = await self.evaluate_partial_victories(conflict_id)
                side_effects.append(
                    self._build_resolution_event(
                        conflict_id,
                        achievements,
                        partial,
                        event.event_id,
                    )
                )
                data['victories_achieved'] = len(achievements)

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=data,
            side_effects=side_effects,
        )

    async def _on_conflict_resolved(self, event) -> SubsystemResponse:
        payload = event.payload or {}
        request_type = payload.get('request')
        conflict_id = int(payload.get('conflict_id') or 0)

        if request_type == 'generate_epilogue' and conflict_id > 0:
            interactive = bool(payload.get('interactive', False))
            resolution_context = payload.get('context') or {}
            achievements = payload.get('achievements') or []
            if achievements:
                resolution_context['achievements'] = achievements
            epilogue = await self.generate_epilogue(
                conflict_id,
                interactive=interactive,
                resolution_data=resolution_context,
            )
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'conflict_id': conflict_id, 'epilogue': epilogue, 'interactive': interactive},
                side_effects=[],
            )

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'conflict_id': conflict_id, 'ack': 'conflict_resolved'},
            side_effects=[],
        )

    async def _on_health_check(self, event) -> SubsystemResponse:
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=await self.health_check(),
            side_effects=[],
        )

    def _build_resolution_event(
        self,
        conflict_id: int,
        achievements: List[Dict[str, Any]],
        partial: List[Dict[str, Any]],
        source_event_id: str,
    ) -> SystemEvent:
        return SystemEvent(
            event_id=f"victory_resolution_{conflict_id}_{source_event_id}",
            event_type=EventType.CONFLICT_RESOLVED,
            source_subsystem=self.subsystem_type,
            payload={
                'conflict_id': conflict_id,
                'achievements': achievements,
                'partial_victories': partial,
                'resolution_type': 'victory_conditions_met',
            },
            priority=2,
        )
    
    async def health_check(self) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            stuck = await conn.fetchval("""
                SELECT COUNT(*) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
                  AND progress > 0.5 AND progress < 1.0
                  AND updated_at < CURRENT_TIMESTAMP - INTERVAL '3 days'
            """, self.user_id, self.conversation_id)
            
            total = await conn.fetchval("""
                SELECT COUNT(*) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        
        return {
            'healthy': (stuck or 0) < 5,
            'total_conditions': int(total or 0),
            'stuck_conditions': int(stuck or 0),
            'issue': 'Too many stuck victory conditions' if (stuck or 0) >= 5 else None
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            conditions = await conn.fetch("""
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1
                ORDER BY progress DESC
            """, conflict_id)
            achieved = await conn.fetch("""
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1 AND is_achieved = true
            """, conflict_id)
        
        return {
            'total_conditions': len(conditions),
            'achieved_conditions': len(achieved),
            'leading_condition': dict(conditions[0]) if conditions else None,
            'victory_paths': [
                {
                    'stakeholder_id': c['stakeholder_id'],
                    'victory_type': c['victory_type'],
                    'progress': float(c['progress'] or 0.0)
                }
                for c in conditions[:3]
            ]
        }
    
    async def get_state(self) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            active = await conn.fetchval("""
                SELECT COUNT(DISTINCT conflict_id) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
                  AND conflict_id IN (
                      SELECT id FROM Conflicts
                      WHERE is_active = true
                  )
            """, self.user_id, self.conversation_id)
            near_victory = await conn.fetchval("""
                SELECT COUNT(*) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
                  AND progress > 0.8 AND is_achieved = false
            """, self.user_id, self.conversation_id)
        
        return {
            'conflicts_with_conditions': int(active or 0),
            'conditions_near_victory': int(near_victory or 0)
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        if scene_context.get('conflict_active'):
            return True
        async with get_db_connection_context() as conn:
            near = await conn.fetchval("""
                SELECT COUNT(*) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
                  AND progress > 0.7 AND is_achieved = false
            """, self.user_id, self.conversation_id)
        return (near or 0) > 0
    
    # ========== Agent Properties ==========
    
    @property
    def victory_generator(self) -> Agent:
        if self._victory_generator is None:
            self._victory_generator = Agent(
                name="Victory Condition Generator",
                instructions="""
                Generate nuanced victory conditions for conflicts.
                """,
                model=WARMUP_MODEL,
            )
        return self._victory_generator
    
    @property
    def achievement_narrator(self) -> Agent:
        if self._achievement_narrator is None:
            self._achievement_narrator = Agent(
                name="Achievement Narrator",
                instructions="""
                Narrate victory achievements with emotional depth.
                """,
                model="gpt-5-nano",
            )
        return self._achievement_narrator
    
    @property
    def consequence_calculator(self) -> Agent:
        if self._consequence_calculator is None:
            self._consequence_calculator = Agent(
                name="Victory Consequence Calculator",
                instructions="""
                Calculate the ripple effects of victory/defeat.
                """,
                model="gpt-5-nano",
            )
        return self._consequence_calculator
    
    @property
    def epilogue_writer(self) -> Agent:
        if self._epilogue_writer is None:
            self._epilogue_writer = Agent(
                name="Conflict Epilogue Writer",
                instructions="""
                Write epilogues that provide closure while opening new possibilities.
                """,
                model=WARMUP_MODEL,
            )
        return self._epilogue_writer
    
    # ========== Victory Management Methods ==========
    
    async def _generate_initial_conditions(self, conflict_id: int) -> List[VictoryCondition]:
        """Generate initial victory conditions when conflict is created"""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT id, conflict_type FROM Conflicts WHERE id = $1
            """, conflict_id)
            stakeholders = await conn.fetch("""
                SELECT stakeholder_id, role FROM stakeholders
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, self.user_id, self.conversation_id)
        
        if not conflict or not stakeholders:
            return []
        
        stakeholder_data = [
            {
                'id': int(s['stakeholder_id']),
                'role': (s.get('role') or 'participant'),
                'involvement': 'primary'
            }
            for s in stakeholders
        ]
        
        return await self.generate_victory_conditions(
            conflict_id,
            conflict['conflict_type'],
            stakeholder_data
        )

    def _rows_to_conditions(
        self,
        rows: Sequence[Mapping[str, Any]]
    ) -> List[VictoryCondition]:
        conditions: List[VictoryCondition] = []
        for row in rows:
            data = dict(row)
            requirements = data.get('requirements') or {}
            if isinstance(requirements, str):
                try:
                    requirements = json.loads(requirements)
                except json.JSONDecodeError:
                    requirements = {}
            metadata = conflict_victory_hotpath.get_condition_metadata(data)
            raw_vtype = self._victory_type_value(data.get('victory_type'))
            vtype = raw_vtype.lower()
            try:
                enum_vtype = VictoryType(vtype)
            except Exception:
                enum_vtype = VictoryType.NARRATIVE
            conditions.append(VictoryCondition(
                condition_id=int(data.get('condition_id') or 0),
                conflict_id=int(data.get('conflict_id') or 0),
                stakeholder_id=int(data.get('stakeholder_id') or 0),
                victory_type=enum_vtype,
                description=str(data.get('description') or ''),
                requirements=requirements,
                progress=float(data.get('progress') or 0.0),
                is_achieved=bool(data.get('is_achieved')), 
                achievement_impact=dict(data.get('achievement_impact') or {}),
                task_metadata=metadata,
            ))
        return conditions

    def _victory_type_value(self, raw: Any) -> str:
        if isinstance(raw, VictoryType):
            return raw.value
        if isinstance(raw, str):
            return raw
        return str(raw or VictoryType.NARRATIVE.value)
    
    async def generate_victory_conditions(
        self,
        conflict_id: int,
        conflict_type: str,
        stakeholders: List[Dict[str, Any]]
    ) -> List[VictoryCondition]:
        return await self._generate_victory_conditions(conflict_id, conflict_type, stakeholders)

    async def _generate_victory_conditions(
        self,
        conflict_id: int,
        conflict_type: str,
        stakeholders: List[Dict[str, Any]]
    ) -> List[VictoryCondition]:
        """Generate dynamic victory conditions for each stakeholder."""
        conditions: List[VictoryCondition] = []

        for stakeholder in stakeholders:
            stakeholder_id = int(stakeholder.get('id') or 0)
            async with get_db_connection_context() as conn:
                existing_rows = await conn.fetch(
                    """
                    SELECT * FROM victory_conditions
                    WHERE conflict_id = $1 AND stakeholder_id = $2
                    ORDER BY condition_id
                    """,
                    conflict_id,
                    stakeholder_id,
                )

            if existing_rows:
                conditions.extend(self._rows_to_conditions(existing_rows))
                continue

            fallback_conditions = conflict_victory_hotpath.fallback_victory_conditions(
                conflict_type,
                stakeholder,
            )
            inserted_ids: List[int] = []

            async with get_db_connection_context() as conn:
                for fallback in fallback_conditions:
                    condition_id = await conn.fetchval(
                        """
                        INSERT INTO victory_conditions
                            (user_id, conversation_id, conflict_id, stakeholder_id,
                             victory_type, description, requirements, progress, is_achieved,
                             created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 0.0, false, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING condition_id
                        """,
                        self.user_id,
                        self.conversation_id,
                        conflict_id,
                        stakeholder_id,
                        self._victory_type_value(fallback.get('victory_type')),
                        str(fallback.get('description') or ''),
                        json.dumps(fallback.get('requirements') or {}),
                    )
                    inserted_ids.append(int(condition_id or 0))

            if not inserted_ids:
                continue

            task_payload = {
                'condition_ids': inserted_ids,
                'conflict_id': conflict_id,
                'conflict_type': conflict_type,
                'stakeholder': stakeholder,
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
            }
            task_id = conflict_victory_hotpath.enqueue_task(
                'nyx.tasks.background.conflict_victory.generate_victory_conditions',
                task_payload,
            )

            metadata_updates: List[tuple[int, Dict[str, Any]]] = []
            for idx, condition_id in enumerate(inserted_ids):
                fallback_data = fallback_conditions[min(idx, len(fallback_conditions) - 1)]
                entry = conflict_victory_hotpath.build_entry(
                    'queued' if task_id else 'fallback',
                    task_id=task_id,
                    result=fallback_data,
                )
                metadata_updates.append((condition_id, entry))

            if metadata_updates:
                await conflict_victory_hotpath.write_many_condition_metadata(
                    metadata_updates,
                    conflict_victory_hotpath.TaskKey.GENERATOR,
                )

            async with get_db_connection_context() as conn:
                new_rows = await conn.fetch(
                    """
                    SELECT * FROM victory_conditions
                    WHERE condition_id = ANY($1::int[])
                    ORDER BY condition_id
                    """,
                    inserted_ids,
                )
            conditions.extend(self._rows_to_conditions(new_rows))

        return conditions
    
    async def check_victory_conditions(
        self,
        conflict_id: int,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Check if any victory conditions have been met (single-connection fix)"""
        achievements: List[Dict[str, Any]] = []

        state_snapshot = current_state or {}

        async with get_db_connection_context() as conn:
            conditions = await conn.fetch("""
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1 AND is_achieved = false
            """, conflict_id)

            for condition in conditions:
                condition_data = dict(condition)
                try:
                    requirements = condition_data.get('requirements')
                    if isinstance(requirements, str):
                        requirements = json.loads(requirements or '{}')
                except Exception:
                    requirements = {}

                progress = self._calculate_progress(requirements, state_snapshot)

                # Update progress
                await conn.execute("""
                    UPDATE victory_conditions
                    SET progress = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE condition_id = $2
                """, float(progress), int(condition_data.get('condition_id') or 0))

                if progress >= 1.0:
                    metadata = conflict_victory_hotpath.get_condition_metadata(condition_data)
                    condition_data['requirements'] = requirements
                    condition_data['task_metadata'] = metadata
                    condition_data['progress'] = float(progress)
                    achievement = await self._process_victory_achievement(condition_data, state_snapshot)
                    # Enrich with a simple DTO shape
                    achievement['id'] = int(condition['condition_id'])
                    achievement['name'] = f"{str(condition['victory_type']).title()} achieved"
                    achievement['description'] = str(condition.get('description', '') or '')
                    achievement['points'] = 10.0
                    achievements.append(achievement)
                    
                    # Mark as achieved
                    await conn.execute("""
                        UPDATE victory_conditions
                        SET is_achieved = true, achieved_at = CURRENT_TIMESTAMP
                        WHERE condition_id = $1
                    """, int(condition['condition_id']))
        
        return achievements
    
    def _calculate_progress(
        self,
        requirements: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """Calculate progress toward victory requirements"""
        total_progress = 0.0
        requirement_count = 0
        
        specific = (requirements or {}).get('specific', {}) or {}
        for key, target in specific.items():
            if key in (current_state or {}):
                current = current_state[key]
                if isinstance(target, (int, float)) and float(target) > 0:
                    progress = min(1.0, float(current) / float(target))
                else:
                    progress = 1.0 if current == target else 0.0
                total_progress += progress
                requirement_count += 1
        
        narrative = (requirements or {}).get('narrative', {}) or {}
        if narrative:
            narrative_progress = float((current_state or {}).get('narrative_progress', 0.5))
            total_progress += max(0.0, min(1.0, narrative_progress))
            requirement_count += 1
        
        return (total_progress / requirement_count) if requirement_count > 0 else 0.0
    
    async def _process_victory_achievement(
        self,
        condition: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a victory achievement"""
        vtype_str = self._victory_type_value(condition.get('victory_type')).lower()
        try:
            vtype = VictoryType(vtype_str)
        except Exception:
            vtype = VictoryType.NARRATIVE

        narration = await self._generate_achievement_summary(condition, current_state)
        consequences = await self._calculate_victory_consequences(condition, current_state)

        return {
            'condition_id': int(condition['condition_id']),
            'stakeholder_id': int(condition['stakeholder_id']),
            'victory_type': vtype.value,
            'narration': narration,
            'consequences': consequences,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _generate_achievement_summary(
        self,
        condition: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> str:
        metadata = conflict_victory_hotpath.get_condition_metadata(condition)
        entry = conflict_victory_hotpath.get_task_entry(
            metadata,
            conflict_victory_hotpath.TaskKey.SUMMARY,
        )
        ready = conflict_victory_hotpath.resolved_result(entry)
        if ready is not None:
            return str(ready)

        fallback = conflict_victory_hotpath.fallback_result(entry)
        if fallback is None:
            fallback = conflict_victory_hotpath.fallback_achievement_summary(
                condition,
                current_state,
            )

        condition_id = int(condition.get('condition_id') or 0)
        if condition_id > 0:
            if conflict_victory_hotpath.should_queue_task(entry):
                payload = {
                    'condition_id': condition_id,
                    'victory_type': self._victory_type_value(condition.get('victory_type')),
                    'description': str(condition.get('description') or ''),
                    'current_state': current_state,
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                }
                task_id = conflict_victory_hotpath.enqueue_task(
                    'nyx.tasks.background.conflict_victory.generate_achievement_summary',
                    payload,
                )
                entry = conflict_victory_hotpath.build_entry(
                    'queued' if task_id else 'fallback',
                    task_id=task_id,
                    result=fallback,
                )
                await conflict_victory_hotpath.write_condition_metadata(
                    condition_id,
                    conflict_victory_hotpath.TaskKey.SUMMARY,
                    entry,
                )
            elif not entry or entry.get('result') is None:
                status = entry.get('status', 'fallback') if entry else 'fallback'
                entry = conflict_victory_hotpath.build_entry(
                    status,
                    task_id=entry.get('task_id') if entry else None,
                    result=fallback,
                )
                await conflict_victory_hotpath.write_condition_metadata(
                    condition_id,
                    conflict_victory_hotpath.TaskKey.SUMMARY,
                    entry,
                )

        return fallback

    async def calculate_victory_consequences(
        self,
        condition: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._calculate_victory_consequences(condition, current_state)

    async def _calculate_victory_consequences(
        self,
        condition: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        metadata = conflict_victory_hotpath.get_condition_metadata(condition)
        entry = conflict_victory_hotpath.get_task_entry(
            metadata,
            conflict_victory_hotpath.TaskKey.CONSEQUENCES,
        )
        ready = conflict_victory_hotpath.resolved_result(entry)
        if ready is not None:
            return dict(ready)

        fallback = conflict_victory_hotpath.fallback_result(entry)
        if fallback is None:
            fallback = conflict_victory_hotpath.fallback_victory_consequences(
                condition,
                current_state,
            )

        condition_id = int(condition.get('condition_id') or 0)
        if condition_id > 0:
            if conflict_victory_hotpath.should_queue_task(entry):
                payload = {
                    'condition_id': condition_id,
                    'victory_type': self._victory_type_value(condition.get('victory_type')),
                    'description': str(condition.get('description') or ''),
                    'current_state': current_state,
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                }
                task_id = conflict_victory_hotpath.enqueue_task(
                    'nyx.tasks.background.conflict_victory.calculate_victory_consequences',
                    payload,
                )
                entry = conflict_victory_hotpath.build_entry(
                    'queued' if task_id else 'fallback',
                    task_id=task_id,
                    result=fallback,
                )
                await conflict_victory_hotpath.write_condition_metadata(
                    condition_id,
                    conflict_victory_hotpath.TaskKey.CONSEQUENCES,
                    entry,
                )
            elif not entry or entry.get('result') is None:
                status = entry.get('status', 'fallback') if entry else 'fallback'
                entry = conflict_victory_hotpath.build_entry(
                    status,
                    task_id=entry.get('task_id') if entry else None,
                    result=fallback,
                )
                await conflict_victory_hotpath.write_condition_metadata(
                    condition_id,
                    conflict_victory_hotpath.TaskKey.CONSEQUENCES,
                    entry,
                )

        return dict(fallback)

    async def generate_conflict_epilogue(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> str:
        return await self._generate_conflict_epilogue(conflict_id, resolution_data)

    async def generate_epilogue(
        self,
        conflict_id: int,
        interactive: bool = False,
        resolution_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        resolution_payload: Dict[str, Any] = dict(resolution_data or {})

        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow(
                """
                SELECT id, conflict_name, conflict_type, description
                FROM Conflicts
                WHERE id = $1
                """,
                conflict_id,
            )
            achievements = await conn.fetch(
                """
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1 AND is_achieved = true
                ORDER BY achieved_at NULLS LAST
                """,
                conflict_id,
            )

        if not conflict:
            return "The conflict fades quietly without a lasting epilogue."

        conflict_dict = dict(conflict)
        achievement_dicts = [dict(a) for a in achievements]
        model_override = INTERACTIVE_MODEL if interactive else WARMUP_MODEL

        prompt = (
            "Write a resonant epilogue for a conflict that has reached resolution.\n\n"
            f"Conflict: {conflict_dict.get('conflict_name', 'Unnamed Conflict')}\n"
            f"Type: {conflict_dict.get('conflict_type', 'unknown')}\n"
            f"Summary: {conflict_dict.get('description', '')}\n"
            f"Achieved conditions: {json.dumps(achievement_dicts, default=str)}\n"
            f"Resolution context: {json.dumps(resolution_payload, default=str)}\n\n"
            "Deliver 2-3 paragraphs that acknowledge triumphs, note lingering tensions, "
            "and tease future possibilities without reopening the conflict."
        )

        epilogue_text = ""
        try:
            response = await llm_gateway.execute(
                LLMRequest(
                    prompt=prompt,
                    agent=self.epilogue_writer,
                    model_override=model_override,
                )
            )
            epilogue_text = extract_runner_response(response).strip()
        except Exception:
            logger.exception(
                "Epilogue generation failed for conflict_id=%s (interactive=%s)",
                conflict_id,
                interactive,
            )

        if not epilogue_text:
            epilogue_text = conflict_victory_hotpath.fallback_conflict_epilogue(
                conflict_dict,
                achievement_dicts,
                resolution_payload,
            )

        return epilogue_text

    async def _generate_conflict_epilogue(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> str:
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT id, conflict_name, conflict_type, description FROM Conflicts WHERE id = $1
            """, conflict_id)
            achievements = await conn.fetch("""
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1 AND is_achieved = true
            """, conflict_id)

        if not conflict:
            return "The conflict fades into memory."

        achievement_dicts = [dict(a) for a in achievements]
        metadata_entries = [
            conflict_victory_hotpath.get_task_entry(
                conflict_victory_hotpath.get_condition_metadata(a_dict),
                conflict_victory_hotpath.TaskKey.EPILOGUE,
            )
            for a_dict in achievement_dicts
        ]
        for entry in metadata_entries:
            ready = conflict_victory_hotpath.resolved_result(entry)
            if ready is not None:
                return str(ready)

        fallback = None
        for entry in metadata_entries:
            cached = conflict_victory_hotpath.fallback_result(entry)
            if cached:
                fallback = cached
                break
        if fallback is None:
            fallback = conflict_victory_hotpath.fallback_conflict_epilogue(
                conflict,
                achievement_dicts,
                resolution_data,
            )

        condition_ids = [
            int(a_dict.get('condition_id') or 0)
            for a_dict in achievement_dicts
            if int(a_dict.get('condition_id') or 0) > 0
        ]

        if condition_ids:
            requires_queue = any(
                conflict_victory_hotpath.should_queue_task(entry)
                for entry in metadata_entries
            ) or not metadata_entries

            if requires_queue:
                payload = {
                    'condition_ids': condition_ids,
                    'conflict_id': conflict_id,
                    'conflict': dict(conflict),
                    'achievements': achievement_dicts,
                    'resolution': resolution_data,
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                }
                task_id = conflict_victory_hotpath.enqueue_task(
                    'nyx.tasks.background.conflict_victory.generate_conflict_epilogue',
                    payload,
                )
                entry = conflict_victory_hotpath.build_entry(
                    'queued' if task_id else 'fallback',
                    task_id=task_id,
                    result=fallback,
                )
                await conflict_victory_hotpath.write_many_condition_metadata(
                    [(cid, entry) for cid in condition_ids],
                    conflict_victory_hotpath.TaskKey.EPILOGUE,
                )
            else:
                missing_result = any(
                    (entry.get('result') is None)
                    for entry in metadata_entries
                )
                if missing_result:
                    entry = conflict_victory_hotpath.build_entry(
                        metadata_entries[0].get('status', 'fallback') if metadata_entries else 'fallback',
                        task_id=metadata_entries[0].get('task_id') if metadata_entries else None,
                        result=fallback,
                    )
                    await conflict_victory_hotpath.write_many_condition_metadata(
                        [(cid, entry) for cid in condition_ids],
                        conflict_victory_hotpath.TaskKey.EPILOGUE,
                    )

        return fallback
    
    async def evaluate_partial_victories(
        self,
        conflict_id: int
    ) -> List[Dict[str, Any]]:
        """Evaluate partial victories when conflict ends without clear winner"""
        async with get_db_connection_context() as conn:
            conditions = await conn.fetch("""
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1
                ORDER BY progress DESC
            """, conflict_id)
        
        partial_victories: List[Dict[str, Any]] = []
        for condition in conditions:
            condition_data = dict(condition)
            prog = float(condition_data.get('progress') or 0.0)
            if 0.3 <= prog < 1.0:
                condition_data['task_metadata'] = conflict_victory_hotpath.get_condition_metadata(condition_data)
                condition_data['progress'] = prog
                victory_label = self._victory_type_value(condition_data.get('victory_type'))
                partial_victory = {
                    'stakeholder_id': int(condition_data.get('stakeholder_id') or 0),
                    'victory_type': f"partial_{victory_label}",
                    'progress': prog,
                    'description': f"Partially achieved: {condition_data.get('description','')}",
                    'consolation': await self._generate_consolation(condition_data)
                }
                partial_victories.append(partial_victory)
        return partial_victories
    
    async def _generate_consolation(self, condition: Dict[str, Any]) -> str:
        metadata = conflict_victory_hotpath.get_condition_metadata(condition)
        entry = conflict_victory_hotpath.get_task_entry(
            metadata,
            conflict_victory_hotpath.TaskKey.CONSOLATION,
        )
        ready = conflict_victory_hotpath.resolved_result(entry)
        if ready is not None:
            return str(ready)

        fallback = conflict_victory_hotpath.fallback_result(entry)
        if fallback is None:
            fallback = conflict_victory_hotpath.fallback_consolation(condition)

        condition_id = int(condition.get('condition_id') or 0)
        if condition_id > 0:
            if conflict_victory_hotpath.should_queue_task(entry):
                payload = {
                    'condition_id': condition_id,
                    'victory_type': self._victory_type_value(condition.get('victory_type')),
                    'description': str(condition.get('description') or ''),
                    'progress': float(condition.get('progress') or 0.0),
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                }
                task_id = conflict_victory_hotpath.enqueue_task(
                    'nyx.tasks.background.conflict_victory.generate_consolation',
                    payload,
                )
                entry = conflict_victory_hotpath.build_entry(
                    'queued' if task_id else 'fallback',
                    task_id=task_id,
                    result=fallback,
                )
                await conflict_victory_hotpath.write_condition_metadata(
                    condition_id,
                    conflict_victory_hotpath.TaskKey.CONSOLATION,
                    entry,
                )
            elif not entry or entry.get('result') is None:
                status = entry.get('status', 'fallback') if entry else 'fallback'
                entry = conflict_victory_hotpath.build_entry(
                    status,
                    task_id=entry.get('task_id') if entry else None,
                    result=fallback,
                )
                await conflict_victory_hotpath.write_condition_metadata(
                    condition_id,
                    conflict_victory_hotpath.TaskKey.CONSOLATION,
                    entry,
                )

        return fallback
    
    # ========== Helper Methods ==========
    
    async def _accelerate_victory_progress(self, conflict_id: int):
        """Accelerate victory condition progress during climax"""
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE victory_conditions
                SET progress = LEAST(1.0, progress * 1.2),
                    updated_at = CURRENT_TIMESTAMP
                WHERE conflict_id = $1 AND is_achieved = false
            """, conflict_id)
    
    async def _update_stakeholder_progress(self, stakeholder_id: int, action_type: str):
        """Update victory progress based on stakeholder action"""
        progress_modifiers = {
            'attack': 0.1, 'aggressive': 0.1,
            'defend': 0.05, 'defensive': 0.05,
            'negotiate': 0.15, 'diplomatic': 0.15, 'mediate': 0.12,
            'ally': 0.1, 'supportive': 0.08,
            'betray': 0.2,
            'retreat': -0.1, 'withdraw': -0.1, 'evasive': -0.05
        }
        modifier = float(progress_modifiers.get(action_type, 0.05))
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE victory_conditions
                SET progress = LEAST(1.0, GREATEST(0.0, progress + $1)),
                    updated_at = CURRENT_TIMESTAMP
                WHERE stakeholder_id = $2 AND is_achieved = false
            """, modifier, int(stakeholder_id or 0))


# ===============================================================================
# PUBLIC API FUNCTIONS (via orchestrator)
# ===============================================================================

@function_tool
async def check_for_victory(
    ctx: RunContextWrapper,
    conflict_id: int,
) -> CheckVictoryResponse:
    """Check if any victory conditions have been met"""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    conflict_state = await synthesizer.get_conflict_state(conflict_id)

    event = SystemEvent(
        event_id=f"check_victory_{conflict_id}_{datetime.now().timestamp()}",
        event_type=EventType.CONFLICT_UPDATED,
        source_subsystem=SubsystemType.FLOW,
        payload={'conflict_id': conflict_id, 'state': conflict_state},
        target_subsystems={SubsystemType.VICTORY},
        requires_response=True,
        priority=3,
    )

    achievements: List[AchievementDTO] = []
    epilogue = ""
    victory_achieved = False

    responses = await synthesizer.emit_event(event)
    if responses:
        for r in responses:
            if r.subsystem == SubsystemType.VICTORY:
                data: Dict = r.data or {}
                raw_achs = data.get('achievements') or []
                for a in raw_achs:
                    achievements.append({
                        'id': int(a.get('id', 0) or 0),
                        'name': str(a.get('name', "")),
                        'description': str(a.get('description', "")),
                        'points': float(a.get('points', 0.0) or 0.0),
                    })
                victory_achieved = bool(data.get('victories_achieved', 0) or achievements)

    if victory_achieved:
        ep_event = SystemEvent(
            event_id=f"epilogue_{conflict_id}_{datetime.now().timestamp()}",
            event_type=EventType.CONFLICT_RESOLVED,
            source_subsystem=SubsystemType.VICTORY,
            payload={
                'conflict_id': conflict_id,
                'achievements': achievements,
                'request': 'generate_epilogue'
            },
            target_subsystems={SubsystemType.VICTORY},
            requires_response=True,
            priority=3,
        )
        ep_responses = await synthesizer.emit_event(ep_event)
        if ep_responses:
            for r in ep_responses:
                if r.subsystem == SubsystemType.VICTORY:
                    epilogue = str((r.data or {}).get('epilogue', ""))

    partial_victories: List[PartialVictoryDTO] = []
    if not victory_achieved:
        pv_event = SystemEvent(
            event_id=f"partial_victories_{conflict_id}_{datetime.now().timestamp()}",
            event_type=EventType.STATE_SYNC,
            source_subsystem=SubsystemType.VICTORY,
            payload={'request': 'evaluate_partial_victories', 'conflict_id': conflict_id},
            target_subsystems={SubsystemType.VICTORY},
            requires_response=True,
            priority=4,
        )
        pv_responses = await synthesizer.emit_event(pv_event)
        if pv_responses:
            for r in pv_responses:
                if r.subsystem == SubsystemType.VICTORY:
                    for pv in (r.data or {}).get('partial_victories', []):
                        partial_victories.append({
                            'stakeholder_id': int(pv.get('stakeholder_id', 0) or 0),
                            'description': str(pv.get('description', "")),
                            'progress': float(pv.get('progress', 0.0) or 0.0),
                        })

    return {
        'victory_achieved': victory_achieved,
        'achievements': achievements,
        'partial_victories': partial_victories,
        'epilogue': epilogue,
        'conflict_continues': not victory_achieved,
        'error': "",
    }


@function_tool
async def generate_victory_paths(
    ctx: RunContextWrapper,
    conflict_id: int,
) -> VictoryPathsResponse:
    """Generate possible victory paths for a conflict"""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    async with get_db_connection_context() as conn:
        conflict = await conn.fetchrow(
            "SELECT id, conflict_type FROM Conflicts WHERE id = $1", conflict_id
        )
        stakeholders = await conn.fetch(
            "SELECT stakeholder_id, role FROM stakeholders WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3",
            conflict_id, user_id, conversation_id
        )

    if not conflict:
        return {'conflict_id': conflict_id, 'victory_paths': [], 'error': "Conflict not found"}

    stakeholder_data = [
        {'id': int(s['stakeholder_id']), 'role': str(s.get('role') or 'participant')}
        for s in stakeholders
    ]

    req = SystemEvent(
        event_id=f"victory_paths_{conflict_id}_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.VICTORY,
        payload={
            'request': 'generate_victory_conditions',
            'conflict_id': conflict_id,
            'conflict_type': str(conflict['conflict_type']),
            'stakeholders': stakeholder_data,
        },
        target_subsystems={SubsystemType.VICTORY},
        requires_response=True,
        priority=4,
    )

    paths: List[VictoryPathDTO] = []
    responses = await synthesizer.emit_event(req)
    if responses:
        for r in responses:
            if r.subsystem == SubsystemType.VICTORY:
                for c in (r.data or {}).get('conditions', []):
                    paths.append({
                        'stakeholder': int(c.get('stakeholder_id') or 0),
                        'type': str(c.get('victory_type') or ""),
                        'description': str(c.get('description', "")),
                        'requirements': [str(x) for x in (c.get('requirements') or [])],
                        'current_progress': float(c.get('progress', 0.0) or 0.0),
                    })

    return {'conflict_id': conflict_id, 'victory_paths': paths, 'error': ""}
