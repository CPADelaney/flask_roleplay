# logic/conflict_system/conflict_victory.py
"""
Victory Conditions System with LLM-generated dynamic content
Integrated with ConflictSynthesizer as the central orchestrator
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from agents import Agent, function_tool, RunContextWrapper, Runner
from db.connection import get_db_connection_context

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
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.PHASE_TRANSITION,
            EventType.STAKEHOLDER_ACTION,
            EventType.CONFLICT_RESOLVED,  # for epilogue on demand
            EventType.STATE_SYNC,         # targeted requests
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
        
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                # Generate initial victory conditions
                conflict_id = event.payload.get('conflict_id')
                if conflict_id:
                    conditions = await self._generate_initial_conditions(conflict_id)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={
                            'victory_conditions_created': len(conditions),
                            'conditions': [c.victory_type.value for c in conditions]
                        },
                        side_effects=[]
                    )
            
            elif event.event_type == EventType.CONFLICT_UPDATED:
                # Check victory conditions
                conflict_id = event.payload.get('conflict_id')
                current_state = event.payload.get('state', {}) or {}
                achievements = await self.check_victory_conditions(conflict_id, current_state)
                
                side_effects = []
                if achievements:
                    # Notify orchestrator of resolution via side effect
                    for achievement in achievements:
                        side_effects.append(SystemEvent(
                            event_id=f"victory_{event.event_id}_{achievement['condition_id']}",
                            event_type=EventType.CONFLICT_RESOLVED,
                            source_subsystem=self.subsystem_type,
                            payload={
                                'conflict_id': conflict_id,
                                'victory_type': achievement['victory_type'],
                                'stakeholder_id': achievement['stakeholder_id'],
                                'resolution_type': 'victory',
                                'context': achievement
                            },
                            priority=2
                        ))
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'victories_achieved': len(achievements), 'achievements': achievements},
                    side_effects=side_effects
                )
            
            elif event.event_type == EventType.PHASE_TRANSITION:
                # Update victory condition progress based on phase
                conflict_id = event.payload.get('conflict_id')
                new_phase = event.payload.get('to_phase') or event.payload.get('phase')
                if new_phase == 'climax':
                    await self._accelerate_victory_progress(conflict_id)
                elif new_phase == 'resolution':
                    partial = await self.evaluate_partial_victories(conflict_id)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'partial_victories': partial},
                        side_effects=[]
                    )
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'phase_acknowledged': True},
                    side_effects=[]
                )
            
            elif event.event_type == EventType.STAKEHOLDER_ACTION:
                stakeholder_id = event.payload.get('stakeholder_id')
                action_type = (event.payload.get('action_type') or '').lower()
                await self._update_stakeholder_progress(stakeholder_id, action_type)
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'progress_updated': True},
                    side_effects=[]
                )
            
            elif event.event_type == EventType.CONFLICT_RESOLVED:
                # Support epilogue generation by targeted request
                if (event.payload or {}).get('request') == 'generate_epilogue':
                    conflict_id = event.payload.get('conflict_id')
                    res_data = {
                        'achievements': event.payload.get('achievements', []),
                        'context': event.payload.get('context', {})
                    }
                    ep = await self.generate_conflict_epilogue(conflict_id, res_data)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'epilogue': ep},
                        side_effects=[]
                    )
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'ack': 'conflict_resolved'},
                    side_effects=[]
                )
            
            elif event.event_type == EventType.STATE_SYNC:
                # Targeted requests routed through orchestrator function tools
                req = (event.payload or {}).get('request')
                if req == 'generate_victory_conditions':
                    cid = int(event.payload.get('conflict_id', 0) or 0)
                    ctype = str(event.payload.get('conflict_type', 'unknown'))
                    stakeholders = event.payload.get('stakeholders', []) or []
                    # This writes to DB and returns dataclasses
                    conds = await self.generate_victory_conditions(cid, ctype, stakeholders)
                    # Coerce to DTOs for the tool
                    out = []
                    for c in conds:
                        out.append({
                            'condition_id': c.condition_id,
                            'stakeholder_id': c.stakeholder_id,
                            'victory_type': c.victory_type.value,
                            'description': c.description,
                            'requirements': list((c.requirements or {}).keys()),
                            'progress': float(c.progress or 0.0),
                        })
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'conditions': out},
                        side_effects=[]
                    )
                if req == 'evaluate_partial_victories':
                    cid = int(event.payload.get('conflict_id', 0) or 0)
                    partial = await self.evaluate_partial_victories(cid)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'partial_victories': partial},
                        side_effects=[]
                    )
                if req == 'generate_epilogue':
                    conflict_id = int(event.payload.get('conflict_id', 0) or 0)
                    res_data = {
                        'achievements': event.payload.get('achievements', []),
                        'context': event.payload.get('context', {})
                    }
                    ep = await self.generate_conflict_epilogue(conflict_id, res_data)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'epilogue': ep},
                        side_effects=[]
                    )
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'status': 'no_action_taken'},
                    side_effects=[]
                )
            
            elif event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check(),
                    side_effects=[]
                )
            
            # Default
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={},
                side_effects=[]
            )
            
        except Exception as e:
            logger.error(f"Victory subsystem error: {e}")
            from logic.conflict_system.conflict_synthesizer import SubsystemResponse
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
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
                model="gpt-5-nano",
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
                model="gpt-5-nano",
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
    
    async def generate_victory_conditions(
        self,
        conflict_id: int,
        conflict_type: str,
        stakeholders: List[Dict[str, Any]]
    ) -> List[VictoryCondition]:
        """Generate dynamic victory conditions for each stakeholder"""
        conditions: List[VictoryCondition] = []
        
        for stakeholder in stakeholders:
            prompt = f"""
Generate victory conditions for this stakeholder:

Conflict Type: {conflict_type}
Stakeholder ID: {stakeholder.get('id')}
Role: {stakeholder.get('role', 'participant')}
Involvement: {stakeholder.get('involvement', 'primary')}

Return JSON:
{{
  "conditions": [
    {{
      "victory_type": "dominance|submission|compromise|pyrrhic|tactical|moral|escape|transformation|stalemate|narrative",
      "description": "What this victory looks like",
      "requirements": {{"specific": {{}}, "narrative": {{}}}},
      "impact": {{"relationship": 0.0, "power": 0.0, "satisfaction": 0.0}}
    }}
  ]
}}
"""
            response = await Runner.run(self.victory_generator, prompt)
            try:
                data = json.loads(extract_runner_response(response))
            except Exception:
                data = {"conditions": []}
            
            async with get_db_connection_context() as conn:
                for cond in data.get('conditions', []):
                    vtype = str(cond.get('victory_type', 'narrative') or 'narrative').lower()
                    desc = str(cond.get('description', ''))
                    reqs = cond.get('requirements') or {}
                    impact = cond.get('impact') or {}
                    
                    condition_id = await conn.fetchval("""
                        INSERT INTO victory_conditions
                            (user_id, conversation_id, conflict_id, stakeholder_id, 
                             victory_type, description, requirements, progress, is_achieved, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 0.0, false, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING condition_id
                    """, self.user_id, self.conversation_id, conflict_id,
                       int(stakeholder.get('id') or 0), vtype, desc, json.dumps(reqs))
                    
                    try:
                        enum_vtype = VictoryType(vtype)
                    except Exception:
                        enum_vtype = VictoryType.NARRATIVE
                    
                    conditions.append(VictoryCondition(
                        condition_id=int(condition_id or 0),
                        conflict_id=conflict_id,
                        stakeholder_id=int(stakeholder.get('id') or 0),
                        victory_type=enum_vtype,
                        description=desc,
                        requirements=reqs,
                        progress=0.0,
                        is_achieved=False,
                        achievement_impact=impact
                    ))
        return conditions
    
    async def check_victory_conditions(
        self,
        conflict_id: int,
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if any victory conditions have been met (single-connection fix)"""
        achievements: List[Dict[str, Any]] = []
        
        async with get_db_connection_context() as conn:
            conditions = await conn.fetch("""
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1 AND is_achieved = false
            """, conflict_id)
            
            for condition in conditions:
                try:
                    requirements = condition['requirements']
                    if isinstance(requirements, str):
                        requirements = json.loads(requirements or '{}')
                except Exception:
                    requirements = {}
                
                progress = self._calculate_progress(requirements, current_state or {})
                
                # Update progress
                await conn.execute("""
                    UPDATE victory_conditions
                    SET progress = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE condition_id = $2
                """, float(progress), int(condition['condition_id']))
                
                if progress >= 1.0:
                    achievement = await self._process_victory_achievement(condition, current_state or {})
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
        vtype_str = str(condition['victory_type']).lower()
        try:
            vtype = VictoryType(vtype_str)
        except Exception:
            vtype = VictoryType.NARRATIVE
        
        narration = await self.generate_victory_narration(
            vtype,
            condition.get('description', ''),
            current_state
        )
        consequences = await self.calculate_victory_consequences(condition, current_state)
        
        return {
            'condition_id': int(condition['condition_id']),
            'stakeholder_id': int(condition['stakeholder_id']),
            'victory_type': vtype.value,
            'narration': narration,
            'consequences': consequences,
            'timestamp': datetime.now().isoformat()
        }
    
    async def generate_victory_narration(
        self,
        victory_type: VictoryType,
        description: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate dynamic victory narration"""
        prompt = f"""
Narrate this victory achievement:

Victory Type: {victory_type.value}
Description: {description}
Context: {json.dumps(context)}
Write a powerful 2-3 paragraph narration.
"""
        response = await Runner.run(self.achievement_narrator, prompt)
        return extract_runner_response(response)
    
    async def calculate_victory_consequences(
        self,
        condition: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate consequences of victory"""
        prompt = f"""
Calculate consequences of this victory:

Victory Type: {condition['victory_type']}
Description: {condition.get('description','')}
Current State: {json.dumps(current_state)}

Return JSON:
{{"immediate": {{}}, "long_term": {{}}, "hidden_consequences": []}}
"""
        response = await Runner.run(self.consequence_calculator, prompt)
        try:
            return json.loads(extract_runner_response(response))
        except Exception:
            return {"immediate": {}, "long_term": {}, "hidden_consequences": []}
    
    async def generate_conflict_epilogue(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> str:
        """Generate an epilogue for resolved conflict"""
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
        
        prompt = f"""
Write an epilogue for this resolved conflict:

Conflict: {conflict['conflict_name']}
Type: {conflict['conflict_type']}
Description: {conflict.get('description','')}
Achievements: {json.dumps([dict(a) for a in achievements])}
Resolution: {json.dumps(resolution_data)}

Write 3-4 paragraphs that feel like the end of a chapter, not the end of the story.
"""
        response = await Runner.run(self.epilogue_writer, prompt)
        return extract_runner_response(response)
    
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
            prog = float(condition['progress'] or 0.0)
            if 0.3 <= prog < 1.0:
                partial_victory = {
                    'stakeholder_id': int(condition['stakeholder_id']),
                    'victory_type': f"partial_{condition['victory_type']}",
                    'progress': prog,
                    'description': f"Partially achieved: {condition.get('description','')}",
                    'consolation': await self._generate_consolation(condition)
                }
                partial_victories.append(partial_victory)
        return partial_victories
    
    async def _generate_consolation(self, condition: Dict[str, Any]) -> str:
        """Generate consolation for partial victory"""
        prompt = f"""
Generate a consolation for this partial victory:

Victory Type: {condition['victory_type']}
Progress: {float(condition.get('progress',0.0))*100:.0f}%
Description: {condition.get('description','')}

Write a single encouraging paragraph.
"""
        response = await Runner.run(self.achievement_narrator, prompt)
        return extract_runner_response(response)
    
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
