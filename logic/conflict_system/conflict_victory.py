# logic/conflict_system/conflict_victory.py
"""
Victory Conditions System with LLM-generated dynamic content
Integrated with ConflictSynthesizer as the central orchestrator
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from logic.conflict_system.dynamic_conflict_template import extract_runner_response


from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
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
        
        # Reference to synthesizer
        self.synthesizer = None
    
    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.VICTORY
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
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
        """Return other subsystems this depends on"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return {
            SubsystemType.STAKEHOLDER,  # Need stakeholder info
            SubsystemType.FLOW,  # Need conflict flow state
            SubsystemType.CANON  # Victories can become canonical
        }
    
    @property
    def event_subscriptions(self) -> Set:
        """Return events this subsystem wants to receive"""
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.PHASE_TRANSITION,
            EventType.STAKEHOLDER_ACTION,
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
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
                        }
                    )
                    
            elif event.event_type == EventType.CONFLICT_UPDATED:
                # Check victory conditions
                conflict_id = event.payload.get('conflict_id')
                current_state = event.payload.get('state', {})
                
                achievements = await self.check_victory_conditions(conflict_id, current_state)
                
                side_effects = []
                if achievements:
                    # Notify synthesizer of victory
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
                    data={
                        'victories_achieved': len(achievements),
                        'achievements': achievements
                    },
                    side_effects=side_effects
                )
                
            elif event.event_type == EventType.PHASE_TRANSITION:
                # Update victory condition progress based on phase
                conflict_id = event.payload.get('conflict_id')
                new_phase = event.payload.get('phase')
                
                if new_phase == 'climax':
                    # Accelerate victory condition checking
                    await self._accelerate_victory_progress(conflict_id)
                elif new_phase == 'resolution':
                    # Check for partial victories
                    partial = await self.evaluate_partial_victories(conflict_id)
                    
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={
                            'partial_victories': partial
                        }
                    )
                    
            elif event.event_type == EventType.STAKEHOLDER_ACTION:
                # Update relevant victory conditions
                stakeholder_id = event.payload.get('stakeholder_id')
                action_type = event.payload.get('action_type')
                
                await self._update_stakeholder_progress(stakeholder_id, action_type)
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'progress_updated': True}
                )
                
            elif event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check()
                )
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={}
            )
            
        except Exception as e:
            logger.error(f"Victory subsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        async with get_db_connection_context() as conn:
            # Check for stuck victory conditions
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
            'healthy': stuck < 5,
            'total_conditions': total,
            'stuck_conditions': stuck,
            'issue': 'Too many stuck victory conditions' if stuck >= 5 else None
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get victory-related data for a specific conflict"""
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
                    'progress': c['progress']
                }
                for c in conditions[:3]
            ]
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of victory system"""
        async with get_db_connection_context() as conn:
            active = await conn.fetchval("""
                SELECT COUNT(DISTINCT conflict_id) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
                AND conflict_id IN (
                    SELECT conflict_id FROM Conflicts
                    WHERE is_active = true
                )
            """, self.user_id, self.conversation_id)
            
            near_victory = await conn.fetchval("""
                SELECT COUNT(*) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
                AND progress > 0.8 AND is_achieved = false
            """, self.user_id, self.conversation_id)
        
        return {
            'conflicts_with_conditions': active,
            'conditions_near_victory': near_victory
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if victory system is relevant to scene"""
        # Victory is relevant when conflicts are active and progressing
        if scene_context.get('conflict_active'):
            return True
            
        # Check if any victories are close
        async with get_db_connection_context() as conn:
            near = await conn.fetchval("""
                SELECT COUNT(*) FROM victory_conditions
                WHERE user_id = $1 AND conversation_id = $2
                AND progress > 0.7 AND is_achieved = false
            """, self.user_id, self.conversation_id)
        
        return near > 0
    
    # ========== Agent Properties ==========
    
    @property
    def victory_generator(self) -> Agent:
        if self._victory_generator is None:
            self._victory_generator = Agent(
                name="Victory Condition Generator",
                instructions="""
                Generate nuanced victory conditions for conflicts.
                
                Create conditions that:
                - Reflect complex power dynamics
                - Allow multiple paths to victory
                - Include psychological victories
                - Consider pyrrhic outcomes
                - Enable narrative satisfaction
                
                Victory isn't always about winning - sometimes it's about
                changing the game, maintaining dignity, or transforming relationships.
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
                
                Create narratives that:
                - Capture the complexity of victory
                - Show character growth
                - Acknowledge costs and consequences
                - Build on established dynamics
                - Set up future developments
                
                Focus on psychological and emotional impact over simple win/loss.
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
                
                Consider:
                - Relationship changes
                - Power balance shifts
                - Reputation impacts
                - Emotional consequences
                - Long-term implications
                - Unintended effects
                
                Victory shapes future dynamics - make consequences meaningful.
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
                
                Create epilogues that:
                - Provide emotional resolution
                - Acknowledge all participants
                - Hint at future dynamics
                - Reflect on lessons learned
                - Set new status quo
                
                The end of one conflict is the seed of future stories.
                """,
                model="gpt-5-nano",
            )
        return self._epilogue_writer
    
    # ========== Victory Management Methods ==========
    
    async def _generate_initial_conditions(self, conflict_id: int) -> List[VictoryCondition]:
        """Generate initial victory conditions when conflict is created"""
        
        # Get conflict and stakeholder info
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            stakeholders = await conn.fetch("""
                SELECT * FROM conflict_stakeholders WHERE conflict_id = $1
            """, conflict_id)
        
        if not conflict or not stakeholders:
            return []
        
        # Format stakeholder data
        stakeholder_data = [
            {
                'id': s['stakeholder_id'],
                'role': s.get('faction', 'participant'),
                'involvement': s.get('involvement_level', 'primary')
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
        
        conditions = []
        
        for stakeholder in stakeholders:
            prompt = f"""
            Generate victory conditions for this stakeholder:
            
            Conflict Type: {conflict_type}
            Stakeholder ID: {stakeholder.get('id')}
            Role: {stakeholder.get('role', 'participant')}
            Involvement: {stakeholder.get('involvement', 'primary')}
            
            Create 2-3 different victory conditions that reflect different
            ways they could "win" - not just defeating others, but achieving
            their deeper goals or maintaining their values.
            
            Return JSON:
            {{
                "conditions": [
                    {{
                        "victory_type": "dominance/submission/compromise/etc",
                        "description": "What this victory looks like",
                        "requirements": {{
                            "specific": "measurable requirements",
                            "narrative": "story requirements"
                        }},
                        "impact": {{
                            "relationship": -1.0 to 1.0,
                            "power": -1.0 to 1.0,
                            "satisfaction": 0.0 to 1.0
                        }},
                        "hidden_cost": "What winning this way costs"
                    }}
                ]
            }}
            """
            
            response = await Runner.run(self.victory_generator, prompt)
            data = json.loads(extract_runner_response(response))
            
            # Store conditions in database
            async with get_db_connection_context() as conn:
                for cond in data['conditions']:
                    condition_id = await conn.fetchval("""
                        INSERT INTO victory_conditions
                        (user_id, conversation_id, conflict_id, stakeholder_id, 
                         victory_type, description, requirements, progress, is_achieved)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 0.0, false)
                        RETURNING condition_id
                    """, self.user_id, self.conversation_id, conflict_id,
                    stakeholder['id'], cond['victory_type'],
                    cond['description'], json.dumps(cond['requirements']))
                    
                    conditions.append(VictoryCondition(
                        condition_id=condition_id,
                        conflict_id=conflict_id,
                        stakeholder_id=stakeholder['id'],
                        victory_type=VictoryType(cond['victory_type']),
                        description=cond['description'],
                        requirements=cond['requirements'],
                        progress=0.0,
                        is_achieved=False,
                        achievement_impact=cond['impact']
                    ))
        
        return conditions
    
    async def check_victory_conditions(
        self,
        conflict_id: int,
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if any victory conditions have been met"""
        
        # Get all conditions for conflict
        async with get_db_connection_context() as conn:
            conditions = await conn.fetch("""
                SELECT * FROM victory_conditions
                WHERE conflict_id = $1 AND NOT is_achieved
            """, conflict_id)
        
        achievements = []
        
        for condition in conditions:
            requirements = json.loads(condition['requirements'])
            progress = self._calculate_progress(requirements, current_state)
            
            # Update progress
            await conn.execute("""
                UPDATE victory_conditions
                SET progress = $1, updated_at = CURRENT_TIMESTAMP
                WHERE condition_id = $2
            """, progress, condition['condition_id'])
            
            if progress >= 1.0:
                # Victory achieved!
                achievement = await self._process_victory_achievement(
                    condition, current_state
                )
                achievements.append(achievement)
                
                # Mark as achieved
                await conn.execute("""
                    UPDATE victory_conditions
                    SET is_achieved = true, achieved_at = CURRENT_TIMESTAMP
                    WHERE condition_id = $1
                """, condition['condition_id'])
        
        return achievements
    
    def _calculate_progress(
        self,
        requirements: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """Calculate progress toward victory requirements"""
        
        total_progress = 0.0
        requirement_count = 0
        
        # Check specific requirements
        specific = requirements.get('specific', {})
        for key, target in specific.items():
            if key in current_state:
                current = current_state[key]
                if isinstance(target, (int, float)):
                    progress = min(1.0, current / target)
                else:
                    progress = 1.0 if current == target else 0.0
                total_progress += progress
                requirement_count += 1
        
        # Check narrative requirements (simplified)
        narrative = requirements.get('narrative', {})
        if narrative:
            # This would normally check story flags
            narrative_progress = current_state.get('narrative_progress', 0.5)
            total_progress += narrative_progress
            requirement_count += 1
        
        return total_progress / requirement_count if requirement_count > 0 else 0.0
    
    async def _process_victory_achievement(
        self,
        condition: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a victory achievement"""
        
        # Generate victory narration
        narration = await self.generate_victory_narration(
            VictoryType(condition['victory_type']),
            condition['description'],
            current_state
        )
        
        # Calculate consequences
        consequences = await self.calculate_victory_consequences(
            condition,
            current_state
        )
        
        return {
            'condition_id': condition['condition_id'],
            'stakeholder_id': condition['stakeholder_id'],
            'victory_type': condition['victory_type'],
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
        
        Create a powerful 2-3 paragraph narration that:
        - Captures the emotional weight of victory
        - Acknowledges what was gained and lost
        - Shows character development
        - Sets up future dynamics
        - Feels earned and meaningful
        
        Focus on psychological and relational impact.
        """
        
        response = await Runner.run(self.achievement_narrator, prompt)
        return response.output
    
    async def calculate_victory_consequences(
        self,
        condition: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate consequences of victory"""
        
        prompt = f"""
        Calculate consequences of this victory:
        
        Victory Type: {condition['victory_type']}
        Description: {condition['description']}
        Current State: {json.dumps(current_state)}
        
        Return JSON with immediate and long-term consequences:
        {{
            "immediate": {{
                "relationship_changes": {{"entity": change_value}},
                "power_shifts": {{"entity": change_value}},
                "emotional_impact": {{"entity": "impact description"}},
                "reputation_change": -1.0 to 1.0
            }},
            "long_term": {{
                "new_dynamics": ["future relationship patterns"],
                "seeds_of_conflict": ["potential future tensions"],
                "character_growth": ["how characters changed"],
                "world_changes": ["how the world shifted"]
            }},
            "hidden_consequences": ["unexpected results that emerge later"]
        }}
        """
        
        response = await Runner.run(self.consequence_calculator, prompt)
        return json.loads(response.output)
    
    async def generate_conflict_epilogue(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> str:
        """Generate an epilogue for resolved conflict"""
        
        # Get conflict details
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
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
        Description: {conflict['description']}
        Achievements: {json.dumps([dict(a) for a in achievements])}
        Resolution: {json.dumps(resolution_data)}
        
        Create a meaningful epilogue that:
        - Provides emotional closure
        - Acknowledges all participants' journeys
        - Reflects on what was learned
        - Hints at how relationships have changed
        - Suggests future possibilities
        - Captures the bittersweet nature of resolution
        
        Write 3-4 paragraphs that feel like the end of a chapter,
        not the end of the story.
        """
        
        response = await Runner.run(self.epilogue_writer, prompt)
        return response.output
    
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
        
        partial_victories = []
        
        for condition in conditions:
            if 0.3 <= condition['progress'] < 1.0:
                partial_victory = {
                    'stakeholder_id': condition['stakeholder_id'],
                    'victory_type': f"partial_{condition['victory_type']}",
                    'progress': condition['progress'],
                    'description': f"Partially achieved: {condition['description']}",
                    'consolation': await self._generate_consolation(condition)
                }
                partial_victories.append(partial_victory)
        
        return partial_victories
    
    async def _generate_consolation(self, condition: Dict[str, Any]) -> str:
        """Generate consolation for partial victory"""
        
        prompt = f"""
        Generate a consolation for this partial victory:
        
        Victory Type: {condition['victory_type']}
        Progress: {condition['progress']:.1%}
        Description: {condition['description']}
        
        Write a single paragraph that:
        - Acknowledges the effort
        - Recognizes partial achievement
        - Suggests growth occurred
        - Leaves dignity intact
        
        Be encouraging but realistic.
        """
        
        response = await Runner.run(self.achievement_narrator, prompt)
        return response.output
    
    # ========== Helper Methods ==========
    
    async def _accelerate_victory_progress(self, conflict_id: int):
        """Accelerate victory condition progress during climax"""
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE victory_conditions
                SET progress = LEAST(1.0, progress * 1.2)
                WHERE conflict_id = $1 AND is_achieved = false
            """, conflict_id)
    
    async def _update_stakeholder_progress(self, stakeholder_id: int, action_type: str):
        """Update victory progress based on stakeholder action"""
        # Different actions contribute differently to victory
        progress_modifiers = {
            'attack': 0.1,
            'defend': 0.05,
            'negotiate': 0.15,
            'ally': 0.1,
            'betray': 0.2,
            'retreat': -0.1
        }
        
        modifier = progress_modifiers.get(action_type, 0.05)
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE victory_conditions
                SET progress = LEAST(1.0, GREATEST(0.0, progress + $1))
                WHERE stakeholder_id = $2 AND is_achieved = false
            """, modifier, stakeholder_id)


# ===============================================================================
# PUBLIC API FUNCTIONS
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

    # Get current conflict state (optional for the VICTORY system to use)
    conflict_state = await synthesizer.get_conflict_state(conflict_id)

    # Ask VICTORY to evaluate
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
                # Coerce whatever shape into our DTO
                for a in raw_achs:
                    achievements.append({
                        'id': int(a.get('id', 0) or 0),
                        'name': str(a.get('name', "")),
                        'description': str(a.get('description', "")),
                        'points': float(a.get('points', 0.0) or 0.0),
                    })
                victory_achieved = bool(data.get('victory_achieved', bool(achievements)))

    # If victory: request epilogue from VICTORY via event
    if victory_achieved:
        ep_event = SystemEvent(
            event_id=f"epilogue_{conflict_id}_{datetime.now().timestamp()}",
            event_type=EventType.CONFLICT_RESOLVED,
            source_subsystem=SubsystemType.VICTORY,
            payload={'conflict_id': conflict_id, 'achievements': achievements, 'request': 'generate_epilogue'},
            target_subsystems={SubsystemType.VICTORY},
            requires_response=True,
            priority=3,
        )
        ep_responses = await synthesizer.emit_event(ep_event)
        if ep_responses:
            for r in ep_responses:
                if r.subsystem == SubsystemType.VICTORY:
                    epilogue = str((r.data or {}).get('epilogue', ""))

    # If not victory: ask for partial victories
    partial_victories: List[PartialVictoryDTO] = []
    if not victory_achieved:
        pv_event = SystemEvent(
            event_id=f"partial_victories_{conflict_id}_{datetime.now().timestamp()}",
            event_type=EventType.CONFLICT_UPDATED,
            source_subsystem=SubsystemType.FLOW,
            payload={'conflict_id': conflict_id, 'request': 'evaluate_partial_victories'},
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

    # Load conflict + stakeholders (keep shapes simple)
    async with get_db_connection_context() as conn:
        conflict = await conn.fetchrow(
            "SELECT conflict_id, conflict_type FROM Conflicts WHERE conflict_id = $1",
            conflict_id
        )
        stakeholders = await conn.fetch(
            "SELECT stakeholder_id, faction FROM conflict_stakeholders WHERE conflict_id = $1",
            conflict_id
        )

    if not conflict:
        return {'conflict_id': conflict_id, 'victory_paths': [], 'error': "Conflict not found"}

    stakeholder_data = [
        {'id': int(s['stakeholder_id']), 'role': str(s.get('faction') or 'participant')}
        for s in stakeholders
    ]

    # Ask VICTORY to generate victory conditions
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
                    # Coerce into our DTO
                    paths.append({
                        'stakeholder': int(
                            c.get('stakeholder_id') or c.get('stakeholder') or 0
                        ),
                        'type': str(
                            (c.get('victory_type') or c.get('type') or "")
                            if not isinstance(c.get('victory_type'), dict)
                            else c['victory_type'].get('value', "")
                        ),
                        'description': str(c.get('description', "")),
                        'requirements': [str(x) for x in (c.get('requirements') or [])],
                        'current_progress': float(c.get('progress', 0.0) or 0.0),
                    })

    return {'conflict_id': conflict_id, 'victory_paths': paths, 'error': ""}
