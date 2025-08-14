# logic/conflict_system/conflict_victory.py
"""
Victory Conditions System with LLM-generated dynamic content
Manages victory conditions, achievements, and resolution narratives
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

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


# ===============================================================================
# VICTORY MANAGER
# ===============================================================================

class ConflictVictoryManager:
    """Manages victory conditions and resolutions with LLM generation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._victory_generator = None
        self._achievement_narrator = None
        self._consequence_calculator = None
        self._epilogue_writer = None
    
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
    
    # ========== Dynamic Victory Generation ==========
    
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
            Stakeholder: {stakeholder.get('name', 'Unknown')}
            Role: {stakeholder.get('role', 'participant')}
            Personality: {json.dumps(stakeholder.get('personality', {}))}
            Goals: {json.dumps(stakeholder.get('goals', []))}
            
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
            data = json.loads(response.output)
            
            # Store conditions in database
            async with get_db_connection_context() as conn:
                for cond in data['conditions']:
                    condition_id = await conn.fetchval("""
                        INSERT INTO victory_conditions
                        (conflict_id, stakeholder_id, victory_type, description,
                         requirements, progress, is_achieved)
                        VALUES ($1, $2, $3, $4, $5, 0.0, false)
                        RETURNING condition_id
                    """, conflict_id, stakeholder['id'], cond['victory_type'],
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
                SET progress = $1
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
            'timestamp': datetime.now()
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


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def check_for_victory(
    ctx: RunContextWrapper,
    conflict_id: int
) -> Dict[str, Any]:
    """Check if any victory conditions have been met"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = ConflictVictoryManager(user_id, conversation_id)
    
    # Get current conflict state
    async with get_db_connection_context() as conn:
        conflict = await conn.fetchrow("""
            SELECT * FROM Conflicts WHERE conflict_id = $1
        """, conflict_id)
    
    current_state = {
        'progress': conflict['progress'],
        'phase': conflict['phase'],
        'intensity': conflict['intensity']
        # Add more state data as needed
    }
    
    # Check victory conditions
    achievements = await manager.check_victory_conditions(conflict_id, current_state)
    
    if achievements:
        # Generate epilogue if conflict is resolved
        epilogue = await manager.generate_conflict_epilogue(
            conflict_id,
            {'achievements': achievements}
        )
        
        return {
            'victory_achieved': True,
            'achievements': achievements,
            'epilogue': epilogue
        }
    else:
        # Check for partial victories
        partial = await manager.evaluate_partial_victories(conflict_id)
        
        return {
            'victory_achieved': False,
            'partial_victories': partial,
            'conflict_continues': True
        }


@function_tool
async def generate_victory_paths(
    ctx: RunContextWrapper,
    conflict_id: int
) -> Dict[str, Any]:
    """Generate possible victory paths for a conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = ConflictVictoryManager(user_id, conversation_id)
    
    # Get conflict and stakeholders
    async with get_db_connection_context() as conn:
        conflict = await conn.fetchrow("""
            SELECT * FROM Conflicts WHERE conflict_id = $1
        """, conflict_id)
        
        stakeholders = await conn.fetch("""
            SELECT cs.*, n.npc_name as name
            FROM conflict_stakeholders cs
            JOIN NPCStats n ON cs.npc_id = n.npc_id
            WHERE cs.conflict_id = $1
        """, conflict_id)
    
    # Generate victory conditions
    conditions = await manager.generate_victory_conditions(
        conflict_id,
        conflict['conflict_type'],
        [dict(s) for s in stakeholders]
    )
    
    return {
        'conflict_id': conflict_id,
        'victory_paths': [
            {
                'stakeholder': c.stakeholder_id,
                'type': c.victory_type.value,
                'description': c.description,
                'requirements': c.requirements,
                'current_progress': c.progress
            }
            for c in conditions
        ]
    }
