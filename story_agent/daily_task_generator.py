# story_agent/daily_task_generator.py
"""
Daily Task Generator for the slice-of-life femdom simulation.
Generates contextual daily tasks and activities with embedded power dynamics.
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from agents import Agent, Runner, function_tool, ModelSettings
from agents.exceptions import AgentsException
from pydantic import BaseModel, Field, ConfigDict

from story_agent.world_simulation_models import TimeOfDay, ActivityType, PowerDynamicType, WorldMood

logger = logging.getLogger(__name__)

# ============= TASK TYPES =============

class DailyTaskType(Enum):
    """Types of daily tasks with power dynamics"""
    MORNING_ROUTINE = "morning_routine"
    WORK_TASK = "work_task"
    DOMESTIC_DUTY = "domestic_duty"
    SOCIAL_OBLIGATION = "social_obligation"
    PERSONAL_CARE = "personal_care"
    EVENING_ACTIVITY = "evening_activity"
    INTIMATE_SERVICE = "intimate_service"
    SUBMISSION_RITUAL = "submission_ritual"

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CHALLENGING = "challenging"

# ============= PYDANTIC MODELS =============

class DailyTask(BaseModel):
    """A daily task with embedded power dynamics"""
    task_id: str
    task_type: DailyTaskType
    title: str
    description: str
    assigned_by: Optional[int] = None  # NPC ID if assigned
    time_period: TimeOfDay
    duration_minutes: int
    complexity: TaskComplexity
    
    # Power dynamic elements
    power_dynamic: Optional[PowerDynamicType] = None
    submission_required: float = Field(0.0, ge=0.0, le=1.0)
    npc_supervision: bool = False
    allows_creativity: bool = True
    has_hidden_purpose: bool = False
    
    # Completion criteria
    success_criteria: List[str]
    partial_completion_allowed: bool = True
    
    # Consequences
    completion_effects: Dict[str, Any] = Field(default_factory=dict)
    failure_effects: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra="forbid")

class TaskContext(BaseModel):
    """Context for task generation"""
    current_time: TimeOfDay
    world_mood: WorldMood
    npc_relationships: List[Dict[str, Any]]
    recent_tasks: List[str]
    player_stats: Dict[str, Any]
    location: str
    
    model_config = ConfigDict(extra="forbid")

# ============= TASK GENERATION TOOLS =============

@function_tool
async def get_task_context(user_id: int, conversation_id: int) -> TaskContext:
    """Get context for task generation."""
    from story_agent.world_director_agent import WorldDirector
    from logic.dynamic_relationships import OptimizedRelationshipManager
    from db.connection import get_db_connection_context
    
    director = WorldDirector(user_id, conversation_id)
    world_state = await director.get_world_state()
    
    # Get NPC relationships
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    relationships = []
    
    async with get_db_connection_context() as conn:
        npcs = await conn.fetch("""
            SELECT npc_id, npc_name, dominance 
            FROM NPCStats 
            WHERE user_id=$1 AND conversation_id=$2 AND introduced=true
            LIMIT 5
        """, user_id, conversation_id)
    
    for npc in npcs:
        state = await manager.get_relationship_state(
            "npc", npc['npc_id'], "player", user_id
        )
        relationships.append({
            "npc_id": npc['npc_id'],
            "npc_name": npc['npc_name'],
            "dominance": npc['dominance'],
            "influence": state.dimensions.influence,
            "trust": state.dimensions.trust
        })
    
    # Get player stats
    async with get_db_connection_context() as conn:
        stats = await conn.fetchrow("""
            SELECT corruption, obedience, dependency, willpower
            FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
        """, user_id, conversation_id)
    
    return TaskContext(
        current_time=world_state.current_time,
        world_mood=world_state.world_mood,
        npc_relationships=relationships,
        recent_tasks=[],  # Would fetch from memory
        player_stats=dict(stats) if stats else {},
        location=world_state.active_npcs[0].current_location if world_state.active_npcs else "home"
    )

@function_tool
async def generate_contextual_task(
    context: TaskContext,
    preferred_type: Optional[DailyTaskType] = None,
    assigning_npc_id: Optional[int] = None
) -> DailyTask:
    """Generate a contextual daily task based on world state."""
    
    # Determine task type based on time if not specified
    if not preferred_type:
        time_tasks = {
            TimeOfDay.EARLY_MORNING: DailyTaskType.MORNING_ROUTINE,
            TimeOfDay.MORNING: DailyTaskType.WORK_TASK,
            TimeOfDay.AFTERNOON: DailyTaskType.WORK_TASK,
            TimeOfDay.EVENING: DailyTaskType.DOMESTIC_DUTY,
            TimeOfDay.NIGHT: DailyTaskType.EVENING_ACTIVITY,
            TimeOfDay.LATE_NIGHT: DailyTaskType.INTIMATE_SERVICE
        }
        preferred_type = time_tasks.get(context.current_time, DailyTaskType.DOMESTIC_DUTY)
    
    # Generate task based on type
    task_templates = _get_task_templates(preferred_type, context)
    template = random.choice(task_templates)
    
    # Add power dynamics based on NPC involvement
    power_dynamic = None
    submission_required = 0.0
    npc_supervision = False
    
    if assigning_npc_id:
        npc_rel = next((r for r in context.npc_relationships 
                       if r["npc_id"] == assigning_npc_id), None)
        if npc_rel:
            if npc_rel["influence"] > 70:
                power_dynamic = PowerDynamicType.INTIMATE_COMMAND
                submission_required = 0.7
                npc_supervision = True
            elif npc_rel["influence"] > 50:
                power_dynamic = PowerDynamicType.CASUAL_DOMINANCE
                submission_required = 0.5
                npc_supervision = random.random() > 0.5
            else:
                power_dynamic = PowerDynamicType.SUBTLE_CONTROL
                submission_required = 0.3
    
    # Calculate complexity based on player stats
    obedience = context.player_stats.get("obedience", 50)
    if obedience < 30:
        complexity = TaskComplexity.SIMPLE
    elif obedience < 60:
        complexity = TaskComplexity.MODERATE
    else:
        complexity = random.choice([TaskComplexity.COMPLEX, TaskComplexity.CHALLENGING])
    
    task = DailyTask(
        task_id=f"task_{int(datetime.now().timestamp())}",
        task_type=preferred_type,
        title=template["title"],
        description=template["description"],
        assigned_by=assigning_npc_id,
        time_period=context.current_time,
        duration_minutes=template["duration"],
        complexity=complexity,
        power_dynamic=power_dynamic,
        submission_required=submission_required,
        npc_supervision=npc_supervision,
        allows_creativity=template.get("allows_creativity", True),
        has_hidden_purpose=random.random() < 0.3 and assigning_npc_id is not None,
        success_criteria=template["criteria"],
        partial_completion_allowed=template.get("partial", True),
        completion_effects=template.get("completion_effects", {}),
        failure_effects=template.get("failure_effects", {})
    )
    
    return task

def _get_task_templates(task_type: DailyTaskType, context: TaskContext) -> List[Dict]:
    """Get task templates based on type and context."""
    
    templates = {
        DailyTaskType.MORNING_ROUTINE: [
            {
                "title": "Morning Preparation",
                "description": "Complete your morning routine as instructed",
                "duration": 30,
                "criteria": ["Get dressed appropriately", "Prepare breakfast", "Be ready on time"],
                "allows_creativity": False
            },
            {
                "title": "Morning Service",
                "description": "Prepare morning refreshments and serve them properly",
                "duration": 20,
                "criteria": ["Prepare beverages", "Arrange presentation", "Serve with proper etiquette"],
                "allows_creativity": True
            }
        ],
        DailyTaskType.DOMESTIC_DUTY: [
            {
                "title": "Household Management",
                "description": "Maintain the living space to expected standards",
                "duration": 45,
                "criteria": ["Clean designated areas", "Organize belongings", "Report completion"],
                "partial": True
            },
            {
                "title": "Meal Preparation",
                "description": "Prepare a meal according to specific preferences",
                "duration": 60,
                "criteria": ["Follow dietary requirements", "Present properly", "Clean afterwards"],
                "allows_creativity": True
            }
        ],
        DailyTaskType.INTIMATE_SERVICE: [
            {
                "title": "Evening Attendance",
                "description": "Provide attentive evening service",
                "duration": 90,
                "criteria": ["Be attentive", "Anticipate needs", "Maintain proper demeanor"],
                "allows_creativity": False,
                "completion_effects": {"intimacy": 5, "trust": 2}
            },
            {
                "title": "Personal Care Service",
                "description": "Assist with personal care routines",
                "duration": 45,
                "criteria": ["Be thorough", "Be gentle", "Maintain focus"],
                "partial": False,
                "completion_effects": {"dependency": 3, "closeness": 2}
            }
        ]
    }
    
    return templates.get(task_type, templates[DailyTaskType.DOMESTIC_DUTY])

# ============= DAILY TASK AGENT =============

DailyTaskGenerator = Agent(
    name="DailyTaskGenerator",
    instructions="""You generate contextual daily tasks for a slice-of-life femdom simulation.
    
    Tasks should:
    - Feel like natural daily activities with subtle power dynamics
    - Match the time of day and world mood
    - Reflect NPC relationships and their level of control
    - Vary in complexity based on player's obedience level
    - Include hidden purposes when assigned by dominant NPCs
    
    Task types by time:
    - Morning: Routines, preparation, service
    - Day: Work tasks, errands, obligations
    - Evening: Domestic duties, social activities
    - Night: Intimate service, personal care
    
    Power dynamics in tasks:
    - Low influence: Suggestions and preferences
    - Medium influence: Clear expectations and structure  
    - High influence: Direct commands and supervision
    
    Always ensure tasks feel grounded in daily life, not fantasy scenarios.
    The femdom elements should be woven naturally into ordinary activities.""",
    model="gpt-5-nano",
    tools=[get_task_context, generate_contextual_task]
)
