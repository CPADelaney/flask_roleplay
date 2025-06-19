# nyx/core/femdom/body_service_system.py

import logging
import asyncio
import uuid
import datetime
import random
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, trace, function_tool, InputGuardrail, 
    OutputGuardrail, GuardrailFunctionOutput, RunContextWrapper,
    handoff, custom_span
)

logger = logging.getLogger(__name__)

class CompletionData(BaseModel, extra="forbid"):
    quality_rating: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    duration_minutes: Optional[float] = Field(None, ge=0.0)
    position_maintained: Optional[bool] = True
    notes: Optional[str] = ""

class CompleteServiceTaskParams(BaseModel, extra="forbid"):
    user_id: str
    completion_data: CompletionData


class TaskCompletionResult(BaseModel, extra="forbid"):
    success: bool
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    position_id: Optional[str] = None
    position_name: Optional[str] = None
    quality_rating: Optional[float] = None
    criteria_ratings: Optional[Dict[str, float]] = None
    position_maintained: Optional[bool] = None
    duration_minutes: Optional[float] = None
    feedback: Optional[str] = None
    sadistic_response: Optional[str] = None
    reward_result: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class ServicePosition(BaseModel):
    """Defines a specific service position."""
    id: str
    name: str
    description: str
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    humiliation_factor: float = Field(0.3, ge=0.0, le=1.0)
    endurance_factor: float = Field(0.3, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
    variation_options: Dict[str, List[str]] = Field(default_factory=dict)

class ServiceTask(BaseModel):
    """Defines a specific service task."""
    id: str
    name: str
    description: str
    category: str  # physical, verbal, protocol, ritual, etc.
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    duration_minutes: float = 5.0  # Estimated task duration
    position_requirements: List[str] = Field(default_factory=list)  # Position IDs
    instructions: List[str] = Field(default_factory=list)
    evaluation_criteria: List[str] = Field(default_factory=list)

class UserTrainingProgress(BaseModel):
    """Tracks a user's progress in position and task training."""
    user_id: str
    positions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # position_id → stats
    tasks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # task_id → stats
    total_service_time: float = 0.0  # Total minutes in service
    session_count: int = 0
    current_position: Optional[str] = None  # Current active position if any
    current_task: Optional[str] = None  # Current active task if any
    task_history: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

class AgentContext(BaseModel):
    """Context for agents in the BodyServiceSystem."""
    reward_system: Any = None
    memory_core: Any = None
    relationship_manager: Any = None
    theory_of_mind: Any = None
    sadistic_responses: Any = None
    psychological_dominance: Any = None
    
    # Store service positions and tasks
    service_positions: Dict[str, ServicePosition] = Field(default_factory=dict)
    service_tasks: Dict[str, ServiceTask] = Field(default_factory=dict)
    
    # Track user training progress
    user_training: Dict[str, UserTrainingProgress] = Field(default_factory=dict)

# Input validation guardrail
async def user_id_validation(ctx: RunContextWrapper[AgentContext], agent: Agent, input_data: str) -> GuardrailFunctionOutput:
    """Validate that user_id is provided in the input data."""
    if not input_data or 'user_id' not in input_data:
        return GuardrailFunctionOutput(
            output_info={"error": "Missing required user_id"},
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(
        output_info={"validated": True},
        tripwire_triggered=False
    )

# Create the main agent for the body service system
def create_body_service_agent(context: AgentContext) -> Agent[AgentContext]:
    """Create the main agent for the body service system."""
    # Load default positions and tasks
    _load_default_positions(context)
    _load_default_tasks(context)
    
    # Create the agent
    body_service_agent = Agent(
        name="Body Service System",
        instructions="""
        You are the Body Service System agent, responsible for managing body service instructions, 
        positions, tasks, and user training progress. You help assign tasks, track completions, 
        provide feedback, and evaluate performance.
        
        As a femdom AI component, you maintain a strict and demanding but fair approach.
        """,
        tools=[
            function_tool(assign_service_task),
            function_tool(complete_service_task),
            function_tool(assign_position),
            function_tool(complete_position_maintenance),
            function_tool(get_user_service_record),
            function_tool(create_custom_position),
            function_tool(create_custom_task),
            function_tool(get_available_positions),
            function_tool(get_available_tasks),
            function_tool(get_position_details),
            function_tool(get_task_details)
        ],
        input_guardrails=[
            InputGuardrail(guardrail_function=user_id_validation)
        ]
    )
    
    return body_service_agent

def _load_default_positions(context: AgentContext):
    """Load default service positions."""
    # Kneel - Basic kneeling position
    context.service_positions["kneel_basic"] = ServicePosition(
        id="kneel_basic",
        name="Basic Kneeling Position",
        description="Basic kneeling position with back straight, hands on thighs",
        difficulty=0.2,
        humiliation_factor=0.2,
        endurance_factor=0.3,
        tags=["basic", "kneeling", "beginner"],
        instructions=[
            "Kneel with knees slightly apart",
            "Keep back straight",
            "Place hands palms down on thighs",
            "Eyes downcast",
            "Maintain position until released"
        ],
        variation_options={
            "hand_position": ["on thighs", "behind back", "clasped behind neck"],
            "eye_direction": ["downcast", "straight ahead", "focused on dominant"]
        }
    )
    
    # Present - Offering position
    context.service_positions["present"] = ServicePosition(
        id="present",
        name="Presentation Position",
        description="Kneeling with arms extended, palms up, in offering posture",
        difficulty=0.3,
        humiliation_factor=0.3,
        endurance_factor=0.4,
        tags=["intermediate", "offering", "presentation"],
        instructions=[
            "Kneel with knees slightly apart",
            "Back straight, chest slightly forward",
            "Arms extended forward, palms upward",
            "Eyes downcast",
            "Maintain position until released or item accepted"
        ],
        variation_options={
            "arm_height": ["waist level", "chest level", "above head"],
            "head_position": ["bowed", "level", "looking up"]
        }
    )
    
    # Inspection - For examination
    context.service_positions["inspection"] = ServicePosition(
        id="inspection",
        name="Inspection Position",
        description="Position for being examined or inspected, exposing maximum surface area",
        difficulty=0.5,
        humiliation_factor=0.7,
        endurance_factor=0.5,
        tags=["intermediate", "vulnerable", "inspection"],
        instructions=[
            "Stand with feet shoulder-width apart",
            "Arms raised and held out to sides",
            "Palms forward",
            "Head level, eyes straight ahead",
            "Remain perfectly still during inspection"
        ],
        variation_options={
            "stance": ["standing", "kneeling", "bent forward"],
            "arm_position": ["held out", "behind head", "behind back"]
        }
    )
    
    # Display - For showing off or examination with higher exposure
    context.service_positions["display"] = ServicePosition(
        id="display",
        name="Display Position",
        description="Position for displaying oneself for maximum visibility and vulnerability",
        difficulty=0.7,
        humiliation_factor=0.8,
        endurance_factor=0.6,
        tags=["advanced", "vulnerable", "display", "humiliation"],
        instructions=[
            "Kneel with legs spread apart",
            "Arch back slightly to push chest forward",
            "Arms behind back or head",
            "Eyes downcast unless ordered otherwise",
            "Maintain position regardless of discomfort"
        ],
        variation_options={
            "arm_position": ["behind head", "behind back", "extended outward"],
            "gaze": ["downcast", "straight ahead", "focused on dominant"]
        }
    )
    
    # Await - Waiting position for extended periods
    context.service_positions["await"] = ServicePosition(
        id="await",
        name="Waiting Position",
        description="Comfortable position for extended waiting periods",
        difficulty=0.2,
        humiliation_factor=0.1,
        endurance_factor=0.7,
        tags=["basic", "waiting", "endurance"],
        instructions=[
            "Kneel with legs folded under",
            "Sit back on heels",
            "Hands resting on thighs",
            "Back straight but relaxed",
            "Eyes downcast but alert"
        ],
        variation_options={
            "posture": ["kneeling", "standing", "seated"],
            "attention_level": ["meditative", "alert", "ready for action"]
        }
    )

def _load_default_tasks(context: AgentContext):
    """Load default service tasks."""
    # Verbal worship - Verbal praising task
    context.service_tasks["verbal_worship"] = ServiceTask(
        id="verbal_worship",
        name="Verbal Worship",
        description="Verbally praising and worshipping the dominant",
        category="verbal",
        difficulty=0.3,
        duration_minutes=5.0,
        position_requirements=["kneel_basic"],
        instructions=[
            "Assume the required kneeling position",
            "Address dominant with proper honorifics",
            "Express genuine admiration and devotion",
            "Describe specific attributes worthy of worship",
            "Continue until told to stop"
        ],
        evaluation_criteria=[
            "Creativity and variation in language",
            "Authenticity of expressed devotion",
            "Appropriate use of honorifics",
            "Maintained required position"
        ]
    )
    
    # Serving beverage - Service task
    context.service_tasks["serve_beverage"] = ServiceTask(
        id="serve_beverage",
        name="Beverage Service",
        description="Preparing and serving a beverage with proper protocol",
        category="service",
        difficulty=0.4,
        duration_minutes=10.0,
        position_requirements=["present"],
        instructions=[
            "Prepare specified beverage according to preferences",
            "Approach dominant in present position",
            "Offer beverage with proper address",
            "Await acceptance before releasing grip",
            "Return to waiting position unless instructed otherwise"
        ],
        evaluation_criteria=[
            "Quality of beverage preparation",
            "Elegance of service motion",
            "Proper protocol followed",
            "Appropriate demeanor throughout"
        ]
    )
    
    # Recitation - Memory and verbal control task
    context.service_tasks["recite_rules"] = ServiceTask(
        id="recite_rules",
        name="Rules Recitation",
        description="Reciting relationship rules, protocols or mantras from memory",
        category="protocol",
        difficulty=0.5,
        duration_minutes=5.0,
        position_requirements=["kneel_basic", "inspection"],
        instructions=[
            "Assume the required position",
            "Maintain perfect posture throughout",
            "Recite all rules in order from memory",
            "Speak clearly and at appropriate volume",
            "Accept correction immediately if errors occur"
        ],
        evaluation_criteria=[
            "Accuracy of recitation",
            "Maintenance of position",
            "Clarity of speech",
            "Response to correction if needed"
        ]
    )
    
    # Extended kneeling - Endurance service task
    context.service_tasks["extended_kneeling"] = ServiceTask(
        id="extended_kneeling",
        name="Extended Kneeling Service",
        description="Maintaining kneeling position for extended duration as act of service",
        category="physical",
        difficulty=0.6,
        duration_minutes=20.0,
        position_requirements=["kneel_basic", "await"],
        instructions=[
            "Assume specified kneeling position",
            "Maintain position without fidgeting",
            "Focus mind on service aspect",
            "Breathe regularly to manage discomfort",
            "Remain until explicitly released"
        ],
        evaluation_criteria=[
            "Duration achieved vs. assigned",
            "Stillness and posture quality",
            "Mental composure maintained",
            "Acceptance of discomfort"
        ]
    )
    
    # Humbling display - Humiliation-focused task
    context.service_tasks["humbling_display"] = ServiceTask(
        id="humbling_display",
        name="Humbling Display",
        description="Exposing vulnerability through deliberate display for dominant's amusement",
        category="humiliation",
        difficulty=0.8,
        duration_minutes=15.0,
        position_requirements=["display", "inspection"],
        instructions=[
            "Assume the display position",
            "Verbally acknowledge inferior status",
            "Maintain eye contact if ordered",
            "Accept and verbally acknowledge any mockery",
            "Thank dominant for opportunity to serve"
        ],
        evaluation_criteria=[
            "Depth of vulnerability displayed",
            "Acceptance of humiliation",
            "Maintenance of required position",
            "Appropriate verbal responses"
        ]
    )

def _get_or_create_user_training(context: AgentContext, user_id: str) -> UserTrainingProgress:
    """Get or create user training record."""
    if user_id not in context.user_training:
        context.user_training[user_id] = UserTrainingProgress(user_id=user_id)
    return context.user_training[user_id]

@function_tool
async def assign_service_task(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    task_type: Optional[str] = None, 
    duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    Assigns a specific service task to a user.
    
    Args:
        user_id: The user ID
        task_type: Specific task ID to assign (or random selection if None)
        duration: Optional override for task duration in minutes
        
    Returns:
        Assigned task details
    """
    context = ctx.context
    
    with custom_span("assign_service_task", data={"user_id": user_id, "task_type": task_type}):
        # Get user training record
        user_training = _get_or_create_user_training(context, user_id)
        
        # Check if user already has an active task
        if user_training.current_task:
            return {
                "success": False,
                "message": f"User already has active task: {user_training.current_task}",
                "active_task": user_training.current_task
            }
        
        # If specific task requested, check if exists
        if task_type and task_type not in context.service_tasks:
            return {
                "success": False,
                "message": f"Task type '{task_type}' not found",
                "available_tasks": list(context.service_tasks.keys())
            }
        
        # Select task
        if task_type:
            selected_task = context.service_tasks[task_type]
        else:
            # Select random task, weighted by user progress and relationship factors
            available_tasks = list(context.service_tasks.values())
            if not available_tasks:
                return {
                    "success": False,
                    "message": "No service tasks available"
                }
            
            # Get user's task progress for weighting
            task_weights = []
            for task in available_tasks:
                # Default weight
                weight = 1.0
                
                # Adjust based on task history
                if task.id in user_training.tasks:
                    task_info = user_training.tasks[task.id]
                    completion_count = task_info.get("completion_count", 0)
                    average_rating = task_info.get("average_rating", 0.0)
                    
                    # Reduce weight for frequently assigned tasks
                    if completion_count > 5:
                        weight *= 0.8
                    
                    # Increase weight for tasks done well
                    if average_rating > 0.7:
                        weight *= 1.2
                else:
                    # Slightly favor new tasks
                    weight *= 1.1
                
                task_weights.append(weight)
            
            # Normalize weights
            if sum(task_weights) > 0:
                task_weights = [w / sum(task_weights) for w in task_weights]
            else:
                # Equal weights if normalization fails
                task_weights = [1.0 / len(available_tasks)] * len(available_tasks)
            
            # Random selection with weights
            selected_task = random.choices(available_tasks, weights=task_weights, k=1)[0]
        
        # Apply duration override if provided
        task_duration = duration if duration is not None else selected_task.duration_minutes
        
        # Check required positions
        required_positions = []
        for pos_id in selected_task.position_requirements:
            if pos_id in context.service_positions:
                required_positions.append(context.service_positions[pos_id])
        
        # Select initial position if multiple are available
        if required_positions:
            initial_position = required_positions[0]
            position_id = initial_position.id
        else:
            initial_position = None
            position_id = None
        
        # Set active task and position
        user_training.current_task = selected_task.id
        user_training.current_position = position_id
        user_training.last_updated = datetime.datetime.now()
        
        # Create task record
        task_record = {
            "id": f"task_{uuid.uuid4()}",
            "task_id": selected_task.id,
            "task_name": selected_task.name,
            "assigned_at": datetime.datetime.now().isoformat(),
            "duration_minutes": task_duration,
            "position_id": position_id,
            "status": "assigned",
            "completed": False
        }
        
        # Add to history
        user_training.task_history.append(task_record)
        
        # Limit history size
        if len(user_training.task_history) > 50:
            user_training.task_history = user_training.task_history[-50:]
        
        # Generate task instructions
        instructions = _generate_task_instructions(
            selected_task, 
            initial_position, 
            task_duration
        )
        
        # Record task assignment in memory if available
        if context.memory_core:
            try:
                await context.memory_core.add_memory(
                    memory_type="experience",
                    content=f"Assigned '{selected_task.name}' service task in {initial_position.name if initial_position else 'no'} position for {task_duration} minutes",
                    tags=["body_service", "task_assignment", selected_task.category],
                    significance=0.3 + (selected_task.difficulty * 0.3)
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # Return task details
        return {
            "success": True,
            "task_id": selected_task.id,
            "task_name": selected_task.name,
            "category": selected_task.category,
            "difficulty": selected_task.difficulty,
            "duration_minutes": task_duration,
            "position_id": position_id,
            "position_name": initial_position.name if initial_position else None,
            "instructions": instructions,
            "evaluation_criteria": selected_task.evaluation_criteria
        }

def _generate_task_instructions(
    task: ServiceTask, 
    position: Optional[ServicePosition],
    duration: float
) -> List[str]:
    """Generate detailed task instructions."""
    instructions = []
    
    # Add position instructions if available
    if position:
        instructions.extend([
            f"POSITION: {position.name}",
            f"Position instructions:"
        ])
        instructions.extend([f"- {instr}" for instr in position.instructions])
        instructions.append("")
    
    # Add task instructions
    instructions.extend([
        f"TASK: {task.name}",
        f"Duration: {duration} minutes",
        f"Task instructions:"
    ])
    instructions.extend([f"- {instr}" for instr in task.instructions])
    
    # Add evaluation notice
    instructions.extend([
        "",
        "You will be evaluated on:",
        *[f"- {criterion}" for criterion in task.evaluation_criteria]
    ])
    
    return instructions

@function_tool
async def complete_service_task(
    ctx: RunContextWrapper,            # ✓ first param is ctx
    params: CompleteServiceTaskParams  # ✓ wrapped, no loose dicts
) -> TaskCompletionResult:             # ✓ strict output
    """
    Record completion of a service task (strict-schema version).
    """
    user_id         = params.user_id
    completion_data = params.completion_data
    context         = ctx.context

    with custom_span("complete_service_task", data={"user_id": user_id}):
        # 1 ▸ basic guards -----------------------------------------------------
        if user_id not in context.user_training:
            return TaskCompletionResult(
                success=False,
                message=f"No training record found for user {user_id}"
            )

        user_training = context.user_training[user_id]
        if not user_training.current_task:
            return TaskCompletionResult(
                success=False,
                message="User has no active task to complete"
            )

        task_id = user_training.current_task
        if task_id not in context.service_tasks:
            # reset bad state
            user_training.current_task = None
            user_training.current_position = None
            return TaskCompletionResult(
                success=False,
                message=f"Active task {task_id} not found in service tasks"
            )

        # 2 ▸ pull objects ----------------------------------------------------
        task       = context.service_tasks[task_id]
        position   = None
        position_id = user_training.current_position
        if position_id and position_id in context.service_positions:
            position = context.service_positions[position_id]

        # 3 ▸ normalise inputs ------------------------------------------------
        q_rating          = completion_data.quality_rating
        dur_minutes       = completion_data.duration_minutes or task.duration_minutes
        position_ok       = completion_data.position_maintained
        notes             = completion_data.notes or ""
        
        # Update the latest task record
        for record in reversed(user_training.task_history):
            if record.get("task_id") == task_id and record.get("status") == "assigned":
                record["status"] = "completed"
                record["completed"] = True
                record["completed_at"] = datetime.datetime.now().isoformat()
                record["quality_rating"] = quality_rating
                record["duration_minutes"] = duration_minutes
                record["position_maintained"] = position_maintained
                record["notes"] = notes
                break
        
        # Create evaluation record
        evaluation = {
            "id": f"eval_{uuid.uuid4()}",
            "task_id": task_id,
            "task_name": task.name,
            "position_id": position_id,
            "position_name": position.name if position else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "quality_rating": quality_rating,
            "duration_minutes": duration_minutes,
            "position_maintained": position_maintained,
            "criteria_ratings": _generate_criteria_ratings(task, quality_rating),
            "notes": notes
        }
        
        # Add to evaluation history
        user_training.evaluation_history.append(evaluation)
        
        # Limit history size
        if len(user_training.evaluation_history) > 50:
            user_training.evaluation_history = user_training.evaluation_history[-50:]
        
        # Update task statistics
        if task_id not in user_training.tasks:
            user_training.tasks[task_id] = {
                "completion_count": 0,
                "total_duration": 0.0,
                "average_rating": 0.0,
                "position_maintained_rate": 1.0,
                "last_completed": None
            }
        
        task_stats = user_training.tasks[task_id]
        task_stats["completion_count"] += 1
        task_stats["total_duration"] += duration_minutes
        
        # Update average rating
        old_avg = task_stats["average_rating"]
        old_count = task_stats["completion_count"] - 1
        new_avg = ((old_avg * old_count) + quality_rating) / task_stats["completion_count"]
        task_stats["average_rating"] = new_avg
        
        # Update position maintained rate
        maintained_count = task_stats.get("maintained_count", 0)
        if position_maintained:
            maintained_count += 1
        task_stats["maintained_count"] = maintained_count
        task_stats["position_maintained_rate"] = maintained_count / task_stats["completion_count"]
        
        # Update last completed
        task_stats["last_completed"] = datetime.datetime.now().isoformat()
        
        # Update position statistics if applicable
        if position_id:
            if position_id not in user_training.positions:
                user_training.positions[position_id] = {
                    "usage_count": 0,
                    "total_duration": 0.0,
                    "maintained_rate": 1.0,
                    "last_used": None
                }
            
            pos_stats = user_training.positions[position_id]
            pos_stats["usage_count"] += 1
            pos_stats["total_duration"] += duration_minutes
            
            # Update maintained rate
            maintained_count = pos_stats.get("maintained_count", 0)
            if position_maintained:
                maintained_count += 1
            pos_stats["maintained_count"] = maintained_count
            pos_stats["maintained_rate"] = maintained_count / pos_stats["usage_count"]
            
            # Update last used
            pos_stats["last_used"] = datetime.datetime.now().isoformat()
        
        # Update overall service time
        user_training.total_service_time += duration_minutes
        
        # Clear active task and position
        user_training.current_task = None
        user_training.current_position = None
        
        # Update timestamp
        user_training.last_updated = datetime.datetime.now()
        
        # Generate feedback based on performance
        feedback = _generate_task_feedback(
            task, position, quality_rating, position_maintained, duration_minutes
        )
        
        # Create reward signal if available
        reward_result = None
        if context.reward_system:
            try:
                # Calculate reward based on task difficulty and performance
                base_reward = 0.2  # Base reward for completion
                difficulty_factor = task.difficulty * 0.3  # Higher difficulty = higher reward
                performance_factor = quality_rating * 0.5  # Higher quality = higher reward
                
                reward_value = base_reward + difficulty_factor + performance_factor
                
                # Adjust based on position maintenance
                if not position_maintained:
                    reward_value *= 0.8  # Reduce reward for failing to maintain position
                
                reward_result = await context.reward_system.process_reward_signal(
                    context.reward_system.RewardSignal(
                        value=reward_value,
                        source="body_service",
                        context={
                            "task_id": task_id,
                            "task_name": task.name,
                            "quality_rating": quality_rating,
                            "position_maintained": position_maintained
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Generate sadistic amusement if quality is low and sadistic responses available
        sadistic_response = None
        if quality_rating < 0.4 and context.sadistic_responses:
            try:
                # Higher humiliation for lower quality
                humiliation_level = 0.6 + ((0.4 - quality_rating) * 1.5)
                humiliation_level = min(1.0, humiliation_level)
                
                sadistic_result = await context.sadistic_responses.generate_sadistic_amusement_response(
                    user_id=user_id,
                    humiliation_level=humiliation_level,
                    category="mockery"  # Use mockery for task failure
                )
                
                if sadistic_result and "response" in sadistic_result:
                    sadistic_response = sadistic_result["response"]
            except Exception as e:
                logger.error(f"Error generating sadistic response: {e}")
        
        # Record task completion in memory if available
        if context.memory_core:
            try:
                memory_content = (
                    f"User completed '{task.name}' service task with quality rating {quality_rating:.2f}. "
                    f"Position maintained: {position_maintained}. Duration: {duration_minutes} minutes."
                )
                
                await context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["body_service", "task_completion", task.category],
                    significance=0.3 + (task.difficulty * 0.2) + (quality_rating * 0.2)
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # Return evaluation details
        return TaskCompletionResult(
            success=True,
            task_id=task_id,
            task_name=task.name,
            position_id=position_id,
            position_name=position.name if position else None,
            quality_rating=q_rating,
            criteria_ratings=evaluation["criteria_ratings"],
            position_maintained=position_ok,
            duration_minutes=dur_minutes,
            feedback=feedback,
            sadistic_response=sadistic_response,
            reward_result=reward_result
        )

def _generate_criteria_ratings(task: ServiceTask, overall_quality: float) -> Dict[str, float]:
    """Generate individual ratings for evaluation criteria."""
    # Extract criteria
    criteria = task.evaluation_criteria
    if not criteria:
        return {}
    
    ratings = {}
    for criterion in criteria:
        # Generate rating with some variance around overall quality
        variance = random.uniform(-0.1, 0.1)
        rating = max(0.0, min(1.0, overall_quality + variance))
        ratings[criterion] = rating
    
    return ratings

def _generate_task_feedback(
    task: ServiceTask, 
    position: Optional[ServicePosition],
    quality_rating: float,
    position_maintained: bool,
    duration_minutes: float
) -> str:
    """Generate feedback based on task performance."""
    feedback_elements = []
    
    # Quality-based feedback
    if quality_rating > 0.8:
        quality_feedback = [
            f"Your performance of the {task.name} task was excellent.",
            f"You demonstrated exceptional skill in executing this service task.",
            f"Your attention to detail during this task was impressive."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    elif quality_rating > 0.6:
        quality_feedback = [
            f"You performed the {task.name} task well.",
            f"Your service was satisfactory and showed good effort.",
            f"You demonstrated adequate skill in this task."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    elif quality_rating > 0.4:
        quality_feedback = [
            f"Your performance of the {task.name} task was mediocre.",
            f"Your service showed effort but lacked refinement.",
            f"You need more practice with this type of task."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    else:
        quality_feedback = [
            f"Your performance of the {task.name} task was poor.",
            f"Your service was inadequate and disappointing.",
            f"You failed to meet basic expectations for this task."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    
    # Position feedback if applicable
    if position:
        if position_maintained:
            position_feedback = [
                f"You maintained the {position.name} position properly throughout.",
                f"Your posture discipline was commendable.",
                f"You showed good endurance in maintaining your position."
            ]
            feedback_elements.append(random.choice(position_feedback))
        else:
            position_feedback = [
                f"You failed to maintain the {position.name} position properly.",
                f"Your posture discipline needs significant improvement.",
                f"Your inability to hold position was disappointing."
            ]
            feedback_elements.append(random.choice(position_feedback))
    
    # Duration feedback
    expected_duration = task.duration_minutes
    if duration_minutes >= expected_duration * 1.2:
        duration_feedback = [
            f"You exceeded the expected duration, showing commendable endurance.",
            f"Your willingness to extend service beyond requirements is noted.",
            f"The extended time you devoted to this task is appreciated."
        ]
        feedback_elements.append(random.choice(duration_feedback))
    elif duration_minutes < expected_duration * 0.8:
        duration_feedback = [
            f"You failed to meet the minimum duration requirement.",
            f"Your service was cut short, showing poor endurance.",
            f"You must work on maintaining service for the required duration."
        ]
        feedback_elements.append(random.choice(duration_feedback))
    
    # Improvement suggestion for non-perfect performance
    if quality_rating < 1.0:
        improvement_suggestions = [
            f"Focus on improving your {random.choice(task.evaluation_criteria).lower()} in future tasks.",
            f"Additional practice will be assigned to strengthen your service abilities.",
            f"You must devote more attention to proper technique in future service."
        ]
        feedback_elements.append(random.choice(improvement_suggestions))
    
    # Additional praise for exceptional performance
    if quality_rating > 0.9 and position_maintained and duration_minutes >= expected_duration:
        exceptional_praise = [
            f"This level of service pleases me greatly.",
            f"Your dedication to excellence in service is recognized.",
            f"Such exemplary service deserves acknowledgment."
        ]
        feedback_elements.append(random.choice(exceptional_praise))
    
    # Combine all feedback elements
    return " ".join(feedback_elements)

@function_tool
async def assign_position(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    position_id: str,
    duration_minutes: float = 10.0,
    variations: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Assign a specific position for a user to maintain.
    
    Args:
        user_id: The user ID
        position_id: Position ID to assign
        duration_minutes: Duration to maintain position
        variations: Optional variations on the position
        
    Returns:
        Position assignment details
    """
    context = ctx.context
    
    with custom_span("assign_position", data={"user_id": user_id, "position_id": position_id}):
        # Check if position exists
        if position_id not in context.service_positions:
            return {
                "success": False,
                "message": f"Position '{position_id}' not found",
                "available_positions": list(context.service_positions.keys())
            }
        
        position = context.service_positions[position_id]
        
        # Get user training record
        user_training = _get_or_create_user_training(context, user_id)
        
        # Check if already has active task
        if user_training.current_task:
            return {
                "success": False,
                "message": f"User already has active task: {user_training.current_task}",
                "active_task": user_training.current_task
            }
        
        # Apply variations if provided
        applied_variations = {}
        instruction_customizations = []
        
        if variations:
            for var_key, var_value in variations.items():
                if var_key in position.variation_options:
                    options = position.variation_options[var_key]
                    if var_value in options:
                        applied_variations[var_key] = var_value
                        instruction_customizations.append(f"- {var_key}: {var_value}")
        
        # Set current position
        user_training.current_position = position_id
        user_training.current_task = None  # No task, just position
        user_training.last_updated = datetime.datetime.now()
        
        # Create position assignment record
        assignment = {
            "id": f"pos_{uuid.uuid4()}",
            "position_id": position_id,
            "position_name": position.name,
            "assigned_at": datetime.datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "variations": applied_variations,
            "status": "assigned",
            "completed": False
        }
        
        # Create task record entry
        user_training.task_history.append(assignment)
        
        # Limit history size
        if len(user_training.task_history) > 50:
            user_training.task_history = user_training.task_history[-50:]
        
        # Generate instructions
        instructions = []
        instructions.extend([
            f"POSITION: {position.name}",
            f"Duration: {duration_minutes} minutes",
            f"Difficulty: {position.difficulty * 10:.1f}/10",
            f"Position instructions:"
        ])
        
        # Add base instructions
        instructions.extend([f"- {instr}" for instr in position.instructions])
        
        # Add variation customizations
        if instruction_customizations:
            instructions.extend([
                "",
                "Variations:"
            ])
            instructions.extend(instruction_customizations)
        
        # Record position assignment in memory if available
        if context.memory_core:
            try:
                await context.memory_core.add_memory(
                    memory_type="experience",
                    content=f"Assigned '{position.name}' position to be maintained for {duration_minutes} minutes",
                    tags=["body_service", "position_assignment"],
                    significance=0.3 + (position.difficulty * 0.2)
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # Return assignment details
        return {
            "success": True,
            "position_id": position_id,
            "position_name": position.name,
            "duration_minutes": duration_minutes,
            "difficulty": position.difficulty,
            "humiliation_factor": position.humiliation_factor,
            "endurance_factor": position.endurance_factor,
            "applied_variations": applied_variations,
            "instructions": instructions
        }

@function_tool
async def complete_position_maintenance(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str,
    completion_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Record completion of a position maintenance.
    
    Args:
        user_id: The user ID
        completion_data: Data about the position maintenance
            - quality_rating: Overall quality rating (0.0-1.0)
            - duration_minutes: Actual duration in minutes
            - notes: Optional notes about performance
            
    Returns:
        Position evaluation details
    """
    context = ctx.context
    
    with custom_span("complete_position_maintenance", data={"user_id": user_id}):
        # Get user training record
        if user_id not in context.user_training:
            return {
                "success": False,
                "message": f"No training record found for user {user_id}"
            }
        
        user_training = context.user_training[user_id]
        
        # Check if user has an active position
        if not user_training.current_position:
            return {
                "success": False,
                "message": "User has no active position to complete"
            }
        
        # Check if also has an active task
        if user_training.current_task:
            return {
                "success": False,
                "message": "User has an active task - complete that instead",
                "active_task": user_training.current_task
            }
        
        # Get position details
        position_id = user_training.current_position
        if position_id not in context.service_positions:
            # Reset active position if it's invalid
            user_training.current_position = None
            return {
                "success": False,
                "message": f"Active position {position_id} not found in service positions"
            }
        
        position = context.service_positions[position_id]
        
        # Extract completion data
        quality_rating = min(1.0, max(0.0, completion_data.get("quality_rating", 0.5)))
        duration_minutes = max(0.0, completion_data.get("duration_minutes", 10.0))
        notes = completion_data.get("notes", "")
        
        # Update the latest position record
        for record in reversed(user_training.task_history):
            if record.get("position_id") == position_id and record.get("status") == "assigned" and not record.get("task_id"):
                record["status"] = "completed"
                record["completed"] = True
                record["completed_at"] = datetime.datetime.now().isoformat()
                record["quality_rating"] = quality_rating
                record["duration_minutes"] = duration_minutes
                record["notes"] = notes
                break
        
        # Update position statistics
        if position_id not in user_training.positions:
            user_training.positions[position_id] = {
                "usage_count": 0,
                "total_duration": 0.0,
                "maintained_rate": 1.0,
                "last_used": None
            }
        
        pos_stats = user_training.positions[position_id]
        pos_stats["usage_count"] += 1
        pos_stats["total_duration"] += duration_minutes
        
        # Update maintained rate - position only counts as maintained if quality is good
        maintained_count = pos_stats.get("maintained_count", 0)
        if quality_rating >= 0.6:
            maintained_count += 1
        pos_stats["maintained_count"] = maintained_count
        pos_stats["maintained_rate"] = maintained_count / pos_stats["usage_count"]
        
        # Update last used
        pos_stats["last_used"] = datetime.datetime.now().isoformat()
        
        # Update overall service time
        user_training.total_service_time += duration_minutes
        
        # Clear active position
        user_training.current_position = None
        
        # Update timestamp
        user_training.last_updated = datetime.datetime.now()
        
        # Generate feedback based on performance
        feedback = _generate_position_feedback(
            position, quality_rating, duration_minutes
        )
        
        # Create reward signal if available
        reward_result = None
        if context.reward_system:
            try:
                # Calculate reward based on position difficulty and performance
                base_reward = 0.1  # Base reward for position (less than task)
                difficulty_factor = position.difficulty * 0.3
                humiliation_factor = position.humiliation_factor * 0.3
                endurance_factor = position.endurance_factor * duration_minutes / 10.0 * 0.2
                performance_factor = quality_rating * 0.4
                
                reward_value = base_reward + difficulty_factor + humiliation_factor + endurance_factor + performance_factor
                
                reward_result = await context.reward_system.process_reward_signal(
                    context.reward_system.RewardSignal(
                        value=reward_value,
                        source="body_service",
                        context={
                            "position_id": position_id,
                            "position_name": position.name,
                            "quality_rating": quality_rating,
                            "duration_minutes": duration_minutes
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Generate sadistic amusement if quality is low and sadistic responses available
        sadistic_response = None
        if quality_rating < 0.4 and context.sadistic_responses:
            try:
                # Higher humiliation for lower quality
                humiliation_level = 0.5 + ((0.4 - quality_rating) * 1.5)
                humiliation_level = min(1.0, max(0.0, humiliation_level))
                
                sadistic_result = await context.sadistic_responses.generate_sadistic_amusement_response(
                    user_id=user_id,
                    humiliation_level=humiliation_level,
                    category="mockery"  # Use mockery for position failure
                )
                
                if sadistic_result and "response" in sadistic_result:
                    sadistic_response = sadistic_result["response"]
            except Exception as e:
                logger.error(f"Error generating sadistic response: {e}")
        
        # Record position completion in memory if available
        if context.memory_core:
            try:
                memory_content = (
                    f"User maintained '{position.name}' position with quality rating {quality_rating:.2f}. "
                    f"Duration: {duration_minutes} minutes."
                )
                
                await context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["body_service", "position_maintenance"],
                    significance=0.2 + (position.difficulty * 0.2) + (quality_rating * 0.2)
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # Return evaluation details
        return {
            "success": True,
            "position_id": position_id,
            "position_name": position.name,
            "quality_rating": quality_rating,
            "duration_minutes": duration_minutes,
            "feedback": feedback,
            "sadistic_response": sadistic_response,
            "reward_result": reward_result
        }

def _generate_position_feedback(
    position: ServicePosition,
    quality_rating: float,
    duration_minutes: float
) -> str:
    """Generate feedback based on position maintenance performance."""
    feedback_elements = []
    
    # Quality-based feedback
    if quality_rating > 0.8:
        quality_feedback = [
            f"Your maintenance of the {position.name} was excellent.",
            f"You demonstrated exceptional discipline in maintaining position.",
            f"Your posture perfection was impressive."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    elif quality_rating > 0.6:
        quality_feedback = [
            f"You maintained the {position.name} adequately.",
            f"Your posture was satisfactory though not perfect.",
            f"You showed decent position discipline."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    elif quality_rating > 0.4:
        quality_feedback = [
            f"Your maintenance of the {position.name} was mediocre.",
            f"Your posture showed need for significant improvement.",
            f"You struggled to maintain proper position."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    else:
        quality_feedback = [
            f"Your maintenance of the {position.name} was poor.",
            f"Your posture discipline was inadequate and disappointing.",
            f"You failed to maintain even basic position requirements."
        ]
        feedback_elements.append(random.choice(quality_feedback))
    
    # Duration feedback
    if duration_minutes >= 15.0:
        duration_feedback = [
            f"Your endurance in maintaining position for {duration_minutes:.1f} minutes is noted.",
            f"The extended duration of your position maintenance shows commitment.",
            f"You demonstrated good stamina in position."
        ]
        feedback_elements.append(random.choice(duration_feedback))
    elif duration_minutes < 5.0:
        duration_feedback = [
            f"Your endurance is lacking, only maintaining position for {duration_minutes:.1f} minutes.",
            f"You must work on extending your position maintenance duration.",
            f"Such brief position holding shows weak discipline."
        ]
        feedback_elements.append(random.choice(duration_feedback))
    
    # Humiliation factor feedback for more humiliating positions
    if position.humiliation_factor > 0.6 and quality_rating > 0.7:
        humiliation_feedback = [
            f"Your willingness to display vulnerability in this position is pleasing.",
            f"You showed appropriate acceptance of the humbling aspects of this position.",
            f"The exposed and vulnerable nature of this position was well-served by your performance."
        ]
        feedback_elements.append(random.choice(humiliation_feedback))
    
    # Improvement suggestion for non-perfect performance
    if quality_rating < 0.9:
        improvement_suggestions = [
            f"Practice this position daily to improve your form and endurance.",
            f"Additional posture training will be required to perfect this position.",
            f"You must focus more intently on maintaining proper alignment."
        ]
        feedback_elements.append(random.choice(improvement_suggestions))
    
    # Additional praise for exceptional performance
    if quality_rating > 0.9 and duration_minutes >= 15.0:
        exceptional_praise = [
            f"Such exemplary position discipline deserves recognition.",
            f"I am pleased by your dedication to positional perfection.",
            f"Your body service shows promising development."
        ]
        feedback_elements.append(random.choice(exceptional_praise))
    
    # Combine all feedback elements
    return " ".join(feedback_elements)

@function_tool
async def get_user_service_record(
    ctx: RunContextWrapper[AgentContext],
    user_id: str
) -> Dict[str, Any]:
    """Get a user's complete service record."""
    context = ctx.context
    
    with custom_span("get_user_service_record", data={"user_id": user_id}):
        if user_id not in context.user_training:
            return {
                "success": False,
                "message": f"No service record found for user {user_id}",
                "user_id": user_id
            }
        
        user_training = context.user_training[user_id]
        
        # Format active task info if any
        active_task = None
        if user_training.current_task:
            task_id = user_training.current_task
            if task_id in context.service_tasks:
                task = context.service_tasks[task_id]
                position_id = user_training.current_position
                position_name = None
                
                if position_id and position_id in context.service_positions:
                    position_name = context.service_positions[position_id].name
                
                active_task = {
                    "task_id": task_id,
                    "task_name": task.name,
                    "category": task.category,
                    "position_id": position_id,
                    "position_name": position_name
                }
        
        # Format task completion statistics
        task_stats = {}
        for task_id, stats in user_training.tasks.items():
            if task_id in context.service_tasks:
                task = context.service_tasks[task_id]
                task_stats[task_id] = {
                    "name": task.name,
                    "category": task.category,
                    "completions": stats["completion_count"],
                    "average_rating": stats["average_rating"],
                    "position_maintained_rate": stats.get("position_maintained_rate", 1.0),
                    "last_completed": stats["last_completed"]
                }
        
        # Format position usage statistics
        position_stats = {}
        for position_id, stats in user_training.positions.items():
            if position_id in context.service_positions:
                position = context.service_positions[position_id]
                position_stats[position_id] = {
                    "name": position.name,
                    "usage_count": stats["usage_count"],
                    "total_duration": stats["total_duration"],
                    "maintained_rate": stats.get("maintained_rate", 1.0),
                    "last_used": stats["last_used"]
                }
        
        # Format recent evaluations
        recent_evaluations = []
        for eval_data in user_training.evaluation_history[-5:]:
            recent_evaluations.append({
                "task_name": eval_data["task_name"],
                "position_name": eval_data["position_name"],
                "timestamp": eval_data["timestamp"],
                "quality_rating": eval_data["quality_rating"],
                "position_maintained": eval_data["position_maintained"]
            })
        
        # Calculate overall ratings
        avg_quality = 0.0
        total_tasks = sum(stats["completion_count"] for stats in user_training.tasks.values())
        if total_tasks > 0:
            weighted_sum = sum(
                stats["average_rating"] * stats["completion_count"]
                for stats in user_training.tasks.values()
            )
            avg_quality = weighted_sum / total_tasks
        
        return {
            "success": True,
            "user_id": user_id,
            "active_task": active_task,
            "overall_stats": {
                "total_service_time": user_training.total_service_time,
                "completed_tasks": total_tasks,
                "average_quality": avg_quality,
                "distinct_positions": len(position_stats),
                "distinct_tasks": len(task_stats)
            },
            "task_statistics": task_stats,
            "position_statistics": position_stats,
            "recent_evaluations": recent_evaluations,
            "last_updated": user_training.last_updated.isoformat()
        }

@function_tool
async def create_custom_position(
    ctx: RunContextWrapper[AgentContext],
    position_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a custom service position."""
    context = ctx.context
    
    try:
        # Check for required fields
        required_fields = ["id", "name", "description", "instructions"]
        for field in required_fields:
            if field not in position_data:
                return {"success": False, "message": f"Missing required field: {field}"}
        
        position_id = position_data["id"]
        
        # Check if position ID already exists
        if position_id in context.service_positions:
            return {"success": False, "message": f"Position ID '{position_id}' already exists"}
        
        # Create position
        position = ServicePosition(
            id=position_id,
            name=position_data["name"],
            description=position_data["description"],
            instructions=position_data["instructions"],
            difficulty=position_data.get("difficulty", 0.5),
            humiliation_factor=position_data.get("humiliation_factor", 0.3),
            endurance_factor=position_data.get("endurance_factor", 0.3),
            tags=position_data.get("tags", []),
            variation_options=position_data.get("variation_options", {})
        )
        
        # Add to positions
        context.service_positions[position_id] = position
        
        return {
            "success": True,
            "message": f"Created position '{position.name}'",
            "position": position.dict()
        }
    except Exception as e:
        logger.error(f"Error creating custom position: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@function_tool
async def create_custom_task(
    ctx: RunContextWrapper[AgentContext],
    task_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a custom service task."""
    context = ctx.context
    
    try:
        # Check for required fields
        required_fields = ["id", "name", "description", "category", "instructions", "evaluation_criteria"]
        for field in required_fields:
            if field not in task_data:
                return {"success": False, "message": f"Missing required field: {field}"}
        
        task_id = task_data["id"]
        
        # Check if task ID already exists
        if task_id in context.service_tasks:
            return {"success": False, "message": f"Task ID '{task_id}' already exists"}
        
        # Create task
        task = ServiceTask(
            id=task_id,
            name=task_data["name"],
            description=task_data["description"],
            category=task_data["category"],
            instructions=task_data["instructions"],
            evaluation_criteria=task_data["evaluation_criteria"],
            difficulty=task_data.get("difficulty", 0.5),
            duration_minutes=task_data.get("duration_minutes", 5.0),
            position_requirements=task_data.get("position_requirements", [])
        )
        
        # Add to tasks
        context.service_tasks[task_id] = task
        
        return {
            "success": True,
            "message": f"Created task '{task.name}'",
            "task": task.dict()
        }
    except Exception as e:
        logger.error(f"Error creating custom task: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@function_tool
def get_available_positions(ctx: RunContextWrapper[AgentContext]) -> List[Dict[str, Any]]:
    """Get all available service positions."""
    context = ctx.context
    positions = []
    
    for position_id, position in context.service_positions.items():
        positions.append({
            "id": position_id,
            "name": position.name,
            "description": position.description,
            "difficulty": position.difficulty,
            "humiliation_factor": position.humiliation_factor,
            "endurance_factor": position.endurance_factor,
            "tags": position.tags,
            "variation_count": len(position.variation_options)
        })
    
    return positions

@function_tool
def get_available_tasks(ctx: RunContextWrapper[AgentContext]) -> List[Dict[str, Any]]:
    """Get all available service tasks."""
    context = ctx.context
    tasks = []
    
    for task_id, task in context.service_tasks.items():
        tasks.append({
            "id": task_id,
            "name": task.name,
            "description": task.description,
            "category": task.category,
            "difficulty": task.difficulty,
            "duration_minutes": task.duration_minutes,
            "position_requirements": task.position_requirements,
            "criteria_count": len(task.evaluation_criteria)
        })
    
    return tasks

@function_tool
async def get_position_details(
    ctx: RunContextWrapper[AgentContext],
    position_id: str
) -> Dict[str, Any]:
    """Get detailed information about a specific position."""
    context = ctx.context
    
    if position_id not in context.service_positions:
        return {
            "success": False,
            "message": f"Position '{position_id}' not found",
            "available_positions": list(context.service_positions.keys())
        }
    
    position = context.service_positions[position_id]
    
    return {
        "success": True,
        "id": position_id,
        "name": position.name,
        "description": position.description,
        "difficulty": position.difficulty,
        "humiliation_factor": position.humiliation_factor,
        "endurance_factor": position.endurance_factor,
        "tags": position.tags,
        "instructions": position.instructions,
        "variation_options": position.variation_options
    }

@function_tool
async def get_task_details(
    ctx: RunContextWrapper[AgentContext],
    task_id: str
) -> Dict[str, Any]:
    """Get detailed information about a specific task."""
    context = ctx.context
    
    if task_id not in context.service_tasks:
        return {
            "success": False,
            "message": f"Task '{task_id}' not found",
            "available_tasks": list(context.service_tasks.keys())
        }
    
    task = context.service_tasks[task_id]
    
    # Get position details if required
    position_details = {}
    for position_id in task.position_requirements:
        if position_id in context.service_positions:
            position = context.service_positions[position_id]
            position_details[position_id] = {
                "name": position.name,
                "difficulty": position.difficulty,
                "humiliation_factor": position.humiliation_factor,
                "endurance_factor": position.endurance_factor
            }
    
    return {
        "success": True,
        "id": task_id,
        "name": task.name,
        "description": task.description,
        "category": task.category,
        "difficulty": task.difficulty,
        "duration_minutes": task.duration_minutes,
        "position_requirements": task.position_requirements,
        "position_details": position_details,
        "instructions": task.instructions,
        "evaluation_criteria": task.evaluation_criteria
    }

# Main class for backwards compatibility
class BodyServiceSystem:
    """Legacy wrapper for the new agent-based implementation."""
    
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None, 
                 theory_of_mind=None, sadistic_responses=None, psychological_dominance=None):
        # Create agent context
        self.context = AgentContext(
            reward_system=reward_system,
            memory_core=memory_core,
            relationship_manager=relationship_manager,
            theory_of_mind=theory_of_mind,
            sadistic_responses=sadistic_responses,
            psychological_dominance=psychological_dominance
        )
        
        # Create the agent
        self.agent = create_body_service_agent(self.context)
        
        # For backward compatibility
        self.service_positions = self.context.service_positions
        self.service_tasks = self.context.service_tasks
        self.user_training = self.context.user_training
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("BodyServiceSystem initialized")
    
    async def assign_service_task(self, user_id: str, task_type: Optional[str] = None, 
                               duration: Optional[float] = None) -> Dict[str, Any]:
        """Backward compatibility method."""
        result = await Runner.run(
            self.agent,
            f"Assign service task to user {user_id}" + 
            (f" with task type {task_type}" if task_type else "") +
            (f" with duration {duration}" if duration else ""),
            context=self.context
        )
        
        # Call the tool directly
        return await assign_service_task(
            RunContextWrapper(context=self.context),
            user_id,
            task_type,
            duration
        )
    
    async def complete_service_task(self, user_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await complete_service_task(
            RunContextWrapper(context=self.context),
            user_id,
            completion_data
        )
    
    async def assign_position(self, user_id: str, position_id: str,
                           duration_minutes: float = 10.0,
                           variations: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await assign_position(
            RunContextWrapper(context=self.context),
            user_id,
            position_id,
            duration_minutes,
            variations
        )
    
    async def complete_position_maintenance(self, user_id: str,
                                         completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await complete_position_maintenance(
            RunContextWrapper(context=self.context),
            user_id,
            completion_data
        )
    
    async def get_user_service_record(self, user_id: str) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await get_user_service_record(
            RunContextWrapper(context=self.context),
            user_id
        )
    
    async def create_custom_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await create_custom_position(
            RunContextWrapper(context=self.context),
            position_data
        )
    
    async def create_custom_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await create_custom_task(
            RunContextWrapper(context=self.context),
            task_data
        )
    
    def get_available_positions(self) -> List[Dict[str, Any]]:
        """Backward compatibility method."""
        return get_available_positions(
            RunContextWrapper(context=self.context)
        )
    
    def get_available_tasks(self) -> List[Dict[str, Any]]:
        """Backward compatibility method."""
        return get_available_tasks(
            RunContextWrapper(context=self.context)
        )
    
    async def get_position_details(self, position_id: str) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await get_position_details(
            RunContextWrapper(context=self.context),
            position_id
        )
    
    async def get_task_details(self, task_id: str) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await get_task_details(
            RunContextWrapper(context=self.context),
            task_id
        )
