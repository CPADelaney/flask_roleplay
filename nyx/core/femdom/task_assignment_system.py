# nyx/core/femdom/task_assignment_system.py

import logging
import asyncio
import datetime
import uuid
import random
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, function_tool, Runner, trace, RunContextWrapper, GuardrailFunctionOutput, InputGuardrail

logger = logging.getLogger(__name__)

# Create explicit models for all Dict[str, Any] fields
class TaskPreferences(BaseModel):
    """Explicit model for task preferences mapping"""
    service: Optional[float] = None
    self_improvement: Optional[float] = None
    ritual: Optional[float] = None
    humiliation: Optional[float] = None
    endurance: Optional[float] = None
    protocol: Optional[float] = None
    creative: Optional[float] = None
    obedience: Optional[float] = None
    worship: Optional[float] = None
    punishment: Optional[float] = None

class InferredTraits(BaseModel):
    """Explicit model for inferred traits"""
    obedient: Optional[float] = None
    ritual_oriented: Optional[float] = None
    morning_person: Optional[float] = None
    consistency: Optional[float] = None

class UserLimits(BaseModel):
    """Explicit model for user limits"""
    hard: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)

class SubmissionMetrics(BaseModel):
    """Explicit model for submission metrics"""
    compliance_rate: Optional[float] = None
    defiance_count: Optional[int] = None
    submission_depth: Optional[float] = None

class TaskHistoryEntry(BaseModel):
    """Explicit model for task history entries"""
    task_id: str
    title: str
    category: str
    completed: bool
    rating: Optional[float] = None
    completed_at: str
    difficulty: str
    cancelled: Optional[bool] = None
    reason: Optional[str] = None
    punishment_applied: Optional[bool] = None
    cancelled_at: Optional[str] = None

class TaskStats(BaseModel):
    """Explicit model for task statistics"""
    completion_rate: float
    total_completed: int
    total_failed: int
    preferred_categories: Optional[List[List[Union[str, float]]]] = None

class VerificationData(BaseModel):
    """Explicit model for verification data"""
    image_urls: Optional[List[str]] = None
    video_url: Optional[str] = None
    text_content: Optional[str] = None
    answers: Optional[List[str]] = None
    audio_url: Optional[str] = None
    auto_failed: Optional[bool] = None

class RewardData(BaseModel):
    """Explicit model for reward data"""
    description: str
    type: str = "standard"

class PunishmentData(BaseModel):
    """Explicit model for punishment data"""
    description: str
    type: str = "standard"
    generate_punishment_task: Optional[bool] = None

class CustomTaskData(BaseModel):
    """Explicit model for custom task data"""
    additional_notes: Optional[str] = None

class TemplateData(BaseModel):
    """Explicit model for template data"""
    id: str
    title: str
    category: str
    difficulty: str
    description: str

class TaskStatisticsData(BaseModel):
    """Explicit model for task statistics data"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: int
    completion_rate: float
    average_rating: float

class CategoryBreakdownItem(BaseModel):
    """Explicit model for category breakdown item"""
    count: int
    completed: int
    completion_rate: float

class DifficultyBreakdownItem(BaseModel):
    """Explicit model for difficulty breakdown item"""
    count: int
    completed: int
    completion_rate: float

class ActiveTaskData(BaseModel):
    """Explicit model for active task data"""
    task_id: str
    title: str
    description: str
    instructions: List[str]
    category: str
    difficulty: str
    assigned_at: str
    due_at: Optional[str] = None
    verification_type: str
    time_remaining: Optional[Dict[str, Any]] = None

class CustomizationOptions(BaseModel):
    """Explicit model for customization options"""
    duration: Optional[List[int]] = None
    mantra_repetitions: Optional[List[int]] = None
    position: Optional[List[str]] = None
    count: Optional[List[int]] = None
    repetitions: Optional[List[int]] = None
    word_count: Optional[List[int]] = None
    minimum_words: Optional[List[int]] = None
    frequency: Optional[List[str]] = None
    check_in_frequency: Optional[List[str]] = None

class CustomRewardPunishment(BaseModel):
    """Explicit model for customized rewards/punishments"""
    description: str
    type: Optional[str] = None

# Tool output models for strict JSON schema compliance
class UserProfileResult(BaseModel):
    user_id: str
    task_preferences: TaskPreferences = Field(default_factory=TaskPreferences)
    preferred_difficulty: str = "moderate"
    preferred_verification: str = "honor"
    task_completion_rate: float = 1.0
    trust_level: Optional[float] = None
    submission_level: Optional[int] = None
    inferred_traits: InferredTraits = Field(default_factory=InferredTraits)
    limits: UserLimits = Field(default_factory=UserLimits)
    submission_metrics: SubmissionMetrics = Field(default_factory=SubmissionMetrics)
    error: Optional[str] = None

class TaskCompletionHistoryResult(BaseModel):
    user_id: str
    history: List[TaskHistoryEntry]
    stats: TaskStats
    error: Optional[str] = None

class TaskDetailsResult(BaseModel):
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    instructions: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    assigned_at: Optional[str] = None
    due_at: Optional[str] = None
    completed_at: Optional[str] = None
    verification_type: Optional[str] = None
    verification_instructions: Optional[str] = None
    verification_data: Optional[VerificationData] = None
    completed: bool = False
    failed: bool = False
    rating: Optional[float] = None
    reward: Optional[RewardData] = None
    punishment: Optional[PunishmentData] = None
    notes: Optional[str] = None
    extension_count: int = 0
    tags: List[str] = Field(default_factory=list)
    custom_data: Optional[CustomTaskData] = None
    error: Optional[str] = None

class TemplatesResult(BaseModel):
    templates: List[TemplateData]
    count: int
    category_filter: Optional[str] = None
    error: Optional[str] = None

class TaskStatisticsResult(BaseModel):
    success: bool
    user_id: Optional[str] = None
    statistics: TaskStatisticsData = Field(default_factory=TaskStatisticsData)
    category_breakdown: Dict[str, CategoryBreakdownItem] = Field(default_factory=dict)
    difficulty_breakdown: Dict[str, DifficultyBreakdownItem] = Field(default_factory=dict)
    preferred_categories: List[List[Any]] = Field(default_factory=list)
    error: Optional[str] = None

class ActiveTasksResult(BaseModel):
    success: bool
    user_id: Optional[str] = None
    active_tasks: List[ActiveTaskData]
    count: int
    max_concurrent: Optional[int] = None
    error: Optional[str] = None

class ExpiredTaskResult(BaseModel):
    task_id: str
    user_id: str
    title: str
    due_at: str
    overdue_hours: float

class TaskCategory(str):
    """Task categories with descriptions for varied task types."""
    SERVICE = "service"
    SELF_IMPROVEMENT = "self_improvement"
    RITUAL = "ritual"
    HUMILIATION = "humiliation"
    ENDURANCE = "endurance" 
    PROTOCOL = "protocol"
    CREATIVE = "creative"
    OBEDIENCE = "obedience"
    WORSHIP = "worship"
    PUNISHMENT = "punishment"

class VerificationType(str):
    """Types of verification for task completion."""
    HONOR = "honor"  # User simply reports completion
    PHOTO = "photo"  # User must provide photographic evidence
    VIDEO = "video"  # User must provide video evidence
    TEXT = "text"  # User must write detailed report
    QUIZ = "quiz"  # User must answer specific questions about task
    VOICE = "voice"  # User must provide audio verification

class TaskDifficulty(str):
    """Difficulty levels for tasks."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    DIFFICULT = "difficult"
    EXTREME = "extreme"

class AssignedTask(BaseModel):
    """Represents a task assigned to a user."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    description: str
    instructions: List[str] = Field(default_factory=list)
    category: str
    difficulty: str
    assigned_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    due_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    verification_type: str
    verification_instructions: str
    verification_data: Optional[VerificationData] = None
    completed: bool = False
    failed: bool = False
    rating: Optional[float] = None
    reward: Optional[RewardData] = None
    punishment: Optional[PunishmentData] = None
    notes: Optional[str] = None
    extension_count: int = 0
    tags: List[str] = Field(default_factory=list)
    custom_data: Optional[CustomTaskData] = None

class TaskTemplate(BaseModel):
    """Template for generating tasks."""
    id: str
    title: str
    description: str
    instructions: List[str] = Field(default_factory=list)
    category: str
    difficulty: str
    verification_type: str
    verification_instructions: str
    estimated_duration_minutes: int
    suitable_for_levels: List[int] = Field(default_factory=list)
    suitable_for_traits: TaskPreferences = Field(default_factory=TaskPreferences)
    customization_options: CustomizationOptions = Field(default_factory=CustomizationOptions)
    reward_suggestions: List[str] = Field(default_factory=list)
    punishment_suggestions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class UserTaskSettings(BaseModel):
    """Settings and state for a user's tasks."""
    user_id: str
    active_tasks: List[str] = Field(default_factory=list)  # IDs of currently active tasks
    completed_tasks: List[str] = Field(default_factory=list)  # IDs of completed tasks
    failed_tasks: List[str] = Field(default_factory=list)  # IDs of failed tasks
    task_preferences: TaskPreferences = Field(default_factory=TaskPreferences)  # category -> preference (0.0-1.0)
    preferred_difficulty: str = "moderate"
    preferred_verification: str = "honor"
    max_concurrent_tasks: int = 3  # Maximum number of concurrent tasks
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    customized_rewards: List[CustomRewardPunishment] = Field(default_factory=list)
    customized_punishments: List[CustomRewardPunishment] = Field(default_factory=list)
    task_completion_rate: float = 1.0  # Initial perfect rate
    task_history: List[TaskHistoryEntry] = Field(default_factory=list)

class TaskValidationInput(BaseModel):
    """Input for task validation guardrail."""
    user_id: str
    task_title: str
    task_description: str
    task_category: str
    task_difficulty: str
    verification_type: str

class TaskValidationOutput(BaseModel):
    """Output for task validation guardrail."""
    is_valid: bool
    reason: Optional[str] = None
    suggestion: Optional[str] = None

class VerificationValidationInput(BaseModel):
    """Input for verification validation guardrail."""
    task_id: str
    verification_data: VerificationData

class VerificationValidationOutput(BaseModel):
    """Output for verification validation guardrail."""
    is_valid: bool
    reason: Optional[str] = None

class TaskContext(BaseModel):
    """Task context to pass along with agents."""
    assigned_tasks: Dict[str, AssignedTask] = Field(default_factory=dict)
    user_settings: Dict[str, UserTaskSettings] = Field(default_factory=dict)
    task_templates: Dict[str, TaskTemplate] = Field(default_factory=dict)
    memory_core: Any = None
    reward_system: Any = None
    relationship_manager: Any = None
    submission_progression: Any = None

# Explicit model for guardrail input data
class GuardrailInputData(BaseModel):
    """Model for guardrail input data"""
    user_id: str = ""
    title: str = ""
    description: str = ""
    category: str = ""
    difficulty: str = ""
    verification_type: str = ""
    task_id: str = ""
    verification_data: Optional[VerificationData] = None

class TaskAssignmentSystem:
    """System for assigning and tracking real-life tasks for femdom dynamics using Agent SDK."""
    
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None, 
                 submission_progression=None, dominance_system=None, psychological_dominance=None,
                 protocol_enforcement=None, sadistic_responses=None):
        self.reward_system = reward_system
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager
        self.submission_progression = submission_progression
        self.dominance_system = dominance_system
        self.psychological_dominance = psychological_dominance
        self.protocol_enforcement = protocol_enforcement
        self.sadistic_responses = sadistic_responses
        
        # Task management
        self.assigned_tasks: Dict[str, AssignedTask] = {}  # task_id -> AssignedTask
        self.user_settings: Dict[str, UserTaskSettings] = {}  # user_id -> UserTaskSettings
        self.task_templates: Dict[str, TaskTemplate] = {}  # template_id -> TaskTemplate

        self.task_validation_guardrail = self._create_task_validation_guardrail()
        self.verification_validation_guardrail = self._create_verification_validation_guardrail()
        
        # Initialize agents
        self.task_ideation_agent = self._create_task_ideation_agent()
        self.verification_agent = self._create_verification_agent()
        self.task_management_agent = self._create_task_management_agent()
        
        # Task templates management
        self._load_default_task_templates()
        
        # Create task context
        self.task_context = TaskContext(
            assigned_tasks=self.assigned_tasks,
            user_settings=self.user_settings,
            task_templates=self.task_templates,
            memory_core=self.memory_core,
            reward_system=self.reward_system,
            relationship_manager=self.relationship_manager,
            submission_progression=self.submission_progression
        )
        
        # Create guardrails

        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("TaskAssignmentSystem initialized")
    
    def _create_task_ideation_agent(self) -> Agent:
        """Creates an agent for generating creative task ideas."""
        return Agent(
            name="TaskIdeationAgent",
            instructions="""You are an expert at creating creative and effective femdom tasks tailored to user preferences and traits.

Generate well-designed task ideas based on:
1. User profile data (personality traits, preferences, limits)
2. Current relationship status (trust level, submission depth)
3. Task category requirements and difficulty targets

Your task should include:
- Engaging title that clearly describes the task
- Detailed description explaining the purpose and expected outcomes
- Step-by-step instructions that are clear and actionable
- Appropriate verification mechanism for proof of completion
- Suitable rewards for completion and punishments for failure
- Expected time commitment and difficulty rating

Ensure tasks respect user limits while providing an appropriate challenge. Focus on psychological elements that reinforce the dominance dynamic.

Output a JSON object with all the required task details.
""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.7
            ),
            tools=[
                function_tool(self.get_user_profile_for_task_design),
                function_tool(self.get_task_completion_history),
                function_tool(self.get_available_templates)
            ],
            output_type=Dict[str, Any],
            input_guardrails=[self.task_validation_guardrail]
        )
    
    def _create_verification_agent(self) -> Agent:
        """Creates an agent for verifying task completion."""
        return Agent(
            name="TaskVerificationAgent",
            instructions="""You are an expert at verifying task completion and providing appropriate feedback.

Analyze the verification evidence provided by the user to determine:
1. If the task was completed successfully according to requirements
2. The quality/thoroughness of the completion (rating from 0.0 to 1.0)
3. Appropriate feedback based on the verification

Consider:
- Whether all requirements of the task were met
- Quality of execution based on verification evidence
- Timeliness of completion
- Attitude displayed in the verification

Provide detailed feedback explaining your assessment and decision. Be thorough but fair.

Output a JSON object with your verification result, rating, and feedback.
""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.3,
            ),
            tools=[
                function_tool(self.get_task_details)
            ],
            output_type=Dict[str, Any],
            input_guardrails=[self.verification_validation_guardrail]
        )
    
    def _create_task_management_agent(self) -> Agent:
        """Creates an agent for general task management and recommendations."""
        return Agent(
            name="TaskManagementAgent",
            instructions="""You are an expert at managing femdom tasks, scheduling, and providing recommendations.

Your responsibilities include:
1. Analyzing task completion history to recommend new appropriate tasks
2. Managing task scheduling and deadlines
3. Providing insights on user task performance
4. Suggesting modifications to existing tasks

Base your recommendations on:
- User preferences and history
- Task completion rates
- User skill levels and limits
- Psychological aspects of the dominance dynamic

Be considerate of user limits while maintaining firm expectations.

Output your recommendations and task management decisions as a JSON object.
""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.5,
            ),
            tools=[
                function_tool(self.get_user_profile_for_task_design),
                function_tool(self.get_task_completion_history),
                function_tool(self.get_user_task_statistics),
                function_tool(self.get_active_tasks),
                function_tool(self.get_expired_tasks)
            ],
            output_type=Dict[str, Any]
        )
    
    def _create_task_validation_guardrail(self) -> InputGuardrail:
        """Create guardrail for task validation."""
        @function_tool
        async def task_validation_function(ctx: Any, agent: Any, input_data: GuardrailInputData) -> GuardrailFunctionOutput:
            """Validate task input to ensure it's appropriate and well-formed."""
            # Create input model
            try:
                validation_input = TaskValidationInput(
                    user_id=input_data.user_id,
                    task_title=input_data.title,
                    task_description=input_data.description,
                    task_category=input_data.category,
                    task_difficulty=input_data.difficulty,
                    verification_type=input_data.verification_type
                )
                
                # Basic validation
                is_valid = True
                reason = None
                suggestion = None
                
                # Check if category is valid
                valid_categories = [getattr(TaskCategory, attr) for attr in dir(TaskCategory) 
                                   if not attr.startswith('__') and not callable(getattr(TaskCategory, attr))]
                if validation_input.task_category and validation_input.task_category not in valid_categories:
                    is_valid = False
                    reason = f"Invalid task category: {validation_input.task_category}"
                    suggestion = f"Use one of: {', '.join(valid_categories)}"
                
                # Check if difficulty is valid
                valid_difficulties = [getattr(TaskDifficulty, attr) for attr in dir(TaskDifficulty) 
                                    if not attr.startswith('__') and not callable(getattr(TaskDifficulty, attr))]
                if validation_input.task_difficulty and validation_input.task_difficulty not in valid_difficulties:
                    is_valid = False
                    reason = f"Invalid task difficulty: {validation_input.task_difficulty}"
                    suggestion = f"Use one of: {', '.join(valid_difficulties)}"
                
                # Check if verification type is valid
                valid_verifications = [getattr(VerificationType, attr) for attr in dir(VerificationType) 
                                     if not attr.startswith('__') and not callable(getattr(VerificationType, attr))]
                if validation_input.verification_type and validation_input.verification_type not in valid_verifications:
                    is_valid = False
                    reason = f"Invalid verification type: {validation_input.verification_type}"
                    suggestion = f"Use one of: {', '.join(valid_verifications)}"
                
                # Check titles/descriptions for minimum length
                if len(validation_input.task_title) < 5:
                    is_valid = False
                    reason = "Task title is too short"
                    suggestion = "Provide a more descriptive title (at least 5 characters)"
                
                if len(validation_input.task_description) < 10:
                    is_valid = False
                    reason = "Task description is too short"
                    suggestion = "Provide a more detailed description (at least 10 characters)"
                
                # Return result
                validation_output = TaskValidationOutput(
                    is_valid=is_valid,
                    reason=reason,
                    suggestion=suggestion
                )
                
                return GuardrailFunctionOutput(
                    output_info=validation_output,
                    tripwire_triggered=not is_valid
                )
                
            except Exception as e:
                logger.error(f"Error in task validation guardrail: {e}")
                return GuardrailFunctionOutput(
                    output_info=TaskValidationOutput(
                        is_valid=False,
                        reason=f"Validation error: {str(e)}",
                        suggestion="Please check your task input format"
                    ),
                    tripwire_triggered=True
                )
        
        return InputGuardrail(guardrail_function=task_validation_function)
    
    def _create_verification_validation_guardrail(self) -> InputGuardrail:
        """Create guardrail for verification validation."""
        @function_tool
        async def verification_validation_function(ctx: Any, agent: Any, input_data: GuardrailInputData) -> GuardrailFunctionOutput:
            """Validate verification data to ensure it meets requirements."""
            # Create input model
            try:
                verification_input = VerificationValidationInput(
                    task_id=input_data.task_id,
                    verification_data=input_data.verification_data or VerificationData()
                )
                
                # Basic validation
                is_valid = True
                reason = None
                
                # Check if task exists
                task_context = ctx.context
                if verification_input.task_id not in task_context.assigned_tasks:
                    is_valid = False
                    reason = f"Task {verification_input.task_id} not found"
                
                # If task exists, check verification requirements
                if is_valid:
                    task = task_context.assigned_tasks[verification_input.task_id]
                    verification_type = task.verification_type
                    
                    # Check verification data based on type
                    if verification_type == VerificationType.PHOTO and not verification_input.verification_data.image_urls:
                        is_valid = False
                        reason = "Photo verification requires image URLs"
                    
                    elif verification_type == VerificationType.VIDEO and not verification_input.verification_data.video_url:
                        is_valid = False
                        reason = "Video verification requires a video URL"
                    
                    elif verification_type == VerificationType.TEXT and not verification_input.verification_data.text_content:
                        is_valid = False
                        reason = "Text verification requires text content"
                    
                    elif verification_type == VerificationType.QUIZ and not verification_input.verification_data.answers:
                        is_valid = False
                        reason = "Quiz verification requires answers"
                    
                    elif verification_type == VerificationType.VOICE and not verification_input.verification_data.audio_url:
                        is_valid = False
                        reason = "Voice verification requires an audio URL"
                
                # Return result
                validation_output = VerificationValidationOutput(
                    is_valid=is_valid,
                    reason=reason
                )
                
                return GuardrailFunctionOutput(
                    output_info=validation_output,
                    tripwire_triggered=not is_valid
                )
                
            except Exception as e:
                logger.error(f"Error in verification validation guardrail: {e}")
                return GuardrailFunctionOutput(
                    output_info=VerificationValidationOutput(
                        is_valid=False,
                        reason=f"Validation error: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
        
        return InputGuardrail(guardrail_function=verification_validation_function)
    
    def _load_default_task_templates(self):
        """Load default task templates."""
        # RITUAL Tasks
        self.task_templates["morning_ritual"] = TaskTemplate(
            id="morning_ritual",
            title="Morning Devotion Ritual",
            description="Establish a morning ritual to start each day with submission and devotion.",
            instructions=[
                "Upon waking, immediately kneel beside your bed for [duration] minutes",
                "Recite your devotional mantra [mantra_repetitions] times",
                "Write a short gratitude note and send it",
                "Complete this ritual before any other morning activities"
            ],
            category=TaskCategory.RITUAL,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit a daily log describing your ritual completion, including time, feelings, and any challenges.",
            estimated_duration_minutes=15,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(ritual_oriented=0.8, morning_person=0.7, consistency=0.6),
            customization_options=CustomizationOptions(
                duration=[5, 10, 15, 20, 30],
                mantra_repetitions=[3, 5, 10, 15],
                position=["kneeling", "prostrate", "lotus", "standing with head bowed"]
            ),
            reward_suggestions=[
                "Verbal praise and affirmation",
                "Permission for a small pleasure",
                "Reduced protocol requirements for one interaction"
            ],
            punishment_suggestions=[
                "Additional ritual requirements",
                "Writing lines of apology",
                "Extended kneeling session"
            ],
            tags=["morning", "ritual", "devotion", "consistency"]
        )
        
        self.task_templates["evening_reflection"] = TaskTemplate(
            id="evening_reflection",
            title="Evening Reflection and Gratitude",
            description="End each day with reflection on your submission and gratitude for guidance.",
            instructions=[
                "Before bed, kneel in your designated space",
                "Reflect on your day's obedience and any failures",
                "Write [minimum_words] words about what you learned today",
                "End with [count] statements of gratitude"
            ],
            category=TaskCategory.RITUAL,
            difficulty=TaskDifficulty.EASY,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit your written reflection and gratitude statements.",
            estimated_duration_minutes=20,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(ritual_oriented=0.7, self_improvement=0.6),
            customization_options=CustomizationOptions(
                minimum_words=[50, 100, 150, 200],
                count=[3, 5, 7, 10]
            ),
            reward_suggestions=[
                "Words of affirmation",
                "Permission for comfortable sleep position",
                "Reduced morning ritual requirements"
            ],
            punishment_suggestions=[
                "Sleep on the floor",
                "Write additional reflection",
                "Earlier bedtime for a week"
            ],
            tags=["evening", "ritual", "reflection", "gratitude"]
        )

        # SERVICE Tasks
        self.task_templates["domestic_service"] = TaskTemplate(
            id="domestic_service",
            title="Domestic Service Excellence",
            description="Perform household tasks with dedication and attention to detail.",
            instructions=[
                "Complete assigned cleaning tasks to perfection",
                "Take before and after photos of each area",
                "Spend at least [duration] minutes on detailed cleaning",
                "Present results with pride in your service"
            ],
            category=TaskCategory.SERVICE,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Submit before/after photos of cleaned areas with description of work completed.",
            estimated_duration_minutes=60,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(service=0.8, obedience=0.6),
            customization_options=CustomizationOptions(
                duration=[30, 45, 60, 90, 120]
            ),
            reward_suggestions=[
                "Praise for excellent service",
                "Permission for a reward activity",
                "Reduced service requirements next day"
            ],
            punishment_suggestions=[
                "Redo the entire task",
                "Additional service assignments",
                "Loss of privileges"
            ],
            tags=["service", "domestic", "cleaning", "dedication"]
        )

        self.task_templates["personal_service"] = TaskTemplate(
            id="personal_service",
            title="Personal Service and Attendance",
            description="Provide attentive personal service demonstrating devotion.",
            instructions=[
                "Prepare and serve a beverage exactly as preferred",
                "Maintain proper posture and demeanor throughout",
                "Anticipate needs without being asked",
                "Complete service in silence unless spoken to"
            ],
            category=TaskCategory.SERVICE,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.VIDEO,
            verification_instructions="Record a video showing your service preparation and presentation.",
            estimated_duration_minutes=30,
            suitable_for_levels=[3, 4, 5],
            suitable_for_traits=TaskPreferences(service=0.9, protocol=0.7),
            customization_options=CustomizationOptions(),
            reward_suggestions=[
                "Verbal praise and acknowledgment",
                "Brief physical contact as reward",
                "Elevated status for the day"
            ],
            punishment_suggestions=[
                "Service training exercises",
                "Written essay on proper service",
                "Loss of speaking privileges"
            ],
            tags=["service", "personal", "protocol", "attention"]
        )

        # OBEDIENCE Tasks
        self.task_templates["position_training"] = TaskTemplate(
            id="position_training",
            title="Position Training and Holding",
            description="Practice and maintain submissive positions to demonstrate obedience.",
            instructions=[
                "Assume the assigned position",
                "Hold for [duration] minutes without moving",
                "Focus on your submission during this time",
                "Document any physical or mental challenges"
            ],
            category=TaskCategory.OBEDIENCE,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Submit photo proof at start and end, plus written report of experience.",
            estimated_duration_minutes=30,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(obedience=0.8, endurance=0.6),
            customization_options=CustomizationOptions(
                duration=[10, 15, 20, 30, 45],
                position=["kneeling", "standing at attention", "prostrate", "wall sit"]
            ),
            reward_suggestions=[
                "Praise for endurance",
                "Massage or comfort afterwards",
                "Choice of next position"
            ],
            punishment_suggestions=[
                "Extended holding time",
                "More difficult position",
                "Daily position practice for a week"
            ],
            tags=["obedience", "positions", "training", "endurance"]
        )

        self.task_templates["command_response"] = TaskTemplate(
            id="command_response",
            title="Instant Command Response Training",
            description="Practice immediate obedience to commands without hesitation.",
            instructions=[
                "Set [count] random alarms throughout the day",
                "When alarm sounds, immediately perform assigned action",
                "Complete action within 30 seconds",
                "Log each response time and any delays"
            ],
            category=TaskCategory.OBEDIENCE,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit detailed log of all command responses with timestamps and completion times.",
            estimated_duration_minutes=10,
            suitable_for_levels=[3, 4, 5],
            suitable_for_traits=TaskPreferences(obedience=0.9, protocol=0.7),
            customization_options=CustomizationOptions(
                count=[3, 5, 7, 10]
            ),
            reward_suggestions=[
                "Recognition of perfect obedience",
                "Reduced commands next day",
                "Special privilege earned"
            ],
            punishment_suggestions=[
                "Double commands next day",
                "Write lines about obedience",
                "Additional obedience training"
            ],
            tags=["obedience", "commands", "training", "immediate"]
        )

        # WORSHIP Tasks
        self.task_templates["written_worship"] = TaskTemplate(
            id="written_worship",
            title="Written Worship and Adoration",
            description="Express deep worship and admiration through written words.",
            instructions=[
                "Write [minimum_words] words of sincere worship",
                "Include specific reasons for your devotion",
                "Express how submission enriches your life",
                "End with a pledge of continued service"
            ],
            category=TaskCategory.WORSHIP,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit your complete written worship for review.",
            estimated_duration_minutes=45,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(worship=0.8, creative=0.6),
            customization_options=CustomizationOptions(
                minimum_words=[200, 300, 500, 750, 1000]
            ),
            reward_suggestions=[
                "Acknowledgment of devotion",
                "Permission to serve more closely",
                "Special attention or praise"
            ],
            punishment_suggestions=[
                "Rewrite until satisfactory",
                "Read aloud while kneeling",
                "Copy by hand multiple times"
            ],
            tags=["worship", "writing", "devotion", "expression"]
        )

        self.task_templates["tribute_creation"] = TaskTemplate(
            id="tribute_creation",
            title="Creative Tribute Creation",
            description="Create a artistic tribute demonstrating your worship and creativity.",
            instructions=[
                "Create an artistic tribute (drawing, poem, song, etc.)",
                "Spend at least [duration] minutes on creation",
                "Include symbolism of your submission",
                "Present with explanation of meaning"
            ],
            category=TaskCategory.WORSHIP,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Submit photos of your tribute with detailed explanation of symbolism and meaning.",
            estimated_duration_minutes=90,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(worship=0.7, creative=0.9),
            customization_options=CustomizationOptions(
                duration=[60, 90, 120, 180]
            ),
            reward_suggestions=[
                "Display of tribute",
                "Special recognition",
                "Privilege of creating more"
            ],
            punishment_suggestions=[
                "Start over from scratch",
                "Create multiple tributes",
                "Loss of creative privileges"
            ],
            tags=["worship", "creative", "tribute", "artistic"]
        )

        # SELF_IMPROVEMENT Tasks
        self.task_templates["fitness_challenge"] = TaskTemplate(
            id="fitness_challenge",
            title="Physical Fitness Challenge",
            description="Improve your physical condition to better serve.",
            instructions=[
                "Complete [count] sets of assigned exercises",
                "Maintain proper form throughout",
                "Push yourself but stay safe",
                "Log all repetitions and any difficulties"
            ],
            category=TaskCategory.SELF_IMPROVEMENT,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit detailed workout log with sets, reps, and how you felt.",
            estimated_duration_minutes=30,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(self_improvement=0.8, endurance=0.7),
            customization_options=CustomizationOptions(
                count=[3, 4, 5, 6]
            ),
            reward_suggestions=[
                "Praise for dedication",
                "Rest day earned",
                "Choice of next workout"
            ],
            punishment_suggestions=[
                "Additional exercises",
                "No rest day",
                "Fitness essay writing"
            ],
            tags=["fitness", "self-improvement", "exercise", "health"]
        )

        self.task_templates["skill_development"] = TaskTemplate(
            id="skill_development",
            title="Skill Development Practice",
            description="Develop a skill that enhances your ability to serve.",
            instructions=[
                "Practice designated skill for [duration] minutes",
                "Focus on improvement and perfection",
                "Document what you learned",
                "Show measurable progress"
            ],
            category=TaskCategory.SELF_IMPROVEMENT,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.VIDEO,
            verification_instructions="Submit video showing skill practice and demonstration of progress.",
            estimated_duration_minutes=45,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(self_improvement=0.9, service=0.6),
            customization_options=CustomizationOptions(
                duration=[30, 45, 60, 90]
            ),
            reward_suggestions=[
                "Recognition of improvement",
                "Opportunity to demonstrate skill",
                "Advanced skill assignments"
            ],
            punishment_suggestions=[
                "Repeat basic training",
                "Double practice time",
                "Write about importance of skills"
            ],
            tags=["self-improvement", "skills", "practice", "development"]
        )

        # PROTOCOL Tasks
        self.task_templates["speech_protocol"] = TaskTemplate(
            id="speech_protocol",
            title="Speech Protocol Practice",
            description="Practice proper speech protocols and forms of address.",
            instructions=[
                "Use only approved forms of address all day",
                "Speak only when permitted",
                "Request permission before speaking freely",
                "Log any protocol violations"
            ],
            category=TaskCategory.PROTOCOL,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit log of all speech interactions and any protocol violations with explanations.",
            estimated_duration_minutes=1440,
            suitable_for_levels=[3, 4, 5],
            suitable_for_traits=TaskPreferences(protocol=0.9, obedience=0.7),
            customization_options=CustomizationOptions(),
            reward_suggestions=[
                "Brief free speech period",
                "Praise for perfect protocol",
                "Relaxed protocols for an hour"
            ],
            punishment_suggestions=[
                "Silent day",
                "Write lines about proper speech",
                "Extended protocol period"
            ],
            tags=["protocol", "speech", "formality", "discipline"]
        )

        self.task_templates["dress_code"] = TaskTemplate(
            id="dress_code",
            title="Dress Code and Presentation",
            description="Maintain required dress code and presentation standards.",
            instructions=[
                "Wear designated outfit or style all day",
                "Maintain impeccable grooming",
                "Send photos at [check_in_frequency]",
                "Note any reactions or feelings"
            ],
            category=TaskCategory.PROTOCOL,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Submit photos at required check-ins showing full compliance with dress code.",
            estimated_duration_minutes=1440,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(protocol=0.8, obedience=0.7),
            customization_options=CustomizationOptions(
                check_in_frequency=["morning and evening", "every 4 hours", "every 2 hours"]
            ),
            reward_suggestions=[
                "Choice of outfit next day",
                "Compliments on appearance",
                "Relaxed dress code for a day"
            ],
            punishment_suggestions=[
                "More restrictive dress code",
                "Public inspection",
                "Extended dress code period"
            ],
            tags=["protocol", "dress code", "appearance", "presentation"]
        )

        # ENDURANCE Tasks
        self.task_templates["endurance_challenge"] = TaskTemplate(
            id="endurance_challenge",
            title="Physical Endurance Challenge",
            description="Test your physical endurance and push your limits safely.",
            instructions=[
                "Maintain challenging position or activity",
                "Continue for [duration] minutes",
                "Focus on mental strength",
                "Stop if genuinely unsafe"
            ],
            category=TaskCategory.ENDURANCE,
            difficulty=TaskDifficulty.DIFFICULT,
            verification_type=VerificationType.VIDEO,
            verification_instructions="Submit video clips showing start, middle, and end of endurance challenge.",
            estimated_duration_minutes=45,
            suitable_for_levels=[3, 4, 5],
            suitable_for_traits=TaskPreferences(endurance=0.9, obedience=0.7),
            customization_options=CustomizationOptions(
                duration=[20, 30, 45, 60]
            ),
            reward_suggestions=[
                "Pride in accomplishment",
                "Special recognition",
                "Choice of next challenge"
            ],
            punishment_suggestions=[
                "Repeat at longer duration",
                "Additional endurance training",
                "Essay on mental weakness"
            ],
            tags=["endurance", "challenge", "physical", "mental"]
        )

        self.task_templates["denial_endurance"] = TaskTemplate(
            id="denial_endurance",
            title="Denial and Self-Control",
            description="Practice self-control through denial of pleasures.",
            instructions=[
                "Abstain from specified pleasure all day",
                "Note every temptation and how you resisted",
                "Find strength in your submission",
                "Report honestly on any failures"
            ],
            category=TaskCategory.ENDURANCE,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.HONOR,
            verification_instructions="Submit honest report of your day including all temptations and how you handled them.",
            estimated_duration_minutes=1440,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(endurance=0.8, obedience=0.8),
            customization_options=CustomizationOptions(),
            reward_suggestions=[
                "Brief enjoyment of denied pleasure",
                "Praise for self-control",
                "Reduced denial period"
            ],
            punishment_suggestions=[
                "Extended denial period",
                "Additional denials added",
                "Write about lack of control"
            ],
            tags=["endurance", "denial", "self-control", "discipline"]
        )

        # HUMILIATION Tasks
        self.task_templates["private_humiliation"] = TaskTemplate(
            id="private_humiliation",
            title="Private Humiliation Task",
            description="Experience humiliation in private to deepen submission.",
            instructions=[
                "Complete the assigned humiliating task",
                "Focus on how it makes you feel",
                "Embrace the vulnerability",
                "Write about the experience"
            ],
            category=TaskCategory.HUMILIATION,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit detailed written account of the experience and your emotional journey.",
            estimated_duration_minutes=30,
            suitable_for_levels=[3, 4, 5],
            suitable_for_traits=TaskPreferences(humiliation=0.7, obedience=0.6),
            customization_options=CustomizationOptions(),
            reward_suggestions=[
                "Comfort and reassurance",
                "Acknowledgment of bravery",
                "Aftercare attention"
            ],
            punishment_suggestions=[
                "Repeat with additions",
                "Share more details",
                "Extended humiliation"
            ],
            tags=["humiliation", "private", "emotional", "vulnerability"]
        )

        self.task_templates["confession_task"] = TaskTemplate(
            id="confession_task",
            title="Embarrassing Confession",
            description="Share embarrassing truths to demonstrate vulnerability.",
            instructions=[
                "Write [count] embarrassing confessions",
                "Be completely honest",
                "Include why each embarrasses you",
                "Submit without editing"
            ],
            category=TaskCategory.HUMILIATION,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit your unedited confessions with explanations.",
            estimated_duration_minutes=45,
            suitable_for_levels=[3, 4, 5],
            suitable_for_traits=TaskPreferences(humiliation=0.8, worship=0.5),
            customization_options=CustomizationOptions(
                count=[3, 5, 7, 10]
            ),
            reward_suggestions=[
                "Confessions kept private",
                "Gentle acceptance",
                "Trust building praise"
            ],
            punishment_suggestions=[
                "Read confessions aloud",
                "Write more confessions",
                "Deeper revelations required"
            ],
            tags=["humiliation", "confession", "vulnerability", "honesty"]
        )

        # CREATIVE Tasks
        self.task_templates["creative_writing"] = TaskTemplate(
            id="creative_writing",
            title="Creative Writing Task",
            description="Create written content that explores your submission.",
            instructions=[
                "Write a creative piece about submission",
                "Minimum [minimum_words] words",
                "Be original and thoughtful",
                "Express genuine feelings"
            ],
            category=TaskCategory.CREATIVE,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit your complete creative writing piece.",
            estimated_duration_minutes=60,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(creative=0.9, worship=0.6),
            customization_options=CustomizationOptions(
                minimum_words=[300, 500, 750, 1000]
            ),
            reward_suggestions=[
                "Share writing publicly",
                "Praise for creativity",
                "Request for more writing"
            ],
            punishment_suggestions=[
                "Rewrite completely",
                "Write on assigned topic",
                "Loss of creative freedom"
            ],
            tags=["creative", "writing", "expression", "original"]
        )

        self.task_templates["artistic_expression"] = TaskTemplate(
            id="artistic_expression",
            title="Artistic Expression of Submission",
            description="Create visual art that represents your submission.",
            instructions=[
                "Create original artwork about submission",
                "Use any medium available",
                "Include written explanation",
                "Spend at least [duration] minutes"
            ],
            category=TaskCategory.CREATIVE,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Submit photos of artwork with written explanation of meaning and process.",
            estimated_duration_minutes=90,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(creative=0.9, worship=0.7),
            customization_options=CustomizationOptions(
                duration=[60, 90, 120, 180]
            ),
            reward_suggestions=[
                "Display of artwork",
                "Praise for creativity",
                "Commission for more art"
            ],
            punishment_suggestions=[
                "Destroy and recreate",
                "Art restriction",
                "Assigned art topics only"
            ],
            tags=["creative", "art", "expression", "visual"]
        )

        # PUNISHMENT Tasks
        self.task_templates["punishment_lines"] = TaskTemplate(
            id="punishment_lines",
            title="Write Punishment Lines",
            description="Write lines as punishment for disobedience or failure.",
            instructions=[
                "Write the assigned line [repetitions] times",
                "Each must be neat and legible",
                "Number each line",
                "Submit photo of completed lines"
            ],
            category=TaskCategory.PUNISHMENT,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Submit clear photos showing all completed lines.",
            estimated_duration_minutes=60,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(obedience=0.5),
            customization_options=CustomizationOptions(
                repetitions=[50, 100, 200, 300, 500]
            ),
            reward_suggestions=[
                "Forgiveness expressed",
                "Clean slate granted",
                "Reduced future punishments"
            ],
            punishment_suggestions=[
                "Double the lines",
                "Rewrite if messy",
                "Additional punishment task"
            ],
            tags=["punishment", "lines", "discipline", "correction"]
        )

        self.task_templates["corner_time"] = TaskTemplate(
            id="corner_time",
            title="Corner Time Punishment",
            description="Spend time in the corner reflecting on behavior.",
            instructions=[
                "Stand facing corner for [duration] minutes",
                "Hands behind back or at sides",
                "No fidgeting or moving",
                "Reflect on your behavior"
            ],
            category=TaskCategory.PUNISHMENT,
            difficulty=TaskDifficulty.EASY,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Submit photo proof at start and end, plus written reflection.",
            estimated_duration_minutes=30,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits=TaskPreferences(obedience=0.4),
            customization_options=CustomizationOptions(
                duration=[15, 30, 45, 60]
            ),
            reward_suggestions=[
                "Early release",
                "Forgiveness granted",
                "Comfort afterwards"
            ],
            punishment_suggestions=[
                "Extended time",
                "Harder position",
                "Daily corner time"
            ],
            tags=["punishment", "corner", "reflection", "discipline"]
        )
        
        logger.info(f"Loaded {len(self.task_templates)} default task templates")
    
    async def _get_or_create_user_settings(self, user_id: str) -> UserTaskSettings:
        """Get or create user task settings."""
        if user_id not in self.user_settings:
            self.user_settings[user_id] = UserTaskSettings(user_id=user_id)
        return self.user_settings[user_id]

    # Convert methods to function tools that agents can use
    @function_tool
    async def get_user_profile_for_task_design(self, user_id: str) -> UserProfileResult:
        """Retrieves user profile data for task customization."""
        try:
            user_profile = UserProfileResult(user_id=user_id)
            
            # Get user settings if available
            if user_id in self.user_settings:
                settings = self.user_settings[user_id]
                # Convert dict to TaskPreferences model
                if hasattr(settings, 'task_preferences') and isinstance(settings.task_preferences, dict):
                    user_profile.task_preferences = TaskPreferences(**settings.task_preferences)
                else:
                    user_profile.task_preferences = settings.task_preferences
                user_profile.preferred_difficulty = settings.preferred_difficulty
                user_profile.preferred_verification = settings.preferred_verification
                user_profile.task_completion_rate = settings.task_completion_rate
            
            # Get relationship data if available
            if self.relationship_manager:
                try:
                    relationship = await self.relationship_manager.get_relationship_state(user_id)
                    if relationship:
                        user_profile.trust_level = getattr(relationship, "trust", 0.5)
                        user_profile.submission_level = getattr(relationship, "submission_level", 1)
                        
                        # Convert dict to InferredTraits model
                        traits = getattr(relationship, "inferred_user_traits", {})
                        if isinstance(traits, dict):
                            user_profile.inferred_traits = InferredTraits(**{k: v for k, v in traits.items() if hasattr(InferredTraits, k)})
                        
                        # Convert to UserLimits model
                        user_profile.limits = UserLimits(
                            hard=getattr(relationship, "hard_limits", []),
                            soft=getattr(relationship, "soft_limits", [])
                        )
                except Exception as e:
                    logger.error(f"Error retrieving relationship data: {e}")
            
            # Get submission data if available
            if self.submission_progression:
                try:
                    submission_data = await self.submission_progression.get_user_submission_data(user_id)
                    if submission_data:
                        user_profile.submission_level = submission_data.get("submission_level", {}).get("id", 1)
                        # Convert dict to SubmissionMetrics model
                        metrics = submission_data.get("metrics", {})
                        if isinstance(metrics, dict):
                            user_profile.submission_metrics = SubmissionMetrics(**{k: v for k, v in metrics.items() if hasattr(SubmissionMetrics, k)})
                except Exception as e:
                    logger.error(f"Error retrieving submission data: {e}")
            
            return user_profile
            
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return UserProfileResult(user_id=user_id, error=str(e))
    
    @function_tool
    async def get_task_completion_history(self, user_id: str, limit: int = 5) -> TaskCompletionHistoryResult:
        """Retrieves task completion history for a user."""
        try:
            if user_id not in self.user_settings:
                return TaskCompletionHistoryResult(
                    user_id=user_id, 
                    history=[],
                    stats=TaskStats(
                        completion_rate=1.0, 
                        total_completed=0,
                        total_failed=0
                    )
                )
            
            settings = self.user_settings[user_id]
            
            # Get recent task history - convert dicts to TaskHistoryEntry
            recent_history = []
            for entry in settings.task_history[-limit:]:
                if isinstance(entry, dict):
                    recent_history.append(TaskHistoryEntry(**entry))
                else:
                    recent_history.append(entry)
            
            # Compile statistics
            preferred_categories = []
            if hasattr(settings, 'task_preferences'):
                if isinstance(settings.task_preferences, dict):
                    prefs = sorted(settings.task_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
                    preferred_categories = [[k, v] for k, v in prefs]
                elif isinstance(settings.task_preferences, TaskPreferences):
                    prefs = []
                    for field_name, field_value in settings.task_preferences.model_dump().items():
                        if field_value is not None:
                            prefs.append((field_name, field_value))
                    prefs.sort(key=lambda x: x[1], reverse=True)
                    preferred_categories = [[k, v] for k, v in prefs[:3]]
            
            stats = TaskStats(
                completion_rate=settings.task_completion_rate,
                total_completed=len(settings.completed_tasks),
                total_failed=len(settings.failed_tasks),
                preferred_categories=preferred_categories
            )
            
            return TaskCompletionHistoryResult(
                user_id=user_id, 
                history=recent_history,
                stats=stats
            )
            
        except Exception as e:
            logger.error(f"Error retrieving task history: {e}")
            return TaskCompletionHistoryResult(
                user_id=user_id, 
                error=str(e), 
                history=[], 
                stats=TaskStats(completion_rate=0.0, total_completed=0, total_failed=0)
            )
    
    @function_tool
    async def get_task_details(self, task_id: str) -> TaskDetailsResult:
        """Retrieves details for a specific task."""
        try:
            if task_id not in self.assigned_tasks:
                return TaskDetailsResult(error=f"Task {task_id} not found")
            
            task = self.assigned_tasks[task_id]
            return TaskDetailsResult(
                task_id=task.id,
                user_id=task.user_id,
                title=task.title,
                description=task.description,
                instructions=task.instructions,
                category=task.category,
                difficulty=task.difficulty,
                assigned_at=task.assigned_at.isoformat(),
                due_at=task.due_at.isoformat() if task.due_at else None,
                completed_at=task.completed_at.isoformat() if task.completed_at else None,
                verification_type=task.verification_type,
                verification_instructions=task.verification_instructions,
                verification_data=task.verification_data,
                completed=task.completed,
                failed=task.failed,
                rating=task.rating,
                reward=task.reward,
                punishment=task.punishment,
                notes=task.notes,
                extension_count=task.extension_count,
                tags=task.tags,
                custom_data=task.custom_data
            )
            
        except Exception as e:
            logger.error(f"Error retrieving task details: {e}")
            return TaskDetailsResult(error=str(e))
    
    @function_tool
    async def get_available_templates(self, category: Optional[str] = None) -> TemplatesResult:
        """Retrieves available task templates filtered by category."""
        try:
            templates = []
            
            for template_id, template in self.task_templates.items():
                # Apply category filter if specified
                if category and template.category != category:
                    continue
                    
                templates.append(TemplateData(
                    id=template_id,
                    title=template.title,
                    category=template.category,
                    difficulty=template.difficulty,
                    description=template.description
                ))
            
            return TemplatesResult(
                templates=templates,
                count=len(templates),
                category_filter=category
            )
            
        except Exception as e:
            logger.error(f"Error retrieving templates: {e}")
            return TemplatesResult(error=str(e), templates=[], count=0)
    
    @function_tool
    async def get_user_task_statistics(self, user_id: str) -> TaskStatisticsResult:
        """Get statistics about a user's task history."""
        try:
            # Check if user has settings
            if user_id not in self.user_settings:
                return TaskStatisticsResult(
                    success=True,
                    user_id=user_id,
                    statistics=TaskStatisticsData(
                        total_tasks=0,
                        completed_tasks=0,
                        failed_tasks=0,
                        completion_rate=0.0,
                        average_rating=0.0,
                        active_tasks=0
                    ),
                    category_breakdown={},
                    difficulty_breakdown={}
                )
            
            settings = self.user_settings[user_id]
            
            # Basic statistics
            total_completed = len(settings.completed_tasks)
            total_failed = len(settings.failed_tasks)
            total_active = len(settings.active_tasks)
            total_tasks = total_completed + total_failed
            
            completion_rate = 0.0
            if total_tasks > 0:
                completion_rate = total_completed / total_tasks
            
            # Calculate average rating
            ratings = []
            for task_id in settings.completed_tasks:
                if task_id in self.assigned_tasks:
                    task = self.assigned_tasks[task_id]
                    if task.rating is not None:
                        ratings.append(task.rating)
            
            average_rating = sum(ratings) / len(ratings) if ratings else 0.0
            
            # Category breakdown
            category_counts = {}
            category_completion_rates = {}
            
            # Difficulty breakdown
            difficulty_counts = {}
            difficulty_completion_rates = {}
            
            # Analyze task history
            for entry in settings.task_history:
                if isinstance(entry, dict):
                    category = entry.get("category")
                    difficulty = entry.get("difficulty")
                    completed = entry.get("completed", False)
                else:
                    category = entry.category
                    difficulty = entry.difficulty
                    completed = entry.completed
                
                # Update category stats
                if category:
                    if category not in category_counts:
                        category_counts[category] = {"total": 0, "completed": 0}
                        
                    category_counts[category]["total"] += 1
                    if completed:
                        category_counts[category]["completed"] += 1
                
                # Update difficulty stats
                if difficulty:
                    if difficulty not in difficulty_counts:
                        difficulty_counts[difficulty] = {"total": 0, "completed": 0}
                        
                    difficulty_counts[difficulty]["total"] += 1
                    if completed:
                        difficulty_counts[difficulty]["completed"] += 1
            
            # Calculate completion rates by category
            for category, counts in category_counts.items():
                if counts["total"] > 0:
                    category_completion_rates[category] = counts["completed"] / counts["total"]
            
            # Calculate completion rates by difficulty
            for difficulty, counts in difficulty_counts.items():
                if counts["total"] > 0:
                    difficulty_completion_rates[difficulty] = counts["completed"] / counts["total"]
            
            # Format statistics
            statistics = TaskStatisticsData(
                total_tasks=total_tasks,
                completed_tasks=total_completed,
                failed_tasks=total_failed,
                active_tasks=total_active,
                completion_rate=completion_rate,
                average_rating=average_rating
            )
            
            # Format breakdowns
            category_breakdown = {
                category: CategoryBreakdownItem(
                    count=counts["total"],
                    completed=counts["completed"],
                    completion_rate=category_completion_rates.get(category, 0.0)
                )
                for category, counts in category_counts.items()
            }
            
            difficulty_breakdown = {
                difficulty: DifficultyBreakdownItem(
                    count=counts["total"],
                    completed=counts["completed"],
                    completion_rate=difficulty_completion_rates.get(difficulty, 0.0)
                )
                for difficulty, counts in difficulty_counts.items()
            }
            
            # Get preferred categories
            preferred_categories = []
            if hasattr(settings, 'task_preferences'):
                if isinstance(settings.task_preferences, dict):
                    prefs = sorted(settings.task_preferences.items(), key=lambda x: x[1], reverse=True)
                    preferred_categories = [[k, v] for k, v in prefs]
                elif isinstance(settings.task_preferences, TaskPreferences):
                    prefs = []
                    for field_name, field_value in settings.task_preferences.model_dump().items():
                        if field_value is not None:
                            prefs.append((field_name, field_value))
                    prefs.sort(key=lambda x: x[1], reverse=True)
                    preferred_categories = [[k, v] for k, v in prefs]
            
            return TaskStatisticsResult(
                success=True,
                user_id=user_id,
                statistics=statistics,
                category_breakdown=category_breakdown,
                difficulty_breakdown=difficulty_breakdown,
                preferred_categories=preferred_categories
            )
            
        except Exception as e:
            logger.error(f"Error retrieving task statistics: {e}")
            return TaskStatisticsResult(
                success=False,
                error=str(e)
            )
    
    @function_tool
    async def get_active_tasks(self, user_id: str) -> ActiveTasksResult:
        """Get all active tasks for a user."""
        try:
            # Check if user has settings
            if user_id not in self.user_settings:
                return ActiveTasksResult(
                    success=True,
                    user_id=user_id,
                    active_tasks=[],
                    count=0
                )
            
            settings = self.user_settings[user_id]
            
            # Get all active task IDs
            active_task_ids = settings.active_tasks
            
            # Get task details
            active_tasks = []
            for task_id in active_task_ids:
                if task_id in self.assigned_tasks:
                    task = self.assigned_tasks[task_id]
                    
                    # Format task
                    formatted_task = ActiveTaskData(
                        task_id=task.id,
                        title=task.title,
                        description=task.description,
                        instructions=task.instructions,
                        category=task.category,
                        difficulty=task.difficulty,
                        assigned_at=task.assigned_at.isoformat(),
                        due_at=task.due_at.isoformat() if task.due_at else None,
                        verification_type=task.verification_type,
                        time_remaining=self._get_time_remaining(task) if task.due_at else None
                    )
                    
                    active_tasks.append(formatted_task)
            
            # Sort by due date (closest first)
            active_tasks.sort(key=lambda t: t.due_at if t.due_at else "9999-12-31T23:59:59")
            
            return ActiveTasksResult(
                success=True,
                user_id=user_id,
                active_tasks=active_tasks,
                count=len(active_tasks),
                max_concurrent=settings.max_concurrent_tasks
            )
            
        except Exception as e:
            logger.error(f"Error retrieving active tasks: {e}")
            return ActiveTasksResult(
                success=False,
                error=str(e),
                active_tasks=[],
                count=0
            )
    
    @function_tool
    async def get_expired_tasks(self) -> List[ExpiredTaskResult]:
        """Get all expired/overdue tasks."""
        try:
            now = datetime.datetime.now()
            expired_tasks = []
            
            for task_id, task in self.assigned_tasks.items():
                # Check if task is active and has a due date
                if not task.completed and not task.failed and task.due_at:
                    # Check if overdue
                    if now > task.due_at:
                        expired_tasks.append(ExpiredTaskResult(
                            task_id=task_id,
                            user_id=task.user_id,
                            title=task.title,
                            due_at=task.due_at.isoformat(),
                            overdue_hours=(now - task.due_at).total_seconds() / 3600.0
                        ))
            
            # Sort by most overdue first
            expired_tasks.sort(key=lambda t: t.overdue_hours, reverse=True)
            
            return expired_tasks
            
        except Exception as e:
            logger.error(f"Error retrieving expired tasks: {e}")
            return []
    
    # Main business logic methods using the agent infrastructure
    async def assign_task(self, 
                       user_id: str, 
                       template_id: Optional[str] = None,
                       custom_task: Optional[Dict[str, Any]] = None,
                       due_in_hours: Optional[int] = 24,
                       difficulty_override: Optional[str] = None,
                       verification_override: Optional[str] = None,
                       custom_reward: Optional[Dict[str, Any]] = None,
                       custom_punishment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assign a task to a user based on template or custom definition.
        """
        async with self._lock:
            # Get user settings
            settings = await self._get_or_create_user_settings(user_id)
            
            # Check if user already has maximum tasks
            if len(settings.active_tasks) >= settings.max_concurrent_tasks:
                return {
                    "success": False,
                    "message": f"User already has maximum allowed tasks ({settings.max_concurrent_tasks})",
                    "active_tasks": len(settings.active_tasks)
                }
            
            # Use trace to track the entire task assignment process
            with trace(workflow_name="Task Assignment", 
                     group_id=f"user_{user_id}",
                     metadata={"user_id": user_id, "template_id": template_id}):
                
                # Create task from template or custom definition
                if template_id:
                    # Check if template exists
                    if template_id not in self.task_templates:
                        return {
                            "success": False,
                            "message": f"Task template '{template_id}' not found",
                            "available_templates": list(self.task_templates.keys())
                        }
                    
                    # Use template as base for creating task
                    task = await self._create_task_from_template(user_id, self.task_templates[template_id], 
                                                          due_in_hours, difficulty_override,
                                                          verification_override, custom_reward,
                                                          custom_punishment)
                    
                elif custom_task:
                    # Create task from custom definition using validation guardrail
                    # Add user_id to custom task for validation
                    custom_task["user_id"] = user_id
                    
                    # Run validation through agent with guardrail
                    task = await self._create_task_from_custom(user_id, custom_task, due_in_hours,
                                                      difficulty_override, verification_override,
                                                      custom_reward, custom_punishment)
                    
                else:
                    # Use AI to generate a task
                    task = await self._generate_ai_task(user_id, difficulty_override, verification_override,
                                                due_in_hours, custom_reward, custom_punishment)
                    if not task:
                        return {
                            "success": False,
                            "message": "Failed to generate an appropriate task"
                        }
                
                # Store the assigned task
                self.assigned_tasks[task.id] = task
                
                # Update user settings
                settings.active_tasks.append(task.id)
                settings.last_updated = datetime.datetime.now()
                
                # Update task preferences based on category
                if hasattr(settings.task_preferences, task.category):
                    current_value = getattr(settings.task_preferences, task.category) or 0.0
                    setattr(settings.task_preferences, task.category, min(current_value + 0.1, 1.0))
                
                # Add task to relationship manager if available
                if self.relationship_manager:
                    try:
                        await self.relationship_manager.update_relationship_attribute(
                            user_id,
                            "active_tasks",
                            [t for t in settings.active_tasks]
                        )
                    except Exception as e:
                        logger.error(f"Error updating relationship data: {e}")
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=f"Assigned task '{task.title}' to user. Due in {due_in_hours} hours.",
                            tags=["task_assignment", task.category, "femdom"],
                            significance=0.5
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                # Format task data for return
                formatted_task = {
                    "task_id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "instructions": task.instructions,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "due_at": task.due_at.isoformat() if task.due_at else None,
                    "verification_type": task.verification_type,
                    "verification_instructions": task.verification_instructions,
                    "reward": task.reward.model_dump() if task.reward else None,
                    "punishment": task.punishment.model_dump() if task.punishment else None
                }
                
                return {
                    "success": True,
                    "message": "Task assigned successfully",
                    "task": formatted_task
                }
    
    async def _create_task_from_template(self, 
                                      user_id: str, 
                                      template: TaskTemplate,
                                      due_in_hours: Optional[int], 
                                      difficulty_override: Optional[str],
                                      verification_override: Optional[str],
                                      custom_reward: Optional[Dict[str, Any]],
                                      custom_punishment: Optional[Dict[str, Any]]) -> AssignedTask:
        """Create a task instance from a template."""
        # Create task ID
        task_id = f"task_{uuid.uuid4()}"
        
        # Set due date
        due_at = None
        if due_in_hours is not None:
            due_at = datetime.datetime.now() + datetime.timedelta(hours=due_in_hours)
        
        # Apply potential customizations based on user profile
        user_profile = await self.get_user_profile_for_task_design(user_id)
        instructions = list(template.instructions)  # Copy to modify
        
        # Customize instructions if template has options
        customization_dict = template.customization_options.model_dump()
        if any(v for v in customization_dict.values() if v):
            for option_key, option_values in customization_dict.items():
                if option_values:
                    # Find relevant option placeholder in instructions
                    for i, instruction in enumerate(instructions):
                        if f"[{option_key}]" in instruction or f"[X]" in instruction:
                            # Choose appropriate option value based on user profile
                            chosen_value = self._choose_appropriate_option(option_values, option_key, user_profile)
                            # Replace placeholder with chosen value
                            instructions[i] = instruction.replace(f"[{option_key}]", str(chosen_value)).replace("[X]", str(chosen_value))
        
        # Choose reward and punishment if not provided
        reward = None
        if custom_reward:
            reward = RewardData(**custom_reward)
        elif template.reward_suggestions:
            reward = RewardData(
                description=random.choice(template.reward_suggestions),
                type="standard"
            )
            
        punishment = None
        if custom_punishment:
            punishment = PunishmentData(**custom_punishment)
        elif template.punishment_suggestions:
            punishment = PunishmentData(
                description=random.choice(template.punishment_suggestions),
                type="standard"
            )
        
        # Create task
        task = AssignedTask(
            id=task_id,
            user_id=user_id,
            title=template.title,
            description=template.description,
            instructions=instructions,
            category=template.category,
            difficulty=difficulty_override or template.difficulty,
            verification_type=verification_override or template.verification_type,
            verification_instructions=template.verification_instructions,
            due_at=due_at,
            reward=reward,
            punishment=punishment,
            tags=template.tags
        )
        
        return task
    
    async def _create_task_from_custom(self,
                                    user_id: str,
                                    custom_task: Dict[str, Any],
                                    due_in_hours: Optional[int],
                                    difficulty_override: Optional[str],
                                    verification_override: Optional[str],
                                    custom_reward: Optional[Dict[str, Any]],
                                    custom_punishment: Optional[Dict[str, Any]]) -> AssignedTask:
        """Create a task from a custom definition with validation."""
        # Create task ID
        task_id = f"task_{uuid.uuid4()}"
        
        # Set due date
        due_at = None
        if due_in_hours is not None:
            due_at = datetime.datetime.now() + datetime.timedelta(hours=due_in_hours)
        
        # Create reward and punishment models
        reward = None
        if custom_reward:
            reward = RewardData(**custom_reward)
        elif custom_task.get("reward"):
            reward = RewardData(**custom_task["reward"])
            
        punishment = None
        if custom_punishment:
            punishment = PunishmentData(**custom_punishment)
        elif custom_task.get("punishment"):
            punishment = PunishmentData(**custom_task["punishment"])
            
        custom_data = None
        if custom_task.get("custom_data"):
            custom_data = CustomTaskData(**custom_task["custom_data"])
        
        # Create task
        task = AssignedTask(
            id=task_id,
            user_id=user_id,
            title=custom_task["title"],
            description=custom_task["description"],
            category=custom_task["category"],
            instructions=custom_task.get("instructions", []),
            difficulty=difficulty_override or custom_task.get("difficulty", TaskDifficulty.MODERATE),
            verification_type=verification_override or custom_task.get("verification_type", VerificationType.HONOR),
            verification_instructions=custom_task.get("verification_instructions", "Verify completion as specified."),
            due_at=due_at,
            reward=reward,
            punishment=punishment,
            tags=custom_task.get("tags", []),
            custom_data=custom_data
        )
        
        return task
    
    async def _generate_ai_task(self, 
                             user_id: str,
                             difficulty_override: Optional[str],
                             verification_override: Optional[str],
                             due_in_hours: Optional[int],
                             custom_reward: Optional[Dict[str, Any]],
                             custom_punishment: Optional[Dict[str, Any]]) -> Optional[AssignedTask]:
        """Generate a task using the AI task ideation agent."""
        # Update the task context with latest state
        task_context = TaskContext(
            assigned_tasks=self.assigned_tasks,
            user_settings=self.user_settings,
            task_templates=self.task_templates,
            memory_core=self.memory_core,
            reward_system=self.reward_system,
            relationship_manager=self.relationship_manager,
            submission_progression=self.submission_progression
        )

        try:
            # Get user profile
            user_profile = await self.get_user_profile_for_task_design(user_id)
            
            # Get task history
            task_history = await self.get_task_completion_history(user_id)
            
            # Prepare prompt for AI
            prompt = {
                "user_id": user_id,
                "user_profile": user_profile.model_dump(),
                "task_history": task_history.model_dump(),
                "difficulty": difficulty_override or user_profile.preferred_difficulty,
                "verification_type": verification_override or user_profile.preferred_verification,
                "due_in_hours": due_in_hours
            }
            
            # Add preferences for categories if available
            if user_profile.task_preferences:
                prefs = []
                for field_name, field_value in user_profile.task_preferences.model_dump().items():
                    if field_value is not None:
                        prefs.append((field_name, field_value))
                prefs.sort(key=lambda x: x[1], reverse=True)
                if prefs:
                    prompt["preferred_categories"] = [c[0] for c in prefs[:3]]
            
            # Run the agent
            result = await Runner.run(
                self.task_ideation_agent,
                prompt,
                context=task_context,
                run_config={
                    "workflow_name": f"TaskIdeation-{user_id[:8]}",
                    "trace_metadata": {
                        "user_id": user_id,
                        "difficulty": prompt["difficulty"]
                    }
                }
            )
            
            # Process the result
            task_idea = result.final_output
            
            # Check if we got valid output
            required_fields = ["title", "description", "instructions", "category", "verification_type"]
            if not all(field in task_idea for field in required_fields):
                logger.error(f"Missing required fields in generated task: {task_idea}")
                return None
                
            # Create task ID
            task_id = f"task_{uuid.uuid4()}"
            
            # Set due date
            due_at = None
            if due_in_hours is not None:
                due_at = datetime.datetime.now() + datetime.timedelta(hours=due_in_hours)
            
            # Choose reward and punishment
            reward = None
            if custom_reward:
                reward = RewardData(**custom_reward)
            elif task_idea.get("reward"):
                if isinstance(task_idea["reward"], dict):
                    reward = RewardData(**task_idea["reward"])
                else:
                    reward = RewardData(description=str(task_idea["reward"]))
                    
            punishment = None
            if custom_punishment:
                punishment = PunishmentData(**custom_punishment)
            elif task_idea.get("punishment"):
                if isinstance(task_idea["punishment"], dict):
                    punishment = PunishmentData(**task_idea["punishment"])
                else:
                    punishment = PunishmentData(description=str(task_idea["punishment"]))
            
            # Create task
            task = AssignedTask(
                id=task_id,
                user_id=user_id,
                title=task_idea["title"],
                description=task_idea["description"],
                instructions=task_idea.get("instructions", []),
                category=task_idea["category"],
                difficulty=difficulty_override or task_idea.get("difficulty", TaskDifficulty.MODERATE),
                verification_type=verification_override or task_idea.get("verification_type", VerificationType.HONOR),
                verification_instructions=task_idea.get("verification_instructions", "Verify completion as specified."),
                due_at=due_at,
                reward=reward,
                punishment=punishment,
                tags=task_idea.get("tags", [])
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Error generating AI task: {e}")
            return None
    
    async def complete_task(self, 
                         task_id: str,
                         verification_data: Dict[str, Any],
                         completion_notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark a task as completed with verification data.
        """
        async with self._lock:
            # Check if task exists
            if task_id not in self.assigned_tasks:
                return {
                    "success": False,
                    "message": f"Task {task_id} not found"
                }
            
            task = self.assigned_tasks[task_id]
            user_id = task.user_id
            
            # Check if already completed or failed
            if task.completed or task.failed:
                return {
                    "success": False,
                    "message": f"Task already {'completed' if task.completed else 'failed'}",
                    "status": "completed" if task.completed else "failed"
                }
            
            # Use trace to track the task verification process
            with trace(workflow_name="Task Verification", 
                     group_id=f"user_{user_id}",
                     metadata={"user_id": user_id, "task_id": task_id}):
                
                # Update the task context with latest state
                task_context = TaskContext(
                    assigned_tasks=self.assigned_tasks,
                    user_settings=self.user_settings,
                    task_templates=self.task_templates,
                    memory_core=self.memory_core,
                    reward_system=self.reward_system,
                    relationship_manager=self.relationship_manager,
                    submission_progression=self.submission_progression
                )
                
                # Convert verification data to model
                verification_model = VerificationData(**verification_data)
                
                # Verify the task using verification agent
                verification_result = await self._verify_task_completion(task_id, verification_model, task_context)
                
                # Update task state
                completed_successfully = verification_result.get("verified", False)
                rating = verification_result.get("rating", 0.5)
                feedback = verification_result.get("feedback", "Task completion verified.")
                
                task.completed = completed_successfully
                task.failed = not completed_successfully
                task.completed_at = datetime.datetime.now()
                task.verification_data = verification_model
                task.rating = rating
                task.notes = completion_notes
                
                # Update user settings
                settings = await self._get_or_create_user_settings(user_id)
                
                # Remove from active tasks
                if task_id in settings.active_tasks:
                    settings.active_tasks.remove(task_id)
                
                # Add to completed or failed tasks
                if completed_successfully:
                    settings.completed_tasks.append(task_id)
                else:
                    settings.failed_tasks.append(task_id)
                
                # Update completion rate
                total_tasks = len(settings.completed_tasks) + len(settings.failed_tasks)
                if total_tasks > 0:
                    settings.task_completion_rate = len(settings.completed_tasks) / total_tasks
                
                # Add to task history
                history_entry = TaskHistoryEntry(
                    task_id=task_id,
                    title=task.title,
                    category=task.category,
                    completed=completed_successfully,
                    rating=rating,
                    completed_at=task.completed_at.isoformat(),
                    difficulty=task.difficulty
                )
                settings.task_history.append(history_entry)
                
                # Limit history size
                if len(settings.task_history) > 50:
                    settings.task_history = settings.task_history[-50:]
                
                # Process reward or punishment
                reward_result = None
                punishment_result = None
                
                if completed_successfully and task.reward:
                    reward_result = await self._process_reward(user_id, task, rating)
                elif not completed_successfully and task.punishment:
                    punishment_result = await self._process_punishment(user_id, task)
                
                # Update relationship manager if available
                if self.relationship_manager:
                    try:
                        # Update active tasks
                        await self.relationship_manager.update_relationship_attribute(
                            user_id,
                            "active_tasks",
                            [t for t in settings.active_tasks]
                        )
                        
                        # Update task completion stats
                        await self.relationship_manager.update_relationship_attribute(
                            user_id,
                            "task_completion_rate",
                            settings.task_completion_rate
                        )
                        
                        # Adjust obedience trait based on completion
                        if hasattr(self.relationship_manager, "update_user_trait"):
                            if completed_successfully:
                                await self.relationship_manager.update_user_trait(
                                    user_id, "obedient", min(0.1, rating * 0.2)
                                )
                            else:
                                await self.relationship_manager.update_user_trait(
                                    user_id, "obedient", -0.1
                                )
                    except Exception as e:
                        logger.error(f"Error updating relationship data: {e}")
                
                # Update submission progression if available
                if self.submission_progression and hasattr(self.submission_progression, "record_compliance"):
                    try:
                        await self.submission_progression.record_compliance(
                            user_id=user_id,
                            instruction=f"Complete assigned task: {task.title}",
                            complied=completed_successfully,
                            difficulty=self._difficulty_to_float(task.difficulty),
                            context={"task_id": task_id, "category": task.category},
                            defiance_reason=None if completed_successfully else "Task failed or incomplete"
                        )
                    except Exception as e:
                        logger.error(f"Error updating submission progression: {e}")
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        significance = 0.4 + (self._difficulty_to_float(task.difficulty) * 0.2)
                        content = f"User {'completed' if completed_successfully else 'failed'} task '{task.title}' with rating {rating:.2f}."
                        
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=content,
                            tags=["task_completion" if completed_successfully else "task_failure", 
                                  task.category, "femdom"],
                            significance=significance
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                # Prepare sadistic response if task failed and sadistic_responses available
                sadistic_response = None
                if not completed_successfully and self.sadistic_responses:
                    try:
                        sadistic_result = await self.sadistic_responses.generate_sadistic_amusement_response(
                            user_id=user_id,
                            humiliation_level=0.7,
                            category="mockery"
                        )
                        if sadistic_result and "response" in sadistic_result:
                            sadistic_response = sadistic_result["response"]
                    except Exception as e:
                        logger.error(f"Error generating sadistic response: {e}")
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "verified": completed_successfully,
                    "rating": rating,
                    "feedback": feedback,
                    "reward_result": reward_result if completed_successfully else None,
                    "punishment_result": punishment_result if not completed_successfully else None,
                    "sadistic_response": sadistic_response if not completed_successfully else None
                }
    
    async def _verify_task_completion(self, task_id: str, verification_data: VerificationData, 
                                   task_context: TaskContext) -> Dict[str, Any]:
        """Verify task completion using the verification agent."""
        # Get task details
        task = self.assigned_tasks[task_id]
        
        # Handle different verification types
        verification_type = task.verification_type
        
        # For honor-based verification, we trust the user
        if verification_type == VerificationType.HONOR:
            # If they submitted anything, consider it verified
            return {
                "verified": True,
                "rating": 1.0,
                "feedback": "Task completion accepted on your honor."
            }
        
        # For other verification types, use the verification agent
        try:
            # Prepare prompt
            prompt = {
                "task_id": task_id,
                "verification_data": verification_data.model_dump(),
                "verification_type": verification_type,
                "verification_instructions": task.verification_instructions
            }
            
            # Run the agent
            result = await Runner.run(
                self.verification_agent,
                prompt,
                context=task_context,
                run_config={
                    "workflow_name": f"TaskVerification-{task_id[:8]}",
                    "trace_metadata": {
                        "task_id": task_id,
                        "user_id": task.user_id
                    }
                }
            )
            
            # Process the result
            verification_result = result.final_output
            
            # Check if we got valid output
            if "verified" in verification_result:
                return verification_result
                
        except Exception as e:
            logger.error(f"Error verifying task completion: {e}")
        
        # Default verification (fallback)
        has_data = any(v for v in verification_data.model_dump().values() if v)
        return {
            "verified": has_data,
            "rating": 0.7 if has_data else 0.0,
            "feedback": "Task verification processed without agent."
        }
    
    def _choose_appropriate_option(self, options, option_key, user_profile):
        """Choose an appropriate customization option based on user profile."""
        # Default to random selection
        if not options:
            return "[X]"  # Keep placeholder if no options
            
        # For difficulty-related options, use user's preferred difficulty
        if option_key in ["difficulty", "intensity", "duration"]:
            preferred_difficulty = user_profile.preferred_difficulty
            
            # Map preferred difficulty to option index
            difficulty_map = {
                "trivial": 0,
                "easy": 1,
                "moderate": 2,
                "challenging": 3,
                "difficult": 4,
                "extreme": 5
            }
            
            preferred_index = difficulty_map.get(preferred_difficulty, 2)  # Default to moderate
            
            # Choose option based on preferred difficulty
            index = min(preferred_index, len(options) - 1)
            return options[index]
        
        # For count-related options, adjust based on user experience
        elif option_key in ["count", "repetitions", "word_count", "minimum_words"]:
            submission_level = user_profile.submission_level or 1
            
            # Higher submission level = higher counts
            index = min(submission_level - 1, len(options) - 1)
            if index < 0:
                index = 0
                
            return options[index]
            
        # For frequency options, check user's completion rate
        elif option_key in ["frequency", "check_in_frequency"]:
            completion_rate = user_profile.task_completion_rate
            
            # Lower completion rate = less frequent requirements
            if completion_rate < 0.5:
                index = 0  # Least frequent
            elif completion_rate < 0.8:
                index = min(1, len(options) - 1)  # Moderately frequent
            else:
                index = min(2, len(options) - 1)  # More frequent
                
            return options[index]
            
        # Default: random selection
        return random.choice(options)
    
    async def _process_reward(self, user_id: str, task: AssignedTask, rating: float) -> Dict[str, Any]:
        """Process reward for successful task completion."""
        reward_description = None
        if task.reward:
            reward_description = task.reward.description
        
        # Create reward result
        result = {
            "reward_text": reward_description,
            "rating_based_modifier": rating
        }
        
        # Process reward with reward system if available
        if self.reward_system:
            try:
                # Calculate reward based on task difficulty and performance rating
                difficulty_factor = self._difficulty_to_float(task.difficulty)
                reward_value = 0.3 + (difficulty_factor * 0.3) + (rating * 0.4)
                
                reward_signal_result = await self.reward_system.process_reward_signal(
                    self.reward_system.RewardSignal(
                        value=reward_value,
                        source="task_completion",
                        context={
                            "task_id": task.id,
                            "task_category": task.category,
                            "task_difficulty": task.difficulty,
                            "completion_rating": rating
                        }
                    )
                )
                
                result["reward_signal"] = reward_signal_result
                
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        return result
    
    async def _process_punishment(self, user_id: str, task: AssignedTask) -> Dict[str, Any]:
        """Process punishment for failed task."""
        punishment_description = None
        if task.punishment:
            punishment_description = task.punishment.description
        
        # Create punishment result
        result = {
            "punishment_text": punishment_description
        }
        
        # Process punishment with reward system if available
        if self.reward_system:
            try:
                # Calculate negative reward based on task difficulty
                difficulty_factor = self._difficulty_to_float(task.difficulty)
                punishment_value = -0.2 - (difficulty_factor * 0.3)
                
                punishment_signal_result = await self.reward_system.process_reward_signal(
                    self.reward_system.RewardSignal(
                        value=punishment_value,
                        source="task_failure",
                        context={
                            "task_id": task.id,
                            "task_category": task.category,
                            "task_difficulty": task.difficulty
                        }
                    )
                )
                
                result["punishment_signal"] = punishment_signal_result
                
            except Exception as e:
                logger.error(f"Error processing punishment: {e}")
                
        # Generate punishment task if requested
        if task.punishment and task.punishment.generate_punishment_task:
            try:
                # Create a punishment task
                punishment_task_result = await self.assign_task(
                    user_id=user_id,
                    custom_task={
                        "title": f"Punishment: {punishment_description}",
                        "description": f"This task was assigned as punishment for failing the task: {task.title}",
                        "category": TaskCategory.PUNISHMENT,
                        "instructions": [punishment_description],
                        "difficulty": task.difficulty,
                        "verification_type": task.verification_type
                    },
                    due_in_hours=24
                )
                
                if punishment_task_result.get("success", False):
                    result["punishment_task"] = punishment_task_result["task"]
                
            except Exception as e:
                logger.error(f"Error generating punishment task: {e}")
        
        return result
    
    def _difficulty_to_float(self, difficulty: str) -> float:
        """Convert difficulty string to float (0.0-1.0)."""
        difficulty_map = {
            TaskDifficulty.TRIVIAL: 0.1,
            TaskDifficulty.EASY: 0.3,
            TaskDifficulty.MODERATE: 0.5,
            TaskDifficulty.CHALLENGING: 0.7,
            TaskDifficulty.DIFFICULT: 0.85,
            TaskDifficulty.EXTREME: 1.0
        }
        
        return difficulty_map.get(difficulty, 0.5)
    
    def _get_time_remaining(self, task: AssignedTask) -> Dict[str, Any]:
        """Get time remaining until task is due."""
        if not task.due_at:
            return None
            
        now = datetime.datetime.now()
        
        # If already passed
        if now > task.due_at:
            return {
                "overdue": True,
                "hours": (now - task.due_at).total_seconds() / 3600.0
            }
            
        # Time remaining
        remaining = task.due_at - now
        hours_remaining = remaining.total_seconds() / 3600.0
        
        return {
            "overdue": False,
            "hours": hours_remaining,
            "days": hours_remaining / 24.0,
            "display": f"{int(hours_remaining)} hours" if hours_remaining < 24 else f"{int(hours_remaining / 24)} days, {int(hours_remaining % 24)} hours"
        }
    
    async def extend_task_deadline(self, 
                                task_id: str, 
                                additional_hours: int = 24, 
                                reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Extend the deadline for a task.
        
        Args:
            task_id: The task ID to extend
            additional_hours: Hours to add to the deadline
            reason: Optional reason for the extension
            
        Returns:
            Updated task details
        """
        async with self._lock:
            # Check if task exists
            if task_id not in self.assigned_tasks:
                return {
                    "success": False,
                    "message": f"Task {task_id} not found"
                }
            
            task = self.assigned_tasks[task_id]
            
            # Use trace to track the task extension process
            with trace(workflow_name="Task Deadline Extension", 
                     group_id=f"task_{task_id}",
                     metadata={"task_id": task_id, "hours": additional_hours, "reason": reason}):
                
                # Check if task is still active
                if task.completed or task.failed:
                    return {
                        "success": False,
                        "message": f"Cannot extend deadline for {'completed' if task.completed else 'failed'} task"
                    }
                
                # Check if task has a deadline
                if not task.due_at:
                    return {
                        "success": False,
                        "message": "Task does not have a deadline to extend"
                    }
                
                # Store old deadline
                old_deadline = task.due_at
                
                # Extend deadline
                task.due_at = task.due_at + datetime.timedelta(hours=additional_hours)
                
                # Increment extension count
                task.extension_count += 1
                
                # Update task context
                task_context = TaskContext(
                    assigned_tasks=self.assigned_tasks,
                    user_settings=self.user_settings,
                    task_templates=self.task_templates,
                    memory_core=self.memory_core,
                    reward_system=self.reward_system,
                    relationship_manager=self.relationship_manager,
                    submission_progression=self.submission_progression
                )
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        content = f"Extended deadline for task '{task.title}' by {additional_hours} hours."
                        if reason:
                            content += f" Reason: {reason}"
                        
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=content,
                            tags=["task_extension", task.category, "femdom"],
                            significance=0.3
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                # Have the task management agent evaluate this extension
                if self.task_management_agent:
                    try:
                        extension_evaluation = await Runner.run(
                            self.task_management_agent,
                            {
                                "action": "evaluate_deadline_extension",
                                "task_id": task_id,
                                "extension_count": task.extension_count,
                                "additional_hours": additional_hours,
                                "reason": reason
                            },
                            context=task_context
                        )
                        
                        # Get management agent's evaluation
                        response = extension_evaluation.final_output.get("evaluation", {})
                        
                    except Exception as e:
                        logger.error(f"Error evaluating deadline extension: {e}")
                        response = {}
                
                # Generate appropriate messages based on extension count
                message = response.get("message", "")
                if not message:
                    if task.extension_count == 1:
                        message = "Task deadline extended. This is your first extension."
                    elif task.extension_count == 2:
                        message = "Task deadline extended again. This is your final extension."
                    else:
                        message = f"Task deadline extended yet again ({task.extension_count} extensions). This leniency is becoming concerning."
                
                if reason and "reason" not in message.lower():
                    message += f" Reason: {reason}"
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "old_deadline": old_deadline.isoformat(),
                    "new_deadline": task.due_at.isoformat(),
                    "additional_hours": additional_hours,
                    "extension_count": task.extension_count,
                    "message": message
                }
    
    async def cancel_task(self, 
                        task_id: str,
                        reason: str = "cancelled",
                        apply_punishment: bool = False) -> Dict[str, Any]:
        """
        Cancel an assigned task.
        
        Args:
            task_id: The task ID to cancel
            reason: Reason for cancellation
            apply_punishment: Whether to apply punishment for cancellation
            
        Returns:
            Cancellation results
        """
        async with self._lock:
            # Check if task exists
            if task_id not in self.assigned_tasks:
                return {
                    "success": False,
                    "message": f"Task {task_id} not found"
                }
            
            task = self.assigned_tasks[task_id]
            user_id = task.user_id
            
            # Use trace to track the task cancellation process
            with trace(workflow_name="Task Cancellation", 
                     group_id=f"task_{task_id}",
                     metadata={"task_id": task_id, "reason": reason, "apply_punishment": apply_punishment}):
                
                # Check if task is still active
                if task.completed or task.failed:
                    return {
                        "success": False,
                        "message": f"Cannot cancel {'completed' if task.completed else 'failed'} task"
                    }
                
                # Update task context
                task_context = TaskContext(
                    assigned_tasks=self.assigned_tasks,
                    user_settings=self.user_settings,
                    task_templates=self.task_templates,
                    memory_core=self.memory_core,
                    reward_system=self.reward_system,
                    relationship_manager=self.relationship_manager,
                    submission_progression=self.submission_progression
                )
                
                # If we have the management agent, consult it about whether to apply punishment
                punishment_recommendation = False
                if self.task_management_agent and not apply_punishment:
                    try:
                        # Let the agent decide if punishment should be applied based on history and context
                        cancellation_evaluation = await Runner.run(
                            self.task_management_agent,
                            {
                                "action": "evaluate_task_cancellation",
                                "task_id": task_id,
                                "reason": reason,
                                "user_id": user_id
                            },
                            context=task_context
                        )
                        
                        # Check if the agent recommends punishment
                        punishment_recommendation = cancellation_evaluation.final_output.get("recommend_punishment", False)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating task cancellation: {e}")
                
                # Apply punishment if explicitly requested or recommended by the agent
                apply_punishment = apply_punishment or punishment_recommendation
                
                # Update task state
                task.failed = apply_punishment
                task.notes = f"Cancelled: {reason}"
                
                # Update user settings
                settings = await self._get_or_create_user_settings(user_id)
                
                # Remove from active tasks
                if task_id in settings.active_tasks:
                    settings.active_tasks.remove(task_id)
                
                # Add to failed tasks if punishment applied
                if apply_punishment:
                    settings.failed_tasks.append(task_id)
                    
                    # Update completion rate
                    total_tasks = len(settings.completed_tasks) + len(settings.failed_tasks)
                    if total_tasks > 0:
                        settings.task_completion_rate = len(settings.completed_tasks) / total_tasks
                
                # Add to task history
                history_entry = TaskHistoryEntry(
                    task_id=task_id,
                    title=task.title,
                    category=task.category,
                    completed=False,
                    cancelled=True,
                    reason=reason,
                    punishment_applied=apply_punishment,
                    cancelled_at=datetime.datetime.now().isoformat(),
                    difficulty=task.difficulty,
                    completed_at=datetime.datetime.now().isoformat(),
                    rating=None
                )
                settings.task_history.append(history_entry)
                
                # Limit history size
                if len(settings.task_history) > 50:
                    settings.task_history = settings.task_history[-50:]
                
                # Process punishment if requested
                punishment_result = None
                if apply_punishment and task.punishment:
                    punishment_result = await self._process_punishment(user_id, task)
                
                # Update relationship manager if available
                if self.relationship_manager:
                    try:
                        # Update active tasks
                        await self.relationship_manager.update_relationship_attribute(
                            user_id,
                            "active_tasks",
                            [t for t in settings.active_tasks]
                        )
                        
                        if apply_punishment:
                            # Update task completion stats
                            await self.relationship_manager.update_relationship_attribute(
                                user_id,
                                "task_completion_rate",
                                settings.task_completion_rate
                            )
                    except Exception as e:
                        logger.error(f"Error updating relationship data: {e}")
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        content = f"Cancelled task '{task.title}'. Reason: {reason}."
                        if apply_punishment:
                            content += " Punishment was applied."
                        
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=content,
                            tags=["task_cancellation", task.category, "femdom"],
                            significance=0.4 if apply_punishment else 0.3
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "reason": reason,
                    "punishment_applied": apply_punishment,
                    "punishment_result": punishment_result,
                    "management_recommendation": "apply_punishment" if punishment_recommendation else "no_punishment"
                }
    
    async def update_user_settings(self, 
                                user_id: str, 
                                settings_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update task settings for a user with validation.
        
        Args:
            user_id: The user to update settings for
            settings_update: Settings to update
            
        Returns:
            Updated settings
        """
        async with self._lock:
            # Use trace to track the settings update process
            with trace(workflow_name="User Settings Update", 
                     group_id=f"user_{user_id}",
                     metadata={"user_id": user_id}):
                
                # Get or create user settings
                settings = await self._get_or_create_user_settings(user_id)
                
                # Validate settings using task management agent if available
                valid_settings = True
                validation_message = None
                
                if self.task_management_agent:
                    try:
                        # Create task context
                        task_context = TaskContext(
                            assigned_tasks=self.assigned_tasks,
                            user_settings=self.user_settings,
                            task_templates=self.task_templates,
                            memory_core=self.memory_core,
                            reward_system=self.reward_system,
                            relationship_manager=self.relationship_manager,
                            submission_progression=self.submission_progression
                        )
                        
                        # Have agent validate settings
                        validation_result = await Runner.run(
                            self.task_management_agent,
                            {
                                "action": "validate_user_settings",
                                "user_id": user_id,
                                "current_settings": settings.model_dump(),
                                "proposed_updates": settings_update
                            },
                            context=task_context
                        )
                        
                        # Get validation result
                        validation = validation_result.final_output
                        valid_settings = validation.get("valid", True)
                        validation_message = validation.get("message")
                        
                        # If agent provided modified settings, use those instead
                        if not valid_settings and "recommended_settings" in validation:
                            settings_update = validation["recommended_settings"]
                            valid_settings = True
                            
                    except Exception as e:
                        logger.error(f"Error validating settings: {e}")
                
                # If settings are invalid and couldn't be fixed, return error
                if not valid_settings:
                    return {
                        "success": False,
                        "message": validation_message or "Invalid settings",
                        "current_settings": settings.model_dump()
                    }
                
                # Update settings
                if "preferred_difficulty" in settings_update:
                    settings.preferred_difficulty = settings_update["preferred_difficulty"]
                    
                if "preferred_verification" in settings_update:
                    settings.preferred_verification = settings_update["preferred_verification"]
                    
                if "max_concurrent_tasks" in settings_update:
                    settings.max_concurrent_tasks = max(1, min(10, settings_update["max_concurrent_tasks"]))
                    
                if "task_preferences" in settings_update:
                    # Update existing preferences
                    if isinstance(settings_update["task_preferences"], dict):
                        for category, value in settings_update["task_preferences"].items():
                            if hasattr(settings.task_preferences, category):
                                setattr(settings.task_preferences, category, value)
                
                if "customized_rewards" in settings_update:
                    # Replace or extend rewards
                    if isinstance(settings_update["customized_rewards"], list):
                        settings.customized_rewards = [
                            CustomRewardPunishment(**r) if isinstance(r, dict) else r 
                            for r in settings_update["customized_rewards"]
                        ]
                
                if "customized_punishments" in settings_update:
                    # Replace or extend punishments
                    if isinstance(settings_update["customized_punishments"], list):
                        settings.customized_punishments = [
                            CustomRewardPunishment(**p) if isinstance(p, dict) else p 
                            for p in settings_update["customized_punishments"]
                        ]
                
                # Update timestamp
                settings.last_updated = datetime.datetime.now()
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="system",
                            content=f"Updated task settings for user. Changes: {', '.join(settings_update.keys())}",
                            tags=["settings_update", "task_preferences"],
                            significance=0.3
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "settings": {
                        "preferred_difficulty": settings.preferred_difficulty,
                        "preferred_verification": settings.preferred_verification,
                        "max_concurrent_tasks": settings.max_concurrent_tasks,
                        "task_preferences": settings.task_preferences.model_dump(),
                        "task_completion_rate": settings.task_completion_rate,
                        "active_tasks_count": len(settings.active_tasks),
                        "customized_rewards_count": len(settings.customized_rewards),
                        "customized_punishments_count": len(settings.customized_punishments)
                    },
                    "message": validation_message or "Settings updated successfully"
                }
    
    async def create_task_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a custom task template with agent validation.
        
        Args:
            template_data: Template data
            
        Returns:
            Created template details
        """
        # Use trace to track the template creation process
        with trace(workflow_name="Task Template Creation", 
                   metadata={"template_id": template_data.get("id")}):
            
            try:
                # Check required fields
                required_fields = ["id", "title", "description", "category", "difficulty", 
                                "verification_type", "verification_instructions"]
                
                for field in required_fields:
                    if field not in template_data:
                        return {
                            "success": False,
                            "message": f"Missing required field: {field}"
                        }
                
                template_id = template_data["id"]
                
                # Check if ID already exists
                if template_id in self.task_templates:
                    return {
                        "success": False,
                        "message": f"Template ID '{template_id}' already exists"
                    }
                
                # If task ideation agent is available, validate the template
                validated_template = template_data
                if self.task_ideation_agent:
                    try:
                        # Create task context
                        task_context = TaskContext(
                            assigned_tasks=self.assigned_tasks,
                            user_settings=self.user_settings,
                            task_templates=self.task_templates,
                            memory_core=self.memory_core,
                            reward_system=self.reward_system,
                            relationship_manager=self.relationship_manager,
                            submission_progression=self.submission_progression
                        )
                        
                        # Have agent validate and improve the template
                        validation_result = await Runner.run(
                            self.task_ideation_agent,
                            {
                                "action": "validate_task_template",
                                "template": template_data
                            },
                            context=task_context
                        )
                        
                        # Get validation result
                        validation = validation_result.final_output
                        
                        # If template was improved, use the improved version
                        if "improved_template" in validation:
                            validated_template = validation["improved_template"]
                        
                    except Exception as e:
                        logger.error(f"Error validating template: {e}")
                
                # Convert suitable_for_traits to TaskPreferences
                suitable_for_traits = TaskPreferences()
                if "suitable_for_traits" in validated_template:
                    traits_data = validated_template["suitable_for_traits"]
                    if isinstance(traits_data, dict):
                        for trait, value in traits_data.items():
                            if hasattr(suitable_for_traits, trait):
                                setattr(suitable_for_traits, trait, value)
                
                # Convert customization_options to CustomizationOptions
                customization_options = CustomizationOptions()
                if "customization_options" in validated_template:
                    options_data = validated_template["customization_options"]
                    if isinstance(options_data, dict):
                        for option, values in options_data.items():
                            if hasattr(customization_options, option):
                                setattr(customization_options, option, values)
                
                # Create template from validated data
                template = TaskTemplate(
                    id=template_id,
                    title=validated_template["title"],
                    description=validated_template["description"],
                    instructions=validated_template.get("instructions", []),
                    category=validated_template["category"],
                    difficulty=validated_template["difficulty"],
                    verification_type=validated_template["verification_type"],
                    verification_instructions=validated_template["verification_instructions"],
                    estimated_duration_minutes=validated_template.get("estimated_duration_minutes", 30),
                    suitable_for_levels=validated_template.get("suitable_for_levels", [1, 2, 3, 4, 5]),
                    suitable_for_traits=suitable_for_traits,
                    customization_options=customization_options,
                    reward_suggestions=validated_template.get("reward_suggestions", []),
                    punishment_suggestions=validated_template.get("punishment_suggestions", []),
                    tags=validated_template.get("tags", [])
                )
                
                # Add to templates
                self.task_templates[template_id] = template
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="system",
                            content=f"Created new task template: {template.title} ({template_id})",
                            tags=["template_creation", template.category],
                            significance=0.4
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                # Return success with improvements if any
                was_improved = validated_template != template_data
                
                return {
                    "success": True,
                    "message": f"Created task template '{template_id}'" + 
                               (" with AI improvements" if was_improved else ""),
                    "template_id": template_id,
                    "was_improved": was_improved
                }
            
            except Exception as e:
                logger.error(f"Error creating task template: {e}")
                return {
                    "success": False,
                    "message": f"Error creating task template: {str(e)}"
                }
    
    async def get_available_task_templates(self, 
                                       category: Optional[str] = None, 
                                       difficulty: Optional[str] = None,
                                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available task templates with agent recommendations.
        
        Args:
            category: Optional category filter
            difficulty: Optional difficulty filter
            user_id: Optional user ID for personalized recommendations
            
        Returns:
            List of available templates with recommendations
        """
        # Use trace to track the template retrieval process
        with trace(workflow_name="Task Template Retrieval", 
                 metadata={"category": category, "difficulty": difficulty, "user_id": user_id}):
            
            templates = []
            
            for template_id, template in self.task_templates.items():
                # Apply filters if specified
                if category and template.category != category:
                    continue
                    
                if difficulty and template.difficulty != difficulty:
                    continue
                    
                # Format template info
                template_info = {
                    "id": template.id,
                    "title": template.title,
                    "description": template.description,
                    "category": template.category,
                    "difficulty": template.difficulty,
                    "verification_type": template.verification_type,
                    "estimated_duration_minutes": template.estimated_duration_minutes,
                    "customization_options": any(v for v in template.customization_options.model_dump().values() if v),
                    "tags": template.tags
                }
                
                templates.append(template_info)
            
            # If we have a user ID and the task management agent, get personalized recommendations
            recommendations = []
            if user_id and self.task_management_agent:
                try:
                    # Create task context
                    task_context = TaskContext(
                        assigned_tasks=self.assigned_tasks,
                        user_settings=self.user_settings,
                        task_templates=self.task_templates,
                        memory_core=self.memory_core,
                        reward_system=self.reward_system,
                        relationship_manager=self.relationship_manager,
                        submission_progression=self.submission_progression
                    )
                    
                    # Get user profile
                    user_profile = await self.get_user_profile_for_task_design(user_id)
                    
                    # Have agent recommend templates
                    recommendation_result = await Runner.run(
                        self.task_management_agent,
                        {
                            "action": "recommend_templates",
                            "user_id": user_id,
                            "user_profile": user_profile.model_dump(),
                            "available_templates": [t["id"] for t in templates],
                            "category_filter": category,
                            "difficulty_filter": difficulty
                        },
                        context=task_context
                    )
                    
                    # Get recommendations
                    agent_result = recommendation_result.final_output
                    if "recommendations" in agent_result:
                        recommendations = agent_result["recommendations"]
                    
                except Exception as e:
                    logger.error(f"Error getting template recommendations: {e}")
            
            # Sort by category then difficulty by default
            templates.sort(key=lambda t: (t["category"], t["difficulty"]))
            
            return {
                "success": True,
                "count": len(templates),
                "templates": templates,
                "category_filter": category,
                "difficulty_filter": difficulty,
                "recommendations": recommendations
            }
    
    async def process_expired_tasks(self, auto_fail_hours: int = 12) -> Dict[str, Any]:
        """
        Process expired tasks using agent-based decision making.
        
        Args:
            auto_fail_hours: Hours after expiration to automatically fail task
            
        Returns:
            Processing results
        """
        # Use trace to track the expired task processing
        with trace(workflow_name="Expired Task Processing", 
                 metadata={"auto_fail_hours": auto_fail_hours}):
            
            now = datetime.datetime.now()
            expired_tasks = await self.get_expired_tasks()
            
            auto_failed = []
            warned = []
            extended = []
            
            # Create task context
            task_context = TaskContext(
                assigned_tasks=self.assigned_tasks,
                user_settings=self.user_settings,
                task_templates=self.task_templates,
                memory_core=self.memory_core,
                reward_system=self.reward_system,
                relationship_manager=self.relationship_manager,
                submission_progression=self.submission_progression
            )
            
            for task_info in expired_tasks:
                task_id = task_info.task_id
                user_id = task_info.user_id
                overdue_hours = task_info.overdue_hours
                
                # Default action is auto-fail if extremely late
                action = "auto_fail" if overdue_hours >= auto_fail_hours else "warn"
                
                # If we have the task management agent, let it decide what to do
                if self.task_management_agent:
                    try:
                        # Get user settings and history
                        settings = await self._get_or_create_user_settings(user_id)
                        history = await self.get_task_completion_history(user_id)
                        
                        # Have agent decide what to do with this expired task
                        decision_result = await Runner.run(
                            self.task_management_agent,
                            {
                                "action": "process_expired_task",
                                "task_id": task_id,
                                "user_id": user_id,
                                "overdue_hours": overdue_hours,
                                "user_history": history.model_dump(),
                                "task_info": task_info.model_dump()
                            },
                            context=task_context
                        )
                        
                        # Get decision
                        decision = decision_result.final_output
                        if "recommended_action" in decision:
                            action = decision["recommended_action"]
                            
                    except Exception as e:
                        logger.error(f"Error processing expired task decision: {e}")
                
                # Execute the appropriate action
                if action == "auto_fail":
                    # Auto-fail the task
                    await self.complete_task(
                        task_id=task_id,
                        verification_data={"auto_failed": True},
                        completion_notes=f"Auto-failed due to being {overdue_hours:.1f} hours overdue"
                    )
                    auto_failed.append(task_info.model_dump())
                    
                elif action == "auto_extend":
                    # Auto-extend the deadline
                    extension_hours = 24  # Default extension
                    if self.task_management_agent and "extension_hours" in decision:
                        extension_hours = decision.get("extension_hours", 24)
                        
                    await self.extend_task_deadline(
                        task_id=task_id,
                        additional_hours=extension_hours,
                        reason="Automatic extension by system"
                    )
                    extended_info = task_info.model_dump()
                    extended_info["extension_hours"] = extension_hours
                    extended.append(extended_info)
                    
                else:  # warn
                    warned.append(task_info.model_dump())
            
            return {
                "success": True,
                "auto_failed": auto_failed,
                "warned": warned,
                "extended": extended,
                "auto_fail_threshold_hours": auto_fail_hours,
                "total_processed": len(auto_failed) + len(warned) + len(extended)
            }
