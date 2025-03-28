# nyx/core/femdom/task_assignment_system.py

import logging
import asyncio
import datetime
import uuid
import random
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, function_tool, Runner, trace, RunContextWrapper

logger = logging.getLogger(__name__)

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
    verification_data: Optional[Dict[str, Any]] = None
    completed: bool = False
    failed: bool = False
    rating: Optional[float] = None
    reward: Optional[Dict[str, Any]] = None
    punishment: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    extension_count: int = 0
    tags: List[str] = Field(default_factory=list)
    custom_data: Optional[Dict[str, Any]] = None

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
    suitable_for_traits: Dict[str, float] = Field(default_factory=dict)
    customization_options: Dict[str, List[Any]] = Field(default_factory=dict)
    reward_suggestions: List[str] = Field(default_factory=list)
    punishment_suggestions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class UserTaskSettings(BaseModel):
    """Settings and state for a user's tasks."""
    user_id: str
    active_tasks: List[str] = Field(default_factory=list)  # IDs of currently active tasks
    completed_tasks: List[str] = Field(default_factory=list)  # IDs of completed tasks
    failed_tasks: List[str] = Field(default_factory=list)  # IDs of failed tasks
    task_preferences: Dict[str, float] = Field(default_factory=dict)  # category -> preference (0.0-1.0)
    preferred_difficulty: str = "moderate"
    preferred_verification: str = "honor"
    max_concurrent_tasks: int = 3  # Maximum number of concurrent tasks
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    customized_rewards: List[Dict[str, Any]] = Field(default_factory=list)
    customized_punishments: List[Dict[str, Any]] = Field(default_factory=list)
    task_completion_rate: float = 1.0  # Initial perfect rate
    task_history: List[Dict[str, Any]] = Field(default_factory=list)

class TaskAssignmentSystem:
    """System for assigning and tracking real-life tasks for femdom dynamics."""
    
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
        
        # Task ideation agent
        self.task_ideation_agent = self._create_task_ideation_agent()
        self.verification_agent = self._create_verification_agent()
        
        # Task templates management
        self._load_default_task_templates()
        
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
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.7,
                response_format={"type": "json_object"}
            ),
            tools=[
                self.get_user_profile_for_task_design,
                self.get_task_completion_history
            ],
            output_type=Dict[str, Any]
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
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.3,
                response_format={"type": "json_object"}
            ),
            tools=[
                self.get_task_details
            ],
            output_type=Dict[str, Any]
        )
    
    def _load_default_task_templates(self):
        """Load default task templates."""
        # Service Tasks
        self.task_templates["morning_ritual"] = TaskTemplate(
            id="morning_ritual",
            title="Morning Devotion Ritual",
            description="Establish a morning ritual to start each day with submission and devotion.",
            instructions=[
                "Upon waking, immediately kneel beside your bed for [X] minutes",
                "Recite your devotional mantra [X] times",
                "Write a short gratitude note and send it",
                "Complete this ritual before any other morning activities"
            ],
            category=TaskCategory.RITUAL,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit a daily log describing your ritual completion, including time, feelings, and any challenges.",
            estimated_duration_minutes=15,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits={"ritual_oriented": 0.8, "morning_person": 0.7, "consistency": 0.6},
            customization_options={
                "duration": [5, 10, 15, 20, 30],
                "mantra_repetitions": [3, 5, 10, 15],
                "position": ["kneeling", "prostrate", "lotus", "standing with head bowed"]
            },
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
        
        self.task_templates["workout_regimen"] = TaskTemplate(
            id="workout_regimen",
            title="Prescribed Exercise Regimen",
            description="Complete a specific workout routine as prescribed, demonstrating obedience and self-improvement.",
            instructions=[
                "Perform the prescribed workout exactly as specified",
                "Document your progress with before/after photos",
                "Maintain the specified form for all exercises",
                "Follow the exact schedule without deviation"
            ],
            category=TaskCategory.SELF_IMPROVEMENT,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Take photos of yourself before, during, and after the workout. Include timestamp verification.",
            estimated_duration_minutes=45,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits={"fitness_oriented": 0.8, "body_conscious": 0.7, "discipline": 0.6},
            customization_options={
                "workout_type": ["cardio", "strength", "flexibility", "HIIT", "endurance"],
                "duration": [20, 30, 45, 60, 90],
                "intensity": ["light", "moderate", "challenging", "intense"],
                "frequency": ["daily", "3x weekly", "5x weekly"]
            },
            reward_suggestions=[
                "Praise for body improvements",
                "Special attention to worked body parts",
                "Progress tracking rewards"
            ],
            punishment_suggestions=[
                "Additional workout sessions",
                "More challenging exercise variants",
                "Public accountability posting"
            ],
            tags=["fitness", "self-improvement", "discipline", "body"]
        )
        
        self.task_templates["written_submission"] = TaskTemplate(
            id="written_submission",
            title="Written Submission Essay",
            description="Write a detailed essay exploring your feelings of submission and devotion.",
            instructions=[
                "Write [X] words on the assigned topic related to submission",
                "Include personal reflections and genuine feelings",
                "Structure with introduction, body, and conclusion",
                "Use proper grammar and expressive language",
                "Submit by the specified deadline"
            ],
            category=TaskCategory.CREATIVE,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit your complete essay as a text document, following all specified formatting requirements.",
            estimated_duration_minutes=120,
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits={"articulate": 0.8, "introspective": 0.7, "expressive": 0.6},
            customization_options={
                "word_count": [500, 1000, 1500, 2000, 3000],
                "topic_focus": ["service", "submission", "obedience", "devotion", "worship", "desires"],
                "style": ["formal", "personal", "poetic", "analytical", "confessional"]
            },
            reward_suggestions=[
                "Detailed feedback and discussion",
                "Publishing/sharing the writing (with consent)",
                "Using excerpts in future interactions"
            ],
            punishment_suggestions=[
                "Rewriting with stricter requirements",
                "Public reading of the inferior work",
                "Writing lines related to the failure"
            ],
            tags=["writing", "introspection", "creativity", "expression"]
        )
        
        self.task_templates["positional_training"] = TaskTemplate(
            id="positional_training",
            title="Positional Training Session",
            description="Practice and perfect specified positions to demonstrate obedience and body control.",
            instructions=[
                "Assume each assigned position for the specified duration",
                "Practice transitions between positions as directed",
                "Maintain proper form throughout the session",
                "Document your practice with video or photos as instructed"
            ],
            category=TaskCategory.PROTOCOL,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.VIDEO,
            verification_instructions="Record a video demonstrating each position and the transitions between them. Ensure your full body is visible.",
            estimated_duration_minutes=30,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits={"body_conscious": 0.7, "detail_oriented": 0.8, "discipline": 0.6},
            customization_options={
                "positions": ["kneel", "present", "inspection", "await", "display", "custom"],
                "duration_per_position": [1, 2, 5, 10, 15],
                "total_positions": [3, 5, 7, 10],
                "environment": ["private", "semi-private", "public"]
            },
            reward_suggestions=[
                "Advancement to more complex positions",
                "Recognition of skill improvement",
                "Reduced position requirements for one session"
            ],
            punishment_suggestions=[
                "Extended practice sessions",
                "More difficult position variants",
                "Additional positions to master"
            ],
            tags=["positions", "protocol", "training", "body-control"]
        )
        
        self.task_templates["service_project"] = TaskTemplate(
            id="service_project",
            title="Household Service Project",
            description="Complete a specific household task or project as an act of service and devotion.",
            instructions=[
                "Plan the specified service project thoroughly",
                "Gather all necessary supplies before beginning",
                "Complete the project to high standards",
                "Document before, during, and after with photos",
                "Write a reflection on your feelings during service"
            ],
            category=TaskCategory.SERVICE,
            difficulty=TaskDifficulty.CHALLENGING,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Take before and after photos of the project area. Include photos of yourself working on the project. Write a brief reflection on your service.",
            estimated_duration_minutes=180,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits={"service_oriented": 0.9, "detail_oriented": 0.7, "handy": 0.6},
            customization_options={
                "project_type": ["cleaning", "organizing", "decorating", "repairing", "cooking", "gardening"],
                "scale": ["single item", "area", "room", "multiple rooms", "entire home"],
                "complexity": ["simple", "moderate", "complex", "very complex"]
            },
            reward_suggestions=[
                "Praise for quality service",
                "Recognition of effort and skills",
                "Special attention or privilege"
            ],
            punishment_suggestions=[
                "Additional service projects",
                "Redoing aspects that don't meet standards",
                "More demanding service in the future"
            ],
            tags=["service", "household", "domestic", "project"]
        )
        
        self.task_templates["public_symbol"] = TaskTemplate(
            id="public_symbol",
            title="Wearing Symbol of Submission in Public",
            description="Wear a designated item symbolizing your submission in public settings.",
            instructions=[
                "Wear the designated symbol in public for the specified duration",
                "Interact naturally in public while wearing it",
                "Note reactions or feelings throughout the experience",
                "Document with discreet photos as proof of compliance"
            ],
            category=TaskCategory.HUMILIATION,
            difficulty=TaskDifficulty.DIFFICULT,
            verification_type=VerificationType.PHOTO,
            verification_instructions="Take discreet selfies in various public locations showing the symbolic item. Include proof of location when possible.",
            estimated_duration_minutes=240,
            suitable_for_levels=[3, 4, 5],
            suitable_for_traits={"exhibitionist": 0.7, "risk_comfortable": 0.8, "public_comfort": 0.6},
            customization_options={
                "symbol_type": ["collar", "wristband", "anklet", "necklace", "specific clothing item", "subtle mark"],
                "visibility": ["subtle", "noticeable", "obvious"],
                "location_type": ["work", "shopping", "social gathering", "restaurant", "public transport"],
                "duration": [1, 2, 4, 8, "full day"]
            },
            reward_suggestions=[
                "Praise for bravery",
                "Special recognition of public submission",
                "Increasing the intensity/meaning of the symbol"
            ],
            punishment_suggestions=[
                "More obvious symbols in the future",
                "Extended wearing periods",
                "Additional public tasks"
            ],
            tags=["public", "symbol", "humiliation", "exhibition"]
        )
        
        self.task_templates["denial_challenge"] = TaskTemplate(
            id="denial_challenge",
            title="Pleasure Denial Challenge",
            description="Abstain from specified pleasures or activities for a designated period.",
            instructions=[
                "Abstain completely from the specified pleasure/activity",
                "Document any temptations or difficulties experienced",
                "Maintain daily check-ins about your experience",
                "Complete the full duration without any lapses"
            ],
            category=TaskCategory.ENDURANCE,
            difficulty=TaskDifficulty.DIFFICULT,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit daily check-in reports describing your experience, temptations, and how you overcame them.",
            estimated_duration_minutes=0,  # Variable - depends on denial period
            suitable_for_levels=[2, 3, 4, 5],
            suitable_for_traits={"self_control": 0.8, "endurance": 0.7, "discipline": 0.9},
            customization_options={
                "denied_pleasure": ["favorite food", "desserts", "social media", "entertainment", "hobby", "sexual release"],
                "duration_days": [1, 3, 7, 14, 30],
                "check_in_frequency": ["daily", "twice daily", "every other day"]
            },
            reward_suggestions=[
                "Special enjoyment of the denied pleasure afterward",
                "Recognition of willpower and discipline",
                "Symbolic reward related to the denial"
            ],
            punishment_suggestions=[
                "Extended denial period",
                "Additional denials",
                "More stringent reporting requirements"
            ],
            tags=["denial", "endurance", "willpower", "discipline"]
        )
        
        self.task_templates["mantra_repetition"] = TaskTemplate(
            id="mantra_repetition",
            title="Submission Mantra Repetition",
            description="Repeatedly recite assigned submission mantras to internalize their meaning.",
            instructions=[
                "Memorize the assigned submission mantras completely",
                "Recite each mantra the specified number of times daily",
                "Record yourself reciting mantras with proper tone and pacing",
                "Reflect on the meaning of each mantra in writing"
            ],
            category=TaskCategory.OBEDIENCE,
            difficulty=TaskDifficulty.EASY,
            verification_type=VerificationType.VOICE,
            verification_instructions="Record your voice clearly reciting all mantras in the proper manner. Include a brief reflection on their meaning to you.",
            estimated_duration_minutes=20,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits={"verbal": 0.7, "meditative": 0.6, "routine_oriented": 0.7},
            customization_options={
                "mantra_count": [1, 3, 5, 7, 10],
                "repetitions": [10, 25, 50, 100],
                "position": ["kneeling", "standing", "prostrate", "sitting"],
                "recitation_style": ["whispered", "normal voice", "loud and clear", "rhythmic"]
            },
            reward_suggestions=[
                "Praise for memorization and delivery",
                "New mantras of deeper significance",
                "Permission to suggest personal mantras"
            ],
            punishment_suggestions=[
                "Increased repetitions",
                "More difficult mantras",
                "Public recitation (if appropriate)"
            ],
            tags=["mantras", "repetition", "psychological", "conditioning"]
        )
        
        self.task_templates["journaling"] = TaskTemplate(
            id="journaling",
            title="Submission Journaling Practice",
            description="Maintain a detailed journal of your submission experiences and feelings.",
            instructions=[
                "Write at least [X] words in your submission journal daily",
                "Focus on the suggested topics and prompts",
                "Be honest and detailed about your experiences and feelings",
                "Include specific examples and reflections"
            ],
            category=TaskCategory.SELF_IMPROVEMENT,
            difficulty=TaskDifficulty.MODERATE,
            verification_type=VerificationType.TEXT,
            verification_instructions="Submit your journal entries for the specified period, ensuring they meet the minimum word count and address the assigned topics.",
            estimated_duration_minutes=30,
            suitable_for_levels=[1, 2, 3, 4, 5],
            suitable_for_traits={"introspective": 0.9, "articulate": 0.7, "analytical": 0.6},
            customization_options={
                "minimum_words": [200, 300, 500, 750, 1000],
                "frequency": ["daily", "3x weekly", "weekly"],
                "focus_topics": ["progress", "challenges", "desires", "gratitude", "service", "obedience"],
                "format": ["digital", "handwritten", "audio recording", "structured", "free-form"]
            },
            reward_suggestions=[
                "Thoughtful responses to your reflections",
                "Deeper discussion of interesting points",
                "Progressive journal prompts"
            ],
            punishment_suggestions=[
                "More structured journaling requirements",
                "Additional word count requirements",
                "Specific challenging topics to address"
            ],
            tags=["journaling", "reflection", "writing", "self-awareness"]
        )
        
        # Add more task templates as desired...
        logger.info(f"Loaded {len(self.task_templates)} default task templates")
    
    @function_tool
    async def get_user_profile_for_task_design(self, user_id: str) -> Dict[str, Any]:
        """Retrieves user profile data for task customization."""
        try:
            user_profile = {"user_id": user_id}
            
            # Get user settings if available
            if user_id in self.user_settings:
                settings = self.user_settings[user_id]
                user_profile["task_preferences"] = settings.task_preferences
                user_profile["preferred_difficulty"] = settings.preferred_difficulty
                user_profile["preferred_verification"] = settings.preferred_verification
                user_profile["task_completion_rate"] = settings.task_completion_rate
            else:
                # Default values if not set
                user_profile["task_preferences"] = {}
                user_profile["preferred_difficulty"] = "moderate"
                user_profile["preferred_verification"] = "honor"
                user_profile["task_completion_rate"] = 1.0
            
            # Get relationship data if available
            if self.relationship_manager:
                try:
                    relationship = await self.relationship_manager.get_relationship_state(user_id)
                    if relationship:
                        user_profile["trust_level"] = getattr(relationship, "trust", 0.5)
                        user_profile["submission_level"] = getattr(relationship, "submission_level", 1)
                        user_profile["inferred_traits"] = getattr(relationship, "inferred_user_traits", {})
                        user_profile["limits"] = {
                            "hard": getattr(relationship, "hard_limits", []),
                            "soft": getattr(relationship, "soft_limits", [])
                        }
                except Exception as e:
                    logger.error(f"Error retrieving relationship data: {e}")
            
            # Get submission data if available
            if self.submission_progression:
                try:
                    submission_data = await self.submission_progression.get_user_submission_data(user_id)
                    if submission_data:
                        user_profile["submission_level"] = submission_data.get("submission_level", {}).get("id", 1)
                        user_profile["submission_metrics"] = submission_data.get("metrics", {})
                except Exception as e:
                    logger.error(f"Error retrieving submission data: {e}")
            
            return user_profile
            
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    @function_tool
    async def get_task_completion_history(self, user_id: str, limit: int = 5) -> Dict[str, Any]:
        """Retrieves task completion history for a user."""
        try:
            if user_id not in self.user_settings:
                return {"user_id": user_id, "history": [], "stats": {
                    "completion_rate": 1.0, 
                    "total_completed": 0,
                    "total_failed": 0
                }}
            
            settings = self.user_settings[user_id]
            
            # Get recent task history
            recent_history = settings.task_history[-limit:] if settings.task_history else []
            
            # Compile statistics
            stats = {
                "completion_rate": settings.task_completion_rate,
                "total_completed": len(settings.completed_tasks),
                "total_failed": len(settings.failed_tasks),
                "preferred_categories": sorted(
                    settings.task_preferences.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3] if settings.task_preferences else []
            }
            
            return {
                "user_id": user_id, 
                "history": recent_history,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error retrieving task history: {e}")
            return {"user_id": user_id, "error": str(e), "history": []}
    
    @function_tool
    async def get_task_details(self, task_id: str) -> Dict[str, Any]:
        """Retrieves details for a specific task."""
        try:
            if task_id not in self.assigned_tasks:
                return {"error": f"Task {task_id} not found"}
            
            task = self.assigned_tasks[task_id]
            return task.dict()
            
        except Exception as e:
            logger.error(f"Error retrieving task details: {e}")
            return {"error": str(e)}
    
    async def _get_or_create_user_settings(self, user_id: str) -> UserTaskSettings:
        """Get or create user task settings."""
        if user_id not in self.user_settings:
            self.user_settings[user_id] = UserTaskSettings(user_id=user_id)
        return self.user_settings[user_id]
    
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
        
        Args:
            user_id: The user to assign the task to
            template_id: Optional template ID to use
            custom_task: Optional custom task definition
            due_in_hours: Hours until the task is due
            difficulty_override: Optional override for difficulty level
            verification_override: Optional override for verification type
            custom_reward: Optional custom reward for completion
            custom_punishment: Optional custom punishment for failure
            
        Returns:
            Assigned task details
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
            
            # Create task from template or custom definition
            if template_id:
                # Check if template exists
                if template_id not in self.task_templates:
                    return {
                        "success": False,
                        "message": f"Task template '{template_id}' not found",
                        "available_templates": list(self.task_templates.keys())
                    }
                
                # Use template as base
                template = self.task_templates[template_id]
                
                # Create task from template
                task = await self._create_task_from_template(user_id, template, 
                                                        due_in_hours, difficulty_override,
                                                        verification_override, custom_reward,
                                                        custom_punishment)
                
            elif custom_task:
                # Create task from custom definition
                required_fields = ["title", "description", "category", "instructions"]
                for field in required_fields:
                    if field not in custom_task:
                        return {
                            "success": False,
                            "message": f"Missing required field in custom task: {field}"
                        }
                
                # Create task ID
                task_id = f"task_{uuid.uuid4()}"
                
                # Set due date
                due_at = None
                if due_in_hours is not None:
                    due_at = datetime.datetime.now() + datetime.timedelta(hours=due_in_hours)
                
                # Create task
                task = AssignedTask(
                    id=task_id,
                    user_id=user_id,
                    title=custom_task["title"],
                    description=custom_task["description"],
                    category=custom_task["category"],
                    instructions=custom_task["instructions"],
                    difficulty=difficulty_override or custom_task.get("difficulty", TaskDifficulty.MODERATE),
                    verification_type=verification_override or custom_task.get("verification_type", VerificationType.HONOR),
                    verification_instructions=custom_task.get("verification_instructions", "Verify completion as specified."),
                    due_at=due_at,
                    reward=custom_reward or custom_task.get("reward"),
                    punishment=custom_punishment or custom_task.get("punishment"),
                    tags=custom_task.get("tags", []),
                    custom_data=custom_task.get("custom_data")
                )
                
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
            if task.category in settings.task_preferences:
                settings.task_preferences[task.category] += 0.1
            else:
                settings.task_preferences[task.category] = 0.5
            
            # Normalize preferences to keep them in 0-1 range
            max_pref = max(settings.task_preferences.values()) if settings.task_preferences else 1.0
            if max_pref > 1.0:
                for key in settings.task_preferences:
                    settings.task_preferences[key] /= max_pref
            
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
                "reward": task.reward,
                "punishment": task.punishment
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
        if template.customization_options:
            for option_key, option_values in template.customization_options.items():
                if option_values:
                    # Find relevant option placeholder in instructions
                    for i, instruction in enumerate(instructions):
                        if f"[{option_key}]" in instruction or f"[X]" in instruction:
                            # Choose appropriate option value based on user profile
                            chosen_value = self._choose_appropriate_option(option_values, option_key, user_profile)
                            # Replace placeholder with chosen value
                            instructions[i] = instruction.replace(f"[{option_key}]", str(chosen_value)).replace("[X]", str(chosen_value))
        
        # Choose reward and punishment if not provided
        reward = custom_reward
        if not reward and template.reward_suggestions:
            reward = {
                "description": random.choice(template.reward_suggestions),
                "type": "standard"
            }
            
        punishment = custom_punishment
        if not punishment and template.punishment_suggestions:
            punishment = {
                "description": random.choice(template.punishment_suggestions),
                "type": "standard"
            }
        
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
    
    def _choose_appropriate_option(self, options, option_key, user_profile):
        """Choose an appropriate customization option based on user profile."""
        # Default to random selection
        if not options:
            return "[X]"  # Keep placeholder if no options
            
        # For difficulty-related options, use user's preferred difficulty
        if option_key in ["difficulty", "intensity", "duration"]:
            preferred_difficulty = user_profile.get("preferred_difficulty", "moderate")
            
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
            submission_level = user_profile.get("submission_level", 1)
            
            # Higher submission level = higher counts
            index = min(submission_level - 1, len(options) - 1)
            if index < 0:
                index = 0
                
            return options[index]
            
        # For frequency options, check user's completion rate
        elif option_key in ["frequency", "check_in_frequency"]:
            completion_rate = user_profile.get("task_completion_rate", 1.0)
            
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
    
    async def _generate_ai_task(self, 
                             user_id: str,
                             difficulty_override: Optional[str],
                             verification_override: Optional[str],
                             due_in_hours: Optional[int],
                             custom_reward: Optional[Dict[str, Any]],
                             custom_punishment: Optional[Dict[str, Any]]) -> Optional[AssignedTask]:
        """Generate a task using the AI task ideation agent."""
        if not self.task_ideation_agent:
            logger.error("Task ideation agent not available")
            return None
            
        try:
            # Get user profile
            user_profile = await self.get_user_profile_for_task_design(user_id)
            
            # Get task history
            task_history = await self.get_task_completion_history(user_id)
            
            # Prepare prompt for AI
            prompt = {
                "user_id": user_id,
                "user_profile": user_profile,
                "task_history": task_history,
                "difficulty": difficulty_override or user_profile.get("preferred_difficulty", "moderate"),
                "verification_type": verification_override or user_profile.get("preferred_verification", "honor"),
                "due_in_hours": due_in_hours
            }
            
            # Add preferences for categories if available
            if "task_preferences" in user_profile and user_profile["task_preferences"]:
                preferred_categories = sorted(
                    user_profile["task_preferences"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                if preferred_categories:
                    prompt["preferred_categories"] = [c[0] for c in preferred_categories[:3]]
            
            # Run the agent
            result = await Runner.run(
                self.task_ideation_agent,
                prompt,
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
            reward = custom_reward or task_idea.get("reward")
            punishment = custom_punishment or task_idea.get("punishment")
            
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
        
        Args:
            task_id: The task ID to complete
            verification_data: Data verifying the task was completed
            completion_notes: Optional notes about the completion
            
        Returns:
            Completion results
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
            
            # Verify the task using verification agent
            verification_result = await self._verify_task_completion(task_id, verification_data)
            
            # Update task state
            completed_successfully = verification_result.get("verified", False)
            rating = verification_result.get("rating", 0.5)
            feedback = verification_result.get("feedback", "Task completion verified.")
            
            task.completed = completed_successfully
            task.failed = not completed_successfully
            task.completed_at = datetime.datetime.now()
            task.verification_data = verification_data
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
            history_entry = {
                "task_id": task_id,
                "title": task.title,
                "category": task.category,
                "completed": completed_successfully,
                "rating": rating,
                "completed_at": task.completed_at.isoformat(),
                "difficulty": task.difficulty
            }
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
    
    async def _verify_task_completion(self, task_id: str, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify task completion using the verification agent."""
        # Get task details
        task = self.assigned_tasks[task_id]
        
        # Handle different verification types
        verification_type = task.verification_type
        
        # For honor-based verification, we trust the user
        if verification_type == VerificationType.HONOR:
            # If they submitted anything, consider it verified
            if verification_data:
                return {
                    "verified": True,
                    "rating": 1.0,
                    "feedback": "Task completion accepted on your honor."
                }
        
        # For other verification types, use the verification agent
        if self.verification_agent:
            try:
                # Prepare prompt
                prompt = {
                    "task_id": task_id,
                    "verification_data": verification_data,
                    "verification_type": verification_type,
                    "verification_instructions": task.verification_instructions
                }
                
                # Run the agent
                result = await Runner.run(
                    self.verification_agent,
                    prompt,
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
        return {
            "verified": bool(verification_data),
            "rating": 0.7 if verification_data else 0.0,
            "feedback": "Task verification processed without agent."
        }
    
    async def _process_reward(self, user_id: str, task: AssignedTask, rating: float) -> Dict[str, Any]:
        """Process reward for successful task completion."""
        reward_description = None
        if isinstance(task.reward, dict):
            reward_description = task.reward.get("description", "Task completion reward")
        else:
            reward_description = str(task.reward)
        
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
        if isinstance(task.punishment, dict):
            punishment_description = task.punishment.get("description", "Task failure punishment")
        else:
            punishment_description = str(task.punishment)
        
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
        if isinstance(task.punishment, dict) and task.punishment.get("generate_punishment_task", False):
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
            
            # Generate appropriate messages based on extension count
            message = ""
            if task.extension_count == 1:
                message = "Task deadline extended. This is your first extension."
            elif task.extension_count == 2:
                message = "Task deadline extended again. This is your final extension."
            else:
                message = f"Task deadline extended yet again ({task.extension_count} extensions). This leniency is becoming concerning."
            
            if reason:
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
            
            # Check if task is still active
            if task.completed or task.failed:
                return {
                    "success": False,
                    "message": f"Cannot cancel {'completed' if task.completed else 'failed'} task"
                }
            
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
            history_entry = {
                "task_id": task_id,
                "title": task.title,
                "category": task.category,
                "cancelled": True,
                "reason": reason,
                "punishment_applied": apply_punishment,
                "cancelled_at": datetime.datetime.now().isoformat(),
                "difficulty": task.difficulty
            }
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
                "punishment_result": punishment_result
            }
    
    async def get_active_tasks(self, user_id: str) -> Dict[str, Any]:
        """
        Get all active tasks for a user.
        
        Args:
            user_id: The user to get tasks for
            
        Returns:
            List of active tasks
        """
        # Check if user has settings
        if user_id not in self.user_settings:
            return {
                "success": True,
                "user_id": user_id,
                "active_tasks": [],
                "count": 0
            }
        
        settings = self.user_settings[user_id]
        
        # Get all active task IDs
        active_task_ids = settings.active_tasks
        
        # Get task details
        active_tasks = []
        for task_id in active_task_ids:
            if task_id in self.assigned_tasks:
                task = self.assigned_tasks[task_id]
                
                # Format task
                formatted_task = {
                    "task_id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "instructions": task.instructions,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "assigned_at": task.assigned_at.isoformat(),
                    "due_at": task.due_at.isoformat() if task.due_at else None,
                    "verification_type": task.verification_type,
                    "time_remaining": self._get_time_remaining(task) if task.due_at else None
                }
                
                active_tasks.append(formatted_task)
        
        # Sort by due date (closest first)
        active_tasks.sort(key=lambda t: t["due_at"] if t["due_at"] else "9999-12-31T23:59:59")
        
        return {
            "success": True,
            "user_id": user_id,
            "active_tasks": active_tasks,
            "count": len(active_tasks),
            "max_concurrent": settings.max_concurrent_tasks
        }
    
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
    
    async def get_task_details(self, task_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific task.
        
        Args:
            task_id: The task ID to get details for
            
        Returns:
            Task details
        """
        # Check if task exists
        if task_id not in self.assigned_tasks:
            return {
                "success": False,
                "message": f"Task {task_id} not found"
            }
        
        task = self.assigned_tasks[task_id]
        
        # Format task details
        details = {
            "task_id": task.id,
            "user_id": task.user_id,
            "title": task.title,
            "description": task.description,
            "instructions": task.instructions,
            "category": task.category,
            "difficulty": task.difficulty,
            "assigned_at": task.assigned_at.isoformat(),
            "due_at": task.due_at.isoformat() if task.due_at else None,
            "verification_type": task.verification_type,
            "verification_instructions": task.verification_instructions,
            "completed": task.completed,
            "failed": task.failed,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "rating": task.rating,
            "reward": task.reward,
            "punishment": task.punishment,
            "notes": task.notes,
            "extension_count": task.extension_count,
            "tags": task.tags,
            "time_remaining": self._get_time_remaining(task) if task.due_at else None
        }
        
        return {
            "success": True,
            "task": details
        }
    
    async def get_user_task_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's task history.
        
        Args:
            user_id: The user to get statistics for
            
        Returns:
            Task statistics
        """
        # Check if user has settings
        if user_id not in self.user_settings:
            return {
                "success": True,
                "user_id": user_id,
                "statistics": {
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "completion_rate": 0.0,
                    "average_rating": 0.0,
                    "active_tasks": 0
                },
                "category_breakdown": {},
                "difficulty_breakdown": {}
            }
        
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
            category = entry.get("category")
            difficulty = entry.get("difficulty")
            completed = entry.get("completed", False)
            
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
        statistics = {
            "total_tasks": total_tasks,
            "completed_tasks": total_completed,
            "failed_tasks": total_failed,
            "active_tasks": total_active,
            "completion_rate": completion_rate,
            "average_rating": average_rating
        }
        
        # Format breakdowns
        category_breakdown = {
            category: {
                "count": counts["total"],
                "completed": counts["completed"],
                "completion_rate": category_completion_rates.get(category, 0.0)
            }
            for category, counts in category_counts.items()
        }
        
        difficulty_breakdown = {
            difficulty: {
                "count": counts["total"],
                "completed": counts["completed"],
                "completion_rate": difficulty_completion_rates.get(difficulty, 0.0)
            }
            for difficulty, counts in difficulty_counts.items()
        }
        
        return {
            "success": True,
            "user_id": user_id,
            "statistics": statistics,
            "category_breakdown": category_breakdown,
            "difficulty_breakdown": difficulty_breakdown,
            "preferred_categories": sorted(
                settings.task_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            ) if settings.task_preferences else []
        }
    
    async def update_user_settings(self, 
                                user_id: str, 
                                settings_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update task settings for a user.
        
        Args:
            user_id: The user to update settings for
            settings_update: Settings to update
            
        Returns:
            Updated settings
        """
        async with self._lock:
            # Get or create user settings
            settings = await self._get_or_create_user_settings(user_id)
            
            # Update settings
            if "preferred_difficulty" in settings_update:
                settings.preferred_difficulty = settings_update["preferred_difficulty"]
                
            if "preferred_verification" in settings_update:
                settings.preferred_verification = settings_update["preferred_verification"]
                
            if "max_concurrent_tasks" in settings_update:
                settings.max_concurrent_tasks = max(1, min(10, settings_update["max_concurrent_tasks"]))
                
            if "task_preferences" in settings_update:
                # Update existing preferences
                for category, value in settings_update["task_preferences"].items():
                    settings.task_preferences[category] = value
            
            if "customized_rewards" in settings_update:
                # Replace or extend rewards
                if isinstance(settings_update["customized_rewards"], list):
                    settings.customized_rewards = settings_update["customized_rewards"]
            
            if "customized_punishments" in settings_update:
                # Replace or extend punishments
                if isinstance(settings_update["customized_punishments"], list):
                    settings.customized_punishments = settings_update["customized_punishments"]
            
            # Update timestamp
            settings.last_updated = datetime.datetime.now()
            
            return {
                "success": True,
                "user_id": user_id,
                "settings": {
                    "preferred_difficulty": settings.preferred_difficulty,
                    "preferred_verification": settings.preferred_verification,
                    "max_concurrent_tasks": settings.max_concurrent_tasks,
                    "task_preferences": settings.task_preferences,
                    "task_completion_rate": settings.task_completion_rate,
                    "active_tasks_count": len(settings.active_tasks),
                    "customized_rewards_count": len(settings.customized_rewards),
                    "customized_punishments_count": len(settings.customized_punishments)
                }
            }
    
    async def create_task_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a custom task template.
        
        Args:
            template_data: Template data
            
        Returns:
            Created template details
        """
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
            
            # Create template
            template = TaskTemplate(
                id=template_id,
                title=template_data["title"],
                description=template_data["description"],
                instructions=template_data.get("instructions", []),
                category=template_data["category"],
                difficulty=template_data["difficulty"],
                verification_type=template_data["verification_type"],
                verification_instructions=template_data["verification_instructions"],
                estimated_duration_minutes=template_data.get("estimated_duration_minutes", 30),
                suitable_for_levels=template_data.get("suitable_for_levels", [1, 2, 3, 4, 5]),
                suitable_for_traits=template_data.get("suitable_for_traits", {}),
                customization_options=template_data.get("customization_options", {}),
                reward_suggestions=template_data.get("reward_suggestions", []),
                punishment_suggestions=template_data.get("punishment_suggestions", []),
                tags=template_data.get("tags", [])
            )
            
            # Add to templates
            self.task_templates[template_id] = template
            
            return {
                "success": True,
                "message": f"Created task template '{template_id}'",
                "template_id": template_id
            }
        
        except Exception as e:
            logger.error(f"Error creating task template: {e}")
            return {
                "success": False,
                "message": f"Error creating task template: {str(e)}"
            }
    
    def get_available_task_templates(self, category: Optional[str] = None, 
                                  difficulty: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available task templates, optionally filtered.
        
        Args:
            category: Optional category filter
            difficulty: Optional difficulty filter
            
        Returns:
            List of available templates
        """
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
                "customization_options": bool(template.customization_options),
                "tags": template.tags
            }
            
            templates.append(template_info)
        
        # Sort by category then difficulty
        templates.sort(key=lambda t: (t["category"], t["difficulty"]))
        
        return {
            "success": True,
            "count": len(templates),
            "templates": templates,
            "category_filter": category,
            "difficulty_filter": difficulty
        }
    
    async def get_expired_tasks(self) -> List[Dict[str, Any]]:
        """Get all expired/overdue tasks."""
        now = datetime.datetime.now()
        expired_tasks = []
        
        for task_id, task in self.assigned_tasks.items():
            # Check if task is active and has a due date
            if not task.completed and not task.failed and task.due_at:
                # Check if overdue
                if now > task.due_at:
                    expired_tasks.append({
                        "task_id": task_id,
                        "user_id": task.user_id,
                        "title": task.title,
                        "due_at": task.due_at.isoformat(),
                        "overdue_hours": (now - task.due_at).total_seconds() / 3600.0
                    })
        
        # Sort by most overdue first
        expired_tasks.sort(key=lambda t: t["overdue_hours"], reverse=True)
        
        return expired_tasks
    
    async def process_expired_tasks(self, auto_fail_hours: int = 12) -> Dict[str, Any]:
        """
        Process expired tasks, automatically failing those that are extremely late.
        
        Args:
            auto_fail_hours: Hours after expiration to automatically fail task
            
        Returns:
            Processing results
        """
        now = datetime.datetime.now()
        expired_tasks = await self.get_expired_tasks()
        
        auto_failed = []
        warned = []
        
        for task_info in expired_tasks:
            task_id = task_info["task_id"]
            user_id = task_info["user_id"]
            overdue_hours = task_info["overdue_hours"]
            
            # Auto-fail extremely late tasks
            if overdue_hours >= auto_fail_hours:
                await self.complete_task(
                    task_id=task_id,
                    verification_data={"auto_failed": True},
                    completion_notes=f"Auto-failed due to being {overdue_hours:.1f} hours overdue"
                )
                auto_failed.append(task_info)
            else:
                warned.append(task_info)
        
        return {
            "success": True,
            "auto_failed": auto_failed,
            "warned": warned,
            "auto_fail_threshold_hours": auto_fail_hours
        }
