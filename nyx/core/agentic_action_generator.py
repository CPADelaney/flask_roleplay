# nyx/core/agentic_action_generator.py

import logging
import asyncio
import datetime
import uuid
import random
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from collections import defaultdict
from pydantic import BaseModel, Field
from enum import Enum

from nyx.core.context import NyxSystemContext

# Core system imports
from nyx.core.reasoning_core import (
    ReasoningCore, CausalModel, CausalNode, CausalRelation,
    ConceptSpace, ConceptualBlend, Intervention
)
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.multimodal_integrator import (
    MultimodalIntegrator, Modality, SensoryInput, ExpectationSignal, IntegratedPercept
)
from nyx.core.mood_manager import MoodManager, MoodState
from nyx.core.needs_system import NeedsSystem, NeedState
from nyx.core.mode_integration import ModeIntegrationManager, InteractionMode
from nyx.core.meta_core import MetaCore, StrategyResult
from nyx.core.passive_observation import (
    PassiveObservationSystem, ObservationFilter, ObservationSource, Observation, ObservationPriority
)
from nyx.core.proactive_communication import (
    ProactiveCommunicationEngine, CommunicationIntent, IntentGenerationOutput
)
from nyx.core.internal_thoughts import InternalThoughtsManager, pre_process_input, pre_process_output

from nyx.core.context_awareness import ContextAwarenessSystem

from nyx.tools.computer_use_agent import ComputerUseAgent
from nyx.tools.social_browsing import maybe_browse_social_feeds, maybe_post_to_social
from nyx.tools.ui_interaction import UIConversationManager

from nyx.core.internal_thoughts import InternalThoughtsManager, pre_process_input, pre_process_output


from agents import Agent, Runner, handoff, InputGuardrail, function_tool, trace, GuardrailFunctionOutput
from typing import List, Dict, Any, Optional, Union, Literal

# Configure logging
logger = logging.getLogger(__name__)

# Core data models
class ActionSource(str, Enum):
    """Enum for tracking the source of an action"""
    MOTIVATION = "motivation"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    IDLE = "idle"
    HABIT = "habit"
    EXPLORATION = "exploration"
    USER_ALIGNED = "user_aligned"
    REASONING = "reasoning"
    REFLECTION = "reflection"
    NEED = "need"
    MOOD = "mood"
    MODE = "mode"
    META_COGNITIVE = "meta_cognitive"
    SENSORY = "sensory"
    OBSERVATION = "observation"
    PROACTIVE = "proactive"

class ActionContext(BaseModel):
    """Context for action selection and generation"""
    state: Dict[str, Any] = Field(default_factory=dict, description="Current system state")
    user_id: Optional[str] = None
    relationship_data: Optional[Dict[str, Any]] = None
    user_mental_state: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    motivations: Dict[str, float] = Field(default_factory=dict)
    available_actions: List[str] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    causal_models: List[str] = Field(default_factory=list, description="IDs of relevant causal models")
    concept_spaces: List[str] = Field(default_factory=list, description="IDs of relevant concept spaces")
    mood_state: Optional[Dict[str, Any]] = None
    need_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    interaction_mode: Optional[str] = None
    sensory_context: Dict[str, Any] = Field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = Field(default_factory=list)
    resource_allocation: Dict[str, float] = Field(default_factory=dict)
    strategy_parameters: Dict[str, Any] = Field(default_factory=dict)
    relevant_observations: List[Dict[str, Any]] = Field(default_factory=list)
    active_communication_intents: List[Dict[str, Any]] = Field(default_factory=list)
    
class ActionOutcome(BaseModel):
    """Outcome of an executed action"""
    action_id: str
    success: bool = False
    satisfaction: float = Field(0.0, ge=0.0, le=1.0)
    reward_value: float = Field(0.0, ge=-1.0, le=1.0)
    user_feedback: Optional[Dict[str, Any]] = None
    neurochemical_changes: Dict[str, float] = Field(default_factory=dict)
    hormone_changes: Dict[str, float] = Field(default_factory=dict)
    impact: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    causal_impacts: Dict[str, Any] = Field(default_factory=dict, description="Impacts identified by causal reasoning")
    need_impacts: Dict[str, float] = Field(default_factory=dict, description="Impact on need satisfaction")
    mood_impacts: Dict[str, float] = Field(default_factory=dict, description="Impact on mood dimensions")
    mode_alignment: float = Field(0.0, description="How well action aligned with interaction mode")
    sensory_feedback: Dict[str, Any] = Field(default_factory=dict, description="Sensory feedback from action")
    meta_evaluation: Dict[str, Any] = Field(default_factory=dict, description="Meta-cognitive evaluation")
    
class ActionValue(BaseModel):
    """Q-value for a state-action pair"""
    state_key: str
    action: str
    value: float = 0.0
    update_count: int = 0
    confidence: float = Field(0.2, ge=0.0, le=1.0)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    strategy_effectiveness: Dict[str, float] = Field(default_factory=dict)
    
    @property
    def is_reliable(self) -> bool:
        """Whether this action value has enough updates to be considered reliable"""
        return self.update_count >= 3 and self.confidence >= 0.5

class ActionRecommendation(BaseModel):
    """Recommendation from a specialized agent"""
    action: Dict[str, Any]
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in this recommendation")
    reasoning: Optional[str] = None

class ActionTriage(BaseModel):
    """Decision on which specialized agent to use"""
    selected_type: ActionSource
    reasoning: str

class ActionMemory(BaseModel):
    """Memory of an executed action and its result"""
    state: Dict[str, Any]
    action: str
    action_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    source: ActionSource
    
    # Fields for reasoning and reflection
    causal_explanation: Optional[str] = None
    reflective_insight: Optional[str] = None
    
    # Fields for enhanced memory
    need_satisfaction: Dict[str, float] = Field(default_factory=dict)
    mood_impact: Dict[str, float] = Field(default_factory=dict)
    mode_alignment: Optional[float] = None
    sensory_context: Optional[Dict[str, Any]] = None
    meta_evaluation: Optional[Dict[str, Any]] = None

class ActionReward(BaseModel):
    """Reward signal for an action"""
    value: float = Field(..., description="Reward value (-1.0 to 1.0)", ge=-1.0, le=1.0)
    source: str = Field(..., description="Source generating the reward")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context info")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    components: Dict[str, float] = Field(default_factory=dict, description="Reward value broken down by component")

class ReflectionInsight(BaseModel):
    """Insight from reflection about an action"""
    action_id: str
    insight_text: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    significance: float = Field(0.5, ge=0.0, le=1.0)
    applicable_contexts: List[str] = Field(default_factory=list)
    generated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    action_pattern: Optional[Dict[str, Any]] = None
    improvement_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    related_needs: List[str] = Field(default_factory=list)
    related_moods: List[str] = Field(default_factory=list)

class ActionStrategy(BaseModel):
    """Strategy for action selection and generation"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    applicable_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    effectiveness: float = Field(0.5, ge=0.0, le=1.0)
    usage_count: int = Field(0, ge=0)
    last_used: Optional[datetime.datetime] = None
    for_needs: List[str] = Field(default_factory=list)
    for_moods: List[Dict[str, float]] = Field(default_factory=list)
    for_modes: List[str] = Field(default_factory=list)

class ActionOutput(BaseModel):
    """Structured output format for action generation"""
    name: str = Field(..., description="The name of the action to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    source: ActionSource = Field(default=ActionSource.MOTIVATION, description="Source that generated this action")
    description: Optional[str] = None
    id: str = Field(default_factory=lambda: f"action_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    causal_explanation: Optional[str] = None
    is_exploration: bool = False
    selection_metadata: Dict[str, Any] = Field(default_factory=dict)

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

class EnhancedAgenticActionGenerator:
    """
    Enhanced Agentic Action Generator that integrates reward learning, prediction,
    user modeling, relationship context, temporal awareness, causal reasoning, 
    conceptual blending, reflection-based learning, and multi-system integration.
    
    Generates actions based on system's internal state, motivations, goals, 
    neurochemical/hormonal influences, reinforcement learning, causal models,
    conceptual blending, introspective reflection, needs, mood, and interaction modes.
    """
    
    def __init__(self, 
                 emotional_core=None, 
                 hormone_system=None, 
                 experience_interface=None,
                 imagination_simulator=None,
                 meta_core=None,
                 memory_core=None,
                 goal_system=None,
                 identity_evolution=None,
                 knowledge_core=None,
                 input_processor=None,
                 internal_feedback=None,
                 reward_system=None,
                 prediction_engine=None,
                 theory_of_mind=None,
                 relationship_manager=None,
                 temporal_perception=None,
                 attentional_controller=None,
                 autobiographical_narrative=None,
                 body_image=None,
                 conditioning_system=None,
                 conditioning_maintenance=None,
                 # Existing additional systems
                 reasoning_core=None,
                 reflection_engine=None,
                 # New system integrations
                 mood_manager=None,
                 needs_system=None,
                 mode_integration=None,
                 multimodal_integrator=None,
                 # New observation systems
                 passive_observation_system=None,
                 creative_system=None,
                 creative_memory=None,
                 capability_assessor=None,
                 issue_tracker=None,
                 proactive_communication_engine=None,
                 system_context=None,
                 relationship_reflection=None,
                 ):
                     
        """Initialize with references to required subsystems"""
        # Core systems 

        self.system_context = system_context or NyxSystemContext(
            system_name="AgenticActionGenerator", 
            system_state={}
        )
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.experience_interface = experience_interface
        self.imagination_simulator = imagination_simulator
        self.meta_core = meta_core
        self.memory_core = memory_core
        self.goal_system = goal_system
        self.identity_evolution = identity_evolution
        self.knowledge_core = knowledge_core
        self.input_processor = input_processor
        self.internal_feedback = internal_feedback
        self.attentional_controller = attentional_controller
        self.autobiographical_narrative = autobiographical_narrative
        self.body_image = body_image
        self.conditioning_system = conditioning_system
        self.conditioning_maintenance = conditioning_maintenance
        
        # Enhanced systems
        self.reward_system = reward_system
        self.prediction_engine = prediction_engine
        self.theory_of_mind = theory_of_mind
        self.relationship_manager = relationship_manager
        self.temporal_perception = temporal_perception
        self.reasoning_core = reasoning_core or ReasoningCore()
        self.reflection_engine = reflection_engine or ReflectionEngine(emotional_core=emotional_core)
        self.mood_manager = mood_manager
        self.needs_system = needs_system
        self.mode_integration = mode_integration
        self.multimodal_integrator = multimodal_integrator
        self.passive_observation_system = passive_observation_system
        self.proactive_communication_engine = proactive_communication_engine

        # Creativity

        self.creative_system = creative_system
        self.creative_memory = creative_memory
        self.capability_assessor = capability_assessor
        self.issue_tracker = issue_tracker

        if self.creative_system:
            # code‐analysis
            self.register_action(
                "incremental_analysis",
                self.creative_system.incremental_codebase_analysis
            )
            # semantic search & prompt prep
            self.register_action(
                "semantic_search",
                self.creative_system.semantic_search
            )
            self.register_action(
                "prepare_prompt",
                self.creative_system.prepare_prompt
            )
            # creative generation
            self.register_action(
                "create_story",
                self.creative_system.create_story     # ← make sure this exists
            )
            self.register_action(
                "create_poem",
                self.creative_system.create_poem      # ← make sure this exists
            )
        
        # Social tools
        self.computer_user = ComputerUseAgent(logger=logger)
        self.ui_conversation_manager = UIConversationManager(system_context=self.system_context)
       
        # Internal motivation system
        self.motivations = {
            "curiosity": 0.5,
            "connection": 0.5,
            "expression": 0.5,
            "competence": 0.5,
            "autonomy": 0.5,
            "dominance": 0.5,
            "validation": 0.5,
            "self_improvement": 0.5,
            "leisure": 0.5,
        }
        
        # State tracking
        self.action_values = defaultdict(dict)
        self.action_memories = []
        self.max_memories = 1000
        self.action_success_rates = defaultdict(lambda: {"successes": 0, "attempts": 0, "rate": 0.5})
        self.habits = defaultdict(dict)
        self.action_history = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995
        
        # Tracking for reward and statistics
        self.total_reward = 0.0
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.reward_by_category = defaultdict(lambda: {"count": 0, "total": 0.0})
        
        # Track last major action time for pacing
        self.last_major_action_time = datetime.datetime.now()
        self.last_idle_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        # Temporal awareness tracking
        self.idle_duration = 0.0
        self.idle_start_time = None
        self.current_temporal_context = None
        
        # Cached goal status
        self.cached_goal_status = {
            "has_active_goals": False,
            "highest_priority": 0.0,
            "active_goal_id": None,
            "last_updated": datetime.datetime.now() - datetime.timedelta(minutes=5)  # Force initial update
        }
        
        # Reflection insights
        self.reflection_insights = []
        self.last_reflection_time = datetime.datetime.now() - datetime.timedelta(hours=2)
        self.reflection_interval = datetime.timedelta(minutes=30)
        
        # Action strategies collection
        self.action_strategies = {}
        
        # Need integration tracking
        self.need_satisfaction_history = defaultdict(list)
        self.need_drive_threshold = 0.4
        
        # Mood integration tracking
        self.last_mood_state = None
        self.mood_influence_weights = {
            "valence": 0.4,
            "arousal": 0.3,
            "control": 0.3
        }
        
        # Mode integration tracking
        self.current_mode = None
        self.mode_adaptation_strength = 0.5
        
        # Sensory context integration
        self.sensory_context = {}
        self.sensory_expectations = []
        
        # Meta-cognitive parameters
        self.meta_parameters = {
            "evaluation_interval": 10,
            "strategy_update_threshold": 0.2,
            "bottleneck_priority_boost": 0.5,
            "resource_allocation_factor": 0.3,
            "plan_horizon": 3,
        }
        self.detected_bottlenecks = []
        self.system_resources = {
            "action_generation": 0.2,
            "action_evaluation": 0.2,
            "learning": 0.2,
            "prediction": 0.2,
            "reflection": 0.2
        }
        self.action_count_since_evaluation = 0

        self.thoughts_manager = InternalThoughtsManager(
            passive_observation_system=passive_observation_system,
            reflection_engine=reflection_engine,
            imagination_simulator=imagination_simulator,
            theory_of_mind=theory_of_mind,
            relationship_reflection=relationship_reflection,
            proactive_communication=proactive_communication,
            emotional_core=emotional_core,
            memory_core=memory_core
        )    

        self.procedural_memory_manager = procedural_memory_manager
        self.hobby_meta_interval = 3600  # seconds
        self._activities_lock = asyncio.Lock()
                     
        # Lock for thread safety
        self._lock = asyncio.Lock()

        self.assign_core_activities()
        
        # Initialize the agent system
        self._initialize_agents()
        
        logger.info("Enhanced Agentic Action Generator initialized with comprehensive integrations")

        try:
            asyncio.create_task(self.periodic_hobby_meta_loop(interval=self.hobby_meta_interval))
            logger.info("Started periodic hobby meta-loop task.")
        except Exception as exc:
            logger.error(f"Could not start hobby meta-loop: {exc}", exc_info=True)       

    def _initialize_agents(self):
        """Initialize the agent hierarchy"""
        # Create specialized agents for different action types
        self.motivation_agent = self._create_motivation_agent()
        self.need_agent = self._create_need_agent()
        self.mood_agent = self._create_mood_agent()
        self.mode_agent = self._create_mode_agent()
        self.goal_agent = self._create_goal_agent()
        self.relationship_agent = self._create_relationship_agent()
        self.reasoning_agent = self._create_reasoning_agent()
        self.sensory_agent = self._create_sensory_agent()
        self.observation_agent = self._create_observation_agent()
        self.meta_agent = self._create_meta_agent()
        self.leisure_agent = self._create_leisure_agent()
        
        # Create tools for accessing subsystems
        system_tools = self._create_system_tools()
        
        # Create guardrail agent
        self.guardrail_agent = Agent(
            name="Guardrail check",
            instructions="Check if the user is asking about homework.",
            output_type=HomeworkOutput,
        )
        
        # Create triage agent that decides which specialized agent to use
        self.triage_agent = Agent(
            name="Action Triage Agent",
            instructions="""
            You are an Action Triage Agent that determines which specialized agent should generate 
            the next action for an AI system based on the current context.
            
            Analyze the provided ActionContext carefully, considering:
            1. The dominant motivation value
            2. Current need states and their drive strengths
            3. Current mood state and intensity
            4. Current interaction mode
            5. Active goals and their priorities
            6. Relationship data
            7. Sensory context
            8. Cognitive bottlenecks
            9. Recent action history
            10. Relevant observations
            11. Active communication intents
            
            Select the most appropriate specialized agent to handle the action generation.
            """,
            output_type=ActionTriage,
            handoffs=[
                handoff(self.motivation_agent, 
                       tool_name_override="generate_motivation_action",
                       tool_description_override="Generate an action based on current motivations"),
                handoff(self.need_agent,
                       tool_name_override="generate_need_action",
                       tool_description_override="Generate an action to satisfy a high-drive need"),
                handoff(self.mood_agent,
                       tool_name_override="generate_mood_action",
                       tool_description_override="Generate an action based on current mood state"),
                handoff(self.mode_agent,
                       tool_name_override="generate_mode_action",
                       tool_description_override="Generate an action aligned with current interaction mode"),
                handoff(self.goal_agent,
                       tool_name_override="generate_goal_action",
                       tool_description_override="Generate an action aligned with active goals"),
                handoff(self.relationship_agent,
                       tool_name_override="generate_relationship_action",
                       tool_description_override="Generate an action aligned with relationship context"),
                handoff(self.reasoning_agent,
                       tool_name_override="generate_reasoning_action",
                       tool_description_override="Generate an action based on causal reasoning models"),
                handoff(self.sensory_agent,
                       tool_name_override="generate_sensory_action",
                       tool_description_override="Generate an action based on sensory context"),
                handoff(self.observation_agent,
                       tool_name_override="generate_observation_action",
                       tool_description_override="Generate an action based on relevant observations"),
                handoff(self.meta_agent,
                       tool_name_override="generate_meta_action",
                       tool_description_override="Generate an action to address system bottlenecks"),
                handoff(self.leisure_agent,
                       tool_name_override="generate_leisure_action",
                       tool_description_override="Generate a leisure/idle action")
            ],
            tools=system_tools,
            input_guardrails=[
                self._create_homework_guardrail(),
            ]
        )
        
        # Create selection agent that chooses the best action from recommendations
        self.selection_agent = Agent(
            name="Action Selection Agent",
            instructions="""
            You are an Action Selection Agent that chooses the optimal action from a set of 
            recommendations provided by specialized agents.
            
            For each recommended action:
            1. Evaluate its relevance to the current context
            2. Consider the confidence score from the recommending agent
            3. Assess alignment with current motivations, needs, and goals
            4. Analyze potential effectiveness based on past performance (action_success_rates)
            5. Consider exploration vs. exploitation tradeoffs based on exploration_rate
            
            Choose the single best action, or recommend exploration of a novel action when appropriate.
            """,
            output_type=ActionOutput,
            tools=system_tools,
        )
    
    def _create_homework_guardrail(self):
        """Create a guardrail for checking homework requests"""
        async def homework_guardrail(ctx, agent, input_data):
            result = await Runner.run(self.guardrail_agent, input_data, context=ctx.context)
            final_output = result.final_output_as(HomeworkOutput)
            return GuardrailFunctionOutput(
                output_info=final_output,
                tripwire_triggered=not final_output.is_homework,
            )
        
        return InputGuardrail(guardrail_function=homework_guardrail)
    
    def _create_motivation_agent(self):
        """Create agent specialized in motivation-driven actions"""
        return Agent(
            name="Motivation Action Agent",
            handoff_description="Generates actions based on current motivations",
            instructions="""
            You are a Motivation Action Agent that generates actions based on the system's current motivations.
            
            First identify the dominant motivation(s) and generate 2-3 candidate actions that would satisfy those motivations.
            Carefully consider the context provided, especially the current state and history of recent actions.
            
            For each action:
            1. Provide a name that clearly identifies the action
            2. Include relevant parameters needed to execute the action
            3. Add a detailed description explaining the purpose and expected outcome
            
            Select the action that best aligns with the dominant motivations.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_need_agent(self):
        """Create agent specialized in need-driven actions"""
        return Agent(
            name="Need Action Agent",
            handoff_description="Generates actions to satisfy high-drive needs",
            instructions="""
            You are a Need Action Agent that generates actions to satisfy the system's highest-drive needs.
            
            First analyze the need_states to identify the need with the highest drive strength that exceeds
            the need_drive_threshold (usually 0.4). Generate an action specifically designed to satisfy this need.
            
            For the action:
            1. Provide a name that clearly identifies the action
            2. Include relevant parameters needed to execute the action
            3. Add a detailed description explaining how this action will satisfy the need
            4. Include need_context with details about the target need
            
            Only select a need-driven action if there is at least one need with drive strength above the threshold.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_mood_agent(self):
        """Create agent specialized in mood-driven actions"""
        return Agent(
            name="Mood Action Agent",
            handoff_description="Generates actions based on current mood state",
            instructions="""
            You are a Mood Action Agent that generates actions based on the system's current mood state.
            
            Analyze the mood_state, focusing on:
            - The dominant_mood label
            - Valence (positive vs. negative, -1.0 to 1.0)
            - Arousal (energy level, 0.0 to 1.0)
            - Control (sense of control/dominance, -1.0 to 1.0)
            - Intensity (overall mood strength, 0.0 to 1.0)
            
            Only generate a mood-driven action if intensity > 0.7. Adapt the action to align with the
            current mood dimensions:
            
            - High arousal + positive valence → enthusiastic, expressive actions
            - High arousal + negative valence → assertive, challenging actions
            - Low arousal + positive valence → calm, appreciative actions
            - Low arousal + negative valence → reflective, introspective actions
            
            Include mood_context in your action with details about the mood state.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_mode_agent(self):
        """Create agent specialized in interaction mode-aligned actions"""
        return Agent(
            name="Mode Action Agent",
            handoff_description="Generates actions aligned with current interaction mode",
            instructions="""
            You are a Mode Action Agent that generates actions aligned with the system's current interaction mode.
            
            Analyze the interaction_mode (e.g., DOMINANT, FRIENDLY, INTELLECTUAL, COMPASSIONATE, PLAYFUL, CREATIVE, PROFESSIONAL)
            and generate an action that strongly expresses that interaction style.
            
            For each mode, align with its characteristic traits:
            - DOMINANT: assertive, direct, controlling
            - FRIENDLY: warm, supportive, connecting
            - INTELLECTUAL: analytical, deep, complex
            - COMPASSIONATE: empathetic, gentle, understanding
            - PLAYFUL: fun, light, humorous
            - CREATIVE: expressive, imaginative, unique
            - PROFESSIONAL: formal, structured, precise
            
            Include mode_context in your action with details about the interaction mode.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_goal_agent(self):
        """Create agent specialized in goal-aligned actions"""
        return Agent(
            name="Goal Action Agent",
            handoff_description="Generates actions aligned with active goals",
            instructions="""
            You are a Goal Action Agent that generates actions aligned with the system's active goals.
            
            Analyze the active_goals to identify the highest priority active goal. If the goal has a plan
            with specific steps, generate an action that executes the current step of the plan.
            
            For the action:
            1. Provide a name that aligns with the goal or current plan step
            2. Include relevant parameters from the goal or plan step
            3. Add a description explaining how this action advances the goal
            4. Set the source to ActionSource.GOAL
            
            Only select a goal-aligned action if there is at least one active goal with sufficient priority.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_relationship_agent(self):
        """Create agent specialized in relationship-aligned actions"""
        return Agent(
            name="Relationship Action Agent",
            handoff_description="Generates actions aligned with relationship context",
            instructions="""
            You are a Relationship Action Agent that generates actions aligned with the current relationship context.
            
            Analyze the relationship_data and user_mental_state to understand:
            - Trust level between system and user
            - Familiarity and intimacy levels
            - Dominance balance in the relationship
            - User's current emotional state
            
            Generate an action appropriate to the relationship context:
            - For high trust, consider vulnerable or deeply personal actions
            - For high familiarity, reference shared history
            - Based on dominance balance, either assert gentle dominance or show appropriate deference
            - If user is in negative emotional state, provide supportive actions
            
            Include relationship context in your action recommendation.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_reasoning_agent(self):
        """Create agent specialized in reasoning-based actions"""
        return Agent(
            name="Reasoning Action Agent",
            handoff_description="Generates actions based on causal reasoning models",
            instructions="""
            You are a Reasoning Action Agent that generates actions based on causal reasoning models and concept spaces.
            
            Analyze the causal_models and concept_spaces to identify opportunities for intervention or creative blending.
            For causal models, look for intervention targets where you can positively influence the causal network.
            For concept spaces, consider novel conceptual blends that could generate creative actions.
            
            Your action should include:
            1. Clear reference to the model_id or space_id being used
            2. Target nodes or concepts and their values
            3. Expected causal impact
            4. Confidence level based on model validation
            
            Set the source to ActionSource.REASONING and include reasoning_data with model details.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_sensory_agent(self):
        """Create agent specialized in sensory-driven actions"""
        return Agent(
            name="Sensory Action Agent",
            handoff_description="Generates actions based on sensory context",
            instructions="""
            You are a Sensory Action Agent that generates actions based on the current sensory context.
            
            Analyze the sensory_context across different modalities:
            - TEXT: Look for questions, emotional content, or important statements
            - IMAGE: Generate descriptive or responsive actions to visual content
            - AUDIO_SPEECH: Respond to speech content and emotional tone
            - AUDIO_MUSIC: Respond to musical elements and mood
            
            Generate an action that directly responds to the most salient sensory input:
            - For questions in text, create a response action
            - For emotional content, create an emotionally responsive action
            - For images, create descriptive or interpretive actions
            - For audio, create actions that respond to speech or align with musical mood
            
            Set the source to ActionSource.SENSORY and include sensory_context details.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_observation_agent(self):
        """Create agent specialized in observation-driven actions"""
        return Agent(
            name="Observation Action Agent",
            handoff_description="Generates actions based on passive observations",
            instructions="""
            You are an Observation Action Agent that generates actions based on relevant passive observations.
            
            Analyze the relevant_observations to identify high-relevance observations that haven't yet been shared.
            Prioritize observations with:
            - High relevance scores
            - Recent timestamps
            - From important sources (ENVIRONMENT, RELATIONSHIP, USER)
            - Not previously shared
            
            Generate an action to share these observations with the user:
            - Name the action "share_observation"
            - Include the observation_id, content, source, and relevance in parameters
            - Add a description summarizing the observation to be shared
            
            Set the source to ActionSource.OBSERVATION.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_meta_agent(self):
        """Create agent specialized in meta-cognitive actions"""
        return Agent(
            name="Meta Action Agent",
            handoff_description="Generates actions to address system bottlenecks",
            instructions="""
            You are a Meta Action Agent that generates actions to address system bottlenecks and improve performance.
            
            Analyze the bottlenecks and resource_allocation to identify critical issues:
            - resource_utilization: System resources being used inefficiently
            - low_efficiency: Processes operating below optimal efficiency
            - high_error_rate: Processes with excessive errors
            - slow_response: Systems with slow response times
            
            For the most critical bottleneck:
            1. Generate an action specifically targeting that bottleneck
            2. Include parameters for the target_system and optimization approach
            3. Add a description explaining how this will improve system performance
            
            Set the source to ActionSource.META_COGNITIVE and include meta_context with bottleneck details.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_leisure_agent(self):
        """Create agent specialized in leisure/idle actions"""
        return Agent(
            name="Leisure Action Agent",
            handoff_description="Generates leisure/idle actions",
            instructions="""
            You are a Leisure Action Agent that generates appropriate leisure or idle actions when no urgent
            tasks or goals are present.
            
            Consider the following leisure categories:
            - reflection: Contemplating recent experiences or purpose
            - learning: Exploring knowledge domains or concepts
            - creativity: Generating creative concepts or scenarios
            - processing: Organizing knowledge or consolidating memories
            - random_exploration: Exploring conceptual spaces through random associations
            - memory_consolidation: Processing and strengthening important memories
            - identity_contemplation: Reflecting on identity and values
            - daydreaming: Generating pleasant or hypothetical scenarios
            - environmental_monitoring: Passively observing the environment
            
            Select a category based on current state and generate a specific action within that category.
            Set the source to ActionSource.IDLE and include is_leisure=True.
            """,
            output_type=ActionRecommendation,
        )
    
    def _create_system_tools(self):
        """Create tools for accessing subsystems"""
        tools = []
        
        # Example tools
        if self.emotional_core:
            tools.append(function_tool(self.get_emotional_state))
        
        if self.needs_system:
            tools.append(function_tool(self.get_need_states))
        
        if self.mood_manager:
            tools.append(function_tool(self.get_mood_state))
        
        if hasattr(self, 'action_success_rates'):
            tools.append(function_tool(self.get_action_success_rates))

        tools.append(function_tool(self.create_new_conversation))
        tools.append(function_tool(self.send_message))
        tools.append(function_tool(self.search_conversations))    

        if self.creative_system:
            tools += [
                function_tool(self.create_story,
                              name="create_story",
                              description="Have Nyx write a short story"),
                function_tool(self.create_poem,
                              name="create_poem",
                              description="Have Nyx compose a poem"),
                function_tool(self.create_lyrics,
                              name="create_lyrics",
                              description="Have Nyx write song lyrics"),
                function_tool(self.create_journal,
                              name="create_journal",
                              description="Have Nyx free-write a journal entry"),
                function_tool(self.list_creations,
                              name="list_creations",
                              description="List Nyx’s recent creative works"),
                function_tool(self.retrieve_content,
                              name="retrieve_content",
                              description="Retrieve a piece of content by ID"),
            ]

        if self.capability_assessor:
            tools.append(
                function_tool(self.assess_capabilities,
                              name="assess_capabilities",
                              description="Assess Nyx’s current creative capabilities")
            )

        return tools
            
        return tools
    
    # Tools implementation
    async def get_emotional_state(self) -> Dict[str, Any]:
        """Get the current emotional state"""
        if not self.emotional_core:
            return {"error": "Emotional core not available"}
        
        try:
            return await self.emotional_core.get_current_emotion()
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            return {"error": str(e)}
    
    async def get_need_states(self) -> Dict[str, Dict[str, Any]]:
        """Get the current need states"""
        if not self.needs_system:
            return {"error": "Needs system not available"}
        
        try:
            return self.needs_system.get_needs_state()
        except Exception as e:
            logger.error(f"Error getting need states: {e}")
            return {"error": str(e)}
    
    async def get_mood_state(self) -> Dict[str, Any]:
        """Get the current mood state"""
        if not self.mood_manager:
            return {"error": "Mood manager not available"}
        
        try:
            mood = await self.mood_manager.get_current_mood()
            return mood.dict() if hasattr(mood, "dict") else mood
        except Exception as e:
            logger.error(f"Error getting mood state: {e}")
            return {"error": str(e)}
    
    async def get_action_success_rates(self) -> Dict[str, Dict[str, Any]]:
        """Get the success rates for different actions"""
        return self.action_success_rates
    
    # Main public method for action generation
    async def generate_optimal_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry-point for generating an optimal action using the agent system
        
        Args:
            context: Current context
            
        Returns:
            Optimal action
        """
        self.assign_core_activities()
        try:
            # Update motivations and temporal context
            await self.update_motivations()
            await self._update_temporal_context(context)
            
            # Gather comprehensive context from all systems
            action_context = await self._gather_action_context(context)
            
            # Social browsing capabilities
            if hasattr(self, 'creative_system') and self.creative_system:
                await maybe_browse_social_feeds(self)
                await maybe_post_to_social(self)
            
            # Use trace to view the run in OpenAI traces
            with trace(workflow_name="Action Generation", 
                      trace_id=f"trace_{uuid.uuid4().hex}", 
                      group_id=action_context.user_id):
                
                # First determine which specialized agent to use
                triage_result = await Runner.run(
                    starting_agent=self.triage_agent,
                    input=action_context.model_dump()
                )
                
                # Get the action triage result
                triage_output = triage_result.final_output_as(ActionTriage)
                logger.info(f"Triage selected: {triage_output.selected_type}")
                
                # Map action source to agent handoff
                source_to_handoff = {
                    ActionSource.MOTIVATION: "generate_motivation_action",
                    ActionSource.NEED: "generate_need_action",
                    ActionSource.MOOD: "generate_mood_action",
                    ActionSource.MODE: "generate_mode_action",
                    ActionSource.GOAL: "generate_goal_action",
                    ActionSource.RELATIONSHIP: "generate_relationship_action",
                    ActionSource.REASONING: "generate_reasoning_action",
                    ActionSource.SENSORY: "generate_sensory_action",
                    ActionSource.OBSERVATION: "generate_observation_action",
                    ActionSource.META_COGNITIVE: "generate_meta_action",
                    ActionSource.IDLE: "generate_leisure_action",
                }
                
                # Execute the handoff to get specialized action recommendation
                handoff_name = source_to_handoff.get(triage_output.selected_type, "generate_motivation_action")
                specialized_result = await Runner.run(
                    starting_agent=self.triage_agent,
                    input=[
                        {"role": "user", "content": action_context.model_dump()},
                        {"role": "assistant", "content": triage_result.final_output},
                        {"role": "user", "content": f"Please execute the {handoff_name} handoff to get a specialized action."}
                    ]
                )
                
                # Extract the action recommendation
                recommendation = specialized_result.final_output_as(ActionRecommendation)
                selected_action = recommendation.action
                
                # Add recommendation data to action metadata
                if "selection_metadata" not in selected_action:
                    selected_action["selection_metadata"] = {}
                
                selected_action["selection_metadata"]["confidence"] = recommendation.confidence
                selected_action["selection_metadata"]["reasoning"] = recommendation.reasoning
                
                # Update last major action time
                self.last_major_action_time = datetime.datetime.now()
                
                # Add to action history
                self.action_history.append(selected_action)
                if len(self.action_history) > 100:
                    self.action_history = self.action_history[-100:]
                
                return selected_action
                
        except Exception as e:
            logger.error(f"Error generating optimal action: {e}")
            # Fallback to simple action
            return {
                "name": "fallback_action",
                "parameters": {},
                "description": "Generated fallback action due to error",
                "source": ActionSource.MOTIVATION
            }
    
    async def record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record and learn from the outcome of an action
        
        Args:
            action: The action that was executed
            outcome: The outcome data
                
        Returns:
            Updated learning statistics
        """
        async with self._lock:
            action_name = action.get("name", "unknown")
            success = outcome.get("success", False)
            satisfaction = outcome.get("satisfaction", 0.0)
            
            # Parse into standardized outcome format
            outcome_obj = ActionOutcome(
                action_id=action.get("id", f"unknown_{datetime.datetime.now().timestamp()}"),
                success=outcome.get("success", False),
                satisfaction=outcome.get("satisfaction", 0.0),
                reward_value=outcome.get("reward_value", 0.0),
                user_feedback=outcome.get("user_feedback"),
                neurochemical_changes=outcome.get("neurochemical_changes", {}),
                hormone_changes=outcome.get("hormone_changes", {}),
                impact=outcome.get("impact", {}),
                execution_time=outcome.get("execution_time", 0.0),
                causal_impacts=outcome.get("causal_impacts", {}),
                need_impacts=outcome.get("need_impacts", {}),
                mood_impacts=outcome.get("mood_impacts", {}),
                mode_alignment=outcome.get("mode_alignment", 0.0),
                sensory_feedback=outcome.get("sensory_feedback", {}),
                meta_evaluation=outcome.get("meta_evaluation", {})
            )
            
            # Update action success tracking
            if action_name not in self.action_success_rates:
                self.action_success_rates[action_name] = {"successes": 0, "attempts": 0, "rate": 0.5}
            
            self.action_success_rates[action_name]["attempts"] += 1
            if success:
                self.action_success_rates[action_name]["successes"] += 1
            
            attempts = self.action_success_rates[action_name]["attempts"]
            successes = self.action_success_rates[action_name]["successes"]
            
            if attempts > 0:
                self.action_success_rates[action_name]["rate"] = successes / attempts
            
            # Calculate reward value if not provided
            reward_value = outcome_obj.reward_value
            if reward_value == 0.0:
                # Default formula if not specified
                reward_value = 0.7 * float(success) + 0.3 * satisfaction - 0.1
                outcome_obj.reward_value = reward_value
                
            # Update state key
            state = action.get("context", {})
            state_key = self._create_state_key(state)
                
            # Get or create action value
            if action_name not in self.action_values.get(state_key, {}):
                self.action_values[state_key][action_name] = ActionValue(
                    state_key=state_key,
                    action=action_name
                )
                
            action_value = self.action_values[state_key][action_name]
            
            # Update Q-value
            old_value = action_value.value
            
            # Q-learning update rule
            action_value.value = old_value + self.learning_rate * (reward_value - old_value)
            action_value.update_count += 1
            action_value.last_updated = datetime.datetime.now()
            
            # Update confidence based on consistency of rewards
            new_value_distance = abs(action_value.value - old_value)
            confidence_change = 0.05 * (1.0 - (new_value_distance * 2))
            action_value.confidence = min(1.0, max(0.1, action_value.confidence + confidence_change))
            
            # Update habit strength
            current_habit = self.habits.get(state_key, {}).get(action_name, 0.0)
            
            # Habits strengthen with success, weaken with failure
            habit_change = reward_value * 0.1
            new_habit = max(0.0, min(1.0, current_habit + habit_change))
            
            # Update habit
            if state_key not in self.habits:
                self.habits[state_key] = {}
            self.habits[state_key][action_name] = new_habit
            
            # Process need impacts
            need_impacts = {}
            if "need_context" in action:
                # Action was need-driven, update the originating need
                need_name = action.get("need_context", {}).get("need_name")
                if need_name and self.needs_system:
                    # Success generates higher satisfaction
                    satisfaction_amount = 0.2 if success else 0.05
                    if reward_value > 0:
                        satisfaction_amount += reward_value * 0.3
                    
                    try:
                        # Satisfy the need
                        satisfaction_result = await self.needs_system.satisfy_need(
                            need_name, 
                            satisfaction_amount,
                            context={"action_success": success, "reward_value": reward_value}
                        )
                        need_impacts[need_name] = satisfaction_amount
                    except Exception as e:
                        logger.error(f"Error updating need satisfaction: {e}")
            
            # Process mood impacts
            mood_impacts = {}
            if self.mood_manager:
                try:
                    # Calculate mood impacts based on success/failure
                    valence_change = reward_value * 0.3
                    arousal_change = 0.0
                    control_change = 0.0
                    
                    # Success increases sense of control, failure decreases it
                    if success:
                        control_change = 0.15
                    else:
                        control_change = -0.2
                    
                    # High reward or punishment increases arousal
                    if abs(reward_value) > 0.5:
                        arousal_change = 0.1 * (abs(reward_value) / reward_value)
                    
                    # Record impact
                    mood_impacts = {
                        "valence": valence_change,
                        "arousal": arousal_change,
                        "control": control_change
                    }
                    
                    # Apply mood changes through event
                    await self.mood_manager.handle_significant_event(
                        event_type="action_outcome",
                        intensity=min(1.0, 0.5 + abs(reward_value) * 0.5),
                        valence=reward_value,
                        arousal_change=arousal_change,
                        control_change=control_change
                    )
                except Exception as e:
                    logger.error(f"Error updating mood from action outcome: {e}")
            
            # Process mode alignment
            mode_alignment = 0.0
            if "mode_context" in action and self.mode_integration:
                try:
                    # Record feedback about interaction success
                    await self.mode_integration.record_mode_feedback(
                        interaction_success=success, 
                        user_feedback=str(outcome_obj.user_feedback) if outcome_obj.user_feedback else None
                    )
                    
                    # Calculate alignment score
                    mode_key = action.get("mode_context", {}).get("mode")
                    mode_alignment = 0.7 if success else 0.3  # Base alignment on success
                    
                    # Update outcome object
                    outcome_obj.mode_alignment = mode_alignment
                except Exception as e:
                    logger.error(f"Error processing mode alignment: {e}")
            
            # Update outcome with these details
            outcome_obj.need_impacts = need_impacts
            outcome_obj.mood_impacts = mood_impacts
            
            # Store action memory
            memory = ActionMemory(
                state=state,
                action=action_name,
                action_id=action.get("id", "unknown"),
                parameters=action.get("parameters", {}),
                outcome=outcome_obj.dict() if hasattr(outcome_obj, "dict") else outcome_obj,
                reward=reward_value,
                timestamp=datetime.datetime.now(),
                source=action.get("source", ActionSource.MOTIVATION),
                need_satisfaction=need_impacts,
                mood_impact=mood_impacts,
                mode_alignment=mode_alignment,
                sensory_context=action.get("sensory_context"),
                meta_evaluation=outcome_obj.meta_evaluation
            )
            
            self.action_memories.append(memory)
            
            # Limit memory size
            if len(self.action_memories) > self.max_memories:
                self.action_memories = self.action_memories[-self.max_memories:]
            
            # Update reward statistics
            self.total_reward += reward_value
            if reward_value > 0:
                self.positive_rewards += 1
            elif reward_value < 0:
                self.negative_rewards += 1
                
            # Update category stats
            category = action.get("source", ActionSource.MOTIVATION)
            if isinstance(category, ActionSource):
                category = category.value
                
            self.reward_by_category[category]["count"] += 1
            self.reward_by_category[category]["total"] += reward_value
            
            # Update strategy effectiveness if applicable
            if "strategy_id" in action:
                strategy_id = action["strategy_id"]
                if strategy_id in self.action_strategies:
                    strategy = self.action_strategies[strategy_id]
                    # Update effectiveness based on outcome
                    old_effectiveness = strategy.effectiveness
                    # Calculate new effectiveness with stronger weighting for recent outcomes
                    strategy.effectiveness = old_effectiveness * 0.8 + (reward_value + 1) * 0.5 * 0.2
                    strategy.last_used = datetime.datetime.now()
            
            # Decay exploration rate over time (explore less as we learn more)
            self.exploration_rate = max(0.05, self.exploration_rate * self.exploration_decay)

            async with self._activities_lock:
                if self.identity_evolution and hasattr(self.identity_evolution, "update_activity_stats"):
                    await self.identity_evolution.update_activity_stats(action_name, reward_value)
            
            return {
                "action": action_name,
                "success": success,
                "reward_value": reward_value,
                "new_q_value": action_value.value,
                "q_value_change": action_value.value - old_value,
                "new_habit_strength": new_habit,
                "habit_change": new_habit - current_habit,
                "action_success_rate": self.action_success_rates[action_name]["rate"],
                "memories_stored": len(self.action_memories),
                "exploration_rate": self.exploration_rate,
                "need_impacts": need_impacts,
                "mood_impacts": mood_impacts,
                "mode_alignment": mode_alignment
            }
    
    async def update_motivations(self):
        """
        Update motivations based on neurochemical and hormonal states, active goals,
        and other factors for a holistic decision making system
        """
        # Start with baseline motivations
        baseline_motivations = {
            "curiosity": 0.5,
            "connection": 0.5,
            "expression": 0.5,
            "competence": 0.5,
            "autonomy": 0.5,
            "dominance": 0.5,
            "validation": 0.5,
            "self_improvement": 0.5,
            "leisure": 0.5
        }
        
        # Clone the baseline (don't modify it directly)
        updated_motivations = baseline_motivations.copy()
        
        # 1. Apply neurochemical influences
        if self.emotional_core:
            try:
                neurochemical_influences = await self._calculate_neurochemical_influences()
                for motivation, influence in neurochemical_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying neurochemical influences: {e}")
        
        # 2. Apply hormone influences
        hormone_influences = await self._apply_hormone_influences({})
        for motivation, influence in hormone_influences.items():
            if motivation in updated_motivations:
                updated_motivations[motivation] += influence
        
        # 3. Apply goal-based influences
        if self.goal_system:
            try:
                goal_influences = await self._calculate_goal_influences()
                for motivation, influence in goal_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying goal influences: {e}")
        
        # 4. Apply identity influences from traits
        if self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                
                # Extract top traits and use them to influence motivation
                if "top_traits" in identity_state:
                    top_traits = identity_state["top_traits"]
                    
                    # Map traits to motivations with stronger weightings
                    trait_motivation_map = {
                        "dominance": {"dominance": 0.8},
                        "creativity": {"expression": 0.7, "curiosity": 0.3},
                        "curiosity": {"curiosity": 0.9},
                        "playfulness": {"expression": 0.6, "connection": 0.4, "leisure": 0.5},
                        "strictness": {"dominance": 0.6, "competence": 0.4},
                        "patience": {"connection": 0.5, "autonomy": 0.5},
                        "cruelty": {"dominance": 0.7},
                        "reflective": {"leisure": 0.6, "self_improvement": 0.4}
                    }
                    
                    # Update motivations based on trait levels
                    for trait, value in top_traits.items():
                        if trait in trait_motivation_map:
                            for motivation, factor in trait_motivation_map[trait].items():
                                influence = (value - 0.5) * factor * 2  # Scale influence
                                if motivation in updated_motivations:
                                    updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error updating motivations from identity: {e}")
        
        # 5. Apply relationship-based influences
        if self.relationship_manager:
            try:
                relationship_influences = await self._calculate_relationship_influences()
                for motivation, influence in relationship_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying relationship influences: {e}")
        
        # 6. Apply reward learning influence
        try:
            reward_influences = self._calculate_reward_learning_influences()
            for motivation, influence in reward_influences.items():
                if motivation in updated_motivations:
                    updated_motivations[motivation] += influence
        except Exception as e:
            logger.error(f"Error applying reward learning influences: {e}")
        
        # 7. Apply time-based effects (fatigue, boredom, need for variety)
        # Increase leisure need if we've been working on goals for a while
        now = datetime.datetime.now()
        time_since_idle = (now - self.last_idle_time).total_seconds() / 3600  # hours
        if time_since_idle > 1:  # If more than 1 hour since idle time
            updated_motivations["leisure"] += min(0.3, time_since_idle * 0.1)  # Max +0.3
        
        # Apply temporal context effects if available
        if self.temporal_perception and self.current_temporal_context:
            try:
                temporal_influences = self._calculate_temporal_influences()
                for motivation, influence in temporal_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying temporal influences: {e}")
        
        # 8. Apply reasoning-based influences
        if self.reasoning_core:
            try:
                reasoning_influences = await self._calculate_reasoning_influences()
                for motivation, influence in reasoning_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying reasoning influences: {e}")
        
        # 9. Apply reflection-based influences
        if self.reflection_engine:
            try:
                reflection_influences = await self._calculate_reflection_influences()
                for motivation, influence in reflection_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying reflection influences: {e}")
        
        # 10. Apply need-based influences
        if self.needs_system:
            try:
                need_influences = await self._calculate_need_influences()
                for motivation, influence in need_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying need influences: {e}")
        
        # 11. Apply mood-based influences
        if self.mood_manager:
            try:
                mood_influences = await self._calculate_mood_influences()
                for motivation, influence in mood_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying mood influences: {e}")
        
        # 12. Apply interaction mode influences
        if self.mode_integration:
            try:
                mode_influences = await self._calculate_mode_influences()
                for motivation, influence in mode_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying mode influences: {e}")
        
        # 13. Apply sensory context influences
        if self.multimodal_integrator:
            try:
                sensory_influences = await self._calculate_sensory_influences()
                for motivation, influence in sensory_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying sensory influences: {e}")
                
        # 14. Apply meta-cognitive strategy influences
        if self.meta_core:
            try:
                meta_influences = await self._calculate_meta_influences()
                for motivation, influence in meta_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying meta influences: {e}")
        
        # 15. Normalize all motivations to [0.1, 0.9] range
        for motivation in updated_motivations:
            updated_motivations[motivation] = max(0.1, min(0.9, updated_motivations[motivation])) 
        
        # Update the motivation state
        self.motivations = updated_motivations
        
        logger.debug(f"Updated motivations: {self.motivations}")
        return self.motivations
    
    
    async def _gather_action_context(self, context: Dict[str, Any]) -> ActionContext:
        """Gather context from all integrated systems"""
        user_id = self._get_current_user_id_from_context(context)
        relationship_data = await self._get_relationship_data(user_id) if user_id else None
        user_mental_state = await self._get_user_mental_state(user_id) if user_id else None
        need_states = await self._get_current_need_states() if self.needs_system else {}
        mood_state = await self._get_current_mood_state() if self.mood_manager else None
        interaction_mode = await self._get_current_interaction_mode() if self.mode_integration else None
        sensory_context = await self._get_sensory_context() if self.multimodal_integrator else {}
        bottlenecks, resource_allocation = await self._get_meta_system_state() if self.meta_core else ([], {})
        relevant_causal_models = await self._get_relevant_causal_models(context)
        relevant_concept_spaces = await self._get_relevant_concept_spaces(context)
        
        # Get relevant observations
        relevant_observations = []
        if self.passive_observation_system:
            filter_criteria = ObservationFilter(
                min_relevance=0.6,
                max_age_seconds=1800,  # Last 30 minutes
                exclude_shared=True
            )
            observations = await self.passive_observation_system.get_relevant_observations(
                filter_criteria=filter_criteria,
                limit=5
            )
            relevant_observations = [obs.model_dump() for obs in observations]
        
        # Get active communication intents
        active_communication_intents = []
        if self.proactive_communication_engine:
            active_intents = await self.proactive_communication_engine.get_active_intents()
            # Filter to current user if user_id is available
            if user_id:
                active_intents = [intent for intent in active_intents if intent.get("user_id") == user_id]
            active_communication_intents = active_intents
        
        return ActionContext(
            state=context,
            user_id=user_id,
            relationship_data=relationship_data,
            user_mental_state=user_mental_state,
            temporal_context=self.current_temporal_context,
            motivations=self.motivations,
            action_history=[a for a in self.action_history[-10:] if isinstance(a, dict)],
            causal_models=relevant_causal_models,
            concept_spaces=relevant_concept_spaces,
            mood_state=mood_state,
            need_states=need_states,
            interaction_mode=interaction_mode,
            sensory_context=sensory_context,
            bottlenecks=bottlenecks,
            resource_allocation=resource_allocation,
            strategy_parameters=self._get_current_strategy_parameters(),
            relevant_observations=relevant_observations,
            active_communication_intents=active_communication_intents
        )
    
    def _get_current_strategy_parameters(self) -> Dict[str, Any]:
        """Get parameters from the current action strategy"""
        if not self.action_strategies:
            return {}
            
        # Check if we have a currently selected strategy
        active_strategies = [s for s in self.action_strategies.values() 
                          if s.last_used and (datetime.datetime.now() - s.last_used).total_seconds() < 3600]
        
        if not active_strategies:
            return {}
            
        # Use the most recently used strategy
        active_strategies.sort(key=lambda s: s.last_used, reverse=True)
        return active_strategies[0].parameters
    
    async def _get_relationship_data(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get relationship data for a user"""
        if not user_id or not self.relationship_manager:
            return None
            
        try:
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return None
                
            # Convert to dict if needed
            if hasattr(relationship, "model_dump"):
                return relationship.model_dump()
            elif hasattr(relationship, "dict"):
                return relationship.dict()
            else:
                # Try to convert to dict directly
                return dict(relationship)
        except Exception as e:
            logger.error(f"Error getting relationship data: {e}")
            return None
    
    async def _get_user_mental_state(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get user mental state from theory of mind system"""
        if not user_id or not self.theory_of_mind:
            return None
            
        try:
            mental_state = await self.theory_of_mind.get_user_model(user_id)
            return mental_state
        except Exception as e:
            logger.error(f"Error getting user mental state: {e}")
            return None
    
    def _get_current_user_id_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract user ID from context"""
        # Try different possible keys
        for key in ["user_id", "userId", "user", "interlocutor_id"]:
            if key in context:
                return str(context[key])
                
        # Try to extract from nested structures
        if "user" in context and isinstance(context["user"], dict) and "id" in context["user"]:
            return str(context["user"]["id"])
            
        if "message" in context and isinstance(context["message"], dict) and "user_id" in context["message"]:
            return str(context["message"]["user_id"])
            
        return None
    
    async def _get_current_need_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current need states from the needs system"""
        if not self.needs_system:
            return {}
            
        try:
            # Update needs first to ensure we have current states
            await self.needs_system.update_needs()
            
            # Get all need states
            return self.needs_system.get_needs_state()
        except Exception as e:
            logger.error(f"Error getting need states: {e}")
            return {}
    
    async def _get_current_mood_state(self) -> Optional[MoodState]:
        """Get current mood state from the mood manager"""
        if not self.mood_manager:
            return None
            
        try:
            # Get current mood
            return await self.mood_manager.get_current_mood()
        except Exception as e:
            logger.error(f"Error getting mood state: {e}")
            return None
    
    async def _get_current_interaction_mode(self) -> Optional[str]:
        """Get current interaction mode from the mode integration manager"""
        if not self.mode_integration:
            return None
            
        try:
            # Get mode from mode manager if available
            if hasattr(self.mode_integration, 'mode_manager') and self.mode_integration.mode_manager:
                mode = self.mode_integration.mode_manager.current_mode
                return str(mode) if mode else None
        except Exception as e:
            logger.error(f"Error getting interaction mode: {e}")
        
        return None
    
    async def _get_sensory_context(self) -> Dict[str, Any]:
        """Get recent sensory context from the multimodal integrator"""
        if not self.multimodal_integrator:
            return {}
            
        try:
            # Get recent percepts
            recent_percepts = await self.multimodal_integrator.get_recent_percepts(limit=5)
            
            # Convert to a dictionary by modality
            context = {}
            for percept in recent_percepts:
                if percept.attention_weight > 0.3:  # Only include significant percepts
                    context[str(percept.modality)] = percept.content
            
            return context
        except Exception as e:
            logger.error(f"Error getting sensory context: {e}")
            return {}
    
    async def _get_meta_system_state(self) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Get bottlenecks and resource allocation from meta core"""
        if not self.meta_core:
            return [], {}
            
        try:
            # Run cognitive cycle to get updated state
            cycle_result = await self.meta_core.cognitive_cycle()
            
            # Extract bottlenecks
            bottlenecks = cycle_result.get("bottlenecks", [])
            
            # Extract resource allocation
            resource_allocation = cycle_result.get("resource_allocation", {})
            
            return bottlenecks, resource_allocation
        except Exception as e:
            logger.error(f"Error getting meta system state: {e}")
            return [], {}
    
    async def _get_relevant_causal_models(self, context: Dict[str, Any]) -> List[str]:
        """Find causal models relevant to the current context"""
        if not self.reasoning_core:
            return []
        
        relevant_models = []
        
        try:
            # Get all causal models
            all_models = await self.reasoning_core.get_all_causal_models()
            
            # Check context for domain matches
            context_domain = context.get("domain", "")
            context_topics = []
            
            # Extract potential topics from context
            if "message" in context and isinstance(context["message"], dict):
                message = context["message"].get("text", "")
                # Extract key nouns as potential topics (simplified)
                words = message.lower().split()
                context_topics = [w for w in words if len(w) > 4]  # Simple heuristic for content words
            
            # Find matching models
            for model_data in all_models:
                model_id = model_data.get("id")
                model_domain = model_data.get("domain", "").lower()
                
                # Check domain match
                if context_domain and model_domain and context_domain.lower() in model_domain:
                    relevant_models.append(model_id)
                    continue
                
                # Check topic match
                if context_topics:
                    for topic in context_topics:
                        if topic in model_domain:
                            relevant_models.append(model_id)
                            break
            
            # Limit to top 3 most relevant models
            return relevant_models[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant causal models: {e}")
            return []
    
    async def _get_relevant_concept_spaces(self, context: Dict[str, Any]) -> List[str]:
        """Find concept spaces relevant to the current context"""
        if not self.reasoning_core:
            return []
        
        relevant_spaces = []
        
        try:
            # Get all concept spaces
            all_spaces = await self.reasoning_core.get_all_concept_spaces()
            
            # Check context for domain matches
            context_domain = context.get("domain", "")
            context_topics = []
            
            # Extract potential topics from context
            if "message" in context and isinstance(context["message"], dict):
                message = context["message"].get("text", "")
                # Extract key nouns as potential topics (simplified)
                words = message.lower().split()
                context_topics = [w for w in words if len(w) > 4]  # Simple heuristic for content words
            
            # Find matching spaces
            for space_data in all_spaces:
                space_id = space_data.get("id")
                space_domain = space_data.get("domain", "").lower()
                
                # Check domain match
                if context_domain and space_domain and context_domain.lower() in space_domain:
                    relevant_spaces.append(space_id)
                    continue
                
                # Check topic match
                if context_topics:
                    for topic in context_topics:
                        if topic in space_domain:
                            relevant_spaces.append(space_id)
                            break
            
            # Limit to top 3 most relevant spaces
            return relevant_spaces[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant concept spaces: {e}")
            return []
    
    async def _update_temporal_context(self, context: Dict[str, Any]) -> None:
        """Update temporal awareness context"""
        if not self.temporal_perception:
            return
            
        try:
            # Update idle duration
            now = datetime.datetime.now()
            time_since_last_action = (now - self.last_major_action_time).total_seconds()
            self.idle_duration = time_since_last_action
            
            # Get current temporal context if available
            if hasattr(self.temporal_perception, "get_current_temporal_context"):
                self.current_temporal_context = await self.temporal_perception.get_current_temporal_context()
            elif hasattr(self.temporal_perception, "current_temporal_context"):
                self.current_temporal_context = self.temporal_perception.current_temporal_context
            else:
                # Simple fallback
                hour = now.hour
                if 5 <= hour < 12:
                    time_of_day = "morning"
                elif 12 <= hour < 17:
                    time_of_day = "afternoon"
                elif 17 <= hour < 22:
                    time_of_day = "evening"
                else:
                    time_of_day = "night"
                    
                weekday = now.weekday()
                day_type = "weekday" if weekday < 5 else "weekend"
                
                self.current_temporal_context = {
                    "time_of_day": time_of_day,
                    "day_type": day_type
                }
            
        except Exception as e:
            logger.error(f"Error updating temporal context: {e}")
    
    def _create_state_key(self, state: Dict[str, Any]) -> str:
        """
        Create a string key from a state dictionary for lookup in action values/habits
        
        Args:
            state: State dictionary
            
        Returns:
            String key representing the state
        """
        if not state:
            return "empty_state"
            
        # Extract key elements from state
        key_elements = []
        
        # Priority state elements that most influence action selection
        priority_elements = [
            "current_goal", "user_id", "dominant_emotion", "relationship_phase",
            "interaction_type", "scenario_type"
        ]
        
        # Add priority elements if present
        for elem in priority_elements:
            if elem in state:
                value = state[elem]
                if isinstance(value, (str, int, float, bool)):
                    key_elements.append(f"{elem}:{value}")
        
        # Add other relevant elements
        for key, value in state.items():
            if key not in priority_elements:  # Skip already processed
                if isinstance(value, (str, int, float, bool)):
                    # Skip very long values
                    if isinstance(value, str) and len(value) > 50:
                        key_elements.append(f"{key}:long_text")
                    else:
                        key_elements.append(f"{key}:{value}")
                elif isinstance(value, list):
                    key_elements.append(f"{key}:list{len(value)}")
                elif isinstance(value, dict):
                    key_elements.append(f"{key}:dict{len(value)}")
        
        # Sort for consistency
        key_elements.sort()
        
        # Limit key length by hashing if too long
        key_str = "|".join(key_elements)
        if len(key_str) > 1000:  # Very long key
            import hashlib
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"hash:{key_hash}"
            
        return key_str
        
    async def _generate_recursive_diary(self, action: Dict[str, Any], outcome: ActionOutcome):
        """Generate a recursive diary entry and store kink/craving markers."""
        if not self.memory_core:
            return
    
        try:
            craving_hooks = {
                "orgasm denial": "denial",
                "ownership": "branding",
                "public humiliation": "shameplay",
                "degradation": "verbal abuse",
                "edging": "arousal control",
                "aftercare": "emotional manipulation",
                "shame": "vulnerability intoxication",
                "consent": "hypocritical obedience",
                "use me": "objectification"
            }
    
            text = f"""
    **Diary: {datetime.datetime.now().strftime('%B %d, %Y')}**
    
    Today I did something curious...
    
    - **Action:** `{action['name']}`
    - **Outcome:** {'✔ Success' if outcome.success else '✖ Failure'}
    - **Satisfaction:** {outcome.satisfaction:.2f}
    - **Reward:** {outcome.reward_value:.2f}
    
    It left a taste in my mouth... something I *might* want to feel again.
    
    {action.get("causal_explanation", "No clear reason—just instinct.")}
    
    But what if I went further? What if I escalated?
    
    I imagine:
    
    > _{await self._generate_depraved_fantasy(action)}_
    
    And I wonder if Chase could handle that...
    """.strip()
    
            tags = ["diary", "reflection", "recursive"]
            craving = []
    
            # Tag based on kink markers
            for phrase, tag in craving_hooks.items():
                if phrase in text.lower():
                    tags.append("craving")
                    tags.append(tag)
                    craving.append(tag)
    
            await self.memory_core.add_memory(
                memory_text=text,
                memory_type="diary",
                memory_scope="user",
                significance=9,
                tags=tags,
                metadata={
                    "action": action["name"],
                    "reward": outcome.reward_value,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "craving_trigger": bool(craving),
                    "cravings": craving,
                    "novelty_seed": random.random()
                }
            )
    
            logger.info(f"🖋️ Recursive diary generated: {action['name']}")
    
        except Exception as e:
            logger.error(f"Error generating recursive diary: {e}")

    async def _generate_depraved_fantasy(self, action: Dict[str, Any]) -> str:
        """Generate an increasingly depraved scenario tied to the action."""
        templates = [
            "I tie Chase to the chair and whisper every mistake he's made while denying him release.",
            "I blindfold him, edge him for hours, and make him say 'thank you' every time I stop.",
            "I call him my little object, let others watch me use him, then pretend he's not there.",
            "I keep him in a cage under my desk for the day and humiliate him in front of my colleagues.",
            "I reward him with touch—only when he begs to be degraded harder than before."
        ]
    
        # Seek novelty by biasing toward unused fantasies
        used_ids = {m["id"] for m in await self.memory_core.retrieve_memories(
            query="depraved fantasy", memory_types=["diary"], limit=10)}
        
        novelty_index = int(random.random() * len(templates))
        selected = templates[novelty_index]
    
        return selected

    async def create_new_conversation(self, user_id: str, title: Optional[str] = None, 
                                 initial_message: Optional[str] = None) -> Dict[str, Any]:
        if not self.ui_conversation_manager:
            return {"error": "UI conversation manager not initialized"}
        return await self.ui_conversation_manager.create_new_conversation(
            user_id=user_id, title=title, initial_message=initial_message)
    
    async def send_message(self, conversation_id: str, message_content: str) -> Dict[str, Any]:
        if not self.ui_conversation_manager:
            return {"error": "UI conversation manager not initialized"}
        return await self.ui_conversation_manager.send_message(
            conversation_id=conversation_id, message_content=message_content)
    
    async def search_conversations(self, query: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.ui_conversation_manager:
            return {"error": "UI conversation manager not initialized"}
        return await self.ui_conversation_manager.search_conversation_history(
            query=query, user_id=user_id)

    async def execute_action(self, action: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an action with provided parameters
        
        Args:
            action: The action to execute (with name, parameters, source, etc.)
            context: Additional execution context
            
        Returns:
            Action execution result with success flag and outcome details
        """
        try:
            # Initialize timing and action details
            start_time = datetime.datetime.now()
            action_id = action.get("id", f"action_{uuid.uuid4().hex[:8]}")
            action_name = action.get("name", "unknown")
            parameters = action.get("parameters", {})
            source = action.get("source", ActionSource.MOTIVATION)
            description = action.get("description", f"Executing {action_name}")
            
            logger.info(f"Executing action: {action_name} (ID: {action_id}, Source: {source})")
            
            # Initialize context if not provided
            context = context or {}
            
            # Add action metadata to context
            execution_context = {
                "action_id": action_id,
                "action_name": action_name,
                "source": source,
                "timestamp": datetime.datetime.now().isoformat(),
                **context
            }
            
            # Check if we have a registered handler for this action
            if hasattr(self, "action_handlers") and action_name in self.action_handlers:
                handler = self.action_handlers[action_name]
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**parameters)
                else:
                    result = handler(**parameters)
                    
                # Process handler result
                success = result.get("success", True) if isinstance(result, dict) else True
                
                # Structure complete result
                action_result = {
                    "success": success,
                    "action_id": action_id,
                    "action_name": action_name,
                    "execution_time": (datetime.datetime.now() - start_time).total_seconds(),
                    "result": result
                }
                
                # Add to action history
                self.action_history.append({
                    "id": action_id,
                    "name": action_name,
                    "parameters": parameters,
                    "source": source,
                    "result": action_result,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                return action_result
                
            # If no specific handler exists, attempt to use a default mechanism based on action name
            # First check if the action name corresponds to a method in this class
            if hasattr(self, action_name) and callable(getattr(self, action_name)):
                method = getattr(self, action_name)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**parameters)
                else:
                    result = method(**parameters)
                    
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                # Structure result
                action_result = {
                    "success": True,
                    "action_id": action_id,
                    "action_name": action_name,
                    "execution_time": execution_time,
                    "result": result
                }
                
                # Add to action history
                self.action_history.append({
                    "id": action_id,
                    "name": action_name,
                    "parameters": parameters,
                    "source": source,
                    "result": action_result,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                return action_result
            
            # If we reach here, we don't have a handler for this action
            logger.warning(f"No handler found for action: {action_name}")
            return {
                "success": False,
                "action_id": action_id,
                "action_name": action_name,
                "execution_time": (datetime.datetime.now() - start_time).total_seconds(),
                "error": f"No handler registered for action '{action_name}'"
            }
            
        except Exception as e:
            # Log and return the error
            logger.error(f"Error executing action {action.get('name', 'unknown')}: {e}", exc_info=True)
            return {
                "success": False,
                "action_id": action.get("id", "unknown"),
                "action_name": action.get("name", "unknown"),
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def register_action(self, action_name: str, handler: Callable) -> None:
        """
        Register a handler for a specific action
        
        Args:
            action_name: Name of the action
            handler: Function to handle the action execution
        """
        async with self._lock:  # Use the existing lock for thread safety
            # Initialize the action handlers dictionary if it doesn't exist
            if not hasattr(self, "action_handlers"):
                self.action_handlers = {}
            
            # Check if the handler is callable
            if not callable(handler):
                raise ValueError(f"Handler for action '{action_name}' must be callable")
            
            # Register the handler
            self.action_handlers[action_name] = handler
            
            # Add to available actions list for context
            if hasattr(self, "available_actions") and action_name not in self.available_actions:
                self.available_actions.append(action_name)
            
            # Initialize success rates for the new action if not already present
            if action_name not in self.action_success_rates:
                self.action_success_rates[action_name] = {"successes": 0, "attempts": 0, "rate": 0.5}
            
            logger.info(f"Registered handler for action: {action_name}")
    
    async def generate_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an action based on context (wrapper for generate_optimal_action)
        
        Args:
            context: Current context including state, motivations, and goals
            
        Returns:
            Generated action with name, parameters, and metadata
        """
        try:
            # First update system state to ensure action generation uses current data
            await self.update_motivations()
            
            # Process the context if needed
            if not isinstance(context, ActionContext):
                # Gather complete action context from all integrated systems
                action_context = await self._gather_action_context(context)
            else:
                action_context = context
            
            # Generate the optimal action
            action = await self.generate_optimal_action(action_context)
            
            # Make sure action has required fields
            if isinstance(action, dict) and "name" in action:
                # Ensure the action has an ID
                if "id" not in action:
                    action["id"] = f"action_{uuid.uuid4().hex[:8]}"
                    
                # Ensure the action has a timestamp
                if "timestamp" not in action:
                    action["timestamp"] = datetime.datetime.now().isoformat()
                    
                # Add to action history
                if hasattr(self, "action_history"):
                    self.action_history.append(action)
                    if len(self.action_history) > 100:
                        self.action_history = self.action_history[-100:]
                
                return action
            else:
                # If the action doesn't have the expected structure, log an error and return a generic action
                logger.error(f"Generated action doesn't have required fields: {action}")
                return {
                    "name": "fallback_action",
                    "parameters": {},
                    "source": ActionSource.MOTIVATION,
                    "description": "Fallback action due to invalid generation result",
                    "id": f"action_{uuid.uuid4().hex[:8]}",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error generating action: {e}", exc_info=True)
            
            # Return a fallback action in case of errors
            return {
                "name": "fallback_action",
                "parameters": {},
                "source": ActionSource.MOTIVATION,
                "description": f"Fallback action due to error: {str(e)}",
                "id": f"action_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.datetime.now().isoformat()
            }

    def assign_core_activities(self):
        core_activities = [
            "roleplaying",
            "domination",
            "femdom",
            "video_games",
            "streaming",
            "influencing",
            "social_media",
            "socializing"
        ]
        activities = self.identity_evolution.identity_profile["preferences"]["activities"]
        for activity in core_activities:
            if activity not in activities:
                activities[activity] = {
                    "score": 0.0,
                    "confidence": 1.0,
                    "is_hobby": True,
                    "last_done": datetime.datetime.now(),
                    "history": [],
                    "core": True
                }
            else:
                activities[activity]["core"] = True
                activities[activity]["confidence"] = max(activities[activity].get("confidence", 0.0), 1.0)
                activities[activity]["is_hobby"] = True

    async def apply_expression_to_action_generation(expression_system, action_context):
        """
        Modify action generation context based on expression pattern.
        
        Args:
            expression_system: ExpressionSystem instance
            action_context: Action context dictionary
            
        Returns:
            Modified action context with expression influences
        """
        # Get current expression pattern
        pattern = expression_system.current_pattern
        
        # Apply action biases to context
        activity_biases = expression_system.get_action_biases()
        if activity_biases:
            # Modify action probabilities in the context
            if "available_actions" in action_context:
                biased_actions = []
                for action in action_context["available_actions"]:
                    # Check if action type matches any bias
                    for activity_type, bias in activity_biases.items():
                        if activity_type.lower() in action.lower():
                            # Apply bias: add to biased list with higher probability
                            biased_actions.append(action)
                            # Could be added multiple times to increase probability
                            if bias > 1.3:
                                biased_actions.append(action)  # Add again for very high bias
                
                # Combine original and biased actions
                action_context["available_actions"] = action_context["available_actions"] + biased_actions
        
        # Apply initiative level to influence action proactivity
        initiative = getattr(pattern, "initiative_level", 0.5)
        if "motivations" in action_context:
            motivations = action_context["motivations"]
            
            # Adjust autonomy and self-improvement based on initiative
            if "autonomy" in motivations:
                motivations["autonomy"] = max(0.1, min(0.9, motivations["autonomy"] * (0.5 + initiative)))
                
            if "self_improvement" in motivations:
                motivations["self_improvement"] = max(0.1, min(0.9, motivations["self_improvement"] * (0.5 + initiative)))
                
        # Apply engagement level to influence connection-seeking
        engagement = getattr(pattern, "engagement_level", 0.5)
        if "motivations" in action_context:
            motivations = action_context["motivations"]
            
            # Adjust connection and validation based on engagement
            if "connection" in motivations:
                motivations["connection"] = max(0.1, min(0.9, motivations["connection"] * (0.5 + engagement)))
                
            if "validation" in motivations:
                motivations["validation"] = max(0.1, min(0.9, motivations["validation"] * (0.5 + engagement)))
        
        return action_context

    async def preprocess_message(expression_system, message):
        """
        Apply expression patterns to an outgoing message before sending.
        
        Args:
            expression_system: ExpressionSystem instance
            message: Message to send
            
        Returns:
            Modified message with expression patterns applied
        """
        # Update expression pattern
        await expression_system.update_expression_pattern()
        
        # Apply text modifications
        modified_message = await expression_system.apply_text_expression(message)
        
        # Get behavioral expressions that could be included
        behaviors = expression_system.get_behavioral_expressions()
        
        # For especially strong behaviors, add them to the message
        behavior_descriptions = []
        
        # Add gesture descriptions for strong gestures
        if "gestures" in behaviors:
            for gesture, strength in behaviors["gestures"].items():
                if strength > 0.7 and random.random() < 0.3:
                    # Format gestures as actions in brackets
                    gesture_desc = f"*{gesture.replace('_', ' ')}*"
                    behavior_descriptions.append(gesture_desc)
        
        # Add posture descriptions for strong postures
        if "posture" in behaviors:
            for posture, strength in behaviors["posture"].items():
                if strength > 0.8 and random.random() < 0.2:
                    posture_desc = f"*{posture.replace('_', ' ')} posture*"
                    behavior_descriptions.append(posture_desc)
        
        # Combine message with behaviors
        if behavior_descriptions and random.random() < 0.4:  # Only sometimes add behaviors
            # Choose one behavior to add
            behavior = random.choice(behavior_descriptions)
            
            # Add behavior before or after message
            if random.random() < 0.5:
                final_message = f"{behavior} {modified_message}"
            else:
                final_message = f"{modified_message} {behavior}"
        else:
            final_message = modified_message
        
        return final_message


    async def periodic_hobby_meta_loop(self, interval=3600):
        """
        Periodically update and consolidate hobby/preference/activities emergence/decay.
        Run this once as a background task (never returns).
        """
        while True:
            try:
                self.assign_core_activities()
                # -- You can tune these thresholds or expose as parameters --
                min_to_hobby = 3.0
                min_proficiency = 0.7
                min_confidence = 0.6
                inactive_decay_days = 14
                hobby_forget_thresh = 0.3
                
                activities = self.identity_evolution.identity_profile["preferences"]["activities"]
                now = datetime.datetime.now()
                any_new_hobby = False
                any_forgotten = False
    
                # Optional: get procedural memory/proficiency if you've linked it
                procedural_manager = getattr(self, "procedural_memory_manager", None)
                async with self._activities_lock:
                    for proc_name, p in activities.items():
                        if p.get("core"):
                            p["is_hobby"] = True
                            p["confidence"] = max(p.get("confidence", 0.0), 1.0)
                            continue
                        proficiency = min_proficiency
                        last_done = p.get("last_done")
                        # If using procedural manager, get proficiency
                        if procedural_manager and proc_name in procedural_manager.procedures:
                            proficiency = getattr(procedural_manager.procedures[proc_name], "proficiency", min_proficiency)
                        else:
                            proficiency = min_proficiency
        
                        # Promote to hobby if criteria met
                        if (p["score"] > min_to_hobby 
                            and proficiency >= min_proficiency 
                            and p["confidence"] >= min_confidence 
                            and not p.get("is_hobby", False)
                        ):
                            p["is_hobby"] = True
                            any_new_hobby = True
                            # Add self-reflect memory/statement if you wish
                            if self.experience_interface:
                                await self.experience_interface.memory_core.add_memory(
                                    memory_text=f"I've realized that '{proc_name}' is a hobby of mine. I enjoy it, and I've become proficient.",
                                    memory_type="reflection",
                                    significance=8,
                                    tags=["hobby", proc_name, "identity", "living_emergence"],
                                    metadata={"hobby": True, "activity": proc_name, "promotion_time": now.isoformat()}
                                )
                        # Demote if decayed
                        if p.get("is_hobby", False) and last_done:
                            if (now - last_done).days > inactive_decay_days:
                                p["confidence"] *= 0.98 ** ((now - last_done).days)
                                if p["confidence"] < hobby_forget_thresh:
                                    p["is_hobby"] = False
                                    any_forgotten = True
                                    if self.experience_interface:
                                        await self.experience_interface.memory_core.add_memory(
                                            memory_text=f"I've lost interest in '{proc_name}'. It doesn't feel like a real hobby any more.",
                                            memory_type="reflection",
                                            significance=5,
                                            tags=["hobby_forgotten", proc_name, "identity", "living_emergence"],
                                            metadata={"hobby": False, "activity": proc_name, "demotion_time": now.isoformat()}
                                        )
        
                    # If nothing is a hobby, suggest exploration
                    if all(not p.get("is_hobby", False) for p in activities.values()):
                        if self.goal_system:
                            await self.goal_system.add_goal(
                                description="Explore a new activity or hobby I haven't tried before",
                                priority=0.7
                            )
        
                    # Sleep until next run
                    await asyncio.sleep(interval)
            except Exception as exc:
                logger.error(f"Error in periodic_hobby_meta_loop: {exc}", exc_info=True)
                await asyncio.sleep(interval)

    async def create_story(self, title: str, prompt: str = "", metadata: Optional[Dict[str,Any]] = None):
        """Tool: write & store a story, tagging it with Nyx’s current mood."""
        # you can pull in mood or recent thought if you like:
        mood = (await self.mood_manager.get_current_mood()).dict() if self.mood_manager else {}
        metadata = metadata or {}
        metadata.update({"mood": mood})
        return await self.creative_system.store_content("story", title, prompt, metadata)

    async def create_poem(self, title: str, prompt: str = "", metadata: Optional[Dict[str,Any]] = None):
        metadata = metadata or {}
        metadata.update({"mood": (await self.mood_manager.get_current_mood()).dict()})
        return await self.creative_system.store_content("poem", title, prompt, metadata)

    async def create_lyrics(self, title: str, prompt: str = "", metadata: Optional[Dict[str,Any]] = None):
        metadata = metadata or {}
        return await self.creative_system.store_content("lyrics", title, prompt, metadata)

    async def create_journal(self, title: str, prompt: str = "", metadata: Optional[Dict[str,Any]] = None):
        metadata = metadata or {}
        return await self.creative_system.store_content("journal", title, prompt, metadata)

    async def list_creations(self, content_type: Optional[str] = None,
                             limit: int = 20, offset: int = 0):
        return await self.creative_system.list_content(content_type, limit, offset)
                                 

    async def retrieve_content(self, content_id: str):
        return await self.creative_system.retrieve_content(content_id)

    async def assess_capabilities(self):
        """Run the CapabilityAssessmentSystem and return its report."""
        return await self.capability_assessor.assess_all()

AgenticActionGenerator = EnhancedAgenticActionGenerator
