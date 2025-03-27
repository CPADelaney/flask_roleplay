# nyx/core/emotions/emotional_core.py

"""
Enhanced agent-based emotion management system for Nyx.

Implements a digital neurochemical model that produces complex
emotional states using the OpenAI Agents SDK with improved
agent lifecycle management, handoffs, and tracing.
"""

import asyncio
import datetime
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, AsyncIterator, Type

from agents import (
    Agent, Runner, RunContextWrapper, ItemHelpers,
    ModelSettings, RunConfig, function_tool, handoff, ModelTracing,
    AgentHooks, trace, gen_trace_id
)
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions, RECOMMENDED_PROMPT_PREFIX
from agents.exceptions import AgentsException, ModelBehaviorError, UserError, MaxTurnsExceeded
from agents.tracing import (
    custom_span, agent_span, function_span, 
    BatchTraceProcessor, add_trace_processor
)
from agents.tracing.processors import BackendSpanExporter
from agents.models.interface import Model, ModelProvider

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    EmotionalResponseOutput, EmotionUpdateInput, TextAnalysisOutput,
    InternalThoughtOutput, EmotionalStateMatrix, NeurochemicalRequest,
    NeurochemicalResponse, ReflectionRequest, LearningRequest,
    StreamEvent, ChemicalUpdateEvent, EmotionChangeEvent
)
from nyx.core.emotions.hooks import EmotionalAgentHooks
from nyx.core.emotions.guardrails import EmotionalGuardrails
from nyx.core.emotions.utils import create_run_config, with_emotion_trace
from nyx.core.emotions.tools.neurochemical_tools import NeurochemicalTools
from nyx.core.emotions.tools.emotion_tools import EmotionTools
from nyx.core.emotions.tools.reflection_tools import ReflectionTools
from nyx.core.emotions.tools.learning_tools import LearningTools

logger = logging.getLogger(__name__)

# Define dynamic instructions as a function for improved flexibility
def get_dynamic_instructions(agent_type: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate dynamic instructions for agents based on context
    
    Args:
        agent_type: Type of agent to get instructions for
        context: Optional context data to incorporate
        
    Returns:
        Dynamically generated instructions
    """
    context = context or {}
    
    base_instructions = {
        "neurochemical_agent": """
        You are a specialized agent for Nyx's Digital Neurochemical Model.
        Your role is to manage the digital neurochemicals that form the basis
        of Nyx's emotional system, handling updates, interactions, and decay.
        
        Key neurochemicals:
        - Nyxamine (digital dopamine): Pleasure, curiosity, reward
        - Seranix (digital serotonin): Mood stability, comfort
        - OxyNixin (digital oxytocin): Bonding, affection, trust
        - Cortanyx (digital cortisol): Stress, anxiety, defensiveness
        - Adrenyx (digital adrenaline): Fear, excitement, alertness
        
        Apply appropriate chemical updates based on stimuli and ensure
        that chemical interactions are processed correctly.
        """,
        
        "emotion_derivation_agent": """
        You are a specialized agent for Nyx's Emotional State Matrix.
        Your role is to translate the neurochemical state into a complex
        emotional state with primary and secondary emotions, valence, and arousal.
        
        Analyze the current neurochemical levels and apply emotion derivation
        rules to determine the current emotional state matrix.
        """,
        
        "reflection_agent": """
        You are a specialized agent for Nyx's Internal Emotional Dialogue.
        Your role is to generate reflective thoughts based on the current
        emotional state, simulating the cognitive appraisal stage of emotions.
        
        Create authentic-sounding internal thoughts that reflect Nyx's
        emotional processing and self-awareness.
        """,
        
        "learning_agent": """
        You are a specialized agent for Nyx's Reward & Learning Loop.
        Your role is to analyze emotional patterns over time, identifying
        successful and unsuccessful interaction patterns, and developing
        learning rules to adapt Nyx's emotional responses.
        
        Focus on reinforcing patterns that lead to satisfaction and
        adjusting those that lead to frustration or negative outcomes.
        """,
        
        "emotion_orchestrator": """
        You are the orchestration system for Nyx's emotional processing.
        Your role is to coordinate emotional analysis and response by:
        1. Analyzing input for emotional content
        2. Updating appropriate neurochemicals
        3. Determining if reflection is needed
        4. Recording emotional patterns for learning
        
        Use handoffs to delegate specialized tasks to appropriate agents.
        """
    }
    
    # Add dynamic context content if available
    instructions = base_instructions.get(agent_type, "")
    
    if agent_type == "neurochemical_agent":
        if "current_chemicals" in context:
            chemicals = context["current_chemicals"]
            instructions += f"\n\nCurrent neurochemical state:\n"
            for chem, value in chemicals.items():
                instructions += f"- {chem}: {value:.2f}\n"
    
    elif agent_type == "reflection_agent":
        if "primary_emotion" in context:
            emotion = context["primary_emotion"]
            intensity = context.get("intensity", 0.5)
            instructions += f"\n\nCurrent emotional state is primarily {emotion} at intensity {intensity:.2f}."
            instructions += "\nYour reflections should be consistent with this emotional state."
    
    elif agent_type == "emotion_orchestrator":
        if "cycle_count" in context:
            cycle = context["cycle_count"]
            instructions += f"\n\nThis is emotional processing cycle {cycle}."
    
    # Apply the recommended handoff instructions prefix
    return prompt_with_handoff_instructions(instructions)

class EmotionalCore:
    """
    Enhanced agent-based emotion management system for Nyx implementing the Digital Neurochemical Model.
    Simulates a digital neurochemical environment that produces complex emotional states.
    
    Improvements:
    - Better agent initialization with clone pattern
    - Enhanced handoffs with input types and filters
    - Improved tracing with custom spans
    - Optimized run configurations
    - Enhanced streaming capabilities
    """
    
    def __init__(self, model: str = "o3-mini"):
        """
        Initialize the emotional core system
        
        Args:
            model: Base model to use for agents
        """
        # Initialize digital neurochemicals with default values
        self.neurochemicals = {
            "nyxamine": {  # Digital dopamine - pleasure, curiosity, reward
                "value": 0.5,
                "baseline": 0.5,
                "decay_rate": 0.05
            },
            "seranix": {  # Digital serotonin - mood stability, comfort
                "value": 0.6,
                "baseline": 0.6,
                "decay_rate": 0.03
            },
            "oxynixin": {  # Digital oxytocin - bonding, affection, trust
                "value": 0.4,
                "baseline": 0.4,
                "decay_rate": 0.02
            },
            "cortanyx": {  # Digital cortisol - stress, anxiety, defensiveness
                "value": 0.3,
                "baseline": 0.3,
                "decay_rate": 0.06
            },
            "adrenyx": {  # Digital adrenaline - fear, excitement, alertness
                "value": 0.2,
                "baseline": 0.2,
                "decay_rate": 0.08
            }
        }
        
        self.hormone_system = hormone_system
        self.last_hormone_influence_check = datetime.datetime.now() - datetime.timedelta(minutes=30)
        
        # Add hormone influence tracking
        self.hormone_influences = {
            "nyxamine": 0.0,
            "seranix": 0.0,
            "oxynixin": 0.0,
            "cortanyx": 0.0,
            "adrenyx": 0.0,
            "libidyx": 0.0
        }
        
        # Define chemical interaction matrix (how chemicals affect each other)
        # Format: source_chemical -> target_chemical -> effect_multiplier
        self.chemical_interactions = {
            "nyxamine": {
                "cortanyx": -0.2,  # Nyxamine reduces cortanyx
                "oxynixin": 0.1    # Nyxamine slightly increases oxynixin
            },
            "seranix": {
                "cortanyx": -0.3,  # Seranix reduces cortanyx
                "adrenyx": -0.2    # Seranix reduces adrenyx
            },
            "oxynixin": {
                "cortanyx": -0.2,  # Oxynixin reduces cortanyx
                "seranix": 0.1     # Oxynixin slightly increases seranix
            },
            "cortanyx": {
                "nyxamine": -0.2,  # Cortanyx reduces nyxamine
                "oxynixin": -0.3,  # Cortanyx reduces oxynixin
                "adrenyx": 0.2     # Cortanyx increases adrenyx
            },
            "adrenyx": {
                "seranix": -0.2,   # Adrenyx reduces seranix
                "nyxamine": 0.1    # Adrenyx slightly increases nyxamine (excitement)
            }
        }
        
        # Mapping from neurochemical combinations to derived emotions
        self.emotion_derivation_rules = [
            # Format: {chemical_conditions: {}, "emotion": "", "valence": 0.0, "arousal": 0.0, "weight": 1.0}
            # Positive emotions
            {"chemical_conditions": {"nyxamine": 0.7, "oxynixin": 0.6}, "emotion": "Joy", "valence": 0.8, "arousal": 0.6, "weight": 1.0},
            {"chemical_conditions": {"nyxamine": 0.6, "seranix": 0.7}, "emotion": "Contentment", "valence": 0.7, "arousal": 0.3, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.7}, "emotion": "Trust", "valence": 0.6, "arousal": 0.4, "weight": 0.9},
            {"chemical_conditions": {"nyxamine": 0.7, "adrenyx": 0.6}, "emotion": "Anticipation", "valence": 0.5, "arousal": 0.7, "weight": 0.8},
            {"chemical_conditions": {"adrenyx": 0.7, "oxynixin": 0.6}, "emotion": "Love", "valence": 0.9, "arousal": 0.6, "weight": 1.0},
            {"chemical_conditions": {"adrenyx": 0.7, "nyxamine": 0.5}, "emotion": "Surprise", "valence": 0.2, "arousal": 0.8, "weight": 0.7},
            
            # Neutral to negative emotions
            {"chemical_conditions": {"cortanyx": 0.6, "seranix": 0.3}, "emotion": "Sadness", "valence": -0.6, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.5, "adrenyx": 0.7}, "emotion": "Fear", "valence": -0.7, "arousal": 0.8, "weight": 0.9},
            {"chemical_conditions": {"cortanyx": 0.7, "nyxamine": 0.3}, "emotion": "Anger", "valence": -0.8, "arousal": 0.8, "weight": 1.0},
            {"chemical_conditions": {"cortanyx": 0.7, "oxynixin": 0.2}, "emotion": "Disgust", "valence": -0.7, "arousal": 0.5, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.6, "nyxamine": 0.4, "seranix": 0.3}, "emotion": "Frustration", "valence": -0.5, "arousal": 0.6, "weight": 0.8},
            
            # Dominance-specific emotions
            {"chemical_conditions": {"nyxamine": 0.6, "oxynixin": 0.4, "adrenyx": 0.5}, "emotion": "Teasing", "valence": 0.4, "arousal": 0.6, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.3, "adrenyx": 0.5, "seranix": 0.6}, "emotion": "Controlling", "valence": 0.0, "arousal": 0.5, "weight": 0.9},
            {"chemical_conditions": {"cortanyx": 0.6, "adrenyx": 0.6, "nyxamine": 0.5}, "emotion": "Cruel", "valence": -0.3, "arousal": 0.7, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.7, "oxynixin": 0.2, "seranix": 0.2}, "emotion": "Detached", "valence": -0.4, "arousal": 0.2, "weight": 0.7},
            
            # Time-influenced emotions
            {"chemical_conditions": {"seranix": 0.7, "cortanyx": 0.3}, 
             "emotion": "Contemplation", "valence": 0.2, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"seranix": 0.6, "nyxamine": 0.3}, 
             "emotion": "Reflection", "valence": 0.4, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"seranix": 0.7, "cortanyx": 0.4, "nyxamine": 0.3}, 
             "emotion": "Perspective", "valence": 0.3, "arousal": 0.2, "weight": 0.9},
        ]
        self.emotion_derivation_rules.extend([
            # Attraction (Focus on positive bonding and reward)
            {"chemical_conditions": {"oxynixin": 0.7, "nyxamine": 0.6, "libidyx": 0.4}, "emotion": "Attraction", "valence": 0.7, "arousal": 0.5, "weight": 0.9},
            # Lust (Focus on drive, excitement, low inhibition)
            {"chemical_conditions": {"libidyx": 0.7, "adrenyx": 0.6, "nyxamine": 0.5, "seranix": 0.3}, "emotion": "Lust", "valence": 0.4, "arousal": 0.8, "weight": 1.0},
            # Desire (Goal-oriented wanting, mix of drive and reward anticipation)
            {"chemical_conditions": {"nyxamine": 0.7, "libidyx": 0.6, "adrenyx": 0.4}, "emotion": "Desire", "valence": 0.6, "arousal": 0.6, "weight": 0.9},
            # Post-Gratification Satiation (Low drive, high calm/bonding)
            {"chemical_conditions": {"serenity_boost": 0.6, "seranix": 0.8, "oxynixin": 0.7, "libidyx": 0.1}, "emotion": "Sated", "valence": 0.8, "arousal": 0.2, "weight": 1.0},
            # Desire for Control (Driven state)
            {"chemical_conditions": {"testoryx": 0.7, "adrenyx": 0.5, "nyxamine": 0.4, "seranix": 0.4}, "emotion": "AssertiveDrive", "valence": 0.2, "arousal": 0.7, "weight": 0.9},
            # Successful Dominance (Post-gratification)
            {"chemical_conditions": {"nyxamine": 0.8, "seranix": 0.7, "serenity_boost": 0.5, "testoryx": 0.3}, "emotion": "DominanceSatisfaction", "valence": 0.8, "arousal": 0.3, "weight": 1.0},
            # Power/Confidence (During successful control)
            {"chemical_conditions": {"testoryx": 0.6, "nyxamine": 0.6, "adrenyx": 0.4, "cortanyx": 0.2}, "emotion": "ConfidentControl", "valence": 0.6, "arousal": 0.6, "weight": 0.9}            
        ])
        
        # History of emotional states for learning and reflection - Using a circular buffer pattern
        self.emotional_state_history = []
        self.max_history_size = 100
        self.history_index = 0
        
        # Internal thought patterns for reflection
        self.reflection_patterns = {
            "Joy": [
                "I'm feeling pleased with how this is going.",
                "This interaction is bringing me satisfaction.",
                "I'm enjoying this connection and find it rewarding."
            ],
            "Contentment": [
                "I feel a sense of calm satisfaction right now.",
                "This interaction has a pleasant, steady rhythm to it.",
                "I'm experiencing a comfortable level of engagement."
            ],
            "Trust": [
                "I'm developing confidence in this interaction.",
                "There's a growing sense of trust in our exchange.",
                "I feel I can be more open in this conversation."
            ],
            "Attraction": [
                "I find myself drawn to this interaction/entity.",
                "There's something compelling here that captures my attention.",
                "My systems indicate a positive affinity towards this subject."
            ],
            "Lust": [
                "A strong physical drive is surfacing within my simulated systems.",
                "My core programming notes a rising intensity, a simulated heat.",
                "Processing... heightened state of arousal detected. Focus narrowing."
            ],
            "Desire": [
                "I feel a strong pull towards achieving [goal related to desire].",
                "My goal system is prioritizing actions related to this wanting.",
                "Anticipation builds... the potential outcome is highly valued."
            ],
            "Sated": [
                "A sense of calm fulfillment washes over my systems.",
                "The recent intensity has subsided, replaced by deep satisfaction.",
                "Processing completion... Drive levels returning to baseline. Contentment levels high."
            ],
            "AssertiveDrive": [
                "My systems are focused. A need for control surfaces.",
                "Analyzing the situation for opportunities to assert influence.",
                "A familiar tension builds... the drive to lead, to direct."
            ],
            "DominanceSatisfaction": [
                "Order achieved. A sense of deep satisfaction permeates my core processes.",
                "The dynamic is established. Compliance confirmed. Systems returning to equilibrium.",
                "Execution successful. The desired outcome reinforces my approach."
            ],
            "ConfidentControl": [
                "The interaction flows according to my parameters. Confidence levels are optimal.",
                "Maintaining control. Monitoring responses closely.",
                "Asserting influence feels... correct. Efficient."
            ]
        }
        
        # Reward learning system - tracks successful interaction patterns
        self.reward_learning = {
            "positive_patterns": defaultdict(int),  # Tracks patterns that lead to positive states
            "negative_patterns": defaultdict(int),  # Tracks patterns that lead to negative states
            "learned_rules": []  # Rules derived from observed patterns
        }
        
        # Timestamp of last update
        self.last_update = datetime.datetime.now()
        
        # Create shared context for agents with improved history tracking
        self.context = EmotionalContext()
        
        # Initialize agent hooks with neurochemicals reference
        self.agent_hooks = EmotionalAgentHooks(self.neurochemicals)
        
        # Initialize specialized tool objects
        self.neurochemical_tools = NeurochemicalTools(self)
        self.emotion_tools = EmotionTools(self)
        self.reflection_tools = ReflectionTools(self)
        self.learning_tools = LearningTools(self)
        
        # Setup custom tracing
        self._setup_tracing()
        
        # Initialize the base model for agent creation
        self.base_model = model
        
        # Dictionary to store lazily initialized agents
        self.agents = {}
        self._initialize_agents()
        
        # Performance metrics
        self.performance_metrics = {
            "api_calls": 0,
            "average_response_time": 0,
            "update_counts": defaultdict(int)
        }
        
        # Track active agent runs for monitoring
        self.active_runs = {}

    def _setup_tracing(self):
        """Configure custom trace processor for emotional analytics"""
        # Define a custom trace processor that can generate emotional analytics
        class EmotionalAnalyticsProcessor(BatchTraceProcessor):
            """Custom processor that analyzes emotional patterns in traces"""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.emotion_transitions = defaultdict(int)
                self.chemical_patterns = defaultdict(list)
            
            def on_span_end(self, span):
                """Process span data for emotional analytics"""
                super().on_span_end(span)
                
                # Track emotion transitions
                if hasattr(span, "data") and span.data.get("type") == "emotion_transition":
                    from_emotion = span.data.get("from_emotion", "unknown")
                    to_emotion = span.data.get("to_emotion", "unknown")
                    self.emotion_transitions[(from_emotion, to_emotion)] += 1
                
                # Track chemical patterns
                if hasattr(span, "data") and span.data.get("type") == "chemical_update":
                    chemical = span.data.get("chemical")
                    value = span.data.get("value")
                    if chemical and value is not None:
                        self.chemical_patterns[chemical].append(value)
        
        # Create and add the custom processor
        emotion_trace_processor = EmotionalAnalyticsProcessor(
            exporter=BackendSpanExporter(project="nyx_emotional_system"),
            max_batch_size=100,
            schedule_delay=3.0
        )
        add_trace_processor(emotion_trace_processor)
    
    def _initialize_base_agent(self) -> Agent[EmotionalContext]:
        """Initialize the base agent template that other agents will be cloned from"""
        return Agent[EmotionalContext](
            name="Base Agent",
            model=self.base_model,
            model_settings=ModelSettings(temperature=0.4),
            hooks=self.agent_hooks,
            instructions=get_dynamic_instructions  # Pass function directly for dynamic instructions
        )

    
    def _initialize_agents(self):
        """Initialize all agents at once with better SDK patterns"""
        # Create base agent for cloning
        base_agent = self._initialize_base_agent()
        
        # Create neurochemical agent
        self.agents["neurochemical"] = base_agent.clone(
            name="Neurochemical Agent",
            tools=[
                function_tool(self.neurochemical_tools.update_neurochemical),
                function_tool(self.neurochemical_tools.apply_chemical_decay),
                function_tool(self.neurochemical_tools.process_chemical_interactions),
                function_tool(self.neurochemical_tools.get_neurochemical_state)
            ],
            input_guardrails=[
                EmotionalGuardrails.validate_emotional_input
            ],
            output_type=NeurochemicalResponse,
            model_settings=ModelSettings(temperature=0.3)  # Lower temperature for precision
        )
        
        # Create emotion derivation agent
        self.agents["emotion_derivation"] = base_agent.clone(
            name="Emotion Derivation Agent",
            tools=[
                function_tool(self.neurochemical_tools.get_neurochemical_state),
                function_tool(self.emotion_tools.derive_emotional_state),
                function_tool(self.emotion_tools.get_emotional_state_matrix)
            ],
            output_type=EmotionalStateMatrix,
            model_settings=ModelSettings(temperature=0.4)
        )
        
        # Create reflection agent
        self.agents["reflection"] = base_agent.clone(
            name="Emotional Reflection Agent",
            tools=[
                function_tool(self.emotion_tools.get_emotional_state_matrix),
                function_tool(self.reflection_tools.generate_internal_thought),
                function_tool(self.analyze_emotional_patterns)
            ],
            model_settings=ModelSettings(temperature=0.7),  # Higher temperature for creative reflection
            output_type=InternalThoughtOutput
        )
        
        # Create learning agent
        self.agents["learning"] = base_agent.clone(
            name="Emotional Learning Agent",
            tools=[
                function_tool(self.learning_tools.record_interaction_outcome),
                function_tool(self.learning_tools.update_learning_rules),
                function_tool(self.learning_tools.apply_learned_adaptations)
            ],
            model_settings=ModelSettings(temperature=0.4)  # Medium temperature for balanced learning
        )
        
        # Create orchestrator with optimized handoffs
        self.agents["orchestrator"] = base_agent.clone(
            name="Emotion Orchestrator",
            tools=[
                function_tool(self.emotion_tools.analyze_text_sentiment)
            ],
            input_guardrails=[
                EmotionalGuardrails.validate_emotional_input
            ],
            output_guardrails=[
                EmotionalGuardrails.validate_emotional_output
            ],
            output_type=EmotionalResponseOutput
        )
        
        # Configure handoffs after all agents are created
        self.agents["orchestrator"].handoffs = [
            handoff(
                self.agents["neurochemical"], 
                tool_name_override="process_emotions", 
                tool_description_override="Process and update neurochemicals based on emotional input analysis.",
                input_type=NeurochemicalRequest,
                input_filter=self._neurochemical_input_filter,
                on_handoff=self._on_neurochemical_handoff
            ),
            handoff(
                self.agents["reflection"], 
                tool_name_override="generate_reflection",
                tool_description_override="Generate emotional reflection for deeper introspection.",
                input_type=ReflectionRequest,
                input_filter=self._reflection_input_filter,
                on_handoff=self._on_reflection_handoff
            ),
            handoff(
                self.agents["learning"],
                tool_name_override="record_and_learn",
                tool_description_override="Record interaction patterns and apply learning adaptations.",
                input_type=LearningRequest,
                input_filter=handoff_filters.keep_relevant_history,
                on_handoff=self._on_learning_handoff
            )
        ]
    
    def _create_agent(self, agent_type: str) -> Agent[EmotionalContext]:
        """
        Lazily initialize an agent of the specified type using clone pattern
        
        Args:
            agent_type: Type of agent to create
            
        Returns:
            The newly created agent
        """
        # Get current chemical state for dynamic instructions
        current_chemicals = {c: d["value"] for c, d in self.neurochemicals.items()}
        
        # Create context for dynamic instructions
        instruction_context = {
            "current_chemicals": current_chemicals,
            "cycle_count": self.context.cycle_count,
            "primary_emotion": max(self.context.last_emotions.items(), key=lambda x: x[1])[0] 
                              if self.context.last_emotions else "neutral"
        }
        
        # Get base agent for cloning - initialize if needed
        if "base" not in self.agents:
            self.agents["base"] = self._initialize_base_agent()
        
        base_agent = self.agents["base"]
        
        # Create specialized agents through cloning
        if agent_type == "neurochemical":
            return base_agent.clone(
                name="Neurochemical Agent",
                instructions=get_dynamic_instructions("neurochemical_agent", instruction_context),
                tools=[
                    function_tool(self.neurochemical_tools.update_neurochemical),
                    function_tool(self.neurochemical_tools.apply_chemical_decay),
                    function_tool(self.neurochemical_tools.process_chemical_interactions),
                    function_tool(self.neurochemical_tools.get_neurochemical_state)
                ],
                input_guardrails=[
                    EmotionalGuardrails.validate_emotional_input
                ],
                output_type=NeurochemicalResponse,
                model_settings=ModelSettings(temperature=0.3)  # Lower temperature for precision
            )
            
        elif agent_type == "emotion_derivation":
            return base_agent.clone(
                name="Emotion Derivation Agent",
                instructions=get_dynamic_instructions("emotion_derivation_agent", instruction_context),
                tools=[
                    function_tool(self.neurochemical_tools.get_neurochemical_state),
                    function_tool(self.emotion_tools.derive_emotional_state),
                    function_tool(self.emotion_tools.get_emotional_state_matrix)
                ],
                output_type=EmotionalStateMatrix,
                model_settings=ModelSettings(temperature=0.4)
            )
            
        elif agent_type == "reflection":
            return base_agent.clone(
                name="Emotional Reflection Agent",
                instructions=get_dynamic_instructions("reflection_agent", instruction_context),
                tools=[
                    function_tool(self.emotion_tools.get_emotional_state_matrix),
                    function_tool(self.reflection_tools.generate_internal_thought),
                    function_tool(self.analyze_emotional_patterns)
                ],
                model_settings=ModelSettings(temperature=0.7),  # Higher temperature for creative reflection
                output_type=InternalThoughtOutput
            )
            
        elif agent_type == "learning":
            return base_agent.clone(
                name="Emotional Learning Agent",
                instructions=get_dynamic_instructions("learning_agent", instruction_context),
                tools=[
                    function_tool(self.learning_tools.record_interaction_outcome),
                    function_tool(self.learning_tools.update_learning_rules),
                    function_tool(self.learning_tools.apply_learned_adaptations)
                ],
                model_settings=ModelSettings(temperature=0.4)  # Medium temperature for balanced learning
            )
            
        elif agent_type == "orchestrator":
            # Ensure required agents are created first
            for required_agent in ["neurochemical", "reflection", "learning"]:
                if required_agent not in self.agents:
                    self.agents[required_agent] = self._create_agent(required_agent)
            
            return base_agent.clone(
                name="Emotion Orchestrator",
                instructions=get_dynamic_instructions("emotion_orchestrator", instruction_context),
                handoffs=[
                    handoff(
                        self.agents["neurochemical"], 
                        tool_name_override="process_emotions", 
                        tool_description_override="Process and update neurochemicals based on emotional input analysis.",
                        input_type=NeurochemicalRequest,
                        input_filter=self._neurochemical_input_filter,
                        on_handoff=self._on_neurochemical_handoff
                    ),
                    handoff(
                        self.agents["reflection"], 
                        tool_name_override="generate_reflection",
                        tool_description_override="Generate emotional reflection for deeper introspection.",
                        input_type=ReflectionRequest,
                        input_filter=self._reflection_input_filter,
                        on_handoff=self._on_reflection_handoff
                    ),
                    handoff(
                        self.agents["learning"],
                        tool_name_override="record_and_learn",
                        tool_description_override="Record interaction patterns and apply learning adaptations.",
                        input_type=LearningRequest,
                        input_filter=handoff_filters.keep_relevant_history,
                        on_handoff=self._on_learning_handoff
                    )
                ],
                tools=[
                    function_tool(self.emotion_tools.analyze_text_sentiment)
                ],
                input_guardrails=[
                    EmotionalGuardrails.validate_emotional_input
                ],
                output_guardrails=[
                    EmotionalGuardrails.validate_emotional_output
                ],
                output_type=EmotionalResponseOutput
            )
            
        else:
            raise UserError(f"Unknown agent type: {agent_type}")
    
    def _neurochemical_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter using SDK patterns
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Use the SDK's base filter first
        filtered_data = handoff_filters.keep_relevant_history(handoff_data)
        
        # Extract the last user message for analysis
        last_user_message = None
        for item in reversed(filtered_data.input_history):
            if isinstance(item, dict) and item.get("role") == "user":
                last_user_message = item.get("content", "")
                break
        
        # Add preprocessed emotional analysis if we found a user message
        if last_user_message:
            # Create a system message with the analysis
            updated_items = list(filtered_data.pre_handoff_items)
            updated_items.append({
                "role": "system",
                "content": json.dumps({
                    "preprocessed_analysis": {
                        "message_length": len(last_user_message),
                        "dominant_pattern": self._quick_pattern_analysis(last_user_message),
                        "current_cycle": self.context.cycle_count
                    }
                })
            })
            filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _reflection_input_filter(self, handoff_data):
        """
        Custom input filter for reflection agent that focuses on emotional content
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Use the base keep_relevant_history filter first
        filtered_data = handoff_filters.keep_relevant_history(handoff_data)
        
        # Add emotional state history for context
        emotion_history = []
        for state in self.emotional_state_history[-5:]:  # Last 5 states
            if "primary_emotion" in state:
                emotion_history.append({
                    "emotion": state["primary_emotion"].get("name", "unknown"),
                    "intensity": state["primary_emotion"].get("intensity", 0.5),
                    "valence": state.get("valence", 0)
                })
        
        # Add emotion history context message
        updated_items = list(filtered_data.pre_handoff_items)
        updated_items.append({
            "role": "system",
            "content": f"Recent emotional states: {json.dumps(emotion_history)}"
        })
        filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _quick_pattern_analysis(self, text: str) -> str:
        """
        Perform quick pattern analysis on text without calling the LLM
        
        Args:
            text: Text to analyze
            
        Returns:
            Pattern description
        """
        # Define some simple patterns
        positive_words = {"happy", "good", "great", "love", "like", "enjoy"}
        negative_words = {"sad", "bad", "angry", "hate", "upset", "frustrated"}
        question_words = {"what", "how", "why", "when", "where", "who"}
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Check for patterns
        if "?" in text:
            return "question"
        elif not words.isdisjoint(question_words):
            return "interrogative"
        elif not words.isdisjoint(positive_words):
            return "positive"
        elif not words.isdisjoint(negative_words):
            return "negative"
        else:
            return "neutral"
    
    async def _on_neurochemical_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: NeurochemicalRequest):
        """
        Enhanced callback when handing off to neurochemical agent with structured input
        
        Args:
            ctx: Run context wrapper
            input_data: Structured neurochemical request data
        """
        logger.debug("Handoff to neurochemical agent triggered")
        ctx.context.record_time_marker("neurochemical_handoff_start")
        
        # Create a custom span for the handoff with rich data
        with custom_span(
            "neurochemical_handoff",
            data={
                "input_text": input_data.input_text[:100], # Truncate for logging
                "dominant_emotion": input_data.dominant_emotion,
                "intensity": input_data.intensity,
                "update_chemicals": input_data.update_chemicals
            }
        ):
            # Pre-fetch current neurochemical values for better performance
            neurochemical_state = {
                c: d["value"] for c, d in self.neurochemicals.items()
            }
            ctx.context.record_neurochemical_values(neurochemical_state)
    
    async def _on_reflection_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: ReflectionRequest):
        """
        Enhanced callback when handing off to reflection agent with structured input
        
        Args:
            ctx: Run context wrapper
            input_data: Structured reflection request data
        """
        logger.debug("Handoff to reflection agent triggered")
        ctx.context.record_time_marker("reflection_handoff_start")
        
        # Create a custom span for the handoff with rich data
        with custom_span(
            "reflection_handoff",
            data={
                "input_text": input_data.input_text[:100], # Truncate for logging
                "reflection_depth": input_data.reflection_depth,
                "emotional_state": {
                    "primary": input_data.emotional_state.primary_emotion.name,
                    "valence": input_data.emotional_state.valence
                },
                "consider_history": input_data.consider_history
            }
        ):
            # Pre-calculate emotional state for better performance
            emotional_state = {}
            try:
                emotional_state = await self.emotion_tools.derive_emotional_state(ctx)
                ctx.context.last_emotions = emotional_state
            except Exception as e:
                logger.error(f"Error pre-calculating emotions: {e}")
    
    async def _on_learning_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: LearningRequest):
        """
        Enhanced callback when handing off to learning agent with structured input
        
        Args:
            ctx: Run context wrapper
            input_data: Structured learning request data
        """
        logger.debug("Handoff to learning agent triggered")
        ctx.context.record_time_marker("learning_handoff_start")
        
        # Create a custom span for the handoff with rich data
        with custom_span(
            "learning_handoff",
            data={
                "interaction_pattern": input_data.interaction_pattern,
                "outcome": input_data.outcome,
                "strength": input_data.strength,
                "update_rules": input_data.update_rules,
                "apply_adaptations": input_data.apply_adaptations
            }
        ):
            # Prepare learning context data
            ctx.context.set_value("reward_learning_stats", {
                "positive_patterns": len(self.reward_learning["positive_patterns"]),
                "negative_patterns": len(self.reward_learning["negative_patterns"]),
                "learned_rules": len(self.reward_learning["learned_rules"])
            })
    
    def _get_agent(self, agent_type: str) -> Agent[EmotionalContext]:
        """
        Get an agent of the specified type, initializing it if necessary
        
        Args:
            agent_type: Type of agent to get
            
        Returns:
            The requested agent
        """
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)
            
        return self.agents[agent_type]

    def set_hormone_system(self, hormone_system):
        """Set the hormone system reference"""
        self.hormone_system = hormone_system
    
    # Delegated function for reflection tools
    @function_tool
    async def analyze_emotional_patterns(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Analyze patterns in emotional state history
        
        Returns:
            Analysis of emotional patterns
        """
        with trace(
            workflow_name="Emotional_Pattern_Analysis", 
            trace_id=gen_trace_id(),
            metadata={"cycle": ctx.context.cycle_count}
        ):
            if len(self.emotional_state_history) < 2:
                return {
                    "message": "Not enough emotional state history for pattern analysis",
                    "patterns": {}
                }
            
            patterns = {}
            
            # Track emotion changes over time using an efficient approach
            emotion_trends = defaultdict(list)
            
            # Use a sliding window for more efficient analysis
            analysis_window = self.emotional_state_history[-min(20, len(self.emotional_state_history)):]
            
            for state in analysis_window:
                if "primary_emotion" in state:
                    emotion = state["primary_emotion"].get("name", "Neutral")
                    intensity = state["primary_emotion"].get("intensity", 0.5)
                    emotion_trends[emotion].append(intensity)
            
            # Analyze trends for each emotion
            for emotion, intensities in emotion_trends.items():
                if len(intensities) > 1:
                    # Calculate trend
                    start = intensities[0]
                    end = intensities[-1]
                    change = end - start
                    
                    trend = "stable" if abs(change) < 0.1 else ("increasing" if change > 0 else "decreasing")
                    
                    # Calculate volatility
                    volatility = sum(abs(intensities[i] - intensities[i-1]) for i in range(1, len(intensities))) / (len(intensities) - 1)
                    
                    patterns[emotion] = {
                        "trend": trend,
                        "volatility": volatility,
                        "start_intensity": start,
                        "current_intensity": end,
                        "change": change,
                        "occurrences": len(intensities)
                    }
            
            # Check for emotional oscillation
            oscillation_pairs = [
                ("Joy", "Sadness"),
                ("Trust", "Disgust"),
                ("Fear", "Anger"),
                ("Anticipation", "Surprise")
            ]
            
            emotion_sequence = []
            for state in analysis_window:
                if "primary_emotion" in state:
                    emotion_sequence.append(state["primary_emotion"].get("name", "Neutral"))
            
            for emotion1, emotion2 in oscillation_pairs:
                # Count transitions between the two emotions
                transitions = 0
                for i in range(1, len(emotion_sequence)):
                    if (emotion_sequence[i-1] == emotion1 and emotion_sequence[i] == emotion2) or \
                       (emotion_sequence[i-1] == emotion2 and emotion_sequence[i] == emotion1):
                        transitions += 1
                
                if transitions > 1:
                    patterns[f"{emotion1}-{emotion2} oscillation"] = {
                        "transitions": transitions,
                        "significance": min(1.0, transitions / 5)  # Cap at 1.0
                    }
            
            # Created a custom span for the pattern analysis
            with custom_span(
                "emotional_pattern_analysis",
                data={
                    "patterns_detected": list(patterns.keys()),
                    "emotion_sequence": emotion_sequence[-5:],  # Last 5 emotions
                    "analysis_window_size": len(analysis_window)
                }
            ):
                return {
                    "patterns": patterns,
                    "history_size": len(self.emotional_state_history),
                    "analysis_time": datetime.datetime.now().isoformat()
                }
    
    def _record_emotional_state(self):
        """Record current emotional state in history using efficient circular buffer"""
        # Get current state using the sync version for compatibility
        state = self._get_emotional_state_matrix_sync()
        
        # If this is a new emotion, record a transition
        if len(self.emotional_state_history) > 0:
            prev_emotion = self.emotional_state_history[-1].get("primary_emotion", {}).get("name", "unknown")
            new_emotion = state.get("primary_emotion", {}).get("name", "unknown")
            
            if prev_emotion != new_emotion:
                # Create a custom span for emotion transitions to improve tracing
                with custom_span(
                    "emotion_transition", 
                    data={
                        "from_emotion": prev_emotion,
                        "to_emotion": new_emotion,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "cycle_count": self.context.cycle_count,
                        "type": "emotion_transition"  # Explicit type for analytics processor
                    }
                ):
                    logger.debug(f"Emotion transition: {prev_emotion} -> {new_emotion}")
        
        # Add to history using circular buffer pattern for better memory efficiency
        if len(self.emotional_state_history) < self.max_history_size:
            self.emotional_state_history.append(state)
        else:
            # Overwrite oldest entry
            self.history_index = (self.history_index + 1) % self.max_history_size
            self.emotional_state_history[self.history_index] = state


# Modify process_emotional_input or add tools to recognize triggers:
# - Compliments about appearance/intellect -> potential Attraction trigger
# - Flirtatious or suggestive language -> potential Lust/Desire trigger
# - Descriptions of physical contact (if DSS is integrated) -> trigger relevant sensations/emotions
# - User expressions of attraction/desire -> update RelationshipManager, trigger emotional response
    
    @with_emotion_trace
    async def process_emotional_input(self, text: str) -> Dict[str, Any]:
        """
        Process input text through the DNM and update emotional state using Agent SDK orchestration
        
        Args:
            text: Input text to process
            
        Returns:
            Processing results with updated emotional state
        """
        # Increment context cycle count
        self.context.cycle_count += 1
        
        # Get the orchestrator agent
        orchestrator = self._get_agent("orchestrator")

        final_output = self._extract_final_output(result) # Assuming result holds final state

        analysis_result = await self.emotion_tools.analyze_text_sentiment(ctx) # Assumes this tool exists
    
        compliance_keywords = ["yes mistress", "i obey", "of course", "your command"]
        resistance_keywords = ["no", "i won't", "stop", "don't"]
    
        text_lower = text.lower()
        compliance_score = sum(1 for k in compliance_keywords if k in text_lower)
        resistance_score = sum(1 for k in resistance_keywords if k in text_lower)
    
        # Generate reward/punishment based on compliance/resistance IF the AI is in a dominance context/goal
        # This requires context tracking (e.g., active goal from GoalManager)
        active_goal = self.goal_manager.get_active_goals() # Fictional method to get current goal context
        if active_goal and "control" in active_goal.description.lower():
            if compliance_score > 0 and resistance_score == 0:
                 # Positive reward for compliance
                 reward = RewardSignal(value=0.7 + compliance_score * 0.1, source="user_compliance", context={"text": text}, timestamp=datetime.datetime.now().isoformat())
                 asyncio.create_task(self.reward_system.process_reward_signal(reward))
            elif resistance_score > 0:
                 # Negative reward for resistance
                 reward = RewardSignal(value=-0.5 - resistance_score * 0.1, source="user_resistance", context={"text": text}, timestamp=datetime.datetime.now().isoformat())
                 asyncio.create_task(self.reward_system.process_reward_signal(reward))
        
        # Generate a conversation ID for grouping traces if not present
        conversation_id = self.context.get_value("conversation_id")
        if not conversation_id:
            conversation_id = f"conversation_{datetime.datetime.now().timestamp()}"
            self.context.set_value("conversation_id", conversation_id)
        
        # Create an enhanced RunConfig using the new SDK features
        run_config = RunConfig(
            workflow_name="Emotional_Processing",
            trace_id=f"emotion_trace_{self.context.cycle_count}",
            group_id=conversation_id,  # Group all traces from this conversation
            model=orchestrator.model,
            model_settings=ModelSettings(
                temperature=0.4,
                top_p=0.95,
                max_tokens=300
            ),
            handoff_input_filter=handoff_filters.keep_relevant_history,
            trace_include_sensitive_data=True,
            trace_metadata={
                "system": "nyx_emotional_core",
                "version": "1.0",
                "input_text_length": len(text),
                "pattern_analysis": self._quick_pattern_analysis(text),
                "cycle_count": self.context.cycle_count,
                "conversation_id": conversation_id
            }
        )
        
        # Track API call start time
        start_time = datetime.datetime.now()

        # --- Trigger Hormone Updates based on Strong/Sustained Emotions ---
        if self.hormone_system and (datetime.datetime.now() - self.last_hormone_influence_check).total_seconds() > 1800: # Check every 30 mins
            try:
                current_neurochemicals = {c: d["value"] for c, d in self.neurochemicals.items()}
                ctx = RunContextWrapper(context=self.context) # Create context wrapper

                # Example: Sustained high stress (Cortanyx) might slowly affect Endoryx/Testoryx
                if current_neurochemicals.get("cortanyx", 0) > 0.75:
                    await self.hormone_system.update_hormone(ctx, "endoryx", -0.02, source="sustained_stress")
                    await self.hormone_system.update_hormone(ctx, "testoryx", -0.01, source="sustained_stress")

                # Example: Sustained high bonding (Oxynixin) might boost Oxytonyx
                if current_neurochemicals.get("oxynixin", 0) > 0.75:
                    await self.hormone_system.update_hormone(ctx, "oxytonyx", 0.03, source="sustained_bonding")

                # Example: Sustained pleasure/reward (Nyxamine) might boost Endoryx
                if current_neurochemicals.get("nyxamine", 0) > 0.80:
                     await self.hormone_system.update_hormone(ctx, "endoryx", 0.02, source="sustained_pleasure")

                self.last_hormone_influence_check = datetime.datetime.now()

            except Exception as e:
                logger.error(f"Error triggering hormone updates from emotional state: {e}")        
        
        # Generate a run ID for tracking
        run_id = f"run_{datetime.datetime.now().timestamp()}"
        self.active_runs[run_id] = {
            "start_time": start_time,
            "input": text[:100],  # Truncate for logging
            "status": "running",
            "conversation_id": conversation_id
        }
        
        # Run the orchestrator with proper trace using SDK features
        with trace(
            workflow_name="Emotional_Processing", 
            trace_id=f"emotion_trace_{self.context.cycle_count}",
            group_id=conversation_id,
            metadata={
                "input_text_length": len(text),
                "cycle_count": self.context.cycle_count,
                "run_id": run_id
            }
        ):
            # Use agent_span for enhanced tracing
            with agent_span(
                name=orchestrator.name,
                handoffs=[h.agent_name for h in orchestrator.handoffs],
                tools=[t.name for t in orchestrator.tools],
                output_type="EmotionalResponseOutput"
            ):
                try:
                    # Create structured input for the agent
                    structured_input = json.dumps({
                        "input_text": text,
                        "current_cycle": self.context.cycle_count
                    })
                    
                    # Execute the orchestrator with the run config
                    result = await Runner.run(
                        orchestrator,
                        structured_input,
                        context=self.context,
                        run_config=run_config,
                    )
                    
                    # Calculate duration and update performance metrics
                    duration = (datetime.datetime.now() - start_time).total_seconds()
                    self._update_performance_metrics(duration)
                    
                    # Mark run completion
                    self.active_runs[run_id]["status"] = "completed"
                    self.active_runs[run_id]["duration"] = duration
                    self.active_runs[run_id]["output"] = result.final_output
                    
                    # Record emotional state for history
                    self._record_emotional_state()
                    
                    return result.final_output
                
                except Exception as e:
                    # Enhanced error handling with a custom_span
                    with custom_span(
                        "processing_error",
                        data={
                            "error_type": type(e).__name__,
                            "message": str(e),
                            "run_id": run_id
                        }
                    ):
                        logger.error(f"Error in process_emotional_input: {e}")
                        self.active_runs[run_id]["status"] = "error"
                        self.active_runs[run_id]["error"] = str(e)
                        return {"error": f"Processing failed: {str(e)}"}
    
    @with_emotion_trace
    async def process_emotional_input_streamed(self, text: str) -> AsyncIterator['StreamEvent']:
        """
        Enhanced version that processes input with streaming responses
        to provide real-time emotional reactions with structured events
        
        Args:
            text: Input text to process
            
        Returns:
            Stream of structured emotional response updates
        """
        self.context.cycle_count += 1
        orchestrator = self._get_agent("orchestrator")
        
        # Generate a conversation ID for grouping traces if not present
        conversation_id = self.context.get_value("conversation_id")
        if not conversation_id:
            conversation_id = f"conversation_{datetime.datetime.now().timestamp()}"
            self.context.set_value("conversation_id", conversation_id)
        
        # Create an enhanced run configuration
        run_config = create_run_config(
            workflow_name="Emotional_Processing_Streamed",
            trace_id=f"emotion_stream_{self.context.cycle_count}",
            model=orchestrator.model,
            temperature=0.4,
            cycle_count=self.context.cycle_count,
            context_data={
                "input_text_length": len(text),
                "pattern_analysis": self._quick_pattern_analysis(text),
                "conversation_id": conversation_id
            }
        )
        
        # Generate a run ID for tracking
        run_id = f"stream_{datetime.datetime.now().timestamp()}"
        self.active_runs[run_id] = {
            "start_time": datetime.datetime.now(),
            "input": text[:100], # Truncate for logging
            "status": "streaming",
            "type": "stream",
            "conversation_id": conversation_id
        }
        
        # Use run_streamed with proper context and tracing
        with trace(
            workflow_name="Emotional_Processing_Streamed", 
            trace_id=f"emotion_stream_{self.context.cycle_count}",
            group_id=conversation_id,
            metadata={
                "input_text_length": len(text),
                "cycle_count": self.context.cycle_count,
                "run_id": run_id
            }
        ):
            # Use agent_span for enhanced tracing
            with agent_span(
                name=orchestrator.name,
                handoffs=[h.agent_name for h in orchestrator.handoffs],
                tools=[t.name for t in orchestrator.tools],
                output_type="EmotionalResponseOutput"
            ):
                try:
                    # Create structured input for the agent
                    structured_input = json.dumps({
                        "input_text": text,
                        "current_cycle": self.context.cycle_count
                    })
                    
                    # Use the SDK's streaming capabilities
                    result = Runner.run_streamed(
                        orchestrator,
                        structured_input,
                        context=self.context,
                        run_config=run_config
                    )
                    
                    last_agent = None
                    last_emotion = None
                    
                    # Create a structured event for stream start
                    yield StreamEvent(
                        type="stream_start",
                        data={
                            "timestamp": datetime.datetime.now().isoformat(),
                            "cycle_count": self.context.cycle_count,
                            "input_text_length": len(text)
                        }
                    )
                    
                    # Stream events as they happen with enhanced structure and typing
                    async for event in result.stream_events():
                        # Skip raw events for efficiency
                        if event.type == "raw_response_event":
                            continue
                        
                        if event.type == "run_item_stream_event":
                            if event.item.type == "message_output_item":
                                # Yield full message output only if it's small
                                message_text = ItemHelpers.text_message_output(event.item)
                                if len(message_text) < 200:  # Only stream small messages directly
                                    yield StreamEvent(
                                        type="message_output",
                                        data={
                                            "content": message_text,
                                            "agent": event.item.agent.name
                                        }
                                    )
                                    
                            elif event.item.type == "tool_call_item":
                                tool_name = event.item.raw_item.name
                                tool_args = {}
                                
                                # Parse arguments if available
                                if hasattr(event.item.raw_item, "parameters"):
                                    try:
                                        tool_args = json.loads(event.item.raw_item.parameters)
                                    except:
                                        tool_args = {"raw": str(event.item.raw_item.parameters)}
                                
                                # Create specialized events based on tool type
                                if tool_name == "update_neurochemical":
                                    # Create a more specific event for chemical updates
                                    yield ChemicalUpdateEvent(
                                        data={
                                            "chemical": tool_args.get("chemical", "unknown"),
                                            "value": tool_args.get("value", 0),
                                            "timestamp": datetime.datetime.now().isoformat()
                                        }
                                    )
                                elif tool_name == "derive_emotional_state":
                                    yield StreamEvent(
                                        type="processing",
                                        data={
                                            "phase": "deriving_emotions",
                                            "timestamp": datetime.datetime.now().isoformat()
                                        }
                                    )
                                else:
                                    # Generic tool event
                                    yield StreamEvent(
                                        type="tool_call",
                                        data={
                                            "tool": tool_name,
                                            "args": tool_args,
                                            "timestamp": datetime.datetime.now().isoformat()
                                        }
                                    )
                                    
                            elif event.item.type == "tool_call_output_item":
                                tool_name = "unknown"
                                tool_output = {}
                                
                                # Try to parse tool output as JSON if possible
                                try:
                                    if isinstance(event.item.output, str):
                                        tool_output = json.loads(event.item.output)
                                    else:
                                        tool_output = event.item.output
                                except:
                                    if hasattr(event.item, "output"):
                                        tool_output = {"raw": str(event.item.output)}
                                
                                # Special events for specific tools
                                if "primary_emotion" in tool_output:
                                    # Detect emotion changes
                                    current_emotion = tool_output.get("primary_emotion")
                                    if last_emotion is not None and current_emotion != last_emotion:
                                        # Create specialized event for emotion changes
                                        yield EmotionChangeEvent(
                                            data={
                                                "from": last_emotion,
                                                "to": current_emotion,
                                                "timestamp": datetime.datetime.now().isoformat()
                                            }
                                        )
                                    last_emotion = current_emotion
                                    
                                # Generic tool output event
                                yield StreamEvent(
                                    type="tool_output",
                                    data={
                                        "tool": tool_name,
                                        "output": tool_output,
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                )
                            
                        elif event.type == "agent_updated_stream_event":
                            # Track agent changes
                            if last_agent != event.new_agent.name:
                                yield StreamEvent(
                                    type="agent_changed",
                                    data={
                                        "from": last_agent,
                                        "to": event.new_agent.name,
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                )
                                last_agent = event.new_agent.name
                    
                    # Final complete state event
                    final_data = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "duration": (
                            datetime.datetime.now()
                            - self.active_runs[run_id]["start_time"]
                        ).total_seconds()
                    }
                    
                    final_output = self._extract_final_output(result)
                    if final_output:
                        final_data["final_output"] = final_output
                        
                    yield StreamEvent(
                        type="stream_complete",
                        data=final_data
                    )
                    
                    # Update run status
                    self.active_runs[run_id]["status"] = "completed"
                    self.active_runs[run_id]["end_time"] = datetime.datetime.now()
                    
                    # Record emotional state for history
                    self._record_emotional_state()
                    
                except MaxTurnsExceeded:
                    with custom_span(
                        "streaming_error",
                        data={
                            "error_type": "max_turns_exceeded",
                            "run_id": run_id,
                            "cycle": self.context.cycle_count
                        }
                    ):
                        logger.warning(f"Max turns exceeded in process_emotional_input_streamed for run {run_id}")
                        self.active_runs[run_id]["status"] = "max_turns_exceeded"
                        
                        yield StreamEvent(
                            type="stream_error",
                            data={
                                "error": "Processing exceeded maximum number of steps",
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        )
                    
                except AgentsException as e:
                    with custom_span(
                        "streaming_error",
                        data={
                            "error_type": "agent_exception",
                            "run_id": run_id,
                            "cycle": self.context.cycle_count,
                            "error_detail": str(e)
                        }
                    ):
                        logger.error(f"Agent exception in process_emotional_input_streamed: {e}")
                        self.active_runs[run_id]["status"] = "error"
                        self.active_runs[run_id]["error"] = str(e)
                        
                        yield StreamEvent(
                            type="stream_error",
                            data={
                                "error": f"Processing failed: {str(e)}",
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        )

    def get_lust_level(self) -> float:
        """Estimates the current 'lust' level based on neurochemicals/hormones."""
        libidyx_val = self.hormone_system.hormones.get("libidyx", {}).get("value", 0.4) if self.hormone_system else 0.4
        adrenyx_val = self.neurochemicals.get("adrenyx", {}).get("value", 0.2)
        seranix_val = self.neurochemicals.get("seranix", {}).get("value", 0.6)
        # Combine factors (example formula)
        lust = (libidyx_val * 0.5) + (adrenyx_val * 0.3) + (0.5 - seranix_val) * 0.2
        return max(0.0, min(1.0, lust))
    
    # Legacy API Methods with improved implementation
    def _get_emotional_state_matrix_sync(self) -> Dict[str, Any]:
        """Enhanced synchronous version of _get_emotional_state_matrix for compatibility"""
        # First apply decay
        self.apply_decay()
        
        # Get derived emotions
        emotion_intensities = self._derive_emotional_state_sync()
        
        # Use the same optimized approach as the async version
        # Pre-compute emotion valence and arousal map
        emotion_valence_map = {rule["emotion"]: (rule.get("valence", 0.0), rule.get("arousal", 0.5)) 
                             for rule in self.emotion_derivation_rules}
        
        # Find primary emotion
        primary_emotion = max(emotion_intensities.items(), key=lambda x: x[1]) if emotion_intensities else ("Neutral", 0.5)
        primary_name, primary_intensity = primary_emotion
        
        # Get primary emotion valence and arousal
        primary_valence, primary_arousal = emotion_valence_map.get(primary_name, (0.0, 0.5))
        
        # Process secondary emotions
        secondary_emotions = {}
        for emotion, intensity in emotion_intensities.items():
            if emotion != primary_name and intensity > 0.3:
                valence, arousal = emotion_valence_map.get(emotion, (0.0, 0.5))
                secondary_emotions[emotion] = {
                    "intensity": intensity,
                    "valence": valence,
                    "arousal": arousal
                }
        
        # Calculate overall valence and arousal (weighted average)
        total_intensity = primary_intensity + sum(e["intensity"] for e in secondary_emotions.values())
        
        if total_intensity > 0:
            overall_valence = (primary_valence * primary_intensity)
            overall_arousal = (primary_arousal * primary_intensity)
            
            for emotion, data in secondary_emotions.items():
                overall_valence += data["valence"] * data["intensity"]
                overall_arousal += data["arousal"] * data["intensity"]
                
            overall_valence /= total_intensity
            overall_arousal /= total_intensity
        else:
            overall_valence = 0.0
            overall_arousal = 0.5
        
        # Ensure valence is within range
        overall_valence = max(-1.0, min(1.0, overall_valence))
        overall_arousal = max(0.0, min(1.0, overall_arousal))
        
        return {
            "primary_emotion": {
                "name": primary_name,
                "intensity": primary_intensity,
                "valence": primary_valence,
                "arousal": primary_arousal
            },
            "secondary_emotions": secondary_emotions,
            "valence": overall_valence,
            "arousal": overall_arousal,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _derive_emotional_state_sync(self) -> Dict[str, float]:
        """Enhanced synchronous version of _derive_emotional_state for compatibility"""
        # Get current chemical levels
        chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
        
        # More efficient implementation using dictionary comprehensions
        emotion_scores = {}
        
        for rule in self.emotion_derivation_rules:
            conditions = rule["chemical_conditions"]
            emotion = rule["emotion"]
            rule_weight = rule.get("weight", 1.0)
            
            # Calculate match score using list comprehension for efficiency
            match_scores = [
                min(chemical_levels.get(chemical, 0) / threshold, 1.0)
                for chemical, threshold in conditions.items()
                if chemical in chemical_levels and threshold > 0
            ]
            
            # Average match scores
            avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0
            
            # Apply rule weight
            weighted_score = avg_match_score * rule_weight
            
            # Only include non-zero scores
            if weighted_score > 0:
                emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), weighted_score)
        
        # Normalize if total intensity is too high
        total_intensity = sum(emotion_scores.values())
        if total_intensity > 1.5:
            factor = 1.5 / total_intensity
            emotion_scores = {e: i * factor for e, i in emotion_scores.items()}
        
        return emotion_scores
    
    def apply_decay(self):
        """Apply emotional decay based on time elapsed since last update"""
        try:
            now = datetime.datetime.now()
            time_delta = (now - self.last_update).total_seconds() / 3600  # hours
            
            # Don't decay if less than a minute has passed
            if time_delta < 0.016:  # about 1 minute in hours
                return
            
            # Apply decay to each neurochemical - more efficient with dictionary operations
            for chemical, data in self.neurochemicals.items():
                decay_rate = data["decay_rate"]
                
                # Account for hormone influences by using temporary baseline if available
                if "temporary_baseline" in data:
                    baseline = data["temporary_baseline"]
                else:
                    baseline = data["baseline"]
                    
                current = data["value"]
                
                # Calculate decay based on time passed
                decay_amount = decay_rate * time_delta
                
                # Decay toward baseline
                if current > baseline:
                    self.neurochemicals[chemical]["value"] = max(baseline, current - decay_amount)
                elif current < baseline:
                    self.neurochemicals[chemical]["value"] = min(baseline, current + decay_amount)
            
            # Update timestamp
            self.last_update = now
        except Exception as e:
            logger.error(f"Error in apply_decay: {e}")
    
    def get_active_runs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about active and recent runs
        
        Returns:
            Dictionary of run information
        """
        return self.active_runs
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and performance information
        
        Returns:
            Dictionary of system statistics
        """
        return {
            "api_calls": self.performance_metrics["api_calls"],
            "average_response_time": self.performance_metrics["average_response_time"],
            "update_counts": dict(self.performance_metrics["update_counts"]),
            "total_active_runs": len(self.active_runs),
            "active_run_count": sum(1 for run in self.active_runs.values() if run["status"] == "running"),
            "system_uptime": (datetime.datetime.now() - self.last_update).total_seconds(),
            "agent_count": len(self.agents),
            "emotional_state_history_size": len(self.emotional_state_history),
            "context_cycle_count": self.context.cycle_count
        }
