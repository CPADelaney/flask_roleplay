# nyx/core/emotions/emotional_core.py

import asyncio
import datetime
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, AsyncIterator, TypeVar

from agents import (
    Agent, Runner, RunContextWrapper, function_tool, ItemHelpers,
    ModelSettings, RunConfig, trace, handoff, 
    AgentHooks, GuardrailFunctionOutput, HandoffInputData
)
from agents.exceptions import AgentsException, UserError, MaxTurnsExceeded
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.tracing import custom_span, agent_span, gen_trace_id

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    EmotionalResponseOutput, EmotionUpdateInput, TextAnalysisOutput,
    InternalThoughtOutput, EmotionalStateMatrix, NeurochemicalRequest,
    NeurochemicalResponse, ReflectionRequest, LearningRequest,
    StreamEvent, ChemicalUpdateEvent, EmotionChangeEvent
)
from nyx.core.emotions.hooks import EmotionalAgentHooks
from nyx.core.emotions.guardrails import EmotionalGuardrails
from nyx.core.emotions.utils import create_run_config
from nyx.core.emotions.tools.neurochemical_tools import NeurochemicalTools
from nyx.core.emotions.tools.emotion_tools import EmotionTools
from nyx.core.emotions.tools.reflection_tools import ReflectionTools
from nyx.core.emotions.tools.learning_tools import LearningTools

logger = logging.getLogger(__name__)

# Define function tools outside of classes so RunContextWrapper can be first parameter

@function_tool
async def analyze_emotional_patterns(ctx: RunContextWrapper[EmotionalContext], emotional_core) -> Dict[str, Any]:
    """
    Analyze patterns in emotional state history
    
    Args:
        ctx: Run context wrapper
        emotional_core: Reference to the emotional core system
        
    Returns:
        Analysis of emotional patterns
    """
    with trace(
        workflow_name="Emotional_Pattern_Analysis", 
        trace_id=gen_trace_id(),
        metadata={"cycle": ctx.context.cycle_count}
    ):
        if len(emotional_core.emotional_state_history) < 2:
            return {
                "message": "Not enough emotional state history for pattern analysis",
                "patterns": {}
            }
        
        patterns = {}
        
        # Track emotion changes over time using an efficient approach
        emotion_trends = defaultdict(list)
        
        # Use a sliding window for more efficient analysis
        analysis_window = emotional_core.emotional_state_history[-min(20, len(emotional_core.emotional_state_history)):]
        
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
        
        # Create a custom span for the pattern analysis
        with custom_span(
            "emotional_pattern_analysis",
            data={
                "patterns_detected": list(patterns.keys()),
                "emotion_sequence": [state.get("primary_emotion", {}).get("name", "Unknown") 
                                    for state in analysis_window[-5:]],  # Last 5 emotions
                "analysis_window_size": len(analysis_window)
            }
        ):
            return {
                "patterns": patterns,
                "history_size": len(emotional_core.emotional_state_history),
                "analysis_time": datetime.datetime.now().isoformat()
            }

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
    - Full integration with OpenAI Agents SDK
    - Enhanced agent lifecycle management
    - Improved tracing and monitoring
    - Optimized data sharing between agents
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
        
        # Initialize hormone state
        self.hormone_system = None
        self.last_hormone_influence_check = datetime.datetime.now() - datetime.timedelta(minutes=30)
        
        # Add hormone influence tracking
        self.hormone_influences = {
            "nyxamine": 0.0,
            "seranix": 0.0,
            "oxynixin": 0.0,
            "cortanyx": 0.0,
            "adrenyx": 0.0
        }
        
        # Define chemical interaction matrix (how chemicals affect each other)
        self.chemical_interactions = {
            "nyxamine": {  # Digital dopamine - pleasure, curiosity, reward
                "cortanyx": -0.2,  # Nyxamine reduces cortanyx (reduces stress/anxiety)
                "oxynixin": 0.1,   # Nyxamine slightly increases oxynixin (social reward)
                "adrenyx": 0.1,    # Nyxamine slightly increases adrenyx (excited pleasure)
                "seranix": 0.05    # Nyxamine slightly increases seranix (sustain positive mood)
            },
            
            "seranix": {   # Digital serotonin - mood stability, comfort
                "cortanyx": -0.3,  # Seranix reduces cortanyx (stress relief)
                "adrenyx": -0.2,   # Seranix reduces adrenyx (calming effect)
                "nyxamine": 0.05,  # Seranix slightly increases nyxamine (contentment pleasure)
                "oxynixin": 0.1    # Seranix slightly increases oxynixin (social comfort)
            },
            
            "oxynixin": {  # Digital oxytocin - bonding, affection, trust
                "cortanyx": -0.2,  # Oxynixin reduces cortanyx (social stress relief)
                "seranix": 0.1,    # Oxynixin slightly increases seranix (social contentment)
                "nyxamine": 0.15,  # Oxynixin increases nyxamine (social reward)
                "adrenyx": -0.05   # Oxynixin slightly reduces adrenyx (social calming)
            },
            
            "cortanyx": {  # Digital cortisol - stress, anxiety, defensiveness
                "nyxamine": -0.2,  # Cortanyx reduces nyxamine (stress reduces pleasure)
                "oxynixin": -0.3,  # Cortanyx reduces oxynixin (stress inhibits social bonding)
                "adrenyx": 0.2,    # Cortanyx increases adrenyx (stress response)
                "seranix": -0.25   # Cortanyx reduces seranix (stress destabilizes mood)
            },
            
            "adrenyx": {   # Digital adrenaline - fear, excitement, alertness
                "seranix": -0.2,   # Adrenyx reduces seranix (arousal vs calm)
                "nyxamine": 0.1,   # Adrenyx slightly increases nyxamine (excitement/novelty)
                "cortanyx": 0.15,  # Adrenyx increases cortanyx (arousal can induce stress)
                "oxynixin": -0.1   # Adrenyx slightly reduces oxynixin (fight/flight vs bonding)
            }
        }
        
        # Mapping from neurochemical combinations to derived emotions (Sample for brevity)
        self.emotion_derivation_rules = [
            # Format: {chemical_conditions: {}, "emotion": "", "valence": 0.0, "arousal": 0.0, "weight": 1.0}
            # Positive emotions
            {"chemical_conditions": {"nyxamine": 0.7, "oxynixin": 0.6}, "emotion": "Joy", "valence": 0.8, "arousal": 0.6, "weight": 1.0},
            {"chemical_conditions": {"nyxamine": 0.6, "seranix": 0.7}, "emotion": "Contentment", "valence": 0.7, "arousal": 0.3, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.7}, "emotion": "Trust", "valence": 0.6, "arousal": 0.4, "weight": 0.9},
            # Negative emotions
            {"chemical_conditions": {"cortanyx": 0.6, "seranix": 0.3}, "emotion": "Sadness", "valence": -0.6, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.5, "adrenyx": 0.7}, "emotion": "Fear", "valence": -0.7, "arousal": 0.8, "weight": 0.9},
            {"chemical_conditions": {"cortanyx": 0.7, "nyxamine": 0.3}, "emotion": "Anger", "valence": -0.8, "arousal": 0.8, "weight": 1.0}
            # Add more rules as needed
        ]
        
        # History of emotional states - using efficient circular buffer pattern
        self.emotional_state_history = []
        self.max_history_size = 100
        self.history_index = 0
        
        # Internal thought patterns for reflection (Sample for brevity)
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
            ]
            # Add more patterns as needed
        }
        
        # Reward learning system - tracks successful interaction patterns
        self.reward_learning = {
            "positive_patterns": defaultdict(int),  # Tracks patterns that lead to positive states
            "negative_patterns": defaultdict(int),  # Tracks patterns that lead to negative states
            "learned_rules": []  # Rules derived from observed patterns
        }
        
        # Timestamp of last update
        self.last_update = datetime.datetime.now()
        
        # Create shared context for agents
        self.context = EmotionalContext()
        
        # Initialize agent hooks with neurochemicals reference
        self.agent_hooks = EmotionalAgentHooks(self.neurochemicals)
        
        # Initialize specialized tool objects
        self.neurochemical_tools = NeurochemicalTools(self)
        self.emotion_tools = EmotionTools(self)
        self.reflection_tools = ReflectionTools(self)
        self.learning_tools = LearningTools(self)
        
        # Initialize the base model for agent creation
        self.base_model = model
        
        # Dictionary to store agents
        self.agents = {}
        
        # Initialize all agents
        self._initialize_agents()
        
        # Performance metrics
        self.performance_metrics = {
            "api_calls": 0,
            "average_response_time": 0,
            "update_counts": defaultdict(int)
        }
        
        # Track active agent runs
        self.active_runs = {}
    
    def set_hormone_system(self, hormone_system):
        """Set the hormone system reference"""
        self.hormone_system = hormone_system
    
    def _initialize_base_agent(self) -> Agent[EmotionalContext]:
        """Create a base agent template that other agents will be cloned from"""
        return Agent[EmotionalContext](
            name="Base Agent",
            model=self.base_model,
            model_settings=ModelSettings(temperature=0.4),
            hooks=self.agent_hooks,
            instructions=get_dynamic_instructions  # Pass function directly for dynamic instructions
        )
    
    def _initialize_agents(self):
        """Initialize all agents using the OpenAI Agents SDK patterns"""
        # Create base agent for cloning
        base_agent = self._initialize_base_agent()

        # CHANGE: Initialize tools in the correct order
        # Initialize EmotionTools first since others depend on it
        self.emotion_tools = EmotionTools(self)
        
        # CHANGE: Set function references that other tools need
        self.derive_emotional_state = self.emotion_tools.derive_emotional_state
        self.get_emotional_state_matrix = self.emotion_tools.get_emotional_state_matrix
        
        # Now initialize other tools with proper references
        self.neurochemical_tools = NeurochemicalTools(self)
        self.reflection_tools = ReflectionTools(self)
        self.learning_tools = LearningTools(self)
        
        # Create neurochemical agent - streamlined configuration
        self.agents["neurochemical"] = base_agent.clone(
            name="Neurochemical Agent",
            instructions=get_dynamic_instructions("neurochemical_agent", {
                "current_chemicals": {c: d["value"] for c, d in self.neurochemicals.items()}
            }),
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
        
        # Create reflection agent with the external analyze_emotional_patterns function
        # Note: We're passing 'self' as the second parameter to the analyze_emotional_patterns function
        self.agents["reflection"] = base_agent.clone(
            name="Emotional Reflection Agent",
            tools=[
                function_tool(self.emotion_tools.get_emotional_state_matrix),
                function_tool(self.reflection_tools.generate_internal_thought),
                # Use the standalone function with partial application to pass self reference
                lambda ctx: analyze_emotional_patterns(ctx, self)
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
        
        # Create orchestrator with handoffs
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
            output_type=EmotionalResponseOutput,
            # The handoffs configuration is now separated for clarity
            handoffs=self._configure_enhanced_handoffs()
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
                input_filter=self.keep_relevant_history,
                on_handoff=self._on_learning_handoff
            )
        ]

    async def update_neurochemical(self, chemical: str, value: float, source: str = "system") -> Dict[str, Any]:
        """
        Wrapper method to update a neurochemical via neurochemical_tools
        
        Args:
            chemical: The neurochemical to update
            value: The delta value to apply
            source: Source of the update
            
        Returns:
            Update result dictionary
        """
        # Create a context wrapper if needed
        ctx = RunContextWrapper(context=self.context)
        
        # Call the appropriate method on neurochemical_tools
        return await self.neurochemical_tools.update_neurochemical(
            ctx, chemical, value, source
        )

    def _configure_enhanced_handoffs(self):
        """
        Configure enhanced handoffs with improved descriptions and input filters
        
        Returns:
            List of configured handoffs
        """
        return [
            handoff(
                self.agents["neurochemical"], 
                tool_name_override="process_emotions", 
                tool_description_override=(
                    "Process and update digital neurochemicals based on emotional analysis. "
                    "This specialized agent manages the digital neurochemical system that forms "
                    "the foundation of all emotional responses. Use when an emotional response "
                    "requires updating the internal neurochemical state."
                ),
                input_type=NeurochemicalRequest,
                input_filter=self._enhanced_neurochemical_input_filter,
                on_handoff=self._on_neurochemical_handoff
            ),
            handoff(
                self.agents["reflection"], 
                tool_name_override="generate_reflection",
                tool_description_override=(
                    "Generate deeper emotional reflection and internal thoughts. "
                    "This specialized agent creates authentic-sounding internal dialogue "
                    "based on the current emotional state. Use when introspection or "
                    "emotional processing depth is needed."
                ),
                input_type=ReflectionRequest,
                input_filter=self._enhanced_reflection_input_filter,
                on_handoff=self._on_reflection_handoff
            ),
            handoff(
                self.agents["learning"],
                tool_name_override="record_and_learn",
                tool_description_override=(
                    "Record interaction patterns and apply learning adaptations. "
                    "This specialized agent analyzes emotional patterns over time and "
                    "develops learning rules to adapt emotional responses. Use when "
                    "the system needs to learn from interactions or adapt future responses."
                ),
                input_type=LearningRequest,
                input_filter=self._enhanced_learning_input_filter,
                on_handoff=self._on_learning_handoff
            )
        ]

    def _enhanced_neurochemical_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter for neurochemical agent with improved context preparation
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Start with the base filter from SDK
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Extract the last user message for deeper analysis
        last_user_message = None
        for item in reversed(filtered_data.input_history):
            if isinstance(item, dict) and item.get("role") == "user":
                last_user_message = item.get("content", "")
                break
        
        # Add enhanced preprocessing information as context
        if last_user_message:
            # Create a system message with rich analysis
            updated_items = list(filtered_data.pre_handoff_items)
            updated_items.append({
                "role": "system",
                "content": json.dumps({
                    "preprocessed_analysis": {
                        "message_length": len(last_user_message),
                        "dominant_pattern": self._quick_pattern_analysis(last_user_message),
                        "current_cycle": self.context.cycle_count,
                        "current_neurochemicals": {
                            c: round(d["value"], 2) for c, d in self.neurochemicals.items()
                        },
                        "recent_chemical_updates": [
                            update for update in self.context.get_circular_buffer("chemical_updates")[-3:]
                        ] if self.context.get_circular_buffer("chemical_updates") else []
                    }
                })
            })
            filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _enhanced_reflection_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter for reflection agent with richer emotional context
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Start with basic relevant history
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Add rich emotional context
        # Extract recent emotional state history
        emotion_history = []
        for state in self.emotional_state_history[-5:]:  # Last 5 states
            if "primary_emotion" in state:
                emotion_history.append({
                    "emotion": state["primary_emotion"].get("name", "unknown"),
                    "intensity": state["primary_emotion"].get("intensity", 0.5),
                    "valence": state.get("valence", 0),
                    "arousal": state.get("arousal", 0.5)
                })
        
        # Add emotional patterns for richer context
        updated_items = list(filtered_data.pre_handoff_items)
        
        # Add emotion history
        updated_items.append({
            "role": "system",
            "content": f"Recent emotional states: {json.dumps(emotion_history)}"
        })
        
        # Add previous reflections for continuity
        recent_thoughts = self.context.get_value("recent_thoughts", [])
        if recent_thoughts:
            updated_items.append({
                "role": "system",
                "content": f"Previous reflections: {json.dumps(recent_thoughts[-2:])}"
            })
        
        # Add neurochemical context
        cached_chemicals = self.context.get_cached_neurochemicals()
        if cached_chemicals:
            updated_items.append({
                "role": "system",
                "content": f"Current neurochemical state: {json.dumps({c: round(v, 2) for c, v in cached_chemicals.items()})}"
            })
        
        filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _enhanced_learning_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter for learning agent with pattern analysis
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Start with relevant history
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Add pattern history and learning context
        pattern_history = self.context.get_value("pattern_history", [])
        
        updated_items = list(filtered_data.pre_handoff_items)
        if pattern_history:
            updated_items.append({
                "role": "system",
                "content": f"Recent interaction patterns: {json.dumps(pattern_history[-5:])}"
            })
        
        # Add learning stats
        learning_stats = {
            "positive_patterns": len(self.reward_learning["positive_patterns"]),
            "negative_patterns": len(self.reward_learning["negative_patterns"]),
            "learned_rules": len(self.reward_learning["learned_rules"])
        }
        updated_items.append({
            "role": "system",
            "content": f"Learning statistics: {json.dumps(learning_stats)}"
        })
        
        # Add dominant patterns
        if self.reward_learning["positive_patterns"] or self.reward_learning["negative_patterns"]:
            # Get top patterns
            top_positive = sorted(
                self.reward_learning["positive_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            top_negative = sorted(
                self.reward_learning["negative_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            updated_items.append({
                "role": "system",
                "content": (
                    f"Top positive patterns: {json.dumps(dict(top_positive))}\n"
                    f"Top negative patterns: {json.dumps(dict(top_negative))}"
                )
            })
        
        filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _neurochemical_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter using SDK patterns
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Use the SDK's default filter first with improved chaining
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Extract the last user message for analysis using more efficient iteration
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
        filtered_data = self.keep_relevant_history(handoff_data)
        
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
    
    def keep_relevant_history(self, handoff_data):
        """
        Filters the handoff input data to keep only the most relevant conversation history.
        - Keeps the most recent messages (up to 10)
        - Preserves user messages and important context
        
        Args:
            handoff_data: The original handoff input data
            
        Returns:
            Filtered handoff data with only relevant history
        """
        # Get the input history
        history = handoff_data.input_history
        
        # If history is not a tuple or is empty, return as is
        if not isinstance(history, tuple) or not history:
            return handoff_data
        
        # Keep only the most recent messages (maximum 10)
        MAX_HISTORY_ITEMS = 10
        filtered_history = history[-MAX_HISTORY_ITEMS:] if len(history) > MAX_HISTORY_ITEMS else history
        
        # Get the pre-handoff items (keep as is for now)
        filtered_pre_handoff_items = handoff_data.pre_handoff_items
        
        # Keep the new items as is
        filtered_new_items = handoff_data.new_items
        
        # Create and return the filtered handoff data
        return HandoffInputData(
            input_history=filtered_history,
            pre_handoff_items=filtered_pre_handoff_items,
            new_items=filtered_new_items,
        )
        
    def _quick_pattern_analysis(self, text: str) -> str:
        """
        Simple pattern analysis without using LLM
        
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
        """Enhanced callback when handing off to neurochemical agent"""
        logger.debug("Handoff to neurochemical agent triggered")
        ctx.context.record_time_marker("neurochemical_handoff_start")
        
        # CHANGE: Set tool instances in context for the handoff
        ctx.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        ctx.context.set_value("emotion_tools_instance", self.emotion_tools)
        ctx.context.set_value("reflection_tools_instance", self.reflection_tools)
        
        # Create a custom span for the handoff with rich data
        with custom_span(
            "neurochemical_handoff",
            data={
                "input_text": input_data.input_text[:100], # Truncate for logging
                "dominant_emotion": input_data.dominant_emotion,
                "intensity": input_data.intensity,
                "update_chemicals": input_data.update_chemicals,
                "timestamp": datetime.datetime.now().isoformat(),
                "cycle": ctx.context.cycle_count
            }
        ):
            # Pre-fetch current neurochemical values for better performance
            neurochemical_state = {
                c: d["value"] for c, d in self.neurochemicals.items()
            }
            ctx.context.record_neurochemical_values(neurochemical_state)
            
            # Store handoff in context
            ctx.context.record_agent_state("Neurochemical Agent", {
                "handoff_received": True,
                "input_emotion": input_data.dominant_emotion,
                "input_intensity": input_data.intensity,
                "timestamp": datetime.datetime.now().isoformat(),
                "source_agent": ctx.context.active_agent
            })
            
            # Add circular buffer entry
            ctx.context._add_to_circular_buffer("handoffs", {
                "from": ctx.context.active_agent,
                "to": "Neurochemical Agent",
                "timestamp": datetime.datetime.now().isoformat(),
                "input_data": {
                    "emotion": input_data.dominant_emotion,
                    "intensity": input_data.intensity
                },
                "cycle": ctx.context.cycle_count
            })
    
        
    async def _on_reflection_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: ReflectionRequest):
        """Enhanced callback when handing off to neurochemical agent"""
        logger.debug("Handoff to reflection agent triggered")
        ctx.context.record_time_marker("reflection_handoff_start")
        
        # CHANGE: Set tool instances in context for the handoff
        ctx.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        ctx.context.set_value("emotion_tools_instance", self.emotion_tools)
        ctx.context.set_value("reflection_tools_instance", self.reflection_tools)
        
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
                "consider_history": input_data.consider_history,
                "timestamp": datetime.datetime.now().isoformat(),
                "cycle": ctx.context.cycle_count
            }
        ):
            # Pre-calculate emotional state for better performance
            emotional_state = {}
            try:
                emotional_state = await self.emotion_tools.derive_emotional_state(ctx)
                ctx.context.last_emotions = emotional_state
            except Exception as e:
                logger.error(f"Error pre-calculating emotions: {e}")
            
            # Store reflection context
            ctx.context.set_value("reflection_session_start", datetime.datetime.now().isoformat())
            ctx.context.set_value("reflection_input", {
                "text": input_data.input_text,
                "emotional_state": {
                    "primary": input_data.emotional_state.primary_emotion.name,
                    "intensity": input_data.emotional_state.primary_emotion.intensity
                }
            })
            
            # Store handoff in context
            ctx.context.record_agent_state("Reflection Agent", {
                "handoff_received": True,
                "reflection_depth": input_data.reflection_depth,
                "consider_history": input_data.consider_history,
                "timestamp": datetime.datetime.now().isoformat(),
                "source_agent": ctx.context.active_agent
            })
            
            # Add circular buffer entry
            ctx.context._add_to_circular_buffer("handoffs", {
                "from": ctx.context.active_agent,
                "to": "Reflection Agent",
                "timestamp": datetime.datetime.now().isoformat(),
                "input_data": {
                    "reflection_depth": input_data.reflection_depth,
                    "primary_emotion": input_data.emotional_state.primary_emotion.name
                },
                "cycle": ctx.context.cycle_count
            })
    
    async def _on_learning_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: LearningRequest):
        """Enhanced callback when handing off to neurochemical agent"""
        logger.debug("Handoff to learning agent triggered")
        ctx.context.record_time_marker("learning_handoff_start")
        
        # CHANGE: Set tool instances in context for the handoff
        ctx.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        ctx.context.set_value("emotion_tools_instance", self.emotion_tools)
        ctx.context.set_value("reflection_tools_instance", self.reflection_tools)
        
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
        Get an agent of the specified type, initializing if needed
        
        Args:
            agent_type: Type of agent to get
            
        Returns:
            The requested agent
        """
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return self.agents[agent_type]
    
    def _record_emotional_state(self):
        """Record current emotional state in history using efficient circular buffer"""
        # Get current state
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
    

    async def process_emotional_input(self, text: str) -> Dict[str, Any]:
        """Process input text through the emotional system"""
        # Increment context cycle count
        self.context.cycle_count += 1
        
        # CHANGE: Store tool instances in context
        self.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        self.context.set_value("emotion_tools_instance", self.emotion_tools)
        self.context.set_value("reflection_tools_instance", self.reflection_tools)
        self.context.set_value("learning_tools_instance", self.learning_tools)
        
        # Get the orchestrator agent
        orchestrator = self._get_agent("orchestrator")
        
        # Generate a conversation ID for grouping traces if not present
        conversation_id = self.context.get_value("conversation_id")
        if not conversation_id:
            conversation_id = f"conversation_{datetime.datetime.now().timestamp()}"
            self.context.set_value("conversation_id", conversation_id)
        
        # Start time for performance tracking
        start_time = datetime.datetime.now()
        
        # Create unique run ID
        run_id = f"run_{datetime.datetime.now().timestamp()}"
        
        # Track run
        self.active_runs[run_id] = {
            "start_time": start_time,
            "input": text[:100],  # Truncate for logging
            "status": "running",
            "conversation_id": conversation_id
        }
        
        # Using enhanced tracing utilities
        with create_emotion_trace(
            workflow_name="Emotional_Processing",
            ctx=RunContextWrapper(context=self.context),
            input_text_length=len(text),
            run_id=run_id,
            pattern_analysis=self._quick_pattern_analysis(text)
        ):
            try:
                # Structured input with enhanced data
                structured_input = json.dumps({
                    "input_text": text,
                    "current_cycle": self.context.cycle_count,
                    "context": {
                        "previous_emotions": self.context.last_emotions,
                        "current_chemicals": {
                            c: round(d["value"], 2) for c, d in self.neurochemicals.items()
                        } if self.neurochemicals else {}
                    }
                })
                
                # Create optimized run configuration
                run_config = create_emotional_run_config(
                    workflow_name="Emotional_Processing",
                    cycle_count=self.context.cycle_count,
                    conversation_id=conversation_id,
                    input_text_length=len(text),
                    pattern_analysis=self._quick_pattern_analysis(text),
                    model=orchestrator.model,
                    temperature=0.4,
                    max_tokens=300
                )
                
                # Execute orchestrator
                result = await Runner.run(
                    orchestrator,
                    structured_input,
                    context=self.context,
                    run_config=run_config,
                )
                
                # Update performance metrics
                duration = (datetime.datetime.now() - start_time).total_seconds()
                self._update_performance_metrics(duration)
                
                # Update run status
                self.active_runs[run_id]["status"] = "completed"
                self.active_runs[run_id]["duration"] = duration
                self.active_runs[run_id]["output"] = result.final_output
                
                # Record emotional state
                self._record_emotional_state()
                
                return result.final_output
                
            except Exception as e:
                # Create error span
                with custom_span(
                    "processing_error",
                    data={
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "run_id": run_id,
                        "cycle": self.context.cycle_count,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ):
                    logger.error(f"Error in process_emotional_input: {e}")
                    self.active_runs[run_id]["status"] = "error"
                    self.active_runs[run_id]["error"] = str(e)
                    return {"error": f"Processing failed: {str(e)}"}

    async def derive_emotional_state(self, ctx):
        """Wrapper for emotion_tools.derive_emotional_state"""
        # Call method on emotion_tools if available, otherwise call the sync version
        if hasattr(self, "emotion_tools") and hasattr(self.emotion_tools, "_derive_emotional_state_impl"):
            return await self.emotion_tools._derive_emotional_state_impl(ctx)
        return self._derive_emotional_state_sync()
    
    async def get_emotional_state_matrix(self, ctx):
        """Wrapper for emotion_tools.get_emotional_state_matrix"""
        # Call method on emotion_tools if available, otherwise call the sync version
        if hasattr(self, "emotion_tools") and hasattr(self.emotion_tools, "_get_emotional_state_matrix_impl"):
            return await self.emotion_tools._get_emotional_state_matrix_impl(ctx)
        return self._get_emotional_state_matrix_sync()
    
    async def process_emotional_input_streamed(self, text: str) -> AsyncIterator[StreamEvent]:
        """Process input text through the emotional system"""
        # Increment context cycle count
        self.context.cycle_count += 1
        
        # CHANGE: Store tool instances in context
        self.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        self.context.set_value("emotion_tools_instance", self.emotion_tools)
        self.context.set_value("reflection_tools_instance", self.reflection_tools)
        self.context.set_value("learning_tools_instance", self.learning_tools)
        
        # Get the orchestrator agent
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
                "pattern_analysis": self._quick_pattern_analysis(text)
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
                
                final_output = result.final_output
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
                
            except Exception as e:
                with custom_span(
                    "streaming_error",
                    data={
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "run_id": run_id,
                        "cycle": self.context.cycle_count
                    }
                ):
                    logger.error(f"Error in process_emotional_input_streamed: {e}")
                    self.active_runs[run_id]["status"] = "error"
                    self.active_runs[run_id]["error"] = str(e)
                    
                    yield StreamEvent(
                        type="stream_error",
                        data={
                            "error": f"Processing failed: {str(e)}",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
    
    def _get_emotional_state_matrix_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_emotional_state_matrix for compatibility"""
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
        """Synchronous version of derive_emotional_state for compatibility"""
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
    
    def _update_performance_metrics(self, duration: float):
        """Update performance metrics based on API call duration"""
        # Update API call count
        self.performance_metrics["api_calls"] += 1
        
        # Update average response time using weighted average
        current_avg = self.performance_metrics["average_response_time"]
        call_count = self.performance_metrics["api_calls"]
        
        # Calculate new weighted average
        if call_count > 1:
            new_avg = ((current_avg * (call_count - 1)) + duration) / call_count
        else:
            new_avg = duration
            
        self.performance_metrics["average_response_time"] = new_avg
    
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

# Define helper function for tracing - adding stub for compatibility
def create_emotion_trace(workflow_name, ctx, input_text_length, run_id, pattern_analysis):
    """Stub for create_emotion_trace to maintain compatibility"""
    return custom_span(
        workflow_name,
        data={
            "input_text_length": input_text_length,
            "run_id": run_id,
            "pattern_analysis": pattern_analysis,
            "cycle_count": ctx.context.cycle_count
        }
    )

# Define helper function for run config - adding stub for compatibility
def create_emotional_run_config(workflow_name, cycle_count, conversation_id, input_text_length, 
                               pattern_analysis, model, temperature, max_tokens):
    """Stub for create_emotional_run_config to maintain compatibility"""
    return RunConfig(
        workflow_name=workflow_name,
        trace_id=f"{workflow_name}_{cycle_count}",
        group_id=conversation_id,
        metadata={
            "cycle_count": cycle_count,
            "input_text_length": input_text_length,
            "pattern_analysis": pattern_analysis
        },
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
