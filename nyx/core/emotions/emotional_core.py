# nyx/core/emotions/emotional_core.py

"""
Enhanced agent-based emotion management system for Nyx.

Implements a digital neurochemical model that produces complex
emotional states using the OpenAI Agents SDK.
"""

import asyncio
import datetime
import functools
import json
import logging
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set, AsyncIterator

from agents import (
    Agent, Runner, trace, RunContextWrapper, ItemHelpers,
    ModelSettings, RunConfig, input_guardrail, output_guardrail,
    function_tool, handoff
)
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.exceptions import AgentsException, ModelBehaviorError, UserError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    EmotionalResponseOutput, EmotionUpdateInput, TextAnalysisOutput,
    InternalThoughtOutput, EmotionalStateMatrix
)
from nyx.core.emotions.hooks import EmotionalAgentHooks
from nyx.core.emotions.guardrails import EmotionalGuardrails
from nyx.core.emotions.utils import handle_errors, create_run_config
from nyx.core.emotions.tools.neurochemical_tools import NeurochemicalTools
from nyx.core.emotions.tools.emotion_tools import EmotionTools
from nyx.core.emotions.tools.reflection_tools import ReflectionTools
from nyx.core.emotions.tools.learning_tools import LearningTools

logger = logging.getLogger(__name__)

# Agent instructions with proper prompt template
AGENT_INSTRUCTIONS = {
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

class EmotionalCore:
    """
    Enhanced agent-based emotion management system for Nyx implementing the Digital Neurochemical Model.
    Simulates a digital neurochemical environment that produces complex emotional states.
    """
    
    def __init__(self):
        """Initialize the emotional core system"""
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
        
        # Store reference to hormone system
        self.hormone_system = None  # Will be set later
        
        # Add hormone influence tracking
        self.hormone_influences = {
            "nyxamine": 0.0,
            "seranix": 0.0,
            "oxynixin": 0.0,
            "cortanyx": 0.0,
            "adrenyx": 0.0
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
            # [More patterns omitted for brevity but would be included in the actual code]
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
        
        # Initialize agents dict - will use factory pattern with agent cloning
        self.agents = {}
        
        # Performance metrics
        self.performance_metrics = {
            "api_calls": 0,
            "average_response_time": 0,
            "update_counts": defaultdict(int)
        }

    def _setup_tracing(self):
        """Configure custom trace processor for emotional analytics"""
        from agents.tracing import add_trace_processor, BatchTraceProcessor
        from agents.tracing.processors import BackendSpanExporter
        
        emotion_trace_processor = BatchTraceProcessor(
            exporter=BackendSpanExporter(project="nyx_emotional_system"),
            max_batch_size=100,
            schedule_delay=3.0
        )
        add_trace_processor(emotion_trace_processor)
    
    def _initialize_agents(self):
        """Initialize all agents using factory pattern with agent cloning"""
        # Create a base agent with common settings as template
        base_agent = Agent[EmotionalContext](
            name="Base Agent",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.4),
            hooks=self.agent_hooks
        )
        
        # Create neurochemical agent
        self.agents["neurochemical"] = base_agent.clone(
            name="Neurochemical Agent",
            instructions=prompt_with_handoff_instructions(AGENT_INSTRUCTIONS["neurochemical_agent"]),
            tools=[
                function_tool(self.neurochemical_tools.update_neurochemical),
                function_tool(self.neurochemical_tools.apply_chemical_decay),
                function_tool(self.neurochemical_tools.process_chemical_interactions),
                function_tool(self.neurochemical_tools.get_neurochemical_state)
            ],
            input_guardrails=[input_guardrail(EmotionalGuardrails.validate_emotional_input)]
        )
        
        # Create emotion derivation agent
        self.agents["emotion_derivation"] = base_agent.clone(
            name="Emotion Derivation Agent",
            instructions=prompt_with_handoff_instructions(AGENT_INSTRUCTIONS["emotion_derivation_agent"]),
            tools=[
                function_tool(self.neurochemical_tools.get_neurochemical_state),
                function_tool(self.emotion_tools.derive_emotional_state),
                function_tool(self.emotion_tools.get_emotional_state_matrix)
            ],
            output_type=EmotionalStateMatrix
        )
        
        # Create reflection agent
        self.agents["reflection"] = base_agent.clone(
            name="Emotional Reflection Agent",
            instructions=prompt_with_handoff_instructions(AGENT_INSTRUCTIONS["reflection_agent"]),
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
            instructions=prompt_with_handoff_instructions(AGENT_INSTRUCTIONS["learning_agent"]),
            tools=[
                function_tool(self.learning_tools.record_interaction_outcome),
                function_tool(self.learning_tools.update_learning_rules),
                function_tool(self.learning_tools.apply_learned_adaptations)
            ],
            model_settings=ModelSettings(temperature=0.4)  # Medium temperature for balanced learning
        )
        
        # Create orchestrator with enhanced handoffs
        self.agents["orchestrator"] = base_agent.clone(
            name="Emotion_Orchestrator",
            instructions=prompt_with_handoff_instructions(AGENT_INSTRUCTIONS["emotion_orchestrator"]),
            handoffs=[
                handoff(
                    self.agents["neurochemical"], 
                    tool_name_override="process_emotions", 
                    tool_description_override="Process and update neurochemicals based on emotional input analysis. Use this when you need to trigger emotional changes.",
                    input_filter=handoff_filters.keep_relevant_history,
                    on_handoff=self._on_neurochemical_handoff
                ),
                handoff(
                    self.agents["reflection"], 
                    tool_name_override="generate_reflection",
                    tool_description_override="Generate emotional reflection when user input triggers significant emotional response. Use when deeper introspection is needed.",
                    input_filter=handoff_filters.keep_relevant_history,
                    on_handoff=self._on_reflection_handoff
                ),
                handoff(
                    self.agents["learning"],
                    tool_name_override="record_and_learn",
                    tool_description_override="Record interaction patterns and apply learning to adapt emotional responses. Use after completing emotional processing.",
                    input_filter=handoff_filters.keep_relevant_history,
                    on_handoff=self._on_learning_handoff
                )
            ],
            tools=[
                function_tool(self.emotion_tools.analyze_text_sentiment)
            ],
            input_guardrails=[input_guardrail(EmotionalGuardrails.validate_emotional_input)],
            output_guardrails=[output_guardrail(EmotionalGuardrails.validate_emotional_output)],
            output_type=EmotionalResponseOutput  # Specify structured output
        )
    
    async def _on_neurochemical_handoff(self, ctx: RunContextWrapper[EmotionalContext]):
        """Callback when handing off to neurochemical agent"""
        logger.debug("Handoff to neurochemical agent triggered")
        ctx.context.record_time_marker("neurochemical_handoff_start")
        
        # Pre-fetch current neurochemical values for better performance
        neurochemical_state = {
            c: d["value"] for c, d in self.neurochemicals.items()
        }
        ctx.context.record_neurochemical_values(neurochemical_state)
    
    async def _on_reflection_handoff(self, ctx: RunContextWrapper[EmotionalContext]):
        """Callback when handing off to reflection agent"""
        logger.debug("Handoff to reflection agent triggered")
        ctx.context.record_time_marker("reflection_handoff_start")
        
        # Pre-calculate emotional state for better performance
        emotional_state = {}
        try:
            emotional_state = await self.emotion_tools.derive_emotional_state(ctx)
            ctx.context.last_emotions = emotional_state
        except Exception as e:
            logger.error(f"Error pre-calculating emotions: {e}")
    
    async def _on_learning_handoff(self, ctx: RunContextWrapper[EmotionalContext]):
        """Callback when handing off to learning agent"""
        logger.debug("Handoff to learning agent triggered")
        ctx.context.record_time_marker("learning_handoff_start")
        
        # Prepare learning context data
        ctx.context.set_value("reward_learning_stats", {
            "positive_patterns": len(self.reward_learning["positive_patterns"]),
            "negative_patterns": len(self.reward_learning["negative_patterns"]),
            "learned_rules": len(self.reward_learning["learned_rules"])
        })
    
    def _ensure_agent(self, agent_type: str) -> Agent[EmotionalContext]:
        """
        Ensure a specific agent is initialized
        
        Args:
            agent_type: Type of agent to ensure
            
        Returns:
            The requested agent
        """
        if not self.agents:
            self._initialize_agents()
            
        if agent_type not in self.agents:
            raise UserError(f"Unknown agent type: {agent_type}")
        
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
        with trace(workflow_name="Emotional_Pattern_Analysis", 
                  trace_id=f"pattern_analysis_{datetime.datetime.now().timestamp()}",
                  metadata={"cycle": ctx.context.cycle_count}):
            
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
            
            return {
                "patterns": patterns,
                "history_size": len(self.emotional_state_history),
                "analysis_time": datetime.datetime.now().isoformat()
            }
    
    def _record_emotional_state(self):
        """Record current emotional state in history using efficient circular buffer"""
        # Get current state using the sync version for compatibility
        state = self._get_emotional_state_matrix_sync()
        
        # Add to history using circular buffer pattern for better memory efficiency
        if len(self.emotional_state_history) < self.max_history_size:
            self.emotional_state_history.append(state)
        else:
            # Overwrite oldest entry
            self.emotional_state_history[self.history_index] = state
            self.history_index = (self.history_index + 1) % self.max_history_size
    
    @handle_errors("Error processing emotional input")
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
        orchestrator = self._ensure_agent("orchestrator")
        
        # Define efficient run configuration
        run_config = create_run_config(
            workflow_name="Emotional_Processing",
            trace_id=f"emotion_trace_{self.context.cycle_count}",
            temperature=0.4,
            max_tokens=300,
            cycle_count=self.context.cycle_count
        )
        
        # Track API call start time
        start_time = datetime.datetime.now()
        
        # Run the orchestrator with context sharing and proper trace
        with trace(
            workflow_name="Emotional_Processing", 
            trace_id=f"emotion_trace_{self.context.cycle_count}",
            metadata={
                "input_text_length": len(text),
                "cycle_count": self.context.cycle_count
            }
        ):
            result = await Runner.run(
                orchestrator,
                json.dumps({
                    "input_text": text,
                    "current_cycle": self.context.cycle_count
                }),
                context=self.context,
                run_config=run_config
            )
            
            # Update performance metrics
            duration = (datetime.datetime.now() - start_time).total_seconds()
            self.performance_metrics["api_calls"] += 1
            
            # Update running average of response time
            prev_avg = self.performance_metrics["average_response_time"]
            prev_count = self.performance_metrics["api_calls"] - 1
            if prev_count > 0:
                self.performance_metrics["average_response_time"] = (prev_avg * prev_count + duration) / self.performance_metrics["api_calls"]
            else:
                self.performance_metrics["average_response_time"] = duration
            
            return result.final_output
    
    @handle_errors("Error processing emotional input with streaming")
    async def process_emotional_input_streamed(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Process input with streaming responses to provide real-time emotional reactions
        
        Args:
            text: Input text to process
            
        Returns:
            Stream of emotional response updates
        """
        self.context.cycle_count += 1
        orchestrator = self._ensure_agent("orchestrator")
        run_config = create_run_config(
            workflow_name="Emotional_Processing_Streamed",
            trace_id=f"emotion_stream_{self.context.cycle_count}",
            temperature=0.4,
            cycle_count=self.context.cycle_count
        )
        
        # Use run_streamed instead of run
        result = Runner.run_streamed(
            orchestrator,
            json.dumps({"input_text": text, "current_cycle": self.context.cycle_count}),
            context=self.context,
            run_config=run_config
        )
        
        # Stream events as they happen
        async for event in result.stream_events():
            if event.type == "run_item_stream_event":
                if event.item.type == "message_output_item":
                    yield {
                        "type": "emotional_update",
                        "content": ItemHelpers.text_message_output(event.item)
                    }
                elif event.item.type == "tool_call_item":
                    yield {
                        "type": "tool_call",
                        "tool": event.item.raw_item.name,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
            elif event.type == "agent_updated_stream_event":
                yield {
                    "type": "agent_changed",
                    "agent": event.new_agent.name,
                    "timestamp": datetime.datetime.now().isoformat()
                }
    
    # Legacy API Methods for backward compatibility
    def _get_emotional_state_matrix_sync(self) -> Dict[str, Any]:
        """Synchronous version of _get_emotional_state_matrix for compatibility"""
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
        """Synchronous version of _derive_emotional_state for compatibility"""
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
