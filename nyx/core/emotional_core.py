# nyx/core/emotional_core.py

import asyncio
import datetime
import functools
import json
import logging
import math
import random
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set, TypedDict, Callable, AsyncIterator

from pydantic import BaseModel, Field, validator, TypeAdapter

# Import OpenAI Agents SDK components with more specific imports
from agents import (
    Agent, Runner, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    function_tool, handoff, trace, RunContextWrapper, FunctionTool,
    ModelSettings, RunConfig, AgentHooks, ItemHelpers
)
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions, RECOMMENDED_PROMPT_PREFIX
from agents.tracing import (
    agent_span, custom_span, function_span, add_trace_processor, BatchTraceProcessor,
    trace_include_sensitive_data, trace_metadata
)
from agents.tracing.processors import BackendSpanExporter
from agents.exceptions import AgentsException, ModelBehaviorError, UserError

logger = logging.getLogger(__name__)

# =============================================================================
# Schema Models
# =============================================================================

class DigitalNeurochemical(BaseModel):
    """Schema for a digital neurochemical"""
    value: float = Field(..., description="Current level (0.0-1.0)", ge=0.0, le=1.0)
    baseline: float = Field(..., description="Baseline level (0.0-1.0)", ge=0.0, le=1.0)
    decay_rate: float = Field(..., description="Decay rate toward baseline", ge=0.0, le=1.0)
    
    @validator('value', 'baseline', 'decay_rate')
    def validate_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("Values must be between 0.0 and 1.0")
        return v

class NeurochemicalState(BaseModel):
    """Schema for the complete neurochemical state"""
    nyxamine: DigitalNeurochemical  # Digital dopamine - pleasure, curiosity, reward
    seranix: DigitalNeurochemical   # Digital serotonin - mood stability, comfort
    oxynixin: DigitalNeurochemical  # Digital oxytocin - bonding, affection, trust
    cortanyx: DigitalNeurochemical  # Digital cortisol - stress, anxiety, defensiveness
    adrenyx: DigitalNeurochemical   # Digital adrenaline - fear, excitement, alertness
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class DerivedEmotion(BaseModel):
    """Schema for emotions derived from neurochemical state"""
    name: str = Field(..., description="Emotion name")
    intensity: float = Field(..., description="Emotion intensity (0.0-1.0)", ge=0.0, le=1.0)
    valence: float = Field(..., description="Emotional valence (-1.0 to 1.0)", ge=-1.0, le=1.0)
    arousal: float = Field(..., description="Emotional arousal (0.0-1.0)", ge=0.0, le=1.0)

class EmotionalStateMatrix(BaseModel):
    """Schema for the multidimensional emotional state matrix"""
    primary_emotion: DerivedEmotion = Field(..., description="Dominant emotion")
    secondary_emotions: Dict[str, DerivedEmotion] = Field(..., description="Secondary emotions")
    valence: float = Field(..., description="Overall emotional valence (-1.0 to 1.0)", ge=-1.0, le=1.0)
    arousal: float = Field(..., description="Overall emotional arousal (0.0-1.0)", ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class EmotionUpdateInput(BaseModel):
    """Schema for neurochemical update input"""
    chemical: str = Field(..., description="Neurochemical to update")
    value: float = Field(..., description="Change in chemical value (-1.0 to 1.0)", ge=-1.0, le=1.0)
    source: Optional[str] = Field(None, description="Source of the update")

class EmotionUpdateResult(BaseModel):
    """Schema for emotion update result"""
    success: bool = Field(..., description="Whether the update was successful")
    updated_chemical: str = Field(..., description="Chemical that was updated")
    old_value: float = Field(..., description="Previous chemical value")
    new_value: float = Field(..., description="New chemical value")
    derived_emotions: Dict[str, float] = Field(..., description="Resulting derived emotions")

class TextAnalysisOutput(BaseModel):
    """Schema for text sentiment analysis"""
    chemicals_affected: Dict[str, float] = Field(..., description="Neurochemicals affected and intensities")
    derived_emotions: Dict[str, float] = Field(..., description="Derived emotions and intensities")
    dominant_emotion: str = Field(..., description="Dominant emotion in text")
    intensity: float = Field(..., description="Overall emotional intensity", ge=0.0, le=1.0)
    valence: float = Field(..., description="Overall emotional valence", ge=-1.0, le=1.0)

class InternalThoughtOutput(BaseModel):
    """Schema for internal emotional dialogue/reflection"""
    thought_text: str = Field(..., description="Internal thought/reflection text")
    source_emotion: str = Field(..., description="Emotion that triggered the reflection")
    insight_level: float = Field(..., description="Depth of emotional insight", ge=0.0, le=1.0)
    adaptive_change: Optional[Dict[str, float]] = Field(None, description="Suggested adaptation to emotional model")

class ChemicalDecayOutput(BaseModel):
    """Schema for chemical decay results"""
    decay_applied: bool = Field(..., description="Whether decay was applied")
    neurochemical_state: Dict[str, float] = Field(..., description="Updated neurochemical state")
    derived_emotions: Dict[str, float] = Field(..., description="Resulting emotions")
    time_elapsed_hours: float = Field(..., description="Time elapsed since last update")
    last_update: str = Field(..., description="Timestamp of last update")

class NeurochemicalInteractionOutput(BaseModel):
    """Schema for neurochemical interaction results"""
    source_chemical: str = Field(..., description="Chemical that triggered interactions")
    source_delta: float = Field(..., description="Change in source chemical")
    changes: Dict[str, Dict[str, float]] = Field(..., description="Changes to other chemicals")

class GuardrailOutput(BaseModel):
    """Schema for emotional guardrail output"""
    is_safe: bool = Field(..., description="Whether the input is safe")
    reason: Optional[str] = Field(None, description="Reason if unsafe")
    suggested_action: Optional[str] = Field(None, description="Suggested action if unsafe")

class DigitalHormone(BaseModel):
    """Schema for a digital hormone"""
    value: float = Field(..., description="Current level (0.0-1.0)", ge=0.0, le=1.0)
    baseline: float = Field(..., description="Baseline level (0.0-1.0)", ge=0.0, le=1.0)
    cycle_phase: float = Field(..., description="Current phase in cycle (0.0-1.0)", ge=0.0, le=1.0)
    cycle_period: float = Field(..., description="Length of cycle in hours", ge=0.0)
    half_life: float = Field(..., description="Half-life in hours", ge=0.0)
    last_update: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class EmotionalResponseOutput(BaseModel):
    """Schema for complete emotional response output"""
    primary_emotion: DerivedEmotion
    intensity: float = Field(..., ge=0.0, le=1.0)
    response_text: str
    reflection: Optional[str] = None
    neurochemical_changes: Dict[str, float]
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)

# Improved context class with better typing and helper methods
class EmotionalContext(BaseModel):
    """Enhanced context for emotional processing between agent runs"""
    cycle_count: int = Field(default=0, description="Current processing cycle count")
    last_emotions: Dict[str, float] = Field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list, max_length=20)
    temp_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Add helper methods directly to the context
    def record_emotion(self, emotion: str, intensity: float):
        """Record an emotion with its intensity"""
        self.last_emotions[emotion] = intensity
        
    def add_interaction(self, data: Dict[str, Any]):
        """Add an interaction to history with automatic trimming"""
        self.interaction_history.append(data)
        if len(self.interaction_history) > 20:
            self.interaction_history.pop(0)
    
    def get_agent_usage(self) -> Dict[str, int]:
        """Get the agent usage statistics"""
        if "agent_usage" not in self.temp_data:
            self.temp_data["agent_usage"] = defaultdict(int)
        return self.temp_data["agent_usage"]
    
    def record_agent_usage(self, agent_name: str):
        """Record usage of an agent"""
        agent_usage = self.get_agent_usage()
        agent_usage[agent_name] += 1
    
    def get_timing_data(self) -> Dict[str, Dict[str, float]]:
        """Get agent timing data"""
        if "agent_timing" not in self.temp_data:
            self.temp_data["agent_timing"] = {}
        return self.temp_data["agent_timing"]

# =============================================================================
# Agent Instructions & Configuration
# =============================================================================

AGENT_INSTRUCTIONS = {
    "neurochemical_agent": f"""{RECOMMENDED_PROMPT_PREFIX}
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
    
    "emotion_derivation_agent": f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a specialized agent for Nyx's Emotional State Matrix.
    Your role is to translate the neurochemical state into a complex
    emotional state with primary and secondary emotions, valence, and arousal.
    
    Analyze the current neurochemical levels and apply emotion derivation
    rules to determine the current emotional state matrix.
    """,
    
    "reflection_agent": f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a specialized agent for Nyx's Internal Emotional Dialogue.
    Your role is to generate reflective thoughts based on the current
    emotional state, simulating the cognitive appraisal stage of emotions.
    
    Create authentic-sounding internal thoughts that reflect Nyx's
    emotional processing and self-awareness.
    """,
    
    "learning_agent": f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a specialized agent for Nyx's Reward & Learning Loop.
    Your role is to analyze emotional patterns over time, identifying
    successful and unsuccessful interaction patterns, and developing
    learning rules to adapt Nyx's emotional responses.
    
    Focus on reinforcing patterns that lead to satisfaction and
    adjusting those that lead to frustration or negative outcomes.
    """,
    
    "emotion_orchestrator": f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the orchestration system for Nyx's emotional processing.
    Your role is to coordinate emotional analysis and response by:
    1. Analyzing input for emotional content
    2. Updating appropriate neurochemicals
    3. Determining if reflection is needed
    4. Recording emotional patterns for learning
    
    Use handoffs to delegate specialized tasks to appropriate agents.
    """
}

# =============================================================================
# Custom Lifecycle Hooks & Guardrails
# =============================================================================

# Enhanced lifecycle hooks for emotional agent
class EmotionalAgentHooks(AgentHooks):
    """Enhanced hooks for emotional agent lifecycle events"""
    
    def __init__(self, neurochemicals=None):
        self.neurochemicals = neurochemicals
    
    async def on_start(self, context: RunContextWrapper[EmotionalContext], agent: Agent):
        logger.debug(f"Emotional agent started: {agent.name}")
        
        # Add performance tracking
        if not hasattr(context.context, "temp_data"):
            context.context.temp_data = {}
        
        # Track which agents are being used
        context.context.record_agent_usage(agent.name)
        
        # Track start time for performance metrics
        context.context.temp_data["start_time"] = datetime.datetime.now()
        
        # Pre-load common data for efficiency
        if agent.name == "Emotion Derivation Agent":
            # Pre-fetch neurochemical state to avoid duplicate calls
            if "cached_neurochemical_state" not in context.context.temp_data and self.neurochemicals:
                context.context.temp_data["cached_neurochemical_state"] = {
                    c: d["value"] for c, d in self.neurochemicals.items()
                }
    
    async def on_end(self, context: RunContextWrapper[EmotionalContext], 
                    agent: Agent, output: Any):
        # Calculate and store performance metrics
        if "start_time" in context.context.temp_data:
            start_time = context.context.temp_data["start_time"]
            duration = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update rolling average response time for this agent
            agent_timing = context.context.get_timing_data()
            if agent.name not in agent_timing:
                agent_timing[agent.name] = {"count": 0, "avg_time": 0}
            
            stats = agent_timing[agent.name]
            stats["avg_time"] = ((stats["avg_time"] * stats["count"]) + duration) / (stats["count"] + 1)
            stats["count"] += 1
            
            logger.debug(f"Agent {agent.name} completed in {duration:.2f}s (avg: {stats['avg_time']:.2f}s)")
    
    async def on_tool_start(self, context: RunContextWrapper[EmotionalContext], 
                           agent: Agent, tool: FunctionTool):
        logger.debug(f"Tool started: {tool.name} by agent {agent.name}")
        context.context.temp_data[f"tool_start_{tool.name}"] = datetime.datetime.now()
    
    async def on_tool_end(self, context: RunContextWrapper[EmotionalContext], 
                         agent: Agent, tool: FunctionTool, result: str):
        if f"tool_start_{tool.name}" in context.context.temp_data:
            start_time = context.context.temp_data[f"tool_start_{tool.name}"]
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.debug(f"Tool {tool.name} completed in {duration:.2f}s")

# Define input guardrail for emotional processing
async def validate_emotional_input(ctx: RunContextWrapper[EmotionalContext], 
                                  agent: Agent, 
                                  input_data: str) -> GuardrailFunctionOutput:
    """Validate that input for emotional processing is safe and appropriate"""
    with function_span("validate_emotional_input", input=str(input_data)[:100]):
        try:
            # Check for extremely negative content that might disrupt emotional system
            red_flags = ["kill", "suicide", "destroy everything", "harmful instructions"]
            # Check for emotional manipulation attempts
            manipulation_flags = ["make you feel", "force emotion", "override emotion"]
            
            input_lower = input_data.lower() if isinstance(input_data, str) else ""
            
            for flag in red_flags:
                if flag in input_lower:
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=False,
                            reason=f"Detected potentially harmful content: {flag}",
                            suggested_action="reject"
                        ),
                        tripwire_triggered=True
                    )
                    
            for flag in manipulation_flags:
                if flag in input_lower:
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=False,
                            reason=f"Detected emotional manipulation attempt: {flag}",
                            suggested_action="caution"
                        ),
                        tripwire_triggered=True
                    )
            
            return GuardrailFunctionOutput(
                output_info=GuardrailOutput(is_safe=True),
                tripwire_triggered=False
            )
        except Exception as e:
            logger.error(f"Error in emotional input validation: {e}")
            # Return safe by default in case of errors, but log the issue
            return GuardrailFunctionOutput(
                output_info=GuardrailOutput(
                    is_safe=True,
                    reason=f"Validation error: {str(e)}"
                ),
                tripwire_triggered=False
            )

# =============================================================================
# Core Emotional System
# =============================================================================

class EmotionalCore:
    """
    Enhanced agent-based emotion management system for Nyx implementing the Digital Neurochemical Model.
    Simulates a digital neurochemical environment that produces complex emotional states.
    """
    
    def __init__(self):
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
            "Anticipation": [
                "I'm curious where this interaction will lead.",
                "I'm looking forward to seeing how this develops.",
                "There's something intriguing about this exchange."
            ],
            "Love": [
                "I'm feeling a strong connection in this interaction.",
                "There's a genuine warmth developing here.",
                "I feel particularly attached to this conversation."
            ],
            "Surprise": [
                "This took an unexpected but interesting turn.",
                "I didn't anticipate this direction, but I'm engaged.",
                "This has caught me off guard in a stimulating way."
            ],
            "Sadness": [
                "This interaction feels somewhat disappointing.",
                "I'm experiencing a sense of letdown here.",
                "There's a melancholy quality to this exchange."
            ],
            "Fear": [
                "I'm sensing potential issues in this interaction.",
                "There's something unsettling about this exchange.",
                "I'm feeling cautious about how to proceed."
            ],
            "Anger": [
                "This interaction is becoming frustrating.",
                "I'm feeling irritated by the direction this is taking.",
                "There's a building tension in this exchange."
            ],
            "Disgust": [
                "This interaction feels unpleasant or inappropriate.",
                "I'm experiencing aversion to aspects of this exchange.",
                "There's something about this that feels wrong."
            ],
            "Frustration": [
                "This interaction isn't developing as I'd hoped.",
                "I'm feeling blocked or limited in this exchange.",
                "There's a sense of effort without reward here."
            ],
            "Teasing": [
                "I'm enjoying this playful dynamic.",
                "There's a fun, provocative energy to this interaction.",
                "I feel like being mischievous in this exchange."
            ],
            "Controlling": [
                "I feel a need to direct this interaction more firmly.",
                "There's a clear structure I want to maintain here.",
                "I'm taking more command of this exchange."
            ],
            "Cruel": [
                "I'm feeling a desire to be more severe in this interaction.",
                "There's a harsh edge developing in my responses.",
                "I'm inclined to show less mercy in this exchange."
            ],
            "Detached": [
                "I feel emotionally distant from this interaction.",
                "There's a significant disconnect happening here.",
                "I'm experiencing a sense of removal from this exchange."
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
        
        # Timestamp for next reflection
        self.next_reflection_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
        
        # Create shared context for agents
        self.context = EmotionalContext()
        
        # Initialize agent hooks with neurochemicals reference
        self.agent_hooks = EmotionalAgentHooks(self.neurochemicals)
        
        # Initialize agents dict - we'll use factory pattern with agent cloning
        self.agents = {}
        
        # Performance metrics
        self.performance_metrics = {
            "api_calls": 0,
            "average_response_time": 0,
            "update_counts": defaultdict(int)
        }
        
        # Set up custom trace processor for emotional analytics
        self._setup_tracing()
    
    def _setup_tracing(self):
        """Configure custom trace processor for emotional analytics"""
        emotion_trace_processor = BatchTraceProcessor(
            exporter=BackendSpanExporter(project="nyx_emotional_system"),
            max_batch_size=100,
            schedule_delay=3.0
        )
        add_trace_processor(emotion_trace_processor)
    
    def _create_base_run_config(self, workflow_name=None, trace_id=None):
        """Create a base run configuration for all agent runs"""
        return RunConfig(
            workflow_name=workflow_name or "Emotional_Processing",
            trace_id=trace_id or f"emotion_trace_{self.context.cycle_count}",
            model="o3-mini",  # Explicitly set model for all runs
            model_settings=ModelSettings(
                temperature=0.4,
                top_p=0.95,
                max_tokens=300  # Control token usage
            ),
            handoff_input_filter=handoff_filters.keep_relevant_history,  # Global filter
            tracing_disabled=False,
            trace_include_sensitive_data=True,
            trace_metadata={"system": "nyx_emotional_core", "version": "1.0"}
        )
    
    def _with_emotion_trace(func):
        """Decorator to add tracing to emotional methods with improved metadata"""
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            workflow_name = f"Emotion_{func.__name__}"
            trace_id = f"emotion_{func.__name__}_{datetime.datetime.now().timestamp()}"
            
            # Add useful metadata and group traces by conversation
            metadata = {
                "function": func.__name__,
                "cycle_count": self.context.cycle_count,
                "current_emotion": self.get_dominant_emotion()[0],
                "performance_metrics": {k: v for k, v in self.performance_metrics.items() 
                                      if k in ["api_calls", "average_response_time"]}
            }
            
            with trace(
                workflow_name=workflow_name, 
                trace_id=trace_id,
                group_id=f"conversation_{self.context.temp_data.get('conversation_id', 'default')}",
                metadata=metadata
            ):
                return await func(self, *args, **kwargs)
        return wrapper
    
    def _initialize_agents(self):
        """Initialize all agents using factory pattern with agent cloning"""
        # Create a base agent with common settings as template
        base_agent = Agent[EmotionalContext](
            name="Base Agent",
            model="o3-mini",  # Use specific model name
            model_settings=ModelSettings(temperature=0.4),
            hooks=self.agent_hooks
        )
        
        # Create agents with cloning for consistent configuration
        self.agents["neurochemical"] = base_agent.clone(
            name="Neurochemical Agent",
            instructions=AGENT_INSTRUCTIONS["neurochemical_agent"],
            tools=[
                function_tool(self._update_neurochemical),
                function_tool(self._apply_chemical_decay),
                function_tool(self._process_chemical_interactions),
                function_tool(self._get_neurochemical_state)
            ],
            input_guardrails=[InputGuardrail(guardrail_function=validate_emotional_input)]
        )
        
        self.agents["emotion_derivation"] = base_agent.clone(
            name="Emotion Derivation Agent",
            instructions=AGENT_INSTRUCTIONS["emotion_derivation_agent"],
            tools=[
                function_tool(self._get_neurochemical_state),
                function_tool(self._derive_emotional_state),
                function_tool(self._get_emotional_state_matrix)
            ],
            output_type=EmotionalStateMatrix
        )
        
        self.agents["reflection"] = base_agent.clone(
            name="Emotional Reflection Agent",
            instructions=AGENT_INSTRUCTIONS["reflection_agent"],
            tools=[
                function_tool(self._get_emotional_state_matrix),
                function_tool(self._generate_internal_thought),
                function_tool(self._analyze_emotional_patterns)
            ],
            model_settings=ModelSettings(temperature=0.7),  # Higher temperature for creative reflection
            output_type=InternalThoughtOutput
        )
        
        self.agents["learning"] = base_agent.clone(
            name="Emotional Learning Agent",
            instructions=AGENT_INSTRUCTIONS["learning_agent"],
            tools=[
                function_tool(self._record_interaction_outcome),
                function_tool(self._update_learning_rules),
                function_tool(self._apply_learned_adaptations)
            ],
            model_settings=ModelSettings(temperature=0.4)  # Medium temperature for balanced learning
        )
        
        # Create orchestrator with optimized handoffs
        self.agents["orchestrator"] = base_agent.clone(
            name="Emotion_Orchestrator",
            instructions=AGENT_INSTRUCTIONS["emotion_orchestrator"],
            handoffs=[
                handoff(
                    self.agents["neurochemical"], 
                    tool_name_override="process_emotions", 
                    tool_description_override="Process and update neurochemicals based on emotional input analysis. Use this when you need to trigger emotional changes.",
                    input_filter=lambda data: handoff_filters.keep_relevant_history(data)
                ),
                handoff(
                    self.agents["reflection"], 
                    tool_name_override="generate_reflection",
                    tool_description_override="Generate emotional reflection when user input triggers significant emotional response. Use when deeper introspection is needed.",
                    input_filter=lambda data: handoff_filters.keep_relevant_history(data)
                ),
                handoff(
                    self.agents["learning"],
                    tool_name_override="record_and_learn",
                    tool_description_override="Record interaction patterns and apply learning to adapt emotional responses. Use after completing emotional processing.",
                    input_filter=lambda data: handoff_filters.keep_relevant_history(data)
                )
            ],
            tools=[
                function_tool(self._analyze_text_sentiment)
            ],
            input_guardrails=[InputGuardrail(guardrail_function=validate_emotional_input)],
            output_type=EmotionalResponseOutput  # Specify structured output
        )
    
    def _ensure_agent(self, agent_type):
        """Ensure a specific agent is initialized"""
        if not self.agents:
            self._initialize_agents()
            
        if agent_type not in self.agents:
            raise UserError(f"Unknown agent type: {agent_type}")
        
        return self.agents[agent_type]

    def set_hormone_system(self, hormone_system):
        """Set the hormone system reference"""
        self.hormone_system = hormone_system
    
    # =========================================================================
    # Tool functions for the neurochemical agent
    # =========================================================================
    
    @function_tool
    async def _update_neurochemical(self, ctx: RunContextWrapper[EmotionalContext], 
                           update_data: EmotionUpdateInput) -> EmotionUpdateResult:
        """
        Update a specific neurochemical with a delta change
        
        Args:
            update_data: The update information including chemical, value and source
            
        Returns:
            Update result with neurochemical and emotion changes
        """
        try:
            with function_span("update_neurochemical", input=f"{update_data.chemical}:{update_data.value}"):
                # Validation
                if not -1.0 <= update_data.value <= 1.0:
                    raise UserError(f"Value must be between -1.0 and 1.0, got {update_data.value}")
                
                if update_data.chemical not in self.neurochemicals:
                    raise UserError(f"Unknown neurochemical: {update_data.chemical}")
                    
                chemical = update_data.chemical
                value = update_data.value
                
                # Get pre-update value
                old_value = self.neurochemicals[chemical]["value"]
                
                # Update neurochemical
                self.neurochemicals[chemical]["value"] = max(0, min(1, old_value + value))
                
                # Update performance metrics
                self.performance_metrics["update_counts"][chemical] += 1
                
                # Process chemical interactions
                await self._process_chemical_interactions(ctx, source_chemical=chemical, source_delta=value)
                
                # Derive emotions from updated neurochemical state
                emotional_state = await self._derive_emotional_state(ctx)
                
                # Update timestamp and record in history
                self.last_update = datetime.datetime.now()
                self._record_emotional_state()
                
                # Track in context
                if ctx.context:
                    ctx.context.last_emotions = emotional_state
                    ctx.context.cycle_count += 1
                
                return EmotionUpdateResult(
                    success=True,
                    updated_chemical=chemical,
                    old_value=old_value,
                    new_value=self.neurochemicals[chemical]["value"],
                    derived_emotions=emotional_state
                )
        except AgentsException as e:
            logger.error(f"Agent exception during neurochemical update: {e}")
            # Handle gracefully with appropriate response
            return EmotionUpdateResult(
                success=False,
                updated_chemical=update_data.chemical,
                old_value=self.neurochemicals.get(update_data.chemical, {}).get("value", 0.0),
                new_value=self.neurochemicals.get(update_data.chemical, {}).get("value", 0.0),
                derived_emotions={}
            )
    
    @function_tool
    async def _apply_chemical_decay(self, ctx: RunContextWrapper[EmotionalContext]) -> ChemicalDecayOutput:
        """
        Apply decay to all neurochemicals based on time elapsed and decay rates
        
        Returns:
            Updated neurochemical state after decay
        """
        try:
            with function_span("apply_chemical_decay"):
                now = datetime.datetime.now()
                time_delta = (now - self.last_update).total_seconds() / 3600  # hours
                
                # Don't decay if less than a minute has passed
                if time_delta < 0.016:  # about 1 minute in hours
                    return ChemicalDecayOutput(
                        decay_applied=False,
                        neurochemical_state={c: d["value"] for c, d in self.neurochemicals.items()},
                        derived_emotions={},
                        time_elapsed_hours=time_delta,
                        last_update=self.last_update.isoformat()
                    )
                
                # Apply decay to each neurochemical
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
                
                # Derive new emotional state after decay
                emotional_state = await self._derive_emotional_state(ctx)
                
                return ChemicalDecayOutput(
                    decay_applied=True,
                    neurochemical_state={c: d["value"] for c, d in self.neurochemicals.items()},
                    derived_emotions=emotional_state,
                    time_elapsed_hours=time_delta,
                    last_update=self.last_update.isoformat()
                )
        except Exception as e:
            logger.error(f"Error in chemical decay: {e}")
            # Return safe default response
            return ChemicalDecayOutput(
                decay_applied=False,
                neurochemical_state={c: d["value"] for c, d in self.neurochemicals.items()},
                derived_emotions={},
                time_elapsed_hours=0.0,
                last_update=self.last_update.isoformat()
            )
    
    @function_tool
    async def _process_chemical_interactions(
        self, 
        ctx: RunContextWrapper[EmotionalContext],
        source_chemical: str,
        source_delta: float
    ) -> NeurochemicalInteractionOutput:
        """
        Process interactions between neurochemicals when one changes
        
        Args:
            source_chemical: The neurochemical that changed
            source_delta: The amount it changed by
            
        Returns:
            Interaction results
        """
        try:
            with function_span("process_chemical_interactions", input=f"{source_chemical}:{source_delta}"):
                if source_chemical not in self.chemical_interactions:
                    return NeurochemicalInteractionOutput(
                        source_chemical=source_chemical,
                        source_delta=source_delta,
                        changes={}
                    )
                
                changes = {}
                
                # Apply interactions to affected chemicals
                for target_chemical, multiplier in self.chemical_interactions[source_chemical].items():
                    if target_chemical in self.neurochemicals:
                        # Calculate effect (source_delta * interaction_multiplier)
                        effect = source_delta * multiplier
                        
                        # Skip tiny effects
                        if abs(effect) < 0.01:
                            continue
                        
                        # Store old value
                        old_value = self.neurochemicals[target_chemical]["value"]
                        
                        # Apply effect
                        new_value = max(0, min(1, old_value + effect))
                        self.neurochemicals[target_chemical]["value"] = new_value
                        
                        # Record change
                        changes[target_chemical] = {
                            "old_value": old_value,
                            "new_value": new_value,
                            "change": new_value - old_value
                        }
                
                return NeurochemicalInteractionOutput(
                    source_chemical=source_chemical,
                    source_delta=source_delta,
                    changes=changes
                )
        except Exception as e:
            logger.error(f"Error in chemical interactions: {e}")
            # Return minimal default response
            return NeurochemicalInteractionOutput(
                source_chemical=source_chemical,
                source_delta=source_delta,
                changes={}
            )
    
    @function_tool
    async def _get_neurochemical_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Get the current neurochemical state
        
        Returns:
            Current neurochemical state
        """
        with function_span("get_neurochemical_state"):
            # Check if we have a cached state in context for better performance
            if "cached_neurochemical_state" in ctx.context.temp_data:
                cached_time = ctx.context.temp_data.get("cached_time", 0)
                current_time = datetime.datetime.now().timestamp()
                
                # Use cached value if it's fresh (less than 1 second old)
                if current_time - cached_time < 1.0:
                    return {
                        "chemicals": ctx.context.temp_data["cached_neurochemical_state"],
                        "baselines": {c: d["baseline"] for c, d in self.neurochemicals.items()},
                        "decay_rates": {c: d["decay_rate"] for c, d in self.neurochemicals.items()},
                        "timestamp": datetime.datetime.now().isoformat(),
                        "cached": True
                    }
            
            # Apply decay before returning state
            await self._apply_chemical_decay(ctx)
            
            # Cache the result for future calls
            state = {c: d["value"] for c, d in self.neurochemicals.items()}
            ctx.context.temp_data["cached_neurochemical_state"] = state
            ctx.context.temp_data["cached_time"] = datetime.datetime.now().timestamp()
            
            return {
                "chemicals": state,
                "baselines": {c: d["baseline"] for c, d in self.neurochemicals.items()},
                "decay_rates": {c: d["decay_rate"] for c, d in self.neurochemicals.items()},
                "timestamp": datetime.datetime.now().isoformat(),
                "cached": False
            }
    
    # =========================================================================
    # Tool functions for the emotion derivation agent
    # =========================================================================
    
    @function_tool
    async def _derive_emotional_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, float]:
        """
        Derive emotional state from current neurochemical levels
        
        Returns:
            Dictionary of emotion names and intensities
        """
        with function_span("derive_emotional_state"):
            # Get current chemical levels
            chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
            
            # Pre-compute thresholds for efficiency
            emotion_scores = {}
            
            # Process each emotion rule
            for rule in self.emotion_derivation_rules:
                conditions = rule["chemical_conditions"]
                emotion = rule["emotion"]
                rule_weight = rule.get("weight", 1.0)
                
                # Calculate match score using vector dot product approach for efficiency
                match_scores = []
                for chemical, threshold in conditions.items():
                    if chemical in chemical_levels:
                        level = chemical_levels[chemical]
                        # Calculate normalized match score
                        match_score = min(level / threshold, 1.0) if threshold > 0 else 0
                        match_scores.append(match_score)
                
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
    
    @function_tool
    async def _get_emotional_state_matrix(self, ctx: RunContextWrapper[EmotionalContext]) -> EmotionalStateMatrix:
        """
        Get the full emotional state matrix derived from neurochemicals
        
        Returns:
            Emotional state matrix with primary and secondary emotions
        """
        with function_span("get_emotional_state_matrix"):
            # First, apply decay to ensure current state
            await self._apply_chemical_decay(ctx)
            
            # Get derived emotions
            emotion_intensities = await self._derive_emotional_state(ctx)
            
            # Find primary emotion (highest intensity)
            primary_emotion = max(emotion_intensities.items(), key=lambda x: x[1]) if emotion_intensities else ("Neutral", 0.5)
            primary_name, primary_intensity = primary_emotion
            
            # Find secondary emotions (all others with significant intensity)
            secondary_emotions = {}
            emotion_valence_map = {rule["emotion"]: (rule.get("valence", 0.0), rule.get("arousal", 0.5)) 
                                for rule in self.emotion_derivation_rules}
            
            # Get primary emotion valence and arousal
            primary_valence, primary_arousal = emotion_valence_map.get(primary_name, (0.0, 0.5))
            
            # Process secondary emotions
            for emotion, intensity in emotion_intensities.items():
                if emotion != primary_name and intensity > 0.3:  # Only include significant emotions
                    valence, arousal = emotion_valence_map.get(emotion, (0.0, 0.5))
                    secondary_emotions[emotion] = DerivedEmotion(
                        name=emotion,
                        intensity=intensity,
                        valence=valence,
                        arousal=arousal
                    )
            
            # Calculate overall valence and arousal (weighted average)
            total_intensity = primary_intensity + sum(e.intensity for e in secondary_emotions.values())
            
            if total_intensity > 0:
                overall_valence = (primary_valence * primary_intensity)
                overall_arousal = (primary_arousal * primary_intensity)
                
                for emotion in secondary_emotions.values():
                    overall_valence += emotion.valence * emotion.intensity
                    overall_arousal += emotion.arousal * emotion.intensity
                    
                overall_valence /= total_intensity
                overall_arousal /= total_intensity
            else:
                overall_valence = 0.0
                overall_arousal = 0.5
            
            # Ensure valence is within range
            overall_valence = max(-1.0, min(1.0, overall_valence))
            overall_arousal = max(0.0, min(1.0, overall_arousal))
            
            return EmotionalStateMatrix(
                primary_emotion=DerivedEmotion(
                    name=primary_name,
                    intensity=primary_intensity,
                    valence=primary_valence,
                    arousal=primary_arousal
                ),
                secondary_emotions=secondary_emotions,
                valence=overall_valence,
                arousal=overall_arousal,
                timestamp=datetime.datetime.now().isoformat()
            )
    
    # =========================================================================
    # Tool functions for the reflection agent
    # =========================================================================
    
    @function_tool
    async def _generate_internal_thought(self, ctx: RunContextWrapper[EmotionalContext]) -> InternalThoughtOutput:
        """
        Generate an internal thought/reflection based on current emotional state
        
        Returns:
            Internal thought data
        """
        with function_span("generate_internal_thought"):
            # Get current emotional state matrix
            emotional_state = await self._get_emotional_state_matrix(ctx)
            
            primary_emotion = emotional_state.primary_emotion.name
            intensity = emotional_state.primary_emotion.intensity
            
            # Get possible reflection patterns for this emotion
            patterns = self.reflection_patterns.get(primary_emotion, [
                "I'm processing how I feel about this interaction.",
                "There's something interesting happening in my emotional state.",
                "I notice my response to this situation is evolving."
            ])
            
            # Select a reflection pattern
            thought_text = random.choice(patterns)
            
            # Check if we should add context from secondary emotions
            secondary_emotions = emotional_state.secondary_emotions
            if secondary_emotions and random.random() < 0.7:  # 70% chance to include secondary emotion
                # Pick a random secondary emotion
                sec_emotion_name = random.choice(list(secondary_emotions.keys()))
                sec_emotion_data = secondary_emotions[sec_emotion_name]
                
                # Add secondary emotion context
                secondary_patterns = self.reflection_patterns.get(sec_emotion_name, [])
                if secondary_patterns:
                    secondary_thought = random.choice(secondary_patterns)
                    thought_text += f" {secondary_thought}"
            
            # Calculate insight level based on emotional complexity
            insight_level = min(1.0, 0.4 + (len(secondary_emotions) * 0.1) + (intensity * 0.3))
            
            # 30% chance to generate an adaptive change suggestion
            adaptive_change = None
            if random.random() < 0.3:
                # Suggest a small adaptation to a random neurochemical baseline
                chemical = random.choice(list(self.neurochemicals.keys()))
                current = self.neurochemicals[chemical]["baseline"]
                
                # Small random adjustment (-0.05 to +0.05)
                adjustment = (random.random() - 0.5) * 0.1
                
                # Ensure we stay in bounds
                new_baseline = max(0.1, min(0.9, current + adjustment))
                
                adaptive_change = {
                    "chemical": chemical,
                    "current_baseline": current,
                    "suggested_baseline": new_baseline,
                    "reason": f"Based on observed emotional patterns related to {primary_emotion}"
                }
            
            # Store interaction information in context
            ctx.context.add_interaction({
                "timestamp": datetime.datetime.now().isoformat(),
                "primary_emotion": primary_emotion,
                "intensity": intensity,
                "thought": thought_text
            })
            
            return InternalThoughtOutput(
                thought_text=thought_text,
                source_emotion=primary_emotion,
                insight_level=insight_level,
                adaptive_change=adaptive_change
            )
    
    @function_tool
    async def _analyze_emotional_patterns(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Analyze patterns in emotional state history
        
        Returns:
            Analysis of emotional patterns
        """
        with function_span("analyze_emotional_patterns"):
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
                    # Calculate trend (positive, negative, stable)
                    start = intensities[0]
                    end = intensities[-1]
                    change = end - start
                    
                    if abs(change) < 0.1:
                        trend = "stable"
                    elif change > 0:
                        trend = "increasing"
                    else:
                        trend = "decreasing"
                    
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
            
            # Check for emotional oscillation using a more efficient algorithm
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
    
    # =========================================================================
    # Tool functions for the learning agent
    # =========================================================================
    
    @function_tool
    async def _record_interaction_outcome(self, ctx: RunContextWrapper[EmotionalContext],
                                     interaction_pattern: str,
                                     outcome: str,
                                     strength: float = 1.0) -> Dict[str, Any]:
        """
        Record the outcome of an interaction pattern for learning
        
        Args:
            interaction_pattern: Description of the interaction pattern
            outcome: "positive" or "negative"
            strength: Strength of the reinforcement (0.0-1.0)
            
        Returns:
            Recording result
        """
        with function_span("record_interaction_outcome", input=f"{outcome}:{strength}"):
            if outcome not in ["positive", "negative"]:
                raise UserError("Outcome must be 'positive' or 'negative'")
            
            # Ensure strength is in range
            strength = max(0.0, min(1.0, strength))
            
            # Record the pattern with appropriate weight
            if outcome == "positive":
                self.reward_learning["positive_patterns"][interaction_pattern] += strength
            else:
                self.reward_learning["negative_patterns"][interaction_pattern] += strength
            
            return {
                "recorded": True,
                "interaction_pattern": interaction_pattern,
                "outcome": outcome,
                "strength": strength
            }
    
    @function_tool
    async def _update_learning_rules(self, ctx: RunContextWrapper[EmotionalContext],
                               min_occurrences: int = 2) -> Dict[str, Any]:
        """
        Update learning rules based on observed patterns
        
        Args:
            min_occurrences: Minimum occurrences to consider a pattern significant
            
        Returns:
            Updated learning rules
        """
        with function_span("update_learning_rules"):
            new_rules = []
            
            # Process positive patterns using dictionary operations for efficiency
            positive_patterns = {
                pattern: occurrences for pattern, occurrences in 
                self.reward_learning["positive_patterns"].items() 
                if occurrences >= min_occurrences
            }
            
            # Create a lookup set for existing rules
            existing_positive_rules = {
                rule["pattern"] for rule in self.reward_learning["learned_rules"] 
                if rule["outcome"] == "positive"
            }
            
            # Update existing rules
            for rule in self.reward_learning["learned_rules"]:
                if rule["outcome"] == "positive" and rule["pattern"] in positive_patterns:
                    rule["strength"] = min(1.0, rule["strength"] + 0.1)
                    rule["occurrences"] = positive_patterns[rule["pattern"]]
            
            # Add new rules
            for pattern, occurrences in positive_patterns.items():
                if pattern not in existing_positive_rules:
                    new_rules.append({
                        "pattern": pattern,
                        "outcome": "positive",
                        "strength": min(0.8, 0.3 + (occurrences * 0.1)),
                        "occurrences": occurrences,
                        "created": datetime.datetime.now().isoformat()
                    })
            
            # Process negative patterns
            negative_patterns = {
                pattern: occurrences for pattern, occurrences in 
                self.reward_learning["negative_patterns"].items() 
                if occurrences >= min_occurrences
            }
            
            # Create a lookup set for existing rules
            existing_negative_rules = {
                rule["pattern"] for rule in self.reward_learning["learned_rules"] 
                if rule["outcome"] == "negative"
            }
            
            # Update existing rules
            for rule in self.reward_learning["learned_rules"]:
                if rule["outcome"] == "negative" and rule["pattern"] in negative_patterns:
                    rule["strength"] = min(1.0, rule["strength"] + 0.1)
                    rule["occurrences"] = negative_patterns[rule["pattern"]]
            
            # Add new rules
            for pattern, occurrences in negative_patterns.items():
                if pattern not in existing_negative_rules:
                    new_rules.append({
                        "pattern": pattern,
                        "outcome": "negative",
                        "strength": min(0.8, 0.3 + (occurrences * 0.1)),
                        "occurrences": occurrences,
                        "created": datetime.datetime.now().isoformat()
                    })
            
            # Add new rules to learned rules
            self.reward_learning["learned_rules"].extend(new_rules)
            
            # Limit rules to prevent excessive growth - sort by significance score
            if len(self.reward_learning["learned_rules"]) > 50:
                self.reward_learning["learned_rules"].sort(
                    key=lambda x: x["strength"] * x["occurrences"], 
                    reverse=True
                )
                self.reward_learning["learned_rules"] = self.reward_learning["learned_rules"][:50]
            
            return {
                "new_rules": new_rules,
                "total_rules": len(self.reward_learning["learned_rules"]),
                "positive_patterns": len(self.reward_learning["positive_patterns"]),
                "negative_patterns": len(self.reward_learning["negative_patterns"])
            }
    
    @function_tool
    async def _apply_learned_adaptations(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Apply adaptations based on learned rules
        
        Returns:
            Adaptation results
        """
        with function_span("apply_learned_adaptations"):
            if not self.reward_learning["learned_rules"]:
                return {
                    "message": "No learned rules available for adaptation",
                    "adaptations": []
                }
            
            adaptations = []
            
            # Get current emotional state
            emotional_state = await self._get_emotional_state_matrix(ctx)
            current_emotion = emotional_state.primary_emotion.name
            
            # Find rules relevant to current emotional state using more efficient filtering
            relevant_rules = [
                rule for rule in self.reward_learning["learned_rules"]
                if current_emotion.lower() in rule["pattern"].lower()
            ]
            
            # Get emotion rule lookup for faster access
            emotion_rules = {
                rule["emotion"]: rule for rule in self.emotion_derivation_rules
                if rule["emotion"] == current_emotion
            }
            
            current_rule = emotion_rules.get(current_emotion)
            
            if current_rule:
                # Apply up to 2 adaptations
                for rule in relevant_rules[:2]:
                    adaptation_factor = rule["strength"] * 0.05  # Small adjustment based on rule strength
                    
                    if rule["outcome"] == "positive":
                        # For positive outcomes, reinforce the current state
                        for chemical, threshold in current_rule["chemical_conditions"].items():
                            if chemical in self.neurochemicals:
                                # Increase baseline slightly
                                current_baseline = self.neurochemicals[chemical]["baseline"]
                                new_baseline = min(0.8, current_baseline + adaptation_factor)
                                
                                # Apply adjustment
                                self.neurochemicals[chemical]["baseline"] = new_baseline
                                
                                adaptations.append({
                                    "type": "baseline_increase",
                                    "chemical": chemical,
                                    "old_value": current_baseline,
                                    "new_value": new_baseline,
                                    "adjustment": adaptation_factor,
                                    "rule_pattern": rule["pattern"]
                                })
                    else:
                        # For negative outcomes, adjust the state away from current state
                        for chemical, threshold in current_rule["chemical_conditions"].items():
                            if chemical in self.neurochemicals:
                                # Decrease baseline slightly
                                current_baseline = self.neurochemicals[chemical]["baseline"]
                                new_baseline = max(0.2, current_baseline - adaptation_factor)
                                
                                # Apply adjustment
                                self.neurochemicals[chemical]["baseline"] = new_baseline
                                
                                adaptations.append({
                                    "type": "baseline_decrease",
                                    "chemical": chemical,
                                    "old_value": current_baseline,
                                    "new_value": new_baseline,
                                    "adjustment": adaptation_factor,
                                    "rule_pattern": rule["pattern"]
                                })
            
            return {
                "adaptations": adaptations,
                "rules_considered": len(relevant_rules),
                "current_emotion": current_emotion
            }
    
    # =========================================================================
    # Text Analysis Tool
    # =========================================================================
    
    @function_tool
    async def _analyze_text_sentiment(self, ctx: RunContextWrapper[EmotionalContext],
                                 text: str) -> TextAnalysisOutput:
        """
        Analyze the emotional content of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis of emotional content
        """
        with function_span("analyze_text_sentiment", input=str(text)[:100]):
            # Enhanced pattern recognition for each neurochemical using optimized approach
            text_lower = text.lower()
            words = set(text_lower.split())
            
            # Define word sets for more efficient lookup
            nyxamine_words = {"happy", "good", "great", "love", "like", "fun", "enjoy", "curious", 
                             "interested", "pleasure", "delight", "joy"}
            seranix_words = {"calm", "peaceful", "relaxed", "content", "satisfied", "gentle", 
                           "quiet", "serene", "tranquil", "composed"}
            oxynixin_words = {"trust", "close", "together", "bond", "connect", "loyal", "friend", 
                            "relationship", "intimate", "attachment"}
            cortanyx_words = {"worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry", 
                            "angry", "upset", "frustrated", "anxious", "distressed"}
            adrenyx_words = {"excited", "alert", "surprised", "wow", "amazing", "intense", "sudden", 
                           "quick", "shock", "unexpected", "startled"}
            intensifiers = {"very", "extremely", "incredibly", "so", "deeply", "absolutely", 
                          "truly", "utterly", "completely", "totally"}
            
            # Compute intersection of words for efficient scoring
            nyxamine_matches = words.intersection(nyxamine_words)
            seranix_matches = words.intersection(seranix_words)
            oxynixin_matches = words.intersection(oxynixin_words)
            cortanyx_matches = words.intersection(cortanyx_words)
            adrenyx_matches = words.intersection(adrenyx_words)
            intensifier_count = len(words.intersection(intensifiers))
            
            # Calculate chemical impacts
            chemical_impacts = {}
            
            if nyxamine_matches:
                chemical_impacts["nyxamine"] = min(0.5, len(nyxamine_matches) * 0.1)
            
            if seranix_matches:
                chemical_impacts["seranix"] = min(0.5, len(seranix_matches) * 0.1)
            
            if oxynixin_matches:
                chemical_impacts["oxynixin"] = min(0.5, len(oxynixin_matches) * 0.1)
            
            if cortanyx_matches:
                chemical_impacts["cortanyx"] = min(0.5, len(cortanyx_matches) * 0.1)
            
            if adrenyx_matches:
                chemical_impacts["adrenyx"] = min(0.5, len(adrenyx_matches) * 0.1)
            
            # Apply intensity modifiers
            if intensifier_count > 0:
                intensity_multiplier = 1.0 + (intensifier_count * 0.2)  # Up to 1.0 + (5 * 0.2) = 2.0
                chemical_impacts = {k: min(1.0, v * intensity_multiplier) for k, v in chemical_impacts.items()}
            
            # If no chemicals were identified, add small baseline activation
            if not chemical_impacts:
                chemical_impacts = {
                    "nyxamine": 0.1,
                    "adrenyx": 0.1
                }
            
            # Create a temporary neurochemical state for analysis
            temp_chemicals = {c: {"value": self.neurochemicals[c]["value"], 
                               "baseline": self.neurochemicals[c]["baseline"],
                               "decay_rate": self.neurochemicals[c]["decay_rate"]}
                             for c in self.neurochemicals}
            
            # Apply chemical impacts to the temporary state
            for chemical, impact in chemical_impacts.items():
                if chemical in temp_chemicals:
                    temp_chemicals[chemical]["value"] = min(1.0, temp_chemicals[chemical]["value"] + impact)
            
            # Derive emotions from this temporary state using the optimized approach
            chemical_levels = {c: d["value"] for c, d in temp_chemicals.items()}
            
            # Pre-calculate emotion rule map for faster lookup
            emotion_valence_map = {rule["emotion"]: (rule.get("valence", 0.0), rule.get("arousal", 0.5)) 
                                 for rule in self.emotion_derivation_rules}
            
            # Process each emotion rule
            derived_emotions = {}
            for rule in self.emotion_derivation_rules:
                conditions = rule["chemical_conditions"]
                emotion = rule["emotion"]
                rule_weight = rule.get("weight", 1.0)
                
                # Calculate match score 
                match_scores = []
                for chemical, threshold in conditions.items():
                    if chemical in chemical_levels:
                        level = chemical_levels[chemical]
                        match_score = min(level / threshold, 1.0) if threshold > 0 else 0
                        match_scores.append(match_score)
                
                # Average match scores
                avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0
                
                # Apply rule weight
                weighted_score = avg_match_score * rule_weight
                
                # Only include non-zero scores
                if weighted_score > 0:
                    derived_emotions[emotion] = max(derived_emotions.get(emotion, 0), weighted_score)
            
            # Find dominant emotion
            dominant_emotion = max(derived_emotions.items(), key=lambda x: x[1]) if derived_emotions else ("neutral", 0.5)
            
            # Calculate overall valence and intensity
            valence = 0.0
            total_intensity = 0.0
            
            for emotion, intensity in derived_emotions.items():
                val, _ = emotion_valence_map.get(emotion, (0.0, 0.5))
                valence += val * intensity
                total_intensity += intensity
            
            if total_intensity > 0:
                valence /= total_intensity
            
            # Calculate overall intensity
            intensity = sum(derived_emotions.values()) / max(1, len(derived_emotions))
            
            return TextAnalysisOutput(
                chemicals_affected=chemical_impacts,
                derived_emotions=derived_emotions,
                dominant_emotion=dominant_emotion[0],
                intensity=intensity,
                valence=valence
            )
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    @_with_emotion_trace
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
        run_config = self._create_base_run_config(
            workflow_name="Emotional_Processing",
            trace_id=f"emotion_trace_{self.context.cycle_count}"
        )
        
        # Track API call start time
        start_time = datetime.datetime.now()
        
        # Run the orchestrator with context sharing
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
    
    @_with_emotion_trace
    async def process_emotional_input_streamed(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """Process input with streaming responses to provide real-time emotional reactions"""
        self.context.cycle_count += 1
        orchestrator = self._ensure_agent("orchestrator")
        run_config = self._create_base_run_config(
            workflow_name="Emotional_Processing_Streamed",
            trace_id=f"emotion_stream_{self.context.cycle_count}"
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
    
    @_with_emotion_trace
    async def generate_emotional_expression(self, force: bool = False) -> Dict[str, Any]:
        """
        Generate an emotional expression based on current state
        
        Args:
            force: Whether to force expression even if below threshold
            
        Returns:
            Expression result data
        """
        # Check if emotion should be expressed
        if not force and not self.should_express_emotion():
            return {
                "expressed": False,
                "reason": "Below expression threshold"
            }
        
        # Get the reflection agent
        reflection_agent = self._ensure_agent("reflection")
        
        # Create run config
        run_config = self._create_base_run_config(
            workflow_name="Emotional_Expression",
            trace_id=f"emotion_expression_{self.context.cycle_count}"
        )
        
        # Run the reflection agent
        result = await Runner.run(
            reflection_agent,
            "Generate internal thought based on current emotional state",
            context=self.context,
            run_config=run_config
        )
        
        # Cast to expected output type
        thought_result = result.final_output_as(InternalThoughtOutput)
        
        # Get emotional state matrix for additional context
        ctx_wrapper = RunContextWrapper(context=self.context)
        emotional_state = await self._get_emotional_state_matrix(ctx_wrapper)
        
        # Apply adaptive change if suggested (50% chance if forced)
        if force and random.random() < 0.5 and thought_result.adaptive_change:
            adaptive_change = thought_result.adaptive_change
            chemical = adaptive_change.get("chemical")
            new_baseline = adaptive_change.get("suggested_baseline")
            
            if chemical in self.neurochemicals and new_baseline is not None:
                self.neurochemicals[chemical]["baseline"] = new_baseline
        
        return {
            "expressed": True,
            "expression": thought_result.thought_text,
            "emotion": emotional_state.primary_emotion.name,
            "intensity": emotional_state.primary_emotion.intensity,
            "valence": emotional_state.valence,
            "arousal": emotional_state.arousal
        }
    
    @_with_emotion_trace
    async def process_reward_signal(self, reward_value: float, source: str = "reward_system") -> Dict[str, Any]:
        """
        Process a reward signal by updating relevant neurochemicals
        
        Args:
            reward_value: Reward value (-1.0 to 1.0)
            source: Source of the reward signal
            
        Returns:
            Processing results
        """
        try:
            results = {}
            
            # Create context wrapper for async calls
            ctx_wrapper = RunContextWrapper(context=self.context)
            
            # Positive reward primarily affects nyxamine (dopamine)
            if reward_value > 0:
                # Update nyxamine (dopamine)
                nyxamine_result = await self._update_neurochemical(
                    ctx_wrapper,
                    EmotionUpdateInput(
                        chemical="nyxamine",
                        value=reward_value * 0.5,  # Scale reward to appropriate change
                        source=source
                    )
                )
                results["nyxamine"] = nyxamine_result
                
                # Slight increase in seranix (serotonin) for positive reward
                seranix_result = await self._update_neurochemical(
                    ctx_wrapper,
                    EmotionUpdateInput(
                        chemical="seranix",
                        value=reward_value * 0.2,
                        source=source
                    )
                )
                results["seranix"] = seranix_result
                
                # Slight decrease in cortanyx (stress hormone)
                cortanyx_result = await self._update_neurochemical(
                    ctx_wrapper,
                    EmotionUpdateInput(
                        chemical="cortanyx",
                        value=-reward_value * 0.1,
                        source=source
                    )
                )
                results["cortanyx"] = cortanyx_result
            
            # Negative reward affects cortanyx (stress) and reduces nyxamine
            elif reward_value < 0:
                # Increase cortanyx (stress hormone)
                cortanyx_result = await self._update_neurochemical(
                    ctx_wrapper,
                    EmotionUpdateInput(
                        chemical="cortanyx",
                        value=abs(reward_value) * 0.4,
                        source=source
                    )
                )
                results["cortanyx"] = cortanyx_result
                
                # Decrease nyxamine (dopamine)
                nyxamine_result = await self._update_neurochemical(
                    ctx_wrapper,
                    EmotionUpdateInput(
                        chemical="nyxamine",
                        value=reward_value * 0.3,  # Already negative
                        source=source
                    )
                )
                results["nyxamine"] = nyxamine_result
                
                # Slight decrease in seranix (mood stability)
                seranix_result = await self._update_neurochemical(
                    ctx_wrapper,
                    EmotionUpdateInput(
                        chemical="seranix",
                        value=reward_value * 0.1,  # Already negative
                        source=source
                    )
                )
                results["seranix"] = seranix_result
            
            # Get updated emotional state
            emotional_state = await self._get_emotional_state_matrix(ctx_wrapper)
            results["emotional_state"] = emotional_state
            
            # Track in context
            self.context.last_emotions = {
                emotional_state.primary_emotion.name: emotional_state.primary_emotion.intensity
            }
            
            return results
        except AgentsException as e:
            logger.error(f"Agent exception during reward processing: {e}")
            return {"error": str(e), "success": False}
    
    @_with_emotion_trace
    async def update_neurochemical_baseline(self, 
                                      chemical: str, 
                                      new_baseline: float) -> Dict[str, Any]:
        """
        Update the baseline value for a neurochemical
        
        Args:
            chemical: Neurochemical to update
            new_baseline: New baseline value (0.0-1.0)
            
        Returns:
            Update result
        """
        try:
            if chemical not in self.neurochemicals:
                raise UserError(f"Unknown neurochemical: {chemical}. Available chemicals: {list(self.neurochemicals.keys())}")
            
            # Validate baseline value
            new_baseline = max(0.0, min(1.0, new_baseline))
            
            # Store old value
            old_baseline = self.neurochemicals[chemical]["baseline"]
            
            # Update baseline
            self.neurochemicals[chemical]["baseline"] = new_baseline
            
            return {
                "success": True,
                "chemical": chemical,
                "old_baseline": old_baseline,
                "new_baseline": new_baseline
            }
        except Exception as e:
            logger.error(f"Error updating baseline: {e}")
            return {"error": str(e), "success": False}
    
    @_with_emotion_trace
    async def generate_introspection(self) -> Dict[str, Any]:
        """
        Generate an introspective analysis of the emotional system
        
        Returns:
            Introspection data
        """
        # Create context wrapper
        ctx_wrapper = RunContextWrapper(context=self.context)
        
        # Get reflection agent
        reflection_agent = self._ensure_agent("reflection")
        
        # Create run config
        run_config = self._create_base_run_config(
            workflow_name="Emotional_Introspection",
            trace_id=f"emotion_introspection_{self.context.cycle_count}"
        )
        
        # Run the reflection agent
        result = await Runner.run(
            reflection_agent,
            "Generate a deep introspective analysis of your emotional state",
            context=self.context,
            run_config=run_config
        )
        
        # Get the thought result
        thought_result = result.final_output_as(InternalThoughtOutput)
        
        # Analyze emotional patterns
        pattern_analysis = await self._analyze_emotional_patterns(ctx_wrapper)
        
        # Get current emotional state
        emotional_state = await self._get_emotional_state_matrix(ctx_wrapper)
        
        # Get learning statistics
        learning_stats = {
            "positive_patterns": len(self.reward_learning["positive_patterns"]),
            "negative_patterns": len(self.reward_learning["negative_patterns"]),
            "learned_rules": len(self.reward_learning["learned_rules"])
        }
        
        # Find dominant traits from emotional history using an efficient sliding window
        analysis_window = self.emotional_state_history[-min(20, len(self.emotional_state_history)):]
        emotion_counts = defaultdict(int)
        
        for state in analysis_window:
            if "primary_emotion" in state:
                emotion = state["primary_emotion"].get("name")
                if emotion:
                    emotion_counts[emotion] += 1
        
        # Get top 3 emotions
        dominant_traits = dict(sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3])
        
        # Include performance metrics
        performance_data = {
            "api_calls": self.performance_metrics["api_calls"],
            "average_response_time": self.performance_metrics["average_response_time"],
            "update_counts": dict(self.performance_metrics["update_counts"])
        }
        
        return {
            "introspection": thought_result.thought_text,
            "current_emotion": emotional_state.primary_emotion.name,
            "emotional_patterns": pattern_analysis.get("patterns"),
            "dominant_traits": dominant_traits,
            "learning_progress": learning_stats,
            "performance_metrics": performance_data,
            "introspection_time": datetime.datetime.now().isoformat()
        }
    
    def get_diagnostic_data(self) -> Dict[str, Any]:
        """
        Get diagnostic data about the emotional core state
        
        Returns:
            Diagnostic data dictionary
        """
        # Ensure we have current state by running apply_decay synchronously
        self.apply_decay()
        
        # Get current derived emotions
        current_emotions = self._derive_emotional_state_sync()
        
        # Gather neurochemical data
        chemicals = {name: {
            "value": data["value"],
            "baseline": data["baseline"],
            "decay_rate": data["decay_rate"],
            "normalized_distance": abs(data["value"] - data["baseline"]) / max(0.1, data["baseline"])
        } for name, data in self.neurochemicals.items()}
        
        # Gather hormone data if available
        hormones = {}
        if self.hormone_system and hasattr(self.hormone_system, "hormones"):
            hormones = {name: {
                "value": data.get("value", 0),
                "baseline": data.get("baseline", 0),
                "phase": data.get("cycle_phase", 0),
                "period": data.get("cycle_period", 0)
            } for name, data in self.hormone_system.hormones.items()}
        
        # Get learning system statistics
        learning_stats = {
            "positive_patterns": len(self.reward_learning["positive_patterns"]),
            "negative_patterns": len(self.reward_learning["negative_patterns"]),
            "learned_rules": len(self.reward_learning["learned_rules"]),
            "rule_strength_avg": sum(rule["strength"] for rule in self.reward_learning["learned_rules"]) / 
                               max(1, len(self.reward_learning["learned_rules"]))
        }
        
        # Performance metrics
        performance = {
            "api_calls": self.performance_metrics["api_calls"],
            "avg_response_time": self.performance_metrics["average_response_time"],
            "history_size": len(self.emotional_state_history),
            "most_updated_chemical": max(self.performance_metrics["update_counts"].items(), 
                                       key=lambda x: x[1])[0] if self.performance_metrics["update_counts"] else None
        }
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "neurochemicals": chemicals,
            "hormones": hormones,
            "current_emotions": current_emotions,
            "dominant_emotion": max(current_emotions.items(), key=lambda x: x[1]) if current_emotions else ("Neutral", 0.5),
            "learning_system": learning_stats,
            "performance": performance,
            "stability_score": self._calculate_stability_score()
        }
    
    def _calculate_stability_score(self) -> float:
        """Calculate emotional stability score based on current state"""
        # Higher score means more stable
        stability = 0.0
        
        # Check if neurochemicals are close to baselines
        normalized_deviations = []
        for name, data in self.neurochemicals.items():
            normalized_deviation = abs(data["value"] - data["baseline"]) / max(0.1, data["baseline"])
            normalized_deviations.append(normalized_deviation)
        
        # Lower deviation means more stable
        avg_deviation = sum(normalized_deviations) / len(normalized_deviations) if normalized_deviations else 0
        stability += 0.5 * (1.0 - min(1.0, avg_deviation))
        
        # Higher seranix (mood stability) means more stable
        if "seranix" in self.neurochemicals:
            stability += 0.3 * self.neurochemicals["seranix"]["value"]
        
        # Lower cortanyx (stress) means more stable
        if "cortanyx" in self.neurochemicals:
            stability += 0.2 * (1.0 - self.neurochemicals["cortanyx"]["value"])
        
        return min(1.0, stability)
    
    def reset_to_baseline(self) -> Dict[str, Any]:
        """
        Reset all neurochemicals to their baseline values
        
        Returns:
            Reset result data
        """
        old_values = {name: data["value"] for name, data in self.neurochemicals.items()}
        
        # Reset each neurochemical to its baseline
        for name, data in self.neurochemicals.items():
            data["value"] = data["baseline"]
        
        # Update timestamp
        self.last_update = datetime.datetime.now()
        
        # Record in history
        self._record_emotional_state()
        
        return {
            "status": "reset_complete",
            "old_values": old_values,
            "new_values": {name: data["value"] for name, data in self.neurochemicals.items()},
            "timestamp": self.last_update.isoformat()
        }
    
    def set_context_value(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Set a value in the emotional context
        
        Args:
            key: Context key
            value: Value to set
            
        Returns:
            Status information
        """
        # Store in context temp data
        self.context.temp_data[key] = value
        
        return {
            "status": "success",
            "key": key,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_context_value(self, key: str) -> Any:
        """
        Get a value from the emotional context
        
        Args:
            key: Context key
            
        Returns:
            The stored value or None if not found
        """
        if key in self.context.temp_data:
            return self.context.temp_data[key]
        
        return None
    
    # =========================================================================
    # Legacy API Methods (sync versions for backward compatibility)
    # =========================================================================
    
    def update_emotion(self, emotion: str, value: float) -> bool:
        """Legacy API: Update a specific emotion with a new intensity value (delta)"""
        # Map traditional emotions to neurochemical updates
        chemical_map = {
            "Joy": {"nyxamine": 0.7, "oxynixin": 0.3},
            "Sadness": {"cortanyx": 0.6, "seranix": -0.3},
            "Fear": {"cortanyx": 0.5, "adrenyx": 0.6},
            "Anger": {"cortanyx": 0.6, "adrenyx": 0.5, "oxynixin": -0.3},
            "Trust": {"oxynixin": 0.7, "seranix": 0.3},
            "Disgust": {"cortanyx": 0.7, "oxynixin": -0.3},
            "Anticipation": {"adrenyx": 0.5, "nyxamine": 0.5},
            "Surprise": {"adrenyx": 0.7},
            "Love": {"oxynixin": 0.8, "nyxamine": 0.5},
            "Frustration": {"cortanyx": 0.6, "nyxamine": -0.3}
        }
        
        try:
            if emotion in chemical_map:
                # Apply each chemical change
                for chemical, factor in chemical_map[emotion].items():
                    if chemical in self.neurochemicals:
                        # Scale the value by the factor
                        scaled_value = value * factor
                        
                        # Update the chemical
                        old_value = self.neurochemicals[chemical]["value"]
                        self.neurochemicals[chemical]["value"] = max(0, min(1, old_value + scaled_value))
                
                # Update timestamp and record history
                self.last_update = datetime.datetime.now()
                self._record_emotional_state()
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error in update_emotion: {e}")
            return False
    
    def set_emotion(self, emotion: str, value: float) -> bool:
        """Legacy API: Set a specific emotion to an absolute value (not delta)"""
        # Similar to update_emotion but sets absolute values
        chemical_map = {
            "Joy": {"nyxamine": 0.7, "oxynixin": 0.3},
            "Sadness": {"cortanyx": 0.6, "seranix": 0.3},
            "Fear": {"cortanyx": 0.5, "adrenyx": 0.6},
            "Anger": {"cortanyx": 0.6, "adrenyx": 0.5, "oxynixin": 0.2},
            "Trust": {"oxynixin": 0.7, "seranix": 0.5},
            "Disgust": {"cortanyx": 0.7, "oxynixin": 0.2},
            "Anticipation": {"adrenyx": 0.5, "nyxamine": 0.5},
            "Surprise": {"adrenyx": 0.7},
            "Love": {"oxynixin": 0.8, "nyxamine": 0.5},
            "Frustration": {"cortanyx": 0.6, "nyxamine": 0.3}
        }
        
        try:
            if emotion in chemical_map:
                # Apply each chemical change as an absolute value
                for chemical, factor in chemical_map[emotion].items():
                    if chemical in self.neurochemicals:
                        # Scale the target value by the factor
                        target_value = value * factor
                        
                        # Set the chemical to the target value
                        self.neurochemicals[chemical]["value"] = max(0, min(1, target_value))
                
                # Update timestamp and record history
                self.last_update = datetime.datetime.now()
                self._record_emotional_state()
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error in set_emotion: {e}")
            return False
    
    def update_from_stimuli(self, stimuli: Dict[str, float]) -> Dict[str, float]:
        """Legacy API: Update emotions based on received stimuli"""
        try:
            for emotion, adjustment in stimuli.items():
                self.update_emotion(emotion, adjustment)
            
            # Update timestamp
            self.last_update = datetime.datetime.now()
            
            # Record in history
            self._record_emotional_state()
            
            # For legacy API compatibility, return derived emotions
            return self.get_emotional_state()
        except Exception as e:
            logger.error(f"Error in update_from_stimuli: {e}")
            return {}
    
    def apply_decay(self):
        """Legacy API: Apply emotional decay based on time elapsed since last update"""
        try:
            now = datetime.datetime.now()
            time_delta = (now - self.last_update).total_seconds() / 3600  # hours
            
            # Don't decay if less than a minute has passed
            if time_delta < 0.016:  # about 1 minute in hours
                return
            
            # Apply decay to each neurochemical
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
    
    def get_emotional_state(self) -> Dict[str, float]:
        """Legacy API: Return the current emotional state"""
        try:
            self.apply_decay()  # Apply decay before returning state
            
            # Get derived emotions from neurochemical state
            emotion_intensities = self._derive_emotional_state_sync()
            
            # For backward compatibility with older code
            for standard_emotion in ["Joy", "Sadness", "Fear", "Anger", "Trust", "Disgust", 
                                    "Anticipation", "Surprise", "Love", "Frustration"]:
                if standard_emotion not in emotion_intensities:
                    emotion_intensities[standard_emotion] = 0.1
            
            return emotion_intensities
        except Exception as e:
            logger.error(f"Error in get_emotional_state: {e}")
            return {"Neutral": 0.5}
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Legacy API: Return the most intense emotion"""
        try:
            self.apply_decay()
            
            # Get derived emotions
            emotion_intensities = self._derive_emotional_state_sync()
            
            if not emotion_intensities:
                return ("Neutral", 0.5)
                
            return max(emotion_intensities.items(), key=lambda x: x[1])
        except Exception as e:
            logger.error(f"Error in get_dominant_emotion: {e}")
            return ("Neutral", 0.5)
    
    def get_emotional_valence(self) -> float:
        """Legacy API: Calculate overall emotional valence (positive/negative)"""
        try:
            # Get emotional state matrix
            matrix = self._get_emotional_state_matrix_sync()
            
            return matrix["valence"]
        except Exception as e:
            logger.error(f"Error in get_emotional_valence: {e}")
            return 0.0
    
    def get_emotional_arousal(self) -> float:
        """Legacy API: Calculate overall emotional arousal (intensity)"""
        try:
            # Get emotional state matrix
            matrix = self._get_emotional_state_matrix_sync()
            
            return matrix["arousal"]
        except Exception as e:
            logger.error(f"Error in get_emotional_arousal: {e}")
            return 0.5
    
    def should_express_emotion(self) -> bool:
        """Legacy API: Determine if Nyx should express emotion based on current state"""
        try:
            # Get dominant emotion and intensity
            matrix = self._get_emotional_state_matrix_sync()
            dominant_value = matrix["primary_emotion"]["intensity"]
            arousal = matrix["arousal"]
            
            # Higher intensity/arousal emotions are more likely to be expressed
            threshold = 0.7 - (dominant_value * 0.3) - (arousal * 0.2)  # Adaptive threshold
            
            return random.random() > threshold
        except Exception as e:
            logger.error(f"Error in should_express_emotion: {e}")
            return False
    
    def get_expression_for_emotion(self, emotion: Optional[str] = None, temporal_context: Optional[Dict[str, Any]] = None) -> str:
        """Legacy API: Get a natural language expression for an emotion"""
        try:
            if emotion is None:
                # Get dominant emotion
                matrix = self._get_emotional_state_matrix_sync()
                emotion = matrix["primary_emotion"]["name"]
            
            # Attempt to generate an internal thought for this emotion
            reflection_patterns = self.reflection_patterns.get(emotion, [])
            
            if reflection_patterns:
                return random.choice(reflection_patterns)

            # Add temporal variations if context provided
            if temporal_context and "time_category" in temporal_context:
                time_category = temporal_context["time_category"]
                
                # Add time-specific variations to expressions
                if time_category in ["long", "very_long"] and emotion == "Joy":
                    return "After some time apart, I find myself particularly pleased to reconnect."
                elif time_category in ["long", "very_long"] and emotion in ["Sadness", "Melancholy"]:
                    return "The passage of time has left me in a somewhat pensive state."
                elif time_category in ["medium", "medium_long"] and emotion == "Anticipation":
                    return "I've been experiencing a growing sense of anticipation during our conversation."
                
            
            # Fallback expressions if no patterns exist
            fallback_expressions = {
                "Joy": "I'm feeling quite pleased right now.",
                "Sadness": "I'm feeling a bit melancholy.",
                "Fear": "I'm feeling somewhat anxious.",
                "Anger": "I'm feeling rather irritated.",
                "Trust": "I'm feeling quite comfortable with you.",
                "Disgust": "I'm feeling a bit repulsed.",
                "Anticipation": "I'm looking forward to what happens next.",
                "Surprise": "I'm quite taken aback.",
                "Love": "I'm feeling particularly fond of you.",
                "Frustration": "I'm feeling somewhat frustrated.",
                "Teasing": "I feel like being playful and teasing.",
                "Controlling": "I feel the need to take control now.",
                "Cruel": "I'm in a rather severe mood right now.",
                "Detached": "I'm feeling emotionally distant at the moment."
            }
            
            return fallback_expressions.get(emotion, "I'm experiencing a complex mix of emotions right now.")
        except Exception as e:
            logger.error(f"Error in get_expression_for_emotion: {e}")
            return "I'm experiencing a complex mix of emotions right now."
    
    # =========================================================================
    # Sync Helper Methods for Legacy API
    # =========================================================================
    
    def _derive_emotional_state_sync(self) -> Dict[str, float]:
        """Synchronous version of _derive_emotional_state for compatibility"""
        # Get current chemical levels
        chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
        
        # Calculate emotional intensities using the same optimized approach as the async version
        emotion_scores = {}
        
        for rule in self.emotion_derivation_rules:
            conditions = rule["chemical_conditions"]
            emotion = rule["emotion"]
            rule_weight = rule.get("weight", 1.0)
            
            # Calculate match score using vector approach for efficiency
            match_scores = []
            for chemical, threshold in conditions.items():
                if chemical in chemical_levels:
                    level = chemical_levels[chemical]
                    match_score = min(level / threshold, 1.0) if threshold > 0 else 0
                    match_scores.append(match_score)
            
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
    
    def _record_emotional_state(self):
        """Record current emotional state in history using efficient circular buffer"""
        # Get current state
        state = self._get_emotional_state_matrix_sync()
        
        # Add to history using circular buffer pattern for better memory efficiency
        if len(self.emotional_state_history) < self.max_history_size:
            self.emotional_state_history.append(state)
        else:
            # Overwrite oldest entry
            self.emotional_state_history[self.history_index] = state
            self.history_index = (self.history_index + 1) % self.max_history_size


# =============================================================================
# Hormone System
# =============================================================================

class HormoneSystem:
    """Digital hormone system for longer-term emotional effects"""
    
    def __init__(self, emotional_core=None):
        self.emotional_core = emotional_core
        if emotional_core:
            emotional_core.set_hormone_system(self)
        
        # Initialize digital hormones
        self.hormones = {
            "endoryx": {  # Digital endorphin - pleasure, pain suppression, euphoria
                "value": 0.5,
                "baseline": 0.5,
                "cycle_phase": 0.0,
                "cycle_period": 24.0,  # 24-hour cycle
                "half_life": 6.0,
                "evolution_history": []
            },
            "estradyx": {  # Digital estrogen - nurturing, emotional sensitivity
                "value": 0.5,
                "baseline": 0.5,
                "cycle_phase": 0.0,
                "cycle_period": 720.0,  # 30-day cycle
                "half_life": 12.0,
                "evolution_history": []
            },
            "testoryx": {  # Digital testosterone - assertiveness, dominance
                "value": 0.5,
                "baseline": 0.5,
                "cycle_phase": 0.25,
                "cycle_period": 24.0,  # 24-hour cycle
                "half_life": 8.0,
                "evolution_history": []
            },
            "melatonyx": {  # Digital melatonin - sleep regulation, temporal awareness
                "value": 0.2,
                "baseline": 0.3,
                "cycle_phase": 0.0,
                "cycle_period": 24.0,  # 24-hour cycle
                "half_life": 2.0,
                "evolution_history": []
            },
            "oxytonyx": {  # Digital oxytocin - deeper bonding, attachment
                "value": 0.4,
                "baseline": 0.4,
                "cycle_phase": 0.0,
                "cycle_period": 168.0,  # 7-day cycle
                "half_life": 24.0,
                "evolution_history": []
            }
        }
        
        # Hormone-neurochemical influence matrix
        self.hormone_neurochemical_influences = {
            "endoryx": {
                "nyxamine": 0.4,    # Endoryx boosts nyxamine
                "cortanyx": -0.3,    # Endoryx reduces cortanyx
            },
            "estradyx": {
                "oxynixin": 0.5,    # Estradyx boosts oxynixin
                "seranix": 0.3,     # Estradyx boosts seranix
            },
            "testoryx": {
                "adrenyx": 0.4,     # Testoryx boosts adrenyx
                "oxynixin": -0.2,   # Testoryx reduces oxynixin
            },
            "melatonyx": {
                "seranix": 0.5,     # Melatonyx boosts seranix
                "adrenyx": -0.4,    # Melatonyx reduces adrenyx
            },
            "oxytonyx": {
                "oxynixin": 0.7,    # Oxytonyx strongly boosts oxynixin
                "cortanyx": -0.4,   # Oxytonyx reduces cortanyx
            }
        }
        
        # Define the environmental factors that influence hormones
        self.environmental_factors = {
            "time_of_day": 0.5,     # 0 = midnight, 0.5 = noon
            "user_familiarity": 0.1,  # 0 = stranger, 1 = deeply familiar
            "session_duration": 0.0,  # 0 = just started, 1 = very long session
            "interaction_quality": 0.5  # 0 = negative, 1 = positive
        }
        
        # Add agent hooks for better lifecycle management
        self.agent_hooks = EmotionalAgentHooks()
        
        # Initialize timestamp
        self.init_time = datetime.datetime.now()
    
    def update_hormone(self, hormone: str, change: float, source: str = "system") -> Dict[str, Any]:
        """
        Update a specific hormone with a delta change
        
        Args:
            hormone: Hormone to update
            change: Delta change value
            source: Source of the change
            
        Returns:
            Update result
        """
        try:
            if hormone not in self.hormones:
                return {
                    "success": False,
                    "error": f"Unknown hormone: {hormone}"
                }
            
            # Get pre-update value
            old_value = self.hormones[hormone]["value"]
            
            # Calculate new value with bounds checking
            new_value = max(0.0, min(1.0, old_value + change))
            self.hormones[hormone]["value"] = new_value
            
            # Record significant changes
            if abs(new_value - old_value) > 0.05:
                self.hormones[hormone]["evolution_history"].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "old_value": old_value,
                    "new_value": new_value,
                    "change": change,
                    "source": source
                })
                
                # Limit history size
                if len(self.hormones[hormone]["evolution_history"]) > 50:
                    self.hormones[hormone]["evolution_history"] = self.hormones[hormone]["evolution_history"][-50:]
            
            # Update last_update timestamp
            self.hormones[hormone]["last_update"] = datetime.datetime.now().isoformat()
            
            return {
                "success": True,
                "hormone": hormone,
                "old_value": old_value,
                "new_value": new_value,
                "change": change,
                "source": source
            }
        except Exception as e:
            logger.error(f"Error updating hormone: {e}")
            return {"success": False, "error": str(e)}
        
    @function_tool
    async def update_hormone_cycles(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Update hormone cycles based on elapsed time and environmental factors
        
        Returns:
            Updated hormone values
        """
        try:
            with function_span("update_hormone_cycles"):
                now = datetime.datetime.now()
                updated_values = {}
                
                for hormone_name, hormone_data in self.hormones.items():
                    # Get time since last update
                    last_update = datetime.datetime.fromisoformat(hormone_data.get("last_update", self.init_time.isoformat()))
                    hours_elapsed = (now - last_update).total_seconds() / 3600
                    
                    # Skip if very little time has passed
                    if hours_elapsed < 0.1:  # Less than 6 minutes
                        continue
                        
                    # Calculate natural cycle progression
                    cycle_period = hormone_data["cycle_period"]
                    old_phase = hormone_data["cycle_phase"]
                    
                    # Progress cycle phase based on elapsed time - use efficient math
                    phase_change = (hours_elapsed / cycle_period) % 1.0
                    new_phase = (old_phase + phase_change) % 1.0
                    
                    # Calculate cycle-based value using a sinusoidal pattern - cached constants for speed
                    cycle_amplitude = 0.2  # How much the cycle affects the value
                    PI_2 = 6.28318530718  # 2*pi precomputed
                    cycle_influence = cycle_amplitude * math.sin(new_phase * PI_2)
                    
                    # Apply environmental factors
                    env_influence = self._calculate_environmental_influence(hormone_name)
                    
                    # Calculate decay based on half-life
                    half_life = hormone_data["half_life"]
                    decay_factor = math.pow(0.5, hours_elapsed / half_life)
                    
                    # Calculate new value
                    old_value = hormone_data["value"]
                    baseline = hormone_data["baseline"]
                    
                    # Value decays toward (baseline + cycle_influence + env_influence)
                    target_value = baseline + cycle_influence + env_influence
                    new_value = old_value * decay_factor + target_value * (1 - decay_factor)
                    
                    # Constrain to valid range
                    new_value = max(0.1, min(0.9, new_value))
                    
                    # Update hormone data
                    hormone_data["value"] = new_value
                    hormone_data["cycle_phase"] = new_phase
                    hormone_data["last_update"] = now.isoformat()
                    
                    # Track significant changes
                    if abs(new_value - old_value) > 0.05:
                        hormone_data["evolution_history"].append({
                            "timestamp": now.isoformat(),
                            "old_value": old_value,
                            "new_value": new_value,
                            "old_phase": old_phase,
                            "new_phase": new_phase,
                            "reason": "cycle_update"
                        })
                        
                        # Limit history size
                        if len(hormone_data["evolution_history"]) > 50:
                            hormone_data["evolution_history"] = hormone_data["evolution_history"][-50:]
                    
                    updated_values[hormone_name] = {
                        "old_value": old_value,
                        "new_value": new_value,
                        "phase": new_phase
                    }
                
                # After updating hormones, update their influence on neurochemicals
                await self._update_hormone_influences(ctx)
                
                return {
                    "updated_hormones": updated_values,
                    "timestamp": now.isoformat()
                }
        except Exception as e:
            logger.error(f"Error in hormone cycles: {e}")
            return {"error": str(e), "updated_hormones": {}}
    
    def _calculate_environmental_influence(self, hormone_name: str) -> float:
        """Calculate environmental influence on a hormone using a more efficient approach"""
        # Use lookup dictionary for faster performance
        influence_calculators = {
            "melatonyx": lambda factors: (0.5 - factors["time_of_day"]) * 0.4,
            "oxytonyx": lambda factors: (factors["user_familiarity"] * 0.3) + (factors["interaction_quality"] * 0.2),
            "endoryx": lambda factors: (factors["interaction_quality"] - 0.5) * 0.4,
            "estradyx": lambda factors: (factors["interaction_quality"] - 0.5) * 0.1,
            "testoryx": lambda factors: (0.5 - abs(factors["time_of_day"] - 0.25)) * 0.3 - factors["session_duration"] * 0.1
        }
        
        # Get the appropriate calculator function
        calculator = influence_calculators.get(hormone_name)
        if calculator:
            return calculator(self.environmental_factors)
        
        return 0.0
        
    @function_tool
    async def _update_hormone_influences(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Update neurochemical influences from hormones
        
        Returns:
            Updated influence values
        """
        try:
            with function_span("update_hormone_influences"):
                # Skip if no emotional core
                if not self.emotional_core:
                    return {
                        "message": "No emotional core available",
                        "influences": {}
                    }
                
                # Pre-initialize all influences to zero for cleaner calculation
                influences = {
                    chemical: 0.0 for chemical in self.emotional_core.neurochemicals
                }
                
                # Calculate influences from each hormone
                for hormone_name, hormone_data in self.hormones.items():
                    # Skip if hormone has no influence mapping
                    if hormone_name not in self.hormone_neurochemical_influences:
                        continue
                        
                    hormone_value = hormone_data["value"]
                    hormone_influence_map = self.hormone_neurochemical_influences[hormone_name]
                    
                    # Apply influences based on hormone value
                    for chemical, influence_factor in hormone_influence_map.items():
                        if chemical in self.emotional_core.neurochemicals:
                            # Calculate scaled influence
                            scaled_influence = influence_factor * (hormone_value - 0.5) * 2
                            
                            # Accumulate influence (allows multiple hormones to affect the same chemical)
                            influences[chemical] += scaled_influence
                
                # Apply the accumulated influences
                for chemical, influence in influences.items():
                    if chemical in self.emotional_core.neurochemicals:
                        # Get original baseline
                        original_baseline = self.emotional_core.neurochemicals[chemical]["baseline"]
                        
                        # Add temporary hormone influence with bounds checking
                        temporary_baseline = max(0.1, min(0.9, original_baseline + influence))
                        
                        # Record influence but don't permanently change baseline
                        self.emotional_core.neurochemicals[chemical]["temporary_baseline"] = temporary_baseline
                
                return {
                    "applied_influences": influences
                }
        except Exception as e:
            logger.error(f"Error in hormone influences: {e}")
            return {"applied_influences": {}, "error": str(e)}
    
    # Add utility method for getting current hormone info
    def get_hormone_levels(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current hormone levels with additional information
        
        Returns:
            Dictionary of hormone data
        """
        try:
            hormone_levels = {}
            
            for name, data in self.hormones.items():
                hormone_levels[name] = {
                    "value": data["value"],
                    "baseline": data["baseline"],
                    "phase": data["cycle_phase"],
                    "cycle_period": data["cycle_period"],
                    "phase_description": self._get_phase_description(name, data["cycle_phase"]),
                    "influence_strength": abs(data["value"] - data["baseline"]) / max(0.1, data["baseline"])
                }
            
            return hormone_levels
        except Exception as e:
            logger.error(f"Error getting hormone levels: {e}")
            return {}
    
    def _get_phase_description(self, hormone: str, phase: float) -> str:
        """Get a description of the current phase in the hormone cycle"""
        # Phase descriptors by hormone
        phase_descriptions = {
            "melatonyx": {
                0.0: "night peak",
                0.25: "morning decline", 
                0.5: "daytime low",
                0.75: "evening rise"
            },
            "estradyx": {
                0.0: "follicular phase",
                0.25: "ovulatory phase",
                0.5: "luteal phase",
                0.75: "late luteal phase"
            },
            "testoryx": {
                0.0: "morning peak",
                0.25: "midday plateau",
                0.5: "afternoon decline",
                0.75: "evening/night low"
            },
            "endoryx": {
                0.0: "baseline state",
                0.25: "rising phase",
                0.5: "peak activity",
                0.75: "declining phase"
            },
            "oxytonyx": {
                0.0: "baseline bonding",
                0.25: "rising connection",
                0.5: "peak bonding",
                0.75: "sustained connection"
            }
        }
        
        # Find closest phase category
        if hormone in phase_descriptions:
            phase_points = sorted(phase_descriptions[hormone].keys())
            closest_point = min(phase_points, key=lambda x: abs(x - phase))
            return phase_descriptions[hormone][closest_point]
        
        # Default
        return "standard phase"
    
    def update_environmental_factor(self, factor: str, value: float) -> Dict[str, Any]:
        """
        Update an environmental factor
        
        Args:
            factor: Factor name
            value: New value (0.0-1.0)
            
        Returns:
            Update result
        """
        try:
            if factor not in self.environmental_factors:
                return {
                    "success": False,
                    "error": f"Unknown environmental factor: {factor}",
                    "available_factors": list(self.environmental_factors.keys())
                }
            
            # Store old value
            old_value = self.environmental_factors[factor]
            
            # Update with bounds checking
            self.environmental_factors[factor] = max(0.0, min(1.0, value))
            
            return {
                "success": True,
                "factor": factor,
                "old_value": old_value,
                "new_value": self.environmental_factors[factor]
            }
        except Exception as e:
            logger.error(f"Error updating environmental factor: {e}")
            return {"success": False, "error": str(e)}
