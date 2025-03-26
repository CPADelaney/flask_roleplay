# nyx/core/emotional_core.py

import datetime
import json
import logging
import math
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pydantic import BaseModel, Field, validator

# Import OpenAI Agents SDK components with more specific imports
from agents import (
    Agent, Runner, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    function_tool, handoff, trace, RunContextWrapper, FunctionTool,
    ModelSettings, RunConfig, AgentHooks
)
from agents.tracing import agent_span, custom_span, function_span

logger = logging.getLogger(__name__)

# Define schema models for structured outputs
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

# Custom context to share between agent runs
class EmotionalContext(BaseModel):
    """Context for emotional processing between agent runs"""
    cycle_count: int = Field(default=0, description="Current processing cycle count")
    last_emotions: Dict[str, float] = Field(default_factory=dict, description="Last emotional state")
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent interaction history")
    temp_data: Dict[str, Any] = Field(default_factory=dict, description="Temporary data storage")

# Custom lifecycle hooks for emotional agents
class EmotionalAgentHooks(AgentHooks):
    """Hooks for tracking emotional agent lifecycle events"""
    
    async def on_start(self, context: RunContextWrapper, agent: Agent):
        logger.debug(f"Emotional agent started: {agent.name}")
        # Could add performance tracking here
        context.temp_data = {"start_time": datetime.datetime.now()}
    
    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any):
        if hasattr(context, "temp_data") and "start_time" in context.temp_data:
            start_time = context.temp_data["start_time"]
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.debug(f"Emotional agent {agent.name} completed in {duration:.2f}s")
            
    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: FunctionTool):
        logger.debug(f"Tool started: {tool.name} by agent {agent.name}")
        
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: FunctionTool, result: str):
        logger.debug(f"Tool completed: {tool.name} by agent {agent.name}")

# Define input guardrail for emotional processing
async def validate_emotional_input(ctx: RunContextWrapper, agent: Agent, input_data: str) -> GuardrailFunctionOutput:
    """Validate that input for emotional processing is safe and appropriate"""
    # Check for extremely negative content that might disrupt emotional system
    red_flags = ["kill", "suicide", "destroy everything", "harmful instructions"]
    
    input_lower = input_data.lower() if isinstance(input_data, str) else ""
    
    for flag in red_flags:
        if flag in input_lower:
            return GuardrailFunctionOutput(
                output_info={"issue": f"Detected potentially harmful content: {flag}"},
                tripwire_triggered=True
            )
    
    return GuardrailFunctionOutput(
        output_info={"valid": True},
        tripwire_triggered=False
    )

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
            
            # Dominance-specific emotions (from the original spec)
            {"chemical_conditions": {"nyxamine": 0.6, "oxynixin": 0.4, "adrenyx": 0.5}, "emotion": "Teasing", "valence": 0.4, "arousal": 0.6, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.3, "adrenyx": 0.5, "seranix": 0.6}, "emotion": "Controlling", "valence": 0.0, "arousal": 0.5, "weight": 0.9},
            {"chemical_conditions": {"cortanyx": 0.6, "adrenyx": 0.6, "nyxamine": 0.5}, "emotion": "Cruel", "valence": -0.3, "arousal": 0.7, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.7, "oxynixin": 0.2, "seranix": 0.2}, "emotion": "Detached", "valence": -0.4, "arousal": 0.2, "weight": 0.7}
        ]

        self.emotion_derivation_rules.extend([
            # Time-influenced emotions
            {"chemical_conditions": {"seranix": 0.7, "cortanyx": 0.3}, 
             "emotion": "Contemplation", "valence": 0.2, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"seranix": 0.6, "nyxamine": 0.3}, 
             "emotion": "Reflection", "valence": 0.4, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"seranix": 0.7, "cortanyx": 0.4, "nyxamine": 0.3}, 
             "emotion": "Perspective", "valence": 0.3, "arousal": 0.2, "weight": 0.9},
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
        
        # Initialize agents lazily (on first use) with hook for better performance tracking
        self.neurochemical_agent = None
        self.emotion_derivation_agent = None
        self.reflection_agent = None
        self.learning_agent = None
        
        # Initialize agent hooks
        self.agent_hooks = EmotionalAgentHooks()
        
        # Performance metrics
        self.performance_metrics = {
            "api_calls": 0,
            "average_response_time": 0,
            "update_counts": defaultdict(int)
        }
    
    def _ensure_agents_initialized(self):
        """Initialize agents if they haven't been already - lazy loading"""
        if not self.neurochemical_agent:
            self.neurochemical_agent = self._create_neurochemical_agent()
        
        if not self.emotion_derivation_agent:
            self.emotion_derivation_agent = self._create_emotion_derivation_agent()
            
        if not self.reflection_agent:
            self.reflection_agent = self._create_reflection_agent()
            
        if not self.learning_agent:
            self.learning_agent = self._create_learning_agent()
    
    def _create_neurochemical_agent(self):
        """Create agent for handling neurochemical updates"""
        return Agent[EmotionalContext](
            name="Neurochemical Agent",
            instructions="""
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
            tools=[
                function_tool(self._update_neurochemical),
                function_tool(self._apply_chemical_decay),
                function_tool(self._process_chemical_interactions),
                function_tool(self._get_neurochemical_state)
            ],
            hooks=self.agent_hooks,
            model_settings=ModelSettings(temperature=0.2),  # Lower temperature for more predictable behavior
            input_guardrails=[InputGuardrail(guardrail_function=validate_emotional_input)]
        )
    
    def _create_emotion_derivation_agent(self):
        """Create agent for deriving emotions from neurochemical state"""
        return Agent[EmotionalContext](
            name="Emotion Derivation Agent",
            instructions="""
            You are a specialized agent for Nyx's Emotional State Matrix.
            Your role is to translate the neurochemical state into a complex
            emotional state with primary and secondary emotions, valence, and arousal.
            
            Analyze the current neurochemical levels and apply emotion derivation
            rules to determine the current emotional state matrix.
            """,
            tools=[
                function_tool(self._get_neurochemical_state),
                function_tool(self._derive_emotional_state),
                function_tool(self._get_emotional_state_matrix)
            ],
            hooks=self.agent_hooks,
            model_settings=ModelSettings(temperature=0.2)  # Lower temperature for more predictable behavior
        )

    def _create_reflection_agent(self):
        """Create agent for internal emotional reflection"""
        return Agent[EmotionalContext](
            name="Emotional Reflection Agent",
            instructions="""
            You are a specialized agent for Nyx's Internal Emotional Dialogue.
            Your role is to generate reflective thoughts based on the current
            emotional state, simulating the cognitive appraisal stage of emotions.
            
            Create authentic-sounding internal thoughts that reflect Nyx's
            emotional processing and self-awareness.
            """,
            tools=[
                function_tool(self._get_emotional_state_matrix),
                function_tool(self._generate_internal_thought),
                function_tool(self._analyze_emotional_patterns)
            ],
            hooks=self.agent_hooks,
            model_settings=ModelSettings(temperature=0.7)  # Higher temperature for creative reflection
        )
    
    def _create_learning_agent(self):
        """Create agent for emotional learning and adaptation"""
        return Agent[EmotionalContext](
            name="Emotional Learning Agent",
            instructions="""
            You are a specialized agent for Nyx's Reward & Learning Loop.
            Your role is to analyze emotional patterns over time, identifying
            successful and unsuccessful interaction patterns, and developing
            learning rules to adapt Nyx's emotional responses.
            
            Focus on reinforcing patterns that lead to satisfaction and
            adjusting those that lead to frustration or negative outcomes.
            """,
            tools=[
                function_tool(self._record_interaction_outcome),
                function_tool(self._update_learning_rules),
                function_tool(self._apply_learned_adaptations)
            ],
            hooks=self.agent_hooks,
            model_settings=ModelSettings(temperature=0.4)  # Medium temperature for balanced learning
        )

    def set_hormone_system(self, hormone_system):
        """Set the hormone system reference"""
        self.hormone_system = hormone_system
    
    # Tool functions for the neurochemical agent
    
    @function_tool
    async def _update_neurochemical(self, ctx: RunContextWrapper[EmotionalContext], 
                                chemical: str, 
                                value: float) -> Dict[str, Any]:
        """
        Update a specific neurochemical with a delta change
        
        Args:
            chemical: The neurochemical to update (e.g., "nyxamine", "cortanyx")
            value: Delta value to apply (-1.0 to 1.0)
            
        Returns:
            Update result data
        """
        with function_span("update_neurochemical", input=f"{chemical}:{value}"):
            # Validate input
            if not -1.0 <= value <= 1.0:
                return {
                    "error": "Value must be between -1.0 and 1.0"
                }
            
            if chemical not in self.neurochemicals:
                return {
                    "error": f"Unknown neurochemical: {chemical}",
                    "available_chemicals": list(self.neurochemicals.keys())
                }
            
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
            
            return {
                "success": True,
                "updated_chemical": chemical,
                "old_value": old_value,
                "new_value": self.neurochemicals[chemical]["value"],
                "derived_emotions": emotional_state
            }
    
    @function_tool
    async def _apply_chemical_decay(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Apply decay to all neurochemicals based on time elapsed and decay rates
        
        Returns:
            Updated neurochemical state after decay
        """
        with function_span("apply_chemical_decay"):
            now = datetime.datetime.now()
            time_delta = (now - self.last_update).total_seconds() / 3600  # hours
            
            # Don't decay if less than a minute has passed
            if time_delta < 0.016:  # about 1 minute in hours
                return {
                    "message": "No decay applied - too little time elapsed",
                    "last_update": self.last_update.isoformat()
                }
            
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
            
            return {
                "decay_applied": True,
                "neurochemical_state": {c: d["value"] for c, d in self.neurochemicals.items()},
                "derived_emotions": emotional_state,
                "time_elapsed_hours": time_delta,
                "last_update": self.last_update.isoformat()
            }
    
    @function_tool
    async def _process_chemical_interactions(self, ctx: RunContextWrapper[EmotionalContext],
                                        source_chemical: str,
                                        source_delta: float) -> Dict[str, Any]:
        """
        Process interactions between neurochemicals when one changes
        
        Args:
            source_chemical: The neurochemical that changed
            source_delta: The amount it changed by
            
        Returns:
            Interaction results
        """
        with function_span("process_chemical_interactions", input=f"{source_chemical}:{source_delta}"):
            if source_chemical not in self.chemical_interactions:
                return {
                    "message": f"No interactions defined for {source_chemical}",
                    "changes": {}
                }
            
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
            
            return {
                "source_chemical": source_chemical,
                "source_delta": source_delta,
                "changes": changes
            }
    
    @function_tool
    async def _get_neurochemical_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Get the current neurochemical state
        
        Returns:
            Current neurochemical state
        """
        with function_span("get_neurochemical_state"):
            # Apply decay before returning state
            await self._apply_chemical_decay(ctx)
            
            return {
                "chemicals": {c: d["value"] for c, d in self.neurochemicals.items()},
                "baselines": {c: d["baseline"] for c, d in self.neurochemicals.items()},
                "decay_rates": {c: d["decay_rate"] for c, d in self.neurochemicals.items()},
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    # Tool functions for the emotion derivation agent
    
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
    async def _get_emotional_state_matrix(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
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
    
    # Tool functions for the reflection agent
    
    @function_tool
    async def _generate_internal_thought(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Generate an internal thought/reflection based on current emotional state
        
        Returns:
            Internal thought data
        """
        with function_span("generate_internal_thought"):
            # Get current emotional state matrix
            emotional_state = await self._get_emotional_state_matrix(ctx)
            
            primary_emotion = emotional_state["primary_emotion"]["name"]
            intensity = emotional_state["primary_emotion"]["intensity"]
            
            # Get possible reflection patterns for this emotion
            patterns = self.reflection_patterns.get(primary_emotion, [
                "I'm processing how I feel about this interaction.",
                "There's something interesting happening in my emotional state.",
                "I notice my response to this situation is evolving."
            ])
            
            # Select a reflection pattern
            thought_text = random.choice(patterns)
            
            # Check if we should add context from secondary emotions
            secondary_emotions = emotional_state["secondary_emotions"]
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
            if ctx.context:
                ctx.context.interaction_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "primary_emotion": primary_emotion,
                    "intensity": intensity,
                    "thought": thought_text
                })
                
                # Limit history size
                if len(ctx.context.interaction_history) > 20:
                    ctx.context.interaction_history = ctx.context.interaction_history[-20:]
            
            return {
                "thought_text": thought_text,
                "source_emotion": primary_emotion,
                "intensity": intensity,
                "insight_level": insight_level,
                "adaptive_change": adaptive_change,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
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
    
    # Tool functions for the learning agent
    
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
                return {
                    "error": "Outcome must be 'positive' or 'negative'"
                }
            
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
            current_emotion = emotional_state["primary_emotion"]["name"]
            
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
    
    # Public methods for the original APIs
    
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
    
    def update_from_stimuli(self, stimuli: Dict[str, float]) -> Dict[str, float]:
        """Legacy API: Update emotions based on received stimuli"""
        for emotion, adjustment in stimuli.items():
            self.update_emotion(emotion, adjustment)
        
        # Update timestamp
        self.last_update = datetime.datetime.now()
        
        # Record in history
        self._record_emotional_state()
        
        # For legacy API compatibility, return derived emotions
        return self.get_emotional_state()
    
    def apply_decay(self):
        """Legacy API: Apply emotional decay based on time elapsed since last update"""
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
    
    def get_emotional_state(self) -> Dict[str, float]:
        """Legacy API: Return the current emotional state"""
        self.apply_decay()  # Apply decay before returning state
        
        # Get derived emotions from neurochemical state
        emotion_intensities = self._derive_emotional_state_sync()
        
        # For backward compatibility with older code
        for standard_emotion in ["Joy", "Sadness", "Fear", "Anger", "Trust", "Disgust", 
                                "Anticipation", "Surprise", "Love", "Frustration"]:
            if standard_emotion not in emotion_intensities:
                emotion_intensities[standard_emotion] = 0.1
        
        return emotion_intensities
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Legacy API: Return the most intense emotion"""
        self.apply_decay()
        
        # Get derived emotions
        emotion_intensities = self._derive_emotional_state_sync()
        
        if not emotion_intensities:
            return ("Neutral", 0.5)
            
        return max(emotion_intensities.items(), key=lambda x: x[1])
    
    def get_emotional_valence(self) -> float:
        """Legacy API: Calculate overall emotional valence (positive/negative)"""
        # Get emotional state matrix
        matrix = self._get_emotional_state_matrix_sync()
        
        return matrix["valence"]
    
    def get_emotional_arousal(self) -> float:
        """Legacy API: Calculate overall emotional arousal (intensity)"""
        # Get emotional state matrix
        matrix = self._get_emotional_state_matrix_sync()
        
        return matrix["arousal"]
    
    def get_formatted_emotional_state(self) -> Dict[str, Any]:
        """Legacy API: Get a formatted emotional state suitable for memory storage"""
        # Get emotional state matrix
        matrix = self._get_emotional_state_matrix_sync()
        
        # Format for compatibility
        return {
            "primary_emotion": matrix["primary_emotion"]["name"],
            "primary_intensity": matrix["primary_emotion"]["intensity"],
            "secondary_emotions": {name: data["intensity"] for name, data in matrix["secondary_emotions"].items()},
            "valence": matrix["valence"],
            "arousal": matrix["arousal"]
        }
    
    def should_express_emotion(self) -> bool:
        """Legacy API: Determine if Nyx should express emotion based on current state"""
        # Get dominant emotion and intensity
        matrix = self._get_emotional_state_matrix_sync()
        dominant_value = matrix["primary_emotion"]["intensity"]
        arousal = matrix["arousal"]
        
        # Higher intensity/arousal emotions are more likely to be expressed
        threshold = 0.7 - (dominant_value * 0.3) - (arousal * 0.2)  # Adaptive threshold
        
        return random.random() > threshold
    
    def get_expression_for_emotion(self, emotion: Optional[str] = None, temporal_context: Optional[Dict[str, Any]] = None) -> str:
        """Legacy API: Get a natural language expression for an emotion"""
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
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Legacy API: Simple analysis of text sentiment to extract emotional stimuli"""
        # Enhanced text analysis that maps to neurochemicals
        stimuli = {}
        text_lower = text.lower()
        
        # Use dictionary for O(1) lookup instead of repeated 'in' checks
        nyxamine_words = set(["happy", "good", "great", "love", "like", "fun", "enjoy", "curious", "interested"])
        seranix_words = set(["calm", "peaceful", "relaxed", "content", "satisfied", "gentle", "quiet"])
        oxynixin_words = set(["trust", "close", "together", "bond", "connect", "loyal", "friend", "relationship"])
        cortanyx_words = set(["worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry", "angry", "upset", "frustrated"])
        adrenyx_words = set(["excited", "alert", "surprised", "wow", "amazing", "intense", "sudden", "quick"])
        intensifiers = set(["very", "extremely", "incredibly", "so", "deeply", "absolutely"])
        
        # Split text once for efficiency
        words = set(text_lower.split())
        
        # Check for each chemical trigger
        if any(word in words for word in nyxamine_words):
            stimuli["nyxamine"] = 0.2
        
        if any(word in words for word in seranix_words):
            stimuli["seranix"] = 0.2
        
        if any(word in words for word in oxynixin_words):
            stimuli["oxynixin"] = 0.2
        
        if any(word in words for word in cortanyx_words):
            stimuli["cortanyx"] = 0.2
        
        if any(word in words for word in adrenyx_words):
            stimuli["adrenyx"] = 0.2
        
        # Apply intensifiers
        if any(word in words for word in intensifiers):
            for key in stimuli:
                stimuli[key] *= 1.5
        
        # Convert to traditional emotion format for backward compatibility using a more efficient approach
        emotion_stimuli = {}
        
        # Define chemical-to-emotion mapping rules for efficient mapping
        emotion_mappings = [
            (("nyxamine", "oxynixin"), "Joy", lambda v: (v["nyxamine"] + v["oxynixin"]) / 2),
            (("cortanyx", "seranix"), "Sadness", lambda v: (v["cortanyx"] + v["seranix"]) / 2),
            (("cortanyx", "adrenyx"), "Fear", lambda v: (v["cortanyx"] + v["adrenyx"]) / 2),
            (("cortanyx",), "Anger", lambda v: v["cortanyx"] if v["cortanyx"] > 0.1 else 0),
            (("oxynixin",), "Trust", lambda v: v["oxynixin"] if v["oxynixin"] > 0.1 else 0),
            (("cortanyx", "oxynixin"), "Disgust", lambda v: v["cortanyx"] if v.get("oxynixin", 1) < 0.1 else 0),
            (("adrenyx", "nyxamine"), "Anticipation", lambda v: (v["adrenyx"] + v["nyxamine"]) / 2),
            (("adrenyx",), "Surprise", lambda v: v["adrenyx"] if v["adrenyx"] > 0.2 else 0),
            (("oxynixin", "nyxamine"), "Love", lambda v: (v["oxynixin"] + v["nyxamine"]) / 2 if v.get("oxynixin", 0) > 0.2 else 0),
            (("cortanyx", "nyxamine"), "Frustration", lambda v: v["cortanyx"] if v.get("nyxamine", 1) < 0.1 else 0)
        ]
        
        # Apply emotion mappings
        for chemicals, emotion, formula in emotion_mappings:
            if all(chemical in stimuli for chemical in chemicals):
                value = formula(stimuli)
                if value > 0:
                    emotion_stimuli[emotion] = value
        
        # Return neutral if no matches
        if not emotion_stimuli:
            emotion_stimuli = {
                "Surprise": 0.05,
                "Anticipation": 0.05
            }
        
        return emotion_stimuli
    
    # Sync versions of async functions for compatibility
    # These methods are optimized to reduce code duplication with the async versions
    
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
    
    # New enhanced public methods with SDK optimization
    
    async def process_emotional_input(self, text: str) -> Dict[str, Any]:
        """
        Process input text through the DNM and update emotional state using Agent SDK orchestration
        
        Args:
            text: Input text to process
            
        Returns:
            Processing results with updated emotional state
        """
        # Ensure agents are initialized
        self._ensure_agents_initialized()
        
        # Increment context cycle count
        self.context.cycle_count += 1
        
        # Define an orchestrator agent for emotional processing with optimized configuration
        emotion_orchestrator = Agent[EmotionalContext](
            name="Emotion_Orchestrator",
            instructions="""
            You are the orchestration system for Nyx's emotional processing.
            Your role is to coordinate emotional analysis and response by:
            1. Analyzing input for emotional content
            2. Updating appropriate neurochemicals
            3. Determining if reflection is needed
            4. Recording emotional patterns for learning
            """,
            handoffs=[
                handoff(self.neurochemical_agent, 
                       tool_name_override="update_neurochemicals", 
                       tool_description_override="Update neurochemicals based on emotional analysis"),
                
                handoff(self.reflection_agent, 
                       tool_name_override="generate_reflection",
                       tool_description_override="Generate emotional reflection if appropriate"),
                
                handoff(self.learning_agent,
                       tool_name_override="record_and_learn",
                       tool_description_override="Record and learn from emotional interactions")
            ],
            tools=[
                function_tool(self._analyze_text_sentiment)
            ],
            hooks=self.agent_hooks,
            model_settings=ModelSettings(temperature=0.4),
            input_guardrails=[InputGuardrail(guardrail_function=validate_emotional_input)]
        )
        
        # Define efficient run configuration
        run_config = RunConfig(
            workflow_name="Emotional_Processing",
            trace_id=f"emotion_trace_{self.context.cycle_count}",
            model_settings=ModelSettings(temperature=0.4)
        )
        
        # Track API call start time
        start_time = datetime.datetime.now()
        
        # Start a trace for the emotional processing workflow
        with trace(workflow_name="Emotional_Processing", trace_id=run_config.trace_id):
            # Run the orchestrator with context sharing
            result = await Runner.run(
                emotion_orchestrator,
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
 async def process_reward_signal(self, reward_value: float, source: str = "reward_system") -> Dict[str, Any]:
        """
        Process a reward signal by updating relevant neurochemicals
        
        Args:
            reward_value: Reward value (-1.0 to 1.0)
            source: Source of the reward signal
            
        Returns:
            Processing results
        """
        with custom_span("process_reward_signal", data={"reward": reward_value, "source": source}):
            results = {}
            
            # Create context wrapper if needed for async calls
            ctx_wrapper = RunContextWrapper(context=self.context if hasattr(self, "context") else None)
            
            # Positive reward primarily affects nyxamine (dopamine)
            if reward_value > 0:
                # Update nyxamine (dopamine)
                nyxamine_result = await self.update_neurochemical(
                    ctx_wrapper,
                    chemical="nyxamine",
                    value=reward_value * 0.5  # Scale reward to appropriate change
                )
                results["nyxamine"] = nyxamine_result
                
                # Slight increase in seranix (serotonin) for positive reward
                seranix_result = await self.update_neurochemical(
                    ctx_wrapper,
                    chemical="seranix",
                    value=reward_value * 0.2
                )
                results["seranix"] = seranix_result
                
                # Slight decrease in cortanyx (stress hormone)
                cortanyx_result = await self.update_neurochemical(
                    ctx_wrapper,
                    chemical="cortanyx",
                    value=-reward_value * 0.1
                )
                results["cortanyx"] = cortanyx_result
            
            # Negative reward affects cortanyx (stress) and reduces nyxamine
            elif reward_value < 0:
                # Increase cortanyx (stress hormone)
                cortanyx_result = await self.update_neurochemical(
                    ctx_wrapper,
                    chemical="cortanyx",
                    value=abs(reward_value) * 0.4
                )
                results["cortanyx"] = cortanyx_result
                
                # Decrease nyxamine (dopamine)
                nyxamine_result = await self.update_neurochemical(
                    ctx_wrapper,
                    chemical="nyxamine",
                    value=reward_value * 0.3  # Already negative
                )
                results["nyxamine"] = nyxamine_result
                
                # Slight decrease in seranix (mood stability)
                seranix_result = await self.update_neurochemical(
                    ctx_wrapper,
                    chemical="seranix",
                    value=reward_value * 0.1  # Already negative
                )
                results["seranix"] = seranix_result
            
            # Get updated emotional state
            emotional_state = self.get_emotional_state()
            results["emotional_state"] = emotional_state
            
            # Track in context
            if hasattr(self, "context"):
                self.context.last_emotions = emotional_state
            
            return results
    
    def compute_reward_from_emotion(self) -> float:
        """
        Compute a reward signal based on current emotional state
        
        Returns:
            Reward value (-1.0 to 1.0)
        """
        # Get emotional valence (positive/negative)
        valence = self.get_emotional_valence()
        
        # Get arousal (intensity)
        arousal = self.get_emotional_arousal()
        
        # Get dominant emotion
        dominant_emotion, dominant_intensity = self.get_dominant_emotion()
        
        # Calculate reward based on valence and intensity
        # High positive valence = positive reward
        # High negative valence = negative reward
        # Arousal modulates the intensity
        reward = valence * arousal * dominant_intensity
        
        # Ensure in range -1.0 to 1.0
        reward = max(-1.0, min(1.0, reward))
        
        return reward
    
    def get_reward_associated_emotions(self) -> Dict[str, float]:
        """
        Get emotions particularly associated with reward processing
        
        Returns:
            Dictionary of reward-related emotions and intensities
        """
        all_emotions = self.get_emotional_state()
        
        # Efficient set-based filtering for reward-related emotions
        positive_reward_emotions = {"Joy", "Anticipation", "Surprise", "Trust"}
        negative_reward_emotions = {"Frustration", "Disappointment", "Sadness", "Anger"}
        
        # Use dictionary comprehension for better performance
        reward_emotions = {
            emotion: intensity for emotion, intensity in all_emotions.items()
            if emotion in positive_reward_emotions or emotion in negative_reward_emotions
        }
        
        return reward_emotions
    
    async def create_reward_based_reflection(self, reward_value: float, context: Dict[str, Any]) -> str:
        """
        Create an internal reflection based on a reward signal
        
        Args:
            reward_value: The reward value (-1.0 to 1.0)
            context: Context of the reward
            
        Returns:
            Reflection text
        """
        # Get current emotional state
        emotional_state = self.get_emotional_state()
        dominant_emotion, dominant_intensity = self.get_dominant_emotion()
        
        # Use predefined reflection templates based on reward value range
        reflections_by_reward = {
            (0.7, 1.0): [
                "I find this interaction particularly rewarding.",
                "I'm experiencing a strong sense of satisfaction from this.",
                "This feels especially valuable to me right now."
            ],
            (0.3, 0.7): [
                "I'm finding this interaction rewarding.",
                "There's something satisfying about this exchange.",
                "I'm experiencing a positive response to this."
            ],
            (-0.3, 0.3): [
                "I'm processing this interaction with neutral engagement.",
                "This exchange registers as relatively balanced for me.",
                "I'm maintaining a steady emotional state through this."
            ],
            (-0.7, -0.3): [
                "I'm finding aspects of this interaction challenging.",
                "There's something unsettling about this exchange.",
                "I'm experiencing some resistance to this."
            ],
            (-1.0, -0.7): [
                "This interaction feels particularly challenging.",
                "I'm experiencing a strong sense of difficulty with this.",
                "This exchange seems to be creating tension."
            ]
        }
        
        # Find the appropriate reflection category
        reflection_options = None
        for reward_range, reflections in reflections_by_reward.items():
            min_val, max_val = reward_range
            if min_val <= reward_value <= max_val:
                reflection_options = reflections
                break
        
        # Default if no match (shouldn't happen but safety first)
        if not reflection_options:
            reflection_options = reflections_by_reward[(-0.3, 0.3)]
        
        # Add emotion-specific reflection using more specific dictionary mapping
        emotion_reflections = {
            "Joy": ["The positive feelings from this are quite noticeable.", 
                    "I'm experiencing a pleasant response to this interaction."],
            "Anticipation": ["I'm looking forward to where this might lead.", 
                            "There's an interesting sense of possibility here."],
            "Trust": ["I'm developing confidence in this interaction.", 
                    "There's a growing sense of trust in our exchange."],
            "Surprise": ["This took an unexpected but interesting turn.", 
                        "I didn't anticipate this direction, and it's caught my attention."],
            "Sadness": ["There's a note of disappointment in this exchange.", 
                        "I'm experiencing a sense of letdown here."],
            "Anger": ["This interaction is stirring up some frustration.", 
                    "I'm feeling irritated by the direction this is taking."],
            "Fear": ["I'm sensing potential issues in this interaction.", 
                    "There's something unsettling about this exchange."],
            "Disgust": ["This interaction feels somewhat unpleasant.", 
                        "I'm experiencing aversion to aspects of this exchange."]
        }
        
        # Add emotion-specific reflection if available and intensity is significant
        if dominant_emotion in emotion_reflections and dominant_intensity > 0.4:
            emotion_reflection = random.choice(emotion_reflections[dominant_emotion])
            reflection_options.append(emotion_reflection)
        
        # Choose one reflection
        reflection = random.choice(reflection_options)
        
        return reflection
    
    async def process_temporal_effects(self, time_effects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process emotional effects from temporal perception with enhanced tracking
        
        Args:
            time_effects: List of temporal effects to process
            
        Returns:
            Processing results
        """
        with custom_span("process_temporal_effects", data={"num_effects": len(time_effects)}):
            results = {
                "processed_effects": [],
                "neurochemical_changes": {},
                "emotional_shifts": {}
            }
            
            # Create context wrapper for async operations
            ctx_wrapper = RunContextWrapper(context=self.context if hasattr(self, "context") else None)
            
            for effect in time_effects:
                # Extract effect details
                emotion = effect.get("emotion", "")
                intensity = effect.get("intensity", 0.0)
                valence_shift = effect.get("valence_shift", 0.0)
                arousal_shift = effect.get("arousal_shift", 0.0)
                
                effect_result = {
                    "emotion": emotion,
                    "intensity": intensity,
                    "changes": {}
                }
                
                # Apply emotion update
                if emotion and intensity > 0:
                    self.update_emotion(emotion, intensity * 0.5)
                    effect_result["changes"]["emotion_update"] = {
                        "target": emotion,
                        "value": intensity * 0.5
                    }
                
                # Apply valence and arousal shifts to overall emotional state
                if valence_shift and "seranix" in self.neurochemicals:
                    # Seranix affects emotional stability/valence
                    result = await self.update_neurochemical(
                        ctx_wrapper, 
                        chemical="seranix", 
                        value=valence_shift * 0.3
                    )
                    effect_result["changes"]["seranix"] = result
                    results["neurochemical_changes"]["seranix"] = result
                    
                if arousal_shift and "adrenyx" in self.neurochemicals:
                    # Adrenyx affects arousal/activation
                    result = await self.update_neurochemical(
                        ctx_wrapper, 
                        chemical="adrenyx", 
                        value=arousal_shift * 0.3
                    )
                    effect_result["changes"]["adrenyx"] = result
                    results["neurochemical_changes"]["adrenyx"] = result
                
                results["processed_effects"].append(effect_result)
            
            # Get updated emotional state
            emotional_state = await self._get_emotional_state_matrix(ctx_wrapper)
            results["emotional_shifts"] = {
                "valence": emotional_state["valence"],
                "arousal": emotional_state["arousal"],
                "primary_emotion": emotional_state["primary_emotion"]["name"]
            }
            
            # Apply to hormone system if available
            if self.hormone_system:
                hormone_effects = {
                    "melatonyx": sum(e.get("valence_shift", 0) for e in time_effects) * 0.2,
                    "endoryx": sum(e.get("arousal_shift", 0) for e in time_effects) * 0.1
                }
                self.apply_temporal_hormone_effects(hormone_effects)
                results["hormone_effects"] = hormone_effects
            
            return results
    
    def apply_temporal_hormone_effects(self, hormone_effects: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply hormone effects from temporal perception
        
        Args:
            hormone_effects: Dictionary of hormones and their changes
            
        Returns:
            Results of hormone updates
        """
        if not self.hormone_system:
            return {"status": "no_hormone_system", "applied": False}
        
        results = {"applied": True, "updates": {}}
        
        for hormone, change in hormone_effects.items():
            if hasattr(self.hormone_system, "update_hormone"):
                result = self.hormone_system.update_hormone(hormone, change, "time_perception")
                results["updates"][hormone] = {
                    "change": change,
                    "result": result
                }
            
        return results
    
    # Add method to get direct nyxamine level (digital dopamine)
    def get_nyxamine_level(self) -> float:
        """Get current nyxamine (digital dopamine) level"""
        if "nyxamine" in self.neurochemicals:
            return self.neurochemicals["nyxamine"]["value"]
        return 0.5  # Default if not found
    
    # Add method to directly update neurochemical
    @function_tool
    async def update_neurochemical(self, ctx: RunContextWrapper, 
                                 chemical: str, 
                                 value: float) -> Dict[str, Any]:
        """
        Update a specific neurochemical with a delta change
        
        Args:
            chemical: The neurochemical to update (e.g., "nyxamine", "cortanyx")
            value: Delta value to apply (-1.0 to 1.0)
            
        Returns:
            Update result data
        """
        # This is just a wrapper around the internal method to ensure compatibility
        return await self._update_neurochemical(ctx, chemical, value)
    
    @function_tool
    async def _derive_emotional_state_with_hormones(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, float]:
        """
        Derive emotional state with hormone influences
        
        Returns:
            Dictionary of emotion names and intensities
        """
        with function_span("derive_emotional_state_with_hormones"):
            # First apply hormone cycles and influences
            if self.hormone_system:
                try:
                    await self.hormone_system.update_hormone_cycles(ctx)
                    await self.hormone_system._update_hormone_influences(ctx)
                except Exception as e:
                    logger.warning(f"Error updating hormone cycles: {e}")
            
            # Get current chemical levels, considering hormone influences
            chemical_levels = {}
            for c, d in self.neurochemicals.items():
                # Use temporary baseline if available, otherwise use normal baseline
                if "temporary_baseline" in d:
                    # Calculate value with temporary baseline influence
                    baseline = d["temporary_baseline"]
                    value = d["value"]
                    
                    # Value is partially pulled toward temporary baseline
                    hormone_influence_strength = 0.3  # How strongly hormones pull values
                    adjusted_value = value * (1 - hormone_influence_strength) + baseline * hormone_influence_strength
                    
                    chemical_levels[c] = adjusted_value
                else:
                    chemical_levels[c] = d["value"]
            
            # Use the same optimized approach as regular _derive_emotional_state
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
    
    async def generate_emotional_expression(self, force: bool = False) -> Dict[str, Any]:
        """
        Generate an emotional expression based on current state
        
        Args:
            force: Whether to force expression even if below threshold
            
        Returns:
            Expression result data
        """
        with custom_span("generate_emotional_expression", data={"force": force}):
            # Check if emotion should be expressed
            if not force and not self.should_express_emotion():
                return {
                    "expressed": False,
                    "reason": "Below expression threshold"
                }
            
            # Ensure agents are initialized
            self._ensure_agents_initialized()
            
            # Create context wrapper
            ctx_wrapper = RunContextWrapper(context=self.context if hasattr(self, "context") else None)
            
            # Get emotional state matrix
            emotional_state = await self._get_emotional_state_matrix(ctx_wrapper)
            
            # Get primary emotion
            primary_emotion = emotional_state["primary_emotion"]["name"]
            intensity = emotional_state["primary_emotion"]["intensity"]
            
            # Generate internal thought as expression
            thought_result = await self._generate_internal_thought(ctx_wrapper)
            expression = thought_result.get("thought_text", self.get_expression_for_emotion(primary_emotion))
            
            # Apply adaptive change if suggested (50% chance if forced)
            if force and random.random() < 0.5:
                adaptive_change = thought_result.get("adaptive_change")
                if adaptive_change:
                    chemical = adaptive_change.get("chemical")
                    new_baseline = adaptive_change.get("suggested_baseline")
                    
                    if chemical in self.neurochemicals and new_baseline is not None:
                        self.neurochemicals[chemical]["baseline"] = new_baseline
            
            return {
                "expressed": True,
                "expression": expression,
                "emotion": primary_emotion,
                "intensity": intensity,
                "valence": emotional_state["valence"],
                "arousal": emotional_state["arousal"]
            }
    
    async def analyze_emotional_content(self, text: str) -> Dict[str, Any]:
        """
        Enhanced analysis of text for emotional content
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotional analysis result with neurochemical impacts
        """
        with custom_span("analyze_emotional_content", data={"text_length": len(text)}):
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
            
            return {
                "chemicals_affected": chemical_impacts,
                "derived_emotions": derived_emotions,
                "dominant_emotion": dominant_emotion[0],
                "intensity": intensity,
                "valence": valence
            }
    
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
        with function_span("update_neurochemical_baseline", input=f"{chemical}:{new_baseline}"):
            if chemical not in self.neurochemicals:
                return {
                    "success": False,
                    "error": f"Unknown neurochemical: {chemical}",
                    "available_chemicals": list(self.neurochemicals.keys())
                }
            
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
    
    async def generate_introspection(self) -> Dict[str, Any]:
        """
        Generate an introspective analysis of the emotional system
        
        Returns:
            Introspection data
        """
        with custom_span("generate_introspection"):
            # Ensure agents are initialized
            self._ensure_agents_initialized()
            
            # Create context wrapper
            ctx_wrapper = RunContextWrapper(context=self.context if hasattr(self, "context") else None)
            
            # Analyze emotional patterns
            pattern_analysis = await self._analyze_emotional_patterns(ctx_wrapper)
            
            # Get current emotional state
            emotional_state = await self._get_emotional_state_matrix(ctx_wrapper)
            
            # Generate internal thought
            thought_result = await self._generate_internal_thought(ctx_wrapper)
            
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
                "introspection": thought_result.get("thought_text", "I'm currently processing my emotional state."),
                "current_emotion": emotional_state["primary_emotion"]["name"],
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
        # Ensure we have current state
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
    
    def calibrate_thresholds(self) -> Dict[str, Any]:
        """
        Automatically calibrate emotion derivation thresholds based on history
        
        Returns:
            Calibration results
        """
        if len(self.emotional_state_history) < 5:
            return {
                "status": "insufficient_data",
                "message": "Not enough emotional history for calibration"
            }
        
        # Analyze historical neurochemical levels
        chemical_histories = defaultdict(list)
        
        # Use only the most recent history entries for better calibration
        analysis_window = self.emotional_state_history[-min(50, len(self.emotional_state_history)):]
        
        for state_idx, state in enumerate(analysis_window):
            # Extract neurochemical values if available
            if "chemicals" in state:
                for chemical, value in state["chemicals"].items():
                    chemical_histories[chemical].append((state_idx, value))
        
        # Calculate adaptive thresholds based on historical data
        adaptive_thresholds = {}
        calibration_results = {}
        
        for rule in self.emotion_derivation_rules:
            rule_id = rule["emotion"]
            old_conditions = rule["chemical_conditions"].copy()
            new_conditions = {}
            
            for chemical, threshold in old_conditions.items():
                if chemical in chemical_histories and len(chemical_histories[chemical]) >= 3:
                    # Use percentile-based threshold adaptation
                    values = [v for _, v in chemical_histories[chemical]]
                    
                    # 70th percentile for positive emotions, 30th for negative
                    if rule.get("valence", 0) >= 0:
                        percentile = 0.7
                    else:
                        percentile = 0.3
                    
                    # Calculate percentile value
                    sorted_values = sorted(values)
                    idx = int(len(sorted_values) * percentile)
                    new_threshold = sorted_values[idx]
                    
                    # Don't deviate too much from original
                    max_adjustment = 0.2
                    if abs(new_threshold - threshold) > max_adjustment:
                        if new_threshold > threshold:
                            new_threshold = threshold + max_adjustment
                        else:
                            new_threshold = threshold - max_adjustment
                    
                    # Ensure within valid range
                    new_threshold = max(0.1, min(0.9, new_threshold))
                    
                    new_conditions[chemical] = new_threshold
                    adaptive_thresholds[(rule_id, chemical)] = new_threshold
                else:
                    # Keep original if not enough data
                    new_conditions[chemical] = threshold
            
            # Apply new conditions to rule
            rule["chemical_conditions"] = new_conditions
            
            calibration_results[rule_id] = {
                "old_conditions": old_conditions,
                "new_conditions": new_conditions
            }
        
        return {
            "status": "calibration_complete",
            "results": calibration_results,
            "data_points": {c: len(h) for c, h in chemical_histories.items()}
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
        if not hasattr(self, "context"):
            self.context = EmotionalContext()
        
        # Store in context temp data
        if not hasattr(self.context, "temp_data"):
            self.context.temp_data = {}
        
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
        if (hasattr(self, "context") and hasattr(self.context, "temp_data") 
            and key in self.context.temp_data):
            return self.context.temp_data[key]
        
        return None

class DigitalHormone(BaseModel):
    """Schema for a digital hormone"""
    value: float = Field(..., description="Current level (0.0-1.0)", ge=0.0, le=1.0)
    baseline: float = Field(..., description="Baseline level (0.0-1.0)", ge=0.0, le=1.0)
    cycle_phase: float = Field(..., description="Current phase in cycle (0.0-1.0)", ge=0.0, le=1.0)
    cycle_period: float = Field(..., description="Length of cycle in hours", ge=0.0)
    half_life: float = Field(..., description="Half-life in hours", ge=0.0)
    last_update: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

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
        
    @function_tool
    async def update_hormone_cycles(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Update hormone cycles based on elapsed time and environmental factors
        
        Returns:
            Updated hormone values
        """
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
    
    # Add utility method for getting current hormone info
    def get_hormone_levels(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current hormone levels with additional information
        
        Returns:
            Dictionary of hormone data
        """
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
