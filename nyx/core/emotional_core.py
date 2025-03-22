# nyx/core/emotional_core.py

import datetime
import json
import logging
import math
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pydantic import BaseModel, Field, validator

# Import OpenAI Agents SDK components 
from agents import (
    Agent, Runner, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    function_tool, handoff, trace, RunContextWrapper, FunctionTool
)

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
        self.hormone_system = hormone_system
        
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
        
        # History of emotional states for learning and reflection
        self.emotional_state_history = []
        self.max_history_size = 100
        
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
        
        # Initialize agents for processing
        self._init_agents()
    
    def _init_agents(self):
        """Initialize the agents for emotional processing"""
        self.neurochemical_agent = self._create_neurochemical_agent()
        self.emotion_derivation_agent = self._create_emotion_derivation_agent()
        self.reflection_agent = self._create_reflection_agent()
        self.learning_agent = self._create_learning_agent()
    
    def _create_neurochemical_agent(self):
        """Create agent for handling neurochemical updates"""
        return Agent(
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
            ]
        )
    
    def _create_emotion_derivation_agent(self):
        """Create agent for deriving emotions from neurochemical state"""
        return Agent(
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
            ]
        )
        
    @function_tool
    async def _derive_emotional_state_with_hormones(self, ctx: RunContextWrapper) -> Dict[str, float]:
        """
        Derive emotional state with hormone influences
        
        Returns:
            Dictionary of emotion names and intensities
        """
        # First apply hormone cycles and influences
        if self.hormone_system:
            await self.hormone_system.update_hormone_cycles(ctx)
            await self.hormone_system._update_hormone_influences(ctx)
        
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
    
    def _create_reflection_agent(self):
        """Create agent for internal emotional reflection"""
        return Agent(
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
            ]
        )
    
    def _create_learning_agent(self):
        """Create agent for emotional learning and adaptation"""
        return Agent(
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
            ]
        )
    
    # Tool functions for the neurochemical agent
    
    @function_tool
    async def _update_neurochemical(self, ctx: RunContextWrapper, 
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
        
        # Process chemical interactions
        await self._process_chemical_interactions(ctx, source_chemical=chemical, source_delta=value)
        
        # Derive emotions from updated neurochemical state
        emotional_state = await self._derive_emotional_state(ctx)
        
        # Update timestamp and record in history
        self.last_update = datetime.datetime.now()
        self._record_emotional_state()
        
        return {
            "success": True,
            "updated_chemical": chemical,
            "old_value": old_value,
            "new_value": self.neurochemicals[chemical]["value"],
            "derived_emotions": emotional_state
        }
    
    @function_tool
    async def _apply_chemical_decay(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Apply decay to all neurochemicals based on time elapsed and decay rates
        
        Returns:
            Updated neurochemical state after decay
        """
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
    async def _process_chemical_interactions(self, ctx: RunContextWrapper,
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
    async def _get_neurochemical_state(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current neurochemical state
        
        Returns:
            Current neurochemical state
        """
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
    async def _derive_emotional_state(self, ctx: RunContextWrapper) -> Dict[str, float]:
        """
        Derive emotional state from current neurochemical levels
        
        Returns:
            Dictionary of emotion names and intensities
        """
        # Get current chemical levels
        chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
        
        # Calculate emotional intensities based on derivation rules
        emotion_intensities = {}
        
        for rule in self.emotion_derivation_rules:
            conditions = rule["chemical_conditions"]
            emotion = rule["emotion"]
            rule_weight = rule.get("weight", 1.0)
            
            # Check if chemical levels meet the conditions
            match_score = 0
            for chemical, threshold in conditions.items():
                if chemical in chemical_levels:
                    # Calculate how well this condition matches (0.0 to 1.0)
                    level = chemical_levels[chemical]
                    if level >= threshold:
                        match_score += 1
                    else:
                        # Partial match based on percentage of threshold
                        match_score += level / threshold
            
            # Normalize match score (0.0 to 1.0)
            if conditions:
                match_score = match_score / len(conditions)
            else:
                match_score = 0
            
            # Apply rule weight to match score
            weighted_score = match_score * rule_weight
            
            # Only include emotions with non-zero intensity
            if weighted_score > 0:
                if emotion in emotion_intensities:
                    # Take the higher intensity if this emotion is already present
                    emotion_intensities[emotion] = max(emotion_intensities[emotion], weighted_score)
                else:
                    emotion_intensities[emotion] = weighted_score
        
        # Normalize emotion intensities to ensure they sum to a reasonable value
        total_intensity = sum(emotion_intensities.values())
        if total_intensity > 1.5:  # If total is too high, normalize
            factor = 1.5 / total_intensity
            emotion_intensities = {e: i * factor for e, i in emotion_intensities.items()}
        
        return emotion_intensities
    
    @function_tool
    async def _get_emotional_state_matrix(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the full emotional state matrix derived from neurochemicals
        
        Returns:
            Emotional state matrix with primary and secondary emotions
        """
        # First, apply decay to ensure current state
        await self._apply_chemical_decay(ctx)
        
        # Get derived emotions
        emotion_intensities = await self._derive_emotional_state(ctx)
        
        # Find primary emotion (highest intensity)
        primary_emotion = max(emotion_intensities.items(), key=lambda x: x[1]) if emotion_intensities else ("Neutral", 0.5)
        primary_name, primary_intensity = primary_emotion
        
        # Find secondary emotions (all others with significant intensity)
        secondary_emotions = {}
        for emotion, intensity in emotion_intensities.items():
            if emotion != primary_name and intensity > 0.3:  # Only include significant emotions
                # Find valence and arousal for this emotion from the rules
                valence = 0.0
                arousal = 0.5
                for rule in self.emotion_derivation_rules:
                    if rule["emotion"] == emotion:
                        valence = rule.get("valence", 0.0)
                        arousal = rule.get("arousal", 0.5)
                        break
                
                secondary_emotions[emotion] = {
                    "intensity": intensity,
                    "valence": valence,
                    "arousal": arousal
                }
        
        # Find valence and arousal for primary emotion
        primary_valence = 0.0
        primary_arousal = 0.5
        for rule in self.emotion_derivation_rules:
            if rule["emotion"] == primary_name:
                primary_valence = rule.get("valence", 0.0)
                primary_arousal = rule.get("arousal", 0.5)
                break
        
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
    async def _generate_internal_thought(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Generate an internal thought/reflection based on current emotional state
        
        Returns:
            Internal thought data
        """
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
        
        return {
            "thought_text": thought_text,
            "source_emotion": primary_emotion,
            "intensity": intensity,
            "insight_level": insight_level,
            "adaptive_change": adaptive_change,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    @function_tool
    async def _analyze_emotional_patterns(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Analyze patterns in emotional state history
        
        Returns:
            Analysis of emotional patterns
        """
        if len(self.emotional_state_history) < 2:
            return {
                "message": "Not enough emotional state history for pattern analysis",
                "patterns": {}
            }
        
        patterns = {}
        
        # Track emotion changes over time
        emotion_trends = defaultdict(list)
        for state in self.emotional_state_history:
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
        
        # Check for emotional oscillation
        oscillation_pairs = [
            ("Joy", "Sadness"),
            ("Trust", "Disgust"),
            ("Fear", "Anger"),
            ("Anticipation", "Surprise")
        ]
        
        for emotion1, emotion2 in oscillation_pairs:
            if emotion1 in emotion_trends and emotion2 in emotion_trends:
                # Count alternations between these emotions
                alternations = 0
                last_emotion = None
                
                for state in self.emotional_state_history:
                    if "primary_emotion" not in state:
                        continue
                        
                    current = state["primary_emotion"].get("name", "Neutral")
                    if current in (emotion1, emotion2):
                        if last_emotion and current != last_emotion:
                            alternations += 1
                        last_emotion = current
                
                if alternations > 1:
                    patterns[f"{emotion1}-{emotion2} oscillation"] = {
                        "alternations": alternations,
                        "significance": min(1.0, alternations / 5)  # Cap at 1.0
                    }
        
        return {
            "patterns": patterns,
            "history_size": len(self.emotional_state_history),
            "analysis_time": datetime.datetime.now().isoformat()
        }
    
    # Tool functions for the learning agent
    
    @function_tool
    async def _record_interaction_outcome(self, ctx: RunContextWrapper,
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
    async def _update_learning_rules(self, ctx: RunContextWrapper,
                               min_occurrences: int = 2) -> Dict[str, Any]:
        """
        Update learning rules based on observed patterns
        
        Args:
            min_occurrences: Minimum occurrences to consider a pattern significant
            
        Returns:
            Updated learning rules
        """
        new_rules = []
        
        # Process positive patterns
        for pattern, occurrences in self.reward_learning["positive_patterns"].items():
            if occurrences >= min_occurrences:
                # Check if a similar rule already exists
                existing = False
                for rule in self.reward_learning["learned_rules"]:
                    if rule["pattern"] == pattern and rule["outcome"] == "positive":
                        # Update existing rule
                        rule["strength"] = min(1.0, rule["strength"] + 0.1)
                        rule["occurrences"] = occurrences
                        existing = True
                        break
                
                if not existing:
                    # Create new rule
                    new_rules.append({
                        "pattern": pattern,
                        "outcome": "positive",
                        "strength": min(0.8, 0.3 + (occurrences * 0.1)),  # Start at 0.3, increase with occurrences
                        "occurrences": occurrences,
                        "created": datetime.datetime.now().isoformat()
                    })
        
        # Process negative patterns
        for pattern, occurrences in self.reward_learning["negative_patterns"].items():
            if occurrences >= min_occurrences:
                # Check if a similar rule already exists
                existing = False
                for rule in self.reward_learning["learned_rules"]:
                    if rule["pattern"] == pattern and rule["outcome"] == "negative":
                        # Update existing rule
                        rule["strength"] = min(1.0, rule["strength"] + 0.1)
                        rule["occurrences"] = occurrences
                        existing = True
                        break
                
                if not existing:
                    # Create new rule
                    new_rules.append({
                        "pattern": pattern,
                        "outcome": "negative",
                        "strength": min(0.8, 0.3 + (occurrences * 0.1)),
                        "occurrences": occurrences,
                        "created": datetime.datetime.now().isoformat()
                    })
        
        # Add new rules to learned rules
        self.reward_learning["learned_rules"].extend(new_rules)
        
        # Limit rules to prevent excessive growth
        if len(self.reward_learning["learned_rules"]) > 50:
            # Keep the most significant rules
            self.reward_learning["learned_rules"].sort(key=lambda x: x["strength"] * x["occurrences"], reverse=True)
            self.reward_learning["learned_rules"] = self.reward_learning["learned_rules"][:50]
        
        return {
            "new_rules": new_rules,
            "total_rules": len(self.reward_learning["learned_rules"]),
            "positive_patterns": len(self.reward_learning["positive_patterns"]),
            "negative_patterns": len(self.reward_learning["negative_patterns"])
        }
    
    @function_tool
    async def _apply_learned_adaptations(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Apply adaptations based on learned rules
        
        Returns:
            Adaptation results
        """
        if not self.reward_learning["learned_rules"]:
            return {
                "message": "No learned rules available for adaptation",
                "adaptations": 0
            }
        
        adaptations = []
        
        # Get current emotional state
        emotional_state = await self._get_emotional_state_matrix(ctx)
        current_emotion = emotional_state["primary_emotion"]["name"]
        
        # Find rules relevant to current emotional state
        relevant_rules = []
        for rule in self.reward_learning["learned_rules"]:
            # Check if rule mentions current emotion
            if current_emotion.lower() in rule["pattern"].lower():
                relevant_rules.append(rule)
        
        # Apply up to 2 adaptations
        for rule in relevant_rules[:2]:
            # Apply different adaptations based on outcome
            if rule["outcome"] == "positive":
                # For positive outcomes, reinforce the current state
                # Slightly increase the baselines for neurochemicals associated with this emotion
                for emotion_rule in self.emotion_derivation_rules:
                    if emotion_rule["emotion"] == current_emotion:
                        # Find key chemicals for this emotion
                        for chemical, threshold in emotion_rule["chemical_conditions"].items():
                            if chemical in self.neurochemicals:
                                # Increase baseline slightly
                                current_baseline = self.neurochemicals[chemical]["baseline"]
                                adjustment = rule["strength"] * 0.05  # Small adjustment based on rule strength
                                
                                # Don't adjust beyond sensible limits
                                new_baseline = min(0.8, current_baseline + adjustment)
                                
                                # Apply adjustment
                                self.neurochemicals[chemical]["baseline"] = new_baseline
                                
                                adaptations.append({
                                    "type": "baseline_increase",
                                    "chemical": chemical,
                                    "old_value": current_baseline,
                                    "new_value": new_baseline,
                                    "rule_pattern": rule["pattern"]
                                })
            else:
                # For negative outcomes, adjust the state away from current state
                # Slightly decrease the baselines for neurochemicals associated with this emotion
                for emotion_rule in self.emotion_derivation_rules:
                    if emotion_rule["emotion"] == current_emotion:
                        # Find key chemicals for this emotion
                        for chemical, threshold in emotion_rule["chemical_conditions"].items():
                            if chemical in self.neurochemicals:
                                # Decrease baseline slightly
                                current_baseline = self.neurochemicals[chemical]["baseline"]
                                adjustment = rule["strength"] * 0.05  # Small adjustment based on rule strength
                                
                                # Don't adjust beyond sensible limits
                                new_baseline = max(0.2, current_baseline - adjustment)
                                
                                # Apply adjustment
                                self.neurochemicals[chemical]["baseline"] = new_baseline
                                
                                adaptations.append({
                                    "type": "baseline_decrease",
                                    "chemical": chemical,
                                    "old_value": current_baseline,
                                    "new_value": new_baseline,
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
    
    def get_expression_for_emotion(self, emotion: Optional[str] = None) -> str:
        """Legacy API: Get a natural language expression for an emotion"""
        if emotion is None:
            # Get dominant emotion
            matrix = self._get_emotional_state_matrix_sync()
            emotion = matrix["primary_emotion"]["name"]
        
        # Attempt to generate an internal thought for this emotion
        reflection_patterns = self.reflection_patterns.get(emotion, [])
        
        if reflection_patterns:
            return random.choice(reflection_patterns)
        
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
        
        # Nyxamine (pleasure, curiosity) triggers
        if any(word in text_lower for word in ["happy", "good", "great", "love", "like", "fun", "enjoy", "curious", "interested"]):
            stimuli["nyxamine"] = 0.2
        
        # Seranix (calm, satisfaction) triggers
        if any(word in text_lower for word in ["calm", "peaceful", "relaxed", "content", "satisfied", "gentle", "quiet"]):
            stimuli["seranix"] = 0.2
        
        # OxyNixin (bonding, trust) triggers
        if any(word in text_lower for word in ["trust", "close", "together", "bond", "connect", "loyal", "friend", "relationship"]):
            stimuli["oxynixin"] = 0.2
        
        # Cortanyx (stress, anxiety) triggers
        if any(word in text_lower for word in ["worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry", "angry", "upset", "frustrated"]):
            stimuli["cortanyx"] = 0.2
        
        # Adrenyx (excitement, alertness) triggers
        if any(word in text_lower for word in ["excited", "alert", "surprised", "wow", "amazing", "intense", "sudden", "quick"]):
            stimuli["adrenyx"] = 0.2
        
        # Intensity modifiers
        intensifiers = ["very", "extremely", "incredibly", "so", "deeply", "absolutely"]
        if any(word in text_lower for word in intensifiers):
            for key in stimuli:
                stimuli[key] *= 1.5
        
        # Convert to traditional emotion format for backward compatibility
        emotion_stimuli = {}
        
        # Map chemical combinations to emotions
        if "nyxamine" in stimuli and "oxynixin" in stimuli:
            emotion_stimuli["Joy"] = (stimuli["nyxamine"] + stimuli["oxynixin"]) / 2
        
        if "cortanyx" in stimuli and "seranix" in stimuli:
            emotion_stimuli["Sadness"] = (stimuli["cortanyx"] + stimuli["seranix"]) / 2
        
        if "cortanyx" in stimuli and "adrenyx" in stimuli:
            emotion_stimuli["Fear"] = (stimuli["cortanyx"] + stimuli["adrenyx"]) / 2
        
        if "cortanyx" in stimuli and stimuli["cortanyx"] > 0.1:
            emotion_stimuli["Anger"] = stimuli["cortanyx"]
        
        if "oxynixin" in stimuli and stimuli["oxynixin"] > 0.1:
            emotion_stimuli["Trust"] = stimuli["oxynixin"]
        
        if "cortanyx" in stimuli and "oxynixin" in stimuli and stimuli["oxynixin"] < 0.1:
            emotion_stimuli["Disgust"] = stimuli["cortanyx"]
        
        if "adrenyx" in stimuli and "nyxamine" in stimuli:
            emotion_stimuli["Anticipation"] = (stimuli["adrenyx"] + stimuli["nyxamine"]) / 2
        
        if "adrenyx" in stimuli and stimuli["adrenyx"] > 0.2:
            emotion_stimuli["Surprise"] = stimuli["adrenyx"]
        
        if "oxynixin" in stimuli and "nyxamine" in stimuli and stimuli["oxynixin"] > 0.2:
            emotion_stimuli["Love"] = (stimuli["oxynixin"] + stimuli["nyxamine"]) / 2
        
        if "cortanyx" in stimuli and "nyxamine" in stimuli and stimuli["nyxamine"] < 0.1:
            emotion_stimuli["Frustration"] = stimuli["cortanyx"]
        
        # Return neutral if no matches
        if not emotion_stimuli:
            emotion_stimuli = {
                "Surprise": 0.05,
                "Anticipation": 0.05
            }
        
        return emotion_stimuli
    
    # Sync versions of async functions for compatibility
    
    def _derive_emotional_state_sync(self) -> Dict[str, float]:
        """Synchronous version of _derive_emotional_state for compatibility"""
        # Get current chemical levels
        chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
        
        # Calculate emotional intensities based on derivation rules
        emotion_intensities = {}
        
        for rule in self.emotion_derivation_rules:
            conditions = rule["chemical_conditions"]
            emotion = rule["emotion"]
            rule_weight = rule.get("weight", 1.0)
            
            # Check if chemical levels meet the conditions
            match_score = 0
            for chemical, threshold in conditions.items():
                if chemical in chemical_levels:
                    # Calculate how well this condition matches (0.0 to 1.0)
                    level = chemical_levels[chemical]
                    if level >= threshold:
                        match_score += 1
                    else:
                        # Partial match based on percentage of threshold
                        match_score += level / threshold
            
            # Normalize match score (0.0 to 1.0)
            if conditions:
                match_score = match_score / len(conditions)
            else:
                match_score = 0
            
            # Apply rule weight to match score
            weighted_score = match_score * rule_weight
            
            # Only include emotions with non-zero intensity
            if weighted_score > 0:
                if emotion in emotion_intensities:
                    # Take the higher intensity if this emotion is already present
                    emotion_intensities[emotion] = max(emotion_intensities[emotion], weighted_score)
                else:
                    emotion_intensities[emotion] = weighted_score
        
        # Normalize emotion intensities to ensure they sum to a reasonable value
        total_intensity = sum(emotion_intensities.values())
        if total_intensity > 1.5:  # If total is too high, normalize
            factor = 1.5 / total_intensity
            emotion_intensities = {e: i * factor for e, i in emotion_intensities.items()}
        
        return emotion_intensities
    
    def _get_emotional_state_matrix_sync(self) -> Dict[str, Any]:
        """Synchronous version of _get_emotional_state_matrix for compatibility"""
        # First apply decay
        self.apply_decay()
        
        # Get derived emotions
        emotion_intensities = self._derive_emotional_state_sync()
        
        # Find primary emotion (highest intensity)
        primary_emotion = max(emotion_intensities.items(), key=lambda x: x[1]) if emotion_intensities else ("Neutral", 0.5)
        primary_name, primary_intensity = primary_emotion
        
        # Find secondary emotions (all others with significant intensity)
        secondary_emotions = {}
        for emotion, intensity in emotion_intensities.items():
            if emotion != primary_name and intensity > 0.3:  # Only include significant emotions
                # Find valence and arousal for this emotion from the rules
                valence = 0.0
                arousal = 0.5
                for rule in self.emotion_derivation_rules:
                    if rule["emotion"] == emotion:
                        valence = rule.get("valence", 0.0)
                        arousal = rule.get("arousal", 0.5)
                        break
                
                secondary_emotions[emotion] = {
                    "intensity": intensity,
                    "valence": valence,
                    "arousal": arousal
                }
        
        # Find valence and arousal for primary emotion
        primary_valence = 0.0
        primary_arousal = 0.5
        for rule in self.emotion_derivation_rules:
            if rule["emotion"] == primary_name:
                primary_valence = rule.get("valence", 0.0)
                primary_arousal = rule.get("arousal", 0.5)
                break
        
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
        """Record current emotional state in history"""
        # Get current state
        state = self._get_emotional_state_matrix_sync()
        
        # Add to history
        self.emotional_state_history.append(state)
        
        # Limit history size
        if len(self.emotional_state_history) > self.max_history_size:
            self.emotional_state_history = self.emotional_state_history[-self.max_history_size:]
    
    # New enhanced public methods
    
    async def process_emotional_input(self, text: str) -> Dict[str, Any]:
        """
        Process input text through the DNM and update emotional state
        
        Args:
            text: Input text to process
            
        Returns:
            Processing results with updated emotional state
        """
        # Check for reflection trigger
        now = datetime.datetime.now()
        should_reflect = now > self.next_reflection_time
        
        # Use the neurochemical agent
        with trace(workflow_name="EmotionalInput"):
            try:
                # Analyze text for neurochemical impacts
                stimuli = {}
                text_lower = text.lower()
                
                # Enhanced pattern recognition for each neurochemical
                # Nyxamine (pleasure, curiosity) triggers
                if any(word in text_lower for word in ["happy", "good", "great", "love", "like", "fun", "enjoy", "curious", "interested"]):
                    stimuli["nyxamine"] = 0.2
                
                # Seranix (calm, satisfaction) triggers
                if any(word in text_lower for word in ["calm", "peaceful", "relaxed", "content", "satisfied", "gentle", "quiet"]):
                    stimuli["seranix"] = 0.2
                
                # OxyNixin (bonding, trust) triggers
                if any(word in text_lower for word in ["trust", "close", "together", "bond", "connect", "loyal", "friend", "relationship"]):
                    stimuli["oxynixin"] = 0.2
                
                # Cortanyx (stress, anxiety) triggers
                if any(word in text_lower for word in ["worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry", "angry", "upset", "frustrated"]):
                    stimuli["cortanyx"] = 0.2
                
                # Adrenyx (excitement, alertness) triggers
                if any(word in text_lower for word in ["excited", "alert", "surprised", "wow", "amazing", "intense", "sudden", "quick"]):
                    stimuli["adrenyx"] = 0.2
                
                # Update each affected neurochemical
                updated_chemicals = {}
                for chemical, value in stimuli.items():
                    result = await self._update_neurochemical(
                        RunContextWrapper(context=None),
                        chemical=chemical,
                        value=value
                    )
                    
                    if result.get("success", False):
                        updated_chemicals[chemical] = result
                
                # Get the derived emotional state
                emotional_state_matrix = await self._get_emotional_state_matrix(RunContextWrapper(context=None))
                
                # Generate internal reflection if due
                reflection = None
                if should_reflect:
                    reflection_result = await self._generate_internal_thought(RunContextWrapper(context=None))
                    reflection = reflection_result.get("thought_text")
                    
                    # Apply any adaptive change suggested
                    adaptive_change = reflection_result.get("adaptive_change")
                    if adaptive_change:
                        chemical = adaptive_change.get("chemical")
                        new_baseline = adaptive_change.get("suggested_baseline")
                        
                        if chemical in self.neurochemicals and new_baseline is not None:
                            self.neurochemicals[chemical]["baseline"] = new_baseline
                    
                    # Set next reflection time
                    self.next_reflection_time = now + datetime.timedelta(minutes=random.randint(5, 15))
                
                # Learn from this interaction if appropriate
                if len(self.emotional_state_history) >= 2:
                    # Get previous emotional state
                    prev_state = self.emotional_state_history[-2]
                    current_state = emotional_state_matrix
                    
                    # Evaluate if the change was positive or negative
                    prev_valence = prev_state.get("valence", 0)
                    current_valence = current_state.get("valence", 0)
                    
                    valence_change = current_valence - prev_valence
                    
                    # Record pattern with outcome
                    if abs(valence_change) > 0.2:  # Only record significant changes
                        pattern = f"Text input with {[key for key in stimuli.keys()]} triggers"
                        
                        if valence_change > 0:
                            await self._record_interaction_outcome(
                                RunContextWrapper(context=None),
                                interaction_pattern=pattern,
                                outcome="positive",
                                strength=min(1.0, abs(valence_change) * 2)
                            )
                        else:
                            await self._record_interaction_outcome(
                                RunContextWrapper(context=None),
                                interaction_pattern=pattern,
                                outcome="negative",
                                strength=min(1.0, abs(valence_change) * 2)
                            )
                        
                        # Periodically update learning rules
                        if random.random() < 0.2:  # 20% chance each time
                            await self._update_learning_rules(RunContextWrapper(context=None))
                
                return {
                    "input_processed": True,
                    "chemicals_affected": {c: v for c, v in stimuli.items() if v > 0},
                    "emotional_state": {
                        "primary_emotion": emotional_state_matrix["primary_emotion"],
                        "secondary_emotions": emotional_state_matrix["secondary_emotions"],
                        "valence": emotional_state_matrix["valence"],
                        "arousal": emotional_state_matrix["arousal"]
                    },
                    "internal_reflection": reflection
                }
                
            except Exception as e:
                logger.error(f"Error processing emotional input: {e}")
                
                # Fallback: use simple stimuli processing
                stimuli = self.analyze_text_sentiment(text)
                self.update_from_stimuli(stimuli)
                
                return {
                    "input_processed": True,
                    "detected_emotions": stimuli,
                    "updated_state": self.get_formatted_emotional_state(),
                    "error": str(e)
                }
    
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
        
        # Get emotional state matrix
        emotional_state = await self._get_emotional_state_matrix(RunContextWrapper(context=None))
        
        # Get primary emotion
        primary_emotion = emotional_state["primary_emotion"]["name"]
        intensity = emotional_state["primary_emotion"]["intensity"]
        
        # Generate internal thought as expression
        thought_result = await self._generate_internal_thought(RunContextWrapper(context=None))
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
        # Enhanced pattern recognition for each neurochemical
        chemical_impacts = {}
        text_lower = text.lower()
                
        # Nyxamine (pleasure, curiosity) triggers
        nyxamine_score = 0
        nyxamine_words = ["happy", "good", "great", "love", "like", "fun", "enjoy", "curious", "interested", "pleasure", "delight", "joy"]
        for word in nyxamine_words:
            if word in text_lower:
                nyxamine_score += 0.1
        if nyxamine_score > 0:
            chemical_impacts["nyxamine"] = min(0.5, nyxamine_score)
        
        # Seranix (calm, satisfaction) triggers
        seranix_score = 0
        seranix_words = ["calm", "peaceful", "relaxed", "content", "satisfied", "gentle", "quiet", "serene", "tranquil", "composed"]
        for word in seranix_words:
            if word in text_lower:
                seranix_score += 0.1
        if seranix_score > 0:
            chemical_impacts["seranix"] = min(0.5, seranix_score)
        
        # OxyNixin (bonding, trust) triggers
        oxynixin_score = 0
        oxynixin_words = ["trust", "close", "together", "bond", "connect", "loyal", "friend", "relationship", "intimate", "attachment"]
        for word in oxynixin_words:
            if word in text_lower:
                oxynixin_score += 0.1
        if oxynixin_score > 0:
            chemical_impacts["oxynixin"] = min(0.5, oxynixin_score)
        
        # Cortanyx (stress, anxiety) triggers
        cortanyx_score = 0
        cortanyx_words = ["worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry", "angry", "upset", "frustrated", "anxious", "distressed"]
        for word in cortanyx_words:
            if word in text_lower:
                cortanyx_score += 0.1
        if cortanyx_score > 0:
            chemical_impacts["cortanyx"] = min(0.5, cortanyx_score)
        
        # Adrenyx (excitement, alertness) triggers
        adrenyx_score = 0
        adrenyx_words = ["excited", "alert", "surprised", "wow", "amazing", "intense", "sudden", "quick", "shock", "unexpected", "startled"]
        for word in adrenyx_words:
            if word in text_lower:
                adrenyx_score += 0.1
        if adrenyx_score > 0:
            chemical_impacts["adrenyx"] = min(0.5, adrenyx_score)
        
        # Intensity modifiers
        intensifiers = ["very", "extremely", "incredibly", "so", "deeply", "absolutely", "truly", "utterly", "completely", "totally"]
        intensifier_count = sum(1 for word in intensifiers if word in text_lower)
        
        if intensifier_count > 0:
            intensity_multiplier = 1.0 + (intensifier_count * 0.2)  # Up to 1.0 + (5 * 0.2) = 2.0
            chemical_impacts = {k: min(1.0, v * intensity_multiplier) for k, v in chemical_impacts.items()}
        
        # If no chemicals were identified, add small baseline activation
        if not chemical_impacts:
            chemical_impacts = {
                "nyxamine": 0.1,
                "adrenyx": 0.1
            }
        
        # Derive emotions from these chemical impacts
        # Create a temporary neurochemical state for analysis
        temp_chemicals = {c: {"value": self.neurochemicals[c]["value"], 
                           "baseline": self.neurochemicals[c]["baseline"],
                           "decay_rate": self.neurochemicals[c]["decay_rate"]}
                         for c in self.neurochemicals}
        
        # Apply chemical impacts to the temporary state
        for chemical, impact in chemical_impacts.items():
            if chemical in temp_chemicals:
                temp_chemicals[chemical]["value"] = min(1.0, temp_chemicals[chemical]["value"] + impact)
        
        # Derive emotions from this temporary state
        derived_emotions = {}
        # Similar logic to _derive_emotional_state but using temp_chemicals
        chemical_levels = {c: d["value"] for c, d in temp_chemicals.items()}
        
        for rule in self.emotion_derivation_rules:
            conditions = rule["chemical_conditions"]
            emotion = rule["emotion"]
            rule_weight = rule.get("weight", 1.0)
            
            # Check if chemical levels meet the conditions
            match_score = 0
            for chemical, threshold in conditions.items():
                if chemical in chemical_levels:
                    # Calculate how well this condition matches (0.0 to 1.0)
                    level = chemical_levels[chemical]
                    if level >= threshold:
                        match_score += 1
                    else:
                        # Partial match based on percentage of threshold
                        match_score += level / threshold
            
            # Normalize match score (0.0 to 1.0)
            if conditions:
                match_score = match_score / len(conditions)
            else:
                match_score = 0
            
            # Apply rule weight to match score
            weighted_score = match_score * rule_weight
            
            # Only include emotions with non-zero intensity
            if weighted_score > 0:
                if emotion in derived_emotions:
                    # Take the higher intensity if this emotion is already present
                    derived_emotions[emotion] = max(derived_emotions[emotion], weighted_score)
                else:
                    derived_emotions[emotion] = weighted_score
        
        # Find dominant emotion
        dominant_emotion = max(derived_emotions.items(), key=lambda x: x[1]) if derived_emotions else ("neutral", 0.5)
        
        # Calculate overall valence from the temporary emotional state
        valence = 0.0
        total_intensity = 0.0
        
        for emotion, intensity in derived_emotions.items():
            # Find valence for this emotion
            for rule in self.emotion_derivation_rules:
                if rule["emotion"] == emotion:
                    valence += rule.get("valence", 0.0) * intensity
                    total_intensity += intensity
                    break
        
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
        # Analyze emotional patterns
        pattern_analysis = await self._analyze_emotional_patterns(RunContextWrapper(context=None))
        
        # Get current emotional state
        emotional_state = await self._get_emotional_state_matrix(RunContextWrapper(context=None))
        
        # Generate internal thought
        thought_result = await self._generate_internal_thought(RunContextWrapper(context=None))
        
        # Get learning statistics
        learning_stats = {
            "positive_patterns": len(self.reward_learning["positive_patterns"]),
            "negative_patterns": len(self.reward_learning["negative_patterns"]),
            "learned_rules": len(self.reward_learning["learned_rules"])
        }
        
        # Find dominant traits from emotional history
        dominant_traits = {}
        for state in self.emotional_state_history[-20:]:  # Look at last 20 states
            if "primary_emotion" in state:
                emotion = state["primary_emotion"].get("name")
                if emotion:
                    if emotion not in dominant_traits:
                        dominant_traits[emotion] = 0
                    dominant_traits[emotion] += 1
        
        # Sort by frequency
        dominant_traits = dict(sorted(dominant_traits.items(), key=lambda x: x[1], reverse=True)[:3])
        
        return {
            "introspection": thought_result.get("thought_text", "I'm currently processing my emotional state."),
            "current_emotion": emotional_state["primary_emotion"]["name"],
            "emotional_patterns": pattern_analysis.get("patterns"),
            "dominant_traits": dominant_traits,
            "learning_progress": learning_stats,
            "introspection_time": datetime.datetime.now().isoformat()
        }
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
        
        # Initialize timestamp
        self.init_time = datetime.datetime.now()
        
    @function_tool
    async def update_hormone_cycles(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Update hormone cycles based on elapsed time and environmental factors
        
        Returns:
            Updated hormone values
        """
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
            
            # Progress cycle phase based on elapsed time
            phase_change = (hours_elapsed / cycle_period) % 1.0
            new_phase = (old_phase + phase_change) % 1.0
            
            # Calculate cycle-based value using a sinusoidal pattern
            cycle_amplitude = 0.2  # How much the cycle affects the value
            cycle_influence = cycle_amplitude * math.sin(new_phase * 2 * math.pi)
            
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
        """Calculate environmental influence on a hormone"""
        
        # Different hormones respond to different environmental factors
        if hormone_name == "melatonyx":
            # Melatonyx is strongly affected by time of day (high at night, low at day)
            time_factor = 0.5 - self.environmental_factors["time_of_day"]
            return time_factor * 0.4  # Up to 0.4 variation
            
        elif hormone_name == "oxytonyx":
            # Oxytonyx increases with user familiarity and positive interactions
            familiarity = self.environmental_factors["user_familiarity"]
            quality = self.environmental_factors["interaction_quality"]
            return (familiarity * 0.3) + (quality * 0.2)
            
        elif hormone_name == "endoryx":
            # Endoryx responds to interaction quality
            quality = self.environmental_factors["interaction_quality"]
            return (quality - 0.5) * 0.4  # -0.2 to +0.2
            
        elif hormone_name == "estradyx":
            # Estradyx has a complex monthly cycle, with minor environmental influence
            return (self.environmental_factors["interaction_quality"] - 0.5) * 0.1
            
        elif hormone_name == "testoryx":
            # Testoryx has a diurnal cycle with peaks in morning, affected by session length
            time_factor = 0.5 - abs(self.environmental_factors["time_of_day"] - 0.25)
            session_factor = self.environmental_factors["session_duration"] * 0.1
            return (time_factor * 0.3) - session_factor
        
        return 0.0
        
    @function_tool
    async def _update_hormone_influences(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Update neurochemical influences from hormones
        
        Returns:
            Updated influence values
        """
        # Skip if no emotional core
        if not self.emotional_core:
            return {
                "message": "No emotional core available",
                "influences": {}
            }
        
        # Initialize hormone influences
        hormone_influences = {
            "nyxamine": 0.0,
            "seranix": 0.0,
            "oxynixin": 0.0,
            "cortanyx": 0.0,
            "adrenyx": 0.0
        }
        
        # Calculate influences from each hormone
        for hormone_name, hormone_data in self.hormones.items():
            # Skip if hormone has no influence mapping
            if hormone_name not in self.hormone_neurochemical_influences:
                continue
                
            hormone_value = hormone_data["value"]
            hormone_influences = self.hormone_neurochemical_influences[hormone_name]
            
            # Apply influences based on hormone value
            for chemical, influence_factor in hormone_influences.items():
                if chemical in self.emotional_core.neurochemicals:
                    # Calculate scaled influence
                    scaled_influence = influence_factor * (hormone_value - 0.5) * 2
                    
                    # Get original baseline
                    original_baseline = self.emotional_core.neurochemicals[chemical]["baseline"]
                    
                    # Add temporary hormone influence
                    temporary_baseline = max(0.1, min(0.9, original_baseline + scaled_influence))
                    
                    # Record influence but don't permanently change baseline
                    self.emotional_core.neurochemicals[chemical]["temporary_baseline"] = temporary_baseline
        
        return {
            "applied_influences": hormone_influences
        }
