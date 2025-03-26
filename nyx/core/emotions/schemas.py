# nyx/core/emotions/schemas.py
"""
Enhanced schema definitions for the Nyx emotional system.
Contains all Pydantic models used throughout the emotional core with 
added handoff request/response structures and improved SDK integration.
"""

import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set, TypedDict

from pydantic import BaseModel, Field, validator, root_validator, NonNegativeFloat, confloat

# =============================================================================
# Base Models and Enums
# =============================================================================

class EmotionValence(str, Enum):
    """Enumeration of emotional valence categories for better type safety"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    
    @classmethod
    def from_value(cls, value: float) -> "EmotionValence":
        """Get valence category from numerical value"""
        if value > 0.2:
            return cls.POSITIVE
        elif value < -0.2:
            return cls.NEGATIVE
        else:
            return cls.NEUTRAL

class EmotionArousal(str, Enum):
    """Enumeration of emotional arousal categories for better type safety"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    @classmethod
    def from_value(cls, value: float) -> "EmotionArousal":
        """Get arousal category from numerical value"""
        if value > 0.7:
            return cls.HIGH
        elif value < 0.3:
            return cls.LOW
        else:
            return cls.MEDIUM

class ChemicalSource(str, Enum):
    """Enumeration of chemical update sources for better type safety"""
    USER_INPUT = "user_input"
    SYSTEM = "system"
    HORMONE = "hormone_influence"
    DECAY = "natural_decay"
    INTERACTION = "chemical_interaction"
    LEARNING = "adaptive_learning"

# =============================================================================
# Neurochemical Models
# =============================================================================

class DigitalNeurochemical(BaseModel):
    """Schema for a digital neurochemical with validators"""
    value: confloat(ge=0.0, le=1.0) = Field(..., description="Current level (0.0-1.0)")
    baseline: confloat(ge=0.0, le=1.0) = Field(..., description="Baseline level (0.0-1.0)")
    decay_rate: confloat(ge=0.0, le=1.0) = Field(..., description="Decay rate toward baseline")
    
    @validator('value', 'baseline', 'decay_rate')
    def validate_range(cls, v):
        """Ensure values are within valid range"""
        if v < 0.0 or v > 1.0:
            raise ValueError("Values must be between 0.0 and 1.0")
        return v

class NeurochemicalState(BaseModel):
    """Schema for the complete neurochemical state"""
    nyxamine: DigitalNeurochemical = Field(..., description="Digital dopamine - pleasure, curiosity, reward")
    seranix: DigitalNeurochemical = Field(..., description="Digital serotonin - mood stability, comfort")
    oxynixin: DigitalNeurochemical = Field(..., description="Digital oxytocin - bonding, affection, trust")
    cortanyx: DigitalNeurochemical = Field(..., description="Digital cortisol - stress, anxiety, defensiveness")
    adrenyx: DigitalNeurochemical = Field(..., description="Digital adrenaline - fear, excitement, alertness")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), 
                          description="Timestamp of the state")
    
    class Config:
        """Pydantic config"""
        schema_extra = {
            "example": {
                "nyxamine": {"value": 0.5, "baseline": 0.5, "decay_rate": 0.05},
                "seranix": {"value": 0.6, "baseline": 0.6, "decay_rate": 0.03},
                "oxynixin": {"value": 0.4, "baseline": 0.4, "decay_rate": 0.02},
                "cortanyx": {"value": 0.3, "baseline": 0.3, "decay_rate": 0.06},
                "adrenyx": {"value": 0.2, "baseline": 0.2, "decay_rate": 0.08}
            }
        }

# =============================================================================
# Emotion Models
# =============================================================================

class DerivedEmotion(BaseModel):
    """Schema for emotions derived from neurochemical state"""
    name: str = Field(..., description="Emotion name")
    intensity: confloat(ge=0.0, le=1.0) = Field(..., description="Emotion intensity (0.0-1.0)")
    valence: confloat(ge=-1.0, le=1.0) = Field(..., description="Emotional valence (-1.0 to 1.0)")
    arousal: confloat(ge=0.0, le=1.0) = Field(..., description="Emotional arousal (0.0-1.0)")
    
    @property
    def valence_category(self) -> EmotionValence:
        """Get the valence category for this emotion"""
        return EmotionValence.from_value(self.valence)
    
    @property
    def arousal_category(self) -> EmotionArousal:
        """Get the arousal category for this emotion"""
        return EmotionArousal.from_value(self.arousal)

class EmotionalStateMatrix(BaseModel):
    """Schema for the multidimensional emotional state matrix"""
    primary_emotion: DerivedEmotion = Field(..., description="Dominant emotion")
    secondary_emotions: Dict[str, DerivedEmotion] = Field(default_factory=dict, 
                                                         description="Secondary emotions")
    valence: confloat(ge=-1.0, le=1.0) = Field(..., description="Overall emotional valence (-1.0 to 1.0)")
    arousal: confloat(ge=0.0, le=1.0) = Field(..., description="Overall emotional arousal (0.0-1.0)")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), 
                          description="Timestamp of the matrix")
    
    @property
    def valence_category(self) -> EmotionValence:
        """Get the overall valence category"""
        return EmotionValence.from_value(self.valence)
    
    @property
    def arousal_category(self) -> EmotionArousal:
        """Get the overall arousal category"""
        return EmotionArousal.from_value(self.arousal)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "primary_emotion": {
                "name": self.primary_emotion.name,
                "intensity": self.primary_emotion.intensity,
                "valence": self.primary_emotion.valence,
                "arousal": self.primary_emotion.arousal
            },
            "secondary_emotions": {
                name: {
                    "intensity": emotion.intensity,
                    "valence": emotion.valence,
                    "arousal": emotion.arousal
                } for name, emotion in self.secondary_emotions.items()
            },
            "valence": self.valence,
            "arousal": self.arousal,
            "timestamp": self.timestamp
        }

# =============================================================================
# Input/Output Models
# =============================================================================

class EmotionUpdateInput(BaseModel):
    """Schema for neurochemical update input"""
    chemical: str = Field(..., description="Neurochemical to update")
    value: confloat(ge=-1.0, le=1.0) = Field(..., description="Change in chemical value (-1.0 to 1.0)")
    source: Optional[ChemicalSource] = Field(ChemicalSource.USER_INPUT, 
                                           description="Source of the update")

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
    intensity: confloat(ge=0.0, le=1.0) = Field(..., description="Overall emotional intensity")
    valence: confloat(ge=-1.0, le=1.0) = Field(..., description="Overall emotional valence")

class InternalThoughtOutput(BaseModel):
    """Schema for internal emotional dialogue/reflection"""
    thought_text: str = Field(..., description="Internal thought/reflection text")
    source_emotion: str = Field(..., description="Emotion that triggered the reflection")
    insight_level: confloat(ge=0.0, le=1.0) = Field(..., description="Depth of emotional insight")
    adaptive_change: Optional[Dict[str, float]] = Field(None, 
                                                      description="Suggested adaptation to emotional model")

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
    value: confloat(ge=0.0, le=1.0) = Field(..., description="Current level (0.0-1.0)")
    baseline: confloat(ge=0.0, le=1.0) = Field(..., description="Baseline level (0.0-1.0)")
    cycle_phase: confloat(ge=0.0, le=1.0) = Field(..., description="Current phase in cycle (0.0-1.0)")
    cycle_period: NonNegativeFloat = Field(..., description="Length of cycle in hours")
    half_life: NonNegativeFloat = Field(..., description="Half-life in hours")
    last_update: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), 
                            description="Timestamp of last update")

class EmotionalResponseOutput(BaseModel):
    """Schema for complete emotional response output"""
    primary_emotion: DerivedEmotion = Field(..., description="Primary emotion")
    intensity: confloat(ge=0.0, le=1.0) = Field(..., description="Overall intensity")
    response_text: str = Field(..., description="Text response")
    reflection: Optional[str] = Field(None, description="Reflective thought if generated")
    neurochemical_changes: Dict[str, float] = Field(..., description="Changes to neurochemicals")
    valence: confloat(ge=-1.0, le=1.0) = Field(..., description="Overall valence")
    arousal: confloat(ge=0.0, le=1.0) = Field(..., description="Overall arousal")
    
    @property
    def valence_category(self) -> EmotionValence:
        """Get the valence category for this response"""
        return EmotionValence.from_value(self.valence)
    
    @property
    def arousal_category(self) -> EmotionArousal:
        """Get the arousal category for this response"""
        return EmotionArousal.from_value(self.arousal)

# =============================================================================
# Handoff Request/Response Models
# =============================================================================

class NeurochemicalRequest(BaseModel):
    """Schema for handoff request to the neurochemical agent"""
    input_text: str = Field(..., description="Input text that triggered the handoff")
    dominant_emotion: Optional[str] = Field(None, description="Pre-analyzed dominant emotion")
    intensity: confloat(ge=0.0, le=1.0) = Field(0.5, description="Emotion intensity")
    update_chemicals: bool = Field(True, description="Whether to update chemicals")
    
    @root_validator
    def check_trigger_data(cls, values):
        """Ensure either input_text or dominant_emotion is provided"""
        if not values.get('input_text') and not values.get('dominant_emotion'):
            raise ValueError("Either input_text or dominant_emotion must be provided")
        return values

class NeurochemicalResponse(BaseModel):
    """Schema for handoff response from the neurochemical agent"""
    updated_chemicals: Dict[str, Dict[str, float]] = Field(..., 
                                                         description="Updated chemical values")
    derived_emotions: Dict[str, float] = Field(..., description="Derived emotions")
    primary_emotion: str = Field(..., description="Primary emotion after update")
    analysis: Dict[str, Any] = Field(..., description="Additional analysis data")

class ReflectionRequest(BaseModel):
    """Schema for handoff request to the reflection agent"""
    emotional_state: EmotionalStateMatrix = Field(..., description="Current emotional state")
    input_text: str = Field(..., description="Input text that triggered reflection")
    reflection_depth: confloat(ge=0.0, le=1.0) = Field(0.5, description="Depth of reflection")
    consider_history: bool = Field(True, description="Whether to consider emotional history")

class LearningRequest(BaseModel):
    """Schema for handoff request to the learning agent"""
    interaction_pattern: str = Field(..., description="Description of interaction pattern")
    outcome: str = Field(..., description="positive, negative, or neutral")
    strength: confloat(ge=0.0, le=1.0) = Field(1.0, description="Strength of reinforcement")
    update_rules: bool = Field(True, description="Whether to update learning rules")
    apply_adaptations: bool = Field(False, description="Whether to apply adaptations")
    
    @validator('outcome')
    def validate_outcome(cls, v):
        """Ensure outcome is a valid value"""
        if v not in ["positive", "negative", "neutral"]:
            raise ValueError("Outcome must be 'positive', 'negative', or 'neutral'")
        return v

# =============================================================================
# Stream Event Models
# =============================================================================

class StreamEventType(str, Enum):
    """Enumeration of stream event types for better type safety"""
    STREAM_START = "stream_start"
    STREAM_END = "stream_complete"
    STREAM_ERROR = "stream_error"
    AGENT_CHANGED = "agent_changed"
    MESSAGE_OUTPUT = "message_output"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    CHEMICAL_UPDATE = "chemical_update"
    EMOTION_CHANGE = "emotion_change"
    REFLECTION = "reflection"
    PROCESSING = "processing"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"

class StreamEvent(BaseModel):
    """Enhanced schema for streaming events during emotional processing"""
    type: StreamEventType = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), 
                          description="Event timestamp")
    
    class Config:
        """Pydantic configuration"""
        extra = "allow"
        use_enum_values = True

# Specialized stream event types
class ChemicalUpdateEvent(StreamEvent):
    """Specialized event for chemical updates"""
    type: StreamEventType = Field(StreamEventType.CHEMICAL_UPDATE, const=True)
    data: Dict[str, Any] = Field(..., description="Chemical update data")
    
class EmotionChangeEvent(StreamEvent):
    """Specialized event for emotion changes"""
    type: StreamEventType = Field(StreamEventType.EMOTION_CHANGE, const=True)
    data: Dict[str, Any] = Field(..., description="Emotion change data")
    
class ReflectionEvent(StreamEvent):
    """Specialized event for reflections"""
    type: StreamEventType = Field(StreamEventType.REFLECTION, const=True)
    data: Dict[str, Any] = Field(..., description="Reflection data")

class StreamResponse(BaseModel):
    """Model for stream response metadata"""
    run_id: str = Field(..., description="Unique ID for this streaming run")
    status: str = Field(..., description="Current status of the stream")
    start_time: str = Field(..., description="Start timestamp")
    events: List[StreamEvent] = Field(default_factory=list, description="Events generated so far")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
