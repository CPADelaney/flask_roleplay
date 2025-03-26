# nyx/core/emotions/schemas.py
"""
Schema definitions for the Nyx emotional system.
Contains all Pydantic models used throughout the emotional core.

"""

import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set, TypedDict

from pydantic import BaseModel, Field, validator

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
    arousal: float = Field(..., ge=0.0, le=1.0)c
