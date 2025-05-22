# nyx/core/input_processing_config.py
from typing import Dict, Any
from pydantic import BaseModel, Field

class InputProcessingConfig(BaseModel):
    """Unified configuration for input processing systems"""
    
    # Core settings
    pattern_sensitivity_base: float = Field(0.5, ge=0.0, le=1.0)
    mode_blending_enabled: bool = True
    context_distribution_enabled: bool = True
    conditioning_enabled: bool = True
    
    # Pattern detection thresholds
    pattern_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "submission_language": 0.6,
        "defiance": 0.7,
        "flattery": 0.5,
        "disrespect": 0.8,
        "embarrassment": 0.5
    })
    
    # Behavior selection weights
    behavior_weights: Dict[str, float] = Field(default_factory=lambda: {
        "dominant_response": 1.0,
        "teasing_response": 0.8,
        "nurturing_response": 0.9,
        "direct_response": 0.7,
        "playful_response": 0.8,
        "strict_response": 0.9
    })
    
    # Context influence settings
    emotional_influence_strength: float = Field(0.3, ge=0.0, le=1.0)
    mode_influence_strength: float = Field(0.4, ge=0.0, le=1.0)
    relationship_influence_strength: float = Field(0.35, ge=0.0, le=1.0)
    
    # Processing limits
    max_pattern_detections: int = 10
    max_behavior_evaluations: int = 6
    context_update_batch_size: int = 5
    
    class Config:
        validate_assignment = True
