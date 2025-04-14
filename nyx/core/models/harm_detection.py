# nyx/core/models/harm_detection.py

from pydantic import BaseModel, Field
from typing import List, Optional

class HarmIntentDetectionInput(BaseModel):
    """Input for harm intent detection"""
    text: str = Field(..., description="Text to analyze for harmful intent")
    context: Optional[str] = Field(None, description="Optional context for improved detection")
    action_type: Optional[str] = Field(None, description="Type of action if known (physical, verbal, etc.)")
    
class HarmIntentDetectionOutput(BaseModel):
    """Output from harm intent detection"""
    is_harmful: bool = Field(..., description="Whether harmful intent was detected")
    confidence: float = Field(..., description="Confidence in the detection (0.0-1.0)")
    detected_terms: List[str] = Field(default_factory=list, description="Terms that triggered detection")
    harmful_type: Optional[str] = Field(None, description="Type of harm (physical, emotional, etc.)")
    severity: Optional[float] = Field(None, description="Estimated severity of harm (0.0-1.0)")
    method: str = Field("keyword_detection", description="Method used for detection")
    
class ProtectedResponseOutput(BaseModel):
    """Output for a protected response to harmful content"""
    protected: bool = Field(..., description="Whether protection was applied")
    original_stimulus: dict = Field(..., description="Original stimulus that was protected against")
    detection_result: HarmIntentDetectionOutput = Field(..., description="Results of harm detection")
    message: str = Field(..., description="Message explaining the protection")
    response_suggestion: str = Field(..., description="Suggested character response")

