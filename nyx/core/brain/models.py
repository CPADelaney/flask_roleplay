# nyx/core/brain/models.py
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    """Input from a user to be processed"""
    user_id: int = Field(..., description="User ID")
    text: str = Field(..., description="User's input text")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ProcessResult(BaseModel):
    """Result of processing user input"""
    user_input: str = Field(..., description="Original user input")
    emotional_state: Dict[str, Any] = Field(..., description="Current emotional state")
    memories: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved memories")
    memory_count: int = Field(0, description="Number of memories retrieved")
    has_experience: bool = Field(False, description="Whether an experience was found")
    experience_response: Optional[str] = Field(None, description="Experience response if available")
    cross_user_experience: bool = Field(False, description="Whether experience is from another user")
    memory_id: Optional[str] = Field(None, description="ID of stored memory")
    response_time: float = Field(0.0, description="Processing time in seconds")
    context_change: Optional[Dict[str, Any]] = Field(None, description="Context change detection")
    identity_impact: Optional[Dict[str, Any]] = Field(None, description="Impact on identity")

class ResponseResult(BaseModel):
    """Result of generating a response"""
    message: str = Field(..., description="Main response message")
    response_type: str = Field(..., description="Type of response")
    emotional_state: Dict[str, Any] = Field(..., description="Current emotional state")
    emotional_expression: Optional[str] = Field(None, description="Emotional expression if any")
    memories_used: List[str] = Field(default_factory=list, description="IDs of memories used")
    memory_count: int = Field(0, description="Number of memories used")
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Response evaluation if available")
    experience_sharing_adapted: bool = Field(False, description="Whether experience sharing was adapted")

class AdaptationResult(BaseModel):
    """Result of adaptation process"""
    strategy_id: str = Field(..., description="ID of the selected strategy")
    context_change: Dict[str, Any] = Field(..., description="Context change information")
    confidence: float = Field(..., description="Confidence in strategy selection")
    adaptations: Dict[str, Any] = Field(..., description="Applied adaptations")

class IdentityState(BaseModel):
    """Current state of Nyx's identity"""
    top_preferences: Dict[str, float] = Field(..., description="Top preferences with scores")
    top_traits: Dict[str, float] = Field(..., description="Top traits with scores")
    identity_reflection: str = Field(..., description="Reflection on identity")
    identity_evolution: Dict[str, Any] = Field(..., description="Identity evolution metrics")

class StimulusData(BaseModel):
    """Input stimulus that may trigger a reflexive response"""
    data: Dict[str, Any] = Field(..., description="Stimulus data patterns")
    domain: Optional[str] = Field(None, description="Optional domain to limit patterns")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    priority: Optional[str] = Field("normal", description="Processing priority (normal, high, critical)")

class ReflexRegistrationInput(BaseModel):
    """Input for registering a new reflex pattern"""
    name: str = Field(..., description="Unique name for this pattern")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern definition data")
    procedure_name: str = Field(..., description="Name of procedure to execute when triggered")
    threshold: float = Field(0.7, description="Matching threshold (0.0-1.0)")
    priority: int = Field(1, description="Priority level (higher values take precedence)")
    domain: Optional[str] = Field(None, description="Optional domain for specialized responses")
    context_template: Optional[Dict[str, Any]] = Field(None, description="Template for context to pass to procedure")

class ReflexResponse(BaseModel):
    """Result of processing a stimulus with reflexes"""
    success: bool = Field(..., description="Whether a reflex was successfully triggered")
    pattern_name: Optional[str] = Field(None, description="Name of the triggered pattern if successful")
    reaction_time_ms: float = Field(..., description="Reaction time in milliseconds")
    output: Optional[Dict[str, Any]] = Field(None, description="Output from the procedure execution")
    match_score: Optional[float] = Field(None, description="Match score for the pattern")

class SensoryInput(BaseModel):
    """Input from a sensory modality"""
    modality: str = Field(..., description="Sensory modality type (text, image, etc.)")
    data: Any = Field(..., description="The actual sensory input data")
    confidence: float = Field(1.0, description="Confidence in this input (0.0-1.0)")
    timestamp: str = Field(..., description="Timestamp of the input")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ExpectationSignal(BaseModel):
    """Top-down expectation signal for perception modulation"""
    source: str = Field(..., description="Source of this expectation")
    target_modality: str = Field(..., description="Target modality to influence")
    pattern: Any = Field(..., description="Expected pattern")
    strength: float = Field(0.5, description="Strength of expectation (0.0-1.0)")
    duration: Optional[int] = Field(None, description="Duration of this expectation (in interactions)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
