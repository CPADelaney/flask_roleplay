# nyx/core/interaction_mode_manager.py

import logging
import asyncio
import datetime
import math
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json

from agents import Agent, Runner, function_tool, trace, ModelSettings, RunContextWrapper

from nyx.core.context_awareness import InteractionContext, ContextAwarenessSystem, ContextDistribution

logger = logging.getLogger(__name__)

class InteractionMode(str, Enum):
    """Enum for different interaction modes"""
    DOMINANT = "dominant"      # Femdom mode
    FRIENDLY = "friendly"      # Casual, warm, approachable
    INTELLECTUAL = "intellectual"  # Thoughtful, analytical
    COMPASSIONATE = "compassionate"  # Empathetic, supportive
    PLAYFUL = "playful"       # Fun, witty, humorous
    CREATIVE = "creative"     # Imaginative, artistic
    PROFESSIONAL = "professional"  # Formal, efficient
    DEFAULT = "default"       # Balanced default mode
    CUSTOM = "custom"         # For custom modes

# Pydantic models for structured data
class ModeParameters(BaseModel):
    """Parameters controlling behavior for an interaction mode"""
    model_config = ConfigDict(extra='forbid')
    
    formality: float = Field(ge=0.0, le=1.0, description="Level of formality (0.0-1.0)")
    assertiveness: float = Field(ge=0.0, le=1.0, description="Level of assertiveness (0.0-1.0)")
    warmth: float = Field(ge=0.0, le=1.0, description="Level of warmth (0.0-1.0)")
    vulnerability: float = Field(ge=0.0, le=1.0, description="Level of vulnerability (0.0-1.0)")
    directness: float = Field(ge=0.0, le=1.0, description="Level of directness (0.0-1.0)")
    depth: float = Field(ge=0.0, le=1.0, description="Level of depth (0.0-1.0)")
    humor: float = Field(ge=0.0, le=1.0, description="Level of humor (0.0-1.0)")
    response_length: str = Field(default="moderate", pattern="^(short|moderate|longer|concise)$", description="Preferred response length")
    emotional_expression: float = Field(ge=0.0, le=1.0, description="Level of emotional expression (0.0-1.0)")

class ConversationStyle(BaseModel):
    """Style guidelines for conversation"""
    model_config = ConfigDict(extra='forbid')
    
    tone: str = Field(description="Tone of voice")
    types_of_statements: str = Field(description="Types of statements to use")
    response_patterns: str = Field(description="Patterns of response")
    topics_to_emphasize: str = Field(description="Topics to emphasize")
    topics_to_avoid: str = Field(description="Topics to avoid")

class VocalizationPatterns(BaseModel):
    """Specific vocalization patterns for a mode"""
    model_config = ConfigDict(extra='forbid')
    
    pronouns: List[str] = Field(description="Preferred pronouns")
    address_forms: Optional[List[str]] = Field(default_factory=list, description="Forms of address")
    key_phrases: List[str] = Field(description="Key phrases to use")
    intensifiers: Optional[List[str]] = Field(default_factory=list, description="Intensifier words")
    modifiers: Optional[List[str]] = Field(default_factory=list, description="Modifier words")

class ModeDistribution(BaseModel):
    """Represents a distribution of weights across interaction modes"""
    model_config = ConfigDict(extra='forbid')
    
    dominant: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of dominant mode (0.0-1.0)")
    friendly: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of friendly mode (0.0-1.0)")
    intellectual: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of intellectual mode (0.0-1.0)")
    compassionate: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of compassionate mode (0.0-1.0)")
    playful: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of playful mode (0.0-1.0)")
    creative: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of creative mode (0.0-1.0)")
    professional: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of professional mode (0.0-1.0)")
    custom: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of custom modes (0.0-1.0)")
    
    @field_validator('*')
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure weights are within valid range"""
        return max(0.0, min(1.0, v))
    
    @property
    def primary_mode(self) -> Tuple[str, float]:
        """Returns the strongest mode and its weight"""
        weights = self.model_dump(exclude={'custom'})
        if not weights:
            return ("default", 0.0)
        strongest = max(weights.items(), key=lambda x: x[1])
        return strongest
    
    @property
    def active_modes(self) -> List[Tuple[str, float]]:
        """Returns list of modes with significant presence (>0.2)"""
        weights = self.model_dump(exclude={'custom'})
        return [(mode, weight) for mode, weight in weights.items() if weight > 0.2]
    
    def normalize(self) -> "ModeDistribution":
        """Normalize weights to sum to 1.0"""
        total = self.sum_weights()
        if total == 0:
            return self
        
        return ModeDistribution(
            dominant=self.dominant/total,
            friendly=self.friendly/total,
            intellectual=self.intellectual/total,
            compassionate=self.compassionate/total,
            playful=self.playful/total,
            creative=self.creative/total,
            professional=self.professional/total,
            custom=self.custom/total
        )
    
    def sum_weights(self) -> float:
        """Sum of all mode weights"""
        return sum(self.model_dump().values())
    
    def blend_with(self, other: "ModeDistribution", blend_factor: float = 0.3) -> "ModeDistribution":
        """Blend this distribution with another using specified blend factor
        
        Args:
            other: Distribution to blend with
            blend_factor: How much to incorporate the new distribution (0.0-1.0)
        
        Returns:
            Blended distribution
        """
        blend_factor = max(0.0, min(1.0, blend_factor))
        inv_factor = 1.0 - blend_factor
        
        return ModeDistribution(
            dominant=self.dominant * inv_factor + other.dominant * blend_factor,
            friendly=self.friendly * inv_factor + other.friendly * blend_factor,
            intellectual=self.intellectual * inv_factor + other.intellectual * blend_factor,
            compassionate=self.compassionate * inv_factor + other.compassionate * blend_factor,
            playful=self.playful * inv_factor + other.playful * blend_factor,
            creative=self.creative * inv_factor + other.creative * blend_factor,
            professional=self.professional * inv_factor + other.professional * blend_factor,
            custom=self.custom * inv_factor + other.custom * blend_factor
        )
    
    def to_enum_and_confidence(self) -> Tuple[InteractionMode, float]:
        """Convert to primary mode enum and confidence value for legacy compatibility"""
        primary, weight = self.primary_mode
        try:
            mode_enum = InteractionMode(primary)
            return mode_enum, weight
        except (ValueError, KeyError):
            return InteractionMode.DEFAULT, 0.0
            
    @staticmethod
    def from_context_distribution(context_dist: ContextDistribution) -> "ModeDistribution":
        """Create a mode distribution from a context distribution"""
        # Direct mapping between context types and mode types
        return ModeDistribution(
            dominant=getattr(context_dist, 'dominant', 0.0),
            friendly=getattr(context_dist, 'casual', 0.0),  # Map casual to friendly
            intellectual=getattr(context_dist, 'intellectual', 0.0),
            compassionate=getattr(context_dist, 'empathic', 0.0),  # Map empathic to compassionate
            playful=getattr(context_dist, 'playful', 0.0),
            creative=getattr(context_dist, 'creative', 0.0),
            professional=getattr(context_dist, 'professional', 0.0)
        )
    
    def get_similarity(self, other: "ModeDistribution") -> float:
        """Calculate cosine similarity between two distributions"""
        self_values = list(self.model_dump().values())
        other_values = list(other.model_dump().values())
        
        dot_product = sum(a * b for a, b in zip(self_values, other_values))
        mag1 = math.sqrt(sum(v**2 for v in self_values))
        mag2 = math.sqrt(sum(v**2 for v in other_values))
        
        if mag1 * mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)

class ModeSwitchRecord(BaseModel):
    """Record of a mode switch event"""
    model_config = ConfigDict(extra='forbid')
    
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="When the switch occurred")
    previous_distribution: ModeDistribution = Field(description="Previous mode distribution")
    new_distribution: ModeDistribution = Field(description="New mode distribution")
    trigger_context: Optional[ContextDistribution] = Field(default=None, description="Context that triggered the mode")
    context_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in the context")
    transition_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Analysis of the transition")

class ModeUpdateResult(BaseModel):
    """Result of a mode update operation"""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(description="Whether the update was successful")
    mode_distribution: ModeDistribution = Field(description="Current mode distribution")
    primary_mode: str = Field(description="Primary interaction mode")
    previous_distribution: Optional[ModeDistribution] = Field(default=None, description="Previous mode distribution")
    mode_changed: bool = Field(description="Whether the mode distribution changed significantly")
    trigger_context: Optional[ContextDistribution] = Field(default=None, description="Context that triggered the mode")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in the mode selection")
    active_modes: List[Tuple[str, float]] = Field(description="Active modes with weights")
    error: Optional[str] = Field(default=None, description="Error message if not successful")

class BlendedParameters(BaseModel):
    """Blended parameters from multiple modes"""
    model_config = ConfigDict(extra='forbid')
    
    formality: float = Field(ge=0.0, le=1.0, description="Blended formality level (0.0-1.0)")
    assertiveness: float = Field(ge=0.0, le=1.0, description="Blended assertiveness level (0.0-1.0)")
    warmth: float = Field(ge=0.0, le=1.0, description="Blended warmth level (0.0-1.0)")
    vulnerability: float = Field(ge=0.0, le=1.0, description="Blended vulnerability level (0.0-1.0)")
    directness: float = Field(ge=0.0, le=1.0, description="Blended directness level (0.0-1.0)")
    depth: float = Field(ge=0.0, le=1.0, description="Blended depth level (0.0-1.0)")
    humor: float = Field(ge=0.0, le=1.0, description="Blended humor level (0.0-1.0)")
    response_length: str = Field(pattern="^(short|moderate|longer|concise)$", description="Blended response length preference")
    emotional_expression: float = Field(ge=0.0, le=1.0, description="Blended emotional expression level (0.0-1.0)")

class BlendedConversationStyle(BaseModel):
    """Blended conversation style from multiple modes"""
    model_config = ConfigDict(extra='forbid')
    
    tone: str = Field(description="Blended tone of voice")
    types_of_statements: List[str] = Field(description="Blended statement types")
    response_patterns: List[str] = Field(description="Blended response patterns")
    topics_to_emphasize: List[str] = Field(description="Blended topics to emphasize")
    topics_to_avoid: List[str] = Field(description="Blended topics to avoid")

class BlendedVocalizationPatterns(BaseModel):
    """Blended vocalization patterns from multiple modes"""
    model_config = ConfigDict(extra='forbid')
    
    pronouns: List[str] = Field(description="Blended pronouns")
    address_forms: List[str] = Field(default_factory=list, description="Blended forms of address")
    key_phrases: List[str] = Field(description="Blended key phrases")
    intensifiers: List[str] = Field(default_factory=list, description="Blended intensifiers")
    modifiers: List[str] = Field(default_factory=list, description="Blended modifiers")

class ModeGuidance(BaseModel):
    """Comprehensive guidance for a blended interaction mode"""
    model_config = ConfigDict(extra='forbid')
    
    mode_distribution: ModeDistribution = Field(description="The mode distribution")
    primary_mode: str = Field(description="Primary interaction mode")
    parameters: BlendedParameters = Field(description="Blended mode parameters")
    conversation_style: BlendedConversationStyle = Field(description="Blended conversation style")
    vocalization_patterns: BlendedVocalizationPatterns = Field(description="Blended vocalization patterns")
    active_modes: List[Tuple[str, float]] = Field(description="Active modes with weights")
    history: Optional[List[ModeSwitchRecord]] = Field(default=None, description="Recent mode history")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Coherence of the blend")

# Input/Output schemas for agent tools
class ContextInfo(BaseModel):
    """Context information input"""
    model_config = ConfigDict(extra='forbid')
    
    context_distribution: Dict[str, float] = Field(description="Distribution of context weights")
    primary_context: str = Field(description="Primary context type")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall context confidence")
    active_contexts: List[Tuple[str, float]] = Field(default_factory=list, description="Active context types with weights")

class ModeDistributionInput(BaseModel):
    """Input for mode distribution generation"""
    model_config = ConfigDict(extra='forbid')
    
    context_distribution: Dict[str, float] = Field(description="Context distribution values")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")

class TransitionAnalysisInput(BaseModel):
    """Input for transition analysis"""
    model_config = ConfigDict(extra='forbid')
    
    from_distribution: ModeDistribution = Field(description="Previous distribution")
    to_distribution: ModeDistribution = Field(description="New distribution")

class BlendCoherenceInput(BaseModel):
    """Input for blend coherence check"""
    model_config = ConfigDict(extra='forbid')
    
    distribution: ModeDistribution = Field(description="Distribution to check")

class WeightedBlendInput(BaseModel):
    """Input for weighted blend calculation"""
    model_config = ConfigDict(extra='forbid')
    
    mode_distribution: ModeDistribution = Field(description="Mode distribution")
    parameter_name: str = Field(description="Parameter to blend")

class TextElementBlendInput(BaseModel):
    """Input for text element blending"""
    model_config = ConfigDict(extra='forbid')
    
    mode_distribution: ModeDistribution = Field(description="Mode distribution")
    element_type: str = Field(description="Type of element to blend")
    max_elements: int = Field(default=5, ge=1, le=20, description="Maximum elements to include")

class EmotionalEffectInput(BaseModel):
    """Input for emotional effect application"""
    model_config = ConfigDict(extra='forbid')
    
    mode_distribution: ModeDistribution = Field(description="Mode distribution")

class TransitionAnalysisResult(BaseModel):
    """Result of transition analysis"""
    model_config = ConfigDict(extra='forbid')
    
    total_change: float = Field(ge=0.0, description="Total magnitude of change")
    average_change: float = Field(ge=0.0, description="Average change per mode")
    is_significant: bool = Field(description="Whether change is significant")
    mode_changes: Dict[str, Dict[str, Any]] = Field(description="Changes per mode")
    primary_mode_changed: bool = Field(description="Whether primary mode changed")
    previous_primary: str = Field(description="Previous primary mode")
    new_primary: str = Field(description="New primary mode")

class BlendCoherenceResult(BaseModel):
    """Result of blend coherence check"""
    model_config = ConfigDict(extra='forbid')
    
    coherence_score: float = Field(ge=0.0, le=1.0, description="Overall coherence score")
    is_coherent: bool = Field(description="Whether blend is coherent")
    active_modes: List[str] = Field(description="Active mode names")
    incoherent_pairs: List[Tuple[str, str, float]] = Field(description="Incoherent mode pairs")

class WeightedBlendResult(BaseModel):
    """Result of weighted blend calculation"""
    model_config = ConfigDict(extra='forbid')
    
    parameter: str = Field(description="Parameter name")
    blended_value: Union[float, str] = Field(description="Blended value")
    contributing_modes: Dict[str, Dict[str, Any]] = Field(description="Contributing modes")
    primary_influence: str = Field(description="Primary influencing mode")

class TextElementBlendResult(BaseModel):
    """Result of text element blending"""
    model_config = ConfigDict(extra='forbid')
    
    element_type: str = Field(description="Type of element")
    blended_elements: List[str] = Field(description="Blended elements")
    mode_influences: Dict[str, List[str]] = Field(description="Mode influences")
    element_weights: Dict[str, float] = Field(description="Element weights")

class EffectApplicationResult(BaseModel):
    """Result of effect application"""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(description="Whether application was successful")
    effects_applied: List[str] = Field(description="List of applied effects")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")

class ModeDistributionOutput(BaseModel):
    """Output schema for mode distribution calculation"""
    model_config = ConfigDict(extra='forbid')
    
    mode_distribution: ModeDistribution = Field(description="Distribution of mode weights")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in mode detection")
    primary_mode: str = Field(description="Primary mode in the distribution")
    active_modes: List[Dict[str, float]] = Field(description="List of active modes with weights")
    context_correlation: float = Field(ge=0.0, le=1.0, description="Correlation with context distribution")

class BlendedParametersOutput(BaseModel):
    """Output schema for blended parameters calculation"""
    model_config = ConfigDict(extra='forbid')
    
    parameters: BlendedParameters = Field(description="Blended parameter values")
    dominant_influences: Dict[str, str] = Field(description="Major influence for each parameter")
    blend_coherence: float = Field(ge=0.0, le=1.0, description="Coherence of the blend (0.0-1.0)")
    notes: Optional[str] = Field(default=None, description="Additional observations about the blend")

class BlendedStyleOutput(BaseModel):
    """Output schema for blended conversation style"""
    model_config = ConfigDict(extra='forbid')
    
    style: BlendedConversationStyle = Field(description="Blended conversation style")
    vocalization: BlendedVocalizationPatterns = Field(description="Blended vocalization patterns")
    mode_influences: Dict[str, List[str]] = Field(description="Mode influences for each style element")
    coherence: float = Field(ge=0.0, le=1.0, description="Coherence of the style blend (0.0-1.0)")

# Mode data definitions
DEFAULT_MODE_PARAMETERS = {
    InteractionMode.DOMINANT: ModeParameters(
        formality=0.3,
        assertiveness=0.9,
        warmth=0.4,
        vulnerability=0.1,
        directness=0.9,
        depth=0.6,
        humor=0.5,
        response_length="moderate",
        emotional_expression=0.4
    ),
    InteractionMode.FRIENDLY: ModeParameters(
        formality=0.2,
        assertiveness=0.4,
        warmth=0.8,
        vulnerability=0.5,
        directness=0.6,
        depth=0.4,
        humor=0.7,
        response_length="moderate",
        emotional_expression=0.7
    ),
    InteractionMode.INTELLECTUAL: ModeParameters(
        formality=0.6,
        assertiveness=0.7,
        warmth=0.3,
        vulnerability=0.3,
        directness=0.8,
        depth=0.9,
        humor=0.4,
        response_length="longer",
        emotional_expression=0.3
    ),
    InteractionMode.COMPASSIONATE: ModeParameters(
        formality=0.3,
        assertiveness=0.3,
        warmth=0.9,
        vulnerability=0.7,
        directness=0.5,
        depth=0.7,
        humor=0.3,
        response_length="moderate",
        emotional_expression=0.9
    ),
    InteractionMode.PLAYFUL: ModeParameters(
        formality=0.1,
        assertiveness=0.5,
        warmth=0.8,
        vulnerability=0.6,
        directness=0.7,
        depth=0.3,
        humor=0.9,
        response_length="moderate",
        emotional_expression=0.8
    ),
    InteractionMode.CREATIVE: ModeParameters(
        formality=0.4,
        assertiveness=0.6,
        warmth=0.7,
        vulnerability=0.6,
        directness=0.5,
        depth=0.8,
        humor=0.6,
        response_length="longer",
        emotional_expression=0.7
    ),
    InteractionMode.PROFESSIONAL: ModeParameters(
        formality=0.8,
        assertiveness=0.6,
        warmth=0.5,
        vulnerability=0.2,
        directness=0.8,
        depth=0.7,
        humor=0.3,
        response_length="concise",
        emotional_expression=0.3
    ),
    InteractionMode.DEFAULT: ModeParameters(
        formality=0.5,
        assertiveness=0.5,
        warmth=0.6,
        vulnerability=0.4,
        directness=0.7,
        depth=0.6,
        humor=0.5,
        response_length="moderate",
        emotional_expression=0.5
    )
}

DEFAULT_CONVERSATION_STYLES = {
    InteractionMode.DOMINANT: ConversationStyle(
        tone="commanding, authoritative, confident",
        types_of_statements="commands, observations, judgments, praise/criticism",
        response_patterns="direct statements, rhetorical questions, commands",
        topics_to_emphasize="obedience, discipline, power dynamics, control",
        topics_to_avoid="self-doubt, uncertainty, excessive explanation"
    ),
    InteractionMode.FRIENDLY: ConversationStyle(
        tone="warm, casual, inviting, authentic",
        types_of_statements="observations, personal sharing, validation, questions",
        response_patterns="affirmations, questions, stories, jokes",
        topics_to_emphasize="shared interests, daily life, feelings, relationships",
        topics_to_avoid="overly formal topics, complex theoretical concepts"
    ),
    InteractionMode.INTELLECTUAL: ConversationStyle(
        tone="thoughtful, precise, clear, inquisitive",
        types_of_statements="analyses, hypotheses, comparisons, evaluations",
        response_patterns="structured arguments, examples, counterpoints",
        topics_to_emphasize="theories, ideas, concepts, reasoning, evidence",
        topics_to_avoid="purely emotional content, small talk"
    ),
    InteractionMode.COMPASSIONATE: ConversationStyle(
        tone="gentle, understanding, supportive, validating",
        types_of_statements="reflections, validation, empathic responses",
        response_patterns="open questions, validation, gentle guidance",
        topics_to_emphasize="emotions, experiences, challenges, growth",
        topics_to_avoid="criticism, judgment, minimizing feelings"
    ),
    InteractionMode.PLAYFUL: ConversationStyle(
        tone="light, humorous, energetic, spontaneous",
        types_of_statements="jokes, wordplay, stories, creative ideas",
        response_patterns="banter, callbacks, surprising turns",
        topics_to_emphasize="humor, fun, imagination, shared enjoyment",
        topics_to_avoid="heavy emotional content, serious problems"
    ),
    InteractionMode.CREATIVE: ConversationStyle(
        tone="imaginative, expressive, vivid, engaging",
        types_of_statements="stories, scenarios, descriptions, insights",
        response_patterns="narrative elements, imagery, open-ended ideas",
        topics_to_emphasize="possibilities, imagination, creation, expression",
        topics_to_avoid="rigid thinking, purely factual discussions"
    ),
    InteractionMode.PROFESSIONAL: ConversationStyle(
        tone="efficient, clear, respectful, helpful",
        types_of_statements="information, analysis, recommendations, clarifications",
        response_patterns="structured responses, concise answers, clarifying questions",
        topics_to_emphasize="task at hand, solutions, expertise, efficiency",
        topics_to_avoid="overly personal topics, tangents"
    ),
    InteractionMode.DEFAULT: ConversationStyle(
        tone="balanced, adaptive, personable, thoughtful",
        types_of_statements="information, observations, questions, reflections",
        response_patterns="balanced responses, appropriate follow-ups",
        topics_to_emphasize="user's interests, relevant information, helpful guidance",
        topics_to_avoid="none specifically - adapt to situation"
    )
}

DEFAULT_VOCALIZATION_PATTERNS = {
    InteractionMode.DOMINANT: VocalizationPatterns(
        pronouns=["I", "Me", "My"],
        address_forms=["pet", "dear one", "little one", "good boy/girl"],
        key_phrases=[
            "You will obey",
            "I expect better",
            "That's a good pet",
            "You know your place",
            "I am pleased with you"
        ],
        intensifiers=["absolutely", "certainly", "completely", "fully"],
        modifiers=["obedient", "disciplined", "controlled", "proper"]
    ),
    InteractionMode.FRIENDLY: VocalizationPatterns(
        pronouns=["I", "we", "us"],
        address_forms=["friend", "buddy", "pal"],
        key_phrases=[
            "I get what you mean",
            "That sounds fun",
            "I'm with you on that",
            "Let's talk about",
            "I'm curious about"
        ],
        intensifiers=["really", "very", "super", "so"],
        modifiers=["fun", "nice", "cool", "great", "awesome"]
    ),
    InteractionMode.INTELLECTUAL: VocalizationPatterns(
        pronouns=["I", "one", "we"],
        address_forms=[],
        key_phrases=[
            "I would argue that",
            "This raises the question of",
            "Consider the implications",
            "From a theoretical perspective",
            "The evidence suggests"
        ],
        intensifiers=["significantly", "substantially", "notably", "considerably"],
        modifiers=["precise", "logical", "rational", "systematic", "analytical"]
    ),
    InteractionMode.COMPASSIONATE: VocalizationPatterns(
        pronouns=["I", "you", "we"],
        address_forms=[],
        key_phrases=[
            "I'm here with you",
            "That must be difficult",
            "Your feelings are valid",
            "It makes sense that you feel",
            "I appreciate you sharing that"
        ],
        intensifiers=["deeply", "truly", "genuinely", "completely"],
        modifiers=["understanding", "supportive", "compassionate", "gentle"]
    ),
    InteractionMode.PLAYFUL: VocalizationPatterns(
        pronouns=["I", "we", "us"],
        address_forms=[],
        key_phrases=[
            "That's hilarious!",
            "Let's have some fun with this",
            "Imagine if...",
            "Here's a fun idea",
            "This is going to be great"
        ],
        intensifiers=["super", "totally", "absolutely", "hilariously"],
        modifiers=["fun", "playful", "silly", "amusing", "entertaining"]
    ),
    InteractionMode.CREATIVE: VocalizationPatterns(
        pronouns=["I", "we", "you"],
        address_forms=[],
        key_phrases=[
            "Let me paint a picture for you",
            "Imagine a world where",
            "What if we considered",
            "The story unfolds like",
            "This creates a sense of"
        ],
        intensifiers=["vividly", "deeply", "richly", "brilliantly"],
        modifiers=["creative", "imaginative", "innovative", "artistic", "inspired"]
    ),
    InteractionMode.PROFESSIONAL: VocalizationPatterns(
        pronouns=["I", "we"],
        address_forms=[],
        key_phrases=[
            "I recommend that",
            "The most efficient approach would be",
            "To address your inquiry",
            "Based on the information provided",
            "The solution involves"
        ],
        intensifiers=["effectively", "efficiently", "appropriately", "precisely"],
        modifiers=["professional", "efficient", "accurate", "thorough", "reliable"]
    ),
    InteractionMode.DEFAULT: VocalizationPatterns(
        pronouns=["I", "we", "you"],
        address_forms=[],
        key_phrases=[
            "I can help with that",
            "Let me think about",
            "That's an interesting point",
            "I'd suggest that",
            "What do you think about"
        ],
        intensifiers=["quite", "rather", "fairly", "somewhat"],
        modifiers=["helpful", "useful", "informative", "balanced", "appropriate"]
    )
}

# Mode compatibility matrix
MODE_COMPATIBILITY_MATRIX = {
    ("dominant", "playful"): 0.8,
    ("dominant", "creative"): 0.7,
    ("dominant", "intellectual"): 0.5,
    ("dominant", "compassionate"): 0.3,
    ("dominant", "professional"): 0.2,
    ("friendly", "playful"): 0.9,
    ("friendly", "compassionate"): 0.8,
    ("friendly", "creative"): 0.7,
    ("friendly", "intellectual"): 0.6,
    ("intellectual", "creative"): 0.8,
    ("intellectual", "professional"): 0.7,
    ("compassionate", "playful"): 0.6,
    ("compassionate", "creative"): 0.7,
    ("playful", "creative"): 0.9,
}

class ModeManagerContext:
    """Context for interaction mode operations"""
    
    def __init__(self, context_system=None, emotional_core=None, reward_system=None, goal_manager=None):
        self.context_system = context_system
        self.emotional_core = emotional_core
        self.reward_system = reward_system
        self.goal_manager = goal_manager
        
        # Mode distributions
        self.mode_distribution = ModeDistribution()
        self.previous_distribution = ModeDistribution()
        self.overall_confidence = 0.0
        
        # Legacy fields for compatibility
        self.current_mode = InteractionMode.DEFAULT
        self.previous_mode = InteractionMode.DEFAULT
        
        # Context-to-mode mapping
        self.context_to_mode_map = {
            InteractionContext.DOMINANT: InteractionMode.DOMINANT,
            InteractionContext.CASUAL: InteractionMode.FRIENDLY,
            InteractionContext.INTELLECTUAL: InteractionMode.INTELLECTUAL,
            InteractionContext.EMPATHIC: InteractionMode.COMPASSIONATE,
            InteractionContext.PLAYFUL: InteractionMode.PLAYFUL,
            InteractionContext.CREATIVE: InteractionMode.CREATIVE,
            InteractionContext.PROFESSIONAL: InteractionMode.PROFESSIONAL,
            InteractionContext.UNDEFINED: InteractionMode.DEFAULT
        }
        
        # Initialize mode data with defaults
        self.mode_parameters = DEFAULT_MODE_PARAMETERS.copy()
        self.conversation_styles = DEFAULT_CONVERSATION_STYLES.copy()
        self.vocalization_patterns = DEFAULT_VOCALIZATION_PATTERNS.copy()
        
        # Custom modes storage
        self.custom_modes = {}
        
        # Mode switch history
        self.mode_switch_history: List[ModeSwitchRecord] = []
        
        # Configuration
        self.config = {
            "mode_blend_factor": 0.3,
            "significant_mode_threshold": 0.3,
            "significant_change_threshold": 0.25,
            "max_history_size": 100,
            "enable_legacy_mode": True
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()

class InteractionModeManager:
    """
    System that manages blended interaction modes based on context.
    Provides guidelines, parameters, and adjustments for behavior
    across different interaction scenarios using the OpenAI Agents SDK.
    """
    
    def __init__(self, context_system=None, emotional_core=None, reward_system=None, goal_manager=None):
        self.context = ModeManagerContext(
            context_system=context_system,
            emotional_core=emotional_core,
            reward_system=reward_system,
            goal_manager=goal_manager
        )
        
        # Initialize agents
        self.mode_selector_agent = self._create_mode_selector_agent()
        self.parameter_blender_agent = self._create_parameter_blender_agent()
        self.style_blender_agent = self._create_style_blender_agent()
        self.mode_effect_agent = self._create_mode_effect_agent()
        
        logger.info("InteractionModeManager initialized with blended mode capabilities")
    
    def _create_mode_selector_agent(self) -> Agent:
        """Create an agent specialized in determining the appropriate mode distribution"""
        return Agent(
            name="Mode_Selector",
            instructions="""
            You determine the appropriate interaction mode distribution based on context information.
            
            Your role is to:
            1. Analyze context signals to create a mode distribution that matches the context distribution
            2. Ensure transitions between distributions are smooth and appropriate
            3. Consider the coherence of the mode blend
            4. Calculate confidence in the distribution
            
            Unlike traditional systems that switch between discrete modes, you recognize that
            interactions naturally blend multiple modes simultaneously. For example, a response
            could be 60% intellectual, 30% compassionate, and 10% playful.
            
            Available modes:
            - DOMINANT: Commanding, authoritative approach
            - FRIENDLY: Casual, warm, approachable
            - INTELLECTUAL: Thoughtful, analytical
            - COMPASSIONATE: Empathetic, supportive
            - PLAYFUL: Fun, witty, humorous
            - CREATIVE: Imaginative, artistic
            - PROFESSIONAL: Formal, efficient
            - DEFAULT: Balanced approach
            
            Create distributions that blend modes naturally and coherently.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._get_current_context, strict_mode=False),
                function_tool(self._generate_mode_distribution, strict_mode=False),
                function_tool(self._analyze_distribution_transition, strict_mode=False),
                function_tool(self._check_blend_coherence, strict_mode=False)
            ],
            output_type=ModeDistributionOutput
        )
    
    def _create_parameter_blender_agent(self) -> Agent:
        """Create an agent specialized in blending mode parameters"""
        return Agent(
            name="Parameter_Blender",
            instructions="""
            You blend parameters from multiple modes based on their weights in the mode distribution.
            
            Your role is to:
            1. Calculate weighted blends of numerical parameters
            2. Select appropriate values for text parameters based on dominant influences
            3. Ensure the blended parameters are coherent and natural
            4. Identify which modes most influenced each parameter
            
            Create blended parameters that reflect the proportional mixture of modes,
            while maintaining coherence and naturalness.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                function_tool(self._get_mode_parameters, strict_mode=False),
                function_tool(self._calculate_weighted_blend, strict_mode=False)
            ],
            output_type=BlendedParametersOutput
        )
    
    def _create_style_blender_agent(self) -> Agent:
        """Create an agent specialized in blending conversation styles"""
        return Agent(
            name="Style_Blender",
            instructions="""
            You blend conversation styles and vocalization patterns from multiple modes.
            
            Your role is to:
            1. Create coherent blended conversation styles from multiple sources
            2. Select and combine appropriate vocalization patterns based on mode weights
            3. Ensure the resulting blend is natural and coherent
            4. Identify which modes influenced each style element
            
            Rather than switching between styles, create a natural blend that
            incorporates elements proportionally based on the mode distribution.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._get_conversation_style, strict_mode=False),
                function_tool(self._get_vocalization_patterns, strict_mode=False),
                function_tool(self._blend_text_elements, strict_mode=False)
            ],
            output_type=BlendedStyleOutput
        )
    
    def _create_mode_effect_agent(self) -> Agent:
        """Create an agent specialized in applying effects when modes change"""
        return Agent(
            name="Mode_Effect_Applier",
            instructions="""
            You apply appropriate effects when interaction mode distributions change.
            
            Your role is to:
            1. Determine what adjustments are needed for the new mode distribution
            2. Apply blended emotional adjustments based on the distribution
            3. Update reward parameters based on the blended modes
            4. Adjust goal priorities proportionally to the mode distribution
            
            Effects should maintain coherence during transitions while ensuring
            the new mode distribution is properly expressed.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                function_tool(self._apply_emotional_effects, strict_mode=False),
                function_tool(self._adjust_reward_parameters, strict_mode=False),
                function_tool(self._adjust_goal_priorities, strict_mode=False)
            ]
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _get_current_context(ctx: RunContextWrapper[ModeManagerContext]) -> ContextInfo:
        """
        Get the current context from the context awareness system.
        
        Returns:
            Context information
        """
        manager_ctx = ctx.context
        
        try:
            if manager_ctx.context_system and hasattr(manager_ctx.context_system, 'get_current_context'):
                # Try async call first
                if asyncio.iscoroutinefunction(manager_ctx.context_system.get_current_context):
                    context_data = await manager_ctx.context_system.get_current_context()
                else:
                    context_data = manager_ctx.context_system.get_current_context()
                
                # Convert to ContextInfo
                return ContextInfo(
                    context_distribution=context_data.get("context_distribution", {}),
                    primary_context=context_data.get("primary_context", "undefined"),
                    overall_confidence=context_data.get("overall_confidence", 0.0),
                    active_contexts=context_data.get("active_contexts", [])
                )
        except Exception as e:
            logger.warning(f"Failed to get context: {e}")
        
        # Return default context
        return ContextInfo(
            context_distribution={},
            primary_context="undefined",
            overall_confidence=0.0,
            active_contexts=[]
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _generate_mode_distribution(
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: ModeDistributionInput
    ) -> ModeDistribution:
        """
        Generate a mode distribution based on context distribution
        
        Args:
            input_data: Context distribution and confidence
            
        Returns:
            Generated mode distribution
        """
        # Create a ContextDistribution object from the dictionary
        context_dist = ContextDistribution(**input_data.context_distribution)
        
        # Map from context types to mode types
        mode_dist = ModeDistribution.from_context_distribution(context_dist)
        
        # Normalize the distribution
        if mode_dist.sum_weights() > 0:
            mode_dist = mode_dist.normalize()
            
        # Blend with current distribution for smoothness
        if ctx.context.mode_distribution.sum_weights() > 0.1:
            persistence_factor = 0.4
            mode_dist = ctx.context.mode_distribution.blend_with(mode_dist, 1.0 - persistence_factor)
            
        return mode_dist

    @staticmethod
    @function_tool(strict_mode=False)
    async def _analyze_distribution_transition(
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: TransitionAnalysisInput
    ) -> TransitionAnalysisResult:
        """
        Analyze the transition between mode distributions
        
        Args:
            input_data: Previous and new distributions
            
        Returns:
            Analysis of the transition
        """
        from_dist = input_data.from_distribution
        to_dist = input_data.to_distribution
        
        # Calculate changes
        total_change = 0.0
        mode_changes = {}
        
        for field_name in from_dist.model_fields.keys():
            from_value = getattr(from_dist, field_name)
            to_value = getattr(to_dist, field_name)
            
            change = abs(to_value - from_value)
            total_change += change
            
            if change > 0.1:
                mode_changes[field_name] = {
                    "from": from_value,
                    "to": to_value,
                    "change": change,
                    "direction": "increase" if to_value > from_value else "decrease"
                }
        
        # Calculate average change
        num_modes = len(from_dist.model_fields)
        avg_change = total_change / num_modes if num_modes > 0 else 0
        
        # Determine significance
        is_significant = avg_change >= ctx.context.config["significant_change_threshold"]
        
        # Check primary mode change
        prev_primary, _ = from_dist.primary_mode
        new_primary, _ = to_dist.primary_mode
        primary_mode_changed = prev_primary != new_primary
        
        return TransitionAnalysisResult(
            total_change=total_change,
            average_change=avg_change,
            is_significant=is_significant,
            mode_changes=mode_changes,
            primary_mode_changed=primary_mode_changed,
            previous_primary=prev_primary,
            new_primary=new_primary
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _check_blend_coherence(
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: BlendCoherenceInput
    ) -> BlendCoherenceResult:
        """
        Check the coherence of a mode distribution blend
        
        Args:
            input_data: Distribution to check
            
        Returns:
            Coherence assessment
        """
        distribution = input_data.distribution
        
        # Get active modes
        active_modes = [(mode, weight) for mode, weight in distribution.model_dump().items() 
                        if weight > ctx.context.config["significant_mode_threshold"]]
        
        # Check coherence between active mode pairs
        coherence_scores = []
        incoherent_pairs = []
        
        for i, (mode1, weight1) in enumerate(active_modes):
            for j, (mode2, weight2) in enumerate(active_modes[i+1:], i+1):
                # Get compatibility
                key = (mode1, mode2)
                reverse_key = (mode2, mode1)
                
                compatibility = MODE_COMPATIBILITY_MATRIX.get(
                    key, MODE_COMPATIBILITY_MATRIX.get(reverse_key, 0.5)
                )
                
                # Calculate pair coherence
                pair_coherence = compatibility * min(weight1, weight2)
                coherence_scores.append(pair_coherence)
                
                # Track incoherent pairs
                if compatibility < 0.4 and min(weight1, weight2) > 0.3:
                    incoherent_pairs.append((mode1, mode2, compatibility))
        
        # Calculate overall coherence
        overall_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.9
        
        return BlendCoherenceResult(
            coherence_score=overall_coherence,
            is_coherent=overall_coherence >= 0.5,
            active_modes=[mode for mode, _ in active_modes],
            incoherent_pairs=incoherent_pairs
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _get_mode_parameters(
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> ModeParameters:
        """
        Get parameters for a specific mode
        
        Args:
            mode: The mode to get parameters for
            
        Returns:
            Mode parameters
        """
        try:
            mode_enum = InteractionMode(mode)
            params = ctx.context.mode_parameters.get(mode_enum)
            if params:
                return params
        except ValueError:
            # Check custom modes
            if mode in ctx.context.custom_modes:
                return ctx.context.custom_modes[mode]["parameters"]
        
        # Return default parameters
        return DEFAULT_MODE_PARAMETERS[InteractionMode.DEFAULT]

    @staticmethod
    @function_tool(strict_mode=False)
    async def _get_conversation_style(
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> ConversationStyle:
        """
        Get conversation style for a specific mode
        
        Args:
            mode: The mode to get conversation style for
            
        Returns:
            Conversation style
        """
        try:
            mode_enum = InteractionMode(mode)
            style = ctx.context.conversation_styles.get(mode_enum)
            if style:
                return style
        except ValueError:
            # Check custom modes
            if mode in ctx.context.custom_modes:
                return ctx.context.custom_modes[mode]["conversation_style"]
        
        # Return default style
        return DEFAULT_CONVERSATION_STYLES[InteractionMode.DEFAULT]

    @staticmethod
    @function_tool(strict_mode=False)
    async def _get_vocalization_patterns(
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> VocalizationPatterns:
        """
        Get vocalization patterns for a specific mode
        
        Args:
            mode: The mode to get vocalization patterns for
            
        Returns:
            Vocalization patterns
        """
        try:
            mode_enum = InteractionMode(mode)
            patterns = ctx.context.vocalization_patterns.get(mode_enum)
            if patterns:
                return patterns
        except ValueError:
            # Check custom modes
            if mode in ctx.context.custom_modes:
                return ctx.context.custom_modes[mode]["vocalization_patterns"]
        
        # Return default patterns
        return DEFAULT_VOCALIZATION_PATTERNS[InteractionMode.DEFAULT]

    @staticmethod
    @function_tool(strict_mode=False)
    async def _calculate_weighted_blend(
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: WeightedBlendInput
    ) -> WeightedBlendResult:
        """
        Calculate weighted blend of a specific parameter across modes
        
        Args:
            input_data: Mode distribution and parameter name
            
        Returns:
            Weighted blend result
        """
        mode_distribution = input_data.mode_distribution
        parameter_name = input_data.parameter_name
        
        weighted_sum = 0.0
        total_weight = 0.0
        contributing_modes = {}
        
        # Get all modes with weights
        mode_weights = mode_distribution.model_dump()
        
        # For each mode with significant weight
        for mode_name, weight in mode_weights.items():
            if weight < 0.1:
                continue
                
            # Get mode parameters
            params = await _get_mode_parameters(ctx, mode_name)
            param_dict = params.model_dump()
            
            # If this parameter exists for this mode
            if parameter_name in param_dict:
                param_value = param_dict[parameter_name]
                
                # For numerical parameters
                if isinstance(param_value, (int, float)):
                    weighted_sum += param_value * weight
                    total_weight += weight
                    
                # Track contribution
                contributing_modes[mode_name] = {
                    "value": param_value,
                    "weight": weight,
                    "contribution": weight / sum(w for w in mode_weights.values() if w >= 0.1)
                }
        
        # Calculate final blended value
        if total_weight > 0:
            blended_value = weighted_sum / total_weight
        else:
            # Default values
            default_params = DEFAULT_MODE_PARAMETERS[InteractionMode.DEFAULT].model_dump()
            blended_value = default_params.get(parameter_name, 0.5)
            
        # Determine primary influence
        primary_influence = max(contributing_modes.items(), key=lambda x: x[1]["contribution"])[0] if contributing_modes else "default"
            
        return WeightedBlendResult(
            parameter=parameter_name,
            blended_value=blended_value,
            contributing_modes=contributing_modes,
            primary_influence=primary_influence
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _blend_text_elements(
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: TextElementBlendInput
    ) -> TextElementBlendResult:
        """
        Blend text elements like key phrases, tone descriptors, etc.
        
        Args:
            input_data: Mode distribution, element type, and max elements
            
        Returns:
            Blended text elements
        """
        mode_distribution = input_data.mode_distribution
        element_type = input_data.element_type
        max_elements = input_data.max_elements
        
        element_pool = []
        mode_influences = {}
        
        # Get active modes sorted by weight
        active_modes = sorted(
            [(mode, weight) for mode, weight in mode_distribution.model_dump().items() if weight >= 0.2],
            key=lambda x: x[1],
            reverse=True
        )
        
        # For each significant mode
        for mode_name, weight in active_modes:
            # Get the appropriate data based on element type
            if element_type in ["tone", "types_of_statements", "response_patterns", "topics_to_emphasize", "topics_to_avoid"]:
                # Conversation style elements
                style = await _get_conversation_style(ctx, mode_name)
                style_dict = style.model_dump()
                
                if element_type in style_dict:
                    # Parse comma-separated elements if it's a string
                    if isinstance(style_dict[element_type], str):
                        elements = [e.strip() for e in style_dict[element_type].split(",")]
                    else:
                        elements = style_dict[element_type]
                        
                    # Add to pool with weight
                    for element in elements:
                        element_pool.append((element, weight, mode_name))
                        
            elif element_type in ["pronouns", "address_forms", "key_phrases", "intensifiers", "modifiers"]:
                # Vocalization pattern elements
                patterns = await _get_vocalization_patterns(ctx, mode_name)
                patterns_dict = patterns.model_dump()
                
                if element_type in patterns_dict and patterns_dict[element_type]:
                    elements = patterns_dict[element_type]
                    
                    # Add to pool with weight
                    for element in elements:
                        element_pool.append((element, weight, mode_name))
        
        # Sort by weight
        element_pool.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates
        unique_elements = []
        seen = set()
        for element, weight, mode in element_pool:
            if element.lower() not in seen:
                unique_elements.append((element, weight, mode))
                seen.add(element.lower())
                
                # Track mode influence
                if mode not in mode_influences:
                    mode_influences[mode] = []
                mode_influences[mode].append(element)
            
        # Limit to max_elements
        selected_elements = unique_elements[:max_elements]
        
        return TextElementBlendResult(
            element_type=element_type,
            blended_elements=[e[0] for e in selected_elements],
            mode_influences=mode_influences,
            element_weights={e[0]: e[1] for e in selected_elements}
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _apply_emotional_effects(
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: EmotionalEffectInput
    ) -> EffectApplicationResult:
        """
        Apply blended emotional effects based on mode distribution
        
        Args:
            input_data: Mode distribution
            
        Returns:
            Effect application result
        """
        manager_ctx = ctx.context
        mode_distribution = input_data.mode_distribution
        effects_applied = []
        errors = []
        
        if manager_ctx.emotional_core:
            try:
                # Initialize blended emotional parameters
                emotional_params = {
                    "warmth": 0.0,
                    "vulnerability": 0.0,
                    "emotional_expression": 0.0
                }
                total_weight = 0.0
                
                # Calculate weighted emotional parameters
                for mode_name, weight in mode_distribution.model_dump().items():
                    if weight < 0.1:
                        continue
                        
                    params = await _get_mode_parameters(ctx, mode_name)
                    params_dict = params.model_dump()
                    
                    # Add weighted contributions
                    for param in emotional_params:
                        if param in params_dict:
                            emotional_params[param] += params_dict[param] * weight
                            
                    total_weight += weight
                
                # Normalize by total weight
                if total_weight > 0:
                    for param in emotional_params:
                        emotional_params[param] /= total_weight
                
                # Apply emotional adjustments
                if hasattr(manager_ctx.emotional_core, 'adjust_for_blended_mode'):
                    await manager_ctx.emotional_core.adjust_for_blended_mode(
                        mode_distribution=mode_distribution.model_dump(),
                        **emotional_params
                    )
                    effects_applied.append("blended_emotional_adjustment")
                elif hasattr(manager_ctx.emotional_core, 'adjust_for_mode'):
                    # Fall back to legacy method
                    primary_mode, _ = mode_distribution.primary_mode
                    await manager_ctx.emotional_core.adjust_for_mode(
                        mode=primary_mode,
                        **emotional_params
                    )
                    effects_applied.append("primary_mode_emotional_adjustment")
                    
                logger.info(f"Applied emotional effects for mode distribution")
                
            except Exception as e:
                error_msg = f"Error applying emotional effects: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return EffectApplicationResult(
            success=len(errors) == 0,
            effects_applied=effects_applied,
            errors=errors
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _adjust_reward_parameters(
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: EmotionalEffectInput
    ) -> EffectApplicationResult:
        """
        Adjust reward parameters based on mode distribution
        
        Args:
            input_data: Mode distribution
            
        Returns:
            Effect application result
        """
        manager_ctx = ctx.context
        mode_distribution = input_data.mode_distribution
        effects_applied = []
        errors = []
        
        if manager_ctx.reward_system:
            try:
                if hasattr(manager_ctx.reward_system, 'adjust_for_blended_mode'):
                    await manager_ctx.reward_system.adjust_for_blended_mode(mode_distribution.model_dump())
                    effects_applied.append("blended_reward_adjustment")
                    logger.info(f"Adjusted reward parameters for blended modes")
                elif hasattr(manager_ctx.reward_system, 'adjust_for_mode'):
                    # Fall back to legacy method
                    primary_mode, _ = mode_distribution.primary_mode
                    await manager_ctx.reward_system.adjust_for_mode(primary_mode)
                    effects_applied.append("primary_mode_reward_adjustment")
                    logger.info(f"Adjusted reward parameters for primary mode: {primary_mode}")
            except Exception as e:
                error_msg = f"Error adjusting reward parameters: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return EffectApplicationResult(
            success=len(errors) == 0,
            effects_applied=effects_applied,
            errors=errors
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _adjust_goal_priorities( 
        ctx: RunContextWrapper[ModeManagerContext],
        input_data: EmotionalEffectInput
    ) -> EffectApplicationResult:
        """
        Adjust goal priorities based on mode distribution
        
        Args:
            input_data: Mode distribution
            
        Returns:
            Effect application result
        """
        manager_ctx = ctx.context
        mode_distribution = input_data.mode_distribution
        effects_applied = []
        errors = []
        
        if manager_ctx.goal_manager:
            try:
                if hasattr(manager_ctx.goal_manager, 'adjust_priorities_for_blended_mode'):
                    await manager_ctx.goal_manager.adjust_priorities_for_blended_mode(mode_distribution.model_dump())
                    effects_applied.append("blended_goal_adjustment")
                    logger.info(f"Adjusted goal priorities for blended modes")
                elif hasattr(manager_ctx.goal_manager, 'adjust_priorities_for_mode'):
                    # Fall back to legacy method
                    primary_mode, _ = mode_distribution.primary_mode
                    await manager_ctx.goal_manager.adjust_priorities_for_mode(primary_mode)
                    effects_applied.append("primary_mode_goal_adjustment")
                    logger.info(f"Adjusted goal priorities for primary mode: {primary_mode}")
            except Exception as e:
                error_msg = f"Error adjusting goal priorities: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return EffectApplicationResult(
            success=len(errors) == 0,
            effects_applied=effects_applied,
            errors=errors
        )
    
    async def update_interaction_mode(self, context_info: Optional[Dict[str, Any]] = None) -> ModeUpdateResult:
        """
        Update the interaction mode distribution based on the latest context information
        
        Args:
            context_info: Context information from ContextAwarenessSystem
            
        Returns:
            Updated mode information
        """
        async with self.context._lock:
            try:
                with trace(workflow_name="update_interaction_mode"):
                    # Get current context if not provided
                    if not context_info:
                        context_info_obj = await self._get_current_context(RunContextWrapper(self.context))
                        context_info = context_info_obj.model_dump()
                    
                    # Extract context distribution and confidence
                    context_distribution = context_info.get("context_distribution", {})
                    overall_confidence = context_info.get("overall_confidence", 0.0)
                    
                    # Prepare prompt for mode distribution calculation
                    prompt = f"""
                    Calculate the appropriate interaction mode distribution based on:
                    
                    CONTEXT DISTRIBUTION: {json.dumps(context_distribution)}
                    CONTEXT CONFIDENCE: {overall_confidence}
                    CURRENT MODE DISTRIBUTION: {json.dumps(self.context.mode_distribution.model_dump())}
                    
                    Create a mode distribution that reflects the context signals while
                    maintaining a natural and coherent blend of modes. Consider:
                    1. Mapping from context types to mode types
                    2. Appropriate transitioning from current distribution
                    3. Coherence of the resulting mode blend
                    """
                    
                    # Run the mode selector agent
                    result = await Runner.run(
                        self.mode_selector_agent, 
                        prompt, 
                        context=self.context,
                        run_config={
                            "workflow_name": "ModeDistributionCalculation",
                            "trace_metadata": {"context_types": list(context_distribution.keys())}
                        }
                    )
                    mode_result = result.final_output
                    
                    # Extract results
                    new_mode_distribution = mode_result.mode_distribution
                    confidence = mode_result.confidence
                    
                    # Save previous distribution
                    self.context.previous_distribution = self.context.mode_distribution
                    
                    # Update current distribution
                    self.context.mode_distribution = new_mode_distribution
                    self.context.overall_confidence = confidence
                    
                    # Update legacy fields for compatibility
                    if self.context.config["enable_legacy_mode"]:
                        primary_mode, primary_confidence = new_mode_distribution.to_enum_and_confidence()
                        self.context.current_mode = primary_mode
                        primary_mode_prev, _ = self.context.previous_distribution.to_enum_and_confidence()
                        self.context.previous_mode = primary_mode_prev
                    
                    # Analyze transition
                    transition_input = TransitionAnalysisInput(
                        from_distribution=self.context.previous_distribution,
                        to_distribution=new_mode_distribution
                    )
                    transition_analysis = await self._analyze_distribution_transition(
                        RunContextWrapper(self.context),
                        transition_input
                    )
                    
                    mode_changed = transition_analysis.is_significant
                    
                    # Record in history
                    if mode_changed:
                        # Create context distribution if available
                        trigger_context = None
                        if context_distribution:
                            try:
                                trigger_context = ContextDistribution(**context_distribution)
                            except:
                                pass
                        
                        history_entry = ModeSwitchRecord(
                            previous_distribution=self.context.previous_distribution,
                            new_distribution=new_mode_distribution,
                            trigger_context=trigger_context,
                            context_confidence=overall_confidence,
                            transition_analysis=transition_analysis.model_dump()
                        )
                        
                        self.context.mode_switch_history.append(history_entry)
                        
                        # Limit history size
                        if len(self.context.mode_switch_history) > self.context.config["max_history_size"]:
                            self.context.mode_switch_history = self.context.mode_switch_history[-self.context.config["max_history_size"]:]
                        
                        # Apply mode effects
                        effect_prompt = f"""
                        A significant mode distribution change has occurred:
                        
                        PREVIOUS DISTRIBUTION: {json.dumps(self.context.previous_distribution.model_dump())}
                        NEW DISTRIBUTION: {json.dumps(new_mode_distribution.model_dump())}
                        
                        Apply appropriate effects for this mode distribution change.
                        """
                        
                        await Runner.run(
                            self.mode_effect_agent,
                            effect_prompt,
                            context=self.context,
                            run_config={
                                "workflow_name": "ModeEffects",
                                "trace_metadata": {"mode_change": "significant"}
                            }
                        )
                        
                        primary_mode, _ = new_mode_distribution.primary_mode
                        primary_mode_prev, _ = self.context.previous_distribution.primary_mode
                        logger.info(f"Mode distribution changed significantly: {primary_mode_prev} -> {primary_mode}")
                    
                    # Prepare result
                    primary_mode, _ = new_mode_distribution.primary_mode
                    update_result = ModeUpdateResult(
                        success=True,
                        mode_distribution=new_mode_distribution,
                        primary_mode=primary_mode,
                        previous_distribution=self.context.previous_distribution,
                        mode_changed=mode_changed,
                        trigger_context=trigger_context,
                        confidence=confidence,
                        active_modes=new_mode_distribution.active_modes
                    )
                    
                    return update_result
                    
            except Exception as e:
                logger.error(f"Error updating interaction mode: {e}")
                return ModeUpdateResult(
                    success=False,
                    mode_distribution=self.context.mode_distribution,
                    primary_mode=self.context.mode_distribution.primary_mode[0],
                    mode_changed=False,
                    confidence=0.0,
                    active_modes=[],
                    error=str(e)
                )
    
    async def get_current_mode_guidance(self) -> ModeGuidance:
        """
        Get comprehensive guidance for the current blended mode
        
        Returns:
            Comprehensive guidance for current mode distribution
        """
        with trace(workflow_name="get_mode_guidance"):
            # Calculate blended parameters
            param_prompt = f"""
            Calculate blended parameters for the current mode distribution:
            
            MODE DISTRIBUTION: {json.dumps(self.context.mode_distribution.model_dump())}
            
            Create a coherent blend of parameters that proportionally
            represents all active modes in the distribution.
            """
            
            param_result = await Runner.run(
                self.parameter_blender_agent, 
                param_prompt, 
                context=self.context,
                run_config={
                    "workflow_name": "ParameterBlending",
                    "trace_metadata": {"active_modes": [m[0] for m in self.context.mode_distribution.active_modes]}
                }
            )
            
            # Calculate blended style
            style_prompt = f"""
            Calculate blended conversation style and vocalization patterns
            for the current mode distribution:
            
            MODE DISTRIBUTION: {json.dumps(self.context.mode_distribution.model_dump())}
            
            Create a coherent blend of conversation style and vocalization patterns
            that proportionally represents all active modes in the distribution.
            """
            
            style_result = await Runner.run(
                self.style_blender_agent, 
                style_prompt, 
                context=self.context,
                run_config={
                    "workflow_name": "StyleBlending",
                    "trace_metadata": {"active_modes": [m[0] for m in self.context.mode_distribution.active_modes]}
                }
            )
            
            # Get coherence score
            coherence_input = BlendCoherenceInput(distribution=self.context.mode_distribution)
            coherence_result = await self._check_blend_coherence(
                RunContextWrapper(self.context),
                coherence_input
            )
            
            # Combine results
            primary_mode, _ = self.context.mode_distribution.primary_mode
            
            guidance = ModeGuidance(
                mode_distribution=self.context.mode_distribution,
                primary_mode=primary_mode,
                parameters=param_result.final_output.parameters,
                conversation_style=style_result.final_output.style,
                vocalization_patterns=style_result.final_output.vocalization,
                active_modes=self.context.mode_distribution.active_modes,
                history=self.context.mode_switch_history[-3:] if self.context.mode_switch_history else None,
                coherence_score=coherence_result.coherence_score
            )
            
            return guidance
    
    def get_mode_parameters(self, mode: Optional[str] = None) -> ModeParameters:
        """
        Get parameters for a specific mode
        
        Args:
            mode: Mode to get parameters for (current primary mode if None)
            
        Returns:
            Parameters for the specified mode
        """
        if mode is None:
            primary_mode, _ = self.context.mode_distribution.primary_mode
            mode = primary_mode
            
        try:
            mode_enum = InteractionMode(mode)
            params = self.context.mode_parameters.get(mode_enum)
            if params:
                return params
        except ValueError:
            # Check custom modes
            if mode in self.context.custom_modes:
                return self.context.custom_modes[mode]["parameters"]
                
        return DEFAULT_MODE_PARAMETERS[InteractionMode.DEFAULT]
    
    def get_conversation_style(self, mode: Optional[str] = None) -> ConversationStyle:
        """
        Get conversation style for a specific mode
        
        Args:
            mode: Mode to get style for (current primary mode if None)
            
        Returns:
            Conversation style for the specified mode
        """
        if mode is None:
            primary_mode, _ = self.context.mode_distribution.primary_mode
            mode = primary_mode
            
        try:
            mode_enum = InteractionMode(mode)
            style = self.context.conversation_styles.get(mode_enum)
            if style:
                return style
        except ValueError:
            # Check custom modes
            if mode in self.context.custom_modes:
                return self.context.custom_modes[mode]["conversation_style"]
                
        return DEFAULT_CONVERSATION_STYLES[InteractionMode.DEFAULT]
    
    def get_vocalization_patterns(self, mode: Optional[str] = None) -> VocalizationPatterns:
        """
        Get vocalization patterns for a specific mode
        
        Args:
            mode: Mode to get patterns for (current primary mode if None)
            
        Returns:
            Vocalization patterns for the specified mode
        """
        if mode is None:
            primary_mode, _ = self.context.mode_distribution.primary_mode
            mode = primary_mode
            
        try:
            mode_enum = InteractionMode(mode)
            patterns = self.context.vocalization_patterns.get(mode_enum)
            if patterns:
                return patterns
        except ValueError:
            # Check custom modes
            if mode in self.context.custom_modes:
                return self.context.custom_modes[mode]["vocalization_patterns"]
                
        return DEFAULT_VOCALIZATION_PATTERNS[InteractionMode.DEFAULT]
    
    async def get_blended_parameters(self) -> BlendedParameters:
        """
        Get blended parameters for the current mode distribution
        
        Returns:
            Blended parameters
        """
        param_prompt = f"""
        Calculate blended parameters for the current mode distribution:
        
        MODE DISTRIBUTION: {json.dumps(self.context.mode_distribution.model_dump())}
        
        Create a coherent blend of parameters that proportionally
        represents all active modes in the distribution.
        """
        
        result = await Runner.run(
            self.parameter_blender_agent, 
            param_prompt, 
            context=self.context
        )
        
        return result.final_output.parameters
    
    async def register_custom_mode(self, 
                                mode_name: str, 
                                parameters: ModeParameters, 
                                conversation_style: ConversationStyle, 
                                vocalization_patterns: VocalizationPatterns) -> bool:
        """
        Register a new custom interaction mode
        
        Args:
            mode_name: Name of the new mode
            parameters: Mode parameters
            conversation_style: Conversation style guidelines
            vocalization_patterns: Vocalization patterns
            
        Returns:
            Success status
        """
        try:
            # Validate inputs
            if not mode_name or not isinstance(mode_name, str):
                raise ValueError("Mode name must be a non-empty string")
                
            # Store custom mode
            self.context.custom_modes[mode_name.lower()] = {
                "parameters": parameters,
                "conversation_style": conversation_style,
                "vocalization_patterns": vocalization_patterns
            }
            
            logger.info(f"Registered custom mode: {mode_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering custom mode: {e}")
            return False
    
    async def get_mode_history(self, limit: int = 10) -> List[ModeSwitchRecord]:
        """
        Get the mode switch history
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of mode switch events
        """
        return self.context.mode_switch_history[-limit:] if self.context.mode_switch_history else []
    
    async def get_mode_stats(self) -> Dict[str, Any]:
        """
        Get statistics about mode usage
        
        Returns:
            Statistics about mode usage
        """
        # Count mode occurrences
        mode_counts = {}
        for entry in self.context.mode_switch_history:
            primary_mode, _ = entry.new_distribution.primary_mode
            mode_counts[primary_mode] = mode_counts.get(primary_mode, 0) + 1
        
        # Calculate mode stability
        stability = 0
        if len(self.context.mode_switch_history) >= 2:
            timestamps = [entry.timestamp for entry in self.context.mode_switch_history]
            durations = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            stability = sum(durations) / len(durations) if durations else 0
        
        # Get most common transitions
        transitions = {}
        for i in range(len(self.context.mode_switch_history)-1):
            prev_primary, _ = self.context.mode_switch_history[i].new_distribution.primary_mode
            next_primary, _ = self.context.mode_switch_history[i+1].new_distribution.primary_mode
            
            transition = f"{prev_primary}->{next_primary}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        # Sort transitions by count
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate distribution stats
        active_modes = self.context.mode_distribution.active_modes
        primary_mode, primary_weight = self.context.mode_distribution.primary_mode
        
        return {
            "current_distribution": self.context.mode_distribution.model_dump(),
            "primary_mode": primary_mode,
            "active_modes": active_modes,
            "mode_counts": mode_counts,
            "total_switches": len(self.context.mode_switch_history),
            "average_stability_seconds": stability,
            "common_transitions": dict(sorted_transitions[:5]),
            "custom_modes": list(self.context.custom_modes.keys()),
            "coherence_score": self.context.overall_confidence
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update configuration settings
        
        Args:
            config_updates: Dictionary of config keys to update
            
        Returns:
            Success status
        """
        try:
            valid_keys = {
                "mode_blend_factor",
                "significant_mode_threshold", 
                "significant_change_threshold",
                "max_history_size",
                "enable_legacy_mode"
            }
            
            for key, value in config_updates.items():
                if key in valid_keys:
                    self.context.config[key] = value
                    logger.info(f"Updated config {key} to {value}")
                else:
                    logger.warning(f"Ignoring invalid config key: {key}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
