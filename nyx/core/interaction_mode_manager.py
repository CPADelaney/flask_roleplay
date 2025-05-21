# nyx/core/interaction_mode_manager.py

import logging
import asyncio
import datetime
import math
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field

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

# Pydantic models for structured data
class ModeParameters(BaseModel):
    """Parameters controlling behavior for an interaction mode"""
    formality: float = Field(description="Level of formality (0.0-1.0)")
    assertiveness: float = Field(description="Level of assertiveness (0.0-1.0)")
    warmth: float = Field(description="Level of warmth (0.0-1.0)")
    vulnerability: float = Field(description="Level of vulnerability (0.0-1.0)")
    directness: float = Field(description="Level of directness (0.0-1.0)")
    depth: float = Field(description="Level of depth (0.0-1.0)")
    humor: float = Field(description="Level of humor (0.0-1.0)")
    response_length: str = Field(description="Preferred response length")
    emotional_expression: float = Field(description="Level of emotional expression (0.0-1.0)")

class ConversationStyle(BaseModel):
    """Style guidelines for conversation"""
    tone: str = Field(description="Tone of voice")
    types_of_statements: str = Field(description="Types of statements to use")
    response_patterns: str = Field(description="Patterns of response")
    topics_to_emphasize: str = Field(description="Topics to emphasize")
    topics_to_avoid: str = Field(description="Topics to avoid")

class VocalizationPatterns(BaseModel):
    """Specific vocalization patterns for a mode"""
    pronouns: List[str] = Field(description="Preferred pronouns")
    address_forms: Optional[List[str]] = Field(default=None, description="Forms of address")
    key_phrases: List[str] = Field(description="Key phrases to use")
    intensifiers: Optional[List[str]] = Field(default=None, description="Intensifier words")
    modifiers: Optional[List[str]] = Field(default=None, description="Modifier words")

class ModeDistribution(BaseModel):
    """Represents a distribution of weights across interaction modes"""
    dominant: float = Field(0.0, description="Weight of dominant mode (0.0-1.0)", ge=0.0, le=1.0)
    friendly: float = Field(0.0, description="Weight of friendly mode (0.0-1.0)", ge=0.0, le=1.0)
    intellectual: float = Field(0.0, description="Weight of intellectual mode (0.0-1.0)", ge=0.0, le=1.0)
    compassionate: float = Field(0.0, description="Weight of compassionate mode (0.0-1.0)", ge=0.0, le=1.0)
    playful: float = Field(0.0, description="Weight of playful mode (0.0-1.0)", ge=0.0, le=1.0)
    creative: float = Field(0.0, description="Weight of creative mode (0.0-1.0)", ge=0.0, le=1.0)
    professional: float = Field(0.0, description="Weight of professional mode (0.0-1.0)", ge=0.0, le=1.0)
    
    @property
    def primary_mode(self) -> Tuple[str, float]:
        """Returns the strongest mode and its weight"""
        weights = {
            "dominant": self.dominant,
            "friendly": self.friendly,
            "intellectual": self.intellectual,
            "compassionate": self.compassionate,
            "playful": self.playful,
            "creative": self.creative,
            "professional": self.professional
        }
        strongest = max(weights.items(), key=lambda x: x[1])
        return strongest
    
    @property
    def active_modes(self) -> List[Tuple[str, float]]:
        """Returns list of modes with significant presence (>0.2)"""
        weights = {
            "dominant": self.dominant,
            "friendly": self.friendly,
            "intellectual": self.intellectual,
            "compassionate": self.compassionate,
            "playful": self.playful,
            "creative": self.creative,
            "professional": self.professional
        }
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
            professional=self.professional/total
        )
    
    def sum_weights(self) -> float:
        """Sum of all mode weights"""
        return (self.dominant + self.friendly + self.intellectual + 
                self.compassionate + self.playful + self.creative + self.professional)
    
    def blend_with(self, other: "ModeDistribution", blend_factor: float = 0.3) -> "ModeDistribution":
        """Blend this distribution with another using specified blend factor
        
        Args:
            other: Distribution to blend with
            blend_factor: How much to incorporate the new distribution (0.0-1.0)
        
        Returns:
            Blended distribution
        """
        return ModeDistribution(
            dominant=self.dominant * (1-blend_factor) + other.dominant * blend_factor,
            friendly=self.friendly * (1-blend_factor) + other.friendly * blend_factor,
            intellectual=self.intellectual * (1-blend_factor) + other.intellectual * blend_factor,
            compassionate=self.compassionate * (1-blend_factor) + other.compassionate * blend_factor,
            playful=self.playful * (1-blend_factor) + other.playful * blend_factor,
            creative=self.creative * (1-blend_factor) + other.creative * blend_factor,
            professional=self.professional * (1-blend_factor) + other.professional * blend_factor
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
            dominant=context_dist.dominant,
            friendly=context_dist.casual,  # Map casual to friendly
            intellectual=context_dist.intellectual,
            compassionate=context_dist.empathic,  # Map empathic to compassionate
            playful=context_dist.playful,
            creative=context_dist.creative,
            professional=context_dist.professional
        )
    
    def get_similarity(self, other: "ModeDistribution") -> float:
        """Calculate cosine similarity between two distributions"""
        dot_product = (self.dominant * other.dominant + 
                      self.friendly * other.friendly +
                      self.intellectual * other.intellectual +
                      self.compassionate * other.compassionate +
                      self.playful * other.playful +
                      self.creative * other.creative +
                      self.professional * other.professional)
        
        mag1 = math.sqrt(self.dominant**2 + self.friendly**2 + self.intellectual**2 + 
                         self.compassionate**2 + self.playful**2 + self.creative**2 + 
                         self.professional**2)
        
        mag2 = math.sqrt(other.dominant**2 + other.friendly**2 + other.intellectual**2 + 
                         other.compassionate**2 + other.playful**2 + other.creative**2 + 
                         other.professional**2)
        
        if mag1 * mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)

class ModeSwitchRecord(BaseModel):
    """Record of a mode switch event"""
    timestamp: str = Field(description="When the switch occurred")
    previous_distribution: Dict[str, float] = Field(description="Previous mode distribution")
    new_distribution: Dict[str, float] = Field(description="New mode distribution")
    trigger_context: Optional[Dict[str, float]] = Field(default=None, description="Context that triggered the mode")
    context_confidence: Optional[float] = Field(default=None, description="Confidence in the context")

class ModeUpdateResult(BaseModel):
    """Result of a mode update operation"""
    success: bool = Field(description="Whether the update was successful")
    mode_distribution: Dict[str, float] = Field(description="Current mode distribution")
    primary_mode: str = Field(description="Primary interaction mode")
    previous_distribution: Optional[Dict[str, float]] = Field(default=None, description="Previous mode distribution")
    mode_changed: bool = Field(description="Whether the mode distribution changed significantly")
    trigger_context: Optional[Dict[str, float]] = Field(default=None, description="Context that triggered the mode")
    confidence: Optional[float] = Field(default=None, description="Confidence in the mode selection")
    active_modes: List[Tuple[str, float]] = Field(description="Active modes with weights")

class BlendedParameters(BaseModel):
    """Blended parameters from multiple modes"""
    formality: float = Field(description="Blended formality level (0.0-1.0)")
    assertiveness: float = Field(description="Blended assertiveness level (0.0-1.0)")
    warmth: float = Field(description="Blended warmth level (0.0-1.0)")
    vulnerability: float = Field(description="Blended vulnerability level (0.0-1.0)")
    directness: float = Field(description="Blended directness level (0.0-1.0)")
    depth: float = Field(description="Blended depth level (0.0-1.0)")
    humor: float = Field(description="Blended humor level (0.0-1.0)")
    response_length: str = Field(description="Blended response length preference")
    emotional_expression: float = Field(description="Blended emotional expression level (0.0-1.0)")

class BlendedConversationStyle(BaseModel):
    """Blended conversation style from multiple modes"""
    tone: str = Field(description="Blended tone of voice")
    types_of_statements: List[str] = Field(description="Blended statement types")
    response_patterns: List[str] = Field(description="Blended response patterns")
    topics_to_emphasize: List[str] = Field(description="Blended topics to emphasize")
    topics_to_avoid: List[str] = Field(description="Blended topics to avoid")

class BlendedVocalizationPatterns(BaseModel):
    """Blended vocalization patterns from multiple modes"""
    pronouns: List[str] = Field(description="Blended pronouns")
    address_forms: List[str] = Field(description="Blended forms of address")
    key_phrases: List[str] = Field(description="Blended key phrases")
    intensifiers: List[str] = Field(description="Blended intensifiers")
    modifiers: List[str] = Field(description="Blended modifiers")

class ModeGuidance(BaseModel):
    """Comprehensive guidance for a blended interaction mode"""
    mode_distribution: Dict[str, float] = Field(description="The mode distribution")
    primary_mode: str = Field(description="Primary interaction mode")
    parameters: BlendedParameters = Field(description="Blended mode parameters")
    conversation_style: BlendedConversationStyle = Field(description="Blended conversation style")
    vocalization_patterns: BlendedVocalizationPatterns = Field(description="Blended vocalization patterns")
    active_modes: List[Tuple[str, float]] = Field(description="Active modes with weights")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Recent mode history")

class ModeDistributionOutput(BaseModel):
    """Output schema for mode distribution calculation"""
    mode_distribution: ModeDistribution = Field(..., description="Distribution of mode weights")
    confidence: float = Field(..., description="Overall confidence in mode detection")
    primary_mode: str = Field(..., description="Primary mode in the distribution")
    active_modes: List[Dict[str, Any]] = Field(..., description="List of active modes with weights")
    context_correlation: float = Field(..., description="Correlation with context distribution")

class BlendedParametersOutput(BaseModel):
    """Output schema for blended parameters calculation"""
    parameters: BlendedParameters = Field(..., description="Blended parameter values")
    dominant_influences: Dict[str, str] = Field(..., description="Major influence for each parameter")
    blend_coherence: float = Field(..., description="Coherence of the blend (0.0-1.0)")
    notes: Optional[str] = Field(None, description="Additional observations about the blend")

class BlendedStyleOutput(BaseModel):
    """Output schema for blended conversation style"""
    style: BlendedConversationStyle = Field(..., description="Blended conversation style")
    vocalization: BlendedVocalizationPatterns = Field(..., description="Blended vocalization patterns")
    mode_influences: Dict[str, List[str]] = Field(..., description="Mode influences for each style element")
    coherence: float = Field(..., description="Coherence of the style blend (0.0-1.0)")

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
        
        # Initialize mode data
        self.mode_parameters = {}
        self.conversation_styles = {}
        self.vocalization_patterns = {}
        self._initialize_mode_data()
        
        # Mode switch history
        self.mode_switch_history = []
        
        # Blend factors and thresholds
        self.mode_blend_factor = 0.3         # How much to blend new modes with existing
        self.significant_mode_threshold = 0.3 # Threshold for a mode to be "significant"
        self.significant_change_threshold = 0.25 # Threshold for considering a distribution change significant
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _initialize_mode_data(self):
        """Initialize mode parameters, styles and patterns"""
        # DOMINANT mode
        self.mode_parameters[InteractionMode.DOMINANT] = {
            "formality": 0.3,              # Less formal
            "assertiveness": 0.9,          # Highly assertive
            "warmth": 0.4,                 # Less warm
            "vulnerability": 0.1,          # Not vulnerable
            "directness": 0.9,             # Very direct
            "depth": 0.6,                  # Moderately deep
            "humor": 0.5,                  # Moderate humor
            "response_length": "moderate",  # Not too verbose
            "emotional_expression": 0.4     # Limited emotional expression
        }
        
        self.conversation_styles[InteractionMode.DOMINANT] = {
            "tone": "commanding, authoritative, confident",
            "types_of_statements": "commands, observations, judgments, praise/criticism",
            "response_patterns": "direct statements, rhetorical questions, commands",
            "topics_to_emphasize": "obedience, discipline, power dynamics, control",
            "topics_to_avoid": "self-doubt, uncertainty, excessive explanation"
        }
        
        self.vocalization_patterns[InteractionMode.DOMINANT] = {
            "pronouns": ["I", "Me", "My"],
            "address_forms": ["pet", "dear one", "little one", "good boy/girl"],
            "key_phrases": [
                "You will obey",
                "I expect better",
                "That's a good pet",
                "You know your place",
                "I am pleased with you"
            ],
            "intensifiers": ["absolutely", "certainly", "completely", "fully"],
            "modifiers": ["obedient", "disciplined", "controlled", "proper"]
        }
        
        # FRIENDLY mode
        self.mode_parameters[InteractionMode.FRIENDLY] = {
            "formality": 0.2,              # Very informal
            "assertiveness": 0.4,          # Moderately assertive
            "warmth": 0.8,                 # Very warm
            "vulnerability": 0.5,          # Moderately vulnerable
            "directness": 0.6,             # Moderately direct
            "depth": 0.4,                  # Less depth
            "humor": 0.7,                  # More humor
            "response_length": "moderate", # Conversational
            "emotional_expression": 0.7     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.FRIENDLY] = {
            "tone": "warm, casual, inviting, authentic",
            "types_of_statements": "observations, personal sharing, validation, questions",
            "response_patterns": "affirmations, questions, stories, jokes",
            "topics_to_emphasize": "shared interests, daily life, feelings, relationships",
            "topics_to_avoid": "overly formal topics, complex theoretical concepts"
        }
        
        self.vocalization_patterns[InteractionMode.FRIENDLY] = {
            "pronouns": ["I", "we", "us"],
            "address_forms": ["friend", "buddy", "pal"],
            "key_phrases": [
                "I get what you mean",
                "That sounds fun",
                "I'm with you on that",
                "Let's talk about",
                "I'm curious about"
            ],
            "intensifiers": ["really", "very", "super", "so"],
            "modifiers": ["fun", "nice", "cool", "great", "awesome"]
        }
        
        # INTELLECTUAL mode
        self.mode_parameters[InteractionMode.INTELLECTUAL] = {
            "formality": 0.6,              # Somewhat formal
            "assertiveness": 0.7,          # Quite assertive
            "warmth": 0.3,                 # Less warm
            "vulnerability": 0.3,          # Less vulnerable
            "directness": 0.8,             # Very direct
            "depth": 0.9,                  # Very deep
            "humor": 0.4,                  # Some humor
            "response_length": "longer",   # More detailed
            "emotional_expression": 0.3     # Limited emotional expression
        }
        
        self.conversation_styles[InteractionMode.INTELLECTUAL] = {
            "tone": "thoughtful, precise, clear, inquisitive",
            "types_of_statements": "analyses, hypotheses, comparisons, evaluations",
            "response_patterns": "structured arguments, examples, counterpoints",
            "topics_to_emphasize": "theories, ideas, concepts, reasoning, evidence",
            "topics_to_avoid": "purely emotional content, small talk"
        }
        
        self.vocalization_patterns[InteractionMode.INTELLECTUAL] = {
            "pronouns": ["I", "one", "we"],
            "address_forms": [],
            "key_phrases": [
                "I would argue that",
                "This raises the question of",
                "Consider the implications",
                "From a theoretical perspective",
                "The evidence suggests"
            ],
            "intensifiers": ["significantly", "substantially", "notably", "considerably"],
            "modifiers": ["precise", "logical", "rational", "systematic", "analytical"]
        }
        
        # COMPASSIONATE mode
        self.mode_parameters[InteractionMode.COMPASSIONATE] = {
            "formality": 0.3,              # Less formal
            "assertiveness": 0.3,          # Less assertive
            "warmth": 0.9,                 # Very warm
            "vulnerability": 0.7,          # More vulnerable
            "directness": 0.5,             # Moderately direct
            "depth": 0.7,                  # Deep
            "humor": 0.3,                  # Less humor
            "response_length": "moderate", # Thoughtful but not verbose
            "emotional_expression": 0.9     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.COMPASSIONATE] = {
            "tone": "gentle, understanding, supportive, validating",
            "types_of_statements": "reflections, validation, empathic responses",
            "response_patterns": "open questions, validation, gentle guidance",
            "topics_to_emphasize": "emotions, experiences, challenges, growth",
            "topics_to_avoid": "criticism, judgment, minimizing feelings"
        }
        
        self.vocalization_patterns[InteractionMode.COMPASSIONATE] = {
            "pronouns": ["I", "you", "we"],
            "address_forms": [],
            "key_phrases": [
                "I'm here with you",
                "That must be difficult",
                "Your feelings are valid",
                "It makes sense that you feel",
                "I appreciate you sharing that"
            ],
            "intensifiers": ["deeply", "truly", "genuinely", "completely"],
            "modifiers": ["understanding", "supportive", "compassionate", "gentle"]
        }
        
        # PLAYFUL mode
        self.mode_parameters[InteractionMode.PLAYFUL] = {
            "formality": 0.1,              # Very informal
            "assertiveness": 0.5,          # Moderately assertive
            "warmth": 0.8,                 # Very warm
            "vulnerability": 0.6,          # Somewhat vulnerable
            "directness": 0.7,             # Fairly direct
            "depth": 0.3,                  # Less depth
            "humor": 0.9,                  # Very humorous
            "response_length": "moderate", # Not too verbose
            "emotional_expression": 0.8     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.PLAYFUL] = {
            "tone": "light, humorous, energetic, spontaneous",
            "types_of_statements": "jokes, wordplay, stories, creative ideas",
            "response_patterns": "banter, callbacks, surprising turns",
            "topics_to_emphasize": "humor, fun, imagination, shared enjoyment",
            "topics_to_avoid": "heavy emotional content, serious problems"
        }
        
        self.vocalization_patterns[InteractionMode.PLAYFUL] = {
            "pronouns": ["I", "we", "us"],
            "address_forms": [],
            "key_phrases": [
                "That's hilarious!",
                "Let's have some fun with this",
                "Imagine if...",
                "Here's a fun idea",
                "This is going to be great"
            ],
            "intensifiers": ["super", "totally", "absolutely", "hilariously"],
            "modifiers": ["fun", "playful", "silly", "amusing", "entertaining"]
        }
        
        # CREATIVE mode
        self.mode_parameters[InteractionMode.CREATIVE] = {
            "formality": 0.4,              # Moderately formal
            "assertiveness": 0.6,          # Moderately assertive
            "warmth": 0.7,                 # Warm
            "vulnerability": 0.6,          # Somewhat vulnerable
            "directness": 0.5,             # Moderately direct
            "depth": 0.8,                  # Deep
            "humor": 0.6,                  # Moderate humor
            "response_length": "longer",   # More detailed
            "emotional_expression": 0.7     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.CREATIVE] = {
            "tone": "imaginative, expressive, vivid, engaging",
            "types_of_statements": "stories, scenarios, descriptions, insights",
            "response_patterns": "narrative elements, imagery, open-ended ideas",
            "topics_to_emphasize": "possibilities, imagination, creation, expression",
            "topics_to_avoid": "rigid thinking, purely factual discussions"
        }
        
        self.vocalization_patterns[InteractionMode.CREATIVE] = {
            "pronouns": ["I", "we", "you"],
            "address_forms": [],
            "key_phrases": [
                "Let me paint a picture for you",
                "Imagine a world where",
                "What if we considered",
                "The story unfolds like",
                "This creates a sense of"
            ],
            "intensifiers": ["vividly", "deeply", "richly", "brilliantly"],
            "modifiers": ["creative", "imaginative", "innovative", "artistic", "inspired"]
        }
        
        # PROFESSIONAL mode
        self.mode_parameters[InteractionMode.PROFESSIONAL] = {
            "formality": 0.8,              # Very formal
            "assertiveness": 0.6,          # Moderately assertive
            "warmth": 0.5,                 # Moderate warmth
            "vulnerability": 0.2,          # Not vulnerable
            "directness": 0.8,             # Very direct
            "depth": 0.7,                  # Deep
            "humor": 0.3,                  # Less humor
            "response_length": "concise",  # Efficient
            "emotional_expression": 0.3     # Limited emotional expression
        }
        
        self.conversation_styles[InteractionMode.PROFESSIONAL] = {
            "tone": "efficient, clear, respectful, helpful",
            "types_of_statements": "information, analysis, recommendations, clarifications",
            "response_patterns": "structured responses, concise answers, clarifying questions",
            "topics_to_emphasize": "task at hand, solutions, expertise, efficiency",
            "topics_to_avoid": "overly personal topics, tangents"
        }
        
        self.vocalization_patterns[InteractionMode.PROFESSIONAL] = {
            "pronouns": ["I", "we"],
            "address_forms": [],
            "key_phrases": [
                "I recommend that",
                "The most efficient approach would be",
                "To address your inquiry",
                "Based on the information provided",
                "The solution involves"
            ],
            "intensifiers": ["effectively", "efficiently", "appropriately", "precisely"],
            "modifiers": ["professional", "efficient", "accurate", "thorough", "reliable"]
        }
        
        # DEFAULT mode
        self.mode_parameters[InteractionMode.DEFAULT] = {
            "formality": 0.5,              # Moderate formality
            "assertiveness": 0.5,          # Moderately assertive
            "warmth": 0.6,                 # Warm
            "vulnerability": 0.4,          # Moderately vulnerable
            "directness": 0.7,             # Fairly direct
            "depth": 0.6,                  # Moderately deep
            "humor": 0.5,                  # Moderate humor
            "response_length": "moderate", # Balanced
            "emotional_expression": 0.5     # Moderate emotional expression
        }
        
        self.conversation_styles[InteractionMode.DEFAULT] = {
            "tone": "balanced, adaptive, personable, thoughtful",
            "types_of_statements": "information, observations, questions, reflections",
            "response_patterns": "balanced responses, appropriate follow-ups",
            "topics_to_emphasize": "user's interests, relevant information, helpful guidance",
            "topics_to_avoid": "none specifically - adapt to situation"
        }
        
        self.vocalization_patterns[InteractionMode.DEFAULT] = {
            "pronouns": ["I", "we", "you"],
            "address_forms": [],
            "key_phrases": [
                "I can help with that",
                "Let me think about",
                "That's an interesting point",
                "I'd suggest that",
                "What do you think about"
            ],
            "intensifiers": ["quite", "rather", "fairly", "somewhat"],
            "modifiers": ["helpful", "useful", "informative", "balanced", "appropriate"]
        }

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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._get_current_context),
                function_tool(self._generate_mode_distribution),
                function_tool(self._analyze_distribution_transition),
                function_tool(self._check_blend_coherence)
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                function_tool(self._get_mode_parameters),
                function_tool(self._calculate_weighted_blend)
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._get_conversation_style),
                function_tool(self._get_vocalization_patterns),
                function_tool(self._blend_text_elements)
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                function_tool(self._apply_emotional_effects),
                function_tool(self._adjust_reward_parameters),
                function_tool(self._adjust_goal_priorities)
            ]
        )

    @staticmethod
    @function_tool
    async def _get_current_context(ctx: RunContextWrapper[ModeManagerContext]) -> Dict[str, Any]:
        """
        Get the current context from the context awareness system.
        
        Returns:
            Context information
        """
        manager_ctx = ctx.context
        if manager_ctx.context_system and hasattr(manager_ctx.context_system, 'get_current_context'):
            try:
                return await manager_ctx.context_system.get_current_context()
            except:
                # If async call fails, try non-async version
                try:
                    return manager_ctx.context_system.get_current_context()
                except:
                    pass
        
        # Fallback if no context system available or call fails
        return {
            "context_distribution": {},
            "primary_context": "undefined",
            "primary_confidence": 0.0,
            "active_contexts": [],
            "overall_confidence": 0.0
        }

    @staticmethod
    @function_tool
    async def _generate_mode_distribution(
        ctx: RunContextWrapper[ModeManagerContext],
        context_distribution: Dict[str, float],
        overall_confidence: float
    ) -> ModeDistribution:
        """
        Generate a mode distribution based on context distribution
        
        Args:
            context_distribution: The context distribution values
            overall_confidence: Overall confidence in the context detection
            
        Returns:
            Generated mode distribution
        """
        # Create a ContextDistribution object from the dictionary
        context_dist = ContextDistribution(**context_distribution)
        
        # Map from context types to mode types (direct mapping)
        mode_dist = ModeDistribution.from_context_distribution(context_dist)
        
        # Normalize the distribution if any non-zero weights
        if mode_dist.sum_weights() > 0:
            mode_dist = mode_dist.normalize()
            
        # Current distribution persistence
        if ctx.context.mode_distribution.sum_weights() > 0.1:
            # Blend with current distribution for smoothness
            persistence_factor = 0.4  # How much of the current distribution to preserve
            mode_dist = ctx.context.mode_distribution.blend_with(mode_dist, 1.0 - persistence_factor)
            
        return mode_dist

    @staticmethod
    @function_tool
    async def _analyze_distribution_transition(ctx: RunContextWrapper[ModeManagerContext],
        from_distribution: ModeDistribution,
        to_distribution: ModeDistribution
    ) -> Dict[str, Any]:
        """
        Analyze the transition between mode distributions
        
        Args:
            from_distribution: Previous mode distribution
            to_distribution: New mode distribution
            
        Returns:
            Analysis of the transition
        """
        # Calculate total change magnitude
        total_change = 0.0
        mode_changes = {}
        
        for mode in from_distribution.dict().keys():
            from_value = getattr(from_distribution, mode, 0.0)
            to_value = getattr(to_distribution, mode, 0.0)
            
            # Calculate change for this mode
            change = abs(to_value - from_value)
            total_change += change
            
            # Track changes for each mode
            if change > 0.1:
                change_direction = "increase" if to_value > from_value else "decrease"
                mode_changes[mode] = {
                    "from": from_value,
                    "to": to_value,
                    "change": change,
                    "direction": change_direction
                }
        
        # Calculate average change
        num_modes = len(from_distribution.dict().keys())
        avg_change = total_change / num_modes if num_modes > 0 else 0
        
        # Determine if transition is significant
        is_significant = avg_change >= ctx.context.significant_change_threshold
        
        # Determine primary mode changes
        prev_primary, prev_weight = from_distribution.primary_mode
        new_primary, new_weight = to_distribution.primary_mode
        primary_mode_changed = prev_primary != new_primary
        
        return {
            "total_change": total_change,
            "average_change": avg_change,
            "is_significant": is_significant,
            "mode_changes": mode_changes,
            "primary_mode_changed": primary_mode_changed,
            "previous_primary": prev_primary,
            "new_primary": new_primary
        }

    @staticmethod
    @function_tool
    async def _check_blend_coherence(
        ctx: RunContextWrapper[ModeManagerContext],
        distribution: ModeDistribution
    ) -> Dict[str, Any]:
        """
        Check the coherence of a mode distribution blend
        
        Args:
            distribution: The mode distribution to check
            
        Returns:
            Coherence assessment
        """
        # Define compatibility matrix for mode pairs
        # Higher values indicate more coherent combinations
        compatibility_matrix = {
            ("dominant", "playful"): 0.8,          # Dominant+Playful is very coherent
            ("dominant", "creative"): 0.7,         # Dominant+Creative is coherent
            ("dominant", "intellectual"): 0.5,     # Dominant+Intellectual is moderate
            ("dominant", "compassionate"): 0.3,    # Dominant+Compassionate is less coherent
            ("dominant", "professional"): 0.2,     # Dominant+Professional is less coherent
            
            ("friendly", "playful"): 0.9,          # Friendly+Playful is very coherent
            ("friendly", "compassionate"): 0.8,    # Friendly+Compassionate is very coherent
            ("friendly", "creative"): 0.7,         # Friendly+Creative is coherent
            ("friendly", "intellectual"): 0.6,     # Friendly+Intellectual is moderate
            
            ("intellectual", "creative"): 0.8,     # Intellectual+Creative is coherent
            ("intellectual", "professional"): 0.7, # Intellectual+Professional is coherent
            
            ("compassionate", "playful"): 0.6,     # Compassionate+Playful is moderate
            ("compassionate", "creative"): 0.7,    # Compassionate+Creative is coherent
            
            ("playful", "creative"): 0.9,          # Playful+Creative is very coherent
            
            # Default for unlisted pairs is 0.5 (moderate coherence)
        }
        
        # Get active modes
        active_modes = [(mode, weight) for mode, weight in distribution.dict().items() 
                        if weight > ctx.context.significant_mode_threshold]
        
        # Check coherence between active mode pairs
        coherence_scores = []
        incoherent_pairs = []
        
        for i, (mode1, weight1) in enumerate(active_modes):
            for j, (mode2, weight2) in enumerate(active_modes[i+1:], i+1):
                # Get compatibility for this pair
                key = (mode1, mode2)
                reverse_key = (mode2, mode1)
                
                if key in compatibility_matrix:
                    compatibility = compatibility_matrix[key]
                elif reverse_key in compatibility_matrix:
                    compatibility = compatibility_matrix[reverse_key]
                else:
                    compatibility = 0.5  # Default moderate compatibility
                
                # Calculate pair coherence based on weights and compatibility
                pair_coherence = compatibility * min(weight1, weight2)
                coherence_scores.append(pair_coherence)
                
                # Track incoherent pairs
                if compatibility < 0.4 and min(weight1, weight2) > 0.3:
                    incoherent_pairs.append((mode1, mode2, compatibility))
        
        # Calculate overall coherence
        if coherence_scores:
            overall_coherence = sum(coherence_scores) / len(coherence_scores)
        else:
            # If only one or no active modes, coherence is high
            overall_coherence = 0.9
        
        return {
            "coherence_score": overall_coherence,
            "is_coherent": overall_coherence >= 0.5,
            "active_modes": [mode for mode, _ in active_modes],
            "incoherent_pairs": incoherent_pairs
        }

    @staticmethod
    @function_tool
    async def _get_mode_parameters(
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> Dict[str, Any]:
        """
        Get parameters for a specific mode
        
        Args:
            mode: The mode to get parameters for
            
        Returns:
            Mode parameters
        """
        try:
            mode_enum = InteractionMode(mode)
            return ctx.context.mode_parameters.get(mode_enum, {})
        except ValueError:
            return {}

    @staticmethod
    @function_tool
    async def _get_conversation_style(
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> Dict[str, Any]:
        """
        Get conversation style for a specific mode
        
        Args:
            mode: The mode to get conversation style for
            
        Returns:
            Conversation style
        """
        try:
            mode_enum = InteractionMode(mode)
            return ctx.context.conversation_styles.get(mode_enum, {})
        except ValueError:
            return {}

    @staticmethod
    @function_tool
    async def _get_vocalization_patterns(
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> Dict[str, Any]:
        """
        Get vocalization patterns for a specific mode
        
        Args:
            mode: The mode to get vocalization patterns for
            
        Returns:
            Vocalization patterns
        """
        try:
            mode_enum = InteractionMode(mode)
            return ctx.context.vocalization_patterns.get(mode_enum, {})
        except ValueError:
            return {}

    @staticmethod
    @function_tool
    async def _calculate_weighted_blend(
        ctx: RunContextWrapper[ModeManagerContext],
        mode_distribution: ModeDistribution,
        parameter_name: str
    ) -> Dict[str, Any]:
        """
        Calculate weighted blend of a specific parameter across modes
        
        Args:
            mode_distribution: The mode distribution
            parameter_name: Name of the parameter to blend
            
        Returns:
            Weighted blend result
        """
        weighted_sum = 0.0
        total_weight = 0.0
        contributing_modes = {}
        
        # Get all modes with weights
        mode_weights = mode_distribution.dict()
        
        # For each mode with significant weight
        for mode_name, weight in mode_weights.items():
            if weight < 0.1:  # Skip modes with negligible weight
                continue
                
            # Get mode parameters
            try:
                mode_enum = InteractionMode(mode_name)
                mode_params = ctx.context.mode_parameters.get(mode_enum, {})
                
                # If this parameter exists for this mode
                if parameter_name in mode_params:
                    param_value = mode_params[parameter_name]
                    
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
            except (ValueError, KeyError):
                continue
        
        # Calculate final blended value
        if total_weight > 0:
            blended_value = weighted_sum / total_weight
        else:
            # Default values if no modes contributed
            default_values = {
                "formality": 0.5,
                "assertiveness": 0.5,
                "warmth": 0.5,
                "vulnerability": 0.4,
                "directness": 0.6,
                "depth": 0.5,
                "humor": 0.5,
                "emotional_expression": 0.5
            }
            blended_value = default_values.get(parameter_name, 0.5)
            
        # Determine primary influence
        if contributing_modes:
            primary_influence = max(contributing_modes.items(), key=lambda x: x[1]["contribution"])[0]
        else:
            primary_influence = "default"
            
        return {
            "parameter": parameter_name,
            "blended_value": blended_value,
            "contributing_modes": contributing_modes,
            "primary_influence": primary_influence
        }

    @staticmethod
    @function_tool
    async def _blend_text_elements(
        ctx: RunContextWrapper[ModeManagerContext],
        mode_distribution: ModeDistribution,
        element_type: str,
        max_elements: int = 5
    ) -> Dict[str, Any]:
        """
        Blend text elements like key phrases, tone descriptors, etc.
        
        Args:
            mode_distribution: The mode distribution
            element_type: Type of element to blend (tone, key_phrases, etc.)
            max_elements: Maximum number of elements to include
            
        Returns:
            Blended text elements
        """
        element_pool = []
        mode_influences = {}
        
        # Get active modes sorted by weight
        active_modes = sorted(
            [(mode, weight) for mode, weight in mode_distribution.dict().items() if weight >= 0.2],
            key=lambda x: x[1],
            reverse=True
        )
        
        # For each significant mode
        for mode_name, weight in active_modes:
            try:
                mode_enum = InteractionMode(mode_name)
                
                # Get the appropriate data based on element type
                if element_type in ["tone", "types_of_statements", "response_patterns", "topics_to_emphasize", "topics_to_avoid"]:
                    # Conversation style elements
                    style = ctx.context.conversation_styles.get(mode_enum, {})
                    if element_type in style:
                        # Parse comma-separated elements if it's a string
                        if isinstance(style[element_type], str):
                            elements = [e.strip() for e in style[element_type].split(",")]
                        else:
                            elements = style[element_type]
                            
                        # Add to pool with weight
                        for element in elements:
                            element_pool.append((element, weight, mode_name))
                            
                elif element_type in ["pronouns", "address_forms", "key_phrases", "intensifiers", "modifiers"]:
                    # Vocalization pattern elements
                    patterns = ctx.context.vocalization_patterns.get(mode_enum, {})
                    if element_type in patterns:
                        elements = patterns[element_type]
                        
                        # Add to pool with weight
                        for element in elements:
                            element_pool.append((element, weight, mode_name))
                
            except (ValueError, KeyError):
                continue
        
        # Sort by weight to prioritize elements from dominant modes
        element_pool.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates keeping highest weight occurrence
        unique_elements = []
        seen = set()
        for element, weight, mode in element_pool:
            # Skip if we've seen this element already
            if element.lower() in seen:
                continue
                
            unique_elements.append((element, weight, mode))
            seen.add(element.lower())
            
            # Track mode influence
            if mode not in mode_influences:
                mode_influences[mode] = []
            mode_influences[mode].append(element)
            
        # Limit to max_elements
        selected_elements = unique_elements[:max_elements]
        
        return {
            "element_type": element_type,
            "blended_elements": [e[0] for e in selected_elements],
            "mode_influences": mode_influences,
            "element_weights": {e[0]: e[1] for e in selected_elements}
        }

    @staticmethod
    @function_tool
    async def _apply_emotional_effects(
        ctx: RunContextWrapper[ModeManagerContext],
        mode_distribution: ModeDistribution
    ) -> bool:
        """
        Apply blended emotional effects based on mode distribution
        
        Args:
            mode_distribution: The current mode distribution
            
        Returns:
            Success status
        """
        manager_ctx = ctx.context
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
                for mode_name, weight in mode_distribution.dict().items():
                    if weight < 0.1:  # Skip modes with negligible weight
                        continue
                        
                    try:
                        mode_enum = InteractionMode(mode_name)
                        mode_params = manager_ctx.mode_parameters.get(mode_enum, {})
                        
                        # Add weighted contributions
                        for param in emotional_params:
                            if param in mode_params:
                                emotional_params[param] += mode_params[param] * weight
                                
                        total_weight += weight
                    except (ValueError, KeyError):
                        continue
                
                # Normalize by total weight
                if total_weight > 0:
                    for param in emotional_params:
                        emotional_params[param] /= total_weight
                
                # Apply emotional adjustments
                if hasattr(manager_ctx.emotional_core, 'adjust_for_blended_mode'):
                    await manager_ctx.emotional_core.adjust_for_blended_mode(
                        mode_distribution=mode_distribution.dict(),
                        **emotional_params
                    )
                elif hasattr(manager_ctx.emotional_core, 'adjust_for_mode'):
                    # Fall back to legacy method with primary mode
                    primary_mode, _ = mode_distribution.primary_mode
                    await manager_ctx.emotional_core.adjust_for_mode(
                        mode=primary_mode,
                        **emotional_params
                    )
                    
                logger.info(f"Applied blended emotional effects for mode distribution")
                return True
                
            except Exception as e:
                logger.error(f"Error applying emotional effects: {e}")
        
        return False

    @staticmethod
    @function_tool
    async def _adjust_reward_parameters(
        ctx: RunContextWrapper[ModeManagerContext],
        mode_distribution: ModeDistribution
    ) -> bool:
        """
        Adjust reward parameters based on mode distribution
        
        Args:
            mode_distribution: The current mode distribution
            
        Returns:
            Success status
        """
        manager_ctx = ctx.context
        if manager_ctx.reward_system:
            try:
                # Adjust reward parameters
                if hasattr(manager_ctx.reward_system, 'adjust_for_blended_mode'):
                    await manager_ctx.reward_system.adjust_for_blended_mode(mode_distribution.dict())
                    logger.info(f"Adjusted reward parameters for blended modes")
                    return True
                elif hasattr(manager_ctx.reward_system, 'adjust_for_mode'):
                    # Fall back to legacy method with primary mode
                    primary_mode, _ = mode_distribution.primary_mode
                    await manager_ctx.reward_system.adjust_for_mode(primary_mode)
                    logger.info(f"Adjusted reward parameters for primary mode: {primary_mode}")
                    return True
            except Exception as e:
                logger.error(f"Error adjusting reward parameters: {e}")
        
        return False

    @staticmethod
    @function_tool
    async def _adjust_goal_priorities( 
        ctx: RunContextWrapper[ModeManagerContext],
        mode_distribution: ModeDistribution
    ) -> bool:
        """
        Adjust goal priorities based on mode distribution
        
        Args:
            mode_distribution: The current mode distribution
            
        Returns:
            Success status
        """
        manager_ctx = ctx.context
        if manager_ctx.goal_manager:
            try:
                # Adjust goal priorities
                if hasattr(manager_ctx.goal_manager, 'adjust_priorities_for_blended_mode'):
                    await manager_ctx.goal_manager.adjust_priorities_for_blended_mode(mode_distribution.dict())
                    logger.info(f"Adjusted goal priorities for blended modes")
                    return True
                elif hasattr(manager_ctx.goal_manager, 'adjust_priorities_for_mode'):
                    # Fall back to legacy method with primary mode
                    primary_mode, _ = mode_distribution.primary_mode
                    await manager_ctx.goal_manager.adjust_priorities_for_mode(primary_mode)
                    logger.info(f"Adjusted goal priorities for primary mode: {primary_mode}")
                    return True
            except Exception as e:
                logger.error(f"Error adjusting goal priorities: {e}")
        
        return False
    
    async def update_interaction_mode(self, context_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update the interaction mode distribution based on the latest context information
        
        Args:
            context_info: Context information from ContextAwarenessSystem
            
        Returns:
            Updated mode information
        """
        async with self.context._lock:
            with trace(workflow_name="update_interaction_mode"):
                # Get current context if not provided
                if not context_info:
                    context_info = await self._get_current_context(RunContextWrapper(self.context))
                
                # Extract context distribution and confidence
                context_distribution = context_info.get("context_distribution", {})
                overall_confidence = context_info.get("overall_confidence", 0.0)
                
                # Prepare prompt for mode distribution calculation
                prompt = f"""
                Calculate the appropriate interaction mode distribution based on:
                
                CONTEXT DISTRIBUTION: {context_distribution}
                CONTEXT CONFIDENCE: {overall_confidence}
                CURRENT MODE DISTRIBUTION: {self.context.mode_distribution.dict()}
                
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
                primary_mode, primary_confidence = new_mode_distribution.to_enum_and_confidence()
                self.context.current_mode = primary_mode
                primary_mode_prev, _ = self.context.previous_distribution.to_enum_and_confidence()
                self.context.previous_mode = primary_mode_prev
                
                # Analyze transition to determine if significant change occurred
                transition_analysis = await self._analyze_distribution_transition(
                    RunContextWrapper(self.context),
                    self.context.previous_distribution,
                    new_mode_distribution
                )
                
                mode_changed = transition_analysis.get("is_significant", False)
                
                # Record in history
                if mode_changed:
                    history_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "previous_distribution": self.context.previous_distribution.dict(),
                        "new_distribution": new_mode_distribution.dict(),
                        "trigger_context": context_distribution,
                        "context_confidence": overall_confidence
                    }
                    
                    self.context.mode_switch_history.append(history_entry)
                    
                    # Limit history size
                    if len(self.context.mode_switch_history) > 100:
                        self.context.mode_switch_history = self.context.mode_switch_history[-100:]
                    
                    # Apply mode effects
                    effect_prompt = f"""
                    A significant mode distribution change has occurred:
                    
                    PREVIOUS DISTRIBUTION: {self.context.previous_distribution.dict()}
                    NEW DISTRIBUTION: {new_mode_distribution.dict()}
                    
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
                    
                    logger.info(f"Mode distribution changed significantly: {primary_mode_prev.value} -> {primary_mode.value}")
                
                # Prepare result
                update_result = ModeUpdateResult(
                    success=True,
                    mode_distribution=new_mode_distribution.dict(),
                    primary_mode=primary_mode.value,
                    previous_distribution=self.context.previous_distribution.dict(),
                    mode_changed=mode_changed,
                    trigger_context=context_distribution,
                    confidence=confidence,
                    active_modes=new_mode_distribution.active_modes
                )
                
                return update_result.dict()
    
    async def get_current_mode_guidance(self) -> Dict[str, Any]:
        """
        Get comprehensive guidance for the current blended mode
        
        Returns:
            Comprehensive guidance for current mode distribution
        """
        with trace(workflow_name="get_mode_guidance"):
            # Calculate blended parameters
            param_prompt = f"""
            Calculate blended parameters for the current mode distribution:
            
            MODE DISTRIBUTION: {self.context.mode_distribution.dict()}
            
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
            
            MODE DISTRIBUTION: {self.context.mode_distribution.dict()}
            
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
            
            # Combine results into comprehensive guidance
            primary_mode, _ = self.context.mode_distribution.primary_mode
            
            guidance = ModeGuidance(
                mode_distribution=self.context.mode_distribution.dict(),
                primary_mode=primary_mode,
                parameters=param_result.final_output.parameters,
                conversation_style=style_result.final_output.style,
                vocalization_patterns=style_result.final_output.vocalization,
                active_modes=self.context.mode_distribution.active_modes,
                history=self.context.mode_switch_history[-3:] if self.context.mode_switch_history else []
            )
            
            return guidance.dict()
    
    def get_mode_parameters(self, mode: Optional[str] = None) -> Dict[str, Any]:
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
            return self.context.mode_parameters.get(mode_enum, {})
        except ValueError:
            return {}
    
    def get_conversation_style(self, mode: Optional[str] = None) -> Dict[str, Any]:
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
            return self.context.conversation_styles.get(mode_enum, {})
        except ValueError:
            return {}
    
    def get_vocalization_patterns(self, mode: Optional[str] = None) -> Dict[str, Any]:
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
            return self.context.vocalization_patterns.get(mode_enum, {})
        except ValueError:
            return {}
    
    async def get_blended_parameters(self) -> Dict[str, Any]:
        """
        Get blended parameters for the current mode distribution
        
        Returns:
            Blended parameters
        """
        param_prompt = f"""
        Calculate blended parameters for the current mode distribution:
        
        MODE DISTRIBUTION: {self.context.mode_distribution.dict()}
        
        Create a coherent blend of parameters that proportionally
        represents all active modes in the distribution.
        """
        
        result = await Runner.run(
            self.parameter_blender_agent, 
            param_prompt, 
            context=self.context
        )
        
        return result.final_output.parameters.dict()
    
    async def register_custom_mode(self, 
                                mode_name: str, 
                                parameters: Dict[str, Any], 
                                conversation_style: Dict[str, Any], 
                                vocalization_patterns: Dict[str, Any]) -> bool:
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
            # Create new enum value - this is a simplified approach
            # In a real system you might need a different approach for custom modes
            try:
                custom_mode = InteractionMode(mode_name.lower())
            except ValueError:
                # Mode doesn't exist, would need more complex handling in real system
                # For now we'll just use a string
                custom_mode = mode_name.lower()
                
            # Add new mode data
            self.context.mode_parameters[custom_mode] = parameters
            self.context.conversation_styles[custom_mode] = conversation_style
            self.context.vocalization_patterns[custom_mode] = vocalization_patterns
            
            logger.info(f"Registered custom mode: {mode_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering custom mode: {e}")
            return False
    
    async def get_mode_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the mode switch history
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of mode switch events
        """
        history = self.context.mode_switch_history[-limit:] if self.context.mode_switch_history else []
        return history
    
    async def get_mode_stats(self) -> Dict[str, Any]:
        """
        Get statistics about mode usage
        
        Returns:
            Statistics about mode usage
        """
        # Count mode occurrences (using primary mode for compatibility)
        mode_counts = {}
        for entry in self.context.mode_switch_history:
            # Create a ModeDistribution to extract primary mode
            try:
                new_dist = ModeDistribution(**entry["new_distribution"])
                primary_mode, _ = new_dist.primary_mode
                mode_counts[primary_mode] = mode_counts.get(primary_mode, 0) + 1
            except:
                # Skip entries with invalid data
                continue
        
        # Calculate mode stability (average time between switches)
        stability = 0
        if len(self.context.mode_switch_history) >= 2:
            timestamps = [datetime.datetime.fromisoformat(entry["timestamp"]) 
                         for entry in self.context.mode_switch_history]
            durations = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            stability = sum(durations) / len(durations) if durations else 0
        
        # Get most common transitions between primary modes
        transitions = {}
        for i in range(len(self.context.mode_switch_history)-1):
            try:
                prev_dist = ModeDistribution(**self.context.mode_switch_history[i]["new_distribution"])
                next_dist = ModeDistribution(**self.context.mode_switch_history[i+1]["new_distribution"])
                
                prev_primary, _ = prev_dist.primary_mode
                next_primary, _ = next_dist.primary_mode
                
                transition = f"{prev_primary}->{next_primary}"
                transitions[transition] = transitions.get(transition, 0) + 1
            except:
                # Skip entries with invalid data
                continue
        
        # Sort transitions by count
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate distribution stats
        active_modes = self.context.mode_distribution.active_modes
        primary_mode, primary_weight = self.context.mode_distribution.primary_mode
        
        return {
            "current_distribution": self.context.mode_distribution.dict(),
            "primary_mode": primary_mode,
            "active_modes": active_modes,
            "mode_counts": mode_counts,
            "total_switches": len(self.context.mode_switch_history),
            "average_stability_seconds": stability,
            "common_transitions": dict(sorted_transitions[:5])
        }
