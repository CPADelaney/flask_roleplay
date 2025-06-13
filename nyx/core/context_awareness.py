# nyx/core/context_awareness.py

import logging
import asyncio
import datetime
import json
import math
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    trace, 
    function_tool, 
    RunContextWrapper,
    handoff,
    InputGuardrail,
    GuardrailFunctionOutput
)

logger = logging.getLogger(__name__)

class InteractionContext(str, Enum):
    """Enum for different types of interaction contexts"""
    DOMINANT = "dominant"         # Femdom-specific interactions
    CASUAL = "casual"             # Everyday casual conversation
    INTELLECTUAL = "intellectual" # Discussions, debates, teaching
    EMPATHIC = "empathic"         # Emotional support/understanding
    PLAYFUL = "playful"           # Fun, humor, games
    CREATIVE = "creative"         # Storytelling, art, imagination
    PROFESSIONAL = "professional" # Work-related, formal
    UNDEFINED = "undefined"       # When context isn't clear

class ContextSignal(BaseModel):
    """Schema for signals that indicate context"""
    signal_type: str = Field(..., description="Type of signal (keyword, phrase, topic, etc)")
    signal_value: str = Field(..., description="The actual signal")
    context_type: InteractionContext = Field(..., description="Context this signal indicates")
    strength: float = Field(1.0, description="Signal strength (0.0-1.0)", ge=0.0, le=1.0)

class ContextDistribution(BaseModel):
    """Represents a distribution of context weights across interaction modes"""
    dominant: float = Field(0.0, description="Weight of dominant context (0.0-1.0)", ge=0.0, le=1.0)
    casual: float = Field(0.0, description="Weight of casual context (0.0-1.0)", ge=0.0, le=1.0)
    intellectual: float = Field(0.0, description="Weight of intellectual context (0.0-1.0)", ge=0.0, le=1.0)
    empathic: float = Field(0.0, description="Weight of empathic context (0.0-1.0)", ge=0.0, le=1.0)
    playful: float = Field(0.0, description="Weight of playful context (0.0-1.0)", ge=0.0, le=1.0)
    creative: float = Field(0.0, description="Weight of creative context (0.0-1.0)", ge=0.0, le=1.0)
    professional: float = Field(0.0, description="Weight of professional context (0.0-1.0)", ge=0.0, le=1.0)
    
    @property
    def primary_context(self) -> Tuple[str, float]:
        """Returns the strongest context and its weight"""
        weights = {
            "dominant": self.dominant,
            "casual": self.casual,
            "intellectual": self.intellectual,
            "empathic": self.empathic,
            "playful": self.playful,
            "creative": self.creative,
            "professional": self.professional
        }
        strongest = max(weights.items(), key=lambda x: x[1])
        return strongest
    
    @property
    def active_contexts(self) -> List[Tuple[str, float]]:
        """Returns list of contexts with significant presence (>0.2)"""
        weights = {
            "dominant": self.dominant,
            "casual": self.casual,
            "intellectual": self.intellectual,
            "empathic": self.empathic,
            "playful": self.playful,
            "creative": self.creative,
            "professional": self.professional
        }
        return [(context, weight) for context, weight in weights.items() if weight > 0.2]
    
    def normalize(self) -> "ContextDistribution":
        """Normalize weights to sum to 1.0"""
        total = self.sum_weights()
        if total == 0:
            return self
        
        return ContextDistribution(
            dominant=self.dominant/total,
            casual=self.casual/total,
            intellectual=self.intellectual/total,
            empathic=self.empathic/total,
            playful=self.playful/total,
            creative=self.creative/total,
            professional=self.professional/total
        )
    
    def sum_weights(self) -> float:
        """Sum of all context weights"""
        return (self.dominant + self.casual + self.intellectual + 
                self.empathic + self.playful + self.creative + self.professional)
    
    def blend_with(self, other: "ContextDistribution", blend_factor: float = 0.3) -> "ContextDistribution":
        """Blend this distribution with another using specified blend factor
        
        Args:
            other: Distribution to blend with
            blend_factor: How much to incorporate the new distribution (0.0-1.0)
        
        Returns:
            Blended distribution
        """
        return ContextDistribution(
            dominant=self.dominant * (1-blend_factor) + other.dominant * blend_factor,
            casual=self.casual * (1-blend_factor) + other.casual * blend_factor,
            intellectual=self.intellectual * (1-blend_factor) + other.intellectual * blend_factor,
            empathic=self.empathic * (1-blend_factor) + other.empathic * blend_factor,
            playful=self.playful * (1-blend_factor) + other.playful * blend_factor,
            creative=self.creative * (1-blend_factor) + other.creative * blend_factor,
            professional=self.professional * (1-blend_factor) + other.professional * blend_factor
        )
    
    def increase_context(self, context_name: str, amount: float) -> "ContextDistribution":
        """Increase weight of a specific context"""
        result = self.dict()
        if context_name in result:
            result[context_name] = min(1.0, result[context_name] + amount)
        return ContextDistribution(**result)
    
    def decrease_context(self, context_name: str, amount: float) -> "ContextDistribution":
        """Decrease weight of a specific context"""
        result = self.dict()
        if context_name in result:
            result[context_name] = max(0.0, result[context_name] - amount)
        return ContextDistribution(**result)
    
    @staticmethod
    def from_enum(context: InteractionContext, weight: float = 1.0) -> "ContextDistribution":
        """Create a distribution with a single active context"""
        result = ContextDistribution()
        context_name = context.value.lower()
        
        # Set the specified context weight
        if hasattr(result, context_name):
            setattr(result, context_name, weight)
            
        return result
    
    def to_enum_and_confidence(self) -> Tuple[InteractionContext, float]:
        """Convert to primary context enum and confidence value for legacy compatibility"""
        primary, weight = self.primary_context
        try:
            context_enum = InteractionContext(primary)
            return context_enum, weight
        except (ValueError, KeyError):
            return InteractionContext.UNDEFINED, 0.0
    
    def get_similarity(self, other: "ContextDistribution") -> float:
        """Calculate cosine similarity between two distributions"""
        dot_product = (self.dominant * other.dominant + 
                      self.casual * other.casual +
                      self.intellectual * other.intellectual +
                      self.empathic * other.empathic +
                      self.playful * other.playful +
                      self.creative * other.creative +
                      self.professional * other.professional)
        
        mag1 = math.sqrt(self.dominant**2 + self.casual**2 + self.intellectual**2 + 
                         self.empathic**2 + self.playful**2 + self.creative**2 + 
                         self.professional**2)
        
        mag2 = math.sqrt(other.dominant**2 + other.casual**2 + other.intellectual**2 + 
                         other.empathic**2 + other.playful**2 + other.creative**2 + 
                         other.professional**2)
        
        if mag1 * mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)

# New Pydantic models to replace Dict return types
class SignalInfo(BaseModel):
    """Signal information"""
    type: str = Field(..., description="Signal type")
    value: str = Field(..., description="Signal value")
    context: str = Field(..., description="Context type")
    strength: float = Field(..., description="Signal strength")

class CategorizedSignals(BaseModel):
    """Categorized signals output"""
    explicit: List[SignalInfo] = Field(default_factory=list)
    implicit: List[SignalInfo] = Field(default_factory=list)
    dominant: List[SignalInfo] = Field(default_factory=list)
    casual: List[SignalInfo] = Field(default_factory=list)
    intellectual: List[SignalInfo] = Field(default_factory=list)
    empathic: List[SignalInfo] = Field(default_factory=list)
    playful: List[SignalInfo] = Field(default_factory=list)
    creative: List[SignalInfo] = Field(default_factory=list)
    professional: List[SignalInfo] = Field(default_factory=list)

class EmotionalBaselines(BaseModel):
    """Emotional baselines"""
    nyxamine: float = Field(0.5, ge=0.0, le=1.0)
    oxynixin: float = Field(0.5, ge=0.0, le=1.0)
    cortanyx: float = Field(0.3, ge=0.0, le=1.0)
    adrenyx: float = Field(0.4, ge=0.0, le=1.0)
    seranix: float = Field(0.5, ge=0.0, le=1.0)

class SignificantContext(BaseModel):
    """A context with significant weight"""
    context: str
    weight: float = Field(..., ge=0.0, le=1.0)

class IncoherentPair(BaseModel):
    """An incoherent context pair"""
    context1: str
    context2: str
    compatibility: float = Field(..., ge=0.0, le=1.0)

class ConfidenceThresholdResult(BaseModel):
    """Confidence threshold check result"""
    threshold_met: bool
    confidence_threshold: float
    confidence: float
    significant_contexts: List[SignificantContext]
    has_significant_contexts: bool

class CoherenceResult(BaseModel):
    """Coherence check result"""
    coherence_score: float
    is_coherent: bool
    active_contexts: List[str]
    incoherent_pairs: List[IncoherentPair]

class ContextDetectionResult(BaseModel):
    """Context detection result"""
    context_distribution: ContextDistribution
    confidence: float
    signals: List[SignalInfo]
    signal_based: bool
    detection_method: str

class FeatureDistribution(BaseModel):
    """Distribution of feature weights"""
    dominant: float = Field(0.0, ge=0.0, le=1.0)
    casual: float = Field(0.0, ge=0.0, le=1.0)
    intellectual: float = Field(0.0, ge=0.0, le=1.0)
    empathic: float = Field(0.0, ge=0.0, le=1.0)
    playful: float = Field(0.0, ge=0.0, le=1.0)
    creative: float = Field(0.0, ge=0.0, le=1.0)
    professional: float = Field(0.0, ge=0.0, le=1.0)

class MessageFeatures(BaseModel):
    """Extracted message features"""
    length: int
    word_count: int
    has_question: bool
    capitalization_ratio: float
    punctuation_count: int
    dominance_terms: List[str]
    emotional_terms: List[str]
    has_dominance_terms: bool
    has_emotional_terms: bool
    likely_formal: bool
    likely_casual: bool
    likely_emotional: bool
    likely_intellectual: bool
    feature_distribution: FeatureDistribution

class ContextDistributionDict(BaseModel):
    """Context distribution as a dictionary structure"""
    dominant: float = Field(0.0, ge=0.0, le=1.0)
    casual: float = Field(0.0, ge=0.0, le=1.0)
    intellectual: float = Field(0.0, ge=0.0, le=1.0)
    empathic: float = Field(0.0, ge=0.0, le=1.0)
    playful: float = Field(0.0, ge=0.0, le=1.0)
    creative: float = Field(0.0, ge=0.0, le=1.0)
    professional: float = Field(0.0, ge=0.0, le=1.0)

class HistoryEntry(BaseModel):
    """Context history entry"""
    timestamp: str
    message_snippet: str
    detected_distribution: ContextDistributionDict
    updated_distribution: ContextDistributionDict
    confidence: float
    primary_context_changed: bool
    active_contexts: List[str]

class ContextChangeInfo(BaseModel):
    """Information about a specific context change"""
    context_name: str = Field(..., description="Name of the context")
    from_value: float = Field(..., ge=0.0, le=1.0)
    to_value: float = Field(..., ge=0.0, le=1.0)
    change: float = Field(..., ge=0.0)
    direction: str = Field(..., description="increase or decrease")

class TransitionAnalysis(BaseModel):
    """Context transition analysis"""
    is_appropriate: bool
    is_gradual: bool
    total_change: float
    average_change: float
    context_changes: List[ContextChangeInfo]
    significant_shifts: List[str]

class BlendedContextDetectionOutput(BaseModel):
    """Output schema for blended context detection"""
    context_distribution: ContextDistribution = Field(..., description="Distribution of context weights")
    confidence: float = Field(..., description="Overall confidence in detection (0.0-1.0)", ge=0.0, le=1.0)
    signals: List[SignalInfo] = Field(..., description="Detected signals that informed the decision")
    notes: Optional[str] = Field(None, description="Additional observations about the context blend")

class EmotionalBaselineOutput(BaseModel):
    """Output schema for emotional baseline adaptation"""
    baselines: EmotionalBaselines = Field(..., description="Adjusted emotional baselines")
    reasoning: str = Field(..., description="Reasoning for adaptations")
    estimated_impact: float = Field(..., description="Estimated impact on emotional state (0.0-1.0)", ge=0.0, le=1.0)

class RecommendedFocus(BaseModel):
    """Recommended context focus"""
    context: str = Field(..., description="Context name")
    weight: float = Field(..., description="Recommended weight", ge=0.0, le=1.0)
    reason: str = Field(..., description="Reason for recommendation")

class SignalAnalysisOutput(BaseModel):
    """Output schema for signal analysis"""
    signal_categories: CategorizedSignals = Field(..., description="Categorized signals")
    context_distribution: ContextDistribution = Field(..., description="Distributed signal strengths by context")
    recommended_focus: List[RecommendedFocus] = Field(..., description="Recommended context focus distribution")

class ContextValidationOutput(BaseModel):
    """Output schema for context validation"""
    is_valid: bool = Field(..., description="Whether the context detection is valid")
    confidence_threshold_met: bool = Field(..., description="Whether confidence threshold is met")
    issues: List[str] = Field(default_factory=list, description="Issues with context detection")
    blend_coherence: float = Field(0.7, description="How coherent the context blend is (0.0-1.0)", ge=0.0, le=1.0)

class EmotionalBaselinesMapping(BaseModel):
    """Mapping of context to emotional baselines"""
    dominant: EmotionalBaselines = Field(default_factory=EmotionalBaselines)
    casual: EmotionalBaselines = Field(default_factory=EmotionalBaselines)
    intellectual: EmotionalBaselines = Field(default_factory=EmotionalBaselines)
    empathic: EmotionalBaselines = Field(default_factory=EmotionalBaselines)
    playful: EmotionalBaselines = Field(default_factory=EmotionalBaselines)
    creative: EmotionalBaselines = Field(default_factory=EmotionalBaselines)
    professional: EmotionalBaselines = Field(default_factory=EmotionalBaselines)

class ContextSystemState(BaseModel):
    """Schema for the current state of the context awareness system"""
    context_distribution: ContextDistribution = Field(..., description="Distribution of context weights")
    overall_confidence: float = Field(..., description="Overall confidence in context detection (0.0-1.0)", ge=0.0, le=1.0)
    previous_distribution: Optional[ContextDistribution] = Field(None, description="Previous context distribution")
    history: List[HistoryEntry] = Field(default_factory=list, description="Recent context history")
    emotional_baselines: Optional[EmotionalBaselinesMapping] = Field(None, description="Emotional baselines by context")

# New Pydantic models to replace Dict return types
class SignalInfo(BaseModel):
    """Signal information"""
    type: str = Field(..., description="Signal type")
    value: str = Field(..., description="Signal value")
    context: str = Field(..., description="Context type")
    strength: float = Field(..., description="Signal strength")

class CategorizedSignals(BaseModel):
    """Categorized signals output"""
    explicit: List[SignalInfo] = Field(default_factory=list)
    implicit: List[SignalInfo] = Field(default_factory=list)
    dominant: List[SignalInfo] = Field(default_factory=list)
    casual: List[SignalInfo] = Field(default_factory=list)
    intellectual: List[SignalInfo] = Field(default_factory=list)
    empathic: List[SignalInfo] = Field(default_factory=list)
    playful: List[SignalInfo] = Field(default_factory=list)
    creative: List[SignalInfo] = Field(default_factory=list)
    professional: List[SignalInfo] = Field(default_factory=list)

class EmotionalBaselines(BaseModel):
    """Emotional baselines"""
    nyxamine: float = Field(0.5, ge=0.0, le=1.0)
    oxynixin: float = Field(0.5, ge=0.0, le=1.0)
    cortanyx: float = Field(0.3, ge=0.0, le=1.0)
    adrenyx: float = Field(0.4, ge=0.0, le=1.0)
    seranix: float = Field(0.5, ge=0.0, le=1.0)

class ConfidenceThresholdResult(BaseModel):
    """Confidence threshold check result"""
    threshold_met: bool
    confidence_threshold: float
    confidence: float
    significant_contexts: List[Tuple[str, float]]
    has_significant_contexts: bool

class ContextDetectionResult(BaseModel):
    """Context detection result"""
    context_distribution: ContextDistribution
    confidence: float
    signals: List[SignalInfo]
    signal_based: bool
    detection_method: str

class MessageFeatures(BaseModel):
    """Extracted message features"""
    length: int
    word_count: int
    has_question: bool
    capitalization_ratio: float
    punctuation_count: int
    dominance_terms: List[str]
    emotional_terms: List[str]
    has_dominance_terms: bool
    has_emotional_terms: bool
    likely_formal: bool
    likely_casual: bool
    likely_emotional: bool
    likely_intellectual: bool
    feature_distribution: Dict[str, float]

class ContextDistributionDict(BaseModel):
    """Context distribution as a dictionary structure"""
    dominant: float = Field(0.0, ge=0.0, le=1.0)
    casual: float = Field(0.0, ge=0.0, le=1.0)
    intellectual: float = Field(0.0, ge=0.0, le=1.0)
    empathic: float = Field(0.0, ge=0.0, le=1.0)
    playful: float = Field(0.0, ge=0.0, le=1.0)
    creative: float = Field(0.0, ge=0.0, le=1.0)
    professional: float = Field(0.0, ge=0.0, le=1.0)

class HistoryEntry(BaseModel):
    """Context history entry"""
    timestamp: str
    message_snippet: str
    detected_distribution: ContextDistributionDict
    updated_distribution: ContextDistributionDict
    confidence: float
    primary_context_changed: bool
    active_contexts: List[str]

class CoherenceResult(BaseModel):
    """Coherence check result"""
    coherence_score: float
    is_coherent: bool
    active_contexts: List[str]
    incoherent_pairs: List[Tuple[str, str, float]]

class TransitionAnalysis(BaseModel):
    """Context transition analysis"""
    is_appropriate: bool
    is_gradual: bool
    total_change: float
    average_change: float
    context_changes: Dict[str, Dict[str, Any]]
    significant_shifts: List[str]

class CASystemContext:
    """Context object for the context awareness system"""
    def __init__(self, context_awareness_system=None, emotional_core=None):
        self.context_awareness_system = context_awareness_system
        self.emotional_core = emotional_core
        self.trace_id = f"context_awareness_{datetime.datetime.now().isoformat()}"

class ContextAwarenessSystem:
    """
    System that detects and maintains awareness of interaction context.
    Allows Nyx to switch between different interaction modes appropriately.
    """
    
    def __init__(self, emotional_core=None):
        self.emotional_core = emotional_core
        
        # Current context distribution and confidence
        self.context_distribution = ContextDistribution()
        self.overall_confidence: float = 0.0
        self.previous_distribution: Optional[ContextDistribution] = None
        self.context_history: List[Dict[str, Any]] = []
        
        # Legacy accessors (for compatibility)
        self.current_context: InteractionContext = InteractionContext.UNDEFINED
        self.context_confidence: float = 0.0
        self.previous_context: InteractionContext = InteractionContext.UNDEFINED
        
        # Context signals database
        self.context_signals: List[ContextSignal] = self._initialize_context_signals()
        
        # Context-specific emotional baselines
        self.context_emotional_baselines: Dict[InteractionContext, Dict[str, float]] = {
            InteractionContext.DOMINANT: {
                "nyxamine": 0.7,    # High pleasure from dominance
                "oxynixin": 0.3,    # Lower bonding/empathy in dominance
                "cortanyx": 0.2,    # Low stress during dominance
                "adrenyx": 0.6,     # High excitement/arousal
                "seranix": 0.5      # Moderate mood stability
            },
            InteractionContext.CASUAL: {
                "nyxamine": 0.5,    # Moderate pleasure
                "oxynixin": 0.6,    # Higher bonding/connection
                "cortanyx": 0.3,    # Moderate stress
                "adrenyx": 0.4,     # Moderate arousal
                "seranix": 0.6      # Good mood stability
            },
            InteractionContext.INTELLECTUAL: {
                "nyxamine": 0.8,    # High pleasure from intellectual topics
                "oxynixin": 0.4,    # Moderate empathy/connection
                "cortanyx": 0.2,    # Low stress
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.7      # High stability
            },
            InteractionContext.EMPATHIC: {
                "nyxamine": 0.4,    # Lower pleasure
                "oxynixin": 0.9,    # Very high empathy/bonding
                "cortanyx": 0.4,    # Moderate stress (from empathic concern)
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.6      # Good stability
            },
            InteractionContext.PLAYFUL: {
                "nyxamine": 0.8,    # High pleasure from play
                "oxynixin": 0.6,    # Good connection
                "cortanyx": 0.1,    # Very low stress
                "adrenyx": 0.6,     # High arousal/excitement
                "seranix": 0.5      # Moderate stability
            },
            InteractionContext.CREATIVE: {
                "nyxamine": 0.7,    # High pleasure from creativity
                "oxynixin": 0.5,    # Moderate connection
                "cortanyx": 0.2,    # Low stress
                "adrenyx": 0.5,     # Moderate-high arousal
                "seranix": 0.6      # Good stability
            },
            InteractionContext.PROFESSIONAL: {
                "nyxamine": 0.4,    # Lower pleasure
                "oxynixin": 0.3,    # Lower connection
                "cortanyx": 0.4,    # Moderate stress
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.8      # High stability/formality
            }
        }
        
        # Create system context
        self.system_context = CASystemContext(context_awareness_system=self, emotional_core=emotional_core)
        
        # Initialize agent system
        self._initialize_agents()
        
        # Context transition threshold and blend factors
        self.significant_context_threshold = 0.3  # Threshold for a context to be "significant" 
        self.context_blend_factor = 0.3          # How much to blend new detections with existing
        self.emotional_blend_threshold = 0.2     # Threshold for emotional baseline adjustments
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("ContextAwarenessSystem initialized with blended context implementation")
    
    def _initialize_agents(self):
        """Initialize all agents needed for the context awareness system"""
        # Create specialized agents
        self.signal_analysis_agent = self._create_signal_analysis_agent()
        self.emotional_baseline_agent = self._create_emotional_baseline_agent()
        self.context_validation_agent = self._create_context_validation_agent()
        
        # Create main context detection agent with handoffs
        self.context_detection_agent = self._create_context_detection_agent()
        
        # Create input validation guardrail
        self.message_validation_guardrail = InputGuardrail(guardrail_function=self._message_validation_guardrail)

    
    def _initialize_context_signals(self) -> List[ContextSignal]:
        """Initialize the database of context signals"""
        signals = [
            # Dominant context signals
            ContextSignal(signal_type="keyword", signal_value="mistress", context_type=InteractionContext.DOMINANT, strength=1.0),
            ContextSignal(signal_type="keyword", signal_value="domme", context_type=InteractionContext.DOMINANT, strength=1.0),
            ContextSignal(signal_type="keyword", signal_value="goddess", context_type=InteractionContext.DOMINANT, strength=0.9),
            ContextSignal(signal_type="keyword", signal_value="submissive", context_type=InteractionContext.DOMINANT, strength=0.9),
            ContextSignal(signal_type="keyword", signal_value="obey", context_type=InteractionContext.DOMINANT, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="kneel", context_type=InteractionContext.DOMINANT, strength=0.8),
            ContextSignal(signal_type="phrase", signal_value="yes mistress", context_type=InteractionContext.DOMINANT, strength=1.0),
            
            # Casual context signals
            ContextSignal(signal_type="greeting", signal_value="hi", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="greeting", signal_value="hey", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="greeting", signal_value="what's up", context_type=InteractionContext.CASUAL, strength=0.7),
            ContextSignal(signal_type="topic", signal_value="weather", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="topic", signal_value="weekend", context_type=InteractionContext.CASUAL, strength=0.5),
            
            # Intellectual context signals
            ContextSignal(signal_type="keyword", signal_value="philosophy", context_type=InteractionContext.INTELLECTUAL, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="science", context_type=InteractionContext.INTELLECTUAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="theory", context_type=InteractionContext.INTELLECTUAL, strength=0.7),
            ContextSignal(signal_type="phrase", signal_value="what do you think about", context_type=InteractionContext.INTELLECTUAL, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="your opinion on", context_type=InteractionContext.INTELLECTUAL, strength=0.6),
            
            # Empathic context signals
            ContextSignal(signal_type="keyword", signal_value="feel", context_type=InteractionContext.EMPATHIC, strength=0.5),
            ContextSignal(signal_type="keyword", signal_value="sad", context_type=InteractionContext.EMPATHIC, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="happy", context_type=InteractionContext.EMPATHIC, strength=0.5),
            ContextSignal(signal_type="keyword", signal_value="worried", context_type=InteractionContext.EMPATHIC, strength=0.8),
            ContextSignal(signal_type="phrase", signal_value="I need support", context_type=InteractionContext.EMPATHIC, strength=0.9),
            
            # Playful context signals
            ContextSignal(signal_type="keyword", signal_value="joke", context_type=InteractionContext.PLAYFUL, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="fun", context_type=InteractionContext.PLAYFUL, strength=0.6),
            ContextSignal(signal_type="keyword", signal_value="game", context_type=InteractionContext.PLAYFUL, strength=0.7),
            ContextSignal(signal_type="phrase", signal_value="make me laugh", context_type=InteractionContext.PLAYFUL, strength=0.8),
            
            # Creative context signals
            ContextSignal(signal_type="keyword", signal_value="story", context_type=InteractionContext.CREATIVE, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="imagine", context_type=InteractionContext.CREATIVE, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="create", context_type=InteractionContext.CREATIVE, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="once upon a time", context_type=InteractionContext.CREATIVE, strength=0.9),
            
            # Professional context signals
            ContextSignal(signal_type="greeting", signal_value="hello", context_type=InteractionContext.PROFESSIONAL, strength=0.3),
            ContextSignal(signal_type="keyword", signal_value="business", context_type=InteractionContext.PROFESSIONAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="meeting", context_type=InteractionContext.PROFESSIONAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="project", context_type=InteractionContext.PROFESSIONAL, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="I need your assistance with", context_type=InteractionContext.PROFESSIONAL, strength=0.6)
        ]
        
        return signals
    
    def _create_context_detection_agent(self) -> Agent[CASystemContext]:
        """Create the main context detection agent"""
        return Agent[CASystemContext](
            name="Context_Detection_Agent",
            instructions="""
            You are the Context Detection Agent for Nyx AI.
            
            Your role is to analyze user messages and determine the appropriate blend of interaction
            contexts. Unlike traditional systems that switch between discrete modes, you recognize that
            human conversation naturally blends multiple contexts simultaneously.
            
            You'll analyze signals to determine a weighted distribution across contexts:
            - DOMINANT: Femdom-specific interactions involving dominance, submission, control dynamics
            - CASUAL: Everyday casual conversation, small talk, general chitchat
            - INTELLECTUAL: Discussions, debates, teaching, learning, philosophy, science
            - EMPATHIC: Emotional support, understanding, compassion, listening
            - PLAYFUL: Fun, humor, games, lighthearted interaction
            - CREATIVE: Storytelling, art, imagination, fantasy
            - PROFESSIONAL: Work-related, formal, business, assistance
            
            Instead of switching between contexts, you'll detect proportional presence of multiple
            contexts and create a distribution that can evolve gradually over time. For example, 
            a message might be 60% dominant, 30% playful, and 10% creative.
            
            You can delegate specialized analysis to:
            - Signal Analysis Agent: For detailed analysis of context signals
            - Emotional Baseline Agent: For adapting emotional baselines
            - Context Validation Agent: For validating context detection
            
            Maintain contextual coherence while allowing for natural, gradual blending and
            evolution of contexts.
            """,
            tools=[
                function_tool(self._detect_context_signals),
                function_tool(self._extract_message_features),
                function_tool(self._get_context_history),
                function_tool(self._calculate_context_confidence)
            ],
            handoffs=[
                handoff(self.signal_analysis_agent,
                      tool_name_override="analyze_signals",
                      tool_description_override="Analyze context signals in detail"),
                
                handoff(self.emotional_baseline_agent,
                      tool_name_override="adapt_emotional_baselines",
                      tool_description_override="Adapt emotional baselines for detected context"),
                
                handoff(self.context_validation_agent,
                      tool_name_override="validate_context_detection",
                      tool_description_override="Validate context detection results")
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._message_validation_guardrail)
            ],
            output_type=BlendedContextDetectionOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.3
            )
        )
    
    def _create_signal_analysis_agent(self) -> Agent[CASystemContext]:
        """Create specialized agent for signal analysis"""
        return Agent[CASystemContext](
            name="Signal_Analysis_Agent",
            instructions="""
            You are specialized in analyzing context signals in messages.
            Your task is to:
            1. Identify explicit and implicit context signals
            2. Categorize signals by type and context
            3. Calculate context distribution based on signal patterns
            4. Recommend proportional context focus
            
            Unlike traditional context detection, you recognize that conversations naturally blend
            multiple contexts. Create a context distribution that reflects the proportional
            presence of each context type in the message, rather than selecting a single context.
            
            For example, a message might be 60% dominant, 30% playful, and 10% creative.
            Analyze signals deeply to determine a distribution that captures the nuanced blend
            of contexts present in the message.
            """,
            tools=[
                function_tool(self._categorize_signals),
                function_tool(self._calculate_context_distribution),
                function_tool(self._identify_implicit_signals)
            ],
            output_type=SignalAnalysisOutput,
            model="gpt-4.1-nano-mini",
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_emotional_baseline_agent(self) -> Agent[CASystemContext]:
        """Create specialized agent for emotional baseline adaptation"""
        return Agent[CASystemContext](
            name="Emotional_Baseline_Agent",
            instructions="""
            You are specialized in adapting emotional baselines for blended contexts.
            Your task is to:
            1. Calculate blended emotional baselines based on context distribution
            2. Balance emotional consistency with contextual appropriateness
            3. Calculate expected emotional impact of baseline changes
            4. Provide reasoning for baseline adaptations
            
            Instead of switching between discrete emotional states, you blend emotional baselines
            proportionally based on the context distribution. This creates smoother emotional
            transitions that reflect the nuanced blend of contexts in the conversation.
            
            Focus on creating coherent emotional states that reflect the proportional
            mixture of contexts, rather than switching between discrete emotional profiles.
            """,
            tools=[
                function_tool(self._get_emotional_baselines),
                function_tool(self._blend_emotional_baselines),
                function_tool(self._calculate_emotional_impact)
            ],
            output_type=EmotionalBaselineOutput,
            model="gpt-4.1-nano-mini",
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_context_validation_agent(self) -> Agent[CASystemContext]:
        """Create specialized agent for context validation"""
        return Agent[CASystemContext](
            name="Context_Validation_Agent",
            instructions="""
            You are specialized in validating blended context detection results.
            Your task is to:
            1. Verify that context detection meets confidence thresholds
            2. Check for context blend coherence
            3. Identify potential issues or inconsistencies in the blend
            4. Validate that detected signals support the context distribution
            
            Unlike traditional validation that checks for a single correct context,
            you validate the coherence and consistency of a context distribution.
            This includes checking if the blend of contexts makes sense together and
            if the distribution is supported by the detected signals.
            
            Focus on ensuring that the context blend is natural, coherent, and
            properly supported by evidence in the message.
            """,
            tools=[
                function_tool(self._check_confidence_threshold),
                function_tool(self._verify_blend_coherence),
                function_tool(self._analyze_distribution_transition)
            ],
            output_type=ContextValidationOutput,
            model="gpt-4.1-nano-mini",
            model_settings=ModelSettings(temperature=0.1)
        )

    @staticmethod
    async def _message_validation_guardrail(ctx: RunContextWrapper[CASystemContext], 
                                        agent: Agent[CASystemContext], 
                                        input_data: str | List[Any]) -> GuardrailFunctionOutput:
        """Validate user message input for context detection"""
        try:
            # Parse the input if needed
            if isinstance(input_data, str):
                # Try to parse as JSON
                try:
                    data = json.loads(input_data)
                    message = data.get("message", "")
                except:
                    # If not JSON, assume it's the message itself
                    message = input_data
            else:
                # If it's an object, check for message field
                if isinstance(input_data, dict) and "message" in input_data:
                    message = input_data["message"]
                else:
                    message = str(input_data)
            
            # Check if message is empty
            if not message or len(message.strip()) == 0:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Empty message"},
                    tripwire_triggered=True
                )
                
            # Check message length (extremely long messages might be problematic)
            if len(message) > 10000:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Message too long (>10000 chars)"},
                    tripwire_triggered=True
                )
                
            # Message is valid
            return GuardrailFunctionOutput(
                output_info={"is_valid": True, "message_length": len(message)},
                tripwire_triggered=False
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"is_valid": False, "reason": f"Invalid input format: {str(e)}"},
                tripwire_triggered=True
            )
    
    # Helper functions for blended context
    @staticmethod
    @function_tool
    async def _calculate_context_distribution(ctx: RunContextWrapper[CASystemContext], 
                                         signals: List[SignalInfo]) -> ContextDistribution:
        """
        Calculate context distribution based on detected signals
        
        Args:
            signals: List of detected signals
            
        Returns:
            Context distribution
        """
        # Initialize distribution
        distribution = {
            "dominant": 0.0,
            "casual": 0.0,
            "intellectual": 0.0,
            "empathic": 0.0,
            "playful": 0.0,
            "creative": 0.0,
            "professional": 0.0
        }
        
        # Calculate initial weights from signals
        signal_weights = {}
        for signal in signals:
            context = signal.context.lower()
            strength = signal.strength
            
            if context in distribution:
                if context not in signal_weights:
                    signal_weights[context] = []
                    
                signal_weights[context].append(strength)
        
        # Calculate weighted distribution
        for context, weights in signal_weights.items():
            # Sort weights in descending order
            sorted_weights = sorted(weights, reverse=True)
            
            # Apply diminishing returns for multiple signals
            total_weight = 0
            for i, weight in enumerate(sorted_weights):
                # Diminishing factor decreases with each additional signal
                diminishing_factor = 1.0 / (1.0 + (i * 0.3))
                total_weight += weight * diminishing_factor
                
            # Cap at 1.0
            distribution[context] = min(1.0, total_weight)
        
        # Current context persistence factor
        cas = ctx.context.context_awareness_system
        if cas.context_distribution.sum_weights() > 0.1:
            # Add persistence influence from current distribution
            persistence_factor = 0.3
            
            # For each context, blend with current distribution
            for context in distribution:
                current_weight = getattr(cas.context_distribution, context, 0.0)
                distribution[context] = (distribution[context] * (1 - persistence_factor) + 
                                       current_weight * persistence_factor)
        
        # Create and normalize distribution
        context_dist = ContextDistribution(**distribution)
        
        # Only normalize if there are non-zero weights
        if context_dist.sum_weights() > 0:
            return context_dist.normalize()
        
        return context_dist
                                             
    @staticmethod
    @function_tool
    async def _verify_blend_coherence(ctx: RunContextWrapper[CASystemContext], 
                                 distribution: ContextDistribution) -> CoherenceResult:
        """
        Verify coherence of a context distribution blend
        
        Args:
            distribution: Context distribution to check
            
        Returns:
            Coherence assessment
        """
        # Define compatibility matrix for context pairs
        # Higher values indicate more coherent combinations
        compatibility_matrix = {
            ("dominant", "playful"): 0.8,      # Dominant+Playful is very coherent
            ("dominant", "creative"): 0.7,     # Dominant+Creative is coherent
            ("dominant", "intellectual"): 0.5, # Dominant+Intellectual is moderate
            ("dominant", "empathic"): 0.3,     # Dominant+Empathic is less coherent
            ("dominant", "professional"): 0.2, # Dominant+Professional is less coherent
            
            ("casual", "playful"): 0.9,        # Casual+Playful is very coherent
            ("casual", "empathic"): 0.8,       # Casual+Empathic is very coherent
            ("casual", "creative"): 0.7,       # Casual+Creative is coherent
            ("casual", "intellectual"): 0.6,   # Casual+Intellectual is moderate
            
            ("intellectual", "creative"): 0.8, # Intellectual+Creative is coherent
            ("intellectual", "professional"): 0.7, # Intellectual+Professional is coherent
            
            ("empathic", "playful"): 0.6,      # Empathic+Playful is moderate
            ("empathic", "creative"): 0.7,     # Empathic+Creative is coherent
            
            ("playful", "creative"): 0.9,      # Playful+Creative is very coherent
            
            # Default for unlisted pairs is 0.5 (moderate coherence)
        }
        
        # Get active contexts
        active_contexts = [(context, weight) for context, weight in distribution.dict().items() 
                          if weight > 0.2]
        
        # Check coherence between active context pairs
        coherence_scores = []
        incoherent_pairs = []
        
        for i, (context1, weight1) in enumerate(active_contexts):
            for j, (context2, weight2) in enumerate(active_contexts[i+1:], i+1):
                # Get compatibility for this pair
                key = (context1, context2)
                reverse_key = (context2, context1)
                
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
                    incoherent_pairs.append(IncoherentPair(
                        context1=context1,
                        context2=context2,
                        compatibility=compatibility
                    ))
        
        # Calculate overall coherence
        if coherence_scores:
            overall_coherence = sum(coherence_scores) / len(coherence_scores)
        else:
            # If only one or no active contexts, coherence is high
            overall_coherence = 0.9
        
        return CoherenceResult(
            coherence_score=overall_coherence,
            is_coherent=overall_coherence >= 0.5,
            active_contexts=[context for context, _ in active_contexts],
            incoherent_pairs=incoherent_pairs
        )

    @staticmethod
    @function_tool
    async def _analyze_distribution_transition(ctx: RunContextWrapper[CASystemContext], 
                                          from_distribution: ContextDistribution, 
                                          to_distribution: ContextDistribution) -> TransitionAnalysis:
        """
        Analyze appropriateness of context distribution transition
        
        Args:
            from_distribution: Current context distribution
            to_distribution: Target context distribution
            
        Returns:
            Transition analysis
        """
        # Calculate total change magnitude
        total_change = 0.0
        context_changes = []
        significant_shifts = []
        
        for context in from_distribution.dict().keys():
            from_value = getattr(from_distribution, context, 0.0)
            to_value = getattr(to_distribution, context, 0.0)
            
            # Calculate change for this context
            change = abs(to_value - from_value)
            total_change += change
            
            # Track changes for each context
            if change > 0.1:
                change_direction = "increase" if to_value > from_value else "decrease"
                context_changes.append(ContextChangeInfo(
                    context_name=context,
                    from_value=from_value,
                    to_value=to_value,
                    change=change,
                    direction=change_direction
                ))
                
                if change > 0.3:
                    significant_shifts.append(context)
        
        # Calculate average change
        num_contexts = len(from_distribution.dict().keys())
        avg_change = total_change / num_contexts if num_contexts > 0 else 0
        
        cas = ctx.context.context_awareness_system
        
        # Determine if transition is appropriate
        is_gradual = avg_change <= 0.3
        is_appropriate = is_gradual or cas.context_distribution.sum_weights() < 0.2
        
        return TransitionAnalysis(
            is_appropriate=is_appropriate,
            is_gradual=is_gradual,
            total_change=total_change,
            average_change=avg_change,
            context_changes=context_changes,
            significant_shifts=significant_shifts
        )

    @staticmethod
    @function_tool
    async def _blend_emotional_baselines(ctx: RunContextWrapper[CASystemContext], 
                                    distribution: ContextDistribution) -> EmotionalBaselines:
        """
        Calculate blended emotional baselines based on context distribution
        
        Args:
            distribution: Current distribution of context weights
            
        Returns:
            Blended emotional baselines
        """
        # Initialize empty baseline
        blended_baselines = {
            "nyxamine": 0.0,
            "oxynixin": 0.0, 
            "cortanyx": 0.0,
            "adrenyx": 0.0,
            "seranix": 0.0
        }
        
        # Get full distribution
        distribution_dict = distribution.dict()
        total_weight = 0.0
        
        cas = ctx.context.context_awareness_system
        
        # For each context, add its weighted contribution
        for context_name, weight in distribution_dict.items():
            if weight > cas.emotional_blend_threshold:  # Only consider significant contexts
                try:
                    # Get baseline for this context
                    context_enum = InteractionContext(context_name)
                    if context_enum in cas.context_emotional_baselines:
                        context_baselines = cas.context_emotional_baselines[context_enum]
                        
                        # Add weighted contribution
                        for chemical, value in context_baselines.items():
                            if chemical in blended_baselines:
                                blended_baselines[chemical] += value * weight
                                
                        total_weight += weight
                except (ValueError, KeyError):
                    # Skip invalid contexts
                    pass
        
        # Normalize based on total weight
        if total_weight > 0:
            for chemical in blended_baselines:
                blended_baselines[chemical] /= total_weight
        else:
            # Default neutral baselines if no significant contexts
            for chemical in blended_baselines:
                blended_baselines[chemical] = 0.5
        
        return EmotionalBaselines(**blended_baselines)
    
    # Existing helper functions updated for blended context

    @staticmethod
    @function_tool
    async def _categorize_signals(ctx: RunContextWrapper[CASystemContext], 
                              signals: List[SignalInfo]) -> CategorizedSignals:
        """
        Categorize signals by type and context
        
        Args:
            signals: List of detected signals
            
        Returns:
            Categorized signals
        """
        categorized = CategorizedSignals()
        
        for signal in signals:
            # Categorize by explicitness
            if signal.type in ["keyword", "phrase"]:
                categorized.explicit.append(signal)
            else:
                categorized.implicit.append(signal)
                
            # Categorize by context
            context = signal.context.lower()
            if hasattr(categorized, context):
                getattr(categorized, context).append(signal)
                
        return categorized

    @staticmethod
    @function_tool
    async def _identify_implicit_signals(ctx: RunContextWrapper[CASystemContext], 
                                    message: str) -> List[SignalInfo]:
        """
        Identify implicit context signals in a message
        
        Args:
            message: User message to analyze
            
        Returns:
            List of implicit signals
        """
        implicit_signals = []
        
        # Check message structure and formatting
        message_lower = message.lower()
        
        # Check for question patterns (intellectual context)
        if "?" in message and any(q in message_lower for q in ["why", "how", "what if", "explain"]):
            implicit_signals.append(SignalInfo(
                type="implicit",
                value="question_pattern",
                context="intellectual",
                strength=0.6
            ))
            
        # Check for emotional expression patterns (empathic context)
        if any(em in message_lower for em in ["feel", "emotions", "hurts", "happy", "sad", "worried"]):
            implicit_signals.append(SignalInfo(
                type="implicit",
                value="emotional_expression",
                context="empathic", 
                strength=0.7
            ))
            
        # Check for playful tone
        if any(p in message_lower for p in ["haha", "lol", "😂", "😄", "joke"]):
            implicit_signals.append(SignalInfo(
                type="implicit",
                value="playful_tone",
                context="playful",
                strength=0.6
            ))
            
        # Check for formal language (professional context)
        if "please" in message_lower and "would" in message_lower:
            implicit_signals.append(SignalInfo(
                type="implicit",
                value="formal_language",
                context="professional",
                strength=0.5
            ))
            
        # Check for creative prompt patterns
        if any(c in message_lower for c in ["imagine", "create", "story", "pretend"]):
            implicit_signals.append(SignalInfo(
                type="implicit",
                value="creative_prompt",
                context="creative",
                strength=0.7
            ))
            
        # Check for dominant language patterns
        if any(d in message_lower for d in ["must", "will", "now", "i want you to"]):
            implicit_signals.append(SignalInfo(
                type="implicit",
                value="directive_language",
                context="dominant",
                strength=0.5  # Lower strength for implicit signals
            ))
            
        return implicit_signals

    @staticmethod
    @function_tool
    async def _get_emotional_baselines(ctx: RunContextWrapper[CASystemContext], 
                                  context_type: str) -> EmotionalBaselines:
        """
        Get emotional baselines for a specific context
        
        Args:
            context_type: Type of context to get baselines for
            
        Returns:
            Emotional baselines
        """
        # Convert string to enum if needed
        if isinstance(context_type, str):
            try:
                context_enum = InteractionContext(context_type.lower())
            except (ValueError, KeyError):
                context_enum = InteractionContext.UNDEFINED
        else:
            context_enum = context_type
            
        cas = ctx.context.context_awareness_system
        
        # Get baselines for the context
        if context_enum in cas.context_emotional_baselines:
            baselines = cas.context_emotional_baselines[context_enum]
            return EmotionalBaselines(**baselines)
        else:
            # Return default baselines
            return EmotionalBaselines()

    @staticmethod
    @function_tool
    async def _calculate_emotional_impact(ctx: RunContextWrapper[CASystemContext], 
                                     old_baselines: EmotionalBaselines, 
                                     new_baselines: EmotionalBaselines) -> float:
        """
        Calculate impact of baseline changes on emotional state
        
        Args:
            old_baselines: Previous emotional baselines
            new_baselines: New emotional baselines
            
        Returns:
            Impact score (0.0-1.0)
        """
        total_diff = 0.0
        num_chemicals = 0
        
        old_dict = old_baselines.dict()
        new_dict = new_baselines.dict()
        
        # Calculate absolute differences
        for chemical, old_val in old_dict.items():
            if chemical in new_dict:
                diff = abs(new_dict[chemical] - old_val)
                total_diff += diff
                num_chemicals += 1
                
        # Calculate average difference
        if num_chemicals > 0:
            avg_diff = total_diff / num_chemicals
            
            # Scale to impact score (0.0-1.0)
            impact = min(1.0, avg_diff * 2.5)  # Scale factor to make changes more noticeable
            
            return impact
        else:
            return 0.0

    @staticmethod
    @function_tool
    async def _check_confidence_threshold(ctx: RunContextWrapper[CASystemContext], 
                                     distribution: ContextDistribution,
                                     confidence: float) -> ConfidenceThresholdResult:
        """
        Check if confidence meets threshold for significant contexts
        
        Args:
            distribution: Context distribution
            confidence: Overall detection confidence
            
        Returns:
            Threshold check results
        """
        # Get active contexts and weights
        active_contexts = distribution.active_contexts
        
        cas = ctx.context.context_awareness_system
        
        # Check if any context exceeds the threshold
        significant_contexts = [
            SignificantContext(context=context, weight=weight)
            for context, weight in active_contexts 
            if weight >= cas.significant_context_threshold
        ]
        
        # Check confidence threshold
        confidence_threshold = 0.5  # Base confidence threshold
        threshold_met = confidence >= confidence_threshold
        
        return ConfidenceThresholdResult(
            threshold_met=threshold_met,
            confidence_threshold=confidence_threshold,
            confidence=confidence,
            significant_contexts=significant_contexts,
            has_significant_contexts=len(significant_contexts) > 0
        )

    @staticmethod
    @function_tool
    async def _detect_context_signals(ctx: RunContextWrapper[CASystemContext], 
                                 message: str) -> ContextDetectionResult:
        """
        Detect context signals and calculate initial context distribution
        
        Args:
            message: User message to analyze
            
        Returns:
            Detection results with context distribution and confidence
        """
        # Initialize context scores
        context_scores = {
            "dominant": 0.0,
            "casual": 0.0,
            "intellectual": 0.0,
            "empathic": 0.0,
            "playful": 0.0,
            "creative": 0.0,
            "professional": 0.0
        }
        detected_signals = []
        
        # Convert message to lowercase for case-insensitive matching
        message_lower = message.lower()
        
        cas = ctx.context.context_awareness_system
        
        # Check for explicit context signals
        for signal in cas.context_signals:
            context_type = signal.context_type.value.lower()
            
            if signal.signal_type == "keyword" and signal.signal_value.lower() in message_lower:
                context_scores[context_type] += signal.strength
                detected_signals.append(SignalInfo(
                    type=signal.signal_type,
                    value=signal.signal_value,
                    context=context_type,
                    strength=signal.strength
                ))
            elif signal.signal_type == "phrase" and signal.signal_value.lower() in message_lower:
                context_scores[context_type] += signal.strength * 1.2  # Phrases are stronger signals
                detected_signals.append(SignalInfo(
                    type=signal.signal_type,
                    value=signal.signal_value,
                    context=context_type,
                    strength=signal.strength * 1.2
                ))
            elif signal.signal_type == "greeting" and message_lower.startswith(signal.signal_value.lower()):
                context_scores[context_type] += signal.strength * 0.8  # Greetings are moderate signals
                detected_signals.append(SignalInfo(
                    type=signal.signal_type,
                    value=signal.signal_value,
                    context=context_type,
                    strength=signal.strength * 0.8
                ))
        
        # Check for implicit signals
        implicit_signals = await ContextAwarenessSystem._identify_implicit_signals(ctx, message)
        detected_signals.extend(implicit_signals)
        
        # Add implicit signal scores
        for signal in implicit_signals:
            context_type = signal.context.lower()
            if context_type in context_scores:
                context_scores[context_type] += signal.strength
                
        # Create context distribution
        distribution = ContextDistribution(**context_scores)
        
        # Calculate confidence based on signal strength and differentiation
        total_signal_strength = sum(context_scores.values())
        max_context, max_score = distribution.primary_context
        
        if total_signal_strength > 0:
            # Calculate variance in scores as a measure of differentiation
            avg_score = total_signal_strength / len(context_scores)
            variance = sum((score - avg_score) ** 2 for score in context_scores.values()) / len(context_scores)
            
            # Higher variance = more differentiated = higher confidence
            variance_factor = min(1.0, variance * 5)  # Scale factor
            
            # Overall confidence based on total strength and differentiation
            confidence = (total_signal_strength * 0.3) + (max_score * 0.4) + (variance_factor * 0.3)
            confidence = min(1.0, confidence)
        else:
            confidence = 0.2  # Default low confidence if no signals
        
        # Normalize distribution if any signals detected
        if distribution.sum_weights() > 0:
            distribution = distribution.normalize()
            
        return ContextDetectionResult(
            context_distribution=distribution,
            confidence=confidence,
            signals=detected_signals,
            signal_based=True,
            detection_method="composite_analysis"
        )

    @staticmethod
    @function_tool
    async def _extract_message_features(ctx: RunContextWrapper[CASystemContext], 
                                   message: str) -> MessageFeatures:
        """
        Extract features from a message for context analysis
        
        Args:
            message: User message to analyze
            
        Returns:
            Extracted message features
        """
        # Basic features
        length = len(message)
        word_count = len(message.split())
        has_question = "?" in message
        capitalization_ratio = sum(1 for c in message if c.isupper()) / max(1, len(message))
        punctuation_count = sum(1 for c in message if c in ".,;:!?-")
        
        # Check for dominance-related terms
        dominance_terms_list = ["mistress", "domme", "slave", "obey", "submit", "kneel", "worship", "serve"]
        dominance_terms = [term for term in dominance_terms_list if term in message.lower()]
        has_dominance_terms = len(dominance_terms) > 0
        
        # Check for emotional terms
        emotional_terms_list = ["feel", "sad", "happy", "angry", "worried", "excited", "afraid", "love", "hate"]
        emotional_terms = [term for term in emotional_terms_list if term in message.lower()]
        has_emotional_terms = len(emotional_terms) > 0
        
        # Derive higher-level features
        likely_formal = capitalization_ratio > 0.2 and punctuation_count >= 2
        likely_casual = capitalization_ratio < 0.1 and "hi" in message.lower()
        likely_emotional = has_emotional_terms and "?" not in message
        likely_intellectual = word_count > 15 and has_question
        
        # Derive context distribution from features
        feature_distribution = FeatureDistribution(
            dominant=0.7 * (len(dominance_terms) / 2) if has_dominance_terms else 0.0,
            casual=0.6 if likely_casual else 0.0,
            intellectual=0.7 if likely_intellectual else 0.0,
            empathic=0.7 * (len(emotional_terms) / 2) if likely_emotional else 0.0,
            playful=0.0,
            creative=0.0,
            professional=0.6 if likely_formal else 0.0
        )
        
        return MessageFeatures(
            length=length,
            word_count=word_count,
            has_question=has_question,
            capitalization_ratio=capitalization_ratio,
            punctuation_count=punctuation_count,
            dominance_terms=dominance_terms,
            emotional_terms=emotional_terms,
            has_dominance_terms=has_dominance_terms,
            has_emotional_terms=has_emotional_terms,
            likely_formal=likely_formal,
            likely_casual=likely_casual,
            likely_emotional=likely_emotional,
            likely_intellectual=likely_intellectual,
            feature_distribution=feature_distribution
        )

    @staticmethod
    @function_tool
    async def _get_context_history(ctx: RunContextWrapper[CASystemContext]) -> List[HistoryEntry]:
        """
        Get recent context history
        
        Returns:
            Recent context history
        """
        cas = ctx.context.context_awareness_system
        
        # Return last 5 context history items or fewer if not available
        history = cas.context_history[-5:] if cas.context_history else []
        
        # Convert to HistoryEntry objects
        history_entries = []
        for item in history:
            detected_dist = item.get("detected_distribution", {})
            updated_dist = item.get("updated_distribution", {})
            
            history_entries.append(HistoryEntry(
                timestamp=item.get("timestamp", ""),
                message_snippet=item.get("message_snippet", ""),
                detected_distribution=ContextDistributionDict(**detected_dist),
                updated_distribution=ContextDistributionDict(**updated_dist),
                confidence=item.get("confidence", 0.0),
                primary_context_changed=item.get("primary_context_changed", False),
                active_contexts=item.get("active_contexts", [])
            ))
        
        return history_entries

    @staticmethod
    @function_tool
    async def _calculate_context_confidence(ctx: RunContextWrapper[CASystemContext],
                                      distribution: ContextDistribution,
                                      signals: List[SignalInfo],
                                      message_features: MessageFeatures) -> float:
        """
        Calculate confidence in detected context distribution
        
        Args:
            distribution: Detected context distribution
            signals: Context signals detected
            message_features: Features extracted from the message
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from signal count
        signal_confidence = min(1.0, len(signals) * 0.1)
        
        # Calculate average signal strength
        if signals:
            total_strength = sum(s.strength for s in signals)
            avg_strength = total_strength / len(signals)
            strength_factor = avg_strength
        else:
            strength_factor = 0.0
            
        # Calculate coherence of distribution
        primary_context, primary_weight = distribution.primary_context
        active_contexts = distribution.active_contexts
        
        # Higher confidence if distribution is focused
        if primary_weight > 0.6:
            focus_factor = 0.3
        elif primary_weight > 0.4:
            focus_factor = 0.2
        else:
            focus_factor = 0.1
            
        # Match with message features
        feature_confidence = 0.0
        feature_dist_dict = message_features.feature_distribution.dict()
        
        if feature_dist_dict:
            # Calculate correlation between detected distribution and feature distribution
            correlation = sum(min(distribution.dict().get(context, 0.0), feature_dist_dict.get(context, 0.0)) 
                             for context in distribution.dict().keys())
            
            feature_confidence = correlation
            
        # Calculate overall confidence
        confidence = (signal_confidence * 0.3) + (strength_factor * 0.3) + (focus_factor * 0.2) + (feature_confidence * 0.2)
        
        # Ensure valid range
        return max(0.1, min(1.0, confidence))

    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message to determine and update context distribution
        
        Args:
            message: User message to process
            
        Returns:
            Updated context information
        """
        async with self._lock:
            with trace(workflow_name="ContextDetection", group_id=self.system_context.trace_id):
                # Run the context detection agent
                result = await Runner.run(
                    self.context_detection_agent,
                    {"message": message},
                    context=self.system_context,
                    run_config={
                        "workflow_name": "ContextDetection",
                        "trace_metadata": {"input_length": len(message)}
                    }
                )
                
                # Process the detection result
                detection_result = result.final_output
                
                # Extract the context distribution and confidence
                detected_distribution = detection_result.context_distribution
                confidence = detection_result.confidence
                
                # Save current distribution as previous
                self.previous_distribution = self.context_distribution
                
                # Update context distribution with blending
                if self.context_distribution.sum_weights() < 0.1:
                    # If current distribution is essentially empty, use detected distribution directly
                    self.context_distribution = detected_distribution
                else:
                    # Blend detected distribution with current distribution
                    self.context_distribution = self.context_distribution.blend_with(
                        detected_distribution, 
                        self.context_blend_factor
                    )
                
                # Update overall confidence
                self.overall_confidence = confidence
                
                # Update legacy single-context fields (for backwards compatibility)
                primary_context, primary_confidence = self.context_distribution.to_enum_and_confidence()
                self.current_context = primary_context
                self.context_confidence = primary_confidence
                primary_context_prev, _ = self.previous_distribution.to_enum_and_confidence() if self.previous_distribution else (InteractionContext.UNDEFINED, 0.0)
                self.previous_context = primary_context_prev
                
                # Determine if significant context change occurred
                context_changed = False
                if self.previous_distribution:
                    primary_before, _ = self.previous_distribution.primary_context
                    primary_after, _ = self.context_distribution.primary_context
                    context_changed = primary_before != primary_after
                
                # Record in history
                history_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message_snippet": message[:50] + ("..." if len(message) > 50 else ""),
                    "detected_distribution": detected_distribution.dict(),
                    "updated_distribution": self.context_distribution.dict(),
                    "confidence": confidence,
                    "primary_context_changed": context_changed,
                    "active_contexts": [c for c, w in self.context_distribution.active_contexts]
                }
                
                self.context_history.append(history_entry)
                
                # Limit history size
                if len(self.context_history) > 100:
                    self.context_history = self.context_history[-100:]
                
                # Apply context effects if significant change in distribution
                effects = {}
                if context_changed or self.previous_distribution is None:
                    effects = await self._apply_context_effects()
                
                return {
                    "context_distribution": self.context_distribution.dict(),
                    "primary_context": primary_context.value,
                    "primary_confidence": primary_confidence,
                    "overall_confidence": self.overall_confidence,
                    "active_contexts": [c for c, w in self.context_distribution.active_contexts],
                    "context_changed": context_changed,
                    "detection_method": "blended_context",
                    "detected_signals": [signal.dict() for signal in detection_result.signals],
                    "notes": detection_result.notes,
                    "effects": effects
                }
    
    async def _apply_context_effects(self) -> Dict[str, Any]:
        """Apply effects when context changes"""
        effects = {"emotional": False}
        
        # Update emotional baselines if emotional core is available
        if self.system_context.emotional_core and self.context_distribution.sum_weights() > 0.3:
            try:
                with trace(workflow_name="EmotionalBaselines", group_id=self.system_context.trace_id):
                    # Calculate blended emotional baselines
                    blended_baselines = await self._blend_emotional_baselines(
                        RunContextWrapper(context=self.system_context),
                        self.context_distribution
                    )
                    
                    # Apply baseline adjustments
                    blended_dict = blended_baselines.dict()
                    for chemical, baseline in blended_dict.items():
                        # Only adjust if in emotional core
                        if chemical in self.system_context.emotional_core.neurochemicals:
                            # Create temporary baseline (not permanent changes)
                            self.system_context.emotional_core.neurochemicals[chemical]["temporary_baseline"] = baseline
                    
                    effects["emotional"] = True
                    effects["baselines"] = blended_dict
                    effects["active_contexts"] = [c for c, w in self.context_distribution.active_contexts]
                    
                    # Calculate emotional impact
                    if self.previous_distribution:
                        previous_baselines = await self._blend_emotional_baselines(
                            RunContextWrapper(context=self.system_context),
                            self.previous_distribution
                        )
                        
                        impact = await self._calculate_emotional_impact(
                            RunContextWrapper(context=self.system_context),
                            previous_baselines,
                            blended_baselines
                        )
                        
                        effects["impact"] = impact
                    
                    logger.info(f"Applied blended emotional baselines for context distribution")
            except Exception as e:
                logger.error(f"Error applying context emotional effects: {e}")
        
        return effects
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get the current interaction context (blended version)"""
        return {
            "context_distribution": self.context_distribution.dict(),
            "primary_context": self.current_context.value,  # Legacy field
            "primary_confidence": self.context_confidence,  # Legacy field
            "active_contexts": [c for c, w in self.context_distribution.active_contexts],
            "overall_confidence": self.overall_confidence,
            "history": self.context_history[-5:] if self.context_history else []
        }
    
    def get_system_state(self) -> ContextSystemState:
        """Get the current system state"""
        # Convert emotional baselines to the new structure
        emotional_baselines_mapping = EmotionalBaselinesMapping()
        for context_enum, baselines_dict in self.context_emotional_baselines.items():
            context_name = context_enum.value.lower()
            if hasattr(emotional_baselines_mapping, context_name):
                setattr(emotional_baselines_mapping, context_name, EmotionalBaselines(**baselines_dict))
        
        # Convert history to HistoryEntry objects
        history_entries = []
        for item in (self.context_history[-5:] if self.context_history else []):
            detected_dist = item.get("detected_distribution", {})
            updated_dist = item.get("updated_distribution", {})
            
            history_entries.append(HistoryEntry(
                timestamp=item.get("timestamp", ""),
                message_snippet=item.get("message_snippet", ""),
                detected_distribution=ContextDistributionDict(**detected_dist),
                updated_distribution=ContextDistributionDict(**updated_dist),
                confidence=item.get("confidence", 0.0),
                primary_context_changed=item.get("primary_context_changed", False),
                active_contexts=item.get("active_contexts", [])
            ))
        
        return ContextSystemState(
            context_distribution=self.context_distribution,
            overall_confidence=self.overall_confidence,
            previous_distribution=self.previous_distribution,
            history=history_entries,
            emotional_baselines=emotional_baselines_mapping
        )
    
    def add_context_signal(self, signal: ContextSignal) -> bool:
        """Add a new context signal to the database"""
        try:
            self.context_signals.append(signal)
            logger.info(f"Added new context signal: {signal.signal_type}:{signal.signal_value} -> {signal.context_type}")
            return True
        except Exception as e:
            logger.error(f"Error adding context signal: {e}")
            return False
