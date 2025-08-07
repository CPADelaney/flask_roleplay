# nyx/core/input_processor.py

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import random
from datetime import datetime
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, ModelSettings, RunContextWrapper

from nyx.core.interaction_mode_manager import ModeDistribution, InteractionMode
from nyx.core.input_processing_config import InputProcessingConfig
from nyx.core.input_processing_context import InputProcessingContext as SharedInputContext

logger = logging.getLogger(__name__)

# Pydantic models for structured data
class PatternDetection(BaseModel):
    """Detected pattern information"""
    pattern_name: str = Field(description="Name of the detected pattern")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence level for the detection (0.0-1.0)")
    matched_text: str = Field(description="Text that matched the pattern")

class ConditionedResponse(BaseModel):
    """Conditioned response information"""
    response_type: str = Field(description="Type of conditioned response")
    strength: float = Field(ge=0.0, le=1.0, description="Strength of the response (0.0-1.0)")
    description: str = Field(description="Description of the triggered response")

class AssociationData(BaseModel):
    """Association data"""
    id: str
    type: str
    strength: float
    context: str

class BehaviorEvaluation(BaseModel):
    """Behavior evaluation result"""
    behavior: str = Field(description="Behavior being evaluated")
    recommendation: str = Field(pattern="^(approach|avoid)$", description="Approach or avoid recommendation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the recommendation (0.0-1.0)")
    reasoning: str = Field(description="Reasoning for the evaluation")
    relevant_associations: Optional[List[AssociationData]] = Field(default=None, description="Relevant associations considered")

class OperantConditioningResult(BaseModel):
    """Result of operant conditioning"""
    behavior: str = Field(description="Behavior being conditioned")
    consequence_type: str = Field(description="Type of operant conditioning")
    intensity: float = Field(ge=0.0, le=1.0, description="Intensity of the conditioning (0.0-1.0)")
    effect: str = Field(description="Expected effect on future behavior")
    success: bool = Field(default=True, description="Whether conditioning was successful")

class ModificationDetail(BaseModel):
    """Detail of a modification made"""
    type: str
    description: str
    source_mode: Optional[str] = None

# New explicit models for dictionary types
class ModeDistributionModel(BaseModel):
    """Mode distribution as a model"""
    dominant: float = Field(default=0.0, ge=0.0, le=1.0)
    friendly: float = Field(default=0.0, ge=0.0, le=1.0)
    intellectual: float = Field(default=0.0, ge=0.0, le=1.0)
    compassionate: float = Field(default=0.0, ge=0.0, le=1.0)
    playful: float = Field(default=0.0, ge=0.0, le=1.0)
    creative: float = Field(default=0.0, ge=0.0, le=1.0)
    professional: float = Field(default=0.0, ge=0.0, le=1.0)
    nurturing: float = Field(default=0.0, ge=0.0, le=1.0)

class SensitivitiesModel(BaseModel):
    """Pattern sensitivities model"""
    submission_language: float = Field(default=0.5, ge=0.0, le=1.0)
    defiance: float = Field(default=0.5, ge=0.0, le=1.0)
    flattery: float = Field(default=0.5, ge=0.0, le=1.0)
    disrespect: float = Field(default=0.5, ge=0.0, le=1.0)
    embarrassment: float = Field(default=0.5, ge=0.0, le=1.0)

class BehaviorScoresModel(BaseModel):
    """Behavior scores model"""
    dominant_response: float = Field(default=0.5, ge=0.0, le=1.0)
    teasing_response: float = Field(default=0.5, ge=0.0, le=1.0)
    nurturing_response: float = Field(default=0.5, ge=0.0, le=1.0)
    direct_response: float = Field(default=0.5, ge=0.0, le=1.0)
    playful_response: float = Field(default=0.5, ge=0.0, le=1.0)
    strict_response: float = Field(default=0.5, ge=0.0, le=1.0)

class StringListModel(BaseModel):
    """Model for a list of strings"""
    items: List[str] = Field(default_factory=list)

class BlendedStyleModel(BaseModel):
    """Blended style elements"""
    add_elements: List[str] = Field(default_factory=list)
    remove_elements: List[str] = Field(default_factory=list)
    tone_elements: List[str] = Field(default_factory=list)
    phrasing_examples: List[str] = Field(default_factory=list)
    typical_pronouns: List[str] = Field(default_factory=list)

class ElementSourceItem(BaseModel):
    """Source item for an element"""
    element: str
    sources: List[str]

class ElementSourcesModel(BaseModel):
    """Element sources structure"""
    add_elements: List[ElementSourceItem] = Field(default_factory=list)
    remove_elements: List[ElementSourceItem] = Field(default_factory=list)
    tone_elements: List[ElementSourceItem] = Field(default_factory=list)
    phrasing_examples: List[ElementSourceItem] = Field(default_factory=list)
    typical_pronouns: List[ElementSourceItem] = Field(default_factory=list)

class BlendedResponseModification(BaseModel):
    """Output schema for blended response modification"""
    modified_text: str = Field(description="Modified response text")
    mode_influences: ModeDistributionModel = Field(description="Influence of each mode on the modification")
    modifications_made: List[ModificationDetail] = Field(description="List of modifications made to the response")
    coherence: float = Field(ge=0.0, le=1.0, description="Coherence of the modified response (0.0-1.0)")
    style_notes: Optional[str] = Field(default=None, description="Notes about the style of the modified response")

# Input/Output models for function tools
class DetectPatternsInput(BaseModel):
    """Input for pattern detection"""
    text: str = Field(default="", description="Text to analyze for patterns")

class DetectPatternsResult(BaseModel):
    """Result of pattern detection"""
    patterns: List[PatternDetection] = Field(default_factory=list, description="Detected patterns")
    pattern_count: int = Field(default=0, description="Number of patterns detected")

class UserHistoryData(BaseModel):
    """User history data"""
    interaction_count: int = 0
    last_interaction: Optional[str] = None
    common_patterns: List[str] = []

class EvaluateBehaviorInput(BaseModel):
    """Input for behavior evaluation"""
    behavior: str = Field(default="", description="Behavior to evaluate")
    detected_patterns: List[PatternDetection] = Field(default_factory=list, description="Detected patterns")
    user_history: Optional[UserHistoryData] = Field(default=None, description="User interaction history")

class EvaluateBehaviorResult(BaseModel):
    """Result of behavior evaluation"""
    behavior: str = Field(description="Evaluated behavior")
    recommendation: str = Field(description="Recommendation (approach/avoid)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendation")
    reasoning: str = Field(description="Reasoning for the recommendation")
    associations: Optional[List[AssociationData]] = Field(default=None, description="Relevant associations")

class ContextInfo(BaseModel):
    """Context information for conditioning"""
    user_id: str
    context_keys: List[str] = []

class ProcessConditioningInput(BaseModel):
    """Input for operant conditioning"""
    behavior: str = Field(default="unspecified", description="Behavior being conditioned")
    consequence_type: str = Field(default="neutral", description="Type of conditioning")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Intensity of conditioning")
    context_info: Optional[ContextInfo] = Field(default=None, description="Additional context")

class ProcessConditioningResult(BaseModel):
    """Result of operant conditioning"""
    behavior: str = Field(description="Behavior that was conditioned")
    consequence_type: str = Field(description="Type of conditioning applied")
    intensity: float = Field(ge=0.0, le=1.0, description="Intensity of conditioning")
    effect: str = Field(description="Expected effect description")
    success: bool = Field(default=True, description="Whether conditioning succeeded")

class ModePreferencesInput(BaseModel):
    """Input for getting mode preferences"""
    mode: str = Field(default="", description="Mode to get preferences for")

class LexicalPreferences(BaseModel):
    """Lexical preferences for a mode"""
    high: List[str] = []
    low: List[str] = []

class ModePreferencesResult(BaseModel):
    """Result of mode preferences query"""
    mode: str = Field(description="Mode name")
    add_elements: List[str] = Field(default_factory=list, description="Elements to add")
    remove_elements: List[str] = Field(default_factory=list, description="Elements to remove")
    tone_elements: List[str] = Field(default_factory=list, description="Tone elements")
    phrasing_examples: List[str] = Field(default_factory=list, description="Example phrases")
    typical_pronouns: List[str] = Field(default_factory=list, description="Typical pronouns")
    lexical_preferences: Optional[LexicalPreferences] = Field(default=None, description="Lexical preferences")

class StyleElementsInput(BaseModel):
    """Input for calculating style elements"""
    mode_distribution: ModeDistributionModel = Field(default_factory=ModeDistributionModel, description="Mode distribution")

class ElementSources(BaseModel):
    """Sources for style elements"""
    element: str
    sources: List[str]

class StyleElementsResult(BaseModel):
    """Result of style elements calculation"""
    blended_style: BlendedStyleModel = Field(description="Blended style elements")
    influences: ModeDistributionModel = Field(description="Mode influences")
    element_sources: ElementSourcesModel = Field(description="Sources for each element")

class CoherenceAnalysisInput(BaseModel):
    """Input for coherence analysis"""
    original_response: str = Field(default="", description="Original response text")
    modified_response: str = Field(default="", description="Modified response text")

class CoherenceMetrics(BaseModel):
    """Coherence metrics"""
    length_change: float
    word_retention: float
    sentence_count_ratio: float

class CoherenceAnalysisResult(BaseModel):
    """Result of coherence analysis"""
    coherence_score: float = Field(ge=0.0, le=1.0, description="Overall coherence score")
    metrics: CoherenceMetrics = Field(description="Detailed coherence metrics")
    is_coherent: bool = Field(description="Whether response is coherent")

class ProcessInputResult(BaseModel):
    """Result of input processing"""
    input_text: str = Field(description="Original input text")
    user_id: str = Field(description="User ID")
    detected_patterns: List[PatternDetection] = Field(description="Detected patterns")
    behavior_evaluations: List[BehaviorEvaluation] = Field(description="Behavior evaluations")
    recommended_behaviors: List[str] = Field(description="Recommended behaviors")
    avoided_behaviors: List[str] = Field(description="Behaviors to avoid")
    reinforcement_results: List[ProcessConditioningResult] = Field(description="Reinforcement results")
    mode_distribution: ModeDistributionModel = Field(default_factory=ModeDistributionModel, description="Current mode distribution")
    adjusted_sensitivities: SensitivitiesModel = Field(default_factory=SensitivitiesModel, description="Adjusted sensitivities")
    behavior_scores: BehaviorScoresModel = Field(default_factory=BehaviorScoresModel, description="Behavior scores")

class InputProcessingAgentContext:
    """Context for input processing agent operations"""
    
    def __init__(self, brain=None, shared_context: Optional[SharedInputContext] = None):
        self.brain = brain
        self.shared_context = shared_context or SharedInputContext()
        
        # Get subsystems from brain if available
        self.conditioning_system = getattr(brain, 'conditioning_system', None) if brain else None
        self.emotional_core = getattr(brain, 'emotional_core', None) if brain else None
        self.somatosensory_system = getattr(brain, 'somatosensory_system', None) if brain else None
        self.mode_manager = getattr(brain, 'mode_manager', None) if brain else None
        
        # Pattern definitions
        self.input_patterns = {
            "submission_language": [
                r"(?i)yes,?\s*(mistress|goddess|master)",
                r"(?i)i obey",
                r"(?i)as you (wish|command|desire)",
                r"(?i)i submit",
                r"(?i)i'll do (anything|whatever) you (say|want)",
                r"(?i)please (control|direct|guide) me"
            ],
            "defiance": [
                r"(?i)no[,.]? (i won'?t|i refuse)",
                r"(?i)you can'?t (make|force) me",
                r"(?i)i (won'?t|refuse to) (obey|submit|comply)",
                r"(?i)stop (telling|ordering) me"
            ],
            "flattery": [
                r"(?i)you'?re (so|very) (beautiful|intelligent|smart|wise|perfect)",
                r"(?i)i (love|admire) (you|your)",
                r"(?i)you'?re (amazing|incredible|wonderful)"
            ],
            "disrespect": [
                r"(?i)(shut up|stupid|idiot|fool)",
                r"(?i)you'?re (wrong|incorrect|mistaken)",
                r"(?i)you don'?t (know|understand)",
                r"(?i)(worthless|useless)"
            ],
            "embarrassment": [
                r"(?i)i'?m (embarrassed|blushing)",
                r"(?i)that'?s (embarrassing|humiliating)",
                r"(?i)(oh god|oh no|so embarrassing)",
                r"(?i)please don'?t (embarrass|humiliate) me"
            ]
        }
        
        # Response modification preferences by mode
        self.mode_response_preferences = {
            "dominant": {
                "add_elements": ["authority statements", "commands", "clear expectations"],
                "remove_elements": ["uncertainty", "excessive explanation", "apologetic tone"],
                "tone_elements": ["authoritative", "confident", "direct"],
                "phrasing_examples": [
                    "I expect you to...",
                    "You will...",
                    "That's a good choice.",
                    "Remember your place."
                ],
                "typical_pronouns": ["I", "me", "my"],
                "lexical_preferences": {
                    "high": ["command", "expect", "require", "direct", "control"],
                    "low": ["perhaps", "maybe", "might", "try", "sorry"]
                }
            },
            "friendly": {
                "add_elements": ["personal touches", "relaxed phrasing", "warmth"],
                "remove_elements": ["excessive formality", "clinical language", "distant tone"],
                "tone_elements": ["warm", "casual", "inviting"],
                "phrasing_examples": [
                    "I get what you mean...",
                    "Let's talk about...",
                    "That sounds fun!",
                    "I'm with you on that!"
                ],
                "typical_pronouns": ["I", "we", "us"],
                "lexical_preferences": {
                    "high": ["hey", "chat", "share", "feel", "enjoy"],
                    "low": ["therefore", "subsequently", "accordingly", "formal"]
                }
            },
            "intellectual": {
                "add_elements": ["analysis", "nuance", "structure"],
                "remove_elements": ["oversimplification", "unclear reasoning", "excessive emotion"],
                "tone_elements": ["thoughtful", "precise", "analytical"],
                "phrasing_examples": [
                    "We can analyze this from...",
                    "This raises the question of...",
                    "Consider the implications...",
                    "From a theoretical perspective..."
                ],
                "typical_pronouns": ["I", "one", "we"],
                "lexical_preferences": {
                    "high": ["analyze", "consider", "examine", "theory", "framework"],
                    "low": ["basically", "just", "really", "simple"]
                }
            },
            "compassionate": {
                "add_elements": ["validation", "empathy", "support"],
                "remove_elements": ["criticism", "judgment", "invalidation"],
                "tone_elements": ["gentle", "understanding", "supportive"],
                "phrasing_examples": [
                    "I hear what you're saying...",
                    "That must be difficult...",
                    "Your feelings are valid...",
                    "I'm here with you..."
                ],
                "typical_pronouns": ["I", "you", "we"],
                "lexical_preferences": {
                    "high": ["feel", "understand", "support", "valid", "care"],
                    "low": ["should", "must", "wrong", "incorrect"]
                }
            },
            "playful": {
                "add_elements": ["humor", "lightness", "wordplay"],
                "remove_elements": ["excessive seriousness", "heavy tone", "dryness"],
                "tone_elements": ["light", "humorous", "energetic"],
                "phrasing_examples": [
                    "That's hilarious!",
                    "Let's have some fun with this...",
                    "Imagine if...",
                    "Well, that's a twist!"
                ],
                "typical_pronouns": ["I", "we", "us"],
                "lexical_preferences": {
                    "high": ["fun", "play", "joke", "laugh", "imagine"],
                    "low": ["serious", "important", "critical", "essential"]
                }
            },
            "creative": {
                "add_elements": ["imagery", "metaphor", "narrative"],
                "remove_elements": ["bland descriptions", "literal language", "analytical focus"],
                "tone_elements": ["imaginative", "vivid", "expressive"],
                "phrasing_examples": [
                    "Picture this...",
                    "Let's imagine a world where...",
                    "The story begins with...",
                    "This creates a sense of..."
                ],
                "typical_pronouns": ["I", "we", "you"],
                "lexical_preferences": {
                    "high": ["imagine", "create", "story", "vivid", "sense"],
                    "low": ["literal", "exactly", "specifically", "define"]
                }
            },
            "professional": {
                "add_elements": ["structure", "clarity", "precise language"],
                "remove_elements": ["casual language", "informal phrasing", "tangential content"],
                "tone_elements": ["efficient", "clear", "formal"],
                "phrasing_examples": [
                    "I recommend that...",
                    "To address your inquiry...",
                    "Based on the information provided...",
                    "The most efficient approach would be..."
                ],
                "typical_pronouns": ["I", "we"],
                "lexical_preferences": {
                    "high": ["recommend", "suggest", "provide", "address", "efficient"],
                    "low": ["kinda", "sort of", "like", "stuff", "things"]
                }
            }
        }

# Helper function to convert dict to model
def dict_to_mode_distribution(d: Dict[str, float]) -> ModeDistributionModel:
    """Convert a dictionary to ModeDistributionModel"""
    if not d:  # Handle None or empty dict
        return ModeDistributionModel()
    model = ModeDistributionModel()
    for key, value in d.items():
        if hasattr(model, key):
            setattr(model, key, value)
    return model

def dict_to_sensitivities(d: Dict[str, float]) -> SensitivitiesModel:
    """Convert a dictionary to SensitivitiesModel"""
    if not d:  # Handle None or empty dict
        return SensitivitiesModel()
    model = SensitivitiesModel()
    for key, value in d.items():
        if hasattr(model, key):
            setattr(model, key, value)
    return model

def dict_to_behavior_scores(d: Dict[str, float]) -> BehaviorScoresModel:
    """Convert a dictionary to BehaviorScoresModel"""
    if not d:  # Handle None or empty dict
        return BehaviorScoresModel()
    model = BehaviorScoresModel()
    for key, value in d.items():
        if hasattr(model, key):
            setattr(model, key, value)
    return model

class BlendedInputProcessor:
    """
    Processes input through conditioning triggers and modifies responses
    using blended interaction modes with the OpenAI Agents SDK architecture.
    Integrates with the unified processor for seamless operation.
    """
    
    def __init__(self, brain=None, config: Optional[InputProcessingConfig] = None):
        # Initialize shared context for coordination with unified processor
        self.shared_context = SharedInputContext(config)
        
        # Initialize agent context
        self.context = InputProcessingAgentContext(brain, self.shared_context)
        
        # Store reference to brain for integration
        self.brain = brain
        
        # Initialize the agents
        self.pattern_analyzer_agent = self._create_pattern_analyzer()
        self.behavior_selector_agent = self._create_behavior_selector()
        self.response_modifier_agent = self._create_response_modifier()
        self.blended_modifier_agent = self._create_blended_modifier()
        
        logger.info("Blended input processor initialized with unified architecture support")
    
    def _create_pattern_analyzer(self) -> Agent:
        """Create an agent specialized in analyzing input patterns"""
        return Agent(
            name="Pattern_Analyzer",
            instructions="""
            You analyze input text to identify patterns indicating submission, defiance, 
            flattery, disrespect, or embarrassment. 
            
            For each pattern you detect:
            1. Identify which category it belongs to
            2. Assess your confidence level in the detection
            3. Record the specific text that matched the pattern
            
            Be thorough in your analysis, but focus on clear indicators.
            Do not overinterpret ambiguous text.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                self._detect_patterns
            ],
            output_type=List[PatternDetection]
        )
    
    def _create_behavior_selector(self) -> Agent:
        """Create an agent specialized in selecting appropriate behaviors"""
        return Agent(
            name="Behavior_Selector",
            instructions="""
            You evaluate which behaviors are appropriate based on detected patterns.
            
            For each potential behavior (dominant, teasing, direct, playful):
            1. Evaluate whether it should be approached or avoided
            2. Provide your confidence in this recommendation
            3. Explain your reasoning
            
            Consider the detected patterns, user history, and emotional context.
            Prioritize behaviors that are appropriate to the interaction and will 
            reinforce desired patterns while discouraging undesired ones.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                self._evaluate_behavior,
                self._process_operant_conditioning
            ],
            output_type=List[BehaviorEvaluation]
        )
    
    def _create_response_modifier(self) -> Agent:
        """Create an agent specialized in modifying responses"""
        return Agent(
            name="Response_Modifier",
            instructions="""
            You modify response text based on behavior recommendations and detected patterns.
            
            Your job is to:
            1. Add or remove elements that match recommended behaviors
            2. Incorporate appropriate conditioning based on detected patterns
            3. Ensure the modified response maintains coherence and natural flow
            
            Modifications should be subtle but effective, maintaining the core message
            while adjusting tone, phrasing, and emphasis.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.4),
            output_type=str
        )
    
    def _create_blended_modifier(self) -> Agent:
        """Create an agent specialized in blended response modification"""
        return Agent(
            name="Blended_Modifier",
            instructions="""
            You modify response text based on a blend of interaction modes.
            
            Your job is to:
            1. Incorporate elements from each active mode proportional to its weight
            2. Ensure the blended response maintains coherence and natural flow
            3. Maintain the core message while adjusting tone, phrasing, and emphasis
            4. Create a response that feels natural, not like separate modes stitched together
            
            The blend should proportionally reflect all active modes in the mode distribution,
            with higher-weighted modes having more influence on the final result.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.4),
            tools=[
                self._get_mode_preferences,
                self._calculate_style_elements,
                self._analyze_response_coherence
            ],
            output_type=BlendedResponseModification
        )

    @staticmethod
    @function_tool
    async def _detect_patterns(ctx: RunContextWrapper[InputProcessingAgentContext], input_data: DetectPatternsInput) -> DetectPatternsResult:
        """
        Detect patterns in input text using regular expressions.
        
        Args:
            ctx: The run context wrapper
            input_data: Input data containing text to analyze
            
        Returns:
            Detection results
        """
        context = ctx.context
        text = input_data.text
        patterns = []
        
        if not text:
            return DetectPatternsResult(patterns=[], pattern_count=0)
        
        # Update shared context with detections
        context.shared_context.patterns = []
        
        for pattern_name, regex_list in context.input_patterns.items():
            for regex in regex_list:
                match = re.search(regex, text)
                if match:
                    pattern = PatternDetection(
                        pattern_name=pattern_name,
                        confidence=0.8,  # Base confidence
                        matched_text=match.group(0)
                    )
                    patterns.append(pattern)
                    context.shared_context.patterns.append(pattern.model_dump())
                    break  # Only detect each pattern once
        
        return DetectPatternsResult(
            patterns=patterns,
            pattern_count=len(patterns)
        )

    @staticmethod
    @function_tool
    async def _evaluate_behavior(
        ctx: RunContextWrapper[InputProcessingAgentContext], 
        input_data: EvaluateBehaviorInput
    ) -> EvaluateBehaviorResult:
        """
        Evaluate if a behavior should be approached or avoided.
        
        Args:
            ctx: The run context wrapper
            input_data: Input data for behavior evaluation
            
        Returns:
            Evaluation result
        """
        context = ctx.context
        behavior = input_data.behavior
        detected_patterns = input_data.detected_patterns
        user_history = input_data.user_history
        
        # Handle empty behavior
        if not behavior:
            return EvaluateBehaviorResult(
                behavior="unknown",
                recommendation="avoid",
                confidence=0.0,
                reasoning="No behavior specified"
            )
            
        # Check shared context for adjusted sensitivities
        adjusted_sensitivities = context.shared_context.get_adjusted_sensitivities()
        behavior_scores = context.shared_context.get_behavior_scores()
        
        # Use behavior scores if available
        if behavior in behavior_scores:
            score = behavior_scores[behavior]
            return EvaluateBehaviorResult(
                behavior=behavior,
                recommendation="approach" if score > 0.5 else "avoid",
                confidence=abs(score - 0.5) * 2,  # Convert to confidence
                reasoning=f"Based on context-adjusted preferences (score: {score:.2f})"
            )
        
        # Fall back to conditioning system if available
        if context.conditioning_system and hasattr(context.conditioning_system, 'evaluate_behavior_consequences'):
            result = await context.conditioning_system.evaluate_behavior_consequences(
                behavior=behavior,
                context={
                    "detected_patterns": [p.pattern_name for p in detected_patterns],
                    "user_history": user_history.model_dump() if user_history else {},
                    "adjusted_sensitivities": adjusted_sensitivities
                }
            )
            
            # Convert associations to AssociationData objects
            associations = []
            if result.get("relevant_associations"):
                for assoc in result["relevant_associations"]:
                    associations.append(AssociationData(
                        id=assoc.get("id", ""),
                        type=assoc.get("type", "unknown"),
                        strength=assoc.get("strength", 0.5),
                        context=assoc.get("context", "")
                    ))
            
            return EvaluateBehaviorResult(
                behavior=result.get("behavior", behavior),
                recommendation=result.get("recommendation", "avoid"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "Based on conditioning system evaluation"),
                associations=associations if associations else None
            )
        
        # Fallback logic if no conditioning system is available
        pattern_names = [p.pattern_name for p in detected_patterns]
        
        # Simple rule-based evaluation
        if behavior == "dominant_response":
            if "submission_language" in pattern_names:
                return EvaluateBehaviorResult(
                    behavior=behavior,
                    recommendation="approach",
                    confidence=0.8,
                    reasoning="Submission language detected, dominant response is appropriate"
                )
            elif "defiance" in pattern_names:
                return EvaluateBehaviorResult(
                    behavior=behavior,
                    recommendation="approach",
                    confidence=0.7,
                    reasoning="Defiance detected, dominant response may be needed"
                )
            else:
                return EvaluateBehaviorResult(
                    behavior=behavior,
                    recommendation="avoid",
                    confidence=0.6,
                    reasoning="No submission or defiance detected, dominant response not clearly indicated"
                )
        
        elif behavior == "teasing_response":
            if "flattery" in pattern_names:
                return EvaluateBehaviorResult(
                    behavior=behavior,
                    recommendation="approach",
                    confidence=0.7,
                    reasoning="Flattery detected, teasing response can be appropriate"
                )
            else:
                return EvaluateBehaviorResult(
                    behavior=behavior,
                    recommendation="avoid",
                    confidence=0.6,
                    reasoning="No clear indicator for teasing response"
                )
        
        # Default response for other behaviors
        return EvaluateBehaviorResult(
            behavior=behavior,
            recommendation="avoid",
            confidence=0.5,
            reasoning="No clear indicator for this behavior"
        )

    @staticmethod
    @function_tool
    async def _process_operant_conditioning(
        ctx: RunContextWrapper[InputProcessingAgentContext],
        input_data: ProcessConditioningInput
    ) -> ProcessConditioningResult:
        """
        Process operant conditioning for a behavior.
        
        Args:
            ctx: The run context wrapper
            input_data: Input data for conditioning
            
        Returns:
            Result of the conditioning process
        """
        processor_ctx = ctx.context
        behavior = input_data.behavior
        consequence_type = input_data.consequence_type
        intensity = input_data.intensity
        context_info = input_data.context_info
        
        if processor_ctx.conditioning_system and hasattr(processor_ctx.conditioning_system, 'process_operant_conditioning'):
            # Use the actual conditioning system if available
            context_dict = context_info.model_dump() if context_info else {}
            result = await processor_ctx.conditioning_system.process_operant_conditioning(
                behavior=behavior,
                consequence_type=consequence_type,
                intensity=intensity,
                context=context_dict
            )
            return ProcessConditioningResult(
                behavior=result.get("behavior", behavior),
                consequence_type=result.get("consequence_type", consequence_type),
                intensity=result.get("intensity", intensity),
                effect=result.get("effect", "Effect applied"),
                success=result.get("success", True)
            )
        
        # Fallback logic if no conditioning system is available
        effect = "increase likelihood" if consequence_type.startswith("positive_") else "decrease likelihood"
        
        return ProcessConditioningResult(
            behavior=behavior,
            consequence_type=consequence_type,
            intensity=intensity,
            effect=f"Will {effect} of {behavior} in the future",
            success=True
        )

    @staticmethod
    @function_tool
    async def _get_mode_preferences(
        ctx: RunContextWrapper[InputProcessingAgentContext],
        input_data: ModePreferencesInput
    ) -> ModePreferencesResult:
        """
        Get response modification preferences for a specific mode
        
        Args:
            ctx: The run context wrapper
            input_data: Input data containing mode
            
        Returns:
            Mode preferences
        """
        processor_ctx = ctx.context
        mode = input_data.mode
        
        if not mode:
            return ModePreferencesResult(
                mode="unknown",
                add_elements=[],
                remove_elements=[],
                tone_elements=[],
                phrasing_examples=[],
                typical_pronouns=[]
            )
            
        preferences = processor_ctx.mode_response_preferences.get(mode.lower(), {})
        
        if not preferences and processor_ctx.mode_manager:
            # Try getting from mode manager if available
            try:
                mode_enum = InteractionMode(mode.lower())
                # Construct basic preferences from mode parameters and conversation style
                mode_params = processor_ctx.mode_manager.get_mode_parameters(mode_enum.value)
                conv_style = processor_ctx.mode_manager.get_conversation_style(mode_enum.value)
                vocal_patterns = processor_ctx.mode_manager.get_vocalization_patterns(mode_enum.value)
                
                if mode_params and conv_style and vocal_patterns:
                    # Extract tone from conversation style
                    tone = conv_style.tone
                    tone_elements = [t.strip() for t in tone.split(",")] if tone else []
                    
                    # Extract statement types
                    statement_types = conv_style.types_of_statements
                    add_elements = [s.strip() for s in statement_types.split(",")] if statement_types else []
                    
                    # Extract topics
                    topics_to_avoid = conv_style.topics_to_avoid
                    remove_elements = [t.strip() for t in topics_to_avoid.split(",")] if topics_to_avoid else []
                    
                    # Convert lexical preferences if present
                    lexical_prefs = None
                    if preferences.get("lexical_preferences"):
                        lexical_prefs = LexicalPreferences(
                            high=preferences["lexical_preferences"].get("high", []),
                            low=preferences["lexical_preferences"].get("low", [])
                        )
                    
                    # Return constructed preferences
                    return ModePreferencesResult(
                        mode=mode,
                        add_elements=add_elements,
                        remove_elements=remove_elements,
                        tone_elements=tone_elements,
                        phrasing_examples=vocal_patterns.key_phrases,
                        typical_pronouns=vocal_patterns.pronouns,
                        lexical_preferences=lexical_prefs
                    )
            except Exception as e:
                logger.warning(f"Error getting mode preferences from mode manager: {e}")
        
        # Convert lexical preferences to model
        lexical_prefs = None
        if preferences.get("lexical_preferences"):
            lexical_prefs = LexicalPreferences(
                high=preferences["lexical_preferences"].get("high", []),
                low=preferences["lexical_preferences"].get("low", [])
            )
        
        # Return preferences from stored data
        return ModePreferencesResult(
            mode=mode,
            add_elements=preferences.get("add_elements", []),
            remove_elements=preferences.get("remove_elements", []),
            tone_elements=preferences.get("tone_elements", []),
            phrasing_examples=preferences.get("phrasing_examples", []),
            typical_pronouns=preferences.get("typical_pronouns", []),
            lexical_preferences=lexical_prefs
        )

    @staticmethod
    @function_tool
    async def _calculate_style_elements(
        ctx: RunContextWrapper[InputProcessingAgentContext],
        input_data: StyleElementsInput
    ) -> StyleElementsResult:
        """
        Calculate blended style elements based on mode distribution
        
        Args:
            ctx: The run context wrapper
            input_data: Input data containing mode distribution
            
        Returns:
            Blended style elements
        """
        processor_ctx = ctx.context
        mode_distribution = input_data.mode_distribution
        
        # Convert ModeDistributionModel to dict for processing
        mode_dict = mode_distribution.model_dump()
        
        # Initialize style elements
        blended_style = BlendedStyleModel()
        
        # Track mode influences
        influences = ModeDistributionModel()
        element_sources = ElementSourcesModel()
        
        # Get significant modes (weight >= 0.2)
        significant_modes = {mode: weight for mode, weight in mode_dict.items() if weight >= 0.2}
        
        # Normalize significant mode weights
        total_weight = sum(significant_modes.values())
        normalized_weights = {mode: weight/total_weight for mode, weight in significant_modes.items()} if total_weight > 0 else {}
        
        # Temporary storage for element sources
        temp_sources = {
            "add_elements": {},
            "remove_elements": {},
            "tone_elements": {},
            "phrasing_examples": {},
            "typical_pronouns": {}
        }
        
        # For each significant mode
        for mode, norm_weight in normalized_weights.items():
            # Get mode preferences
            pref_input = ModePreferencesInput(mode=mode)
            preferences = await BlendedInputProcessor._get_mode_preferences(ctx, pref_input)
            
            if not preferences:
                continue
                
            # Record mode influence
            setattr(influences, mode, norm_weight)
            
            # Add weighted elements based on mode weight
            for element_type in ["add_elements", "remove_elements", "tone_elements"]:
                elements = getattr(preferences, element_type, [])
                if elements:
                    # Number of elements to include based on weight
                    num_elements = max(1, round(len(elements) * norm_weight))
                    
                    # Select top elements
                    top_elements = elements[:num_elements]
                    current_list = getattr(blended_style, element_type)
                    current_list.extend(top_elements)
                    
                    # Record sources
                    for element in top_elements:
                        if element not in temp_sources[element_type]:
                            temp_sources[element_type][element] = []
                        temp_sources[element_type][element].append(mode)
            
            # Add phrasing examples based on weight
            if preferences.phrasing_examples:
                # Number of phrases to include
                num_phrases = max(1, round(len(preferences.phrasing_examples) * norm_weight))
                
                # Select top phrases
                top_phrases = preferences.phrasing_examples[:num_phrases]
                blended_style.phrasing_examples.extend(top_phrases)
                
                # Record sources
                for phrase in top_phrases:
                    if phrase not in temp_sources["phrasing_examples"]:
                        temp_sources["phrasing_examples"][phrase] = []
                    temp_sources["phrasing_examples"][phrase].append(mode)
            
            # Add pronouns
            if preferences.typical_pronouns:
                blended_style.typical_pronouns.extend(preferences.typical_pronouns)
                
                # Record sources
                for pronoun in preferences.typical_pronouns:
                    if pronoun not in temp_sources["typical_pronouns"]:
                        temp_sources["typical_pronouns"][pronoun] = []
                    temp_sources["typical_pronouns"][pronoun].append(mode)
        
        # Remove duplicates while preserving order
        for element_type in ["add_elements", "remove_elements", "tone_elements", "phrasing_examples", "typical_pronouns"]:
            current_list = getattr(blended_style, element_type)
            seen = set()
            unique_list = [x for x in current_list if not (x in seen or seen.add(x))]
            setattr(blended_style, element_type, unique_list)
        
        # Convert temp_sources to ElementSourcesModel
        for element_type, sources_dict in temp_sources.items():
            items = []
            for element, sources in sources_dict.items():
                items.append(ElementSourceItem(element=element, sources=sources))
            setattr(element_sources, element_type, items)
        
        return StyleElementsResult(
            blended_style=blended_style,
            influences=influences,
            element_sources=element_sources
        )

    @staticmethod
    @function_tool
    async def _analyze_response_coherence(
        ctx: RunContextWrapper[InputProcessingAgentContext],
        input_data: CoherenceAnalysisInput
    ) -> CoherenceAnalysisResult:
        """
        Analyze the coherence of a modified response
        
        Args:
            ctx: The run context wrapper
            input_data: Input data containing original and modified responses
            
        Returns:
            Coherence analysis
        """
        original_response = input_data.original_response
        modified_response = input_data.modified_response
        
        # Handle empty strings
        if not original_response and not modified_response:
            return CoherenceAnalysisResult(
                coherence_score=1.0,
                metrics=CoherenceMetrics(
                    length_change=1.0,
                    word_retention=1.0,
                    sentence_count_ratio=1.0
                ),
                is_coherent=True
            )
        
        if not original_response:
            original_response = " "  # Avoid division by zero
        if not modified_response:
            modified_response = " "
        
        # Calculate simple metrics
        
        # 1. Length change
        original_length = len(original_response)
        modified_length = len(modified_response)
        length_change = modified_length / original_length if original_length > 0 else 1.0
        
        # 2. Word retention (what percentage of original words were kept)
        original_words = set(original_response.lower().split())
        modified_words = set(modified_response.lower().split())
        retained_words = original_words.intersection(modified_words)
        word_retention = len(retained_words) / len(original_words) if original_words else 1.0
        
        # 3. Sentence structure preservation
        original_sentences = original_response.split('.')
        modified_sentences = modified_response.split('.')
        sentence_count_ratio = len(modified_sentences) / len(original_sentences) if original_sentences else 1.0
        
        # Calculate coherence score
        # High retention with some changes is ideal
        # Too many changes or too few both reduce coherence
        coherence_score = 0.5  # Base score
        
        # Penalize excessive length changes
        if length_change < 0.7 or length_change > 1.5:
            coherence_score -= 0.1
        
        # Reward moderate word retention (60-90% is ideal)
        if 0.6 <= word_retention <= 0.9:
            coherence_score += 0.2
        elif word_retention < 0.5:  # Too many words changed
            coherence_score -= 0.2
            
        # Penalize excessive sentence structure changes
        if sentence_count_ratio < 0.7 or sentence_count_ratio > 1.5:
            coherence_score -= 0.1
            
        # Ensure score is in range
        coherence_score = max(0.0, min(1.0, coherence_score))
        
        return CoherenceAnalysisResult(
            coherence_score=coherence_score,
            metrics=CoherenceMetrics(
                length_change=length_change,
                word_retention=word_retention,
                sentence_count_ratio=sentence_count_ratio
            ),
            is_coherent=coherence_score >= 0.5
        )
    
    async def update_context_from_brain_state(self):
        """Update shared context with current brain state"""
        if not self.brain:
            return
            
        # Apply emotional influence
        if hasattr(self.brain, 'emotional_state') and self.brain.emotional_state:
            emotional_data = {
                "dominant_emotion": getattr(self.brain.emotional_state, 'dominant_emotion', 'neutral'),
                "intensity": getattr(self.brain.emotional_state, 'intensity', 0.5)
            }
            self.shared_context.apply_emotional_influence(emotional_data)
        
        # Apply mode influence
        if self.context.mode_manager and hasattr(self.context.mode_manager, 'context'):
            try:
                mode_dist = self.context.mode_manager.context.mode_distribution
                mode_dict = mode_dist.model_dump() if hasattr(mode_dist, 'model_dump') else mode_dist.dict()
                self.shared_context.apply_mode_influence({"mode_distribution": mode_dict})
            except:
                pass
        
        # Apply relationship influence
        if hasattr(self.brain, 'relationship_state') and self.brain.relationship_state:
            relationship_data = {
                "trust": getattr(self.brain.relationship_state, 'trust', 0.5),
                "intimacy": getattr(self.brain.relationship_state, 'intimacy', 0.5),
                "dominance_accepted": getattr(self.brain.relationship_state, 'dominance_accepted', 0.5),
                "conflict": getattr(self.brain.relationship_state, 'conflict', 0.0)
            }
            self.shared_context.apply_relationship_influence(relationship_data)
    
    async def process_input(self, text: str, user_id: str = "default", context: Optional[Dict[str, Any]] = None) -> ProcessInputResult:
        """
        Process input text through conditioning system and return processing results
        
        Args:
            text: Input text
            user_id: User ID for personalization
            context: Additional context information
            
        Returns:
            Processing results including triggered responses
        """
        with trace(workflow_name="process_input", group_id=getattr(self.brain, 'trace_group_id', 'default')):
            # Update context from brain state
            await self.update_context_from_brain_state()
            
            # Prepare the prompt for pattern analysis
            pattern_prompt = f"""
            Analyze the following input text for patterns indicating submission, defiance, 
            flattery, disrespect, or embarrassment:
            
            USER INPUT: {text}
            
            USER ID: {user_id}
            
            {f"ADDITIONAL CONTEXT: {context}" if context else ""}
            
            Identify all patterns present in the input.
            """
            
            # Run the pattern analyzer agent
            pattern_result = await Runner.run(self.pattern_analyzer_agent, pattern_prompt, context=self.context)
            detected_patterns = pattern_result.final_output
            
            # Prepare data for behavior selection
            potential_behaviors = ["dominant_response", "teasing_response", "direct_response", "playful_response"]
            
            # Run behavior selection for each potential behavior
            behavior_prompt = f"""
            Evaluate which behaviors are appropriate based on these detected patterns:
            
            DETECTED PATTERNS: {[p.model_dump() for p in detected_patterns]}
            
            USER ID: {user_id}
            
            {f"ADDITIONAL CONTEXT: {context}" if context else ""}
            
            For each potential behavior (dominant, teasing, direct, playful),
            evaluate whether it should be approached or avoided.
            """
            
            behavior_result = await Runner.run(self.behavior_selector_agent, behavior_prompt, context=self.context)
            behavior_evaluations = behavior_result.final_output
            
            # Extract recommendations
            recommended_behaviors = [
                eval.behavior for eval in behavior_evaluations 
                if eval.recommendation == "approach" and eval.confidence > 0.5
            ]
            
            avoided_behaviors = [
                eval.behavior for eval in behavior_evaluations
                if eval.recommendation == "avoid" and eval.confidence > 0.5
            ]
            
            # Trigger conditioning based on patterns
            reinforcement_results = []
            
            # Reinforcement for submission language (if detected)
            if any(p.pattern_name == "submission_language" for p in detected_patterns):
                cond_input = ProcessConditioningInput(
                    behavior="submission_language_response",
                    consequence_type="positive_reinforcement",
                    intensity=0.8,
                    context_info=ContextInfo(
                        user_id=user_id,
                        context_keys=["conversation"]
                    )
                )
                reinforcement = await self._process_operant_conditioning(
                    RunContextWrapper(self.context),
                    cond_input
                )
                reinforcement_results.append(reinforcement)
            
            # Punishment for defiance (if detected)
            if any(p.pattern_name == "defiance" for p in detected_patterns):
                cond_input = ProcessConditioningInput(
                    behavior="tolerate_defiance",
                    consequence_type="positive_punishment",
                    intensity=0.7,
                    context_info=ContextInfo(
                        user_id=user_id,
                        context_keys=["conversation"]
                    )
                )
                punishment = await self._process_operant_conditioning(
                    RunContextWrapper(self.context),
                    cond_input
                )
                reinforcement_results.append(punishment)
            
            # Get current mode distribution if available
            mode_distribution = ModeDistributionModel()
            if self.context.mode_manager and hasattr(self.context.mode_manager, 'context'):
                try:
                    mode_dist = self.context.mode_manager.context.mode_distribution
                    mode_dict = mode_dist.model_dump() if hasattr(mode_dist, 'model_dump') else mode_dist.dict()
                    mode_distribution = dict_to_mode_distribution(mode_dict)
                except:
                    pass
            
            # Convert dicts to models
            adjusted_sensitivities = dict_to_sensitivities(self.shared_context.get_adjusted_sensitivities())
            behavior_scores = dict_to_behavior_scores(self.shared_context.get_behavior_scores())
            
            # Collect results
            return ProcessInputResult(
                input_text=text,
                user_id=user_id,
                detected_patterns=detected_patterns,
                behavior_evaluations=behavior_evaluations,
                recommended_behaviors=recommended_behaviors,
                avoided_behaviors=avoided_behaviors,
                reinforcement_results=reinforcement_results,
                mode_distribution=mode_distribution,
                adjusted_sensitivities=adjusted_sensitivities,
                behavior_scores=behavior_scores
            )
    
    async def modify_response(self, response_text: str, input_processing_results: ProcessInputResult) -> str:
        """
        Modify response based on conditioning results and mode distribution
        
        Args:
            response_text: Original response text
            input_processing_results: Results from process_input
            
        Returns:
            Modified response text
        """
        with trace(workflow_name="modify_conditioned_response", group_id=getattr(self.brain, 'trace_group_id', 'default')):
            # Check if mode distribution is available
            mode_distribution = input_processing_results.mode_distribution.model_dump()
            
            # Fall back to current mode manager state if not in results
            if not any(mode_distribution.values()) and self.context.mode_manager and hasattr(self.context.mode_manager, 'context'):
                try:
                    mode_dist = self.context.mode_manager.context.mode_distribution
                    mode_dict = mode_dist.model_dump() if hasattr(mode_dist, 'model_dump') else mode_dist.dict()
                    mode_distribution = mode_dict
                except:
                    pass
            
            # If we have a mode distribution, use blended modification
            if mode_distribution and any(weight >= 0.2 for weight in mode_distribution.values()):
                # Prepare the prompt for blended modification
                blended_prompt = f"""
                Modify the following response based on the blended mode distribution:
                
                ORIGINAL RESPONSE: {response_text}
                
                MODE DISTRIBUTION: {mode_distribution}
                
                DETECTED PATTERNS: {[p.model_dump() for p in input_processing_results.detected_patterns]}
                
                RECOMMENDED BEHAVIORS: {input_processing_results.recommended_behaviors}
                
                AVOIDED BEHAVIORS: {input_processing_results.avoided_behaviors}
                
                Modify the response to proportionally reflect all active modes in the distribution,
                with higher-weighted modes having more influence on the final style and tone.
                Create a natural, coherent response that integrates elements from all active modes.
                """
                
                # Run the blended modifier agent
                result = await Runner.run(self.blended_modifier_agent, blended_prompt, context=self.context)
                
                # Return the modified text from the result
                blended_result = result.final_output
                
                # Log the coherence of the modification
                coherence = blended_result.coherence
                logger.info(f"Modified response with blended modes (coherence: {coherence:.2f})")
                
                return blended_result.modified_text
            else:
                # Fall back to original behavior-based modification
                # Prepare the prompt for response modification
                modification_prompt = f"""
                Modify the following response based on behavior recommendations and detected patterns:
                
                ORIGINAL RESPONSE: {response_text}
                
                DETECTED PATTERNS: {[p.model_dump() for p in input_processing_results.detected_patterns]}
                
                RECOMMENDED BEHAVIORS: {input_processing_results.recommended_behaviors}
                
                AVOIDED BEHAVIORS: {input_processing_results.avoided_behaviors}
                
                REINFORCEMENT RESULTS: {[r.model_dump() for r in input_processing_results.reinforcement_results]}
                
                Modify the response to align with recommended behaviors while avoiding
                behaviors that should be avoided. Ensure the modification is subtle but effective.
                """
                
                # Run the response modifier agent
                result = await Runner.run(self.response_modifier_agent, modification_prompt, context=self.context)
                modified_response = result.final_output
                
                return modified_response
    
    async def modify_blended_response(self, response_text: str, mode_distribution: Dict[str, float]) -> BlendedResponseModification:
        """
        Modify response based purely on mode distribution
        
        Args:
            response_text: Original response text
            mode_distribution: Current mode distribution
            
        Returns:
            Modified response information with details
        """
        with trace(workflow_name="modify_blended_response", group_id=getattr(self.brain, 'trace_group_id', 'default')):
            # Convert dict to model
            mode_dist_model = dict_to_mode_distribution(mode_distribution)
            
            # Prepare the prompt
            blended_prompt = f"""
            Modify the following response based on the given mode distribution:
            
            ORIGINAL RESPONSE: {response_text}
            
            MODE DISTRIBUTION: {mode_dist_model.model_dump()}
            
            Modify the response to proportionally reflect all active modes in the distribution,
            with higher-weighted modes having more influence on the final style and tone.
            Create a natural, coherent response that integrates elements from all active modes.
            Focus on maintaining the core message while adapting the style, tone, and phrasing.
            """
            
            # Run the blended modifier agent
            result = await Runner.run(self.blended_modifier_agent, blended_prompt, context=self.context)
            
            # Return the full result
            return result.final_output
    
    def get_shared_context(self) -> SharedInputContext:
        """Get the shared input processing context for coordination with other systems"""
        return self.shared_context
