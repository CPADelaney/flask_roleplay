# nyx/core/input_processor.py

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import random
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, ModelSettings, RunContextWrapper

from nyx.core.interaction_mode_manager import ModeDistribution, InteractionMode

logger = logging.getLogger(__name__)

# Pydantic models for structured data
class PatternDetection(BaseModel):
    pattern_name: str = Field(description="Name of the detected pattern")
    confidence: float = Field(description="Confidence level for the detection (0.0-1.0)")
    matched_text: str = Field(description="Text that matched the pattern")

class ConditionedResponse(BaseModel):
    response_type: str = Field(description="Type of conditioned response")
    strength: float = Field(description="Strength of the response (0.0-1.0)")
    description: str = Field(description="Description of the triggered response")

class BehaviorEvaluation(BaseModel):
    behavior: str = Field(description="Behavior being evaluated")
    recommendation: str = Field(description="Approach or avoid recommendation")
    confidence: float = Field(description="Confidence in the recommendation (0.0-1.0)")
    reasoning: str = Field(description="Reasoning for the evaluation")

class OperantConditioningResult(BaseModel):
    behavior: str = Field(description="Behavior being conditioned")
    consequence_type: str = Field(description="Type of operant conditioning")
    intensity: float = Field(description="Intensity of the conditioning (0.0-1.0)")
    effect: str = Field(description="Expected effect on future behavior")

class BlendedResponseModification(BaseModel):
    """Output schema for blended response modification"""
    modified_text: str = Field(description="Modified response text")
    mode_influences: Dict[str, float] = Field(description="Influence of each mode on the modification")
    modifications_made: List[Dict[str, Any]] = Field(description="List of modifications made to the response")
    coherence: float = Field(description="Coherence of the modified response (0.0-1.0)", ge=0.0, le=1.0)
    style_notes: Optional[str] = Field(None, description="Notes about the style of the modified response")

class ProcessingContext:
    """Context for input processing operations"""
    
    def __init__(self, conditioning_system=None, emotional_core=None, somatosensory_system=None, mode_manager=None):
        self.conditioning_system = conditioning_system
        self.emotional_core = emotional_core
        self.somatosensory_system = somatosensory_system
        self.mode_manager = mode_manager
        
        # Pattern definitions (could be moved to a configuration file)
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

class BlendedInputProcessor:
    """
    Processes input through conditioning triggers and modifies responses
    using blended interaction modes with the OpenAI Agents SDK architecture.
    """
    
    def __init__(self, conditioning_system=None, emotional_core=None, somatosensory_system=None, mode_manager=None):
        self.context = ProcessingContext(
            conditioning_system=conditioning_system,
            emotional_core=emotional_core,
            somatosensory_system=somatosensory_system,
            mode_manager=mode_manager
        )
        
        # Initialize the agents
        self.pattern_analyzer_agent = self._create_pattern_analyzer()
        self.behavior_selector_agent = self._create_behavior_selector()
        self.response_modifier_agent = self._create_response_modifier()
        self.blended_modifier_agent = self._create_blended_modifier()
        
        logger.info("Blended input processor initialized with agents")
    
    def _create_pattern_analyzer(self) -> Agent:
        """Create an agent specialized in analyzing input patterns"""
        return Agent(
            name="Pattern Analyzer",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                function_tool(self._detect_patterns)
            ],
            output_type=List[PatternDetection]
        )
    
    def _create_behavior_selector(self) -> Agent:
        """Create an agent specialized in selecting appropriate behaviors"""
        return Agent(
            name="Behavior Selector",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._evaluate_behavior),
                function_tool(self._process_operant_conditioning)
            ],
            output_type=List[BehaviorEvaluation]
        )
    
    def _create_response_modifier(self) -> Agent:
        """Create an agent specialized in modifying responses"""
        return Agent(
            name="Response Modifier",
            instructions="""
            You modify response text based on behavior recommendations and detected patterns.
            
            Your job is to:
            1. Add or remove elements that match recommended behaviors
            2. Incorporate appropriate conditioning based on detected patterns
            3. Ensure the modified response maintains coherence and natural flow
            
            Modifications should be subtle but effective, maintaining the core message
            while adjusting tone, phrasing, and emphasis.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.4),
            output_type=str
        )
    
    def _create_blended_modifier(self) -> Agent:
        """Create an agent specialized in blended response modification"""
        return Agent(
            name="Blended Modifier",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.4),
            tools=[
                function_tool(self._get_mode_preferences),
                function_tool(self._calculate_style_elements),
                function_tool(self._analyze_response_coherence)
            ],
            output_type=BlendedResponseModification
        )

    @staticmethod
    @function_tool
    async def _detect_patterns(ctx: RunContextWrapper[ProcessingContext], text: str) -> List[Dict[str, Any]]:
        """
        Detect patterns in input text using regular expressions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected patterns with confidence scores
        """
        context = ctx.context
        detected = []
        
        for pattern_name, regex_list in context.input_patterns.items():
            for regex in regex_list:
                match = re.search(regex, text)
                if match:
                    detected.append({
                        "pattern_name": pattern_name,
                        "confidence": 0.8,  # Base confidence
                        "matched_text": match.group(0)
                    })
                    break  # Only detect each pattern once
        
        return detected

    @staticmethod
    @function_tool
    async def _evaluate_behavior(
        ctx: RunContextWrapper[ProcessingContext], 
        behavior: str,
        detected_patterns: List[Dict[str, Any]],
        user_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if a behavior should be approached or avoided.
        
        Args:
            behavior: Behavior to evaluate
            detected_patterns: Patterns detected in the input
            user_history: Optional history of user interactions
            
        Returns:
            Evaluation result with recommendation
        """
        context = ctx.context
        if context.conditioning_system and hasattr(context.conditioning_system, 'evaluate_behavior_consequences'):
            # Use the actual conditioning system if available
            result = await context.conditioning_system.evaluate_behavior_consequences(
                behavior=behavior,
                context={
                    "detected_patterns": [p["pattern_name"] for p in detected_patterns],
                    "user_history": user_history or {}
                }
            )
            return result
        
        # Fallback logic if no conditioning system is available
        pattern_names = [p["pattern_name"] for p in detected_patterns]
        
        # Simple rule-based evaluation
        if behavior == "dominant_response":
            if "submission_language" in pattern_names:
                return {
                    "behavior": behavior,
                    "recommendation": "approach",
                    "confidence": 0.8,
                    "reasoning": "Submission language detected, dominant response is appropriate"
                }
            elif "defiance" in pattern_names:
                return {
                    "behavior": behavior,
                    "recommendation": "approach",
                    "confidence": 0.7,
                    "reasoning": "Defiance detected, dominant response may be needed"
                }
            else:
                return {
                    "behavior": behavior,
                    "recommendation": "avoid",
                    "confidence": 0.6,
                    "reasoning": "No submission or defiance detected, dominant response not clearly indicated"
                }
        
        elif behavior == "teasing_response":
            if "flattery" in pattern_names:
                return {
                    "behavior": behavior,
                    "recommendation": "approach",
                    "confidence": 0.7,
                    "reasoning": "Flattery detected, teasing response can be appropriate"
                }
            else:
                return {
                    "behavior": behavior,
                    "recommendation": "avoid",
                    "confidence": 0.6,
                    "reasoning": "No clear indicator for teasing response"
                }
        
        # Default response for other behaviors
        return {
            "behavior": behavior,
            "recommendation": "avoid",
            "confidence": 0.5,
            "reasoning": "No clear indicator for this behavior"
        }

    @staticmethod
    @function_tool
    async def _process_operant_conditioning(
        ctx: RunContextWrapper[ProcessingContext],
        behavior: str,
        consequence_type: str,
        intensity: float = 0.5,
        context_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process operant conditioning for a behavior.
        
        Args:
            behavior: Behavior being conditioned
            consequence_type: Type of operant conditioning (positive/negative reinforcement/punishment)
            intensity: Intensity of the conditioning (0.0-1.0)
            context_info: Additional context information
            
        Returns:
            Result of the conditioning process
        """
        processor_ctx = ctx.context
        if processor_ctx.conditioning_system and hasattr(processor_ctx.conditioning_system, 'process_operant_conditioning'):
            # Use the actual conditioning system if available
            result = await processor_ctx.conditioning_system.process_operant_conditioning(
                behavior=behavior,
                consequence_type=consequence_type,
                intensity=intensity,
                context=context_info or {}
            )
            return result
        
        # Fallback logic if no conditioning system is available
        effect = "increase likelihood" if consequence_type.startswith("positive_") else "decrease likelihood"
        
        return {
            "behavior": behavior,
            "consequence_type": consequence_type,
            "intensity": intensity,
            "effect": f"Will {effect} of {behavior} in the future",
            "success": True
        }

    @staticmethod
    @function_tool
    async def _get_mode_preferences(
        ctx: RunContextWrapper[ProcessingContext],
        mode: str
    ) -> Dict[str, Any]:
        """
        Get response modification preferences for a specific mode
        
        Args:
            mode: The interaction mode
            
        Returns:
            Response modification preferences for the mode
        """
        processor_ctx = ctx.context
        preferences = processor_ctx.mode_response_preferences.get(mode.lower(), {})
        
        if not preferences and processor_ctx.mode_manager:
            # Try getting from mode manager if available
            try:
                mode_enum = InteractionMode(mode.lower())
                # Construct basic preferences from mode parameters and conversation style
                mode_params = processor_ctx.mode_manager.get_mode_parameters(mode_enum)
                conv_style = processor_ctx.mode_manager.get_conversation_style(mode_enum)
                vocal_patterns = processor_ctx.mode_manager.get_vocalization_patterns(mode_enum)
                
                if mode_params and conv_style:
                    # Extract tone from conversation style
                    tone = conv_style.get("tone", "")
                    tone_elements = [t.strip() for t in tone.split(",")] if tone else []
                    
                    # Extract statement types
                    statement_types = conv_style.get("types_of_statements", "")
                    add_elements = [s.strip() for s in statement_types.split(",")] if statement_types else []
                    
                    # Extract topics to emphasize/avoid
                    topics_to_emphasize = conv_style.get("topics_to_emphasize", "")
                    topics_to_avoid = conv_style.get("topics_to_avoid", "")
                    
                    # Extract key phrases
                    key_phrases = vocal_patterns.get("key_phrases", []) if vocal_patterns else []
                    
                    # Construct preferences
                    preferences = {
                        "add_elements": add_elements,
                        "remove_elements": [t.strip() for t in topics_to_avoid.split(",")] if topics_to_avoid else [],
                        "tone_elements": tone_elements,
                        "phrasing_examples": key_phrases,
                        "typical_pronouns": vocal_patterns.get("pronouns", []) if vocal_patterns else []
                    }
            except Exception as e:
                logger.warning(f"Error getting mode preferences from mode manager: {e}")
        
        return preferences

    @staticmethod
    @function_tool
    async def _calculate_style_elements(
        ctx: RunContextWrapper[ProcessingContext],
        mode_distribution: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate blended style elements based on mode distribution
        
        Args:
            mode_distribution: The mode distribution
            
        Returns:
            Blended style elements
        """
        processor_ctx = ctx.context
        
        # Initialize style elements
        blended_style = {
            "add_elements": [],
            "remove_elements": [],
            "tone_elements": [],
            "phrasing_examples": [],
            "typical_pronouns": []
        }
        
        # Track mode influences
        influences = {}
        element_sources = {
            "add_elements": {},
            "remove_elements": {},
            "tone_elements": {},
            "phrasing_examples": {},
            "typical_pronouns": {}
        }
        
        # Get significant modes (weight >= 0.2)
        significant_modes = {mode: weight for mode, weight in mode_distribution.items() if weight >= 0.2}
        
        # Normalize significant mode weights
        total_weight = sum(significant_modes.values())
        normalized_weights = {mode: weight/total_weight for mode, weight in significant_modes.items()} if total_weight > 0 else {}
        
        # For each significant mode
        for mode, norm_weight in normalized_weights.items():
            # Get mode preferences
            preferences = await self._get_mode_preferences(ctx, mode)
            
            if not preferences:
                continue
                
            # Record mode influence
            influences[mode] = norm_weight
            
            # Add weighted elements based on mode weight
            for element_type in ["add_elements", "remove_elements", "tone_elements"]:
                if element_type in preferences:
                    # Number of elements to include based on weight
                    num_elements = max(1, round(len(preferences[element_type]) * norm_weight))
                    
                    # Select top elements
                    top_elements = preferences[element_type][:num_elements]
                    blended_style[element_type].extend(top_elements)
                    
                    # Record sources
                    for element in top_elements:
                        if element not in element_sources[element_type]:
                            element_sources[element_type][element] = []
                        element_sources[element_type][element].append(mode)
            
            # Add phrasing examples based on weight
            if "phrasing_examples" in preferences:
                # Number of phrases to include
                num_phrases = max(1, round(len(preferences["phrasing_examples"]) * norm_weight))
                
                # Select top phrases
                top_phrases = preferences["phrasing_examples"][:num_phrases]
                blended_style["phrasing_examples"].extend(top_phrases)
                
                # Record sources
                for phrase in top_phrases:
                    if phrase not in element_sources["phrasing_examples"]:
                        element_sources["phrasing_examples"][phrase] = []
                    element_sources["phrasing_examples"][phrase].append(mode)
            
            # Add pronouns
            if "typical_pronouns" in preferences:
                blended_style["typical_pronouns"].extend(preferences["typical_pronouns"])
                
                # Record sources
                for pronoun in preferences["typical_pronouns"]:
                    if pronoun not in element_sources["typical_pronouns"]:
                        element_sources["typical_pronouns"][pronoun] = []
                    element_sources["typical_pronouns"][pronoun].append(mode)
        
        # Remove duplicates while preserving order
        for element_type in blended_style:
            seen = set()
            blended_style[element_type] = [x for x in blended_style[element_type] if not (x in seen or seen.add(x))]
        
        return {
            "blended_style": blended_style,
            "influences": influences,
            "element_sources": element_sources
        }

    @staticmethod
    @function_tool
    async def _analyze_response_coherence(
        ctx: RunContextWrapper[ProcessingContext],
        original_response: str,
        modified_response: str
    ) -> Dict[str, Any]:
        """
        Analyze the coherence of a modified response
        
        Args:
            original_response: Original response text
            modified_response: Modified response text
            
        Returns:
            Coherence analysis
        """
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
        
        return {
            "coherence_score": coherence_score,
            "metrics": {
                "length_change": length_change,
                "word_retention": word_retention,
                "sentence_count_ratio": sentence_count_ratio
            },
            "is_coherent": coherence_score >= 0.5
        }
    
    async def process_input(self, text: str, user_id: str = "default", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input text through conditioning system and return processing results
        
        Args:
            text: Input text
            user_id: User ID for personalization
            context: Additional context information
            
        Returns:
            Processing results including triggered responses
        """
        with trace(workflow_name="process_conditioned_input"):
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
            
            DETECTED PATTERNS: {[p.dict() for p in detected_patterns]}
            
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
                reinforcement = await self._process_operant_conditioning(
                    RunContextWrapper(self.context),
                    behavior="submission_language_response",
                    consequence_type="positive_reinforcement",
                    intensity=0.8,
                    context_info={
                        "user_id": user_id,
                        "context_keys": ["conversation"]
                    }
                )
                reinforcement_results.append(reinforcement)
            
            # Punishment for defiance (if detected)
            if any(p.pattern_name == "defiance" for p in detected_patterns):
                punishment = await self._process_operant_conditioning(
                    RunContextWrapper(self.context),
                    behavior="tolerate_defiance",
                    consequence_type="positive_punishment",
                    intensity=0.7,
                    context_info={
                        "user_id": user_id,
                        "context_keys": ["conversation"]
                    }
                )
                reinforcement_results.append(punishment)
            
            # Get current mode distribution if available
            mode_distribution = {}
            if self.context.mode_manager and hasattr(self.context.mode_manager, 'context'):
                try:
                    mode_distribution = self.context.mode_manager.context.mode_distribution.dict()
                except:
                    pass
            
            # Collect results
            return {
                "input_text": text,
                "user_id": user_id,
                "detected_patterns": [p.dict() for p in detected_patterns],
                "behavior_evaluations": [eval.dict() for eval in behavior_evaluations],
                "recommended_behaviors": recommended_behaviors,
                "avoided_behaviors": avoided_behaviors,
                "reinforcement_results": reinforcement_results,
                "mode_distribution": mode_distribution
            }
    
    async def modify_response(self, response_text: str, input_processing_results: Dict[str, Any]) -> str:
        """
        Modify response based on conditioning results and mode distribution
        
        Args:
            response_text: Original response text
            input_processing_results: Results from process_input
            
        Returns:
            Modified response text
        """
        with trace(workflow_name="modify_conditioned_response"):
            # Check if mode distribution is available
            mode_distribution = input_processing_results.get("mode_distribution", {})
            
            # Fall back to current mode manager state if not in results
            if not mode_distribution and self.context.mode_manager and hasattr(self.context.mode_manager, 'context'):
                try:
                    mode_distribution = self.context.mode_manager.context.mode_distribution.dict()
                except:
                    pass
            
            # If we have a mode distribution, use blended modification
            if mode_distribution and any(weight >= 0.2 for weight in mode_distribution.values()):
                # Prepare the prompt for blended modification
                blended_prompt = f"""
                Modify the following response based on the blended mode distribution:
                
                ORIGINAL RESPONSE: {response_text}
                
                MODE DISTRIBUTION: {mode_distribution}
                
                DETECTED PATTERNS: {input_processing_results.get('detected_patterns', [])}
                
                RECOMMENDED BEHAVIORS: {input_processing_results.get('recommended_behaviors', [])}
                
                AVOIDED BEHAVIORS: {input_processing_results.get('avoided_behaviors', [])}
                
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
                
                DETECTED PATTERNS: {input_processing_results.get('detected_patterns', [])}
                
                RECOMMENDED BEHAVIORS: {input_processing_results.get('recommended_behaviors', [])}
                
                AVOIDED BEHAVIORS: {input_processing_results.get('avoided_behaviors', [])}
                
                REINFORCEMENT RESULTS: {input_processing_results.get('reinforcement_results', [])}
                
                Modify the response to align with recommended behaviors while avoiding
                behaviors that should be avoided. Ensure the modification is subtle but effective.
                """
                
                # Run the response modifier agent
                result = await Runner.run(self.response_modifier_agent, modification_prompt, context=self.context)
                modified_response = result.final_output
                
                return modified_response
    
    async def modify_blended_response(self, response_text: str, mode_distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Modify response based purely on mode distribution
        
        Args:
            response_text: Original response text
            mode_distribution: Current mode distribution
            
        Returns:
            Modified response information with details
        """
        with trace(workflow_name="modify_blended_response"):
            # Prepare the prompt
            blended_prompt = f"""
            Modify the following response based on the given mode distribution:
            
            ORIGINAL RESPONSE: {response_text}
            
            MODE DISTRIBUTION: {mode_distribution}
            
            Modify the response to proportionally reflect all active modes in the distribution,
            with higher-weighted modes having more influence on the final style and tone.
            Create a natural, coherent response that integrates elements from all active modes.
            Focus on maintaining the core message while adapting the style, tone, and phrasing.
            """
            
            # Run the blended modifier agent
            result = await Runner.run(self.blended_modifier_agent, blended_prompt, context=self.context)
            
            # Return the full result
            return result.final_output
