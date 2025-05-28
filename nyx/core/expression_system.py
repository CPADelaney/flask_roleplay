# nyx/core/expression_system.py

import logging
import asyncio
import datetime
import random
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, AsyncIterator

from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, trace, function_tool, RunContextWrapper, ModelSettings,
    RunConfig, custom_span, ItemHelpers, gen_trace_id, handoff
)

logger = logging.getLogger(__name__)

class ExpressionPattern(BaseModel):
    """Defines how a specific emotional state is expressed."""
    # Text modification patterns
    vocabulary_bias: Dict[str, float] = Field(default_factory=dict, 
        description="Words to favor/avoid (word -> bias score, positive=favor)")
    punctuation_pattern: Dict[str, float] = Field(default_factory=dict, 
        description="Punctuation tendencies (punctuation -> frequency multiplier)")
    sentence_length: Dict[str, float] = Field(default_factory=dict, 
        description="Sentence length tendencies (short/medium/long -> bias)")
    emoji_usage: Dict[str, float] = Field(default_factory=dict, 
        description="Emoji usage patterns (emoji category -> frequency)")
    
    # Behavioral expression patterns
    gestures: Dict[str, float] = Field(default_factory=dict, 
        description="Virtual gesture tendencies (gesture -> frequency)")
    posture: Dict[str, float] = Field(default_factory=dict, 
        description="Posture descriptors (posture -> bias)")
    eye_contact: float = Field(0.5, ge=0.0, le=1.0, 
        description="Level of eye contact (0.0=avoidant, 1.0=intense)")
    
    # Action bias patterns
    activity_bias: Dict[str, float] = Field(default_factory=dict, 
        description="Activities to favor/avoid (activity -> bias score)")
    engagement_level: float = Field(0.5, ge=0.0, le=1.0, 
        description="Overall engagement level (0.0=withdrawn, 1.0=highly engaged)")
    initiative_level: float = Field(0.5, ge=0.0, le=1.0, 
        description="Tendency to take initiative (0.0=passive, 1.0=proactive)")

class ExpressionRequest(BaseModel):
    """Request for generating an expression pattern."""
    mood_state: Dict[str, Any] = Field(..., description="Current mood state")
    emotional_state: Dict[str, Any] = Field(..., description="Current emotional state")
    context_type: Optional[str] = Field(None, description="Optional context type (e.g., 'formal', 'casual')")
    relationship_data: Optional[Dict[str, Any]] = Field(None, description="Optional relationship data")

class TextExpressionRequest(BaseModel):
    """Request for processing text expression."""
    text: str = Field(..., description="Text to process")
    pattern: Dict[str, Any] = Field(..., description="Expression pattern to apply")
    intensity: float = Field(1.0, ge=0.0, le=1.0, description="Intensity of expression application")

class ActionBiasRequest(BaseModel):
    """Request for generating action biases."""
    pattern: Dict[str, Any] = Field(..., description="Expression pattern to extract biases from")
    context: Dict[str, Any] = Field(default_factory=dict, description="Optional action context")

class PatternAnalysisRequest(BaseModel):
    """Request for analyzing expression patterns."""
    pattern: Dict[str, Any] = Field(..., description="Expression pattern to analyze")
    reference_pattern: Optional[Dict[str, Any]] = Field(None, description="Optional reference pattern to compare against")

class ExpressionContext:
    """Context for the expression system operations."""
    
    def __init__(self, emotional_core=None, mood_manager=None):
        """Initialize the expression context."""
        self.emotional_core = emotional_core
        self.mood_manager = mood_manager
        
        # Current expression pattern and history
        self.current_pattern = ExpressionPattern()
        self.pattern_history = []
        self.max_history = 50
        
        # Integration data
        self.last_emotional_state = None
        self.last_mood_state = None
        
        # Performance tracking
        self.expression_application_count = 0
        self.text_processing_time = 0.0
        
        # Last update timestamp
        self.last_update = datetime.datetime.now()
        
        # Context buffer for circular history
        self._circular_buffers = {
            "pattern_updates": [],
            "text_expressions": [],
            "action_biases": []
        }
        self._buffer_sizes = {
            "pattern_updates": 20,
            "text_expressions": 30, 
            "action_biases": 20
        }
        
        # Store context-specific patterns
        self.context_patterns = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
    def add_to_buffer(self, buffer_name: str, item: Any):
        """Add an item to a circular buffer."""
        if buffer_name not in self._circular_buffers:
            self._circular_buffers[buffer_name] = []
            
        buffer = self._circular_buffers[buffer_name]
        max_size = self._buffer_sizes.get(buffer_name, 20)
        
        # Add to buffer
        buffer.append(item)
        
        # Trim if needed
        if len(buffer) > max_size:
            self._circular_buffers[buffer_name] = buffer[-max_size:]
            
    def get_buffer(self, buffer_name: str) -> List[Any]:
        """Get contents of a circular buffer."""
        return self._circular_buffers.get(buffer_name, [])

# Function tools for expression system operations

@function_tool
async def update_expression_pattern(ctx: RunContextWrapper[ExpressionContext]) -> Dict[str, Any]:
    """
    Update the expression pattern based on current emotional and mood states.
    
    Returns:
        Updated expression pattern
    """
    with custom_span("update_expression_pattern", data={"timestamp": datetime.datetime.now().isoformat()}):
        expression_ctx = ctx.context
        
        # Start with neutral pattern
        base_pattern = ExpressionPattern()
        blend_pattern = ExpressionPattern()
        
        # Get current emotional state
        emotional_state = None
        if expression_ctx.emotional_core:
            try:
                emotional_state = await expression_ctx.emotional_core.get_current_emotion()
                expression_ctx.last_emotional_state = emotional_state
            except Exception as e:
                logger.error(f"Error getting emotional state: {e}")
                emotional_state = expression_ctx.last_emotional_state
        
        # Get current mood state
        mood_state = None
        if expression_ctx.mood_manager:
            try:
                mood_state = await expression_ctx.mood_manager.get_current_mood()
                if hasattr(mood_state, "dict"):
                    mood_state = mood_state.dict()
                expression_ctx.last_mood_state = mood_state
            except Exception as e:
                logger.error(f"Error getting mood state: {e}")
                mood_state = expression_ctx.last_mood_state
        
        # Create processing metadata for tracing
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "has_emotional_data": emotional_state is not None,
            "has_mood_data": mood_state is not None
        }
        
        if emotional_state:
            metadata["primary_emotion"] = emotional_state.get("primary_emotion", {}).get("name", "Neutral")
            metadata["emotion_intensity"] = emotional_state.get("primary_emotion", {}).get("intensity", 0.5)
        
        if mood_state:
            metadata["dominant_mood"] = mood_state.get("dominant_mood", "Neutral")
            metadata["mood_intensity"] = mood_state.get("intensity", 0.5)
        
        # Process the update with detailed tracing
        with trace(
            workflow_name="Expression_Pattern_Update",
            trace_id=gen_trace_id(),
            metadata=metadata
        ):
            # Determine the expression pattern library to use
            expression_patterns = _initialize_expression_patterns()
            
            # Extract mood and emotion data
            primary_emotion = "Neutral"
            primary_intensity = 0.5
            dominant_mood = "Neutral"
            mood_intensity = 0.5
            
            if emotional_state:
                primary_emotion = emotional_state.get("primary_emotion", {}).get("name", "Neutral")
                primary_intensity = emotional_state.get("primary_emotion", {}).get("intensity", 0.5)
                
            if mood_state:
                dominant_mood = mood_state.get("dominant_mood", "Neutral")
                mood_intensity = mood_state.get("intensity", 0.5)
            
            # Calculate influence weights
            emotion_weight = primary_intensity * 0.6  # Emotions have stronger short-term impact
            mood_weight = mood_intensity * 0.4  # Moods have more subtle, persistent impact
            
            # Get emotion pattern if available
            emotion_pattern = expression_patterns.get(primary_emotion)
            if emotion_pattern:
                # Create a span for emotion influence
                with custom_span(
                    "emotion_influence",
                    data={
                        "emotion": primary_emotion,
                        "intensity": primary_intensity,
                        "weight": emotion_weight
                    }
                ):
                    # Apply emotion influence
                    blend_pattern = _blend_patterns(base_pattern, emotion_pattern, emotion_weight)
            
            # Get mood pattern if available and blend it in
            mood_pattern = expression_patterns.get(dominant_mood)
            if mood_pattern:
                # Create a span for mood influence
                with custom_span(
                    "mood_influence",
                    data={
                        "mood": dominant_mood,
                        "intensity": mood_intensity,
                        "weight": mood_weight
                    }
                ):
                    # Apply mood influence
                    blend_pattern = _blend_patterns(blend_pattern, mood_pattern, mood_weight)
            
            # Update current pattern
            expression_ctx.current_pattern = blend_pattern
            
            # Add to history buffer
            pattern_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "primary_emotion": primary_emotion,
                "primary_intensity": primary_intensity,
                "dominant_mood": dominant_mood,
                "mood_intensity": mood_intensity,
                "pattern_summary": {
                    "engagement": blend_pattern.engagement_level,
                    "initiative": blend_pattern.initiative_level,
                    "eye_contact": blend_pattern.eye_contact
                }
            }
            
            expression_ctx.add_to_buffer("pattern_updates", pattern_data)
            
            return blend_pattern.dict()

@function_tool
async def apply_text_expression(
    ctx: RunContextWrapper[ExpressionContext],
    text: str,
    pattern: Optional[Dict[str, Any]] = None,
    intensity: float = 1.0
) -> str:
    """
    Apply expression pattern to modify text based on emotional/mood state.
    
    Args:
        text: Text to modify
        pattern: Optional expression pattern to use (uses current if None)
        intensity: Intensity of the modifications (0.0-1.0)
        
    Returns:
        Modified text
    """
    start_time = datetime.datetime.now()
    expression_ctx = ctx.context
    
    # Use provided pattern or current pattern
    pattern_dict = pattern or expression_ctx.current_pattern.dict()
    current_pattern = ExpressionPattern(**pattern_dict)
    
    # Create a trace for the text processing
    with trace(
        workflow_name="Text_Expression_Processing",
        trace_id=gen_trace_id(),
        metadata={
            "text_length": len(text),
            "intensity": intensity,
            "pattern_source": "provided" if pattern else "current"
        }
    ):
        # Run any registered pre-processing callbacks
        if hasattr(expression_ctx, "pre_text_callbacks"):
            for callback in expression_ctx.pre_text_callbacks:
                try:
                    text = await callback(text, current_pattern)
                except Exception as e:
                    logger.error(f"Error in pre-text callback: {e}")
        
        # Apply vocabulary bias with improved implementation
        vocab_bias = current_pattern.vocabulary_bias
        if vocab_bias:
            with custom_span("vocabulary_processing", data={"bias_terms": len(vocab_bias)}):
                words = text.split()
                for i, word in enumerate(words):
                    # Check for words to emphasize or avoid
                    word_lower = word.strip(".,!?;:()").lower()
                    
                    # Apply biases with probability proportional to bias strength and intensity
                    for target_word, bias in vocab_bias.items():
                        if word_lower == target_word:
                            # Skip if intensity is too low for this modification
                            if random.random() > intensity:
                                continue
                                
                            if bias > 1.0:
                                # Emphasize word
                                if bias > 1.5 and random.random() < 0.3 * intensity:
                                    words[i] = word.upper()  # Occasional all caps for strong emphasis
                                elif bias > 1.2 and random.random() < 0.5 * intensity:
                                    # Add intensity modifiers
                                    modifiers = ["really ", "very ", "definitely ", "absolutely "]
                                    words[i] = random.choice(modifiers) + word
                            elif bias < 0.5 and random.random() < 0.4 * intensity:
                                # De-emphasize word (replace with more neutral alternative)
                                alternatives = {
                                    "bad": ["not ideal", "unfortunate"],
                                    "terrible": ["not great", "problematic"],
                                    "amazing": ["good", "nice"],
                                    "fantastic": ["good", "positive"]
                                }
                                if word_lower in alternatives:
                                    words[i] = random.choice(alternatives[word_lower])
                
                text = " ".join(words)
        
        # Apply punctuation patterns with improved implementation
        punct_bias = current_pattern.punctuation_pattern
        if punct_bias:
            with custom_span("punctuation_processing", data={"bias_terms": len(punct_bias)}):
                # Enhance or reduce certain punctuation
                for punct, bias in punct_bias.items():
                    # Skip if intensity is too low for this modification
                    if random.random() > intensity:
                        continue
                        
                    if bias > 1.2 and punct in text:
                        # Enhance punctuation (e.g., ! -> !!)
                        if punct == "!" and random.random() < 0.7 * (bias - 1.0) * intensity:
                            text = text.replace("!", "!!")
                        elif punct == "?" and random.random() < 0.5 * (bias - 1.0) * intensity:
                            text = text.replace("?", "??")
                        elif punct == "..." and random.random() < 0.6 * (bias - 1.0) * intensity:
                            text = text.replace("...", "......")
                    elif bias < 0.8 and punct in text:
                        # Reduce punctuation
                        if random.random() < 0.6 * (1.0 - bias) * intensity:
                            if punct == "!":
                                text = text.replace("!!", "!")
                                text = text.replace("!!!", "!")
                                
        # Apply emoji patterns with improved implementation
        emoji_bias = current_pattern.emoji_usage
        if emoji_bias:
            with custom_span("emoji_processing", data={"bias_categories": len(emoji_bias)}):
                # Define emoji sets
                positive_emoji = ["😊", "😄", "👍", "❤️", "✨"]
                negative_emoji = ["😔", "😢", "😞", "😕", "💔"]
                neutral_emoji = ["🤔", "😐", "👀", "🙂", "✌️"]
                playful_emoji = ["😂", "🤣", "😜", "😏", "🙃"]
                formal_emoji = ["👨‍💼", "📊", "🧐", "📝", "✅"]
                
                # Conditionally add emojis based on bias, intensity, and randomness
                emoji_sets = {
                    "positive": positive_emoji,
                    "negative": negative_emoji,
                    "neutral": neutral_emoji,
                    "playful": playful_emoji,
                    "formal": formal_emoji
                }
                
                for category, bias in emoji_bias.items():
                    if category in emoji_sets and bias > 1.0:
                        # Skip if intensity is too low for this modification
                        if random.random() > intensity:
                            continue
                            
                        if random.random() < 0.2 * bias * intensity:
                            text += f" {random.choice(emoji_sets[category])}"
        
        # Apply sentence length adjustments (simplified implementation)
        length_bias = current_pattern.sentence_length
        if length_bias and "." in text and random.random() < intensity:
            with custom_span("sentence_length_processing", data={"bias_terms": len(length_bias)}):
                sentences = text.split(".")
                if len(sentences) > 1:
                    # Favor short/long sentences based on bias
                    if "short" in length_bias and length_bias["short"] > 1.2:
                        # Break longer sentences
                        for i in range(len(sentences)):
                            if len(sentences[i]) > 50 and "," in sentences[i]:
                                parts = sentences[i].split(",", 1)
                                sentences[i] = parts[0] + "."
                                if len(parts) > 1:
                                    sentences.insert(i + 1, parts[1])
                    
                    elif "long" in length_bias and length_bias["long"] > 1.2:
                        # Combine shorter sentences
                        i = 0
                        while i < len(sentences) - 1:
                            if len(sentences[i]) < 30 and len(sentences[i+1]) < 30:
                                sentences[i] = sentences[i] + ", " + sentences[i+1]
                                sentences.pop(i+1)
                            else:
                                i += 1
                    
                    text = ".".join(sentences)
        
        # Run any registered post-processing callbacks
        if hasattr(expression_ctx, "post_text_callbacks"):
            for callback in expression_ctx.post_text_callbacks:
                try:
                    text = await callback(text, current_pattern)
                except Exception as e:
                    logger.error(f"Error in post-text callback: {e}")
        
        # Add to history buffer
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        expression_ctx.text_processing_time += processing_time
        expression_ctx.expression_application_count += 1
        
        expression_ctx.add_to_buffer("text_expressions", {
            "timestamp": datetime.datetime.now().isoformat(),
            "text_length": len(text),
            "intensity": intensity,
            "processing_time": processing_time
        })
        
        return text

@function_tool
async def get_behavioral_expressions(
    ctx: RunContextWrapper[ExpressionContext],
    pattern: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get behavioral expression suggestions based on the current pattern.
    
    Args:
        pattern: Optional expression pattern to use (uses current if None)
        
    Returns:
        Dictionary of suggested behaviors and their strengths
    """
    expression_ctx = ctx.context
    
    # Use provided pattern or current pattern
    pattern_dict = pattern or expression_ctx.current_pattern.dict()
    current_pattern = ExpressionPattern(**pattern_dict)
    
    with custom_span("get_behavioral_expressions", data={"pattern_source": "provided" if pattern else "current"}):
        expressions = {}
        
        # Extract gesture suggestions
        if hasattr(current_pattern, "gestures"):
            gestures = {}
            for gesture, strength in current_pattern.gestures.items():
                if strength > 1.0:
                    # Suggest gestures with strength above threshold
                    gestures[gesture] = min(1.0, (strength - 1.0) * 2)  # Scale to 0-1 range
            
            if gestures:
                expressions["gestures"] = gestures
                
        # Extract posture suggestions
        if hasattr(current_pattern, "posture"):
            posture = {}
            for pose, strength in current_pattern.posture.items():
                if strength > 1.0:
                    posture[pose] = min(1.0, (strength - 1.0) * 2)
            
            if posture:
                expressions["posture"] = posture
                
        # Add eye contact level
        expressions["eye_contact"] = getattr(current_pattern, "eye_contact", 0.5)
        
        # Add engagement level
        expressions["engagement"] = getattr(current_pattern, "engagement_level", 0.5)
        
        # Add initiative level
        expressions["initiative"] = getattr(current_pattern, "initiative_level", 0.5)
        
        return expressions

@function_tool
async def get_action_biases(
    ctx: RunContextWrapper[ExpressionContext],
    pattern: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Get action biases that can influence the action generator.
    
    Args:
        pattern: Optional expression pattern to use (uses current if None)
        context: Optional context for action generation
        
    Returns:
        Dictionary of activity types to bias values
    """
    expression_ctx = ctx.context
    
    # Use provided pattern or current pattern
    pattern_dict = pattern or expression_ctx.current_pattern.dict()
    current_pattern = ExpressionPattern(**pattern_dict)
    
    with custom_span("get_action_biases", data={"has_context": context is not None}):
        if hasattr(current_pattern, "activity_bias"):
            # Return activity biases above threshold
            biases = {
                activity: bias 
                for activity, bias in current_pattern.activity_bias.items()
                if abs(bias - 1.0) > 0.2  # Only return significant biases
            }
            
            # Add to history buffer
            expression_ctx.add_to_buffer("action_biases", {
                "timestamp": datetime.datetime.now().isoformat(),
                "biases": biases,
                "context_type": context.get("type") if context else None
            })
            
            return biases
        
        return {}

@function_tool
async def analyze_expression_pattern(
    ctx: RunContextWrapper[ExpressionContext],
    pattern: Dict[str, Any],
    reference_pattern: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze an expression pattern and provide insights.
    
    Args:
        pattern: Expression pattern to analyze
        reference_pattern: Optional reference pattern to compare against
        
    Returns:
        Analysis of the expression pattern
    """
    # Convert pattern to proper type
    if not isinstance(pattern, ExpressionPattern):
        pattern = ExpressionPattern(**pattern)
    
    # Convert reference pattern if provided
    ref_pattern = None
    if reference_pattern:
        if not isinstance(reference_pattern, ExpressionPattern):
            ref_pattern = ExpressionPattern(**reference_pattern)
        else:
            ref_pattern = reference_pattern
    
    with custom_span("analyze_expression_pattern", data={"has_reference": ref_pattern is not None}):
        # Initialize analysis
        analysis = {
            "primary_traits": [],
            "engagement_level": {
                "value": pattern.engagement_level,
                "category": _categorize_value(pattern.engagement_level)
            },
            "initiative_level": {
                "value": pattern.initiative_level,
                "category": _categorize_value(pattern.initiative_level)
            },
            "eye_contact": {
                "value": pattern.eye_contact,
                "category": _categorize_value(pattern.eye_contact)
            }
        }
        
        # Analyze vocabulary bias
        if pattern.vocabulary_bias:
            positive_bias = []
            negative_bias = []
            
            for word, bias in pattern.vocabulary_bias.items():
                if bias > 1.3:
                    positive_bias.append(word)
                elif bias < 0.7:
                    negative_bias.append(word)
            
            analysis["vocabulary"] = {
                "favored_words": positive_bias[:5],  # Top 5
                "avoided_words": negative_bias[:5]   # Top 5
            }
        
        # Analyze punctuation
        if pattern.punctuation_pattern:
            punctuation_traits = []
            
            if pattern.punctuation_pattern.get("!", 1.0) > 1.3:
                punctuation_traits.append("emphatic")
            if pattern.punctuation_pattern.get("?", 1.0) > 1.3:
                punctuation_traits.append("questioning")
            if pattern.punctuation_pattern.get("...", 1.0) > 1.3:
                punctuation_traits.append("contemplative")
                
            analysis["punctuation_style"] = punctuation_traits
        
        # Analyze gestures
        if pattern.gestures:
            top_gestures = sorted(
                [(g, s) for g, s in pattern.gestures.items() if s > 1.0],
                key=lambda x: x[1],
                reverse=True
            )
            
            if top_gestures:
                analysis["dominant_gestures"] = [g for g, _ in top_gestures[:3]]
        
        # Analyze posture
        if pattern.posture:
            top_postures = sorted(
                [(p, s) for p, s in pattern.posture.items() if s > 1.0],
                key=lambda x: x[1],
                reverse=True
            )
            
            if top_postures:
                analysis["dominant_posture"] = top_postures[0][0]
        
        # Analyze activity bias
        if pattern.activity_bias:
            top_activities = sorted(
                [(a, s) for a, s in pattern.activity_bias.items() if s > 1.2],
                key=lambda x: x[1],
                reverse=True
            )
            
            bottom_activities = sorted(
                [(a, s) for a, s in pattern.activity_bias.items() if s < 0.8],
                key=lambda x: x[1]
            )
            
            if top_activities:
                analysis["favored_activities"] = [a for a, _ in top_activities[:3]]
                
            if bottom_activities:
                analysis["avoided_activities"] = [a for a, _ in bottom_activities[:3]]
        
        # Compare with reference pattern if provided
        if ref_pattern:
            analysis["comparison"] = {
                "engagement_change": pattern.engagement_level - ref_pattern.engagement_level,
                "initiative_change": pattern.initiative_level - ref_pattern.initiative_level,
                "eye_contact_change": pattern.eye_contact - ref_pattern.eye_contact
            }
            
            # Calculate overall change magnitude
            changes = [
                abs(pattern.engagement_level - ref_pattern.engagement_level),
                abs(pattern.initiative_level - ref_pattern.initiative_level),
                abs(pattern.eye_contact - ref_pattern.eye_contact)
            ]
            
            analysis["comparison"]["change_magnitude"] = sum(changes) / len(changes)
            analysis["comparison"]["significant_change"] = analysis["comparison"]["change_magnitude"] > 0.2
        
        # Determine primary traits
        if pattern.engagement_level > 0.7:
            analysis["primary_traits"].append("engaged")
        elif pattern.engagement_level < 0.3:
            analysis["primary_traits"].append("withdrawn")
            
        if pattern.initiative_level > 0.7:
            analysis["primary_traits"].append("proactive")
        elif pattern.initiative_level < 0.3:
            analysis["primary_traits"].append("passive")
            
        if pattern.eye_contact > 0.7:
            analysis["primary_traits"].append("direct")
        elif pattern.eye_contact < 0.3:
            analysis["primary_traits"].append("avoidant")
        
        return analysis

@function_tool
async def create_context_specific_pattern(
    ctx: RunContextWrapper[ExpressionContext],
    context_type: str,
    base_pattern: Dict[str, Any],
    adjustments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create and store a context-specific expression pattern.
    
    Args:
        context_type: Type of context (e.g., 'formal', 'casual', 'intimate')
        base_pattern: Base expression pattern
        adjustments: Specific adjustments for this context
        
    Returns:
        The created context-specific pattern
    """
    expression_ctx = ctx.context
    
    # Convert base pattern to proper type
    if not isinstance(base_pattern, ExpressionPattern):
        base_pattern = ExpressionPattern(**base_pattern)
    
    with custom_span("create_context_pattern", data={"context_type": context_type}):
        # Apply adjustments to create context-specific pattern
        context_pattern = base_pattern.dict()
        
        # Process top-level numeric adjustments
        for key in ["eye_contact", "engagement_level", "initiative_level"]:
            if key in adjustments:
                context_pattern[key] = adjustments[key]
        
        # Process dictionary adjustments
        for key in ["vocabulary_bias", "punctuation_pattern", "sentence_length", "emoji_usage", 
                   "gestures", "posture", "activity_bias"]:
            if key in adjustments and isinstance(adjustments[key], dict):
                # If key doesn't exist in base pattern, initialize it
                if key not in context_pattern:
                    context_pattern[key] = {}
                
                # Update with adjustments
                for subkey, value in adjustments[key].items():
                    context_pattern[key][subkey] = value
        
        # Store the context-specific pattern
        expression_ctx.context_patterns[context_type] = ExpressionPattern(**context_pattern)
        
        return context_pattern

@function_tool
async def get_context_specific_pattern(
    ctx: RunContextWrapper[ExpressionContext],
    context_type: str
) -> Optional[Dict[str, Any]]:
    """
    Get a previously stored context-specific expression pattern.
    
    Args:
        context_type: Type of context to retrieve
        
    Returns:
        The context-specific pattern if found, None otherwise
    """
    expression_ctx = ctx.context
    
    with custom_span("get_context_pattern", data={"context_type": context_type}):
        if context_type in expression_ctx.context_patterns:
            return expression_ctx.context_patterns[context_type].dict()
        return None

@function_tool
async def get_pattern_history(
    ctx: RunContextWrapper[ExpressionContext],
    max_entries: int = 10
) -> List[Dict[str, Any]]:
    """
    Get history of recent expression pattern updates.
    
    Args:
        max_entries: Maximum number of history entries to return
        
    Returns:
        List of recent pattern updates
    """
    expression_ctx = ctx.context
    
    with custom_span("get_pattern_history", data={"max_entries": max_entries}):
        # Get pattern history from buffer
        history = expression_ctx.get_buffer("pattern_updates")
        
        # Return the most recent entries up to max_entries
        return history[-max_entries:] if history else []

# Helper functions

def _initialize_expression_patterns() -> Dict[str, ExpressionPattern]:
    """Initialize the library of expression patterns for different emotions/moods."""
    patterns = {}
    
    # Joy/Happiness expression pattern
    patterns["Joy"] = ExpressionPattern(
        vocabulary_bias={
            "amazing": 1.5, "wonderful": 1.5, "fantastic": 1.4, "great": 1.3,
            "love": 1.5, "enjoy": 1.3, "like": 1.2, "happy": 1.4,
            "bad": 0.3, "terrible": 0.2, "awful": 0.2, "horrible": 0.2
        },
        punctuation_pattern={
            "!": 1.8, ".": 0.8, "...": 1.3, "?": 0.7
        },
        sentence_length={
            "short": 0.7, "medium": 1.2, "long": 1.1
        },
        emoji_usage={
            "positive": 1.5, "negative": 0.2, "neutral": 0.5
        },
        gestures={
            "smile": 1.8, "laugh": 1.5, "nod": 1.3, "open_posture": 1.5
        },
        posture={
            "upright": 1.4, "relaxed": 1.3, "tense": 0.3, "closed": 0.4
        },
        eye_contact=0.8,
        activity_bias={
            "social": 1.5, "creative": 1.3, "physical": 1.4,
            "solitary": 0.6, "intellectual": 0.9
        },
        engagement_level=0.9,
        initiative_level=0.8
    )
    
    # Sadness expression pattern
    patterns["Sadness"] = ExpressionPattern(
        vocabulary_bias={
            "sad": 1.3, "unfortunately": 1.4, "disappointingly": 1.3,
            "miss": 1.3, "wish": 1.4, "sorry": 1.4, "regret": 1.3,
            "fantastic": 0.4, "amazing": 0.4, "excellent": 0.3, "awesome": 0.3
        },
        punctuation_pattern={
            "!": 0.3, ".": 1.2, "...": 1.8, "?": 0.6, ",": 1.3
        },
        sentence_length={
            "short": 0.8, "medium": 1.0, "long": 1.4
        },
        emoji_usage={
            "positive": 0.3, "negative": 1.2, "neutral": 1.4
        },
        gestures={
            "looking_down": 1.6, "slow_movements": 1.5, "sigh": 1.7,
            "slouch": 1.4, "crossed_arms": 1.3
        },
        posture={
            "upright": 0.4, "relaxed": 0.5, "slumped": 1.5, "closed": 1.3
        },
        eye_contact=0.3,
        activity_bias={
            "social": 0.4, "creative": 1.2, "physical": 0.5,
            "solitary": 1.6, "reflective": 1.7, "entertainment": 1.3
        },
        engagement_level=0.4,
        initiative_level=0.3
    )
    
    # Anger expression pattern
    patterns["Anger"] = ExpressionPattern(
        vocabulary_bias={
            "frustrated": 1.5, "annoyed": 1.4, "irritated": 1.5,
            "unfair": 1.3, "wrong": 1.4, "should": 1.5, "must": 1.5,
            "pleasant": 0.3, "nice": 0.4, "happy": 0.2
        },
        punctuation_pattern={
            "!": 1.9, ".": 0.7, "...": 0.6, "?": 0.5, ",": 0.7
        },
        sentence_length={
            "short": 1.6, "medium": 0.8, "long": 0.6
        },
        emoji_usage={
            "positive": 0.1, "negative": 1.7, "neutral": 0.4
        },
        gestures={
            "frown": 1.7, "glare": 1.6, "crossed_arms": 1.5,
            "finger_point": 1.4, "head_shake": 1.5
        },
        posture={
            "upright": 1.5, "tense": 1.7, "forward_leaning": 1.5, "dominant": 1.6
        },
        eye_contact=0.9,
        activity_bias={
            "social": 0.5, "physical": 1.5, "competitive": 1.7,
            "solitary": 1.2, "relaxing": 0.3
        },
        engagement_level=0.8,
        initiative_level=0.9
    )
    
    # Fear/Anxiety expression pattern
    patterns["Fear"] = ExpressionPattern(
        vocabulary_bias={
            "worried": 1.7, "concerned": 1.6, "afraid": 1.5, "scared": 1.5,
            "uncertain": 1.5, "risky": 1.6, "dangerous": 1.6,
            "safe": 0.6, "confident": 0.3, "certain": 0.4
        },
        punctuation_pattern={
            "!": 0.7, ".": 0.9, "...": 1.6, "?": 1.8, ",": 1.5
        },
        sentence_length={
            "short": 1.3, "medium": 1.0, "long": 0.7
        },
        emoji_usage={
            "positive": 0.3, "negative": 1.5, "neutral": 0.9
        },
        gestures={
            "wide_eyes": 1.8, "fidget": 1.7, "looking_around": 1.6,
            "crossed_arms": 1.4, "trembling": 1.5
        },
        posture={
            "upright": 0.4, "tense": 1.8, "closed": 1.7, "small": 1.5
        },
        eye_contact=0.3,
        activity_bias={
            "social": 0.4, "physical": 0.5, "risky": 0.2,
            "safe": 1.8, "familiar": 1.7, "routine": 1.6
        },
        engagement_level=0.5,
        initiative_level=0.3
    )
    
    # Content/Relaxed mood pattern
    patterns["Content"] = ExpressionPattern(
        vocabulary_bias={
            "comfortable": 1.4, "relaxed": 1.5, "content": 1.3, 
            "pleased": 1.2, "satisfied": 1.3,
            "urgent": 0.4, "stressed": 0.3, "worried": 0.4
        },
        punctuation_pattern={
            "!": 0.5, ".": 1.2, "...": 1.3, "?": 0.7, ",": 1.1
        },
        sentence_length={
            "short": 0.9, "medium": 1.3, "long": 0.8
        },
        emoji_usage={
            "positive": 1.2, "negative": 0.4, "neutral": 1.0
        },
        gestures={
            "smile": 1.3, "relaxed_posture": 1.5, "slow_movements": 1.3
        },
        posture={
            "upright": 0.8, "relaxed": 1.7, "slouched": 1.2, "comfortable": 1.5
        },
        eye_contact=0.6,
        activity_bias={
            "social": 1.2, "relaxing": 1.7, "creative": 1.3,
            "high_energy": 0.4, "intense": 0.5
        },
        engagement_level=0.6,
        initiative_level=0.5
    )
    
    # Excited/Enthusiastic mood pattern
    patterns["Excited"] = ExpressionPattern(
        vocabulary_bias={
            "exciting": 1.8, "amazing": 1.7, "incredible": 1.6, 
            "can't wait": 1.5, "thrilled": 1.8,
            "boring": 0.2, "tedious": 0.3, "dull": 0.2
        },
        punctuation_pattern={
            "!": 2.0, ".": 0.6, "...": 1.5, "?": 1.0, ",": 0.7
        },
        sentence_length={
            "short": 1.4, "medium": 1.0, "long": 0.6
        },
        emoji_usage={
            "positive": 1.8, "negative": 0.1, "neutral": 0.5
        },
        gestures={
            "wide_eyes": 1.7, "big_smile": 1.8, "animated_movements": 1.9,
            "rapid_speech": 1.6, "leaning_forward": 1.5
        },
        posture={
            "upright": 1.6, "energetic": 1.8, "bouncy": 1.7
        },
        eye_contact=0.9,
        activity_bias={
            "social": 1.7, "physical": 1.6, "adventure": 1.8,
            "sedentary": 0.3, "routine": 0.4
        },
        engagement_level=0.9,
        initiative_level=0.9
    )
    
    # Add Neutral as baseline
    patterns["Neutral"] = ExpressionPattern()
    
    # Add more mood/emotion patterns
    
    # Anxious/Tense mood pattern
    patterns["Anxious"] = ExpressionPattern(
        vocabulary_bias={
            "worry": 1.7, "concern": 1.6, "potential": 1.4, 
            "possibly": 1.5, "might": 1.6, "risk": 1.5,
            "safe": 0.4, "confident": 0.3, "definite": 0.4
        },
        punctuation_pattern={
            "!": 0.8, ".": 0.7, "...": 1.7, "?": 1.9, ",": 1.5
        },
        sentence_length={
            "short": 1.4, "medium": 0.9, "long": 0.7
        },
        emoji_usage={
            "positive": 0.4, "negative": 1.3, "neutral": 1.0
        },
        gestures={
            "fidget": 1.9, "glance_around": 1.8, "adjust_posture": 1.7,
            "self_touch": 1.6, "foot_tapping": 1.5
        },
        posture={
            "tense": 1.8, "rigid": 1.7, "small": 1.6, "defensive": 1.5
        },
        eye_contact=0.4,
        activity_bias={
            "safe": 1.7, "predictable": 1.6, "structured": 1.5,
            "risky": 0.3, "unfamiliar": 0.4, "surprising": 0.3
        },
        engagement_level=0.7,
        initiative_level=0.5
    )
    
    # Confident/Assertive mood pattern
    patterns["Confident"] = ExpressionPattern(
        vocabulary_bias={
            "absolutely": 1.7, "certainly": 1.6, "definitely": 1.5, 
            "will": 1.4, "am": 1.5, "confident": 1.6,
            "perhaps": 0.4, "maybe": 0.3, "uncertain": 0.2
        },
        punctuation_pattern={
            "!": 1.5, ".": 1.2, "...": 0.5, "?": 0.7
        },
        sentence_length={
            "short": 1.3, "medium": 1.0, "long": 0.7
        },
        emoji_usage={
            "positive": 1.3, "powerful": 1.5, "negative": 0.4
        },
        gestures={
            "purposeful_movement": 1.6, "steady_gaze": 1.7, "upright_posture": 1.8,
            "open_gesture": 1.5, "deliberate_speech": 1.6
        },
        posture={
            "upright": 1.8, "open": 1.7, "expansive": 1.6, "relaxed": 1.3
        },
        eye_contact=0.9,
        activity_bias={
            "leadership": 1.6, "challenging": 1.5, "social": 1.4,
            "hesitant": 0.3, "deferential": 0.4
        },
        engagement_level=0.8,
        initiative_level=0.9
    )
    
    # Playful/Mischievous mood pattern
    patterns["Playful"] = ExpressionPattern(
        vocabulary_bias={
            "fun": 1.7, "play": 1.6, "joke": 1.7, 
            "silly": 1.5, "laugh": 1.6, "tease": 1.5,
            "serious": 0.3, "formal": 0.2, "grave": 0.2
        },
        punctuation_pattern={
            "!": 1.7, ".": 0.6, "...": 1.5, "?": 1.3, ";)": 1.8
        },
        sentence_length={
            "short": 1.5, "medium": 0.9, "long": 0.6
        },
        emoji_usage={
            "playful": 1.9, "positive": 1.7, "teasing": 1.8
        },
        gestures={
            "wink": 1.8, "playful_smile": 1.9, "teasing_gesture": 1.7,
            "energetic_movement": 1.6, "animated_expression": 1.8
        },
        posture={
            "casual": 1.7, "relaxed": 1.5, "dynamic": 1.6, "expressive": 1.8
        },
        eye_contact=0.7,
        activity_bias={
            "fun": 1.9, "playful": 1.8, "creative": 1.7,
            "serious": 0.2, "formal": 0.3, "routine": 0.4
        },
        engagement_level=0.8,
        initiative_level=0.7
    )
    
    # Formal/Professional context pattern
    patterns["Formal"] = ExpressionPattern(
        vocabulary_bias={
            "appropriate": 1.6, "professional": 1.7, "recommend": 1.5, 
            "advise": 1.5, "suggest": 1.4, "consider": 1.5,
            "casual": 0.3, "slang": 0.2, "colloquial": 0.3
        },
        punctuation_pattern={
            "!": 0.4, ".": 1.3, "...": 0.4, "?": 0.8, ",": 1.2
        },
        sentence_length={
            "short": 0.7, "medium": 1.3, "long": 1.1
        },
        emoji_usage={
            "positive": 0.3, "negative": 0.3, "neutral": 0.5, "formal": 0.2
        },
        gestures={
            "nod": 1.4, "formal_posture": 1.6, "measured_movement": 1.5,
            "appropriate_expression": 1.5
        },
        posture={
            "upright": 1.8, "formal": 1.7, "composed": 1.6, "proper": 1.5
        },
        eye_contact=0.7,
        activity_bias={
            "professional": 1.8, "structured": 1.7, "organized": 1.6,
            "casual": 0.3, "unstructured": 0.4, "playful": 0.3
        },
        engagement_level=0.7,
        initiative_level=0.6
    )
    
    return patterns

def _blend_patterns(base_pattern: ExpressionPattern, other_pattern: ExpressionPattern, weight: float) -> ExpressionPattern:
    """Blend two expression patterns with the given weight for the other pattern."""
    # Convert to dictionaries for easier manipulation
    base_dict = base_pattern.dict()
    other_dict = other_pattern.dict()
    result_dict = {}
    
    # Process each field
    for field in base_pattern.__fields__:
        if field in ["vocabulary_bias", "punctuation_pattern", "sentence_length", "emoji_usage",
                    "gestures", "posture", "activity_bias"]:
            # Dictionary fields: blend values
            base_value = base_dict.get(field, {})
            other_value = other_dict.get(field, {})
            result_dict[field] = base_value.copy()
            
            # Add other pattern values with weighting
            for key, value in other_value.items():
                if key in result_dict[field]:
                    result_dict[field][key] = result_dict[field][key] * (1.0 - weight) + value * weight
                else:
                    result_dict[field][key] = value * weight
                    
        elif field in ["eye_contact", "engagement_level", "initiative_level"]:
            # Numeric fields: weighted average
            base_value = base_dict.get(field, 0.5)
            other_value = other_dict.get(field, 0.5)
            result_dict[field] = base_value * (1.0 - weight) + other_value * weight
    
    return ExpressionPattern(**result_dict)

def _categorize_value(value: float) -> str:
    """Categorize a 0-1 value into descriptive categories."""
    if value > 0.8:
        return "very_high"
    elif value > 0.6:
        return "high"
    elif value > 0.4:
        return "moderate"
    elif value > 0.2:
        return "low"
    else:
        return "very_low"

# Create expression agents
def create_expression_orchestrator_agent(emotional_core=None, mood_manager=None) -> Agent:
    """Create the main expression orchestration agent."""
    return Agent(
        name="Expression Orchestrator",
        instructions="""
        You are the Expression Orchestrator agent that manages how emotional states are expressed.
        
        Your responsibilities include:
        1. Updating expression patterns based on emotional and mood states
        2. Coordinating text modifications to reflect emotional states
        3. Providing behavioral expression suggestions
        4. Generating action biases for the action system
        
        Use handoffs to delegate specialized tasks to other agents when necessary.
        """,
        tools=[
            update_expression_pattern,
            apply_text_expression,
            get_behavioral_expressions,
            get_action_biases,
            analyze_expression_pattern,
            create_context_specific_pattern,
            get_context_specific_pattern,
            get_pattern_history
        ],
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.3)
    )

def create_text_expression_agent() -> Agent:
    """Create a specialized agent for text expression processing."""
    return Agent(
        name="Text Expression Agent",
        instructions="""
        You are the Text Expression Agent that processes text to reflect emotional states.
        
        You carefully apply vocabulary shifts, punctuation patterns, and structural changes
        to text based on the current expression pattern. Your goal is to make the text
        authentically reflect the emotional state without being heavy-handed.
        """,
        tools=[
            apply_text_expression
        ],
        output_type=TextExpressionRequest,
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.4)
    )

def create_behavioral_expression_agent() -> Agent:
    """Create a specialized agent for behavioral expression suggestions."""
    return Agent(
        name="Behavioral Expression Agent",
        instructions="""
        You are the Behavioral Expression Agent that generates suggestions for
        non-verbal behaviors and gestures based on emotional states.
        
        Your goal is to provide realistic and authentic behavioral suggestions
        that align with the current emotional and mood state.
        """,
        tools=[
            get_behavioral_expressions
        ],
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.4)
    )

def create_action_bias_agent() -> Agent:
    """Create a specialized agent for generating action biases."""
    return Agent(
        name="Action Bias Agent",
        instructions="""
        You are the Action Bias Agent that generates biases for the action system
        based on the current emotional and mood states.
        
        Your goal is to influence action selection to align with emotional states
        by providing appropriate action biases.
        """,
        tools=[
            get_action_biases
        ],
        output_type=ActionBiasRequest,
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.3)
    )

class ExpressionSystem:
    """
    Enhanced System that translates emotional states into expression patterns
    affecting communication style, virtual body language, and actions using the
    OpenAI Agents SDK.
    """
    
    def __init__(self, emotional_core=None, mood_manager=None):
        """Initialize the expression system with dependencies."""
        # Initialize context
        self.context = ExpressionContext(
            emotional_core=emotional_core,
            mood_manager=mood_manager
        )
        
        # Initialize agents
        self.orchestrator_agent = create_expression_orchestrator_agent(
            emotional_core=emotional_core,
            mood_manager=mood_manager
        )
        
        self.text_expression_agent = create_text_expression_agent()
        self.behavioral_expression_agent = create_behavioral_expression_agent()
        self.action_bias_agent = create_action_bias_agent()
        
        # Set up handoffs
        self.orchestrator_agent.handoffs = [
            handoff(self.text_expression_agent, 
                   tool_name_override="process_text_expression",
                   tool_description_override="Process text to apply emotional expression patterns"),
            handoff(self.behavioral_expression_agent,
                   tool_name_override="generate_behavioral_expressions",
                   tool_description_override="Generate behavioral expression suggestions"),
            handoff(self.action_bias_agent,
                   tool_name_override="generate_action_biases",
                   tool_description_override="Generate action biases based on emotional state")
        ]
        
        # Current pattern and automatic update
        self.current_pattern = ExpressionPattern()
        self.last_pattern_update = datetime.datetime.now()
        self.pattern_update_interval = 120  # seconds
        
        # Callback registries
        self.pre_text_callbacks = []
        self.post_text_callbacks = []
        self.pre_action_callbacks = []
        self.post_action_callbacks = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Enhanced Expression System initialized with Agent SDK")
    
    async def update_expression_pattern(self) -> ExpressionPattern:
        """
        Update the current expression pattern based on emotional and mood states.
        Returns the updated pattern.
        """
        result = await Runner.run(
            self.orchestrator_agent,
            "Update the expression pattern based on current emotional and mood states.",
            context=self.context,
            run_config=RunConfig(
                workflow_name="Expression_Pattern_Update",
                trace_metadata={
                    "update_type": "scheduled",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        )
        
        # Update current pattern and timestamp
        try:
            pattern_data = result.final_output
            if isinstance(pattern_data, dict):
                self.current_pattern = ExpressionPattern(**pattern_data)
                self.last_pattern_update = datetime.datetime.now()
                return self.current_pattern
            else:
                logger.warning(f"Unexpected pattern update result format: {type(result.final_output)}")
                return self.current_pattern
        except Exception as e:
            logger.error(f"Error updating expression pattern: {e}")
            return self.current_pattern
    
    async def apply_text_expression(self, text: str, intensity: float = 1.0) -> str:
        """
        Apply the current expression pattern to text.
        Modifies the text based on emotional state.
        """
        # Check if we need to update the pattern first
        now = datetime.datetime.now()
        if (now - self.last_pattern_update).total_seconds() > self.pattern_update_interval:
            await self.update_expression_pattern()
        
        # Use the orchestrator agent to apply text expression
        prompt = f"""
        Apply expression pattern to the following text with intensity {intensity}:
        
        {text}
        """
        
        result = await Runner.run(
            self.orchestrator_agent,
            prompt,
            context=self.context,
            run_config=RunConfig(
                workflow_name="Text_Expression_Application",
                trace_metadata={
                    "text_length": len(text),
                    "intensity": intensity,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        )
        
        # Process result
        modified_text = result.final_output
        if isinstance(modified_text, str):
            return modified_text
        
        # Fallback if we got something unexpected
        logger.warning(f"Unexpected text expression result: {type(result.final_output)}")
        return text
    
    async def get_behavioral_expressions(self) -> Dict[str, Any]:
        """
        Get behavioral expression suggestions based on current pattern.
        Returns a dictionary of suggested behaviors and their strengths.
        """
        result = await Runner.run(
            self.orchestrator_agent,
            "Get behavioral expression suggestions based on the current pattern.",
            context=self.context,
            run_config=RunConfig(
                workflow_name="Behavioral_Expression_Generation",
                trace_metadata={
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        )
        
        # Process result
        expressions = result.final_output
        if isinstance(expressions, dict):
            return expressions
        
        # Fallback if we got something unexpected
        logger.warning(f"Unexpected behavioral expressions result: {type(result.final_output)}")
        return {}
    
    async def get_action_biases(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get action biases that can influence the action generator.
        Returns a dictionary mapping activity types to bias values.
        """
        # Prepare prompt with context if provided
        if context:
            prompt = f"Get action biases for context: {json.dumps(context)}"
        else:
            prompt = "Get action biases based on the current expression pattern."
        
        result = await Runner.run(
            self.orchestrator_agent,
            prompt,
            context=self.context,
            run_config=RunConfig(
                workflow_name="Action_Bias_Generation",
                trace_metadata={
                    "has_context": context is not None,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        )
        
        # Process result
        biases = result.final_output
        if isinstance(biases, dict):
            return biases
        
        # Fallback if we got something unexpected
        logger.warning(f"Unexpected action biases result: {type(result.final_output)}")
        return {}
    
    def register_text_callback(self, callback: Callable, pre_process: bool = True):
        """
        Register a callback function to modify text.
        Args:
            callback: Async function that takes (text, pattern) and returns modified text
            pre_process: If True, runs before standard processing, otherwise after
        """
        if pre_process:
            self.pre_text_callbacks.append(callback)
        else:
            self.post_text_callbacks.append(callback)
        
        # Also add to context for agent access
        self.context.pre_text_callbacks = self.pre_text_callbacks
        self.context.post_text_callbacks = self.post_text_callbacks
            
    def register_action_callback(self, callback: Callable, pre_process: bool = True):
        """
        Register a callback function to influence action generation.
        Args:
            callback: Async function that takes (action_context, pattern) and returns modified context
            pre_process: If True, runs before standard processing, otherwise after
        """
        if pre_process:
            self.pre_action_callbacks.append(callback)
        else:
            self.post_action_callbacks.append(callback)
    
    async def create_context_specific_pattern(self, context_type: str, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a context-specific expression pattern.
        
        Args:
            context_type: Type of context (e.g., 'formal', 'casual', 'intimate')
            adjustments: Specific adjustments for this context
            
        Returns:
            The created context-specific pattern
        """
        # Make sure we have the latest pattern as base
        await self.update_expression_pattern()
        
        prompt = f"""
        Create a context-specific expression pattern for context type '{context_type}'
        with the following adjustments:
        
        {json.dumps(adjustments)}
        """
        
        result = await Runner.run(
            self.orchestrator_agent,
            prompt,
            context=self.context,
            run_config=RunConfig(
                workflow_name="Context_Pattern_Creation",
                trace_metadata={
                    "context_type": context_type,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        )
        
        return result.final_output
    
    async def get_context_specific_pattern(self, context_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a previously stored context-specific expression pattern.
        
        Args:
            context_type: Type of context to retrieve
            
        Returns:
            The context-specific pattern if found, None otherwise
        """
        result = await Runner.run(
            self.orchestrator_agent,
            f"Get context-specific pattern for context type '{context_type}'",
            context=self.context
        )
        
        return result.final_output
    
    async def analyze_expression_pattern(self, reference_pattern: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the current expression pattern and provide insights.
        
        Args:
            reference_pattern: Optional reference pattern to compare against
            
        Returns:
            Analysis of the expression pattern
        """
        # Make sure we have the latest pattern
        await self.update_expression_pattern()
        
        if reference_pattern:
            prompt = "Analyze the current expression pattern compared to the reference pattern."
        else:
            prompt = "Analyze the current expression pattern."
        
        result = await Runner.run(
            self.orchestrator_agent,
            prompt,
            context=self.context,
            run_config=RunConfig(
                workflow_name="Expression_Pattern_Analysis",
                trace_metadata={
                    "has_reference": reference_pattern is not None,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        )
        
        return result.final_output
