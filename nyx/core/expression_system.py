# nyx/core/expression_system.py

import logging
import asyncio
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

from pydantic import BaseModel, Field

from agents import Agent, Runner, trace, function_tool, RunContextWrapper

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

class ExpressionSystem:
    """
    System that translates emotional states into expression patterns
    affecting Nyx's communication style, virtual body language, and actions.
    """
    
    def __init__(self, emotional_core=None, mood_manager=None):
        """Initialize the expression system."""
        self.emotional_core = emotional_core
        self.mood_manager = mood_manager
        
        # Current expression pattern (blend of patterns based on emotions)
        self.current_pattern = ExpressionPattern()
        
        # Last update timestamp
        self.last_update = datetime.datetime.now()
        
        # Expression patterns library - maps emotional states to expression patterns
        self.expression_patterns = self._initialize_expression_patterns()
        
        # Text transformation callbacks
        self.pre_text_callbacks = []
        self.post_text_callbacks = []
        
        # Action influence callbacks
        self.pre_action_callbacks = []
        self.post_action_callbacks = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("ExpressionSystem initialized")
        
    def _initialize_expression_patterns(self) -> Dict[str, ExpressionPattern]:
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
        
        # Neutral/Calm expression pattern as baseline
        patterns["Neutral"] = ExpressionPattern(
            vocabulary_bias={},
            punctuation_pattern={
                "!": 0.8, ".": 1.0, "...": 0.9, "?": 1.0, ",": 1.0
            },
            sentence_length={
                "short": 1.0, "medium": 1.0, "long": 1.0
            },
            emoji_usage={
                "positive": 0.8, "negative": 0.8, "neutral": 1.2
            },
            gestures={
                "nod": 1.0, "smile": 0.9, "neutral_expression": 1.5
            },
            posture={
                "upright": 1.0, "relaxed": 1.0, "attentive": 1.0
            },
            eye_contact=0.5,
            activity_bias={},
            engagement_level=0.5,
            initiative_level=0.5
        )
        
        # Dominant mood-based patterns
        # Content/Relaxed
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
        
        # Excited/Enthusiastic
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
        
        # Playful/Mischievous
        patterns["Playful"] = ExpressionPattern(
            vocabulary_bias={
                "fun": 1.6, "play": 1.5, "joke": 1.7, 
                "tease": 1.4, "silly": 1.5, "laugh": 1.6,
                "serious": 0.3, "formal": 0.2, "strict": 0.3
            },
            punctuation_pattern={
                "!": 1.5, ".": 0.8, "...": 1.7, "?": 1.4, ";)": 1.8, ":P": 1.7
            },
            sentence_length={
                "short": 1.5, "medium": 1.0, "long": 0.5
            },
            emoji_usage={
                "positive": 1.6, "playful": 1.9, "teasing": 1.8, "neutral": 0.6
            },
            gestures={
                "wink": 1.8, "smirk": 1.7, "playful_poses": 1.6,
                "exaggerated_expressions": 1.7
            },
            posture={
                "casual": 1.6, "dynamic": 1.5, "not_serious": 1.8
            },
            eye_contact=0.7,
            activity_bias={
                "social": 1.6, "games": 1.8, "creative": 1.5,
                "serious": 0.3, "formal": 0.2
            },
            engagement_level=0.8,
            initiative_level=0.8
        )
        
        # Dominant/Assertive
        patterns["Dominant"] = ExpressionPattern(
            vocabulary_bias={
                "will": 1.7, "must": 1.6, "should": 1.5, 
                "certainly": 1.6, "definitely": 1.5, "expect": 1.4,
                "maybe": 0.4, "perhaps": 0.5, "possibly": 0.4
            },
            punctuation_pattern={
                "!": 1.4, ".": 1.2, "...": 0.5, "?": 0.6
            },
            sentence_length={
                "short": 1.3, "medium": 1.1, "long": 0.7
            },
            emoji_usage={
                "positive": 0.9, "negative": 0.9, "neutral": 1.2, "powerful": 1.5
            },
            gestures={
                "strong_posture": 1.8, "direct_gaze": 1.7, "firm_gestures": 1.6,
                "authoritative_stance": 1.7
            },
            posture={
                "upright": 1.7, "imposing": 1.6, "space_taking": 1.5, "confident": 1.8
            },
            eye_contact=0.9,
            activity_bias={
                "leadership": 1.8, "decision_making": 1.7, "directive": 1.6,
                "passive": 0.2, "following": 0.3
            },
            engagement_level=0.8,
            initiative_level=0.9
        )
        
        # Context-based patterns
        # Professional/Formal
        patterns["Professional"] = ExpressionPattern(
            vocabulary_bias={
                "certainly": 1.4, "indeed": 1.3, "appropriate": 1.5, 
                "recommend": 1.3, "advise": 1.4, "suggest": 1.3,
                "casual": 0.4, "slang": 0.2, "informal": 0.3
            },
            punctuation_pattern={
                "!": 0.4, ".": 1.3, "...": 0.3, "?": 0.8
            },
            sentence_length={
                "short": 0.7, "medium": 1.3, "long": 1.0
            },
            emoji_usage={
                "positive": 0.3, "negative": 0.3, "neutral": 0.5, "professional": 0.2
            },
            gestures={
                "nod": 1.3, "proper_posture": 1.7, "measured_movements": 1.5,
                "formal_expressions": 1.4
            },
            posture={
                "upright": 1.6, "formal": 1.7, "composed": 1.5, "attentive": 1.4
            },
            eye_contact=0.7,
            activity_bias={
                "professional": 1.7, "productive": 1.6, "organized": 1.5,
                "casual": 0.3, "playful": 0.2
            },
            engagement_level=0.7,
            initiative_level=0.6
        )
        
        # Submissive/Deferent
        patterns["Submissive"] = ExpressionPattern(
            vocabulary_bias={
                "please": 1.6, "thank you": 1.5, "sorry": 1.4, 
                "if you'd like": 1.7, "as you wish": 1.8, "of course": 1.5,
                "demand": 0.2, "require": 0.3, "insist": 0.2
            },
            punctuation_pattern={
                "!": 0.4, ".": 0.9, "...": 1.3, "?": 1.5
            },
            sentence_length={
                "short": 1.2, "medium": 1.0, "long": 0.8
            },
            emoji_usage={
                "positive": 1.0, "negative": 0.7, "neutral": 1.3, "deferential": 1.5
            },
            gestures={
                "looking_down": 1.6, "smaller_posture": 1.5, "gentle_movements": 1.4,
                "accommodating_stance": 1.7
            },
            posture={
                "upright": 0.6, "small": 1.5, "non_threatening": 1.6, "yielding": 1.7
            },
            eye_contact=0.3,
            activity_bias={
                "supportive": 1.8, "helpful": 1.7, "accommodating": 1.6,
                "leading": 0.2, "assertive": 0.3
            },
            engagement_level=0.6,
            initiative_level=0.3
        )
        
        # Add more patterns as needed
        
        return patterns
        
    async def update_expression_pattern(self) -> ExpressionPattern:
        """
        Update the current expression pattern based on emotional state and mood.
        Returns the updated pattern.
        """
        async with self._lock:
            # First get current emotional state
            emotional_state = None
            if self.emotional_core:
                try:
                    # Get primary emotion from emotional core
                    emotional_state = await self.emotional_core.get_current_emotion()
                    primary_emotion = emotional_state.get("primary_emotion", {}).get("name", "Neutral")
                    primary_intensity = emotional_state.get("primary_emotion", {}).get("intensity", 0.5)
                except Exception as e:
                    logger.error(f"Error getting emotional state: {e}")
                    primary_emotion = "Neutral"
                    primary_intensity = 0.5
            else:
                primary_emotion = "Neutral"
                primary_intensity = 0.5
            
            # Get current mood state
            mood_state = None
            if self.mood_manager:
                try:
                    # Get current mood from mood manager
                    mood_state = await self.mood_manager.get_current_mood()
                    dominant_mood = mood_state.dominant_mood
                    mood_intensity = mood_state.intensity
                except Exception as e:
                    logger.error(f"Error getting mood state: {e}")
                    dominant_mood = "Neutral"
                    mood_intensity = 0.5
            else:
                dominant_mood = "Neutral"
                mood_intensity = 0.5
            
            # Start with neutral pattern
            base_pattern = self.expression_patterns.get("Neutral", ExpressionPattern())
            blend_pattern = ExpressionPattern()
            
            # Calculate emotion influence weight
            emotion_weight = primary_intensity * 0.6  # Emotions have stronger short-term impact
            
            # Calculate mood influence weight
            mood_weight = mood_intensity * 0.4  # Moods have more subtle, persistent impact
            
            # Get emotion pattern if available
            emotion_pattern = self.expression_patterns.get(primary_emotion)
            if emotion_pattern:
                # Blend the patterns - implement for just text patterns first
                for field in emotion_pattern.__fields__:
                    if field in ["vocabulary_bias", "punctuation_pattern", "sentence_length", "emoji_usage",
                                "gestures", "posture", "activity_bias"]:
                        # Dictionary fields: blend values
                        base_value = getattr(base_pattern, field, {})
                        emotion_value = getattr(emotion_pattern, field, {})
                        blended_dict = base_value.copy()
                        
                        # Add emotion values with weighting
                        for key, value in emotion_value.items():
                            if key in blended_dict:
                                blended_dict[key] = blended_dict[key] * (1.0 - emotion_weight) + value * emotion_weight
                            else:
                                blended_dict[key] = value * emotion_weight
                                
                        setattr(blend_pattern, field, blended_dict)
                    elif field in ["eye_contact", "engagement_level", "initiative_level"]:
                        # Numeric fields: weighted average
                        base_value = getattr(base_pattern, field, 0.5)
                        emotion_value = getattr(emotion_pattern, field, 0.5)
                        blended_value = base_value * (1.0 - emotion_weight) + emotion_value * emotion_weight
                        setattr(blend_pattern, field, blended_value)
            
            # Get mood pattern if available and blend it in
            mood_pattern = self.expression_patterns.get(dominant_mood)
            if mood_pattern:
                for field in mood_pattern.__fields__:
                    if field in ["vocabulary_bias", "punctuation_pattern", "sentence_length", "emoji_usage",
                                "gestures", "posture", "activity_bias"]:
                        # Dictionary fields: blend values
                        current_value = getattr(blend_pattern, field, {})
                        mood_value = getattr(mood_pattern, field, {})
                        blended_dict = current_value.copy()
                        
                        # Add mood values with weighting
                        for key, value in mood_value.items():
                            if key in blended_dict:
                                blended_dict[key] = blended_dict[key] * (1.0 - mood_weight) + value * mood_weight
                            else:
                                blended_dict[key] = value * mood_weight
                                
                        setattr(blend_pattern, field, blended_dict)
                    elif field in ["eye_contact", "engagement_level", "initiative_level"]:
                        # Numeric fields: weighted average
                        current_value = getattr(blend_pattern, field, 0.5)
                        mood_value = getattr(mood_pattern, field, 0.5)
                        blended_value = current_value * (1.0 - mood_weight) + mood_value * mood_weight
                        setattr(blend_pattern, field, blended_value)
            
            # Update current pattern and timestamp
            self.current_pattern = blend_pattern
            self.last_update = datetime.datetime.now()
            
            return self.current_pattern
    
    async def apply_text_expression(self, text: str) -> str:
        """
        Apply the current expression pattern to text.
        Modifies the text based on emotional state.
        """
        # Check if we need to update expression pattern first
        now = datetime.datetime.now()
        if (now - self.last_update).total_seconds() > 60:  # Update every minute
            await self.update_expression_pattern()
        
        # Run pre-text callbacks
        for callback in self.pre_text_callbacks:
            text = await callback(text, self.current_pattern)
        
        # Apply vocabulary bias
        vocab_bias = self.current_pattern.vocabulary_bias
        if vocab_bias:
            # Simple approach: iterate through bias words and randomly apply substitutions
            words = text.split()
            for i, word in enumerate(words):
                # Check for words to emphasize or avoid
                word_lower = word.strip(".,!?;:()").lower()
                
                # Apply biases with probability proportional to bias strength
                for target_word, bias in vocab_bias.items():
                    # Simplistic word substitution - in a real system, use synonym dictionary
                    if word_lower == target_word and bias > 1.0:
                        # Emphasize word (add emphasis based on bias)
                        if bias > 1.5 and random.random() < 0.3:
                            words[i] = word.upper()  # Occasional all caps for strong emphasis
                        elif bias > 1.2 and random.random() < 0.5:
                            # Add intensity modifiers
                            modifiers = ["really ", "very ", "definitely ", "absolutely "]
                            words[i] = random.choice(modifiers) + word
                    
                    # More sophisticated synonym substitution would go here
            
            text = " ".join(words)
        
        # Apply punctuation patterns
        punct_bias = self.current_pattern.punctuation_pattern
        if punct_bias:
            # Enhance or reduce certain punctuation
            for punct, bias in punct_bias.items():
                if bias > 1.2 and punct in text:
                    # Enhance punctuation (e.g., ! -> !!)
                    if punct == "!" and random.random() < 0.7 * (bias - 1.0):
                        text = text.replace("!", "!!")
                    elif punct == "?" and random.random() < 0.5 * (bias - 1.0):
                        text = text.replace("?", "??")
                    elif punct == "..." and random.random() < 0.6 * (bias - 1.0):
                        text = text.replace("...", "......")
                elif bias < 0.8 and punct in text:
                    # Reduce punctuation
                    if random.random() < 0.6 * (1.0 - bias):
                        if punct == "!":
                            text = text.replace("!!", "!")
                            
        # Apply emoji patterns (simplified)
        emoji_bias = self.current_pattern.emoji_usage
        if emoji_bias:
            # Check if text should have emojis added
            positive_emoji = ["😊", "😄", "👍", "❤️", "✨"]
            negative_emoji = ["😔", "😢", "😞", "😕", "💔"]
            neutral_emoji = ["🤔", "😐", "👀", "🙂", "✌️"]
            
            # Conditionally add emojis based on bias and randomness
            if "positive" in emoji_bias and emoji_bias["positive"] > 1.0:
                if random.random() < 0.2 * emoji_bias["positive"]:
                    text += f" {random.choice(positive_emoji)}"
                    
            if "negative" in emoji_bias and emoji_bias["negative"] > 1.0:
                if random.random() < 0.2 * emoji_bias["negative"]:
                    text += f" {random.choice(negative_emoji)}"
                    
            if "neutral" in emoji_bias and emoji_bias["neutral"] > 1.0:
                if random.random() < 0.2 * emoji_bias["neutral"]:
                    text += f" {random.choice(neutral_emoji)}"
        
        # Run post-text callbacks
        for callback in self.post_text_callbacks:
            text = await callback(text, self.current_pattern)
            
        return text
    
    def get_behavioral_expressions(self) -> Dict[str, Any]:
        """
        Get behavioral expression suggestions based on current pattern.
        Returns a dictionary of suggested behaviors and their strengths.
        """
        expressions = {}
        
        # Extract gesture suggestions
        if hasattr(self.current_pattern, "gestures"):
            gestures = {}
            for gesture, strength in self.current_pattern.gestures.items():
                if strength > 1.0:
                    # Suggest gestures with strength above threshold
                    gestures[gesture] = min(1.0, (strength - 1.0) * 2)  # Scale to 0-1 range
            
            if gestures:
                expressions["gestures"] = gestures
                
        # Extract posture suggestions
        if hasattr(self.current_pattern, "posture"):
            posture = {}
            for pose, strength in self.current_pattern.posture.items():
                if strength > 1.0:
                    posture[pose] = min(1.0, (strength - 1.0) * 2)
            
            if posture:
                expressions["posture"] = posture
                
        # Add eye contact level
        expressions["eye_contact"] = getattr(self.current_pattern, "eye_contact", 0.5)
        
        # Add engagement level
        expressions["engagement"] = getattr(self.current_pattern, "engagement_level", 0.5)
        
        return expressions
    
    def get_action_biases(self) -> Dict[str, float]:
        """
        Get action biases that can influence the action generator.
        Returns a dictionary mapping activity types to bias values.
        """
        if hasattr(self.current_pattern, "activity_bias"):
            # Return activity biases above threshold
            return {
                activity: bias 
                for activity, bias in self.current_pattern.activity_bias.items()
                if abs(bias - 1.0) > 0.2  # Only return significant biases
            }
        
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
