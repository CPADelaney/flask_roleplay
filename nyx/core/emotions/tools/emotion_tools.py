# nyx/core/emotions/tools/emotion_tools.py

"""
Function tools for emotion derivation and analysis.
These tools handle deriving emotions from neurochemicals and analyzing text.
"""

import datetime
import logging
from typing import Dict, Any, Set, List, Optional

from agents import function_tool, RunContextWrapper, function_span

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    EmotionalStateMatrix, DerivedEmotion, TextAnalysisOutput
)

logger = logging.getLogger(__name__)

class EmotionTools:
    """Function tools for emotion derivation and analysis"""
    
    def __init__(self, emotion_system):
        """
        Initialize with reference to the emotion system
        
        Args:
            emotion_system: The emotion system to interact with
        """
        self.neurochemicals = emotion_system.neurochemicals
        self.emotion_derivation_rules = emotion_system.emotion_derivation_rules
        self.apply_chemical_decay = emotion_system._apply_chemical_decay
    
    @function_tool
    async def derive_emotional_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, float]:
        """
        Derive emotional state from current neurochemical levels
        
        Returns:
            Dictionary of emotion names and intensities
        """
        with function_span("derive_emotional_state"):
            # Get current chemical levels - use cached values if available
            cached_state = ctx.context.get_cached_neurochemicals()
            
            if cached_state is not None:
                chemical_levels = cached_state
            else:
                # Apply decay if needed
                await self.apply_chemical_decay(ctx)
                chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
            
            # Process each emotion rule using a more efficient approach
            emotion_scores = {}
            
            for rule in self.emotion_derivation_rules:
                conditions = rule["chemical_conditions"]
                emotion = rule["emotion"]
                rule_weight = rule.get("weight", 1.0)
                
                # Calculate match score more efficiently using list comprehension
                match_scores = [
                    min(chemical_levels.get(chemical, 0) / threshold, 1.0)
                    for chemical, threshold in conditions.items()
                    if chemical in chemical_levels and threshold > 0
                ]
                
                # Average match scores if any exist
                avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0
                
                # Apply rule weight
                weighted_score = avg_match_score * rule_weight
                
                # Only include non-zero scores
                if weighted_score > 0:
                    emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), weighted_score)
            
            # Normalize if total intensity is too high
            total_intensity = sum(emotion_scores.values())
            if total_intensity > 1.5:
                factor = 1.5 / total_intensity
                emotion_scores = {e: i * factor for e, i in emotion_scores.items()}
            
            # Cache the results in context
            ctx.context.last_emotions = emotion_scores
            
            return emotion_scores
    
    @function_tool
    async def get_emotional_state_matrix(self, ctx: RunContextWrapper[EmotionalContext]) -> EmotionalStateMatrix:
        """
        Get the full emotional state matrix derived from neurochemicals
        
        Returns:
            Emotional state matrix with primary and secondary emotions
        """
        with function_span("get_emotional_state_matrix"):
            # First, apply decay to ensure current state
            await self.apply_chemical_decay(ctx)
            
            # Get derived emotions
            emotion_intensities = await self.derive_emotional_state(ctx)
            
            # Pre-compute emotion valence and arousal map for efficiency
            emotion_valence_map = {
                rule["emotion"]: (rule.get("valence", 0.0), rule.get("arousal", 0.5)) 
                for rule in self.emotion_derivation_rules
            }
            
            # Find primary emotion (highest intensity)
            primary_emotion = max(emotion_intensities.items(), key=lambda x: x[1]) if emotion_intensities else ("Neutral", 0.5)
            primary_name, primary_intensity = primary_emotion
            
            # Get primary emotion valence and arousal
            primary_valence, primary_arousal = emotion_valence_map.get(primary_name, (0.0, 0.5))
            
            # Process secondary emotions more efficiently
            secondary_emotions = {
                emotion: DerivedEmotion(
                    name=emotion,
                    intensity=intensity,
                    valence=emotion_valence_map.get(emotion, (0.0, 0.5))[0],
                    arousal=emotion_valence_map.get(emotion, (0.0, 0.5))[1]
                )
                for emotion, intensity in emotion_intensities.items()
                if emotion != primary_name and intensity > 0.3
            }
            
            # Calculate overall valence and arousal (weighted average) more efficiently
            total_intensity = primary_intensity + sum(e.intensity for e in secondary_emotions.values())
            
            if total_intensity > 0:
                # Start with primary emotion contribution
                overall_valence = primary_valence * primary_intensity
                overall_arousal = primary_arousal * primary_intensity
                
                # Add contributions from secondary emotions
                for emotion in secondary_emotions.values():
                    overall_valence += emotion.valence * emotion.intensity
                    overall_arousal += emotion.arousal * emotion.intensity
                    
                # Normalize by total intensity
                overall_valence /= total_intensity
                overall_arousal /= total_intensity
            else:
                overall_valence = 0.0
                overall_arousal = 0.5
            
            # Ensure values are within range
            overall_valence = max(-1.0, min(1.0, overall_valence))
            overall_arousal = max(0.0, min(1.0, overall_arousal))
            
            # Create the complete state matrix
            state_matrix = EmotionalStateMatrix(
                primary_emotion=DerivedEmotion(
                    name=primary_name,
                    intensity=primary_intensity,
                    valence=primary_valence,
                    arousal=primary_arousal
                ),
                secondary_emotions=secondary_emotions,
                valence=overall_valence,
                arousal=overall_arousal,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            # Record primary emotion in context for quick access
            ctx.context.record_emotion(primary_name, primary_intensity)
            
            return state_matrix
    
    @function_tool
    async def analyze_text_sentiment(self, ctx: RunContextWrapper[EmotionalContext],
                                 text: str) -> TextAnalysisOutput:
        """
        Analyze the emotional content of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis of emotional content
        """
        with function_span("analyze_text_sentiment", input=str(text)[:100]):
            # Enhanced pattern recognition using efficient set operations
            text_lower = text.lower()
            words = set(text_lower.split())
            
            # Define word sets for more efficient lookup
            word_categories = {
                "nyxamine": {"happy", "good", "great", "love", "like", "fun", "enjoy", "curious", 
                            "interested", "pleasure", "delight", "joy"},
                "seranix": {"calm", "peaceful", "relaxed", "content", "satisfied", "gentle", 
                           "quiet", "serene", "tranquil", "composed"},
                "oxynixin": {"trust", "close", "together", "bond", "connect", "loyal", "friend", 
                            "relationship", "intimate", "attachment"},
                "cortanyx": {"worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry", 
                            "angry", "upset", "frustrated", "anxious", "distressed"},
                "adrenyx": {"excited", "alert", "surprised", "wow", "amazing", "intense", "sudden", 
                           "quick", "shock", "unexpected", "startled"}
            }
            
            intensifiers = {"very", "extremely", "incredibly", "so", "deeply", "absolutely", 
                          "truly", "utterly", "completely", "totally"}
            
            # Compute intersection of words for efficient scoring using dictionary comprehension
            matches = {
                category: words.intersection(word_set)
                for category, word_set in word_categories.items()
            }
            
            intensifier_count = len(words.intersection(intensifiers))
            
            # Calculate chemical impacts more efficiently
            chemical_impacts = {
                chemical: min(0.5, len(matches[chemical]) * 0.1)
                for chemical in word_categories.keys()
                if matches[chemical]
            }
            
            # Apply intensity modifiers
            if intensifier_count > 0:
                intensity_multiplier = 1.0 + (intensifier_count * 0.2)  # Up to 1.0 + (5 * 0.2) = 2.0
                chemical_impacts = {
                    k: min(1.0, v * intensity_multiplier) for k, v in chemical_impacts.items()
                }
            
            # If no chemicals were identified, add small baseline activation
            if not chemical_impacts:
                chemical_impacts = {
                    "nyxamine": 0.1,
                    "adrenyx": 0.1
                }
            
            # Create a temporary neurochemical state for analysis
            temp_chemicals = {
                c: {
                    "value": self.neurochemicals[c]["value"] + chemical_impacts.get(c, 0),
                    "baseline": self.neurochemicals[c]["baseline"],
                    "decay_rate": self.neurochemicals[c]["decay_rate"]
                }
                for c in self.neurochemicals
            }
            
            # Apply bounds checking to values
            for chemical in temp_chemicals:
                temp_chemicals[chemical]["value"] = max(0.0, min(1.0, temp_chemicals[chemical]["value"]))
            
            # Derive emotions from this temporary state using emotion rules
            chemical_levels = {c: d["value"] for c, d in temp_chemicals.items()}
            
            # Pre-calculate emotion rule map for faster lookup
            emotion_valence_map = {
                rule["emotion"]: (rule.get("valence", 0.0), rule.get("arousal", 0.5)) 
                for rule in self.emotion_derivation_rules
            }
            
            # Process each emotion rule more efficiently
            derived_emotions = {}
            for rule in self.emotion_derivation_rules:
                conditions = rule["chemical_conditions"]
                emotion = rule["emotion"]
                rule_weight = rule.get("weight", 1.0)
                
                # Calculate match score using comprehension for efficiency
                match_scores = [
                    min(chemical_levels.get(chemical, 0) / threshold, 1.0)
                    for chemical, threshold in conditions.items()
                    if chemical in chemical_levels and threshold > 0
                ]
                
                # Average match scores
                avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0
                
                # Apply rule weight
                weighted_score = avg_match_score * rule_weight
                
                # Only include non-zero scores
                if weighted_score > 0:
                    derived_emotions[emotion] = max(derived_emotions.get(emotion, 0), weighted_score)
            
            # Find dominant emotion
            dominant_emotion = max(derived_emotions.items(), key=lambda x: x[1]) if derived_emotions else ("neutral", 0.5)
            
            # Calculate overall valence and intensity more efficiently
            valence_contributions = [
                emotion_valence_map.get(emotion, (0.0, 0.5))[0] * intensity
                for emotion, intensity in derived_emotions.items()
            ]
            
            valence = sum(valence_contributions) / sum(derived_emotions.values()) if derived_emotions else 0.0
            
            # Calculate overall intensity
            intensity = sum(derived_emotions.values()) / max(1, len(derived_emotions))
            
            return TextAnalysisOutput(
                chemicals_affected=chemical_impacts,
                derived_emotions=derived_emotions,
                dominant_emotion=dominant_emotion[0],
                intensity=intensity,
                valence=valence
            )
