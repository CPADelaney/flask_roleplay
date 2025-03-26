# nyx/core/emotions/tools/emotion_tools.py

"""
Enhanced function tools for emotion derivation and analysis.
These tools handle deriving emotions from neurochemicals and analyzing text
with improved performance and error handling, leveraging the OpenAI Agents SDK.
"""

import datetime
import logging
from typing import Dict, Any, Set, List, Optional, Tuple, TypedDict, Union, cast
import json

from agents import (
    function_tool, RunContextWrapper, function_span, custom_span, Agent,
    GuardrailFunctionOutput, FunctionTool, Tool, trace, gen_trace_id
)
from agents.exceptions import UserError, ModelBehaviorError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    EmotionalStateMatrix, DerivedEmotion, TextAnalysisOutput,
    EmotionValence, EmotionArousal
)
from nyx.core.emotions.utils import handle_errors, EmotionalToolUtils

logger = logging.getLogger(__name__)

# Define types for improved type hinting
EmotionData = Dict[str, float]
NeurochemicalState = Dict[str, float]
EmotionRule = Dict[str, Any]

class EmotionTools:
    """Enhanced function tools for emotion derivation and analysis with SDK integration"""
    
    def __init__(self, emotion_system):
        """
        Initialize with reference to the emotion system
        
        Args:
            emotion_system: The emotion system to interact with
        """
        self.neurochemicals = emotion_system.neurochemicals
        self.emotion_derivation_rules = emotion_system.emotion_derivation_rules
        
        # Store reference to functions we need
        if hasattr(emotion_system, "apply_chemical_decay"):
            self.apply_chemical_decay = emotion_system.apply_chemical_decay
        elif hasattr(emotion_system, "_apply_chemical_decay"):
            self.apply_chemical_decay = emotion_system._apply_chemical_decay
        
        # Create indexed lookup tables for more efficient processing
        self._emotion_rule_index = self._index_emotion_rules()
        self._valence_arousal_map = self._create_valence_arousal_map()
    
    def _index_emotion_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create an indexed version of emotion rules for faster lookups
        
        Returns:
            Dictionary mapping chemicals to relevant emotion rules
        """
        chemical_rules = {}
        
        for chemical in set(chem for rule in self.emotion_derivation_rules 
                            for chem in rule.get("chemical_conditions", {})):
            # For each chemical, find all rules where it appears
            chemical_rules[chemical] = [
                rule for rule in self.emotion_derivation_rules
                if chemical in rule.get("chemical_conditions", {})
            ]
        
        return chemical_rules
    
    def _create_valence_arousal_map(self) -> Dict[str, Tuple[float, float]]:
        """
        Create a mapping of emotions to their valence and arousal values
        
        Returns:
            Dictionary mapping emotion names to (valence, arousal) tuples
        """
        return {
            rule["emotion"]: (rule.get("valence", 0.0), rule.get("arousal", 0.5))
            for rule in self.emotion_derivation_rules
        }
    
    @function_tool
    async def derive_emotional_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, float]:
        """
        Derive emotional state from current neurochemical levels with optimized processing
        
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
                if hasattr(self, "apply_chemical_decay"):
                    await self.apply_chemical_decay(ctx)
                chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
            
            # Process each emotion rule using a more efficient approach
            emotion_scores = {}
            
            # Use a two-phase approach for better efficiency
            # First phase: determine relevant rules based on chemical levels
            relevant_emotions = set()
            
            # Track which chemicals exceed their threshold for faster processing
            exceeded_thresholds = {}
            
            for chemical, level in chemical_levels.items():
                if chemical not in self._emotion_rule_index:
                    continue
                    
                # Track which rules have this chemical exceeding threshold
                for rule in self._emotion_rule_index[chemical]:
                    threshold = rule["chemical_conditions"].get(chemical, 0)
                    
                    # Only consider rules where chemical level is significant
                    if level >= threshold * 0.7:  # At least 70% of threshold
                        relevant_emotions.add(rule["emotion"])
                        
                        # Track this chemical as exceeding threshold for this rule
                        rule_id = rule["emotion"]
                        if rule_id not in exceeded_thresholds:
                            exceeded_thresholds[rule_id] = set()
                        
                        exceeded_thresholds[rule_id].add(chemical)
            
            # Second phase: score only the relevant emotions for efficiency
            with custom_span(
                "emotion_scoring",
                data={
                    "relevant_emotions": list(relevant_emotions),
                    "chemical_levels": {k: round(v, 2) for k, v in chemical_levels.items()}
                }
            ):
                for rule in self.emotion_derivation_rules:
                    # Skip rules for emotions that aren't relevant
                    if rule["emotion"] not in relevant_emotions:
                        continue
                        
                    # Skip rules where not all required chemicals exceed threshold
                    rule_id = rule["emotion"]
                    if rule_id not in exceeded_thresholds:
                        continue
                    
                    conditions = rule["chemical_conditions"]
                    emotion = rule["emotion"]
                    rule_weight = rule.get("weight", 1.0)
                    
                    # Only process rules where all chemicals are available
                    chemicals_available = all(chemical in chemical_levels for chemical in conditions)
                    
                    if chemicals_available:
                        # Calculate match score using list comprehension for efficiency
                        match_scores = [
                            min(chemical_levels.get(chemical, 0) / threshold, 1.0)
                            for chemical, threshold in conditions.items()
                            if threshold > 0  # Skip chemicals with zero threshold
                        ]
                        
                        # Only process valid scores
                        if match_scores:
                            # Average match scores
                            avg_match_score = sum(match_scores) / len(match_scores)
                            
                            # Apply rule weight
                            weighted_score = avg_match_score * rule_weight
                            
                            # Only include significant scores to reduce noise
                            if weighted_score > 0.1:
                                emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), weighted_score)
            
            # If no emotions were detected, add neutral state
            if not emotion_scores:
                emotion_scores["Neutral"] = 0.5
            
            # Normalize if total intensity is too high
            total_intensity = sum(emotion_scores.values())
            if total_intensity > 1.5:
                factor = 1.5 / total_intensity
                emotion_scores = {e: i * factor for e, i in emotion_scores.items()}
            
            # Cache the results in context
            ctx.context.last_emotions = emotion_scores
            
            # Create a custom span for emotion state
            with custom_span(
                "emotional_state", 
                data={
                    "emotions": {
                        k: round(v, 2) for k, v in emotion_scores.items() 
                        if v > 0.1  # Only include significant emotions
                    },
                    "cycle": ctx.context.cycle_count,
                    "primary": max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "Neutral"
                }
            ):
                return emotion_scores
    
    @function_tool
    async def get_emotional_state_matrix(self, ctx: RunContextWrapper[EmotionalContext]) -> EmotionalStateMatrix:
        """
        Get the full emotional state matrix derived from neurochemicals with optimized calculations
        
        Returns:
            Emotional state matrix with primary and secondary emotions
        """
        with function_span("get_emotional_state_matrix"):
            # First, apply decay to ensure current state
            if hasattr(self, "apply_chemical_decay"):
                await self.apply_chemical_decay(ctx)
            
            # Get derived emotions
            emotion_intensities = await self.derive_emotional_state(ctx)
            
            # Use pre-computed emotion valence and arousal map for efficiency
            emotion_valence_map = self._valence_arousal_map
            
            # Find primary emotion (highest intensity)
            primary_emotion = max(emotion_intensities.items(), key=lambda x: x[1]) if emotion_intensities else ("Neutral", 0.5)
            primary_name, primary_intensity = primary_emotion
            
            # Get primary emotion valence and arousal
            primary_valence, primary_arousal = emotion_valence_map.get(primary_name, (0.0, 0.5))
            
            # Process secondary emotions more efficiently
            secondary_emotions = {}
            
            # Create a span for matrix calculation
            with custom_span(
                "calculate_emotion_matrix",
                data={
                    "primary": primary_name,
                    "intensity": primary_intensity,
                    "secondary_count": len([e for e, i in emotion_intensities.items() 
                                            if e != primary_name and i > 0.2])
                }
            ):
                for emotion, intensity in emotion_intensities.items():
                    if emotion != primary_name and intensity > 0.2:  # Lower threshold to include more emotions
                        valence, arousal = emotion_valence_map.get(emotion, (0.0, 0.5))
                        secondary_emotions[emotion] = DerivedEmotion(
                            name=emotion,
                            intensity=intensity,
                            valence=valence,
                            arousal=arousal
                        )
                
                # Calculate overall valence and arousal (weighted average) more efficiently
                total_intensity = primary_intensity + sum(e.intensity for e in secondary_emotions.values())
                
                if total_intensity > 0:
                    # Use EmotionalToolUtils for cleaner calculation
                    valence_values = [primary_valence] + [e.valence for e in secondary_emotions.values()]
                    arousal_values = [primary_arousal] + [e.arousal for e in secondary_emotions.values()]
                    intensity_weights = [primary_intensity] + [e.intensity for e in secondary_emotions.values()]
                    
                    overall_valence = EmotionalToolUtils.calculate_weighted_average(valence_values, intensity_weights)
                    overall_arousal = EmotionalToolUtils.calculate_weighted_average(arousal_values, intensity_weights)
                else:
                    overall_valence = 0.0
                    overall_arousal = 0.5
                
                # Ensure values are within range
                overall_valence = EmotionalToolUtils.normalize_value(overall_valence, -1.0, 1.0)
                overall_arousal = EmotionalToolUtils.normalize_value(overall_arousal, 0.0, 1.0)
            
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
            
            # Create a custom span for the final emotional state
            with custom_span(
                "emotional_state_matrix", 
                data={
                    "primary": primary_name,
                    "intensity": round(primary_intensity, 2),
                    "valence": round(overall_valence, 2),
                    "arousal": round(overall_arousal, 2),
                    "secondary_count": len(secondary_emotions),
                    "cycle": ctx.context.cycle_count,
                    "valence_category": EmotionValence.from_value(overall_valence),
                    "arousal_category": EmotionArousal.from_value(overall_arousal)
                }
            ):
                return state_matrix
    
    @function_tool
    async def analyze_text_sentiment(self, ctx: RunContextWrapper[EmotionalContext],
                                text: str) -> TextAnalysisOutput:
        """
        Analyze the emotional content of text with improved efficiency and accuracy
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis of emotional content
        """
        with function_span("analyze_text_sentiment", input=str(text)[:100]):
            # Create a trace for sentiment analysis
            with trace(
                workflow_name="Text_Sentiment_Analysis",
                trace_id=gen_trace_id(),
                metadata={
                    "text_length": len(text),
                    "cycle": ctx.context.cycle_count,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            ):
                # Enhanced pattern recognition using efficient set operations
                text_lower = text.lower()
                words = set(text_lower.split())
                
                # Define word sets for more efficient lookup
                # Use a more structured approach with emotional categories
                emotional_categories = {
                    "positive": {
                        "nyxamine": {"happy", "good", "great", "love", "like", "fun", "enjoy", "curious", 
                                    "interested", "pleasure", "delight", "joy", "excited", "awesome",
                                    "excellent", "wonderful", "amazing", "fantastic", "terrific"}
                    },
                    "calm": {
                        "seranix": {"calm", "peaceful", "relaxed", "content", "satisfied", "gentle", 
                                   "quiet", "serene", "tranquil", "composed", "patient", "steady",
                                   "stable", "balanced", "harmonious", "comfortable"}
                    },
                    "social": {
                        "oxynixin": {"trust", "close", "together", "bond", "connect", "loyal", "friend", 
                                    "relationship", "intimate", "attachment", "connection", "team",
                                    "family", "care", "support", "help", "understanding", "empathy"}
                    },
                    "negative": {
                        "cortanyx": {"worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry", 
                                    "angry", "upset", "frustrated", "anxious", "distressed", "hurt",
                                    "pain", "suffering", "miserable", "terrible", "awful", "horrible"}
                    },
                    "arousing": {
                        "adrenyx": {"excited", "alert", "surprised", "wow", "amazing", "intense", "sudden", 
                                   "quick", "shock", "unexpected", "startled", "astonished", "astounded",
                                   "urgent", "emergency", "crisis", "danger", "important", "critical"}
                    }
                }
                
                intensifiers = {"very", "extremely", "incredibly", "so", "deeply", "absolutely", 
                              "truly", "utterly", "completely", "totally", "highly", "exceptionally",
                              "remarkably", "extraordinarily", "insanely", "super", "immensely"}
                
                negators = {"not", "never", "no", "none", "neither", "nor", "hardly", "barely",
                           "scarcely", "seldom", "rarely"}
                
                # Check for sentence structure patterns
                has_negation = any(negator in words for negator in negators)
                exclamation_count = text.count("!")
                question_count = text.count("?")
                
                # Create a span for pattern detection
                with custom_span(
                    "text_pattern_detection",
                    data={
                        "text_length": len(text),
                        "word_count": len(words),
                        "has_negation": has_negation,
                        "exclamations": exclamation_count,
                        "questions": question_count
                    }
                ):
                    # Compute chemical impacts more efficiently
                    chemical_impacts = {}
                    
                    # First pass - find direct word matches
                    for category, chemicals in emotional_categories.items():
                        for chemical, word_set in chemicals.items():
                            # Calculate the intersection of words
                            matches = words.intersection(word_set)
                            if matches:
                                # Calculate base impact
                                base_impact = min(0.5, len(matches) * 0.1)
                                
                                # Adjust for negation
                                if has_negation:
                                    # Convert to opposite valence
                                    if category in ["positive", "calm", "social"]:
                                        chemical = "cortanyx"  # Convert positive to negative
                                    elif category == "negative":
                                        chemical = "nyxamine"  # Convert negative to positive
                                
                                # Add to chemical impacts
                                chemical_impacts[chemical] = max(chemical_impacts.get(chemical, 0), base_impact)
                
                # Second pass - look for phrases and patterns
                phrase_patterns = [
                    # Positive patterns
                    ({"thank you", "thanks", "appreciate"}, {"nyxamine": 0.3}),
                    ({"congratulations", "congrats", "well done"}, {"nyxamine": 0.4, "oxynixin": 0.2}),
                    ({"miss you", "thinking of you"}, {"oxynixin": 0.4}),
                    
                    # Negative patterns
                    ({"go away", "leave me alone"}, {"cortanyx": 0.3}),
                    ({"help me", "please help"}, {"cortanyx": 0.2, "adrenyx": 0.3}),
                    ({"angry with", "mad at"}, {"cortanyx": 0.4}),
                    
                    # Arousal patterns
                    ({"can't wait", "looking forward"}, {"adrenyx": 0.3, "nyxamine": 0.2}),
                    ({"hurry", "quickly", "emergency"}, {"adrenyx": 0.5})
                ]
                
                # Create a span for phrase pattern detection
                with custom_span("phrase_pattern_detection"):
                    for pattern_words, impacts in phrase_patterns:
                        # Check if any of the pattern words exist in the text
                        if any(phrase in text_lower for phrase in pattern_words):
                            # Apply the impacts
                            for chemical, impact in impacts.items():
                                chemical_impacts[chemical] = max(chemical_impacts.get(chemical, 0), impact)
                
                # Account for punctuation and structure
                if exclamation_count > 0:
                    # Exclamations increase arousal
                    chemical_impacts["adrenyx"] = max(chemical_impacts.get("adrenyx", 0), 
                                                   min(0.5, exclamation_count * 0.15))
                
                if question_count > 0:
                    # Questions indicate curiosity
                    chemical_impacts["nyxamine"] = max(chemical_impacts.get("nyxamine", 0), 
                                                    min(0.3, question_count * 0.1))
                
                # Intensifier count affects all chemicals
                intensifier_count = len(words.intersection(intensifiers))
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
                
                # Record the chemical impacts for future reference
                ctx.context.set_value("last_text_analysis", {
                    "text": text[:100],  # Truncate for storage
                    "chemicals": chemical_impacts,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Create a temporary neurochemical state for analysis
                with custom_span("temporary_chemical_state"):
                    temp_chemicals = {
                        c: {
                            "value": min(1.0, self.neurochemicals[c]["value"] + chemical_impacts.get(c, 0)),
                            "baseline": self.neurochemicals[c]["baseline"],
                            "decay_rate": self.neurochemicals[c]["decay_rate"]
                        }
                        for c in self.neurochemicals
                    }
                    
                    # Apply bounds checking to values
                    for chemical in temp_chemicals:
                        temp_chemicals[chemical]["value"] = EmotionalToolUtils.normalize_value(
                            temp_chemicals[chemical]["value"])
                
                # Derive emotions from this temporary state using emotion rules
                with custom_span("derive_emotions_from_text"):
                    chemical_levels = {c: d["value"] for c, d in temp_chemicals.items()}
                    
                    # Use optimized emotion derivation approach
                    derived_emotions = {}
                    valence_sum = 0.0
                    intensity_sum = 0.0
                    
                    # Similar logic to derive_emotional_state but specialized for text analysis
                    for rule in self.emotion_derivation_rules:
                        conditions = rule["chemical_conditions"]
                        emotion = rule["emotion"]
                        rule_weight = rule.get("weight", 1.0)
                        
                        # Calculate match score using list comprehension
                        match_scores = [
                            min(chemical_levels.get(chemical, 0) / threshold, 1.0)
                            for chemical, threshold in conditions.items()
                            if chemical in chemical_levels and threshold > 0
                        ]
                        
                        # Average match scores if they exist
                        if match_scores and len(match_scores) == len(conditions):
                            avg_match_score = sum(match_scores) / len(match_scores)
                            
                            # Apply rule weight
                            weighted_score = avg_match_score * rule_weight
                            
                            # Apply a threshold to reduce noise
                            if weighted_score > 0.1:
                                derived_emotions[emotion] = max(derived_emotions.get(emotion, 0), weighted_score)
                                valence_sum += rule.get("valence", 0.0) * weighted_score
                                intensity_sum += weighted_score
                
                # Ensure we have at least one emotion - add neutral if none found
                if not derived_emotions:
                    derived_emotions["Neutral"] = 0.5
                    intensity_sum = 0.5
                
                # Find dominant emotion
                dominant_emotion = max(derived_emotions.items(), key=lambda x: x[1]) if derived_emotions else ("Neutral", 0.5)
                
                # Calculate overall intensity and valence
                intensity = intensity_sum / len(derived_emotions) if derived_emotions else 0.5
                valence = valence_sum / intensity_sum if intensity_sum > 0 else 0.0
                
                # Create the analysis output
                analysis = TextAnalysisOutput(
                    chemicals_affected=chemical_impacts,
                    derived_emotions=derived_emotions,
                    dominant_emotion=dominant_emotion[0],
                    intensity=intensity,
                    valence=valence
                )
                
                # Create a custom span for the text sentiment
                with custom_span(
                    "text_sentiment_result", 
                    data={
                        "text_length": len(text),
                        "dominant_emotion": dominant_emotion[0],
                        "intensity": round(intensity, 2),
                        "valence": round(valence, 2),
                        "chemicals": {
                            k: round(v, 2) for k, v in chemical_impacts.items()
                        },
                        "cycle": ctx.context.cycle_count,
                        "valence_category": EmotionValence.from_value(valence),
                        "arousal_category": EmotionArousal.from_value(intensity)
                    }
                ):
                    return analysis
    
    @function_tool
    async def get_emotion_trends(self, ctx: RunContextWrapper[EmotionalContext], 
                           limit: int = 10) -> Dict[str, Any]:
        """
        Get emotional trends over time
        
        Args:
            limit: Maximum number of historical data points to consider
            
        Returns:
            Dictionary of emotion trends
        """
        with function_span("get_emotion_trends"):
            # Get emotion history from context
            emotion_history = ctx.context.get_emotion_trends(limit=limit)
            
            if not emotion_history:
                return {
                    "message": "Not enough emotion history data",
                    "trends": {}
                }
            
            trends = {}
            
            # Create a span for trend analysis
            with custom_span(
                "emotion_trend_analysis",
                data={
                    "emotions_analyzed": list(emotion_history.keys()),
                    "history_points": max(len(data) for data in emotion_history.values()) if emotion_history else 0,
                    "limit": limit
                }
            ):
                # Analyze each emotion's trend
                for emotion, data_points in emotion_history.items():
                    if len(data_points) < 2:
                        continue
                        
                    intensities = [point["intensity"] for point in data_points]
                    
                    # Calculate trend statistics
                    start_value = intensities[0]
                    end_value = intensities[-1]
                    change = end_value - start_value
                    
                    # Determine trend direction
                    if abs(change) < 0.1:
                        direction = "stable"
                    elif change > 0:
                        direction = "increasing"
                    else:
                        direction = "decreasing"
                    
                    # Calculate volatility (average change between consecutive points)
                    volatility = sum(abs(intensities[i] - intensities[i-1]) 
                                   for i in range(1, len(intensities))) / (len(intensities) - 1)
                    
                    trends[emotion] = {
                        "direction": direction,
                        "change": change,
                        "volatility": volatility,
                        "current_value": end_value,
                        "points": len(data_points)
                    }
            
            # Calculate overall emotional stability
            if trends:
                avg_volatility = sum(t["volatility"] for t in trends.values()) / len(trends)
                stability = 1.0 - min(1.0, avg_volatility * 5.0)  # Invert and scale volatility
                
                # Add stability rating
                stability_rating = "very stable" if stability > 0.8 else \
                                  "stable" if stability > 0.6 else \
                                  "somewhat unstable" if stability > 0.4 else \
                                  "unstable" if stability > 0.2 else "very unstable"
            else:
                stability = 0.5
                stability_rating = "unknown"
            
            # Create a custom span for the trend analysis
            with custom_span(
                "emotion_trends_result", 
                data={
                    "emotions_analyzed": list(trends.keys()),
                    "stability": round(stability, 2),
                    "stability_rating": stability_rating,
                    "cycle": ctx.context.cycle_count
                }
            ):
                return {
                    "trends": trends,
                    "stability": stability,
                    "stability_rating": stability_rating,
                    "analysis_time": datetime.datetime.now().isoformat()
                }
