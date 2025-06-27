# nyx/core/emotions/tools/emotion_tools.py

"""
Enhanced function tools for emotion derivation and analysis.
These tools handle deriving emotions from neurochemicals and analyzing text
with improved performance and error handling, leveraging the OpenAI Agents SDK.
"""

import datetime
import logging
from typing import Dict, Any, Set, List, Optional, Tuple, Union

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


class EmotionTools:
    """
    Enhanced function tools for emotion derivation and analysis with SDK integration.

    IMPORTANT: Because each @function_tool must have RunContextWrapper[...] as its first
    parameter, we split each tool method into:
      - A private `_xxx_impl(self, ctx, ...)`
      - A public `@staticmethod @function_tool` with `ctx` as first param
        that fetches the instance from `ctx.context.get_value("emotion_tools_instance")`
        and calls `_xxx_impl(...)`.

    You'll need to ensure something like:
        ctx.context.set_value("emotion_tools_instance", self)
    is done so that the static methods can retrieve the instance.
    """

    def __init__(self, emotion_system):
        """
        Initialize with reference to the emotion system.

        Args:
            emotion_system: The emotion system to interact with
        """
        self.neurochemicals = emotion_system.neurochemicals
        self.emotion_derivation_rules = emotion_system.emotion_derivation_rules

        # If there's a method apply_chemical_decay in emotion_system, store it
        if hasattr(emotion_system, "apply_chemical_decay"):
            self.apply_chemical_decay = emotion_system.apply_chemical_decay
        elif hasattr(emotion_system, "_apply_chemical_decay"):
            self.apply_chemical_decay = emotion_system._apply_chemical_decay
        else:
            self.apply_chemical_decay = None

        # Index emotion rules for faster lookups
        self._emotion_rule_index = self._index_emotion_rules()
        self._valence_arousal_map = self._create_valence_arousal_map()

    def _index_emotion_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create an indexed version of emotion rules for faster lookups.
        """
        chemical_rules: Dict[str, List[Dict[str, Any]]] = {}
        for chemical in {
            chem
            for rule in self.emotion_derivation_rules
            for chem in rule.get("chemical_conditions", {})
        }:
            chemical_rules[chemical] = [
                rule for rule in self.emotion_derivation_rules
                if chemical in rule.get("chemical_conditions", {})
            ]
        return chemical_rules

    def _create_valence_arousal_map(self) -> Dict[str, Tuple[float, float]]:
        """
        Create a mapping of emotions to their valence and arousal values.
        """
        return {
            rule["emotion"]: (
                rule.get("valence", 0.0),
                rule.get("arousal", 0.5)
            )
            for rule in self.emotion_derivation_rules
        }

    # -------------------------------------------------------------------------
    # 1) derive_emotional_state
    # -------------------------------------------------------------------------
    async def _derive_emotional_state_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, float]:
        """
        Actual implementation that uses 'self' to derive emotional state from neurochemicals.
        """
        with function_span("derive_emotional_state"):
            # Get current chemical levels - use cached if available
            cached_state = ctx.context.get_cached_neurochemicals()
            if cached_state is not None:
                chemical_levels = cached_state
            else:
                if self.apply_chemical_decay:
                    # Apply decay if needed
                    await self.apply_chemical_decay(ctx)
                chemical_levels = {
                    c: d["value"] for c, d in self.neurochemicals.items()
                }

            emotion_scores: Dict[str, float] = {}
            relevant_emotions = set()
            exceeded_thresholds = {}

            # Phase 1: find relevant emotions by checking chemical thresholds
            for chemical, level in chemical_levels.items():
                if chemical not in self._emotion_rule_index:
                    continue
                for rule in self._emotion_rule_index[chemical]:
                    threshold = rule["chemical_conditions"].get(chemical, 0)
                    if level >= threshold * 0.7:
                        relevant_emotions.add(rule["emotion"])
                        rule_id = rule["emotion"]
                        if rule_id not in exceeded_thresholds:
                            exceeded_thresholds[rule_id] = set()
                        exceeded_thresholds[rule_id].add(chemical)

            # Phase 2: score relevant emotions
            with custom_span(
                "emotion_scoring",
                data={
                    "relevant_emotions": list(relevant_emotions),
                    "chemical_levels": {k: round(v, 2) for k, v in chemical_levels.items()}
                }
            ):
                for rule in self.emotion_derivation_rules:
                    emotion = rule["emotion"]
                    if emotion not in relevant_emotions:
                        continue
                    if emotion not in exceeded_thresholds:
                        continue

                    conditions = rule["chemical_conditions"]
                    rule_weight = rule.get("weight", 1.0)

                    # Check if all relevant chemicals are available
                    if all(chem in chemical_levels for chem in conditions):
                        match_scores = [
                            min(chemical_levels.get(chem, 0) / thres, 1.0)
                            for chem, thres in conditions.items()
                            if thres > 0
                        ]
                        if match_scores:
                            avg_match = sum(match_scores) / len(match_scores)
                            weighted_score = avg_match * rule_weight
                            if weighted_score > 0.1:
                                emotion_scores[emotion] = max(
                                    emotion_scores.get(emotion, 0),
                                    weighted_score
                                )

            if not emotion_scores:
                emotion_scores["Neutral"] = 0.5

            total_intensity = sum(emotion_scores.values())
            if total_intensity > 1.5:
                factor = 1.5 / total_intensity
                emotion_scores = {e: i * factor for e, i in emotion_scores.items()}

            ctx.context.last_emotions = emotion_scores

            # Create a custom span for the final emotional state
            with custom_span(
                "emotional_state",
                data={
                    "emotions": {
                        k: round(v, 2) for k, v in emotion_scores.items() if v > 0.1
                    },
                    "cycle": str(ctx.context.cycle_count),
                    "primary": max(emotion_scores.items(), key=lambda x: x[1])[0]
                    if emotion_scores else "Neutral"
                }
            ):
                return emotion_scores

    @staticmethod
    @function_tool
    async def derive_emotional_state(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, float]:
        """
        Derive emotional state from current neurochemical levels with optimized processing.

        Returns:
            Dictionary of emotion names and intensities
        """
        instance = ctx.context.get_value("emotion_tools_instance")
        if not instance:
            raise UserError("No EmotionTools instance found in context.")

        return await instance._derive_emotional_state_impl(ctx)

    # -------------------------------------------------------------------------
    # 2) get_emotional_state_matrix
    # -------------------------------------------------------------------------
    async def _get_emotional_state_matrix_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext]
    ) -> EmotionalStateMatrix:
        """
        Actual implementation that returns the full emotional state matrix.
        """
        with function_span("get_emotional_state_matrix"):
            # First, apply decay if available
            if self.apply_chemical_decay:
                await self.apply_chemical_decay(ctx)

            # Derive emotions
            emotion_intensities = await self._derive_emotional_state_impl(ctx)

            # Use precomputed valence/arousal map
            emotion_valence_map = self._valence_arousal_map

            # Primary emotion
            if emotion_intensities:
                primary_name, primary_intensity = max(emotion_intensities.items(), key=lambda x: x[1])
            else:
                primary_name, primary_intensity = ("Neutral", 0.5)

            primary_valence, primary_arousal = emotion_valence_map.get(primary_name, (0.0, 0.5))

            secondary_emotions: Dict[str, DerivedEmotion] = {}

            with custom_span(
                "calculate_emotion_matrix",
                data={
                    "primary": primary_name,
                    "intensity": primary_intensity,
                    "secondary_count": len([
                        e for e, i in emotion_intensities.items()
                        if e != primary_name and i > 0.2
                    ])
                }
            ):
                for emotion, intensity in emotion_intensities.items():
                    if emotion != primary_name and intensity > 0.2:
                        valence, arousal = emotion_valence_map.get(emotion, (0.0, 0.5))
                        secondary_emotions[emotion] = DerivedEmotion(
                            name=emotion,
                            intensity=intensity,
                            valence=valence,
                            arousal=arousal
                        )

                total_intensity = primary_intensity + sum(e.intensity for e in secondary_emotions.values())
                if total_intensity > 0:
                    valence_values = [primary_valence] + [e.valence for e in secondary_emotions.values()]
                    arousal_values = [primary_arousal] + [e.arousal for e in secondary_emotions.values()]
                    intensity_weights = [primary_intensity] + [e.intensity for e in secondary_emotions.values()]

                    overall_valence = EmotionalToolUtils.calculate_weighted_average(valence_values, intensity_weights)
                    overall_arousal = EmotionalToolUtils.calculate_weighted_average(arousal_values, intensity_weights)
                else:
                    overall_valence = 0.0
                    overall_arousal = 0.5

                overall_valence = EmotionalToolUtils.normalize_value(overall_valence, -1.0, 1.0)
                overall_arousal = EmotionalToolUtils.normalize_value(overall_arousal, 0.0, 1.0)

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

            # Record primary emotion
            ctx.context.record_emotion(primary_name, primary_intensity)

            with custom_span(
                "emotional_state_matrix",
                data={
                    "primary": primary_name,
                    "intensity": round(primary_intensity, 2),
                    "valence": round(overall_valence, 2),
                    "arousal": round(overall_arousal, 2),
                    "secondary_count": len(secondary_emotions),
                    "cycle": str(ctx.context.cycle_count),
                    "valence_category": EmotionValence.from_value(overall_valence),
                    "arousal_category": EmotionArousal.from_value(overall_arousal)
                }
            ):
                return state_matrix

    @staticmethod
    @function_tool
    async def get_emotional_state_matrix(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> EmotionalStateMatrix:
        """
        Get the full emotional state matrix derived from neurochemicals with optimized calculations.

        Returns:
            Emotional state matrix with primary and secondary emotions
        """
        instance = ctx.context.get_value("emotion_tools_instance")
        if not instance:
            raise UserError("No EmotionTools instance found in context.")
        return await instance._get_emotional_state_matrix_impl(ctx)

    # -------------------------------------------------------------------------
    # 3) analyze_text_sentiment
    # -------------------------------------------------------------------------
    async def _analyze_text_sentiment_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext],
        text: str
    ) -> TextAnalysisOutput:
        """
        Actual implementation that analyzes the emotional content of text.
        """
        with function_span("analyze_text_sentiment", input=str(text)[:100]):
            with trace(
                workflow_name="Text_Sentiment_Analysis",
                trace_id=gen_trace_id(),
                metadata={
                    "text_length": len(text),
                    "cycle": str(ctx.context.cycle_count),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            ):
                text_lower = text.lower()
                words = set(text_lower.split())

                emotional_categories = {
                    "positive": {
                        "nyxamine": {
                            "happy", "good", "great", "love", "like", "fun", "enjoy",
                            "curious", "interested", "pleasure", "delight", "joy", "excited",
                            "awesome", "excellent", "wonderful", "amazing", "fantastic", "terrific"
                        }
                    },
                    "calm": {
                        "seranix": {
                            "calm", "peaceful", "relaxed", "content", "satisfied", "gentle",
                            "quiet", "serene", "tranquil", "composed", "patient", "steady",
                            "stable", "balanced", "harmonious", "comfortable"
                        }
                    },
                    "social": {
                        "oxynixin": {
                            "trust", "close", "together", "bond", "connect", "loyal", "friend",
                            "relationship", "intimate", "attachment", "connection", "team",
                            "family", "care", "support", "help", "understanding", "empathy"
                        }
                    },
                    "negative": {
                        "cortanyx": {
                            "worried", "scared", "afraid", "nervous", "stressed", "sad", "sorry",
                            "angry", "upset", "frustrated", "anxious", "distressed", "hurt",
                            "pain", "suffering", "miserable", "terrible", "awful", "horrible"
                        }
                    },
                    "arousing": {
                        "adrenyx": {
                            "excited", "alert", "surprised", "wow", "amazing", "intense",
                            "sudden", "quick", "shock", "unexpected", "startled", "astonished",
                            "astounded", "urgent", "emergency", "crisis", "danger", "important",
                            "critical"
                        }
                    }
                }

                intensifiers = {
                    "very", "extremely", "incredibly", "so", "deeply", "absolutely",
                    "truly", "utterly", "completely", "totally", "highly", "exceptionally",
                    "remarkably", "extraordinarily", "insanely", "super", "immensely"
                }
                negators = {
                    "not", "never", "no", "none", "neither", "nor", "hardly", "barely",
                    "scarcely", "seldom", "rarely"
                }

                has_negation = any(negator in words for negator in negators)
                exclamation_count = text.count("!")
                question_count = text.count("?")

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
                    chemical_impacts: Dict[str, float] = {}

                    for category, chemicals in emotional_categories.items():
                        for chem, word_set in chemicals.items():
                            matches = words.intersection(word_set)
                            if matches:
                                base_impact = min(0.5, len(matches) * 0.1)
                                if has_negation:
                                    # invert chemical if negative => positive or positive => negative
                                    if category in ["positive", "calm", "social"]:
                                        chem = "cortanyx"
                                    elif category == "negative":
                                        chem = "nyxamine"
                                chemical_impacts[chem] = max(
                                    chemical_impacts.get(chem, 0),
                                    base_impact
                                )

                phrase_patterns = [
                    ({"thank you", "thanks", "appreciate"}, {"nyxamine": 0.3}),
                    ({"congratulations", "congrats", "well done"}, {"nyxamine": 0.4, "oxynixin": 0.2}),
                    ({"miss you", "thinking of you"}, {"oxynixin": 0.4}),

                    ({"go away", "leave me alone"}, {"cortanyx": 0.3}),
                    ({"help me", "please help"}, {"cortanyx": 0.2, "adrenyx": 0.3}),
                    ({"angry with", "mad at"}, {"cortanyx": 0.4}),

                    ({"can't wait", "looking forward"}, {"adrenyx": 0.3, "nyxamine": 0.2}),
                    ({"hurry", "quickly", "emergency"}, {"adrenyx": 0.5}),
                ]

                with custom_span("phrase_pattern_detection"):
                    for pattern_words, impacts in phrase_patterns:
                        if any(phrase in text_lower for phrase in pattern_words):
                            for chem, impact in impacts.items():
                                chemical_impacts[chem] = max(
                                    chemical_impacts.get(chem, 0),
                                    impact
                                )

                if exclamation_count > 0:
                    chemical_impacts["adrenyx"] = max(
                        chemical_impacts.get("adrenyx", 0),
                        min(0.5, exclamation_count * 0.15)
                    )
                if question_count > 0:
                    chemical_impacts["nyxamine"] = max(
                        chemical_impacts.get("nyxamine", 0),
                        min(0.3, question_count * 0.1)
                    )

                intensifier_count = len(words.intersection(intensifiers))
                if intensifier_count > 0:
                    intensity_multiplier = 1.0 + (intensifier_count * 0.2)
                    chemical_impacts = {
                        k: min(1.0, v * intensity_multiplier)
                        for k, v in chemical_impacts.items()
                    }

                if not chemical_impacts:
                    chemical_impacts = {"nyxamine": 0.1, "adrenyx": 0.1}

                ctx.context.set_value("last_text_analysis", {
                    "text": text[:100],
                    "chemicals": chemical_impacts,
                    "timestamp": datetime.datetime.now().isoformat()
                })

                with custom_span("temporary_chemical_state"):
                    temp_chemicals = {
                        c: {
                            "value": min(1.0, self.neurochemicals[c]["value"] + chemical_impacts.get(c, 0)),
                            "baseline": self.neurochemicals[c]["baseline"],
                            "decay_rate": self.neurochemicals[c]["decay_rate"]
                        }
                        for c in self.neurochemicals
                    }
                    for chemical in temp_chemicals:
                        temp_chemicals[chemical]["value"] = EmotionalToolUtils.normalize_value(
                            temp_chemicals[chemical]["value"]
                        )

                with custom_span("derive_emotions_from_text"):
                    chemical_levels = {c: d["value"] for c, d in temp_chemicals.items()}
                    derived_emotions: Dict[str, float] = {}
                    valence_sum = 0.0
                    intensity_sum = 0.0

                    for rule in self.emotion_derivation_rules:
                        conditions = rule["chemical_conditions"]
                        emotion = rule["emotion"]
                        rule_weight = rule.get("weight", 1.0)

                        match_scores = [
                            min(chemical_levels.get(ch, 0) / th, 1.0)
                            for ch, th in conditions.items()
                            if ch in chemical_levels and th > 0
                        ]
                        if match_scores and len(match_scores) == len(conditions):
                            avg_score = sum(match_scores) / len(match_scores)
                            weighted_score = avg_score * rule_weight
                            if weighted_score > 0.1:
                                derived_emotions[emotion] = max(
                                    derived_emotions.get(emotion, 0),
                                    weighted_score
                                )
                                valence_sum += rule.get("valence", 0.0) * weighted_score
                                intensity_sum += weighted_score

                if not derived_emotions:
                    derived_emotions["Neutral"] = 0.5
                    intensity_sum = 0.5

                dominant_emotion = max(derived_emotions.items(), key=lambda x: x[1]) if derived_emotions else ("Neutral", 0.5)
                intensity = intensity_sum / len(derived_emotions) if derived_emotions else 0.5
                valence = valence_sum / intensity_sum if intensity_sum > 0 else 0.0

                analysis = TextAnalysisOutput(
                    chemicals_affected=chemical_impacts,
                    derived_emotions=derived_emotions,
                    dominant_emotion=dominant_emotion[0],
                    intensity=intensity,
                    valence=valence
                )

                with custom_span(
                    "text_sentiment_result",
                    data={
                        "text_length": len(text),
                        "dominant_emotion": dominant_emotion[0],
                        "intensity": round(intensity, 2),
                        "valence": round(valence, 2),
                        "chemicals": {k: round(v, 2) for k, v in chemical_impacts.items()},
                        "cycle": str(ctx.context.cycle_count),
                        "valence_category": EmotionValence.from_value(valence),
                        "arousal_category": EmotionArousal.from_value(intensity)
                    }
                ):
                    return analysis

    @staticmethod
    @function_tool
    async def analyze_text_sentiment(
        ctx: RunContextWrapper[EmotionalContext],
        text: str
    ) -> TextAnalysisOutput:
        """
        Analyze the emotional content of text with improved efficiency and accuracy.

        Args:
            text: Text to analyze

        Returns:
            Analysis of emotional content
        """
        instance = ctx.context.get_value("emotion_tools_instance")
        if not instance:
            raise UserError("No EmotionTools instance found in context.")
        return await instance._analyze_text_sentiment_impl(ctx, text)

    # -------------------------------------------------------------------------
    # 4) get_emotion_trends
    # -------------------------------------------------------------------------
    async def _get_emotion_trends_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext],
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Actual implementation that returns emotional trends over time.
        """
        with function_span("get_emotion_trends"):
            emotion_history = ctx.context.get_emotion_trends(limit=limit)
            if not emotion_history:
                return {
                    "message": "Not enough emotion history data",
                    "trends": {}
                }

            trends = {}
            with custom_span(
                "emotion_trend_analysis",
                data={
                    "emotions_analyzed": list(emotion_history.keys()),
                    "history_points": max(len(data) for data in emotion_history.values()) if emotion_history else 0,
                    "limit": limit
                }
            ):
                for emotion, data_points in emotion_history.items():
                    if len(data_points) < 2:
                        continue
                    intensities = [p["intensity"] for p in data_points]
                    start_value = intensities[0]
                    end_value = intensities[-1]
                    change = end_value - start_value

                    if abs(change) < 0.1:
                        direction = "stable"
                    elif change > 0:
                        direction = "increasing"
                    else:
                        direction = "decreasing"

                    volatility = sum(abs(intensities[i] - intensities[i - 1]) for i in range(1, len(intensities))) / (len(intensities) - 1)
                    trends[emotion] = {
                        "direction": direction,
                        "change": change,
                        "volatility": volatility,
                        "current_value": end_value,
                        "points": len(data_points)
                    }

            if trends:
                avg_volatility = sum(t["volatility"] for t in trends.values()) / len(trends)
                stability = 1.0 - min(1.0, avg_volatility * 5.0)
                stability_rating = (
                    "very stable" if stability > 0.8 else
                    "stable" if stability > 0.6 else
                    "somewhat unstable" if stability > 0.4 else
                    "unstable" if stability > 0.2 else
                    "very unstable"
                )
            else:
                stability = 0.5
                stability_rating = "unknown"

            with custom_span(
                "emotion_trends_result",
                data={
                    "emotions_analyzed": list(trends.keys()),
                    "stability": round(stability, 2),
                    "stability_rating": stability_rating,
                    "cycle": str(ctx.context.cycle_count)
                }
            ):
                return {
                    "trends": trends,
                    "stability": stability,
                    "stability_rating": stability_rating,
                    "analysis_time": datetime.datetime.now().isoformat()
                }

    @staticmethod
    @function_tool
    async def get_emotion_trends(
        ctx: RunContextWrapper[EmotionalContext],
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get emotional trends over time.

        Args:
            limit: Maximum number of historical data points to consider

        Returns:
            Dictionary of emotion trends
        """
        instance = ctx.context.get_value("emotion_tools_instance")
        if not instance:
            raise UserError("No EmotionTools instance found in context.")
        return await instance._get_emotion_trends_impl(ctx, limit)
