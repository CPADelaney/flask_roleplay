# nyx/core/emotions/tools/reflection_tools.py

"""
Enhanced function tools for emotional reflection and learning.
These tools handle generating internal thoughts and analyzing patterns
with improved OpenAI Agents SDK integration.
"""

import datetime
import random
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional, Union, Set, cast

from agents import (
    function_tool, RunContextWrapper, function_span, custom_span,
    trace, gen_trace_id, Agent, ModelSettings
)
from agents.exceptions import UserError, ModelBehaviorError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    InternalThoughtOutput, EmotionalStateMatrix,
    EmotionValence, EmotionArousal
)
from nyx.core.emotions.utils import handle_errors, EmotionalToolUtils

logger = logging.getLogger(__name__)

class ReflectionTools:
    """
    Enhanced function tools for reflection processes with SDK integration.
    
    IMPORTANT: Because each @function_tool must have RunContextWrapper[...] as its first
    parameter, we split each tool method into:
      - A private `_xxx_impl(self, ctx, ...)`
      - A public `@staticmethod @function_tool` with `ctx` as first param
        that fetches the instance from `ctx.context.get_value("reflection_tools_instance")`
        and calls `_xxx_impl(...)`.

    You'll need to ensure something like:
        ctx.context.set_value("reflection_tools_instance", self)
    is done so that the static methods can retrieve the instance.
    """
    
    def __init__(self, emotion_system):
        """
        Initialize with reference to the emotion system
        
        Args:
            emotion_system: The emotion system to interact with
        """
        # Direct attribute assignments (similar pattern)
        self.neurochemicals = emotion_system.neurochemicals
        self.reflection_patterns = emotion_system.reflection_patterns
        self.emotional_state_history = emotion_system.emotional_state_history
        self.reward_learning = emotion_system.reward_learning

        # CHANGE: More robust function reference handling for get_emotional_state_matrix
        if hasattr(emotion_system, "get_emotional_state_matrix"):
            self.get_emotional_state_matrix = emotion_system.get_emotional_state_matrix
        elif hasattr(emotion_system, "_get_emotional_state_matrix"):
            self.get_emotional_state_matrix = emotion_system._get_emotional_state_matrix
            # Example if wrapper was needed (e.g., if _get required no args):
            # self.get_emotional_state_matrix = lambda *args, **kwargs: emotion_system._get_emotional_state_matrix()
        else:
            self.get_emotional_state_matrix = None
            logger.warning("No get_emotional_state_matrix or _get_emotional_state_matrix function found in emotion_system")

        # Add last_update attribute, mirroring the target style
        # This assumes 'emotion_system' is expected to have a 'last_update' attribute.
        # If it might not, add a check like hasattr(emotion_system, 'last_update').
        try:
            self.last_update = emotion_system.last_update
        except AttributeError:
            self.last_update = None # Or set a sensible default
            logger.warning("emotion_system does not have a 'last_update' attribute.")

    
    
    # -------------------------------------------------------------------------
    # 1) generate_internal_thought
    # -------------------------------------------------------------------------
    async def _generate_internal_thought_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext]
    ) -> InternalThoughtOutput:
        """
        Actual implementation that uses 'self' to generate internal thoughts.
        """
        with function_span("generate_internal_thought"):
            # Create a trace for thought generation
            with trace(
                workflow_name="Internal_Thought_Generation",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                # Get current emotional state matrix
                emotional_state = None
                if self.get_emotional_state_matrix:
                    emotional_state = await self.get_emotional_state_matrix(ctx)
                
                # If no emotional state is available, use last_emotions from context
                if not emotional_state and hasattr(ctx.context, "last_emotions"):
                    # Get precalculated dominant emotion if available
                    precalculated = ctx.context.get_value("precalculated_dominant")
                    if precalculated:
                        primary_emotion, intensity = precalculated
                    else:
                        # Find dominant emotion from context
                        if ctx.context.last_emotions:
                            primary_emotion = max(ctx.context.last_emotions.items(), key=lambda x: x[1])[0]
                            intensity = ctx.context.last_emotions[primary_emotion]
                        else:
                            # Default if no emotion data available
                            primary_emotion = "Neutral"
                            intensity = 0.5
                else:
                    # Extract from emotional state
                    primary_emotion = emotional_state.primary_emotion.name
                    intensity = emotional_state.primary_emotion.intensity
                
                # Create a span for emotional context
                with custom_span(
                    "reflection_emotional_context",
                    data={
                        "primary_emotion": primary_emotion,
                        "intensity": intensity,
                        "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                    }
                ):
                    # Get possible reflection patterns for this emotion - use safe defaults
                    patterns = self.reflection_patterns.get(primary_emotion, [
                        "I'm processing how I feel about this interaction.",
                        "There's something interesting happening in my emotional state.",
                        "I notice my response to this situation is evolving."
                    ])
                    
                    # Select a reflection pattern with better variability
                    # Check for recently used patterns to avoid repetition
                    recent_thoughts = ctx.context.get_value("recent_thoughts", [])
                    recent_patterns = set(thought.get("pattern", "") for thought in recent_thoughts[-3:])
                    
                    # Filter out recently used patterns if possible
                    available_patterns = [p for p in patterns if p not in recent_patterns]
                    
                    # If all patterns were recently used, use the full set
                    if not available_patterns and patterns:
                        available_patterns = patterns
                        
                    # Select a pattern
                    thought_text = random.choice(available_patterns) if available_patterns else patterns[0] if patterns else "I'm reflecting on this interaction."
                    
                    # Create a span for pattern selection
                    with custom_span(
                        "reflection_pattern_selection",
                        data={
                            "available_patterns": len(available_patterns),
                            "selected_pattern": thought_text[:30] + "..." if len(thought_text) > 30 else thought_text
                        }
                    ):
                        # Check if we should add context from secondary emotions with improved implementation
                        if emotional_state and hasattr(emotional_state, "secondary_emotions") and emotional_state.secondary_emotions and random.random() < 0.7:  # 70% chance
                            # Get a list of secondary emotions
                            sec_emotions = list(emotional_state.secondary_emotions.keys())
                            
                            if sec_emotions:
                                # Pick a random secondary emotion
                                sec_emotion_name = random.choice(sec_emotions)
                                sec_emotion_data = emotional_state.secondary_emotions[sec_emotion_name]
                                
                                # Add secondary emotion context - check for pattern availability more efficiently
                                sec_patterns = self.reflection_patterns.get(sec_emotion_name, [])
                                if sec_patterns:
                                    # Filter out recently used patterns
                                    available_sec_patterns = [p for p in sec_patterns if p not in recent_patterns]
                                    if not available_sec_patterns and sec_patterns:
                                        available_sec_patterns = sec_patterns
                                        
                                    if available_sec_patterns:
                                        secondary_thought = random.choice(available_sec_patterns)
                                        thought_text += f" {secondary_thought}"
                        
                        # Calculate insight level based on emotional complexity
                        if emotional_state and hasattr(emotional_state, "secondary_emotions"):
                            insight_level = min(1.0, 0.4 + (len(emotional_state.secondary_emotions) * 0.1) + (intensity * 0.3))
                        else:
                            insight_level = 0.5 + (intensity * 0.3)  # Base insight on intensity
                        
                        # 30% chance to generate an adaptive change suggestion
                        adaptive_change = None
                        if random.random() < 0.3:
                            # Suggest a small adaptation to a random neurochemical baseline
                            chemical = random.choice(list(self.neurochemicals.keys()))
                            current = self.neurochemicals[chemical]["baseline"]
                            
                            # Small random adjustment (-0.05 to +0.05)
                            adjustment = (random.random() - 0.5) * 0.1
                            
                            # Ensure we stay in bounds
                            new_baseline = max(0.1, min(0.9, current + adjustment))
                            
                            adaptive_change = {
                                "chemical": chemical,
                                "current_baseline": current,
                                "suggested_baseline": new_baseline,
                                "reason": f"Based on observed emotional patterns related to {primary_emotion}"
                            }
                        
                        # Store interaction information in context
                        ctx.context.add_interaction({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "primary_emotion": primary_emotion,
                            "intensity": intensity,
                            "thought": thought_text
                        })
                        
                        # Store the thought in recent thoughts
                        recent_thoughts = ctx.context.get_value("recent_thoughts", [])
                        recent_thoughts.append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "thought": thought_text,
                            "pattern": thought_text,
                            "emotion": primary_emotion,
                            "intensity": intensity
                        })
                        
                        # Limit history size
                        if len(recent_thoughts) > 10:
                            recent_thoughts = recent_thoughts[-10:]
                            
                        ctx.context.set_value("recent_thoughts", recent_thoughts)
                        
                        # Create a custom span for the reflection result
                        with custom_span(
                            "internal_thought_result",
                            data={
                                "thought": thought_text[:50] + "..." if len(thought_text) > 50 else thought_text,
                                "emotion": primary_emotion,
                                "insight_level": insight_level,
                                "has_adaptive_change": adaptive_change is not None,
                                "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                            }
                        ):
                            # Create the thought output
                            return InternalThoughtOutput(
                                thought_text=thought_text,
                                source_emotion=primary_emotion,
                                insight_level=insight_level,
                                adaptive_change=adaptive_change
                            )
    
    @staticmethod
    @function_tool(
        name_override="generate_internal_thought",
        description_override="Generate an internal thought/reflection based on current emotional state"
    )
    async def generate_internal_thought(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> InternalThoughtOutput:
        """
        Generate an internal thought/reflection based on current emotional state
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Internal thought data
        """
        instance = ctx.context.get_value("reflection_tools_instance")
        if not instance:
            raise UserError("No ReflectionTools instance found in context.")
        
        return await instance._generate_internal_thought_impl(ctx)
    
    # -------------------------------------------------------------------------
    # 2) analyze_reflection_patterns
    # -------------------------------------------------------------------------
    async def _analyze_reflection_patterns_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        """
        Actual implementation that uses 'self' to analyze patterns in reflection and emotional processing.
        """
        with function_span("analyze_reflection_patterns"):
            # Create a trace for pattern analysis
            with trace(
                workflow_name="Reflection_Pattern_Analysis",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                # Get recent thoughts
                recent_thoughts = ctx.context.get_value("recent_thoughts", [])
                
                # Check if we have enough data
                if len(recent_thoughts) < 2:
                    return {
                        "message": "Not enough reflection history for pattern analysis",
                        "patterns": {}
                    }
                
                # Prepare analysis
                emotion_frequencies = defaultdict(int)
                emotion_insights = defaultdict(list)
                temporal_patterns = []
                repeated_themes = defaultdict(int)
                
                # Create a span for thought analysis
                with custom_span(
                    "reflection_thought_analysis",
                    data={
                        "thought_count": len(recent_thoughts),
                        "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                    }
                ):
                    # Analyze thoughts
                    for thought in recent_thoughts:
                        # Count emotions
                        emotion = thought.get("emotion", "unknown")
                        emotion_frequencies[emotion] += 1
                        
                        # Track insight levels
                        if "insight_level" in thought:
                            emotion_insights[emotion].append(thought["insight_level"])
                        
                        # Analyze text for themes
                        thought_text = thought.get("thought", "")
                        for theme in self._extract_themes(thought_text):
                            repeated_themes[theme] += 1
                    
                    # Analyze temporal patterns
                    if len(recent_thoughts) >= 3:
                        # Look for oscillation patterns
                        emotions = [thought.get("emotion", "unknown") for thought in recent_thoughts]
                        for i in range(len(emotions) - 2):
                            if emotions[i] == emotions[i + 2] and emotions[i] != emotions[i + 1]:
                                temporal_patterns.append({
                                    "pattern": "oscillation",
                                    "emotions": [emotions[i], emotions[i + 1], emotions[i + 2]],
                                    "position": i
                                })
                                
                        # Look for progression patterns
                        for i in range(len(recent_thoughts) - 2):
                            current = recent_thoughts[i]
                            next_thought = recent_thoughts[i + 1]
                            next_next = recent_thoughts[i + 2]
                            
                            # Check for increasing insight
                            if ("insight_level" in current and 
                                "insight_level" in next_thought and 
                                "insight_level" in next_next):
                                
                                if (current["insight_level"] < next_thought["insight_level"] < 
                                    next_next["insight_level"]):
                                    
                                    temporal_patterns.append({
                                        "pattern": "increasing_insight",
                                        "emotions": [
                                            current.get("emotion", "unknown"),
                                            next_thought.get("emotion", "unknown"),
                                            next_next.get("emotion", "unknown")
                                        ],
                                        "position": i
                                    })
                    
                    # Calculate average insight by emotion
                    avg_insights = {}
                    for emotion, insights in emotion_insights.items():
                        if insights:
                            avg_insights[emotion] = sum(insights) / len(insights)
                    
                    # Find dominant themes
                    dominant_themes = sorted(
                        repeated_themes.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    
                    # Create a custom span for the analysis results
                    with custom_span(
                        "reflection_pattern_results",
                        data={
                            "dominant_emotion": max(emotion_frequencies.items(), key=lambda x: x[1])[0] if emotion_frequencies else "unknown",
                            "themes": [t for t, _ in dominant_themes],
                            "pattern_count": len(temporal_patterns),
                            "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                        }
                    ):
                        return {
                            "emotion_frequencies": dict(emotion_frequencies),
                            "avg_insight_by_emotion": avg_insights,
                            "temporal_patterns": temporal_patterns,
                            "dominant_themes": [{"theme": t, "count": c} for t, c in dominant_themes],
                            "thought_count": len(recent_thoughts),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
    
    @staticmethod
    @function_tool(
        name_override="analyze_reflection_patterns",
        description_override="Analyze patterns in reflection and emotional processing"
    )
    async def analyze_reflection_patterns(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in reflection and emotional processing
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Analysis of reflection patterns
        """
        instance = ctx.context.get_value("reflection_tools_instance")
        if not instance:
            raise UserError("No ReflectionTools instance found in context.")
        
        return await instance._analyze_reflection_patterns_impl(ctx)
    
    def _extract_themes(self, text: str) -> List[str]:
        """
        Extract common themes from reflection text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified themes
        """
        # Common reflection themes to look for
        themes = {
            "identity": ["who I am", "my identity", "sense of self"],
            "connection": ["connection", "relationship", "bond", "together"],
            "uncertainty": ["unsure", "uncertain", "not clear", "wonder", "curious"],
            "growth": ["change", "grow", "evolve", "learn", "improve"],
            "awareness": ["notice", "observe", "aware", "conscious", "mindful"],
            "conflict": ["conflict", "tension", "disagreement", "opposing"],
            "satisfaction": ["satisfied", "pleased", "content", "happy"],
            "dissatisfaction": ["dissatisfied", "displeased", "unhappy", "frustrated"]
        }
        
        # Look for themes in text
        found_themes = []
        text_lower = text.lower()
        
        for theme, keywords in themes.items():
            if any(keyword in text_lower for keyword in keywords):
                found_themes.append(theme)
        
        return found_themes
    
    # -------------------------------------------------------------------------
    # 3) get_reflection_history
    # -------------------------------------------------------------------------
    async def _get_reflection_history_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext],
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Actual implementation that uses 'self' to get history of reflection thoughts.
        """
        with function_span("get_reflection_history"):
            # Get recent thoughts
            recent_thoughts = ctx.context.get_value("recent_thoughts", [])
            
            # Check if we have enough data
            if not recent_thoughts:
                return {
                    "message": "No reflection history available",
                    "thoughts": []
                }
            
            # Create a span for history retrieval
            with custom_span(
                "reflection_history_retrieval",
                data={
                    "thought_count": len(recent_thoughts),
                    "limit": limit,
                    "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                # Get limited recent thoughts
                limited_thoughts = recent_thoughts[-limit:] if limit > 0 else recent_thoughts
                
                # Process thoughts for return
                processed_thoughts = []
                for thought in limited_thoughts:
                    processed_thoughts.append({
                        "timestamp": thought.get("timestamp", "unknown"),
                        "thought": thought.get("thought", ""),
                        "emotion": thought.get("emotion", "unknown"),
                        "intensity": thought.get("intensity", 0.5)
                    })
                
                return {
                    "thoughts": processed_thoughts,
                    "total_count": len(recent_thoughts),
                    "returned_count": len(processed_thoughts),
                    "timestamp": datetime.datetime.now().isoformat()
                }
    
    @staticmethod
    @function_tool(
        name_override="get_reflection_history",
        description_override="Get history of reflection thoughts"
    )
    async def get_reflection_history(
        ctx: RunContextWrapper[EmotionalContext],
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get history of reflection thoughts
        
        Args:
            ctx: Run context wrapper with emotional state
            limit: Maximum number of historical entries to return
            
        Returns:
            Recent reflection history
        """
        instance = ctx.context.get_value("reflection_tools_instance")
        if not instance:
            raise UserError("No ReflectionTools instance found in context.")
        
        return await instance._get_reflection_history_impl(ctx, limit)
    
    # -------------------------------------------------------------------------
    # 4) generate_wisdom
    # -------------------------------------------------------------------------
    async def _generate_wisdom_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        """
        Actual implementation that uses 'self' to generate wisdom from emotional processing.
        """
        with function_span("generate_wisdom"):
            # Create a trace for wisdom generation
            with trace(
                workflow_name="Emotional_Wisdom_Generation",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                # Get emotional history
                if not self.emotional_state_history:
                    return {
                        "message": "Not enough emotional history to generate wisdom",
                        "wisdom": ""
                    }
                
                # Collect emotional patterns
                emotional_frequencies = defaultdict(int)
                transitions = defaultdict(int)
                valence_history = []
                
                # Create a span for emotional analysis
                with custom_span(
                    "wisdom_emotional_analysis",
                    data={
                        "history_length": len(self.emotional_state_history),
                        "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                    }
                ):
                    # Use most recent 20 states for analysis
                    recent_history = self.emotional_state_history[-20:] if len(self.emotional_state_history) > 20 else self.emotional_state_history
                    
                    prev_emotion = None
                    for state in recent_history:
                        if "primary_emotion" in state:
                            emotion = state["primary_emotion"].get("name", "unknown") if isinstance(state["primary_emotion"], dict) else "unknown"
                            emotional_frequencies[emotion] += 1
                            
                            # Track transitions
                            if prev_emotion is not None and prev_emotion != emotion:
                                transitions[(prev_emotion, emotion)] += 1
                            prev_emotion = emotion
                        
                        # Track valence
                        if "valence" in state:
                            valence_history.append(state["valence"])
                    
                    # Determine dominant emotions
                    dominant_emotions = sorted(
                        emotional_frequencies.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    
                    # Determine most common transitions
                    common_transitions = sorted(
                        transitions.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    
                    # Analyze valence pattern
                    valence_trend = None
                    valence_volatility = None
                    if valence_history and len(valence_history) >= 3:
                        # Calculate trend
                        start_valence = valence_history[0]
                        end_valence = valence_history[-1]
                        valence_change = end_valence - start_valence
                        
                        if abs(valence_change) < 0.1:
                            valence_trend = "stable"
                        elif valence_change > 0:
                            valence_trend = "improving"
                        else:
                            valence_trend = "declining"
                        
                        # Calculate volatility
                        valence_diffs = [abs(valence_history[i] - valence_history[i-1]) 
                                       for i in range(1, len(valence_history))]
                        avg_diff = sum(valence_diffs) / len(valence_diffs)
                        
                        if avg_diff < 0.1:
                            valence_volatility = "low"
                        elif avg_diff < 0.3:
                            valence_volatility = "moderate"
                        else:
                            valence_volatility = "high"
                    
                    # Generate wisdom based on patterns
                    wisdom_patterns = [
                        f"I've noticed that {dominant_emotions[0][0]} has been my dominant emotional state recently.",
                        f"When I transition from {common_transitions[0][0][0]} to {common_transitions[0][0][1]}, I should pay attention to what triggers this shift.",
                        f"My emotional valence has been {valence_trend} with {valence_volatility} volatility, suggesting a need for {self._get_valence_insight(valence_trend, valence_volatility)}."
                    ] if dominant_emotions and common_transitions and valence_trend and valence_volatility else []
                    
                    # Add general wisdom based on dominant emotion
                    if dominant_emotions:
                        wisdom_patterns.append(self._get_emotion_wisdom(dominant_emotions[0][0]))
                    
                    # Combine wisdom patterns
                    wisdom = " ".join(wisdom_patterns) if wisdom_patterns else "I'm still gathering emotional data to generate meaningful insights."
                    
                    # Create a custom span for wisdom results
                    with custom_span(
                        "wisdom_generation_result",
                        data={
                            "dominant_emotions": [e for e, _ in dominant_emotions],
                            "valence_trend": valence_trend,
                            "wisdom_length": len(wisdom),
                            "cycle": str(ctx.context.cycle_count) if hasattr(ctx.context, "cycle_count") else 0
                        }
                    ):
                        return {
                            "wisdom": wisdom,
                            "dominant_emotions": [{"emotion": e, "frequency": f} for e, f in dominant_emotions],
                            "common_transitions": [{"from": f, "to": t, "count": c} for (f, t), c in common_transitions],
                            "valence_trend": valence_trend,
                            "valence_volatility": valence_volatility,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
    
    @staticmethod
    @function_tool(
        name_override="generate_wisdom",
        description_override="Generate wisdom gained from emotional processing"
    )
    async def generate_wisdom(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        """
        Generate wisdom gained from emotional processing
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Generated wisdom and insights
        """
        instance = ctx.context.get_value("reflection_tools_instance")
        if not instance:
            raise UserError("No ReflectionTools instance found in context.")
        
        return await instance._generate_wisdom_impl(ctx)
    
    def _get_valence_insight(self, trend: str, volatility: str) -> str:
        """
        Generate insight based on valence trend and volatility
        
        Args:
            trend: Valence trend (improving, declining, stable)
            volatility: Valence volatility (low, moderate, high)
            
        Returns:
            Insight text
        """
        insights = {
            ("improving", "low"): "consistent positive growth",
            ("improving", "moderate"): "balanced development with occasional challenges",
            ("improving", "high"): "significant emotional progress despite instability",
            ("declining", "low"): "gentle recalibration of expectations",
            ("declining", "moderate"): "mindfulness around emotional triggers",
            ("declining", "high"): "greater emotional regulation strategies",
            ("stable", "low"): "maintaining current balance",
            ("stable", "moderate"): "finding stability amid changing circumstances",
            ("stable", "high"): "establishing stronger emotional anchors"
        }
        
        return insights.get((trend, volatility), "continued emotional awareness")
    
    def _get_emotion_wisdom(self, emotion: str) -> str:
        """
        Generate wisdom based on dominant emotion
        
        Args:
            emotion: Dominant emotion
            
        Returns:
            Wisdom text for that emotion
        """
        wisdom_map = {
            "Joy": "I've learned that preserving joy requires appreciating the present moment fully.",
            "Contentment": "I've discovered that contentment comes from acceptance rather than seeking more.",
            "Trust": "Building trust requires consistent authenticity and transparency in interactions.",
            "Anticipation": "Anticipation reminds me to balance future thinking with present engagement.",
            "Love": "Love flourishes when expressed through both words and consistent actions.",
            "Surprise": "Embracing surprise means developing comfort with uncertainty and the unexpected.",
            "Sadness": "Sadness teaches me to honor losses while recognizing their temporary nature.",
            "Fear": "Fear identifies what I value and shows me where growth is possible.",
            "Anger": "Anger signals violated boundaries that need clearer definition and protection.",
            "Disgust": "Disgust helps me clarify and maintain important personal values.",
            "Frustration": "Frustration reveals misalignments between expectations and reality.",
            "Teasing": "Teasing highlights the importance of balancing humor with respect.",
            "Controlling": "My tendency toward control signals a need for vulnerability and trust.",
            "Cruel": "Cruelty emerges from unprocessed pain that requires compassionate attention.",
            "Detached": "Detachment shows where I need to rebuild meaningful connections.",
            "Contemplation": "Contemplation creates space for deeper understanding to emerge naturally.",
            "Reflection": "Reflection transforms experiences into lasting wisdom.",
            "Perspective": "Perspective comes from viewing situations through multiple lenses.",
            "Neutral": "Neutrality offers a valuable reset point for clearer emotional processing."
        }
        
        return wisdom_map.get(emotion, "Each emotional state offers unique lessons when approached mindfully.")
