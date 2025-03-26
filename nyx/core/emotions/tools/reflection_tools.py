# nyx/core/emotions/tools/reflection_tools.py

"""
Function tools for emotional reflection and learning.
These tools handle generating internal thoughts and analyzing patterns.
"""

import datetime
import random
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional

from agents import function_tool, RunContextWrapper, function_span

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    InternalThoughtOutput, EmotionalStateMatrix
)

logger = logging.getLogger(__name__)

class ReflectionTools:
    """Function tools for reflection and learning processes"""
    
    def __init__(self, emotion_system):
        """
        Initialize with reference to the emotion system
        
        Args:
            emotion_system: The emotion system to interact with
        """
        self.neurochemicals = emotion_system.neurochemicals
        self.reflection_patterns = emotion_system.reflection_patterns
        self.emotional_state_history = emotion_system.emotional_state_history
        self.reward_learning = emotion_system.reward_learning
        self.get_emotional_state_matrix = emotion_system._get_emotional_state_matrix
    
    @function_tool
    async def generate_internal_thought(self, ctx: RunContextWrapper[EmotionalContext]) -> InternalThoughtOutput:
        """
        Generate an internal thought/reflection based on current emotional state
        
        Returns:
            Internal thought data
        """
        with function_span("generate_internal_thought"):
            # Get current emotional state matrix
            emotional_state = await self.get_emotional_state_matrix(ctx)
            
            primary_emotion = emotional_state.primary_emotion.name
            intensity = emotional_state.primary_emotion.intensity
            
            # Get possible reflection patterns for this emotion - use safe defaults
            patterns = self.reflection_patterns.get(primary_emotion, [
                "I'm processing how I feel about this interaction.",
                "There's something interesting happening in my emotional state.",
                "I notice my response to this situation is evolving."
            ])
            
            # Select a reflection pattern
            thought_text = random.choice(patterns)
            
            # Check if we should add context from secondary emotions - more efficient implementation
            if emotional_state.secondary_emotions and random.random() < 0.7:  # 70% chance
                # Get a list of secondary emotions
                sec_emotions = list(emotional_state.secondary_emotions.keys())
                
                if sec_emotions:
                    # Pick a random secondary emotion
                    sec_emotion_name = random.choice(sec_emotions)
                    sec_emotion_data = emotional_state.secondary_emotions[sec_emotion_name]
                    
                    # Add secondary emotion context - check for pattern availability more efficiently
                    sec_patterns = self.reflection_patterns.get(sec_emotion_name, [])
                    if sec_patterns:
                        secondary_thought = random.choice(sec_patterns)
                        thought_text += f" {secondary_thought}"
            
            # Calculate insight level based on emotional complexity
            insight_level = min(1.0, 0.4 + (len(emotional_state.secondary_emotions) * 0.1) + (intensity * 0.3))
            
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
            
            return InternalThoughtOutput(
                thought_text=thought_text,
                source_emotion=primary_emotion,
                insight_level=insight_level,
                adaptive_change=adaptive_change
            )
