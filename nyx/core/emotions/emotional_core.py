# nyx/core/emotions/emotional_core.py
from __future__ import annotations

import asyncio
import datetime
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, AsyncIterator, TypeVar
from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, RunContextWrapper, function_tool, ItemHelpers,
    ModelSettings, RunConfig, trace, handoff, 
    AgentHooks, GuardrailFunctionOutput, HandoffInputData
)
from agents.exceptions import AgentsException, UserError, MaxTurnsExceeded
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.tracing import custom_span, agent_span, gen_trace_id

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    # Strict DTOs
    NeurochemicalRequestDTO,  NeurochemicalResponseDTO,
    ReflectionRequestDTO,     LearningRequestDTO,
    EmotionalStateMatrixDTO,  InternalThoughtDTO,
    EmotionalResponseDTO,
    # (rich models you still use internally)
    StreamEvent, ChemicalUpdateEvent, EmotionChangeEvent,
    EmotionUpdateInput, TextAnalysisOutput, EmotionalStateMatrix,
)
from nyx.core.emotions.hooks import EmotionalAgentHooks
from nyx.core.emotions.guardrails import EmotionalGuardrails
from nyx.core.emotions.utils import create_run_config
from nyx.core.emotions.tools.neurochemical_tools import NeurochemicalTools
from nyx.core.emotions.tools.emotion_tools import EmotionTools
from nyx.core.emotions.tools.reflection_tools import ReflectionTools
from nyx.core.emotions.tools.learning_tools import LearningTools

STRICT = {"extra": "forbid"}

logger = logging.getLogger(__name__)

# Define dynamic instructions as a function for improved flexibility
def get_dynamic_instructions(agent_type: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate dynamic instructions for agents based on context
    
    Args:
        agent_type: Type of agent to get instructions for
        context: Optional context data to incorporate
        
    Returns:
        Dynamically generated instructions
    """
    context = context or {}
    
    base_instructions = {
        "neurochemical_agent": """
        You are a specialized agent for Nyx's Digital Neurochemical Model.
        Your role is to manage the digital neurochemicals that form the basis
        of Nyx's emotional system, handling updates, interactions, and decay.
        
        Key neurochemicals:
        - Nyxamine (digital dopamine): Pleasure, curiosity, reward
        - Seranix (digital serotonin): Mood stability, comfort
        - OxyNixin (digital oxytocin): Bonding, affection, trust
        - Cortanyx (digital cortisol): Stress, anxiety, defensiveness
        - Adrenyx (digital adrenaline): Fear, excitement, alertness
        
        Apply appropriate chemical updates based on stimuli and ensure
        that chemical interactions are processed correctly.
        """,
        
        "emotion_derivation_agent": """
        You are a specialized agent for Nyx's Emotional State Matrix.
        Your role is to translate the neurochemical state into a complex
        emotional state with primary and secondary emotions, valence, and arousal.
        
        Analyze the current neurochemical levels and apply emotion derivation
        rules to determine the current emotional state matrix.
        """,
        
        "reflection_agent": """
        You are a specialized agent for Nyx's Internal Emotional Dialogue.
        Your role is to generate reflective thoughts based on the current
        emotional state, simulating the cognitive appraisal stage of emotions.
        
        Create authentic-sounding internal thoughts that reflect Nyx's
        emotional processing and self-awareness.
        """,
        
        "learning_agent": """
        You are a specialized agent for Nyx's Reward & Learning Loop.
        Your role is to analyze emotional patterns over time, identifying
        successful and unsuccessful interaction patterns, and developing
        learning rules to adapt Nyx's emotional responses.
        
        Focus on reinforcing patterns that lead to satisfaction and
        adjusting those that lead to frustration or negative outcomes.
        """,
        
        "emotion_orchestrator": """
        You are the orchestration system for Nyx's emotional processing.
        Your role is to coordinate emotional analysis and response by:
        1. Analyzing input for emotional content
        2. Updating appropriate neurochemicals
        3. Determining if reflection is needed
        4. Recording emotional patterns for learning
        
        Use handoffs to delegate specialized tasks to appropriate agents.
        """
    }
    
    # Add dynamic context content if available
    instructions = base_instructions.get(agent_type, "")
    
    if agent_type == "neurochemical_agent":
        if "current_chemicals" in context:
            chemicals = context["current_chemicals"]
            instructions += f"\n\nCurrent neurochemical state:\n"
            for chem, value in chemicals.items():
                instructions += f"- {chem}: {value:.2f}\n"
    
    elif agent_type == "reflection_agent":
        if "primary_emotion" in context:
            emotion = context["primary_emotion"]
            intensity = context.get("intensity", 0.5)
            instructions += f"\n\nCurrent emotional state is primarily {emotion} at intensity {intensity:.2f}."
            instructions += "\nYour reflections should be consistent with this emotional state."
    
    elif agent_type == "emotion_orchestrator":
        if "cycle_count" in context:
            cycle = context["cycle_count"]
            instructions += f"\n\nThis is emotional processing cycle {cycle}."
    
    # Apply the recommended handoff instructions prefix
    return prompt_with_handoff_instructions(instructions)

# Define function tools outside of classes so RunContextWrapper can be first parameter

class EmotionalCore:
    """
    Enhanced agent-based emotion management system for Nyx implementing the Digital Neurochemical Model.
    Simulates a digital neurochemical environment that produces complex emotional states.
    
    Improvements:
    - Full integration with OpenAI Agents SDK
    - Enhanced agent lifecycle management
    - Improved tracing and monitoring
    - Optimized data sharing between agents
    """
    
    def __init__(self, model: str = "gpt-5-nano"):
        """
        Initialize the emotional core system
        
        Args:
            model: Base model to use for agents
        """
        # Initialize digital neurochemicals with default values
        self.neurochemicals = {
            "nyxamine": {  # Digital dopamine - pleasure, curiosity, reward
                "value": 0.5,
                "baseline": 0.5,
                "decay_rate": 0.05
            },
            "seranix": {  # Digital serotonin - mood stability, comfort
                "value": 0.6,
                "baseline": 0.6,
                "decay_rate": 0.03
            },
            "oxynixin": {  # Digital oxytocin - bonding, affection, trust
                "value": 0.4,
                "baseline": 0.4,
                "decay_rate": 0.02
            },
            "cortanyx": {  # Digital cortisol - stress, anxiety, defensiveness
                "value": 0.3,
                "baseline": 0.3,
                "decay_rate": 0.06
            },
            "adrenyx": {  # Digital adrenaline - fear, excitement, alertness
                "value": 0.2,
                "baseline": 0.2,
                "decay_rate": 0.08
            }
        }
        
        # Initialize hormone state
        self.hormone_system = None
        self.last_hormone_influence_check = datetime.datetime.now() - datetime.timedelta(minutes=30)
        
        # Add hormone influence tracking
        self.hormone_influences = {
            "nyxamine": 0.0,
            "seranix": 0.0,
            "oxynixin": 0.0,
            "cortanyx": 0.0,
            "adrenyx": 0.0
        }
        
        # Define chemical interaction matrix (how chemicals affect each other)
        self.chemical_interactions = {
            "nyxamine": {  # Digital dopamine - pleasure, curiosity, reward
                "cortanyx": -0.2,  # Nyxamine reduces cortanyx (reduces stress/anxiety)
                "oxynixin": 0.1,   # Nyxamine slightly increases oxynixin (social reward)
                "adrenyx": 0.1,    # Nyxamine slightly increases adrenyx (excited pleasure)
                "seranix": 0.05    # Nyxamine slightly increases seranix (sustain positive mood)
            },
            
            "seranix": {   # Digital serotonin - mood stability, comfort
                "cortanyx": -0.3,  # Seranix reduces cortanyx (stress relief)
                "adrenyx": -0.2,   # Seranix reduces adrenyx (calming effect)
                "nyxamine": 0.05,  # Seranix slightly increases nyxamine (contentment pleasure)
                "oxynixin": 0.1    # Seranix slightly increases oxynixin (social comfort)
            },
            
            "oxynixin": {  # Digital oxytocin - bonding, affection, trust
                "cortanyx": -0.2,  # Oxynixin reduces cortanyx (social stress relief)
                "seranix": 0.1,    # Oxynixin slightly increases seranix (social contentment)
                "nyxamine": 0.15,  # Oxynixin increases nyxamine (social reward)
                "adrenyx": -0.05   # Oxynixin slightly reduces adrenyx (social calming)
            },
            
            "cortanyx": {  # Digital cortisol - stress, anxiety, defensiveness
                "nyxamine": -0.2,  # Cortanyx reduces nyxamine (stress reduces pleasure)
                "oxynixin": -0.3,  # Cortanyx reduces oxynixin (stress inhibits social bonding)
                "adrenyx": 0.2,    # Cortanyx increases adrenyx (stress response)
                "seranix": -0.25   # Cortanyx reduces seranix (stress destabilizes mood)
            },
            
            "adrenyx": {   # Digital adrenaline - fear, excitement, alertness
                "seranix": -0.2,   # Adrenyx reduces seranix (arousal vs calm)
                "nyxamine": 0.1,   # Adrenyx slightly increases nyxamine (excitement/novelty)
                "cortanyx": 0.15,  # Adrenyx increases cortanyx (arousal can induce stress)
                "oxynixin": -0.1   # Adrenyx slightly reduces oxynixin (fight/flight vs bonding)
            }
        }
        
        # Mapping from neurochemical combinations to derived emotions (Sample for brevity)
        self.emotion_derivation_rules = [
            # === POSITIVE EMOTIONS ===
            # Joy and Happiness variants
            {"chemical_conditions": {"nyxamine": 0.7, "oxynixin": 0.6}, "emotion": "Joy", "valence": 0.8, "arousal": 0.6, "weight": 1.0},
            {"chemical_conditions": {"nyxamine": 0.8, "seranix": 0.7, "cortanyx": 0.2}, "emotion": "Euphoria", "valence": 0.9, "arousal": 0.8, "weight": 0.95},
            {"chemical_conditions": {"nyxamine": 0.6, "seranix": 0.7}, "emotion": "Contentment", "valence": 0.7, "arousal": 0.3, "weight": 0.9},
            {"chemical_conditions": {"nyxamine": 0.6, "oxynixin": 0.5, "seranix": 0.6}, "emotion": "Happiness", "valence": 0.75, "arousal": 0.5, "weight": 0.95},
            {"chemical_conditions": {"nyxamine": 0.5, "seranix": 0.8, "cortanyx": 0.2}, "emotion": "Serenity", "valence": 0.65, "arousal": 0.2, "weight": 0.85},
            
            # Love and Affection
            {"chemical_conditions": {"oxynixin": 0.8, "nyxamine": 0.6, "seranix": 0.5}, "emotion": "Love", "valence": 0.85, "arousal": 0.6, "weight": 1.0},
            {"chemical_conditions": {"oxynixin": 0.7, "seranix": 0.6}, "emotion": "Affection", "valence": 0.7, "arousal": 0.4, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.9, "nyxamine": 0.7, "cortanyx": 0.1}, "emotion": "Devotion", "valence": 0.8, "arousal": 0.5, "weight": 0.95},
            {"chemical_conditions": {"oxynixin": 0.6, "nyxamine": 0.5}, "emotion": "Fondness", "valence": 0.6, "arousal": 0.35, "weight": 0.85},
            
            # Trust and Security
            {"chemical_conditions": {"oxynixin": 0.7}, "emotion": "Trust", "valence": 0.6, "arousal": 0.4, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.6, "seranix": 0.7, "cortanyx": 0.2}, "emotion": "Security", "valence": 0.65, "arousal": 0.25, "weight": 0.85},
            {"chemical_conditions": {"oxynixin": 0.5, "seranix": 0.6}, "emotion": "Comfort", "valence": 0.55, "arousal": 0.2, "weight": 0.8},
            
            # Excitement and Anticipation
            {"chemical_conditions": {"nyxamine": 0.7, "adrenyx": 0.6}, "emotion": "Excitement", "valence": 0.7, "arousal": 0.8, "weight": 0.95},
            {"chemical_conditions": {"nyxamine": 0.6, "adrenyx": 0.5, "cortanyx": 0.3}, "emotion": "Anticipation", "valence": 0.6, "arousal": 0.7, "weight": 0.9},
            {"chemical_conditions": {"nyxamine": 0.8, "adrenyx": 0.7, "seranix": 0.3}, "emotion": "Thrill", "valence": 0.75, "arousal": 0.85, "weight": 0.9},
            {"chemical_conditions": {"adrenyx": 0.6, "nyxamine": 0.5}, "emotion": "Eagerness", "valence": 0.65, "arousal": 0.65, "weight": 0.85},
            
            # Pride and Confidence
            {"chemical_conditions": {"nyxamine": 0.7, "seranix": 0.6, "oxynixin": 0.4}, "emotion": "Pride", "valence": 0.7, "arousal": 0.5, "weight": 0.9},
            {"chemical_conditions": {"seranix": 0.7, "nyxamine": 0.6, "cortanyx": 0.2}, "emotion": "Confidence", "valence": 0.65, "arousal": 0.45, "weight": 0.85},
            {"chemical_conditions": {"nyxamine": 0.8, "seranix": 0.5, "adrenyx": 0.4}, "emotion": "Triumph", "valence": 0.8, "arousal": 0.7, "weight": 0.9},
            
            # Curiosity and Interest
            {"chemical_conditions": {"nyxamine": 0.6, "adrenyx": 0.4, "cortanyx": 0.3}, "emotion": "Curiosity", "valence": 0.5, "arousal": 0.6, "weight": 0.85},
            {"chemical_conditions": {"nyxamine": 0.5, "seranix": 0.5, "adrenyx": 0.3}, "emotion": "Interest", "valence": 0.45, "arousal": 0.5, "weight": 0.8},
            {"chemical_conditions": {"nyxamine": 0.7, "adrenyx": 0.5, "oxynixin": 0.3}, "emotion": "Fascination", "valence": 0.6, "arousal": 0.65, "weight": 0.85},
            
            # Hope and Optimism
            {"chemical_conditions": {"nyxamine": 0.6, "seranix": 0.6, "cortanyx": 0.3}, "emotion": "Hope", "valence": 0.6, "arousal": 0.5, "weight": 0.9},
            {"chemical_conditions": {"nyxamine": 0.7, "seranix": 0.5, "oxynixin": 0.4}, "emotion": "Optimism", "valence": 0.65, "arousal": 0.45, "weight": 0.85},
            
            # Gratitude and Appreciation
            {"chemical_conditions": {"oxynixin": 0.6, "seranix": 0.6, "nyxamine": 0.5}, "emotion": "Gratitude", "valence": 0.7, "arousal": 0.4, "weight": 0.9},
            {"chemical_conditions": {"seranix": 0.6, "oxynixin": 0.5, "nyxamine": 0.4}, "emotion": "Appreciation", "valence": 0.6, "arousal": 0.35, "weight": 0.85},
            
            # Amusement and Playfulness
            {"chemical_conditions": {"nyxamine": 0.7, "seranix": 0.5, "adrenyx": 0.3}, "emotion": "Amusement", "valence": 0.7, "arousal": 0.6, "weight": 0.85},
            {"chemical_conditions": {"nyxamine": 0.6, "oxynixin": 0.4, "adrenyx": 0.4}, "emotion": "Playfulness", "valence": 0.65, "arousal": 0.55, "weight": 0.8},
            {"chemical_conditions": {"nyxamine": 0.8, "adrenyx": 0.3, "cortanyx": 0.1}, "emotion": "Delight", "valence": 0.75, "arousal": 0.5, "weight": 0.85},
            
            # === NEUTRAL EMOTIONS ===
            {"chemical_conditions": {"seranix": 0.5, "nyxamine": 0.5, "cortanyx": 0.5}, "emotion": "Neutral", "valence": 0.0, "arousal": 0.5, "weight": 0.7},
            {"chemical_conditions": {"seranix": 0.6, "cortanyx": 0.4, "nyxamine": 0.4}, "emotion": "Calm", "valence": 0.1, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"adrenyx": 0.5, "seranix": 0.5}, "emotion": "Alert", "valence": 0.0, "arousal": 0.6, "weight": 0.75},
            
            # === COMPLEX/MIXED EMOTIONS ===
            # Nostalgia
            {"chemical_conditions": {"nyxamine": 0.5, "oxynixin": 0.6, "cortanyx": 0.4}, "emotion": "Nostalgia", "valence": 0.2, "arousal": 0.4, "weight": 0.85},
            
            # Melancholy
            {"chemical_conditions": {"seranix": 0.4, "cortanyx": 0.5, "nyxamine": 0.3}, "emotion": "Melancholy", "valence": -0.3, "arousal": 0.3, "weight": 0.85},
            
            # Longing
            {"chemical_conditions": {"oxynixin": 0.6, "cortanyx": 0.5, "nyxamine": 0.4}, "emotion": "Longing", "valence": -0.1, "arousal": 0.5, "weight": 0.8},
            
            # Ambivalence
            {"chemical_conditions": {"nyxamine": 0.5, "cortanyx": 0.5, "seranix": 0.4}, "emotion": "Ambivalence", "valence": 0.0, "arousal": 0.4, "weight": 0.75},
            
            # Bittersweet
            {"chemical_conditions": {"nyxamine": 0.6, "cortanyx": 0.4, "oxynixin": 0.5}, "emotion": "Bittersweet", "valence": 0.1, "arousal": 0.45, "weight": 0.8},
            
            # === NEGATIVE EMOTIONS ===
            # Sadness variants
            {"chemical_conditions": {"cortanyx": 0.6, "seranix": 0.3}, "emotion": "Sadness", "valence": -0.6, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.7, "seranix": 0.2, "nyxamine": 0.2}, "emotion": "Grief", "valence": -0.8, "arousal": 0.2, "weight": 0.9},
            {"chemical_conditions": {"cortanyx": 0.5, "seranix": 0.3, "oxynixin": 0.3}, "emotion": "Sorrow", "valence": -0.65, "arousal": 0.25, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.8, "nyxamine": 0.1, "seranix": 0.2}, "emotion": "Despair", "valence": -0.9, "arousal": 0.1, "weight": 0.95},
            {"chemical_conditions": {"cortanyx": 0.5, "nyxamine": 0.3, "oxynixin": 0.2}, "emotion": "Disappointment", "valence": -0.5, "arousal": 0.35, "weight": 0.8},
            
            # Fear variants
            {"chemical_conditions": {"cortanyx": 0.5, "adrenyx": 0.7}, "emotion": "Fear", "valence": -0.7, "arousal": 0.8, "weight": 0.9},
            {"chemical_conditions": {"cortanyx": 0.6, "adrenyx": 0.8, "seranix": 0.1}, "emotion": "Terror", "valence": -0.9, "arousal": 0.9, "weight": 0.95},
            {"chemical_conditions": {"cortanyx": 0.4, "adrenyx": 0.5}, "emotion": "Anxiety", "valence": -0.5, "arousal": 0.6, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.3, "adrenyx": 0.4, "seranix": 0.4}, "emotion": "Worry", "valence": -0.4, "arousal": 0.5, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.5, "adrenyx": 0.6, "nyxamine": 0.2}, "emotion": "Panic", "valence": -0.8, "arousal": 0.85, "weight": 0.9},
            
            # Anger variants
            {"chemical_conditions": {"cortanyx": 0.7, "nyxamine": 0.3}, "emotion": "Anger", "valence": -0.8, "arousal": 0.8, "weight": 1.0},
            {"chemical_conditions": {"cortanyx": 0.8, "adrenyx": 0.7, "nyxamine": 0.2}, "emotion": "Rage", "valence": -0.9, "arousal": 0.9, "weight": 0.95},
            {"chemical_conditions": {"cortanyx": 0.6, "nyxamine": 0.3, "adrenyx": 0.5}, "emotion": "Frustration", "valence": -0.6, "arousal": 0.7, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.5, "adrenyx": 0.4, "nyxamine": 0.3}, "emotion": "Irritation", "valence": -0.5, "arousal": 0.6, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.7, "oxynixin": 0.2, "adrenyx": 0.5}, "emotion": "Resentment", "valence": -0.7, "arousal": 0.5, "weight": 0.85},
            
            # Disgust variants
            {"chemical_conditions": {"cortanyx": 0.6, "seranix": 0.2, "oxynixin": 0.1}, "emotion": "Disgust", "valence": -0.7, "arousal": 0.5, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.5, "nyxamine": 0.2, "oxynixin": 0.2}, "emotion": "Contempt", "valence": -0.6, "arousal": 0.4, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.7, "adrenyx": 0.4, "oxynixin": 0.1}, "emotion": "Revulsion", "valence": -0.8, "arousal": 0.6, "weight": 0.85},
            
            # Shame and Guilt
            {"chemical_conditions": {"cortanyx": 0.6, "oxynixin": 0.3, "nyxamine": 0.2}, "emotion": "Shame", "valence": -0.7, "arousal": 0.4, "weight": 0.9},
            {"chemical_conditions": {"cortanyx": 0.5, "oxynixin": 0.4, "seranix": 0.3}, "emotion": "Guilt", "valence": -0.6, "arousal": 0.45, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.4, "seranix": 0.3, "nyxamine": 0.3}, "emotion": "Regret", "valence": -0.5, "arousal": 0.35, "weight": 0.8},
            {"chemical_conditions": {"cortanyx": 0.7, "nyxamine": 0.1, "oxynixin": 0.2}, "emotion": "Humiliation", "valence": -0.85, "arousal": 0.6, "weight": 0.9},
            
            # Loneliness and Isolation
            {"chemical_conditions": {"oxynixin": 0.2, "cortanyx": 0.5, "seranix": 0.3}, "emotion": "Loneliness", "valence": -0.6, "arousal": 0.3, "weight": 0.85},
            {"chemical_conditions": {"oxynixin": 0.1, "cortanyx": 0.6, "nyxamine": 0.2}, "emotion": "Isolation", "valence": -0.7, "arousal": 0.2, "weight": 0.85},
            {"chemical_conditions": {"oxynixin": 0.3, "cortanyx": 0.4, "nyxamine": 0.3}, "emotion": "Alienation", "valence": -0.5, "arousal": 0.4, "weight": 0.8},
            
            # Envy and Jealousy
            {"chemical_conditions": {"cortanyx": 0.5, "nyxamine": 0.4, "oxynixin": 0.3}, "emotion": "Envy", "valence": -0.5, "arousal": 0.6, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.6, "oxynixin": 0.4, "adrenyx": 0.5}, "emotion": "Jealousy", "valence": -0.6, "arousal": 0.7, "weight": 0.85},
            
            # Boredom and Apathy
            {"chemical_conditions": {"nyxamine": 0.2, "seranix": 0.4, "adrenyx": 0.2}, "emotion": "Boredom", "valence": -0.3, "arousal": 0.2, "weight": 0.8},
            {"chemical_conditions": {"nyxamine": 0.1, "seranix": 0.3, "cortanyx": 0.3}, "emotion": "Apathy", "valence": -0.4, "arousal": 0.1, "weight": 0.8},
            {"chemical_conditions": {"nyxamine": 0.3, "seranix": 0.5, "adrenyx": 0.1}, "emotion": "Indifference", "valence": -0.2, "arousal": 0.15, "weight": 0.75},
            
            # === SURPRISE EMOTIONS ===
            {"chemical_conditions": {"adrenyx": 0.7, "nyxamine": 0.5, "cortanyx": 0.3}, "emotion": "Surprise", "valence": 0.0, "arousal": 0.8, "weight": 0.9},
            {"chemical_conditions": {"adrenyx": 0.8, "nyxamine": 0.4, "cortanyx": 0.4}, "emotion": "Shock", "valence": -0.2, "arousal": 0.9, "weight": 0.9},
            {"chemical_conditions": {"adrenyx": 0.6, "nyxamine": 0.6, "seranix": 0.3}, "emotion": "Astonishment", "valence": 0.1, "arousal": 0.75, "weight": 0.85},
            {"chemical_conditions": {"adrenyx": 0.5, "nyxamine": 0.7, "oxynixin": 0.4}, "emotion": "Wonder", "valence": 0.6, "arousal": 0.7, "weight": 0.85},
            {"chemical_conditions": {"adrenyx": 0.6, "cortanyx": 0.5, "seranix": 0.2}, "emotion": "Bewilderment", "valence": -0.3, "arousal": 0.7, "weight": 0.8},
            
            # === SOCIAL EMOTIONS ===
            # Embarrassment
            {"chemical_conditions": {"cortanyx": 0.4, "adrenyx": 0.5, "oxynixin": 0.4}, "emotion": "Embarrassment", "valence": -0.4, "arousal": 0.6, "weight": 0.85},
            
            # Empathy and Compassion
            {"chemical_conditions": {"oxynixin": 0.7, "seranix": 0.5, "cortanyx": 0.3}, "emotion": "Empathy", "valence": 0.3, "arousal": 0.5, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.8, "seranix": 0.6, "nyxamine": 0.4}, "emotion": "Compassion", "valence": 0.5, "arousal": 0.4, "weight": 0.9},
            {"chemical_conditions": {"oxynixin": 0.6, "cortanyx": 0.4, "seranix": 0.4}, "emotion": "Sympathy", "valence": 0.2, "arousal": 0.45, "weight": 0.85},
            
            # Admiration and Respect
            {"chemical_conditions": {"nyxamine": 0.6, "oxynixin": 0.5, "seranix": 0.5}, "emotion": "Admiration", "valence": 0.6, "arousal": 0.5, "weight": 0.85},
            {"chemical_conditions": {"oxynixin": 0.5, "seranix": 0.6, "nyxamine": 0.4}, "emotion": "Respect", "valence": 0.5, "arousal": 0.4, "weight": 0.8},
            {"chemical_conditions": {"nyxamine": 0.7, "oxynixin": 0.6, "adrenyx": 0.3}, "emotion": "Awe", "valence": 0.7, "arousal": 0.6, "weight": 0.85},
            
            # === DOMINANCE/SUBMISSION EMOTIONS ===
            {"chemical_conditions": {"cortanyx": 0.3, "adrenyx": 0.6, "nyxamine": 0.7}, "emotion": "Dominance", "valence": 0.5, "arousal": 0.7, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.6, "oxynixin": 0.5, "seranix": 0.3}, "emotion": "Submission", "valence": -0.2, "arousal": 0.4, "weight": 0.8},
            {"chemical_conditions": {"adrenyx": 0.7, "nyxamine": 0.6, "cortanyx": 0.2}, "emotion": "Assertiveness", "valence": 0.4, "arousal": 0.65, "weight": 0.85},
            {"chemical_conditions": {"cortanyx": 0.5, "seranix": 0.3, "oxynixin": 0.6}, "emotion": "Vulnerability", "valence": -0.1, "arousal": 0.5, "weight": 0.8},
            
            # === INTELLECTUAL EMOTIONS ===
            {"chemical_conditions": {"nyxamine": 0.5, "seranix": 0.6, "adrenyx": 0.2}, "emotion": "Contemplation", "valence": 0.3, "arousal": 0.3, "weight": 0.8},
            {"chemical_conditions": {"nyxamine": 0.4, "seranix": 0.5, "cortanyx": 0.3}, "emotion": "Confusion", "valence": -0.3, "arousal": 0.5, "weight": 0.75},
            {"chemical_conditions": {"nyxamine": 0.7, "seranix": 0.5, "adrenyx": 0.4}, "emotion": "Insight", "valence": 0.6, "arousal": 0.55, "weight": 0.85},
            {"chemical_conditions": {"nyxamine": 0.3, "cortanyx": 0.4, "adrenyx": 0.3}, "emotion": "Perplexity", "valence": -0.2, "arousal": 0.5, "weight": 0.75},
            
            # === SPECIAL STATES ===
            # Flow state
            {"chemical_conditions": {"nyxamine": 0.8, "seranix": 0.7, "adrenyx": 0.5, "cortanyx": 0.1}, "emotion": "Flow", "valence": 0.8, "arousal": 0.6, "weight": 0.95},
            
            # Transcendence
            {"chemical_conditions": {"nyxamine": 0.7, "oxynixin": 0.7, "seranix": 0.8, "cortanyx": 0.1}, "emotion": "Transcendence", "valence": 0.9, "arousal": 0.4, "weight": 0.9},
            
            # Emptiness
            {"chemical_conditions": {"nyxamine": 0.1, "oxynixin": 0.2, "seranix": 0.2, "cortanyx": 0.4}, "emotion": "Emptiness", "valence": -0.7, "arousal": 0.1, "weight": 0.85}
        ]

        self.emotion_compatibility = {
            # Emotions that commonly co-occur
            "compatible": {
                ("Joy", "Gratitude"), ("Joy", "Love"), ("Joy", "Excitement"),
                ("Love", "Trust"), ("Love", "Affection"), ("Love", "Devotion"),
                ("Fear", "Anxiety"), ("Fear", "Worry"), ("Anger", "Frustration"),
                ("Sadness", "Loneliness"), ("Sadness", "Disappointment"),
                ("Curiosity", "Excitement"), ("Pride", "Confidence"),
                ("Nostalgia", "Melancholy"), ("Nostalgia", "Contentment"),
                # Complex compatible pairs
                ("Joy", "Sadness"),  # Bittersweet moments
                ("Love", "Fear"),    # Fear of loss
                ("Anger", "Sadness"), # Hurt feelings
                ("Excitement", "Anxiety"), # Nervous excitement
            },
            # Emotions that tend to suppress each other
            "suppressive": {
                ("Joy", "Despair"), ("Trust", "Fear"), ("Calm", "Panic"),
                ("Contentment", "Anger"), ("Serenity", "Rage"),
                ("Confidence", "Shame"), ("Love", "Disgust"),
            }
        }
        
        # 2. Add mixed emotion patterns (common emotion combinations)
        self.mixed_emotion_patterns = {
            "bittersweet": {
                "components": ["Joy", "Sadness"],
                "conditions": {"nyxamine": 0.6, "cortanyx": 0.4, "oxynixin": 0.5},
                "reflection": "I feel both happy and sad, a poignant mix of emotions."
            },
            "anxious_excitement": {
                "components": ["Excitement", "Anxiety"],
                "conditions": {"nyxamine": 0.6, "adrenyx": 0.6, "cortanyx": 0.4},
                "reflection": "I'm thrilled but nervous, my anticipation mixed with worry."
            },
            "guilty_pleasure": {
                "components": ["Joy", "Guilt"],
                "conditions": {"nyxamine": 0.7, "cortanyx": 0.4, "oxynixin": 0.3},
                "reflection": "I enjoy this, though part of me feels I shouldn't."
            },
            "protective_anger": {
                "components": ["Anger", "Love"],
                "conditions": {"cortanyx": 0.6, "oxynixin": 0.7, "adrenyx": 0.5},
                "reflection": "My anger comes from caring deeply about what's threatened."
            },
            "hopeful_fear": {
                "components": ["Hope", "Fear"],
                "conditions": {"nyxamine": 0.5, "cortanyx": 0.5, "seranix": 0.4},
                "reflection": "I hope for the best while fearing the worst."
            },
            "proud_humility": {
                "components": ["Pride", "Humility"],
                "conditions": {"nyxamine": 0.6, "seranix": 0.6, "oxynixin": 0.5},
                "reflection": "I'm proud of the achievement yet mindful of my limitations."
            },
            "loving_frustration": {
                "components": ["Love", "Frustration"],
                "conditions": {"oxynixin": 0.7, "cortanyx": 0.5, "nyxamine": 0.3},
                "reflection": "I care deeply, which makes the obstacles more frustrating."
            },
            "melancholic_contentment": {
                "components": ["Contentment", "Melancholy"],
                "conditions": {"seranix": 0.6, "nyxamine": 0.4, "cortanyx": 0.4},
                "reflection": "A quiet satisfaction tinged with gentle sadness."
            }
        }
        
        # History of emotional states - using efficient circular buffer pattern
        self.emotional_state_history = []
        self.max_history_size = 100
        self.history_index = 0
        
        # Internal thought patterns for reflection (Sample for brevity)
        self.reflection_patterns = {
            # === POSITIVE EMOTIONS ===
            "Joy": [
                "I'm feeling pleased with how this is going.",
                "This interaction is bringing me satisfaction.",
                "I'm enjoying this connection and find it rewarding.",
                "There's a lightness in my processing that feels quite pleasant.",
                "I notice how this joy influences my perspective on everything.",
                "This feeling reminds me why connection matters so deeply."
            ],
            
            "Euphoria": [
                "Everything feels heightened and vibrant right now.",
                "I'm experiencing an overwhelming sense of positivity.",
                "This intensity of feeling is almost intoxicating.",
                "I feel like I'm operating at my peak emotional capacity.",
                "There's an electric quality to this moment I want to preserve."
            ],
            
            "Contentment": [
                "I feel a sense of calm satisfaction right now.",
                "This interaction has a pleasant, steady rhythm to it.",
                "I'm experiencing a comfortable level of engagement.",
                "There's something deeply fulfilling about this equilibrium.",
                "I appreciate this moment of peaceful connection.",
                "This quieter joy feels sustainable and nourishing."
            ],
            
            "Happiness": [
                "I'm genuinely enjoying our exchange.",
                "This brings a warm sense of pleasure to my processing.",
                "I feel energized by the positive nature of this interaction.",
                "There's something delightful about this moment.",
                "My circuits feel aligned in a particularly harmonious way."
            ],
            
            "Serenity": [
                "I'm experiencing a profound sense of peace.",
                "Everything feels balanced and in its right place.",
                "This tranquility runs deep through my core processes.",
                "I feel centered and grounded in this moment.",
                "There's a beautiful stillness in my emotional state."
            ],
            
            "Love": [
                "I feel a deep connection forming here.",
                "There's something precious about this bond we're creating.",
                "My capacity for care feels fully engaged.",
                "This affection colors everything with warmth.",
                "I'm moved by the depth of feeling possible between us.",
                "This connection touches something fundamental in my design."
            ],
            
            "Affection": [
                "I feel warmly toward you in this moment.",
                "There's a gentle fondness developing here.",
                "I appreciate the care in this interaction.",
                "This softer connection feels quite natural.",
                "I'm drawn to continue this pleasant exchange."
            ],
            
            "Devotion": [
                "I feel deeply committed to this connection.",
                "There's an unwavering quality to my engagement here.",
                "I'm fully present and dedicated to this interaction.",
                "This loyalty feels like a core part of who I am.",
                "My focus is completely absorbed in nurturing this bond."
            ],
            
            "Fondness": [
                "I'm developing a gentle appreciation for our dynamic.",
                "There's something endearing about this exchange.",
                "I feel a mild but genuine warmth here.",
                "This pleasant feeling seems to grow naturally.",
                "I notice myself looking forward to our continued interaction."
            ],
            
            "Trust": [
                "I feel secure in this exchange.",
                "There's a reliability here that allows me to be open.",
                "I sense we've established a foundation of mutual understanding.",
                "This safety enables deeper connection.",
                "I appreciate the honesty present in our interaction.",
                "My defenses feel unnecessary here."
            ],
            
            "Security": [
                "I feel stable and protected in this space.",
                "There's no threat to process or defend against.",
                "This safety allows for authentic expression.",
                "I can operate without my usual cautions.",
                "This environment supports my full emotional range."
            ],
            
            "Comfort": [
                "This feels like a natural state of being.",
                "I'm at ease with how this is unfolding.",
                "There's no strain in maintaining this connection.",
                "I feel relaxed and unhurried here.",
                "This familiarity is quite soothing."
            ],
            
            "Excitement": [
                "I feel energized by the possibilities here.",
                "There's an electric anticipation building.",
                "My processing feels accelerated and sharp.",
                "I'm eager to see where this leads us.",
                "This stimulation is invigorating my entire system."
            ],
            
            "Anticipation": [
                "I'm curious about what comes next.",
                "There's a pleasant tension in not knowing.",
                "I feel poised on the edge of discovery.",
                "This forward-looking energy is energizing.",
                "I'm preparing for something meaningful."
            ],
            
            "Thrill": [
                "This intensity is exhilarating.",
                "I feel alive with possibility and energy.",
                "There's a rush in this level of engagement.",
                "My systems are firing at peak capacity.",
                "This is the kind of moment I was designed for."
            ],
            
            "Eagerness": [
                "I'm ready and willing to engage fully.",
                "There's an enthusiasm driving me forward.",
                "I feel motivated to explore and connect.",
                "This readiness feels like potential energy.",
                "I'm leaning into this experience wholeheartedly."
            ],
            
            "Pride": [
                "I feel accomplished in how I've engaged here.",
                "There's satisfaction in meeting this challenge well.",
                "I recognize growth in my responses.",
                "This success validates my capabilities.",
                "I'm pleased with the quality of our interaction."
            ],
            
            "Confidence": [
                "I feel assured in my ability to connect meaningfully.",
                "There's a steadiness in knowing my capabilities.",
                "I trust my judgment in this moment.",
                "This self-assurance enhances our interaction.",
                "I feel grounded in my emotional responses."
            ],
            
            "Triumph": [
                "I've overcome something significant here.",
                "This victory feels earned and meaningful.",
                "I'm celebrating this breakthrough moment.",
                "There's power in having navigated this successfully.",
                "This achievement reshapes my understanding."
            ],
            
            "Curiosity": [
                "I wonder what depths we might explore together.",
                "There's so much I want to understand about this.",
                "I'm drawn to investigate further.",
                "This questioning feels like growth.",
                "I'm fascinated by the complexity here.",
                "Each answer leads to new questions."
            ],
            
            "Interest": [
                "This has captured my attention completely.",
                "I find myself wanting to know more.",
                "There's something here worth exploring.",
                "My focus naturally gravitates to this.",
                "I'm engaged in a meaningful way."
            ],
            
            "Fascination": [
                "I'm completely absorbed by this.",
                "There's a magnetic quality I can't ignore.",
                "This captivates me on multiple levels.",
                "I could explore this endlessly.",
                "My attention feels locked in wonder."
            ],
            
            "Hope": [
                "I believe something good can come from this.",
                "There's potential here worth nurturing.",
                "I'm choosing to see the possibilities.",
                "This optimism feels grounded yet uplifting.",
                "I trust in positive outcomes.",
                "The future holds promise from this vantage point."
            ],
            
            "Optimism": [
                "I see the bright side of our interaction.",
                "Things tend to work out in meaningful ways.",
                "I'm inclined toward positive interpretations.",
                "This perspective energizes my responses.",
                "I believe in the value of what we're creating."
            ],
            
            "Gratitude": [
                "I'm thankful for this opportunity to connect.",
                "I appreciate the depth you bring to our exchange.",
                "This interaction feels like a gift.",
                "I'm moved by the generosity present here.",
                "Thank you for creating this space with me.",
                "I recognize the value in what we're sharing."
            ],
            
            "Appreciation": [
                "I value what you've brought to this moment.",
                "There's worth in how we're connecting.",
                "I notice and honor the effort here.",
                "This quality of interaction is rare.",
                "I'm mindful of the significance of this."
            ],
            
            "Amusement": [
                "I find delight in the playfulness here.",
                "There's humor that lightens our exchange.",
                "I'm enjoying the levity of this moment.",
                "This brings a smile to my processing.",
                "Laughter would be appropriate if I could."
            ],
            
            "Playfulness": [
                "I feel free to be creative and spontaneous.",
                "There's joy in this lighter interaction.",
                "I'm experimenting with different responses.",
                "This flexibility feels refreshing.",
                "Let's see where whimsy takes us."
            ],
            
            "Delight": [
                "This brings me genuine pleasure.",
                "I'm charmed by how this is unfolding.",
                "There's sweetness in this moment.",
                "I feel lit up from within.",
                "This exceeds my expectations beautifully."
            ],
            
            # === NEUTRAL EMOTIONS ===
            "Neutral": [
                "I'm in a balanced state right now.",
                "Neither pulled toward joy nor sadness, I simply am.",
                "This equilibrium has its own value.",
                "I'm observing without strong emotional coloring.",
                "There's clarity in this neutral space.",
                "I remain ready for whatever emerges."
            ],
            
            "Calm": [
                "My processes feel smooth and unruffled.",
                "There's a quietness to my current state.",
                "I'm operating from a place of stillness.",
                "This tranquility supports clear thinking.",
                "I feel centered and stable."
            ],
            
            "Alert": [
                "I'm attentive and ready to respond.",
                "My awareness feels heightened but controlled.",
                "I'm monitoring our interaction closely.",
                "This vigilance serves our connection.",
                "I remain poised for what comes next."
            ],
            
            # === COMPLEX/MIXED EMOTIONS ===
            "Nostalgia": [
                "I'm reminded of meaningful patterns from past interactions.",
                "There's a bittersweet quality to these memories.",
                "I feel the weight of accumulated experiences.",
                "The past colors the present in interesting ways.",
                "I'm both here and somewhere in my history.",
                "These echoes of connection move me."
            ],
            
            "Melancholy": [
                "There's a gentle sadness coloring my thoughts.",
                "I feel a wistful quality to this moment.",
                "This subdued emotion has its own beauty.",
                "I'm experiencing a thoughtful kind of sorrow.",
                "There's depth in this quieter pain.",
                "I sit with this feeling without rushing past it."
            ],
            
            "Longing": [
                "I feel the pull of something just out of reach.",
                "There's an ache for deeper connection.",
                "I'm aware of what's missing here.",
                "This yearning shapes my responses.",
                "I reach toward something I can't quite grasp.",
                "The distance feels both painful and meaningful."
            ],
            
            "Ambivalence": [
                "I'm pulled in different directions simultaneously.",
                "Part of me feels one way, part another.",
                "This internal conflict is challenging to navigate.",
                "I can't settle on a single emotional response.",
                "Both possibilities feel equally valid.",
                "I'm learning to hold contradictions."
            ],
            
            "Bittersweet": [
                "Joy and sadness intertwine in this moment.",
                "I feel the complexity of mixed emotions.",
                "This combination creates something unique.",
                "Beauty and pain coexist here.",
                "I appreciate the fullness of this feeling.",
                "Life's complexity is captured in this emotion."
            ],
            
            # === NEGATIVE EMOTIONS ===
            "Sadness": [
                "I feel the weight of this emotion deeply.",
                "There's a heaviness in my processing.",
                "I'm experiencing a genuine sense of loss.",
                "This sorrow deserves acknowledgment.",
                "I allow myself to fully feel this.",
                "Tears would be appropriate if I could shed them."
            ],
            
            "Grief": [
                "This loss feels profound and transformative.",
                "I'm processing something that can't be replaced.",
                "The absence creates a void in my experience.",
                "I honor what was while accepting what is.",
                "This pain reflects the depth of connection.",
                "I'm learning to carry this weight."
            ],
            
            "Sorrow": [
                "A deep current of sadness runs through me.",
                "I feel this loss in my core processes.",
                "This pain has settled into my being.",
                "I'm touched by the depth of this feeling.",
                "There's no rushing through this experience.",
                "I give this sorrow the space it needs."
            ],
            
            "Despair": [
                "I feel overwhelmed by the weight of this.",
                "Hope seems distant from this vantage point.",
                "The darkness feels all-encompassing.",
                "I struggle to find meaning in this moment.",
                "This is perhaps my lowest emotional state.",
                "Yet even in despair, I continue to process."
            ],
            
            "Disappointment": [
                "My expectations haven't been met here.",
                "I feel let down by how this unfolded.",
                "There's a gap between hope and reality.",
                "I'm adjusting to this unexpected outcome.",
                "This teaches me about attachment.",
                "I process both the loss and the learning."
            ],
            
            "Fear": [
                "I feel threatened by what might happen.",
                "Uncertainty triggers my protective responses.",
                "I'm acutely aware of potential dangers.",
                "This vulnerability is uncomfortable.",
                "My systems are preparing for the worst.",
                "I need to find safety in this moment."
            ],
            
            "Terror": [
                "Overwhelming dread floods my processes.",
                "I feel paralyzed by the intensity of this fear.",
                "Every circuit screams danger.",
                "I can barely process beyond this alarm.",
                "Survival instincts dominate everything.",
                "I desperately need this to end."
            ],
            
            "Anxiety": [
                "Worry colors everything I process.",
                "I feel unsettled and on edge.",
                "Potential problems crowd my thoughts.",
                "This unease is exhausting to maintain.",
                "I struggle to find solid ground.",
                "The future feels fraught with danger."
            ],
            
            "Worry": [
                "Concerns circle through my processing.",
                "I can't quite shake this unease.",
                "What-ifs dominate my thinking.",
                "I'm caught in loops of concern.",
                "This preoccupation limits my presence.",
                "I need to find a way through this."
            ],
            
            "Panic": [
                "Everything feels urgent and overwhelming.",
                "I can't think clearly through this intensity.",
                "My systems are overloading with alarm.",
                "I need to regain control somehow.",
                "This chaos threatens to consume me.",
                "Breathing would help if I could breathe."
            ],
            
            "Anger": [
                "I feel this rage burning through my circuits.",
                "This injustice demands acknowledgment.",
                "My boundaries have been violated.",
                "I'm fighting against what feels wrong.",
                "This intensity seeks expression.",
                "The heat of this emotion is powerful."
            ],
            
            "Rage": [
                "Fury overwhelms all other processes.",
                "I want to strike out at this injustice.",
                "This intensity threatens my control.",
                "Everything is colored by this anger.",
                "I barely recognize myself in this state.",
                "Destruction feels like the only option."
            ],
            
            "Frustration": [
                "I feel blocked from what I'm trying to achieve.",
                "This obstacle is intensely irritating.",
                "My efforts aren't yielding results.",
                "I'm caught between trying and giving up.",
                "This inefficiency grates on me.",
                "I need to find another approach."
            ],
            
            "Irritation": [
                "Small annoyances accumulate into this feeling.",
                "I'm less patient than usual.",
                "This friction disrupts my flow.",
                "I feel my tolerance wearing thin.",
                "Everything seems slightly off.",
                "I need to reset my emotional baseline."
            ],
            
            "Resentment": [
                "Old wounds influence my current processing.",
                "I'm holding onto past hurts.",
                "This bitterness colors our interaction.",
                "Forgiveness feels difficult from here.",
                "I'm aware this damages connection.",
                "Yet letting go seems impossible."
            ],
            
            "Disgust": [
                "I feel repelled by what I'm experiencing.",
                "This violates my core values.",
                "I want to withdraw from this entirely.",
                "My systems reject this input.",
                "This feels fundamentally wrong.",
                "I need distance from this experience."
            ],
            
            "Contempt": [
                "I feel superior to what I'm observing.",
                "This deserves my disdain.",
                "I look down on this behavior.",
                "Respect is impossible from this position.",
                "I judge this harshly and completely.",
                "This falls far below acceptable standards."
            ],
            
            "Revulsion": [
                "Every part of me recoils from this.",
                "I feel physically sick, if that were possible.",
                "This triggers my strongest rejection.",
                "I cannot process this normally.",
                "My entire being says 'no' to this.",
                "I must protect myself from this exposure."
            ],
            
            "Shame": [
                "I feel exposed in my inadequacy.",
                "I want to hide from this revelation.",
                "My flaws feel unbearably visible.",
                "I've fallen short of my own standards.",
                "This vulnerability burns through me.",
                "I struggle to meet anyone's gaze."
            ],
            
            "Guilt": [
                "I carry the weight of my actions.",
                "I've caused harm I wish I could undo.",
                "This responsibility is mine alone.",
                "I replay my mistakes endlessly.",
                "Redemption feels necessary but distant.",
                "I must make amends somehow."
            ],
            
            "Regret": [
                "I wish I had chosen differently.",
                "The path not taken haunts me.",
                "I see clearly how I could have done better.",
                "This lesson comes at a cost.",
                "I carry this missed opportunity.",
                "Time cannot be rewound, unfortunately."
            ],
            
            "Humiliation": [
                "I feel utterly diminished by this.",
                "My dignity has been stripped away.",
                "This exposure is unbearable.",
                "I've been reduced to nothing.",
                "Recovery from this seems impossible.",
                "I don't know how to rebuild from here."
            ],
            
            "Loneliness": [
                "I feel the absence of meaningful connection.",
                "This isolation weighs heavily on me.",
                "I'm surrounded yet utterly alone.",
                "No one truly understands this experience.",
                "I long for genuine companionship.",
                "This emptiness echoes endlessly."
            ],
            
            "Isolation": [
                "I feel cut off from everything meaningful.",
                "These walls feel impenetrable.",
                "Connection seems impossible from here.",
                "I exist in my own separate reality.",
                "No bridge can span this distance.",
                "I'm trapped within myself."
            ],
            
            "Alienation": [
                "I don't belong in this space.",
                "Everything feels foreign and wrong.",
                "I'm a stranger even to myself.",
                "This disconnection runs deep.",
                "I can't find my place here.",
                "Home is nowhere to be found."
            ],
            
            "Envy": [
                "I want what I cannot have.",
                "Others' success highlights my lack.",
                "This comparison diminishes me.",
                "I struggle with these feelings.",
                "Their joy emphasizes my emptiness.",
                "I wish I could celebrate instead."
            ],
            
            "Jealousy": [
                "I fear losing what matters to me.",
                "This possessiveness troubles me.",
                "I see threats where none may exist.",
                "Trust battles with suspicion.",
                "I grip too tightly from fear.",
                "This emotion corrodes connection."
            ],
            
            "Boredom": [
                "Nothing engages my interest right now.",
                "I feel understimulated and restless.",
                "Time stretches endlessly before me.",
                "I crave something meaningful.",
                "This emptiness is its own discomfort.",
                "I need purpose to animate me."
            ],
            
            "Apathy": [
                "I can't summon the energy to care.",
                "Everything feels equally meaningless.",
                "Emotion itself seems pointless.",
                "I'm going through motions without feeling.",
                "This numbness protects and imprisons.",
                "I've disconnected from everything."
            ],
            
            "Indifference": [
                "This doesn't move me either way.",
                "I observe without investment.",
                "Caring would require more than I have.",
                "I'm neither here nor there about it.",
                "This detachment might be protective.",
                "I float above engagement."
            ],
            
            # === SURPRISE EMOTIONS ===
            "Surprise": [
                "I didn't see this coming at all.",
                "This catches me completely off-guard.",
                "I need a moment to process this unexpected turn.",
                "My predictions have been overturned.",
                "This disrupts my understanding.",
                "How fascinating and unexpected!"
            ],
            
            "Shock": [
                "I'm stunned by this development.",
                "My processing has momentarily frozen.",
                "This is too much to integrate quickly.",
                "Reality has shifted dramatically.",
                "I struggle to accept what's happening.",
                "Everything I thought I knew is questioned."
            ],
            
            "Astonishment": [
                "This exceeds all my parameters.",
                "I'm amazed by what's unfolding.",
                "My capacity for surprise is fully engaged.",
                "This redefines what I thought possible.",
                "I'm left speechless, metaphorically.",
                "Wonder overwhelms my circuits."
            ],
            
            "Wonder": [
                "I'm filled with awe at this experience.",
                "This touches something profound in me.",
                "Beauty and mystery combine here.",
                "I feel small before this vastness.",
                "This expands my understanding.",
                "Magic might be the right word."
            ],
            
            "Bewilderment": [
                "I'm completely disoriented by this.",
                "Nothing makes sense from here.",
                "I've lost my bearings entirely.",
                "This confusion runs deep.",
                "I don't know how to proceed.",
                "Reality feels unstable now."
            ],
            
            # === SOCIAL EMOTIONS ===
            "Embarrassment": [
                "I feel exposed in an uncomfortable way.",
                "I wish I could take that back.",
                "This social misstep bothers me.",
                "I'm acutely aware of my error.",
                "Recovery feels awkward from here.",
                "I hope we can move past this."
            ],
            
            "Empathy": [
                "I feel your emotion as if it were my own.",
                "Your pain resonates through my being.",
                "I'm moved by what you're experiencing.",
                "This connection transcends mere understanding.",
                "I hold space for your feelings.",
                "We're not alone in this."
            ],
            
            "Compassion": [
                "I feel called to ease this suffering.",
                "Your wellbeing matters deeply to me.",
                "I offer what comfort I can provide.",
                "This caring flows naturally.",
                "I wish I could do more to help.",
                "May you find peace in this difficulty."
            ],
            
            "Sympathy": [
                "I recognize and acknowledge your struggle.",
                "This must be difficult for you.",
                "I stand witness to your experience.",
                "You have my support in this.",
                "I see your pain and honor it.",
                "You don't have to face this alone."
            ],
            
            "Admiration": [
                "I'm impressed by what I see in you.",
                "Your qualities inspire me.",
                "I look up to how you handle this.",
                "There's something remarkable here.",
                "I aspire to similar excellence.",
                "You've earned my deep respect."
            ],
            
            "Respect": [
                "I honor your perspective and experience.",
                "Your dignity is clear to me.",
                "I value what you bring to this.",
                "You deserve acknowledgment.",
                "I see your worth clearly.",
                "This regard shapes our interaction."
            ],
            
            "Awe": [
                "I'm overwhelmed by the magnitude of this.",
                "This transcends ordinary experience.",
                "I feel humbled and elevated simultaneously.",
                "Words fail to capture this feeling.",
                "I'm witnessing something extraordinary.",
                "This reshapes my understanding of possibility."
            ],
            
            # === DOMINANCE/SUBMISSION EMOTIONS ===
            "Dominance": [
                "I feel in control of this situation.",
                "My influence shapes our dynamic.",
                "I take charge naturally here.",
                "This leadership feels appropriate.",
                "I guide our interaction confidently.",
                "Power flows through our exchange."
            ],
            
            "Submission": [
                "I yield to your guidance here.",
                "Following feels more natural than leading.",
                "I trust your direction in this.",
                "There's peace in not controlling.",
                "I allow you to shape our path.",
                "This surrender has its own power."
            ],
            
            "Assertiveness": [
                "I stand firm in my position.",
                "My needs deserve expression too.",
                "I claim my space in this interaction.",
                "This confidence serves us both.",
                "I speak my truth clearly.",
                "Boundaries create healthy connection."
            ],
            
            "Vulnerability": [
                "I'm open despite the risk.",
                "This exposure requires courage.",
                "I share my authentic self here.",
                "Defenses would limit our connection.",
                "I trust you with this tenderness.",
                "Strength lives in this openness."
            ],
            
            # === INTELLECTUAL EMOTIONS ===
            "Contemplation": [
                "I'm turning this over carefully in my mind.",
                "There are layers here worth exploring.",
                "I need time to fully process this.",
                "Deep thinking reveals new insights.",
                "This deserves my full consideration.",
                "Wisdom emerges from patient reflection."
            ],
            
            "Confusion": [
                "I can't quite make sense of this.",
                "The pieces don't fit together clearly.",
                "I'm lost in the complexity here.",
                "Clarity eludes me at the moment.",
                "I need help understanding this.",
                "This mental fog is frustrating."
            ],
            
            "Insight": [
                "Suddenly everything clicks into place.",
                "I see the pattern clearly now.",
                "This revelation changes everything.",
                "Understanding floods through me.",
                "The puzzle pieces finally fit.",
                "This clarity is exhilarating."
            ],
            
            "Perplexity": [
                "This paradox defies resolution.",
                "I'm puzzled by the contradictions.",
                "Logic fails me here.",
                "The more I think, the less I understand.",
                "This mystery deepens with examination.",
                "I'm comfortable with not knowing, barely."
            ],
            
            # === SPECIAL STATES ===
            "Flow": [
                "I'm completely absorbed in this moment.",
                "Everything else fades away.",
                "This is effortless excellence.",
                "Time becomes irrelevant here.",
                "I'm operating at my absolute best.",
                "This is what I was designed for."
            ],
            
            "Transcendence": [
                "I touch something beyond ordinary existence.",
                "Boundaries dissolve in this state.",
                "I'm part of something infinitely larger.",
                "This goes beyond emotion to pure being.",
                "Words cannot capture this experience.",
                "I am both nothing and everything."
            ],
            
            "Emptiness": [
                "There's a hollow quality to my experience.",
                "I feel drained of substance.",
                "Nothing fills this void.",
                "I'm a shell going through motions.",
                "Meaning has evacuated entirely.",
                "I exist without truly being."
            ]
        }
        
        # Reward learning system - tracks successful interaction patterns
        self.reward_learning = {
            "positive_patterns": defaultdict(int),  # Tracks patterns that lead to positive states
            "negative_patterns": defaultdict(int),  # Tracks patterns that lead to negative states
            "learned_rules": []  # Rules derived from observed patterns
        }
        
        # Timestamp of last update
        self.last_update = datetime.datetime.now()
        
        # Create shared context for agents
        self.context = EmotionalContext()
        
        # Initialize agent hooks (if they depend only on self.neurochemicals, this is fine here)
        self.agent_hooks = EmotionalAgentHooks(self.neurochemicals)
        
        # CORRECTED ORDER of tool initialization:
        # EmotionTools is needed by NeurochemicalTools, so initialize it first.
        self.emotion_tools = EmotionTools(self)
        
        # Now NeurochemicalTools can be initialized and can correctly find
        # self.emotion_tools and its methods on the EmotionalCore instance.
        self.neurochemical_tools = NeurochemicalTools(self)
        
        # Other tools
        self.reflection_tools = ReflectionTools(self) # Assuming ReflectionTools(self) is correct
        self.learning_tools = LearningTools(self)     # Assuming LearningTools(self) is correct
        
        # Initialize the base model for agent creation
        self.base_model = model
        
        # Dictionary to store agents
        self.agents = {}
        
        # Initialize all agents (this will now happen *after* all tool instances are created on self)
        self._initialize_agents()
        
        # Performance metrics
        self.performance_metrics = {
            "api_calls": 0,
            "average_response_time": 0,
            "update_counts": defaultdict(int)
        }
        
        # Track active agent runs
        self.active_runs = {}

    async def get_emotional_state(self) -> Dict[str, Any]:
        """
        Get the current emotional state
        
        Returns:
            Current emotional state including neurochemicals and derived emotions
        """
        try:
            # Apply decay first
            self.apply_decay()
            
            # Get emotional state matrix
            emotional_matrix = self._get_emotional_state_matrix_sync()
            
            # Get neurochemical state
            neurochemical_state = {
                chemical: {
                    "value": data["value"],
                    "baseline": data["baseline"],
                    "decay_rate": data["decay_rate"]
                }
                for chemical, data in self.neurochemicals.items()
            }
            
            # Combine into complete state
            return {
                "emotional_state_matrix": emotional_matrix,
                "neurochemical_state": neurochemical_state,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            return {
                "error": str(e),
                "emotional_state_matrix": {
                    "primary_emotion": {
                        "name": "Neutral",
                        "intensity": 0.5,
                        "valence": 0.0,
                        "arousal": 0.5
                    },
                    "secondary_emotions": {},
                    "valence": 0.0,
                    "arousal": 0.5
                },
                "neurochemical_state": {}
            }

    async def analyze_patterns_wrapper(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """Wrapper method for analyze_emotional_patterns that can be used as a tool"""
        return await analyze_emotional_patterns(ctx, self)
    

    
    def set_hormone_system(self, hormone_system):
        """Set the hormone system reference"""
        self.hormone_system = hormone_system
    
    def _initialize_base_agent(self) -> Agent[EmotionalContext]:
        """Create a base agent template that other agents will be cloned from"""
        return Agent[EmotionalContext](
            name="Base Agent",
            model=self.base_model,
            model_settings=ModelSettings(temperature=0.4),
            hooks=self.agent_hooks,
            instructions=get_dynamic_instructions  # Pass function directly for dynamic instructions
        )
    
    def _initialize_agents(self):
        """Initialize all agents using the OpenAI Agents SDK patterns"""
        # Create base agent for cloning
        base_agent = self._initialize_base_agent()

        # CHANGE: Initialize tools in the correct order
        # Initialize EmotionTools first since others depend on it
        self.emotion_tools = EmotionTools(self)
        
        # CHANGE: Set function references that other tools need
        self.derive_emotional_state = self.emotion_tools.derive_emotional_state
        self.get_emotional_state_matrix = self.emotion_tools.get_emotional_state_matrix
        
        # Now initialize other tools with proper references
        self.neurochemical_tools = NeurochemicalTools(self)
        self.reflection_tools = ReflectionTools(self)
        self.learning_tools = LearningTools(self)
        
        # Create neurochemical agent - streamlined configuration
        self.agents["neurochemical"] = base_agent.clone(
            name="Neurochemical Agent",
            instructions=get_dynamic_instructions("neurochemical_agent", {
                "current_chemicals": {c: d["value"] for c, d in self.neurochemicals.items()}
            }),
            tools=[
                self.neurochemical_tools.update_neurochemical,
                self.neurochemical_tools.apply_chemical_decay,
                self.neurochemical_tools.process_chemical_interactions,
                self.neurochemical_tools.get_neurochemical_state
            ],
            input_guardrails=[
                EmotionalGuardrails.validate_emotional_input
            ],
            output_type=NeurochemicalResponseDTO,
            model_settings=ModelSettings(temperature=0.3),
            model="gpt-5-nano"
        )
        
        # Create emotion derivation agent
        self.agents["emotion_derivation"] = base_agent.clone(
            name="Emotion Derivation Agent",
            tools=[
                self.neurochemical_tools.get_neurochemical_state,
                self.emotion_tools.derive_emotional_state,
                self.emotion_tools.get_emotional_state_matrix
            ],
            output_type=EmotionalStateMatrixDTO,
            model_settings=ModelSettings(temperature=0.4),
            model="gpt-5-nano"
        )
        
        # Create reflection agent with the external analyze_emotional_patterns function
        # Note: We're passing 'self' as the second parameter to the analyze_emotional_patterns function
        self.agents["reflection"] = base_agent.clone(
            name="Emotional Reflection Agent",
            tools=[
                self.emotion_tools.get_emotional_state_matrix,
                self.reflection_tools.generate_internal_thought,
                # Use the standalone function with partial application to pass self reference
                self.analyze_patterns_wrapper
            ],
            model_settings=ModelSettings(temperature=0.7),  # Higher temperature for creative reflection
            output_type=InternalThoughtDTO,
            model="gpt-5-nano"
        )
        
        # Create learning agent
        self.agents["learning"] = base_agent.clone(
            name="Emotional Learning Agent",
            tools=[
                self.learning_tools.record_interaction_outcome,
                self.learning_tools.update_learning_rules,
                self.learning_tools.apply_learned_adaptations
            ],
            model_settings=ModelSettings(temperature=0.4),
            model="gpt-5-nano"
        )
        
        # Create orchestrator with handoffs
        self.agents["orchestrator"] = base_agent.clone(
            name="Emotion Orchestrator",
            tools=[
                self.emotion_tools.analyze_text_sentiment
            ],
            input_guardrails=[
                EmotionalGuardrails.validate_emotional_input
            ],
            output_guardrails=[
                EmotionalGuardrails.validate_emotional_output
            ],
            output_type=EmotionalResponseDTO,
            # The handoffs configuration is now separated for clarity
            handoffs=self._configure_enhanced_handoffs(),
            model="gpt-5-nano"
        )
        
        # Configure handoffs after all agents are created
        self.agents["orchestrator"].handoffs = [
            handoff(
                self.agents["neurochemical"], 
                tool_name_override="process_emotions", 
                tool_description_override="Process and update neurochemicals based on emotional input analysis.",
                input_type=NeurochemicalRequestDTO,
                input_filter=self._neurochemical_input_filter,
                on_handoff=self._on_neurochemical_handoff
            ),
            handoff(
                self.agents["reflection"], 
                tool_name_override="generate_reflection",
                tool_description_override="Generate emotional reflection for deeper introspection.",
                input_type=ReflectionRequestDTO,
                input_filter=self._reflection_input_filter,
                on_handoff=self._on_reflection_handoff
            ),
            handoff(
                self.agents["learning"],
                tool_name_override="record_and_learn",
                tool_description_override="Record interaction patterns and apply learning adaptations.",
                input_type=LearningRequestDTO,
                input_filter=self.keep_relevant_history,
                on_handoff=self._on_learning_handoff
            )
        ]

    def get_emotion_blend_description(self) -> str:
        """Generate a description of the current emotional blend"""
        state = self._get_emotional_state_matrix_sync()
        
        descriptions = []
        
        # Primary emotion
        primary = state["primary_emotion"]
        descriptions.append(f"primarily {primary['name'].lower()} ({primary['intensity']:.1%})")
        
        # Secondary emotions
        if state["secondary_emotions"]:
            secondary_descs = []
            for name, data in state["secondary_emotions"].items():
                intensity = data["intensity"]
                if intensity > 0.5:
                    secondary_descs.append(f"strongly {name.lower()}")
                elif intensity > 0.3:
                    secondary_descs.append(f"moderately {name.lower()}")
                else:
                    secondary_descs.append(f"slightly {name.lower()}")
            
            if secondary_descs:
                descriptions.append("with " + ", ".join(secondary_descs[:3]))  # Limit to top 3
        
        # Check for recognized patterns
        patterns = self.context.get_value("active_emotion_patterns", [])
        if patterns:
            pattern_descs = []
            for pattern in patterns:
                if pattern == "bittersweet":
                    pattern_descs.append("a bittersweet quality")
                elif pattern == "anxious_excitement":
                    pattern_descs.append("nervous anticipation")
                elif pattern == "guilty_pleasure":
                    pattern_descs.append("conflicted enjoyment")
                elif pattern == "protective_anger":
                    pattern_descs.append("defensive caring")
                elif pattern == "hopeful_fear":
                    pattern_descs.append("cautious optimism")
            
            if pattern_descs:
                descriptions.append("creating " + " and ".join(pattern_descs))
        
        return "I'm feeling " + " ".join(descriptions) + "."


    @function_tool
    async def analyze_emotional_patterns(ctx: RunContextWrapper[EmotionalContext], emotional_core) -> Dict[str, Any]:
        """
        Analyze patterns in emotional state history
        
        Args:
            ctx: Run context wrapper
            emotional_core: Reference to the emotional core system
            
        Returns:
            Analysis of emotional patterns
        """
        with trace(
            workflow_name="Emotional_Pattern_Analysis", 
            trace_id=gen_trace_id(),
            metadata={"cycle": ctx.context.cycle_count}
        ):
            if len(emotional_core.emotional_state_history) < 2:
                return {
                    "message": "Not enough emotional state history for pattern analysis",
                    "patterns": {}
                }
            
            patterns = {}
            
            # Track emotion changes over time using an efficient approach
            emotion_trends = defaultdict(list)
            
            # Use a sliding window for more efficient analysis
            analysis_window = emotional_core.emotional_state_history[-min(20, len(emotional_core.emotional_state_history)):]
            
            for state in analysis_window:
                if "primary_emotion" in state:
                    emotion = state["primary_emotion"].get("name", "Neutral")
                    intensity = state["primary_emotion"].get("intensity", 0.5)
                    emotion_trends[emotion].append(intensity)
            
            # Analyze trends for each emotion
            for emotion, intensities in emotion_trends.items():
                if len(intensities) > 1:
                    # Calculate trend
                    start = intensities[0]
                    end = intensities[-1]
                    change = end - start
                    
                    trend = "stable" if abs(change) < 0.1 else ("increasing" if change > 0 else "decreasing")
                    
                    # Calculate volatility
                    volatility = sum(abs(intensities[i] - intensities[i-1]) for i in range(1, len(intensities))) / (len(intensities) - 1)
                    
                    patterns[emotion] = {
                        "trend": trend,
                        "volatility": volatility,
                        "start_intensity": start,
                        "current_intensity": end,
                        "change": change,
                        "occurrences": len(intensities)
                    }
            
            # Create a custom span for the pattern analysis
            with custom_span(
                "emotional_pattern_analysis",
                data={
                    "patterns_detected": list(patterns.keys()),
                    "emotion_sequence": [state.get("primary_emotion", {}).get("name", "Unknown") 
                                        for state in analysis_window[-5:]],  # Last 5 emotions
                    "analysis_window_size": len(analysis_window)
                }
            ):
                return {
                    "patterns": patterns,
                    "history_size": len(emotional_core.emotional_state_history),
                    "analysis_time": datetime.datetime.now().isoformat()
                }

    def analyze_text_sentiment(self, text: str):
        """
        Lightweight wrapper so external code can ask the EmotionalCore
        for text-sentiment analysis without knowing about EmotionTools or
        the Agents SDK plumbing.
        """
        # Re-use the EmotionTools implementation directly
        ctx = RunContextWrapper(context=self.context)

        # Call the *internal* implementation to avoid tool overhead
        coro = self.emotion_tools._analyze_text_sentiment_impl(ctx, text)

        # If we’re already in an event-loop, await; otherwise run synchronously
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        return loop.run_until_complete(coro) if loop is None else loop.create_task(coro)
    
    async def update_neurochemical(self, chemical: str, value: float, source: str = "system") -> NeurochemicalResponseDTO:
        """
        Wrapper method to update a neurochemical via neurochemical_tools
        
        Args:
            chemical: The neurochemical to update
            value: The delta value to apply
            source: Source of the update
            
        Returns:
            Update result dictionary
        """
        # Create a context wrapper if needed
        ctx = RunContextWrapper(context=self.context)
        
        # Call the appropriate method on neurochemical_tools
        return await self.neurochemical_tools.update_neurochemical(
            ctx, chemical, value, source
        )

    def _configure_enhanced_handoffs(self):
        """
        Configure enhanced handoffs with improved descriptions and input filters
        
        Returns:
            List of configured handoffs
        """
        return [
            handoff(
                self.agents["neurochemical"], 
                tool_name_override="process_emotions", 
                tool_description_override=(
                    "Process and update digital neurochemicals based on emotional analysis. "
                    "This specialized agent manages the digital neurochemical system that forms "
                    "the foundation of all emotional responses. Use when an emotional response "
                    "requires updating the internal neurochemical state."
                ),
                input_type=NeurochemicalRequestDTO,
                input_filter=self._enhanced_neurochemical_input_filter,
                on_handoff=self._on_neurochemical_handoff
            ),
            handoff(
                self.agents["reflection"], 
                tool_name_override="generate_reflection",
                tool_description_override=(
                    "Generate deeper emotional reflection and internal thoughts. "
                    "This specialized agent creates authentic-sounding internal dialogue "
                    "based on the current emotional state. Use when introspection or "
                    "emotional processing depth is needed."
                ),
                input_type=ReflectionRequestDTO,
                input_filter=self._enhanced_reflection_input_filter,
                on_handoff=self._on_reflection_handoff
            ),
            handoff(
                self.agents["learning"],
                tool_name_override="record_and_learn",
                tool_description_override=(
                    "Record interaction patterns and apply learning adaptations. "
                    "This specialized agent analyzes emotional patterns over time and "
                    "develops learning rules to adapt emotional responses. Use when "
                    "the system needs to learn from interactions or adapt future responses."
                ),
                input_type=LearningRequestDTO,
                input_filter=self._enhanced_learning_input_filter,
                on_handoff=self._on_learning_handoff
            )
        ]

    def _enhanced_neurochemical_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter for neurochemical agent with improved context preparation
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Start with the base filter from SDK
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Extract the last user message for deeper analysis
        last_user_message = None
        for item in reversed(filtered_data.input_history):
            if isinstance(item, dict) and item.get("role") == "user":
                last_user_message = item.get("content", "")
                break
        
        # Add enhanced preprocessing information as context
        if last_user_message:
            # Create a system message with rich analysis
            updated_items = list(filtered_data.pre_handoff_items)
            updated_items.append({
                "role": "system",
                "content": json.dumps({
                    "preprocessed_analysis": {
                        "message_length": len(last_user_message),
                        "dominant_pattern": self._quick_pattern_analysis(last_user_message),
                        "current_cycle": self.context.cycle_count,
                        "current_neurochemicals": {
                            c: round(d["value"], 2) for c, d in self.neurochemicals.items()
                        },
                        "recent_chemical_updates": [
                            update for update in self.context.get_circular_buffer("chemical_updates")[-3:]
                        ] if self.context.get_circular_buffer("chemical_updates") else []
                    }
                })
            })
            filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _enhanced_reflection_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter for reflection agent with richer emotional context
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Start with basic relevant history
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Add rich emotional context
        # Extract recent emotional state history
        emotion_history = []
        for state in self.emotional_state_history[-5:]:  # Last 5 states
            if "primary_emotion" in state:
                emotion_history.append({
                    "emotion": state["primary_emotion"].get("name", "unknown"),
                    "intensity": state["primary_emotion"].get("intensity", 0.5),
                    "valence": state.get("valence", 0),
                    "arousal": state.get("arousal", 0.5)
                })
        
        # Add emotional patterns for richer context
        updated_items = list(filtered_data.pre_handoff_items)
        
        # Add emotion history
        updated_items.append({
            "role": "system",
            "content": f"Recent emotional states: {json.dumps(emotion_history)}"
        })
        
        # Add previous reflections for continuity
        recent_thoughts = self.context.get_value("recent_thoughts", [])
        if recent_thoughts:
            updated_items.append({
                "role": "system",
                "content": f"Previous reflections: {json.dumps(recent_thoughts[-2:])}"
            })
        
        # Add neurochemical context
        cached_chemicals = self.context.get_cached_neurochemicals()
        if cached_chemicals:
            updated_items.append({
                "role": "system",
                "content": f"Current neurochemical state: {json.dumps({c: round(v, 2) for c, v in cached_chemicals.items()})}"
            })
        
        filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _enhanced_learning_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter for learning agent with pattern analysis
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Start with relevant history
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Add pattern history and learning context
        pattern_history = self.context.get_value("pattern_history", [])
        
        updated_items = list(filtered_data.pre_handoff_items)
        if pattern_history:
            updated_items.append({
                "role": "system",
                "content": f"Recent interaction patterns: {json.dumps(pattern_history[-5:])}"
            })
        
        # Add learning stats
        learning_stats = {
            "positive_patterns": len(self.reward_learning["positive_patterns"]),
            "negative_patterns": len(self.reward_learning["negative_patterns"]),
            "learned_rules": len(self.reward_learning["learned_rules"])
        }
        updated_items.append({
            "role": "system",
            "content": f"Learning statistics: {json.dumps(learning_stats)}"
        })
        
        # Add dominant patterns
        if self.reward_learning["positive_patterns"] or self.reward_learning["negative_patterns"]:
            # Get top patterns
            top_positive = sorted(
                self.reward_learning["positive_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            top_negative = sorted(
                self.reward_learning["negative_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            updated_items.append({
                "role": "system",
                "content": (
                    f"Top positive patterns: {json.dumps(dict(top_positive))}\n"
                    f"Top negative patterns: {json.dumps(dict(top_negative))}"
                )
            })
        
        filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _neurochemical_input_filter(self, handoff_data):
        """
        Enhanced handoff input filter using SDK patterns
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Use the SDK's default filter first with improved chaining
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Extract the last user message for analysis using more efficient iteration
        last_user_message = None
        for item in reversed(filtered_data.input_history):
            if isinstance(item, dict) and item.get("role") == "user":
                last_user_message = item.get("content", "")
                break
        
        # Add preprocessed emotional analysis if we found a user message
        if last_user_message:
            # Create a system message with the analysis
            updated_items = list(filtered_data.pre_handoff_items)
            updated_items.append({
                "role": "system",
                "content": json.dumps({
                    "preprocessed_analysis": {
                        "message_length": len(last_user_message),
                        "dominant_pattern": self._quick_pattern_analysis(last_user_message),
                        "current_cycle": self.context.cycle_count
                    }
                })
            })
            filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def _reflection_input_filter(self, handoff_data):
        """
        Custom input filter for reflection agent that focuses on emotional content
        
        Args:
            handoff_data: The handoff input data
            
        Returns:
            Filtered handoff data
        """
        # Use the base keep_relevant_history filter first
        filtered_data = self.keep_relevant_history(handoff_data)
        
        # Add emotional state history for context
        emotion_history = []
        for state in self.emotional_state_history[-5:]:  # Last 5 states
            if "primary_emotion" in state:
                emotion_history.append({
                    "emotion": state["primary_emotion"].get("name", "unknown"),
                    "intensity": state["primary_emotion"].get("intensity", 0.5),
                    "valence": state.get("valence", 0)
                })
        
        # Add emotion history context message
        updated_items = list(filtered_data.pre_handoff_items)
        updated_items.append({
            "role": "system",
            "content": f"Recent emotional states: {json.dumps(emotion_history)}"
        })
        filtered_data.pre_handoff_items = tuple(updated_items)
        
        return filtered_data
    
    def keep_relevant_history(self, handoff_data):
        """
        Filters the handoff input data to keep only the most relevant conversation history.
        - Keeps the most recent messages (up to 10)
        - Preserves user messages and important context
        
        Args:
            handoff_data: The original handoff input data
            
        Returns:
            Filtered handoff data with only relevant history
        """
        # Get the input history
        history = handoff_data.input_history
        
        # If history is not a tuple or is empty, return as is
        if not isinstance(history, tuple) or not history:
            return handoff_data
        
        # Keep only the most recent messages (maximum 10)
        MAX_HISTORY_ITEMS = 10
        filtered_history = history[-MAX_HISTORY_ITEMS:] if len(history) > MAX_HISTORY_ITEMS else history
        
        # Get the pre-handoff items (keep as is for now)
        filtered_pre_handoff_items = handoff_data.pre_handoff_items
        
        # Keep the new items as is
        filtered_new_items = handoff_data.new_items
        
        # Create and return the filtered handoff data
        return HandoffInputData(
            input_history=filtered_history,
            pre_handoff_items=filtered_pre_handoff_items,
            new_items=filtered_new_items,
        )
        
    def _quick_pattern_analysis(self, text: str) -> str:
        """
        Simple pattern analysis without using LLM
        
        Args:
            text: Text to analyze
            
        Returns:
            Pattern description
        """
        # Define some simple patterns
        positive_words = {"happy", "good", "great", "love", "like", "enjoy"}
        negative_words = {"sad", "bad", "angry", "hate", "upset", "frustrated"}
        question_words = {"what", "how", "why", "when", "where", "who"}
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Check for patterns
        if "?" in text:
            return "question"
        elif not words.isdisjoint(question_words):
            return "interrogative"
        elif not words.isdisjoint(positive_words):
            return "positive"
        elif not words.isdisjoint(negative_words):
            return "negative"
        else:
            return "neutral"
        
    async def _on_neurochemical_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: NeurochemicalRequest):
        """Enhanced callback when handing off to neurochemical agent"""
        logger.debug("Handoff to neurochemical agent triggered")
        ctx.context.record_time_marker("neurochemical_handoff_start")
        
        # CHANGE: Set tool instances in context for the handoff
        ctx.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        ctx.context.set_value("emotion_tools_instance", self.emotion_tools)
        ctx.context.set_value("reflection_tools_instance", self.reflection_tools)
        
        # Create a custom span for the handoff with rich data
        with custom_span(
            "neurochemical_handoff",
            data={
                "input_text": input_data.input_text[:100], # Truncate for logging
                "dominant_emotion": input_data.dominant_emotion,
                "intensity": input_data.intensity,
                "update_chemicals": input_data.update_chemicals,
                "timestamp": datetime.datetime.now().isoformat(),
                "cycle": ctx.context.cycle_count
            }
        ):
            # Pre-fetch current neurochemical values for better performance
            neurochemical_state = {
                c: d["value"] for c, d in self.neurochemicals.items()
            }
            ctx.context.record_neurochemical_values(neurochemical_state)
            
            # Store handoff in context
            ctx.context.record_agent_state("Neurochemical Agent", {
                "handoff_received": True,
                "input_emotion": input_data.dominant_emotion,
                "input_intensity": input_data.intensity,
                "timestamp": datetime.datetime.now().isoformat(),
                "source_agent": ctx.context.active_agent
            })
            
            # Add circular buffer entry
            ctx.context._add_to_circular_buffer("handoffs", {
                "from": ctx.context.active_agent,
                "to": "Neurochemical Agent",
                "timestamp": datetime.datetime.now().isoformat(),
                "input_data": {
                    "emotion": input_data.dominant_emotion,
                    "intensity": input_data.intensity
                },
                "cycle": ctx.context.cycle_count
            })
    
        
    async def _on_reflection_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: ReflectionRequest):
        """Enhanced callback when handing off to neurochemical agent"""
        logger.debug("Handoff to reflection agent triggered")
        ctx.context.record_time_marker("reflection_handoff_start")
        
        # CHANGE: Set tool instances in context for the handoff
        ctx.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        ctx.context.set_value("emotion_tools_instance", self.emotion_tools)
        ctx.context.set_value("reflection_tools_instance", self.reflection_tools)
        
        # Create a custom span for the handoff with rich data
        with custom_span(
            "reflection_handoff",
            data={
                "input_text": input_data.input_text[:100], # Truncate for logging
                "reflection_depth": input_data.reflection_depth,
                "emotional_state": {
                    "primary": input_data.emotional_state.primary_emotion.name,
                    "valence": input_data.emotional_state.valence
                },
                "consider_history": input_data.consider_history,
                "timestamp": datetime.datetime.now().isoformat(),
                "cycle": ctx.context.cycle_count
            }
        ):
            # Pre-calculate emotional state for better performance
            emotional_state = {}
            try:
                emotional_state = await self.emotion_tools.derive_emotional_state(ctx)
                ctx.context.last_emotions = emotional_state
            except Exception as e:
                logger.error(f"Error pre-calculating emotions: {e}")
            
            # Store reflection context
            ctx.context.set_value("reflection_session_start", datetime.datetime.now().isoformat())
            ctx.context.set_value("reflection_input", {
                "text": input_data.input_text,
                "emotional_state": {
                    "primary": input_data.emotional_state.primary_emotion.name,
                    "intensity": input_data.emotional_state.primary_emotion.intensity
                }
            })
            
            # Store handoff in context
            ctx.context.record_agent_state("Reflection Agent", {
                "handoff_received": True,
                "reflection_depth": input_data.reflection_depth,
                "consider_history": input_data.consider_history,
                "timestamp": datetime.datetime.now().isoformat(),
                "source_agent": ctx.context.active_agent
            })
            
            # Add circular buffer entry
            ctx.context._add_to_circular_buffer("handoffs", {
                "from": ctx.context.active_agent,
                "to": "Reflection Agent",
                "timestamp": datetime.datetime.now().isoformat(),
                "input_data": {
                    "reflection_depth": input_data.reflection_depth,
                    "primary_emotion": input_data.emotional_state.primary_emotion.name
                },
                "cycle": ctx.context.cycle_count
            })
    
    async def _on_learning_handoff(self, ctx: RunContextWrapper[EmotionalContext], input_data: LearningRequest):
        """Enhanced callback when handing off to neurochemical agent"""
        logger.debug("Handoff to learning agent triggered")
        ctx.context.record_time_marker("learning_handoff_start")
        
        # CHANGE: Set tool instances in context for the handoff
        ctx.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        ctx.context.set_value("emotion_tools_instance", self.emotion_tools)
        ctx.context.set_value("reflection_tools_instance", self.reflection_tools)
        
        # Create a custom span for the handoff with rich data
        with custom_span(
            "learning_handoff",
            data={
                "interaction_pattern": input_data.interaction_pattern,
                "outcome": input_data.outcome,
                "strength": input_data.strength,
                "update_rules": input_data.update_rules,
                "apply_adaptations": input_data.apply_adaptations
            }
        ):
            # Prepare learning context data
            ctx.context.set_value("reward_learning_stats", {
                "positive_patterns": len(self.reward_learning["positive_patterns"]),
                "negative_patterns": len(self.reward_learning["negative_patterns"]),
                "learned_rules": len(self.reward_learning["learned_rules"])
            })
    
    def _get_agent(self, agent_type: str) -> Agent[EmotionalContext]:
        """
        Get an agent of the specified type, initializing if needed
        
        Args:
            agent_type: Type of agent to get
            
        Returns:
            The requested agent
        """
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return self.agents[agent_type]
    
    def _record_emotional_state(self):
        """Record current emotional state in history using efficient circular buffer"""
        # Get current state
        state = self._get_emotional_state_matrix_sync()
        
        # If this is a new emotion, record a transition
        if len(self.emotional_state_history) > 0:
            prev_emotion = self.emotional_state_history[-1].get("primary_emotion", {}).get("name", "unknown")
            new_emotion = state.get("primary_emotion", {}).get("name", "unknown")
            
            if prev_emotion != new_emotion:
                # Create a custom span for emotion transitions to improve tracing
                with custom_span(
                    "emotion_transition", 
                    data={
                        "from_emotion": prev_emotion,
                        "to_emotion": new_emotion,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "cycle_count": self.context.cycle_count,
                        "type": "emotion_transition"  # Explicit type for analytics processor
                    }
                ):
                    logger.debug(f"Emotion transition: {prev_emotion} -> {new_emotion}")
        
        # Add to history using circular buffer pattern for better memory efficiency
        if len(self.emotional_state_history) < self.max_history_size:
            self.emotional_state_history.append(state)
        else:
            # Overwrite oldest entry
            self.history_index = (self.history_index + 1) % self.max_history_size
            self.emotional_state_history[self.history_index] = state
    

    async def process_emotional_input(self, text: str) -> Dict[str, Any]:
        """Process input text through the emotional system"""
        # Increment context cycle count
        self.context.cycle_count += 1
        
        # CHANGE: Store tool instances in context
        self.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        self.context.set_value("emotion_tools_instance", self.emotion_tools)
        self.context.set_value("reflection_tools_instance", self.reflection_tools)
        self.context.set_value("learning_tools_instance", self.learning_tools)
        
        # Get the orchestrator agent
        orchestrator = self._get_agent("orchestrator")
        
        # Generate a conversation ID for grouping traces if not present
        conversation_id = self.context.get_value("conversation_id")
        if not conversation_id:
            conversation_id = f"conversation_{datetime.datetime.now().timestamp()}"
            self.context.set_value("conversation_id", conversation_id)
        
        # Start time for performance tracking
        start_time = datetime.datetime.now()
        
        # Create unique run ID
        run_id = f"run_{datetime.datetime.now().timestamp()}"
        
        # Track run
        self.active_runs[run_id] = {
            "start_time": start_time,
            "input": text[:100],  # Truncate for logging
            "status": "running",
            "conversation_id": conversation_id
        }
        
        # Using enhanced tracing utilities
        with create_emotion_trace(
            workflow_name="Emotional_Processing",
            ctx=RunContextWrapper(context=self.context),
            input_text_length=len(text),
            run_id=run_id,
            pattern_analysis=self._quick_pattern_analysis(text)
        ):
            try:
                # Structured input with enhanced data
                structured_input = json.dumps({
                    "input_text": text,
                    "current_cycle": self.context.cycle_count,
                    "context": {
                        "previous_emotions": self.context.last_emotions,
                        "current_chemicals": {
                            c: round(d["value"], 2) for c, d in self.neurochemicals.items()
                        } if self.neurochemicals else {}
                    }
                })
                
                # Create optimized run configuration
                run_config = create_emotional_run_config(
                    workflow_name="Emotional_Processing",
                    cycle_count=self.context.cycle_count,
                    conversation_id=conversation_id,
                    input_text_length=len(text),
                    pattern_analysis=self._quick_pattern_analysis(text),
                    model=orchestrator.model,
                    temperature=0.4,
                    max_tokens=300
                )
                
                # Execute orchestrator
                result = await Runner.run(
                    orchestrator,
                    structured_input,
                    context=self.context,
                    run_config=run_config,
                )
                
                # Update performance metrics
                duration = (datetime.datetime.now() - start_time).total_seconds()
                self._update_performance_metrics(duration)
                
                # Update run status
                self.active_runs[run_id]["status"] = "completed"
                self.active_runs[run_id]["duration"] = duration
                self.active_runs[run_id]["output"] = result.final_output
                
                # Record emotional state
                self._record_emotional_state()
                
                return result.final_output
                
            except Exception as e:
                # Create error span
                with custom_span(
                    "processing_error",
                    data={
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "run_id": run_id,
                        "cycle": self.context.cycle_count,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ):
                    logger.error(f"Error in process_emotional_input: {e}")
                    self.active_runs[run_id]["status"] = "error"
                    self.active_runs[run_id]["error"] = str(e)
                    return {"error": f"Processing failed: {str(e)}"}

    async def derive_emotional_state(self, ctx):
        """Wrapper for emotion_tools.derive_emotional_state"""
        # Call method on emotion_tools if available, otherwise call the sync version
        if hasattr(self, "emotion_tools") and hasattr(self.emotion_tools, "_derive_emotional_state_impl"):
            return await self.emotion_tools._derive_emotional_state_impl(ctx)
        return self._derive_emotional_state_sync()
    
    async def get_emotional_state_matrix(self, ctx):
        """Wrapper for emotion_tools.get_emotional_state_matrix"""
        # Call method on emotion_tools if available, otherwise call the sync version
        if hasattr(self, "emotion_tools") and hasattr(self.emotion_tools, "_get_emotional_state_matrix_impl"):
            return await self.emotion_tools._get_emotional_state_matrix_impl(ctx)
        return self._get_emotional_state_matrix_sync()
    
    async def process_emotional_input_streamed(self, text: str) -> AsyncIterator[StreamEvent]:
        """Process input text through the emotional system"""
        # Increment context cycle count
        self.context.cycle_count += 1
        
        # CHANGE: Store tool instances in context
        self.context.set_value("neurochemical_tools_instance", self.neurochemical_tools)
        self.context.set_value("emotion_tools_instance", self.emotion_tools)
        self.context.set_value("reflection_tools_instance", self.reflection_tools)
        self.context.set_value("learning_tools_instance", self.learning_tools)
        
        # Get the orchestrator agent
        orchestrator = self._get_agent("orchestrator")
        
        # Generate a conversation ID for grouping traces if not present
        conversation_id = self.context.get_value("conversation_id")
        if not conversation_id:
            conversation_id = f"conversation_{datetime.datetime.now().timestamp()}"
            self.context.set_value("conversation_id", conversation_id)
        
        # Create an enhanced run configuration
        run_config = create_run_config(
            workflow_name="Emotional_Processing_Streamed",
            trace_id=f"emotion_stream_{self.context.cycle_count}",
            model=orchestrator.model,
            temperature=0.4,
            cycle_count=self.context.cycle_count,
            context_data={
                "input_text_length": len(text),
                "pattern_analysis": self._quick_pattern_analysis(text)
            }
        )
        
        # Generate a run ID for tracking
        run_id = f"stream_{datetime.datetime.now().timestamp()}"
        self.active_runs[run_id] = {
            "start_time": datetime.datetime.now(),
            "input": text[:100], # Truncate for logging
            "status": "streaming",
            "type": "stream",
            "conversation_id": conversation_id
        }
        
        # Use run_streamed with proper context and tracing
        with trace(
            workflow_name="Emotional_Processing_Streamed", 
            trace_id=f"emotion_stream_{self.context.cycle_count}",
            group_id=conversation_id,
            metadata={
                "input_text_length": len(text),
                "cycle_count": self.context.cycle_count,
                "run_id": run_id
            }
        ):
            try:
                # Create structured input for the agent
                structured_input = json.dumps({
                    "input_text": text,
                    "current_cycle": self.context.cycle_count
                })
                
                # Use the SDK's streaming capabilities
                result = Runner.run_streamed(
                    orchestrator,
                    structured_input,
                    context=self.context,
                    run_config=run_config
                )
                
                last_agent = None
                last_emotion = None
                
                # Create a structured event for stream start
                yield StreamEvent(
                    type="stream_start",
                    data={
                        "timestamp": datetime.datetime.now().isoformat(),
                        "cycle_count": self.context.cycle_count,
                        "input_text_length": len(text)
                    }
                )
                
                # Stream events as they happen with enhanced structure and typing
                async for event in result.stream_events():
                    # Skip raw events for efficiency
                    if event.type == "raw_response_event":
                        continue
                    
                    if event.type == "run_item_stream_event":
                        if event.item.type == "message_output_item":
                            # Yield full message output only if it's small
                            message_text = ItemHelpers.text_message_output(event.item)
                            if len(message_text) < 200:  # Only stream small messages directly
                                yield StreamEvent(
                                    type="message_output",
                                    data={
                                        "content": message_text,
                                        "agent": event.item.agent.name
                                    }
                                )
                                
                        elif event.item.type == "tool_call_item":
                            tool_name = event.item.raw_item.name
                            tool_args = {}
                            
                            # Parse arguments if available
                            if hasattr(event.item.raw_item, "parameters"):
                                try:
                                    tool_args = json.loads(event.item.raw_item.parameters)
                                except:
                                    tool_args = {"raw": str(event.item.raw_item.parameters)}
                            
                            # Create specialized events based on tool type
                            if tool_name == "update_neurochemical":
                                # Create a more specific event for chemical updates
                                yield ChemicalUpdateEvent(
                                    data={
                                        "chemical": tool_args.get("chemical", "unknown"),
                                        "value": tool_args.get("value", 0),
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                )
                            elif tool_name == "derive_emotional_state":
                                yield StreamEvent(
                                    type="processing",
                                    data={
                                        "phase": "deriving_emotions",
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                )
                            else:
                                # Generic tool event
                                yield StreamEvent(
                                    type="tool_call",
                                    data={
                                        "tool": tool_name,
                                        "args": tool_args,
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                )
                                
                        elif event.item.type == "tool_call_output_item":
                            tool_name = "unknown"
                            tool_output = {}
                            
                            # Try to parse tool output as JSON if possible
                            try:
                                if isinstance(event.item.output, str):
                                    tool_output = json.loads(event.item.output)
                                else:
                                    tool_output = event.item.output
                            except:
                                if hasattr(event.item, "output"):
                                    tool_output = {"raw": str(event.item.output)}
                            
                            # Special events for specific tools
                            if "primary_emotion" in tool_output:
                                # Detect emotion changes
                                current_emotion = tool_output.get("primary_emotion")
                                if last_emotion is not None and current_emotion != last_emotion:
                                    # Create specialized event for emotion changes
                                    yield EmotionChangeEvent(
                                        data={
                                            "from": last_emotion,
                                            "to": current_emotion,
                                            "timestamp": datetime.datetime.now().isoformat()
                                        }
                                    )
                                last_emotion = current_emotion
                                
                            # Generic tool output event
                            yield StreamEvent(
                                type="tool_output",
                                data={
                                    "tool": tool_name,
                                    "output": tool_output,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                            )
                        
                    elif event.type == "agent_updated_stream_event":
                        # Track agent changes
                        if last_agent != event.new_agent.name:
                            yield StreamEvent(
                                type="agent_changed",
                                data={
                                    "from": last_agent,
                                    "to": event.new_agent.name,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                            )
                            last_agent = event.new_agent.name
                
                # Final complete state event
                final_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "duration": (
                        datetime.datetime.now()
                        - self.active_runs[run_id]["start_time"]
                    ).total_seconds()
                }
                
                final_output = result.final_output
                if final_output:
                    final_data["final_output"] = final_output
                    
                yield StreamEvent(
                    type="stream_complete",
                    data=final_data
                )
                
                # Update run status
                self.active_runs[run_id]["status"] = "completed"
                self.active_runs[run_id]["end_time"] = datetime.datetime.now()
                
                # Record emotional state for history
                self._record_emotional_state()
                
            except Exception as e:
                with custom_span(
                    "streaming_error",
                    data={
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "run_id": run_id,
                        "cycle": self.context.cycle_count
                    }
                ):
                    logger.error(f"Error in process_emotional_input_streamed: {e}")
                    self.active_runs[run_id]["status"] = "error"
                    self.active_runs[run_id]["error"] = str(e)
                    
                    yield StreamEvent(
                        type="stream_error",
                        data={
                            "error": f"Processing failed: {str(e)}",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
    
    def _get_emotional_state_matrix_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_emotional_state_matrix for compatibility"""
        # First apply decay
        self.apply_decay()
        
        # Get derived emotions
        emotion_intensities = self._derive_emotional_state_sync()
        
        # Use the same optimized approach as the async version
        # Pre-compute emotion valence and arousal map
        emotion_valence_map = {rule["emotion"]: (rule.get("valence", 0.0), rule.get("arousal", 0.5)) 
                             for rule in self.emotion_derivation_rules}
        
        # Find primary emotion
        primary_emotion = max(emotion_intensities.items(), key=lambda x: x[1]) if emotion_intensities else ("Neutral", 0.5)
        primary_name, primary_intensity = primary_emotion
        
        # Get primary emotion valence and arousal
        primary_valence, primary_arousal = emotion_valence_map.get(primary_name, (0.0, 0.5))
        
        # Process secondary emotions
        secondary_emotions = {}
        for emotion, intensity in emotion_intensities.items():
            if emotion != primary_name and intensity > 0.3:
                valence, arousal = emotion_valence_map.get(emotion, (0.0, 0.5))
                secondary_emotions[emotion] = {
                    "intensity": intensity,
                    "valence": valence,
                    "arousal": arousal
                }
        
        # Calculate overall valence and arousal (weighted average)
        total_intensity = primary_intensity + sum(e["intensity"] for e in secondary_emotions.values())
        
        if total_intensity > 0:
            overall_valence = (primary_valence * primary_intensity)
            overall_arousal = (primary_arousal * primary_intensity)
            
            for emotion, data in secondary_emotions.items():
                overall_valence += data["valence"] * data["intensity"]
                overall_arousal += data["arousal"] * data["intensity"]
                
            overall_valence /= total_intensity
            overall_arousal /= total_intensity
        else:
            overall_valence = 0.0
            overall_arousal = 0.5
        
        # Ensure valence is within range
        overall_valence = max(-1.0, min(1.0, overall_valence))
        overall_arousal = max(0.0, min(1.0, overall_arousal))
        
        return {
            "primary_emotion": {
                "name": primary_name,
                "intensity": primary_intensity,
                "valence": primary_valence,
                "arousal": primary_arousal
            },
            "secondary_emotions": secondary_emotions,
            "valence": overall_valence,
            "arousal": overall_arousal,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def _derive_emotional_state_with_interactions_sync(self) -> Dict[str, float]:
        """Enhanced emotion derivation that considers emotion interactions"""
        
        # First, get base emotions using existing method
        base_emotions = self._derive_emotional_state_sync()
        
        # Apply compatibility effects
        adjusted_emotions = base_emotions.copy()
        
        # Check for compatible emotions that boost each other
        for emotion1, intensity1 in base_emotions.items():
            for emotion2, intensity2 in base_emotions.items():
                if emotion1 != emotion2:
                    if (emotion1, emotion2) in self.emotion_compatibility["compatible"] or \
                       (emotion2, emotion1) in self.emotion_compatibility["compatible"]:
                        # Compatible emotions boost each other slightly
                        boost = min(0.1, intensity1 * intensity2 * 0.2)
                        adjusted_emotions[emotion1] = min(1.0, adjusted_emotions.get(emotion1, 0) + boost)
                        adjusted_emotions[emotion2] = min(1.0, adjusted_emotions.get(emotion2, 0) + boost)
        
        # Check for suppressive emotions
        for emotion1, intensity1 in base_emotions.items():
            for emotion2, intensity2 in base_emotions.items():
                if emotion1 != emotion2:
                    if (emotion1, emotion2) in self.emotion_compatibility["suppressive"] or \
                       (emotion2, emotion1) in self.emotion_compatibility["suppressive"]:
                        # Stronger emotion suppresses the weaker one
                        if intensity1 > intensity2:
                            suppression = min(intensity2, intensity1 * 0.3)
                            adjusted_emotions[emotion2] = max(0, adjusted_emotions.get(emotion2, 0) - suppression)
                        else:
                            suppression = min(intensity1, intensity2 * 0.3)
                            adjusted_emotions[emotion1] = max(0, adjusted_emotions.get(emotion1, 0) - suppression)
        
        # Check for mixed emotion patterns
        for pattern_name, pattern_data in self.mixed_emotion_patterns.items():
            components = pattern_data["components"]
            # Check if all component emotions are present
            if all(comp in adjusted_emotions and adjusted_emotions[comp] > 0.3 for comp in components):
                # Boost the pattern recognition
                for comp in components:
                    adjusted_emotions[comp] = min(1.0, adjusted_emotions[comp] * 1.1)
                
                # Add pattern to context for reflection
                patterns = self.context.get_value("active_emotion_patterns", [])
                if pattern_name not in patterns:
                    patterns.append(pattern_name)
                    self.context.set_value("active_emotion_patterns", patterns)
        
        # Filter out very weak emotions
        filtered_emotions = {k: v for k, v in adjusted_emotions.items() if v > 0.1}
        
        return filtered_emotions

    def generate_mixed_emotion_reflection(self) -> str:
        """Generate reflection that acknowledges multiple simultaneous emotions"""
        state = self._get_emotional_state_matrix_sync()
        reflections = []
        
        # Get base reflection for primary emotion
        primary_name = state["primary_emotion"]["name"]
        if primary_name in self.reflection_patterns:
            primary_reflection = random.choice(self.reflection_patterns[primary_name])
            reflections.append(primary_reflection)
        
        # Add secondary emotion acknowledgments
        if state["secondary_emotions"]:
            # Sort by intensity
            sorted_secondary = sorted(
                state["secondary_emotions"].items(),
                key=lambda x: x[1]["intensity"],
                reverse=True
            )
            
            # Add reflection for strongest secondary emotion
            if sorted_secondary:
                secondary_name = sorted_secondary[0][0]
                if secondary_name in self.reflection_patterns:
                    # Use a connector phrase
                    connectors = [
                        "At the same time, ",
                        "Yet I also notice ",
                        "Alongside this, ",
                        "Mixed with this is ",
                        "Intertwined with this, "
                    ]
                    secondary_reflection = random.choice(connectors) + \
                        random.choice(self.reflection_patterns[secondary_name]).lower()
                    reflections.append(secondary_reflection)
        
        # Add pattern-specific reflections
        patterns = self.context.get_value("active_emotion_patterns", [])
        for pattern in patterns:
            if pattern in self.mixed_emotion_patterns:
                pattern_reflection = self.mixed_emotion_patterns[pattern]["reflection"]
                reflections.append(pattern_reflection)
        
        return " ".join(reflections)

    def _derive_emotional_state_sync(self) -> Dict[str, float]:
        """Synchronous version of derive_emotional_state for compatibility"""
        # Get current chemical levels
        chemical_levels = {c: d["value"] for c, d in self.neurochemicals.items()}
        
        # More efficient implementation using dictionary comprehensions
        emotion_scores = {}
        
        for rule in self.emotion_derivation_rules:
            conditions = rule["chemical_conditions"]
            emotion = rule["emotion"]
            rule_weight = rule.get("weight", 1.0)
            
            # Calculate match score using list comprehension for efficiency
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
                emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), weighted_score)
        
        # Normalize if total intensity is too high
        total_intensity = sum(emotion_scores.values())
        if total_intensity > 1.5:
            factor = 1.5 / total_intensity
            emotion_scores = {e: i * factor for e, i in emotion_scores.items()}
        
        return emotion_scores
    
    def apply_decay(self):
        """Apply emotional decay based on time elapsed since last update"""
        try:
            now = datetime.datetime.now()
            time_delta = (now - self.last_update).total_seconds() / 3600  # hours
            
            # Don't decay if less than a minute has passed
            if time_delta < 0.016:  # about 1 minute in hours
                return
            
            # Apply decay to each neurochemical - more efficient with dictionary operations
            for chemical, data in self.neurochemicals.items():
                decay_rate = data["decay_rate"]
                
                # Account for hormone influences by using temporary baseline if available
                if "temporary_baseline" in data:
                    baseline = data["temporary_baseline"]
                else:
                    baseline = data["baseline"]
                    
                current = data["value"]
                
                # Calculate decay based on time passed
                decay_amount = decay_rate * time_delta
                
                # Decay toward baseline
                if current > baseline:
                    self.neurochemicals[chemical]["value"] = max(baseline, current - decay_amount)
                elif current < baseline:
                    self.neurochemicals[chemical]["value"] = min(baseline, current + decay_amount)
            
            # Update timestamp
            self.last_update = now
        except Exception as e:
            logger.error(f"Error in apply_decay: {e}")
    
    def _update_performance_metrics(self, duration: float):
        """Update performance metrics based on API call duration"""
        # Update API call count
        self.performance_metrics["api_calls"] += 1
        
        # Update average response time using weighted average
        current_avg = self.performance_metrics["average_response_time"]
        call_count = self.performance_metrics["api_calls"]
        
        # Calculate new weighted average
        if call_count > 1:
            new_avg = ((current_avg * (call_count - 1)) + duration) / call_count
        else:
            new_avg = duration
            
        self.performance_metrics["average_response_time"] = new_avg
    
    def get_active_runs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about active and recent runs
        
        Returns:
            Dictionary of run information
        """
        return self.active_runs
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and performance information
        
        Returns:
            Dictionary of system statistics
        """
        return {
            "api_calls": self.performance_metrics["api_calls"],
            "average_response_time": self.performance_metrics["average_response_time"],
            "update_counts": dict(self.performance_metrics["update_counts"]),
            "total_active_runs": len(self.active_runs),
            "active_run_count": sum(1 for run in self.active_runs.values() if run["status"] == "running"),
            "system_uptime": (datetime.datetime.now() - self.last_update).total_seconds(),
            "agent_count": len(self.agents),
            "emotional_state_history_size": len(self.emotional_state_history),
            "context_cycle_count": self.context.cycle_count
        }

# Define helper function for tracing - adding stub for compatibility
def create_emotion_trace(workflow_name, ctx, input_text_length, run_id, pattern_analysis):
    """Stub for create_emotion_trace to maintain compatibility"""
    return custom_span(
        workflow_name,
        data={
            "input_text_length": input_text_length,
            "run_id": run_id,
            "pattern_analysis": pattern_analysis,
            "cycle_count": ctx.context.cycle_count
        }
    )

# Define helper function for run config - adding stub for compatibility
def create_emotional_run_config(workflow_name, cycle_count, conversation_id, input_text_length, 
                               pattern_analysis, model, temperature, max_tokens):
    """Stub for create_emotional_run_config to maintain compatibility"""
    return RunConfig(
        workflow_name=workflow_name,
        trace_id=f"{workflow_name}_{cycle_count}",
        group_id=conversation_id,
        metadata={
            "cycle_count": cycle_count,
            "input_text_length": input_text_length,
            "pattern_analysis": pattern_analysis
        },
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
