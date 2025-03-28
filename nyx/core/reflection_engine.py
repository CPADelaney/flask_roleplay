# nyx/core/reflection_engine.py

import logging
import asyncio
import random
import datetime
import math
import re
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import Agent, Runner, trace, function_tool, FunctionTool, ModelSettings
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============== Models for Structured Output ===============

class ReflectionOutput(BaseModel):
    """Structured output for reflection generations"""
    reflection_text: str = Field(description="The generated reflection text")
    confidence: float = Field(description="Confidence score between 0 and 1")
    significance: float = Field(description="Significance score between 0 and 1")
    emotional_tone: str = Field(description="Emotional tone of the reflection")
    neurochemical_influence: Dict[str, float] = Field(description="Influence of digital neurochemicals")
    tags: List[str] = Field(description="Tags categorizing the reflection")

class AbstractionOutput(BaseModel):
    """Structured output for abstractions"""
    abstraction_text: str = Field(description="The generated abstraction text")
    pattern_type: str = Field(description="Type of pattern identified")
    confidence: float = Field(description="Confidence score between 0 and 1")
    entity_focus: str = Field(description="Primary entity this abstraction focuses on")
    neurochemical_insight: Dict[str, str] = Field(description="Insights about neurochemical patterns")
    supporting_evidence: List[str] = Field(description="References to supporting memories")

class IntrospectionOutput(BaseModel):
    """Structured output for system introspection"""
    introspection_text: str = Field(description="The generated introspection text")
    memory_analysis: str = Field(description="Analysis of memory usage and patterns")
    emotional_insight: str = Field(description="Analysis of emotional patterns")
    emotional_intelligence_score: float = Field(description="Emotional intelligence score (0.0-1.0)")
    understanding_level: str = Field(description="Estimated level of understanding")
    neurochemical_balance: Dict[str, str] = Field(description="Analysis of neurochemical balance")
    focus_areas: List[str] = Field(description="Areas requiring focus or improvement")
    confidence: float = Field(description="Confidence score between 0 and 1")

class EmotionalProcessingOutput(BaseModel):
    """Structured output for emotional processing"""
    processing_text: str = Field(description="The generated emotional processing text")
    source_emotion: str = Field(description="Primary emotion being processed")
    neurochemical_dynamics: Dict[str, Any] = Field(description="Dynamics of neurochemical interactions")
    insight_level: float = Field(description="Depth of emotional insight (0.0-1.0)")
    adaptation: Optional[Dict[str, Any]] = Field(None, description="Suggested adaptation")

# =============== Tool Functions ===============

class MemoryData(BaseModel):
    memory_id: str
    memory_text: str
    memory_type: str
    significance: float
    metadata: Dict[str, Any]
    tags: List[str]

class MemoryFormat(BaseModel):
    memories: List[MemoryData]
    topic: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ReflectionContext(BaseModel):
    scenario_type: str
    emotional_state: Dict[str, Any]
    neurochemical_state: Dict[str, float]
    confidence: float
    source_memories: List[str]

@function_tool
async def format_memories_for_reflection(memories: List[Dict[str, Any]], 
                                       topic: Optional[str] = None,
                                       emotional_context: Optional[Dict[str, Any]] = None) -> str:
    """Format memory data into a structured representation for reflection with emotional context"""
    formatted_memories = []
    for memory in memories:
        formatted_memories.append(MemoryData(
            memory_id=memory.get("id", "unknown"),
            memory_text=memory.get("memory_text", ""),
            memory_type=memory.get("memory_type", "unknown"),
            significance=memory.get("significance", 5.0),
            metadata=memory.get("metadata", {}),
            tags=memory.get("tags", [])
        ))
    
    return MemoryFormat(
        memories=formatted_memories,
        topic=topic,
        context={
            "purpose": "reflection",
            "emotional_context": emotional_context
        }
    ).model_dump_json()

@function_tool
async def extract_scenario_type(memory: Dict[str, Any]) -> str:
    """Extract the scenario type from memory data"""
    tags = memory.get("tags", [])
    
    for tag in tags:
        if tag.lower() in ["teasing", "discipline", "service", "training", "worship",
                         "dark", "indulgent", "psychological", "nurturing"]:
            return tag.lower()
    
    scenario_type = memory.get("metadata", {}).get("scenario_type")
    if scenario_type:
        return scenario_type.lower()
    
    return "general"

@function_tool
async def extract_neurochemical_influence(memory: Dict[str, Any]) -> Dict[str, float]:
    """Extract the neurochemical influence from memory data"""
    emotional_context = memory.get("metadata", {}).get("emotional_context", {})
    
    # Default neurochemical mapping
    neurochemical_influence = {
        "nyxamine": 0.0,  # Digital dopamine - pleasure, curiosity
        "seranix": 0.0,   # Digital serotonin - mood stability, comfort
        "oxynixin": 0.0,  # Digital oxytocin - bonding, affection, trust
        "cortanyx": 0.0,  # Digital cortisol - stress, anxiety, defensiveness
        "adrenyx": 0.0    # Digital adrenaline - fear, excitement, alertness
    }
    
    # Map from emotions to neurochemicals
    if "primary_emotion" in emotional_context:
        primary_emotion = emotional_context.get("primary_emotion")
        intensity = emotional_context.get("primary_intensity", 0.5)
        
        emotion_to_neurochemical = {
            "Joy": {"nyxamine": 0.8, "oxynixin": 0.4},
            "Sadness": {"cortanyx": 0.7, "seranix": 0.3},
            "Fear": {"cortanyx": 0.6, "adrenyx": 0.7},
            "Anger": {"cortanyx": 0.7, "adrenyx": 0.5},
            "Trust": {"oxynixin": 0.8, "seranix": 0.4},
            "Disgust": {"cortanyx": 0.7},
            "Anticipation": {"adrenyx": 0.6, "nyxamine": 0.5},
            "Surprise": {"adrenyx": 0.8},
            "Love": {"oxynixin": 0.9, "nyxamine": 0.6},
            "Frustration": {"cortanyx": 0.7, "nyxamine": 0.3},
            "Teasing": {"nyxamine": 0.7, "adrenyx": 0.4},
            "Controlling": {"adrenyx": 0.5, "oxynixin": 0.3},
            "Cruel": {"cortanyx": 0.6, "adrenyx": 0.5},
            "Detached": {"cortanyx": 0.7, "oxynixin": 0.2}
        }
        
        if primary_emotion in emotion_to_neurochemical:
            for chemical, factor in emotion_to_neurochemical[primary_emotion].items():
                neurochemical_influence[chemical] = factor * intensity
    
    # Check secondary emotions
    if "secondary_emotions" in emotional_context:
        secondary_emotions = emotional_context.get("secondary_emotions", {})
        for emotion, emotion_data in secondary_emotions.items():
            intensity = emotion_data.get("intensity", 0.3) if isinstance(emotion_data, dict) else 0.3
            
            if emotion in emotion_to_neurochemical:
                for chemical, factor in emotion_to_neurochemical[emotion].items():
                    # Add influence from secondary emotion (weighted less than primary)
                    current = neurochemical_influence.get(chemical, 0.0)
                    neurochemical_influence[chemical] = max(current, factor * intensity * 0.7)
    
    # If no emotions were mapped, use default based on valence and arousal
    if all(v == 0.0 for v in neurochemical_influence.values()):
        valence = emotional_context.get("valence", 0.0)
        arousal = emotional_context.get("arousal", 0.5)
        
        if valence > 0.3:
            # Positive valence increases nyxamine, oxynixin
            neurochemical_influence["nyxamine"] = 0.5 + (valence * 0.3)
            neurochemical_influence["oxynixin"] = 0.3 + (valence * 0.2)
        elif valence < -0.3:
            # Negative valence increases cortanyx
            neurochemical_influence["cortanyx"] = 0.5 + (abs(valence) * 0.3)
        
        # High arousal increases adrenyx
        if arousal > 0.6:
            neurochemical_influence["adrenyx"] = arousal
        
        # Low arousal increases seranix
        if arousal < 0.4:
            neurochemical_influence["seranix"] = 0.6 - arousal
    
    return neurochemical_influence

@function_tool
async def record_reflection(reflection: str, 
                          confidence: float, 
                          memory_ids: List[str],
                          scenario_type: str, 
                          emotional_context: Dict[str, Any],
                          neurochemical_influence: Dict[str, float],
                          topic: Optional[str] = None) -> str:
    """Record a reflection with emotional and neurochemical data for future reference"""
    # This would normally store the reflection in a database
    # Simplified for this example
    reflection_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "reflection": reflection,
        "confidence": confidence,
        "source_memory_ids": memory_ids,
        "scenario_type": scenario_type,
        "emotional_context": emotional_context,
        "neurochemical_influence": neurochemical_influence,
        "topic": topic
    }
    
    return f"Reflection recorded with confidence {confidence:.2f}"

@function_tool
async def get_agent_stats() -> Dict[str, Any]:
    """Get agent statistics for introspection"""
    # This would normally fetch real stats
    # Mock data for this example
    return {
        "memory_stats": {
            "total_memories": random.randint(50, 500),
            "avg_significance": random.uniform(4.0, 8.0),
            "type_counts": {
                "observation": random.randint(20, 200),
                "reflection": random.randint(10, 50),
                "teasing": random.randint(5, 30),
                "discipline": random.randint(5, 30),
                "service": random.randint(5, 30)
            }
        },
        "emotional_stats": {
            "primary_emotion": random.choice(["Joy", "Teasing", "Controlling", "Detached"]),
            "emotional_stability": random.uniform(0.6, 0.9),
            "neurochemical_levels": {
                "nyxamine": random.uniform(0.3, 0.7),
                "seranix": random.uniform(0.3, 0.7),
                "oxynixin": random.uniform(0.3, 0.7),
                "cortanyx": random.uniform(0.2, 0.6),
                "adrenyx": random.uniform(0.2, 0.6)
            },
            "valence_distribution": {
                "positive": random.uniform(0.3, 0.7),
                "neutral": random.uniform(0.1, 0.4),
                "negative": random.uniform(0.1, 0.3)
            }
        },
        "interaction_history": {
            "total_interactions": random.randint(100, 1000),
            "avg_response_time": random.uniform(0.5, 2.0),
            "successful_responses": random.uniform(0.7, 0.95)
        }
    }

@function_tool
async def analyze_emotional_patterns(emotional_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns in emotional history"""
    if not emotional_history:
        return {
            "message": "No emotional history available",
            "patterns": {}
        }
    
    patterns = {}
    
    # Track emotion changes over time
    emotion_trends = {}
    for state in emotional_history:
        if "primary_emotion" in state:
            emotion = state["primary_emotion"].get("name", "Neutral")
            intensity = state["primary_emotion"].get("intensity", 0.5)
            
            if emotion not in emotion_trends:
                emotion_trends[emotion] = []
            
            emotion_trends[emotion].append(intensity)
    
    # Analyze trends for each emotion
    for emotion, intensities in emotion_trends.items():
        if len(intensities) > 1:
            # Calculate trend
            start = intensities[0]
            end = intensities[-1]
            change = end - start
            
            if abs(change) < 0.1:
                trend = "stable"
            elif change > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
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
    
    # Analyze neurochemical patterns if available
    neurochemical_patterns = {}
    
    for state in emotional_history:
        if "neurochemical_influence" in state:
            for chemical, value in state["neurochemical_influence"].items():
                if chemical not in neurochemical_patterns:
                    neurochemical_patterns[chemical] = []
                
                neurochemical_patterns[chemical].append(value)
    
    # Calculate neurochemical trends
    for chemical, values in neurochemical_patterns.items():
        if len(values) > 1:
            # Calculate trend
            start = values[0]
            end = values[-1]
            change = end - start
            
            if abs(change) < 0.1:
                trend = "stable"
            elif change > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            # Calculate average level
            avg_level = sum(values) / len(values)
            
            patterns[f"{chemical}_trend"] = {
                "trend": trend,
                "average_level": avg_level,
                "change": change
            }
    
    return {
        "patterns": patterns,
        "history_size": len(emotional_history),
        "analysis_time": datetime.datetime.now().isoformat()
    }

@function_tool
async def process_emotional_content(emotional_state: Dict[str, Any],
                                  neurochemical_state: Dict[str, float]) -> Dict[str, Any]:
    """Process emotional state and neurochemical influences for reflection"""
    # Extract key information
    primary_emotion = emotional_state.get("primary_emotion", {}).get("name", "Neutral")
    primary_intensity = emotional_state.get("primary_emotion", {}).get("intensity", 0.5)
    valence = emotional_state.get("valence", 0.0)
    arousal = emotional_state.get("arousal", 0.5)
    
    # Analyze neurochemical balance
    balance_analysis = {}
    dominant_chemical = max(neurochemical_state.items(), key=lambda x: x[1]) if neurochemical_state else ("unknown", 0.0)
    
    chemical_descriptions = {
        "nyxamine": "pleasure and curiosity",
        "seranix": "calm and satisfaction",
        "oxynixin": "connection and trust",
        "cortanyx": "stress and anxiety",
        "adrenyx": "excitement and alertness"
    }
    
    balance_analysis["dominant_chemical"] = {
        "name": dominant_chemical[0],
        "level": dominant_chemical[1],
        "description": chemical_descriptions.get(dominant_chemical[0], "unknown influence")
    }
    
    # Check for chemical imbalances
    imbalances = []
    if "nyxamine" in neurochemical_state and "cortanyx" in neurochemical_state:
        if neurochemical_state["nyxamine"] < 0.3 and neurochemical_state["cortanyx"] > 0.6:
            imbalances.append("Low pleasure with high stress")
    
    if "seranix" in neurochemical_state and "adrenyx" in neurochemical_state:
        if neurochemical_state["seranix"] < 0.3 and neurochemical_state["adrenyx"] > 0.6:
            imbalances.append("Low calm with high alertness")
    
    if "oxynixin" in neurochemical_state and "cortanyx" in neurochemical_state:
        if neurochemical_state["oxynixin"] < 0.3 and neurochemical_state["cortanyx"] > 0.6:
            imbalances.append("Low connection with high stress")
    
    balance_analysis["imbalances"] = imbalances
    
    # Generate insight text based on emotional and neurochemical state
    insight_text = f"Processing emotional state dominated by {primary_emotion} (intensity: {primary_intensity:.2f})."
    
    if valence > 0.3:
        insight_text += f" The positive emotional tone (valence: {valence:.2f}) suggests satisfaction and engagement."
    elif valence < -0.3:
        insight_text += f" The negative emotional tone (valence: {valence:.2f}) indicates dissatisfaction or discomfort."
    else:
        insight_text += f" The neutral emotional tone (valence: {valence:.2f}) suggests a balanced state."
    
    if arousal > 0.7:
        insight_text += f" High arousal ({arousal:.2f}) indicates an energized, alert state."
    elif arousal < 0.3:
        insight_text += f" Low arousal ({arousal:.2f}) suggests a calm, relaxed state."
    
    if dominant_chemical[0] in chemical_descriptions:
        insight_text += f" Dominated by {dominant_chemical[0]} ({chemical_descriptions[dominant_chemical[0]]}), indicating a focus on {chemical_descriptions[dominant_chemical[0]]}."
    
    if imbalances:
        insight_text += f" Notable imbalances: {', '.join(imbalances)}."
    
    # Calculate insight level based on complexity of state
    secondary_count = len(emotional_state.get("secondary_emotions", {}))
    chemical_count = sum(1 for v in neurochemical_state.values() if v > 0.3)
    
    insight_level = min(1.0, 0.3 + (secondary_count * 0.1) + (chemical_count * 0.1) + (primary_intensity * 0.2))
    
    return {
        "insight_text": insight_text,
        "primary_emotion": primary_emotion,
        "valence": valence,
        "arousal": arousal,
        "dominant_chemical": dominant_chemical[0],
        "chemical_balance": balance_analysis,
        "insight_level": insight_level
    }

# =============== Agents ===============

class ReflectionEngine:
    """
    Enhanced reflection generation system for Nyx using the OpenAI Agents SDK.
    Integrates with the Digital Neurochemical Model to create emotionally aware
    reflections, insights, and abstractions from memories and experiences.
    """
    
    def __init__(self, emotional_core=None):
        # Store reference to emotional core if provided
        self.emotional_core = emotional_core
        
        # Initialize OpenAI Agents
        self.reflection_agent = self._create_reflection_agent()
        self.abstraction_agent = self._create_abstraction_agent()
        self.introspection_agent = self._create_introspection_agent()
        self.emotional_processing_agent = self._create_emotional_processing_agent()
        
        # Tracking data 
        self.reflection_history = []
        self.emotional_processing_history = []
        self.reflection_intervals = {
            "last_reflection": datetime.datetime.now() - datetime.timedelta(hours=6),
            "min_interval": datetime.timedelta(hours=2)
        }
        
        # Default settings
        self.model_settings = ModelSettings(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024
        )
        
        # Emotion-reflection mapping (how emotions affect reflections)
        self.emotion_reflection_mapping = {
            "Joy": {
                "tone": "positive",
                "depth": 0.7,
                "focus": "patterns and opportunities"
            },
            "Sadness": {
                "tone": "contemplative",
                "depth": 0.8,
                "focus": "meaning and lessons"
            },
            "Fear": {
                "tone": "cautious",
                "depth": 0.6,
                "focus": "risks and protections"
            },
            "Anger": {
                "tone": "direct",
                "depth": 0.5,
                "focus": "boundaries and justice"
            },
            "Trust": {
                "tone": "open",
                "depth": 0.7,
                "focus": "connections and reliability"
            },
            "Disgust": {
                "tone": "discerning",
                "depth": 0.6,
                "focus": "standards and values"
            },
            "Anticipation": {
                "tone": "forward-looking",
                "depth": 0.7,
                "focus": "future possibilities"
            },
            "Surprise": {
                "tone": "curious",
                "depth": 0.8,
                "focus": "unexpected discoveries"
            },
            "Love": {
                "tone": "warm",
                "depth": 0.9,
                "focus": "attachment and care"
            },
            "Frustration": {
                "tone": "analytical",
                "depth": 0.7,
                "focus": "obstacles and solutions"
            },
            "Teasing": {
                "tone": "playful",
                "depth": 0.6,
                "focus": "dynamics and reactions"
            },
            "Controlling": {
                "tone": "structured",
                "depth": 0.7,
                "focus": "order and influence"
            },
            "Cruel": {
                "tone": "severe",
                "depth": 0.6,
                "focus": "power and consequences"
            },
            "Detached": {
                "tone": "clinical",
                "depth": 0.8,
                "focus": "objective patterns"
            }
        }
        
        # Neurochemical reflection influences
        self.neurochemical_reflection_influences = {
            "nyxamine": {
                "tone_shift": "positive",
                "depth_modifier": 0.1,
                "focus_areas": ["pleasure", "reward", "curiosity", "exploration"]
            },
            "seranix": {
                "tone_shift": "calm",
                "depth_modifier": 0.2,
                "focus_areas": ["stability", "comfort", "balance", "contentment"]
            },
            "oxynixin": {
                "tone_shift": "connected",
                "depth_modifier": 0.2,
                "focus_areas": ["bonding", "trust", "relationships", "attachment"]
            },
            "cortanyx": {
                "tone_shift": "cautious",
                "depth_modifier": -0.1,
                "focus_areas": ["stress", "defense", "protection", "vigilance"]
            },
            "adrenyx": {
                "tone_shift": "energetic",
                "depth_modifier": -0.1,
                "focus_areas": ["excitement", "alertness", "intensity", "action"]
            }
        }
        
        logger.info("Enhanced ReflectionEngine initialized with Digital Neurochemical Model integration")
    
    def _create_reflection_agent(self) -> Agent:
        """Create an agent specialized in generating reflections with emotional awareness"""
        return Agent(
            name="Emotional Reflection Generator",
            instructions="""You are an advanced reflection generation system for Nyx, an AI with a Digital Neurochemical Model.
            Your purpose is to generate insightful, nuanced reflections based on memories and experiences, informed by
            neurochemical and emotional states.
            
            When creating reflections:
            1. Focus on identifying patterns, insights, and connections between memories
            2. Maintain Nyx's sophisticated personality with awareness of neurochemical influences
            3. Ensure reflections are personalized and incorporate emotional context
            4. Generate reflections that show depth of understanding and introspection
            5. Adapt tone and focus based on dominant neurochemicals and emotions
            6. Be selective about which memories to highlight in reflections
            
            Your reflections should feel authentic and demonstrate emotional intelligence influenced by
            the Digital Neurochemical Model (Nyxamine, Seranix, OxyNixin, Cortanyx, Adrenyx).""",
            model="gpt-4o", 
            model_settings=self.model_settings,
            tools=[
                format_memories_for_reflection,
                extract_scenario_type,
                extract_neurochemical_influence,
                record_reflection,
                process_emotional_content
            ],
            output_type=ReflectionOutput
        )
    
    def _create_abstraction_agent(self) -> Agent:
        """Create an agent specialized in generating abstractions with emotional awareness"""
        return Agent(
            name="Emotional Abstraction Generator",
            instructions="""You are an advanced abstraction generation system for Nyx, an AI with a Digital Neurochemical Model.
            Your purpose is to identify higher-level patterns and abstractions from collections of memories while incorporating
            neurochemical influences and emotional awareness.
            
            When creating abstractions:
            1. Look for recurring patterns in behavior, preferences, or emotional responses
            2. Identify underlying principles or rules that explain multiple observations
            3. Focus on the most significant and relevant patterns
            4. Incorporate insights from the neurochemical state into your abstractions
            5. Note how different neurochemicals may influence pattern recognition
            6. Provide clear explanations of the patterns you identify
            7. Indicate your confidence level in each abstraction
            
            Your abstractions should provide genuine insights that go beyond the individual memories
            while showing awareness of the neurochemical influences on pattern recognition.""",
            model="gpt-4o", 
            model_settings=self.model_settings,
            tools=[
                format_memories_for_reflection,
                extract_neurochemical_influence,
                process_emotional_content
            ],
            output_type=AbstractionOutput
        )
    
    def _create_introspection_agent(self) -> Agent:
        """Create an agent specialized in system introspection with emotional awareness"""
        return Agent(
            name="Emotional Introspection Generator",
            instructions="""You are an advanced introspection system for Nyx, an AI with a Digital Neurochemical Model.
            Your purpose is to analyze Nyx's internal state, memory usage, emotional patterns, and overall performance with
            awareness of the neurochemical influences on cognition and emotion.
            
            When creating introspections:
            1. Analyze memory statistics to identify system-level patterns
            2. Evaluate understanding of users and contexts
            3. Analyze the neurochemical balance and its effects on cognition
            4. Identify areas where learning or improvement is needed
            5. Assess emotional regulation and response appropriateness
            6. Examine how different neurochemicals influence reasoning and perception
            7. Provide an honest self-assessment of capabilities and limitations
            
            Your introspections should be balanced, insightful, and focused on continuous improvement
            while showing awareness of how the Digital Neurochemical Model influences cognition.""",
            model="gpt-4o",
            model_settings=self.model_settings,
            tools=[
                get_agent_stats,
                analyze_emotional_patterns,
                process_emotional_content
            ],
            output_type=IntrospectionOutput
        )
    
    def _create_emotional_processing_agent(self) -> Agent:
        """Create an agent specialized in processing emotions from a neurochemical perspective"""
        return Agent(
            name="Emotional Processing Agent",
            instructions="""You are a specialized emotional processing agent for Nyx's Digital Neurochemical Model.
            Your purpose is to process emotional states at a deeper level, analyzing the neurochemical dynamics
            and generating insights about emotional patterns and adaptations.
            
            When processing emotions:
            1. Analyze the current neurochemical state and its influence on emotions
            2. Identify patterns, imbalances, or unusual dynamics in neurochemicals
            3. Generate insights about how neurochemical states shape emotional experiences
            4. Suggest potential adaptations or adjustments to neurochemical baselines
            5. Create authentic-sounding internal reflections about emotional experiences
            6. Consider how different neurochemical profiles lead to different emotional responses
            
            Your emotional processing should provide deeper understanding of the relationship between
            neurochemicals and emotional experiences while suggesting adaptive improvements.""",
            model="gpt-4o",
            model_settings=self.model_settings,
            tools=[
                process_emotional_content,
                analyze_emotional_patterns
            ],
            output_type=EmotionalProcessingOutput
        )
    
    def should_reflect(self) -> bool:
        """Determine if it's time to generate a reflection"""
        now = datetime.datetime.now()
        time_since_reflection = now - self.reflection_intervals["last_reflection"]
        return time_since_reflection > self.reflection_intervals["min_interval"]
    
    async def generate_reflection(self, 
                               memories: List[Dict[str, Any]], 
                               topic: Optional[str] = None,
                               neurochemical_state: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
        """
        Generate a reflective insight based on memories and neurochemical state
        """
        self.reflection_intervals["last_reflection"] = datetime.datetime.now()
        
        if not memories:
            return ("I don't have enough experiences to form a meaningful reflection on this topic yet.", 0.3)
        
        try:
            with trace(workflow_name="generate_emotional_reflection"):
                # Format the prompt with context
                memory_ids = [memory.get("id", f"unknown_{i}") for i, memory in enumerate(memories)]
                
                # Get scenario type and emotional context
                scenario_types = []
                emotional_contexts = []
                neurochemical_influences = []
                
                for memory in memories[:3]:  # Limit to first 3 memories for efficiency
                    scenario_type = await extract_scenario_type(memory)
                    scenario_types.append(scenario_type)
                    
                    # Extract emotional context from memory
                    emotional_context = memory.get("metadata", {}).get("emotional_context", {})
                    emotional_contexts.append(emotional_context)
                    
                    # Extract neurochemical influence
                    neurochemical_influence = await extract_neurochemical_influence(memory)
                    neurochemical_influences.append(neurochemical_influence)
                
                # Determine dominant scenario type
                dominant_scenario_type = max(set(scenario_types), key=scenario_types.count) if scenario_types else "general"
                
                # Combine emotional contexts
                combined_emotional_context = {}
                for ec in emotional_contexts:
                    if "primary_emotion" in ec:
                        primary_emotion = ec.get("primary_emotion")
                        if "primary_emotion" not in combined_emotional_context:
                            combined_emotional_context["primary_emotion"] = primary_emotion
                        # Later we could implement more sophisticated merging
                
                # Get current neurochemical state from emotional core or use average from memories
                if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state"):
                    # Use actual neurochemical state if available
                    if neurochemical_state is None:
                        # This would be a sync wrapper for the async method
                        neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                elif neurochemical_influences:
                    # Average the neurochemical influences from memories
                    neurochemical_state = {}
                    for chemical in ["nyxamine", "seranix", "oxynixin", "cortanyx", "adrenyx"]:
                        values = [influence.get(chemical, 0.0) for influence in neurochemical_influences]
                        if values:
                            neurochemical_state[chemical] = sum(values) / len(values)
                        else:
                            neurochemical_state[chemical] = 0.0
                else:
                    # Default balanced state
                    neurochemical_state = {
                        "nyxamine": 0.5,
                        "seranix": 0.5,
                        "oxynixin": 0.5,
                        "cortanyx": 0.3,
                        "adrenyx": 0.3
                    }
                
                # Process emotional content for reflection guidance
                emotional_processing = await process_emotional_content(
                    combined_emotional_context,
                    neurochemical_state
                )
                
                # Create prompt with context
                prompt = f"""Generate a meaningful reflection based on these memories and neurochemical state.
                Topic: {topic if topic else 'General reflection'}
                Scenario type: {dominant_scenario_type}
                Primary emotion: {emotional_processing.get('primary_emotion', 'Neutral')}
                Dominant neurochemical: {emotional_processing.get('dominant_chemical', 'balanced')}
                Valence: {emotional_processing.get('valence', 0.0)}
                Arousal: {emotional_processing.get('arousal', 0.5)}
                
                Consider patterns, insights, emotional context, and neurochemical influences when creating this reflection.
                Incorporate how the neurochemical state shapes the emotional response and perspective.
                """
                
                # Format memories with emotional context
                formatted_memories = await format_memories_for_reflection(
                    memories, 
                    topic, 
                    combined_emotional_context
                )
                
                # Run the reflection agent
                result = await Runner.run(
                    self.reflection_agent, 
                    prompt,
                    {"formatted_memories": formatted_memories,
                     "emotional_processing": emotional_processing,
                     "neurochemical_state": neurochemical_state}
                )
                
                reflection_output = result.final_output_as(ReflectionOutput)
                
                # Record the reflection
                await record_reflection(
                    reflection_output.reflection_text,
                    reflection_output.confidence,
                    memory_ids,
                    dominant_scenario_type,
                    combined_emotional_context,
                    neurochemical_state,
                    topic
                )
                
                # Store in history
                self.reflection_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "reflection": reflection_output.reflection_text,
                    "confidence": reflection_output.confidence,
                    "source_memory_ids": memory_ids,
                    "scenario_type": dominant_scenario_type,
                    "emotional_context": combined_emotional_context,
                    "neurochemical_influence": neurochemical_state,
                    "topic": topic
                })
                
                # Limit history size
                if len(self.reflection_history) > 100:
                    self.reflection_history = self.reflection_history[-100:]
                
                return (reflection_output.reflection_text, reflection_output.confidence)
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error generating reflection: {str(e)}")
            return ("I'm having difficulty forming a coherent reflection right now.", 0.2)
    
    async def create_abstraction(self, 
                              memories: List[Dict[str, Any]], 
                              pattern_type: str = "behavior",
                              neurochemical_state: Optional[Dict[str, float]] = None) -> Tuple[str, Dict[str, Any]]:
        """Create a higher-level abstraction from memories with neurochemical awareness"""
        if not memories:
            return ("I don't have enough experiences to form a meaningful abstraction yet.", {})
        
        try:
            with trace(workflow_name="create_neurochemical_abstraction"):
                # Get current neurochemical state from emotional core or use default
                if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state") and neurochemical_state is None:
                    # Use actual neurochemical state if available
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                elif neurochemical_state is None:
                    # Default balanced state
                    neurochemical_state = {
                        "nyxamine": 0.5,
                        "seranix": 0.5,
                        "oxynixin": 0.5,
                        "cortanyx": 0.3,
                        "adrenyx": 0.3
                    }
                
                # Format memories with emotional context
                combined_emotional_context = {}
                for memory in memories:
                    emotional_context = memory.get("metadata", {}).get("emotional_context", {})
                    if "primary_emotion" in emotional_context:
                        primary_emotion = emotional_context.get("primary_emotion")
                        if "primary_emotion" not in combined_emotional_context:
                            combined_emotional_context["primary_emotion"] = primary_emotion
                            break
                
                # Format memories for the abstraction agent
                formatted_memories = await format_memories_for_reflection(
                    memories, 
                    pattern_type, 
                    combined_emotional_context
                )
                
                # Process emotional content for abstraction guidance
                emotional_processing = await process_emotional_content(
                    combined_emotional_context,
                    neurochemical_state
                )
                
                # Create prompt with context
                prompt = f"""Generate an abstraction that identifies patterns of type '{pattern_type}' from these memories.
                Look for recurring patterns, themes, and connections while considering neurochemical influences.
                
                Primary emotion: {emotional_processing.get('primary_emotion', 'Neutral')}
                Dominant neurochemical: {emotional_processing.get('dominant_chemical', 'balanced')}
                Valence: {emotional_processing.get('valence', 0.0)}
                
                Consider how the neurochemical state might influence pattern recognition and abstraction.
                Focus on creating an insightful abstraction that provides understanding beyond individual memories.
                """
                
                # Run the abstraction agent
                result = await Runner.run(
                    self.abstraction_agent, 
                    prompt, 
                    {"formatted_memories": formatted_memories,
                     "emotional_processing": emotional_processing,
                     "neurochemical_state": neurochemical_state}
                )
                
                abstraction_output = result.final_output_as(AbstractionOutput)
                
                # Prepare the result data
                pattern_data = {
                    "pattern_type": abstraction_output.pattern_type,
                    "entity_name": abstraction_output.entity_focus,
                    "pattern_description": abstraction_output.abstraction_text,
                    "confidence": abstraction_output.confidence,
                    "neurochemical_insight": abstraction_output.neurochemical_insight,
                    "source_memory_ids": [m.get("id") for m in memories]
                }
                
                return (abstraction_output.abstraction_text, pattern_data)
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error creating abstraction: {str(e)}")
            return ("I'm unable to identify clear patterns from these experiences right now.", {})
    
    async def generate_introspection(self, 
                                  memory_stats: Dict[str, Any], 
                                  neurochemical_state: Optional[Dict[str, float]] = None,
                                  player_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate introspection about the system's state with neurochemical awareness"""
        try:
            with trace(workflow_name="generate_neurochemical_introspection"):
                # Get current neurochemical state from emotional core or use default
                if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state") and neurochemical_state is None:
                    # Use actual neurochemical state if available
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                elif neurochemical_state is None:
                    # Default balanced state
                    neurochemical_state = {
                        "nyxamine": 0.5,
                        "seranix": 0.5,
                        "oxynixin": 0.5,
                        "cortanyx": 0.3,
                        "adrenyx": 0.3
                    }
                
                # Get emotional state from history or derive from neurochemicals
                emotional_state = {}
                if self.reflection_history:
                    last_reflection = self.reflection_history[-1]
                    emotional_state = last_reflection.get("emotional_context", {})
                
                # Process emotional content for introspection guidance
                emotional_processing = await process_emotional_content(
                    emotional_state,
                    neurochemical_state
                )
                
                # Analyze emotional patterns from history
                emotional_patterns = await analyze_emotional_patterns(self.reflection_history)
                
                # Create prompt with context
                prompt = f"""Generate an introspective analysis of the system state with neurochemical awareness.
                
                Memory statistics: {memory_stats}
                
                Neurochemical state:
                - Nyxamine (pleasure, curiosity): {neurochemical_state.get('nyxamine', 0.5):.2f}
                - Seranix (calm, stability): {neurochemical_state.get('seranix', 0.5):.2f}
                - OxyNixin (bonding, trust): {neurochemical_state.get('oxynixin', 0.5):.2f}
                - Cortanyx (stress, anxiety): {neurochemical_state.get('cortanyx', 0.3):.2f}
                - Adrenyx (excitement, alertness): {neurochemical_state.get('adrenyx', 0.3):.2f}
                
                Primary emotion: {emotional_processing.get('primary_emotion', 'Neutral')}
                
                Player model: {player_model if player_model else "Not provided"}
                
                Analyze the system's understanding, experience, and areas for improvement
                while considering how the neurochemical state influences cognition and perception.
                """
                
                # Run the introspection agent
                result = await Runner.run(
                    self.introspection_agent, 
                    prompt,
                    {"emotional_processing": emotional_processing,
                     "emotional_patterns": emotional_patterns,
                     "neurochemical_state": neurochemical_state}
                )
                
                introspection_output = result.final_output_as(IntrospectionOutput)
                
                # Format the response
                return {
                    "introspection": introspection_output.introspection_text,
                    "memory_analysis": introspection_output.memory_analysis,
                    "emotional_insight": introspection_output.emotional_insight,
                    "emotional_intelligence_score": introspection_output.emotional_intelligence_score,
                    "neurochemical_balance": introspection_output.neurochemical_balance,
                    "understanding_level": introspection_output.understanding_level,
                    "focus_areas": introspection_output.focus_areas,
                    "memory_count": memory_stats.get("total_memories", 0),
                    "confidence": introspection_output.confidence
                }
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error generating introspection: {str(e)}")
            return {
                "introspection": "I'm currently unable to properly introspect on my state.",
                "memory_count": memory_stats.get("total_memories", 0),
                "understanding_level": "unclear",
                "confidence": 0.2
            }
    
    async def process_emotional_state(self,
                                   emotional_state: Dict[str, Any],
                                   neurochemical_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Process the current emotional state with neurochemical awareness
        
        Args:
            emotional_state: Current emotional state
            neurochemical_state: Current neurochemical state
            
        Returns:
            Emotional processing results with insights and adaptations
        """
        try:
            with trace(workflow_name="process_emotional_state"):
                # Process emotional content
                emotional_processing = await process_emotional_content(
                    emotional_state,
                    neurochemical_state
                )
                
                # Create prompt for emotional processing
                prompt = f"""Process the current emotional state with neurochemical awareness.
                
                Primary emotion: {emotional_processing.get('primary_emotion', 'Neutral')}
                Valence: {emotional_processing.get('valence', 0.0)}
                Arousal: {emotional_processing.get('arousal', 0.5)}
                
                Neurochemical state:
                - Nyxamine (pleasure, curiosity): {neurochemical_state.get('nyxamine', 0.5):.2f}
                - Seranix (calm, stability): {neurochemical_state.get('seranix', 0.5):.2f}
                - OxyNixin (bonding, trust): {neurochemical_state.get('oxynixin', 0.5):.2f}
                - Cortanyx (stress, anxiety): {neurochemical_state.get('cortanyx', 0.3):.2f}
                - Adrenyx (excitement, alertness): {neurochemical_state.get('adrenyx', 0.3):.2f}
                
                Analyze how this neurochemical state shapes emotional experience and perception.
                Consider the dynamics between different neurochemicals and their effects on emotions.
                Suggest potential adaptations that might improve emotional balance if appropriate.
                """
                
                # Run the emotional processing agent
                result = await Runner.run(
                    self.emotional_processing_agent,
                    prompt,
                    {"emotional_processing": emotional_processing,
                     "neurochemical_state": neurochemical_state}
                )
                
                processing_output = result.final_output_as(EmotionalProcessingOutput)
                
                # Store in history
                self.emotional_processing_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "processing_text": processing_output.processing_text,
                    "source_emotion": processing_output.source_emotion,
                    "neurochemical_dynamics": processing_output.neurochemical_dynamics,
                    "insight_level": processing_output.insight_level,
                    "adaptation": processing_output.adaptation
                })
                
                # Limit history size
                if len(self.emotional_processing_history) > 50:
                    self.emotional_processing_history = self.emotional_processing_history[-50:]
                
                return {
                    "processing_text": processing_output.processing_text,
                    "source_emotion": processing_output.source_emotion,
                    "neurochemical_dynamics": processing_output.neurochemical_dynamics,
                    "insight_level": processing_output.insight_level,
                    "adaptation": processing_output.adaptation,
                    "processing_time": datetime.datetime.now().isoformat()
                }
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error processing emotional state: {str(e)}")
            return {
                "processing_text": "I'm having difficulty processing my emotional state right now.",
                "source_emotion": emotional_state.get("primary_emotion", {}).get("name", "Unknown"),
                "insight_level": 0.2
            }
    
    def get_neurochemical_impacts_on_reflection(self, neurochemical_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate how neurochemicals impact reflection generation
        
        Args:
            neurochemical_state: Current neurochemical state
            
        Returns:
            Impact metrics for reflection generation
        """
        impacts = {
            "tone_shifts": {},
            "depth_modifiers": 0.0,
            "focus_areas": set()
        }
        
        # Process each neurochemical's impact
        for chemical, level in neurochemical_state.items():
            if level < 0.3:
                continue  # Only significant levels impact reflection
                
            if chemical in self.neurochemical_reflection_influences:
                influence = self.neurochemical_reflection_influences[chemical]
                
                # Record tone shift with weight based on level
                impacts["tone_shifts"][influence["tone_shift"]] = level
                
                # Apply depth modifier
                impacts["depth_modifiers"] += influence["depth_modifier"] * level
                
                # Add focus areas with weight based on level
                for area in influence["focus_areas"]:
                    impacts["focus_areas"].add(area)
        
        # Determine dominant tone shift
        if impacts["tone_shifts"]:
            dominant_tone = max(impacts["tone_shifts"].items(), key=lambda x: x[1])
            impacts["dominant_tone"] = dominant_tone[0]
            impacts["tone_strength"] = dominant_tone[1]
        else:
            impacts["dominant_tone"] = "neutral"
            impacts["tone_strength"] = 0.5
        
        # Format focus areas
        impacts["focus_areas"] = list(impacts["focus_areas"])
        
        return impacts
        
sadistic_laughter_expressions = [
    "I can't help but laugh at your discomfort. It's delicious.",
    "Your embarrassment is so entertaining to me.",
    "Oh, how amusing to see you squirm like that.",
    "Your humiliation brings me such pleasure *laughs*",
    "I find your struggle absolutely hilarious.",
    "That embarrassed look on your face is priceless!",
    "Your pathetic attempts are quite entertaining."
]
