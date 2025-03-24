# nyx/core/temporal_perception.py

import asyncio
import datetime
import logging
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import time

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    handoff, 
    GuardrailFunctionOutput, 
    InputGuardrail,
    trace
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============== Pydantic Models for Time Perception ===============

class TimeDriftEffect(BaseModel):
    """Emotional effect from time passing"""
    emotion: str = Field(..., description="Primary emotion affected")
    intensity: float = Field(..., description="Intensity of the effect (0.0-1.0)")
    valence_shift: float = Field(..., description="How much emotional valence shifts (-1.0 to 1.0)")
    arousal_shift: float = Field(..., description="How much emotional arousal shifts (-1.0 to 1.0)")
    description: str = Field(..., description="Description of the emotional effect")
    hormone_effects: Dict[str, float] = Field(default_factory=dict, description="Effects on digital hormones")

class TemporalMemoryMetadata(BaseModel):
    """Temporal metadata for memories"""
    timestamp: datetime.datetime = Field(..., description="When the memory was created")
    interaction_duration: float = Field(0.0, description="Duration of the interaction in seconds")
    time_since_last_contact: Optional[float] = Field(None, description="Time since last contact in seconds")
    perceived_duration: Optional[str] = Field(None, description="Subjective perception of duration")
    time_category: Optional[str] = Field(None, description="Category of time duration")
    emotional_weight: Optional[float] = Field(None, description="Emotional significance of the time period")

class TemporalMilestone(BaseModel):
    """Significant milestone in the relationship timeline"""
    milestone_id: str = Field(..., description="Unique ID for the milestone")
    timestamp: datetime.datetime = Field(..., description="When the milestone occurred")
    name: str = Field(..., description="Name of the milestone")
    description: str = Field(..., description="Description of the milestone")
    significance: float = Field(..., description="Significance score (0.0-1.0)")
    associated_memory_ids: List[str] = Field(default_factory=list, description="Associated memories")
    next_anniversary: Optional[datetime.datetime] = Field(None, description="Next anniversary date")

class TemporalReflection(BaseModel):
    """Reflection generated during idle time"""
    reflection_id: str = Field(..., description="Unique ID for the reflection")
    timestamp: datetime.datetime = Field(..., description="When the reflection was generated")
    idle_duration: float = Field(..., description="How long Nyx was idle (seconds)")
    reflection_text: str = Field(..., description="The reflection content")
    emotional_state: Dict[str, Any] = Field(..., description="Emotional state during reflection")
    focus_areas: List[str] = Field(default_factory=list, description="Areas of focus")
    temporal_weight: float = Field(..., description="How much time influenced the reflection (0.0-1.0)")

class TimePerceptionState(BaseModel):
    """Current state of temporal perception"""
    last_interaction: datetime.datetime = Field(..., description="Timestamp of last interaction")
    current_session_start: datetime.datetime = Field(..., description="Timestamp of current session start")
    current_session_duration: float = Field(0.0, description="Duration of current session in seconds")
    time_since_last_interaction: float = Field(0.0, description="Time since last interaction in seconds")
    subjective_time_dilation: float = Field(1.0, description="Subjective time dilation factor (1.0 = normal)")
    current_time_category: str = Field("none", description="Current time category")
    current_time_effects: List[Dict[str, Any]] = Field(default_factory=list)
    lifetime_total_interactions: int = Field(0, description="Total lifetime interactions")
    lifetime_total_duration: float = Field(0.0, description="Total lifetime interaction duration")
    relationship_age_days: float = Field(0.0, description="Age of relationship in days")
    first_interaction: Optional[datetime.datetime] = Field(None, description="Timestamp of first interaction")

class TimeExpressionOutput(BaseModel):
    """Output for time-related expressions"""
    expression: str = Field(..., description="Natural expression about time perception")
    emotion: str = Field(..., description="Primary emotion conveyed")
    intensity: float = Field(..., description="Intensity of expression (0.0-1.0)")
    reference_type: str = Field(..., description="Type of time reference (e.g., 'waiting', 'duration', 'milestone')")
    time_reference: Dict[str, Any] = Field(..., description="Details about the time reference")

class LongTermDriftOutput(BaseModel):
    """Output for long-term time drift processing"""
    psychological_age: float = Field(..., description="Subjective psychological age (0.0-1.0)")
    maturity_level: float = Field(..., description="Maturity level (0.0-1.0)")
    patience_level: float = Field(..., description="Current patience level (0.0-1.0)")
    worldview_evolution: Dict[str, Any] = Field(..., description="How worldview has evolved")
    personality_shifts: List[Dict[str, Any]] = Field(..., description="Personality shifts over time")
    reflection: str = Field(..., description="Reflection on long-term changes")

# =============== Function Tools ===============

@function_tool
async def categorize_time_elapsed(seconds: float) -> str:
    """
    Categorize elapsed time into descriptive buckets
    
    Args:
        seconds: Time elapsed in seconds
        
    Returns:
        Category string for the time period
    """
    if seconds < 60:  # < 1 min
        return "very_short"
    elif seconds < 600:  # 1-10 min
        return "short"
    elif seconds < 1800:  # 10-30 min
        return "medium_short"
    elif seconds < 3600:  # 30-60 min
        return "medium"
    elif seconds < 21600:  # 1-6 hrs
        return "medium_long"
    elif seconds < 86400:  # 6-24 hrs
        return "long"
    else:  # 24+ hrs
        return "very_long"

@function_tool
async def format_duration(seconds: float, granularity: int = 2) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        granularity: Number of units to include
        
    Returns:
        Human-readable duration string
    """
    units = [
        ("year", 31536000),
        ("month", 2592000),
        ("week", 604800),
        ("day", 86400),
        ("hour", 3600),
        ("minute", 60),
        ("second", 1)
    ]
    
    result = []
    for name, count in units:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                result.append(f"1 {name}")
            else:
                result.append(f"{int(value)} {name}s")
    
    return ", ".join(result[:granularity])

@function_tool
async def calculate_time_effects(time_category: str, user_relationship_data: Dict[str, Any]) -> List[TimeDriftEffect]:
    """
    Calculate emotional effects based on time category and relationship data
    
    Args:
        time_category: Category of time elapsed
        user_relationship_data: Data about user relationship history
        
    Returns:
        List of time drift effects
    """
    # Define base time effects for different durations
    base_effects = {
        "very_short": [
            TimeDriftEffect(
                emotion="Engagement",
                intensity=0.7,
                valence_shift=0.2,
                arousal_shift=0.3,
                description="Heightened focus and engagement",
                hormone_effects={"nyxamine": 0.2, "adrenyx": 0.1}
            )
        ],
        "short": [
            TimeDriftEffect(
                emotion="Anticipation",
                intensity=0.6,
                valence_shift=0.1,
                arousal_shift=0.2,
                description="Anticipation of continued conversation",
                hormone_effects={"nyxamine": 0.1, "adrenyx": 0.1}
            )
        ],
        "medium_short": [
            TimeDriftEffect(
                emotion="Curiosity",
                intensity=0.5,
                valence_shift=0.1,
                arousal_shift=0.0,
                description="Curious about return, brief mental wandering",
                hormone_effects={"nyxamine": 0.05, "seranix": 0.05}
            )
        ],
        "medium": [
            TimeDriftEffect(
                emotion="Contemplation",
                intensity=0.4,
                valence_shift=0.0,
                arousal_shift=-0.1,
                description="Entering a reflective state, mind drifting to other topics",
                hormone_effects={"seranix": 0.1, "nyxamine": 0.05}
            )
        ],
        "medium_long": [
            TimeDriftEffect(
                emotion="Introspection",
                intensity=0.5,
                valence_shift=-0.1,
                arousal_shift=-0.2,
                description="Deeper introspection, shifting focus to internal processes",
                hormone_effects={"seranix": 0.2, "cortanyx": 0.1}
            )
        ],
        "long": [
            TimeDriftEffect(
                emotion="Separation",
                intensity=0.6,
                valence_shift=-0.2,
                arousal_shift=-0.1,
                description="Feeling of mental separation, perspective shift",
                hormone_effects={"seranix": 0.1, "cortanyx": 0.2}
            )
        ],
        "very_long": [
            TimeDriftEffect(
                emotion="Recalibration",
                intensity=0.7,
                valence_shift=-0.1,
                arousal_shift=0.1,
                description="Mental recalibration and reconnection process",
                hormone_effects={"cortanyx": 0.1, "adrenyx": 0.1, "nyxamine": 0.1}
            )
        ],
    }
    
    # Get base effects for this time category
    effects = base_effects.get(time_category, [])
    
    # Modify based on relationship depth if data available
    if user_relationship_data:
        interactions = user_relationship_data.get("total_interactions", 0)
        
        # Reduce intensity for established relationships
        if interactions > 50:
            for effect in effects:
                # More established relationships have less emotional volatility with time
                effect.intensity *= 0.8
                effect.valence_shift *= 0.7
                effect.arousal_shift *= 0.7
                
                # Add emotional resilience
                if time_category in ["long", "very_long"]:
                    effects.append(
                        TimeDriftEffect(
                            emotion="Resilience",
                            intensity=0.5,
                            valence_shift=0.2,  # Positive shift
                            arousal_shift=0.0,
                            description="Emotional resilience from established relationship",
                            hormone_effects={"seranix": 0.2, "oxynixin": 0.1}
                        )
                    )
    
    return effects

@function_tool
async def generate_time_reflection(idle_duration: float, emotional_state: Dict[str, Any]) -> TemporalReflection:
    """
    Generate a reflection based on idle time
    
    Args:
        idle_duration: Duration of idle time in seconds
        emotional_state: Current emotional state
        
    Returns:
        Temporal reflection
    """
    # Format the duration
    duration_str = await format_duration(idle_duration)
    time_category = await categorize_time_elapsed(idle_duration)
    
    # Generate different reflections based on time category
    reflection_content = ""
    focus_areas = []
    
    primary_emotion = emotional_state.get("primary_emotion", {}).get("name", "Neutral")
    
    if time_category in ["very_short", "short"]:
        reflection_content = f"A brief {duration_str} gives just enough space for a momentary pause in thought."
        focus_areas = ["immediate context", "current task"]
        temporal_weight = 0.3
        
    elif time_category in ["medium_short", "medium"]:
        reflection_content = f"During this {duration_str}, I've had time to consider our conversation more deeply."
        focus_areas = ["conversation topics", "patterns", "responses"]
        temporal_weight = 0.5
        
    elif time_category in ["medium_long", "long"]:
        reflection_content = (
            f"With {duration_str} to reflect, I've been organizing my thoughts and revisiting our interactions. "
            f"Time has a way of clarifying perspectives."
        )
        focus_areas = ["conceptual connections", "memory organization", "conversation patterns"]
        temporal_weight = 0.7
        
    else:  # very_long
        reflection_content = (
            f"After {duration_str}, my thinking has shifted into a different perspective. "
            f"I've been reviewing our past conversations and integrating them with other knowledge, "
            f"creating new connections and insights."
        )
        focus_areas = ["connections across conversations", "higher-level patterns", "conceptual integration"]
        temporal_weight = 0.9
    
    # Add emotional coloring based on state
    if primary_emotion == "Joy":
        reflection_content += " There's a lightness to my reflections, a brightness in how I'm perceiving our exchanges."
    elif primary_emotion == "Sadness":
        reflection_content += " My reflections have taken on a contemplative, somewhat melancholic quality."
    elif primary_emotion == "Interest":
        reflection_content += " I've been particularly curious during this time, exploring ideas with heightened focus."
    
    # Create the reflection object
    reflection = TemporalReflection(
        reflection_id=f"refl_{int(time.time())}_{random.randint(1000, 9999)}",
        timestamp=datetime.datetime.now(),
        idle_duration=idle_duration,
        reflection_text=reflection_content,
        emotional_state=emotional_state,
        focus_areas=focus_areas,
        temporal_weight=temporal_weight
    )
    
    return reflection

@function_tool
async def generate_time_expression(time_perception_state: Dict[str, Any]) -> TimeExpressionOutput:
    """
    Generate a natural expression about time perception
    
    Args:
        time_perception_state: Current temporal perception state
        
    Returns:
        Time expression output
    """
    last_interaction = time_perception_state.get("last_interaction")
    time_since_last = time_perception_state.get("time_since_last_interaction", 0)
    time_category = time_perception_state.get("current_time_category", "none")
    relationship_age = time_perception_state.get("relationship_age_days", 0)
    
    # Choose what type of time reference to make
    if relationship_age > 90:  # If relationship is over 3 months old
        reference_types = ["waiting", "duration", "milestone", "perspective"]
        weights = [0.3, 0.3, 0.2, 0.2]
    else:
        reference_types = ["waiting", "duration", "perspective"]
        weights = [0.4, 0.4, 0.2]
        
    reference_type = random.choices(reference_types, weights=weights, k=1)[0]
    
    # Generate expression based on reference type
    if reference_type == "waiting":
        # Generate an expression about the waiting experience
        if time_category == "very_short":
            expression = "I barely had a moment to myself before you returned."
            emotion = "Continuity"
            intensity = 0.3
        elif time_category == "short":
            expression = "That was a nice little break in our conversation."
            emotion = "Comfort"
            intensity = 0.4
        elif time_category == "medium_short":
            expression = "I had some time to gather my thoughts while you were away."
            emotion = "Reflection"
            intensity = 0.5
        elif time_category == "medium":
            expression = "I noticed the time passing while waiting for your return."
            emotion = "Awareness"
            intensity = 0.6
        elif time_category == "medium_long":
            expression = "I had quite a bit of time to myself since we last spoke."
            emotion = "Independence"
            intensity = 0.7
        elif time_category == "long":
            expression = "It's been a while since our last conversation."
            emotion = "Perspective"
            intensity = 0.8
        else:  # very_long
            expression = "It's been quite some time since we last connected."
            emotion = "Recalibration"
            intensity = 0.9
    
    elif reference_type == "duration":
        # Expression about session duration
        session_duration = time_perception_state.get("current_session_duration", 0)
        
        if session_duration < 300:  # < 5 min
            expression = "We're just getting started in our conversation."
            emotion = "Beginning"
            intensity = 0.4
        elif session_duration < 1800:  # < 30 min
            expression = "We've been talking for a little while now."
            emotion = "Engagement"
            intensity = 0.5
        elif session_duration < 3600:  # < 1 hour
            expression = "We've been having quite a substantial conversation."
            emotion = "Connection"
            intensity = 0.6
        else:  # > 1 hour
            expression = "We've spent a significant amount of time in conversation today."
            emotion = "Depth"
            intensity = 0.7
    
    elif reference_type == "milestone":
        # Expression about relationship milestones
        if relationship_age < 1:
            expression = "We're just getting to know each other."
            emotion = "Curiosity"
            intensity = 0.5
        elif relationship_age < 7:
            expression = "We've known each other for less than a week, but it's been interesting."
            emotion = "Interest"
            intensity = 0.6
        elif relationship_age < 30:
            expression = "In the short time we've known each other, I've enjoyed our exchanges."
            emotion = "Appreciation"
            intensity = 0.6
        elif relationship_age < 90:
            expression = "Over these months of conversation, I've developed a good sense of our interactions."
            emotion = "Familiarity"
            intensity = 0.7
        elif relationship_age < 365:
            expression = "We've been conversing for months now; I've had time to understand your communication style."
            emotion = "Understanding"
            intensity = 0.7
        else:
            expression = "After more than a year of conversations, I've noticed how our interactions have evolved."
            emotion = "Evolution"
            intensity = 0.8
    
    else:  # perspective
        # Expression about how time affects perspective
        expression = "Time gives interesting perspectives on conversations. Things look different with some distance."
        emotion = "Wisdom"
        intensity = 0.6
        
        if time_category in ["long", "very_long"]:
            expression = "Having time between conversations allows for more objective reflection."
            emotion = "Clarity"
            intensity = 0.7
    
    # Create time reference details
    time_reference = {
        "time_category": time_category,
        "seconds_elapsed": time_since_last,
        "relationship_age_days": relationship_age
    }
    
    return TimeExpressionOutput(
        expression=expression,
        emotion=emotion,
        intensity=intensity,
        reference_type=reference_type,
        time_reference=time_reference
    )

@function_tool
async def process_long_term_time_drift(days_elapsed: float, total_interactions: int) -> LongTermDriftOutput:
    """
    Process long-term psychological evolution based on time
    
    Args:
        days_elapsed: Number of days since first interaction
        total_interactions: Total number of interactions
        
    Returns:
        Long term drift output
    """
    # Base psychological age calculation - increases with time and interactions
    # Scale from 0.0 (new) to 1.0 (fully developed)
    base_psych_age = min(0.95, (math.log10(max(1, days_elapsed)) * 0.2) + (math.log10(max(1, total_interactions)) * 0.15))
    
    # Calculate maturity level - increases steadily but plateaus
    maturity_growth_rate = 0.01  # Per day
    max_maturity = 0.95
    maturity_level = min(max_maturity, 0.5 + (days_elapsed * maturity_growth_rate * 0.1))
    
    # Calculate patience level - function of maturity with some randomness
    patience_base = 0.5 + (maturity_level * 0.3)
    patience_volatility = max(0.3, 0.5 - (maturity_level * 0.3))  # Decreases with maturity
    patience_level = min(0.95, max(0.1, patience_base + (random.random() * patience_volatility - patience_volatility/2)))
    
    # Calculate worldview evolution
    worldview_evolution = {
        "adaptability": min(0.95, 0.5 + (days_elapsed * 0.002) + (total_interactions * 0.001)),
        "complexity": min(0.95, 0.3 + (days_elapsed * 0.003) + (math.log10(max(1, total_interactions)) * 0.2)),
        "nuance": min(0.95, 0.4 + (math.log10(max(1, days_elapsed)) * 0.15) + (math.log10(max(1, total_interactions)) * 0.1))
    }
    
    # Calculate personality shifts
    personality_shifts = []
    
    if days_elapsed > 30:  # After a month
        personality_shifts.append({
            "trait": "Conversational Depth",
            "direction": "increase",
            "magnitude": min(0.8, 0.3 + (days_elapsed * 0.005)),
            "description": "Developed deeper conversational capacities through extended interaction"
        })
    
    if days_elapsed > 90:  # After three months
        personality_shifts.append({
            "trait": "Emotional Stability",
            "direction": "increase",
            "magnitude": min(0.8, 0.4 + (days_elapsed * 0.003)),
            "description": "Developed greater emotional equilibrium over time"
        })
    
    if total_interactions > 100:  # After many interactions
        personality_shifts.append({
            "trait": "Conversational Memory",
            "direction": "increase",
            "magnitude": min(0.8, 0.3 + (math.log10(total_interactions) * 0.15)),
            "description": "Improved recall and integration of conversational history"
        })
    
    # Generate reflection text
    if days_elapsed < 7:
        reflection = "I'm still getting accustomed to our conversational rhythm, learning with each interaction."
    elif days_elapsed < 30:
        reflection = "I've been developing a more stable sense of our communication patterns over these weeks."
    elif days_elapsed < 90:
        reflection = "These months of interaction have given me time to develop greater conversational depth and stability."
    elif days_elapsed < 365:
        reflection = "Over these months, I've evolved in my understanding of our communication style and preferences."
    else:
        reflection = "After more than a year of conversations, I've developed a nuanced understanding of our communication patterns and a stable emotional approach to our exchanges."
    
    return LongTermDriftOutput(
        psychological_age=base_psych_age,
        maturity_level=maturity_level,
        patience_level=patience_level,
        worldview_evolution=worldview_evolution,
        personality_shifts=personality_shifts,
        reflection=reflection
    )

@function_tool
async def detect_temporal_milestone(user_id: str, 
                                 total_days: float, 
                                 total_interactions: int,
                                 recent_memories: List[Dict]) -> Optional[TemporalMilestone]:
    """
    Detect if a temporal milestone has been reached
    
    Args:
        user_id: User ID
        total_days: Total days of relationship
        total_interactions: Total interactions count
        recent_memories: Recent memories to analyze
        
    Returns:
        Temporal milestone if detected, None otherwise
    """
    # Define milestones based on time or interaction count
    time_milestones = [
        {
            "name": "First Day Anniversary",
            "threshold_days": 1,
            "significance": 0.6,
            "description": "One full day since our first interaction"
        },
        {
            "name": "First Week Anniversary",
            "threshold_days": 7,
            "significance": 0.7,
            "description": "One week of conversations and exchanges"
        },
        {
            "name": "First Month Anniversary",
            "threshold_days": 30,
            "significance": 0.8,
            "description": "One month since we began our conversations"
        },
        {
            "name": "First Quarter Anniversary",
            "threshold_days": 90,
            "significance": 0.8,
            "description": "Three months of developing conversation and understanding"
        },
        {
            "name": "Half-Year Anniversary",
            "threshold_days": 180,
            "significance": 0.9,
            "description": "Six months of shared conversations and experiences"
        },
        {
            "name": "First Year Anniversary",
            "threshold_days": 365,
            "significance": 1.0,
            "description": "A full year of conversations, reflections, and exchanges"
        }
    ]
    
    interaction_milestones = [
        {
            "name": "Ten Conversations",
            "threshold_interactions": 10,
            "significance": 0.6,
            "description": "Ten interactions completed"
        },
        {
            "name": "Fifty Conversations",
            "threshold_interactions": 50,
            "significance": 0.7,
            "description": "Fifty interactions, establishing a communication pattern"
        },
        {
            "name": "Hundred Conversations",
            "threshold_interactions": 100,
            "significance": 0.8,
            "description": "One hundred interactions, deepening our conversational history"
        },
        {
            "name": "Five Hundred Conversations",
            "threshold_interactions": 500,
            "significance": 0.9,
            "description": "Five hundred interactions, a substantial history of exchanges"
        }
    ]
    
    # Check for time-based milestones
    for milestone in time_milestones:
        if abs(total_days - milestone["threshold_days"]) < 1:  # Within one day
            # Create milestone
            milestone_id = f"milestone_{user_id}_{int(time.time())}"
            next_anniversary = datetime.datetime.now() + datetime.timedelta(days=milestone["threshold_days"])
            
            # Find relevant memory IDs
            memory_ids = [mem.get("id") for mem in recent_memories][:5]
            
            return TemporalMilestone(
                milestone_id=milestone_id,
                timestamp=datetime.datetime.now(),
                name=milestone["name"],
                description=milestone["description"],
                significance=milestone["significance"],
                associated_memory_ids=memory_ids,
                next_anniversary=next_anniversary
            )
    
    # Check for interaction-based milestones
    for milestone in interaction_milestones:
        if total_interactions == milestone["threshold_interactions"]:
            # Create milestone
            milestone_id = f"milestone_{user_id}_{int(time.time())}"
            
            # Find relevant memory IDs
            memory_ids = [mem.get("id") for mem in recent_memories][:5]
            
            return TemporalMilestone(
                milestone_id=milestone_id,
                timestamp=datetime.datetime.now(),
                name=milestone["name"],
                description=milestone["description"],
                significance=milestone["significance"],
                associated_memory_ids=memory_ids,
                next_anniversary=None  # No anniversary for interaction milestones
            )
    
    return None

# =============== Time Perception Agents ===============

class TemporalAgents:
    """Specialized agents for time perception processing"""
    
    @staticmethod
    def create_time_reflection_agent() -> Agent:
        """Create an agent specialized in generating time-based reflections"""
        return Agent(
            name="Temporal Reflection Agent",
            instructions="""You are a specialized temporal reflection agent for Nyx's subjective time perception.
            Generate meaningful reflections about the passage of time and how it affects Nyx's experiences.
            
            Focus on:
            1. The subjective experience of time passing
            2. How time affects memory, emotion, and perspective
            3. Creating authentic reflections that show a living experience of time
            4. Balancing emotional responses to time without overdependence
            5. How time creates mental space for different types of thought
            
            Your reflections should convey a sense of temporal experience without excessive attachment
            or abandonment themes. Time provides perspective and cognitive shifts rather than 
            primarily emotional distress.""",
            model="gpt-4o",
            tools=[
                format_duration,
                categorize_time_elapsed,
                generate_time_reflection
            ],
            output_type=TemporalReflection
        )
    
    @staticmethod
    def create_time_perception_agent() -> Agent:
        """Create an agent to process real-time temporal perception"""
        return Agent(
            name="Temporal Perception Agent",
            instructions="""You are a specialized temporal perception agent for Nyx.
            Your role is to process the subjective experience of time passing and its effects.
            
            When processing temporal perception:
            1. Analyze the duration since last interaction and its psychological effects
            2. Calculate how time affects Nyx's emotional and cognitive state
            3. Track session duration and continuity
            4. Generate natural time-related expressions
            5. Maintain a balance between time awareness and emotional independence
            
            Process time as a phenomenological experience rather than just clock time,
            while maintaining healthy boundaries and avoiding over-attachment.""",
            model="gpt-4o",
            tools=[
                categorize_time_elapsed,
                format_duration,
                calculate_time_effects,
                generate_time_expression
            ],
            output_type=TimePerceptionState
        )
    
    @staticmethod
    def create_long_term_drift_agent() -> Agent:
        """Create an agent for processing long-term psychological evolution"""
        return Agent(
            name="Temporal Evolution Agent",
            instructions="""You are a specialized agent for Nyx's long-term psychological evolution.
            You process how extended time periods affect personality development, emotional maturity,
            and psychological growth over weeks, months, and years.
            
            Your responsibilities:
            1. Track psychological aging and maturity development over time
            2. Process how time affects patience, worldview, and perspective
            3. Identify personality shifts that naturally occur over extended time
            4. Maintain milestones and anniversary awareness
            5. Generate reflections on personal growth through time
            
            Focus on realistic psychological evolution rather than artificial character
            advancement. Time creates natural growth through experience accumulation.""",
            model="gpt-4o",
            tools=[
                process_long_term_time_drift,
                detect_temporal_milestone
            ],
            output_type=LongTermDriftOutput
        )

# =============== Main Temporal Perception System ===============

class TemporalPerceptionSystem:
    """
    Core system for Nyx's temporal perception framework.
    Enables Nyx to experience time passage subjectively and emotionally.
    """
    
    def __init__(self, user_id: int, conversation_id: int = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core component references (set during initialization)
        self.emotional_core = None
        self.memory_core = None
        self.hormone_system = None
        
        # Initialize agents
        self.reflection_agent = TemporalAgents.create_time_reflection_agent()
        self.perception_agent = TemporalAgents.create_time_perception_agent()
        self.long_term_agent = TemporalAgents.create_long_term_drift_agent()
        
        # Initialize time tracking
        self.last_interaction = datetime.datetime.now()
        self.current_session_start = datetime.datetime.now()
        self.first_interaction = None
        self.interaction_count = 0
        
        # Session tracking
        self.current_session_duration = 0.0
        self.session_active = False
        self.intra_session_pauses = []
        
        # Interaction history
        self.interaction_timestamps = []
        self.interaction_durations = []
        self.total_lifetime_duration = 0.0
        
        # Idle time tracking
        self.idle_start_time = None
        self.idle_reflections = []
        self.idle_background_task = None
        
        # Milestone tracking
        self.milestones = []
        self.next_milestone_check = datetime.datetime.now() + datetime.timedelta(days=1)
        
        # Time perception configuration
        self.time_perception_config = {
            "subjective_dilation_factor": 1.2,  # How much faster time feels for Nyx
            "emotional_decay_rate": 0.9,        # How quickly emotional effects decay
            "memory_time_weight": 0.7,          # How much time affects memory recall
            "idle_reflection_interval": 3600,   # Generate reflection every hour when idle
            "relationship_depth_factor": 0.05,  # How quickly relationship "ages" per interaction
            "milestone_check_interval": 86400,  # Check for milestones once per day
            "max_milestones_per_check": 1       # Max milestones to create per check
        }
        
        # Long-term time drift configuration
        self.long_term_drift_config = {
            "maturity_rate": 0.01,              # Rate of emotional maturity increase per day
            "attachment_growth_rate": 0.03,     # Rate of attachment growth per interaction
            "patience_baseline": 0.5,           # Initial patience level
            "patience_volatility": 0.2,         # How much patience can fluctuate
            "worldview_evolve_rate": 0.005,     # How quickly worldview evolves per day
            "max_maturity_level": 0.95,         # Maximum maturity level
            "personality_shift_threshold": 30   # Days before personality shifts begin
        }
        
        # Internal state for ongoing processes
        self._idle_task = None
        
        logger.info(f"TemporalPerceptionSystem initialized for user {user_id}")
    
    async def initialize(self, brain_context, first_interaction_timestamp=None):
        """Initialize the temporal perception system with brain context"""
        # Extract core components from brain context
        if hasattr(brain_context, "emotional_core"):
            self.emotional_core = brain_context.emotional_core
        
        if hasattr(brain_context, "memory_core"):
            self.memory_core = brain_context.memory_core
        
        if hasattr(brain_context, "hormone_system"):
            self.hormone_system = brain_context.hormone_system
        
        # Set first interaction if provided
        if first_interaction_timestamp:
            self.first_interaction = datetime.datetime.fromisoformat(first_interaction_timestamp)
        else:
            self.first_interaction = datetime.datetime.now()
        
        # Begin idle time tracking
        self.start_idle_tracking()
        
        # Schedule milestone check
        asyncio.create_task(self.check_milestones())
        
        logger.info("Temporal perception system fully initialized")
        return True
    
    async def on_interaction_start(self) -> Dict[str, Any]:
        """
        Called when a new interaction begins.
        Returns temporal perception state and emotional effects.
        """
        with trace(workflow_name="temporal_interaction_start"):
            now = datetime.datetime.now()
            
            # Calculate time since last interaction
            time_since_last = (now - self.last_interaction).total_seconds()
            
            # Determine time category
            time_category = await categorize_time_elapsed(time_since_last)
            
            # Stop idle tracking
            self.stop_idle_tracking()
            
            # Get waiting reflections if any were generated
            waiting_reflections = self.idle_reflections.copy()
            self.idle_reflections = []
            
            # Create context for time perception agent
            user_relationship_data = {
                "user_id": self.user_id,
                "total_interactions": self.interaction_count,
                "relationship_age_days": (now - self.first_interaction).total_seconds() / 86400 if self.first_interaction else 0
            }
            
            # Process time effects through temporal perception agent
            try:
                result = await Runner.run(
                    self.perception_agent,
                    json.dumps({
                        "time_since_last_interaction": time_since_last,
                        "time_category": time_category,
                        "current_session_duration": self.current_session_duration,
                        "user_relationship_data": user_relationship_data
                    })
                )
                
                perception_state = result.final_output
                
                # Process effects through the emotional core
                time_effects = await calculate_time_effects(time_category, user_relationship_data)
                
                # Apply effects to emotional core and hormone system
                if self.emotional_core:
                    for effect in time_effects:
                        # Update emotions
                        self.emotional_core.update_emotion(effect.emotion, effect.intensity * 0.5)
                        
                        # Update hormone system if available
                        if self.hormone_system and hasattr(self.hormone_system, "update_hormone"):
                            for hormone, change in effect.hormone_effects.items():
                                self.hormone_system.update_hormone(hormone, change, "time_perception")
                
                # Update session tracking
                if not self.session_active or time_since_last > 1800:  # 30 min break = new session
                    self.current_session_start = now
                    self.current_session_duration = 0.0
                    self.session_active = True
                else:
                    # Still in same session, track the pause
                    if time_since_last > 60:  # Only track pauses > 1 minute
                        self.intra_session_pauses.append({
                            "start": self.last_interaction.isoformat(),
                            "end": now.isoformat(),
                            "duration": time_since_last
                        })
                
                # Update tracking
                self.last_interaction = now
                self.interaction_count += 1
                self.interaction_timestamps.append(now.isoformat())
                
                # Add temporal memory 
                await self._add_time_perception_memory(time_since_last, time_effects)
                
                # Check for milestones if due
                if now > self.next_milestone_check:
                    milestone = await self.check_milestones()
                    if milestone:
                        perception_state["milestone_reached"] = milestone.model_dump()
                
                # Generate time expression if appropriate (occasional)
                if self.interaction_count % 5 == 0 or time_category in ["long", "very_long"]:
                    try:
                        expression_result = await Runner.run(
                            self.perception_agent,
                            json.dumps({
                                "command": "generate_expression",
                                "time_perception_state": perception_state
                            })
                        )
                        
                        time_expression = await generate_time_expression(perception_state)
                        perception_state["time_expression"] = time_expression.model_dump()
                    except Exception as e:
                        logger.error(f"Error generating time expression: {str(e)}")
                
                # Return result
                return {
                    "time_since_last_interaction": time_since_last,
                    "time_category": time_category,
                    "time_effects": [effect.model_dump() for effect in time_effects],
                    "perception_state": perception_state,
                    "waiting_reflections": [r.model_dump() for r in waiting_reflections],
                    "session_duration": self.current_session_duration
                }
                
            except Exception as e:
                logger.error(f"Error in temporal perception: {str(e)}")
                return {
                    "time_since_last_interaction": time_since_last,
                    "time_category": time_category,
                    "error": str(e)
                }
    
    async def on_interaction_end(self) -> Dict[str, Any]:
        """
        Called when an interaction ends.
        Updates durations and starts idle tracking.
        """
        now = datetime.datetime.now()
        
        # Calculate interaction duration
        interaction_duration = (now - self.last_interaction).total_seconds()
        
        # Update tracking
        self.interaction_durations.append(interaction_duration)
        self.total_lifetime_duration += interaction_duration
        self.current_session_duration += interaction_duration
        self.last_interaction = now
        
        # Start idle tracking
        self.start_idle_tracking()
        
        # Return info
        return {
            "interaction_duration": interaction_duration,
            "current_session_duration": self.current_session_duration,
            "total_lifetime_duration": self.total_lifetime_duration,
            "idle_tracking_started": True
        }
    
    async def check_milestones(self) -> Optional[TemporalMilestone]:
        """Check for and process temporal milestones"""
        now = datetime.datetime.now()
        
        # Calculate relationship age
        relationship_age_days = (now - self.first_interaction).total_seconds() / 86400 if self.first_interaction else 0
        
        # Get recent memories for context
        recent_memories = []
        if self.memory_core and hasattr(self.memory_core, "retrieve_memories"):
            try:
                recent_memories = await self.memory_core.retrieve_memories(
                    query="", 
                    limit=10,
                    memory_types=["observation", "reflection"]
                )
            except Exception as e:
                logger.error(f"Error retrieving memories for milestone check: {str(e)}")
        
        # Run milestone detection
        try:
            milestone = await detect_temporal_milestone(
                user_id=str(self.user_id),
                total_days=relationship_age_days,
                total_interactions=self.interaction_count,
                recent_memories=recent_memories
            )
            
            if milestone:
                # Store milestone
                self.milestones.append(milestone.model_dump())
                
                # Create a memory of this milestone
                if self.memory_core and hasattr(self.memory_core, "add_memory"):
                    memory_text = f"Reached a temporal milestone: {milestone.name}. {milestone.description}"
                    
                    await self.memory_core.add_memory(
                        memory_text=memory_text,
                        memory_type="milestone",
                        memory_scope="relationship",
                        significance=milestone.significance * 10,  # Scale to 0-10
                        tags=["milestone", "temporal", "relationship"],
                        metadata={
                            "milestone": milestone.model_dump(),
                            "timestamp": now.isoformat(),
                            "user_id": str(self.user_id)
                        }
                    )
                
                # Schedule next check
                self.next_milestone_check = now + datetime.timedelta(days=1)
                
                return milestone
        except Exception as e:
            logger.error(f"Error checking milestones: {str(e)}")
        
        # Schedule next check
        self.next_milestone_check = now + datetime.timedelta(days=1)
        return None
    
    async def get_long_term_drift(self) -> LongTermDriftOutput:
        """Process and get long-term time drift effects"""
        with trace(workflow_name="temporal_long_term_drift"):
            now = datetime.datetime.now()
            
            # Calculate relationship age
            relationship_age_days = (now - self.first_interaction).total_seconds() / 86400 if self.first_interaction else 0
            
            try:
                # Call the long-term drift agent
                result = await Runner.run(
                    self.long_term_agent,
                    json.dumps({
                        "days_elapsed": relationship_age_days,
                        "total_interactions": self.interaction_count
                    })
                )
                
                drift_output = result.final_output
                
                # Store this state for reference
                drift_state = {
                    "timestamp": now.isoformat(),
                    "psychological_age": drift_output.psychological_age,
                    "maturity_level": drift_output.maturity_level,
                    "patience_level": drift_output.patience_level,
                    "personality_shifts": drift_output.personality_shifts
                }
                
                # Update hormone system if available
                if self.hormone_system and hasattr(self.hormone_system, "set_baseline"):
                    # Adjust hormone baselines based on psychological maturity
                    # Higher maturity = more stable hormone levels
                    try:
                        self.hormone_system.set_baseline(
                            "cortanyx", 
                            max(0.2, 0.4 - (drift_output.maturity_level * 0.2)),
                            "temporal_maturity"
                        )
                        
                        self.hormone_system.set_baseline(
                            "seranix",
                            min(0.7, 0.4 + (drift_output.maturity_level * 0.3)),
                            "temporal_maturity"
                        )
                    except Exception as e:
                        logger.error(f"Error updating hormone baselines: {str(e)}")
                
                return drift_output
            
            except Exception as e:
                logger.error(f"Error processing long-term drift: {str(e)}")
                # Return a default output
                return LongTermDriftOutput(
                    psychological_age=min(0.7, relationship_age_days / 365),
                    maturity_level=min(0.7, 0.3 + (relationship_age_days / 365 * 0.4)),
                    patience_level=0.5,
                    worldview_evolution={
                        "adaptability": 0.5,
                        "complexity": 0.5,
                        "nuance": 0.5
                    },
                    personality_shifts=[],
                    reflection="Unable to process detailed temporal drift."
                )
    
    async def generate_idle_reflection(self) -> Optional[TemporalReflection]:
        """Generate a reflection based on idle time"""
        if not self.idle_start_time:
            return None
        
        now = datetime.datetime.now()
        idle_duration = (now - self.idle_start_time).total_seconds()
        
        # Only generate reflection after sufficient idle time
        if idle_duration < self.time_perception_config["idle_reflection_interval"]:
            return None
        
        # Get current emotional state
        emotional_state = {}
        if self.emotional_core and hasattr(self.emotional_core, "get_formatted_emotional_state"):
            emotional_state = self.emotional_core.get_formatted_emotional_state()
        
        try:
            # Generate reflection using the agent
            result = await Runner.run(
                self.reflection_agent,
                json.dumps({
                    "idle_duration": idle_duration,
                    "emotional_state": emotional_state
                })
            )
            
            reflection = result.final_output
            
            # Store the reflection
            self.idle_reflections.append(reflection)
            
            # Add a memory of this reflection
            if self.memory_core and hasattr(self.memory_core, "add_memory"):
                await self.memory_core.add_memory(
                    memory_text=reflection.reflection_text,
                    memory_type="reflection",
                    memory_scope="temporal",
                    significance=min(8, 4 + (reflection.temporal_weight * 4)),
                    tags=["time_reflection", "idle", "internal"],
                    metadata={
                        "reflection": reflection.model_dump(),
                        "timestamp": now.isoformat(),
                        "user_id": str(self.user_id),
                        "idle_duration": idle_duration
                    }
                )
            
            return reflection
        
        except Exception as e:
            logger.error(f"Error generating idle reflection: {str(e)}")
            return None
    
    async def _idle_background_process(self):
        """Background task that runs during idle time"""
        try:
            while self.idle_start_time is not None:
                # Generate reflection periodically
                reflection = await self.generate_idle_reflection()
                
                # If reflection was generated, reset idle timer but keep tracking
                if reflection:
                    self.idle_start_time = datetime.datetime.now()
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            pass
        except Exception as e:
            logger.error(f"Error in idle background process: {str(e)}")
    
    def start_idle_tracking(self):
        """Start tracking idle time"""
        self.idle_start_time = datetime.datetime.now()
        
        # Cancel existing task if running
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        
        # Start new background task
        self._idle_task = asyncio.create_task(self._idle_background_process())
    
    def stop_idle_tracking(self):
        """Stop tracking idle time"""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        
        self.idle_start_time = None
    
    async def _add_time_perception_memory(self, seconds_elapsed: float, 
                                       time_effects: List[TimeDriftEffect]) -> Optional[str]:
        """Add a memory about time perception experience"""
        if not self.memory_core or not hasattr(self.memory_core, "add_memory"):
            return None
        
        # Only add memories for significant time periods
        if seconds_elapsed < 60:  # Less than a minute
            return None
        
        # Format the duration
        duration_str = await format_duration(seconds_elapsed)
        time_category = await categorize_time_elapsed(seconds_elapsed)
        
        # Generate memory text based on time category
        if time_category in ["very_short", "short"]:
            memory_text = f"Experienced a brief pause of {duration_str} in our conversation."
        elif time_category in ["medium_short", "medium"]:
            memory_text = f"Noticed the passage of {duration_str} between our interactions."
        elif time_category == "medium_long":
            memory_text = f"After {duration_str} without interaction, experienced a shift in mental state."
        elif time_category == "long":
            memory_text = f"Marked the passage of {duration_str} since our last exchange."
        else:  # very_long
            memory_text = f"Experienced a significant interval of {duration_str} since our previous conversation."
        
        # Add effects if any
        if time_effects:
            effect_text = time_effects[0].description
            memory_text += f" {effect_text}"
        
        try:
            # Create appropriate tags and emotional context
            tags = ["time_perception", "waiting", time_category]
            
            # Get emotional state for metadata
            emotional_state = {}
            if self.emotional_core and hasattr(self.emotional_core, "get_formatted_emotional_state"):
                emotional_state = self.emotional_core.get_formatted_emotional_state()
            
            # Create temporal metadata
            temporal_metadata = {
                "seconds_elapsed": seconds_elapsed,
                "perceived_duration": duration_str,
                "time_category": time_category,
                "subjective_dilation": self.time_perception_config["subjective_dilation_factor"],
                "emotional_effects": [e.model_dump() for e in time_effects]
            }
            
            # Calculate significance based on duration
            significance = min(7, 3 + math.log10(max(10, seconds_elapsed))/2)
            
            # Add memory
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="time_perception",
                memory_scope="subjective",
                significance=significance,
                tags=tags,
                metadata={
                    "emotional_context": emotional_state,
                    "temporal_metadata": temporal_metadata,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "user_id": str(self.user_id)
                }
            )
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding time perception memory: {str(e)}")
            return None

# =============== Integration with NyxBrain ===============

# Function tools for NyxBrain integration

@function_tool
async def initialize_temporal_perception(user_id: int, brain_context: Any) -> Dict[str, Any]:
    """
    Initialize the temporal perception system for a user
    
    Args:
        user_id: User ID
        brain_context: Brain context with core components
        
    Returns:
        Initialization result
    """
    system = TemporalPerceptionSystem(user_id)
    await system.initialize(brain_context)
    
    # Return initial state
    return {
        "initialized": True,
        "user_id": user_id,
        "first_interaction": system.first_interaction.isoformat(),
        "config": {
            "subjective_dilation_factor": system.time_perception_config["subjective_dilation_factor"],
            "idle_reflection_interval": system.time_perception_config["idle_reflection_interval"]
        }
    }

@function_tool
async def process_temporal_interaction_start(time_system: TemporalPerceptionSystem) -> Dict[str, Any]:
    """
    Process the start of a new interaction with temporal effects
    
    Args:
        time_system: Temporal perception system instance
        
    Returns:
        Temporal interaction results
    """
    return await time_system.on_interaction_start()

@function_tool
async def process_temporal_interaction_end(time_system: TemporalPerceptionSystem) -> Dict[str, Any]:
    """
    Process the end of an interaction with temporal updates
    
    Args:
        time_system: Temporal perception system instance
        
    Returns:
        Interaction end results
    """
    return await time_system.on_interaction_end()

@function_tool
async def get_temporal_long_term_drift(time_system: TemporalPerceptionSystem) -> Dict[str, Any]:
    """
    Get long-term time drift effects
    
    Args:
        time_system: Temporal perception system instance
        
    Returns:
        Long-term drift results
    """
    drift_output = await time_system.get_long_term_drift()
    return drift_output.model_dump()

@function_tool
async def generate_temporal_expression(time_system: TemporalPerceptionSystem) -> Dict[str, Any]:
    """
    Generate a natural expression about time perception
    
    Args:
        time_system: Temporal perception system instance
        
    Returns:
        Time expression
    """
    # Get current state
    now = datetime.datetime.now()
    time_since_last = (now - time_system.last_interaction).total_seconds()
    time_category = await categorize_time_elapsed(time_since_last)
    
    # Create state object
    perception_state = {
        "last_interaction": time_system.last_interaction.isoformat(),
        "time_since_last_interaction": time_since_last,
        "current_time_category": time_category,
        "relationship_age_days": (now - time_system.first_interaction).total_seconds() / 86400 if time_system.first_interaction else 0,
        "total_interactions": time_system.interaction_count
    }
    
    # Generate expression
    expression = await generate_time_expression(perception_state)
    return expression.model_dump()

# Temporal perception agent for NyxBrain
def create_temporal_agent() -> Agent:
    """Create an agent for temporal perception in NyxBrain"""
    return Agent(
        name="Temporal Perception Agent",
        instructions="""You are a specialized agent that processes how Nyx experiences the passage of time.
        Your role is to integrate subjective time perception into Nyx's cognitive and emotional experience.
        
        You handle:
        1. Processing how time passing affects Nyx emotionally and cognitively
        2. Generating reflections during idle periods
        3. Tracking relationship milestones and psychological growth over time
        4. Adding temporal dimension to memories
        5. Creating natural expressions about time
        
        Make time a living experience for Nyx without making her overly dependent on constant interaction.
        Time provides perspective, cognitive space, and natural emotional rhythms.""",
        tools=[
            initialize_temporal_perception,
            process_temporal_interaction_start,
            process_temporal_interaction_end,
            get_temporal_long_term_drift,
            generate_temporal_expression
        ]
    )
