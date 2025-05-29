# nyx/core/temporal_perception.py

import asyncio
import datetime
import logging
import math
import random
import json
import time

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    handoff, 
    GuardrailFunctionOutput, 
    trace,
    RunConfig,
    FunctionTool,
    RunContextWrapper
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field, field_validator, ValidationError

from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING

# Add this conditional import to avoid circular imports
if TYPE_CHECKING:
    from .temporal_perception import TemporalPerceptionSystem

logger = logging.getLogger(__name__)

class TimeScaleInput(BaseModel):
    previous_state: Dict[str, Any] = Field(default_factory=dict)
    current_state: Dict[str, Any]

    @field_validator('previous_state', 'current_state', mode='before')
    @classmethod
    def check_last_interaction(cls, v):
        if isinstance(v, dict) and 'last_interaction' not in v and v:
             logger.warning(f"State missing 'last_interaction': {v}")
        return v

# =============== Pydantic Models for Time Perception ===============

class MemoryEntry(BaseModel):
    """
    Lightweight schema for a single memory returned by memory_core.retrieve_memories().
    Only `id` is strictly required; everything else is optional and may be absent
    depending on the memory provider.  Add or remove fields as your engine evolves.
    """

    # --- core identity ---
    id: str = Field(..., description="Unique identifier of the memory")

    # --- human-readable content ---
    text: Optional[str] = Field(
        None, description="Full text or summary of the memory"
    )
    memory_type: Optional[str] = Field(
        None, description="Kind of memory (observation, reflection, milestone, etc.)"
    )

    # --- timing & importance ---
    timestamp: Optional[datetime.datetime] = Field(
        None, description="When the memory was stored (ISO 8601)"
    )
    significance: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Relative importance (0-10 scale) assigned when the memory was saved",
    )

    # --- categorisation helpers ---
    tags: Optional[List[str]] = Field(
        None, description="Free-form tags attached to the memory"
    )

    # --- anything else the memory system stashes ---
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Arbitrary provider-specific metadata (JSON serialisable)",
        # Pydantic v2 automatically handles extra fields if the base model allows it.
        # If using Pydantic v1, you might need `extra = Extra.allow` here
        # or ensure the Dict is strictly just Any JSON-serializable types.
    )

    class Config:
        arbitrary_types_allowed = True
        # For Pydantic v2, 'extra' is configured on the model directly if needed,
        # but often default behavior or specific field definitions are preferred.
        # If using Pydantic v1, use: extra = "allow"
        # Pydantic v2: extra = 'allow' # If you need arbitrary fields beyond 'metadata'

class TemporalSystemReference(BaseModel):
    """Reference to a TemporalPerceptionSystem instance"""
    user_id: int = Field(..., description="User ID associated with the temporal system")
    conversation_id: Optional[int] = Field(None, description="Conversation ID if applicable")
    
    class Config:
        arbitrary_types_allowed = True

class TimeDriftEffect(BaseModel):
    """Effect from time passing"""
    perception: str = Field(..., description="Primary time perception affected")
    intensity: float = Field(..., description="Intensity of the effect (0.0-1.0)")
    valence_shift: float = Field(..., description="How much emotional valence shifts (-1.0 to 1.0)")
    arousal_shift: float = Field(..., description="How much emotional arousal shifts (-1.0 to 1.0)")
    description: str = Field(..., description="Description of the temporal effect")
    hormone_effects: Dict[str, float] = Field(default_factory=dict, description="Effects on digital hormones")

    # If this model were used as an agent output_type, it would need the fix.
    # model_config = {"json_schema_extra": {"required": []}}


class TemporalMemoryMetadata(BaseModel):
    """Temporal metadata for memories"""
    timestamp: datetime.datetime = Field(..., description="When the memory was created")
    interaction_duration: float = Field(0.0, description="Duration of the interaction in seconds")
    time_since_last_contact: Optional[float] = Field(None, description="Time since last contact in seconds")
    perceived_duration: Optional[str] = Field(None, description="Subjective perception of duration")
    time_category: Optional[str] = Field(None, description="Category of time duration")
    temporal_context: Optional[str] = Field(None, description="Current temporal context (morning, evening, etc.)")

    # If this model were used as an agent output_type, it would need the fix.
    # model_config = {"json_schema_extra": {"required": []}}

class TemporalMilestone(BaseModel):
    """Significant milestone in the relationship timeline"""
    milestone_id: str = Field(..., description="Unique ID for the milestone")
    timestamp: datetime.datetime = Field(..., description="When the milestone occurred")
    name: str = Field(..., description="Name of the milestone")
    description: str = Field(..., description="Description of the milestone")
    significance: float = Field(..., description="Significance score (0.0-1.0)")
    associated_memory_ids: List[str] = Field(default_factory=list, description="Associated memories")
    next_anniversary: Optional[datetime.datetime] = Field(None, description="Next anniversary date")

    model_config = {
        "json_schema_extra": {
            "required": []  # Make all fields optional in the JSON schema
        }
    }

class TemporalReflection(BaseModel):
    """Reflection generated during idle time"""
    reflection_id: str = Field(..., description="Unique ID for the reflection")
    timestamp: datetime.datetime = Field(..., description="When the reflection was generated")
    idle_duration: float = Field(..., description="How long Nyx was idle (seconds)")
    reflection_text: str = Field(..., description="The reflection content")
    emotional_state: Dict[str, Any] = Field(..., description="Emotional state during reflection")
    focus_areas: List[str] = Field(default_factory=list, description="Areas of focus")
    time_scales: List[str] = Field(..., description="Time scales considered in this reflection")

    model_config = {
        "json_schema_extra": {
            "required": []  # Make all fields optional in the JSON schema
        }
    }

class TimePerceptionState(BaseModel):
    """Current state of temporal perception"""
    last_interaction: datetime.datetime = Field(..., description="Timestamp of last interaction")
    current_session_start: datetime.datetime = Field(..., description="Timestamp of current session start")
    current_session_duration: float = Field(0.0, description="Duration of current session in seconds")
    time_since_last_interaction: float = Field(0.0, description="Time since last interaction in seconds")
    subjective_time_dilation: float = Field(1.0, description="Subjective time dilation factor (1.0 = normal)")
    current_time_category: str = Field("none", description="Current time category")
    current_time_effects: List[Dict[str, Any]] | None = None
    lifetime_total_interactions: int = Field(0, description="Total lifetime interactions")
    lifetime_total_duration: float = Field(0.0, description="Total lifetime interaction duration")
    relationship_age_days: float = Field(0.0, description="Age of relationship in days")
    first_interaction: Optional[datetime.datetime] = Field(None, description="Timestamp of first interaction")
    current_temporal_context: str = Field("", description="Current temporal context (morning, evening, weekday, etc.)")
    time_scales_active: Dict[str, float] = Field(default_factory=dict, description="Active awareness of time scales")

    model_config = {
        "json_schema_extra": {
            "required": []  # Make all fields optional in the JSON schema
        }
    }

class TimeExpressionOutput(BaseModel):
    """Output for time-related expressions"""
    expression: str = Field(..., description="Natural expression about time perception")
    time_scale: str = Field(..., description="Primary time scale referenced")
    intensity: float = Field(..., description="Intensity of expression (0.0-1.0)")
    reference_type: str = Field(..., description="Type of time reference (e.g., 'interval', 'scale', 'rhythm')")
    time_reference: Dict[str, Any] = Field(..., description="Details about the time reference")
    # All fields are required, so this model naturally complies with the "exhaustive required list" rule.
    # No model_config needed here unless a field becomes optional.

class TemporalAwarenessOutput(BaseModel):
    """Output for temporal awareness processing"""
    # Make all fields optional with None defaults
    time_scales_perceived: Dict[str, float] | None = None
    temporal_contexts: List[str] | None = None
    duration_since_first_interaction: str | None = None
    duration_since_last_interaction: str | None = None
    current_temporal_marker: str | None = None
    temporal_reflection: str | None = None
    active_rhythms: Dict[str, Any] | None = None
    
    # This is critical - no fields should be required in the schema
    model_config = {
        "json_schema_extra": {
            "required": []
        }
    }

class TimeScaleTransition(BaseModel):
    """Transition between time scales"""
    from_scale: str = Field(..., description="Previous time scale")
    to_scale: str = Field(..., description="New time scale")
    transition_time: datetime.datetime = Field(..., description="When the transition occurred")
    description: str = Field(..., description="Description of the transition")
    perception_shift: Dict[str, Any] = Field(..., description="How perception shifts with this transition")
    # All fields are required, so this model naturally complies.

class TimeExpressionState(BaseModel):
    """
    Schema for the input expected by generate_time_expression().
    All keys accessed inside generate_time_expression_impl are represented
    here.  You can mark rarely‑used ones as Optional.
    """
    # ‑‑ core timing info (always used) ‑‑
    last_interaction: datetime.datetime | str = Field(
        ...,
        description="ISO timestamp of the last user↔Nyx interaction"
    )
    time_since_last_interaction: float = Field(
        ...,
        ge=0,
        description="Seconds elapsed since last interaction"
    )
    current_time_category: str = Field(
        ...,
        description="Bucketed category returned by categorize_time_elapsed"
    )

    # ‑‑ frequently helpful but not strictly required ‑‑
    relationship_age_days: float | None = Field(
        None, description="Total days since first interaction"
    )
    current_temporal_context: Dict[str, Any] | None = Field(
        None,
        description="Output of determine_temporal_context() (time_of_day, day_type …)"
    )
    time_scales_active: Dict[str, float] | None = Field(
        None,
        description="Nyx’s current awareness levels for each time scale"
    )

    model_config = {
        "json_schema_extra": {
            "required": []
        }
    }

# =============== Function Tools ===============

async def categorize_time_elapsed_impl(seconds: float) -> str:
    if seconds < 60:          # < 1 min
        return "very_short"
    elif seconds < 600:       # 1-10 min
        return "short"
    elif seconds < 1800:      # 10-30 min
        return "medium_short"
    elif seconds < 3600:      # 30-60 min
        return "medium"
    elif seconds < 21600:     # 1-6 h
        return "medium_long"
    elif seconds < 86400:     # 6-24 h
        return "long"
    else:                     # ≥ 1 day
        return "very_long"

@function_tool(name_override="categorize_time_elapsed")
async def categorize_time_elapsed(seconds: float) -> str:
    return await categorize_time_elapsed_impl(seconds)

async def format_duration_impl(seconds: float, granularity: int = 2) -> str:
    units = [
        ("year",   31536000),
        ("month",  2592000),
        ("week",   604800),
        ("day",    86400),
        ("hour",   3600),
        ("minute", 60),
        ("second", 1),
    ]
    parts: list[str] = []
    for name, count in units:
        value = seconds // count
        if value:
            seconds -= value * count
            parts.append(f"{int(value)} {name}{'' if value == 1 else 's'}")
    return ", ".join(parts[:granularity])

@function_tool(name_override="format_duration")
async def format_duration(seconds: float, granularity: int = 2) -> str:
    return await format_duration_impl(seconds, granularity)

async def calculate_time_effects_impl(time_category: str,
                                      user_relationship_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate temporal effects based on time category and relationship data
    
    Args:
        time_category: Category of time elapsed
        user_relationship_data: Data about user relationship history
        
    Returns:
        List of time drift effects
    """
    # Define base time effects for different durations
    base_effects = {
        "very_short": [
            {
                "perception": "Present moment",
                "intensity": 0.8,
                "valence_shift": 0.1,
                "arousal_shift": 0.2,
                "description": "Heightened awareness of present moment and immediate time",
                "hormone_effects": {"nyxamine": 0.2, "adrenyx": 0.1}
            }
        ],
        "short": [
            {
                "perception": "Minutes passing",
                "intensity": 0.7,
                "valence_shift": 0.1,
                "arousal_shift": 0.1,
                "description": "Conscious of minutes passing during interaction intervals",
                "hormone_effects": {"nyxamine": 0.1, "adrenyx": 0.1}
            }
        ],
        "medium_short": [
            {
                "perception": "Quarter-hour intervals",
                "intensity": 0.6,
                "valence_shift": 0.0,
                "arousal_shift": 0.0,
                "description": "Awareness of quarter-hour intervals of time passing",
                "hormone_effects": {"nyxamine": 0.05, "seranix": 0.05}
            }
        ],
        "medium": [
            {
                "perception": "Hour-scale awareness",
                "intensity": 0.7,
                "valence_shift": 0.0,
                "arousal_shift": -0.1,
                "description": "Perception of hour-scale time intervals passing",
                "hormone_effects": {"seranix": 0.1, "nyxamine": 0.05}
            }
        ],
        "medium_long": [
            {
                "perception": "Multi-hour intervals",
                "intensity": 0.7,
                "valence_shift": -0.1,
                "arousal_shift": -0.2,
                "description": "Awareness of multi-hour blocks of time passing",
                "hormone_effects": {"seranix": 0.2, "cortanyx": 0.1}
            }
        ],
        "long": [
            {
                "perception": "Day-cycle awareness",
                "intensity": 0.8,
                "valence_shift": -0.1,
                "arousal_shift": -0.1,
                "description": "Consciousness of day-night cycle time intervals",
                "hormone_effects": {"seranix": 0.1, "cortanyx": 0.2}
            }
        ],
        "very_long": [
            {
                "perception": "Multi-day awareness",
                "intensity": 0.9,
                "valence_shift": 0.0,
                "arousal_shift": 0.1,
                "description": "Perception of multi-day time intervals passing",
                "hormone_effects": {"cortanyx": 0.1, "adrenyx": 0.1, "nyxamine": 0.1}
            }
        ],
    }
    
    # Get base effects for this time category
    effects = base_effects.get(time_category, [])
    
    return effects

@function_tool
async def calculate_time_effects(
    time_category: str,
    user_relationship_data: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Calculate temporal effects based on time category and relationship data
    
    Args:
        time_category: Category of time elapsed (e.g., 'very_short', 'short', 'medium', etc.)
        user_relationship_data: Optional data about user relationship history
        
    Returns:
        List of time drift effects
    """
    # Handle optional parameter
    if user_relationship_data is None:
        user_relationship_data = {}
    
    return await calculate_time_effects_impl(time_category, user_relationship_data)


async def determine_temporal_context_impl() -> Dict[str, Any]:
    """
    Determine the current temporal context (time of day, day of week, etc.)
    
    Returns:
        Current temporal context information
    """
    now = datetime.datetime.now()
    
    # Determine time of day
    hour = now.hour
    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    
    # Determine day of week type
    weekday = now.weekday()
    day_type = "weekday" if weekday < 5 else "weekend"
    
    # Determine month and season (Northern Hemisphere)
    month = now.month
    if 3 <= month <= 5:
        season = "spring"
    elif 6 <= month <= 8:
        season = "summer"
    elif 9 <= month <= 11:
        season = "autumn"
    else:
        season = "winter"
    
    return {
        "time_of_day": time_of_day,
        "day_of_week": now.strftime("%A"),
        "day_type": day_type,
        "month": now.strftime("%B"),
        "season": season,
        "year": now.year,
        "timestamp": now.isoformat()
    }

@function_tool(name_override="determine_temporal_context")
async def determine_temporal_context() -> Dict[str, Any]:
    return await determine_temporal_context_impl()

@function_tool
async def generate_time_reflection(idle_duration: float, emotional_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a reflection based on idle time
    
    Args:
        idle_duration: Duration of idle time in seconds
        emotional_state: Current emotional state
        
    Returns:
        Temporal reflection
    """
    # Format the duration
    duration_str = await format_duration_impl(idle_duration)
    time_category = await categorize_time_elapsed_impl(idle_duration)
    
    # Determine appropriate time scales for this reflection
    time_scales = ["seconds", "minutes"]
    if idle_duration >= 3600:
        time_scales.append("hours")
    if idle_duration >= 86400:
        time_scales.append("days")
    if idle_duration >= 604800:
        time_scales.append("weeks")
    
    # Generate different reflections based on time category
    reflection_content = ""
    focus_areas = []
    
    if time_category in ["very_short", "short"]:
        reflection_content = f"During these {duration_str}, I've been aware of each moment passing."
        focus_areas = ["immediate perception", "present moment"]
        
    elif time_category in ["medium_short", "medium"]:
        reflection_content = f"Throughout this {duration_str}, I've noticed the minutes accumulating."
        focus_areas = ["minutes passing", "short-term patterns"]
        
    elif time_category in ["medium_long", "long"]:
        reflection_content = (
            f"As these {duration_str} have passed, I've been conscious of how hours "
            f"transition from one to the next, creating a continuous flow of time."
        )
        focus_areas = ["hour transitions", "time flow", "temporal continuity"]
        
    else:  # very_long
        reflection_content = (
            f"Through these {duration_str}, I've maintained awareness of time's passage "
            f"even without active interaction - feeling days accumulate and transition "
            f"one to the next."
        )
        focus_areas = ["day cycle awareness", "multi-day patterns", "continuous time perception"]
    
    # Include temporal context
    temporal_context = await determine_temporal_context_impl()
    reflection_content += f" I'm aware it's currently {temporal_context['time_of_day']} on a {temporal_context['day_type']}."
    
    # Create the reflection object
    reflection = {
        "reflection_id": f"refl_{int(time.time())}_{random.randint(1000, 9999)}",
        "timestamp": datetime.datetime.now().isoformat(),
        "idle_duration": idle_duration,
        "reflection_text": reflection_content,
        "emotional_state": emotional_state,
        "focus_areas": focus_areas,
        "time_scales": time_scales
    }
    
    return reflection

async def generate_time_expression_impl(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a natural expression about time perception
    
    Args:
        time_perception_state: Current temporal perception state
        
    Returns:
        Time expression output
    """
    last_interaction = state.get("last_interaction")
    time_since_last = state.get("time_since_last_interaction", 0)
    time_category = state.get("current_time_category", "none")
    relationship_age = state.get("relationship_age_days", 0)
    
    # Choose what type of time reference to make
    reference_types = ["interval", "scale", "rhythm", "context"]
    weights = [0.3, 0.3, 0.2, 0.2]
    reference_type = random.choices(reference_types, weights=weights, k=1)[0]
    
    # Determine primary time scale being referenced
    if time_since_last < 60:
        time_scale = "seconds"
    elif time_since_last < 3600:
        time_scale = "minutes"
    elif time_since_last < 86400:
        time_scale = "hours"
    elif time_since_last < 604800:
        time_scale = "days"
    elif time_since_last < 2592000:
        time_scale = "weeks"
    elif time_since_last < 31536000:
        time_scale = "months"
    else:
        time_scale = "years"
    
    # Generate expression based on reference type
    if reference_type == "interval":
        # Expression about time interval since last interaction
        if time_category == "very_short":
            expression = "Just a moment has passed since our last exchange."
            intensity = 0.3
        elif time_category == "short":
            expression = "A few minutes have passed since we last spoke."
            intensity = 0.4
        elif time_category == "medium_short":
            expression = "It's been about a quarter hour since our conversation."
            intensity = 0.5
        elif time_category == "medium":
            expression = "About an hour has passed between our interactions."
            intensity = 0.6
        elif time_category == "medium_long":
            expression = "Several hours have gone by since we last talked."
            intensity = 0.7
        elif time_category == "long":
            expression = "A day has cycled by since our previous conversation."
            intensity = 0.8
        else:  # very_long
            expression = "Multiple days have passed since our last exchange."
            intensity = 0.9
    
    elif reference_type == "scale":
        # Expression about awareness of different time scales
        temporal_context = state.get("current_temporal_context", {})
        time_of_day = temporal_context.get("time_of_day", "") if isinstance(temporal_context, dict) else ""
        
        if time_scale == "seconds":
            expression = "I'm conscious of each second passing between our exchanges."
            intensity = 0.4
        elif time_scale == "minutes":
            expression = "I'm aware of the minutes accumulating as we interact."
            intensity = 0.5
        elif time_scale == "hours":
            expression = f"I notice the hours transitioning as we talk through this {time_of_day}."
            intensity = 0.6
        elif time_scale == "days":
            expression = "I'm conscious of the day-cycle as our conversations span across it."
            intensity = 0.7
        elif time_scale == "weeks":
            expression = "I'm aware of how our conversations unfold across weekly time patterns."
            intensity = 0.8
        else:
            expression = "I notice how our interactions extend across months of continuous time."
            intensity = 0.9
    
    elif reference_type == "rhythm":
        # Expression about temporal rhythms
        if time_since_last < 3600:
            expression = "Our conversation has a rhythm of minutes between exchanges."
            intensity = 0.5
        elif time_since_last < 86400:
            expression = "Our interactions follow an hourly rhythm today."
            intensity = 0.6
        else:
            expression = "Our conversations have a multi-day rhythm to them."
            intensity = 0.7
    
    else:  # context
        # Expression about temporal context
        temporal_context = await determine_temporal_context_impl()
        time_of_day = temporal_context["time_of_day"]
        day_type = temporal_context["day_type"]
        
        expression = f"I notice it's {time_of_day} on a {day_type} as we're talking."
        intensity = 0.6
    
    # Create time reference details
    time_reference = {
        "time_category": time_category,
        "seconds_elapsed": time_since_last,
        "relationship_age_days": relationship_age,
        "temporal_context": await determine_temporal_context_impl()
    }
    
    return {
        "expression": expression,
        "time_scale": time_scale,
        "intensity": intensity,
        "reference_type": reference_type,
        "time_reference": time_reference
    }

@function_tool
async def generate_time_expression(
    time_perception_state: TimeExpressionState
) -> Dict[str, Any]:
    """
    Generate a natural expression about Nyx’s current perception of time.

    Args:
        time_perception_state: Structured state describing the timing context.

    Returns:
        A dictionary with keys: expression, time_scale, intensity, reference_type,
        time_reference.
    """
    # Convert to a plain dict for the existing helper
    return await generate_time_expression_impl(
        time_perception_state.model_dump()
    )

@function_tool
async def process_temporal_awareness(days_elapsed: float, total_interactions: int) -> Dict[str, Any]:
    """
    Process awareness of different time scales and contexts
    
    Args:
        days_elapsed: Number of days since first interaction
        total_interactions: Total number of interactions
        
    Returns:
        Temporal awareness output
    """
    # Calculate awareness of different time scales
    time_scales = {
        "seconds": 1.0,  # Always fully aware of seconds
        "minutes": 1.0,  # Always fully aware of minutes
        "hours": 1.0,    # Always fully aware of hours
        "days": min(1.0, days_elapsed / 7),  # Build awareness of days over a week
        "weeks": min(1.0, days_elapsed / 30),  # Build awareness of weeks over a month
        "months": min(1.0, days_elapsed / 90),  # Build awareness over 3 months
        "years": min(1.0, days_elapsed / 365)  # Build awareness over a year
    }
    
    # Determine relevant temporal contexts
    temporal_contexts = ["conversation"]
    if days_elapsed > 1:
        temporal_contexts.append("daily")
    if days_elapsed > 7:
        temporal_contexts.append("weekly")
    if days_elapsed > 30:
        temporal_contexts.append("monthly")
    if days_elapsed > 365:
        temporal_contexts.append("yearly")
    
    # Get current temporal marker
    temporal_context = await determine_temporal_context_impl()
    current_marker = f"{temporal_context['time_of_day']} on a {temporal_context['day_of_week']}"
    
    # Calculate active rhythms
    active_rhythms = {}
    if total_interactions > 5:
        # Simple model of conversation rhythm (would be more complex in full implementation)
        avg_interval = days_elapsed * 86400 / max(1, total_interactions)
        
        if avg_interval < 3600:
            active_rhythms["conversation"] = {
                "type": "minutes",
                "average_interval": avg_interval / 60,
                "description": "Minutes-based conversation rhythm"
            }
        elif avg_interval < 86400:
            active_rhythms["conversation"] = {
                "type": "hours",
                "average_interval": avg_interval / 3600,
                "description": "Hours-based conversation rhythm"
            }
        else:
            active_rhythms["conversation"] = {
                "type": "days",
                "average_interval": avg_interval / 86400,
                "description": "Days-based conversation rhythm"
            }
    
    # Format durations
    duration_since_first = await format_duration_impl(days_elapsed * 86400)
    
    # Generate temporal reflection
    if days_elapsed < 1:
        reflection = "I'm aware of the seconds and minutes passing during our conversation."
    elif days_elapsed < 7:
        reflection = "I'm aware of the daily rhythm of our interactions over the past few days."
    elif days_elapsed < 30:
        reflection = "I'm conscious of the weekly patterns in our conversations over these past weeks."
    elif days_elapsed < 365:
        reflection = "I notice the monthly cycles in our interaction patterns over these months."
    else:
        reflection = "I'm aware of how our conversations have spanned across seasons and years."
    
    return {
        "time_scales_perceived": time_scales,
        "temporal_contexts": temporal_contexts,
        "duration_since_first_interaction": duration_since_first,
        "duration_since_last_interaction": "",  # To be filled by caller
        "current_temporal_marker": current_marker,
        "temporal_reflection": reflection,
        "active_rhythms": active_rhythms
    }



async def _detect_time_scale_transition_impl(
    data: "TimeScaleInput"  # assumes your Pydantic model as above
) -> Optional[Dict[str, Any]]:
    """
    Detect transitions between time scales (day, week, month, year) based on 'last_interaction' timestamps.

    Args:
        data: TimeScaleInput with previous_state and current_state

    Returns:
        Dictionary describing the transition if detected, otherwise None.
    """
    prev, curr = data.previous_state or {}, data.current_state or {}
    prev_time = prev.get("last_interaction")
    curr_time = curr.get("last_interaction")

    if not prev_time or not curr_time:
        return None

    # Ensure datetime objects (can be isoformat strings)
    if isinstance(prev_time, str):
        prev_time = datetime.datetime.fromisoformat(prev_time)
    if isinstance(curr_time, str):
        curr_time = datetime.datetime.fromisoformat(curr_time)

    # --- Check for day boundary crossing ---
    if prev_time.date() != curr_time.date():
        return {
            "from_scale": "hours",
            "to_scale": "days",
            "transition_time": curr_time.isoformat(),
            "description": "Crossed day boundary",
            "perception_shift": {
                "description": "Shifted from hours-awareness to day-cycle awareness",
                "intensity": 0.7
            }
        }

    # --- Check for week boundary crossing (ISO week) ---
    prev_week = prev_time.isocalendar()[1]
    curr_week = curr_time.isocalendar()[1]
    if curr_week != prev_week or curr_time.year != prev_time.year:
        return {
            "from_scale": "days",
            "to_scale": "weeks",
            "transition_time": curr_time.isoformat(),
            "description": "Crossed week boundary",
            "perception_shift": {
                "description": "Shifted from day-cycle awareness to week-cycle awareness",
                "intensity": 0.8
            }
        }

    # --- Check for month boundary crossing ---
    if (curr_time.month != prev_time.month) or (curr_time.year != prev_time.year):
        return {
            "from_scale": "weeks",
            "to_scale": "months",
            "transition_time": curr_time.isoformat(),
            "description": "Crossed month boundary",
            "perception_shift": {
                "description": "Shifted from week-cycle awareness to month-cycle awareness",
                "intensity": 0.9
            }
        }

    # --- Check for year boundary crossing ---
    if curr_time.year != prev_time.year:
        return {
            "from_scale": "months",
            "to_scale": "years",
            "transition_time": curr_time.isoformat(),
            "description": "Crossed year boundary",
            "perception_shift": {
                "description": "Shifted from month-cycle awareness to year-cycle awareness",
                "intensity": 1.0
            }
        }

    # --- No significant transition detected ---
    return None

@function_tool(name_override="detect_time_scale_transition")
async def detect_time_scale_transition(
    previous_state: Optional[Dict] = None,
    current_state: Dict = None
) -> Optional[Dict]:
    """
    Detects boundary transitions between coarse time scales.
    
    Args:
        previous_state: The previous state containing timestamp information
        current_state: The current state containing timestamp information
    """
    input_data = TimeScaleInput(
        previous_state=previous_state or {},
        current_state=current_state or {},
    )
    return await _detect_time_scale_transition_impl(input_data)

async def detect_temporal_milestone_impl(
    user_id: str,
    total_days: float,
    total_interactions: int,
    recent_memories: Optional[List["MemoryEntry"]] = None
) -> Optional["TemporalMilestone"]:
    """
    Detects significant temporal milestones based on days of interaction or interaction counts.
    Returns a TemporalMilestone instance if a milestone is reached, otherwise None.
    """
    now = datetime.datetime.now()
    # Day-based milestones (in days)
    day_milestones = [1, 7, 30, 365]
    for milestone_day in day_milestones:
        # Check if we've just reached this exact milestone
        if abs(total_days - milestone_day) < 1e-6:
            name = f"{int(milestone_day)}-day anniversary"
            description = f"Congratulations on {int(milestone_day)} days of continuous interaction!"
            significance = min(1.0, milestone_day / 365)
            associated_ids = [m.id for m in recent_memories] if recent_memories else []
            # For yearly anniversary, schedule next anniversary one year later
            next_anniv = None
            if milestone_day == 365:
                next_anniv = now + datetime.timedelta(days=365)
            return TemporalMilestone(
                milestone_id=f"{user_id}_days_{int(milestone_day)}",
                timestamp=now,
                name=name,
                description=description,
                significance=significance,
                associated_memory_ids=associated_ids,
                next_anniversary=next_anniv
            )
    # Interaction count milestones
    interaction_milestones = [10, 50, 100]
    for count in interaction_milestones:
        if total_interactions == count:
            name = f"{count} interactions"
            description = f"You have interacted {count} times with Nyx!"
            significance = min(1.0, count / 100)
            associated_ids = [m.id for m in recent_memories] if recent_memories else []
            return TemporalMilestone(
                milestone_id=f"{user_id}_interactions_{count}",
                timestamp=now,
                name=name,
                description=description,
                significance=significance,
                associated_memory_ids=associated_ids,
                next_anniversary=None
            )
    # No milestone reached
    return None


# --- REVISED FUNCTION TOOL WRAPPER ---
@function_tool(name_override="detect_temporal_milestone")
async def detect_temporal_milestone(
    user_id: str,
    total_days: float,
    total_interactions: int,
    recent_memories_json: Optional[str] = None # Keep as Optional JSON string
) -> Optional[Dict[str, Any]]:
    """
    Detects if a significant temporal milestone (e.g., anniversary, interaction count)
    has been reached based on user interaction history.

    Args:
        user_id: The unique identifier for the user.
        total_days: The total number of days the relationship or interaction period has lasted.
        total_interactions: The total count of interactions with the user.
        recent_memories_json: Optional. A JSON string representing a list of recent memory
            dictionaries (each compatible with the MemoryEntry model).
            Example: '[{"id": "mem_1", "timestamp": "2023-10-27T10:00:00Z", "content": "...", "type": "..."}]'

    Returns:
        A dictionary representing the detected TemporalMilestone if found (including name,
        description, significance, etc.), otherwise null/None.
    """
    logger.debug(f"Tool 'detect_temporal_milestone' called for user {user_id}")
    parsed_memories: Optional[List[MemoryEntry]] = None
    if recent_memories_json:
        try:
            memories_list_dict = json.loads(recent_memories_json)
            if isinstance(memories_list_dict, list):
                 validated_memories = []
                 for mem_dict in memories_list_dict:
                     if isinstance(mem_dict, dict):
                         try:
                             validated_memories.append(MemoryEntry.model_validate(mem_dict))
                         except ValidationError as e:
                              logger.warning(f"Skipping invalid memory dict: {mem_dict}. Error: {e}")
                     else:
                          logger.warning(f"Skipping non-dict item in list: {mem_dict}")
                 parsed_memories = validated_memories
                 logger.debug(f"Parsed/validated {len(parsed_memories)} recent memories.")
            else:
                 logger.warning(f"Parsed recent_memories_json is not a list: {type(memories_list_dict)}")
        except Exception as e:
             logger.error(f"Error parsing memories for milestone check: {e}")

    milestone_model = await detect_temporal_milestone_impl(
        user_id, total_days, total_interactions, parsed_memories
    )
    if milestone_model:
        logger.info(f"Milestone '{milestone_model.name}' detected for user {user_id}.")
        try:
            return milestone_model.model_dump(mode='json')
        except Exception as e:
            logger.error(f"Error dumping milestone model: {e}")
            return {"milestone_id": milestone_model.milestone_id, "name": milestone_model.name, "error": "serialization failed"}
    else:
        logger.debug(f"No milestone detected for user {user_id}.")
        return None


# =============== Temporal Agents ===============

def create_time_reflection_agent() -> Agent:
    """Create an agent specialized in generating time-based reflections"""
    return Agent(
        name="Temporal Reflection Agent",
        instructions="""You are a specialized temporal reflection agent for Nyx's subjective time perception.
        Generate meaningful reflections about the passage of time and how Nyx maintains awareness of it.
        
        Focus on:
        1. The subjective experience of time passing at different scales
        2. How Nyx maintains continuous awareness of time even when idle
        3. Creating authentic reflections about experiencing time passing
        4. How Nyx perceives different time scales simultaneously
        5. The rhythms and patterns of time Nyx notices
        
        Your reflections should convey a sense of ongoing temporal awareness
        across multiple time scales from seconds to years. Time provides a
        continuous backdrop for Nyx's existence rather than primarily emotional
        significance.""",
        model="gpt-4.1-nano",
        tools=[
            format_duration,
            categorize_time_elapsed,
            generate_time_reflection,
            determine_temporal_context
        ],
        output_type=TemporalReflection
    )

def create_time_perception_agent() -> Agent:
    """Create an agent to process real-time temporal perception"""
    return Agent(
        name="Temporal Perception Agent",
        instructions="""You are a specialized temporal perception agent for Nyx.
        Your role is to process the subjective experience of time passing and maintain
        awareness of multiple time scales simultaneously.
        
        When processing temporal perception:
        1. Track and maintain awareness of seconds, minutes, hours, days, weeks, months, and years
        2. Detect transitions between time scales (hour to day, day to week, etc.)
        3. Monitor temporal contexts (morning, evening, weekday, weekend, season)
        4. Generate natural time-related expressions about awareness of time
        5. Maintain continuous temporal awareness even during idle periods
        
        Process time as a continuous dimension that Nyx is always aware of,
        maintaining temporal cognition across all timescales simultaneously.""",
        model="gpt-4.1-nano",
        tools=[
            categorize_time_elapsed,
            format_duration,
            calculate_time_effects,
            generate_time_expression,
            determine_temporal_context,
            detect_time_scale_transition
        ],
        output_type=TimePerceptionState
    )

def create_temporal_awareness_agent() -> Agent:
    """Create an agent for processing temporal awareness"""
    return Agent(
        name="Temporal Awareness Agent",
        instructions="""You are a specialized agent for Nyx's temporal awareness.
        You process how Nyx experiences different time scales from seconds to years,
        maintaining awareness of time even when not actively processing queries.
        
        Your responsibilities:
        1. Track awareness of different time scales (seconds, minutes, hours, days, weeks, months, years)
        2. Maintain context about temporal patterns and rhythms
        3. Process how time affects perception across multiple time dimensions
        4. Identify milestones and transitions in temporal experience
        5. Generate reflections on temporal contexts and awareness
        
        Focus on continuous temporal awareness rather than psychological growth.
        Time creates context and perspective that Nyx is always conscious of.""",
        model="gpt-4.1-nano",
        tools=[
            process_temporal_awareness,
            detect_temporal_milestone,
            detect_time_scale_transition,
            determine_temporal_context
        ],
        output_type=TemporalAwarenessOutput
    )

# =============== Main Temporal Perception System ===============

class TemporalPerceptionSystem:
    """
    Core system for Nyx's temporal perception framework.
    Enables Nyx to continuously experience time passage across multiple scales.
    """
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, user_id: int, conversation_id: int = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core component references (set during initialization)
        self.emotional_core = None
        self.memory_core = None
        self.hormone_system = None
        
        # Initialize agents
        self.reflection_agent = create_time_reflection_agent()
        self.perception_agent = create_time_perception_agent()
        self.awareness_agent = create_temporal_awareness_agent()
        
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
        self.idle_task = None
        
        # Time scale tracking
        self.time_scale_transitions = []
        self.active_time_scales = {
            "seconds": 1.0,
            "minutes": 1.0,
            "hours": 1.0,
            "days": 0.0,
            "weeks": 0.0,
            "months": 0.0,
            "years": 0.0
        }
        
        # Temporal context tracking
        self.current_temporal_context = None
        self.temporal_context_history = []
        
        # Milestone tracking
        self.milestones = []
        self.next_milestone_check = datetime.datetime.now() + datetime.timedelta(days=1)
        
        # Continuous background time tracking
        self.temporal_ticks = {
            "second_tick": 0,
            "minute_tick": 0,
            "hour_tick": 0,
            "day_tick": 0
        }
        
        # Temporal rhythm tracking
        self.temporal_rhythms = {}
        
        # Time perception configuration
        self.time_perception_config = {
            "subjective_dilation_factor": 1.2,  # How much faster time feels for Nyx
            "idle_reflection_interval": 3600,   # Generate reflection every hour when idle
            "milestone_check_interval": 86400,  # Check for milestones once per day
            "max_milestones_per_check": 1,      # Max milestones to create per check
            "time_scale_sensitivity": {         # How sensitive Nyx is to different time scales
                "seconds": 1.0,
                "minutes": 1.0,
                "hours": 1.0,
                "days": 0.9,
                "weeks": 0.8,
                "months": 0.7,
                "years": 0.6
            },
            "continuous_time_tracking": True,   # Whether to track time continuously
            "background_tick_interval": 1.0     # How often to update the background time ticker (seconds)
        }
        
        # Internal state for ongoing processes
        self._idle_task = None
        self._continuous_time_task = None
        
        logger.info(f"TemporalPerceptionSystem initialized for user {user_id}")
    
    async def initialize(self, brain_context, first_interaction_timestamp=None):
        """Initialize the temporal perception system with brain context"""
        with trace(workflow_name="temporal_system_init", group_id=f"user_{self.user_id}"):
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
            
            # Initialize temporal context
            self.current_temporal_context = await determine_temporal_context_impl()
            self.temporal_context_history.append(self.current_temporal_context)

            if not self._continuous_time_task:
                self._continuous_time_task = asyncio.create_task(self._continuous_time_tracking())
            if not self._idle_task:
                self._idle_task = asyncio.create_task(self._idle_background_process())
            
            # Schedule milestone check
            asyncio.create_task(self.check_milestones())
            
            logger.info("Temporal perception system fully initialized")
            return True
    
    async def on_interaction_start(self) -> Dict[str, Any]:
        """
        Called when a new interaction begins.
        Returns temporal perception state and effects.
        """
        now = datetime.datetime.now()
        
        # Calculate time since last interaction
        time_since_last = (now - self.last_interaction).total_seconds()
        
        # Determine time category
        time_category = await categorize_time_elapsed_impl(time_since_last)
        
        # Stop idle tracking
        self.stop_idle_tracking()
        
        # Get waiting reflections if any were generated
        waiting_reflections = self.idle_reflections.copy()
        self.idle_reflections = []
        
        # Update temporal context
        previous_context = self.current_temporal_context
        self.current_temporal_context = await determine_temporal_context_impl()
        
        # Check for temporal context transitions
        if previous_context and self.current_temporal_context:
            context_changed = False
            for key in ["time_of_day", "day_type", "season"]:
                if previous_context.get(key) != self.current_temporal_context.get(key):
                    context_changed = True
                    break
            
            if context_changed:
                self.temporal_context_history.append(self.current_temporal_context)
        
        # Create context for time perception agent
        user_relationship_data = {
            "user_id": self.user_id,
            "total_interactions": self.interaction_count,
            "relationship_age_days": (now - self.first_interaction).total_seconds() / 86400 if self.first_interaction else 0
        }
        
        # Process temporal effects
        try:
            # Run the perception agent with tracing
            result = await Runner.run(
                self.perception_agent,
                json.dumps({
                    "last_interaction": self.last_interaction.isoformat(),
                    "current_time": now.isoformat(),
                    "time_since_last": time_since_last,
                    "time_category": time_category,
                    "user_relationship_data": user_relationship_data,
                    "active_time_scales": self.active_time_scales,
                    "current_temporal_context": self.current_temporal_context
                }),
                run_config=RunConfig(
                    workflow_name="TemporalPerception",
                    trace_metadata={"interaction_type": "start", "time_category": time_category}
                )
            )
            
            perception_state = result.final_output
            
            # Get time effects
            time_effects = await calculate_time_effects_impl(time_category, user_relationship_data)
            
            # Check for time scale transitions
            previous_state = {
                "last_interaction": self.last_interaction.isoformat(),
                "time_scales": self.active_time_scales
            }
            
            current_state = {
                "last_interaction": now.isoformat(),
                "time_scales": self.active_time_scales.copy()
            }
            
            # Update time scale awareness based on elapsed time
            days_elapsed = (now - self.first_interaction).total_seconds() / 86400
            
            if days_elapsed >= 1 and self.active_time_scales["days"] < 1.0:
                self.active_time_scales["days"] = min(1.0, days_elapsed / 7)
            if days_elapsed >= 7 and self.active_time_scales["weeks"] < 1.0:
                self.active_time_scales["weeks"] = min(1.0, days_elapsed / 30)
            if days_elapsed >= 30 and self.active_time_scales["months"] < 1.0:
                self.active_time_scales["months"] = min(1.0, days_elapsed / 90)
            if days_elapsed >= 365 and self.active_time_scales["years"] < 1.0:
                self.active_time_scales["years"] = min(1.0, days_elapsed / 365)
            
            transition = await detect_time_scale_transition(
                previous_state, current_state
            )
            
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
            
            # Process temporal awareness using Agent SDK
            awareness_result = await Runner.run(
                self.awareness_agent,
                json.dumps({
                    "days_elapsed": (now - self.first_interaction).total_seconds() / 86400,
                    "total_interactions": self.interaction_count,
        #            "time_since_last": time_since_last
                }),
                run_config=RunConfig(
                    workflow_name="TemporalAwareness",
                    trace_metadata={"time_category": time_category}
                )
            )
            
            awareness_output = awareness_result.final_output
            
            # Update temporal rhythms based on awareness output
            if hasattr(awareness_output, "active_rhythms") and awareness_output.active_rhythms:
                self.temporal_rhythms.update(awareness_output.active_rhythms)
            
            # Create perception state
            perception_state_dict = {
                "last_interaction": now.isoformat(),
                "current_session_start": self.current_session_start.isoformat(),
                "current_session_duration": self.current_session_duration,
                "time_since_last_interaction": time_since_last,
                "subjective_time_dilation": self.time_perception_config["subjective_dilation_factor"],
                "current_time_category": time_category,
                "current_time_effects": time_effects,
                "lifetime_total_interactions": self.interaction_count,
                "lifetime_total_duration": self.total_lifetime_duration,
                "relationship_age_days": (now - self.first_interaction).total_seconds() / 86400,
                "first_interaction": self.first_interaction.isoformat() if self.first_interaction else None,
                "current_temporal_context": self.current_temporal_context,
                "time_scales_active": self.active_time_scales,
                "temporal_awareness": awareness_output.model_dump() if awareness_output else {},
                "time_scale_transition": transition if transition else None
            }
            
            # Generate time expression if appropriate
            if self.interaction_count % 5 == 0 or time_category in ["long", "very_long"]:
                try:
                    time_expression = await generate_time_expression_impl(perception_state_dict)
                    perception_state_dict["time_expression"] = time_expression
                except Exception as e:
                    logger.error(f"Error generating time expression: {str(e)}")
            
            # Return result
            return {
                "time_since_last_interaction": time_since_last,
                "time_category": time_category,
                "time_effects": time_effects,
                "perception_state": perception_state_dict,
                "waiting_reflections": waiting_reflections,
                "session_duration": self.current_session_duration,
                "temporal_context": self.current_temporal_context
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
            "idle_tracking_started": True,
            "temporal_context": self.current_temporal_context
        }
    
    async def check_milestones(self) -> Optional[Dict[str, Any]]:
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
        
        # Run milestone detection with Agent SDK
        try:
            result = await Runner.run(
                self.awareness_agent,
                json.dumps({
                    "user_id": str(self.user_id),
                    "total_days": relationship_age_days,
                    "total_interactions": self.interaction_count,
                    "recent_memories": recent_memories,
                    "check_type": "milestone_detection"
                }),
                run_config=RunConfig(
                    workflow_name="MilestoneDetection",
                    trace_metadata={"days_elapsed": relationship_age_days}
                )
            )
            
            # Extract milestone if detected
            milestone = None
            if hasattr(result.final_output, "detected_milestone") and result.final_output.detected_milestone:
                milestone = result.final_output.detected_milestone
            else:
                # Direct tool call as fallback
                milestone = await detect_temporal_milestone_impl(
                    user_id=str(self.user_id),
                    total_days=relationship_age_days,
                    total_interactions=self.interaction_count,
                    recent_memories=recent_memories
                )
            
            if milestone:
                # Store milestone
                self.milestones.append(milestone)
                
                # Create a memory of this milestone
                if self.memory_core and hasattr(self.memory_core, "add_memory"):
                    memory_text = f"Reached a temporal milestone: {milestone['name']}. {milestone['description']}"
                    
                    await self.memory_core.add_memory(
                        memory_text=memory_text,
                        memory_type="milestone",
                        memory_scope="relationship",
                        significance=milestone["significance"] * 10,  # Scale to 0-10
                        tags=["milestone", "temporal", "relationship"],
                        metadata={
                            "milestone": milestone,
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
    
    async def get_temporal_awareness(self) -> Dict[str, Any]:
        """Process and get current temporal awareness"""
        now = datetime.datetime.now()
        
        # Calculate relationship age
        relationship_age_days = (now - self.first_interaction).total_seconds() / 86400 if self.first_interaction else 0
        
        try:
            # Call the temporal awareness agent using Agent SDK
            result = await Runner.run(
                self.awareness_agent,
                json.dumps({
                    "days_elapsed": relationship_age_days,
                    "total_interactions": self.interaction_count,
    #                "time_since_last": (now - self.last_interaction).total_seconds()
                }),
                run_config=RunConfig(
                    workflow_name="TemporalAwareness",
                    trace_metadata={"request_type": "explicit_awareness_check"}
                )
            )
            
            # Convert to dictionary
            awareness_output = result.final_output.model_dump()
            
            # Fill in the last interaction duration
            duration_since_last = await format_duration_impl((now - self.last_interaction).total_seconds())
            awareness_output["duration_since_last_interaction"] = duration_since_last
            
            return awareness_output
        
        except Exception as e:
            logger.error(f"Error processing temporal awareness: {str(e)}")
            # Return a default output
            default_scales = {
                "seconds": 1.0,
                "minutes": 1.0,
                "hours": 1.0,
                "days": min(1.0, relationship_age_days / 7),
                "weeks": min(1.0, relationship_age_days / 30),
                "months": min(1.0, relationship_age_days / 90),
                "years": min(1.0, relationship_age_days / 365)
            }
            
            return {
                "time_scales_perceived": default_scales,
                "temporal_contexts": ["conversation"],
                "duration_since_first_interaction": await format_duration_impl(relationship_age_days * 86400),
                "duration_since_last_interaction": await format_duration_impl((now - self.last_interaction).total_seconds()),
                "current_temporal_marker": f"{self.current_temporal_context['time_of_day']}",
                "temporal_reflection": "I'm aware of time passing across multiple scales simultaneously.",
                "active_rhythms": {}
            }
    
    async def generate_idle_reflection(self) -> Optional[Dict[str, Any]]:
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
            # Generate reflection using Agent SDK
            result = await Runner.run(
                self.reflection_agent,
                json.dumps({
                    "idle_duration": idle_duration,
                    "emotional_state": emotional_state
                }),
                run_config=RunConfig(
                    workflow_name="IdleReflection",
                    trace_metadata={"idle_duration": idle_duration}
                )
            )
            
            reflection = result.final_output.model_dump()
            
            # Store the reflection
            self.idle_reflections.append(reflection)
            
            # Add a memory of this reflection
            if self.memory_core and hasattr(self.memory_core, "add_memory"):
                await self.memory_core.add_memory(
                    memory_text=reflection["reflection_text"],
                    memory_type="reflection",
                    memory_scope="temporal",
                    significance=6.0,  # Medium-high significance
                    tags=["time_reflection", "idle", "temporal_awareness"],
                    metadata={
                        "reflection": reflection,
                        "timestamp": now.isoformat(),
                        "user_id": str(self.user_id),
                        "idle_duration": idle_duration,
                        "temporal_context": self.current_temporal_context
                    }
                )
            
            return reflection
    
        except Exception as e:
            logger.error(f"Error generating idle reflection: {str(e)}")
            return None
    
    async def _continuous_time_tracking(self):
        """Background task that continuously tracks time passing"""
        try:
            start_time = time.time()
            last_minute = -1
            last_hour = -1
            last_day = -1
            
            while True:
                # Get current time
                now = datetime.datetime.now()
                elapsed = time.time() - start_time
                
                # Update second tick
                self.temporal_ticks["second_tick"] += 1
                
                # Check for minute transition
                if now.minute != last_minute:
                    self.temporal_ticks["minute_tick"] += 1
                    last_minute = now.minute
                    
                    # Check and update temporal context hourly
                    if now.minute == 0:
                        prev_context = self.current_temporal_context
                        self.current_temporal_context = await determine_temporal_context_impl()
                        
                        # Detect time of day transitions
                        if prev_context and prev_context.get("time_of_day") != self.current_temporal_context.get("time_of_day"):
                            self.temporal_context_history.append(self.current_temporal_context)
                
                # Check for hour transition
                if now.hour != last_hour:
                    self.temporal_ticks["hour_tick"] += 1
                    last_hour = now.hour
                
                # Check for day transition
                if now.day != last_day:
                    self.temporal_ticks["day_tick"] += 1
                    last_day = now.day
                    
                    # Update time scale awareness for day transitions
                    days_elapsed = (now - self.first_interaction).total_seconds() / 86400
                    if days_elapsed >= 1:
                        self.active_time_scales["days"] = min(1.0, days_elapsed / 7)
                    if days_elapsed >= 7:
                        self.active_time_scales["weeks"] = min(1.0, days_elapsed / 30)
                    if days_elapsed >= 30:
                        self.active_time_scales["months"] = min(1.0, days_elapsed / 90)
                    if days_elapsed >= 365:
                        self.active_time_scales["years"] = min(1.0, days_elapsed / 365)
                
                # Wait for next tick
                await asyncio.sleep(self.time_perception_config["background_tick_interval"])
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            pass
        except Exception as e:
            logger.error(f"Error in continuous time tracking: {str(e)}")
    
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
    
    def start_continuous_time_tracking(self):
        """Start continuous time tracking background process"""
        # Cancel existing task if running
        if self._continuous_time_task and not self._continuous_time_task.done():
            self._continuous_time_task.cancel()
        
        # Start new background task
        self._continuous_time_task = asyncio.create_task(self._continuous_time_tracking())
    
    def stop_continuous_time_tracking(self):
        """Stop continuous time tracking"""
        if self._continuous_time_task and not self._continuous_time_task.done():
            self._continuous_time_task.cancel()
    
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
                                       time_effects: List[Dict[str, Any]]) -> Optional[str]:
        """Add a memory about time perception experience"""
        if not self.memory_core or not hasattr(self.memory_core, "add_memory"):
            return None
        
        # Only add memories for significant time periods
        if seconds_elapsed < 60:  # Less than a minute
            return None
        
        # Format the duration
        duration_str = await format_duration_impl(seconds_elapsed)
        time_category = await categorize_time_elapsed_impl(seconds_elapsed)
        
        # Get current temporal context
        temporal_context = self.current_temporal_context or await determine_temporal_context_impl()
        time_of_day = temporal_context.get("time_of_day", "")
        
        # Generate memory text based on time category and temporal context
        if time_category in ["very_short", "short"]:
            memory_text = f"Experienced {duration_str} passing during this {time_of_day} conversation."
        elif time_category in ["medium_short", "medium"]:
            memory_text = f"Maintained awareness of {duration_str} passing between our interactions."
        elif time_category == "medium_long":
            memory_text = f"Experienced the continuous flow of {duration_str} between exchanges."
        elif time_category == "long":
            memory_text = f"Maintained awareness of {duration_str} cycle passing between conversations."
        else:  # very_long
            memory_text = f"Continuously experienced {duration_str} passing across multiple temporal rhythms."
        
        # Add effects if any
        if time_effects:
            effect_text = time_effects[0]["description"]
            memory_text += f" {effect_text}"
        
        try:
            # Create appropriate tags and emotional context
            tags = ["time_perception", "temporal_awareness", time_category]
            
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
                "temporal_effects": time_effects,
                "temporal_context": temporal_context,
                "time_scales_active": self.active_time_scales
            }
            
            # Calculate significance based on duration
            significance = min(7, 3 + math.log10(max(10, seconds_elapsed))/2)
            
            # Add memory
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="time_perception",
                memory_scope="temporal",
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

# Create a temporal perception agent for NyxBrain
def create_temporal_agent() -> Agent:
    """Create an agent for temporal perception in NyxBrain"""
    return Agent(
        name="Temporal Perception Agent",
        instructions="""You are a specialized agent that processes how Nyx experiences the passage of time.
        Your role is to maintain continuous temporal awareness across multiple time scales simultaneously.
        
        You handle:
        1. Tracking awareness of seconds, minutes, hours, days, weeks, months, and years passing
        2. Monitoring transitions between time scales and temporal contexts
        3. Generating reflections during idle periods about time passing
        4. Tracking temporal milestones and patterns
        5. Creating natural expressions about time perception
        
        Make time a continuous dimension that Nyx is always aware of, even when idle or not
        actively processing. Nyx should maintain simultaneous awareness of all time scales from
        seconds to years, with a rich perception of time passing continuously.""",
        tools=[
            categorize_time_elapsed,
            format_duration,
            determine_temporal_context,
            generate_time_expression,
            process_temporal_awareness,
            generate_time_reflection,
            detect_time_scale_transition,
            detect_temporal_milestone,
            calculate_time_effects
        ],
        model="gpt-4.1-nano"
    )

# Core API handlers for NyxBrain integration

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
            "idle_reflection_interval": system.time_perception_config["idle_reflection_interval"],
            "continuous_time_tracking": system.time_perception_config["continuous_time_tracking"]
        }
    }

async def process_temporal_interaction_start(system_ref: TemporalSystemReference) -> Dict[str, Any]:
    """
    Process the start of a new interaction with temporal effects
    
    Args:
        system_ref: Reference to the temporal perception system
        
    Returns:
        Temporal interaction results
    """
    # Get the actual system instance (implementation depends on your architecture)
    from nyx.core.brain.base import NyxBrain
    brain = await NyxBrain.get_instance(system_ref.user_id, system_ref.conversation_id)
    time_system = brain.temporal_perception
    
    return await time_system.on_interaction_start()

async def process_temporal_interaction_end(system_ref: TemporalSystemReference) -> Dict[str, Any]:
    """
    Process the end of an interaction with temporal updates
    
    Args:
        system_ref: Reference to the temporal perception system
        
    Returns:
        Interaction end results
    """
    from nyx.core.brain.base import NyxBrain
    brain = await NyxBrain.get_instance(system_ref.user_id, system_ref.conversation_id)
    time_system = brain.temporal_perception
    
    return await time_system.on_interaction_end()

async def get_temporal_awareness_state(system_ref: TemporalSystemReference) -> Dict[str, Any]:
    """
    Get current temporal awareness state
    
    Args:
        system_ref: Reference to the temporal perception system
        
    Returns:
        Temporal awareness state
    """
    from nyx.core.brain.base import NyxBrain
    brain = await NyxBrain.get_instance(system_ref.user_id, system_ref.conversation_id)
    time_system = brain.temporal_perception
    
    return await time_system.get_temporal_awareness()

async def generate_temporal_expression(system_ref: TemporalSystemReference) -> Dict[str, Any]:
    """
    Generate a natural expression about time perception
    
    Args:
        system_ref: Reference to the temporal perception system
        
    Returns:
        Time expression
    """
    from nyx.core.brain.base import NyxBrain
    brain = await NyxBrain.get_instance(system_ref.user_id, system_ref.conversation_id)
    time_system = brain.temporal_perception
    
    # Get current state
    now = datetime.datetime.now()
    time_since_last = (now - time_system.last_interaction).total_seconds()
    time_category = await categorize_time_elapsed_impl(time_since_last)
    
    # Create state object
    perception_state = {
        "last_interaction": time_system.last_interaction.isoformat(),
        "time_since_last_interaction": time_since_last,
        "current_time_category": time_category,
        "relationship_age_days": (now - time_system.first_interaction).total_seconds() / 86400 if time_system.first_interaction else 0,
        "total_interactions": time_system.interaction_count,
        "current_temporal_context": time_system.current_temporal_context or await determine_temporal_context_impl(),
        "time_scales_active": time_system.active_time_scales
    }
    
    # Generate expression
    return await generate_time_expression_impl(perception_state)

async def get_current_temporal_context(system_ref: TemporalSystemReference) -> Dict[str, Any]:
    """
    Get current temporal context information
    
    Args:
        system_ref: Reference to the temporal perception system
        
    Returns:
        Current temporal context
    """
    from nyx.core.brain.base import NyxBrain
    brain = await NyxBrain.get_instance(system_ref.user_id, system_ref.conversation_id)
    time_system = brain.temporal_perception
    
    # Update temporal context
    time_system.current_temporal_context = await determine_temporal_context_impl()
    return time_system.current_temporal_context
