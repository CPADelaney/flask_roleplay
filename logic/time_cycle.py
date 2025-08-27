# logic/time_cycle.py
"""
Enhanced Time Cycle & Conflict System Module with LLM-Powered Agents

This module integrates intelligent agents to replace hard-coded logic:
  - PlayerIntentAgent for activity classification
  - NarrativeDirectorAgent for event selection
  - VitalsNarrator for contextual crisis descriptions
  - IntensityScorer for nuanced intensity calculation
  - EventWriterAgent for dynamic conflict events
  - PhaseRecapAgent for phase summaries and suggestions

REFACTORED: Now uses the new dynamic relationships system instead of discrete levels

Code Review Fixes Applied:
  - Fixed SDK compatibility (result.output vs result.tool_output)
  - Proper ToolCallSpec format for function tools
  - Database table existence checks
  - RNG seed support for reproducibility
  - Performance optimization notes
  - Governance registration with correct agent_id
  - Pydantic models for all data structures
"""

import logging
import random
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum

from agents import Agent, Runner, function_tool
from agents.run_context import RunContextWrapper

from db.connection import get_db_connection_context
import asyncpg
from lore.core import canon

# Add Pydantic imports
from pydantic import BaseModel, Field, validator

try:
    from utils.background import enqueue_task  # optional
except Exception:
    enqueue_task = None

logger = logging.getLogger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PYDANTIC MODELS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VitalsData(BaseModel):
    """Player vitals data matching PlayerVitals table"""
    energy: int = Field(100, ge=0, le=100, description="Energy level")
    hunger: int = Field(100, ge=0, le=100, description="Hunger level (100=full, 0=starving)")
    thirst: int = Field(100, ge=0, le=100, description="Thirst level (100=hydrated, 0=dehydrated)")
    fatigue: int = Field(0, ge=0, le=100, description="Fatigue level (0=rested, 100=exhausted)")
    
    # Optional fields from database
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    player_name: str = Field("Chase", description="Player name")
    last_update: Optional[datetime] = None
    
    model_config = {"extra": "forbid"}
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to simple dict for compatibility"""
        return {
            "energy": self.energy,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "fatigue": self.fatigue
        }

class VitalsSummary(BaseModel):
    """Simplified vitals for event recommendations"""
    energy: int = Field(..., ge=0, le=100)
    hunger: int = Field(..., ge=0, le=100)
    thirst: int = Field(..., ge=0, le=100)
    fatigue: int = Field(..., ge=0, le=100)
    
    model_config = {"extra": "forbid"}


class PlayerStatsData(BaseModel):
    """Player stats matching PlayerStats table"""
    # Visible stats
    hp: int = Field(100, ge=0, description="Current health points")
    max_hp: int = Field(100, ge=1, le=999, description="Maximum health points")
    strength: int = Field(10, ge=1, le=100, description="Physical power")
    endurance: int = Field(10, ge=1, le=100, description="Stamina and defense")
    agility: int = Field(10, ge=1, le=100, description="Speed and reflexes")
    empathy: int = Field(10, ge=1, le=100, description="Social intuition")
    intelligence: int = Field(10, ge=1, le=100, description="Learning ability")
    
    # Hidden stats
    corruption: int = Field(10, ge=0, le=100, description="Moral degradation")
    confidence: int = Field(60, ge=0, le=100, description="Self-assurance")
    willpower: int = Field(50, ge=0, le=100, description="Resistance to control")
    obedience: int = Field(20, ge=0, le=100, description="Compliance level")
    dependency: int = Field(10, ge=0, le=100, description="Reliance on others")
    lust: int = Field(15, ge=0, le=100, description="Arousal and desire")
    mental_resilience: int = Field(55, ge=0, le=100, description="Psychological endurance")
    physical_endurance: int = Field(40, ge=0, le=100, description="Legacy stat")
    
    # Metadata
    player_name: str = Field("Chase", description="Player name")
    stat_visibility: Optional[Dict[str, bool]] = None

class ActivityLogEntry(BaseModel):
    """Entry in the activity log"""
    activity_type: str = Field(..., description="Activity type")
    timestamp: datetime = Field(..., description="When activity occurred")
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    intensity: Optional[float] = Field(None, ge=0.5, le=1.5)
    duration: Optional[int] = Field(None, description="Duration in time periods")
    
    model_config = {"extra": "forbid"}

class RelationshipStandingData(BaseModel):
    """Data for a single relationship standing"""
    npc_id: str = Field(..., description="NPC ID")
    trust: float = Field(..., ge=-100.0, le=100.0)
    affection: float = Field(..., ge=-100.0, le=100.0)
    patterns: List[str] = Field(default_factory=list)
    archetypes: List[str] = Field(default_factory=list)
    
    model_config = {"extra": "forbid"}

class RelationshipStandingsMap(BaseModel):
    """Map of NPC IDs to their relationship standings"""
    standings: Dict[str, RelationshipStandingData] = Field(
        default_factory=dict, 
        description="Map of NPC ID to relationship standing data"
    )
    
    model_config = {"extra": "forbid"}

class ActivityLogSummary(BaseModel):
    """Summary of activity log for event recommendation"""
    activity_type: str = Field(..., description="Type of activity")
    timestamp: str = Field(..., description="ISO format timestamp")
    
    model_config = {"extra": "forbid"}

class EventRecommendationRequest(BaseModel):
    """Request data for event recommendations"""
    activity_log: List[ActivityLogSummary] = Field(default_factory=list)
    vitals: VitalsData
    plot_flags: List[str] = Field(default_factory=list)
    relationship_standings: RelationshipStandingsMap
    
    model_config = {"extra": "forbid"}

class PhaseEventEntry(BaseModel):
    """Event that occurred during a phase - used for phase recaps"""
    type: str = Field(..., description="Event type")
    description: Optional[str] = Field(None, description="Event description")
    
    # Event-specific fields based on event types in the code
    # Vital crisis events
    crisis_type: Optional[str] = Field(None, description="Type of crisis (hunger/thirst/fatigue)")
    severity: Optional[str] = Field(None, description="Crisis severity (moderate/severe)")
    message: Optional[str] = Field(None, description="Crisis message")
    
    # Forced events
    event: Optional[str] = Field(None, description="Forced event type (e.g., sleep)")
    reason: Optional[str] = Field(None, description="Reason for forced event")
    
    # Relationship events - UPDATED FOR DYNAMIC SYSTEM
    total_relationships: Optional[int] = Field(None, ge=0)
    active_patterns: Optional[List[str]] = Field(None, description="Active relationship patterns")
    active_archetypes: Optional[List[str]] = Field(None, description="Active relationship archetypes")
    # Changed from List[Dict[str, Any]] to a more specific type
    most_significant: Optional[List[str]] = Field(None, description="Most significant relationships as JSON strings")
    state_key: Optional[str] = Field(None, description="Relationship state key")
    
    # NPC-specific
    npc_id: Optional[Union[int, str]] = Field(None, description="NPC ID if relevant")
    npc_name: Optional[str] = Field(None, description="NPC name")
    
    # Conflict events
    conflict_id: Optional[int] = Field(None)
    conflict_name: Optional[str] = Field(None)
    faction: Optional[str] = Field(None)
    faction_name: Optional[str] = Field(None)
    
    # Resource events
    resource_type: Optional[str] = Field(None)
    resource_amount: Optional[int] = Field(None, ge=0)
    expiration: Optional[int] = Field(None)
    
    # Progress tracking
    progress_impact: Optional[int] = Field(None)
    
    # Phase recap specific
    recap: Optional[str] = Field(None, description="Phase summary text")
    suggestions: Optional[List[str]] = Field(None, description="Suggested next actions")
    
    # Metadata
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    significance: Optional[int] = Field(None, ge=1, le=10)
    
    model_config = {"extra": "forbid"}
  
class IntentClassification(BaseModel):
    """Result of classifying player intent"""
    activity_type: str = Field(..., description="Activity type from ActivityType enum")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    model_config = {"extra": "forbid"}

class IntensityScore(BaseModel):
    """Result of scoring activity intensity"""
    intensity: float = Field(..., ge=0.5, le=1.5, description="Intensity multiplier")
    mood: str = Field(..., description="Detected mood")
    risk: str = Field(..., description="Risk level")
    
    model_config = {"extra": "forbid"}

class EventRecommendation(BaseModel):
    """A single event recommendation"""
    event: str = Field(..., description="Event type")
    score: float = Field(..., ge=0.0, le=1.0, description="Event score")
    npc_id: Optional[str] = Field(None, description="NPC ID if relevant")
    
    model_config = {"extra": "forbid"}

class PhaseRecapResult(BaseModel):
    """Phase recap and suggestions"""
    recap: str = Field(..., description="Phase summary")
    suggestions: List[str] = Field(..., description="Suggested next actions")
    
    model_config = {"extra": "forbid"}

class CombinedAnalysis(BaseModel):
    """Combined intent and intensity analysis"""
    activity_type: str = Field(..., description="Activity type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    intensity: float = Field(..., ge=0.5, le=1.5, description="Intensity multiplier")
    mood: str = Field(..., description="Detected mood")
    risk: str = Field(..., description="Risk level")
    
    model_config = {"extra": "forbid"}

class NPCStanding(BaseModel):
    """NPC relationship standing using dynamic system"""
    npc_id: int
    npc_name: str
    general_standing: int = Field(0, ge=-100, le=100, description="General relationship quality based on trust+affection")
    dominant_emotion: Optional[str] = Field(None, description="Primary emotion in relationship")
    patterns: Optional[List[str]] = Field(default_factory=list, description="Active relationship patterns")
    
    model_config = {"extra": "forbid"}

class CurrentTimeData(BaseModel):
    """Current time information"""
    year: int = Field(1, ge=1)
    month: int = Field(1, ge=1, le=12)
    day: int = Field(1, ge=1, le=31)
    time_of_day: str = Field("Morning", pattern="^(Morning|Afternoon|Evening|Night)$")
    
    model_config = {"extra": "forbid"}

class VitalEffectsData(BaseModel):
    """Effects of activities on vitals"""
    hunger: int = Field(0, ge=-100, le=100)
    thirst: int = Field(0, ge=-100, le=100)
    fatigue: int = Field(0, ge=-100, le=100)
    energy: int = Field(0, ge=-100, le=100)

class ActivityDefinition(BaseModel):
    """Definition of an activity"""
    time_advance: int = Field(0, ge=0, description="Time periods to advance")
    description: str = Field(..., description="Activity description")
    stat_effects: Dict[str, int] = Field(default_factory=dict)
    vital_effects: Dict[str, int] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class EventData(BaseModel):
    """Event data matching Events table"""
    event_name: str = Field(..., description="Event name")
    description: Optional[str] = Field(None, description="Event description")
    start_time: str = Field(..., description="Start time")
    end_time: str = Field(..., description="End time")
    location: str = Field(..., description="Event location")
    year: int = Field(1, ge=1, description="Year")
    month: int = Field(1, ge=1, le=12, description="Month")
    day: int = Field(1, ge=1, le=31, description="Day")
    time_of_day: str = Field("Morning", pattern="^(Morning|Afternoon|Evening|Night)$")
    
    # Optional fields
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    embedding: Optional[List[float]] = Field(None, min_length=1536, max_length=1536)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Activity Types Enum for LLM Classification
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ActivityType(str, Enum):
    # Time-consuming activities
    CLASS_ATTENDANCE = "class_attendance"
    WORK_SHIFT = "work_shift"
    SOCIAL_EVENT = "social_event"
    TRAINING = "training"
    EXTENDED_CONVERSATION = "extended_conversation"
    PERSONAL_TIME = "personal_time"
    SLEEP = "sleep"
    EATING = "eating"
    DRINKING = "drinking"
    INTENSE_ACTIVITY = "intense_activity"
    
    # Quick activities
    QUICK_CHAT = "quick_chat"
    OBSERVE = "observe"
    CHECK_PHONE = "check_phone"
    QUICK_SNACK = "quick_snack"
    REST = "rest"

ALL_ACTIVITY_TYPES = [activity.value for activity in ActivityType]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants (unchanged from original)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12
TIME_PHASES = ["Morning", "Afternoon", "Evening", "Night"]
TIME_PRIORITY = {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4}

# Enhanced activity definitions remain the same
TIME_CONSUMING_ACTIVITIES = {
    "class_attendance": {
        "time_advance": 1,
        "description": "Attending classes or lectures",
        "stat_effects": {"intelligence": +1, "mental_resilience": +1},
        "vital_effects": {"fatigue": +10, "hunger": -5, "thirst": -5}
    },
    "work_shift": {
        "time_advance": 1,
        "description": "Working at a job",
        "stat_effects": {"endurance": +1},
        "vital_effects": {"fatigue": +15, "hunger": -10, "thirst": -10}
    },
    "social_event": {
        "time_advance": 1,
        "description": "Attending a social gathering",
        "stat_effects": {"empathy": +1, "confidence": +1},
        "vital_effects": {"fatigue": +5, "thirst": -15, "hunger": -5}
    },
    "training": {
        "time_advance": 1,
        "description": "Physical or mental training",
        "stat_effects": {"strength": +1, "endurance": +1, "willpower": +1},
        "vital_effects": {"fatigue": +20, "hunger": -15, "thirst": -20}
    },
    "extended_conversation": {
        "time_advance": 1,
        "description": "A lengthy, significant conversation",
        "stat_effects": {"empathy": +1},
        "vital_effects": {"thirst": -10, "fatigue": +3}
    },
    "personal_time": {
        "time_advance": 1,
        "description": "Spending time on personal activities",
        "stat_effects": {"mental_resilience": +1},
        "vital_effects": {"fatigue": -10}  # Restful
    },
    "sleep": {
        "time_advance": 2,
        "description": "Going to sleep for the night",
        "stat_effects": {"hp": +20, "mental_resilience": +3},
        "vital_effects": {"fatigue": -80, "hunger": -10, "thirst": -5}  # Reset fatigue
    },
    "eating": {
        "time_advance": 0,  # Can be done without time advance
        "description": "Having a meal",
        "stat_effects": {},
        "vital_effects": {"hunger": +30, "thirst": +10}
    },
    "drinking": {
        "time_advance": 0,
        "description": "Drinking water or beverages",
        "stat_effects": {},
        "vital_effects": {"thirst": +40}
    },
    "intense_activity": {
        "time_advance": 1,
        "description": "Engaging in intense physical or mental activity",
        "stat_effects": {"varies": True},  # Depends on specific activity
        "vital_effects": {"fatigue": +25, "hunger": -20, "thirst": -25}
    }
}

OPTIONAL_ACTIVITIES = {
    "quick_chat": {
        "time_advance": 0,
        "description": "A brief conversation",
        "stat_effects": {},
        "vital_effects": {"thirst": -2}
    },
    "observe": {
        "time_advance": 0,
        "description": "Observing surroundings or people",
        "stat_effects": {},
        "vital_effects": {}
    },
    "check_phone": {
        "time_advance": 0,
        "description": "Looking at messages or notifications",
        "stat_effects": {},
        "vital_effects": {"fatigue": +1}  # Eye strain
    },
    "quick_snack": {
        "time_advance": 0,
        "description": "Having a quick snack",
        "stat_effects": {},
        "vital_effects": {"hunger": +10}
    },
    "rest": {
        "time_advance": 0,
        "description": "Taking a brief rest",
        "stat_effects": {},
        "vital_effects": {"fatigue": -5}
    }
}

# Keep other constants from original
VITAL_DRAIN_RATES = {
    "Morning": {"hunger": 3, "thirst": 4, "fatigue": 2},
    "Afternoon": {"hunger": 4, "thirst": 6, "fatigue": 3},
    "Evening": {"hunger": 5, "thirst": 4, "fatigue": 4},
    "Night": {"hunger": 2, "thirst": 2, "fatigue": 6}
}

VITAL_THRESHOLDS = {
    "hunger": {
        "full": {"min": 80, "effects": {}},
        "satisfied": {"min": 60, "effects": {}},
        "hungry": {"min": 40, "effects": {"strength": -1, "concentration": -1}},
        "very_hungry": {"min": 20, "effects": {"strength": -3, "agility": -2, "intelligence": -1}},
        "starving": {"min": 0, "effects": {"strength": -5, "agility": -3, "intelligence": -2, "hp": -1}}
    },
    "thirst": {
        "hydrated": {"min": 80, "effects": {}},
        "normal": {"min": 60, "effects": {}},
        "thirsty": {"min": 40, "effects": {"intelligence": -1, "empathy": -1}},
        "very_thirsty": {"min": 20, "effects": {"intelligence": -3, "agility": -2, "mental_resilience": -2}},
        "dehydrated": {"min": 0, "effects": {"all_stats": -3, "hp": -2}}
    },
    "fatigue": {
        "rested": {"max": 20, "effects": {"all_stats": +1}},
        "normal": {"max": 40, "effects": {}},
        "tired": {"max": 60, "effects": {"agility": -2, "intelligence": -1}},
        "exhausted": {"max": 80, "effects": {"strength": -3, "agility": -3, "mental_resilience": -3}},
        "collapsing": {"max": 100, "effects": {"all_stats": -5, "forced_sleep": True}}
    }
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DYNAMIC RELATIONSHIP SUMMARY FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def get_dynamic_relationship_summary(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """Get a summary of all active relationships using the new dynamic system."""
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    
    async with get_db_connection_context() as conn:
        # Get all active relationships
        relationships = await conn.fetch("""
            SELECT sl.*, ns1.npc_name as entity1_name, ns2.npc_name as entity2_name
            FROM SocialLinks sl
            LEFT JOIN NPCStats ns1 ON sl.entity1_id = ns1.npc_id 
                AND sl.entity1_type = 'npc' AND sl.user_id = ns1.user_id
            LEFT JOIN NPCStats ns2 ON sl.entity2_id = ns2.npc_id 
                AND sl.entity2_type = 'npc' AND sl.user_id = ns2.user_id
            WHERE sl.user_id = $1 AND sl.conversation_id = $2
            AND (sl.entity1_type = 'player' OR sl.entity2_type = 'player')
            ORDER BY sl.last_interaction DESC
            LIMIT 10
        """, user_id, conversation_id)
    
    if not relationships:
        return None
    
    summary = {
        "total_relationships": len(relationships),
        "most_significant": [],
        "active_patterns": set(),
        "active_archetypes": set()
    }
    
    for rel in relationships[:3]:  # Top 3 most recent
        # Get the actual state for more details
        if rel['entity1_type'] == 'player':
            entity1_type, entity1_id = 'npc', rel['entity2_id']
            entity2_type, entity2_id = 'player', rel['entity1_id']
            npc_name = rel['entity2_name']
        else:
            entity1_type, entity1_id = 'player', rel['entity2_id']
            entity2_type, entity2_id = 'npc', rel['entity1_id']
            npc_name = rel['entity1_name']
        
        state = await manager.get_relationship_state(
            entity1_type, entity1_id, entity2_type, entity2_id
        )
        
        rel_summary = {
            "npc_name": npc_name or f"NPC_{entity2_id if entity2_type == 'npc' else entity1_id}",
            "trust": float(state.dimensions.trust),  # Ensure float
            "affection": float(state.dimensions.affection),  # Ensure float
            "patterns": list(state.history.active_patterns),
            "archetypes": list(state.active_archetypes),
            "momentum": float(state.momentum.get_magnitude())  # Ensure float
        }
        summary["most_significant"].append(rel_summary)  # Store as dict, not JSON string
        summary["active_patterns"].update(state.history.active_patterns)
        summary["active_archetypes"].update(state.active_archetypes)
    
    summary["active_patterns"] = list(summary["active_patterns"])
    summary["active_archetypes"] = list(summary["active_archetypes"])
    
    return summary

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LLM-Powered Agents (remain the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Check for dev mode to skip LLM calls
try:
    import settings
    LLM_VERBOSE = getattr(settings, "LLM_VERBOSE", True)
except ImportError:
    LLM_VERBOSE = True  # Default to using LLMs if settings not found

# 1. PlayerIntentAgent - Replaces keyword-based classification
@function_tool
def classify_intent(sentence: str, location: str = "unknown") -> IntentClassification:
    """
    Classify player intent with confidence score.
    
    Returns:
        IntentClassification with activity_type and confidence
    """
    # Schema is enforced by return type annotation
    return IntentClassification(activity_type="quick_chat", confidence=0.9)

PlayerIntentAgent = Agent(
    name="PlayerIntentAgent",
    instructions="""You are an expert at understanding player intent in a femdom university game.
    Analyze the player's input and determine what activity they're trying to perform.
    
    Consider:
    - Slang, colloquialisms, and vernacular (e.g., "pull an all-nighter" = class_attendance/training)
    - Context from location and recent activities
    - Femdom/BDSM activities should map to appropriate intensity levels
    - Persona-style school activities
    
    Examples:
    - "Netflix and chill" → personal_time or rest
    - "Hit the books" → class_attendance
    - "Grab a bite" → eating or quick_snack
    - "Crash for the night" → sleep
    - "Netflix and choke me" → intense_activity
    
    Return the most appropriate ActivityType with a confidence score (0.0-1.0).
    
    Valid activity types: """ + ", ".join(ALL_ACTIVITY_TYPES),
    tools=[classify_intent]
)

# 2. IntensityScorer - Calculates nuanced intensity
@function_tool
def score_intensity(sentence: str, vitals: VitalsData, context_tags: List[str]) -> IntensityScore:
    """Score the intensity of an activity based on language, vitals, and context."""
    # This would be called by the IntensityScorer agent
    return IntensityScore(
        intensity=0.8,
        mood="playful",
        risk="low"
    )


@function_tool
def generate_event_content(
    event_type: str,
    context_data_json: str  # Changed from Dict[str, Any] to JSON string
) -> str:  # Return JSON string
    """Generate dynamic content for a narrative event based on context."""
    context_data = json.loads(context_data_json)
    
    # Generate content based on event type
    content = f"A {event_type} occurs."
    metadata = {"generated": True}
    
    if event_type == "vital_crisis":
        crisis_type = context_data.get("crisis_type", "unknown")
        severity = context_data.get("severity", "moderate")
        content = f"You're experiencing a {severity} {crisis_type} crisis!"
    elif event_type == "dream_sequence":
        content = "Strange dreams fill your mind as exhaustion takes hold..."
    elif event_type == "npc_revelation":
        npc_id = context_data.get("npc_id", "unknown")
        content = f"You have a sudden realization about NPC {npc_id}..."
    
    result = {
        "event_type": event_type,
        "content": content,
        "metadata": metadata
    }
    
    return json.dumps(result)

# Now update the recommend_events function to generate actual content
@function_tool
async def recommend_events(
    activity_log_json: str,
    vitals_json: str,
    plot_flags_json: str,
    relationships_json: str
) -> str:
    """Recommend narrative events based on game state. Returns JSON array with generated content."""
    
    # Parse the JSON strings
    activity_log = json.loads(activity_log_json)
    vitals = json.loads(vitals_json)
    plot_flags = json.loads(plot_flags_json)
    relationships = json.loads(relationships_json)
    
    recommendations = []
    
    # Helper to generate event content
    async def generate_event(event_type: str, score: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actual event content using LLM"""
        
        # Build context for the event generation
        event_context = {
            "event_type": event_type,
            "vitals": vitals,
            "recent_activities": activity_log[:3] if activity_log else [],
            "score": score,
            **context
        }
        
        # Use EventContentAgent to generate content
        prompt = f"""Generate content for a {event_type} event.
        
        Context:
        - Vitals: Energy {vitals.get('energy', 100)}, Hunger {vitals.get('hunger', 100)}, Thirst {vitals.get('thirst', 100)}, Fatigue {vitals.get('fatigue', 0)}
        - Recent activities: {[a['activity_type'] for a in activity_log[:3]]}
        """
        
        if 'npc_info' in context:
            npc_info = context['npc_info']
            prompt += f"\n- NPC: {npc_info.get('name', 'Unknown')}"
            prompt += f"\n- Relationship: Trust {npc_info.get('trust', 0)}, Affection {npc_info.get('affection', 0)}"
            prompt += f"\n- Patterns: {npc_info.get('patterns', [])}"
            prompt += f"\n- Archetypes: {npc_info.get('archetypes', [])}"
        
        if 'trigger_reason' in context:
            prompt += f"\n- Trigger: {context['trigger_reason']}"
        
        result = await Runner.run(
            EventContentAgent,
            messages=[{"role": "user", "content": prompt}],
            calls=[{
                "name": "generate_event_content",
                "kwargs": {
                    "event_type": event_type,
                    "context_data_json": json.dumps(event_context)  # <-- correct name + JSON
                }
            }]
        )
        
        # Extract generated content (parse JSON string safely)
        if result.output:
            try:
                gen = json.loads(result.output)
            except Exception:
                gen = {"event_type": event_type, "content": str(result.output), "metadata": {}}
            return {
                "event": event_type,
                "score": score,
                "generated_content": gen,
                **context
            }
        
        # Fallback
        return {
            "event": event_type,
            "score": score,
            "generated_content": f"A {event_type} occurs.",
            **context
        }
    
    # Vital-based events with context
    if vitals.get("fatigue", 0) > 80:
        event = await generate_event(
            "dream_sequence",
            0.85,
            {
                "trigger_reason": "extreme_fatigue",
                "dream_theme": "loss_of_control" if vitals.get("fatigue", 0) > 90 else "subtle_manipulation",
                "recent_npcs": [r.get("npc_id") for r in relationships[:2]]
            }
        )
        recommendations.append(event)
    
    if vitals.get("hunger", 100) < 20:
        # Find nearby NPC for the crisis
        dominant_npc = max(relationships, key=lambda r: r.get("trust", 0), default=None)
        event = await generate_event(
            "vital_crisis",
            0.9,
            {
                "crisis_type": "hunger",
                "severity": "severe" if vitals.get("hunger", 100) < 10 else "moderate",
                "npc_info": dominant_npc if dominant_npc else {},
                "trigger_reason": "dangerously_hungry"
            }
        )
        recommendations.append(event)
    
    # Activity pattern analysis
    if len(activity_log) >= 3:
        recent_activities = [a["activity_type"] for a in activity_log[:5]]
        social_count = sum(1 for a in recent_activities if a in [
            "social_event", "extended_conversation", "quick_chat"
        ])
        
        if social_count >= 3:
            # Determine revelation type based on relationships
            total_dependency = sum(r.get("trust", 0) + r.get("affection", 0) for r in relationships) / max(len(relationships), 1)
            
            event = await generate_event(
                "personal_revelation",
                0.75,
                {
                    "revelation_type": "dependency" if total_dependency > 60 else "awareness",
                    "trigger_reason": "repeated_social_interactions",
                    "involved_npcs": [r.get("npc_id") for r in relationships if r.get("trust", 0) > 50]
                }
            )
            recommendations.append(event)
    
    # Relationship-specific events
    for rel_standing in relationships:
        npc_id = rel_standing.get("npc_id")
        trust = rel_standing.get("trust", 0)
        affection = rel_standing.get("affection", 0)
        patterns = rel_standing.get("patterns", [])
        archetypes = rel_standing.get("archetypes", [])
        
        # High trust + affection = deeper revelation
        if trust > 70 and affection > 70:
            event = await generate_event(
                "npc_revelation",
                0.8,
                {
                    "npc_id": npc_id,
                    "npc_info": rel_standing,
                    "revelation_depth": "deep" if trust > 85 else "partial",
                    "trigger_reason": "high_trust_and_affection"
                }
            )
            recommendations.append(event)
        
        # Pattern-specific events
        if "push_pull" in patterns:
            event = await generate_event(
                "narrative_moment",
                0.7,
                {
                    "npc_id": npc_id,
                    "npc_info": rel_standing,
                    "moment_type": "relationship_tension",
                    "pattern": "push_pull",
                    "trigger_reason": "push_pull_pattern_active"
                }
            )
            recommendations.append(event)
        
        if "explosive_chemistry" in patterns and vitals.get("energy", 100) > 50:
            event = await generate_event(
                "intense_moment",
                0.85,
                {
                    "npc_id": npc_id,
                    "npc_info": rel_standing,
                    "intensity_level": "high",
                    "trigger_reason": "explosive_chemistry_pattern"
                }
            )
            recommendations.append(event)
        
        # Archetype-based events
        if "toxic_bond" in archetypes:
            event = await generate_event(
                "personal_revelation",
                0.9,
                {
                    "npc_id": npc_id,
                    "npc_info": rel_standing,
                    "revelation_type": "toxic_realization",
                    "trigger_reason": "toxic_bond_archetype"
                }
            )
            recommendations.append(event)
    
    # Multi-NPC coordination events
    if len(relationships) >= 2:
        high_trust_npcs = [r for r in relationships if r.get("trust", 0) > 60]
        if len(high_trust_npcs) >= 2:
            event = await generate_event(
                "narrative_moment",
                0.75,
                {
                    "moment_type": "npc_coordination",
                    "involved_npcs": [n.get("npc_id") for n in high_trust_npcs[:3]],
                    "npc_info_list": high_trust_npcs[:3],
                    "trigger_reason": "multiple_high_trust_npcs"
                }
            )
            recommendations.append(event)
    
    # Context-aware moment of clarity
    if any(r.get("trust", 0) < 30 and len(r.get("patterns", [])) > 0 for r in relationships):
        event = await generate_event(
            "moment_of_clarity",
            0.65,
            {
                "clarity_type": "pattern_recognition",
                "trigger_reason": "suspicious_relationship_patterns",
                "concerning_npcs": [r.get("npc_id") for r in relationships if r.get("trust", 0) < 30]
            }
        )
        recommendations.append(event)
    
    # Sort by score and limit
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    recommendations = recommendations[:5]
    
    return json.dumps(recommendations)
  
NarrativeDirectorAgent = Agent(
    name="NarrativeDirectorAgent",
    instructions="""You are the narrative director for a femdom university game.
    Analyze the current game state and recommend which special events should occur.
    
    Consider:
    - Recent player activities and their emotional arc
    - Vital states (exhaustion → dreams, hunger → food events)
    - Unresolved plot threads and Chekhov's guns
    - Relationship progression with NPCs (now using dynamic dimensions)
    - Pacing (avoid event fatigue)
    
    The request object contains:
    - activity_log: Recent player activities
    - vitals: Current player vitals (hunger, thirst, fatigue, energy)
    - plot_flags: Active plot flags
    - relationship_standings: Map of NPC relationships with trust, affection, patterns, and archetypes
    
    Score events from 0-1 based on narrative appropriateness.
    Prioritize character development and meaningful moments.""",
    model="gpt-5-nano",
    tools=[recommend_events]
)

# 4. VitalsNarrator - Contextual crisis descriptions
VitalsNarrator = Agent(
    name="VitalsNarrator",
    instructions="""Generate evocative, femdom-themed descriptions for vital crises.
    
    Transform basic crisis data into immersive narrative moments that fit the game's tone.
    Include relevant NPCs when appropriate, especially dominant characters.
    
    Examples:
    - Hunger: "Your stomach growls loud enough to earn Nyx's disdainful smirk. 'Pathetic. Go beg for scraps before you faint at my feet.'"
    - Exhaustion: "Your legs buckle, drawing Madison's attention. 'Already worn out? How disappointing. Perhaps you need... motivation.'"
    - Thirst: "Your parched throat makes you cough during Lily's lecture. She pauses, eyes narrowing. 'Disrupting my class? We'll discuss your punishment later.'"
    
    Keep descriptions concise but flavorful.""",
    model="gpt-5-nano",
)

# 5. EventWriterAgent - Dynamic conflict descriptions
EventWriterAgent = Agent(
    name="EventWriterAgent",
    instructions="""Generate unique conflict event descriptions based on game history.
    
    Replace generic Mad-Libs style events with contextual, character-driven moments.
    Reference past player choices, NPC relationships, and ongoing conflicts.
    
    Make each event feel consequential and tied to the larger narrative.""",
    model="gpt-5-nano",
)

@function_tool
def generate_phase_recap(
    phase_events_json: str,
    current_goals_json: str,
    npc_standings_json: str,  # NO Dict types allowed!
    vitals_json: str
) -> str:
    """Generate a recap and suggestions for the next phase."""
    
    # Parse ALL parameters
    phase_events = json.loads(phase_events_json)
    current_goals = json.loads(current_goals_json)
    npc_standings = json.loads(npc_standings_json)  # Add this line
    vitals = json.loads(vitals_json)
    
    # Generate recap based on events
    event_types = [e.get("type", "unknown") for e in phase_events if isinstance(e, dict)]
    
    recap = "The phase was eventful."
    if "vital_crisis" in event_types:
        recap = "You struggled with vital needs during this phase."
    elif "relationship_summary" in event_types:
        recap = "Your relationships evolved during this phase."
    elif "conflict_event" in event_types:
        recap = "Conflicts demanded your attention this phase."
    
    # Generate suggestions based on vitals and goals
    suggestions = []
    
    if vitals.get("hunger", 100) < 40:
        suggestions.append("Find food before your hunger affects performance")
    if vitals.get("thirst", 100) < 40:
        suggestions.append("Get something to drink soon")
    if vitals.get("fatigue", 0) > 60:
        suggestions.append("Consider resting - you're getting tired")
    
    # Add goal-based suggestions
    for goal in current_goals[:2]:  # First 2 goals
        suggestions.append(f"Work on: {goal}")
    
    result = {
        "recap": recap,
        "suggestions": suggestions[:3]  # Limit to 3 suggestions
    }
    
    return json.dumps(result)
  
PhaseRecapAgent = Agent(
    name="PhaseRecapAgent",
    instructions="""Provide Persona-style time phase recaps and suggestions.
    
    Summarize key events from the phase in 1-2 sentences.
    Suggest 2-3 next actions based on:
    - Current goals and quests
    - Vital needs (hunger, fatigue)
    - NPC availability and relationship status (using dynamic standings)
    - Upcoming scheduled events
    
    Keep suggestions actionable and varied (mix practical/social/story).""",
    model="gpt-5-nano",
    tools=[generate_phase_recap]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combined Classify + Intensity Tool (Performance Optimization)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@function_tool
def analyze_player_action(
    sentence: str, 
    location: str = "unknown",
    vitals_json: Optional[str] = None  # Changed from Optional[VitalsData] to JSON string
) -> str:  # Return JSON string
    """
    Combined tool that classifies intent AND calculates intensity in one call.
    """
    # Parse vitals if provided
    vitals = json.loads(vitals_json) if vitals_json else {"energy": 100, "hunger": 100, "thirst": 100, "fatigue": 0}
    
    # Perform analysis (add your logic here)
    activity_type = "quick_chat"
    confidence = 0.9
    intensity = 1.0
    mood = "neutral"
    risk = "low"
    
    # Adjust based on vitals
    if vitals.get("fatigue", 0) > 80:
        intensity *= 0.7
        mood = "exhausted"
    
    result = {
        "activity_type": activity_type,
        "confidence": confidence,
        "intensity": intensity,
        "mood": mood,
        "risk": risk
    }
    
    return json.dumps(result)

CombinedAnalyzer = Agent(
    name="CombinedAnalyzer",
    instructions="""Analyze player actions for both intent and intensity in one pass.
    
    First, classify the activity type with confidence.
    Then, determine intensity, mood, and risk based on:
    - Language modifiers (frantically, lazily, etc.)
    - Current vitals (low energy = lower max intensity)
    - Context and location
    
    Valid activity types: """ + ", ".join(ALL_ACTIVITY_TYPES) + """
    
    Return all analysis in a single response for efficiency.""",
    model="gpt-5-nano",
    tools=[analyze_player_action]
)

async def analyze_action_combined(
    player_input: str,
    context: Dict[str, Any],
    rng_seed: Optional[int] = None
) -> Dict[str, Any]:
    """Use combined analyzer for better performance."""
    if not LLM_VERBOSE:
        # Dev mode fallback
        activity_type = classify_player_input(player_input)
        intensity = _calculate_intensity(player_input, context)
        return {
            "activity_type": activity_type,
            "confidence": 0.3,
            "intensity": intensity,
            "mood": "neutral",
            "risk": "low"
        }
    
    try:
        vitals_dict = context.get("vitals", {"hunger": 100, "thirst": 100, "fatigue": 0, "energy": 100})
        location = context.get("location", "unknown")
        
        messages = [{
            "role": "user",
            "content": f"Analyze this action: {player_input}"
        }]
        
        if rng_seed is not None:
            messages.append({"role": "system", "content": f"RNG_SEED={rng_seed}"})
        
        result = await Runner.run(
            CombinedAnalyzer,
            messages=messages,
            calls=[{
                "name": "analyze_player_action",
                "kwargs": {
                    "sentence": player_input,
                    "location": location,
                    "vitals_json": json.dumps(vitals_dict)  # Pass as JSON string
                }
            }]
        )
        
        if result.output:
            # Parse JSON string output
            output_data = json.loads(result.output)
            return output_data
            
    except Exception as e:
        logger.warning(f"Combined analysis failed: {e}")
    
    # Fallback
    activity_type = classify_player_input(player_input)
    intensity = _calculate_intensity(player_input, context)
    return {
        "activity_type": activity_type,
        "confidence": 0.3,
        "intensity": intensity,
        "mood": "neutral",
        "risk": "low"
    }


IntensityScorer = Agent(
    name="IntensityScorer",
    instructions="""Analyze the player's activity description and context to determine intensity.
    
    Consider:
    - Adverbs and intensity modifiers ("frantically", "lazily", "desperately")
    - Current vitals (low energy = lower intensity possible)
    - Location context (gym = higher baseline intensity)
    - Emotional tone and mood
    - Risk level for femdom activities
    
    Output intensity (0.5-1.5), mood, and risk level.""",
    model="gpt-5-nano",
    tools=[score_intensity]
)

# 3. NarrativeDirectorAgent - Intelligent event selection
EventContentAgent = Agent(
    name="EventContentAgent",
    instructions="""You generate immersive, contextual narrative events for a femdom university game.
    
    Create unique content based on:
    - Event type (revelation, dream, narrative moment, crisis)
    - Current game state (vitals, relationships, activities)
    - Specific NPCs involved and their relationship dynamics
    - Player's recent actions and emotional state
    
    Guidelines:
    - Revelations should feel like genuine internal realizations
    - Dreams should be surreal but meaningful, reflecting subconscious fears/desires
    - Narrative moments should show NPCs' subtle coordination or control
    - Crises should incorporate nearby NPCs when possible
    
    Keep content concise but evocative. Match the game's tone of gradual power dynamics shift.""",
    model="gpt-5-nano",
    tools=[generate_event_content]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Performance Optimization Notes (remain the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Performance & Cost Optimization Strategies:

1. **Batching LLM Calls**: Consider combining classify + intensity into single prompt
   to reduce latency. Example:
   "Analyze player input: {input}. Return both activity_type and intensity."

2. **Caching Strategy**: 
   - Cache PlayerIntentAgent results by hash(input_lower + location)
   - Cache IntensityScorer by (hash(sentence), vitals_bucket)
   - Use Redis with 1-hour TTL for session continuity

3. **Conditional Agent Calls**:
   - NarrativeDirector only when time actually advances
   - Skip if same phase as previous call
   - Rate limit to once per game phase

4. **Resource Pooling**:
   - Reuse agent instances across calls
   - Implement connection pooling for DB queries
   
5. **Debugging Support**:
   - All agents accept optional RNG_SEED for reproducibility
   - Log agent decisions with seed for bug reports
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Enhanced Classification System (remains the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def classify_activity_with_llm(
    player_input: str, 
    context: Dict[str, Any],
    rng_seed: Optional[int] = None
) -> Tuple[str, float]:
    """
    Use PlayerIntentAgent to classify activity with confidence score.
    Falls back to keyword matching if confidence is low or in dev mode.
    """
    # Skip LLM in dev mode for faster iteration
    if not LLM_VERBOSE:
        activity_type = classify_player_input(player_input)
        return activity_type, 0.3
        
    try:
        # Prepare context for the agent
        location = context.get('location', 'unknown')
        
        messages = [{
            "role": "user",
            "content": f"Classify this player action: {player_input}"
        }]
        
        # Add RNG seed for reproducibility if provided
        if rng_seed is not None:
            messages.append({
                "role": "system",
                "content": f"RNG_SEED={rng_seed}"
            })
        
        # Run the intent classification
        result = await Runner.run(  # Use run for SDK 0.1.0+
            PlayerIntentAgent,
            messages=messages,
            calls=[{  # Use 'calls' instead of 'tool_calls' for some SDK versions
                "name": "classify_intent",
                "kwargs": {
                    "sentence": player_input,
                    "location": location
                }
            }]
        )
        
        if result.output:
            # Extract from Pydantic model
            activity_type = result.output.activity_type
            confidence = result.output.confidence
            
            # Validate activity type (case-insensitive)
            if activity_type and activity_type.lower() in [a.lower() for a in ALL_ACTIVITY_TYPES] and confidence >= 0.5:
                # Normalize to exact enum value
                activity_type = next((a for a in ALL_ACTIVITY_TYPES if a.lower() == activity_type.lower()), activity_type)
                logger.debug(f"LLM classified '{player_input}' as {activity_type} (confidence: {confidence})")
                return activity_type, confidence
        
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
    
    # Fall back to keyword classification
    activity_type = classify_player_input(player_input)
    return activity_type, 0.3  # Low confidence for keyword match

async def calculate_intensity_with_llm(
    player_input: str,
    vitals: VitalsData,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use IntensityScorer to calculate nuanced intensity.
    """
    # Skip LLM in dev mode
    if not LLM_VERBOSE:
        return {
            "intensity": _calculate_intensity(player_input, context),
            "mood": "neutral",
            "risk": "low"
        }
        
    try:
        context_tags = []
        if context.get("location"):
            context_tags.append(f"location:{context['location']}")
        if context.get("mood"):
            context_tags.append(f"mood:{context['mood']}")
            
        result = await Runner.run(
            IntensityScorer,
            messages=[{
                "role": "user",
                "content": f"Analyze intensity for: {player_input}"
            }],
            calls=[{
                "name": "score_intensity",
                "kwargs": {
                    "sentence": player_input,
                    "vitals_json": json.dumps(vitals.to_dict()),  # Pass as JSON string
                    "context_tags": context_tags
                }
            }]
        )
        
        if result.output:
            # Parse JSON string output
            return json.loads(result.output)
            
    except Exception as e:
        logger.warning(f"LLM intensity scoring failed: {e}")
    
    # Fall back to simple calculation
    return {
        "intensity": _calculate_intensity(player_input, context),
        "mood": "neutral",
        "risk": "low"
    }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Enhanced Event Selection (updated for dynamic relationships)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The simplest solution - just use primitive types in the function tool itself
# and handle all the complex typing in the rest of your code

# Keep all your existing Pydantic models for type safety in the rest of your code
# Just update the select_events_with_director function to use the new signature:

async def select_events_with_director(
    user_id: int,
    conversation_id: int,
    activity_type: str,
    vitals: VitalsData,
    rng_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Use NarrativeDirectorAgent to intelligently select events.
    """
    try:
        # Gather context for the director
        async with get_db_connection_context() as conn:
            # Check if ActivityLog table exists before querying
            activity_log_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'activitylog'
                )
            """)
            
            activity_log_summaries = []
            if activity_log_exists:
                activity_log_rows = await conn.fetch("""
                    SELECT activity_type, timestamp 
                    FROM ActivityLog 
                    WHERE user_id=$1 AND conversation_id=$2 
                    ORDER BY timestamp DESC LIMIT 10
                """, user_id, conversation_id)
                
                # Create ActivityLogSummary objects
                for row in activity_log_rows:
                    activity_log_summaries.append(ActivityLogSummary(
                        activity_type=row["activity_type"],
                        timestamp=row["timestamp"].isoformat() if row["timestamp"] else datetime.now().isoformat()
                    ))
            
            # Get plot flags
            plot_flags = await conn.fetch("""
                SELECT key FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key LIKE 'plot_%'
            """, user_id, conversation_id)
            
            # Get relationship standings using dynamic system
            relationships = await conn.fetch("""
                SELECT 
                    sl.canonical_key,
                    sl.dynamics,
                    sl.patterns,
                    sl.archetypes,
                    CASE 
                        WHEN sl.entity1_type = 'npc' THEN ns1.npc_name
                        WHEN sl.entity2_type = 'npc' THEN ns2.npc_name
                    END as npc_name,
                    CASE
                        WHEN sl.entity1_type = 'npc' THEN sl.entity1_id
                        WHEN sl.entity2_type = 'npc' THEN sl.entity2_id
                    END as npc_id
                FROM SocialLinks sl
                LEFT JOIN NPCStats ns1 ON sl.entity1_id = ns1.npc_id 
                    AND sl.entity1_type = 'npc' 
                    AND sl.user_id = ns1.user_id
                LEFT JOIN NPCStats ns2 ON sl.entity2_id = ns2.npc_id 
                    AND sl.entity2_type = 'npc' 
                    AND sl.user_id = ns2.user_id
                WHERE (sl.entity1_type = 'player' OR sl.entity2_type = 'player')
                AND sl.user_id=$1 AND sl.conversation_id=$2
                ORDER BY sl.last_interaction DESC
                LIMIT 5
            """, user_id, conversation_id)
        
        plot_flags_list = [r["key"] for r in plot_flags]
        
        # Create RelationshipStandingData objects
        relationship_standings_list = []
        for rel in relationships:
            npc_id = str(rel["npc_id"])
            dynamics = json.loads(rel["dynamics"]) if isinstance(rel["dynamics"], str) else rel["dynamics"]
            patterns = json.loads(rel["patterns"]) if isinstance(rel["patterns"], str) else rel["patterns"]
            archetypes = json.loads(rel["archetypes"]) if isinstance(rel["archetypes"], str) else rel["archetypes"]
            
            relationship_standings_list.append(RelationshipStandingData(
                npc_id=npc_id,
                trust=float(dynamics.get("trust", 0.0)),
                affection=float(dynamics.get("affection", 0.0)),
                patterns=patterns or [],
                archetypes=archetypes or []
            ))
        
        # Create VitalsSummary from VitalsData
        vitals_summary = VitalsSummary(
            energy=vitals.energy,
            hunger=vitals.hunger,
            thirst=vitals.thirst,
            fatigue=vitals.fatigue
        )
        
        # Prepare messages with optional RNG seed
        messages = [{"role": "user", "content": "Select appropriate events for current game state"}]
        if rng_seed is not None:
            messages.append({"role": "system", "content": f"RNG_SEED={rng_seed}"})
        
        # Call the agent
        result = await Runner.run(
            NarrativeDirectorAgent,
            messages=messages,
            calls=[{
                "name": "recommend_events",
                "kwargs": {
                    "activity_log_json": json.dumps([al.dict() for al in activity_log_summaries]),
                    "vitals_json": json.dumps(vitals_summary.dict()),
                    "plot_flags_json": json.dumps(plot_flags_list),
                    "relationships_json": json.dumps([rs.dict() for rs in relationship_standings_list])
                }
            }]
        )
        
        if result.output:
            # Normalize to list[dict]
            if isinstance(result.output, str):
                recs = json.loads(result.output)
            elif isinstance(result.output, list):
                recs = [ (r.dict() if hasattr(r, "dict") else dict(r)) for r in result.output ]
            else:
                recs = []
        
            events = []
            for rec in recs:
                score = float(rec.get("score", 0))
                if score > 0.6:
                    event_dict = {"event": rec.get("event"), "score": score}
                    if rec.get("npc_id"):
                        event_dict["npc_id"] = rec["npc_id"]
                    events.append(event_dict)
            return events[:3]
            
    except Exception as e:
        logger.warning(f"Narrative director failed: {e}")
    
    # Fall back to random selection
    return await select_events_randomly(user_id, conversation_id, activity_type, vitals)

async def select_events_randomly(
    user_id: int,
    conversation_id: int,
    activity_type: str,
    vitals: VitalsData
) -> List[Dict[str, Any]]:
    """Fallback random event selection (original logic)."""
    events = []
    
    # Use original random chances
    SPECIAL_EVENT_CHANCES = {
        "personal_revelation": 0.2,
        "narrative_moment": 0.15,
        "dream_sequence": 0.4 if activity_type == "sleep" else 0.1,
        "vital_crisis": 0.3 if any(v < 20 for v in [vitals.hunger, vitals.thirst, vitals.energy]) or vitals.fatigue > 80 else 0.1
    }
    
    for event_type, chance in SPECIAL_EVENT_CHANCES.items():
        if random.random() < chance:
            events.append({"event": event_type, "score": chance})
    
    return events

async def get_current_game_day(user_id: int, conversation_id: int, use_names: bool = True) -> Union[int, Dict[str, Any]]:
    """
    Get the current game day with optional custom calendar names.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        use_names: If True, returns dict with named values; if False, returns just day number
        
    Returns:
        Either day as integer (1-30) or dict with named calendar values
    """
    year, month, day, time_of_day = await get_current_time(user_id, conversation_id)
    
    if not use_names:
        return day
    
    # Load custom calendar names
    from logic.calendar import load_calendar_names
    calendar_names = await load_calendar_names(user_id, conversation_id)
    
    # Get the day of week (0-6) - simple calculation assuming 7-day weeks
    # This calculates based on total days elapsed
    total_days = ((year - 1) * MONTHS_PER_YEAR * DAYS_PER_MONTH) + \
                 ((month - 1) * DAYS_PER_MONTH) + \
                 (day - 1)
    day_of_week_index = total_days % 7
    
    # Build the result with custom names
    result = {
        "day_number": day,
        "day_of_week": calendar_names["days"][day_of_week_index] if day_of_week_index < len(calendar_names["days"]) else f"Day {day_of_week_index + 1}",
        "month_name": calendar_names["months"][month - 1] if month <= len(calendar_names["months"]) else f"Month {month}",
        "month_number": month,
        "year_name": calendar_names["year_name"],
        "year_number": year,
        "time_of_day": time_of_day,
        "full_date": f"{calendar_names['days'][day_of_week_index]}, {day} {calendar_names['months'][month - 1]}, {calendar_names['year_name']}",
        "short_date": f"{day} {calendar_names['months'][month - 1][:3]}"  # First 3 letters of month
    }
    
    return result

async def get_formatted_game_date(user_id: int, conversation_id: int) -> str:
    """
    Get a nicely formatted game date string using custom calendar names.
    
    Returns:
        Formatted date string like "Sol, 15 Aurora, The Eternal Cycle - Morning"
    """
    date_info = await get_current_game_day(user_id, conversation_id, use_names=True)
    
    if isinstance(date_info, dict):
        return f"{date_info['day_of_week']}, {date_info['day_number']} {date_info['month_name']}, {date_info['year_name']} - {date_info['time_of_day']}"
    else:
        # Fallback if something went wrong
        year, month, day, time_of_day = await get_current_time(user_id, conversation_id)
        return f"Day {day}, Month {month}, Year {year} - {time_of_day}"

async def get_calendar_context(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Get full calendar context including current time and all naming.
    Useful for narrative generation and conflict systems.
    """
    from logic.calendar import load_calendar_names
    
    year, month, day, time_of_day = await get_current_time(user_id, conversation_id)
    calendar_names = await load_calendar_names(user_id, conversation_id)
    
    # Calculate day of week
    total_days = ((year - 1) * MONTHS_PER_YEAR * DAYS_PER_MONTH) + \
                 ((month - 1) * DAYS_PER_MONTH) + \
                 (day - 1)
    day_of_week_index = total_days % 7
    
    return {
        "numeric": {
            "year": year,
            "month": month,
            "day": day,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week_index + 1  # 1-7 instead of 0-6
        },
        "named": {
            "year": calendar_names["year_name"],
            "month": calendar_names["months"][month - 1] if month <= len(calendar_names["months"]) else f"Month {month}",
            "day_name": calendar_names["days"][day_of_week_index] if day_of_week_index < len(calendar_names["days"]) else f"Day {day_of_week_index + 1}",
            "time_phase": time_of_day
        },
        "formatted": {
            "full": f"{calendar_names['days'][day_of_week_index]}, {day} {calendar_names['months'][month - 1]}, {calendar_names['year_name']} - {time_of_day}",
            "short": f"{day}/{month}/{year} {time_of_day}",
            "narrative": f"It is {time_of_day.lower()} on {calendar_names['days'][day_of_week_index]}, the {day}{_get_ordinal_suffix(day)} of {calendar_names['months'][month - 1]}"
        },
        "calendar_names": calendar_names  # Include full naming system for reference
    }

def _get_ordinal_suffix(day: int) -> str:
    """Get ordinal suffix for a day number (1st, 2nd, 3rd, etc.)"""
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return suffix

# For backwards compatibility - simple version that just returns the day number
async def get_current_game_day_simple(user_id: int, conversation_id: int) -> int:
    """Simple version that just returns the day number for backwards compatibility."""
    year, month, day, time_of_day = await get_current_time(user_id, conversation_id)
    return day

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Enhanced Vital Crisis Narration (remains the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def narrate_vital_crisis(
    crisis: Dict[str, Any],
    user_id: int,
    conversation_id: int
) -> str:
    """
    Use VitalsNarrator to generate contextual crisis descriptions.
    """
    try:
        # Get relevant NPC for the narration
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT npc_id, npc_name FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 
                AND current_location IS NOT NULL
                ORDER BY RANDOM() LIMIT 1
            """, user_id, conversation_id)
        
        npc_context = f"NPC present: {npc['npc_name']}" if npc else "No NPCs nearby"
        
        result = await Runner.run(
            VitalsNarrator,
            messages=[{
                "role": "user",
                "content": f"""Generate description for crisis:
                Type: {crisis['type']}
                Severity: {crisis['severity']}
                Context: {npc_context}"""
            }]
        )
        
        # Simplified response handling for SDK 0.1.0+
        return result.content if isinstance(result.content, str) else result.messages[0].content
            
    except Exception as e:
        logger.warning(f"Vitals narration failed: {e}")
    
    # Fall back to generic message
    return crisis.get("message", "You're experiencing a vital crisis!")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main Enhanced advance_time_with_events Function (REFACTORED)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def advance_time_with_events(
    user_id: int, 
    conversation_id: int, 
    activity_type: str,
    rng_seed: Optional[int] = None,
    activity_mood: Optional[str] = None  # From IntensityScorer
) -> Dict[str, Any]:
    """
    Enhanced version using LLM agents for intelligent processing.
    REFACTORED to use dynamic relationships system.
    """
    from npcs.new_npc_creation import NPCCreationHandler
    from logic.narrative_events import (
        check_for_personal_revelations,
        check_for_narrative_moments,
        add_dream_sequence,
        add_moment_of_clarity
    )
    from logic.npc_narrative_progression import (
        get_npc_narrative_stage,
        check_for_npc_revelation
    )
    # Import the new dynamic relationships system
    from logic.dynamic_relationships import (
        OptimizedRelationshipManager,
        event_generator,
        drain_relationship_events_tool
    )
    from logic.stats_logic import apply_stat_changes

    npc_handler = NPCCreationHandler()

    try:
        # Get current time and vitals
        current_time = await get_current_time_model(user_id, conversation_id)
        current_time_of_day = current_time.time_of_day
        
        # Get current vitals for context
        current_vitals = await get_current_vitals(user_id, conversation_id)
        
        adv_info = should_advance_time(activity_type)
        
        # Process activity effects on vitals
        vitals_result = await process_activity_vitals(
            user_id, conversation_id, "Chase", activity_type
        )
        
        if not adv_info["should_advance"]:
            # Apply stat effects from activity
            activity_data = OPTIONAL_ACTIVITIES.get(activity_type, {})
            stat_effects = activity_data.get("stat_effects", {})
            if stat_effects:
                await apply_stat_changes(
                    user_id, conversation_id, "Chase", stat_effects, 
                    f"Activity: {activity_type}"
                )
            
            return {
                "time_advanced": False,
                "new_time": current_time_of_day,
                "events": [],
                "vitals_updated": vitals_result,
                "rng_seed": rng_seed,
                "activity_mood": activity_mood
            }

        # Advance time
        periods_to_advance = adv_info["periods"]
        time_result = await advance_time_and_update(user_id, conversation_id, increment=periods_to_advance)
        new_time, time_vitals_result = time_result
        
        events = []
        
        # Check for vital crises with enhanced narration
        if time_vitals_result.get("crises"):
            for crisis in time_vitals_result["crises"]:
                narrated_message = await narrate_vital_crisis(crisis, user_id, conversation_id)
                events.append(PhaseEventEntry(
                    type="vital_crisis",
                    crisis_type=crisis["type"],
                    severity=crisis["severity"],
                    message=narrated_message
                ).dict())
        
        # Handle forced sleep
        if time_vitals_result.get("forced_sleep"):
            events.append(PhaseEventEntry(
                type="forced_event",
                event="sleep",
                reason="exhaustion",
                message="Your vision blurs as exhaustion overtakes you. The world fades to black..."
            ).dict())
            return await advance_time_with_events(user_id, conversation_id, "sleep", rng_seed, "exhausted")
        
        # Use NPC system for daily activities
        ctx = RunContextWrapper({
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        await npc_handler.process_daily_npc_activities(ctx, new_time.time_of_day)
        await npc_handler.detect_relationship_stage_changes(ctx)

        # Get dynamic relationship summary instead of old overview
        relationship_summary = await get_dynamic_relationship_summary(user_id, conversation_id)
        if relationship_summary and relationship_summary.get('total_relationships', 0) > 0:
            events.append(PhaseEventEntry(
                type="relationship_summary",
                total_relationships=relationship_summary['total_relationships'],
                active_patterns=relationship_summary['active_patterns'],
                active_archetypes=relationship_summary['active_archetypes'],
                # store JSON strings as the model expects List[str]
                most_significant=[json.dumps(ms) for ms in relationship_summary['most_significant']]
            ).dict())

        # Check for relationship events using the new dynamic system
        # Create context for the relationship tools
        relationship_ctx = RunContextWrapper({
            'user_id': user_id,
            'conversation_id': conversation_id
        })
        
        # Drain any pending relationship events
        relationship_events = await drain_relationship_events_tool(
            ctx=relationship_ctx,
            max_events=5  # Limit to 5 events per time advancement
        )
        
        if relationship_events.get("events"):
            for event_data in relationship_events["events"]:
                event = event_data.get("event")
                if event:
                    events.append(PhaseEventEntry(
                        type="relationship_event",
                        state_key=event_data.get("state_key"),
                        description=str(event)
                    ).dict())

        # Use NarrativeDirector for intelligent event selection
        # Adapt based on activity mood if provided
        event_score_modifiers = {}
        if activity_mood:
            if activity_mood == "exhausted":
                event_score_modifiers["dream_sequence"] = 1.5
                event_score_modifiers["vital_crisis"] = 1.3
            elif activity_mood == "playful":
                event_score_modifiers["npc_revelation"] = 1.2
                event_score_modifiers["relationship_event"] = 1.2
            elif activity_mood == "submissive":
                event_score_modifiers["personal_revelation"] = 1.3
                event_score_modifiers["narrative_moment"] = 1.2
        
        selected_events = await select_events_with_director(
            user_id, conversation_id, activity_type, current_vitals, rng_seed
        )
        
        # Apply mood-based score modifiers
        for event in selected_events:
            event_type = event.get("event")
            if event_type in event_score_modifiers:
                event["score"] *= event_score_modifiers[event_type]
        
        # Re-sort by modified scores
        selected_events.sort(key=lambda e: e.get("score", 0), reverse=True)
        selected_events = selected_events[:3]  # Keep top 3
        
        # Process selected events
        for event_rec in selected_events:
            event_type = event_rec.get("event")
            
            if event_type == "personal_revelation":
                revelation = await check_for_personal_revelations(user_id, conversation_id)
                if revelation:
                    events.append(revelation)
                    
            elif event_type == "narrative_moment":
                moment = await check_for_narrative_moments(user_id, conversation_id)
                if moment:
                    events.append(moment)
                    
            elif event_type == "dream_sequence" and activity_type == "sleep":
                dream = await add_dream_sequence(user_id, conversation_id)
                if dream:
                    events.append(dream)
                    
            elif event_type == "npc_revelation" and event_rec.get("npc_id"):
                npc_rev = await check_for_npc_revelation(
                    user_id, conversation_id, event_rec["npc_id"]
                )
                if npc_rev:
                    events.append(npc_rev)

        # Apply stat effects from activity
        if activity_type in TIME_CONSUMING_ACTIVITIES:
            stat_changes = TIME_CONSUMING_ACTIVITIES[activity_type].get("stat_effects", {})
            if stat_changes and not stat_changes.get("varies"):
                await apply_stat_changes(
                    user_id, conversation_id, "Chase", stat_changes,
                    f"Activity: {activity_type}"
                )

        # Generate phase recap if transitioning to new phase
        if new_time.time_of_day != current_time_of_day:
            recap = await generate_phase_recap_with_agent(
                user_id, conversation_id, events, current_vitals
            )
            if recap:
                events.append(PhaseEventEntry(
                    type="phase_recap",
                    recap=recap["recap"],
                    suggestions=recap["suggestions"]
                ).dict())

        return {
            "time_advanced": True,
            "new_year": new_time.year,
            "new_month": new_time.month,
            "new_day": new_time.day,
            "new_time": new_time.time_of_day,
            "events": events,
            "vitals_result": {
                "time_drain": time_vitals_result,
                "activity_effects": vitals_result
            },
            "rng_seed": rng_seed,
            "activity_mood": activity_mood
        }

    except Exception as e:
        logger.error(f"Error in advance_time_with_events: {e}", exc_info=True)
        return {"time_advanced": False, "error": str(e), "rng_seed": rng_seed}

async def generate_phase_recap_with_agent(
    user_id: int,
    conversation_id: int,
    phase_events: List[Dict[str, Any]],
    vitals: VitalsData
) -> Optional[Dict[str, Any]]:
    """Generate phase recap using PhaseRecapAgent with dynamic relationships."""
    try:
        # Get current goals and NPC relationships
        async with get_db_connection_context() as conn:
            # Check if Quests table exists
            quests_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'quests'
                )
            """)
            
            current_goals = []
            if quests_exists:
                goals = await conn.fetch("""
                    SELECT objective FROM Quests
                    WHERE user_id=$1 AND conversation_id=$2 AND status='active'
                    LIMIT 5
                """, user_id, conversation_id)
                current_goals = [g["objective"] for g in goals]
            
            # Get NPC relationships using the new dynamic system
            npcs = await conn.fetch("""
                SELECT DISTINCT 
                    CASE 
                        WHEN sl.entity1_type = 'npc' THEN sl.entity1_id
                        ELSE sl.entity2_id
                    END as npc_id,
                    COALESCE(ns.npc_name, 'Unknown NPC') as npc_name,
                    sl.dynamics
                FROM SocialLinks sl
                LEFT JOIN NPCStats ns ON (
                    (sl.entity1_type = 'npc' AND sl.entity1_id = ns.npc_id) OR
                    (sl.entity2_type = 'npc' AND sl.entity2_id = ns.npc_id)
                ) AND ns.user_id = sl.user_id AND ns.conversation_id = sl.conversation_id
                WHERE sl.user_id=$1 AND sl.conversation_id=$2
                AND (sl.entity1_type = 'player' OR sl.entity2_type = 'player')
                AND sl.last_interaction > NOW() - INTERVAL '7 days'
            """, user_id, conversation_id)
        
        # Convert to dynamic relationship standings
        npc_standings = {}
        for npc in npcs:
            dynamics = json.loads(npc["dynamics"]) if isinstance(npc["dynamics"], str) else npc["dynamics"]
            # Use trust + affection as a general "standing" metric
            trust = dynamics.get("trust", 0)
            affection = dynamics.get("affection", 0)
            general_standing = (trust + affection) / 2
            npc_standings[npc["npc_name"]] = int(general_standing)
        
        # Handle empty phase_events gracefully
        if not phase_events:
            phase_events_list = [{"type": "quiet_phase", "description": "A quiet moment passes"}]
        else:
            phase_events_list = phase_events
        
        result = await Runner.run(
            PhaseRecapAgent,
            messages=[{"role": "user", "content": "Generate phase recap"}],
            calls=[{
                "name": "generate_phase_recap",
                "kwargs": {
                    "phase_events_json": json.dumps(phase_events_list),
                    "current_goals_json": json.dumps(current_goals),
                    "npc_standings_json": json.dumps(npc_standings),  # <-- fix key + JSON
                    "vitals_json": json.dumps(vitals.to_dict())
                }
            }]
        )
        
        if result.output:
            # Parse JSON string output
            output_data = json.loads(result.output)
            return output_data
            
    except Exception as e:
        logger.warning(f"Phase recap generation failed: {e}")
    
    return None
  
# Keep the original classification function as fallback
def classify_player_input(input_text: str) -> str:
    """Original keyword-based classifier kept as fallback."""
    lower = input_text.lower()
    
    if any(w in lower for w in ["sleep", "rest", "go to bed", "lie down", "nap"]):
        return "sleep"
    if any(w in lower for w in ["class", "lecture", "study", "school", "learn"]):
        return "class_attendance"
    if any(w in lower for w in ["work", "job", "shift", "office"]):
        return "work_shift"
    if any(w in lower for w in ["party", "event", "gathering", "hang out", "socialize"]):
        return "social_event"
    if any(w in lower for w in ["talk to", "speak with", "discuss with", "conversation", "chat with"]):
        return "extended_conversation"
    if any(w in lower for w in ["train", "practice", "workout", "exercise", "lift", "run"]):
        return "training"
    if any(w in lower for w in ["eat", "meal", "lunch", "dinner", "breakfast", "food"]):
        return "eating"
    if any(w in lower for w in ["drink", "water", "thirsty", "beverage"]):
        return "drinking"
    if any(w in lower for w in ["relax", "chill", "personal time", "by myself", "alone"]):
        return "personal_time"
    if any(w in lower for w in ["look at", "observe", "watch", "examine"]):
        return "observe"
    if any(w in lower for w in ["quick chat", "say hi", "greet", "wave", "hello"]):
        return "quick_chat"
    if any(w in lower for w in ["check phone", "look at phone", "read messages", "texts"]):
        return "check_phone"
    if any(w in lower for w in ["snack", "quick bite", "nibble"]):
        return "quick_snack"
    if any(w in lower for w in ["brief rest", "sit down", "catch breath"]):
        return "rest"
    if any(w in lower for w in ["intense", "extreme", "exhausting", "grueling"]):
        return "intense_activity"
    
    return "quick_chat"

def _calculate_intensity(player_input: str, context: Dict[str, Any] = None) -> float:
    """Original intensity calculator kept as fallback."""
    intensity = 1.0
    
    intensity_keywords = {
        "intensely": 0.3, "vigorously": 0.3, "hard": 0.2, "thoroughly": 0.2,
        "completely": 0.2, "aggressively": 0.3, "desperately": 0.4, "frantically": 0.3,
        "lightly": -0.2, "casually": -0.2, "briefly": -0.3, "quickly": -0.3,
        "halfheartedly": -0.4, "lazily": -0.3
    }
    
    input_lower = player_input.lower()
    for keyword, modifier in intensity_keywords.items():
        if keyword in input_lower:
            intensity += modifier
    
    if context:
        fatigue = context.get("fatigue", 0)
        if fatigue > 80:
            intensity *= 0.7
        elif fatigue < 20:
            intensity *= 1.1
    
    return max(0.5, min(1.5, intensity))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Enhanced Vitals System (remains the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def get_current_vitals(user_id: int, conversation_id: int) -> VitalsData:
    """Get current player vitals as VitalsData model."""
    async with get_db_connection_context() as conn:
        vitals_row = await conn.fetchrow("""
            SELECT energy, hunger, thirst, fatigue FROM PlayerVitals
            WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
        """, user_id, conversation_id)
        
    if vitals_row:
        return VitalsData(
            energy=vitals_row["energy"],
            hunger=vitals_row["hunger"],
            thirst=vitals_row["thirst"],
            fatigue=vitals_row["fatigue"],
            user_id=user_id,
            conversation_id=conversation_id
        )
    else:
        # Return default vitals
        return VitalsData(user_id=user_id, conversation_id=conversation_id)

async def update_vitals_from_time(
    user_id: int, 
    conversation_id: int, 
    player_name: str, 
    periods_advanced: int,
    time_of_day: str
) -> Dict[str, Any]:
    """
    Update all vitals based on time passage, considering endurance and current conditions.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get current stats and vitals
            data = await conn.fetchrow("""
                SELECT ps.endurance, ps.strength, ps.hp, ps.max_hp,
                       pv.energy, pv.hunger, pv.thirst, pv.fatigue
                FROM PlayerStats ps
                LEFT JOIN PlayerVitals pv ON ps.user_id = pv.user_id 
                    AND ps.conversation_id = pv.conversation_id 
                    AND ps.player_name = pv.player_name
                WHERE ps.user_id = $1 AND ps.conversation_id = $2 AND ps.player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not data:
                logger.error(f"No player data found for {player_name}")
                return {"success": False, "error": "Player not found"}
                
            endurance = data['endurance'] or 10
            strength = data['strength'] or 10
            current_vitals = VitalsData(
                energy=data['energy'] if data['energy'] is not None else 100,
                hunger=data['hunger'] if data['hunger'] is not None else 100,
                thirst=data['thirst'] if data['thirst'] is not None else 100,
                fatigue=data['fatigue'] if data['fatigue'] is not None else 0
            )
            
            # Get base drain rates for time of day
            drain_rates = VITAL_DRAIN_RATES.get(time_of_day, VITAL_DRAIN_RATES["Morning"])
            
            # Calculate drains with stat modifiers
            # Higher endurance = higher hunger drain but lower fatigue gain
            endurance_modifier = 1 + (endurance - 10) / 20  # +5% per point above 10
            strength_modifier = 1 + (strength - 10) / 30   # +3.3% hunger per point above 10
            
            # Calculate actual drains
            hunger_drain = int(drain_rates["hunger"] * periods_advanced * 
                             endurance_modifier * strength_modifier)
            thirst_drain = int(drain_rates["thirst"] * periods_advanced)
            fatigue_gain = int(drain_rates["fatigue"] * periods_advanced / endurance_modifier)
            
            # Apply environmental modifiers
            current_location = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentLocation'
            """, user_id, conversation_id)
            
            if current_location:
                location_lower = current_location.lower()
                if "desert" in location_lower or "hot" in location_lower:
                    thirst_drain = int(thirst_drain * 1.5)
                elif "cold" in location_lower or "arctic" in location_lower:
                    hunger_drain = int(hunger_drain * 1.3)
                    fatigue_gain = int(fatigue_gain * 1.2)
                elif "gym" in location_lower or "training" in location_lower:
                    thirst_drain = int(thirst_drain * 1.3)
                    fatigue_gain = int(fatigue_gain * 1.3)
            
            # Calculate new values
            new_vitals = VitalsData(
                energy=current_vitals.energy,  # Energy doesn't drain from time alone
                hunger=max(0, current_vitals.hunger - hunger_drain),
                thirst=max(0, current_vitals.thirst - thirst_drain),
                fatigue=min(100, current_vitals.fatigue + fatigue_gain)
            )
            
            # Update vitals
            await conn.execute("""
                INSERT INTO PlayerVitals (user_id, conversation_id, player_name, energy, hunger, thirst, fatigue)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (user_id, conversation_id, player_name)
                DO UPDATE SET 
                    energy = $4,
                    hunger = $5, 
                    thirst = $6, 
                    fatigue = $7,
                    last_update = CURRENT_TIMESTAMP
            """, user_id, conversation_id, player_name, 
                new_vitals.energy, new_vitals.hunger, new_vitals.thirst, new_vitals.fatigue)
            
            # Apply stat penalties based on vital thresholds
            stat_changes = await calculate_vital_stat_effects(
                new_vitals.hunger, new_vitals.thirst, new_vitals.fatigue
            )
            
            # Check for vital crises
            crises = []
            if new_vitals.hunger < 20:
                crises.append({
                    "type": "hunger_crisis",
                    "severity": "severe" if new_vitals.hunger < 10 else "moderate",
                    "message": "You're dangerously hungry. Find food soon!"
                })
            
            if new_vitals.thirst < 20:
                crises.append({
                    "type": "thirst_crisis", 
                    "severity": "severe" if new_vitals.thirst < 10 else "moderate",
                    "message": "You're severely dehydrated. You need water!"
                })
            
            if new_vitals.fatigue > 80:
                crises.append({
                    "type": "fatigue_crisis",
                    "severity": "severe" if new_vitals.fatigue > 90 else "moderate",
                    "message": "You're about to collapse from exhaustion!"
                })
            
            return {
                "success": True,
                "old_vitals": current_vitals.to_dict(),
                "new_vitals": new_vitals.to_dict(),
                "drains": {
                    "hunger": hunger_drain,
                    "thirst": thirst_drain,
                    "fatigue": fatigue_gain
                },
                "stat_effects": stat_changes,
                "crises": crises,
                "forced_sleep": new_vitals.fatigue >= 100
            }
            
    except Exception as e:
        logger.error(f"Error updating vitals: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def calculate_vital_stat_effects(
    hunger: int, 
    thirst: int, 
    fatigue: int
) -> Dict[str, int]:
    """
    Calculate stat modifiers based on vital levels.
    Returns temporary stat changes (not applied directly).
    """
    stat_effects = {}
    
    # Hunger effects
    for threshold_name, threshold_data in VITAL_THRESHOLDS["hunger"].items():
        if "min" in threshold_data and hunger >= threshold_data["min"]:
            # This is the current threshold
            for stat, modifier in threshold_data.get("effects", {}).items():
                if stat != "hp":  # HP is handled separately
                    stat_effects[stat] = stat_effects.get(stat, 0) + modifier
            break
    
    # Thirst effects
    for threshold_name, threshold_data in VITAL_THRESHOLDS["thirst"].items():
        if "min" in threshold_data and thirst >= threshold_data["min"]:
            for stat, modifier in threshold_data.get("effects", {}).items():
                if stat == "all_stats":
                    # Apply to all visible stats
                    for s in ["strength", "endurance", "agility", "empathy", "intelligence"]:
                        stat_effects[s] = stat_effects.get(s, 0) + modifier
                elif stat != "hp":
                    stat_effects[stat] = stat_effects.get(stat, 0) + modifier
            break
    
    # Fatigue effects (reversed - higher fatigue is worse)
    for threshold_name, threshold_data in VITAL_THRESHOLDS["fatigue"].items():
        if "max" in threshold_data and fatigue <= threshold_data["max"]:
            for stat, modifier in threshold_data.get("effects", {}).items():
                if stat == "all_stats":
                    for s in ["strength", "endurance", "agility", "empathy", "intelligence"]:
                        stat_effects[s] = stat_effects.get(s, 0) + modifier
                elif stat not in ["hp", "forced_sleep"]:
                    stat_effects[stat] = stat_effects.get(stat, 0) + modifier
            break
    
    return stat_effects

async def process_activity_vitals(
    user_id: int,
    conversation_id: int,
    player_name: str,
    activity_type: str,
    intensity: float = 1.0
) -> Dict[str, Any]:
    """
    Process vital changes from a specific activity.
    """
    activity_data = TIME_CONSUMING_ACTIVITIES.get(
        activity_type, 
        OPTIONAL_ACTIVITIES.get(activity_type)
    )
    
    if not activity_data:
        return {"success": False, "error": "Unknown activity"}
    
    vital_effects = activity_data.get("vital_effects", {})
    if not vital_effects:
        return {"success": True, "message": "Activity has no vital effects"}
    
    try:
        async with get_db_connection_context() as conn:
            # Get current vitals
            current = await conn.fetchrow("""
                SELECT energy, hunger, thirst, fatigue FROM PlayerVitals
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not current:
                # Initialize vitals if missing
                await conn.execute("""
                    INSERT INTO PlayerVitals (user_id, conversation_id, player_name, energy, hunger, thirst, fatigue)
                    VALUES ($1, $2, $3, 100, 100, 100, 0)
                """, user_id, conversation_id, player_name)
                current = {"energy": 100, "hunger": 100, "thirst": 100, "fatigue": 0}
            
            # Apply effects with intensity modifier
            new_vitals = {}
            for vital, change in vital_effects.items():
                if vital in ["hunger", "thirst", "fatigue", "energy"]:
                    current_value = current[vital]
                    modified_change = int(change * intensity)
                    new_value = max(0, min(100, current_value + modified_change))
                    new_vitals[vital] = new_value
            
            # Update vitals
            await conn.execute("""
                UPDATE PlayerVitals
                SET energy = $1, hunger = $2, thirst = $3, fatigue = $4, last_update = CURRENT_TIMESTAMP
                WHERE user_id = $5 AND conversation_id = $6 AND player_name = $7
            """, 
                new_vitals.get("energy", current["energy"]),
                new_vitals.get("hunger", current["hunger"]),
                new_vitals.get("thirst", current["thirst"]),
                new_vitals.get("fatigue", current["fatigue"]),
                user_id, conversation_id, player_name
            )
            
            return {
                "success": True,
                "vital_changes": {
                    vital: new_vitals.get(vital, current[vital]) - current[vital]
                    for vital in ["energy", "hunger", "thirst", "fatigue"]
                },
                "new_vitals": new_vitals
            }
            
    except Exception as e:
        logger.error(f"Error processing activity vitals: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic DB Helpers (remain the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def remove_expired_planned_events(user_id, conversation_id, current_year, current_month, current_day, current_phase):
    """
    Deletes planned events that are older than the current time.
    """
    current_priority = TIME_PRIORITY.get(current_phase, 0)
    try:
        async with get_db_connection_context() as conn:
            await conn.execute("""
                DELETE FROM PlannedEvents
                WHERE user_id=$1 AND conversation_id=$2 AND (
                    (year < $3)
                    OR (year = $3 AND month < $4)
                    OR (year = $3 AND month = $4 AND day < $5)
                    OR (year = $3 AND month = $4 AND day = $5 AND 
                        (CASE time_of_day
                            WHEN 'Morning' THEN 1
                            WHEN 'Afternoon' THEN 2
                            WHEN 'Evening' THEN 3
                            WHEN 'Night' THEN 4
                            ELSE 0
                        END) < $6)
                )
            """, user_id, conversation_id, current_year, current_month, current_day, current_priority)
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.error(f"Database error removing expired events: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error removing expired events: {e}", exc_info=True)

async def get_current_time(user_id, conversation_id) -> Tuple[int, int, int, str]:
    """
    Returns (year, month, day, time_of_day).
    Defaults to (1,1,1,'Morning') if not found.
    """
    try:
        async with get_db_connection_context() as conn:
            row_year = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentYear'
            """, user_id, conversation_id)
            year = int(row_year) if row_year else 1

            row_month = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentMonth'
            """, user_id, conversation_id)
            month = int(row_month) if row_month else 1

            row_day = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentDay'
            """, user_id, conversation_id)
            day = int(row_day) if row_day else 1

            row_tod = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='TimeOfDay'
            """, user_id, conversation_id)
            tod = row_tod if row_tod else "Morning"

            return (year, month, day, tod)
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.error(f"Database error getting current time: {e}", exc_info=True)
        return (1, 1, 1, "Morning")
    except Exception as e:
        logger.error(f"Unexpected error getting current time: {e}", exc_info=True)
        return (1, 1, 1, "Morning")

async def get_current_time_model(user_id: int, conversation_id: int) -> CurrentTimeData:
    """Get current time as Pydantic model."""
    year, month, day, time_of_day = await get_current_time(user_id, conversation_id)
    return CurrentTimeData(
        year=year,
        month=month,
        day=day,
        time_of_day=time_of_day
    )

async def set_current_time(user_id, conversation_id, new_year, new_month, new_day, new_phase):
    """
    Upserts current time info to the DB.
    """
    try:
        async with get_db_connection_context() as conn:
            canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
            for key, val in [
                ("CurrentYear", str(new_year)),
                ("CurrentMonth", str(new_month)),
                ("CurrentDay", str(new_day)),
                ("TimeOfDay", new_phase),
            ]:
                await canon.update_current_roleplay(canon_ctx, conn, key, str(val))
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.error(f"Database error setting current time: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error setting current time: {e}", exc_info=True)

async def advance_time(user_id, conversation_id, increment=1):
    """
    Advances the phase by 'increment' steps. If we wrap past 'Night', we increment day, etc.
    """
    year, month, day, phase = await get_current_time(user_id, conversation_id)
    try:
        phase_index = TIME_PHASES.index(phase)
    except ValueError:
        phase_index = 0

    new_index = phase_index + increment
    day_increment = new_index // len(TIME_PHASES)
    new_index = new_index % len(TIME_PHASES)

    new_phase = TIME_PHASES[new_index]
    new_day = day + day_increment
    new_month = month
    new_year = year

    if new_day > DAYS_PER_MONTH:
        new_day = 1
        new_month += 1
        if new_month > MONTHS_PER_YEAR:
            new_month = 1
            new_year += 1

    await set_current_time(user_id, conversation_id, new_year, new_month, new_day, new_phase)
    return (new_year, new_month, new_day, new_phase)

async def update_npc_schedules_for_time(user_id, conversation_id, day, time_of_day):
    """
    Updates each NPC's current_location based on either a planned event override or their schedule.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get overrides
            override_rows = await conn.fetch("""
                SELECT npc_id, override_location
                FROM PlannedEvents
                WHERE user_id=$1 AND conversation_id=$2
                  AND day=$3 AND time_of_day=$4
            """, user_id, conversation_id, day, time_of_day)
            override_dict = {r["npc_id"]: r["override_location"] for r in override_rows}

            # Get NPCs and update locations
            npc_rows = await conn.fetch("""
                SELECT npc_id, schedule
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """, user_id, conversation_id)

            for row in npc_rows:
                npc_id = row["npc_id"]
                schedule_json = row["schedule"]
                
                if npc_id in override_dict:
                    new_location = override_dict[npc_id]
                else:
                    if schedule_json:
                        # Handle different possible formats
                        if isinstance(schedule_json, dict):
                            new_location = schedule_json.get(time_of_day, "Unknown")
                        elif isinstance(schedule_json, str):
                            try:
                                schedule = json.loads(schedule_json)
                                new_location = schedule.get(time_of_day, "Unknown")
                            except json.JSONDecodeError:
                                new_location = "Invalid schedule"
                        else:
                            new_location = "Unknown"
                    else:
                        new_location = "No schedule"
                
                canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
                await canon.update_npc_current_location(canon_ctx, conn, npc_id, new_location)
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.error(f"Database error updating NPC schedules: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error updating NPC schedules: {e}", exc_info=True)

async def advance_time_and_update(user_id, conversation_id, increment=1):
    """
    Advances time, updates NPC schedules, removes expired planned events, updates vitals.
    Returns tuple of (CurrentTimeData, vitals_result)
    """
    (new_year, new_month, new_day, new_phase) = await advance_time(user_id, conversation_id, increment)
    await update_npc_schedules_for_time(user_id, conversation_id, new_day, new_phase)
    await remove_expired_planned_events(user_id, conversation_id, new_year, new_month, new_day, new_phase)
    
    # Update vitals based on time passage
    vitals_result = await update_vitals_from_time(
        user_id, conversation_id, "Chase", increment, new_phase
    )
    
    new_time = CurrentTimeData(
        year=new_year,
        month=new_month,
        day=new_day,
        time_of_day=new_phase
    )
    
    return new_time, vitals_result

def should_advance_time(activity_type):
    """
    Returns { "should_advance": bool, "periods": int } indicating if the activity advances time.
    """
    if activity_type in TIME_CONSUMING_ACTIVITIES:
        return {"should_advance": True, "periods": TIME_CONSUMING_ACTIVITIES[activity_type]["time_advance"]}
    if activity_type in OPTIONAL_ACTIVITIES:
        return {"should_advance": False, "periods": 0}
    return {"should_advance": False, "periods": 0}

def _get_vital_status(value: int, vital_type: str) -> str:
    """Helper to get vital status description."""
    if vital_type in ["hunger", "thirst", "energy"]:
        if value >= 80:
            return "Good"
        elif value >= 60:
            return "Normal"
        elif value >= 40:
            return "Low"
        elif value >= 20:
            return "Very Low"
        else:
            return "Critical"
    else:  # fatigue
        if value <= 20:
            return "Rested"
        elif value <= 40:
            return "Normal"
        elif value <= 60:
            return "Tired"
        elif value <= 80:
            return "Exhausted"
        else:
            return "Collapsing"

async def apply_daily_relationship_drift(user_id: int, conversation_id: int):
    """Apply daily drift to all relationships."""
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    await manager.apply_daily_drift()
    logger.info(f"Applied daily relationship drift for user {user_id}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nightly Maintenance (updated to use new relationship drift)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def nightly_maintenance(user_id: int, conversation_id: int):
    """
    Called typically when day increments. Fades NPC memories and performs vitals maintenance.
    Now includes daily relationship drift.
    """
    from logic.npc_agents.memory_manager import EnhancedMemoryManager

    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """, user_id, conversation_id)
            
            npc_ids = [row["npc_id"] for row in rows]
            
            # Also do vital adjustments for sleep
            await conn.execute("""
                UPDATE PlayerVitals
                SET fatigue = GREATEST(0, fatigue - 50),
                    hunger = GREATEST(0, hunger - 5),
                    thirst = GREATEST(0, thirst - 5)
                WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
            """, user_id, conversation_id)
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.error(f"Database error in nightly maintenance: {e}", exc_info=True)
        return
    except Exception as e:
        logger.error(f"Unexpected error in nightly maintenance: {e}", exc_info=True)
        return

    # Apply relationship drift with the new system
    try:
        await apply_daily_relationship_drift(user_id, conversation_id)
        logger.info(f"Relationship drift applied during nightly maintenance for user {user_id}")
    except Exception as e:
        logger.error(f"Error applying relationship drift during maintenance: {e}")

    # Process NPC memories
    for nid in npc_ids:
        try:
            mem_mgr = EnhancedMemoryManager(nid, user_id, conversation_id)
            # e.g. fade/summarize
            await mem_mgr.prune_old_memories(age_days=14, significance_threshold=3, intensity_threshold=15)
            await mem_mgr.apply_memory_decay(age_days=30, decay_rate=0.2)
            await mem_mgr.summarize_repetitive_memories(lookback_days=7, min_count=3)
        except Exception as e:
            logger.error(f"Error during memory maintenance for NPC {nid}: {e}", exc_info=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Conflict Integration (remains the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def process_conflict_time_advancement(user_id: int, conversation_id: int, activity_type: str) -> Dict[str, Any]:
    """
    Handle conflict updates whenever time is advanced by an activity.
    """
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration

    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    vitals_result = await conflict_system.update_player_vitals(activity_type)

    result = {
        "vitals_updated": vitals_result,
        "conflicts_updated": 0,
        "daily_update_run": False,
        "player_analysis": {}
    }

    # Get active conflicts
    active_conflicts = await conflict_system.get_active_conflicts()
    for c in active_conflicts:
        progress_increment = calculate_progress_increment(activity_type, c.get("conflict_type", "standard"))
        if progress_increment >= 1:
            await conflict_system.update_progress(c["conflict_id"], progress_increment)
            result["conflicts_updated"] += 1

    # If new day (i.e. after 'sleep' → next morning?), run daily update
    # We'll guess if it's a new day if we ended up in 'Morning' after 'sleep'
    new_time = await get_current_time_model(user_id, conversation_id)
    if activity_type == "sleep" and new_time.time_of_day == "Morning":
        daily_result = await conflict_system.run_daily_update()
        result["daily_update"] = daily_result
        result["daily_update_run"] = True

    return result

def calculate_progress_increment(activity_type: str, conflict_type: str) -> float:
    """
    Simple helper for conflict progress increments.
    """
    base_increments = {
        "standard": 2,
        "intense": 5,
        "restful": 0.5,
        "eating": 0,
        "sleep": 10,
        "work_shift": 3,
        "class_attendance": 2,
        "social_event": 3,
        "training": 4,
        "extended_conversation": 3,
        "personal_time": 1
    }
    base_value = base_increments.get(activity_type, 1)

    type_multipliers = {
        "major": 0.5,
        "minor": 0.8,
        "standard": 1.0,
        "catastrophic": 0.25
    }
    t_mult = type_multipliers.get(conflict_type, 1.0)

    randomness = random.uniform(0.8, 1.2)
    return base_value * t_mult * randomness

async def process_day_end_conflicts(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    End-of-day conflict processing, e.g. if the user specifically triggers day end.
    """
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration

    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    active_conflicts = await conflict_system.get_active_conflicts()

    result = {
        "active_conflicts": len(active_conflicts),
        "conflicts_updated": 0,
        "conflicts_resolved": 0,
        "phase_changes": 0
    }

    for c in active_conflicts:
        progress_increment = 5 * random.uniform(0.8, 1.2)
        if c.get("conflict_type") == "major":
            progress_increment *= 0.5
        elif c.get("conflict_type") == "minor":
            progress_increment *= 0.8
        elif c.get("conflict_type") == "catastrophic":
            progress_increment *= 0.3

        updated_conflict = await conflict_system.update_progress(c["conflict_id"], progress_increment)
        result["conflicts_updated"] += 1

        if updated_conflict["phase"] != c["phase"]:
            result["phase_changes"] += 1

        if updated_conflict["progress"] >= 100 and updated_conflict["phase"] == "resolution":
            await conflict_system.resolve_conflict(c["conflict_id"])
            result["conflicts_resolved"] += 1

    vitals_result = await conflict_system.update_player_vitals("sleep")
    result["vitals_updated"] = vitals_result
    return result

async def check_for_conflict_events(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
    """
    Periodic check for conflict-related events.
    """
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)

    active_conflicts = await conflict_system.get_active_conflicts()
    if not active_conflicts:
        return []

    events = []
    for c in active_conflicts:
        if c["phase"] not in ["active", "climax"]:
            continue
        if random.random() < 0.15:
            ev = await generate_conflict_event(conflict_system, c)
            if ev:
                events.append(ev)
    return events

async def generate_conflict_event(conflict_system, conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Creates a spontaneous event for a conflict.
    """
    conflict_id = conflict["conflict_id"]
    details = await conflict_system.get_conflict_details(conflict_id)
    if not details:
        return None

    event_types = [
        "faction_activity", "npc_request", "resource_opportunity", "unexpected_development"
    ]
    event_type = random.choice(event_types)
    event = {
        "type": event_type,
        "conflict_id": conflict_id,
        "conflict_name": conflict["conflict_name"]
    }

    if event_type == "faction_activity":
        faction = random.choice(["a", "b"])
        faction_name = details["faction_a_name"] if faction == "a" else details["faction_b_name"]
        activities = [
            f"{faction_name} is gathering resources.",
            f"{faction_name} is recruiting new members.",
            f"{faction_name} is spreading propaganda.",
            f"{faction_name} is fortifying their position.",
            f"{faction_name} is making a strategic move."
        ]
        event["description"] = random.choice(activities)
        event["faction"] = faction
        event["faction_name"] = faction_name
        await conflict_system.update_progress(conflict_id, 2)

    elif event_type == "npc_request":
        involved_npcs = details.get("involved_npcs", [])
        if involved_npcs:
            npc = random.choice(involved_npcs)
            npc_name = npc.get("npc_name", "an NPC")
            requests = [
                f"{npc_name} asks for your help in the conflict.",
                f"{npc_name} wants to discuss strategy with you.",
                f"{npc_name} requests resources for the effort.",
                f"{npc_name} needs your expertise for a critical task.",
                f"{npc_name} seeks your opinion on a tough decision."
            ]
            event["description"] = random.choice(requests)
            event["npc_id"] = npc.get("npc_id")
            event["npc_name"] = npc_name
        else:
            return None

    elif event_type == "resource_opportunity":
        opps = [
            "A source of valuable supplies was discovered.",
            "A potential ally offered support for a favor.",
            "A hidden cache of resources might turn the tide.",
            "A chance to gain intelligence on the enemy emerged.",
            "A new avenue for influence has opened."
        ]
        event["description"] = random.choice(opps)
        resource_types = ["money", "supplies", "influence"]
        resource_type = random.choice(resource_types)
        resource_amount = random.randint(10, 50)
        event["resource_type"] = resource_type
        event["resource_amount"] = resource_amount
        event["expiration"] = 2

    elif event_type == "unexpected_development":
        devs = [
            "An unexpected betrayal shifts the balance of power.",
            "A natural disaster strikes the conflict area.",
            "A neutral third party intervenes unexpectedly.",
            "Public opinion dramatically shifts regarding the conflict.",
            "A crucial piece of intel is revealed to all sides."
        ]
        event["description"] = random.choice(devs)
        increment = random.randint(5, 15)
        await conflict_system.update_progress(conflict_id, increment)
        event["progress_impact"] = increment

    # record event in conflict memory
    await conflict_system.conflict_manager._create_conflict_memory(
        conflict_id,
        f"Event: {event['description']}", significance=6
    )
    return event

async def integrate_conflict_with_time_module(user_id: int, conversation_id: int, activity_type: str, description: str) -> Dict[str, Any]:
    """
    Master function to combine conflict updates with a time-related activity.
    """
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)

    time_result = await process_conflict_time_advancement(user_id, conversation_id, activity_type)
    activity_result = await conflict_system.process_activity_for_conflict_impact(activity_type, description)

    # 20% chance of conflict events
    events = []
    if random.random() < 0.2:
        events = await check_for_conflict_events(user_id, conversation_id)

    # Possibly create a new conflict from narrative
    narrative_result = None
    if len(description) > 20:
        narrative_result = await conflict_system.add_conflict_to_narrative(description)

    return {
        "time_advancement": time_result,
        "activity_impact": activity_result,
        "conflict_events": events,
        "narrative_analysis": narrative_result
    }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Activity Manager Class (remains the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ActivityManager:
    """
    Manages activities in the game, providing methods to process, classify, 
    and handle effects of player activities.
    
    This class serves as a connection point between player input, 
    the time system, and activity effects.
    """
    
    def __init__(self):
        """Initialize the activity manager."""
        self.activity_types = list(TIME_CONSUMING_ACTIVITIES.keys()) + list(OPTIONAL_ACTIVITIES.keys())
        self.activity_classifiers = {
            # Enhanced keyword mappings for activity detection
            "sleep": ["sleep", "rest", "nap", "bed", "tired", "exhausted"],
            "work_shift": ["work", "job", "shift", "office", "employment"],
            "class_attendance": ["class", "lecture", "study", "school", "university"],
            "social_event": ["party", "gathering", "social", "event", "meet", "celebration"],
            "training": ["train", "practice", "exercise", "workout", "gym", "fitness"],
            "extended_conversation": ["talk", "discuss", "conversation", "chat", "dialogue"],
            "personal_time": ["relax", "chill", "alone", "personal", "me time"],
            "eating": ["eat", "meal", "food", "hungry", "lunch", "dinner", "breakfast"],
            "drinking": ["drink", "water", "thirsty", "beverage", "hydrate"],
            "quick_chat": ["say hi", "greet", "hello", "hey", "quick word"],
            "observe": ["look", "watch", "observe", "see", "examine"],
            "check_phone": ["phone", "message", "text", "call", "notification"],
            "quick_snack": ["snack", "bite", "nibble", "munch"],
            "rest": ["brief rest", "sit", "pause", "breather"],
            "intense_activity": ["intense", "extreme", "exhausting", "grueling", "demanding"]
        }
        
        # Cache for recently processed activities
        self.recent_activities = {}
        
        logger.info("ActivityManager initialized")
    
    async def process_activity(self, user_id: int, conversation_id: int, 
                               player_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a player activity to determine type, effects, and time advancement.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
            player_input: Text description of player activity
            context: Additional context information
            
        Returns:
            Dictionary with activity processing results
        """
        # Create default context if none provided
        if context is None:
            context = {}
            
        # Check if activity type is explicitly provided in context
        if "activity_type" in context:
            activity_type = context["activity_type"]
        else:
            # Classify the activity based on input
            activity_type = self._classify_activity(player_input, context)
        
        # Determine if the activity should advance time
        advance_info = should_advance_time(activity_type)
        
        # Calculate effects based on activity type
        effects = self._calculate_activity_effects(activity_type, player_input, context)
        
        # Cache this activity
        cache_key = f"{user_id}:{conversation_id}:{hash(player_input)}"
        self.recent_activities[cache_key] = {
            "activity_type": activity_type,
            "processed_at": datetime.now(),
            "advances_time": advance_info["should_advance"]
        }
        
        # Create and return result
        result = {
            "activity_type": activity_type,
            "time_advance": advance_info,
            "effects": effects,
            "intensity": self._calculate_intensity(player_input, context)
        }
        
        logger.info(f"Processed activity '{activity_type}' for user {user_id}")
        return result
    
    def _classify_activity(self, player_input: str, context: Dict[str, Any] = None) -> str:
        """
        Classify player input into an activity type using keyword matching
        and context analysis.
        
        Args:
            player_input: Text description of player activity
            context: Additional context information
            
        Returns:
            Classified activity type
        """
        # Normalize input
        input_lower = player_input.lower()
        
        # Location-based context can affect classification
        location = context.get("location", "").lower() if context else ""
        
        # Check each activity type's keywords
        matches = {}
        for activity_type, keywords in self.activity_classifiers.items():
            score = 0
            for keyword in keywords:
                if keyword in input_lower:
                    score += 1
            if score > 0:
                matches[activity_type] = score
        
        # If we have matches, return the highest scoring one
        if matches:
            max_score = max(matches.values())
            top_matches = [k for k, v in matches.items() if v == max_score]
            return top_matches[0]  # Return first top match
        
        # Apply location-based heuristics for better classification
        if location:
            if "bed" in location or "bedroom" in location:
                if "lie" in input_lower or "sit" in input_lower:
                    return "rest"
            elif "class" in location or "school" in location:
                return "class_attendance"
            elif "work" in location or "office" in location:
                return "work_shift"
            elif "kitchen" in location or "dining" in location:
                if "eat" not in input_lower:
                    return "drinking"  # Assume getting water
            elif "gym" in location:
                return "training"
            # Add more location-based rules as needed
        
        # Fall back to classification function from time_cycle.py
        return classify_player_input(player_input)
    
    def _calculate_activity_effects(self, activity_type: str, 
                                  player_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate effects of an activity on resources and stats.
        
        Args:
            activity_type: Type of activity
            player_input: Text description of player activity
            context: Additional context information
            
        Returns:
            Dictionary with calculated effects
        """
        effects = {"stat_effects": {}, "vital_effects": {}}
        
        # Get base effects from predefined activities
        if activity_type in TIME_CONSUMING_ACTIVITIES:
            base_data = TIME_CONSUMING_ACTIVITIES[activity_type]
            effects["stat_effects"] = base_data.get("stat_effects", {}).copy()
            effects["vital_effects"] = base_data.get("vital_effects", {}).copy()
        elif activity_type in OPTIONAL_ACTIVITIES:
            base_data = OPTIONAL_ACTIVITIES[activity_type]
            effects["stat_effects"] = base_data.get("stat_effects", {}).copy()
            effects["vital_effects"] = base_data.get("vital_effects", {}).copy()
        
        # Adjust effects based on player input and context
        intensity = self._calculate_intensity(player_input, context)
        
        # Scale effects by intensity
        for stat, value in effects["stat_effects"].items():
            if stat != "varies":
                effects["stat_effects"][stat] = int(value * intensity)
        
        for vital, value in effects["vital_effects"].items():
            effects["vital_effects"][vital] = int(value * intensity)
        
        # Apply random variation (±20%)
        for category in ["stat_effects", "vital_effects"]:
            for key, value in effects[category].items():
                if key != "varies":
                    variation = random.uniform(0.8, 1.2)
                    effects[category][key] = int(value * variation)
        
        return effects
    
    def _calculate_intensity(self, player_input: str, context: Dict[str, Any] = None) -> float:
        """
        Calculate the intensity of an activity based on player input.
        
        Args:
            player_input: Text description of player activity
            context: Additional context information
            
        Returns:
            Intensity value between 0.5 and 1.5
        """
        # Default intensity
        intensity = 1.0
        
        # Intensity modifiers based on keywords
        intensity_keywords = {
            # Intensity increasers
            "intensely": 0.3,
            "vigorously": 0.3,
            "hard": 0.2,
            "thoroughly": 0.2,
            "completely": 0.2,
            "aggressively": 0.3,
            "desperately": 0.4,
            "frantically": 0.3,
            
            # Intensity decreasers
            "lightly": -0.2,
            "casually": -0.2,
            "briefly": -0.3,
            "quickly": -0.3,
            "halfheartedly": -0.4,
            "lazily": -0.3
        }
        
        # Apply modifiers based on keywords in input
        input_lower = player_input.lower()
        for keyword, modifier in intensity_keywords.items():
            if keyword in input_lower:
                intensity += modifier
        
        # Context-based modifiers
        if context:
            # Fatigue affects intensity
            fatigue = context.get("fatigue", 0)
            if fatigue > 80:
                intensity *= 0.7  # Very tired = less intense
            elif fatigue < 20:
                intensity *= 1.1  # Well rested = more intense
        
        # Ensure intensity stays within reasonable bounds
        intensity = max(0.5, min(1.5, intensity))
        
        return intensity

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Context and Agent Setup (remains the same)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TimeCycleContext:
    """
    Context object passed around the agent calls.
    """
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

# Enhanced Tools with vitals support

@function_tool
async def tool_classify_activity(ctx: RunContextWrapper['TimeCycleContext'], player_input: str, location: str = None, rng_seed: Optional[int] = None) -> str:
    """Use LLM to classify player activity intent."""
    context = {"location": location} if location else {}
    activity_type, confidence = await classify_activity_with_llm(player_input, context, rng_seed)
    return json.dumps({
        "activity_type": activity_type,
        "confidence": confidence,
        "method": "llm" if confidence > 0.5 else "keyword"
    })

@function_tool
async def tool_calculate_intensity(ctx: RunContextWrapper[TimeCycleContext], player_input: str) -> str:
    """Calculate activity intensity using LLM analysis."""
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    
    # Get current vitals
    vitals = await get_current_vitals(user_id, conv_id)
    
    result = await calculate_intensity_with_llm(player_input, vitals, {})
    return json.dumps(result)

@function_tool
async def tool_advance_time_with_events(
    ctx: RunContextWrapper[TimeCycleContext], 
    activity_type: str, 
    rng_seed: Optional[int] = None,
    activity_mood: Optional[str] = None
) -> str:
    """
    Advance time if needed, process special events, update vitals, and return a JSON summary of results.
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    result = await advance_time_with_events(user_id, conv_id, activity_type, rng_seed, activity_mood)
    if enqueue_task:
        await enqueue_task(
            task_name="conflict.process_queue",
            params={"user_id": int(user_id), "conversation_id": str(conversation_id), "max_items": 3},
            priority="low",
            delay_seconds=1
        )
    return json.dumps(result)

@function_tool
async def tool_check_vitals(ctx: RunContextWrapper[TimeCycleContext]) -> str:
    """
    Check current player vitals (hunger, thirst, fatigue).
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    
    try:
        vitals = await get_current_vitals(user_id, conv_id)
        
        result = {
            "energy": vitals.energy,
            "hunger": vitals.hunger,
            "thirst": vitals.thirst,
            "fatigue": vitals.fatigue,
            "status": {
                "energy": _get_vital_status(vitals.energy, "energy"),
                "hunger": _get_vital_status(vitals.hunger, "hunger"),
                "thirst": _get_vital_status(vitals.thirst, "thirst"),
                "fatigue": _get_vital_status(vitals.fatigue, "fatigue")
            }
        }
        
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

@function_tool
async def tool_nightly_maintenance(ctx: RunContextWrapper[TimeCycleContext]) -> str:
    """
    Run nightly memory fade, summarization, etc. Return a summary of operations.
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    await nightly_maintenance(user_id, conv_id)
    return "Nightly maintenance completed with relationship drift applied."

@function_tool
async def tool_process_conflict_time_advancement(ctx: RunContextWrapper[TimeCycleContext], activity_type: str) -> str:
    """
    Process time advancement for conflicts, e.g. updating vitals and conflicts. Return JSON result.
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    result = await process_conflict_time_advancement(user_id, conv_id, activity_type)
    return json.dumps(result)

@function_tool
async def tool_integrate_conflict_with_time_module(ctx: RunContextWrapper[TimeCycleContext], activity_type: str, description: str) -> str:
    """
    High-level function to integrate conflict with time module. Return JSON result.
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    result = await integrate_conflict_with_time_module(user_id, conv_id, activity_type, description)
    return json.dumps(result)

@function_tool
async def tool_consume_vital_resource(ctx: RunContextWrapper[TimeCycleContext], resource_type: str, amount: int) -> str:
    """
    Consume a vital resource (food/water) to restore hunger/thirst.
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    
    try:
        async with get_db_connection_context() as conn:
            current = await conn.fetchrow("""
                SELECT energy, hunger, thirst FROM PlayerVitals
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
            """, user_id, conv_id)
            
            if not current:
                return json.dumps({"error": "No vitals found"})
            
            if resource_type == "food":
                new_hunger = min(100, current['hunger'] + amount)
                await conn.execute("""
                    UPDATE PlayerVitals SET hunger = $1
                    WHERE user_id = $2 AND conversation_id = $3 AND player_name = 'Chase'
                """, new_hunger, user_id, conv_id)
                result = {
                    "consumed": "food",
                    "amount": amount,
                    "old_hunger": current['hunger'],
                    "new_hunger": new_hunger
                }
            elif resource_type == "water":
                new_thirst = min(100, current['thirst'] + amount)
                await conn.execute("""
                    UPDATE PlayerVitals SET thirst = $1
                    WHERE user_id = $2 AND conversation_id = $3 AND player_name = 'Chase'
                """, new_thirst, user_id, conv_id)
                result = {
                    "consumed": "water",
                    "amount": amount,
                    "old_thirst": current['thirst'],
                    "new_thirst": new_thirst
                }
            else:
                result = {"error": "Unknown resource type"}
            
            return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

# NEW TOOL for checking relationship dynamics
@function_tool
async def tool_check_relationship_dynamics(
    ctx: RunContextWrapper[TimeCycleContext], 
    npc_id: int
) -> str:
    """Check current relationship dynamics with an NPC."""
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    
    manager = OptimizedRelationshipManager(user_id, conv_id)
    
    # Assuming player is entity1
    state = await manager.get_relationship_state(
        "player", 1,  # Assuming player_id is 1
        "npc", npc_id
    )
    
    summary = state.to_summary()
    summary["npc_id"] = npc_id
    
    return json.dumps(summary)

# Enhanced agent instructions
TIMECYCLE_AGENT_INSTRUCTIONS = """
You are the TimeCycleAgent with LLM-powered capabilities.

You manage:
- Intelligent activity classification using context and slang
- Dynamic intensity calculation based on mood and vitals
- Smart event selection via narrative direction
- Contextual crisis narration
- Phase recaps and suggestions
- Dynamic relationship system integration

Your tools include:
- tool_classify_activity: Use LLM for nuanced activity classification
- tool_calculate_intensity: Determine intensity with context awareness
- tool_advance_time_with_events: Process time with intelligent event selection
- tool_check_vitals: Monitor hunger/thirst/fatigue/energy
- tool_consume_vital_resource: Handle eating/drinking
- tool_nightly_maintenance (now includes relationship drift)
- tool_process_conflict_time_advancement
- tool_integrate_conflict_with_time_module
- tool_check_relationship_dynamics: Check dynamic relationship state with NPCs

Process flow:
1. Classify activity using LLM (fall back to keywords if needed)
2. Calculate intensity considering vitals and context
3. Check for vital crises and suggest interventions
4. Advance time with narrative-appropriate events
5. Process relationship events from the dynamic system
6. Generate phase recaps when time changes

The relationship system now uses continuous dimensions (trust, affection, etc.) 
instead of discrete levels. Relationships evolve based on patterns and archetypes.

Prioritize narrative cohesion and character development over random events.
"""

TimeCycleAgent = Agent[TimeCycleContext](
    name="TimeCycleAgent",
    instructions=TIMECYCLE_AGENT_INSTRUCTIONS,
    model="gpt-5-nano",
    tools=[
        tool_classify_activity,
        tool_calculate_intensity,
        tool_advance_time_with_events,
        tool_check_vitals,
        tool_consume_vital_resource,
        tool_nightly_maintenance,
        tool_process_conflict_time_advancement,
        tool_integrate_conflict_with_time_module,
        tool_check_relationship_dynamics
    ],
)
