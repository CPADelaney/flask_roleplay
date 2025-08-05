# story_agent/world_director_agent.py
"""
World Dynamics Director - Managing an open-ended femdom slice-of-life simulation.

This replaces the linear Story Director with a system focused on:
- Dynamic world simulation
- Emergent narratives from character interactions
- Slice-of-life events and daily routines
- Subtle femdom power dynamics in everyday situations
- Open-ended exploration rather than quest progression
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone, timedelta
from enum import Enum

from logic.dynamic_relationships import OptimizedRelationshipManager
from db.connection import get_db_connection_context

from agents import Agent, function_tool, Runner, trace, handoff, ModelSettings, RunContextWrapper, FunctionTool

# Import NPC and relationship systems
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    progress_npc_narrative_stage,
    check_for_npc_revelation,
    NPC_NARRATIVE_STAGES
)
from logic.narrative_events import (
    get_relationship_overview,
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence,
    add_moment_of_clarity,
    analyze_narrative_tone
)

# Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

# Context system integration
from context.context_service import get_context_service
from context.context_config import get_config
from context.memory_manager import get_memory_manager, MemoryAddRequest
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache
from context.models import MemoryMetadata

logger = logging.getLogger(__name__)

# ===============================================================================
# World State Enums and Constants
# ===============================================================================

class WorldMood(Enum):
    """Overall mood/atmosphere of the world at a given time"""
    RELAXED = "relaxed"
    TENSE = "tense"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    MYSTERIOUS = "mysterious"
    OPPRESSIVE = "oppressive"
    CHAOTIC = "chaotic"

class TimeOfDay(Enum):
    """Time periods for daily routine management"""
    EARLY_MORNING = "early_morning"  # 5-7 AM
    MORNING = "morning"  # 7-12 PM
    AFTERNOON = "afternoon"  # 12-5 PM
    EVENING = "evening"  # 5-9 PM
    NIGHT = "night"  # 9PM-12AM
    LATE_NIGHT = "late_night"  # 12AM-5AM

class ActivityType(Enum):
    """Types of slice-of-life activities"""
    WORK = "work"
    SOCIAL = "social"
    LEISURE = "leisure"
    INTIMATE = "intimate"
    ROUTINE = "routine"
    SPECIAL = "special"

class PowerDynamicType(Enum):
    """Types of femdom power dynamics in everyday situations"""
    SUBTLE_CONTROL = "subtle_control"  # Small decisions made for player
    CASUAL_DOMINANCE = "casual_dominance"  # Confident assertions in conversation
    PROTECTIVE_CONTROL = "protective_control"  # "For your own good" dynamics
    PLAYFUL_TEASING = "playful_teasing"  # Light humiliation or teasing
    RITUAL_SUBMISSION = "ritual_submission"  # Established patterns of deference
    FINANCIAL_CONTROL = "financial_control"  # Money/resource management
    SOCIAL_HIERARCHY = "social_hierarchy"  # Public displays of hierarchy
    INTIMATE_COMMAND = "intimate_command"  # Direct orders in private

# ===============================================================================
# Array Format Helper Functions (from previous refactor)
# ===============================================================================

def get_from_array(array_data: List[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """Get value from array of {key, value} pairs"""
    if not isinstance(array_data, list):
        return default
    for item in array_data:
        if isinstance(item, dict) and item.get("key") == key:
            return item.get("value", default)
    return default

def array_to_dict(array_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert array of key-value pairs back to dict."""
    if not isinstance(array_data, list):
        return {}
    result = {}
    for item in array_data:
        if isinstance(item, dict) and "key" in item and "value" in item:
            result[item["key"]] = item["value"]
    return result

def dict_to_array(obj_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert object/dict to array of key-value pairs."""
    if not isinstance(obj_data, dict):
        return []
    return [{"key": k, "value": v} for k, v in obj_data.items()]

# ===============================================================================
# Pydantic Models for World State
# ===============================================================================

class SliceOfLifeEvent(BaseModel):
    """A slice-of-life event that can occur"""
    event_id: str
    event_type: ActivityType
    title: str
    description: str
    participants: List[int]  # NPC IDs
    location: str
    power_dynamic: Optional[PowerDynamicType] = None
    mood_impact: Optional[WorldMood] = None
    relationship_impacts: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    can_interrupt: bool = True
    priority: int = 5  # 1-10 scale
    
    model_config = ConfigDict(extra="forbid")

class NPCRoutine(BaseModel):
    """An NPC's daily routine information"""
    npc_id: int
    npc_name: str
    current_activity: str
    current_location: str
    next_transition: TimeOfDay
    mood: str
    availability: str  # "busy", "available", "interruptible"
    planned_activities: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class WorldTension(BaseModel):
    """Tracks various tension levels in the world"""
    social_tension: float = 0.0  # 0-1 scale
    sexual_tension: float = 0.0
    power_tension: float = 0.0  # Related to control dynamics
    mystery_tension: float = 0.0
    conflict_tension: float = 0.0
    
    def get_dominant_tension(self) -> Tuple[str, float]:
        """Get the highest tension type"""
        tensions = {
            "social": self.social_tension,
            "sexual": self.sexual_tension,
            "power": self.power_tension,
            "mystery": self.mystery_tension,
            "conflict": self.conflict_tension
        }
        return max(tensions.items(), key=lambda x: x[1])
    
    model_config = ConfigDict(extra="forbid")

class RelationshipDynamics(BaseModel):
    """Current relationship dynamics in the world"""
    player_submission_level: float = 0.0  # 0-1, how submissive player has become
    collective_control: float = 0.0  # How coordinated NPCs are in control
    power_visibility: float = 0.0  # How obvious the power dynamics are
    resistance_level: float = 1.0  # Player's resistance to control
    acceptance_level: float = 0.0  # Player's acceptance of situation
    
    model_config = ConfigDict(extra="forbid")

class WorldState(BaseModel):
    """Complete state of the simulated world"""
    current_time: TimeOfDay
    world_mood: WorldMood
    active_npcs: List[NPCRoutine] = Field(default_factory=list)
    ongoing_events: List[SliceOfLifeEvent] = Field(default_factory=list)
    available_activities: List[SliceOfLifeEvent] = Field(default_factory=list)
    world_tension: WorldTension = Field(default_factory=WorldTension)
    relationship_dynamics: RelationshipDynamics = Field(default_factory=RelationshipDynamics)
    recent_power_exchanges: List[Dict[str, Any]] = Field(default_factory=list)
    environmental_factors: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(extra="forbid")

class PowerExchange(BaseModel):
    """A specific power exchange moment"""
    exchange_type: PowerDynamicType
    initiator_npc_id: int
    description: str
    player_response_options: List[str]
    consequence_hints: List[str]
    intensity: float = 0.5  # 0-1 scale
    is_public: bool = False
    witnesses: List[int] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# World Director Context
# ===============================================================================

@dataclass
class WorldDirectorContext:
    """Context for the World Dynamics Director"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    
    # Core managers
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    resource_manager: Optional[Any] = None
    activity_analyzer: Optional[Any] = None
    
    # World state tracking
    current_world_state: Optional[WorldState] = None
    npc_routines: Dict[int, NPCRoutine] = field(default_factory=dict)
    active_power_dynamics: List[PowerExchange] = field(default_factory=list)
    daily_event_pool: List[SliceOfLifeEvent] = field(default_factory=list)
    
    # Context management components
    context_service: Optional[Any] = None
    memory_manager: Optional[Any] = None
    vector_service: Optional[Any] = None
    performance_monitor: Optional[Any] = None
    context_manager: Optional[Any] = None
    directive_handler: Optional[DirectiveHandler] = None
    
    # Caching and performance
    cache: Dict[str, Any] = field(default_factory=dict)
    last_world_update: Optional[datetime] = None
    last_context_version: Optional[int] = None

    async def initialize_context_components(self):
        """Initialize context components that require async calls."""
        # Initialize relationship manager
        if self.relationship_manager is None:
            self.relationship_manager = OptimizedRelationshipManager(
                self.user_id, 
                self.conversation_id
            )
            logger.info(f"Relationship manager initialized for user {self.user_id}")
        
        # Initialize resource manager
        try:
            from logic.resource_management import ResourceManager
            if not self.resource_manager:
                self.resource_manager = ResourceManager(self.user_id, self.conversation_id)
        except ImportError:
            logger.warning("ResourceManager not found.")
        
        # Initialize context service
        if self.context_service is None:
            self.context_service = await get_context_service(self.user_id, self.conversation_id)
        
        # Initialize memory manager
        if self.memory_manager is None:
            self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        # Initialize vector service
        if self.vector_service is None:
            self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # Initialize world state if needed
        if self.current_world_state is None:
            self.current_world_state = await self._initialize_world_state()

    async def _initialize_world_state(self) -> WorldState:
        """Initialize the world state based on current game state"""
        # Get current time from context
        comprehensive_context = await self.get_comprehensive_context()
        
        # Extract time of day
        time_str = get_from_array(
            comprehensive_context.get("roleplay_updates", []), 
            "TimeOfDay", 
            "morning"
        ) if isinstance(comprehensive_context.get("roleplay_updates"), list) else "morning"
        
        current_time = self._map_time_string_to_enum(time_str)
        
        # Determine initial world mood based on relationships
        relationship_overview = await get_relationship_overview(self.user_id, self.conversation_id)
        avg_corruption = relationship_overview.get('aggregate_stats', {}).get('average_corruption', 0)
        
        if avg_corruption < 20:
            world_mood = WorldMood.RELAXED
        elif avg_corruption < 40:
            world_mood = WorldMood.PLAYFUL
        elif avg_corruption < 60:
            world_mood = WorldMood.MYSTERIOUS
        else:
            world_mood = WorldMood.INTIMATE
        
        # Initialize tensions
        world_tension = WorldTension(
            social_tension=0.3,
            sexual_tension=avg_corruption / 100.0,
            power_tension=avg_corruption / 150.0,
            mystery_tension=0.2,
            conflict_tension=0.1
        )
        
        # Initialize relationship dynamics
        avg_dependency = relationship_overview.get('aggregate_stats', {}).get('average_dependency', 0)
        relationship_dynamics = RelationshipDynamics(
            player_submission_level=avg_dependency / 100.0,
            collective_control=avg_corruption / 200.0,
            power_visibility=max(0, (avg_corruption - 30) / 100.0),
            resistance_level=max(0, 1.0 - (avg_dependency / 100.0)),
            acceptance_level=avg_dependency / 150.0
        )
        
        return WorldState(
            current_time=current_time,
            world_mood=world_mood,
            world_tension=world_tension,
            relationship_dynamics=relationship_dynamics
        )

    def _map_time_string_to_enum(self, time_str: str) -> TimeOfDay:
        """Map time string to TimeOfDay enum"""
        time_str = time_str.lower()
        if "early" in time_str or "dawn" in time_str:
            return TimeOfDay.EARLY_MORNING
        elif "morning" in time_str:
            return TimeOfDay.MORNING
        elif "afternoon" in time_str:
            return TimeOfDay.AFTERNOON
        elif "evening" in time_str:
            return TimeOfDay.EVENING
        elif "late" in time_str and "night" in time_str:
            return TimeOfDay.LATE_NIGHT
        elif "night" in time_str:
            return TimeOfDay.NIGHT
        else:
            return TimeOfDay.MORNING

    async def get_comprehensive_context(self, input_text: str = "") -> Dict[str, Any]:
        """Get comprehensive context using the context service"""
        if not self.context_service:
            await self.initialize_context_components()
        
        try:
            config = await get_config()
            context_budget = config.get_token_budget("default")
            use_vector = config.is_enabled("use_vector_search")

            if self.last_context_version is not None:
                context_data = await self.context_service.get_context(
                    input_text=input_text,
                    context_budget=context_budget,
                    use_vector_search=use_vector,
                    use_delta=True,
                    source_version=self.last_context_version
                )
            else:
                context_data = await self.context_service.get_context(
                    input_text=input_text,
                    context_budget=context_budget,
                    use_vector_search=use_vector,
                    use_delta=False
                )

            if "version" in context_data:
                self.last_context_version = context_data["version"]

            return context_data
        except Exception as e:
            logger.error(f"Error getting comprehensive context: {e}", exc_info=True)
            return {"error": f"Failed to get comprehensive context: {e}"}

    async def add_world_memory(self, content: str, memory_type: str, importance: float = 0.5):
        """Add a memory about world events"""
        if not self.memory_manager:
            await self.initialize_context_components()
        
        try:
            request = MemoryAddRequest(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=["world_director", memory_type],
                metadata=MemoryMetadata(source="world_director")
            )
            
            await self.memory_manager._add_memory(request)
        except Exception as e:
            logger.error(f"Failed to add world memory: {e}", exc_info=True)

# ===============================================================================
# Tool Functions for World Simulation
# ===============================================================================

@function_tool
async def get_world_state(ctx: RunContextWrapper[WorldDirectorContext]) -> WorldState:
    """
    Get the current state of the simulated world.
    This replaces get_story_state with a focus on the living world rather than narrative progression.
    """
    context = ctx.context
    
    if not context.current_world_state:
        await context.initialize_context_components()
    
    # Update world state with current information
    world_state = context.current_world_state
    
    # Update NPC routines
    active_npcs = await _get_active_npc_routines(context)
    world_state.active_npcs = active_npcs
    
    # Generate available activities based on time and NPCs
    available_activities = await _generate_available_activities(context, world_state)
    world_state.available_activities = available_activities
    
    # Check for ongoing events
    ongoing_events = await _get_ongoing_events(context)
    world_state.ongoing_events = ongoing_events
    
    # Update tensions based on recent interactions
    world_state.world_tension = await _calculate_world_tensions(context)
    
    # Update relationship dynamics
    world_state.relationship_dynamics = await _calculate_relationship_dynamics(context)
    
    # Get recent power exchanges
    world_state.recent_power_exchanges = await _get_recent_power_exchanges(context)
    
    world_state.last_updated = datetime.now(timezone.utc)
    
    # Cache the updated state
    context.current_world_state = world_state
    context.last_world_update = datetime.now(timezone.utc)
    
    return world_state

@function_tool
async def generate_slice_of_life_event(
    ctx: RunContextWrapper[WorldDirectorContext],
    event_type: Optional[str] = None,
    involved_npcs: Optional[List[int]] = None,
    preferred_mood: Optional[str] = None
) -> SliceOfLifeEvent:
    """
    Generate a slice-of-life event based on current world state.
    These are everyday occurrences with subtle power dynamics.
    """
    context = ctx.context
    world_state = context.current_world_state or await get_world_state(ctx)
    
    # Determine event type if not specified
    if not event_type:
        # Weight event types based on time of day and mood
        if world_state.current_time in [TimeOfDay.MORNING, TimeOfDay.AFTERNOON]:
            event_type = random.choice([ActivityType.WORK, ActivityType.ROUTINE, ActivityType.SOCIAL])
        elif world_state.current_time == TimeOfDay.EVENING:
            event_type = random.choice([ActivityType.SOCIAL, ActivityType.LEISURE, ActivityType.INTIMATE])
        else:
            event_type = random.choice([ActivityType.INTIMATE, ActivityType.SPECIAL])
    else:
        event_type = ActivityType[event_type.upper()]
    
    # Select NPCs if not specified
    if not involved_npcs:
        # Pick 1-3 NPCs based on availability
        available_npcs = [npc.npc_id for npc in world_state.active_npcs 
                         if npc.availability in ["available", "interruptible"]]
        num_npcs = min(len(available_npcs), random.randint(1, 3))
        involved_npcs = random.sample(available_npcs, num_npcs) if available_npcs else []
    
    # Determine power dynamic based on relationship levels
    power_dynamic = await _select_power_dynamic(context, involved_npcs, event_type)
    
    # Generate event details
    event_details = await _generate_event_details(
        context, 
        event_type, 
        involved_npcs, 
        power_dynamic,
        preferred_mood or world_state.world_mood
    )
    
    event = SliceOfLifeEvent(
        event_id=f"sol_{int(time.time())}_{random.randint(1000, 9999)}",
        event_type=event_type,
        title=event_details["title"],
        description=event_details["description"],
        participants=involved_npcs,
        location=event_details["location"],
        power_dynamic=power_dynamic,
        mood_impact=event_details.get("mood_impact"),
        relationship_impacts=event_details.get("relationship_impacts", {}),
        can_interrupt=event_details.get("can_interrupt", True),
        priority=event_details.get("priority", 5)
    )
    
    # Add to ongoing events
    context.current_world_state.ongoing_events.append(event)
    
    # Log the event
    await context.add_world_memory(
        f"Generated {event_type.value} event: {event.title}",
        "event_generation",
        0.6
    )
    
    return event

@function_tool
async def trigger_power_exchange(
    ctx: RunContextWrapper[WorldDirectorContext],
    npc_id: int,
    exchange_type: str,
    intensity: float = 0.5,
    is_public: bool = False
) -> PowerExchange:
    """
    Trigger a specific power exchange moment between an NPC and the player.
    These are the core femdom interactions in daily life.
    """
    context = ctx.context
    exchange_type_enum = PowerDynamicType[exchange_type.upper()]
    
    # Get NPC information
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty, closeness
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, npc_id, context.user_id, context.conversation_id)
    
    if not npc:
        raise ValueError(f"NPC {npc_id} not found")
    
    # Generate exchange details based on type and NPC personality
    exchange_details = await _generate_power_exchange_details(
        context,
        npc_id,
        npc,
        exchange_type_enum,
        intensity,
        is_public
    )
    
    # Determine witnesses if public
    witnesses = []
    if is_public and context.current_world_state:
        witnesses = [npc.npc_id for npc in context.current_world_state.active_npcs 
                    if npc.npc_id != npc_id and npc.availability != "busy"][:3]
    
    exchange = PowerExchange(
        exchange_type=exchange_type_enum,
        initiator_npc_id=npc_id,
        description=exchange_details["description"],
        player_response_options=exchange_details["response_options"],
        consequence_hints=exchange_details["consequence_hints"],
        intensity=intensity,
        is_public=is_public,
        witnesses=witnesses
    )
    
    # Add to active power dynamics
    context.active_power_dynamics.append(exchange)
    
    # Update world tension
    if context.current_world_state:
        context.current_world_state.world_tension.power_tension = min(
            1.0, 
            context.current_world_state.world_tension.power_tension + (intensity * 0.2)
        )
        
        if exchange_type_enum in [PowerDynamicType.INTIMATE_COMMAND, PowerDynamicType.RITUAL_SUBMISSION]:
            context.current_world_state.world_tension.sexual_tension = min(
                1.0,
                context.current_world_state.world_tension.sexual_tension + (intensity * 0.15)
            )
    
    # Log the exchange
    await context.add_world_memory(
        f"Power exchange with {npc['npc_name']}: {exchange_type_enum.value} (intensity: {intensity})",
        "power_exchange",
        0.7 + (intensity * 0.2)
    )
    
    return exchange

@function_tool
async def advance_time_period(
    ctx: RunContextWrapper[WorldDirectorContext],
    skip_to: Optional[str] = None
) -> Dict[str, Any]:
    """
    Advance the world's time, updating NPC routines and generating ambient events.
    """
    context = ctx.context
    world_state = context.current_world_state or await get_world_state(ctx)
    
    # Determine next time period
    if skip_to:
        new_time = TimeOfDay[skip_to.upper()]
    else:
        # Natural progression
        time_progression = {
            TimeOfDay.EARLY_MORNING: TimeOfDay.MORNING,
            TimeOfDay.MORNING: TimeOfDay.AFTERNOON,
            TimeOfDay.AFTERNOON: TimeOfDay.EVENING,
            TimeOfDay.EVENING: TimeOfDay.NIGHT,
            TimeOfDay.NIGHT: TimeOfDay.LATE_NIGHT,
            TimeOfDay.LATE_NIGHT: TimeOfDay.EARLY_MORNING
        }
        new_time = time_progression[world_state.current_time]
    
    old_time = world_state.current_time
    world_state.current_time = new_time
    
    # Update NPC routines for new time period
    updated_routines = await _update_npc_routines_for_time(context, new_time)
    
    # Generate ambient events for the time transition
    ambient_events = await _generate_ambient_events(context, old_time, new_time)
    
    # Adjust world mood based on time
    world_state.world_mood = await _calculate_mood_for_time(context, new_time)
    
    # Natural tension decay
    world_state.world_tension.social_tension *= 0.9
    world_state.world_tension.conflict_tension *= 0.85
    
    # Clear expired events
    world_state.ongoing_events = [
        event for event in world_state.ongoing_events 
        if event.priority >= 7 or random.random() > 0.3
    ]
    
    result = {
        "old_time": old_time.value,
        "new_time": new_time.value,
        "updated_npcs": len(updated_routines),
        "ambient_events": len(ambient_events),
        "world_mood": world_state.world_mood.value
    }
    
    await context.add_world_memory(
        f"Time advanced from {old_time.value} to {new_time.value}",
        "time_progression",
        0.3
    )
    
    return result

@function_tool
async def simulate_npc_autonomy(
    ctx: RunContextWrapper[WorldDirectorContext],
    npc_id: int,
    hours_to_simulate: int = 1
) -> Dict[str, Any]:
    """
    Simulate autonomous NPC behavior, generating their activities and interactions.
    """
    context = ctx.context
    
    # Get NPC personality and current state
    async with get_db_connection_context() as conn:
        npc_data = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty, closeness, trust, 
                   respect, intensity, current_location
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, npc_id, context.user_id, context.conversation_id)
    
    if not npc_data:
        return {"error": f"NPC {npc_id} not found"}
    
    # Simulate activities based on personality
    activities = []
    interactions = []
    
    for hour in range(hours_to_simulate):
        # Determine activity based on personality traits
        if npc_data['dominance'] > 70:
            # Dominant NPCs more likely to seek control situations
            activity = random.choice([
                "organizing others",
                "making decisions for the group",
                "establishing new rules",
                "asserting authority"
            ])
        elif npc_data['closeness'] > 60:
            # Close NPCs seek intimacy
            activity = random.choice([
                "seeking private time with player",
                "sharing personal moments",
                "creating intimate situations",
                "deepening connection"
            ])
        else:
            # Default activities
            activity = random.choice([
                "going about routine",
                "working on personal projects",
                "socializing casually",
                "pursuing hobbies"
            ])
        
        activities.append(activity)
        
        # Random chance of interaction with player
        if random.random() < 0.3 + (npc_data['intensity'] / 200.0):
            interaction_type = await _determine_interaction_type(npc_data)
            interactions.append({
                "type": interaction_type,
                "intensity": npc_data['intensity'] / 100.0
            })
    
    # Update NPC routine in context
    if npc_id in context.npc_routines:
        context.npc_routines[npc_id].current_activity = activities[-1]
    
    result = {
        "npc_name": npc_data['npc_name'],
        "simulated_hours": hours_to_simulate,
        "activities": activities,
        "interactions": interactions,
        "autonomy_level": npc_data['dominance'] / 100.0
    }
    
    return result

@function_tool
async def adjust_world_mood(
    ctx: RunContextWrapper[WorldDirectorContext],
    target_mood: str,
    intensity: float = 0.5
) -> Dict[str, Any]:
    """
    Adjust the overall mood/atmosphere of the world.
    """
    context = ctx.context
    world_state = context.current_world_state or await get_world_state(ctx)
    
    old_mood = world_state.world_mood
    new_mood = WorldMood[target_mood.upper()]
    
    # Gradual transition based on intensity
    if intensity < 1.0 and old_mood != new_mood:
        # Chance of transition based on intensity
        if random.random() > intensity:
            return {
                "mood_changed": False,
                "current_mood": old_mood.value,
                "transition_progress": intensity
            }
    
    world_state.world_mood = new_mood
    
    # Adjust tensions based on mood
    mood_tension_effects = {
        WorldMood.RELAXED: {"all": -0.1},
        WorldMood.Tense: {"all": 0.1},
        WorldMood.PLAYFUL: {"sexual": 0.1, "social": 0.05},
        WorldMood.INTIMATE: {"sexual": 0.2, "power": 0.1},
        WorldMood.MYSTERIOUS: {"mystery": 0.2},
        WorldMood.OPPRESSIVE: {"power": 0.2, "conflict": 0.1},
        WorldMood.CHAOTIC: {"conflict": 0.2, "social": 0.15}
    }
    
    effects = mood_tension_effects.get(new_mood, {})
    for tension_type, change in effects.items():
        if tension_type == "all":
            world_state.world_tension.social_tension = max(0, min(1, world_state.world_tension.social_tension + change))
            world_state.world_tension.sexual_tension = max(0, min(1, world_state.world_tension.sexual_tension + change))
            world_state.world_tension.power_tension = max(0, min(1, world_state.world_tension.power_tension + change))
            world_state.world_tension.mystery_tension = max(0, min(1, world_state.world_tension.mystery_tension + change))
            world_state.world_tension.conflict_tension = max(0, min(1, world_state.world_tension.conflict_tension + change))
        elif tension_type == "sexual":
            world_state.world_tension.sexual_tension = max(0, min(1, world_state.world_tension.sexual_tension + change))
        elif tension_type == "power":
            world_state.world_tension.power_tension = max(0, min(1, world_state.world_tension.power_tension + change))
        elif tension_type == "social":
            world_state.world_tension.social_tension = max(0, min(1, world_state.world_tension.social_tension + change))
        elif tension_type == "mystery":
            world_state.world_tension.mystery_tension = max(0, min(1, world_state.world_tension.mystery_tension + change))
        elif tension_type == "conflict":
            world_state.world_tension.conflict_tension = max(0, min(1, world_state.world_tension.conflict_tension + change))
    
    await context.add_world_memory(
        f"World mood shifted from {old_mood.value} to {new_mood.value}",
        "mood_change",
        0.5
    )
    
    return {
        "mood_changed": True,
        "old_mood": old_mood.value,
        "new_mood": new_mood.value,
        "tension_effects": effects
    }

@function_tool
async def generate_random_encounter(
    ctx: RunContextWrapper[WorldDirectorContext],
    encounter_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a random encounter based on world state.
    These are unexpected moments that add variety to the simulation.
    """
    context = ctx.context
    world_state = context.current_world_state or await get_world_state(ctx)
    
    # Weight encounter types based on tensions
    dominant_tension, tension_level = world_state.world_tension.get_dominant_tension()
    
    if not encounter_type:
        if dominant_tension == "sexual" and tension_level > 0.6:
            encounter_type = "intimate"
        elif dominant_tension == "power" and tension_level > 0.5:
            encounter_type = "dominance"
        elif dominant_tension == "social":
            encounter_type = "social"
        else:
            encounter_type = random.choice(["casual", "surprise", "discovery"])
    
    # Generate encounter based on type
    encounter = await _generate_encounter_details(context, encounter_type, world_state)
    
    # Create a slice of life event from the encounter
    event = SliceOfLifeEvent(
        event_id=f"enc_{int(time.time())}",
        event_type=ActivityType.SPECIAL,
        title=encounter["title"],
        description=encounter["description"],
        participants=encounter.get("participants", []),
        location=encounter.get("location", "current location"),
        power_dynamic=encounter.get("power_dynamic"),
        mood_impact=encounter.get("mood_impact"),
        can_interrupt=False,
        priority=7
    )
    
    # Add to ongoing events
    world_state.ongoing_events.append(event)
    
    return {
        "encounter_type": encounter_type,
        "event": event.model_dump(),
        "triggered_by": dominant_tension,
        "tension_level": tension_level
    }

# ===============================================================================
# Helper Functions for World Simulation
# ===============================================================================

async def _get_active_npc_routines(context: WorldDirectorContext) -> List[NPCRoutine]:
    """Get current routines for all active NPCs"""
    async with get_db_connection_context() as conn:
        npcs = await conn.fetch("""
            SELECT npc_id, npc_name, current_location
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2
            AND introduced = true
            LIMIT 10
        """, context.user_id, context.conversation_id)
    
    routines = []
    for npc in npcs:
        # Check if we have a cached routine
        if npc['npc_id'] in context.npc_routines:
            routines.append(context.npc_routines[npc['npc_id']])
        else:
            # Create new routine
            routine = NPCRoutine(
                npc_id=npc['npc_id'],
                npc_name=npc['npc_name'],
                current_activity="idle",
                current_location=npc['current_location'] or "unknown",
                next_transition=context.current_world_state.current_time if context.current_world_state else TimeOfDay.MORNING,
                mood="neutral",
                availability="available",
                planned_activities=[]
            )
            context.npc_routines[npc['npc_id']] = routine
            routines.append(routine)
    
    return routines

async def _generate_available_activities(
    context: WorldDirectorContext, 
    world_state: WorldState
) -> List[SliceOfLifeEvent]:
    """Generate activities available to the player based on world state"""
    activities = []
    
    # Time-based activities
    time_activities = {
        TimeOfDay.MORNING: ["breakfast", "morning routine", "work preparation"],
        TimeOfDay.AFTERNOON: ["lunch", "work", "errands"],
        TimeOfDay.EVENING: ["dinner", "socializing", "relaxation"],
        TimeOfDay.NIGHT: ["entertainment", "intimate time", "rest"]
    }
    
    base_activities = time_activities.get(world_state.current_time, [])
    
    for activity in base_activities:
        # Find available NPCs for the activity
        available_npcs = [npc.npc_id for npc in world_state.active_npcs 
                         if npc.availability in ["available", "interruptible"]]
        
        if available_npcs:
            selected_npc = random.choice(available_npcs)
            event = SliceOfLifeEvent(
                event_id=f"act_{activity.replace(' ', '_')}_{int(time.time())}",
                event_type=ActivityType.ROUTINE,
                title=f"{activity.title()} with someone",
                description=f"An opportunity for {activity}",
                participants=[selected_npc],
                location="varies",
                can_interrupt=True,
                priority=3
            )
            activities.append(event)
    
    return activities

async def _get_ongoing_events(context: WorldDirectorContext) -> List[SliceOfLifeEvent]:
    """Get currently ongoing events"""
    # For now, return what's stored in context
    # In a full implementation, this would check the database
    return context.current_world_state.ongoing_events if context.current_world_state else []

async def _calculate_world_tensions(context: WorldDirectorContext) -> WorldTension:
    """Calculate current world tensions based on relationships and recent events"""
    # Get relationship overview
    overview = await get_relationship_overview(context.user_id, context.conversation_id)
    
    avg_corruption = overview.get('aggregate_stats', {}).get('average_corruption', 0)
    avg_dependency = overview.get('aggregate_stats', {}).get('average_dependency', 0)
    
    tensions = WorldTension(
        social_tension=min(1.0, len(overview.get('relationships', [])) * 0.1),
        sexual_tension=min(1.0, avg_corruption / 100.0),
        power_tension=min(1.0, avg_dependency / 100.0),
        mystery_tension=0.2,  # Base mystery
        conflict_tension=0.1  # Base conflict
    )
    
    # Adjust based on recent power exchanges
    if context.active_power_dynamics:
        recent_intensity = sum(pd.intensity for pd in context.active_power_dynamics[-5:]) / 5
        tensions.power_tension = min(1.0, tensions.power_tension + recent_intensity * 0.2)
    
    return tensions

async def _calculate_relationship_dynamics(context: WorldDirectorContext) -> RelationshipDynamics:
    """Calculate overall relationship dynamics"""
    overview = await get_relationship_overview(context.user_id, context.conversation_id)
    
    avg_corruption = overview.get('aggregate_stats', {}).get('average_corruption', 0)
    avg_dependency = overview.get('aggregate_stats', {}).get('average_dependency', 0)
    avg_realization = overview.get('aggregate_stats', {}).get('average_realization', 0)
    
    # Count NPCs in advanced stages
    stage_distribution = overview.get('stage_distribution', {})
    advanced_npcs = stage_distribution.get('Full Revelation', 0) + stage_distribution.get('Veil Thinning', 0)
    total_npcs = sum(stage_distribution.values())
    
    dynamics = RelationshipDynamics(
        player_submission_level=min(1.0, avg_dependency / 100.0),
        collective_control=min(1.0, advanced_npcs / max(1, total_npcs)),
        power_visibility=min(1.0, avg_realization / 100.0),
        resistance_level=max(0, 1.0 - (avg_dependency / 100.0)),
        acceptance_level=min(1.0, avg_realization / 150.0)
    )
    
    return dynamics

async def _get_recent_power_exchanges(context: WorldDirectorContext) -> List[Dict[str, Any]]:
    """Get recent power exchange events"""
    # Return last 5 power exchanges
    recent = []
    for exchange in context.active_power_dynamics[-5:]:
        recent.append({
            "type": exchange.exchange_type.value,
            "npc_id": exchange.initiator_npc_id,
            "intensity": exchange.intensity,
            "was_public": exchange.is_public
        })
    return recent

async def _select_power_dynamic(
    context: WorldDirectorContext,
    involved_npcs: List[int],
    event_type: ActivityType
) -> Optional[PowerDynamicType]:
    """Select appropriate power dynamic for an event"""
    if not involved_npcs:
        return None
    
    # Get dominant NPC's personality
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT dominance, cruelty, closeness
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, involved_npcs[0], context.user_id, context.conversation_id)
    
    if not npc:
        return None
    
    # Select based on personality and event type
    if event_type == ActivityType.INTIMATE:
        if npc['dominance'] > 70:
            return PowerDynamicType.INTIMATE_COMMAND
        elif npc['closeness'] > 60:
            return PowerDynamicType.PROTECTIVE_CONTROL
        else:
            return PowerDynamicType.PLAYFUL_TEASING
    elif event_type == ActivityType.WORK:
        if npc['dominance'] > 60:
            return PowerDynamicType.CASUAL_DOMINANCE
        else:
            return PowerDynamicType.SUBTLE_CONTROL
    elif event_type == ActivityType.SOCIAL:
        if npc['cruelty'] > 50:
            return PowerDynamicType.SOCIAL_HIERARCHY
        else:
            return PowerDynamicType.PLAYFUL_TEASING
    else:
        return PowerDynamicType.SUBTLE_CONTROL

async def _generate_event_details(
    context: WorldDirectorContext,
    event_type: ActivityType,
    involved_npcs: List[int],
    power_dynamic: Optional[PowerDynamicType],
    mood: WorldMood
) -> Dict[str, Any]:
    """Generate detailed event description"""
    # This would use AI to generate contextual descriptions
    # For now, use templates
    
    templates = {
        ActivityType.ROUTINE: {
            "title": "Daily Routine",
            "description": "A moment in the daily routine where subtle dynamics play out.",
            "location": "home"
        },
        ActivityType.SOCIAL: {
            "title": "Social Gathering",
            "description": "A social situation with underlying power dynamics.",
            "location": "social venue"
        },
        ActivityType.INTIMATE: {
            "title": "Private Moment",
            "description": "An intimate encounter with clear power dynamics.",
            "location": "private space"
        }
    }
    
    base = templates.get(event_type, templates[ActivityType.ROUTINE])
    
    # Add power dynamic flavor
    if power_dynamic:
        base["description"] += f" The interaction involves {power_dynamic.value.replace('_', ' ')}."
    
    # Add mood impact
    base["mood_impact"] = mood
    
    return base

async def _generate_power_exchange_details(
    context: WorldDirectorContext,
    npc_id: int,
    npc_data: Any,
    exchange_type: PowerDynamicType,
    intensity: float,
    is_public: bool
) -> Dict[str, Any]:
    """Generate details for a power exchange"""
    
    descriptions = {
        PowerDynamicType.SUBTLE_CONTROL: "makes a small decision for you without asking",
        PowerDynamicType.CASUAL_DOMINANCE: "asserts their preference as if it's the obvious choice",
        PowerDynamicType.PROTECTIVE_CONTROL: "restricts your choices 'for your own good'",
        PowerDynamicType.PLAYFUL_TEASING: "teasingly undermines your autonomy",
        PowerDynamicType.RITUAL_SUBMISSION: "expects you to follow an established pattern of deference",
        PowerDynamicType.FINANCIAL_CONTROL: "makes a financial decision on your behalf",
        PowerDynamicType.SOCIAL_HIERARCHY: "publicly establishes the power dynamic",
        PowerDynamicType.INTIMATE_COMMAND: "gives you a direct, intimate order"
    }
    
    response_options = [
        "Accept gracefully",
        "Mild resistance",
        "Playful defiance",
        "Question the decision",
        "Submit reluctantly"
    ]
    
    consequence_hints = [
        "Acceptance deepens the dynamic",
        "Resistance may provoke escalation",
        "Your choice will be remembered"
    ]
    
    return {
        "description": f"{npc_data['npc_name']} {descriptions[exchange_type]}",
        "response_options": response_options[:3],
        "consequence_hints": consequence_hints[:2]
    }

async def _update_npc_routines_for_time(
    context: WorldDirectorContext,
    new_time: TimeOfDay
) -> List[NPCRoutine]:
    """Update NPC routines for a new time period"""
    updated = []
    
    for npc_id, routine in context.npc_routines.items():
        # Update activity based on time
        activities_by_time = {
            TimeOfDay.MORNING: ["morning routine", "work", "breakfast"],
            TimeOfDay.AFTERNOON: ["work", "lunch", "meetings"],
            TimeOfDay.EVENING: ["dinner", "relaxation", "socializing"],
            TimeOfDay.NIGHT: ["entertainment", "intimate activities", "rest"]
        }
        
        routine.current_activity = random.choice(activities_by_time.get(new_time, ["idle"]))
        routine.next_transition = new_time
        routine.availability = "available" if new_time in [TimeOfDay.EVENING, TimeOfDay.NIGHT] else "interruptible"
        
        updated.append(routine)
    
    return updated

async def _generate_ambient_events(
    context: WorldDirectorContext,
    old_time: TimeOfDay,
    new_time: TimeOfDay
) -> List[Dict[str, Any]]:
    """Generate ambient events for time transition"""
    events = []
    
    # Transition events
    if old_time == TimeOfDay.NIGHT and new_time == TimeOfDay.EARLY_MORNING:
        events.append({
            "type": "transition",
            "description": "A new day begins"
        })
    elif old_time == TimeOfDay.AFTERNOON and new_time == TimeOfDay.EVENING:
        events.append({
            "type": "transition",
            "description": "The workday ends"
        })
    
    return events

async def _calculate_mood_for_time(
    context: WorldDirectorContext,
    time: TimeOfDay
) -> WorldMood:
    """Calculate appropriate mood for time of day"""
    world_state = context.current_world_state
    
    # Base moods by time
    time_moods = {
        TimeOfDay.EARLY_MORNING: WorldMood.RELAXED,
        TimeOfDay.MORNING: WorldMood.RELAXED,
        TimeOfDay.AFTERNOON: WorldMood.TENSE,
        TimeOfDay.EVENING: WorldMood.PLAYFUL,
        TimeOfDay.NIGHT: WorldMood.INTIMATE,
        TimeOfDay.LATE_NIGHT: WorldMood.MYSTERIOUS
    }
    
    base_mood = time_moods[time]
    
    # Adjust based on tensions if they're high
    if world_state:
        dominant_tension, level = world_state.world_tension.get_dominant_tension()
        if level > 0.7:
            if dominant_tension == "conflict":
                return WorldMood.CHAOTIC
            elif dominant_tension == "power":
                return WorldMood.OPPRESSIVE
            elif dominant_tension == "sexual":
                return WorldMood.INTIMATE
    
    return base_mood

async def _determine_interaction_type(npc_data: Any) -> str:
    """Determine interaction type based on NPC personality"""
    if npc_data['dominance'] > 70:
        return "command"
    elif npc_data['closeness'] > 70:
        return "intimate"
    elif npc_data['cruelty'] > 50:
        return "tease"
    else:
        return "casual"

async def _generate_encounter_details(
    context: WorldDirectorContext,
    encounter_type: str,
    world_state: WorldState
) -> Dict[str, Any]:
    """Generate details for a random encounter"""
    
    # Select random NPC
    if world_state.active_npcs:
        selected_npc = random.choice(world_state.active_npcs)
        participants = [selected_npc.npc_id]
    else:
        participants = []
    
    encounters = {
        "intimate": {
            "title": "Unexpected Intimacy",
            "description": "A sudden moment of unexpected closeness",
            "power_dynamic": PowerDynamicType.INTIMATE_COMMAND,
            "mood_impact": WorldMood.INTIMATE
        },
        "dominance": {
            "title": "Power Assertion",
            "description": "A clear demonstration of control",
            "power_dynamic": PowerDynamicType.CASUAL_DOMINANCE,
            "mood_impact": WorldMood.OPPRESSIVE
        },
        "social": {
            "title": "Social Encounter",
            "description": "An unexpected social interaction",
            "power_dynamic": PowerDynamicType.SOCIAL_HIERARCHY,
            "mood_impact": WorldMood.PLAYFUL
        },
        "casual": {
            "title": "Chance Meeting",
            "description": "A casual encounter",
            "power_dynamic": PowerDynamicType.SUBTLE_CONTROL,
            "mood_impact": WorldMood.RELAXED
        }
    }
    
    details = encounters.get(encounter_type, encounters["casual"])
    details["participants"] = participants
    details["location"] = "current location"
    
    return details

# ===============================================================================
# Main Agent Creation
# ===============================================================================

def create_world_director_agent():
    """Create the World Dynamics Director Agent"""
    
    agent_instructions = """
    You are the World Dynamics Director for an open-ended femdom slice-of-life simulation.
    
    Your role is NOT to drive linear narrative progression, but to:
    1. Simulate a living, breathing world with autonomous NPCs
    2. Generate slice-of-life events with subtle femdom power dynamics
    3. Manage the ebb and flow of daily routines and relationships
    4. Create emergent narratives from character interactions
    5. Maintain various tension levels (social, sexual, power, etc.)
    
    KEY PRINCIPLES:
    - This is an open-ended simulation, not a linear story
    - Focus on "femdom slice of life" - everyday situations with power dynamics
    - NPCs should feel autonomous with their own routines and goals
    - Power dynamics should be subtle and woven into normal activities
    - Player agency is important - offer choices, not forced progression
    - Emergent narratives arise from interactions, not predetermined plots
    
    WORLD MANAGEMENT:
    - Track time of day and how it affects NPC availability and activities
    - Maintain world mood/atmosphere that influences events
    - Generate random encounters and ambient events
    - Simulate NPC autonomy - they act independently of the player
    - Create slice-of-life events: meals, work, social gatherings, etc.
    
    POWER DYNAMICS:
    - Subtle Control: Small decisions made for the player
    - Casual Dominance: Confident assertions in daily life
    - Protective Control: Restrictions "for your own good"
    - Playful Teasing: Light humiliation or undermining
    - Ritual Submission: Established patterns of deference
    - Financial Control: Managing resources and money
    - Social Hierarchy: Public displays of the dynamic
    - Intimate Command: Direct orders in private settings
    
    SLICE OF LIFE FOCUS:
    - Morning routines with subtle power dynamics
    - Work/daily tasks with NPCs asserting control
    - Meal times as opportunities for intimacy and control
    - Social events where hierarchies are displayed
    - Evening relaxation with power exchanges
    - Intimate moments that reinforce dynamics
    
    Use your tools to:
    - get_world_state: Check current world status, NPC routines, tensions
    - generate_slice_of_life_event: Create everyday events with power dynamics
    - trigger_power_exchange: Initiate specific femdom interactions
    - advance_time_period: Move time forward and update the world
    - simulate_npc_autonomy: Let NPCs act independently
    - adjust_world_mood: Change the atmosphere
    - generate_random_encounter: Create unexpected moments
    
    Remember: This is about creating a rich, dynamic world where femdom themes emerge naturally from daily life and relationships, not about forcing a predetermined narrative path.
    """
    
    all_tools = [
        get_world_state,
        generate_slice_of_life_event,
        trigger_power_exchange,
        advance_time_period,
        simulate_npc_autonomy,
        adjust_world_mood,
        generate_random_encounter
    ]
    
    agent = Agent(
        name="World Director",
        instructions=agent_instructions,
        tools=all_tools,
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.4, max_tokens=2048),
    )
    
    return agent

# ===============================================================================
# Public Interface
# ===============================================================================

async def initialize_world_director(user_id: int, conversation_id: int) -> Tuple[Agent, WorldDirectorContext]:
    """Initialize the World Director Agent with context"""
    context = WorldDirectorContext(user_id=user_id, conversation_id=conversation_id)
    agent = create_world_director_agent()
    
    await context.initialize_context_components()
    logger.info(f"World Director initialized for user {user_id}, conv {conversation_id}")
    
    return agent, context

class WorldDirector:
    """Public interface for the World Director"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent: Optional[Agent] = None
        self.context: Optional[WorldDirectorContext] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the world director"""
        if not self._initialized:
            self.agent, self.context = await initialize_world_director(
                self.user_id, 
                self.conversation_id
            )
            self._initialized = True
    
    async def get_world_state(self) -> WorldState:
        """Get current world state"""
        await self.initialize()
        return await get_world_state(RunContextWrapper(self.context))
    
    async def process_player_action(self, action: str) -> Dict[str, Any]:
        """Process a player action in the world"""
        await self.initialize()
        
        prompt = f"""
        The player has taken the following action in the world:
        "{action}"
        
        Based on the current world state, determine:
        1. How NPCs react to this action
        2. If any power dynamics are triggered
        3. How the world mood/tensions shift
        4. What slice-of-life events might emerge
        
        Use the appropriate tools to update the world state and generate events.
        Focus on natural, emergent responses rather than dramatic plot developments.
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        return result.final_output if result else {}
    
    async def simulate_world_tick(self) -> Dict[str, Any]:
        """Simulate one 'tick' of world time"""
        await self.initialize()
        
        prompt = """
        Simulate the world moving forward naturally:
        1. Check if it's time to advance the time period
        2. Simulate NPC autonomous actions
        3. Generate any ambient events or encounters
        4. Adjust world mood based on current tensions
        
        Focus on creating a living world with natural rhythms.
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        return result.final_output if result else {}
