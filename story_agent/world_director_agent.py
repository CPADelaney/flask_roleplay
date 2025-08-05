# story_agent/world_director_agent.py
"""
World Dynamics Director - Managing an open-ended femdom slice-of-life simulation.
REFACTORED: Integrated with existing game systems
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

from db.connection import get_db_connection_context
from agents import Agent, function_tool, Runner, trace, ModelSettings, RunContextWrapper

# ===============================================================================
# INTEGRATED SYSTEM IMPORTS
# ===============================================================================

# Time and Calendar Systems
from logic.time_cycle import (
    get_current_time_model,
    advance_time_with_events,
    get_current_vitals,
    VitalsData,
    CurrentTimeData,
    TimeOfDay as TimeOfDayOriginal,
    process_activity_vitals,
    ActivityManager
)
from logic.calendar import (
    load_calendar_names,
    update_calendar_names
)

# Dynamic Relationships System
from logic.dynamic_relationships import (
    OptimizedRelationshipManager, 
    event_generator,
    drain_relationship_events_tool,
    RelationshipState,
    RelationshipDimensions,
    RelationshipPatternDetector,
    RelationshipArchetypes
)

# Narrative Events System  
from logic.narrative_events import (
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence,
    add_moment_of_clarity,
    get_relationship_overview
)

# Event System
from logic.event_system import EventSystem

# Addiction System
from logic.addiction_system_sdk import (
    AddictionContext,
    process_addiction_update,
    check_addiction_status,
    get_addiction_status
)

# Currency System
from logic.currency_generator import CurrencyGenerator

# NPC Systems
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    check_for_npc_revelation,
    NPC_NARRATIVE_STAGES
)

# Context system integration
from context.context_service import get_context_service
from context.context_config import get_config
from context.memory_manager import get_memory_manager, MemoryAddRequest
from context.models import MemoryMetadata

# Nyx governance
from nyx.governance_helpers import with_governance
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType

logger = logging.getLogger(__name__)

# ===============================================================================
# Enhanced World State Models
# ===============================================================================

class WorldMood(Enum):
    """Overall mood/atmosphere of the world"""
    RELAXED = "relaxed"
    TENSE = "tense"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    MYSTERIOUS = "mysterious"
    OPPRESSIVE = "oppressive"
    CHAOTIC = "chaotic"
    EXHAUSTED = "exhausted"  # Added based on vitals
    DESPERATE = "desperate"   # Added for crisis states

class ActivityType(Enum):
    """Types of slice-of-life activities"""
    WORK = "work"
    SOCIAL = "social"
    LEISURE = "leisure"
    INTIMATE = "intimate"
    ROUTINE = "routine"
    SPECIAL = "special"
    ADDICTION = "addiction"  # Added for addiction-related events
    VITAL = "vital"         # Added for hunger/thirst/rest events

class PowerDynamicType(Enum):
    """Types of femdom power dynamics"""
    SUBTLE_CONTROL = "subtle_control"
    CASUAL_DOMINANCE = "casual_dominance"
    PROTECTIVE_CONTROL = "protective_control"
    PLAYFUL_TEASING = "playful_teasing"
    RITUAL_SUBMISSION = "ritual_submission"
    FINANCIAL_CONTROL = "financial_control"
    SOCIAL_HIERARCHY = "social_hierarchy"
    INTIMATE_COMMAND = "intimate_command"
    ADDICTION_EXPLOITATION = "addiction_exploitation"  # Added

class SliceOfLifeEvent(BaseModel):
    """Enhanced slice-of-life event with system integration"""
    event_id: str
    event_type: ActivityType
    title: str
    description: str
    participants: List[int]  # NPC IDs
    location: str
    power_dynamic: Optional[PowerDynamicType] = None
    mood_impact: Optional[WorldMood] = None
    relationship_impacts: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    addiction_triggers: Dict[str, float] = Field(default_factory=dict)  # Added
    vital_costs: Dict[str, int] = Field(default_factory=dict)  # Added
    currency_cost: Optional[int] = None  # Added
    can_interrupt: bool = True
    priority: int = 5
    
    model_config = ConfigDict(extra="forbid")

class WorldTension(BaseModel):
    """Enhanced tension tracking"""
    social_tension: float = 0.0
    sexual_tension: float = 0.0
    power_tension: float = 0.0
    mystery_tension: float = 0.0
    conflict_tension: float = 0.0
    addiction_tension: float = 0.0  # Added
    vital_tension: float = 0.0      # Added
    
    def get_dominant_tension(self) -> Tuple[str, float]:
        tensions = {
            "social": self.social_tension,
            "sexual": self.sexual_tension,
            "power": self.power_tension,
            "mystery": self.mystery_tension,
            "conflict": self.conflict_tension,
            "addiction": self.addiction_tension,
            "vital": self.vital_tension
        }
        return max(tensions.items(), key=lambda x: x[1])
    
    model_config = ConfigDict(extra="forbid")

class WorldState(BaseModel):
    """Enhanced world state with integrated systems"""
    # Time using existing system
    current_time: CurrentTimeData
    calendar_names: Dict[str, Any] = Field(default_factory=dict)
    
    # Vitals from existing system
    player_vitals: VitalsData
    
    # Mood and atmosphere
    world_mood: WorldMood
    world_tension: WorldTension = Field(default_factory=WorldTension)
    
    # NPCs and relationships
    active_npcs: List[Dict[str, Any]] = Field(default_factory=list)
    relationship_summary: Optional[Dict[str, Any]] = None
    
    # Events
    ongoing_events: List[SliceOfLifeEvent] = Field(default_factory=list)
    available_activities: List[SliceOfLifeEvent] = Field(default_factory=list)
    pending_relationship_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Addictions
    addiction_status: Dict[str, Any] = Field(default_factory=dict)
    
    # Currency
    currency_system: Dict[str, Any] = Field(default_factory=dict)
    player_money: int = 0
    
    # Environmental factors
    environmental_factors: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Enhanced World Director Context
# ===============================================================================

@dataclass
class WorldDirectorContext:
    """Enhanced context with integrated systems"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    
    # Integrated system managers
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    event_system: Optional[EventSystem] = None
    addiction_context: Optional[AddictionContext] = None
    currency_generator: Optional[CurrencyGenerator] = None
    activity_manager: Optional[ActivityManager] = None
    
    # State tracking
    current_world_state: Optional[WorldState] = None
    last_time_update: Optional[datetime] = None
    
    # Context management
    context_service: Optional[Any] = None
    memory_manager: Optional[Any] = None
    directive_handler: Optional[DirectiveHandler] = None
    
    # Caching
    cache: Dict[str, Any] = field(default_factory=dict)
    calendar_names: Optional[Dict[str, Any]] = None

    async def initialize_context_components(self):
        """Initialize all integrated systems"""
        # Initialize relationship manager
        self.relationship_manager = OptimizedRelationshipManager(
            self.user_id, self.conversation_id
        )
        
        # Initialize event system
        self.event_system = EventSystem(self.user_id, self.conversation_id)
        await self.event_system.initialize()
        
        # Initialize addiction context
        self.addiction_context = AddictionContext(self.user_id, self.conversation_id)
        await self.addiction_context.initialize()
        
        # Initialize currency generator
        self.currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
        
        # Initialize activity manager
        self.activity_manager = ActivityManager()
        
        # Load calendar names for immersive time references
        self.calendar_names = await load_calendar_names(
            self.user_id, self.conversation_id
        )
        
        # Initialize context service
        self.context_service = await get_context_service(self.user_id, self.conversation_id)
        self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        # Initialize world state
        if self.current_world_state is None:
            self.current_world_state = await self._initialize_world_state()
        
        logger.info(f"World Director context initialized with all systems")

    async def _initialize_world_state(self) -> WorldState:
        """Initialize world state using integrated systems"""
        # Get current time from time_cycle
        current_time = await get_current_time_model(self.user_id, self.conversation_id)
        
        # Get current vitals
        vitals = await get_current_vitals(self.user_id, self.conversation_id)
        
        # Get relationship overview
        rel_overview = await get_relationship_overview(self.user_id, self.conversation_id)
        
        # Get addiction status
        addiction_status = await get_addiction_status(
            self.user_id, self.conversation_id, self.player_name
        )
        
        # Get currency system
        currency_system = await self.currency_generator.get_currency_system()
        
        # Determine initial mood based on vitals and relationships
        world_mood = self._calculate_mood_from_state(vitals, rel_overview)
        
        # Calculate tensions
        world_tension = await self._calculate_initial_tensions(
            vitals, rel_overview, addiction_status
        )
        
        return WorldState(
            current_time=current_time,
            calendar_names=self.calendar_names or {},
            player_vitals=vitals,
            world_mood=world_mood,
            world_tension=world_tension,
            relationship_summary=rel_overview,
            addiction_status=addiction_status,
            currency_system=currency_system,
            player_money=100  # Default starting money
        )
    
    def _calculate_mood_from_state(
        self, 
        vitals: VitalsData, 
        rel_overview: Dict[str, Any]
    ) -> WorldMood:
        """Calculate world mood from game state"""
        # Check vitals first
        if vitals.fatigue > 80:
            return WorldMood.EXHAUSTED
        if vitals.hunger < 20 or vitals.thirst < 20:
            return WorldMood.DESPERATE
        
        # Then check relationships
        avg_corruption = rel_overview.get('aggregate_stats', {}).get('average_corruption', 0)
        if avg_corruption < 20:
            return WorldMood.RELAXED
        elif avg_corruption < 40:
            return WorldMood.PLAYFUL
        elif avg_corruption < 60:
            return WorldMood.MYSTERIOUS
        else:
            return WorldMood.INTIMATE
    
    async def _calculate_initial_tensions(
        self,
        vitals: VitalsData,
        rel_overview: Dict[str, Any],
        addiction_status: Dict[str, Any]
    ) -> WorldTension:
        """Calculate initial world tensions"""
        tensions = WorldTension()
        
        # Vital-based tensions
        if vitals.hunger < 40 or vitals.thirst < 40:
            tensions.vital_tension = 0.7
        if vitals.fatigue > 60:
            tensions.vital_tension = max(tensions.vital_tension, 0.5)
        
        # Addiction-based tensions
        if addiction_status.get("has_addictions"):
            addiction_count = len(addiction_status.get("addictions", {}))
            tensions.addiction_tension = min(1.0, addiction_count * 0.2)
        
        # Relationship-based tensions
        avg_corruption = rel_overview.get('aggregate_stats', {}).get('average_corruption', 0)
        avg_dependency = rel_overview.get('aggregate_stats', {}).get('average_dependency', 0)
        
        tensions.sexual_tension = min(1.0, avg_corruption / 100.0)
        tensions.power_tension = min(1.0, avg_dependency / 100.0)
        tensions.social_tension = min(1.0, len(rel_overview.get('relationships', [])) * 0.1)
        
        return tensions

# ===============================================================================
# Integrated Tool Functions
# ===============================================================================

@function_tool
async def get_world_state(ctx: RunContextWrapper[WorldDirectorContext]) -> WorldState:
    """Get current world state using all integrated systems"""
    context = ctx.context
    
    if not context.current_world_state:
        await context.initialize_context_components()
    
    # Update from integrated systems
    world_state = context.current_world_state
    
    # Update time
    world_state.current_time = await get_current_time_model(
        context.user_id, context.conversation_id
    )
    
    # Update vitals
    world_state.player_vitals = await get_current_vitals(
        context.user_id, context.conversation_id
    )
    
    # Update relationships
    world_state.relationship_summary = await get_relationship_overview(
        context.user_id, context.conversation_id
    )
    
    # Check for pending relationship events
    rel_events = await drain_relationship_events_tool(
        ctx=RunContextWrapper({
            'user_id': context.user_id,
            'conversation_id': context.conversation_id
        }),
        max_events=5
    )
    if rel_events.get('events'):
        world_state.pending_relationship_events = rel_events['events']
    
    # Update addiction status
    world_state.addiction_status = await get_addiction_status(
        context.user_id, context.conversation_id, context.player_name
    )
    
    # Update tensions based on current state
    world_state.world_tension = await _update_world_tensions(context, world_state)
    
    # Update mood
    world_state.world_mood = context._calculate_mood_from_state(
        world_state.player_vitals,
        world_state.relationship_summary
    )
    
    # Generate available activities
    world_state.available_activities = await _generate_integrated_activities(
        context, world_state
    )
    
    world_state.last_updated = datetime.now(timezone.utc)
    context.current_world_state = world_state
    
    return world_state

@function_tool
async def advance_time_period_integrated(
    ctx: RunContextWrapper[WorldDirectorContext],
    activity_type: Optional[str] = None
) -> Dict[str, Any]:
    """Advance time using the integrated time_cycle system"""
    context = ctx.context
    
    # Use the time_cycle system for advancement
    result = await advance_time_with_events(
        context.user_id,
        context.conversation_id,
        activity_type or "personal_time"
    )
    
    # Update world state based on time advancement
    if result['time_advanced']:
        # Update context's current time
        context.current_world_state.current_time = await get_current_time_model(
            context.user_id, context.conversation_id
        )
        
        # Map game time to world mood
        new_time = result['new_time']
        context.current_world_state.world_mood = await _calculate_mood_for_integrated_time(
            context, new_time
        )
        
        # Process any events from time advancement
        for event in result.get('events', []):
            if event['type'] == 'vital_crisis':
                # Create a vital event
                await _create_vital_event(context, event)
            elif event['type'] == 'relationship_summary':
                # Update relationship dynamics
                await _update_relationship_dynamics_from_summary(context, event)
    
    # Check and drain relationship events
    rel_events = await drain_relationship_events_tool(
        ctx=RunContextWrapper({
            'user_id': context.user_id,
            'conversation_id': context.conversation_id
        }),
        max_events=5
    )
    
    if rel_events.get('events'):
        for event_data in rel_events['events']:
            # Convert relationship events to world events
            await _process_relationship_event_for_world(context, event_data)
    
    return result

@function_tool
async def generate_integrated_slice_of_life_event(
    ctx: RunContextWrapper[WorldDirectorContext],
    event_type: Optional[str] = None,
    consider_addictions: bool = True,
    consider_vitals: bool = True
) -> SliceOfLifeEvent:
    """Generate events considering all integrated systems"""
    context = ctx.context
    world_state = context.current_world_state or await get_world_state(ctx)
    
    # Priority checks
    if consider_vitals and _needs_vital_event(world_state.player_vitals):
        return await _generate_vital_priority_event(context, world_state)
    
    if consider_addictions and world_state.addiction_status.get('has_addictions'):
        if random.random() < 0.3:  # 30% chance for addiction event
            return await _generate_addiction_event(context, world_state)
    
    # Check for narrative moments
    narrative_moment = await check_for_narrative_moments(
        context.user_id, context.conversation_id
    )
    if narrative_moment:
        return await _convert_narrative_to_event(context, narrative_moment)
    
    # Generate regular slice of life event
    if not event_type:
        # Weight based on current tensions
        dominant_tension, level = world_state.world_tension.get_dominant_tension()
        if dominant_tension == "addiction" and level > 0.5:
            event_type = ActivityType.ADDICTION
        elif dominant_tension == "vital" and level > 0.5:
            event_type = ActivityType.VITAL
        else:
            event_type = _select_activity_by_time(world_state.current_time)
    else:
        event_type = ActivityType[event_type.upper()]
    
    # Select participating NPCs based on relationships
    involved_npcs = await _select_npcs_by_relationship_stage(context, world_state)
    
    # Generate event with integrated details
    event = await _generate_full_integrated_event(
        context, world_state, event_type, involved_npcs
    )
    
    # Add to ongoing events
    world_state.ongoing_events.append(event)
    
    # Process through event system
    await context.event_system.create_event(
        event_type=event.event_type.value,
        event_data={
            "event_id": event.event_id,
            "description": event.description,
            "participants": event.participants
        },
        priority=event.priority
    )
    
    return event

@function_tool
async def process_activity_with_integration(
    ctx: RunContextWrapper[WorldDirectorContext],
    player_input: str
) -> Dict[str, Any]:
    """Process player activity using integrated systems"""
    context = ctx.context
    
    # Use activity manager to process
    activity_result = await context.activity_manager.process_activity(
        context.user_id,
        context.conversation_id,
        player_input,
        {"vitals": context.current_world_state.player_vitals.to_dict()}
    )
    
    activity_type = activity_result['activity_type']
    intensity = activity_result.get('intensity', 1.0)
    
    # Process vitals
    vitals_result = await process_activity_vitals(
        context.user_id,
        context.conversation_id,
        context.player_name,
        activity_type,
        intensity
    )
    
    # Check for addiction triggers
    addiction_results = []
    if activity_type in ['social_event', 'intimate_activity', 'extended_conversation']:
        # Get involved NPCs
        involved_npcs = await _get_npcs_in_current_scene(context)
        for npc_id in involved_npcs:
            # Random chance to trigger addiction progression
            if random.random() < 0.2 * intensity:
                addiction_type = random.choice(['dependency', 'submission', 'fascination'])
                result = await process_addiction_update(
                    context.user_id,
                    context.conversation_id,
                    context.player_name,
                    addiction_type,
                    intensity,
                    npc_id
                )
                addiction_results.append(result)
    
    # Update relationship dynamics based on activity
    relationship_updates = []
    if activity_type in ['extended_conversation', 'intimate_activity']:
        for npc_id in await _get_npcs_in_current_scene(context):
            state = await context.relationship_manager.get_relationship_state(
                'player', 1, 'npc', npc_id
            )
            # Process interaction
            interaction_result = await context.relationship_manager.process_interaction(
                'player', 1, 'npc', npc_id,
                {'type': activity_type, 'intensity': intensity}
            )
            relationship_updates.append(interaction_result)
    
    return {
        "activity": activity_result,
        "vitals": vitals_result,
        "addictions": addiction_results,
        "relationships": relationship_updates
    }

@function_tool
async def check_for_emergent_events(
    ctx: RunContextWrapper[WorldDirectorContext]
) -> List[Dict[str, Any]]:
    """Check all systems for emergent events"""
    context = ctx.context
    emergent_events = []
    
    # Check for personal revelations
    revelation = await check_for_personal_revelations(
        context.user_id, context.conversation_id
    )
    if revelation:
        emergent_events.append({
            "type": "revelation",
            "data": revelation
        })
    
    # Check for NPC revelations
    npcs = await _get_active_npcs(context)
    for npc in npcs:
        npc_revelation = await check_for_npc_revelation(
            context.user_id, context.conversation_id, npc['npc_id']
        )
        if npc_revelation:
            emergent_events.append({
                "type": "npc_revelation",
                "data": npc_revelation
            })
    
    # Check event system
    active_events = await context.event_system.get_active_events()
    for event in active_events:
        if event['priority'] >= 7:  # High priority events
            emergent_events.append({
                "type": "system_event",
                "data": event
            })
    
    # Check for critical vitals
    vitals = context.current_world_state.player_vitals
    if vitals.hunger < 10 or vitals.thirst < 10 or vitals.fatigue > 95:
        emergent_events.append({
            "type": "vital_crisis",
            "data": {
                "hunger": vitals.hunger,
                "thirst": vitals.thirst,
                "fatigue": vitals.fatigue
            }
        })
    
    return emergent_events

@function_tool
async def handle_currency_transaction(
    ctx: RunContextWrapper[WorldDirectorContext],
    amount: int,
    reason: str,
    npc_id: Optional[int] = None
) -> Dict[str, Any]:
    """Handle currency transactions with proper formatting"""
    context = ctx.context
    
    # Format currency
    formatted_amount = await context.currency_generator.format_currency(abs(amount))
    
    old_money = context.current_world_state.player_money
    new_money = max(0, old_money + amount)
    context.current_world_state.player_money = new_money
    
    # Format new balance
    formatted_balance = await context.currency_generator.format_currency(new_money)
    
    # If NPC involved, might affect power dynamics
    if npc_id and amount < 0:  # Player spending/giving money
        # This could trigger financial control dynamics
        if random.random() < 0.3:
            power_exchange = {
                "type": PowerDynamicType.FINANCIAL_CONTROL,
                "npc_id": npc_id,
                "description": f"Controls your spending on {reason}"
            }
            context.current_world_state.world_tension.power_tension += 0.1
    
    return {
        "transaction": "debit" if amount < 0 else "credit",
        "amount": formatted_amount,
        "reason": reason,
        "old_balance": old_money,
        "new_balance": new_money,
        "formatted_balance": formatted_balance
    }

# ===============================================================================
# Helper Functions for Integration
# ===============================================================================

async def _update_world_tensions(
    context: WorldDirectorContext,
    world_state: WorldState
) -> WorldTension:
    """Update tensions based on all systems"""
    tensions = world_state.world_tension
    
    # Vital tensions
    vitals = world_state.player_vitals
    vital_crisis_level = 0
    if vitals.hunger < 30:
        vital_crisis_level += (30 - vitals.hunger) / 30
    if vitals.thirst < 30:
        vital_crisis_level += (30 - vitals.thirst) / 30
    if vitals.fatigue > 70:
        vital_crisis_level += (vitals.fatigue - 70) / 30
    tensions.vital_tension = min(1.0, vital_crisis_level / 2)
    
    # Addiction tensions
    if world_state.addiction_status.get('has_addictions'):
        max_level = 0
        for addiction_data in world_state.addiction_status.get('addictions', {}).values():
            max_level = max(max_level, addiction_data.get('level', 0))
        tensions.addiction_tension = min(1.0, max_level / 4.0)
    
    # Relationship tensions from dynamic system
    if world_state.relationship_summary:
        for rel in world_state.relationship_summary.get('relationships', []):
            # Check for patterns that increase tension
            if 'explosive_chemistry' in rel.get('patterns', []):
                tensions.sexual_tension = min(1.0, tensions.sexual_tension + 0.1)
            if 'toxic_bond' in rel.get('archetypes', []):
                tensions.conflict_tension = min(1.0, tensions.conflict_tension + 0.15)
    
    return tensions

def _needs_vital_event(vitals: VitalsData) -> bool:
    """Check if vitals require immediate attention"""
    return (vitals.hunger < 20 or 
            vitals.thirst < 20 or 
            vitals.fatigue > 85 or
            vitals.energy < 20)

async def _generate_vital_priority_event(
    context: WorldDirectorContext,
    world_state: WorldState
) -> SliceOfLifeEvent:
    """Generate a vital-priority event"""
    vitals = world_state.player_vitals
    
    if vitals.hunger < 20:
        title = "Desperate Hunger"
        description = "You need food immediately"
        vital_type = "hunger"
    elif vitals.thirst < 20:
        title = "Severe Dehydration"
        description = "You desperately need water"
        vital_type = "thirst"
    elif vitals.fatigue > 85:
        title = "Exhaustion Takes Over"
        description = "You can barely keep your eyes open"
        vital_type = "fatigue"
    else:
        title = "Energy Depleted"
        description = "You need to rest and recover"
        vital_type = "energy"
    
    return SliceOfLifeEvent(
        event_id=f"vital_{vital_type}_{int(time.time())}",
        event_type=ActivityType.VITAL,
        title=title,
        description=description,
        participants=[],
        location="current",
        mood_impact=WorldMood.DESPERATE,
        priority=9,
        can_interrupt=False
    )

async def _generate_addiction_event(
    context: WorldDirectorContext,
    world_state: WorldState
) -> SliceOfLifeEvent:
    """Generate an addiction-related event"""
    addictions = world_state.addiction_status.get('addictions', {})
    if not addictions:
        return None
    
    # Pick the strongest addiction
    strongest = max(addictions.items(), key=lambda x: x[1].get('level', 0))
    addiction_key, addiction_data = strongest
    
    # Generate event based on addiction type
    if addiction_data.get('type') == 'npc_specific':
        npc_id = addiction_data.get('npc_id')
        npc_name = addiction_data.get('npc_name', 'Someone')
        title = f"Craving {npc_name}'s Presence"
        description = f"You feel a deep need related to {npc_name}"
        participants = [npc_id] if npc_id else []
    else:
        title = f"Overwhelming {addiction_key.title()} Craving"
        description = f"The addiction to {addiction_key} demands attention"
        participants = []
    
    return SliceOfLifeEvent(
        event_id=f"addiction_{addiction_key}_{int(time.time())}",
        event_type=ActivityType.ADDICTION,
        title=title,
        description=description,
        participants=participants,
        location="mind",
        power_dynamic=PowerDynamicType.ADDICTION_EXPLOITATION,
        mood_impact=WorldMood.TENSE,
        addiction_triggers={addiction_key: 0.2},
        priority=7,
        can_interrupt=True
    )

async def _convert_narrative_to_event(
    context: WorldDirectorContext,
    narrative_moment: Dict[str, Any]
) -> SliceOfLifeEvent:
    """Convert a narrative moment to a world event"""
    return SliceOfLifeEvent(
        event_id=f"narrative_{narrative_moment.get('type')}_{int(time.time())}",
        event_type=ActivityType.SPECIAL,
        title=narrative_moment.get('name', 'Narrative Moment'),
        description=narrative_moment.get('scene_text', ''),
        participants=[],
        location="mindscape",
        mood_impact=WorldMood.MYSTERIOUS,
        priority=8,
        can_interrupt=False
    )

def _select_activity_by_time(current_time: CurrentTimeData) -> ActivityType:
    """Select activity type based on time of day"""
    time_str = current_time.time_of_day
    
    if time_str in ["Morning", "Afternoon"]:
        return random.choice([ActivityType.WORK, ActivityType.ROUTINE])
    elif time_str == "Evening":
        return random.choice([ActivityType.SOCIAL, ActivityType.LEISURE])
    else:  # Night
        return random.choice([ActivityType.INTIMATE, ActivityType.SPECIAL])

async def _select_npcs_by_relationship_stage(
    context: WorldDirectorContext,
    world_state: WorldState
) -> List[int]:
    """Select NPCs based on relationship stages"""
    if not world_state.relationship_summary:
        return []
    
    relationships = world_state.relationship_summary.get('relationships', [])
    
    # Prioritize NPCs with higher corruption/dependency
    sorted_rels = sorted(
        relationships, 
        key=lambda r: r.get('corruption', 0) + r.get('dependency', 0),
        reverse=True
    )
    
    # Select top 1-3
    selected = []
    for rel in sorted_rels[:3]:
        if rel.get('npc_id'):
            selected.append(rel['npc_id'])
    
    return selected

async def _generate_full_integrated_event(
    context: WorldDirectorContext,
    world_state: WorldState,
    event_type: ActivityType,
    involved_npcs: List[int]
) -> SliceOfLifeEvent:
    """Generate a fully integrated event"""
    # Base event details
    event_id = f"{event_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Get NPC details for personalization
    npc_names = []
    power_dynamic = None
    
    if involved_npcs:
        async with get_db_connection_context() as conn:
            for npc_id in involved_npcs[:2]:  # Limit to 2 for readability
                npc = await conn.fetchrow("""
                    SELECT npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2
                """, npc_id, context.user_id)
                if npc:
                    npc_names.append(npc['npc_name'])
                    # Determine power dynamic based on dominance
                    if npc['dominance'] > 70:
                        power_dynamic = PowerDynamicType.CASUAL_DOMINANCE
                    elif npc['dominance'] > 50:
                        power_dynamic = PowerDynamicType.SUBTLE_CONTROL
    
    # Generate title and description
    if event_type == ActivityType.WORK:
        title = f"Work Time"
        description = "Time to focus on tasks and responsibilities"
        vital_costs = {"energy": -10, "fatigue": 10}
    elif event_type == ActivityType.SOCIAL:
        if npc_names:
            title = f"Socializing with {' and '.join(npc_names)}"
        else:
            title = "Social Gathering"
        description = "A chance to interact and build connections"
        vital_costs = {"energy": -5, "thirst": -10}
    elif event_type == ActivityType.INTIMATE:
        if npc_names:
            title = f"Private Time with {npc_names[0]}"
        else:
            title = "Intimate Moment"
        description = "A moment of closeness and vulnerability"
        vital_costs = {"energy": -15, "fatigue": 5}
        if power_dynamic is None:
            power_dynamic = PowerDynamicType.INTIMATE_COMMAND
    else:
        title = f"{event_type.value.title()} Activity"
        description = f"An opportunity for {event_type.value}"
        vital_costs = {"energy": -5}
    
    # Check for addiction triggers
    addiction_triggers = {}
    if event_type in [ActivityType.INTIMATE, ActivityType.SOCIAL] and involved_npcs:
        # These activities might trigger addictions
        addiction_triggers["dependency"] = 0.1
        if event_type == ActivityType.INTIMATE:
            addiction_triggers["submission"] = 0.15
    
    # Determine currency cost
    currency_cost = None
    if event_type in [ActivityType.SOCIAL, ActivityType.LEISURE]:
        currency_cost = random.randint(10, 50)
    
    return SliceOfLifeEvent(
        event_id=event_id,
        event_type=event_type,
        title=title,
        description=description,
        participants=involved_npcs,
        location=await _determine_location(context, event_type),
        power_dynamic=power_dynamic,
        mood_impact=world_state.world_mood,
        vital_costs=vital_costs,
        addiction_triggers=addiction_triggers,
        currency_cost=currency_cost,
        priority=5,
        can_interrupt=True
    )

async def _determine_location(
    context: WorldDirectorContext,
    event_type: ActivityType
) -> str:
    """Determine location for an event type"""
    location_map = {
        ActivityType.WORK: "workplace",
        ActivityType.SOCIAL: "social venue",
        ActivityType.LEISURE: "relaxation spot",
        ActivityType.INTIMATE: "private space",
        ActivityType.ROUTINE: "home",
        ActivityType.VITAL: "nearest resource",
        ActivityType.ADDICTION: "mindspace",
        ActivityType.SPECIAL: "varies"
    }
    return location_map.get(event_type, "current location")

async def _get_active_npcs(context: WorldDirectorContext) -> List[Dict[str, Any]]:
    """Get currently active NPCs"""
    async with get_db_connection_context() as conn:
        npcs = await conn.fetch("""
            SELECT npc_id, npc_name, current_location, dominance, intensity
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2
            AND introduced = true
            LIMIT 10
        """, context.user_id, context.conversation_id)
    
    return [dict(npc) for npc in npcs]

async def _get_npcs_in_current_scene(context: WorldDirectorContext) -> List[int]:
    """Get NPCs in the current scene"""
    # This would check current location and active NPCs
    # For now, return a sample
    active_npcs = await _get_active_npcs(context)
    return [npc['npc_id'] for npc in active_npcs[:3]]

async def _calculate_mood_for_integrated_time(
    context: WorldDirectorContext,
    time_of_day: str
) -> WorldMood:
    """Calculate mood based on integrated time system"""
    world_state = context.current_world_state
    
    # Check vitals first
    if world_state.player_vitals.fatigue > 80:
        return WorldMood.EXHAUSTED
    if world_state.player_vitals.hunger < 20 or world_state.player_vitals.thirst < 20:
        return WorldMood.DESPERATE
    
    # Base mood by time
    time_moods = {
        "Morning": WorldMood.RELAXED,
        "Afternoon": WorldMood.TENSE,
        "Evening": WorldMood.PLAYFUL,
        "Night": WorldMood.INTIMATE
    }
    
    return time_moods.get(time_of_day, WorldMood.RELAXED)

async def _create_vital_event(
    context: WorldDirectorContext,
    event_data: Dict[str, Any]
):
    """Create a vital crisis event"""
    event = SliceOfLifeEvent(
        event_id=f"vital_crisis_{int(time.time())}",
        event_type=ActivityType.VITAL,
        title="Vital Crisis",
        description=event_data.get('message', 'Critical vital levels'),
        participants=[],
        location="current",
        mood_impact=WorldMood.DESPERATE,
        priority=10,
        can_interrupt=False
    )
    context.current_world_state.ongoing_events.append(event)

async def _update_relationship_dynamics_from_summary(
    context: WorldDirectorContext,
    event_data: Dict[str, Any]
):
    """Update world state from relationship summary"""
    # Update tensions based on active patterns
    for pattern in event_data.get('active_patterns', []):
        if pattern == 'explosive_chemistry':
            context.current_world_state.world_tension.sexual_tension += 0.1
        elif pattern == 'toxic_bond':
            context.current_world_state.world_tension.conflict_tension += 0.1
    
    # Clamp tensions
    for attr in ['sexual_tension', 'conflict_tension']:
        val = getattr(context.current_world_state.world_tension, attr)
        setattr(context.current_world_state.world_tension, attr, min(1.0, val))

async def _process_relationship_event_for_world(
    context: WorldDirectorContext,
    event_data: Dict[str, Any]
):
    """Process a relationship event into world state"""
    event_info = event_data.get('event', {})
    
    # Create a special event
    event = SliceOfLifeEvent(
        event_id=f"rel_event_{int(time.time())}",
        event_type=ActivityType.SPECIAL,
        title=event_info.get('title', 'Relationship Development'),
        description=str(event_info),
        participants=[],  # Could extract NPC IDs from state_key
        location="emotional_space",
        mood_impact=WorldMood.INTIMATE,
        priority=7,
        can_interrupt=False
    )
    
    context.current_world_state.ongoing_events.append(event)

async def _generate_integrated_activities(
    context: WorldDirectorContext,
    world_state: WorldState
) -> List[SliceOfLifeEvent]:
    """Generate available activities using all systems"""
    activities = []
    
    # Vital-based activities if needed
    if world_state.player_vitals.hunger < 50:
        activities.append(SliceOfLifeEvent(
            event_id=f"eat_{int(time.time())}",
            event_type=ActivityType.VITAL,
            title="Get Food",
            description="Find something to eat",
            participants=[],
            location="kitchen or restaurant",
            vital_costs={"hunger": 40},
            currency_cost=20,
            priority=6,
            can_interrupt=True
        ))
    
    if world_state.player_vitals.thirst < 50:
        activities.append(SliceOfLifeEvent(
            event_id=f"drink_{int(time.time())}",
            event_type=ActivityType.VITAL,
            title="Get Water",
            description="Hydrate yourself",
            participants=[],
            location="anywhere",
            vital_costs={"thirst": 40},
            priority=6,
            can_interrupt=True
        ))
    
    # Time-based regular activities
    time_str = world_state.current_time.time_of_day
    if time_str in ["Morning", "Afternoon"]:
        activities.append(SliceOfLifeEvent(
            event_id=f"work_{int(time.time())}",
            event_type=ActivityType.WORK,
            title="Work Session",
            description="Focus on work tasks",
            participants=[],
            location="workplace",
            vital_costs={"energy": -10, "fatigue": 15},
            priority=4,
            can_interrupt=True
        ))
    
    # Relationship-based activities
    if world_state.relationship_summary:
        for rel in world_state.relationship_summary.get('relationships', [])[:3]:
            npc_id = rel.get('npc_id')
            npc_name = rel.get('npc_name', 'Someone')
            
            activities.append(SliceOfLifeEvent(
                event_id=f"social_{npc_id}_{int(time.time())}",
                event_type=ActivityType.SOCIAL,
                title=f"Spend time with {npc_name}",
                description=f"Interact with {npc_name}",
                participants=[npc_id] if npc_id else [],
                location="varies",
                vital_costs={"energy": -5},
                priority=5,
                can_interrupt=True
            ))
    
    return activities

# ===============================================================================
# Main Agent Creation
# ===============================================================================

def create_integrated_world_director_agent():
    """Create the World Director Agent with full system integration"""
    
    agent_instructions = """
    You are the World Dynamics Director for an open-ended femdom slice-of-life simulation
    with fully integrated game systems.
    
    INTEGRATED SYSTEMS YOU MANAGE:
    
    1. TIME & VITALS (from time_cycle):
       - Use advance_time_period_integrated for time progression
       - Monitor hunger, thirst, fatigue, energy
       - Generate vital crises when needed
       - Use calendar names for immersion
    
    2. DYNAMIC RELATIONSHIPS (from dynamic_relationships):
       - Multi-dimensional relationship tracking
       - Patterns (push_pull, slow_burn, explosive_chemistry, etc.)
       - Archetypes (soulmates, toxic_bond, rivals, etc.)
       - Momentum and drift over time
    
    3. ADDICTION SYSTEM:
       - Track player addictions to NPCs and behaviors
       - Generate addiction-related events
       - Exploitation of addictions by dominant NPCs
    
    4. NARRATIVE EVENTS:
       - Personal revelations about the player's situation
       - NPC revelations as relationships deepen
       - Narrative moments and dream sequences
    
    5. CURRENCY SYSTEM:
       - Use setting-appropriate currency
       - Financial control dynamics
       - Resource management
    
    6. EVENT SYSTEM:
       - Central event processing and prioritization
       - Event propagation across systems
       - Emergent event generation
    
    KEY PRINCIPLES:
    - Balance all systems: Don't let one dominate
    - Vital needs create urgency and vulnerability
    - Addictions create dependency and control opportunities
    - Relationships evolve dynamically with patterns
    - Currency enables financial control dynamics
    - Events emerge from system interactions
    
    SLICE OF LIFE WITH DEPTH:
    - Daily routines hide power dynamics
    - Vital needs create opportunities for control
    - Addictions are exploited subtly
    - Financial dependency develops naturally
    - Relationships follow complex patterns
    
    Your tools:
    - get_world_state: Full integrated world state
    - advance_time_period_integrated: Time with all systems
    - generate_integrated_slice_of_life_event: Events considering all systems
    - process_activity_with_integration: Activity processing through all systems
    - check_for_emergent_events: Check all systems for emergent narratives
    - handle_currency_transaction: Financial interactions
    
    Remember: The depth comes from system integration. A simple meal becomes:
    - Vital need satisfaction (hunger)
    - Potential addiction trigger (if with certain NPC)
    - Relationship interaction opportunity
    - Financial control moment (who pays?)
    - Power dynamic display (who chooses what to eat?)
    """
    
    all_tools = [
        get_world_state,
        advance_time_period_integrated,
        generate_integrated_slice_of_life_event,
        process_activity_with_integration,
        check_for_emergent_events,
        handle_currency_transaction
    ]
    
    agent = Agent(
        name="Integrated World Director",
        instructions=agent_instructions,
        tools=all_tools,
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.4, max_tokens=2048),
    )
    
    return agent

# ===============================================================================
# Public Interface
# ===============================================================================

class IntegratedWorldDirector:
    """Public interface for the Integrated World Director"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent: Optional[Agent] = None
        self.context: Optional[WorldDirectorContext] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize with all integrated systems"""
        if not self._initialized:
            self.context = WorldDirectorContext(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            await self.context.initialize_context_components()
            self.agent = create_integrated_world_director_agent()
            self._initialized = True
            logger.info(f"Integrated World Director initialized for user {self.user_id}")
    
    async def get_world_state(self) -> WorldState:
        """Get current integrated world state"""
        await self.initialize()
        return await get_world_state(RunContextWrapper(self.context))
    
    async def process_player_action(self, action: str) -> Dict[str, Any]:
        """Process player action through all systems"""
        await self.initialize()
        
        # First process through activity system
        activity_result = await process_activity_with_integration(
            RunContextWrapper(self.context),
            action
        )
        
        # Then let the agent handle narrative consequences
        prompt = f"""
        The player performed: "{action}"
        
        Activity processing results: {json.dumps(activity_result, default=str)}
        
        Based on these results and the current world state:
        1. Generate appropriate slice-of-life events
        2. Check for emergent narrative moments
        3. Process any addiction or relationship triggers
        4. Update world mood and tensions
        
        Focus on natural consequences that emerge from the integrated systems.
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        
        return {
            "activity_result": activity_result,
            "world_response": result.messages[-1].content if result else None
        }
    
    async def advance_world_time(self) -> Dict[str, Any]:
        """Advance world time with all systems"""
        await self.initialize()
        
        prompt = """
        Advance the world time appropriately:
        1. Use advance_time_period_integrated
        2. Check for emergent events from all systems
        3. Generate slice-of-life events if appropriate
        4. Update world state based on time passage
        
        Consider:
        - Vital drain over time
        - Relationship drift
        - Addiction cravings
        - NPC autonomous actions
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        return result.messages[-1].content if result else {}
    
    async def get_available_activities(self) -> List[SliceOfLifeEvent]:
        """Get activities available to player"""
        await self.initialize()
        world_state = await self.get_world_state()
        return world_state.available_activities
