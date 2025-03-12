# logic/time_cycle_conflict_integration.py

"""
Time Cycle Integration for the Dynamic Conflict System

This module extends the time_cycle.py module to integrate the conflict system.
"""

import logging
import random
from typing import Dict, Any, List, Optional

from logic.conflict_system.conflict_integration import ConflictSystemIntegration

logger = logging.getLogger(__name__)

async def process_conflict_time_advancement(user_id: int, conversation_id: int, activity_type: str) -> Dict[str, Any]:
    """
    Process time advancement for conflicts based on player activity.
    This should be called when time advances in the game.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        activity_type: Type of activity that caused time to advance
        
    Returns:
        Dict with results of conflict processing
    """
    # Initialize the conflict system
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    
    # Update player vitals based on activity type
    vitals_result = await conflict_system.update_player_vitals(activity_type)
    
    # Process the effects of the activity on conflicts
    result = {
        "vitals_updated": vitals_result,
        "conflicts_updated": 0,
        "daily_update_run": False,
        "player_analysis": {}
    }
    
    # Get current active conflicts
    active_conflicts = await conflict_system.get_active_conflicts()
    
    # Process each active conflict
    for conflict in active_conflicts:
        # Determine progress increment based on activity type
        progress_increment = calculate_progress_increment(
            activity_type, conflict.get("conflict_type", "standard")
        )
        
        # Only update if there's a meaningful increment
        if progress_increment >= 1:
            await conflict_system.update_progress(conflict["conflict_id"], progress_increment)
            result["conflicts_updated"] += 1
    
    # Check if this was a night -> morning transition (new day)
    is_new_day = activity_type == "sleep" and result.get("vitals_updated", {}).get("time_of_day") == "Morning"
    
    # If it's a new day, run the daily conflict update
    if is_new_day:
        daily_result = await conflict_system.run_daily_update()
        result["daily_update"] = daily_result
        result["daily_update_run"] = True
    
    # Return the results
    return result

def calculate_progress_increment(activity_type: str, conflict_type: str) -> float:
    """
    Calculate how much a specific activity should progress a conflict.
    
    Args:
        activity_type: Type of activity
        conflict_type: Type of conflict
        
    Returns:
        Float representing progress increment
    """
    # Base progress increments by activity type
    base_increments = {
        "standard": 2,
        "intense": 5,
        "restful": 0.5,
        "eating": 0,
        "sleep": 10,  # Larger increment for sleep as it advances to next day
        "work_shift": 3,
        "class_attendance": 2,
        "social_event": 3,
        "training": 4,
        "extended_conversation": 3,
        "personal_time": 1
    }
    
    # Get base increment value, default to 1 if not found
    base_value = base_increments.get(activity_type, 1)
    
    # Adjust based on conflict type
    type_multipliers = {
        "major": 0.5,        # Major conflicts progress more slowly
        "minor": 0.8,        # Minor conflicts progress more slowly than standard
        "standard": 1.0,     # Standard conflicts progress at the base rate
        "catastrophic": 0.25  # Catastrophic conflicts progress very slowly
    }
    
    # Get type multiplier, default to 1.0 if not found
    type_multiplier = type_multipliers.get(conflict_type, 1.0)
    
    # Add slight randomness
    randomness = random.uniform(0.8, 1.2)
    
    # Calculate final increment
    return base_value * type_multiplier * randomness

async def process_day_end_conflicts(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Process conflicts at the end of the day.
    This should be called when the player sleeps or the day ends.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        
    Returns:
        Dict with results of end-of-day processing
    """
    # Initialize the conflict system
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    
    # Get current active conflicts
    active_conflicts = await conflict_system.get_active_conflicts()
    
    result = {
        "active_conflicts": len(active_conflicts),
        "conflicts_updated": 0,
        "conflicts_resolved": 0,
        "phase_changes": 0
    }
    
    # Process each active conflict
    for conflict in active_conflicts:
        # Apply end-of-day progress increment
        progress_increment = 5 * random.uniform(0.8, 1.2)  # Base 5% progress with randomness
        
        # Adjust based on conflict type
        if conflict.get("conflict_type") == "major":
            progress_increment *= 0.5
        elif conflict.get("conflict_type") == "minor":
            progress_increment *= 0.8
        elif conflict.get("conflict_type") == "catastrophic":
            progress_increment *= 0.3
        
        # Update the conflict progress
        updated_conflict = await conflict_system.update_progress(conflict["conflict_id"], progress_increment)
        result["conflicts_updated"] += 1
        
        # Check if phase changed
        if updated_conflict["phase"] != conflict["phase"]:
            result["phase_changes"] += 1
        
        # Check if conflict should be resolved
        if updated_conflict["progress"] >= 100 and updated_conflict["phase"] == "resolution":
            await conflict_system.resolve_conflict(conflict["conflict_id"])
            result["conflicts_resolved"] += 1
    
    # Update player vitals for sleep
    vitals_result = await conflict_system.update_player_vitals("sleep")
    result["vitals_updated"] = vitals_result
    
    # Return the results
    return result

async def check_for_conflict_events(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
    """
    Check for conflict-related events that might occur regardless of player action.
    This can be called periodically to generate spontaneous events.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        
    Returns:
        List of conflict events
    """
    # Initialize the conflict system
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    
    # Get current active conflicts
    active_conflicts = await conflict_system.get_active_conflicts()
    
    # If no active conflicts, return empty list
    if not active_conflicts:
        return []
    
    events = []
    
    # Random chance for events based on conflicts
    for conflict in active_conflicts:
        # Only generate events for active or climax phase
        if conflict["phase"] not in ["active", "climax"]:
            continue
        
        # 15% chance of an event per conflict
        if random.random() < 0.15:
            # Generate event based on conflict
            event = await generate_conflict_event(conflict_system, conflict)
            if event:
                events.append(event)
    
    return events

async def generate_conflict_event(
    conflict_system: ConflictSystemIntegration, 
    conflict: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Generate a spontaneous event for a conflict.
    
    Args:
        conflict_system: ConflictSystemIntegration instance
        conflict: Conflict data
        
    Returns:
        Event data or None
    """
    # Get full conflict details
    conflict_id = conflict["conflict_id"]
    conflict_details = await conflict_system.get_conflict_details(conflict_id)
    
    if not conflict_details:
        return None
    
    # Determine event type
    event_types = [
        "faction_activity",
        "npc_request",
        "resource_opportunity",
        "unexpected_development"
    ]
    
    event_type = random.choice(event_types)
    event = {"type": event_type, "conflict_id": conflict_id, "conflict_name": conflict["conflict_name"]}
    
    # Generate event based on type
    if event_type == "faction_activity":
        # Choose a faction
        faction = random.choice(["a", "b"])
        faction_name = conflict_details["faction_a_name"] if faction == "a" else conflict_details["faction_b_name"]
        
        # Generate activity description
        activities = [
            f"{faction_name} is gathering resources for the conflict.",
            f"{faction_name} is recruiting new members to their cause.",
            f"{faction_name} is spreading propaganda against their opponents.",
            f"{faction_name} is fortifying their position in the conflict.",
            f"{faction_name} is making a strategic move in the conflict."
        ]
        
        event["description"] = random.choice(activities)
        event["faction"] = faction
        event["faction_name"] = faction_name
        
        # Small progress increment for the conflict
        await conflict_system.update_progress(conflict_id, 2)
    
    elif event_type == "npc_request":
        # Get an NPC involved in the conflict
        involved_npcs = conflict_details.get("involved_npcs", [])
        if involved_npcs:
            npc = random.choice(involved_npcs)
            npc_name = npc.get("npc_name", "an NPC")
            npc_faction = npc.get("faction", "neutral")
            
            # Generate request description
            requests = [
                f"{npc_name} asks for your help in the conflict.",
                f"{npc_name} wants to discuss strategy with you.",
                f"{npc_name} requests resources for the conflict effort.",
                f"{npc_name} needs your expertise for a critical task.",
                f"{npc_name} seeks your opinion on a difficult decision."
            ]
            
            event["description"] = random.choice(requests)
            event["npc_id"] = npc.get("npc_id")
            event["npc_name"] = npc_name
            event["npc_faction"] = npc_faction
        else:
            # No NPCs to request, skip this event
            return None
    
    elif event_type == "resource_opportunity":
        # Generate opportunity description
        opportunities = [
            "You've discovered a source of valuable supplies for the conflict.",
            "A potential ally has offered their support in exchange for a favor.",
            "You've learned of a hidden cache of resources that could turn the tide.",
            "An opportunity to gain intelligence on the opposing faction has emerged.",
            "A chance to secure additional influence for your faction has appeared."
        ]
        
        event["description"] = random.choice(opportunities)
        
        # Generate resource details
        resource_types = ["money", "supplies", "influence"]
        resource_type = random.choice(resource_types)
        resource_amount = random.randint(10, 50)
        
        event["resource_type"] = resource_type
        event["resource_amount"] = resource_amount
        event["expiration"] = 2  # Expires in 2 game days
    
    elif event_type == "unexpected_development":
        # Generate development description
        developments = [
            "An unexpected betrayal has shifted the balance of power.",
            "A natural disaster has affected the conflict area.",
            "A neutral third party has decided to intervene.",
            "Public opinion has suddenly shifted regarding the conflict.",
            "A crucial piece of information has been revealed to all parties."
        ]
        
        event["description"] = random.choice(developments)
        
        # Add a significant progress increment to the conflict
        progress_increment = random.randint(5, 15)
        await conflict_system.update_progress(conflict_id, progress_increment)
        
        event["progress_impact"] = progress_increment
    
    # Record the event as a conflict memory
    await conflict_system.conflict_manager._create_conflict_memory(
        conflict_id, 
        f"Event: {event['description']}",
        6  # Significance level
    )
    
    return event

async def integrate_conflict_with_time_module(
    user_id: int, conversation_id: int, activity_type: str, description: str
) -> Dict[str, Any]:
    """
    Main integration function to process conflict system alongside time advancement.
    This is the primary function to call from the time_cycle module.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        activity_type: Type of activity
        description: Description of the activity
        
    Returns:
        Dict with integrated results
    """
    # Initialize the conflict system
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    
    # Process time advancement for conflicts
    time_result = await process_conflict_time_advancement(user_id, conversation_id, activity_type)
    
    # Process the specific activity's impact on conflicts
    activity_result = await conflict_system.process_activity_for_conflict_impact(activity_type, description)
    
    # Check for spontaneous conflict events (20% chance)
    events = []
    if random.random() < 0.2:
        events = await check_for_conflict_events(user_id, conversation_id)
    
    # Also check if the narrative in the description might generate a new conflict
    narrative_result = None
    if len(description) > 20:  # Only process substantial descriptions
        narrative_result = await conflict_system.add_conflict_to_narrative(description)
    
    # Combine all results
    result = {
        "time_advancement": time_result,
        "activity_impact": activity_result,
        "conflict_events": events,
        "narrative_analysis": narrative_result
    }
    
    return result
