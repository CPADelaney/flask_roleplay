# logic/time_cycle.py
"""
Unified Time Cycle & Conflict System Module in an Agentic Framework

This module merges:
  - time_cycle.py
  - enhanced_time_cycle.py
  - time_cycle_conflict_integration.py

All integrated as a single "TimeCycleAgent" using the OpenAI Agents SDK.
Updated to use NPC-specific narrative progression and comprehensive vitals system.
"""

import logging
import random
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from agents import Agent, Runner, function_tool
from agents.run_context import RunContextWrapper

from db.connection import get_db_connection_context
import asyncpg
from lore.core import canon

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Merged Constants and Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12
TIME_PHASES = ["Morning", "Afternoon", "Evening", "Night"]
TIME_PRIORITY = {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4}

# Enhanced activity definitions with stat and vital effects
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

# Vital drain rates per time period
VITAL_DRAIN_RATES = {
    "Morning": {
        "hunger": 3,
        "thirst": 4,
        "fatigue": 2
    },
    "Afternoon": {
        "hunger": 4,
        "thirst": 6,  # Higher in afternoon
        "fatigue": 3
    },
    "Evening": {
        "hunger": 5,  # Dinner time
        "thirst": 4,
        "fatigue": 4
    },
    "Night": {
        "hunger": 2,
        "thirst": 2,
        "fatigue": 6  # Get tired faster at night
    }
}

# Vital thresholds and their effects
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

SPECIAL_EVENT_CHANCES = {
    "personal_revelation": 0.2,
    "narrative_moment": 0.15,
    "relationship_crossroads": 0.1,
    "relationship_ritual": 0.1,
    "dream_sequence": 0.4,
    "moment_of_clarity": 0.25,
    "mask_slippage": 0.3,
    "npc_revelation": 0.25,
    "vital_crisis": 0.3  # New: chance for hunger/thirst/fatigue events
}

logger = logging.getLogger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Enhanced Vitals System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                       pv.hunger, pv.thirst, pv.fatigue
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
            current_hunger = data['hunger'] if data['hunger'] is not None else 100
            current_thirst = data['thirst'] if data['thirst'] is not None else 100
            current_fatigue = data['fatigue'] if data['fatigue'] is not None else 0
            
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
            new_hunger = max(0, current_hunger - hunger_drain)
            new_thirst = max(0, current_thirst - thirst_drain)
            new_fatigue = min(100, current_fatigue + fatigue_gain)
            
            # Update vitals
            await conn.execute("""
                INSERT INTO PlayerVitals (user_id, conversation_id, player_name, hunger, thirst, fatigue)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, conversation_id, player_name)
                DO UPDATE SET 
                    hunger = $4, 
                    thirst = $5, 
                    fatigue = $6,
                    last_update = CURRENT_TIMESTAMP
            """, user_id, conversation_id, player_name, new_hunger, new_thirst, new_fatigue)
            
            # Apply stat penalties based on vital thresholds
            stat_changes = await calculate_vital_stat_effects(
                new_hunger, new_thirst, new_fatigue
            )
            
            # Check for vital crises
            crises = []
            if new_hunger < 20:
                crises.append({
                    "type": "hunger_crisis",
                    "severity": "severe" if new_hunger < 10 else "moderate",
                    "message": "You're dangerously hungry. Find food soon!"
                })
            
            if new_thirst < 20:
                crises.append({
                    "type": "thirst_crisis", 
                    "severity": "severe" if new_thirst < 10 else "moderate",
                    "message": "You're severely dehydrated. You need water!"
                })
            
            if new_fatigue > 80:
                crises.append({
                    "type": "fatigue_crisis",
                    "severity": "severe" if new_fatigue > 90 else "moderate",
                    "message": "You're about to collapse from exhaustion!"
                })
            
            return {
                "success": True,
                "old_vitals": {
                    "hunger": current_hunger,
                    "thirst": current_thirst,
                    "fatigue": current_fatigue
                },
                "new_vitals": {
                    "hunger": new_hunger,
                    "thirst": new_thirst,
                    "fatigue": new_fatigue
                },
                "drains": {
                    "hunger": hunger_drain,
                    "thirst": thirst_drain,
                    "fatigue": fatigue_gain
                },
                "stat_effects": stat_changes,
                "crises": crises,
                "forced_sleep": new_fatigue >= 100
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
                SELECT hunger, thirst, fatigue FROM PlayerVitals
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not current:
                # Initialize vitals if missing
                await conn.execute("""
                    INSERT INTO PlayerVitals (user_id, conversation_id, player_name, hunger, thirst, fatigue)
                    VALUES ($1, $2, $3, 100, 100, 0)
                """, user_id, conversation_id, player_name)
                current = {"hunger": 100, "thirst": 100, "fatigue": 0}
            
            # Apply effects with intensity modifier
            new_vitals = {}
            for vital, change in vital_effects.items():
                current_value = current[vital]
                modified_change = int(change * intensity)
                new_value = max(0, min(100, current_value + modified_change))
                new_vitals[vital] = new_value
            
            # Update vitals
            await conn.execute("""
                UPDATE PlayerVitals
                SET hunger = $1, thirst = $2, fatigue = $3, last_update = CURRENT_TIMESTAMP
                WHERE user_id = $4 AND conversation_id = $5 AND player_name = $6
            """, 
                new_vitals.get("hunger", current["hunger"]),
                new_vitals.get("thirst", current["thirst"]),
                new_vitals.get("fatigue", current["fatigue"]),
                user_id, conversation_id, player_name
            )
            
            return {
                "success": True,
                "vital_changes": {
                    vital: new_vitals.get(vital, current[vital]) - current[vital]
                    for vital in ["hunger", "thirst", "fatigue"]
                },
                "new_vitals": new_vitals
            }
            
    except Exception as e:
        logger.error(f"Error processing activity vitals: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic DB Helpers (unchanged)
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
            """, user_id, conversation_id, current_year,
                current_year, current_month,
                current_year, current_month, current_day,
                current_year, current_month, current_day,
                current_priority)
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
        logger.error(f"Database error removing expired events: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error removing expired events: {e}", exc_info=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Base Time Logic (unchanged except for adding vitals update)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                await canon.update_current_roleplay(canon_ctx, conn, user_id, conversation_id, key, val)
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
    """
    (new_year, new_month, new_day, new_phase) = await advance_time(user_id, conversation_id, increment)
    await update_npc_schedules_for_time(user_id, conversation_id, new_day, new_phase)
    await remove_expired_planned_events(user_id, conversation_id, new_year, new_month, new_day, new_phase)
    
    # Update vitals based on time passage
    vitals_result = await update_vitals_from_time(
        user_id, conversation_id, "Chase", increment, new_phase
    )
    
    return (new_year, new_month, new_day, new_phase), vitals_result

def should_advance_time(activity_type):
    """
    Returns { "should_advance": bool, "periods": int } indicating if the activity advances time.
    """
    if activity_type in TIME_CONSUMING_ACTIVITIES:
        return {"should_advance": True, "periods": TIME_CONSUMING_ACTIVITIES[activity_type]["time_advance"]}
    if activity_type in OPTIONAL_ACTIVITIES:
        return {"should_advance": False, "periods": 0}
    return {"should_advance": False, "periods": 0}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Higher-level Function: advance_time_with_events (ENHANCED)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def advance_time_with_events(user_id: int, conversation_id: int, activity_type: str) -> Dict[str, Any]:
    """
    Advances time (if the activity is time-consuming), triggers special events, 
    updates player stats and vitals.
    
    Enhanced with comprehensive vitals management.
    """
    # Update imports to use new NPC creation system and NPC-specific progression
    from npcs.new_npc_creation import NPCCreationHandler, RunContextWrapper
    from logic.narrative_events import (
        get_relationship_overview,
        check_for_personal_revelations,
        check_for_narrative_moments,
        add_dream_sequence,
        add_moment_of_clarity
    )
    from logic.npc_narrative_progression import (
        get_npc_narrative_stage,
        check_for_npc_revelation
    )
    from logic.social_links import check_for_relationship_crossroads, check_for_relationship_ritual
    from logic.stats_logic import apply_stat_changes

    # Create NPCCreationHandler instance
    npc_handler = NPCCreationHandler()

    try:
        # Get current time_of_day to see if we need to do anything
        _, _, _, current_time_of_day = await get_current_time(user_id, conversation_id)

        adv_info = should_advance_time(activity_type)
        
        # Process activity effects on vitals first (even if time doesn't advance)
        vitals_result = await process_activity_vitals(
            user_id, conversation_id, "Chase", activity_type
        )
        
        if not adv_info["should_advance"]:
            # Apply stat effects from activity even without time advance
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
                "vitals_updated": vitals_result
            }

        periods_to_advance = adv_info["periods"]
        time_result = await advance_time_and_update(user_id, conversation_id, increment=periods_to_advance)
        (new_year, new_month, new_day, new_time), time_vitals_result = time_result

        events = []
        
        # Check for vital crises
        if time_vitals_result.get("crises"):
            for crisis in time_vitals_result["crises"]:
                events.append({
                    "type": "vital_crisis",
                    "crisis_type": crisis["type"],
                    "severity": crisis["severity"],
                    "message": crisis["message"]
                })
        
        # Handle forced sleep from exhaustion
        if time_vitals_result.get("forced_sleep"):
            events.append({
                "type": "forced_event",
                "event": "sleep",
                "reason": "exhaustion",
                "message": "You collapse from exhaustion and fall into a deep sleep."
            })
            # Recursively call with sleep activity
            return await advance_time_with_events(user_id, conversation_id, "sleep")
        
        # Use the new NPC system for daily activities and stage changes
        ctx = RunContextWrapper({
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        # Process daily activities
        await npc_handler.process_daily_npc_activities(ctx, new_time)
        
        # Detect relationship stages
        await npc_handler.detect_relationship_stage_changes(ctx)

        # Get relationship overview instead of single narrative stage
        relationship_overview = await get_relationship_overview(user_id, conversation_id)
        if relationship_overview and relationship_overview.get('total_relationships', 0) > 0:
            events.append({
                "type": "relationship_overview",
                "total_relationships": relationship_overview['total_relationships'],
                "stage_distribution": relationship_overview['stage_distribution'],
                "most_advanced": relationship_overview['most_advanced_npcs'][:1]  # Just the top one
            })

        # Random checks for various events (with vital state modifiers)
        hunger = vitals_result.get("new_vitals", {}).get("hunger", 100)
        thirst = vitals_result.get("new_vitals", {}).get("thirst", 100)
        fatigue = vitals_result.get("new_vitals", {}).get("fatigue", 0)
        
        # Modify event chances based on vitals
        event_chance_modifiers = {
            "personal_revelation": 1.0 if fatigue < 50 else 0.5,
            "narrative_moment": 1.0,
            "relationship_crossroads": 1.0 if hunger > 40 and thirst > 40 else 0.7,
            "dream_sequence": 1.5 if fatigue > 70 else 1.0,
            "moment_of_clarity": 1.2 if hunger > 60 and thirst > 60 and fatigue < 40 else 0.8,
        }
        
        # Check for special events with modified chances
        if random.random() < SPECIAL_EVENT_CHANCES["personal_revelation"] * event_chance_modifiers.get("personal_revelation", 1.0):
            revelation = await check_for_personal_revelations(user_id, conversation_id)
            if revelation:
                events.append(revelation)

        if random.random() < SPECIAL_EVENT_CHANCES["narrative_moment"] * event_chance_modifiers.get("narrative_moment", 1.0):
            moment = await check_for_narrative_moments(user_id, conversation_id)
            if moment:
                events.append(moment)

        if random.random() < SPECIAL_EVENT_CHANCES["relationship_crossroads"] * event_chance_modifiers.get("relationship_crossroads", 1.0):
            crossroads = await check_for_relationship_crossroads(user_id, conversation_id)
            if crossroads:
                events.append(crossroads)

        if random.random() < SPECIAL_EVENT_CHANCES["relationship_ritual"]:
            ritual = await check_for_relationship_ritual(user_id, conversation_id)
            if ritual:
                events.append(ritual)

        if activity_type == "sleep" and random.random() < SPECIAL_EVENT_CHANCES["dream_sequence"] * event_chance_modifiers.get("dream_sequence", 1.0):
            dream = await add_dream_sequence(user_id, conversation_id)
            if dream:
                events.append(dream)

        if random.random() < SPECIAL_EVENT_CHANCES["moment_of_clarity"] * event_chance_modifiers.get("moment_of_clarity", 1.0):
            clarity = await add_moment_of_clarity(user_id, conversation_id)
            if clarity:
                events.append(clarity)

        # Update mask slippage check to use the new NPCCreationHandler
        if random.random() < SPECIAL_EVENT_CHANCES["mask_slippage"]:
            async with get_db_connection_context() as conn:
                npc_row = await conn.fetchrow("""
                    SELECT npc_id FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                    ORDER BY RANDOM() LIMIT 1
                """, user_id, conversation_id)
                
                if npc_row:
                    npc_id = npc_row["npc_id"]
                    # Use the new NPCCreationHandler for mask slippage
                    slip_events = await npc_handler.check_for_mask_slippage(user_id, conversation_id, npc_id)
                    if slip_events:
                        events.append({
                            "type": "mask_slippage",
                            "npc_id": npc_id,
                            "events": slip_events
                        })

        # Check for NPC-specific revelations for multiple NPCs
        if random.random() < SPECIAL_EVENT_CHANCES["npc_revelation"]:
            async with get_db_connection_context() as conn:
                # Get NPCs who aren't in the innocent beginning stage
                npc_rows = await conn.fetch("""
                    SELECT np.npc_id 
                    FROM NPCNarrativeProgression np
                    WHERE np.user_id=$1 AND np.conversation_id=$2
                    AND np.narrative_stage != 'Innocent Beginning'
                    ORDER BY RANDOM() LIMIT 3
                """, user_id, conversation_id)
                
                for npc_row in npc_rows:
                    npc_id = npc_row["npc_id"]
                    npc_rev = await check_for_npc_revelation(user_id, conversation_id, npc_id)
                    if npc_rev:
                        events.append(npc_rev)
                        break  # Only one revelation per time period

        # Stat effects from activity
        if activity_type in TIME_CONSUMING_ACTIVITIES:
            stat_changes = TIME_CONSUMING_ACTIVITIES[activity_type].get("stat_effects", {})
            if stat_changes and not stat_changes.get("varies"):
                # Apply stat changes
                await apply_stat_changes(
                    user_id, conversation_id, "Chase", stat_changes,
                    f"Activity: {activity_type}"
                )

        return {
            "time_advanced": True,
            "new_year": new_year,
            "new_month": new_month,
            "new_day": new_day,
            "new_time": new_time,
            "events": events,
            "vitals_result": {
                "time_drain": time_vitals_result,
                "activity_effects": vitals_result
            }
        }

    except Exception as e:
        logger.error(f"Error in advance_time_with_events: {e}", exc_info=True)
        return {"time_advanced": False, "error": str(e)}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3) Nightly Maintenance (enhanced with vitals reset)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def nightly_maintenance(user_id: int, conversation_id: int):
    """
    Called typically when day increments. Fades NPC memories and performs vitals maintenance.
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
# 4) Conflict Integration (unchanged)
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
    new_year, new_month, new_day, new_time = await get_current_time(user_id, conversation_id)
    if activity_type == "sleep" and new_time == "Morning":
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
# 5) Classification Helper (enhanced with more activities)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def classify_player_input(input_text: str) -> str:
    """
    An enhanced classifier for the activity type based on user text.
    """
    lower = input_text.lower()
    
    # Sleep/rest activities
    if any(w in lower for w in ["sleep", "rest", "go to bed", "lie down", "nap"]):
        return "sleep"
    
    # Work/education activities
    if any(w in lower for w in ["class", "lecture", "study", "school", "learn"]):
        return "class_attendance"
    if any(w in lower for w in ["work", "job", "shift", "office"]):
        return "work_shift"
    
    # Social activities
    if any(w in lower for w in ["party", "event", "gathering", "hang out", "socialize"]):
        return "social_event"
    if any(w in lower for w in ["talk to", "speak with", "discuss with", "conversation", "chat with"]):
        return "extended_conversation"
    
    # Physical activities
    if any(w in lower for w in ["train", "practice", "workout", "exercise", "lift", "run"]):
        return "training"
    
    # Consumption activities
    if any(w in lower for w in ["eat", "meal", "lunch", "dinner", "breakfast", "food"]):
        return "eating"
    if any(w in lower for w in ["drink", "water", "thirsty", "beverage"]):
        return "drinking"
    
    # Personal activities
    if any(w in lower for w in ["relax", "chill", "personal time", "by myself", "alone"]):
        return "personal_time"
    
    # Quick activities
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
    
    # Intense activities
    if any(w in lower for w in ["intense", "extreme", "exhausting", "grueling"]):
        return "intense_activity"
    
    # Default to quick chat
    return "quick_chat"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6) Agentic Merged: "TimeCycleAgent" (with new tools)
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
async def tool_advance_time_with_events(ctx: RunContextWrapper[TimeCycleContext], activity_type: str) -> str:
    """
    Advance time if needed, process special events, update vitals, and return a JSON summary of results.
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    result = await advance_time_with_events(user_id, conv_id, activity_type)
    return json.dumps(result)

@function_tool
async def tool_check_vitals(ctx: RunContextWrapper[TimeCycleContext]) -> str:
    """
    Check current player vitals (hunger, thirst, fatigue).
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    
    try:
        async with get_db_connection_context() as conn:
            vitals = await conn.fetchrow("""
                SELECT hunger, thirst, fatigue FROM PlayerVitals
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
            """, user_id, conv_id)
            
            if vitals:
                result = {
                    "hunger": vitals['hunger'],
                    "thirst": vitals['thirst'],
                    "fatigue": vitals['fatigue'],
                    "status": {
                        "hunger": _get_vital_status(vitals['hunger'], "hunger"),
                        "thirst": _get_vital_status(vitals['thirst'], "thirst"),
                        "fatigue": _get_vital_status(vitals['fatigue'], "fatigue")
                    }
                }
            else:
                result = {"error": "No vitals found"}
            
            return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def _get_vital_status(value: int, vital_type: str) -> str:
    """Helper to get vital status description."""
    if vital_type in ["hunger", "thirst"]:
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

@function_tool
async def tool_nightly_maintenance(ctx: RunContextWrapper[TimeCycleContext]) -> str:
    """
    Run nightly memory fade, summarization, etc. Return a summary of operations.
    """
    user_id = ctx.context.user_id
    conv_id = ctx.context.conversation_id
    await nightly_maintenance(user_id, conv_id)
    return "Nightly maintenance completed."

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
                SELECT hunger, thirst FROM PlayerVitals
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

# Enhanced Agent with vitals awareness

AGENT_INSTRUCTIONS = """
You are the TimeCycleAgent. You manage in-game time, vitals (hunger/thirst/fatigue), 
daily maintenance, conflict integration, and special events.

You have the following tools at your disposal:
- tool_advance_time_with_events: Advance time and process events/vitals
- tool_check_vitals: Check current hunger/thirst/fatigue levels
- tool_consume_vital_resource: Consume food/water to restore vitals
- tool_nightly_maintenance: Run end-of-day maintenance
- tool_process_conflict_time_advancement: Handle conflict time progression
- tool_integrate_conflict_with_time_module: Full conflict-time integration

When the user provides an activity or command:
1. Classify the activity type
2. Check if vitals are critically low (suggest eating/drinking/resting)
3. Process the activity with appropriate time advancement
4. Report vital changes and any special events

Monitor for vital crises and forced events (like collapsing from exhaustion).
"""

TimeCycleAgent = Agent[TimeCycleContext](
    name="TimeCycleAgent",
    instructions=AGENT_INSTRUCTIONS,
    tools=[
        tool_advance_time_with_events,
        tool_check_vitals,
        tool_consume_vital_resource,
        tool_nightly_maintenance,
        tool_process_conflict_time_advancement,
        tool_integrate_conflict_with_time_module
    ],
    # You could also add guardrails or more advanced instructions here.
)

async def register_with_governance(user_id: int, conversation_id: int):
    """Register time cycle agent with Nyx governance system."""
    from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
    from nyx.integrate import get_central_governance
    
    governor = await get_central_governance(user_id, conversation_id)
    await governor.register_agent(
        agent_type=AgentType.UNIVERSAL_UPDATER,  # Or create a specific TIME_MANAGER type
        agent_instance=TimeCycleAgent,
        agent_id="time_cycle"
    )
    # Issue directive for time management
    await governor.issue_directive(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="time_cycle",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Manage time advancement, vitals, and associated events",
            "scope": "game"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60  # 24 hours
    )
    logging.info("Time cycle agent registered with Nyx governance")

# Add the ActivityManager class (unchanged from before)

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
