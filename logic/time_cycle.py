# logic.time_cycle.py

# time_cycle.py - Merged Time Management Module

import random
import json
import logging
from datetime import datetime
from db.connection import get_db_connection
from logic.npc_creation import (
    process_daily_npc_activities,
    check_for_mask_slippage,
    detect_relationship_stage_changes
)
from logic.narrative_progression import (
    get_current_narrative_stage,
    check_for_personal_revelations,
    check_for_narrative_moments,
    check_for_npc_revelations,
    add_dream_sequence,
    add_moment_of_clarity
)
from logic.social_links import (
    check_for_relationship_crossroads,
    check_for_relationship_ritual
)

from logic.npc_agents.memory_manager import EnhancedMemoryManager  # <--- NEW IMPORT

# Define constants
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12  # Adjust if needed

# Define your phases (this remains the same)
TIME_PHASES = ["Morning", "Afternoon", "Evening", "Night"]

TIME_PRIORITY = {
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4
}

# Define time-consuming activities that should advance time
TIME_CONSUMING_ACTIVITIES = {
    # Key activities that would advance time in the game
    "class_attendance": {
        "time_advance": 1,  # Advances time by 1 period (e.g., Morning → Afternoon)
        "description": "Attending classes or lectures",
        "stat_effects": {"mental_resilience": +2}
    },
    "work_shift": {
        "time_advance": 1,
        "description": "Working at a job",
        "stat_effects": {"physical_endurance": +2}
    },
    "social_event": {
        "time_advance": 1,
        "description": "Attending a social gathering",
        "stat_effects": {"confidence": +1}
    },
    "training": {
        "time_advance": 1,
        "description": "Physical or mental training",
        "stat_effects": {"willpower": +2}
    },
    "extended_conversation": {
        "time_advance": 1,
        "description": "A lengthy, significant conversation",
        "stat_effects": {}  # Effects would depend on the conversation
    },
    "personal_time": {
        "time_advance": 1,
        "description": "Spending time on personal activities",
        "stat_effects": {}  # Effects would depend on the activity
    },
    "sleep": {
        "time_advance": 2,  # Sleeping advances time from Evening → Morning (skipping Night)
        "description": "Going to sleep for the night",
        "stat_effects": {"physical_endurance": +3, "mental_resilience": +3}
    }
}

# Optional activities that can occur during time periods
OPTIONAL_ACTIVITIES = {
    "quick_chat": {
        "time_advance": 0,  # Doesn't advance time
        "description": "A brief conversation",
        "stat_effects": {}
    },
    "observe": {
        "time_advance": 0,
        "description": "Observing surroundings or people",
        "stat_effects": {}
    },
    "check_phone": {
        "time_advance": 0,
        "description": "Looking at messages or notifications",
        "stat_effects": {}
    }
}

# Define chances for special events based on time advancement
SPECIAL_EVENT_CHANCES = {
    "personal_revelation": 0.2,  # 20% chance per time advancement
    "narrative_moment": 0.15,    # 15% chance per time advancement
    "relationship_crossroads": 0.1, # 10% chance per time advancement
    "relationship_ritual": 0.1,  # 10% chance per time advancement
    "dream_sequence": 0.4,       # 40% chance when sleeping
    "moment_of_clarity": 0.25,   # 25% chance per time advancement
    "mask_slippage": 0.3,        # 30% chance per time advancement
}

def remove_expired_planned_events(user_id, conversation_id, current_year, current_month, current_day, current_phase):
    """
    Delete planned events whose scheduled time has passed.
    This function assumes that an event is considered expired if its date/phase
    is earlier than the current in-game time.
    """
    current_priority = TIME_PRIORITY.get(current_phase, 0)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM PlannedEvents
        WHERE user_id=%s AND conversation_id=%s AND (
            (year < %s)
            OR (year = %s AND month < %s)
            OR (year = %s AND month = %s AND day < %s)
            OR (year = %s AND month = %s AND day = %s AND 
                (CASE time_of_day
                    WHEN 'Morning' THEN 1
                    WHEN 'Afternoon' THEN 2
                    WHEN 'Evening' THEN 3
                    WHEN 'Night' THEN 4
                    ELSE 0
                END) < %s)
        )
    """, (user_id, conversation_id, current_year,
          current_year, current_month,
          current_year, current_month, current_day,
          current_year, current_month, current_day, current_priority))
    conn.commit()
    conn.close()

def get_current_time(user_id, conversation_id):
    """
    Returns a tuple (current_year, current_month, current_day, time_of_day) for the given user_id and conversation_id.
    Falls back to defaults: year=1, month=1, day=1, phase="Morning".
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get CurrentYear
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='CurrentYear'
    """, (user_id, conversation_id))
    row_year = cursor.fetchone()
    current_year = int(row_year[0]) if row_year else 1

    # Get CurrentMonth
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='CurrentMonth'
    """, (user_id, conversation_id))
    row_month = cursor.fetchone()
    current_month = int(row_month[0]) if row_month else 1

    # Get CurrentDay
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'
    """, (user_id, conversation_id))
    row_day = cursor.fetchone()
    current_day = int(row_day[0]) if row_day else 1

    # Get TimeOfDay
    cursor.execute("""
        SELECT value
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='TimeOfDay'
    """, (user_id, conversation_id))
    row_time = cursor.fetchone()
    time_of_day = row_time[0] if row_time else "Morning"

    conn.close()
    return current_year, current_month, current_day, time_of_day

def set_current_time(user_id, conversation_id, new_year, new_month, new_day, new_phase):
    """
    Upserts CurrentYear, CurrentMonth, CurrentDay, and TimeOfDay in the CurrentRoleplay table for the given user_id and conversation_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Upsert CurrentYear
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'CurrentYear', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, str(new_year)))

    # Upsert CurrentMonth
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'CurrentMonth', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, str(new_month)))

    # Upsert CurrentDay
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'CurrentDay', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, str(new_day)))

    # Upsert TimeOfDay
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'TimeOfDay', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, new_phase))

    conn.commit()
    conn.close()

def advance_time(user_id, conversation_id, increment=1):
    """
    Advances the current time by a given number of phases.
    
    1. Reads the current year, month, day, and time_of_day.
    2. Advances the time_of_day by 'increment' steps (wrapping around if necessary).
       Every full cycle of phases increases the day by 1.
    3. If the new day exceeds DAYS_PER_MONTH, resets day to 1 and increments the month.
       If the new month exceeds MONTHS_PER_YEAR, resets month to 1 and increments the year.
    4. Saves the new year, month, day, and time_of_day back to the database.
    5. Returns (new_year, new_month, new_day, new_phase).
    """
    current_year, current_month, current_day, current_phase = get_current_time(user_id, conversation_id)

    try:
        phase_index = TIME_PHASES.index(current_phase)
    except ValueError:
        phase_index = 0

    new_index = phase_index + increment
    day_increment = new_index // len(TIME_PHASES)
    new_index = new_index % len(TIME_PHASES)

    new_phase = TIME_PHASES[new_index]
    new_day = current_day + day_increment

    new_month = current_month
    new_year = current_year

    # Check if day exceeds the maximum for the month.
    if new_day > DAYS_PER_MONTH:
        new_day = 1  # Reset day to 1
        new_month += 1  # Increment the month
        # Check if month exceeds the maximum for the year.
        if new_month > MONTHS_PER_YEAR:
            new_month = 1  # Reset month to 1
            new_year += 1  # Increment the year

    # Save the updated time values.
    set_current_time(user_id, conversation_id, new_year, new_month, new_day, new_phase)
    return new_year, new_month, new_day, new_phase

def update_npc_schedules_for_time(user_id, conversation_id, day, time_of_day):
    """
    For each NPC, checks if there is an override (in PlannedEvents) for the given day and time_of_day.
    If so, uses that; otherwise, it uses the NPC's default schedule from their schedule JSON.
    Updates each NPC's current_location accordingly.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch planned events for overrides.
    cursor.execute("""
        SELECT npc_id, override_location
        FROM PlannedEvents
        WHERE user_id=%s AND conversation_id=%s
          AND day=%s AND time_of_day=%s
    """, (user_id, conversation_id, day, time_of_day))
    override_rows = cursor.fetchall()
    override_dict = {r[0]: r[1] for r in override_rows}

    # Fetch all NPC schedules.
    cursor.execute("""
        SELECT npc_id, schedule
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    npc_rows = cursor.fetchall()

    # For each NPC, update the current_location.
    for (npc_id, schedule_json) in npc_rows:
        if npc_id in override_dict:
            new_location = override_dict[npc_id]
        else:
            if schedule_json:
                # If the schedule is stored as a JSON object, e.g.
                # {"Morning": "Cafe", "Afternoon": "Park", ...}
                # then get the location for the current time_of_day.
                new_location = schedule_json.get(time_of_day, "Unknown")
            else:
                new_location = "No schedule"

        cursor.execute("""
            UPDATE NPCStats
            SET current_location=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (new_location, user_id, conversation_id, npc_id))

    conn.commit()
    conn.close()

def advance_time_and_update(user_id, conversation_id, increment=1):
    """
    Advances time by a given number of phases, updates NPC schedules,
    and removes any expired planned events.
    Returns (new_year, new_month, new_day, new_phase).
    """
    new_year, new_month, new_day, new_phase = advance_time(user_id, conversation_id, increment)
    update_npc_schedules_for_time(user_id, conversation_id, new_day, new_phase)
    # Now remove planned events that are past due:
    remove_expired_planned_events(user_id, conversation_id, new_year, new_month, new_day, new_phase)
    return new_year, new_month, new_day, new_phase

#
# Enhanced Time Cycle Functionality
#

def should_advance_time(activity_type, current_time_of_day=None):
    """
    Determines if the specified activity should cause time to advance.
    
    Args:
        activity_type: The type of activity (string)
        current_time_of_day: Current time of day (if needed for special cases)
        
    Returns:
        Boolean indicating if time should advance and by how much
    """
    # Check if activity is in TIME_CONSUMING_ACTIVITIES
    if activity_type in TIME_CONSUMING_ACTIVITIES:
        return {
            "should_advance": True,
            "periods": TIME_CONSUMING_ACTIVITIES[activity_type]["time_advance"]
        }
    
    # Check if activity is in OPTIONAL_ACTIVITIES
    if activity_type in OPTIONAL_ACTIVITIES:
        return {
            "should_advance": False,
            "periods": 0
        }
    
    # Default case
    return {
        "should_advance": False,
        "periods": 0
    }

async def advance_time_with_events(user_id, conversation_id, activity_type):
    """
    Advanced version of time advancement that includes event processing.
    Only advances time if the activity warrants it, and processes
    appropriate events when time changes.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        activity_type: Type of activity that may trigger time advancement
        
    Returns:
        A dictionary containing:
        - time_advanced: Boolean indicating if time was advanced
        - new_time: New time of day if advanced
        - events: List of events that occurred
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get current time information
        cursor.execute("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=%s AND conversation_id=%s AND key='TimeOfDay'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        current_time = row[0] if row else "Morning"
        
        # Check if this activity should advance time
        advance_info = should_advance_time(activity_type, current_time)
        
        if not advance_info["should_advance"]:
            return {
                "time_advanced": False,
                "new_time": current_time,
                "events": []
            }
        
        # If we should advance time, do so
        
        # We'll advance by the number of periods specified
        periods_to_advance = advance_info["periods"]
        new_year, new_month, new_day, new_time = advance_time_and_update(
            user_id, conversation_id, increment=periods_to_advance
        )
        
        # Process events that should occur with time advancement
        events = []
        
        # Process NPC activities for the new time period
        await process_daily_npc_activities(user_id, conversation_id, new_time)
        
        # Check for relationship stage changes
        await detect_relationship_stage_changes(user_id, conversation_id)
        
        # Check for narrative stage changes
        narrative_stage = await get_current_narrative_stage(user_id, conversation_id)
        if narrative_stage:
            events.append({
                "type": "narrative_stage",
                "stage": narrative_stage.name,
                "description": narrative_stage.description
            })
        
        # Now check for various special events based on chance
        # Personal revelation
        if random.random() < SPECIAL_EVENT_CHANCES["personal_revelation"]:
            revelation = await check_for_personal_revelations(user_id, conversation_id)
            if revelation:
                events.append(revelation)
        
        # Narrative moment
        if random.random() < SPECIAL_EVENT_CHANCES["narrative_moment"]:
            moment = await check_for_narrative_moments(user_id, conversation_id)
            if moment:
                events.append(moment)
        
        # Relationship crossroads
        if random.random() < SPECIAL_EVENT_CHANCES["relationship_crossroads"]:
            crossroads = await check_for_relationship_crossroads(user_id, conversation_id)
            if crossroads:
                events.append(crossroads)
        
        # Relationship ritual
        if random.random() < SPECIAL_EVENT_CHANCES["relationship_ritual"]:
            ritual = await check_for_relationship_ritual(user_id, conversation_id)
            if ritual:
                events.append(ritual)
        
        # Dream sequence (if sleeping)
        if activity_type == "sleep" and random.random() < SPECIAL_EVENT_CHANCES["dream_sequence"]:
            dream = await add_dream_sequence(user_id, conversation_id)
            if dream:
                events.append(dream)
        
        # Moment of clarity
        if random.random() < SPECIAL_EVENT_CHANCES["moment_of_clarity"]:
            clarity = await add_moment_of_clarity(user_id, conversation_id)
            if clarity:
                events.append(clarity)
        
        # NPC mask slippage
        if random.random() < SPECIAL_EVENT_CHANCES["mask_slippage"]:
            # Get random NPC
            cursor.execute("""
                SELECT npc_id FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                ORDER BY RANDOM() LIMIT 1
            """, (user_id, conversation_id))
            
            npc_row = cursor.fetchone()
            if npc_row:
                npc_id = npc_row[0]
                slippage_events = await check_for_mask_slippage(user_id, conversation_id, npc_id)
                if slippage_events:
                    events.append({
                        "type": "mask_slippage",
                        "npc_id": npc_id,
                        "events": slippage_events
                    })
        
        # NPC revelation
        npc_revelation = await check_for_npc_revelations(user_id, conversation_id)
        if npc_revelation:
            events.append(npc_revelation)
        
        # Apply activity stat effects if defined
        if activity_type in TIME_CONSUMING_ACTIVITIES and TIME_CONSUMING_ACTIVITIES[activity_type]["stat_effects"]:
            stat_effects = TIME_CONSUMING_ACTIVITIES[activity_type]["stat_effects"]
            
            # Build SQL update for stats
            updates = []
            values = []
            
            for stat, change in stat_effects.items():
                updates.append(f"{stat} = {stat} + %s")
                values.append(change)
            
            if updates:
                values.extend([user_id, conversation_id])
                cursor.execute(f"""
                    UPDATE PlayerStats
                    SET {", ".join(updates)}
                    WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                """, values)
        
        conn.commit()
        
        return {
            "time_advanced": True,
            "new_time": new_time,
            "new_day": new_day,
            "new_month": new_month,
            "new_year": new_year,
            "events": events
        }
    
    except Exception as e:
        conn.rollback()
        logging.error(f"Error in advance_time_with_events: {e}")
        return {
            "time_advanced": False,
            "error": str(e)
        }
    finally:
        cursor.close()
        conn.close()

def classify_player_input(input_text):
    """
    Analyzes player input to determine the implied activity type.
    This is a simplified example - in a real implementation, you might use
    more sophisticated NLP techniques or a dedicated classifier.
    
    Args:
        input_text: The player's input text
        
    Returns:
        The classified activity type
    """
    input_lower = input_text.lower()
    
    # Check for sleep-related commands
    if any(term in input_lower for term in ["sleep", "go to bed", "rest", "go to sleep"]):
        return "sleep"
    
    # Check for class/work attendance
    if any(term in input_lower for term in ["go to class", "attend lecture", "go to work", "work shift"]):
        return "class_attendance" if "class" in input_lower or "lecture" in input_lower else "work_shift"
    
    # Check for social events
    if any(term in input_lower for term in ["party", "event", "gathering", "meetup", "meet up", "hang out"]):
        return "social_event"
    
    # Check for training
    if any(term in input_lower for term in ["train", "practice", "workout", "exercise"]):
        return "training"
    
    # Check for extended conversations
    if any(term in input_lower for term in ["talk to", "speak with", "discuss with", "have a conversation"]):
        return "extended_conversation"
    
    # Check for personal time
    if any(term in input_lower for term in ["relax", "chill", "personal time", "free time", "by myself"]):
        return "personal_time"
    
    # Check for observation
    if any(term in input_lower for term in ["look at", "observe", "watch"]):
        return "observe"
    
    # Check for quick chats
    if any(term in input_lower for term in ["quick chat", "say hi", "greet", "wave"]):
        return "quick_chat"
    
    # Check for phone usage
    if any(term in input_lower for term in ["check phone", "look at phone", "read messages"]):
        return "check_phone"
    
    # Default: If we can't classify, assume it's a non-time-advancing interaction
    return "quick_chat"

class ActivityManager:
    """
    Manages activity classification and time advancement decisions.
    Provides a central point for determining if an activity should advance time.
    """
    
    @staticmethod
    def get_activity_type(player_input, context=None):
        """
        Determines the activity type from player input and context.
        
        Args:
            player_input: The player's input text
            context: Additional context (location, NPCs present, etc.)
            
        Returns:
            The activity type
        """
        # This could be expanded to use more sophisticated analysis,
        # including the current context, location, time of day, etc.
        return classify_player_input(player_input)
    
    @staticmethod
    async def process_activity(user_id, conversation_id, player_input, context=None):
        """
        Processes the player's activity, including determining if time should advance
        and handling any resulting events.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
            player_input: The player's input text
            context: Additional context
            
        Returns:
            A dictionary with processing results
        """
        # Determine the activity type
        activity_type = ActivityManager.get_activity_type(player_input, context)
        
        # Check if this should advance time and process events
        result = await advance_time_with_events(user_id, conversation_id, activity_type)
        
        # Add activity type to result
        result["activity_type"] = activity_type
        
        # If this is a TIME_CONSUMING_ACTIVITY, include its description
        if activity_type in TIME_CONSUMING_ACTIVITIES:
            result["activity_description"] = TIME_CONSUMING_ACTIVITIES[activity_type]["description"]
        elif activity_type in OPTIONAL_ACTIVITIES:
            result["activity_description"] = OPTIONAL_ACTIVITIES[activity_type]["description"]
        
        return result

# ----------------------------------------------------------------
# NIGHTLY MAINTENANCE: Memory Fading, Summarizing, etc.
# ----------------------------------------------------------------

async def nightly_maintenance(user_id: int, conversation_id: int):
    """
    For each NPC in the conversation, run memory fade, summarization, 
    or other cleanup tasks that simulate a nightly 'brain housekeeping'.
    Typically called after advancing to the next day.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Gather all NPCs for this user+conversation
        cursor.execute("""
            SELECT npc_id
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s
        """, (user_id, conversation_id))
        npc_ids = [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()
    
    # For each NPC, create a memory manager and do fade steps
    for nid in npc_ids:
        mem_mgr = EnhancedMemoryManager(nid, user_id, conversation_id)
        
        # 1) Prune older trivial memories
        await mem_mgr.prune_old_memories(age_days=14, significance_threshold=3, intensity_threshold=15)
        
        # 2) Decay older but still relevant memories
        await mem_mgr.apply_memory_decay(age_days=30, decay_rate=0.2)
        
        # 3) Summarize repetitive ones
        await mem_mgr.summarize_repetitive_memories(lookback_days=7, min_count=3)

    logging.info(f"[nightly_maintenance] Completed for user={user_id}, conversation={conversation_id}")
