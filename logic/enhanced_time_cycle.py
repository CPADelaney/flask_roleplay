# logic/enhanced_time_cycle.py

import logging
import json
import random
from datetime import datetime
from db.connection import get_db_connection
from logic.npc_evolution import (
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
from logic.enhanced_social_links import (
    check_for_relationship_crossroads,
    check_for_relationship_ritual
)

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
        from logic.time_cycle import advance_time_and_update
        
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

# Example of integration with request handling in Flask
"""
@app.route('/player_action', methods=['POST'])
async def player_action():
    data = request.get_json()
    user_id = session.get('user_id')
    conversation_id = data.get('conversation_id')
    player_input = data.get('input')
    
    # Process the activity
    result = await ActivityManager.process_activity(user_id, conversation_id, player_input)
    
    # Build response based on result
    response = {
        "message": "Action processed successfully",
        "time_advanced": result["time_advanced"]
    }
    
    # If time advanced, include the new time
    if result["time_advanced"]:
        response["new_time"] = result["new_time"]
        
        # If any events occurred, include them
        if result["events"]:
            response["events"] = result["events"]
    
    return jsonify(response)
"""
