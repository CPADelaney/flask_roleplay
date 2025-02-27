# logic/enhanced_stats.py

import random
import json
import logging
from db.connection import get_db_connection

# Define stat threshold tiers that trigger different game behaviors
STAT_THRESHOLDS = {
    "corruption": [
        {"level": 30, "name": "Curious", "behaviors": ["occasional_submission", "light_teasing"]},
        {"level": 50, "name": "Accepting", "behaviors": ["regular_submission", "moderate_humiliation", "voluntary_service"]},
        {"level": 70, "name": "Craving", "behaviors": ["eager_submission", "heavy_humiliation", "ritual_service"]},
        {"level": 90, "name": "Devoted", "behaviors": ["complete_submission", "public_humiliation", "permanent_servitude"]}
    ],
    "obedience": [
        {"level": 30, "name": "Compliant", "behaviors": ["follows_direct_orders", "minimal_resistance"]},
        {"level": 50, "name": "Dutiful", "behaviors": ["anticipates_orders", "apologizes_for_hesitation"]},
        {"level": 70, "name": "Conditioned", "behaviors": ["automatic_obedience", "discomfort_when_disobeying"]},
        {"level": 90, "name": "Programmed", "behaviors": ["cannot_disobey", "distress_without_orders"]}
    ],
    "dependency": [
        {"level": 30, "name": "Attached", "behaviors": ["seeks_approval", "mild_separation_anxiety"]},
        {"level": 50, "name": "Reliant", "behaviors": ["frequent_check-ins", "moderate_separation_anxiety"]},
        {"level": 70, "name": "Dependent", "behaviors": ["constant_presence", "severe_separation_anxiety"]},
        {"level": 90, "name": "Addicted", "behaviors": ["panic_when_alone", "physical_symptoms_when_separated"]}
    ],
    # Add similar tiers for other stats
}

# Define combinations of stats that trigger special states
STAT_COMBINATIONS = [
    {
        "name": "Willing Prisoner",
        "description": "Player actively seeks confinement and restriction",
        "conditions": {"corruption": 60, "obedience": 70, "dependency": 50},
        "behaviors": ["requests_restraints", "thanks_for_control", "anxiety_without_boundaries"]
    },
    {
        "name": "Breaking Point",
        "description": "Player exhibits signs of psychological fracturing",
        "conditions": {"mental_resilience": -10, "confidence": -20, "willpower": -30},
        "behaviors": ["disassociation", "fugue_states", "confused_identity"]
    },
    {
        "name": "Stockholm Syndrome",
        "description": "Player has formed unhealthy attachments to controlling NPCs",
        "conditions": {"dependency": 80, "corruption": 60, "mental_resilience": -40},
        "behaviors": ["defends_abusers", "reinterprets_cruelty_as_love", "fears_rescue"]
    },
    # Add more complex combinations
]

def get_player_current_tier(user_id, conversation_id, stat_name):
    """
    Determine which tier a player is in for a given stat
    Returns the threshold dict with level, name and behaviors
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            SELECT {stat_name} FROM PlayerStats 
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        row = cursor.fetchone()
        if not row:
            return None
        
        stat_value = row[0]
        thresholds = STAT_THRESHOLDS.get(stat_name, [])
        
        # Find the highest threshold the player meets
        current_tier = None
        for tier in thresholds:
            if stat_value >= tier["level"]:
                current_tier = tier
            else:
                break
                
        return current_tier
    finally:
        cursor.close()
        conn.close()

def check_for_combination_triggers(user_id, conversation_id):
    """
    Check if player stats trigger any special combination states
    Returns a list of triggered combinations
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, dependency, 
                   lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        row = cursor.fetchone()
        if not row:
            return []
            
        player_stats = {
            "corruption": row[0],
            "confidence": row[1],
            "willpower": row[2],
            "obedience": row[3],
            "dependency": row[4],
            "lust": row[5],
            "mental_resilience": row[6],
            "physical_endurance": row[7]
        }
        
        triggered_combinations = []
        
        for combo in STAT_COMBINATIONS:
            meets_all_conditions = True
            for stat_name, required_value in combo["conditions"].items():
                if player_stats.get(stat_name, 0) < required_value:
                    meets_all_conditions = False
                    break
                    
            if meets_all_conditions:
                triggered_combinations.append(combo)
                
        return triggered_combinations
    finally:
        cursor.close()
        conn.close()

def record_stat_change_event(user_id, conversation_id, stat_name, old_value, new_value, cause=""):
    """
    Record significant stat changes in a new StatsHistory table
    """
    # Only record significant changes (more than 5 points)
    if abs(new_value - old_value) < 5:
        return
        
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO StatsHistory 
            (user_id, conversation_id, player_name, stat_name, old_value, new_value, cause, timestamp)
            VALUES (%s, %s, 'Chase', %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """, (user_id, conversation_id, stat_name, old_value, new_value, cause))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Error recording stat change: {e}")
    finally:
        cursor.close()
        conn.close()

def apply_stat_change(user_id, conversation_id, changes, cause=""):
    """
    Apply multiple stat changes at once and record the history
    
    Example:
    changes = {
        "corruption": +5,
        "willpower": -3,
        "confidence": -2
    }
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # First get current values
        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, dependency, 
                   lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        row = cursor.fetchone()
        if not row:
            return False
            
        current_stats = {
            "corruption": row[0],
            "confidence": row[1],
            "willpower": row[2],
            "obedience": row[3],
            "dependency": row[4],
            "lust": row[5],
            "mental_resilience": row[6],
            "physical_endurance": row[7]
        }
        
        # Prepare updates
        updates = []
        values = []
        
        for stat_name, change in changes.items():
            if stat_name not in current_stats:
                continue
                
            old_value = current_stats[stat_name]
            new_value = max(0, min(100, old_value + change))  # Clamp between 0-100
            
            updates.append(f"{stat_name} = %s")
            values.append(new_value)
            
            # Record the change
            record_stat_change_event(user_id, conversation_id, stat_name, old_value, new_value, cause)
        
        if not updates:
            return False
            
        # Apply the updates
        sql = "UPDATE PlayerStats SET " + ", ".join(updates)
        sql += " WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'"
        values.extend([user_id, conversation_id])
        
        cursor.execute(sql, values)
        conn.commit()
        
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error applying stat changes: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# Define specific activities that modify stats
ACTIVITY_EFFECTS = {
    "public_humiliation": {
        "corruption": +3,
        "confidence": -5,
        "obedience": +7,
        "mental_resilience": -3
    },
    "praise_and_reward": {
        "dependency": +5,
        "corruption": +2,
        "confidence": +3
    },
    "punishment": {
        "obedience": +8,
        "willpower": -4,
        "physical_endurance": +2
    },
    "isolation": {
        "dependency": +10,
        "mental_resilience": -6,
        "willpower": -3
    },
    "training_session": {
        "obedience": +6,
        "corruption": +4,
        "physical_endurance": +5
    },
    "service_task": {
        "obedience": +3,
        "corruption": +2,
        "dependency": +3
    },
    "resistance_attempt": {
        "willpower": +5,
        "confidence": +3,
        "obedience": -4,
        "corruption": -2
    },
    # Add more activities with effects
}

def apply_activity_effects(user_id, conversation_id, activity_name, intensity=1.0):
    """
    Apply stat changes based on a specific activity
    Intensity multiplier can increase or decrease the effect
    """
    if activity_name not in ACTIVITY_EFFECTS:
        return False
        
    # Get the base effects and scale by intensity
    base_effects = ACTIVITY_EFFECTS[activity_name]
    scaled_effects = {stat: change * intensity for stat, change in base_effects.items()}
    
    # Apply the changes
    return apply_stat_change(
        user_id, conversation_id, 
        scaled_effects, 
        f"Activity: {activity_name} (intensity: {intensity})"
    )
