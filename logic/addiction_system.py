# logic/addiction_system.py

"""
Module for managing player addictions to various stimuli in an extreme femdom roleplay setting.
This version supports enhanced thematic messaging, bidirectional addiction changes,
additional addiction types (e.g., humiliation, submission), and integrated stat penalties.
"""

import logging
import random
import json
from datetime import datetime
from db.connection import get_db_connection

# Addiction levels
ADDICTION_LEVELS = {
    0: "None",
    1: "Mild",
    2: "Moderate", 
    3: "Heavy",
    4: "Extreme"
}

# ------------------------------------------------------------------------------
# Helper Functions for Thematic Addiction Effects
# ------------------------------------------------------------------------------

def process_socks_addiction(level):
    messages = []
    if level >= 1:
        messages.append("You occasionally steal glances at sumptuous stockings.")
    if level >= 2:
        messages.append("A subtle craving for the delicate feel of silk emerges within you.")
    if level >= 3:
        messages.append("The allure of sensuous socks overwhelms your thoughts.")
    if level >= 4:
        messages.append("Under your Mistress’s commanding presence, your obsession with exquisite socks leaves you trembling in servile adoration.")
    return messages

def process_feet_addiction(level):
    messages = []
    if level >= 1:
        messages.append("Your eyes frequently wander to the graceful arch of bare feet.")
    if level >= 2:
        messages.append("A surge of forbidden excitement courses through you at the mere glimpse of uncovered toes.")
    if level >= 3:
        messages.append("Distracted by the sight of enticing feet, you find it difficult to focus on anything else.")
    if level >= 4:
        messages.append("In the presence of your dominant Mistress, your fixation on every tantalizing curve of feet renders you utterly submissive.")
    return messages

def process_sweat_addiction(level):
    messages = []
    if level >= 1:
        messages.append("The scent of perspiration begins to evoke an unspoken thrill within you.")
    if level >= 2:
        messages.append("Each drop of sweat stokes a simmering desire you dare not fully acknowledge.")
    if level >= 3:
        messages.append("Your senses heighten as the aroma of exertion casts a spell over your inhibitions.")
    if level >= 4:
        messages.append("Overwhelmed by the intoxicating allure of sweat, you are compelled to seek it out under your Mistress’s relentless command.")
    return messages

def process_ass_addiction(level):
    messages = []
    if level >= 1:
        messages.append("Your gaze lingers a little longer on the curves of a well-shaped rear.")
    if level >= 2:
        messages.append("A subtle, forbidden thrill courses through you at the sight of a pert backside.")
    if level >= 3:
        messages.append("You find yourself fixated on every tantalizing detail of exposed derrieres, your mind wandering into submissive fantasies.")
    if level >= 4:
        messages.append("Under your Mistress’s unwavering control, your obsession with perfectly sculpted rear ends drives you to desperate submission.")
    return messages

def process_scent_addiction(level):
    messages = []
    if level >= 1:
        messages.append("You become acutely aware of natural pheromones and subtle scents around you.")
    if level >= 2:
        messages.append("Every hint of an enticing aroma sends a shiver down your spine, awakening deep desires.")
    if level >= 3:
        messages.append("You begin to collect memories of scents, each evoking a surge of submissive longing.")
    if level >= 4:
        messages.append("In the grip of your extreme addiction, the mere whiff of a scent under your Mistress’s watchful eye reduces you to euphoric submission.")
    return messages

def process_humiliation_addiction(level):
    messages = []
    if level >= 1:
        messages.append("The sting of humiliation sparks a curious thrill in your submissive heart.")
    if level >= 2:
        messages.append("You find yourself yearning for more degrading scenarios as your pride withers under each slight.")
    if level >= 3:
        messages.append("Every act of public embarrassment intensifies your craving to be dominated and humiliated.")
    if level >= 4:
        messages.append("In the presence of your ruthless Mistress, the exquisite agony of humiliation consumes you, binding your will entirely to her desires.")
    return messages

def process_submission_addiction(level):
    messages = []
    if level >= 1:
        messages.append("The taste of obedience becomes subtly intoxicating as you seek her approval in every glance.")
    if level >= 2:
        messages.append("Your need to surrender grows, craving the reassurance that only your Mistress can provide.")
    if level >= 3:
        messages.append("In every command, you find a deeper satisfaction in your subjugated state, yearning to be molded by her hand.")
    if level >= 4:
        messages.append("Your identity dissolves in the overwhelming tide of submission, as your Mistress’s word becomes the sole law governing your existence.")
    return messages

# Mapping of addiction types to their effect processors
EFFECT_PROCESSORS = {
    "socks": process_socks_addiction,
    "feet": process_feet_addiction,
    "sweat": process_sweat_addiction,
    "ass": process_ass_addiction,
    "scent": process_scent_addiction,
    "humiliation": process_humiliation_addiction,
    "submission": process_submission_addiction,
}

# ------------------------------------------------------------------------------
# Helper Function for Stat Penalties
# ------------------------------------------------------------------------------

def apply_stat_penalty(user_id, conversation_id, player_name, stat="willpower", penalty=5):
    """
    Apply a penalty to a player's stat (e.g., willpower) in the PlayerStats table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Note: Using string formatting for the stat name is acceptable if the stat names are controlled.
        query = f"""
            UPDATE PlayerStats
            SET {stat} = GREATEST({stat} - %s, 0)
            WHERE user_id = %s AND conversation_id = %s AND player_name = %s
        """
        cursor.execute(query, (penalty, user_id, conversation_id, player_name))
        conn.commit()
        logging.info(f"Applied {penalty} penalty to {stat} for player {player_name}")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error applying {stat} penalty: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# ------------------------------------------------------------------------------
# Database Operations and Addiction Management
# ------------------------------------------------------------------------------

async def check_addiction_levels(user_id, conversation_id, player_name):
    """
    Check current addiction levels for all addiction types.
    
    Returns:
        dict: Dictionary of addiction types and their current levels, plus NPC-specific addictions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Ensure the addiction table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS PlayerAddictions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                player_name VARCHAR(255) NOT NULL,
                addiction_type VARCHAR(50) NOT NULL,
                level INTEGER NOT NULL DEFAULT 0,
                target_npc_id INTEGER NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, conversation_id, player_name, addiction_type, target_npc_id)
            )
        """)
        conn.commit()
        
        # Get all addiction levels for the player
        cursor.execute("""
            SELECT addiction_type, level, target_npc_id
            FROM PlayerAddictions
            WHERE user_id = %s AND conversation_id = %s AND player_name = %s
        """, (user_id, conversation_id, player_name))
        
        addiction_data = {}
        npc_specific = []
        
        for row in cursor.fetchall():
            addiction_type, level, target_npc_id = row
            
            if target_npc_id is None:
                addiction_data[addiction_type] = level
            else:
                # For NPC-specific addictions, fetch the NPC's name for better messaging
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id = %s AND conversation_id = %s AND npc_id = %s
                """, (user_id, conversation_id, target_npc_id))
                npc_row = cursor.fetchone()
                if npc_row:
                    npc_name = npc_row[0]
                    npc_specific.append({
                        "addiction_type": addiction_type,
                        "level": level,
                        "npc_id": target_npc_id,
                        "npc_name": npc_name
                    })
        
        return {
            "addiction_levels": addiction_data,
            "npc_specific_addictions": npc_specific,
            "has_addictions": any(level > 0 for level in addiction_data.values()) or len(npc_specific) > 0
        }
        
    except Exception as e:
        logging.error(f"Error checking addiction levels: {str(e)}")
        return {
            "addiction_levels": {},
            "npc_specific_addictions": [],
            "has_addictions": False,
            "error": str(e)
        }
    finally:
        cursor.close()
        conn.close()

async def update_addiction_level(user_id, conversation_id, player_name, addiction_type,
                                 progression_chance=0.2, progression_multiplier=1.0,
                                 regression_chance=0.1, target_npc_id=None):
    """
    Update addiction level with a chance of progression or regression.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Player name
        addiction_type: Type of addiction (e.g., socks, feet, sweat, ass, scent, humiliation, submission)
        progression_chance: Base chance of progression (0.0-1.0)
        progression_multiplier: Multiplier to adjust progression chance (e.g., for intense scenes)
        regression_chance: Chance of regression (0.0-1.0)
        target_npc_id: Optional NPC ID for NPC-specific addictions
        
    Returns:
        dict: Updated addiction info.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check current addiction level
        if target_npc_id is None:
            cursor.execute("""
                SELECT level FROM PlayerAddictions
                WHERE user_id = %s AND conversation_id = %s AND player_name = %s 
                AND addiction_type = %s AND target_npc_id IS NULL
            """, (user_id, conversation_id, player_name, addiction_type))
        else:
            cursor.execute("""
                SELECT level FROM PlayerAddictions
                WHERE user_id = %s AND conversation_id = %s AND player_name = %s 
                AND addiction_type = %s AND target_npc_id = %s
            """, (user_id, conversation_id, player_name, addiction_type, target_npc_id))
        
        row = cursor.fetchone()
        current_level = row[0] if row else 0
        
        roll = random.random()
        new_level = current_level
        
        # Attempt progression
        if roll < progression_chance * progression_multiplier:
            if current_level < 4:
                new_level = current_level + 1
                logging.info(f"Addiction '{addiction_type}' progressing from {current_level} to {new_level}")
        # Otherwise, check for regression
        elif roll > (1 - regression_chance) and current_level > 0:
            new_level = current_level - 1
            logging.info(f"Addiction '{addiction_type}' regressing from {current_level} to {new_level}")
        
        # Update the addiction level using UPSERT
        if target_npc_id is None:
            cursor.execute("""
                INSERT INTO PlayerAddictions 
                (user_id, conversation_id, player_name, addiction_type, level, last_updated)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                DO UPDATE SET level = %s, last_updated = NOW()
            """, (user_id, conversation_id, player_name, addiction_type, new_level, new_level))
        else:
            cursor.execute("""
                INSERT INTO PlayerAddictions 
                (user_id, conversation_id, player_name, addiction_type, level, target_npc_id, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                DO UPDATE SET level = %s, last_updated = NOW()
            """, (user_id, conversation_id, player_name, addiction_type, new_level, target_npc_id, new_level))
        
        conn.commit()
        
        return {
            "addiction_type": addiction_type,
            "previous_level": current_level,
            "new_level": new_level,
            "level_name": ADDICTION_LEVELS[new_level],
            "progressed": new_level > current_level,
            "regressed": new_level < current_level,
            "target_npc_id": target_npc_id
        }
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating addiction level for '{addiction_type}': {str(e)}")
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

async def process_addiction_effects(user_id, conversation_id, player_name, addiction_status):
    """
    Process effects based on current addiction levels and generate thematic messages.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Player name
        addiction_status: Dictionary of current addiction levels
        
    Returns:
        dict: Effects generated by addictions.
    """
    effects = []
    addiction_levels = addiction_status.get("addiction_levels", {})
    npc_specific = addiction_status.get("npc_specific_addictions", [])
    
    # Process general (player) addictions using the modular processors
    for addiction_type, level in addiction_levels.items():
        if level <= 0:
            continue
        processor = EFFECT_PROCESSORS.get(addiction_type)
        if processor:
            messages = processor(level)
            effects.extend(messages)
        else:
            effects.append(f"Your addiction to {addiction_type} is at level {level}.")
        
        # Apply stat penalty if at extreme level
        if level == 4:
            logging.info(f"Extreme '{addiction_type}' addiction detected; applying willpower penalty.")
            apply_stat_penalty(user_id, conversation_id, player_name, stat="willpower", penalty=5)
    
    # Process NPC-specific addictions with additional messaging
    for npc_addiction in npc_specific:
        npc_name = npc_addiction.get("npc_name", "Someone")
        addiction_type = npc_addiction.get("addiction_type", "")
        level = npc_addiction.get("level", 0)
        
        if level >= 3:
            effects.append(f"You have a powerful {ADDICTION_LEVELS[level].lower()} addiction to {npc_name}'s {addiction_type}.")
            if level >= 4:
                effects.append(f"When {npc_name} is near, your submission intensifies—you can barely control your impulses regarding their {addiction_type}.")
                logging.info(f"Extreme NPC-specific addiction for '{addiction_type}' with {npc_name}; applying willpower penalty.")
                apply_stat_penalty(user_id, conversation_id, player_name, stat="willpower", penalty=5)
    
    return {
        "effects": effects,
        "has_effects": len(effects) > 0
    }

async def get_addiction_status(user_id, conversation_id, player_name):
    """
    Get current addiction status for a player.
    This is a wrapper for check_addiction_levels.
    """
    return await check_addiction_levels(user_id, conversation_id, player_name)

def get_addiction_label(level):
    """Helper function to get the textual label for an addiction level."""
    return ADDICTION_LEVELS.get(level, "Unknown")
