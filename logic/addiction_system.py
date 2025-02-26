# logic/addiction_system.py

"""
Module for managing player addictions to various stimuli.
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

async def check_addiction_levels(user_id, conversation_id, player_name):
    """
    Check current addiction levels for all addiction types.
    
    Returns:
        dict: Dictionary of addiction types and their current levels
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if addiction table exists, if not create it
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
        
        # Get all addiction levels for player
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
                # General addiction
                addiction_data[addiction_type] = level
            else:
                # NPC-specific addiction
                # Get NPC name
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

async def update_addiction_level(user_id, conversation_id, player_name, addiction_type, progression_chance=0.2, target_npc_id=None):
    """
    Update addiction level with a chance of progression or regression.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Player name
        addiction_type: Type of addiction (socks, feet, sweat, ass, scent)
        progression_chance: Chance of addiction progressing (0.0-1.0)
        target_npc_id: Optional NPC ID for NPC-specific addictions
        
    Returns:
        dict: Updated addiction data
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
        
        # Determine if addiction progresses
        roll = random.random()
        new_level = current_level
        
        if roll < progression_chance:
            # Progress addiction unless already at max
            if current_level < 4:  # Max level is 4 (Extreme)
                new_level = current_level + 1
                logging.info(f"Addiction {addiction_type} progressing from {current_level} to {new_level}")
        
        # Update addiction level in database with UPSERT
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
        
        # Return updated addiction info
        return {
            "addiction_type": addiction_type,
            "previous_level": current_level,
            "new_level": new_level,
            "level_name": ADDICTION_LEVELS[new_level],
            "progressed": new_level > current_level,
            "target_npc_id": target_npc_id
        }
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating addiction level: {str(e)}")
        return {
            "error": str(e)
        }
    finally:
        cursor.close()
        conn.close()

async def process_addiction_effects(user_id, conversation_id, player_name, addiction_status):
    """
    Process effects based on current addiction levels.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Player name
        addiction_status: Dictionary of current addiction levels
        
    Returns:
        dict: Effects generated by addictions
    """
    effects = []
    addiction_levels = addiction_status.get("addiction_levels", {})
    npc_specific = addiction_status.get("npc_specific_addictions", [])
    
    # Process general addiction effects
    for addiction_type, level in addiction_levels.items():
        if level <= 0:
            continue
            
        # Add effects based on addiction type and level
        if addiction_type == "socks":
            if level >= 1:
                effects.append(f"You find yourself occasionally glancing at people's socks")
            if level >= 2:
                effects.append(f"You feel a mild urge to touch or smell socks")
            if level >= 3:
                effects.append(f"You become distracted when seeing socks and have intrusive thoughts about them")
            if level >= 4:
                effects.append(f"You feel an overwhelming need to engage with socks - collecting, smelling, or touching them")
                
        elif addiction_type == "feet":
            if level >= 1:
                effects.append(f"You notice yourself looking at people's feet more often")
            if level >= 2:
                effects.append(f"You feel a rush of excitement when someone removes their shoes")
            if level >= 3:
                effects.append(f"You struggle to focus on conversations when bare feet are visible")
            if level >= 4:
                effects.append(f"You're consumed by thoughts about feet and look for any opportunity to see, touch, or smell them")
                
        elif addiction_type == "sweat":
            if level >= 1:
                effects.append(f"You notice the scent of sweat more than before")
            if level >= 2:
                effects.append(f"You feel a subtle excitement when detecting someone's sweat")
            if level >= 3:
                effects.append(f"You find yourself drawn to people after they've exercised")
            if level >= 4:
                effects.append(f"The smell of sweat is intoxicating to you, and you seek it out constantly")
                
        elif addiction_type == "ass":
            if level >= 1:
                effects.append(f"You catch yourself looking at people's rear ends more often")
            if level >= 2:
                effects.append(f"You have frequent thoughts about touching or smelling people's behinds")
            if level >= 3:
                effects.append(f"You struggle to focus when someone's behind is prominently displayed")
            if level >= 4:
                effects.append(f"You're obsessed with rear ends and will do almost anything to get closer to them")
                
        elif addiction_type == "scent":
            if level >= 1:
                effects.append(f"You're more aware of people's natural scents")
            if level >= 2:
                effects.append(f"You find yourself leaning in closer to people to catch their scent")
            if level >= 3:
                effects.append(f"You've begun collecting items with people's scents on them")
            if level >= 4:
                effects.append(f"You're constantly thinking about capturing and experiencing people's scents")
    
    # Process NPC-specific addiction effects
    for npc_addiction in npc_specific:
        npc_name = npc_addiction.get("npc_name", "Someone")
        addiction_type = npc_addiction.get("addiction_type", "")
        level = npc_addiction.get("level", 0)
        
        if level >= 3:  # Only show effects for significant NPC addictions
            effects.append(f"You have a powerful {ADDICTION_LEVELS[level].lower()} addiction to {npc_name}'s {addiction_type}")
            
            # Add specific effects based on the combination
            if level >= 4:
                effects.append(f"When {npc_name} is around, you can barely control your impulses related to their {addiction_type}")
                
                # Apply stat penalties when extremely addicted
                conn = get_db_connection()
                cursor = conn.cursor()
                try:
                    # Reduce willpower when extremely addicted
                    cursor.execute("""
                        UPDATE PlayerStats
                        SET willpower = GREATEST(willpower - 5, 0)
                        WHERE user_id = %s AND conversation_id = %s AND player_name = %s
                    """, (user_id, conversation_id, player_name))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logging.error(f"Error applying addiction stat penalty: {str(e)}")
                finally:
                    cursor.close()
                    conn.close()
    
    # Return addiction effects
    return {
        "effects": effects,
        "has_effects": len(effects) > 0
    }

async def get_addiction_status(user_id, conversation_id, player_name):
    """
    Get current addiction status for a player.
    This is just a wrapper for check_addiction_levels.
    """
    return await check_addiction_levels(user_id, conversation_id, player_name)

def get_addiction_label(level):
    """Helper function to get text label for addiction level"""
    return ADDICTION_LEVELS.get(level, "Unknown")
