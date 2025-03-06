# logic/npc_agents/environment_perception.py

"""
Handles NPC perception of their environment
"""

import json
import logging
from typing import Dict, Any, Optional
from db.connection import get_db_connection

async def fetch_environment_data(user_id, conversation_id, context):
    """Fetch relevant environment data based on the current context"""
    env_data = {
        "location": "Unknown",
        "time_of_day": "Unknown",
        "day": "Unknown",
        "entities_present": []
    }

    # Get current location
    location = context.get("location")
    if not location:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND key = 'CurrentLocation'
            """, (user_id, conversation_id))
            row = cursor.fetchone()
            if row:
                location = row[0]

            # Time info
            cursor.execute("""
                SELECT key, value
                FROM CurrentRoleplay
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND key IN ('TimeOfDay', 'CurrentDay')
            """, (user_id, conversation_id))
            for row in cursor.fetchall():
                key, value = row
                if key == "TimeOfDay":
                    env_data["time_of_day"] = value
                elif key == "CurrentDay":
                    env_data["day"] = value

            # Entities present if location known
            if location:
                cursor.execute("""
                    SELECT npc_id, npc_name
                    FROM NPCStats
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND current_location = %s
                """, (user_id, conversation_id, location))
                entities = []
                for row in cursor.fetchall():
                    npc_id, npc_name = row
                    entities.append({
                        "type": "npc",
                        "id": npc_id,
                        "name": npc_name
                    })
                # Assume player is present
                entities.append({
                    "type": "player",
                    "id": user_id,
                    "name": "Chase"
                })
                env_data["entities_present"] = entities

        except Exception as e:
            logging.error(f"Error fetching environment data: {e}")
        finally:
            conn.close()

    if location:
        env_data["location"] = location

    return env_data

def is_significant_action(action, result):
    """Determine if an action is significant enough to create a memory"""
    if action["type"] in ["talk", "command", "mock", "confide", "praise"]:
        return True

    outcome = result.get("outcome", "")
    if "surprising" in outcome or "unexpected" in outcome:
        return True

    emotional_impact = result.get("emotional_impact", 0)
    if abs(emotional_impact) >= 3:
        return True

    return False

async def execute_npc_action(npc_id, user_id, conversation_id, action, context):
    """Execute an NPC action and return the result"""
    result = {
        "outcome": f"NPC performed action: {action['description']}",
        "emotional_impact": 0,
        "target_reactions": []
    }

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT npc_name
            FROM NPCStats
            WHERE npc_id = %s
              AND user_id = %s
              AND conversation_id = %s
        """, (npc_id, user_id, conversation_id))
        row = cursor.fetchone()
        npc_name = row[0] if row else f"NPC_{npc_id}"

        # Action handling
        if action["type"] == "talk":
            if action["target"] == "player":
                result["outcome"] = f"{npc_name} engages in conversation with the player."
                result["emotional_impact"] = 1
                result["target_reactions"] = ["Player listens"]
            elif action["target"] == "group":
                result["outcome"] = f"{npc_name} addresses the group."
                result["emotional_impact"] = 1
                result["target_reactions"] = ["The group listens"]
            else:
                target_id = action["target"]
                cursor.execute("""
                    SELECT npc_name
                    FROM NPCStats
                    WHERE npc_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """, (target_id, user_id, conversation_id))
                target_row = cursor.fetchone()
                target_name = target_row[0] if target_row else f"NPC_{target_id}"
                result["outcome"] = f"{npc_name} talks to {target_name}."
                result["emotional_impact"] = 1
                result["target_reactions"] = [f"{target_name} listens"]

        elif action["type"] == "command":
            if action["target"] == "player":
                result["outcome"] = f"{npc_name} gives a command to the player."
                result["emotional_impact"] = 2
                result["target_reactions"] = ["Player considers the command"]
            else:
                target_id = action["target"]
                cursor.execute("""
                    SELECT npc_name
                    FROM NPCStats
                    WHERE npc_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """, (target_id, user_id, conversation_id))
                target_row = cursor.fetchone()
                target_name = target_row[0] if target_row else f"NPC_{target_id}"
                result["outcome"] = f"{npc_name} commands {target_name}."
                result["emotional_impact"] = 2
                result["target_reactions"] = [f"{target_name} responds to the command"]

        elif action["type"] == "mock":
            if action["target"] == "player":
                result["outcome"] = f"{npc_name} mocks the player."
                result["emotional_impact"] = -2
                result["target_reactions"] = ["Player feels mocked"]
            else:
                target_id = action["target"]
                cursor.execute("""
                    SELECT npc_name
                    FROM NPCStats
                    WHERE npc_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """, (target_id, user_id, conversation_id))
                target_row = cursor.fetchone()
                target_name = target_row[0] if target_row else f"NPC_{target_id}"
                result["outcome"] = f"{npc_name} mocks {target_name}."
                result["emotional_impact"] = -2
                result["target_reactions"] = [f"{target_name} feels mocked"]

        elif action["type"] == "observe":
            result["outcome"] = f"{npc_name} observes quietly."
            result["emotional_impact"] = 0
            result["target_reactions"] = []

        elif action["type"] == "leave":
            result["outcome"] = f"{npc_name} leaves the area."
            result["emotional_impact"] = 0
            result["target_reactions"] = []

        elif action["type"] == "scheduled":
            result["outcome"] = f"{npc_name} {action['description']}"
            result["emotional_impact"] = 0
            result["target_reactions"] = []

        else:
            result["outcome"] = f"{npc_name} {action['description']}"
            result["emotional_impact"] = 0
            result["target_reactions"] = []

        return result
    except Exception as e:
        logging.error(f"Error executing NPC action: {e}")
        return result
    finally:
        conn.close()
