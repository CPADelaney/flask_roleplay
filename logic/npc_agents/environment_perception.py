# logic/npc_agents/environment_perception.py

import logging
from typing import Dict, Any, Optional, List
import asyncio

from db.connection import get_db_connection

logger = logging.getLogger(__name__)


async def fetch_environment_data(
    user_id: int,
    conversation_id: int,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fetch relevant environment data based on the current context.

    Args:
        user_id: The user/player ID
        conversation_id: The current conversation or scene ID
        context: A dictionary that may contain 'location', 'time_of_day', etc.

    Returns:
        A dictionary with keys:
          - "location"
          - "time_of_day"
          - "day"
          - "entities_present" (list of NPC/player descriptors)
    """
    env_data = {
        "location": context.get("location", "Unknown"),
        "time_of_day": context.get("time_of_day", "Unknown"),
        "day": context.get("day", "Unknown"),
        "entities_present": []
    }

    if env_data["location"] == "Unknown":
        def _fetch_db_info():
            with get_db_connection() as conn, conn.cursor() as cursor:
                # Attempt to get location from CurrentRoleplay
                location_query = """
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND key = 'CurrentLocation'
                """
                cursor.execute(location_query, (user_id, conversation_id))
                row = cursor.fetchone()
                if row:
                    env_data["location"] = row[0]

                # Attempt to fetch time info
                time_query = """
                    SELECT key, value
                    FROM CurrentRoleplay
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND key IN ('TimeOfDay', 'CurrentDay')
                """
                cursor.execute(time_query, (user_id, conversation_id))
                time_info_rows = cursor.fetchall()
                for key, value in time_info_rows:
                    if key == "TimeOfDay":
                        env_data["time_of_day"] = value
                    elif key == "CurrentDay":
                        env_data["day"] = value

                # If we now have a location, fetch entities
                if env_data["location"] != "Unknown":
                    entity_query = """
                        SELECT npc_id, npc_name
                        FROM NPCStats
                        WHERE user_id = %s
                          AND conversation_id = %s
                          AND current_location = %s
                    """
                    cursor.execute(
                        entity_query,
                        (user_id, conversation_id, env_data["location"])
                    )
                    entities = []
                    for npc_id, npc_name in cursor.fetchall():
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

        await asyncio.to_thread(_fetch_db_info)

    return env_data


def is_significant_action(action: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """
    Determine if an action is significant enough to create a memory.

    Args:
        action: A dictionary describing the action (type, description, etc.)
        result: The outcome of that action (including 'outcome', 'emotional_impact')

    Returns:
        True if the action/outcome is notable enough to warrant memory creation.
    """
    always_significant = {"talk", "command", "mock", "confide", "praise"}
    if action.get("type") in always_significant:
        return True

    outcome = result.get("outcome", "")
    if "surprising" in outcome.lower() or "unexpected" in outcome.lower():
        return True

    emotional_impact = result.get("emotional_impact", 0)
    if abs(emotional_impact) >= 3:
        return True

    return False


async def execute_npc_action(
    npc_id: int,
    user_id: int,
    conversation_id: int,
    action: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute an NPC action and return the result.
    """
    result = {
        "outcome": f"NPC performed action: {action.get('description', 'unknown')}",
        "emotional_impact": 0,
        "target_reactions": []
    }

    def _fetch_npc_name():
        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(
                    """
                    SELECT npc_name
                    FROM NPCStats
                    WHERE npc_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                    """,
                    (npc_id, user_id, conversation_id)
                )
                row = cursor.fetchone()
                return row[0] if row else f"NPC_{npc_id}"
            except Exception as e:
                logger.error("Error fetching NPC name for npc_id=%s: %s", npc_id, e)
                return f"NPC_{npc_id}"

    npc_name = await asyncio.to_thread(_fetch_npc_name)

    action_type = action.get("type")
    target = action.get("target", "group")

    if action_type == "talk":
        result.update(_handle_talk_action(npc_name, target, user_id, conversation_id))
    elif action_type == "command":
        result.update(_handle_command_action(npc_name, target, user_id, conversation_id))
    elif action_type == "mock":
        result.update(_handle_mock_action(npc_name, target, user_id, conversation_id))
    elif action_type == "observe":
        result["outcome"] = f"{npc_name} observes quietly."
        result["emotional_impact"] = 0
    elif action_type == "leave":
        result["outcome"] = f"{npc_name} leaves the area."
        result["emotional_impact"] = 0
    else:
        desc = action.get("description", "does something")
        result["outcome"] = f"{npc_name} {desc}"
        result["emotional_impact"] = 0

    return result


def _handle_talk_action(npc_name: str, target: Any, user_id: int, conversation_id: int) -> Dict[str, Any]:
    outcome = {"outcome": "", "emotional_impact": 1, "target_reactions": []}
    if target == "player":
        outcome["outcome"] = f"{npc_name} engages in conversation with the player."
        outcome["target_reactions"] = ["Player listens"]
    elif target == "group":
        outcome["outcome"] = f"{npc_name} addresses the group."
        outcome["target_reactions"] = ["The group listens"]
    else:
        try:
            target_id = int(target)
            target_name = _fetch_target_name(target_id, user_id, conversation_id)
            outcome["outcome"] = f"{npc_name} talks to {target_name}."
            outcome["target_reactions"] = [f"{target_name} listens"]
        except ValueError:
            outcome["outcome"] = f"{npc_name} tries to talk, but the target is unclear."
    return outcome


def _handle_command_action(npc_name: str, target: Any, user_id: int, conversation_id: int) -> Dict[str, Any]:
    outcome = {"outcome": "", "emotional_impact": 2, "target_reactions": []}
    if target == "player":
        outcome["outcome"] = f"{npc_name} gives a command to the player."
        outcome["target_reactions"] = ["Player considers the command"]
    else:
        try:
            target_id = int(target)
            target_name = _fetch_target_name(target_id, user_id, conversation_id)
            outcome["outcome"] = f"{npc_name} commands {target_name}."
            outcome["target_reactions"] = [f"{target_name} responds to the command"]
        except ValueError:
            outcome["outcome"] = f"{npc_name} issues a command, but the target is unclear."
    return outcome


def _handle_mock_action(npc_name: str, target: Any, user_id: int, conversation_id: int) -> Dict[str, Any]:
    outcome = {"outcome": "", "emotional_impact": -2, "target_reactions": []}
    if target == "player":
        outcome["outcome"] = f"{npc_name} mocks the player."
        outcome["target_reactions"] = ["Player feels mocked"]
    else:
        try:
            target_id = int(target)
            target_name = _fetch_target_name(target_id, user_id, conversation_id)
            outcome["outcome"] = f"{npc_name} mocks {target_name}."
            outcome["target_reactions"] = [f"{target_name} feels mocked"]
        except ValueError:
            outcome["outcome"] = f"{npc_name} attempts to mock someone, but the target is unclear."
    return outcome


def _fetch_target_name(target_id: int, user_id: int, conversation_id: int) -> str:
    """
    Helper to fetch target NPC name from DB, fallback is NPC_<id>.
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT npc_name
                FROM NPCStats
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
                """,
                (target_id, user_id, conversation_id)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return f"NPC_{target_id}"
    except Exception as e:
        logger.error("Error fetching target NPC name for npc_id=%s: %s", target_id, e)
        return f"NPC_{target_id}"
