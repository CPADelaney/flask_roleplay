# logic/npc_agents/environment_perception.py

"""
Environment perception tools for NPC agents using OpenAI Agents SDK.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from agents import function_tool, trace
from db.connection import get_db_connection

logger = logging.getLogger(__name__)

class EnvironmentData(BaseModel):
    """Data about the environment an NPC is in."""
    location: str = "Unknown"
    time_of_day: str = "Unknown"
    day: str = "Unknown"
    entities_present: List[Dict[str, Any]] = []
    description: Optional[str] = None
    
class ActionSignificance(BaseModel):
    """Evaluation of an action's significance."""
    is_significant: bool
    level: int
    reason: str
    
class ActionResult(BaseModel):
    """Result of executing an NPC action."""
    outcome: str
    emotional_impact: int = 0
    target_reactions: List[str] = []
    success: bool = True
    error: Optional[str] = None

@function_tool
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
        A dictionary with environment information
    """
    # Create a trace for debugging
    with trace(f"fetch_environment_{user_id}_{conversation_id}"):
        env_data = {
            "location": context.get("location", "Unknown"),
            "time_of_day": context.get("time_of_day", "Unknown"),
            "day": context.get("day", "Unknown"),
            "entities_present": []
        }
    
        # If location is unknown, try to fetch from database
        if env_data["location"] == "Unknown":
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    # Attempt to get location from CurrentRoleplay
                    cursor.execute("""
                        SELECT value
                        FROM CurrentRoleplay
                        WHERE user_id = %s
                          AND conversation_id = %s
                          AND key = 'CurrentLocation'
                    """, (user_id, conversation_id))
                    row = cursor.fetchone()
                    if row:
                        env_data["location"] = row[0]
    
                    # Attempt to fetch time info
                    cursor.execute("""
                        SELECT key, value
                        FROM CurrentRoleplay
                        WHERE user_id = %s
                          AND conversation_id = %s
                          AND key IN ('TimeOfDay', 'CurrentDay')
                    """, (user_id, conversation_id))
                    time_info_rows = cursor.fetchall()
                    for key, value in time_info_rows:
                        if key == "TimeOfDay":
                            env_data["time_of_day"] = value
                        elif key == "CurrentDay":
                            env_data["day"] = value
            except Exception as e:
                logger.error(f"Error fetching environment data: {e}")
        
        # If we have a location, fetch entities present
        if env_data["location"] != "Unknown":
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    # Get NPCs in this location
                    cursor.execute("""
                        SELECT npc_id, npc_name
                        FROM NPCStats
                        WHERE user_id = %s
                          AND conversation_id = %s
                          AND current_location = %s
                    """, (user_id, conversation_id, env_data["location"]))
                    entities = []
                    for npc_id, npc_name in cursor.fetchall():
                        entities.append({
                            "type": "npc",
                            "id": npc_id,
                            "name": npc_name
                        })
    
                    # Assume player is present (in a real system, you'd check)
                    entities.append({
                        "type": "player",
                        "id": user_id,
                        "name": "Player"
                    })
                    env_data["entities_present"] = entities
            except Exception as e:
                logger.error(f"Error fetching entities in location: {e}")
                
        # Get location description if available
        if env_data["location"] != "Unknown":
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT description
                        FROM Locations
                        WHERE name = %s
                    """, (env_data["location"],))
                    row = cursor.fetchone()
                    if row:
                        env_data["description"] = row[0]
            except Exception as e:
                logger.error(f"Error fetching location description: {e}")
    
        return env_data

@function_tool
def evaluate_action_significance(
    action: Dict[str, Any], 
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine if an action is significant enough to create a memory.

    Args:
        action: A dictionary describing the action (type, description, etc.)
        result: The outcome of that action (including 'outcome', 'emotional_impact')

    Returns:
        Evaluation of the action's significance
    """
    # Actions that are always significant
    always_significant = {"talk", "command", "mock", "confide", "praise", "emotional_outburst", "mask_slip"}
    if action.get("type") in always_significant:
        return {
            "is_significant": True,
            "level": 2,  # Medium significance
            "reason": f"Action type '{action.get('type')}' is inherently significant"
        }

    # Check for surprising outcomes
    outcome = result.get("outcome", "")
    if any(term in outcome.lower() for term in ["surprising", "unexpected", "shocked", "unusual"]):
        return {
            "is_significant": True,
            "level": 2,
            "reason": "Outcome was surprising or unexpected"
        }

    # Check for emotional impact
    emotional_impact = result.get("emotional_impact", 0)
    if abs(emotional_impact) >= 3:
        return {
            "is_significant": True,
            "level": 3,  # High significance
            "reason": f"Strong emotional impact ({emotional_impact})"
        }
    elif abs(emotional_impact) >= 1:
        return {
            "is_significant": True,
            "level": 1,  # Low significance
            "reason": f"Moderate emotional impact ({emotional_impact})"
        }

    # Default - not significant enough
    return {
        "is_significant": False,
        "level": 0,
        "reason": "Action was routine with minimal impact"
    }

@function_tool
async def execute_npc_action(
    npc_id: int,
    user_id: int,
    conversation_id: int,
    action: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute an NPC action and return the result.
    
    Args:
        npc_id: ID of the NPC
        user_id: ID of the user
        conversation_id: ID of the conversation
        action: Action to execute
        context: Context information
        
    Returns:
        Result of the action
    """
    # Create a trace for debugging
    with trace(f"execute_action_npc_{npc_id}", group_id=f"user_{user_id}_conv_{conversation_id}"):
        # Default result template
        result = {
            "outcome": f"NPC performed action: {action.get('description', 'unknown')}",
            "emotional_impact": 0,
            "target_reactions": [],
            "success": True
        }
    
        try:
            # Get NPC name
            npc_name = await _fetch_npc_name(npc_id, user_id, conversation_id)
            
            # Process different action types
            action_type = action.get("type", "unknown")
            target = action.get("target", "environment")
            
            if action_type == "talk":
                result = _handle_talk_action(npc_name, target, user_id, conversation_id)
            elif action_type == "command":
                result = _handle_command_action(npc_name, target, user_id, conversation_id)
            elif action_type == "mock":
                result = _handle_mock_action(npc_name, target, user_id, conversation_id)
            elif action_type == "observe":
                result["outcome"] = f"{npc_name} observes the surroundings carefully."
                result["emotional_impact"] = 0
            elif action_type == "leave":
                result["outcome"] = f"{npc_name} leaves the area."
                result["emotional_impact"] = 0
            elif action_type == "emotional_outburst":
                result["outcome"] = f"{npc_name} has an emotional outburst."
                result["emotional_impact"] = 3
                result["target_reactions"] = ["Others are surprised"]
            elif action_type == "mask_slip":
                result["outcome"] = f"{npc_name}'s mask slips, revealing a glimpse of their true nature."
                result["emotional_impact"] = 2
                result["target_reactions"] = ["Others notice the momentary change"]
            elif action_type == "scheduled":
                result["outcome"] = f"{npc_name} {action.get('description', 'performs a scheduled activity')}."
                result["emotional_impact"] = 0
            else:
                desc = action.get("description", "does something")
                result["outcome"] = f"{npc_name} {desc}"
                result["emotional_impact"] = 0
                
        except Exception as e:
            logger.error(f"Error executing NPC action: {e}")
            result["success"] = False
            result["error"] = str(e)
    
        return result

async def _fetch_npc_name(npc_id: int, user_id: int, conversation_id: int) -> str:
    """
    Fetch an NPC's name from the database.
    
    Args:
        npc_id: ID of the NPC
        user_id: ID of the user
        conversation_id: ID of the conversation
        
    Returns:
        Name of the NPC
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT npc_name
                FROM NPCStats
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (npc_id, user_id, conversation_id))
            row = cursor.fetchone()
            if row:
                return row[0]
            return f"NPC_{npc_id}"
    except Exception as e:
        logger.error(f"Error fetching NPC name: {e}")
        return f"NPC_{npc_id}"

def _handle_talk_action(npc_name: str, target: Any, user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Handle a talk action.
    """
    result = {
        "outcome": "",
        "emotional_impact": 1,
        "target_reactions": []
    }
    
    if target == "player":
        result["outcome"] = f"{npc_name} initiates a conversation with the player."
        result["target_reactions"] = ["Player listens attentively"]
    elif target == "group":
        result["outcome"] = f"{npc_name} speaks to the group."
        result["target_reactions"] = ["The group listens"]
    else:
        try:
            target_id = int(target)
            target_name = _fetch_target_name_sync(target_id, user_id, conversation_id)
            result["outcome"] = f"{npc_name} converses with {target_name}."
            result["target_reactions"] = [f"{target_name} responds to the conversation"]
        except (ValueError, TypeError):
            result["outcome"] = f"{npc_name} speaks, but the target is unclear."
            result["target_reactions"] = ["Others look confused"]
    
    return result

def _handle_command_action(npc_name: str, target: Any, user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Handle a command action.
    """
    result = {
        "outcome": "",
        "emotional_impact": 2,
        "target_reactions": []
    }
    
    if target == "player":
        result["outcome"] = f"{npc_name} gives a command to the player."
        result["target_reactions"] = ["Player considers whether to comply"]
    elif target == "group":
        result["outcome"] = f"{npc_name} issues a command to everyone present."
        result["target_reactions"] = ["The group reacts to the commanding tone"]
    else:
        try:
            target_id = int(target)
            target_name = _fetch_target_name_sync(target_id, user_id, conversation_id)
            result["outcome"] = f"{npc_name} commands {target_name} to do something."
            result["target_reactions"] = [f"{target_name} responds to the command"]
        except (ValueError, TypeError):
            result["outcome"] = f"{npc_name} attempts to give a command, but the target is unclear."
            result["target_reactions"] = ["Others look confused"]
    
    return result

def _handle_mock_action(npc_name: str, target: Any, user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Handle a mock action.
    """
    result = {
        "outcome": "",
        "emotional_impact": -2,
        "target_reactions": []
    }
    
    if target == "player":
        result["outcome"] = f"{npc_name} mockingly teases the player."
        result["target_reactions"] = ["Player feels targeted"]
    elif target == "group":
        result["outcome"] = f"{npc_name} makes mocking comments about the group."
        result["target_reactions"] = ["The group reacts with discomfort"]
    else:
        try:
            target_id = int(target)
            target_name = _fetch_target_name_sync(target_id, user_id, conversation_id)
            result["outcome"] = f"{npc_name} mocks {target_name} with cutting remarks."
            result["target_reactions"] = [f"{target_name} feels belittled"]
        except (ValueError, TypeError):
            result["outcome"] = f"{npc_name} makes mocking comments, but the target is unclear."
            result["target_reactions"] = ["Others look around uncomfortably"]
    
    return result

def _fetch_target_name_sync(target_id: int, user_id: int, conversation_id: int) -> str:
    """
    Fetch target NPC name synchronously.
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT npc_name
                FROM NPCStats
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (target_id, user_id, conversation_id))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return f"NPC_{target_id}"
    except Exception as e:
        logger.error(f"Error fetching target NPC name: {e}")
        return f"NPC_{target_id}"
