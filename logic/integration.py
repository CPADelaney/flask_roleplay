# logic/integration.py

"""
Central integration module for connecting the IntegratedNPCSystem with game systems.
This module handles initialization, caching, and provides utility functions.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from functools import lru_cache

from logic.fully_integrated_npc_system import IntegratedNPCSystem
from db.connection import get_db_connection_context, get_db_connection_pool

# Cache of active NPC systems
_npc_systems = {}

async def get_npc_system(user_id, conversation_id):
    key = f"{user_id}:{conversation_id}"
    if key not in _npc_systems:
        connection_pool = await get_db_connection_pool()
        _npc_systems[key] = IntegratedNPCSystem(user_id, conversation_id, connection_pool)
    return _npc_systems[key]

async def initialize_game_world(user_id: int, conversation_id: int, environment_desc: str) -> Dict[str, Any]:
    """
    Initialize a game world with NPCs, locations, and basic setup.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        environment_desc: Description of the environment
        
    Returns:
        Dictionary with initialization results
    """
    # Get the NPC system
    npc_system = await get_npc_system(user_id, conversation_id)
    
    # Get calendar names (days of the week, etc.)
    calendar_data = await get_calendar_info(user_id, conversation_id)
    day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    # Create initial NPCs (3-5)
    num_npcs = 5
    npc_ids = await npc_system.create_multiple_npcs(environment_desc, day_names, count=num_npcs)
    
    # Get NPC details
    npcs = []
    for npc_id in npc_ids:
        npc_details = await npc_system.get_npc_details(npc_id)
        if npc_details:
            npcs.append(npc_details)
    
    # Initialize time
    await npc_system.set_game_time(1040, 6, 15, "Morning")
    
    # Create initial relationships between NPCs
    relationships = []
    for i in range(len(npc_ids)):
        for j in range(i+1, len(npc_ids)):
            npc1_id = npc_ids[i]
            npc2_id = npc_ids[j]
            
            # Create relationship
            rel = await npc_system.create_relationship(
                "npc", npc1_id, 
                "npc", npc2_id
            )
            
            relationships.append(rel)
    
    # Create a player-NPC relationship for each NPC
    for npc_id in npc_ids:
        await npc_system.create_relationship(
            "player", user_id,
            "npc", npc_id
        )
    
    # Possibly create an NPC group
    group = await npc_system.create_npc_group(
        "Core Group", 
        "The main social circle in this environment", 
        npc_ids[:3]  # First 3 NPCs
    )
    
    return {
        "initialized": True,
        "npcs_created": len(npcs),
        "relationships_created": len(relationships),
        "groups_created": 1 if group else 0,
        "environment": environment_desc
    }

async def get_calendar_info(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Get calendar information for the game world.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        
    Returns:
        Dictionary with calendar information
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
            """, user_id, conversation_id)
            
            if not row:
                # Default calendar
                return {
                    "year_name": "The Eternal Cycle",
                    "months": [
                        "Aurora", "Blaze", "Crimson", "Dusk",
                        "Ember", "Frost", "Gleam", "Haze",
                        "Iris", "Jade", "Knell", "Lumen"
                    ],
                    "days": [
                        "Sol", "Luna", "Terra", "Vesta", 
                        "Mercury", "Venus", "Mars"
                    ]
                }
            
            # Parse calendar data
            calendar_data = row['value']
            if isinstance(calendar_data, str):
                try:
                    calendar_data = json.loads(calendar_data)
                except json.JSONDecodeError:
                    # Default calendar on error
                    return {
                        "year_name": "The Eternal Cycle",
                        "months": [
                            "Aurora", "Blaze", "Crimson", "Dusk",
                            "Ember", "Frost", "Gleam", "Haze",
                            "Iris", "Jade", "Knell", "Lumen"
                        ],
                        "days": [
                            "Sol", "Luna", "Terra", "Vesta", 
                            "Mercury", "Venus", "Mars"
                        ]
                    }
            
            return calendar_data
    except Exception as e:
        logging.error(f"Error getting calendar info: {e}")
        # Default calendar on error
        return {
            "year_name": "The Eternal Cycle",
            "months": [
                "Aurora", "Blaze", "Crimson", "Dusk",
                "Ember", "Frost", "Gleam", "Haze",
                "Iris", "Jade", "Knell", "Lumen"
            ],
            "days": [
                "Sol", "Luna", "Terra", "Vesta", 
                "Mercury", "Venus", "Mars"
            ]
        }

async def process_end_of_day(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Process end-of-day actions: memory fading, relationship maintenance, etc.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        
    Returns:
        Dictionary with processing results
    """
    # Get the NPC system
    npc_system = await get_npc_system(user_id, conversation_id)
    
    # 1. Get current time
    year, month, day, time_of_day = await npc_system.get_current_game_time()
    
    # 2. Advance to "Night" if needed
    if time_of_day != "Night":
        # Advance to Night
        await npc_system.set_game_time(year, month, day, "Night")
    
    # 3. Process night activities for each NPC
    from logic.time_cycle import nightly_maintenance
    await nightly_maintenance(user_id, conversation_id)
    
    # 4. Generate dream sequence
    dream = None
    from logic.narrative_events import add_dream_sequence
    dream_result = await add_dream_sequence(user_id, conversation_id)
    if dream_result:
        dream = dream_result.get("text")
    
    # 5. Advance to next day, Morning
    new_day = day + 1
    new_month = month
    new_year = year
    
    # Check month/year boundaries
    if new_day > 30:  # Simplified 30-day months
        new_day = 1
        new_month += 1
        
        if new_month > 12:
            new_month = 1
            new_year += 1
    
    await npc_system.set_game_time(new_year, new_month, new_day, "Morning")
    
    return {
        "processed": True,
        "dream": dream,
        "new_date": f"{new_year}-{new_month}-{new_day}, Morning",
        "previous_date": f"{year}-{month}-{day}, Night"
    }

async def handle_game_event(user_id: int, conversation_id: int, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a game event using the IntegratedNPCSystem.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        event_type: Type of event
        event_data: Event data
        
    Returns:
        Dictionary with event handling results
    """
    # Get the NPC system
    npc_system = await get_npc_system(user_id, conversation_id)
    
    result = {"processed": False}
    
    # Handle different event types
    if event_type == "player_action":
        # Process player action
        action_type = event_data.get("action_type", "talk")
        content = event_data.get("content", "")
        location = event_data.get("location")
        
        context = {"location": location} if location else {}
        
        # Create player action
        player_action = {
            "type": action_type,
            "description": content,
            "target_location": location
        }
        
        # Handle with integrated system
        nearby_npcs = await get_nearby_npcs(user_id, conversation_id, location)
        responses = []
        
        for npc in nearby_npcs[:3]:  # Limit to 3 NPCs
            interaction_result = await npc_system.handle_npc_interaction(
                npc["npc_id"],
                determine_interaction_type(content),
                content,
                context
            )
            
            if interaction_result:
                responses.append({
                    "npc_id": npc["npc_id"],
                    "npc_name": npc["npc_name"],
                    "response": interaction_result
                })
        
        result = {
            "processed": True,
            "responses": responses
        }
        
    elif event_type == "relationship_event":
        # Handle relationship event
        relationship_type = event_data.get("relationship_type")
        entity1_type = event_data.get("entity1_type")
        entity1_id = event_data.get("entity1_id")
        entity2_type = event_data.get("entity2_type")
        entity2_id = event_data.get("entity2_id")
        
        if all([relationship_type, entity1_type, entity1_id, entity2_type, entity2_id]):
            relationship = await npc_system.create_relationship(
                entity1_type, entity1_id,
                entity2_type, entity2_id,
                relationship_type
            )
            
            result = {
                "processed": True,
                "relationship": relationship
            }
    
    elif event_type == "time_advance":
        # Advance time
        activity_type = event_data.get("activity_type", "extended_conversation")
        time_result = await npc_system.advance_time_with_activity(activity_type)
        
        result = {
            "processed": True,
            "time_result": time_result
        }
    
    elif event_type == "multi_npc_scene":
        # Generate multi-NPC scene
        npc_ids = event_data.get("npc_ids", [])
        location = event_data.get("location")
        include_player = event_data.get("include_player", True)
        
        if npc_ids:
            scene = await npc_system.generate_multi_npc_scene(
                npc_ids, location, include_player
            )
            
            result = {
                "processed": True,
                "scene": scene
            }
    
    return result

async def get_nearby_npcs(user_id: int, conversation_id: int, location: str = None) -> List[Dict[str, Any]]:
    """
    Get NPCs that are at the current location or nearby.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        location: Location to check
        
    Returns:
        List of NPC data
    """
    nearby_npcs = []
    
    try:
        async with get_db_connection_context() as conn:
            if location:
                # Get NPCs at the specific location
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 
                    AND current_location=$3
                    LIMIT 5
                """, user_id, conversation_id, location)
            else:
                # Get any NPCs (prioritize ones that are introduced)
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY introduced DESC
                    LIMIT 5
                """, user_id, conversation_id)
                
            for row in rows:
                nearby_npcs.append({
                    "npc_id": row['npc_id'],
                    "npc_name": row['npc_name'],
                    "current_location": row['current_location'],
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty']
                })
            
        return nearby_npcs
    except Exception as e:
        logging.error(f"Error getting nearby NPCs: {e}")
        return []

def determine_interaction_type(player_input: str) -> str:
    """
    Determine the type of interaction based on player input.
    
    Args:
        player_input: Player's input text
        
    Returns:
        Interaction type
    """
    player_input_lower = player_input.lower()
    
    if "no" in player_input_lower or "won't" in player_input_lower or "refuse" in player_input_lower:
        return "defiant_response"
    elif "yes" in player_input_lower or "okay" in player_input_lower or "sure" in player_input_lower:
        return "submissive_response"
    elif any(word in player_input_lower for word in ["cute", "pretty", "hot", "sexy", "beautiful", "attractive"]):
        return "flirtatious_remark"
    else:
        return "extended_conversation"
