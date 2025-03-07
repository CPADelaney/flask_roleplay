# logic/enhanced_time_cycle.py

"""
Enhanced time cycle module that integrates with IntegratedNPCSystem.
This module handles time advancement, event scheduling, and related operations.
"""

import logging
import random
import json
from typing import Dict, List, Optional, Union, Any, Tuple

from db.connection import get_db_connection
from logic.fully_integrated_npc_system import IntegratedNPCSystem
from logic.time_cycle import (
    get_current_time, set_current_time, nightly_maintenance,
    TIME_PHASES, TIME_CONSUMING_ACTIVITIES, OPTIONAL_ACTIVITIES
)

# Constants
EVENT_PROBABILITY = 0.3  # 30% chance of an event occurring when time advances

async def process_time_advancement(user_id: int, conversation_id: int, activity_type: str) -> Dict[str, Any]:
    """
    Process time advancement with full integration of IntegratedNPCSystem.
    This is a comprehensive function that handles time changes, NPC updates,
    event generation, and stat effects.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        activity_type: Type of activity
        
    Returns:
        Dictionary with time advancement results
    """
    # Initialize IntegratedNPCSystem
    npc_system = IntegratedNPCSystem(user_id, conversation_id)
    
    # Use the integrated system to advance time and process related events
    result = await npc_system.advance_time_with_activity(activity_type)
    
    # Check if time was advanced
    if result.get("time_advanced", False):
        # Get old and new time information
        old_year, old_month, old_day, old_time = get_current_time(user_id, conversation_id)
        new_year = result.get("new_year", old_year)
        new_month = result.get("new_month", old_month)
        new_day = result.get("new_day", old_day)
        new_time = result.get("new_time", old_time)
        
        # Check if day rolled over
        if new_day != old_day and new_time == "Morning":
            # Perform nightly maintenance
            await nightly_maintenance(user_id, conversation_id)
            logging.info(f"Performed nightly maintenance for user={user_id}, conversation={conversation_id}")
        
        # Process NPC activities for the new time period
        await update_npcs_for_new_time(user_id, conversation_id, npc_system, new_day, new_time)
        
        # Generate random events with a certain probability
        random_events = []
        if random.random() < EVENT_PROBABILITY:
            events = await generate_random_events(user_id, conversation_id, npc_system, new_time)
            if events:
                random_events.extend(events)
                result["random_events"] = random_events
    
    return result

async def update_npcs_for_new_time(user_id: int, conversation_id: int, 
                                 npc_system: IntegratedNPCSystem,
                                 day: int, time_of_day: str) -> None:
    """
    Update NPCs for a new time of day using the integrated system.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        npc_system: IntegratedNPCSystem instance
        day: Current day
        time_of_day: Current time of day
    """
    # Get all NPCs
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT npc_id, npc_name 
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    
    npcs = []
    for row in cursor.fetchall():
        npc_id, npc_name = row
        npcs.append({"npc_id": npc_id, "npc_name": npc_name})
    
    cursor.close()
    conn.close()
    
    # For each NPC, perform daily activity update
    for npc in npcs:
        try:
            await npc_system.perform_npc_daily_activity(npc["npc_id"], time_of_day)
        except Exception as e:
            logging.warning(f"Error updating NPC {npc['npc_name']} for time {time_of_day}: {e}")
    
    # Additionally, check for mask slippage for 30% of NPCs
    if npcs:
        npcs_to_check = random.sample(npcs, min(len(npcs), max(1, len(npcs) // 3)))
        for npc in npcs_to_check:
            try:
                slippage_events = await npc_system.check_for_mask_slippage(npc["npc_id"])
                if slippage_events:
                    logging.info(f"Mask slippage detected for NPC {npc['npc_name']} at {time_of_day}")
            except Exception as e:
                logging.warning(f"Error checking mask slippage for NPC {npc['npc_name']}: {e}")

async def generate_random_events(user_id: int, conversation_id: int, 
                               npc_system: IntegratedNPCSystem,
                               time_of_day: str) -> List[Dict[str, Any]]:
    """
    Generate random events appropriate for the time of day.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        npc_system: IntegratedNPCSystem instance
        time_of_day: Current time of day
        
    Returns:
        List of generated event data
    """
    events = []
    
    # Get context data
    current_location = await get_current_location(user_id, conversation_id)
    nearby_npcs = await get_nearby_npcs(user_id, conversation_id, current_location)
    
    # If we have NPCs, potentially generate:
    if nearby_npcs:
        # 1. Overheard conversation
        if time_of_day in ["Morning", "Afternoon"] and len(nearby_npcs) >= 2 and random.random() < 0.4:
            npc_ids = [npc["npc_id"] for npc in nearby_npcs[:2]]
            about_player = random.choice([True, False])
            
            try:
                conversation = await npc_system.generate_overheard_conversation(
                    npc_ids, 
                    topic=None, 
                    about_player=about_player
                )
                
                if conversation:
                    events.append({
                        "type": "overheard_conversation",
                        "data": conversation
                    })
            except Exception as e:
                logging.warning(f"Error generating overheard conversation: {e}")
        
        # 2. Multi-NPC scene
        elif time_of_day in ["Evening", "Night"] and len(nearby_npcs) >= 2 and random.random() < 0.3:
            npc_ids = [npc["npc_id"] for npc in nearby_npcs[:3]]
            
            try:
                scene = await npc_system.generate_multi_npc_scene(
                    npc_ids,
                    location=current_location,
                    include_player=True
                )
                
                if scene:
                    events.append({
                        "type": "multi_npc_scene",
                        "data": scene
                    })
            except Exception as e:
                logging.warning(f"Error generating multi-NPC scene: {e}")
    
    # 3. Narrative progression events
    from logic.narrative_progression import (
        check_for_personal_revelations, 
        check_for_narrative_moments,
        check_for_npc_revelations
    )
    
    try:
        # Personal revelation
        if random.random() < 0.2:
            revelation = await check_for_personal_revelations(user_id, conversation_id)
            if revelation:
                events.append(revelation)
        
        # Narrative moment
        if random.random() < 0.15:
            moment = await check_for_narrative_moments(user_id, conversation_id)
            if moment:
                events.append(moment)
        
        # NPC revelation
        if random.random() < 0.25:
            npc_revelation = await check_for_npc_revelations(user_id, conversation_id)
            if npc_revelation:
                events.append(npc_revelation)
    except Exception as e:
        logging.warning(f"Error generating narrative progression events: {e}")
    
    # 4. Relationship events
    from logic.social_links import (
        check_for_relationship_crossroads,
        check_for_relationship_ritual
    )
    
    try:
        # Relationship crossroads
        if random.random() < 0.1:
            crossroads = await check_for_relationship_crossroads(user_id, conversation_id)
            if crossroads:
                events.append({
                    "type": "relationship_crossroads",
                    "data": crossroads
                })
        
        # Relationship ritual
        if random.random() < 0.1:
            ritual = await check_for_relationship_ritual(user_id, conversation_id)
            if ritual:
                events.append({
                    "type": "relationship_ritual",
                    "data": ritual
                })
    except Exception as e:
        logging.warning(f"Error generating relationship events: {e}")
    
    return events

async def get_current_location(user_id: int, conversation_id: int) -> Optional[str]:
    """
    Get the current location from CurrentRoleplay.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        
    Returns:
        Current location or None
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s AND conversation_id=%s AND key='CurrentLocation'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        if row:
            return row[0]
        
        return None
    finally:
        cursor.close()
        conn.close()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if location:
            # Get NPCs at the specific location
            cursor.execute("""
                SELECT npc_id, npc_name, current_location, dominance, cruelty
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s 
                AND current_location=%s
                LIMIT 5
            """, (user_id, conversation_id, location))
        else:
            # Get any NPCs (prioritize ones that are introduced)
            cursor.execute("""
                SELECT npc_id, npc_name, current_location, dominance, cruelty
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s
                ORDER BY introduced DESC
                LIMIT 5
            """, (user_id, conversation_id))
            
        nearby_npcs = []
        for row in cursor.fetchall():
            npc_id, npc_name, current_location, dominance, cruelty = row
            nearby_npcs.append({
                "npc_id": npc_id,
                "npc_name": npc_name,
                "current_location": current_location,
                "dominance": dominance,
                "cruelty": cruelty
            })
        
        return nearby_npcs
    finally:
        cursor.close()
        conn.close()

class EnhancedTimeManager:
    """
    Enhanced time manager that integrates with IntegratedNPCSystem.
    Provides a unified interface for time-related operations.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the time manager.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_system = IntegratedNPCSystem(user_id, conversation_id)
    
    async def get_current_time(self) -> Tuple[int, int, int, str]:
        """
        Get the current game time.
        
        Returns:
            Tuple of (year, month, day, time_of_day)
        """
        return await self.npc_system.get_current_game_time()
    
    async def set_time(self, year: int, month: int, day: int, time_of_day: str) -> bool:
        """
        Set the game time.
        
        Args:
            year: Year to set
            month: Month to set
            day: Day to set
            time_of_day: Time of day to set
            
        Returns:
            True if successful
        """
        return await self.npc_system.set_game_time(year, month, day, time_of_day)
    
    async def advance_time(self, activity_type: str) -> Dict[str, Any]:
        """
        Advance time based on an activity type.
        
        Args:
            activity_type: Type of activity
            
        Returns:
            Dictionary with time advancement results
        """
        return await process_time_advancement(self.user_id, self.conversation_id, activity_type)
    
    async def process_activity(self, player_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a player's activity, determining if time should advance.
        
        Args:
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Dictionary with processing results
        """
        return await self.npc_system.process_player_activity(player_input, context)
    
    async def end_day(self) -> Dict[str, Any]:
        """
        Process end-of-day actions.
        
        Returns:
            Dictionary with processing results
        """
        # Get current time
        year, month, day, time_of_day = await self.get_current_time()
        
        # Advance to Night if needed
        if time_of_day != "Night":
            await self.set_time(year, month, day, "Night")
        
        # Run nightly maintenance
        await nightly_maintenance(self.user_id, self.conversation_id)
        
        # Generate dream
        from logic.narrative_progression import add_dream_sequence
        dream_result = await add_dream_sequence(self.user_id, self.conversation_id)
        
        # Advance to next day, Morning
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
        
        await self.set_time(new_year, new_month, new_day, "Morning")
        
        return {
            "processed": True,
            "dream": dream_result.get("text") if dream_result else None,
            "new_date": f"{new_year}-{new_month}-{new_day}, Morning",
            "previous_date": f"{year}-{month}-{day}, Night"
        }
