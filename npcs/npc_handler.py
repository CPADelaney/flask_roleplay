# npcs/npc_handler.py

"""
NPC interaction handler for managing interactions between NPCs and players.
Refactored from npc_handler_agent.py.
"""

import logging
import json
import asyncio
import random
from typing import List, Dict, Any, Optional
import os

from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field

from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from logic.activities_logic import get_all_activities, filter_activities_for_npc
from npcs.npc_learning_adaptation import NPCLearningAdaptation, NPCLearningManager

# Configuration
DB_DSN = os.getenv("DB_DSN")

class NPCInteractionInput(BaseModel):
    npc_id: int
    player_input: str
    context: Dict[str, Any] = Field(default_factory=dict)
    interaction_type: str = "standard_interaction"

class NPCInteractionOutput(BaseModel):
    npc_id: int
    npc_name: str
    response: str
    stat_changes: Dict[str, int] = Field(default_factory=dict)
    memory_created: bool = False

class NPCHandler:
    """Handles NPC interactions with players and other NPCs"""
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the NPC handler.
        
        Args:
            user_id: User/player ID
            conversation_id: Conversation/scene ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize the learning manager for handling multiple NPCs
        self.learning_manager = NPCLearningManager(user_id, conversation_id)
        
        # Initialize agent for handling NPC interactions
        self.interaction_agent = Agent(
            name="InteractionHandler",
            instructions="""
            You generate realistic NPC responses to player interactions in a roleplaying game with subtle femdom elements.
            
            Each response should:
            - Match the NPC's personality and stats (dominance, cruelty, etc.)
            - Be appropriate to the context and location
            - Incorporate subtle hints of control when appropriate
            - Consider the NPC's history with the player
            - Possibly suggest stat changes based on the interaction
            
            The responses should maintain a balance between mundane everyday interactions and 
            subtle power dynamics, with control elements hidden beneath friendly facades.
            """,
            output_type=NPCInteractionOutput,
            tools=[
                function_tool(self.get_npc_details),
                function_tool(self.get_npc_memory),
                function_tool(self.get_relationship_details)
            ]
        )

    async def handle_interaction(
        self,
        npc_id: int,
        interaction_type: str,
        player_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an interaction between the player and an NPC.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction (e.g., "standard_interaction", "defiant_response", etc.)
            player_input: Player's input text
            context: Additional context for the interaction (optional)
            
        Returns:
            Dictionary with the NPC's response
        """
        # Get NPC details
        npc_details = await self.get_npc_details(npc_id)
        
        # Get NPC memories
        memories = await self.get_npc_memory(npc_id)
        
        # Get player relationship
        relationship = await self.get_relationship_details("npc", npc_id, "player", 0)
        
        # Create prompt for the interaction handler
        context_str = json.dumps(context) if context else "{}"
        memories_str = json.dumps(memories)
        relationship_str = json.dumps(relationship)
        
        prompt = f"""
        Generate a response for {npc_details['npc_name']} to the player's input:
        
        "{player_input}"
        
        NPC Details:
        {json.dumps(npc_details, indent=2)}
        
        Recent memories:
        {memories_str}
        
        Relationship with player:
        {relationship_str}
        
        Interaction type: {interaction_type}
        Additional context: {context_str}
        
        Generate a response that:
        - Is consistent with the NPC's personality and stats
        - Considers their memories and relationship with the player
        - Fits the interaction type and context
        - Includes subtle elements of control when appropriate
        - Suggests any relevant stat changes
        
        The response should maintain a balance between mundane interaction
        and subtle power dynamics appropriate to the NPC's character.
        """
        
        # Run the interaction handler
        result = await Runner.run(
            self.interaction_agent,
            prompt
        )
        
        response = result.final_output
        
        # Store the interaction in memory if it's significant
        if response.memory_created:
            await self._store_interaction_memory(
                npc_id, 
                player_input, 
                response.response
            )
        
        # Apply stat changes if any
        if response.stat_changes:
            await self._apply_stat_changes(npc_id, response.stat_changes)
        
        # INTEGRATION: Record interaction with the learning adaptation system
        await self._record_interaction_for_learning(
            npc_id,
            interaction_type,
            player_input,
            response.response,
            context
        )
        
        return response.dict()

    async def _store_interaction_memory(self, npc_id: int, player_input: str, response: str) -> None:
        """
        Store an interaction in the NPC's memory using canon system.
        
        Args:
            npc_id: ID of the NPC
            player_input: Player's input
            response: NPC's response
        """
        try:
            # Create unified memory entry (allowed)
            memory_text = f"Interaction with player: {player_input} - Response: {response}"
            memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="medium",
                tags=["interaction", "player_interaction"],
                emotional=True
            )
            
            # Also store in unified_memories table for compatibility
            async with get_db_connection_context() as conn:
                # Create context for canon
                ctx = type('obj', (object,), {
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                    'npc_id': npc_id
                })
                
                # Use canon to create memory entry
                await canon.create_journal_entry(
                    ctx, conn,
                    entry_type="npc_interaction",
                    entry_text=memory_text,
                    tags=["interaction", "npc", f"npc_{npc_id}"],
                    importance=0.5
                )
                
        except Exception as e:
            logger.error(f"Error storing interaction memory: {e}")

    async def _apply_stat_changes(self, npc_id: int, stat_changes: Dict[str, int]) -> None:
        """
        Apply stat changes to an NPC using LoreSystem.
        
        Args:
            npc_id: ID of the NPC
            stat_changes: Dictionary of stat changes
        """
        try:
            # Get LoreSystem instance
            from lore.lore_system import LoreSystem
            lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            
            # Create context for governance
            ctx = type('obj', (object,), {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'npc_id': npc_id
            })
            
            # Get current stats first
            async with get_db_connection_context() as conn:
                current_stats = {}
                valid_stats = ["dominance", "cruelty", "closeness", "trust", "respect", "intensity"]
                
                for stat in valid_stats:
                    if stat in stat_changes:
                        row = await conn.fetchrow(
                            f"SELECT {stat} FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3",
                            self.user_id, self.conversation_id, npc_id
                        )
                        if row:
                            current_stats[stat] = row[stat]
            
            # Calculate new values and prepare updates
            updates = {}
            for stat, change in stat_changes.items():
                if stat in current_stats:
                    new_value = max(0, min(100, current_stats[stat] + change))
                    if new_value != current_stats[stat]:
                        updates[stat] = new_value
            
            # Apply updates if any
            if updates:
                result = await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCStats",
                    entity_identifier={"npc_id": npc_id},
                    updates=updates,
                    reason=f"Interaction caused stat changes: {stat_changes}"
                )
                
                if result.get("status") != "committed":
                    logger.error(f"Failed to apply stat changes: {result}")
                    
        except Exception as e:
            logger.error(f"Error applying stat changes: {e}")

    async def get_npc_details(self, npc_id: int) -> Dict[str, Any]:
        """
        Get details about a specific NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with NPC details
        """
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_id, npc_name, introduced, archetypes, archetype_summary, 
                       archetype_extras_summary, physical_description, relationships,
                       dominance, cruelty, closeness, trust, respect, intensity,
                       hobbies, personality_traits, likes, dislikes, affiliations,
                       schedule, current_location, sex, age
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                LIMIT 1
            """, self.user_id, self.conversation_id, npc_id)
            
            if not row:
                return {"error": "NPC not found"}
            
            # Process JSON fields
            archetypes = row["archetypes"]
            if archetypes:
                if isinstance(archetypes, str):
                    try:
                        archetypes = json.loads(archetypes)
                    except (json.JSONDecodeError, TypeError):
                        archetypes = []
            else:
                archetypes = []
            
            relationships = row["relationships"]
            if relationships:
                if isinstance(relationships, str):
                    try:
                        relationships = json.loads(relationships)
                    except (json.JSONDecodeError, TypeError):
                        relationships = {}
            else:
                relationships = {}
            
            hobbies = row["hobbies"]
            if hobbies:
                if isinstance(hobbies, str):
                    try:
                        hobbies = json.loads(hobbies)
                    except (json.JSONDecodeError, TypeError):
                        hobbies = []
            else:
                hobbies = []
            
            personality_traits = row["personality_traits"]
            if personality_traits:
                if isinstance(personality_traits, str):
                    try:
                        personality_traits = json.loads(personality_traits)
                    except (json.JSONDecodeError, TypeError):
                        personality_traits = []
            else:
                personality_traits = []
            
            likes = row["likes"]
            if likes:
                if isinstance(likes, str):
                    try:
                        likes = json.loads(likes)
                    except (json.JSONDecodeError, TypeError):
                        likes = []
            else:
                likes = []
            
            dislikes = row["dislikes"]
            if dislikes:
                if isinstance(dislikes, str):
                    try:
                        dislikes = json.loads(dislikes)
                    except (json.JSONDecodeError, TypeError):
                        dislikes = []
            else:
                dislikes = []
            
            affiliations = row["affiliations"]
            if affiliations:
                if isinstance(affiliations, str):
                    try:
                        affiliations = json.loads(affiliations)
                    except (json.JSONDecodeError, TypeError):
                        affiliations = []
            else:
                affiliations = []
            
            schedule = row["schedule"]
            if schedule:
                if isinstance(schedule, str):
                    try:
                        schedule = json.loads(schedule)
                    except (json.JSONDecodeError, TypeError):
                        schedule = {}
            else:
                schedule = {}
            
            return {
                "npc_id": row["npc_id"],
                "npc_name": row["npc_name"],
                "introduced": row["introduced"],
                "archetypes": archetypes,
                "archetype_summary": row["archetype_summary"],
                "archetype_extras_summary": row["archetype_extras_summary"],
                "physical_description": row["physical_description"],
                "relationships": relationships,
                "dominance": row["dominance"],
                "cruelty": row["cruelty"],
                "closeness": row["closeness"],
                "trust": row["trust"],
                "respect": row["respect"],
                "intensity": row["intensity"],
                "hobbies": hobbies,
                "personality_traits": personality_traits,
                "likes": likes,
                "dislikes": dislikes,
                "affiliations": affiliations,
                "schedule": schedule,
                "current_location": row["current_location"],
                "sex": row["sex"],
                "age": row["age"]
            }

    async def get_npc_memory(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get memories associated with an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of memory objects
        """
        async with get_db_connection_context() as conn:
            # First check if the NPC has memories in the NPCStats table
            row = await conn.fetchrow("""
                SELECT memory
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                LIMIT 1
            """, self.user_id, self.conversation_id, npc_id)
            
            if row and row["memory"]:
                try:
                    if isinstance(row["memory"], str):
                        memories = json.loads(row["memory"])
                    else:
                        memories = row["memory"]
                    
                    return memories
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Try the legacy NPCMemories table
            rows = await conn.fetch("""
                SELECT id, memory_text, emotional_intensity, significance, memory_type
                FROM NPCMemories
                WHERE npc_id=$1 AND status='active'
                ORDER BY timestamp DESC
                LIMIT 10
            """, npc_id)
            
            if rows:
                memories = []
                for row in rows:
                    memories.append({
                        "id": row["id"],
                        "memory_text": row["memory_text"],
                        "emotional_intensity": row["emotional_intensity"],
                        "significance": row["significance"],
                        "memory_type": row["memory_type"]
                    })
                
                return memories
            
            # Finally, try the unified_memories table
            rows = await conn.fetch("""
                SELECT id, memory_text, emotional_intensity, significance, memory_type
                FROM unified_memories
                WHERE entity_type='npc' AND entity_id=$1 AND user_id=$2 AND conversation_id=$3 AND status='active'
                ORDER BY timestamp DESC
                LIMIT 10
            """, npc_id, self.user_id, self.conversation_id)
            
            memories = []
            for row in rows:
                memories.append({
                    "id": row["id"],
                    "memory_text": row["memory_text"],
                    "emotional_intensity": row["emotional_intensity"],
                    "significance": row["significance"],
                    "memory_type": row["memory_type"]
                })
            
            return memories

    async def get_relationship_details(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int
    ) -> Dict[str, Any]:
        """
        Get relationship details between two entities.
        
        Args:
            entity1_type: Type of the first entity (e.g., "npc", "player")
            entity1_id: ID of the first entity
            entity2_type: Type of the second entity
            entity2_id: ID of the second entity
            
        Returns:
            Dictionary with relationship details
        """
        async with get_db_connection_context() as conn:
            # Try both orientations of the relationship
            for e1t, e1i, e2t, e2i in [(entity1_type, entity1_id, entity2_type, entity2_id),
                                       (entity2_type, entity2_id, entity1_type, entity1_id)]:
                row = await conn.fetchrow("""
                    SELECT link_id, link_type, link_level, link_history, dynamics, 
                           group_interaction, relationship_stage, experienced_crossroads,
                           experienced_rituals
                    FROM SocialLinks
                    WHERE user_id=$1 AND conversation_id=$2
                      AND entity1_type=$3 AND entity1_id=$4
                      AND entity2_type=$5 AND entity2_id=$6
                    LIMIT 1
                """, self.user_id, self.conversation_id, e1t, e1i, e2t, e2i)
                
                if row:
                    # Process JSON fields
                    link_history = row["link_history"]
                    if link_history:
                        try:
                            if isinstance(link_history, str):
                                link_history = json.loads(link_history)
                        except (json.JSONDecodeError, TypeError):
                            link_history = []
                    else:
                        link_history = []
                    
                    dynamics = row["dynamics"]
                    if dynamics:
                        try:
                            if isinstance(dynamics, str):
                                dynamics = json.loads(dynamics)
                        except (json.JSONDecodeError, TypeError):
                            dynamics = {}
                    else:
                        dynamics = {}
                    
                    experienced_crossroads = row["experienced_crossroads"]
                    if experienced_crossroads:
                        try:
                            if isinstance(experienced_crossroads, str):
                                experienced_crossroads = json.loads(experienced_crossroads)
                        except (json.JSONDecodeError, TypeError):
                            experienced_crossroads = {}
                    else:
                        experienced_crossroads = {}
                    
                    experienced_rituals = row["experienced_rituals"]
                    if experienced_rituals:
                        try:
                            if isinstance(experienced_rituals, str):
                                experienced_rituals = json.loads(experienced_rituals)
                        except (json.JSONDecodeError, TypeError):
                            experienced_rituals = {}
                    else:
                        experienced_rituals = {}
                    
                    return {
                        "link_id": row["link_id"],
                        "entity1_type": e1t,
                        "entity1_id": e1i,
                        "entity2_type": e2t,
                        "entity2_id": e2i,
                        "link_type": row["link_type"],
                        "link_level": row["link_level"],
                        "link_history": link_history,
                        "dynamics": dynamics,
                        "group_interaction": row["group_interaction"],
                        "relationship_stage": row["relationship_stage"],
                        "experienced_crossroads": experienced_crossroads,
                        "experienced_rituals": experienced_rituals
                    }
            
            # No relationship found
            return {
                "entity1_type": entity1_type,
                "entity1_id": entity1_id,
                "entity2_type": entity2_type,
                "entity2_id": entity2_id,
                "link_type": "none",
                "link_level": 0,
                "link_history": [],
                "dynamics": {},
                "relationship_stage": "strangers",
                "experienced_crossroads": {},
                "experienced_rituals": {}
            }

    async def get_nearby_npcs(self, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get NPCs that are at a specific location.
        
        Args:
            location: Location to filter by (optional)
            
        Returns:
            List of nearby NPCs
        """
        async with get_db_connection_context() as conn:
            if location:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 
                    AND current_location=$3
                    ORDER BY introduced DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id, location)
            else:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY introduced DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
            
            nearby_npcs = []
            for row in rows:
                nearby_npcs.append({
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"],
                    "current_location": row["current_location"],
                    "dominance": row["dominance"],
                    "cruelty": row["cruelty"]
                })
            
            return nearby_npcs

    async def process_daily_npc_activities(self) -> Dict[str, Any]:
        """
        Process daily activities for all NPCs using canon system.
        Update locations, create memories, and handle NPC interactions.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Get LoreSystem instance
            from lore.core import canon
            from lore.lore_system import LoreSystem
            lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            
            # Create context for governance
            ctx = type('obj', (object,), {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            })
            
            async with get_db_connection_context() as conn:
                # Get current time data
                year, month, day, time_of_day = 1, 1, 1, "Morning"
                
                for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
                    row = await conn.fetchrow(
                        "SELECT value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key=$3",
                        self.user_id, self.conversation_id, key
                    )
                    
                    if row:
                        if key == "CurrentYear":
                            year = int(row["value"]) if row["value"].isdigit() else 1
                        elif key == "CurrentMonth":
                            month = int(row["value"]) if row["value"].isdigit() else 1
                        elif key == "CurrentDay":
                            day = int(row["value"]) if row["value"].isdigit() else 1
                        elif key == "TimeOfDay":
                            time_of_day = row["value"]
                
                # Get day name
                row = await conn.fetchrow(
                    "SELECT value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'",
                    self.user_id, self.conversation_id
                )
                
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                if row:
                    try:
                        calendar_data = json.loads(row["value"])
                        if "days" in calendar_data:
                            day_names = calendar_data["days"]
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                day_of_week = day_names[day % len(day_names)]
                
                # Get all NPCs
                rows = await conn.fetch(
                    "SELECT npc_id, npc_name, schedule, current_location FROM NPCStats WHERE user_id=$1 AND conversation_id=$2",
                    self.user_id, self.conversation_id
                )
                
                results = []
                
                for row in rows:
                    npc_id = row["npc_id"]
                    npc_name = row["npc_name"]
                    schedule = row["schedule"]
                    current_location = row["current_location"]
                    
                    # Parse schedule
                    if schedule:
                        try:
                            if isinstance(schedule, str):
                                schedule_data = json.loads(schedule)
                            else:
                                schedule_data = schedule
                            
                            # Check if this day is in the schedule
                            if day_of_week in schedule_data:
                                day_schedule = schedule_data[day_of_week]
                                
                                # Check if the current time period is in the schedule
                                if time_of_day in day_schedule:
                                    new_location = day_schedule[time_of_day]
                                    
                                    # Update location if different using LoreSystem
                                    if new_location and new_location != current_location:
                                        # Update context for this NPC
                                        ctx.npc_id = npc_id
                                        
                                        result = await lore_system.propose_and_enact_change(
                                            ctx=ctx,
                                            entity_type="NPCStats",
                                            entity_identifier={"npc_id": npc_id},
                                            updates={"current_location": new_location},
                                            reason=f"Scheduled movement to {new_location} during {time_of_day} on {day_of_week}"
                                        )
                                        
                                        if result.get("status") == "committed":
                                            # Create memory of location change using canon
                                            memory_text = f"Moved to {new_location} during {time_of_day} on {day_of_week}."
                                            
                                            await canon.create_journal_entry(
                                                ctx, conn,
                                                entry_type="movement",
                                                entry_text=memory_text,
                                                tags=["movement", "location_change", f"npc_{npc_id}"],
                                                importance=0.2
                                            )
                                            
                                            results.append({
                                                "npc_id": npc_id,
                                                "npc_name": npc_name,
                                                "action": "moved",
                                                "old_location": current_location,
                                                "new_location": new_location
                                            })
                        except (json.JSONDecodeError, TypeError):
                            pass
            
            # Handle NPC interactions
            # Get NPCs at the same location
            location_npcs = {}
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, current_location
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """, self.user_id, self.conversation_id)
            
            for row in rows:
                location = row["current_location"]
                if location not in location_npcs:
                    location_npcs[location] = []
                
                location_npcs[location].append({
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"]
                })
            
            # Process interactions for NPCs at the same location
            for location, npcs in location_npcs.items():
                if len(npcs) >= 2:
                    # Randomly select some NPC pairs for interaction
                    for _ in range(min(3, len(npcs))):
                        npc1, npc2 = random.sample(npcs, 2)
                        
                        # Create memories of their interaction using canon
                        interaction_text1 = f"Interacted with {npc2['npc_name']} at {location} during {time_of_day}."
                        interaction_text2 = f"Interacted with {npc1['npc_name']} at {location} during {time_of_day}."
                        
                        # Create memory entries through canon
                        await canon.create_journal_entry(
                            ctx, conn,
                            entry_type="npc_interaction",
                            entry_text=interaction_text1,
                            tags=["interaction", "npc", f"npc_{npc1['npc_id']}", f"with_npc_{npc2['npc_id']}"],
                            importance=0.3,
                            metadata={
                                "entity_type": "npc",
                                "entity_id": npc1["npc_id"],
                                "target_npc_id": npc2["npc_id"],
                                "location": location,
                                "time_of_day": time_of_day
                            }
                        )
                        
                        await canon.create_journal_entry(
                            ctx, conn,
                            entry_type="npc_interaction",
                            entry_text=interaction_text2,
                            tags=["interaction", "npc", f"npc_{npc2['npc_id']}", f"with_npc_{npc1['npc_id']}"],
                            importance=0.3,
                            metadata={
                                "entity_type": "npc",
                                "entity_id": npc2["npc_id"],
                                "target_npc_id": npc1["npc_id"],
                                "location": location,
                                "time_of_day": time_of_day
                            }
                        )
                        
                        # Update relationship using canon system
                        await self._update_npc_relationship_canonical(ctx, conn, npc1["npc_id"], npc2["npc_id"])
                        
                        results.append({
                            "type": "interaction",
                            "npc1": npc1["npc_name"],
                            "npc2": npc2["npc_name"],
                            "location": location
                        })
            
            return {
                "year": year,
                "month": month,
                "day": day,
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "results": results
            }
            
        except Exception as e:  # This line should align with 'try:'
            logger.error(f"Error processing daily activities: {e}")
            return {"error": str(e), "results": []}

    async def _update_npc_relationship(self, ctx, conn, npc1_id: int, npc2_id: int) -> None:
        """
        Update the relationship between two NPCs using the canon system.
        
        Args:
            ctx: Context with governance info
            conn: Database connection
            npc1_id: ID of the first NPC
            npc2_id: ID of the second NPC
        """
        from lore.core import canon
        
        # Check if relationship exists
        row = await conn.fetchrow("""
            SELECT link_id, link_level
            FROM SocialLinks
            WHERE user_id=$1 AND conversation_id=$2
              AND ((entity1_type='npc' AND entity1_id=$3 AND entity2_type='npc' AND entity2_id=$4)
               OR  (entity1_type='npc' AND entity1_id=$4 AND entity2_type='npc' AND entity2_id=$3))
            LIMIT 1
        """, self.user_id, self.conversation_id, npc1_id, npc2_id)
        
        if row:
            # Update existing relationship
            link_id = row["link_id"]
            current_level = row["link_level"]
            
            # Small random change to relationship
            change = random.randint(-1, 2)
            new_level = max(0, min(100, current_level + change))
            
            if new_level != current_level:
                await canon.update_entity_with_governance(
                    ctx, conn, "SocialLinks", link_id,
                    {"link_level": new_level},
                    f"Daily interaction between NPCs adjusting relationship level",
                    significance=2
                )
        else:
            # Create new relationship through canon
            link_id = await canon.find_or_create_social_link(
                ctx, conn,
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                entity1_type="npc",
                entity1_id=npc1_id,
                entity2_type="npc",
                entity2_id=npc2_id,
                link_type="neutral",
                link_level=50
            )
            
            await canon.log_canonical_event(
                ctx, conn,
                f"New relationship established between NPC {npc1_id} and NPC {npc2_id}",
                tags=["relationship", "npc_relationship", "creation"],
                significance=4
            )

    async def _record_interaction_for_learning(
        self, 
        npc_id: int, 
        interaction_type: str, 
        player_input: str, 
        npc_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an interaction with the NPC learning adaptation system.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction
            player_input: Player's input
            npc_response: NPC's response
            context: Optional context information
        """
        try:
            # Get or create a learning system for this NPC
            learning_system = self.learning_manager.get_learning_system_for_npc(npc_id)
            await learning_system.initialize()
            
            # Determine player response characteristics based on the input
            # Enhanced version with more comprehensive analysis
            player_input_lower = player_input.lower()
            
            # Initialize variables
            compliance_level = 0
            emotional_response = "neutral"
            intensity_level = 0
            respect_level = 0
            fear_level = 0
            
            # ENHANCED: Structured emotion and response detection
            emotion_patterns = {
                # Compliance patterns
                "strong_compliance": [
                    "absolutely", "of course", "my pleasure", "happy to", "gladly",
                    "at your service", "as you command", "without question", "immediately",
                    "right away", "certainly", "with pleasure", "i'd be delighted"
                ],
                "compliance": [
                    "yes", "okay", "sure", "fine", "will do", "as you wish", "alright",
                    "i'll do it", "agreed", "very well", "i understand", "if you say so"
                ],
                "reluctant_compliance": [
                    "i guess", "if i must", "if i have to", "suppose so", "whatever",
                    "fine then", "sigh", "hesitantly", "reluctantly", "not like i have a choice"
                ],
                
                # Defiance patterns
                "strong_defiance": [
                    "absolutely not", "never", "no way", "forget it", "i refuse",
                    "not a chance", "hell no", "over my dead body", "i won't", "not happening"
                ],
                "defiance": [
                    "no", "won't", "refuse", "not going to", "can't make me", "i don't think so",
                    "nope", "not interested", "i decline", "not now", "i don't want to"
                ],
                "mild_defiance": [
                    "maybe later", "not right now", "i'd rather not", "do i have to",
                    "is this necessary", "can we do something else", "i don't feel like it"
                ],
                
                # Emotional responses
                "fear": [
                    "afraid", "scared", "terrified", "fear", "worried", "nervous", "anxious",
                    "frightened", "petrified", "dread", "panic", "horror", "trembling"
                ],
                "anger": [
                    "angry", "mad", "furious", "outraged", "annoyed", "irritated", "enraged",
                    "hostile", "resent", "hate", "despise", "contempt", "disgusted"
                ],
                "sadness": [
                    "sad", "upset", "unhappy", "miserable", "depressed", "heartbroken",
                    "devastated", "disappointed", "dejected", "hopeless", "grief", "sorrow"
                ],
                "joy": [
                    "happy", "excited", "delighted", "pleased", "glad", "joyful", "thrilled",
                    "ecstatic", "content", "satisfied", "elated", "jubilant", "overjoyed"
                ],
                "surprise": [
                    "surprised", "shocked", "astonished", "amazed", "stunned", "startled",
                    "bewildered", "confused", "taken aback", "unexpected", "wow", "oh"
                ],
                
                # Respect and submission
                "respect": [
                    "respect", "admire", "look up to", "honor", "esteem", "regard highly",
                    "value your", "appreciate your", "trust your", "your wisdom", "your guidance"
                ],
                "submission": [
                    "submit", "yield", "surrender", "obey", "comply", "follow", "serve",
                    "bow to", "defer to", "at your mercy", "under your control", "your command"
                ]
            }
            
            # Analyze for each emotional pattern
            detected_patterns = {}
            for category, patterns in emotion_patterns.items():
                detected_patterns[category] = 0
                for pattern in patterns:
                    if pattern in player_input_lower:
                        detected_patterns[category] += 1
            
            # Calculate strongest emotional response
            strongest_emotion = "neutral"
            strongest_value = 0
            emotional_categories = ["fear", "anger", "sadness", "joy", "surprise"]
            for emotion in emotional_categories:
                if detected_patterns[emotion] > strongest_value:
                    strongest_value = detected_patterns[emotion]
                    strongest_emotion = emotion
            
            # Only set emotional response if we detected something
            if strongest_value > 0:
                emotional_response = strongest_emotion
            
            # Calculate compliance level
            if detected_patterns["strong_compliance"] > 0:
                compliance_level = 8
            elif detected_patterns["compliance"] > 0:
                compliance_level = 5
            elif detected_patterns["reluctant_compliance"] > 0:
                compliance_level = 2
            elif detected_patterns["mild_defiance"] > 0:
                compliance_level = -3
            elif detected_patterns["defiance"] > 0:
                compliance_level = -5
            elif detected_patterns["strong_defiance"] > 0:
                compliance_level = -8
            
            # Calculate respect level
            respect_level = detected_patterns["respect"] * 2
            
            # Calculate submission level
            submission_level = detected_patterns["submission"] * 2
            
            # Calculate fear level
            fear_level = detected_patterns["fear"] * 2
            
            # Adjust compliance level based on fear (scared compliance is different)
            if compliance_level > 0 and fear_level > 0:
                # If complying out of fear, mark it differently
                compliance_type = "fearful_compliance"
            elif compliance_level > 0 and respect_level > 0:
                # If complying out of respect, mark it differently
                compliance_type = "respectful_compliance"
            elif compliance_level > 0 and submission_level > 0:
                # If complying out of submission, mark it differently
                compliance_type = "submissive_compliance"
            elif compliance_level > 0:
                compliance_type = "willing_compliance"
            elif compliance_level < 0:
                compliance_type = "defiance"
            else:
                compliance_type = "neutral"
            
            # Calculate intensity based on the strength of the emotional response
            intensity_level = max(
                detected_patterns["strong_compliance"] * 2,
                detected_patterns["strong_defiance"] * 2,
                detected_patterns["fear"] * 2,
                detected_patterns["anger"] * 2,
                detected_patterns["submission"] * 2
            )
            
            # Create a detailed analysis result
            analysis_result = {
                "compliance_level": compliance_level,
                "compliance_type": compliance_type,
                "emotional_response": emotional_response,
                "intensity_level": intensity_level,
                "respect_level": respect_level,
                "fear_level": fear_level,
                "submission_level": submission_level,
                "pattern_matches": detected_patterns
            }
            
            # Log detailed analysis for debugging
            logging.debug(f"Player input analysis for NPC {npc_id}: {analysis_result}")
            
            # Record the learning interaction with enhanced analysis
            await learning_system.record_player_interaction(
                interaction_type=interaction_type,
                player_input=player_input,
                npc_response=npc_response,
                compliance_level=compliance_level,
                emotional_response=emotional_response,
                context=context,
                analysis_result=analysis_result
            )
            
            # Log the learning interaction
            logging.info(f"Recorded learning interaction for NPC {npc_id}: {interaction_type}")
            
        except Exception as e:
            logging.error(f"Error recording interaction for learning: {e}", exc_info=True)

    # Add a method to explicitly trigger memory-based learning
    async def process_npc_learning_cycle(self, npc_id: int) -> Dict[str, Any]:
        """
        Trigger a learning cycle for an NPC to process memories and adapt behavior.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with learning results
        """
        try:
            # Get learning system for this NPC
            learning_system = self.learning_manager.get_learning_system_for_npc(npc_id)
            await learning_system.initialize()
            
            # Process memories for learning
            memory_result = await learning_system.process_recent_memories_for_learning()
            
            # Process relationship changes
            relationship_result = await learning_system.adapt_to_relationship_changes()
            
            return {
                "npc_id": npc_id,
                "memory_learning": memory_result,
                "relationship_adaptation": relationship_result,
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Error processing NPC learning cycle: {e}", exc_info=True)
            return {
                "npc_id": npc_id,
                "success": False,
                "error": str(e)
            }
