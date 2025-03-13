# npcs/npc_handler_agent.py

import logging
import json
import asyncio
import random
from typing import List, Dict, Any, Optional
import os
import asyncpg

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail
from pydantic import BaseModel, Field

# Import your existing modules
from logic.npc_creation import spawn_multiple_npcs_enhanced, create_and_refine_npc
from logic.activities_logic import get_all_activities, filter_activities_for_npc
from db.connection import get_db_connection

# Configuration
DB_DSN = os.getenv("DB_DSN")

# Models for input/output
class NPCContext(BaseModel):
    user_id: int
    conversation_id: int
    environment_desc: str = ""
    day_names: List[str] = Field(default_factory=list)

class NPCCreationInput(BaseModel):
    npc_name: str
    sex: str = "female"
    archetype_names: List[str] = Field(default_factory=list)
    physical_description: str = ""
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    introduced: bool = False
    dominance: int = 50
    cruelty: int = 50
    closeness: int = 50
    trust: int = 50
    respect: int = 50
    intensity: int = 50

class NPCUpdateInput(BaseModel):
    npc_id: int
    npc_name: Optional[str] = None
    physical_description: Optional[str] = None
    archetype_summary: Optional[str] = None
    archetype_extras_summary: Optional[str] = None
    dominance: Optional[int] = None
    cruelty: Optional[int] = None
    closeness: Optional[int] = None
    trust: Optional[int] = None
    respect: Optional[int] = None
    intensity: Optional[int] = None
    hobbies: Optional[List[str]] = None
    personality_traits: Optional[List[str]] = None
    likes: Optional[List[str]] = None
    dislikes: Optional[List[str]] = None
    memory: Optional[List[str]] = None
    current_location: Optional[str] = None
    introduced: Optional[bool] = None

class NPCScheduleDay(BaseModel):
    Morning: str = ""
    Afternoon: str = ""
    Evening: str = ""
    Night: str = ""

class NPCSchedule(BaseModel):
    schedule: Dict[str, NPCScheduleDay] = Field(default_factory=dict)

class NPCRelationship(BaseModel):
    npc1_id: int
    npc2_id: int
    relationship_type: str
    relationship_level: int
    dynamics: Dict[str, Any] = Field(default_factory=dict)

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

class NPCHandlerAgent:
    """Agent for handling NPC creation, updates, and interactions"""
    
    def __init__(self):
        self.npc_creator = Agent(
            name="NPCCreator",
            instructions="""
            You are a specialized agent for creating detailed NPCs for a roleplaying game with subtle femdom elements.
            
            Create NPCs with:
            - Consistent and coherent personalities
            - Realistic motivations and backgrounds
            - Subtle dominance traits hidden behind friendly facades
            - Detailed physical and personality descriptions
            - Appropriate archetypes that fit the game's themes
            
            The NPCs should feel like real individuals with complex personalities and hidden agendas,
            while maintaining a balance between mundane everyday characteristics and subtle control dynamics.
            """,
            output_type=NPCCreationInput,
            tools=[
                function_tool(self.suggest_archetypes),
                function_tool(self.get_environment_details)
            ]
        )
        
        self.schedule_creator = Agent(
            name="ScheduleCreator",
            instructions="""
            You create detailed, realistic daily schedules for NPCs in a roleplaying game.
            
            Each schedule should:
            - Fit the NPC's personality, interests, and social status
            - Follow a realistic pattern throughout the week
            - Include variations for different days
            - Place the NPC in appropriate locations at appropriate times
            - Include opportunities for player interactions
            
            The schedules should feel natural and mundane while creating opportunities for
            subtle power dynamics to emerge during player encounters.
            """,
            output_type=NPCSchedule,
            tools=[
                function_tool(self.get_locations),
                function_tool(self.get_npc_details)
            ]
        )
        
        self.interaction_handler = Agent(
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
        
        self.relationship_manager = Agent(
            name="RelationshipManager",
            instructions="""
            You manage and evolve NPC relationships within a roleplaying game with subtle femdom elements.
            
            Create and update relationships that:
            - Feel natural and evolve organically
            - Reflect each NPC's personality and stats
            - Include complex dynamics and potential conflicts
            - Incorporate subtle power imbalances when appropriate
            - Create interesting narrative possibilities
            
            The relationships should provide depth to the game world while maintaining
            the balance between mundane social dynamics and subtle control elements.
            """,
            output_type=NPCRelationship,
            tools=[
                function_tool(self.get_npc_details),
                function_tool(self.get_relationship_details)
            ]
        )
        
        # Main coordinating agent
        self.agent = Agent(
            name="NPCManager",
            instructions="""
            You coordinate NPC management for a roleplaying game with subtle femdom elements.
            
            Your responsibilities include:
            - Creating new NPCs with detailed characteristics
            - Generating realistic schedules for NPCs
            - Handling NPC interactions with the player
            - Managing relationships between NPCs
            - Evolving NPCs based on game events
            
            Maintain a balance between realistic, mundane NPCs and subtle power dynamics,
            ensuring NPCs feel like unique individuals with hidden depths and agendas.
            """,
            tools=[
                function_tool(self.create_npc),
                function_tool(self.update_npc),
                function_tool(self.create_npc_schedule),
                function_tool(self.handle_npc_interaction),
                function_tool(self.manage_npc_relationship),
                function_tool(self.spawn_multiple_npcs),
                function_tool(self.get_nearby_npcs),
                function_tool(self.process_daily_npc_activities)
            ]
        )
    
    async def suggest_archetypes(self, ctx):
        """
        Suggest appropriate archetypes for NPCs.
        
        Returns:
            List of archetype objects with id and name
        """
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            rows = await conn.fetch("""
                SELECT id, name
                FROM Archetypes
                ORDER BY id
            """)
            
            archetypes = []
            for row in rows:
                archetypes.append({
                    "id": row["id"],
                    "name": row["name"]
                })
            
            return archetypes
        finally:
            await conn.close()
    
    async def get_environment_details(self, ctx):
        """
        Get details about the game environment.
        
        Returns:
            Dictionary with environment details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            # Get environment description
            row = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
                LIMIT 1
            """, user_id, conversation_id)
            
            environment_desc = row["value"] if row else "No environment description available."
            
            # Get setting name
            row = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentSetting'
                LIMIT 1
            """, user_id, conversation_id)
            
            setting_name = row["value"] if row else "Unknown Setting"
            
            # Get locations
            rows = await conn.fetch("""
                SELECT location_name, description
                FROM Locations
                WHERE user_id=$1 AND conversation_id=$2
                LIMIT 10
            """, user_id, conversation_id)
            
            locations = []
            for row in rows:
                locations.append({
                    "name": row["location_name"],
                    "description": row["description"]
                })
            
            return {
                "environment_desc": environment_desc,
                "setting_name": setting_name,
                "locations": locations
            }
        finally:
            await conn.close()
    
    async def get_locations(self, ctx):
        """
        Get all locations in the game world.
        
        Returns:
            List of location objects
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            rows = await conn.fetch("""
                SELECT id, location_name, description, open_hours
                FROM Locations
                WHERE user_id=$1 AND conversation_id=$2
                ORDER BY id
            """, user_id, conversation_id)
            
            locations = []
            for row in rows:
                location = {
                    "id": row["id"],
                    "location_name": row["location_name"],
                    "description": row["description"]
                }
                
                # Parse open_hours if available
                if row["open_hours"]:
                    try:
                        if isinstance(row["open_hours"], str):
                            location["open_hours"] = json.loads(row["open_hours"])
                        else:
                            location["open_hours"] = row["open_hours"]
                    except (json.JSONDecodeError, TypeError):
                        location["open_hours"] = []
                else:
                    location["open_hours"] = []
                
                locations.append(location)
            
            return locations
        finally:
            await conn.close()
    
    async def get_npc_details(self, ctx, npc_id=None, npc_name=None):
        """
        Get details about a specific NPC.
        
        Args:
            npc_id: ID of the NPC to get details for (optional)
            npc_name: Name of the NPC to get details for (optional)
            
        Returns:
            Dictionary with NPC details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            query = """
                SELECT npc_id, npc_name, introduced, archetypes, archetype_summary, 
                       archetype_extras_summary, physical_description, relationships,
                       dominance, cruelty, closeness, trust, respect, intensity,
                       hobbies, personality_traits, likes, dislikes, affiliations,
                       schedule, current_location, sex, age
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """
            
            params = [user_id, conversation_id]
            
            if npc_id is not None:
                query += " AND npc_id=$3"
                params.append(npc_id)
            elif npc_name is not None:
                query += " AND LOWER(npc_name)=LOWER($3)"
                params.append(npc_name)
            else:
                return {"error": "No NPC ID or name provided"}
            
            query += " LIMIT 1"
            
            row = await conn.fetchrow(query, *params)
            
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
        finally:
            await conn.close()
    
    async def get_npc_memory(self, ctx, npc_id):
        """
        Get memories associated with an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of memory objects
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            # First check if the NPC has memories in the NPCStats table
            row = await conn.fetchrow("""
                SELECT memory
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                LIMIT 1
            """, user_id, conversation_id, npc_id)
            
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
            """, npc_id, user_id, conversation_id)
            
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
        finally:
            await conn.close()
    
    async def get_relationship_details(self, ctx, entity1_type, entity1_id, entity2_type, entity2_id):
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
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
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
                """, user_id, conversation_id, e1t, e1i, e2t, e2i)
                
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
        finally:
            await conn.close()
    
    async def create_npc(self, ctx, archetype_names=None, physical_desc=None, starting_traits=None):
        """
        Create a new NPC with detailed characteristics.
        
        Args:
            archetype_names: List of archetype names to use (optional)
            physical_desc: Physical description of the NPC (optional)
            starting_traits: Initial personality traits (optional)
            
        Returns:
            Dictionary with the created NPC details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get environment details for context
        env_details = await self.get_environment_details(ctx)
        
        # Create prompt for the NPC creator
        archetypes_str = ", ".join(archetype_names) if archetype_names else "to be determined"
        physical_desc_str = physical_desc if physical_desc else "to be determined"
        traits_str = ", ".join(starting_traits) if starting_traits else "to be determined"
        
        prompt = f"""
        Create a detailed NPC for a roleplaying game set in:
        {env_details['setting_name']}: {env_details['environment_desc']}
        
        The NPC should have:
        - Suggested archetypes: {archetypes_str}
        - Physical description: {physical_desc_str}
        - Suggested personality traits: {traits_str}
        
        Create a complete, coherent NPC with appropriate:
        - Name (if not already provided)
        - Sex (default to female)
        - Physical description
        - Personality traits
        - Likes and dislikes
        - Hobbies and interests
        - Affiliations and connections
        - Stats (dominance, cruelty, etc.)
        
        The NPC should feel like a real person with complex motivations,
        while incorporating subtle elements of control and influence.
        """
        
        # Run the NPC creator
        result = await Runner.run(
            self.npc_creator,
            prompt,
            context=ctx.context
        )
        
        npc_data = result.final_output
        
        # Create the NPC in the database
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            # Extract archetype IDs if names were provided
            archetype_ids = []
            if archetype_names:
                for name in archetype_names:
                    row = await conn.fetchrow("""
                        SELECT id FROM Archetypes WHERE name=$1 LIMIT 1
                    """, name)
                    if row:
                        archetype_ids.append(row["id"])
            
            # Build archetypes JSON
            archetypes_json = []
            for arch_id in archetype_ids:
                row = await conn.fetchrow("SELECT name FROM Archetypes WHERE id=$1", arch_id)
                if row:
                    archetypes_json.append({"id": arch_id, "name": row["name"]})
            
            # Insert the NPC
            row = await conn.fetchrow("""
                INSERT INTO NPCStats (
                    user_id, conversation_id, npc_name, introduced, archetypes,
                    dominance, cruelty, closeness, trust, respect, intensity,
                    hobbies, personality_traits, likes, dislikes, affiliations,
                    physical_description, sex
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
                RETURNING npc_id
            """, 
            user_id, conversation_id, npc_data.npc_name, npc_data.introduced,
            json.dumps(archetypes_json), 
            npc_data.dominance, npc_data.cruelty, npc_data.closeness,
            npc_data.trust, npc_data.respect, npc_data.intensity,
            json.dumps(npc_data.hobbies), json.dumps(npc_data.personality_traits),
            json.dumps(npc_data.likes), json.dumps(npc_data.dislikes),
            json.dumps(npc_data.affiliations), npc_data.physical_description,
            npc_data.sex)
            
            npc_id = row["npc_id"]
            
            # Call the existing create_and_refine_npc function for additional refinements
            from logic.npc_creation import create_and_refine_npc
            await create_and_refine_npc(
                user_id,
                conversation_id,
                npc_id,
                npc_data.physical_description,
                env_details["environment_desc"]
            )
            
            # Return the created NPC
            return await self.get_npc_details(ctx, npc_id=npc_id)
        finally:
            await conn.close()
    
    async def update_npc(self, ctx, update_data):
        """
        Update an existing NPC.
        
        Args:
            update_data: NPCUpdateInput object
            
        Returns:
            Dictionary with the updated NPC details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        npc_id = update_data.npc_id
        
        # Validate that the NPC exists
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            row = await conn.fetchrow("""
                SELECT npc_id FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                LIMIT 1
            """, user_id, conversation_id, npc_id)
            
            if not row:
                return {"error": f"NPC with ID {npc_id} not found"}
            
            # Build update query
            update_parts = []
            params = []
            param_index = 1
            
            fields = [
                ("npc_name", update_data.npc_name),
                ("physical_description", update_data.physical_description),
                ("archetype_summary", update_data.archetype_summary),
                ("archetype_extras_summary", update_data.archetype_extras_summary),
                ("dominance", update_data.dominance),
                ("cruelty", update_data.cruelty),
                ("closeness", update_data.closeness),
                ("trust", update_data.trust),
                ("respect", update_data.respect),
                ("intensity", update_data.intensity),
                ("current_location", update_data.current_location),
                ("introduced", update_data.introduced)
            ]
            
            for field_name, value in fields:
                if value is not None:
                    update_parts.append(f"{field_name} = ${param_index}")
                    params.append(value)
                    param_index += 1
            
            # Handle JSON fields
            if update_data.hobbies is not None:
                update_parts.append(f"hobbies = ${param_index}")
                params.append(json.dumps(update_data.hobbies))
                param_index += 1
            
            if update_data.personality_traits is not None:
                update_parts.append(f"personality_traits = ${param_index}")
                params.append(json.dumps(update_data.personality_traits))
                param_index += 1
            
            if update_data.likes is not None:
                update_parts.append(f"likes = ${param_index}")
                params.append(json.dumps(update_data.likes))
                param_index += 1
            
            if update_data.dislikes is not None:
                update_parts.append(f"dislikes = ${param_index}")
                params.append(json.dumps(update_data.dislikes))
                param_index += 1
            
            if update_data.memory is not None:
                # For memory, we append to the existing memory
                row = await conn.fetchrow("""
                    SELECT memory FROM NPCStats
                    WHERE npc_id=$1
                """, npc_id)
                
                existing_memory = []
                if row and row["memory"]:
                    try:
                        if isinstance(row["memory"], str):
                            existing_memory = json.loads(row["memory"])
                        else:
                            existing_memory = row["memory"]
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                new_memory = existing_memory + update_data.memory
                update_parts.append(f"memory = ${param_index}")
                params.append(json.dumps(new_memory))
                param_index += 1
            
            # Execute the update if there are fields to update
            if update_parts:
                params.extend([user_id, conversation_id, npc_id])
                
                query = f"""
                    UPDATE NPCStats
                    SET {", ".join(update_parts)}
                    WHERE user_id=${param_index} AND conversation_id=${param_index + 1} AND npc_id=${param_index + 2}
                """
                
                await conn.execute(query, *params)
            
            # Return the updated NPC
            return await self.get_npc_details(ctx, npc_id=npc_id)
        finally:
            await conn.close()
    
    async def create_npc_schedule(self, ctx, npc_id, day_names=None):
        """
        Create a detailed schedule for an NPC.
        
        Args:
            npc_id: ID of the NPC
            day_names: List of day names (e.g., ["Monday", "Tuesday", ...])
            
        Returns:
            Dictionary with the NPC's schedule
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get NPC details
        npc_details = await self.get_npc_details(ctx, npc_id=npc_id)
        
        # Get day names if not provided
        if not day_names:
            conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                row = await conn.fetchrow("""
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                    LIMIT 1
                """, user_id, conversation_id)
                
                if row:
                    try:
                        calendar_data = json.loads(row["value"])
                        day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                    except (json.JSONDecodeError, TypeError):
                        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                else:
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            finally:
                await conn.close()
        
        # Create prompt for the schedule creator
        locations = await self.get_locations(ctx)
        locations_str = "\n".join([f"- {loc['location_name']}: {loc['description'][:100]}..." for loc in locations[:10]])
        
        prompt = f"""
        Create a detailed weekly schedule for:
        {npc_details['npc_name']}, a {npc_details['sex']} with the following characteristics:
        
        Physical description: {npc_details['physical_description']}
        Personality traits: {json.dumps(npc_details['personality_traits'])}
        Hobbies: {json.dumps(npc_details['hobbies'])}
        Likes: {json.dumps(npc_details['likes'])}
        Dislikes: {json.dumps(npc_details['dislikes'])}
        Affiliations: {json.dumps(npc_details['affiliations'])}
        
        Stats:
        - Dominance: {npc_details['dominance']}
        - Cruelty: {npc_details['cruelty']}
        - Closeness: {npc_details['closeness']}
        - Trust: {npc_details['trust']}
        - Respect: {npc_details['respect']}
        - Intensity: {npc_details['intensity']}
        
        Available locations:
        {locations_str}
        
        Create a schedule for each day of the week ({', '.join(day_names)})
        with activities for Morning, Afternoon, Evening, and Night time periods.
        
        The schedule should:
        - Fit the NPC's personality and interests
        - Use available locations appropriately
        - Include variations for different days
        - Create opportunities for player interactions
        - Include subtle hints of the NPC's dominance style when appropriate
        """
        
        # Run the schedule creator
        result = await Runner.run(
            self.schedule_creator,
            prompt,
            context=ctx.context
        )
        
        schedule = result.final_output.schedule
        
        # Save the schedule to the database
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            await conn.execute("""
                UPDATE NPCStats
                SET schedule = $1
                WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
            """, json.dumps(schedule), user_id, conversation_id, npc_id)
            
            return {"npc_id": npc_id, "schedule": schedule}
        finally:
            await conn.close()
    
    async def handle_npc_interaction(self, ctx, npc_id, interaction_type, player_input, context=None):
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
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get NPC details
        npc_details = await self.get_npc_details(ctx, npc_id=npc_id)
        
        # Get NPC memories
        memories = await self.get_npc_memory(ctx, npc_id)
        
        # Get player relationship
        relationship = await self.get_relationship_details(ctx, "npc", npc_id, "player", 0)
        
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
            self.interaction_handler,
            prompt,
            context=ctx.context
        )
        
        response = result.final_output
        
        # Store the interaction in memory if it's significant
        if response.memory_created:
            conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                memory_text = f"Interaction with player: {player_input} - Response: {response.response}"
                
                # Add memory to NPCStats
                row = await conn.fetchrow("""
                    SELECT memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    LIMIT 1
                """, user_id, conversation_id, npc_id)
                
                existing_memory = []
                if row and row["memory"]:
                    try:
                        if isinstance(row["memory"], str):
                            existing_memory = json.loads(row["memory"])
                        else:
                            existing_memory = row["memory"]
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                existing_memory.append(memory_text)
                
                await conn.execute("""
                    UPDATE NPCStats
                    SET memory = $1
                    WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                """, json.dumps(existing_memory), user_id, conversation_id, npc_id)
                
                # Also add to unified_memories for better compatibility
                await conn.execute("""
                    INSERT INTO unified_memories (
                        entity_type, entity_id, user_id, conversation_id,
                        memory_text, memory_type, significance, emotional_intensity
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, "npc", npc_id, user_id, conversation_id, memory_text, "interaction", 3, 30)
            finally:
                await conn.close()
        
        # Apply stat changes if any
        if response.stat_changes:
            conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                for stat, change in response.stat_changes.items():
                    if stat in ["dominance", "cruelty", "closeness", "trust", "respect", "intensity"]:
                        # Get current value
                        row = await conn.fetchrow(f"""
                            SELECT {stat}
                            FROM NPCStats
                            WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                            LIMIT 1
                        """, user_id, conversation_id, npc_id)
                        
                        if row:
                            current_value = row[stat]
                            new_value = max(0, min(100, current_value + change))
                            
                            await conn.execute(f"""
                                UPDATE NPCStats
                                SET {stat} = $1
                                WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                            """, new_value, user_id, conversation_id, npc_id)
            finally:
                await conn.close()
        
        return response.dict()
    
    async def manage_npc_relationship(self, ctx, npc1_id, npc2_id=None, entity2_type="player", entity2_id=0, relationship_type=None, level_change=0):
        """
        Manage or update a relationship between NPCs or between an NPC and the player.
        
        Args:
            npc1_id: ID of the first NPC
            npc2_id: ID of the second NPC (optional, defaults to None)
            entity2_type: Type of the second entity (defaults to "player")
            entity2_id: ID of the second entity (defaults to 0 for player)
            relationship_type: Type of relationship (optional)
            level_change: Change in relationship level (optional)
            
        Returns:
            Dictionary with the updated relationship details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Determine entity types and IDs
        entity1_type = "npc"
        entity1_id = npc1_id
        
        if npc2_id is not None:
            entity2_type = "npc"
            entity2_id = npc2_id
        
        # Get existing relationship if any
        existing_relationship = await self.get_relationship_details(ctx, entity1_type, entity1_id, entity2_type, entity2_id)
        
        # Get details for both entities
        npc1_details = await self.get_npc_details(ctx, npc_id=npc1_id)
        
        if entity2_type == "npc" and entity2_id:
            npc2_details = await self.get_npc_details(ctx, npc_id=entity2_id)
            entity2_name = npc2_details["npc_name"]
        else:
            entity2_name = "Player (Chase)"
        
        # Create prompt for the relationship manager
        existing_str = json.dumps(existing_relationship)
        
        prompt = f"""
        Manage the relationship between:
        
        Entity 1: {npc1_details['npc_name']} (NPC, ID: {npc1_id})
        Entity 2: {entity2_name} ({entity2_type}, ID: {entity2_id})
        
        Existing relationship: {existing_str}
        
        Suggested relationship type: {relationship_type or "to be determined"}
        Suggested level change: {level_change}
        
        Evaluate and update this relationship considering:
        - Entity personalities and characteristics
        - Existing relationship history
        - Appropriate dynamics for their interaction
        - Potential for interesting narrative development
        
        Create or update the relationship with appropriate:
        - Relationship type (e.g., friend, rival, mentor, etc.)
        - Relationship level (0-100)
        - Dynamics and potential crossroads
        - Group interaction context if relevant
        """
        
        # Run the relationship manager
        result = await Runner.run(
            self.relationship_manager,
            prompt,
            context=ctx.context
        )
        
        relationship_data = result.final_output
        
        # Update or create the relationship in the database
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            # Check if relationship exists
            row = await conn.fetchrow("""
                SELECT link_id
                FROM SocialLinks
                WHERE user_id=$1 AND conversation_id=$2
                  AND ((entity1_type=$3 AND entity1_id=$4 AND entity2_type=$5 AND entity2_id=$6)
                   OR  (entity1_type=$5 AND entity1_id=$6 AND entity2_type=$3 AND entity2_id=$4))
                LIMIT 1
            """, user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
            
            if row:
                # Update existing relationship
                link_id = row["link_id"]
                
                await conn.execute("""
                    UPDATE SocialLinks
                    SET link_type=$1, link_level=$2, dynamics=$3
                    WHERE link_id=$4
                """, 
                relationship_data.relationship_type,
                relationship_data.relationship_level,
                json.dumps(relationship_data.dynamics),
                link_id)
            else:
                # Create new relationship
                row = await conn.fetchrow("""
                    INSERT INTO SocialLinks (
                        user_id, conversation_id, entity1_type, entity1_id,
                        entity2_type, entity2_id, link_type, link_level, dynamics
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING link_id
                """,
                user_id, conversation_id, entity1_type, entity1_id,
                entity2_type, entity2_id, relationship_data.relationship_type,
                relationship_data.relationship_level, json.dumps(relationship_data.dynamics))
                
                link_id = row["link_id"]
            
            # Get updated relationship
            return await self.get_relationship_details(ctx, entity1_type, entity1_id, entity2_type, entity2_id)
        finally:
            await conn.close()
    
    async def spawn_multiple_npcs(self, ctx, count=5):
        """
        Spawn multiple NPCs for the game world.
        
        Args:
            count: Number of NPCs to spawn
            
        Returns:
            List of spawned NPC IDs
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get environment description
        env_details = await self.get_environment_details(ctx)
        
        # Get day names
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            row = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                LIMIT 1
            """, user_id, conversation_id)
            
            if row:
                try:
                    calendar_data = json.loads(row["value"])
                    day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                except (json.JSONDecodeError, TypeError):
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            else:
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        finally:
            await conn.close()
        
        # Use existing spawn_multiple_npcs_enhanced function
        npc_ids = await spawn_multiple_npcs_enhanced(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=env_details["environment_desc"],
            day_names=day_names,
            count=count
        )
        
        return npc_ids
    
    async def get_nearby_npcs(self, ctx, location=None):
        """
        Get NPCs that are at a specific location.
        
        Args:
            location: Location to filter by (optional)
            
        Returns:
            List of nearby NPCs
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            if location:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 
                    AND current_location=$3
                    ORDER BY introduced DESC
                    LIMIT 5
                """, user_id, conversation_id, location)
            else:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY introduced DESC
                    LIMIT 5
                """, user_id, conversation_id)
            
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
        finally:
            await conn.close()
    
    async def process_daily_npc_activities(self, ctx):
        """
        Process daily activities for all NPCs.
        Update locations, create memories, and handle NPC interactions.
        
        Returns:
            Dictionary with processing results
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get current time
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            year, month, day, time_of_day = 1, 1, 1, "Morning"
            
            for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
                row = await conn.fetchrow("""
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                """, user_id, conversation_id, key)
                
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
            row = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                LIMIT 1
            """, user_id, conversation_id)
            
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
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, schedule, current_location
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """, user_id, conversation_id)
            
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
                                
                                # Update location if different
                                if new_location and new_location != current_location:
                                    await conn.execute("""
                                        UPDATE NPCStats
                                        SET current_location=$1
                                        WHERE npc_id=$2
                                    """, new_location, npc_id)
                                    
                                    # Create memory of location change
                                    memory_text = f"Moved to {new_location} during {time_of_day} on {day_of_week}."
                                    
                                    # Add to unified_memories
                                    await conn.execute("""
                                        INSERT INTO unified_memories (
                                            entity_type, entity_id, user_id, conversation_id,
                                            memory_text, memory_type, significance, emotional_intensity
                                        )
                                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                                    """, "npc", npc_id, user_id, conversation_id, memory_text, "movement", 2, 10)
                                    
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
            """, user_id, conversation_id)
            
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
                        
                        # Create a memory of their interaction
                        interaction_text = f"Interacted with {npc2['npc_name']} at {location} during {time_of_day}."
                        
                        # Add to unified_memories for both NPCs
                        await conn.execute("""
                            INSERT INTO unified_memories (
                                entity_type, entity_id, user_id, conversation_id,
                                memory_text, memory_type, significance, emotional_intensity
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, "npc", npc1["npc_id"], user_id, conversation_id, interaction_text, "interaction", 2, 20)
                        
                        interaction_text2 = f"Interacted with {npc1['npc_name']} at {location} during {time_of_day}."
                        
                        await conn.execute("""
                            INSERT INTO unified_memories (
                                entity_type, entity_id, user_id, conversation_id,
                                memory_text, memory_type, significance, emotional_intensity
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, "npc", npc2["npc_id"], user_id, conversation_id, interaction_text2, "interaction", 2, 20)
                        
                        # Update or create relationship if needed
                        await self.manage_npc_relationship(ctx, npc1["npc_id"], npc2["npc_id"])
                        
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
        finally:
            await conn.close()
