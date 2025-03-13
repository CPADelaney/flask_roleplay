# npcs/new_npc_creation.py

"""
Unified NPC creation functionality.
Refactored from npc_creation_agent.py and parts of npc_handler_agent.py.
"""

import logging
import json
import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import os
import asyncpg
from datetime import datetime

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail, RunContextWrapper
from db.connection import get_db_connection
from memory.wrapper import MemorySystem

# Import existing NPC creation utilities
from logic.npc_creation import (
    create_and_refine_npc, 
    spawn_multiple_npcs_enhanced,
    gpt_generate_physical_description,
    gpt_generate_schedule,
    gpt_generate_memories,
    gpt_generate_affiliations,
    gpt_generate_relationship_specific_memories,
    store_npc_memories,
    propagate_shared_memories,
    integrate_femdom_elements
)
from logic.calendar import load_calendar_names

logger = logging.getLogger(__name__)

# Configuration
DB_DSN = os.getenv("DB_DSN")

# Models for input/output
class NPCCreationContext(BaseModel):
    user_id: int
    conversation_id: int
    db_dsn: str = DB_DSN

class NPCPersonalityData(BaseModel):
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)

class NPCStatsData(BaseModel):
    dominance: int = 50
    cruelty: int = 30
    closeness: int = 50
    trust: int = 0
    respect: int = 0
    intensity: int = 40

class NPCArchetypeData(BaseModel):
    archetype_names: List[str] = Field(default_factory=list)
    archetype_summary: str = ""
    archetype_extras_summary: str = ""

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

class NPCCreationResult(BaseModel):
    npc_id: int
    npc_name: str
    physical_description: str
    personality: NPCPersonalityData
    stats: NPCStatsData
    archetypes: NPCArchetypeData
    schedule: Dict[str, Any] = Field(default_factory=dict)
    memories: List[str] = Field(default_factory=list)
    current_location: str = ""

class EnvironmentGuardrailOutput(BaseModel):
    is_valid: bool = True
    reasoning: str = ""

class NPCCreationHandler:
    """
    Unified handler for NPC creation and management.
    Combines functionality from NPCCreationAgent and parts of NPCHandlerAgent.
    """
    
    def __init__(self):
        # Initialize input validation guardrail
        @InputGuardrail
        async def environment_guardrail(ctx, agent, input_str):
            """Validate that the environment description is appropriate for NPC creation"""
            try:
                # Check if the input has minimum required information
                if len(input_str) < 50:
                    return GuardrailFunctionOutput(
                        output_info=EnvironmentGuardrailOutput(
                            is_valid=False, 
                            reasoning="Environment description is too short for effective NPC creation"
                        ),
                        tripwire_triggered=True
                    )
                
                # Check for required elements
                required_elements = ["setting", "environment", "world", "location"]
                if not any(element in input_str.lower() for element in required_elements):
                    return GuardrailFunctionOutput(
                        output_info=EnvironmentGuardrailOutput(
                            is_valid=False, 
                            reasoning="Environment description lacks essential setting information"
                        ),
                        tripwire_triggered=True
                    )
                
                return GuardrailFunctionOutput(
                    output_info=EnvironmentGuardrailOutput(
                        is_valid=True, 
                        reasoning="Environment description is valid"
                    ),
                    tripwire_triggered=False
                )
            except Exception as e:
                logging.error(f"Error in environment guardrail: {e}")
                return GuardrailFunctionOutput(
                    output_info=EnvironmentGuardrailOutput(
                        is_valid=False, 
                        reasoning=f"Error validating environment: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
                
        # Personality designer agent
        self.personality_designer = Agent(
            name="NPCPersonalityDesigner",
            instructions="""
            You are a specialist in designing unique and consistent NPC personalities.
            
            Create personalities with:
            - 3-5 distinct personality traits that form a coherent character
            - 3-5 likes that align with the personality
            - 3-5 dislikes that create interesting tension
            - 2-4 hobbies or interests that make the character feel three-dimensional
            
            The personalities should feel like real individuals with subtle psychological
            depth. Include traits that suggest hidden layers, secret motivations, or
            potential for character growth.
            
            For femdom-themed worlds, incorporate subtle traits related to control,
            authority, or psychological dominance without being explicit or overt.
            These should be woven naturally into the personality.
            """,
            output_type=NPCPersonalityData
        )
        
        # Stats calibrator agent
        self.stats_calibrator = Agent(
            name="NPCStatsCalibrator",
            instructions="""
            You calibrate NPC stats to match their personality and archetypes.
            
            Determine appropriate values (0-100) for:
            - dominance: How naturally controlling/authoritative the NPC is
            - cruelty: How willing the NPC is to cause discomfort/distress
            - closeness: How emotionally available/connected the NPC is
            - trust: Trust toward the player (-100 to 100)
            - respect: Respect toward the player (-100 to 100)
            - intensity: Overall emotional/psychological intensity
            
            The stats should align coherently with the personality traits and archetypes.
            For femdom-themed NPCs, calibrate dominance higher (50-90) while ensuring
            other stats create a balanced, nuanced character. Cruelty can vary widely
            depending on personality (10-80).
            """,
            output_type=NPCStatsData
        )
        
        # Archetype synthesizer agent
        self.archetype_synthesizer = Agent(
            name="NPCArchetypeSynthesizer",
            instructions="""
            You synthesize multiple archetypes into a coherent character concept.
            
            Given multiple archetypes, create:
            - A cohesive archetype summary that blends the archetypes
            - An extras summary explaining how the archetype fusion affects the character
            
            The synthesis should feel natural rather than forced, identifying common
            themes and resolving contradictions between archetypes. Focus on how the
            archetypes interact to create a unique character foundation.
            
            For femdom-themed archetypes, emphasize subtle dominance dynamics while
            maintaining psychological realism and depth.
            """,
            output_type=NPCArchetypeData,
            tools=[function_tool(self.get_available_archetypes)]
        )
        
        # Main NPC creator agent 
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
        
        # Schedule creator agent
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
            tools=[
                function_tool(self.get_locations),
                function_tool(self.get_npc_details)
            ]
        )
        
        # Main agent with input guardrail
        self.agent = Agent(
            name="NPCCreator",
            instructions="""
            You are an expert NPC creator for immersive, psychologically realistic role-playing games.
            
            Create detailed NPCs with:
            - Unique names and physical descriptions
            - Consistent personalities and motivations
            - Appropriate stat calibrations
            - Coherent archetype synthesis
            - Detailed schedules and memories
            - Subtle elements of control and influence where appropriate
            
            The NPCs should feel like real people with histories, quirks, and hidden depths.
            Focus on psychological realism and subtle complexity rather than explicit themes.
            
            For femdom-themed worlds, incorporate subtle dominance dynamics
            into the character design without being heavy-handed or explicit.
            These elements should be woven naturally into the character's psychology.
            """,
            tools=[
                function_tool(self.generate_npc_name),
                function_tool(self.generate_physical_description),
                function_tool(self.design_personality),
                function_tool(self.calibrate_stats),
                function_tool(self.synthesize_archetypes),
                function_tool(self.generate_schedule),
                function_tool(self.generate_memories),
                function_tool(self.create_npc_in_database),
                function_tool(self.get_environment_details),
                function_tool(self.get_day_names)
            ],
            input_guardrails=[environment_guardrail]
        )
    
    async def get_available_archetypes(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Get available archetypes from the database.
        
        Returns:
            List of archetype data
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, baseline_stats, progression_rules, 
                       setting_examples, unique_traits
                FROM Archetypes
                ORDER BY name
            """)
            
            archetypes = []
            for row in cursor.fetchall():
                archetype = {
                    "id": row[0],
                    "name": row[1]
                }
                
                # Add detailed information if available
                if row[2]:  # baseline_stats
                    try:
                        if isinstance(row[2], str):
                            archetype["baseline_stats"] = json.loads(row[2])
                        else:
                            archetype["baseline_stats"] = row[2]
                    except:
                        pass
                
                if row[5]:  # unique_traits
                    try:
                        if isinstance(row[5], str):
                            archetype["unique_traits"] = json.loads(row[5])
                        else:
                            archetype["unique_traits"] = row[5]
                    except:
                        pass
                
                archetypes.append(archetype)
            
            conn.close()
            return archetypes
        except Exception as e:
            logging.error(f"Error getting archetypes: {e}")
            return []
    
    async def suggest_archetypes(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Suggest appropriate archetypes for NPCs.
        
        Returns:
            List of archetype objects with id and name
        """
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
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
            
            await conn.close()
            return archetypes
        except Exception as e:
            logging.error(f"Error suggesting archetypes: {e}")
            return []
    
    async def get_environment_details(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get details about the game environment.
        
        Returns:
            Dictionary with environment details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get environment description
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
            """, (user_id, conversation_id))
            
            row = cursor.fetchone()
            environment_desc = row[0] if row else "No environment description available"
            
            # Get current setting
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentSetting'
            """, (user_id, conversation_id))
            
            row = cursor.fetchone()
            setting_name = row[0] if row else "Unknown Setting"
            
            # Get locations
            cursor.execute("""
                SELECT location_name, description FROM Locations
                WHERE user_id=%s AND conversation_id=%s
                LIMIT 10
            """, (user_id, conversation_id))
            
            locations = []
            for row in cursor.fetchall():
                locations.append({
                    "name": row[0],
                    "description": row[1]
                })
            
            conn.close()
            
            return {
                "environment_desc": environment_desc,
                "setting_name": setting_name,
                "locations": locations
            }
        except Exception as e:
            logging.error(f"Error getting environment details: {e}")
            return {
                "environment_desc": "Error retrieving environment",
                "setting_name": "Unknown",
                "locations": []
            }
    
    async def get_locations(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Get all locations in the game world.
        
        Returns:
            List of location objects
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, location_name, description, open_hours
                FROM Locations
                WHERE user_id=%s AND conversation_id=%s
                ORDER BY id
            """, (user_id, conversation_id))
            
            locations = []
            for row in cursor.fetchall():
                location = {
                    "id": row[0],
                    "location_name": row[1],
                    "description": row[2]
                }
                
                # Parse open_hours if available
                if row[3]:
                    try:
                        if isinstance(row[3], str):
                            location["open_hours"] = json.loads(row[3])
                        else:
                            location["open_hours"] = row[3]
                    except:
                        location["open_hours"] = []
                else:
                    location["open_hours"] = []
                
                locations.append(location)
            
            conn.close()
            return locations
        except Exception as e:
            logging.error(f"Error getting locations: {e}")
            return []
    
    async def get_day_names(self, ctx: RunContextWrapper) -> List[str]:
        """
        Get custom day names from the calendar system.
        
        Returns:
            List of day names
        """
        try:
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            calendar_data = load_calendar_names(user_id, conversation_id)
            day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
            return day_names
        except Exception as e:
            logging.error(f"Error getting day names: {e}")
            return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    async def get_npc_details(self, ctx: RunContextWrapper, npc_id=None, npc_name=None) -> Dict[str, Any]:
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
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
                SELECT npc_id, npc_name, introduced, archetypes, archetype_summary, 
                       archetype_extras_summary, physical_description, relationships,
                       dominance, cruelty, closeness, trust, respect, intensity,
                       hobbies, personality_traits, likes, dislikes, affiliations,
                       schedule, current_location, sex, age
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s
            """
            
            params = [user_id, conversation_id]
            
            if npc_id is not None:
                query += " AND npc_id=%s"
                params.append(npc_id)
            elif npc_name is not None:
                query += " AND LOWER(npc_name)=LOWER(%s)"
                params.append(npc_name)
            else:
                return {"error": "No NPC ID or name provided"}
            
            query += " LIMIT 1"
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if not row:
                return {"error": "NPC not found"}
            
            # Process JSON fields
            def parse_json_field(field):
                if field is None:
                    return []
                if isinstance(field, str):
                    try:
                        return json.loads(field)
                    except:
                        return []
                return field
            
            npc_id, npc_name = row[0], row[1]
            archetypes = parse_json_field(row[3])
            archetype_summary = row[4]
            archetype_extras_summary = row[5]
            physical_description = row[6]
            relationships = parse_json_field(row[7])
            dominance, cruelty = row[8], row[9]
            closeness, trust = row[10], row[11]
            respect, intensity = row[12], row[13]
            hobbies = parse_json_field(row[14])
            personality_traits = parse_json_field(row[15])
            likes = parse_json_field(row[16])
            dislikes = parse_json_field(row[17])
            affiliations = parse_json_field(row[18])
            schedule = parse_json_field(row[19])
            current_location = row[20]
            sex, age = row[21], row[22]
            
            conn.close()
            
            return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "introduced": row[2],
                "archetypes": archetypes,
                "archetype_summary": archetype_summary,
                "archetype_extras_summary": archetype_extras_summary,
                "physical_description": physical_description,
                "relationships": relationships,
                "dominance": dominance,
                "cruelty": cruelty,
                "closeness": closeness,
                "trust": trust,
                "respect": respect,
                "intensity": intensity,
                "hobbies": hobbies,
                "personality_traits": personality_traits,
                "likes": likes,
                "dislikes": dislikes,
                "affiliations": affiliations,
                "schedule": schedule,
                "current_location": current_location,
                "sex": sex,
                "age": age
            }
        except Exception as e:
            logging.error(f"Error getting NPC details: {e}")
            return {"error": f"Error retrieving NPC details: {str(e)}"}
    
    async def generate_npc_name(self, ctx: RunContextWrapper, desired_gender="female", style="unique", forbidden_names=None) -> str:
        """
        Generate a unique name for an NPC.
        
        Args:
            desired_gender: Preferred gender for name generation
            style: Style of name (unique, fantasy, modern, etc.)
            forbidden_names: List of names to avoid
            
        Returns:
            Generated name
        """
        try:
            from logic.npc_creation import get_unique_npc_name, fetch_npc_name
            
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Get environment for context
            env_details = await self.get_environment_details(ctx)
            
            # Get existing NPC names to avoid duplicates
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s
            """, (user_id, conversation_id))
            
            existing_names = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if forbidden_names:
                existing_names.extend(forbidden_names)
            
            # Use existing function to get a name
            name = await fetch_npc_name(desired_gender, existing_names, env_details["environment_desc"], style)
            
            # Ensure uniqueness
            unique_name = get_unique_npc_name(name)
            
            return unique_name
        except Exception as e:
            logging.error(f"Error generating NPC name: {e}")
            
            # Fallback name generation
            import random
            first_names = ["Elara", "Thalia", "Vespera", "Lyra", "Nadia", "Corin", "Isadora", "Maren", "Octavia", "Quinn"]
            last_names = ["Valen", "Nightshade", "Wolfe", "Thorn", "Blackwood", "Frost", "Stone", "Rivers", "Skye", "Ash"]
            
            if forbidden_names:
                for name in list(first_names):
                    if name in forbidden_names:
                        first_names.remove(name)
            
            if not first_names:
                first_names = ["Unnamed"]
            
            return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    async def generate_physical_description(self, ctx: RunContextWrapper, npc_name, archetype_summary="", environment_desc=None) -> str:
        """
        Generate a detailed physical description for an NPC.
        
        Args:
            npc_name: Name of the NPC
            archetype_summary: Summary of the NPC's archetypes
            environment_desc: Description of the environment
            
        Returns:
            Physical description
        """
        try:
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Get environment if not provided
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]
            
            # Create a basic NPC data structure for the existing function
            npc_data = {
                "npc_name": npc_name,
                "archetype_summary": archetype_summary,
                "dominance": 50,
                "cruelty": 30,
                "intensity": 40,
                "personality_traits": [],
                "likes": [],
                "dislikes": []
            }
            
            # Use existing function
            description = await gpt_generate_physical_description(
                user_id, conversation_id, npc_data, environment_desc
            )
            
            return description
        except Exception as e:
            logging.error(f"Error generating physical description: {e}")
            return f"{npc_name} has an appearance that matches their personality and role in this environment."
    
    async def design_personality(self, ctx: RunContextWrapper, npc_name, archetype_summary="", environment_desc=None) -> NPCPersonalityData:
        """
        Design a coherent personality for an NPC.
        
        Args:
            npc_name: Name of the NPC
            archetype_summary: Summary of the NPC's archetypes
            environment_desc: Description of the environment
            
        Returns:
            NPCPersonalityData object
        """
        try:
            # Get environment if not provided
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]
            
            prompt = f"""
            Design a unique personality for {npc_name} in this environment:
            
            Environment: {environment_desc}
            
            Archetype Summary: {archetype_summary}
            
            Create a coherent personality with:
            - 3-5 distinct personality traits
            - 3-5 likes that align with the personality
            - 3-5 dislikes that create interesting tension
            - 2-4 hobbies or interests
            
            The personality should feel like a real individual with subtle psychological depth.
            Include traits that suggest hidden layers, motivations, or potential for character growth.
            """
            
            # Run the personality designer agent
            result = await Runner.run(
                self.personality_designer,
                prompt,
                context=ctx.context
            )
            
            return result.final_output
        except Exception as e:
            logging.error(f"Error designing personality: {e}")
            
            # Fallback personality generation
            return NPCPersonalityData(
                personality_traits=["confident", "observant", "private"],
                likes=["structure", "competence", "subtle control"],
                dislikes=["vulnerability", "unpredictability", "unnecessary conflict"],
                hobbies=["psychology", "strategic games"]
            )
    
    async def calibrate_stats(self, ctx: RunContextWrapper, npc_name, personality=None, archetype_summary="") -> NPCStatsData:
        """
        Calibrate NPC stats based on personality and archetypes.
        
        Args:
            npc_name: Name of the NPC
            personality: NPCPersonalityData object
            archetype_summary: Summary of the NPC's archetypes
            
        Returns:
            NPCStatsData object
        """
        try:
            personality_str = ""
            if personality:
                personality_str = f"""
                Personality Traits: {", ".join(personality.personality_traits)}
                Likes: {", ".join(personality.likes)}
                Dislikes: {", ".join(personality.dislikes)}
                Hobbies: {", ".join(personality.hobbies)}
                """
            
            prompt = f"""
            Calibrate stats for {npc_name} with:
            
            {personality_str}
            
            Archetype Summary: {archetype_summary}
            
            Determine appropriate values (0-100) for:
            - dominance: How naturally controlling/authoritative the NPC is
            - cruelty: How willing the NPC is to cause discomfort/distress
            - closeness: How emotionally available/connected the NPC is
            - trust: Trust toward the player (-100 to 100)
            - respect: Respect toward the player (-100 to 100)
            - intensity: Overall emotional/psychological intensity
            
            The stats should align coherently with the personality traits and archetypes.
            """
            
            # Run the stats calibrator agent
            result = await Runner.run(
                self.stats_calibrator,
                prompt,
                context=ctx.context
            )
            
            return result.final_output
        except Exception as e:
            logging.error(f"Error calibrating stats: {e}")
            
            # Fallback stats generation
            # Slight femdom bias as per the game's theme
            return NPCStatsData(
                dominance=60,
                cruelty=40,
                closeness=50,
                trust=20,
                respect=30,
                intensity=55
            )
    
    async def synthesize_archetypes(self, ctx: RunContextWrapper, archetype_names=None, npc_name="") -> NPCArchetypeData:
        """
        Synthesize multiple archetypes into a coherent character concept.
        
        Args:
            archetype_names: List of archetype names
            npc_name: Name of the NPC
            
        Returns:
            NPCArchetypeData object
        """
        try:
            if not archetype_names:
                # Get available archetypes
                available_archetypes = await self.get_available_archetypes(ctx)
                
                # Select a few random archetypes
                if available_archetypes:
                    selected = random.sample(available_archetypes, min(3, len(available_archetypes)))
                    archetype_names = [arch["name"] for arch in selected]
                else:
                    archetype_names = ["Mentor", "Authority Figure", "Hidden Depth"]
            
            archetypes_str = ", ".join(archetype_names)
            
            prompt = f"""
            Synthesize these archetypes for {npc_name}:
            
            Archetypes: {archetypes_str}
            
            Create:
            1. A cohesive archetype summary that blends these archetypes
            2. An extras summary explaining how the archetype fusion affects the character
            
            The synthesis should feel natural rather than forced, identifying common
            themes and resolving contradictions between archetypes.
            """
            
            # Run the archetype synthesizer agent
            result = await Runner.run(
                self.archetype_synthesizer,
                prompt,
                context=ctx.context
            )
            
            # Ensure archetype_names is preserved
            result.final_output.archetype_names = archetype_names
            
            return result.final_output
        except Exception as e:
            logging.error(f"Error synthesizing archetypes: {e}")
            
            # Fallback archetype synthesis
            return NPCArchetypeData(
                archetype_names=archetype_names or ["Authority Figure"],
                archetype_summary="A complex character with layers of authority and hidden depth.",
                archetype_extras_summary="This character's authority is expressed through subtle psychological control rather than overt dominance."
            )
    
    async def generate_schedule(self, ctx: RunContextWrapper, npc_name, environment_desc=None, day_names=None) -> Dict[str, Any]:
        """
        Generate a detailed schedule for an NPC.
        
        Args:
            npc_name: Name of the NPC
            environment_desc: Description of the environment
            day_names: List of day names
            
        Returns:
            Dictionary with the NPC's schedule
        """
        try:
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Get environment if not provided
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]
            
            # Get day names if not provided
            if not day_names:
                day_names = await self.get_day_names(ctx)
            
            # Get NPC data from the database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_id, archetypes, hobbies, personality_traits
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_name=%s
                LIMIT 1
            """, (user_id, conversation_id, npc_name))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                npc_id = row[0]
                
                # Parse JSON fields
                def parse_json_field(field):
                    if field is None:
                        return []
                    if isinstance(field, str):
                        try:
                            return json.loads(field)
                        except:
                            return []
                    return field
                
                archetypes = parse_json_field(row[1])
                hobbies = parse_json_field(row[2])
                personality_traits = parse_json_field(row[3])
                
                # Create NPC data for the existing function
                npc_data = {
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "archetypes": archetypes,
                    "hobbies": hobbies,
                    "personality_traits": personality_traits
                }
                
                # Use existing function
                schedule = await gpt_generate_schedule(
                    user_id, conversation_id, npc_data, environment_desc, day_names
                )
                
                return schedule
            else:
                # Fallback: create a simple schedule
                schedule = {}
                for day in day_names:
                    schedule[day] = {
                        "Morning": f"{npc_name} starts their day with personal routines.",
                        "Afternoon": f"{npc_name} attends to their primary responsibilities.",
                        "Evening": f"{npc_name} engages in social activities or hobbies.",
                        "Night": f"{npc_name} returns home and rests."
                    }
                
                return schedule
        except Exception as e:
            logging.error(f"Error generating schedule: {e}")
            
            # Fallback schedule generation
            day_names = day_names or ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            schedule = {}
            for day in day_names:
                schedule[day] = {
                    "Morning": f"{npc_name} starts their day with personal routines.",
                    "Afternoon": f"{npc_name} attends to their primary responsibilities.",
                    "Evening": f"{npc_name} engages in social activities or hobbies.",
                    "Night": f"{npc_name} returns home and rests."
                }
            
            return schedule
    
    async def generate_memories(self, ctx: RunContextWrapper, npc_name, environment_desc=None) -> List[str]:
        """
        Generate detailed memories for an NPC.
        
        Args:
            npc_name: Name of the NPC
            environment_desc: Description of the environment
            
        Returns:
            List of memory strings
        """
        try:
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Get environment if not provided
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]
            
            # Get NPC data from the database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_id, archetypes, archetype_summary, relationships
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_name=%s
                LIMIT 1
            """, (user_id, conversation_id, npc_name))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                npc_id = row[0]
                
                # Parse relationships
                def parse_json_field(field):
                    if field is None:
                        return []
                    if isinstance(field, str):
                        try:
                            return json.loads(field)
                        except:
                            return []
                    return field
                
                relationships = parse_json_field(row[3])
                
                # Create NPC data for the existing function
                npc_data = {
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "archetype_summary": row[2] if row[2] else ""
                }
                
                # Use existing function
                memories = await gpt_generate_memories(
                    user_id, conversation_id, npc_data, environment_desc, relationships
                )
                
                return memories
            else:
                # Fallback: create simple memories
                return [
                    f"I remember when I first arrived in this place. The atmosphere was both familiar and strange, like I belonged here but didn't yet know why.",
                    f"There was that conversation last month where I realized how easily people shared their secrets with me. It was fascinating how a simple question, asked the right way, could reveal so much.",
                    f"Sometimes I think about my position here and the subtle influence I've cultivated. Few realize how carefully I've positioned myself within the social dynamics."
                ]
        except Exception as e:
            logging.error(f"Error generating memories: {e}")
            
            # Fallback memory generation
            return [
                f"I remember when I first arrived in this place. The atmosphere was both familiar and strange, like I belonged here but didn't yet know why.",
                f"There was that conversation last month where I realized how easily people shared their secrets with me. It was fascinating how a simple question, asked the right way, could reveal so much.",
                f"Sometimes I think about my position here and the subtle influence I've cultivated. Few realize how carefully I've positioned myself within the social dynamics."
            ]
    
    async def create_npc_in_database(self, ctx: RunContextWrapper, npc_data) -> Dict[str, Any]:
        """
        Create an NPC in the database with complete details.
        
        Args:
            npc_data: Complete NPC data
            
        Returns:
            Dictionary with the created NPC details including ID
        """
        try:
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Extract values from npc_data
            npc_name = npc_data.get("npc_name", "Unnamed NPC")
            physical_description = npc_data.get("physical_description", "")
            introduced = npc_data.get("introduced", False)
            
            personality = npc_data.get("personality", {})
            personality_traits = personality.get("personality_traits", [])
            likes = personality.get("likes", [])
            dislikes = personality.get("dislikes", [])
            hobbies = personality.get("hobbies", [])
            
            stats = npc_data.get("stats", {})
            dominance = stats.get("dominance", 50)
            cruelty = stats.get("cruelty", 30)
            closeness = stats.get("closeness", 50)
            trust = stats.get("trust", 0)
            respect = stats.get("respect", 0)
            intensity = stats.get("intensity", 40)
            
            archetypes = npc_data.get("archetypes", {})
            archetype_names = archetypes.get("archetype_names", [])
            archetype_summary = archetypes.get("archetype_summary", "")
            archetype_extras_summary = archetypes.get("archetype_extras_summary", "")
            
            schedule = npc_data.get("schedule", {})
            memories = npc_data.get("memories", [])
            current_location = npc_data.get("current_location", "")
            
            # Use the existing create_and_refine_npc function
            npc_id = await create_and_refine_npc(
                user_id=user_id,
                conversation_id=conversation_id,
                npc_name=npc_name,
                physical_description=physical_description,
                environment_desc=npc_data.get("environment_desc", ""),
                day_names=await self.get_day_names(ctx),
                sex="female",  # Default to female NPCs as per game theme
                introduced=introduced,
                personality_traits=personality_traits,
                likes=likes,
                dislikes=dislikes,
                hobbies=hobbies,
                dominance=dominance,
                cruelty=cruelty,
                closeness=closeness,
                trust=trust,
                respect=respect,
                intensity=intensity,
                archetype_names=archetype_names,
                archetype_summary=archetype_summary,
                archetype_extras_summary=archetype_extras_summary,
                schedule=schedule,
                memories=memories,
                current_location=current_location
            )
            
            # Return the created NPC details
            return await self.get_npc_details(ctx, npc_id=npc_id)
        except Exception as e:
            logging.error(f"Error creating NPC in database: {e}")
            return {"error": f"Failed to create NPC: {str(e)}"}
    
    async def create_npc(self, ctx: RunContextWrapper, archetype_names=None, physical_desc=None, starting_traits=None) -> Dict[str, Any]:
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
        
        # Now create the NPC in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Extract archetype IDs if names were provided
        archetype_ids = []
        if archetype_names:
            for name in archetype_names:
                cursor.execute("""
                    SELECT id FROM Archetypes WHERE name=%s LIMIT 1
                """, (name,))
                row = cursor.fetchone()
                if row:
                    archetype_ids.append(row[0])
        
        # Build archetypes JSON
        archetypes_json = []
        for arch_id in archetype_ids:
            cursor.execute("SELECT name FROM Archetypes WHERE id=%s", (arch_id,))
            row = cursor.fetchone()
            if row:
                archetypes_json.append({"id": arch_id, "name": row[0]})
        
        # Insert the NPC
        cursor.execute("""
            INSERT INTO NPCStats (
                user_id, conversation_id, npc_name, introduced, archetypes,
                dominance, cruelty, closeness, trust, respect, intensity,
                hobbies, personality_traits, likes, dislikes, affiliations,
                physical_description, sex
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING npc_id
        """, 
        (user_id, conversation_id, npc_data.npc_name, npc_data.introduced,
        json.dumps(archetypes_json), 
        npc_data.dominance, npc_data.cruelty, npc_data.closeness,
        npc_data.trust, npc_data.respect, npc_data.intensity,
        json.dumps(npc_data.hobbies), json.dumps(npc_data.personality_traits),
        json.dumps(npc_data.likes), json.dumps(npc_data.dislikes),
        json.dumps(npc_data.affiliations), npc_data.physical_description,
        npc_data.sex))
        
        npc_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        
        # Call the existing create_and_refine_npc function for additional refinements
        await create_and_refine_npc(
            user_id,
            conversation_id,
            npc_id,
            npc_data.physical_description,
            env_details["environment_desc"]
        )
        
        # Return the created NPC
        return await self.get_npc_details(ctx, npc_id=npc_id)
    
    async def spawn_multiple_npcs(self, ctx: RunContextWrapper, count=5) -> List[int]:
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
        day_names = await self.get_day_names(ctx)
        
        # Use existing spawn_multiple_npcs_enhanced function
        npc_ids = await spawn_multiple_npcs_enhanced(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=env_details["environment_desc"],
            day_names=day_names,
            count=count
        )
        
        return npc_ids
    
    async def create_npc_with_context(self, environment_desc=None, archetype_names=None, specific_traits=None, user_id=None, conversation_id=None, db_dsn=None) -> NPCCreationResult:
        """
        Main function to create a complete NPC using the agent.
        
        Args:
            environment_desc: Description of the environment (optional)
            archetype_names: List of desired archetype names (optional)
            specific_traits: Dictionary with specific traits to incorporate (optional)
            user_id: User ID (required)
            conversation_id: Conversation ID (required)
            db_dsn: Database connection string (optional)
            
        Returns:
            NPCCreationResult object
        """
        if not user_id or not conversation_id:
            raise ValueError("user_id and conversation_id are required")
        
        # Create context
        context = NPCCreationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            db_dsn=db_dsn or DB_DSN
        )
        
        ctx = RunContextWrapper(context.dict())
        
        # Get environment description if not provided
        if not environment_desc:
            env_details = await self.get_environment_details(ctx)
            environment_desc = env_details["environment_desc"]
        
        # Build prompt
        archetypes_str = ", ".join(archetype_names) if archetype_names else "to be determined based on setting"
        traits_str = ""
        if specific_traits:
            traits_str = "Please incorporate these specific traits:\n"
            for trait_type, traits in specific_traits.items():
                if isinstance(traits, list):
                    traits_str += f"- {trait_type}: {', '.join(traits)}\n"
                else:
                    traits_str += f"- {trait_type}: {traits}\n"
        
        prompt = f"""
        Create a detailed, psychologically realistic NPC for this environment:
        
        {environment_desc}
        
        Desired archetypes: {archetypes_str}
        
        {traits_str}
        
        Generate a complete NPC with:
        1. A unique name and physical description
        2. A coherent personality with traits, likes, dislikes, and hobbies
        3. Appropriate stats (dominance, cruelty, etc.)
        4. A synthesis of the desired archetypes
        5. A detailed weekly schedule
        6. Rich, diverse memories
        
        The NPC should feel like a real person with psychological depth and subtle complexity.
        For femdom-themed worlds, incorporate natural elements of control and influence
        that feel organic to the character rather than forced or explicit.
        """
        
        # Generate a name
        npc_name = await self.generate_npc_name(ctx)
        
        # Synthesize archetypes
        archetypes = await self.synthesize_archetypes(ctx, archetype_names, npc_name)
        
        # Generate a physical description
        physical_description = await self.generate_physical_description(
            ctx, npc_name, archetypes.archetype_summary, environment_desc
        )
        
        # Design personality
        personality = await self.design_personality(
            ctx, npc_name, archetypes.archetype_summary, environment_desc
        )
        
        # Calibrate stats
        stats = await self.calibrate_stats(
            ctx, npc_name, personality, archetypes.archetype_summary
        )
        
        # Create the NPC in the database
        npc_data = {
            "npc_name": npc_name,
            "physical_description": physical_description,
            "personality": personality.dict(),
            "stats": stats.dict(),
            "archetypes": archetypes.dict(),
            "environment_desc": environment_desc
        }
        
        created_npc = await self.create_npc_in_database(ctx, npc_data)
        
        # Extract the NPC ID
        npc_id = created_npc.get("npc_id")
        
        # Generate schedule
        schedule = await self.generate_schedule(ctx, npc_name, environment_desc)
        
        # Generate memories
        memories = await self.generate_memories(ctx, npc_name, environment_desc)
        
        # Update schedule and memories in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE NPCStats
            SET schedule = %s, memory = %s
            WHERE npc_id = %s
        """, (json.dumps(schedule), json.dumps(memories), npc_id))
        conn.commit()
        conn.close()
        
        # Return the final NPC
        return NPCCreationResult(
            npc_id=npc_id,
            npc_name=npc_name,
            physical_description=physical_description,
            personality=personality,
            stats=stats,
            archetypes=archetypes,
            schedule=schedule,
            memories=memories,
            current_location=created_npc.get("current_location", "")
        )
