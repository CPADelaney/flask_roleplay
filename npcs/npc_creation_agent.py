# npcs/npc_creation_agent.py

import logging
import json
import asyncio
import random
from typing import List, Dict, Any, Optional
import os
import asyncpg
from datetime import datetime

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail
from pydantic import BaseModel, Field

# Import your existing modules
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
from db.connection import get_db_connection
from memory.wrapper import MemorySystem

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

class NPCCreationAgent:
    """Agent for dedicated NPC creation process"""
    
    def __init__(self):
        # Input validation guardrail
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
        
        # Main NPC creation agent
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
    
    async def get_available_archetypes(self, ctx):
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
    
    async def get_environment_details(self, ctx):
        """
        Get environment details from the database.
        
        Returns:
            Dictionary with environment details
        """
        try:
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
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
    
    async def get_day_names(self, ctx):
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
    
    async def generate_npc_name(self, ctx, desired_gender="female", style="unique", forbidden_names=None):
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
    
    async def generate_physical_description(self, ctx, npc_name, archetype_summary="", environment_desc=None):
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
    
    async def design_personality(self, ctx, npc_name, archetype_summary="", environment_desc=None):
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
    
    async def calibrate_stats(self, ctx, npc_name, personality=None, archetype_summary=""):
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
    
    async def synthesize_archetypes(self, ctx, archetype_names=None, npc_name=""):
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
    
    async def generate_schedule(self, ctx, npc_name, environment_desc=None, day_names=None):
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
                archetypes = []
                if row[1]:
                    try:
                        if isinstance(row[1], str):
                            archetypes = json.loads(row[1])
                        else:
                            archetypes = row[1]
                    except:
                        pass
                
                hobbies = []
                if row[2]:
                    try:
                        if isinstance(row[2], str):
                            hobbies = json.loads(row[2])
                        else:
                            hobbies = row[2]
                    except:
                        pass
                
                personality_traits = []
                if row[3]:
                    try:
                        if isinstance(row[3], str):
                            personality_traits = json.loads(row[3])
                        else:
                            personality_traits = row[3]
                    except:
                        pass
                
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
    
    async def generate_memories(self, ctx, npc_name, environment_desc=None):
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
                relationships = []
                if row[3]:
                    try:
                        if isinstance(row[3], str):
                            relationships = json.loads(row[3])
                        else:
                            relationships = row[3]
                    except:
                        pass
                
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
    
    async def create_npc_in_database(self, ctx, npc_data):
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
            
            # Get day names for scheduling
            day_names = await self.get_day_names(ctx)
            
            # Get environment details
            env_details = await self.get_environment_details(ctx)
            
            # Use the existing create_and_refine_npc function
            npc_id = await create_and_refine_npc(
                user_id=user_id,
                conversation_id=conversation_id,
                environment_desc=env_details["environment_desc"],
                day_names=day_names,
                sex="female"  # Default to female NPCs as per game theme
            )
            
            # Retrieve the created NPC details
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_id, npc_name, physical_description, archetypes, archetype_summary,
                       archetype_extras_summary, hobbies, personality_traits, likes, dislikes,
                       dominance, cruelty, closeness, trust, respect, intensity,
                       schedule, memory, current_location
                FROM NPCStats
                WHERE npc_id=%s
                LIMIT 1
            """, (npc_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Parse JSON fields
                archetypes = []
                if row[3]:
                    try:
                        if isinstance(row[3], str):
                            archetypes = json.loads(row[3])
                        else:
                            archetypes = row[3]
                    except:
                        archetypes = []
                
                hobbies = []
                if row[6]:
                    try:
                        if isinstance(row[6], str):
                            hobbies = json.loads(row[6])
                        else:
                            hobbies = row[6]
                    except:
                        hobbies = []
                
                personality_traits = []
                if row[7]:
                    try:
                        if isinstance(row[7], str):
                            personality_traits = json.loads(row[7])
                        else:
                            personality_traits = row[7]
                    except:
                        personality_traits = []
                
                likes = []
                if row[8]:
                    try:
                        if isinstance(row[8], str):
                            likes = json.loads(row[8])
                        else:
                            likes = row[8]
                    except:
                        likes = []
                
                dislikes = []
                if row[9]:
                    try:
                        if isinstance(row[9], str):
                            dislikes = json.loads(row[9])
                        else:
                            dislikes = row[9]
                    except:
                        dislikes = []
                
                schedule = {}
                if row[16]:
                    try:
                        if isinstance(row[16], str):
                            schedule = json.loads(row[16])
                        else:
                            schedule = row[16]
                    except:
                        schedule = {}
                
                memories = []
                if row[17]:
                    try:
                        if isinstance(row[17], str):
                            memories = json.loads(row[17])
                        else:
                            memories = row[17]
                    except:
                        memories = []
                
                archetype_names = []
                for arch in archetypes:
                    if isinstance(arch, dict) and "name" in arch:
                        archetype_names.append(arch["name"])
                
                return {
                    "npc_id": npc_id,
                    "npc_name": row[1],
                    "physical_description": row[2],
                    "personality": {
                        "personality_traits": personality_traits,
                        "likes": likes,
                        "dislikes": dislikes,
                        "hobbies": hobbies
                    },
                    "stats": {
                        "dominance": row[10],
                        "cruelty": row[11],
                        "closeness": row[12],
                        "trust": row[13],
                        "respect": row[14],
                        "intensity": row[15]
                    },
                    "archetypes": {
                        "archetype_names": archetype_names,
                        "archetype_summary": row[4] if row[4] else "",
                        "archetype_extras_summary": row[5] if row[5] else ""
                    },
                    "schedule": schedule,
                    "memories": memories,
                    "current_location": row[18] if row[18] else ""
                }
            else:
                return {"error": f"Failed to retrieve created NPC with ID {npc_id}"}
        except Exception as e:
            logging.error(f"Error creating NPC in database: {e}")
            return {"error": f"Failed to create NPC: {str(e)}"}
    
    async def create_npc(self, environment_desc=None, archetype_names=None, specific_traits=None, user_id=None, conversation_id=None, db_dsn=None):
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
        
        # Get environment description if not provided
        if not environment_desc:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
            """, (user_id, conversation_id))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                environment_desc = row[0]
            else:
                environment_desc = "A detailed, immersive world with subtle layers of control and influence."
        
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
        
        # Run the agent
        result = await Runner.run(
            self.agent,
            prompt,
            context=context.dict()
        )
        
        return result.final_output
    
    async def spawn_multiple_npcs(self, count=3, environment_desc=None, user_id=None, conversation_id=None, db_dsn=None):
        """
        Spawn multiple NPCs.
        
        Args:
            count: Number of NPCs to create
            environment_desc: Description of the environment (optional)
            user_id: User ID (required)
            conversation_id: Conversation ID (required)
            db_dsn: Database connection string (optional)
            
        Returns:
            List of NPC IDs
        """
        if not user_id or not conversation_id:
            raise ValueError("user_id and conversation_id are required")
        
        # Get environment description if not provided
        if not environment_desc:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
            """, (user_id, conversation_id))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                environment_desc = row[0]
            else:
                environment_desc = "A detailed, immersive world with subtle layers of control and influence."
        
        # Get day names
        day_names = await self.get_day_names(
            type("Context", (), {"context": {"user_id": user_id, "conversation_id": conversation_id}})
        )
        
        # Use existing function
        npc_ids = await spawn_multiple_npcs_enhanced(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=environment_desc,
            day_names=day_names,
            count=count
        )
        
        return npc_ids
