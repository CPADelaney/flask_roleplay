# npcs/new_npc_creation.py

"""
Unified NPC creation functionality.
This module consolidates and replaces functionality from:
- logic/npc_creation.py
- npcs/npc_creation_agent.py
- npcs/npc_handler_agent.py
"""

import logging
import json
import asyncio
import random
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import os
import asyncpg
from datetime import datetime

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail, RunContextWrapper
from db.connection import get_db_connection
from memory.wrapper import MemorySystem
from memory.core import Memory, MemoryType, MemorySignificance
from memory.managers import NPCMemoryManager
from memory.emotional import EmotionalMemoryManager
from memory.schemas import MemorySchemaManager
from memory.flashbacks import FlashbackManager
from memory.semantic import SemanticMemoryManager
from memory.masks import ProgressiveRevealManager, RevealType, RevealSeverity
from memory.reconsolidation import ReconsolidationManager

from logic.chatgpt_integration import get_openai_client, get_chatgpt_response
from logic.gpt_utils import spaced_gpt_call
from logic.gpt_helpers import fetch_npc_name
from logic.social_links import create_social_link
from logic.calendar import load_calendar_names

logger = logging.getLogger(__name__)

# Configuration
DB_DSN = os.getenv("DB_DSN")

# NPC mask slippage triggers - these are moments when the NPC's true nature begins to show
MASK_SLIPPAGE_TRIGGERS = {
    "dominance": [
        {"threshold": 30, "event": "subtle_control", "memory": "I let my control slip through a bit today. Nobody seemed to notice the subtle shift in dynamic."},
        {"threshold": 50, "event": "mask_adjustment", "memory": "It's getting harder to maintain this facade. I caught myself giving commands too firmly, had to play it off as a joke."},
        {"threshold": 70, "event": "partial_revelation", "memory": "I showed a glimpse of my true self today. The flash of fear in their eyes was... intoxicating."},
        {"threshold": 90, "event": "mask_removal", "memory": "I'm barely pretending anymore. Those who understand appreciate the honesty. Those who don't will learn."}
    ],
    "cruelty": [
        {"threshold": 30, "event": "sharp_comment", "memory": "I said something cutting today and quickly covered it with a laugh. The momentary hurt in their eyes was satisfying."},
        {"threshold": 50, "event": "testing_boundaries", "memory": "I'm pushing further each time to see what I can get away with. People are so willing to excuse 'playful' cruelty."},
        {"threshold": 70, "event": "deliberate_harm", "memory": "I orchestrated a situation today that caused genuine distress. I maintained plausible deniability, of course."},
        {"threshold": 90, "event": "overt_sadism", "memory": "My reputation for 'intensity' is established enough that I barely need to hide my enjoyment of others' suffering now."}
    ],
    "intensity": [
        {"threshold": 30, "event": "piercing_gaze", "memory": "Someone commented on my intense stare today. I've learned to soften it in public, but sometimes I forget."},
        {"threshold": 50, "event": "forceful_presence", "memory": "People naturally move aside when I walk through a room now. My presence is becoming harder to disguise."},
        {"threshold": 70, "event": "commanding_aura", "memory": "I no longer need to raise my voice to be obeyed. My quiet commands carry weight that surprises even me."},
        {"threshold": 90, "event": "overwhelming_presence", "memory": "The mask has become nearly transparent. My true nature radiates from me, drawing submission from the vulnerable."}
    ]
}

# Relationship stages that track the evolution of NPC-NPC and NPC-player relationships
RELATIONSHIP_STAGES = {
    "dominant": [
        {"level": 10, "name": "Initial Interest", "description": "Beginning to notice potential for control"},
        {"level": 30, "name": "Strategic Friendship", "description": "Establishing trust while assessing vulnerabilities"},
        {"level": 50, "name": "Subtle Influence", "description": "Exercising increasing control through 'guidance'"},
        {"level": 70, "name": "Open Control", "description": "Dropping pretense of equality in the relationship"},
        {"level": 90, "name": "Complete Dominance", "description": "Relationship is explicitly based on control and submission"}
    ],
    "alliance": [
        {"level": 10, "name": "Mutual Recognition", "description": "Recognizing similar controlling tendencies"},
        {"level": 30, "name": "Cautious Cooperation", "description": "Sharing limited information and techniques"},
        {"level": 50, "name": "Strategic Partnership", "description": "Actively collaborating while maintaining independence"},
        {"level": 70, "name": "Power Coalition", "description": "Forming a unified front with clear internal hierarchy"},
        {"level": 90, "name": "Dominant Cabal", "description": "Operating as a coordinated group to control others"}
    ],
    "rivalry": [
        {"level": 10, "name": "Veiled Competition", "description": "Competing subtly while maintaining cordial appearance"},
        {"level": 30, "name": "Strategic Undermining", "description": "Actively working to diminish the other's influence"},
        {"level": 50, "name": "Open Challenge", "description": "Directly competing for control and resources"},
        {"level": 70, "name": "Psychological Warfare", "description": "Actively attempting to break the other's control"},
        {"level": 90, "name": "Domination Contest", "description": "All-out struggle for supremacy"}
    ]
}

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
    Combines functionality from NPCCreationAgent and logic/npc_creation.py.
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
        
        # Memory creator agent
        self.memory_creator = Agent(
            name="MemoryCreator",
            instructions="""
            You create psychologically rich memories for NPCs that establish their background, relationships, and personality.
            
            Each memory should:
            - Be written in first-person perspective
            - Include sensory details and emotional responses
            - Reveal aspects of the NPC's true nature and personality
            - Establish connections to the environment and other characters
            - For femdom-themed worlds, subtly hint at control dynamics and authority patterns
            
            Memories should feel authentic and provide insight into how the NPC views themselves and others.
            """,
            tools=[
                function_tool(self.get_npc_details),
                function_tool(self.get_environment_details)
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
        
        # Keep track of existing NPC names to ensure uniqueness
        self.existing_npc_names = set()
    
    # --- Helper methods for robust parsing ---
    
    def safe_json_loads(self, text, default=None):
        """
        Safely parse JSON with multiple fallback methods.
        
        Args:
            text: Text to parse as JSON
            default: Default value to return if parsing fails
            
        Returns:
            Parsed JSON or default value
        """
        if not text:
            return default if default is not None else {}
        
        # Method 1: Direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Look for JSON object within text
        try:
            json_match = re.search(r'(\{[\s\S]*\})', text)
            if json_match:
                return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
        
        # Method 3: Try to fix common JSON syntax errors
        try:
            # Replace single quotes with double quotes
            fixed_text = text.replace("'", '"')
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        # Method 4: Extract field from markdown code block
        try:
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_block_match:
                return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
        
        # Return default if all parsing attempts fail
        return default if default is not None else {}
    
    def extract_field_from_text(self, text, field_name):
        """
        Extract a specific field from text that might contain JSON or key-value patterns.
        
        Args:
            text: Text to extract field from
            field_name: Name of the field to extract
            
        Returns:
            The field value or empty string if not found
        """
        # Try parsing as JSON first
        data = self.safe_json_loads(text)
        if data and field_name in data:
            return data[field_name]
        
        # Try regex patterns for field extraction
        patterns = [
            rf'"{field_name}"\s*:\s*"([^"]*)"',      # For string values: "field": "value"
            rf'"{field_name}"\s*:\s*(\[[^\]]*\])',     # For array values: "field": [...]
            rf'"{field_name}"\s*:\s*(\{{[^}}]*\}})',  # For object values: "field": {...}
            rf'{field_name}:\s*(.*?)(?:\n|$)',          # For plain text: field: value
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def get_unique_npc_name(self, proposed_name: str, existing_names=None) -> str:
        """
        Ensures a name is unique by checking against existing names and unwanted names.
        
        Args:
            proposed_name: The suggested name
            existing_names: List of names already in use
            
        Returns:
            A unique name
        """
        # If the name is "Seraphina" or already exists, choose an alternative from a predefined list.
        unwanted_names = {"seraphina"}
        existing_set = set(existing_names) if existing_names else self.existing_npc_names
        
        if proposed_name.strip().lower() in unwanted_names or proposed_name in existing_set:
            alternatives = ["Aurora", "Celeste", "Luna", "Nova", "Ivy", "Evelyn", "Isolde", "Marina"]
            # Filter out any alternatives already in use
            available = [name for name in alternatives if name not in existing_set and name.lower() not in unwanted_names]
            if available:
                new_name = random.choice(available)
            else:
                # If none available, simply append a random number
                new_name = f"{proposed_name}{random.randint(2, 99)}"
            return new_name
        
        # Add the name to our tracking set
        self.existing_npc_names.add(proposed_name)
        return proposed_name
    
    def clamp(self, value, min_val, max_val):
        """
        Clamp a value between a minimum and maximum.
        
        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Clamped value
        """
        return max(min_val, min(value, max_val))
    
    def dynamic_reciprocal_relationship(self, rel_type: str, archetype_summary: str = "") -> str:
        """
        Given a relationship label, return a reciprocal label in a context‐sensitive way.
        
        Args:
            rel_type: Relationship type (e.g., "thrall", "underling", "friend")
            archetype_summary: Summary of the target's archetype for context
            
        Returns:
            Reciprocal relationship label
        """
        fixed = {
            "mother": "child",
            "sister": "younger sibling",
            "aunt": "nephew/niece"
        }
        rel_lower = rel_type.lower()
        if rel_lower in fixed:
            return fixed[rel_lower]
        if rel_lower in ["friend", "best friend"]:
            return rel_type  # mutual relationship
        dynamic_options = {
            "underling": ["boss", "leader", "overseer"],
            "thrall": ["master", "controller", "dominator"],
            "enemy": ["rival", "adversary"],
            "lover": ["lover", "beloved"],
            "colleague": ["colleague"],
            "neighbor": ["neighbor"],
            "classmate": ["classmate"],
            "teammate": ["teammate"],
            "rival": ["rival", "competitor"],
        }
        if rel_lower in dynamic_options:
            if "dominant" in archetype_summary.lower() or "domina" in archetype_summary.lower():
                if rel_lower in ["underling", "thrall"]:
                    return "boss"
            return random.choice(dynamic_options[rel_lower])
        return "associate"
    
    def get_entity_name(self, conn, entity_type, entity_id, user_id, conversation_id):
        """
        Get the name of an entity (NPC or player).
        
        Args:
            conn: Database connection
            entity_type: Type of entity ("npc" or "player")
            entity_id: ID of the entity
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Name of the entity
        """
        if entity_type == 'player':
            return "Chase"
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT npc_name FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, entity_id))
        
        row = cursor.fetchone()
        cursor.close()
        
        return row[0] if row else "Unknown"
    
    def get_reciprocal_description(self, description):
        """
        Generate a reciprocal description from the perspective of the other entity.
        
        Args:
            description: Original description
            
        Returns:
            Reciprocal description
        """
        # Simple replacements for now
        replacements = {
            "control": "being controlled",
            "dominance": "submission",
            "manipulating": "being influenced",
            "assessing vulnerabilities": "being evaluated",
            "control and submission": "submission and control"
        }
        
        result = description
        for original, replacement in replacements.items():
            result = result.replace(original, replacement)
        
        return result
    
    def add_npc_memory(self, conn, user_id, conversation_id, npc_id, memory_text):
        """
        Add a memory entry for an NPC.
        
        Args:
            conn: Database connection
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            memory_text: Memory text to add
        """
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT memory FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            row = cursor.fetchone()
            if row and row[0]:
                if isinstance(row[0], str):
                    try:
                        memory = json.loads(row[0])
                    except:
                        memory = []
                else:
                    memory = row[0]
            else:
                memory = []
            
            memory.append(memory_text)
            
            cursor.execute("""
                UPDATE NPCStats
                SET memory = %s
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (json.dumps(memory), user_id, conversation_id, npc_id))
            
        finally:
            cursor.close()
    
    # --- NPC data retrieval methods ---
    
    async def get_available_archetypes(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Get available archetypes from the database.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
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
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            List of archetype objects with id and name
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name
                FROM Archetypes
                ORDER BY id
            """)
            
            archetypes = []
            for row in cursor.fetchall():
                archetypes.append({
                    "id": row[0],
                    "name": row[1]
                })
            
            conn.close()
            return archetypes
        except Exception as e:
            logging.error(f"Error suggesting archetypes: {e}")
            return []
    
    async def get_environment_details(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get details about the game environment.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            Dictionary with environment details
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        if not user_id or not conversation_id:
            return {
                "environment_desc": "A detailed, immersive world with subtle layers of control and influence.",
                "setting_name": "Default Setting",
                "locations": []
            }
        
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
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            List of location objects
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        if not user_id or not conversation_id:
            return []
        
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
        
        Args:
            ctx: Context wrapper with user and conversation IDs
        
        Returns:
            List of day names
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
            if not user_id or not conversation_id:
                return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
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
            ctx: Context wrapper with user and conversation IDs
            npc_id: ID of the NPC to get details for (optional)
            npc_name: Name of the NPC to get details for (optional)
            
        Returns:
            Dictionary with NPC details
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        if not user_id or not conversation_id:
            return {"error": "Missing user_id or conversation_id"}
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
                SELECT npc_id, npc_name, introduced, archetypes, archetype_summary, 
                       archetype_extras_summary, physical_description, relationships,
                       dominance, cruelty, closeness, trust, respect, intensity,
                       hobbies, personality_traits, likes, dislikes, affiliations,
                       schedule, current_location, sex, age, memory
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
            memories = parse_json_field(row[23])
            
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
                "age": age,
                "memories": memories
            }
        except Exception as e:
            logging.error(f"Error getting NPC details: {e}")
            return {"error": f"Error retrieving NPC details: {str(e)}"}
    
    # --- Helper methods for NPC generation ---
    
    async def integrate_femdom_elements(self, npc_data):
        """
        Analyze NPC data and subtly integrate femdom elements based on dominance level.
        This doesn't add overt femdom content, but rather plants seeds through traits.
        
        Args:
            npc_data: NPC data to enhance
            
        Returns:
            Updated NPC data with subtle femdom elements
        """
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 30)
        
        # Don't add femdom tendencies if dominance is very low
        if dominance < 20:
            return npc_data
            
        # Determine femdom intensity based on dominance and cruelty
        femdom_intensity = (dominance + cruelty) / 2
        is_high_intensity = femdom_intensity > 70
        is_medium_intensity = 40 <= femdom_intensity <= 70
        
        # Copy existing traits
        personality_traits = npc_data.get("personality_traits", [])
        likes = npc_data.get("likes", [])
        dislikes = npc_data.get("dislikes", [])
        
        # Add subtle femdom personality traits based on intensity
        potential_traits = []
        
        if is_high_intensity:
            potential_traits += [
                "enjoys being obeyed",
                "naturally commanding",
                "good at reading weaknesses",
                "expects compliance",
                "subtle manipulator",
                "finds pleasure in control",
                "notices when others defer to her",
            ]
        elif is_medium_intensity:
            potential_traits += [
                "prefers making decisions",
                "naturally takes charge",
                "surprisingly assertive at times",
                "notices small power dynamics",
                "enjoys being respected",
                "secretly enjoys having influence",
                "feels comfortable setting rules"
            ]
        else:
            potential_traits += [
                "occasionally assertive",
                "selective with permissions",
                "expects politeness",
                "notices disrespect quickly",
                "sometimes tests boundaries",
                "appreciates deference"
            ]
        
        # Select 1-2 traits to add without being too obvious
        traits_to_add = random.sample(potential_traits, min(2, len(potential_traits)))
        for trait in traits_to_add:
            if trait not in personality_traits:
                personality_traits.append(trait)
        
        # Add subtle femdom likes/dislikes based on intensity
        potential_likes = []
        potential_dislikes = []
        
        if is_high_intensity:
            potential_likes += [
                "seeing others follow her lead",
                "making important decisions",
                "being the center of attention",
                "quiet acknowledgment of her authority",
                "setting clear expectations"
            ]
            potential_dislikes += [
                "being interrupted",
                "unexpected defiance",
                "having her judgment questioned",
                "people who overstep boundaries",
                "being ignored"
            ]
        elif is_medium_intensity:
            potential_likes += [
                "receiving prompt responses",
                "being asked for permission",
                "planning events for others",
                "mentoring those who listen well",
                "being consulted on decisions"
            ]
            potential_dislikes += [
                "tardiness",
                "people who speak over her",
                "having to repeat herself",
                "casual dismissals of her opinions"
            ]
        else:
            potential_likes += [
                "well-mannered individuals",
                "being respected in conversations",
                "when others remember her preferences",
                "thoughtful attentiveness"
            ]
            potential_dislikes += [
                "poor etiquette",
                "being contradicted publicly",
                "presumptuous behavior"
            ]
        
        # Add 1 like and 1 dislike
        if potential_likes:
            new_like = random.choice(potential_likes)
            if new_like not in likes:
                likes.append(new_like)
                
        if potential_dislikes:
            new_dislike = random.choice(potential_dislikes)
            if new_dislike not in dislikes:
                dislikes.append(new_dislike)
        
        # Update the data and return
        updated_data = npc_data.copy()
        updated_data["personality_traits"] = personality_traits
        updated_data["likes"] = likes
        updated_data["dislikes"] = dislikes
        
        return updated_data
    
    # --- NPC generation methods ---
    
    async def generate_npc_name(self, ctx: RunContextWrapper, desired_gender="female", style="unique", forbidden_names=None) -> str:
        """
        Generate a unique name for an NPC.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
            desired_gender: Preferred gender for name generation
            style: Style of name (unique, fantasy, modern, etc.)
            forbidden_names: List of names to avoid
            
        Returns:
            Generated name
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
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
            unique_name = self.get_unique_npc_name(name, existing_names)
            
            return unique_name
        except Exception as e:
            logging.error(f"Error generating NPC name: {e}")
            
            # Fallback name generation
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
            ctx: Context wrapper with user and conversation IDs
            npc_name: Name of the NPC
            archetype_summary: Summary of the NPC's archetypes
            environment_desc: Description of the environment
            
        Returns:
            Physical description
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
            # Get environment if not provided
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]
            
            # Create a basic NPC data structure
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
            
            # Build the prompt
            prompt = f"""
            Generate a detailed physical description for {npc_name}, a female NPC in this femdom-themed environment:
            {environment_desc}

            IMPORTANT NPC DETAILS TO INCORPORATE:
            Archetype summary: {archetype_summary}
            Stats: Dominance 50/100, Cruelty 30/100, Intensity 40/100

            YOUR TASK:
            Create a detailed physical description that deeply integrates the archetype summary into the NPC's appearance. The archetype summary contains essential character information that should be physically manifested.

            The description must:
            1. Be 2-3 paragraphs with vivid, sensual details appropriate for a mature audience
            2. Directly translate key elements from the archetype summary into visible physical features
            3. Ensure clothing, accessories, and physical appearance reflect her specific archetype role
            4. Include distinctive physical features that immediately signal her archetype to observers
            5. Describe her characteristic expressions, posture, and mannerisms that reveal her personality
            6. Use sensory details beyond just visual (voice quality, scent, the feeling of her presence)
            7. Be written in third-person perspective with evocative, descriptive language
            8. Make sure to describe this character's curves in detail

            The description should allow someone to immediately understand the character's archetype and role from her appearance alone.

            Return a valid JSON object with the key "physical_description" containing the description as a string.
            """
            
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            description_json = response.choices[0].message.content
            data = self.safe_json_loads(description_json)
            
            if data and "physical_description" in data:
                return data["physical_description"]
            
            # Fallback: Extract description using regex if JSON parsing failed
            description = self.extract_field_from_text(description_json, "physical_description")
            if description and len(description) > 50:
                return description
                
        except Exception as e:
            logging.error(f"Error generating physical description for {npc_name}: {e}")
        
        # Final fallback
        return f"{npc_name} has an appearance that matches their personality and role in this environment."
    
    async def design_personality(self, ctx: RunContextWrapper, npc_name, archetype_summary="", environment_desc=None) -> NPCPersonalityData:
        """
        Design a coherent personality for an NPC.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
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
            ctx: Context wrapper with user and conversation IDs
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
            ctx: Context wrapper with user and conversation IDs
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
            themes and resolving contradictions between archetypes. Focus on how the
            archetypes interact to create a unique character foundation.
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
            ctx: Context wrapper with user and conversation IDs
            npc_name: Name of the NPC
            environment_desc: Description of the environment
            day_names: List of day names
            
        Returns:
            Dictionary with the NPC's schedule
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
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
                
                # Create NPC data for the prompt
                npc_data = {
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "archetypes": archetypes,
                    "hobbies": hobbies,
                    "personality_traits": personality_traits
                }
                
                # Build day example
                example_day = {
                    "Morning": "Activity description",
                    "Afternoon": "Activity description",
                    "Evening": "Activity description",
                    "Night": "Activity description"
                }
                example_schedule = {day: example_day for day in day_names}
                
                # Build prompt
                archetype_names = [a.get("name", "") for a in archetypes]
                prompt = f"""
                Generate a weekly schedule for {npc_name}, an NPC in this environment:
                {environment_desc}

                NPC Details:
                - Archetypes: {archetype_names}
                - Personality: {personality_traits}
                - Hobbies: {hobbies}

                The schedule must include all these days: {day_names}
                Each day must have activities for: Morning, Afternoon, Evening, and Night

                Return a valid JSON object with a single "schedule" key containing the complete weekly schedule.
                Example format:
                {json.dumps({"schedule": example_schedule}, indent=2)}

                Activities should reflect the NPC's personality, archetypes, and the environment.
                """
                
                client = get_openai_client()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}  # Force JSON output
                )
                
                schedule_json = response.choices[0].message.content
                data = self.safe_json_loads(schedule_json)
                
                if data and "schedule" in data and isinstance(data["schedule"], dict):
                    # Validate schedule has all required days and time periods
                    schedule = data["schedule"]
                    is_valid = True
                    
                    for day in day_names:
                        if day not in schedule:
                            is_valid = False
                            break
                            
                        day_schedule = schedule[day]
                        if not isinstance(day_schedule, dict):
                            is_valid = False
                            break
                            
                        for period in ["Morning", "Afternoon", "Evening", "Night"]:
                            if period not in day_schedule:
                                is_valid = False
                                break
                    
                    if is_valid:
                        return schedule
            
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
            ctx: Context wrapper with user and conversation IDs
            npc_name: Name of the NPC
            environment_desc: Description of the environment
            
        Returns:
            List of memory strings
        """
        try:
            user_id = ctx.context.get("user_id") 
            conversation_id = ctx.context.get("conversation_id")
            
            # Get environment if not provided
            if not environment_desc:
                env_details = await self.get_environment_details(ctx)
                environment_desc = env_details["environment_desc"]
            
            # Get NPC data from the database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_id, archetypes, archetype_summary, relationships,
                       dominance, cruelty, personality_traits
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_name=%s
                LIMIT 1
            """, (user_id, conversation_id, npc_name))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                npc_id = row[0]
                dominance = row[4]
                cruelty = row[5]
                
                # Parse relationships
                relationships = []
                if row[3]:
                    try:
                        if isinstance(row[3], str):
                            relationships = json.loads(row[3])
                        else:
                            relationships = row[3]
                    except:
                        relationships = []
                
                # Parse personality traits
                personality_traits = []
                if row[6]:
                    try:
                        if isinstance(row[6], str):
                            personality_traits = json.loads(row[6])
                        else:
                            personality_traits = row[6]
                    except:
                        personality_traits = []
                
                # Create prompt
                personality_context = ", ".join(personality_traits) if personality_traits else "complex personality"
                
                prompt = f"""
                Create 3-5 vivid, detailed memories for {npc_name} in this environment:
                
                {environment_desc}
                
                NPC INFORMATION:
                - Name: {npc_name}
                - Archetype: {row[2] if row[2] else "Unknown"}
                - Dominance Level: {dominance}/100
                - Cruelty Level: {cruelty}/100
                - Personality: {personality_context}
                
                MEMORY REQUIREMENTS:
                1. Each memory must be a SPECIFIC EVENT with concrete details - not vague impressions
                2. Include sensory details (sights, sounds, smells, textures)
                3. Include precise emotional responses and internal thoughts
                4. Include dialogue snippets with actual quoted speech
                5. Write in first-person perspective from {npc_name}'s viewpoint
                6. Each memory should be 3-5 sentences minimum with specific details
                
                IMPORTANT THEME GUIDANCE:
                * Include subtle hints of control dynamics without being overtly femdom
                * Show instances where {npc_name} momentarily revealed her true nature before quickly masking it
                * Show moments where {npc_name} tested boundaries or enjoyed having influence
                
                Return a valid JSON object with a single "memories" key containing an array of memory strings.
                """
                
                client = get_openai_client()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.8,
                    response_format={"type": "json_object"}
                )
                
                memories_json = response.choices[0].message.content
                data = self.safe_json_loads(memories_json)
                
                if data and "memories" in data and isinstance(data["memories"], list):
                    memories = data["memories"]
                    if memories and all(isinstance(m, str) for m in memories):
                        return memories
            
            # Fallback: create basic memories
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
            ctx: Context wrapper with user and conversation IDs
            npc_data: Complete NPC data
            
        Returns:
            Dictionary with the created NPC details including ID
        """
        try:
            user_id = ctx.context.get("user_id")
            conversation_id = ctx.context.get("conversation_id")
            
            # Get day names for scheduling
            day_names = await self.get_day_names(ctx)
            
            # Get environment details
            env_details = await self.get_environment_details(ctx)
            environment_desc = env_details["environment_desc"]
            
            # Extract values from npc_data
            npc_name = npc_data.get("npc_name", "Unnamed NPC")
            physical_description = npc_data.get("physical_description", "")
            introduced = npc_data.get("introduced", False)
            sex = npc_data.get("sex", "female")
            
            # Extract personality data
            personality = npc_data.get("personality", {})
            if isinstance(personality, dict):
                personality_traits = personality.get("personality_traits", [])
                likes = personality.get("likes", [])
                dislikes = personality.get("dislikes", [])
                hobbies = personality.get("hobbies", [])
            else:
                personality_traits = getattr(personality, "personality_traits", [])
                likes = getattr(personality, "likes", [])
                dislikes = getattr(personality, "dislikes", [])
                hobbies = getattr(personality, "hobbies", [])
            
            # Extract stats data
            stats = npc_data.get("stats", {})
            if isinstance(stats, dict):
                dominance = stats.get("dominance", 50)
                cruelty = stats.get("cruelty", 30)
                closeness = stats.get("closeness", 50)
                trust = stats.get("trust", 0)
                respect = stats.get("respect", 0)
                intensity = stats.get("intensity", 40)
            else:
                dominance = getattr(stats, "dominance", 50)
                cruelty = getattr(stats, "cruelty", 30)
                closeness = getattr(stats, "closeness", 50)
                trust = getattr(stats, "trust", 0)
                respect = getattr(stats, "respect", 0)
                intensity = getattr(stats, "intensity", 40)
            
            # Extract archetype data
            archetypes = npc_data.get("archetypes", {})
            if isinstance(archetypes, dict):
                archetype_names = archetypes.get("archetype_names", [])
                archetype_summary = archetypes.get("archetype_summary", "")
                archetype_extras_summary = archetypes.get("archetype_extras_summary", "")
            else:
                archetype_names = getattr(archetypes, "archetype_names", [])
                archetype_summary = getattr(archetypes, "archetype_summary", "")
                archetype_extras_summary = getattr(archetypes, "archetype_extras_summary", "")
            
            # Format archetypes for database
            archetype_objs = [{"name": name} for name in archetype_names]
            
            # Get or generate schedule
            schedule = npc_data.get("schedule", {})
            if not schedule:
                schedule = await self.generate_schedule(
                    ctx, npc_name, environment_desc, day_names
                )
            
            # Get or generate memories
            memories = npc_data.get("memories", [])
            if not memories:
                memories = await self.generate_memories(
                    ctx, npc_name, environment_desc
                )
            
            # Get current location based on schedule
            current_location = npc_data.get("current_location", "")
            if not current_location:
                # Determine current time of day and day
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value FROM CurrentRoleplay WHERE user_id=%s AND conversation_id=%s AND key='TimeOfDay'",
                    (user_id, conversation_id)
                )
                time_of_day = cursor.fetchone()
                time_of_day = time_of_day[0] if time_of_day else "Morning"
                
                cursor.execute(
                    "SELECT value FROM CurrentRoleplay WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'",
                    (user_id, conversation_id)
                )
                current_day_num = cursor.fetchone()
                current_day_num = int(current_day_num[0]) if current_day_num and current_day_num[0].isdigit() else 1
                
                # Calculate day index
                day_index = (current_day_num - 1) % len(day_names)
                current_day = day_names[day_index]
                
                # Extract current location from schedule
                if schedule and current_day in schedule and time_of_day in schedule[current_day]:
                    activity = schedule[current_day][time_of_day]
                    # Extract location from activity description
                    location_keywords = ["at the", "in the", "at", "in"]
                    for keyword in location_keywords:
                        if keyword in activity:
                            parts = activity.split(keyword, 1)
                            if len(parts) > 1:
                                potential_location = parts[1].split(".")[0].split(",")[0].strip()
                                if len(potential_location) > 3:  # Avoid very short fragments
                                    current_location = potential_location
                                    break
                
                # If we couldn't extract a location, use a random location from the database
                if not current_location:
                    cursor.execute(
                        "SELECT location_name FROM Locations WHERE user_id=%s AND conversation_id=%s ORDER BY RANDOM() LIMIT 1",
                        (user_id, conversation_id)
                    )
                    random_location = cursor.fetchone()
                    if random_location:
                        current_location = random_location[0]
                    else:
                        current_location = "Unknown"
                
                conn.close()
            
            # Generate age and birthdate
            age = random.randint(20, 50)  # Default age range
            # Get calendar data for birthdate
            calendar_data = load_calendar_names(user_id, conversation_id)
            months_list = calendar_data.get("months", [
                "Frostmoon", "Windspeak", "Bloomrise", "Dawnsveil",
                "Emberlight", "Goldencrest", "Shadowleaf", "Harvesttide",
                "Stormcall", "Nightwhisper", "Snowbound", "Yearsend"
            ])
            birth_month = random.choice(months_list)
            birth_day = random.randint(1, 28)
            birthdate = f"{birth_month} {birth_day}"
            
            # Apply subtle femdom elements
            npc_data_with_femdom = await self.integrate_femdom_elements({
                "npc_name": npc_name,
                "dominance": dominance,
                "cruelty": cruelty,
                "personality_traits": personality_traits,
                "likes": likes,
                "dislikes": dislikes
            })
            
            # Get updated traits
            personality_traits = npc_data_with_femdom.get("personality_traits", personality_traits)
            likes = npc_data_with_femdom.get("likes", likes)
            dislikes = npc_data_with_femdom.get("dislikes", dislikes)
            
            # Insert into database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO NPCStats (
                    user_id, conversation_id,
                    npc_name, introduced, sex,
                    dominance, cruelty, closeness, trust, respect, intensity,
                    archetypes, archetype_summary, archetype_extras_summary,
                    likes, dislikes, hobbies, personality_traits,
                    age, birthdate,
                    relationships, memory, schedule,
                    physical_description, current_location
                )
                VALUES (%s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s,
                        '[]'::jsonb, %s, %s,
                        %s, %s
                )
                RETURNING npc_id
            """, (
                user_id, conversation_id,
                npc_name, introduced, sex,
                dominance, cruelty, closeness, trust, respect, intensity,
                json.dumps(archetype_objs), archetype_summary, archetype_extras_summary,
                json.dumps(likes), json.dumps(dislikes), json.dumps(hobbies), json.dumps(personality_traits),
                age, birthdate,
                json.dumps(memories), json.dumps(schedule),
                physical_description, current_location
            ))
            
            npc_id = cursor.fetchone()[0]
            conn.commit()
            
            # Assign random relationships
            await self.assign_random_relationships(
                user_id, conversation_id, npc_id, npc_name, archetype_objs
            )
            
            # Initialize memory system
            try:
                memory_system = await MemorySystem.get_instance(user_id, conversation_id)
                
                # Store memories in the memory system
                await self.store_npc_memories(user_id, conversation_id, npc_id, memories)
                
                # Initialize emotional state
                await self.initialize_npc_emotional_state(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "archetype_summary": archetype_summary
                }, memories)
                
                # Generate initial beliefs
                await self.generate_npc_beliefs(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "archetype_summary": archetype_summary
                })
                
                # Initialize memory schemas
                await self.initialize_npc_memory_schemas(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "archetype_summary": archetype_summary
                })
                
                # Setup trauma model if appropriate
                await self.setup_npc_trauma_model(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "archetype_summary": archetype_summary
                }, memories)
                
                # Setup flashback triggers
                await self.setup_npc_flashback_triggers(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "archetype_summary": archetype_summary
                })
                
                # Generate counterfactual memories
                await self.generate_counterfactual_memories(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "archetype_summary": archetype_summary
                })
                
                # Plan mask revelations
                await self.plan_mask_revelations(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "archetype_summary": archetype_summary
                })
                
                # Setup relationship evolution tracking
                await self.setup_relationship_evolution_tracking(user_id, conversation_id, npc_id, relationships=[])
                
                # Build semantic networks
                await self.build_initial_semantic_network(user_id, conversation_id, npc_id, {
                    "npc_name": npc_name,
                    "archetype_summary": archetype_summary
                })
                
                # Detect memory patterns
                await self.detect_memory_patterns(user_id, conversation_id, npc_id)
                
                # Schedule memory maintenance
                await self.schedule_npc_memory_maintenance(user_id, conversation_id, npc_id)
                
            except Exception as e:
                logging.error(f"Error initializing memory system for NPC {npc_id}: {e}")
            
            conn.close()
            
            # Return the created NPC details
            return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "physical_description": physical_description,
                "personality": {
                    "personality_traits": personality_traits,
                    "likes": likes,
                    "dislikes": dislikes,
                    "hobbies": hobbies
                },
                "stats": {
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust,
                    "respect": respect,
                    "intensity": intensity
                },
                "archetypes": {
                    "archetype_names": archetype_names,
                    "archetype_summary": archetype_summary,
                    "archetype_extras_summary": archetype_extras_summary
                },
                "schedule": schedule,
                "memories": memories,
                "current_location": current_location
            }
        except Exception as e:
            logging.error(f"Error creating NPC in database: {e}")
            return {"error": f"Failed to create NPC: {str(e)}"}
    
    # --- Main NPC creation method ---
    
    async def create_npc(self, ctx: RunContextWrapper, archetype_names=None, physical_desc=None, starting_traits=None) -> Dict[str, Any]:
        """
        Create a new NPC with detailed characteristics.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
            archetype_names: List of archetype names to use (optional)
            physical_desc: Physical description of the NPC (optional)
            starting_traits: Initial personality traits (optional)
            
        Returns:
            Dictionary with the created NPC details
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
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
        created_npc = await self.create_npc_in_database(ctx, npc_data)
        
        return created_npc
    
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
    
    # --- Multiple NPC creation ---
    
    async def spawn_multiple_npcs(self, ctx: RunContextWrapper, count=3) -> List[int]:
        """
        Spawn multiple NPCs for the game world.
        
        Args:
            ctx: Context wrapper with user and conversation IDs
            count: Number of NPCs to spawn
            
        Returns:
            List of spawned NPC IDs
        """
        user_id = ctx.context.get("user_id")
        conversation_id = ctx.context.get("conversation_id")
        
        # Get environment description
        env_details = await self.get_environment_details(ctx)
        
        # Get day names
        day_names = await self.get_day_names(ctx)
        
        # Spawn NPCs one by one
        npc_ids = []
        for i in range(count):
            # Generate a unique NPC
            npc_data = await self.create_npc_with_context(
                environment_desc=env_details["environment_desc"],
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            npc_ids.append(npc_data.npc_id)
            
            # Add a small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        return npc_ids
    
    # --- Memory system methods ---
    
    async def store_npc_memories(self, user_id, conversation_id, npc_id, memories):
        """
        Store NPC memories using the memory system with enhanced features.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            memories: List of memory strings
        """
        if not memories:
            return
        
        # Get memory system instance
        memory_system = await MemorySystem.get_instance(user_id, conversation_id)
        
        # Create specialized NPC memory manager
        npc_memory_manager = await memory_system.npc_memory(npc_id)
        
        # Get NPC stats for contextualization
        npc_stats = await npc_memory_manager.get_npc_stats()
        dominance = npc_stats.get('dominance', 50)
        
        # Store each memory with appropriate tags and emotional analysis
        for i, memory_text in enumerate(memories):
            # Determine appropriate significance based on content and position
            # First memories are often more foundational
            significance = "high" if i < 2 else "medium"
            
            # Add appropriate tags based on content
            tags = ["initial_memory"]
            
            # Simple keyword tagging for demonstration
            if "childhood" in memory_text.lower() or "young" in memory_text.lower():
                tags.append("childhood")
            
            if "power" in memory_text.lower() or "control" in memory_text.lower():
                tags.append("power_dynamics")
                
            if "family" in memory_text.lower() or "parent" in memory_text.lower():
                tags.append("family")
                
            # Store with emotional analysis
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance=significance,
                emotional=True,  # Perform emotional analysis
                tags=tags
            )
            
            # For key memories, create semantic abstractions that represent general beliefs
            if significance == "high" and random.random() < 0.7:  # 70% chance for high significance memories
                semantic_manager = SemanticMemoryManager(user_id, conversation_id)
                
                # Get the recently added memory's ID
                recent_memories = await npc_memory_manager.retrieve_memories(limit=1)
                if recent_memories:
                    await semantic_manager.generate_semantic_memory(
                        source_memory_id=recent_memories[0].id,
                        entity_type="npc",
                        entity_id=npc_id,
                        abstraction_level=0.7  # Higher abstraction
                    )
    
    async def propagate_shared_memories(self, user_id, conversation_id, source_npc_id, source_npc_name, memories):
        """
        Propagate shared memories to related NPCs.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            source_npc_id: ID of the NPC whose memories to propagate
            source_npc_name: Name of the source NPC
            memories: List of memory strings to potentially propagate
        """
        if not memories:
            return
        
        # Get connections to other NPCs
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity2_id, link_type, link_level
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s 
            AND entity1_type='npc' AND entity1_id=%s
            AND entity2_type='npc'
        """, (user_id, conversation_id, source_npc_id))
        
        connections = []
        for row in cursor.fetchall():
            connections.append({
                "npc_id": row[0],
                "relationship": row[1],
                "strength": row[2]
            })
        
        cursor.close()
        conn.close()
        
        if not connections:
            return
        
        # Get memory system instance
        memory_system = await MemorySystem.get_instance(user_id, conversation_id)
        
        # Select a subset of memories to propagate (not all memories should be shared)
        for connection in connections:
            target_npc_id = connection["npc_id"]
            relationship = connection["relationship"]
            strength = connection["strength"]
            
            # Only propagate memories to closer relationships
            if strength < 30:
                continue
            
            # Get target NPC name
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT npc_name FROM NPCStats WHERE npc_id=%s AND user_id=%s AND conversation_id=%s",
                (target_npc_id, user_id, conversation_id)
            )
            row = cursor.fetchone()
            target_npc_name = row[0] if row else f"NPC {target_npc_id}"
            cursor.close()
            conn.close()
            
            # Decide how many memories to share (stronger connections share more)
            num_to_share = 1
            if strength > 50:
                num_to_share = 2
            if strength > 80:
                num_to_share = 3
                
            # Select random memories to propagate
            memories_to_share = random.sample(memories, min(num_to_share, len(memories)))
            
            # Transform memories to second-hand perspective
            for memory in memories_to_share:
                # Create a second-hand version of the memory
                second_hand_memory = f"{source_npc_name} told me about when {memory[0].lower() + memory[1:]}"
                
                # Store the transformed memory
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=target_npc_id,
                    memory_text=second_hand_memory,
                    importance="low",  # Lower importance for second-hand memories
                    emotional=True,
                    tags=["secondhand", "propagated", relationship]
                )
    
    async def initialize_npc_emotional_state(self, user_id, conversation_id, npc_id, npc_data, memories):
        """
        Initialize emotional state based on NPC traits and memories.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
            memories: List of memory strings
        
        Returns:
            Boolean indicating success
        """
        try:
            emotional_manager = EmotionalMemoryManager(user_id, conversation_id)
            
            # Determine base emotional state from NPC traits
            dominance = npc_data.get("dominance", 50)
            cruelty = npc_data.get("cruelty", 30)
            
            # Higher dominance tends toward confident emotions
            # Higher cruelty tends toward colder emotions
            primary_emotion = "neutral"
            if dominance > 70:
                primary_emotion = "confidence" if cruelty < 50 else "pride"
            elif cruelty > 70:
                primary_emotion = "contempt"
            elif dominance > 50 and cruelty > 50:
                primary_emotion = "satisfaction"
            
            # Set intensity based on traits
            intensity = ((dominance + cruelty) / 200) + 0.3  # 0.3-0.8 range
            
            # Create emotional state
            current_emotion = {
                "primary_emotion": primary_emotion,
                "intensity": intensity,
                "secondary_emotions": {},
                "valence": 0.1 if cruelty > 50 else 0.3,  # Slight positive bias
                "arousal": dominance / 200  # Higher dominance = higher arousal
            }
            
            await emotional_manager.update_entity_emotional_state(
                entity_type="npc",
                entity_id=npc_id,
                current_emotion=current_emotion
            )
            
            # Additionally, analyze the emotional content of the first memory
            if memories:
                await emotional_manager.add_emotional_memory(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memories[0],
                    primary_emotion=primary_emotion,
                    emotion_intensity=intensity
                )
            
            return True
        except Exception as e:
            logging.error(f"Error initializing emotional state for NPC {npc_id}: {e}")
            return False
    
    async def generate_npc_beliefs(self, user_id, conversation_id, npc_id, npc_data):
        """
        Generate initial beliefs based on NPC archetype.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Boolean indicating success
        """
        try:
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
            archetype_summary = npc_data.get("archetype_summary", "")
            
            # Generate beliefs based on archetype
            beliefs = []
            
            # Dominance-related beliefs
            if npc_data.get("dominance", 50) > 60:
                beliefs.append("I deserve to be in control of social situations.")
                beliefs.append("Those who submit easily are meant to be guided by stronger personalities.")
            
            # Cruelty-related beliefs
            if npc_data.get("cruelty", 30) > 60:
                beliefs.append("A little discomfort is necessary for growth in others.")
                beliefs.append("Emotional reactions reveal useful vulnerabilities in people.")
            
            # Archetype-specific beliefs (simple keyword matching for demonstration)
            if "maternal" in archetype_summary.lower() or "mother" in archetype_summary.lower():
                beliefs.append("I know what's best for those under my care.")
                beliefs.append("Guidance requires a firm hand and clear boundaries.")
            
            if "mentor" in archetype_summary.lower() or "teacher" in archetype_summary.lower():
                beliefs.append("Knowledge is a form of power to be carefully dispensed.")
                beliefs.append("Those who learn from me owe me their loyalty and respect.")
            
            # Add beliefs to memory system
            for belief_text in beliefs:
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=npc_id,
                    belief_text=belief_text,
                    confidence=0.8
                )
            
            return True
        except Exception as e:
            logging.error(f"Error generating beliefs for NPC {npc_id}: {e}")
            return False
    
    async def initialize_npc_memory_schemas(self, user_id, conversation_id, npc_id, npc_data):
        """
        Initialize basic memory schemas based on NPC archetype.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Boolean indicating success
        """
        try:
            schema_manager = MemorySchemaManager(user_id, conversation_id)
            archetype_summary = npc_data.get("archetype_summary", "")
            
            # Create a basic schema for interactions with the player
            await schema_manager.create_schema(
                entity_type="npc",
                entity_id=npc_id,
                schema_name="Player Interactions",
                description="Patterns in how the player behaves toward me",
                category="social",
                attributes={
                    "compliance_level": "unknown",
                    "respect_shown": "moderate",
                    "vulnerability_signs": "to be observed"
                }
            )
            
            # Create archetype-specific schemas
            if npc_data.get("dominance", 50) > 60:
                await schema_manager.create_schema(
                    entity_type="npc",
                    entity_id=npc_id,
                    schema_name="Control Dynamics",
                    description="Patterns of establishing and maintaining control",
                    category="power",
                    attributes={
                        "submission_triggers": "to be identified",
                        "resistance_patterns": "to be analyzed",
                        "effective_techniques": "to be developed"
                    }
                )
            
            return True
        except Exception as e:
            logging.error(f"Error initializing memory schemas for NPC {npc_id}: {e}")
            return False
    
    async def setup_npc_trauma_model(self, user_id, conversation_id, npc_id, npc_data, memories):
        """
        Set up trauma model for NPCs with traumatic backgrounds.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID  
            npc_id: NPC ID
            npc_data: Dict containing NPC data
            memories: List of memory strings
        
        Returns:
            Dict with trauma model information
        """
        try:
            # Only setup trauma for NPCs with higher cruelty or intense backgrounds
            cruelty = npc_data.get("cruelty", 0)
            archetype_summary = npc_data.get("archetype_summary", "").lower()
            
            has_traumatic_background = (
                cruelty > 70 or
                "trauma" in archetype_summary or 
                "tragic" in archetype_summary or
                "abused" in archetype_summary
            )
            
            if not has_traumatic_background:
                # Check memories for trauma indicators
                trauma_keywords = ["hurt", "pain", "suffer", "trauma", "abuse", "betray", "abandon"]
                memory_has_trauma = any(
                    any(keyword in memory.lower() for keyword in trauma_keywords)
                    for memory in memories
                )
                
                if not memory_has_trauma:
                    return {"trauma_model_needed": False}
            
            # At this point, we've determined trauma modeling is appropriate
            emotional_manager = EmotionalMemoryManager(user_id, conversation_id)
            
            # Create traumatic event record
            traumatic_memory = next(
                (memory for memory in memories 
                 if any(keyword in memory.lower() for keyword in ["hurt", "pain", "suffer", "trauma", "abuse", "betray", "abandon"])),
                memories[0] if memories else "A traumatic experience from the past that still affects me today."
            )
            
            # Analyze emotional content
            emotion_analysis = await emotional_manager.analyze_emotional_content(traumatic_memory)
            
            # Create trauma event
            trauma_event = {
                "memory_text": traumatic_memory,
                "emotion": emotion_analysis.get("primary_emotion", "fear"),
                "intensity": emotion_analysis.get("intensity", 0.8),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update emotional state with trauma
            await emotional_manager.update_entity_emotional_state(
                entity_type="npc",
                entity_id=npc_id,
                trauma_event=trauma_event
            )
            
            # Generate trauma triggers
            words = traumatic_memory.split()
            significant_words = [w for w in words if len(w) > 4 and w.isalpha()]
            triggers = random.sample(significant_words, min(3, len(significant_words)))
            
            # Store triggers in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE NPCStats
                SET trauma_triggers = %s
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """,
                (json.dumps(triggers), user_id, conversation_id, npc_id)
            )
            conn.commit()
            conn.close()
            
            return {
                "trauma_model_created": True,
                "trauma_triggers": triggers,
                "primary_emotion": emotion_analysis.get("primary_emotion")
            }
        except Exception as e:
            logging.error(f"Error setting up trauma model for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def setup_npc_flashback_triggers(self, user_id, conversation_id, npc_id, npc_data):
        """
        Set up potential flashback triggers based on NPC traits and memories.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Dict with flashback trigger information
        """
        try:
            flashback_manager = FlashbackManager(user_id, conversation_id)
            
            # Get NPC's memories to extract potential triggers
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
            npc_memories = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                limit=10
            )
            
            # Identify potential trigger words from memories
            trigger_words = []
            high_intensity_memories = [m for m in npc_memories.get("memories", []) 
                                      if m.get("emotional_intensity", 0) > 60]
            
            for memory in high_intensity_memories:
                # Extract significant words as potential triggers
                content = memory.get("text", "")
                words = [w for w in content.split() if len(w) > 4 and w.isalpha()]
                
                if words:
                    # Select 1-2 significant words as triggers
                    selected = random.sample(words, min(2, len(words)))
                    trigger_words.extend(selected)
            
            # If no triggers found from memories, use archetype-based triggers
            if not trigger_words:
                archetype = npc_data.get("archetype_summary", "").lower()
                
                if "mother" in archetype or "maternal" in archetype:
                    trigger_words = ["child", "mother", "family", "responsibility"]
                elif "teacher" in archetype or "mentor" in archetype:
                    trigger_words = ["student", "failure", "potential", "discipline"]
                elif "dominant" in archetype:
                    trigger_words = ["control", "power", "obedience", "submission"]
                else:
                    trigger_words = ["past", "mistake", "secret", "fear"]
            
            # Create a test flashback
            if trigger_words:
                test_flashback = await flashback_manager.check_for_triggered_flashback(
                    entity_type="npc",
                    entity_id=npc_id,
                    trigger_words=trigger_words,
                    chance=1.0  # Ensure creation for testing
                )
                
                # Store trigger words in database for future reference
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE NPCStats
                    SET flashback_triggers = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """,
                    (json.dumps(trigger_words), user_id, conversation_id, npc_id)
                )
                conn.commit()
                conn.close()
                
                return {
                    "triggers_established": len(trigger_words),
                    "trigger_words": trigger_words,
                    "test_flashback": test_flashback is not None
                }
        except Exception as e:
            logging.error(f"Error setting up flashback triggers for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def generate_counterfactual_memories(self, user_id, conversation_id, npc_id, npc_data):
        """
        Generate 'what-if' alternative versions of key memories to deepen personality.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Dict with counterfactual generation results
        """
        try:
            semantic_manager = SemanticMemoryManager(user_id, conversation_id)
            
            # Get NPC's existing memories
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
            memories_result = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                limit=5
            )
            
            # Find a significant memory for counterfactual generation
            significant_memories = [m for m in memories_result.get("memories", []) 
                                  if m.get("significance", 0) >= 3]
            
            if not significant_memories:
                return {"counterfactuals_generated": 0}
                
            # Generate counterfactuals for the most significant memory
            target_memory = significant_memories[0]
            
            # Generate opposite outcome counterfactual
            opposite_cf = await semantic_manager.generate_counterfactual(
                memory_id=target_memory["id"],
                entity_type="npc",
                entity_id=npc_id,
                variation_type="opposite"
            )
            
            # Generate exaggerated outcome counterfactual
            exaggerated_cf = await semantic_manager.generate_counterfactual(
                memory_id=target_memory["id"],
                entity_type="npc",
                entity_id=npc_id,
                variation_type="exaggeration"
            )
            
            return {
                "counterfactuals_generated": 2,
                "based_on_memory": target_memory["text"],
                "counterfactuals": [
                    opposite_cf.get("counterfactual_text"),
                    exaggerated_cf.get("counterfactual_text")
                ]
            }
        except Exception as e:
            logging.error(f"Error generating counterfactuals for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def plan_mask_revelations(self, user_id, conversation_id, npc_id, npc_data):
        """
        Create a revelation plan for gradually exposing the NPC's true nature.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Dict with revelation plan information
        """
        try:
            mask_manager = ProgressiveRevealManager(user_id, conversation_id)
            
            # Get current mask info
            mask_info = await mask_manager.get_npc_mask(npc_id)
            if "error" in mask_info:
                return {"error": mask_info["error"]}
            
            # Hidden traits that need to be revealed
            hidden_traits = mask_info.get("hidden_traits", {})
            if not hidden_traits:
                return {"revelation_plan_needed": False}
            
            # Create a progressive revelation plan
            revelation_plan = []
            
            # Plan subtle revelations for early encounters
            for trait_name in hidden_traits.keys():
                # Early stage - subtle hints through physical tells
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.SUBTLE,
                    "type": RevealType.PHYSICAL,
                    "stage": "early",
                    "integrity_threshold": 90,
                    "trigger_contexts": ["stress", "unexpected", "authority challenged"]
                })
                
                # Mid stage - verbal slips
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.MINOR,
                    "type": RevealType.VERBAL_SLIP,
                    "stage": "mid",
                    "integrity_threshold": 70,
                    "trigger_contexts": ["command", "confrontation", "private conversation"]
                })
                
                # Later stage - behavioral inconsistencies
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.MODERATE,
                    "type": RevealType.BEHAVIOR,
                    "stage": "later",
                    "integrity_threshold": 50,
                    "trigger_contexts": ["frustration", "opportunity", "unexpected behavior"]
                })
                
                # Final stage - direct revelation
                revelation_plan.append({
                    "trait": trait_name,
                    "severity": RevealSeverity.MAJOR,
                    "type": RevealType.EMOTIONAL,
                    "stage": "final",
                    "integrity_threshold": 30,
                    "trigger_contexts": ["confronted", "cornered", "powerful position"]
                })
            
            # Store the revelation plan
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE NPCStats
                SET revelation_plan = %s
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """,
                (json.dumps(revelation_plan), user_id, conversation_id, npc_id)
            )
            conn.commit()
            conn.close()
            
            return {
                "revelation_plan_created": True,
                "planned_revelations": len(revelation_plan),
                "traits_covered": list(hidden_traits.keys())
            }
        except Exception as e:
            logging.error(f"Error planning mask revelations for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def setup_relationship_evolution_tracking(self, user_id, conversation_id, npc_id, relationships):
        """
        Setup tracking for relationship evolution based on the memory system.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            relationships: List of relationship data
        
        Returns:
            Dict with tracking setup results
        """
        try:
            if not relationships:
                return {"relationships_tracked": 0}
            
            # For each relationship, establish evolution parameters
            for relationship in relationships:
                entity_type = relationship.get("entity_type")
                entity_id = relationship.get("entity_id")
                relationship_label = relationship.get("relationship_label", "associate")
                
                if not entity_type or not entity_id:
                    continue
                
                # Create a relationship tracker entry
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO RelationshipEvolution (
                        user_id, conversation_id, npc1_id, entity2_type, entity2_id, 
                        relationship_type, current_stage, progress_to_next, evolution_history
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, conversation_id, npc1_id, entity2_type, entity2_id) 
                    DO NOTHING
                    """,
                    (
                        user_id, conversation_id, npc_id, entity_type, entity_id,
                        relationship_label, "initial", 0,
                        json.dumps([{
                            "stage": "initial",
                            "date": datetime.now().isoformat(),
                            "note": f"Relationship as {relationship_label} established"
                        }])
                    )
                )
                conn.commit()
                conn.close()
            
            return {"relationships_tracked": len(relationships)}
        except Exception as e:
            logging.error(f"Error setting up relationship evolution for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def build_initial_semantic_network(self, user_id, conversation_id, npc_id, npc_data):
        """
        Build initial semantic networks for the NPC's knowledge structure.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_data: Dict containing NPC data
        
        Returns:
            Dict with semantic network information
        """
        try:
            semantic_manager = SemanticMemoryManager(user_id, conversation_id)
            
            # Determine central topics based on archetype
            archetype_summary = npc_data.get("archetype_summary", "").lower()
            central_topics = []
            
            # Extract key themes from archetype
            if "mother" in archetype_summary or "maternal" in archetype_summary:
                central_topics.append("Family")
            if "teacher" in archetype_summary or "mentor" in archetype_summary:
                central_topics.append("Education")
            if "dominant" in archetype_summary or "control" in archetype_summary:
                central_topics.append("Power")
            if "professional" in archetype_summary or "career" in archetype_summary:
                central_topics.append("Career")
            
            # Add default topic if none detected
            if not central_topics:
                central_topics.append("Self")
            
            # Build semantic networks for each central topic
            networks = []
            for topic in central_topics:
                network = await semantic_manager.build_semantic_network(
                    entity_type="npc",
                    entity_id=npc_id,
                    central_topic=topic,
                    depth=1  # Start with shallow networks
                )
                
                # Store network in database
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO SemanticNetworks (
                        user_id, conversation_id, entity_type, entity_id,
                        central_topic, network_data, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """,
                    (user_id, conversation_id, "npc", npc_id, topic, json.dumps(network))
                )
                conn.commit()
                conn.close()
                
                networks.append({
                    "topic": topic,
                    "nodes": len(network.get("nodes", [])),
                    "edges": len(network.get("edges", []))
                })
            
            return {
                "semantic_networks_created": len(networks),
                "networks": networks
            }
        except Exception as e:
            logging.error(f"Error building semantic networks for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def detect_memory_patterns(self, user_id, conversation_id, npc_id):
        """
        Detect patterns in memories to establish consistent personality traits.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
        
        Returns:
            Dict with pattern detection results
        """
        try:
            schema_manager = MemorySchemaManager(user_id, conversation_id)
            
            # Attempt to detect schemas from existing memories
            result = await schema_manager.detect_schema_from_memories(
                entity_type="npc",
                entity_id=npc_id,
                min_memories=2  # Lower threshold for initial detection
            )
            
            if result.get("schema_detected", False):
                # Schema detected - we have a pattern to work with
                schema_id = result.get("schema_id")
                schema_name = result.get("schema_name")
                
                # Store this as a personality pattern
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE NPCStats
                    SET personality_patterns = personality_patterns || %s::jsonb
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """,
                    (json.dumps([{
                        "pattern_name": schema_name,
                        "schema_id": schema_id,
                        "confidence": result.get("confidence", 0.7),
                        "detected_at": datetime.now().isoformat()
                    }]), user_id, conversation_id, npc_id)
                )
                conn.commit()
                conn.close()
                
                return {
                    "pattern_detected": True,
                    "pattern_name": schema_name,
                    "schema_id": schema_id
                }
            
            return {"pattern_detected": False}
        except Exception as e:
            logging.error(f"Error detecting memory patterns for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def schedule_npc_memory_maintenance(self, user_id, conversation_id, npc_id):
        """
        Schedule regular memory maintenance for an NPC.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
        
        Returns:
            Dict with scheduling results
        """
        try:
            # Create maintenance schedule entry in database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if we already have a schedule
            cursor.execute(
                """
                SELECT 1 FROM MemoryMaintenanceSchedule
                WHERE user_id=%s AND conversation_id=%s AND entity_type='npc' AND entity_id=%s
                """,
                (user_id, conversation_id, npc_id)
            )
            
            if cursor.fetchone():
                # Already scheduled
                conn.close()
                return {"already_scheduled": True}
            
            # Create maintenance schedule
            # Different maintenance types happen at different intervals:
            maintenance_types = [
                {
                    "type": "consolidation",
                    "description": "Consolidate related memories",
                    "interval_days": 3,
                    "last_run": None
                },
                {
                    "type": "decay",
                    "description": "Apply memory decay to old memories",
                    "interval_days": 7,
                    "last_run": None
                },
                {
                    "type": "schema_update",
                    "description": "Update memory schemas based on new experiences",
                    "interval_days": 5,
                    "last_run": None
                },
                {
                    "type": "mask_update",
                    "description": "Evolve mask integrity based on interactions",
                    "interval_days": 2,
                    "last_run": None
                }
            ]
            
            cursor.execute(
                """
                INSERT INTO MemoryMaintenanceSchedule (
                    user_id, conversation_id, entity_type, entity_id,
                    maintenance_schedule, next_maintenance_date
                )
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP + INTERVAL '1 day')
                """,
                (user_id, conversation_id, "npc", npc_id, json.dumps(maintenance_types))
            )
            conn.commit()
            conn.close()
            
            return {
                "maintenance_scheduled": True,
                "maintenance_types": len(maintenance_types)
            }
        except Exception as e:
            logging.error(f"Error scheduling memory maintenance for NPC {npc_id}: {e}")
            return {"error": str(e)}
    
    async def check_for_mask_slippage(self, user_id, conversation_id, npc_id):
        """
        Check if an NPC has reached thresholds where their true nature begins to show.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            
        Returns:
            List of triggered events or None if error
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get NPC's current stats
            cursor.execute("""
                SELECT npc_name, dominance, cruelty, intensity, memory
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            npc_name, dominance, cruelty, intensity, memory_json = row
            
            # Parse memory
            if memory_json:
                if isinstance(memory_json, str):
                    try:
                        memory = json.loads(memory_json)
                    except:
                        memory = []
                else:
                    memory = memory_json
            else:
                memory = []
                
            # Get slippage history
            cursor.execute("""
                SELECT mask_slippage_events
                FROM NPCEvolution
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            row = cursor.fetchone()
            if row and row[0]:
                if isinstance(row[0], str):
                    try:
                        slippage_history = json.loads(row[0])
                    except:
                        slippage_history = []
                else:
                    slippage_history = row[0]
            else:
                slippage_history = []
                
            # Check each stat for slippage triggers
            triggered_events = []
            
            for stat_name, thresholds in MASK_SLIPPAGE_TRIGGERS.items():
                stat_value = locals().get(stat_name, 0)  # Get the stat value from local vars
                
                for threshold in thresholds:
                    event_name = threshold["event"]
                    
                    # Skip if this slippage has already occurred
                    if any(e.get("event") == event_name for e in slippage_history):
                        continue
                        
                    # Check if threshold is met
                    if stat_value >= threshold["threshold"]:
                        # Record this slippage event
                        event_data = {
                            "event": event_name,
                            "stat": stat_name,
                            "threshold": threshold["threshold"],
                            "timestamp": datetime.now().isoformat()
                        }
                        triggered_events.append(event_data)
                        
                        # Add the memory
                        if "memory" in threshold:
                            memory.append(threshold["memory"])
                            
                        # Subtle physical description changes based on mask slippage
                        if stat_name == "dominance" and threshold["threshold"] >= 50:
                            cursor.execute("""
                                UPDATE NPCStats
                                SET physical_description = CONCAT(physical_description, ' In unguarded moments, her demeanor shifts subtly—a barely perceptible straightening of posture, a flash of something commanding in her eyes that quickly vanishes when noticed.')
                                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                            """, (user_id, conversation_id, npc_id))
                        
                        if stat_name == "cruelty" and threshold["threshold"] >= 50:
                            cursor.execute("""
                                UPDATE NPCStats
                                SET physical_description = CONCAT(physical_description, ' Occasionally her smile doesn't quite reach her eyes, revealing a momentary coldness before she adjusts her expression back to warmth.')
                                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                            """, (user_id, conversation_id, npc_id))
                            
                        if stat_name == "intensity" and threshold["threshold"] >= 50:
                            cursor.execute("""
                                UPDATE NPCStats
                                SET physical_description = CONCAT(physical_description, ' Sometimes when she thinks no one is watching, her gaze becomes unnervingly focused, studying others with an analytical intensity that disappears behind a pleasant mask when attention returns to her.')
                                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                            """, (user_id, conversation_id, npc_id))
            
            # Update memory
            cursor.execute("""
                UPDATE NPCStats
                SET memory = %s
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (json.dumps(memory), user_id, conversation_id, npc_id))
            
            # Update slippage history
            if triggered_events:
                slippage_history.extend(triggered_events)
                
                # Check if NPCEvolution record exists
                cursor.execute("""
                    SELECT 1 FROM NPCEvolution
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                if cursor.fetchone():
                    cursor.execute("""
                        UPDATE NPCEvolution
                        SET mask_slippage_events = %s
                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """, (json.dumps(slippage_history), user_id, conversation_id, npc_id))
                else:
                    cursor.execute("""
                        INSERT INTO NPCEvolution
                        (user_id, conversation_id, npc_id, mask_slippage_events)
                        VALUES (%s, %s, %s, %s)
                    """, (user_id, conversation_id, npc_id, json.dumps(slippage_history)))
            
            conn.commit()
            return triggered_events
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error checking mask slippage: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    async def assign_random_relationships(self, user_id, conversation_id, npc_id, npc_name, npc_archetypes=None):
        """
        Assign random relationships between the new NPC and other entities.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: NPC ID
            npc_name: NPC name
            npc_archetypes: List of NPC archetypes (optional)
        """
        logging.info(f"Assigning relationships for NPC {npc_id} ({npc_name})")
        
        relationships = []

        # Define explicit mapping for archetypes to relationship labels.
        explicit_role_map = {
            "mother": "mother",
            "stepmother": "stepmother",
            "aunt": "aunt",
            "older sister": "older sister",
            "stepsister": "stepsister",
            "babysitter": "babysitter",
            "friend from online interactions": "online friend",
            "neighbor": "neighbor",
            "rival": "rival",
            "classmate": "classmate",
            "lover": "lover",
            "colleague": "colleague",
            "teammate": "teammate",
            "boss/supervisor": "boss/supervisor",
            "teacher/principal": "teacher/principal",
            "landlord": "landlord",
            "roommate/housemate": "roommate",
            "ex-girlfriend/ex-wife": "ex-partner",
            "therapist": "therapist",
            "domestic authority": "head of household",
            "the one who got away": "the one who got away",
            "childhood friend": "childhood friend",
            "friend's wife": "friend",
            "friend's girlfriend": "friend",
            "best friend's sister": "friend's sister"
        }
        
        # First, add relationships based on explicit archetype mapping.
        if npc_archetypes:
            for arc in npc_archetypes:
                arc_name = arc.get("name", "").strip().lower()
                if arc_name in explicit_role_map:
                    rel_label = explicit_role_map[arc_name]
                    # Add relationship from NPC to player using the explicit role.
                    relationships.append({
                        "target_entity_type": "player",
                        "target_entity_id": user_id,  # player ID
                        "relationship_label": rel_label
                    })
                    logging.info(f"Added explicit relationship '{rel_label}' for NPC {npc_id} to player.")
        
        # Next, determine which explicit roles (if any) were already added.
        explicit_roles_added = {rel["relationship_label"] for rel in relationships}
        
        # Define default lists for random selection.
        default_familial = ["mother", "sister", "aunt"]
        default_non_familial = ["enemy", "friend", "best friend", "lover", "neighbor",
                              "colleague", "classmate", "teammate", "underling", "rival", "ex-girlfriend", "ex-wife", "boss", "roommate", "childhood friend"]
        
        # If no explicit familial role was added, consider assigning a random non-familial relationship with the player.
        if not (explicit_roles_added & set(default_familial)):
            if random.random() < 0.5:
                rel_type = random.choice(default_non_familial)
                relationships.append({
                    "target_entity_type": "player",
                    "target_entity_id": user_id,
                    "relationship_label": rel_type
                })
                logging.info(f"Randomly added non-familial relationship '{rel_type}' for NPC {npc_id} to player.")
        
        # Now add relationships with other NPCs.
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT npc_id, npc_name, archetype_summary
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
        """, (user_id, conversation_id, npc_id))
        rows = cursor.fetchall()
        conn.close()
        
        # For each other NPC, use explicit mapping if possible; otherwise, fall back to random choice.
        for (old_npc_id, old_npc_name, old_arche_summary) in rows:
            if random.random() < 0.3:
                # Check if the current NPC's explicit roles should be used.
                if explicit_roles_added:
                    # Prefer one of the explicit roles if available.
                    rel_type = random.choice(list(explicit_roles_added))
                else:
                    rel_type = random.choice(default_non_familial)
                relationships.append({
                    "target_entity_type": "npc",
                    "target_entity_id": old_npc_id,
                    "relationship_label": rel_type,
                    "target_archetype_summary": old_arche_summary or ""
                })
                logging.info(f"Added relationship '{rel_type}' between NPC {npc_id} and NPC {old_npc_id}.")
        
        # Finally, create these relationships in the database and generate associated memories.
        memory_system = await MemorySystem.get_instance(user_id, conversation_id)
        
        for rel in relationships:
            if rel["target_entity_type"] == "player":
                create_social_link(
                    user_id, conversation_id,
                    entity1_type="npc", entity1_id=npc_id,
                    entity2_type="player", entity2_id=rel["target_entity_id"],
                    link_type=rel["relationship_label"]
                )
                
                # Add to NPCStats.relationships
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT relationships FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                row = cursor.fetchone()
                relationships_json = row[0] if row else "[]"
                
                if isinstance(relationships_json, str):
                    try:
                        current_relationships = json.loads(relationships_json)
                    except:
                        current_relationships = []
                else:
                    current_relationships = relationships_json
                
                # Add the new relationship
                current_relationships.append({
                    "relationship_label": rel["relationship_label"],
                    "entity_type": "player", 
                    "entity_id": rel["target_entity_id"]
                })
                
                # Update the NPCStats
                cursor.execute("""
                    UPDATE NPCStats
                    SET relationships = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(current_relationships), user_id, conversation_id, npc_id))
                
                conn.commit()
                conn.close()
                
            else:  # NPC to NPC relationship
                old_npc_id = rel["target_entity_id"]
                old_arche_summary = rel.get("target_archetype_summary", "")
                
                # Create forward link
                create_social_link(
                    user_id, conversation_id,
                    entity1_type="npc", entity1_id=npc_id,
                    entity2_type="npc", entity2_id=old_npc_id,
                    link_type=rel["relationship_label"]
                )
                
                # Add to NPCStats.relationships
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT relationships FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                row = cursor.fetchone()
                relationships_json = row[0] if row else "[]"
                
                if isinstance(relationships_json, str):
                    try:
                        current_relationships = json.loads(relationships_json)
                    except:
                        current_relationships = []
                else:
                    current_relationships = relationships_json
                
                # Add the new relationship
                current_relationships.append({
                    "relationship_label": rel["relationship_label"],
                    "entity_type": "npc", 
                    "entity_id": old_npc_id
                })
                
                # Update the NPCStats
                cursor.execute("""
                    UPDATE NPCStats
                    SET relationships = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(current_relationships), user_id, conversation_id, npc_id))
                
                conn.commit()
                
                # Create reverse link with reciprocal relationship
                rec_type = self.dynamic_reciprocal_relationship(
                    rel["relationship_label"],
                    old_arche_summary
                )
                
                create_social_link(
                    user_id, conversation_id,
                    entity1_type="npc", entity1_id=old_npc_id,
                    entity2_type="npc", entity2_id=npc_id,
                    link_type=rec_type
                )
                
                # Add to target NPC's relationships
                cursor.execute("""
                    SELECT relationships FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, old_npc_id))
                
                row = cursor.fetchone()
                target_relationships_json = row[0] if row else "[]"
                
                if isinstance(target_relationships_json, str):
                    try:
                        target_relationships = json.loads(target_relationships_json)
                    except:
                        target_relationships = []
                else:
                    target_relationships = target_relationships_json
                
                # Add the reciprocal relationship
                target_relationships.append({
                    "relationship_label": rec_type,
                    "entity_type": "npc", 
                    "entity_id": npc_id
                })
                
                # Update the target NPC's relationships
                cursor.execute("""
                    UPDATE NPCStats
                    SET relationships = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(target_relationships), user_id, conversation_id, old_npc_id))
                
                conn.commit()
                conn.close()
                
                # Generate shared memories for both NPCs
                try:
                    # Generate a relationship-specific memory for the new NPC
                    specific_memory = await self.generate_relationship_specific_memory(
                        user_id, conversation_id, 
                        npc_data, 
                        {"npc_id": old_npc_id, "npc_name": old_npc_name},
                        rel["relationship_label"]
                    )
                    
                    if specific_memory:
                        # Store in memory system
                        await memory_system.remember(
                            entity_type="npc",
                            entity_id=npc_id,
                            memory_text=specific_memory,
                            importance="medium",
                            emotional=True,
                            tags=["relationship", rel["relationship_label"]]
                        )
                        
                        # Create a reciprocal memory for the target NPC
                        reciprocal_memory = self.create_reciprocal_memory(
                            specific_memory, npc_name, old_npc_name, 
                            rel["relationship_label"], rec_type
                        )
                        
                        if reciprocal_memory:
                            await memory_system.remember(
                                entity_type="npc",
                                entity_id=old_npc_id,
                                memory_text=reciprocal_memory,
                                importance="medium",
                                emotional=True,
                                tags=["relationship", rec_type]
                            )
                except Exception as e:
                    logging.error(f"Error generating relationship memories: {e}")
        
        logging.info(f"Finished assigning relationships for NPC {npc_id}.")
    
    async def generate_relationship_specific_memory(self, user_id, conversation_id, npc_data, target_data, relationship_type):
        """
        Generate a memory specific to the relationship between two NPCs.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID  
            npc_data: Dict with data about the NPC
            target_data: Dict with data about the target NPC
            relationship_type: Type of relationship
            
        Returns:
            Memory string or None if error
        """
        try:
            npc_name = npc_data.get("npc_name", "Unknown")
            target_name = target_data.get("npc_name", "Unknown")
            
            # Get a random location for the memory
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT location_name FROM Locations 
                WHERE user_id=%s AND conversation_id=%s
                ORDER BY RANDOM() LIMIT 1
            """, (user_id, conversation_id))
            row = cursor.fetchone()
            location = row[0] if row else "a familiar place"
            conn.close()
            
            # Different memory templates based on relationship type
            memory_templates = {
                "mother": [
                    f"I remember when {target_name} was younger and had their first real failure at {location}. The disappointment was visible in their eyes, but I knew this was a teaching moment. 'Sometimes we need to fail to understand the value of success,' I told them, my hand firm but comforting on their shoulder. I could see the resistance at first—that stubborn set of the jaw they inherited from me—but slowly, understanding dawned. How they looked at me then, with a mixture of frustration and reluctant recognition, showed me they were growing in the way I had hoped."
                ],
                "friend": [
                    f"Last month, {target_name} and I spent an afternoon at {location}, supposedly just catching up, but I was carefully observing their reactions to certain topics. 'You always know exactly what to say,' they told me, not realizing how deliberately I choose my words. I smiled and deflected the compliment, but privately noted how easily they opened up about their insecurities. These casual get-togethers are perfect for gathering the small details that might be useful later."
                ],
                "colleague": [
                    f"During the project deadline at {location}, I noticed how {target_name} deferred to my judgment on the final presentation. 'What do you think we should emphasize?' they asked, though technically we were equals on the team. I suggested an approach that showcased my contributions while still acknowledging theirs. The subtle way they nodded, relieved to have direction, confirmed my growing influence in our professional relationship. I've been cultivating this dynamic carefully, one small interaction at a time."
                ],
                "rival": [
                    f"I encountered {target_name} at {location} during the quarterly review, and we engaged in our usual verbal chess match. 'Impressive results,' they said with that smile that never quite reaches their eyes. I returned an equally measured compliment, both of us aware of the real competition beneath our cordial exchange. What they don't realize is how much I study their strategies, cataloging weaknesses for future reference. There's a certain thrill in these encounters—measuring myself against someone who thinks they're my equal."
                ],
                "mentor": [
                    f"I've been guiding {target_name} through their professional development at {location}. Yesterday, I intentionally gave them a task just slightly beyond their current abilities. 'I know you can handle this,' I said, watching them try to hide their uncertainty. The subtle way they looked to me for reassurance was exactly what I wanted—establishing that I'm the source of both challenge and validation. After they completed it with my carefully timed guidance, their gratitude was palpable. These moments of manufactured growth strengthen our mentor-student dynamic."
                ]
            }
            
            # Get relevant template or use a generic one
            if relationship_type.lower() in memory_templates:
                return random.choice(memory_templates[relationship_type.lower()])
            else:
                return f"I remember an interaction with {target_name} at {location} that defined our {relationship_type} relationship. There was a moment of genuine connection mixed with the subtle power dynamic that's always been present between us. 'I understand you better than most people do,' I told them, which wasn't exactly what they were expecting to hear. The look of surprise followed by thoughtful consideration showed me they were reassessing our connection. These little moments of revelation always give me a particular satisfaction."
        
        except Exception as e:
            logging.error(f"Error generating relationship memory: {e}")
            return None
    
    def create_reciprocal_memory(self, original_memory, npc1_name, npc2_name, relationship_type, reciprocal_type):
        """
        Create a reciprocal memory from the perspective of the second NPC.
        
        Args:
            original_memory: The original memory text
            npc1_name: Name of the first NPC (original memory owner)
            npc2_name: Name of the second NPC (reciprocal memory owner)
            relationship_type: Original relationship type
            reciprocal_type: Reciprocal relationship type
            
        Returns:
            Reciprocal memory string or None if error
        """
        try:
            # Simple conversion for prototype
            # In a full implementation, this would use GPT to create a properly
            # transformed memory from the other perspective
            
            # Replace names
            memory = original_memory.replace(npc1_name, "THE_OTHER_PERSON")
            memory = memory.replace(npc2_name, "MYSELF")
            memory = memory.replace("THE_OTHER_PERSON", npc1_name)
            memory = memory.replace("MYSELF", "I")
            
            # Flip perspective words
            memory = memory.replace("I told them", f"{npc1_name} told me")
            memory = memory.replace("I noticed", f"{npc1_name} seemed to notice")
            memory = memory.replace("my hand", "her hand")
            memory = memory.replace("I suggested", f"{npc1_name} suggested")
            
            # Convert to first person
            memory = memory.replace(f"{npc2_name} was", "I was")
            memory = memory.replace(f"{npc2_name} had", "I had")
            memory = memory.replace(f"{npc2_name} and I", f"{npc1_name} and I")
            
            return memory
        except Exception as e:
            logging.error(f"Error creating reciprocal memory: {e}")
            return None
    
    # --- API Methods ---
    
    async def create_npc_api(self, request):
        """
        API endpoint to create a new NPC.
        
        Args:
            request: NPCCreationRequest object
            
        Returns:
            Dict with basic NPC info
        """
        try:
            # Extract request data
            user_id = request.user_id
            conversation_id = request.conversation_id
            environment_desc = request.environment_desc
            archetype_names = request.archetype_names
            specific_traits = request.specific_traits
            
            # Create context wrapper
            ctx = RunContextWrapper({
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            # Use existing function to create NPC
            result = await self.create_npc(
                ctx, 
                archetype_names=archetype_names, 
                physical_desc=None, 
                starting_traits=specific_traits.get("personality_traits") if specific_traits else None
            )
            
            # Extract basic info for immediate return
            npc_id = result.get("npc_id")
            npc_name = result.get("npc_name")
            physical_description = result.get("physical_description", "")
            
            # Get personality traits from result
            personality = result.get("personality", {})
            if isinstance(personality, dict):
                personality_traits = personality.get("personality_traits", [])
            else:
                personality_traits = []
            
            # Get archetypes from result
            archetypes = result.get("archetypes", {})
            if isinstance(archetypes, dict):
                archetype_names = archetypes.get("archetype_names", [])
            else:
                archetype_names = []
            
            return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "physical_description": physical_description[:100] + "..." if len(physical_description) > 100 else physical_description,
                "personality_traits": personality_traits,
                "archetypes": archetype_names,
                "current_location": result.get("current_location", ""),
                "message": "NPC creation in progress. Basic information is available now."
            }
        except Exception as e:
            logging.error(f"Error in create_npc_api: {e}")
            return {"error": str(e)}
    
    async def spawn_multiple_npcs_api(self, request):
        """
        API endpoint to spawn multiple NPCs.
        
        Args:
            request: MultipleNPCRequest object
            
        Returns:
            Dict with NPC IDs
        """
        try:
            # Extract request data
            user_id = request.user_id
            conversation_id = request.conversation_id
            count = request.count
            environment_desc = request.environment_desc
            
            # Create context wrapper
            ctx = RunContextWrapper({
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            # Use existing function to spawn NPCs
            npc_ids = await self.spawn_multiple_npcs(ctx, count)
            
            return {
                "npc_ids": npc_ids,
                "message": f"Successfully spawned {len(npc_ids)} NPCs."
            }
        except Exception as e:
            logging.error(f"Error in spawn_multiple_npcs_api: {e}")
            return {"error": str(e)}
    
    async def get_npc_api(self, npc_id, user_id, conversation_id):
        """
        API endpoint to get NPC details.
        
        Args:
            npc_id: NPC ID
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Dict with NPC details
        """
        try:
            # Create context wrapper
            ctx = RunContextWrapper({
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            # Use existing function to get NPC details
            result = await self.get_npc_details(ctx, npc_id=npc_id)
            
            if "error" in result:
                return result
            
            # Extract relevant info for API response
            return {
                "npc_id": result["npc_id"],
                "npc_name": result["npc_name"],
                "physical_description": result["physical_description"],
                "personality_traits": result["personality_traits"],
                "archetypes": [a.get("name", "") for a in result["archetypes"]],
                "current_location": result["current_location"]
            }
        except Exception as e:
            logging.error(f"Error in get_npc_api: {e}")
            return {"error": str(e)}
