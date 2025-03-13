# logic/universal_updater_agent.py

import json
import logging
import asyncio
import os
import asyncpg
from datetime import datetime

from agents import Agent, Runner, function_tool, ModelSettings
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union

# DB connection
DB_DSN = os.getenv("DB_DSN")

# Define Pydantic models for structured data
class NPCCreation(BaseModel):
    npc_name: str
    introduced: bool = False
    sex: str = "female"
    dominance: Optional[int] = None
    cruelty: Optional[int] = None
    closeness: Optional[int] = None
    trust: Optional[int] = None
    respect: Optional[int] = None
    intensity: Optional[int] = None
    archetypes: List[Dict[str, Any]] = Field(default_factory=list)
    archetype_summary: Optional[str] = None
    archetype_extras_summary: Optional[str] = None
    physical_description: Optional[str] = None
    hobbies: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    schedule: Dict[str, Any] = Field(default_factory=dict)
    memory: Union[List[str], str, None] = None
    monica_level: Optional[int] = None
    age: Optional[int] = None
    birthdate: Optional[str] = None

class NPCUpdate(BaseModel):
    npc_id: int
    npc_name: Optional[str] = None
    introduced: Optional[bool] = None
    archetype_summary: Optional[str] = None
    archetype_extras_summary: Optional[str] = None
    physical_description: Optional[str] = None
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
    sex: Optional[str] = None
    memory: Optional[Union[List[str], str]] = None
    schedule: Optional[Dict[str, Any]] = None
    schedule_updates: Optional[Dict[str, Any]] = None
    affiliations: Optional[List[str]] = None
    current_location: Optional[str] = None

class NPCIntroduction(BaseModel):
    npc_id: int

class PlayerStats(BaseModel):
    corruption: Optional[int] = None
    confidence: Optional[int] = None
    willpower: Optional[int] = None
    obedience: Optional[int] = None
    dependency: Optional[int] = None
    lust: Optional[int] = None
    mental_resilience: Optional[int] = None
    physical_endurance: Optional[int] = None

class CharacterStatUpdates(BaseModel):
    player_name: str = "Chase"
    stats: PlayerStats

class RelationshipUpdate(BaseModel):
    npc_id: int
    affiliations: List[str]

class SocialLink(BaseModel):
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    link_type: Optional[str] = None
    level_change: Optional[int] = None
    new_event: Optional[str] = None
    group_context: Optional[str] = None

class Location(BaseModel):
    location_name: str
    description: Optional[str] = None
    open_hours: List[str] = Field(default_factory=list)

class Event(BaseModel):
    event_name: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None
    npc_id: Optional[int] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    time_of_day: Optional[str] = None
    override_location: Optional[str] = None
    fantasy_level: str = "realistic"

class Quest(BaseModel):
    quest_id: Optional[int] = None
    quest_name: Optional[str] = None
    status: Optional[str] = None
    progress_detail: Optional[str] = None
    quest_giver: Optional[str] = None
    reward: Optional[str] = None

class InventoryItem(BaseModel):
    item_name: str
    item_description: Optional[str] = None
    item_effect: Optional[str] = None
    category: Optional[str] = None

class InventoryUpdates(BaseModel):
    player_name: str = "Chase"
    added_items: List[Union[str, InventoryItem]] = Field(default_factory=list)
    removed_items: List[Union[str, Dict[str, str]]] = Field(default_factory=list)

class Perk(BaseModel):
    player_name: str = "Chase"
    perk_name: str
    perk_description: Optional[str] = None
    perk_effect: Optional[str] = None

class Activity(BaseModel):
    activity_name: str
    purpose: Optional[Dict[str, Any]] = None
    stat_integration: Optional[Dict[str, Any]] = None
    intensity_tier: Optional[int] = None
    setting_variant: Optional[str] = None

class JournalEntry(BaseModel):
    entry_type: str
    entry_text: str
    fantasy_flag: bool = False
    intensity_level: Optional[int] = None

class ImageGeneration(BaseModel):
    generate: bool = False
    priority: str = "low"
    focus: str = "balanced"
    framing: str = "medium_shot"
    reason: Optional[str] = None

class UniversalUpdateInput(BaseModel):
    user_id: int
    conversation_id: int
    narrative: str
    roleplay_updates: Dict[str, Any] = Field(default_factory=dict)
    ChaseSchedule: Optional[Dict[str, Any]] = None
    MainQuest: Optional[str] = None
    PlayerRole: Optional[str] = None
    npc_creations: List[NPCCreation] = Field(default_factory=list)
    npc_updates: List[NPCUpdate] = Field(default_factory=list)
    character_stat_updates: Optional[CharacterStatUpdates] = None
    relationship_updates: List[RelationshipUpdate] = Field(default_factory=list)
    npc_introductions: List[NPCIntroduction] = Field(default_factory=list)
    location_creations: List[Location] = Field(default_factory=list)
    event_list_updates: List[Event] = Field(default_factory=list)
    inventory_updates: Optional[InventoryUpdates] = None
    quest_updates: List[Quest] = Field(default_factory=list)
    social_links: List[SocialLink] = Field(default_factory=list)
    perk_unlocks: List[Perk] = Field(default_factory=list)
    activity_updates: List[Activity] = Field(default_factory=list)
    journal_updates: List[JournalEntry] = Field(default_factory=list)
    image_generation: Optional[ImageGeneration] = None

class UniversalUpdaterAgent:
    """Agent for processing and applying updates to the game state"""
    
    def __init__(self):
        # Primary agent for universal updates
        self.agent = Agent(
            name="UniversalUpdater",
            instructions="""
            You are the UniversalUpdater agent for a roleplaying game with subtle femdom themes.
            Your job is to analyze narrative text and extract appropriate game state updates.
            
            Based on the provided narrative and context, you need to:
            1. Identify changes to NPCs (creation, updates, introductions)
            2. Track changes to player stats
            3. Detect changes to relationships and social dynamics
            4. Register new locations or events
            5. Update inventory, quests, activities, and journal entries
            6. Determine if image generation is warranted
            
            Your output must be a well-structured JSON object conforming to the expected schema.
            Focus on extracting concrete changes from the narrative rather than inferring too much.
            Be subtle in handling femdom themes - identify power dynamics but keep them understated.
            """,
            output_type=UniversalUpdateInput,
            tools=[
                function_tool(self._normalize_json),
                function_tool(self._check_npc_exists)
            ],
            model_settings=ModelSettings(temperature=0.2)  # Lower temperature for precision
        )
        
        # Helper agent for extracting stats and state changes
        self.extractor_agent = Agent(
            name="StateExtractor",
            instructions="""
            You identify and extract state changes from narrative text in a roleplaying game.
            Carefully analyze the provided text to detect explicit and implied changes to:
            
            - NPC stats, location, or status
            - Player stats and resources
            - Relationships between characters
            - New items, locations, or events
            - Tone, atmosphere, and environment changes
            
            Be precise and avoid over-interpretation. Only extract changes that are clearly
            indicated in the text or strongly implied.
            """,
            tools=[
                function_tool(self._extract_player_stats),
                function_tool(self._extract_npc_changes),
                function_tool(self._extract_relationship_changes)
            ],
            model_settings=ModelSettings(temperature=0.1)  # Very low temperature for accuracy
        )
    
    async def _normalize_json(self, ctx, json_str: str) -> Dict[str, Any]:
        """
        Normalize JSON string, fixing common errors:
        - Replace curly quotes with straight quotes
        - Add missing quotes around keys
        - Fix trailing commas
        
        Args:
            json_str: A potentially malformed JSON string
            
        Returns:
            Parsed JSON object as a dictionary
        """
        try:
            # Try to parse as-is first
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Simple normalization - replace curly quotes
            normalized = json_str.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
            
            try:
                return json.loads(normalized)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to normalize JSON: {e}")
                # Return a simple dict with error info
                return {"error": "Failed to parse JSON", "message": str(e), "original": json_str}
    
    async def _check_npc_exists(self, ctx, npc_id: int) -> bool:
        """
        Check if an NPC with the given ID exists in the database.
        
        Args:
            npc_id: NPC ID to check
            
        Returns:
            Boolean indicating if the NPC exists
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                row = await conn.fetchrow("""
                    SELECT npc_id FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, user_id, conversation_id)
                
                return row is not None
            finally:
                await conn.close()
        except Exception as e:
            logging.error(f"Error checking if NPC exists: {e}")
            return False
    
    async def _extract_player_stats(self, ctx, narrative: str) -> Dict[str, Any]:
        """
        Extract player stat changes from narrative text.
        
        Args:
            narrative: The narrative text to analyze
            
        Returns:
            Dictionary of player stat changes
        """
        # The stats to look for
        stats = ["corruption", "confidence", "willpower", "obedience", 
                "dependency", "lust", "mental_resilience", "physical_endurance"]
        
        changes = {}
        
        # Extract explicit mentions of stats increasing or decreasing
        for stat in stats:
            # Look for patterns like "confidence increased", "willpower drops", etc.
            if f"{stat} increase" in narrative.lower() or f"{stat} rose" in narrative.lower() or f"{stat} grows" in narrative.lower():
                changes[stat] = "+5"  # Default modest increase
            elif f"{stat} decrease" in narrative.lower() or f"{stat} drop" in narrative.lower() or f"{stat} falls" in narrative.lower():
                changes[stat] = "-5"  # Default modest decrease
        
        return {"player_name": "Chase", "stats": changes}
    
    async def _extract_npc_changes(self, ctx, narrative: str) -> List[Dict[str, Any]]:
        """
        Extract NPC changes from narrative text.
        
        Args:
            narrative: The narrative text to analyze
            
        Returns:
            List of NPC updates
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get existing NPCs
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            npcs = {row["npc_name"]: {"npc_id": row["npc_id"], "current_location": row["current_location"]} 
                   for row in rows}
        finally:
            await conn.close()
        
        updates = []
        
        # Check each NPC for mentions and changes
        for npc_name, npc_data in npcs.items():
            # Skip NPCs not mentioned in the narrative
            if npc_name not in narrative:
                continue
            
            npc_update = {"npc_id": npc_data["npc_id"]}
            
            # Check for location changes
            location_indicators = ["moved to", "arrived at", "entered", "stood in", "was at"]
            for indicator in location_indicators:
                if f"{npc_name} {indicator}" in narrative:
                    # Extract location after the indicator
                    idx = narrative.find(f"{npc_name} {indicator}") + len(f"{npc_name} {indicator}")
                    end_idx = narrative.find(".", idx)
                    if end_idx != -1:
                        location_text = narrative[idx:end_idx].strip()
                        # Extract just the location name - use a simple approach
                        for word in ["the", "a", "an"]:
                            if location_text.startswith(word + " "):
                                location_text = location_text[len(word) + 1:]
                        npc_update["current_location"] = location_text.strip()
                        break
            
            # Only add the update if we found changes
            if len(npc_update) > 1:  # More than just npc_id
                updates.append(npc_update)
        
        return updates
    
    async def _extract_relationship_changes(self, ctx, narrative: str) -> List[Dict[str, Any]]:
        """
        Extract relationship changes from narrative text.
        
        Args:
            narrative: The narrative text to analyze
            
        Returns:
            List of relationship changes
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get existing NPCs
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            npcs = {row["npc_name"]: row["npc_id"] for row in rows}
        finally:
            await conn.close()
        
        changes = []
        
        # Check for relationship indicators between player and NPCs
        for npc_name, npc_id in npcs.items():
            # Skip NPCs not mentioned in the narrative
            if npc_name not in narrative:
                continue
            
            # Look for relationship indicators
            positive_indicators = ["smiled at you", "touched your", "praised you", "thanked you"]
            negative_indicators = ["frowned at you", "scolded you", "ignored you", "dismissed you"]
            
            # Check for specific relationship changes
            relationship_change = None
            
            for indicator in positive_indicators:
                if f"{npc_name} {indicator}" in narrative:
                    relationship_change = {
                        "entity1_type": "player",
                        "entity1_id": 0,  # Player ID
                        "entity2_type": "npc",
                        "entity2_id": npc_id,
                        "level_change": 5,  # Modest increase
                        "new_event": f"{npc_name} {indicator}"
                    }
                    break
            
            if not relationship_change:
                for indicator in negative_indicators:
                    if f"{npc_name} {indicator}" in narrative:
                        relationship_change = {
                            "entity1_type": "player",
                            "entity1_id": 0,  # Player ID
                            "entity2_type": "npc",
                            "entity2_id": npc_id,
                            "level_change": -5,  # Modest decrease
                            "new_event": f"{npc_name} {indicator}"
                        }
                        break
            
            if relationship_change:
                changes.append(relationship_change)
        
        return changes
    
    async def process_universal_update(self, user_id: int, conversation_id: int, narrative: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a universal update based on narrative text.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            narrative: Narrative text to process
            context: Additional context (optional)
            
        Returns:
            Dictionary with update results
        """
        # Set up context
        ctx_data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "db_dsn": DB_DSN
        }
        if context:
            ctx_data.update(context)
        
        # Create prompt for the agent
        prompt = f"""
        Analyze the following narrative text and extract appropriate game state updates.
        
        Narrative:
        {narrative}
        
        Based on this narrative, identify:
        1. NPC creations or updates (changes in location, stats, etc.)
        2. Player stat changes (increases or decreases in corruption, confidence, etc.)
        3. Relationship changes between characters
        4. New locations, events, items, or quests
        5. Journal entries or activity updates
        6. Whether an image should be generated for this scene
        
        Provide a structured output conforming to the UniversalUpdateInput schema.
        Include the narrative text in the 'narrative' field and fill in other fields as appropriate.
        Only include fields where you have identified changes or updates.
        """
        
        # Run the agent to extract updates
        result = await Runner.run(
            self.agent,
            prompt,
            context=ctx_data
        )
        
        update_data = result.final_output
        
        # Add user_id and conversation_id if not already present
        if not hasattr(update_data, 'user_id') or not update_data.user_id:
            update_data.user_id = user_id
        if not hasattr(update_data, 'conversation_id') or not update_data.conversation_id:
            update_data.conversation_id = conversation_id
        
        # Apply the updates
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            # Convert to dict for apply_universal_updates_async
            # Import locally to avoid circular import
            from logic.universal_updater import apply_universal_updates_async
            
            update_result = await apply_universal_updates_async(
                user_id,
                conversation_id,
                update_data.dict(),
                conn
            )
            
            return update_result
        except Exception as e:
            logging.error(f"Error applying universal updates: {e}")
            return {"error": str(e)}
        finally:
            await conn.close()

# Helper function to apply updates
async def apply_updates(user_id, conversation_id, data):
    """
    Helper function to apply universal updates.
    This is a wrapper around the main agent functionality.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        data: Update data dictionary or narrative string
        
    Returns:
        Dictionary with update results
    """
    updater = UniversalUpdaterAgent()
    
    # If data is a string, treat it as narrative
    if isinstance(data, str):
        narrative = data
        return await updater.process_universal_update(user_id, conversation_id, narrative)
    
    # If data is a dict, extract narrative and other fields
    elif isinstance(data, dict):
        narrative = data.get("narrative", "")
        return await updater.process_universal_update(user_id, conversation_id, narrative, data)
    
    # Otherwise, raise an error
    else:
        raise ValueError("Data must be a string or dictionary")
