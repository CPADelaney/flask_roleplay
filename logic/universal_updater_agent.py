# logic/universal_updater_sdk.py

"""
Universal Updater SDK using OpenAI's Agents SDK with Nyx Governance integration.

This module is responsible for analyzing narrative text and extracting appropriate 
game state updates. It replaces the previous class-based approach in universal_updater_agent.py
with a more agentic system that integrates with Nyx governance.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# OpenAI Agents SDK imports
from agents import (
    Agent, 
    ModelSettings, 
    Runner, 
    function_tool, 
    RunContextWrapper,
    GuardrailFunctionOutput,
    InputGuardrail,
    trace,
    handoff
)
from pydantic import BaseModel, Field

# DB connection
from db.connection import get_db_connection_context
import asyncpg

# Nyx governance integration
from nyx.nyx_governance import (
    NyxUnifiedGovernor,
    AgentType,
    DirectiveType,
    DirectivePriority
)
from nyx.integrate import get_central_governance

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Pydantic Models for Structured Outputs (migrated from universal_updater_agent.py)
# -------------------------------------------------------------------------------

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

class ContentSafety(BaseModel):
    """Output for content moderation guardrail"""
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

# -------------------------------------------------------------------------------
# Agent Context
# -------------------------------------------------------------------------------

class UniversalUpdaterContext:
    """Context object for universal updater agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        
    async def initialize(self):
        """Initialize context with governance integration"""
        self.governor = await get_central_governance(self.user_id, self.conversation_id)

# -------------------------------------------------------------------------------
# Function Tools
# -------------------------------------------------------------------------------

@function_tool
async def normalize_json(ctx, json_str: str) -> Dict[str, Any]:
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

@function_tool
async def check_npc_exists(ctx, npc_id: int) -> bool:
    """
    Check if an NPC with the given ID exists in the database.
    
    Args:
        npc_id: NPC ID to check
        
    Returns:
        Boolean indicating if the NPC exists
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Check permission with governance system
    governor = ctx.context.governor
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="check_npc_exists",
        action_details={"npc_id": npc_id}
    )
    
    if not permission["approved"]:
        return False
    
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_id FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, user_id, conversation_id)
            
            exists = row is not None
            
            # Report action to governance
            await governor.process_agent_action_report(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="universal_updater",
                action={"type": "check_npc_exists", "npc_id": npc_id},
                result={"exists": exists}
            )
            
            return exists
    except Exception as e:
        logging.error(f"Error checking if NPC exists: {e}")
        return False

@function_tool
async def extract_player_stats(ctx, narrative: str) -> Dict[str, Any]:
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
    
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="extract_player_stats",
        action_details={"narrative_length": len(narrative)}
    )
    
    if not permission["approved"]:
        return {"player_name": "Chase", "stats": {}}
    
    changes = {}
    
    # Extract explicit mentions of stats increasing or decreasing
    for stat in stats:
        # Look for patterns like "confidence increased", "willpower drops", etc.
        if f"{stat} increase" in narrative.lower() or f"{stat} rose" in narrative.lower() or f"{stat} grows" in narrative.lower():
            changes[stat] = 5  # Default modest increase
        elif f"{stat} decrease" in narrative.lower() or f"{stat} drop" in narrative.lower() or f"{stat} falls" in narrative.lower():
            changes[stat] = -5  # Default modest decrease
    
    # Report action to governance
    await governor.process_agent_action_report(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action={"type": "extract_player_stats"},
        result={"stats_changed": len(changes)}
    )
    
    return {"player_name": "Chase", "stats": changes}

@function_tool
async def extract_npc_changes(ctx, narrative: str) -> List[Dict[str, Any]]:
    """
    Extract NPC changes from narrative text.
    
    Args:
        narrative: The narrative text to analyze
        
    Returns:
        List of NPC updates
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="extract_npc_changes",
        action_details={"narrative_length": len(narrative)}
    )
    
    if not permission["approved"]:
        return []
    
    # Get existing NPCs
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            npcs = {row["npc_name"]: {"npc_id": row["npc_id"], "current_location": row["current_location"]} 
                   for row in rows}
        
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
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="universal_updater",
            action={"type": "extract_npc_changes"},
            result={"npc_updates": len(updates)}
        )
        
        return updates
    except Exception as e:
        logging.error(f"Error extracting NPC changes: {e}")
        return []

@function_tool
async def extract_relationship_changes(ctx, narrative: str) -> List[Dict[str, Any]]:
    """
    Extract relationship changes from narrative text.
    
    Args:
        narrative: The narrative text to analyze
        
    Returns:
        List of relationship changes
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="extract_relationship_changes",
        action_details={"narrative_length": len(narrative)}
    )
    
    if not permission["approved"]:
        return []
    
    # Get existing NPCs
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            npcs = {row["npc_name"]: row["npc_id"] for row in rows}
        
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
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="universal_updater",
            action={"type": "extract_relationship_changes"},
            result={"relationship_changes": len(changes)}
        )
        
        return changes
    except Exception as e:
        logging.error(f"Error extracting relationship changes: {e}")
        return []

# Add this function to universal_updater_agent.py, before the apply_universal_updates function

async def apply_universal_updates_async(
    user_id: int,
    conversation_id: int,
    updates: Dict[str, Any],
    conn: asyncpg.Connection
) -> Dict[str, Any]:
    """
    Apply universal updates to the database.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        updates: Dictionary containing all the updates to apply
        conn: Database connection
        
    Returns:
        Dictionary with update results
    """
    try:
        # Initialize counters and results
        results = {
            "success": True,
            "updates_applied": 0,
            "details": {}
        }
        
        # Process NPC creations
        if "npc_creations" in updates and updates["npc_creations"]:
            npc_creation_count = await process_npc_creations(
                user_id, conversation_id, updates["npc_creations"], conn
            )
            results["details"]["npc_creations"] = npc_creation_count
            results["updates_applied"] += npc_creation_count
        
        # Process NPC updates
        if "npc_updates" in updates and updates["npc_updates"]:
            npc_update_count = await process_npc_updates(
                user_id, conversation_id, updates["npc_updates"], conn
            )
            results["details"]["npc_updates"] = npc_update_count
            results["updates_applied"] += npc_update_count
        
        # Process character stat updates
        if "character_stat_updates" in updates and updates["character_stat_updates"]:
            stat_update_count = await process_character_stats(
                user_id, conversation_id, updates["character_stat_updates"], conn
            )
            results["details"]["stat_updates"] = stat_update_count
            results["updates_applied"] += stat_update_count
        
        # Process social links
        if "social_links" in updates and updates["social_links"]:
            social_link_count = await process_social_links(
                user_id, conversation_id, updates["social_links"], conn
            )
            results["details"]["social_links"] = social_link_count
            results["updates_applied"] += social_link_count
        
        # Process roleplay updates
        if "roleplay_updates" in updates and updates["roleplay_updates"]:
            roleplay_update_count = await process_roleplay_updates(
                user_id, conversation_id, updates["roleplay_updates"], conn
            )
            results["details"]["roleplay_updates"] = roleplay_update_count
            results["updates_applied"] += roleplay_update_count
        
        # Return results
        return results
    except Exception as e:
        logger.error(f"Error applying universal updates: {e}")
        return {"success": False, "error": str(e)}

# Helper functions for specific update types

async def process_npc_creations(
    user_id: int,
    conversation_id: int,
    npc_creations: List[Dict[str, Any]],
    conn: asyncpg.Connection
) -> int:
    """Process NPC creations and return count of NPCs created."""
    count = 0
    for npc in npc_creations:
        # Convert any list/dict fields to JSON strings
        for field in ["archetypes", "hobbies", "personality_traits", "likes", 
                      "dislikes", "affiliations", "schedule", "memory"]:
            if field in npc and npc[field] is not None:
                if isinstance(npc[field], (list, dict)):
                    npc[field] = json.dumps(npc[field])
        
        # Insert NPC record
        await conn.execute("""
            INSERT INTO NPCStats (
                user_id, conversation_id, npc_name, introduced, sex,
                dominance, cruelty, closeness, trust, respect, intensity,
                archetypes, archetype_summary, archetype_extras_summary,
                physical_description, hobbies, personality_traits,
                likes, dislikes, affiliations, schedule, memory,
                monica_level, age, birthdate
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
            )
        """, user_id, conversation_id, npc["npc_name"], npc.get("introduced", False),
        npc.get("sex", "female"), npc.get("dominance"), npc.get("cruelty"),
        npc.get("closeness"), npc.get("trust"), npc.get("respect"),
        npc.get("intensity"), npc.get("archetypes"), npc.get("archetype_summary"),
        npc.get("archetype_extras_summary"), npc.get("physical_description"),
        npc.get("hobbies"), npc.get("personality_traits"), npc.get("likes"),
        npc.get("dislikes"), npc.get("affiliations"), npc.get("schedule"),
        npc.get("memory"), npc.get("monica_level"), npc.get("age"),
        npc.get("birthdate"))
        
        count += 1
    
    return count

async def process_npc_updates(
    user_id: int,
    conversation_id: int,
    npc_updates: List[Dict[str, Any]],
    conn: asyncpg.Connection
) -> int:
    """Process NPC updates and return count of NPCs updated."""
    count = 0
    for npc in npc_updates:
        # Skip if no npc_id
        if "npc_id" not in npc:
            continue
        
        # Build update statement dynamically based on provided fields
        fields = []
        values = [user_id, conversation_id, npc["npc_id"]]
        param_index = 4  # Starting after the first 3 parameters
        
        update_fields = [
            "npc_name", "introduced", "archetype_summary", "archetype_extras_summary",
            "physical_description", "dominance", "cruelty", "closeness", "trust",
            "respect", "intensity", "sex", "current_location"
        ]
        
        for field in update_fields:
            if field in npc and npc[field] is not None:
                fields.append(f"{field} = ${param_index}")
                values.append(npc[field])
                param_index += 1
        
        # Handle JSON fields separately
        json_fields = ["hobbies", "personality_traits", "likes", "dislikes", 
                       "affiliations", "memory", "schedule"]
        
        for field in json_fields:
            if field in npc and npc[field] is not None:
                fields.append(f"{field} = ${param_index}")
                values.append(json.dumps(npc[field]) if isinstance(npc[field], (list, dict)) else npc[field])
                param_index += 1
        
        # Skip if no fields to update
        if not fields:
            continue
        
        # Execute update
        query = f"""
            UPDATE NPCStats SET {', '.join(fields)}
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """
        await conn.execute(query, *values)
        count += 1
    
    return count

async def process_character_stats(
    user_id: int,
    conversation_id: int,
    stat_updates: Dict[str, Any],
    conn: asyncpg.Connection
) -> int:
    """Process character stat updates and return count of updates."""
    if not stat_updates or "stats" not in stat_updates:
        return 0
    
    player_name = stat_updates.get("player_name", "Chase")
    stats = stat_updates["stats"]
    
    # Skip if no stats to update
    if not stats:
        return 0
    
    # First, check if player exists
    player_exists = await conn.fetchval("""
        SELECT COUNT(*) FROM PlayerStats
        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
    """, user_id, conversation_id, player_name)
    
    if not player_exists:
        # Insert new player stats record with defaults
        await conn.execute("""
            INSERT INTO PlayerStats (
                user_id, conversation_id, player_name,
                corruption, confidence, willpower, obedience,
                dependency, lust, mental_resilience, physical_endurance
            ) VALUES (
                $1, $2, $3, 0, 0, 0, 0, 0, 0, 0, 0
            )
        """, user_id, conversation_id, player_name)
    
    # Count updates actually made
    update_count = 0
    
    # Update each stat
    for stat, value in stats.items():
        if value is not None:
            # Get current value
            current_value = await conn.fetchval(f"""
                SELECT {stat} FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if current_value is not None:
                # Calculate new value (ensuring it stays within 0-100 range)
                new_value = max(0, min(100, current_value + value))
                
                # Update the stat
                await conn.execute(f"""
                    UPDATE PlayerStats SET {stat} = $4
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                """, user_id, conversation_id, player_name, new_value)
                
                # Log stat change in history
                await conn.execute("""
                    INSERT INTO StatsHistory (
                        user_id, conversation_id, player_name, stat_name,
                        old_value, new_value, cause
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, user_id, conversation_id, player_name, stat,
                current_value, new_value, "Narrative update")
                
                update_count += 1
    
    return update_count

async def process_social_links(
    user_id: int,
    conversation_id: int,
    social_links: List[Dict[str, Any]],
    conn: asyncpg.Connection
) -> int:
    """Process social link updates and return count of links updated."""
    count = 0
    for link in social_links:
        # Check if link exists
        existing_link = await conn.fetchrow("""
            SELECT link_id, link_level FROM SocialLinks
            WHERE user_id = $1 AND conversation_id = $2 
            AND entity1_type = $3 AND entity1_id = $4
            AND entity2_type = $5 AND entity2_id = $6
        """, user_id, conversation_id, link["entity1_type"], link["entity1_id"],
        link["entity2_type"], link["entity2_id"])
        
        if existing_link:
            # Update existing link
            if "level_change" in link and link["level_change"] is not None:
                current_level = existing_link["link_level"] or 0
                new_level = current_level + link["level_change"]
                
                await conn.execute("""
                    UPDATE SocialLinks SET link_level = $7
                    WHERE link_id = $8
                """, new_level, existing_link["link_id"])
            
            # Add new event if provided
            if "new_event" in link and link["new_event"]:
                # Get current history
                current_history = await conn.fetchval("""
                    SELECT link_history FROM SocialLinks
                    WHERE link_id = $1
                """, existing_link["link_id"])
                
                # Append new event
                history = json.loads(current_history or '[]')
                history.append({
                    "event": link["new_event"],
                    "timestamp": str(conn.get_server_timestamp())
                })
                
                await conn.execute("""
                    UPDATE SocialLinks SET link_history = $1
                    WHERE link_id = $2
                """, json.dumps(history), existing_link["link_id"])
            
            count += 1
        else:
            # Create new link
            initial_level = link.get("level_change", 0)
            history = []
            
            if "new_event" in link and link["new_event"]:
                history.append({
                    "event": link["new_event"],
                    "timestamp": str(conn.get_server_timestamp())
                })
            
            await conn.execute("""
                INSERT INTO SocialLinks (
                    user_id, conversation_id, entity1_type, entity1_id,
                    entity2_type, entity2_id, link_type, link_level,
                    link_history, group_interaction
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, user_id, conversation_id, link["entity1_type"], link["entity1_id"],
            link["entity2_type"], link["entity2_id"], link.get("link_type"),
            initial_level, json.dumps(history), link.get("group_context"))
            
            count += 1
    
    return count

async def process_roleplay_updates(
    user_id: int,
    conversation_id: int,
    roleplay_updates: Dict[str, Any],
    conn: asyncpg.Connection
) -> int:
    """Process roleplay updates and return count of updates."""
    count = 0
    for key, value in roleplay_updates.items():
        if value is not None:
            # Check if key exists
            existing = await conn.fetchval("""
                SELECT COUNT(*) FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = $3
            """, user_id, conversation_id, key)
            
            if existing:
                # Update existing key
                await conn.execute("""
                    UPDATE CurrentRoleplay SET value = $4
                    WHERE user_id = $1 AND conversation_id = $2 AND key = $3
                """, user_id, conversation_id, key, str(value))
            else:
                # Insert new key
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, $3, $4)
                """, user_id, conversation_id, key, str(value))
            
            count += 1
    
    return count

@function_tool
async def apply_universal_updates(ctx, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply universal updates to the database.
    
    Args:
        updates: Dictionary containing all the updates to apply
        
    Returns:
        Dictionary with update results
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="apply_updates",
        action_details={"update_count": sum(len(updates.get(k, [])) for k in updates if isinstance(updates.get(k), list))}
    )
    
    if not permission["approved"]:
        return {"success": False, "reason": permission["reasoning"]}
    
    try:
        # Ensure user_id and conversation_id are set in updates
        updates["user_id"] = user_id
        updates["conversation_id"] = conversation_id
        
        async with get_db_connection_context() as conn:
            # Apply updates using the function defined in this same file
            result = await apply_universal_updates_async(
                user_id,
                conversation_id,
                updates,
                conn
            )
            
            # Report action to governance
            await governor.process_agent_action_report(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="universal_updater",
                action={"type": "apply_updates"},
                result={"success": True, "updates_applied": result.get("updates_applied", 0)}
            )
            
            return result
    except Exception as e:
        logging.error(f"Error applying universal updates: {e}")
        return {"success": False, "error": str(e)}



# -------------------------------------------------------------------------------
# Guardrail Functions
# -------------------------------------------------------------------------------

async def content_safety_guardrail(ctx, agent, input_data):
    """Input guardrail for content moderation"""
    content_moderator = Agent(
        name="Content Moderator",
        instructions="""
        You check if content is appropriate for a femdom roleplay game. 
        Allow adult themes within the context of a consensual femdom relationship,
        but flag anything that might be genuinely harmful or problematic.
        """,
        output_type=ContentSafety
    )
    
    result = await Runner.run(content_moderator, input_data, context=ctx.context)
    final_output = result.final_output_as(ContentSafety)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# -------------------------------------------------------------------------------
# Agent Definitions
# -------------------------------------------------------------------------------

# Extraction agent for initial analysis
extraction_agent = Agent[UniversalUpdaterContext](
    name="StateExtractor",
    instructions="""
    You identify and extract state changes from narrative text in a femdom roleplaying game.
    
    Your role is to:
    1. Analyze narrative text to detect explicit and implied changes
    2. Extract changes to NPC stats, locations, or status
    3. Identify player stat changes and relationships
    4. Note new items, locations, or events mentioned
    5. Detect tone, atmosphere, and environment changes
    
    Be precise and avoid over-interpretation. Only extract changes that are clearly
    indicated in the text or strongly implied.
    """,
    tools=[
        extract_player_stats,
        extract_npc_changes,
        extract_relationship_changes
    ],
    model_settings=ModelSettings(temperature=0.1)  # Low temperature for accuracy
)

# Main Universal Updater Agent
universal_updater_agent = Agent[UniversalUpdaterContext](
    name="UniversalUpdater",
    instructions="""
    You analyze narrative text and extract appropriate game state updates for a femdom roleplaying game.
    
    Your role is to:
    1. Analyze narrative text for important state changes
    2. Extract NPC creations, updates, and introductions
    3. Track player stat changes and social relationship changes
    4. Identify new locations, events, quests, and inventory items
    5. Organize all changes into a structured format
    
    Focus on extracting concrete changes rather than inferring too much.
    Be subtle in handling femdom themes - identify power dynamics but keep them understated.
    """,
    tools=[
        normalize_json,
        check_npc_exists,
        extract_player_stats,
        extract_npc_changes,
        extract_relationship_changes,
        apply_universal_updates
    ],
    handoffs=[
        handoff(extraction_agent, tool_name_override="extract_state_changes")
    ],
    output_type=UniversalUpdateInput,
    input_guardrails=[
        InputGuardrail(guardrail_function=content_safety_guardrail),
    ],
    model_settings=ModelSettings(temperature=0.2)  # Low temperature for precision
)

# -------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------

async def process_universal_update(
    user_id: int, 
    conversation_id: int, 
    narrative: str, 
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a universal update based on narrative text with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        narrative: Narrative text to process
        context: Additional context (optional)
        
    Returns:
        Dictionary with update results
    """
    # Create and initialize the updater context
    updater_context = UniversalUpdaterContext(user_id, conversation_id)
    await updater_context.initialize()
    
    # Set up context data
    ctx_data = context or {}
    
    # Create trace for monitoring
    with trace(
        workflow_name="Universal Update",
        trace_id=f"universal-update-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
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
            universal_updater_agent,
            prompt,
            context=updater_context
        )
        
        # Get the output
        update_data = result.final_output
        
        # Apply the updates
        if update_data:
            update_result = await apply_universal_updates(RunContextWrapper(updater_context), update_data.dict())
            return update_result
        else:
            return {"success": False, "error": "No updates extracted"}

async def register_with_governance(user_id: int, conversation_id: int):
    """
    Register universal updater agents with Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    # Get governor
    governor = await get_central_governance(user_id, conversation_id)
    
    # Register main agent
    await governor.register_agent(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_instance=universal_updater_agent,
        agent_id="universal_updater"
    )
    
    # Issue directive for universal updating
    await governor.issue_directive(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Process narrative updates and extract game state changes",
            "scope": "game"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60  # 24 hours
    )
    
    logging.info("Universal Updater registered with Nyx governance")

async def initialize_universal_updater(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Initialize the Universal Updater system and register with governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary containing initialized context and status
    """
    try:
        # Create and initialize the updater context
        updater_context = UniversalUpdaterContext(user_id, conversation_id)
        await updater_context.initialize()
        
        # Register with governance system
        await register_with_governance(user_id, conversation_id)
        
        logging.info(f"Universal Updater initialized for user {user_id}, conversation {conversation_id}")
        
        return {
            "context": updater_context,
            "agents": {
                "extraction_agent": extraction_agent,
                "universal_updater_agent": universal_updater_agent
            },
            "status": "initialized"
        }
    except Exception as e:
        logging.error(f"Error initializing Universal Updater: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "status": "failed"
        }
