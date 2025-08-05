# logic/universal_updater_sdk.py

"""
Universal Updater SDK using OpenAI's Agents SDK with Nyx Governance integration.

This module is responsible for analyzing narrative text and extracting appropriate 
game state updates. It replaces the previous class-based approach in universal_updater_agent.py
with a more agentic system that integrates with Nyx governance.

REFACTORED: All direct database writes now go through canon or LoreSystem
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
from pydantic import BaseModel, Field, ConfigDict

# DB connection
from db.connection import get_db_connection_context
import asyncpg

# Import canon and lore system
from lore.core import canon
from lore.core.lore_system import LoreSystem

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

class StrictBaseModel(BaseModel):
    """Base class enforcing a strict schema for OpenAI Agents"""

    model_config = ConfigDict(extra='forbid')

# ADD NEW STRICT OUTPUT MODELS FOR TOOLS
class NormalizedJson(StrictBaseModel):
    """Strict model for normalized JSON tool output"""
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    original: Optional[str] = None

class PlayerStatsExtraction(StrictBaseModel):
    """Strict model for player stats extraction tool output"""
    player_name: str = "Chase"
    stats: Dict[str, int] = Field(default_factory=dict)

class NPCSimpleUpdate(StrictBaseModel):
    """Strict model for NPC update tool output"""
    npc_id: int
    current_location: Optional[str] = None
    npc_name: Optional[str] = None

class RelationshipSimpleChange(StrictBaseModel):
    """Strict model for relationship change tool output"""
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    level_change: Optional[int] = None
    new_event: Optional[str] = None
    group_context: Optional[str] = None

class ApplyUpdatesResult(StrictBaseModel):
    """Strict model for apply updates tool output"""
    success: bool
    updates_applied: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    reason: Optional[str] = None

class NPCCreation(StrictBaseModel):
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
    memory: List[str] = Field(default_factory=list)
    monica_level: Optional[int] = None
    age: Optional[int] = None
    birthdate: Optional[str] = None

class NPCUpdate(StrictBaseModel):
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
    memory: Optional[List[str]] = None
    schedule: Optional[Dict[str, Any]] = None
    schedule_updates: Optional[Dict[str, Any]] = None
    affiliations: Optional[List[str]] = None
    current_location: Optional[str] = None

class NPCIntroduction(StrictBaseModel):
    npc_id: int

class PlayerStats(StrictBaseModel):
    corruption: Optional[int] = None
    confidence: Optional[int] = None
    willpower: Optional[int] = None
    obedience: Optional[int] = None
    dependency: Optional[int] = None
    lust: Optional[int] = None
    mental_resilience: Optional[int] = None
    physical_endurance: Optional[int] = None

class CharacterStatUpdates(StrictBaseModel):
    player_name: str = "Chase"
    stats: PlayerStats

class RelationshipUpdate(StrictBaseModel):
    npc_id: int
    affiliations: List[str]

class SocialLink(StrictBaseModel):
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    link_type: Optional[str] = None
    level_change: Optional[int] = None
    new_event: Optional[str] = None
    group_context: Optional[str] = None

class Location(StrictBaseModel):
    location_name: str
    description: Optional[str] = None
    open_hours: List[str] = Field(default_factory=list)

class Event(StrictBaseModel):
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

class Quest(StrictBaseModel):
    quest_id: Optional[int] = None
    quest_name: Optional[str] = None
    status: Optional[str] = None
    progress_detail: Optional[str] = None
    quest_giver: Optional[str] = None
    reward: Optional[str] = None

class InventoryItem(StrictBaseModel):
    item_name: str
    item_description: Optional[str] = None
    item_effect: Optional[str] = None
    category: Optional[str] = None

class InventoryUpdates(StrictBaseModel):
    player_name: str = "Chase"
    added_items: List[InventoryItem] = Field(default_factory=list)
    removed_items: List[str] = Field(default_factory=list) # Assuming you just need names to remove

class Perk(StrictBaseModel):
    player_name: str = "Chase"
    perk_name: str
    perk_description: Optional[str] = None
    perk_effect: Optional[str] = None

class Activity(StrictBaseModel):
    activity_name: str
    purpose: Optional[Dict[str, Any]] = None
    stat_integration: Optional[Dict[str, Any]] = None
    intensity_tier: Optional[int] = None
    setting_variant: Optional[str] = None

class JournalEntry(StrictBaseModel):
    entry_type: str
    entry_text: str
    fantasy_flag: bool = False
    intensity_level: Optional[int] = None

class ImageGeneration(StrictBaseModel):
    generate: bool = False
    priority: str = "low"
    focus: str = "balanced"
    framing: str = "medium_shot"
    reason: Optional[str] = None

class UniversalUpdateInput(StrictBaseModel):
    user_id: int
    conversation_id: int
    narrative: str
    roleplay_updates: Dict[str, str] = Field(default_factory=dict)
    ChaseSchedule: Optional[Dict[str, str]] = None
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

class ContentSafety(StrictBaseModel):
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
        self.lore_system = None
        
    async def initialize(self):
        """Initialize context with governance integration"""
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        self.lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)

# -------------------------------------------------------------------------------
# Function Tools (UPDATED TO RETURN STRICT MODELS)
# -------------------------------------------------------------------------------

@function_tool
async def normalize_json(ctx, json_str: str) -> NormalizedJson:
    """
    Normalize JSON string, fixing common errors:
    - Replace curly quotes with straight quotes
    - Add missing quotes around keys
    - Fix trailing commas
    
    Args:
        json_str: A potentially malformed JSON string
        
    Returns:
        NormalizedJson with parsed data or error info
    """
    try:
        # Try to parse as-is first
        data = json.loads(json_str)
        return NormalizedJson(ok=True, data=data)
    except json.JSONDecodeError:
        # Simple normalization - replace curly quotes using unicode escapes
        normalized = (json_str
            .replace("\u201c", '"').replace("\u201d", '"')  # Curly double quotes
            .replace("\u2018", "'").replace("\u2019", "'")  # Curly single quotes
        )
        
        try:
            data = json.loads(normalized)
            return NormalizedJson(ok=True, data=data)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to normalize JSON: {e}")
            # Return structured failure info
            return NormalizedJson(
                ok=False,
                error="Failed to parse JSON",
                message=str(e),
                original=json_str
            )

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
async def extract_player_stats(ctx, narrative: str) -> PlayerStatsExtraction:
    """
    Extract player stat changes from narrative text.
    
    Args:
        narrative: The narrative text to analyze
        
    Returns:
        PlayerStatsExtraction with player stat changes
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
        return PlayerStatsExtraction(player_name="Chase", stats={})
    
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
    
    return PlayerStatsExtraction(player_name="Chase", stats=changes)

@function_tool
async def extract_npc_changes(ctx, narrative: str) -> List[NPCSimpleUpdate]:
    """
    Extract NPC changes from narrative text.
    
    Args:
        narrative: The narrative text to analyze
        
    Returns:
        List of NPCSimpleUpdate models
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
                updates.append(NPCSimpleUpdate(**npc_update))
        
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
async def extract_relationship_changes(ctx, narrative: str) -> List[RelationshipSimpleChange]:
    """
    Extract relationship changes from narrative text.
    
    Args:
        narrative: The narrative text to analyze
        
    Returns:
        List of RelationshipSimpleChange models
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
                    relationship_change = RelationshipSimpleChange(
                        entity1_type="player",
                        entity1_id=0,  # Player ID
                        entity2_type="npc",
                        entity2_id=npc_id,
                        level_change=5,  # Modest increase
                        new_event=f"{npc_name} {indicator}"
                    )
                    break
            
            if not relationship_change:
                for indicator in negative_indicators:
                    if f"{npc_name} {indicator}" in narrative:
                        relationship_change = RelationshipSimpleChange(
                            entity1_type="player",
                            entity1_id=0,  # Player ID
                            entity2_type="npc",
                            entity2_id=npc_id,
                            level_change=-5,  # Modest decrease
                            new_event=f"{npc_name} {indicator}"
                        )
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

# REFACTORED: Now uses canon and LoreSystem instead of direct database operations
async def apply_universal_updates_async(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    updates: Dict[str, Any],
    conn: asyncpg.Connection
) -> Dict[str, Any]:
    """
    Apply universal updates using canon and LoreSystem.
    
    Args:
        ctx: UniversalUpdaterContext with lore_system
        user_id: User ID
        conversation_id: Conversation ID
        updates: Dictionary containing all the updates to apply
        conn: Database connection (passed by LoreSystem)
        
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
            npc_creation_count = await process_npc_creations_canonical(
                ctx, user_id, conversation_id, updates["npc_creations"], conn
            )
            results["details"]["npc_creations"] = npc_creation_count
            results["updates_applied"] += npc_creation_count
        
        # Process NPC updates
        if "npc_updates" in updates and updates["npc_updates"]:
            npc_update_count = await process_npc_updates_canonical(
                ctx, user_id, conversation_id, updates["npc_updates"], conn
            )
            results["details"]["npc_updates"] = npc_update_count
            results["updates_applied"] += npc_update_count
        
        # Process character stat updates
        if "character_stat_updates" in updates and updates["character_stat_updates"]:
            stat_update_count = await process_character_stats_canonical(
                ctx, user_id, conversation_id, updates["character_stat_updates"], conn
            )
            results["details"]["stat_updates"] = stat_update_count
            results["updates_applied"] += stat_update_count
        
        # Process social links
        if "social_links" in updates and updates["social_links"]:
            social_link_count = await process_social_links_canonical(
                ctx, user_id, conversation_id, updates["social_links"], conn
            )
            results["details"]["social_links"] = social_link_count
            results["updates_applied"] += social_link_count
        
        # Process roleplay updates
        if "roleplay_updates" in updates and updates["roleplay_updates"]:
            roleplay_update_count = await process_roleplay_updates_canonical(
                ctx, user_id, conversation_id, updates["roleplay_updates"], conn
            )
            results["details"]["roleplay_updates"] = roleplay_update_count
            results["updates_applied"] += roleplay_update_count
        
        # Return results
        return results
    except Exception as e:
        logger.error(f"Error applying universal updates: {e}")
        return {"success": False, "error": str(e)}

# REFACTORED: Helper functions now use canon

async def process_npc_creations_canonical(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    npc_creations: List[Dict[str, Any]],
    conn: asyncpg.Connection
) -> int:
    """Process NPC creations using canon."""
    count = 0
    canon_ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
    for npc in npc_creations:
        # Prepare NPC data package
        npc_data = {
            'npc_name': npc['npc_name'],
            'introduced': npc.get('introduced', False),
            'sex': npc.get('sex', 'female'),
            'dominance': npc.get('dominance'),
            'cruelty': npc.get('cruelty'),
            'closeness': npc.get('closeness'),
            'trust': npc.get('trust'),
            'respect': npc.get('respect'),
            'intensity': npc.get('intensity'),
            'archetypes': json.dumps(npc.get('archetypes', [])) if npc.get('archetypes') else None,
            'archetype_summary': npc.get('archetype_summary'),
            'archetype_extras_summary': npc.get('archetype_extras_summary'),
            'physical_description': npc.get('physical_description'),
            'hobbies': json.dumps(npc.get('hobbies', [])) if npc.get('hobbies') else None,
            'personality_traits': json.dumps(npc.get('personality_traits', [])) if npc.get('personality_traits') else None,
            'likes': json.dumps(npc.get('likes', [])) if npc.get('likes') else None,
            'dislikes': json.dumps(npc.get('dislikes', [])) if npc.get('dislikes') else None,
            'affiliations': json.dumps(npc.get('affiliations', [])) if npc.get('affiliations') else None,
            'schedule': json.dumps(npc.get('schedule', {})) if npc.get('schedule') else None,
            'memory': json.dumps(npc.get('memory')) if npc.get('memory') else None,
            'monica_level': npc.get('monica_level'),
            'age': npc.get('age'),
            'birthdate': npc.get('birthdate')
        }
        
        # Remove None values
        npc_data = {k: v for k, v in npc_data.items() if v is not None}
        
        # Use canon to create NPC
        npc_id = await canon.find_or_create_npc(canon_ctx, conn, **npc_data)
        if npc_id:
            count += 1
    
    return count

async def process_npc_updates_canonical(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    npc_updates: List[Dict[str, Any]],
    conn: asyncpg.Connection
) -> int:
    """Process NPC updates using LoreSystem."""
    count = 0
    
    for npc in npc_updates:
        # Skip if no npc_id
        if "npc_id" not in npc:
            continue
        
        # Build update dictionary
        updates = {}
        
        # Simple fields
        simple_fields = [
            "npc_name", "introduced", "archetype_summary", "archetype_extras_summary",
            "physical_description", "dominance", "cruelty", "closeness", "trust",
            "respect", "intensity", "sex", "current_location"
        ]
        
        for field in simple_fields:
            if field in npc and npc[field] is not None:
                updates[field] = npc[field]
        
        # JSON fields
        json_fields = ["hobbies", "personality_traits", "likes", "dislikes", 
                       "affiliations", "memory", "schedule"]
        
        for field in json_fields:
            if field in npc and npc[field] is not None:
                updates[field] = json.dumps(npc[field]) if isinstance(npc[field], (list, dict)) else npc[field]
        
        # Skip if no fields to update
        if not updates:
            continue
        
        # Use LoreSystem to update
        result = await ctx.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier={"npc_id": npc["npc_id"], "user_id": user_id, "conversation_id": conversation_id},
            updates=updates,
            reason=f"Narrative update for NPC {npc.get('npc_name', npc['npc_id'])}"
        )
        
        if result.get("status") in ["committed", "conflict_generated"]:
            count += 1
    
    return count

async def process_character_stats_canonical(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    stat_updates: Dict[str, Any],
    conn: asyncpg.Connection
) -> int:
    """Process character stat updates using canon and LoreSystem."""
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
        # Create player using canon
        canon_ctx = RunContextWrapper(context={
            'user_id': user_id,
            'conversation_id': conversation_id
        })
        await canon.find_or_create_player_stats(
            canon_ctx, conn, player_name,
            corruption=0, confidence=0, willpower=0, obedience=0,
            dependency=0, lust=0, mental_resilience=0, physical_endurance=0
        )
    
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
                
                # Use LoreSystem to update
                result = await ctx.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="PlayerStats",
                    entity_identifier={
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "player_name": player_name
                    },
                    updates={stat: new_value},
                    reason=f"Narrative update: {stat} changed by {value}"
                )
                
                if result.get("status") in ["committed", "conflict_generated"]:
                    # Log stat change in history using canon
                    canon_ctx = type('obj', (object,), {'user_id': user_id, 'conversation_id': conversation_id})
                    await canon.log_stat_change(
                        canon_ctx, conn, player_name, stat,
                        current_value, new_value, "Narrative update"
                    )
                    update_count += 1
    
    return update_count

async def process_social_links_canonical(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    social_links: List[Dict[str, Any]],
    conn: asyncpg.Connection
) -> int:
    """Process relationship updates using the new dynamic system."""
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    count = 0
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    
    for link in social_links:
        # Convert old social link format to new interaction format
        interaction_type = None
        
        # Map link events to interaction types
        if link.get('new_event'):
            event_text = link['new_event'].lower()
            if 'help' in event_text or 'support' in event_text:
                interaction_type = 'helpful_action'
            elif 'betray' in event_text:
                interaction_type = 'betrayal'
            elif 'praise' in event_text or 'compliment' in event_text:
                interaction_type = 'genuine_compliment'
            else:
                interaction_type = 'social_interaction'
        
        # Process the interaction
        if interaction_type:
            result = await manager.process_interaction(
                entity1_type=link['entity1_type'],
                entity1_id=link['entity1_id'],
                entity2_type=link['entity2_type'],
                entity2_id=link['entity2_id'],
                interaction={
                    'type': interaction_type,
                    'context': link.get('group_context', 'casual'),
                    'description': link.get('new_event', '')
                }
            )
            
            if result.get('success'):
                count += 1
        
        # Handle direct level changes
        elif link.get('level_change'):
            # Map old level change to dimension changes
            level_change = link['level_change']
            dimension_changes = {
                'affection': level_change * 0.5,
                'trust': level_change * 0.3,
                'closeness': level_change * 0.2
            }
            
            # Get current state
            state = await manager.get_relationship_state(
                entity1_type=link['entity1_type'],
                entity1_id=link['entity1_id'],
                entity2_type=link['entity2_type'],
                entity2_id=link['entity2_id']
            )
            
            # Apply changes
            for dim, change in dimension_changes.items():
                current = getattr(state.dimensions, dim)
                setattr(state.dimensions, dim, current + change)
            
            state.dimensions.clamp()
            await manager._queue_update(state)
            count += 1
    
    # Flush all updates
    await manager._flush_updates()
    
    return count

async def process_roleplay_updates_canonical(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    roleplay_updates: Dict[str, Any],
    conn: asyncpg.Connection
) -> int:
    """Process roleplay updates using canon."""
    count = 0
    canon_ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
    for key, value in roleplay_updates.items():
        if value is not None:
            # Use canon to update current roleplay
            await canon.update_current_roleplay(
                canon_ctx, conn, user_id, conversation_id, key, str(value)
            )
            count += 1
    
    return count

@function_tool
async def apply_universal_updates(ctx, updates_json: str) -> ApplyUpdatesResult:
    """
    Apply universal updates to the database.
    
    Args:
        updates_json: JSON string containing all the updates to apply
        
    Returns:
        ApplyUpdatesResult with update results
    """
    # Parse the JSON string - handle NormalizedJson if needed
    try:
        # First try direct parsing
        updates = json.loads(updates_json)
    except json.JSONDecodeError:
        # If it fails, try normalizing with our tool
        normalized = await normalize_json(ctx, updates_json)
        if normalized.ok and normalized.data:
            updates = normalized.data
        else:
            return ApplyUpdatesResult(
                success=False, 
                error=f"Invalid JSON: {normalized.error or 'Unknown error'}"
            )
    
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
        return ApplyUpdatesResult(success=False, reason=permission["reasoning"])
    
    try:
        # Ensure user_id and conversation_id are set in updates
        updates["user_id"] = user_id
        updates["conversation_id"] = conversation_id
        
        async with get_db_connection_context() as conn:
            # Apply updates using the canonical function
            result = await apply_universal_updates_async(
                ctx.context,
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
            
            return ApplyUpdatesResult(**result)
    except Exception as e:
        logging.error(f"Error applying universal updates: {e}")
        return ApplyUpdatesResult(success=False, error=str(e))

# -------------------------------------------------------------------------------
# Guardrail Functions
# -------------------------------------------------------------------------------

async def content_safety_guardrail(ctx, agent, input_data):
    """Input guardrail for content moderation"""
    try:
        content_moderator = Agent(
            name="Content Moderator",
            instructions="""
            You check if content is appropriate for a femdom roleplay game. 
            Allow adult themes within the context of a consensual femdom relationship,
            but flag anything that might be genuinely harmful or problematic.
            """,
            output_type=ContentSafety,
            output_schema_strict=True  # Ensure strict output
        )
        
        result = await Runner.run(content_moderator, input_data, context=ctx.context)
        final_output = result.final_output_as(ContentSafety)
        
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=not final_output.is_appropriate,
        )
    except Exception as e:
        logging.error(f"Error in content safety guardrail: {str(e)}", exc_info=True)
        # Return safe default on error
        return GuardrailFunctionOutput(
            output_info=ContentSafety(
                is_appropriate=True,
                reasoning="Error in content moderation, defaulting to safe",
                suggested_adjustment=None
            ),
            tripwire_triggered=False,
        )

# -------------------------------------------------------------------------------
# Agent Definitions (UPDATED WITH STRICT SCHEMAS)
# -------------------------------------------------------------------------------

# Extraction agent for initial analysis - now with output_type
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
    
    Return your findings as plain text describing the extracted changes.
    """,
    tools=[
        extract_player_stats,
        extract_npc_changes,
        extract_relationship_changes
    ],
    output_type=str,  # Explicitly set as string output
    # output_schema_strict=False,  # Optional: explicitly disable strict schema for text output
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
    
    IMPORTANT: Only include fields that are part of the UniversalUpdateInput schema.
    Do not include any fields that are not defined in the schema.
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
    output_schema_strict=True,  # Explicitly enable strict schema
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
    
    # Create trace for monitoring - Fix the trace ID format
    with trace(
        workflow_name="Universal Update",
        trace_id=f"trace_universal_update_{conversation_id}_{int(datetime.now().timestamp())}",
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
        Do not include any additional fields not defined in the schema.
        """
        
        try:
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
                # Convert the Pydantic model to dict, then to JSON string
                update_dict = update_data.dict()
                update_json = json.dumps(update_dict)
                
                # Wrap the context for the tool call
                wrapped_ctx = RunContextWrapper(updater_context)
                update_result = await apply_universal_updates(wrapped_ctx, update_json)
                
                # Convert ApplyUpdatesResult back to dict for backward compatibility
                return update_result.model_dump()
            else:
                return {"success": False, "error": "No updates extracted"}
                
        except Exception as e:
            logging.error(f"Error in universal updater agent execution: {str(e)}", exc_info=True)
            logging.error(f"Agent: universal_updater_agent, Context: user_id={user_id}, conversation_id={conversation_id}")
            return {"success": False, "error": f"Agent execution error: {str(e)}"}

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
        # Create the wrapper agent class for governance compatibility
        updater_agent = UniversalUpdaterAgent(user_id, conversation_id)
        await updater_agent.initialize()
        
        # Register with governance system
        await register_with_governance(user_id, conversation_id)
        
        logging.info(f"Universal Updater initialized for user {user_id}, conversation {conversation_id}")
        
        return {
            "agent": updater_agent,
            "context": updater_agent.context,
            "status": "initialized"
        }
    except Exception as e:
        logging.error(f"Error initializing Universal Updater: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "status": "failed"
        }
        
# Add this class to provide compatibility with existing governance code
class UniversalUpdaterAgent:
    """
    Compatibility wrapper for the OpenAI Agents SDK implementation of Universal Updater.
    This class serves as an adapter between the SDK implementation and the
    governance system which expects a class-based approach.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the Universal Updater Agent with user context."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the updater agent and context."""
        if not self.initialized:
            # Create and initialize the updater context
            self.context = UniversalUpdaterContext(self.user_id, self.conversation_id)
            await self.context.initialize()
            self.initialized = True
        return self
    
    async def process_update(self, narrative: str, context: Dict[str, Any] = None):
        """Process a universal update based on narrative text."""
        # Ensure initialization
        if not self.initialized:
            await self.initialize()
            
        # Use the SDK-based function for processing
        return await process_universal_update(
            self.user_id, 
            self.conversation_id, 
            narrative, 
            context
        )
    
    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a directive from the governance system."""
        # Ensure initialization
        if not self.initialized:
            await self.initialize()
            
        # Process different directive types
        directive_type = directive.get("type")
        directive_data = directive.get("data", {})
        
        if directive_type == "process_narrative":
            return await self.process_update(
                directive_data.get("narrative", ""),
                directive_data.get("context")
            )
        
        return {
            "success": False,
            "error": f"Unsupported directive type: {directive_type}"
        }
    
    async def get_capabilities(self) -> List[str]:
        """Return agent capabilities for coordination."""
        return ["narrative_analysis", "state_extraction", "state_updating"]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for the agent."""
        return {
            "updates_processed": 0,  # You would track this in a real implementation
            "success_rate": 0.0,
            "average_processing_time": 0.0,
            "strategies": {}
        }
    
    async def get_learning_state(self) -> Dict[str, Any]:
        """Return learning state for the agent."""
        return {
            "patterns": {},
            "adaptations": []
        }

# Add these helper functions that were missing from canon
async def find_or_create_player_stats(ctx, conn, player_name: str, **kwargs) -> None:
    """Helper function to create player stats if they don't exist."""
    await conn.execute("""
        INSERT INTO PlayerStats (
            user_id, conversation_id, player_name,
            corruption, confidence, willpower, obedience,
            dependency, lust, mental_resilience, physical_endurance
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
    """, ctx.user_id, ctx.conversation_id, player_name,
    kwargs.get('corruption', 0), kwargs.get('confidence', 0),
    kwargs.get('willpower', 0), kwargs.get('obedience', 0),
    kwargs.get('dependency', 0), kwargs.get('lust', 0),
    kwargs.get('mental_resilience', 0), kwargs.get('physical_endurance', 0))

async def log_stat_change(ctx, conn, player_name: str, stat_name: str, old_value: int, new_value: int, cause: str) -> None:
    """Helper function to log stat changes."""
    await conn.execute("""
        INSERT INTO StatsHistory (
            user_id, conversation_id, player_name, stat_name,
            old_value, new_value, cause
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    """, ctx.user_id, ctx.conversation_id, player_name, stat_name,
    old_value, new_value, cause)
