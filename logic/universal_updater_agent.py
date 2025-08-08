# logic/universal_updater_agent.py

"""
Universal Updater SDK using OpenAI's Agents SDK with Nyx Governance integration.

This module is responsible for analyzing narrative text and extracting appropriate 
game state updates. It uses the new array format for roleplay_updates, ChaseSchedule,
character stats, and other fields that were previously dicts.

REFACTORED: Now handles array format for schema compliance while maintaining
backward compatibility with database storage (which still expects dicts).
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

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

# ===============================================================================
# Array Format Helper Functions
# ===============================================================================

def get_from_array(array_data: List[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """Get value from array of {key, value} pairs"""
    if not isinstance(array_data, list):
        return default
    for item in array_data:
        if isinstance(item, dict) and item.get("key") == key:
            return item.get("value", default)
    return default

def set_in_array(array_data: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
    """Set or update value in array of {key, value} pairs"""
    if not isinstance(array_data, list):
        array_data = []
    
    # Update existing
    for item in array_data:
        if isinstance(item, dict) and item.get("key") == key:
            item["value"] = value
            return array_data
    
    # Add new
    array_data.append({"key": key, "value": value})
    return array_data

def remove_from_array(array_data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Remove entry from array of {key, value} pairs"""
    if not isinstance(array_data, list):
        return []
    return [item for item in array_data if item.get("key") != key]

def array_to_dict(array_data: List[Dict[str, Any]], key_name: str = "key", value_name: str = "value") -> Dict[str, Any]:
    """Convert array of key-value pairs back to dict."""
    if not isinstance(array_data, list):
        return {}
    result = {}
    for item in array_data:
        if isinstance(item, dict) and key_name in item and value_name in item:
            result[item[key_name]] = item[value_name]
    return result

def dict_to_array(obj_data: Dict[str, Any], key_name: str = "key", value_name: str = "value") -> List[Dict[str, Any]]:
    """Convert object/dict to array of key-value pairs."""
    if not isinstance(obj_data, dict):
        return []
    return [
        {key_name: k, value_name: v} 
        for k, v in obj_data.items()
    ]

def ensure_array_format(data: Union[Dict, List]) -> List[Dict[str, Any]]:
    """Ensure data is in array format"""
    if isinstance(data, dict):
        # It's still in dict format, convert it
        return dict_to_array(data)
    elif isinstance(data, list):
        # Already in array format
        return data
    else:
        return []

def ensure_dict_format(data: Union[Dict, List]) -> Dict[str, Any]:
    """Ensure data is in dict format (for database storage)"""
    if isinstance(data, list):
        # It's in array format, convert it
        return array_to_dict(data)
    elif isinstance(data, dict):
        # Already in dict format
        return data
    else:
        return {}

# ===============================================================================
# Pydantic Models for Structured Outputs (Updated for Array Format)
# ===============================================================================

class StrictBaseModel(BaseModel):
    """Base class enforcing a strict schema for OpenAI Agents"""
    model_config = ConfigDict(extra='forbid')

# ---- JSON-safe aliases (strict) ----
# Allowed scalar types in strict output
JsonScalar = Union[str, int, float, bool, None]
# Allowed “value” types in {key,value} pairs:
# - scalar
# - list of scalars
# - DaySchedule (defined below)
# NOTE: If you need more object types here later, add them explicitly.
# (We define KeyValuePair AFTER DaySchedule so the type exists.)

class KeyValueStr(StrictBaseModel):
    """String key-value pair"""
    key: str
    value: str

class KeyValueInt(StrictBaseModel):
    """Integer key-value pair"""
    key: str
    value: int

# Schedule models using array format
class DaySchedule(StrictBaseModel):
    """Schedule for a single day"""
    Morning: Optional[str] = None
    Afternoon: Optional[str] = None
    Evening: Optional[str] = None
    Night: Optional[str] = None

class ScheduleEntry(StrictBaseModel):
    """Single schedule entry in array format"""
    key: str  # Day name (Monday, Tuesday, etc.)
    value: DaySchedule

# Now that DaySchedule exists, define the strict KeyValuePair
JsonList = List[JsonScalar]
JsonValue = Union[JsonScalar, JsonList, DaySchedule]

class KeyValuePair(StrictBaseModel):
    """Generic key-value pair for array format (strict)"""
    key: str
    value: JsonValue

# NPC models
class NPCArchetype(StrictBaseModel):
    """NPC archetype entry"""
    id: Optional[int] = None
    name: Optional[str] = None

class NPCCreation(StrictBaseModel):
    """NPC creation with array format for schedule"""
    npc_name: str
    introduced: bool = False
    sex: str = "female"
    dominance: Optional[int] = None
    cruelty: Optional[int] = None
    closeness: Optional[int] = None
    trust: Optional[int] = None
    respect: Optional[int] = None
    intensity: Optional[int] = None
    archetypes: List[NPCArchetype] = Field(default_factory=list)
    archetype_summary: Optional[str] = None
    archetype_extras_summary: Optional[str] = None
    physical_description: Optional[str] = None
    hobbies: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    schedule: List[ScheduleEntry] = Field(default_factory=list)  # Array format
    memory: Union[str, List[str], None] = None
    monica_level: Optional[int] = None
    age: Optional[int] = None
    birthdate: Optional[str] = None

class NPCUpdate(StrictBaseModel):
    """NPC update with array format for schedule"""
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
    memory: Union[str, List[str], None] = None
    schedule: Optional[List[ScheduleEntry]] = None  # Array format
    schedule_updates: Optional[List[ScheduleEntry]] = None  # Array format
    affiliations: Optional[List[str]] = None
    current_location: Optional[str] = None

class NPCIntroduction(StrictBaseModel):
    npc_id: int

# Character stats using array format
class StatEntry(StrictBaseModel):
    """Single stat entry in array format"""
    key: str  # Stat name (corruption, confidence, etc.)
    value: int

class CharacterStatUpdates(StrictBaseModel):
    """Character stat updates with array format"""
    player_name: str = "Chase"
    stats: List[StatEntry] = Field(default_factory=list)  # Array format

# Relationship models
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

# Location and Event models
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

# Quest model
class Quest(StrictBaseModel):
    quest_id: Optional[int] = None
    quest_name: Optional[str] = None
    status: Optional[str] = None
    progress_detail: Optional[str] = None
    quest_giver: Optional[str] = None
    reward: Optional[str] = None

# Inventory models
class InventoryItem(StrictBaseModel):
    item_name: str
    item_description: Optional[str] = None
    item_effect: Optional[str] = None
    category: Optional[str] = None

class InventoryRemovedItem(StrictBaseModel):
    """Item removed from inventory"""
    name: str

class InventoryUpdates(StrictBaseModel):
    player_name: str = "Chase"
    added_items: List[Union[str, InventoryItem]] = Field(default_factory=list)
    removed_items: List[Union[str, InventoryRemovedItem]] = Field(default_factory=list)

# Perk model
class Perk(StrictBaseModel):
    player_name: str = "Chase"
    perk_name: str
    perk_description: Optional[str] = None
    perk_effect: Optional[str] = None

# Activity models with array format
class ActivityPurpose(StrictBaseModel):
    description: Optional[str] = None
    fantasy_level: str = "realistic"

class StatIntegrationEntry(StrictBaseModel):
    """Single stat integration entry in array format"""
    key: str
    value: JsonScalar

class Activity(StrictBaseModel):
    """Activity with array format for stat_integration"""
    activity_name: str
    purpose: Optional[ActivityPurpose] = None
    stat_integration: List[StatIntegrationEntry] = Field(default_factory=list)  # Array format
    intensity_tier: Optional[int] = None
    setting_variant: Optional[str] = None

# Journal model
class JournalEntry(StrictBaseModel):
    entry_type: str
    entry_text: str
    fantasy_flag: bool = False
    intensity_level: Optional[int] = None

# Image generation model
class ImageGeneration(StrictBaseModel):
    generate: bool = False
    priority: str = "low"
    focus: str = "balanced"
    framing: str = "medium_shot"
    reason: Optional[str] = None

# Main update input model with array formats
class UniversalUpdateInput(StrictBaseModel):
    """Main input model using array formats where appropriate"""
    user_id: int
    conversation_id: int
    narrative: str
    roleplay_updates: List[KeyValuePair] = Field(default_factory=list)  # Array format
    ChaseSchedule: List[ScheduleEntry] = Field(default_factory=list)  # Array format
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

# Tool output models
class NormalizedJson(StrictBaseModel):
    ok: bool
    data: List[KeyValuePair] = Field(default_factory=list)
    error: Optional[str] = None
    message: Optional[str] = None
    original: Optional[str] = None

class PlayerStatsExtraction(StrictBaseModel):
    player_name: str = "Chase"
    stats: List[StatEntry] = Field(default_factory=list)

class NPCSimpleUpdate(StrictBaseModel):
    npc_id: int
    current_location: Optional[str] = None
    npc_name: Optional[str] = None

class RelationshipSimpleChange(StrictBaseModel):
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    level_change: Optional[int] = None
    new_event: Optional[str] = None
    group_context: Optional[str] = None

class ApplyUpdatesResult(StrictBaseModel):
    success: bool
    updates_applied: Optional[int] = None
    details: Optional[List[KeyValuePair]] = None
    error: Optional[str] = None
    reason: Optional[str] = None

class ContentSafety(StrictBaseModel):
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

# ===============================================================================
# Agent Context
# ===============================================================================

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

# ===============================================================================
# Conversion Functions for Database Storage
# ===============================================================================

def convert_updates_for_database(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert array formats back to dicts for database storage.
    The database still expects dict format for backward compatibility.
    """
    db_updates = updates.copy()
    
    # Convert roleplay_updates from array to dict
    if "roleplay_updates" in db_updates and isinstance(db_updates["roleplay_updates"], list):
        db_updates["roleplay_updates"] = array_to_dict(db_updates["roleplay_updates"])
    
    # Convert ChaseSchedule from array to dict
    if "ChaseSchedule" in db_updates and isinstance(db_updates["ChaseSchedule"], list):
        db_updates["ChaseSchedule"] = array_to_dict(db_updates["ChaseSchedule"])
    
    # Convert character stats from array to dict
    if "character_stat_updates" in db_updates and db_updates["character_stat_updates"]:
        stats = db_updates["character_stat_updates"].get("stats", [])
        if isinstance(stats, list):
            db_updates["character_stat_updates"]["stats"] = array_to_dict(stats)
    
    # Convert NPC schedules from array to dict
    for npc in db_updates.get("npc_creations", []):
        if "schedule" in npc and isinstance(npc["schedule"], list):
            npc["schedule"] = array_to_dict(npc["schedule"])
    
    for npc in db_updates.get("npc_updates", []):
        if "schedule" in npc and isinstance(npc["schedule"], list):
            npc["schedule"] = array_to_dict(npc["schedule"])
        if "schedule_updates" in npc and isinstance(npc["schedule_updates"], list):
            npc["schedule_updates"] = array_to_dict(npc["schedule_updates"])
    
    # Convert activity stat_integration from array to dict
    for activity in db_updates.get("activity_updates", []):
        if "stat_integration" in activity and isinstance(activity["stat_integration"], list):
            activity["stat_integration"] = array_to_dict(activity["stat_integration"])
    
    return db_updates

def convert_from_database_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert database dict format to array format for schema compliance.
    """
    result = data.copy()
    
    # Convert roleplay_updates to array
    if "roleplay_updates" in result and isinstance(result["roleplay_updates"], dict):
        result["roleplay_updates"] = dict_to_array(result["roleplay_updates"])
    
    # Convert ChaseSchedule to array
    if "ChaseSchedule" in result and isinstance(result["ChaseSchedule"], dict):
        result["ChaseSchedule"] = dict_to_array(result["ChaseSchedule"])
    
    # Convert character stats to array
    if "character_stat_updates" in result and result["character_stat_updates"]:
        stats = result["character_stat_updates"].get("stats", {})
        if isinstance(stats, dict):
            result["character_stat_updates"]["stats"] = dict_to_array(stats)
    
    # Convert NPC schedules to array
    for npc in result.get("npc_creations", []):
        if "schedule" in npc and isinstance(npc["schedule"], dict):
            npc["schedule"] = dict_to_array(npc["schedule"])
    
    for npc in result.get("npc_updates", []):
        if "schedule" in npc and isinstance(npc["schedule"], dict):
            npc["schedule"] = dict_to_array(npc["schedule"])
        if "schedule_updates" in npc and isinstance(npc["schedule_updates"], dict):
            npc["schedule_updates"] = dict_to_array(npc["schedule_updates"])
    
    # Convert activity stat_integration to array
    for activity in result.get("activity_updates", []):
        if "stat_integration" in activity and isinstance(activity["stat_integration"], dict):
            activity["stat_integration"] = dict_to_array(activity["stat_integration"])
    
    return result

# ===============================================================================
# Function Tools
# ===============================================================================

@function_tool
async def normalize_json(ctx: RunContextWrapper, json_str: str) -> NormalizedJson:
    """
    Normalize JSON string, fixing common errors.
    """
    try:
        # Try to parse as-is first
        data = json.loads(json_str)
        if isinstance(data, dict):
            data = dict_to_array(data)
        elif not isinstance(data, list):
            data = []
        return NormalizedJson(ok=True, data=data)
    except json.JSONDecodeError:
        # Simple normalization - replace curly quotes
        normalized = (json_str
            .replace("\u201c", '"').replace("\u201d", '"')  # Curly double quotes
            .replace("\u2018", "'").replace("\u2019", "'")  # Curly single quotes
        )

        try:
            data = json.loads(normalized)
            if isinstance(data, dict):
                data = dict_to_array(data)
            elif not isinstance(data, list):
                data = []
            return NormalizedJson(ok=True, data=data)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to normalize JSON: {e}")
            return NormalizedJson(
                ok=False,
                error="Failed to parse JSON",
                message=str(e),
                original=json_str
            )

@function_tool
async def check_npc_exists(ctx: RunContextWrapper, npc_id: int) -> bool:
    """Check if an NPC with the given ID exists in the database."""
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
async def extract_player_stats(ctx: RunContextWrapper, narrative: str) -> PlayerStatsExtraction:
    """Extract player stat changes from narrative text."""
    stats = ["corruption", "confidence", "willpower", "obedience", 
            "dependency", "lust", "mental_resilience", "physical_endurance"]
    
    governor = ctx.context.governor
    
    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="extract_player_stats",
        action_details={"narrative_length": len(narrative)}
    )

    if not permission["approved"]:
        return PlayerStatsExtraction(player_name="Chase", stats=[])
    
    changes = {}
    
    # Extract explicit mentions of stats
    for stat in stats:
        if f"{stat} increase" in narrative.lower() or f"{stat} rose" in narrative.lower():
            changes[stat] = 5
        elif f"{stat} decrease" in narrative.lower() or f"{stat} drop" in narrative.lower():
            changes[stat] = -5
    
    # Report action
    await governor.process_agent_action_report(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action={"type": "extract_player_stats"},
        result={"stats_changed": len(changes)}
    )
    
    return PlayerStatsExtraction(
        player_name="Chase",
        stats=[StatEntry(key=k, value=v) for k, v in changes.items()]
    )

@function_tool
async def apply_universal_updates(ctx: RunContextWrapper, updates_json: str) -> ApplyUpdatesResult:
    """
    Apply universal updates to the database.
    Handles conversion between array format (schema) and dict format (database).
    """
    # Parse the JSON string
    try:
        updates = json.loads(updates_json)
    except json.JSONDecodeError:
        normalized = await normalize_json(ctx, updates_json)
        if normalized.ok and normalized.data:
            updates = array_to_dict([item.model_dump() for item in normalized.data])
        else:
            return ApplyUpdatesResult(
                success=False,
                error=f"Invalid JSON: {normalized.error or 'Unknown error'}"
            )
    
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="apply_updates",
        action_details={"update_count": sum(len(updates.get(k, [])) for k in updates if isinstance(updates.get(k), list))}
    )
    
    if not permission["approved"]:
        return ApplyUpdatesResult(success=False, reason=permission["reasoning"])
    
    try:
        # Convert array formats to dict formats for database storage
        db_updates = convert_updates_for_database(updates)
        
        # Set user_id and conversation_id
        db_updates["user_id"] = user_id
        db_updates["conversation_id"] = conversation_id
        
        async with get_db_connection_context() as conn:
            # Apply updates
            result = await apply_universal_updates_async(
                ctx.context,
                user_id,
                conversation_id,
                db_updates,
                conn
            )
            
            # Report action
            await governor.process_agent_action_report(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="universal_updater",
                action={"type": "apply_updates"},
                result={"success": True, "updates_applied": result.get("updates_applied", 0)}
            )

            if "details" in result and isinstance(result["details"], dict):
                result["details"] = dict_to_array(result["details"])
            return ApplyUpdatesResult(**result)
    except Exception as e:
        logging.error(f"Error applying universal updates: {e}")
        return ApplyUpdatesResult(success=False, error=str(e))

# ===============================================================================
# Database Processing Functions (Updated for Array/Dict Conversion)
# ===============================================================================

async def apply_universal_updates_async(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    updates: Dict[str, Any],
    conn: asyncpg.Connection
) -> Dict[str, Any]:
    """
    Apply universal updates using canon and LoreSystem.
    Expects updates in dict format (after conversion from array format).
    """
    try:
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
        
        return results
    except Exception as e:
        logger.error(f"Error applying universal updates: {e}")
        return {"success": False, "error": str(e)}

async def process_npc_creations_canonical(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    npc_creations: List[Dict[str, Any]],
    conn: asyncpg.Connection
) -> int:
    """Process NPC creations using canon. Expects dict format for database."""
    count = 0
    canon_ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
    for npc in npc_creations:
        # Prepare JSON fields
        archetypes_json = json.dumps(npc.get('archetypes', [])) if npc.get('archetypes') else None
        schedule_json = json.dumps(npc.get('schedule', {})) if npc.get('schedule') else None
        hobbies_json = json.dumps(npc.get('hobbies', [])) if npc.get('hobbies') else None
        personality_json = json.dumps(npc.get('personality_traits', [])) if npc.get('personality_traits') else None
        likes_json = json.dumps(npc.get('likes', [])) if npc.get('likes') else None
        dislikes_json = json.dumps(npc.get('dislikes', [])) if npc.get('dislikes') else None
        affiliations_json = json.dumps(npc.get('affiliations', [])) if npc.get('affiliations') else None
        memory_json = json.dumps(npc.get('memory')) if npc.get('memory') else None
        
        # Prepare NPC data
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
            'archetypes': archetypes_json,
            'archetype_summary': npc.get('archetype_summary'),
            'archetype_extras_summary': npc.get('archetype_extras_summary'),
            'physical_description': npc.get('physical_description'),
            'hobbies': hobbies_json,
            'personality_traits': personality_json,
            'likes': likes_json,
            'dislikes': dislikes_json,
            'affiliations': affiliations_json,
            'schedule': schedule_json,
            'memory': memory_json,
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
    """Process NPC updates using LoreSystem. Expects dict format for database."""
    count = 0
    
    for npc in npc_updates:
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
                       "affiliations", "memory", "schedule", "schedule_updates"]
        
        for field in json_fields:
            if field in npc and npc[field] is not None:
                updates[field] = json.dumps(npc[field]) if isinstance(npc[field], (list, dict)) else npc[field]
        
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
    """Process character stat updates. Expects dict format for stats."""
    if not stat_updates or "stats" not in stat_updates:
        return 0
    
    player_name = stat_updates.get("player_name", "Chase")
    stats = stat_updates["stats"]
    
    # Stats should be in dict format after conversion
    if not stats or not isinstance(stats, dict):
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
        await find_or_create_player_stats(
            canon_ctx, conn, player_name,
            corruption=0, confidence=0, willpower=0, obedience=0,
            dependency=0, lust=0, mental_resilience=0, physical_endurance=0
        )
    
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
                # Calculate new value
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
                    # Log stat change
                    await log_stat_change(
                        ctx, conn, player_name, stat,
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
        # Map link events to interaction types
        interaction_type = None
        
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
    roleplay_updates: Union[Dict[str, Any], List[Dict[str, Any]]],
    conn: asyncpg.Connection
) -> int:
    """Process roleplay updates. Can handle both dict and array format."""
    count = 0
    canon_ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
    # Convert to dict format if needed
    if isinstance(roleplay_updates, list):
        roleplay_updates = array_to_dict(roleplay_updates)
    
    for key, value in roleplay_updates.items():
        if value is not None:
            await canon.update_current_roleplay(canon_ctx, conn, key, str(value))
            count += 1
    
    return count

# ===============================================================================
# Helper Functions for Canon Integration
# ===============================================================================

async def find_or_create_player_stats(ctx, conn, player_name: str, **kwargs) -> None:
    """Helper function to create player stats if they don't exist."""
    await conn.execute("""
        INSERT INTO PlayerStats (
            user_id, conversation_id, player_name,
            corruption, confidence, willpower, obedience,
            dependency, lust, mental_resilience, physical_endurance
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
    """, ctx.context.user_id if hasattr(ctx, 'context') else ctx.user_id,
    ctx.context.conversation_id if hasattr(ctx, 'context') else ctx.conversation_id,
    player_name,
    kwargs.get('corruption', 0), kwargs.get('confidence', 0),
    kwargs.get('willpower', 0), kwargs.get('obedience', 0),
    kwargs.get('dependency', 0), kwargs.get('lust', 0),
    kwargs.get('mental_resilience', 0), kwargs.get('physical_endurance', 0))

async def log_stat_change(ctx, conn, player_name: str, stat_name: str, old_value: int, new_value: int, cause: str) -> None:
    """Helper function to log stat changes."""
    user_id = ctx.context.user_id if hasattr(ctx, 'context') else ctx.user_id
    conversation_id = ctx.context.conversation_id if hasattr(ctx, 'context') else ctx.conversation_id
    
    await conn.execute("""
        INSERT INTO StatsHistory (
            user_id, conversation_id, player_name, stat_name,
            old_value, new_value, cause, changed_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
    """, user_id, conversation_id, player_name, stat_name,
    old_value, new_value, cause)

# ===============================================================================
# Guardrail Functions
# ===============================================================================

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
        )
        
        result = await Runner.run(content_moderator, input_data, context=ctx.context)
        final_output = result.final_output_as(ContentSafety)
        
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=not final_output.is_appropriate,
        )
    except Exception as e:
        logging.error(f"Error in content safety guardrail: {str(e)}", exc_info=True)
        return GuardrailFunctionOutput(
            output_info=ContentSafety(
                is_appropriate=True,
                reasoning="Error in content moderation, defaulting to safe",
                suggested_adjustment=None
            ),
            tripwire_triggered=False,
        )

# ===============================================================================
# Agent Definitions
# ===============================================================================

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
        check_npc_exists,
    ],
    output_type=str,
    model_settings=ModelSettings(temperature=0.1)
)

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
    
    IMPORTANT: All list-based fields that were previously dicts are now arrays of key-value pairs:
    - roleplay_updates: Array of {key, value} pairs
    - ChaseSchedule: Array of {key: day_name, value: {Morning, Afternoon, Evening, Night}}
    - character_stat_updates.stats: Array of {key: stat_name, value: stat_value}
    - NPC schedule: Array of {key: day_name, value: day_schedule}
    - Activity stat_integration: Array of {key, value} pairs
    
    Focus on extracting concrete changes rather than inferring too much.
    Be subtle in handling femdom themes - identify power dynamics but keep them understated.
    """,
    tools=[
        normalize_json,
        check_npc_exists,
        extract_player_stats,
        apply_universal_updates
    ],
    handoffs=[
        handoff(extraction_agent, tool_name_override="extract_state_changes")
    ],
    output_type=UniversalUpdateInput,
    input_guardrails=[
        InputGuardrail(guardrail_function=content_safety_guardrail),
    ],
    model_settings=ModelSettings(temperature=0.2)
)

# ===============================================================================
# Main Functions
# ===============================================================================

async def process_universal_update(
    user_id: int, 
    conversation_id: int, 
    narrative: str, 
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a universal update based on narrative text with governance oversight.
    """
    # Create and initialize the updater context
    updater_context = UniversalUpdaterContext(user_id, conversation_id)
    await updater_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Universal Update",
        trace_id=f"trace_universal_update_{conversation_id}_{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
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
        
        IMPORTANT: Use array format for these fields:
        - roleplay_updates: Array of {{key, value}} pairs
        - ChaseSchedule: Array of {{key: day_name, value: day_schedule}}
        - character_stat_updates.stats: Array of {{key: stat_name, value: stat_change}}
        - NPC schedule: Array format
        - Activity stat_integration: Array format
        
        Provide a structured output conforming to the UniversalUpdateInput schema.
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
                try:
                    if isinstance(update_data, str):
                        update_dict = json.loads(update_data)
                    elif hasattr(update_data, "model_dump"):
                        update_dict = update_data.model_dump()
                    elif isinstance(update_data, dict):
                        update_dict = update_data
                    else:
                        return {"success": False, "error": "Unsupported update data type"}
                    update_json = json.dumps(update_dict)
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON from updater agent: {e}")
                    return {"success": False, "error": f"Invalid JSON: {e}"}

                # Wrap the context for the tool call
                wrapped_ctx = RunContextWrapper(updater_context)
                update_result = await apply_universal_updates(wrapped_ctx, update_json)
                
                # Convert ApplyUpdatesResult back to dict
                return update_result.model_dump()
            else:
                return {"success": False, "error": "No updates extracted"}
                
        except Exception as e:
            logging.error(f"Error in universal updater agent execution: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Agent execution error: {str(e)}"}

async def register_with_governance(user_id: int, conversation_id: int):
    """Register universal updater agents with Nyx governance system."""
    governor = await get_central_governance(user_id, conversation_id)
    
    # Register main agent
    await governor.register_agent(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_instance=universal_updater_agent,
        agent_id="universal_updater"
    )
    
    # Issue directive
    await governor.issue_directive(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Process narrative updates and extract game state changes",
            "scope": "game"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60
    )
    
    logging.info("Universal Updater registered with Nyx governance")

async def initialize_universal_updater(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """Initialize the Universal Updater system and register with governance."""
    try:
        # Create the wrapper agent class
        updater_agent = UniversalUpdaterAgent(user_id, conversation_id)
        await updater_agent.initialize()
        
        # Register with governance
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

# ===============================================================================
# Compatibility Wrapper Class
# ===============================================================================

class UniversalUpdaterAgent:
    """
    Compatibility wrapper for the OpenAI Agents SDK implementation of Universal Updater.
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
            self.context = UniversalUpdaterContext(self.user_id, self.conversation_id)
            await self.context.initialize()
            self.initialized = True
        return self
    
    async def process_update(self, narrative: str, context: Dict[str, Any] = None):
        """Process a universal update based on narrative text."""
        if not self.initialized:
            await self.initialize()
            
        return await process_universal_update(
            self.user_id, 
            self.conversation_id, 
            narrative, 
            context
        )
    
    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a directive from the governance system."""
        if not self.initialized:
            await self.initialize()
            
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
        return ["narrative_analysis", "state_extraction", "state_updating", "array_format_handling"]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for the agent."""
        return {
            "updates_processed": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0,
            "strategies": {},
            "array_format_support": True
        }
    
    async def get_learning_state(self) -> Dict[str, Any]:
        """Return learning state for the agent."""
        return {
            "patterns": {},
            "adaptations": [],
            "format_version": "array_v2"
        }
