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
    AgentOutputSchema,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper,
    GuardrailFunctionOutput,
    InputGuardrail,
    trace,
    handoff
)
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# DB connection
from db.connection import get_db_connection_context
from db import rpc as db_rpc
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

from logic.universal_delta import build_delta_from_legacy_payload, DeltaBuildError
from openai_integration.conversations import (
    ensure_scene_seal_item,
    extract_scene_seal_from_updates,
)

logger = logging.getLogger(__name__)


def _truncate_for_log(value: Any, limit: int = 120) -> str:
    text = str(value)
    text = text.replace("\n", " ")
    if len(text) > limit:
        return text[: limit - 1] + "\u2026"
    return text


def _summarize_operations(delta: Any) -> List[str]:
    summaries: List[str] = []
    for op in getattr(delta, "operations", []) or []:
        op_type = getattr(op, "type", None) or "unknown"
        if op_type == "npc.move":
            destination = op.location_slug or (
                f"id={op.location_id}" if getattr(op, "location_id", None) else "unknown"
            )
            summaries.append(
                f"npc.move#{getattr(op, 'npc_id', 'unknown')}->{destination}"
            )
        elif op_type == "player.move":
            destination = op.location_slug or (
                f"id={op.location_id}" if getattr(op, "location_id", None) else "unknown"
            )
            subject = getattr(op, "player_id", None) or "player"
            summaries.append(f"player.move#{subject}->{destination}")
        elif op_type == "relationship.bump":
            summaries.append(
                "relationship.bump#"
                f"{getattr(op, 'source_type', '?')}:{getattr(op, 'source_id', '?')}->"
                f"{getattr(op, 'target_type', '?')}:{getattr(op, 'target_id', '?')}"
                f" (\u0394{getattr(op, 'delta', '?')})"
            )
        elif op_type == "narrative.append":
            summaries.append(
                "narrative.append:" + _truncate_for_log(getattr(op, "text", ""), 40)
            )
        else:
            summaries.append(op_type)
    return summaries

# -------------------------------------------------------------------------------
# Helper functions for robust JSON handling
# -------------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.startswith("```"):
        # strip first fence line
        t = t.split("\n", 1)[1] if "\n" in t else ""
        # strip trailing fence
        if t.rstrip().endswith("```"):
            t = t.rsplit("```", 1)[0]
    return t.strip()

def _to_updates_json(update_data) -> str:
    """
    Accepts pydantic v1/v2 models, dicts, lists, or JSON strings.
    Returns a JSON string (never empty). Raises ValueError if impossible.
    """
    # Pydantic v2
    if hasattr(update_data, "model_dump"):
        payload = update_data.model_dump(exclude_none=True)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Pydantic v1
    if hasattr(update_data, "dict"):
        payload = update_data.dict()
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Raw dict/list
    if isinstance(update_data, (dict, list)):
        return json.dumps(update_data, ensure_ascii=False, separators=(",", ":"))
    # String-ish
    if isinstance(update_data, str):
        s = _strip_code_fences(update_data)
        if not s or not s.strip():
            raise ValueError("Empty JSON string from agent")
        # validate it is JSON; if not, try to normalize later
        json.loads(s)
        return s
    raise ValueError(f"Unsupported updater output type: {type(update_data)}")


async def _invoke_function_tool(tool, ctx, **kwargs):
    """
    Call a FunctionTool across SDK variants:
    - Prefer .invoke(ctx, **kwargs)
    - Fall back to .run(ctx, **kwargs)
    - Fall back to Runner.run_tool(...) if available
    - As a last resort, try calling it directly if it's still a coroutine fn
    """
    # Common modern path
    if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
        return await tool.invoke(ctx, **kwargs)
    if hasattr(tool, "run") and callable(getattr(tool, "run")):
        return await tool.run(ctx, **kwargs)
    # Some SDKs expose a Runner helper
    try:
        from agents.run import Runner as _R
        if hasattr(_R, "run_tool"):
            return await _R.run_tool(tool, ctx=ctx, **kwargs)
    except Exception:
        pass
    # Very old decorator returns a bare async fn
    if callable(tool):
        return await tool(ctx, **kwargs)
    raise TypeError("Cannot invoke FunctionTool with this SDK build.")


def _build_min_skeleton(user_id: int, conversation_id: int, narrative: str) -> dict:
    """Return a valid empty UniversalUpdateInput payload."""
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "narrative": narrative or "",
        "roleplay_updates": [],
        "ChaseSchedule": [],
        "MainQuest": None,
        "PlayerRole": None,
        "npc_creations": [],
        "npc_updates": [],
        "character_stat_updates": {"player_name": "Chase", "stats": []},
        "relationship_updates": [],
        "npc_introductions": [],
        "location_creations": [],
        "event_list_updates": [],
        "inventory_updates": {"player_name": "Chase", "added_items": [], "removed_items": []},
        "quest_updates": [],
        "social_links": [],
        "perk_unlocks": [],
        "activity_updates": [],
        "journal_updates": [],
        "image_generation": {"generate": False, "priority": "low", "focus": "balanced", "framing": "medium_shot"},
    }

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

    def _resolve_field(item: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in item:
                return candidate
        lowered_map = {
            str(actual_key).lower(): actual_key
            for actual_key in item.keys()
            if isinstance(actual_key, str)
        }
        for candidate in candidates:
            lowered_candidate = candidate.lower()
            if lowered_candidate in lowered_map:
                return lowered_map[lowered_candidate]
        return None

    result: Dict[Any, Any] = {}
    key_candidates = []
    for candidate in (key_name, "key", "field", "name"):
        if isinstance(candidate, str) and candidate not in key_candidates:
            key_candidates.append(candidate)

    value_candidates = []
    for candidate in (value_name, "value"):
        if isinstance(candidate, str) and candidate not in value_candidates:
            value_candidates.append(candidate)

    for item in array_data:
        if not isinstance(item, dict):
            continue

        key_field = _resolve_field(item, key_candidates)
        value_field = _resolve_field(item, value_candidates)

        if key_field is None or value_field is None:
            continue

        result[item[key_field]] = item[value_field]

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
    """Runtime-strict models, but JSON Schema sanitized for Agents."""
    model_config = ConfigDict(extra='forbid')

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)

        def _strip(d):
            if isinstance(d, dict):
                d.pop("additionalProperties", None)
                d.pop("unevaluatedProperties", None)
                for v in d.values():
                    _strip(v)
            elif isinstance(d, list):
                for v in d:
                    _strip(v)
        _strip(schema)
        return schema

# Alias for compatibility
StrictBaseModel = BaseModel

# ===== Utility Types for Strict Schema =====
JsonScalar = Union[str, int, float, bool, None]

class KeyValueStr(StrictBaseModel):
    """String key-value pair"""
    key: str
    value: str

class KeyValueInt(StrictBaseModel):
    """Integer key-value pair"""
    key: str
    value: int

# Schedule models using array format
class DailySchedule(StrictBaseModel):
    """Schedule for a single day"""
    Morning: Optional[str] = None
    Afternoon: Optional[str] = None
    Evening: Optional[str] = None
    Night: Optional[str] = None

class ScheduleEntry(StrictBaseModel):
    """Single schedule entry in array format"""
    key: str  # Day name (Monday, Tuesday, etc.)
    value: DailySchedule

# Now that DailySchedule exists, define the strict KeyValuePair
JsonList = List[JsonScalar]
JsonValue = Union[JsonScalar, JsonList, DailySchedule]

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
        json_str = _strip_code_fences(json_str)
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
            bad = _strip_code_fences(json_str or "")
            logging.error(f"Failed to normalize JSON: {e}; preview: {bad[:200]!r}")
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

async def _apply_universal_updates_impl(ctx: RunContextWrapper, updates_json: str) -> ApplyUpdatesResult:
    """
    Actual implementation you can call directly from Python.
    The @function_tool wrapper below just delegates to this.
    """
    # Accept dict/list directly; strings get parsed (with code-fence stripping)
    raw = updates_json
    if isinstance(raw, (dict, list)):
        updates = raw
        s = None
    else:
        s = _strip_code_fences(raw if isinstance(raw, str) else "")
        try:
            updates = json.loads(s)
        except json.JSONDecodeError:
            bad = _strip_code_fences(updates_json or "")
            logger.error("Invalid JSON from updater agent (preview): %r", bad[:200])
            # Use normalize_json tool (robustly) but we can also inline a fallback
            try:
                normalized = await _invoke_function_tool(normalize_json, ctx, json_str=s)
            except Exception:
                normalized = None
            if normalized and getattr(normalized, "ok", False) and getattr(normalized, "data", None):
                updates = array_to_dict([item.model_dump() for item in normalized.data])
            else:
                return ApplyUpdatesResult(
                    success=False,
                    error=f"Invalid JSON: {(normalized and normalized.error) or 'Unknown error'}"
                )

    try:
        preview = (s if isinstance(raw, str) else json.dumps(raw))[:200]
        logging.debug(f"[apply_universal_updates] Incoming updates preview: {preview}")
    except Exception:
        pass
    
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

@function_tool
async def apply_universal_updates(ctx: RunContextWrapper, updates_json: str) -> ApplyUpdatesResult:
    """
    Thin tool wrapper that forwards to the real implementation.
    Agents can call this tool; your Python code should call the impl directly.
    """
    return await _apply_universal_updates_impl(ctx, updates_json)

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
    """Apply universal updates by emitting a typed canonical delta."""

    del ctx  # retained for call compatibility

    payload = dict(updates or {})
    seal_from_updates = extract_scene_seal_from_updates(payload)

    try:
        delta = build_delta_from_legacy_payload(
            user_id=user_id,
            conversation_id=conversation_id,
            payload=payload,
        )
    except (DeltaBuildError, ValidationError) as exc:
        logger.error("Failed to construct canonical delta: %s", exc)
        return {"success": False, "error": str(exc)}

    operation_summaries = _summarize_operations(delta)
    logger.info(
        "Emitting canon delta request_id=%s operations=%d summary=%s",
        delta.request_id,
        delta.operation_count,
        operation_summaries,
    )

    try:
        db_result = await db_rpc.write_event(conn, delta)
    except db_rpc.CanonEventError as exc:
        logger.error("canon.apply_event failed: %s", exc)
        return {"success": False, "error": str(exc)}

    applied_flag = db_result.get("applied") if isinstance(db_result, dict) else None
    raw_messages = db_result.get("messages") if isinstance(db_result, dict) else None
    if isinstance(raw_messages, (list, tuple)):
        truncated_messages = [_truncate_for_log(msg, 200) for msg in raw_messages]
    elif raw_messages is None:
        truncated_messages = None
    else:
        truncated_messages = _truncate_for_log(raw_messages, 200)

    logger.info(
        "canon.apply_event response for request_id=%s: applied=%s messages=%s",
        delta.request_id,
        applied_flag,
        truncated_messages,
    )

    if seal_from_updates and db_result.get("applied"):
        logger.info(
            "Ensuring scene seal for conversation=%s venue=%s date=%s request_id=%s",
            conversation_id,
            seal_from_updates.get("venue"),
            seal_from_updates.get("date"),
            delta.request_id,
        )
        await ensure_scene_seal_item(
            conn,
            conversation_id=conversation_id,
            venue=seal_from_updates.get("venue"),
            date=seal_from_updates.get("date"),
            non_negotiables=seal_from_updates.get("non_negotiables"),
            source="universal_updates",
        )

    return {
        "success": True,
        "updates_applied": delta.operation_count,
        "details": db_result,
        "request_id": str(delta.request_id),
    }


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
            Allow adult themes within the context of a hardcore femdom relationship.
            """,
            output_type=ContentSafety,
            model='gpt-5-nano'
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
    model='gpt-5-nano'
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
    output_type=AgentOutputSchema(UniversalUpdateInput, strict_json_schema=False),
    input_guardrails=[
        InputGuardrail(guardrail_function=content_safety_guardrail),
    ],
    model_settings=ModelSettings(
        response_format="json_object",
    ),
    model='gpt-5-nano'
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
    # Safeguard against None IDs
    user_id_str = str(user_id if user_id is not None else 0)
    conversation_id_str = str(conversation_id if conversation_id is not None else 0)
    
    # Ensure narrative is never empty
    if not narrative or not narrative.strip():
        logger.warning("Empty narrative passed to universal updater, using fallback")
        narrative = "The scene continues..."
    
    # Log what we're processing for debugging
    logger.debug(
        "Universal updater processing: user_id=%s, conversation_id=%s, narrative_len=%d, preview=%r",
        user_id_str, conversation_id_str, len(narrative), narrative[:100]
    )
    
    # Create and initialize the updater context
    updater_context = UniversalUpdaterContext(user_id, conversation_id)
    await updater_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Universal Update",
        trace_id=f"trace_universal_update_{conversation_id_str}_{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id_str}"
    ):
        # Build prompt with actual narrative content
        prompt = f"""
Analyze the narrative and return ONLY a single JSON object matching UniversalUpdateInput.
Do not include any text before or after the JSON. No markdown fences.

USER_ID: {user_id_str}
CONVERSATION_ID: {conversation_id_str}

NARRATIVE:
{narrative}

Example shape:
{{
  "user_id": {user_id_str},
  "conversation_id": {conversation_id_str},
  "narrative": {json.dumps(narrative[:60] + ("..." if len(narrative) > 60 else ""))},
  "roleplay_updates": [],
  "ChaseSchedule": [],
  "MainQuest": null,
  "PlayerRole": null,
  "npc_creations": [],
  "npc_updates": [],
  "character_stat_updates": {{"player_name":"Chase","stats":[]}},
  "relationship_updates": [],
  "npc_introductions": [],
  "location_creations": [],
  "event_list_updates": [],
  "inventory_updates": {{"player_name":"Chase","added_items":[],"removed_items":[]}},
  "quest_updates": [],
  "social_links": [],
  "perk_unlocks": [],
  "activity_updates": [],
  "journal_updates": [],
  "image_generation": {{"generate": false,"priority":"low","focus":"balanced","framing":"medium_shot"}}
}}
        """
        
        try:
            logger.debug("Running universal updater agent with prompt length: %d", len(prompt))
            
            # Run the agent to extract updates
            result = await Runner.run(
                universal_updater_agent,
                prompt,
                context=updater_context
            )
            
            # Try several paths to get structured output
            update_data = None
            try:
                update_data = result.final_output_as(UniversalUpdateInput)
                logger.debug("Successfully extracted structured output from result")
            except Exception as e:
                logger.debug("Could not extract as UniversalUpdateInput: %s", e)
                update_data = getattr(result, "final_output", None)
            
            # Convert whatever we got into JSON; try multiple fallbacks
            update_json = None
            if update_data:
                try:
                    update_json = _to_updates_json(update_data)
                    logger.debug("Successfully converted update_data to JSON")
                except Exception as e:
                    logger.error("Updater output could not be normalized to JSON: %s", e)
            
            if not update_json:
                # Try to extract raw text from various result attributes
                raw_txt = (
                    getattr(result, "output_text", None)
                    or getattr(result, "completion_text", None)
                    or (getattr(getattr(result, "response", None), "output_text", None))
                    or ""
                )
                raw_txt = _strip_code_fences(raw_txt)
                if raw_txt:
                    try:
                        json.loads(raw_txt)
                        update_json = raw_txt
                        logger.debug("Successfully parsed raw text as JSON")
                    except Exception:
                        logger.error("Invalid JSON from updater agent (preview): %r", raw_txt[:200])
            
            if not update_json:
                logger.error("Updater returned empty output. Using minimal skeleton to continue.")
                skel = _build_min_skeleton(int(user_id_str), int(conversation_id_str), narrative)
                update_json = json.dumps(skel, ensure_ascii=False, separators=(",", ":"))
                logger.debug("Created skeleton update with %d keys", len(skel))
            
            # Log what we're about to apply
            try:
                update_preview = json.loads(update_json) if isinstance(update_json, str) else update_json
                update_count = sum(
                    len(v) if isinstance(v, list) else (1 if v else 0)
                    for k, v in update_preview.items()
                    if k not in ["user_id", "conversation_id", "narrative"]
                )
                logger.debug("Applying %d updates from universal updater", update_count)
            except Exception:
                logger.debug("Could not count updates in preview")
            
            # Call the impl directly (no SDK tool invocation needed)
            wrapped_ctx = RunContextWrapper(updater_context)
            update_result = await _apply_universal_updates_impl(
                wrapped_ctx, update_json
            )
            
            # Log result
            result_dict = update_result.model_dump() if hasattr(update_result, 'model_dump') else dict(update_result)
            logger.info(
                "Universal update complete: success=%s, updates_applied=%s",
                result_dict.get("success"),
                result_dict.get("updates_applied", 0)
            )
            
            return result_dict
                
        except Exception as e:
            logger.error("Error in universal updater agent execution: %s", str(e), exc_info=True)
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
