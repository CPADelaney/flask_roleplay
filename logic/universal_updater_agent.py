# logic/universal_updater_agent.py
"""
Universal Updater SDK using OpenAI's Agents SDK with Nyx Governance integration.

REFRESHED (location-robust):
- convert_updates_for_database: mirrors location into legacy keys (CurrentLocation, etc.)
- apply_universal_updates_async: invalidates aggregated context cache after canon write

This module analyzes narrative text and extracts appropriate game state updates.
It supports the new array format for structures that used to be dicts while
remaining backward compatible with DB storage expectations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Literal

# OpenAI Agents SDK-style interfaces
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
    handoff,
)

# DB connection
from db.connection import get_db_connection_context
from db import rpc as db_rpc

# Lore / governance
from lore.core import canon
from lore.core.lore_system import LoreSystem
from nyx.integrate import get_central_governance
from nyx.nyx_governance import NyxUnifiedGovernor, AgentType, DirectiveType, DirectivePriority

# Delta builder (typed canonical ops)
from logic.universal_delta import build_delta_from_legacy_payload, DeltaBuildError

# Scene helpers
from openai_integration.conversations import (
    ensure_scene_seal_item,
    extract_scene_seal_from_updates,
)

logger = logging.getLogger(__name__)

DEFAULT_NARRATIVE_FALLBACK = "The scene continues..."

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    """Remove ``` fences that models sometimes add around JSON."""
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

def _ensure_narrative(payload: Dict[str, Any], fallback: str = DEFAULT_NARRATIVE_FALLBACK) -> Dict[str, Any]:
    """Ensure a payload has a non-empty narrative value."""
    if isinstance(payload, dict):
        narrative = payload.get("narrative")
        if not isinstance(narrative, str) or not narrative.strip():
            payload["narrative"] = fallback
    return payload


def _to_updates_json(update_data: Any, fallback_narrative: str = DEFAULT_NARRATIVE_FALLBACK) -> str:
    """
    Accept pydantic v1/v2 models, dicts, lists, or JSON strings.
    Return a JSON string; raise ValueError if impossible.
    """
    # Pydantic v2
    if hasattr(update_data, "model_dump"):
        payload = update_data.model_dump(exclude_none=True)
        _ensure_narrative(payload, fallback_narrative)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Pydantic v1
    if hasattr(update_data, "dict"):
        payload = update_data.dict()
        _ensure_narrative(payload, fallback_narrative)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Raw dict/list
    if isinstance(update_data, (dict, list)):
        if isinstance(update_data, dict):
            _ensure_narrative(update_data, fallback_narrative)
        return json.dumps(update_data, ensure_ascii=False, separators=(",", ":"))
    # String-ish
    if isinstance(update_data, str):
        s = _strip_code_fences(update_data)
        if not s or not s.strip():
            raise ValueError("Empty JSON string from agent")
        json.loads(s)  # validate
        return s
    raise ValueError(f"Unsupported updater output type: {type(update_data)}")

def _first(*vals):
    """Return the first non-empty value."""
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None

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
            destination = op.location_slug or (f"id={op.location_id}" if getattr(op, "location_id", None) else "unknown")
            summaries.append(f"npc.move#{getattr(op, 'npc_id', 'unknown')}->{destination}")
        elif op_type == "player.move":
            destination = op.location_slug or (f"id={op.location_id}" if getattr(op, "location_id", None) else "unknown")
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
            summaries.append("narrative.append:" + _truncate_for_log(getattr(op, "text", ""), 40))
        else:
            summaries.append(op_type)
    return summaries

async def _invoke_function_tool(tool, ctx, **kwargs):
    """
    Call a FunctionTool across SDK variants.
    """
    on_invoke = getattr(tool, "on_invoke_tool", None)
    if callable(on_invoke):
        payload = kwargs or {}
        return await on_invoke(ctx, payload)
    if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
        return await tool.invoke(ctx, **kwargs)
    if hasattr(tool, "run") and callable(getattr(tool, "run")):
        return await tool.run(ctx, **kwargs)
    try:
        from agents.run import Runner as _R
        if hasattr(_R, "run_tool"):
            return await _R.run_tool(tool, ctx=ctx, **kwargs)
    except Exception:
        pass
    if callable(tool):
        return await tool(ctx, **kwargs)
    raise TypeError("Cannot invoke FunctionTool with this SDK build.")

# ------------------------------------------------------------------------------
# Array <-> Dict helpers
# ------------------------------------------------------------------------------

def array_to_dict(array_data: List[Dict[str, Any]], key_name: str = "key", value_name: str = "value") -> Dict[str, Any]:
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
    if not isinstance(obj_data, dict):
        return []
    return [{key_name: k, value_name: v} for k, v in obj_data.items()]

# ------------------------------------------------------------------------------
# Pydantic models (minimal set retained from your version)
# ------------------------------------------------------------------------------

# Runtime-strict models, JSON schema sanitized by your Agent glue; leave as BaseModel
StrictBaseModel = BaseModel

JsonScalar = Union[str, int, float, bool, None]

class KeyValuePair(StrictBaseModel):
    key: str
    value: Union[JsonScalar, Dict[str, Any], List[JsonScalar]]

class DailySchedule(StrictBaseModel):
    Morning: Optional[str] = None
    Afternoon: Optional[str] = None
    Evening: Optional[str] = None
    Night: Optional[str] = None

class ScheduleEntry(StrictBaseModel):
    key: str  # Day
    value: DailySchedule

class NPCArchetype(StrictBaseModel):
    id: Optional[int] = None
    name: Optional[str] = None

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
    archetypes: List[NPCArchetype] = Field(default_factory=list)
    archetype_summary: Optional[str] = None
    archetype_extras_summary: Optional[str] = None
    physical_description: Optional[str] = None
    hobbies: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    schedule: List[ScheduleEntry] = Field(default_factory=list)
    memory: Union[str, List[str], None] = None
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
    memory: Union[str, List[str], None] = None
    schedule: Optional[List[ScheduleEntry]] = None
    schedule_updates: Optional[List[ScheduleEntry]] = None
    affiliations: Optional[List[str]] = None
    current_location: Optional[str] = None

class NPCIntroduction(StrictBaseModel):
    npc_id: int

class StatEntry(StrictBaseModel):
    key: str
    value: int

class CharacterStatUpdates(StrictBaseModel):
    player_name: str = "Chase"
    stats: List[StatEntry] = Field(default_factory=list)

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

class InventoryRemovedItem(StrictBaseModel):
    name: str

class InventoryUpdates(StrictBaseModel):
    player_name: str = "Chase"
    added_items: List[Union[str, InventoryItem]] = Field(default_factory=list)
    removed_items: List[Union[str, InventoryRemovedItem]] = Field(default_factory=list)

class ActivityPurpose(StrictBaseModel):
    description: Optional[str] = None
    fantasy_level: str = "realistic"

class StatIntegrationEntry(StrictBaseModel):
    key: str
    value: JsonScalar

class Activity(StrictBaseModel):
    activity_name: str
    purpose: Optional[ActivityPurpose] = None
    stat_integration: List[StatIntegrationEntry] = Field(default_factory=list)
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
    narrative: Optional[str] = None
    roleplay_updates: List[KeyValuePair] = Field(default_factory=list)
    ChaseSchedule: List[ScheduleEntry] = Field(default_factory=list)
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
    perk_unlocks: List[Dict[str, Any]] = Field(default_factory=list)  # simplified
    activity_updates: List[Activity] = Field(default_factory=list)
    journal_updates: List[JournalEntry] = Field(default_factory=list)
    image_generation: Optional[ImageGeneration] = None

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
    is_appropriate: bool
    reasoning: str
    suggested_adjustment: Optional[str] = None

# ------------------------------------------------------------------------------
# Agent Context
# ------------------------------------------------------------------------------

class UniversalUpdaterContext:
    """Context object for universal updater agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor: Optional[NyxUnifiedGovernor] = None
        self.lore_system: Optional[LoreSystem] = None

    async def initialize(self):
        """Initialize context with governance integration"""
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        self.lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)

# ------------------------------------------------------------------------------
# Format conversions (UPDATED: location mirroring for DB)
# ------------------------------------------------------------------------------

def convert_updates_for_database(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert array formats back to dicts for database storage.
    Adds robust location mirroring into legacy keys so readers always see a location.
    """
    db_updates = dict(updates or {})

    # roleplay_updates: array -> dict + location mirroring
    if isinstance(db_updates.get("roleplay_updates"), list):
        rp = array_to_dict(db_updates["roleplay_updates"])

        # Support both flat and nested location payloads
        loc_obj = rp.get("location") if isinstance(rp.get("location"), dict) else None

        candidate_display = _first(
            rp.get("CurrentLocation"),
            rp.get("current_location"),
            rp.get("currentLocation"),
            rp.get("Location"),
            (loc_obj or {}).get("name"),
            (loc_obj or {}).get("label"),
            (loc_obj or {}).get("display"),
            (loc_obj or {}).get("slug"),
        )
        candidate_id = _first(
            rp.get("CurrentLocationId"),
            rp.get("current_location_id"),
            rp.get("currentLocationId"),
            rp.get("LocationId"),
            (loc_obj or {}).get("id"),
            (loc_obj or {}).get("location_id"),
        )

        # Mirror into canonical/legacy slots expected by DB readers
        if candidate_display:
            rp.setdefault("CurrentLocation", candidate_display)
            rp.setdefault("current_location", candidate_display)
        if candidate_id is not None:
            rp.setdefault("CurrentLocationId", candidate_id)
            rp.setdefault("current_location_id", candidate_id)

        db_updates["roleplay_updates"] = rp

    # ChaseSchedule array -> dict
    if isinstance(db_updates.get("ChaseSchedule"), list):
        db_updates["ChaseSchedule"] = array_to_dict(db_updates["ChaseSchedule"])

    # character_stat_updates.stats array -> dict
    csu = db_updates.get("character_stat_updates")
    if isinstance(csu, dict) and isinstance(csu.get("stats"), list):
        csu["stats"] = array_to_dict(csu["stats"])
        db_updates["character_stat_updates"] = csu

    # NPC schedules array -> dict
    for npc in db_updates.get("npc_creations", []) or []:
        if isinstance(npc.get("schedule"), list):
            npc["schedule"] = array_to_dict(npc["schedule"])
    for npc in db_updates.get("npc_updates", []) or []:
        if isinstance(npc.get("schedule"), list):
            npc["schedule"] = array_to_dict(npc["schedule"])
        if isinstance(npc.get("schedule_updates"), list):
            npc["schedule_updates"] = array_to_dict(npc["schedule_updates"])

    # Activities stat_integration array -> dict
    for act in db_updates.get("activity_updates", []) or []:
        if isinstance(act.get("stat_integration"), list):
            act["stat_integration"] = array_to_dict(act["stat_integration"])

    return db_updates

def convert_from_database_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """(unchanged behavior) dict -> array for schema compliance."""
    result = dict(data or {})

    if isinstance(result.get("roleplay_updates"), dict):
        result["roleplay_updates"] = dict_to_array(result["roleplay_updates"])
    if isinstance(result.get("ChaseSchedule"), dict):
        result["ChaseSchedule"] = dict_to_array(result["ChaseSchedule"])

    csu = result.get("character_stat_updates")
    if isinstance(csu, dict) and isinstance(csu.get("stats"), dict):
        csu["stats"] = dict_to_array(csu["stats"])

    for npc in result.get("npc_creations", []) or []:
        if isinstance(npc.get("schedule"), dict):
            npc["schedule"] = dict_to_array(npc["schedule"])
    for npc in result.get("npc_updates", []) or []:
        if isinstance(npc.get("schedule"), dict):
            npc["schedule"] = dict_to_array(npc["schedule"])
        if isinstance(npc.get("schedule_updates"), dict):
            npc["schedule_updates"] = dict_to_array(npc["schedule_updates"])

    for act in result.get("activity_updates", []) or []:
        if isinstance(act.get("stat_integration"), dict):
            act["stat_integration"] = dict_to_array(act["stat_integration"])

    return result

# ------------------------------------------------------------------------------
# Function tools
# ------------------------------------------------------------------------------

@function_tool
async def normalize_json(ctx: RunContextWrapper, json_str: str) -> NormalizedJson:
    """Normalize JSON string, fixing common errors."""
    try:
        json_str = _strip_code_fences(json_str)
        data = json.loads(json_str)
        if isinstance(data, dict):
            data = dict_to_array(data)
        elif not isinstance(data, list):
            data = []
        return NormalizedJson(ok=True, data=data)
    except json.JSONDecodeError:
        normalized = (
            json_str.replace("\u201c", '"').replace("\u201d", '"')
                    .replace("\u2018", "'").replace("\u2019", "'")
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
            return NormalizedJson(ok=False, error="Failed to parse JSON", message=str(e), original=json_str)

@function_tool
async def check_npc_exists(ctx: RunContextWrapper, npc_id: int) -> bool:
    """Check if an NPC with the given ID exists in the database (with governance permission)."""
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor

    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="check_npc_exists",
        action_details={"npc_id": npc_id},
    )
    if not permission["approved"]:
        return False

    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                "SELECT npc_id FROM NPCStats WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3",
                npc_id, user_id, conversation_id
            )
            exists = row is not None
            await governor.process_agent_action_report(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="universal_updater",
                action={"type": "check_npc_exists", "npc_id": npc_id},
                result={"exists": exists},
            )
            return exists
    except Exception as e:
        logging.error(f"Error checking if NPC exists: {e}")
        return False

@function_tool
async def extract_player_stats(ctx: RunContextWrapper, narrative: str) -> PlayerStatsExtraction:
    """Very light heuristic extraction of player stat mentions."""
    stats = ["corruption", "confidence", "willpower", "obedience", "dependency", "lust", "mental_resilience", "physical_endurance"]
    governor = ctx.context.governor

    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="extract_player_stats",
        action_details={"narrative_length": len(narrative)},
    )
    if not permission["approved"]:
        return PlayerStatsExtraction(player_name="Chase", stats=[])

    changes = {}
    low = narrative.lower()
    for stat in stats:
        if f"{stat} increase" in low or f"{stat} rose" in low:
            changes[stat] = 5
        elif f"{stat} decrease" in low or f"{stat} drop" in low:
            changes[stat] = -5

    await governor.process_agent_action_report(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action={"type": "extract_player_stats"},
        result={"stats_changed": len(changes)},
    )

    return PlayerStatsExtraction(
        player_name="Chase",
        stats=[StatEntry(key=k, value=v) for k, v in changes.items()],
    )

async def _apply_universal_updates_impl(ctx: RunContextWrapper, updates_json: str) -> ApplyUpdatesResult:
    """
    Actual implementation invoked by the tool and by process_universal_update.
    Includes robust JSON handling and DB write.
    """
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
            try:
                normalized = await _invoke_function_tool(normalize_json, ctx, json_str=s)
            except Exception:
                normalized = None
            if normalized and getattr(normalized, "ok", False) and getattr(normalized, "data", None):
                updates = array_to_dict([item.model_dump() for item in normalized.data])
            else:
                return ApplyUpdatesResult(success=False, error=f"Invalid JSON: {(normalized and normalized.error) or 'Unknown error'}")

    updates_payload: Dict[str, Any] = updates if isinstance(updates, dict) else {}
    _ensure_narrative(updates_payload, DEFAULT_NARRATIVE_FALLBACK)

    try:
        preview = (s if isinstance(raw, str) else json.dumps(raw))[:200]
        logging.debug(f"[apply_universal_updates] Incoming updates preview: {preview}")
    except Exception:
        pass

    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor

    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        action_type="apply_updates",
        action_details={"update_count": sum(len(updates.get(k, [])) for k in updates if isinstance(updates.get(k), list))},
    )
    if not permission["approved"]:
        return ApplyUpdatesResult(success=False, reason=permission["reasoning"])

    try:
        # **IMPORTANT**: unify array formats and mirror location to legacy keys
        if isinstance(updates, dict):
            updates = dict(updates_payload)
        db_updates = convert_updates_for_database(updates)
        _ensure_narrative(db_updates, DEFAULT_NARRATIVE_FALLBACK)
        db_updates["user_id"] = user_id
        db_updates["conversation_id"] = conversation_id

        async with get_db_connection_context() as conn:
            result = await apply_universal_updates_async(ctx.context, user_id, conversation_id, db_updates, conn)

            await governor.process_agent_action_report(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="universal_updater",
                action={"type": "apply_updates"},
                result={"success": True, "updates_applied": result.get("updates_applied", 0)},
            )

            if "details" in result and isinstance(result["details"], dict):
                result["details"] = dict_to_array(result["details"])
            return ApplyUpdatesResult(**result)
    except Exception as e:
        logging.error(f"Error applying universal updates: {e}")
        return ApplyUpdatesResult(success=False, error=str(e))

@function_tool
async def apply_universal_updates(ctx: RunContextWrapper, updates_json: str) -> ApplyUpdatesResult:
    """Tool wrapper; delegates to the implementation."""
    return await _apply_universal_updates_impl(ctx, updates_json)

# ------------------------------------------------------------------------------
# Database Processing (UPDATED: cache invalidation on success)
# ------------------------------------------------------------------------------

async def apply_universal_updates_async(
    ctx: UniversalUpdaterContext,
    user_id: int,
    conversation_id: int,
    updates: Dict[str, Any],
    conn: asyncpg.Connection,
) -> Dict[str, Any]:
    """Apply universal updates by emitting a typed canonical delta to canon."""
    del ctx  # retained for signature compatibility

    payload = dict(updates or {})
    seal_from_updates = extract_scene_seal_from_updates(payload)

    # Build canonical delta from legacy-ish payload
    try:
        delta = build_delta_from_legacy_payload(
            user_id=user_id,
            conversation_id=conversation_id,
            payload=payload,
        )
    except ValidationError as exc:
        logger.error("Failed to construct canonical delta: %s", exc)
        return {"success": False, "error": str(exc)}
    except DeltaBuildError as exc:
        message = str(exc)
        if "no canonical operations could be extracted" in message.lower():
            logger.info(
                "Universal updater produced no canonical ops; treating as noop for conversation=%s",
                conversation_id,
            )
            return {"success": True, "updates_applied": 0, "reason": message}
        logger.error("Failed to construct canonical delta: %s", exc)
        return {"success": False, "error": message}

    operation_summaries = _summarize_operations(delta)
    logger.info(
        "Emitting canon delta request_id=%s operations=%d summary=%s",
        delta.request_id,
        delta.operation_count,
        operation_summaries,
    )

    # Persist event via RPC
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
        "canon.apply_event response request_id=%s: applied=%s messages=%s",
        delta.request_id,
        applied_flag,
        truncated_messages,
    )

    # Optional: ensure scene seal item
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

    # **NEW**: Invalidate context caches so location/time refresh immediately
    try:
        # Import lazily to avoid circular import at module load
        from logic.aggregator_sdk import context_cache as _agg_cache
        if _agg_cache:
            _agg_cache.invalidate(f"context:{user_id}:{conversation_id}")
    except Exception:
        logger.debug("Context cache invalidation skipped (no cache or older runtime)")

    return {
        "success": True,
        "updates_applied": delta.operation_count,
        "details": db_result,
        "request_id": str(delta.request_id),
    }

# ------------------------------------------------------------------------------
# Guardrail
# ------------------------------------------------------------------------------

async def content_safety_guardrail(ctx, agent, input_data):
    """Very light input guardrail example (leave permissive for your domain)."""
    try:
        content_moderator = Agent(
            name="Content Moderator",
            instructions="""
            You check if content is appropriate for a femdom roleplay game.
            Allow adult themes within the context of a consensual hardcore femdom relationship.
            """,
            output_type=ContentSafety,
            model='gpt-5-nano',
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
                suggested_adjustment=None,
            ),
            tripwire_triggered=False,
        )

# ------------------------------------------------------------------------------
# Agent definitions
# ------------------------------------------------------------------------------

extraction_agent = Agent[UniversalUpdaterContext](
    name="StateExtractor",
    instructions="""
    Extract concrete, explicit state changes from narrative. Avoid over-inference.
    """,
    tools=[extract_player_stats, check_npc_exists],
    output_type=str,
    model='gpt-5-nano',
)

universal_updater_agent = Agent[UniversalUpdaterContext](
    name="UniversalUpdater",
    instructions="""
    Analyze narrative text and return ONLY a single JSON object (UniversalUpdateInput).
    Use array formats for lists of key/value pairs (roleplay_updates, schedules, stats, etc).
    """,
    tools=[normalize_json, check_npc_exists, extract_player_stats, apply_universal_updates],
    handoffs=[handoff(extraction_agent, tool_name_override="extract_state_changes")],
    output_type=AgentOutputSchema(UniversalUpdateInput, strict_json_schema=False),
    input_guardrails=[InputGuardrail(guardrail_function=content_safety_guardrail)],
    model_settings=ModelSettings(response_format="json_object"),
    model='gpt-5-nano',
)

# ------------------------------------------------------------------------------
# Primary public API
# ------------------------------------------------------------------------------

def _build_min_skeleton(user_id: int, conversation_id: int, narrative: str) -> dict:
    """Return a valid, empty UniversalUpdateInput payload."""
    narrative_fallback = narrative if narrative and narrative.strip() else DEFAULT_NARRATIVE_FALLBACK
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "narrative": narrative_fallback,
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

async def process_universal_update(
    user_id: int,
    conversation_id: int,
    narrative: str,
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Run the agent and apply updates with governance oversight."""
    user_id_str = str(user_id if user_id is not None else 0)
    conversation_id_str = str(conversation_id if conversation_id is not None else 0)

    if not narrative or not narrative.strip():
        logger.warning("Empty narrative passed to universal updater, using fallback")
        narrative = DEFAULT_NARRATIVE_FALLBACK

    updater_context = UniversalUpdaterContext(user_id, conversation_id)
    await updater_context.initialize()

    with trace(
        workflow_name="Universal Update",
        trace_id=f"trace_universal_update_{conversation_id_str}_{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id_str}",
    ):
        prompt = f"""
Analyze the narrative and return ONLY a single JSON object matching UniversalUpdateInput.
No prose, no markdown fences.

USER_ID: {user_id_str}
CONVERSATION_ID: {conversation_id_str}

NARRATIVE:
{narrative}
"""
        try:
            logger.debug("Running universal updater agent (prompt len=%d)", len(prompt))
            result = await Runner.run(universal_updater_agent, prompt, context=updater_context)

            update_data = None
            try:
                update_data = result.final_output_as(UniversalUpdateInput)
            except Exception as e:
                logger.debug("Could not coerce to UniversalUpdateInput directly: %s", e)
                update_data = getattr(result, "final_output", None)

            update_json = None
            if update_data:
                try:
                    update_json = _to_updates_json(update_data, fallback_narrative=narrative)
                except Exception as e:
                    logger.error("Updater output could not be normalized to JSON: %s", e)

            if not update_json:
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
                    except Exception:
                        logger.error("Invalid JSON from updater agent (preview): %r", raw_txt[:200])

            if not update_json:
                logger.error("Updater returned empty output. Using minimal skeleton to continue.")
                update_json = json.dumps(_build_min_skeleton(int(user_id_str), int(conversation_id_str), narrative))

            try:
                payload_dict: Dict[str, Any] = json.loads(update_json) if isinstance(update_json, str) else dict(update_json)
            except Exception as exc:
                logger.error("Updater payload was not a JSON object: %s", exc)
                payload_dict = _build_min_skeleton(int(user_id_str), int(conversation_id_str), narrative)

            _ensure_narrative(payload_dict, narrative)

            update_json = json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":"))

            # Apply
            wrapped_ctx = RunContextWrapper(updater_context)
            update_result = await _apply_universal_updates_impl(wrapped_ctx, update_json)

            result_dict = update_result.model_dump() if hasattr(update_result, 'model_dump') else dict(update_result)
            logger.info(
                "Universal update complete: success=%s, updates_applied=%s",
                result_dict.get("success"),
                result_dict.get("updates_applied", 0),
            )
            return result_dict

        except Exception as e:
            logger.error("Error in universal updater agent execution: %s", str(e), exc_info=True)
            return {"success": False, "error": f"Agent execution error: {str(e)}"}

# ------------------------------------------------------------------------------
# Governance registration / wrapper
# ------------------------------------------------------------------------------

async def register_with_governance(user_id: int, conversation_id: int):
    governor = await get_central_governance(user_id, conversation_id)
    await governor.register_agent(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_instance=universal_updater_agent,
        agent_id="universal_updater",
    )
    await governor.issue_directive(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="universal_updater",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Process narrative updates and extract game state changes",
            "scope": "game",
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24 * 60,
    )
    logging.info("Universal Updater registered with Nyx governance")

async def initialize_universal_updater(user_id: int, conversation_id: int) -> Dict[str, Any]:
    try:
        updater_agent = UniversalUpdaterAgent(user_id, conversation_id)
        await updater_agent.initialize()
        await register_with_governance(user_id, conversation_id)
        logging.info(f"Universal Updater initialized for user {user_id}, conversation {conversation_id}")
        return {"agent": updater_agent, "context": updater_agent.context, "status": "initialized"}
    except Exception as e:
        logging.error(f"Error initializing Universal Updater: {str(e)}", exc_info=True)
        return {"error": str(e), "status": "failed"}

class UniversalUpdaterAgent:
    """Compatibility wrapper for the OpenAI Agents SDK implementation of Universal Updater."""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context: Optional[UniversalUpdaterContext] = None
        self.initialized = False

    async def initialize(self):
        if not self.initialized:
            self.context = UniversalUpdaterContext(self.user_id, self.conversation_id)
            await self.context.initialize()
            self.initialized = True
        return self

    async def process_update(self, narrative: str, context: Dict[str, Any] = None):
        if not self.initialized:
            await self.initialize()
        return await process_universal_update(self.user_id, self.conversation_id, narrative, context)

    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        directive_type = directive.get("type")
        data = directive.get("data", {})
        if directive_type == "process_narrative":
            return await self.process_update(data.get("narrative", ""), data.get("context"))
        return {"success": False, "error": f"Unsupported directive type: {directive_type}"}

    async def get_capabilities(self) -> List[str]:
        return ["narrative_analysis", "state_extraction", "state_updating", "array_format_handling"]

    async def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "updates_processed": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0,
            "strategies": {},
            "array_format_support": True,
        }

    async def get_learning_state(self) -> Dict[str, Any]:
        return {"patterns": {}, "adaptations": [], "format_version": "array_v2"}
