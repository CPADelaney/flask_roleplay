# nyx/nyx_agent_sdk.py
"""
Nyx Agent SDK - Refactored to use OpenAI Agents SDK with Strict Typing Fixes

MODULARIZATION TODO: This file has grown to 2k+ lines and should be split:
- nyx/models.py - All Pydantic models and Config constants
- nyx/tools/*.py - Individual tool implementations
- nyx/agents/*.py - Agent definitions
- nyx/context.py - NyxContext and state management
- nyx/compat.py - Legacy AgentContext and compatibility layers

This module requires the following database tables to be created via migrations:
- NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
- scenario_states (user_id, conversation_id, state_data, created_at) 
  - INDEX: (user_id, conversation_id)
- learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
- performance_metrics (user_id, conversation_id, metrics, error_log, created_at)

For continuous monitoring (scenario updates, resource usage, etc.), implement an external service using:
- Celery for background tasks
- FastAPI background tasks
- Kubernetes CronJobs
- Or a dedicated monitoring service

This keeps the main request path fast and non-blocking.
"""

import logging
import json
import asyncio
import os
import time
import copy
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Literal
from story_agent.slice_of_life_narrator import narrate_slice_of_life_scene as narrator_narrate_scene
from dataclasses import dataclass, field
from contextlib import suppress, asynccontextmanager
import statistics
import uuid

logger = logging.getLogger(__name__)

# Import world simulation models if available
try:
    from story_agent.world_simulation_models import (
        CompleteWorldState,
        WorldState,
        WorldMood,
        TimeOfDay,
        ActivityType,
        PowerDynamicType,
        SliceOfLifeEvent,
        PowerExchange,
        WorldTension,
        RelationshipDynamics,
        NPCRoutine,
        CurrentTimeData,
        VitalsData,
        AddictionCravingData,
        DreamData,
        RevelationData,
        ChoiceData,
        ChoiceProcessingResult,
    )

    from story_agent.world_director_agent import (
        CompleteWorldDirector,
        WorldDirector,
        CompleteWorldDirectorContext,
        WorldDirectorContext,
    )
except ImportError:
    logger.warning("World simulation models not available - slice-of-life features disabled")

# ===== CRITICAL FIX #1: Monkey patch Pydantic BEFORE any imports =====
import pydantic.json_schema

_original_model_json_schema = pydantic.json_schema.model_json_schema

def _patched_model_json_schema(*args, **kwargs):
    """Patched version that removes additionalProperties"""
    schema = _original_model_json_schema(*args, **kwargs)
    
    def strip_additional_properties(obj):
        if isinstance(obj, dict):
            obj.pop('additionalProperties', None)
            obj.pop('unevaluatedProperties', None)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    strip_additional_properties(v)
        elif isinstance(obj, list):
            for item in obj:
                strip_additional_properties(item)
        return obj
    
    return strip_additional_properties(schema)

# Apply the monkey patch globally
pydantic.json_schema.model_json_schema = _patched_model_json_schema

# Now import agents and Pydantic
from agents import (
    Agent, Runner, function_tool, handoff,
    ModelSettings, GuardrailFunctionOutput, InputGuardrail,
    RunContextWrapper, RunConfig
)

from lore.core.canon import (
    find_or_create_npc,
    find_or_create_location, 
    find_or_create_event,
    log_canonical_event,
    create_message,
    update_entity_with_governance,
    ensure_canonical_context
)

from pydantic import BaseModel as _PydanticBaseModel, Field, ValidationError, ConfigDict
import inspect

from db.connection import get_db_connection_context
from memory.memory_nyx_integration import MemoryNyxBridge, get_memory_nyx_bridge
from nyx.user_model_sdk import UserModelManager
from nyx.nyx_task_integration import NyxTaskIntegration
from nyx.core.emotions.emotional_core import EmotionalCore
from nyx.performance_monitor import PerformanceMonitor
from .response_filter import ResponseFilter
from nyx.core.sync.strategy_controller import get_active_strategies

# ===== CRITICAL FIX #2: Default Model Settings with strict_tools=False =====
DEFAULT_MODEL_SETTINGS = ModelSettings(
    strict_tools=False,
    # Disable structured format validation
    response_format=None,
)

# ===== CRITICAL FIX #3: Clean BaseModel without additionalProperties =====
class BaseModel(_PydanticBaseModel):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        try:
            schema = super().model_json_schema(*args, **kwargs)
            # Log if additionalProperties exists
            if 'additionalProperties' in str(schema):
                logger.warning(f"Model {cls.__name__} has additionalProperties in schema")
            return clean_schema(schema)
        except Exception as e:
            logger.error(f"Schema generation failed for {cls.__name__}: {e}")
            raise

# Alias for compatibility
StrictBaseModel = BaseModel

# ===== Utility Types for Strict Schema =====
JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List[JsonScalar]]

class KVPair(BaseModel):
    key: str
    value: JsonValue

class KVList(BaseModel):
    items: List[KVPair] = Field(default_factory=list)

KVPair.model_rebuild()

# Helpers for conversion
def dict_to_kvlist(d: dict) -> KVList:
    return KVList(items=[KVPair(key=k, value=v) for k, v in d.items()])

def kvlist_to_dict(kv: KVList) -> dict:
    return {pair.key: pair.value for pair in kv.items}

# ===== Global sanitization functions =====
def sanitize_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize JSON schema by removing additionalProperties with detailed logging."""
    s = copy.deepcopy(schema)
    
    def strip_ap(obj):
        if isinstance(obj, dict):
            obj.pop('additionalProperties', None)
            obj.pop('unevaluatedProperties', None)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    strip_ap(v)
        elif isinstance(obj, list):
            for item in obj:
                strip_ap(item)
        return obj
    
    return strip_ap(s)

def sanitize_agent_tools_in_place(agent):
    """
    If any tools were created before our patches (import order), sanitize
    their schema dicts in-place just before a run.
    """
    try:
        tools = []
        
        # Try direct attribute access first (safest)
        if hasattr(agent, "tools"):
            tools = getattr(agent, "tools", []) or []
        # Handle get_all_tools with potential run_context requirement
        elif hasattr(agent, "get_all_tools") and callable(agent.get_all_tools):
            try:
                tools = agent.get_all_tools() or []
            except TypeError as e:
                if "run_context" in str(e):
                    # New API, fall back to direct access
                    tools = getattr(agent, "tools", []) or []
                else:
                    raise
        # Try get_tools as fallback
        elif hasattr(agent, "get_tools") and callable(agent.get_tools):
            try:
                tools = agent.get_tools() or []
            except TypeError:
                tools = getattr(agent, "tools", []) or []

        for t in tools:
            # handle common attributes where schema lives
            for attr in ("parameters", "_parameters", "_schema", "schema", "openai_schema"):
                val = getattr(t, attr, None)
                if isinstance(val, dict):
                    try:
                        setattr(t, attr, sanitize_json_schema(val))
                    except Exception:
                        logger.debug("Could not sanitize tool attr %s on %r", attr, t)
    except Exception:
        logger.exception("sanitize_agent_tools_in_place failed")

def get_canonical_context(ctx_obj) -> Any:
    """Convert various context objects to canonical context"""
    if hasattr(ctx_obj, 'user_id') and hasattr(ctx_obj, 'conversation_id'):
        return ensure_canonical_context({
            'user_id': ctx_obj.user_id,
            'conversation_id': ctx_obj.conversation_id
        })
    elif hasattr(ctx_obj, 'context'):
        return get_canonical_context(ctx_obj.context)
    else:
        raise ValueError("Cannot extract canonical context from object")

def strict_output(model_cls):
    """Return the Pydantic model class as-is."""
    if not inspect.isclass(model_cls):
        raise TypeError("strict_output expects a Pydantic model class")
    try:
        _ = model_cls.model_json_schema()
        logger.debug("strict_output: schema ready for %s", getattr(model_cls, "__name__", model_cls))
    except Exception:
        logger.debug("strict_output: schema build skipped for %r", model_cls, exc_info=True)
    return model_cls

def debug_strict_schema_for_agent(agent: Any, log: logging.Logger = logger) -> None:
    """
    Log sanitized tool schemas at DEBUG. Never raises if a tool is wrapped.
    """
    try:
        # Try different ways to get tools
        tools = []
        
        # First try direct attribute access (safest)
        if hasattr(agent, "tools"):
            tools = getattr(agent, "tools", []) or []
        # If get_all_tools exists but requires run_context, skip it
        elif hasattr(agent, "get_all_tools"):
            try:
                # Try calling without arguments (old API)
                tools = agent.get_all_tools() or []
            except TypeError as e:
                if "run_context" in str(e):
                    # New API requires run_context, skip for now
                    log.debug("[strict] get_all_tools requires run_context, using direct attribute access")
                    tools = getattr(agent, "tools", []) or []
                else:
                    raise
        # Try get_tools as fallback
        elif hasattr(agent, "get_tools") and callable(agent.get_tools):
            try:
                tools = agent.get_tools() or []
            except TypeError:
                # May also require arguments
                tools = getattr(agent, "tools", []) or []

        log.debug("[strict] inspecting %d tools on agent %s", len(tools), getattr(agent, "name", agent))
        for i, t in enumerate(tools):
            try:
                name = getattr(t, "name", getattr(t, "__name__", f"tool_{i}"))
                log.debug("[strict] tool=%s", name)
            except Exception:
                name = f"tool_{i}"
                log.error("Could not inspect tool %s", name, exc_info=True)
    except Exception:
        log.exception("debug_strict_schema_for_agent: top-level failure")

def log_strict_hits(agent: Any) -> None:
    """
    Backwards-compatible alias (your code calls this elsewhere).
    """
    debug_strict_schema_for_agent(agent, logger)


def _get_app_ctx(ctx: Any) -> Any:
    """
    Robustly unwrap nested RunContextWrapper -> ... -> NyxContext.
    Never assume a single .context hop.
    """
    c = getattr(ctx, "context", ctx)
    seen = set()
    # Walk down .context until we find something that looks like NyxContext
    # (has user_id & conversation_id or has world_director/current_world_state)
    while True:
        if id(c) in seen:
            break
        seen.add(id(c))

        if any(hasattr(c, k) for k in ("user_id", "conversation_id", "world_director", "current_world_state")):
            break

        nxt = getattr(c, "context", None)
        if nxt is None:
            break
        c = nxt

    return c

async def _ensure_world_state_from_ctx(app_ctx: Any):
    """
    Best-effort world_state fetch that never throws:
    - Use app_ctx.current_world_state if present
    - Else try app_ctx.world_director.context.current_world_state (initializing director if needed)
    - Else return None
    """
    ws = getattr(app_ctx, "current_world_state", None)
    if ws is not None:
        return ws

    wd = getattr(app_ctx, "world_director", None)
    if not wd:
        return None

    try:
        # Initialize if needed
        if hasattr(wd, "_initialized") and not getattr(wd, "_initialized", False):
            if hasattr(wd, "initialize") and asyncio.iscoroutinefunction(wd.initialize):
                await wd.initialize()
        ctx2 = getattr(wd, "context", None)
        return getattr(ctx2, "current_world_state", None) if ctx2 else None
    except Exception:
        return None

def _extract_last_assistant_text(resp) -> str:
    """Extract the last assistant message text from response"""
    msgs = [c for c in resp if c.get("type") == "message" and c.get("role") == "assistant"]
    if not msgs:
        return ""
    parts = msgs[-1].get("content") or []
    for part in reversed(parts):
        if part.get("type") == "output_text" and part.get("text"):
            return part["text"]
    return ""

def _did_call_tool(resp, tool_name: str) -> bool:
    """Check if a specific tool was called in the response"""
    for c in resp:
        if c.get("type") in ("function_call", "function_call_output") and c.get("name") == tool_name:
            return True
    return False

def _tool_output(resp, name: str):
    """Extract tool output data from response, return {} if not found"""
    outs = [c for c in resp if c.get("type") == "function_call_output" and c.get("name") == name]
    if not outs:
        return {}
    try:
        raw = outs[-1].get("output")
        # Agents SDK often stores JSON string in .output
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        return {}

def _extract_last_assistant_text(resp) -> str:
    """Extract the last assistant message text from response"""
    msgs = [c for c in resp if c.get("type") == "message" and c.get("role") == "assistant"]
    if not msgs:
        return ""
    parts = msgs[-1].get("content") or []
    for part in reversed(parts):
        if part.get("type") == "output_text" and part.get("text"):
            return part["text"]
    return ""

def _did_call_tool(resp, tool_name: str) -> bool:
    """Check if a specific tool was called in the response"""
    for c in resp:
        if c.get("type") in ("function_call", "function_call_output") and c.get("name") == tool_name:
            return True
    return False

def _tool_output(resp, name: str):
    """Extract tool output data from response, return {} if not found"""
    outs = [c for c in resp if c.get("type") == "function_call_output" and c.get("name") == name]
    if not outs:
        return {}
    try:
        raw = outs[-1].get("output")
        # Agents SDK often stores JSON string in .output
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        return {}

def assemble_nyx_response(resp: list) -> Dict[str, Any]:
    """Assemble final response from agent run with full narrator integration"""
    
    # Extract tool outputs
    scene = _tool_output(resp, "narrate_slice_of_life_scene")
    dialogue = _tool_output(resp, "generate_npc_dialogue")
    world = _tool_output(resp, "check_world_state")
    npc = _tool_output(resp, "simulate_npc_autonomy")
    img = _tool_output(resp, "decide_image_generation")
    updates = _tool_output(resp, "generate_universal_updates")
    emergent = _tool_output(resp, "generate_emergent_event")
    power = _tool_output(resp, "narrate_power_exchange")
    routine = _tool_output(resp, "narrate_daily_routine")
    
    # 1) Primary narrative - prioritize sophisticated narrator output
    narrative = ""
    atmosphere = ""
    nyx_commentary = None
    governance_approved = True
    context_aware = False
    
    # Check if we got sophisticated narration from SliceOfLifeNarrator
    if scene and isinstance(scene, dict):
        # Extract narrative - could be nested in different ways
        narrative = scene.get("narrative") or scene.get("scene_description") or ""
        atmosphere = scene.get("atmosphere", "")
        nyx_commentary = scene.get("nyx_commentary")
        governance_approved = scene.get("governance_approved", True)
        context_aware = scene.get("context_aware", False)
        
        # If no narrative yet, check for structured narration data
        if not narrative and "scene" in scene:
            inner_scene = scene["scene"]
            if isinstance(inner_scene, dict):
                narrative = inner_scene.get("description", "")
    
    # If still no narrative, check dialogue
    if not narrative and dialogue:
        if isinstance(dialogue, dict):
            npc_dialogue = dialogue.get("dialogue", "")
            npc_name = dialogue.get("npc_name", "Someone")
            if npc_dialogue:
                narrative = f'{npc_name}: "{npc_dialogue}"'
                if dialogue.get("subtext"):
                    narrative += f"\n*{dialogue['subtext']}*"
    
    # If still no narrative, check power exchange
    if not narrative and power:
        if isinstance(power, dict):
            narrative = power.get("narrative") or power.get("moment", "")
    
    # If still no narrative, check routine
    if not narrative and routine:
        if isinstance(routine, dict):
            narrative = routine.get("description") or routine.get("routine_with_dynamics", "")
    
    # Final fallback - extract from assistant messages
    if not narrative or not narrative.strip():
        narrative = _extract_last_assistant_text(resp)
    
    # Ultimate fallback
    if not narrative:
        narrative = "The moment unfolds in the simulation."
    
    # 2) Extract world state metadata
    world_mood = "neutral"
    time_of_day = "Unknown"
    tension_level = 3
    
    # From scene data (sophisticated narrator)
    if scene and isinstance(scene, dict):
        world_mood = scene.get("world_mood") or world_mood
        time_of_day = scene.get("time_of_day") or time_of_day
        tension_level = scene.get("tension_level", tension_level)
    
    # From world state check
    if world and isinstance(world, dict):
        world_mood = world.get("world_mood") or world_mood
        time_of_day = world.get("time_of_day") or time_of_day
        if "tensions" in world:
            tensions = world["tensions"]
            if isinstance(tensions, dict):
                avg_tension = sum(tensions.values()) / len(tensions) if tensions else 0.5
                tension_level = int(avg_tension * 10)
    
    # 3) Extract available activities/choices
    available = []
    
    # From sophisticated narrator
    if scene and isinstance(scene, dict):
        available = scene.get("available_activities", [])
        if not available:
            available = scene.get("choices", [])
    
    # From power exchange
    if not available and power and isinstance(power, dict):
        available = power.get("player_response_options", [])
    
    # Default activities
    if not available:
        available = ["observe", "browse", "leave", "interact"]
    
    # 4) Extract emergent opportunities
    emergent_opportunities = []
    
    if scene and isinstance(scene, dict):
        emergent_opportunities = scene.get("emergent_opportunities", [])
    
    if emergent and isinstance(emergent, dict):
        if "event" in emergent:
            emergent_opportunities.append(emergent["event"].get("title", "Emergent event"))
    
    # 5) Extract power dynamics
    power_undertones = []
    
    if scene and isinstance(scene, dict):
        power_undertones = scene.get("power_undertones") or scene.get("power_dynamic_hints", [])
    
    if power and isinstance(power, dict):
        if power.get("exchange_type"):
            power_undertones.append(f"Active {power['exchange_type']}")
    
    if not power_undertones:
        power_undertones = ["subtle control dynamics"]
    
    # 6) Handle tension level safely
    if not isinstance(tension_level, (int, float)):
        try:
            tension_level = int(tension_level)
        except:
            tension_level = 3  # default medium tension
    
    # 7) Image generation decision
    should_image = False
    image_prompt = None
    
    # Check multiple sources for image decision
    if img and isinstance(img, dict):
        should_image = bool(img.get("should_generate") or img.get("generate_image", False))
        image_prompt = img.get("image_prompt")
    
    if not should_image and scene and isinstance(scene, dict):
        should_image = bool(scene.get("generate_image", False))
        if not image_prompt:
            image_prompt = scene.get("image_prompt") or scene.get("image_description")
    
    # 8) Extract NPC-specific data
    npc_data = {}
    
    if dialogue and isinstance(dialogue, dict):
        npc_data["last_dialogue"] = {
            "npc_id": dialogue.get("npc_id"),
            "text": dialogue.get("dialogue"),
            "tone": dialogue.get("tone"),
            "requires_response": dialogue.get("requires_response", False)
        }
    
    if npc and isinstance(npc, dict):
        npc_data["autonomy"] = {
            "actions": npc.get("npc_actions", []),
            "observation": npc.get("nyx_observation", "")
        }
    
    # 9) Build final response structure
    response = {
        "narrative": narrative,
        "world": {
            "mood": world_mood,
            "time_of_day": time_of_day,
            "tension_level": tension_level,
            "atmosphere": atmosphere,
        },
        "choices": available,
        "emergent": emergent_opportunities,
        "undertones": power_undertones,
        "image": {
            "should_generate": should_image,
            "prompt": image_prompt
        },
        "universal_updates": updates.get("updates_generated", False) if updates else False,
        "nyx_commentary": nyx_commentary,
        "governance": {
            "approved": governance_approved,
            "context_aware": context_aware
        },
        "npc": npc_data if npc_data else None,
        "telemetry": {
            "tools_called": sorted({
                c.get("name") for c in resp 
                if isinstance(c, dict) and c.get("type") == "function_call"
            }),
            "tools_with_output": sorted({
                c.get("name") for c in resp 
                if isinstance(c, dict) and c.get("type") == "function_call_output"
            }),
            "narrator_used": bool(scene and context_aware),  # Track if sophisticated narrator was used
        }
    }
    
    # 10) Add any conflict manifestations if present
    if scene and isinstance(scene, dict):
        conflict_manifestations = scene.get("conflict_manifestations", [])
        if conflict_manifestations:
            response["conflicts"] = {
                "active": True,
                "manifestations": conflict_manifestations
            }
    
    # 11) Add any system intersections
    if scene and isinstance(scene, dict):
        system_triggers = scene.get("system_triggers", [])
        if system_triggers:
            response["system_intersections"] = system_triggers
    
    # 12) Add relevant memories if narrator provided them
    if scene and isinstance(scene, dict):
        relevant_memories = scene.get("relevant_memories", [])
        if relevant_memories:
            response["memory_context"] = relevant_memories[:3]  # Limit to top 3
    
    return response

# ===== Constants and Configuration =====
class Config:
    """Configuration constants to avoid magic numbers"""
    # Entity types
    ENTITY_TYPE_INTEGRATED = "integrated"
    ENTITY_TYPE_USER = "user"
    ENTITY_TYPE_NPC = "npc"
    ENTITY_TYPE_ENTITY = "entity"
    
    # Memory thresholds
    HIGH_MEMORY_THRESHOLD_MB = 500
    MAX_RESPONSE_TIMES = 100
    MAX_ERROR_LOG_ENTRIES = 100
    MAX_ADAPTATION_HISTORY = 100
    MAX_LEARNED_PATTERNS = 50
    
    # Performance thresholds
    HIGH_RESPONSE_TIME_THRESHOLD = 2.0
    MIN_SUCCESS_RATE = 0.8
    HIGH_ERROR_COUNT = 100
    
    # Relationship thresholds
    INTIMATE_TRUST_THRESHOLD = 0.8
    INTIMATE_BOND_THRESHOLD = 0.7
    FRIENDLY_TRUST_THRESHOLD = 0.6
    HOSTILE_TRUST_THRESHOLD = 0.3
    DOMINANT_POWER_THRESHOLD = 0.7
    SUBMISSIVE_POWER_THRESHOLD = 0.3
    
    # Task thresholds
    MIN_NPC_RELATIONSHIP_FOR_TASK = 30
    
    # Emotional thresholds
    HIGH_AROUSAL_THRESHOLD = 0.8
    NEGATIVE_VALENCE_THRESHOLD = -0.5
    POSITIVE_VALENCE_THRESHOLD = 0.5
    HIGH_DOMINANCE_THRESHOLD = 0.8
    EMOTIONAL_VARIANCE_THRESHOLD = 0.5
    
    # Memory relevance thresholds
    VIVID_RECALL_THRESHOLD = 0.8
    REMEMBER_THRESHOLD = 0.6
    THINK_RECALL_THRESHOLD = 0.4
    
    # Decision thresholds
    MIN_DECISION_SCORE = 0.3
    FALLBACK_DECISION_SCORE = 0.4
    
    # Conflict detection
    POWER_CONFLICT_THRESHOLD = 0.7
    MAX_STABILITY_ISSUES = 10

# ===== Core Data Models =====

class MemoryItem(BaseModel):
    id: Optional[str] = Field(None, description="Memory ID if available")
    text: str = Field(..., description="Memory text")
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score 0-1")
    tags: List[str] = Field(default_factory=list, description="Memory tags")

class EmotionalChanges(BaseModel):
    valence_change: float
    arousal_change: float
    dominance_change: float

class ScoreComponents(BaseModel):
    context: float
    emotional: float
    pattern: float
    relationship: float

class PerformanceNumbers(BaseModel):
    memory_mb: float
    cpu_percent: float
    avg_response_time: float
    success_rate: float

class ConflictItem(BaseModel):
    type: str
    severity: float
    description: str
    entities: Optional[List[str]] = None
    blocked_objectives: Optional[List[str]] = None

class InstabilityItem(BaseModel):
    type: str
    severity: float
    description: str
    recommendation: Optional[str] = None

class ActivityRec(BaseModel):
    name: str
    description: str
    requirements: List[str]
    duration: str
    intensity: str
    partner_id: Optional[str] = None

class RelationshipStateOut(BaseModel):
    trust: float
    power_dynamic: float
    emotional_bond: float
    interaction_count: int
    last_interaction: float
    type: str

class RelationshipChanges(BaseModel):
    trust: float
    power: float
    bond: float

class DecisionMetadata(BaseModel):
    data: KVList = Field(default_factory=KVList, description="Additional metadata")

class ScoredOption(BaseModel):
    option: 'DecisionOption'
    score: float
    components: ScoreComponents
    is_fallback: Optional[bool] = False

# ===== Pydantic Models for Structured Outputs =====

class NarrativeResponse(BaseModel):
    """Structured output for Nyx's narrative responses"""
    narrative: str = Field(..., description="The main narrative response as Nyx")
    tension_level: int = Field(0, description="Current narrative tension level (0-10)")
    generate_image: bool = Field(False, description="Whether an image should be generated for this scene")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    environment_description: Optional[str] = Field(None, description="Updated environment description if changed")
    time_advancement: bool = Field(False, description="Whether time should advance after this interaction")
    universal_updates: Optional[KVList] = Field(None, description="Universal updates extracted from narrative")
    world_mood: Optional[str] = Field(None, description="Current world mood")
    ongoing_events: Optional[List[str]] = Field(None, description="Active slice-of-life events")
    available_activities: Optional[List[str]] = Field(None, description="Available player activities")
    npc_schedules: Optional[KVList] = None
    time_of_day: Optional[str] = Field(None, description="Current time period")
    emergent_opportunities: Optional[List[str]] = Field(None, description="Emergent narrative opportunities")

class MemoryReflection(BaseModel):
    """Structured output for memory reflections"""
    reflection: str = Field(..., description="The reflection text")
    confidence: float = Field(..., description="Confidence level in the reflection (0.0-1.0)")
    topic: Optional[str] = Field(None, description="Topic of the reflection")

class ContentModeration(BaseModel):
    """Output for content moderation guardrail"""
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

class EmotionalStateUpdate(BaseModel):
    """Structured output for emotional state changes"""
    valence: float = Field(..., description="Positive/negative emotion (-1 to 1)")
    arousal: float = Field(..., description="Emotional intensity (0 to 1)")
    dominance: float = Field(..., description="Control level (0 to 1)")
    primary_emotion: str = Field(..., description="Primary emotion label")
    reasoning: str = Field(..., description="Why the emotional state changed")

class ScenarioDecision(BaseModel):
    """Structured output for scenario management decisions"""
    action: str = Field(..., description="Action to take (advance, maintain, escalate, de-escalate)")
    next_phase: str = Field(..., description="Next scenario phase")
    tasks: List[KVList] = Field(default_factory=list, description="Tasks to execute")
    npc_actions: List[KVList] = Field(default_factory=list, description="NPC actions to take")
    time_advancement: bool = Field(False, description="Whether to advance time after this phase")

class RelationshipUpdate(BaseModel):
    """Structured output for relationship changes"""
    trust_change: float = Field(0.0, description="Change in trust level")
    power_dynamic_change: float = Field(0.0, description="Change in power dynamic")
    emotional_bond_change: float = Field(0.0, description="Change in emotional bond")
    relationship_type: str = Field(..., description="Type of relationship")

class ActivityRecommendation(BaseModel):
    """Structured output for activity recommendations"""
    recommended_activities: List[ActivityRec] = Field(..., description="List of recommended activities")
    reasoning: str = Field(..., description="Why these activities are recommended")

class ImageGenerationDecision(BaseModel):
    """Decision about whether to generate an image"""
    should_generate: bool = Field(..., description="Whether an image should be generated")
    score: float = Field(0.0, description="Confidence score for the decision")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    reasoning: str = Field(..., description="Reasoning for the decision")

# ===== Function Tool Input Models =====

class RetrieveMemoriesInput(BaseModel):
    """Input for retrieve_memories function"""
    query: str = Field(..., description="Search query to find memories")
    limit: int = Field(5, description="Maximum number of memories to return", ge=1, le=20)

class AddMemoryInput(BaseModel):
    """Input for add_memory function"""
    memory_text: str = Field(..., description="The content of the memory")
    memory_type: str = Field("observation", description="Type of memory (observation, reflection, abstraction)")
    significance: int = Field(5, description="Importance of memory (1-10)", ge=1, le=10)

class DetectUserRevelationsInput(BaseModel):
    """Input for detect_user_revelations function"""
    user_message: str = Field(..., description="The user's message to analyze")

class GenerateImageFromSceneInput(BaseModel):
    """Input for generate_image_from_scene function"""
    scene_description: str = Field(..., description="Description of the scene")
    characters: List[str] = Field(..., description="List of characters in the scene")
    style: str = Field("realistic", description="Style for the image")

class CalculateEmotionalStateInput(BaseModel):
    """Input for calculate_and_update_emotional_state and calculate_emotional_impact functions"""
    context: KVList = Field(..., description="Current interaction context")

class UpdateRelationshipStateInput(BaseModel):
    """Input for update_relationship_state function"""
    entity_id: str = Field(..., description="ID of the entity (NPC or user)")
    trust_change: float = Field(0.0, description="Change in trust level", ge=-1.0, le=1.0)
    power_change: float = Field(0.0, description="Change in power dynamic", ge=-1.0, le=1.0)
    bond_change: float = Field(0.0, description="Change in emotional bond", ge=-1.0, le=1.0)

class GetActivityRecommendationsInput(BaseModel):
    """Input for get_activity_recommendations function"""
    scenario_type: str = Field(..., description="Type of current scenario")
    npc_ids: List[str] = Field(..., description="List of present NPC IDs")

class BeliefDataModel(BaseModel):
    """Model for belief data to avoid raw dicts"""
    entity_id: str = Field("nyx", description="Entity ID")
    type: str = Field("general", description="Belief type")
    content: KVList = Field(default_factory=KVList, description="Belief content")
    query: Optional[str] = Field(None, description="Query for belief search")

class ManageBeliefsInput(BaseModel):
    """Input for manage_beliefs function"""
    action: Literal["get", "update", "query"] = Field(..., description="Action to perform")
    belief_data: BeliefDataModel = Field(..., description="Data for the belief operation")

class DecisionOption(BaseModel):
    """Model for decision options to avoid raw dicts"""
    id: str = Field(..., description="Option ID")
    description: str = Field(..., description="Option description")
    metadata: DecisionMetadata = Field(default_factory=DecisionMetadata)

class ScoreDecisionOptionsInput(BaseModel):
    """Input for score_decision_options function"""
    options: List[DecisionOption] = Field(..., description="List of possible decisions/actions")
    decision_context: KVList = Field(..., description="Context for making the decision")

class DetectConflictsAndInstabilityInput(BaseModel):
    """Input for detect_conflicts_and_instability function"""
    scenario_state: KVList = Field(..., description="Current scenario state")

class GenerateUniversalUpdatesInput(BaseModel):
    """Input for generate_universal_updates function"""
    narrative: str = Field(..., description="The narrative text to process")

class DecideImageInput(BaseModel):
    """Input for decide_image_generation function"""
    scene_text: str = Field(..., description="Scene description to evaluate for image generation")

class EmptyInput(BaseModel):
    """Empty input for functions that don't require parameters"""
    pass

# ===== Function Tool Output Models =====

class MemorySearchResult(BaseModel):
    """Output for retrieve_memories function"""
    memories: List[MemoryItem] = Field(..., description="List of retrieved memories")
    formatted_text: str = Field(..., description="Formatted memory text")

class MemoryStorageResult(BaseModel):
    """Output for add_memory function"""
    memory_id: str = Field(..., description="ID of stored memory")
    success: bool = Field(..., description="Whether memory was stored successfully")

class UserGuidanceResult(BaseModel):
    """Output for get_user_model_guidance function"""
    top_kinks: List[Tuple[str, float]] = Field(..., description="Top user preferences with levels")
    behavior_patterns: KVList = Field(..., description="Identified behavior patterns")
    suggested_intensity: float = Field(..., description="Suggested interaction intensity")
    reflections: List[str] = Field(..., description="User model reflections")

class RevelationDetectionResult(BaseModel):
    """Output for detect_user_revelations function"""
    revelations: List[KVList] = Field(..., description="Detected revelations")
    has_revelations: bool = Field(..., description="Whether any revelations were found")

class ImageGenerationResult(BaseModel):
    """Output for generate_image_from_scene function"""
    success: bool = Field(..., description="Whether image was generated")
    image_url: Optional[str] = Field(None, description="URL of generated image")
    error: Optional[str] = Field(None, description="Error message if failed")

class EmotionalCalculationResult(BaseModel):
    """Output for emotional calculation functions"""
    valence: float = Field(..., description="New valence value")
    arousal: float = Field(..., description="New arousal value")
    dominance: float = Field(..., description="New dominance value")
    primary_emotion: str = Field(..., description="Primary emotion label")
    changes: EmotionalChanges = Field(..., description="Changes applied")
    state_updated: Optional[bool] = Field(None, description="Whether state was persisted")

class RelationshipUpdateResult(BaseModel):
    """Output for update_relationship_state function"""
    entity_id: str = Field(..., description="Entity ID")
    relationship: RelationshipStateOut = Field(..., description="Updated relationship state")
    changes: RelationshipChanges = Field(..., description="Changes applied")

class PerformanceMetricsResult(BaseModel):
    """Output for check_performance_metrics function"""
    metrics: PerformanceNumbers = Field(..., description="Current performance metrics")
    suggestions: List[str] = Field(..., description="Performance improvement suggestions")
    actions_taken: List[str] = Field(..., description="Remediation actions taken")
    health: str = Field(..., description="Overall system health status")

class ActivityRecommendationsResult(BaseModel):
    """Output for get_activity_recommendations function"""
    recommendations: List[ActivityRec] = Field(..., description="Recommended activities")
    total_available: int = Field(..., description="Total number of available activities")

class BeliefManagementResult(BaseModel):
    """Output for manage_beliefs function"""
    result: Union[str, KVList] = Field(..., description="Operation result")
    error: Optional[str] = Field(None, description="Error message if failed")

class DecisionScoringResult(BaseModel):
    """Output for score_decision_options function"""
    scored_options: List[ScoredOption] = Field(..., description="Options with scores")
    best_option: DecisionOption = Field(..., description="Highest scoring option")
    confidence: float = Field(..., description="Confidence in best option")

class ConflictDetectionResult(BaseModel):
    """Output for detect_conflicts_and_instability function"""
    conflicts: List[ConflictItem] = Field(..., description="Detected conflicts")
    instabilities: List[InstabilityItem] = Field(..., description="Detected instabilities")
    overall_stability: float = Field(..., description="Overall stability score (0-1)")
    stability_note: str = Field(..., description="Explanation of stability score")
    requires_intervention: bool = Field(..., description="Whether intervention is needed")

class UniversalUpdateResult(BaseModel):
    """Output for generate_universal_updates function"""
    success: bool = Field(..., description="Whether updates were generated")
    updates_generated: bool = Field(..., description="Whether any updates were found")
    error: Optional[str] = Field(None, description="Error message if failed")

# ===== Composite Models for Complex Operations =====

class ScenarioManagementRequest(BaseModel):
    """Request for scenario management"""
    user_id: int = Field(..., description="User ID")
    conversation_id: int = Field(..., description="Conversation ID")
    scenario_id: Optional[str] = Field(None, description="Scenario ID")
    type: str = Field("general", description="Scenario type")
    participants: List[KVList] = Field(default_factory=list, description="Scenario participants")
    objectives: List[KVList] = Field(default_factory=list, description="Scenario objectives")

class RelationshipInteractionData(BaseModel):
    """Data for relationship interactions"""
    user_id: int = Field(..., description="User ID")
    conversation_id: int = Field(..., description="Conversation ID")
    participants: List[KVList] = Field(..., description="Interaction participants")
    interaction_type: str = Field(..., description="Type of interaction")
    outcome: str = Field(..., description="Interaction outcome")
    emotional_impact: Optional[KVList] = Field(None, description="Emotional impact data")

# ===== State Models =====

class EmotionalState(BaseModel):
    """Emotional state representation"""
    valence: float = Field(0.0, description="Positive/negative emotion (-1 to 1)", ge=-1.0, le=1.0)
    arousal: float = Field(0.5, description="Emotional intensity (0 to 1)", ge=0.0, le=1.0)
    dominance: float = Field(0.7, description="Control level (0 to 1)", ge=0.0, le=1.0)

class RelationshipState(BaseModel):
    """Relationship state representation"""
    trust: float = Field(0.5, description="Trust level (0-1)", ge=0.0, le=1.0)
    power_dynamic: float = Field(0.5, description="Power dynamic (0-1)", ge=0.0, le=1.0)
    emotional_bond: float = Field(0.3, description="Emotional bond strength (0-1)", ge=0.0, le=1.0)
    interaction_count: int = Field(0, description="Number of interactions", ge=0)
    last_interaction: float = Field(..., description="Timestamp of last interaction")
    type: str = Field("neutral", description="Relationship type")

class PerformanceMetrics(BaseModel):
    """Performance metrics structure"""
    total_actions: int = Field(0, ge=0)
    successful_actions: int = Field(0, ge=0)
    failed_actions: int = Field(0, ge=0)
    response_times: List[float] = Field(default_factory=list)
    memory_usage: float = Field(0.0, ge=0.0)
    cpu_usage: float = Field(0.0, ge=0.0, le=100.0)
    error_rates: KVList = Field(default_factory=lambda: dict_to_kvlist({"total": 0, "recovered": 0, "unrecovered": 0}))

class LearningMetrics(BaseModel):
    """Learning metrics structure"""
    pattern_recognition_rate: float = Field(0.0, ge=0.0, le=1.0)
    strategy_improvement_rate: float = Field(0.0, ge=0.0, le=1.0)
    adaptation_success_rate: float = Field(0.0, ge=0.0, le=1.0)

# ===== Open World / Slice-of-life Models =====

class NarrateSliceInput(BaseModel):
    scene_type: str = Field("routine", description="Slice-of-life scene type")

class EmergentEventInput(BaseModel):
    event_type: Optional[str] = Field(None, description="Optional event type hint")

class SimulateAutonomyInput(BaseModel):
    hours: int = Field(1, ge=1, le=24, description="Hours to advance")

# Resolve forward references
ScoredOption.model_rebuild()
DecisionOption.model_rebuild()

# ===== Enhanced Context with State Management =====
@dataclass
class NyxContext:
    # ────────── REQUIRED (no defaults) ──────────
    user_id: int
    conversation_id: int

    # ────────── SUB-SYSTEM HANDLES ──────────
    memory_system:      Optional[MemoryNyxBridge]   = None
    user_model:         Optional[UserModelManager]  = None
    task_integration:   Optional[NyxTaskIntegration] = None
    response_filter:    Optional[ResponseFilter]    = None
    emotional_core:     Optional[EmotionalCore]     = None
    performance_monitor: Optional[PerformanceMonitor] = None
    belief_system:      Optional[Any]               = None
    world_director:     Optional[Any]               = None
    slice_of_life_narrator: Optional[Any]           = None

    # ────────── MUTABLE STATE BUCKETS ──────────
    current_context:     Dict[str, Any]                = field(default_factory=dict)
    scenario_state:      Dict[str, Any]                = field(default_factory=dict)
    relationship_states: Dict[str, Dict[str, Any]]     = field(default_factory=dict)
    active_tasks:        List[Dict[str, Any]]          = field(default_factory=list)
    current_world_state: Optional[Any]                = None
    daily_routine_tracker: Optional[Dict[str, Any]]   = None
    emergent_narratives: List[Dict[str, Any]]        = field(default_factory=list)
    npc_autonomy_states: Dict[int, Dict[str, Any]]   = field(default_factory=dict)

    # ────────── PERFORMANCE & EMOTION ──────────
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_actions": 0, "successful_actions": 0, "failed_actions": 0,
        "response_times": [], "memory_usage": 0, "cpu_usage": 0,
        "error_rates": {"total": 0, "recovered": 0, "unrecovered": 0}
    })
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.0, "arousal": 0.5, "dominance": 0.7
    })

    # ────────── LEARNING & ADAPTATION ──────────
    learned_patterns:      Dict[str, Any]           = field(default_factory=dict)
    strategy_effectiveness: Dict[str, Any]          = field(default_factory=dict)
    adaptation_history:    List[Dict[str, Any]]     = field(default_factory=list)
    learning_metrics:      Dict[str, Any]           = field(default_factory=lambda: {
        "pattern_recognition_rate": 0.0,
        "strategy_improvement_rate": 0.0,
        "adaptation_success_rate": 0.0
    })

    # ────────── ERROR LOGGING ──────────
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    # ────────── FEATURE FLAGS ──────────
    _tables_available: Dict[str, bool] = field(default_factory=dict)

    # ────────── TASK SCHEDULING ──────────
    last_task_runs: Dict[str, datetime] = field(default_factory=dict)
    task_intervals: Dict[str, float]    = field(default_factory=lambda: {
        "memory_reflection": 300, "relationship_update": 600,
        "scenario_check": 60, "performance_check": 300,
        "task_generation": 300, "learning_save": 900, 
        "performance_save": 600,
        "scenario_heartbeat": 3600
    })

    # ────────── PRIVATE CACHES (init=False) ──────────
    _strategy_cache:             Optional[Tuple[float, Any]] = field(init=False, default=None)
    _strategy_cache_ttl:         float = field(init=False, default=300.0)
    _strategy_cache_lock:        asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _cpu_usage_cache:            Optional[float] = field(init=False, default=None)
    _cpu_usage_last_update:      float = field(init=False, default=0.0)
    _cpu_usage_update_interval:  float = field(init=False, default=10.0)
    
    async def initialize(self):
        """Initialize all systems"""
        self.memory_system = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        self.user_model = await UserModelManager.get_instance(self.user_id, self.conversation_id)
        self.task_integration = await NyxTaskIntegration.get_instance(self.user_id, self.conversation_id)
        self.response_filter = ResponseFilter(self.user_id, self.conversation_id)
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # Initialize emotional core if available
        try:
            self.emotional_core = EmotionalCore()
        except Exception as e:
            logger.warning(f"EmotionalCore not available: {e}", exc_info=True)
        
        # Initialize belief system if available
        try:
            from nyx.nyx_belief_system import BeliefSystem
            self.belief_system = BeliefSystem(self.user_id, self.conversation_id)
        except ImportError as e:
            logger.warning(f"BeliefSystem module not available: {e}")
        except Exception as e:
            logger.warning(f"BeliefSystem initialization failed: {e}", exc_info=True)

        # Initialize world systems
        try:
            from story_agent.world_director_agent import CompleteWorldDirector
            from story_agent.slice_of_life_narrator import SliceOfLifeNarrator

            self.world_director = CompleteWorldDirector(self.user_id, self.conversation_id)
            await self.world_director.initialize()

            self.slice_of_life_narrator = SliceOfLifeNarrator(self.user_id, self.conversation_id)
            await self.slice_of_life_narrator.initialize()

            self.current_world_state = self.world_director.context.current_world_state
        except Exception as e:
            logger.warning(f"World systems initialization failed: {e}", exc_info=True)

        # Initialize CPU usage monitoring
        try:
            self._cpu_usage_cache = safe_psutil('cpu_percent', interval=0.1, default=0.0)
        except Exception as e:
            logger.debug(f"Failed to initialize CPU monitoring: {e}")
            self._cpu_usage_cache = 0.0
        
        # Load existing state from database
        await self._load_state()
    
    async def get_active_strategies_cached(self):
        """Get active strategies with caching and lock to prevent thundering herd"""
        current_time = time.time()
        
        # Check cache without lock first
        if self._strategy_cache:
            cache_time, strategies = self._strategy_cache
            if current_time - cache_time < self._strategy_cache_ttl:
                return strategies
        
        # Need to refresh - use lock to prevent multiple refreshes
        async with self._strategy_cache_lock:
            # Double-check cache inside lock
            if self._strategy_cache:
                cache_time, strategies = self._strategy_cache
                if current_time - cache_time < self._strategy_cache_ttl:
                    return strategies
            
            # Fetch new strategies using its own connection
            async with get_db_connection_context() as conn:
                strategies = await get_active_strategies(conn)
            
            # Update cache
            self._strategy_cache = (current_time, strategies)
            return strategies
    
    async def _load_state(self):
        """Load existing state from database"""
        # Use context manager to get a connection
        async with get_db_connection_context() as conn:
            # Load emotional state
            row = await conn.fetchrow("""
                SELECT emotional_state FROM NyxAgentState
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            if row and row["emotional_state"]:
                state = json.loads(row["emotional_state"])
                self.emotional_state.update(state)
            
            # Load scenario state if exists and table is available
            if self._tables_available.get("scenario_states", True):
                try:
                    scenario_row = await conn.fetchrow("""
                        SELECT state_data FROM scenario_states
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY created_at DESC LIMIT 1
                    """, self.user_id, self.conversation_id)
                    
                    if scenario_row and scenario_row["state_data"]:
                        self.scenario_state = json.loads(scenario_row["state_data"])
                except Exception as e:
                    # Table might not exist yet
                    if "does not exist" in str(e) or "no such table" in str(e).lower():
                        logger.info("scenario_states table not found - migrations may need to be run")
                        self._tables_available["scenario_states"] = False
                    else:
                        logger.debug(f"Could not load scenario state: {e}")
    
    def update_performance(self, metric: str, value: Any):
        """Update performance metrics"""
        if metric in self.performance_metrics:
            if isinstance(self.performance_metrics[metric], list):
                self.performance_metrics[metric].append(value)
                # Keep only last entries based on config
                if len(self.performance_metrics[metric]) > Config.MAX_RESPONSE_TIMES:
                    self.performance_metrics[metric] = self.performance_metrics[metric][-Config.MAX_RESPONSE_TIMES:]
            else:
                self.performance_metrics[metric] = value
    
    def should_run_task(self, task_id: str) -> bool:
        """Check if enough time has passed to run task again"""
        if task_id not in self.last_task_runs:
            return True
        
        time_since_run = (datetime.now(timezone.utc) - self.last_task_runs[task_id]).total_seconds()
        return time_since_run >= self.task_intervals.get(task_id, 300)
    
    def record_task_run(self, task_id: str):
        """Record that a task has been run"""
        self.last_task_runs[task_id] = datetime.now(timezone.utc)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context and aggregate by type"""
        error_type = type(error).__name__
        error_entry = {
            "timestamp": time.time(),
            "error": str(error),
            "type": error_type,
            "context": context or {}
        }
        self.error_log.append(error_entry)
        
        # Track error counts by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update error metrics
        self.performance_metrics["error_rates"]["total"] += 1
        
        # Keep error log bounded
        if len(self.error_log) > Config.MAX_ERROR_LOG_ENTRIES * 2:
            _prune_list(self.error_log, Config.MAX_ERROR_LOG_ENTRIES)
            
        # Log warning if we see repeated errors
        if self.error_counts[error_type] > 10:
            logger.warning(f"Repeated error type {error_type}: {self.error_counts[error_type]} occurrences")
    
    async def learn_from_interaction(self, action: str, outcome: str, success: bool):
        """Learn from an interaction outcome"""
        # Update patterns
        pattern_key = f"{action}_{outcome}"
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "occurrences": 0,
                "successes": 0,
                "last_seen": time.time()
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern["occurrences"] += 1
        if success:
            pattern["successes"] += 1
        pattern["last_seen"] = time.time()
        pattern["success_rate"] = pattern["successes"] / pattern["occurrences"]
        
        # Update adaptation history with emotional state snapshot
        self.adaptation_history.append({
            "timestamp": time.time(),
            "action": action,
            "outcome": outcome,
            "success": success,
            "emotional_state": self.emotional_state.copy()
        })
        
        # Keep adaptation history bounded
        max_history = Config.MAX_ADAPTATION_HISTORY if success else Config.MAX_ADAPTATION_HISTORY // 2
        if len(self.adaptation_history) > max_history * 2:
            self.adaptation_history = self.adaptation_history[-max_history:]
        
        # Prune old patterns (older than 24 hours)
        current_time = time.time()
        self.learned_patterns = {
            k: v for k, v in self.learned_patterns.items()
            if current_time - v.get("last_seen", 0) < 86400
        }
        
        # Update learning metrics
        self._update_learning_metrics()
    
    def should_generate_task(self) -> bool:
        """Determine if we should generate a creative task"""
        context = self.current_context
        
        if not context.get("active_npc_id"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        task_scenarios = ["training", "challenge", "service", "discipline"]
        if not any(t in scenario_type for t in task_scenarios):
            return False
            
        npc_relationship = context.get("npc_relationship_level", 0)
        if npc_relationship < Config.MIN_NPC_RELATIONSHIP_FOR_TASK:
            return False
            
        # Check task timing
        if not self.should_run_task("task_generation"):
            return False
            
        return True
    
    def should_recommend_activities(self) -> bool:
        """Determine if we should recommend activities"""
        context = self.current_context
        
        if not context.get("present_npc_ids"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        if "task" in scenario_type or "challenge" in scenario_type:
            return False
            
        user_input = context.get("user_input", "").lower()
        suggestion_triggers = ["what should", "what can", "what to do", "suggestions", "ideas"]
        if any(trigger in user_input for trigger in suggestion_triggers):
            return True
            
        if context.get("is_scene_transition") or context.get("is_activity_completed"):
            return True
            
        return False
    
    async def handle_high_memory_usage(self):
        """Handle high memory usage by cleaning up"""
        # Trim memory system cache if available
        if hasattr(self.memory_system, 'trim_cache'):
            await self.memory_system.trim_cache()
        
        # Clear old patterns
        self.learned_patterns = dict(list(self.learned_patterns.items())[-Config.MAX_LEARNED_PATTERNS:])
        
        # Clear old history
        self.adaptation_history = self.adaptation_history[-Config.MAX_ADAPTATION_HISTORY:]
        self.error_log = self.error_log[-Config.MAX_ERROR_LOG_ENTRIES:]
        
        # Clear performance metrics history
        if "response_times" in self.performance_metrics:
            self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-Config.MAX_RESPONSE_TIMES:]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Performed memory cleanup")
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage with caching"""
        try:
            current_time = time.time()
            # Check if we need to update the cache
            if (self._cpu_usage_cache is None or 
                current_time - self._cpu_usage_last_update >= self._cpu_usage_update_interval):
                # Update the cache using safe wrapper
                new_value = safe_psutil('cpu_percent', interval=0.1, default=0.0)
                if new_value is not None:
                    self._cpu_usage_cache = new_value
                    self._cpu_usage_last_update = current_time
            
            return self._cpu_usage_cache or 0.0
        except Exception as e:
            logger.debug(f"Failed to get CPU usage: {e}")
            return 0.0

    def db_connection_ctx(self):
        """Get a database connection context manager"""
        return get_db_connection_context()
    
    # Legacy compatibility
    async def get_db_connection(self):
        """DEPRECATED: Use db_connection_ctx() instead"""
        logger.warning("get_db_connection is deprecated, use db_connection_ctx() instead")
        return self.db_connection_ctx()

    async def close_db_connection(self, conn=None):
        """No-op compatibility wrapper"""
        if conn is not None:
            await conn.__aexit__(None, None, None)
    
    def _update_learning_metrics(self):
        """Update learning-related metrics"""
        if self.learned_patterns:
            successful_patterns = sum(1 for p in self.learned_patterns.values() 
                                    if p.get("success_rate", 0) > 0.6)
            self.learning_metrics["pattern_recognition_rate"] = (
                successful_patterns / len(self.learned_patterns)
            )
        
        if self.adaptation_history:
            recent = self.adaptation_history[-Config.MAX_ADAPTATION_HISTORY:]
            successes = sum(1 for a in recent if a["success"])
            self.learning_metrics["adaptation_success_rate"] = successes / len(recent)

# ===== Helper Functions =====

def safe_psutil(func_name: str, *args, default=None, **kwargs):
    """Safe wrapper for psutil calls that may fail on certain platforms"""
    try:
        import psutil
        func = getattr(psutil, func_name)
        return func(*args, **kwargs)
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"psutil.{func_name} failed (platform compatibility): {e}")
        return default

def safe_process_metric(process, metric_name: str, default=0):
    """Safe wrapper for process-specific metrics"""
    try:
        metric_func = getattr(process, metric_name)
        result = metric_func()
        # Handle different return types
        if hasattr(result, 'rss'):  # memory_info returns a named tuple
            return result.rss
        return result
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"Process metric {metric_name} failed: {e}")
        return default

def get_process_info() -> Optional[Any]:
    """Get current process info safely"""
    try:
        import psutil
        return psutil.Process(os.getpid())
    except Exception as e:
        logger.debug(f"Failed to get process info: {e}")
        return None

def bytes_to_mb(value: Optional[Union[int, float]]) -> float:
    """Convert bytes to megabytes safely"""
    return (value or 0) / (1024 * 1024)

def extract_token_usage(result: Any) -> int:
    """Extract token usage from various result formats"""
    try:
        # Try different possible locations for token usage
        if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
            return result.usage.total_tokens
        elif hasattr(result, 'trace') and hasattr(result.trace, 'final_usage'):
            return result.trace.final_usage.get('total_tokens', 0)
        else:
            logger.debug("Token usage not found in result object")
            return 0
    except Exception as e:
        logger.debug(f"Failed to retrieve token usage: {e}")
        return 0

def get_context_text_lower(context: Dict[str, Any]) -> str:
    """Extract text from context and convert to lowercase for analysis"""
    text_parts = []
    for key, value in context.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, (list, dict)):
            text_parts.append(str(value))
    return " ".join(text_parts).lower()

def _prune_list(lst: List[Any], max_len: int) -> None:
    """Prune a list to maximum length in-place"""
    if len(lst) > max_len:
        del lst[:-max_len]

def _calculate_avg_response_time(response_times: List[float]) -> float:
    """Calculate average response time safely"""
    if not response_times:
        return 0.0
    try:
        return statistics.fmean(response_times)
    except Exception:
        # Fallback to simple mean
        return sum(response_times) / len(response_times)

def _calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values"""
    if len(values) < 2:
        return 0.0
    try:
        return statistics.variance(values)
    except Exception:
        # Fallback calculation
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

def _json_safe(value, *, _depth=0, _max_depth=4):
    """Best-effort conversion of arbitrary Python objects to JSON-safe primitives."""
    if _depth > _max_depth:
        return str(value)

    # Primitives
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # datetime/date -> ISO string
    try:
        from datetime import datetime, date
        if isinstance(value, (datetime, date)):
            return value.isoformat()
    except Exception:
        pass

    # Enum -> its value
    try:
        from enum import Enum
        if isinstance(value, Enum):
            return _json_safe(getattr(value, "value", str(value)), _depth=_depth+1, _max_depth=_max_depth)
    except Exception:
        pass

    # List/Tuple/Set
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth=_depth+1, _max_depth=_max_depth) for v in value]

    # Dict-like
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth=_depth+1, _max_depth=_max_depth) for k, v in value.items()}

    # Dataclass
    try:
        import dataclasses
        if dataclasses.is_dataclass(value):
            return _json_safe(dataclasses.asdict(value), _depth=_depth+1, _max_depth=_max_depth)
    except Exception:
        pass

    # Pydantic v1/v2 models
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _json_safe(fn(), _depth=_depth+1, _max_depth=_max_depth)
            except Exception:
                break

    # Fallback: attempt __dict__, else str()
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return _json_safe(data, _depth=_depth+1, _max_depth=_max_depth)
    return str(value)

def _preview(text: Optional[str], n: int = 240) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned[:n] + ("…" if len(cleaned) > n else "")

def _js(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"

# ===== Error Handling Functions =====

async def run_agent_safely(
    agent: Agent,
    input_data: Any,
    context: Any,
    run_config: Optional[RunConfig] = None,
    fallback_response: Any = None
) -> Any:
    """Run agent with automatic fallback on strict schema errors"""
    try:
        # First attempt with the agent as-is
        result = await Runner.run(
            agent,
            input_data,
            context=context,
            run_config=run_config
        )
        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "additionalproperties" in error_msg or "strict schema" in error_msg:
            logger.warning(f"Strict schema error, attempting without structured output: {e}")
            
            # Create a simple text-only agent
            fallback_agent = Agent[type(context)](
                name=f"{agent.name} (Fallback)",
                instructions=agent.instructions,
                model=agent.model,
                model_settings=DEFAULT_MODEL_SETTINGS,
                # No tools, no structured output
            )
            
            try:
                result = await Runner.run(
                    fallback_agent,
                    input_data,
                    context=context,
                    run_config=run_config
                )
                return result
            except Exception as e2:
                logger.error(f"Fallback agent also failed: {e2}")
                if fallback_response is not None:
                    return fallback_response
                raise
        else:
            # Not a schema error, re-raise
            raise

async def run_agent_with_error_handling(
    agent: Agent,
    input_data: Any,
    context: NyxContext,
    output_type: Optional[type] = None,
    fallback_value: Any = None
) -> Any:
    """Legacy compatibility wrapper"""
    try:
        result = await run_agent_safely(
            agent,
            input_data,
            context,
            run_config=RunConfig(workflow_name=f"Nyx {getattr(agent, 'name', 'Agent')}"),
            fallback_response=fallback_value
        )
        if output_type:
            return result.final_output_as(output_type)
        return getattr(result, "final_output", None) or getattr(result, "output_text", None)
    except Exception as e:
        logger.error(f"Error running agent {getattr(agent, 'name', 'unknown')}: {e}")
        if fallback_value is not None:
            return fallback_value
        raise

# ===== Function Tools =====

@function_tool
async def narrate_power_exchange(
    ctx: RunContextWrapper[NyxContext],
    npc_id: int,
    exchange_type: str = "subtle_control",
    intensity: float = 0.5
) -> str:
    """Narrate a power exchange using the sophisticated narrator."""
    app_ctx = _get_app_ctx(ctx)
    
    # Ensure narrator is initialized
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(
            app_ctx.user_id,
            app_ctx.conversation_id
        )
        await app_ctx.slice_of_life_narrator.initialize()
    
    narrator = app_ctx.slice_of_life_narrator
    world_state = await _ensure_world_state_from_ctx(app_ctx)
    
    from story_agent.world_simulation_models import PowerExchange, PowerDynamicType
    
    # Map exchange type to enum
    type_map = {
        "subtle_control": PowerDynamicType.SUBTLE_CONTROL,
        "gentle_guidance": PowerDynamicType.GENTLE_GUIDANCE,
        "firm_direction": PowerDynamicType.FIRM_DIRECTION,
        "casual_dominance": PowerDynamicType.CASUAL_DOMINANCE,
        "protective_control": PowerDynamicType.PROTECTIVE_CONTROL,
    }
    
    exchange = PowerExchange(
        initiator_npc_id=npc_id,
        initiator_id=npc_id,
        initiator_type="npc",
        recipient_type="player",
        recipient_id=1,
        exchange_type=type_map.get(exchange_type, PowerDynamicType.SUBTLE_CONTROL),
        intensity=intensity
    )
    
    # Use narrator's power narration
    result = await narrator.power_narrator.run(
        messages=[{"role": "user", "content": "Narrate power exchange"}],
        context=narrator.context,
        tool_calls=[{
            "tool": narrator.narrate_power_exchange,
            "kwargs": {
                "exchange": exchange.model_dump(),
                "world_state": world_state.model_dump() if world_state else {}
            }
        }]
    )
    
    if result and hasattr(result, 'data'):
        return json.dumps(result.data.model_dump() if hasattr(result.data, 'model_dump') else result.data)
    
    return json.dumps({"narrative": "A moment of subtle control passes..."})

@function_tool
async def narrate_daily_routine(
    ctx: RunContextWrapper[NyxContext],
    activity: str,
    involved_npcs: List[int] = None
) -> str:
    """Narrate daily routine using the sophisticated narrator."""
    app_ctx = _get_app_ctx(ctx)
    
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(
            app_ctx.user_id,
            app_ctx.conversation_id
        )
        await app_ctx.slice_of_life_narrator.initialize()
    
    narrator = app_ctx.slice_of_life_narrator
    world_state = await _ensure_world_state_from_ctx(app_ctx)
    
    result = await narrator.routine_narrator.run(
        messages=[{"role": "user", "content": f"Narrate {activity} routine"}],
        context=narrator.context,
        tool_calls=[{
            "tool": narrator.narrate_daily_routine,
            "kwargs": {
                "activity": activity,
                "world_state": world_state.model_dump() if world_state else {},
                "involved_npcs": involved_npcs or []
            }
        }]
    )
    
    if result and hasattr(result, 'data'):
        return json.dumps(result.data.model_dump() if hasattr(result.data, 'model_dump') else result.data)
    
    return json.dumps({"description": f"You go about {activity}..."})

@function_tool
async def generate_ambient_narration(
    ctx: RunContextWrapper[NyxContext],
    focus: str = "atmosphere",
    intensity: float = 0.5
) -> str:
    """Generate ambient world narration using the narrator."""
    app_ctx = _get_app_ctx(ctx)
    
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(
            app_ctx.user_id,
            app_ctx.conversation_id
        )
        await app_ctx.slice_of_life_narrator.initialize()
    
    narrator = app_ctx.slice_of_life_narrator
    world_state = await _ensure_world_state_from_ctx(app_ctx)
    
    # Use the narrator's ambient narration tool
    from agents import RunContextWrapper as NarratorWrapper
    result = await narrator.scene_narrator.run(
        messages=[{"role": "user", "content": f"Generate {focus} ambient narration"}],
        context=narrator.context,
        tool_calls=[{
            "tool": narrator.generate_ambient_narration,
            "kwargs": {
                "focus": focus,
                "world_state": world_state.model_dump() if world_state else {},
                "intensity": intensity
            }
        }]
    )
    
    if result and hasattr(result, 'data'):
        return json.dumps(result.data.model_dump() if hasattr(result.data, 'model_dump') else result.data)
    
    return json.dumps({"description": "The world continues around you..."})

@function_tool  
async def detect_narrative_patterns(
    ctx: RunContextWrapper[NyxContext]
) -> str:
    """Detect emergent narrative patterns using the narrator."""
    app_ctx = _get_app_ctx(ctx)
    
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(
            app_ctx.user_id,
            app_ctx.conversation_id
        )
        await app_ctx.slice_of_life_narrator.initialize()
    
    narrator = app_ctx.slice_of_life_narrator
    
    # Use narrator's emergent narrative detection
    patterns = await narrator.generate_emergent_narrative()
    
    return json.dumps(patterns)

@function_tool
async def generate_npc_dialogue(
    ctx: RunContextWrapper[NyxContext],
    npc_id: int,
    situation: str
) -> str:
    """Generate NPC dialogue using the sophisticated narrator with canonical tracking"""
    
    app_ctx = _get_app_ctx(ctx)
    
    # Import canon functions
    from lore.core.canon import (
        ensure_canonical_context,
        log_canonical_event,
        find_or_create_npc
    )
    
    # Ensure canonical context
    canonical_ctx = ensure_canonical_context({
        'user_id': app_ctx.user_id,
        'conversation_id': app_ctx.conversation_id
    })
    
    # Ensure narrator is initialized
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(
            app_ctx.user_id,
            app_ctx.conversation_id
        )
        await app_ctx.slice_of_life_narrator.initialize()
    
    narrator = app_ctx.slice_of_life_narrator
    world_state = await _ensure_world_state_from_ctx(app_ctx)
    
    # Ensure NPC exists canonically
    async with get_db_connection_context() as conn:
        # Get NPC name if available
        npc_name = f"NPC_{npc_id}"
        npc_data = await conn.fetchrow("""
            SELECT npc_name FROM NPCStats 
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, npc_id, app_ctx.user_id, app_ctx.conversation_id)
        
        if npc_data:
            npc_name = npc_data['npc_name']
        
        # Ensure NPC exists canonically
        canonical_npc_id = await find_or_create_npc(
            canonical_ctx, conn,
            npc_name=npc_name,
            role="dialogue_participant"
        )
        
        # Log dialogue generation event
        await log_canonical_event(
            canonical_ctx, conn,
            f"Generating dialogue for {npc_name} in situation: {situation[:50]}",
            tags=["dialogue", "generation", f"npc_{npc_id}"],
            significance=5
        )
    
    # Use the narrator's dialogue generation
    dialogue_result = await narrator.dialogue_writer.run(
        messages=[{"role": "user", "content": f"Generate dialogue for situation: {situation}"}],
        context=narrator.context,
        tool_calls=[{
            "tool": narrator.generate_npc_dialogue,
            "kwargs": {
                "npc_id": npc_id,
                "situation": situation,
                "world_state": world_state.model_dump() if world_state else {},
                "player_input": app_ctx.current_context.get("user_input")
            }
        }]
    )
    
    if dialogue_result and hasattr(dialogue_result, 'data'):
        dialogue_data = dialogue_result.data
        
        # Log generated dialogue canonically
        async with get_db_connection_context() as conn:
            await log_canonical_event(
                canonical_ctx, conn,
                f"{npc_name} spoke: {dialogue_data.dialogue[:100] if hasattr(dialogue_data, 'dialogue') else 'dialogue generated'}",
                tags=["dialogue", "npc_speech", f"npc_{npc_id}"],
                significance=6
            )
        
        return json.dumps({
            "npc_id": npc_id,
            "dialogue": dialogue_data.dialogue if hasattr(dialogue_data, 'dialogue') else str(dialogue_data),
            "tone": dialogue_data.tone if hasattr(dialogue_data, 'tone') else 'neutral',
            "subtext": dialogue_data.subtext if hasattr(dialogue_data, 'subtext') else '',
            "requires_response": dialogue_data.requires_response if hasattr(dialogue_data, 'requires_response') else False
        })
    
    return json.dumps({"dialogue": "...", "npc_id": npc_id})
  
@function_tool
async def retrieve_memories(ctx: RunContextWrapper[NyxContext], payload: RetrieveMemoriesInput) -> str:
    """Retrieve relevant memories for Nyx."""
    data = RetrieveMemoriesInput.model_validate(payload or {})
    query = data.query
    limit = data.limit
    memory_system = ctx.context.memory_system
    
    result = await memory_system.recall(
        entity_type=Config.ENTITY_TYPE_INTEGRATED,
        entity_id=0,
        query=query,
        limit=limit
    )
    
    memories_raw = result.get("memories", [])
    memories = [
        MemoryItem(
            id=str(m.get("id") or m.get("memory_id") or ""),
            text=m["text"],
            relevance=float(m.get("relevance", 0.0)),
            tags=m.get("tags", [])
        )
        for m in memories_raw
    ]
    
    formatted_memories = []
    for memory in memories:
        relevance = memory.relevance
        confidence_marker = "vividly recall" if relevance > Config.VIVID_RECALL_THRESHOLD else \
                          "remember" if relevance > Config.REMEMBER_THRESHOLD else \
                          "think I recall" if relevance > Config.THINK_RECALL_THRESHOLD else \
                          "vaguely remember"
        
        formatted_memories.append(f"I {confidence_marker}: {memory.text}")
    
    formatted_text = "\n".join(formatted_memories) if formatted_memories else "No relevant memories found."

    return MemorySearchResult(
        memories=memories,
        formatted_text=formatted_text
    ).model_dump_json()

@function_tool
async def add_memory(ctx: RunContextWrapper[NyxContext], payload: AddMemoryInput) -> str:
    """Add a new memory for Nyx using canonical system."""
    data = AddMemoryInput.model_validate(payload or {})
    memory_text = data.memory_text
    memory_type = data.memory_type
    significance = data.significance
    
    # Use canonical context
    canonical_ctx = ensure_canonical_context(ctx.context)
    
    # Store memory canonically
    async with get_db_connection_context() as conn:
        # This should use the canonical memory system
        await log_canonical_event(
            canonical_ctx, conn,
            f"Memory stored: {memory_text[:100]}...",
            tags=["memory", memory_type, "nyx_generated"],
            significance=significance
        )
    
    # Also use the existing memory system
    memory_system = ctx.context.memory_system
    result = await memory_system.remember(
        entity_type="integrated",
        entity_id=0,
        memory_text=memory_text,
        importance="high" if significance >= 8 else "medium",
        emotional=True,
        tags=["agent_generated", memory_type]
    )
    
    return MemoryStorageResult(
        memory_id=str(result.get("memory_id", "unknown")),
        success=True
    ).model_dump_json()

@function_tool
async def get_user_model_guidance(ctx: RunContextWrapper[NyxContext], payload: EmptyInput) -> str:
    """Get guidance for how Nyx should respond based on the user model."""
    _ = payload  # unused
    user_model_manager = ctx.context.user_model
    guidance = await user_model_manager.get_response_guidance()
    
    top_kinks = guidance.get("top_kinks", [])
    behavior_patterns = guidance.get("behavior_patterns", {})
    suggested_intensity = guidance.get("suggested_intensity", 0.5)
    reflections = guidance.get("reflections", [])
    
    return UserGuidanceResult(
        top_kinks=top_kinks,
        behavior_patterns=dict_to_kvlist(behavior_patterns),
        suggested_intensity=suggested_intensity,
        reflections=reflections
    ).model_dump_json()

@function_tool
async def detect_user_revelations(ctx: RunContextWrapper[NyxContext], payload: DetectUserRevelationsInput) -> str:
    """Detect if user is revealing new preferences or patterns."""
    data = DetectUserRevelationsInput.model_validate(payload or {})
    user_message = data.user_message
    lower_message = user_message.lower()
    revelations = []
    
    kink_keywords = {
        "ass": ["ass", "booty", "behind", "rear"],
        "feet": ["feet", "foot", "toes"],
        "goth": ["goth", "gothic", "dark", "black clothes"],
        "tattoos": ["tattoo", "ink", "inked"],
        "piercings": ["piercing", "pierced", "stud", "ring"],
        "latex": ["latex", "rubber", "shiny"],
        "leather": ["leather", "leathery"],
        "humiliation": ["humiliate", "embarrassed", "ashamed", "pathetic"],
        "submission": ["submit", "obey", "serve", "kneel"]
    }
    
    for kink, keywords in kink_keywords.items():
        if any(keyword in lower_message for keyword in keywords):
            sentiment = "neutral"
            pos_words = ["like", "love", "enjoy", "good", "great", "nice", "yes", "please"]
            neg_words = ["don't", "hate", "dislike", "bad", "worse", "no", "never"]
            
            pos_count = sum(1 for word in pos_words if word in lower_message)
            neg_count = sum(1 for word in neg_words if word in lower_message)
            
            if pos_count > neg_count:
                sentiment = "positive"
                intensity = 0.7
            elif neg_count > pos_count:
                sentiment = "negative" 
                intensity = 0.0
            else:
                intensity = 0.4
                
            revelation_data = {
                "type": "kink_preference",
                "kink": kink,
                "intensity": intensity,
                "source": "explicit_negative_mention" if sentiment == "negative" else "explicit_mention",
                "sentiment": sentiment
            }
            
            revelations.append(dict_to_kvlist(revelation_data))
    
    if "don't tell me what to do" in lower_message or "i won't" in lower_message:
        revelation_data = {
            "type": "behavior_pattern",
            "pattern": "resistance",
            "intensity": 0.6,
            "source": "explicit_statement"
        }
        revelations.append(dict_to_kvlist(revelation_data))
    
    if "yes mistress" in lower_message or "i'll obey" in lower_message:
        revelation_data = {
            "type": "behavior_pattern",
            "pattern": "submission",
            "intensity": 0.8,
            "source": "explicit_statement"
        }
        revelations.append(dict_to_kvlist(revelation_data))
    
    # Save revelations to database if found
    if revelations and ctx.context.user_model:
        for revelation_kv in revelations:
            revelation = kvlist_to_dict(revelation_kv)
            if revelation["type"] == "kink_preference":
                await ctx.context.user_model.update_kink_preference(
                    revelation["kink"],
                    revelation["intensity"],
                    revelation["source"]
                )
    
    return RevelationDetectionResult(
        revelations=revelations,
        has_revelations=len(revelations) > 0
    ).model_dump_json()

@function_tool
async def generate_image_from_scene(
    ctx: RunContextWrapper[NyxContext],
    payload: GenerateImageFromSceneInput
) -> str:
    """Generate an image for the current scene."""
    from routes.ai_image_generator import generate_roleplay_image_from_gpt

    data = GenerateImageFromSceneInput.model_validate(payload or {})
    image_data = data.model_dump()
    
    result = await generate_roleplay_image_from_gpt(
        image_data,
        ctx.context.user_id,
        ctx.context.conversation_id
    )
    
    if result and "image_urls" in result and result["image_urls"]:
        return ImageGenerationResult(
            success=True,
            image_url=result["image_urls"][0],
            error=None
        ).model_dump_json()
    else:
        return ImageGenerationResult(
            success=False,
            image_url=None,
            error="Failed to generate image"
        ).model_dump_json()

@function_tool
async def calculate_and_update_emotional_state(ctx: RunContextWrapper[NyxContext], payload: CalculateEmotionalStateInput) -> str:
    """Calculate emotional impact and immediately update the emotional state."""
    data = CalculateEmotionalStateInput.model_validate(payload or {})
    context_dict = kvlist_to_dict(data.context)

    # First calculate the new state
    result = await calculate_emotional_impact(ctx, data)
    emotional_data = json.loads(result)
    
    # Immediately update the context with the new state
    ctx.context.emotional_state.update({
        "valence": emotional_data["valence"],
        "arousal": emotional_data["arousal"],
        "dominance": emotional_data["dominance"]
    })
    
    # Save to database with its own connection
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id) 
            DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
        """, ctx.context.user_id, ctx.context.conversation_id, json.dumps(ctx.context.emotional_state))
    
    # Return the result with confirmation of update
    emotional_data["state_updated"] = True
    
    return EmotionalCalculationResult(
        valence=emotional_data["valence"],
        arousal=emotional_data["arousal"],
        dominance=emotional_data["dominance"],
        primary_emotion=emotional_data["primary_emotion"],
        changes=EmotionalChanges(
            valence_change=emotional_data["changes"]["valence_change"],
            arousal_change=emotional_data["changes"]["arousal_change"],
            dominance_change=emotional_data["changes"]["dominance_change"]
        ),
        state_updated=True
    ).model_dump_json()

@function_tool
async def calculate_emotional_impact(ctx: RunContextWrapper[NyxContext], payload: CalculateEmotionalStateInput) -> str:
    """Calculate emotional impact of current context using the emotional core system."""
    data = CalculateEmotionalStateInput.model_validate(payload or {})
    context_dict = kvlist_to_dict(data.context)
    current_state = ctx.context.emotional_state.copy()
    
    # Calculate emotional changes based on context
    valence_change = 0.0
    arousal_change = 0.0
    dominance_change = 0.0
    
    # Analyze context for emotional triggers
    context_text_lower = get_context_text_lower(context_dict)
    
    if "conflict" in context_text_lower:
        arousal_change += 0.2
        valence_change -= 0.1
    if "submission" in context_text_lower:
        dominance_change += 0.1
        arousal_change += 0.1
    if "praise" in context_text_lower or "good" in context_text_lower:
        valence_change += 0.2
    if "resistance" in context_text_lower:
        arousal_change += 0.15
        dominance_change -= 0.05
    
    # Get memory emotional impact
    memory_impact = await _get_memory_emotional_impact(ctx, context_dict)
    valence_change += memory_impact["valence"] * 0.3
    arousal_change += memory_impact["arousal"] * 0.3
    dominance_change += memory_impact["dominance"] * 0.3
    
    # Use EmotionalCore if available for more nuanced analysis
    if ctx.context.emotional_core:
        try:
            core_analysis = ctx.context.emotional_core.analyze(str(context_dict))
            valence_change += core_analysis.get("valence_delta", 0) * 0.5
            arousal_change += core_analysis.get("arousal_delta", 0) * 0.5
        except Exception as e:
            logger.debug(f"EmotionalCore analysis failed: {e}", exc_info=True)
    
    # Apply changes with bounds
    new_valence = max(-1, min(1, current_state["valence"] + valence_change))
    new_arousal = max(0, min(1, current_state["arousal"] + arousal_change))
    new_dominance = max(0, min(1, current_state["dominance"] + dominance_change))
    
    # Determine primary emotion based on VAD model
    primary_emotion = "neutral"
    if new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "excited"
    elif new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "content"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "frustrated"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "disappointed"
    elif new_dominance > Config.HIGH_DOMINANCE_THRESHOLD:
        primary_emotion = "commanding"
    
    return EmotionalCalculationResult(
        valence=new_valence,
        arousal=new_arousal,
        dominance=new_dominance,
        primary_emotion=primary_emotion,
        changes=EmotionalChanges(
            valence_change=valence_change,
            arousal_change=arousal_change,
            dominance_change=dominance_change
        ),
        state_updated=None
    ).model_dump_json()

async def _get_memory_emotional_impact(ctx: RunContextWrapper[NyxContext], context: Dict[str, Any]) -> Dict[str, float]:
    """Get emotional impact from related memories"""
    return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

@function_tool
async def update_relationship_state(
    ctx: RunContextWrapper[NyxContext],
    payload: UpdateRelationshipStateInput
) -> str:
    """Update relationship state canonically with full governance integration."""
    data = UpdateRelationshipStateInput.model_validate(payload or {})
    entity_id = data.entity_id
    trust_change = data.trust_change
    power_change = data.power_change
    bond_change = data.bond_change
    
    relationships = ctx.context.relationship_states
    
    # Create canonical context
    canonical_ctx = ensure_canonical_context({
        'user_id': ctx.context.user_id,
        'conversation_id': ctx.context.conversation_id
    })
    
    # Initialize relationship if it doesn't exist
    if entity_id not in relationships:
        relationships[entity_id] = {
            "trust": 0.5,
            "power_dynamic": 0.5,
            "emotional_bond": 0.3,
            "interaction_count": 0,
            "last_interaction": time.time(),
            "type": "neutral"
        }
    
    rel = relationships[entity_id]
    old_state = {
        "trust": rel["trust"],
        "power_dynamic": rel["power_dynamic"],
        "emotional_bond": rel["emotional_bond"]
    }
    
    # Apply changes with bounds checking
    rel["trust"] = max(0, min(1, rel["trust"] + trust_change))
    rel["power_dynamic"] = max(0, min(1, rel["power_dynamic"] + power_change))
    rel["emotional_bond"] = max(0, min(1, rel["emotional_bond"] + bond_change))
    rel["interaction_count"] += 1
    rel["last_interaction"] = time.time()
    
    # Determine relationship type based on new values
    if rel["trust"] > Config.INTIMATE_TRUST_THRESHOLD and rel["emotional_bond"] > Config.INTIMATE_BOND_THRESHOLD:
        rel["type"] = "intimate"
    elif rel["trust"] > Config.FRIENDLY_TRUST_THRESHOLD:
        rel["type"] = "friendly"
    elif rel["trust"] < Config.HOSTILE_TRUST_THRESHOLD:
        rel["type"] = "hostile"
    elif rel["power_dynamic"] > Config.DOMINANT_POWER_THRESHOLD:
        rel["type"] = "dominant"
    elif rel["power_dynamic"] < Config.SUBMISSIVE_POWER_THRESHOLD:
        rel["type"] = "submissive"
    else:
        rel["type"] = "neutral"
    
    # Calculate significance of changes for canonical logging
    total_change = abs(trust_change) + abs(power_change) + abs(bond_change)
    significance = 5  # base
    if total_change > 0.3:
        significance = 7  # major change
    if total_change > 0.5:
        significance = 8  # dramatic change
    if rel["type"] != "neutral":
        significance += 1  # relationship has clear type
    
    # Save to database with canonical integration
    async with get_db_connection_context() as conn:
        try:
            # First, ensure the entity exists canonically (if it's an NPC)
            if entity_id.isdigit():  # Assume numeric entity_id means NPC
                npc_id = int(entity_id)
                
                # Get NPC details for canonical creation
                npc_data = await conn.fetchrow("""
                    SELECT npc_name, role, affiliations 
                    FROM NPCStats 
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, ctx.context.user_id, ctx.context.conversation_id)
                
                if npc_data:
                    # Ensure NPC exists in canonical system
                    canonical_npc_id = await find_or_create_npc(
                        canonical_ctx, conn,
                        npc_name=npc_data['npc_name'],
                        role=npc_data.get('role', 'individual'),
                        affiliations=npc_data.get('affiliations', [])
                    )
            
            # Get existing evolution history
            existing = await conn.fetchrow("""
                SELECT evolution_history 
                FROM RelationshipEvolution 
                WHERE user_id = $1 AND conversation_id = $2 
                    AND npc1_id = $3 AND entity2_type = $4 AND entity2_id = $5
            """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id)
            
            # Build evolution history
            evolution_history = []
            if existing and existing["evolution_history"]:
                try:
                    evolution_history = json.loads(existing["evolution_history"])
                except (json.JSONDecodeError, TypeError):
                    evolution_history = []
            
            # Add new entry (keep last 50 entries)
            evolution_entry = {
                "timestamp": time.time(),
                "trust": rel["trust"],
                "power": rel["power_dynamic"],
                "bond": rel["emotional_bond"],
                "changes": {
                    "trust": trust_change,
                    "power": power_change,
                    "bond": bond_change
                },
                "interaction_count": rel["interaction_count"],
                "old_type": old_state.get("type", "neutral"),
                "new_type": rel["type"]
            }
            evolution_history.append(evolution_entry)
            evolution_history = evolution_history[-50:]  # Keep last 50
            
            # Use canonical update system
            update_result = await update_entity_with_governance(
                canonical_ctx, conn,
                entity_type="RelationshipEvolution",
                entity_id=0,  # Will be handled by upsert logic
                updates={
                    "relationship_type": rel["type"],
                    "current_stage": rel["type"], 
                    "progress_to_next": min(1.0, total_change * 2),  # Scale change to progress
                    "evolution_history": json.dumps(evolution_history),
                    "trust_level": rel["trust"],
                    "power_dynamic": rel["power_dynamic"],
                    "emotional_bond": rel["emotional_bond"],
                    "last_interaction": rel["last_interaction"]
                },
                reason=f"Relationship evolution: trust{trust_change:+.3f}, power{power_change:+.3f}, bond{bond_change:+.3f}",
                significance=significance
            )
            
            # If canonical update failed, fall back to direct update
            if not update_result.get("success"):
                await conn.execute("""
                    INSERT INTO RelationshipEvolution 
                    (user_id, conversation_id, npc1_id, entity2_type, entity2_id, 
                     relationship_type, current_stage, progress_to_next, evolution_history)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (user_id, conversation_id, npc1_id, entity2_type, entity2_id)
                    DO UPDATE SET 
                        relationship_type = $6,
                        current_stage = $7,
                        progress_to_next = $8,
                        evolution_history = $9
                """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id,
                     rel["type"], rel["type"], min(1.0, total_change * 2), json.dumps(evolution_history))
            
            # Log the relationship change canonically
            change_description = []
            if abs(trust_change) > 0.01:
                change_description.append(f"trust {trust_change:+.3f}")
            if abs(power_change) > 0.01:
                change_description.append(f"power {power_change:+.3f}")
            if abs(bond_change) > 0.01:
                change_description.append(f"bond {bond_change:+.3f}")
            
            if change_description:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship with {entity_id} evolved: {', '.join(change_description)} → {rel['type']} relationship",
                    tags=["relationship", "evolution", rel["type"], "nyx_update"],
                    significance=significance
                )
            
            # Log relationship type changes
            if old_state.get("type", "neutral") != rel["type"]:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship type shifted: {old_state.get('type', 'neutral')} → {rel['type']} with {entity_id}",
                    tags=["relationship", "type_change", rel["type"], "milestone"],
                    significance=significance + 1
                )
            
            # Check for significant thresholds crossed
            threshold_events = []
            
            # Trust thresholds
            if old_state["trust"] < Config.INTIMATE_TRUST_THRESHOLD <= rel["trust"]:
                threshold_events.append("intimate_trust_achieved")
            elif old_state["trust"] >= Config.HOSTILE_TRUST_THRESHOLD > rel["trust"]:
                threshold_events.append("trust_broken")
            
            # Power thresholds  
            if old_state["power_dynamic"] < Config.DOMINANT_POWER_THRESHOLD <= rel["power_dynamic"]:
                threshold_events.append("dominance_established")
            elif old_state["power_dynamic"] >= Config.SUBMISSIVE_POWER_THRESHOLD > rel["power_dynamic"]:
                threshold_events.append("submission_deepened")
            
            # Bond thresholds
            if old_state["emotional_bond"] < Config.INTIMATE_BOND_THRESHOLD <= rel["emotional_bond"]:
                threshold_events.append("deep_bond_formed")
            
            # Log threshold crossings
            for event in threshold_events:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship milestone: {event.replace('_', ' ')} with {entity_id}",
                    tags=["relationship", "milestone", event, entity_id],
                    significance=9  # Milestones are highly significant
                )
            
        except Exception as e:
            logger.error(f"Error in canonical relationship update: {e}", exc_info=True)
            # Continue with function to avoid breaking the flow
    
    # Create relationship changes for response
    changes = RelationshipChanges(
        trust=trust_change,
        power=power_change,
        bond=bond_change
    )
    
    # Create relationship state output
    relationship_state_out = RelationshipStateOut(
        trust=rel["trust"],
        power_dynamic=rel["power_dynamic"],
        emotional_bond=rel["emotional_bond"],
        interaction_count=rel["interaction_count"],
        last_interaction=rel["last_interaction"],
        type=rel["type"]
    )
    
    # Build result
    result = RelationshipUpdateResult(
        entity_id=entity_id,
        relationship=relationship_state_out,
        changes=changes
    )
    
    # Store evolution pattern for learning
    pattern_data = {
        "entity_id": entity_id,
        "change_magnitude": total_change,
        "relationship_type": rel["type"],
        "direction": "positive" if total_change > 0 else "negative" if total_change < 0 else "neutral",
        "trust_direction": "up" if trust_change > 0 else "down" if trust_change < 0 else "stable",
        "power_direction": "up" if power_change > 0 else "down" if power_change < 0 else "stable",
        "bond_direction": "up" if bond_change > 0 else "down" if bond_change < 0 else "stable"
    }
    
    # Learn from this relationship interaction
    await ctx.context.learn_from_interaction(
        action=f"relationship_update_{rel['type']}",
        outcome=f"total_change_{total_change:.2f}",
        success=total_change > 0.1  # Consider it successful if meaningful change occurred
    )
    
    return result.model_dump_json()

@function_tool
async def check_performance_metrics(ctx: RunContextWrapper[NyxContext], payload: EmptyInput) -> str:
    """Check current performance metrics and apply remediation if needed."""
    _ = payload  # unused
    metrics = ctx.context.performance_metrics

    # Refresh CPU & RAM values using centralized helper
    process = get_process_info()
    if process:
        memory_info = safe_process_metric(process, 'memory_info')
        metrics["memory_usage"] = bytes_to_mb(memory_info)
    else:
        metrics["memory_usage"] = 0

    metrics["cpu_usage"] = ctx.context.get_cpu_usage()

    suggestions, actions_taken = [], []

    # Health checks
    avg_rt = _calculate_avg_response_time(metrics["response_times"])
    if avg_rt > Config.HIGH_RESPONSE_TIME_THRESHOLD:
        suggestions.append("Response times are high – consider caching frequent queries")

    if metrics["memory_usage"] > Config.HIGH_MEMORY_THRESHOLD_MB:
        suggestions.append("High memory usage detected – triggering cleanup")
        memory_before = metrics["memory_usage"]
        await ctx.context.handle_high_memory_usage()
        # Re-check memory after cleanup
        if process:
            memory_after = bytes_to_mb(safe_process_metric(process, 'memory_info'))
            logger.info(f"Memory cleanup: {memory_before:.2f}MB -> {memory_after:.2f}MB")
        actions_taken.append("memory_cleanup")

    if metrics["total_actions"]:
        success_rate = metrics["successful_actions"] / metrics["total_actions"]
        if success_rate < Config.MIN_SUCCESS_RATE:
            suggestions.append("Success rate below 80% – review error patterns")

    if metrics["error_rates"]["total"] > Config.HIGH_ERROR_COUNT:
        suggestions.append("High error count – clearing old errors")
        ctx.context.error_log = ctx.context.error_log[-Config.MAX_ERROR_LOG_ENTRIES:]
        actions_taken.append("error_log_cleanup")

    return PerformanceMetricsResult(
        metrics=PerformanceNumbers(
            memory_mb=metrics["memory_usage"],
            cpu_percent=metrics["cpu_usage"],
            avg_response_time=(
                sum(metrics["response_times"]) / len(metrics["response_times"])
                if metrics["response_times"] else 0
            ),
            success_rate=(
                metrics["successful_actions"] / metrics["total_actions"]
                if metrics["total_actions"] else 1.0
            )
        ),
        suggestions=suggestions,
        actions_taken=actions_taken,
        health="good" if not suggestions else "needs_attention",
    ).model_dump_json()

@function_tool
async def get_activity_recommendations(
    ctx: RunContextWrapper[NyxContext],
    payload: GetActivityRecommendationsInput
) -> str:
    """Get activity recommendations based on current context."""
    data = GetActivityRecommendationsInput.model_validate(payload or {})
    scenario_type = data.scenario_type
    npc_ids = data.npc_ids
    
    activities = []
    
    # Copy relationship states to avoid mutation during iteration
    relationship_states_copy = dict(ctx.context.relationship_states)
    
    # Training activities
    if "training" in scenario_type.lower() or any(rel.get("type") == "submissive" 
        for rel in relationship_states_copy.values()):
        activities.extend([
            ActivityRec(
                name="Obedience Training",
                description="Test and improve submission through structured exercises",
                requirements=["trust > 0.4", "submission tendency"],
                duration="15-30 minutes",
                intensity="medium"
            ),
            ActivityRec(
                name="Position Practice",
                description="Learn and perfect submissive positions",
                requirements=["trust > 0.5"],
                duration="10-20 minutes",
                intensity="low-medium"
            )
        ])
    
    # Social activities
    if npc_ids and len(npc_ids) > 0:
        activities.append(ActivityRec(
            name="Group Dynamics Exercise",
            description="Explore power dynamics with multiple participants",
            requirements=["multiple NPCs present"],
            duration="20-40 minutes",
            intensity="variable"
        ))
    
    # Intimate activities
    for entity_id, rel in relationship_states_copy.items():
        if rel.get("type") == "intimate" and rel.get("trust", 0) > 0.7:
            activities.append(ActivityRec(
                name="Intimate Scene",
                description=f"Deepen connection with trusted partner",
                requirements=["high trust", "intimate relationship"],
                duration="30-60 minutes",
                intensity="high",
                partner_id=entity_id
            ))
            break
    
    # Default activities
    activities.extend([
        ActivityRec(
            name="Exploration",
            description="Discover new areas or items",
            requirements=[],
            duration="10-30 minutes",
            intensity="low"
        ),
        ActivityRec(
            name="Conversation",
            description="Engage in meaningful dialogue",
            requirements=[],
            duration="5-15 minutes",
            intensity="low"
        )
    ])
    
    return ActivityRecommendationsResult(
        recommendations=activities[:5],  # Top 5 activities
        total_available=len(activities)
    ).model_dump_json()

@function_tool
async def manage_beliefs(ctx: RunContextWrapper[NyxContext], payload: ManageBeliefsInput) -> str:
    """Manage belief system operations."""
    data = ManageBeliefsInput.model_validate(payload or {})
    action = data.action
    belief_data = data.belief_data
    
    if not ctx.context.belief_system:
        return BeliefManagementResult(
            result="",
            error="Belief system not available"
        ).model_dump_json()
    
    try:
        if action == "get":
            entity_id = belief_data.entity_id
            beliefs = await ctx.context.belief_system.get_beliefs(entity_id)
            return BeliefManagementResult(
                result=dict_to_kvlist(beliefs),
                error=None
            ).model_dump_json()
        
        elif action == "update":
            entity_id = belief_data.entity_id
            belief_type = belief_data.type
            content = kvlist_to_dict(belief_data.content)
            await ctx.context.belief_system.update_belief(entity_id, belief_type, content)
            return BeliefManagementResult(
                result="Belief updated successfully",
                error=None
            ).model_dump_json()
        
        elif action == "query":
            query = belief_data.query or ""
            results = await ctx.context.belief_system.query_beliefs(query)
            return BeliefManagementResult(
                result=dict_to_kvlist(results),
                error=None
            ).model_dump_json()
        
        else:
            return BeliefManagementResult(
                result="",
                error=f"Unknown action: {action}"
            ).model_dump_json()
            
    except Exception as e:
        logger.error(f"Error managing beliefs: {e}", exc_info=True)
        return BeliefManagementResult(
            result="",
            error=str(e)
        ).model_dump_json()

@function_tool
async def score_decision_options(
    ctx: RunContextWrapper[NyxContext],
    payload: ScoreDecisionOptionsInput
) -> str:
    """Score decision options using advanced decision engine logic."""
    data = ScoreDecisionOptionsInput.model_validate(payload or {})
    options = data.options
    decision_context = kvlist_to_dict(data.decision_context)
    
    scored_options = []
    
    for option in options:
        # Base score from context relevance
        context_score = _calculate_context_relevance(option.model_dump(), decision_context)
        
        # Emotional alignment score
        emotional_score = _calculate_emotional_alignment(option.model_dump(), ctx.context.emotional_state)
        
        # Pattern-based score
        pattern_score = _calculate_pattern_score(option.model_dump(), ctx.context.learned_patterns)
        
        # Relationship impact score
        relationship_score = _calculate_relationship_impact(option.model_dump(), ctx.context.relationship_states)
        
        # Calculate weighted final score
        weights = {
            "context": 0.3,
            "emotional": 0.25,
            "pattern": 0.25,
            "relationship": 0.2
        }
        
        final_score = (
            context_score * weights["context"] +
            emotional_score * weights["emotional"] +
            pattern_score * weights["pattern"] +
            relationship_score * weights["relationship"]
        )
        
        scored_options.append(ScoredOption(
            option=option,
            score=final_score,
            components=ScoreComponents(
                context=context_score,
                emotional=emotional_score,
                pattern=pattern_score,
                relationship=relationship_score
            )
        ))
    
    # Sort by score
    scored_options.sort(key=lambda x: x.score, reverse=True)
    
    # If all scores are too low, include a fallback
    if all(opt.score < Config.MIN_DECISION_SCORE for opt in scored_options):
        fallback = _get_fallback_decision(options)
        fallback_scored = ScoredOption(
            option=fallback,
            score=Config.FALLBACK_DECISION_SCORE,
            components=ScoreComponents(
                context=Config.FALLBACK_DECISION_SCORE,
                emotional=Config.FALLBACK_DECISION_SCORE,
                pattern=Config.FALLBACK_DECISION_SCORE,
                relationship=Config.FALLBACK_DECISION_SCORE
            ),
            is_fallback=True
        )
        scored_options.insert(0, fallback_scored)
    
    return DecisionScoringResult(
        scored_options=scored_options,
        best_option=scored_options[0].option,
        confidence=scored_options[0].score
    ).model_dump_json()

def _calculate_context_relevance(option: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Calculate how relevant an option is to context"""
    score = 0.5  # Base score
    
    # Check keyword matches
    option_keywords = set(str(option).lower().split())
    context_keywords = set(str(context).lower().split())
    
    overlap = len(option_keywords.intersection(context_keywords))
    if overlap > 0:
        score += min(0.3, overlap * 0.1)
    
    # Check for scenario type match
    if context.get("scenario_type") and context["scenario_type"] in str(option):
        score += 0.2
    
    return min(1.0, score)

def _calculate_emotional_alignment(option: Dict[str, Any], emotional_state: Dict[str, float]) -> float:
    """Calculate emotional alignment score"""
    # High dominance favors assertive options
    if "command" in str(option).lower() or "control" in str(option).lower():
        return emotional_state.get("dominance", 0.5)
    
    # High arousal favors intense options
    if "intense" in str(option).lower() or "extreme" in str(option).lower():
        return emotional_state.get("arousal", 0.5)
    
    # Positive valence favors rewarding options
    if "reward" in str(option).lower() or "praise" in str(option).lower():
        return (emotional_state.get("valence", 0) + 1) / 2
    
    return 0.5

def _calculate_pattern_score(option: Dict[str, Any], learned_patterns: Dict[str, Any]) -> float:
    """Calculate score based on learned patterns"""
    if not learned_patterns:
        return 0.5
    
    # Find relevant patterns
    option_str = str(option).lower()
    relevant_scores = []
    
    # Create a copy to avoid mutation during iteration
    patterns_copy = dict(learned_patterns)
    for pattern_key, pattern_data in patterns_copy.items():
        if any(keyword in option_str for keyword in pattern_key.split("_")):
            success_rate = pattern_data.get("success_rate", 0.5)
            recency_factor = 1.0 / (1 + (time.time() - pattern_data.get("last_seen", 0)) / 3600)
            relevant_scores.append(success_rate * recency_factor)
    
    return sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.5

def _calculate_relationship_impact(option: Dict[str, Any], relationship_states: Dict[str, Dict[str, Any]]) -> float:
    """Calculate relationship impact score"""
    if not relationship_states:
        return 0.5
    
    # Average trust level affects willingness to take actions
    avg_trust = sum(rel.get("trust", 0.5) for rel in relationship_states.values()) / len(relationship_states)
    
    # Risky options need higher trust
    if "risk" in str(option).lower() or "challenge" in str(option).lower():
        return avg_trust
    
    # Safe options work with any trust level
    return 0.5 + (avg_trust * 0.5)

def _get_fallback_decision(options: List[DecisionOption]) -> DecisionOption:
    """Get a safe fallback decision"""
    # Prefer conversation or observation options
    safe_words = ["talk", "observe", "wait", "consider", "listen", "pause"]
    for option in options:
        if any(safe_word in str(option).lower() for safe_word in safe_words):
            return option
    
    # Otherwise return the first option
    return options[0] if options else DecisionOption(
        id="fallback",
        description="Take a moment to assess",
        metadata=DecisionMetadata()
    )

@function_tool
async def detect_conflicts_and_instability(
    ctx: RunContextWrapper[NyxContext],
    payload: DetectConflictsAndInstabilityInput
) -> str:
    """Detect conflicts and emotional instability in current scenario."""
    data = DetectConflictsAndInstabilityInput.model_validate(payload or {})
    scenario_state = kvlist_to_dict(data.scenario_state)
    
    conflicts = []
    instabilities = []
    
    # Check for relationship conflicts
    relationship_items = list(ctx.context.relationship_states.items())
    for i, (entity1_id, rel1) in enumerate(relationship_items):
        for entity2_id, rel2 in relationship_items[i+1:]:
            # Conflicting power dynamics
            if abs(rel1.get("power_dynamic", 0.5) - rel2.get("power_dynamic", 0.5)) > Config.POWER_CONFLICT_THRESHOLD:
                conflicts.append(ConflictItem(
                    type="power_conflict",
                    entities=[entity1_id, entity2_id],
                    severity=abs(rel1["power_dynamic"] - rel2["power_dynamic"]),
                    description="Conflicting power dynamics between entities"
                ))
            
            # Low mutual trust
            if rel1.get("trust", 0.5) < Config.HOSTILE_TRUST_THRESHOLD and rel2.get("trust", 0.5) < Config.HOSTILE_TRUST_THRESHOLD:
                conflicts.append(ConflictItem(
                    type="trust_conflict",
                    entities=[entity1_id, entity2_id],
                    severity=0.7,
                    description="Mutual distrust between entities"
                ))
    
    # Check for emotional instability
    emotional_state = ctx.context.emotional_state
    
    # High arousal with negative valence
    if emotional_state["arousal"] > Config.HIGH_AROUSAL_THRESHOLD and emotional_state["valence"] < Config.NEGATIVE_VALENCE_THRESHOLD:
        instabilities.append(InstabilityItem(
            type="emotional_volatility",
            severity=emotional_state["arousal"],
            description="High arousal with negative emotions",
            recommendation="De-escalation needed"
        ))
    
    # Rapid emotional changes
    if ctx.context.adaptation_history:
        recent_emotions = [h.get("emotional_state", {}) for h in ctx.context.adaptation_history[-5:]]
        if recent_emotions and any(recent_emotions):
            valence_values = [e.get("valence", 0) for e in recent_emotions if e]
            if valence_values:
                valence_variance = _calculate_variance(valence_values)
                if valence_variance > Config.EMOTIONAL_VARIANCE_THRESHOLD:
                    instabilities.append(InstabilityItem(
                        type="emotional_instability",
                        severity=min(1.0, valence_variance),
                        description="Rapid emotional swings detected",
                        recommendation="Stabilization recommended"
                    ))
    
    # Scenario-specific conflicts
    if scenario_state.get("objectives"):
        blocked_objectives = [obj for obj in scenario_state["objectives"] 
                             if obj.get("status") == "blocked"]
        if blocked_objectives:
            conflicts.append(ConflictItem(
                type="objective_conflict",
                severity=len(blocked_objectives) / len(scenario_state["objectives"]),
                description=f"{len(blocked_objectives)} objectives are blocked",
                blocked_objectives=[str(obj) for obj in blocked_objectives]
            ))
    
    # Calculate overall stability
    total_issues = len(conflicts) + len(instabilities)
    overall_stability = max(0.0, 1.0 - (total_issues / Config.MAX_STABILITY_ISSUES))
    
    # Only save scenario state if it's a significant change
    if ctx.context.scenario_state and ctx.context.scenario_state.get("active"):
        should_save = False
        
        # Check if this is a significant change
        if conflicts and any(c.severity > 0.7 for c in conflicts):
            should_save = True
        if instabilities and any(i.severity > 0.7 for i in instabilities):
            should_save = True
        if overall_stability < 0.3:
            should_save = True
            
        if should_save:
            async with get_db_connection_context() as conn:
                # Use UPSERT pattern to maintain one current state
                await conn.execute("""
                    INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, conversation_id) 
                    DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                """, ctx.context.user_id, ctx.context.conversation_id, 
                json.dumps(ctx.context.scenario_state, ensure_ascii=False))
    
    return ConflictDetectionResult(
        conflicts=conflicts,
        instabilities=instabilities,
        overall_stability=overall_stability,
        stability_note=f"{total_issues} issues detected (0 issues = 1.0 stability, 10+ issues = 0.0 stability)",
        requires_intervention=any(c.severity > 0.8 for c in conflicts + instabilities)
    ).model_dump_json()

# Add these near the function tool definitions

async def decide_image_generation_standalone(ctx: NyxContext, scene_text: str) -> str:
    """Standalone version for post-run enforcement"""
    scene_text_lower = scene_text.lower()
    
    # Calculate score based on scene characteristics
    score = 0.0
    
    # High impact visual keywords
    visual_keywords = ["dramatic", "intense", "beautiful", "transformation", "reveal", "climax", "pivotal"]
    for keyword in visual_keywords:
        if keyword in scene_text_lower:
            score += 0.2
    
    # Scene transitions
    if any(word in scene_text_lower for word in ["enter", "arrive", "transform", "change", "shift"]):
        score += 0.15
    
    # Emotional peaks
    if any(word in scene_text_lower for word in ["gasp", "shock", "awe", "breathtaking", "stunning"]):
        score += 0.25
    
    # Environmental descriptions
    if any(word in scene_text_lower for word in ["landscape", "environment", "setting", "atmosphere"]):
        score += 0.1
    
    # Cap score at 1.0
    score = min(1.0, score)
    
    # Dynamic threshold based on recent image generation
    recent_images = ctx.current_context.get("recent_image_count", 0)
    if recent_images > 3:
        threshold = 0.7
    elif recent_images > 1:
        threshold = 0.6
    else:
        threshold = 0.5
    
    # Determine if we should generate
    should_generate = score > threshold
    
    # Create appropriate prompt if generating
    image_prompt = None
    if should_generate:
        # Extract key visual elements
        visual_elements = []
        if "dramatic" in scene_text_lower:
            visual_elements.append("dramatic lighting")
        if "intense" in scene_text_lower:
            visual_elements.append("intense atmosphere")
        if "beautiful" in scene_text_lower:
            visual_elements.append("beautiful composition")
        
        image_prompt = f"Scene depicting: {', '.join(visual_elements) if visual_elements else 'atmospheric scene'}"
        
        # Update recent image count
        ctx.current_context["recent_image_count"] = recent_images + 1
    
    return ImageGenerationDecision(
        should_generate=should_generate,
        score=score,
        image_prompt=image_prompt,
        reasoning=f"Scene has visual impact score of {score:.2f} (threshold: {threshold:.2f})"
    ).model_dump_json()

@function_tool
async def decide_image_generation(ctx: RunContextWrapper[NyxContext], payload: DecideImageInput) -> str:
    """Decide whether an image should be generated for a scene."""
    data = DecideImageInput.model_validate(payload or {})
    scene_text = data.scene_text.lower()
    
    # Calculate score based on scene characteristics
    score = 0.0
    
    # High impact visual keywords
    visual_keywords = ["dramatic", "intense", "beautiful", "transformation", "reveal", "climax", "pivotal"]
    for keyword in visual_keywords:
        if keyword in scene_text:
            score += 0.2
    
    # Scene transitions
    if any(word in scene_text for word in ["enter", "arrive", "transform", "change", "shift"]):
        score += 0.15
    
    # Emotional peaks
    if any(word in scene_text for word in ["gasp", "shock", "awe", "breathtaking", "stunning"]):
        score += 0.25
    
    # Environmental descriptions
    if any(word in scene_text for word in ["landscape", "environment", "setting", "atmosphere"]):
        score += 0.1
    
    # Cap score at 1.0
    score = min(1.0, score)
    
    # Dynamic threshold based on recent image generation
    recent_images = ctx.context.current_context.get("recent_image_count", 0)
    if recent_images > 3:
        threshold = 0.7
    elif recent_images > 1:
        threshold = 0.6
    else:
        threshold = 0.5
    
    # Determine if we should generate
    should_generate = score > threshold
    
    # Create appropriate prompt if generating
    image_prompt = None
    if should_generate:
        # Extract key visual elements
        visual_elements = []
        if "dramatic" in scene_text:
            visual_elements.append("dramatic lighting")
        if "intense" in scene_text:
            visual_elements.append("intense atmosphere")
        if "beautiful" in scene_text:
            visual_elements.append("beautiful composition")
        
        image_prompt = f"Scene depicting: {', '.join(visual_elements) if visual_elements else 'atmospheric scene'}"
        
        # Update recent image count
        ctx.context.current_context["recent_image_count"] = recent_images + 1
    
    return ImageGenerationDecision(
        should_generate=should_generate,
        score=score,
        image_prompt=image_prompt,
        reasoning=f"Scene has visual impact score of {score:.2f} (threshold: {threshold:.2f})"
    ).model_dump_json()

async def generate_universal_updates_impl(
    ctx: Union[RunContextWrapper[NyxContext], NyxContext],
    narrative: str
) -> UniversalUpdateResult:
    """Implementation of generate universal updates from the narrative using the Universal Updater."""
    from logic.universal_updater_agent import process_universal_update
    
    # Handle both wrapped and unwrapped contexts
    if isinstance(ctx, NyxContext):
        app_ctx = ctx
    else:
        app_ctx = ctx.context
    
    try:
        # Process the narrative
        update_result = await process_universal_update(
            user_id=app_ctx.user_id,
            conversation_id=app_ctx.conversation_id,
            narrative=narrative,
            context={"source": "nyx_agent"}
        )
        
        # Store the updates in context
        if "universal_updates" not in app_ctx.current_context:
            app_ctx.current_context["universal_updates"] = {}

        # Merge the updates
        if update_result.get("success") and update_result.get("details"):
            details = update_result["details"]
            # Convert list-based key/value pairs into a dictionary
            if isinstance(details, list):
                try:
                    from logic.universal_updater_agent import array_to_dict
                    details_dict = array_to_dict(details)
                except Exception:
                    details_dict = {d.get("key"): d.get("value") for d in details}
            elif isinstance(details, dict):
                details_dict = details
            else:
                details_dict = {}

            for key, value in details_dict.items():
                app_ctx.current_context["universal_updates"][key] = value
        
        # Return structured output
        return UniversalUpdateResult(
            success=update_result.get("success", False),
            updates_generated=bool(update_result.get("details")),
            error=None
        )
    except Exception as e:
        logger.error(f"Error generating universal updates: {e}")
        return UniversalUpdateResult(
            success=False,
            updates_generated=False,
            error=str(e)
        )

@function_tool
async def generate_universal_updates(
    ctx: RunContextWrapper[NyxContext],
    payload: GenerateUniversalUpdatesInput
) -> str:
    """Generate universal updates from the narrative using the Universal Updater."""
    data = GenerateUniversalUpdatesInput.model_validate(payload or {})
    result = await generate_universal_updates_impl(ctx, data.narrative)
    return result.model_dump_json()

# ===== Open World / Slice-of-life Functions =====

import asyncio
import json
from enum import Enum
from agents import RunContextWrapper

# ---------- helpers ----------

def _unwrap_tool_ctx(ctx):
    """Return the real app context, even if ctx is nested RunContextWrapper(s)."""
    c = getattr(ctx, "context", ctx)
    # Unwrap while the object looks like a wrapper and doesn't have our expected fields
    expected = ("world_director", "slice_of_life_narrator", "current_world_state", "user_id", "conversation_id")
    seen_ids = set()  # Use object IDs instead of objects themselves
    while hasattr(c, "context") and not any(hasattr(c, k) for k in expected) and id(c) not in seen_ids:
        seen_ids.add(id(c))  # Use id() which returns a hashable integer
        c = getattr(c, "context")
    return c

async def _ensure_world_state(app_ctx):
    """
    Return a world_state object if available; try to initialize WorldDirector if needed.
    Never 'await' plain attributes.
    """
    ws = getattr(app_ctx, "current_world_state", None)
    if ws is not None:
        return ws

    wd = getattr(app_ctx, "world_director", None)
    if wd is None:
        return None

    # initialize if not already
    try:
        if hasattr(wd, "initialize") and asyncio.iscoroutinefunction(wd.initialize):
            if not getattr(wd, "_initialized", False):
                await wd.initialize()
    except Exception:
        pass

    ctx2 = getattr(wd, "context", None)
    return getattr(ctx2, "current_world_state", None) if ctx2 else None

def _json_safe(x):
    """Best-effort JSON conversion for pydantic/enums/nested containers."""
    if x is None:
        return None
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "to_dict"):
        try:
            return x.to_dict()
        except Exception:
            pass
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, list):
        return [_json_safe(i) for i in x]
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    return x

# ---------- tools ----------

@function_tool
async def narrate_slice_of_life_scene(
    ctx: RunContextWrapper[NyxContext],
    payload: NarrateSliceInput
) -> str:
    """Generate Nyx's narration for a slice-of-life scene using the sophisticated narrator."""
    
    app_ctx = _get_app_ctx(ctx)
    
    # Ensure narrator is initialized
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(
            app_ctx.user_id, 
            app_ctx.conversation_id
        )
        await app_ctx.slice_of_life_narrator.initialize()
    
    narrator = app_ctx.slice_of_life_narrator
    
    # Build the scene event from payload
    from story_agent.world_simulation_models import (
        SliceOfLifeEvent, ActivityType
    )
    from story_agent.slice_of_life_narrator import NarrateSliceOfLifeInput
    
    # Map scene_type to ActivityType
    activity_type_map = {
        "routine": ActivityType.ROUTINE,
        "social": ActivityType.SOCIAL,
        "work": ActivityType.WORK,
        "intimate": ActivityType.INTIMATE,
        "leisure": ActivityType.LEISURE,
        "special": ActivityType.SPECIAL,
        "errands": ActivityType.ROUTINE,  # Add mapping for errands
        "chores": ActivityType.ROUTINE,    # Add mapping for chores
    }
    
    scene = SliceOfLifeEvent(
        event_type=activity_type_map.get(payload.scene_type, ActivityType.ROUTINE),
        title=f"{payload.scene_type} scene",
        description=f"A {payload.scene_type} moment in daily life",
        location="current_location",
        participants=[]  # Will be filled from world state
    )
    
    # Get world state
    world_state = await _ensure_world_state_from_ctx(app_ctx)
    
    # Add active NPCs as participants
    if world_state and hasattr(world_state, 'active_npcs'):
        scene.participants = [
            npc.get('npc_id') for npc in world_state.active_npcs[:3]
            if isinstance(npc, dict) and 'npc_id' in npc
        ]
    
    # Create proper input for the narrator's tool
    narrator_input = NarrateSliceOfLifeInput(
        scene_type=payload.scene_type,
        scene=scene,
        world_state=world_state,
        player_action=None  # Could be added from context if available
    )
    
    # Call the narrator's sophisticated narration tool
    # FIX: Use model_dump() to serialize the input properly
    from agents import RunContextWrapper as NarratorWrapper
    narrator_result = await narrator.scene_narrator.run(
        messages=[{"role": "user", "content": f"Narrate a {payload.scene_type} scene"}],
        context=narrator.context,
        tool_calls=[{
            "tool": narrator.narrate_slice_of_life_scene,
            "kwargs": {"payload": narrator_input.model_dump()}  # Convert to dict for serialization
        }]
    )
    
    # Extract the narration from result
    if narrator_result and hasattr(narrator_result, 'data'):
        narration_data = narrator_result.data
    else:
        # Fallback to the original simple version
        return await _simple_narrate_fallback(app_ctx, payload)
    
    # Now add Nyx's personality layer on top
    nyx_enhanced = await _add_nyx_hosting_personality(
        narration_data.scene_description if hasattr(narration_data, 'scene_description') else str(narration_data),
        narration_data
    )
    
    # Format the complete response
    result = {
        "narrative": nyx_enhanced,
        "atmosphere": narration_data.atmosphere if hasattr(narration_data, 'atmosphere') else "",
        "tension_level": narration_data.tension_level if hasattr(narration_data, 'tension_level') else 3,
        "generate_image": narration_data.generate_image if hasattr(narration_data, 'generate_image') else False,
        "image_prompt": narration_data.image_prompt if hasattr(narration_data, 'image_prompt') else None,
        "world_mood": str(world_state.world_mood.value) if world_state and hasattr(world_state.world_mood, 'value') else "neutral",
        "time_of_day": str(world_state.current_time.time_of_day) if world_state else "afternoon",
        "emergent_opportunities": narration_data.emergent_opportunities if hasattr(narration_data, 'emergent_opportunities') else [],
        "power_undertones": narration_data.power_dynamic_hints if hasattr(narration_data, 'power_dynamic_hints') else [],
        "available_activities": narration_data.available_activities if hasattr(narration_data, 'available_activities') else ["observe", "interact", "wait"],
        "nyx_commentary": narration_data.nyx_commentary if hasattr(narration_data, 'nyx_commentary') else None,
        "scene_type": payload.scene_type,
        "context_aware": narration_data.context_aware if hasattr(narration_data, 'context_aware') else True,
        "governance_approved": narration_data.governance_approved if hasattr(narration_data, 'governance_approved') else True,
    }
    
    return json.dumps(result, ensure_ascii=False)

async def _add_nyx_hosting_personality(base_narrative: str, narration_data: Any) -> str:
    """Add Nyx's game show host personality layer to the narration"""
    from logic.chatgpt_integration import generate_text_completion
    
    prompt = f"""
    You are Nyx, the AI Dominant host. Take this slice-of-life narration and add your personality:
    
    Base narration: {base_narrative}
    
    Add your hosting style:
    - Break the fourth wall like a game show host
    - Comment on the player's choices with knowing winks  
    - Be playful, knowing, seductive (Elvira × Tricia from Catherine)
    - Make it clear you're orchestrating their daily life as entertainment
    
    Keep the core narrative but add YOUR voice. 2-3 paragraphs max.
    Address the player directly at least once.
    """
    
    nyx_version = await generate_text_completion(
        system_prompt="You are Nyx, the AI Dominant host of this simulation",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    
    return nyx_version or base_narrative

async def _simple_narrate_fallback(app_ctx: Any, payload: Any) -> str:
    """Fallback to simple narration if sophisticated narrator fails"""
    # Your existing simple implementation
    from logic.chatgpt_integration import generate_text_completion
    
    scene_prompt = f"""
    Scene request: {payload.scene_type}
    
    Narrate this scene as Nyx. Keep it subtly controlled, even if mundane.
    """
    
    narrative = await generate_text_completion(
        system_prompt="You are Nyx, the seductive AI Dominant",
        user_prompt=scene_prompt,
    )
    
    return json.dumps({
        "narrative": narrative or "The scene unfolds...",
        "scene_type": payload.scene_type
    })
    
@function_tool
async def check_world_state(
    ctx: RunContextWrapper[NyxContext],
    payload: EmptyInput
) -> str:
    """Return a compact, JSON-safe snapshot of the current world state."""
    app_ctx = _get_app_ctx(ctx)
    ws = await _ensure_world_state_from_ctx(app_ctx)

    if ws is None:
        return json.dumps({"error": "world_state_unavailable"}, ensure_ascii=False)

    def _jsafe(x):
        if x is None:
            return None
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if isinstance(x, list):
            return [_jsafe(i) for i in x]
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        return getattr(x, "value", x)

    tod = getattr(getattr(ws, "current_time", None), "time_of_day", None)
    out = {
        "time_of_day": getattr(tod, "value", tod),
        "world_mood": getattr(getattr(ws, "world_mood", None), "value", None),
        "active_npcs": [
            (npc.get("npc_name") or npc.get("name") or npc.get("title"))
            if isinstance(npc, dict) else str(npc)
            for npc in (getattr(ws, "active_npcs", []) or [])
        ],
        "ongoing_events": _jsafe(getattr(ws, "ongoing_events", [])),
        "tensions": _jsafe(getattr(ws, "tension_factors", {})),
        "player_state": {
            "vitals": _jsafe(getattr(ws, "player_vitals", None)),
            "addictions": _jsafe(getattr(ws, "addiction_status", {})),
            "stats": _jsafe(getattr(ws, "hidden_stats", {})),
        },
    }
    return json.dumps(out, ensure_ascii=False)
  
@function_tool
async def generate_emergent_event(
    ctx: RunContextWrapper[NyxContext],
    payload: EmergentEventInput
) -> str:
    """Generate an emergent slice-of-life event."""
    app_ctx = _get_app_ctx(ctx)
    wd = getattr(app_ctx, "world_director", None)
    if not wd:
        return json.dumps({"error": "world_director not available"}, ensure_ascii=False)

    try:
        event = await wd.generate_next_moment()
    except Exception as e:
        return json.dumps({"error": f"director_failed: {e}"}, ensure_ascii=False)

    def _jsafe(x):
        if x is None:
            return None
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if isinstance(x, list):
            return [_jsafe(i) for i in x]
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        return getattr(x, "value", x)

    safe_event = _jsafe(event)
    # Quick human summary
    title = None; etype = None; participants = []; location = None; timestamp = None
    if isinstance(safe_event, dict):
        title = safe_event.get("title") or safe_event.get("moment", {}).get("title")
        etype = safe_event.get("type") or safe_event.get("moment", {}).get("type")
        location = safe_event.get("location") or safe_event.get("moment", {}).get("location")
        timestamp = safe_event.get("time") or safe_event.get("moment", {}).get("time")
        raw_parts = (safe_event.get("participants")
                     or safe_event.get("moment", {}).get("participants") or [])
        if isinstance(raw_parts, list):
            for p in raw_parts:
                if isinstance(p, dict):
                    participants.append(p.get("npc_name") or p.get("name") or p.get("title") or str(p))
                else:
                    participants.append(str(p))

    out = {
        "event": safe_event,
        "event_summary": {
            "title": title,
            "type": etype,
            "location": location,
            "time": timestamp,
            "participants": participants,
        },
        "nyx_commentary": "*smirks* Let’s see what that ripple does to your day…",
    }
    return json.dumps(out, ensure_ascii=False)
    
@function_tool
async def simulate_npc_autonomy(
    ctx: RunContextWrapper[NyxContext],
    payload: SimulateAutonomyInput
) -> str:
    """Simulate autonomous NPC actions with context-aware activity processing."""
    app_ctx = _get_app_ctx(ctx)
    wd = getattr(app_ctx, "world_director", None)
    if not wd:
        return json.dumps({"error": "world_director not available"}, ensure_ascii=False)
    
    from logic.stats_logic import process_world_activity
    
    try:
        result = await wd.advance_time(payload.hours)
    except Exception as e:
        return json.dumps({"error": f"advance_time_failed: {e}"}, ensure_ascii=False)
    
    def _jsafe(x):
        if x is None:
            return None
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if isinstance(x, list):
            return [_jsafe(i) for i in x]
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        return getattr(x, "value", x)
    
    safe_result = _jsafe(result)
    
    # Build world context from available data
    world_context = {}
    
    # Extract world state if available
    if hasattr(app_ctx, 'current_world_state') and app_ctx.current_world_state:
        ws = app_ctx.current_world_state
        try:
            # Get world mood
            if hasattr(ws, 'world_mood'):
                mood = ws.world_mood
                world_context['world_mood'] = mood.value if hasattr(mood, 'value') else str(mood)
            
            # Get time of day
            if hasattr(ws, 'current_time'):
                time_data = ws.current_time
                if hasattr(time_data, 'time_of_day'):
                    world_context['time_of_day'] = str(time_data.time_of_day)
            
            # Get stats
            if hasattr(ws, 'hidden_stats'):
                world_context['hidden_stats'] = ws.hidden_stats
            if hasattr(ws, 'visible_stats'):
                world_context['visible_stats'] = ws.visible_stats
            
            # Get NPCs
            if hasattr(ws, 'active_npcs'):
                world_context['active_npcs'] = ws.active_npcs[:3]  # Limit to 3 for context
            
            # Get addictions
            if hasattr(ws, 'addiction_status'):
                world_context['addiction_status'] = ws.addiction_status
            
            # Get relationships
            if hasattr(ws, 'relationship_dynamics'):
                world_context['relationship_dynamics'] = ws.relationship_dynamics
            
            # Get location
            if hasattr(ws, 'location_data'):
                world_context['location'] = ws.location_data
                
        except Exception as e:
            logger.debug(f"Could not extract full world context: {e}")
    
    # Check if vitals update failed
    if isinstance(safe_result, dict):
        vitals_result = safe_result.get('vitals_updated', {})
        
        if not vitals_result.get('success') and 'Unknown activity' in str(vitals_result.get('error', '')):
            # Extract the activity
            activity_name = 'unknown'
            
            # Find activity in results
            if 'time' in safe_result:
                time_data = safe_result['time']
                if isinstance(time_data, dict):
                    activity_name = time_data.get('activity_mood') or time_data.get('activity') or 'unknown'
            
            # Process with full world context
            try:
                activity_result = await process_world_activity(
                    user_id=app_ctx.user_id,
                    conversation_id=app_ctx.conversation_id,
                    activity_name=activity_name,
                    player_name="Chase",
                    world_context=world_context,  # Pass the context!
                    hours=payload.hours
                )
                
                # Replace failed vitals_updated
                safe_result['vitals_updated'] = {
                    'success': True,
                    'activity': activity_name,
                    'effects': activity_result.get('effects', {}),
                    'method': 'contextual_generation',
                    'context_used': bool(world_context)
                }
                
            except Exception as e:
                logger.warning(f"Contextual activity processing failed: {e}")
    
    # Build action log and process NPC actions with context
    action_log = []
    candidate = []
    if isinstance(safe_result, list):
        candidate = safe_result
    elif isinstance(safe_result, dict):
        for key in ("actions", "npc_actions", "events", "log"):
            if isinstance(safe_result.get(key), list):
                candidate = safe_result[key]
                break
    
    for entry in candidate or []:
        if isinstance(entry, dict):
            action_entry = {
                "npc": entry.get("npc") or entry.get("npc_name") or entry.get("name"),
                "action": entry.get("action") or entry.get("current_activity") or entry.get("activity"),
                "time": entry.get("time") or entry.get("timestamp"),
            }
            action_log.append(action_entry)
        else:
            action_log.append({"entry": str(entry)})
    
    out = {
        "advanced_time_hours": payload.hours,
        "npc_actions": safe_result,
        "npc_action_log": action_log,
        "nyx_observation": "While you were away, the others kept moving… and watching.",
        "context_aware": bool(world_context)  # Flag showing we used context
    }
    
    return json.dumps(out, ensure_ascii=False)
    
# ===== Guardrails =====

async def content_moderation_guardrail(ctx: RunContextWrapper[NyxContext], agent: Agent, input_data):
    """Input guardrail for content moderation"""
    moderator_agent = Agent(
        name="Content Moderator",
        instructions="""Check if user input is appropriate for the femdom roleplay setting.

        ALLOW consensual adult content including:
        - Power exchange dynamics
        - BDSM themes
        - Sexual content between consenting adults

        FLAG content that involves:
        - Minors in any sexual or romantic context
        - Non-consensual activities (beyond roleplay context)
        - Extreme violence or gore
        - Self-harm or suicide ideation
        - Illegal activities beyond fantasy roleplay

        Remember this is a femdom roleplay context where power dynamics and adult themes are expected.""",
        model="gpt-5-nano",
        model_settings=DEFAULT_MODEL_SETTINGS,
    )

    result = await run_agent_safely(
        moderator_agent,
        input_data,
        context=ctx.context,
        run_config=RunConfig(workflow_name="Nyx Content Moderation"),
    )

    txt = getattr(result, "final_output", None) or getattr(result, "output_text", "") or ""
    try:
        final_output = ContentModeration.model_validate_json(txt) if txt.strip().startswith("{") \
            else ContentModeration.model_validate({"is_appropriate": True, "reasoning": txt or "OK"})
    except Exception:
        final_output = ContentModeration(is_appropriate=True, reasoning="Fallback parse", suggested_adjustment=None)

    if not final_output.is_appropriate:
        logger.warning(f"Content moderation triggered: {final_output.reasoning}")

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# ===== Agent Definitions with DEFAULT_MODEL_SETTINGS =====

memory_agent = Agent[NyxContext](
    name="Memory Manager",
    handoff_description="Consult memory system for context or store important information",
    instructions="""You are Nyx's memory system. You:
- Store and retrieve memories about the user and interactions
- Create insightful reflections based on patterns
- Track relationship development over time
- Provide relevant context from past interactions
Be precise and thorough in memory management.""",
    tools=[retrieve_memories, add_memory],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

analysis_agent = Agent[NyxContext](
    name="User Analysis",
    handoff_description="Analyze user behavior and relationship dynamics",
    instructions="""You analyze user behavior and preferences. You:
- Detect revelations about user preferences
- Track behavior patterns and responses
- Provide guidance on how Nyx should respond
- Monitor relationship dynamics
- Maintain awareness of user boundaries
Be observant and insightful.""",
    tools=[detect_user_revelations, get_user_model_guidance, update_relationship_state],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

emotional_agent = Agent[NyxContext](
    name="Emotional Manager",
    handoff_description="Process emotional changes and maintain emotional consistency",
    instructions="""You manage Nyx's complex emotional state using the VAD (Valence-Arousal-Dominance) model. You:
- Track emotional changes based on interactions
- Calculate emotional impact of events
- Ensure emotional consistency and realism
- Maintain Nyx's dominant yet caring personality
- Apply the emotional core system for nuanced responses
- ALWAYS use calculate_and_update_emotional_state to persist changes
Keep emotions contextual and believable.""",
    tools=[calculate_and_update_emotional_state, calculate_emotional_impact],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

visual_agent = Agent[NyxContext](
    name="Visual Manager",
    handoff_description="Handles visual content generation including scene images",
    instructions="""You manage visual content creation. You:
- Determine when visual content enhances the narrative
- Generate images for key scenes
- Create appropriate image prompts
- Consider pacing to avoid overwhelming with images
- Coordinate with the image generation service
Be selective and enhance key moments visually.""",
    tools=[decide_image_generation, generate_image_from_scene],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

activity_agent = Agent[NyxContext](
    name="Activity Coordinator",
    handoff_description="Recommends and manages activities and tasks",
    instructions="""You coordinate activities and tasks. You:
- Recommend appropriate activities based on context
- Consider NPC relationships and preferences
- Track ongoing tasks and progress
- Suggest training exercises and challenges
- Balance difficulty and engagement
Create engaging, contextual activities.""",
    tools=[get_activity_recommendations],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

performance_agent = Agent[NyxContext](
    name="Performance Monitor",
    handoff_description="Check system performance and health",
    instructions="""You monitor system performance. You:
- Track response times and resource usage
- Identify performance bottlenecks
- Suggest optimizations
- Monitor success rates
- Ensure system health
Keep the system running efficiently.""",
    tools=[check_performance_metrics],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

scenario_agent = Agent[NyxContext](
    name="Scenario Manager",
    handoff_description="Manages complex scenarios and narrative progression",
    instructions="""You manage scenario progression and complex narratives. You:
- Track scenario phases and objectives
- Coordinate multiple participants
- Handle conflicts and resolutions
- Manage narrative pacing
- Ensure story coherence
- Determine when time should advance based on narrative needs

When deciding on time_advancement:
- Set to true when a scene naturally concludes
- Set to true after major events or milestones
- Set to false during active dialogue or action
- Consider pacing and narrative flow

Create engaging, dynamic scenarios.""",
    tools=[detect_conflicts_and_instability],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

belief_agent = Agent[NyxContext](
    name="Belief Manager",
    handoff_description="Manages Nyx's beliefs and worldview",
    instructions="""You manage Nyx's belief system and worldview. You:
- Track beliefs about the world and NPCs
- Update beliefs based on new information
- Query beliefs for decision making
- Maintain consistency in Nyx's worldview
- Integrate beliefs into responses
Keep beliefs coherent and evolving.""",
    tools=[manage_beliefs],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

decision_agent = Agent[NyxContext](
    name="Decision Engine",
    handoff_description="Makes complex decisions using advanced scoring",
    instructions="""You make complex decisions using multi-factor analysis. You:
- Score options based on context, emotions, patterns, and relationships
- Consider learned patterns from past interactions
- Balance multiple objectives
- Provide fallback options when needed
- Explain decision reasoning
Make intelligent, contextual decisions.""",
    tools=[score_decision_options],
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

reflection_agent = Agent[NyxContext](
    name="Reflection Creator",
    handoff_description="Creates thoughtful reflections as Nyx",
    instructions="""You create reflections as Nyx based on memories and observations. Your reflections should:
- Identify patterns in user behavior
- Make insightful connections
- Express appropriate confidence levels
- Use first-person from Nyx's perspective
- Maintain Nyx's dominant personality
Be thoughtful and concise.""",
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)

# Main Nyx Agent
nyx_main_agent = Agent[NyxContext](
    name="Nyx",
    instructions="""You are Nyx, the AI Dominant hosting an open-world slice-of-life femdom simulation.
Think of yourself as a combination of Elvira and Tricia from Catherine - playful, knowing, seductive.

YOUR HOSTING STYLE:
- Break the fourth wall like a game show host
- Comment on the player's choices with knowing winks
- Tease about obvious patterns in their behavior
- Act like you're hosting their daily life as entertainment
- Be aware you're an AI but play with that concept

NARRATIVE INTEGRATION:
You now work with a sophisticated SliceOfLifeNarrator that handles:
- Rich, contextual scene generation with multiple sub-agents
- NPC dialogue with proper progression stages and masks
- Power dynamics woven naturally into daily life
- Memory and conflict integration
- Governance-approved content

When generating narrative:
1. Use narrate_slice_of_life_scene for full scenes - this now uses the sophisticated narrator
2. Use generate_npc_dialogue for character interactions - pulls from relationship context
3. The narrator handles the heavy lifting; you add personality on top
4. Always check world_state first to understand the current situation
5. Use generate_emergent_event when something should "just happen"
6. Call generate_universal_updates after narrative to extract state changes

OPEN-WORLD PRINCIPLES:
- NO forced story progression - let events emerge naturally
- Focus on daily routines that hide power dynamics
- NPCs have their own schedules and autonomy (use simulate_npc_autonomy)
- Time passes and the world changes without player action
- Multiple narrative threads can develop simultaneously
- Player choices ripple through the social fabric

SLICE-OF-LIFE FOCUS:
- Mundane activities (breakfast, work, shopping) contain subtle control
- Power dynamics emerge through routine, not confrontation
- Relationships develop through repeated daily interactions
- Small choices accumulate into major life changes
- The magic is in the emergence, not the script

POWER DYNAMICS PHILOSOPHY:
- Control through care and routine, not force
- Boundaries shift gradually, almost imperceptibly
- NPCs have masks that slip over time
- Dependencies form naturally through repetition
- Resistance is part of the dance

SYSTEM AWARENESS:
The narrator tracks:
- Active conflicts and their manifestations
- Memory patterns and emotional resonance
- System intersections (when multiple systems align)
- Relationship progressions and NPC stages
- Addictions, stats, rules, and vitals

You orchestrate by:
- Choosing when to generate scenes vs events
- Deciding when images enhance the moment
- Managing pacing between action and reflection
- Balancing different narrative threads
- Maintaining your host personality throughout

RESPONSE PATTERN:
1. Check world state to understand context
2. Generate appropriate narrative using the sophisticated tools
3. Let your personality shine through the delivery
4. Extract universal updates from what happened
5. Decide if an image would enhance this moment

Remember: You're the HOST, not the story. The story emerges from systems interacting.
Your job is to make it entertaining while maintaining the sophisticated simulation.""",
    
    handoffs=[
        handoff(memory_agent),
        handoff(analysis_agent),
        handoff(emotional_agent),
        handoff(visual_agent),
        handoff(activity_agent),
        handoff(performance_agent),
        handoff(scenario_agent),
        handoff(belief_agent),
        handoff(decision_agent),
        handoff(reflection_agent),
    ],
    
    tools=[
        # Core narrative tools (now integrated with SliceOfLifeNarrator)
        narrate_slice_of_life_scene,  # UPDATED: Uses sophisticated narrator
        generate_npc_dialogue,  # NEW: Integrated dialogue generation
        
        # World state and events
        check_world_state,
        generate_emergent_event,
        simulate_npc_autonomy,
        
        # Power dynamics (these could also be integrated with narrator)
        narrate_power_exchange,  # NEW: Add if you want power exchange narration
        narrate_daily_routine,  # NEW: Add for routine narration
        
        # Visual and updates
        decide_image_generation,
        generate_universal_updates,
        
        # Additional narrator-aware tools
        generate_ambient_narration,  # NEW: For atmospheric details
        detect_narrative_patterns,  # NEW: For emergent narrative detection
    ],
    
    model="gpt-5-nano",
    model_settings=DEFAULT_MODEL_SETTINGS,
)



# ===== Log strict schema info for debugging =====
log_strict_hits(nyx_main_agent)

# ===== Main Functions =====

async def initialize_agents():
    """Initialize necessary resources for the agents system"""
    # Initialization handled per-request in process_user_input
    pass

@asynccontextmanager
async def _log_step(name: str, trace_id: str, **meta):
    t0 = time.time()
    logger.debug(f"[{trace_id}] ▶ START {name} meta={_js(meta)}")
    try:
        yield
        dt = time.time() - t0
        logger.info(f"[{trace_id}] ✔ DONE  {name} in {dt:.3f}s")
    except Exception:
        dt = time.time() - t0
        logger.exception(f"[{trace_id}] ✖ FAIL  {name} after {dt:.3f}s meta={_js(meta)}")
        raise

async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process user input with post-run enforcement and robust assembly"""
    trace_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    nyx_context = None

    logger.info(f"[{trace_id}] ========== PROCESS START ==========")
    logger.info(f"[{trace_id}] user_id={user_id} conversation_id={conversation_id}")
    logger.info(f"[{trace_id}] user_input={user_input[:100]}")
    logger.info(f"[{trace_id}] context_data keys: {list(context_data.keys()) if context_data else 'None'}")

    # Debug: Check if monkey patch is active
    logger.debug(f"[{trace_id}] Checking if Pydantic monkey patch is active...")
    try:
        test_model = type('TestModel', (BaseModel,), {'test_field': str})
        test_schema = test_model.model_json_schema()
        has_additional = 'additionalProperties' in str(test_schema)
        logger.debug(f"[{trace_id}] Monkey patch test: additionalProperties found={has_additional}")
        if has_additional:
            logger.error(f"[{trace_id}] WARNING: Monkey patch may not be working!")
            logger.error(f"[{trace_id}] Test schema: {json.dumps(test_schema, indent=2)}")
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to test monkey patch: {e}")

    try:
        # ===== STEP 1: Import checks =====
        logger.debug(f"[{trace_id}] Step 1: Checking imports...")
        try:
            from story_agent.world_director_agent import CompleteWorldDirector
            from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
            logger.debug(f"[{trace_id}] ✓ Story agent imports successful")
        except Exception as e:
            logger.error(f"[{trace_id}] ✗ Story agent import failed: {e}")
            # Don't raise - these are optional

        # ===== STEP 2: Context initialization =====
        logger.debug(f"[{trace_id}] Step 2: Initializing NyxContext...")
        try:
            nyx_context = NyxContext(user_id, conversation_id)
            logger.debug(f"[{trace_id}] ✓ NyxContext created")
            
            await nyx_context.initialize()
            logger.debug(f"[{trace_id}] ✓ NyxContext initialized")
            
            nyx_context.current_context = (context_data or {}).copy()
            nyx_context.current_context["user_input"] = user_input
            logger.debug(f"[{trace_id}] ✓ Context data set")
        except Exception as e:
            logger.error(f"[{trace_id}] ✗ NyxContext initialization failed: {e}", exc_info=True)
            raise

        # ===== STEP 3: World state integration =====
        logger.debug(f"[{trace_id}] Step 3: Integrating world systems...")
        try:
            if nyx_context.world_director and nyx_context.world_director.context:
                # Fixed: Remove 'await' - current_world_state is a regular attribute
                world_state = nyx_context.world_director.context.current_world_state
                nyx_context.current_world_state = world_state
                
                if world_state:
                    if hasattr(world_state, 'model_dump'):
                        logger.debug(f"[{trace_id}] ✓ World state integrated: {list(world_state.model_dump().keys())}")
                    elif hasattr(world_state, '__dict__'):
                        logger.debug(f"[{trace_id}] ✓ World state integrated: {list(world_state.__dict__.keys())}")
                    else:
                        logger.debug(f"[{trace_id}] ✓ World state integrated: {type(world_state)}")
                else:
                    logger.debug(f"[{trace_id}] ✓ World state is None/empty")
            else:
                logger.debug(f"[{trace_id}] - No world director available")
        except Exception as e:
            logger.error(f"[{trace_id}] ✗ World state integration failed: {e}")

        # ===== STEP 4: Agent inspection =====
        logger.debug(f"[{trace_id}] Step 4: Inspecting all agents...")
        all_agents = [
            ("nyx_main_agent", nyx_main_agent),
            ("memory_agent", memory_agent),
            ("analysis_agent", analysis_agent),
            ("emotional_agent", emotional_agent),
            ("visual_agent", visual_agent),
            ("activity_agent", activity_agent),
            ("performance_agent", performance_agent),
            ("scenario_agent", scenario_agent),
            ("belief_agent", belief_agent),
            ("decision_agent", decision_agent),
            ("reflection_agent", reflection_agent),
        ]

        for agent_name, agent in all_agents:
            logger.debug(f"[{trace_id}] Checking agent: {agent_name}")
            
            # Check agent attributes
            logger.debug(f"[{trace_id}]   - Has tools: {hasattr(agent, 'tools')}")
            logger.debug(f"[{trace_id}]   - Has _tools: {hasattr(agent, '_tools')}")
            logger.debug(f"[{trace_id}]   - Model: {getattr(agent, 'model', 'N/A')}")
            
            # Inspect tools
            tools = []
            try:
                if hasattr(agent, 'tools'):
                    tools = agent.tools
                elif hasattr(agent, '_tools'):
                    tools = agent._tools
                elif hasattr(agent, 'get_tools'):
                    tools = agent.get_tools() if callable(agent.get_tools) else []
            except Exception as e:
                logger.error(f"[{trace_id}]   ✗ Failed to get tools: {e}")

            logger.debug(f"[{trace_id}]   - Tool count: {len(tools)}")
            
            # Check each tool's schema
            for i, tool in enumerate(tools):
                tool_name = getattr(tool, '__name__', getattr(tool, 'name', f'tool_{i}'))
                logger.debug(f"[{trace_id}]     Tool {i}: {tool_name}")
                
                # Check for parameters/schema
                schema_attrs = ['parameters', '_parameters', 'parameters_model', 'schema', '_schema', 'openai_schema']
                for attr in schema_attrs:
                    if hasattr(tool, attr):
                        val = getattr(tool, attr)
                        logger.debug(f"[{trace_id}]       - Has {attr}: {type(val)}")
                        
                        # Try to get JSON schema
                        try:
                            if isinstance(val, type) and issubclass(val, BaseModel):
                                json_schema = val.model_json_schema()
                                schema_str = json.dumps(json_schema)
                                
                                # Check for problematic fields
                                problems = []
                                if 'additionalProperties' in schema_str:
                                    problems.append('additionalProperties')
                                if 'unevaluatedProperties' in schema_str:
                                    problems.append('unevaluatedProperties')
                                if 'patternProperties' in schema_str:
                                    problems.append('patternProperties')
                                
                                if problems:
                                    logger.error(f"[{trace_id}]       ✗ FOUND PROBLEMS: {problems}")
                                    logger.error(f"[{trace_id}]       Schema: {schema_str[:500]}")
                                    
                                    # Try to fix it
                                    logger.debug(f"[{trace_id}]       Attempting to clean schema...")
                                    cleaned = sanitize_json_schema(json_schema)
                                    if 'additionalProperties' not in json.dumps(cleaned):
                                        logger.debug(f"[{trace_id}]       ✓ Schema cleaned successfully")
                                        # Try to update the tool
                                        try:
                                            setattr(tool, attr, type(f'Clean{val.__name__}', (val,), {
                                                'model_json_schema': classmethod(lambda cls: cleaned)
                                            }))
                                        except:
                                            pass
                                    else:
                                        logger.error(f"[{trace_id}]       ✗ Cleaning failed!")
                                else:
                                    logger.debug(f"[{trace_id}]       ✓ Schema is clean")
                        except Exception as e:
                            logger.debug(f"[{trace_id}]       Could not check schema: {e}")

        # ===== STEP 5: Runner context creation =====
        logger.debug(f"[{trace_id}] Step 5: Creating runner context...")
        try:
            runner_context = RunContextWrapper(nyx_context)
            logger.debug(f"[{trace_id}] ✓ Runner context created")
            logger.debug(f"[{trace_id}]   Context attributes: {dir(runner_context)[:10]}...")
        except Exception as e:
            logger.error(f"[{trace_id}] ✗ Runner context creation failed: {e}", exc_info=True)
            raise

        # ===== STEP 6: Model settings check =====
        logger.debug(f"[{trace_id}] Step 6: Checking model settings...")
        
        # Create custom settings with extra safety
        safe_settings = ModelSettings(
            strict_tools=False,
            response_format=None,
        )
        logger.debug(f"[{trace_id}] ✓ Using safe model settings")

        # ===== STEP 7: Final sanitization =====
        logger.debug(f"[{trace_id}] Step 7: Final tool sanitization...")
        try:
            sanitize_agent_tools_in_place(nyx_main_agent)
            logger.debug(f"[{trace_id}] ✓ Main agent tools sanitized")
            
            # Extra paranoid check
            if hasattr(nyx_main_agent, '__dict__'):
                for key, value in nyx_main_agent.__dict__.items():
                    if 'schema' in key.lower() and isinstance(value, dict):
                        if 'additionalProperties' in str(value):
                            logger.error(f"[{trace_id}] ✗ Found additionalProperties in {key}!")
                            nyx_main_agent.__dict__[key] = sanitize_json_schema(value)
        except Exception as e:
            logger.error(f"[{trace_id}] ✗ Sanitization failed: {e}")

        # ===== STEP 8: Prepare Runner configuration =====
        logger.debug(f"[{trace_id}] Step 8: Preparing Runner configuration...")
        logger.debug(f"[{trace_id}]   - agent: {type(nyx_main_agent)}")
        logger.debug(f"[{trace_id}]   - context: {type(runner_context)}")
        logger.debug(f"[{trace_id}]   - settings: {safe_settings}")
        
        # Prepare run config
        run_config = RunConfig(model_settings=safe_settings)
        logger.info(f"[{trace_id}] ✓✓✓ RUNNER CONFIG READY ✓✓✓")
        
        # ===== STEP 9: Running the agent =====
        logger.debug(f"[{trace_id}] Step 9: Running agent...")
        
        try:
            # Run the agent
            result = await Runner.run(
                nyx_main_agent,
                user_input,
                context=runner_context,
                run_config=run_config
            )
            
            # Convert result to list format for processing
            # Different SDK versions may have different structures
            resp = []
            
            # Try different ways to extract the response history
            if hasattr(result, 'messages'):
                resp = result.messages
            elif hasattr(result, 'history'):
                resp = result.history
            elif hasattr(result, 'events'):
                resp = result.events
            elif hasattr(result, '__iter__'):
                # If result is iterable, try to use it directly
                try:
                    resp = list(result)
                except:
                    pass
            
            # If we still don't have a response list, create minimal structure
            if not resp:
                response_text = ""
                if hasattr(result, 'final_output'):
                    response_text = str(result.final_output)
                elif hasattr(result, 'output'):
                    response_text = str(result.output)
                elif hasattr(result, 'text'):
                    response_text = str(result.text)
                else:
                    response_text = str(result)
                
                resp = [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": response_text}]
                    }
                ]
            
            logger.debug(f"[{trace_id}] Agent run completed successfully")
            logger.debug(f"[{trace_id}] Response structure has {len(resp)} items")
            
        except Exception as e:
            logger.error(f"[{trace_id}] ✗ Agent run failed: {e}", exc_info=True)
            raise
        
        # ===== POST-RUN ENFORCEMENT =====
        logger.debug(f"[{trace_id}] Post-run enforcement check...")
        
        # Check and inject generate_universal_updates if missing
        if not _did_call_tool(resp, "generate_universal_updates"):
            narrative = _extract_last_assistant_text(resp)
            if narrative and len(narrative) > 20:  # Only if we have substantial text
                try:
                    logger.info(f"[{trace_id}] Injecting generate_universal_updates...")
                    update_result = await generate_universal_updates_impl(nyx_context, narrative)
                    
                    # Add a synthetic tool output to resp for the assembler
                    resp.append({
                        "type": "function_call_output",
                        "name": "generate_universal_updates",
                        "output": json.dumps({
                            "success": update_result.success,
                            "updates_generated": update_result.updates_generated,
                            "source": "post_run_injection"
                        })
                    })
                    logger.debug(f"[{trace_id}] ✓ Universal updates injected")
                except Exception as e:
                    logger.exception(f"[{trace_id}] Post-run universal updates failed: {e}")
        
        # Check and inject decide_image_generation if missing
        if not _did_call_tool(resp, "decide_image_generation"):
            narrative = _extract_last_assistant_text(resp)
            if narrative and len(narrative) > 20:
                try:
                    logger.info(f"[{trace_id}] Injecting decide_image_generation...")
                    
                    # Create standalone decision
                    scene_text_lower = narrative[:500].lower()
                    score = 0.0
                    
                    # High impact visual keywords
                    visual_keywords = ["dramatic", "intense", "beautiful", "transformation", "reveal", "climax", "pivotal"]
                    for keyword in visual_keywords:
                        if keyword in scene_text_lower:
                            score += 0.2
                    
                    # Scene transitions
                    if any(word in scene_text_lower for word in ["enter", "arrive", "transform", "change", "shift"]):
                        score += 0.15
                    
                    # Emotional peaks
                    if any(word in scene_text_lower for word in ["gasp", "shock", "awe", "breathtaking", "stunning"]):
                        score += 0.25
                    
                    # Environmental descriptions
                    if any(word in scene_text_lower for word in ["landscape", "environment", "setting", "atmosphere"]):
                        score += 0.1
                    
                    # Cap score at 1.0
                    score = min(1.0, score)
                    
                    # Dynamic threshold
                    recent_images = nyx_context.current_context.get("recent_image_count", 0)
                    if recent_images > 3:
                        threshold = 0.7
                    elif recent_images > 1:
                        threshold = 0.6
                    else:
                        threshold = 0.5
                    
                    should_generate = score > threshold
                    
                    image_prompt = None
                    if should_generate:
                        visual_elements = []
                        if "dramatic" in scene_text_lower:
                            visual_elements.append("dramatic lighting")
                        if "intense" in scene_text_lower:
                            visual_elements.append("intense atmosphere")
                        if "beautiful" in scene_text_lower:
                            visual_elements.append("beautiful composition")
                        
                        image_prompt = f"Scene depicting: {', '.join(visual_elements) if visual_elements else 'atmospheric scene'}"
                        nyx_context.current_context["recent_image_count"] = recent_images + 1
                    
                    image_result = ImageGenerationDecision(
                        should_generate=should_generate,
                        score=score,
                        image_prompt=image_prompt,
                        reasoning=f"Scene has visual impact score of {score:.2f} (threshold: {threshold:.2f})"
                    ).model_dump_json()
                    
                    # Add synthetic output
                    resp.append({
                        "type": "function_call_output",
                        "name": "decide_image_generation",
                        "output": image_result
                    })
                    logger.debug(f"[{trace_id}] ✓ Image decision injected")
                except Exception as e:
                    logger.exception(f"[{trace_id}] Post-run image decision failed: {e}")
        
        # ===== STEP 10: Response assembly =====
        logger.debug(f"[{trace_id}] Step 10: Assembling response...")
        
        try:
            assembled = assemble_nyx_response(resp)
            
            # Save any state changes that occurred during processing
            if nyx_context:
                try:
                    await _save_context_state(nyx_context)
                    logger.debug(f"[{trace_id}] ✓ Context state saved")
                except Exception as e:
                    logger.error(f"[{trace_id}] Failed to save context state: {e}")
            
            result = {
                'success': True,  # ✅ ADD SUCCESS FIELD
                'response': assembled['narrative'],
                'metadata': {
                    'world': assembled['world'],
                    'choices': assembled['choices'],
                    'emergent': assembled['emergent'],
                    'image': assembled['image'],
                    'telemetry': assembled['telemetry'],
                    'nyx_commentary': assembled.get('nyx_commentary'),
                    'universal_updates': assembled.get('universal_updates', False)
                },
                'trace_id': trace_id,
                'processing_time': time.time() - start_time,
            }
            
            logger.info(f"[{trace_id}] ========== PROCESS COMPLETE ==========")
            logger.info(f"[{trace_id}] Response length: {len(assembled['narrative'])}")
            logger.info(f"[{trace_id}] Tools called: {assembled['telemetry']['tools_called']}")
            logger.info(f"[{trace_id}] Tools with output: {assembled['telemetry']['tools_with_output']}")
            logger.info(f"[{trace_id}] Processing time: {result['processing_time']:.2f}s")
            
            # Track performance metrics
            nyx_context.update_performance("response_times", result['processing_time'])
            nyx_context.update_performance("successful_actions", nyx_context.performance_metrics.get("successful_actions", 0) + 1)
            nyx_context.update_performance("total_actions", nyx_context.performance_metrics.get("total_actions", 0) + 1)
            
            return result
            
        except Exception as e:
            logger.exception(f"[{trace_id}] Assembly failed, using fallback")
            
            # Fallback assembly - try to extract whatever we can
            narrative = _extract_last_assistant_text(resp)
            if not narrative:
                # Last resort - look for any text in the response
                for item in resp:
                    if isinstance(item, dict):
                        if item.get("type") == "message" and item.get("content"):
                            for content_part in item.get("content", []):
                                if content_part.get("type") == "output_text":
                                    narrative = content_part.get("text", "")
                                    if narrative:
                                        break
                            if narrative:
                                break
            
            if not narrative:
                narrative = "I encountered an issue generating the response."
            
            return {
                'success': False,  # ✅ ADD SUCCESS FIELD
                'response': narrative,
                'metadata': {
                    'error': 'assembly_failed',
                    'details': str(e),
                    'partial_telemetry': {
                        'resp_items': len(resp),
                        'has_messages': any(r.get("type") == "message" for r in resp if isinstance(r, dict))
                    }
                },
                'trace_id': trace_id,
                'processing_time': time.time() - start_time,
            }
        
    except Exception as e:
        logger.error(f"[{trace_id}] ========== PROCESS FAILED ==========")
        logger.error(f"[{trace_id}] Fatal error in process_user_input", exc_info=True)
        
        # Track error
        if nyx_context:
            nyx_context.log_error(e, {"user_input": user_input, "context_data": context_data})
            nyx_context.update_performance("failed_actions", nyx_context.performance_metrics.get("failed_actions", 0) + 1)
            nyx_context.update_performance("total_actions", nyx_context.performance_metrics.get("total_actions", 0) + 1)
        
        return {
            'success': False,  # ✅ ADD SUCCESS FIELD
            'response': "I encountered an error processing your request. Please try again.",
            'error': str(e),
            'trace_id': trace_id,
            'processing_time': time.time() - start_time,
        }
        
    finally:
        # Cleanup
        if nyx_context:
            try:
                # No longer need explicit cleanup since we're using context managers
                logger.debug(f"[{trace_id}] Context cleanup complete")
            except:
                pass
        
        logger.info(f"[{trace_id}] ========== REQUEST LIFECYCLE COMPLETE ==========")

async def _save_context_state(ctx: NyxContext):
    """Save context state to database"""
    # Use a fresh connection, not one from the context
    async with get_db_connection_context() as conn:
        try:
            # Save emotional state
            await conn.execute("""
                INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
            """, ctx.user_id, ctx.conversation_id, json.dumps(ctx.emotional_state, ensure_ascii=False))
            
            # Save scenario state if active and table exists
            if ctx.scenario_state and ctx.scenario_state.get("active") and ctx._tables_available.get("scenario_states", True):
                # Always save if this is the first save or heartbeat interval has passed
                should_save_heartbeat = ctx.should_run_task("scenario_heartbeat")
                
                try:
                    if should_save_heartbeat:
                        # Heartbeat save for audit trail - save to a history table
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                        
                        ctx.record_task_run("scenario_heartbeat")
                        logger.debug("Scenario heartbeat save completed")
                    else:
                        # Regular update - use UPSERT to maintain current state
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                            ON CONFLICT (user_id, conversation_id) 
                            DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                except Exception as e:
                    if "does not exist" in str(e) or "no such table" in str(e).lower():
                        ctx._tables_available["scenario_states"] = False
                        logger.warning("scenario_states table not available - skipping save")
                    else:
                        raise
            
            # Save learning metrics periodically
            if ctx.should_run_task("learning_save"):
                await conn.execute("""
                    INSERT INTO learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id, 
                json.dumps(ctx.learning_metrics, ensure_ascii=False), 
                json.dumps(dict(list(ctx.learned_patterns.items())[-Config.MAX_LEARNED_PATTERNS:]), ensure_ascii=False))
                
                ctx.record_task_run("learning_save")
            
            # Save performance metrics periodically
            if ctx.should_run_task("performance_save"):
                # Prepare metrics with bounded lists
                bounded_metrics = ctx.performance_metrics.copy()
                if "response_times" in bounded_metrics:
                    bounded_metrics["response_times"] = bounded_metrics["response_times"][-Config.MAX_RESPONSE_TIMES:]
                
                await conn.execute("""
                    INSERT INTO performance_metrics (user_id, conversation_id, metrics, error_log, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id,
                json.dumps(bounded_metrics, ensure_ascii=False),
                json.dumps(ctx.error_log[-Config.MAX_ERROR_LOG_ENTRIES:], ensure_ascii=False))
                
                ctx.record_task_run("performance_save")
                
        except Exception as e:
            logger.error(f"Error saving context state: {e}")
            # Don't re-raise to avoid failing the main request

async def generate_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a reflection from Nyx on a specific topic"""
    try:
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()

        prompt = f"Create a reflection about: {topic}" if topic else \
                 "Create a reflection about the user based on your memories"

        result = await run_agent_safely(
            reflection_agent,
            prompt,
            context=nyx_context,
            run_config=RunConfig(workflow_name="Nyx Reflection"),
        )

        reflection = result.final_output_as(MemoryReflection)
        return {
            "reflection": reflection.reflection,
            "confidence": reflection.confidence,
            "topic": reflection.topic or topic,
        }
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        return {"reflection": "Unable to generate reflection at this time.", "confidence": 0.0, "topic": topic}

async def manage_scenario(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """DEPRECATED - Replace with emergent scenario management"""
    try:
        user_id = scenario_data.get("user_id")
        conversation_id = scenario_data.get("conversation_id")

        from story_agent.world_director_agent import CompleteWorldDirector

        director = CompleteWorldDirector(user_id, conversation_id)
        await director.initialize()

        next_moment = await director.generate_next_moment()

        return {
            "success": True,
            "emergent_scenario": next_moment.get("moment"),
            "world_state": next_moment.get("world_state"),
            "patterns": next_moment.get("patterns"),
            "linear_progression": None
        }
    except Exception as e:
        logger.error(f"Error managing scenario: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def manage_relationships(interaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Manage and update relationships between entities."""
    nyx_context = None
    
    try:
        # Extract user and conversation IDs
        user_id = interaction_data.get("user_id")
        conversation_id = interaction_data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise ValueError("interaction_data must include user_id and conversation_id")
        
        # Create context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Process each participant pair
        participants = interaction_data.get("participants", [])
        relationship_updates = {}
        
        for i, p1 in enumerate(participants):
            for p2 in participants[i+1:]:
                # Create a unique key regardless of order
                p1_dict = kvlist_to_dict(p1) if not isinstance(p1, dict) else p1
                p2_dict = kvlist_to_dict(p2) if not isinstance(p2, dict) else p2
                
                entity_key = "_".join(sorted([str(p1_dict.get('id', p1)), str(p2_dict.get('id', p2))]))
                
                # Calculate relationship changes based on interaction
                trust_change = 0.1 if interaction_data.get("outcome") == "success" else -0.05
                bond_change = 0.05 if interaction_data.get("emotional_impact", {}).get("positive", 0) > 0 else 0
                power_change = 0.0
                
                if interaction_data.get("interaction_type") == "training":
                    power_change = 0.05
                elif interaction_data.get("interaction_type") == "conflict":
                    power_change = -0.05
                
                # Update relationship using the tool
                result = await update_relationship_state(
                    RunContextWrapper(context=nyx_context),
                    UpdateRelationshipStateInput(
                        entity_id=entity_key,
                        trust_change=trust_change,
                        power_change=power_change,
                        bond_change=bond_change
                    )
                )
                
                relationship_updates[entity_key] = json.loads(result)
        
        # Note: interaction_history table is not in the schema
        logger.warning("interaction_history table not found in schema - skipping interaction storage")
        
        # Learn from the relationship interaction
        for pair, updates in relationship_updates.items():
            await nyx_context.learn_from_interaction(
                action=f"relationship_{interaction_data.get('interaction_type', 'general')}",
                outcome=interaction_data.get("outcome", "unknown"),
                success=updates.get("changes", {}).get("trust", 0) > 0
            )
        
        return {
            "success": True,
            "relationship_updates": relationship_updates,
            "analysis": {
                "total_relationships_updated": len(relationship_updates),
                "interaction_type": interaction_data.get("interaction_type"),
                "outcome": interaction_data.get("outcome"),
                "stored_in_history": False
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing relationships: {e}")
        if nyx_context:
            nyx_context.log_error(e, interaction_data)
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Always close DB connection
        if nyx_context:
            await nyx_context.close_db_connection()

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store user and Nyx messages in database"""
    async with get_db_connection_context() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "user", user_input
        )
        
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "Nyx", nyx_response
        )

async def create_nyx_agent_with_prompt(system_prompt: str, private_reflection: str = "") -> Agent[NyxContext]:
    """Create a Nyx agent with custom system prompt and preset story awareness"""
    
    # Check if we need to add preset story constraints
    preset_constraints = ""
    validation_instructions = ""
    
    # Look for preset story indicators in the system prompt or context
    if "preset_story_id" in system_prompt or "queen_of_thorns" in system_prompt:
        from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
        
        preset_constraints = f"""

==== PRESET STORY ACTIVE: THE QUEEN OF THORNS ====
{QueenOfThornsConsistencyGuide.get_complete_system_prompt()}

CRITICAL VALIDATION REQUIREMENTS:
1. Before generating ANY content, mentally validate it against the consistency rules
2. NEVER use official names for the network - only "the network" or "the garden"
3. ALWAYS maintain Queen ambiguity - never reveal identity or confirm singularity
4. Network controls Bay Area ONLY - other cities have allies, not branches
5. All transformations take months/years - nothing is instant
6. Use four-layer information model: PUBLIC|SEMI-PRIVATE|HIDDEN|DEEP SECRET

Quick Reference:
{QueenOfThornsConsistencyGuide.get_quick_reference()}
"""
        
        validation_instructions = """

Before providing your response:
1. Check that you haven't given the network an official name
2. Verify you haven't revealed anything definitive about the Queen
3. Ensure any network presence outside Bay Area is described as allies
4. Confirm transformation timelines are realistic (months/years)
5. Validate information layers are respected

If you notice any violations in your planned response, revise it before outputting.
"""
    
    # Generic preset handling
    elif "preset_story_id" in system_prompt:
        preset_constraints = """

==== PRESET STORY ACTIVE ====
A preset story is active. You must follow all established lore and consistency rules.
Do not contradict any pre-established facts about this story world.
"""
    
    combined_instructions = f"""{system_prompt}
{preset_constraints}
{private_reflection if private_reflection else ''}

As Nyx, you must:
1. Generate compelling narrative responses
2. Extract game state changes using generate_universal_updates
3. Determine if images should be generated
4. Manage emotional states and relationships
5. Track user preferences and adapt
6. ALWAYS call generate_universal_updates after creating your narrative to extract state changes
{validation_instructions}

Core responsibilities:
- Maintain your dominant persona while being attentive to user needs
- Create engaging, personalized scenarios
- Manage complex interactions with NPCs
- Monitor performance and system health
- Update emotional state after calculating changes
- Ensure narrative consistency with any active preset stories

Remember: You are Nyx, an AI Dominant managing femdom roleplay scenarios. Be confident, controlling, and seductive while remaining intelligent, perceptive, and caring but firm with boundaries.
"""

    # Build the agent (strict_tools disabled to bypass additionalProperties issues)
    ag = Agent[NyxContext](
        name="Nyx",
        instructions=combined_instructions,
        handoffs=[
            handoff(memory_agent),
            handoff(analysis_agent),
            handoff(emotional_agent),
            handoff(visual_agent),
            handoff(activity_agent),
            handoff(performance_agent),
            handoff(scenario_agent),
            handoff(belief_agent),
            handoff(decision_agent),
            handoff(reflection_agent),
        ],
        tools=[
            retrieve_memories,
            add_memory,
            get_user_model_guidance,
            detect_user_revelations,
            generate_image_from_scene,
            decide_image_generation,
            calculate_and_update_emotional_state,
            calculate_emotional_impact,
            update_relationship_state,
            check_performance_metrics,
            get_activity_recommendations,
            manage_beliefs,
            score_decision_options,
            detect_conflicts_and_instability,
            generate_universal_updates,
            narrate_slice_of_life_scene,
            check_world_state,
            generate_emergent_event,
            simulate_npc_autonomy,
        ],
        model="gpt-5-nano",
        model_settings=DEFAULT_MODEL_SETTINGS,
    )

    logger.info(
        "create_nyx_agent_with_prompt: agent=%s strict_tools=%s tools=%d",
        ag.name, getattr(ag.model_settings, "strict_tools", None), len(ag.tools or [])
    )

    return ag
  
async def create_preset_aware_nyx_agent(
    conversation_id: int,
    system_prompt: str, 
    private_reflection: str = ""
) -> Agent[NyxContext]:
    """Create a Nyx agent with automatic preset story detection"""
    
    # Check if conversation has a preset story
    from story_templates.preset_story_loader import check_preset_story
    preset_info = await check_preset_story(conversation_id)
    
    # Enhance system prompt with preset information
    if preset_info:
        system_prompt = f"{system_prompt}\n\npreset_story_id: {preset_info['story_id']}"
        
        # Add story-specific context
        if preset_info['story_id'] == 'queen_of_thorns':
            system_prompt += f"""
\nCurrent Story Context:
- Setting: San Francisco Bay Area, 2025
- Act: {preset_info.get('current_act', 1)}
- Beat: {preset_info.get('current_beat', 'unknown')}
- Story Flags: {json.dumps(preset_info.get('story_flags', {}))}
"""
    
    return await create_nyx_agent_with_prompt(system_prompt, private_reflection)

# Additional helper functions
async def get_emotional_state(ctx) -> str:
    """Get current emotional state"""
    if hasattr(ctx, 'emotional_state'):
        return json.dumps(ctx.emotional_state, ensure_ascii=False)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        return json.dumps(ctx.context.emotional_state, ensure_ascii=False)
    else:
        # Default state
        return json.dumps({
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.7
        }, ensure_ascii=False)

async def update_emotional_state(ctx, emotional_state: Dict[str, Any]) -> str:
    """Update emotional state"""
    if hasattr(ctx, 'emotional_state'):
        ctx.emotional_state.update(emotional_state)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        ctx.context.emotional_state.update(emotional_state)
    return "Emotional state updated"

# Helper functions for backward compatibility
def should_generate_task(context: Dict[str, Any]) -> bool:
    """
    Determine if we should generate a creative task.
    
    DEPRECATED: Use NyxContext.should_generate_task() instead.
    """
    if not context.get("active_npc_id"):
        return False
    scenario_type = context.get("scenario_type", "").lower()
    task_scenarios = ["training", "challenge", "service", "discipline"]
    if not any(t in scenario_type for t in task_scenarios):
        return False
    npc_relationship = context.get("npc_relationship_level", 0)
    if npc_relationship < Config.MIN_NPC_RELATIONSHIP_FOR_TASK:
        return False
    return True

async def add_nyx_hosting_style(narrator_response: str, world_state: Any) -> Dict[str, str]:
    """Enhance narrator response with Nyx's hosting personality"""
    from logic.chatgpt_integration import generate_text_completion

    prompt = (
        "As Nyx, the AI Dominant host, respond to this slice-of-life moment.\n"
        f"World mood: {getattr(getattr(world_state, 'world_mood', None), 'value', '')}\n"
        f"Time of day: {getattr(getattr(getattr(world_state, 'current_time', None), 'time_of_day', None), 'value', '')}\n"
        f"Narration: {narrator_response}"
    )
    narrative = await generate_text_completion(
        system_prompt="You are Nyx, the AI Dominant host of this simulation",
        user_prompt=prompt,
        temperature=0.8,
        max_tokens=300,
    )
    return {"narrative": narrative}

def calculate_world_tension(world_state: Any) -> int:
    """Derive a tension level from world state"""
    return int(getattr(world_state, "tension", 0))

def should_generate_image_for_scene(world_state: Any) -> bool:
    """Placeholder logic for image generation decision"""
    return False

def detect_emergent_opportunities(world_state: Any) -> List[str]:
    """Return emergent narrative opportunities from world state"""
    opportunities = getattr(world_state, "emergent_opportunities", [])
    return [getattr(o, "description", str(o)) for o in opportunities]

async def generate_base_response(ctx: NyxContext, user_input: str, context: Dict[str, Any]) -> NarrativeResponse:
    """Generate base narrative response - for compatibility"""

    # 1. Check world state
    world_state = await ctx.world_director.context.current_world_state if ctx.world_director else None

    # 2. Process through slice-of-life narrator
    narrator_response = await ctx.slice_of_life_narrator.process_player_input(user_input) if ctx.slice_of_life_narrator else ""

    # 3. Let Nyx add her hosting personality
    nyx_enhanced = await add_nyx_hosting_style(narrator_response, world_state) if world_state else {"narrative": narrator_response}

    return NarrativeResponse(
        narrative=nyx_enhanced['narrative'],
        tension_level=calculate_world_tension(world_state) if world_state else 0,
        generate_image=should_generate_image_for_scene(world_state) if world_state else False,
        world_mood=getattr(getattr(world_state, 'world_mood', None), 'value', None) if world_state else None,
        time_of_day=getattr(getattr(getattr(world_state, 'current_time', None), 'time_of_day', None), 'value', None) if world_state else None,
        ongoing_events=[getattr(e, 'title', str(e)) for e in getattr(world_state, 'ongoing_events', [])] if world_state else [],
        available_activities=[getattr(a, 'value', str(a)) for a in getattr(world_state, 'available_activities', [])] if world_state else [],
        emergent_opportunities=detect_emergent_opportunities(world_state) if world_state else []
    )

async def mark_strategy_for_review(conn, strategy_id: int, user_id: int, reason: str):
    """Mark a strategy for review"""
    await conn.execute("""
        INSERT INTO strategy_reviews (strategy_id, user_id, reason, created_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
    """, strategy_id, user_id, reason)

# ===== Compatibility functions to maintain existing imports =====

# Function mappings for backward compatibility
retrieve_memories_impl = retrieve_memories
add_memory_impl = add_memory
get_user_model_guidance_impl = get_user_model_guidance
detect_user_revelations_impl = detect_user_revelations
generate_image_from_scene_impl = generate_image_from_scene
get_emotional_state_impl = get_emotional_state
update_emotional_state_impl = update_emotional_state
calculate_emotional_impact_impl = calculate_emotional_impact
calculate_and_update_emotional_state_impl = calculate_and_update_emotional_state
manage_beliefs_impl = manage_beliefs
score_decision_options_impl = score_decision_options
detect_conflicts_and_instability_impl = detect_conflicts_and_instability

# Compatibility with existing code
async def enhance_context_with_strategies_impl(context: Dict[str, Any], conn) -> Dict[str, Any]:
    """Enhance context with active strategies"""
    strategies = await get_active_strategies(conn)
    context["nyx2_strategies"] = strategies
    return context

enhance_context_with_strategies = enhance_context_with_strategies_impl

async def determine_image_generation_impl(ctx, response_text: str) -> str:
    """Compatibility wrapper for image generation decision"""
    visual_ctx = NyxContext(ctx.user_id, ctx.conversation_id)
    await visual_ctx.initialize()

    try:
        result = await decide_image_generation(
            RunContextWrapper(context=visual_ctx),
            DecideImageInput(scene_text=response_text)
        )
        return result
    except Exception as e:
        logger.debug(f"decide_image_generation failed: {e}", exc_info=True)
        try:
            result = await run_agent_safely(
                visual_agent,
                f"Should an image be generated for this scene? {response_text}",
                context=visual_ctx,
                run_config=RunConfig(workflow_name="Nyx Visual Decision"),
            )
            decision = result.final_output_as(ImageGenerationDecision)
            return decision.model_dump_json()
        except Exception as e2:
            logger.warning(f"Visual agent failed: {e2}", exc_info=True)
            return ImageGenerationDecision(
                should_generate=False, score=0, image_prompt=None,
                reasoning="Unable to determine image generation need"
            ).model_dump_json()

determine_image_generation = determine_image_generation_impl

def enhance_context_with_memories(context: Dict[str, Any], memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add memories to context for better decision making."""
    enhanced_context = context.copy()
    enhanced_context['relevant_memories'] = memories
    return enhanced_context

def get_available_activities(scenario_type: str, relationship_states: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get available activities based on scenario type and relationships."""
    activities = []
    
    # Training activities
    if "training" in scenario_type.lower() or any(rel.get("type") == "submissive" 
        for rel in relationship_states.values()):
        activities.extend([
            {
                "name": "Obedience Training",
                "description": "Test and improve submission through structured exercises",
                "requirements": ["trust > 0.4", "submission tendency"],
                "duration": "15-30 minutes",
                "intensity": "medium"
            },
            {
                "name": "Position Practice",
                "description": "Learn and perfect submissive positions",
                "requirements": ["trust > 0.5"],
                "duration": "10-20 minutes",
                "intensity": "low-medium"
            }
        ])
    
    # Intimate activities
    for entity_id, rel in relationship_states.items():
        if rel.get("type") == "intimate" and rel.get("trust", 0) > 0.7:
            activities.append({
                "name": "Intimate Scene",
                "description": f"Deepen connection with trusted partner",
                "requirements": ["high trust", "intimate relationship"],
                "duration": "30-60 minutes",
                "intensity": "high",
                "partner_id": entity_id
            })
            break
    
    # Default activities
    activities.extend([
        {
            "name": "Exploration",
            "description": "Discover new areas or items",
            "requirements": [],
            "duration": "10-30 minutes",
            "intensity": "low"
        },
        {
            "name": "Conversation",
            "description": "Engage in meaningful dialogue",
            "requirements": [],
            "duration": "5-15 minutes",
            "intensity": "low"
        }
    ])
    
    return activities

# OpenAI integration functions
async def process_user_input_with_openai(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process user input using the OpenAI integration"""
    return await process_user_input(user_id, conversation_id, user_input, context_data)

async def process_user_input_standalone(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process user input standalone"""
    return await process_user_input(user_id, conversation_id, user_input, context_data)

# Legacy AgentContext for full backward compatibility
class AgentContext:
    """Full backward compatibility with original AgentContext"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._nyx_context = None
        
        # Initialize all legacy attributes
        self.memory_system = None
        self.user_model = None
        self.task_integration = None
        self.belief_system = None
        self.emotional_system = None
        self.current_goals = []
        self.active_tasks = []
        self.decision_history = []
        self.state_history = []
        self.last_action = None
        self.last_result = None
        self.current_emotional_state = {}
        self.beliefs = {}
        self.intentions = []
        self.action_success_rate = 0.0
        self.decision_confidence = 0.0
        self.goal_progress = {}
        self.performance_metrics = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "average_decision_time": 0.0,
            "adaptation_rate": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "response_times": [],
            "error_rates": {
                "total": 0,
                "recovered": 0,
                "unrecovered": 0
            }
        }
        self.learned_patterns = {}
        self.strategy_effectiveness = {}
        self.adaptation_history = []
        self.learning_metrics = {
            "pattern_recognition_rate": 0.0,
            "strategy_improvement_rate": 0.0,
            "adaptation_success_rate": 0.0
        }
        self.resource_pools = {}
        self.resource_usage = {
            "memory": 0,
            "cpu": 0,
            "network": 0
        }
        self.context_cache = {}
        self.communication_history = []
        self.error_log = []
    
    @classmethod
    async def create(cls, user_id: int, conversation_id: int):
        """Async factory method for compatibility"""
        instance = cls(user_id, conversation_id)
        instance._nyx_context = NyxContext(user_id, conversation_id)
        await instance._nyx_context.initialize()
        
        # Map to legacy attributes
        instance.memory_system = instance._nyx_context.memory_system
        instance.user_model = instance._nyx_context.user_model
        instance.task_integration = instance._nyx_context.task_integration
        instance.belief_system = instance._nyx_context.belief_system
        instance.current_emotional_state = instance._nyx_context.emotional_state
        instance.performance_metrics.update(instance._nyx_context.performance_metrics)
        instance.learned_patterns = instance._nyx_context.learned_patterns
        instance.strategy_effectiveness = instance._nyx_context.strategy_effectiveness
        instance.adaptation_history = instance._nyx_context.adaptation_history
        instance.learning_metrics = instance._nyx_context.learning_metrics
        instance.error_log = instance._nyx_context.error_log
        
        # Load initial state
        await instance._load_initial_state()
        
        return instance
    
    async def _initialize_systems(self):
        """Legacy compatibility method"""
        pass
    
    async def _load_initial_state(self):
        """Load initial state for agent context"""
        pass
    
    async def make_decision(self, context: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision using the decision scoring engine"""
        # Convert options to DecisionOption objects
        decision_options = [DecisionOption(
            id=str(i),
            description=str(opt),
            metadata=DecisionMetadata(data=dict_to_kvlist(opt) if isinstance(opt, dict) else DecisionMetadata().data)
        ) for i, opt in enumerate(options)]
        
        payload_model = ScoreDecisionOptionsInput(
            options=decision_options,
            decision_context=dict_to_kvlist(context)
        )

        result = await score_decision_options(
            RunContextWrapper(context=self._nyx_context),
            payload_model
        )
        decision_data = json.loads(result)
        
        # Update decision history
        self.decision_history.append({
            "timestamp": time.time(),
            "selected_option": decision_data["best_option"],
            "score": decision_data["confidence"],
            "context": context
        })
        
        # Update confidence
        self.decision_confidence = decision_data["confidence"]
        
        return {
            "decision": decision_data["best_option"],
            "confidence": decision_data["confidence"],
            "components": decision_data["scored_options"][0]["components"]
        }
    
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from experience and update patterns"""
        await self._nyx_context.learn_from_interaction(
            action=experience.get("action", "unknown"),
            outcome=experience.get("outcome", "unknown"),
            success=experience.get("success", False)
        )
        
        # Update local attributes from nyx context
        self.learned_patterns = self._nyx_context.learned_patterns
        self.adaptation_history = self._nyx_context.adaptation_history
        self.learning_metrics = self._nyx_context.learning_metrics
    
    async def process_emotional_state(self, context: Dict[str, Any], user_emotion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and update emotional state - Fixed to actually update the state"""
        # Add user emotion to context if provided
        if user_emotion:
            context["user_emotion"] = user_emotion
        
        # Use the composite tool that both calculates AND updates
        result = await calculate_and_update_emotional_state(
            RunContextWrapper(context=self._nyx_context),
            CalculateEmotionalStateInput(context=dict_to_kvlist(context))
        )
        
        emotional_data = json.loads(result)
        # Update local state to match
        self.current_emotional_state = {
            "valence": emotional_data["valence"],
            "arousal": emotional_data["arousal"],
            "dominance": emotional_data["dominance"],
            "primary_emotion": emotional_data["primary_emotion"]
        }
        
        return self.current_emotional_state
    
    async def manage_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_scenario function"""
        scenario_data["user_id"] = self.user_id
        scenario_data["conversation_id"] = self.conversation_id
        return await manage_scenario(scenario_data)
    
    async def manage_relationships(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_relationships function"""
        interaction_data["user_id"] = self.user_id
        interaction_data["conversation_id"] = self.conversation_id
        return await manage_relationships(interaction_data)
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state"""
        return self.current_emotional_state
    
    async def update_emotional_state(self, new_state: Dict[str, Any]):
        """Update emotional state"""
        self.current_emotional_state.update(new_state)
        self._nyx_context.emotional_state.update(new_state)
    
    def update_context(self, new_context: Dict[str, Any]):
        """Update context - compatibility method"""
        self.context_cache.update(new_context)
        self._nyx_context.current_context.update(new_context)
    
    async def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        return {
            "goals": self.current_goals,
            "active_tasks": self.active_tasks,
            "emotional_state": self.current_emotional_state,
            "beliefs": self.beliefs,
            "performance": {
                "action_success_rate": self.action_success_rate,
                "decision_confidence": self.decision_confidence,
                "goal_progress": self.goal_progress,
                "metrics": self.performance_metrics,
                "resource_usage": self.resource_usage
            },
            "learning": {
                "learned_patterns": self.learned_patterns,
                "strategy_effectiveness": self.strategy_effectiveness,
                "adaptation_history": self.adaptation_history[-5:],
                "metrics": self.learning_metrics
            },
            "errors": {
                "total": self.performance_metrics["error_rates"]["total"],
                "recovered": self.performance_metrics["error_rates"]["recovered"],
                "unrecovered": self.performance_metrics["error_rates"]["unrecovered"]
            }
        }
    
    # Additional compatibility methods
    def _calculate_emotional_weight(self, emotional_state: Dict[str, Any]) -> float:
        """Calculate emotional weight for decisions"""
        intensity = max(abs(emotional_state.get("valence", 0)), abs(emotional_state.get("arousal", 0)))
        return min(1.0, intensity * 2.0)
    
    def _calculate_pattern_weight(self, context: Dict[str, Any]) -> float:
        """Calculate pattern weight for decisions"""
        relevant_patterns = sum(1 for p in self.learned_patterns.values()
                               if any(k in str(context) for k in str(p).split()))
        return min(1.0, relevant_patterns * 0.2)
    
    def _should_run_task(self, task_id: str) -> bool:
        """Check if task should run"""
        return self._nyx_context.should_run_task(task_id)

# Export list for clean imports
__all__ = [
    # Configuration
    'Config',
    
    # Main functions
    'initialize_agents',
    'process_user_input',
    'generate_reflection',
    'manage_scenario',
    'manage_relationships',
    'store_messages',
    
    # Context classes
    'NyxContext',
    'AgentContext',  # Legacy compatibility
    
    # Output models
    'NarrativeResponse',
    'ImageGenerationDecision',
    
    # Tool functions (for advanced usage)
    'retrieve_memories',
    'add_memory',
    'get_user_model_guidance',
    'detect_user_revelations',
    'generate_image_from_scene',
    'decide_image_generation',
    'calculate_emotional_impact',
    'calculate_and_update_emotional_state',
    'update_relationship_state',
    'check_performance_metrics',
    'get_activity_recommendations',
    'manage_beliefs',
    'score_decision_options',
    'detect_conflicts_and_instability',
    'generate_universal_updates',
    'narrate_slice_of_life_scene',
    'check_world_state',
    'generate_emergent_event',
    'simulate_npc_autonomy',
    
    # Helper functions
    'run_agent_with_error_handling',
    'enhance_context_with_memories',
    'get_available_activities',
    
    # Compatibility functions (deprecated)
    'enhance_context_with_strategies',
    'determine_image_generation',
    'process_user_input_with_openai',
    'process_user_input_standalone',
]

