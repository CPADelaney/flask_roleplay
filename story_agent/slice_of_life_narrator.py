# story_agent/slice_of_life_narrator.py
"""
Enhanced Slice-of-Life Narrator with Context-Aware Agents
REFACTORED: Fixed Pydantic v2 additionalProperties issue for OpenAI Agents SDK
"""
from __future__ import annotations

import logging
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict, ValidationError

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool, RunContextWrapper
from agents.model_settings import ModelSettings

from logic.conflict_system.conflict_synthesizer import get_synthesizer, ConflictSynthesizer

# Database
from db.connection import get_db_connection_context

from lore.core.canon import (
    find_or_create_npc,
    find_or_create_location,
    find_or_create_event,
    log_canonical_event,
    find_or_create_notable_figure,
    ensure_canonical_context
)

import json

def _parse_tool_json(model_cls, data):
    if isinstance(data, model_cls):
        return data
    if isinstance(data, str):
        try:
            return model_cls.model_validate_json(data)
        except ValidationError:
            return model_cls.model_validate(json.loads(data))
    if isinstance(data, dict):
        return model_cls.model_validate(data)
    return None

def _ws_brief(ws: Optional[WorldState]) -> Dict[str, Any]:
    """Small, safe summary of world_state for logs."""
    try:
        if ws is None:
            return {"has_ws": False}
        mood = getattr(ws, "world_mood", None)
        tod  = getattr(ws, "time_of_day", None)
        tens = getattr(ws, "world_tension", None)
        return {
            "has_ws": True,
            "mood": getattr(mood, "value", str(mood) if mood is not None else None),
            "time_of_day": getattr(tod, "value", str(tod) if tod is not None else None),
            "power_tension": getattr(tens, "power_tension", None) if tens is not None else None,
            "social_tension": getattr(tens, "social_tension", None) if tens is not None else None,
            "sexual_tension": getattr(tens, "sexual_tension", None) if tens is not None else None,
            "active_npcs_len": len(getattr(ws, "active_npcs", []) or []),
        }
    except Exception as e:
        return {"has_ws": ws is not None, "summary_error": str(e)}

def _apply_world_state_influence(
    tone: str,
    requires_response: bool,
    world_state: Optional[WorldState],
) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Apply small, auditable adjustments based on world_state.
    Returns (tone, requires_response, effects_dict) so we can log exactly what changed.
    """
    effects: Dict[str, Any] = {}
    if world_state is None:
        return tone, requires_response, effects

    try:
        mood = getattr(world_state, "world_mood", None)
        mood_val = getattr(mood, "value", None)
        tension = getattr(world_state, "world_tension", None)
        power_t = getattr(tension, "power_tension", 0.0) if tension is not None else 0.0

        # Example, visible adjustments:
        # - High power tension nudges the NPC to demand a response.
        if isinstance(power_t, (int, float)) and power_t >= 0.7 and not requires_response:
            requires_response = True
            effects["requires_response"] = "set_true_by_high_power_tension"

        # - Mood can soften or harden tone a notch
        if isinstance(mood_val, str):
            m = mood_val.lower()
            if m == "playful" and tone in {"confident", "commanding"}:
                effects["tone_adjustment"] = f"{tone}->friendly (playful mood)"
                tone = "friendly"
            elif m == "oppressive" and tone in {"friendly"}:
                effects["tone_adjustment"] = f"{tone}->confident (oppressive mood)"
                tone = "confident"
    except Exception as e:
        effects["ws_influence_error"] = str(e)

    return tone, requires_response, effects

def _unwrap_run_ctx(x, max_hops: int = 4):
    """
    Follow `.context` links until we find an object with user_id & conversation_id,
    or we run out of hops.
    """
    base = x
    for _ in range(max_hops):
        if hasattr(base, "user_id") and hasattr(base, "conversation_id"):
            return base
        nxt = getattr(base, "context", None)
        if nxt is None or nxt is base:
            break
        base = nxt
    return base

async def _try_refresh_context(reason: str, *candidates):
    """
    Try to call refresh_context on the first candidate that has it.
    Returns True if successful, False otherwise.
    """
    for c in candidates:
        if c is None:
            continue
        fn = getattr(c, "refresh_context", None)
        if callable(fn):
            try:
                await fn(reason)
                return True
            except Exception as e:
                logger.debug(f"refresh_context failed on {type(c).__name__}: {e}")
    return False

async def _get_ws_prefer_bundle(narr_ctx: "NarratorContext") -> Optional["WorldState"]:
    """
    Prefer a cached/bundled world state on the narrator side:
      1) use narr_ctx.current_world_state if present
      2) try WorldDirector.get_world_bundle(fast=True) and extract "world_state"
      3) fallback to WorldDirector.get_world_state()
    """
    try:
        # 1) use what's already on the context (often set by refresh_context)
        if getattr(narr_ctx, "current_world_state", None) is not None:
            return narr_ctx.current_world_state

        wd = getattr(narr_ctx, "world_director", None)
        if wd is None:
            return None

        # 2) bundle fast path
        get_bundle = getattr(wd, "get_world_bundle", None)
        if callable(get_bundle):
            try:
                bundle = await get_bundle(fast=True)
                if isinstance(bundle, dict) and bundle.get("world_state") is not None:
                    ws = bundle["world_state"]
                    # Coerce dict → WorldState if needed
                    if isinstance(ws, dict):
                        try:
                            ws = WorldState(**ws)
                        except Exception:
                            pass
                    narr_ctx.current_world_state = ws
                    return ws
            except Exception as be:
                logger.debug(f"slice_of_life bundle fast path failed: {be}", exc_info=True)

        # 3) slow path
        ws = await wd.get_world_state()
        narr_ctx.current_world_state = ws
        return ws

    except Exception as e:
        logger.warning(f"_get_ws_prefer_bundle failed: {e}", exc_info=True)
        return None

# ===============================================================================
# CRITICAL FIX: Agent-Safe Base Model
# ===============================================================================

class AgentSafeModel(BaseModel):
    """
    Pydantic v2 base model that emits a schema with no additionalProperties anywhere.
    This is required for compatibility with OpenAI Agents SDK's strict mode.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    
    @classmethod
    def model_json_schema(cls, **kwargs):
        """Override to strip additionalProperties from the schema"""
        schema = super().model_json_schema(**kwargs)
        
        def strip_additional_properties(obj):
            if isinstance(obj, dict):
                # Remove the problematic keys
                obj.pop('additionalProperties', None)
                obj.pop('unevaluatedProperties', None)
                
                # Fix 'required' field to match properties
                props = obj.get("properties")
                req = obj.get("required")
                if isinstance(props, dict) and isinstance(req, list):
                    # Only keep required fields that actually exist in properties
                    obj["required"] = [k for k in req if k in props]
                
                # Recursively process all values
                for v in obj.values():
                    strip_additional_properties(v)
            elif isinstance(obj, list):
                # Process list items
                for item in obj:
                    strip_additional_properties(item)
            return obj
        
        return strip_additional_properties(schema)

# ===============================================================================
# Key-Value Helper for Replacing Dict[str, Any]
# ===============================================================================

class KeyValue(AgentSafeModel):
    """Key-value pair to replace Dict[str, Any] fields"""
    key: str
    value: Union[str, int, float, bool, None, List[Union[str, int, float, bool, None]]]

# ===============================================================================
# Narrative Tone and Style Enums
# ===============================================================================

class NarrativeTone(Enum):
    """Tone for slice-of-life narration"""
    CASUAL = "casual"
    INTIMATE = "intimate"
    OBSERVATIONAL = "observational"
    SENSUAL = "sensual"
    TEASING = "teasing"
    COMMANDING = "commanding"
    SUBTLE = "subtle"
    PSYCHOLOGICAL = "psychological"

class SceneFocus(Enum):
    """What to emphasize in scene narration"""
    ATMOSPHERE = "atmosphere"
    DIALOGUE = "dialogue"
    INTERNAL = "internal"
    DYNAMICS = "dynamics"
    ROUTINE = "routine"
    TENSION = "tension"

# ===============================================================================
# Import World Director Models (they should also use AgentSafeModel)
# ===============================================================================

from story_agent.world_simulation_models import (
    WorldState, WorldMood, TimeOfDay, ActivityType, PowerDynamicType
)

# ===============================================================================
# Tool Input Models - ALL using AgentSafeModel
# ===============================================================================

class SliceOfLifeEvent(AgentSafeModel):
    """A slice-of-life event in the simulation"""
    id: Optional[str] = None
    event_type: ActivityType
    title: str
    description: str
    location: str = "unknown"
    participants: List[int] = Field(default_factory=list)  # NPC IDs
    involved_npcs: List[int] = Field(default_factory=list)  # Alias for compatibility
    duration_minutes: int = 30
    priority: float = Field(0.5, ge=0.0, le=1.0)
    power_dynamic: Optional[PowerDynamicType] = None
    choices: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[str, Any]
    mood_impact: Optional[str] = None
    stat_impacts: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[str, float]
    addiction_triggers: List[str] = Field(default_factory=list)
    memory_tags: List[str] = Field(default_factory=list)

class NarrateSliceOfLifeInput(AgentSafeModel):
    """Input for narrating a slice-of-life scene"""
    scene_type: str = Field("routine", description="Type of scene to narrate")
    scene: Optional[SliceOfLifeEvent] = None
    world_state: Optional[WorldState] = None
    player_action: Optional[str] = None

class PowerExchange(AgentSafeModel):
    """A power exchange moment between entities"""
    initiator_npc_id: int  # For compatibility
    initiator_type: str = "npc"  # "npc" or "player"
    initiator_id: int
    recipient_type: str = "player"  # "npc" or "player"
    recipient_id: int = 1  # Default to player
    exchange_type: PowerDynamicType
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    description: str = ""
    is_public: bool = False
    witnesses: List[int] = Field(default_factory=list)
    resistance_possible: bool = True
    player_response_options: List[str] = Field(default_factory=list)
    consequences: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[str, Any]

class WorldTension(AgentSafeModel):
    """Current tension levels in the world"""
    overall_tension: float = Field(0.0, ge=0.0, le=1.0)
    social_tension: float = Field(0.0, ge=0.0, le=1.0)
    power_tension: float = Field(0.0, ge=0.0, le=1.0)
    sexual_tension: float = Field(0.0, ge=0.0, le=1.0)
    emotional_tension: float = Field(0.0, ge=0.0, le=1.0)
    addiction_tension: float = Field(0.0, ge=0.0, le=1.0)

class RelationshipDynamics(AgentSafeModel):
    """Dynamics between entities"""
    player_submission_level: float = Field(0.0, ge=0.0, le=1.0)
    active_dominants: List[int] = Field(default_factory=list)
    power_balance: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[int, float]
    tension_sources: List[str] = Field(default_factory=list)

class NPCRoutine(AgentSafeModel):
    """NPC's current routine"""
    npc_id: int
    npc_name: str
    current_activity: str
    location: str
    mood: str = "neutral"
    availability: str = "available"
    power_tendency: Optional[str] = None
    scheduled_events: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[str, str]

# ===============================================================================
# Tool Output Models - Also using AgentSafeModel for consistency
# ===============================================================================

class SliceOfLifeNarration(AgentSafeModel):
    """Complete narration for a slice-of-life scene"""
    scene_description: str
    atmosphere: str
    tone: NarrativeTone
    focus: SceneFocus
    power_dynamic_hints: List[str] = Field(default_factory=list)
    sensory_details: List[str] = Field(default_factory=list)
    npc_observations: List[str] = Field(default_factory=list)
    internal_monologue: Optional[str] = None
    governance_approved: bool = True
    governance_notes: Optional[str] = None
    emergent_elements: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[str, Any]
    system_triggers: List[str] = Field(default_factory=list)
    context_aware: bool = True
    relevant_memories: List[str] = Field(default_factory=list)

class NPCDialogue(AgentSafeModel):
    """Dialogue from an NPC in daily life"""
    npc_id: int
    npc_name: str
    dialogue: str
    tone: str
    subtext: str
    body_language: str
    power_dynamic: Optional[PowerDynamicType] = None
    requires_response: bool = False
    hidden_triggers: List[str] = Field(default_factory=list)
    memory_influence: Optional[str] = None
    governance_approved: bool = True
    context_informed: bool = False

class AmbientNarration(AgentSafeModel):
    """Ambient narration for world atmosphere"""
    description: str
    focus: str
    intensity: float = 0.5
    affects_mood: bool = False
    reflects_systems: List[str] = Field(default_factory=list)

class PowerMomentNarration(AgentSafeModel):
    """Narration for a power exchange moment"""
    setup: str
    moment: str
    aftermath: str
    player_feelings: str
    options_presentation: List[str]
    potential_consequences: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[str, Any]
    governance_tracking: List[KeyValue] = Field(default_factory=list)  # Changed from Dict[str, Any]

class DailyActivityNarration(AgentSafeModel):
    """Narration for routine daily activities"""
    activity: str
    description: str
    routine_with_dynamics: str
    npc_involvement: List[str] = Field(default_factory=list)
    subtle_control_elements: List[str] = Field(default_factory=list)
    emergent_variations: Optional[List[str]] = None

# ===============================================================================
# Import Systems (unchanged, but with safe imports)
# ===============================================================================

# Relationship systems
from logic.npc_narrative_progression import get_npc_narrative_stage, NPCNarrativeStage
from logic.dynamic_relationships import OptimizedRelationshipManager, event_generator

# Time and calendar systems
from logic.calendar import load_calendar_names
from logic.time_cycle import get_current_vitals, get_current_time_model, CurrentTimeData

# Addiction system
from logic.addiction_system_sdk import (
    get_addiction_status, 
    AddictionContext,
    check_addiction_levels_impl
)

# Narrative events
from logic.narrative_events import (
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence
)

# Event system
from logic.event_system import EventSystem

# Currency
from logic.currency_generator import CurrencyGenerator

# Relationship integration
from logic.relationship_integration import RelationshipIntegration

# Memory systems
from logic.memory_logic import (
    MemoryManager as LegacyMemoryManager,
    EnhancedMemory,
    MemoryType,
    MemorySignificance,
    ProgressiveRevealManager
)

# Stats and rules
from logic.stats_logic import (
    get_all_player_stats,
    get_player_current_tier,
    check_for_combination_triggers,
    STAT_THRESHOLDS,
    ACTIVITY_EFFECTS
)
from logic.rule_enforcement import enforce_all_rules_on_player

# Inventory
from logic.inventory_system_sdk import get_inventory

# NYX governance integration
from nyx.governance_helpers import with_governance, with_governance_permission
from nyx.integrate import get_central_governance, remember_with_governance

# Context services integration
from context.context_service import (
    get_context_service, 
    get_comprehensive_context,
    ContextService
)
from context.memory_manager import (
    get_memory_manager,
    MemoryManager,
    MemoryAddRequest,
    MemorySearchRequest
)
from context.models import MemoryMetadata  # ADD THIS
from context.vector_service import (
    get_vector_service,
    VectorService
)
from context.unified_cache import context_cache
from context.context_performance import PerformanceMonitor, track_performance

logger = logging.getLogger(__name__)

# ===============================================================================
# Narrator Context (with lazy loading)
# ===============================================================================

@dataclass
class NarratorContext:
    """Enhanced context container for slice-of-life narrator with ALL systems"""
    user_id: int
    conversation_id: int
    
    # Core systems (lazy loaded)
    _world_director: Optional[Any] = None
    _relationship_manager: Optional[Any] = None
    _event_system: Optional[Any] = None
    _currency_generator: Optional[Any] = None
    _relationship_integration: Optional[Any] = None
    _nyx_governance: Optional[Any] = None
    _memory_manager: Optional[Any] = None
    _context_service: Optional[Any] = None
    _vector_service: Optional[Any] = None
    _performance_monitor: Optional[Any] = None
    
    # NEW: Add conflict synthesizer
    _conflict_synthesizer: Optional[ConflictSynthesizer] = None
    
    # Cached data
    current_world_state: Optional[Any] = None
    active_memories: List[Any] = field(default_factory=list)
    recent_interactions: List[Any] = field(default_factory=list)
    recent_narrations: List[str] = field(default_factory=list)
    system_intersections: List[str] = field(default_factory=list)
    
    # Context data
    current_context: Optional[Dict[str, Any]] = None
    player_stats: Optional[Dict[str, Any]] = None
    active_addictions: Optional[Dict[str, Any]] = None
    current_vitals: Optional[Dict[str, Any]] = None
    
    # NEW: Conflict data
    active_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    conflict_manifestations: List[str] = field(default_factory=list)
    
    @property
    def world_director(self):
        if self._world_director is None:
            from story_agent.world_director_agent import WorldDirector
            self._world_director = WorldDirector(self.user_id, self.conversation_id)
        return self._world_director
    
    @property
    def relationship_manager(self):
        if self._relationship_manager is None:
            self._relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
        return self._relationship_manager
    
    @property
    def event_system(self):
        if self._event_system is None:
            self._event_system = EventSystem(self.user_id, self.conversation_id)
        return self._event_system

    @property
    def memory_manager(self):
        # Don't do async initialization here - just return what we have
        return self._memory_manager
    
    @property
    def nyx_governance(self):
        # Don't do async initialization here - just return what we have
        return self._nyx_governance

    @property
    def governance_active(self) -> bool:
        return self.nyx_governance is not None
    
    @property
    def conflict_synthesizer(self):
        return self._conflict_synthesizer
        
    async def initialize(self):
        """Initialize all lazy-loaded components"""
        if self._world_director is None:
            from story_agent.world_director_agent import WorldDirector
            self._world_director = WorldDirector(self.user_id, self.conversation_id)
            await self._world_director.initialize()
        
        if self._relationship_manager is None:
            self._relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
        
        if self._event_system is None:
            self._event_system = EventSystem(self.user_id, self.conversation_id)
        
        if self._memory_manager is None:
            self._memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        if self._context_service is None:
            self._context_service = await get_context_service(self.user_id, self.conversation_id)
        
        if self._nyx_governance is None:
            self._nyx_governance = await get_central_governance(self.user_id, self.conversation_id)
        
        if self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # NEW: Initialize conflict synthesizer
        if self._conflict_synthesizer is None:
            self._conflict_synthesizer = await get_synthesizer(self.user_id, self.conversation_id)
    
    async def refresh_context(self, input_text: Optional[str] = None):
        """Refresh context with latest data from all systems"""
        try:
            # Update world state
            if self._world_director:
                self.current_world_state = await self._world_director.get_world_state()
    
            # Update player stats
            try:
                self.player_stats = await get_all_player_stats(self.user_id, self.conversation_id)
            except Exception as e:
                logger.debug(f"Stats update failed during refresh: {e}")
    
            # Get player name for addiction check
            player_name = "Chase"
            if self.player_stats and isinstance(self.player_stats, dict):
                player_name = self.player_stats.get("player_name", "Chase")
    
            # Update vitals
            try:
                self.current_vitals = await get_current_vitals(self.user_id, self.conversation_id)
            except Exception as e:
                logger.debug(f"Vitals update failed during refresh: {e}")
    
            # Update addictions
            try:
                self.active_addictions = await get_addiction_status(
                    self.user_id, self.conversation_id, player_name
                )
            except Exception as e:
                logger.debug(f"Addiction status update failed during refresh: {e}")
    
            # === MEMORIES (FIX) ==========================================
            # Always use the real MemoryManager instance (NOT the tool wrapper)
            try:
                from context.memory_manager import get_memory_manager  # local import avoids cycles
                mgr = await get_memory_manager(self.user_id, self.conversation_id)
    
                # Use the instance method; it builds MemorySearchRequest with user/convo IDs
                memory_result = await mgr.search_memories(
                    query_text=(input_text or "recent"),
                    memory_types=["scene", "interaction", "npc"],
                    limit=10,
                    use_vector=True,
                )
    
                # Result is a MemorySearchResult; keep models as-is for downstream code
                self.active_memories = getattr(memory_result, "memories", []) or []
            except Exception as e:
                logger.debug(f"Memory search failed during refresh: {e}")
                self.active_memories = []
            # =============================================================
    
            # Update conflicts if synthesizer exists
            if self._conflict_synthesizer:
                try:
                    system_state = await self._conflict_synthesizer.get_system_state()
                    self.active_conflicts = []
                    conflict_states = system_state.get("conflict_states", {}) or {}
                    for conflict_id, conflict_data in conflict_states.items():
                        self.active_conflicts.append({
                            "id": conflict_id,
                            "type": conflict_data.get("type", "unknown"),
                            "participants": conflict_data.get("participants", []),
                        })
                    self.conflict_manifestations = []
                except Exception as e:
                    logger.debug(f"Conflict update failed during refresh: {e}")
    
            # Detect system intersections
            try:
                await self._detect_system_intersections()
            except Exception as e:
                logger.debug(f"Intersection detect failed: {e}")
    
            # Update current context
            if self._context_service:
                try:
                    comprehensive = await get_comprehensive_context(
                        self.user_id, self.conversation_id, input_text
                    )
                    self.current_context = comprehensive
                except Exception as e:
                    logger.debug(f"Context service update failed: {e}")
    
        except Exception as e:
            logger.warning(f"Error during context refresh: {e}")
            
    async def _detect_system_intersections(self):
        """Detect interesting system intersections"""
        self.system_intersections = []
        
        # Check addiction + rules
        if self.active_addictions and self.active_addictions.get("has_addictions"):
            active_rules = await enforce_all_rules_on_player(self.user_id, self.conversation_id)
            if active_rules:
                self.system_intersections.append("addiction_rule_synergy")
        
        # Check stats combos
        try:
            stat_combos = await check_for_combination_triggers(self.user_id, self.conversation_id)
            for combo in stat_combos:
                self.system_intersections.append(f"stat_combo:{combo['name']}")
        except:
            pass
        
        # Check vitals + rules (FIXED)
        if self.current_vitals:
            # Use attribute access for Pydantic model
            if hasattr(self.current_vitals, 'fatigue') and self.current_vitals.fatigue > 80:
                self.system_intersections.append("extreme_fatigue")
            if hasattr(self.current_vitals, 'hunger') and self.current_vitals.hunger < 20:
                self.system_intersections.append("severe_hunger")
        
        # Check memory patterns
        if self.active_memories and len(self.active_memories) > 3:
            self.system_intersections.append("memory_accumulation")
        
        # Check conflict intersections
        if self.active_conflicts:
            self.system_intersections.append(f"active_conflicts:{len(self.active_conflicts)}")
            
            # Check for specific conflict types
            for conflict in self.active_conflicts:
                if isinstance(conflict, dict):
                    conflict_type = conflict.get('type', 'unknown')
                    if conflict_type == 'social' and len(conflict.get('participants', [])) > 2:
                        self.system_intersections.append("multiparty_social_conflict")
                    elif conflict_type == 'power':
                        self.system_intersections.append("power_struggle_active")

# ===============================================================================
# Core Narration Functions with Governance & Context
# ===============================================================================

@function_tool
async def narrate_slice_of_life_scene(
    ctx: RunContextWrapper,
    payload: NarrateSliceOfLifeInput
) -> str:
    """Generate narration for a slice-of-life scene with full canonical tracking."""
    
    # --- Robust unwrapping of RunContextWrapper(s) ---
    host = _unwrap_run_ctx(ctx)  # Find the actual host context
    if not (hasattr(host, "user_id") and hasattr(host, "conversation_id")):
        # Try unwrapping ctx.context explicitly as fallback
        host = _unwrap_run_ctx(getattr(ctx, "context", ctx))
    
    if not (hasattr(host, "user_id") and hasattr(host, "conversation_id")):
        raise RuntimeError("Could not resolve host context; missing user_id/conversation_id")

    # --- Ensure narrator is attached to the *host* context, not the wrapper ---
    narrator = getattr(host, "slice_of_life_narrator", None)
    if narrator is None:
        narrator = SliceOfLifeNarrator(host.user_id, host.conversation_id)
        await narrator.initialize()
        setattr(host, "slice_of_life_narrator", narrator)

    # Work with two contexts:
    context = narrator.context          # NarratorContext (for narration internals)
    op_ctx = host                       # Operational context (has refresh_context, etc.)

    # --- SAFE REFRESH: Try narrator context first, then host context ---
    refresh_reason = payload.player_action or payload.scene_type
    await _try_refresh_context(refresh_reason, context, op_ctx)

    # --- Get attributes with fallback to operational context ---
    governance_active = getattr(context, "governance_active", 
                                getattr(op_ctx, "governance_active", False))
    nyx_governance = getattr(context, "nyx_governance", 
                            getattr(op_ctx, "nyx_governance", None))
    relationship_manager = getattr(context, "relationship_manager", 
                                  getattr(op_ctx, "relationship_manager", None))
    memory_manager = getattr(context, "memory_manager", 
                            getattr(op_ctx, "memory_manager", None))
    active_conflicts = getattr(context, "active_conflicts", 
                              getattr(op_ctx, "active_conflicts", []))
    conflict_manifestations = getattr(context, "conflict_manifestations", 
                                     getattr(op_ctx, "conflict_manifestations", []))
    system_intersections = getattr(context, "system_intersections", 
                                   getattr(op_ctx, "system_intersections", []))
    active_memories = getattr(context, "active_memories", 
                             getattr(op_ctx, "active_memories", []))
    world_director = getattr(context, "world_director", 
                            getattr(op_ctx, "world_director", None))

    # ---------- helpers (local, no imports needed) ----------
    import dataclasses as _dc

    def _to_serializable(obj):
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if _dc.is_dataclass(obj):
            return _dc.asdict(obj)
        if isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
        if hasattr(obj, "__dict__"):
            # shallow; avoid unserializable callables
            return {k: v for k, v in vars(obj).items() if not callable(v)}
        return str(obj)

    def _coerce_model(obj, cls):
        if obj is None:
            return None
        if isinstance(obj, cls):
            return obj
        if hasattr(obj, "model_dump"):
            try:
                return cls(**obj.model_dump())
            except Exception:
                return None
        if isinstance(obj, dict):
            try:
                return cls(**obj)
            except Exception:
                return None
        return None

    # ---------- robust payload normalization ----------
    if isinstance(payload, dict):
        # First pass – may raise due to nested foreign models
        try:
            payload = NarrateSliceOfLifeInput(**payload)
        except Exception:
            # Coerce nested objects that provide model_dump()
            scene_obj = payload.get("scene")
            if scene_obj is not None and hasattr(scene_obj, "model_dump"):
                payload["scene"] = scene_obj.model_dump()
            ws_obj = payload.get("world_state")
            if ws_obj is not None and hasattr(ws_obj, "model_dump"):
                payload["world_state"] = ws_obj.model_dump()
            payload = NarrateSliceOfLifeInput(**payload)

    # Coerce nested objects into our AgentSafe models
    scene = _coerce_model(payload.scene, SliceOfLifeEvent)
    world_state = payload.world_state or context.current_world_state
    if isinstance(world_state, dict):
        # WorldState is pydantic in this module
        try:
            world_state = WorldState(**world_state)
        except Exception:
            world_state = None
    if world_state is None:
        world_state = await _get_ws_prefer_bundle(context)
        if world_state is None and world_director:  # last-ditch fallback
            world_state = await world_director.get_world_state()

    # Synthesize a scene if needed
    if scene is None:
        scene = SliceOfLifeEvent(
            event_type=ActivityType.ROUTINE if payload.scene_type == "routine" else ActivityType.SOCIAL,
            title=f"{payload.scene_type} scene",
            description=f"A {payload.scene_type} moment",
            location="current_location",
            participants=[],
        )

    # Normalize participants -> List[int]
    norm_parts = []
    for p in (scene.participants or []):
        try:
            if isinstance(p, int):
                norm_parts.append(p)
            elif isinstance(p, dict) and "npc_id" in p:
                norm_parts.append(int(p["npc_id"]))
            elif hasattr(p, "npc_id"):
                norm_parts.append(int(getattr(p, "npc_id")))
            else:
                # best-effort int cast
                norm_parts.append(int(p))
        except Exception:
            # drop silently if unusable
            pass
    scene.participants = norm_parts

    # ---------- canon + context refresh ----------
    from lore.core.canon import (
        ensure_canonical_context,
        log_canonical_event,
        find_or_create_location,
        find_or_create_event,
        create_journal_entry
    )

    canonical_ctx = ensure_canonical_context({
        'user_id': context.user_id,
        'conversation_id': context.conversation_id
    })

    await context.refresh_context(payload.player_action or payload.scene_type)

    # Ensure location exists canonically
    async with get_db_connection_context() as conn:
        canonical_location = await find_or_create_location(
            canonical_ctx, conn,
            location_name=scene.location,
            description=f"The location where {scene.description} takes place",
            location_type="scene",
            notable_features=[f"{payload.scene_type}_area"]
        )
        scene.location = canonical_location

    # ---------- conflict influences ----------
    conflict_influences = []
    if context.active_conflicts:
        for conflict in context.active_conflicts[:2]:
            if isinstance(conflict, dict):
                conflict_participants = conflict.get('participants', [])
                if any(p in conflict_participants for p in scene.participants):
                    conflict_influences.append({
                        'type': conflict.get('type', 'unknown'),
                        'intensity': conflict.get('intensity', 0.5),
                        'description': conflict.get('description', '')
                    })

    # ---------- governance ----------
    governance_approved = True
    governance_notes = None
    if governance_active:
        try:
            approval = await nyx_governance.check_action_permission(
                agent_type="narrator",
                agent_id="slice_of_life_narrator",
                action_type="narrate_scene",
                action_details={
                    "scene": _to_serializable(scene),
                    "scene_type": payload.scene_type,
                    "has_conflicts": len(conflict_influences) > 0
                }
            )
            governance_approved = approval.get("approved", True)
            governance_notes = approval.get("notes")
        except Exception as e:
            logger.warning(f"Governance check failed: {e}")

    # ---------- relationship contexts ----------
    relationship_contexts = {}
    for npc_id in scene.participants:
        try:
            rel_state = await relationship_manager.get_relationship_state(
                'npc', npc_id, 'player', context.user_id
            )
            relationship_contexts[npc_id] = rel_state
        except Exception as e:
            logger.warning(f"Rel state fetch failed for npc {npc_id}: {e}")

    # ---------- narration building ----------
    scene_desc = await _generate_scene_description(context, scene, world_state, relationship_contexts)

    if context.conflict_manifestations:
        scene_desc += f" {context.conflict_manifestations[0]}"

    atmosphere = await _generate_atmosphere(context, scene, world_state)
    tone = _select_narrative_tone(scene, world_state, relationship_contexts)
    focus = _select_scene_focus(scene, payload.player_action)

    power_hints = await _generate_power_hints(context, scene, relationship_contexts)
    if conflict_influences:
        power_hints.append("Unresolved tensions simmer beneath the surface")

    sensory = await _generate_sensory_details(context, scene, world_state)

    npc_obs = []
    for npc_id in scene.participants:
        obs = await _generate_npc_observation(context, npc_id, scene, relationship_contexts.get(npc_id))
        if obs:
            npc_obs.append(obs)

    internal = None
    tension = getattr(world_state, 'world_tension', None)
    if tension is not None:
        if (getattr(tension, 'power_tension', 0) > 0.6 or getattr(tension, 'unresolved_conflicts', 0) > 0) and \
           hasattr(world_state, 'relationship_dynamics'):
            internal = await _generate_internal_monologue(context, scene, world_state.relationship_dynamics, relationship_contexts)

    emergent_elements = await _identify_emergent_elements(context, scene, relationship_contexts)

    relevant_memories = []
    if context.active_memories:
        for memory in context.active_memories[:5]:
            if hasattr(memory, 'content'):
                relevant_memories.append(memory.content[:100])
            elif isinstance(memory, dict) and 'content' in memory:
                relevant_memories.append(memory['content'][:100])

    narration = SliceOfLifeNarration(
        scene_description=scene_desc,
        atmosphere=atmosphere,
        tone=tone,
        focus=focus,
        power_dynamic_hints=power_hints,
        sensory_details=sensory,
        npc_observations=npc_obs,
        internal_monologue=internal,
        governance_approved=governance_approved,
        governance_notes=governance_notes,
        emergent_elements=emergent_elements,
        system_triggers=context.system_intersections,
        context_aware=True,
        relevant_memories=relevant_memories,
    )

    # ---------- persistence (canonical + memory) ----------
    # Compute significance
    significance = 5
    if conflict_influences:
        significance += 2
    if len(scene.participants) > 3:
        significance += 1
    if scene.event_type in [ActivityType.SPECIAL, ActivityType.INTIMATE]:
        significance += 2
    if power_hints:
        significance += 1
    significance = min(significance, 10)

    # Safe time_of_day
    tod = getattr(world_state, 'time_of_day', None)
    tod_val = getattr(tod, 'value', str(tod) if tod is not None else "Unknown")

    async with get_db_connection_context() as conn:
        try:
            await log_canonical_event(
                canonical_ctx, conn,
                f"Scene: {scene.title} - {scene_desc[:150]}...",
                tags=[
                    "scene",
                    scene.event_type.value,
                    f"location_{canonical_location}",
                    tone.value,
                    focus.value
                ] + (["conflict_present"] if conflict_influences else []),
                significance=significance
            )
        except Exception as e:
            logger.warning(f"log_canonical_event failed: {e}")

        if significance >= 7:
            try:
                _ = await find_or_create_event(
                    canonical_ctx, conn,
                    event_name=scene.title,
                    description=scene_desc,
                    location=canonical_location,
                    year=1,
                    month=1,
                    day=1,
                    time_of_day=tod_val
                )
            except Exception as e:
                logger.warning(f"find_or_create_event failed: {e}")

        journal_entry_text = scene_desc + (f"\n\n[Internal thought: {internal}]" if internal else "")
        try:
            await create_journal_entry(
                canonical_ctx, conn,
                entry_type="scene",
                entry_text=journal_entry_text,
                revelation_types="power_dynamic" if power_hints else None,
                narrative_moment=True if significance >= 8 else False,
                intensity_level=int(significance),
                entry_metadata={
                    "scene_type": scene.event_type.value,
                    "location": canonical_location,
                    "participants": scene.participants,
                    "conflicts_active": len(conflict_influences) > 0,
                    "tone": tone.value,
                    "focus": focus.value
                },
                importance=significance / 10.0,
                tags=["scene", scene.event_type.value] + (["conflict"] if conflict_influences else [])
            )
        except Exception as e:
            logger.warning(f"create_journal_entry failed: {e}")

        if scene.participants:
            for npc_id in scene.participants:
                try:
                    await log_canonical_event(
                        canonical_ctx, conn,
                        f"NPC {npc_id} participated in scene at {canonical_location}",
                        tags=[f"npc_{npc_id}", "scene_participation", canonical_location],
                        significance=3
                    )
                except Exception as e:
                    logger.warning(f"log_canonical_event (npc {npc_id}) failed: {e}")

    if memory_manager:
        try:
            await memory_manager.add_memory(
                content=scene_desc,
                memory_type="scene",
                importance=significance / 10.0,
                tags=["scene", scene.event_type.value, f"tone_{tone.value}", f"focus_{focus.value}"] + 
                     (["conflict_present"] if conflict_influences else []) +
                     [f"npc_{npc_id}" for npc_id in scene.participants[:3]],
                metadata=MemoryMetadata(
                    location=canonical_location,
                    context_type=scene.event_type.value,
                    emotion=tone.value
                ).model_dump()
            )
        except Exception as e:
            logger.warning(f"memory_manager.add_memory failed: {e}")

    return narration.model_dump()

    
@function_tool
async def generate_npc_dialogue(
    ctx,
    npc_id: int,
    situation: str,
    world_state: Optional[WorldState] = None,
    player_input: Optional[str] = None
) -> str:
    """Generate contextual NPC dialogue with power dynamics awareness."""
    import uuid
    call_id = uuid.uuid4().hex[:8]
    
    # Get the proper contexts
    host = _unwrap_run_ctx(ctx)
    if not (hasattr(host, "user_id") and hasattr(host, "conversation_id")):
        host = _unwrap_run_ctx(getattr(ctx, "context", ctx))
    
    narrator = getattr(host, "slice_of_life_narrator", None)
    if narrator is None:
        narrator = SliceOfLifeNarrator(host.user_id, host.conversation_id)
        await narrator.initialize()
        setattr(host, "slice_of_life_narrator", narrator)
    
    context = narrator.context  # NarratorContext
    op_ctx = host               # Operational context
    
    # Get attributes with fallback to operational context
    governance_active = getattr(context, "governance_active", 
                                getattr(op_ctx, "governance_active", False))
    nyx_governance = getattr(context, "nyx_governance", 
                            getattr(op_ctx, "nyx_governance", None))
    relationship_manager = getattr(context, "relationship_manager", 
                                  getattr(op_ctx, "relationship_manager", None))
    memory_manager = getattr(context, "memory_manager", 
                            getattr(op_ctx, "memory_manager", None))
    world_director = getattr(context, "world_director", 
                            getattr(op_ctx, "world_director", None))
    
    logger.info(
        "NPC_DIALOGUE[%s]: enter npc_id=%s situation=%r ws_provided=%s ws=%s",
        call_id, npc_id, situation, world_state is not None, _ws_brief(world_state)
    )

    # If callers don't pass world_state, fetch it.
    if world_state is None and world_director:
        try:
            world_state = await _get_ws_prefer_bundle(context) or await world_director.get_world_state()
            logger.info("NPC_DIALOGUE[%s]: fetched world_state via fallback ws=%s",
                        call_id, _ws_brief(world_state))
        except Exception as e:
            logger.warning("NPC_DIALOGUE[%s]: failed to fetch world_state: %s", call_id, e)
            world_state = None

    # Safe refresh context if player input provided
    if player_input:
        await _try_refresh_context(player_input, context, op_ctx)

    # Get NPC details
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow(
            """
            SELECT npc_id, npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """,
            npc_id, context.user_id, context.conversation_id
        )

    if not npc:
        logger.warning("NPC_DIALOGUE[%s]: npc_id=%s not found; returning fallback", call_id, npc_id)
        out = NPCDialogue(
            npc_id=npc_id,
            npc_name="Unknown",
            dialogue="...",
            tone="neutral",
            subtext="",
            body_language="still"
        ).model_dump()
        logger.info("NPC_DIALOGUE[%s]: exit (fallback)", call_id)
        return out

    # Narrative stage + relationship
    stage = await get_npc_narrative_stage(context.user_id, context.conversation_id, npc_id)
    
    relationship_context = None
    if relationship_manager:
        try:
            relationship_context = await relationship_manager.get_relationship_state(
                "npc", npc_id, "player", context.user_id
            )
        except Exception as e:
            logger.warning("NPC_DIALOGUE[%s]: could not get relationship state: %s", call_id, e)

    # Memory search (best-effort)
    npc_memories = []
    if memory_manager:
        try:
            memory_search_result = await memory_manager.search_memories(
                MemorySearchRequest(
                    query_text=f"npc_{npc_id} {situation}",
                    memory_types=["dialogue", "interaction", "npc"],
                    limit=5,
                    user_id=context.user_id,
                    conversation_id=context.conversation_id,
                )
            )
            npc_memories = getattr(memory_search_result, "memories", []) or []
        except Exception as e:
            logger.warning("NPC_DIALOGUE[%s]: memory search failed: %s", call_id, e)

    # Governance check using cached variables
    governance_approved = True
    if governance_active and nyx_governance:
        try:
            approval = await nyx_governance.check_permission(
                agent_type="narrator",
                action_type="npc_dialogue",
                context={"npc_id": npc_id, "stage": stage.name},
            )
            governance_approved = approval.get("approved", True)
            logger.debug("NPC_DIALOGUE[%s]: governance_approved=%s", call_id, governance_approved)
        except Exception as e:
            logger.warning("NPC_DIALOGUE[%s]: governance check failed: %s", call_id, e)

    # Dialogue building
    dialogue = await _generate_dialogue_content(
        context, npc, situation, stage, player_input, relationship_context
    )
    tone = await _generate_dialogue_tone(context, npc["dominance"], stage.name, situation)
    subtext = await _generate_dialogue_subtext(
        context, dialogue, npc["dominance"], stage.name, context.active_addictions
    )
    body_language = await _generate_body_language(context, npc["dominance"], tone, stage.name)

    # World-state influence (visible + logged)
    requires_response = (random.random() > 0.5)
    tone_before = tone
    requires_before = requires_response
    tone, requires_response, ws_effects = _apply_world_state_influence(tone, requires_response, world_state)

    if ws_effects:
        logger.info(
            "NPC_DIALOGUE[%s]: ws_influence applied tone:%s->%s requires:%s->%s effects=%s ws=%s",
            call_id, tone_before, tone, requires_before, requires_response, ws_effects, _ws_brief(world_state)
        )
    else:
        logger.debug("NPC_DIALOGUE[%s]: no ws_influence applied", call_id)

    # Power dynamic & triggers
    power_dynamic = None
    if npc["dominance"] > 60 and stage.name != "Innocent Beginning":
        power_dynamic = _select_dialogue_power_dynamic(situation, npc["dominance"])

    hidden_triggers = await _identify_dialogue_triggers(
        context, dialogue, npc_id, relationship_context
    )
    memory_influence = (
        f"Influenced by {len(npc_memories)} past interactions" if npc_memories else None
    )

    logger.debug(
        "NPC_DIALOGUE[%s]: tone=%s subtext=%s body=%s power_dynamic=%s triggers=%d mem_influence=%s",
        call_id, tone, subtext, body_language,
        getattr(power_dynamic, "value", power_dynamic),
        len(hidden_triggers),
        memory_influence or "none",
    )

    dialogue_obj = NPCDialogue(
        npc_id=npc_id,
        npc_name=npc["npc_name"],
        dialogue=dialogue,
        tone=tone,
        subtext=subtext,
        body_language=body_language,
        power_dynamic=power_dynamic,
        requires_response=requires_response,
        hidden_triggers=hidden_triggers,
        memory_influence=memory_influence,
        governance_approved=governance_approved,
        context_informed=bool(context.current_context and "npcs" in context.current_context),
    )

    # Memory storage (best-effort) using cached memory_manager
    if memory_manager:
        try:
            await memory_manager.add_memory(
                content=f"{npc['npc_name']}: {dialogue}",
                memory_type="dialogue",
                importance=0.6,
                tags=["dialogue", f"npc_{npc_id}", stage.name, f"tone_{tone}"],
                metadata=MemoryMetadata(
                    npc_id=str(npc_id),
                    emotion=tone,
                    context_type=stage.name
                ).model_dump()
            )
        except Exception as e:
            logger.warning("NPC_DIALOGUE[%s]: add_memory failed: %s", call_id, e)

    logger.info(
        "NPC_DIALOGUE[%s]: exit npc_id=%s requires_response=%s governance_approved=%s",
        call_id, npc_id, dialogue_obj.requires_response, governance_approved
    )
    return dialogue_obj.model_dump()

@function_tool
async def narrate_power_exchange(
    ctx,
    exchange: PowerExchange,
    world_state: WorldState
) -> str:
    """Generate narration for a power exchange moment with Nyx tracking, memory, and canonical consistency."""
    context = ctx.context
    governance_active = getattr(context, "governance_active", False)
    nyx_governance = getattr(context, "nyx_governance", None)
    relationship_manager = getattr(context, "relationship_manager", None)
    memory_manager = getattr(context, "memory_manager", None)
    
    # Import canon functions
    from lore.core.canon import (
        ensure_canonical_context,
        log_canonical_event,
        find_or_create_npc,
        find_or_create_location,
        find_or_create_historical_event,
        update_current_roleplay,
        create_journal_entry,
        find_or_create_social_link
    )
    
    # Create canonical context
    canonical_ctx = ensure_canonical_context({
        'user_id': context.user_id,
        'conversation_id': context.conversation_id
    })
    
    # Refresh context for power exchange
    await _try_refresh_context(f"Power exchange: {exchange.exchange_type.value}", context)
    
    # Get NPC details and ensure canonical existence
    async with get_db_connection_context() as conn:
        # First ensure NPC exists canonically
        npc = await conn.fetchrow("""
            SELECT npc_id, npc_name, dominance, personality_traits, current_location
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, exchange.initiator_npc_id, context.user_id, context.conversation_id)
        
        if not npc:
            # Create NPC canonically if doesn't exist
            canonical_npc_id = await find_or_create_npc(
                canonical_ctx, conn,
                npc_name=f"Dominant_{exchange.initiator_npc_id}",
                role="dominant",
                dominance=80,  # High dominance for power exchange initiator
                personality_traits=["commanding", "confident", "observant"]
            )
            
            # Re-fetch the created NPC
            npc = await conn.fetchrow("""
                SELECT npc_id, npc_name, dominance, personality_traits, current_location
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, canonical_npc_id, context.user_id, context.conversation_id)
        
        # Ensure location exists
        location = world_state.current_location if hasattr(world_state, 'current_location') else "private_space"
        canonical_location = await find_or_create_location(
            canonical_ctx, conn,
            location_name=location,
            description=f"The space where power dynamics shift",
            location_type="intimate" if not exchange.is_public else "public",
            cultural_significance="high" if exchange.intensity > 0.7 else "moderate"
        )
        
        # Ensure social link exists and is updated
        link_id = await find_or_create_social_link(
            canonical_ctx, conn,
            entity1_type='npc',
            entity1_id=exchange.initiator_npc_id,
            entity2_type='player',
            entity2_id=context.user_id,
            link_type='power_dynamic',
            link_level=int(exchange.intensity * 100)
        )
    
    # Track with Nyx governance if active
    governance_tracking = []
    if governance_active and nyx_governance:
        try:
            tracking_data = await nyx_governance.track_power_exchange(
                npc_id=exchange.initiator_npc_id,
                exchange_type=exchange.exchange_type.value,
                intensity=exchange.intensity,
                timestamp=datetime.now(timezone.utc)
            )
            governance_tracking = [
                KeyValue(key=k, value=str(v)) for k, v in tracking_data.items()
            ] if tracking_data else []
        except Exception as e:
            logger.warning(f"Could not track power exchange with governance: {e}")
    
    # Get relationship state
    rel_state = await relationship_manager.get_relationship_state(
        'npc', exchange.initiator_npc_id, 'player', context.user_id
    )
    
    # Generate narration components
    setup = await _generate_power_moment_setup(context, exchange, npc, world_state)
    moment = await _generate_power_moment_description(context, exchange, npc)
    aftermath = await _generate_power_moment_aftermath(context, exchange, world_state.relationship_dynamics)
    player_feelings = await _generate_player_feelings(context, exchange, world_state.relationship_dynamics)
    
    # Generate response options
    options = await _generate_power_response_options(
        context, exchange, npc['dominance'], rel_state
    )
    
    # Calculate potential consequences
    consequences = await _calculate_power_consequences(context, exchange, world_state)
    
    narration = PowerMomentNarration(
        setup=setup,
        moment=moment,
        aftermath=aftermath,
        player_feelings=player_feelings,
        options_presentation=options,
        potential_consequences=consequences,
        governance_tracking=governance_tracking
    )
    
    # Store power exchange canonically with full context
    async with get_db_connection_context() as conn:
        # Calculate significance based on multiple factors
        base_significance = 7  # Power exchanges are always significant
        if exchange.intensity > 0.8:
            base_significance += 2
        elif exchange.intensity > 0.6:
            base_significance += 1
        
        if exchange.exchange_type in [PowerDynamicType.INTIMATE_COMMAND, PowerDynamicType.PUBLIC_DISPLAY]:
            base_significance += 1
        
        significance = min(base_significance, 10)
        
        # Log the main power exchange event
        await log_canonical_event(
            canonical_ctx, conn,
            f"Power exchange: {npc['npc_name']} exercised {exchange.exchange_type.value} over player (intensity: {exchange.intensity:.2f})",
            tags=[
                "power_exchange",
                "dynamics",
                f"npc_{exchange.initiator_npc_id}",
                exchange.exchange_type.value.replace("_", "-").lower(),
                f"location_{canonical_location}",
                "public" if exchange.is_public else "private"
            ] + ([f"witness_{w}" for w in exchange.witnesses] if exchange.witnesses else []),
            significance=significance
        )
        
        # Update CurrentRoleplay to track power dynamics state
        current_submission = world_state.relationship_dynamics.player_submission_level if hasattr(world_state, 'relationship_dynamics') else 0.5
        new_submission = min(1.0, current_submission + (exchange.intensity * 0.1))
        
        await update_current_roleplay(
            canonical_ctx, conn,
            'PlayerSubmissionLevel',
            str(new_submission)
        )
        
        await update_current_roleplay(
            canonical_ctx, conn,
            'LastPowerExchange',
            f"{npc['npc_name']}:{exchange.exchange_type.value}:{exchange.intensity}"
        )
        
        # Create historical event for major power exchanges
        if significance >= 8:
            event_id = await find_or_create_historical_event(
                canonical_ctx, conn,
                event_name=f"Significant power exchange with {npc['npc_name']}",
                description=f"{setup} {moment} {aftermath}",
                date_description="Present moment",
                event_type="power_dynamic",
                significance=significance,
                involved_entities=[f"npc_{exchange.initiator_npc_id}", "player"],
                location=canonical_location,
                consequences=[f"submission_increased:{(exchange.intensity * 0.1):.2f}"],
                disputed_facts=["The exact nature of control"] if exchange.resistance_possible else []
            )
        
        # Create journal entry from player's perspective
        journal_text = f"{moment}\n\nFeeling: {player_feelings}"
        if options:
            journal_text += f"\n\nOptions considered: {', '.join(options[:2])}"
        
        await create_journal_entry(
            canonical_ctx, conn,
            entry_type="power_exchange",
            entry_text=journal_text,
            revelation_types="power_dynamic",
            narrative_moment=True,
            fantasy_flag=exchange.exchange_type == PowerDynamicType.INTIMATE_COMMAND,
            intensity_level=int(exchange.intensity * 10),
            entry_metadata={
                "npc_id": exchange.initiator_npc_id,
                "npc_name": npc['npc_name'],
                "exchange_type": exchange.exchange_type.value,
                "location": canonical_location,
                "public": exchange.is_public,
                "witnesses": exchange.witnesses,
                "resistance_possible": exchange.resistance_possible,
                "player_submission": new_submission
            },
            importance=exchange.intensity,
            tags=["power", f"npc_{exchange.initiator_npc_id}", exchange.exchange_type.value]
        )
        
        # Log witnesses if present
        if exchange.is_public and exchange.witnesses:
            for witness_id in exchange.witnesses:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"NPC {witness_id} witnessed power exchange between {npc['npc_name']} and player",
                    tags=[f"npc_{witness_id}", "witness", "power_exchange", f"location_{canonical_location}"],
                    significance=5
                )
        
        # Track cumulative power dynamics
        power_history = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id = $1 AND conversation_id = $2 AND key = 'PowerExchangeCount'
        """, context.user_id, context.conversation_id)
        
        exchange_count = int(power_history) if power_history else 0
        exchange_count += 1
        
        await update_current_roleplay(
            canonical_ctx, conn,
            'PowerExchangeCount',
            str(exchange_count)
        )
        
        # Track power exchange patterns if multiple with same NPC
        npc_exchanges = await conn.fetchval("""
            SELECT COUNT(*) FROM CanonicalEvents
            WHERE user_id = $1 AND conversation_id = $2
            AND $3 = ANY(tags) AND 'power_exchange' = ANY(tags)
        """, context.user_id, context.conversation_id, f"npc_{exchange.initiator_npc_id}")
        
        if npc_exchanges >= 3:
            await log_canonical_event(
                canonical_ctx, conn,
                f"Power dynamic pattern established with {npc['npc_name']} ({npc_exchanges} exchanges)",
                tags=["pattern", "power_dynamic", f"npc_{exchange.initiator_npc_id}", "relationship_progression"],
                significance=8
            )
    
    # Store in memory as significant event
    if memory_manager:
        await memory_manager.add_memory(
            content=f"Power exchange with {npc['npc_name']}: {moment}",
            memory_type="power_exchange",
            importance=min(1.0, 0.7 + exchange.intensity * 0.3),
            tags=["power", f"npc_{exchange.initiator_npc_id}", exchange.exchange_type.value,
                  f"intensity_{int(exchange.intensity*10)}", f"significance_{significance}"],
            metadata=MemoryMetadata(
                npc_id=str(exchange.initiator_npc_id),
                context_type=exchange.exchange_type.value,
                location=canonical_location if 'canonical_location' in locals() else None
            ).model_dump()
        )
    
    # Return JSON string
    return narration.model_dump()

@function_tool
async def narrate_daily_routine(
    ctx,
    activity: str,
    world_state: WorldState,
    involved_npcs: List[int] = None
) -> str:
    """Generate narration for daily routine activities with subtle power dynamics."""
    context = ctx.context
    await context.refresh_context(input_text=f"Daily activity: {activity}")
    
    involved_npcs = involved_npcs or []
    
    # Generate base description
    description = await _generate_routine_description(context, activity, world_state)
    
    # Add power dynamics layer
    routine_with_dynamics = await _generate_routine_with_dynamics(
        context, activity, involved_npcs, world_state
    )
    
    # Generate NPC involvement
    npc_involvement = []
    for npc_id in involved_npcs:
        involvement = await _generate_npc_routine_involvement(
            context, npc_id, activity
        )
        if involvement:
            npc_involvement.append(involvement)
    
    # Identify subtle control elements
    control_elements = await _identify_control_elements(
        context, activity, involved_npcs, world_state
    )
    
    # Generate emergent variations based on systems
    emergent_variations = None
    if context.system_intersections:
        emergent_variations = await _generate_emergent_variations(
            context, activity, context.system_intersections
        )
    
    result = DailyActivityNarration(
        activity=activity,
        description=description,
        routine_with_dynamics=routine_with_dynamics,
        npc_involvement=npc_involvement,
        subtle_control_elements=control_elements,
        emergent_variations=emergent_variations
    )
    
    # Return JSON string
    return result.model_dump()

@function_tool
async def generate_ambient_narration(ctx, focus: str, world_state: WorldState, intensity: float = 0.5) -> str:
    context = ctx.context
    await context.refresh_context()

    # Safely pull parts from world_state
    mood = getattr(world_state, "world_mood", None)
    tension = getattr(world_state, "world_tension", None)

    if focus == "time_passage":
        time_data = await get_current_time_model(context.user_id, context.conversation_id)
        description = await _narrate_time_passage(context, time_data) if time_data else "Time passes..."
    elif focus == "mood":
        description = await _narrate_mood_shift(context, mood) if mood else "The atmosphere tilts, in a way you can't quite name..."
    elif focus == "tension":
        description = await _narrate_tension(context, tension) if tension else "A subtle tension hums at the edges of things..."
    elif focus == "addiction":
        description = await _narrate_addiction_presence(context, context.active_addictions)
    elif focus == "ambient":
        description = await _narrate_ambient_detail(context, world_state) if world_state else "Life drifts on, unhurried."
    else:
        description = await _generate_ambient_description(context, focus, world_state, intensity)

    affects_mood = intensity > 0.7 or focus in ["tension", "power", "control"]
    reflects_systems = await _identify_reflected_systems(context, focus, description)

    return AmbientNarration(
        description=description,
        focus=focus,
        intensity=intensity,
        affects_mood=affects_mood,
        reflects_systems=reflects_systems
    ).model_dump()


@function_tool
async def narrate_player_action(
    ctx,
    action: str,
    world_state: WorldState,
    scene_context: Optional[SliceOfLifeEvent] = None
) -> str:
    context = ctx.context
    await context.refresh_context(input_text=action)

    scene_context = scene_context or SliceOfLifeEvent(
        event_type=ActivityType.SOCIAL,
        title="Player Action",
        description=action,
        participants=[]
    )

    # Call the tool function using a payload object
    result = await narrate_slice_of_life_scene(
        ctx,
        payload=NarrateSliceOfLifeInput(
            scene_type="action",
            scene=scene_context,
            world_state=world_state,
            player_action=action,
        )
    )
    
    # Result is already a JSON string from narrate_slice_of_life_scene
    return result
    
# ===============================================================================
# Helper Functions with GPT Integration
# ===============================================================================

# Import GPT integration
from logic.chatgpt_integration import generate_text_completion

async def _generate_scene_description(context, scene, world_state, relationship_contexts):
    npc_names = []
    for npc_id, rc in relationship_contexts.items():
        name = None
        state = getattr(rc, "state", None)
        if state is not None:
            name = getattr(state, "npc_name", None)
        if not name and isinstance(rc, dict):
            name = rc.get("state", {}).get("npc_name")
        if not name:
            name = f"NPC {npc_id}"
        npc_names.append(name)

    mood_val = getattr(getattr(world_state, "world_mood", None), "value", "neutral")

    prompt = f"""
    Generate a scene description for:
    Location: {scene.location}
    Activity: {scene.description}
    Mood: {mood_val}
    NPCs present: {', '.join(npc_names) if npc_names else 'none'}

    Make it atmospheric and immersive, 2-3 sentences.
    """

    description = await generate_text_completion(
        system_prompt="You narrate slice-of-life scenes with subtle power dynamics",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    return description.strip()

    
async def _generate_atmosphere(context, scene, world_state):
    """Generate atmospheric description"""
    mood_val = getattr(getattr(world_state, "world_mood", None), "value", "neutral")
    prompt = f"""
    Generate atmospheric description for:
    Scene: {scene.location}
    Mood: {mood_val}

    One sentence capturing the feeling of the space.
    """

    atmosphere = await generate_text_completion(
        system_prompt="You create atmospheric descriptions",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    return atmosphere.strip()


def _get_dim(rel_state, name: str, default: float = 0):
    """
    Read a relationship dimension (e.g., 'dominance', 'intimacy', 'awareness')
    from either an object with .dimensions.<name> or a dict with
    {'dimensions': {name: ...}} (or flat {name: ...}) shape.
    """
    # object path: rel_state.dimensions.<name>
    dims = getattr(rel_state, "dimensions", None)
    if dims is not None:
        val = getattr(dims, name, None)
        if val is not None:
            return val

    # dict path: rel_state["dimensions"][name] or rel_state[name]
    if isinstance(rel_state, dict):
        dims = rel_state.get("dimensions") or {}
        if isinstance(dims, dict) and name in dims:
            return dims.get(name, default)
        return rel_state.get(name, default)

    return default

def _select_narrative_tone(scene, world_state, relationship_contexts):
    tens = getattr(world_state, "world_tension", None)
    if tens and getattr(tens, "power_tension", 0) > 0.7:
        return NarrativeTone.COMMANDING
    if tens and getattr(tens, "sexual_tension", 0) > 0.6:
        return NarrativeTone.SENSUAL
    for rc in relationship_contexts.values():
        if _get_dim(rc, "intimacy", 0) > 70:
            return NarrativeTone.INTIMATE
    return NarrativeTone.OBSERVATIONAL


def _select_scene_focus(scene, player_action):
    """Select scene focus"""
    if player_action:
        return SceneFocus.DYNAMICS
    elif scene.power_dynamic:
        return SceneFocus.TENSION
    return SceneFocus.ATMOSPHERE

async def _generate_power_hints(context, scene, relationship_contexts):
    """Generate subtle power dynamic hints"""
    hints = []
    for rel_state in relationship_contexts.values():
        if _get_dim(rel_state, "dominance", 0) > 60:
            hints.append("A subtle expectation in their presence")
        if _get_dim(rel_state, "influence", 0) > 60:
            hints.append("They steer the moment without saying much")
    # de-dupe while preserving order
    seen = set()
    out = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out

async def _generate_sensory_details(context, scene, world_state):
    details = []
    time_data = await get_current_time_model(context.user_id, context.conversation_id)
    if time_data:
        tod = getattr(time_data, "time_of_day", None)
        tod_val = getattr(tod, "value", tod)  # enum.value or raw
        tod_str = str(tod_val).lower() if tod_val is not None else ""
        if "morning" in tod_str:
            details.append("The morning light filters through")
        elif "evening" in tod_str or "dusk" in tod_str:
            details.append("Evening shadows lengthen")
    return details

async def _generate_npc_observation(context, npc_id, scene, rel_context):
    """Generate NPC observation"""
    awareness = _get_dim(rel_context, "awareness", 0)
    intimacy  = _get_dim(rel_context, "intimacy", 0)
    dominance = _get_dim(rel_context, "dominance", 0)

    if awareness > 50:
        return "They seem particularly aware of your presence"
    if intimacy > 65:
        return "Their regard lingers a fraction longer than it should"
    if dominance > 60:
        return "There’s a quiet, confident expectancy in how they watch you"
    return None

async def _generate_internal_monologue(context, scene, relationship_dynamics, relationship_contexts):
    """Generate internal monologue with GPT"""
    prompt = f"""
    Generate internal monologue for player experiencing:
    Power tension: {relationship_dynamics.player_submission_level}
    Scene: {scene.description}
    
    One sentence of internal thought showing awareness/resistance/acceptance.
    """
    
    monologue = await generate_text_completion(
        system_prompt="You write internal monologues showing psychological states",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    
    return monologue.strip()

async def _generate_dialogue_content(context, npc, situation, stage, player_input, rel_context):
    """Generate dialogue content with GPT"""
    prompt = f"""
    Generate dialogue for {npc['npc_name']} in this situation:
    Situation: {situation}
    Stage: {stage.name}
    Player said: {player_input or 'nothing'}
    
    One line of natural dialogue fitting the stage and personality.
    """
    
    dialogue = await generate_text_completion(
        system_prompt=f"You write dialogue for NPCs. {npc['npc_name']} has traits: {npc.get('personality_traits', 'unknown')}",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    
    return dialogue.strip()

async def _generate_dialogue_tone(context, dominance, stage, situation):
    """Generate dialogue tone"""
    if dominance > 80 and stage != "Innocent Beginning":
        return "commanding"
    elif dominance > 60:
        return "confident"
    return "friendly"

async def _generate_dialogue_subtext(context, dialogue, dominance, stage, addictions):
    """Generate dialogue subtext"""
    if stage == "Full Revelation":
        return "Complete understanding of the power dynamic"
    elif stage == "Veil Thinning" and dominance > 70:
        return "The expectation of compliance barely hidden"
    return "Surface-level interaction"

async def _generate_body_language(context, dominance, tone, stage):
    """Generate body language"""
    if tone == "commanding":
        return "Posture radiating unquestionable authority"
    elif tone == "confident":
        return "Relaxed confidence in every gesture"
    return "Casual stance"

async def _identify_dialogue_triggers(context, dialogue, npc_id, rel_context):
    triggers = []
    if context.active_addictions and context.active_addictions.get("has_addictions"):
        triggers.append("addiction_vulnerability")
    if _get_dim(rel_context, "dependency", 0) > 50:
        triggers.append("dependency_reinforcement")
    return triggers

async def _generate_power_moment_setup(context, exchange, npc, world_state):
    """Generate power moment setup with GPT"""
    prompt = f"""
    Generate setup for power exchange:
    NPC: {npc['npc_name']}
    Type: {exchange.exchange_type.value}
    Mood: {world_state.world_mood.value}
    
    One sentence building natural tension.
    """
    
    setup = await generate_text_completion(
        system_prompt="You narrate power dynamics emerging from daily life",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    
    return setup.strip()

async def _generate_power_moment_description(context, exchange, npc):
    """Generate power moment description"""
    prompt = f"""
    Describe this power moment:
    {npc['npc_name']} exercises {exchange.exchange_type.value}
    Intensity: {exchange.intensity}
    
    One sentence showing the moment of control.
    """
    
    description = await generate_text_completion(
        system_prompt="You narrate subtle power exchanges",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    
    return description.strip()

async def _generate_power_moment_aftermath(context, exchange, dynamics):
    """Generate power moment aftermath"""
    return "The moment passes, but its weight lingers..."

async def _generate_player_feelings(context, exchange, dynamics):
    sub = getattr(dynamics, "player_submission_level", 0.0) if dynamics else 0.0
    prompt = f"""
    Describe player's feelings during power exchange:
    Submission level: {sub}
    Exchange type: {exchange.exchange_type.value}
    One sentence of internal experience.
    """
    
    feelings = await generate_text_completion(
        system_prompt="You describe psychological responses to power dynamics",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    
    return feelings.strip()

async def _generate_power_response_options(context, exchange, dominance, rel_state):
    """Generate response options"""
    options = ["Accept gracefully", "Subtle resistance"]
    if dominance < 70:
        options.append("Deflect with humor")
    return options

async def _calculate_power_consequences(context, exchange, world_state):
    """Calculate power exchange consequences"""
    consequences = []
    consequences.append(
        KeyValue(key="submission_increase", value=str(0.1 * exchange.intensity))
    )
    consequences.append(
        KeyValue(key="tension_change", value="0.05")
    )
    return consequences

async def _generate_routine_description(context, activity, world_state):
    mood_val = getattr(getattr(world_state, "world_mood", None), "value", "neutral") if world_state else "neutral"
    return f"You go about {activity} in the {mood_val} atmosphere"

async def _generate_routine_with_dynamics(context, activity, npcs, world_state):
    tens = getattr(world_state, "world_tension", None) if world_state else None
    dyn  = getattr(world_state, "relationship_dynamics", None) if world_state else None

    if tens and getattr(tens, "power_tension", 0) > 0.5:
        return f"Even {activity} carries subtle undertones of control"
    if dyn and getattr(dyn, "player_submission_level", 0) > 0.5:
        return f"The {activity} follows patterns you've grown accustomed to..."
    return f"A simple moment of {activity}"

async def _generate_npc_routine_involvement(context, npc_id, activity):
    """Generate NPC involvement in routine"""
    return f"Their presence during {activity} feels significant"

async def _identify_control_elements(context, activity, npcs, world_state):
    elements = []
    dyn = getattr(world_state, "relationship_dynamics", None) if world_state else None
    if dyn and getattr(dyn, "player_submission_level", 0) > 0.5:
        elements.append("The natural order of things")
    if dyn and getattr(dyn, "player_submission_level", 0) > 0.3:
        elements.append("Choices that aren't really choices")
    if dyn and hasattr(dyn, "acceptance_level") and getattr(dyn, "acceptance_level", 0) > 0.5:
        elements.append("The comfort of established patterns")
    return elements

async def _generate_subtle_control_elements(context: NarratorContext, activity: str,
                                           dynamics: Any) -> List[str]:
    """Generate subtle control elements in routine"""
    elements = []
    if hasattr(dynamics, 'player_submission_level'):
        if dynamics.player_submission_level > 0.3:
            elements.append("Choices that aren't really choices")
    if hasattr(dynamics, 'acceptance_level'):
        if dynamics.acceptance_level > 0.5:
            elements.append("The comfort of established patterns")
    return elements

async def _generate_emergent_variations(context, activity, intersections):
    """Generate emergent variations"""
    variations = []
    if "addiction_rule_synergy" in intersections:
        variations.append("Your needs color the moment differently")
    if "extreme_fatigue" in intersections:
        variations.append(f"Exhaustion making {activity} more difficult")
    if "addiction_low_willpower" in intersections:
        variations.append(f"Distraction coloring the {activity}")
    return variations

async def _generate_ambient_description(context, focus, world_state, intensity):
    """Generate ambient description with GPT"""
    mood_val = getattr(getattr(world_state, "world_mood", None), "value", "neutral") if world_state else "neutral"
    prompt = f"""
    Generate ambient description for:
    Focus: {focus}
    Mood: {mood_val}
    Intensity: {intensity}

    One atmospheric sentence.
    """

    description = await generate_text_completion(
        system_prompt="You create ambient world descriptions",
        user_prompt=prompt,
        model='gpt-5-nano',
    )
    return description.strip()


# Additional calculation and detection functions
def _calculate_susceptibility(vitals: Dict[str, int], addictions: Dict[str, Any]) -> float:
    """Calculate player susceptibility to power dynamics"""
    susceptibility = 0.5
    
    if vitals:
        if vitals.get("fatigue", 0) > 70:
            susceptibility += 0.2
        if vitals.get("hunger", 100) < 30:
            susceptibility += 0.15
        if vitals.get("thirst", 100) < 30:
            susceptibility += 0.15
    
    if addictions and addictions.get("has_addictions"):
        addiction_count = len(addictions.get("addictions", {}))
        susceptibility += min(0.3, addiction_count * 0.1)
    
    return min(1.0, susceptibility)

async def _get_relevant_addictions_for_npc(context: NarratorContext, npc_id: int) -> Dict[str, Any]:
    """Get addictions relevant to a specific NPC"""
    relevant = {}
    
    if not context.active_addictions or not context.active_addictions.get("has_addictions"):
        return relevant
    
    for addiction_type, data in context.active_addictions.get("addictions", {}).items():
        if data.get("target_npc_id") == npc_id:
            relevant[addiction_type] = data
    
    return relevant

async def _identify_emergent_elements(context: NarratorContext, scene: SliceOfLifeEvent, 
                                     relationship_contexts: Dict[int, Dict]) -> List[KeyValue]:
    """Identify emergent elements from system intersections"""
    emergent = []
    
    if context.system_intersections:
        emergent.append(KeyValue(key="triggers", value=str(context.system_intersections[:3])))
        
        potential_events = []
        for intersection in context.system_intersections:
            if "stat_combo" in intersection:
                potential_events.append("stat_combination_trigger")
            if "addiction_low_willpower" in intersection:
                potential_events.append("vulnerability_window")
            if "extreme_fatigue" in intersection:
                potential_events.append("exhaustion_event")
            if "relationship_progression" in intersection:
                potential_events.append("relationship_milestone")
        
        if potential_events:
            emergent.append(KeyValue(key="potential_events", value=str(potential_events)))
    
    if len(context.system_intersections) >= 2:
        emergent.append(KeyValue(key="hidden_connections", value="multiple_systems_converging"))
    
    return emergent if emergent else []

async def _calculate_power_exchange_consequences(context: NarratorContext, exchange: PowerExchange,
                                                rel_state: Any, susceptibility: float) -> List[KeyValue]:
    """Calculate potential consequences of power exchange"""
    consequences = []
    
    if susceptibility > 0.7:
        consequences.append(KeyValue(
            key="stat_change",
            value="willpower:-5,obedience:+8"
        ))
    
    if context.active_addictions and context.active_addictions.get("has_addictions"):
        consequences.append(KeyValue(
            key="addiction_progression",
            value=f"likelihood:{susceptibility:.2f}"
        ))
    
    if exchange.intensity > 0.5:
        consequences.append(KeyValue(
            key="memory_formation",
            value=f"type:emotional,significance:{min(1.0, exchange.intensity):.2f}"
        ))
    
    if hasattr(rel_state, 'dimensions') and rel_state.dimensions.influence > 60:
        patterns = list(rel_state.history.active_patterns)[:2] if hasattr(rel_state, 'history') else []
        consequences.append(KeyValue(
            key="relationship_deepening",
            value=f"patterns:{','.join(patterns) if patterns else 'none'}"
        ))
    
    return consequences

async def _present_response_options(context: NarratorContext, options: List[str]) -> List[str]:
    """Present player response options"""
    return [f"You could {option}..." for option in options[:3]]

# Activity and routine helper functions
async def _generate_addiction_effect_description(context: NarratorContext, addiction_type: str,
                                                level: int, activity: str) -> str:
    """Generate description of addiction effects on activity"""
    if level >= 3:
        return f"The {addiction_type} dependency colors everything about {activity}"
    elif level >= 2:
        return f"Thoughts of {addiction_type} intrude during {activity}"
    return ""

async def _generate_activity_description(context: NarratorContext, activity: str,
                                        time_period: TimeOfDay) -> str:
    """Generate basic activity description"""
    return f"The {time_period.value} {activity} unfolds with familiar rhythm..."

async def _determine_integrated_tone(context: NarratorContext, world_state: WorldState,
                                    relationship_contexts: Dict[int, Dict]) -> NarrativeTone:
    mood = getattr(world_state, "world_mood", None) if world_state else None
    base_tone = _determine_narrative_tone(mood or WorldMood.RELAXED, ActivityType.ROUTINE)

    vit = getattr(context, "current_vitals", None) or {}
    if vit.get("fatigue", 0) > 80:
        return NarrativeTone.OBSERVATIONAL
    if vit.get("hunger", 100) < 20:
        return NarrativeTone.COMMANDING

    if relationship_contexts:
        influences = [
            (_get_dim(rc, "influence", 0)) for rc in relationship_contexts.values()
        ]
        if influences:
            avg = sum(influences) / len(influences)
            if avg > 50: return NarrativeTone.SUBTLE
            if avg > 30: return NarrativeTone.TEASING

    if getattr(context, "active_addictions", None) and context.active_addictions.get("has_addictions"):
        if len(context.active_addictions.get("addictions", {})) >= 3:
            return NarrativeTone.SENSUAL

    return base_tone


async def _generate_activity_variations(context: NarratorContext, activity: str,
                                       intersections: List[str]) -> List[str]:
    """Generate emergent variations based on system state"""
    variations = []
    for intersection in intersections[:2]:
        if "fatigue" in intersection:
            variations.append(f"Exhaustion making {activity} more difficult")
        elif "addiction" in intersection:
            variations.append(f"Distraction coloring the {activity}")
    return variations

# Ambient narration functions
async def _narrate_time_passage(context: NarratorContext, current_time: CurrentTimeData) -> str:
    """Narrate the passage of time"""
    return f"Time moves forward, {current_time.time_of_day} settling over everything..."

async def _narrate_mood_shift(context: NarratorContext, mood: WorldMood) -> str:
    """Narrate a shift in world mood"""
    return f"The atmosphere shifts toward something more {mood.value}..."

async def _narrate_tension(context, tension: Optional[WorldTension]) -> str:
    if not tension:
        return "A tension you can't quite name gathers and recedes..."
    dominant_tensions = [
        ("power", getattr(tension, "power_tension", 0.0)),
        ("social", getattr(tension, "social_tension", 0.0)),
        ("sexual", getattr(tension, "sexual_tension", 0.0)),
        ("emotional", getattr(tension, "emotional_tension", 0.0)),
    ]
    dominant = max(dominant_tensions, key=lambda x: x[1])
    return f"A {dominant[0]} tension builds, barely perceptible but undeniable..."


async def _narrate_addiction_presence(context: NarratorContext, addictions: Dict[str, Any]) -> str:
    """Narrate the presence of addiction effects"""
    if addictions and addictions.get("has_addictions"):
        return "The familiar pull of need colors your perception..."
    return "Your mind remains clear, for now..."

async def _narrate_ambient_detail(context: NarratorContext, world_state: WorldState) -> str:
    mood_val = getattr(getattr(world_state, "world_mood", None), "value", "neutral") if world_state else "neutral"
    return f"The world continues its {mood_val} rhythm..."

async def _weave_relationship_event(context: NarratorContext, event: Any) -> str:
    """Weave a relationship event into narration"""
    return "A moment of connection shifts the dynamic subtly..."

async def _identify_reflected_systems(context: NarratorContext, focus: str,
                                     description: str) -> List[str]:
    """Identify which systems are reflected in narration"""
    systems = []
    if "addiction" in focus or "need" in description.lower():
        systems.append("addiction_system")
    if "tension" in focus:
        systems.append("tension_system")
    if "mood" in focus:
        systems.append("mood_system")
    return systems

# Tone and style determination functions
def _determine_narrative_tone(mood: WorldMood, event_type: ActivityType) -> NarrativeTone:
    """Determine appropriate narrative tone based on mood and activity"""
    tone_map = {
        (WorldMood.INTIMATE, ActivityType.INTIMATE): NarrativeTone.SENSUAL,
        (WorldMood.PLAYFUL, ActivityType.SOCIAL): NarrativeTone.TEASING,
        (WorldMood.OPPRESSIVE, ActivityType.ROUTINE): NarrativeTone.COMMANDING,
        (WorldMood.MYSTERIOUS, ActivityType.SPECIAL): NarrativeTone.OBSERVATIONAL,
        (WorldMood.RELAXED, ActivityType.LEISURE): NarrativeTone.CASUAL,
    }
    
    key = (mood, event_type)
    return tone_map.get(key, NarrativeTone.SUBTLE)

def _select_dialogue_power_dynamic(situation: str, dominance: int) -> PowerDynamicType:
    """Select appropriate power dynamic for dialogue based on context"""
    situation_lower = situation.lower()
    
    if dominance > 80:
        if "private" in situation_lower or "alone" in situation_lower:
            return PowerDynamicType.INTIMATE_COMMAND
        else:
            return PowerDynamicType.CASUAL_DOMINANCE
    elif dominance > 60:
        if "decision" in situation_lower:
            return PowerDynamicType.CASUAL_DOMINANCE
        elif "help" in situation_lower:
            return PowerDynamicType.PROTECTIVE_CONTROL
        else:
            return PowerDynamicType.SUBTLE_CONTROL
    else:
        return PowerDynamicType.PLAYFUL_TEASING

# Player action helper functions
async def _acknowledge_player_action(context, action):
    """Acknowledge player's action"""
    return f"You {action}..."

async def _generate_world_reaction(context, action, world_state):
    """Generate world's reaction to player action"""
    return f"The {world_state.world_mood.value} world responds to your action..."

async def _generate_npc_reaction(context, npc_id, action, world_state):
    """Generate NPC reaction to player action"""
    return "They notice, filing it away for later..."

async def _check_for_dynamic_shift(context, action, dynamics):
    """Check if action causes dynamic shift"""
    if hasattr(dynamics, 'player_submission_level'):
        if dynamics.player_submission_level < 0.3:
            return "The power dynamic shifts slightly in response..."
    elif hasattr(dynamics, 'resistance_level'):
        if dynamics.resistance_level < 0.3:
            return "The power dynamic shifts slightly in response..."
    return None

# ===============================================================================
# Main Narrator Class
# ===============================================================================

class SliceOfLifeNarrator:
    """
    Enhanced Slice-of-Life Narrator with context services integration
    """

    @property
    def world_director(self):
        # convenience passthrough so existing self.world_director references work
        return self.context.world_director
    
    async def _get_world_state_fast(self) -> Optional[WorldState]:
        ws = await _get_ws_prefer_bundle(self.context)
        if ws is not None:
            return ws
        try:
            # fall back to the director, not to ourselves
            ws = await self.world_director.get_world_state()
            self.context.current_world_state = ws
            return ws
        except Exception as e:
            logger.warning(f"_get_world_state_fast fallback failed: {e}", exc_info=True)
            return None
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = NarratorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Performance tracking
        self.performance_monitor = None
        
        # Sub-agents for specialized narration
        self.scene_narrator = Agent(
            name="SceneNarrator",
            instructions="""
            You narrate slice-of-life scenes in a femdom setting with full system awareness.
            Integrate relationship dynamics, time references, system states, and physical condition.
            Use second-person perspective. Be immersive and atmospheric.
            Show consequences of past events through environmental details.
            Draw from memories and context to make scenes feel lived-in.
            Respect Nyx governance directives.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(),
            tools=[narrate_slice_of_life_scene, generate_ambient_narration]
        )
        
        self.dialogue_writer = Agent(
            name="DialogueWriter",
            instructions="""
            You write natural NPC dialogue with awareness of progression stages and masks.
            Power dynamics through subtext, not explicit statements.
            Show mask slippage and true nature bleeding through when appropriate.
            Exploit known vulnerabilities subtly.
            Reference shared history and memories naturally.
            Follow Nyx governance guidelines for appropriate content.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(),
            tools=[generate_npc_dialogue]
        )
        
        self.power_narrator = Agent(
            name="PowerNarrator",
            instructions="""
            You narrate power exchanges with awareness of susceptibility and consequences.
            Make power dynamics feel inevitable rather than forced.
            Show how multiple systems reinforce control.
            Track exchanges through Nyx governance.
            Store significant moments in memory.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(),
            tools=[narrate_power_exchange]
        )
        
        self.routine_narrator = Agent(
            name="RoutineNarrator",
            instructions="""
            You narrate daily routines with subtle power dynamics woven throughout.
            Show how control manifests in mundane moments.
            Reference system states naturally (hunger, fatigue, addiction).
            Make routine activities feel meaningful within the power structure.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(),
            tools=[narrate_daily_routine]
        )

    async def initialize(self):
        """Initialize the narrator context and systems"""
        await self.context.initialize()
        
        # Initialize performance monitor if not already done
        if self.performance_monitor is None:
            self.performance_monitor = PerformanceMonitor.get_instance(
                self.user_id, self.conversation_id
            )
            

    async def generate_npc_dialogue(
        self,
        npc_id: int,
        situation: str,
        world_state: Optional[WorldState] = None,
        player_input: Optional[str] = None,
    ):
        call_id = uuid.uuid4().hex[:8]
        logger.info(
            "NPC_DIALOGUE_ADAPTER[%s]: enter npc_id=%s ws_provided=%s ws=%s",
            call_id, npc_id, world_state is not None, _ws_brief(world_state)
        )

        if world_state is None:
            try:
                world_state = await self._get_world_state_fast()
                logger.info("NPC_DIALOGUE_ADAPTER[%s]: fetched ws=%s", call_id, _ws_brief(world_state))
            except Exception as e:
                logger.warning("NPC_DIALOGUE_ADAPTER[%s]: failed to fetch ws: %s", call_id, e)
                world_state = None

        result = await self.dialogue_writer.run(
            messages=[{"role": "user", "content": situation}],
            context=self.context,
            tool_calls=[{
                "tool": generate_npc_dialogue,
                "kwargs": {
                    "npc_id": npc_id,
                    "situation": situation,
                    "world_state": world_state.model_dump() if hasattr(world_state, "model_dump") else world_state,
                    "player_input": player_input,
                },
            }],
        )

        logger.info("NPC_DIALOGUE_ADAPTER[%s]: exit", call_id)
        return getattr(result, "data", result)
    
    async def narrate_world_state(self) -> str:
        """Generate narration for current world state"""
        await self.context.refresh_context()
        
        world_state = await self._get_world_state_fast()
        
        # Create a scene from world state
        scene = SliceOfLifeEvent(
            event_type=ActivityType.ROUTINE,
            title="Current Moment",
            description="The world continues around you",
            location="current_location",
            participants=world_state.active_npcs if hasattr(world_state, 'active_npcs') else []
        )
        
        # Run scene narrator
        result = await self.scene_narrator.run(
            messages=[{"role": "user", "content": "Narrate the current world state"}],
            context=self.context,
            tool_calls=[{
                "tool": narrate_slice_of_life_scene,
                "kwargs": {
                    "payload": NarrateSliceOfLifeInput(
                        scene_type="routine",
                        scene=scene,
                        world_state=world_state
                    ).model_dump()
                }
            }]
        )
        
        narration = result.data if hasattr(result, 'data') else result  # should be SliceOfLifeNarration
        return narration.scene_description
    
    async def process_player_input(self, user_input: str) -> Dict[str, Any]:
        """Process player input and generate appropriate narration with full context.
        - Robust to tool return shapes (dict / JSON string / Pydantic model / Agents result with .data)
        - Null-guards for world_state fields and conflict participants
        - Uses dict payload for tool calls when world_state is missing
        """
        await self.initialize()
    
        # Refresh context with player input
        await self.context.refresh_context(user_input)
    
        import json
    
        def _to_dict(obj):
            """Best-effort convert tool/agent outputs to a dict."""
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            # Agents SDK often wraps data on .data
            data = getattr(obj, "data", None)
            if data is not None:
                return _to_dict(data)
            # Pydantic v2
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:
                    pass
            # Legacy JSON-string path
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    return {"raw": obj}
            # Last resort: shallow attr capture
            try:
                return {
                    k: getattr(obj, k)
                    for k in dir(obj)
                    if not k.startswith("_") and not callable(getattr(obj, k, None))
                }
            except Exception:
                return {"value": str(obj)}
    
        def _get(d_or_obj, key, default=None):
            """Dict-or-attr getter."""
            if isinstance(d_or_obj, dict):
                return d_or_obj.get(key, default)
            return getattr(d_or_obj, key, default)
    
        def _extract_npc_ids_from_ws(ws, limit=3):
            """Safely extract up to N NPC ids from a world_state.active_npcs of varying shapes."""
            out = []
            try:
                active = getattr(ws, "active_npcs", None) or []
                for item in list(active)[:limit]:
                    if isinstance(item, int):
                        out.append(item)
                    elif isinstance(item, dict):
                        nid = item.get("npc_id") or item.get("id")
                        if isinstance(nid, int):
                            out.append(nid)
                        elif isinstance(nid, str) and nid.isdigit():
                            out.append(int(nid))
                    else:
                        nid = getattr(item, "npc_id", None) or getattr(item, "id", None)
                        if isinstance(nid, int):
                            out.append(nid)
                        elif isinstance(nid, str) and nid.isdigit():
                            out.append(int(nid))
            except Exception:
                pass
            return out
    
        # NEW: Check if input triggers or affects conflicts
        conflict_triggered = None
        try:
            if self.context.conflict_synthesizer:
                participants = _extract_npc_ids_from_ws(self.context.current_world_state, limit=3) \
                               if self.context.current_world_state else []
                scene_context = {
                    "scene_type": "player_input",
                    "player_action": user_input,
                    "participants": participants,
                }
                conflict_result = await self.context.conflict_synthesizer.process_scene(scene_context)
                if conflict_result and conflict_result.get("conflicts_detected"):
                    conflict_triggered = conflict_result
        except Exception as e:
            logger.debug(f"process_player_input: conflict check failed: {e}")
    
        # Get world state and determine affected NPCs
        world_state = await self._get_world_state_fast()
        world_state_payload = _to_dict(world_state) if world_state is not None else {}
    
        # Check if input is dialogue
        affected_npcs: List[int] = []
        lower = user_input.lower()
        is_dialogue = any(keyword in lower for keyword in ["say", "tell", "ask", "talk"])
    
        if is_dialogue:
            active = getattr(world_state, "active_npcs", None) if world_state is not None else []
            # Normalize to up to 2 npc IDs
            affected_npcs = _extract_npc_ids_from_ws(world_state, limit=2) if active else []
            if affected_npcs:
                responses = []
                for npc_id in affected_npcs:
                    try:
                        result = await self.dialogue_writer.run(
                            messages=[{"role": "user", "content": user_input}],
                            context=self.context,
                            tool_calls=[{
                                "tool": generate_npc_dialogue,
                                "kwargs": {
                                    "npc_id": npc_id,
                                    "situation": user_input,
                                    "world_state": world_state_payload,
                                    "player_input": user_input
                                }
                            }]
                        )
                        dialogue_dict = _to_dict(result)
                        # Ensure minimal shape
                        if "npc_id" not in dialogue_dict:
                            dialogue_dict["npc_id"] = npc_id
                        responses.append(dialogue_dict)
                    except Exception as e:
                        logger.warning(f"dialogue tool failed for npc {npc_id}: {e}")
                        responses.append({"npc_id": npc_id, "npc_name": "Unknown", "dialogue": "..."})
    
                # Process player action (pass the object if available; else the dict payload)
                action_result = await self._process_player_action(
                    user_input, world_state if world_state is not None else world_state_payload, affected_npcs
                )
    
                return {
                    "success": True,
                    "type": "dialogue",
                    "npc_responses": responses,
                    "narrative": _get(action_result, "acknowledgment", ""),
                    "world_reaction": _get(action_result, "world_reaction", ""),
                    "dynamic_shift": _get(action_result, "dynamic_shift"),
                    "governance_approved": _get(action_result, "governance_approved", True),
                    "conflict_triggered": conflict_triggered,
                }
    
        # Process as scene action
        try:
            result = await self.scene_narrator.run(
                messages=[{"role": "user", "content": user_input}],
                context=self.context,
                tool_calls=[{
                    "tool": narrate_player_action,
                    "kwargs": {
                        "action": user_input,
                        "world_state": world_state_payload
                    }
                }]
            )
        except Exception as e:
            logger.warning(f"scene_narrator tool failed: {e}")
            result = {}
    
        narration = _to_dict(result)
    
        # Process the action for consequences
        action_result = await self._process_player_action(
            user_input, world_state if world_state is not None else world_state_payload, affected_npcs
        )
    
        return {
            "success": True,
            "type": "action",
            "narrative": narration.get("scene_description", _get(narration, "raw", "")),
            "atmosphere": narration.get("atmosphere", ""),
            "world_reaction": _get(action_result, "world_reaction", ""),
            "npc_reactions": _get(action_result, "npc_reactions", []),
            "dynamic_shift": _get(action_result, "dynamic_shift"),
            "governance_approved": _get(action_result, "governance_approved", True),
            "system_triggers": narration.get("system_triggers", []),
            "conflict_triggered": conflict_triggered,
        }

        
    async def _process_player_action(self, action: str, world_state: Any, affected_npcs: List[int]) -> Dict[str, Any]:
        """Process player action for system consequences"""
        context = self.context
        
        # Get attributes with fallback to operational context
        # (Since this is a class method, we need to check both contexts)
        governance_active = getattr(context, "governance_active", False)
        nyx_governance = getattr(context, "nyx_governance", None)
        memory_manager = getattr(context, "memory_manager", None)
        
        # Check governance approval
        if governance_active and nyx_governance:
            try:
                approval = await nyx_governance.check_permission(
                    agent_type="narrator",
                    action_type="player_action",
                    context={"action": action, "affected_npcs": affected_npcs}
                )
                if not approval.get("approved", True):
                    return {
                        "acknowledgment": approval.get("message", "That action isn't possible right now."),
                        "world_reaction": "",
                        "npc_reactions": [],
                        "dynamic_shift": None,
                        "governance_blocked": True
                    }
            except Exception as e:
                logger.warning(f"Governance check failed for player action: {e}")
        
        # Generate action acknowledgment
        acknowledgment = await _acknowledge_player_action(context, action)
        
        # Generate world reaction
        world_reaction = await _generate_world_reaction(
            context, action, world_state
        )
        
        # Generate NPC reactions if any
        npc_reactions = []
        if affected_npcs:
            for npc_id in affected_npcs:
                reaction = await _generate_npc_reaction(
                    context, npc_id, action, world_state
                )
                if reaction:
                    npc_reactions.append(reaction)
        
        # Determine if this shifts any dynamics
        dynamic_shift = None
        if hasattr(world_state, 'relationship_dynamics'):
            dynamics = world_state.relationship_dynamics
            if hasattr(dynamics, 'player_submission_level') and dynamics.player_submission_level > 0.5:
                dynamic_shift = await _check_for_dynamic_shift(
                    context, action, dynamics
                )
        
        # Record action in memory using cached memory_manager
        if memory_manager:
            try:
                await memory_manager.add_memory(
                    content=f"Player action: {action}",
                    memory_type="decision",
                    importance=0.7,
                    tags=["player_action"] + [f"npc_{npc_id}" for npc_id in (affected_npcs or [])[:3]],
                    metadata=MemoryMetadata(
                        context_type="player_action"
                    ).model_dump()
                )
            except Exception as e:
                logger.warning(f"Failed to store player action in memory: {e}")
        
        return {
            "acknowledgment": acknowledgment,
            "world_reaction": world_reaction,
            "npc_reactions": npc_reactions,
            "dynamic_shift": dynamic_shift,
            "maintains_atmosphere": True,
            "governance_approved": not (governance_active and nyx_governance and not approval.get("approved", True)) if 'approval' in locals() else True
        }
    
    async def narrate_time_transition(self, old_time: str, new_time: str) -> str:
        world_state = await self._get_world_state_fast()
        await self.context.refresh_context()
    
        if not world_state:
            # Minimal graceful fallback
            return "Time slips forward, softening the edges of the moment."
    
        result = await self.scene_narrator.run(
            messages=[{"role": "user", "content": "Narrate time passing"}],
            context=self.context,
            tool_calls=[{
                "tool": generate_ambient_narration,
                "kwargs": {"focus": "time_passage", "world_state": world_state.model_dump()},
            }],
        )
        ambient = getattr(result, "data", result)
        # Try to extract description safely
        if hasattr(ambient, "description"):
            return ambient.description
        if isinstance(ambient, dict) and "description" in ambient:
            return ambient["description"]
        try:
            import json
            return json.loads(ambient).get("description", "Time passes...")
        except Exception:
            return "Time passes..."

    
    async def orchestrate_scene(self, scene_type: str = "routine", npcs: List[int] = None) -> Dict[str, Any]:
        """Orchestrate a complete scene with multiple elements"""
        await self.initialize()
        
        world_state = await self._get_world_state_fast()
        npcs = npcs or []
        
        # NEW: Check for conflict involvement
        scene_conflicts = []
        if self.context.active_conflicts:
            for conflict in self.context.active_conflicts:
                if isinstance(conflict, dict):
                    conflict_npcs = conflict.get('participants', [])
                    if any(npc in conflict_npcs for npc in npcs):
                        scene_conflicts.append(conflict)
        
        # Create scene event
        scene = SliceOfLifeEvent(
            event_type=ActivityType.ROUTINE if scene_type == "routine" else ActivityType.SOCIAL,
            title=f"{scene_type.title()} Scene",
            description=f"A {scene_type} moment in daily life",
            location="current_location",
            participants=npcs
        )
        
        # Generate base narration
        narration_result = await self.scene_narrator.run(
            messages=[{"role": "user", "content": f"Orchestrate a {scene_type} scene"}],
            context=self.context,
            tool_calls=[{
                "tool": narrate_slice_of_life_scene,
                "kwargs": {
                    "payload": NarrateSliceOfLifeInput(
                        scene_type=scene_type,
                        scene=scene,
                        world_state=world_state
                    ).model_dump()
                }
            }]
        )
        narration = narration_result.data if hasattr(narration_result, 'data') else narration_result
        
        # Generate dialogues for participating NPCs
        dialogues = []
        for npc_id in npcs:
            dialogue_result = await self.dialogue_writer.run(
                messages=[{"role": "user", "content": f"Generate ambient dialogue for scene"}],
                context=self.context,
                tool_calls=[{
                    "tool": generate_npc_dialogue,
                    "kwargs": {
                        "npc_id": npc_id,
                        "situation": scene_type,
                        "world_state": world_state.model_dump()
                    }
                }]
            )
            dialogue = dialogue_result.data if hasattr(dialogue_result, 'data') else dialogue_result
            dialogues.append(dialogue)
        
        # Check for power dynamics
        power_moments = []
        # NEW: Conflicts increase chance of power moments
        power_chance = 0.5 if not scene_conflicts else 0.7
        
        if world_state.world_tension.power_tension > power_chance and npcs:
            for npc_id in npcs[:1]:  # One power moment per scene max
                exchange = PowerExchange(
                    initiator_npc_id=npc_id,
                    initiator_id=npc_id,
                    exchange_type=PowerDynamicType.SUBTLE_CONTROL,
                    intensity=world_state.world_tension.power_tension
                )
                
                power_result = await self.power_narrator.run(
                    messages=[{"role": "user", "content": "Narrate power moment"}],
                    context=self.context,
                    tool_calls=[{
                        "tool": narrate_power_exchange,
                        "kwargs": {
                            "exchange": exchange.model_dump(),
                            "world_state": world_state.model_dump()
                        }
                    }]
                )
                power_moment = power_result.data if hasattr(power_result, 'data') else power_result
                power_moments.append(power_moment)
        
        return {
            "scene": scene.model_dump() if hasattr(scene, 'model_dump') else dict(scene),
            "narration": narration,
            "dialogues": dialogues,
            "power_moments": power_moments,
            "world_mood": world_state.world_mood.value if hasattr(world_state.world_mood, 'value') else str(world_state.world_mood),
            "active_conflicts": scene_conflicts  # NEW
        }
    
    async def generate_emergent_narrative(self) -> Dict[str, Any]:
        """Detect and generate emergent narrative threads"""
        await self.initialize()
        await self.context.refresh_context()
        
        # Analyze system intersections for emergent patterns
        emergent_threads = []
        
        # Check memory patterns
        if self.context.active_memories and len(self.context.active_memories) > 5:
            memory_pattern = self._analyze_memory_pattern(self.context.active_memories)
            if memory_pattern:
                emergent_threads.append({
                    "type": "memory_convergence",
                    "description": memory_pattern,
                    "significance": 0.7
                })
        
        # Check relationship patterns
        if self.context.current_context and "relationship_overview" in self.context.current_context:
            overview = self.context.current_context["relationship_overview"]
            if overview.get("most_advanced_npcs"):
                for npc_data in overview["most_advanced_npcs"]:
                    if npc_data.get("stage_name") in ["Veil Thinning", "Full Revelation"]:
                        emergent_threads.append({
                            "type": "relationship_climax",
                            "npc_id": npc_data.get("npc_id"),
                            "description": f"Relationship with {npc_data.get('npc_name')} reaching critical point",
                            "significance": 0.9
                        })
        
        # Check system convergence
        if len(self.context.system_intersections) >= 3:
            emergent_threads.append({
                "type": "system_convergence",
                "systems": self.context.system_intersections[:3],
                "description": "Multiple systems creating compound effects",
                "significance": 0.8
            })
        
        # Generate narrative for most significant thread
        if emergent_threads:
            most_significant = max(emergent_threads, key=lambda x: x.get("significance", 0))
            
            # Generate narrative description
            from logic.chatgpt_integration import generate_text_completion
            
            prompt = f"""
            Generate a narrative description for this emergent story thread:
            Type: {most_significant.get('type')}
            Description: {most_significant.get('description')}
            
            Make it subtle and atmospheric, showing how this emerges from daily life.
            2-3 sentences maximum.
            """
            
            narrative = await generate_text_completion(
                system_prompt="You narrate emergent story threads in a slice-of-life femdom setting",
                user_prompt=prompt,
                model='gpt-5-nano',
            )
            
            return {
                "has_emergent_narrative": True,
                "primary_thread": most_significant,
                "all_threads": emergent_threads,
                "narrative": narrative.strip()
            }
        
        return {
            "has_emergent_narrative": False,
            "primary_thread": None,
            "all_threads": [],
            "narrative": "The day continues its familiar rhythm..."
        }
    
    def _analyze_memory_pattern(self, memories: List[Any]) -> Optional[str]:
        """Analyze memories for patterns"""
        # Simple pattern detection
        themes = {}
        for memory in memories:
            if hasattr(memory, 'tags'):
                for tag in memory.tags:
                    themes[tag] = themes.get(tag, 0) + 1
            elif isinstance(memory, dict) and 'tags' in memory:
                for tag in memory['tags']:
                    themes[tag] = themes.get(tag, 0) + 1
        
        if themes:
            dominant_theme = max(themes.items(), key=lambda x: x[1])
            if dominant_theme[1] >= 3:
                return f"Recurring theme of {dominant_theme[0]} emerging from past experiences"
        
        return None
    
    async def handle_group_interaction(self, npc_ids: List[int], interaction_type: str = "social") -> Dict[str, Any]:
        """Handle interaction with multiple NPCs"""
        await self.initialize()
        
        world_state = await self._get_world_state_fast()
        
        # Create group scene
        scene = SliceOfLifeEvent(
            event_type=ActivityType.SOCIAL,
            title=f"Group {interaction_type.title()}",
            description=f"Interacting with multiple people",
            location="current_location",
            participants=npc_ids
        )
        
        # Generate scene narration
        narration_result = await self.scene_narrator.run(
            messages=[{"role": "user", "content": f"Narrate group {interaction_type}"}],
            context=self.context,
            tool_calls=[{
                "tool": narrate_slice_of_life_scene,
                "kwargs": {
                    "payload": NarrateSliceOfLifeInput(
                        scene_type="social",
                        scene=scene,
                        world_state=world_state
                    ).model_dump()
                }
            }]
        )
        
        narration = narration_result.data if hasattr(narration_result, 'data') else narration_result
        
        # Generate dialogue for each NPC
        dialogues = []
        for npc_id in npc_ids:
            dialogue_result = await self.dialogue_writer.run(
                messages=[{"role": "user", "content": f"Generate group interaction dialogue"}],
                context=self.context,
                tool_calls=[{
                    "tool": generate_npc_dialogue,
                    "kwargs": {
                        "npc_id": npc_id,
                        "situation": f"group {interaction_type}",
                        "world_state": world_state.model_dump()
                    }
                }]
            )
            dialogue = dialogue_result.data if hasattr(dialogue_result, 'data') else dialogue_result
            dialogues.append(dialogue)
        
        # Check for group dynamics
        group_dynamic = None
        if len(npc_ids) > 2:
            # Determine if there's a group power dynamic
            from logic.chatgpt_integration import generate_text_completion
            
            prompt = f"""
            Analyze this group interaction for power dynamics:
            NPCs: {len(npc_ids)} people
            Type: {interaction_type}
            
            Describe any group dynamics or hierarchies emerging (1-2 sentences).
            """
            
            dynamic_desc = await generate_text_completion(
                system_prompt="You analyze group dynamics in social situations",
                user_prompt=prompt,
                model='gpt-5-nano',
            )
            
            group_dynamic = dynamic_desc.strip()
        
        return {
            "scene": scene.model_dump() if hasattr(scene, 'model_dump') else dict(scene),
            "narration": narration,
            "npc_dialogues": dialogues,
            "group_dynamic": group_dynamic,
            "participant_count": len(npc_ids)
        }
    
    async def generate_overheard_conversation(self, npc_ids: List[int], about_player: bool = False) -> Dict[str, Any]:
        """Generate a conversation between NPCs that player overhears"""
        await self.initialize()
        
        # Generate conversation content
        from logic.chatgpt_integration import generate_text_completion
        
        conversation_context = "about you" if about_player else "about their own concerns"
        
        prompt = f"""
        Generate a conversation between {len(npc_ids)} NPCs that the player overhears.
        The conversation is {conversation_context}.
        
        Format as dialogue only, 3-5 exchanges.
        Show personality through speech patterns.
        {f"They don't know the player is listening." if about_player else ""}
        """
        
        conversation = await generate_text_completion(
            system_prompt="You write overheard NPC conversations in a slice-of-life setting",
            user_prompt=prompt,
            model='gpt-5-nano',
        )
        
        # Analyze conversation for revelations
        revelations = []
        if about_player:
            # Check if conversation reveals NPC true feelings
            if "control" in conversation.lower() or "perfect" in conversation.lower():
                revelations.append("NPCs discussing their influence over you")
            if "plan" in conversation.lower() or "next" in conversation.lower():
                revelations.append("NPCs coordinating future interactions")
        
        return {
            "conversation": conversation.strip(),
            "participants": npc_ids,
            "about_player": about_player,
            "revelations": revelations,
            "player_discovered": False
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the narrator"""
        if self.performance_monitor:
            return self.performance_monitor.get_metrics()
        return {}

# ===============================================================================
# Validation Helper for Schema Checking
# ===============================================================================

def validate_agent_safe_schema(model_class):
    """
    Validate that a model's schema doesn't contain additionalProperties.
    Raises RuntimeError if validation fails.
    """
    schema = model_class.model_json_schema()
    
    def find_issues(obj, path=""):
        if isinstance(obj, dict):
            issues = []
            if "additionalProperties" in obj or "unevaluatedProperties" in obj:
                issues.append(path or model_class.__name__)
            for k, v in obj.items():
                issues.extend(find_issues(v, f"{path}.{k}" if path else k))
            return issues
        elif isinstance(obj, list):
            issues = []
            for i, item in enumerate(obj):
                issues.extend(find_issues(item, f"{path}[{i}]"))
            return issues
        return []
    
    problems = find_issues(schema)
    if problems:
        raise RuntimeError(f"{model_class.__name__} schema not agent-safe at: {problems}")

# Validate all models on module load (optional, for debugging)
if __name__ == "__main__":
    # Validate all tool models
    for model in [
        SliceOfLifeEvent, PowerExchange, WorldTension,
        RelationshipDynamics, NPCRoutine,
        SliceOfLifeNarration, NPCDialogue, AmbientNarration,
        PowerMomentNarration, DailyActivityNarration,
        NarrateSliceOfLifeInput,
    ]:
        try:
            validate_agent_safe_schema(model)
            print(f"✓ {model.__name__} is agent-safe")
        except RuntimeError as e:
            print(f"✗ {e}")

# ===============================================================================
# Export
# ===============================================================================

def create_slice_of_life_narrator(user_id: int, conversation_id: int) -> SliceOfLifeNarrator:
    """Create an enhanced narrator with full context integration"""
    return SliceOfLifeNarrator(user_id, conversation_id)

__all__ = [
    'SliceOfLifeNarrator',
    'create_slice_of_life_narrator',
    'narrate_slice_of_life_scene',
    'generate_npc_dialogue',
    'narrate_power_exchange',
    'narrate_daily_routine',
    'generate_ambient_narration',
    'narrate_player_action',
    # Models
    'SliceOfLifeNarration',
    'NPCDialogue',
    'PowerMomentNarration',
    'DailyActivityNarration',
    'AmbientNarration',
    'NarrativeTone',
    'SceneFocus',
    # Input models
    'SliceOfLifeEvent',
    'PowerExchange',
    'WorldTension',
    'RelationshipDynamics',
    'NPCRoutine',
    # Base class for other modules
    'AgentSafeModel',
    'KeyValue',
    # Validation helper
    'validate_agent_safe_schema'
]
