# logic/addiction_system_sdk.py
"""
Refactored Addiction System with full Nyx Governance integration.

REFACTORED: All database writes now go through canon or LoreSystem
FIXED: Separated implementation functions from decorated tools to avoid 'FunctionTool' not callable errors
FIXED: Incorporated feedback from code review
NEW: Smart gating system to only show addiction messages when contextually appropriate

Features:
1) Complete integration with Nyx central governance
2) Permission checking before all operations
3) Action reporting for monitoring and tracing
4) Directive handling for system control
5) Registration with proper agent types and constants
6) Smart context-aware message gating with cooldowns
"""

import logging
import random
import json
import asyncio
import os
import time
import hashlib
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Iterable, Set, Tuple

# OpenAI Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    function_tool,
    RunContextWrapper,
    GuardrailFunctionOutput,
    InputGuardrail,
    trace,
    handoff
)
from agents.models.openai_responses import OpenAIResponsesModel
from pydantic import BaseModel, Field, field_validator, model_validator

# DB connection - UPDATED: Using new async context manager
from db.connection import get_db_connection_context

# Import lore system for canonical writes
from lore.core.lore_system import LoreSystem

# Nyx governance integration
# Moved imports to function level to avoid circular imports
import nyx.gateway.llm_gateway as llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

from nyx.nyx_governance import (
    AgentType,
    DirectiveType,
    DirectivePriority
)
from nyx.governance_helpers import (
    with_governance_permission,
    with_action_reporting,
    with_governance
)
from nyx.directive_handler import DirectiveHandler

# -------------------------------------------------------------------------------
# Persistent Gate State (per-conversation, survives context recreation)
# -------------------------------------------------------------------------------

@dataclass
class _PersistentGate:
    """Persistent cooldown/dedupe state per conversation"""
    last_any_turn: int = -1_000_000
    last_any_ts: float = 0.0
    last_ambient_turn: int = -1_000_000
    last_ambient_ts: float = 0.0
    seen_stimuli: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    rng_counter: int = 0  # NEW: Counter for deterministic RNG

_GATE_STATE: Dict[Tuple[int, int], _PersistentGate] = {}  # (user_id, conversation_id) -> state
_GATE_LOCKS: Dict[Tuple[int, int], asyncio.Lock] = {}
_GATE_TTL_SECONDS = 60 * 60 * 12  # 12h idle GC
_GATE_TOUCH: Dict[Tuple[int, int], float] = {}

def _gate_for(uid: int, cid: int) -> Tuple[_PersistentGate, asyncio.Lock]:
    """Get or create gate state with lock, perform opportunistic GC"""
    key = (uid, cid)
    st = _GATE_STATE.setdefault(key, _PersistentGate())
    lk = _GATE_LOCKS.setdefault(key, asyncio.Lock())
    _GATE_TOUCH[key] = time.time()
    # opportunistic GC
    cutoff = time.time() - _GATE_TTL_SECONDS
    for k, ts in list(_GATE_TOUCH.items()):
        if ts < cutoff:
            _GATE_TOUCH.pop(k, None)
            _GATE_STATE.pop(k, None)
            _GATE_LOCKS.pop(k, None)
    return st, lk

def _next_rand(pg: _PersistentGate, user_id: int, conversation_id: int, salt: str = "") -> float:
    """Generate deterministic random float in [0,1) using counter and salt"""
    import hashlib
    pg.rng_counter += 1
    seed = f"{user_id}:{conversation_id}:{pg.rng_counter}:{salt}".encode()
    digest = hashlib.blake2b(seed, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)

def purge_gate_state(user_id: int, conversation_id: int):
    """Explicitly purge gate state for a conversation"""
    key = (user_id, conversation_id)
    for d in (_GATE_STATE, _GATE_LOCKS, _GATE_TOUCH):
        d.pop(key, None)

# -------------------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# -------------------------------------------------------------------------------

class AddictionUpdate(BaseModel):
    """Structure for addiction update results"""
    addiction_type: str = Field(..., description="Type of addiction")
    previous_level: int = Field(..., description="Previous addiction level")
    new_level: int = Field(..., description="New addiction level")
    level_name: str = Field(..., description="Name of the current level")
    progressed: bool = Field(..., description="Whether addiction progressed")
    regressed: bool = Field(..., description="Whether addiction regressed")
    target_npc_id: Optional[int] = Field(None, description="Target NPC ID if applicable")

class AddictionStatus(BaseModel):
    """Structure for overall addiction status"""
    addiction_levels: Dict[str, int] = Field(default_factory=dict, description="General addiction levels")
    npc_specific_addictions: List[Dict[str, Any]] = Field(default_factory=list, description="NPC-specific addictions")
    has_addictions: bool = Field(False, description="Whether the player has any addictions")

class AddictionEffects(BaseModel):
    """Structure for addiction narrative effects"""
    effects: List[str] = Field(default_factory=list, description="Narrative effects from addictions")
    has_effects: bool = Field(False, description="Whether there are any effects to display")

class AddictionSafety(BaseModel):
    """Output for addiction content moderation guardrail"""
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

class ThematicMessage(BaseModel):
    level: int = Field(..., ge=1, le=4, description="Addiction severity tier 1-4")
    text: str = Field(..., description="Short in-world narrative line; 1-2 sentences max.")

    @field_validator("text", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

class ThematicAddictionMessages(BaseModel):
    addiction_type: str = Field(..., description="e.g., 'feet', 'humiliation'")
    messages: List[ThematicMessage] = Field(..., min_items=4, max_items=4)

    @model_validator(mode="after")
    def _levels_cover_1_to_4(self) -> "ThematicAddictionMessages":
        levels = sorted({m.level for m in self.messages})
        if levels != [1, 2, 3, 4]:
            raise ValueError("Must include levels 1-4 exactly once each.")
        return self

class ThematicMessagesBundle(BaseModel):
    """Top-level object returned by the generator agent."""
    addictions: List[ThematicAddictionMessages]

# -------------------------------------------------------------------------------
# Trigger Configuration
# -------------------------------------------------------------------------------

@dataclass
class AddictionTriggerConfig:
    """Configuration for when addiction messages should appear"""
    # Only fire in scenes that smell like kink/fetish—or when feasibility tagged it.
    allowed_scene_tags: Set[str] = field(default_factory=lambda: {
        "erotic", "sex", "tease", "humiliation", "punishment", "aftercare", "ritual", "fetish",
        "kink", "dominance", "submissive", "intimate"
    })
    intent_markers: Set[str] = field(default_factory=lambda: {
        "fetish", "submission", "addiction_trigger", "dominance", "humiliation", "arousal", "kink"
    })

    # Ambient in neutral scenes if a relevant stimulus is present (e.g., sandals -> feet)
    ambient_in_neutral: bool = True
    ambient_prob_base: float = 0.10      # baseline chance if relevant + cooldown
    ambient_prob_lvl4: float = 0.28      # stronger cravings at higher levels
    ambient_cooldown_turns: int = 2
    ambient_cooldown_seconds: float = 45.0

    # Global spam guards (for any tier)
    min_turn_gap: int = 3
    min_seconds_between: float = 75.0

    # Which stimuli map to which addictions (extend as needed)
    stimuli_affinity: dict = field(default_factory=lambda: {
        "feet": {"feet","toes","ankle","barefoot","sandals","flipflops","heels"},
        "scent": {"perfume","musk","sweat","locker","gym","laundry","socks"},
        "socks": {"socks","ankle_socks","knee_highs","thigh_highs","stockings"},
        "ass": {"hips","ass","shorts","tight_skirt"},
        "humiliation": {"snicker","laugh","eye_roll","dismissive"},
        "submission": {"order","command","kneel","obedience"},
        "sweat": {"gym","workout","locker","perspiration","moist"},
    })

    # Tier thresholds (soft vs major)
    priority_level: int = 3      # >= this pushes toward soft/major

# -------------------------------------------------------------------------------
# Global Constants & Thematic Messages
# -------------------------------------------------------------------------------

ADDICTION_LEVELS = {
    0: "None",
    1: "Mild",
    2: "Moderate",
    3: "Heavy",
    4: "Extreme"
}

# Default fallback if external JSON is missing
ADDICTION_TYPES = ["socks", "feet", "sweat", "ass", "scent", "humiliation", "submission"]

# Minimal bare fallback used only if generation fails catastrophically.
_MIN_FALLBACK_MSG = "You feel a tug of desire you can't quite ignore."
_DEFAULT_THEMATIC_MESSAGES_MIN = {
    t: {str(i): _MIN_FALLBACK_MSG for i in range(1, 5)} for t in ADDICTION_TYPES
}

# Config helpers --------------------------------------------------------------

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


THEMATIC_MESSAGES_FILE = os.getenv("THEMATIC_MESSAGES_FILE", "thematic_messages.json")
THEMATIC_GENERATION_TIMEOUT_ACTIVE = _get_env_float(
    "ADDICTION_THEMATIC_GENERATION_TIMEOUT", 60.0
)
_DEFAULT_THEMATIC_MESSAGES = _DEFAULT_THEMATIC_MESSAGES_MIN

################################################################################
# Thematic Message Loader - Singleton, Async & Dynamic
################################################################################

class ThematicMessages:
    _instance: Optional["ThematicMessages"] = None
    _lock = asyncio.Lock()

    def __init__(self, fallback: dict, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        self.messages = fallback
        self.file_source = None
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._generated = False

    @classmethod
    async def get(
        cls,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        refresh: bool = False,
        defer_generation: bool = False,
        ensure_generated: bool = False,
        generation_timeout: Optional[float] = None,
        force_generate: bool = False,
    ):
        """
        Global singleton. Pass user & convo if available for agent generation / governance.
        refresh=True forces regeneration (ignoring cached file).
        """
        async with cls._lock:
            instance = cls._instance
            if instance is None or refresh:
                instance = cls(_DEFAULT_THEMATIC_MESSAGES_MIN, user_id, conversation_id)
                await instance._load(
                    refresh=refresh,
                    allow_generate=not defer_generation,
                    generation_timeout=None if defer_generation else generation_timeout,
                    force_generate=force_generate and not defer_generation,
                )
                cls._instance = instance
            else:
                if user_id is not None:
                    instance.user_id = user_id
                if conversation_id is not None:
                    instance.conversation_id = conversation_id

            if ensure_generated:
                await cls._instance.ensure_generated(
                    generation_timeout=generation_timeout,
                    force=force_generate,
                )

            return cls._instance

    async def _load(
        self,
        refresh: bool = False,
        allow_generate: bool = True,
        generation_timeout: Optional[float] = None,
        force_generate: bool = False,
    ):
        """
        Load from file if present & not refreshing; optionally generate via agent.
        Merge user overrides over generated; fill gaps w/ min fallback.
        """
        generated: Dict[str, Dict[str, str]] = {}
        file_msgs: Dict[str, Dict[str, str]] = {}

        # Load file overrides if available
        try:
            if not refresh and os.path.exists(THEMATIC_MESSAGES_FILE):
                with open(THEMATIC_MESSAGES_FILE, "r") as f:
                    file_msgs = json.load(f)
                    self.file_source = THEMATIC_MESSAGES_FILE
                logging.info(f"Thematic messages loaded from {THEMATIC_MESSAGES_FILE}")
        except Exception as e:
            logging.warning(f"Could not load external thematic messages: {e}")

        should_generate = allow_generate and (force_generate or refresh or not file_msgs)
        if should_generate:
            try:
                gen_coro = generate_thematic_messages_via_agent(
                    user_id=self.user_id or 0,
                    conversation_id=self.conversation_id or 0,
                    timeout=generation_timeout,
                )
                generated = await gen_coro
                self.file_source = "generated"
                self._generated = True
                try:
                    with open(THEMATIC_MESSAGES_FILE, "w") as f:
                        json.dump(generated, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logging.warning(f"Failed to persist generated thematic messages: {e}")
            except asyncio.TimeoutError:
                timeout_desc = generation_timeout if generation_timeout is not None else "unknown"
                logging.error(
                    f"Thematic message generation timed out after {timeout_desc} seconds; using fallback."
                )
            except Exception as e:
                logging.error(f"Thematic message generation error: {e}")

        # Merge precedence: file overrides > generated > min fallback
        merged: Dict[str, Dict[str, str]] = {}
        for t in ADDICTION_TYPES:
            merged[t] = _DEFAULT_THEMATIC_MESSAGES_MIN[t].copy()
            if generated and t in generated:
                merged[t].update({str(k): v for k, v in generated[t].items()})
            if file_msgs and t in file_msgs:
                merged[t].update({str(k): v for k, v in file_msgs[t].items()})
        self.messages = merged

    async def ensure_generated(
        self,
        generation_timeout: Optional[float] = None,
        force: bool = False,
    ) -> None:
        if self._generated and not force:
            return
        await self._load(
            refresh=False,
            allow_generate=True,
            generation_timeout=generation_timeout,
            force_generate=force or not self._generated,
        )

    def get_for(self, addiction_type: str, level: Union[int, str]) -> str:
        level_str = str(level)
        return self.messages.get(addiction_type, {}).get(level_str, "")

    def get_levels(self, addiction_type: str, up_to_level: int) -> List[str]:
        return [
            msg for lvl in range(1, up_to_level + 1)
            if (msg := self.get_for(addiction_type, lvl))
        ]

################################################################################
# Main Context with Smart Gating
################################################################################

class AddictionContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.thematic_messages: Optional[ThematicMessages] = None
        self.thematic_generation_timeout = THEMATIC_GENERATION_TIMEOUT_ACTIVE
        self.directive_handler: Optional[DirectiveHandler] = None
        self.prohibited_addictions: set = set()
        self.directive_task = None
        self.lore_system = None

        # Trigger governance (tiered, stimulus-aware)
        self.trigger_cfg = AddictionTriggerConfig()
        self.current_context: Dict[str, Any] = {}
        self._latched_window = False
        
        # Use persistent gate state with lock that survives context recreation
        self._pg, self._pg_lock = _gate_for(user_id, conversation_id)

    async def initialize(self, start_background: bool = False):
        """Initialize the addiction context. Only set start_background=True for long-lived contexts."""
        from nyx.integrate import get_central_governance
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        self.thematic_messages = await ThematicMessages.get(
            self.user_id,
            self.conversation_id,
            defer_generation=True,
        )
        self.lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
        self.directive_handler = DirectiveHandler(
            self.user_id, self.conversation_id,
            AgentType.UNIVERSAL_UPDATER, "addiction_system", governance=self.governor
        )
        self.directive_handler.register_handler(DirectiveType.ACTION, self._handle_action_directive)
        self.directive_handler.register_handler(DirectiveType.PROHIBITION, self._handle_prohibition_directive)
        
        # Only start background task for long-lived contexts to avoid task leak
        if start_background and not self.directive_task:
            self.directive_task = self.directive_handler.start_background_processing(interval=60.0)

    async def get_thematic_messages(
        self,
        require_generation: bool = False,
        generation_timeout: Optional[float] = None,
        force: bool = False,
    ) -> ThematicMessages:
        """Retrieve thematic messages, generating them lazily when required."""
        if generation_timeout is None and require_generation:
            generation_timeout = self.thematic_generation_timeout

        if self.thematic_messages is None:
            self.thematic_messages = await ThematicMessages.get(
                self.user_id,
                self.conversation_id,
                defer_generation=not require_generation,
                ensure_generated=require_generation,
                generation_timeout=generation_timeout,
                force_generate=force,
            )
        elif require_generation:
            await self.thematic_messages.ensure_generated(
                generation_timeout=generation_timeout,
                force=force,
            )

        return self.thematic_messages

    # --- trigger helpers (tiered, stimulus-aware) ---------------------------------
    def _cooldowns_ok(self, turn_idx: int, now: float) -> bool:
        if (turn_idx - self._pg.last_any_turn) < self.trigger_cfg.min_turn_gap:
            return False
        if (now - self._pg.last_any_ts) < self.trigger_cfg.min_seconds_between:
            return False
        return True

    def _ambient_cooldowns_ok(self, turn_idx: int, now: float) -> bool:
        if (turn_idx - self._pg.last_ambient_turn) < self.trigger_cfg.ambient_cooldown_turns:
            return False
        if (now - self._pg.last_ambient_ts) < self.trigger_cfg.ambient_cooldown_seconds:
            return False
        return True

    def _mark_emit(self, turn_idx: int, tier: str):
        """Mark emission with thread-safe locking"""
        now = time.time()
        # Create task to handle lock acquisition without blocking
        async def _do():
            async with self._pg_lock:
                self._pg.last_any_turn, self._pg.last_any_ts = turn_idx, now
                if tier == "ambient":
                    self._pg.last_ambient_turn, self._pg.last_ambient_ts = turn_idx, now
        asyncio.create_task(_do())

    def _scene_allows(self, scene_tags: Iterable[str]) -> bool:
        return bool(set(scene_tags or []) & self.trigger_cfg.allowed_scene_tags)

    def _intents_allow(self, feas: Optional[dict]) -> bool:
        if not isinstance(feas, dict):
            return False
        per = feas.get("per_intent") or []
        for it in per:
            if set((it or {}).get("tags", [])) & self.trigger_cfg.intent_markers:
                return True
        overall = feas.get("overall") or {}
        return bool(set(overall.get("tags", [])) & self.trigger_cfg.intent_markers)

    def _affinity_hit(self, active_types: Set[str], stimuli: Iterable[str]) -> Optional[str]:
        """Return one addiction type that matches present stimuli; thread-safe."""
        stim = set(stimuli or [])
        if not stim:
            return None
        for a_type in active_types:
            if stim & self.trigger_cfg.stimuli_affinity.get(a_type, set()):
                # de-dupe same stimulus burst per type
                key = "|".join(sorted(stim & self.trigger_cfg.stimuli_affinity.get(a_type, set())))
                h = hashlib.sha1(key.encode()).hexdigest()[:8]
                last = self._pg.seen_stimuli.get(a_type)
                if not last or last[0] != h or (time.time() - last[1]) > 120.0:
                    # Thread-safe update
                    async def _remember(a_type: str, h: str):
                        async with self._pg_lock:
                            self._pg.seen_stimuli[a_type] = (h, time.time())
                    asyncio.create_task(_remember(a_type, h))
                    return a_type
        return None

    def latch_effect_window(self):
        """Allow exactly one emission attempt on next effects call."""
        self._latched_window = True

    def decide_effect_tier(
        self,
        meta: Dict[str, Any],
        changed: bool,
        highest_level: int,
        active_types: Set[str]
    ) -> Optional[str]:
        """
        Decide None|'ambient'|'soft'|'major' given scene, intents, stimuli, cooldowns.
        Uses deterministic RNG with counter for reproducible but non-repeating behavior.
        """
        turn_idx = int(meta.get("turn_index", 0))
        now = time.time()
        scene_tags = ((meta.get("scene") or {}).get("tags")) or meta.get("scene_tags") or []
        feas = meta.get("feasibility")
        stimuli = set(meta.get("stimuli", []))  # e.g., {"sandals","cashier"} from orchestrator

        # explicit force respects global cooldowns
        if meta.get("addiction_force") is True and self._cooldowns_ok(turn_idx, now):
            return "major"

        relevant = self._scene_allows(scene_tags) or self._intents_allow(feas)

        if changed:
            # If a stat changed and we're in a relevant slice, escalate by severity
            if relevant and self._cooldowns_ok(turn_idx, now):
                return "major" if highest_level >= self.trigger_cfg.priority_level else "soft"
            # If not a relevant slice, fall back to ambient only if stimuli say so
            if self.trigger_cfg.ambient_in_neutral:
                hit = self._affinity_hit(active_types, stimuli)
                if hit and self._ambient_cooldowns_ok(turn_idx, now):
                    return "ambient"
            return None

        # No stat change:
        if relevant and self._cooldowns_ok(turn_idx, now):
            # soft craving in relevant scenes even without change, controlled by prob
            prob = self.trigger_cfg.ambient_prob_base if highest_level < self.trigger_cfg.priority_level else self.trigger_cfg.ambient_prob_lvl4
            r = _next_rand(self._pg, self.user_id, self.conversation_id, salt=f"tier_soft:{turn_idx}")
            return "soft" if r < prob else None

        # Neutral scene: allow ambient if a matching stimulus is present and ambient cooldown passes
        if self.trigger_cfg.ambient_in_neutral:
            hit = self._affinity_hit(active_types, stimuli)
            if hit and self._ambient_cooldowns_ok(turn_idx, now):
                # ambient probability scales with severity
                prob = self.trigger_cfg.ambient_prob_base + 0.05 * max(0, highest_level - 1)
                r = _next_rand(self._pg, self.user_id, self.conversation_id, salt=f"tier_ambient:{turn_idx}:{hit}")
                if r < min(prob, self.trigger_cfg.ambient_prob_lvl4):
                    return "ambient"

        # one-shot latch (e.g., something upstream decided we should hint once)
        if self._latched_window and self._cooldowns_ok(turn_idx, now):
            return "soft"

        return None

    async def _handle_action_directive(self, directive):
        instruction = directive.get("instruction", "")
        meta = directive.get("meta") or {}
        # Make directive meta available to gate with stimuli support
        self.current_context = {
            "scene": {"tags": meta.get("scene_tags", [])},
            "scene_tags": meta.get("scene_tags", []),
            "stimuli": meta.get("stimuli", []),  # NEW: pass stimuli from directive
            "feasibility": meta.get("feasibility"),
            "turn_index": meta.get("turn_index", 0),
            "addiction_force": meta.get("addiction_force", False),
        }

        if "monitor addictions" in instruction.lower():
            # state only; no latch here; gate will decide later
            ctx_wrapper = RunContextWrapper(context=self)
            return await _check_addiction_levels_impl(ctx_wrapper, directive.get("player_name", "player"))

        if "apply addiction effect" in instruction.lower():
            # We do NOT force emission here. We respect the same unified gate.
            ctx_wrapper = RunContextWrapper(context=self)
            return await _update_addiction_level_impl(
                ctx_wrapper,
                directive.get("player_name", "player"),
                directive.get("addiction_type"),
                progression_multiplier=directive.get("multiplier", 1.0),
                target_npc_id=directive.get("target_npc_id")
            )

        return {"status": "unknown_directive", "instruction": instruction}

    async def _handle_prohibition_directive(self, directive):
        prohibited = directive.get("prohibited_actions", [])
        self.prohibited_addictions = set(prohibited)
        return {"status": "prohibition_registered", "prohibited": prohibited}

################################################################################
# Database Helper Functions
################################################################################

async def ensure_addiction_table_exists(context: AddictionContext, conn):
    """Ensure the PlayerAddictions table exists."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS PlayerAddictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            player_name VARCHAR(255) NOT NULL,
            addiction_type VARCHAR(50) NOT NULL,
            level INTEGER NOT NULL DEFAULT 0,
            target_npc_id INTEGER NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, conversation_id, player_name, addiction_type, target_npc_id)
        )
    """)

async def find_or_create_addiction(
    context: AddictionContext, 
    conn, 
    player_name: str, 
    addiction_type: str, 
    level: int, 
    target_npc_id: Optional[int] = None
) -> int:
    """Find or create an addiction entry."""
    insert_stmt = """
        INSERT INTO PlayerAddictions
        (user_id, conversation_id, player_name, addiction_type, level, target_npc_id, last_updated)
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
        ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
        DO UPDATE SET level=EXCLUDED.level, last_updated=NOW()
        RETURNING id
    """
    
    addiction_id = await conn.fetchval(
        insert_stmt,
        context.user_id, context.conversation_id, player_name, addiction_type,
        level, target_npc_id if target_npc_id is not None else None
    )
    
    return addiction_id

################################################################################
# IMPLEMENTATION FUNCTIONS (not decorated, for internal use)
################################################################################

async def _check_addiction_levels_impl(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str
) -> Dict[str, Any]:
    """Internal implementation of check_addiction_levels"""
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    try:
        async with get_db_connection_context() as conn:
            await ensure_addiction_table_exists(ctx.context, conn)
            
            rows = await conn.fetch(
                "SELECT addiction_type, level, target_npc_id FROM PlayerAddictions WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3",
                user_id, conversation_id, player_name
            )
            
            addiction_data = {}
            npc_specific = []
            
            for row in rows:
                addiction_type, level, target_npc_id = row
                if target_npc_id is None:
                    addiction_data[addiction_type] = level
                else:
                    npc_row = await conn.fetchrow(
                        "SELECT npc_name FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3",
                        user_id, conversation_id, target_npc_id
                    )
                    npc_name = npc_row["npc_name"] if npc_row and "npc_name" in npc_row else f"NPC#{target_npc_id}"
                    npc_specific.append({
                        "addiction_type": addiction_type,
                        "level": level,
                        "npc_id": target_npc_id,
                        "npc_name": npc_name
                    })
                    
        has_addictions = any(lvl > 0 for lvl in addiction_data.values()) or bool(npc_specific)
        return {
            "addiction_levels": addiction_data,
            "npc_specific_addictions": npc_specific,
            "has_addictions": has_addictions
        }
    except Exception as e:
        logging.error(f"Error checking addiction levels: {e}")
        return {"error": str(e), "has_addictions": False}

async def _update_addiction_level_impl(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str,
    addiction_type: str,
    progression_chance: float = 0.2,
    progression_multiplier: float = 1.0,
    regression_chance: float = 0.1,
    target_npc_id: Optional[int] = None
) -> Dict[str, Any]:
    """Internal implementation of update_addiction_level"""
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id

    prohibited = getattr(ctx.context, "prohibited_addictions", set())
    if addiction_type in prohibited:
        return {
            "error": f"Addiction type '{addiction_type}' is prohibited by governance directive",
            "addiction_type": addiction_type,
            "prohibited": True
        }

    try:
        async with get_db_connection_context() as conn:
            await ensure_addiction_table_exists(ctx.context, conn)

            if target_npc_id is None:
                row = await conn.fetchrow("""
                    SELECT level FROM PlayerAddictions
                    WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
                    AND addiction_type=$4 AND target_npc_id IS NULL
                """, user_id, conversation_id, player_name, addiction_type)
            else:
                row = await conn.fetchrow("""
                    SELECT level FROM PlayerAddictions
                    WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
                    AND addiction_type=$4 AND target_npc_id=$5
                """, user_id, conversation_id, player_name, addiction_type, target_npc_id)

            current_level = row["level"] if row else 0
            prev_level = current_level
            
            # Use deterministic RNG with counter for consistent but non-repeating behavior
            turn_idx = ctx.context.current_context.get("turn_index", 0)
            roll = _next_rand(ctx.context._pg, user_id, conversation_id, 
                            salt=f"update:{player_name}:{addiction_type}:{turn_idx}:{target_npc_id}")

            # Dynamic progression regression handling
            if roll < (progression_chance * progression_multiplier) and current_level < 4:
                current_level += 1
                logging.info(f"Addiction ({addiction_type}) progressed: {prev_level} → {current_level}")
            elif roll > (1 - regression_chance) and current_level > 0:
                current_level -= 1
                logging.info(f"Addiction ({addiction_type}) regressed: {prev_level} → {current_level}")

            # Use helper function to update addiction
            addiction_id = await find_or_create_addiction(
                ctx.context, conn, player_name, addiction_type, current_level, target_npc_id
            )

        # If addiction reached level 4, update player stats through LoreSystem
        if current_level == 4:
            result = await ctx.context.lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="PlayerStats",
                entity_identifier={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "player_name": player_name
                },
                updates={"willpower": "GREATEST(willpower - 5, 0)"},
                reason=f"Extreme addiction to {addiction_type} affecting willpower"
            )

        return {
            "addiction_type": addiction_type,
            "previous_level": prev_level,
            "new_level": current_level,
            "level_name": ADDICTION_LEVELS.get(current_level, "Unknown"),
            "progressed": current_level > prev_level,
            "regressed": current_level < prev_level,
            "target_npc_id": target_npc_id
        }
    except Exception as e:
        logging.error(f"Error updating addiction: {e}")
        return {"error": str(e)}

async def _generate_addiction_effects_impl(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str,
    addiction_status: AddictionStatus
) -> Dict[str, Any]:
    """Internal implementation of generate_addiction_effects with tiered output"""
    meta = getattr(ctx.context, "current_context", {}) or {}
    tier = (meta.get("addiction_effect_tier") or "").lower()
    turn_idx = int(meta.get("turn_index", 0))

    # If no tier was decided, let the context make a last-second decision with no-change assumption
    if not tier:
        highest = max(list(addiction_status.addiction_levels.values()) + [0])
        active_types = set(a for a, lvl in addiction_status.addiction_levels.items() if lvl > 0)
        tier = ctx.context.decide_effect_tier(meta, changed=False, highest_level=highest, active_types=active_types) or ""

    if not tier:
        return {"effects": [], "has_effects": False}

    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    thematic = await ctx.context.get_thematic_messages(require_generation=True)
    if thematic is None:
        return {"effects": [], "has_effects": False}

    # Prioritize addictions by level and stimulus relevance
    stim = set((meta.get("stimuli") or []))
    def _priority_key(item):
        a_type, lvl = item
        stim_hit = bool(stim & ctx.context.trigger_cfg.stimuli_affinity.get(a_type, set()))
        return (lvl, stim_hit)

    addiction_levels = addiction_status.addiction_levels
    ordered = sorted(
        ((a, lvl) for a, lvl in addiction_levels.items() if lvl > 0),
        key=_priority_key, reverse=True
    )

    effects = []
    for a_type, lvl in ordered:
        effects.extend(thematic.get_levels(a_type, lvl))

    npc_specific = addiction_status.npc_specific_addictions
    
    async with get_db_connection_context() as conn:
        for entry in npc_specific:
            addiction_type = entry["addiction_type"]
            npc_name = entry.get("npc_name", f"NPC#{entry['npc_id']}")
            level = entry["level"]
            if level >= 3:
                effects.append(f"You have a {ADDICTION_LEVELS[level]} addiction to {npc_name}'s {addiction_type}.")
                # Only generate special events for major tier
                if level >= 4 and tier == "major":
                    try:
                        npc_data = await conn.fetchrow("""
                            SELECT npc_name, archetype_summary, personality_traits, dominance, cruelty
                            FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """, user_id, conversation_id, entry["npc_id"])
                        if npc_data:
                            # Convert asyncpg.Record to dict for safe access
                            rec = dict(npc_data)
                            archetype = rec.get("archetype_summary", "Unknown")
                            traits = rec.get("personality_traits") or []
                            dom = rec.get("dominance", 50)
                            cru = rec.get("cruelty", 50)
                            
                            prompt = (
                                f"Generate a 2-3 paragraph intense narrative scene about the player's extreme addiction "
                                f"to {npc_name}'s {addiction_type}. This is for a femdom roleplaying game.\n\n"
                                f"NPC Details:\n"
                                f"- Name: {npc_name}\n"
                                f"- Archetype: {archetype}\n"
                                f"- Dominance: {dom}/100\n"
                                f"- Cruelty: {cru}/100\n"
                                f"- Personality: {', '.join(traits[:3]) if traits else 'Unknown'}\n\n"
                                "Write an intense, immersive scene that shows how this addiction is affecting the player."
                            )
                            result = await llm_gateway.execute(
                                LLMRequest(
                                    agent=special_event_agent,
                                    prompt=prompt,
                                    context=ctx.context,
                                )
                            )
                            # Safer result extraction
                            special_event = result.text or None
                            if not special_event and result.raw is not None:
                                raw = result.raw
                                if hasattr(raw, "final_output"):
                                    special_event = raw.final_output
                                elif hasattr(raw, "output_text"):
                                    special_event = raw.output_text
                                else:
                                    special_event = str(raw)
                    
                            if special_event:
                                effects.append(special_event)
                    except Exception as e:
                        logging.error(f"Error generating special event: {e}")
    
    # Dedupe effects before pruning
    effects = list(dict.fromkeys(effects))
    
    # Prune by tier:
    # - ambient: exactly 1 short line (pick the strongest active addiction)
    # - soft: up to 2 lines; no special event
    # - major: keep as-is (including special event at lv4)
    if tier == "ambient":
        # Filter to short, single-line effects
        effects = [e for e in effects if len(e) <= 200 and e.count('\n') <= 1][:1]
        ctx.context._mark_emit(turn_idx, "ambient")
    elif tier == "soft":
        # Filter to medium, mostly single-line effects
        effects = [e for e in effects if len(e) <= 280 and e.count('\n') <= 1][:2]
        ctx.context._mark_emit(turn_idx, "soft")
    else:
        # major - no filtering needed
        ctx.context._mark_emit(turn_idx, "major")

    # consume one-shot latch
    ctx.context._latched_window = False
    
    return {"effects": effects, "has_effects": bool(effects)}

################################################################################
# DECORATED TOOL FUNCTIONS (for agent framework use)
################################################################################

@function_tool
@with_governance(
    agent_type=AgentType.UNIVERSAL_UPDATER,
    action_type="view_addictions",
    action_description="Checking addiction levels for {player_name}",
    id_from_context=lambda ctx: "addiction_system"
)
async def check_addiction_levels(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str
) -> Dict[str, Any]:
    """Check all addiction levels for a player (decorated tool version)"""
    return await _check_addiction_levels_impl(ctx, player_name)

@function_tool
@with_governance(
    agent_type=AgentType.UNIVERSAL_UPDATER,
    action_type="update_addiction",
    action_description="Updating addiction level for {player_name}: {addiction_type}",
    id_from_context=lambda ctx: "addiction_system"
)
async def update_addiction_level(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str,
    addiction_type: str,
    progression_chance: float = 0.2,
    progression_multiplier: float = 1.0,
    regression_chance: float = 0.1,
    target_npc_id: Optional[int] = None
) -> Dict[str, Any]:
    """Update addiction level for a player (decorated tool version)"""
    return await _update_addiction_level_impl(
        ctx, player_name, addiction_type,
        progression_chance, progression_multiplier,
        regression_chance, target_npc_id
    )

@function_tool(strict_mode=False)
@with_governance(
    agent_type=AgentType.UNIVERSAL_UPDATER,
    action_type="generate_effects",
    action_description="Generating narrative effects for {player_name}'s addictions",
    id_from_context=lambda ctx: "addiction_system"
)
async def generate_addiction_effects(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str,
    addiction_status: AddictionStatus
) -> Dict[str, Any]:
    """Generate narrative effects for addictions (decorated tool version)"""
    return await _generate_addiction_effects_impl(ctx, player_name, addiction_status)

################################################################################
# Guardrail Functions
################################################################################

async def addiction_content_safety(ctx, agent, input_data):
    content_moderator = Agent(
        name="Addiction Content Moderator",
        instructions=(
            "You check if addiction content is appropriate for the game setting. "
            "Allow adult themes in a femdom context but flag anything that might be genuinely harmful "
            "or that trivializes real addiction issues in a way that's ethically problematic."
        ),
        output_type=AddictionSafety,
        model=OpenAIResponsesModel(model="gpt-5-nano", openai_client=get_openai_client()),
    )
    result = await llm_gateway.execute(
        LLMRequest(
            agent=content_moderator,
            prompt=input_data,
            context=ctx.context,
        )
    )
    raw_result = result.raw
    if raw_result is None:
        raise ValueError("Content moderator returned no result")
    final_output = raw_result.final_output_as(AddictionSafety)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

################################################################################
# AGENTS - Configuration-Friendly
################################################################################

def get_openai_client():
    """Get OpenAI client instance for agents"""
    from openai import AsyncOpenAI
    import os
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

special_event_agent = Agent[AddictionContext](
    name="Special Event Generator",
    instructions=(
        "You generate vivid, immersive narrative events for extreme addiction situations. "
        "Scenes should be immersive, impactful, psychologically realistic, and maintain a femdom theme. "
    ),
    model=OpenAIResponsesModel(model="gpt-5-nano", openai_client=get_openai_client()),
)

addiction_progression_agent = Agent[AddictionContext](
    name="Addiction Progression Agent",
    instructions=(
        "Analyze events and context to determine addiction changes. "
        "Handle progression, regression, speed, and thresholds. Respect directives."
    ),
    tools=[update_addiction_level],
    output_type=AddictionUpdate,
    model=OpenAIResponsesModel(model="gpt-5-nano", openai_client=get_openai_client()),
)

addiction_narrative_agent = Agent[AddictionContext](
    name="Addiction Narrative Agent",
    instructions=(
        "Generate narrative effects for addictions, varying with type/level. "
        "Incorporate femdom themes subtly."
    ),
    tools=[generate_addiction_effects],
    output_type=AddictionEffects,
    model=OpenAIResponsesModel(model="gpt-5-nano", openai_client=get_openai_client()),
)

addiction_system_agent = Agent[AddictionContext](
    name="Addiction System Agent",
    instructions=(
        "Central addiction management system for a femdom RPG. "
        "Tracks and manages player and NPC-specific addictions, progression, regression, effects, and special events."
        "Use subagents and always respect governance directives. "
    ),
    handoffs=[
        handoff(addiction_progression_agent, tool_name_override="manage_addiction_progression"),
        handoff(addiction_narrative_agent, tool_name_override="generate_narrative_effects"),
        handoff(special_event_agent, tool_name_override="create_special_event")
    ],
    tools=[
        check_addiction_levels,
        update_addiction_level,
        generate_addiction_effects
    ],
    input_guardrails=[
        InputGuardrail(guardrail_function=addiction_content_safety)
    ],
    model=OpenAIResponsesModel(model="gpt-5-nano", openai_client=get_openai_client()),
)

def get_thematic_message_agent():
    """
    Agent that generates level-tiered thematic messages per addiction type
    in a femdom RPG. 
    """
    return Agent(
        name="ThematicMessageAgent",
        instructions=(
            "You generate short narrative addiction progression blurbs for a femdom RPG.\n"
            "Input: JSON specifying addiction types.\n"
            "Output: JSON object matching ThematicMessagesBundle schema.\n\n"
            "Rules:\n"
            "- Provide EXACTLY 4 levels (1=mild tease, 4=intense submission) per addiction.\n"
            "- 1-2 sentences; <=160 characters preferred.\n"
            "- Tone: immersive, suggestive, explicit sexual detail.\n"
            "- Reference 'Mistress' or dominant presence sparingly; more intensity at higher levels.\n"
            "- Return ONLY JSON.\n"
        ),
        model=OpenAIResponsesModel(model="gpt-5-nano", openai_client=get_openai_client()),
        model_settings=ModelSettings(),
        output_type=ThematicMessagesBundle,
    )

async def generate_thematic_messages_via_agent(
    user_id: int,
    conversation_id: int,
    addiction_types: List[str] = ADDICTION_TYPES,
    governor=None,
    timeout: Optional[float] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Ask LLM to synthesize messages. Returns {addiction_type: {1:txt,...,4:txt}, ...}
    Falls back to _DEFAULT_THEMATIC_MESSAGES_MIN on failure.
    """
    if governor is None:
        from nyx.integrate import get_central_governance
        try:
            governor = await get_central_governance(user_id, conversation_id)
            perm = await governor.check_action_permission(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="addiction_thematic_generator",
                action_type="generate_thematic_messages",
                action_details={"addiction_types": addiction_types},
            )
            if not perm.get("approved", True):
                logging.warning("Governance denied thematic message generation; using min fallback.")
                return _DEFAULT_THEMATIC_MESSAGES_MIN
        except Exception as e:
            logging.warning(f"Governance check failed; continuing anyway: {e}")

    agent = get_thematic_message_agent()

    payload = {
        "addiction_types": addiction_types,
        "tone": "femdom",
        "max_length": 160,
    }

    run_ctx = RunContextWrapper(context={
        "user_id": user_id,
        "conversation_id": conversation_id,
        "purpose": "generate_thematic_messages",
    })

    try:
        request = LLMRequest(
            agent=agent,
            prompt=json.dumps(payload),
            context=run_ctx.context,
        )
        run_task = llm_gateway.execute(request)
        if timeout is not None:
            resp = await asyncio.wait_for(run_task, timeout=timeout)
        else:
            resp = await run_task
        raw_resp = resp.raw
        if raw_resp is None:
            raise ValueError("Thematic message agent returned no result")
        bundle = raw_resp.final_output_as(ThematicMessagesBundle)

        out: Dict[str, Dict[str, str]] = {}
        entries = bundle.addictions or []
        for entry in entries:
            out[entry.addiction_type] = {str(m.level): m.text for m in entry.messages}
        for t in addiction_types:
            out.setdefault(t, _DEFAULT_THEMATIC_MESSAGES_MIN[t])
            for lvl in ("1", "2", "3", "4"):
                out[t].setdefault(lvl, _MIN_FALLBACK_MSG)
        return out
    except asyncio.TimeoutError:
        timeout_desc = timeout if timeout is not None else "unknown"
        logging.error(
            f"Thematic message agent timed out after {timeout_desc} seconds; propagation to caller."
        )
        raise
    except Exception as e:
        logging.error(f"Thematic message generation failed: {e}")
        return _DEFAULT_THEMATIC_MESSAGES_MIN

################################################################################
# MAIN ENTRY / UTILITY FUNCTIONS (Extensible) - REFACTORED
################################################################################

def get_addiction_label(level: int) -> str:
    return ADDICTION_LEVELS.get(level, "Unknown")

async def process_addiction_update(
    user_id: int, conversation_id: int, player_name: str,
    addiction_type: str, progression_multiplier: float = 1.0, 
    target_npc_id: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process addiction update with smart tiered gating"""
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize(start_background=False)  # No background task for transient context
    
    # Set context from metadata if provided (from orchestrator/SDK)
    if metadata:
        addiction_context.current_context = {
            "scene": {"tags": metadata.get("scene_tags", [])},
            "scene_tags": metadata.get("scene_tags", []),
            "stimuli": metadata.get("stimuli", []),  # NEW: stimuli from scene
            "feasibility": metadata.get("feasibility"),
            "turn_index": metadata.get("turn_index", 0),
            "addiction_force": metadata.get("addiction_force", False),
        }
    
    with trace(
        workflow_name="Addiction System",
        trace_id=f"trace_addiction-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        ctx_wrapper = RunContextWrapper(context=addiction_context)
        
        # Update addiction level
        update_result = await _update_addiction_level_impl(
            ctx_wrapper, player_name, addiction_type,
            progression_multiplier=progression_multiplier,
            target_npc_id=target_npc_id
        )
        
        # Get current addiction state
        levels_dict = await _check_addiction_levels_impl(ctx_wrapper, player_name)
        addiction_levels = levels_dict.get("addiction_levels", {}) or {}
        active_types = {a for a, lvl in addiction_levels.items() if lvl > 0}
        highest = max(list(addiction_levels.values()) + [0])
        
        meta = addiction_context.current_context or {}
        changed = bool(update_result.get("progressed") or update_result.get("regressed"))
        
        # Decide effect tier based on context, stimuli, and changes
        tier = addiction_context.decide_effect_tier(
            meta=meta, changed=changed, highest_level=highest, active_types=active_types
        )
        
        if tier:
            addiction_context.latch_effect_window()  # consume on next effects call
            # Tag the chosen tier into meta so generator can prune appropriately
            addiction_context.current_context["addiction_effect_tier"] = tier
            addiction_status = AddictionStatus(
                addiction_levels=addiction_levels,
                npc_specific_addictions=levels_dict.get("npc_specific_addictions", []),
                has_addictions=levels_dict.get("has_addictions", False)
            )
            narrative_effects = await _generate_addiction_effects_impl(ctx_wrapper, player_name, addiction_status)
        else:
            narrative_effects = {"effects": [], "has_effects": False}
        
    return {
        "update": update_result, 
        "narrative_effects": narrative_effects, 
        "addiction_type": addiction_type, 
        "target_npc_id": target_npc_id,
        "meta": {
            "tier": tier,
            "changed": changed,
            "highest": highest,
            "stimuli": list(meta.get("stimuli", [])),
            "scene_tags": list(meta.get("scene_tags", [])),
        }
    }

async def process_addiction_effects(
    user_id: int, conversation_id: int, player_name: str, 
    addiction_status: dict,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process addiction effects with smart gating and telemetry"""
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize(start_background=False)
    
    # Set context from metadata if provided
    if metadata:
        addiction_context.current_context = {
            "scene": {"tags": metadata.get("scene_tags", [])},
            "scene_tags": metadata.get("scene_tags", []),
            "stimuli": metadata.get("stimuli", []),
            "feasibility": metadata.get("feasibility"),
            "turn_index": metadata.get("turn_index", 0),
            "addiction_force": metadata.get("addiction_force", False),
        }
    
    addiction_status_obj = AddictionStatus(
        addiction_levels=addiction_status.get("addiction_levels", {}),
        npc_specific_addictions=addiction_status.get("npc_specific_addictions", []),
        has_addictions=addiction_status.get("has_addictions", False)
    )
    effects_result = await _generate_addiction_effects_impl(
        RunContextWrapper(context=addiction_context), player_name, addiction_status_obj
    )
    
    # Mirror telemetry like process_addiction_update
    highest = max(list(addiction_status_obj.addiction_levels.values()) + [0])
    decided_tier = addiction_context.current_context.get("addiction_effect_tier", "")
    
    return {
        **effects_result,
        "meta": {
            "tier": decided_tier,
            "highest": highest,
            "stimuli": addiction_context.current_context.get("stimuli", []),
            "scene_tags": addiction_context.current_context.get("scene_tags", []),
        }
    }

async def check_addiction_status(
    user_id: int, conversation_id: int, player_name: str
) -> Dict[str, Any]:
    """Check addiction status with effects generation"""
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize(start_background=False)
    
    with trace(
        workflow_name="Addiction System",
        trace_id=f"trace_addiction-status-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        ctx_wrapper = RunContextWrapper(context=addiction_context)
        levels_result = await _check_addiction_levels_impl(ctx_wrapper, player_name)
        effects_result = {"effects": [], "has_effects": False}
        if levels_result.get("has_addictions", False):
            addiction_status = AddictionStatus(
                addiction_levels=levels_result.get("addiction_levels", {}),
                npc_specific_addictions=levels_result.get("npc_specific_addictions", []),
                has_addictions=levels_result.get("has_addictions", False)
            )
            effects_result = await _generate_addiction_effects_impl(ctx_wrapper, player_name, addiction_status)
    
    return {"status": levels_result, "effects": effects_result}

async def get_addiction_status(
    user_id: int, conversation_id: int, player_name: str
) -> Dict[str, Any]:
    """Get addiction status without triggering effects"""
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize(start_background=False)
    
    ctx_wrapper = RunContextWrapper(context=addiction_context)
    levels_result = await _check_addiction_levels_impl(ctx_wrapper, player_name)
    
    result = {"has_addictions": levels_result.get("has_addictions", False), "addictions": {}}
    for addiction_type, level in levels_result.get("addiction_levels", {}).items():
        if level > 0:
            result["addictions"][addiction_type] = {"level": level, "label": get_addiction_label(level), "type": "general"}
    for addiction in levels_result.get("npc_specific_addictions", []):
        addiction_type = addiction.get("addiction_type")
        npc_id = addiction.get("npc_id")
        npc_name = addiction.get("npc_name", f"NPC#{npc_id}")
        level = addiction.get("level", 0)
        if level > 0:
            key = f"{addiction_type}_{npc_id}"
            result["addictions"][key] = {
                "level": level,
                "label": get_addiction_label(level),
                "type": "npc_specific",
                "npc_id": npc_id,
                "npc_name": npc_name,
                "addiction_type": addiction_type
            }
    return result

async def register_with_governance(user_id: int, conversation_id: int):
    """Register addiction system with governance (for long-lived contexts)"""
    from nyx.integrate import get_central_governance
    governor = await get_central_governance(user_id, conversation_id)
    await governor.register_agent(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_instance=addiction_system_agent,
        agent_id="addiction_system"
    )
    await governor.issue_directive(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="addiction_system",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Monitor player addictions and apply appropriate effects",
            "scope": "game"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60
    )
    logging.info("Addiction system registered with Nyx governance")

async def process_addiction_directive(directive_data: Dict[str, Any], user_id: int, conversation_id: int) -> Dict[str, Any]:
    """Process directives from governance"""
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize(start_background=False)
    if not addiction_context.directive_handler:
        addiction_context.directive_handler = DirectiveHandler(
            user_id, conversation_id, AgentType.UNIVERSAL_UPDATER, "addiction_system"
        )
    # Unified action for both types
    if directive_data.get("type") == "prohibition" or directive_data.get("directive_type") == DirectiveType.PROHIBITION:
        return await addiction_context._handle_prohibition_directive(directive_data)
    return await addiction_context._handle_action_directive(directive_data)

# Export implementation functions for external use if needed
check_addiction_levels_impl = _check_addiction_levels_impl
update_addiction_level_impl = _update_addiction_level_impl
generate_addiction_effects_impl = _generate_addiction_effects_impl

# Export gate management for SDK cleanup
__all__ = [
    'process_addiction_update',
    'process_addiction_effects', 
    'check_addiction_status',
    'get_addiction_status',
    'register_with_governance',
    'process_addiction_directive',
    'purge_gate_state',
    '_GATE_STATE',
    '_GATE_LOCKS',
    '_GATE_TOUCH'
]
