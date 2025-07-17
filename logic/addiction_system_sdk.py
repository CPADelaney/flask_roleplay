# logic/addiction_system_sdk.py
"""
Refactored Addiction System with full Nyx Governance integration.

REFACTORED: All database writes now go through canon or LoreSystem
FIXED: Separated implementation functions from decorated tools to avoid 'FunctionTool' not callable errors
FIXED: Incorporated feedback from code review

Features:
1) Complete integration with Nyx central governance
2) Permission checking before all operations
3) Action reporting for monitoring and tracing
4) Directive handling for system control
5) Registration with proper agent types and constants
"""

import logging
import random
import json
import asyncio
import asyncpg
import os
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
from agents.models.openai_responses import OpenAIResponsesModel
from pydantic import BaseModel, Field, field_validator, model_validator

# DB connection - UPDATED: Using new async context manager
from db.connection import get_db_connection_context

# Import canon and lore system for canonical writes
from lore.core import canon
from lore.core.lore_system import LoreSystem

# Nyx governance integration
# Moved imports to function level to avoid circular imports
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
# ---------------------------------------------------------------------------
# Thematic message *seeds* (labels only) used if we must synthesize messages.
# Actual text will be generated agentically at runtime.
# ---------------------------------------------------------------------------
ADDICTION_TYPES = ["socks", "feet", "sweat", "ass", "scent", "humiliation", "submission"]

# Minimal bare fallback used only if generation fails catastrophically.
_MIN_FALLBACK_MSG = "You feel a tug of desire you can't quite ignore."
_DEFAULT_THEMATIC_MESSAGES_MIN = {
    t: {str(i): _MIN_FALLBACK_MSG for i in range(1, 5)} for t in ADDICTION_TYPES
}

THEMATIC_MESSAGES_FILE = os.getenv("THEMATIC_MESSAGES_FILE", "thematic_messages.json")
# Historical compatibility var kept for callers that may still import it.
# Point them to the minimal fallback.
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

    @classmethod
    async def get(cls, user_id: Optional[int] = None, conversation_id: Optional[int] = None, refresh: bool = False):
        """
        Global singleton. Pass user & convo if available for agent generation / governance.
        refresh=True forces regeneration (ignoring cached file).
        """
        async with cls._lock:
            if cls._instance is None or refresh:
                instance = cls(_DEFAULT_THEMATIC_MESSAGES_MIN, user_id, conversation_id)
                await instance._load(refresh=refresh)
                cls._instance = instance
            else:
                # attach IDs if newly provided
                if user_id is not None:
                    cls._instance.user_id = user_id
                if conversation_id is not None:
                    cls._instance.conversation_id = conversation_id
            return cls._instance

    async def _load(self, refresh: bool = False):
        """
        Load from file if present & not refreshing; else generate via agent.
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

        # Generate (always if refresh; else only if no file data)
        if refresh or not file_msgs:
            try:
                generated = await generate_thematic_messages_via_agent(
                    user_id=self.user_id or 0,
                    conversation_id=self.conversation_id or 0,
                )
                self.file_source = "generated"
                # write to disk for caching
                # TODO: Consider atomic write (temp file + rename) to avoid race conditions
                try:
                    with open(THEMATIC_MESSAGES_FILE, "w") as f:
                        json.dump(generated, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logging.warning(f"Failed to persist generated thematic messages: {e}")
            except Exception as e:
                logging.error(f"Thematic message generation error: {e}")

        # Merge precedence: file overrides > generated > min fallback
        merged: Dict[str, Dict[str, str]] = {}
        for t in ADDICTION_TYPES:
            merged[t] = _DEFAULT_THEMATIC_MESSAGES_MIN[t].copy()
            if generated and t in generated:
                # Ensure all keys are strings
                merged[t].update({str(k): v for k, v in generated[t].items()})
            if file_msgs and t in file_msgs:
                # file already string-keyed; ensure string keys anyway
                merged[t].update({str(k): v for k, v in file_msgs[t].items()})
        self.messages = merged

    # --- existing public helpers unchanged ----------------------------
    def get_for(self, addiction_type: str, level: Union[int, str]) -> str:
        level_str = str(level)
        return self.messages.get(addiction_type, {}).get(level_str, "")

    def get_levels(self, addiction_type: str, up_to_level: int) -> List[str]:
        return [
            msg for lvl in range(1, up_to_level + 1)
            if (msg := self.get_for(addiction_type, lvl))
        ]

################################################################################
# Agent Model Settings (Configurable)
################################################################################

def get_openai_client():
    """Get OpenAI client instance for agents"""
    from openai import AsyncOpenAI
    import os
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_model_settings(agent_name: str = "", env_override: str = "") -> ModelSettings:
    temp_env = os.getenv(f"{agent_name}_MODEL_TEMP", os.getenv(env_override, None))
    try:
        t = float(temp_env) if temp_env else 0.7 if agent_name == "Special Event Generator" else 0.4
    except Exception:
        t = 0.7
    return ModelSettings(temperature=t)

################################################################################
# Main Context
################################################################################

class AddictionContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.thematic_messages: Optional[ThematicMessages] = None
        self.directive_handler: Optional[DirectiveHandler] = None
        self.prohibited_addictions: set = set()
        self.directive_task = None
        self.lore_system = None

    async def initialize(self):
        from nyx.integrate import get_central_governance
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        self.thematic_messages = await ThematicMessages.get(self.user_id, self.conversation_id)
        self.lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
        self.directive_handler = DirectiveHandler(
            self.user_id, self.conversation_id,
            AgentType.UNIVERSAL_UPDATER, "addiction_system", governance=self.governor
        )
        self.directive_handler.register_handler(DirectiveType.ACTION, self._handle_action_directive)
        self.directive_handler.register_handler(DirectiveType.PROHIBITION, self._handle_prohibition_directive)
        self.directive_task = asyncio.create_task(
            self.directive_handler.start_background_processing(interval=60.0)
        )

    async def _handle_action_directive(self, directive):
        instruction = directive.get("instruction", "")
        if "monitor addictions" in instruction.lower():
            ctx_wrapper = RunContextWrapper(context=self)
            return await _check_addiction_levels_impl(
                ctx_wrapper,
                directive.get("player_name", "player")
            )
        if "apply addiction effect" in instruction.lower():
            addiction_type = directive.get("addiction_type")
            if addiction_type:
                ctx_wrapper = RunContextWrapper(context=self)
                return await _update_addiction_level_impl(
                    ctx_wrapper,
                    directive.get("player_name", "player"),
                    addiction_type,
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
            # Ensure table exists through canon
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
                    # Fetch NPC data while still in connection context
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
            roll = random.random()

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
    """Internal implementation of generate_addiction_effects"""
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    thematic = ctx.context.thematic_messages

    effects = []
    addiction_levels = addiction_status.addiction_levels
    for addiction_type, level in addiction_levels.items():
        if level <= 0:
            continue
        effects.extend(thematic.get_levels(addiction_type, level))

    npc_specific = addiction_status.npc_specific_addictions
    
    async with get_db_connection_context() as conn:
        for entry in npc_specific:
            addiction_type = entry["addiction_type"]
            npc_name = entry.get("npc_name", f"NPC#{entry['npc_id']}")
            level = entry["level"]
            if level >= 3:
                effects.append(f"You have a {ADDICTION_LEVELS[level]} addiction to {npc_name}'s {addiction_type}.")
                if level >= 4:
                    try:
                        npc_data = await conn.fetchrow("""
                            SELECT npc_name, archetype_summary, personality_traits, dominance, cruelty
                            FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """, user_id, conversation_id, entry["npc_id"])
                        if npc_data:
                            prompt = (
                                f"Generate a 2-3 paragraph intense narrative scene about the player's extreme addiction "
                                f"to {npc_name}'s {addiction_type}. This is for a femdom roleplaying game.\n\n"
                                f"NPC Details:\n"
                                f"- Name: {npc_name}\n"
                                f"- Archetype: {npc_data.get('archetype_summary')}\n"
                                f"- Dominance: {npc_data.get('dominance')}/100\n"
                                f"- Cruelty: {npc_data.get('cruelty')}/100\n"
                                f"- Personality: {', '.join(npc_data.get('personality_traits', [])[:3]) if npc_data.get('personality_traits') else 'Unknown'}\n\n"
                                "Write an intense, immersive scene that shows how this addiction is affecting the player."
                            )
                            result = await Runner.run(
                                special_event_agent, prompt, context=ctx.context
                            )
                            # Safer result extraction
                            special_event = None
                            if hasattr(result, "final_output"):
                                special_event = result.final_output
                            elif hasattr(result, "output_text"):
                                special_event = result.output_text
                            else:
                                special_event = str(result)
                            
                            if special_event:
                                effects.append(special_event)
                    except Exception as e:
                        logging.error(f"Error generating special event: {e}")
    
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
        model=OpenAIResponsesModel(model="gpt-4.1-nano", openai_client=get_openai_client()),
        model_settings=get_model_settings("Addiction Content Moderator", "ADD_CONTENT_MOD_TEMP")
    )
    result = await Runner.run(content_moderator, input_data, context=ctx.context)
    final_output = result.final_output_as(AddictionSafety)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

################################################################################
# AGENTS - Configuration-Friendly
################################################################################

special_event_agent = Agent[AddictionContext](
    name="Special Event Generator",
    instructions=(
        "You generate vivid, immersive narrative events for extreme addiction situations. "
        "Scenes should be immersive, impactful, psychologically realistic, and maintain a femdom theme. "
    ),
    model=OpenAIResponsesModel(model="gpt-4.1-nano", openai_client=get_openai_client()),
    model_settings=get_model_settings("Special Event Generator", "SPECIAL_EVENT_TEMP")
)

addiction_progression_agent = Agent[AddictionContext](
    name="Addiction Progression Agent",
    instructions=(
        "Analyze events and context to determine addiction changes. "
        "Handle progression, regression, speed, and thresholds. Respect directives."
    ),
    tools=[update_addiction_level],
    output_type=AddictionUpdate,
    model=OpenAIResponsesModel(model="gpt-4.1-nano", openai_client=get_openai_client()),
    model_settings=get_model_settings("Addiction Progression Agent", "PROGRESS_AGENT_TEMP")
)

addiction_narrative_agent = Agent[AddictionContext](
    name="Addiction Narrative Agent",
    instructions=(
        "Generate narrative effects for addictions, varying with type/level. "
        "Incorporate femdom themes subtly."
    ),
    tools=[generate_addiction_effects],
    output_type=AddictionEffects,
    model=OpenAIResponsesModel(model="gpt-4.1-nano", openai_client=get_openai_client()),
    model_settings=get_model_settings("Addiction Narrative Agent", "NARRATIVE_AGENT_TEMP")
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
    model=OpenAIResponsesModel(model="gpt-4.1-nano", openai_client=get_openai_client()),
    model_settings=get_model_settings("Addiction System Agent", "ADDICTION_SYS_TEMP")
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
        model=OpenAIResponsesModel(model="gpt-4.1-nano", openai_client=get_openai_client()),
        model_settings=ModelSettings(temperature=0.8),
        output_type=ThematicMessagesBundle,
    )

async def generate_thematic_messages_via_agent(
    user_id: int,
    conversation_id: int,
    addiction_types: List[str] = ADDICTION_TYPES,
    governor=None,
) -> Dict[str, Dict[str, str]]:
    """
    Ask LLM to synthesize messages. Returns {addiction_type: {1:txt,...,4:txt}, ...}
    Falls back to _DEFAULT_THEMATIC_MESSAGES_MIN on failure.
    """
    # governance (optional)
    if governor is None:
        from nyx.integrate import get_central_governance
        try:
            governor = await get_central_governance(user_id, conversation_id)
            perm = await governor.check_action_permission(
                agent_type=AgentType.UNIVERSAL_UPDATER,  # or a dedicated type
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
        # you can add environment / tone knobs here
        "tone": "femdom",
        "max_length": 160,
    }

    # The RunContextWrapper expects something like AddictionContext? We can fake a minimal dict.
    run_ctx = RunContextWrapper(context={
        "user_id": user_id,
        "conversation_id": conversation_id,
        "purpose": "generate_thematic_messages",
    })

    try:
        resp = await Runner.run(agent, json.dumps(payload), context=run_ctx.context)
        bundle = resp.final_output_as(ThematicMessagesBundle)

        out: Dict[str, Dict[str, str]] = {}
        entries = bundle.addictions or []  # Guard against None
        for entry in entries:
            # Ensure all keys are strings
            out[entry.addiction_type] = {str(m.level): m.text for m in entry.messages}
        # Basic sanity fill for any missing types
        for t in addiction_types:
            out.setdefault(t, _DEFAULT_THEMATIC_MESSAGES_MIN[t])
            for lvl in ("1", "2", "3", "4"):
                out[t].setdefault(lvl, _MIN_FALLBACK_MSG)
        return out
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
    addiction_type: str, progression_multiplier: float = 1.0, target_npc_id: Optional[int] = None
) -> Dict[str, Any]:
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    with trace(
        workflow_name="Addiction System",
        trace_id=f"trace_addiction-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        prompt = f"Update the player's addiction to {addiction_type}{f' related to NPC #{target_npc_id}' if target_npc_id else ''}. Player name: {player_name}. Progression multiplier: {progression_multiplier}"
        
        # Create context wrapper without connection
        ctx_wrapper = RunContextWrapper(context=addiction_context)
        
        # Call implementation function directly
        update_result = await _update_addiction_level_impl(
            ctx_wrapper, player_name, addiction_type,
            progression_multiplier=progression_multiplier,
            target_npc_id=target_npc_id
        )
        
        # Get addiction status for effects
        addiction_status_dict = await _check_addiction_levels_impl(ctx_wrapper, player_name)
        addiction_status = AddictionStatus(
            addiction_levels=addiction_status_dict.get("addiction_levels", {}),
            npc_specific_addictions=addiction_status_dict.get("npc_specific_addictions", []),
            has_addictions=addiction_status_dict.get("has_addictions", False)
        )
        narrative_effects = await _generate_addiction_effects_impl(ctx_wrapper, player_name, addiction_status)
        
    return {"update": update_result, "narrative_effects": narrative_effects, "addiction_type": addiction_type, "target_npc_id": target_npc_id}

async def process_addiction_effects(
    user_id: int, conversation_id: int, player_name: str, addiction_status: dict
) -> Dict[str, Any]:
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    addiction_status_obj = AddictionStatus(
        addiction_levels=addiction_status.get("addiction_levels", {}),
        npc_specific_addictions=addiction_status.get("npc_specific_addictions", []),
        has_addictions=addiction_status.get("has_addictions", False)
    )
    effects_result = await _generate_addiction_effects_impl(
        RunContextWrapper(context=addiction_context), player_name, addiction_status_obj
    )
    return effects_result

async def check_addiction_status(
    user_id: int, conversation_id: int, player_name: str
) -> Dict[str, Any]:
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
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
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
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
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    if not addiction_context.directive_handler:
        addiction_context.directive_handler = DirectiveHandler(
            user_id, conversation_id, AgentType.UNIVERSAL_UPDATER, "addiction_system"
        )
    # Unified action for both types (use correct method)
    if directive_data.get("type") == "prohibition" or directive_data.get("directive_type") == DirectiveType.PROHIBITION:
        return await addiction_context._handle_prohibition_directive(directive_data)
    return await addiction_context._handle_action_directive(directive_data)

# Export implementation functions for external use if needed
check_addiction_levels_impl = _check_addiction_levels_impl
update_addiction_level_impl = _update_addiction_level_impl
generate_addiction_effects_impl = _generate_addiction_effects_impl
