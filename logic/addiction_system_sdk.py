# logic/addiction_system_sdk.py
"""
Refactored Addiction System with full Nyx Governance integration.

REFACTORED: All database writes now go through canon or LoreSystem

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
from pydantic import BaseModel, Field

# DB connection - UPDATED: Using new async context manager
from db.connection import get_db_connection_context

# Import canon and lore system for canonical writes
from lore.core import canon
from lore.lore_system import LoreSystem

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
DEFAULT_THEMATIC_MESSAGES = {
    "socks": {
        "1": "You occasionally steal glances at sumptuous stockings.",
        "2": "A subtle craving for the delicate feel of silk emerges within you.",
        "3": "The allure of sensuous socks overwhelms your thoughts.",
        "4": "Under your Mistress's commanding presence, your obsession with exquisite socks leaves you trembling in servile adoration."
    },
    "feet": {
        "1": "Your eyes frequently wander to the graceful arch of bare feet.",
        "2": "A surge of forbidden excitement courses through you at the mere glimpse of uncovered toes.",
        "3": "Distracted by the sight of enticing feet, you find it difficult to focus on anything else.",
        "4": "In the presence of your dominant Mistress, your fixation on every tantalizing curve of feet renders you utterly submissive."
    },
    "sweat": {
        "1": "The scent of perspiration begins to evoke an unspoken thrill within you.",
        "2": "Each drop of sweat stokes a simmering desire you dare not fully acknowledge.",
        "3": "Your senses heighten as the aroma of exertion casts a spell over your inhibitions.",
        "4": "Overwhelmed by the intoxicating allure of sweat, you are compelled to seek it out under your Mistress's relentless command."
    },
    "ass": {
        "1": "Your gaze lingers a little longer on the curves of a well-shaped rear.",
        "2": "A subtle, forbidden thrill courses through you at the sight of a pert backside.",
        "3": "You find yourself fixated on every tantalizing detail of exposed derrieres, your mind wandering into submissive fantasies.",
        "4": "Under your Mistress's unwavering control, your obsession with perfectly sculpted rear ends drives you to desperate submission."
    },
    "scent": {
        "1": "You become acutely aware of natural pheromones and subtle scents around you.",
        "2": "Every hint of an enticing aroma sends a shiver down your spine, awakening deep desires.",
        "3": "You begin to collect memories of scents, each evoking a surge of submissive longing.",
        "4": "In the grip of your extreme addiction, the mere whiff of a scent under your Mistress's watchful eye reduces you to euphoric submission."
    },
    "humiliation": {
        "1": "The sting of humiliation sparks a curious thrill in your submissive heart.",
        "2": "You find yourself yearning for more degrading scenarios as your pride withers under each slight.",
        "3": "Every act of public embarrassment intensifies your craving to be dominated and humiliated.",
        "4": "In the presence of your ruthless Mistress, the exquisite agony of humiliation consumes you, binding your will entirely to her desires."
    },
    "submission": {
        "1": "The taste of obedience becomes subtly intoxicating as you seek her approval in every glance.",
        "2": "Your need to surrender grows, craving the reassurance that only your Mistress can provide.",
        "3": "In every command, you find a deeper satisfaction in your subjugated state, yearning to be molded by her hand.",
        "4": "Your identity dissolves in the overwhelming tide of submission, as your Mistress's word becomes the sole law governing your existence."
    }
}

THEMATIC_MESSAGES_FILE = os.getenv("THEMATIC_MESSAGES_FILE", "thematic_messages.json")
_DEFAULT_THEMATIC_MESSAGES = DEFAULT_THEMATIC_MESSAGES  # Store for later use

################################################################################
# Thematic Message Loader - Singleton, Async & Dynamic
################################################################################

class ThematicMessages:
    _instance: Optional["ThematicMessages"] = None
    _lock = asyncio.Lock()

    def __init__(self, fallback: dict):
        self.messages = fallback
        self.file_source = None

    @classmethod
    async def get(cls) -> "ThematicMessages":
        async with cls._lock:
            if cls._instance is None:
                instance = cls(_DEFAULT_THEMATIC_MESSAGES)
                await instance._load()
                cls._instance = instance
            return cls._instance

    async def _load(self):
        try:
            if os.path.exists(THEMATIC_MESSAGES_FILE):
                with open(THEMATIC_MESSAGES_FILE, "r") as f:
                    self.messages = json.load(f)
                    self.file_source = THEMATIC_MESSAGES_FILE
                logging.info(f"Thematic messages loaded from {THEMATIC_MESSAGES_FILE}")
            else:
                raise FileNotFoundError()
        except Exception:
            self.file_source = "default"
            logging.warning("Could not load external thematic messages, using defaults.")

    def get_for(self, addiction_type: str, level: Union[int, str]) -> str:
        level_str = str(level)
        return self.messages.get(addiction_type, {}).get(level_str, "")

    def get_levels(self, addiction_type: str, up_to_level: int) -> List[str]:
        """Get all non-empty messages for a type up to a given level."""
        return [
            msg for lvl in range(1, up_to_level + 1)
            if (msg := self.get_for(addiction_type, lvl))
        ]

################################################################################
# Agent Model Settings (Configurable)
################################################################################

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
        self.thematic_messages = await ThematicMessages.get()
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
            return await check_addiction_status(
                self.user_id, self.conversation_id, directive.get("player_name", "player")
            )
        if "apply addiction effect" in instruction.lower():
            addiction_type = directive.get("addiction_type")
            if addiction_type:
                return await update_addiction_level(
                    RunContextWrapper(self),
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
# Core Functions as Agent Tools (Governance-wrapped) - REFACTORED
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

@function_tool
@with_governance(
    agent_type=AgentType.UNIVERSAL_UPDATER,
    action_type="generate_effects",
    action_description="Generating narrative effects for {player_name}'s addictions",
    id_from_context=lambda ctx: "addiction_system"
)
async def generate_addiction_effects(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str,
    addiction_status: Dict[str, Any]
) -> Dict[str, Any]:
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    thematic = ctx.context.thematic_messages

    effects = []
    addiction_levels = addiction_status.get("addiction_levels", {})
    for addiction_type, level in addiction_levels.items():
        if level <= 0:
            continue
        effects.extend(thematic.get_levels(addiction_type, level))

    npc_specific = addiction_status.get("npc_specific_addictions", [])
    
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
                            special_event = result.final_output
                            if special_event:
                                effects.append(special_event)
                    except Exception as e:
                        logging.error(f"Error generating special event: {e}")
    
    return {"effects": effects, "has_effects": bool(effects)}
    
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
        "Avoid explicit content. Use second person."
    ),
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
    model_settings=get_model_settings("Addiction System Agent", "ADDICTION_SYS_TEMP")
)

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
    
    # Remove the connection context here
    with trace(
        workflow_name="Addiction System",
        trace_id=f"addiction-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        prompt = f"Update the player's addiction to {addiction_type}{f' related to NPC #{target_npc_id}' if target_npc_id else ''}. Player name: {player_name}. Progression multiplier: {progression_multiplier}"
        
        # Create context wrapper without connection
        ctx_wrapper = RunContextWrapper(addiction_context)
        
        # Call update function without connection parameter
        update_result = await update_addiction_level(
            ctx_wrapper, player_name, addiction_type,
            progression_multiplier=progression_multiplier,
            target_npc_id=target_npc_id
        )
        
        # Get addiction status for effects
        addiction_status = await check_addiction_levels(ctx_wrapper, player_name)
        narrative_effects = await generate_addiction_effects(ctx_wrapper, player_name, addiction_status)
        
    return {"update": update_result, "narrative_effects": narrative_effects, "addiction_type": addiction_type, "target_npc_id": target_npc_id}

async def process_addiction_effects(
    user_id: int, conversation_id: int, player_name: str, addiction_status: dict
) -> Dict[str, Any]:
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    effects_result = await generate_addiction_effects(
        RunContextWrapper(addiction_context), player_name, addiction_status
    )
    return effects_result

async def check_addiction_status(
    user_id: int, conversation_id: int, player_name: str
) -> Dict[str, Any]:
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    with trace(
        workflow_name="Addiction System",
        trace_id=f"addiction-status-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        ctx_wrapper = RunContextWrapper(addiction_context)
        levels_result = await check_addiction_levels(ctx_wrapper, player_name)
        effects_result = {"effects": [], "has_effects": False}
        if levels_result.get("has_addictions", False):
            effects_result = await generate_addiction_effects(ctx_wrapper, player_name, levels_result)
    
    return {"status": levels_result, "effects": effects_result}

async def get_addiction_status(
    user_id: int, conversation_id: int, player_name: str
) -> Dict[str, Any]:
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    levels_result = await check_addiction_levels(RunContextWrapper(addiction_context), player_name)
    
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

# Add helper functions for canon that don't exist yet
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
