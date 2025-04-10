# logic/addiction_system_sdk.py
"""
Refactored Addiction System with full Nyx Governance integration.

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
# Agent Context and Directive Handler
# -------------------------------------------------------------------------------

class AddictionContext:
    """Context object for addiction agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.thematic_messages = {}
        self.directive_handler = None
        
    async def initialize(self):
        """Initialize context with governance integration"""
        # Lazy import to avoid circular dependency
        from nyx.integrate import get_central_governance
        
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        
        # Initialize directive handler
        self.directive_handler = DirectiveHandler(
            self.user_id, 
            self.conversation_id, 
            AgentType.UNIVERSAL_UPDATER,
            "addiction_system",
            governance=self.governor  # pass the object here
        )
        
        # Register handlers for different directive types
        self.directive_handler.register_handler(DirectiveType.ACTION, self._handle_action_directive)
        self.directive_handler.register_handler(DirectiveType.PROHIBITION, self._handle_prohibition_directive)
        
        # Start background processing of directives
        self.directive_task = await self.directive_handler.start_background_processing(interval=60.0)
        
        # Load thematic messages
        try:
            with open("thematic_messages.json", "r") as f:
                self.thematic_messages = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load external thematic messages; using defaults.")
            self.thematic_messages = DEFAULT_THEMATIC_MESSAGES
            
    async def _handle_action_directive(self, directive):
        """Handle action directives from Nyx"""
        instruction = directive.get("instruction", "")
        
        if "monitor addictions" in instruction.lower():
            # Trigger a monitoring scan
            return await check_addiction_status(
                self.user_id,
                self.conversation_id,
                directive.get("player_name", "player")
            )
        elif "apply addiction effect" in instruction.lower():
            # Apply a specific addiction effect
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
        """Handle prohibition directives from Nyx"""
        # Mark certain addiction types as prohibited
        prohibited = directive.get("prohibited_actions", [])
        
        # Store these in context for later checking
        self.prohibited_addictions = prohibited
        
        return {"status": "prohibition_registered", "prohibited": prohibited}

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

try:
    with open("thematic_messages.json", "r") as f:
        THEMATIC_MESSAGES = json.load(f)
except Exception as e:
    logging.warning("Could not load external thematic messages; using defaults.")
    THEMATIC_MESSAGES = DEFAULT_THEMATIC_MESSAGES

# -------------------------------------------------------------------------------
# Function Tools with Governance Integration
# -------------------------------------------------------------------------------

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
    """
    Checks the player's addiction levels with Nyx governance oversight.
    
    Args:
        player_name: Name of the player
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Connect to database using async context manager
    try:
        async with get_db_connection_context() as conn:
            # Ensure table exists
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
            
            # Query addictions
            rows = await conn.fetch("""
                SELECT addiction_type, level, target_npc_id
                FROM PlayerAddictions
                WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
            """, user_id, conversation_id, player_name)
            
            addiction_data = {}
            npc_specific = []
            
            for row in rows:
                addiction_type, level, target_npc_id = row
                if target_npc_id is None:
                    addiction_data[addiction_type] = level
                else:
                    # Get NPC name
                    npc_row = await conn.fetchrow("""
                        SELECT npc_name FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """, user_id, conversation_id, target_npc_id)
                    
                    npc_name = npc_row["npc_name"] if npc_row else f"NPC#{target_npc_id}"
                    npc_specific.append({
                        "addiction_type": addiction_type,
                        "level": level,
                        "npc_id": target_npc_id,
                        "npc_name": npc_name
                    })
            
            has_addictions = any(lvl > 0 for lvl in addiction_data.values()) or bool(npc_specific)
            
            result = {
                "addiction_levels": addiction_data,
                "npc_specific_addictions": npc_specific,
                "has_addictions": has_addictions
            }
            
            return result
            
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
    """
    Update or create an addiction entry for a player with Nyx governance oversight.
    
    Args:
        player_name: Name of the player
        addiction_type: Type of addiction to update
        progression_chance: Chance for addiction to increase (0.0-1.0)
        progression_multiplier: Multiplier for progression chance
        regression_chance: Chance for addiction to decrease (0.0-1.0)
        target_npc_id: Optional NPC ID for NPC-specific addictions
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Check if this addiction type is prohibited by governance
    if hasattr(ctx.context, 'prohibited_addictions') and addiction_type in ctx.context.prohibited_addictions:
        return {
            "error": f"Addiction type '{addiction_type}' is prohibited by governance directive",
            "addiction_type": addiction_type,
            "prohibited": True
        }
    
    # Connect to database using async context manager
    try:
        async with get_db_connection_context() as conn:
            # Ensure table exists
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
            
            # Get current level if exists
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
            
            # Calculate new level
            prev_level = current_level
            roll = random.random()
            
            if roll < (progression_chance * progression_multiplier) and current_level < 4:
                current_level += 1
                logging.info(f"Addiction progressed from {prev_level} to {current_level}")
            elif roll > (1 - regression_chance) and current_level > 0:
                current_level -= 1
                logging.info(f"Addiction regressed from {prev_level} to {current_level}")
            
            # Insert or update
            if target_npc_id is None:
                await conn.execute("""
                    INSERT INTO PlayerAddictions
                        (user_id, conversation_id, player_name, addiction_type, level, last_updated)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                    DO UPDATE SET level=EXCLUDED.level, last_updated=NOW()
                """, user_id, conversation_id, player_name, addiction_type, current_level)
            else:
                await conn.execute("""
                    INSERT INTO PlayerAddictions
                        (user_id, conversation_id, player_name, addiction_type, level, target_npc_id, last_updated)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW())
                    ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                    DO UPDATE SET level=EXCLUDED.level, last_updated=NOW()
                """, user_id, conversation_id, player_name, addiction_type, current_level, target_npc_id)
            
            # Apply stat penalty if extreme
            if current_level == 4:
                await conn.execute("""
                    UPDATE PlayerStats
                    SET willpower = GREATEST(willpower - $1, 0)
                    WHERE user_id = $2 AND conversation_id = $3 AND player_name = $4
                """, 5, user_id, conversation_id, player_name)
            
            result = {
                "addiction_type": addiction_type,
                "previous_level": prev_level,
                "new_level": current_level,
                "level_name": ADDICTION_LEVELS.get(current_level, "Unknown"),
                "progressed": current_level > prev_level,
                "regressed": current_level < prev_level,
                "target_npc_id": target_npc_id
            }
            
            return result
            
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
    """
    Generate narrative effects for a player's addictions with Nyx governance oversight.
    
    Args:
        player_name: Name of the player
        addiction_status: Addiction status from check_addiction_levels
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    thematic_messages = ctx.context.thematic_messages
    
    effects = []
    
    # General addictions
    addiction_levels = addiction_status.get("addiction_levels", {})
    for addiction_type, level in addiction_levels.items():
        if level <= 0:
            continue
            
        # Get messages for this addiction type and level
        type_messages = thematic_messages.get(addiction_type, {})
        messages = [type_messages.get(str(lvl), "") for lvl in range(1, level + 1)]
        # Filter out empty strings
        effects.extend(msg for msg in messages if msg)
    
    # NPC-specific addictions
    npc_specific = addiction_status.get("npc_specific_addictions", [])
    for entry in npc_specific:
        addiction_type = entry["addiction_type"]
        npc_name = entry.get("npc_name", f"NPC#{entry['npc_id']}")
        level = entry["level"]
        
        if level >= 3:
            effects.append(f"You have a {ADDICTION_LEVELS[level]} addiction to {npc_name}'s {addiction_type}.")
            
            # For extreme level, generate a special event
            if level >= 4:
                try:
                    # Get NPC data - UPDATED to use async context manager
                    async with get_db_connection_context() as conn:
                        npc_data = await conn.fetchrow("""
                            SELECT npc_name, archetype_summary, personality_traits, dominance, cruelty
                            FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """, user_id, conversation_id, entry["npc_id"])
                        
                        if npc_data:
                            # Create a special event prompt
                            prompt = f"""
                            Generate a 2-3 paragraph intense narrative scene about the player's extreme addiction 
                            to {npc_name}'s {addiction_type}. This is for a femdom roleplaying game.
                            
                            NPC Details:
                            - Name: {npc_name}
                            - Archetype: {npc_data["archetype_summary"]}
                            - Dominance: {npc_data["dominance"]}/100
                            - Cruelty: {npc_data["cruelty"]}/100
                            - Personality: {', '.join(npc_data["personality_traits"][:3]) if npc_data["personality_traits"] else "Unknown"}
                            
                            Write an intense, immersive scene that shows how this addiction is affecting the player.
                            """
                            
                            # Use the LLM to generate the special event
                            result = await Runner.run(
                                special_event_agent,
                                prompt,
                                context=ctx.context
                            )
                            
                            special_event = result.final_output
                            
                            if special_event:
                                effects.append(special_event)
                except Exception as e:
                    logging.error(f"Error generating special event: {e}")
    
    result = {
        "effects": effects,
        "has_effects": bool(effects)
    }
    
    return result

# -------------------------------------------------------------------------------
# Guardrail Functions
# -------------------------------------------------------------------------------

async def addiction_content_safety(ctx, agent, input_data):
    """Input guardrail for addiction content moderation"""
    content_moderator = Agent(
        name="Addiction Content Moderator",
        instructions="""
        You check if addiction content is appropriate for the game setting. 
        Allow adult themes in a femdom context but flag anything that might be genuinely harmful
        or that trivializes real addiction issues in a way that's ethically problematic.
        """,
        output_type=AddictionSafety
    )
    
    result = await Runner.run(content_moderator, input_data, context=ctx.context)
    final_output = result.final_output_as(AddictionSafety)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# -------------------------------------------------------------------------------
# Agent Definitions
# -------------------------------------------------------------------------------

# Agent for generating special events for extreme addictions
special_event_agent = Agent[AddictionContext](
    name="Special Event Generator",
    instructions="""
    You generate vivid, immersive narrative events for extreme addiction situations.
    Your scenes should:
    1. Be immersive and emotionally impactful
    2. Show the psychological effects of the addiction
    3. Include appropriate physical reactions
    4. Maintain the femdom theme without being explicitly sexual
    5. Be 2-3 paragraphs in length
    
    Write in second person perspective, addressing the player directly.
    """,
    model_settings=ModelSettings(temperature=0.7)
)

# Agent for handling addiction progression
addiction_progression_agent = Agent[AddictionContext](
    name="Addiction Progression Agent",
    instructions="""
    You handle the progression and regression of addictions.
    You analyze events and context to determine:
    1. When an addiction should progress or regress
    2. How quickly addiction should develop
    3. What multipliers should apply to progression chance
    4. What special thresholds might be reached
    
    Focus on gradual, realistic progression that aligns with player choices and experiences.
    
    Always respect directives from the Nyx governance system.
    """,
    tools=[update_addiction_level],
    output_type=AddictionUpdate
)

# Agent for generating narrative effects
addiction_narrative_agent = Agent[AddictionContext](
    name="Addiction Narrative Agent",
    instructions="""
    You generate narrative effects for addictions.
    Your narrative effects should:
    1. Be immersive and psychologically realistic
    2. Vary based on addiction level (mild, moderate, heavy, extreme)
    3. Reflect the specific addiction type
    4. Subtly incorporate femdom themes
    5. Show how the addiction affects the player's mind and perceptions
    
    Create effects that enhance the roleplaying experience without being too intrusive.
    
    Always respect directives from the Nyx governance system.
    """,
    tools=[generate_addiction_effects],
    output_type=AddictionEffects
)

# Main addiction system agent
addiction_system_agent = Agent[AddictionContext](
    name="Addiction System Agent",
    instructions="""
    You are the central addiction management system for a femdom roleplaying game.
    
    Your role is to:
    1. Track player addictions to various stimuli
    2. Progress or regress addiction levels based on exposure and choices
    3. Generate appropriate narrative effects based on addiction levels
    4. Manage NPC-specific addictions alongside general addictions
    5. Create special events for extreme addiction levels
    
    Handle addictions in a psychologically realistic way, using the addiction level system:
    - Level 0: None
    - Level 1: Mild
    - Level 2: Moderate
    - Level 3: Heavy
    - Level 4: Extreme
    
    Use specialized sub-agents for specific tasks as needed.
    
    Always respect directives from the Nyx governance system and check permissions
    before performing any actions.
    """,
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
        InputGuardrail(guardrail_function=addiction_content_safety),
    ],
    model_settings=ModelSettings(temperature=0.4)
)

# -------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------

async def process_addiction_update(
    user_id: int,
    conversation_id: int,
    player_name: str,
    addiction_type: str,
    progression_multiplier: float = 1.0,
    target_npc_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process an addiction update with Nyx governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player
        addiction_type: Type of addiction to update
        progression_multiplier: Multiplier for progression chance
        target_npc_id: Optional NPC ID for NPC-specific addictions
        
    Returns:
        Update results
    """
    # Create addiction context
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Addiction System",
        trace_id=f"addiction-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Create prompt
        prompt = f"""
        Update the player's addiction to {addiction_type}{f" related to NPC #{target_npc_id}" if target_npc_id else ""}.
        Player name: {player_name}
        Progression multiplier: {progression_multiplier}
        """
        
        # Run the agent
        result = await Runner.run(
            addiction_system_agent,
            prompt,
            context=addiction_context
        )
    
    # Process the result
    update_result = None
    narrative_effects = None
    
    for item in result.new_items:
        if item.type == "handoff_output_item":
            if "manage_addiction_progression" in str(item.raw_item):
                try:
                    update_result = json.loads(item.raw_item.content)
                except Exception as e:
                    logging.error(f"Error parsing addiction update result: {e}")
            elif "generate_narrative_effects" in str(item.raw_item):
                try:
                    narrative_effects = json.loads(item.raw_item.content)
                except Exception as e:
                    logging.error(f"Error parsing narrative effects: {e}")
    
    return {
        "update": update_result,
        "narrative_effects": narrative_effects,
        "addiction_type": addiction_type,
        "target_npc_id": target_npc_id
    }

async def process_addiction_effects(
    user_id: int,
    conversation_id: int,
    player_name: str,
    addiction_status: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process addiction effects based on the provided addiction status.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player
        addiction_status: Addiction status from get_addiction_status
        
    Returns:
        Dictionary with addiction effects
    """
    # Create addiction context
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    # Generate effects using existing function
    effects_result = await generate_addiction_effects(
        RunContextWrapper(addiction_context),
        player_name,
        addiction_status
    )
    
    return effects_result

def get_addiction_label(level: int) -> str:
    """
    Get the label for an addiction level.
    
    Args:
        level: Addiction level (0-4)
        
    Returns:
        Label for the addiction level
    """
    return ADDICTION_LEVELS.get(level, "Unknown")

async def check_addiction_status(
    user_id: int,
    conversation_id: int,
    player_name: str
) -> Dict[str, Any]:
    """
    Check a player's addiction status with Nyx governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player
        
    Returns:
        Addiction status and effects
    """
    # Create addiction context
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Addiction System",
        trace_id=f"addiction-status-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Get addiction levels
        levels_result = await check_addiction_levels(
            RunContextWrapper(addiction_context),
            player_name
        )
        
        # If there are addictions, get effects
        effects_result = {"effects": [], "has_effects": False}
        if levels_result.get("has_addictions", False):
            effects_result = await generate_addiction_effects(
                RunContextWrapper(addiction_context),
                player_name,
                levels_result
            )
    
    return {
        "status": levels_result,
        "effects": effects_result
    }

# Register with Nyx governance
async def register_with_governance(user_id: int, conversation_id: int):
    """
    Register addiction agents with Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    # Lazy import to avoid circular dependency
    from nyx.integrate import get_central_governance
    
    # Get governor
    governor = await get_central_governance(user_id, conversation_id)
    
    # Register main agent
    await governor.register_agent(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_instance=addiction_system_agent,
        agent_id="addiction_system"
    )
    
    # Issue directive for addiction monitoring
    await governor.issue_directive(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="addiction_system",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Monitor player addictions and apply appropriate effects",
            "scope": "game"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60  # 24 hours
    )
    
    logging.info("Addiction system registered with Nyx governance")

# Handle directives from Nyx
async def process_addiction_directive(directive_data: Dict[str, Any], user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Process a directive from Nyx governance system.
    
    Args:
        directive_data: The directive data
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Result of processing the directive
    """
    # Create addiction context
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    # Initialize directive handler if needed
    if not addiction_context.directive_handler:
        addiction_context.directive_handler = DirectiveHandler(
            user_id, 
            conversation_id, 
            AgentType.UNIVERSAL_UPDATER,
            "addiction_system"
        )
        
    # Process the directive
    result = await addiction_context.directive_handler._handle_action_directive(directive_data)
    
    return result
async def get_addiction_status(
    user_id: int,
    conversation_id: int,
    player_name: str
) -> Dict[str, Any]:
    """
    Get comprehensive addiction status for a player.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player
        
    Returns:
        Dictionary with addiction status
    """
    # Create addiction context
    addiction_context = AddictionContext(user_id, conversation_id)
    await addiction_context.initialize()
    
    # Use existing function to get addiction levels
    levels_result = await check_addiction_levels(
        RunContextWrapper(addiction_context),
        player_name
    )
    
    # Format the addiction status with labels
    result = {
        "has_addictions": levels_result.get("has_addictions", False),
        "addictions": {}
    }
    
    # Process general addictions
    addiction_levels = levels_result.get("addiction_levels", {})
    for addiction_type, level in addiction_levels.items():
        if level > 0:  # Only include active addictions
            result["addictions"][addiction_type] = {
                "level": level,
                "label": get_addiction_label(level),
                "type": "general"
            }
    
    # Process NPC-specific addictions
    npc_specific = levels_result.get("npc_specific_addictions", [])
    for addiction in npc_specific:
        addiction_type = addiction.get("addiction_type")
        npc_id = addiction.get("npc_id")
        npc_name = addiction.get("npc_name", f"NPC#{npc_id}")
        level = addiction.get("level", 0)
        
        if level > 0:  # Only include active addictions
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
