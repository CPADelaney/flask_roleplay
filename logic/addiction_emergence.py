# logic/addiction_emergence.py

"""
REFACTORED: Now properly delegates all database writes to the addiction_system_sdk
which in turn uses canon/LoreSystem
"""

from logic.addiction_system_sdk import AddictionContext, update_addiction_level
from agents import RunContextWrapper
from pydantic import BaseModel, Field
import os, json
from typing import List, Dict, Any, Optional
from agents import Agent, Runner, ModelSettings
from db.connection import get_db_connection_context

################################################################################
# Model for emergent suggestion
################################################################################
class AddictionSuggestion(BaseModel):
    addiction_type: str
    is_npc_specific: bool = False
    target_npc_id: Optional[int] = None
    intensity: int = 1  # 1-4
    reason: str = ""
    suggested_themes: List[str] = Field(default_factory=list)  # Optional effect lines

################################################################################
# Emergent Addiction Analyzer agent
################################################################################

emergent_addiction_agent = Agent(
    name="EmergentAddictionAnalyzer",
    instructions="""
    You analyze recent narrative or events from a femdom roleplaying game session for signs that the player is developing *new* or emerging addictions.
    - Suggest plausible addiction types or variants that fit with the session story and player's behavior.
    - You can invent new addiction categories (e.g., "shoelaces", "gazes", "worship", "discipline") if the story makes sense.
    - For each suggested addiction, specify:
        - Type (short string, e.g. "shoelaces", "discipline")
        - Level (1 Mild - 4 Extreme)
        - If the addiction is specific to an NPC (and to who, if possible)
        - 1-3 very short theme sentences (for effects/messaging)
        - Why you suggest this (using evidence from the events)
    Output as a JSON array of objects.
    Examples:
      [{"addiction_type": "shoelaces", "is_npc_specific": false, "intensity": 1, "reason": "Player repeatedly fixated on Mistress' shoes.", "suggested_themes": ["You feel oddly drawn towards shoes and their laces."]}]
      [{"addiction_type": "praise", "is_npc_specific": true, "target_npc_id": 5, "intensity": 2, "reason": "Player responded extremely positively to Mistress Lyra's praise.", "suggested_themes": ["Her praise makes your heart race."]}]
    """,
    output_type=List[AddictionSuggestion],
    model_settings=ModelSettings(temperature=0.8),
)

################################################################################
# Emergent suggestion function (core workflow) - REFACTORED
################################################################################

@function_tool
async def suggest_addictions_from_events(
    ctx: RunContextWrapper[AddictionContext],
    player_name: str,
    recent_events: str,
    npcs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Analyze recent events for emergent/addiction discovery.
    If new addictions detected, creates them dynamically, applies effects, etc.
    
    REFACTORED: Now properly uses connection from context and delegates writes
    """
    # Prepare prompt from narrative/logs/events
    prompt = f"""
    Narrative and events log:
    {recent_events}

    Player: {player_name}
    """
    # Let LLM agent analyze and propose addictions.
    result = await Runner.run(
        emergent_addiction_agent,
        prompt,
        context=ctx.context,
    )
    suggestions: List[AddictionSuggestion] = result.final_output

    # 1. Dynamically add any new types to the theme messages system.
    thematics = ctx.context.thematic_messages
    updated_message_types = False
    for suggestion in suggestions:
        typ = suggestion.addiction_type
        level = suggestion.intensity
        # If not present, add with effects
        if typ not in thematics.messages:
            thematics.messages[typ] = {}
        if str(level) not in thematics.messages[typ]:
            effect = suggestion.suggested_themes[0] if suggestion.suggested_themes else f"You begin to develop a fixation on {typ}."
            thematics.messages[typ][str(level)] = effect
            updated_message_types = True
        # Optionally add *all* suggestions to that type/level
        for i, line in enumerate(suggestion.suggested_themes):
            thematics.messages[typ][str(level+i)] = line
            updated_message_types = True

    # If any new addictions/themes, optionally write back to file.
    if updated_message_types:
        try:
            THEMATIC_MESSAGES_FILE = os.getenv("THEMATIC_MESSAGES_FILE", "thematic_messages.json")
            with open(THEMATIC_MESSAGES_FILE, "w") as f:
                json.dump(thematics.messages, f, indent=2)
            print(f"New thematic addictions written to {THEMATIC_MESSAGES_FILE}.")
        except Exception as e:
            print(f"Warning: Could not save updated themes dynamically: {e}")

    # 2. For each suggestion, apply/add the addiction using update_addiction_level

    from logic.addiction_system_sdk import process_addiction_update
    
    applied = []
    for suggestion in suggestions:
        upd = await process_addiction_update(
            user_id=ctx.context.user_id,
            conversation_id=ctx.context.conversation_id,
            player_name=player_name,
            addiction_type=suggestion.addiction_type,
            progression_multiplier=1.0,
            target_npc_id=suggestion.target_npc_id if suggestion.is_npc_specific else None
        )
        applied.append(upd["update"])

    # Optionally, retrieve and return summary status after all changes
    from logic.addiction_system_sdk import get_addiction_status
    full_status = await get_addiction_status(
        ctx.context.user_id, ctx.context.conversation_id, player_name
    )

    return {
        "applied_suggestions": [s.dict() for s in suggestions],
        "update_results": applied,
        "player_addiction_status": full_status
    }

################################################################################
# Example high-level usage
################################################################################

async def analyze_and_apply_emergent_addictions(
    user_id: int, conversation_id: int, player_name: str, recent_narrative: str, npcs: Optional[List[dict]] = None
) -> Dict[str, Any]:
    ctx = AddictionContext(user_id, conversation_id)
    await ctx.initialize()
    result = await suggest_addictions_from_events(
        RunContextWrapper(ctx),
        player_name,
        recent_events=recent_narrative,
        npcs=npcs,
    )
    return result
