# logic/gpt_parser.py

import json
import logging
import asyncio
from logic.gpt_utils import spaced_gpt_call  # Your helper for GPT calls

# Define your extraction prompt – this is the part where you “rip” the JSON payload stuff.
EXTRACTION_PROMPT = """
Based on the following narrative and context, produce a strictly valid JSON object that summarizes all game state updates according to the following schema. Output ONLY the JSON object with no additional commentary or formatting:

{
  "roleplay_updates": {
    "CurrentYear": <number>,
    "CurrentMonth": <number>,
    "CurrentDay": <number>,
    "TimeOfDay": "<string>"
  },
  "ChaseSchedule": { /* complete weekly schedule or {} if unchanged */ },
  "MainQuest": "<string>",
  "PlayerRole": "<string>",
  "npc_creations": [ /* array of new NPC objects or [] */ ],
  "npc_updates": [ /* array of NPC update objects or [] */ ],
  "character_stat_updates": { "player_name": "Chase", "stats": { /* stat changes or {} */ } },
  "relationship_updates": [ /* array of relationship update objects or [] */ ],
  "npc_introductions": [ /* array of NPC introduction objects or [] */ ],
  "location_creations": [ /* array of location creation objects or [] */ ],
  "event_list_updates": [ /* array of event objects or [] */ ],
  "inventory_updates": { "player_name": "Chase", "added_items": [], "removed_items": [] },
  "quest_updates": [ /* array of quest update objects or [] */ ],
  "social_links": [ /* array of social link objects or [] */ ],
  "perk_unlocks": [ /* array of perk unlock objects or [] */ ]
}

Narrative:
{narrative}

Context:
{context}
"""

async def generate_narrative(conversation_id, aggregator_text, user_input):
    """
    Calls GPT to generate the narrative response.
    """
    # Using your _with_retry helper
    narrative_resp = await _with_retry(conversation_id, aggregator_text, user_input)
    narrative = narrative_resp.get("response", "").strip()
    if not narrative:
        narrative = "[No narrative generated]"
    return narrative

async def extract_state_updates(conversation_id, aggregator_text, narrative):
    """
    Calls GPT to produce a JSON payload that extracts state updates based on the narrative and context.
    """
    # Fill in the extraction prompt with the current narrative and context.
    prompt = EXTRACTION_PROMPT.format(narrative=narrative, context=aggregator_text)
    # Force a function call if desired or simply get the text response.
    state_resp = await spaced_gpt_call(conversation_id, aggregator_text, prompt)
    
    updates_payload = {}
    # If the response is a function call, use its arguments.
    if state_resp.get("type") == "function_call":
        updates_payload = state_resp.get("function_args", {})
    else:
        try:
            updates_payload = json.loads(state_resp.get("response", ""))
        except Exception as e:
            logging.error("Error parsing state update JSON: %s", e)
    return updates_payload

async def generate_narrative_and_updates(conversation_id, aggregator_text, user_input):
    """
    Generates the narrative and extracts the state update payload.
    Returns a tuple of (narrative, updates_payload).
    """
    # First, generate the narrative response
    narrative = await generate_narrative(conversation_id, aggregator_text, user_input)
    # Then, extract state updates from the narrative and context
    updates_payload = await extract_state_updates(conversation_id, aggregator_text, narrative)
    return narrative, updates_payload

# Example usage:
# In your /next_storybeat route, after preparing aggregator_text and getting conversation_id:
#
# narrative, updates_payload = await generate_narrative_and_updates(conv_id, aggregator_text, user_input)
#
# Then:
# 1. Insert the narrative text into your messages table.
# 2. Call your apply_universal_updates function with the updates_payload.
