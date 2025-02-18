# logic/gpt_parser.py

import json
import logging
import asyncio
from logic.gpt_utils import spaced_gpt_call  # Your helper for GPT calls

# Define your extraction prompt – this is the part where you “rip” the JSON payload stuff.
EXTRACTION_PROMPT = """
Based on the following narrative and context, produce a strictly valid JSON object that summarizes all game state updates according to the following schema. Output ONLY the JSON object with no additional commentary or formatting.

The schema is as follows:
{{
  "roleplay_updates": {{
    "CurrentYear": <number>,
    "CurrentMonth": <number>,
    "CurrentDay": <number>,
    "TimeOfDay": "<string>"
  }},
  "ChaseSchedule": {{ /* provide a complete weekly schedule if it has changed, otherwise output an empty object {{}} */ }},
  "MainQuest": "<string>",
  "PlayerRole": "<string>",
  "npc_creations": [ /* array of new NPC objects or [] */ ],
  "npc_updates": [ /* array of NPC update objects or [] */ ],
  "character_stat_updates": {{ "player_name": "Chase", "stats": {{ /* stat changes or {{}} */ }} }},
  "relationship_updates": [ /* array of relationship update objects or [] */ ],
  "npc_introductions": [ /* array of NPC introduction objects or [] */ ],
  "location_creations": [ /* array of location creation objects or [] */ ],
  "event_list_updates": [ /* array of event objects or [] */ ],
  "inventory_updates": {{
      "player_name": "Chase", 
      "added_items": [ /* if the narrative or context (or previous responses) mentions new inventory items (e.g., "Debugging Amulet"), list them as objects with keys: item_name, item_description, item_effect, and category; otherwise, output an empty array */ ],
      "removed_items": [ /* similarly, list any items to remove if mentioned; otherwise, output an empty array */ ]
  }},
  "quest_updates": [ /* array of quest update objects or [] */ ],
  "social_links": [ /* array of social link objects or [] */ ],
  "perk_unlocks": [ /* array of perk unlock objects or [] */ ]
}}

Important: If any inventory items (for example, a Debugging Amulet) were mentioned in previous responses or the context, include them as objects in "added_items" with all required fields.

Narrative:
{narrative}

Context:
{context}
"""

async def generate_narrative(conversation_id, aggregator_text, user_input):
    # Using your spaced_gpt_call helper (which implements retry/backoff)
    narrative_resp = await spaced_gpt_call(conversation_id, aggregator_text, user_input)
    # If the response type is a function call, then the narrative field may be missing.
    narrative = narrative_resp.get("response")
    if narrative is None:
        narrative = ""
    narrative = narrative.strip()
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
    Generates the narrative and extracts the state update payload by making two GPT calls:
      1. An initial call to generate the narrative and capture any function call update.
      2. A second call using an extraction prompt to extract state updates from the narrative.
    Then, the two update payloads are merged.
    Returns a tuple of (narrative, merged_update).
    """
    # First GPT call: generate narrative
    initial_resp = await spaced_gpt_call(conversation_id, aggregator_text, user_input)
    if initial_resp is None:
        # If no response was returned, use a default narrative.
        narrative = "[No narrative generated]"
        initial_update = {}
    else:
        narrative = (initial_resp.get("response") or "").strip()
        if not narrative:
            narrative = "[No narrative generated]"
        initial_update = {}
        if initial_resp.get("type") == "function_call":
            initial_update = initial_resp.get("function_args", {})

    # Second GPT call: use extraction prompt to get updates from narrative and context.
    prompt = EXTRACTION_PROMPT.format(narrative=narrative, context=aggregator_text)
    extraction_resp = await spaced_gpt_call(conversation_id, aggregator_text, prompt)
    extracted_update = {}
    if extraction_resp is not None:
        if extraction_resp.get("type") == "function_call":
            extracted_update = extraction_resp.get("function_args", {})
        else:
            try:
                extracted_update = json.loads(extraction_resp.get("response", ""))
            except Exception as e:
                logging.error("Error parsing extracted update JSON: %s", e)
    else:
        logging.error("Extraction GPT call returned None")

    # Merge the two updates from the current interaction.
    from logic.state_update_helper import merge_state_updates  # Import our merge helper
    current_update = merge_state_updates(initial_update, extracted_update)
    
    return narrative, current_update


def merge_state_updates(old_update: dict, new_update: dict) -> dict:
    """
    Merges two state update payloads.
    For the inventory_updates section, if the new update's 'added_items' (or 'removed_items')
    is empty but the old update has values, preserve the old values.
    """
    # Start with a copy of the new update
    merged = new_update.copy()

    # Merge the inventory updates
    old_inv = old_update.get("inventory_updates", {})
    new_inv = new_update.get("inventory_updates", {})

    # Merge 'added_items'
    if not new_inv.get("added_items") and old_inv.get("added_items"):
        merged.setdefault("inventory_updates", {})["added_items"] = old_inv["added_items"]

    # Merge 'removed_items'
    if not new_inv.get("removed_items") and old_inv.get("removed_items"):
        merged.setdefault("inventory_updates", {})["removed_items"] = old_inv["removed_items"]

    # (You can extend this merge logic for other parts as needed)
    return merged
