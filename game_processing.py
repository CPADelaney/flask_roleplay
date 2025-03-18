# game_processing.py
# DEPRECATED: This file is deprecated and should not be used.
# Please use new_game_agent.py instead, which integrates with the Nyx governance system.
# All functionality has been migrated to NewGameAgent which provides a more robust
# implementation with proper governance integration.

import logging
import json
import random
import time
import asyncio
import os
import asyncpg
import re
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from datetime import datetime
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator import get_aggregated_roleplay_context
from logic.gpt_helpers import adjust_npc_complete
from routes.story_routes import build_aggregator_text
from logic.gpt_utils import spaced_gpt_call, safe_int
from logic.universal_updater import apply_universal_updates_async
from logic.calendar import update_calendar_names
from db.connection import get_db_connection
from logic.npc_creation import spawn_multiple_npcs_enhanced, create_and_refine_npc, init_chase_schedule
from logic.gpt_image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from lore.setting_analyzer import SettingAnalyzer
from logic.conflict_system.conflict_integration import ConflictSystemIntegration

DB_DSN = os.getenv("DB_DSN") 

async def spaced_gpt_call_with_retry(conversation_id, context, prompt, delay=1.0, max_retries=5):
    """
    Calls GPT with a short initial delay, then attempts exponential backoff
    if we get a 429 (rate limit) error.
    """
    wait_before_call = delay
    attempt = 1

    while attempt <= max_retries:
        logging.info(
            "Waiting %.1f seconds before calling GPT (conversation_id=%s) (attempt=%d)",
            wait_before_call, conversation_id, attempt
        )
        await asyncio.sleep(wait_before_call)

        try:
            logging.info("Calling GPT (conversation_id=%s, attempt=%d)", conversation_id, attempt)
            result = await asyncio.to_thread(_sync_gpt_request, conversation_id, context, prompt)
            logging.info("GPT returned (attempt=%d): %s", attempt, result)
            return result  # If success, return immediately

        except Exception as e:
            if "429" in str(e) or "RateLimitError" in str(e):
                logging.warning("Got 429 on attempt %d; backing off.", attempt)
                attempt += 1
                wait_before_call *= 2
                if attempt > max_retries:
                    logging.error("Max retries reached after repeated 429 errors.")
                    raise
            else:
                logging.error("Non-429 error: %s", e, exc_info=True)
                raise

    raise RuntimeError("Failed to call GPT after repeated attempts.")

def _sync_gpt_request(conversation_id, context, prompt):
    client = get_openai_client()
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )

    finish_reason = resp.choices[0].finish_reason
    content = resp.choices[0].message.content or ""
    function_call = resp.choices[0].message.function_call

    result = {}
    if function_call:
        # If there's a function_call, parse arguments if present.
        result["type"] = "function_call"
        arguments_str = function_call.arguments or ""
        try:
            result["function_args"] = json.loads(arguments_str)
        except:
            result["function_args"] = {"raw": arguments_str}
    else:
        result["type"] = "text"
        result["response"] = content

    return result


# -------------------------------------------------------------------------
# SINGLE-PASS PROMPTS
# -------------------------------------------------------------------------

ENV_PROMPT = f"""
You are setting up a new daily-life sim environment with subtle, hidden layers of femdom and intrigue.

Below is a merged environment concept blending multiple settings:
Mega Setting Name: {{mega_name}}
Mega Description:
{{mega_desc}}

Using this as inspiration, produce a strictly valid JSON object with the **exact** keys below (no extras). Each key's purpose is described:

1. "setting_name" (string; a short, creative name evoking the merged environment)
2. "environment_desc" (string; 1–3 paragraphs painting a vivid, cozy daily-life setting—its look, feel, routines, and culture. Focus on a mundane yet charming atmosphere with faint, unsettling undertones of structure or influence, avoiding overt dominance or kink. Highlight society's norms subtly hinting at control.)
3. "environment_history" (string; a short paragraph on past events shaping the setting, with a neutral tone and vague echoes of power shifts)
4. "events" (array; objects with):
      - "name"
      - "description" (keep festive, community-focused, with subtle hints of obligation)
      - "start_time"
      - "end_time"
      - "location"
      - "year"
      - "month"
      - "day"
      - "time_of_day"
   (Fill a year with creative, slice-of-life festivals—avoid explicit femdom themes.)
5. "locations" (array; objects with):
      - "location_name"
      - "description" (everyday spots with a touch of charm or oddity, no overt control)
      - "open_hours"
   (At least 10 distinct locations.)
6. "scenario_name" (string; a catchy, neutral title for this setup)
7. "quest_data" (object with):
      - "quest_name" (engaging, mysterious, no femdom cues)
      - "quest_description" (intriguing hook with subtle unease)

Output only the JSON object, no additional text.

Reference these details if provided:
Environment components: {{env_components}}
Enhanced features: {{enhanced_features}}
Stat modifiers: {{stat_modifiers}}
"""


NPC_PROMPT = f"""
You are crafting schedules for NPCs and the player "Chase" in this daily-life sim environment:
{{environment_desc}}

Here's each NPC's refined data (schedules missing):
{{refined_npc_data}}

Return exactly one JSON with keys:
  "npc_creations": [ {{ ... }}, ... ],
  "ChaseSchedule": {{}}

For each NPC in "npc_creations":
  - "npc_name" (from refined_npc_data)
  - "likes", "dislikes", "hobbies", "affiliations"
  - "schedule": day-based subkeys (Morning, Afternoon, Evening, Night) for: {{day_names}}

For "ChaseSchedule": day-based subkeys (Morning, Afternoon, Evening, Night) for: {{day_names}}

Constraints:
- Schedules must fit the setting's cozy, mundane vibe, reflecting each NPC's archetypes/likes/dislikes/hobbies/etc., with subtle hints of influence (e.g., a 'student' NPC attends class but lingers oddly long).
- NPCs wear friendly, charming masks—hide any dominance or control behind routine tasks (e.g., 'coffee run' masking a power play).
- Chase's schedule feels normal yet guided, overlapping with NPCs subtly.
- Output only JSON, no commentary.
"""
print("DEBUG NPC_PROMPT ->", repr(NPC_PROMPT))


# -------------------------------------------------------------------------
# GPT call wrappers (single-block approach)
# -------------------------------------------------------------------------
async def call_gpt_for_environment_data(
    conversation_id,
    env_comps,
    enh_feats,
    stat_mods,
    mega_name,
    mega_desc
):
    """
    Single GPT call for environment-level data.
    Returns a dict with:
    {
      "setting_name": str,
      "environment_desc": str,
      "environment_history": str,
      "events": [...],
      "locations": [...],
      "scenario_name": str,
      "quest_data": {...}
    }
    """
    prompt = ENV_PROMPT.format(
        mega_name=mega_name,
        mega_desc=mega_desc,
        env_components="\n".join(env_comps),
        enhanced_features=", ".join(enh_feats),
        stat_modifiers=", ".join(f"{k}: {v}" for k,v in stat_mods.items())
    )

    result = await spaced_gpt_call_with_retry(conversation_id, "", prompt, delay=1.0)

    if result.get("type") == "function_call":
        fn_args = result.get("function_args", {})
        return fn_args
    else:
        raw_text = result.get("response", "").strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        try:
            data = json.loads(raw_text)
            return data
        except Exception as e:
            logging.error("Error parsing environment JSON: %s", e, exc_info=True)
            return {}


#async def call_gpt_for_npcs_and_chase(conversation_id, environment_desc, day_names):
#    """
#    Single GPT call for multiple NPCs plus ChaseSchedule in one pass.
#    Returns dict:
#    {
#      "npc_creations":[ {...}, {...}],
#      "ChaseSchedule": {...}
#    }
#    """
#    prompt = NPC_PROMPT.format(
#        environment_desc=environment_desc,
#        day_names=", ".join(day_names)
#    )
#
#    result = await spaced_gpt_call_with_retry(conversation_id, environment_desc, prompt, delay=1.0)
#
#    if result.get("type") == "function_call":
#        return result.get("function_args", {})
#    else:
#        raw_text = result.get("response", "").strip()
#        if raw_text.startswith("```"):
#            lines = raw_text.splitlines()
#            if lines and lines[0].startswith("```"):
#                lines = lines[1:]
#            if lines and lines[-1].startswith("```"):
#                lines = lines[:-1]
#            raw_text = "\n".join(lines).strip()
#
#        try:
#            data = json.loads(raw_text)
#            return data
#        except Exception as e:
#            logging.error("Error parsing NPC+Chase JSON: %s", e, exc_info=True)
#            return {}
#
# -------------------------------------------------------------------------
# MAIN NEW GAME FLOW with single-block GPT calls + advanced features
# -------------------------------------------------------------------------
async def async_process_new_game(user_id, conversation_data):
    """
    DEPRECATED: This function is deprecated. Use NewGameAgent.process_new_game instead.
    
    This function exists only for backward compatibility and will be removed in a future release.
    """
    logging.warning("async_process_new_game is deprecated. Use NewGameAgent.process_new_game instead.")
    
    # Import here to avoid circular imports
    from new_game_agent import NewGameAgent
    
    # Create a NewGameAgent instance and use it
    agent = NewGameAgent()
    return await agent.process_new_game(user_id, conversation_data)
