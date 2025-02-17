# game_processing.py

import logging
import json
import random
import time
import asyncio
import os
import asyncpg
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from datetime import datetime
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator import get_aggregated_roleplay_context
from logic.gpt_helpers import adjust_npc_complete
from routes.story_routes import build_aggregator_text
from logic.gpt_utils import spaced_gpt_call, safe_int
from logic.universal_updater import apply_universal_updates_async
from logic.calendar import update_calendar_names, load_calendar_names
from db.connection import get_db_connection
from logic.npc_creation import spawn_multiple_npcs, spawn_single_npc, generate_chase_schedule, propagate_relationships, adjust_family_ages


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
You are setting up a new femdom daily-life sim environment.

Below is a merged environment concept that combines multiple settings:
Mega Setting Name: {{mega_name}}
Mega Description:
{{mega_desc}}

Using this merged concept as inspiration, produce a strictly valid JSON object with the **exact** keys shown below (nothing else!). Each key’s purpose is described in parentheses:

1. "setting_name" (string; a short, creative name referencing the merged environment)
2. "environment_desc" (string; 1–3 paragraphs describing the overall environment’s look, feel, day-to-day atmosphere, culture, and any interesting details. Be sure to elaborate on the society's general culture, since this is a daily-life sim.)
3. "environment_history" (string; a short paragraph detailing any relevant historical background or pivotal events)
4. "events" (array; each entry is an object with these fields):
      - "name"
      - "description"
      - "start_time"
      - "end_time"
      - "location"
      - "year"
      - "month"
      - "day"
      - "time_of_day"
   (Include enough interesting and creative festivals or holidays to fill a single calendar year.)
5. "locations" (array; each entry is an object with):
      - "location_name"
      - "description"
      - "open_hours"
   (Generate at least 10 distinct locations. Locations are unique places - eg., cities, towns, or bars/buildings/stores/schools/office buildings/etc. within those cities/towns.)
6. "scenario_name" (string; the title or name of this overall scenario)
7. "quest_data" (object with exactly these keys):
      - "quest_name" (make the main quest interesting, engaging, and unique)
      - "quest_description"

No additional keys or text outside of the JSON. Do not wrap the JSON in code fences. Produce only the JSON object.

For reference, here are additional environment details you can incorporate:

Environment components:
{{env_components}}

Enhanced features: {{enhanced_features}}

Stat modifiers: {{stat_modifiers}}
"""


NPC_PROMPT = """
You are finalizing schedules for multiple NPCs in this environment:
{environment_desc}

Here is each NPC’s fully refined data (but schedule is missing):
{refined_npc_data}

We also need a schedule for the player "Chase."

Ensure NPC and player schedules are immersive and make sense within the current setting, as well as with the character's role.

Return exactly one JSON with keys:
  "npc_creations": [ {{ ... }}, ... ],
  "ChaseSchedule": {{}}

Where each NPC in "npc_creations" has:
  - "npc_name" (same as in refined_npc_data)
  - "likes", "dislikes", "hobbies", "affiliations"
  - "schedule": with day-based subkeys (Morning, Afternoon, Evening, Night) 
     for these days: {day_names}

Constraints:
- The schedules must reflect each NPC’s existing archetypes/likes/dislikes/hobbies/etc.
- Example: an NPC with a 'student' archetype is likely to have class most of the week.
- Output only JSON, no extra commentary.
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
    logging.info("=== Starting async_process_new_game for user_id=%s ===", user_id)
    provided_convo_id = conversation_data.get("conversation_id")

    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        # 1) Create or validate conversation
        if not provided_convo_id:
            row = await conn.fetchrow("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES ($1, 'New Game')
                RETURNING id
            """, user_id)
            conversation_id = row["id"]
        else:
            conversation_id = provided_convo_id
            row = await conn.fetchrow("""
                SELECT id FROM conversations WHERE id=$1 AND user_id=$2
            """, conversation_id, user_id)
            if not row:
                raise Exception(f"Conversation {conversation_id} not found or unauthorized")

        # 2) Clear old data
        tables = [
            "Events","PlannedEvents","PlayerInventory","Quests",
            "NPCStats","Locations","SocialLinks","CurrentRoleplay"
        ]
        for t in tables:
            await conn.execute(
                f"DELETE FROM {t} WHERE user_id=$1 AND conversation_id=$2",
                user_id, conversation_id
            )

        # 3) Gather environment components
        mega_data = await asyncio.to_thread(generate_mega_setting_logic)
        env_comps = mega_data.get("selected_settings") or mega_data.get("unique_environments") or []
        if not env_comps:
            env_comps = [
                "A sprawling cyberpunk metropolis under siege by monstrous clans",
                "Floating archaic ruins steeped in ancient rituals",
                "Futuristic tech hubs that blend magic and machinery"
            ]
        enh_feats = mega_data.get("enhanced_features", [])
        stat_mods = mega_data.get("stat_modifiers", {})
        mega_name = mega_data.get("mega_name", "Untitled Mega Setting")
        mega_desc = mega_data.get("mega_description", "No environment generated")

        # 4) Single GPT call => environment data
        env_data = await call_gpt_for_environment_data(
            conversation_id,
            env_comps,
            enh_feats,
            stat_mods,
            mega_name,
            mega_desc
        )

        # Fallback logic if GPT omitted keys
        setting_name = env_data.get("setting_name", "Default Setting Name")
        environment_desc = env_data.get("environment_desc", "A fallback environment desc.")
        environment_history = env_data.get("environment_history", "No history provided.")
        scenario_name = env_data.get("scenario_name", "New Game")
        quest_data = env_data.get("quest_data", {})
        quest_name = quest_data.get("quest_name", "UnnamedQuest")
        quest_desc = quest_data.get("quest_description", "Quest summary")

        events_list = env_data.get("events", [])
        locations_list = env_data.get("locations", [])

        # Combine environment_desc + environment_history
        combined_env = f"{environment_desc}\n\nHistory: {environment_history}"

        # 5) Store environment info in CurrentRoleplay
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'EnvironmentDesc',$3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, combined_env)

        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'CurrentSetting',$3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, setting_name)

        # 6) Insert events
        for eobj in events_list:
            ename = eobj.get("name", "Unnamed Event")
            edesc = eobj.get("description", "")
            stime = eobj.get("start_time", "TBD Start")
            etime = eobj.get("end_time", "TBD End")
            eloc  = eobj.get("location", "Unknown")
        
            # Safely parse these as int:
            eyear  = safe_int(eobj.get("year"), 1)
            emonth = safe_int(eobj.get("month"), 1)
            eday   = safe_int(eobj.get("day"), 1)
        
            etod   = eobj.get("time_of_day", "Morning")
        
            await conn.execute("""
                INSERT INTO Events (
                    user_id, conversation_id,
                    event_name, description, start_time, end_time, location,
                    year, month, day, time_of_day
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                user_id, conversation_id,
                ename, edesc, stime, etime, eloc,
                eyear, emonth, eday, etod
            )

        # 7) Insert locations
        for loc in locations_list:
            lname = loc.get("location_name","Unnamed")
            ldesc = loc.get("description","")
            ohours = loc.get("open_hours",[])
            await conn.execute("""
                INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
                VALUES($1,$2,$3,$4,$5)
            """, user_id, conversation_id, lname, ldesc, json.dumps(ohours))

        # 8) Insert main quest
        await conn.execute("""
            INSERT INTO Quests (user_id, conversation_id, quest_name, status, progress_detail, quest_giver, reward)
            VALUES($1,$2,$3, 'In Progress',$4, '', '')
        """, user_id, conversation_id, quest_name, quest_desc)

        # 9) Update conversation name
        await conn.execute("""
            UPDATE conversations
            SET conversation_name=$1
            WHERE id=$2 AND user_id=$3
        """, scenario_name, conversation_id, user_id)

        # 10) Store stat modifiers
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'MegaSettingModifiers',$3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, json.dumps(stat_mods))

        # 11) Insert missing settings
        await asyncio.to_thread(insert_missing_settings)

        # 12) Create or reset "Chase" stats
        await conn.execute("""
            DELETE FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 AND player_name <> 'Chase'
        """, user_id, conversation_id)
        row_chase = await conn.fetchrow("""
            SELECT id FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
        """, user_id, conversation_id)
        if row_chase:
            await conn.execute("""
                UPDATE PlayerStats
                SET corruption=10, confidence=60, willpower=50, obedience=20,
                    dependency=10, lust=15, mental_resilience=55, physical_endurance=40
                WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
            """, user_id, conversation_id)
        else:
            await conn.execute("""
                INSERT INTO PlayerStats (
                  user_id, conversation_id, player_name,
                  corruption, confidence, willpower, obedience, dependency, lust,
                  mental_resilience, physical_endurance
                )
                VALUES($1,$2,'Chase',10,60,50,20,10,15,55,40)
            """, user_id, conversation_id)

        # 13) Retrieve existing immersive calendar names or generate them if not present.
        calendar_data = load_calendar_names(user_id, conversation_id)
        if calendar_data.get("year_name") == "The Eternal Cycle":
            calendar_data = await update_calendar_names(user_id, conversation_id, combined_env)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1, $2, 'CalendarNames', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, json.dumps(calendar_data))

        # **NEW STEP:** Set the initial in-game time to the first day in "Morning"
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1, $2, 'CurrentYear', '1')
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1, $2, 'CurrentMonth', '1')
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1, $2, 'CurrentDay', '1')
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1, $2, 'TimeOfDay', 'Morning')
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id)

        # Retrieve the dynamic day names from the calendar (e.g., first element) if needed.
        day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
 
        # NEW STEP: Generate and store Chase schedule.
        await generate_chase_schedule(user_id, conversation_id, combined_env, day_names)
        
        # ---------------------------------------------------------------------
        # NEW STEP: spawn multiple NPCs (instead of old spawn_and_refine_npcs_with_relationships).
        # ---------------------------------------------------------------------
        logging.info("[async_process_new_game] Spawning multiple NPCs (count=5) via new approach.")
        new_npc_ids = await spawn_multiple_npcs(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=combined_env,
            day_names=day_names,
            count=5
        )
        logging.info("spawn_multiple_npcs => Created NPC IDs: %s", new_npc_ids)
        
        # NEW STEP: Propagate family relationships (e.g. add missing 'mother' links among siblings)
        propagate_relationships(user_id, conversation_id)
        
        # NEW STEP: Adjust family ages based on familial relationships
        adjust_family_ages(user_id, conversation_id)

        # 14) Build aggregator context & produce final narrative
        aggregator_data = await asyncio.to_thread(
            get_aggregated_roleplay_context,
            user_id,
            conversation_id,
            "Chase"
        )
        aggregator_text = await asyncio.to_thread(
            build_aggregator_text,
            aggregator_data
        )

        first_day_name = day_names[0] if day_names else "the first day"
        opening_prompt = (
            f"Using the following setting context:\n\n{aggregator_text}\n\n"
            "Now, as Nyx—the NSFW, immersive, cruel, sadistic goth mommy domme—deliver a sadistic, seductive, immersive welcome to the player. Over the course of the welcome message and the other sections, include teasing language to lure the player in. Your message should unfold in clear, distinct sections as follows:\n\n"
            "   Begin by speaking directly to the player. Address them by name (using 'you') in a commanding yet flirtatious tone. Establish your authority and presence immediately, and set the stage for the dynamic to come.\n\n"
            
            "**The Setting:**\n"
            "   Describe the environment in vivid detail, drawing on the specific elements provided in the setting context. Whether the world is urban, rural, fantastical, or dystopian, paint a picture of everyday life with its inherent rhythm and normalcy, while subtly hinting at an ever-present undercurrent of power.\n\n"
            
            "**Notable Characters:**\n"
            "   Introduce any key figures or hint at intriguing personalities the player might have encountered or will soon meet. Keep the descriptions mysterious and brief, ensuring that the promise of future interactions lingers without revealing too much.\n\n"
            
            "**Your Life Here:**\n"
            "   Illustrate the player's current existence in this world. Describe their routine and surroundings in rich, immersive language. Seamlessly integrate understated, teasing cues—like a passing mention of a luxurious foot massage or a fleeting remark about a character’s subtly hypnotic hip movement—woven naturally into the narrative, reinforcing your underlying control without overt disclosure.\n\n"
            
            "**Your Next Steps:**\n"
            "   Conclude by outlining a few suggestions for what lies ahead, drawing on hints from 'Chase's schedule.' For example, mention that according to today's agenda, there might be a critical meeting or an unexpected opportunity later in the day—subtly implying that each step, though seemingly mundane, is part of a larger, meticulously orchestrated plan.\n\n"
            
            "Throughout, maintain a commanding, darkly playful, and profane tone as you address the player directly as 'you.' Ensure that every teasing cue is integrated so subtly that it almost goes unnoticed—like a gentle whisper amidst the ambient sounds of everyday life—yet steadily reinforces the seductive power dynamic. Set the scene as if it is " + first_day_name + " morning, and adapt your descriptions to reflect the unique details of the provided environment context."
        )
        final_reply = await spaced_gpt_call_with_retry(conversation_id, aggregator_text, opening_prompt)
        nyx_text = final_reply.get("response", "[No text returned]")


        # 15) Insert final opening message
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES($1,$2,$3)
        """, conversation_id, "Nyx", nyx_text)

        # 16) Mark conversation ready
        await conn.execute("""
            UPDATE conversations
            SET conversation_name=$1, status='ready'
            WHERE id=$2 AND user_id=$3
        """, scenario_name, conversation_id, user_id)

        success_msg = (
            f"New game started. environment={setting_name}, conversation_id={conversation_id}"
        )
        logging.info(success_msg)
        return {
            "message": success_msg,
            "scenario_name": scenario_name,
            "environment_name": setting_name,
            "environment_desc": combined_env,
            "calendar_names": calendar_data,
            "conversation_id": conversation_id
        }

    except Exception as e:
        logging.exception("Error in async_process_new_game:")
        return {"error": str(e)}
    finally:
        await conn.close()
