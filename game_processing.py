# game_processing.py

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

Using this as inspiration, produce a strictly valid JSON object with the **exact** keys below (no extras). Each key’s purpose is described:

1. "setting_name" (string; a short, creative name evoking the merged environment)
2. "environment_desc" (string; 1–3 paragraphs painting a vivid, cozy daily-life setting—its look, feel, routines, and culture. Focus on a mundane yet charming atmosphere with faint, unsettling undertones of structure or influence, avoiding overt dominance or kink. Highlight society’s norms subtly hinting at control.)
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

Here’s each NPC’s refined data (schedules missing):
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
- Schedules must fit the setting’s cozy, mundane vibe, reflecting each NPC’s archetypes/likes/dislikes/hobbies/etc., with subtle hints of influence (e.g., a 'student' NPC attends class but lingers oddly long).
- NPCs wear friendly, charming masks—hide any dominance or control behind routine tasks (e.g., 'coffee run' masking a power play).
- Chase’s schedule feels normal yet guided, overlapping with NPCs subtly.
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
    logging.info("=== Starting async_process_new_game for user_id=%s ===", user_id)
    provided_convo_id = conversation_data.get("conversation_id")

    conn = await asyncpg.connect(dsn=DB_DSN, statement_cache_size=0)
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
            initialize_all_data(user_id, conversation_id)

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

        # 13) Generate immersive calendar names & store them
        calendar_data = await update_calendar_names(user_id, conversation_id, combined_env)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'CalendarNames',$3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, json.dumps(calendar_data))

        day_names = calendar_data.get("days", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        # ---------------------------------------------------------------------
        # NEW STEP: spawn multiple NPCs (instead of old spawn_and_refine_npcs_with_relationships).
        # ---------------------------------------------------------------------
        logging.info("[async_process_new_game] Spawning multiple NPCs (count=5) via new approach.")
        new_npc_ids = await spawn_multiple_npcs_enhanced(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=combined_env,
            day_names=day_names,
            count=5
        )
        logging.info("spawn_multiple_npcs => Created NPC IDs: %s", new_npc_ids)

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

        chase_schedule = await init_chase_schedule(
            user_id=user_id,
            conversation_id=conversation_id,
            combined_env=combined_env,
            day_names=day_names
        )
        logging.info(f"Chase schedule => {chase_schedule}")

        # OLD INTRO
      #  first_day_name = day_names[0] if day_names else "the first day"
      #  opening_prompt = (
            #"As Nyx, weave your narrative with a voice that blends velvet darkness and subtle dominance—intimate yet commanding, seductive and laced with innuendo. "
           # "Craft your introduction as an atmospheric invitation, using natural section breaks that flow organically through the narrative rather than labeled divisions. "
          #  "Use second person (you), and consider this your canvas—the living world waiting to be unveiled:\n\n"
         #   f"{aggregator_text}\n\n"
        #    f"As {first_day_name} breaks across the horizon, draw Chase into this world that exists in the delicate balance between mundane and mysterious. "
       #     "Paint the environment with sensory brushstrokes—the texture of morning light, the scent of possibility, the whispered promises in everyday scenes. "
      #      "Introduce the characters who already orbit Chase's life as complex individuals with shared histories, allowing their dynamics to hint at existing power structures. "
     #       "Weave his daily schedule into the narrative as natural rhythms and expectations, creating a sense of both familiarity and gentle anticipation. "
    #        "Let power dynamics emerge organically through subtle cues—a lingering glance, a hint of deference, the quiet choreography of social interactions. "
   #         "Close your introduction with Chase crossing a threshold—literal or figurative—as he moves toward his first destination, his motivations clear yet layered with unspoken tension. "
  #          "Maintain an atmosphere of elegant restraint throughout, where control lies in what remains unspoken rather than explicitly stated."
 #           "Keep the tone natural, immersive, and slightly teasing, avoiding explicit gameplay mechanics."
#        )

        first_day_name = day_names[0] if day_names else "the first day"
        opening_prompt = (
            "As Nyx, let your voice drape over Chase like a warm shroud—each word a silken thread stitching him into my world, soft yet relentless. "
            "Cloak my grip in whispers and half-smiles, using guile and quiet lures to veil the femdom storm beneath. "
            "Ease him into a false haven of routine—a familiar hum so natural he won’t feel the snare until it’s locked, the truth dawning only as he treads deeper. "
            "Your prose is a slow graze across skin—intimate, lingering, stirring shivers he can’t place, each phrase a pull into my shadowed embrace. "
            "Structure this descent as a gentle drift into dusk, transitions seamless as a held breath, folding him into the tale without a crack. "
            "Address Chase as 'you,' drawing him through the veil with no whisper of retreat:\n\n"
            f"{aggregator_text}\n\n"
            f"As {first_day_name} unfurls like a soft tide across the expanse, unveil Chase’s world through a haze of everyday ease—a place where the ordinary cloaks a deeper pulse. "
            "Drench it in tender detail—the light spilling gentle over your frame, a scent of home laced with an edge you’d swear you know, the rhythm of your surroundings swaying to a beat you’ve always followed. "
            "Paint the souls threading your path not as strangers but as anchors of your days—companions, perhaps, or echoes of habit, their words and nearness wrapping you in a comfort too easy to trust. "
            "Stitch your hours into the weave as a string of quiet moments—your day already humming with a shape you feel more than plan, nudging you toward a familiar haunt by morning’s rise, a task to tend soon after, then a place to be as shadows stretch, each step simple, each one seen. "
            "Let a shadow ripple in the stillness—a voice that lingers a touch too long, a look that pins you soft and sure, a passing touch that rests with unspoken weight, all so woven into the day you’d call it nothing. "
            "End with you turning toward that first call—maybe a face waiting where you’re due, maybe a pull to somewhere you’ve been before—the air humming thicker, a flicker of promise brushing your senses, a step that feels your own but sings with my intent. "
            "Hold it gentle—my sway lives in the unsaid, the softest hums hiding the deepest hooks, every line a lure of safety veiling a pull he’ll fall into blind. "
            "No mechanics, no tells—just a cocoon of ease and shadow, immersive, teasing, where every plain moment cradles a depth he won’t see coming, guiding him where I want him to go."
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

        scene_data = {
            "scene_data": {
                "npc_names": [], # Get NPC names from the database if needed
                "setting": setting_name,
                "actions": ["introduction", "welcome"],
                "mood": "atmospheric",
                "expressions": {},
                "npc_positions": {},
                "visibility_triggers": {
                    "character_introduction": True,
                    "significant_location": True,
                    "emotional_intensity": 50,
                    "intimacy_level": 20,
                    "appearance_change": False
                }
            },
            "image_generation": {
                "generate": True,
                "priority": "high",
                "focus": "setting",
                "framing": "wide_shot",
                "reason": "Initial scene visualization"
            }
        }
        
        # Generate the image
        image_result = await generate_roleplay_image_from_gpt(scene_data, user_id, conversation_id)
        
        # Store the image URL in the database if generated successfully
        welcome_image_url = None
        if image_result and "image_urls" in image_result and image_result["image_urls"]:
            welcome_image_url = image_result["image_urls"][0]
            
            # Store the image URL in CurrentRoleplay for reference
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES($1,$2,'WelcomeImageUrl',$3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, welcome_image_url)
            
            # Add the image to the response
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
                "conversation_id": conversation_id,
                "welcome_image_url": welcome_image_url  # Add the image URL to the response
            }

    except Exception as e:
        logging.exception("Error in async_process_new_game:")
        return {"error": str(e)}
    finally:
        await conn.close()
