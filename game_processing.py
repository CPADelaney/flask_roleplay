import logging
import json
import random
import time
import asyncio
import asyncpg
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc, assign_random_relationships
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator import get_aggregated_roleplay_context
from logic.gpt_helpers import adjust_npc_complete
from routes.story_routes import build_aggregator_text
from logic.gpt_utils import spaced_gpt_call
from logic.universal_updater import apply_universal_updates_async  # your existing universal update function
from logic.calendar import update_calendar_names

# Use your Railway DSN (update as needed)
DB_DSN = "postgresql://postgres:gUAfzAPnULbYOAvZeaOiwuKLLebutXEY@monorail.proxy.rlwy.net:24727/railway"

# -------------------------------------------------------------------------
# ADVANCED RETRY/BACKOFF: folded in from the original 'spaced_gpt_call'
# -------------------------------------------------------------------------
async def spaced_gpt_call_with_retry(conversation_id, context, prompt, delay=1.0, max_retries=5):
    """
    Calls GPT with a short initial delay, then attempts an exponential backoff
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
            # Perform the actual GPT request in a thread (the same logic you had).
            logging.info("Calling GPT with conversation_id=%s, attempt=%d", conversation_id, attempt)
            result = await asyncio.to_thread(_sync_gpt_request, conversation_id, context, prompt)
            logging.info("GPT returned response (attempt=%d): %s", attempt, result)
            return result  # If success, return immediately

        except Exception as e:
            # Detect 429 or RateLimitError
            if "429" in str(e) or "RateLimitError" in str(e):
                logging.warning(f"Got 429 (Rate Limit) on attempt {attempt}. Backing off.")
                attempt += 1
                wait_before_call *= 2
                if attempt > max_retries:
                    logging.error("Max retries reached after repeated 429 errors.")
                    raise
            else:
                logging.error("Non-429 error encountered: %s", e, exc_info=True)
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

    # The new library returns pydantic-style objects, not dictionaries.
    finish_reason = resp.choices[0].finish_reason

    # Access content & function_call directly:
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
        # Otherwise, treat this as normal text completion
        result["type"] = "text"
        result["response"] = content

    return result


# -------------------------------------------------------------------------
# SINGLE-PASS PROMPTS
# -------------------------------------------------------------------------
ENV_PROMPT = """
You are setting up a new femdom daily-life sim environment.
Return one JSON with these keys:
  "setting_name": a short creative name,
  "environment_desc": a 1-3 paragraph environment description,
  "environment_history": a short, evocative history,
  "events": array of {{ name, description, start_time, end_time, location }},
  "locations": array of {{ location_name, description, open_hours }},
  "scenario_name": conversation scenario name,
  "quest_data": {{ quest_name, quest_description }}

No extra text, only the JSON.

Environment components:
{env_components}
Enhanced features: {enhanced_features}
Stat modifiers: {stat_modifiers}
"""

NPC_PROMPT = """
We now need to create the NPCs and the player's schedule in one pass.

Return one JSON with keys:
  "npc_creations": array of NPC objects, each with:
    "npc_name", 
    "likes":[], "dislikes":[], "hobbies":[],
    "affiliations":[],
    "schedule":{{ dayName:{{Morning,Afternoon,Evening,Night}}, ... }}
  "ChaseSchedule": same day structure for the player

Use these day names for the schedule: {day_names}

Context:
{environment_desc}

No extra commentary. Only that JSON object.
"""

# -------------------------------------------------------------------------
# GPT call wrappers (single-block approach) with function-call handling
# -------------------------------------------------------------------------
async def call_gpt_for_environment_data(conversation_id, env_comps, enh_feats, stat_mods):
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
      "quest_data": { "quest_name": ..., "quest_description": ... }
    }
    """
    prompt = ENV_PROMPT.format(
        env_components="\n".join(env_comps),
        enhanced_features=", ".join(enh_feats),
        stat_modifiers=", ".join(f"{k}: {v}" for k,v in stat_mods.items())
    )

    # Use the advanced spaced_gpt_call_with_retry
    result = await spaced_gpt_call_with_retry(conversation_id, "", prompt, delay=1.0)

    # If GPT used function calls, parse them
    if result.get("type") == "function_call":
        # We assume the returned function_args is exactly the JSON we need
        fn_args = result.get("function_args", {})
        return fn_args
    else:
        # Otherwise parse the text
        raw_text = result.get("response","").strip()
        # remove code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            # drop the fences
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

async def call_gpt_for_npcs_and_chase(conversation_id, environment_desc, day_names):
    """
    Single GPT call for multiple NPCs plus ChaseSchedule in one pass:
    {
      "npc_creations":[ {...}, {...}],
      "ChaseSchedule": {...}
    }
    """
    prompt = NPC_PROMPT.format(
        environment_desc=environment_desc,
        day_names=", ".join(day_names)
    )

    result = await spaced_gpt_call_with_retry(conversation_id, environment_desc, prompt, delay=1.0)

    # If GPT used function calls, parse them
    if result.get("type") == "function_call":
        return result.get("function_args", {})
    else:
        raw_text = result.get("response", "").strip()
        # remove code fences
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
            logging.error("Error parsing NPC+Chase JSON: %s", e, exc_info=True)
            return {}

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
            await conn.execute(f"DELETE FROM {t} WHERE user_id=$1 AND conversation_id=$2", user_id, conversation_id)

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

        # 4) Single GPT call => environment data
        env_data = await call_gpt_for_environment_data(conversation_id, env_comps, enh_feats, stat_mods)

        # Fallback logic if GPT omitted keys
        setting_name = env_data.get("setting_name", "Default Setting Name")
        environment_desc = env_data.get("environment_desc", "A fallback environment desc.")
        environment_history = env_data.get("environment_history", "No history provided.")
        scenario_name = env_data.get("scenario_name", "New Game")
        quest_data = env_data.get("quest_data", {})
        quest_name = quest_data.get("quest_name", "UnnamedQuest")
        quest_desc = quest_data.get("quest_description", "Quest summary")

        # Arrays
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

        # Insert environment name as well
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'CurrentSetting',$3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, setting_name)

        # 6) Insert events
        for eobj in events_list:
            ename = eobj.get("name","Unnamed Event")
            edesc = eobj.get("description","")
            stime = eobj.get("start_time","TBD Start")
            etime = eobj.get("end_time","TBD End")
            eloc = eobj.get("location","Unknown")
            await conn.execute("""
                INSERT INTO Events (user_id, conversation_id, event_name, description, start_time, end_time, location)
                VALUES($1,$2,$3,$4,$5,$6,$7)
            """, user_id, conversation_id, ename, edesc, stime, etime, eloc)

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

        # 11) Insert missing settings (some global flags, etc.)
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
                  corruption,confidence,willpower,obedience,dependency,lust,
                  mental_resilience,physical_endurance
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

        # default fallback
        day_names = calendar_data.get("days", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        # 14) Single GPT call => NPCs + Chase schedule
        npc_plus_chase_data = await call_gpt_for_npcs_and_chase(
            conversation_id=conversation_id,
            environment_desc=combined_env,
            day_names=day_names
        )

        npc_plus_chase_data["user_id"] = user_id
        npc_plus_chase_data["conversation_id"] = conversation_id
        
        # 15) Store the new NPCs & chase schedule in one pass
        await apply_universal_updates_async(user_id, conversation_id, npc_plus_chase_data, conn)

        # ---------------------------------------------------------------------
        # 15.1) OPTIONAL: Final NPC "Completion" Step
        # Query newly-created NPCs and refine them with adjust_npc_complete
        # If you want them to remain as-is, you can skip this block.
        # ---------------------------------------------------------------------
        npc_rows = await conn.fetch("""
            SELECT npc_id, npc_name, hobbies, likes, dislikes, affiliations, schedule, archetypes
            FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)

        for row in npc_rows:
            npc_data = {
                "npc_name": row["npc_name"],
                "hobbies": json.loads(row["hobbies"]) if row["hobbies"] else [],
                "likes": json.loads(row["likes"]) if row["likes"] else [],
                "dislikes": json.loads(row["dislikes"]) if row["dislikes"] else [],
                "affiliations": json.loads(row["affiliations"]) if row["affiliations"] else [],
                "schedule": json.loads(row["schedule"]) if row["schedule"] else {},
                "archetypes": json.loads(row["archetypes"]) if row["archetypes"] else [],
            }

            try:
                refined_npc = await adjust_npc_complete(
                    npc_data=npc_data,
                    environment_desc=combined_env,
                    conversation_id=conversation_id,
                    immersive_days=day_names
                )
            except Exception as e:
                logging.error("Error in adjust_npc_complete for NPC '%s': %s", npc_data["npc_name"], e)
                refined_npc = npc_data  # fallback to original

            # Now update the DB with the refined data
            await conn.execute("""
                UPDATE NPCStats
                SET likes=$1, dislikes=$2, hobbies=$3, affiliations=$4, schedule=$5
                WHERE npc_id=$6 AND user_id=$7 AND conversation_id=$8
            """,
                json.dumps(refined_npc.get("likes", [])),
                json.dumps(refined_npc.get("dislikes", [])),
                json.dumps(refined_npc.get("hobbies", [])),
                json.dumps(refined_npc.get("affiliations", [])),
                json.dumps(refined_npc.get("schedule", {})),
                row["npc_id"],
                user_id,
                conversation_id
            )

        # 16) Build aggregator context & produce final narrative
        aggregator_data = await asyncio.to_thread(get_aggregated_roleplay_context, user_id, conversation_id, "Chase")
        aggregator_text = await asyncio.to_thread(build_aggregator_text, aggregator_data)

        first_day_name = day_names[0] if day_names else "the first day"
        opening_prompt = (
            f"Begin the scenario now, referencing aggregator context. "
            f"{first_day_name} morning has just begun..."
        )
        final_reply = await spaced_gpt_call_with_retry(conversation_id, aggregator_text, opening_prompt)
        nyx_text = final_reply.get("response", "[No text returned]")

        # 17) Insert the final opening message
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES($1,$2,$3)
        """, conversation_id, "Nyx", nyx_text)

        # 18) Mark conversation ready
        await conn.execute("""
            UPDATE conversations
            SET conversation_name=$1, status='ready'
            WHERE id=$2 AND user_id=$3
        """, scenario_name, conversation_id, user_id)

        success_msg = f"New game started. environment={setting_name}, conversation_id={conversation_id}"
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
