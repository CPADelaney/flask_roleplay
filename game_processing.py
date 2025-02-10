import logging
import json
import random
import time
import asyncio
import asyncpg
import os
import openai
import httpx
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator import get_aggregated_roleplay_context
from logic.gpt_helpers import adjust_npc_complete
from routes.story_routes import build_aggregator_text
from logic.gpt_utils import spaced_gpt_call
from logic.npc_creation import assign_random_relationships
from logic.calendar import update_calendar_names

# Use your Railway DSN (update as needed)
DB_DSN = "postgresql://postgres:gUAfzAPnULbYOAvZeaOiwuKLLebutXEY@monorail.proxy.rlwy.net:24727/railway"

# ---------------------------------------------------------------------
# Helper: spaced-out GPT call.
async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0):
    logging.info("Waiting for %.1f seconds before calling GPT (conversation_id=%s)", delay, conversation_id)
    await asyncio.sleep(delay)
    logging.info("Calling GPT with conversation_id=%s", conversation_id)
    result = await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)
    logging.info("GPT returned response: %s", result)
    return result

# ---------------------------------------------------------------------
# Helper: retry call upon failure

async def call_gpt_with_retry(func, expected_keys: set, retries: int = 3, initial_delay: float = 1, **kwargs):
    """
    Calls a GPT helper function (an async function) with the given keyword arguments.
    Checks that the returned dictionary contains the expected keys.
    Retries the call (with exponential backoff) up to `retries` times if keys are missing or an exception occurs.
    """
    delay = initial_delay
    for attempt in range(1, retries + 1):
        try:
            result = await func(**kwargs)
            # Check which keys are missing
            missing_keys = expected_keys - set(result.keys())
            if missing_keys:
                logging.warning("Attempt %d: %s returned missing keys: %s", attempt, func.__name__, missing_keys)
                raise ValueError("Missing keys: " + ", ".join(missing_keys))
            logging.info("Attempt %d: %s returned all expected keys.", attempt, func.__name__)
            return result
        except Exception as e:
            logging.error("Attempt %d: Error calling %s: %s", attempt, func.__name__, e, exc_info=True)
            if attempt < retries:
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logging.error("Max retries reached for %s", func.__name__)
                raise

###############################################################################
# JSON Schemas for structured outputs
###############################################################################

ENV_NAME_SCHEMA = {
    "name": "EnvironmentName",
    "strict": True,
    "schema": {
        "type":"object",
        "properties":{
            "setting_name":{"type":"string"}
        },
        "required":["setting_name"],
        "additionalProperties":False
    }
}

ENV_DESC_SCHEMA = {
    "name":"EnvironmentDesc",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "setting_desc":{"type":"string"}
        },
        "required":["setting_desc"],
        "additionalProperties":False
    }
}

ENV_HISTORY_SCHEMA = {
    "name":"EnvironmentHistory",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "history":{"type":"string"}
        },
        "required":["history"],
        "additionalProperties":False
    }
}

EVENTS_SCHEMA = {
    "name":"EnvironmentEvents",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "events":{
                "type":"array",
                "items":{
                    "type":"object",
                    "properties":{
                        "name":{"type":"string"},
                        "description":{"type":"string"},
                        "start_time":{"type":"string"},
                        "end_time":{"type":"string"},
                        "location":{"type":"string"}
                    },
                    "required":["name","description","start_time","end_time","location"],
                    "additionalProperties":False
                }
            }
        },
        "required":["events"],
        "additionalProperties":False
    }
}

LOCATIONS_SCHEMA = {
    "name":"EnvironmentLocations",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "locations":{
                "type":"array",
                "items":{
                    "type":"object",
                    "properties":{
                        "location_name":{"type":"string"},
                        "description":{"type":"string"},
                        "open_hours":{"type":"array","items":{"type":"string"}}
                    },
                    "required":["location_name","description","open_hours"],
                    "additionalProperties":False
                }
            }
        },
        "required":["locations"],
        "additionalProperties":False
    }
}

PLAYER_ROLE_SCHEMA = {
    "name":"PlayerRole",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "player_role":{"type":"string"}
        },
        "required":["player_role"],
        "additionalProperties":False
    }
}

MAIN_QUEST_SCHEMA = {
    "name":"MainQuest",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "quest_name":{"type":"string"},
            "progress_detail":{"type":"string"}
        },
        "required":["quest_name","progress_detail"],
        "additionalProperties":False
    }
}

# For "ChaseSchedule", we define an object: { "schedule": { "Alpha": { "Morning": "...", ...}, "Beta": {...}, ... } }
# That means the top-level has "schedule" as a required key.
# Inside "schedule", each day is an object with Morning/Afternoon/Evening/Night. 
# We'll use "additionalProperties" so day-names can vary.
CHASE_SCHEDULE_SCHEMA = {
    "name":"ChaseScheduleSchema",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "schedule":{
                "type":"object",
                "patternProperties":{
                    "^.*$":{
                        "type":"object",
                        "properties":{
                            "Morning":{"type":"string"},
                            "Afternoon":{"type":"string"},
                            "Evening":{"type":"string"},
                            "Night":{"type":"string"}
                        },
                        "required":["Morning","Afternoon","Evening","Night"],
                        "additionalProperties":False
                    }
                },
                "additionalProperties":False
            }
        },
        "required":["schedule"],
        "additionalProperties":False
    }
}

# For the final opening narrative from Nyx, let's store it in a single "narrative" string.
OPENING_SCHEMA = {
    "name":"OpeningNarrative",
    "strict":True,
    "schema":{
        "type":"object",
        "properties":{
            "narrative":{"type":"string"}
        },
        "required":["narrative"],
        "additionalProperties":False
    }
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_4O_MODEL = "gpt-4o"

async def call_gpt_structured(
    system_prompt:str,
    user_prompt:str,
    schema:dict,
    max_retries:int=3,
    temperature:float=0.7
):
    """
    Calls GPT with the provided system + user prompt and a structured response_format (json_schema).
    Returns the parsed object on success, or None on refusal/failure.
    """
    request_body = {
        "model": GPT_4O_MODEL,
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        "response_format":{
            "type":"json_schema",
            "json_schema":schema
        },
        "temperature": temperature
    }

    for attempt in range(1, max_retries+1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type":"application/json"
                    },
                    json=request_body
                )
                resp.raise_for_status()
                data = resp.json()
            choice = data["choices"][0]["message"]
            if "refusal" in choice:
                logging.warning(f"[call_gpt_structured] refusal attempt {attempt}: {choice['refusal']}")
                continue
            if "parsed" in choice:
                return choice["parsed"]
            logging.warning(f"[call_gpt_structured] attempt {attempt}: no 'parsed' found. Full msg => {choice}")
        except Exception as e:
            logging.error(f"[call_gpt_structured] attempt {attempt} error: {e}", exc_info=True)
    return None  # fallback if we never succeed

# Original helper to query stored value
async def get_stored_value(conn, user_id, conversation_id, key):
    row = await conn.fetchrow(
        "SELECT value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key=$3",
        user_id, conversation_id, key
    )
    if row:
        return row["value"]
    return None

# ---------------------------------------------------------------------
# Universal update function (asynchronous version)
async def apply_universal_update(user_id, conversation_id, update_data, conn):
    logging.info("=== [apply_universal_update] START ===")
    logging.info("Update data keys: %s", list(update_data.keys()))
    logging.info("Full update data:\n%s", json.dumps(update_data, indent=2))
    
    # NEW: Check for a top-level "ChaseSchedule" key and store it if found.
    if "ChaseSchedule" in update_data:
        chase_schedule_value = update_data["ChaseSchedule"]
        logging.info("Storing ChaseSchedule from update data.")
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'ChaseSchedule', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, json.dumps(chase_schedule_value))
    
    # 1) roleplay_updates
    rp_updates = update_data.get("roleplay_updates", {})
    logging.info("[apply_universal_update] roleplay_updates: %s", rp_updates)
    # If rp_updates is a list, merge them into a single dictionary.
    if isinstance(rp_updates, list):
        merged_rp_updates = {}
        for d in rp_updates:
            if isinstance(d, dict):
                merged_rp_updates.update(d)
        rp_updates = merged_rp_updates
    for key, val in rp_updates.items():
        stored_val = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, key, stored_val)
        logging.info("Inserted/Updated CurrentRoleplay: key=%s, value=%s", key, stored_val)

    
    # 2) npc_creations
    npc_creations = update_data.get("npc_creations", [])
    logging.info("[apply_universal_update] npc_creations: %s", npc_creations)
    for npc_data in npc_creations:
        name = npc_data.get("npc_name", "Unnamed NPC")
        introduced = npc_data.get("introduced", False)
        arche = npc_data.get("archetypes", [])
        dom = npc_data.get("dominance", 0)
        cru = npc_data.get("cruelty", 0)
        clos = npc_data.get("closeness", 0)
        tru = npc_data.get("trust", 0)
        resp = npc_data.get("respect", 0)
        inten = npc_data.get("intensity", 0)
        hbs = npc_data.get("hobbies", [])
        pers = npc_data.get("personality_traits", [])
        lks = npc_data.get("likes", [])
        dlks = npc_data.get("dislikes", [])
        affil = npc_data.get("affiliations", [])
        sched = npc_data.get("schedule", {})
        mem = npc_data.get("memory", [])
        if isinstance(mem, str):
            mem = [mem]
        monica_lvl = npc_data.get("monica_level", 0)
        sex = npc_data.get("sex", None)
        
        row = await conn.fetchrow("""
            SELECT npc_id FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2 AND LOWER(npc_name)=$3
            LIMIT 1
        """, user_id, conversation_id, name.lower())
        if row:
            logging.info("Skipping NPC creation, '%s' already exists.", name)
            continue
        logging.info("Creating NPC: %s, introduced=%s, dominance=%s, cruelty=%s", name, introduced, dom, cru)
        await conn.execute("""
            INSERT INTO NPCStats (
                user_id, conversation_id, npc_name, introduced,
                archetypes, dominance, cruelty, closeness, trust, respect, intensity,
                hobbies, personality_traits, likes, dislikes,
                affiliations, schedule, memory, monica_level, sex
            )
            VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15,
                $16, $17, $18, $19, $20
            )
        """, user_id, conversation_id, name, introduced, json.dumps(arche),
           dom, cru, clos, tru, resp, inten,
           json.dumps(hbs), json.dumps(pers), json.dumps(lks), json.dumps(dlks),
           json.dumps(affil), json.dumps(sched), json.dumps(mem),
           monica_lvl, sex)
        logging.info("NPC creation insert complete for %s", name)
    
    # 3) npc_updates
    npc_updates = update_data.get("npc_updates", [])
    logging.info("[apply_universal_update] npc_updates: %s", npc_updates)
    for up in npc_updates:
        # First try to get npc_id from the payload
        npc_id = up.get("npc_id")
        # If npc_id is 0 or falsy, attempt a lookup by npc_name
        if not npc_id or npc_id == 0:
            npc_name = up.get("npc_name")
            if npc_name:
                logging.warning("Provided npc_id is invalid (0); looking up NPCStats for npc_name=%s", npc_name)
                row = await conn.fetchrow(
                    "SELECT npc_id FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_name=$3",
                    user_id, conversation_id, npc_name
                )
                if row:
                    npc_id = row["npc_id"]
                    logging.info("Lookup successful: found npc_id=%s for npc_name=%s", npc_id, npc_name)
                else:
                    logging.warning("No NPC found in NPCStats for npc_name=%s; skipping update.", npc_name)
                    continue
            else:
                logging.warning("Skipping npc_update: missing both npc_id and npc_name.")
                continue
    
        # (Optional: add a check to ensure npc_id != user_id if needed)
        if npc_id == user_id:
            logging.warning("Skipping NPC update for npc_id=%s as it matches the player id.", npc_id)
            continue
    
        fields_map = {
            "npc_name": "npc_name",
            "introduced": "introduced",
            "dominance": "dominance",
            "cruelty": "cruelty",
            "closeness": "closeness",
            "trust": "trust",
            "respect": "respect",
            "intensity": "intensity",
            "monica_level": "monica_level",
            "sex": "sex"
        }
        set_clauses = []
        set_vals = []
        for field_key, db_col in fields_map.items():
            if field_key in up:
                set_clauses.append(f"{db_col} = $%d" % (len(set_vals) + 1))
                set_vals.append(up[field_key])
        if set_clauses:
            set_str = ", ".join(set_clauses)
            set_vals.extend([npc_id, user_id, conversation_id])
            query = f"""
                UPDATE NPCStats
                SET {set_str}
                WHERE npc_id=$%d AND user_id=$%d AND conversation_id=$%d
            """ % (len(set_vals)-2, len(set_vals)-1, len(set_vals))
            logging.info("Updating NPC %s with fields %s", npc_id, set_clauses)
            await conn.execute(query, *set_vals)
            logging.info("Updated NPC %s", npc_id)
        if "memory" in up:
            new_mem_entries = up["memory"]
            if isinstance(new_mem_entries, str):
                new_mem_entries = [new_mem_entries]
            logging.info("Appending memory to NPC %s: %s", npc_id, new_mem_entries)
            # Convert the list to a JSON string.
            memory_json = json.dumps(new_mem_entries)
            await conn.execute("""
                UPDATE NPCStats
                SET memory = COALESCE(memory, '[]'::jsonb) || $1::jsonb
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
            """, memory_json, npc_id, user_id, conversation_id)
        if "schedule" in up:
            new_schedule = up["schedule"]
            logging.info("Overwriting schedule for NPC %s: %s", npc_id, new_schedule)
            await conn.execute("""
                UPDATE NPCStats
                SET schedule=$1
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
            """, json.dumps(new_schedule), npc_id, user_id, conversation_id)
        if "schedule_updates" in up:
            partial_sched = up["schedule_updates"]
            logging.info("Merging schedule_updates for NPC %s: %s", npc_id, partial_sched)
            row = await conn.fetchrow("""
                SELECT schedule FROM NPCStats
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, npc_id, user_id, conversation_id)
            if row:
                existing_schedule = row[0] or {}
                for day_key, times_map in partial_sched.items():
                    if day_key not in existing_schedule:
                        existing_schedule[day_key] = {}
                    existing_schedule[day_key].update(times_map)
                await conn.execute("""
                    UPDATE NPCStats
                    SET schedule=$1
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                """, json.dumps(existing_schedule), npc_id, user_id, conversation_id)
        if "affiliations" in up:
            logging.info("Updating affiliations for NPC %s: %s", npc_id, up["affiliations"])
            await conn.execute("""
                UPDATE NPCStats
                SET affiliations = $1
                WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
            """, json.dumps(up["affiliations"]), npc_id, user_id, conversation_id)
    
        if "current_location" in up:
            logging.info("Updating current location for NPC %s: %s", npc_id, up["current_location"])
            await conn.execute("""
                UPDATE NPCStats
                SET current_location = $1
                WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
            """, up["current_location"], npc_id, user_id, conversation_id)
    
    # 4) character_stat_updates
    char_update = update_data.get("character_stat_updates", {})
    logging.info("[apply_universal_update] character_stat_updates: %s", char_update)
    if char_update:
        p_name = char_update.get("player_name", "Chase")
        stats = char_update.get("stats", {})
        stat_map = {
            "corruption": "corruption",
            "confidence": "confidence",
            "willpower": "willpower",
            "obedience": "obedience",
            "dependency": "dependency",
            "lust": "lust",
            "mental_resilience": "mental_resilience",
            "physical_endurance": "physical_endurance"
        }
        set_clauses = []
        set_vals = []
        for k, col in stat_map.items():
            if k in stats:
                set_clauses.append(f"{col}=$%d" % (len(set_vals)+1))
                set_vals.append(stats[k])
        if set_clauses:
            set_str = ", ".join(set_clauses)
            set_vals.extend([p_name, user_id, conversation_id])
            query = f"""
                UPDATE PlayerStats
                SET {set_str}
                WHERE player_name=$%d AND user_id=$%d AND conversation_id=$%d
            """ % (len(set_vals)-2, len(set_vals)-1, len(set_vals))
            logging.info("Updating player stats for %s: %s", p_name, stats)
            await conn.execute(query, *set_vals)
    
    # 5) relationship_updates
    rel_updates = update_data.get("relationship_updates", [])
    logging.info("[apply_universal_update] relationship_updates: %s", rel_updates)
    for r in rel_updates:
        npc_id = r.get("npc_id")
        if not npc_id:
            logging.warning("Skipping relationship update: missing npc_id.")
            continue
        aff_list = r.get("affiliations", None)
        if aff_list is not None:
            logging.info("Updating affiliations for NPC %s: %s", npc_id, aff_list)
            await conn.execute("""
                UPDATE NPCStats
                SET affiliations=$1
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
            """, json.dumps(aff_list), npc_id, user_id, conversation_id)
    
    # 6) shared_memory_updates
    shared_memory_updates = update_data.get("shared_memory_updates", [])
    logging.info("[apply_universal_update] shared_memory_updates: %s", shared_memory_updates)
    for sm_update in shared_memory_updates:
        npc_id = sm_update.get("npc_id")
        relationship = sm_update.get("relationship")
        if not npc_id or not relationship:
            logging.warning("Skipping shared memory update: missing npc_id or relationship data.")
            continue
        row = await conn.fetchrow("""
            SELECT npc_name FROM NPCStats
            WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
        """, npc_id, user_id, conversation_id)
        if not row:
            logging.warning("Shared memory update: NPC with id %s not found.", npc_id)
            continue
        npc_name = row[0]
        from logic.memory import get_shared_memory
        shared_memory_text = get_shared_memory(user_id, conversation_id, relationship, npc_name)
        logging.info("Generated shared memory for NPC %s: %s", npc_id, shared_memory_text)
        await conn.execute("""
            UPDATE NPCStats
            SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb($1::text)
            WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
        """, shared_memory_text, npc_id, user_id, conversation_id)
        logging.info("Appended shared memory to NPC %s", npc_id)

    
    # 7) npc_introductions
    npc_intros = update_data.get("npc_introductions", [])
    logging.info("[apply_universal_update] npc_introductions: %s", npc_intros)
    for intro in npc_intros:
        nid = intro.get("npc_id")
        if nid:
            logging.info("Marking NPC %s as introduced", nid)
            await conn.execute("""
                UPDATE NPCStats
                SET introduced=TRUE
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, nid, user_id, conversation_id)
    
    # 8) location_creations
    loc_creations = update_data.get("location_creations", [])
    logging.info("[apply_universal_update] location_creations: %s", loc_creations)
    for loc in loc_creations:
        loc_name = loc.get("location_name", "Unnamed")
        desc = loc.get("description", "")
        open_hours = loc.get("open_hours", [])
        row = await conn.fetchrow("""
            SELECT id FROM Locations
            WHERE user_id=$1 AND conversation_id=$2 AND LOWER(location_name)=$3
            LIMIT 1
        """, user_id, conversation_id, loc_name.lower())
        if row:
            logging.info("Skipping location creation, '%s' already exists.", loc_name)
            continue
        logging.info("Inserting location => location_name=%s, description=%s, open_hours=%s", loc_name, desc, open_hours)
        await conn.execute("""
            INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, conversation_id, loc_name, desc, json.dumps(open_hours))
        logging.info("Inserted location: %s", loc_name)
    
    # 9) event_list_updates
    event_updates = update_data.get("event_list_updates", [])
    logging.info("[apply_universal_update] event_list_updates: %s", event_updates)
    for ev in event_updates:
        # For PlannedEvents, we now expect npc_id, year, month, day, and time_of_day.
        if "npc_id" in ev and "day" in ev and "time_of_day" in ev:
            npc_id = ev["npc_id"]
            year = ev.get("year", 1)
            month = ev.get("month", 1)
            day = ev["day"]
            tod = ev["time_of_day"]
            ov_loc = ev.get("override_location", "Unknown")
            row = await conn.fetchrow("""
                SELECT event_id FROM PlannedEvents
                WHERE user_id=$1 AND conversation_id=$2
                  AND npc_id=$3 AND year=$4 AND month=$5 AND day=$6 AND time_of_day=$7
                LIMIT 1
            """, user_id, conversation_id, npc_id, year, month, day, tod)
            if row:
                logging.info("Skipping planned event creation; year=%s, month=%s, day=%s, time_of_day=%s, npc=%s already exists.", year, month, day, tod, npc_id)
                continue
            logging.info("Inserting PlannedEvent => npc_id=%s, year=%s, month=%s, day=%s, time_of_day=%s, override_loc=%s", npc_id, year, month, day, tod, ov_loc)
            await conn.execute("""
                INSERT INTO PlannedEvents (user_id, conversation_id, npc_id, year, month, day, time_of_day, override_location)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, user_id, conversation_id, npc_id, year, month, day, tod, ov_loc)
        else:
            # For global events, we also expect the full date info.
            ev_name = ev.get("event_name", "UnnamedEvent")
            ev_desc = ev.get("description", "")
            ev_start = ev.get("start_time", "TBD Start")
            ev_end = ev.get("end_time", "TBD End")
            ev_loc = ev.get("location", "Unknown")
            ev_year = ev.get("year", 1)
            ev_month = ev.get("month", 1)
            ev_day = ev.get("day", 1)
            ev_tod = ev.get("time_of_day", "Morning")
            row = await conn.fetchrow("""
                SELECT id FROM Events
                WHERE user_id=$1 AND conversation_id=$2
                  AND LOWER(event_name)=$3
                LIMIT 1
            """, user_id, conversation_id, ev_name.lower())
            if row:
                logging.info("Skipping event creation; '%s' already exists.", ev_name)
                continue
            logging.info("Inserting Event => %s, loc=%s, times=%s-%s, year=%s, month=%s, day=%s, time_of_day=%s", ev_name, ev_loc, ev_start, ev_end, ev_year, ev_month, ev_day, ev_tod)
            await conn.execute("""
                INSERT INTO Events (
                    user_id, conversation_id,
                    event_name, description, start_time, end_time, location,
                    year, month, day, time_of_day
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, user_id, conversation_id, ev_name, ev_desc, ev_start, ev_end, ev_loc, ev_year, ev_month, ev_day, ev_tod)
    
    # 10) inventory_updates
    inv_updates = update_data.get("inventory_updates", {})
    logging.info("[apply_universal_update] inventory_updates: %s", inv_updates)
    if inv_updates:
        p_n = inv_updates.get("player_name", "Chase")
        added = inv_updates.get("added_items", [])
        removed = inv_updates.get("removed_items", [])
        for item in added:
            if isinstance(item, dict):
                item_name = item.get("item_name", "Unnamed")
                item_desc = item.get("item_description", "")
                item_fx   = item.get("item_effect", "")
                category  = item.get("category", "")
                logging.info("Adding item for %s: %s", p_n, item_name)
                await conn.execute("""
                    INSERT INTO PlayerInventory (user_id, conversation_id, player_name, item_name, item_description, item_effect, category, quantity)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 1)
                    ON CONFLICT (user_id, conversation_id, player_name, item_name)
                    DO UPDATE SET quantity = PlayerInventory.quantity + 1
                """, user_id, conversation_id, p_n, item_name, item_desc, item_fx, category)
            elif isinstance(item, str):
                logging.info("Adding item (string) for %s: %s", p_n, item)
                await conn.execute("""
                    INSERT INTO PlayerInventory (user_id, conversation_id, player_name, item_name, quantity)
                    VALUES ($1, $2, $3, $4, 1)
                    ON CONFLICT (user_id, conversation_id, player_name, item_name)
                    DO UPDATE SET quantity = PlayerInventory.quantity + 1
                """, user_id, conversation_id, p_n, item)
        for item in removed:
            if isinstance(item, dict):
                i_name = item.get("item_name")
                if i_name:
                    logging.info("Removing item for %s: %s", p_n, i_name)
                    await conn.execute("""
                        DELETE FROM PlayerInventory
                        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 AND item_name=$4
                    """, user_id, conversation_id, p_n, i_name)
            elif isinstance(item, str):
                logging.info("Removing item for %s: %s", p_n, item)
                await conn.execute("""
                    DELETE FROM PlayerInventory
                    WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 AND item_name=$4
                """, user_id, conversation_id, p_n, item)
    
    # 11) quest_updates
    quest_updates = update_data.get("quest_updates", [])
    logging.info("[apply_universal_update] quest_updates: %s", quest_updates)
    for qu in quest_updates:
        qid = qu.get("quest_id")
        status = qu.get("status", "In Progress")
        detail = qu.get("progress_detail", "")
        qgiver = qu.get("quest_giver", "")
        reward = qu.get("reward", "")
        qname  = qu.get("quest_name", None)
        if qid:
            logging.info("Updating Quest %s: status=%s, detail=%s", qid, status, detail)
            await conn.execute("""
                UPDATE Quests
                SET status=$1, progress_detail=$2, quest_giver=$3, reward=$4
                WHERE quest_id=$5 AND user_id=$6 AND conversation_id=$7
            """, status, detail, qgiver, reward, qid, user_id, conversation_id)
            row = await conn.fetchrow("""
                SELECT 1 FROM Quests WHERE quest_id=$1 AND user_id=$2 AND conversation_id=$3
            """, qid, user_id, conversation_id)
            if not row:
                logging.info("No existing quest with ID %s, inserting new.", qid)
                await conn.execute("""
                    INSERT INTO Quests (user_id, conversation_id, quest_id, quest_name, status, progress_detail, quest_giver, reward)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, user_id, conversation_id, qid, qname, status, detail, qgiver, reward)
        else:
            logging.info("Inserting new quest: %s, status=%s", qname, status)
            await conn.execute("""
                INSERT INTO Quests (user_id, conversation_id, quest_name, status, progress_detail, quest_giver, reward)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, user_id, conversation_id, qname or "Unnamed Quest", status, detail, qgiver, reward)
    
    logging.info("=== [apply_universal_update] Success! ===")
    return {"message": "Universal update successful"}

# ---------------------------------------------------------------------
# The main function, but replaced calls with structured approach
###############################################################################
# The main function: everything is structured
###############################################################################
async def async_process_new_game(user_id, conversation_data):
    logging.info("=== Starting async_process_new_game for user_id=%s with conversation_data=%s ===", user_id, conversation_data)

    provided_conversation_id = conversation_data.get("conversation_id")
    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        # 1) create or validate conversation
        if not provided_conversation_id:
            preliminary_name = "New Game"
            row = await conn.fetchrow("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES ($1, $2)
                RETURNING id
            """, user_id, preliminary_name)
            conversation_id = row["id"]
        else:
            conversation_id = provided_conversation_id
            row = await conn.fetchrow("""
                SELECT id FROM conversations WHERE id=$1 AND user_id=$2
            """, conversation_id, user_id)
            if not row:
                raise Exception(f"Conversation {conversation_id} not found or unauthorized")

        # 2) clear old data
        tables_to_clear = [
            "Events", "PlannedEvents", "PlayerInventory", "Quests",
            "NPCStats", "Locations", "SocialLinks", "CurrentRoleplay"
        ]
        for table in tables_to_clear:
            await conn.execute(f"DELETE FROM {table} WHERE user_id=$1 AND conversation_id=$2", user_id, conversation_id)
        logging.info("Cleared data for conv=%s", conversation_id)

        # 3) generate environment components
        mega_data = await asyncio.to_thread(generate_mega_setting_logic)
        unique_envs = mega_data.get("selected_settings") or mega_data.get("unique_environments") or []
        if not unique_envs:
            unique_envs = [
                "A sprawling cyberpunk metropolis under siege by monstrous clans",
                "Floating archaic ruins steeped in ancient rituals",
                "Futuristic tech hubs that blend magic and machinery"
            ]
        enhanced_features = mega_data.get("enhanced_features", [])
        stat_modifiers = mega_data.get("stat_modifiers", {})

        # 3.1) environment name (structured)
        system_text = "You produce a single JSON with 'setting_name' for the environment."
        user_text = (
            "Environment components:\n" + "\n".join(unique_envs) + "\n"
            "Generate a short creative name for the overall setting. No extra text."
        )
        env_name_obj = await call_gpt_structured(system_text, user_text, ENV_NAME_SCHEMA)
        if not env_name_obj:
            environment_name = "Default Setting Name"
        else:
            environment_name = env_name_obj["setting_name"]
        logging.info(f"Got environment_name => {environment_name}")

        # 3.2) environment desc (structured)
        system_text = "You produce a single JSON with 'setting_desc' describing the environment."
        user_text = (
            f"Environment components: {unique_envs}\n"
            f"Enhanced features: {enhanced_features}\n"
            f"Stat modifiers: {stat_modifiers}\n"
            "Describe in a cohesive narrative how these combine into a unique, dynamic world. "
            "Output only {\"setting_desc\":\"...\"}."
        )
        env_desc_obj = await call_gpt_structured(system_text, user_text, ENV_DESC_SCHEMA)
        if not env_desc_obj:
            base_environment_desc = "An eclectic realm of monstrous societies, futuristic tech, archaic ruins overhead..."
        else:
            base_environment_desc = env_desc_obj["setting_desc"]
        
        # store environment desc
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'EnvironmentDesc', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, base_environment_desc)

        # 3.3) generate some NPCs
        npc_count = 3
        generated_npcs = []
        for i in range(npc_count):
            new_npc_id = await create_npc(user_id, conversation_id, introduced=False)
            generated_npcs.append(new_npc_id)
        logging.info(f"Generated NPCs => {generated_npcs}")

        # 3.4) environment history (structured)
        system_text = "You produce a single JSON with 'history' containing a brief, evocative setting backstory."
        user_text = (
            f"Environment desc: {base_environment_desc}\n"
            f"NPCs: {generated_npcs}\n"
            "Integrate these NPCs into the history. Return only {\"history\":\"...\"}."
        )
        env_hist_obj = await call_gpt_structured(system_text, user_text, ENV_HISTORY_SCHEMA)
        if not env_hist_obj:
            environment_history = "GPT refused or failed, fallback to a generic backstory."
        else:
            environment_history = env_hist_obj["history"]
        environment_desc = f"{base_environment_desc}\n\nHistory: {environment_history}"

        # 3.5) calendar
        calendar_data = await update_calendar_names(user_id, conversation_id, environment_desc)

        # 4) environment events (structured)
        system_text = "You produce a JSON object with 'events': array of event objects."
        user_text = (
            f"Environment description: {environment_desc}\n"
            "Each event => name, description, start_time, end_time, location.\n"
            "Return only {\"events\":[...]}, no extra text."
        )
        events_obj = await call_gpt_structured(system_text, user_text, EVENTS_SCHEMA)
        if not events_obj:
            events_json = []
        else:
            events_json = events_obj["events"]
        # insert events
        for ev in events_json:
            name = ev.get("name","Unnamed Event")
            desc = ev.get("description","")
            stime = ev.get("start_time","TBD Start")
            etime = ev.get("end_time","TBD End")
            loc = ev.get("location","Unknown")
            await conn.execute("""
                INSERT INTO Events (user_id, conversation_id, event_name, description, start_time, end_time, location)
                VALUES($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT DO NOTHING
            """, user_id, conversation_id, name, desc, stime, etime, loc)

        # 5) environment locations (structured)
        system_text = "You produce a JSON object with 'locations': array of location objects."
        user_text = (
            f"Environment description: {environment_desc}\n"
            "Return only {\"locations\":[...]}, no extra text. Each location => location_name, description, open_hours[]."
        )
        loc_obj = await call_gpt_structured(system_text, user_text, LOCATIONS_SCHEMA)
        if not loc_obj:
            locations_json = []
        else:
            locations_json = loc_obj["locations"]
        # insert locations
        for loc in locations_json:
            loc_name = loc["location_name"]
            loc_desc = loc["description"]
            open_hours = loc["open_hours"]
            await conn.execute("""
                INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
                VALUES($1,$2,$3,$4,$5)
                ON CONFLICT DO NOTHING
            """, user_id, conversation_id, loc_name, loc_desc, json.dumps(open_hours))

        # 6) scenario name & quest summary if new conversation
        scenario_name = "New Game"
        quest_blurb = ""
        if not provided_conversation_id:
            scenario_name, quest_blurb = await asyncio.to_thread(
                gpt_generate_scenario_name_and_quest,
                environment_name,
                base_environment_desc
            )
            await conn.execute("""
                UPDATE conversations SET conversation_name=$1 
                WHERE id=$2
            """, scenario_name, conversation_id)

        # 7) store environment_name
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'CurrentSetting',$3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, environment_name)

        # 8) Insert environment data (MegaSettingModifiers).
        modifiers_json = json.dumps(mega_data.get("stat_modifiers", {}))
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'MegaSettingModifiers',$3)
            ON CONFLICT DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, modifiers_json)

        # 9) Insert missing settings
        await asyncio.to_thread(insert_missing_settings)

        # 10) Reset or create 'Chase' in PlayerStats
        await conn.execute("""
            DELETE FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 
              AND player_name <> 'Chase'
        """, user_id, conversation_id)
        chase_row = await conn.fetchrow("""
            SELECT id FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
        """, user_id, conversation_id)
        if chase_row:
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
                  corruption, confidence, willpower,
                  obedience, dependency, lust,
                  mental_resilience, physical_endurance
                )
                VALUES ($1,$2,'Chase',10,60,50,20,10,15,55,40)
            """, user_id, conversation_id)

        # 11) PlayerRole (structured)
        system_text = "You produce a JSON with 'player_role' describing daily life of Chase in a femdom environment."
        user_text = (
            f"Environment desc: {environment_desc}\n"
            "Chase is 31, but can have any in-world occupation. Return only {\"player_role\":\"...\"}"
        )
        role_obj = await call_gpt_structured(system_text, user_text, PLAYER_ROLE_SCHEMA)
        if not role_obj:
            player_role_text = "Chase works a standard office job, scraping by."
        else:
            player_role_text = role_obj["player_role"]
        # store
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'PlayerRole',$3)
            ON CONFLICT DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, player_role_text)

        # 12) MainQuest (structured)
        system_text = "You produce a JSON with 'quest_name' and 'progress_detail' for a mysterious main quest."
        user_text = (
            f"Environment desc: {environment_desc}\n"
            "Chase is one of the only men in a world of dominant females. Summarize the quest. "
            "No extra text, only {\"quest_name\":\"...\",\"progress_detail\":\"...\"}."
        )
        quest_obj = await call_gpt_structured(system_text, user_text, MAIN_QUEST_SCHEMA)
        if not quest_obj:
            main_quest_text = "Embark on a mysterious journey to prove your worth."
            # progress_detail can be blank or something
            progress_detail = ""
        else:
            main_quest_text = quest_obj["quest_name"]
            progress_detail = quest_obj["progress_detail"]
        # store quest
        await conn.execute("""
            INSERT INTO Quests (user_id, conversation_id, quest_name, status, progress_detail, quest_giver, reward)
            VALUES ($1, $2, $3, 'In Progress', $4, '', '')
        """, user_id, conversation_id, main_quest_text, progress_detail)

        # 13) ChaseSchedule (structured)
        # We have a "CHASE_SCHEDULE_SCHEMA" that returns: { "schedule": { "Alpha": {...}, "Beta": {...}, ... } }
        # We'll store that entire object as 'ChaseSchedule'
        # Retrieve day names from 'CalendarNames' to help GPT
        row_cal = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
        """, user_id, conversation_id)
        if row_cal:
            try:
                cal_json = json.loads(row_cal["value"])
                day_names = cal_json.get("days", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            except:
                day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        else:
            day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        system_text = "You produce a JSON with top-level key 'schedule': an object mapping each day to Morning/Afternoon/Evening/Night."
        user_text = (
            f"Environment desc: {environment_desc}\n"
            f"Chase's role: {player_role_text}\n"
            "Use these day names: " + ", ".join(day_names) + ". "
            "No extra text, only {\"schedule\":{\"DayName\":{\"Morning\":\"...\",\"Afternoon\":\"...\",\"Evening\":\"...\",\"Night\":\"...\"},...}}"
        )
        schedule_obj = await call_gpt_structured(system_text, user_text, CHASE_SCHEDULE_SCHEMA)
        if not schedule_obj:
            # fallback
            fallback = {}
            for d in day_names:
                fallback[d] = {
                    "Morning": f"Wake up on {d}, quick breakfast",
                    "Afternoon": f"Go about tasks on {d}",
                    "Evening": f"Socialize or study on {d}",
                    "Night": f"Return home and sleep on {d}"
                }
            chase_schedule = fallback
        else:
            chase_schedule = schedule_obj["schedule"]

        # store schedule
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES($1,$2,'ChaseSchedule',$3)
            ON CONFLICT DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, json.dumps(chase_schedule))

        # 14) Final NPC adjustments same as before, calling adjust_npc_complete with call_gpt_with_retry
        npc_rows = await conn.fetch("""
            SELECT npc_id, npc_name, hobbies, likes, dislikes, affiliations, schedule, archetypes, archetype_summary
            FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)

        actual_immersive_days = day_names  # or from calendar_data
        for row_npc in npc_rows:
            npc_id = row_npc["npc_id"]
            npc_data_dict = {
                "npc_name": row_npc["npc_name"],
                "hobbies": json.loads(row_npc["hobbies"]),
                "likes": json.loads(row_npc["likes"]),
                "dislikes": json.loads(row_npc["dislikes"]),
                "affiliations": json.loads(row_npc["affiliations"]),
                "schedule": json.loads(row_npc["schedule"]),
                "archetypes": json.loads(row_npc["archetypes"]),
                "archetype_summary": row_npc["archetype_summary"] or ""
            }
            try:
                new_npc_data = await call_gpt_with_retry(
                    func=adjust_npc_complete,
                    expected_keys={"likes","dislikes","hobbies","affiliations","schedule"},
                    npc_data=npc_data_dict,
                    environment_desc=base_environment_desc,
                    conversation_id=conversation_id,
                    immersive_days=actual_immersive_days
                )
            except Exception as e:
                logging.error(f"NPC synergy adjustment failed for {npc_id}: {e}")
                new_npc_data = {
                    "likes": npc_data_dict["likes"],
                    "dislikes": npc_data_dict["dislikes"],
                    "hobbies": npc_data_dict["hobbies"],
                    "affiliations": npc_data_dict["affiliations"],
                    "schedule": npc_data_dict["schedule"]
                }
            # update in DB
            await conn.execute("""
                UPDATE NPCStats
                SET likes=$1, dislikes=$2, hobbies=$3, affiliations=$4, schedule=$5
                WHERE npc_id=$6 AND user_id=$7 AND conversation_id=$8
            """,
            json.dumps(new_npc_data["likes"]),
            json.dumps(new_npc_data["dislikes"]),
            json.dumps(new_npc_data["hobbies"]),
            json.dumps(new_npc_data["affiliations"]),
            json.dumps(new_npc_data["schedule"]),
            npc_id, user_id, conversation_id
            )

        # 15) Opening narrative from Nyx (structured)
        # We'll define a short JSON schema that has { "narrative":"some string" }
        # Then store it as a message from Nyx
        system_text = "You produce a single JSON with 'narrative' that is the final opening text from Nyx."
        user_text = (
            "Begin the scenario now, Nyx. Greet Chase with your sadistic, mocking style..."
            f"Announce that {day_names[0]} morning has just begun. Summarize environment, schedule, main quest..."
            "Output only {\"narrative\":\"some text\"} with no extra keys."
        )
        opening_obj = await call_gpt_structured(system_text, user_text, OPENING_SCHEMA)
        if not opening_obj:
            nyx_text = "[Nyx refuses to speak, so here's a fallback opening.]"
        else:
            nyx_text = opening_obj["narrative"]

        # store in messages
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES($1,$2,$3)
        """, conversation_id, "Nyx", nyx_text)

        # finalize conversation to ready
        await conn.execute("""
            UPDATE conversations
            SET conversation_name=$1, status='ready'
            WHERE id=$2 AND user_id=$3
        """, scenario_name, conversation_id, user_id)

        # return
        success_msg = f"New game started. Environment={environment_name}, conversation_id={conversation_id}"
        logging.info(success_msg)
        return {
            "message": success_msg,
            "scenario_name": scenario_name,
            "environment_name": environment_name,
            "environment_desc": environment_desc,
            "calendar_names": calendar_data,
            "conversation_id": conversation_id
        }

    except Exception as e:
        logging.exception("Error in async_process_new_game:")
        return {"error": str(e)}
    finally:
        await conn.close()

###############################################################################
# Helper function for generating scenario name and quest summary (unchanged)
###############################################################################
def gpt_generate_scenario_name_and_quest(env_name: str, env_desc: str):
    client = get_openai_client()
    unique_token = f"{random.randint(1000,9999)}_{int(time.time())}"
    forbidden_words = ["mistress", "darkness", "manor", "chains", "twilight"]

    system_instructions = f"""
    You are setting up a new femdom daily-life sim scenario with a main quest.
    Environment name: {env_name}
    Environment desc: {env_desc}
    Unique token: {unique_token}

    Please produce:
    1) A single line starting with 'ScenarioName:' followed by a short, creative (1–8 words) name 
       that draws from the environment above. 
       Avoid cliche words like {', '.join(forbidden_words)}.
    2) Then one or two lines summarizing the main quest.

    The conversation name must be unique; do not reuse names from older scenarios 
    (you can ensure uniqueness using the token or environment cues).
    """
    messages = [{"role": "system", "content": system_instructions}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.9,
        max_tokens=120,
        frequency_penalty=0.3
    )
    msg = response.choices[0].message.content.strip()
    logging.info(f"[gpt_generate_scenario_name_and_quest] Raw GPT output: {msg}")
    scenario_name = "New Game"
    quest_blurb = ""
    for line in msg.splitlines():
        line = line.strip()
        if line.lower().startswith("scenarioname:"):
            scenario_name = line.split(":", 1)[1].strip()
        else:
            quest_blurb += line + " "
    return scenario_name.strip(), quest_blurb.strip()


# ---------------------------------------------------------------------
# Helper function for generating scenario name and quest summary.
def gpt_generate_scenario_name_and_quest(env_name: str, env_desc: str):
    client = get_openai_client()
    unique_token = f"{random.randint(1000,9999)}_{int(time.time())}"
    forbidden_words = ["mistress", "darkness", "manor", "chains", "twilight"]

    system_instructions = f"""
    You are setting up a new femdom daily-life sim scenario with a main quest.
    Environment name: {env_name}
    Environment desc: {env_desc}
    Unique token: {unique_token}

    Please produce:
    1) A single line starting with 'ScenarioName:' followed by a short, creative (1–8 words) name 
       that draws from the environment above. 
       Avoid cliche words like {', '.join(forbidden_words)}.
    2) Then one or two lines summarizing the main quest.

    The conversation name must be unique; do not reuse names from older scenarios 
    (you can ensure uniqueness using the token or environment cues).
    """
    messages = [{"role": "system", "content": system_instructions}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.9,
        max_tokens=120,
        frequency_penalty=0.3
    )
    msg = response.choices[0].message.content.strip()
    logging.info(f"[gpt_generate_scenario_name_and_quest] Raw GPT output: {msg}")
    scenario_name = "New Game"
    quest_blurb = ""
    for line in msg.splitlines():
        line = line.strip()
        if line.lower().startswith("scenarioname:"):
            scenario_name = line.split(":", 1)[1].strip()
        else:
            quest_blurb += line + " "
    return scenario_name.strip(), quest_blurb.strip()

# ---------------------------------------------------------------------
# spawn_npcs remains unchanged.
async def spawn_npcs():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400

    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        row = await conn.fetchrow("SELECT id FROM conversations WHERE id=$1 AND user_id=$2", conversation_id, user_id)
        if not row:
            return jsonify({"error": "Conversation not found or unauthorized"}), 403

        spawn_tasks = [
            asyncio.to_thread(create_npc, user_id=user_id, conversation_id=conversation_id, introduced=False)
            for _ in range(5)
        ]
        spawned_npc_ids = await asyncio.gather(*spawn_tasks)
        logging.info("Spawned NPCs concurrently: %s", spawned_npc_ids)
        return jsonify({"message": "NPCs spawned", "npc_ids": spawned_npc_ids}), 200
    except Exception as e:
        logging.exception("Error in /spawn_npcs:")
        return jsonify({"error": str(e)}), 500
    finally:
        await conn.close()
