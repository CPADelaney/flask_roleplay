# logic/npc_creation.py

import os
import json
import random
import logging
import httpx
import asyncio
from db.connection import get_db_connection

logging.basicConfig(level=logging.DEBUG)

# Paths & data loads remain same...
current_dir = os.path.dirname(os.path.abspath(__file__))
archetypes_json_path = os.path.join(current_dir, "..", "data", "archetypes_data.json")
archetypes_json_path = os.path.normpath(archetypes_json_path)

DATA_FILES = {
    "hobbies": "data/npc_hobbies.json",
    "likes": "data/npc_likes.json",
    "dislikes": "data/npc_dislikes.json",
    "personalities": "data/npc_personalities.json"
}

def load_data(file_path):
    """Loads JSON from file_path; returns Python data (dict or list)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return {}

DATA = {
    "hobbies": [],
    "likes": [],
    "dislikes": [],
    "personalities": []
}
for key, path in DATA_FILES.items():
    raw = load_data(path)
    if key == "hobbies":
        DATA["hobbies"] = raw.get("hobbies_pool", [])
    elif key == "likes":
        DATA["likes"] = raw.get("npc_likes", [])
    elif key == "dislikes":
        DATA["dislikes"] = raw.get("dislikes_pool", [])
    elif key == "personalities":
        DATA["personalities"] = raw.get("personality_pool", [])

logging.debug("DATA loaded. hobbies => %s", DATA.get("hobbies"))

REROLL_IDS = list(range(62,73))

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def combine_archetype_stats(archetype_rows):
    # Same as your original code...
    sums = {
        "dominance": 0,
        "cruelty": 0,
        "closeness": 0,
        "trust": 0,
        "respect": 0,
        "intensity": 0
    }
    count = len(archetype_rows)
    if count == 0:
        for k in sums.keys():
            sums[k] = random.randint(0, 30)
        return sums

    for (arc_id, arc_name, bs_json) in archetype_rows:
        if isinstance(bs_json, str):
            bs = json.loads(bs_json)
        else:
            bs = bs_json or {}
        for stat_key in sums.keys():
            rng_key = f"{stat_key}_range"
            mod_key = f"{stat_key}_modifier"
            if rng_key in bs and mod_key in bs:
                low, high = bs[rng_key]
                mod = bs[mod_key]
                val = random.randint(low, high) + mod
            else:
                val = random.randint(0, 30)
            sums[stat_key] += val

    for sk in sums.keys():
        sums[sk] = sums[sk] / count
        if sk in ["trust", "respect"]:
            sums[sk] = clamp(int(sums[sk]), -100, 100)
        else:
            sums[sk] = clamp(int(sums[sk]), 0, 100)
    return sums

def archetypes_to_json(rows):
    arr = []
    for (aid, aname, _bs) in rows:
        arr.append({"id": aid, "name": aname})
    return json.dumps(arr)

#################################
# 1) Synergy JSON Schema
#################################
synergy_schema = {
    "name": "NPCSynergy",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "npc_name": {"type": "string"},
            "archetype_summary": {"type": "string"}
        },
        "required": ["npc_name", "archetype_summary"],
        "additionalProperties": False
    }
}

#################################
# 2) Age+Birth JSON Schema
#################################
age_birth_schema = {
    "name": "NPCAgeBirth",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "age": {"type": "integer"},
            "birthdate": {"type": "string"}
        },
        "required": ["age","birthdate"],
        "additionalProperties": False
    }
}

#################################
# 3) Physical Description Schema
#################################
phys_desc_schema = {
    "name": "NPCPhysicalDescription",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string"}
        },
        "required": ["description"],
        "additionalProperties": False
    }
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_4O_MODEL = "gpt-4o-2024-08-06"   # or the snapshot you have

async def call_gpt_structured(system_prompt, user_prompt, schema_dict, max_retries=3):
    """
    A helper that:
      1) calls the Chat Completions endpoint w/ response_format = "json_schema"
      2) checks for refusal or valid "parsed" in message
      3) on success returns that parsed object
      4) on fail or refusal, returns None
    """
    request_body = {
        "model": GPT_4O_MODEL,
        "messages": [
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema_dict
        }
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=request_body
                )
                resp.raise_for_status()
                data = resp.json()

            choice = data["choices"][0]["message"]
            if "refusal" in choice:
                logging.warning(f"Refusal attempt {attempt+1}: {choice['refusal']}")
                continue

            if "parsed" in choice:
                return choice["parsed"]
            else:
                logging.warning(f"Attempt {attempt+1}: No parsed object in the message. Full message => {choice}")
        except Exception as e:
            logging.error(f"call_gpt_structured attempt {attempt+1} error: {e}", exc_info=True)
    # If we reach here => failure
    return None

#################################
# 4) Synergy function
#################################
async def get_archetype_synergy_description(archetypes_list, provided_npc_name=None):
    """
    Calls GPT with structured outputs. We want exactly:
      {
        "npc_name": str,
        "archetype_summary": str
      }
    If it fails or refuses, we fallback.
    """

    if not archetypes_list:
        default_name = provided_npc_name if provided_npc_name else "Unknown"
        return {
            "npc_name": default_name,
            "archetype_summary": "No special archetype synergy."
        }

    archetype_names = [a["name"] for a in archetypes_list]
    if provided_npc_name:
        name_instruction = f"Use the provided NPC name: '{provided_npc_name}'."
    else:
        name_instruction = "Generate a creative, fitting name for the NPC."

    system_text = (
        "You are an expert creative writer, merging multiple archetypes into one NPC. "
        "Output strictly in the JSON schema: npc_name (string), archetype_summary (string)."
    )
    user_text = (
        f"Archetypes: {', '.join(archetype_names)}.\n"
        f"{name_instruction}\n"
        "No extra keys or text. The final JSON must have exactly npc_name and archetype_summary."
    )

    parsed = await call_gpt_structured(system_text, user_text, synergy_schema)
    if parsed is None:
        # fallback
        default_name = provided_npc_name if provided_npc_name else f"NPC_{random.randint(1000,9999)}"
        return {
            "npc_name": default_name,
            "archetype_summary": "Could not generate synergy text."
        }
    return parsed

#################################
# 5) Age & Birthdate function
#################################
async def generate_npc_age_and_birthdate_gpt_structured(npc_name, relationships=None, archetypes=None, current_year=1000):
    """
    Returns a dict with "age" (int) and "birthdate" (str).
    If fail, fallback to random.
    """
    if relationships:
        rel_strings = []
        for rel in relationships:
            # e.g. mother of Alice
            rel_type = rel.get("type","relation")
            target_name = rel.get("target_name","someone")
            rel_strings.append(f"{rel_type} of {target_name}")
        rel_info = "; ".join(rel_strings)
    else:
        rel_info = "None"

    if archetypes:
        if isinstance(archetypes,list):
            arch_info = ", ".join(archetypes)
        else:
            arch_info = str(archetypes)
    else:
        arch_info = "None"

    system_text = (
        "You will produce a JSON with 'age' (integer) and 'birthdate' (YYYY-MM-DD) "
        "for an NPC in a femdom daily-life sim. Must adhere to the schema, no extra keys."
    )
    user_text = (
        f"NPC name: {npc_name}\n"
        f"Current in-game year: {current_year}\n"
        f"Relationships: {rel_info}\n"
        f"Archetypes: {arch_info}\n"
        "If mother, age≥20 yrs older than child. If 'student', keep age young. "
        "Compute birthdate as (current_year - age) with a random month/day. "
        "Strictly return only {\"age\":..., \"birthdate\":\"...\"} with no extra text."
    )

    parsed = await call_gpt_structured(system_text, user_text, age_birth_schema)
    if not parsed:
        # fallback
        fallback_age = random.randint(18,50)
        fallback_month = random.randint(1,12)
        fallback_day = random.randint(1,28)
        fallback_birthdate = f"{current_year - fallback_age}-{fallback_month:02d}-{fallback_day:02d}"
        return {"age": fallback_age, "birthdate": fallback_birthdate}
    return parsed

#################################
# 6) Physical Description function
#################################
async def get_physical_description_structured(npc_name, final_stats, chosen_arcs):
    """
    Returns { "description": "some text" }
    """
    stats_str = ", ".join([f"{k}:{v}" for k,v in final_stats.items()])
    arcs_str = ", ".join([arc["name"] for arc in chosen_arcs]) if chosen_arcs else "None"

    system_text = (
        "Generate a robust, vivid physical description for an NPC in a femdom sim. "
        "Output must be valid JSON with exactly one key 'description'."
    )
    user_text = (
        f"NPC name: {npc_name}\n"
        f"Stats: {stats_str}\n"
        f"Archetypes: {arcs_str}\n"
        "No extra commentary or markdown. Just {\"description\":\"...\"}."
    )

    parsed = await call_gpt_structured(system_text, user_text, phys_desc_schema)
    if not parsed:
        return {"description": "A striking NPC with an enigmatic, captivating appearance."}
    return parsed

###############################
# 7) create_npc - Rewritten
###############################
async def create_npc(user_id, conversation_id, npc_name=None, introduced=False,
                     sex="female", reroll_extra=False, total_archetypes=4, relationships=None):
    if not npc_name:
        npc_name = ""
    logging.info(f"[create_npc] user_id={user_id}, conv_id={conversation_id}, name={npc_name or '[None]'}, introduced={introduced}, sex={sex}")

    conn = get_db_connection()
    cursor = conn.cursor()

    chosen_arcs_rows = []
    final_stats = {}

    if sex.lower() == "male":
        final_stats = {
            "dominance": random.randint(0, 30),
            "cruelty": random.randint(0, 30),
            "closeness": random.randint(0, 30),
            "trust": random.randint(-30,30),
            "respect": random.randint(-30,30),
            "intensity": random.randint(0,30)
        }
    else:
        # female path => gather archetypes from DB
        cursor.execute("SELECT id, name, baseline_stats FROM Archetypes")
        all_arcs = cursor.fetchall()
        if not all_arcs:
            logging.warning("No archetypes in DB. Using random for female.")
            final_stats = {
                "dominance": random.randint(0,40),
                "cruelty": random.randint(0,40),
                "closeness": random.randint(0,40),
                "trust": random.randint(-40,40),
                "respect": random.randint(-40,40),
                "intensity": random.randint(0,40)
            }
        else:
            reroll_pool = [row for row in all_arcs if row[0] in REROLL_IDS]
            normal_pool = [row for row in all_arcs if row[0] not in REROLL_IDS]
            if len(normal_pool) < total_archetypes:
                chosen_arcs_rows = normal_pool
            else:
                chosen_arcs_rows = random.sample(normal_pool, total_archetypes)
            if reroll_extra and reroll_pool:
                chosen_arcs_rows.append(random.choice(reroll_pool))
            final_stats = combine_archetype_stats(chosen_arcs_rows)

    # Convert chosen archetypes to JSON
    chosen_arcs_json_str = archetypes_to_json(chosen_arcs_rows)
    try:
        chosen_arcs_list = json.loads(chosen_arcs_json_str)
    except:
        logging.error("Error parsing chosen_arcs JSON, fallback to empty.")
        chosen_arcs_list = []

    # 1) synergy step => we want a synergy text + possibly an npc_name
    synergy_data = {}
    if chosen_arcs_list:
        synergy_data = await get_archetype_synergy_description(chosen_arcs_list, npc_name)
        # synergy_data => {"npc_name":"...", "archetype_summary":"..."}
        new_npc_name = synergy_data["npc_name"]
        synergy_text = synergy_data["archetype_summary"]
    else:
        new_npc_name = npc_name if npc_name else f"NPC_{random.randint(1000,9999)}"
        synergy_text = "No synergy text available."

    # 2) extras summary => you can still do a simpler text approach or define a second schema. We'll keep it simpler:
    extras_summary = get_archetype_extras_summary(chosen_arcs_list, new_npc_name)  # same as old code, or rewrite as structured if you prefer

    # 3) age+birth => call structured approach
    archetype_names = [arc["name"] for arc in chosen_arcs_list]
    age_birth_data = await generate_npc_age_and_birthdate_gpt_structured(new_npc_name, relationships, archetype_names, current_year=1000)
    npc_age = age_birth_data["age"]
    birthdate = age_birth_data["birthdate"]

    # 4) physical desc => structured
    phys_desc_obj = await get_physical_description_structured(new_npc_name, final_stats, chosen_arcs_list)
    physical_description = phys_desc_obj["description"]

    # Insert row
    try:
        cursor.execute(
            """
            INSERT INTO NPCStats (
                user_id, conversation_id,
                npc_name, introduced, sex,
                dominance, cruelty, closeness, trust, respect, intensity,
                archetypes, archetype_summary, archetype_extras_summary,
                physical_description, memory, monica_level,
                age, birthdate
            )
            VALUES (
                %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                '[]'::jsonb, 0,
                %s, %s
            )
            RETURNING npc_id
            """,
            (user_id, conversation_id,
             new_npc_name, introduced, sex.lower(),
             final_stats["dominance"], final_stats["cruelty"],
             final_stats["closeness"], final_stats["trust"],
             final_stats["respect"], final_stats["intensity"],
             chosen_arcs_json_str, synergy_text, extras_summary, physical_description,
             npc_age, birthdate)
        )
        new_id = cursor.fetchone()[0]
        conn.commit()
        logging.info(f"[create_npc] Inserted npc_id={new_id}, name={new_npc_name}, stats={final_stats}, age={npc_age}, birthdate={birthdate}, synergy={synergy_text[:40]}...")
    except Exception as e:
        conn.rollback()
        logging.error(f"[create_npc] DB error: {e}", exc_info=True)
        conn.close()
        raise

    # Process relationships:
    if relationships:
        # If relationships are provided externally:
        for rel in relationships:
            num_memories = random.randint(1, 5)
            for _ in range(num_memories):
                memory_text = get_shared_memory(rel, new_npc_name)
                record_npc_event(user_id, conversation_id, new_id, memory_text)
    else:
        # If no relationships are provided, assign random relationships.
        assign_random_relationships(user_id, conversation_id, new_id, new_npc_name)

    # Finally, assign NPC flavor, close the connection, and return the new NPC id.
    assign_npc_flavor(user_id, conversation_id, new_id)
    conn.close()
    return new_id

# 8) NPC Flavor
def assign_npc_flavor(user_id, conversation_id, npc_id: int):
    """
    Randomly pick 3 hobbies, 5 personalities, 3 likes, and 3 dislikes from JSON data,
    then store them in NPCStats.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    hobby_pool = DATA.get("hobbies", [])
    personality_pool = DATA.get("personalities", [])
    likes_pool = DATA.get("likes", [])
    dislikes_pool = DATA.get("dislikes", [])

    hbs = random.sample(hobby_pool, 3) if len(hobby_pool) >= 3 else hobby_pool
    pers = random.sample(personality_pool, 5) if len(personality_pool) >= 5 else personality_pool
    lks = random.sample(likes_pool, 3) if len(likes_pool) >= 3 else likes_pool
    dlks = random.sample(dislikes_pool, 3) if len(dislikes_pool) >= 3 else dislikes_pool

    try:
        cursor.execute(
            """
            UPDATE NPCStats
            SET hobbies=%s, personality_traits=%s, likes=%s, dislikes=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """,
            (json.dumps(hbs), json.dumps(pers), json.dumps(lks), json.dumps(dlks),
             user_id, conversation_id, npc_id)
        )
        conn.commit()
        logging.debug(f"[assign_npc_flavor] Flavor set for npc_id={npc_id}, user={user_id}, conv={conversation_id}")
    except Exception as e:
        conn.rollback()
        logging.error(f"[assign_npc_flavor] DB error: {e}", exc_info=True)
    finally:
        conn.close()
        
def get_physical_description(final_npc_name, final_stats, chosen_arcs_list):
    """
    Uses GPT to generate a robust, vivid physical description for an NPC.
    The prompt considers the NPC's final name, its stats, and the names of the chosen archetypes.
    Returns a plain text description.
    """
    stats_str = ", ".join([f"{k}: {v}" for k, v in final_stats.items()])
    archetypes_str = ", ".join([arc["name"] for arc in chosen_arcs_list]) if chosen_arcs_list else "None"

    prompt = (
        f"Generate a robust, vivid physical description for an NPC in a femdom daily-life sim. "
        f"Use the NPC's name: {final_npc_name}.\n"
        f"Consider the following details:\n"
        f"Stats: {stats_str}\n"
        f"Archetypes: {archetypes_str}\n\n"
        "The description should detail the NPC's physical appearance (e.g., facial features, build, style, and any distinctive traits) "
        "in a way that fits a world of dominant females. Output only the description text with no extra commentary or markdown."
    )
    logging.info("Generating physical description with prompt: %s", prompt)
    gpt_client = get_openai_client()
    messages = [{"role": "system", "content": prompt}]
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=550
        )
        description = response.choices[0].message.content.strip()
        logging.info("Generated physical description: %s", description)
        return description
    except Exception as e:
        logging.error("Error generating physical description: %s", e, exc_info=True)
        return "A physically striking NPC with an enigmatic and captivating appearance."



def update_missing_npc_archetypes(user_id, conversation_id):
    """
    Scans the NPCStats table for any NPCs (for the given user and conversation)
    that have missing archetype fields (i.e.:
      - archetypes is NULL or an empty array,
      - archetype_summary is NULL or blank,
      - archetype_extras_summary is NULL or blank).
    For each such NPC (if female), randomly selects archetypes, computes the synergy
    and extras summaries via GPT, and updates the record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT npc_id, sex
    FROM NPCStats 
    WHERE user_id = %s AND conversation_id = %s
      AND (
          archetypes IS NULL OR archetypes = '[]'
          OR archetype_summary IS NULL OR TRIM(archetype_summary) = ''
          OR archetype_extras_summary IS NULL OR TRIM(archetype_extras_summary) = ''
      )
    """
    cursor.execute(query, (user_id, conversation_id))
    rows = cursor.fetchall()
    if not rows:
        logging.info("No NPCs with missing archetype fields found for user_id=%s, conversation_id=%s", user_id, conversation_id)
        conn.close()
        return

    for npc_id, sex in rows:
        # For this example, we update only female NPCs (as in your creation logic).
        if sex.lower() != "female":
            logging.info("Skipping NPC %s (sex=%s) for archetype update.", npc_id, sex)
            continue

        # Retrieve all archetypes from the database.
        cursor.execute("SELECT id, name, baseline_stats FROM Archetypes")
        all_arcs = cursor.fetchall()
        if not all_arcs:
            logging.warning("No archetypes available in DB to update NPC %s", npc_id)
            continue

        # Define how many archetypes to choose (e.g., 4)
        total_archetypes = 4

        # Filter out any archetypes in the reroll pool (if desired)
        normal_pool = [arc for arc in all_arcs if arc[0] not in REROLL_IDS]
        if len(normal_pool) < total_archetypes:
            chosen_arcs = normal_pool
        else:
            chosen_arcs = random.sample(normal_pool, total_archetypes)

        # Build a Python list of archetype objects (not JSON-dumped yet)
        chosen_arcs_list = [{"id": arc[0], "name": arc[1]} for arc in chosen_arcs]

        # Generate the synergy text and extras summary using your existing GPT functions.
        synergy_text = get_archetype_synergy_description(chosen_arcs_list_for_json, npc_name)
        extras_summary = get_archetype_extras_summary(chosen_arcs_list_for_json, new_npc_name)

        update_query = """
        UPDATE NPCStats
        SET archetypes = %s,
            archetype_summary = %s,
            archetype_extras_summary = %s
        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
        """
        # Here we pass the archetypes as a JSON string since the column is JSONB.
        cursor.execute(update_query, (json.dumps(chosen_arcs_list), synergy_text, extras_summary, npc_id, user_id, conversation_id))
        conn.commit()
        logging.info("Updated NPC %s with archetypes and summaries.", npc_id)

    conn.close()

import random
import json
from db.connection import get_db_connection
from logic.memory_logic import get_shared_memory, record_npc_event

from logic.social_links import create_social_link

def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name):
    familial = ["mother", "sister", "aunt"]
    non_familial = ["enemy", "friend", "lover", "neighbor", "colleague", "classmate", "teammate"]

    relationships = []

    # Chance for relationship with player
    if random.random() < 0.5:
        if random.random() < 0.2:
            rel_type = random.choice(familial)
        else:
            rel_type = random.choice(non_familial)
        relationships.append({"target": "player", "target_name": "the player", "type": rel_type})

    # Chance for relationship with other NPCs:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT npc_id, npc_name FROM NPCStats WHERE user_id = %s AND conversation_id = %s AND npc_id != %s",
        (user_id, conversation_id, new_npc_id)
    )
    existing_npcs = cursor.fetchall()
    conn.close()

    for npc_row in existing_npcs:
        candidate_id, candidate_name = npc_row
        if random.random() < 0.3:
            if random.random() < 0.2:
                rel_type = random.choice(familial)
            else:
                rel_type = random.choice(non_familial)
            if not any(rel.get("target") == candidate_id for rel in relationships):
                relationships.append({
                    "target": candidate_id,
                    "target_name": candidate_name,
                    "type": rel_type
                })

    # For each relationship, generate a memory and create a SocialLinks row.
    for rel in relationships:
        # Update the call to include user_id and conversation_id:
        memory_text = get_shared_memory(user_id, conversation_id, rel, new_npc_name)
        record_npc_event(user_id, conversation_id, new_npc_id, memory_text)
        # Determine entity types:
        if rel["target"] == "player":
            create_social_link(
                user_id, conversation_id,
                "player", 0,          # entity1: the player (ID 0, or your designated player ID)
                "npc", new_npc_id,     # entity2: the new NPC
                link_type=rel["type"], link_level=0
            )
        else:
            create_social_link(
                user_id, conversation_id,
                "npc", rel["target"],
                "npc", new_npc_id,
                link_type=rel["type"], link_level=0
            )
