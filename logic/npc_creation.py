# logic/npc_creation.py

import os
import json
import random
import logging
from db.connection import get_db_connection
from logic.chatgpt_integration import get_openai_client  # or however you import it
from logic.memory_logic import get_shared_memory, record_npc_event

logging.basicConfig(level=logging.DEBUG)

###################
# 1) File Paths
###################
current_dir = os.path.dirname(os.path.abspath(__file__))
archetypes_json_path = os.path.join(current_dir, "..", "data", "archetypes_data.json")
archetypes_json_path = os.path.normpath(archetypes_json_path)

DATA_FILES = {
    "hobbies": "data/npc_hobbies.json",
    "likes": "data/npc_likes.json",
    "dislikes": "data/npc_dislikes.json",
    "personalities": "data/npc_personalities.json"
}

###################
# 2) Utility: Load External Data
###################
def load_data(file_path):
    """Loads JSON from file_path; returns Python data (dict or list)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return {}

###################
# 3) Build Our DATA Dictionary
###################
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

logging.debug("DATA loaded. For example, hobbies => %s", DATA.get("hobbies"))

###################
# 4) Reroll IDs & Clamping
###################
REROLL_IDS = list(range(62, 73))  # e.g. archetype IDs from 62..72

def clamp(value, min_val, max_val):
    """Utility to ensure a stat stays within [min_val..max_val]."""
    return max(min_val, min(value, max_val))

###################
# 5) Archetype Stat Combiner
###################
def combine_archetype_stats(archetype_rows):
    """
    archetype_rows is a list of (id, name, baseline_stats).
    baseline_stats might be JSON or a dict.
    We randomly pick within each stat's range, add the modifier,
    sum for all chosen archetypes, then average & clamp them.
    """
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
    """Convert (id, name, baseline_stats) => [ {"id": id, "name": name}, ... ]"""
    arr = []
    for (aid, aname, _bs) in rows:
        arr.append({"id": aid, "name": aname})
    return json.dumps(arr)

###################
# 6) GPT Synergy Functions
###################
def get_archetype_synergy_description(archetypes_list, provided_npc_name=None):
    """
    Calls GPT to produce a JSON object containing:
      - "npc_name": a creative (or provided) NPC name,
      - "archetype_summary": a short personality/backstory summary.
    
    If provided_npc_name is given, instruct GPT to use that value.
    """
    # If no archetypes, return a default JSON
    if not archetypes_list:
        default_name = provided_npc_name if provided_npc_name else "Unknown"
        return json.dumps({
            "npc_name": default_name,
            "archetype_summary": "No special archetype synergy."
        })
    
    archetype_names = [a["name"] for a in archetypes_list]
    
    if provided_npc_name:
        name_instruction = f"Use the provided NPC name: '{provided_npc_name}'. Do not invent a new name."
    else:
        name_instruction = "Generate a creative, fitting name for the NPC."
    
    system_instructions = (
        f"You are an expert creative writer. You are tasked with writing a personality/backstory summary for an NPC "
        f"who combines these archetypes: {', '.join(archetype_names)}.\n"
        f"{name_instruction}\n"
        "Your response must be a single valid JSON object with exactly two keys:\n"
        "  \"npc_name\": a creative, unique, and fitting name for the NPC,\n"
        "  \"archetype_summary\": describe the unique personality that emerges from merging the independent archetypes into one cohesive, unified archetype.\n"
        "Do not include any extra text, markdown formatting, or newlines outside of the JSON. Output only the JSON object."
    )
    
    gpt_client = get_openai_client()
    messages = [{"role": "system", "content": system_instructions}]
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"[get_archetype_synergy_description] GPT error: {e}")
        default_name = provided_npc_name if provided_npc_name else "Unknown"
        return json.dumps({
            "npc_name": default_name,
            "archetype_summary": "An NPC with these archetypes has a mysterious blend of traits, but GPT call failed."
        })

def get_archetype_extras_summary(archetypes_list, provided_npc_name):
    """
    Produces a concise summary that fuses extra details (progression_rules,
    unique_traits, preferred_kinks) from the chosen archetypes into one cohesive description.
    The summary must explicitly use the provided NPC name (provided_npc_name) so that it matches
    the name stored in the NPCStats table.
    """
    if not archetypes_list:
        return f"No extra details available for {provided_npc_name}."
    
    extras_text_list = []
    for arc in archetypes_list:
        pr = " ".join(arc.get("progression_rules", []))
        ut = " ".join(arc.get("unique_traits", []))
        pk = " ".join(arc.get("preferred_kinks", []))
        extras_text_list.append(f"{arc['name']}: Progression: {pr}; Traits: {ut}; Kinks: {pk}")
    
    combined_extras = "\n\n".join(extras_text_list)
    
    system_instructions = f"""
    You are a creative writer tasked with synthesizing the following extra details from several archetypes in a femdom context into one cohesive, unified description for an NPC named "{provided_npc_name}". Instead of describing each archetype separately, imagine that their traits, progression rules, and unique kinks merge into a single, complex persona. 
    Here are the details:
    {combined_extras}
    
    Please produce a concise description (3-5 sentences) that integrates all these details into a singular, powerful image of an NPC named "{provided_npc_name}". Emphasize how the combined traits reinforce an overall dominant and compelling personality.
    Output only the final description text without any extra commentary or formatting.
    """
    
    gpt_client = get_openai_client()
    messages = [{"role": "system", "content": system_instructions}]
    
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"[get_archetype_extras_summary] GPT error: {e}")
        return f"An extra archetype summary for {provided_npc_name} could not be generated."


###################
# 7) Create NPC
###################
def create_npc(
    user_id,
    conversation_id,
    npc_name=None,
    introduced=False,
    sex="female",
    reroll_extra=False,
    total_archetypes=4,
    relationships=None  # Optional: list of relationship dicts
):
    # If no name is provided, initialize it as an empty string so that GPT will generate a dynamic name.
    if not npc_name:
        npc_name = ""
    logging.info(
        f"[create_npc] user_id={user_id}, conv_id={conversation_id}, "
        f"name={npc_name or '[None]'}, introduced={introduced}, sex={sex}"
    )

    conn = get_db_connection()
    cursor = conn.cursor()

    chosen_arcs_rows = []
    chosen_arcs_list_for_json = []  # for GPT functions
    final_stats = {}

    if sex.lower() == "male":
        final_stats = {
            "dominance": random.randint(0, 30),
            "cruelty": random.randint(0, 30),
            "closeness": random.randint(0, 30),
            "trust": random.randint(-30, 30),
            "respect": random.randint(-30, 30),
            "intensity": random.randint(0, 30)
        }
    else:
        cursor.execute("SELECT id, name, baseline_stats FROM Archetypes")
        all_arcs = cursor.fetchall()
        if not all_arcs:
            logging.warning("[create_npc] No archetypes in DB. Using pure-random for female.")
            final_stats = {
                "dominance": random.randint(0, 40),
                "cruelty": random.randint(0, 40),
                "closeness": random.randint(0, 40),
                "trust": random.randint(-40, 40),
                "respect": random.randint(-40, 40),
                "intensity": random.randint(0, 40)
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
    
    chosen_arcs_json_str = archetypes_to_json(chosen_arcs_rows)
    try:
        chosen_arcs_list_for_json = json.loads(chosen_arcs_json_str)
    except Exception as e:
        logging.error("Error parsing chosen archetypes JSON: %s", e)
        chosen_arcs_list_for_json = []

    # --- Always generate synergy text and obtain a "nice" dynamic NPC name from GPT if archetypes exist ---
    if chosen_arcs_list_for_json:
        synergy_json = get_archetype_synergy_description(chosen_arcs_list_for_json, npc_name)
        if not synergy_json:
            logging.error("GPT returned an empty synergy JSON; using fallback.")
            synergy_json = json.dumps({
                "npc_name": npc_name if npc_name else f"NPC_{random.randint(1000,9999)}",
                "archetype_summary": "No synergy text available."
            })
        try:
            synergy_data = json.loads(synergy_json)
            # Use the GPT-provided name for the NPC, and that will be our final name.
            new_npc_name = synergy_data.get("npc_name", npc_name)
            synergy_text = synergy_data.get("archetype_summary", "")
            if isinstance(synergy_text, list):
                synergy_text = " ".join(synergy_text)
        except Exception as e:
            logging.error("Failed to parse synergy JSON; using fallback.", exc_info=True)
            new_npc_name = npc_name if npc_name else f"NPC_{random.randint(1000,9999)}"
            synergy_text = "No synergy text available."
        extras_summary = get_archetype_extras_summary(chosen_arcs_list_for_json)
    else:
        new_npc_name = npc_name if npc_name else f"NPC_{random.randint(1000,9999)}"
        synergy_text = "No synergy text available."
        extras_summary = "No extra archetype details available."
    
    # --- Generate physical description, now using the final NPC name (new_npc_name) ---
    physical_description = get_physical_description(new_npc_name, final_stats, chosen_arcs_list_for_json)
    
    try:
        cursor.execute(
            """
            INSERT INTO NPCStats (
                user_id, conversation_id,
                npc_name, introduced, sex,
                dominance, cruelty, closeness, trust, respect, intensity,
                archetypes,
                archetype_summary,
                archetype_extras_summary,
                physical_description,
                memory, monica_level
            )
            VALUES (
                %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s,
                %s,
                %s,
                %s,
                '[]'::jsonb, 0
            )
            RETURNING npc_id
            """,
            (
                user_id, conversation_id,
                new_npc_name, introduced, sex.lower(),
                final_stats["dominance"], final_stats["cruelty"],
                final_stats["closeness"], final_stats["trust"],
                final_stats["respect"], final_stats["intensity"],
                chosen_arcs_json_str,
                synergy_text,
                extras_summary,
                physical_description
            )
        )
        new_id = cursor.fetchone()[0]
        conn.commit()
        logging.info(
            f"[create_npc] Inserted npc_id={new_id} with stats={final_stats}. "
            f"Archetypes count={len(chosen_arcs_rows)}. New NPC name: {new_npc_name}"
        )
    except Exception as e:
        conn.rollback()
        logging.error(f"[create_npc] DB error: {e}", exc_info=True)
        conn.close()
        raise

    if relationships:
        for rel in relationships:
            memory_text = get_shared_memory(rel, new_npc_name)
            record_npc_event(user_id, conversation_id, new_id, memory_text)

    assign_npc_flavor(user_id, conversation_id, new_id)
    conn.close()
    return new_id

    # Process relationships if provided
    if relationships:
        for rel in relationships:
            memory_text = get_shared_memory(rel, new_npc_name)
            record_npc_event(user_id, conversation_id, new_id, memory_text)

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

    hobby_pool       = DATA.get("hobbies", [])
    personality_pool = DATA.get("personalities", [])
    likes_pool       = DATA.get("likes", [])
    dislikes_pool    = DATA.get("dislikes", [])

    hbs  = random.sample(hobby_pool, 3)       if len(hobby_pool) >= 3 else hobby_pool
    pers = random.sample(personality_pool, 5) if len(personality_pool) >= 5 else personality_pool
    lks  = random.sample(likes_pool, 3)       if len(likes_pool) >= 3 else likes_pool
    dlks = random.sample(dislikes_pool, 3)    if len(dislikes_pool) >= 3 else dislikes_pool

    try:
        cursor.execute(
            """
            UPDATE NPCStats
            SET hobbies=%s,
                personality_traits=%s,
                likes=%s,
                dislikes=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """,
            (
                json.dumps(hbs),
                json.dumps(pers),
                json.dumps(lks),
                json.dumps(dlks),
                user_id,
                conversation_id,
                npc_id
            )
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
        extras_summary = get_archetype_extras_summary(chosen_arcs_list)

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

