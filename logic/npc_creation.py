import os
import json
import random
import logging
from db.connection import get_db_connection
from logic.chatgpt_integration import get_openai_client  # or however you import it

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

# Load the data
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
        # No archetypes => fallback random
        for k in sums.keys():
            sums[k] = random.randint(0, 30)
        return sums

    for (arc_id, arc_name, bs_json) in archetype_rows:
        # baseline_stats might be a string or a dict
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
                val = random.randint(0, 30)  # fallback
            sums[stat_key] += val

    # average & clamp
    for sk in sums.keys():
        sums[sk] = sums[sk] / count
        # trust/respect can be negative, so clamp -100..100
        if sk in ["trust", "respect"]:
            sums[sk] = clamp(int(sums[sk]), -100, 100)
        else:
            sums[sk] = clamp(int(sums[sk]), 0, 100)

    return sums

def archetypes_to_json(rows):
    """ Convert (id, name, baseline_stats) => [ {"id": id, "name": name}, ... ] """
    arr = []
    for (aid, aname, _bs) in rows:
        arr.append({"id": aid, "name": aname})
    return json.dumps(arr)

###################
# 6) GPT synergy function
###################
def get_archetype_synergy_description(archetypes_list):
    """
    Call GPT to produce a cohesive short paragraph describing how these 
    archetypes blend into a single personality/backstory.
    """
    if not archetypes_list:
        # No archetypes => skip GPT
        return "No special archetype synergy."

    archetype_names = [a["name"] for a in archetypes_list]
    # Build a prompt or system instructions:
    system_instructions = f"""
    You are writing a short personality/backstory summary for an NPC who combines 
    these archetypes: {', '.join(archetype_names)}.

    They exist in a femdom context. 
    Please produce a short paragraph (2-5 sentences) explaining how these archetypes fuse 
    into a single personality with interesting quirks that feel cohesive 
    (not random or contradictory). Keep it concise.
    """

    gpt_client = get_openai_client()
    messages = [{"role": "system", "content": system_instructions}]

    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o",  # or whichever
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        logging.error(f"[get_archetype_synergy_description] GPT error: {e}")
        # fallback
        return "An NPC with these archetypes has a mysterious blend of traits, but GPT call failed."

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
    total_archetypes=4
):
    """
    Creates a new NPC in NPCStats with multiple archetypes logic,
    scoping by user_id + conversation_id.

    - If sex='male', we skip archetypes or do minimal random.
    - total_archetypes: how many normal archetypes to pick.
    - reroll_extra: if True, we add 1 from REROLL_IDS (62..72).
    """
    if not npc_name:
        npc_name = f"NPC_{random.randint(1000,9999)}"
    logging.info(
        f"[create_npc] user_id={user_id}, conv_id={conversation_id}, "
        f"name={npc_name}, introduced={introduced}, sex={sex}"
    )

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Pick archetypes or skip if male
    chosen_arcs_rows = []
    chosen_arcs_list_for_json = []  # for synergy function
    final_stats = {}

    if sex.lower() == "male":
        # fallback minimal random
        final_stats = {
            "dominance": random.randint(0,30),
            "cruelty":   random.randint(0,30),
            "closeness": random.randint(0,30),
            "trust":     random.randint(-30,30),
            "respect":   random.randint(-30,30),
            "intensity": random.randint(0,30)
        }
    else:
        # Gather all archetypes
        cursor.execute("SELECT id, name, baseline_stats FROM Archetypes")
        all_arcs = cursor.fetchall()

        if not all_arcs:
            logging.warning("[create_npc] No archetypes in DB. Using pure-random for female.")
            final_stats = {
                "dominance": random.randint(0,40),
                "cruelty":   random.randint(0,40),
                "closeness": random.randint(0,40),
                "trust":     random.randint(-40,40),
                "respect":   random.randint(-40,40),
                "intensity": random.randint(0,40)
            }
        else:
            reroll_pool  = [row for row in all_arcs if row[0] in REROLL_IDS]
            normal_pool  = [row for row in all_arcs if row[0] not in REROLL_IDS]

            if len(normal_pool) < total_archetypes:
                chosen_arcs_rows = normal_pool
            else:
                chosen_arcs_rows = random.sample(normal_pool, total_archetypes)

            if reroll_extra and reroll_pool:
                chosen_arcs_rows.append(random.choice(reroll_pool))

            final_stats = combine_archetype_stats(chosen_arcs_rows)
    
    # For synergy function & storing
    chosen_arcs_json_str = archetypes_to_json(chosen_arcs_rows)
    try:
        # parse that JSON back to a python list
        chosen_arcs_list_for_json = json.loads(chosen_arcs_json_str)
    except:
        chosen_arcs_list_for_json = []

    # 2) If female, get synergy text from GPT
    synergy_text = ""
    if sex.lower() == "female" and chosen_arcs_list_for_json:
        synergy_text = get_archetype_synergy_description(chosen_arcs_list_for_json)
    else:
        synergy_text = "No synergy text (male or no archetypes)."

    # 3) Insert the NPC row
    try:
        cursor.execute(
            """
            INSERT INTO NPCStats (
                user_id, conversation_id,
                npc_name, introduced, sex,
                dominance, cruelty, closeness, trust, respect, intensity,
                archetypes,
                archetype_summary,
                memory, monica_level
            )
            VALUES (
                %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s,
                %s,
                '[]'::jsonb, 0
            )
            RETURNING npc_id
            """,
            (
                user_id, conversation_id,
                npc_name, introduced, sex.lower(),
                final_stats["dominance"], final_stats["cruelty"],
                final_stats["closeness"], final_stats["trust"],
                final_stats["respect"], final_stats["intensity"],
                chosen_arcs_json_str,       # your JSON array of archetypes
                synergy_text                # your GPT synergy
            )
        )
        new_id = cursor.fetchone()[0]
        conn.commit()

        logging.info(
            f"[create_npc] Inserted npc_id={new_id} with stats={final_stats}. "
            f"Archetypes count={len(chosen_arcs_rows)} synergy={synergy_text[:50]}"
        )
    except Exception as e:
        conn.rollback()
        logging.error(f"[create_npc] DB error: {e}", exc_info=True)
        conn.close()
        raise

    # 4) Assign random flavor (hobbies, personalities, likes, dislikes)
    assign_npc_flavor(user_id, conversation_id, new_id)

    conn.close()
    return new_id

###################
# 8) NPC Flavor
###################
def assign_npc_flavor(user_id, conversation_id, npc_id: int):
    """
    Randomly pick e.g. 3 hobbies, 5 personalities, 3 likes, 3 dislikes 
    from JSON data, then store them in NPCStats
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
