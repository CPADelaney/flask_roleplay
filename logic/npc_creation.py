import random
import json
import logging
from flask import Blueprint, jsonify
from db.connection import get_db_connection

logging.basicConfig(level=logging.DEBUG)

###################
# 1) File Paths
###################
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
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return {}

###################
# 3) Build Our DATA Dictionary
###################
DATA = {}
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
REROLL_IDS = list(range(62, 73))  # All archetype IDs from 62..72

def clamp(value, min_val, max_val):
    """Utility to ensure a stat stays within [min_val..max_val]."""
    return max(min_val, min(value, max_val))

###################
# 5) Archetype Stat Combiner
###################
def combine_archetype_stats(archetype_rows):
    """
    archetype_rows is a list of (id, name, baseline_stats).
    baseline_stats might be JSON or a dict of e.g.:
       { "dominance_range": [20,60], "dominance_modifier": 5, ... }
    We do a random pick within each stat's range, add the modifier,
    sum them for all chosen archetypes, then average/clamp them.
    """
    # We'll accumulate sums for each stat, then average them
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
        # No archetypes -> fallback: random ~30
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

    # Now average & clamp
    for sk in sums.keys():
        sums[sk] = sums[sk] / count
        if sk in ["trust", "respect"]:
            sums[sk] = clamp(int(sums[sk]), -100, 100)
        else:
            sums[sk] = clamp(int(sums[sk]), 0, 100)

    return sums

def archetypes_to_json(rows):
    """
    Convert (id, name, baseline_stats) => 
       [ {"id": id, "name": name}, ... ].
    """
    arr = []
    for (aid, aname, _bs) in rows:
        arr.append({"id": aid, "name": aname})
    return json.dumps(arr)

###################
# 6) Create NPC
###################
def create_npc(npc_name=None, introduced=False, sex="female", reroll_extra=False, total_archetypes=4):
    """
    Creates a new NPC in NPCStats with multiple archetypes logic.
    - If sex='male', we skip archetypes or treat them differently.
    - total_archetypes: how many normal archetypes to pick.
    - reroll_extra: if True, we add an extra from the REROLL_IDS pool (62..72).
    """
    if not npc_name:
        npc_name = f"NPC_{random.randint(1000,9999)}"
    logging.info(f"[create_npc] Creating NPC: name={npc_name}, introduced={introduced}, sex={sex}")

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) If sex = male, skip archetypes or do minimal
    if sex.lower() == "male":
        chosen_arcs = []
        final_stats = {
            "dominance": random.randint(0,30),
            "cruelty":   random.randint(0,30),
            "closeness": random.randint(0,30),
            "trust":     random.randint(-30,30),
            "respect":   random.randint(-30,30),
            "intensity": random.randint(0,30)
        }
    else:
        # 2) We gather all archetypes
        cursor.execute("SELECT id, name, baseline_stats FROM Archetypes")
        all_arcs = cursor.fetchall()
        if not all_arcs:
            logging.warning("[create_npc] No archetypes in DB. Using pure-random for female.")
            chosen_arcs = []
            final_stats = {
                "dominance": random.randint(0,40),
                "cruelty":   random.randint(0,40),
                "closeness": random.randint(0,40),
                "trust":     random.randint(-40,40),
                "respect":   random.randint(-40,40),
                "intensity": random.randint(0,40)
            }
        else:
            # partition normal vs reroll
            reroll_pool  = [row for row in all_arcs if row[0] in REROLL_IDS]
            normal_pool  = [row for row in all_arcs if row[0] not in REROLL_IDS]

            # pick total_archetypes from normal
            if len(normal_pool) < total_archetypes:
                chosen_arcs = normal_pool
            else:
                chosen_arcs = random.sample(normal_pool, total_archetypes)

            # if reroll_extra, add 1 random from reroll_pool
            if reroll_extra and reroll_pool:
                chosen_arcs.append(random.choice(reroll_pool))

            # now combine stats
            final_stats = combine_archetype_stats(chosen_arcs)

    # 3) Insert the NPC
    try:
        cursor.execute("""
            INSERT INTO NPCStats (
              npc_name, introduced, sex,
              dominance, cruelty, closeness, trust, respect, intensity,
              archetypes,
              memory, monica_level
            )
            VALUES (
              %s, %s, %s,
              %s, %s, %s, %s, %s, %s,
              %s,
              '[]'::jsonb, 0
            )
            RETURNING npc_id
        """, (
            npc_name,
            introduced,
            sex.lower(),
            final_stats["dominance"], final_stats["cruelty"], final_stats["closeness"],
            final_stats["trust"], final_stats["respect"], final_stats["intensity"],
            archetypes_to_json(chosen_arcs)
        ))
        new_id = cursor.fetchone()[0]
        conn.commit()
        logging.info(f"[create_npc] Inserted npc_id={new_id} with {len(chosen_arcs)} archetypes. Stats={final_stats}")
    except Exception as e:
        conn.rollback()
        logging.error(f"[create_npc] DB error: {e}", exc_info=True)
        conn.close()
        raise

    # 4) Assign random flavor (hobbies/likes/dislikes/etc.)
    assign_npc_flavor(new_id)

    conn.close()
    return new_id

###################
# 7) NPC Flavor
###################
def assign_npc_flavor(npc_id: int):
    """
    Randomly pick 3 hobbies, 3 likes, 3 dislikes, 5 personalities from the
    JSON data, then store them in NPCStats for that npc_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    hbs    = random.sample(DATA.get("hobbies", []),       k=3) if len(DATA.get("hobbies", []))>=3 else []
    pers   = random.sample(DATA.get("personalities", []), k=5) if len(DATA.get("personalities", []))>=5 else []
    lks    = random.sample(DATA.get("likes", []),         k=3) if len(DATA.get("likes", []))>=3 else []
    dlks   = random.sample(DATA.get("dislikes", []),      k=3) if len(DATA.get("dislikes", []))>=3 else []

    try:
        cursor.execute("""
            UPDATE NPCStats
            SET hobbies = %s,
                personality_traits = %s,
                likes = %s,
                dislikes = %s
            WHERE npc_id = %s
        """, (
            json.dumps(hbs),
            json.dumps(pers),
            json.dumps(lks),
            json.dumps(dlks),
            npc_id
        ))
        conn.commit()
        logging.debug(f"[assign_npc_flavor] Flavor set for npc_id={npc_id}")
    except Exception as e:
        conn.rollback()
        logging.error(f"[assign_npc_flavor] DB error: {e}", exc_info=True)
    finally:
        conn.close()
