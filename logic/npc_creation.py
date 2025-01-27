import random
import json
import logging
from flask import Blueprint, jsonify
from db.connection import get_db_connection
from routes.archetypes import assign_archetypes_to_npc  # or your archetype logic

logging.basicConfig(level=logging.DEBUG)

###################
# 1. File Paths
###################
DATA_FILES = {
    "hobbies": "data/npc_hobbies.json",
    "likes": "data/npc_likes.json",
    "dislikes": "data/npc_dislikes.json",
    "personalities": "data/npc_personalities.json"
    # "occupations": "data/npc_occupations.json" # If you need occupations
}

###################
# 2. Utility: Load external data
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
# 3. Build Our DATA Dictionary
###################
DATA = {}
for key, path in DATA_FILES.items():
    raw = load_data(path)
    
    # If each file has a top-level object with key *_pool*
    # we turn that into a direct list in DATA.
    if key == "hobbies":
        # We expect raw to have {"hobbies_pool": [...]}
        DATA["hobbies"] = raw["hobbies_pool"]
    elif key == "likes":
        DATA["likes"] = raw["likes_pool"]
    elif key == "dislikes":
        DATA["dislikes"] = raw["dislikes_pool"]
    elif key == "personalities":
        # If it's also stored as "personalities_pool", do:
        DATA["personalities"] = raw["personalities_pool"]
    # else: handle other keys if needed

logging.debug("DATA loaded. For example, hobbies => %s", DATA.get("hobbies"))

###################
# 4. Stat Clamping
###################
def clamp(value, min_val, max_val):
    """Utility to ensure a stat stays within [0, 100]."""
    return max(min_val, min(value, max_val))

###################
# 5. Core create_npc Function
###################
def create_npc(npc_name=None, introduced=False):
    """
    Creates a new NPC in NPCStats, using random base stats + an archetype-based adjustment.
    """
    if not npc_name:
        npc_name = f"NPC_{random.randint(1000,9999)}"
    logging.debug(f"[create_npc] Starting with npc_name={npc_name}, introduced={introduced}")

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Pick a random archetype
    cursor.execute("""
        SELECT id, name, baseline_stats
        FROM Archetypes
        ORDER BY RANDOM() LIMIT 1
    """)
    archetype_row = cursor.fetchone()

    if archetype_row is None:
        logging.warning("[create_npc] No archetypes found; using pure random stats with no modifiers.")
        chosen_archetype_id, chosen_archetype_name, baseline_stats = None, "None", {}
    else:
        chosen_archetype_id, chosen_archetype_name, baseline_stats = archetype_row
        # If your column is JSON/JSONB, baseline_stats is already a dict
        # If it's TEXT, you might need baseline_stats = json.loads(baseline_stats)

    logging.debug(f"[create_npc] Picked archetype={chosen_archetype_name}, stats={baseline_stats}")

    # 2) Generate stats based on the archetype's range+modifier
    def get_range_and_modifier(stat_key):
        """Helper to fetch (range, modifier) for a stat."""
        rng = baseline_stats.get(f"{stat_key}_range", [0, 30])
        mod = baseline_stats.get(f"{stat_key}_modifier", 0)
        return rng, mod

    stats = {}
    for stat_key in ["dominance", "cruelty", "closeness", "trust", "respect", "intensity"]:
        r, m = get_range_and_modifier(stat_key)
        val = random.randint(r[0], r[1]) + m
        # clamp trust/respect between -100 & 100, others 0 & 100
        stats[stat_key] = clamp(val, -100 if stat_key in ["trust", "respect"] else 0, 100)

    logging.debug(f"[create_npc] Final stats => {stats}")

    try:
        # 3) Insert NPC
        cursor.execute("""
            INSERT INTO NPCStats (
                npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                memory, monica_level, monica_games_left, introduced
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING npc_id
        """, (
            npc_name,
            stats["dominance"], stats["cruelty"], stats["closeness"],
            stats["trust"], stats["respect"], stats["intensity"],
            '{}',  # memory is an empty JSON object for now
            0, 0,
            introduced
        ))
        new_npc_id = cursor.fetchone()[0]
        conn.commit()
        logging.debug(f"[create_npc] Inserted new NPC with ID={new_npc_id}")

        # 4) Assign archetype (store it in archetypes field as an array of dict)
        cursor.execute("""
            UPDATE NPCStats
            SET archetypes = %s
            WHERE npc_id = %s
        """, (json.dumps([{"id": chosen_archetype_id, "name": chosen_archetype_name}]), new_npc_id))
        conn.commit()

        # 5) Assign random flavor
        assign_npc_flavor(new_npc_id)
        return new_npc_id

    except Exception as e:
        conn.rollback()
        logging.error(f"[create_npc] Error creating NPC: {e}", exc_info=True)
        raise
    finally:
        conn.close()

###################
# 6. Random Flavor
###################
def assign_npc_flavor(npc_id: int):
    """
    Assigns random hobbies, personality traits, likes, and dislikes to the given NPC.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Now that we stored the correct data in DATA["hobbies"] etc.
    hobbies = random.sample(DATA["hobbies"], k=3)
    personality_traits = random.sample(DATA["personalities"], k=5)
    likes = random.sample(DATA["likes"], k=3)
    dislikes = random.sample(DATA["dislikes"], k=3)

    try:
        cursor.execute("""
            UPDATE NPCStats
            SET 
                hobbies = %s,
                personality_traits = %s,
                likes = %s,
                dislikes = %s
            WHERE npc_id = %s
        """, (
            json.dumps(hobbies),
            json.dumps(personality_traits),
            json.dumps(likes),
            json.dumps(dislikes),
            npc_id
        ))
        conn.commit()
        logging.debug(f"[assign_npc_flavor] Flavor assigned for npc_id={npc_id}")
    except Exception as e:
        conn.rollback()
        logging.error(f"[assign_npc_flavor] Error assigning flavor: {e}", exc_info=True)
        raise
    finally:
        conn.close()
