import random
import json
import logging
from db.connection import get_db_connection
from routes.archetypes import assign_archetypes_to_npc  # Import your archetype logic

logging.basicConfig(level=logging.DEBUG)

# 1. Utility: Load external data files
def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return {}

# 2. Lazy-load pools for NPC creation
DATA_FILES = {
    "hobbies": "data/npc_hobbies.json",
    "likes": "data/npc_likes.json",
    "dislikes": "data/npc_dislikes.json",
    # "occupations": "data/npc_occupations.json",  # Commented out
    "personalities": "data/npc_personalities.json"
}

DATA = {key: load_data(path) for key, path in DATA_FILES.items()}

def clamp(value, min_val, max_val):
    """Utility to ensure a stat stays within [0, 100]."""
    return max(min_val, min(value, max_val))

def create_npc(npc_name=None, introduced=False):
    """
    Core function to create an NPC in NPCStats.
    Combines base stats with archetype-based modifiers.
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
        chosen_archetype_id, chosen_archetype_name, baseline_stats_json = archetype_row
        baseline_stats = baseline_stats_db  

    logging.debug(f"[create_npc] Picked archetype={chosen_archetype_name}, stats={baseline_stats}")

    # 2) Generate stats based on archetype
    def get_range_and_modifier(stat_key):
        """Helper to fetch range and modifier for a stat from archetype data."""
        range_key = f"{stat_key}_range"
        mod_key = f"{stat_key}_modifier"
        rng = baseline_stats.get(range_key, [0, 30])  # Default range
        mod = baseline_stats.get(mod_key, 0)         # Default modifier
        return rng, mod

    stats = {}
    for stat_key in ["dominance", "cruelty", "closeness", "trust", "respect", "intensity"]:
        stat_range, stat_mod = get_range_and_modifier(stat_key)
        stat_value = random.randint(stat_range[0], stat_range[1]) + stat_mod
        stats[stat_key] = clamp(stat_value, -100 if stat_key in ["trust", "respect"] else 0, 100)

    logging.debug(f"[create_npc] Final stats: {stats}")

    try:
        # 3) Insert NPC into the database
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
            '{}',  # Memory JSON field
            0, 0,  # Monica-level placeholders
            introduced
        ))
        new_npc_id = cursor.fetchone()[0]
        conn.commit()
        logging.debug(f"[create_npc] Inserted new NPC with ID={new_npc_id}")

        # 4) Assign archetype
        cursor.execute("""
            UPDATE NPCStats
            SET archetypes = %s
            WHERE npc_id = %s
        """, (json.dumps([{"id": chosen_archetype_id, "name": chosen_archetype_name}]), new_npc_id))
        conn.commit()

        # 5) Assign random flavor (hobbies, personality traits, etc.)
        assign_npc_flavor(new_npc_id)
        return new_npc_id

    except Exception as e:
        conn.rollback()
        logging.error(f"[create_npc] Error creating NPC: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def assign_npc_flavor(npc_id: int):
    """
    Assigns random hobbies, personality traits, likes, and dislikes to an NPC.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Randomly sample attributes from loaded data
    # occupation = random.choice(DATA["occupations"])  # Commented out
    hobbies = random.sample(DATA["hobbies"], k=3)
    personality_traits = random.sample(DATA["personalities"], k=5)
    likes = random.sample(DATA["likes"], k=3)
    dislikes = random.sample(DATA["dislikes"], k=3)

    try:
        cursor.execute("""
            UPDATE NPCStats
            SET 
                -- occupation = %s,  # Commented out
                hobbies = %s,
                personality_traits = %s,
                likes = %s,
                dislikes = %s
            WHERE npc_id = %s
        """, (
            # occupation,  # Commented out
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
