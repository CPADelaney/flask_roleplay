import random
import json
import logging
from db.connection import get_db_connection
from routes.archetypes import assign_archetypes_to_npc  # or whatever you call it

logging.basicConfig(level=logging.DEBUG)

def clamp(value, min_val, max_val):
    """Utility to ensure a stat stays within [0, 100]."""
    return max(min_val, min(value, max_val))

def create_npc(npc_name=None, introduced=False):
    """
    Core function that creates a new NPC in NPCStats,
    combining random base stats + an archetype-based adjustment.
    """
    if not npc_name:
        npc_name = f"NPC_{random.randint(1000,9999)}"

    logging.debug(f"[create_npc] Starting with npc_name={npc_name}, introduced={introduced}")

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Pick a random archetype from the Archetypes table
    #    We'll assume the "baseline_stats" column is JSON with keys like:
    #    "dominance_range": [10, 40], "dominance_modifier": 20, etc.
    cursor.execute("""
        SELECT id, name, baseline_stats
        FROM Archetypes
        ORDER BY RANDOM() LIMIT 1
    """)
    archetype_row = cursor.fetchone()

    if archetype_row is None:
        # If no archetypes exist in the DB, fallback to some default
        logging.warning("[create_npc] No archetypes found; using pure random stats with no modifiers.")
        chosen_archetype_id = None
        chosen_archetype_name = "None"
        baseline_stats = {}
    else:
        chosen_archetype_id, chosen_archetype_name, baseline_stats_json = archetype_row
        # parse the JSON
        baseline_stats = json.loads(baseline_stats_json)

    logging.debug(f"[create_npc] Picked archetype={chosen_archetype_name}, data={baseline_stats}")

    # Helper to get a range [min, max] + a modifier from the baseline_stats
    def get_range_and_modifier(stat_key):
        # e.g. stat_key might be "dominance"
        # then we look for "dominance_range" and "dominance_modifier" in baseline_stats
        range_key = f"{stat_key}_range"
        mod_key = f"{stat_key}_modifier"

        # Provide defaults if not found
        rng = baseline_stats.get(range_key, [0, 30])  # e.g. default range
        mod = baseline_stats.get(mod_key, 0)          # default modifier
        return rng, mod

    # 2) For each stat, compute final value
    dom_range, dom_mod = get_range_and_modifier("dominance")
    dominance = random.randint(dom_range[0], dom_range[1]) + dom_mod
    dominance = clamp(dominance, 0, 100)

    cru_range, cru_mod = get_range_and_modifier("cruelty")
    cruelty = random.randint(cru_range[0], cru_range[1]) + cru_mod
    cruelty = clamp(cruelty, 0, 100)

    clos_range, clos_mod = get_range_and_modifier("closeness")
    closeness = random.randint(clos_range[0], clos_range[1]) + clos_mod
    closeness = clamp(closeness, 0, 100)

    trust_range, trust_mod = get_range_and_modifier("trust")
    trust = random.randint(trust_range[0], trust_range[1]) + trust_mod
    trust = clamp(trust, -100, 100)  # if you allow negatives for trust

    resp_range, resp_mod = get_range_and_modifier("respect")
    respect = random.randint(resp_range[0], resp_range[1]) + resp_mod
    respect = clamp(respect, -100, 100)

    inten_range, inten_mod = get_range_and_modifier("intensity")
    intensity = random.randint(inten_range[0], inten_range[1]) + inten_mod
    intensity = clamp(intensity, 0, 100)

    logging.debug(
        "[create_npc] Final stats => dom=%s cru=%s clos=%s trust=%s resp=%s inten=%s",
        dominance, cruelty, closeness, trust, respect, intensity
    )

    try:
        # 3) Insert new NPC with final stats - FIXED JSON FIELD
        cursor.execute("""
            INSERT INTO NPCStats (
                npc_name,
                dominance, cruelty, closeness, trust, respect, intensity,
                memory, monica_level, monica_games_left, introduced
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)  <!-- Added parameter placeholder
            RETURNING npc_id
        """, (
            npc_name,
            dominance, cruelty, closeness, trust, respect, intensity,
            '{}',  # Changed from '' to proper empty JSON object
            0, 0,  # monica_level and monica_games_left
            introduced
        ))
        new_npc_id = cursor.fetchone()[0]
        conn.commit()

        logging.debug(f"[create_npc] Inserted new NPC: npc_id={new_npc_id}")

        # 4) Store the chosen archetype in the NPCStats (archetypes JSON field),
        #    so we remember this NPC's main archetype. Or skip if you use a separate pivot table.
        #    We'll store a single item array: [{"id":..., "name":...}]
        assigned = [{"id": chosen_archetype_id, "name": chosen_archetype_name}]
        cursor.execute("""
            UPDATE NPCStats
            SET archetypes = %s
            WHERE npc_id = %s
        """, (
            json.dumps(assigned),
            new_npc_id
        ))
        conn.commit()

        # 5) Optionally assign random flavor
        assign_npc_flavor(new_npc_id)
        logging.debug(f"[create_npc] Flavor assigned for npc_id={new_npc_id}")

        return new_npc_id

    except Exception as e:
        conn.rollback()
        logging.error("[create_npc] ERROR: %s", e, exc_info=True)
        raise
    finally:
        conn.close()

def assign_npc_flavor(npc_id: int):
    """
    (unchanged from your original)
    Assigns random occupation, hobbies, personality traits, 
    likes, and dislikes to the given NPC.
    """
    # 1) Prepare lists...
    occupations = [...]
    hobbies_pool = [...]
    personality_pool = [...]
    likes_pool = [...]
    dislikes_pool = [...]

    # 2) Random picks
    occupation = random.choice(occupations)
    hobbies = random.sample(hobbies_pool, k=3)
    personality_traits = random.sample(personality_pool, k=5)
    likes = random.sample(likes_pool, k=3)
    dislikes = random.sample(dislikes_pool, k=3)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE NPCStats
            SET occupation = %s,
                hobbies = %s,
                personality_traits = %s,
                likes = %s,
                dislikes = %s
            WHERE npc_id = %s
        """, (
            occupation,
            json.dumps(hobbies) if hobbies else '[]',  # Ensure empty array instead of null
            json.dumps(personality_traits) if personality_traits else '[]',
            json.dumps(likes) if likes else '[]',
            json.dumps(dislikes) if dislikes else '[]',
            npc_id
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

def introduce_random_npc():
    """
    (unchanged from your original)
    Finds an unintroduced NPC in NPCStats (introduced=FALSE), 
    flips introduced=TRUE, and returns the npc_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id
        FROM NPCStats
        WHERE introduced = FALSE
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    npc_id = row[0]
    cursor.execute("""
        UPDATE NPCStats
        SET introduced = TRUE
        WHERE npc_id = %s
    """, (npc_id,))
    conn.commit()
    conn.close()
    return npc_id
