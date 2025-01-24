# logic/npc_creation.py

import random
import json
from db.connection import get_db_connection
from logic.meltdown_logic import meltdown_dialog_gpt  # If needed
from routes.archetypes import assign_archetypes_to_npc

def create_npc(npc_name=None, introduced=False):
    """
    Creates a new NPC in NPCStats, with random stats/archetypes.
    'introduced' is a boolean controlling whether the aggregator sees them now or later.
    """
    if not npc_name:
        npc_name = f"NPC_{random.randint(1000,9999)}"

    dominance = random.randint(10, 40)
    cruelty = random.randint(10, 40)
    closeness = random.randint(0, 30)
    trust = random.randint(-20, 20)
    respect = random.randint(-20, 20)
    intensity = random.randint(0, 40)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO NPCStats (
                npc_name,
                dominance, cruelty, closeness, trust, respect, intensity,
                memory, monica_level, monica_games_left, introduced
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING npc_id
        """, (
            npc_name,
            dominance, cruelty, closeness, trust, respect, intensity,
            "",     # memory
            0,      # monica_level
            0,      # monica_games_left
            introduced
        ))
        new_npc_id = cursor.fetchone()[0]
        conn.commit()

        # Assign 4 random archetypes
        assign_archetypes_to_npc(new_npc_id)

    # Optionally, do more randomization: occupation, hobbies, etc.
    # e.g. assign_random_occupation(new_npc_id)
    # e.g. assign_random_personality(new_npc_id)

        return new_npc_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
