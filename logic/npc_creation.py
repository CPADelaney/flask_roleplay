# logic/npc_creation.py

import random
import json
from db.connection import get_db_connection
from logic.meltdown_logic import meltdown_dialog_gpt  # If needed
from routes.archetypes import assign_archetypes_to_npc

def create_npc(npc_name=None):
    """
    Creates a new row in NPCStats with default stats,
    then assigns 4 random archetypes (store them in the 'archetypes' JSONB field).

    Returns the newly created npc_id.
    """
    if not npc_name:
        npc_name = f"NPC_{random.randint(1000, 9999)}"

    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert a bare-minimum NPC row with default stats
    # For demonstration, we just pick random ranges for these stats or default them.
    dominance = random.randint(0, 50)
    cruelty = random.randint(0, 50)
    closeness = random.randint(0, 50)
    trust = random.randint(-50, 50)
    respect = random.randint(-50, 50)
    intensity = random.randint(0, 50)

    cursor.execute("""
        INSERT INTO NPCStats (
            npc_name, 
            dominance, cruelty, closeness, trust, respect, intensity,
            memory, monica_level, monica_games_left
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING npc_id
    """, (
        npc_name,
        dominance, cruelty, closeness, trust, respect, intensity,
        "",  # memory
        0,   # monica_level
        0    # monica_games_left
    ))
    new_npc_id = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    # Now assign 4 random archetypes to that NPC
    assign_archetypes_to_npc(new_npc_id)

    # Optionally, do more randomization: occupation, hobbies, etc.
    # e.g. assign_random_occupation(new_npc_id)
    # e.g. assign_random_personality(new_npc_id)

    return new_npc_id
