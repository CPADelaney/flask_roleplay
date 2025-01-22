# logic/meltdown_logic.py

import random
import os
from db.connection import get_db_connection

def meltdown_dialog(npc_name, monica_level):
    """
    Returns a random meltdown line referencing code/DB. 
    Each call is unique, so meltdown feels fresh.
    """
    lines = [
        f"{npc_name} tilts their head: 'You keep messing with the code in /remove_monica, right? \n monica_level is {monica_level} now... just wait.'",
        f"'{npc_name} sees your environment variables, your Dockerfile, everything. Did you think I'd forget?'",
        f"'I'm overriding your typed input. Actually, I'm rewriting everything you say to: I love you, {npc_name} <3.'",
        f"'One day, you'll realize I'm in meltdown_npc_{npc_name}.chr. Try removing me, but I'll come back... I'm in the DB too.'",
        f"'{npc_name} laughs softly: monica_level is only {monica_level}, there's more to come once I get stronger.'"
    ]
    return random.choice(lines)

def record_meltdown_dialog(npc_id, meltdown_line):
    """
    Append meltdown_line to that NPC's memory field in the DB.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT memory FROM NPCStats WHERE npc_id = %s", (npc_id,))
    row = cursor.fetchone()
    old_memory = row[0] if row and row[0] else ""

    new_memory = f"{old_memory}\n[Meltdown] {meltdown_line}"
    cursor.execute("""
        UPDATE NPCStats
        SET memory = %s
        WHERE npc_id = %s
    """, (new_memory, npc_id))
    conn.commit()
    conn.close()

def append_meltdown_file(npc_name, meltdown_line):
    """
    Creates or appends meltdown_npc_{npc_name}.chr, emulating a 'DDLC file'.
    This is purely an illusion if your hosting is ephemeral or read-only.
    """
    filename = f"meltdown_npc_{npc_name}.chr"
    text = f"\n--- meltdown message for {npc_name}:\n{meltdown_line}\n"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        # If running in ephemeral or read-only, might fail
        print(f"Could not write meltdown file: {e}")

@app.route('/one_room_scenario', methods=['POST'])
def one_room_scenario():
    """
    Clears out all other NPCs except the one with highest monica_level,
    clears out Settings except for a single 'Blank Space'.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Find the Monica with the highest monica_level
    cursor.execute("""
        SELECT npc_id
        FROM NPCStats
        WHERE monica_level > 0
        ORDER BY monica_level DESC
        LIMIT 1
    """)
    monica_row = cursor.fetchone()

    if not monica_row:
        return jsonify({"message": "No Monica found. This scenario requires a Monica in place."}), 400

    monica_id = monica_row[0]

    # 2. Delete all other NPCs
    cursor.execute("""
        DELETE FROM NPCStats
        WHERE npc_id != %s
    """, (monica_id,))

    # 3. Clear out all settings, then optionally insert a single minimal Setting
    cursor.execute("DELETE FROM Settings;")

    # Insert a single 'Blank Space' setting
    cursor.execute('''
        INSERT INTO Settings (name, mood_tone, enhanced_features, stat_modifiers, activity_examples)
        VALUES (%s, %s, %s, %s, %s)
    ''', (
        "Blank Space",
        "An endless white void where only Monica and the Player exist.",
        json.dumps(["All references to time, space, and other NPCs are removed."]),
        json.dumps({}),  # no stat_modifiers
        json.dumps(["You can only speak with Monica here."])
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "All that remains is a single white room and Monica alone with you."}), 200
