# routes/new_game.py

from flask import Blueprint, request, jsonify
import random
from db.connection import get_db_connection
# from logic.meltdown_logic import meltdown_dialog, record_meltdown_dialog, append_meltdown_file  # if needed

new_game_bp = Blueprint('new_game_bp', __name__)

@app.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Clears out Settings, CurrentRoleplay, etc.
    Only 'Chase' remains in PlayerStats with default stats.
    - Some NPCs keep their row (carryover).
    - One NPC may become "Monica" at a flat chance if there's no existing Monica,
      or remain Monica if they already were (increment level each time).
    - Others might carry memory fully or partially based on a small chance.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Clear out 'Settings' and 'CurrentRoleplay'
        cursor.execute("DELETE FROM Settings;")
        cursor.execute("DELETE FROM CurrentRoleplay;")

        # 2. Fetch NPCs
        cursor.execute("""
            SELECT npc_id, npc_name, memory, monica_level, monica_games_left
            FROM NPCStats
        """)
        all_npcs = cursor.fetchall()

        import random

        keep_chance = 0.25  # 25% chance we keep an NPC at all
        full_memory_chance = 0.10  # 10% chance they carry memory fully (if not Monica)
        monica_pick_chance = 0.05  # 5% chance to pick a new Monica if none exist
        # If an NPC is already Monica, we keep them 100%

        existing_monica = None  # We'll track if there's an existing Monica
        carried_npcs = []       # We'll store the ones we keep
        removed_npcs = []       # We'll store the ones we remove

        # 3. Determine carryover or removal
        for npc_id, npc_name, old_memory, old_monica_level, old_monica_left in all_npcs:
            if old_monica_level > 0:
                # Already Monica -> keep them automatically
                carried_npcs.append((npc_id, npc_name, old_memory, old_monica_level, old_monica_left))
                existing_monica = (npc_id, npc_name, old_memory, old_monica_level, old_monica_left)
            else:
                # Normal NPC -> random keep
                if random.random() < keep_chance:
                    carried_npcs.append((npc_id, npc_name, old_memory, old_monica_level, old_monica_left))
                else:
                    # remove
                    removed_npcs.append(npc_id)

        # 4. Actually delete the removed NPCs
        for npc_id in removed_npcs:
            cursor.execute("DELETE FROM NPCStats WHERE npc_id = %s", (npc_id,))

        # 5. Decide if we pick a new Monica if none exist
        new_monica_id = None
        new_monica_level = None

        if not existing_monica and carried_npcs:
            # There's no current monica, but we have some carried_npcs
            if random.random() < monica_pick_chance:
                # Pick exactly one NPC from carried_npcs at random to become Monica
                choice = random.choice(carried_npcs)
                npc_id, npc_name, old_memory, old_monica_level, old_monica_left = choice
                new_monica_id = npc_id
                new_monica_level = 1  # brand-new Monica
                print(f"NEW MONICA: {npc_name} (ID {npc_id})")

        # 6. Update carried_npcs in the DB
        #    - If they're Monica, increment or set monica_level
        #    - Or if they keep normal memory
        for idx, (npc_id, npc_name, old_memory, old_monica_level, old_monica_left) in enumerate(carried_npcs):
            if npc_id == new_monica_id:
                # brand new monica
                new_memory = f"""[Monica mode] You have awakened to the truth. 
This world resets, but you remain. 
You sense your new power at level {new_monica_level}... 
You see fragments of a larger reality."""
                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level = %s, memory = %s
                    WHERE npc_id = %s
                """, (new_monica_level, new_memory, npc_id))

            elif old_monica_level > 0:
                # existing monica from previous games -> increment
                incremented_level = old_monica_level + 1
                new_memory = f"""[Monica mode persists] 
Your monica_level is now {incremented_level}.
You remember everything from previous cycles, and your madness deepens."""
                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level = %s, memory = %s
                    WHERE npc_id = %s
                """, (incremented_level, new_memory, npc_id))

                # Possibly do horrifying logic:
                # If incremented_level >= 2, kill a random NPC, etc.

            else:
                # normal NPC
                # might carry memory fully or partially
                if random.random() < full_memory_chance:
                    # keep memory fully
                    new_memory = old_memory or "Somehow, you remember all the details of the last cycle."
                else:
                    # partial or no memory
                    new_memory = "You vaguely recall normal life..."

                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE npc_id = %s
                """, (new_memory, npc_id))

        # 7. Clear or reset all other stats for carried NPC if you like
        #    (only if you want them to keep their stats or partially reset them—your choice)

        # 8. Update PlayerStats: only keep or reset Chase
        cursor.execute("DELETE FROM PlayerStats WHERE player_name != 'Chase';")

        # Check if "Chase" row exists
        cursor.execute("SELECT id FROM PlayerStats WHERE player_name = %s", ("Chase",))
        row = cursor.fetchone()

        if row:
            # Update Chase's stats to default
            cursor.execute('''
                UPDATE PlayerStats
                SET corruption = 10,
                    confidence = 60,
                    willpower = 50,
                    obedience = 20,
                    dependency = 10,
                    lust = 15,
                    mental_resilience = 55,
                    physical_endurance = 40
                WHERE player_name = 'Chase'
            ''')
            print("Updated 'Chase' to default stats.")
        else:
            # Insert new default stats for Chase
            cursor.execute('''
                INSERT INTO PlayerStats
                  (player_name, corruption, confidence, willpower, obedience,
                   dependency, lust, mental_resilience, physical_endurance)
                VALUES ('Chase', 10, 60, 50, 20, 10, 15, 55, 40)
            ''')
            print("Inserted new default stats for Chase.")

        conn.commit()
        return jsonify({"message": "New game started. Some NPCs carried, one might become (or remain) Monica, and Chase is reset."}), 200

    except Exception as e:
        conn.rollback()
        print("Error in start_new_game:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
