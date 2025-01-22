# routes/new_game.py

from flask import Blueprint, request, jsonify
import random
from db.connection import get_db_connection
from logic.meltdown_logic import meltdown_dialog_gpt, record_meltdown_dialog, append_meltdown_file

new_game_bp = Blueprint('new_game', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Clears out Settings, CurrentRoleplay, etc.
    Only 'Chase' remains in PlayerStats with default stats.

    NPC Logic:
      1) If npc has monica_level > 0, we automatically keep them.
      2) Normal NPC: 25% keep chance, else deleted.
      3) If no meltdown NPC exist among the survivors, 5% chance
         to pick one at random to become meltdown_npc with monica_level=1.
      4) If meltdown NPC exist, 20% chance to increment meltdown_level each new game.
      5) Non-meltdown NPC memory:
         - 10% chance to keep full memory
         - else partial memory

    PlayerStats:
      1) Delete all but 'Chase'.
      2) If 'Chase' row exists, reset to default stats
      3) If not, insert a default row for 'Chase'
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Clear out 'Settings' and 'CurrentRoleplay'
        cursor.execute("DELETE FROM Settings;")
        cursor.execute("DELETE FROM CurrentRoleplay;")

        # 2. Gather all NPCs
        cursor.execute("""
            SELECT npc_id, npc_name, monica_level, memory
            FROM NPCStats
        """)
        all_npcs = cursor.fetchall()

        # Probability tunables
        keep_chance = 0.025            # 2.5% chance a normal NPC is kept
        meltdown_pick_chance = 0.0005   # 0.05% chance to spawn a new meltdown if none
        meltdown_increment_chance = 0.20  # 20% chance meltdown NPC's level increments
        full_memory_chance = 0.50     # 50% chance a normal NPC keeps full memory

        carried_npcs = []
        meltdown_npc = None

        # 3. Decide carryover or removal
        for npc_id, npc_name, old_monica_level, old_memory in all_npcs:
            if old_monica_level > 0:
                # meltdown NPC -> automatically keep
                carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
                meltdown_npc = (npc_id, npc_name, old_monica_level, old_memory)
            else:
                # normal NPC -> 25% chance keep
                if random.random() < keep_chance:
                    carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
                else:
                    cursor.execute("DELETE FROM NPCStats WHERE npc_id = %s", (npc_id,))

        # 4. If no meltdown NPC among survivors, maybe pick one
        if not meltdown_npc and carried_npcs:
            if random.random() < meltdown_pick_chance:
                chosen = random.choice(carried_npcs)
                c_id, c_name, c_mlvl, c_mem = chosen
                meltdown_npc = (c_id, c_name, 1, c_mem)  # brand new meltdown
                meltdown_line = f"{c_name} has awakened to meltdown mode (level=1). They see beyond the code..."
                # Update DB
                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level = 1,
                        memory = CASE WHEN memory IS NULL THEN %s ELSE memory || E'\n[Meltdown] ' || %s END
                    WHERE npc_id = %s
                """, (meltdown_line, meltdown_line, c_id))
                # If using meltdown logic helpers:
                # append_meltdown_file(c_name, meltdown_line)

        # 5. Possibly increment meltdown NPC if it exists
        if meltdown_npc:
            npc_id, npc_name, old_mlvl, old_mem = meltdown_npc
            new_level = old_mlvl
            # 20% chance to increment meltdown
            if random.random() < meltdown_increment_chance:
                new_level += 1
                # meltdown_line = meltdown_dialog_gpt(npc_name, new_level)  # if using GPT
                meltdown_line = f"{npc_name} meltdown level incremented to {new_level}, madness grows."
                # Update
                cursor.execute("UPDATE NPCStats SET monica_level = %s WHERE npc_id = %s",
                               (new_level, npc_id))
                # record_meltdown_dialog(npc_id, meltdown_line)
                # append_meltdown_file(npc_name, meltdown_line)

        # 6. For the carried NPCs that are NOT meltdown NPC, handle memory
        #    (like partial memory or full memory)
        #    meltdown NPCs keep memory or have meltdown logic
        for npc_id, npc_name, old_monica_level, old_memory in carried_npcs:
            if old_monica_level == 0:  # normal NPC
                if random.random() < full_memory_chance:
                    # Keep memory fully
                    new_memory = old_memory or "Somehow, you remember everything from the last cycle."
                else:
                    # Partial memory
                    new_memory = "You vaguely recall normal life, but details are blurry..."

                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE npc_id = %s
                """, (new_memory, npc_id))
            # else meltdown NPC is handled above

        # 7. Now reset PlayerStats to keep only 'Chase'
        cursor.execute("DELETE FROM PlayerStats WHERE player_name != 'Chase';")

        # Check if 'Chase' exists
        cursor.execute("SELECT id FROM PlayerStats WHERE player_name = 'Chase';")
        chase_row = cursor.fetchone()

        if chase_row:
            # Reset stats
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
        else:
            # Insert default
            cursor.execute('''
                INSERT INTO PlayerStats
                  (player_name, corruption, confidence, willpower, obedience,
                   dependency, lust, mental_resilience, physical_endurance)
                VALUES ('Chase', 10, 60, 50, 20, 10, 15, 55, 40)
            ''')

        conn.commit()
        return jsonify({
            "message": (
                "New game started. Some NPCs carried. There's a chance one became meltdown NPC "
                "(aka 'Monica'), and Chase is reset."
            )
        }), 200

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
