from flask import Blueprint, request, jsonify
import random
from db.connection import get_db_connection
from logic.meltdown_logic import meltdown_dialog_gpt, record_meltdown_dialog, append_meltdown_file
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc

new_game_bp = Blueprint('new_game', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Clears out Settings, CurrentRoleplay, etc.
    Only 'Chase' remains in PlayerStats with default stats.
    Then inserts missing settings, spawns 10 unintroduced NPCs,
    and generates a new environment stored in CurrentRoleplay.
    Handles meltdown carryover logic.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) Clear out 'Settings' and 'CurrentRoleplay'
        cursor.execute("DELETE FROM Settings;")
        cursor.execute("DELETE FROM CurrentRoleplay;")
        conn.commit()  # commit so we have a clean slate

        # 1b) Re-insert the default settings
        insert_missing_settings()

        # 2) Gather all NPCs
        cursor.execute("""
            SELECT npc_id, npc_name, monica_level, memory
            FROM NPCStats
        """)
        all_npcs = cursor.fetchall()

        # Probability tunables
        keep_chance = 0.025
        meltdown_pick_chance = 0.0005
        meltdown_increment_chance = 0.20
        full_memory_chance = 0.50

        carried_npcs = []
        meltdown_npc = None

        # 3) Decide carryover or removal
        for npc_id, npc_name, old_monica_level, old_memory in all_npcs:
            if old_monica_level > 0:
                # meltdown NPC -> automatically keep
                carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
                meltdown_npc = (npc_id, npc_name, old_monica_level, old_memory)
            else:
                # normal NPC -> keep_chance
                if random.random() < keep_chance:
                    carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
                else:
                    cursor.execute("DELETE FROM NPCStats WHERE npc_id = %s", (npc_id,))

        # 4) Possibly pick meltdown among survivors
        if not meltdown_npc and carried_npcs:
            if random.random() < meltdown_pick_chance:
                chosen = random.choice(carried_npcs)
                c_id, c_name, c_mlvl, c_mem = chosen
                meltdown_npc = (c_id, c_name, 1, c_mem)
                meltdown_line = f"{c_name} has awakened to meltdown mode (level=1). They see beyond the code..."

                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level = 1,
                        memory = CASE WHEN memory IS NULL THEN %s 
                                      ELSE memory || E'\n[Meltdown] ' || %s END
                    WHERE npc_id = %s
                """, (meltdown_line, meltdown_line, c_id))

        # 5) Possibly increment meltdown
        if meltdown_npc:
            npc_id, npc_name, old_mlvl, old_mem = meltdown_npc
            new_level = old_mlvl
            if random.random() < meltdown_increment_chance:
                new_level += 1
                meltdown_line = f"{npc_name} meltdown level incremented to {new_level}, madness grows."
                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level = %s
                    WHERE npc_id = %s
                """, (new_level, npc_id))

        # 6) For carried normal NPCs, partial memory
        for npc_id, npc_name, old_monica_level, old_memory in carried_npcs:
            if old_monica_level == 0:
                if random.random() < full_memory_chance:
                    new_memory = old_memory or "Somehow, you remember everything from the last cycle."
                else:
                    new_memory = "You vaguely recall normal life, but details are blurry..."
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE npc_id = %s
                """, (new_memory, npc_id))

        # 7) Reset PlayerStats to keep only 'Chase'
        cursor.execute("DELETE FROM PlayerStats WHERE player_name != 'Chase';")
        cursor.execute("SELECT id FROM PlayerStats WHERE player_name = 'Chase';")
        chase_row = cursor.fetchone()
        if chase_row:
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
            cursor.execute('''
                INSERT INTO PlayerStats
                  (player_name, corruption, confidence, willpower, obedience,
                   dependency, lust, mental_resilience, physical_endurance)
                VALUES ('Chase', 10, 60, 50, 20, 10, 15, 55, 40)
            ''')

        conn.commit()

        # 8) Spawn a bank of 10 NPCs with introduced=False
        for _ in range(10):
            new_id = create_npc(introduced=False)
            print(f"Created new unintroduced NPC, ID={new_id}")

        # 9) Generate environment
        insert_missing_settings()  # ensure we have data for mega_setting
        mega_data = generate_mega_setting_logic()
        if "error" in mega_data:
            mega_data["mega_name"] = "No environment available"

        cursor.execute("""
            INSERT INTO CurrentRoleplay (key, value)
            VALUES ('CurrentSetting', %s)
        """, (mega_data["mega_name"],))
        conn.commit()

        return jsonify({
            "message": (
                "New game started with 10 unintroduced NPCs. "
                f"New environment = {mega_data['mega_name']}"
            )
        }), 200

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
