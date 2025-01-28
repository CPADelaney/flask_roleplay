# routes/new_game.py

from flask import Blueprint, request, jsonify
import random
from db.connection import get_db_connection
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc

new_game_bp = Blueprint('new_game', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Clears out Settings, CurrentRoleplay, etc.
    Only 'Chase' remains in PlayerStats (default stats).
    Then re-inserts missing settings, spawns 10 unintroduced NPCs,
    and sets a new environment in CurrentRoleplay.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) Get the request data, unwrap if "params" is used by the GPT front-end
        data = request.get_json() or {}
        if "params" in data:
            data = data["params"]

        # If the user passes keepChance, use it. Otherwise default to 0.025
        keep_chance = data.get("keepChance", 0.025)

        # 2) Clear relevant tables
        cursor.execute("DELETE FROM Events;")
        cursor.execute("DELETE FROM PlannedEvents;")
        cursor.execute("DELETE FROM PlayerInventory;")
        cursor.execute("DELETE FROM Quests;")
        cursor.execute("DELETE FROM Locations;")
        cursor.execute("DELETE FROM CurrentRoleplay;")
        conn.commit()

        # Reinsert default settings
        insert_missing_settings()

        # Now gather existing NPCs
        cursor.execute("""
            SELECT npc_id, npc_name, monica_level, memory
            FROM NPCStats
        """)
        all_npcs = cursor.fetchall()

        # meltdown_pick_chance = 0.0005
        # meltdown_increment_chance = 0.20
        full_memory_chance = 0.50

        carried_npcs = []
        # meltdown_npc = None  # remove meltdown references altogether

        # 3) Decide which NPCs to keep or delete
        for npc_id, npc_name, old_monica_level, old_memory in all_npcs:
            # For now, treat all NPCs the same—no meltdown auto-keep
            if random.random() < keep_chance:
                carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
            else:
                cursor.execute("DELETE FROM NPCStats WHERE npc_id = %s", (npc_id,))

        # (Meltdown logic removed completely)

        # 4) For carried NPCs, partial memory if old_monica_level == 0
        for npc_id, npc_name, old_monica_level, old_memory in carried_npcs:
            if old_monica_level == 0:
                # 50% chance to fully keep memory
                if random.random() < full_memory_chance:
                    new_entry = "Somehow, you remember everything from the last cycle."
                else:
                    new_entry = "You vaguely recall your old life, but details are blurry."
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s)
                    WHERE npc_id = %s
                """, (new_entry, npc_id))

        # 5) Reset PlayerStats to keep only 'Chase'
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
                INSERT INTO PlayerStats (
                  player_name, corruption, confidence, willpower, obedience,
                  dependency, lust, mental_resilience, physical_endurance
                ) VALUES (
                  'Chase', 10, 60, 50, 20, 10, 15, 55, 40
                )
            ''')

        conn.commit()

        # 6) Spawn 10 new unintroduced NPCs
        for _ in range(10):
            new_id = create_npc(introduced=False)
            print(f"Created new unintroduced NPC, ID={new_id}")

        # 7) Generate environment & update CurrentRoleplay
        insert_missing_settings()
        mega_data = generate_mega_setting_logic()
        if "error" in mega_data:
            mega_data["mega_name"] = "No environment available"

        cursor.execute("""
            INSERT INTO CurrentRoleplay (key, value)
            VALUES ('CurrentSetting', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (mega_data["mega_name"],))
        conn.commit()

        # 8) Return the top-level "message" for the openapi "SuccessResponse"
        return jsonify({
            "message": (
                f"New game started with 10 unintroduced NPCs and keepChance={keep_chance}. "
                f"New environment = {mega_data['mega_name']}"
            )
        }), 200

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
