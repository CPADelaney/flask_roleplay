# routes/new_game.py

import logging
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

    logging.info("=== START: /start_new_game CALLED ===")
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) Get the request data, unwrap if "params" is used by the GPT front-end
        data = request.get_json() or {}
        logging.info(f"Raw incoming JSON data: {data}")
        if "params" in data:
            data = data["params"]
            logging.info(f"After unwrapping 'params': {data}")

        keep_chance = data.get("keepChance", 0.025)
        logging.info(f"Parsed keep_chance = {keep_chance}")

        # 2) Clear relevant tables
        logging.info("Deleting from Events, PlannedEvents, PlayerInventory, Quests, Locations, CurrentRoleplay.")
        cursor.execute("DELETE FROM Events;")
        cursor.execute("DELETE FROM PlannedEvents;")
        cursor.execute("DELETE FROM PlayerInventory;")
        cursor.execute("DELETE FROM Quests;")
        cursor.execute("DELETE FROM Locations;")
        cursor.execute("DELETE FROM CurrentRoleplay;")
        conn.commit()
        logging.info("Tables cleared successfully.")

        # 3) Reinsert default settings
        logging.info("Calling insert_missing_settings()")
        insert_missing_settings()
        logging.info("insert_missing_settings() completed.")

        # 4) Gather existing NPCs
        logging.info("Fetching NPCStats (npc_id, npc_name, monica_level, memory)")
        cursor.execute("""
            SELECT npc_id, npc_name, monica_level, memory
            FROM NPCStats
        """)
        all_npcs = cursor.fetchall()
        logging.info(f"Fetched {len(all_npcs)} NPCs from NPCStats.")

        full_memory_chance = 0.50
        carried_npcs = []

        # 5) Decide which NPCs to keep or delete
        logging.info(f"Processing each NPC with keepChance={keep_chance}")
        for npc_id, npc_name, old_monica_level, old_memory in all_npcs:
            if random.random() < keep_chance:
                carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
            else:
                logging.info(f"Deleting NPC (ID={npc_id}, name={npc_name}) due to keepChance logic.")
                cursor.execute("DELETE FROM NPCStats WHERE npc_id = %s", (npc_id,))

        conn.commit()
        logging.info(f"Carried {len(carried_npcs)} NPCs forward.")

        # 6) For carried NPCs, partial memory if old_monica_level == 0
        logging.info("Applying partial memory logic for normal NPCs.")
        for npc_id, npc_name, old_monica_level, old_memory in carried_npcs:
            if old_monica_level == 0:
                if random.random() < full_memory_chance:
                    new_entry = "Somehow, you remember everything from the last cycle."
                else:
                    new_entry = "You vaguely recall your old life, but details are blurry."

                logging.info(f"Adding memory entry to NPC (ID={npc_id}, name={npc_name}): {new_entry[:60]}...")
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s)
                    WHERE npc_id = %s
                """, (new_entry, npc_id))
        conn.commit()

        # 7) Reset PlayerStats to keep only 'Chase'
        logging.info("Resetting PlayerStats to keep only 'Chase'.")
        cursor.execute("DELETE FROM PlayerStats WHERE player_name != 'Chase';")
        cursor.execute("SELECT id FROM PlayerStats WHERE player_name = 'Chase';")
        chase_row = cursor.fetchone()

        if chase_row:
            logging.info("Updating existing 'Chase' stats.")
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
            logging.info("Inserting fresh row for 'Chase'.")
            cursor.execute('''
                INSERT INTO PlayerStats (
                  player_name, corruption, confidence, willpower, obedience,
                  dependency, lust, mental_resilience, physical_endurance
                ) VALUES (
                  'Chase', 10, 60, 50, 20, 10, 15, 55, 40
                )
            ''')
        conn.commit()
        logging.info("PlayerStats reset complete.")

        # 8) Spawn 10 new unintroduced NPCs
        logging.info("Spawning 10 new unintroduced NPCs via create_npc(introduced=False).")
        for i in range(10):
            new_id = create_npc(introduced=False)
            logging.info(f"Created new unintroduced NPC {i+1}/10, ID={new_id}")

        # 9) Generate environment & update CurrentRoleplay
        logging.info("Calling insert_missing_settings() again, just to be safe.")
        insert_missing_settings()
        logging.info("Generating new mega setting via generate_mega_setting_logic()")
        mega_data = generate_mega_setting_logic()
        if "error" in mega_data:
            logging.info("mega_data contained an error, forcing mega_name to 'No environment available'")
            mega_data["mega_name"] = "No environment available"

        logging.info(f"Setting CurrentSetting in CurrentRoleplay to: {mega_data['mega_name']}")
        cursor.execute("""
            INSERT INTO CurrentRoleplay (key, value)
            VALUES ('CurrentSetting', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (mega_data["mega_name"],))
        conn.commit()

        # 10) Return the top-level "message" for the openapi "SuccessResponse"
        success_msg = (
            f"New game started with 10 unintroduced NPCs and keepChance={keep_chance}. "
            f"New environment = {mega_data['mega_name']}"
        )
        logging.info(f"Success! Returning 200 with message: {success_msg}")
        return jsonify({"message": success_msg}), 200

    except Exception as e:
        logging.exception("Error in /start_new_game:")  # Log full traceback
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        logging.info("=== END: /start_new_game ===")
