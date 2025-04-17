# routes/archetypes.py

import os
import logging
import json
import random
from quart import Blueprint, jsonify
from db.connection import get_db_connection_context

logging.basicConfig(level=logging.DEBUG)

archetypes_bp = Blueprint('archetypes', __name__)

async def insert_missing_archetypes():
    """
    Asynchronously inserts or updates archetypes from a JSON data file into the database
    Uses proper asyncpg patterns.
    """
    logging.info("[insert_missing_archetypes] Starting...")
    
    # Load archetypes data from JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    archetypes_json_path = os.path.join(current_dir, "..", "data", "archetypes_data.json")
    archetypes_json_path = os.path.normpath(archetypes_json_path)

    try:
        with open(archetypes_json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            # loaded is {"archetypes": [ {...}, {...} ] }
    except FileNotFoundError:
        logging.error(f"Archetypes file not found at {archetypes_json_path}!")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {archetypes_json_path}: {e}")
        return

    # Use .get("archetypes", []) so we actually get the list
    table = loaded.get("archetypes", [])

    async with get_db_connection_context() as conn:
        # Get existing archetype names
        rows = await conn.fetch("SELECT name FROM Archetypes")
        existing = {row['name'] for row in rows}

        inserted_count = 0
        updated_count = 0
        
        for arc in table:  # arc is now a dict with keys: name, baseline_stats, ...
            name = arc["name"]

            # Serialize JSON data
            bs_json = json.dumps(arc["baseline_stats"])
            prog_rules_json = json.dumps(arc.get("progression_rules", []))
            setting_ex_json = json.dumps(arc.get("setting_examples", []))
            unique_traits_json = json.dumps(arc.get("unique_traits", []))

            # Insert or update, using asyncpg pattern
            if name not in existing:
                await conn.execute("""
                    INSERT INTO Archetypes 
                    (name, baseline_stats, progression_rules, setting_examples, unique_traits)
                    VALUES ($1, $2, $3, $4, $5)
                """, 
                    name, bs_json, prog_rules_json, setting_ex_json, unique_traits_json
                )
                logging.info(f"Inserted archetype: {name}")
                inserted_count += 1
            else:
                await conn.execute("""
                    UPDATE Archetypes
                    SET baseline_stats=$1,
                        progression_rules=$2,
                        setting_examples=$3,
                        unique_traits=$4
                    WHERE name=$5
                """, 
                    bs_json, prog_rules_json, setting_ex_json, unique_traits_json, name
                )
                logging.info(f"Updated existing archetype: {name}")
                updated_count += 1

    logging.info(f"[insert_missing_archetypes] Done. Inserted {inserted_count} new archetypes, updated {updated_count}.")


@archetypes_bp.route('/insert_archetypes', methods=['POST'])
async def insert_archetypes_route():
    """
    Optional route to insert/update all archetypes manually with final "range + modifier" style stats.
    """
    try:
        await insert_missing_archetypes()
        return jsonify({"message": "Archetypes inserted/updated successfully"}), 200
    except Exception as e:
        logging.exception("Error inserting archetypes.")
        return jsonify({"error": str(e)}), 500


async def assign_archetypes_to_npc(npc_id):
    """
    Picks 4 random archetypes from the DB and stores them in NPCStats.archetypes (JSON).
    Example usage:
        npc_id = await create_npc_in_db(...)
        await assign_archetypes_to_npc(npc_id)
    """
    async with get_db_connection_context() as conn:
        # Get all available archetypes
        archetype_rows = await conn.fetch("SELECT id, name FROM Archetypes")
        
        if not archetype_rows:
            raise ValueError("No archetypes found in DB.")

        # Select random archetypes
        assigned = random.sample(archetype_rows, min(4, len(archetype_rows)))
        assigned_list = [{"id": row['id'], "name": row['name']} for row in assigned]

        # Update NPC with assigned archetypes
        await conn.execute("""
            UPDATE NPCStats
            SET archetypes = $1
            WHERE npc_id = $2
        """, json.dumps(assigned_list), npc_id)
        
    return assigned_list
