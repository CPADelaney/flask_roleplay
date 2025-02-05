import os
import logging
import json
from flask import Blueprint, jsonify
from db.connection import get_db_connection

logging.basicConfig(level=logging.DEBUG)

archetypes_bp = Blueprint('archetypes', __name__)

def insert_missing_archetypes():
    """
    Inserts or updates Archetypes with final "range + modifier" style baseline_stats.
    Loads the data from an external "archetypes_data.json" file.
    """
    # Optionally locate the file relative to this script's directory,
    # so you don't rely on the working directory:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    archetypes_json_path = os.path.join(current_dir, "..", "data", "archetypes_data.json")
    archetypes_json_path = os.path.normpath(archetypes_json_path)

    try:
        with open(archetypes_json_path, "r", encoding="utf-8") as f:
            archetypes_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"archetypes_data.json not found at path: {archetypes_json_path}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Could not decode archetypes_data.json: {e}")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check existing archetype names
    cursor.execute("SELECT name FROM Archetypes")
    existing = {row[0] for row in cursor.fetchall()}

    for arc in archetypes_data:
        name = arc["name"]

        bs_json = json.dumps(arc["baseline_stats"])
        prog_rules_json = json.dumps(arc.get("progression_rules", []))
        setting_ex_json = json.dumps(arc.get("setting_examples", []))
        unique_traits_json = json.dumps(arc.get("unique_traits", []))

        if name not in existing:
            cursor.execute("""
                INSERT INTO Archetypes 
                    (name, baseline_stats, progression_rules, setting_examples, unique_traits)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, bs_json, prog_rules_json, setting_ex_json, unique_traits_json))
            logging.info(f"Inserted archetype: {name}")
        else:
            cursor.execute("""
                UPDATE Archetypes
                SET baseline_stats = %s,
                    progression_rules = %s,
                    setting_examples = %s,
                    unique_traits = %s
                WHERE name = %s
            """, (bs_json, prog_rules_json, setting_ex_json, unique_traits_json, name))
            logging.info(f"Updated existing archetype: {name}")

    conn.commit()
    conn.close()


@archetypes_bp.route('/insert_archetypes', methods=['POST'])
def insert_archetypes_route():
    """
    Optional route to insert/update all archetypes manually with final "range + modifier" style stats.
    """
    try:
        insert_missing_archetypes()
        return jsonify({"message": "Archetypes inserted/updated successfully"}), 200
    except Exception as e:
        logging.exception("Error inserting archetypes.")
        return jsonify({"error": str(e)}), 500


def assign_archetypes_to_npc(npc_id):
    """
    Picks 4 random archetypes from the DB and stores them in NPCStats.archetypes (JSON).
    Example usage:
        npc_id = create_npc_in_db(...)
        assign_archetypes_to_npc(npc_id)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM Archetypes")
    archetype_rows = cursor.fetchall()
    if not archetype_rows:
        conn.close()
        raise ValueError("No archetypes found in DB.")

    import random
    assigned = random.sample(archetype_rows, min(4, len(archetype_rows)))
    assigned_list = [{"id": row[0], "name": row[1]} for row in assigned]

    cursor.execute("""
        UPDATE NPCStats
        SET archetypes = %s
        WHERE npc_id = %s
    """, (json.dumps(assigned_list), npc_id))

    conn.commit()
    conn.close()
    return assigned_list
