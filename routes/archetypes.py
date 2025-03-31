# routes/archetypes.py

import os
import logging
import json
import asyncio
from flask import Blueprint, jsonify
from db.connection import get_db_connection_context

logging.basicConfig(level=logging.DEBUG)

archetypes_bp = Blueprint('archetypes', __name__)

async def insert_missing_archetypes():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    archetypes_json_path = os.path.join(current_dir, "..", "data", "archetypes_data.json")
    archetypes_json_path = os.path.normpath(archetypes_json_path)

    with open(archetypes_json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        # loaded is {"archetypes": [ {...}, {...} ] }

    # Use .get("archetypes", []) so we actually get the list
    table = loaded.get("archetypes", [])

    async with get_db_connection_context() as conn:
        # set() of existing names
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT name FROM Archetypes")
            rows = await cursor.fetchall()
            existing = {row[0] for row in rows}

        for arc in table:  # arc is now a dict with keys: name, baseline_stats, ...
            name = arc["name"]

            bs_json = json.dumps(arc["baseline_stats"])
            prog_rules_json = json.dumps(arc.get("progression_rules", []))
            setting_ex_json = json.dumps(arc.get("setting_examples", []))
            unique_traits_json = json.dumps(arc.get("unique_traits", []))

            # Insert or update, etc.
            if name not in existing:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        INSERT INTO Archetypes (name, baseline_stats, progression_rules, setting_examples, unique_traits)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (name, bs_json, prog_rules_json, setting_ex_json, unique_traits_json))
                logging.info(f"Inserted archetype: {name}")
            else:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        UPDATE Archetypes
                        SET baseline_stats=%s,
                            progression_rules=%s,
                            setting_examples=%s,
                            unique_traits=%s
                        WHERE name=%s
                    """, (bs_json, prog_rules_json, setting_ex_json, unique_traits_json, name))
                logging.info(f"Updated existing archetype: {name}")

        await conn.commit()


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
        npc_id = create_npc_in_db(...)
        assign_archetypes_to_npc(npc_id)
    """
    async with get_db_connection_context() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT id, name FROM Archetypes")
            archetype_rows = await cursor.fetchall()
            if not archetype_rows:
                raise ValueError("No archetypes found in DB.")

            import random
            assigned = random.sample(archetype_rows, min(4, len(archetype_rows)))
            assigned_list = [{"id": row[0], "name": row[1]} for row in assigned]

            await cursor.execute("""
                UPDATE NPCStats
                SET archetypes = %s
                WHERE npc_id = %s
            """, (json.dumps(assigned_list), npc_id))

        await conn.commit()
    return assigned_list
