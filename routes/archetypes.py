from flask import Blueprint, jsonify
import json
from db.connection import get_db_connection

archetypes_bp = Blueprint('archetypes', __name__)

def insert_missing_archetypes():
    """
    Similar to how you did for settings. Insert ~60 archetypes if not present.
    We'll show a partial example using the 'Empress/Queen' archetype you described.
    """
    archetypes_data = [
        {
            "name": "Empress/Queen",
            "baseline_stats": {
                "dominance": "80–100",
                "cruelty": "60–90",
                "closeness": "40–60",
                "trust": "20–40",
                "respect": "30–50",
                "intensity": "50–90"
            },
            "progression_rules": (
                "• Dominance rises rapidly with every act of defiance or failure.\n"
                "• Intensity spikes during public ceremonies, turning submission into a spectacle."
            ),
            "setting_examples": (
                "• Palace: Uses formal rituals to enforce submission and respect.\n"
                "• Corporate Office: Wields bureaucratic power to manipulate personal life.\n"
            ),
            "unique_traits": (
                "• Demands elaborate displays of submission.\n"
                "• Punishes defiance harshly, often using public humiliation."
            )
        },
        # ... add your other ~59 archetypes here ...
    ]

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check existing archetype names
    cursor.execute("SELECT name FROM Archetypes")
    existing = {row[0] for row in cursor.fetchall()}

    for arc in archetypes_data:
        if arc["name"] not in existing:
            cursor.execute("""
                INSERT INTO Archetypes (name, baseline_stats, progression_rules, setting_examples, unique_traits)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                arc["name"],
                json.dumps(arc["baseline_stats"]),     # store baseline_stats as JSONB
                arc.get("progression_rules", ""),
                arc.get("setting_examples", ""),
                arc.get("unique_traits", "")
            ))
            print(f"Inserted archetype: {arc['name']}")
        else:
            print(f"Skipped existing archetype: {arc['name']}")

    conn.commit()
    conn.close()
    print("All archetypes processed or skipped (already existed).")


@archetypes_bp.route('/insert_archetypes', methods=['POST'])
def insert_archetypes_route():
    """
    Optional route to insert archetypes manually (like your /admin or /settings approach).
    """
    try:
        insert_missing_archetypes()
        return jsonify({"message": "Archetypes inserted/updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def assign_archetypes_to_npc(npc_id):
    """
    Picks 4 random archetypes from the DB and stores them in NPCStats.archetypes (a JSON field).
    Example usage whenever you create a new NPC:
        npc_id = create_npc_in_db(...)
        assign_archetypes_to_npc(npc_id)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Fetch all archetypes
    cursor.execute("SELECT id, name FROM Archetypes")
    archetype_rows = cursor.fetchall()
    if not archetype_rows:
        conn.close()
        raise ValueError("No archetypes found in DB.")

    import random
    four = random.sample(archetype_rows, min(4, len(archetype_rows)))
    # e.g. four = [(1, "Empress/Queen"), (5, "Tsundere"), ( ... )]

    # We'll store them in a JSON array in NPCStats. 
    # So let's get the names or IDs
    # Example storing just the ID or name. Let's store {id, name} for clarity.
    assigned_list = [{"id": row[0], "name": row[1]} for row in four]

    # 2) Update the NPCStats table. 
    # We'll assume we have a column "archetypes" of type JSONB in NPCStats.
    cursor.execute("""
        UPDATE NPCStats
        SET archetypes = %s
        WHERE npc_id = %s
    """, (json.dumps(assigned_list), npc_id))

    conn.commit()
    conn.close()

    return assigned_list  # for debugging

