# logic/memory_logic.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
import json
import psycopg2

memory_bp = Blueprint('memory_bp', __name__)

@memory_bp.route('/get_current_roleplay', methods=['GET'])
def get_current_roleplay():
    """
    Returns an array of {key, value} objects from the currentroleplay table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM currentroleplay")
    rows = cursor.fetchall()
    conn.close()

    data = [{"key": r[0], "value": r[1]} for r in rows]
    return jsonify(data), 200

def record_npc_event(npc_id, event_description):
    """
    Appends a new event to the NPC's memory field in a thread-safe, scalable way.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Use a single atomic update with JSONB array manipulation
        cursor.execute("""
            UPDATE NPCStats
            SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s::text)
            WHERE npc_id = %s
            RETURNING memory
        """, (event_description, npc_id))

        updated_memory = cursor.fetchone()
        if not updated_memory:
            print(f"NPC with ID {npc_id} not found. Skipping event recording.")

        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error recording NPC event: {e}")
    finally:
        conn.close()

@memory_bp.route('/store_roleplay_segment', methods=['POST'])
def store_roleplay_segment():
    """
    Stores or updates a key-value pair in the currentroleplay table.
    Uses an upsert to handle duplicates.
    """
    try:
        payload = request.get_json()
        segment_key = payload.get("key")
        segment_value = payload.get("value")

        if not segment_key or segment_value is None:
            return jsonify({"error": "Missing key or value"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Use an upsert for atomicity
        cursor.execute("""
            INSERT INTO currentroleplay (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (segment_key, segment_value))

        conn.commit()
        return jsonify({"message": "Stored successfully"}), 200
    except Exception as e:
        print(f"Error in store_roleplay_segment: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()
