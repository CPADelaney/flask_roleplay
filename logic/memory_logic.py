from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection
import json
import psycopg2

memory_bp = Blueprint('memory_bp', __name__)

@memory_bp.route('/get_current_roleplay', methods=['GET'])
def get_current_roleplay():
    """
    Returns an array of {key, value} objects from CurrentRoleplay,
    scoped to user_id + conversation_id.
    The front-end or route call must pass ?conversation_id=XX or in session/headers.
    """
    # 1) Check user login
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # 2) Get conversation_id from query params or session or however you prefer
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "No conversation_id provided"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Query only rows matching (user_id, conversation_id)
        cursor.execute("""
            SELECT key, value
            FROM currentroleplay
            WHERE user_id=%s
              AND conversation_id=%s
            ORDER BY key
        """, (user_id, conversation_id))
        rows = cursor.fetchall()

        data = [{"key": r[0], "value": r[1]} for r in rows]
        return jsonify(data), 200
    finally:
        conn.close()

def record_npc_event(user_id, conversation_id, npc_id, event_description):
    """
    Appends a new event to the NPC's memory field, for a specific user_id + conversation_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # We'll match the row by (npc_id, user_id, conversation_id)
        cursor.execute("""
            UPDATE NPCStats
            SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s::text)
            WHERE npc_id=%s
              AND user_id=%s
              AND conversation_id=%s
            RETURNING memory
        """, (event_description, npc_id, user_id, conversation_id))

        updated_row = cursor.fetchone()
        if not updated_row:
            print(f"NPC with ID={npc_id} (user_id={user_id}, conversation_id={conversation_id}) not found.")
        else:
            print(f"Updated memory for NPC={npc_id} => {updated_row[0]}")

        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error recording NPC event: {e}")
    finally:
        conn.close()

@memory_bp.route('/store_roleplay_segment', methods=['POST'])
def store_roleplay_segment():
    """
    Stores or updates a key-value pair in the CurrentRoleplay table,
    scoped to user_id + conversation_id.
    The payload should have:
      { "conversation_id": X, "key": "abc", "value": "..." }
    """
    # 1) Check user login
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    try:
        payload = request.get_json() or {}
        conversation_id = payload.get("conversation_id")
        segment_key = payload.get("key")
        segment_value = payload.get("value")

        if not conversation_id:
            return jsonify({"error": "No conversation_id provided"}), 400
        if not segment_key or segment_value is None:
            return jsonify({"error": "Missing 'key' or 'value'"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # 2) Insert or update the row with (user_id, conversation_id, key)
        # We'll need a unique or primary key on (user_id, conversation_id, key) in CurrentRoleplay
        # so we can do ON CONFLICT
        cursor.execute("""
            INSERT INTO currentroleplay (user_id, conversation_id, key, value)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, (user_id, conversation_id, segment_key, segment_value))

        conn.commit()
        return jsonify({"message": "Stored successfully"}), 200
    except Exception as e:
        print(f"Error in store_roleplay_segment: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()
