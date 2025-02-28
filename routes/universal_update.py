# routes/universal_update.py
from flask import Blueprint, request, jsonify, session
import asyncio
import asyncpg
import os
from db.connection import get_db_connection
from logic.universal_updater import apply_universal_updates_async

DB_DSN = os.getenv("DB_DSN")

universal_bp = Blueprint("universal_bp", __name__)

@universal_bp.route("/universal_update", methods=["POST"])
async def universal_update():
    # 1) Get user_id from session
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # 2) Get JSON payload and conversation_id
    # Remove the 'await' since request.get_json() is synchronous in Flask
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "No conversation_id provided"}), 400

    # 3) Open an async DB connection
    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        # 4) Call the updater with all required parameters
        result = await apply_universal_updates_async(user_id, conversation_id, data, conn)
        if "error" in result:
            return jsonify(result), 500
        else:
            return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        await conn.close()

@universal_bp.route("/get_roleplay_value", methods=["GET"])
def get_roleplay_value():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
        
    conversation_id = request.args.get("conversation_id")
    key = request.args.get("key")
    
    if not conversation_id or not key:
        return jsonify({"error": "Missing required parameters"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT value FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key=%s
    """, (user_id, conversation_id, key))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if row:
        return jsonify({"value": row[0]})
    else:
        return jsonify({"value": None})
