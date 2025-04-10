# routes/universal_update.py

from flask import Blueprint, request, jsonify, session
import asyncio
import asyncpg
import os
from db.connection import get_db_connection_context
from logic.universal_updater_agent import apply_universal_updates_async 

universal_bp = Blueprint("universal_bp", __name__)

@universal_bp.route("/universal_update", methods=["POST"])
async def universal_update():
    # 1) Get user_id from session
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # 2) Get JSON payload and conversation_id
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "No conversation_id provided"}), 400

    # 3) Use the async context manager
    async with get_db_connection_context() as conn:
        try:
            # 4) Call the updater with all required parameters
            result = await apply_universal_updates_async(user_id, conversation_id, data, conn)
            if "error" in result:
                return jsonify(result), 500
            else:
                return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@universal_bp.route("/get_roleplay_value", methods=["GET"])
async def get_roleplay_value():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
        
    conversation_id = request.args.get("conversation_id")
    key = request.args.get("key")
    
    if not conversation_id or not key:
        return jsonify({"error": "Missing required parameters"}), 400
    
    async with get_db_connection_context() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key=%s
            """, (user_id, conversation_id, key))
            row = await cursor.fetchone()
    
    if row:
        return jsonify({"value": row[0]})
    else:
        return jsonify({"value": None})
