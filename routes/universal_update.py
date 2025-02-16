# routes/universal_update.py
from flask import Blueprint, request, jsonify, session
import asyncio
import asyncpg
from logic.universal_updater import apply_universal_updates_async
import os

DB_DSN = os.getenv("DB_DSN")

@universal_bp.route("/universal_update", methods=["POST"])
async def universal_update():
    # 1) Get user_id from session
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # 2) Get JSON payload and conversation_id
    data = await request.get_json()  # note: requires an async request handler
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
