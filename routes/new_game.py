# routes/new_game.py

import logging
import json
import random
import time
import os
import asyncio
from flask import Blueprint, request, jsonify, session
import asyncpg
import openai

from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator_sdk import get_aggregated_roleplay_context
from db.connection import get_db_connection  # Not used anymore if using asyncpg
from routes.story_routes import build_aggregator_text

# Use your Railway DSN (public URL for local development)
DB_DSN = os.getenv("DB_DSN") 
logging.info(f"[new_game] Using DB_DSN={DB_DSN}")  # Add right after retrieving


new_game_bp = Blueprint('new_game_bp', __name__)

def create_conversation_sync(user_id):
    async def _create_conversation():
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            preliminary_name = "New Game"
            status = "processing"
            row = await conn.fetchrow("""
                INSERT INTO conversations (user_id, conversation_name, status)
                VALUES ($1, $2, $3)
                RETURNING id
            """, user_id, preliminary_name, status)
            conversation_id = row["id"]
            logging.info(f"Created conversation_id: {conversation_id} for user_id: {user_id}")
            return conversation_id
        finally:
            await conn.close()
    return asyncio.run(_create_conversation())


# --- Helper: a spaced-out GPT call (used by the background task if needed) ---
async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0):
    """
    Waits for a short delay and then calls get_chatgpt_response in a thread.
    Adjust delay (in seconds) as needed to spread out API calls.
    """
    await asyncio.sleep(delay)
    return await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    from tasks import process_new_game_task  # Import task that uses NewGameAgent and Nyx governance
    logging.info("=== /start_new_game endpoint called (offloading to background task using NewGameAgent) ===")
    
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conversation_data = request.get_json() or {}

    if not conversation_data.get("conversation_id"):
        conversation_id = create_conversation_sync(user_id)
        conversation_data["conversation_id"] = conversation_id
    else:
        conversation_id = conversation_data["conversation_id"]

    task = process_new_game_task.delay(user_id, conversation_data)
    logging.info(f"Enqueued process_new_game_task with NewGameAgent for user_id={user_id}, conversation_id={conversation_id}, task id: {task.id}")

    return jsonify({"job_id": task.id, "conversation_id": conversation_id}), 202

@new_game_bp.route('/conversation_status', methods=['GET'])
def conversation_status():
    conversation_id_str = request.args.get("conversation_id")
    user_id = session.get("user_id")
    if not user_id or not conversation_id_str:
        return jsonify({"error": "Missing parameters"}), 400

    try:
        conversation_id = int(conversation_id_str)
    except ValueError:
        return jsonify({"error": "Invalid conversation_id parameter"}), 400

    async def _get_status():
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            row = await conn.fetchrow(
                """
                SELECT conversation_name, status 
                FROM conversations 
                WHERE id=$1 AND user_id=$2
                """,
                conversation_id, user_id
            )
            if row:
                return {"conversation_name": row["conversation_name"], "status": row["status"]}
            else:
                return {"error": "Conversation not found"}
        finally:
            await conn.close()

    return jsonify(asyncio.run(_get_status()))
