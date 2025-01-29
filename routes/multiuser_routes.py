# routes/multiuser_routes.py

from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection  # <--- Instead of from db import db

multiuser_bp = Blueprint("multiuser_bp", __name__)

@multiuser_bp.route("/conversations", methods=["GET"])
def list_conversations():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    cur = db.cursor()
    cur.execute("""
        SELECT id, conversation_name
        FROM conversations
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    conversations = [{"id": r[0], "name": r[1]} for r in rows]
    return jsonify(conversations)

@multiuser_bp.route("/conversations", methods=["POST"])
def create_conversation():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    name = data.get("conversation_name", "Untitled Session")

    cur = db.cursor()
    cur.execute("""
        INSERT INTO conversations (user_id, conversation_name)
        VALUES (%s, %s) RETURNING id
    """, (user_id, name))
    new_id = cur.fetchone()[0]
    db.commit()

    return jsonify({"conversation_id": new_id, "conversation_name": name})

@multiuser_bp.route("/conversations/<int:conv_id>/messages", methods=["GET"])
def get_messages(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # Check conversation ownership
    cur = db.cursor()
    cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conv_id,))
    row = cur.fetchone()
    if not row:
        return jsonify({"error": "Conversation not found"}), 404
    if row[0] != user_id:
        return jsonify({"error": "Unauthorized"}), 403

    # Fetch messages
    cur.execute("""
        SELECT sender, content, created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY id ASC
    """, (conv_id,))
    rows = cur.fetchall()
    messages = [
        {
            "sender": r[0],
            "content": r[1],
            "created_at": r[2].isoformat()
        } 
        for r in rows
    ]
    return jsonify({"messages": messages})

@multiuser_bp.route("/conversations/<int:conv_id>/messages", methods=["POST"])
def add_message(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # Check ownership again
    cur = db.cursor()
    cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conv_id,))
    row = cur.fetchone()
    if not row:
        return jsonify({"error": "Conversation not found"}), 404
    if row[0] != user_id:
        return jsonify({"error": "Unauthorized"}), 403

    # Insert the message
    data = request.get_json()
    sender = data.get("sender", "user")
    content = data.get("content", "")

    cur.execute("""
        INSERT INTO messages (conversation_id, sender, content)
        VALUES (%s, %s, %s)
    """, (conv_id, sender, content))
    db.commit()

    return jsonify({"status": "ok"})
