from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection  # We import the function here

multiuser_bp = Blueprint("multiuser_bp", __name__)

@multiuser_bp.route("/folders", methods=["POST"])
def create_folder():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    folder_name = data.get("folder_name", "").strip()
    if not folder_name:
        return jsonify({"error": "No folder name provided"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    # Insert new folder
    cur.execute("""
        INSERT INTO folders (user_id, folder_name)
        VALUES (%s, %s)
        RETURNING id
    """, (user_id, folder_name))
    folder_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"folder_id": folder_id, "folder_name": folder_name})

@multiuser_bp.route("/folders", methods=["GET"])
def list_folders():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, folder_name, created_at
        FROM folders
        WHERE user_id = %s
        ORDER BY created_at
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "folder_id": r[0],
            "folder_name": r[1],
            "created_at": r[2].isoformat() if r[2] else None
        })
    return jsonify({"folders": results})

@multiuser_bp.route("/folders/<int:folder_id>", methods=["PUT"])
def rename_folder(folder_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    new_name = data.get("folder_name", "").strip()
    if not new_name:
        return jsonify({"error": "No folder name"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    # check folder ownership
    cur.execute("SELECT user_id FROM folders WHERE id=%s", (folder_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"error":"Folder not found"}), 404
    if row[0] != user_id:
        conn.close()
        return jsonify({"error":"Unauthorized"}), 403

    # rename
    cur.execute("UPDATE folders SET folder_name=%s WHERE id=%s", (new_name, folder_id))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message":"Folder renamed"})

@multiuser_bp.route("/conversations/<int:conv_id>/move_folder", methods=["POST"])
def move_folder_auto_create(conv_id):
    """
    Creates folder if needed, then moves conversation to folder_id
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    folder_name = data.get("folder_name", "").strip()
    if not folder_name:
        return jsonify({"error": "No folder name provided"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    # Check conversation ownership
    cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"error":"Conversation not found"}),404
    if row[0] != user_id:
        conn.close()
        return jsonify({"error":"Unauthorized"}),403

    # Check if folder already exists for this user
    cur.execute("""
        SELECT id FROM folders
        WHERE user_id=%s AND folder_name ILIKE %s
    """, (user_id, folder_name))
    frow = cur.fetchone()

    if frow:
        # folder already exists
        folder_id = frow[0]
    else:
        # Create a new folder
        cur.execute("""
            INSERT INTO folders (user_id, folder_name)
            VALUES (%s, %s) RETURNING id
        """, (user_id, folder_name))
        folder_id = cur.fetchone()[0]

    # Now update the conversation's folder_id
    cur.execute("""
        UPDATE conversations
        SET folder_id=%s
        WHERE id=%s
    """, (folder_id, conv_id))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": f"Conversation moved to folder '{folder_name}' (ID={folder_id})"})


@multiuser_bp.route("/folders/<int:folder_id>", methods=["DELETE"])
def delete_folder(folder_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    # check ownership
    cur.execute("SELECT user_id FROM folders WHERE id=%s", (folder_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"error":"Folder not found"}), 404
    if row[0] != user_id:
        conn.close()
        return jsonify({"error":"Unauthorized"}),403

    # If using ON DELETE SET NULL, removing folder won't remove convos.
    # If you used ON DELETE CASCADE, removing folder would also remove convos.
    cur.execute("DELETE FROM folders WHERE id=%s", (folder_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message":"Folder deleted"})

@multiuser_bp.route("/conversations", methods=["GET"])
def list_conversations():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, conversation_name
        FROM conversations
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    conversations = [{"id": r[0], "name": r[1]} for r in rows]
    return jsonify(conversations)

@multiuser_bp.route("/conversations", methods=["POST"])
def create_conversation():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    name = data.get("conversation_name", "Untitled Session")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (user_id, conversation_name)
        VALUES (%s, %s) RETURNING id
    """, (user_id, name))
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"conversation_id": new_id, "conversation_name": name})

@multiuser_bp.route("/conversations/<int:conv_id>/messages", methods=["GET"])
def get_messages(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    # Check conversation ownership
    cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conv_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return jsonify({"error": "Conversation not found"}), 404
    if row[0] != user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403

    # Fetch messages
    cur.execute("""
        SELECT sender, content, created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY id ASC
    """, (conv_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    messages = [
        {"sender": r[0], "content": r[1], "created_at": r[2].isoformat()}
        for r in rows
    ]
    return jsonify({"messages": messages})

@multiuser_bp.route("/conversations/<int:conv_id>/messages", methods=["POST"])
def add_message(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    # Check ownership again
    cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conv_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return jsonify({"error": "Conversation not found"}), 404
    if row[0] != user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403

    # Insert the message
    data = request.get_json()
    sender = data.get("sender", "user")
    content = data.get("content", "")

    cur.execute("""
        INSERT INTO messages (conversation_id, sender, content)
        VALUES (%s, %s, %s)
    """, (conv_id, sender, content))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"status": "ok"})

# RENAME
@multiuser_bp.route("/conversations/<int:conv_id>", methods=["PUT"])
def rename_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    new_name = data.get("conversation_name", "New Chat")

    conn = get_db_connection()
    cur = conn.cursor()

    # Check ownership
    cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Conversation not found"}), 404
    if row[0] != user_id:
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403

    # Rename
    cur.execute("""
        UPDATE conversations
        SET conversation_name = %s
        WHERE id = %s
    """, (new_name, conv_id))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Renamed"})

# Move
@multiuser_bp.route("/conversations/<int:conv_id>/folder", methods=["POST"])
def move_conversation_to_folder(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    new_folder_id = data.get("folder_id")

    # For 'inbox' or 'no folder', you can pass folder_id=None
    # or pass 0, etc.

    conn = get_db_connection()
    cur = conn.cursor()

    # check conversation ownership
    cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"error":"Conversation not found"}), 404
    if row[0] != user_id:
        conn.close()
        return jsonify({"error":"Unauthorized"}), 403

    # If new_folder_id is not None, verify that folder belongs to user
    if new_folder_id:
        cur.execute("SELECT user_id FROM folders WHERE id=%s",(new_folder_id,))
        frow = cur.fetchone()
        if not frow:
            conn.close()
            return jsonify({"error":"Folder not found"}),404
        if frow[0] != user_id:
            conn.close()
            return jsonify({"error":"Unauthorized folder"}),403

    # set folder_id
    cur.execute("""
        UPDATE conversations
        SET folder_id=%s
        WHERE id=%s
    """, (new_folder_id, conv_id))

    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"message":"Conversation moved"})


# DELETE
@multiuser_bp.route("/conversations/<int:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error":"Not logged in"}),401

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT user_id FROM conversations WHERE id=%s",(conv_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return jsonify({"error":"Conversation not found"}),404
    if row[0]!=user_id:
        cur.close()
        conn.close()
        return jsonify({"error":"Unauthorized"}),403

    # This will also delete all messages referencing conversation_id
    cur.execute("DELETE FROM conversations WHERE id=%s",(conv_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message":"Deleted"})

