# routes/universal_update.py

from flask import Blueprint, request, jsonify
from logic.universal_updater import apply_universal_updates

universal_bp = Blueprint("universal_bp", __name__)

@universal_bp.route("/universal_update", methods=["POST"])
def universal_update():
    """
    Endpoint to handle universal updates in a single JSON payload.
    We'll attach user_id and conversation_id so each insert or update
    can store them in the DB.
    """
    try:
        # 1) Get user_id from session
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        # 2) Possibly get conversation_id from JSON or session
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "No conversation_id provided"}), 400

        # 3) Attach user_id and conversation_id to the data
        data["user_id"] = user_id
        data["conversation_id"] = conversation_id

        # 4) Pass augmented data to apply_universal_updates
        result = apply_universal_updates(data)

        if "error" in result:
            return jsonify(result), 500
        else:
            return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

