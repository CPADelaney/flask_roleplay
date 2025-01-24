# routes/universal_update.py

from flask import Blueprint, request, jsonify
from logic.universal_updater import apply_universal_updates

universal_bp = Blueprint("universal_bp", __name__)

@universal_bp.route("/universal_update", methods=["POST"])
def universal_update():
    """
    Endpoint to handle universal updates in a single JSON payload.
    """
    try:
        data = request.get_json() or {}
        result = apply_universal_updates(data)

        if "error" in result:
            return jsonify(result), 500
        else:
            return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
