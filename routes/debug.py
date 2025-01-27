from flask import Blueprint, jsonify
import traceback
import logging
from logic.npc_creation import create_npc

debug_bp = Blueprint("debug_bp", __name__)

@debug_bp.route("/create_npc_safe", methods=["GET"])
def create_npc_safe():
    """
    Calls the create_npc function in a try/except,
    returning any exception & traceback as JSON.
    """
    try:
        new_id = create_npc(introduced=False)
        return jsonify({"message": f"Created new NPC with id {new_id}"}), 200

    except Exception as e:
        # Format the traceback
        tb_str = traceback.format_exc()
        logging.error(f"Exception in create_npc_safe:\n{tb_str}")

        # Return JSON with the error info
        return jsonify({
            "error": str(e),
            "traceback": tb_str
        }), 500
