from flask import Blueprint, jsonify
import traceback
from logic.npc_creation import create_npc_logic

debug_bp = Blueprint("debug_bp", __name__)

@debug_bp.route("/create_npc_safe", methods=["GET"])
def create_npc_safe():
    """
    Calls the create_npc_logic in a try/except,
    returning any exception & traceback as JSON.
    """
    try:
        new_id = create_npc_logic(introduced=False)
        return jsonify({"message": f"Created new NPC with id {new_id}"}), 200
    except Exception as e:
        tb_str = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "traceback": tb_str
        }), 500
