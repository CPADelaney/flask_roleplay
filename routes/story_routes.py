# routes/story_routes.py

from flask import Blueprint, jsonify, request
from logic.aggregator import get_aggregated_roleplay_context

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/aggregated_context", methods=["GET"])
def aggregated_context():
    """
    Returns a single JSON object containing all roleplay data:
    Player stats, NPC stats, meltdown states, environment from CurrentRoleplay, etc.
    """
    player_name = request.args.get("player_name", "Chase")  # or parse differently
    data = get_aggregated_roleplay_context(player_name)
    return jsonify(data), 200
