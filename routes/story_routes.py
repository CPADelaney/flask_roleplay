# routes/story_routes.py

from flask import Blueprint, request, jsonify
from logic.story_flow import next_storybeat  # We'll create this orchestrator

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat_route():
    """
    Orchestrates the entire behind-the-scenes logic, calls GPT, returns the next narrative chunk.
    JSON: {"player_name": "...", "user_input": "..."}
    """
    data = request.get_json() or {}
    player_name = data.get("player_name", "Chase")
    user_input = data.get("user_input", "")

    try:
        story_output = next_storybeat(player_name, user_input)
        return jsonify({"story_output": story_output}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
