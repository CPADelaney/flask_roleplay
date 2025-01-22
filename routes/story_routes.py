# routes/story_routes.py

from flask import Blueprint, request, jsonify
from logic.story_flow import next_storybeat

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat_route():
    """
    Example endpoint. JSON: {"player_name": "...", "user_input": "..."}
    Returns GPT's new narrative chunk.
    """
    data = request.get_json() or {}
    player_name = data.get("player_name", "Chase")
    user_input = data.get("user_input", "")

    gpt_text = next_storybeat(player_name, user_input)
    return jsonify({"story_output": gpt_text}), 200
