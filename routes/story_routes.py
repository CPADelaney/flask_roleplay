# routes/story_routes.py

from flask import Blueprint, request, jsonify
from logic.stats_logic import update_player_stats  # if you want
from logic.meltdown_logic import remove_meltdown_npc
from logic.aggregator import get_aggregated_roleplay_context
# etc. import your aggregator or meltdown calls

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    """
    This route receives {"player_name":..., "user_input":...}.
    1) Parse user_input for meltdown triggers or stat changes
    2) Possibly update DB
    3) Build aggregator context from DB
    4) Return final 'story_output'
    """

    data = request.get_json() or {}
    player_name = data.get("player_name", "Chase")
    user_input = data.get("user_input", "")

    # 1) Check if user_input implies meltdown removal, stats changes, etc.
    user_lower = user_input.lower()
    if "obedience=100" in user_lower:
        # Example of forcing Obedience=100
        # either call your logic or direct DB update
        # e.g.:
        update_player_stat(player_name, "obedience", 100)

    if "remove meltdown" in user_lower:
        # meltdown removal
        remove_meltdown_npc(force=True)

    # 2) Possibly generate new environment if user_input asks for it
    if "generate environment" in user_lower or "mega setting" in user_lower:
        # call your code that does /settings/generate_mega_setting logic
        generate_mega_setting_logic()

    # 3) Now build aggregator context from DB
    aggregator_data = get_aggregated_roleplay_context(player_name)
    # aggregator_data might be a big dict
    story_output = build_aggregator_text(aggregator_data)

    # 4) Return final scenario text
    return jsonify({"story_output": story_output}), 200


def update_player_stat(player_name, stat_key, value):
    # example code to do DB update for a single stat
    # or call logic/stats_logic.py
    pass

def generate_mega_setting_logic():
    # e.g. call a function that merges random settings, updates DB
    pass

def build_aggregator_text(aggregator_data):
    # merges aggregator_data into user-friendly text
    # same approach as the earlier example
    # ...
    return final_text
