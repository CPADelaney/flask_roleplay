# debug.py

debug_bp = Blueprint('debug_bp', __name__)

@debug_bp.route('/create_npc', methods=['GET'])
def debug_create_npc():
    new_id = create_npc(introduced=False)
    return jsonify({"message": f"Created new NPC with id {new_id}"}), 200

@debug_bp.route('/generate_setting', methods=['GET'])
def debug_generate_setting():
    mega_data = generate_mega_setting_logic()
    return jsonify(mega_data), 200
