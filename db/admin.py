# db/admin.py

from flask import Blueprint, jsonify
from db.connection import get_db_connection
from db.initialization import initialize_database
from logic.stats_logic import insert_stat_definitions, insert_or_update_game_rules
from routes.settings_routes import insert_missing_settings
from logic.initialization import initialize_all_data

admin_bp = Blueprint('admin_bp', __name__)

@admin_bp.route('/init_everything', methods=['POST'])
def init_everything():
    """
    Single endpoint to do all DB creation + insertion in one go.
    """
    try:
        initialize_all_data()
        return jsonify({"message": "All data initialized successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
