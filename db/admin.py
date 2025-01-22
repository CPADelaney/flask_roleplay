# routes/admin.py
from flask import Blueprint, jsonify, request
from db.connection import get_db_connection
from db.initialization import initialize_database
from logic.stats_logic import insert_stat_definitions, insert_game_rules, insert_missing_settings

admin_bp = Blueprint('admin_bp', __name__)

@admin_bp.route('/init_db_manual', methods=['POST'])
def init_db_manual():
    try:
        initialize_database()
        insert_game_rules()
        insert_stat_definitions()
        insert_missing_settings()
        return jsonify({"message": "DB initialized and settings inserted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@admin_bp.route('/test_db_connection', methods=['GET'])
def test_db_connection():
