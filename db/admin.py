# db/admin.py
from flask import Blueprint, jsonify
from db.schema_and_seed import initialize_all_data  # or wherever you seeded data

admin_bp = Blueprint('admin_bp', __name__)  # <-- define the Blueprint here

@admin_bp.route('/init_everything', methods=['POST'])
def init_everything():
    try:
        initialize_all_data()
        return jsonify({"message": "All data initialized successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
