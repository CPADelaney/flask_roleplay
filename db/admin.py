# db/admin.py
from quart import Blueprint, jsonify
import asyncio
from db.schema_and_seed import initialize_all_data  # or wherever you seeded data

admin_bp = Blueprint('admin_bp', __name__)  # <-- define the Blueprint here

@admin_bp.route('/init_everything', methods=['POST'])
async def init_everything():
    """
    Initialize all database tables and seed data.
    This is now properly async to match initialize_all_data().
    """
    try:
        await initialize_all_data()
        return jsonify({"message": "All data initialized successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
