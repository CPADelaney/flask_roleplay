from quart import Blueprint, jsonify
import traceback
import logging
from db.connection import get_db_connection_context

debug_bp = Blueprint("debug_bp", __name__)

@debug_bp.route("/debug/connections", methods=["GET"])
async def debug_connections():
    """Debug endpoint to test database connections"""
    try:
        async with get_db_connection_context() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1 as test")
                row = await cursor.fetchone()
                
        return jsonify({
            "status": "success",
            "message": "Database connection is working",
            "test_result": row[0] if row else None
        })
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Database connection error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

from logic.inventory_system_sdk import get_call_report

@debug_bp.route('/inventory/calls', methods=['GET'])
async def debug_inventory_calls():
    """Get inventory call report for debugging"""
    report = await get_call_report()
    return jsonify(report)
