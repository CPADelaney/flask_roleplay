from quart import Blueprint, jsonify
from logic.inventory_system_sdk import get_call_report

debug_inventory_bp = Blueprint('debug_inventory', __name__)

@debug_inventory_bp.route('/debug/inventory/calls', methods=['GET'])
async def debug_inventory_calls():
    """Get inventory call report for debugging"""
    report = await get_call_report()
    return jsonify(report)
