from quart import Blueprint, jsonify, request
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


@debug_bp.route('/ids', methods=['GET'])
async def debug_openai_ids():
    """Return recent OpenAI conversation identifiers for debugging."""

    conversation_id = request.args.get("conversation_id", type=int)
    user_id = request.args.get("user_id", type=int)
    limit = request.args.get("limit", type=int) or 5

    limit = max(1, min(limit, 50))

    params = []
    clauses = []

    if conversation_id is not None:
        params.append(conversation_id)
        clauses.append(f"conversation_id = ${len(params)}")

    if user_id is not None:
        params.append(user_id)
        clauses.append(f"user_id = ${len(params)}")

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    query = f"""
        SELECT
            id,
            user_id,
            conversation_id,
            openai_assistant_id,
            openai_thread_id,
            openai_run_id,
            openai_response_id,
            status,
            metadata,
            updated_at
        FROM openai_conversations
        {where_clause}
        ORDER BY updated_at DESC
        LIMIT ${len(params)}
    """

    async with get_db_connection_context() as conn:
        rows = await conn.fetch(query, *params)

    results = []
    for row in rows:
        record = dict(row)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        openai_conversation_id = metadata.get("openai_conversation_id")
        results.append({
            "row_id": record.get("id"),
            "conversation_id": record.get("conversation_id"),
            "user_id": record.get("user_id"),
            "openai_conversation_id": openai_conversation_id,
            "openai_thread_id": record.get("openai_thread_id"),
            "openai_run_id": record.get("openai_run_id"),
            "openai_response_id": record.get("openai_response_id"),
            "status": record.get("status"),
            "updated_at": record.get("updated_at").isoformat() if record.get("updated_at") else None,
        })

    logging.info(
        "Debug /ids fetched %s rows for conversation=%s user=%s", len(results), conversation_id, user_id
    )

    return jsonify(
        {
            "filters": {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "limit": limit,
            },
            "results": results,
        }
    )
