# nyx/core/sync/strategy_register.py

import asyncpg
from datetime import datetime

async def register_decision(conn, strategy_id, user_id, event_type, reason, snippet, kink_profile):
    await conn.execute("""
        INSERT INTO nyx1_strategy_logs (strategy_id, user_id, event_type, message_snippet, kink_profile, decision_meta)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, strategy_id, user_id, event_type, snippet, kink_profile, {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason
    })
