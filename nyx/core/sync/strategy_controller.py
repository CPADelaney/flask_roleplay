# nyx/core/sync/strategy_controller.py

import asyncpg
from typing import Dict, Any, List
from datetime import datetime, timedelta

async def insert_strategy(conn, name, type_, payload, ttl_hours=72):
    await conn.execute("""
        INSERT INTO nyx1_strategy_injections (strategy_name, strategy_type, payload, expires_at)
        VALUES ($1, $2, $3, $4)
    """, name, type_, payload, datetime.utcnow() + timedelta(hours=ttl_hours))

async def mark_strategy_expired(conn, strategy_id):
    await conn.execute("""
        UPDATE nyx1_strategy_injections SET status = 'expired' WHERE id = $1
    """, strategy_id)

async def dismiss_noise(conn, noise_id):
    await conn.execute("""
        UPDATE nyx1_response_noise SET dismissed = TRUE WHERE id = $1
    """, noise_id)

async def mark_noise_for_review(conn, noise_id):
    await conn.execute("""
        UPDATE nyx1_response_noise SET marked_for_review = TRUE WHERE id = $1
    """, noise_id)
    
async def get_active_strategies(conn) -> List[Dict[str, Any]]:
    records = await conn.fetch("""
        SELECT * FROM nyx1_strategy_injections
        WHERE status = 'active' AND (expires_at IS NULL OR expires_at > now())
        ORDER BY created_at DESC
    """)
    return [dict(r) for r in records]

async def mark_strategy_for_review(conn, strategy_id: int, user_id: int, reason: str):
    await conn.execute("""
        INSERT INTO nyx2_review_queue (strategy_id, user_id, reason)
        VALUES ($1, $2, $3)
    """, strategy_id, user_id, reason)
