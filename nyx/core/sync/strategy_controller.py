# nyx/core/sync/strategy_controller.py

import asyncpg
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
