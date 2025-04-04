# nyx/core/sync/noise_filter.py

import asyncpg

async def get_noisy_responses(conn, limit=20):
    return await conn.fetch("""
        SELECT * FROM nyx1_response_noise
        WHERE marked_for_review = TRUE AND dismissed = FALSE
        ORDER BY created_at DESC LIMIT $1
    """, limit)

async def purge_old_noise(conn, max_age_days=14):
    await conn.execute("""
        DELETE FROM nyx1_response_noise
        WHERE created_at < now() - interval '$1 days'
    """, max_age_days)
