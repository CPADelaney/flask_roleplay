"""Add cached culture summary columns to Nations."""

from __future__ import annotations

import logging


description = "Add cached culture summary columns to Nations"

logger = logging.getLogger(__name__)

ADD_COLUMNS = """
ALTER TABLE Nations
    ADD COLUMN IF NOT EXISTS culture_summary TEXT,
    ADD COLUMN IF NOT EXISTS culture_summary_updated_at TIMESTAMPTZ;
"""

BACKFILL_PLACEHOLDER = """
UPDATE Nations
SET culture_summary = 'Culture summary pending refresh',
    culture_summary_updated_at = NOW()
WHERE culture_summary IS NULL;
"""

DOWNGRADE_DROP = """
ALTER TABLE Nations
    DROP COLUMN IF EXISTS culture_summary,
    DROP COLUMN IF EXISTS culture_summary_updated_at;
"""


async def upgrade(conn):
    """Add cached culture summary columns to Nations and backfill placeholder values."""

    await conn.execute(ADD_COLUMNS)
    await conn.execute(BACKFILL_PLACEHOLDER)
    logger.info("Nations.culture_summary column added and initialized with placeholders")


async def downgrade(conn):
    """Remove cached culture summary columns from Nations."""

    await conn.execute(DOWNGRADE_DROP)
    logger.info("Nations.culture_summary columns dropped")
