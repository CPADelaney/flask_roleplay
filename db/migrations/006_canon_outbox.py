"""Create canon.outbox table for post-turn side effects."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

description = "Create canon.outbox table"


CREATE_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS canon;
"""

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS canon.outbox (
    id BIGSERIAL PRIMARY KEY,
    payload JSONB NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    next_run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dead_lettered BOOLEAN NOT NULL DEFAULT FALSE
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS canon_outbox_next_run_at_idx
    ON canon.outbox (next_run_at)
    WHERE dead_lettered = FALSE;
"""

DROP_INDEX_SQL = """
DROP INDEX IF EXISTS canon_outbox_next_run_at_idx;
"""

DROP_TABLE_SQL = """
DROP TABLE IF EXISTS canon.outbox;
"""


async def upgrade(conn):
    """Install the canon.outbox table used for side-effect dispatch."""

    await conn.execute(CREATE_SCHEMA_SQL)
    await conn.execute(CREATE_TABLE_SQL)
    await conn.execute(CREATE_INDEX_SQL)
    logger.info("Created canon.outbox table and index")


async def downgrade(conn):
    """Remove the canon.outbox table."""

    await conn.execute(DROP_INDEX_SQL)
    await conn.execute(DROP_TABLE_SQL)
    logger.info("Dropped canon.outbox table and index")
