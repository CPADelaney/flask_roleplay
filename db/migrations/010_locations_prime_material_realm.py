"""Adopt Prime Material as the default realm for Locations."""

from __future__ import annotations

import logging


description = "Set Prime Material as Locations.realm default"

logger = logging.getLogger(__name__)

BACKFILL_REALM = """
UPDATE Locations
SET realm = 'Prime Material'
WHERE realm IS NULL
   OR btrim(realm) = ''
   OR lower(realm) = 'physical';
"""

SET_DEFAULT_REALM = """
ALTER TABLE Locations
ALTER COLUMN realm SET DEFAULT 'Prime Material';
"""

DROP_NOT_NULL = """
ALTER TABLE Locations
ALTER COLUMN realm DROP NOT NULL;
"""

DOWNGRADE_BACKFILL = """
UPDATE Locations
SET realm = 'physical'
WHERE realm IS NULL
   OR btrim(realm) = ''
   OR realm = 'Prime Material';
"""

DOWNGRADE_SET_DEFAULT = """
ALTER TABLE Locations
ALTER COLUMN realm SET DEFAULT 'physical';
"""

DOWNGRADE_SET_NOT_NULL = """
ALTER TABLE Locations
ALTER COLUMN realm SET NOT NULL;
"""


async def upgrade(conn):
    """Backfill and relax the Locations.realm column."""

    await conn.execute(BACKFILL_REALM)
    await conn.execute(SET_DEFAULT_REALM)
    await conn.execute(DROP_NOT_NULL)
    logger.info("Locations.realm now defaults to Prime Material and allows NULL values")


async def downgrade(conn):
    """Restore the previous Locations.realm constraints."""

    await conn.execute(DOWNGRADE_BACKFILL)
    await conn.execute(DOWNGRADE_SET_DEFAULT)
    await conn.execute(DOWNGRADE_SET_NOT_NULL)
    logger.info("Locations.realm default reverted to 'physical' and column made NOT NULL")
