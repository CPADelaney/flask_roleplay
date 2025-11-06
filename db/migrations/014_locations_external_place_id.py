"""Add external place identifier column to Locations."""

from __future__ import annotations


description = "Add external_place_id column to Locations"

ADD_EXTERNAL_ID_COLUMN = """
ALTER TABLE Locations
ADD COLUMN IF NOT EXISTS external_place_id TEXT;
"""

CREATE_EXTERNAL_ID_INDEX = """
CREATE INDEX IF NOT EXISTS idx_locations_external_place_id
ON Locations (external_place_id)
WHERE external_place_id IS NOT NULL;
"""

DROP_EXTERNAL_ID_INDEX = """
DROP INDEX IF EXISTS idx_locations_external_place_id;
"""

DROP_EXTERNAL_ID_COLUMN = """
ALTER TABLE Locations
DROP COLUMN IF EXISTS external_place_id;
"""


async def upgrade(conn):
    """Apply the migration."""

    await conn.execute(ADD_EXTERNAL_ID_COLUMN)
    await conn.execute(CREATE_EXTERNAL_ID_INDEX)


async def downgrade(conn):
    """Revert the migration."""

    await conn.execute(DROP_EXTERNAL_ID_INDEX)
    await conn.execute(DROP_EXTERNAL_ID_COLUMN)
