"""Add stored lowercase name column and unique index for Locations."""

from __future__ import annotations


description = "Add generated lowercase column and index for Locations names"

# CREATE UNIQUE INDEX CONCURRENTLY cannot run inside a transaction.
run_in_transaction = False


ADD_GENERATED_COLUMN = """
ALTER TABLE Locations
ADD COLUMN IF NOT EXISTS location_name_lc TEXT
GENERATED ALWAYS AS (lower(location_name)) STORED;
"""


CREATE_UNIQUE_INDEX = """
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_locations_user_conv_name_lc
ON Locations (user_id, conversation_id, location_name_lc);
"""


DROP_EXPRESSION_INDEX = """
DROP INDEX IF EXISTS idx_locations_user_conv_lower_name;
"""


DROP_UNIQUE_INDEX = """
DROP INDEX IF EXISTS idx_locations_user_conv_name_lc;
"""


DROP_GENERATED_COLUMN = """
ALTER TABLE Locations
DROP COLUMN IF EXISTS location_name_lc;
"""


CREATE_LEGACY_INDEX = """
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_locations_user_conv_lower_name
ON Locations (user_id, conversation_id, lower(location_name));
"""


async def upgrade(conn):
    await conn.execute(ADD_GENERATED_COLUMN)
    await conn.execute(CREATE_UNIQUE_INDEX)
    await conn.execute(DROP_EXPRESSION_INDEX)


async def downgrade(conn):
    await conn.execute(DROP_UNIQUE_INDEX)
    await conn.execute(DROP_GENERATED_COLUMN)
    await conn.execute(CREATE_LEGACY_INDEX)
