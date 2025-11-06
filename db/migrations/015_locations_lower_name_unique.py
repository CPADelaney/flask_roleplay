"""Make Locations names unique per conversation case-insensitively."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)

description = "Ensure Locations names are unique per conversation ignoring case"

# Creating an expression index concurrently requires running outside a surrounding
# transaction. The migration runner inspects this flag to disable its automatic
# transaction wrapper for this migration.
run_in_transaction = False


async def _quote_ident(conn, *parts: str) -> str:
    """Return a properly quoted identifier from component parts."""

    if not parts:
        raise ValueError("At least one identifier part is required")
    return await conn.fetchval(
        "SELECT string_agg(format('%I', part), '.') FROM unnest($1::text[]) AS part",
        list(parts),
    )


async def _load_fk_targets(conn) -> Sequence[Tuple[str, str]]:
    """Return (qualified_table, column) pairs referencing locations.id."""

    rows = await conn.fetch(
        """
        SELECT
            n.nspname AS schema_name,
            c.relname AS table_name,
            a.attname AS column_name
        FROM pg_constraint AS fk
        JOIN pg_class AS c ON fk.conrelid = c.oid
        JOIN pg_namespace AS n ON c.relnamespace = n.oid
        JOIN LATERAL unnest(fk.conkey) AS colnum(attnum) ON TRUE
        JOIN pg_attribute AS a ON a.attrelid = fk.conrelid AND a.attnum = colnum.attnum
        WHERE fk.contype = 'f'
          AND fk.confrelid = 'locations'::regclass
        """
    )

    targets: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for row in rows:
        qualified_table = await _quote_ident(conn, row["schema_name"], row["table_name"])
        column_ident = await _quote_ident(conn, row["column_name"])
        key = (qualified_table, column_ident)
        if key in seen:
            continue
        seen.add(key)
        targets.append(key)
    return targets


async def _deduplicate_locations(conn) -> None:
    """Merge case-insensitive duplicate Location rows."""

    duplicate_rows = await conn.fetch(
        """
        SELECT array_agg(id ORDER BY id) AS ids
        FROM Locations
        GROUP BY user_id, conversation_id, lower(location_name)
        HAVING COUNT(*) > 1
        """
    )

    if not duplicate_rows:
        return

    fk_targets = await _load_fk_targets(conn)

    for row in duplicate_rows:
        ids: List[int] = list(row["ids"] or [])
        if len(ids) <= 1:
            continue

        keep_id = ids[0]
        redundant_ids = ids[1:]

        for qualified_table, column_ident in fk_targets:
            await conn.execute(
                f"UPDATE {qualified_table} SET {column_ident} = $1 WHERE {column_ident} = ANY($2::int[])",
                keep_id,
                redundant_ids,
            )

        await conn.execute(
            "DELETE FROM Locations WHERE id = ANY($1::int[])",
            redundant_ids,
        )
        logger.info(
            "Merged %s duplicate Locations rows for canonical id %s",
            len(redundant_ids),
            keep_id,
        )


CREATE_CASE_INSENSITIVE_INDEX = """
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_locations_user_conv_lower_name
ON Locations (user_id, conversation_id, lower(location_name));
"""

DROP_LEGACY_CONSTRAINT = """
ALTER TABLE Locations
DROP CONSTRAINT IF EXISTS idx_locations_user_conversation_name;
"""

DROP_LEGACY_INDEX = """
DROP INDEX IF EXISTS idx_locations_user_conversation_name;
"""

DROP_CASE_INSENSITIVE_INDEX = """
DROP INDEX IF EXISTS idx_locations_user_conv_lower_name;
"""


async def upgrade(conn):
    """Apply the migration."""

    async with conn.transaction():
        await _deduplicate_locations(conn)

    await conn.execute(CREATE_CASE_INSENSITIVE_INDEX)
    await conn.execute(DROP_LEGACY_CONSTRAINT)
    await conn.execute(DROP_LEGACY_INDEX)


async def downgrade(conn):
    """Revert the migration."""

    await conn.execute(DROP_CASE_INSENSITIVE_INDEX)
    await conn.execute(
        """
        DO $$
        BEGIN
            ALTER TABLE Locations
            ADD CONSTRAINT idx_locations_user_conversation_name
            UNIQUE (user_id, conversation_id, location_name);
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )
