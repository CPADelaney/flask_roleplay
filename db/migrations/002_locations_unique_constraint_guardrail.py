import logging
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

description = "Ensure Locations unique index is enforced as a named constraint"


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
    """Remove duplicate Locations rows prior to enforcing uniqueness."""

    duplicate_rows = await conn.fetch(
        """
        SELECT array_agg(id ORDER BY id) AS ids
        FROM Locations
        GROUP BY user_id, conversation_id, location_name
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
            "Removed %s duplicate Locations rows for canonical id %s",
            len(redundant_ids),
            keep_id,
        )


async def _get_unique_constraint(conn) -> Optional[Tuple[str, bool]]:
    """Return the unique constraint for the Locations natural key, if any."""

    row = await conn.fetchrow(
        """
        SELECT
            con.conname AS constraint_name,
            con.convalidated AS is_valid
        FROM pg_constraint AS con
        JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS cols(attnum, ordinality) ON TRUE
        JOIN pg_attribute AS att ON att.attrelid = con.conrelid AND att.attnum = cols.attnum
        WHERE con.conrelid = 'Locations'::regclass
          AND con.contype = 'u'
        GROUP BY con.conname, con.convalidated
        HAVING array_agg(att.attname ORDER BY cols.ordinality) = ARRAY['user_id', 'conversation_id', 'location_name']
        LIMIT 1
        """
    )

    if not row:
        return None

    return row["constraint_name"], bool(row["is_valid"])


async def _get_unique_index(conn) -> Optional[Tuple[str, bool]]:
    """Return a matching unique index for the Locations natural key, if any."""

    row = await conn.fetchrow(
        """
        SELECT
            cls.relname AS index_name,
            idx.indisvalid AS is_valid
        FROM pg_index AS idx
        JOIN pg_class AS tbl ON tbl.oid = idx.indrelid
        JOIN pg_namespace AS ns ON ns.oid = tbl.relnamespace
        JOIN pg_class AS cls ON cls.oid = idx.indexrelid
        JOIN LATERAL unnest(idx.indkey) WITH ORDINALITY AS cols(attnum, ordinality) ON TRUE
        JOIN pg_attribute AS att ON att.attrelid = tbl.oid AND att.attnum = cols.attnum
        WHERE ns.nspname = current_schema()
          AND tbl.relname = 'locations'
          AND idx.indisunique
        GROUP BY cls.relname, idx.indisvalid
        HAVING array_agg(att.attname ORDER BY cols.ordinality) = ARRAY['user_id', 'conversation_id', 'location_name']
        ORDER BY (cls.relname = 'idx_locations_user_conversation_name') DESC
        LIMIT 1
        """
    )

    if not row:
        return None

    return row["index_name"], bool(row["is_valid"])


async def upgrade(conn):
    """Promote the Locations unique index to a named constraint."""

    await _deduplicate_locations(conn)

    desired_name = "idx_locations_user_conversation_name"
    table_ident = await _quote_ident(conn, "Locations")
    desired_ident = await _quote_ident(conn, desired_name)

    constraint_info = await _get_unique_constraint(conn)
    if constraint_info:
        constraint_name, is_valid = constraint_info
        if not is_valid:
            await conn.execute(
                f"ALTER TABLE {table_ident} VALIDATE CONSTRAINT {await _quote_ident(conn, constraint_name)}"
            )
        if constraint_name != desired_name:
            await conn.execute(f"DROP INDEX IF EXISTS {desired_ident}")
            await conn.execute(
                f"ALTER TABLE {table_ident} RENAME CONSTRAINT {await _quote_ident(conn, constraint_name)} TO {desired_ident}"
            )
        return

    index_info = await _get_unique_index(conn)
    if index_info:
        index_name, is_valid = index_info
        index_ident = await _quote_ident(conn, index_name)
        if not is_valid:
            await conn.execute(f"DROP INDEX {index_ident}")
            index_info = None
        else:
            if index_name != desired_name:
                await conn.execute(f"DROP INDEX IF EXISTS {desired_ident}")
                await conn.execute(f"ALTER INDEX {index_ident} RENAME TO {desired_ident}")
                index_ident = desired_ident
            await conn.execute(
                f"ALTER TABLE {table_ident} ADD CONSTRAINT {desired_ident} UNIQUE USING INDEX {index_ident}"
            )
            return

    # No suitable index remains; create the constraint directly.
    await conn.execute(f"DROP INDEX IF EXISTS {desired_ident}")
    await conn.execute(
        f"ALTER TABLE {table_ident} ADD CONSTRAINT {desired_ident} UNIQUE (user_id, conversation_id, location_name)"
    )


async def downgrade(conn):
    """Revert the constraint back to a standalone unique index."""

    desired_name = "idx_locations_user_conversation_name"
    table_ident = await _quote_ident(conn, "Locations")
    desired_ident = await _quote_ident(conn, desired_name)

    await conn.execute(
        f"ALTER TABLE {table_ident} DROP CONSTRAINT IF EXISTS {desired_ident}"
    )
    await conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_locations_user_conversation_name
        ON Locations (user_id, conversation_id, location_name)
        """
    )
