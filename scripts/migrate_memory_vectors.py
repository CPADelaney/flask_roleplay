#!/usr/bin/env python3
"""Align pgvector column dimensions with the configured embedding size."""

from __future__ import annotations

import asyncio
import os
from typing import List, Sequence, Tuple

import asyncpg

from utils.embedding_dimensions import get_target_embedding_dimension


def _quote_ident(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier."""

    return '"' + identifier.replace('"', '""') + '"'


async def _fetch_vector_columns(conn: asyncpg.Connection) -> List[Tuple[str, str, str]]:
    """Return a list of (schema, table, column) for pgvector columns."""

    rows: Sequence[asyncpg.Record] = await conn.fetch(
        """
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        WHERE udt_name = 'vector'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, column_name
        """
    )
    return [(row["table_schema"], row["table_name"], row["column_name"]) for row in rows]


async def _alter_column_dimension(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
    dimension: int,
) -> None:
    qualified_table = f"{_quote_ident(schema)}.{_quote_ident(table)}"
    if not schema or schema == "public":
        qualified_table = _quote_ident(table)

    alter_sql = (
        f"ALTER TABLE {qualified_table} "
        f"ALTER COLUMN {_quote_ident(column)} TYPE vector({dimension});"
    )
    await conn.execute(alter_sql)


async def _validate_column_dimension(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
    dimension: int,
) -> bool:
    qualified_table = f"{_quote_ident(schema)}.{_quote_ident(table)}"
    if not schema or schema == "public":
        qualified_table = _quote_ident(table)

    validation_sql = (
        f"SELECT vector_dims({_quote_ident(column)}) AS dims "
        f"FROM {qualified_table} "
        f"WHERE {_quote_ident(column)} IS NOT NULL "
        "LIMIT 1"
    )
    row = await conn.fetchrow(validation_sql)
    if row is None:
        # No data to validate – treat as success after ALTER TABLE
        return True
    return row["dims"] == dimension


async def main() -> None:
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise SystemExit("DB_DSN environment variable is required for migration")

    target_dimension = get_target_embedding_dimension()
    async with asyncpg.create_pool(dsn) as pool:
        async with pool.acquire() as conn:
            columns = await _fetch_vector_columns(conn)
            if not columns:
                print("No pgvector columns found; nothing to migrate.")
                return

            print(
                f"Updating {len(columns)} vector column(s) to dimension {target_dimension}..."
            )
            for schema, table, column in columns:
                await _alter_column_dimension(conn, schema, table, column, target_dimension)
                is_valid = await _validate_column_dimension(
                    conn, schema, table, column, target_dimension
                )
                status = "OK" if is_valid else "CHECK"
                qualified_table = f"{schema}.{table}" if schema else table
                print(f" - {qualified_table}.{column}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
