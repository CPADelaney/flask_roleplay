#!/usr/bin/env python3
"""Align pgvector column dimensions with the configured embedding size."""

from __future__ import annotations

import asyncio
import os
import re
from typing import List, Optional, Sequence, Tuple

import asyncpg

from utils.embedding_dimensions import get_target_embedding_dimension


def _quote_ident(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier."""

    return '"' + identifier.replace('"', '""') + '"'


def _qualified_table(schema: str, table: str) -> str:
    """Return a fully qualified and quoted table identifier."""

    if not schema:
        raise ValueError("Schema name is required for fully qualified identifiers")
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


async def _fetch_vector_columns(
    conn: asyncpg.Connection,
) -> List[Tuple[str, str, str, Optional[int]]]:
    """Return a list of (schema, table, column, declared_dimension) for pgvector columns."""

    rows: Sequence[asyncpg.Record] = await conn.fetch(
        """
        SELECT
            n.nspname AS table_schema,
            c.relname AS table_name,
            a.attname AS column_name,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS formatted_type
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
        JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        WHERE NOT a.attisdropped
          AND a.attnum > 0
          AND c.relkind IN ('r', 'm', 'p', 'f')
          AND pg_catalog.format_type(a.atttypid, a.atttypmod) LIKE 'vector%'
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, column_name
        """
    )

    dimension_pattern = re.compile(r"vector\((\d+)\)")

    def _parse_dimension(formatted_type: str) -> Optional[int]:
        match = dimension_pattern.search(formatted_type)
        if not match:
            return None
        return int(match.group(1))

    return [
        (
            row["table_schema"],
            row["table_name"],
            row["column_name"],
            _parse_dimension(row["formatted_type"]),
        )
        for row in rows
    ]


async def _alter_column_dimension(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
    dimension: int,
) -> None:
    qualified_table = _qualified_table(schema, table)

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
) -> int:
    return await _count_mismatched_rows(conn, schema, table, column, dimension)


async def _count_mismatched_rows(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
    dimension: int,
) -> int:
    qualified_table = _qualified_table(schema, table)

    mismatch_sql = (
        f"SELECT COUNT(*) AS mismatches "
        f"FROM {qualified_table} "
        f"WHERE {_quote_ident(column)} IS NOT NULL "
        f"  AND vector_dims({_quote_ident(column)}) <> {dimension}"
    )
    row = await conn.fetchrow(mismatch_sql)
    return int(row["mismatches"]) if row is not None else 0


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
            for schema, table, column, declared_dimension in columns:
                qualified_table = f"{schema}.{table}" if schema else table

                if declared_dimension == target_dimension:
                    print(
                        " - "
                        f"{qualified_table}.{column}: declared dimension already {target_dimension}, skipping"
                    )
                    continue

                mismatched_rows = await _count_mismatched_rows(
                    conn, schema, table, column, target_dimension
                )
                if mismatched_rows > 0:
                    print(
                        " - "
                        f"{qualified_table}.{column}: "
                        f"{mismatched_rows} row(s) have vector_dims <> {target_dimension}; skipping"
                    )
                    continue

                await _alter_column_dimension(conn, schema, table, column, target_dimension)
                validation_errors = await _validate_column_dimension(
                    conn, schema, table, column, target_dimension
                )
                if validation_errors:
                    print(
                        " - "
                        f"{qualified_table}.{column}: ERROR - "
                        f"{validation_errors} row(s) still have vector_dims <> {target_dimension}"
                    )
                else:
                    print(f" - {qualified_table}.{column}: OK")


if __name__ == "__main__":
    asyncio.run(main())
