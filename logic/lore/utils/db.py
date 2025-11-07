"""Database utility helpers that rely on the shared asyncpg connection pool."""

import logging
from typing import Any, Dict, List, Optional

import asyncpg

from db.connection import close_connection_pool
from lore.cache_version import get_lore_db_connection_context

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database-related errors."""

    pass


async def execute_query(
    query: str, params: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """Execute a database query with optional parameters."""

    try:
        param_values = list(params.values()) if params else []
        async with get_lore_db_connection_context() as conn:
            if query.strip().upper().startswith("SELECT"):
                result = await conn.fetch(query, *param_values)
                return result

            result = await conn.execute(query, *param_values)
            return [result]

    except asyncpg.PostgresError as e:
        logger.error(f"Query execution failed: {e}")
        raise DatabaseError(f"Query execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in execute_query: {e}")
        raise DatabaseError(f"Unexpected error: {str(e)}")


async def execute_many(query: str, params_list: List[Dict[str, Any]]) -> None:
    """Execute a database query multiple times with different parameters."""

    try:
        async with get_lore_db_connection_context() as conn:
            async with conn.transaction():
                for params in params_list:
                    param_values = list(params.values())
                    await conn.execute(query, *param_values)

    except asyncpg.PostgresError as e:
        logger.error(f"Batch query execution failed: {e}")
        raise DatabaseError(f"Batch query execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in execute_many: {e}")
        raise DatabaseError(f"Unexpected error: {str(e)}")


async def fetch_one(
    query: str, params: Optional[Dict[str, Any]] = None
) -> Optional[asyncpg.Record]:
    """Execute a query and fetch one row as an asyncpg.Record."""

    try:
        param_values = list(params.values()) if params else []
        async with get_lore_db_connection_context() as conn:
            row = await conn.fetchrow(query, *param_values)
            return row

    except asyncpg.PostgresError as e:
        logger.error(f"Query execution failed: {e}")
        raise DatabaseError(f"Query execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in fetch_one: {e}")
        raise DatabaseError(f"Unexpected error: {str(e)}")


async def fetch_value(query: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Execute a query and fetch a single value."""

    try:
        param_values = list(params.values()) if params else []
        async with get_lore_db_connection_context() as conn:
            value = await conn.fetchval(query, *param_values)
            return value

    except asyncpg.PostgresError as e:
        logger.error(f"Query execution failed: {e}")
        raise DatabaseError(f"Query execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in fetch_value: {e}")
        raise DatabaseError(f"Unexpected error: {str(e)}")


async def close_pool() -> None:
    """Close the shared database connection pool using the global manager."""

    await close_connection_pool()
