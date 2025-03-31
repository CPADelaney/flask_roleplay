# logic/lore/utils/db.py

"""
Database utility functions for the Lore System.

Refactored to use asyncpg and async/await pattern.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
import asyncpg
from ..config.settings import config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

async def get_db_pool():
    """Get a connection pool for database operations."""
    try:
        # Create a connection pool
        pool = await asyncpg.create_pool(
            dsn='postgresql://user:password@localhost:5432/lore_db',
            min_size=config.DB_POOL_SIZE,
            max_size=config.DB_POOL_SIZE + config.DB_MAX_OVERFLOW,
            command_timeout=config.DB_POOL_TIMEOUT
        )
        return pool
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        raise DatabaseError(config.ERROR_MESSAGES["db_connection"]) from e

# Global pool variable
_pool = None

async def get_db_connection():
    """Get a connection from the pool."""
    global _pool
    if _pool is None:
        _pool = await get_db_pool()
    return await _pool.acquire()

async def release_db_connection(conn):
    """Release a connection back to the pool."""
    global _pool
    if _pool is not None:
        await _pool.release(conn)

async def execute_query(query: str, params: Optional[dict] = None) -> list:
    """
    Execute a database query with parameters.
    
    Args:
        query: SQL query string
        params: Optional dictionary of query parameters
        
    Returns:
        List of query results
        
    Raises:
        DatabaseError: If query execution fails
    """
    conn = None
    try:
        conn = await get_db_connection()
        
        param_values = list(params.values()) if params else []
        
        # For SELECT queries (fetch data)
        if query.strip().upper().startswith("SELECT"):
            result = await conn.fetch(query, *param_values)
            return result
        
        # For other queries (INSERT, UPDATE, DELETE)
        else:
            result = await conn.execute(query, *param_values)
            return [result]
            
    except asyncpg.PostgresError as e:
        logger.error(f"Query execution failed: {e}")
        raise DatabaseError(f"Query execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in execute_query: {e}")
        raise DatabaseError(f"Unexpected error: {str(e)}")
    finally:
        if conn:
            await release_db_connection(conn)

async def execute_many(query: str, params_list: list) -> None:
    """
    Execute a database query multiple times with different parameters.
    
    Args:
        query: SQL query string
        params_list: List of parameter dictionaries
        
    Raises:
        DatabaseError: If query execution fails
    """
    conn = None
    try:
        conn = await get_db_connection()
        
        # Process each set of parameters
        results = []
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
    finally:
        if conn:
            await release_db_connection(conn)

async def fetch_one(query: str, params: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """
    Execute a query and fetch one row as a dictionary.
    
    Args:
        query: SQL query string
        params: Optional dictionary of query parameters
        
    Returns:
        Dictionary with column names as keys, or None if no row found
        
    Raises:
        DatabaseError: If query execution fails
    """
    conn = None
    try:
        conn = await get_db_connection()
        
        param_values = list(params.values()) if params else []
        row = await conn.fetchrow(query, *param_values)
        
        if row:
            # Convert Row to dictionary
            return dict(row)
        return None
            
    except asyncpg.PostgresError as e:
        logger.error(f"Query execution failed: {e}")
        raise DatabaseError(f"Query execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in fetch_one: {e}")
        raise DatabaseError(f"Unexpected error: {str(e)}")
    finally:
        if conn:
            await release_db_connection(conn)

async def fetch_value(query: str, params: Optional[dict] = None) -> Any:
    """
    Execute a query and fetch a single value.
    
    Args:
        query: SQL query string
        params: Optional dictionary of query parameters
        
    Returns:
        Single value result or None if no result
        
    Raises:
        DatabaseError: If query execution fails
    """
    conn = None
    try:
        conn = await get_db_connection()
        
        param_values = list(params.values()) if params else []
        value = await conn.fetchval(query, *param_values)
        
        return value
            
    except asyncpg.PostgresError as e:
        logger.error(f"Query execution failed: {e}")
        raise DatabaseError(f"Query execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in fetch_value: {e}")
        raise DatabaseError(f"Unexpected error: {str(e)}")
    finally:
        if conn:
            await release_db_connection(conn)

async def close_pool():
    """Close the database connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")
