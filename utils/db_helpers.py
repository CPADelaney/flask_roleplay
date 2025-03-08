# utils/db_helpers.py

import os
import logging
import asyncio
import contextlib
from typing import Optional, Callable, Any, Dict, List, Tuple, Union

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logging.warning("asyncpg not available, falling back to synchronous connections")

# Global connection pool
DB_POOL = None

async def initialize_db_pool():
    """Initialize the database connection pool once at application startup."""
    global DB_POOL
    try:
        if not ASYNCPG_AVAILABLE:
            logging.warning("Cannot initialize pool: asyncpg not available")
            return None
            
        if DB_POOL is None:
            DB_POOL = await asyncpg.create_pool(
                dsn=os.getenv("DB_DSN"),
                min_size=5,
                max_size=20,
                command_timeout=60,
                statement_timeout=60,
                max_cached_statement_lifetime=300,
                max_inactive_connection_lifetime=300
            )
            logging.info("Database connection pool initialized")
            return DB_POOL
    except Exception as e:
        logging.error(f"Error initializing DB pool: {e}")
        return None

async def get_db_connection_async():
    """Get an async database connection from the pool."""
    global DB_POOL
    
    if not ASYNCPG_AVAILABLE:
        raise ImportError("asyncpg not available")
        
    if DB_POOL is not None:
        return await DB_POOL.acquire()
    else:
        # Try to initialize pool
        await initialize_db_pool()
        if DB_POOL is not None:
            return await DB_POOL.acquire()
        else:
            # Fallback to creating individual connections
            return await asyncpg.connect(dsn=os.getenv("DB_DSN"))

@contextlib.asynccontextmanager
async def db_transaction():
    """Context manager for database transactions."""
    conn = None
    try:
        conn = await get_db_connection_async()
        async with conn.transaction():
            yield conn
    finally:
        if conn:
            if DB_POOL is not None:
                await DB_POOL.release(conn)
            else:
                await conn.close()

async def with_transaction(callback, *args, **kwargs):
    """
    Execute callback within a transaction context.
    
    Args:
        callback: Async function to execute within transaction
        *args, **kwargs: Arguments to pass to callback
        
    Returns:
        Result of callback
    """
    async with db_transaction() as conn:
        return await callback(conn, *args, **kwargs)

async def handle_database_operation(operation_name, operation_func, *args, **kwargs):
    """
    Handle database operations with better error classification and logging.
    
    Args:
        operation_name: Name of the operation for logging
        operation_func: Function to execute
        *args, **kwargs: Arguments for the operation function
        
    Returns:
        Result of the operation or error object
    """
    try:
        # Execute the operation
        return await operation_func(*args, **kwargs)
    except asyncpg.PostgresError as e:
        # Database-specific error handling
        error_code = getattr(e, 'sqlstate', None)
        # Handle specific database errors
        if error_code == '23505':  # Unique violation
            logging.warning(f"Unique constraint violation in {operation_name}: {e}")
            return {"error": "duplicate_entry", "message": "This entry already exists"}
        elif error_code == '23503':  # Foreign key violation
            logging.warning(f"Foreign key violation in {operation_name}: {e}")
            return {"error": "reference_error", "message": "Referenced record does not exist"}
        elif error_code == '42P01':  # Undefined table
            logging.error(f"Table not found in {operation_name}: {e}")
            return {"error": "schema_error", "message": "Database schema issue"}
        else:
            logging.error(f"Database error in {operation_name}: {e} ({error_code})")
            return {"error": "database_error", "message": "A database error occurred"}
    except asyncio.TimeoutError:
        logging.error(f"Database operation timeout in {operation_name}")
        return {"error": "timeout", "message": "The operation timed out"}
    except Exception as e:
        # General error handling
        logging.exception(f"Unexpected error in {operation_name}")
        return {"error": "internal_error", "message": "An internal error occurred"}

async def fetch_row_async(query, *args):
    """Fetch a single row asynchronously."""
    async with db_transaction() as conn:
        return await conn.fetchrow(query, *args)

async def fetch_all_async(query, *args):
    """Fetch all rows asynchronously."""
    async with db_transaction() as conn:
        return await conn.fetch(query, *args)

async def execute_async(query, *args):
    """Execute a query asynchronously."""
    async with db_transaction() as conn:
        return await conn.execute(query, *args)
