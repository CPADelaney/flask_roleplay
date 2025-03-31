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

# Import database connections
from db.connection import get_db_connection_context

# Import config values from environment
from os import environ

# Database configuration with defaults
DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb")
DB_MIN_CONN = int(environ.get("DB_MIN_CONN", "5"))
DB_MAX_CONN = int(environ.get("DB_MAX_CONN", "20"))
DB_COMMAND_TIMEOUT = int(environ.get("DB_COMMAND_TIMEOUT", "60"))
DB_STATEMENT_TIMEOUT = int(environ.get("DB_STATEMENT_TIMEOUT", "60"))
DB_STATEMENT_LIFETIME = int(environ.get("DB_STATEMENT_LIFETIME", "300"))
DB_INACTIVE_CONN_LIFETIME = int(environ.get("DB_INACTIVE_CONN_LIFETIME", "300"))

@contextlib.asynccontextmanager
async def db_transaction():
    """Context manager for database transactions."""
    async with get_db_connection_context() as conn:
        async with conn.transaction():
            yield conn

@contextlib.asynccontextmanager
async def db_transaction_with_timeout(timeout_seconds=10):
    """Context manager for database transactions with explicit timeout."""
    timeout_task = None
    
    try:
        # Start timeout
        timeout_task = asyncio.create_task(asyncio.sleep(timeout_seconds))
        
        # Start both tasks
        async with asyncio.timeout(timeout_seconds):
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    yield conn
                    
    except asyncio.TimeoutError:
        logging.error(f"Database operation timed out after {timeout_seconds} seconds")
        raise
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise
    finally:
        # Clean up tasks
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()

async def with_transaction(callback, *args, **kwargs):
    """
    Execute callback within a transaction context.
    
    Args:
        callback: Async function to execute within transaction
        *args, **kwargs: Arguments to pass to callback
        
    Returns:
        Result of callback
    """
    timeout = kwargs.pop('timeout', None)
    
    if timeout:
        async with db_transaction_with_timeout(timeout) as conn:
            return await callback(conn, *args, **kwargs)
    else:
        async with db_transaction() as conn:
            return await callback(conn, *args, **kwargs)

async def handle_database_operation(operation_name, operation_func, *args, timeout=None, **kwargs):
    """
    Handle database operations with better error classification and logging.
    
    Args:
        operation_name: Name of the operation for logging
        operation_func: Function to execute
        *args, **kwargs: Arguments for the operation function
        timeout: Optional timeout in seconds
        
    Returns:
        Result of the operation or error object
    """
    # Import performance tracking
    from utils.performance import STATS
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Create a task with timeout if specified
        if timeout:
            task = asyncio.create_task(operation_func(*args, **kwargs))
            try:
                result = await asyncio.wait_for(task, timeout)
                # Record performance
                elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                STATS.record_db_query_time(elapsed_ms)
                return result
            except asyncio.TimeoutError:
                logging.error(f"Operation {operation_name} timed out after {timeout}s")
                return {"error": "timeout", "message": f"The operation timed out after {timeout} seconds"}
        else:
            # Execute normally if no timeout
            result = await operation_func(*args, **kwargs)
            # Record performance
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            STATS.record_db_query_time(elapsed_ms)
            return result
            
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
        return {"error": "internal_error", "message": f"An internal error occurred: {str(e)}"}

async def fetch_row_async(query, *args, timeout=None):
    """Fetch a single row asynchronously with optional timeout."""
    return await handle_database_operation(
        "fetch_row", 
        _fetch_row_impl,
        query, *args,
        timeout=timeout
    )

async def _fetch_row_impl(query, *args):
    """Implementation for fetch_row_async."""
    async with db_transaction() as conn:
        return await conn.fetchrow(query, *args)

async def fetch_all_async(query, *args, timeout=None):
    """Fetch all rows asynchronously with optional timeout."""
    return await handle_database_operation(
        "fetch_all", 
        _fetch_all_impl,
        query, *args,
        timeout=timeout
    )

async def _fetch_all_impl(query, *args):
    """Implementation for fetch_all_async."""
    async with db_transaction() as conn:
        return await conn.fetch(query, *args)

async def execute_async(query, *args, timeout=None):
    """Execute a query asynchronously with optional timeout."""
    return await handle_database_operation(
        "execute", 
        _execute_impl,
        query, *args,
        timeout=timeout
    )

async def _execute_impl(query, *args):
    """Implementation for execute_async."""
    async with db_transaction() as conn:
        return await conn.execute(query, *args)
