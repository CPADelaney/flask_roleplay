# db/connection.py

import os
import logging
import asyncio
import asyncpg # Use asyncpg
from contextlib import asynccontextmanager
from quart import Quart 
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Global Pool variable
DB_POOL: Optional[asyncpg.Pool] = None

# Default max connections - can be overridden by environment
DEFAULT_MIN_CONNECTIONS = 5
DEFAULT_MAX_CONNECTIONS = 20

def get_db_dsn() -> str:
    """
    Returns the database connection string (DSN) from environment variables.
    Checks for DB_DSN first, then falls back to DATABASE_URL if DB_DSN is not set.

    Returns:
        str: PostgreSQL connection string (DSN).

    Raises:
        EnvironmentError: If neither DB_DSN nor DATABASE_URL environment variable is set.
    """
    # First check for DB_DSN
    dsn = os.getenv("DB_DSN")
    
    # If DB_DSN is not set, check for DATABASE_URL
    if not dsn:
        dsn = os.getenv("DATABASE_URL")
        if dsn:
            logger.info("Using DATABASE_URL as connection string (DB_DSN not found).")
    
    # If still not set, raise an error
    if not dsn:
        logger.critical("Neither DB_DSN nor DATABASE_URL environment variables are set.")
        raise EnvironmentError("Neither DB_DSN nor DATABASE_URL environment variables are set.")
    
    # asyncpg generally works well with standard postgresql:// DSNs
    return dsn

async def initialize_connection_pool(app: Optional[Quart] = None) -> bool:
    """
    Initializes the global DB_POOL for the current process/event loop.
    If 'app' is provided, the pool is also stored on app.db_pool.
    """
    global DB_POOL
    
    # Check if the pool is already initialized and valid
    if DB_POOL is not None and not DB_POOL._closed:
        logger.info(f"DB pool already initialized in process {os.getpid()}.")
        if app and not hasattr(app, 'db_pool'): # Ensure app.db_pool is also set if app is passed now
            app.db_pool = DB_POOL
        return True

    # Check if app already has a pool
    if app and hasattr(app, 'db_pool') and app.db_pool is not None and not app.db_pool._closed:
        logger.info(f"Using existing DB pool from app in process {os.getpid()}.")
        DB_POOL = app.db_pool
        return True

    try:
        dsn = get_db_dsn()
        min_s = int(os.getenv("DB_POOL_MIN_SIZE", DEFAULT_MIN_CONNECTIONS))
        max_s = int(os.getenv("DB_POOL_MAX_SIZE", DEFAULT_MAX_CONNECTIONS))

        logger.info(f"Process {os.getpid()}: Initializing asyncpg pool (min={min_s}, max={max_s})...")
        local_pool = await asyncpg.create_pool(
            dsn=dsn, min_size=min_s, max_size=max_s, statement_cache_size=0
        )
        async with local_pool.acquire() as conn:
            await conn.execute("SELECT 1")

        DB_POOL = local_pool
        if app: # If app instance is provided, store the pool on it
            app.db_pool = DB_POOL
            logger.info(f"Process {os.getpid()}: DB_POOL stored on app.db_pool.")

        logger.info(f"Process {os.getpid()}: Asyncpg pool initialized successfully.")
        return True
    except Exception as e:
        logger.critical(f"Process {os.getpid()}: Failed to initialize asyncpg pool: {e}", exc_info=True)
        DB_POOL = None
        if app: app.db_pool = None # Clear on app too if set attempt failed
        return False

async def get_db_connection_pool():
    global DB_POOL
    if DB_POOL is None or DB_POOL._closed:
        ok = await initialize_connection_pool()
        if not ok or DB_POOL is None:
            raise ConnectionError("Could not initialize DB pool.")
    return DB_POOL


@asynccontextmanager
async def get_db_connection_context(timeout: Optional[float] = 30.0, app: Optional[Quart] = None): # <<< ADDED app: Optional[Quart] = None
    global DB_POOL
    current_pool_to_use: Optional[asyncpg.Pool] = None

    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        current_pool_to_use = app.db_pool
    elif DB_POOL and not DB_POOL._closed:
        current_pool_to_use = DB_POOL
    
    if current_pool_to_use is None:
        logger.warning(f"Process {os.getpid()}: DB pool not initialized. Attempting lazy init for get_db_connection_context.")
        if not await initialize_connection_pool(app=app): # Pass app to lazy init too
            raise ConnectionError("DB pool unavailable and lazy init failed.")
        # Re-fetch pool after lazy init
        current_pool_to_use = app.db_pool if app and hasattr(app, 'db_pool') else DB_POOL
        if current_pool_to_use is None: # Still none
            raise ConnectionError("DB pool is None even after lazy init attempt.")

    conn: Optional[asyncpg.Connection] = None
    # ... (rest of get_db_connection_context as in previous good examples, using current_pool_to_use)
    try:
        conn = await asyncio.wait_for(current_pool_to_use.acquire(), timeout=timeout)
        yield conn
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout}s) acquiring DB connection from pool.")
        raise
    finally:
        if conn:
            try:
                await current_pool_to_use.release(conn)
            except asyncpg.exceptions.InterfaceError as ie:
                logger.error(f"InterfaceError releasing conn {id(conn)}: {ie}. Forcing close.", exc_info=False)
                try:
                    if not conn.is_closed(): await conn.close(timeout=5)
                except Exception as close_err: logger.error(f"Error forcing close on {id(conn)}: {close_err}")
            except Exception as release_err:
                logger.error(f"Error releasing conn {id(conn)}: {release_err}", exc_info=True)

async def close_connection_pool(app: Optional[Quart] = None): # <<< ADDED app: Optional[Quart] = None
    """Closes the DB_POOL, preferentially using app.db_pool if app is provided."""
    global DB_POOL
    pool_to_close: Optional[asyncpg.Pool] = None

    # Prefer pool from app context if available and valid
    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        pool_to_close = app.db_pool
        logger.info(f"Process {os.getpid()}: Will close DB pool from app.db_pool.")
    elif DB_POOL and not DB_POOL._closed: # Fallback to global
        pool_to_close = DB_POOL
        logger.info(f"Process {os.getpid()}: Will close global DB_POOL.")
    
    if pool_to_close:
        # Prevent new acquisitions by clearing references first (if they point to the same object)
        if DB_POOL is pool_to_close:
            DB_POOL = None
        if app and hasattr(app, 'db_pool') and app.db_pool is pool_to_close:
            app.db_pool = None
        
        logger.info(f"Process {os.getpid()}: Closing asyncpg pool ({id(pool_to_close)})...")
        try:
            await pool_to_close.close()
            logger.info(f"Process {os.getpid()}: Asyncpg pool ({id(pool_to_close)}) closed.")
        except Exception as e:
            logger.error(f"Process {os.getpid()}: Error closing asyncpg pool ({id(pool_to_close)}): {e}", exc_info=True)
    else:
        logger.info(f"Process {os.getpid()}: No active asyncpg pool found to close.")


# --- Remove Synchronous/psycopg2 specific functions ---
# get_db_connection (Legacy sync) - REMOVED
# return_db_connection (Legacy sync) - REMOVED
# execute_with_retry (Uses sync context) - REMOVED (Retry logic needs to be implemented around the async context manager usage if needed)
