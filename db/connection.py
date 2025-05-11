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

DB_POOL: Optional[asyncpg.Pool] = None # This will be process-local

async def initialize_connection_pool(app: Optional[Quart] = None) -> bool:
    global DB_POOL
    # Check if a pool is already set for THIS process (via global or app context)
    # The 'app' check is more for if this is called multiple times within the same worker's before_serving
    process_pid = os.getpid()
    existing_pool_on_app = getattr(app, 'db_pool', None) if app else None

    if existing_pool_on_app and not existing_pool_on_app._closed:
        logger.info(f"DB pool already initialized on app context for process {process_pid}.")
        if DB_POOL is not existing_pool_on_app: # Sync global with app's if different
            DB_POOL = existing_pool_on_app
        return True
    if DB_POOL is not None and not DB_POOL._closed:
        logger.info(f"Global DB pool already initialized in process {process_pid}.")
        if app and existing_pool_on_app is None: # Set on app if not already
             app.db_pool = DB_POOL
        return True

    try:
        dsn = get_db_dsn()
        min_s = int(os.getenv("DB_POOL_MIN_SIZE", DEFAULT_MIN_CONNECTIONS))
        max_s = int(os.getenv("DB_POOL_MAX_SIZE", DEFAULT_MAX_CONNECTIONS))

        logger.info(f"Process {process_pid}: Initializing new asyncpg pool (min={min_s}, max={max_s})...")
        # Create a new pool for this process/loop context
        new_pool = await asyncpg.create_pool(
            dsn=dsn, min_size=min_s, max_size=max_s, statement_cache_size=0
        )
        async with new_pool.acquire() as conn: # Test the new pool
            await conn.execute("SELECT 1")

        DB_POOL = new_pool # Set the global for this process
        if app:
            app.db_pool = DB_POOL # Also store on app context if provided
            logger.info(f"Process {process_pid}: DB_POOL stored on app.db_pool.")

        logger.info(f"Process {process_pid}: Asyncpg pool initialized successfully.")
        return True
    except Exception as e:
        logger.critical(f"Process {process_pid}: Failed to initialize asyncpg pool: {e}", exc_info=True)
        DB_POOL = None
        if app: app.db_pool = None
        return False


async def get_db_connection_pool():
    global DB_POOL
    if DB_POOL is None or DB_POOL._closed:
        ok = await initialize_connection_pool()
        if not ok or DB_POOL is None:
            raise ConnectionError("Could not initialize DB pool.")
    return DB_POOL


@asynccontextmanager
async def get_db_connection_context(timeout: Optional[float] = 30.0, app: Optional[Quart] = None):
    global DB_POOL
    pool_to_use: Optional[asyncpg.Pool] = None

    if app and hasattr(app, 'db_pool') and app.db_pool:
        pool_to_use = app.db_pool
    elif DB_POOL: # Fallback to current process's global DB_POOL
        pool_to_use = DB_POOL
    
    if pool_to_use is None or pool_to_use._closed:
        logger.warning(f"Process {os.getpid()}: DB pool not available. Attempting lazy init.")
        # Pass app so it can be set on app.db_pool if successful
        if not await initialize_connection_pool(app=app):
            raise ConnectionError("DB pool unavailable and lazy init failed.")
        # Re-check after lazy init
        pool_to_use = getattr(app, 'db_pool', None) if app else DB_POOL
        if pool_to_use is None:
            raise ConnectionError("DB pool still None after lazy init.")

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
