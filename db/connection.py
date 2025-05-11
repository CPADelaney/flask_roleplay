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
    process_pid = os.getpid()

    # If app context is provided, this is the preferred way to check/store the pool for workers.
    if app:
        if hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
            logger.info(f"Process {process_pid}: DB pool already initialized on app.db_pool.")
            DB_POOL = app.db_pool # Ensure global DB_POOL in this process points to the app's pool
            return True
    # If no app context, or app.db_pool not set, check global DB_POOL (for this process).
    # This covers standalone scripts or the initial asyncio.run() phase.
    elif DB_POOL is not None and not DB_POOL._closed:
        logger.info(f"Process {process_pid}: Global DB_POOL already initialized for this process.")
        # If app was provided but app.db_pool wasn't set, set it now if global exists
        if app and (not hasattr(app, 'db_pool') or app.db_pool is None):
            app.db_pool = DB_POOL
        return True

    # If no suitable pool exists, create a new one.
    try:
        dsn = get_db_dsn()
        min_s = int(os.getenv("DB_POOL_MIN_SIZE", DEFAULT_MIN_CONNECTIONS))
        max_s = int(os.getenv("DB_POOL_MAX_SIZE", DEFAULT_MAX_CONNECTIONS))

        logger.info(f"Process {process_pid}: Creating NEW asyncpg pool (min={min_s}, max={max_s})...")
        new_pool = await asyncpg.create_pool(
            dsn=dsn, min_size=min_s, max_size=max_s, statement_cache_size=0,
            # It's good practice to set the loop explicitly if possible, though create_pool usually does this.
            # loop=asyncio.get_running_loop() # Ensures it's tied to the current loop
        )
        async with new_pool.acquire() as conn:
            await conn.execute("SELECT 1") # Test connection

        if app: # Store primarily on app if provided (for workers)
            app.db_pool = new_pool
            logger.info(f"Process {process_pid}: New DB_POOL stored on app.db_pool.")
        
        DB_POOL = new_pool # Also set/overwrite global DB_POOL for the current process
        logger.info(f"Process {process_pid}: Global DB_POOL set/updated for this process.")
        
        logger.info(f"Process {process_pid}: Asyncpg pool initialization successful.")
        return True
    except Exception as e:
        logger.critical(f"Process {process_pid}: Failed to initialize asyncpg pool: {e}", exc_info=True)
        if app: app.db_pool = None
        DB_POOL = None
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
    # If 'app' is not passed, try to use 'current_app' if in a Quart request context
    effective_app = app or (current_app if current_app else None)
    pid = os.getpid()
    pool_to_use: Optional[asyncpg.Pool] = None

    if effective_app and hasattr(effective_app, 'db_pool') and effective_app.db_pool:
        pool_to_use = effective_app.db_pool
        logger.debug(f"Process {pid}: Using DB pool from app.db_pool for get_db_connection_context.")
    elif DB_POOL: # Fallback to current process's global DB_POOL
        pool_to_use = DB_POOL
        logger.debug(f"Process {pid}: Using global DB_POOL for get_db_connection_context (app context not available/pool not set on app).")
    
    if pool_to_use is None or pool_to_use._closed:
        logger.warning(f"Process {pid}: DB pool is None or closed. Attempting lazy init in get_db_connection_context.")
        # Pass effective_app to ensure if a pool is created, it's set on the app context if possible
        if not await initialize_connection_pool(app=effective_app):
            raise ConnectionError("DB pool unavailable and lazy init failed.")
        
        # Re-fetch after lazy init
        pool_to_use = getattr(effective_app, 'db_pool', None) if effective_app else DB_POOL
        if pool_to_use is None:
            raise ConnectionError("DB pool is still None after lazy init attempt.")
        logger.info(f"Process {pid}: DB pool successfully lazy-initialized/retrieved: {id(pool_to_use)}.")

    conn: Optional[asyncpg.Connection] = None
    try:
        conn = await asyncio.wait_for(pool_to_use.acquire(), timeout=timeout)
        yield conn
    # ... (rest of your finally block for release) ...
    finally:
        if conn:
            try:
                await pool_to_use.release(conn)
            except asyncpg.exceptions.InterfaceError as ie:
                logger.error(f"InterfaceError releasing conn {id(conn)} from pool {id(pool_to_use)}: {ie}. Forcing close.", exc_info=False)
                try:
                    if not conn.is_closed(): await conn.close(timeout=5)
                except Exception as close_err: logger.error(f"Error forcing close on conn {id(conn)}: {close_err}")
            except Exception as release_err:
                logger.error(f"Error releasing conn {id(conn)} from pool {id(pool_to_use)}: {release_err}", exc_info=True)

async def close_connection_pool(app: Optional[Quart] = None): # <<< ADDED app: Optional[Quart] = None
    """Closes the DB_POOL, preferentially using app.db_pool if app is provided."""
    global DB_POOL
    pool_to_close: Optional[asyncpg.Pool] = None
    pid = os.getpid()

    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        pool_to_close = app.db_pool
        logger.info(f"Process {pid}: Closing DB pool from app.db_pool {id(pool_to_close)}.")
        app.db_pool = None # Clear app reference
    elif DB_POOL and not DB_POOL._closed:
        pool_to_close = DB_POOL
        logger.info(f"Process {pid}: Closing global DB_POOL {id(pool_to_close)} for this process.")
    
    if pool_to_close:
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
