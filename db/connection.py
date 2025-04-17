# db/connection.py

import os
import logging
import asyncio
import asyncpg # Use asyncpg
from contextlib import asynccontextmanager
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

async def initialize_connection_pool():
    """
    Asynchronously initialize the asyncpg database connection pool.
    Should be awaited during application startup (e.g., in initialize_systems).

    Returns:
        bool: True if successful or already initialized, False otherwise.
    """
    global DB_POOL

    # Skip if already initialized
    if DB_POOL is not None and not DB_POOL._closed:
        logger.info("Database connection pool already initialized.")
        return True

    try:
        dsn = get_db_dsn() # Get DSN, raises error if not set
        min_connections = int(os.getenv("DB_POOL_MIN_SIZE", DEFAULT_MIN_CONNECTIONS))
        max_connections = int(os.getenv("DB_POOL_MAX_SIZE", DEFAULT_MAX_CONNECTIONS))

        logger.info(f"Initializing asyncpg connection pool (min={min_connections}, max={max_connections})...")
        # Create the connection pool using asyncpg
        DB_POOL = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_connections,
            max_size=max_connections,
            statement_cache_size=0,
            # command_timeout=60, # Optional: Set a default command timeout
            # Add setup/init functions if needed:
            # init=async def init(conn): logger.info(f"Pool init connection {id(conn)}")
        )

        # Test connection (optional but recommended)
        async with DB_POOL.acquire() as conn:
            await conn.execute("SELECT 1")

        logger.info(f"Asyncpg connection pool initialized successfully.")
        return True

    except (asyncpg.PostgresError, OSError, EnvironmentError) as e:
        logger.critical(f"Failed to initialize asyncpg connection pool: {str(e)}", exc_info=True)
        DB_POOL = None # Ensure pool is None on failure
        return False
    except Exception as e: # Catch any other unexpected errors
        logger.critical(f"Unexpected error initializing asyncpg pool: {str(e)}", exc_info=True)
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
async def get_db_connection_context(timeout: Optional[float] = 30.0):
    """
    Provides an asyncpg connection from the pool using an async context manager.
    Handles acquiring and releasing the connection. Does NOT handle transactions.

    Args:
        timeout (Optional[float]): Timeout in seconds to acquire a connection. Defaults to 30.0.

    Yields:
        asyncpg.Connection: An active database connection.

    Raises:
        ConnectionError: If the pool is not initialized or fails to acquire connection.
        asyncio.TimeoutError: If acquiring a connection times out.

    Example:
        try:
            async with get_db_connection_context() as conn:
                # Single statement (implicitly transactional)
                await conn.execute("UPDATE users SET active = TRUE WHERE id = $1", user_id)

                # Multiple statements requiring atomicity
                async with conn.transaction():
                    await conn.execute("INSERT INTO logs (msg) VALUES ($1)", "Log entry 1")
                    await conn.execute("INSERT INTO data (val) VALUES ($1)", 100)

                result = await conn.fetchval("SELECT name FROM products WHERE id = $1", product_id)
        except asyncio.TimeoutError:
            logger.error("Could not get DB connection in time.")
        except asyncpg.PostgresError as db_err:
            logger.error(f"Database operation failed: {db_err}")
        except ConnectionError as pool_err:
            logger.error(f"Pool issue: {pool_err}")

    """
    if DB_POOL is None or DB_POOL._closed:
        # Optionally attempt lazy initialization (can be risky under high concurrency)
        logger.warning("Connection pool not initialized or closed. Attempting lazy init.")
        if not await initialize_connection_pool() or DB_POOL is None:
             raise ConnectionError("Database connection pool is not available.")
        # If lazy init succeeds, fall through to acquire

    conn: Optional[asyncpg.Connection] = None
    try:
        # Acquire connection with timeout
        logger.debug(f"Acquiring connection from pool (timeout={timeout}s)...")
        conn = await asyncio.wait_for(DB_POOL.acquire(), timeout=timeout)
        logger.debug(f"Acquired connection {id(conn)}.")
        yield conn
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout}s) acquiring database connection from pool.")
        raise # Re-raise TimeoutError
    finally:
        if conn:
            try:
                await DB_POOL.release(conn)
                logger.debug(f"Released connection {id(conn)} back to pool.")
            except Exception as release_err:
                # Log error, but don't prevent code flow if release fails (pool might handle)
                logger.error(f"Error releasing connection {id(conn)}: {release_err}", exc_info=True)

async def close_connection_pool():
    """
    Asynchronously close the database connection pool.
    Should be awaited during application shutdown (e.g., via atexit asyncio wrapper).
    """
    global DB_POOL
    pool_to_close = DB_POOL
    if pool_to_close and not pool_to_close._closed:
        DB_POOL = None # Prevent new acquisitions immediately
        try:
            logger.info("Closing asyncpg connection pool...")
            # close() waits for connections to be released
            await pool_to_close.close()
            logger.info("Asyncpg connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing asyncpg connection pool: {str(e)}", exc_info=True)
            # Pool might be partially closed or in error state

# --- Remove Synchronous/psycopg2 specific functions ---
# get_db_connection (Legacy sync) - REMOVED
# return_db_connection (Legacy sync) - REMOVED
# execute_with_retry (Uses sync context) - REMOVED (Retry logic needs to be implemented around the async context manager usage if needed)
