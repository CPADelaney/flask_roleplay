# db/connection.py

import os
import logging
import asyncio
import asyncpg
from contextlib import asynccontextmanager
from quart import Quart 
from typing import Optional
import pgvector.asyncpg as pgvector_asyncpg

# Configure logging
logger = logging.getLogger(__name__)

# Global Pool variable
DB_POOL: Optional[asyncpg.Pool] = None

# Default connection pool settings - increased for better concurrency
DEFAULT_MIN_CONNECTIONS = 10  # Increased from 5
DEFAULT_MAX_CONNECTIONS = 50  # Increased from 20

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
        if app and not hasattr(app, 'db_pool'):
            app.db_pool = DB_POOL
        return True

    # Check if app already has a pool
    if app and hasattr(app, 'db_pool') and app.db_pool is not None and not app.db_pool._closed:
        logger.info(f"Using existing DB pool from app in process {os.getpid()}.")
        DB_POOL = app.db_pool
        return True

    try:
        dsn = get_db_dsn()
        
        # Get pool settings from environment with increased defaults
        min_s = int(os.getenv("DB_POOL_MIN_SIZE", str(DEFAULT_MIN_CONNECTIONS)))
        max_s = int(os.getenv("DB_POOL_MAX_SIZE", str(DEFAULT_MAX_CONNECTIONS)))
        
        # Additional pool settings for better connection management
        max_inactive_lifetime = int(os.getenv("DB_CONNECTION_LIFETIME", "60"))
        command_timeout = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))
        max_queries = int(os.getenv("DB_MAX_QUERIES", "50000"))
        
        logger.info(f"Process {os.getpid()}: Initializing asyncpg pool (min={min_s}, max={max_s}, "
                   f"max_inactive_lifetime={max_inactive_lifetime}s, command_timeout={command_timeout}s)...")
        
        local_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_s,
            max_size=max_s,
            statement_cache_size=0,  # Disable statement caching for pgbouncer compatibility
            max_inactive_connection_lifetime=max_inactive_lifetime,
            command_timeout=command_timeout,
            max_queries=max_queries,
            setup=setup_connection  # Add setup function for each connection
        )
        
        # Test the pool
        async with local_pool.acquire() as conn:
            await conn.execute("SELECT 1")
            logger.info(f"Process {os.getpid()}: Pool test query successful.")

        DB_POOL = local_pool
        if app:
            app.db_pool = DB_POOL
            logger.info(f"Process {os.getpid()}: DB_POOL stored on app.db_pool.")

        logger.info(f"Process {os.getpid()}: Asyncpg pool initialized successfully.")
        return True
        
    except Exception as e:
        logger.critical(f"Process {os.getpid()}: Failed to initialize asyncpg pool: {e}", exc_info=True)
        DB_POOL = None
        if app:
            app.db_pool = None
        return False

async def setup_connection(conn):
    """
    Setup function called for each new connection in the pool.
    Registers extensions and sets any connection-specific settings.
    """
    try:
        # Register pgvector extension for this connection
        await pgvector_asyncpg.register_vector(conn)
        
        # Set any other connection-specific settings here
        # For example, setting search_path, timezone, etc.
        # await conn.execute("SET search_path TO public")
        
    except Exception as e:
        logger.error(f"Error setting up connection: {e}", exc_info=True)
        raise

async def get_db_connection_pool():
    """
    Get the current database connection pool.
    Attempts lazy initialization if pool is not available.
    """
    global DB_POOL
    if DB_POOL is None or DB_POOL._closed:
        ok = await initialize_connection_pool()
        if not ok or DB_POOL is None:
            raise ConnectionError("Could not initialize DB pool.")
    return DB_POOL

@asynccontextmanager
async def get_db_connection_context(timeout: Optional[float] = 30.0, app: Optional[Quart] = None):
    """
    Async context manager for database connections.
    Each call gets a fresh connection from the pool.
    
    Args:
        timeout: Timeout in seconds for acquiring a connection
        app: Optional Quart app instance
        
    Yields:
        asyncpg.Connection: Database connection
        
    Raises:
        asyncio.TimeoutError: If connection acquisition times out
        ConnectionError: If pool is unavailable
    """
    global DB_POOL
    current_pool_to_use: Optional[asyncpg.Pool] = None
    conn: Optional[asyncpg.Connection] = None

    # Determine which pool to use
    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        current_pool_to_use = app.db_pool
        logger.debug(f"Using app.db_pool for connection")
    elif DB_POOL and not DB_POOL._closed:
        current_pool_to_use = DB_POOL
        logger.debug(f"Using global DB_POOL for connection")
    
    # Lazy initialization if needed
    if current_pool_to_use is None:
        logger.warning(f"Process {os.getpid()}: DB pool not initialized. Attempting lazy init for get_db_connection_context.")
        if not await initialize_connection_pool(app=app):
            raise ConnectionError("DB pool unavailable and lazy init failed.")
        
        current_pool_to_use = app.db_pool if app and hasattr(app, 'db_pool') else DB_POOL
        if current_pool_to_use is None:
            raise ConnectionError("DB pool is None even after lazy init attempt.")

    try:
        # Acquire connection with timeout
        logger.debug(f"Acquiring connection from pool (timeout={timeout}s)")
        conn = await asyncio.wait_for(current_pool_to_use.acquire(), timeout=timeout)
        
        # Ensure pgvector is registered for this connection
        # This is redundant if setup_connection is used, but kept for safety
        try:
            await pgvector_asyncpg.register_vector(conn)
        except Exception as vector_err:
            logger.debug(f"pgvector registration notice (may be already registered): {vector_err}")
        
        yield conn
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout}s) acquiring DB connection from pool. "
                    f"Pool stats - size: {current_pool_to_use.get_size() if hasattr(current_pool_to_use, 'get_size') else 'unknown'}, "
                    f"free: {current_pool_to_use.get_idle_size() if hasattr(current_pool_to_use, 'get_idle_size') else 'unknown'}")
        raise
    except asyncpg.exceptions.PostgresError as pg_err:
        logger.error(f"PostgreSQL error during connection operation: {pg_err}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during database operation: {e}", exc_info=True)
        raise
    finally:
        if conn:
            try:
                # Check connection state before attempting release
                if hasattr(conn, '_con') and conn._con is not None and not conn.is_closed():
                    # Connection is valid, release it back to pool
                    await current_pool_to_use.release(conn)
                    logger.debug(f"Released connection {id(conn)} back to pool")
                else:
                    # Connection is closed or invalid
                    logger.warning(f"Connection {id(conn)} was already closed, not releasing to pool")
                    
            except asyncpg.exceptions.InterfaceError as ie:
                # Connection is in a bad state, don't try to release it
                logger.error(f"InterfaceError releasing conn {id(conn)}: {ie}. Connection will be discarded.")
                # The pool should handle removing this bad connection
                
            except asyncpg.exceptions._base.InterfaceWarning as iw:
                # Non-critical warning during release
                logger.warning(f"Interface warning releasing conn {id(conn)}: {iw}")
                
            except Exception as release_err:
                # Any other error during release
                logger.error(f"Unexpected error releasing conn {id(conn)}: {release_err}", exc_info=True)

async def close_connection_pool(app: Optional[Quart] = None):
    """
    Closes the DB_POOL, preferentially using app.db_pool if app is provided.
    
    Args:
        app: Optional Quart app instance
    """
    global DB_POOL
    pool_to_close: Optional[asyncpg.Pool] = None

    # Prefer pool from app context if available and valid
    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        pool_to_close = app.db_pool
        logger.info(f"Process {os.getpid()}: Will close DB pool from app.db_pool.")
    elif DB_POOL and not DB_POOL._closed:
        pool_to_close = DB_POOL
        logger.info(f"Process {os.getpid()}: Will close global DB_POOL.")
    
    if pool_to_close:
        # Clear references first to prevent new acquisitions
        if DB_POOL is pool_to_close:
            DB_POOL = None
        if app and hasattr(app, 'db_pool') and app.db_pool is pool_to_close:
            app.db_pool = None
        
        logger.info(f"Process {os.getpid()}: Closing asyncpg pool ({id(pool_to_close)})...")
        try:
            # Close the pool gracefully
            await pool_to_close.close()
            logger.info(f"Process {os.getpid()}: Asyncpg pool ({id(pool_to_close)}) closed.")
        except Exception as e:
            logger.error(f"Process {os.getpid()}: Error closing asyncpg pool ({id(pool_to_close)}): {e}", exc_info=True)
    else:
        logger.info(f"Process {os.getpid()}: No active asyncpg pool found to close.")

# Health check function for monitoring
async def check_pool_health() -> dict:
    """
    Check the health of the database connection pool.
    
    Returns:
        dict: Health status information
    """
    global DB_POOL
    
    health = {
        "status": "unknown",
        "pool_exists": DB_POOL is not None,
        "pool_closed": DB_POOL._closed if DB_POOL else None,
        "size": None,
        "free": None,
        "used": None,
        "test_query": False
    }
    
    if DB_POOL and not DB_POOL._closed:
        try:
            # Get pool stats if available
            if hasattr(DB_POOL, 'get_size'):
                health["size"] = DB_POOL.get_size()
            if hasattr(DB_POOL, 'get_idle_size'):
                health["free"] = DB_POOL.get_idle_size()
            if health["size"] is not None and health["free"] is not None:
                health["used"] = health["size"] - health["free"]
            
            # Try a test query
            async with get_db_connection_context(timeout=5.0) as conn:
                await conn.fetchval("SELECT 1")
                health["test_query"] = True
                health["status"] = "healthy"
                
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error(f"Pool health check failed: {e}")
    else:
        health["status"] = "not_initialized"
    
    return health

# Export all public functions
__all__ = [
    'get_db_dsn',
    'initialize_connection_pool',
    'get_db_connection_pool',
    'get_db_connection_context',
    'close_connection_pool',
    'check_pool_health',
    'setup_connection'
]
