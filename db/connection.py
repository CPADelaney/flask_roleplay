# db/connection.py

"""
Database connection pool management for asyncpg with Celery worker support.

This module handles:
- Per-process asyncpg connection pools
- Event loop lifecycle management
- Celery worker initialization and cleanup
- Connection context managers for safe connection usage
"""

import os
import logging
import asyncio
import asyncpg
import threading
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from quart import Quart
import pgvector.asyncpg as pgvector_asyncpg

# Celery worker lifecycle signals
from celery.signals import worker_process_init, worker_process_shutdown

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Global State
# ============================================================================

# Connection pool (one per process)
DB_POOL: Optional[asyncpg.Pool] = None
DB_POOL_LOOP: Optional[asyncio.AbstractEventLoop] = None

# Thread-local storage for event loops
_thread_local = threading.local()

# Lock to prevent concurrent pool initialization
_pool_init_lock = threading.Lock()

# Default pool configuration
DEFAULT_MIN_CONNECTIONS = 10
DEFAULT_MAX_CONNECTIONS = 50
DEFAULT_CONNECTION_LIFETIME = 60
DEFAULT_COMMAND_TIMEOUT = 60
DEFAULT_MAX_QUERIES = 50000

# ============================================================================
# Environment Configuration
# ============================================================================

def get_db_dsn() -> str:
    """
    Returns the database connection string (DSN) from environment variables.
    
    Tries DB_DSN first, then DATABASE_URL as fallback.
    
    Returns:
        str: Database connection DSN
        
    Raises:
        EnvironmentError: If neither environment variable is set
    """
    dsn = os.getenv("DB_DSN")
    if not dsn:
        dsn = os.getenv("DATABASE_URL")
        if dsn:
            logger.info("Using DATABASE_URL as connection string (DB_DSN not found).")
    
    if not dsn:
        logger.critical("Neither DB_DSN nor DATABASE_URL environment variables are set.")
        raise EnvironmentError("Neither DB_DSN nor DATABASE_URL environment variables are set.")
    
    return dsn


def get_pool_config() -> Dict[str, int]:
    """
    Get connection pool configuration from environment variables.
    
    Returns:
        dict: Pool configuration parameters
    """
    return {
        'min_size': int(os.getenv("DB_POOL_MIN_SIZE", str(DEFAULT_MIN_CONNECTIONS))),
        'max_size': int(os.getenv("DB_POOL_MAX_SIZE", str(DEFAULT_MAX_CONNECTIONS))),
        'max_inactive_connection_lifetime': int(os.getenv("DB_CONNECTION_LIFETIME", str(DEFAULT_CONNECTION_LIFETIME))),
        'command_timeout': int(os.getenv("DB_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT))),
        'max_queries': int(os.getenv("DB_MAX_QUERIES", str(DEFAULT_MAX_QUERIES))),
    }


# ============================================================================
# Event Loop Management
# ============================================================================

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop or create one if needed.
    
    For Celery workers, this ensures a consistent event loop is used
    throughout the worker's lifetime.
    
    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop
    """
    try:
        # Try to get the currently running loop
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No running loop - check thread-local storage first
        if hasattr(_thread_local, 'event_loop') and \
           _thread_local.event_loop is not None and \
           not _thread_local.event_loop.is_closed():
            # Reuse thread-local loop
            asyncio.set_event_loop(_thread_local.event_loop)
            return _thread_local.event_loop
        else:
            # Create and store new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.event_loop = loop
            logger.debug(f"Created new event loop {id(loop)} for thread {threading.current_thread().name}")
            return loop


def run_async_in_worker_loop(coro):
    """
    Run an async coroutine in the worker's persistent event loop.
    
    This is the bridge between Celery's synchronous task execution
    and our async database operations.
    
    Args:
        coro: Coroutine to execute
        
    Returns:
        Any: Result of the coroutine
        
    Raises:
        RuntimeError: If event loop is already running (shouldn't happen in prefork)
    """
    loop = get_or_create_event_loop()
    
    # Ensure the loop is set as current
    asyncio.set_event_loop(loop)
    
    # If loop is already running, we have a problem
    if loop.is_running():
        raise RuntimeError("Event loop is already running - unexpected in Celery prefork worker")
    
    try:
        # Run the coroutine and return result
        return loop.run_until_complete(coro)
    except Exception:
        # Don't close the loop on error - we want to reuse it
        raise


# ============================================================================
# Connection Pool Management
# ============================================================================

async def setup_connection(conn: asyncpg.Connection):
    """
    Setup function called for each new connection in the pool.
    
    Registers pgvector extension and any other per-connection setup.
    
    Args:
        conn: New connection to set up
    """
    try:
        await pgvector_asyncpg.register_vector(conn)
    except Exception as e:
        logger.error(f"Error setting up connection: {e}", exc_info=True)
        raise


async def close_existing_pool():
    """
    Close the existing connection pool if one exists.
    
    This is used when reinitializing the pool or during shutdown.
    """
    global DB_POOL, DB_POOL_LOOP
    
    if DB_POOL and not DB_POOL._closed:
        try:
            logger.info(f"Closing existing pool in process {os.getpid()}")
            await DB_POOL.close()
        except Exception as e:
            logger.error(f"Error closing existing pool: {e}")
        finally:
            DB_POOL = None
            DB_POOL_LOOP = None


async def initialize_connection_pool(
    app: Optional[Quart] = None, 
    force_new: bool = False
) -> bool:
    """
    Initialize the global DB_POOL for the current process/event loop.
    
    This function is idempotent - calling it multiple times with the same
    event loop will reuse the existing pool.
    
    Args:
        app: Optional Quart app instance to store pool on
        force_new: Force creation of new pool even if one exists
        
    Returns:
        bool: True if pool was successfully initialized or already exists
    """
    global DB_POOL, DB_POOL_LOOP
    
    # Use lock to prevent race conditions during initialization
    with _pool_init_lock:
        current_loop = get_or_create_event_loop()
        
        # Check if we already have a valid pool for this loop
        if not force_new and DB_POOL is not None and not DB_POOL._closed and DB_POOL_LOOP == current_loop:
            try:
                # Test the pool
                async with DB_POOL.acquire() as conn:
                    await conn.execute("SELECT 1")
                logger.info(
                    f"DB pool already initialized for loop {id(current_loop)} "
                    f"in process {os.getpid()}"
                )
                if app and not hasattr(app, 'db_pool'):
                    app.db_pool = DB_POOL
                return True
            except Exception as e:
                logger.warning(f"Existing pool failed health check: {e}")
                await close_existing_pool()
        
        # If pool exists for different loop, close it
        if DB_POOL is not None and DB_POOL_LOOP != current_loop:
            logger.warning(
                f"Pool exists for different event loop "
                f"(pool_loop={id(DB_POOL_LOOP)}, current_loop={id(current_loop)}), "
                f"closing it"
            )
            await close_existing_pool()
        
        # Force close if requested
        if force_new:
            await close_existing_pool()

        # Check if app already has a usable pool
        if app and hasattr(app, 'db_pool') and app.db_pool is not None and not app.db_pool._closed:
            try:
                async with app.db_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                logger.info(f"Using existing DB pool from app in process {os.getpid()}")
                DB_POOL = app.db_pool
                DB_POOL_LOOP = current_loop
                return True
            except Exception:
                logger.warning("App pool exists but is not working, will create new pool")
                app.db_pool = None

        # Create new pool
        try:
            dsn = get_db_dsn()
            config = get_pool_config()
            
            logger.info(
                f"Process {os.getpid()}: Creating new asyncpg pool for event loop "
                f"{id(current_loop)} (min={config['min_size']}, max={config['max_size']})"
            )
            
            local_pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=config['min_size'],
                max_size=config['max_size'],
                statement_cache_size=0,  # Disable for pgbouncer compatibility
                max_inactive_connection_lifetime=config['max_inactive_connection_lifetime'],
                command_timeout=config['command_timeout'],
                max_queries=config['max_queries'],
                setup=setup_connection,
                loop=current_loop  # Explicitly bind to current loop
            )
            
            # Test the pool
            async with local_pool.acquire() as conn:
                await conn.execute("SELECT 1")
                logger.info(f"Process {os.getpid()}: Pool test query successful")

            # Store the pool
            DB_POOL = local_pool
            DB_POOL_LOOP = current_loop
            
            if app:
                app.db_pool = DB_POOL
                logger.info(f"Process {os.getpid()}: DB_POOL stored on app.db_pool")

            logger.info(
                f"Process {os.getpid()}: Asyncpg pool initialized successfully "
                f"for loop {id(current_loop)}"
            )
            return True
            
        except Exception as e:
            logger.critical(
                f"Process {os.getpid()}: Failed to initialize asyncpg pool: {e}", 
                exc_info=True
            )
            DB_POOL = None
            DB_POOL_LOOP = None
            if app:
                app.db_pool = None
            return False


async def get_db_connection_pool() -> asyncpg.Pool:
    """
    Get the current database connection pool.
    
    Attempts lazy initialization if pool is not available, but logs a warning
    in Celery contexts since the pool should already be initialized.
    
    Returns:
        asyncpg.Pool: The connection pool
        
    Raises:
        ConnectionError: If pool cannot be initialized
    """
    global DB_POOL, DB_POOL_LOOP
    current_loop = get_or_create_event_loop()
    
    # Check if pool is ready for current loop
    if DB_POOL is None or DB_POOL._closed or DB_POOL_LOOP != current_loop:
        is_celery = 'celery' in os.environ.get('SERVER_SOFTWARE', '').lower()
        
        if is_celery:
            logger.error(
                f"Celery worker pool not initialized properly! "
                f"pool_exists={DB_POOL is not None}, "
                f"pool_closed={DB_POOL._closed if DB_POOL else 'N/A'}, "
                f"loop_match={DB_POOL_LOOP == current_loop if DB_POOL_LOOP else 'N/A'}"
            )
        else:
            logger.warning(
                f"Pool not ready, attempting lazy initialization "
                f"(pool_exists={DB_POOL is not None})"
            )
        
        ok = await initialize_connection_pool(force_new=False)
        if not ok or DB_POOL is None:
            raise ConnectionError("Could not initialize DB pool")
            
    return DB_POOL


@asynccontextmanager
async def get_db_connection_context(
    timeout: Optional[float] = 30.0, 
    app: Optional[Quart] = None
):
    """
    Async context manager for safe database connection usage.
    
    Example:
        async with get_db_connection_context() as conn:
            result = await conn.fetchval("SELECT 1")
    
    Args:
        timeout: Timeout in seconds for acquiring connection (default: 30s)
        app: Optional Quart app instance
        
    Yields:
        asyncpg.Connection: Database connection
        
    Raises:
        asyncio.TimeoutError: If connection cannot be acquired within timeout
        ConnectionError: If pool is not available
    """
    global DB_POOL, DB_POOL_LOOP
    current_loop = get_or_create_event_loop()
    conn: Optional[asyncpg.Connection] = None

    # Check if pool needs reinitialization due to loop change
    if DB_POOL and DB_POOL_LOOP != current_loop:
        logger.warning("DB pool was created for different event loop, reinitializing")
        await close_existing_pool()

    # Determine which pool to use
    current_pool_to_use: Optional[asyncpg.Pool] = None
    
    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        current_pool_to_use = app.db_pool
    elif DB_POOL and not DB_POOL._closed and DB_POOL_LOOP == current_loop:
        current_pool_to_use = DB_POOL
    
    # Lazy initialization if needed
    if current_pool_to_use is None:
        logger.warning(f"Process {os.getpid()}: DB pool not initialized. Attempting lazy init.")
        is_celery = 'celery' in os.environ.get('SERVER_SOFTWARE', '').lower()
        
        if not await initialize_connection_pool(app=app, force_new=is_celery):
            raise ConnectionError("DB pool unavailable and lazy init failed")
        
        current_pool_to_use = app.db_pool if app and hasattr(app, 'db_pool') else DB_POOL
        if current_pool_to_use is None:
            raise ConnectionError("DB pool is None even after lazy init attempt")

    try:
        # Acquire connection with timeout
        logger.debug(f"Acquiring connection from pool (timeout={timeout}s)")
        conn = await asyncio.wait_for(current_pool_to_use.acquire(), timeout=timeout)
        
        yield conn
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout}s) acquiring DB connection from pool")
        raise
    except asyncpg.exceptions.PostgresError as pg_err:
        logger.error(f"PostgreSQL error: {pg_err}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in connection context: {e}", exc_info=True)
        raise
    finally:
        if conn:
            try:
                if hasattr(conn, '_con') and conn._con is not None and not conn.is_closed():
                    await current_pool_to_use.release(conn)
                    logger.debug("Released connection back to pool")
                else:
                    logger.warning("Connection was already closed")
            except Exception as release_err:
                logger.error(f"Error releasing connection: {release_err}", exc_info=True)


async def close_connection_pool(app: Optional[Quart] = None):
    """
    Close the connection pool.
    
    Args:
        app: Optional Quart app instance
    """
    global DB_POOL, DB_POOL_LOOP
    pool_to_close: Optional[asyncpg.Pool] = None

    # Determine which pool to close
    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        pool_to_close = app.db_pool
        logger.info(f"Process {os.getpid()}: Will close DB pool from app.db_pool")
    elif DB_POOL and not DB_POOL._closed:
        pool_to_close = DB_POOL
        logger.info(f"Process {os.getpid()}: Will close global DB_POOL")
    
    if pool_to_close:
        # Clear references
        if DB_POOL is pool_to_close:
            DB_POOL = None
            DB_POOL_LOOP = None
        if app and hasattr(app, 'db_pool') and app.db_pool is pool_to_close:
            app.db_pool = None
        
        # Close the pool
        try:
            await pool_to_close.close()
            logger.info(f"Process {os.getpid()}: Asyncpg pool closed")
        except Exception as e:
            logger.error(f"Process {os.getpid()}: Error closing pool: {e}", exc_info=True)


async def check_pool_health() -> Dict[str, Any]:
    """
    Check the health of the database connection pool.
    
    Returns:
        dict: Health status including pool size, free connections, and test query result
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
            # Get pool statistics
            if hasattr(DB_POOL, 'get_size'):
                health["size"] = DB_POOL.get_size()
            if hasattr(DB_POOL, 'get_idle_size'):
                health["free"] = DB_POOL.get_idle_size()
            if health["size"] is not None and health["free"] is not None:
                health["used"] = health["size"] - health["free"]
            
            # Test query
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


# ============================================================================
# Celery Worker Integration
# ============================================================================

def init_celery_worker():
    """
    Initialize database pool for Celery worker.
    
    This is called when a worker process starts via the worker_process_init signal.
    """
    pid = os.getpid()
    logger.info(f"Initializing Celery worker database pool in process {pid}")
    
    # Set marker for Celery context
    os.environ['SERVER_SOFTWARE'] = 'celery'
    
    try:
        # Use run_async_in_worker_loop to ensure we use the same event loop
        # for initialization and subsequent task execution
        run_async_in_worker_loop(initialize_connection_pool(force_new=True))
        
        # Log the event loop for debugging
        loop = get_or_create_event_loop()
        logger.info(
            f"Celery worker {pid} database pool initialized successfully "
            f"with event loop {id(loop)}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Celery worker database pool: {e}", exc_info=True)
        raise


@worker_process_init.connect
def init_worker_pool(**kwargs):
    """
    Signal handler: Initialize database pool when worker process starts.
    """
    init_celery_worker()


@worker_process_shutdown.connect
def close_worker_pool(**kwargs):
    """
    Signal handler: Close database pool when worker process shuts down.
    """
    global DB_POOL, DB_POOL_LOOP
    pid = os.getpid()
    
    logger.info(f"Shutting down worker process {pid}")
    
    if DB_POOL:
        logger.info(f"Process {pid}: Will close global DB_POOL")
        try:
            # Use the same event loop that owns the pool
            if DB_POOL_LOOP and not DB_POOL_LOOP.is_closed():
                if DB_POOL_LOOP.is_running():
                    # Shouldn't happen in prefork, but handle it
                    asyncio.run_coroutine_threadsafe(close_connection_pool(), DB_POOL_LOOP)
                else:
                    # Common case: loop exists but isn't running
                    DB_POOL_LOOP.run_until_complete(close_connection_pool())
            else:
                # Fallback: try thread-local loop
                if hasattr(_thread_local, 'event_loop') and not _thread_local.event_loop.is_closed():
                    loop = _thread_local.event_loop
                    if not loop.is_running():
                        loop.run_until_complete(close_connection_pool())
                    else:
                        logger.warning(f"Cannot close pool - event loop is running")
                else:
                    logger.warning(f"Process {pid}: No suitable event loop found for pool cleanup")
                    DB_POOL = None
                    DB_POOL_LOOP = None
                    
            logger.info(f"Worker process {pid} database pool closed successfully")
        except Exception as e:
            logger.error(f"Process {pid}: Error during pool cleanup: {e}", exc_info=True)
            DB_POOL = None
            DB_POOL_LOOP = None
    
    # Clean up event loop
    if hasattr(_thread_local, 'event_loop') and _thread_local.event_loop:
        try:
            if not _thread_local.event_loop.is_closed():
                if not _thread_local.event_loop.is_running():
                    _thread_local.event_loop.close()
                    logger.info(f"Worker {pid} event loop closed")
        except Exception as e:
            logger.error(f"Error closing event loop: {e}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'get_db_dsn',
    'initialize_connection_pool',
    'get_db_connection_pool',
    'get_db_connection_context',
    'close_connection_pool',
    'check_pool_health',
    'setup_connection',
    'init_celery_worker',
    'run_async_in_worker_loop',
    'get_or_create_event_loop'
]
