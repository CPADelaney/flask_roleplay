# db/connection.py

import os
import logging
import asyncio
import asyncpg
from contextlib import asynccontextmanager
from quart import Quart 
from typing import Optional
import pgvector.asyncpg as pgvector_asyncpg
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Global Pool variable - will be initialized per worker
DB_POOL: Optional[asyncpg.Pool] = None

# Store the event loop that owns the pool
DB_POOL_LOOP: Optional[asyncio.AbstractEventLoop] = None

# Thread-local storage for worker event loops
_thread_local = threading.local()

# Default connection pool settings
DEFAULT_MIN_CONNECTIONS = 10
DEFAULT_MAX_CONNECTIONS = 50

def get_db_dsn() -> str:
    """
    Returns the database connection string (DSN) from environment variables.
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

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the current event loop or create one if needed."""
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No running loop
        if hasattr(_thread_local, 'event_loop') and not _thread_local.event_loop.is_closed():
            # Use thread-local loop
            asyncio.set_event_loop(_thread_local.event_loop)
            return _thread_local.event_loop
        else:
            # Create new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.event_loop = loop
            return loop

async def close_existing_pool():
    """Close existing pool if any."""
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

async def initialize_connection_pool(app: Optional[Quart] = None, force_new: bool = False) -> bool:
    """
    Initializes the global DB_POOL for the current process/event loop.
    
    Args:
        app: Optional Quart app instance
        force_new: Force creation of a new pool even if one exists
    """
    global DB_POOL, DB_POOL_LOOP
    
    # Get current event loop
    current_loop = get_or_create_event_loop()
    
    # Check if we have a pool for the current loop
    if not force_new and DB_POOL is not None and not DB_POOL._closed and DB_POOL_LOOP == current_loop:
        try:
            # Test the pool
            async with DB_POOL.acquire() as conn:
                await conn.execute("SELECT 1")
            logger.info(f"DB pool already initialized for current loop in process {os.getpid()}.")
            if app and not hasattr(app, 'db_pool'):
                app.db_pool = DB_POOL
            return True
        except Exception:
            # Pool exists but is not working, close it
            await close_existing_pool()
    
    # If pool is from a different loop, close it
    if DB_POOL is not None and DB_POOL_LOOP != current_loop:
        logger.warning(f"Pool exists for different event loop, closing it")
        await close_existing_pool()
    
    # For Celery workers, always create a new pool if forced
    if force_new:
        await close_existing_pool()

    # Check if app already has a pool for current loop
    if app and hasattr(app, 'db_pool') and app.db_pool is not None and not app.db_pool._closed:
        # Verify it's for the current loop
        try:
            async with app.db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            logger.info(f"Using existing DB pool from app in process {os.getpid()}.")
            DB_POOL = app.db_pool
            DB_POOL_LOOP = current_loop
            return True
        except Exception:
            # App pool is not working
            app.db_pool = None

    try:
        dsn = get_db_dsn()
        
        # Get pool settings from environment
        min_s = int(os.getenv("DB_POOL_MIN_SIZE", str(DEFAULT_MIN_CONNECTIONS)))
        max_s = int(os.getenv("DB_POOL_MAX_SIZE", str(DEFAULT_MAX_CONNECTIONS)))
        
        # Additional pool settings
        max_inactive_lifetime = int(os.getenv("DB_CONNECTION_LIFETIME", "60"))
        command_timeout = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))
        max_queries = int(os.getenv("DB_MAX_QUERIES", "50000"))
        
        logger.info(f"Process {os.getpid()}: Creating new asyncpg pool for event loop {id(current_loop)} (min={min_s}, max={max_s})...")
        
        local_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_s,
            max_size=max_s,
            statement_cache_size=0,  # Disable for pgbouncer compatibility
            max_inactive_connection_lifetime=max_inactive_lifetime,
            command_timeout=command_timeout,
            max_queries=max_queries,
            setup=setup_connection,
            loop=current_loop  # Explicitly bind to current loop
        )
        
        # Test the pool
        async with local_pool.acquire() as conn:
            await conn.execute("SELECT 1")
            logger.info(f"Process {os.getpid()}: Pool test query successful.")

        DB_POOL = local_pool
        DB_POOL_LOOP = current_loop
        
        if app:
            app.db_pool = DB_POOL
            logger.info(f"Process {os.getpid()}: DB_POOL stored on app.db_pool.")

        logger.info(f"Process {os.getpid()}: Asyncpg pool initialized successfully for loop {id(current_loop)}.")
        return True
        
    except Exception as e:
        logger.critical(f"Process {os.getpid()}: Failed to initialize asyncpg pool: {e}", exc_info=True)
        DB_POOL = None
        DB_POOL_LOOP = None
        if app:
            app.db_pool = None
        return False

async def setup_connection(conn):
    """
    Setup function called for each new connection in the pool.
    """
    try:
        await pgvector_asyncpg.register_vector(conn)
    except Exception as e:
        logger.error(f"Error setting up connection: {e}", exc_info=True)
        raise

async def get_db_connection_pool():
    """
    Get the current database connection pool.
    Attempts lazy initialization if pool is not available.
    """
    global DB_POOL, DB_POOL_LOOP
    current_loop = get_or_create_event_loop()
    
    # Check if pool exists and is for current loop
    if DB_POOL is None or DB_POOL._closed or DB_POOL_LOOP != current_loop:
        # In Celery context, force new pool
        is_celery = 'celery' in os.environ.get('SERVER_SOFTWARE', '').lower()
        ok = await initialize_connection_pool(force_new=is_celery)
        if not ok or DB_POOL is None:
            raise ConnectionError("Could not initialize DB pool.")
    return DB_POOL

@asynccontextmanager
async def get_db_connection_context(timeout: Optional[float] = 30.0, app: Optional[Quart] = None):
    """
    Async context manager for database connections.
    """
    global DB_POOL, DB_POOL_LOOP
    current_loop = get_or_create_event_loop()
    current_pool_to_use: Optional[asyncpg.Pool] = None
    conn: Optional[asyncpg.Connection] = None

    # Check if we need to reinitialize due to loop change
    if DB_POOL and DB_POOL_LOOP != current_loop:
        logger.warning(f"DB pool was created for different event loop, reinitializing")
        await close_existing_pool()

    # Determine which pool to use
    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        current_pool_to_use = app.db_pool
    elif DB_POOL and not DB_POOL._closed and DB_POOL_LOOP == current_loop:
        current_pool_to_use = DB_POOL
    
    # Lazy initialization if needed
    if current_pool_to_use is None:
        logger.warning(f"Process {os.getpid()}: DB pool not initialized. Attempting lazy init.")
        # In Celery context, force new pool
        is_celery = 'celery' in os.environ.get('SERVER_SOFTWARE', '').lower()
        if not await initialize_connection_pool(app=app, force_new=is_celery):
            raise ConnectionError("DB pool unavailable and lazy init failed.")
        
        current_pool_to_use = app.db_pool if app and hasattr(app, 'db_pool') else DB_POOL
        if current_pool_to_use is None:
            raise ConnectionError("DB pool is None even after lazy init attempt.")

    try:
        # Acquire connection with timeout
        logger.debug(f"Acquiring connection from pool (timeout={timeout}s)")
        conn = await asyncio.wait_for(current_pool_to_use.acquire(), timeout=timeout)
        
        yield conn
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout}s) acquiring DB connection from pool.")
        raise
    except asyncpg.exceptions.PostgresError as pg_err:
        logger.error(f"PostgreSQL error: {pg_err}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
    finally:
        if conn:
            try:
                if hasattr(conn, '_con') and conn._con is not None and not conn.is_closed():
                    await current_pool_to_use.release(conn)
                    logger.debug(f"Released connection back to pool")
                else:
                    logger.warning(f"Connection was already closed")
            except Exception as release_err:
                logger.error(f"Error releasing connection: {release_err}", exc_info=True)

async def close_connection_pool(app: Optional[Quart] = None):
    """
    Closes the DB_POOL.
    """
    global DB_POOL, DB_POOL_LOOP
    pool_to_close: Optional[asyncpg.Pool] = None

    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        pool_to_close = app.db_pool
        logger.info(f"Process {os.getpid()}: Will close DB pool from app.db_pool.")
    elif DB_POOL and not DB_POOL._closed:
        pool_to_close = DB_POOL
        logger.info(f"Process {os.getpid()}: Will close global DB_POOL.")
    
    if pool_to_close:
        if DB_POOL is pool_to_close:
            DB_POOL = None
            DB_POOL_LOOP = None
        if app and hasattr(app, 'db_pool') and app.db_pool is pool_to_close:
            app.db_pool = None
        
        try:
            await pool_to_close.close()
            logger.info(f"Process {os.getpid()}: Asyncpg pool closed.")
        except Exception as e:
            logger.error(f"Process {os.getpid()}: Error closing pool: {e}", exc_info=True)

async def check_pool_health() -> dict:
    """Check the health of the database connection pool."""
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
            if hasattr(DB_POOL, 'get_size'):
                health["size"] = DB_POOL.get_size()
            if hasattr(DB_POOL, 'get_idle_size'):
                health["free"] = DB_POOL.get_idle_size()
            if health["size"] is not None and health["free"] is not None:
                health["used"] = health["size"] - health["free"]
            
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

# Celery worker event loop management
def run_async_in_worker_loop(coro):
    """
    Run an async coroutine in the worker's event loop.
    Creates and manages a persistent event loop per worker thread.
    """
    loop = get_or_create_event_loop()
    
    # If loop is already running, we can't use run_until_complete
    if loop.is_running():
        # This shouldn't happen in normal Celery operation
        raise RuntimeError("Event loop is already running")
    
    try:
        return loop.run_until_complete(coro)
    except Exception:
        # Don't close the loop on error - we want to reuse it
        raise

def init_celery_worker():
    """
    Initialize database pool for Celery worker.
    This is called from celery_config.py when a worker process starts.
    """
    logger.info(f"Initializing Celery worker database pool in process {os.getpid()}")
    
    # Set marker for Celery context
    os.environ['SERVER_SOFTWARE'] = 'celery'
    
    # Get or create the event loop for this worker
    loop = get_or_create_event_loop()
    
    # Initialize the pool using the worker's event loop
    try:
        run_async_in_worker_loop(initialize_connection_pool(force_new=True))
        logger.info(f"Celery worker {os.getpid()} database pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Celery worker database pool: {e}", exc_info=True)
        raise

# Export all public functions
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
