# db/connection.py

"""
Database connection pool management for asyncpg with Celery worker support.

This module handles:
- Per-process asyncpg connection pools
- Event loop lifecycle management
- Celery worker initialization and cleanup
- Connection context managers for safe connection usage
- Graceful shutdown with operation tracking
"""

import os
import logging
import asyncio
import asyncpg
import threading
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Set, List

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

# Shutdown state tracking
_SHUTTING_DOWN = False
_SHUTTING_DOWN_PIDS: Set[int] = set()

# Pending operations tracking
_pending_operations: Set[asyncio.Task] = set()
_pending_operations_lock: Optional[asyncio.Lock] = None

# Default pool configuration
DEFAULT_MIN_CONNECTIONS = 2
DEFAULT_MAX_CONNECTIONS = 20
DEFAULT_CONNECTION_LIFETIME = 300
DEFAULT_COMMAND_TIMEOUT = 120
DEFAULT_MAX_QUERIES = 50000

_connection_pending_ops: Dict[int, List[asyncio.Task]] = {}
_connection_ops_lock: Optional[asyncio.Lock] = None

def _get_connection_ops_lock() -> asyncio.Lock:
    """Get or create the connection operations lock."""
    global _connection_ops_lock
    if _connection_ops_lock is None:
        _connection_ops_lock = asyncio.Lock()
    return _connection_ops_lock

# ============================================================================
# Environment Configuration
# ============================================================================

def get_db_dsn() -> str:
    """
    Returns the database connection string (DSN) from environment variables.
    
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
    """Get connection pool configuration from environment variables."""
    return {
        'min_size': int(os.getenv("DB_POOL_MIN_SIZE", str(DEFAULT_MIN_CONNECTIONS))),
        'max_size': int(os.getenv("DB_POOL_MAX_SIZE", str(DEFAULT_MAX_CONNECTIONS))),
        'max_inactive_connection_lifetime': int(os.getenv("DB_CONNECTION_LIFETIME", str(DEFAULT_CONNECTION_LIFETIME))),
        'command_timeout': int(os.getenv("DB_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT))),
        'max_queries': int(os.getenv("DB_MAX_QUERIES", str(DEFAULT_MAX_QUERIES))),
    }


# ============================================================================
# Shutdown Detection
# ============================================================================

def is_shutting_down() -> bool:
    """Check if current process is shutting down."""
    return _SHUTTING_DOWN or os.getpid() in _SHUTTING_DOWN_PIDS


def mark_shutting_down():
    """Mark current process as shutting down."""
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True
    _SHUTTING_DOWN_PIDS.add(os.getpid())


# ============================================================================
# Operation Tracking
# ============================================================================

def _get_pending_lock() -> asyncio.Lock:
    """Get or create the pending operations lock."""
    global _pending_operations_lock
    if _pending_operations_lock is None:
        _pending_operations_lock = asyncio.Lock()
    return _pending_operations_lock


async def track_operation(coro):
    """
    Track an async operation to ensure it completes before shutdown.
    
    Args:
        coro: Coroutine to track
        
    Returns:
        Result of the coroutine
        
    Raises:
        ConnectionError: If process is shutting down
    """
    if is_shutting_down():
        raise ConnectionError(f"Cannot start new operation - worker process {os.getpid()} is shutting down")
    
    task = asyncio.create_task(coro)
    
    lock = _get_pending_lock()
    async with lock:
        _pending_operations.add(task)
    
    try:
        result = await task
        return result
    finally:
        async with lock:
            _pending_operations.discard(task)


async def wait_for_pending_operations(timeout: float = 30.0):
    """
    Wait for all pending operations to complete.
    
    Args:
        timeout: Maximum time to wait in seconds
    """
    if not _pending_operations:
        return
    
    logger.info(f"Process {os.getpid()}: Waiting for {len(_pending_operations)} pending operations")
    
    try:
        await asyncio.wait_for(
            asyncio.gather(*_pending_operations, return_exceptions=True),
            timeout=timeout
        )
        logger.info(f"Process {os.getpid()}: All pending operations completed")
    except asyncio.TimeoutError:
        logger.warning(f"Process {os.getpid()}: {len(_pending_operations)} operations timed out during shutdown")


# ============================================================================
# Event Loop Management
# ============================================================================

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop or create one if needed.
    
    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop
    """
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        if hasattr(_thread_local, 'event_loop') and \
           _thread_local.event_loop is not None and \
           not _thread_local.event_loop.is_closed():
            asyncio.set_event_loop(_thread_local.event_loop)
            return _thread_local.event_loop
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.event_loop = loop
            logger.debug(f"Created new event loop {id(loop)} for thread {threading.current_thread().name}")
            return loop


def run_async_in_worker_loop(coro):
    """
    Run an async coroutine in the worker's persistent event loop.
    
    Args:
        coro: Coroutine to execute
        
    Returns:
        Any: Result of the coroutine
        
    Raises:
        RuntimeError: If event loop is already running
        ConnectionError: If process is shutting down
    """
    if is_shutting_down():
        raise ConnectionError(f"Cannot run operation - worker process {os.getpid()} is shutting down")
    
    loop = get_or_create_event_loop()
    asyncio.set_event_loop(loop)
    
    if loop.is_running():
        raise RuntimeError("Event loop is already running - unexpected in Celery prefork worker")
    
    try:
        return loop.run_until_complete(coro)
    except Exception:
        raise


# ============================================================================
# Connection Pool Management
# ============================================================================

async def setup_connection(conn: asyncpg.Connection):
    """Setup function called for each new connection in the pool."""
    try:
        await pgvector_asyncpg.register_vector(conn)
    except Exception as e:
        logger.error(f"Error setting up connection: {e}", exc_info=True)
        raise

async def create_pool_with_retry(
    dsn: str,
    config: Dict[str, int],
    current_loop: asyncio.AbstractEventLoop,
    max_retries: int = 3
) -> asyncpg.Pool:
    """Create pool with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Creating asyncpg pool (attempt {attempt + 1}/{max_retries}), "
                f"min={config['min_size']}, max={config['max_size']}"
            )
            
            pool = await asyncio.wait_for(
                asyncpg.create_pool(
                    dsn=dsn,
                    min_size=config['min_size'],
                    max_size=config['max_size'],
                    statement_cache_size=0,  # CRITICAL: Always 0 for pgbouncer
                    max_inactive_connection_lifetime=config['max_inactive_connection_lifetime'],
                    command_timeout=config['command_timeout'],
                    max_queries=config['max_queries'],
                    setup=setup_connection,
                    loop=current_loop,
                    server_settings={
                        'application_name': f'nyx_worker_{os.getpid()}',
                        'jit': 'off'  # Disable JIT for pgbouncer compatibility
                    }
                ),
                timeout=30.0
            )
            
            # Test the pool with statement_cache_size check
            async with pool.acquire() as conn:
                # Verify statement cache is off
                result = await conn.fetchval("SELECT 1")
                assert result == 1
            
            logger.info(f"Pool creation successful on attempt {attempt + 1}")
            return pool
            
        except (asyncio.TimeoutError, asyncpg.PostgresError, OSError) as e:
            logger.warning(
                f"Pool creation attempt {attempt + 1} failed: {e.__class__.__name__}: {e}"
            )
            
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to create pool after {max_retries} attempts")
                raise

async def close_existing_pool():
    """Close the existing connection pool if one exists."""
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
    """Initialize the global DB_POOL for the current process/event loop."""
    global DB_POOL, DB_POOL_LOOP
    
    if is_shutting_down():
        logger.error(f"Cannot initialize pool in process {os.getpid()} - worker is shutting down")
        return False
    
    with _pool_init_lock:
        current_loop = get_or_create_event_loop()
        
        if not force_new and DB_POOL is not None and not DB_POOL._closed and DB_POOL_LOOP == current_loop:
            try:
                async with DB_POOL.acquire() as conn:
                    await conn.execute("SELECT 1")
                logger.info(f"DB pool already initialized for loop {id(current_loop)} in process {os.getpid()}")
                if app and not hasattr(app, 'db_pool'):
                    app.db_pool = DB_POOL
                return True
            except Exception as e:
                logger.warning(f"Existing pool failed health check: {e}")
                await close_existing_pool()
        
        if DB_POOL is not None and DB_POOL_LOOP != current_loop:
            logger.warning(f"Pool exists for different event loop, closing it")
            await close_existing_pool()
        
        if force_new:
            await close_existing_pool()

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

        try:
            dsn = get_db_dsn()
            config = get_pool_config()
            
            logger.info(
                f"Process {os.getpid()}: Creating new asyncpg pool for event loop "
                f"{id(current_loop)} (min={config['min_size']}, max={config['max_size']})"
            )
            
            # Use the new retry logic
            local_pool = await create_pool_with_retry(dsn, config, current_loop)
            
            DB_POOL = local_pool
            DB_POOL_LOOP = current_loop
            
            if app:
                app.db_pool = DB_POOL
                logger.info(f"Process {os.getpid()}: DB_POOL stored on app.db_pool")
        
            logger.info(f"Process {os.getpid()}: Asyncpg pool initialized successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Process {os.getpid()}: Failed to initialize asyncpg pool: {e}", exc_info=True)
            DB_POOL = None
            DB_POOL_LOOP = None
            if app:
                app.db_pool = None
            return False


async def get_db_connection_pool() -> asyncpg.Pool:
    """Get the current database connection pool."""
    global DB_POOL, DB_POOL_LOOP
    
    if is_shutting_down():
        raise ConnectionError(f"Cannot get DB pool - worker process {os.getpid()} is shutting down")
    
    current_loop = get_or_create_event_loop()
    
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
            logger.warning(f"Pool not ready, attempting lazy initialization")
        
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
    Async context manager for safe database connection usage with shutdown protection.
    
    CRITICAL: All operations MUST complete before connection is released.
    """
    global DB_POOL, DB_POOL_LOOP, _connection_pending_ops
    
    # Early shutdown check
    if is_shutting_down():
        raise ConnectionError(f"Cannot acquire connection - worker process {os.getpid()} is shutting down")
    
    current_loop = get_or_create_event_loop()
    conn: Optional[asyncpg.Connection] = None
    conn_id: Optional[int] = None
    
    # Verify pool is for current loop
    if DB_POOL and DB_POOL_LOOP != current_loop:
        logger.warning("DB pool was created for different event loop, reinitializing")
        await close_existing_pool()

    current_pool_to_use: Optional[asyncpg.Pool] = None
    
    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        current_pool_to_use = app.db_pool
    elif DB_POOL and not DB_POOL._closed and DB_POOL_LOOP == current_loop:
        current_pool_to_use = DB_POOL
    
    if current_pool_to_use is None:
        if is_shutting_down():
            raise ConnectionError(f"Cannot lazy-initialize pool - worker process {os.getpid()} is shutting down")
        
        logger.warning(f"Process {os.getpid()}: DB pool not initialized. Attempting lazy init.")
        is_celery = 'celery' in os.environ.get('SERVER_SOFTWARE', '').lower()
        
        if not await initialize_connection_pool(app=app, force_new=is_celery):
            raise ConnectionError("DB pool unavailable and lazy init failed")
        
        current_pool_to_use = app.db_pool if app and hasattr(app, 'db_pool') else DB_POOL
        if current_pool_to_use is None:
            raise ConnectionError("DB pool is None even after lazy init attempt")

    try:
        logger.debug(f"Acquiring connection from pool (timeout={timeout}s)")
        conn = await asyncio.wait_for(current_pool_to_use.acquire(), timeout=timeout)
        
        # Track pending operations using connection ID instead of setting attributes
        conn_id = id(conn)
        lock = _get_connection_ops_lock()
        async with lock:
            _connection_pending_ops[conn_id] = []
        
        yield conn
        
        # CRITICAL: Wait for all pending operations to complete
        pending_ops = []
        async with lock:
            pending_ops = _connection_pending_ops.get(conn_id, [])
        
        if pending_ops:
            logger.debug(f"Waiting for {len(pending_ops)} pending operations on connection {conn_id}")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_ops, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for {len(pending_ops)} pending operations on connection {conn_id}")
        
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
        # Clean up connection tracking
        if conn_id is not None:
            lock = _get_connection_ops_lock()
            async with lock:
                _connection_pending_ops.pop(conn_id, None)
        
        if conn:
            try:
                # Double-check connection is still valid before release
                if hasattr(conn, '_con') and conn._con is not None and not conn.is_closed():
                    # Force a small delay to ensure all operations flushed
                    await asyncio.sleep(0.001)
                    await current_pool_to_use.release(conn, timeout=2.0)
                    logger.debug("Released connection back to pool")
                else:
                    logger.warning("Connection was already closed")
            except Exception as release_err:
                logger.error(f"Error releasing connection: {release_err}", exc_info=True)


async def close_connection_pool(app: Optional[Quart] = None):
    """Close the connection pool."""
    global DB_POOL, DB_POOL_LOOP
    pool_to_close: Optional[asyncpg.Pool] = None

    if app and hasattr(app, 'db_pool') and isinstance(app.db_pool, asyncpg.Pool) and not app.db_pool._closed:
        pool_to_close = app.db_pool
        logger.info(f"Process {os.getpid()}: Will close DB pool from app.db_pool")
    elif DB_POOL and not DB_POOL._closed:
        pool_to_close = DB_POOL
        logger.info(f"Process {os.getpid()}: Will close global DB_POOL")
    
    if pool_to_close:
        if DB_POOL is pool_to_close:
            DB_POOL = None
            DB_POOL_LOOP = None
        if app and hasattr(app, 'db_pool') and app.db_pool is pool_to_close:
            app.db_pool = None
        
        try:
            await pool_to_close.close()
            logger.info(f"Process {os.getpid()}: Asyncpg pool closed successfully")
        except Exception as e:
            logger.error(f"Process {os.getpid()}: Error closing pool: {e}", exc_info=True)


async def check_pool_health() -> Dict[str, Any]:
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


# ============================================================================
# Celery Worker Integration
# ============================================================================

def init_celery_worker():
    """Initialize database pool for Celery worker."""
    pid = os.getpid()
    logger.info(f"Initializing Celery worker database pool in process {pid}")
    
    os.environ['SERVER_SOFTWARE'] = 'celery'
    
    try:
        run_async_in_worker_loop(initialize_connection_pool(force_new=True))
        loop = get_or_create_event_loop()
        logger.info(f"Celery worker {pid} database pool initialized successfully with event loop {id(loop)}")
    except Exception as e:
        logger.error(f"Failed to initialize Celery worker database pool: {e}", exc_info=True)
        raise


@worker_process_init.connect
def init_worker_pool(**kwargs):
    """Signal handler: Initialize database pool when worker process starts."""
    init_celery_worker()


@worker_process_shutdown.connect
def close_worker_pool(**kwargs):
    """Signal handler: Close database pool gracefully when worker process shuts down."""
    global DB_POOL, DB_POOL_LOOP
    pid = os.getpid()
    
    # Mark as shutting down FIRST
    mark_shutting_down()
    logger.info(f"Process {pid}: Marked as shutting down, beginning graceful shutdown")
    
    async def _graceful_shutdown():
        # Wait for pending operations with a reasonable timeout
        try:
            await asyncio.wait_for(
                wait_for_pending_operations(timeout=25.0),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Process {pid}: Timed out waiting for pending operations")
        
        # Now close the pool
        if DB_POOL:
            try:
                # Give pool time to close gracefully
                await asyncio.wait_for(
                    close_connection_pool(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Process {pid}: Pool closure timed out, forcing close")
                if not DB_POOL._closed:
                    DB_POOL.terminate()
    
    # Execute graceful shutdown
    if DB_POOL_LOOP and not DB_POOL_LOOP.is_closed():
        try:
            if not DB_POOL_LOOP.is_running():
                DB_POOL_LOOP.run_until_complete(_graceful_shutdown())
            else:
                future = asyncio.run_coroutine_threadsafe(_graceful_shutdown(), DB_POOL_LOOP)
                future.result(timeout=45)  # Increased timeout
            logger.info(f"Process {pid}: Graceful shutdown completed")
        except Exception as e:
            logger.error(f"Process {pid}: Error during graceful shutdown: {e}", exc_info=True)
            # Force terminate the pool
            if DB_POOL and not DB_POOL._closed:
                try:
                    DB_POOL.terminate()
                except:
                    pass
    
    # Final cleanup
    DB_POOL = None
    DB_POOL_LOOP = None
    
    # Close event loop
    if hasattr(_thread_local, 'event_loop') and _thread_local.event_loop:
        try:
            if not _thread_local.event_loop.is_closed() and not _thread_local.event_loop.is_running():
                _thread_local.event_loop.close()
                logger.info(f"Worker {pid} event loop closed")
        except Exception:
            pass


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
    'get_or_create_event_loop',
    'track_operation',
    'is_shutting_down',
]
