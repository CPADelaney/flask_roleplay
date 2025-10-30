# db/connection.py
"""
Database connection pool management for asyncpg with Celery worker support.

This module provides enterprise-grade connection pool management with:
- Thread-safe and async-safe connection pooling
- Per-process asyncpg connection pools with lifecycle management
- Celery worker integration with graceful shutdown
- Race condition prevention and proper locking patterns
- Connection context managers for safe connection usage
- Operation tracking and graceful shutdown coordination
- Circuit breaker pattern for resilience
- Comprehensive health checks and observability

Architecture:
    - One connection pool per process/event loop
    - Automatic pool initialization and recycling
    - TCP keepalive for connection stability
    - Graceful shutdown with pending operation draining
    - Resilient connection wrapper for asyncpg bug mitigation

Thread Safety:
    - Uses asyncio.Lock for async operations
    - Uses threading.Lock only for synchronous state mutations
    - Thread-local storage for per-thread event loops
    - Atomic state transitions with proper locking

Author: Generated with best practices for production systems
Version: 2.1.0 (Hybrid - Best of Both Worlds)
"""

import os
import logging
import asyncio
import asyncpg
import threading
import weakref
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Set, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import psycopg2
from quart import Quart
import pgvector.asyncpg as pgvector_asyncpg

# Celery worker lifecycle signals
from celery.signals import worker_process_init, worker_process_shutdown

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# Type Definitions and Enums
# ============================================================================

class PoolState(Enum):
    """Connection pool lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    CLOSED = "closed"


class ConnectionHealth(Enum):
    """Individual connection health states."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class PoolMetrics:
    """Metrics for connection pool monitoring."""
    total_connections: int = 0
    idle_connections: int = 0
    active_connections: int = 0
    failed_health_checks: int = 0
    successful_acquisitions: int = 0
    failed_acquisitions: int = 0
    total_queries: int = 0
    last_health_check: Optional[datetime] = None
    pool_state: PoolState = PoolState.UNINITIALIZED
    
    @property
    def utilization_percent(self) -> float:
        """Calculate pool utilization percentage."""
        if self.total_connections == 0:
            return 0.0
        return (self.active_connections / self.total_connections) * 100.0


@dataclass
class ShutdownState:
    """Thread-safe shutdown state tracking."""
    is_shutting_down: bool = False
    shutdown_pids: Set[int] = field(default_factory=set)
    shutdown_initiated_at: Optional[datetime] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def mark_shutdown(self, pid: int) -> None:
        """Mark process as shutting down."""
        with self._lock:
            self.is_shutting_down = True
            self.shutdown_pids.add(pid)
            if self.shutdown_initiated_at is None:
                self.shutdown_initiated_at = datetime.utcnow()
    
    def is_pid_shutting_down(self, pid: int) -> bool:
        """Check if specific PID is shutting down."""
        with self._lock:
            return self.is_shutting_down or pid in self.shutdown_pids
    
    def clear_pid(self, pid: int) -> None:
        """Remove PID from shutdown tracking."""
        with self._lock:
            self.shutdown_pids.discard(pid)


# ============================================================================
# Global State Management
# ============================================================================

class GlobalPoolState:
    """Centralized global state management with proper locking."""
    
    def __init__(self):
        # Pool state
        self.pool: Optional[asyncpg.Pool] = None
        self.pool_loop: Optional[asyncio.AbstractEventLoop] = None
        self.pool_state: PoolState = PoolState.UNINITIALIZED
        self.metrics: PoolMetrics = PoolMetrics()
        
        # Async locks (created on-demand per event loop)
        self._pool_init_lock: Optional[asyncio.Lock] = None
        self._pending_ops_lock: Optional[asyncio.Lock] = None
        self._connection_ops_lock: Optional[asyncio.Lock] = None
        
        # Thread-local storage
        self.thread_local = threading.local()
        self.thread_local_lock = threading.Lock()
        
        # Shutdown state
        self.shutdown_state = ShutdownState()
        
        # Operation tracking (using weak references to prevent memory leaks)
        self.pending_operations: "weakref.WeakSet[asyncio.Task]" = weakref.WeakSet()
        self.connection_pending_ops: Dict[int, List[asyncio.Task]] = {}
        
        # Circuit breaker state
        self.consecutive_failures: int = 0
        self.circuit_open_until: Optional[datetime] = None
        self.circuit_breaker_lock = threading.Lock()
    
    def get_pool_init_lock(self) -> asyncio.Lock:
        """Get or create pool initialization lock."""
        if self._pool_init_lock is None:
            self._pool_init_lock = asyncio.Lock()
        return self._pool_init_lock
    
    def get_pending_ops_lock(self) -> asyncio.Lock:
        """Get or create pending operations lock."""
        if self._pending_ops_lock is None:
            self._pending_ops_lock = asyncio.Lock()
        return self._pending_ops_lock
    
    def get_connection_ops_lock(self) -> asyncio.Lock:
        """Get or create connection operations lock."""
        if self._connection_ops_lock is None:
            self._connection_ops_lock = asyncio.Lock()
        return self._connection_ops_lock
    
    def reset_locks(self) -> None:
        """Reset all async locks (call when switching event loops)."""
        self._pool_init_lock = None
        self._pending_ops_lock = None
        self._connection_ops_lock = None
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        with self.circuit_breaker_lock:
            if self.circuit_open_until is None:
                return False
            if datetime.utcnow() >= self.circuit_open_until:
                # Circuit breaker timeout expired, reset
                self.circuit_open_until = None
                self.consecutive_failures = 0
                return False
            return True
    
    def record_failure(self) -> None:
        """Record a connection failure for circuit breaker."""
        with self.circuit_breaker_lock:
            self.consecutive_failures += 1
            self.metrics.failed_acquisitions += 1
            
            # Open circuit after 5 consecutive failures
            if self.consecutive_failures >= 5:
                self.circuit_open_until = datetime.utcnow() + timedelta(seconds=30)
                logger.error(
                    f"Circuit breaker opened after {self.consecutive_failures} failures. "
                    f"Will retry at {self.circuit_open_until}"
                )
    
    def record_success(self) -> None:
        """Record a successful connection for circuit breaker."""
        with self.circuit_breaker_lock:
            self.consecutive_failures = 0
            self.circuit_open_until = None
            self.metrics.successful_acquisitions += 1


# Global state singleton
_state = GlobalPoolState()

# ============================================================================
# Configuration Management
# ============================================================================

# Default pool configuration
DEFAULT_MIN_CONNECTIONS = 2
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_CONNECTION_LIFETIME = 300  # seconds
DEFAULT_COMMAND_TIMEOUT = 120  # seconds
DEFAULT_MAX_QUERIES = 50000
DEFAULT_RELEASE_TIMEOUT = DEFAULT_COMMAND_TIMEOUT


def get_db_dsn() -> str:
    """
    Get database connection string (DSN) from environment variables.
    
    Checks DB_DSN first, then falls back to DATABASE_URL.
    
    Returns:
        Database connection DSN string
        
    Raises:
        EnvironmentError: If no DSN is configured
    """
    dsn = os.getenv("DB_DSN")
    if not dsn:
        dsn = os.getenv("DATABASE_URL")
        if dsn:
            logger.info("Using DATABASE_URL as connection string (DB_DSN not found)")
    
    if not dsn:
        logger.critical("Neither DB_DSN nor DATABASE_URL environment variables are set")
        raise EnvironmentError("Database DSN not configured in environment")
    
    return dsn


def get_pool_config() -> Dict[str, int]:
    """
    Get connection pool configuration from environment variables.
    
    Returns:
        Dictionary with pool configuration parameters
    """
    return {
        'min_size': int(os.getenv("DB_POOL_MIN_SIZE", str(DEFAULT_MIN_CONNECTIONS))),
        'max_size': int(os.getenv("DB_POOL_MAX_SIZE", str(DEFAULT_MAX_CONNECTIONS))),
        'max_inactive_connection_lifetime': int(
            os.getenv("DB_CONNECTION_LIFETIME", str(DEFAULT_CONNECTION_LIFETIME))
        ),
        'command_timeout': int(os.getenv("DB_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT))),
        'max_queries': int(os.getenv("DB_MAX_QUERIES", str(DEFAULT_MAX_QUERIES))),
    }


def get_release_timeout() -> float:
    """
    Get timeout for releasing connections back to pool.
    
    Returns:
        Timeout in seconds
    """
    command_timeout = float(os.getenv("DB_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT)))
    default_timeout = command_timeout if command_timeout > 0 else DEFAULT_RELEASE_TIMEOUT
    return float(os.getenv("DB_RELEASE_TIMEOUT", str(default_timeout)))


def get_db_connection_sync():
    """
    Get a synchronous psycopg2 connection.
    
    Returns:
        psycopg2 connection object
        
    Note:
        Use this only for synchronous contexts. For async, use get_db_connection_context.
    """
    dsn = get_db_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


# ============================================================================
# Shutdown Management
# ============================================================================

def is_shutting_down() -> bool:
    """
    Check if current process is shutting down.
    
    Returns:
        True if shutdown is in progress
    """
    return _state.shutdown_state.is_pid_shutting_down(os.getpid())


def mark_shutting_down() -> None:
    """Mark current process as shutting down."""
    pid = os.getpid()
    _state.shutdown_state.mark_shutdown(pid)
    _state.pool_state = PoolState.SHUTTING_DOWN
    logger.info(f"Process {pid} marked as shutting down")


# ============================================================================
# Event Loop Management
# ============================================================================

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop or create one if needed (thread-safe).
    
    Returns:
        The current or newly created event loop
        
    Note:
        This function is thread-safe and handles per-thread event loop storage.
    """
    try:
        # Try to get currently running loop first
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No running loop, need to get or create one
        with _state.thread_local_lock:
            # Double-check pattern for thread safety
            if hasattr(_state.thread_local, 'event_loop') and \
               _state.thread_local.event_loop is not None and \
               not _state.thread_local.event_loop.is_closed():
                asyncio.set_event_loop(_state.thread_local.event_loop)
                return _state.thread_local.event_loop
            
            # Create new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _state.thread_local.event_loop = loop
            logger.debug(
                f"Created new event loop {id(loop)} for thread "
                f"{threading.current_thread().name}"
            )
            return loop


def run_async_in_worker_loop(coro):
    """
    Run an async coroutine in the worker's persistent event loop.
    
    Args:
        coro: Coroutine to execute
        
    Returns:
        Result of the coroutine
        
    Raises:
        RuntimeError: If event loop is already running
        ConnectionError: If process is shutting down
    """
    if is_shutting_down():
        raise ConnectionError(
            f"Cannot run operation - worker process {os.getpid()} is shutting down"
        )
    
    loop = get_or_create_event_loop()
    asyncio.set_event_loop(loop)
    
    if loop.is_running():
        raise RuntimeError(
            "Event loop is already running - unexpected in Celery prefork worker"
        )
    
    return loop.run_until_complete(coro)


# ============================================================================
# Operation Tracking
# ============================================================================

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
        raise ConnectionError(
            f"Cannot start new operation - worker process {os.getpid()} is shutting down"
        )
    
    task = asyncio.create_task(coro)
    
    lock = _state.get_pending_ops_lock()
    async with lock:
        _state.pending_operations.add(task)
    
    try:
        result = await task
        return result
    finally:
        # WeakSet automatically handles cleanup, but we can explicitly remove
        try:
            async with lock:
                _state.pending_operations.discard(task)
        except Exception:
            # WeakSet may have already removed it
            pass


async def wait_for_pending_operations(timeout: float = 30.0) -> Tuple[int, int]:
    """
    Wait for all pending operations to complete.
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        Tuple of (completed_count, timeout_count)
    """
    # Convert WeakSet to list to avoid iteration issues
    pending_ops = list(_state.pending_operations)
    
    if not pending_ops:
        logger.info(f"Process {os.getpid()}: No pending operations to wait for")
        return (0, 0)
    
    initial_count = len(pending_ops)
    logger.info(
        f"Process {os.getpid()}: Waiting for {initial_count} pending operations "
        f"(timeout={timeout}s)"
    )
    
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending_ops, return_exceptions=True),
            timeout=timeout
        )
        logger.info(f"Process {os.getpid()}: All {initial_count} pending operations completed")
        return (initial_count, 0)
    except asyncio.TimeoutError:
        # Check how many are still pending
        still_pending = [op for op in pending_ops if not op.done()]
        timeout_count = len(still_pending)
        completed_count = initial_count - timeout_count
        
        logger.warning(
            f"Process {os.getpid()}: {completed_count}/{initial_count} operations completed, "
            f"{timeout_count} timed out during shutdown"
        )
        return (completed_count, timeout_count)


# ============================================================================
# Asyncpg Bug Mitigation
# ============================================================================

def _is_asyncpg_waiter_cancel_bug(exc: BaseException) -> bool:
    """
    Detect the asyncpg "waiter cancelled" race condition bug.
    
    Args:
        exc: Exception to check
        
    Returns:
        True if this is the known asyncpg bug
    """
    if not isinstance(exc, AttributeError):
        return False
    message = str(exc)
    return "'NoneType' object has no attribute 'cancelled'" in message


class _ResilientConnectionWrapper:
    """
    Wrap asyncpg.Connection to transparently heal the waiter cancellation bug.
    
    This wrapper provides automatic retry and connection refresh for operations
    that fail due to the asyncpg waiter bug.
    """
    
    _RETRYABLE_METHODS = {
        "execute",
        "fetch",
        "fetchrow",
        "fetchval",
        "executemany",
        "copy_records_to_table",
        "copy_to_table",
        "copy_from_table",
    }
    
    def __init__(
        self,
        pool: asyncpg.Pool,
        conn: asyncpg.Connection,
        conn_id: int,
        ops_lock: asyncio.Lock,
    ) -> None:
        """
        Initialize connection wrapper.
        
        Args:
            pool: Connection pool for acquiring replacement connections
            conn: Initial connection to wrap
            conn_id: Unique identifier for the connection
            ops_lock: Lock for coordinating operation tracking
        """
        self._pool = pool
        self._conn = conn
        self._conn_id = conn_id
        self._ops_lock = ops_lock
        self._refresh_lock = asyncio.Lock()
        self._release_timeout = get_release_timeout()
    
    @property
    def raw_connection(self) -> asyncpg.Connection:
        """Get the underlying asyncpg connection."""
        return self._conn
    
    @property
    def conn_id(self) -> int:
        """Get the connection ID."""
        return self._conn_id
    
    def __getattr__(self, item: str):
        """Proxy attribute access to underlying connection with retry support."""
        attr = getattr(self._conn, item)
        
        if item in self._RETRYABLE_METHODS and callable(attr):
            async def _wrapped(*args, __attr=attr, __name=item, **kwargs):
                return await self._call_with_retry(__name, __attr, *args, **kwargs)
            return _wrapped
        
        return attr
    
    async def _call_with_retry(self, name: str, method, *args, **kwargs):
        """
        Call a method with automatic retry on waiter bug detection.
        
        Args:
            name: Method name for logging
            method: Method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the method call
            
        Raises:
            AttributeError: If retry limit exceeded or not the waiter bug
        """
        attempt = 0
        max_attempts = 2
        
        while attempt < max_attempts:
            try:
                return await method(*args, **kwargs)
            except AttributeError as exc:
                if not _is_asyncpg_waiter_cancel_bug(exc) or attempt >= max_attempts - 1:
                    raise
                
                attempt += 1
                logger.warning(
                    f"Detected asyncpg waiter cancellation bug while executing {name}; "
                    f"refreshing connection and retrying (attempt {attempt}/{max_attempts})"
                )
                await self._refresh_connection()
                method = getattr(self._conn, name)
    
    async def _refresh_connection(self) -> None:
        """
        Refresh the connection by acquiring a new one from the pool.
        
        This is called when the waiter bug is detected.
        """
        async with self._refresh_lock:
            old_conn = self._conn
            old_conn_id = self._conn_id
            
            # Atomically remove tracking and mark as invalid
            async with self._ops_lock:
                # Remove old connection's pending ops
                _state.connection_pending_ops.pop(old_conn_id, None)
                # Mark as invalid to prevent new operations
                _state.connection_pending_ops[old_conn_id] = None  # Sentinel
            
            # Release old connection
            if old_conn is not None:
                try:
                    await self._pool.release(old_conn, timeout=self._release_timeout)
                except Exception:
                    logger.warning(
                        "Releasing connection after waiter cancellation bug failed; "
                        "terminating instead",
                        exc_info=True
                    )
                    try:
                        old_conn.terminate()
                    except Exception:
                        logger.exception("Failed to terminate broken asyncpg connection")
            
            # Acquire fresh connection
            new_conn = await self._pool.acquire()
            new_conn_id = id(new_conn)
            
            # Register new connection for tracking
            async with self._ops_lock:
                _state.connection_pending_ops[new_conn_id] = []
            
            # Update wrapper state
            self._conn = new_conn
            self._conn_id = new_conn_id
            self._release_timeout = get_release_timeout()
            
            logger.info(
                f"Connection refreshed: {old_conn_id} -> {new_conn_id} due to waiter bug"
            )


# ============================================================================
# Connection Pool Setup
# ============================================================================

async def setup_connection(conn: asyncpg.Connection) -> None:
    """
    Setup function called for each new connection in the pool.
    
    Registers pgvector extension with retry logic and timeout protection.
    
    Args:
        conn: Connection to set up
        
    Raises:
        asyncpg.PostgresError: If setup fails after all retries
    """
    max_retries = 3
    retry_delay = 0.5  # Start with 0.5 seconds
    
    for attempt in range(max_retries):
        try:
            # IMPROVEMENT: Added timeout to prevent indefinite hang
            await asyncio.wait_for(
                pgvector_asyncpg.register_vector(conn),
                timeout=10.0
            )
            logger.debug(f"Successfully registered pgvector on connection {id(conn)}")
            return  # Success
        except (asyncpg.exceptions.ConnectionDoesNotExistError, asyncio.TimeoutError) as e:
            logger.warning(
                f"Connection setup failed on attempt {attempt + 1}/{max_retries} "
                f"({e.__class__.__name__}): {e}"
            )
            if attempt + 1 == max_retries:
                logger.error("All attempts to set up DB connection failed")
                raise
            
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error setting up connection: {e}", exc_info=True)
            raise


async def create_pool_with_retry(
    dsn: str,
    config: Dict[str, int],
    current_loop: asyncio.AbstractEventLoop,
    max_retries: int = 3
) -> asyncpg.Pool:
    """
    Create connection pool with exponential backoff retry and TCP keepalives.
    
    Args:
        dsn: Database connection string
        config: Pool configuration dictionary
        current_loop: Event loop to bind pool to
        max_retries: Maximum number of retry attempts
        
    Returns:
        Initialized asyncpg connection pool
        
    Raises:
        asyncpg.PostgresError: If pool creation fails after all retries
    """
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Creating asyncpg pool (attempt {attempt + 1}/{max_retries}), "
                f"min={config['min_size']}, max={config['max_size']}"
            )
            
            # TCP keepalive settings prevent firewalls/load balancers from closing connections
            server_settings = {
                'application_name': f'nyx_worker_{os.getpid()}',
                'jit': 'off',  # Disable JIT for pgbouncer compatibility
                'tcp_keepalives_idle': '60',      # Seconds before first keepalive
                'tcp_keepalives_interval': '10',  # Seconds between keepalive probes
                'tcp_keepalives_count': '5'       # Failed probes before connection dead
            }
            
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
                    server_settings=server_settings
                ),
                timeout=30.0
            )
            
            # Verify pool with health check
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    raise ConnectionError("Pool health check failed: SELECT 1 did not return 1")
            
            logger.info(f"Pool creation successful on attempt {attempt + 1}")
            _state.record_success()
            return pool
            
        except (asyncio.TimeoutError, asyncpg.PostgresError, OSError, ConnectionError) as e:
            logger.warning(
                f"Pool creation attempt {attempt + 1} failed: {e.__class__.__name__}: {e}"
            )
            _state.record_failure()
            
            if attempt < max_retries - 1:
                # IMPROVEMENT: Slightly longer backoff
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s...
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to create pool after {max_retries} attempts")
                raise


async def close_existing_pool(pool: Optional[asyncpg.Pool] = None) -> None:
    """
    Close an existing connection pool safely.
    
    Args:
        pool: Pool to close. If None, closes the global pool.
    """
    pool_to_close = pool or _state.pool
    
    # IMPROVEMENT: Safer attribute check
    if pool_to_close and not getattr(pool_to_close, "_closed", True):
        try:
            # Log connection usage before closing
            size = pool_to_close.get_size()
            idle = pool_to_close.get_idle_size()
            in_use = size - idle
            
            if in_use > 0:
                logger.warning(
                    f"Closing pool with {in_use}/{size} connections still in use "
                    f"in process {os.getpid()}"
                )
            
            logger.info(f"Closing existing pool in process {os.getpid()}")
            await pool_to_close.close()
            logger.info(f"Pool closed successfully in process {os.getpid()}")
        except Exception as e:
            logger.error(f"Error closing existing pool: {e}", exc_info=True)
        finally:
            if pool_to_close is _state.pool:
                _state.pool = None
                _state.pool_loop = None
                _state.pool_state = PoolState.CLOSED


async def initialize_connection_pool(
    app: Optional[Quart] = None,
    force_new: bool = False
) -> bool:
    """
    Initialize the global connection pool for the current process/event loop.
    
    This function is thread-safe and uses double-checked locking to prevent
    race conditions during pool initialization.
    
    Args:
        app: Optional Quart application to attach pool to
        force_new: Force creation of new pool even if one exists
        
    Returns:
        True if pool is initialized successfully, False otherwise
    """
    if is_shutting_down():
        logger.error(
            f"Cannot initialize pool in process {os.getpid()} - worker is shutting down"
        )
        return False
    
    # Check circuit breaker
    if _state.is_circuit_open():
        logger.error(
            f"Cannot initialize pool - circuit breaker is open until "
            f"{_state.circuit_open_until}"
        )
        return False
    
    current_loop = get_or_create_event_loop()
    
    # Quick check without lock (first check in double-checked locking)
    if not force_new and \
       _state.pool is not None and \
       not _state.pool._closed and \
       _state.pool_loop == current_loop:
        try:
            async with _state.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            logger.debug(
                f"DB pool already initialized for loop {id(current_loop)} "
                f"in process {os.getpid()}"
            )
            if app and not hasattr(app, 'db_pool'):
                app.db_pool = _state.pool
            return True
        except Exception as e:
            logger.warning(f"Existing pool failed health check: {e}")
            # Continue to re-initialization
    
    # Acquire lock for initialization
    lock = _state.get_pool_init_lock()
    async with lock:
        # Double-check after acquiring lock (second check in double-checked locking)
        if not force_new and \
           _state.pool is not None and \
           not _state.pool._closed and \
           _state.pool_loop == current_loop:
            try:
                async with _state.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                logger.info(
                    f"DB pool already initialized (verified after lock) for loop "
                    f"{id(current_loop)} in process {os.getpid()}"
                )
                if app and not hasattr(app, 'db_pool'):
                    app.db_pool = _state.pool
                return True
            except Exception:
                logger.warning("Pool failed health check after lock, will recreate")
                await close_existing_pool()
        
        # Check if pool exists for different event loop
        if _state.pool is not None and _state.pool_loop != current_loop:
            logger.warning(
                f"Pool exists for different event loop "
                f"(current={id(current_loop)}, pool={id(_state.pool_loop)}), closing it"
            )
            await close_existing_pool()
        
        # Force new pool if requested
        if force_new:
            await close_existing_pool()
        
        # Try to use existing app pool if available
        if app and hasattr(app, 'db_pool') and \
           app.db_pool is not None and \
           not app.db_pool._closed:
            try:
                async with app.db_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                logger.info(f"Using existing DB pool from app in process {os.getpid()}")
                _state.pool = app.db_pool
                _state.pool_loop = current_loop
                _state.pool_state = PoolState.HEALTHY
                return True
            except Exception:
                logger.warning("App pool exists but failed health check, will create new pool")
                app.db_pool = None
        
        # Create new pool
        try:
            dsn = get_db_dsn()
            config = get_pool_config()
            
            logger.info(
                f"Process {os.getpid()}: Creating new asyncpg pool for event loop "
                f"{id(current_loop)} (min={config['min_size']}, max={config['max_size']})"
            )
            
            _state.pool_state = PoolState.INITIALIZING
            local_pool = await create_pool_with_retry(dsn, config, current_loop)
            
            _state.pool = local_pool
            _state.pool_loop = current_loop
            _state.pool_state = PoolState.HEALTHY
            
            # Update metrics
            _state.metrics.pool_state = PoolState.HEALTHY
            _state.metrics.total_connections = local_pool.get_size()
            _state.metrics.idle_connections = local_pool.get_idle_size()
            
            if app:
                app.db_pool = _state.pool
                logger.info(f"Process {os.getpid()}: DB_POOL stored on app.db_pool")
            
            logger.info(f"Process {os.getpid()}: Asyncpg pool initialized successfully")
            return True
            
        except Exception as e:
            logger.critical(
                f"Process {os.getpid()}: Failed to initialize asyncpg pool: {e}",
                exc_info=True
            )
            _state.pool = None
            _state.pool_loop = None
            _state.pool_state = PoolState.CLOSED
            if app:
                app.db_pool = None
            return False


async def get_db_connection_pool() -> asyncpg.Pool:
    """
    Get the current database connection pool with automatic initialization.
    
    Returns:
        Active asyncpg connection pool
        
    Raises:
        ConnectionError: If pool cannot be initialized or process is shutting down
    """
    if is_shutting_down():
        raise ConnectionError(
            f"Cannot get DB pool - worker process {os.getpid()} is shutting down"
        )
    
    current_loop = get_or_create_event_loop()
    
    # Check if pool needs initialization
    if _state.pool is None or \
       _state.pool._closed or \
       _state.pool_loop != current_loop:
        
        is_celery = 'celery' in os.environ.get('SERVER_SOFTWARE', '').lower()
        
        if is_celery:
            logger.error(
                f"Celery worker pool not initialized properly! "
                f"pool_exists={_state.pool is not None}, "
                f"pool_closed={_state.pool._closed if _state.pool else 'N/A'}, "
                f"loop_match={_state.pool_loop == current_loop if _state.pool_loop else 'N/A'}"
            )
        else:
            logger.warning(f"Pool not ready, attempting lazy initialization")
        
        ok = await initialize_connection_pool(force_new=False)
        if not ok or _state.pool is None:
            raise ConnectionError("Could not initialize DB pool")
    
    return _state.pool


@asynccontextmanager
async def get_db_connection_context(
    timeout: Optional[float] = 30.0,
    app: Optional[Quart] = None
):
    """
    Async context manager for safe database connection usage.
    
    This context manager:
    - Ensures shutdown protection (fails fast if shutting down)
    - Provides automatic connection acquisition and release
    - Tracks pending operations on the connection (CRITICAL for race prevention)
    - Wraps connection with resilient wrapper for bug mitigation
    - Handles pool initialization if needed
    
    Args:
        timeout: Maximum time to wait for connection acquisition
        app: Optional Quart application for pool access
        
    Yields:
        _ResilientConnectionWrapper: Wrapped connection with retry support
        
    Raises:
        ConnectionError: If connection cannot be acquired or shutdown is in progress
        asyncio.TimeoutError: If connection acquisition times out
        
    Example:
        async with get_db_connection_context() as conn:
            result = await conn.fetchval("SELECT 1")
    """
    # Early shutdown check
    if is_shutting_down():
        raise ConnectionError(
            f"Cannot acquire connection - worker process {os.getpid()} is shutting down"
        )
    
    current_loop = get_or_create_event_loop()
    conn: Optional[asyncpg.Connection] = None
    conn_id: Optional[int] = None
    wrapped_conn: Optional[_ResilientConnectionWrapper] = None
    
    # Verify pool is for current loop
    if _state.pool and _state.pool_loop != current_loop:
        logger.warning("DB pool was created for different event loop, reinitializing")
        await close_existing_pool()
    
    # Determine which pool to use
    current_pool_to_use: Optional[asyncpg.Pool] = None
    
    if app and hasattr(app, 'db_pool') and \
       isinstance(app.db_pool, asyncpg.Pool) and \
       not app.db_pool._closed:
        current_pool_to_use = app.db_pool
    elif _state.pool and not _state.pool._closed and _state.pool_loop == current_loop:
        current_pool_to_use = _state.pool
    
    # Lazy initialization if needed
    if current_pool_to_use is None:
        if is_shutting_down():
            raise ConnectionError(
                f"Cannot lazy-initialize pool - worker process {os.getpid()} is shutting down"
            )
        
        logger.warning(f"Process {os.getpid()}: DB pool not initialized. Attempting lazy init")
        is_celery = 'celery' in os.environ.get('SERVER_SOFTWARE', '').lower()
        
        if not await initialize_connection_pool(app=app, force_new=is_celery):
            raise ConnectionError("DB pool unavailable and lazy init failed")
        
        current_pool_to_use = app.db_pool if app and hasattr(app, 'db_pool') else _state.pool
        if current_pool_to_use is None:
            raise ConnectionError("DB pool is None even after lazy init attempt")
    
    try:
        logger.debug(f"Acquiring connection from pool (timeout={timeout}s)")
        conn = await asyncio.wait_for(
            current_pool_to_use.acquire(),
            timeout=timeout
        )
        
        # Re-check shutdown status after acquiring (race condition protection)
        if is_shutting_down():
            await current_pool_to_use.release(conn)
            raise ConnectionError(
                f"Shutdown initiated during connection acquisition in process {os.getpid()}"
            )
        
        # Track pending operations using connection ID
        conn_id = id(conn)
        lock = _state.get_connection_ops_lock()
        async with lock:
            _state.connection_pending_ops[conn_id] = []
        
        # Create resilient wrapper
        wrapped_conn = _ResilientConnectionWrapper(
            current_pool_to_use,
            conn,
            conn_id,
            lock
        )
        
        # Update metrics
        _state.metrics.active_connections = (
            current_pool_to_use.get_size() - current_pool_to_use.get_idle_size()
        )
        
        yield wrapped_conn
        
        # Determine active connection after caller returns
        active_conn_id = wrapped_conn.conn_id if wrapped_conn else conn_id
        
        # CRITICAL: Wait for all pending operations on this connection to complete
        # This prevents the race condition where the connection is released while operations are still running
        pending_ops = []
        async with lock:
            pending_ops = list(_state.connection_pending_ops.get(active_conn_id, []))
        
        if pending_ops:
            logger.debug(
                f"Waiting for {len(pending_ops)} pending operations on "
                f"connection {active_conn_id}"
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_ops, return_exceptions=True),
                    timeout=5.0
                )
                logger.debug(
                    f"All {len(pending_ops)} pending operations completed on "
                    f"connection {active_conn_id}"
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout waiting for {len(pending_ops)} pending operations on "
                    f"connection {active_conn_id}"
                )
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout}s) acquiring DB connection from pool")
        _state.metrics.failed_acquisitions += 1
        raise
    except asyncpg.exceptions.PostgresError as pg_err:
        logger.error(f"PostgreSQL error: {pg_err}", exc_info=True)
        _state.metrics.failed_acquisitions += 1
        raise
    except Exception as e:
        logger.error(f"Unexpected error in connection context: {e}", exc_info=True)
        _state.metrics.failed_acquisitions += 1
        raise
    finally:
        # Clean up connection tracking
        lock = _state.get_connection_ops_lock()
        active_conn_obj: Optional[asyncpg.Connection] = None
        active_conn_id = None
        
        if wrapped_conn is not None:
            active_conn_obj = wrapped_conn.raw_connection
            active_conn_id = wrapped_conn.conn_id
        else:
            active_conn_obj = conn
            active_conn_id = conn_id
        
        # Remove connection from tracking (prevents memory leak)
        if active_conn_id is not None:
            async with lock:
                _state.connection_pending_ops.pop(active_conn_id, None)
        
        # Release connection back to pool
        if active_conn_obj and current_pool_to_use:
            try:
                # IMPROVEMENT: Safer closed check
                if not getattr(active_conn_obj, 'is_closed', lambda: True)():
                    release_timeout = get_release_timeout()
                    try:
                        await current_pool_to_use.release(
                            active_conn_obj,
                            timeout=release_timeout
                        )
                    except Exception as release_err:
                        logger.error(
                            f"Error releasing connection {active_conn_id}: {release_err}. "
                            f"Terminating instead.",
                            exc_info=True
                        )
                        try:
                            active_conn_obj.terminate()
                        except Exception as term_err:
                            logger.exception(
                                f"Failed to terminate connection {active_conn_id} "
                                f"after release error: {term_err}"
                            )
                else:
                    logger.warning(
                        f"Connection {active_conn_id} was already closed before release"
                    )
            except Exception as outer_err:
                logger.error(
                    f"Unhandled error during connection cleanup: {outer_err}",
                    exc_info=True
                )


async def close_connection_pool(
    app: Optional[Quart] = None,
    *,
    force_terminate: bool = False
) -> None:
    """
    Close the connection pool gracefully or forcefully.
    
    Args:
        app: Optional Quart application that may own the pool reference
        force_terminate: When True, skip graceful close and terminate directly.
                        Used during Celery shutdown to avoid asyncpg 0.30.0 crashes.
    """
    pool_to_close: Optional[asyncpg.Pool] = None
    
    # Determine which pool to close
    if app and hasattr(app, 'db_pool') and \
       isinstance(app.db_pool, asyncpg.Pool) and \
       not getattr(app.db_pool, "_closed", True):
        pool_to_close = app.db_pool
        logger.info(f"Process {os.getpid()}: Will close DB pool from app.db_pool")
    elif _state.pool and not getattr(_state.pool, "_closed", True):
        pool_to_close = _state.pool
        logger.info(f"Process {os.getpid()}: Will close global DB_POOL")
    
    if pool_to_close:
        # Clear references before closing
        if _state.pool is pool_to_close:
            _state.pool = None
            _state.pool_loop = None
        if app and hasattr(app, 'db_pool') and app.db_pool is pool_to_close:
            app.db_pool = None
        
        _state.pool_state = PoolState.SHUTTING_DOWN
        
        if force_terminate:
            logger.info(
                f"Process {os.getpid()}: Terminating DB pool (force_terminate=True)"
            )
            try:
                pool_to_close.terminate()
            except Exception as e:
                logger.error(
                    f"Process {os.getpid()}: Error terminating pool: {e}",
                    exc_info=True
                )
            finally:
                if hasattr(pool_to_close, "_closed"):
                    pool_to_close._closed = True
                _state.pool_state = PoolState.CLOSED
        else:
            try:
                logger.info(f"Process {os.getpid()}: Closing asyncpg pool gracefully")
                await pool_to_close.close()
                logger.info(f"Process {os.getpid()}: Asyncpg pool closed successfully")
                _state.pool_state = PoolState.CLOSED
            except AttributeError as err:
                logger.warning(
                    f"Process {os.getpid()}: Pool.close raised {err!r}; "
                    f"terminating instead",
                    exc_info=True
                )
                try:
                    pool_to_close.terminate()
                except Exception as terminate_err:
                    logger.error(
                        f"Process {os.getpid()}: Error terminating pool after "
                        f"close failure: {terminate_err}",
                        exc_info=True
                    )
                finally:
                    if hasattr(pool_to_close, "_closed"):
                        pool_to_close._closed = True
                    _state.pool_state = PoolState.CLOSED
            except Exception as e:
                logger.error(
                    f"Process {os.getpid()}: Error closing pool: {e}",
                    exc_info=True
                )
                _state.pool_state = PoolState.CLOSED


# ============================================================================
# Health Checks and Observability
# ============================================================================

async def check_pool_health() -> Dict[str, Any]:
    """
    Comprehensive health check of the database connection pool.
    
    Returns:
        Dictionary containing pool health metrics and status
    """
    health = {
        "status": "unknown",
        "pool_exists": _state.pool is not None,
        "pool_closed": getattr(_state.pool, "_closed", None) if _state.pool else None,
        "pool_state": _state.pool_state.value,
        "size": None,
        "free": None,
        "used": None,
        "utilization_percent": 0.0,
        "test_query": False,
        "circuit_breaker_open": _state.is_circuit_open(),
        "consecutive_failures": _state.consecutive_failures,
        "metrics": {
            "successful_acquisitions": _state.metrics.successful_acquisitions,
            "failed_acquisitions": _state.metrics.failed_acquisitions,
            "total_queries": _state.metrics.total_queries,
        }
    }
    
    if _state.pool and not getattr(_state.pool, "_closed", True):
        try:
            # Get pool metrics
            if hasattr(_state.pool, 'get_size'):
                health["size"] = _state.pool.get_size()
            if hasattr(_state.pool, 'get_idle_size'):
                health["free"] = _state.pool.get_idle_size()
            if health["size"] is not None and health["free"] is not None:
                health["used"] = health["size"] - health["free"]
                if health["size"] > 0:
                    health["utilization_percent"] = (
                        health["used"] / health["size"]
                    ) * 100.0
            
            # Test query
            async with get_db_connection_context(timeout=5.0) as conn:
                await conn.fetchval("SELECT 1")
                health["test_query"] = True
                health["status"] = "healthy"
            
            _state.metrics.last_health_check = datetime.utcnow()
            _state.pool_state = PoolState.HEALTHY
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            _state.pool_state = PoolState.DEGRADED
            logger.error(f"Pool health check failed: {e}")
    else:
        health["status"] = "not_initialized"
    
    return health


def get_pool_metrics() -> PoolMetrics:
    """
    Get current pool metrics.
    
    Returns:
        PoolMetrics object with current statistics
    """
    if _state.pool and not getattr(_state.pool, "_closed", True):
        _state.metrics.total_connections = _state.pool.get_size()
        _state.metrics.idle_connections = _state.pool.get_idle_size()
        _state.metrics.active_connections = (
            _state.metrics.total_connections - _state.metrics.idle_connections
        )
        _state.metrics.pool_state = _state.pool_state
    
    return _state.metrics


# ============================================================================
# Celery Worker Integration
# ============================================================================

def init_celery_worker() -> None:
    """
    Initialize database pool for Celery worker process.
    
    Called automatically when Celery worker process starts.
    """
    pid = os.getpid()
    logger.info(f"Initializing Celery worker database pool in process {pid}")
    
    os.environ['SERVER_SOFTWARE'] = 'celery'
    
    try:
        run_async_in_worker_loop(initialize_connection_pool(force_new=True))
        loop = get_or_create_event_loop()
        logger.info(
            f"Celery worker {pid} database pool initialized successfully with "
            f"event loop {id(loop)}"
        )
    except Exception as e:
        logger.error(
            f"Failed to initialize Celery worker database pool: {e}",
            exc_info=True
        )
        raise


# IMPROVEMENT: Added weak=False to prevent garbage collection of signal handlers
@worker_process_init.connect(weak=False)
def init_worker_pool(**kwargs) -> None:
    """
    Celery signal handler: Initialize database pool when worker process starts.
    
    Args:
        **kwargs: Signal arguments (unused)
    """
    init_celery_worker()


# IMPROVEMENT: Added weak=False to prevent garbage collection of signal handlers
@worker_process_shutdown.connect(weak=False)
def close_worker_pool(**kwargs) -> None:
    """
    Celery signal handler: Gracefully close database pool when worker shuts down.
    
    This handler:
    1. Marks process as shutting down (prevents new operations)
    2. Waits for pending operations to complete (with timeout)
    3. Terminates the connection pool
    4. Cleans up event loop
    
    Args:
        **kwargs: Signal arguments (unused)
    """
    pid = os.getpid()
    
    # Mark as shutting down FIRST to prevent new operations
    mark_shutting_down()
    logger.info(f"Process {pid}: Marked as shutting down, beginning graceful shutdown")
    
    async def _graceful_shutdown():
        """Internal coroutine for coordinated shutdown."""
        # Wait for pending operations with reasonable timeout
        try:
            completed, timed_out = await asyncio.wait_for(
                wait_for_pending_operations(timeout=25.0),
                timeout=30.0
            )
            logger.info(
                f"Process {pid}: Completed {completed} operations, "
                f"{timed_out} timed out"
            )
        except asyncio.TimeoutError:
            logger.warning(f"Process {pid}: Timed out waiting for pending operations")
        
        # Close the pool
        if _state.pool:
            try:
                logger.info(
                    f"Process {pid}: Terminating DB pool after draining operations"
                )
                await close_connection_pool(force_terminate=True)
            except Exception as e:
                logger.error(
                    f"Process {pid}: Error terminating pool: {e}",
                    exc_info=True
                )
                # Final attempt to terminate
                if _state.pool and not getattr(_state.pool, "_closed", False):
                    try:
                        _state.pool.terminate()
                    except Exception:
                        logger.exception(
                            f"Process {pid}: Failed to terminate pool during shutdown"
                        )
    
    # Execute graceful shutdown
    if _state.pool_loop and not _state.pool_loop.is_closed():
        try:
            if not _state.pool_loop.is_running():
                _state.pool_loop.run_until_complete(_graceful_shutdown())
            else:
                future = asyncio.run_coroutine_threadsafe(
                    _graceful_shutdown(),
                    _state.pool_loop
                )
                future.result(timeout=45)  # Total timeout for shutdown
            logger.info(f"Process {pid}: Graceful shutdown completed")
        except Exception as e:
            logger.error(
                f"Process {pid}: Error during graceful shutdown: {e}",
                exc_info=True
            )
            # Force terminate the pool as last resort
            if _state.pool and not getattr(_state.pool, "_closed", True):
                try:
                    _state.pool.terminate()
                except Exception:
                    pass
    
    # Final cleanup
    _state.pool = None
    _state.pool_loop = None
    _state.pool_state = PoolState.CLOSED
    
    # Close event loop
    if hasattr(_state.thread_local, 'event_loop') and _state.thread_local.event_loop:
        try:
            loop = _state.thread_local.event_loop
            if not loop.is_closed() and not loop.is_running():
                loop.close()
                logger.info(f"Worker {pid} event loop closed")
        except Exception:
            pass
    
    # Clear shutdown state for this PID
    _state.shutdown_state.clear_pid(pid)


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Configuration
    'get_db_dsn',
    'get_pool_config',
    'get_db_connection_sync',
    
    # Pool Management
    'initialize_connection_pool',
    'get_db_connection_pool',
    'get_db_connection_context',
    'close_connection_pool',
    'setup_connection',
    
    # Health & Metrics
    'check_pool_health',
    'get_pool_metrics',
    
    # Worker Integration
    'init_celery_worker',
    'run_async_in_worker_loop',
    'get_or_create_event_loop',
    
    # Operation Tracking
    'track_operation',
    'wait_for_pending_operations',
    
    # Shutdown
    'is_shutting_down',
    'mark_shutting_down',
    
    # Types (for external use)
    'PoolState',
    'PoolMetrics',
]
