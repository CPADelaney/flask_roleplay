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
import contextlib
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Optional, Dict, Any, Set, List, Tuple, Union
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

AsyncpgConnection = Union[asyncpg.Connection, '_ResilientConnectionWrapper']


# Use a single, consistent name for the ContextVar
_SKIP_VECTOR_REGISTRATION: ContextVar[bool] = ContextVar(
    "skip_vector_registration",
    default=False,
)

# Truthy/falsey parsing for env var
_VECTOR_TRUTHY = {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}
_VECTOR_FALSEY = {"0", "false", "f", "no", "n", "off", "disable", "disabled", ""}

def get_register_vector() -> bool:
    """
    Return whether pgvector registration should run for new connections.
    Defaults to False if DB_REGISTER_VECTOR is unset or unrecognized.
    """
    raw_value = os.getenv("DB_REGISTER_VECTOR")
    if raw_value is None:
        return False
    value = raw_value.strip().lower()
    if value in _VECTOR_TRUTHY:
        return True
    if value in _VECTOR_FALSEY:
        return False
    logger.warning("Unrecognized DB_REGISTER_VECTOR value '%s'; defaulting to False", raw_value)
    return False

@contextmanager
def skip_vector_registration() -> None:
    """
    Temporarily skip pgvector codec registration for any new connection acquired within this block.
    """
    token = _SKIP_VECTOR_REGISTRATION.set(True)
    try:
        yield
    finally:
        _SKIP_VECTOR_REGISTRATION.reset(token)

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
        self.pool_warmed: bool = False
        
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

# Single persistent event loop for Celery worker processes. We pin all
# database work to this loop to avoid pool thrashing caused by cross-loop
# access patterns inside prefork workers.
WORKER_LOOP: Optional[asyncio.AbstractEventLoop] = None

# ============================================================================
# Configuration Management
# ============================================================================

# Default pool configuration
DEFAULT_MIN_CONNECTIONS = 4
DEFAULT_MAX_CONNECTIONS = 20
DEFAULT_CONNECTION_LIFETIME = 900  # seconds
DEFAULT_COMMAND_TIMEOUT = 120  # seconds
DEFAULT_MAX_QUERIES = 50000
# Default values; env DB_SETUP_TIMEOUT can override
DEFAULT_RELEASE_TIMEOUT = DEFAULT_COMMAND_TIMEOUT
DEFAULT_SETUP_TIMEOUT = 30.0  # Timeout for connection setup (pgvector registration, 30s)
DEFAULT_ACQUIRE_TIMEOUT = 120.0  # Timeout for acquiring connection from pool
DEFAULT_POOL_CREATE_TIMEOUT = 30.0  # Timeout for initial asyncpg.create_pool wait_for


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


def get_setup_timeout() -> float:
    """
    Get timeout for connection setup operations (like pgvector registration).
    
    Returns:
        Timeout in seconds
    """
    return float(os.getenv("DB_SETUP_TIMEOUT", str(DEFAULT_SETUP_TIMEOUT)))


def get_pool_create_timeout() -> float:
    """
    Get timeout for asyncpg.create_pool() gate.

    Returns:
        Timeout in seconds
    """
    return float(os.getenv("DB_POOL_CREATE_TIMEOUT", str(DEFAULT_POOL_CREATE_TIMEOUT)))


def get_acquire_timeout() -> float:
    """
    Get default timeout for acquiring connections from pool.
    
    Returns:
        Timeout in seconds
    """
    return float(os.getenv("DB_ACQUIRE_TIMEOUT", str(DEFAULT_ACQUIRE_TIMEOUT)))


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
    """Get the current event loop, preferring the persistent worker loop."""
    global WORKER_LOOP

    # If we've already created a worker loop for this process, always reuse it.
    if WORKER_LOOP and not WORKER_LOOP.is_closed():
        asyncio.set_event_loop(WORKER_LOOP)
        return WORKER_LOOP

    # Otherwise try to reuse the currently running loop if one exists.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and not loop.is_closed():
        WORKER_LOOP = loop
        asyncio.set_event_loop(WORKER_LOOP)
        return WORKER_LOOP

    # Create a new persistent worker loop and remember it.
    with _state.thread_local_lock:
        if WORKER_LOOP and not WORKER_LOOP.is_closed():
            asyncio.set_event_loop(WORKER_LOOP)
            return WORKER_LOOP

        loop = asyncio.new_event_loop()
        WORKER_LOOP = loop
        asyncio.set_event_loop(WORKER_LOOP)
        _state.thread_local.event_loop = WORKER_LOOP
        _state.reset_locks()
        logger.debug(
            "Created WORKER_LOOP %s for thread %s",
            id(WORKER_LOOP),
            threading.current_thread().name,
        )
        return WORKER_LOOP


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
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    attempt < max_attempts - 1
                    and (
                        "unknown protocol state" in msg
                        or "released back to the pool" in msg
                    )
                ):
                    attempt += 1
                    logger.warning(
                        "InternalClientError during %s; refreshing connection and retrying (%d/%d)",
                        name,
                        attempt,
                        max_attempts,
                    )
                    await self._refresh_connection()
                    method = getattr(self._conn, name)
                    continue
                raise
    
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
                except asyncpg.exceptions.InterfaceError as iface_err:
                    # Connection already released or in invalid state
                    error_msg = str(iface_err)
                    if "released back to the pool" in error_msg:
                        logger.debug(
                            f"Connection {old_conn_id} was already released during refresh"
                        )
                    else:
                        logger.warning(
                            f"InterfaceError releasing old connection {old_conn_id} "
                            f"during refresh: {iface_err}"
                        )
                        try:
                            old_conn.terminate()
                        except Exception:
                            pass  # Best effort
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

# ============================================================================
# setup_connection (merged behavior)
# ============================================================================

async def setup_connection(conn: asyncpg.Connection) -> None:
    """
    Per-connection setup invoked by the pool.
    Honors:
      - DB_REGISTER_VECTOR env to enable/disable pgvector codec registration
      - skip_vector_registration() context manager to temporarily suppress it
      - One-time registration per connection with retry/timeout
    """

    # Respect global env toggle first
    if not get_register_vector():
        logger.debug("Skipping pgvector registration on %s (env disabled)", id(conn))
        return

    # Respect per-scope override (read-heavy or latency-sensitive paths)
    if _SKIP_VECTOR_REGISTRATION.get():
        logger.debug("Skipping pgvector registration on %s (context override)", id(conn))
        return

    # Pull timeouts/retries from helpers if present; otherwise environment defaults
    setup_timeout = get_setup_timeout()

    try:
        max_retries = get_vector_register_retries()
    except NameError:
        max_retries = int(os.getenv("DB_VECTOR_REGISTER_RETRIES", "3"))

    try:
        initial_retry_delay = get_vector_register_retry_delay()
    except NameError:
        initial_retry_delay = float(os.getenv("DB_VECTOR_REGISTER_RETRY_DELAY", "0.2"))

    await _register_vector_with_retry(
        conn,
        setup_timeout=setup_timeout,
        max_retries=max_retries,
        initial_retry_delay=initial_retry_delay,
    )
# ============================================================================
# Retry helper (from main), with small safety polish
# ============================================================================

async def _register_vector_with_retry(
    conn: asyncpg.Connection,
    *,
    setup_timeout: float,
    max_retries: int,
    initial_retry_delay: float,
) -> None:
    """
    Register pgvector codec on the given connection with timeout and backoff.
    Idempotent: safe to call multiple times.
    """
    retry_delay = initial_retry_delay
    last_exception: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            await asyncio.wait_for(
                pgvector_asyncpg.register_vector(conn),
                timeout=setup_timeout,
            )
            logger.debug(
                "pgvector registered on connection %s (attempt %d/%d)",
                id(conn),
                attempt,
                max_retries,
            )
            return
        except asyncio.CancelledError:
            # Do not leave a half-initialized connection in the pool
            with contextlib.suppress(Exception):
                await conn.close()
            raise
        except asyncpg.exceptions.ConnectionDoesNotExistError as exc:
            logger.warning(
                "pgvector registration lost connection %s (attempt %d/%d): %s",
                id(conn),
                attempt,
                max_retries,
                exc,
            )
            last_exception = exc
        except asyncio.TimeoutError as exc:
            logger.warning(
                "pgvector registration timed out on connection %s (attempt %d/%d, timeout=%.1fs)",
                id(conn),
                attempt,
                max_retries,
                setup_timeout,
            )
            last_exception = exc
        except asyncpg.exceptions.UndefinedFunctionError as exc:
            logger.warning(
                "pgvector extension not available on connection %s: %s",
                id(conn),
                exc,
            )
            return
        except Exception as exc:
            logger.warning(
                "pgvector registration failed on connection %s (attempt %d/%d): %s",
                id(conn),
                attempt,
                max_retries,
                exc,
            )
            with contextlib.suppress(Exception):
                await conn.close()
            raise

        if attempt < max_retries:
            # Linear backoff between retry attempts
            await asyncio.sleep(retry_delay)
            retry_delay += initial_retry_delay
        else:
            break

    logger.error(
        "pgvector registration failed on connection %s after %d attempts; closing connection",
        id(conn),
        max_retries,
    )
    with contextlib.suppress(Exception):
        await conn.close()

    if isinstance(last_exception, asyncio.TimeoutError):
        raise TimeoutError("pgvector registration failed after retries") from last_exception
    if last_exception is not None:
        raise last_exception

    raise TimeoutError("pgvector registration failed after retries")




async def create_pool_with_retry(
    dsn: str,
    config: Dict[str, int],
    max_retries: int = 3
) -> asyncpg.Pool:
    """
    Create connection pool with exponential backoff retry and TCP keepalives.
    
    Args:
        dsn: Database connection string
        config: Pool configuration dictionary
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
                    server_settings=server_settings
                ),
                timeout=get_pool_create_timeout()
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


def _clear_global_pool_reference(pool_to_close: asyncpg.Pool) -> None:
    """Reset global pool reference and metrics when a pool is closed."""
    if pool_to_close is _state.pool:
        _state.pool = None
        _state.pool_loop = None
        _state.pool_state = PoolState.CLOSED
        _state.metrics.pool_state = PoolState.CLOSED
        _state.metrics.total_connections = 0
        _state.metrics.idle_connections = 0
        _state.metrics.active_connections = 0
        _state.pool_warmed = False


async def close_existing_pool(pool: Optional[asyncpg.Pool] = None) -> None:
    """
    Close an existing connection pool safely.

    Args:
        pool: Pool to close. If None, closes the global pool.
    """
    pool_to_close = pool or _state.pool

    if pool_to_close is None:
        return

    if getattr(pool_to_close, "_closed", False):
        _clear_global_pool_reference(pool_to_close)
        return

    shutdown_succeeded = False

    owning_loop: Optional[asyncio.AbstractEventLoop] = None
    if pool_to_close is _state.pool:
        owning_loop = _state.pool_loop
    else:
        owning_loop = getattr(pool_to_close, "_loop", None)

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if owning_loop and current_loop is not owning_loop:
        if not owning_loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    close_existing_pool(pool_to_close),
                    owning_loop
                )
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=10.0
                )
                return
            except (RuntimeError, asyncio.TimeoutError) as delegate_error:
                logger.warning(
                    "Delegating pool close to owning loop failed: %s",
                    delegate_error
                )
            except Exception as delegate_error:
                logger.error(
                    "Unexpected error delegating pool close to owning loop: %s",
                    delegate_error,
                    exc_info=True
                )
        # Owning loop is closed or delegation failed; terminate synchronously.
        logger.info(
            "Owning loop %s unavailable; terminating pool from loop %s",
            id(owning_loop),
            id(current_loop) if current_loop else None
        )
        try:
            pool_to_close.terminate()
            shutdown_succeeded = True
            if hasattr(pool_to_close, "_closed"):
                pool_to_close._closed = True
            logger.info("Pool terminated successfully after owning loop shutdown")
        except RuntimeError as terminate_error:
            if "Event loop is closed" in str(terminate_error):
                logger.info(
                    "Pool terminate raised RuntimeError due to closed loop; treating as closed"
                )
                shutdown_succeeded = True
                if hasattr(pool_to_close, "_closed"):
                    pool_to_close._closed = True
            else:
                logger.error(
                    "Pool terminate failed after loop shutdown: %s",
                    terminate_error,
                    exc_info=True
                )
        except Exception as terminate_error:
            logger.error(
                "Pool terminate failed after loop shutdown: %s",
                terminate_error,
                exc_info=True
            )
        if shutdown_succeeded:
            _clear_global_pool_reference(pool_to_close)
        return

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
        shutdown_succeeded = True
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            logger.warning(
                "Pool close failed because event loop is closed; attempting terminate"
            )
            try:
                pool_to_close.terminate()
                shutdown_succeeded = True
                if hasattr(pool_to_close, "_closed"):
                    pool_to_close._closed = True
                logger.info("Pool terminated successfully after loop closure")
            except RuntimeError as terminate_error:
                if "Event loop is closed" in str(terminate_error):
                    logger.info(
                        "Pool terminate raised RuntimeError due to closed loop; treating as closed"
                    )
                    shutdown_succeeded = True
                    if hasattr(pool_to_close, "_closed"):
                        pool_to_close._closed = True
                else:
                    logger.error(
                        "Pool terminate failed after loop closure: %s",
                        terminate_error,
                        exc_info=True
                    )
            except Exception as terminate_error:
                logger.error(
                    "Pool terminate failed after loop closure: %s",
                    terminate_error,
                    exc_info=True
                )
        else:
            logger.error(f"Error closing existing pool: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error closing existing pool: {e}", exc_info=True)

    if shutdown_succeeded:
        _clear_global_pool_reference(pool_to_close)
    elif pool_to_close is _state.pool:
        logger.warning(
            "Pool shutdown did not complete successfully; retaining global reference"
        )


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
    
    # Always bind DB work to the worker loop
    current_loop = get_or_create_event_loop()

    if _state.pool and _state.pool_loop and _state.pool_loop is not current_loop:
        asyncio.set_event_loop(_state.pool_loop)
        current_loop = _state.pool_loop

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
        
        # Check if pool exists for different event loop. Prefer reusing the
        # worker loop rather than tearing the pool down.
        if _state.pool is not None and _state.pool_loop is not current_loop:
            logger.warning(
                "Pool exists for different event loop (current=%s, pool=%s); switching to worker loop",
                id(current_loop),
                id(_state.pool_loop),
            )
            if _state.pool_loop and not _state.pool_loop.is_closed():
                asyncio.set_event_loop(_state.pool_loop)
                current_loop = _state.pool_loop
        
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
            local_pool = await create_pool_with_retry(dsn, config)
            
            _state.pool = local_pool
            _state.pool_loop = current_loop
            _state.pool_state = PoolState.HEALTHY
            _state.pool_warmed = False

            # Update metrics
            _state.metrics.pool_state = PoolState.HEALTHY
            _state.metrics.total_connections = local_pool.get_size()
            _state.metrics.idle_connections = local_pool.get_idle_size()
            
            if app:
                app.db_pool = _state.pool
                logger.info(f"Process {os.getpid()}: DB_POOL stored on app.db_pool")
            
            logger.info(f"Process {os.getpid()}: Asyncpg pool initialized successfully")

            # Optional eager warm-up after init (pre-creates connections & runs pgvector registration)
            try:
                warm_at_init = os.getenv("DB_POOL_WARM_AT_INIT", "true").strip().lower() in {"1", "true", "yes", "on"}
                if warm_at_init and _state.pool and not getattr(_state.pool, "_closed", False):
                    # Default to a small-but-useful warm size; clamp to pool max
                    try:
                        max_size = _state.pool.get_max_size() if hasattr(_state.pool, "get_max_size") else get_pool_config()["max_size"]
                    except Exception:
                        max_size = get_pool_config()["max_size"]
                    try:
                        warm_size = int(os.getenv("DB_POOL_WARM_SIZE", "8"))
                    except Exception:
                        warm_size = 8
                    warm_size = max(1, min(warm_size, max_size))
                    # Use a conservative per-connection timeout; pgvector setup timeout is handled inside setup_connection
                    logger.info(f"Process {os.getpid()}: Warming DB pool (test_count={warm_size})")
                    try:
                        warm_results = await warm_connection_pool(test_count=warm_size, timeout=10.0)
                        if warm_results.get("success"):
                            _state.pool_warmed = True
                        else:
                            _state.pool_warmed = False
                    except Exception as warm_err:
                        _state.pool_warmed = False
                        logger.warning(f"Pool warm-up skipped due to error: {warm_err}")
            except Exception as warm_wrap_err:
                logger.debug(f"Warm-up guard swallowed error: {warm_wrap_err}")

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


async def warm_connection_pool(test_count: int = 5, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Warm up the connection pool by acquiring and testing connections.
    
    This is useful for:
    - Early detection of connection/setup issues
    - Pre-warming connections before production load
    - Validating pool configuration
    
    Args:
        test_count: Number of connections to test (up to pool max_size)
        timeout: Timeout for each connection test
        
    Returns:
        Dictionary with warming results and diagnostics
    """
    results = {
        "success": False,
        "tested": 0,
        "succeeded": 0,
        "failed": 0,
        "errors": [],
        "avg_acquire_time": 0.0,
        "pool_state": None
    }
    
    pool = await get_db_connection_pool()
    max_size = pool.get_max_size() if hasattr(pool, 'get_max_size') else test_count
    test_count = min(test_count, max_size)
    
    logger.info(f"Warming connection pool with {test_count} test connections")
    
    acquire_times = []
    
    for i in range(test_count):
        try:
            start_time = asyncio.get_event_loop().time()
            async with get_db_connection_context(timeout=timeout) as conn:
                await conn.fetchval("SELECT 1")
            acquire_time = asyncio.get_event_loop().time() - start_time
            acquire_times.append(acquire_time)
            
            results["tested"] += 1
            results["succeeded"] += 1
            logger.debug(f"Pool warm-up: connection {i+1}/{test_count} OK ({acquire_time:.3f}s)")
        except Exception as e:
            results["tested"] += 1
            results["failed"] += 1
            results["errors"].append(f"Connection {i+1}: {str(e)}")
            logger.error(f"Pool warm-up: connection {i+1}/{test_count} failed: {e}")
    
    if acquire_times:
        results["avg_acquire_time"] = sum(acquire_times) / len(acquire_times)
    
    results["success"] = results["failed"] == 0
    results["pool_state"] = {
        "size": pool.get_size(),
        "idle": pool.get_idle_size(),
        "in_use": pool.get_size() - pool.get_idle_size()
    }
    
    logger.info(
        f"Pool warm-up complete: {results['succeeded']}/{results['tested']} succeeded, "
        f"avg time: {results['avg_acquire_time']:.3f}s"
    )
    
    return results


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

    if _state.pool and _state.pool_loop and _state.pool_loop is not current_loop:
        asyncio.set_event_loop(_state.pool_loop)
        current_loop = _state.pool_loop

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
    timeout: Optional[float] = None,
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
        timeout: Maximum time to wait for connection acquisition.
                If None, uses DB_ACQUIRE_TIMEOUT env var (default 120s)
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
    # helper: single acquire with timeout
    # Use configurable default timeout if not specified
    if timeout is None:
        timeout = get_acquire_timeout()
    # Early shutdown check
    if is_shutting_down():
        raise ConnectionError(
            f"Cannot acquire connection - worker process {os.getpid()} is shutting down"
        )
    
    current_loop = get_or_create_event_loop()
    conn: Optional[asyncpg.Connection] = None
    conn_id: Optional[int] = None
    wrapped_conn: Optional[_ResilientConnectionWrapper] = None
    connection_already_released = False  # Track if we manually released the connection
    
    # Verify pool is for current loop
    if _state.pool and _state.pool_loop and _state.pool_loop is not current_loop:
        asyncio.set_event_loop(_state.pool_loop)
        current_loop = _state.pool_loop

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
    
    async def _acquire_with_timeout(pool, t: float) -> asyncpg.Connection:
        return await asyncio.wait_for(pool.acquire(), timeout=t)

    try:
        logger.debug(f"Acquiring connection from pool (timeout={timeout}s)")
        try:
            conn = await _acquire_with_timeout(current_pool_to_use, timeout)
        except asyncpg.exceptions.InterfaceError as iface_err:
            if "pool is closing" in str(iface_err).lower():
                logger.warning("Pool was closing during acquire; reinitializing and retrying once")
                await initialize_connection_pool(force_new=True)
                current_pool_to_use = _state.pool
                conn = await _acquire_with_timeout(current_pool_to_use, timeout)
            else:
                raise
        except asyncio.TimeoutError:
            # Enhanced diagnostics for timeout errors + one-shot recovery
            pool_size = current_pool_to_use.get_size() if current_pool_to_use else 0
            pool_free = current_pool_to_use.get_idle_size() if current_pool_to_use else 0
            pool_used = pool_size - pool_free
            logger.error(
                f"Timeout ({timeout}s) acquiring DB connection from pool. "
                f"Pool stats: size={pool_size}, free={pool_free}, in_use={pool_used}. "
                f"Attempting one-shot pool expiry and quick retry (5s)…"
            )
            try:
                expire = getattr(current_pool_to_use, "expire_connections", None)
                if callable(expire):
                    res = expire()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as exp_err:
                logger.warning("expire_connections failed (continuing): %s", exp_err)
            # quick retry
            conn = await _acquire_with_timeout(current_pool_to_use, 5.0)
        
        # Re-check shutdown status after acquiring (race condition protection)
        if is_shutting_down():
            await current_pool_to_use.release(conn)
            connection_already_released = True  # Mark as released
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
            await asyncio.wait_for(
                asyncio.gather(*pending_ops, return_exceptions=True),
                timeout=5.0
            )
        
    except asyncio.TimeoutError:
        # Enhanced diagnostics for timeout errors
        pool_size = current_pool_to_use.get_size() if current_pool_to_use else 0
        pool_free = current_pool_to_use.get_idle_size() if current_pool_to_use else 0
        pool_used = pool_size - pool_free
        
        logger.error(
            f"Timeout ({timeout}s) acquiring DB connection from pool. "
            f"Pool stats: size={pool_size}, free={pool_free}, in_use={pool_used}. "
            f"This may indicate: (1) All connections are busy, (2) Connection setup is slow, "
            f"(3) Database is overloaded. Consider: (1) Increasing DB_POOL_MAX_SIZE, "
            f"(2) Increasing DB_ACQUIRE_TIMEOUT (current: {timeout}s), "
            f"(3) Increasing DB_SETUP_TIMEOUT (current: {get_setup_timeout()}s)"
        )
        _state.metrics.failed_acquisitions += 1
        # No retry here; one-shot retry already attempted above.
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
        # CRITICAL: This block must handle all possible error states during shutdown
        try:
            lock = _state.get_connection_ops_lock()
            active_conn_obj: Optional[asyncpg.Connection] = None
            active_conn_id = None
            
            # Safely extract connection reference
            # During shutdown, even property access can fail
            try:
                if wrapped_conn is not None:
                    active_conn_obj = wrapped_conn.raw_connection
                    active_conn_id = wrapped_conn.conn_id
                else:
                    active_conn_obj = conn
                    active_conn_id = conn_id
            except Exception as extract_err:
                logger.debug(
                    f"Could not extract connection reference during cleanup: {extract_err}"
                )
                # Continue with best effort - use what we have
                active_conn_obj = conn
                active_conn_id = conn_id
            
            # Remove connection from tracking (prevents memory leak)
            if active_conn_id is not None:
                try:
                    async with lock:
                        _state.connection_pending_ops.pop(active_conn_id, None)
                except Exception as tracking_err:
                    logger.debug(
                        f"Could not remove connection tracking for {active_conn_id}: {tracking_err}"
                    )
            
            # Release connection back to pool (if not already released)
            if active_conn_obj and current_pool_to_use and not connection_already_released:
                try:
                    # Try to release the connection
                    release_timeout = get_release_timeout()
                    await current_pool_to_use.release(
                        active_conn_obj,
                        timeout=release_timeout
                    )
                    logger.debug(f"Connection {active_conn_id} released successfully")
                except asyncpg.exceptions.InterfaceError as iface_err:
                    # Connection was already released or is in invalid state
                    # This is expected during shutdown or after errors
                    error_msg = str(iface_err)
                    if "released back to the pool" in error_msg:
                        logger.debug(
                            f"Connection {active_conn_id} was already released to pool"
                        )
                    elif "connection is closed" in error_msg or "connection has been released" in error_msg:
                        logger.debug(
                            f"Connection {active_conn_id} is already closed/released"
                        )
                    else:
                        logger.debug(
                            f"InterfaceError releasing connection {active_conn_id}: {iface_err}"
                        )
                except asyncpg.exceptions.PostgresConnectionError as conn_err:
                    # Connection lost - try to terminate
                    logger.debug(
                        f"Connection {active_conn_id} lost during release: {conn_err}"
                    )
                    try:
                        if hasattr(active_conn_obj, 'terminate'):
                            active_conn_obj.terminate()
                    except Exception:
                        pass  # Best effort termination
                except Exception as release_err:
                    # Other errors - log and try to terminate
                    logger.debug(
                        f"Error releasing connection {active_conn_id}: {release_err}. "
                        f"Attempting termination."
                    )
                    try:
                        if hasattr(active_conn_obj, 'terminate'):
                            active_conn_obj.terminate()
                    except Exception:
                        pass  # Best effort termination
            elif connection_already_released:
                logger.debug(
                    f"Skipping connection release for {active_conn_id} - already released manually"
                )
        except Exception as cleanup_err:
            # Absolute last resort - catch any error in the entire cleanup
            # This ensures the finally block never raises
            logger.debug(
                f"Error in connection cleanup finally block: {cleanup_err}. "
                f"This is likely due to shutdown race conditions."
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
        # Pin a single persistent loop for the worker
        global WORKER_LOOP
        if WORKER_LOOP is None or WORKER_LOOP.is_closed():
            WORKER_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(WORKER_LOOP)
        _state.reset_locks()
        _state.pool_loop = WORKER_LOOP
        WORKER_LOOP.run_until_complete(initialize_connection_pool(force_new=True))
        loop = WORKER_LOOP
        logger.info(
            f"Celery worker {pid} database pool initialized successfully with "
            f"event loop {id(loop)}"
        )
        # Proactively warm a few connections so pgvector registration never lands on the hot path
        warm_size = None
        try:
            warm_at_init = os.getenv("DB_POOL_WARM_AT_INIT", "true").strip().lower() in {"1", "true", "yes", "on"}
            if warm_at_init and not getattr(_state, "pool_warmed", False):
                warm_size = int(os.getenv("DB_POOL_WARM_SIZE", "8"))
                warm_size = max(1, warm_size)
                warm_results = WORKER_LOOP.run_until_complete(
                    warm_connection_pool(test_count=warm_size, timeout=10.0)
                )
                if warm_results.get("success"):
                    _state.pool_warmed = True
                else:
                    _state.pool_warmed = False
        except Exception as warm_err:
            _state.pool_warmed = False
            logger.warning(f"Pool warm-up at worker init failed (continuing): {warm_err}")

        try:
            health_result = WORKER_LOOP.run_until_complete(check_pool_health())
        except Exception as health_exc:
            logger.critical(
                "Fatal error running database pool health check during Celery worker init",
                exc_info=True,
            )
            raise

        status = health_result.get("status")
        if status != "healthy":
            warm_context = (
                f" after warming {warm_size} connections"
                if warm_size is not None
                else ""
            )
            logger.critical(
                "Database pool health check failed%s: %s",
                warm_context,
                health_result,
            )
            raise RuntimeError(
                "Celery worker database pool health check failed; aborting worker startup"
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
    global WORKER_LOOP
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
            # Always run shutdown on the worker loop we created.
            loop = _state.pool_loop
            if not loop.is_running():
                loop.run_until_complete(_graceful_shutdown())
            else:
                future = asyncio.run_coroutine_threadsafe(
                    _graceful_shutdown(), loop
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

    WORKER_LOOP = None
    
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
    'get_setup_timeout',
    'get_acquire_timeout',
    'get_release_timeout',
    
    # Pool Management
    'initialize_connection_pool',
    'warm_connection_pool',
    'get_db_connection_pool',
    'get_db_connection_context',
    'close_connection_pool',
    'setup_connection',
    'skip_vector_registration',
    'get_register_vector',

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
    'AsyncpgConnection',
]
