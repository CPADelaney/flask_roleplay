# memory/connection.py

import asyncio
import logging
import asyncpg
from typing import Optional, Dict, Any
import time
import os

# Import config
from config import DB_CONFIG

logger = logging.getLogger("memory_db")

DB_CONFIG = os.getenv("DB_DSN")

class DBConnectionManager:
    """
    Manages database connections with a connection pool.
    Provides context managers for transaction management.
    """
    _pool: Optional[asyncpg.Pool] = None
    _lock = asyncio.Lock()
    _metrics: Dict[str, Any] = {
        "connections_created": 0,
        "connections_released": 0,
        "active_connections": 0,
        "peak_connections": 0,
        "transactions_started": 0,
        "transactions_committed": 0,
        "transactions_rolled_back": 0,
        "queries_executed": 0
    }
    
    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """
        Get or create the connection pool.
        Uses double-checked locking pattern for thread safety.
        """
        if cls._pool is None:
            async with cls._lock:
                if cls._pool is None:
                    try:
                        # Custom event handlers to track metrics
                        def setup_connection(conn):
                            # Allow client cancelation in case of query timeout
                            conn.add_termination_listener(cls._on_connection_terminated)
                            cls._metrics["connections_created"] += 1
                            cls._metrics["active_connections"] += 1
                            cls._metrics["peak_connections"] = max(
                                cls._metrics["peak_connections"], 
                                cls._metrics["active_connections"]
                            )
                            return conn
                            
                        # Create the connection pool
                        cls._pool = await asyncpg.create_pool(
                            dsn=DB_CONFIG["dsn"],
                            min_size=DB_CONFIG.get("min_connections", 5),
                            max_size=DB_CONFIG.get("max_connections", 20),
                            command_timeout=DB_CONFIG.get("command_timeout", 60),
                            statement_cache_size=DB_CONFIG.get("statement_cache_size", 0),
                            max_inactive_connection_lifetime=DB_CONFIG.get("max_inactive_connection_lifetime", 300),
                            setup=setup_connection
                        )
                        logger.info(f"Database connection pool created with {DB_CONFIG.get('min_connections', 5)}-{DB_CONFIG.get('max_connections', 20)} connections")
                    except Exception as e:
                        logger.critical(f"Failed to create database connection pool: {e}")
                        raise
        return cls._pool
    
    @classmethod
    async def close_pool(cls) -> None:
        """
        Close the connection pool.
        Should be called during application shutdown.
        """
        async with cls._lock:
            if cls._pool is not None:
                await cls._pool.close()
                cls._pool = None
                logger.info("Database connection pool closed")
                
    @classmethod
    def _on_connection_terminated(cls, conn):
        """Callback when a connection is terminated."""
        cls._metrics["connections_released"] += 1
        cls._metrics["active_connections"] -= 1
    
    @classmethod
    async def acquire(cls) -> asyncpg.Connection:
        """
        Acquire a connection from the pool.
        """
        pool = await cls.get_pool()
        return await pool.acquire()
    
    @classmethod
    async def release(cls, connection: asyncpg.Connection) -> None:
        """
        Release a connection back to the pool.
        """
        if cls._pool is None:
            logger.warning("Attempting to release connection to closed pool")
            return
            
        await cls._pool.release(connection)
        cls._metrics["connections_released"] += 1
        cls._metrics["active_connections"] -= 1
    
    @classmethod
    async def get_metrics(cls) -> Dict[str, Any]:
        """
        Get connection pool metrics.
        """
        pool_stats = {}
        if cls._pool:
            pool_stats = {
                "min_size": cls._pool._minsize,
                "max_size": cls._pool._maxsize,
                "size": len(cls._pool._holders),
                "free": cls._pool._queue.qsize()
            }
            
        return {
            **cls._metrics,
            "pool": pool_stats
        }
        
    @classmethod
    def track_query(cls):
        """Track a query execution."""
        cls._metrics["queries_executed"] += 1
        
    @classmethod
    def track_transaction_start(cls):
        """Track transaction start."""
        cls._metrics["transactions_started"] += 1
        
    @classmethod
    def track_transaction_commit(cls):
        """Track transaction commit."""
        cls._metrics["transactions_committed"] += 1
        
    @classmethod
    def track_transaction_rollback(cls):
        """Track transaction rollback."""
        cls._metrics["transactions_rolled_back"] += 1


class Transaction:
    """
    Context manager for transaction handling.
    Ensures proper transaction lifecycle and error handling.
    """
    def __init__(self, conn: asyncpg.Connection, readonly: bool = False):
        self.conn = conn
        self.tx = None
        self.readonly = readonly
        
    async def __aenter__(self):
        # Start transaction
        self.tx = self.conn.transaction(readonly=self.readonly)
        await self.tx.start()
        DBConnectionManager.track_transaction_start()
        return self.tx
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred, rollback
            await self.tx.rollback()
            DBConnectionManager.track_transaction_rollback()
            logger.debug(f"Transaction rolled back due to: {exc_val}")
            return False
        else:
            # No exception, commit
            await self.tx.commit()
            DBConnectionManager.track_transaction_commit()
            return True


class ConnectionContext:
    """
    Context manager for database connections.
    Automatically acquires and releases connections.
    """
    def __init__(self, readonly: bool = False):
        self.conn = None
        self.readonly = readonly
        
    async def __aenter__(self):
        self.conn = await DBConnectionManager.acquire()
        return self.conn
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await DBConnectionManager.release(self.conn)


class TransactionContext:
    """
    Combined context manager for connection and transaction.
    Automatically handles connection acquisition, transaction lifecycle, and proper cleanup.
    """
    def __init__(self, readonly: bool = False):
        self.conn = None
        self.tx = None
        self.readonly = readonly
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        self.conn = await DBConnectionManager.acquire()
        self.tx = self.conn.transaction(readonly=self.readonly)
        await self.tx.start()
        DBConnectionManager.track_transaction_start()
        return self.conn
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                # Exception occurred, rollback
                await self.tx.rollback()
                DBConnectionManager.track_transaction_rollback()
                logger.debug(f"Transaction rolled back due to: {exc_val}")
            else:
                # No exception, commit
                await self.tx.commit()
                DBConnectionManager.track_transaction_commit()
                
                # Log long-running transactions
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:  # More than 1 second
                    logger.warning(f"Long transaction detected: {elapsed:.2f}s")
        finally:
            # Always release the connection
            await DBConnectionManager.release(self.conn)


# Simplified helpers for common operations
async def execute_query(query: str, *args, **kwargs) -> Any:
    """
    Execute a query and return results.
    Handles connection acquisition and release.
    """
    conn = None
    try:
        conn = await DBConnectionManager.acquire()
        DBConnectionManager.track_query()
        return await conn.fetch(query, *args, **kwargs)
    finally:
        if conn:
            await DBConnectionManager.release(conn)


async def execute_transaction(func, *args, readonly: bool = False, **kwargs) -> Any:
    """
    Execute a function within a transaction.
    
    Args:
        func: Async function that receives connection as first argument
        *args: Arguments to pass to func
        readonly: Whether transaction is readonly
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result of func
    """
    async with TransactionContext(readonly=readonly) as conn:
        return await func(conn, *args, **kwargs)
