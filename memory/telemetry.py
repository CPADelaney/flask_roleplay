# memory/telemetry.py

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import json
from datetime import datetime, timedelta

# Lazy import game time functions to avoid circular imports
if TYPE_CHECKING:
    from logic.game_time_helper import get_game_datetime, get_game_iso_string

from memory.connection import get_connection_context

logger = logging.getLogger("memory_telemetry")

# Global lazy import cache
_game_time_module = None

def _lazy_import_game_time():
    """Lazy import game_time_helper to avoid circular dependencies."""
    global _game_time_module
    if _game_time_module is None:
        from logic import game_time_helper as _game_time_module
    return _game_time_module


class MemoryTelemetry:
    """
    Telemetry system for tracking memory operation performance.
    Collects metrics without blocking operations.
    """
    
    # Queue for background processing
    _queue = asyncio.Queue()
    _worker_task = None
    _lock = asyncio.Lock()
    
    # Aggregated metrics for real-time analysis
    _metrics = {
        "operations": {},
        "errors": {},
        "response_times": [],
        "total_operations": 0,
        "total_errors": 0
    }
    
    @classmethod
    async def record(cls,
                    user_id: int,
                    conversation_id: int,
                    operation: str,
                    success: bool = True,
                    duration: float = 0.0,
                    data_size: Optional[int] = None,
                    error: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a telemetry event for a memory operation.
        Queues the event for background processing.
        """
        # Lazy import and get game time
        game_time = _lazy_import_game_time()
        timestamp = await game_time.get_game_datetime(user_id, conversation_id)
        
        record = {
            "timestamp": timestamp,  # Store as datetime object
            "operation": operation,
            "success": success,
            "duration": duration,
            "data_size": data_size,
            "error": error,
            "metadata": metadata or {}
        }
        
        # Update real-time metrics
        async with cls._lock:
            cls._metrics["total_operations"] += 1
            
            # Track by operation type
            if operation not in cls._metrics["operations"]:
                cls._metrics["operations"][operation] = 0
            cls._metrics["operations"][operation] += 1
            
            # Track errors
            if not success:
                cls._metrics["total_errors"] += 1
                error_key = f"{operation}:{error}" if error else operation
                if error_key not in cls._metrics["errors"]:
                    cls._metrics["errors"][error_key] = 0
                cls._metrics["errors"][error_key] += 1
            
            # Track response times with rolling window
            cls._metrics["response_times"].append(duration)
            if len(cls._metrics["response_times"]) > 100:
                cls._metrics["response_times"].pop(0)
        
        # Add to background processing queue
        await cls._queue.put(record)
        
        # Ensure worker is running
        await cls.ensure_worker_running()
    
    @classmethod
    async def ensure_worker_running(cls) -> None:
        """Ensure the background worker task is running."""
        async with cls._lock:
            if cls._worker_task is None or cls._worker_task.done():
                cls._worker_task = asyncio.create_task(cls._background_worker())
    
    @classmethod
    async def _background_worker(cls) -> None:
        """
        Background worker that processes telemetry records from the queue.
        Batches records for more efficient database writes.
        """
        batch = []
        last_flush_time = time.time()
        
        logger.info("Telemetry background worker started")
        
        try:
            while True:
                try:
                    # Get item from queue with timeout
                    record = await asyncio.wait_for(cls._queue.get(), timeout=5.0)
                    batch.append(record)
                    cls._queue.task_done()
                    
                    # Flush batch if it's large enough or enough time has passed
                    current_time = time.time()
                    if len(batch) >= 20 or current_time - last_flush_time > 30:
                        await cls._flush_batch(batch)
                        batch = []
                        last_flush_time = current_time
                        
                except asyncio.TimeoutError:
                    # Timeout - check if we should flush batch
                    if batch:
                        await cls._flush_batch(batch)
                        batch = []
                    last_flush_time = time.time()
                    
                    # Check if queue is empty and has been for a while
                    if cls._queue.empty() and time.time() - last_flush_time > 60:
                        # No activity for a minute, exit worker
                        logger.info("Telemetry background worker shutting down (inactivity)")
                        break
                        
        except asyncio.CancelledError:
            logger.info("Telemetry background worker cancelled")
            # Flush any remaining records
            if batch:
                await cls._flush_batch(batch)
        except Exception as e:
            logger.error(f"Error in telemetry background worker: {e}")
            # Attempt to flush any remaining records
            if batch:
                try:
                    await cls._flush_batch(batch)
                except Exception as flush_e:
                    logger.error(f"Error flushing final batch: {flush_e}")
    
    @classmethod
    async def _flush_batch(cls, batch: List[Dict[str, Any]]) -> None:
        """
        Write a batch of telemetry records to the database.
        """
        if not batch:
            return
            
        try:
            # Get database connection using the proper context manager
            async with get_connection_context() as conn:
                # Insert records in batch
                await conn.executemany("""
                    INSERT INTO memory_telemetry 
                    (timestamp, operation, success, duration, data_size, error, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, [
                    (
                        record["timestamp"],  # Now this is a datetime object
                        record["operation"],
                        record["success"],
                        record["duration"],
                        record["data_size"],
                        record["error"],
                        json.dumps(record["metadata"])
                    )
                    for record in batch
                ])
                
                logger.debug(f"Flushed {len(batch)} telemetry records to database")
                
        except Exception as e:
            logger.error(f"Error flushing telemetry batch: {e}")
    
    @classmethod
    async def get_recent_metrics(cls, user_id: int, conversation_id: int, time_window_minutes: int = 15) -> Dict[str, Any]:
        """
        Get metrics for recent operations.
        """
        try:
            # Lazy import game time
            game_time = _lazy_import_game_time()
            
            # Get database connection using the proper context manager
            async with get_connection_context() as conn:
                cutoff = await game_time.get_game_datetime(user_id, conversation_id) - timedelta(minutes=time_window_minutes)
                
                # Get operation counts
                operation_rows = await conn.fetch("""
                    SELECT operation, COUNT(*) as count, 
                           AVG(duration) as avg_duration,
                           MIN(duration) as min_duration,
                           MAX(duration) as max_duration
                    FROM memory_telemetry
                    WHERE timestamp > $1
                    GROUP BY operation
                    ORDER BY count DESC
                """, cutoff)
                
                operations = {row["operation"]: {
                    "count": row["count"],
                    "avg_duration": row["avg_duration"],
                    "min_duration": row["min_duration"],
                    "max_duration": row["max_duration"]
                } for row in operation_rows}
                
                # Get error counts
                error_rows = await conn.fetch("""
                    SELECT operation, error, COUNT(*) as count
                    FROM memory_telemetry
                    WHERE timestamp > $1 AND success = FALSE
                    GROUP BY operation, error
                    ORDER BY count DESC
                """, cutoff)
                
                errors = {}
                for row in error_rows:
                    error_key = f"{row['operation']}:{row['error']}" if row["error"] else row["operation"]
                    errors[error_key] = row["count"]
                
                # Get overall stats
                stats_row = await conn.fetchrow("""
                    SELECT COUNT(*) as total_operations,
                           COUNT(*) FILTER (WHERE success = FALSE) as total_errors,
                           AVG(duration) as avg_duration,
                           PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration) as p95_duration,
                           PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration) as p99_duration
                    FROM memory_telemetry
                    WHERE timestamp > $1
                """, cutoff)
                
                # Combine with real-time metrics
                async with cls._lock:
                    # Deep copy current metrics to avoid modification during access
                    realtime_metrics = {
                        "operations": cls._metrics["operations"].copy(),
                        "errors": cls._metrics["errors"].copy(),
                        "total_operations": cls._metrics["total_operations"],
                        "total_errors": cls._metrics["total_errors"]
                    }
                    
                    # Calculate real-time response time stats
                    if cls._metrics["response_times"]:
                        times = sorted(cls._metrics["response_times"])
                        realtime_metrics["avg_duration"] = sum(times) / len(times)
                        realtime_metrics["p95_duration"] = times[int(len(times) * 0.95)]
                        realtime_metrics["p99_duration"] = times[int(len(times) * 0.99)]
                
                return {
                    "db_metrics": {
                        "operations": operations,
                        "errors": errors,
                        "total_operations": stats_row["total_operations"],
                        "total_errors": stats_row["total_errors"],
                        "avg_duration": stats_row["avg_duration"],
                        "p95_duration": stats_row["p95_duration"],
                        "p99_duration": stats_row["p99_duration"],
                        "time_window_minutes": time_window_minutes
                    },
                    "realtime_metrics": realtime_metrics
                }

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    @classmethod
    async def get_slow_operations(cls, threshold_ms: float = 500, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent slow operations exceeding the threshold.
        """
        try:
            async with get_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT timestamp, operation, duration, data_size, metadata
                      FROM memory_telemetry
                     WHERE duration > $1
                  ORDER BY duration DESC
                     LIMIT $2
                    """,
                    threshold_ms / 1000.0,
                    limit
                )

            result = []
            for row in rows:
                metadata = (
                    row["metadata"]
                    if isinstance(row["metadata"], dict)
                    else json.loads(row["metadata"] or "{}")
                )
                result.append({
                    "timestamp": row["timestamp"].isoformat(),
                    "operation": row["operation"],
                    "duration_ms": row["duration"] * 1000,
                    "data_size": row["data_size"],
                    "metadata": metadata
                })
            return result

        except Exception as e:
            logger.error(f"Error getting slow operations: {e}")
            return []

    
    @classmethod
    async def cleanup_old_telemetry(cls, user_id: int, conversation_id: int, days_to_keep: int = 30) -> int:
        """
        Clean up old telemetry data.
        """
        try:
            # Lazy import game time
            game_time = _lazy_import_game_time()
            
            async with get_connection_context() as conn:
                cutoff = await game_time.get_game_datetime(user_id, conversation_id) - timedelta(days=days_to_keep)
                result = await conn.execute(
                    """
                    DELETE FROM memory_telemetry
                     WHERE timestamp < $1
                    """,
                    cutoff
                )

            # result is something like "DELETE <n>"
            deleted = int(result.split(" ")[-1]) if " " in result else 0
            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up old telemetry: {e}")
            return 0
