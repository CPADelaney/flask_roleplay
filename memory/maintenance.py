# memory/maintenance.py

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import psutil

from db.connection import get_db_connection_context
from .telemetry import MemoryTelemetry
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
MEMORY_CLEANUP_DURATION = Histogram(
    'memory_cleanup_duration_seconds',
    'Time taken for memory cleanup operation'
)

class MemoryMaintenance:
    """Handles memory system maintenance operations."""
    
    def __init__(self):
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=1)
        self.retention_days = 30
        self.importance_threshold = 0.5
        self.memory_threshold = 85.0  # percentage
    
    async def cleanup_old_memories(self) -> Dict[str, Any]:
        """
        Clean up old, low-importance memories based on configured thresholds.
        """
        # ──------------  DEBUG PROBE  ------------
        logger.debug(
            "DEBUG – get_db_connection_context=%r  MemoryTelemetry.record=%r",
            type(get_db_connection_context), type(MemoryTelemetry.record)
        )
        start_time = datetime.now()
        stats = {
            "deleted_count": 0,
            "archived_count": 0,
            "duration_seconds": 0
        }
        
        try:
            async with get_db_connection_context() as conn:
                # Delete old, unimportant memories
                threshold_date = datetime.now() - timedelta(days=self.retention_days)
                result = await conn.execute("""
                    WITH deleted AS (
                        DELETE FROM unified_memories
                        WHERE last_accessed < $1 
                        AND importance < $2
                        AND NOT is_core_memory
                        RETURNING id
                    )
                    SELECT COUNT(*) FROM deleted
                """, threshold_date, self.importance_threshold)
                
                stats["deleted_count"] = result
                
                # Archive old but important memories
                archive_result = await conn.execute("""
                    WITH archived AS (
                        UPDATE unified_memories
                        SET is_archived = TRUE
                        WHERE last_accessed < $1 
                        AND importance >= $2
                        AND NOT is_archived
                        AND NOT is_core_memory
                        RETURNING id
                    )
                    SELECT COUNT(*) FROM archived
                """, threshold_date, self.importance_threshold)
                
                stats["archived_count"] = archive_result
                
                # Record telemetry
                duration = (datetime.now() - start_time).total_seconds()
                stats["duration_seconds"] = duration
                
                await MemoryTelemetry.record(
                    0,
                    0,
                    operation="memory_cleanup",
                    success=True,
                    duration=duration,
                    metadata=stats
                )
                
                # Update Prometheus metric
                MEMORY_CLEANUP_DURATION.observe(duration)
                
                logger.info(f"Memory cleanup completed: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            await MemoryTelemetry.record(
                0,
                0,
                operation="memory_cleanup",
                success=False,
                error=str(e)
            )
            raise
    
    async def should_run_cleanup(self) -> bool:
        """
        Determine if cleanup should run based on time interval
        and system memory usage.
        """
        if datetime.now() - self.last_cleanup < self.cleanup_interval:
            return False
            
        # Check system memory usage
        memory = psutil.virtual_memory()
        return memory.percent >= self.memory_threshold
    
    async def start_maintenance_loop(self):
        """Start the background maintenance loop."""
        while True:
            try:
                if await self.should_run_cleanup():
                    await self.cleanup_old_memories()
                    self.last_cleanup = datetime.now()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying 
