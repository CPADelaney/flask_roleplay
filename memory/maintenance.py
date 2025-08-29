# memory/maintenance.py

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import psutil

from db.connection import get_db_connection_context
from .telemetry import MemoryTelemetry
from prometheus_client import Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
MEMORY_CLEANUP_DURATION = Histogram(
    'memory_cleanup_duration_seconds',
    'Time taken for memory cleanup operation'
)

class MemoryMaintenance:
    """Handles memory system maintenance operations against unified_memories."""

    def __init__(self):
        # Cleanup pacing and thresholds
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=1)

        # Age thresholds
        self.retention_days = 30               # delete trivial beyond this age
        self.archive_threshold_days = 90       # archive low-significance older than this age

        # Resource-based trigger
        self.memory_threshold = 85.0  # percent RAM used to trigger opportunistic cleanup

        # Significance thresholds (unified_memories.significance is 1..5)
        self.delete_significance_max = 1       # delete when significance <= 1, very old, rarely recalled
        self.archive_significance_max = 2      # archive when significance <= 2 and old

        # Recall thresholds
        self.delete_max_recalls = 0            # delete if never recalled
        self.archive_max_recalls = 1           # archive if recalled rarely

    async def cleanup_old_memories(self) -> Dict[str, Any]:
        """
        Clean up old, low-importance memories using fields available on unified_memories:
          - timestamp (fallback age), last_recalled (preferred age if present),
          - significance (1..5), times_recalled, status.
        Deletes trivially unimportant items and archives older low-significance items.
        """
        start_time = datetime.now()
        stats = {
            "deleted_count": 0,
            "archived_count": 0,
            "duration_seconds": 0.0
        }

        delete_before = datetime.now() - timedelta(days=self.retention_days)
        archive_before = datetime.now() - timedelta(days=self.archive_threshold_days)

        try:
            async with get_db_connection_context() as conn:
                # Delete trivial, never-recalled, very old (not already archived)
                # Use COALESCE(last_recalled, timestamp) as the age anchor.
                deleted_count = await conn.fetchval(
                    """
                    WITH deleted AS (
                        DELETE FROM unified_memories
                        WHERE COALESCE(last_recalled, timestamp) < $1
                          AND significance <= $2
                          AND COALESCE(times_recalled, 0) <= $3
                          AND status != 'archived'
                          AND status != 'deleted'
                        RETURNING id
                    )
                    SELECT COUNT(*) FROM deleted
                    """,
                    delete_before,
                    self.delete_significance_max,
                    self.delete_max_recalls
                ) or 0

                stats["deleted_count"] = int(deleted_count)

                # Archive older, low-significance memories not yet archived
                archived_count = await conn.fetchval(
                    """
                    WITH archived AS (
                        UPDATE unified_memories
                        SET status = 'archived'
                        WHERE COALESCE(last_recalled, timestamp) < $1
                          AND significance <= $2
                          AND COALESCE(times_recalled, 0) <= $3
                          AND status = 'active'
                        RETURNING id
                    )
                    SELECT COUNT(*) FROM archived
                    """,
                    archive_before,
                    self.archive_significance_max,
                    self.archive_max_recalls
                ) or 0

                stats["archived_count"] = int(archived_count)

                # Telemetry
                duration = (datetime.now() - start_time).total_seconds()
                stats["duration_seconds"] = duration

                await MemoryTelemetry.record(
                    0,
                    0,
                    operation="memory_cleanup",
                    success=True,
                    duration=duration,
                    data_size=stats["deleted_count"] + stats["archived_count"]
                )

                MEMORY_CLEANUP_DURATION.observe(duration)
                logger.info(f"Memory cleanup completed: {stats}")
                return stats

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}", exc_info=True)
            await MemoryTelemetry.record(
                0,
                0,
                operation="memory_cleanup",
                success=False,
                duration=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )
            # Reraise if you want supervisors to see failures; for now keep parity:
            raise

    async def should_run_cleanup(self) -> bool:
        """
        Determine if cleanup should run based on time interval and system memory usage.
        """
        if datetime.now() - self.last_cleanup < self.cleanup_interval:
            return False

        # System memory usage gate
        try:
            memory = psutil.virtual_memory()
            if memory and memory.percent >= self.memory_threshold:
                return True
        except Exception:
            # If psutil fails, fall back to time-based cleanup
            return True

        # Also allow time-only periodic cleanup (once interval elapses)
        return True

    async def start_maintenance_loop(self):
        """Start the background maintenance loop."""
        while True:
            try:
                if await self.should_run_cleanup():
                    await self.cleanup_old_memories()
                    self.last_cleanup = datetime.now()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}", exc_info=True)
                # Backoff on error
                await asyncio.sleep(60)
