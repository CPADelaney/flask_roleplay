# nyx/core/brain/nyx_event_log.py

import datetime
import json
import logging
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

class EventLogMixin:
    """
    Mixin for event-sourced logging of every Nyx state change or experience.
    """
    async def log_event(self, event_type, event_payload, event_time=None):
        nyx_id = getattr(self, "nyx_id", None) or getattr(self, "NYX_ID", None)
        instance_id = getattr(self, "instance_id", None) or getattr(self, "INSTANCE_ID", None)
        assert nyx_id and instance_id, "nyx_id and instance_id must be set"
        event_time = event_time or datetime.datetime.utcnow()
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO nyx_brain_events
                (nyx_id, instance_id, event_time, event_type, event_payload)
                VALUES ($1, $2, $3, $4, $5)
            """, nyx_id, instance_id, event_time, event_type, json.dumps(event_payload))
        logger.info(f"Event logged: {event_type} {event_payload}")

    async def get_events_since(self, since_time=None, limit=1000):
        nyx_id = getattr(self, "nyx_id", None) or getattr(self, "NYX_ID", None)
        async with get_db_connection_context() as conn:
            if since_time:
                rows = await conn.fetch("""
                    SELECT * FROM nyx_brain_events
                    WHERE nyx_id = $1 AND event_time > $2
                    ORDER BY event_time ASC
                    LIMIT $3
                """, nyx_id, since_time, limit)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM nyx_brain_events
                    WHERE nyx_id = $1
                    ORDER BY event_time ASC
                    LIMIT $2
                """, nyx_id, limit)
        return list(rows)
