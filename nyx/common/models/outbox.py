"""SQLAlchemy models backing the Nyx persistent outbox."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.types import JSON, Uuid


Base = declarative_base()


class OutboxEventStatus(str, enum.Enum):
    """Lifecycle state for an outbox event."""

    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OutboxEvent(Base):
    """Durable event queued for asynchronous dispatch."""

    __tablename__ = "outbox_events"

    id = Column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    topic = Column(String, nullable=False)
    payload = Column(
        MutableDict.as_mutable(JSONB().with_variant(JSON(), "sqlite")),
        nullable=False,
    )
    dedupe_key = Column(String, nullable=True)
    status = Column(
        String,
        nullable=False,
        default=OutboxEventStatus.PENDING.value,
    )
    attempts = Column(Integer, nullable=False, default=0)
    available_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    last_error = Column(Text, nullable=True)

    __table_args__ = (
        Index(
            "uq_outbox_topic_dedupe_key",
            "topic",
            "dedupe_key",
            unique=True,
            postgresql_where=dedupe_key.isnot(None),
        ),
    )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the event."""

        return {
            "id": str(self.id) if self.id is not None else None,
            "topic": self.topic,
            "payload": dict(self.payload or {}),
            "dedupe_key": self.dedupe_key,
            "status": self.status,
            "attempts": self.attempts,
            "available_at": self.available_at,
            "created_at": self.created_at,
            "last_error": self.last_error,
        }


__all__ = [
    "Base",
    "OutboxEvent",
    "OutboxEventStatus",
]
