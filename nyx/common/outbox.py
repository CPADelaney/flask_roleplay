"""High-level helpers for the Nyx persistent outbox."""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from nyx.common.models.outbox import Base, OutboxEvent, OutboxEventStatus


class DuplicateEventError(RuntimeError):
    """Raised when attempting to append a duplicate deduplicated event."""


_engine_lock = threading.Lock()
_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def _resolve_dsn(dsn: Optional[str] = None) -> str:
    value = dsn or os.getenv("DB_DSN") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("DB_DSN environment variable must be set for outbox operations")
    return value


def get_engine(dsn: Optional[str] = None) -> Engine:
    """Return a singleton SQLAlchemy engine bound to the configured DSN."""

    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = create_engine(_resolve_dsn(dsn), future=True)
            Base.metadata.create_all(_engine)
    return _engine


def get_session_factory(dsn: Optional[str] = None) -> sessionmaker:
    """Return a lazily initialised sessionmaker for outbox operations."""

    global _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory
    with _engine_lock:
        if _SessionFactory is None:
            engine = get_engine(dsn)
            _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return _SessionFactory


@contextmanager
def outbox_session(dsn: Optional[str] = None) -> Iterable[Session]:
    """Context manager returning a SQLAlchemy session bound to the outbox engine."""

    factory = get_session_factory(dsn)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def append_event(
    session: Session,
    *,
    topic: str,
    payload: dict,
    dedupe_key: str | None = None,
    available_at: Optional[datetime] = None,
) -> uuid.UUID:
    """Persist an event into the outbox using the provided session."""

    if dedupe_key:
        existing = session.scalar(
            select(OutboxEvent.id).where(
                OutboxEvent.topic == topic,
                OutboxEvent.dedupe_key == dedupe_key,
            )
        )
        if existing:
            raise DuplicateEventError(f"Outbox event already exists for topic={topic} dedupe={dedupe_key}")

    event = OutboxEvent(
        topic=topic,
        payload=dict(payload or {}),
        dedupe_key=dedupe_key,
        available_at=available_at,
    )
    session.add(event)
    try:
        session.flush()
    except IntegrityError as exc:  # pragma: no cover - race condition safety
        raise DuplicateEventError(str(exc)) from exc
    return event.id


def mark_dispatched(session: Session, event_id) -> None:
    """Mark an event as dispatched."""

    event = session.get(OutboxEvent, event_id)
    if event is None:
        raise LookupError(f"Outbox event {event_id} not found")
    event.status = OutboxEventStatus.DISPATCHED.value
    event.last_error = None
    session.flush()


def mark_failed(session: Session, event_id, error: str) -> None:
    """Mark an event as permanently failed."""

    event = session.get(OutboxEvent, event_id)
    if event is None:
        raise LookupError(f"Outbox event {event_id} not found")
    event.status = OutboxEventStatus.FAILED.value
    event.last_error = error
    event.attempts += 1
    session.flush()


__all__ = [
    "append_event",
    "DuplicateEventError",
    "get_engine",
    "get_session_factory",
    "mark_dispatched",
    "mark_failed",
    "outbox_session",
]
