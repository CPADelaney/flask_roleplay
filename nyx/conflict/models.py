"""SQLAlchemy models for the conflict resolution state machine."""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import Column, DateTime, Float, Index, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.types import JSON, Uuid
from sqlalchemy.ext.mutable import MutableDict


Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConflictResolution(Base):
    """Persisted progress for an asynchronous conflict resolution."""

    __tablename__ = "conflict_resolution"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    conflict_id = Column(Uuid, nullable=False)
    status = Column(String, nullable=False)
    draft_text = Column(Text, nullable=True)
    eval_score = Column(Float, nullable=True)
    eval_notes = Column(Text, nullable=True)
    integrated_changes = Column(
        MutableDict.as_mutable(JSONB().with_variant(JSON(), "sqlite")),
        nullable=True,
    )
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    __table_args__ = (
        Index(
            "uq_conflict_resolution_active",
            "conflict_id",
            "status",
            unique=True,
            postgresql_where=text("status IN ('DRAFT','EVAL','CANON','INTEGRATING')"),
            sqlite_where=text("status IN ('DRAFT','EVAL','CANON','INTEGRATING')"),
        ),
    )


_engine_lock = threading.Lock()
_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def _resolve_dsn(dsn: Optional[str] = None) -> str:
    value = dsn or os.getenv("DB_DSN") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("DB_DSN environment variable must be set for conflict persistence")
    return value


def get_engine(dsn: Optional[str] = None) -> Engine:
    """Return a singleton SQLAlchemy engine for conflict persistence."""

    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = create_engine(_resolve_dsn(dsn), future=True)
            Base.metadata.create_all(_engine)
    return _engine


def get_session_factory(dsn: Optional[str] = None) -> sessionmaker:
    """Return a lazily created session factory bound to the conflict engine."""

    global _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory
    with _engine_lock:
        if _SessionFactory is None:
            engine = get_engine(dsn)
            _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return _SessionFactory


@contextmanager
def conflict_session(dsn: Optional[str] = None) -> Iterable[Session]:
    """Yield a SQLAlchemy session for conflict persistence operations."""

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


__all__ = [
    "Base",
    "ConflictResolution",
    "conflict_session",
    "get_engine",
    "get_session_factory",
]
