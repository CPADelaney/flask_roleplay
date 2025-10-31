"""Tests for the persistent outbox and dispatcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from nyx.common.models.outbox import Base, OutboxEvent, OutboxEventStatus
from nyx.common.outbox import DuplicateEventError, append_event
from nyx.common.outbox_dispatcher import (
    clear_topic_handlers,
    dispatch_once,
    register_default_topic_handlers,
    register_topic_handler,
)


@pytest.fixture()
def session_factory() -> sessionmaker:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    try:
        yield factory
    finally:
        factory.close_all()
        engine.dispose()


@pytest.fixture(autouse=True)
def reset_topic_registry():
    clear_topic_handlers()
    yield
    register_default_topic_handlers()


def _make_recording_handler(store: Dict[str, List[Tuple[Dict, Dict]]], topic: str):
    def _handler(event: OutboxEvent):
        def _apply(headers=None):
            store.setdefault(topic, []).append((event.payload, headers or {}))

        return SimpleNamespace(apply_async=_apply)

    return _handler


def test_append_and_dispatch_happy_path(session_factory: sessionmaker):
    calls: Dict[str, List[Tuple[Dict, Dict]]] = {}
    register_topic_handler("ConflictRouteRequested", _make_recording_handler(calls, "ConflictRouteRequested"))
    register_topic_handler("NPCDecisionNeeded", _make_recording_handler(calls, "NPCDecisionNeeded"))

    with session_factory.begin() as session:
        first_id = append_event(
            session,
            topic="ConflictRouteRequested",
            payload={"conversation_id": 42},
        )
        second_id = append_event(
            session,
            topic="NPCDecisionNeeded",
            payload={"npc_id": 7},
        )

    summary = dispatch_once(session_factory, limit=10)
    assert summary.dispatched == 2
    assert summary.retried == 0

    assert calls["ConflictRouteRequested"][0][0] == {"conversation_id": 42}
    assert calls["ConflictRouteRequested"][0][1]["Idempotency-Key"] == str(first_id)
    assert calls["NPCDecisionNeeded"][0][0] == {"npc_id": 7}
    assert calls["NPCDecisionNeeded"][0][1]["Idempotency-Key"] == str(second_id)

    with session_factory.begin() as session:
        rows = session.execute(
            select(OutboxEvent.topic, OutboxEvent.status)
        ).all()
        statuses = {topic: status for topic, status in rows}
    assert statuses == {
        "ConflictRouteRequested": OutboxEventStatus.DISPATCHED.value,
        "NPCDecisionNeeded": OutboxEventStatus.DISPATCHED.value,
    }


def test_rollback_prevents_event_persistence(session_factory: sessionmaker):
    session: Session = session_factory()
    trans = session.begin()
    append_event(
        session,
        topic="ConflictRouteRequested",
        payload={"conversation_id": 5},
    )
    trans.rollback()
    session.close()

    summary = dispatch_once(session_factory, limit=5)
    assert summary.dispatched == 0

    with session_factory.begin() as session:
        count = session.scalar(select(sa.func.count()).select_from(OutboxEvent))
    assert count == 0


def test_dedupe_blocks_duplicate_event(session_factory: sessionmaker):
    with session_factory.begin() as session:
        append_event(
            session,
            topic="ConflictRouteRequested",
            payload={"conversation_id": 1},
            dedupe_key="scene-1",
        )

    with session_factory.begin() as session:
        with pytest.raises(DuplicateEventError):
            append_event(
                session,
                topic="ConflictRouteRequested",
                payload={"conversation_id": 1},
                dedupe_key="scene-1",
            )

    with session_factory.begin() as session:
        count = session.scalar(select(sa.func.count()).select_from(OutboxEvent))
    assert count == 1


def test_backoff_after_dispatch_failure(session_factory: sessionmaker):
    failure_store: List[Tuple[Dict, Dict]] = []

    def failing_handler(event: OutboxEvent):
        def _apply(headers=None):
            failure_store.append((event.payload, headers or {}))
            raise RuntimeError("boom")

        return SimpleNamespace(apply_async=_apply)

    register_topic_handler("ConflictRouteRequested", failing_handler)

    with session_factory.begin() as session:
        event_id = append_event(
            session,
            topic="ConflictRouteRequested",
            payload={"conversation_id": 10},
        )

    now = datetime.now(timezone.utc)
    summary = dispatch_once(session_factory, limit=1, now=now)
    assert summary.dispatched == 0
    assert summary.retried == 1
    assert failure_store[0][1]["Idempotency-Key"] == str(event_id)

    with session_factory.begin() as session:
        event = session.get(OutboxEvent, event_id)
        assert event is not None
        assert event.status == OutboxEventStatus.PENDING.value
        assert event.attempts == 1
        assert event.last_error == "boom"
        available_at = event.available_at
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=timezone.utc)
        assert now < available_at <= now + timedelta(seconds=2)

