from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from nyx.common.events import ConflictResolved, publish
from nyx.common.models.outbox import Base, OutboxEvent, OutboxEventStatus
from nyx.common.outbox_dispatcher import clear_topic_handlers, dispatch_once, register_topic_handler
from nyx.subscribers import memory_subscribers, npc_subscribers, world_subscribers
from nyx.utils.idempotency import clear_cache


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


def test_publish_persists_event(session_factory: sessionmaker) -> None:
    conflict_id = uuid4()

    with session_factory.begin() as session:
        publish(session, ConflictResolved(conflict_id=conflict_id, outcome={"result": "peace"}, trace_id="t-1"))

    with session_factory.begin() as session:
        rows = session.execute(select(OutboxEvent)).scalars().all()

    assert len(rows) == 1
    event = rows[0]
    assert event.topic == "ConflictResolved"
    assert event.payload["data"]["conflict_id"] == str(conflict_id)
    assert event.status == OutboxEventStatus.PENDING.value


def test_dispatch_fans_out_to_multiple_subscribers(session_factory: sessionmaker) -> None:
    calls: list[tuple[str, str, dict, dict]] = []

    def _handler_factory(name: str):
        def _handler(event: OutboxEvent):
            def _apply(headers=None):
                calls.append((name, str(event.id), dict(event.payload), dict(headers or {})))

            return SimpleNamespace(apply_async=_apply)

        return _handler

    clear_topic_handlers()
    register_topic_handler("ConflictResolved", _handler_factory("memory"))
    register_topic_handler("ConflictResolved", _handler_factory("npc"), overwrite=False)

    conflict_id = uuid4()
    with session_factory.begin() as session:
        publish(session, ConflictResolved(conflict_id=conflict_id, outcome={"winner": "players"}, trace_id=None))

    summary = dispatch_once(session_factory, limit=10)
    assert summary.dispatched == 1
    assert summary.retried == 0
    assert len(calls) == 2

    event_ids = {entry[1] for entry in calls}
    assert len(event_ids) == 1
    event_id = event_ids.pop()
    assert {entry[0] for entry in calls} == {"memory", "npc"}
    header_values = {entry[3]["Idempotency-Key"] for entry in calls}
    assert header_values == {f"{event_id}:0", f"{event_id}:1"}

    with session_factory.begin() as session:
        status = session.execute(
            select(OutboxEvent.status).where(OutboxEvent.topic == "ConflictResolved")
        ).scalar_one()
    assert status == OutboxEventStatus.DISPATCHED.value


def test_domain_subscribers_are_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()

    conflict_id = uuid4()
    npc_id = uuid4()
    memory_owner = uuid4()
    memory_id = uuid4()

    conflict_payload = {
        "type": "ConflictResolved",
        "data": {
            "conflict_id": str(conflict_id),
            "outcome": {"status": "success"},
            "trace_id": "trace-1",
        },
    }
    npc_action_payload = {
        "type": "NPCActionTaken",
        "data": {
            "npc_id": str(npc_id),
            "action": "attack",
            "payload": {"target": "player"},
            "trace_id": "trace-2",
        },
    }
    memory_created_payload = {
        "type": "MemoryCreated",
        "data": {
            "owner_id": str(memory_owner),
            "memory_id": str(memory_id),
            "tags": ["story", "notable"],
            "trace_id": "trace-3",
        },
    }

    memory_calls: list[tuple[UUID, dict, str | None]] = []
    npc_calls: list[tuple[UUID, dict, str | None]] = []
    world_calls: list[tuple[UUID, dict, str | None]] = []
    action_calls: list[tuple[UUID, str, dict, str | None]] = []
    mood_calls: list[tuple[UUID, UUID, list[str], str | None]] = []

    monkeypatch.setattr(
        memory_subscribers,
        "create_conflict_memories",
        lambda cid, outcome, trace: memory_calls.append((cid, outcome, trace)),
    )
    monkeypatch.setattr(
        npc_subscribers,
        "adjust_relationship_stats",
        lambda cid, outcome, trace: npc_calls.append((cid, outcome, trace)),
    )
    monkeypatch.setattr(
        world_subscribers,
        "update_world_tension_counters",
        lambda cid, outcome, trace: world_calls.append((cid, outcome, trace)),
    )
    monkeypatch.setattr(
        memory_subscribers,
        "store_npc_short_term_memory",
        lambda nid, action, payload, trace: action_calls.append((nid, action, payload, trace)),
    )
    monkeypatch.setattr(
        npc_subscribers,
        "adjust_npc_mood",
        lambda owner, mem, tags, trace: mood_calls.append((owner, mem, list(tags), trace)),
    )

    result_memory_first = memory_subscribers.on_conflict_resolved(payload=conflict_payload, trace_id="trace-1")
    result_memory_second = memory_subscribers.on_conflict_resolved(payload=conflict_payload, trace_id="trace-1")
    result_npc_conflict_first = npc_subscribers.on_conflict_resolved(payload=conflict_payload, trace_id="trace-1")
    result_npc_conflict_second = npc_subscribers.on_conflict_resolved(payload=conflict_payload, trace_id="trace-1")
    result_world_first = world_subscribers.on_conflict_resolved(payload=conflict_payload, trace_id="trace-1")
    result_world_second = world_subscribers.on_conflict_resolved(payload=conflict_payload, trace_id="trace-1")

    result_action_first = memory_subscribers.on_npc_action_taken(payload=npc_action_payload, trace_id="trace-2")
    result_action_second = memory_subscribers.on_npc_action_taken(payload=npc_action_payload, trace_id="trace-2")

    result_memory_created_first = npc_subscribers.on_memory_created(payload=memory_created_payload, trace_id="trace-3")
    result_memory_created_second = npc_subscribers.on_memory_created(payload=memory_created_payload, trace_id="trace-3")

    assert result_memory_first["status"] == "applied"
    assert result_memory_second is None
    assert result_npc_conflict_first["status"] == "applied"
    assert result_npc_conflict_second is None
    assert result_world_first["status"] == "applied"
    assert result_world_second is None
    assert result_action_first["status"] == "applied"
    assert result_action_second is None
    assert result_memory_created_first["status"] == "applied"
    assert result_memory_created_second is None

    assert len(memory_calls) == 1
    assert memory_calls[0][0] == conflict_id
    assert len(npc_calls) == 1
    assert npc_calls[0][0] == conflict_id
    assert len(world_calls) == 1
    assert world_calls[0][0] == conflict_id
    assert len(action_calls) == 1
    assert action_calls[0][0] == npc_id
    assert len(mood_calls) == 1
    assert mood_calls[0][0] == memory_owner


__all__ = [
    "session_factory",
    "test_dispatch_fans_out_to_multiple_subscribers",
    "test_domain_subscribers_are_idempotent",
    "test_publish_persists_event",
]

