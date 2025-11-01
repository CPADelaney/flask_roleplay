import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from nyx.conflict import models as conflict_models
from nyx.conflict.fsm import Status, transition
from nyx.conflict.models import Base, ConflictResolution
from nyx.tasks.background import conflict_pipeline


@pytest.fixture()
def conflict_session_factory(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    monkeypatch.setattr(conflict_models, "_engine", engine, raising=False)
    monkeypatch.setattr(conflict_models, "_SessionFactory", factory, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_SESSION_FACTORY", factory, raising=False)

    try:
        yield factory
    finally:
        factory.close_all()
        engine.dispose()


def test_transition_validations(conflict_session_factory):
    session = conflict_session_factory()
    conflict_id = uuid.uuid4()

    with session.begin():
        row = ConflictResolution(conflict_id=conflict_id, status=Status.DRAFT.value)
        session.add(row)
        session.flush()

        transition(session, row, Status.DRAFT, draft_text="initial draft")
        assert row.draft_text == "initial draft"

        transition(session, row, Status.EVAL)
        assert row.status == Status.EVAL.value

        with pytest.raises(ValueError):
            transition(session, row, Status.INTEGRATED)


def test_start_pipeline_resumes_without_regenerating(conflict_session_factory, monkeypatch):
    factory = conflict_session_factory
    conflict_id = uuid.uuid4()

    with factory.begin() as session:
        session.add(
            ConflictResolution(
                conflict_id=conflict_id,
                status=Status.DRAFT.value,
                draft_text="seed draft",
            )
        )

    async def fail_generate(*_args, **_kwargs):  # pragma: no cover - safety guard
        raise AssertionError("draft should not be regenerated")

    queued: list[tuple[str, str]] = []

    monkeypatch.setattr(conflict_pipeline, "_generate_draft_async", fail_generate, raising=False)

    def record_eval(conflict_id_arg, payload):
        queued.append((str(conflict_id_arg), payload.get("conflict_id")))

    monkeypatch.setattr(conflict_pipeline, "_queue_eval", record_eval, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", lambda *_a, **_k: None, raising=False)

    result = conflict_pipeline.start_pipeline({"conflict_id": str(conflict_id)})
    assert result["status"] == Status.EVAL.value

    with factory.begin() as session:
        row = session.scalars(
            select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
        ).one()
        assert row.status == Status.EVAL.value
        assert row.draft_text == "seed draft"

    assert queued  # evaluation should be queued exactly once
    assert queued[0][0] == str(conflict_id)


def test_start_pipeline_idempotent_for_duplicates(conflict_session_factory, monkeypatch):
    factory = conflict_session_factory
    conflict_id = uuid.uuid4()

    draft_calls = 0

    async def generate_draft(*_args, **_kwargs):
        nonlocal draft_calls
        draft_calls += 1
        return "initial draft"

    monkeypatch.setattr(conflict_pipeline, "_generate_draft_async", generate_draft, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_eval", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", lambda *_a, **_k: None, raising=False)

    result_first = conflict_pipeline.start_pipeline({"conflict_id": str(conflict_id)})
    assert result_first["status"] == Status.EVAL.value

    result_second = conflict_pipeline.start_pipeline({"conflict_id": str(conflict_id)})
    assert result_second["status"] == Status.EVAL.value

    assert draft_calls == 1

    with factory.begin() as session:
        rows = session.scalars(select(ConflictResolution)).all()
        assert len(rows) == 1
        assert rows[0].status == Status.EVAL.value
