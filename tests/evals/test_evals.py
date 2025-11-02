import json
import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from nyx.common.evals import combine_results
from nyx.conflict import models as conflict_models
from nyx.conflict.fsm import Status
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


def test_combine_results_merges_flags():
    score, flags = combine_results(
        {"score": 0.8, "flags": ["repetition"]},
        {"score": 0.6, "flags": ["policy_violation", "repetition"]},
        {"score": 1.0, "flags": []},
    )

    assert pytest.approx(score) == 0.8
    assert flags == ["repetition", "policy_violation"]


def _seed_conflict(factory: sessionmaker, conflict_id: uuid.UUID) -> None:
    with factory.begin() as session:
        session.add(
            ConflictResolution(
                conflict_id=conflict_id,
                status=Status.EVAL.value,
                draft_text="seed draft",
            )
        )


def test_eval_draft_fails_when_score_below_threshold(conflict_session_factory, monkeypatch):
    conflict_id = uuid.uuid4()
    _seed_conflict(conflict_session_factory, conflict_id)

    async def low_score(*_args, **_kwargs):
        return 0.3, {"notes": "insufficient"}, []

    queued = []

    monkeypatch.setattr(conflict_pipeline, "_evaluate_draft_async", low_score, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", lambda *_a, **_k: queued.append("integrate"), raising=False)
    monkeypatch.delenv("NYX_EVAL_MIN_SCORE_DEFAULT", raising=False)

    result = conflict_pipeline.eval_draft({"conflict_id": str(conflict_id)})
    assert result["status"] == Status.FAILED.value

    with conflict_session_factory.begin() as session:
        row = session.scalars(
            select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
        ).one()
        assert row.status == Status.FAILED.value
        assert row.eval_score == pytest.approx(0.3)
        assert json.loads(row.eval_notes)["notes"] == "insufficient"

    assert not queued


def test_eval_draft_passes_when_threshold_met(conflict_session_factory, monkeypatch):
    conflict_id = uuid.uuid4()
    _seed_conflict(conflict_session_factory, conflict_id)

    async def high_score(*_args, **_kwargs):
        return 0.92, {"notes": "looks good"}, []

    queued = []

    monkeypatch.setattr(conflict_pipeline, "_evaluate_draft_async", high_score, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", lambda *_a, **_k: queued.append("integrate"), raising=False)
    monkeypatch.setenv("NYX_EVAL_MIN_SCORE_DEFAULT", "0.8")

    result = conflict_pipeline.eval_draft({"conflict_id": str(conflict_id)})
    assert result["status"] == Status.CANON.value

    with conflict_session_factory.begin() as session:
        row = session.scalars(
            select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
        ).one()
        assert row.status == Status.CANON.value
        assert row.eval_score == pytest.approx(0.92)
        assert json.loads(row.eval_notes)["notes"] == "looks good"

    assert queued == ["integrate"]


def test_eval_draft_blocks_on_flag(conflict_session_factory, monkeypatch):
    conflict_id = uuid.uuid4()
    _seed_conflict(conflict_session_factory, conflict_id)

    async def flagged_score(*_args, **_kwargs):
        return 0.95, {"notes": "policy concern"}, ["policy_violation"]

    queued = []

    monkeypatch.setattr(conflict_pipeline, "_evaluate_draft_async", flagged_score, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", lambda *_a, **_k: queued.append("integrate"), raising=False)
    monkeypatch.setenv("NYX_EVAL_BLOCKING_FLAGS", "policy_violation")
    monkeypatch.setenv("NYX_EVAL_MIN_SCORE_DEFAULT", "0.6")

    result = conflict_pipeline.eval_draft({"conflict_id": str(conflict_id)})
    assert result["status"] == Status.FAILED.value

    with conflict_session_factory.begin() as session:
        row = session.scalars(
            select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
        ).one()
        assert row.status == Status.FAILED.value
        stored = json.loads(row.eval_notes)
        assert stored["notes"] == "policy concern"
        assert stored["blocking_flags"] == ["policy_violation"]

    assert not queued
