import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from nyx.common.evals import combine_results
from nyx.conflict.fsm import Status
from nyx.conflict.models import Base, ConflictResolution
from nyx.tasks.background import conflict_pipeline


@pytest.fixture()
def conflict_session_factory(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    monkeypatch.setattr(conflict_pipeline, "_SESSION_FACTORY", factory, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_OUTBOX_SESSION_FACTORY", factory, raising=False)
    monkeypatch.setattr(conflict_pipeline, "metrics_factory", None, raising=False)

    try:
        yield factory
    finally:
        factory.close_all()
        engine.dispose()


def test_combine_results_merges_scores_and_flags():
    result = combine_results(
        {"score": 0.9, "flags": ["alpha"], "notes": "first"},
        {"score": 0.3, "flags": ["beta", "alpha"], "notes": "second"},
    )
    assert result[0] == pytest.approx(0.6, rel=1e-6)
    assert set(result[1]) == {"alpha", "beta"}


def _seed_conflict(factory: sessionmaker) -> uuid.UUID:
    conflict_id = uuid.uuid4()
    with factory.begin() as session:
        session.add(
            ConflictResolution(
                conflict_id=conflict_id,
                status=Status.EVAL.value,
                draft_text="A brave compromise was reached after tense debate.",
            )
        )
    return conflict_id


def test_eval_draft_fails_below_threshold(conflict_session_factory, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NYX_EVAL_BLOCKING_FLAGS", raising=False)
    conflict_id = _seed_conflict(conflict_session_factory)

    async def fake_evaluate(*_args, **_kwargs):
        return 0.4, "too low", []

    monkeypatch.setattr(conflict_pipeline, "_evaluate_draft_async", fake_evaluate, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", lambda *_a, **_k: None, raising=False)

    result = conflict_pipeline.eval_draft({"conflict_id": str(conflict_id)})
    assert result["status"] == Status.FAILED.value

    with conflict_session_factory.begin() as session:
        row = session.scalars(
            select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
        ).one()
        assert row.status == Status.FAILED.value
        assert row.eval_score == 0.4
        assert "too low" in (row.eval_notes or "")


def test_eval_draft_blocks_on_flag(conflict_session_factory, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NYX_EVAL_BLOCKING_FLAGS", "policy_violation")
    conflict_id = _seed_conflict(conflict_session_factory)

    async def flagged(*_args, **_kwargs):
        return 0.92, "looks good", ["policy_violation"]

    monkeypatch.setattr(conflict_pipeline, "_evaluate_draft_async", flagged, raising=False)
    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", lambda *_a, **_k: None, raising=False)

    result = conflict_pipeline.eval_draft({"conflict_id": str(conflict_id)})
    assert result["status"] == Status.FAILED.value

    with conflict_session_factory.begin() as session:
        row = session.scalars(
            select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
        ).one()
        assert row.status == Status.FAILED.value
        assert row.eval_score == 0.92
        assert "flag:policy_violation" in (row.eval_notes or "")


def test_eval_draft_promotes_when_passing(conflict_session_factory, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NYX_EVAL_BLOCKING_FLAGS", raising=False)
    conflict_id = _seed_conflict(conflict_session_factory)
    recorded = []

    async def passing(*_args, **_kwargs):
        return 0.88, "solid", []

    monkeypatch.setattr(conflict_pipeline, "_evaluate_draft_async", passing, raising=False)

    def record_integrate(conflict_id_arg, payload):
        recorded.append((str(conflict_id_arg), payload.get("conflict_id")))

    monkeypatch.setattr(conflict_pipeline, "_queue_integrate", record_integrate, raising=False)

    result = conflict_pipeline.eval_draft({"conflict_id": str(conflict_id)})
    assert result["status"] == Status.CANON.value

    with conflict_session_factory.begin() as session:
        row = session.scalars(
            select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
        ).one()
        assert row.status == Status.CANON.value
        assert row.eval_score == 0.88
        assert "solid" in (row.eval_notes or "")

    assert recorded
    assert recorded[0][0] == str(conflict_id)
