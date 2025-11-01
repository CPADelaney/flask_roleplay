"""Celery tasks orchestrating the conflict resolution pipeline."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Tuple

from sqlalchemy import select
from sqlalchemy.exc import CompileError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from nyx.conflict.fsm import Status, transition
from nyx.conflict.models import ConflictResolution, get_session_factory as get_conflict_session_factory
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest
from nyx.tasks.base import NyxTask, app
from nyx.tasks.utils import run_coro, with_retry
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

_SESSION_FACTORY: sessionmaker | None = None


def _get_session_factory() -> sessionmaker:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is not None:
        return _SESSION_FACTORY
    _SESSION_FACTORY = get_conflict_session_factory()
    return _SESSION_FACTORY


def _parse_conflict_id(payload: Dict[str, Any]) -> uuid.UUID:
    value = payload.get("conflict_id")
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must include a valid conflict_id") from exc


def _select_for_update(session: Session, conflict_id: uuid.UUID) -> ConflictResolution | None:
    stmt = select(ConflictResolution).where(ConflictResolution.conflict_id == conflict_id)
    try:
        return session.scalars(stmt.with_for_update()).first()
    except (CompileError, OperationalError):
        return session.scalars(stmt).first()


def _get_or_create_resolution(session: Session, conflict_id: uuid.UUID) -> ConflictResolution:
    row = _select_for_update(session, conflict_id)
    if row is not None:
        return row
    row = ConflictResolution(conflict_id=conflict_id, status=Status.DRAFT.value)
    session.add(row)
    session.flush()
    session.refresh(row)
    return row


def _serialize_resolution(row: ConflictResolution) -> Dict[str, Any]:
    return {
        "conflict_id": str(row.conflict_id),
        "id": str(row.id) if row.id else None,
        "status": row.status,
        "draft_text": row.draft_text,
        "eval_score": row.eval_score,
        "eval_notes": row.eval_notes,
        "integrated_changes": dict(row.integrated_changes or {}),
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _extract_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    raw = getattr(result, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if raw is not None:
        try:
            return json.dumps(raw, default=str)
        except Exception:  # pragma: no cover - best effort serialisation
            return str(raw)
    return ""


async def _generate_draft_async(conflict_id: uuid.UUID, payload: Dict[str, Any]) -> str:
    request_spec = payload.get("draft_request")
    prompt: str | None = None
    agent: str | None = None
    context = None
    runner_kwargs = None
    metadata: Dict[str, Any] = {}

    if isinstance(request_spec, dict):
        prompt = request_spec.get("prompt")
        agent = request_spec.get("agent")
        context = request_spec.get("context")
        runner_kwargs = request_spec.get("runner_kwargs")
        metadata = dict(request_spec.get("metadata") or {})

    if not prompt:
        context_payload = payload.get("context") or {}
        try:
            context_dump = json.dumps(context_payload, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            context_dump = str(context_payload)
        prompt = (
            "Draft a proposed resolution for the following conflict.\n"
            f"Conflict ID: {conflict_id}\n"
            f"Resolution type: {payload.get('resolution_type', 'resolution')}\n"
            f"Context:\n{context_dump}"
        )

    if not agent:
        agent = os.getenv("NYX_CONFLICT_DRAFT_AGENT")

    metadata.setdefault("operation", "conflict_resolution_draft")
    metadata.setdefault("conflict_id", str(conflict_id))

    if agent:
        result = await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
                context=context,
                metadata=metadata,
                runner_kwargs=runner_kwargs,
            )
        )
        text = _extract_text(result)
        if text:
            return text

    return prompt


async def _evaluate_draft_async(conflict_id: uuid.UUID, draft_text: str, payload: Dict[str, Any]) -> Tuple[float | None, str | None]:
    evaluation = payload.get("evaluation") or {}

    if "score" in evaluation:
        try:
            score = float(evaluation.get("score"))
        except (TypeError, ValueError):
            score = None
        return score, evaluation.get("notes")

    request_spec = evaluation.get("request")
    agent = None
    context = None
    runner_kwargs = None
    metadata: Dict[str, Any] = {"operation": "conflict_resolution_eval", "conflict_id": str(conflict_id)}
    prompt: str | None = None

    if isinstance(request_spec, dict):
        prompt = request_spec.get("prompt")
        agent = request_spec.get("agent")
        context = request_spec.get("context")
        runner_kwargs = request_spec.get("runner_kwargs")
        metadata.update(request_spec.get("metadata") or {})

    if not prompt:
        prompt = (
            "Evaluate the quality of the following conflict resolution draft and return a JSON object with a numeric 'score' "
            "between 0 and 1 and optional 'notes'.\n"
            f"Draft:\n{draft_text}"
        )

    if not agent:
        agent = os.getenv("NYX_CONFLICT_EVAL_AGENT")

    if agent:
        result = await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
                context=context,
                metadata=metadata,
                runner_kwargs=runner_kwargs,
            )
        )
        text = _extract_text(result)
        if text:
            try:
                parsed = json.loads(text)
                score = float(parsed.get("score")) if parsed.get("score") is not None else None
                notes = parsed.get("notes")
                return score, notes
            except (TypeError, ValueError, json.JSONDecodeError):
                try:
                    return float(text), None
                except (TypeError, ValueError):
                    return None, text

    # Default heuristic: accept drafts that are non-empty
    return (1.0 if draft_text else 0.0), "auto-evaluated"


async def _integrate_async(conflict_id: uuid.UUID, payload: Dict[str, Any]) -> Dict[str, Any]:
    integration = payload.get("integration")
    if callable(integration):  # pragma: no cover - defensive hook
        try:
            result = integration()
            if isinstance(result, dict):
                return result
        except Exception:
            logger.exception("Custom integration callable failed", extra={"conflict_id": str(conflict_id)})

    if isinstance(integration, dict):
        changes = dict(integration)
    else:
        changes = {}

    changes.setdefault("conflict_id", str(conflict_id))
    changes.setdefault("resolution_type", payload.get("resolution_type"))
    changes.setdefault("context", payload.get("context") or {})
    return changes


def _queue_eval(conflict_id: uuid.UUID, payload: Dict[str, Any]) -> None:
    app.signature(
        "nyx.tasks.background.conflict.eval_draft",
        args=(dict(payload),),
    ).apply_async()


def _queue_integrate(conflict_id: uuid.UUID, payload: Dict[str, Any]) -> None:
    app.signature(
        "nyx.tasks.background.conflict.integrate_canon",
        args=(dict(payload),),
    ).apply_async()


def _start_key(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    return f"conflict:start:{conflict_id}" if conflict_id else ""


def _eval_key(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    return f"conflict:eval:{conflict_id}" if conflict_id else ""


def _integrate_key(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    return f"conflict:integrate:{conflict_id}" if conflict_id else ""


def _resolve_threshold(payload: Dict[str, Any]) -> float:
    evaluation = payload.get("evaluation") or {}
    try:
        return float(evaluation.get("threshold", 0.7))
    except (TypeError, ValueError):
        return 0.7


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.conflict.start_pipeline",
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_start_key)
def start_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    try:
        conflict_id = _parse_conflict_id(payload)
    except ValueError as exc:
        logger.warning("Invalid conflict resolution payload: %s", exc)
        return {"status": "skipped", "error": str(exc)}

    session_factory = _get_session_factory()
    session = session_factory()
    should_eval = False
    should_integrate = False
    result: Dict[str, Any]

    try:
        with session.begin():
            row = _get_or_create_resolution(session, conflict_id)
            status = Status(row.status)

            if status in (Status.INTEGRATED, Status.FAILED):
                result = _serialize_resolution(row)
            elif status == Status.DRAFT:
                if not row.draft_text:
                    try:
                        draft_text = run_coro(_generate_draft_async(conflict_id, payload))
                    except Exception as exc:  # pragma: no cover - LLM failure path
                        logger.exception("Draft generation failed", extra={"conflict_id": str(conflict_id)})
                        row = transition(session, row, Status.FAILED, eval_notes=f"draft_error: {exc}")
                        result = _serialize_resolution(row)
                    else:
                        row = transition(session, row, Status.DRAFT, draft_text=draft_text)
                        row = transition(session, row, Status.EVAL)
                        should_eval = True
                        result = _serialize_resolution(row)
                else:
                    row = transition(session, row, Status.EVAL)
                    should_eval = True
                    result = _serialize_resolution(row)
            elif status == Status.EVAL:
                should_eval = True
                result = _serialize_resolution(row)
            elif status == Status.CANON:
                should_integrate = True
                result = _serialize_resolution(row)
            elif status == Status.INTEGRATING:
                should_integrate = True
                result = _serialize_resolution(row)
            else:
                result = _serialize_resolution(row)
    finally:
        session.close()

    if should_eval:
        _queue_eval(conflict_id, payload)
    if should_integrate:
        _queue_integrate(conflict_id, payload)

    return result


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.conflict.eval_draft",
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_eval_key)
def eval_draft(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    try:
        conflict_id = _parse_conflict_id(payload)
    except ValueError as exc:
        logger.warning("Invalid conflict resolution payload: %s", exc)
        return {"status": "skipped", "error": str(exc)}

    session_factory = _get_session_factory()
    session = session_factory()
    should_integrate = False
    result: Dict[str, Any]

    try:
        with session.begin():
            row = _select_for_update(session, conflict_id)
            if row is None:
                logger.info("No conflict resolution record for eval", extra={"conflict_id": str(conflict_id)})
                return {"status": "missing", "conflict_id": str(conflict_id)}

            status = Status(row.status)
            if status == Status.EVAL:
                draft_text = row.draft_text or ""
                try:
                    score, notes = run_coro(_evaluate_draft_async(conflict_id, draft_text, payload))
                except Exception as exc:  # pragma: no cover - LLM failure path
                    logger.exception("Draft evaluation failed", extra={"conflict_id": str(conflict_id)})
                    row = transition(session, row, Status.FAILED, eval_notes=f"eval_error: {exc}")
                else:
                    threshold = _resolve_threshold(payload)
                    if score is not None and score >= threshold:
                        row = transition(session, row, Status.CANON, eval_score=score, eval_notes=notes)
                        should_integrate = True
                    else:
                        row = transition(session, row, Status.FAILED, eval_score=score, eval_notes=notes)
                result = _serialize_resolution(row)
            elif status in (Status.CANON, Status.INTEGRATING):
                should_integrate = True
                result = _serialize_resolution(row)
            else:
                result = _serialize_resolution(row)
    finally:
        session.close()

    if should_integrate:
        _queue_integrate(conflict_id, payload)

    return result


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.conflict.integrate_canon",
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_integrate_key)
def integrate_canon(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    try:
        conflict_id = _parse_conflict_id(payload)
    except ValueError as exc:
        logger.warning("Invalid conflict resolution payload: %s", exc)
        return {"status": "skipped", "error": str(exc)}

    session_factory = _get_session_factory()
    session = session_factory()

    try:
        with session.begin():
            row = _select_for_update(session, conflict_id)
            if row is None:
                logger.info("No conflict resolution record for integration", extra={"conflict_id": str(conflict_id)})
                return {"status": "missing", "conflict_id": str(conflict_id)}

            status = Status(row.status)
            if status == Status.CANON:
                row = transition(session, row, Status.INTEGRATING)
                status = Status(row.status)

            if status == Status.INTEGRATING:
                try:
                    changes = run_coro(_integrate_async(conflict_id, payload))
                except Exception as exc:  # pragma: no cover - integration failure path
                    logger.exception("Conflict integration failed", extra={"conflict_id": str(conflict_id)})
                    row = transition(session, row, Status.FAILED, eval_notes=f"integration_error: {exc}")
                else:
                    row = transition(session, row, Status.INTEGRATED, integrated_changes=changes)

            result = _serialize_resolution(row)
    finally:
        session.close()

    return result


__all__ = ["start_pipeline", "eval_draft", "integrate_canon"]
