"""Dispatcher that drains the persistent outbox and sends Celery tasks."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from celery import current_app
from celery.canvas import Signature
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from nyx.common.models.outbox import OutboxEvent, OutboxEventStatus

logger = logging.getLogger(__name__)

TopicHandler = Callable[[OutboxEvent], Signature]

_TOPICS: Dict[str, List[TopicHandler]] = {}

_DEFAULT_TASKS: Dict[str, Tuple[str, ...]] = {
    "ConflictRouteRequested": ("nyx.tasks.background.conflict.route_subsystems",),
    "ConflictResolutionRequested": ("nyx.tasks.background.conflict.start_pipeline",),
    "NPCDecisionNeeded": ("nyx.tasks.background.npc.compute_decision",),
    "ConflictResolved": (
        "nyx.subscribers.memory.on_conflict_resolved",
        "nyx.subscribers.npc.on_conflict_resolved",
        "nyx.subscribers.world.on_conflict_resolved",
    ),
    "NPCActionTaken": ("nyx.subscribers.memory.on_npc_action_taken",),
    "MemoryCreated": ("nyx.subscribers.npc.on_memory_created",),
}


def _celery_signature_factory(task_name: str) -> TopicHandler:
    def _factory(event: OutboxEvent) -> Signature:
        return current_app.signature(task_name, args=(event.payload,))

    return _factory


def register_topic_handler(topic: str, handler: TopicHandler, *, overwrite: bool = True) -> None:
    """Register a handler used to dispatch events for a topic."""

    if overwrite or topic not in _TOPICS:
        _TOPICS[topic] = [handler]
        return
    _TOPICS[topic].append(handler)


def clear_topic_handlers() -> None:
    """Clear the topic registry."""

    _TOPICS.clear()


def register_default_topic_handlers() -> None:
    """Ensure the default topics are present in the registry."""

    for topic, task_names in _DEFAULT_TASKS.items():
        for task_name in task_names:
            register_topic_handler(topic, _celery_signature_factory(task_name), overwrite=False)


register_default_topic_handlers()


@dataclass
class DispatchSummary:
    """Aggregated counts from a dispatcher run."""

    dispatched: int = 0
    retried: int = 0
    failed: int = 0
    skipped: int = 0


def _supports_skip_locked(session: Session) -> bool:
    bind = session.get_bind()
    dialect = getattr(bind, "dialect", None)
    if dialect is None:
        return False
    if getattr(dialect, "name", "") == "sqlite":
        return False
    return getattr(dialect, "supports_select_for_update_with_skip_locked", False) or getattr(
        dialect, "supports_for_update", False
    )


def _select_pending_events(session: Session, *, limit: int, now: datetime) -> Iterable[OutboxEvent]:
    stmt = (
        select(OutboxEvent)
        .where(
            OutboxEvent.status == OutboxEventStatus.PENDING.value,
            OutboxEvent.available_at <= now,
        )
        .order_by(OutboxEvent.created_at)
        .limit(limit)
    )
    if _supports_skip_locked(session):
        stmt = stmt.with_for_update(skip_locked=True)
    return session.scalars(stmt).all()


def dispatch_once(
    session_factory: sessionmaker,
    *,
    limit: int = 100,
    now: Optional[datetime] = None,
) -> DispatchSummary:
    """Drain pending outbox events once using the supplied session factory."""

    summary = DispatchSummary()
    current_time = now or datetime.now(timezone.utc)

    session: Optional[Session] = None
    try:
        session = session_factory()
        with session.begin():
            events = _select_pending_events(session, limit=limit, now=current_time)
            for event in events:
                handlers = _TOPICS.get(event.topic, [])
                if not handlers:
                    logger.error("No handler registered for topic %s", event.topic)
                    event.status = OutboxEventStatus.FAILED.value
                    event.last_error = f"No handler for topic {event.topic}"
                    event.attempts += 1
                    summary.failed += 1
                    continue

                dispatch_errors: List[str] = []
                for index, handler in enumerate(handlers):
                    try:
                        signature = handler(event)
                        signature.apply_async(headers={"Idempotency-Key": f"{event.id}:{index}"})
                    except Exception as exc:  # pragma: no cover - network issues
                        logger.warning(
                            "Failed to dispatch outbox event %s via handler %s: %s",
                            event.id,
                            handler,
                            exc,
                        )
                        dispatch_errors.append(str(exc))

                if dispatch_errors:
                    event.attempts += 1
                    event.last_error = "; ".join(dispatch_errors)
                    backoff_seconds = min(2 ** event.attempts, 300)
                    event.available_at = current_time + timedelta(seconds=backoff_seconds)
                    summary.retried += 1
                else:
                    event.status = OutboxEventStatus.DISPATCHED.value
                    event.last_error = None
                    summary.dispatched += 1
    finally:
        if session is not None:
            session.close()

    return summary


def run_once(limit: int = 100) -> int:
    """Dispatch pending events using the default session factory."""

    from nyx.common.outbox import get_session_factory

    dsn = os.getenv("DB_DSN") or os.getenv("DATABASE_URL")
    session_factory = get_session_factory(dsn)
    summary = dispatch_once(session_factory, limit=limit)
    return summary.dispatched


def main(argv: Optional[Iterable[str]] = None) -> DispatchSummary:
    """Command-line entry point for Celery beat to invoke once."""

    parser = argparse.ArgumentParser(description="Dispatch pending outbox events once")
    parser.add_argument("--limit", type=int, default=100, help="Maximum events to process")
    parser.add_argument(
        "--dsn",
        type=str,
        default=os.getenv("DB_DSN") or os.getenv("DATABASE_URL"),
        help="Database DSN for the outbox connection",
    )
    args = parser.parse_args(argv)

    from nyx.common.outbox import get_session_factory

    session_factory = get_session_factory(args.dsn)

    summary = dispatch_once(session_factory, limit=args.limit)
    logger.info(
        "Outbox dispatcher run complete: dispatched=%s retried=%s failed=%s",  # noqa: G200
        summary.dispatched,
        summary.retried,
        summary.failed,
    )
    return summary


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()


__all__ = [
    "DispatchSummary",
    "clear_topic_handlers",
    "dispatch_once",
    "run_once",
    "register_default_topic_handlers",
    "register_topic_handler",
]
