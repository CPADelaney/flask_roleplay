"""Maintenance Celery tasks for Nyx."""

from __future__ import annotations

import logging

from nyx.common.outbox_dispatcher import run_once
from nyx.tasks.base import NyxTask, app

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.maintenance.dispatch_outbox",
    acks_late=True,
)
def dispatch_outbox(self) -> str:
    """Run a single transactional outbox dispatch iteration."""

    processed = run_once()
    logger.debug("Outbox dispatcher processed %s envelopes", processed)
    return f"processed:{processed}"


__all__ = ["dispatch_outbox"]
