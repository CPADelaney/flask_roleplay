"""Celery tasks for evaluating generative drafts."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence

from nyx.common.evals import (
    EvaluatorResult,
    combine_results,
    eval_coherence,
    eval_consistency,
    eval_policy,
)
from nyx.tasks.base import NyxTask, app

logger = logging.getLogger(__name__)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s: %s", name, raw)
        return default


def get_default_min_score() -> float:
    """Resolve the global minimum passing score."""

    return _get_env_float("NYX_EVAL_MIN_SCORE_DEFAULT", 0.7)


def get_blocking_flags() -> Sequence[str]:
    """Return blocking flag identifiers configured for evaluations."""

    raw = os.getenv("NYX_EVAL_BLOCKING_FLAGS", "")
    if not raw:
        return []
    return [flag.strip() for flag in raw.split(",") if flag.strip()]


def evaluate_text(
    text: str,
    *,
    context: Dict[str, Any] | None = None,
    canon_facts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run all evaluators and return the aggregate result."""

    context = context or {}
    canon_facts = canon_facts or {}

    results: List[EvaluatorResult] = [
        eval_coherence(text, context),
        eval_consistency(text, canon_facts),
        eval_policy(text),
    ]

    score, flags = combine_results(*results)
    notes = "\n".join(filter(None, (result.get("notes") for result in results)))
    return {"score": score, "flags": list(flags), "notes": notes, "details": results}


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.eval_text",
    acks_late=True,
)
def eval_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Celery entrypoint that delegates to :func:`evaluate_text`."""

    payload = dict(payload or {})
    text = str(payload.get("text") or "")
    context = payload.get("context") or {}
    canon_facts = payload.get("canon_facts") or {}

    result = evaluate_text(text, context=context, canon_facts=canon_facts)
    logger.debug(
        "Evaluated text", extra={"score": result.get("score"), "flags": result.get("flags")}
    )
    return result


__all__ = ["eval_text", "evaluate_text", "get_blocking_flags", "get_default_min_score"]
