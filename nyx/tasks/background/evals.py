"""Celery task entrypoints for evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict

from nyx.common.evals import (
    EvaluatorResult,
    combine_results,
    eval_coherence,
    eval_consistency,
    eval_policy,
)
from nyx.tasks.base import NyxTask, app
from nyx.telemetry.metrics import EVAL_SCORE


def _aggregate_notes(*results: EvaluatorResult) -> str:
    parts = [result.get("notes", "") for result in results if result.get("notes")]
    return " | ".join(parts)


def evaluate_text(
    text: str,
    *,
    context: Dict[str, Any] | None = None,
    canon_facts: Dict[str, Any] | None = None,
    kind: str = "generic",
) -> Dict[str, Any]:
    """Execute all evaluators and return a combined result."""

    coherence = eval_coherence(text, context or {})
    policy = eval_policy(text)
    consistency = eval_consistency(text, canon_facts or {})
    score, flags = combine_results(coherence, policy, consistency)

    notes = _aggregate_notes(coherence, policy, consistency)
    details = {
        "coherence": coherence,
        "policy": policy,
        "consistency": consistency,
    }

    if score is not None:
        EVAL_SCORE.labels(kind=kind).observe(score)

    return {"score": score, "flags": flags, "notes": notes, "details": details}


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.eval_text",
    acks_late=True,
)
def eval_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task wrapper for :func:`evaluate_text`."""

    payload = dict(payload or {})
    text = str(payload.get("text") or "")
    context = payload.get("context") or {}
    canon_facts = payload.get("canon_facts") or {}
    kind = payload.get("kind") or "generic"

    return evaluate_text(text, context=context, canon_facts=canon_facts, kind=kind)


__all__ = ["eval_text", "evaluate_text"]
