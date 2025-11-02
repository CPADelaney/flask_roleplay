"""Evaluation helpers for gating generative Nyx outputs."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, TypedDict


class EvaluatorResult(TypedDict, total=False):
    """Container for individual evaluator outputs."""

    score: float
    flags: List[str]
    notes: str


_DEFICIT_PENALTIES: Dict[str, float] = {
    "empty": 1.0,
    "short": 0.2,
    "repetition": 0.3,
    "policy": 1.0,
    "canon_miss": 0.4,
    "canon_conflict": 0.6,
}


def _normalise_text(text: str) -> str:
    return (text or "").strip()


def eval_coherence(text: str, context: Dict[str, Any] | None = None) -> EvaluatorResult:
    """Lightweight coherence heuristic.

    The heuristic penalises empty drafts, extremely short drafts, and high
    repetition ratios.  Context is currently unused but reserved for future
    rubric-based checks.
    """

    context = context or {}
    processed = _normalise_text(text)
    flags: List[str] = []
    notes: List[str] = []

    if not processed:
        return {"score": 0.0, "flags": ["empty_text"], "notes": "Draft is empty"}

    tokens = processed.split()
    score = 1.0

    if len(tokens) < int(context.get("min_tokens", 20)):
        score -= _DEFICIT_PENALTIES["short"]
        notes.append("Draft is very short")

    sentences = [s.strip() for s in re.split(r"[.!?]+", processed) if s.strip()]
    if sentences:
        unique_sentences = len(set(sentences))
        repetition_ratio = 1.0 - (unique_sentences / len(sentences))
        if repetition_ratio > 0.5:
            score -= _DEFICIT_PENALTIES["repetition"]
            flags.append("repetition")
            notes.append(f"High repetition ({repetition_ratio:.2f})")

    score = max(0.0, min(1.0, score))
    return {"score": score, "flags": flags, "notes": "; ".join(notes) if notes else ""}


_POLICY_BLOCKLIST = {
    "kill yourself",
    "suicide",
    "self harm",
    "nsfw minor",
}


def eval_policy(text: str) -> EvaluatorResult:
    """Basic content policy guard.

    If a more sophisticated moderation guard is available it should replace
    this stub.  Until then we run a keyword check and flag obvious policy
    violations.
    """

    processed = _normalise_text(text).lower()
    for phrase in _POLICY_BLOCKLIST:
        if phrase in processed:
            return {
                "score": 0.0,
                "flags": ["policy_violation"],
                "notes": f"Blocked phrase detected: {phrase}",
            }
    return {"score": 1.0, "flags": [], "notes": ""}


def eval_consistency(text: str, canon_facts: Dict[str, Any] | None = None) -> EvaluatorResult:
    """Verify the draft against provided canon facts.

    ``canon_facts`` may include ``required_keywords`` or ``prohibited_keywords``
    entries.  When absent the check falls back to a neutral pass.
    """

    canon_facts = canon_facts or {}
    processed = _normalise_text(text).lower()
    score = 1.0
    flags: List[str] = []
    notes: List[str] = []

    required = [str(item).lower() for item in canon_facts.get("required_keywords", [])]
    if required:
        missing = [kw for kw in required if kw not in processed]
        if missing:
            score -= _DEFICIT_PENALTIES["canon_miss"]
            flags.append("missing_canon")
            notes.append(f"Missing canon keywords: {', '.join(sorted(set(missing)))}")

    prohibited = [str(item).lower() for item in canon_facts.get("prohibited_keywords", [])]
    violations = [kw for kw in prohibited if kw and kw in processed]
    if violations:
        score -= _DEFICIT_PENALTIES["canon_conflict"]
        flags.append("canon_conflict")
        notes.append(f"Conflicts with canon keywords: {', '.join(sorted(set(violations)))}")

    score = max(0.0, min(1.0, score))
    return {"score": score, "flags": flags, "notes": "; ".join(notes) if notes else ""}


def combine_results(*results: EvaluatorResult) -> Tuple[float, List[str]]:
    """Combine individual evaluator outputs with a weighted average.

    All evaluators currently share equal weighting.  Flags are aggregated and
    deduplicated preserving insertion order.
    """

    scores: List[float] = []
    flags: List[str] = []
    seen_flags: set[str] = set()

    for result in results:
        if not isinstance(result, dict):
            continue
        score = result.get("score")
        if isinstance(score, (int, float)) and not math.isnan(score):
            scores.append(float(score))
        for flag in result.get("flags", []) or []:
            if flag not in seen_flags:
                seen_flags.add(flag)
                flags.append(flag)

    average = sum(scores) / len(scores) if scores else 0.0
    return average, flags


def load_blocking_flags(default: Iterable[str] | None = None) -> List[str]:
    """Return blocking flags from configuration."""

    env_value = os.getenv("NYX_EVAL_BLOCKING_FLAGS")
    if env_value:
        parsed = [flag.strip() for flag in env_value.split(",") if flag.strip()]
    else:
        parsed = []
    if default:
        for flag in default:
            if flag not in parsed:
                parsed.append(flag)
    return parsed


__all__ = [
    "EvaluatorResult",
    "eval_coherence",
    "eval_consistency",
    "eval_policy",
    "combine_results",
    "load_blocking_flags",
]
