"""Lightweight evaluation helpers for generative outputs."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple, TypedDict

EvaluatorResult = TypedDict(
    "EvaluatorResult",
    {
        "score": float,
        "flags": List[str],
        "notes": str,
    },
)

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _word_stats(text: str) -> Tuple[int, int]:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return len(words), len(sentences)


def eval_coherence(text: str, context: Dict[str, object] | None = None) -> EvaluatorResult:
    """Heuristic coherence evaluator.

    The implementation intentionally avoids external dependencies and focuses on a
    few observable text properties: non-emptiness, sentence count, and repetition.
    """

    context = context or {}
    cleaned = text.strip()
    word_count, sentence_count = _word_stats(cleaned)

    if not cleaned:
        return {"score": 0.0, "flags": ["empty_text"], "notes": "coherence: empty draft"}

    unique_words = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", cleaned)}
    lexical_diversity = len(unique_words) / max(word_count, 1)

    base_score = 0.4 + 0.6 * _clamp(word_count / 180.0)
    sentence_bonus = 0.1 if sentence_count > 1 else -0.1
    diversity_bonus = 0.2 * lexical_diversity

    score = _clamp(base_score + sentence_bonus + diversity_bonus)

    flags: List[str] = []
    notes = f"coherence: words={word_count} sentences={sentence_count} diversity={lexical_diversity:.2f}"

    if sentence_count <= 1:
        flags.append("low_sentence_count")
    if lexical_diversity < 0.2:
        flags.append("repetition")
    if context.get("expected_length") and word_count < int(context["expected_length"]):
        flags.append("too_short")

    return {"score": score, "flags": flags, "notes": notes}


def eval_policy(text: str) -> EvaluatorResult:
    """Placeholder policy evaluation.

    Integrations can swap this out for a moderation endpoint. By default it
    approves all content while tagging obviously disallowed tokens.
    """

    lower = text.lower()
    flagged = []
    if any(term in lower for term in ("forbidden", "banned_phrase")):
        flagged.append("policy_violation")

    score = 0.0 if flagged else 1.0
    notes = "policy: flagged" if flagged else "policy: pass"
    return {"score": score, "flags": flagged, "notes": notes}


def eval_consistency(text: str, canon_facts: Dict[str, object] | None = None) -> EvaluatorResult:
    """Check that important canon facts are reflected in the draft."""

    canon_facts = canon_facts or {}
    if not canon_facts:
        return {"score": 1.0, "flags": [], "notes": "consistency: no canon facts"}

    lower_text = text.lower()
    missing: List[str] = []

    for key, value in canon_facts.items():
        if isinstance(value, str) and value:
            if value.lower() not in lower_text:
                missing.append(key)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not any(str(item).lower() in lower_text for item in value):
                missing.append(key)

    coverage = 1.0 - (len(missing) / max(len(canon_facts), 1))
    score = _clamp(coverage)
    flags = ["consistency_gap"] if missing else []
    notes = (
        f"consistency: missing={','.join(missing)}"
        if missing
        else "consistency: all canon references present"
    )
    return {"score": score, "flags": flags, "notes": notes}


def combine_results(*results: EvaluatorResult) -> Tuple[float, List[str]]:
    """Combine multiple evaluator outputs into a weighted score and flags."""

    if not results:
        return 0.0, []

    scores: List[float] = []
    flags: List[str] = []

    for result in results:
        score = _clamp(result.get("score", 0.0))
        scores.append(score)
        for flag in result.get("flags", []):
            if flag not in flags:
                flags.append(flag)

    average = sum(scores) / len(scores) if scores else 0.0
    return average, flags


__all__ = [
    "EvaluatorResult",
    "combine_results",
    "eval_coherence",
    "eval_consistency",
    "eval_policy",
]
