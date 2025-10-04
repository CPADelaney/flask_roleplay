"""Lightweight helpers shared by the SDK and orchestrator feasibility gates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


_NYX_TAUNT_PREFIXES: Sequence[str] = (
    "Oh, pet,",
    "Mmm, kitten,",
    "Sweet thing,",
)


def _choose_prefix(violations: Sequence[Dict[str, Any]]) -> str:
    """Pick a playful prefix to keep Nyx's voice varied."""

    if not violations:
        return _NYX_TAUNT_PREFIXES[0]

    idx = sum(len(v.get("reason", "")) for v in violations) % len(_NYX_TAUNT_PREFIXES)
    return _NYX_TAUNT_PREFIXES[idx]


def _clean_sentences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "The scene isn't ready for that little fantasy."
    if text[-1] not in ".!?":
        return f"{text}."
    return text


def _collect_reasons(violations: Iterable[Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    for violation in violations:
        reason = violation.get("reason")
        if isinstance(reason, str) and reason.strip():
            reasons.append(reason.strip())
            continue
        rule = violation.get("rule")
        if isinstance(rule, str) and rule.strip():
            reasons.append(rule.strip())
    if not reasons:
        reasons.append("reality is still missing the groundwork")
    return reasons


def _format_clause(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _build_persona_guidance(guidance: str, violations: List[Dict[str, Any]], leads: Sequence[str]) -> str:
    prefix = _choose_prefix(violations)
    grounded = _clean_sentences(guidance)
    reasons = _collect_reasons(violations)
    reason_clause = _format_clause(reasons)

    persona_lines = [
        f"{prefix} slow down. {grounded}",
        f"Reality keeps its heel on you because {reason_clause}.",
    ]

    if leads:
        lead_clause = _format_clause([lead.strip() for lead in leads if isinstance(lead, str) and lead.strip()])
        if lead_clause:
            persona_lines.append(f"Earn it for me first by {lead_clause}.")

    return " ".join(persona_lines)


def extract_defer_details(feasibility: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
    """Return guidance, suggested steps, and metadata extras for a defer strategy."""

    overall = feasibility.get("overall") or {}
    if (overall.get("strategy") or "").lower() != "defer":
        return "", [], {}

    per_intent = feasibility.get("per_intent") or []
    first = per_intent[0] if per_intent and isinstance(per_intent[0], dict) else {}

    base_guidance = first.get("narrator_guidance") or "Ground the attempt in what this reality actually provides."
    leads = first.get("leads") or first.get("suggested_alternatives") or []
    violations = first.get("violations") or []

    persona_text = _build_persona_guidance(base_guidance, list(violations), leads)

    extra_meta: Dict[str, Any] = {
        "leads": leads,
        "violations": violations,
    }

    return persona_text, list(leads), extra_meta
