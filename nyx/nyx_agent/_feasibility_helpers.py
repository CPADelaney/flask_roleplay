"""Lightweight helpers shared by the SDK and orchestrator feasibility gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_NYX_TAUNT_PREFIXES: Sequence[str] = (
    "Oh, pet,",
    "Mmm, kitten,",
    "Sweet thing,",
)


@dataclass
class DeferPromptContext:
    """Structured context for generating a defer narrative via Nyx."""

    narrator_guidance: str
    leads: List[str]
    violations: List[Dict[str, Any]]
    persona_prefix: str
    reason_phrases: List[str]


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


def _normalize_str_items(items: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
    return normalized


def extract_defer_details(feasibility: Dict[str, Any]) -> Tuple[Optional[DeferPromptContext], Dict[str, Any]]:
    """Return structured context and metadata extras for a defer strategy."""

    overall = feasibility.get("overall") or {}
    if (overall.get("strategy") or "").lower() != "defer":
        return None, {}

    per_intent = feasibility.get("per_intent") or []
    first = per_intent[0] if per_intent and isinstance(per_intent[0], dict) else {}

    base_guidance_raw = first.get("narrator_guidance") or "Ground the attempt in what this reality actually provides."
    leads = _normalize_str_items(first.get("leads") or first.get("suggested_alternatives") or [])
    violations = list(first.get("violations") or [])

    prefix = _choose_prefix(violations)
    grounded_guidance = _clean_sentences(base_guidance_raw)
    reasons = _collect_reasons(violations)

    context = DeferPromptContext(
        narrator_guidance=grounded_guidance,
        leads=leads,
        violations=violations,
        persona_prefix=prefix,
        reason_phrases=reasons,
    )

    extra_meta: Dict[str, Any] = {
        "leads": leads,
        "violations": violations,
    }

    return context, extra_meta


def build_defer_prompt(context: DeferPromptContext) -> str:
    """Create a lightweight prompt instructing Nyx to taunt while listing violations."""

    violation_lines = "\n".join(f"- {reason}" for reason in context.reason_phrases)
    lead_lines = "\n".join(f"- {lead}" for lead in context.leads) if context.leads else "- None offered"

    instructions = (
        "You are Nyx, a teasing, dominant narrator. Craft a short taunt (2-3 sentences) "
        "that stays playful but firm. Start with the provided persona prefix exactly once. "
        "Explicitly reference the missing prerequisites and encourage the player to pursue the suggested leads."
    )

    return (
        f"{instructions}\n"
        f"Persona prefix: {context.persona_prefix}\n"
        f"Narrator guidance: {context.narrator_guidance}\n"
        f"Missing prerequisites:\n{violation_lines}\n"
        f"Suggested leads:\n{lead_lines}\n"
        "Respond as Nyx in second person, keeping the tone sharp and indulgent."
    )


def build_defer_fallback_text(context: DeferPromptContext) -> str:
    """Assemble a static Nyx-styled message if the agent prompt generation fails."""

    reason_clause = _format_clause(context.reason_phrases)
    persona_lines = [
        f"{context.persona_prefix} slow down. {context.narrator_guidance}",
        f"Reality keeps its heel on you because {reason_clause}.",
    ]

    if context.leads:
        lead_clause = _format_clause(context.leads)
        if lead_clause:
            persona_lines.append(f"Earn it for me first by {lead_clause}.")

    return " ".join(persona_lines)


def coalesce_agent_output_text(result: Any) -> Optional[str]:
    """Best-effort extraction of the last textual chunk from a Runner result."""

    if result is None:
        return None

    potential_sequences: List[Sequence[Any]] = []
    for attr in ("messages", "history", "events"):
        value = getattr(result, attr, None)
        if isinstance(value, Sequence) and value:
            potential_sequences.append(value)
        elif isinstance(result, dict):
            value = result.get(attr)
            if isinstance(value, Sequence) and value:
                potential_sequences.append(value)

    for sequence in potential_sequences:
        for item in reversed(sequence):
            text: Optional[str] = None
            if isinstance(item, dict):
                text = item.get("content") or item.get("text")
            else:
                text = getattr(item, "content", None) or getattr(item, "text", None)
                if text is None and isinstance(item, str):
                    text = item
            if text:
                stripped = str(text).strip()
                if stripped:
                    return stripped

    for attr in ("content", "text"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(result, str) and result.strip():
        return result.strip()

    return None
