"""Lightweight helpers shared by the SDK and orchestrator feasibility gates."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def extract_defer_details(feasibility: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
    """Return guidance, suggested steps, and metadata extras for a defer strategy."""

    overall = feasibility.get("overall") or {}
    if (overall.get("strategy") or "").lower() != "defer":
        return "", [], {}

    per_intent = feasibility.get("per_intent") or []
    first = per_intent[0] if per_intent and isinstance(per_intent[0], dict) else {}

    guidance = first.get("narrator_guidance") or "Let's ground that action before we continue."
    leads = first.get("leads") or first.get("suggested_alternatives") or []
    violations = first.get("violations") or []

    extra_meta: Dict[str, Any] = {
        "leads": leads,
        "violations": violations,
    }

    return guidance, list(leads), extra_meta
