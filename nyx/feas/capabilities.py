"""Capability composition helpers for feasibility."""
from typing import Any, Dict, Iterable

from .archetypes.base import ArchetypeCaps


def merge_caps(arche_caps: Iterable[ArchetypeCaps]) -> Dict[str, Any]:
    """Merge capability contributions from multiple archetypes."""

    combined: Dict[str, Any] = {
        "infra": {},
        "economy": {},
        "affordances": set(),
        "prohibitions": set(),
        "analogs": {},
        "weights": {},
    }

    for caps in arche_caps:
        # Merge infra/economy booleans using logical OR semantics.
        for key, value in caps.infra.items():
            combined["infra"][key] = combined["infra"].get(key, False) or bool(value)
        for key, value in caps.economy.items():
            combined["economy"][key] = combined["economy"].get(key, False) or bool(value)

        combined["affordances"].update(caps.affordances)
        combined["prohibitions"].update(caps.prohibitions)
        combined["analogs"].update(caps.analogs)
        combined["weights"].update(caps.weights)

    return combined
