"""Base definitions for feasibility archetypes."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ArchetypeCaps:
    """Small bundle of capabilities supplied by an archetype module."""

    infra: Dict[str, bool] = field(default_factory=dict)
    economy: Dict[str, bool] = field(default_factory=dict)
    affordances: Set[str] = field(default_factory=set)
    prohibitions: Set[str] = field(default_factory=set)
    analogs: Dict[str, str] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)


class Archetype:
    """Interface for capability archetypes."""

    name: str

    def caps(self) -> ArchetypeCaps:
        """Return the capability contribution from this archetype."""

        raise NotImplementedError
