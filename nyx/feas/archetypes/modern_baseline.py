"""Neutral modern-day archetype used as fallback."""

from .base import Archetype, ArchetypeCaps


class ModernBaseline(Archetype):
    """Provides permissive defaults for contemporary settings."""

    name = "modern_baseline"

    def caps(self) -> ArchetypeCaps:
        return ArchetypeCaps(
            infra={
                "electricity": True,
                "printing": True,
                "instant_comms": True,
                "plumbing": True,
                "global_trade": True,
            },
            economy={
                "has_currency": True,
                "bartering_ok": True,
                "markets_common": True,
                "credits": False,
            },
            affordances={"mundane_action", "trade", "movement", "social"},
            prohibitions=set(),
            analogs={
                "market_square": "shopping_district",
                "24h_convenience_store": "24h_convenience_store",
            },
        )
