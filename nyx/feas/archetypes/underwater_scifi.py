"""Futuristic underwater habitat archetype."""

from .base import Archetype, ArchetypeCaps


class UnderwaterSciFi(Archetype):
    """Supports high-tech underwater settlements."""

    name = "underwater_scifi"

    def caps(self) -> ArchetypeCaps:
        return ArchetypeCaps(
            infra={
                "electricity": True,
                "printing": True,
                "instant_comms": True,
                "airlock": True,
            },
            economy={
                "has_currency": True,
                "markets_common": False,
                "credits": True,
            },
            affordances={
                "mundane_action",
                "trade",
                "movement",
                "social",
                "vehicle_operation_water",
            },
            prohibitions=set(),
            analogs={"market_square": "habitat_commissary"},
        )
