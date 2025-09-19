"""Roman Empire inspired capability archetype."""

from .base import Archetype, ArchetypeCaps


class RomanEmpire(Archetype):
    """Classical Roman-era baseline."""

    name = "roman_empire"

    def caps(self) -> ArchetypeCaps:
        return ArchetypeCaps(
            infra={
                "printing": False,
                "electricity": False,
                "plumbing": True,
                "global_trade": True,
            },
            economy={
                "has_currency": True,
                "bartering_ok": True,
                "markets_common": True,
            },
            affordances={"mundane_action", "trade", "movement", "social"},
            prohibitions={"firearm_use", "microprocessors", "instant_comms"},
            analogs={
                "24h_convenience_store": "market_square",
                "phone": "letter",
            },
        )
