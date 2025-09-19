"""Context helpers for mundane feasibility evaluation."""
from typing import Any, Dict


def build_affordance_index(world_caps: Dict[str, Any], scene: Dict[str, Any], player: Dict[str, Any]) -> Dict[str, Any]:
    """Derive quick affordance checks from world, scene, and player state."""

    location_tags = scene.get("location_tags") or []
    nearby = scene.get("nearby") or {}

    idx = {
        "in_shop": bool(location_tags.count("shop") > 0 or scene.get("has_vendor")),
        "nearby_shops": nearby.get("shops", []),
        "open_shop": bool(scene.get("open_shop")),
    }

    if world_caps.get("economy", {}).get("has_currency"):
        balance = player.get("currency", {})
        idx["has_currency"] = any(balance.values()) if isinstance(balance, dict) else False
    else:
        idx["has_currency"] = bool(world_caps.get("economy", {}).get("bartering_ok"))

    idx["balance"] = player.get("currency", {}) if isinstance(player.get("currency"), dict) else {}
    idx["age_ok"] = player.get("age_ok", True)
    idx["reputation"] = player.get("reputation", 0)
    affordances = world_caps.get("affordances", set())
    prohibitions = world_caps.get("prohibitions", set())
    idx["trade_supported"] = "trade" in affordances and "trade" not in prohibitions
    idx["analogs"] = world_caps.get("analogs", {})
    idx["infra"] = world_caps.get("infra", {})
    return idx
