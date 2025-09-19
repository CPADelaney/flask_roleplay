"""Trade action evaluation helpers."""
from typing import Any, Dict, List


def price_estimate(item_name: str, world_caps: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
    """Very small heuristic price estimator used when catalogs are unavailable."""

    economy = world_caps.get("economy", {})
    currency = "credits" if economy.get("credits") else "denarii"
    amount = 1 if item_name and "gum" in item_name.lower() else 5
    return {"currency": currency, "amount": amount}


def evaluate_buy(
    intent: Dict[str, Any],
    world_caps: Dict[str, Any],
    idx: Dict[str, Any],
    scene: Dict[str, Any],
    player: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a mundane "buy" intent with contextual nudges."""

    if "trade" in world_caps.get("prohibitions", set()):
        return _deny("Trading is impossible in this setting.")

    if not idx.get("trade_supported", False):
        analog_loc = idx.get("analogs", {}).get("24h_convenience_store") or idx.get("analogs", {}).get("market_square")
        if analog_loc:
            return _analog(f"Try the {analog_loc}.", {"redirect_location_type": analog_loc})
        return _defer(
            "Trading isn’t established here. Try heading to a market or commissary nearby.",
            leads=_nearby_leads(idx),
        )

    if not idx.get("in_shop", False):
        leads = _nearby_leads(idx)
        if leads:
            return _defer("You’re not in a shop. Nearest options:", leads=leads)
        analog_loc = idx.get("analogs", {}).get("market_square") or idx.get("analogs", {}).get("habitat_commissary")
        if analog_loc:
            return _defer(f"Find a {analog_loc} to buy that.", leads=[{"kind": "district", "name": analog_loc}])
        return _defer("No vendor here—look for a market, stall, or vendor first.")

    direct_object = intent.get("direct_object") or ["item"]
    item = direct_object[0]
    estimate = price_estimate(item, world_caps, scene)

    if not idx.get("has_currency", False):
        if world_caps.get("economy", {}).get("bartering_ok"):
            return _allow(f"You can barter for {item}. Offer an item of similar value.")
        return _defer(f"You need currency to buy {item}.", leads=_nearby_atm_or_exchange(scene))

    balance = idx.get("balance", {}).get(estimate["currency"], 0)
    if balance < estimate["amount"]:
        return _defer(
            f"Price ~{estimate['amount']} {estimate['currency']}, you have {balance}.",
            leads=[{"kind": "earn", "name": "Take a small job"}, {"kind": "exchange", "name": "Currency exchange"}],
        )

    return _allow(
        f"You purchase {item} for {estimate['amount']} {estimate['currency']}.",
        modifications={
            "inventory_add": [item],
            "currency_debit": {estimate["currency"]: estimate["amount"]},
        },
    )


def _allow(msg: str, modifications: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "feasible": True,
        "strategy": "allow",
        "narrator_guidance": msg,
        "modifications": modifications or {},
    }


def _defer(msg: str, leads: Any | None = None) -> Dict[str, Any]:
    return {
        "feasible": False,
        "strategy": "defer",
        "narrator_guidance": msg,
        "leads": leads or [],
    }


def _analog(msg: str, modifications: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "feasible": False,
        "strategy": "analog",
        "narrator_guidance": msg,
        "modifications": modifications or {},
    }


def _deny(msg: str) -> Dict[str, Any]:
    return {
        "feasible": False,
        "strategy": "deny",
        "violations": [{"rule": "established_impossibility", "reason": msg}],
    }


def _nearby_leads(idx: Dict[str, Any]) -> Any:
    leads: List[Dict[str, Any]] = []
    for entry in idx.get("nearby_shops", [])[:3]:
        if isinstance(entry, dict):
            leads.append(
                {
                    "kind": "shop",
                    "name": entry.get("name"),
                    "dist_m": entry.get("dist_m"),
                }
            )
        else:
            try:
                _identifier, name, dist = entry
            except Exception:
                continue
            leads.append({"kind": "shop", "name": name, "dist_m": dist})
    return leads


def _nearby_atm_or_exchange(scene: Dict[str, Any]) -> Any:
    return scene.get("nearby", {}).get("exchanges", []) or []
