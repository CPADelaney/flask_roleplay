"""Helpers for deriving resolver policies from Nyx context metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .types import Anchor, Scope

_ALLOWED_POLICIES = {"allow", "ask", "deny"}


def _clean(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _infer_scope(anchor: Optional[Anchor], setting_context: Optional[Dict[str, Any]]) -> Scope:
    if anchor and anchor.scope:
        return anchor.scope
    if isinstance(setting_context, dict):
        scope = setting_context.get("scope") or setting_context.get("kind") or setting_context.get("type")
        cleaned = _clean(scope)
        if cleaned in {"real", "fictional", "hybrid"}:
            return cleaned  # type: ignore[return-value]
    return "real"


def resolver_policy_for_context(
    setting_context: Optional[Dict[str, Any]],
    *,
    anchor: Optional[Anchor] = None,
) -> Dict[str, Any]:
    """Return resolver policy metadata for hierarchy minting.

    The response contains:
        ``mint_policy`` – allow/ask/deny for minting fictional places.
        ``default_planet`` – fallback world/planet name when the candidate lacks one.
        ``allow_fictional`` – whether fictional locations are generally allowed.
    """

    world_model: Dict[str, Any] = {}
    if isinstance(setting_context, dict):
        wm = setting_context.get("world_model")
        if isinstance(wm, dict):
            world_model = wm

    allow_fictional = bool(world_model.get("allow_fictional_locations"))
    if not allow_fictional and isinstance(setting_context, dict):
        reality = _clean(setting_context.get("reality_context"))
        if reality and reality not in {"normal", "mundane", "realistic"}:
            allow_fictional = True

    resolver_cfg: Dict[str, Any] = {}
    for key in ("resolver", "location_resolver"):
        cfg = world_model.get(key) if isinstance(world_model, dict) else None
        if isinstance(cfg, dict):
            resolver_cfg = cfg
            break

    mint_policy = _clean(
        resolver_cfg.get("fictional_policy")
        or world_model.get("fictional_location_policy")
        or ("allow" if allow_fictional else "deny")
    )
    if not mint_policy or mint_policy not in _ALLOWED_POLICIES:
        mint_policy = "allow" if allow_fictional else "deny"

    default_planet: Optional[str] = None
    candidates = []
    if isinstance(world_model, dict):
        candidates.append(world_model)
    if isinstance(setting_context, dict):
        for key in ("world", "world_meta", "world_info"):
            payload = setting_context.get(key)
            if isinstance(payload, dict):
                candidates.append(payload)
    for candidate in candidates:
        for key in ("planet", "world_name", "name", "label", "title"):
            default_planet = _clean(candidate.get(key))
            if default_planet:
                break
        if default_planet:
            break

    if not default_planet:
        scope = _infer_scope(anchor, setting_context)
        if scope == "fictional":
            default_planet = "Fictional World"
        else:
            default_planet = "Earth"

    return {
        "mint_policy": mint_policy,
        "default_planet": default_planet,
        "allow_fictional": allow_fictional,
    }


__all__ = ["resolver_policy_for_context"]
