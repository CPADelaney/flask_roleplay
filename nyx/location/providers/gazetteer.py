"""Adapters for gazetteer/geocoding providers (Nominatim, etc.)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..nominatim_map import nominatim_to_admin_path
from ..types import Candidate, Place


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def candidate_from_nominatim(payload: Dict[str, Any]) -> Optional[Candidate]:
    """Convert a Nominatim response row into a :class:`Candidate`."""

    if not isinstance(payload, dict):
        return None

    lat = _coerce_float(payload.get("lat"))
    lon = _coerce_float(payload.get("lon"))
    if lat is None or lon is None:
        return None

    importance = _coerce_float(payload.get("importance")) or 0.0
    name = _clean(payload.get("name"))
    if not name:
        display_name = _clean(payload.get("display_name")) or "Unknown"
        name = display_name.split(",")[0].strip() if "," in display_name else display_name
        if not name:
            name = "Unknown"

    address_src = payload.get("address") or {}
    address: Dict[str, Any] = dict(address_src) if isinstance(address_src, dict) else {}
    normalized_path = nominatim_to_admin_path(address)
    if normalized_path:
        address["_normalized_admin_path"] = normalized_path

    place = Place(
        name=name,
        level="venue",
        key=str(payload.get("place_id") or payload.get("osm_id") or name.lower()),
        lat=lat,
        lon=lon,
        address=address,
        meta={
            "category": payload.get("category") or payload.get("type"),
            "source": "nominatim",
            "importance": importance,
            "display_name": payload.get("display_name"),
        },
    )

    confidence = min(0.99, 0.5 + importance / 2.0)

    return Candidate(
        place=place,
        confidence=confidence,
        raw=payload,
    )


__all__ = ["candidate_from_nominatim"]
