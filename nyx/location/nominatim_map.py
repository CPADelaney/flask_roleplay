"""Helpers for translating Nominatim address payloads into Nyx hierarchy keys."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _clean(value: Any) -> Optional[str]:
    if isinstance(value, str):
        s = value.strip()
        if s:
            return s
    return None


def _first(addr: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        val = _clean(addr.get(key))
        if val:
            return val
    return None


def _compose_building(addr: Dict[str, Any]) -> Optional[str]:
    house_name = _clean(addr.get("house_name")) or _clean(addr.get("house"))
    if house_name:
        return house_name
    housenum = _clean(addr.get("house_number"))
    street = _first(
        addr,
        (
            "road",
            "pedestrian",
            "residential",
            "street",
            "square",
            "footway",
            "highway",
        ),
    )
    if housenum and street:
        return f"{housenum} {street}"
    if housenum:
        return housenum
    return None


_LEVEL_CANDIDATES = {
    "feature": (
        "attraction",
        "amenity",
        "tourism",
        "historic",
        "leisure",
        "shop",
        "office",
        "man_made",
    ),
    "campus": ("campus", "institution"),
    "building": ("building", "public_building"),
    "street": (
        "road",
        "pedestrian",
        "residential",
        "street",
        "square",
        "footway",
        "highway",
    ),
    "neighborhood": ("neighbourhood", "neighborhood", "microhood"),
    "district": (
        "suburb",
        "quarter",
        "city_district",
        "district",
        "borough",
        "municipality",
    ),
    "city": ("city", "town", "village", "municipality", "hamlet"),
    "county": ("county", "local_authority", "region_municipality"),
    "region": ("state_district", "state", "province", "region"),
    "country": ("country",),
}


def nominatim_to_admin_path(addr: Dict[str, Any]) -> Dict[str, str]:
    """Normalize an OSM address dictionary into Nyx hierarchy keys."""

    if not isinstance(addr, dict):
        return {}

    normalized: Dict[str, str] = {}

    building = _compose_building(addr)
    if building:
        normalized["building"] = building

    for level, keys in _LEVEL_CANDIDATES.items():
        if level == "building" and "building" in normalized:
            continue
        val = _first(addr, keys)
        if val:
            normalized.setdefault(level, val)

    # Ensure city preference when both municipality+city exist
    if "city" not in normalized:
        alt_city = _first(
            addr,
            (
                "city",
                "town",
                "village",
                "hamlet",
                "municipality",
            ),
        )
        if alt_city:
            normalized["city"] = alt_city

    return normalized


__all__ = ["nominatim_to_admin_path"]
