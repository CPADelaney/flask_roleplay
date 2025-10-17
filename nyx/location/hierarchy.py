"""Utilities for persisting normalized location hierarchies."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import asyncpg

from .types import Anchor, Candidate, PlaceEdge, Scope

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Order for composing deterministic admin paths / identifiers.
_ADMIN_LEVEL_ORDER: Tuple[str, ...] = (
    "world",
    "country",
    "region",
    "state",
    "city",
    "district",
    "neighborhood",
    "venue",
)

# Mapping from canonical place level to keys emitted by the nominatim normalizer.
_LEVEL_KEYS: Dict[str, Sequence[str]] = {
    "country": ("country",),
    "region": ("region",),
    "state": ("state",),
    "city": ("city",),
    "district": ("district", "county"),
    "neighborhood": ("neighborhood",),
}


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = _SLUG_RE.sub("-", value)
    return value.strip("-")


def _make_id(parts: Iterable[str]) -> str:
    """Return a deterministic identifier composed from ``parts``."""

    tokens: List[str] = []
    for part in parts:
        if not part:
            continue
        slug = _slugify(str(part))
        if slug:
            tokens.append(slug)
    if not tokens:
        raise ValueError("Cannot compose id from empty parts")
    return "::".join(tokens)


async def _get_or_create_place(
    conn: asyncpg.Connection,
    *,
    scope: Scope,
    level: str,
    name: str,
    admin_path: Dict[str, str],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure a place row exists and return identifying fields."""

    ordered_path: List[str] = []
    for lvl in _ADMIN_LEVEL_ORDER:
        if lvl == level:
            continue
        val = admin_path.get(lvl)
        if val:
            ordered_path.append(f"{lvl}:{val}")

    place_key = _make_id([scope, level, name, *ordered_path])
    normalized_name = _slugify(name)

    row = await conn.fetchrow(
        """
        INSERT INTO Places (scope, place_key, name, normalized_name, level, admin_path,
                            latitude, longitude, meta)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (place_key) DO UPDATE
        SET name = EXCLUDED.name,
            level = EXCLUDED.level,
            admin_path = COALESCE(Places.admin_path, EXCLUDED.admin_path),
            latitude = COALESCE(Places.latitude, EXCLUDED.latitude),
            longitude = COALESCE(Places.longitude, EXCLUDED.longitude),
            meta = COALESCE(Places.meta, '{}'::jsonb) || EXCLUDED.meta,
            updated_at = NOW()
        RETURNING id, place_key
        """,
        scope,
        place_key,
        name,
        normalized_name or name.lower(),
        level,
        admin_path or {},
        lat,
        lon,
        meta or {},
    )

    assert row is not None
    return {"id": row["id"], "place_key": row["place_key"]}


async def _link(
    conn: asyncpg.Connection,
    parent_id: int,
    child_id: int,
    *,
    kind: str = "contains",
    distance_km: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Create or refresh an edge between two places."""

    if parent_id == child_id:
        return

    await conn.execute(
        """
        INSERT INTO PlaceEdges (parent_id, child_id, kind, distance_km, meta)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (parent_id, child_id, kind) DO UPDATE
        SET distance_km = COALESCE(PlaceEdges.distance_km, EXCLUDED.distance_km),
            meta = COALESCE(PlaceEdges.meta, '{}'::jsonb) || EXCLUDED.meta,
            updated_at = NOW()
        """,
        parent_id,
        child_id,
        kind,
        distance_km,
        meta or {},
    )


async def assign_hierarchy(
    conn: asyncpg.Connection,
    candidate: Candidate,
    *,
    scope: Scope = "real",
    anchor: Optional[Anchor] = None,
    mint_policy: Optional[str] = None,
    default_planet: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a candidate's hierarchy chain and return identifiers."""

    address = candidate.place.address or {}
    normalized = address.get("_normalized_admin_path") or {}

    candidate.place.meta = dict(candidate.place.meta or {})
    meta = candidate.place.meta

    if mint_policy and "resolver_mint_policy" not in meta:
        meta["resolver_mint_policy"] = mint_policy

    world_name = meta.get("world_name")
    if not world_name and anchor and anchor.world_name:
        world_name = anchor.world_name
    if not world_name and default_planet:
        world_name = default_planet
    if not world_name:
        world_name = "Earth" if scope == "real" else "Fictional World"

    path_map: Dict[str, str] = {}
    chain: List[Dict[str, Any]] = []

    path_map["world"] = world_name
    meta.setdefault("world_name", world_name)
    if default_planet and "default_planet" not in meta:
        meta["default_planet"] = default_planet
    world_node = await _get_or_create_place(
        conn,
        scope=scope,
        level="world",
        name=world_name,
        admin_path=dict(path_map),
        meta={"scope": scope},
    )
    chain.append({"level": "world", "name": world_name, **world_node})

    seen: set[Tuple[str, str]] = {("world", world_name.lower())}

    # Use anchor hints for coarse geography before processing normalized address.
    if anchor:
        for level, value in (
            ("country", anchor.country),
            ("region", anchor.region),
            ("city", anchor.primary_city),
        ):
            if value and level not in path_map:
                path_map[level] = value

    for level, keys in _LEVEL_KEYS.items():
        name: Optional[str] = None
        for key in keys:
            candidate_name = normalized.get(key)
            if candidate_name:
                name = candidate_name
                break
        if not name:
            continue
        slug = (level, name.lower())
        if slug in seen:
            continue
        seen.add(slug)
        path_map[level] = name
        node = await _get_or_create_place(
            conn,
            scope=scope,
            level=level,
            name=name,
            admin_path=dict(path_map),
            meta={"scope": scope},
        )
        chain.append({"level": level, "name": name, **node})

    place_level = candidate.place.level
    place_name = candidate.place.name
    if place_level not in _ADMIN_LEVEL_ORDER:
        path_map["venue"] = place_name
    else:
        path_map[place_level] = place_name

    place_node = await _get_or_create_place(
        conn,
        scope=scope,
        level=place_level,
        name=place_name,
        admin_path=dict(path_map),
        lat=candidate.place.lat,
        lon=candidate.place.lon,
        meta=candidate.place.meta,
    )
    chain.append({"level": place_level, "name": place_name, **place_node})

    edges: List[PlaceEdge] = []
    for parent, child in zip(chain, chain[1:]):
        await _link(
            conn,
            parent["id"],
            child["id"],
            kind="contains",
            meta={"scope": scope},
        )
        edges.append(
            PlaceEdge(
                source=str(parent["place_key"]),
                target=str(child["place_key"]),
                kind="contains",
            )
        )

    candidate.edges = edges + list(candidate.edges or [])

    candidate.place.meta.setdefault("place_key", place_node["place_key"])
    candidate.place.meta.setdefault("place_id", place_node["id"])

    return {
        "chain": chain,
        "leaf": place_node,
        "world_name": world_name,
        "mint_policy": mint_policy,
    }


__all__ = [
    "assign_hierarchy",
]
