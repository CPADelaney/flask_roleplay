"""Utilities for persisting normalized location hierarchies."""

from __future__ import annotations

import re
import random
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import logging
import asyncpg

from .nominatim_map import nominatim_to_admin_path
from .types import Anchor, Candidate, Location, PlaceEdge, Scope, DEFAULT_REALM

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

if TYPE_CHECKING:
    from lore.core.context import CanonicalContext

logger = logging.getLogger(__name__)

# Add this helper to avoid circular imports
async def _notify_canon_of_location(
    conn: asyncpg.Connection,
    user_id: int,
    conversation_id: int,
    location: Location,
) -> None:
    """Notify canon system of new location creation."""
    try:
        # Import here to avoid circular dependency
        from lore.core.canon import log_canonical_event, get_canon_memory_orchestrator
        from lore.core.context import CanonicalContext
        from memory.memory_orchestrator import EntityType
        
        ctx = CanonicalContext(user_id=user_id, conversation_id=conversation_id)
        
        # Log the canonical event
        significance = 7 if location.location_type == "city" else 6 if location.location_type == "district" else 5
        
        await log_canonical_event(
            ctx, conn,
            f"Location '{location.location_name}' established in {location.city or 'the world'}",
            tags=['location', 'creation', location.location_type or 'venue', location.scope or 'real'],
            significance=significance,
            persist_memory=True
        )
        
        # Store in memory system with embedding
        memory_orchestrator = await get_canon_memory_orchestrator(user_id, conversation_id)
        
        # Build description for embedding
        description_parts = [location.location_name]
        if location.description:
            description_parts.append(location.description)
        if location.district:
            description_parts.append(f"in {location.district}")
        if location.city:
            description_parts.append(f", {location.city}")
        
        embedding_text = " ".join(description_parts)
        
        # Store as memory
        await memory_orchestrator.store_memory(
            entity_type=EntityType.LORE,
            entity_id=0,
            memory_text=embedding_text,
            significance=0.7 if location.location_type == "venue" else 0.8,
            tags=['location', location.location_type or 'venue', location.scope or 'real'],
            metadata={
                "location_id": location.id,
                "location_name": location.location_name,
                "location_type": location.location_type,
                "city": location.city,
                "district": location.district,
                "scope": location.scope
            }
        )
        
        # Add to vector store for searchability
        await memory_orchestrator.add_to_vector_store(
            text=embedding_text,
            metadata={
                "entity_type": "location",
                "location_id": location.id,
                "location_name": location.location_name,
                "location_type": location.location_type,
                "city": location.city,
                "district": location.district,
                "is_fictional": location.is_fictional,
                "scope": location.scope,
                "user_id": user_id,
                "conversation_id": conversation_id
            },
            entity_type="location"
        )
        
        logger.info(f"✓ Notified canon system of location: {location.location_name}")
        
    except Exception as e:
        # Don't let canon failures break location creation
        logger.warning(f"Failed to notify canon of location creation: {e}", exc_info=True)


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


def _normalize_location_name(value: str) -> str:
    """Return a lower-cased, whitespace-collapsed location identifier."""

    if not isinstance(value, str):
        value = str(value or "")
    normalized = " ".join(value.replace("_", " ").split()).strip().lower()
    return normalized or "unknown location"


def _serialize_admin_path(admin_path: Optional[Dict[str, str]]) -> str:
    """Return a deterministic JSON payload for admin path storage."""

    return json.dumps(admin_path or {}, sort_keys=True)


def _serialize_json_value(value: Any) -> Optional[str]:
    """Serialize JSON-compatible values to text for asyncpg bindings."""

    if value is None:
        return None
    if isinstance(value, (set, tuple)):
        value = list(value)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return None
    return value


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

    serialized_admin_path = _serialize_admin_path(admin_path)

    serialized_meta = json.dumps(meta or {}, sort_keys=True)

    row = await conn.fetchrow(
        """
        INSERT INTO Places (scope, place_key, name, normalized_name, level, admin_path,
                            latitude, longitude, meta)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
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
        serialized_admin_path,
        lat,
        lon,
        serialized_meta,
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

    serialized_meta = json.dumps(meta or {}, sort_keys=True)

    await conn.execute(
        """
        INSERT INTO PlaceEdges (parent_id, child_id, kind, distance_km, meta)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        ON CONFLICT (parent_id, child_id, kind) DO UPDATE
        SET distance_km = COALESCE(PlaceEdges.distance_km, EXCLUDED.distance_km),
            meta = COALESCE(PlaceEdges.meta, '{}'::jsonb) || EXCLUDED.meta,
            updated_at = NOW()
        """,
        parent_id,
        child_id,
        kind,
        distance_km,
        serialized_meta,
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


async def generate_and_persist_hierarchy(
    conn: asyncpg.Connection,
    *,
    user_id: int,
    conversation_id: int,
    candidate: Candidate,
    scope: Scope = "real",
    anchor: Optional[Anchor] = None,
    mint_policy: Optional[str] = None,
    default_planet: Optional[str] = None,
    default_galaxy: Optional[str] = None,
    default_realm: Optional[str] = None,
) -> Location:
    """Persist a candidate as a :class:`Location` with enriched hierarchy data."""

    meta = candidate.place.meta = dict(candidate.place.meta or {})

    # Normalize the address via Nominatim so that admin keys align with Nyx
    # hierarchy expectations.
    address = dict(candidate.place.address or {})
    normalized_admin = address.get("_normalized_admin_path")
    if not normalized_admin:
        normalized_admin = nominatim_to_admin_path(address)
        if normalized_admin:
            address["_normalized_admin_path"] = normalized_admin
    candidate.place.address = address

    admin_path: Dict[str, str] = dict(normalized_admin or {})

    # Enrich with anchor hints for partially specified locations.
    if anchor:
        if anchor.primary_city and not admin_path.get("city"):
            admin_path["city"] = anchor.primary_city
        if anchor.region and not admin_path.get("region"):
            admin_path["region"] = anchor.region
        if anchor.country and not admin_path.get("country"):
            admin_path["country"] = anchor.country

    # Reuse persisted context scoped to the same city when available so that
    # admin hierarchy hints are only shared between matching locations.
    reuse_row: Optional[asyncpg.Record] = None
    fallback_row: Optional[asyncpg.Record] = None
    candidate_city = admin_path.get("city")

    if candidate_city:
        reuse_row = await conn.fetchrow(
            """
            SELECT city, region, country, planet, galaxy, realm
            FROM Locations
            WHERE user_id = $1 AND conversation_id = $2 AND city IS NOT NULL
                  AND LOWER(city) = LOWER($3)
            ORDER BY location_id DESC
            LIMIT 1
            """,
            int(user_id),
            int(conversation_id),
            candidate_city,
        )
        if not reuse_row:
            fallback_row = await conn.fetchrow(
                """
                SELECT city, region, country, planet, galaxy, realm
                FROM Locations
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY location_id DESC
                LIMIT 1
                """,
                int(user_id),
                int(conversation_id),
            )
    else:
        fallback_row = await conn.fetchrow(
            """
            SELECT city, region, country, planet, galaxy, realm
            FROM Locations
            WHERE user_id = $1 AND conversation_id = $2
            ORDER BY location_id DESC
            LIMIT 1
            """,
            int(user_id),
            int(conversation_id),
        )

    if reuse_row:
        for level in ("city", "region", "country"):
            if not admin_path.get(level):
                value = reuse_row.get(level)
                if value:
                    admin_path[level] = value

    if admin_path:
        meta.setdefault("admin_path", admin_path)

    source = str(meta.get("source") or "").strip().lower()
    is_fictional_branch = scope == "fictional" or source == "fictional"

    planet_hint = (
        meta.get("planet")
        or meta.get("world_name")
        or default_planet
        or (anchor.world_name if anchor and anchor.world_name else None)
    )

    hierarchy = await assign_hierarchy(
        conn,
        candidate,
        scope=scope,
        anchor=anchor,
        mint_policy=mint_policy,
        default_planet=planet_hint,
    )

    world_name: Optional[str] = None
    if isinstance(hierarchy, dict):
        chain = hierarchy.get("chain")
        if chain and "hierarchy_chain" not in meta:
            meta["hierarchy_chain"] = chain
        leaf = hierarchy.get("leaf")
        if isinstance(leaf, dict) and "place_leaf_id" not in meta:
            meta["place_leaf_id"] = leaf.get("id")
        world_name = hierarchy.get("world_name")

    display_name = meta.get("display_name") or candidate.place.name or "Unknown"
    normalized_name = _normalize_location_name(display_name)
    meta.setdefault("display_name", display_name)

    description = meta.get("description") or meta.get("display_name")
    if not description:
        contextual = [display_name]
        for key in ("city", "region", "country"):
            val = admin_path.get(key)
            if val and val not in contextual:
                contextual.append(val)
        if len(contextual) > 1:
            description = ", ".join(contextual)

    location_type = meta.get("location_type") or meta.get("category") or candidate.place.level
    parent_location = (
        meta.get("parent_location")
        or admin_path.get("district")
        or admin_path.get("city")
        or admin_path.get("region")
    )

    room = meta.get("room")
    building = meta.get("building") or admin_path.get("building")
    district = meta.get("district") or admin_path.get("district") or admin_path.get("neighborhood")
    district_type = meta.get("district_type")
    city = meta.get("city") or admin_path.get("city")
    region = meta.get("region") or admin_path.get("region")
    country = meta.get("country") or admin_path.get("country")

    for key, value in (("city", city), ("region", region), ("country", country)):
        if value and key not in meta:
            meta[key] = value

    context_row: Optional[asyncpg.Record]
    if reuse_row:
        context_row = reuse_row
    elif candidate_city:
        context_row = None
    else:
        context_row = fallback_row

    planet = meta.get("planet") or world_name or (context_row.get("planet") if context_row else None) or default_planet
    if not planet:
        planet = "Earth" if scope == "real" else (world_name or "Fictional World")

    galaxy = meta.get("galaxy") or (context_row.get("galaxy") if context_row else None) or default_galaxy
    if not galaxy:
        galaxy = "Milky Way" if scope == "real" else "Unknown Galaxy"

    realm = meta.get("realm") or (context_row.get("realm") if context_row else None) or default_realm
    if not realm:
        realm = DEFAULT_REALM if scope == "real" else "fictional"

    meta.setdefault("planet", planet)
    meta.setdefault("galaxy", galaxy)
    meta.setdefault("realm", realm)

    anchor_hints: Dict[str, Any] = {}
    if anchor and isinstance(anchor.hints, dict):
        anchor_hints = anchor.hints

    anchor_geo = anchor_hints.get("geo_anchor") if isinstance(anchor_hints, dict) else None
    anchor_district_hint: Optional[str] = None
    if anchor_geo is not None:
        anchor_district_hint = getattr(anchor_geo, "neighborhood", None) or getattr(anchor_geo, "district", None)
    if not anchor_district_hint and anchor_hints:
        hint_value = anchor_hints.get("district") or anchor_hints.get("neighborhood")
        if isinstance(hint_value, str):
            anchor_district_hint = hint_value
    if not anchor_district_hint and anchor and anchor.focus and anchor.focus.meta:
        focus_meta = anchor.focus.meta
        anchor_district_hint = focus_meta.get("district") or focus_meta.get("neighborhood")

    normalized_district = _normalize_location_name(district) if district else None
    normalized_anchor_district = (
        _normalize_location_name(anchor_district_hint) if anchor_district_hint else None
    )
    same_district_as_anchor = bool(
        normalized_district and normalized_anchor_district == normalized_district
    )

    base_lat = candidate.place.lat
    base_lon = candidate.place.lon
    if base_lat is None and anchor and anchor.lat is not None:
        base_lat = anchor.lat
    if base_lon is None and anchor and anchor.lon is not None:
        base_lon = anchor.lon
    if base_lat is None and anchor_geo is not None:
        geo_lat = getattr(anchor_geo, "lat", None)
        if geo_lat is not None:
            base_lat = geo_lat
    if base_lon is None and anchor_geo is not None:
        geo_lon = getattr(anchor_geo, "lon", None)
        if geo_lon is not None:
            base_lon = geo_lon

    if is_fictional_branch:
        seed_base = f"{user_id}:{conversation_id}:{normalized_name}"
        if base_lat is not None:
            lat_rng = random.Random(f"{seed_base}:lat")
            base_lat = base_lat + lat_rng.uniform(-0.02, 0.02)
        if base_lon is not None:
            lon_rng = random.Random(f"{seed_base}:lon")
            base_lon = base_lon + lon_rng.uniform(-0.02, 0.02)

    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    if normalized_district:
        center_base_lat: Optional[float] = None
        center_base_lon: Optional[float] = None

        if anchor and anchor.lat is not None and anchor.lon is not None:
            center_base_lat = anchor.lat
            center_base_lon = anchor.lon
        elif anchor_geo is not None:
            geo_lat = getattr(anchor_geo, "lat", None)
            geo_lon = getattr(anchor_geo, "lon", None)
            if geo_lat is not None and geo_lon is not None:
                center_base_lat = geo_lat
                center_base_lon = geo_lon
        elif base_lat is not None and base_lon is not None:
            center_base_lat = base_lat
            center_base_lon = base_lon

        if center_base_lat is not None and center_base_lon is not None:
            jitter_scale = 0.0 if same_district_as_anchor else 0.005
            center_seed = f"{conversation_id}:{normalized_district}:center"
            center_rng = random.Random(center_seed)
            lat_jitter = center_rng.uniform(-jitter_scale, jitter_scale) if jitter_scale else 0.0
            lon_jitter = center_rng.uniform(-jitter_scale, jitter_scale) if jitter_scale else 0.0
            center_lat = center_base_lat + lat_jitter
            center_lon = center_base_lon + lon_jitter
            meta.setdefault("district_center", {"lat": round(center_lat, 6), "lon": round(center_lon, 6)})

    if center_lat is not None and center_lon is not None:
        offset_seed = f"{conversation_id}:{normalized_name}:offset"
        offset_rng = random.Random(offset_seed)
        lat_offset = offset_rng.uniform(-0.001, 0.001)
        lon_offset = offset_rng.uniform(-0.001, 0.001)
        lat = round(center_lat + lat_offset, 6)
        lon = round(center_lon + lon_offset, 6)
    else:
        lat = round(base_lat, 6) if base_lat is not None else None
        lon = round(base_lon, 6) if base_lon is not None else None

    candidate.place.lat = lat
    candidate.place.lon = lon

    if lat is not None:
        meta.setdefault("lat", lat)
    if lon is not None:
        meta.setdefault("lon", lon)

    open_hours_raw = meta.get("open_hours")
    if isinstance(open_hours_raw, dict):
        open_hours = open_hours_raw
    elif isinstance(open_hours_raw, str):
        try:
            open_hours = json.loads(open_hours_raw)
        except json.JSONDecodeError:
            open_hours = open_hours_raw
    else:
        open_hours = None

    controlling_faction = meta.get("controlling_faction")
    cultural_significance = meta.get("cultural_significance")
    economic_importance = meta.get("economic_importance")
    strategic_value = meta.get("strategic_value")
    population_density = meta.get("population_density")

    notable_features = meta.get("notable_features")
    if notable_features is not None and not isinstance(notable_features, (list, tuple, set)):
        notable_features = [notable_features]
    hidden_aspects = meta.get("hidden_aspects")
    if hidden_aspects is not None and not isinstance(hidden_aspects, (list, tuple, set)):
        hidden_aspects = [hidden_aspects]
    access_restrictions = meta.get("access_restrictions")
    if access_restrictions is not None and not isinstance(access_restrictions, (list, tuple, set)):
        access_restrictions = [access_restrictions]
    local_customs = meta.get("local_customs")
    if local_customs is not None and not isinstance(local_customs, (list, tuple, set)):
        local_customs = [local_customs]

    embedding = meta.get("embedding")

    location_kwargs: Dict[str, Any] = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "location_name": normalized_name,
        "description": description,
        "location_type": location_type,
        "parent_location": parent_location,
        "room": room,
        "building": building,
        "district": district,
        "district_type": district_type,
        "city": city,
        "region": region,
        "country": country,
        "planet": planet,
        "galaxy": galaxy,
        "realm": realm,
        "scope": scope,
        "lat": lat,
        "lon": lon,
        "is_fictional": bool(meta.get("is_fictional") or is_fictional_branch),
        "open_hours": open_hours,
        "controlling_faction": controlling_faction,
    }

    if cultural_significance is not None:
        location_kwargs["cultural_significance"] = cultural_significance
    if economic_importance is not None:
        location_kwargs["economic_importance"] = economic_importance
    if strategic_value is not None:
        location_kwargs["strategic_value"] = strategic_value
    if population_density is not None:
        location_kwargs["population_density"] = population_density
    if notable_features is not None:
        location_kwargs["notable_features"] = list(notable_features)
    if hidden_aspects is not None:
        location_kwargs["hidden_aspects"] = list(hidden_aspects)
    if access_restrictions is not None:
        location_kwargs["access_restrictions"] = list(access_restrictions)
    if local_customs is not None:
        location_kwargs["local_customs"] = list(local_customs)
    if embedding is not None:
        location_kwargs["embedding"] = list(embedding)

    location_obj = Location(**location_kwargs)
    payload = location_obj.to_dict()

    serialized_payload = dict(payload)
    for field in ("open_hours", "notable_features", "hidden_aspects", "access_restrictions", "local_customs"):
        serialized_payload[field] = _serialize_json_value(serialized_payload.get(field))

    row = await conn.fetchrow(
        """
        INSERT INTO Locations (
            user_id,
            conversation_id,
            location_name,
            description,
            location_type,
            parent_location,
            room,
            building,
            district,
            district_type,
            city,
            region,
            country,
            planet,
            galaxy,
            realm,
            lat,
            lon,
            is_fictional,
            open_hours,
            controlling_faction,
            cultural_significance,
            economic_importance,
            strategic_value,
            population_density,
            notable_features,
            hidden_aspects,
            access_restrictions,
            local_customs,
            embedding
        )
        VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15,
            $16, $17, $18, $19, $20,
            $21, $22, $23, $24, $25,
            $26, $27, $28, $29, $30
        )
        ON CONFLICT (user_id, conversation_id, location_name)
        DO UPDATE SET
            description = COALESCE(EXCLUDED.description, Locations.description),
            location_type = COALESCE(EXCLUDED.location_type, Locations.location_type),
            parent_location = COALESCE(EXCLUDED.parent_location, Locations.parent_location),
            room = COALESCE(EXCLUDED.room, Locations.room),
            building = COALESCE(EXCLUDED.building, Locations.building),
            district = COALESCE(EXCLUDED.district, Locations.district),
            district_type = COALESCE(EXCLUDED.district_type, Locations.district_type),
            city = COALESCE(EXCLUDED.city, Locations.city),
            region = COALESCE(EXCLUDED.region, Locations.region),
            country = COALESCE(EXCLUDED.country, Locations.country),
            planet = COALESCE(EXCLUDED.planet, Locations.planet),
            galaxy = COALESCE(EXCLUDED.galaxy, Locations.galaxy),
            realm = COALESCE(EXCLUDED.realm, Locations.realm),
            lat = COALESCE(EXCLUDED.lat, Locations.lat),
            lon = COALESCE(EXCLUDED.lon, Locations.lon),
            is_fictional = EXCLUDED.is_fictional,
            open_hours = COALESCE(EXCLUDED.open_hours, Locations.open_hours),
            controlling_faction = COALESCE(EXCLUDED.controlling_faction, Locations.controlling_faction),
            cultural_significance = COALESCE(EXCLUDED.cultural_significance, Locations.cultural_significance),
            economic_importance = COALESCE(EXCLUDED.economic_importance, Locations.economic_importance),
            strategic_value = COALESCE(EXCLUDED.strategic_value, Locations.strategic_value),
            population_density = COALESCE(EXCLUDED.population_density, Locations.population_density),
            notable_features = COALESCE(EXCLUDED.notable_features, Locations.notable_features),
            hidden_aspects = COALESCE(EXCLUDED.hidden_aspects, Locations.hidden_aspects),
            access_restrictions = COALESCE(EXCLUDED.access_restrictions, Locations.access_restrictions),
            local_customs = COALESCE(EXCLUDED.local_customs, Locations.local_customs),
            embedding = COALESCE(EXCLUDED.embedding, Locations.embedding)
        RETURNING *
        """,
        serialized_payload["user_id"],
        serialized_payload["conversation_id"],
        serialized_payload["location_name"],
        serialized_payload.get("description"),
        serialized_payload.get("location_type"),
        serialized_payload.get("parent_location"),
        serialized_payload.get("room"),
        serialized_payload.get("building"),
        serialized_payload.get("district"),
        serialized_payload.get("district_type"),
        serialized_payload.get("city"),
        serialized_payload.get("region"),
        serialized_payload.get("country"),
        serialized_payload.get("planet"),
        serialized_payload.get("galaxy"),
        serialized_payload.get("realm"),
        serialized_payload.get("lat"),
        serialized_payload.get("lon"),
        serialized_payload.get("is_fictional"),
        serialized_payload.get("open_hours"),
        serialized_payload.get("controlling_faction"),
        serialized_payload.get("cultural_significance"),
        serialized_payload.get("economic_importance"),
        serialized_payload.get("strategic_value"),
        serialized_payload.get("population_density"),
        serialized_payload.get("notable_features"),
        serialized_payload.get("hidden_aspects"),
        serialized_payload.get("access_restrictions"),
        serialized_payload.get("local_customs"),
        serialized_payload.get("embedding"),
    )

    if row is None:
        raise RuntimeError("Failed to persist location hierarchy")

    persisted = Location.from_record(row)

    if persisted.id is not None:
        meta.setdefault("location_row_id", persisted.id)
    meta.setdefault("location_name", persisted.location_name)
    meta.setdefault("planet", persisted.planet)
    meta.setdefault("galaxy", persisted.galaxy)
    meta.setdefault("realm", persisted.realm)

    # ✨ NEW: Notify canon system
    await _notify_canon_of_location(conn, user_id, conversation_id, persisted)

    return persisted


async def get_or_create_location(
    conn: asyncpg.Connection,
    *,
    user_id: int,
    conversation_id: int,
    candidate: Candidate,
    scope: Scope = "real",
    anchor: Optional[Anchor] = None,
    mint_policy: Optional[str] = None,
    default_planet: Optional[str] = None,
    default_galaxy: Optional[str] = None,
    default_realm: Optional[str] = None,
) -> Location:
    """Return an existing location or persist a new hierarchy-backed entry."""

    name_source = candidate.place.meta.get("display_name") if isinstance(candidate.place.meta, dict) else None
    display_name = name_source or candidate.place.name
    normalized_name = _normalize_location_name(display_name)

    row = await conn.fetchrow(
        """
        SELECT *
        FROM Locations
        WHERE user_id = $1 AND conversation_id = $2 AND location_name = $3
        ORDER BY location_id DESC
        LIMIT 1
        """,
        int(user_id),
        int(conversation_id),
        normalized_name,
    )
    if row:
        return Location.from_record(row)

    candidate.place.meta = dict(candidate.place.meta or {})
    candidate.place.meta.setdefault("display_name", display_name)

    return await generate_and_persist_hierarchy(
        conn,
        user_id=user_id,
        conversation_id=conversation_id,
        candidate=candidate,
        scope=scope,
        anchor=anchor,
        mint_policy=mint_policy,
        default_planet=default_planet,
        default_galaxy=default_galaxy,
        default_realm=default_realm,
    )


__all__ = [
    "assign_hierarchy",
    "generate_and_persist_hierarchy",
    "get_or_create_location",
]
