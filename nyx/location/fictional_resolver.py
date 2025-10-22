# nyx/location/fictional_resolver.py

from __future__ import annotations

import json
import logging
import math
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import asyncpg

from db.connection import get_db_connection_context
from logic.gpt_utils import call_gpt_json

import unicodedata

from nyx.conversation.snapshot_store import ConversationSnapshotStore

from .anchors import derive_geo_anchor
from .hierarchy import generate_and_persist_hierarchy
from .query import PlaceQuery
from .types import (
    Anchor,
    Candidate,
    Location,
    Place,
    ResolveResult,
    STATUS_ASK,
    STATUS_EXACT,
    STATUS_NOT_FOUND,  # --- CHANGE: Import STATUS_NOT_FOUND ---
)

logger = logging.getLogger(__name__)

_DISTRICT_SLUG_RE = re.compile(r"[^a-z0-9]+")

_DEFAULT_DISTRICT_FALLBACKS: List[Dict[str, Any]] = [
    {
        "key": "old_port",
        "name": "Old Port",
        "vibe": "Salt-washed boardwalks buzzing with fishmongers and smugglers' gossip.",
        "layout": "Wraps around the tidal harbor on the eastern edge of the city.",
        "theme": "Harborfront trade and lantern-lit docks",
        "summary": "An old harbor district that never sleeps.",
        "features": [
            "Creaking piers and rusted cranes",
            "Underground tunnels linking warehouses",
            "Nighttime lantern festivals along the quay",
        ],
    },
    {
        "key": "glassline",
        "name": "Glassline",
        "vibe": "Sleek towers of mirrored glass shelter avant-garde tech collectives.",
        "layout": "A linear ridge of skybridges stretching north of the city center.",
        "theme": "Cutting-edge innovation wrapped in neon glow",
        "summary": "The experimental heart of the city where prototypes hit the streets first.",
        "features": [
            "Skyrail arteries weaving between research spires",
            "Holo-galleries that remix memories",
            "A cooperative robotics bazaar at street level",
        ],
    },
    {
        "key": "western_dunes",
        "name": "Western Dunes",
        "vibe": "Wind-carved sandstone rowhouses frame hidden courtyards and spice markets.",
        "layout": "Undulating terraces climbing the sun-baked bluffs to the west.",
        "theme": "Desert craftsmanship meeting city refuge",
        "summary": "A residential quarter shaped by shifting sands and patient artisans.",
        "features": [
            "Cooling cistern gardens shared by neighbors",
            "Rooftop spice auctions at twilight",
            "Whisper tunnels that carry news between homes",
        ],
    },
    {
        "key": "canvas_row",
        "name": "Canvas Row",
        "vibe": "Murals spill onto cobbles while buskers orchestrate the evening air.",
        "layout": "A looping crescent hugging the riverfront south of downtown.",
        "theme": "Art collectives and festival streets",
        "summary": "The city's bohemian artery where every night is opening night.",
        "features": [
            "Shared studios carved into old tram depots",
            "Pop-up night markets curated by guilds",
            "Floating stages drifting along the river",
        ],
    },
]


def _slugify_token(value: str, *, default: str = "district") -> str:
    candidate = (value or "").strip().lower()
    candidate = _DISTRICT_SLUG_RE.sub("-", candidate)
    candidate = candidate.strip("-")
    return candidate or default


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_feature_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [_stringify(item) for item in value]
        return [item for item in items if item]
    if isinstance(value, str):
        text = _stringify(value)
        return [text] if text else []
    return []


def _compose_description(vibe: str, layout: str, theme: str, summary: str) -> str:
    parts: List[str] = []
    for piece in (vibe, layout):
        piece = _stringify(piece)
        if piece:
            parts.append(piece)
    theme_text = _stringify(theme)
    if theme_text:
        parts.append(f"Theme: {theme_text}")
    summary_text = _stringify(summary)
    if summary_text:
        parts.append(summary_text)
    return " ".join(parts).strip()


def _truncate_seed_text(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_world_seed(world_seed: Mapping[str, Any], city_name: str) -> str:
    if not world_seed:
        return f"The city of {city_name} has no additional seed metadata."
    interesting_keys = (
        "themes",
        "tone",
        "genre",
        "travel_rules",
        "technology",
        "technology_level",
        "magic",
        "factions",
        "power_centers",
        "notable_sites",
        "history",
        "geography",
        "climate",
        "culture",
        "conflicts",
    )
    summary: Dict[str, Any] = {}
    for key in interesting_keys:
        value = world_seed.get(key)
        if value:
            summary[key] = value
    if not summary:
        summary = dict(world_seed)
    serialized = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
    return _truncate_seed_text(serialized)


def _extract_nested_value(seed: Mapping[str, Any], path: Iterable[str]) -> Any:
    current: Any = seed
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            return None
    return current


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


_LAT_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("geo_anchor", "lat"),
    ("geo_anchor", "latitude"),
    ("geo", "lat"),
    ("geo", "latitude"),
    ("origin", "lat"),
    ("origin", "latitude"),
    ("city_center", "lat"),
    ("city_center", "latitude"),
    ("coordinates", "lat"),
    ("coordinates", "latitude"),
    ("primary_city", "lat"),
    ("primary_city", "latitude"),
    ("lat",),
    ("latitude",),
)

_LON_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("geo_anchor", "lon"),
    ("geo_anchor", "longitude"),
    ("geo", "lon"),
    ("geo", "longitude"),
    ("origin", "lon"),
    ("origin", "longitude"),
    ("city_center", "lon"),
    ("city_center", "longitude"),
    ("coordinates", "lon"),
    ("coordinates", "longitude"),
    ("primary_city", "lon"),
    ("primary_city", "longitude"),
    ("lon",),
    ("longitude",),
)

_REGION_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("region",),
    ("primary_region",),
    ("geo", "region"),
    ("geo_anchor", "region"),
    ("hints", "geo_anchor", "region"),
)

_COUNTRY_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("country",),
    ("nation",),
    ("geo", "country"),
    ("geo_anchor", "country"),
    ("hints", "geo_anchor", "country"),
)

_PLANET_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("planet",),
    ("world", "planet"),
    ("geo", "planet"),
)

_GALAXY_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("galaxy",),
    ("world", "galaxy"),
)

_REALM_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("realm",),
    ("world", "realm"),
)

_WORLD_NAME_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("name",),
    ("world_name",),
    ("world", "name"),
)


def _extract_lat_lon(world_seed: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat: Optional[float] = None
    lon: Optional[float] = None
    for path in _LAT_PATHS:
        lat = _coerce_float(_extract_nested_value(world_seed, path))
        if lat is not None:
            break
    for path in _LON_PATHS:
        lon = _coerce_float(_extract_nested_value(world_seed, path))
        if lon is not None:
            break
    return lat, lon


def _extract_text_value(world_seed: Mapping[str, Any], paths: Iterable[Iterable[str]]) -> Optional[str]:
    for path in paths:
        value = _extract_nested_value(world_seed, path)
        text = _stringify(value)
        if text:
            return text
    return None


def _fallback_districts(city_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for entry in _DEFAULT_DISTRICT_FALLBACKS:
        clone = dict(entry)
        clone.setdefault(
            "description",
            _compose_description(clone.get("vibe", ""), clone.get("layout", ""), clone.get("theme", ""), clone.get("summary", "")),
        )
        results.append(clone)
    return results


def _normalize_generated_districts(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    districts: Any = payload
    if isinstance(payload, Mapping):
        districts = payload.get("districts", payload)
    if not isinstance(districts, list):
        return []

    normalized: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for entry in districts:
        if not isinstance(entry, Mapping):
            continue
        name = _stringify(entry.get("name") or entry.get("label"))
        if not name:
            continue
        key_source = _stringify(entry.get("key") or entry.get("id"))
        slug = _slugify_token(key_source or name, default=f"district-{len(normalized) + 1}")
        if slug in seen_keys:
            slug = _slugify_token(f"{slug}-{len(seen_keys) + 1}")
        seen_keys.add(slug)

        vibe = _stringify(entry.get("vibe") or entry.get("mood") or entry.get("atmosphere"))
        layout = _stringify(entry.get("layout") or entry.get("relative_layout") or entry.get("placement") or entry.get("orientation"))
        theme = _stringify(entry.get("theme") or entry.get("focus") or entry.get("hook"))
        summary = _stringify(entry.get("summary") or entry.get("tagline") or entry.get("pitch") or entry.get("notes"))
        features = _normalize_feature_list(entry.get("features") or entry.get("landmarks") or entry.get("notable_features"))
        description = _stringify(entry.get("description")) or _compose_description(vibe, layout, theme, summary)

        normalized.append(
            {
                "key": slug,
                "name": name,
                "vibe": vibe,
                "layout": layout,
                "theme": theme,
                "summary": summary,
                "features": features,
                "description": description,
            }
        )
        if len(normalized) >= 5:
            break
    return normalized


def _district_profile_from_location(location: Location) -> Dict[str, Any]:
    customs = list(location.local_customs or [])
    for entry in customs:
        if isinstance(entry, Mapping) and entry.get("kind") == "district_profile":
            profile = dict(entry)
            profile.setdefault("name", entry.get("name") or location.district or location.location_name.title())
            profile.setdefault("key", _slugify_token(str(entry.get("key") or profile["name"])))
            profile.setdefault("vibe", _stringify(entry.get("vibe")) or (location.description or ""))
            profile.setdefault("layout", _stringify(entry.get("layout")))
            profile.setdefault("theme", _stringify(entry.get("theme")))
            profile.setdefault("summary", _stringify(entry.get("summary")))
            return profile

    fallback_name = location.district or location.location_name
    profile = {
        "name": fallback_name.title() if isinstance(fallback_name, str) else "District",
        "key": _slugify_token(str(fallback_name or location.id or "district")),
        "vibe": location.description or "",
        "layout": "",
        "theme": "",
        "summary": "",
    }
    return profile


def _venue_matches_district(venue: Mapping[str, Any], *, key: str, label: str) -> bool:
    if not isinstance(venue, Mapping):
        return False
    v_key = venue.get("district_key")
    if isinstance(v_key, str) and v_key == key:
        return True
    v_label = venue.get("district")
    if isinstance(v_label, str) and v_label == label:
        return True
    return False


import json
from typing import Any, Dict, List, Mapping

# Assuming _normalize_feature_list and Location are defined elsewhere in the file
# def _normalize_feature_list(value: Any) -> List[str]: ...
# class Location: ...

def _extract_world_seed_context(location: Location) -> Dict[str, Any]:
    """
    Extracts world-building context from a Location object, ensuring that nested
    JSON strings within features are correctly parsed.
    """
    context: Dict[str, Any] = {}

    core_fields = {
        "city": location.city or location.parent_location,
        "region": location.region,
        "country": location.country,
        "planet": location.planet,
        "galaxy": location.galaxy,
        "realm": location.realm,
        "description": location.description,
    }
    for key, value in core_fields.items():
        if value:
            context[key] = value

    for entry in location.local_customs or []:
        if isinstance(entry, Mapping):
            kind = str(entry.get("kind") or "").lower()
            if kind in {"world_seed", "world_seed_snapshot", "world_profile", "world_context"}:
                for key, value in entry.items():
                    if key == "kind":
                        continue
                    if value:
                        context.setdefault(key, value)

    # --- FIX STARTS HERE ---
    
    # The 'notable_features' field might contain a single string that is itself a JSON array.
    # We need to parse it to prevent double-serialization in the prompt.
    raw_features = location.notable_features or []
    parsed_features: List[str] = []

    if raw_features and isinstance(raw_features[0], str):
        try:
            # Attempt to parse the first element as a JSON list
            maybe_list = json.loads(raw_features[0])
            if isinstance(maybe_list, list):
                # If successful, use this parsed list as the features
                parsed_features = _normalize_feature_list(maybe_list)
        except (json.JSONDecodeError, TypeError):
            # If it's not a valid JSON string, or not a list,
            # fall back to treating it as a normal list of strings.
            parsed_features = _normalize_feature_list(raw_features)
    else:
        # If the list is empty or doesn't start with a string, process as normal.
        parsed_features = _normalize_feature_list(raw_features)

    if parsed_features:
        context.setdefault("district_features", parsed_features)

    # --- FIX ENDS HERE ---

    return context

def _merge_seed_payload(base: Dict[str, Any], payload: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            existing = base.get(key)
            if isinstance(existing, Mapping):
                base[key] = _merge_seed_payload(dict(existing), value)
            elif key not in base or not base[key]:
                base[key] = dict(value)
        elif key not in base or base[key] in (None, "", [], {}):
            base[key] = value
    return base


def _normalize_search_token(value: Optional[str]) -> str:
    text = _stringify(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_poi_profile(location: Location) -> Dict[str, Any]:
    for entry in location.local_customs or []:
        if isinstance(entry, Mapping) and str(entry.get("kind") or "").lower() == "fictional_poi_profile":
            return dict(entry)
    return {}


def _display_name_for_location(location: Location, profile: Optional[Mapping[str, Any]] = None) -> str:
    if profile is None:
        profile = _extract_poi_profile(location)
    name = _stringify(profile.get("name") if isinstance(profile, Mapping) else None)
    if name:
        return name
    if location.description:
        first_line = location.description.splitlines()[0]
        return _stringify(first_line)
    raw = location.location_name.replace("_", " ")
    return raw.title()


def _score_location_match(
    location: Location,
    profile: Mapping[str, Any],
    phrase_tokens: List[str],
    word_tokens: List[str],
) -> float:
    display_name = _normalize_search_token(_display_name_for_location(location, profile))
    description = _normalize_search_token(location.description)
    lore = _normalize_search_token(profile.get("lore")) if profile else ""
    category = _normalize_search_token(profile.get("category")) if profile else ""
    features = [_normalize_search_token(item) for item in location.notable_features or []]
    score = 0.1

    for phrase in phrase_tokens:
        if not phrase:
            continue
        if phrase == display_name:
            score += 6.0
        elif phrase in display_name:
            score += 3.5
        if phrase and phrase in description:
            score += 2.0
        if phrase and phrase in lore:
            score += 1.5
        for feat in features:
            if phrase and phrase in feat:
                score += 1.0

    for token in word_tokens:
        if not token:
            continue
        if token in display_name:
            score += 0.75
        if token in description:
            score += 0.5
        if token in lore:
            score += 0.4
        if category and token in category:
            score += 0.45
        for feat in features:
            if token in feat:
                score += 0.25

    return score


def _build_candidate_from_location(location: Location) -> Tuple[Candidate, Dict[str, Any]]:
    profile = _extract_poi_profile(location)
    display_name = _display_name_for_location(location, profile)
    address: Dict[str, Any] = {}
    for key in ("district", "city", "region", "country", "planet", "galaxy", "realm"):
        value = getattr(location, key, None)
        if value:
            address[key] = value

    district_key = profile.get("key") if isinstance(profile, Mapping) else None
    if district_key:
        address.setdefault("district_key", district_key)

    place_meta: Dict[str, Any] = {
        "source": "fictional",
        "is_fictional": True,
        "location_id": location.id,
        "notable_features": list(location.notable_features or []),
    }
    if profile:
        for meta_key in ("category", "lore", "travel", "features"):
            if profile.get(meta_key) is not None:
                place_meta[meta_key] = profile.get(meta_key)
        display_name_value = _stringify(profile.get("name"))
        if display_name_value:
            place_meta.setdefault("display_name", display_name_value)

    place = Place(
        name=display_name,
        level="venue",
        lat=location.lat,
        lon=location.lon,
        address=address,
        meta=place_meta,
    )

    candidate = Candidate(
        place=place,
        confidence=0.85,
        rationale="fictional_poi_lookup",
        raw={
            "location_id": location.id,
            "district": location.district,
            "city": location.city,
        },
    )

    return candidate, profile


def _build_navigation_operations(
    location: Location,
    display_name: str,
    profile: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    lore_payload: Dict[str, Any] = {
        "district": location.district,
        "city": location.city,
        "features": list(location.notable_features or []),
    }
    if isinstance(profile, Mapping):
        if profile.get("lore"):
            lore_payload["lore"] = profile.get("lore")
        if profile.get("travel"):
            lore_payload["travel"] = profile.get("travel")
        if profile.get("category"):
            lore_payload["category"] = profile.get("category")

    op = {
        "op": "poi.navigate",
        "label": display_name,
        "lat": location.lat,
        "lon": location.lon,
        "category": profile.get("category") if isinstance(profile, Mapping) else location.location_type,
        "lore": lore_payload,
    }
    return [op]


_BEARING_DEGREES: Dict[str, float] = {
    "e": 0.0,
    "east": 0.0,
    "ne": 45.0,
    "northeast": 45.0,
    "n": 90.0,
    "north": 90.0,
    "nw": 135.0,
    "northwest": 135.0,
    "w": 180.0,
    "west": 180.0,
    "sw": 225.0,
    "southwest": 225.0,
    "s": 270.0,
    "south": 270.0,
    "se": 315.0,
    "southeast": 315.0,
}


def _bearing_to_radians(label: Optional[str]) -> Optional[float]:
    if not label:
        return None
    label = label.strip().lower()
    if not label:
        return None
    if label in _BEARING_DEGREES:
        return math.radians(_BEARING_DEGREES[label])
    try:
        value = float(label)
    except (TypeError, ValueError):
        return None
    return math.radians(value % 360.0)


def _normalize_generated_pois(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pois: Any = payload
    if isinstance(payload, Mapping):
        for key in ("pois", "venues", "points_of_interest", "results"):
            if key in payload:
                pois = payload.get(key)
                break
        else:
            pois = payload

    if not isinstance(pois, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for entry in pois:
        if not isinstance(entry, Mapping):
            continue

        name = _stringify(entry.get("name"))
        if not name:
            continue

        description = _stringify(entry.get("description") or entry.get("summary"))
        category = _stringify(entry.get("category") or entry.get("type") or "venue")
        features = _normalize_feature_list(entry.get("notable_features") or entry.get("features"))
        lore = _stringify(entry.get("lore") or entry.get("hook") or entry.get("story"))

        travel_info = entry.get("travel") if isinstance(entry.get("travel"), Mapping) else {}
        if not travel_info:
            travel_info = entry if isinstance(entry, Mapping) else {}

        travel_modes = _normalize_feature_list(
            (entry.get("travel_modes") if isinstance(entry, Mapping) else [])
            or (travel_info.get("modes") if isinstance(travel_info, Mapping) else [])
            or (travel_info.get("transport") if isinstance(travel_info, Mapping) else [])
        )
        travel_time = _coerce_float(
            (travel_info.get("time_minutes") if isinstance(travel_info, Mapping) else None)
            or (travel_info.get("duration_minutes") if isinstance(travel_info, Mapping) else None)
            or entry.get("travel_time_minutes")
            or entry.get("travel_time")
        )
        travel_note = _stringify(
            (travel_info.get("narrative") if isinstance(travel_info, Mapping) else None)
            or (travel_info.get("hook") if isinstance(travel_info, Mapping) else None)
            or (travel_info.get("notes") if isinstance(travel_info, Mapping) else None)
            or entry.get("travel_hook")
            or entry.get("travel_note")
        )

        coords_payload: Mapping[str, Any] = {}
        if isinstance(entry.get("coordinates"), Mapping):
            coords_payload = entry["coordinates"]
        elif isinstance(entry.get("offset"), Mapping):
            coords_payload = entry["offset"]
        elif isinstance(entry.get("offsets"), Mapping):
            coords_payload = entry["offsets"]

        dx_m = _coerce_float(
            (coords_payload.get("dx_m") if isinstance(coords_payload, Mapping) else None)
            or (coords_payload.get("x") if isinstance(coords_payload, Mapping) else None)
            or (coords_payload.get("east_m") if isinstance(coords_payload, Mapping) else None)
        )
        dy_m = _coerce_float(
            (coords_payload.get("dy_m") if isinstance(coords_payload, Mapping) else None)
            or (coords_payload.get("y") if isinstance(coords_payload, Mapping) else None)
            or (coords_payload.get("north_m") if isinstance(coords_payload, Mapping) else None)
        )

        if (dx_m is None or dy_m is None) and isinstance(coords_payload, Mapping):
            distance = _coerce_float(
                coords_payload.get("distance_m")
                or coords_payload.get("distance")
                or coords_payload.get("radius_m")
            )
            bearing = _stringify(coords_payload.get("bearing") or coords_payload.get("direction"))
            if distance is not None:
                angle = _bearing_to_radians(bearing)
                if angle is None:
                    angle = random.Random(f"poi-bearing:{name}").uniform(0.0, 2 * math.pi)
                dx_m = distance * math.cos(angle)
                dy_m = distance * math.sin(angle)

        normalized.append(
            {
                "name": name,
                "description": description,
                "category": category or "venue",
                "features": features,
                "lore": lore,
                "travel_modes": travel_modes,
                "travel_time_minutes": travel_time,
                "travel_note": travel_note,
                "dx_m": dx_m,
                "dy_m": dy_m,
                "raw": dict(entry),
            }
        )

    return normalized


def _offset_coordinates(
    base_lat: Optional[float],
    base_lon: Optional[float],
    *,
    dx_m: Optional[float],
    dy_m: Optional[float],
    seed_key: str,
) -> Tuple[float, float, float, float]:
    lat = base_lat if base_lat is not None else 0.0
    lon = base_lon if base_lon is not None else 0.0

    if dx_m is None or dy_m is None:
        rng = random.Random(seed_key)
        radius = rng.uniform(60.0, 220.0)
        angle = rng.uniform(0.0, 2 * math.pi)
        dx_m = radius * math.cos(angle)
        dy_m = radius * math.sin(angle)

    lat_offset = (dy_m or 0.0) / 111_111.0
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    lon_offset = (dx_m or 0.0) / (111_111.0 * cos_lat)

    return round(lat + lat_offset, 6), round(lon + lon_offset, 6), dx_m or 0.0, dy_m or 0.0


async def load_or_generate_world_seed(
    anchor: Anchor,
    meta: Mapping[str, Any],
    store: ConversationSnapshotStore,
    user_id: str,
    conversation_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snapshot = store.get(user_id, conversation_id) or {}
    working_snapshot = dict(snapshot)

    seed: Dict[str, Any] = {}

    existing_seed = working_snapshot.get("world_seed")
    if isinstance(existing_seed, Mapping):
        seed = _merge_seed_payload(seed, existing_seed)

    world_payload = (meta or {}).get("world") if isinstance(meta, Mapping) else None
    if isinstance(world_payload, Mapping):
        seed = _merge_seed_payload(seed, world_payload)

    supplemental_keys = ("world_seed", "world_seed_snapshot", "world_profile", "world_context")
    for key in supplemental_keys:
        payload = meta.get(key) if isinstance(meta, Mapping) else None
        if isinstance(payload, Mapping):
            seed = _merge_seed_payload(seed, payload)

    scene_scope = meta.get("scene_scope") if isinstance(meta, Mapping) else None
    if isinstance(scene_scope, Mapping):
        world_scope = scene_scope.get("world")
        if isinstance(world_scope, Mapping):
            seed = _merge_seed_payload(seed, world_scope)

    anchor_hints = anchor.hints if isinstance(getattr(anchor, "hints", None), Mapping) else {}
    world_hint = anchor_hints.get("world") if isinstance(anchor_hints, Mapping) else None
    if isinstance(world_hint, Mapping):
        seed = _merge_seed_payload(seed, world_hint)

    if anchor.primary_city:
        seed.setdefault("primary_city", anchor.primary_city)
    if anchor.region:
        seed.setdefault("region", anchor.region)
    if anchor.country:
        seed.setdefault("country", anchor.country)

    geo_seed = dict(seed.get("geo_anchor") or {})
    if anchor.lat is not None:
        geo_seed.setdefault("lat", anchor.lat)
    if anchor.lon is not None:
        geo_seed.setdefault("lon", anchor.lon)
    if anchor.primary_city and anchor.primary_city.strip():
        geo_seed.setdefault("city", anchor.primary_city)
    if geo_seed:
        seed["geo_anchor"] = geo_seed

    world_name = (
        seed.get("name")
        or seed.get("world_name")
        or anchor.world_name
        or anchor.label
        or "Fictional World"
    )
    seed.setdefault("name", world_name)
    seed.setdefault("world_name", world_name)

    working_snapshot["world_seed"] = seed
    store.put(user_id, conversation_id, working_snapshot)

    return seed, working_snapshot


async def _fetch_fictional_venues(
    user_id: str,
    conversation_id: str,
    *,
    district: Optional[Location],
    query_text: str,
) -> List[Location]:
    try:
        user_key = int(user_id)
        conversation_key = int(conversation_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id and conversation_id must be convertible to int") from exc

    normalized_query = _normalize_search_token(query_text)
    pattern = f"%{normalized_query}%" if normalized_query else None
    district_name = _stringify(district.district or district.location_name) if district else None

    async with get_db_connection_context() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT *
                FROM Locations
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND COALESCE(LOWER(location_type), '') = 'venue'
                  AND LOWER(COALESCE(scope, CASE WHEN is_fictional THEN 'fictional' ELSE 'real' END)) = 'fictional'
                  AND ($3::TEXT IS NULL OR LOWER(COALESCE(district, parent_location, '')) = LOWER($3))
                  AND ($4::TEXT IS NULL OR LOWER(location_name) LIKE $4 OR LOWER(COALESCE(description, '')) LIKE $4)
                ORDER BY location_id
                """,
                user_key,
                conversation_key,
                district_name,
                pattern,
            )
        except asyncpg.UndefinedColumnError:
            rows = await conn.fetch(
                """
                SELECT *
                FROM Locations
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND COALESCE(LOWER(location_type), '') = 'venue'
                  AND is_fictional = TRUE
                  AND ($3::TEXT IS NULL OR LOWER(COALESCE(district, parent_location, '')) = LOWER($3))
                  AND ($4::TEXT IS NULL OR LOWER(location_name) LIKE $4 OR LOWER(COALESCE(description, '')) LIKE $4)
                ORDER BY location_id
                """,
                user_key,
                conversation_key,
                district_name,
                pattern,
            )

    return [Location.from_record(row) for row in rows]


def _select_target_district(
    districts: List[Location],
    query: PlaceQuery,
    anchor: Anchor,
    meta: Mapping[str, Any],
) -> Optional[Location]:
    if not districts:
        return None

    phrase = _normalize_search_token(query.target or query.normalized)
    if phrase:
        for district in districts:
            profile = _district_profile_from_location(district)
            candidate_tokens = {
                _normalize_search_token(profile.get("name")),
                _normalize_search_token(profile.get("key")),
                _normalize_search_token(district.location_name),
                _normalize_search_token(district.district),
            }
            if any(token and token in phrase for token in candidate_tokens):
                return district

    scene_scope = meta.get("scene_scope") if isinstance(meta, Mapping) else None
    if isinstance(scene_scope, Mapping):
        for hint_key in ("district", "neighborhood", "area"):
            hint = _normalize_search_token(scene_scope.get(hint_key))
            if hint:
                for district in districts:
                    profile = _district_profile_from_location(district)
                    candidate_tokens = {
                        _normalize_search_token(profile.get("name")),
                        _normalize_search_token(profile.get("key")),
                        _normalize_search_token(district.location_name),
                        _normalize_search_token(district.district),
                    }
                    if any(token and (token == hint or hint in token or token in hint) for token in candidate_tokens):
                        return district

    anchor_hint: Optional[str] = None
    if anchor and isinstance(anchor.hints, Mapping):
        geo_anchor = anchor.hints.get("geo_anchor")
        if geo_anchor is not None:
            anchor_hint = getattr(geo_anchor, "neighborhood", None) or getattr(geo_anchor, "district", None)
    if not anchor_hint and anchor and anchor.focus and anchor.focus.meta:
        anchor_hint = anchor.focus.meta.get("district")

    normalized_hint = _normalize_search_token(anchor_hint)
    if normalized_hint:
        for district in districts:
            profile = _district_profile_from_location(district)
            candidate_tokens = {
                _normalize_search_token(profile.get("name")),
                _normalize_search_token(profile.get("key")),
                _normalize_search_token(district.location_name),
                _normalize_search_token(district.district),
            }
            if any(token and (token == normalized_hint or normalized_hint in token) for token in candidate_tokens):
                return district

    return districts[0]


def _update_city_graph_snapshot(
    store: ConversationSnapshotStore,
    user_id: str,
    conversation_id: str,
    snapshot: Dict[str, Any],
    districts: List[Location],
    venues: List[Location],
) -> None:
    graph_districts: List[Dict[str, Any]] = []
    for district in districts:
        profile = _district_profile_from_location(district)
        graph_districts.append(
            {
                "key": profile.get("key"),
                "label": profile.get("name"),
                "vibe": profile.get("vibe"),
                "lat": district.lat,
                "lon": district.lon,
            }
        )

    graph_venues: List[Dict[str, Any]] = []
    for venue in venues:
        profile = _extract_poi_profile(venue)
        graph_venues.append(
            {
                "name": _display_name_for_location(venue, profile),
                "district": venue.district,
                "district_key": profile.get("key") if isinstance(profile, Mapping) else None,
                "lat": venue.lat,
                "lon": venue.lon,
                "category": profile.get("category") if isinstance(profile, Mapping) else None,
                "location_id": venue.id,
            }
        )

    snapshot_payload = dict(snapshot)
    snapshot_payload["city_graph"] = {
        "districts": graph_districts,
        "venues": graph_venues,
        "pack_loaded": True,
    }
    store.put(user_id, conversation_id, snapshot_payload)


async def generate_pois_for_district(district: Location, query: str) -> List[Location]:
    normalized_query = unicodedata.normalize("NFKC", query or "").strip()
    pattern = f"%{normalized_query.lower()}%" if normalized_query else None

    district_key = district.location_name
    if not district_key:
        raise ValueError("District must have a location_name for POI generation")

    user_id = int(district.user_id)
    conversation_id = int(district.conversation_id)

    async with get_db_connection_context() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT *
                FROM Locations
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND LOWER(COALESCE(district, parent_location, '')) = LOWER($3)
                  AND COALESCE(LOWER(location_type), '') = 'venue'
                  AND LOWER(COALESCE(scope, CASE WHEN is_fictional THEN 'fictional' ELSE 'real' END)) = 'fictional'
                  AND ($4::TEXT IS NULL OR LOWER(location_name) LIKE $4 OR LOWER(COALESCE(metadata::TEXT, '')) LIKE $4)
                ORDER BY location_id
                """,
                user_id,
                conversation_id,
                district_key,
                pattern,
            )
        except asyncpg.UndefinedColumnError:
            rows = await conn.fetch(
                """
                SELECT *
                FROM Locations
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND LOWER(COALESCE(district, parent_location, '')) = LOWER($3)
                  AND COALESCE(LOWER(location_type), '') = 'venue'
                  AND is_fictional = TRUE
                  AND ($4::TEXT IS NULL OR LOWER(location_name) LIKE $4)
                ORDER BY location_id
                """,
                user_id,
                conversation_id,
                district_key,
                pattern,
            )

    if rows:
        return [Location.from_record(row) for row in rows]

    profile = _district_profile_from_location(district)
    world_context = _extract_world_seed_context(district)

    profile_text = json.dumps(
        {
            key: value
            for key, value in profile.items()
            if key in {"name", "vibe", "layout", "theme", "summary"} and value
        },
        indent=2,
        ensure_ascii=False,
    )
    world_seed_text = (
        json.dumps(world_context, indent=2, ensure_ascii=False)
        if world_context
        else "No explicit world seed guidance available."
    )

    prompt = (
            f"Design evocative fictional points of interest within the district \"{profile.get('name', district_key)}\".\n"
            f"Player query: {normalized_query or 'general exploration request'}.\n"
            f"District profile:\n{profile_text}\n\n"
            f"World seed guidance:\n{world_seed_text}\n\n"
            "Respond with JSON only. Return an object with a `pois` array. Each POI must include: "
            "name, category, description, lore, travel (with modes array and time_minutes number), notable_features, "
            "and coordinates with `dx_m` and `dy_m` offsets in meters from the district center.\n\n"
            "### Example of the required JSON format:\n"
            "```json\n"
            "{\n"
            '  "pois": [\n'
            '    {\n'
            '      "name": "The Gilded Compass",\n'
            '      "category": "Tavern",\n'
            '      "description": "A rowdy tavern built into the hull of a decommissioned airship, known for its smuggled spirits and whispered secrets.",\n'
            '      "lore": "They say the captain of the ship, One-Eyed Maeve, still haunts the bar, searching for her lost treasure map.",\n'
            '      "travel": {\n'
            '        "modes": ["walk", "ferry"],\n'
            '        "time_minutes": 15\n'
            '      },\n'
            '      "notable_features": ["Serves a glowing blue ale", "The bar is carved from a single piece of driftwood"],\n'
            '      "coordinates": {\n'
            '        "dx_m": -250,\n'
            '        "dy_m": 120\n'
            '      }\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```"
        )

    try:
        llm_payload = await call_gpt_json(
            conversation_id,
            context=f"Fictional POIs for {profile.get('name', district_key)}",
            prompt=prompt,
            model="gpt-5-nano",
        )
    except Exception as exc:
        logger.error("Fictional POI generation call failed for %s: %s", district_key, exc, exc_info=True)
        llm_payload = {}

    specs = _normalize_generated_pois(llm_payload if isinstance(llm_payload, Mapping) else {})
    if not specs:
        logger.info("LLM did not return usable POIs for %s", district_key)
        return []

    persisted: List[Location] = []
    async with get_db_connection_context() as conn:
        async with conn.transaction():
            for spec in specs:
                try:
                    lat, lon, dx_m, dy_m = _offset_coordinates(
                        district.lat,
                        district.lon,
                        dx_m=spec.get("dx_m"),
                        dy_m=spec.get("dy_m"),
                        seed_key=f"{user_id}:{conversation_id}:{district_key}:{spec['name']}",
                    )

                    address: Dict[str, Any] = {"district": district.location_name}
                    if district.city:
                        address["city"] = district.city
                    if district.region:
                        address["region"] = district.region
                    if district.country:
                        address["country"] = district.country

                    travel_payload: Dict[str, Any] = {
                        "query": normalized_query,
                        "modes": spec.get("travel_modes") or [],
                        "time_minutes": spec.get("travel_time_minutes"),
                        "note": spec.get("travel_note"),
                        "offset_meters": {"dx_m": dx_m, "dy_m": dy_m},
                    }

                    local_customs = [
                        {
                            "kind": "fictional_poi_profile",
                            "name": spec["name"],
                            "category": spec.get("category"),
                            "lore": spec.get("lore"),
                            "travel": travel_payload,
                            "district": district.location_name,
                        }
                    ]

                    meta: Dict[str, Any] = {
                        "source": "fictional",
                        "display_name": spec["name"],
                        "description": spec.get("description"),
                        "location_type": "venue",
                        "district": district.location_name,
                        "parent_location": district.location_name,
                        "city": district.city,
                        "region": district.region,
                        "country": district.country,
                        "planet": district.planet,
                        "galaxy": district.galaxy,
                        "realm": district.realm,
                        "is_fictional": True,
                        "notable_features": spec.get("features") or [],
                        "local_customs": local_customs,
                        "travel_modes": spec.get("travel_modes") or [],
                        "travel_time_minutes": spec.get("travel_time_minutes"),
                        "travel_note": spec.get("travel_note"),
                        "poi_category": spec.get("category"),
                        "poi_lore": spec.get("lore"),
                        "poi_query": normalized_query,
                        "poi_offset": {"dx_m": dx_m, "dy_m": dy_m},
                    }

                    place = Place(
                        name=spec["name"],
                        level="venue",
                        lat=lat,
                        lon=lon,
                        address=address,
                        meta=meta,
                    )

                    candidate = Candidate(
                        place=place,
                        confidence=0.75,
                        rationale="fictional_district_poi",
                        raw={"query": normalized_query, "spec": spec.get("raw", {})},
                    )

                    location = await generate_and_persist_hierarchy(
                        conn,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        candidate=candidate,
                        scope="fictional",
                    )
                    persisted.append(location)
                except Exception as exc:
                    logger.error(
                        "Failed to persist fictional POI '%s' for district %s: %s",
                        spec.get("name"),
                        district_key,
                        exc,
                        exc_info=True,
                    )

    return persisted


async def get_or_generate_districts(
    world_seed: Dict[str, Any],
    *,
    user_id: str,
    conversation_id: str,
    city_name: str,
) -> List[Location]:
    seed: Dict[str, Any] = dict(world_seed or {})
    city = _stringify(city_name) or seed.get("primary_city") or seed.get("name") or "Fictional City"
    seed.setdefault("primary_city", city)

    try:
        user_key = int(user_id)
        conversation_key = int(conversation_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id and conversation_id must be convertible to int") from exc

    async with get_db_connection_context() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT *
                FROM Locations
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND LOWER(COALESCE(city, parent_location, '')) = LOWER($3)
                  AND COALESCE(LOWER(location_type), '') = 'district'
                  AND LOWER(COALESCE(scope, CASE WHEN is_fictional THEN 'fictional' ELSE 'real' END)) = 'fictional'
                ORDER BY location_id
                """,
                user_key,
                conversation_key,
                city,
            )
        except asyncpg.UndefinedColumnError:
            rows = await conn.fetch(
                """
                SELECT *
                FROM Locations
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND LOWER(COALESCE(city, parent_location, '')) = LOWER($3)
                  AND COALESCE(LOWER(location_type), '') = 'district'
                  AND is_fictional = TRUE
                ORDER BY location_id
                """,
                user_key,
                conversation_key,
                city,
            )

    if rows:
        return [Location.from_record(row) for row in rows]

    seed_summary = _format_world_seed(seed, city)
    context_name = _extract_text_value(seed, _WORLD_NAME_PATHS) or city
    prompt = (
            f"You are a meticulous world-builder designing distinct districts for the fictional city \"{city}\".\n"
            f"World seed insights:\n{seed_summary}\n\n"
            "Return a JSON object with a `districts` array containing three to five entries. "
            "Each district must provide: key, name, vibe, layout, theme, summary, and a list of 2-4 distinctive features. "
            "Focus on relative placement and differentiated vibes. Respond with JSON only.\n\n"
            "### Example of the required JSON format:\n"
            "```json\n"
            "{\n"
            '  "districts": [\n'
            '    {\n'
            '      "key": "iron_quarter",\n'
            '      "name": "The Iron Quarter",\n'
            '      "vibe": "Industrial heart of the city, filled with smog and the clang of machinery.",\n'
            '      "layout": "A rigid grid of warehouses and factories south of the river.",\n'
            '      "theme": "Steampunk industry and labor movements",\n'
            '      "summary": "Where the city\'s gears are forged and its workforce toils.",\n'
            '      "features": ["Automated cargo trains", "Towering smokestacks"]\n'
            '    },\n'
            '    {\n'
            '      "key": "whisperwood",\n'
            '      "name": "Whisperwood",\n'
            '      "vibe": "An ancient, overgrown park that has reclaimed the city\'s northern edge.",\n'
            '      "layout": "A sprawling, untamed forest with winding paths and hidden clearings.",\n'
            '      "theme": "Nature\'s resilience and forgotten magic",\n'
            '      "summary": "A place of quiet mystery, rumored to be older than the city itself.",\n'
            '      "features": ["Glowing flora at night", "The ruins of an old observatory"]\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```"
        )

    try:
        llm_payload = await call_gpt_json(
            conversation_key,
            context=f"World-building for {context_name}",
            prompt=prompt,
            model="gpt-5-nano",
        )
    except Exception as exc:
        logger.warning("Fictional district generation call failed: %s", exc, exc_info=True)
        llm_payload = {}

    specs = _normalize_generated_districts(llm_payload if isinstance(llm_payload, Mapping) else {})
    if not specs:
        logger.info("Falling back to default district templates for %s", city)
        specs = _fallback_districts(city)

    lat_hint, lon_hint = _extract_lat_lon(seed)
    seed_base = f"{user_key}:{conversation_key}:{city.lower()}"
    if lat_hint is None:
        lat_hint = 37.7749 + random.Random(f"{seed_base}:lat").uniform(-0.5, 0.5)
    if lon_hint is None:
        lon_hint = -122.4194 + random.Random(f"{seed_base}:lon").uniform(-0.5, 0.5)

    region = _extract_text_value(seed, _REGION_PATHS)
    country = _extract_text_value(seed, _COUNTRY_PATHS)
    planet = _extract_text_value(seed, _PLANET_PATHS)
    galaxy = _extract_text_value(seed, _GALAXY_PATHS)
    realm = _extract_text_value(seed, _REALM_PATHS)
    world_name = _extract_text_value(seed, _WORLD_NAME_PATHS)

    persisted: List[Location] = []
    async with get_db_connection_context() as conn:
        async with conn.transaction():
            for spec in specs:
                key = spec["key"]
                rng = random.Random(f"{seed_base}:{key}")
                lat = spec.get("lat") or (lat_hint + rng.uniform(-0.03, 0.03))
                lon = spec.get("lon") or (lon_hint + rng.uniform(-0.03, 0.03))
                spec["lat"] = round(lat, 6)
                spec["lon"] = round(lon, 6)

                description = spec.get("description") or _compose_description(
                    spec.get("vibe", ""),
                    spec.get("layout", ""),
                    spec.get("theme", ""),
                    spec.get("summary", ""),
                )
                spec["description"] = description
                features = spec.get("features") or []
                if not features and description:
                    features = [description]
                spec["features"] = features

                address: Dict[str, Any] = {"city": city}
                if region:
                    address["region"] = region
                if country:
                    address["country"] = country

                meta: Dict[str, Any] = {
                    "source": "fictional",
                    "display_name": spec["name"],
                    "description": description,
                    "location_type": "district",
                    "city": city,
                    "region": region,
                    "country": country,
                    "planet": planet,
                    "galaxy": galaxy,
                    "realm": realm,
                    "world_name": world_name,
                    "district": spec["name"],
                    "district_key": key,
                    "district_vibe": spec.get("vibe"),
                    "district_layout": spec.get("layout"),
                    "district_theme": spec.get("theme"),
                    "district_summary": spec.get("summary"),
                    "notable_features": features,
                    "local_customs": [
                        {
                            "kind": "district_profile",
                            "key": key,
                            "name": spec["name"],
                            "vibe": spec.get("vibe"),
                            "layout": spec.get("layout"),
                            "theme": spec.get("theme"),
                            "summary": spec.get("summary"),
                        }
                    ],
                    "is_fictional": True,
                    "lat": spec["lat"],
                    "lon": spec["lon"],
                }

                # Strip None values to keep payload lean
                meta = {key_: value for key_, value in meta.items() if value is not None}

                try:
                    place = Place(
                        name=spec["name"],
                        level="district",
                        lat=spec["lat"],
                        lon=spec["lon"],
                        address=address,
                        meta=meta,
                    )
                    candidate = Candidate(
                        place=place,
                        confidence=0.85,
                        rationale="fictional_district_seed",
                    )
                    location = await generate_and_persist_hierarchy(
                        conn,
                        user_id=user_key,
                        conversation_id=conversation_key,
                        candidate=candidate,
                        scope="fictional",
                    )
                    persisted.append(location)
                except Exception as exc:
                    logger.error(
                        "Failed to persist fictional district '%s' for %s: %s",
                        spec.get("name"),
                        city,
                        exc,
                        exc_info=True,
                    )

    if not persisted:
        async with get_db_connection_context() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM Locations
                    WHERE user_id = $1
                      AND conversation_id = $2
                      AND LOWER(COALESCE(city, parent_location, '')) = LOWER($3)
                      AND COALESCE(LOWER(location_type), '') = 'district'
                    ORDER BY location_id
                    """,
                    user_key,
                    conversation_key,
                    city,
                )
                if rows:
                    return [Location.from_record(row) for row in rows]
            except Exception:
                pass

    return persisted


async def resolve_fictional(
    query: PlaceQuery,
    anchor: Anchor,
    meta: Dict[str, Any],
    store: ConversationSnapshotStore,
    user_id: str,
    conversation_id: str,
) -> ResolveResult:
    if anchor.lat is None or anchor.lon is None:
        try:
            geo = await derive_geo_anchor(meta, user_id=user_id, conversation_id=conversation_id)
            anchor.lat = geo.lat
            anchor.lon = geo.lon
        except Exception:
            pass

    user_key = str(user_id)
    conv_key = str(conversation_id)
    world_seed, snapshot = await load_or_generate_world_seed(anchor, meta, store, user_key, conv_key)

    city_name = (
        world_seed.get("primary_city")
        or anchor.primary_city
        or world_seed.get("name")
        or world_seed.get("world_name")
        or anchor.world_name
        or "Fictional City"
    )

    districts = await get_or_generate_districts(
        world_seed,
        user_id=user_key,
        conversation_id=conv_key,
        city_name=city_name,
    )

    if not (query.target or query.normalized):
        return ResolveResult(
            status=STATUS_ASK,
            message=f"In {city_name}, what kind of place are you seeking?",
            anchor=anchor,
            scope="fictional",
        )

    target_district = _select_target_district(districts, query, anchor, meta)
    if target_district is None:
        return ResolveResult(
            status=STATUS_NOT_FOUND, # --- CHANGE: Return NOT_FOUND if no district can be determined ---
            message="I can shape the city further—but couldn't determine a district from your request.",
            anchor=anchor,
            scope="fictional",
        )

    venues = await _fetch_fictional_venues(
        user_key,
        conv_key,
        district=target_district,
        query_text=query.target or query.normalized or "",
    )

    # --- CHANGE STARTS HERE: Improved resilience for POI generation ---
    if not venues:
        # Attempt to generate POIs, and capture the result directly.
        generated_venues = await generate_pois_for_district(target_district, query.target or query.normalized or "")
        
        # If generation fails to produce any venues, we have definitively not found the location.
        if not generated_venues:
            return ResolveResult(
                status=STATUS_NOT_FOUND,
                message=f"I couldn't find or create a location like '{query.target}' in {target_district.district or city_name}.",
                anchor=anchor,
                scope="fictional",
            )
        # If generation succeeds, use the newly created venues for the rest of this function.
        venues = generated_venues
    # --- CHANGE ENDS HERE ---

    phrase_candidates: List[str] = []
    if query.target:
        phrase_candidates.append(_normalize_search_token(query.target))
    if query.normalized and query.normalized != query.target:
        phrase_candidates.append(_normalize_search_token(query.normalized))
    phrase_candidates = [phrase for phrase in phrase_candidates if phrase]

    word_tokens: List[str] = []
    for phrase in phrase_candidates:
        word_tokens.extend(token for token in phrase.split(" ") if token)
    word_tokens = list({token for token in word_tokens if token})

    scored: List[Tuple[float, Location]] = []
    for venue in venues:
        profile = _extract_poi_profile(venue)
        score = _score_location_match(venue, profile, phrase_candidates, word_tokens)
        scored.append((score, venue))
    
    # --- CHANGE: Handle case where scoring produces no viable candidates ---
    if not scored:
        return ResolveResult(
            status=STATUS_NOT_FOUND,
            message=f"While I found some places, none seemed to match '{query.target}'.",
            anchor=anchor,
            scope="fictional",
        )
    # --- CHANGE ENDS HERE ---

    scored.sort(key=lambda item: item[0], reverse=True)
    best_location = scored[0][1]
    candidate, profile = _build_candidate_from_location(best_location)
    display_name = _display_name_for_location(best_location, profile)
    operations = _build_navigation_operations(best_location, display_name, profile)

    _update_city_graph_snapshot(store, user_key, conv_key, snapshot, districts, venues)

    message_context = best_location.district or target_district.district or city_name
    message = f"Heading to {display_name} in {message_context}."

    return ResolveResult(
        status=STATUS_EXACT,
        message=message,
        candidates=[candidate],
        operations=operations,
        anchor=anchor,
        scope="fictional",
        location=best_location,
    )
