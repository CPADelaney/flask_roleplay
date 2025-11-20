# nyx/location/router.py

from __future__ import annotations

import asyncio
import json
import copy
import logging
import math
import re
import time
import unicodedata
from dataclasses import asdict
from typing import Any, Dict, Optional

import asyncpg

from db.connection import get_db_connection_context, skip_vector_registration

from . import gemini_maps_adapter as gmaps
from .anchors import derive_geo_anchor
from .fictional_resolver import resolve_fictional
from .gemini_maps_adapter import resolve_location_with_gemini
from .query import PlaceQuery
from .search import resolve_real
from .types import (
    Anchor,
    Candidate,
    Place,
    ResolveResult,
    Scope,
    STATUS_EXACT,
    STATUS_MULTIPLE,
    STATUS_TRAVEL_PLAN,
    STATUS_ASK,
    STATUS_NOT_FOUND,
)

from nyx.location.hierarchy import get_or_create_location
from nyx.conversation.snapshot_store import ConversationSnapshotStore
from monitoring.metrics import metrics
from utils.cache_manager import CacheManager
from nyx.tasks.background import place_enrichment

logger = logging.getLogger(__name__)


# Cached metadata for Disneyland Park to allow fast-path resolution when
# the player is already anchored there and issues another movement request.
DISNEYLAND_PARK_SHORTCUT = {
    "name": "Disneyland Park",
    "aliases": (
        "Disneyland Park",
        "Disneyland",
        "the Disneyland Park",
    ),
    "place_id": "ChIJPeY5iQ3t3IARfOMl5TP3f5s",
    "city": "Anaheim",
    "region": "California",
    "country": "United States",
    "lat": 33.8121,
    "lon": -117.9190,
}


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _persist_gmaps_place(
    user_id: str,
    conversation_id: str,
    place: gmaps.PlaceResult,
) -> None:
    """Ensure the resolved real-world place and districts exist in Locations."""

    user_key = _coerce_int(user_id)
    convo_key = _coerce_int(conversation_id)
    if user_key is None or convo_key is None:
        return

    try:
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO public.locations (
                    user_id, conversation_id, location_name,
                    external_place_id, location_type, city, country,
                    lat, lon, scope, is_fictional
                )
                VALUES ($1, $2, $3,
                        $4, $5, $6, $7,
                        $8, $9, 'real', FALSE)
                ON CONFLICT (user_id, conversation_id, location_name_lc) DO UPDATE
                SET external_place_id = COALESCE(EXCLUDED.external_place_id, public.locations.external_place_id),
                    location_type     = COALESCE(EXCLUDED.location_type,     public.locations.location_type),
                    city              = COALESCE(EXCLUDED.city,              public.locations.city),
                    country           = COALESCE(EXCLUDED.country,           public.locations.country),
                    scope             = 'real',
                    lat               = COALESCE(EXCLUDED.lat,               public.locations.lat),
                    lon               = COALESCE(EXCLUDED.lon,               public.locations.lon),
                    is_fictional      = FALSE
                """,
                user_key,
                convo_key,
                place.name,
                place.place_id,
                "venue" if place.districts else None,
                place.city,
                place.country,
                place.lat,
                place.lon,
            )

            for district in place.districts:
                await conn.execute(
                    """
                    INSERT INTO public.locations (
                        user_id, conversation_id, location_name,
                        external_place_id, parent_location, location_type,
                        city, country, lat, lon, scope, is_fictional
                    )
                    VALUES ($1, $2, $3,
                            $4, $5, 'district',
                            $6, $7, $8, $9, 'real', FALSE)
                    ON CONFLICT (user_id, conversation_id, location_name_lc) DO UPDATE
                    SET external_place_id = COALESCE(EXCLUDED.external_place_id, public.locations.external_place_id),
                        parent_location   = COALESCE(EXCLUDED.parent_location,   public.locations.parent_location),
                        city              = COALESCE(EXCLUDED.city,              public.locations.city),
                        country           = COALESCE(EXCLUDED.country,           public.locations.country),
                        scope             = 'real',
                        lat               = COALESCE(EXCLUDED.lat,               public.locations.lat),
                        lon               = COALESCE(EXCLUDED.lon,               public.locations.lon),
                        is_fictional      = FALSE
                    """,
                    user_key,
                    convo_key,
                    district.name,
                    district.place_id,
                    place.name,
                    place.city,
                    place.country,
                    district.lat,
                    district.lon,
                )
    except Exception:
        logger.debug("[ROUTER] Failed to persist Gemini Maps place", exc_info=True)


async def _maybe_resolve_via_gmaps_fastpath(
    query: PlaceQuery,
    anchor: Anchor,
    meta: Dict[str, Any],
    user_id: str,
    conversation_id: str,
) -> Optional[ResolveResult]:
    """Attempt a lightweight Gemini Maps lookup before the heavy real chain."""

    if not gmaps.is_enabled():
        return None
    if query.is_travel:
        return None

    candidate_query = (
        query.target
        or query.raw_text
        or meta.get("location")
        or anchor.label
        or anchor.primary_city
    )
    if not candidate_query:
        return None

    try:
        place = await gmaps.resolve_place_and_districts(candidate_query)
    except Exception:
        logger.debug("[ROUTER] Gemini Maps adapter fast path failed", exc_info=True)
        return None

    if not place:
        return None

    await _persist_gmaps_place(user_id, conversation_id, place)

    level = "venue" if place.districts else ("city" if place.city else "unknown")
    place_meta = {
        "source": "gemini_maps",
        "gmaps_place_id": place.place_id,
    }
    if place.districts:
        place_meta["districts"] = [asdict(d) for d in place.districts]

    candidate_place = Place(
        name=place.name,
        level=level,
        key=place.place_id,
        lat=place.lat,
        lon=place.lon,
        address={
            key: value
            for key, value in {
                "city": place.city,
                "country": place.country,
            }.items()
            if value
        },
        meta=place_meta,
    )

    candidate = Candidate(
        place=candidate_place,
        confidence=0.92,
        rationale="Resolved via Gemini Maps adapter",
    )
    result = ResolveResult(
        status=STATUS_EXACT,
        message=f"Resolved '{candidate_query}' via Gemini Maps adapter",
        candidates=[candidate],
        anchor=anchor,
        scope="real",
    )
    result.metadata = {
        "gmaps": {
            "place_id": place.place_id,
            "city": place.city,
            "country": place.country,
            "districts": [asdict(d) for d in place.districts],
        }
    }
    logger.info(
        "[ROUTER] Using Gemini Maps adapter fast path for '%s'", candidate_query
    )
    return result

LOCATION_RESOLUTION_CACHE = CacheManager(
    name="location_resolution_cache", max_size=200, ttl=120
)


REAL_WORLD_AFC_MAX_CALLS = 3

async def _track_player_movement(
    user_id: str,
    conversation_id: str,
    location_name: str,
    location_type: Optional[str] = None,
    city: Optional[str] = None,
) -> None:
    """Track player movement in canon system."""
    try:
        from db.connection import get_db_connection_context
        from lore.core.canon import update_current_roleplay, log_canonical_event
        from lore.core.context import CanonicalContext
        
        async with get_db_connection_context() as conn:
            ctx = CanonicalContext(
                user_id=int(user_id), 
                conversation_id=int(conversation_id)
            )

            # Update current location
            await update_current_roleplay(ctx, conn, 'CurrentLocation', location_name)

            city_value = city.strip() if isinstance(city, str) else None
            if city_value:
                await update_current_roleplay(ctx, conn, 'CurrentCity', city_value)

            # Log movement with appropriate significance
            significance = 6 if location_type in ['city', 'district'] else 4
            event_details = (
                f"Player moved to {location_name}"
                if not city_value
                else f"Player moved to {location_name} in {city_value}"
            )
            await log_canonical_event(
                ctx, conn,
                event_details,
                tags=['movement', 'location', 'player', location_type or 'venue'],
                significance=significance,
                persist_memory=True
            )

            logger.info(
                "✓ Tracked player movement to: %s (city=%s)",
                location_name,
                city_value or 'unknown',
            )

    except Exception as e:
        logger.warning(f"Failed to track player movement: {e}", exc_info=True)

# Regular expressions to parse user intent for movement or travel.
_GO_TO_RX = re.compile(r"\b(?:go|head|walk|run|drive|get|straight|toward|to)\s+(?:the\s+)?(.+)$", re.IGNORECASE)
_FLY_TO_RX = re.compile(r"\b(?:fly|flight)\s+(?:to|for)\s+(.+)$", re.IGNORECASE)

def _parse_place_query(text: str) -> PlaceQuery:
    """Parses raw user text to extract a target location and travel hints."""
    t = (text or "").strip()
    if not t:
        return PlaceQuery(raw_text="", normalized="")

    m2 = _FLY_TO_RX.search(t)
    if m2:
        target = m2.group(1).strip().rstrip(".!?")
        if target.lower().startswith("to "):
            target = target[3:].lstrip()
        return PlaceQuery(raw_text=t, normalized=target.lower(), is_travel=True, target=target, transport_hint="fly")

    m = _GO_TO_RX.search(t)
    target = (m.group(1) if m else t).strip().rstrip(".!?")
    if target.lower().startswith("to "):
        target = target[3:].lstrip()
    return PlaceQuery(raw_text=t, normalized=target.lower(), target=target)


def _enrich_metadata_with_intent(user_text: str, meta: Dict[str, Any]) -> None:
    """
    Analyze user text for common location patterns and enrich metadata
    to guide world generation appropriately.
    """
    text_lower = user_text.lower()
    world = meta.setdefault("world", {})
    
    # Theme park patterns
    theme_park_keywords = [
        "disneyland",
        "disney",
        "theme park",
        "amusement park",
        "six flags",
        "universal studios",
    ]
    if any(keyword in text_lower for keyword in theme_park_keywords):
        world_type = (world.get("type") or world.get("kind") or "").strip().lower()
        scope_explicit_real = (
            world_type in {"real", "modern_realistic", "realistic", "historical", "modern"}
            or world.get("real_world_based") is True
            or world.get("use_real_locations") is True
            or meta.get("use_google_maps") is True
            or meta.get("enable_google_maps") is True
        )

        if scope_explicit_real:
            logger.info("Detected theme park intent but scope is explicitly real; skipping enrichment")
        else:
            theme_hints = ["entertainment", "theme park", "family-friendly", "attractions"]
            existing_themes = world.get("themes")
            if isinstance(existing_themes, list):
                for hint in theme_hints:
                    if hint not in existing_themes:
                        existing_themes.append(hint)
            elif existing_themes:
                combined_themes = [str(existing_themes), *theme_hints]
                deduped_themes = []
                for hint in combined_themes:
                    if hint not in deduped_themes:
                        deduped_themes.append(hint)
                world["themes"] = deduped_themes
            else:
                world["themes"] = theme_hints

            world.setdefault("tone", "whimsical")
            world.setdefault("technology_level", "modern")

            logger.info("Detected theme park intent, enriching metadata with hints")
            logger.debug(
                "Theme park enrichment added hints without overriding scope (world.type=%s)",
                world.get("type"),
            )
            return
    
    # Fantasy locations
    if any(keyword in text_lower for keyword in ["castle", "kingdom", "realm", "shire", "rivendell", "hogwarts", "narnia"]):
        world.update({
            "type": "fictional",
            "themes": ["fantasy", "magic", "medieval"],
            "tone": "epic",
            "genre": "high fantasy",
            "technology_level": "medieval"
        })
        logger.info("Detected fantasy location intent, enriching metadata")
        return
    
    # Sci-fi locations
    if any(keyword in text_lower for keyword in ["space station", "starship", "colony", "mars", "cyberpunk", "neo tokyo"]):
        world.update({
            "type": "fictional",
            "themes": ["science fiction", "futuristic", "high tech"],
            "tone": "futuristic",
            "genre": "sci-fi",
            "technology_level": "advanced"
        })
        logger.info("Detected sci-fi location intent, enriching metadata")
        return
    
    # Horror/dark locations
    if any(keyword in text_lower for keyword in ["haunted", "cemetery", "crypt", "mansion", "silent hill", "raccoon city"]):
        world.update({
            "type": "fictional",
            "themes": ["horror", "dark", "supernatural"],
            "tone": "ominous",
            "genre": "horror",
            "technology_level": "modern"
        })
        logger.info("Detected horror location intent, enriching metadata")
        return
    
    # Video game/pop culture locations
    game_locations = {
        "hyrule": {"themes": ["fantasy", "adventure"], "tone": "heroic", "genre": "adventure fantasy"},
        "gotham": {"themes": ["urban", "crime", "vigilante"], "tone": "noir", "genre": "superhero"},
        "metropolis": {"themes": ["urban", "superhero"], "tone": "bright", "genre": "superhero"},
        "rapture": {"themes": ["dystopian", "underwater"], "tone": "dark", "genre": "dystopian sci-fi"},
        "columbia": {"themes": ["steampunk", "floating city"], "tone": "fantastical", "genre": "steampunk"},
    }
    
    for location_name, settings in game_locations.items():
        if location_name in text_lower:
            world.update({
                "type": "fictional",
                "technology_level": "varied",
                **settings
            })
            logger.info(f"Detected {location_name} reference, enriching metadata")
            return


async def _anchor_from_meta(meta: Dict[str, Any], user_id: str, conversation_id: str) -> Anchor:
    """Constructs a location resolution anchor from conversation metadata."""
    world = (meta or {}).get("world") or {}
    kind_raw = world.get("type") or world.get("kind")
    kind_txt = (kind_raw or "").strip().lower()
    kind_source = "world.type" if world.get("type") else ("world.kind" if world.get("kind") else "world.type")

    real_signals: list[tuple[str, bool]] = []
    fictional_signals: list[tuple[str, bool]] = []

    def add_signal(target: list[tuple[str, bool]], signal: str, explicit: bool) -> None:
        entry = (signal, explicit)
        if entry not in target:
            target.append(entry)

    def normalize_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "y"}:
                return True
            if normalized in {"false", "no", "0", "n"}:
                return False
        return None

    real_scope_tokens = {
        "real",
        "realistic",
        "historical",
        "earth",
        "contemporary",
        "actual",
        "authentic",
        "nonfiction",
        "canon",
        "irl",
        "realworld",
        "real_world",
        "true",
    }
    fictional_scope_tokens = {
        "fictional",
        "fiction",
        "fantasy",
        "magical",
        "mythic",
        "mythical",
        "mythology",
        "legendary",
        "imaginary",
        "supernatural",
        "dystopian",
        "utopian",
        "cyberpunk",
        "steampunk",
        "futuristic",
        "alternate",
        "otherworldly",
        "realm",
        "kingdom",
        "saga",
        "myth",
        "fable",
        "fairytale",
        "fairy",
        "arcane",
        "enchanted",
        "enchant",
        "mythos",
        "dreamscape",
        "galactic",
        "interstellar",
        "spacefaring",
        "cosmic",
        "scifi",
    }
    sci_fi_hints = ("sci-fi", "science fiction", "science-fiction", "sci fi")

    if kind_txt:
        kind_tokens = [token for token in re.split(r"[^a-z0-9]+", kind_txt) if token]

        for token in kind_tokens:
            if token in real_scope_tokens:
                add_signal(real_signals, f"{kind_source}~{token}", True)
            if token in fictional_scope_tokens:
                add_signal(fictional_signals, f"{kind_source}~{token}", True)

        if any(hint in kind_txt for hint in sci_fi_hints):
            add_signal(fictional_signals, f"{kind_source} contains sci-fi hint", True)

    for container_label, container in (("meta", meta), ("world", world)):
        scope_hint = None
        if isinstance(container, dict):
            scope_hint = container.get("scope")
        if isinstance(scope_hint, str):
            normalized_scope = scope_hint.strip().lower()
            if normalized_scope == "fictional":
                add_signal(fictional_signals, f"{container_label}.scope=fictional", True)
            elif normalized_scope == "real":
                add_signal(real_signals, f"{container_label}.scope=real", True)

    for container_label, container in (("meta", meta), ("world", world)):
        if isinstance(container, dict) and "is_fictional" in container:
            normalized_bool = normalize_bool(container.get("is_fictional"))
            if normalized_bool is True:
                add_signal(fictional_signals, f"{container_label}.is_fictional=True", True)
            elif normalized_bool is False:
                add_signal(real_signals, f"{container_label}.is_fictional=False", True)

    if normalize_bool(world.get("real_world_based")) is True:
        add_signal(real_signals, "world.real_world_based=True", False)
    if normalize_bool(world.get("use_real_locations")) is True:
        add_signal(real_signals, "world.use_real_locations=True", False)
    if normalize_bool(meta.get("use_google_maps")) is True:
        add_signal(real_signals, "meta.use_google_maps=True", False)
    if normalize_bool(meta.get("enable_google_maps")) is True:
        add_signal(real_signals, "meta.enable_google_maps=True", False)

    explicit_fictional = any(explicit for _, explicit in fictional_signals)
    has_real_signals = bool(real_signals)
    has_fictional_signals = bool(fictional_signals)

    if explicit_fictional and not has_real_signals:
        scope: Scope = "fictional"
        decision_reason = "explicit-fictional"
    else:
        scope = "real"
        if has_real_signals and has_fictional_signals:
            decision_reason = "conflict-prefers-real"
        elif any(explicit for _, explicit in real_signals):
            decision_reason = "explicit-real"
        elif has_real_signals:
            decision_reason = "implicit-real"
        elif not kind_txt:
            decision_reason = "default-real-empty-kind"
        elif has_fictional_signals and not explicit_fictional:
            decision_reason = "implicit-fictional-ignored"
        else:
            decision_reason = "default-real-no-fictional"

    def format_signals(signals: list[tuple[str, bool]]) -> list[str]:
        return [f"{'explicit' if explicit else 'implicit'}:{signal}" for signal, explicit in signals]

    logger.info(
        "[ANCHOR] Scope decision -> scope=%s (reason=%s, kind=%s, fictional_signals=%s, real_signals=%s)",
        scope,
        decision_reason,
        kind_txt or "<empty>",
        format_signals(fictional_signals),
        format_signals(real_signals),
    )

    geo = await derive_geo_anchor(meta, user_id, conversation_id)
    primary_city = geo.city or world.get("primary_city")
    region = geo.region or world.get("region")
    country = geo.country or world.get("country")
    label = geo.label or primary_city or world.get("name") or world.get("world_name")

    focus: Optional[Place] = None
    if primary_city:
        focus = Place(
            name=primary_city,
            level="city",
            key=(primary_city.lower().replace(" ", "_") or None),
            lat=geo.lat,
            lon=geo.lon,
            address={
                "city": primary_city,
                "region": region,
                "country": country,
            },
            meta={"source": "geo_anchor"},
        )

    anchor = Anchor(
        scope=scope,
        focus=focus,
        label=label,
        lat=geo.lat,
        lon=geo.lon,
        primary_city=primary_city,
        region=region,
        country=country,
        world_name=world.get("name") or world.get("world_name"),
        hints={
            "world": world,
            "geo_anchor": geo,
        },
    )
    
    logger.debug(
        f"[ANCHOR] Built anchor: scope={anchor.scope}, city={anchor.primary_city}, "
        f"lat={anchor.lat}, lon={anchor.lon}, label={anchor.label}"
    )
    
    return anchor


_ALLOWED_PLACE_LEVELS = {
    "world",
    "country",
    "region",
    "state",
    "city",
    "district",
    "neighborhood",
    "venue",
    "virtual",
    "route",
    "unknown",
}


def _normalize_location_token(value: Any) -> Optional[str]:
    """Normalize free-form location text for fuzzy comparisons."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"^the\s+", "", normalized)
    normalized = re.sub(r"[^0-9a-z]+", "", normalized)
    return normalized or None


_DISNEYLAND_NORMALIZED_TOKENS = {
    token
    for token in (
        _normalize_location_token(alias)
        for alias in DISNEYLAND_PARK_SHORTCUT["aliases"]
    )
    if token
}


def _extract_current_location_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return the best-effort current location payload from metadata."""

    for key in ("currentRoleplay", "current_roleplay"):
        container = meta.get(key)
        if isinstance(container, dict):
            payload = container.get("CurrentLocation") or container.get("currentLocation")
            if isinstance(payload, dict):
                return dict(payload)
            if isinstance(payload, str):
                return {"name": payload}
    return {}


async def _ensure_real_anchor_location(
    anchor: Anchor,
    meta: Dict[str, Any],
    user_id: str,
    conversation_id: str,
) -> None:
    """Ensure the anchor location has a persisted stub for real-world scopes."""

    anchor_payload = _extract_current_location_payload(meta)
    anchor_name = (
        anchor_payload.get("name")
        or anchor_payload.get("label")
        or anchor_payload.get("display_name")
        or anchor_payload.get("location_name")
        or (anchor.focus.name if anchor.focus else None)
        or anchor.label
    )

    if not anchor_name:
        return

    try:
        uid = int(user_id)
        cid = int(conversation_id)
    except (TypeError, ValueError):
        logger.debug(
            "[ANCHOR] Unable to ensure anchor location for non-integer identifiers: user_id=%r conversation_id=%r",
            user_id,
            conversation_id,
        )
        return

    anchor_name = str(anchor_name).strip()
    if not anchor_name:
        return

    city_hint = anchor_payload.get("city") or anchor.primary_city
    country_hint = anchor_payload.get("country") or anchor.country

    try:
        with skip_vector_registration():
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO public.locations (
                      user_id, conversation_id, location_name, city, country, scope, is_fictional
                    ) VALUES ($1,$2,$3,$4,$5,'real',FALSE)
                    ON CONFLICT (user_id, conversation_id, location_name_lc) DO UPDATE
                    SET city         = COALESCE(EXCLUDED.city,    locations.city),
                        country      = COALESCE(EXCLUDED.country, locations.country),
                        scope        = 'real',
                        is_fictional = FALSE;
                    """,
                    uid,
                    cid,
                    anchor_name,
                    city_hint,
                    country_hint,
                )
    except asyncpg.UndefinedColumnError:
        logger.debug(
            "[ANCHOR] Failed to upsert anchor location due to missing generated column",
            exc_info=True,
        )
    except Exception:
        logger.debug(
            "[ANCHOR] Failed to ensure anchor location '%s' exists", anchor_name,
            exc_info=True,
        )


def _collect_location_tokens(
    anchor: Anchor,
    meta: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    """Collect known location identifiers keyed by normalized token."""

    tokens: Dict[str, Dict[str, str]] = {}

    def add(value: Any, reason: str) -> None:
        normalized = _normalize_location_token(value)
        if not normalized:
            return
        tokens.setdefault(normalized, {"value": str(value), "reason": reason})

    current_location = _extract_current_location_payload(meta)
    for key in ("name", "label", "display_name", "location_name", "slug", "key"):
        if key in current_location:
            add(current_location.get(key), f"current_roleplay.{key}")
    for key in ("id", "location_id"):
        if key in current_location:
            add(current_location.get(key), f"current_roleplay.{key}")

    for meta_key in ("location_name", "current_location", "location"):
        value = meta.get(meta_key)
        if isinstance(value, dict):
            for key in ("name", "label", "display_name", "location_name", "id", "slug"):
                if key in value:
                    add(value.get(key), f"meta.{meta_key}.{key}")
        elif isinstance(value, str):
            add(value, f"meta.{meta_key}")

    if isinstance(meta.get("location_id"), str):
        add(meta.get("location_id"), "meta.location_id")

    current_roleplay = None
    for key in ("currentRoleplay", "current_roleplay"):
        if isinstance(meta.get(key), dict):
            current_roleplay = meta[key]
            break

    if isinstance(current_roleplay, dict):
        scene_payload = current_roleplay.get("CurrentScene")
        if isinstance(scene_payload, str):
            try:
                scene_payload = json.loads(scene_payload)
            except Exception:
                scene_payload = None
        if isinstance(scene_payload, dict):
            location_payload = scene_payload.get("location")
            if isinstance(location_payload, dict):
                for key in ("name", "label", "display_name", "id", "slug"):
                    if key in location_payload:
                        add(location_payload.get(key), f"current_scene.location.{key}")
            elif isinstance(location_payload, str):
                add(location_payload, "current_scene.location")

    for key in ("location_name", "scene_id", "scene_name", "label", "slug"):
        if key in snapshot:
            add(snapshot.get(key), f"snapshot.{key}")

    if anchor.label:
        add(anchor.label, "anchor.label")
    if anchor.focus and anchor.focus.name:
        add(anchor.focus.name, "anchor.focus.name")

    hints = anchor.hints if isinstance(anchor.hints, dict) else {}
    geo_anchor = hints.get("geo_anchor")
    if geo_anchor is not None:
        for attr in ("label", "neighborhood"):
            value = getattr(geo_anchor, attr, None)
            if value:
                add(value, f"geo_anchor.{attr}")

    return tokens


def _get_conversation_snapshot(
    store: Optional[ConversationSnapshotStore],
    user_id: str,
    conversation_id: str,
) -> Dict[str, Any]:
    """Fetch the current conversation snapshot if available."""

    if store is None:
        return {}

    try:
        payload = store.get(user_id, conversation_id)
        if isinstance(payload, dict):
            return dict(payload)
    except Exception:
        logger.debug(
            "[ROUTER] Failed to fetch conversation snapshot for disneyland shortcut",
            exc_info=True,
        )
    return {}


_BASIC_MOVE_DISQUALIFIERS = (
    "ticket",
    "reservation",
    "book",
    "dinner",
    "restaurant",
    "parade",
    "event",
    "fireworks",
    "show",
)


def _is_basic_move_intent(user_text: str, query: PlaceQuery) -> bool:
    """Return True when the utterance expresses a simple movement intent."""

    if not query.target or query.is_travel:
        return False

    normalized_text = (user_text or "").strip().lower()
    if not normalized_text:
        return False

    if any(keyword in normalized_text for keyword in _BASIC_MOVE_DISQUALIFIERS):
        return False

    if _GO_TO_RX.search(user_text):
        return True

    for prefix in ("move to ", "proceed to ", "head to "):
        if normalized_text.startswith(prefix):
            return True

    return False


def _anchor_targets_disneyland(
    anchor: Anchor,
    meta: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> bool:
    """Check anchor- and snapshot-derived tokens for Disneyland Park."""

    anchor_label_token = _normalize_location_token(anchor.label)
    if anchor_label_token and anchor_label_token in _DISNEYLAND_NORMALIZED_TOKENS:
        return True

    focus = anchor.focus
    if focus is not None:
        focus_token = _normalize_location_token(focus.name)
        if focus_token and focus_token in _DISNEYLAND_NORMALIZED_TOKENS:
            return True
        focus_meta = focus.meta or {}
        focus_place_id = focus_meta.get("place_id") or focus_meta.get("id") or focus.key
        if focus_place_id and str(focus_place_id) == DISNEYLAND_PARK_SHORTCUT["place_id"]:
            return True

    tokens = _collect_location_tokens(anchor, meta, snapshot)
    if any(token in tokens for token in _DISNEYLAND_NORMALIZED_TOKENS):
        return True

    def _place_id_from_payload(payload: Dict[str, Any]) -> Optional[str]:
        for key in ("place_id", "id", "location_id", "slug"):
            value = payload.get(key)
            if value:
                return str(value)
        return None

    current_location = _extract_current_location_payload(meta)
    place_id = _place_id_from_payload(current_location)
    if place_id == DISNEYLAND_PARK_SHORTCUT["place_id"]:
        return True

    if isinstance(snapshot, dict):
        snapshot_place_id = _place_id_from_payload(snapshot)
        if snapshot_place_id == DISNEYLAND_PARK_SHORTCUT["place_id"]:
            return True

    return False


def _build_disneyland_shortcut_result(anchor: Anchor) -> ResolveResult:
    """Construct a ResolveResult using cached Disneyland Park metadata."""

    place = Place(
        name=DISNEYLAND_PARK_SHORTCUT["name"],
        level="venue",
        key=DISNEYLAND_PARK_SHORTCUT["place_id"],
        lat=DISNEYLAND_PARK_SHORTCUT["lat"],
        lon=DISNEYLAND_PARK_SHORTCUT["lon"],
        address={
            "city": DISNEYLAND_PARK_SHORTCUT["city"],
            "region": DISNEYLAND_PARK_SHORTCUT["region"],
            "country": DISNEYLAND_PARK_SHORTCUT["country"],
        },
        meta={
            "source": "anchor_disneyland_shortcut",
            "place_id": DISNEYLAND_PARK_SHORTCUT["place_id"],
        },
    )

    candidate = Candidate(place=place, confidence=0.99)
    operations = [
        {
            "op": "poi.navigate",
            "label": place.name,
            "lat": place.lat,
            "lon": place.lon,
            "context_hint": {
                "use_geo_anchor": True,
                "source": "anchor_disneyland_shortcut",
            },
        }
    ]

    result = ResolveResult(
        status=STATUS_EXACT,
        candidates=[candidate],
        operations=operations,
        anchor=anchor,
        scope=anchor.scope,
        message=f"Continuing at {place.name}.",
    )

    result.metadata = {
        "router": {
            "disneyland_shortcut": True,
            "place_id": DISNEYLAND_PARK_SHORTCUT["place_id"],
            "city": DISNEYLAND_PARK_SHORTCUT["city"],
        }
    }

    try:
        metrics().LOCATION_ROUTER_DECISIONS.labels(outcome="disneyland_anchor_shortcut").inc()
    except Exception:
        logger.debug(
            "[ROUTER] Failed to emit metrics for Disneyland shortcut",
            exc_info=True,
        )

    logger.info("[ROUTER] Using Disneyland anchor shortcut for cached metadata")

    return result


def _should_skip_maps(
    query: PlaceQuery,
    anchor: Anchor,
    meta: Dict[str, Any],
    store: ConversationSnapshotStore,
    user_id: str,
    conversation_id: str,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[ResolveResult]:
    """Decide whether Gemini/Maps resolution can be skipped based on anchor data."""

    if getattr(query, "is_travel", False):
        return None

    normalized_target = _normalize_location_token(getattr(query, "target", None))
    if not normalized_target:
        return None

    if snapshot is None:
        snapshot = _get_conversation_snapshot(store, user_id, conversation_id)

    tokens = _collect_location_tokens(anchor, meta, snapshot)
    match = tokens.get(normalized_target)
    if not match:
        return None

    current_location = _extract_current_location_payload(meta)
    location_name = (
        current_location.get("name")
        or current_location.get("label")
        or current_location.get("display_name")
        or match["value"]
        or getattr(query, "target", "")
    )

    location_id = (
        current_location.get("id")
        or current_location.get("location_id")
        or current_location.get("slug")
        or meta.get("location_id")
        or snapshot.get("scene_id")
        or snapshot.get("location_id")
    )

    location_type = (
        current_location.get("location_type")
        or current_location.get("level")
        or meta.get("location_type")
        or snapshot.get("location_type")
    )
    level_label = str(location_type).strip().lower() if location_type else "unknown"
    if level_label not in _ALLOWED_PLACE_LEVELS:
        level_label = "unknown"

    lat = None
    lon = None
    for source in (
        current_location,
        meta.get("locationInfo", {}).get("geo")
        if isinstance(meta.get("locationInfo"), dict)
        else {},
        snapshot,
    ):
        if not isinstance(source, dict):
            continue
        if lat is None and source.get("lat") is not None:
            try:
                lat = float(source.get("lat"))
            except (TypeError, ValueError):
                pass
        if lon is None and source.get("lon") is not None:
            try:
                lon = float(source.get("lon"))
            except (TypeError, ValueError):
                pass

    if lat is None and anchor.lat is not None:
        lat = anchor.lat
    if lon is None and anchor.lon is not None:
        lon = anchor.lon

    address: Dict[str, Any] = {}
    for key, attr in (("city", "primary_city"), ("region", "region"), ("country", "country")):
        value = getattr(anchor, attr, None)
        if value:
            address[key] = value
    for key in ("city", "region", "country"):
        value = current_location.get(key)
        if isinstance(value, str) and value:
            address.setdefault(key, value)

    place_meta: Dict[str, Any] = {
        "source": "anchor_snapshot",
        "router_skip": True,
        "match_reason": match["reason"],
    }
    if location_type:
        place_meta["location_type"] = location_type
    if location_id:
        place_meta["location_id"] = location_id
    if snapshot.get("scene_id"):
        place_meta["scene_id"] = snapshot.get("scene_id")
    if snapshot.get("location_name"):
        place_meta["snapshot_location"] = snapshot.get("location_name")

    candidate = Candidate(
        place=Place(
            name=str(location_name),
            level=level_label,
            key=str(location_id) if location_id else _normalize_location_token(location_name),
            lat=lat,
            lon=lon,
            address=address,
            meta=place_meta,
        ),
        confidence=0.98,
    )

    operations = []
    if lat is not None and lon is not None:
        operations.append(
            {
                "op": "poi.navigate",
                "label": candidate.place.name,
                "lat": lat,
                "lon": lon,
                "context_hint": {"use_geo_anchor": True, "source": "anchor_snapshot"},
            }
        )

    result = ResolveResult(
        status=STATUS_EXACT,
        candidates=[candidate],
        operations=operations,
        anchor=anchor,
        scope=anchor.scope,
        message=f"Staying at {candidate.place.name}.",
    )

    result.metadata = {
        "router": {
            "skipped_maps": True,
            "skip_reason": match["reason"],
            "matched_value": match["value"],
        }
    }

    try:
        metrics().LOCATION_ROUTER_DECISIONS.labels(outcome="anchor_match_skip").inc()
    except Exception:
        logger.debug("[ROUTER] Failed to emit metrics for anchor skip", exc_info=True)

    logger.info(
        "[ROUTER] Skipping Gemini/Maps; query '%s' matched current anchor via %s",
        query.target,
        match["reason"],
    )

    return result


def _has_grounded_results(result: ResolveResult) -> bool:
    """Check if any candidates in the result are grounded via Google Maps."""
    if not result or not result.candidates:
        return False

    return any(
        c.place.meta.get("grounded") or c.place.meta.get("google_verified")
        for c in result.candidates
    )


def _extract_external_place_id(result: ResolveResult) -> str:
    """Attempt to pull an external place identifier from the result payload."""

    # Direct field on the result itself
    if getattr(result, "external_place_id", None):
        return str(result.external_place_id)

    # Result-level location object
    location = getattr(result, "location", None)
    if location and getattr(location, "external_place_id", None):
        return str(location.external_place_id)

    # Walk candidates: prefer a dedicated attribute on the Place
    for candidate in getattr(result, "candidates", []) or []:
        place = getattr(candidate, "place", None)
        if place is None:
            continue

        # Hydrated from DB or Maps → often stored directly on the place
        attr_id = getattr(place, "external_place_id", None)
        if attr_id:
            return str(attr_id)

        meta = place.meta if isinstance(getattr(place, "meta", None), dict) else {}
        if meta.get("external_place_id"):
            return str(meta["external_place_id"])

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("external_place_id"):
        return str(metadata["external_place_id"])

    return ""


def _should_skip_fictional_for_real_place(
    real_result: Optional[ResolveResult],
    query: PlaceQuery,
    anchor: Anchor,
    meta: Optional[dict] = None,
) -> bool:
    """
    Decide whether to skip the fictional resolver when a strong real-world
    resolution already exists.

    Intention:
    - If we already have a strongly grounded real-world place (e.g. Disneyland
      Park), do NOT call resolve_fictional in the main path, unless explicitly
      overridden via metadata.
    """

    if real_result is None:
        return False

    # Only care about genuinely real results.
    if getattr(real_result, "scope", None) != "real":
        return False

    meta = meta or {}
    if meta.get("force_fictional_overlay"):
        # Explicit override: caller really wants fictional overlay.
        return False

    external_id = _extract_external_place_id(real_result)
    if isinstance(external_id, str) and external_id.startswith("real::"):
        # Canonical real-world place id, e.g. from your Places hierarchy.
        return True

    # Any candidate explicitly marked as grounded / Google-verified is “real enough”.
    if _has_grounded_results(real_result):
        return True

    # Some adapters attach Maps grounding at the result level.
    meta_dict = getattr(real_result, "metadata", None) or {}
    if isinstance(meta_dict, dict):
        if meta_dict.get("gmaps") or meta_dict.get("has_maps_grounding"):
            return True

    return False


def _normalize_city_name(value: Optional[str]) -> Optional[str]:
    """Normalize city strings for comparison."""
    if not value:
        return None

    cleaned = str(value).strip()
    if not cleaned:
        return None

    primary = cleaned.split(",")[0].strip()
    normalized = unicodedata.normalize("NFKC", primary)
    normalized = re.sub(r"[^0-9a-zA-Z]+", "", normalized.casefold())
    return normalized or None


def _haversine_distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Compute great-circle distance between two coordinates in kilometers."""
    radius_km = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lon = math.radians(b_lon - a_lon)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(a_lat))
        * math.cos(math.radians(b_lat))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def _extract_candidate_city(candidate: Any) -> Optional[str]:
    """Attempt to derive the candidate's city from address or metadata."""
    place = getattr(candidate, "place", None)
    if not place:
        return None

    address = place.address if isinstance(place.address, dict) else {}
    for key in ("city", "town", "municipality", "locality", "village"):
        candidate_city = address.get(key)
        if candidate_city:
            return str(candidate_city)

    normalized_path = address.get("_normalized_admin_path") if isinstance(address, dict) else None
    if isinstance(normalized_path, dict):
        for key in ("city", "town"):
            candidate_city = normalized_path.get(key)
            if candidate_city:
                return str(candidate_city)

    meta = place.meta if isinstance(place.meta, dict) else {}
    for key in ("city", "primary_city", "municipality", "locality"):
        candidate_city = meta.get(key)
        if candidate_city:
            return str(candidate_city)

    return None


def _check_city_change(result: ResolveResult, anchor: Anchor) -> Optional[Dict[str, Any]]:
    """
    Inspect the top candidate and determine if it belongs to a different city
    than the player's current anchor city. Returns city context information
    including travel distance when available.
    """

    if not result or not result.candidates:
        return None

    top_candidate = result.candidates[0]
    candidate_city = _extract_candidate_city(top_candidate)
    if not candidate_city:
        return None

    anchor_city = (anchor.primary_city or "").strip() if anchor else ""
    normalized_anchor = _normalize_city_name(anchor_city)
    normalized_candidate = _normalize_city_name(candidate_city)

    changed = False
    if normalized_anchor and normalized_candidate:
        changed = normalized_anchor != normalized_candidate

    distance = None
    if (
        changed
        and anchor
        and anchor.lat is not None
        and anchor.lon is not None
        and top_candidate.place.lat is not None
        and top_candidate.place.lon is not None
    ):
        distance = _haversine_distance_km(
            float(anchor.lat),
            float(anchor.lon),
            float(top_candidate.place.lat),
            float(top_candidate.place.lon),
        )

    if changed:
        if distance is not None:
            logger.info(
                "[ROUTER] Destination city '%s' differs from anchor city '%s' (~%.1f km)",
                candidate_city,
                anchor_city or "unknown",
                distance,
            )
        else:
            logger.info(
                "[ROUTER] Destination city '%s' differs from anchor city '%s'",
                candidate_city,
                anchor_city or "unknown",
            )

    return {
        "candidate_city": candidate_city,
        "anchor_city": anchor_city or None,
        "changed": changed,
        "distance_km": distance,
        "candidate": top_candidate,
    }


async def resolve_place_or_travel(
    user_text: str,
    meta: Dict[str, Any],
    store: Optional[ConversationSnapshotStore],
    user_id: str,
    conversation_id: str
) -> ResolveResult:
    """
    Main entry point for location resolution. Determines scope (real/fictional)
    and routes the query appropriately.
    
    Resolution strategy for real-world scopes:
    1. Try Gemini with Google Maps grounding (best for real places)
    2. If Gemini returns grounded results -> use them
    3. If Gemini finds nothing or results aren't grounded -> try Overpass/Nominatim
    4. If still nothing -> fall back to fictional resolver
    
    Resolution strategy for fictional scopes:
    1. Try fictional resolver first
    2. If it fails -> fall back to real-world search (mixed worlds)
    """
    if store is None:
        store = ConversationSnapshotStore()

    # Enrich metadata based on user text patterns BEFORE building anchor
    _enrich_metadata_with_intent(user_text, meta)

    anchor = await _anchor_from_meta(meta, user_id, conversation_id)
    q = _parse_place_query(user_text)
    snapshot = _get_conversation_snapshot(store, user_id, conversation_id)

    logger.info(
        f"[ROUTER] Resolving '{q.target}' with scope={anchor.scope}, "
        f"is_travel={q.is_travel}, anchor_city={anchor.primary_city}"
    )

    if anchor.scope == "real":
        # Pre-persist the anchor so we don't fall back to fictional on a known real venue
        await _ensure_real_anchor_location(anchor, meta, user_id, conversation_id)

    res: Optional[ResolveResult] = None
    used_cache = False
    cache_key = f"{user_id}:{conversation_id}:{q.normalized}"
    normalized_anchor_city = _normalize_city_name(anchor.primary_city)

    normalized_target = _normalize_location_token(q.target)
    if (
        _is_basic_move_intent(user_text, q)
        and normalized_target in _DISNEYLAND_NORMALIZED_TOKENS
        and _anchor_targets_disneyland(anchor, meta, snapshot)
    ):
        res = _build_disneyland_shortcut_result(anchor)
        logger.debug(
            "[ROUTER] Disney shortcut triggered (intent move, anchor match, target=%s)",
            q.target,
        )

    if res is None and not q.is_travel:
        cached_payload = await LOCATION_RESOLUTION_CACHE.get(cache_key)
        if cached_payload:
            cached_confidence = float(cached_payload.get("confidence") or 0.0)
            cached_anchor_city = cached_payload.get("anchor_city")
            cached_anchor_normalized = _normalize_city_name(cached_anchor_city)

            if cached_confidence < 0.6:
                logger.debug(
                    "[ROUTER] Cache entry for %s skipped due to low confidence %.2f",
                    cache_key,
                    cached_confidence,
                )
            elif normalized_anchor_city != cached_anchor_normalized:
                logger.debug(
                    "[ROUTER] Cache entry for %s skipped due to anchor city change (%s -> %s)",
                    cache_key,
                    cached_anchor_city,
                    anchor.primary_city,
                )
            else:
                cached_result = cached_payload.get("result")
                if cached_result:
                    res = copy.deepcopy(cached_result)
                    used_cache = True
                    logger.info(
                        "[ROUTER] Using cached resolution for '%s' (age=%.1fs)",
                        q.target,
                        max(
                            0.0,
                            time.time()
                            - float(cached_payload.get("cached_at") or time.time()),
                        ),
                    )

    if res is None and anchor.scope == "real":
        # ============================================================
        # REAL WORLD RESOLUTION CHAIN
        # ============================================================
        logger.info("[ROUTER] Using real-world resolution chain")

        # First, see if we can skip everything via anchor+snapshot tokens
        skip_result = _should_skip_maps(q, anchor, meta, store, user_id, conversation_id)
        if skip_result:
            res = skip_result
        else:
            fastpath_result = await _maybe_resolve_via_gmaps_fastpath(
                q,
                anchor,
                meta,
                user_id,
                conversation_id,
            )
            if fastpath_result is not None:
                res = fastpath_result

            if res is None:
                # Step 1: Try Gemini with Google Maps grounding first
                gemini_result: Optional[ResolveResult] = None
                try:
                    logger.debug(
                        "[ROUTER] Attempting Gemini with Maps grounding (afc_max_calls=%s)...",
                        REAL_WORLD_AFC_MAX_CALLS,
                    )
                    gemini_result = await resolve_location_with_gemini(
                        q,
                        anchor,
                        afc_max_calls=REAL_WORLD_AFC_MAX_CALLS,
                    )

                    if gemini_result.errors:
                        logger.warning(
                            f"[ROUTER] Gemini returned errors: {gemini_result.errors}"
                        )

                    if _has_grounded_results(gemini_result):
                        logger.info(
                            f"[ROUTER] ✓ Gemini found {len(gemini_result.candidates)} Google Maps-grounded results"
                        )
                        res = gemini_result

                    elif gemini_result.candidates and gemini_result.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                        logger.info(
                            "[ROUTER] ⚠ Gemini found results but they're not grounded via Maps. Attempting verification with Overpass/Nominatim..."
                        )

                        try:
                            verified_result = await resolve_real(q, anchor, meta, skip_gemini=True)

                            if verified_result.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                                logger.info("[ROUTER] ✓ Overpass/Nominatim found results")
                                res = verified_result
                            else:
                                logger.info(
                                    f"[ROUTER] Overpass/Nominatim returned status={verified_result.status}"
                                )
                                for candidate in gemini_result.candidates:
                                    candidate.confidence *= 0.7
                                    candidate.place.meta["verification_status"] = "unverified"
                                res = gemini_result

                        except Exception as e:
                            logger.warning(
                                f"[ROUTER] Verification failed: {e}", exc_info=True
                            )
                            res = gemini_result

                    elif gemini_result.status == STATUS_TRAVEL_PLAN and gemini_result.operations:
                        logger.info("[ROUTER] ✓ Gemini created travel plan")
                        res = gemini_result
                    else:
                        logger.info(
                            f"[ROUTER] Gemini returned status={gemini_result.status}, no useful results"
                        )
                        gemini_result = None

                except Exception as e:
                    logger.error(
                        f"[ROUTER] Gemini resolution failed: {e}", exc_info=True
                    )
                    gemini_result = None

            # Step 2: If Gemini didn't work, try Overpass/Nominatim
            if res is None:
                logger.info("[ROUTER] Falling back to Overpass/Nominatim search...")
                try:
                    res = await resolve_real(q, anchor, meta, skip_gemini=True)

                    if res.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                        logger.info(
                            f"[ROUTER] ✓ Overpass/Nominatim found {len(res.candidates)} results"
                        )
                    else:
                        logger.info(
                            f"[ROUTER] Overpass/Nominatim returned status={res.status}"
                        )

                except Exception as e:
                    logger.error(f"[ROUTER] Overpass/Nominatim failed: {e}", exc_info=True)
                    target_label = q.target or q.raw_text or q.normalized or "request"
                    res = ResolveResult(
                        status=STATUS_NOT_FOUND,
                        message=f"Could not find '{target_label}' in the real world.",
                        anchor=anchor,
                        scope="real",
                        errors=[f"real_world_search_failed: {e}"]
                    )

        # Step 3: if real chain fails, enqueue fictional and return ASK quickly
        if res is None:
            target_label = q.target or q.raw_text or q.normalized or "request"
            res = ResolveResult(
                status=STATUS_NOT_FOUND,
                message=f"Could not resolve '{target_label}'.",
                anchor=anchor,
                scope="real",
            )

        if res.status in {STATUS_NOT_FOUND, STATUS_ASK}:
            try:
                place_enrichment.enqueue_fictional_fallback(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query=q,
                    anchor=anchor,
                    meta=meta,
                )
            except Exception:
                logger.warning(
                    "[ROUTER] Failed to enqueue fictional fallback", exc_info=True
                )

            if res.status == STATUS_ASK:
                logger.debug(
                    "[ROUTER] Real resolver requested clarification; preserving ASK payload"
                )
                return res

            return ResolveResult(
                status=STATUS_ASK,
                anchor=anchor,
                scope="real",
                message="Working on it…",
            )

    else:
        # ============================================================
        # FICTIONAL RESOLUTION CHAIN
        # ============================================================
        logger.info("[ROUTER] Using fictional resolution chain")
        real_anchor = Anchor(
            scope="real",
            focus=anchor.focus,
            label=anchor.label,
            lat=anchor.lat,
            lon=anchor.lon,
            primary_city=anchor.primary_city,
            region=anchor.region,
            country=anchor.country,
            world_name=anchor.world_name,
            hints=anchor.hints,
        )

        gemini_result: Optional[ResolveResult] = None
        try:
            gemini_result = await resolve_location_with_gemini(
                q, real_anchor, afc_max_calls=REAL_WORLD_AFC_MAX_CALLS
            )
        except Exception as e:
            logger.error(f"[ROUTER] Gemini resolution failed in fictional chain: {e}", exc_info=True)
            gemini_result = None

        if _should_skip_fictional_for_real_place(gemini_result, q, anchor, meta):
            res = gemini_result
        else:
            # Try fictional resolver first
            res_fictional = await resolve_fictional(q, anchor, meta, store, user_id, conversation_id)

            # If fictional search fails, fall back to real-world search (mixed worlds)
            if res_fictional.status in {STATUS_NOT_FOUND, STATUS_ASK}:
                logger.info(
                    f"[ROUTER] Fictional resolve failed for '{q.target}'. "
                    f"Falling back to real-world search (mixed world scenario)..."
                )

                # Try real-world search with Gemini first (reuse probe if available)
                try:
                    if gemini_result and _has_grounded_results(gemini_result):
                        logger.info(
                            f"[ROUTER] ✓ Found real-world match via Gemini Maps for '{q.target}' "
                            f"in fictional world"
                        )
                        gemini_result.metadata = gemini_result.metadata or {}
                        gemini_result.metadata["mixed_world"] = True
                        gemini_result.metadata["base_scope"] = "fictional"
                        gemini_result.metadata["real_element"] = q.target
                        res = gemini_result
                    else:
                        if gemini_result is None:
                            gemini_result = await resolve_location_with_gemini(
                                q,
                                real_anchor,
                                afc_max_calls=REAL_WORLD_AFC_MAX_CALLS,
                            )

                        if _has_grounded_results(gemini_result):
                            logger.info(
                                f"[ROUTER] ✓ Found real-world match via Gemini Maps for '{q.target}' "
                                f"in fictional world"
                            )
                            gemini_result.metadata = gemini_result.metadata or {}
                            gemini_result.metadata["mixed_world"] = True
                            gemini_result.metadata["base_scope"] = "fictional"
                            gemini_result.metadata["real_element"] = q.target
                            res = gemini_result
                        else:
                            # Try Overpass/Nominatim
                            res_real = gemini_result or await resolve_real(
                                q, real_anchor, meta, skip_gemini=True
                            )

                            if res_real.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                                logger.info(
                                    f"[ROUTER] ✓ Found real-world match via Overpass/Nominatim "
                                    f"for '{q.target}' in fictional world"
                                )
                                res_real.metadata = res_real.metadata or {}
                                res_real.metadata["mixed_world"] = True
                                res_real.metadata["base_scope"] = "fictional"
                                res_real.metadata["real_element"] = q.target
                                res = res_real
                            else:
                                # Both fictional and real searches failed
                                logger.warning(
                                    f"[ROUTER] All resolution methods failed for '{q.target}'"
                                )
                                res = res_fictional

                except Exception as e:
                    logger.error(f"[ROUTER] Real-world fallback failed: {e}", exc_info=True)
                    res = res_fictional
            else:
                # Fictional search succeeded
                res = res_fictional

    # Ensure the final result has the anchor and scope attached
    if res.anchor is None:
        res.anchor = anchor
    if res.scope is None:
        res.scope = anchor.scope

    city_context = _check_city_change(res, anchor)
    candidate_city = city_context["candidate_city"] if city_context else None

    if city_context:
        existing_meta = getattr(res, "metadata", None)
        if isinstance(existing_meta, dict):
            metadata = existing_meta
        else:
            metadata = {}
            res.metadata = metadata
        metadata.setdefault("candidate_city", candidate_city)
        if city_context.get("anchor_city"):
            metadata.setdefault("origin_city", city_context["anchor_city"])
    else:
        metadata = getattr(res, "metadata", None)

    if city_context and city_context["changed"]:
        metadata = metadata if isinstance(metadata, dict) else {}
        res.metadata = metadata
        metadata["requires_travel"] = True
        metadata["destination_city"] = city_context["candidate_city"]
        metadata["origin_city"] = city_context.get("anchor_city")
        distance_km = city_context.get("distance_km")
        if distance_km is not None:
            metadata["travel_distance_km"] = distance_km

        anchor_city = city_context.get("anchor_city")
        distance_note = (
            f" (~{distance_km:.1f} km away)" if distance_km is not None else ""
        )
        travel_msg = (
            f"This destination is in {city_context['candidate_city']}, outside your current city {anchor_city}{distance_note}. "
            "You'll need to travel to reach it."
        )
        metadata["travel_prompt"] = travel_msg
        prompts = metadata.setdefault("prompts", [])
        if travel_msg not in prompts:
            prompts.append(travel_msg)

        if res.message:
            res.message = f"{res.message} {travel_msg}".strip()
        else:
            res.message = travel_msg

        travel_operation = {
            "op": "travel.prompt",
            "from_city": anchor_city,
            "to_city": city_context["candidate_city"],
            "distance_km": distance_km,
        }
        if not any(
            op.get("op") == "travel.prompt"
            and op.get("from_city") == travel_operation["from_city"]
            and op.get("to_city") == travel_operation["to_city"]
            for op in res.operations
        ):
            res.operations.append(travel_operation)

    if not hasattr(res, "metadata"):
        res.metadata = metadata if isinstance(metadata, dict) else {}

    # ✨ NEW: Track player movement if we have an exact match
    if res.status == STATUS_EXACT and res.candidates:
        top_candidate = res.candidates[0]
        location_name = top_candidate.place.name
        location_type = top_candidate.place.meta.get("location_type") or top_candidate.place.level

        # Only persist canonical locations for real-world scopes.
        scope_value = res.scope or anchor.scope
        if scope_value == "real":
            try:
                uid = int(user_id)
                cid = int(conversation_id)
            except (TypeError, ValueError):
                logger.warning(
                    "[ROUTER] Unable to persist location for non-integer identifiers: user_id=%r conversation_id=%r",
                    user_id,
                    conversation_id,
                )
            else:
                try:
                    async with get_db_connection_context() as conn:
                        location_record = await get_or_create_location(
                            conn,
                            user_id=uid,
                            conversation_id=cid,
                            candidate=top_candidate,
                            scope=scope_value,
                            anchor=anchor,
                        )

                    # Mirror the persisted identifier on the candidate metadata.
                    top_candidate.place.meta = dict(top_candidate.place.meta or {})
                    top_candidate.place.meta["location_id"] = location_record.id

                    if location_record.id is not None:
                        if isinstance(res.metadata, dict):
                            res.metadata.setdefault("location_id", location_record.id)
                        elif res.metadata is None:
                            res.metadata = {"location_id": location_record.id}

                    # Seed the snapshot store so downstream turns stay in sync.
                    if store is not None and location_record.id is not None:
                        try:
                            snapshot = store.get(user_id, conversation_id)
                            if snapshot.get("location_id") != location_record.id:
                                snapshot = dict(snapshot)
                                snapshot["location_id"] = location_record.id
                                snapshot.setdefault("location_name", location_record.location_name)
                                store.put(user_id, conversation_id, snapshot)
                        except Exception:
                            logger.warning(
                                "[ROUTER] Failed to seed conversation snapshot with location",
                                exc_info=True,
                            )

                except Exception:
                    logger.warning(
                        "[ROUTER] Failed to persist real-world location for conversation",
                        exc_info=True,
                    )

        # Track movement asynchronously (don't block on failure)
        asyncio.create_task(
            _track_player_movement(
                user_id,
                conversation_id,
                location_name,
                location_type,
                city=candidate_city,
            )
        )

    # ✨ NEW: Also track if we have a location object directly
    if res.location and res.status == STATUS_EXACT:
        location_city = (
            res.location.city.strip()
            if isinstance(res.location.city, str) and res.location.city.strip()
            else candidate_city
        )
        asyncio.create_task(
            _track_player_movement(
                user_id,
                conversation_id,
                res.location.location_name,
                res.location.location_type,
                city=location_city,
            )
        )

    if (
        not used_cache
        and not q.is_travel
        and cache_key
        and res.status in {STATUS_EXACT, STATUS_MULTIPLE}
        and res.candidates
    ):
        top_confidence = float(res.candidates[0].confidence or 0.0)
        if top_confidence >= 0.6:
            metadata_dict = res.metadata if isinstance(res.metadata, dict) else {}
            cache_payload = {
                "result": copy.deepcopy(res),
                "anchor_city": anchor.primary_city,
                "confidence": top_confidence,
                "cached_at": time.time(),
                "requires_travel": bool(metadata_dict.get("requires_travel")),
            }
            await LOCATION_RESOLUTION_CACHE.set(cache_key, cache_payload)
            logger.debug(
                "[ROUTER] Cached resolution for %s with confidence %.2f",
                cache_key,
                top_confidence,
            )

    logger.info(
        f"[ROUTER] Final result: status={res.status}, scope={res.scope}, "
        f"candidates={len(res.candidates)}, mixed_world={res.metadata.get('mixed_world') if res.metadata else False}"
    )

    enrichment_scope = res.scope or anchor.scope
    if enrichment_scope == "real":
        try:
            place_enrichment.enqueue(
                user_id=user_id,
                conversation_id=conversation_id,
                query=q,
                anchor=anchor,
                result=res,
                afc_max_calls=REAL_WORLD_AFC_MAX_CALLS,
            )
        except Exception:
            logger.warning(
                "[ROUTER] Failed to enqueue place enrichment task", exc_info=True
            )

    return res
