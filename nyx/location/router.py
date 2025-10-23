# nyx/location/router.py

from __future__ import annotations

import asyncio
import logging
import math
import re
import unicodedata
from typing import Any, Dict, Optional

from .anchors import derive_geo_anchor
from .fictional_resolver import resolve_fictional
from .gemini_maps_adapter import resolve_location_with_gemini
from .query import PlaceQuery
from .search import resolve_real
from .types import (
    Anchor,
    Place,
    ResolveResult,
    Scope,
    STATUS_EXACT,
    STATUS_MULTIPLE,
    STATUS_TRAVEL_PLAN,
    STATUS_ASK,
    STATUS_NOT_FOUND,
)
from nyx.conversation.snapshot_store import ConversationSnapshotStore

logger = logging.getLogger(__name__)

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
        return PlaceQuery(raw_text=t, normalized=target.lower(), is_travel=True, target=target, transport_hint="fly")
    
    m = _GO_TO_RX.search(t)
    target = (m.group(1) if m else t).strip().rstrip(".!?")
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


def _has_grounded_results(result: ResolveResult) -> bool:
    """Check if any candidates in the result are grounded via Google Maps."""
    if not result or not result.candidates:
        return False

    return any(
        c.place.meta.get("grounded") or c.place.meta.get("google_verified")
        for c in result.candidates
    )


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
    
    logger.info(
        f"[ROUTER] Resolving '{q.target}' with scope={anchor.scope}, "
        f"is_travel={q.is_travel}, anchor_city={anchor.primary_city}"
    )

    res: ResolveResult
    
    if anchor.scope == "real":
        # ============================================================
        # REAL WORLD RESOLUTION CHAIN
        # ============================================================
        logger.info("[ROUTER] Using real-world resolution chain")
        
        # Step 1: Try Gemini with Google Maps grounding first
        gemini_result: Optional[ResolveResult] = None
        try:
            logger.debug("[ROUTER] Attempting Gemini with Maps grounding...")
            gemini_result = await resolve_location_with_gemini(q, anchor)
            
            if gemini_result.errors:
                logger.warning(f"[ROUTER] Gemini returned errors: {gemini_result.errors}")
            
            # Check if we got grounded (verified via Google Maps) results
            has_grounded = _has_grounded_results(gemini_result)
            
            if has_grounded:
                logger.info(
                    f"[ROUTER] ✓ Gemini found {len(gemini_result.candidates)} "
                    f"Google Maps-grounded results"
                )
                res = gemini_result
                
            elif gemini_result.candidates and gemini_result.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                # Gemini found something but it's not grounded
                # Could be fictional place or Gemini making educated guess
                logger.info(
                    f"[ROUTER] ⚠ Gemini found results but they're not grounded via Maps. "
                    f"Attempting verification with Overpass/Nominatim..."
                )
                
                # Try to verify with traditional geocoding
                try:
                    verified_result = await resolve_real(q, anchor, meta)
                    
                    if verified_result.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                        logger.info("[ROUTER] ✓ Verified via Overpass/Nominatim")
                        res = verified_result
                    else:
                        # Overpass/Nominatim couldn't verify either
                        # Use Gemini's best guess but mark confidence as lower
                        logger.info("[ROUTER] Using Gemini's unverified results")
                        for candidate in gemini_result.candidates:
                            candidate.confidence *= 0.7  # Reduce confidence
                            candidate.place.meta["verification_status"] = "unverified"
                        res = gemini_result
                        
                except Exception as e:
                    logger.warning(f"[ROUTER] Verification failed: {e}", exc_info=True)
                    res = gemini_result
                    
            elif gemini_result.status == STATUS_TRAVEL_PLAN and gemini_result.operations:
                # Gemini created a travel plan
                logger.info("[ROUTER] ✓ Gemini created travel plan")
                res = gemini_result
                
            else:
                # Gemini found nothing useful
                logger.info(
                    f"[ROUTER] Gemini returned status={gemini_result.status}, "
                    f"no useful results"
                )
                gemini_result = None
                
        except Exception as e:
            logger.error(f"[ROUTER] Gemini resolution failed: {e}", exc_info=True)
            gemini_result = None
        
        # Step 2: If Gemini didn't work, try Overpass/Nominatim
        if gemini_result is None:
            logger.info("[ROUTER] Falling back to Overpass/Nominatim search...")
            try:
                res = await resolve_real(q, anchor, meta)
                
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
                res = ResolveResult(
                    status=STATUS_NOT_FOUND,
                    message=f"Could not find '{q.target}' in the real world.",
                    anchor=anchor,
                    scope="real",
                    errors=[f"real_world_search_failed: {e}"]
                )
        else:
            res = gemini_result
        
        # Step 3: If everything failed, try fictional as last resort
        if res.status in {STATUS_NOT_FOUND, STATUS_ASK} and q.target:
            logger.info(
                f"[ROUTER] Real-world search failed for '{q.target}'. "
                f"Attempting fictional resolver as fallback (mixed world scenario)..."
            )
            
            try:
                # Temporarily mark as fictional for the resolver
                fictional_anchor = Anchor(
                    scope="fictional",
                    focus=anchor.focus,
                    label=anchor.label,
                    lat=anchor.lat,
                    lon=anchor.lon,
                    primary_city=anchor.primary_city,
                    region=anchor.region,
                    country=anchor.country,
                    world_name=anchor.world_name or anchor.primary_city,
                    hints=anchor.hints,
                )
                
                fictional_result = await resolve_fictional(
                    q, fictional_anchor, meta, store, user_id, conversation_id
                )
                
                if fictional_result.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                    logger.info(
                        f"[ROUTER] ✓ Fictional resolver created '{q.target}' "
                        f"in mixed real/fictional world"
                    )
                    # Mark as mixed world
                    fictional_result.metadata = fictional_result.metadata or {}
                    fictional_result.metadata["mixed_world"] = True
                    fictional_result.metadata["base_scope"] = "real"
                    fictional_result.metadata["fictional_element"] = q.target
                    res = fictional_result
                else:
                    logger.info(
                        f"[ROUTER] Fictional resolver also failed: {fictional_result.status}"
                    )
                    
            except Exception as e:
                logger.error(f"[ROUTER] Fictional fallback failed: {e}", exc_info=True)
    
    else:
        # ============================================================
        # FICTIONAL RESOLUTION CHAIN
        # ============================================================
        logger.info("[ROUTER] Using fictional resolution chain")
        
        # Try fictional resolver first
        res_fictional = await resolve_fictional(q, anchor, meta, store, user_id, conversation_id)

        # If fictional search fails, fall back to real-world search (mixed worlds)
        if res_fictional.status in {STATUS_NOT_FOUND, STATUS_ASK}:
            logger.info(
                f"[ROUTER] Fictional resolve failed for '{q.target}'. "
                f"Falling back to real-world search (mixed world scenario)..."
            )
            
            # Try real-world search with Gemini first
            try:
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
                
                gemini_result = await resolve_location_with_gemini(q, real_anchor)
                
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
                    res_real = await resolve_real(q, real_anchor, meta)
                    
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

        res.operations.append({
            "op": "travel.prompt",
            "from_city": anchor_city,
            "to_city": city_context["candidate_city"],
            "distance_km": distance_km,
        })

    if not hasattr(res, "metadata"):
        res.metadata = metadata if isinstance(metadata, dict) else {}

    # ✨ NEW: Track player movement if we have an exact match
    if res.status == STATUS_EXACT and res.candidates:
        top_candidate = res.candidates[0]
        location_name = top_candidate.place.name
        location_type = top_candidate.place.meta.get("location_type") or top_candidate.place.level

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
    
    logger.info(
        f"[ROUTER] Final result: status={res.status}, scope={res.scope}, "
        f"candidates={len(res.candidates)}, mixed_world={res.metadata.get('mixed_world') if res.metadata else False}"
    )
        
    return res
