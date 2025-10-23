# nyx/location/router.py

from __future__ import annotations
import re
import logging
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
    if any(keyword in text_lower for keyword in ["disneyland", "disney", "theme park", "amusement park", "six flags", "universal studios"]):
        world.update({
            "type": "fictional",
            "themes": ["entertainment", "theme park", "family-friendly", "attractions"],
            "tone": "whimsical",
            "genre": "modern fantasy",
            "technology_level": "modern",
            "primary_city": "Theme Park Resort"
        })
        logger.info("Detected theme park intent, enriching metadata")
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
    kind_txt = (world.get("type") or world.get("kind") or "").strip().lower()
    
    if kind_txt in {"real", "modern_realistic", "realistic", "historical", "modern"}:
        scope: Scope = "real"
    else:
        scope = "fictional"

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

    return Anchor(
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

async def resolve_place_or_travel(
    user_text: str,
    meta: Dict[str, Any],
    store: Optional[ConversationSnapshotStore],
    user_id: str,
    conversation_id: str
) -> ResolveResult:
    """
    Main entry point for location resolution. Determines scope (real/fictional)
    and routes the query, with a fallback from fictional to real-world search on failure.
    """
    if store is None:
        store = ConversationSnapshotStore()
    
    # Enrich metadata based on user text patterns BEFORE building anchor
    _enrich_metadata_with_intent(user_text, meta)
    
    anchor = await _anchor_from_meta(meta, user_id, conversation_id)
    q = _parse_place_query(user_text)

    res: ResolveResult
    if anchor.scope == "real":
        # Handle real-world scopes directly, trying Gemini first.
        gemini_result: Optional[ResolveResult] = None
        try:
            gemini_result = await resolve_location_with_gemini(q, anchor)
        except Exception:
            gemini_result = None

        if gemini_result and (
            (gemini_result.candidates and gemini_result.status in {STATUS_EXACT, STATUS_MULTIPLE})
            or (gemini_result.status == STATUS_TRAVEL_PLAN and gemini_result.operations)
        ):
            res = gemini_result
        else:
            res = await resolve_real(q, anchor, meta)
    else:
        # Fictional scope: try fictional resolver first.
        res_fictional = await resolve_fictional(q, anchor, meta, store, user_id, conversation_id)

        # If the fictional search fails, fall back to a real-world search.
        if res_fictional.status in {STATUS_NOT_FOUND, STATUS_ASK}:
            logger.info(f"Fictional resolve failed for '{q.target}'. Falling back to real-world search.")
            
            # Temporarily change the anchor's scope to perform a real-world search.
            anchor.scope = "real"
            res_real = await resolve_real(q, anchor, meta)

            # If the real-world search found something, use its result.
            if res_real.status in {STATUS_EXACT, STATUS_MULTIPLE}:
                res = res_real
            else:
                # If both fictional and real searches fail, return the original fictional failure.
                res = res_fictional
        else:
            # The fictional search succeeded, so use its result.
            res = res_fictional

    # Ensure the final result has the anchor and scope attached.
    if res.anchor is None:
        res.anchor = anchor
    if res.scope is None:
        res.scope = anchor.scope
        
    return res
