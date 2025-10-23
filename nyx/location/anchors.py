from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.location.types import Location

logger = logging.getLogger(__name__)

@dataclass
class GeoAnchor:
    lat: Optional[float] = None
    lon: Optional[float] = None
    neighborhood: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    label: Optional[str] = None  # human-friendly

_store = ConversationSnapshotStore()

def _deg_box(lat: float, lon: float, km: float) -> Tuple[float, float, float, float]:
    dlat = km / 111.0
    dlon = km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    return (lon - dlon, lat + dlat, lon + dlon, lat - dlat)

async def _geocode_city_once(city: str, region: Optional[str], country: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    q = ", ".join([p for p in [city, region, country] if p])
    if not q.strip():
        return None, None
    
    # Nominatim requires a descriptive User-Agent.
    headers = {"User-Agent": "nyx/worldsense/1.0 (contact: your-admin-email@example.com)"}
    
    # --- MODIFICATION START ---
    # Wrap the network call to gracefully handle connection errors, which are
    # common on PaaS like Render due to shared IPs being blocked by Nominatim.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "jsonv2", "limit": "1", "addressdetails": "1"},
                headers=headers,
            )
            r.raise_for_status()
            data = r.json() or []
            if isinstance(data, list) and data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except httpx.ConnectError:
        logger.warning(
            f"Could not connect to Nominatim to geocode '{q}'. This is a common issue on shared IP platforms like Render. "
            "Proceeding without geocoded anchor, which will allow fallback to Gemini."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while geocoding '{q}': {e}", exc_info=True)
    # --- MODIFICATION END ---
        
    return None, None

async def _fetch_current_location(user_id: str, conversation_id: str) -> Optional[Location]:
    """
    Fetch the player's current location by:
    1. Querying currentroleplay table for CurrentLocation key
    2. Looking up that location_name in Locations table
    """
    try:
        from db.connection import get_db_connection_context
        
        user_key = int(user_id)
        conversation_key = int(conversation_id)
        
        async with get_db_connection_context() as conn:
            # First, get the current location name from currentroleplay
            current_loc_row = await conn.fetchrow(
                """
                SELECT value
                FROM currentroleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                """,
                user_key,
                conversation_key,
            )
            
            if not current_loc_row or not current_loc_row["value"]:
                logger.debug(f"No CurrentLocation found in currentroleplay for user={user_id} conv={conversation_id}")
                return None
            
            current_location_name = str(current_loc_row["value"]).strip()
            logger.info(f"[ANCHOR] CurrentLocation from roleplay: '{current_location_name}'")
            
            # Now look up that location in the Locations table
            location_row = await conn.fetchrow(
                """
                SELECT *
                FROM Locations
                WHERE user_id = $1 
                  AND conversation_id = $2 
                  AND LOWER(TRIM(location_name)) = LOWER($3)
                ORDER BY location_id DESC
                LIMIT 1
                """,
                user_key,
                conversation_key,
                current_location_name,
            )
            
            if location_row:
                logger.info(f"[ANCHOR] Found location in DB: {dict(location_row)}")
                return Location.from_record(location_row)
            else:
                logger.warning(
                    f"[ANCHOR] CurrentLocation '{current_location_name}' not found in Locations table. "
                    f"The location may not have been persisted yet."
                )
                return None
                
    except Exception as e:
        logger.debug(f"Failed to fetch current location: {e}", exc_info=True)
    
    return None

async def derive_geo_anchor(meta: Dict[str, Any], user_id: str = "0", conversation_id: str = "0") -> GeoAnchor:
    """
    Build a robust anchor with lat/lon. Never rely on fictional scene names.
    Priority:
      1) incoming meta.locationInfo.geo lat/lon
      2) last persisted anchor from snapshot_store
      3) current location from currentroleplay + Locations table
      4) world.primary_city (geocode once, cache)
    """
    snap = _store.get(str(user_id), str(conversation_id)) or {}
    a = GeoAnchor()

    li = meta.get("locationInfo") or {}
    world = meta.get("world") or {}
    ss = meta.get("scene_scope") or {}  # used only for neighborhood label; never for city/region/country

    # 1) Direct coordinates (locationInfo.geo or scene_scope if it has numeric lat/lon)
    for scope in (li.get("geo") or {}, ss):
        try:
            if a.lat is None and "lat" in scope and "lon" in scope:
                a.lat = float(scope["lat"])
                a.lon = float(scope["lon"])
        except Exception:
            pass

    # 2) Previous snapshot fallback
    if a.lat is None and isinstance(snap, dict):
        try:
            if "lat" in snap and "lon" in snap:
                a.lat = float(snap["lat"])
                a.lon = float(snap["lon"])
        except Exception:
            pass

    # Neighborhood: ok to use scene_scope for flavor
    for src, dst in [("neighborhood","neighborhood"), ("district","neighborhood")]:
        v = (li.get(src) if li.get(src) is not None else ss.get(src))
        if isinstance(v, str) and getattr(a, dst) is None:
            setattr(a, dst, v.strip())

    # City/region/country: **only** trusted sources (locationInfo and world.*)
    for scope in (li, world):
        for src, dst in [
            ("city","city"), ("region","region"), ("state","region"),
            ("country","country")
        ]:
            v = scope.get(src)
            if isinstance(v, str) and getattr(a, dst) is None:
                setattr(a, dst, v.strip())

    # 3) Try fetching current location from currentroleplay + Locations
    if not a.city or not a.region or not a.country or a.lat is None or a.lon is None:
        current_location = await _fetch_current_location(user_id, conversation_id)
        if current_location:
            logger.info(
                f"[ANCHOR] Using current location from DB: {current_location.location_name} "
                f"(city={current_location.city}, region={current_location.region}, "
                f"country={current_location.country}, lat={current_location.lat}, lon={current_location.lon})"
            )
            
            # Fill in missing anchor data from database
            if not a.city and current_location.city:
                a.city = current_location.city
            if not a.region and current_location.region:
                a.region = current_location.region
            if not a.country and current_location.country:
                a.country = current_location.country
            if a.lat is None and current_location.lat is not None:
                a.lat = current_location.lat
            if a.lon is None and current_location.lon is not None:
                a.lon = current_location.lon
            if not a.neighborhood and current_location.district:
                a.neighborhood = current_location.district
            
            # Update snapshot with database values so we don't need to query again
            snap = dict(snap or {})
            if current_location.lat is not None:
                snap["lat"] = current_location.lat
            if current_location.lon is not None:
                snap["lon"] = current_location.lon
            if current_location.city:
                snap["city"] = current_location.city
            if current_location.region:
                snap["region"] = current_location.region
            if current_location.country:
                snap["country"] = current_location.country
            _store.put(str(user_id), str(conversation_id), snap)

    # 4) If still missing lat/lon, geocode world.primary_city (not scene labels)
    if a.lat is None or a.lon is None:
        city = a.city or world.get("primary_city")
        if city:
            lat, lon = await _geocode_city_once(city, a.region, a.country)
            a.lat, a.lon = lat, lon
            if lat is not None and lon is not None:
                snap = dict(snap or {})
                snap["lat"], snap["lon"] = lat, lon
                if a.city:
                    snap["city"] = a.city
                _store.put(str(user_id), str(conversation_id), snap)

    # Human label
    if a.neighborhood and (a.city or world.get("primary_city")):
        a.label = f"{a.neighborhood}, {a.city or world.get('primary_city')}"
    elif a.city:
        a.label = a.city
    elif world.get("primary_city"):
        a.label = world.get("primary_city")

    logger.debug(f"[ANCHOR] derived anchor lat={a.lat} lon={a.lon} city={a.city} region={a.region} country={a.country} label={a.label}")
    return a

def build_nominatim_params_for_poi(poi: str, anchor: GeoAnchor, *, radius_km: float, limit: int) -> Dict[str, str]:
    def _normalize(s: str) -> str:
        return (s or "").replace("'", "'").strip()
    params = {"format": "jsonv2", "limit": str(limit), "addressdetails": "1"}
    brand = _normalize(poi)
    if anchor.lat is not None and anchor.lon is not None:
        w, n, e, s = _deg_box(anchor.lat, anchor.lon, km=radius_km)
        params["q"] = brand
        params["viewbox"] = f"{w:.6f},{n:.6f},{e:.6f},{s:.6f}"
        params["bounded"] = "1"
    elif anchor.city:
        # Safe to use because we no longer let scene labels populate 'city'
        params["q"] = f"{brand}, {anchor.city}"
    else:
        params["q"] = brand
    return params

def nearest_airport_label(anchor: GeoAnchor):
    if anchor.lat and anchor.lon:
        if 37.60 <= anchor.lat <= 37.90 and -122.55 <= anchor.lon <= -122.20:
            return "San Francisco International Airport", 37.6213, -122.3790
        return "Nearest International Airport", anchor.lat + 0.09, anchor.lon + 0.09
    return "Nearest International Airport", 0.0, 0.0


def derive_anchor_from_hierarchy(current_location: Location) -> Optional[str]:
    """Return a human-readable anchor derived from a location hierarchy."""

    if not isinstance(current_location, Location):
        return None

    if getattr(current_location, "is_fictional", False):
        return None

    parts = []
    for attr in ("district", "city", "region", "country"):
        value = getattr(current_location, attr, None)
        if isinstance(value, str):
            value = value.strip()
        if value:
            parts.append(value)

    if parts:
        return ", ".join(parts)

    return None
