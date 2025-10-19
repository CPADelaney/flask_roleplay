# nyx/location/router.py
from __future__ import annotations
import re
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
)
from nyx.conversation.snapshot_store import ConversationSnapshotStore

_GO_TO_RX = re.compile(r"\b(?:go|head|walk|run|drive|get|straight|toward|to)\s+(?:the\s+)?(.+)$", re.IGNORECASE)
_FLY_TO_RX = re.compile(r"\b(?:fly|flight)\s+(?:to|for)\s+(.+)$", re.IGNORECASE)

def _parse_place_query(text: str) -> PlaceQuery:
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

async def _anchor_from_meta(meta: Dict[str, Any], user_id: str, conversation_id: str) -> Anchor:
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
    if store is None:
        store = ConversationSnapshotStore()
    anchor = await _anchor_from_meta(meta, user_id, conversation_id)
    q = _parse_place_query(user_text)
    if anchor.scope == "real":
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
        res = await resolve_fictional(q, anchor, meta, store, user_id, conversation_id)

    if res.anchor is None:
        res.anchor = anchor
    if res.scope is None:
        res.scope = anchor.scope
    return res
