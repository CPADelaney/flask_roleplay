# nyx/location/router.py
from __future__ import annotations
import re
from typing import Any, Dict

from .types import *
from .anchors import derive_geo_anchor
from .real_world_resolver import resolve_real
from .fictional_resolver import resolve_fictional
from nyx.conversation.snapshot_store import ConversationSnapshotStore

_GO_TO_RX = re.compile(r"\b(?:go|head|walk|run|drive|get|straight|toward|to)\s+(?:the\s+)?(.+)$", re.IGNORECASE)
_FLY_TO_RX = re.compile(r"\b(?:fly|flight)\s+(?:to|for)\s+(.+)$", re.IGNORECASE)

def _setting_from_meta(meta: Dict[str, Any]) -> SettingProfile:
    world = (meta or {}).get("world") or {}
    kind_txt = (world.get("type") or world.get("kind") or "").strip().lower()
    kind = SettingKind.REAL if kind_txt in ("real", "modern_realistic", "realistic") else (
           SettingKind.FICTIONAL if kind_txt in ("fiction", "fictional", "urban_fantasy") else SettingKind.HYBRID)
    a = derive_geo_anchor(meta)
    return SettingProfile(
        kind=kind,
        primary_city=a.city or world.get("primary_city"),
        region=a.region or world.get("region"),
        country=a.country or world.get("country"),
        lat=a.lat, lon=a.lon, label=a.label,
        world_name=world.get("name") or world.get("world_name"),
    )

def _parse_place_query(text: str) -> PlaceQuery:
    t = (text or "").strip()
    if not t:
        return PlaceQuery(raw_text="", normalized="")
    m2 = _FLY_TO_RX.search(t)
    if m2:
        target = m2.group(1).strip()
        return PlaceQuery(raw_text=t, normalized=target.lower(), is_travel=True, target=target, transport_hint="fly")
    m = _GO_TO_RX.search(t)
    target = (m.group(1) if m else t).strip()
    # strip simple punctuation
    target = target.rstrip(".!?")
    return PlaceQuery(raw_text=t, normalized=target.lower(), target=target)

async def resolve_place_or_travel(
    user_text: str, meta: Dict[str, Any], store: ConversationSnapshotStore, user_id: str, conversation_id: str
) -> ResolutionResult:
    setting = _setting_from_meta(meta)
    q = _parse_place_query(user_text)

    if setting.kind == SettingKind.REAL or (setting.kind == SettingKind.HYBRID and setting.primary_city):
        res = await resolve_real(q, setting, meta)
    else:
        res = await resolve_fictional(q, setting, meta, store, user_id, conversation_id)

    # add anchor note
    a = derive_geo_anchor(meta)
    res.anchor_used = a.label or a.city or None
    return res
