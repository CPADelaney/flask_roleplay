# nyx/location/fictional_resolver.py
from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple
from nyx.conversation.snapshot_store import ConversationSnapshotStore

from .types import *
from .anchors import derive_geo_anchor

# Minimal archetypes. Extend as you like.
_DISTRICT_ARCHETYPES = [
    {"key": "old_port", "label": "Old Port", "vibe": "waterfront, brick warehouses, sweets/markets"},
    {"key": "hi_tech", "label": "Glassline", "vibe": "modern towers, neon lanes, cafes"},
    {"key": "sunset_equiv", "label": "Western Dunes", "vibe": "foggy residential blocks, parks, diners"},
    {"key": "arts", "label": "Canvas Row", "vibe": "galleries, night markets, street food"},
]
_VENUE_ARCHETYPES = [
    {"slot": "landmark_sweets", "names": ["Bellweather Confections", "Gilded Chocolatier", "Harbor Confection Hall"], "category": "landmark"},
    {"slot": "dim_sum", "names": ["Dragon Steam House", "Pearl Bamboo Court", "Red Lantern Hall"], "category": "restaurant"},
    {"slot": "festival", "names": ["Blossom Night Market", "Lantern Petal Fair", "Moonflow Parade"], "category": "festival"},
]

def _ensure_city_graph(store: ConversationSnapshotStore, user_key: str, conv_key: str, world_name: str) -> Dict[str, Any]:
    snap = store.get(user_key, conv_key)
    graph = snap.get("city_graph") or {}
    if graph:
        return graph
    # Seed a small, coherent city graph
    random.seed(f"{user_key}:{conv_key}:{world_name}")
    districts = []
    for a in _DISTRICT_ARCHETYPES:
        # Fake coordinates around a rough center if none provided in meta
        lat = (snap.get("lat") or 37.77) + random.uniform(-0.03, 0.03)
        lon = (snap.get("lon") or -122.42) + random.uniform(-0.03, 0.03)
        districts.append({"key": a["key"], "label": a["label"], "vibe": a["vibe"], "lat": lat, "lon": lon, "venues": []})
    graph = {"districts": districts, "venues": []}
    snap["city_graph"] = graph
    store.put(user_key, conv_key, snap)
    return graph

def _find_or_create_venue(graph: Dict[str, Any], slot: str, near_district_key: str) -> Dict[str, Any]:
    # Check if created already
    for v in graph.get("venues", []):
        if v.get("slot") == slot:
            return v
    # Else create new venue in the right district
    dist = next((d for d in graph["districts"] if d["key"] == near_district_key), graph["districts"][0])
    names = next((x["names"] for x in _VENUE_ARCHETYPES if x["slot"] == slot), ["The Place"])
    category = next((x["category"] for x in _VENUE_ARCHETYPES if x["slot"] == slot), "place")
    name = random.choice(names)
    venue = {"slot": slot, "name": name, "lat": dist["lat"] + random.uniform(-0.002, 0.002),
             "lon": dist["lon"] + random.uniform(-0.002, 0.002), "category": category, "district": dist["label"]}
    graph["venues"].append(venue)
    dist["venues"].append(venue)
    return venue

async def resolve_fictional(query: PlaceQuery, setting: SettingProfile, meta: Dict[str, Any],
                            store: ConversationSnapshotStore, user_id: str, conversation_id: str) -> ResolutionResult:
    world_name = setting.world_name or "Velvet City"
    graph = _ensure_city_graph(store, str(user_id), str(conversation_id), world_name)

    # Simple mapping from famous real‑world requests → fictional analogues
    target = (query.target or "").strip().lower()
    if not target:
        return ResolutionResult(status=ResolutionStatus.AMBIGUOUS, message="What kind of place are you seeking?")

    if any(k in target for k in ["ghirardelli", "chocolate", "square", "sweets"]):
        v = _find_or_create_venue(graph, "landmark_sweets", near_district_key="old_port")
        return ResolutionResult(
            status=ResolutionStatus.EXACT,
            candidates=[PlaceCandidate(name=v["name"], lat=v["lat"], lon=v["lon"], address={"district": v["district"]}, category=v["category"], confidence=0.9)],
            canonical_ops=[{"op": "poi.navigate", "label": v["name"], "lat": v["lat"], "lon": v["lon"], "category": v["category"], "lore": {"district": v["district"]}}],
            message=f"Heading to {v['name']} in {v['district']}.",
        )

    if any(k in target for k in ["yank sing", "dim sum", "dumpling"]):
        v = _find_or_create_venue(graph, "dim_sum", near_district_key="arts")
        return ResolutionResult(
            status=ResolutionStatus.EXACT,
            candidates=[PlaceCandidate(name=v["name"], lat=v["lat"], lon=v["lon"], address={"district": v["district"]}, category=v["category"], confidence=0.9)],
            canonical_ops=[{"op": "poi.navigate", "label": v["name"], "lat": v["lat"], "lon": v["lon"], "category": v["category"], "lore": {"district": v["district"]}}],
            message=f"Steam curls from {v['name']} in {v['district']}—let’s go.",
        )

    if any(k in target for k in ["sunset", "west", "residential", "beach", "dunes"]):
        d = next((d for d in graph["districts"] if d["key"] == "sunset_equiv"), graph["districts"][0])
        return ResolutionResult(
            status=ResolutionStatus.EXACT,
            candidates=[PlaceCandidate(name=d["label"], lat=d["lat"], lon=d["lon"], address={}, category="district", confidence=0.85)],
            canonical_ops=[{"op": "poi.navigate", "label": d["label"], "lat": d["lat"], "lon": d["lon"], "category": "district"}],
            message=f"Angling toward {d['label']}—fog and grid streets ahead.",
        )

    if any(k in target for k in ["cherry blossom", "festival", "fair", "parade"]):
        v = _find_or_create_venue(graph, "festival", near_district_key="arts")
        return ResolutionResult(
            status=ResolutionStatus.EXACT,
            candidates=[PlaceCandidate(name=v["name"], lat=v["lat"], lon=v["lon"], address={"district": "Canvas Row"}, category="festival", confidence=0.8)],
            canonical_ops=[{"op": "event.attend", "label": v["name"], "lat": v["lat"], "lon": v["lon"], "category": v["category"]}],
            message=f"Festival lights gather at {v['name']}—we can head there now.",
        )

    # Fallback: soft ask with archetypal suggestions
    suggestions = [
        "night market in Canvas Row",
        "waterfront sweets hall in Old Port",
        "residential dunes on the west side",
    ]
    return ResolutionResult(
        status=ResolutionStatus.ASK,
        message=f"In {world_name}, what kind of place do you want—food, landmark, district, or festival?",
        choices=suggestions,
    )
