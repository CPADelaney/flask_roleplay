# nyx/location/fictional_resolver.py
from __future__ import annotations
import json, os, random
from typing import Any, Dict, List

from nyx.conversation.snapshot_store import ConversationSnapshotStore

from .anchors import derive_geo_anchor
from .query import PlaceQuery
from .types import (
    Anchor,
    Candidate,
    Place,
    ResolveResult,
    STATUS_ASK,
    STATUS_AMBIGUOUS,
    STATUS_EXACT,
)

_CONTENT_PATH = os.environ.get("NYX_CITY_CONTENT", "nyx_data/city_archetypes.json")

def _load_pack() -> Dict[str, Any]:
    if os.path.exists(_CONTENT_PATH):
        try:
            with open(_CONTENT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"districts": [], "archetypes": [], "lexicon": {}}

def _synth_name(patterns: List[str], lexicon: Dict[str, List[str]], rng: random.Random) -> str:
    if not patterns:
        return "The " + "".join(rng.choice("BCDFGHJKLMNPQRSTVWXYZ") + rng.choice("aeiou") for _ in range(3))
    pat = rng.choice(patterns)
    def repl(tok: str) -> str:
        key = tok.strip("{}").lower()
        choices = lexicon.get(key) or [key.title()]
        return rng.choice(choices)
    out, cur, buf = [], False, ""
    for ch in pat:
        if ch == "{":
            if cur: buf += ch
            else:
                cur = True; buf = "{"
        elif ch == "}":
            if cur:
                buf += "}"; out.append(repl(buf)); cur = False; buf = ""
            else:
                out.append("}")
        else:
            if cur: buf += ch
            else: out.append(ch)
    if cur: out.append(buf)
    return "".join(out).strip()

def _ensure_city_graph(store: ConversationSnapshotStore, user_key: str, conv_key: str, world_name: str, anchor) -> Dict[str, Any]:
    snap = store.get(user_key, conv_key) or {}
    graph = snap.get("city_graph") or {}
    if graph: return graph
    pack = _load_pack()
    rng = random.Random(f"{user_key}:{conv_key}:{world_name}")
    districts = []
    base_lat = anchor.lat or 37.77
    base_lon = anchor.lon or -122.42
    if pack["districts"]:
        for d in pack["districts"]:
            districts.append({"key": d["key"], "label": d["label"], "vibe": d.get("vibe",""),
                              "lat": base_lat + rng.uniform(-0.03, 0.03),
                              "lon": base_lon + rng.uniform(-0.03, 0.03),
                              "venues": []})
    else:
        for key,label in [("old_port","Old Port"),("hi_tech","Glassline"),("west_res","Western Dunes"),("arts","Canvas Row")]:
            districts.append({"key": key, "label": label, "vibe": "", "lat": base_lat + rng.uniform(-0.03,0.03), "lon": base_lon + rng.uniform(-0.03,0.03), "venues": []})
    graph = {"districts": districts, "venues": [], "pack_loaded": bool(pack["districts"])}
    snap["city_graph"] = graph
    store.put(user_key, conv_key, snap)
    return graph

def _spawn_archetype(graph: Dict[str, Any], slot: str, pack: Dict[str, Any], rng: random.Random, near_key: str) -> Dict[str, Any]:
    for v in graph.get("venues", []):
        if v.get("slot") == slot: return v
    dist = next((d for d in graph["districts"] if d["key"] == near_key), graph["districts"][0])
    arch = next((a for a in pack.get("archetypes", []) if a.get("slot") == slot), None)
    name = _synth_name(arch.get("name_patterns", []) if arch else [], pack.get("lexicon", {}), rng)
    category = (arch or {}).get("category") or "place"
    venue = {"slot": slot, "name": name,
             "lat": dist["lat"] + rng.uniform(-0.002, 0.002),
             "lon": dist["lon"] + rng.uniform(-0.002, 0.002),
             "category": category, "district": dist["label"]}
    graph["venues"].append(venue); dist["venues"].append(venue)
    return venue

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

    world_meta = meta.get("world") or {}
    world_name = anchor.world_name or world_meta.get("name") or world_meta.get("world_name") or "Fictional City"
    graph = _ensure_city_graph(store, str(user_id), str(conversation_id), world_name, anchor)
    pack = _load_pack()
    rng = random.Random(f"{user_id}:{conversation_id}:{world_name}")

    target = (query.target or "").lower().strip()
    if not target:
        return ResolveResult(status=STATUS_AMBIGUOUS, message="What kind of place are you seeking?", anchor=anchor, scope=anchor.scope)

    if any(k in target for k in ["chocolate","sweets","confection","ghirardelli","square"]):
        v = _spawn_archetype(graph, "landmark_sweets", pack, rng, near_key="old_port")
        place = Place(
            name=v["name"],
            level="venue",
            lat=v.get("lat"),
            lon=v.get("lon"),
            address={"district": v.get("district")},
            meta={"category": v.get("category"), "source": "fictional"},
        )
        cand = Candidate(place=place, confidence=0.9, rationale="fictional_archetype")
        return ResolveResult(
            status=STATUS_EXACT,
            candidates=[cand],
            operations=[{"op":"poi.navigate","label":v["name"],"lat":v.get("lat"),"lon":v.get("lon"),"category":v.get("category"),"lore":{"district": v.get("district")}}],
            message=f"Heading to {v['name']} in {v['district']}.",
            anchor=anchor,
            scope=anchor.scope,
        )

    if any(k in target for k in ["dim sum","dumpling","yum cha","yank sing"]):
        v = _spawn_archetype(graph, "dim_sum", pack, rng, near_key="arts")
        place = Place(
            name=v["name"],
            level="venue",
            lat=v.get("lat"),
            lon=v.get("lon"),
            address={"district": v.get("district")},
            meta={"category": v.get("category"), "source": "fictional"},
        )
        cand = Candidate(place=place, confidence=0.9, rationale="fictional_archetype")
        return ResolveResult(
            status=STATUS_EXACT,
            candidates=[cand],
            operations=[{"op":"poi.navigate","label":v["name"],"lat":v.get("lat"),"lon":v.get("lon"),"category":v.get("category"),"lore":{"district": v.get("district")}}],
            message=f"Steam curls from {v['name']} in {v['district']}—let’s go.",
            anchor=anchor,
            scope=anchor.scope,
        )

    choices: List[str] = []
    if pack.get("archetypes"):
        for a in pack["archetypes"][:3]:
            pretty = a.get("slot","place").replace("_"," ").title()
            choices.append(f"{pretty} in {graph['districts'][0]['label']}")
    else:
        choices = ["night market", "waterfront sweets hall", "residential dunes on the west side"]

    return ResolveResult(
        status=STATUS_ASK,
        message=f"In {world_name}, what kind of place do you want—food, landmark, district, or festival?",
        choices=choices,
        anchor=anchor,
        scope=anchor.scope,
    )
