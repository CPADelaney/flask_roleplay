# nyx/location/fictional_resolver.py
from __future__ import annotations

import json
import logging
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from logic.chatgpt_integration import ALLOWS_TEMPERATURE, OpenAIClientManager
from logic.json_helpers import safe_json_loads
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

LOGGER = logging.getLogger(__name__)

_WORLD_SEED_MODEL = os.getenv("NYX_WORLD_SEED_MODEL", "gpt-4o-mini")
_WORLD_SEED_DIR = Path(os.getenv("NYX_WORLD_SEED_DIR", "nyx_data/world_seeds"))
try:
    _WORLD_SEED_MAX_TOKENS = int(os.getenv("NYX_WORLD_SEED_MAX_TOKENS", "1100"))
except ValueError:
    _WORLD_SEED_MAX_TOKENS = 1100
try:
    _WORLD_SEED_TEMPERATURE = float(os.getenv("NYX_WORLD_SEED_TEMPERATURE", "0.2"))
except ValueError:
    _WORLD_SEED_TEMPERATURE = 0.2

_WORLD_SEED_PROMPT = """You are Nyx's city architect.
Design a fictional urban world seed for collaborative roleplay.

Respond with JSON only, following this schema:
{
  "rules": ["string", ...],
  "themes": ["string", ...],
  "default_travel": {
    "mode": "string",
    "notes": "string",
    "average_minutes": number
  },
  "districts": [
    {"name": "string", "vibe": "string", "signature": "string"}
  ],
  "landmarks": [
    {"name": "string", "hook": "string"}
  ],
  "narrative_hooks": ["string", ...]
}

Concept: {concept}

Keep strings short and game-ready. Include at least three rules and themes.
Do not include markdown fences or commentary.
"""

_OPENAI_MANAGER = OpenAIClientManager()


class WorldSeedGenerationError(RuntimeError):
    """Raised when the world seed cannot be generated or parsed."""


def _slugify_concept(value: str) -> str:
    normalised = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalised.lower()).strip("-")
    return slug or "world"


def _seed_path_for_slug(slug: str) -> Path:
    return _WORLD_SEED_DIR / f"{slug}.json"


def _strip_json_markers(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    fence = re.match(r"```(?:json)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    return text


def _extract_response_text(response: Any) -> str:
    if getattr(response, "output_json", None):
        return json.dumps(response.output_json, ensure_ascii=False)
    if getattr(response, "output_text", None):
        text = response.output_text.strip()
        if text:
            return text
    for message in getattr(response, "output", []) or []:
        for part in getattr(message, "content", []) or []:
            part_type = getattr(part, "type", None)
            if part_type == "output_json" and getattr(part, "json", None) is not None:
                return json.dumps(part.json, ensure_ascii=False)
            if part_type == "output_text":
                text = getattr(part, "text", "").strip()
                if text:
                    return text
    for call in getattr(response, "tool_calls", []) or []:
        fn = getattr(call, "function", None)
        if fn and getattr(fn, "arguments", None):
            return json.dumps(fn.arguments, ensure_ascii=False)
    raise WorldSeedGenerationError("World seed LLM response did not contain parsable output.")


def _parse_seed_payload(raw_text: str) -> Dict[str, Any]:
    cleaned = _strip_json_markers(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        fallback = safe_json_loads(cleaned)
        if fallback:
            return fallback
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            snippet = match.group(0)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                fallback = safe_json_loads(snippet)
                if fallback:
                    return fallback
        raise WorldSeedGenerationError(
            f"World seed JSON parsing failed: {exc}. Snippet: {cleaned[:200]}"
        ) from exc


def _coerce_string_list(values: Any, *, field: str) -> List[str]:
    if not isinstance(values, list):
        raise WorldSeedGenerationError(f"World seed '{field}' must be a list of strings.")
    items: List[str] = []
    for value in values:
        if isinstance(value, str):
            candidate = value.strip()
        elif isinstance(value, dict) and "text" in value:
            candidate = str(value["text"]).strip()
        else:
            candidate = str(value).strip()
        if candidate:
            items.append(candidate)
    if not items:
        raise WorldSeedGenerationError(f"World seed '{field}' cannot be empty.")
    return items


def _normalise_default_travel(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise WorldSeedGenerationError("World seed 'default_travel' must be an object.")
    mode = (
        payload.get("mode")
        or payload.get("method")
        or payload.get("type")
        or payload.get("style")
    )
    if not mode or not str(mode).strip():
        raise WorldSeedGenerationError("World seed 'default_travel' requires a mode description.")
    details: Dict[str, Any] = {"mode": str(mode).strip()}
    notes = payload.get("notes") or payload.get("description") or payload.get("summary")
    if notes and str(notes).strip():
        details["notes"] = str(notes).strip()
    avg_minutes = None
    for key in ("average_minutes", "average_time_minutes", "travel_minutes", "duration_minutes"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            avg_minutes = float(value)
            break
        except (TypeError, ValueError):
            continue
    if avg_minutes is not None:
        details["average_minutes"] = avg_minutes
    for extra_key, extra_value in payload.items():
        if extra_key in {
            "mode",
            "method",
            "type",
            "style",
            "notes",
            "description",
            "summary",
            "average_minutes",
            "average_time_minutes",
            "travel_minutes",
            "duration_minutes",
        }:
            continue
        details.setdefault("extra", {})[extra_key] = extra_value
    return details


def _normalise_seed(payload: Dict[str, Any], *, concept: str, slug: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise WorldSeedGenerationError("World seed payload must be a JSON object.")
    rules = _coerce_string_list(payload.get("rules"), field="rules")
    themes = _coerce_string_list(payload.get("themes"), field="themes")
    default_travel = _normalise_default_travel(payload.get("default_travel"))
    seed: Dict[str, Any] = {
        "concept": concept,
        "slug": slug,
        "rules": rules,
        "themes": themes,
        "default_travel": default_travel,
    }
    for key, value in payload.items():
        if key in {"rules", "themes", "default_travel", "concept", "slug"}:
            continue
        seed[key] = value
    return seed


async def generate_world_seed(
    concept: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> Dict[str, Any]:
    """Generate and persist a fictional world seed for the provided concept."""

    if not concept or not concept.strip():
        raise WorldSeedGenerationError("Concept must be a non-empty string.")
    concept_text = concept.strip()
    slug = _slugify_concept(concept_text)
    prompt = _WORLD_SEED_PROMPT.format(concept=concept_text)
    instructions = (
        "You curate immersive fictional cities for Nyx. "
        "Always answer with strict JSON that matches the requested schema."
    )

    client = _OPENAI_MANAGER.async_client
    params: Dict[str, Any] = {
        "model": model or _WORLD_SEED_MODEL,
        "instructions": instructions,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": _WORLD_SEED_MAX_TOKENS,
    }
    resolved_temperature = _WORLD_SEED_TEMPERATURE if temperature is None else temperature
    if resolved_temperature is not None and params["model"] in ALLOWS_TEMPERATURE:
        params["temperature"] = resolved_temperature

    try:
        response = await client.responses.create(**params)
    except Exception as exc:  # pragma: no cover - network failure guard
        LOGGER.exception("World seed request failed", exc_info=exc)
        raise WorldSeedGenerationError(f"World seed request failed: {exc}") from exc

    raw_text = _extract_response_text(response)
    parsed = _parse_seed_payload(raw_text)
    seed = _normalise_seed(parsed, concept=concept_text, slug=slug)

    cache_path = _seed_path_for_slug(slug)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(seed, handle, ensure_ascii=False, indent=2)

    return seed


async def load_world_seed(
    concept: str,
    *,
    model: str | None = None,
    force_refresh: bool = False,
    temperature: float | None = None,
) -> Dict[str, Any]:
    """Load a cached world seed or generate it on-demand."""

    if not concept or not concept.strip():
        raise WorldSeedGenerationError("Concept must be a non-empty string.")
    concept_text = concept.strip()
    slug = _slugify_concept(concept_text)
    cache_path = _seed_path_for_slug(slug)

    if cache_path.exists() and not force_refresh:
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            return _normalise_seed(cached, concept=concept_text, slug=slug)
        except Exception as exc:
            LOGGER.warning("Failed to load cached world seed '%s': %s", slug, exc)

    return await generate_world_seed(
        concept_text,
        model=model,
        temperature=temperature,
    )

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
