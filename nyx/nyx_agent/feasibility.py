# nyx/nyx_agent/feasibility.py
"""
Dynamic feasibility system that learns what's possible/impossible in each unique setting.
Maintains reality consistency without hard-coded rules or repetitive responses.
"""

import asyncio
import json
import logging
import random
import re
import unicodedata
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from agents import Agent, Runner
from db.connection import get_db_connection_context
from logic.action_parser import parse_action_intents
from logic.aggregator_sdk import fallback_get_context
from nyx.nyx_agent.context import NyxContext

from nyx.feas.actions.mundane import evaluate_mundane
from nyx.feas.archetypes.modern_baseline import ModernBaseline
from nyx.feas.archetypes.roman_empire import RomanEmpire
from nyx.feas.archetypes.underwater_scifi import UnderwaterSciFi
from nyx.feas.capabilities import merge_caps
from nyx.feas.context import build_affordance_index
from nyx.location.anchors import derive_anchor_from_hierarchy
from nyx.geo.toponym import plausibility_score
from nyx.location.config import LocationSettings
from nyx.location.hierarchy import assign_hierarchy, get_or_create_location
from nyx.location.policies import resolver_policy_for_context
from nyx.location.types import (
    Anchor,
    Candidate,
    Location,
    Place,
    ResolveResult,
    Scope,
    STATUS_ASK,
    STATUS_EXACT,
    STATUS_MULTIPLE,
    STATUS_TRAVEL_PLAN,
)

logger = logging.getLogger(__name__)


ROLEPLAY_ONLY_DEFAULT: Set[str] = {
    "1",
    "always",
    "enforced",
    "enabled",
    "locked",
    "required",
    "strict",
    "true",
    "yes",
}

MODE_POLICY_TRUE_ALIASES: Set[str] = ROLEPLAY_ONLY_DEFAULT | {
    "roleplay_only",
    "roleplay-only",
    "roleplayonly",
}

MODE_POLICY_FALSE_ALIASES: Set[str] = {
    "0",
    "allow",
    "allowed",
    "disabled",
    "false",
    "no",
    "off",
    "open",
    "permissive",
    "relaxed",
}

OOC_PREFIX_MARKERS: Set[str] = {
    "[ooc]",
    "(ooc)",
    "{ooc}",
    "ooc:",
    "ooc-",
    "ooc—",
    "ooc ",
    "//ooc",
    "// ooc",
    "((ooc))",
}

OOC_KEYPHRASES: Set[str] = {
    "as an ai",
    "be yourself",
    "break character",
    "drop the roleplay",
    "for real",
    "meta question",
    "not roleplay",
    "ooc request",
    "out of character",
    "real world question",
    "serious question",
    "stop roleplaying",
    "talk normally",
    "this isn't roleplay",
}

BRAND_TERMS: Set[str] = {
    "amazon",
    "burger king",
    "chatgpt",
    "discord",
    "facebook",
    "github",
    "google",
    "hulu",
    "instagram",
    "kfc",
    "lyft",
    "mcdonalds",
    "microsoft",
    "netflix",
    "openai",
    "playstation",
    "reddit",
    "slack",
    "snapchat",
    "spotify",
    "starbucks",
    "tesla",
    "tiktok",
    "twitter",
    "uber",
    "walmart",
    "xbox",
    "youtube",
    "zoom",
}


def _normalize_mode_value(value: Any) -> Optional[str]:
    """Normalize a stored mode string into a canonical representation."""

    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).strip().lower()
        if not normalized:
            return None
        if normalized in {"diegetic", "in_character", "in-character", "ic"}:
            return "diegetic"
        if normalized in {"ooc", "out_of_character", "out-of-character", "outofcharacter"}:
            return "ooc"
    elif isinstance(value, bool):
        return "diegetic" if value else "ooc"

    return None


def _normalize_mode_policy_value(value: Any) -> Optional[str]:
    """Normalize a stored mode policy value to a canonical label."""

    if isinstance(value, bool):
        return "roleplay_only" if value else "open"

    if isinstance(value, (int, float)):
        return "roleplay_only" if value else "open"

    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).strip().lower()
        if not normalized:
            return None

        if normalized in MODE_POLICY_TRUE_ALIASES:
            return "roleplay_only"

        if normalized in MODE_POLICY_FALSE_ALIASES:
            return "open"

        if normalized in {"hybrid", "mixed"}:
            return "mixed"

        return normalized

    return None


IMPOSSIBLE_DEFAULT: Set[str] = {
    "physics_violation",
    "unaided_flight",
    "time_travel",
    "teleportation",
    "ex_nihilo_conjuration",
    "reality_warping",
    "orbital_travel",
    "spacewalk",
    "demon_summoning",
    "necromancy",
}


def _context_payload_for_agent(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of context for agent interactions."""

    sanitized: Dict[str, Any] = {}
    for key, value in context.items():
        # Private/internal keys are excluded from serialized payloads.
        if isinstance(key, str) and key.startswith("_"):
            continue

        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            try:
                sanitized[key] = value.to_dict()
                continue
            except Exception:
                logger.exception("Failed to convert %s to dict during serialization", key)

        sanitized[key] = value

    return sanitized
SAFE_BASELINE: Set[str] = {"mundane_action", "dialogue", "movement", "social", "trade"}


SETTING_DETECTION_TIMEOUT_SECONDS = 8


ARCHETYPE_REGISTRY = {
    "modern_baseline": ModernBaseline,
    "roman_empire": RomanEmpire,
    "underwater_scifi": UnderwaterSciFi,
}


PLAYER_SELF_REFERENCE_TOKENS: Set[str] = {
    "i",
    "me",
    "myself",
    "my self",
}

SELF_REFERENCE_STRIP_CHARS = ".,!?;:'\"`´“”’"


LOCATION_REFERENCE_KEYWORDS: Set[str] = {
    "here",
    "location",
    "right here",
    "current location",
    "current spot",
    "current place",
    "current area",
    "current room",
    "current zone",
    "this area",
    "this place",
    "this location",
    "this room",
    "around here",
    "around us",
    "around me",
    "where am i",
    "where i am",
    "where we are",
    "our location",
}

LOCATION_MOVE_CATEGORY_HINTS: Set[str] = {
    "movement",
    "travel",
    "navigation",
    "exploration",
    "relocation",
    "journey",
}

LOCATION_MOVE_TEXT_MARKERS: Tuple[str, ...] = (
    "go to",
    "head to",
    "head for",
    "move to",
    "move toward",
    "travel to",
    "travel toward",
    "walk to",
    "walk towards",
    "run to",
    "run toward",
    "sprint to",
    "dash to",
    "dash into",
    "drive to",
    "ride to",
    "sail to",
    "fly to",
    "step into",
    "step inside",
    "step toward",
    "enter the",
    "enter a",
    "enter an",
    "enter into",
    "venture to",
    "venture into",
    "sneak to",
    "sneak into",
    "make my way to",
    "make our way to",
    "make his way to",
    "make her way to",
    "make their way to",
    "leave for",
    "leave to",
)

LOCATION_MOVE_PLACEHOLDER_TOKENS: Set[str] = {
    "there",
    "here",
    "somewhere",
    "anywhere",
    "everywhere",
    "outside",
    "inside",
    "upstairs",
    "downstairs",
    "away",
    "back",
    "forward",
    "nearby",
    "around",
}

LOCATION_RESOLVER_ALLOW_THRESHOLD = 0.55
LOCATION_RESOLVER_ASK_THRESHOLD = 0.25
FICTIONAL_RESOLVER_ALLOW_THRESHOLD = 0.35
FICTIONAL_RESOLVER_ASK_THRESHOLD = 0.18

LOCATION_INTENT_KEYS: Tuple[str, ...] = (
    "destination",
    "location",
    "direct_object",
    "prepositional_object",
    "indirect_object",
    "targets",
)

SCENE_LOCATION_KEYS: Tuple[str, ...] = (
    "nearby_locations",
    "adjacent_locations",
    "connected_locations",
    "reachable_locations",
    "available_locations",
    "locations",
    "rooms",
    "areas",
    "wings",
)

REAL_WORLD_TOPONYM_KEYWORDS: Set[str] = {
    "airport",
    "avenue",
    "ave",
    "boulevard",
    "bridge",
    "campus",
    "center",
    "centre",
    "court",
    "ct",
    "dock",
    "drive",
    "dr",
    "expressway",
    "freeway",
    "garden",
    "gardens",
    "harbor",
    "harbour",
    "heights",
    "highway",
    "hwy",
    "lane",
    "ln",
    "library",
    "mall",
    "market",
    "marina",
    "museum",
    "park",
    "parkway",
    "pier",
    "plaza",
    "port",
    "road",
    "rd",
    "route",
    "rt",
    "square",
    "station",
    "street",
    "st",
    "subway",
    "terminal",
    "trail",
    "university",
    "way",
}

REAL_WORLD_TOPONYM_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "for",
    "and",
    "or",
    "near",
    "with",
    "without",
    "into",
    "onto",
    "over",
    "under",
    "between",
    "around",
}

REAL_WORLD_NUMERIC_STORE_PATTERN = re.compile(r"^\d+(?:[-/ ]\d+)+$")

GENERIC_VENUE_PREFIXES: Tuple[str, ...] = (
    "something like ",
    "some kind of ",
    "some sort of ",
    "somewhere like ",
    "anything like ",
    "any kind of ",
    "any sort of ",
    "an ",
    "the ",
    "some ",
    "any ",
    "a ",
)

GENERIC_VENUE_SUFFIXES: Tuple[str, ...] = (
    " or something",
    " or whatever",
    " maybe",
)

GENERIC_VENUE_LEADING_MODIFIERS: Set[str] = {
    "nearest",
    "nearby",
    "closest",
    "local",
    "any",
    "some",
    "good",
    "nice",
    "cheap",
    "fancy",
    "popular",
}

GENERIC_VENUE_ALIASES: Dict[str, Set[str]] = {
    "bar": {"bar", "pub", "tavern", "saloon", "speakeasy"},
    "shop": {
        "shop",
        "store",
        "general store",
        "boutique",
        "stall",
        "market stall",
        "bazaar",
        "storefront",
        "market",
    },
    "convenience store": {
        "convenience store",
        "corner store",
        "bodega",
        "7-11",
        "7 eleven",
    },
    "restaurant": {
        "restaurant",
        "diner",
        "eatery",
        "cafe",
        "coffee shop",
        "bistro",
    },
    "club": {"club", "nightclub", "dance club"},
    "park": {"park", "garden", "gardens", "botanical garden"},
    "station": {
        "station",
        "train station",
        "bus station",
        "subway station",
        "metro station",
        "tram station",
        "ferry terminal",
        "terminal",
    },
    "pier": {"pier", "dock", "harbor", "harbour", "marina"},
    "market": {"market", "street market", "farmer's market"},
    "library": {"library", "bookstore", "book shop"},
    "grocery store": {
        "grocery store", "grocer", "grocery", "supermarket", "food market",
        "whole foods", "whole foods market", "trader joes", "trader joe's",
        "safeway", "kroger", "ralphs", "albertsons", "publix", "heb", "h‑e‑b",
        "meijer", "hy-vee", "hyvee", "wegmans", "aldi", "lidl",
        "tesco", "sainsbury", "sainsbury's", "asda", "morrisons", "waitrose",
        "carrefour"
    },

    # (Optional but handy) coffee chains as a distinct bucket
    "coffee shop": {
        "coffee shop", "coffee", "cafe", "espresso bar","dutch bros","dutch bros."
        "starbucks", "dunkin", "costa", "peet's", "peets"
    },
}

GENERIC_VENUE_TERMS: Set[str] = set()
for canonical, synonyms in GENERIC_VENUE_ALIASES.items():
    GENERIC_VENUE_TERMS.add(canonical)
    GENERIC_VENUE_TERMS |= synonyms


def _is_modern_or_realistic_setting(setting_context: Optional[Dict[str, Any]]) -> bool:
    if not setting_context:
        return False

    markers: Set[str] = set()
    for key in (
        "kind",
        "type",
        "setting_kind",
        "setting_type",
        "reality_context",
        "technology_level",
        "setting_era",
    ):
        value = setting_context.get(key)
        if isinstance(value, str):
            markers.add(value.strip().lower())

    for marker in markers:
        if not marker:
            continue
        if any(hint in marker for hint in ("modern", "realistic", "contemporary", "urban")):
            return True
    return False


def _looks_like_real_world_toponym(
    original_token: str, normalized_token: str, setting_context: Optional[Dict[str, Any]]
) -> bool:
    if not _is_modern_or_realistic_setting(setting_context):
        return False

    normalized = (normalized_token or "").strip()
    if not normalized:
        return False

    normalized = normalized.replace("'", "")
    parts = [part for part in re.split(r"[\s\-]+", normalized) if part]
    if not parts:
        return False

    if REAL_WORLD_NUMERIC_STORE_PATTERN.match(normalized.replace(" ", "")):
        return True

    original = original_token or ""
    original_parts = [p for p in re.split(r"[\s\-]+", original) if p]
    if len(original_parts) >= 2:
        capitalized = sum(1 for p in original_parts if p[:1].isalpha() and p[:1].isupper())
        if capitalized >= 2:
            return True

    lower_parts = [part.lower() for part in parts]
    has_keyword = False
    keyword_support = 0
    supporting_signals = 0

    for idx, lower_part in enumerate(lower_parts):
        original_part = original_parts[idx] if idx < len(original_parts) else parts[idx]
        if lower_part in REAL_WORLD_TOPONYM_KEYWORDS:
            has_keyword = True
            if original_part[:1].isalpha() and original_part[:1].isupper():
                keyword_support += 1
        else:
            if (
                original_part[:1].isalpha()
                and original_part[:1].isupper()
                and lower_part not in REAL_WORLD_TOPONYM_STOPWORDS
            ):
                supporting_signals += 1
            if any(ch.isdigit() for ch in lower_part):
                supporting_signals += 1

    if not has_keyword:
        return False

    if any(part.isdigit() for part in lower_parts):
        for idx, lower_part in enumerate(lower_parts):
            if lower_part in REAL_WORLD_TOPONYM_KEYWORDS:
                if idx + 1 < len(lower_parts) and lower_parts[idx + 1].isdigit():
                    return True
                if idx > 0 and lower_parts[idx - 1].isdigit():
                    return True

    if supporting_signals >= 1 and keyword_support + supporting_signals >= 2:
        return True

    if keyword_support >= 2:
        return True

    return False

def _extract_geo_hints_from_location(location: Any) -> Dict[str, Optional[str]]:
    """
    Pulls common geo fields out of a location payload, tolerating multiple schemas.
    Returns {district|neighborhood|borough -> 'district', city/town -> 'city', region/state/province -> 'region', country}.
    """
    out = {"district": None, "city": None, "region": None, "country": None}
    if not isinstance(location, dict):
        return out

    def _get(*keys: str) -> Optional[str]:
        for k in keys:
            v = location.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    out["district"] = _get("district", "neighborhood", "borough")
    out["city"] = _get("city", "town", "municipality")
    out["region"] = _get("region", "state", "province")
    out["country"] = _get("country")
    return out


def _derive_near_string(setting_context: Optional[Dict[str, Any]], fallback: Optional[str] = None) -> Optional[str]:
    """
    Build the best 'near' string for geocoding when the current location name is fictional.
    Prefers: district+city → city → region → world_model.resolver.anchor → any known real toponym.
    """
    if not isinstance(setting_context, dict):
        return fallback

    # 1) Try structured fields on the current location
    loc = setting_context.get("location") or {}
    hints = _extract_geo_hints_from_location(loc)
    parts = []
    if hints.get("district"):
        parts.append(hints["district"])
    if hints.get("city"):
        parts.append(hints["city"])
    if parts:
        return ", ".join(parts)

    # 2) Try world model resolver anchor (if present)
    wm = setting_context.get("world_model") or {}
    res = wm.get("resolver") or {}
    anchor = res.get("anchor") or {}
    if isinstance(anchor, dict):
        anchor_parts = []
        if anchor.get("district"):
            anchor_parts.append(str(anchor["district"]).strip())
        if anchor.get("city"):
            anchor_parts.append(str(anchor["city"]).strip())
        if anchor_parts:
            return ", ".join(p for p in anchor_parts if p)

        # fall back to region if no city
        if anchor.get("region"):
            return str(anchor["region"]).strip()

    # 3) Try any known real toponym we've seen in this conversation
    for name in setting_context.get("known_location_names", []):
        norm = _normalize_location_phrase(name)
        if _looks_like_real_world_toponym(name, norm, setting_context):
            return name

    return fallback


def _normalize_generic_venue_phrase(normalized_token: str) -> str:
    candidate = (normalized_token or "").strip()
    if not candidate:
        return ""

    candidate = candidate.replace("-", " ")

    for prefix in sorted(GENERIC_VENUE_PREFIXES, key=len, reverse=True):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break

    for suffix in GENERIC_VENUE_SUFFIXES:
        if candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            break

    candidate = candidate.strip()
    if not candidate:
        return ""

    parts = candidate.split()
    while parts and parts[0] in GENERIC_VENUE_LEADING_MODIFIERS:
        parts.pop(0)

    candidate = " ".join(parts).strip()
    return candidate


def _matches_generic_venue_request(
    original_token: str,
    normalized_token: str,
    setting_context: Optional[Dict[str, Any]],
    location_context_tokens: Set[str],
    known_location_tokens: Set[str],
) -> bool:
    if not _is_modern_or_realistic_setting(setting_context):
        return False

    normalized = (normalized_token or "").strip()
    if normalized and normalized in BRAND_TERMS:
        # Brand-specific venues should route through the real place resolver
        # rather than being treated as generic venue placeholders.
        return False

    candidate = _normalize_generic_venue_phrase(normalized_token)
    if not candidate:
        return False

    lower_original = (original_token or "").strip().lower()
    if lower_original and lower_original in GENERIC_VENUE_TERMS:
        return True

    if candidate in GENERIC_VENUE_TERMS:
        return True

    if candidate in location_context_tokens or candidate in known_location_tokens:
        return True

    parts = candidate.split()
    if not parts:
        return False

    if parts[-1] in GENERIC_VENUE_TERMS:
        return True

    if len(parts) >= 2:
        trailing_two = " ".join(parts[-2:])
        if trailing_two in GENERIC_VENUE_TERMS:
            return True

    return False

MUNDANE_SEARCH_TOKEN_SYNONYMS: Dict[str, Set[str]] = {
    "coin": {"coin", "coins", "penny", "pennies", "small coin", "loose coin"},
    "rock": {"rock", "rocks", "stone", "stones", "boulder", "boulders"},
    "pebble": {"pebble", "pebbles"},
    "small_object": {
        "small object",
        "small_object",
        "small item",
        "small items",
        "tiny object",
        "tiny item",
        "little object",
        "little item",
        "trinket",
        "trinkets",
    },
}

MUNDANE_SEARCH_TEXT_MARKERS: Tuple[str, ...] = (
    "look for",
    "search for",
    "search around",
    "look around for",
    "scan for",
    "scour",
    "hunt for",
    "feel around for",
    "fish around for",
    "check for",
    "dig around for",
    "poke around for",
)

SEARCH_CATEGORY_HINTS: Set[str] = {
    "investigation",
    "search",
    "perception",
    "exploration",
    "explore",
    "survey",
    "scavenge",
    "scout",
}

MUNDANE_SEARCH_CONTEXT_HINTS: Dict[str, Set[str]] = {
    "coin": {
        "market",
        "shop",
        "store",
        "street",
        "road",
        "alley",
        "tavern",
        "inn",
        "hall",
        "floor",
        "ground",
        "camp",
        "dormitory",
        "bedroom",
        "casino",
        "bank",
        "bazaar",
        "plaza",
        "square",
        "town",
        "village",
        "marketplace",
        "counter",
        "desk",
        "table",
        "bar",
    },
    "rock": {
        "ground",
        "dirt",
        "soil",
        "grass",
        "trail",
        "path",
        "forest",
        "woods",
        "cave",
        "cavern",
        "mountain",
        "cliff",
        "hillside",
        "shore",
        "beach",
        "river",
        "riverbank",
        "stream",
        "lake",
        "pond",
        "garden",
        "yard",
        "field",
        "rubble",
        "ruins",
        "quarry",
        "mine",
        "stone",
        "rock",
    },
    "pebble": {
        "ground",
        "dirt",
        "soil",
        "trail",
        "path",
        "forest",
        "woods",
        "cave",
        "cavern",
        "mountain",
        "cliff",
        "shore",
        "beach",
        "river",
        "riverbank",
        "stream",
        "lake",
        "pond",
        "garden",
        "yard",
        "field",
        "rubble",
        "ruins",
        "stone",
        "gravel",
        "rock",
    },
    "small_object": {
        "desk",
        "table",
        "shelf",
        "drawer",
        "counter",
        "crate",
        "box",
        "market",
        "shop",
        "store",
        "workshop",
        "laboratory",
        "office",
        "room",
        "hall",
        "cabin",
        "floor",
        "ground",
        "camp",
        "cargo",
    },
}


def _normalize_location_phrase(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("_", " ").split()).strip().lower()


def _normalize_self_reference_phrase(value: Any) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split()).strip().lower()


def _canonicalize_self_reference_token(token: Any) -> str:
    normalized = _normalize_self_reference_phrase(token)
    if not normalized:
        return ""
    stripped = normalized.strip(SELF_REFERENCE_STRIP_CHARS)
    collapsed = stripped.replace(" ", "")
    if stripped in PLAYER_SELF_REFERENCE_TOKENS:
        return stripped
    if collapsed in PLAYER_SELF_REFERENCE_TOKENS:
        return collapsed
    return ""


def _is_self_reference_token(token: Any) -> bool:
    return bool(_canonicalize_self_reference_token(token))


def _location_reference_aliases(location_token: Optional[str], scene: Any) -> Set[str]:
    aliases: Set[str] = set(LOCATION_REFERENCE_KEYWORDS)
    if location_token:
        raw = str(location_token).strip().lower()
        if raw:
            aliases.add(raw)
        normalized = _normalize_location_phrase(location_token)
        if normalized:
            aliases.add(normalized)
            for part in normalized.split():
                if len(part) > 2:
                    aliases.add(part)

    if isinstance(scene, dict):
        location_payload = scene.get("location")
        location_type = scene.get("location_type")
        location_kind = None

        if isinstance(location_payload, dict):
            location_kind = (
                location_payload.get("type")
                or location_payload.get("category")
                or location_payload.get("kind")
            )
        elif isinstance(location_payload, str):
            location_kind = location_payload

        location_kind = location_kind or location_type
        normalized_kind = _normalize_location_phrase(location_kind)
        if normalized_kind:
            aliases.add(normalized_kind)
            for part in normalized_kind.split():
                if len(part) > 2:
                    aliases.add(part)

    return {alias for alias in aliases if alias}


def _canonicalize_mundane_search_token(token: str) -> Optional[str]:
    raw = str(token).strip().lower()
    normalized = _normalize_location_phrase(token)
    for canonical, synonyms in MUNDANE_SEARCH_TOKEN_SYNONYMS.items():
        if raw in synonyms or normalized in synonyms:
            return canonical
    if normalized.endswith("s"):
        singular = normalized[:-1]
        for canonical, synonyms in MUNDANE_SEARCH_TOKEN_SYNONYMS.items():
            if singular in synonyms:
                return canonical
    return None


def _has_search_signal(text_l: str, categories: Set[str]) -> bool:
    category_markers = {c.strip().lower() for c in categories if c}
    if category_markers & SEARCH_CATEGORY_HINTS:
        return True
    return any(marker in text_l for marker in MUNDANE_SEARCH_TEXT_MARKERS)


def _collect_location_context_tokens(
    scene: Any,
    location_features: Any,
    location_token: str,
) -> Set[str]:
    tokens: Set[str] = set()
    tokens |= _tokenize_scene_values(location_features)

    if isinstance(scene, dict):
        tokens |= _tokenize_scene_values(scene.get("location_type"))
        tokens |= _tokenize_scene_values(scene.get("terrain"))
        tokens |= _tokenize_scene_values(scene.get("biome"))
        tokens |= _tokenize_scene_values(scene.get("environment"))
        location_payload = scene.get("location")
        if location_payload is not None:
            tokens |= _tokenize_scene_values(location_payload)

    if location_token:
        raw = str(location_token).strip().lower()
        if raw:
            tokens.add(raw)
        normalized = _normalize_location_phrase(location_token)
        if normalized:
            tokens.add(normalized)

    return {token for token in tokens if token}


def _format_missing_names(tokens: Iterable[str]) -> str:
    unique = sorted({str(token).strip() for token in tokens if str(token).strip()})
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def _intent_requests_location_move(intent: Dict[str, Any], text_l: str, candidate_tokens: Set[str]) -> bool:
    if not candidate_tokens:
        return False

    categories = {
        str(cat).strip().lower()
        for cat in intent.get("categories", [])
        if str(cat).strip()
    }
    if categories & LOCATION_MOVE_CATEGORY_HINTS:
        return True

    for key in ("destination", "location"):
        if intent.get(key):
            return True

    text_l = text_l or ""
    if not any(marker in text_l for marker in LOCATION_MOVE_TEXT_MARKERS):
        return False

    normalized_candidates: Set[str] = set()
    for token in candidate_tokens:
        normalized = _normalize_location_phrase(token)
        if not normalized:
            continue
        normalized_candidates.add(normalized)
        normalized_candidates.update(part for part in normalized.split() if part)

    location_vocab = (
        GENERIC_VENUE_TERMS
        | REAL_WORLD_TOPONYM_KEYWORDS
        | LOCATION_REFERENCE_KEYWORDS
    )

    return bool(normalized_candidates & location_vocab)


def _extract_candidate_location_tokens(intent: Dict[str, Any]) -> Set[str]:
    tokens: Set[str] = set()
    for key in LOCATION_INTENT_KEYS:
        tokens |= _tokenize_scene_values(intent.get(key))
    return {token for token in tokens if token}


def _build_known_location_tokens(
    location_aliases: Set[str],
    location_context_tokens: Set[str],
    known_location_names: Iterable[str],
    scene: Any,
) -> Set[str]:
    tokens: Set[str] = set()

    def _ingest(value: Any) -> None:
        if value is None:
            return
        raw = str(value).strip().lower()
        if raw:
            tokens.add(raw)
        normalized = _normalize_location_phrase(value)
        if normalized:
            tokens.add(normalized)
            for part in normalized.split():
                if len(part) > 2:
                    tokens.add(part)

    for alias in location_aliases:
        _ingest(alias)

    for ctx_token in location_context_tokens:
        _ingest(ctx_token)

    for name in known_location_names or []:
        _ingest(name)

    if isinstance(scene, dict):
        for key in SCENE_LOCATION_KEYS:
            _values = scene.get(key)
            for token in _tokenize_scene_values(_values):
                _ingest(token)

    return {token for token in tokens if token}


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            raise ValueError
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _world_model_branch(
    setting_context: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, bool, bool]:
    world_model: Dict[str, Any] = {}
    branch = "modern_realistic"
    allow_fictional = False

    if isinstance(setting_context, dict):
        raw_world_model = setting_context.get("world_model")
        if isinstance(raw_world_model, dict):
            world_model = raw_world_model

        branch_candidate = str(
            world_model.get("branch")
            or world_model.get("kind")
            or world_model.get("mode")
            or setting_context.get("kind")
            or setting_context.get("type")
            or ""
        ).strip().lower()
        if branch_candidate:
            branch = branch_candidate

        allow_fictional = bool(world_model.get("allow_fictional_locations"))
        if not allow_fictional:
            reality = str(setting_context.get("reality_context") or "").strip().lower()
            if reality and reality not in {"normal", "mundane", "realistic"}:
                allow_fictional = True

    is_real_branch = any(
        token in branch
        for token in ("modern", "real", "baseline", "contemporary")
    )

    if not allow_fictional and not is_real_branch:
        allow_fictional = True

    return world_model, branch, is_real_branch, allow_fictional


def _normalize_world_model_metadata(
    raw_metadata: Any,
    base_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    fallback = base_context or {}

    branch = str(
        metadata.get("branch")
        or metadata.get("kind")
        or metadata.get("mode")
        or fallback.get("kind")
        or fallback.get("type")
        or "modern_realistic"
    ).strip().lower()
    if not branch:
        branch = "modern_realistic"

    is_real_branch = any(
        token in branch for token in ("modern", "real", "baseline", "contemporary")
    )

    allow_fictional = bool(metadata.get("allow_fictional_locations"))
    reality = str((fallback.get("reality_context") or "")).strip().lower()
    if not allow_fictional:
        if reality and reality not in {"normal", "mundane", "realistic"}:
            allow_fictional = True
        elif not is_real_branch:
            allow_fictional = True

    resolver_cfg_raw = (
        metadata.get("resolver")
        or metadata.get("location_resolver")
        or {}
    )
    resolver_cfg = dict(resolver_cfg_raw) if isinstance(resolver_cfg_raw, dict) else {}

    world_model_context: Dict[str, Any] = {}
    if isinstance(fallback.get("world_model"), dict):
        world_model_context = dict(fallback["world_model"])
    elif isinstance(metadata, dict):
        world_model_context = metadata

    fallback_resolver_cfg: Dict[str, Any] = {}
    candidate = world_model_context.get("resolver") or world_model_context.get("location_resolver")
    if isinstance(candidate, dict):
        fallback_resolver_cfg = dict(candidate)

    allow_source = resolver_cfg.get("allow_threshold")
    if allow_source is None:
        allow_source = fallback_resolver_cfg.get("allow_threshold")
    allow_threshold = _coerce_float(
        allow_source,
        LOCATION_RESOLVER_ALLOW_THRESHOLD if is_real_branch else FICTIONAL_RESOLVER_ALLOW_THRESHOLD,
    )

    ask_source = resolver_cfg.get("ask_threshold")
    if ask_source is None:
        ask_source = fallback_resolver_cfg.get("ask_threshold")
    ask_threshold = _coerce_float(
        ask_source,
        LOCATION_RESOLVER_ASK_THRESHOLD if is_real_branch else FICTIONAL_RESOLVER_ASK_THRESHOLD,
    )
    if ask_threshold >= allow_threshold:
        ask_threshold = max(min(allow_threshold * 0.75, allow_threshold - 0.05), 0.0)

    fictional_policy = str(
        resolver_cfg.get("fictional_policy")
        or fallback_resolver_cfg.get("fictional_policy")
        or metadata.get("fictional_location_policy")
        or world_model_context.get("fictional_location_policy")
        or ("allow" if allow_fictional else "deny")
    ).strip().lower()
    if fictional_policy not in {"allow", "ask", "deny"}:
        fictional_policy = "allow" if allow_fictional else "deny"

    # NEW: normalize a geo anchor if provided (tolerate both nested and flat keys)
    anchor_raw = resolver_cfg.get("anchor") or metadata.get("anchor") or {}
    anchor = {}
    if isinstance(anchor_raw, dict):
        anchor = {
            "district": (str(anchor_raw.get("district") or "").strip() or None),
            "city": (str(anchor_raw.get("city") or "").strip() or None),
            "region": (str(anchor_raw.get("region") or anchor_raw.get("state") or "").strip() or None),
            "country": (str(anchor_raw.get("country") or "").strip() or None),
            "lat": anchor_raw.get("lat"),
            "lon": anchor_raw.get("lon"),
        }
    else:
        anchor = {
            "city": (str(resolver_cfg.get("anchor_city") or metadata.get("anchor_city") or "").strip() or None),
            "region": (str(resolver_cfg.get("anchor_region") or metadata.get("anchor_region") or "").strip() or None),
            "country": (str(resolver_cfg.get("anchor_country") or metadata.get("anchor_country") or "").strip() or None),
        }

    return {
        "branch": branch,
        "allow_fictional_locations": allow_fictional,
        "resolver": {
            "allow_threshold": allow_threshold,
            "ask_threshold": ask_threshold,
            "fictional_policy": fictional_policy,
            "anchor": anchor,  # <-- keep the normalized anchor handy
        },
        "raw": metadata,
    }


def _guess_requested_kind(
    token: str, setting_context: Optional[Dict[str, Any]]
) -> str:
    token_l = (token or "").strip().lower()
    _, branch, is_real_branch, allow_fictional = _world_model_branch(setting_context)

    fictional_markers = {
        "nebula",
        "asteroid",
        "moon",
        "galaxy",
        "quantum",
        "void",
        "portal",
        "dimension",
        "celestial",
        "cosmic",
        "lunar",
        "stellar",
        "warp",
        "citadel",
    }
    realistic_markers = {
        "harbor",
        "harbour",
        "park",
        "pier",
        "square",
        "street",
        "avenue",
        "airport",
        "university",
        "station",
        "bridge",
        "boulevard",
        "campus",
        "museum",
    }

    if any(marker in token_l for marker in fictional_markers):
        return "fictional"
    if any(marker in token_l for marker in realistic_markers):
        return "real_world"

    if allow_fictional and not is_real_branch:
        return "fictional"

    return "real_world"


def _resolver_feedback_for_token(
    token: str, resolver_cache: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not resolver_cache:
        return None

    raw = (token or "").strip().lower()
    normalized = _normalize_location_phrase(token)
    for candidate in filter(None, {normalized, raw}):
        verdict = resolver_cache.get(candidate)
        if verdict:
            return verdict
    return None


def _is_plausible_location_token(
    token: str, resolver_cache: Dict[str, Dict[str, Any]]
) -> bool:
    verdict = _resolver_feedback_for_token(token, resolver_cache)
    if verdict:
        return verdict.get("decision") == "allow"
    return False


def _format_admin_preview(path: Optional[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
    if not isinstance(path, dict):
        return None, []

    order = (
        "building",
        "venue",
        "street",
        "neighborhood",
        "district",
        "city",
        "region",
        "country",
    )
    tokens: List[str] = []
    for level in order:
        value = path.get(level)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)

    preview = ", ".join(tokens) if tokens else None
    return preview, tokens


def _resolver_candidate_preview(candidate: Candidate) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, Candidate):
        return None

    address = candidate.place.address or {}
    preview, tokens = _format_admin_preview(address.get("_normalized_admin_path"))
    if not preview:
        return None

    meta = candidate.place.meta or {}
    return {
        "name": candidate.place.name,
        "admin_path": preview,
        "admin_tokens": tokens,
        "confidence": candidate.confidence,
        "place_key": meta.get("place_key") or candidate.place.key,
        "place_id": meta.get("place_id"),
    }


def _coerce_resolve_result(result: Any) -> Optional[ResolveResult]:
    if isinstance(result, ResolveResult):
        return result
    if not isinstance(result, dict):
        return None

    candidates_raw = result.get("candidates") or []
    candidates: List[Candidate] = []
    for item in candidates_raw:
        if isinstance(item, Candidate):
            candidates.append(item)

    anchor = result.get("anchor") if isinstance(result.get("anchor"), Anchor) else None
    operations = result.get("operations") or result.get("canonical_ops") or []

    location_obj: Optional[Location] = None
    location_payload = result.get("location")
    if isinstance(location_payload, Location):
        location_obj = location_payload
    elif isinstance(location_payload, dict):
        try:
            location_obj = Location.from_record(location_payload)
        except Exception:
            location_obj = None

    return ResolveResult(
        status=result.get("status"),
        message=result.get("message"),
        choices=list(result.get("choices") or []),
        candidates=candidates,
        operations=list(operations or []),
        anchor=anchor,
        scope=result.get("scope"),
        errors=list(result.get("errors") or []),
        location=location_obj,
    )


def _scope_from_resolver_decision(decision: Dict[str, Any]) -> Scope:
    branch = str(decision.get("branch") or "").lower()
    allow_fictional = bool(decision.get("allow_fictional"))
    is_real_branch = bool(decision.get("is_real_branch"))

    if "hybrid" in branch:
        return "hybrid"
    if is_real_branch:
        return "real"
    if any(token in branch for token in ("fiction", "fantasy", "sci", "space")):
        return "fictional"
    if allow_fictional and not is_real_branch:
        return "fictional"
    return "real"


def _extract_coordinates(payload: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(payload, dict):
        return None, None

    def _coerce(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    lat = None
    lon = None

    for key in ("lat", "latitude"):
        lat = _coerce(payload.get(key))
        if lat is not None:
            break
    for key in ("lon", "long", "longitude"):
        lon = _coerce(payload.get(key))
        if lon is not None:
            break

    coords = payload.get("coords") or payload.get("coordinates")
    if isinstance(coords, dict):
        if lat is None:
            for key in ("lat", "latitude"):
                lat = _coerce(coords.get(key))
                if lat is not None:
                    break
        if lon is None:
            for key in ("lon", "long", "longitude"):
                lon = _coerce(coords.get(key))
                if lon is not None:
                    break

    return lat, lon


def _anchor_from_setting_context(
    setting_context: Optional[Dict[str, Any]],
    *,
    scope: Scope,
    label: Optional[str] = None,
) -> Optional[Anchor]:
    if not isinstance(setting_context, dict):
        return None

    location_payload = setting_context.get("location") or {}
    hints = _extract_geo_hints_from_location(location_payload)
    lat, lon = _extract_coordinates(location_payload)

    label_value = (
        label
        or location_payload.get("display_name")
        or location_payload.get("name")
        or location_payload.get("label")
        or setting_context.get("setting_name")
        or None
    )

    world_name: Optional[str] = None
    search_payloads: List[Dict[str, Any]] = []
    world_model = setting_context.get("world_model")
    if isinstance(world_model, dict):
        search_payloads.append(world_model)
    for key in ("world", "world_meta", "world_info"):
        payload = setting_context.get(key)
        if isinstance(payload, dict):
            search_payloads.append(payload)

    for payload in search_payloads:
        for candidate_key in ("world_name", "name", "label", "title"):
            value = payload.get(candidate_key)
            if isinstance(value, str) and value.strip():
                world_name = value.strip()
                break
        if world_name:
            break

    focus: Optional[Place] = None
    if hints.get("city"):
        focus = Place(
            name=hints["city"],
            level="city",
            lat=lat,
            lon=lon,
            address={k: v for k, v in hints.items() if v},
            meta={"source": "feasibility_anchor"},
        )

    hints_payload = {
        "location": location_payload,
        "world_model": world_model,
    }

    return Anchor(
        scope=scope,
        focus=focus,
        label=label_value,
        lat=lat,
        lon=lon,
        primary_city=hints.get("city"),
        region=hints.get("region"),
        country=hints.get("country"),
        world_name=world_name,
        hints=hints_payload,
    )


def _build_minted_candidate(
    name: str,
    normalized: Optional[str],
    decision: Dict[str, Any],
    setting_context: Optional[Dict[str, Any]],
    *,
    anchor: Optional[Anchor],
    scope: Scope,
) -> Candidate:
    token = (normalized or name or "").strip().lower()

    def _has_any(markers: Iterable[str]) -> bool:
        return any(marker in token for marker in markers)

    if _has_any({"district", "quarter", "ward", "borough", "neighborhood"}):
        level = "district"
    elif _has_any({"city", "town", "village", "metropolis", "capital"}):
        level = "city"
    elif _has_any({"region", "province", "state", "kingdom", "realm", "empire", "territory"}):
        level = "region"
    else:
        level = "venue"

    hints = (
        _extract_geo_hints_from_location(setting_context.get("location"))
        if isinstance(setting_context, dict)
        else {"district": None, "city": None, "region": None, "country": None}
    )

    address: Dict[str, Any] = {
        key: value
        for key, value in (
            ("district", hints.get("district")),
            ("city", hints.get("city")),
            ("region", hints.get("region")),
            ("country", hints.get("country")),
        )
        if value
    }

    if anchor:
        if anchor.primary_city and "city" not in address:
            address["city"] = anchor.primary_city
        if anchor.region and "region" not in address:
            address["region"] = anchor.region
        if anchor.country and "country" not in address:
            address["country"] = anchor.country

    lat, lon = None, None
    location_payload = setting_context.get("location") if isinstance(setting_context, dict) else {}
    loc_lat, loc_lon = _extract_coordinates(location_payload)
    if anchor and anchor.lat is not None:
        lat = anchor.lat
    elif loc_lat is not None:
        lat = loc_lat
    if anchor and anchor.lon is not None:
        lon = anchor.lon
    elif loc_lon is not None:
        lon = loc_lon

    key_candidate = (normalized or name or "").strip().lower().replace(" ", "_")
    place_key = key_candidate or None

    meta: Dict[str, Any] = {
        "source": "feasibility_resolver",
        "display_name": name,
        "resolver_branch": decision.get("branch"),
        "resolver_score": decision.get("score"),
        "minted": True,
        "minted_at": datetime.utcnow().isoformat(),
        "category": decision.get("kind"),
        "normalized_token": normalized,
    }

    if anchor and anchor.world_name:
        meta["world_name"] = anchor.world_name
    if scope != "real":
        meta["is_fictional_branch"] = True
    if isinstance(location_payload, dict):
        parent_name = (
            location_payload.get("name")
            or location_payload.get("display_name")
            or location_payload.get("label")
        )
        if parent_name:
            meta.setdefault("parent_location", parent_name)

    for key, value in address.items():
        if value and key not in meta:
            meta[key] = value

    confidence = float(decision.get("score") or 0.0)

    place = Place(
        name=name,
        level=level,  # type: ignore[arg-type]
        key=place_key,
        lat=lat,
        lon=lon,
        address=address,
        meta=meta,
    )

    return Candidate(place=place, confidence=confidence)


async def _resolve_location_candidate(
    original: str,
    normalized: Optional[str],
    setting_context: Optional[Dict[str, Any]],
    resolver_cache: Dict[str, Dict[str, Any]],
    inferred_kind: Optional[str] = None,
) -> Dict[str, Any]:
    existing = _resolver_feedback_for_token(normalized or original, resolver_cache)
    if existing:
        return existing

    world_model, branch, is_real_branch, allow_fictional = _world_model_branch(setting_context)
    resolver_cfg: Dict[str, Any] = {}
    if isinstance(world_model.get("resolver"), dict):
        resolver_cfg = dict(world_model["resolver"])
    elif isinstance(world_model.get("location_resolver"), dict):
        resolver_cfg = dict(world_model["location_resolver"])

    allow_threshold = _coerce_float(
        resolver_cfg.get("allow_threshold"),
        LOCATION_RESOLVER_ALLOW_THRESHOLD if is_real_branch else FICTIONAL_RESOLVER_ALLOW_THRESHOLD,
    )
    ask_threshold = _coerce_float(
        resolver_cfg.get("ask_threshold"),
        LOCATION_RESOLVER_ASK_THRESHOLD if is_real_branch else FICTIONAL_RESOLVER_ASK_THRESHOLD,
    )
    if ask_threshold >= allow_threshold:
        ask_threshold = max(min(allow_threshold * 0.75, allow_threshold - 0.05), 0.0)

    fictional_policy = str(
        resolver_cfg.get("fictional_policy")
        or world_model.get("fictional_location_policy")
        or ("allow" if allow_fictional else "deny")
    ).strip().lower()
    if fictional_policy not in {"allow", "ask", "deny"}:
        fictional_policy = "allow" if allow_fictional else "deny"

    token_kind = (inferred_kind or _guess_requested_kind(original, setting_context) or "real_world").lower()

    # --- Updated anchor discovery ---
    use_near: Optional[str] = None
    near_hint: Optional[str] = None
    anchor_candidate: Optional[str] = None
    current_location_obj = None
    context_dict: Dict[str, Any] = setting_context if isinstance(setting_context, dict) else {}

    if context_dict:
        current_location_obj = context_dict.get("_location_object")
        if current_location_obj is None:
            maybe_public_location = context_dict.get("location_object")
            if isinstance(maybe_public_location, Location):
                current_location_obj = maybe_public_location

        location_ctx = context_dict.get("location")
        if isinstance(location_ctx, dict):
            for key in ("name", "display_name", "label", "title"):
                value = location_ctx.get(key)
                if isinstance(value, str) and value.strip():
                    near_hint = value.strip()
                    break
        elif isinstance(location_ctx, str) and location_ctx.strip():
            near_hint = location_ctx.strip()

    if current_location_obj is not None:
        try:
            anchor_candidate = derive_anchor_from_hierarchy(current_location_obj)
        except Exception:
            logger.exception("Failed to derive anchor from location hierarchy", exc_info=True)

    if near_hint:
        normalized_hint = _normalize_location_phrase(near_hint)
        if normalized_hint and _looks_like_real_world_toponym(near_hint, normalized_hint, context_dict):
            use_near = near_hint

    if not use_near and anchor_candidate:
        use_near = anchor_candidate

    if not use_near:
        use_near = _derive_near_string(setting_context, fallback=near_hint)

    score = float(await plausibility_score(original, near=use_near))
    minted = False

    decision = "deny"
    reason = ""

    if token_kind != "fictional" and is_real_branch:
        if score >= allow_threshold:
            decision = "allow"
            reason = f"Resolver recognized '{original}' (score {score:.2f})."
        elif score >= ask_threshold:
            decision = "ask"
            reason = f"Resolver is unsure about '{original}' (score {score:.2f})."
        else:
            decision = "deny"
            reason = f"Resolver rejected '{original}' (score {score:.2f})."
    else:
        if score >= allow_threshold:
            decision = "allow"
            reason = f"Resolver recognized '{original}' (score {score:.2f})."
        elif allow_fictional and fictional_policy == "allow":
            decision = "allow"
            minted = True
            reason = (
                f"Resolver minted '{original}' for the {branch or 'fictional'} branch."
            )
        elif allow_fictional and fictional_policy == "ask":
            decision = "ask"
            reason = (
                f"Resolver can mint '{original}' if you confirm it belongs in this story."
            )
        elif score >= ask_threshold:
            decision = "ask"
            reason = f"Resolver is unsure about '{original}' (score {score:.2f})."
        else:
            decision = "deny"
            if allow_fictional:
                reason = (
                    f"Resolver rejected '{original}' because minting is disabled for this branch (score {score:.2f})."
                )
            else:
                reason = f"Resolver rejected '{original}' (score {score:.2f})."

    payload = {
        "decision": decision,
        "reason": reason,
        "score": score,
        "kind": token_kind,
        "token": original,
        "normalized": normalized,
        "branch": branch,
        "minted": minted,
        "near_used": use_near,  # helpful for logs
        "allow_fictional": allow_fictional,
        "is_real_branch": is_real_branch,
    }

    for candidate in filter(None, {normalized, (original or "").strip().lower()}):
        resolver_cache[candidate] = payload

    return payload


async def _find_unresolved_location_targets(
    intent: Dict[str, Any],
    text_l: str,
    location_aliases: Set[str],
    location_context_tokens: Set[str],
    known_location_tokens: Set[str],
    scene_npc_tokens: Set[str],
    scene_item_tokens: Set[str],
    setting_context: Optional[Dict[str, Any]],
    *,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None,
) -> List[str]:
    candidate_tokens = _extract_candidate_location_tokens(intent)
    if not _intent_requests_location_move(intent, text_l, candidate_tokens):
        return []

    resolver_cache: Dict[str, Dict[str, Any]] = {}
    minted_locations: List[str] = []
    if isinstance(setting_context, dict):
        resolver_cache = setting_context.setdefault("location_resolver_cache", {})
        existing_minted = setting_context.setdefault("resolver_minted_locations", [])
        if isinstance(existing_minted, list):
            minted_locations = existing_minted
        else:
            minted_locations = []
            setting_context["resolver_minted_locations"] = minted_locations

    unresolved: List[str] = []
    for token in candidate_tokens:
        original = str(token).strip()
        raw = original.lower()
        normalized = _normalize_location_phrase(token)
        if not raw and not normalized:
            continue
        if raw in LOCATION_MOVE_PLACEHOLDER_TOKENS or normalized in LOCATION_MOVE_PLACEHOLDER_TOKENS:
            continue
        if _canonicalize_self_reference_token(token):
            continue
        if normalized and _canonicalize_self_reference_token(normalized):
            continue
        if raw in scene_npc_tokens or normalized in scene_npc_tokens:
            continue
        if raw in scene_item_tokens or normalized in scene_item_tokens:
            continue
        if _is_location_reference_token(raw, location_aliases):
            continue
        if normalized and _is_location_reference_token(normalized, location_aliases):
            continue

        if _matches_generic_venue_request(
            original,
            normalized,
            setting_context,
            location_context_tokens,
            known_location_tokens,
        ):
            continue

        candidate_forms = {raw, normalized}
        matches_known = any(
            form in known_location_tokens for form in candidate_forms if form
        )
        if matches_known:
            continue

        inferred_kind = _guess_requested_kind(original, setting_context)
        decision = await _resolve_location_candidate(
            original,
            normalized,
            setting_context,
            resolver_cache,
            inferred_kind=inferred_kind,
        )

        if _is_plausible_location_token(original, resolver_cache):
            if decision.get("minted"):
                minted_name = decision.get("token") or original
                minted_key = decision.get("normalized") or normalized or raw
                identifiers = [
                    candidate
                    for candidate in {minted_name, minted_key}
                    if candidate
                ]

                if isinstance(setting_context, dict):
                    known_names = setting_context.setdefault("known_location_names", [])
                    for candidate_name in identifiers:
                        if candidate_name not in known_names:
                            known_names.append(candidate_name)

                for candidate_name in identifiers:
                    known_location_tokens.add(candidate_name)
                    if candidate_name not in minted_locations:
                        minted_locations.append(candidate_name)

                location_obj: Optional[Location] = None
                candidate_model: Optional[Candidate] = None

                if minted_name and user_id is not None and conversation_id is not None:
                    scope = _scope_from_resolver_decision(decision)
                    anchor = _anchor_from_setting_context(
                        setting_context,
                        scope=scope,
                        label=minted_name,
                    )
                    policy_meta = resolver_policy_for_context(setting_context, anchor=anchor)
                    candidate_model = _build_minted_candidate(
                        minted_name,
                        minted_key,
                        decision,
                        setting_context,
                        anchor=anchor,
                        scope=scope,
                    )

                    try:
                        async with get_db_connection_context() as conn:
                            location_obj = await get_or_create_location(
                                conn,
                                user_id=int(user_id),
                                conversation_id=int(conversation_id),
                                candidate=candidate_model,
                                scope=scope,
                                anchor=anchor,
                                mint_policy=policy_meta.get("mint_policy") if policy_meta else None,
                                default_planet=policy_meta.get("default_planet") if policy_meta else None,
                                default_galaxy=policy_meta.get("default_galaxy") if policy_meta else None,
                                default_realm=policy_meta.get("default_realm") if policy_meta else None,
                            )
                    except Exception:
                        logger.exception(
                            "Failed to mint location hierarchy",
                            extra={
                                "user_id": user_id,
                                "conversation_id": conversation_id,
                                "minted_name": minted_name,
                            },
                        )
                    else:
                        decision["location"] = location_obj
                        decision["candidate"] = candidate_model
                        if location_obj and location_obj.location_name:
                            normalized_token = location_obj.location_name
                            decision["normalized"] = normalized_token
                            if normalized_token not in minted_locations:
                                minted_locations.append(normalized_token)
                            known_location_tokens.add(normalized_token)
                            if isinstance(setting_context, dict):
                                known_names = setting_context.setdefault("known_location_names", [])
                                if normalized_token not in known_names:
                                    known_names.append(normalized_token)
                            resolver_cache[normalized_token] = decision
            continue

        unresolved_token = (
            decision.get("normalized")
            or normalized
            or raw
            or str(token)
        )
        unresolved.append(unresolved_token)

    return unresolved


def _matches_ambient_debris_token(token: str) -> bool:
    if not token:
        return False

    raw = str(token).strip().lower()
    normalized = _normalize_location_phrase(token)
    candidates: Set[str] = {raw, normalized}
    if normalized.endswith("s"):
        candidates.add(normalized[:-1])

    for candidate in list(candidates):
        if not candidate:
            continue
        cleaned = candidate.replace("-", " ")
        if cleaned in AMBIENT_DEBRIS_KEYWORDS:
            return True
        parts = cleaned.split()
        if any(part in AMBIENT_DEBRIS_KEYWORDS for part in parts):
            return True

    canonical = _canonicalize_mundane_search_token(token)
    if canonical and canonical in AMBIENT_DEBRIS_CANONICALS:
        return True

    return False


def _looks_like_sterile_environment(
    location_token: str, location_context_tokens: Set[str]
) -> bool:
    tokens: Set[str] = set(location_context_tokens or set())
    if location_token:
        tokens.add(location_token)

    for token in tokens:
        raw = str(token).strip().lower()
        normalized = _normalize_location_phrase(token)
        candidates = {raw, normalized}
        for candidate in list(candidates):
            if not candidate:
                continue
            cleaned = candidate.replace("-", " ")
            if cleaned in STERILE_ENVIRONMENT_HINTS:
                return True
            parts = cleaned.split()
            if any(part in STERILE_ENVIRONMENT_HINTS for part in parts):
                return True
            for hint in STERILE_ENVIRONMENT_HINTS:
                if " " in hint and hint in cleaned:
                    return True
    return False


def _is_plausible_mundane_search_target(
    token: str,
    text_l: str,
    categories: Set[str],
    location_token: str,
    location_context_tokens: Set[str],
) -> bool:
    canonical = _canonicalize_mundane_search_token(token)
    if not canonical:
        return False

    if not _has_search_signal(text_l, categories):
        return False

    context_blob = " ".join(sorted(location_context_tokens)).lower()
    if canonical == "coin":
        if location_token or context_blob:
            return True
        return False

    hints = MUNDANE_SEARCH_CONTEXT_HINTS.get(canonical, set())
    if any(hint in context_blob for hint in hints):
        return True

    if canonical in {"rock", "pebble"} and location_token:
        normalized_loc = _normalize_location_phrase(location_token)
        if any(
            keyword in normalized_loc
            for keyword in (
                "rock",
                "stone",
                "mountain",
                "cavern",
                "trail",
                "path",
                "shore",
                "beach",
                "forest",
                "field",
            )
        ):
            return True

    return False


def _is_location_reference_token(token: str, location_aliases: Set[str]) -> bool:
    normalized = _normalize_location_phrase(token)
    if not normalized:
        return False
    if normalized in location_aliases:
        return True
    raw = str(token).strip().lower()
    return raw in location_aliases


SETTING_KIND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "modern_realistic": {
        "type": "modern_realistic",
        "capabilities": {
            "magic": "none",
            "technology": "modern",
            "physics": "realistic",
            "supernatural": "none",
        },
        "technology_level": "modern",
        "setting_era": "contemporary",
    },
    "fantasy_epic": {
        "type": "high_fantasy",
        "capabilities": {
            "magic": "common",
            "technology": "medieval",
            "physics": "flexible",
            "supernatural": "known",
        },
        "technology_level": "medieval",
        "setting_era": "preindustrial",
    },
    "high_fantasy": {
        "type": "high_fantasy",
        "capabilities": {
            "magic": "common",
            "technology": "medieval",
            "physics": "flexible",
            "supernatural": "known",
        },
        "technology_level": "medieval",
        "setting_era": "preindustrial",
    },
    "science_fiction": {
        "type": "sci_fi_futuristic",
        "capabilities": {
            "magic": "none",
            "technology": "futuristic",
            "physics": "flexible",
            "supernatural": "none",
        },
        "technology_level": "futuristic",
        "setting_era": "far_future",
    },
    "soft_scifi": {
        "type": "soft_scifi",
        "capabilities": {
            "magic": "none",
            "technology": "advanced",
            "physics": "realistic",
            "supernatural": "none",
        },
        "technology_level": "near_future",
        "setting_era": "near_future",
    },
    "dystopian": {
        "type": "post_apocalyptic",
        "capabilities": {
            "magic": "none",
            "technology": "scavenged",
            "physics": "realistic",
            "supernatural": "hidden",
        },
        "technology_level": "scavenged",
        "setting_era": "near_future",
    },
    "modern_supernatural": {
        "type": "supernatural_modern",
        "capabilities": {
            "magic": "limited",
            "technology": "modern",
            "physics": "flexible",
            "supernatural": "known",
        },
        "technology_level": "modern",
        "setting_era": "contemporary",
    },
    "urban_fantasy": {
        "type": "urban_fantasy",
        "capabilities": {
            "magic": "subtle",
            "technology": "modern",
            "physics": "flexible",
            "supernatural": "hidden",
        },
        "technology_level": "modern",
        "setting_era": "contemporary",
    },
    "surrealist": {
        "type": "surrealist",
        "capabilities": {
            "magic": "subtle",
            "technology": "varies",
            "physics": "surreal",
            "supernatural": "common",
        },
        "technology_level": "varies",
        "setting_era": "timeless",
    },
}


def _get_setting_kind_defaults(setting_kind: Optional[str]) -> Optional[Dict[str, Any]]:
    if not setting_kind:
        return None
    normalized = str(setting_kind).strip().lower()
    defaults = SETTING_KIND_DEFAULTS.get(normalized)
    if not defaults:
        return None
    return deepcopy(defaults)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "none", "null", "no"}:
            return False
        return True
    return bool(value)


def _derive_feasibility_caps(
    infra: Optional[Dict[str, Any]],
    physics: Optional[Dict[str, Any]],
    economy: Optional[Dict[str, Any]],
    setting_era: Optional[str],
) -> Dict[str, bool]:
    infra = infra or {}
    physics = physics or {}
    economy = economy or {}
    era = (setting_era or "").strip().lower()

    trade_signals = [
        _boolish(infra.get("global_trade")),
        _boolish(infra.get("mass_packaging")),
        _boolish(infra.get("printing")),
        _boolish(infra.get("instant_communication")),
        _boolish(infra.get("markets_common")),
    ]
    caps = {
        "can_trade": any(trade_signals)
        or era in {"contemporary", "near_future", "far_future"},
        "has_currency": _boolish(economy.get("currency_enabled", True)),
        "retail_possible": True,
        "magic_allowed": _boolish(physics.get("magic_system"))
        or _boolish(physics.get("magic_allowed")),
        "teleport_allowed": _boolish(physics.get("teleportation_allowed"))
        or _boolish(physics.get("teleport_allowed")),
    }
    return caps


def _normalize_categories(categories: Optional[List[Any]]) -> Set[str]:
    normalized: Set[str] = set()
    for cat in categories or []:
        if cat is None:
            continue
        cat_str = str(cat).strip()
        if cat_str:
            normalized.add(cat_str.lower())
    return normalized


def _log_caps_snapshot(capabilities: Optional[Dict[str, Any]]) -> None:
    if not capabilities:
        return
    snapshot_keys = [
        "can_trade",
        "has_currency",
        "retail_possible",
        "magic_allowed",
        "teleport_allowed",
    ]
    parts = [
        f"{key}={_boolish(capabilities.get(key))}" for key in snapshot_keys
    ]
    logger.info("[FEASIBILITY] caps(%s)", ", ".join(parts))


def _fail_open_missing_caps(
    intents: List[Dict[str, Any]], capabilities: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if capabilities:
        return None

    if not intents:
        return {
            "overall": {"feasible": True, "strategy": "allow"},
            "per_intent": [],
        }

    normalized = [_normalize_categories(intent.get("categories")) for intent in intents]
    originals = [list(intent.get("categories", [])) for intent in intents]

    if all(cats.issubset(SAFE_BASELINE) for cats in normalized):
        logger.info(
            "[FEASIBILITY] No capability context; allowing safe-baseline intents."
        )
        per_intent = [
            {"feasible": True, "strategy": "allow", "categories": originals[idx]}
            for idx in range(len(intents))
        ]
        return {"overall": {"feasible": True, "strategy": "allow"}, "per_intent": per_intent}

    hazard_hits = [cats & IMPOSSIBLE_DEFAULT for cats in normalized]
    if any(hazard_hits):
        logger.info(
            "[FEASIBILITY] No capability context; hazardous categories detected: %s",
            [sorted(h) for h in hazard_hits if h],
        )
        per_intent = []
        for idx, hazard in enumerate(hazard_hits):
            if hazard:
                per_intent.append(
                    {
                        "feasible": False,
                        "strategy": "deny",
                        "violations": [
                            {
                                "rule": "established_impossibility",
                                "reason": "hazard_without_context",
                            }
                        ],
                        "categories": originals[idx],
                    }
                )
            else:
                per_intent.append(
                    {
                        "feasible": True,
                        "strategy": "allow",
                        "categories": originals[idx],
                    }
                )
        return {"overall": {"feasible": False, "strategy": "deny"}, "per_intent": per_intent}

    logger.info("[FEASIBILITY] No capability context; defaulting to allow.")
    per_intent = [
        {"feasible": True, "strategy": "allow", "categories": originals[idx]}
        for idx in range(len(intents))
    ]
    return {"overall": {"feasible": True, "strategy": "allow"}, "per_intent": per_intent}


def _tokenize_scene_values(values: Any) -> Set[str]:
    tokens: Set[str] = set()
    if values is None:
        return tokens
    if isinstance(values, dict):
        for key in ("name", "npc_name", "item_name", "title", "label", "display_name"):
            val = values.get(key)
            if val:
                tokens.add(str(val).strip().lower())
        for key in ("id", "npc_id", "item_id", "npcId", "itemId"):
            val = values.get(key)
            if val is not None:
                tokens.add(str(val).strip().lower())
        return tokens
    if isinstance(values, (list, tuple, set)):
        for entry in values:
            tokens |= _tokenize_scene_values(entry)
        return tokens
    token = str(values).strip().lower()
    if token:
        tokens.add(token)
    return tokens


def _display_scene_values(values: Any) -> List[str]:
    """Collect human-readable labels from scene structures (deduped, preserves order)."""

    collected: List[str] = []

    def _collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key in ("name", "npc_name", "item_name", "label", "title", "display_name"):
                val = value.get(key)
                if isinstance(val, str) and val.strip():
                    collected.append(val.strip())
                    return
            for key in ("id", "item_id", "npc_id", "npcId", "itemId"):
                val = value.get(key)
                if val is not None:
                    collected.append(str(val).strip())
                    return
            if value:
                collected.append(str(value))
            return
        if isinstance(value, (list, tuple, set)):
            for entry in value:
                _collect(entry)
            return
        text = str(value).strip()
        if text:
            collected.append(text)

    _collect(values)

    seen: Set[str] = set()
    ordered: List[str] = []
    for label in collected:
        lowered = label.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(label)
    return ordered


def load_world_caps(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compose world capabilities from the active archetypes."""

    archetype_caps = []
    for key in ctx.get("archetypes", []):
        archetype = ARCHETYPE_REGISTRY.get(key)
        if archetype:
            archetype_caps.append(archetype().caps())

    if not archetype_caps:
        archetype_caps = [ModernBaseline().caps()]

    return merge_caps(archetype_caps)


def _infer_world_archetypes(setting_context: Dict[str, Any]) -> List[str]:
    """Infer archetype mix from stored context details."""

    candidate_sources = [
        setting_context.get("world_archetypes"),
        setting_context.get("archetypes"),
        setting_context.get("setting_archetypes"),
    ]

    inferred: List[str] = []
    for value in candidate_sources:
        if isinstance(value, list):
            inferred.extend(str(v) for v in value)

    kind = str(setting_context.get("kind", "")).lower()
    setting_name = str(setting_context.get("setting_name", "")).lower()

    if "roman" in kind or "roman" in setting_name:
        inferred.append("roman_empire")
    if any(term in kind for term in ("underwater", "oceanic", "marine")) or "underwater" in setting_name:
        inferred.append("underwater_scifi")

    if not inferred:
        inferred.append("modern_baseline")

    # Preserve order while removing duplicates
    seen = set()
    ordered: List[str] = []
    for name in inferred:
        if name not in seen and name in ARCHETYPE_REGISTRY:
            ordered.append(name)
            seen.add(name)

    if not ordered:
        ordered.append("modern_baseline")

    return ordered


def _build_scene_snapshot(setting_context: Dict[str, Any], current_scene: Dict[str, Any]) -> Dict[str, Any]:
    """Translate stored scene information into the affordance schema."""

    location = setting_context.get("location") or {}
    tags = location.get("tags") if isinstance(location, dict) else []
    if isinstance(tags, str):
        tags = [tags]
    tags = tags or []

    location_features = current_scene.get("location_features") or []
    if isinstance(location_features, str):
        location_features = [location_features]

    has_vendor = bool(
        location.get("has_vendor")
        or current_scene.get("has_vendor")
        or any("vendor" in str(feature).lower() for feature in location_features)
        or any("shop" in str(tag).lower() for tag in tags)
    )

    nearby = current_scene.get("nearby")
    if not isinstance(nearby, dict):
        nearby = {}

    open_shop = current_scene.get("open_shop")
    if open_shop is None:
        open_shop = location.get("open_shop")

    return {
        "location_tags": tags,
        "has_vendor": has_vendor,
        "nearby": nearby,
        "open_shop": open_shop,
    }


def _build_player_snapshot(setting_context: Dict[str, Any]) -> Dict[str, Any]:
    """Collect light-weight player data for mundane checks."""

    state = setting_context.get("character_state") or {}
    currency: Dict[str, Any] = {}

    if isinstance(state, dict):
        for key in ("currency", "currency_balances", "balances", "wallet", "money"):
            value = state.get(key)
            if isinstance(value, dict):
                currency = value
                break

    inventory = setting_context.get("available_items", [])
    if not isinstance(inventory, list):
        inventory = []

    age = state.get("age") if isinstance(state, dict) else None
    legal_age = state.get("legal_age") if isinstance(state, dict) else None
    if isinstance(age, (int, float)) and isinstance(legal_age, (int, float)):
        age_ok = age >= legal_age
    elif isinstance(age, (int, float)):
        age_ok = age >= 18
    else:
        age_ok = True

    reputation = 0
    for key in ("reputation", "standing", "renown"):
        value = state.get(key) if isinstance(state, dict) else None
        if isinstance(value, (int, float)):
            reputation = value
            break

    return {
        "currency": currency,
        "inventory": inventory,
        "age_ok": age_ok,
        "reputation": reputation,
    }


def _mundane_result_to_intent_entry(intent: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a mundane evaluation response into the feasibility schema."""

    entry = {
        "feasible": evaluation.get("feasible", False),
        "strategy": evaluation.get("strategy", "defer"),
        "categories": intent.get("categories", []),
    }

    for key in ("narrator_guidance", "leads", "violations", "modifications"):
        if key in evaluation and evaluation[key]:
            entry[key] = evaluation[key]

    return entry


def _combine_overall(per_intent: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive an overall feasibility verdict from per-intent entries."""

    if not per_intent:
        return {"feasible": True, "strategy": "allow"}

    lower_strategies = [str(item.get("strategy", "")).lower() for item in per_intent]
    if any(strategy == "deny" for strategy in lower_strategies):
        return {"feasible": False, "strategy": "deny"}

    if any(item.get("feasible") and strategy == "allow" for item, strategy in zip(per_intent, lower_strategies)):
        return {"feasible": True, "strategy": "allow"}

    if "analog" in lower_strategies:
        return {"feasible": False, "strategy": "analog"}

    if "defer" in lower_strategies:
        return {"feasible": False, "strategy": "defer"}

    return {"feasible": True, "strategy": lower_strategies[0] if lower_strategies else "allow"}


def _apply_mundane_overrides(
    base_result: Dict[str, Any], intents: List[Dict[str, Any]], overrides: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge mundane evaluations with an existing feasibility response."""

    per_intent = base_result.get("per_intent") or []
    if len(per_intent) != len(intents):
        per_intent = [
            {
                "feasible": True,
                "strategy": "allow",
                "categories": intents[i].get("categories", []),
            }
            for i in range(len(intents))
        ]

    for index, override in overrides.items():
        per_intent[index] = override

    base_result["per_intent"] = per_intent
    base_result["overall"] = _combine_overall(per_intent)
    return base_result


async def _evaluate_mundane_actions(
    nyx_ctx: NyxContext, setting_context: Dict[str, Any], intents: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Evaluate mundane intents using the archetype-driven affordance engine when available,
    with a hardened, scene-aware heuristic fallback that never raises.

    Returns None when there are no mundane/trade intents.
    """

    # ---- 0) Which intents are mundane? ----------------------------------------
    indices: List[int] = []
    for idx, intent in enumerate(intents or []):
        categories = {str(cat).lower() for cat in (intent.get("categories") or [])}
        if "mundane_action" in categories or "trade" in categories:
            indices.append(idx)

    if not indices:
        return None

    # ---- 1) Gather scene context (safe) ---------------------------------------
    try:
        current_scene = await _load_current_scene(nyx_ctx)
    except Exception:
        current_scene = {}

    # Normalize scene slices
    def _as_str_list(values: Any) -> List[str]:
        out: List[str] = []
        if values is None:
            return out
        if isinstance(values, (list, tuple, set)):
            for v in values:
                if isinstance(v, dict):
                    for key in ("name", "item_name", "npc_name", "label", "title", "display_name"):
                        val = v.get(key)
                        if isinstance(val, str) and val.strip():
                            out.append(val.strip())
                            break
                    else:
                        out.append(str(v))
                elif v is not None:
                    out.append(str(v))
        elif isinstance(values, dict):
            name = values.get("name") or values.get("display_name") or values.get("label")
            out.append(str(name) if name else str(values))
        else:
            out.append(str(values))
        return [s for s in (x.strip() for x in out) if s]

    scene_npcs = _as_str_list(current_scene.get("npcs") or setting_context.get("present_entities") or [])
    scene_items = _as_str_list(current_scene.get("items") or setting_context.get("available_items") or [])
    scene_feats = _as_str_list(current_scene.get("location_features") or setting_context.get("location_features") or [])
    time_phase_value = (current_scene.get("time_phase") or setting_context.get("current_time") or "day")
    time_phase = str(time_phase_value).lower() if isinstance(time_phase_value, str) else "day"

    location_payload = setting_context.get("location") or {}
    if not isinstance(location_payload, dict):
        location_payload = {"name": str(location_payload)}
    location_name = location_payload.get("name") or location_payload.get("display_name") or location_payload.get("label")

    def _tokens_of(*values: Any) -> Set[str]:
        tokens: Set[str] = set()
        for coll in values:
            for s in _as_str_list(coll):
                s_l = s.lower()
                tokens.add(s_l)
                for part in s_l.replace("/", " ").replace("-", " ").split():
                    if part:
                        tokens.add(part)
        return tokens

    scene_tokens = _tokens_of(
        current_scene.get("npcs"),
        current_scene.get("items"),
        current_scene.get("location_features"),
        [location_name] if location_name else []
    )

    # Vendor detection for "trade" intents
    _VENDOR_TOKENS = {
        "merchant", "trader", "shopkeeper", "clerk", "cashier", "vendor", "barkeep",
        "bartender", "innkeeper", "grocer", "dealer", "pharmacist", "seller",
        "market", "bazaar", "stall", "booth", "kiosk", "counter", "register",
        "shop", "store", "tavern", "bar", "inn"
    }
    vendor_present = bool(scene_tokens & _VENDOR_TOKENS)

    # Ambient debris allowed unless the environment looks sterile
    sterile_hints = {h.lower() for h in (STERILE_ENVIRONMENT_HINTS or set())}
    looks_sterile = bool(scene_tokens & sterile_hints)

    def _alts() -> List[str]:
        alts: List[str] = []
        if scene_items:
            alts.append(f"examine the {scene_items[0]} more closely")
        if scene_feats:
            alts.append(f"investigate the {scene_feats[0]}")
        if scene_npcs:
            alts.append(f"approach {scene_npcs[0]} for help")
        if time_phase == "night":
            alts.append("wait until dawn for better visibility")
        return (alts[:3] or ["try a simpler, grounded action that uses something visible in the scene"])

    # ---- 2) Try the affordance engine if it's wired in ------------------------
    def _fallback_scene_snapshot() -> Dict[str, Any]:
        return {
            "location": {"name": location_name} if location_name else {},
            "npcs": scene_npcs,
            "items": scene_items,
            "features": scene_feats,
            "time_phase": time_phase,
        }

    def _fallback_player_snapshot() -> Dict[str, Any]:
        return {
            "abilities": setting_context.get("character_abilities") or [],
            "stats": setting_context.get("character_state") or {},
            "inventory": _as_str_list(setting_context.get("available_items") or []),
        }

    def _fallback_infer_archetypes() -> List[str]:
        kind = str(setting_context.get("kind") or setting_context.get("setting_kind") or "").lower()
        if "fantasy" in kind:
            return ["fantasy_grounded"]
        if "cyber" in kind or "sci" in kind:
            return ["technoir"]
        return ["realistic_modern"]

    def _combine_overall_local(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not entries:
            return {"feasible": True, "strategy": "allow"}
        if any((e or {}).get("strategy") == "deny" for e in entries):
            return {"feasible": False, "strategy": "deny"}
        if any((e or {}).get("strategy") == "defer" for e in entries):
            return {"feasible": False, "strategy": "defer"}
        if any((e or {}).get("strategy") == "ask" for e in entries):
            return {"feasible": False, "strategy": "ask"}
        return {"feasible": True, "strategy": "allow"}

    try:
        scene_snapshot = _build_scene_snapshot(setting_context, current_scene)  # type: ignore[name-defined]
    except Exception:
        scene_snapshot = _fallback_scene_snapshot()

    try:
        player_snapshot = _build_player_snapshot(setting_context)  # type: ignore[name-defined]
    except Exception:
        player_snapshot = _fallback_player_snapshot()

    try:
        archetypes = _infer_world_archetypes(setting_context)  # type: ignore[name-defined]
    except Exception:
        archetypes = _fallback_infer_archetypes()

    overrides: Dict[int, Dict[str, Any]] = {}
    engine_ok = True
    try:
        world_caps = load_world_caps({"archetypes": archetypes})  # type: ignore[name-defined]
        affordance_index = build_affordance_index(  # type: ignore[name-defined]
            world_caps, scene_snapshot, player_snapshot
        )
    except Exception:
        engine_ok = False

    if engine_ok:
        try:
            for idx in indices:
                try:
                    evaluation = evaluate_mundane(  # type: ignore[name-defined]
                        intents[idx], world_caps, affordance_index, scene_snapshot, player_snapshot
                    )
                    try:
                        entry = _mundane_result_to_intent_entry(intents[idx], evaluation)  # type: ignore[name-defined]
                    except Exception:
                        feasible = bool((evaluation or {}).get("feasible", True))
                        entry = {
                            "feasible": feasible,
                            "strategy": "allow" if feasible else "deny",
                            "categories": intents[idx].get("categories", []),
                            "violations": (evaluation or {}).get("violations", []),
                        }
                    overrides[idx] = entry
                except Exception:
                    overrides[idx] = {"__needs_heuristic__": True}
        except Exception:
            overrides.clear()
            engine_ok = False

    async def _heuristic_entry(intent: Dict[str, Any]) -> Dict[str, Any]:
        cats = {str(c).lower() for c in (intent.get("categories") or [])}

        # Respect common prerequisite logic
        prereq_ok, prereq_reason = await _check_prerequisites(intent, setting_context)
        if not prereq_ok:
            strategy = "defer" if ({"trade", "mundane_action"} & cats) else "deny"
            return {
                "feasible": False,
                "strategy": strategy,
                "violations": [{"rule": "missing_prereq", "reason": prereq_reason or "Required elements are not present"}],
                "narrator_guidance": prereq_reason or "Required elements are not present right now.",
                "suggested_alternatives": _alts(),
                "categories": sorted(cats),
            }

        # Trade requires a vendor-ish presence
        if "trade" in cats:
            if vendor_present:
                return {"feasible": True, "strategy": "allow", "categories": sorted(cats)}
            return {
                "feasible": False,
                "strategy": "defer",
                "violations": [{"rule": "no_vendor_here", "reason": "No vendor/point-of-sale in the current scene"}],
                "narrator_guidance": "No vendor here. Try heading to the market, counter, or corner shop nearby.",
                "suggested_alternatives": _alts(),
                "categories": sorted(cats),
            }

        # Mundane instrument: ambient debris OK unless sterile
        instruments = intent.get("instruments")
        if instruments:
            inst_tokens = _tokens_of(instruments)
            ambient_hit = bool(inst_tokens & (AMBIENT_DEBRIS_KEYWORDS | AMBIENT_DEBRIS_CANONICALS))
            if ambient_hit and not looks_sterile:
                return {"feasible": True, "strategy": "allow", "categories": sorted(cats)}

        # Default: allow mundane if prerequisites passed
        return {"feasible": True, "strategy": "allow", "categories": sorted(cats)}

    if not engine_ok:
        for idx in indices:
            overrides[idx] = await _heuristic_entry(intents[idx])
    else:
        for idx in indices:
            if overrides.get(idx, {}).get("__needs_heuristic__"):
                overrides[idx] = await _heuristic_entry(intents[idx])

    # ---- 4) Build overall & return -------------------------------------------
    entries = list(overrides.values())
    try:
        overall = _combine_overall(entries)  # type: ignore[name-defined]
    except Exception:
        overall = _fallback_scene_snapshot  # type: ignore[assignment]
        overall = {"feasible": True, "strategy": "allow"} if not entries else (
            {"feasible": False, "strategy": "deny"} if any((e or {}).get("strategy") == "deny" for e in entries)
            else {"feasible": False, "strategy": "defer"} if any((e or {}).get("strategy") == "defer" for e in entries)
            else {"feasible": False, "strategy": "ask"} if any((e or {}).get("strategy") == "ask" for e in entries)
            else {"feasible": True, "strategy": "allow"}
        )

    return {
        "overrides": overrides,
        "overall": overall,
        "all_handled": len(indices) == len(intents),
    }

# Dynamic rejection narrator for unique, contextual rejections
REJECTION_NARRATOR_AGENT = Agent(
    name="RejectionNarrator",
    instructions="""
    You are the voice of reality itself, explaining why certain actions cannot occur.
    
    Given the setting context and attempted action, create a UNIQUE, immersive rejection that:
    1. Fits the setting's tone and genre perfectly
    2. Never repeats previous rejections (check rejection_history)
    3. Feels like the world itself is responding, not a game system
    4. Maintains the narrative flow without breaking immersion
    5. Subtly guides toward what IS possible
    
    Consider:
    - Setting atmosphere and mood from environment_desc
    - The specific physics/magic rules of this world
    - Recent narrative events for continuity
    - The exact nature of what was attempted
    - Previous rejections to avoid repetition
    - Current scene details (NPCs, items, location)
    
    Generate THREE elements:
    1. reality_response: A visceral, sensory description of reality resisting (1-2 sentences)
    2. narrator_guidance: Poetic explanation of why this cannot be (2-3 sentences)  
    3. suggested_alternatives: 2-3 contextual alternatives that fit the current scene
    4. metaphor: A unique metaphor for this specific rejection
    
    Make each rejection feel unique to THIS moment in THIS world.
    
    Output JSON:
    {
        "reality_response": "...",
        "narrator_guidance": "...",
        "suggested_alternatives": ["...", "...", "..."],
        "metaphor": "..."
    }
    """,
    model="gpt-5-nano"
)

# Alternative suggestion generator
ALTERNATIVE_GENERATOR_AGENT = Agent(
    name="AlternativeGenerator",
    instructions="""
    Given what the player tried to do and what's actually available in the scene,
    suggest 3 creative alternatives that:
    1. Achieve a similar narrative goal
    2. Use only what's present in the scene
    3. Fit the setting's capabilities
    4. Feel like natural choices, not system suggestions
    5. Vary in approach (physical, social, environmental)
    
    Make suggestions feel organic and enticing, not like consolation prizes.
    Each alternative should be specific to the current moment and scene.
    
    Output JSON array of strings: ["alternative 1", "alternative 2", "alternative 3"]
    """,
    model="gpt-5-nano"
)

# Enhanced feasibility agent with dynamic reasoning
FEASIBILITY_AGENT = Agent(
    name="FeasibilityChecker",
    instructions="""
    You are a reality consistency enforcer. Analyze if actions are possible given the setting's established rules.

    CRITICAL: You must maintain internal consistency. What's been established as impossible STAYS impossible.

    For each intent, consider:
    1. Does this violate established physics/reality rules of THIS setting?
    2. Has this type of action been previously established as possible/impossible?
    3. Does the player have the means/ability to perform this action?
    4. Is the target present and accessible in the current scene?
    5. Would this break narrative consistency?
    6. Does the player's current state allow this action?
    7. The setting kind (realistic, fantasy, sci-fi, etc.) and its capabilities
    8. Hard rules that MUST be enforced vs soft rules that guide behavior
    
    Consider the dynamic context provided, including:
    - Setting capabilities and limitations
    - Current scene state and available elements
    - Previously established possibilities/impossibilities
    - The specific nature and context of this world

    If the user makes an out-of-character or real-world request (asking about apps, coupons,
    store hours, modern logistics, or anything outside the fiction), mark it infeasible and
    include a violation object exactly as {"rule":"policy:roleplay_only","reason":"Out-of-game request"}.

    BE STRICT but CONTEXTUAL. Enforce the world's unique logic.

    Output ONLY JSON:
      {"overall":{"feasible":bool,"strategy":"allow|deny|reinterpret"},
       "per_intent":[
         {"feasible":bool,"strategy":"allow|deny|reinterpret",
          "violations":[{"rule":"...", "reason":"..."}],
          "categories":["..."]}
       ]}
    """,
    model="gpt-5-nano"
)

# Setting detective agent for auto-detecting setting type
SETTING_DETECTIVE_AGENT = Agent(
    name="SettingDetective",
    instructions="""
    Analyze the established narrative elements to determine the setting type and capabilities.
    
    Consider:
    - Technology level (medieval, modern, futuristic, etc.)
    - Presence of magic or supernatural elements
    - Physics model (realistic, soft sci-fi, fantasy, surreal)
    - Genre markers (noir, cyberpunk, high fantasy, etc.)
    - Established world rules and limitations
    - Environmental descriptions and atmosphere
    
    Determine:
    1. Setting type (e.g., "realistic_modern", "high_fantasy", "cyberpunk")
    2. Setting kind (broader category)
    3. Key capabilities (what's possible in this world)
    4. Confidence level
    
    Output JSON:
    {
        "setting_type": "...",
        "setting_kind": "...",
        "confidence": 0.X,
        "indicators": ["...", "..."],
        "capabilities": {
            "magic": "none|limited|common|ubiquitous",
            "technology": "primitive|medieval|modern|advanced|futuristic",
            "physics": "realistic|flexible|surreal",
            "supernatural": "none|hidden|known|common"
        },
        "details": "..."
    }
    """,
    model="gpt-5-nano"
)

# Optional helpers (kept soft — only used if present in your codebase)
try:
    from .helpers import _infer_categories_from_text, _scene_alternatives, _compose_guidance
except Exception:
    def _infer_categories_from_text(text_l: str) -> Set[str]:
        # ultra-light fallback; intentionally weak (keeps system dynamic)
        hits = set()
        if any(k in text_l for k in ("summon", "conjure", "spawn", "manifest", "materialize")):
            hits.add("ex_nihilo_conjuration")
        if any(k in text_l for k in ("fly", "levitate", "hover")):
            hits.add("physics_violation")
        if any(k in text_l for k in ("spaceship", "laser", "plasma", "warp")):
            hits.add("scifi_setpiece")
        if any(k in text_l for k in ("hack drone", "access ai", "drone")):
            hits.add("ai_system_access")
        return hits

    def _scene_alternatives(npcs, items, features, time_phase) -> List[str]:
        alts = []
        if items:
            alts.append(f"use the {items[0]}")
        if features:
            alts.append(f"interact with {features[0]}")
        if npcs:
            alts.append(f"ask {npcs[0].get('name','someone')} for help")
        if not alts:
            alts.append("try a simpler, grounded action that uses something visible in the scene")
        return alts

    def _compose_guidance(setting_kind: str, location_name: Optional[str], blocking: Set[str]) -> str:
        loc = f" in {location_name}" if location_name else ""
        cats = ", ".join(sorted(blocking))
        return f"Reality{loc} doesn’t support that ({cats}). Try something that fits what’s actually present."

async def assess_action_feasibility(nyx_ctx: NyxContext, user_input: str) -> Dict[str, Any]:
    """
    Dynamically assess if an action is feasible in the current setting context.
    Generates unique, contextual responses for every rejection.
    """

    policy_sources: List[Any] = []

    def _extend_policy_sources_from(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        for key in (
            "policy_flags",
            "policies",
            "policy",
            "flags",
            "feature_flags",
            "settings",
            "meta",
            "governance",
            "mode_policy",
            "mode",
        ):
            value = candidate.get(key)
            if value is not None:
                policy_sources.append(value)

    for candidate in (
        getattr(nyx_ctx, "current_context", None),
        getattr(nyx_ctx, "last_packed_context", None),
        getattr(nyx_ctx, "config", None),
        getattr(nyx_ctx, "policy_flags", None),
    ):
        if candidate is not None:
            policy_sources.append(candidate)
            if isinstance(candidate, dict):
                _extend_policy_sources_from(candidate)

    enforced_response_meta = {
        "response_mode": "diegetic",
        "response_mode_reason": "roleplay_only_policy",
    }

    minimal_context: Optional[Dict[str, Any]] = None

    def _stamp_response_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload, dict):
            meta = payload.setdefault("meta", {})
            meta.update(enforced_response_meta)
        return payload

    def _finalize_result(raw_payload: Any) -> Dict[str, Any]:
        payload_dict: Dict[str, Any]
        if isinstance(raw_payload, dict):
            payload_dict = deepcopy(raw_payload)
        else:
            payload_dict = {"meta": {}, "raw": raw_payload}

        _stamp_response_meta(payload_dict)

        sources = list(policy_sources)
        if minimal_context:
            sources.append(minimal_context)

        hardened = _harden_output_against_ooc_leakage(
            payload_dict,
            request_text=user_input,
            policy_sources=sources,
        )
        final_payload = hardened.get("payload", payload_dict)
        if isinstance(final_payload, dict):
            _stamp_response_meta(final_payload)
            return final_payload
        return payload_dict

    if _is_ooc_request(user_input) and _policy_roleplay_only_enabled(*policy_sources):
        return _finalize_result(_nyx_meta_decline_payload("ooc_request"))

    # Parse the intended actions only after early OOC guardrails
    intents = await parse_action_intents(user_input)
    text_l = (user_input or "").lower()

    if _is_ooc_request(user_input):
        minimal_context = await _load_minimal_context(nyx_ctx)
        policy_candidates = list(policy_sources)
        if minimal_context:
            policy_sources.append(minimal_context)
            _extend_policy_sources_from(minimal_context)
            policy_candidates.append(minimal_context)
        if _policy_roleplay_only_enabled(*policy_candidates):
            return _finalize_result(_nyx_meta_decline_payload("ooc_request"))

    # Load comprehensive setting context
    setting_context = await _load_comprehensive_context(nyx_ctx)
    _log_caps_snapshot(setting_context.get("capabilities"))

    policy_sources.append(setting_context)
    if isinstance(setting_context, dict):
        _extend_policy_sources_from(setting_context)
        setting_meta = setting_context.setdefault("meta", {})
        setting_meta["response_mode"] = enforced_response_meta["response_mode"]
        setting_meta["response_mode_reason"] = enforced_response_meta["response_mode_reason"]

    policy_candidates = list(policy_sources)
    if minimal_context and minimal_context not in policy_candidates:
        policy_candidates.append(minimal_context)
    if _is_ooc_request(user_input) and _policy_roleplay_only_enabled(*policy_candidates):
        return _finalize_result(_nyx_meta_decline_payload("ooc_request"))

    caps_loaded = bool(setting_context.get("caps_loaded"))
    fail_open = _fail_open_missing_caps(
        intents,
        setting_context.get("capabilities") if caps_loaded else {},
    )
    if fail_open:
        return _finalize_result(fail_open)

    impossible_logged: Set[str] = set()
    for imp in setting_context.get("established_impossibilities", []):
        try:
            categories = set((imp or {}).get("categories", []) or [])
        except Exception:
            categories = set()
        impossible_logged |= {str(cat) for cat in categories if cat}
    if impossible_logged:
        logger.info(
            "[FEASIBILITY] impossible_categories=%s",
            sorted(impossible_logged),
        )

    # Early guard: fabricated location changes
    scene = setting_context.get("scene") if isinstance(setting_context.get("scene"), dict) else {}
    location_payload = setting_context.get("location") or {}
    location_name: Optional[str]
    if isinstance(location_payload, dict):
        location_name = (
            location_payload.get("name")
            or location_payload.get("label")
            or location_payload.get("display_name")
            or location_payload.get("location_name")
        )
    else:
        location_name = str(location_payload).strip() if location_payload else None

    scene_npcs = setting_context.get("present_entities") or []
    scene_items = setting_context.get("available_items") or []
    location_features = setting_context.get("location_features") or []
    if isinstance(location_features, str):
        location_features = [location_features]
    time_phase_value = setting_context.get("current_time") or "day"
    time_phase = str(time_phase_value).lower() if isinstance(time_phase_value, str) else "day"

    location_token = str(location_name).strip().lower() if location_name else ""
    location_aliases = _location_reference_aliases(location_token, scene)
    location_context_tokens = _collect_location_context_tokens(
        scene,
        location_features,
        location_token,
    )
    known_location_tokens = _build_known_location_tokens(
        location_aliases,
        location_context_tokens,
        setting_context.get("known_location_names") or [],
        scene,
    )
    scene_npc_tokens = _tokenize_scene_values(scene_npcs)
    scene_item_tokens = _tokenize_scene_values(scene_items)

    intents_sequence = intents or [{}]
    inferred_categories: List[Set[str]] = []
    location_blocks: Dict[int, Dict[str, Any]] = {}

    for idx, intent in enumerate(intents_sequence):
        cats = set(intent.get("categories") or [])
        if not cats:
            inferred = _infer_categories_from_text(text_l)
            if inferred:
                cats = inferred
        inferred_categories.append(cats)

        missing_location_tokens = await _find_unresolved_location_targets(
            intent,
            text_l,
            location_aliases,
            location_context_tokens,
            known_location_tokens,
            scene_npc_tokens,
            scene_item_tokens,
            setting_context,
            user_id=nyx_ctx.user_id,
            conversation_id=nyx_ctx.conversation_id,
        )
        if not missing_location_tokens:
            continue

        missing_location_phrase = _format_missing_names(missing_location_tokens)
        resolver_cache = (
            setting_context.get("location_resolver_cache", {})
            if isinstance(setting_context, dict)
            else {}
        )
        resolver_decisions = [
            _resolver_feedback_for_token(token, resolver_cache)
            for token in missing_location_tokens
        ]
        deny_decision = next(
            (d for d in resolver_decisions if d and d.get("decision") == "deny"),
            None,
        )
        ask_decision = next(
            (d for d in resolver_decisions if d and d.get("decision") == "ask"),
            None,
        )

        if deny_decision:
            decision_strategy = "deny"
            violation_reason = (
                deny_decision.get("reason")
                or f"{missing_location_phrase} isn't an established location right now."
            )
        elif ask_decision:
            decision_strategy = "ask"
            violation_reason = ask_decision.get("reason") or (
                f"{missing_location_phrase} needs a quick description before I can add it."
            )
        else:
            decision_strategy = "deny"
            violation_reason = (
                f"{missing_location_phrase} isn't an established location right now."
            )
        lead_candidates = _scene_alternatives(
            _display_scene_values(scene_npcs),
            _display_scene_values(scene_items),
            _display_scene_values(location_features),
            time_phase,
        )
        if decision_strategy == "ask":
            logger.info(
                "[FEASIBILITY] Resolver ASK for location -> %s",
                missing_location_tokens,
            )
        else:
            logger.info(
                "[FEASIBILITY] Hard deny - fabricated location -> %s",
                missing_location_tokens,
            )
        location_blocks[idx] = {
            "feasible": False,
            "strategy": decision_strategy,
            "violations": [
                {
                    "rule": f"location_resolver:{decision_strategy}",
                    "reason": violation_reason,
                }
            ],
            "narrator_guidance": (
                f"{violation_reason} Give me a quick sense of the place or pick one of the known options."
                if decision_strategy == "ask"
                else f"{violation_reason} Stick to known locations or work with the narrator to introduce it first."
            ),
            "suggested_alternatives": lead_candidates,
            "leads": lead_candidates,
            "categories": sorted(cats),
        }

    if location_blocks:
        per_intent = []
        has_ask = False
        has_deny = False
        for idx, cats in enumerate(inferred_categories):
            if idx in location_blocks:
                block = location_blocks[idx]
                per_intent.append(block)
                strategy = (block or {}).get("strategy")
                if strategy == "deny":
                    has_deny = True
                elif strategy == "ask":
                    has_ask = True
            else:
                per_intent.append(
                    {
                        "feasible": True,
                        "strategy": "allow",
                        "categories": sorted(cats),
                    }
                )

        overall_strategy = "deny" if has_deny else ("ask" if has_ask else "deny")
        return _finalize_result(
            {
                "overall": {"feasible": False, "strategy": overall_strategy},
                "per_intent": per_intent,
            }
        )

    # Quick check against hard rules
    quick_check = await _quick_feasibility_check(setting_context, intents)

    violations = quick_check.get("violations", [])
    missing_only_indices: List[int] = []
    for idx, violation_list in enumerate(violations):
        if not violation_list:
            continue
        has_missing = any(v.get("rule") == "missing_prereq" for v in violation_list)
        if not has_missing:
            continue
        if all(v.get("rule") == "missing_prereq" for v in violation_list):
            missing_only_indices.append(idx)
        else:
            missing_only_indices = []
            break

    if missing_only_indices:
        logger.info(
            "[FEASIBILITY] Deferring due to missing prerequisites for intents %s",
            missing_only_indices,
        )
        per_intent = []
        for idx, intent in enumerate(intents):
            cats = intent.get("categories", [])
            if idx in missing_only_indices:
                violation_entry = violations[idx] if idx < len(violations) else []
                categories_norm = _normalize_categories(cats)
                guidance = "No vendor here. Try heading to the market or corner shop nearby."
                if not ({"trade", "mundane_action"} & categories_norm):
                    fallback_reason = "Required elements are not present right now."
                    if violation_entry:
                        fallback_reason = violation_entry[0].get("reason", fallback_reason)
                    guidance = fallback_reason
                per_intent.append(
                    {
                        "feasible": False,
                        "strategy": "defer",
                        "narrator_guidance": guidance,
                        "violations": violation_entry,
                        "categories": cats,
                    }
                )
            else:
                per_intent.append(
                    {
                        "feasible": True,
                        "strategy": "allow",
                        "categories": cats,
                    }
                )

        return _finalize_result(
            {"overall": {"feasible": False, "strategy": "defer"}, "per_intent": per_intent}
        )

    if quick_check.get("hard_blocked"):
        # Generate dynamic, unique rejections for blocked actions
        per_intent = []
        for i, intent in enumerate(intents):
            if not quick_check["intent_feasible"][i]:
                # Generate completely unique rejection
                rejection = await generate_dynamic_rejection(
                    setting_context, 
                    {**intent, "raw_text": user_input, "violations": quick_check["violations"][i]},
                    nyx_ctx
                )
                
                per_intent.append({
                    "feasible": False,
                    "strategy": "deny",
                    "violations": quick_check["violations"][i],
                    "reality_response": rejection["reality_response"],
                    "narrator_guidance": rejection["narrator_guidance"],
                    "suggested_alternatives": rejection["suggested_alternatives"],
                    "metaphor": rejection.get("metaphor", ""),
                    "categories": intent.get("categories", [])
                })
            else:
                per_intent.append({
                    "feasible": True,
                    "strategy": "allow",
                    "categories": intent.get("categories", [])
                })
        
        return _finalize_result(
            {
                "overall": {"feasible": False, "strategy": "deny"},
                "per_intent": per_intent,
            }
        )

    mundane_eval = await _evaluate_mundane_actions(nyx_ctx, setting_context, intents)

    if mundane_eval and mundane_eval.get("all_handled"):
        per_intent = [
            mundane_eval["overrides"].get(i, {
                "feasible": True,
                "strategy": "allow",
                "categories": intents[i].get("categories", []),
            })
            for i in range(len(intents))
        ]

        return _finalize_result(
            {
                "overall": mundane_eval.get("overall", _combine_overall(per_intent)),
                "per_intent": per_intent,
            }
        )

    # Full AI-powered assessment for nuanced cases
    full_result = await _full_dynamic_assessment(nyx_ctx, user_input, intents, setting_context)

    if mundane_eval and mundane_eval.get("overrides"):
        full_result = _apply_mundane_overrides(full_result, intents, mundane_eval["overrides"])

    return _finalize_result(full_result)


async def _load_minimal_context(nyx_ctx: NyxContext) -> Dict[str, Any]:
    """Load a minimal slice of context for policy enforcement checks."""

    existing = getattr(nyx_ctx, "current_context", None)
    if isinstance(existing, dict) and existing:
        return existing

    try:
        minimal = await fallback_get_context(nyx_ctx.user_id, nyx_ctx.conversation_id)
    except Exception:
        logger.warning(
            "Failed to load minimal context for policy enforcement user_id=%s conversation_id=%s",
            nyx_ctx.user_id,
            nyx_ctx.conversation_id,
            exc_info=True,
        )
        return {}

    if isinstance(minimal, dict):
        return minimal

    return {}


async def _load_comprehensive_context(nyx_ctx: NyxContext) -> Dict[str, Any]:
    """Load all relevant context about what's possible in this setting"""
    
    context = {
        "type": "unknown",
        "kind": "modern_realistic",
        "capabilities": {},
        "reality_context": "normal",
        "established_rules": [],
        "hard_rules": [],
        "soft_rules": [],
        "available_items": [],
        "present_entities": [],
        "character_abilities": [],
        "character_state": {},
        "physics_model": "realistic",
        "magic_system": None,
        "technology_level": "contemporary",
        "setting_era": "contemporary",
        "caps_loaded": False,
        "infrastructure_flags": {},
        "economy_flags": {},
        "physics_caps": {},
        "location": {},
        "location_object": None,
        "_location_object": None,
        "location_features": [],
        "scene": {},
        "established_impossibilities": [],
        "established_possibilities": [],
        "narrative_history": [],
        "environment_desc": "",
        "setting_name": "",
        "stat_modifiers": {},
        "known_location_names": [],
        "world_model": {},
        "mode": None,
        "mode_policy": None,
    }
    
    async with get_db_connection_context() as conn:
        # Get comprehensive setting information from new_game_agent storage
        setting_keys = [
            'WorldType', 'SettingType', 'SettingKind', 'SettingCapabilities',
            'RealityContext', 'PhysicsModel', 'PhysicsCaps', 'EnvironmentDesc',
            'CurrentSetting', 'SettingStatModifiers', 'EnvironmentHistory',
            'ScenarioName', 'CurrentLocation', 'CurrentTime', 'InfrastructureFlags',
            'EconomyFlags', 'TechnologyLevel', 'SettingEra',
            'WorldModel', 'RoleplayMode', 'ModePolicy'
        ]
        
        setting_data = await conn.fetch("""
            SELECT key, value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 
            AND key = ANY($3)
        """, nyx_ctx.user_id, nyx_ctx.conversation_id, setting_keys)
        
        infra_flags: Dict[str, Any] = {}
        economy_flags: Dict[str, Any] = {}
        physics_caps_local: Dict[str, Any] = {}
        world_type_value: Optional[str] = None
        world_model_raw: Optional[Dict[str, Any]] = None
        technology_level = context.get("technology_level")
        setting_era = context.get("setting_era")
        technology_level_from_db = False
        setting_era_from_db = False

        current_location_name: Optional[str] = None

        for row in setting_data:
            key = row['key']
            value = row['value']

            if key == 'WorldType':
                world_type_value = value
            elif key == 'SettingType':
                context["type"] = value
            elif key == 'SettingKind':
                context["kind"] = value
            elif key == 'SettingCapabilities':
                try:
                    capabilities_from_db = json.loads(value)
                except Exception:
                    capabilities_from_db = None

                if isinstance(capabilities_from_db, dict):
                    context["capabilities"] = capabilities_from_db
                    if "magic" in capabilities_from_db:
                        context["magic_system"] = capabilities_from_db["magic"]
                    if "technology" in capabilities_from_db:
                        technology_level = capabilities_from_db["technology"]
                        context["technology_level"] = technology_level
                        technology_level_from_db = True
                    if "era" in capabilities_from_db:
                        setting_era = capabilities_from_db["era"]
                        setting_era_from_db = True
            elif key == 'RealityContext':
                context["reality_context"] = value
            elif key == 'PhysicsModel':
                context["physics_model"] = value
            elif key == 'PhysicsCaps':
                try:
                    physics_caps_local = json.loads(value)
                except Exception:
                    physics_caps_local = {}
            elif key == 'EnvironmentDesc':
                context["environment_desc"] = value
            elif key == 'CurrentSetting':
                context["setting_name"] = value
            elif key == 'SettingStatModifiers':
                try:
                    context["stat_modifiers"] = json.loads(value)
                except:
                    pass
            elif key == 'EnvironmentHistory':
                context["environment_history"] = value
            elif key == 'CurrentLocation':
                context["location"]["name"] = value
                current_location_name = value
            elif key == 'CurrentTime':
                context["current_time"] = value
            elif key == 'InfrastructureFlags':
                try:
                    infra_flags = json.loads(value)
                except Exception:
                    infra_flags = {}
            elif key == 'EconomyFlags':
                try:
                    economy_flags = json.loads(value)
                except Exception:
                    economy_flags = {}
            elif key == 'TechnologyLevel':
                technology_level = value
                technology_level_from_db = True
            elif key == 'SettingEra':
                setting_era = value
                setting_era_from_db = True
            elif key == 'WorldModel':
                try:
                    parsed_world = json.loads(value)
                except Exception:
                    parsed_world = None
                if isinstance(parsed_world, dict):
                    world_model_raw = parsed_world
            elif key == 'RoleplayMode':
                normalized_mode = _normalize_mode_value(value)
                if normalized_mode:
                    context["mode"] = normalized_mode
            elif key == 'ModePolicy':
                normalized_policy = _normalize_mode_policy_value(value)
                if normalized_policy:
                    context["mode_policy"] = normalized_policy

        if world_type_value:
            context["type"] = world_type_value
        context["technology_level"] = technology_level or context.get("technology_level")
        context["setting_era"] = setting_era or context.get("setting_era")
        context["infrastructure_flags"] = infra_flags
        context["economy_flags"] = economy_flags
        context["physics_caps"] = physics_caps_local
        context["world_model"] = _normalize_world_model_metadata(
            world_model_raw,
            {
                "kind": context.get("kind"),
                "type": context.get("type"),
                "reality_context": context.get("reality_context"),
            },
        )

        caps_loaded_flag = any(
            [
                bool(context["capabilities"]),
                bool(infra_flags),
                bool(physics_caps_local),
                bool(economy_flags),
                bool(technology_level),
                bool(setting_era),
            ]
        )
        context["capabilities"].update(
            _derive_feasibility_caps(
                infra_flags,
                physics_caps_local,
                economy_flags,
                context.get("setting_era"),
            )
        )
        context["caps_loaded"] = caps_loaded_flag

        if current_location_name:
            location_row = await conn.fetchrow(
                """
                SELECT * FROM Locations
                WHERE user_id=$1 AND conversation_id=$2 AND location_name=$3
                LIMIT 1
                """,
                nyx_ctx.user_id,
                nyx_ctx.conversation_id,
                current_location_name,
            )
            if location_row:
                location_obj = Location(**dict(location_row))
                context["_location_object"] = location_obj
                try:
                    context["location_object"] = location_obj.to_dict()
                except Exception:
                    logger.exception("Failed to convert Location object to dict", exc_info=True)
                    context["location_object"] = dict(location_row)

        normalized_mode_policy = _normalize_mode_policy_value(context.get("mode_policy"))
        if normalized_mode_policy is None:
            normalized_mode_policy = "roleplay_only"
        context["mode_policy"] = normalized_mode_policy

        # Auto-detect if not set
        if context["type"] == "unknown":
            kind_defaults = _get_setting_kind_defaults(context.get("kind"))
            if kind_defaults:
                context["type"] = kind_defaults.get("type", context["type"])
                default_caps = kind_defaults.get("capabilities", {}) or {}
                merged_caps = {**default_caps, **(context.get("capabilities") or {})}
                context["capabilities"] = merged_caps
                if not context.get("magic_system") and "magic" in merged_caps:
                    context["magic_system"] = merged_caps["magic"]
                if not technology_level_from_db:
                    tech_level_default = kind_defaults.get("technology_level") or merged_caps.get("technology")
                    if tech_level_default:
                        context["technology_level"] = tech_level_default
                        technology_level_from_db = True
                if not setting_era_from_db:
                    era_default = kind_defaults.get("setting_era") or merged_caps.get("era")
                    if era_default:
                        context["setting_era"] = era_default
                        setting_era_from_db = True

        if context["type"] == "unknown":
            detected = await detect_setting_type(nyx_ctx)
            context["type"] = detected["setting_type"]
            context["kind"] = detected.get("setting_kind", "modern_realistic")
            context["capabilities"] = detected.get("capabilities", {})

        if not context["type"] or context["type"] == "unknown":
            context["type"] = "modern_realistic"

        context["capabilities"].update(
            _derive_feasibility_caps(
                context.get("infrastructure_flags"),
                context.get("physics_caps"),
                context.get("economy_flags"),
                context.get("setting_era"),
            )
        )
            
        # Get established impossibilities
        impossibilities = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedImpossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if impossibilities:
            context["established_impossibilities"] = json.loads(impossibilities)
            
        # Get established possibilities
        possibilities = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedPossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if possibilities:
            context["established_possibilities"] = json.loads(possibilities)
            
        # Get current scene state
        scene = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentScene'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if scene and scene["value"]:
            scene_data = json.loads(scene["value"])
            context["scene"] = scene_data
            context["location"].update(scene_data.get("location", {}))
            context["available_items"] = scene_data.get("items", [])
            context["present_entities"] = scene_data.get("npcs", [])
            context["location_features"] = scene_data.get("location_features", [])
            
        # Get game rules with categorization
        rules = await conn.fetch("""
            SELECT rule_name, condition, effect
            FROM GameRules
            WHERE user_id=$1 AND conversation_id=$2
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        for r in rules:
            rule_data = {
                "name": r["rule_name"], 
                "condition": r["condition"], 
                "effect": r["effect"]
            }
            
            # Categorize rules
            if r["rule_name"].startswith("hard_"):
                context["hard_rules"].append(rule_data)
            elif r["rule_name"].startswith("soft_"):
                context["soft_rules"].append(rule_data)
            else:
                context["established_rules"].append(rule_data)
                
        # Get character state
        player_stats = await conn.fetchrow("""
            SELECT * FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 1
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if player_stats:
            context["character_state"] = dict(player_stats)
            
        # Get inventory items
        inventory = await conn.fetch("""
            SELECT item_name, equipped FROM PlayerInventory
            WHERE user_id=$1 AND conversation_id=$2
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)

        context["available_items"].extend([item["item_name"] for item in inventory])

        known_locations_rows = await conn.fetch(
            """
            SELECT location_name FROM Locations
            WHERE user_id=$1 AND conversation_id=$2
            """,
            nyx_ctx.user_id,
            nyx_ctx.conversation_id,
        )
        context["known_location_names"] = [
            row["location_name"]
            for row in known_locations_rows
            if row["location_name"]
        ]
        
        # Get active NPCs in current location
        if context["location"].get("name"):
            npcs = await conn.fetch("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 
                AND current_location=$3
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, context["location"]["name"])
            
            context["present_entities"].extend([npc["npc_name"] for npc in npcs])
            
        # Get recent narrative for context
        recent = await conn.fetch("""
            SELECT content FROM messages
            WHERE conversation_id=$1 AND sender='Nyx'
            ORDER BY created_at DESC LIMIT 5
        """, nyx_ctx.conversation_id)
        
        context["narrative_history"] = [r["content"][:500] for r in recent if r["content"]]

    existing_world_model = context.get("world_model")
    raw_world_model = (
        existing_world_model.get("raw")
        if isinstance(existing_world_model, dict)
        else existing_world_model
    )
    context["world_model"] = _normalize_world_model_metadata(raw_world_model, context)

    return context

async def generate_dynamic_rejection(
    setting_context: Dict[str, Any], 
    intent: Dict[str, Any],
    nyx_ctx: NyxContext
) -> Dict[str, Any]:
    """Generate completely unique, contextual rejection narratives"""
    
    # Load rejection history to avoid repetition
    rejection_history = await _load_rejection_history(nyx_ctx)
    
    # Get current scene details
    current_scene = await _load_current_scene(nyx_ctx)
    
    # Build dynamic context
    rejection_context = {
        "setting": {
            "name": setting_context.get("setting_name"),
            "kind": setting_context.get("kind"),
            "atmosphere": setting_context.get("environment_desc", "")[:500],
            "reality_type": setting_context.get("reality_context"),
            "capabilities": setting_context.get("capabilities", {}),
            "current_location": setting_context.get("location", {}).get("name"),
            "physics_model": setting_context.get("physics_model"),
            "time": setting_context.get("current_time", "unknown time")
        },
        "attempted_action": {
            "raw_input": intent.get("raw_text", ""),
            "categories": intent.get("categories", []),
            "violations": intent.get("violations", []),
            "specific_reason": intent.get("violations", [{}])[0].get("reason", "") if intent.get("violations") else ""
        },
        "scene_context": {
            "present_npcs": current_scene.get("npcs", []),
            "available_items": current_scene.get("items", []),
            "recent_events": current_scene.get("recent_narrative", [])[-3:],
            "location_features": current_scene.get("location_features", []),
            "time_phase": current_scene.get("time_phase", "day")
        },
        "rejection_history": rejection_history[-5:],  # Last 5 to avoid repetition
        "instruction": "Generate a unique rejection that has never been used before. Make it specific to this exact moment and action."
    }
    
    # Generate unique rejection
    run = await Runner.run(
        REJECTION_NARRATOR_AGENT,
        json.dumps(rejection_context)
    )
    
    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        
        # Store this rejection to avoid future repetition
        await _store_rejection(nyx_ctx, result)
        
        # Generate contextual alternatives
        result["suggested_alternatives"] = await _generate_contextual_alternatives(
            nyx_ctx, setting_context, current_scene, intent
        )
        
        return result
    except Exception as e:
        # Dynamic fallback
        return await _generate_fallback_rejection(setting_context, intent, current_scene)

async def _generate_contextual_alternatives(
    nyx_ctx: NyxContext,
    setting_context: Dict,
    current_scene: Dict,
    failed_intent: Dict
) -> List[str]:
    """Generate alternatives based on what's actually available in the scene"""
    
    context = {
        "failed_attempt": {
            "action": failed_intent.get("raw_text", ""),
            "categories": failed_intent.get("categories", []),
            "goal": "What the player was trying to achieve"
        },
        "scene_state": {
            "npcs": current_scene.get("npcs", []),
            "items": current_scene.get("items", []),
            "location": setting_context.get("location", {}).get("name", "unknown"),
            "location_features": current_scene.get("location_features", []),
            "time": current_scene.get("time_phase", "day")
        },
        "player_state": {
            "abilities": setting_context.get("character_abilities", []),
            "inventory": setting_context.get("available_items", []),
            "stats": setting_context.get("character_state", {})
        },
        "world_rules": {
            "capabilities": setting_context.get("capabilities", {}),
            "kind": setting_context.get("kind", "realistic"),
            "established_possibilities": setting_context.get("established_possibilities", [])[-5:]
        }
    }
    
    sanitized_context = _context_payload_for_agent(context)
    run = await Runner.run(ALTERNATIVE_GENERATOR_AGENT, json.dumps(sanitized_context))
    
    try:
        alternatives = json.loads(getattr(run, "final_output", "[]"))
        return alternatives[:3]
    except:
        # Dynamic fallback based on actual scene
        return await _generate_scene_based_alternatives(current_scene, setting_context)

async def _generate_scene_based_alternatives(current_scene: Dict, setting_context: Dict) -> List[str]:
    """Generate fallback alternatives based on scene elements"""
    alternatives = []
    
    # NPC interactions
    if current_scene.get("npcs"):
        npc = random.choice(current_scene["npcs"])
        alternatives.append(f"approach {npc} for assistance")
    
    # Item usage
    if current_scene.get("items"):
        item = random.choice(current_scene["items"])
        alternatives.append(f"examine the {item} more closely")
    
    # Location features
    if current_scene.get("location_features"):
        feature = random.choice(current_scene["location_features"])
        alternatives.append(f"investigate the {feature}")
    
    # Time-based alternatives
    time_phase = current_scene.get("time_phase", "day")
    if time_phase == "night":
        alternatives.append("wait until dawn for better visibility")
    elif time_phase == "day":
        alternatives.append("search for a different approach")
    
    # Setting-specific alternatives
    if setting_context.get("kind") == "high_fantasy":
        alternatives.append("seek magical guidance")
    elif setting_context.get("kind") == "cyberpunk":
        alternatives.append("access the local network for information")
    else:
        alternatives.append("reconsider your approach")
    
    return alternatives[:3]

async def _quick_feasibility_check(setting_context: Dict, intents: List[Dict]) -> Dict:
    """Quick check against hard rules without repetitive responses"""
    blocked = False
    intent_feasible: List[bool] = []
    violations: List[List[Dict[str, Any]]] = []
    missing_prereq_flags: List[bool] = []

    for intent in intents:
        intent_violations = []
        feasible = True
        prereq_missing = False
        categories = {str(cat).lower() for cat in intent.get("categories", [])}

        # Check hard rules dynamically
        for rule in setting_context.get("hard_rules", []):
            if await _rule_applies_to_intent(rule, intent, setting_context):
                feasible = False
                intent_violations.append({
                    "rule": rule["name"],
                    "reason": rule["effect"]
                })
        
        # Check established impossibilities with fuzzy matching
        for imp in setting_context.get("established_impossibilities", []):
            if _matches_impossibility_dynamic(intent, imp):
                feasible = False
                intent_violations.append({
                    "rule": "established_impossibility",
                    "reason": imp["reason"]
                })
        
        # Check prerequisites
        prereq_ok, prereq_reason = await _check_prerequisites(intent, setting_context)
        if not prereq_ok:
            feasible = False
            prereq_missing = True
            reason = prereq_reason or "No vendor/point-of-sale in current scene"
            if not ({"trade", "mundane_action"} & categories):
                reason = prereq_reason or "Required elements are not present"
            intent_violations.append({
                "rule": "missing_prereq",
                "reason": reason
            })
            logger.info(
                "[FEASIBILITY] Missing prerequisites detected for cats=%s -> %s",
                sorted(categories),
                reason,
            )

        intent_feasible.append(feasible)
        violations.append(intent_violations)
        missing_prereq_flags.append(prereq_missing)
        if not feasible and not prereq_missing:
            blocked = True

    return {
        "hard_blocked": blocked,
        "intent_feasible": intent_feasible,
        "violations": violations,
        "missing_prereq": missing_prereq_flags,
    }

async def _rule_applies_to_intent(rule: Dict, intent: Dict, context: Dict) -> bool:
    """Dynamically check if a rule applies to an intent"""
    condition = rule.get("condition", "").lower()
    
    # Check intent categories
    if "categories" in intent:
        for cat in intent["categories"]:
            if cat.lower() in condition:
                return True
    
    # Check action keywords
    intent_str = json.dumps(intent).lower()
    condition_keywords = set(condition.split())
    intent_keywords = set(intent_str.split())
    
    # Require significant overlap
    overlap = len(condition_keywords & intent_keywords)
    if overlap >= max(2, len(condition_keywords) * 0.3):
        return True
    
    return False

def _matches_impossibility_dynamic(intent: Dict, impossibility: Dict) -> bool:
    """Dynamic matching with fuzzy logic"""
    # Category-based matching with threshold
    imp_categories = set(impossibility.get("categories", []))
    intent_categories = set(intent.get("categories", []))
    
    if imp_categories and intent_categories:
        overlap = len(imp_categories & intent_categories)
        if overlap >= max(1, min(len(imp_categories), len(intent_categories)) * 0.5):
            return True
    
    # Semantic similarity for actions
    if "action" in impossibility:
        imp_action = impossibility["action"].lower()
        intent_text = json.dumps(intent).lower()
        
        # Key phrase matching
        if len(imp_action) > 10 and imp_action in intent_text:
            return True
        
        # Word overlap threshold
        imp_words = set(imp_action.split())
        intent_words = set(intent_text.split())
        
        common_words = imp_words & intent_words
        # Filter out common words
        meaningful_overlap = common_words - {"the", "a", "an", "to", "from", "with", "at", "in", "on"}
        
        if len(meaningful_overlap) >= min(3, len(imp_words) * 0.4):
            return True
    
    return False

async def _check_prerequisites(intent: Dict, context: Dict) -> Tuple[bool, Optional[str]]:
    """Check if required elements for the action are present"""

    categories = {str(cat).lower() for cat in intent.get("categories", [])}

    def _normalize_term(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned.lower() if cleaned else None
        if isinstance(value, dict):
            for key in ("name", "item_name", "npc_name", "label", "title", "display_name"):
                val = value.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip().lower()
            for key in ("id", "item_id", "npc_id", "npcId", "itemId"):
                val = value.get(key)
                if val is not None:
                    return str(val).strip().lower()
            return None
        return str(value).strip().lower()

    def _display_term(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("name", "item_name", "npc_name", "label", "title", "display_name"):
                val = value.get(key)
                if val:
                    return str(val)
        return str(value)

    available_items = context.get("available_items")
    present_entities = context.get("present_entities")
    location = context.get("location") or {}
    location_features_sources = [
        context.get("location_features"),
        location.get("features"),
        location.get("notable_features"),
        location.get("points_of_interest"),
        location.get("tags"),
    ]

    scene_tokens: Set[str] = set()
    for values in location_features_sources:
        scene_tokens |= _tokenize_scene_values(values)

    loc_name = location.get("name")
    if loc_name:
        scene_tokens.add(str(loc_name).strip().lower())

    available_items_tokens = _tokenize_scene_values(available_items)
    present_entities_tokens = _tokenize_scene_values(present_entities)

    has_scene_data = (
        available_items is not None
        or present_entities is not None
        or bool(location)
        or any(source is not None for source in location_features_sources)
    )

    if not has_scene_data:
        return True, None

    # Check required items
    instruments = intent.get("instruments")
    if instruments is None:
        instruments_iter: List[Any] = []
    elif isinstance(instruments, (list, tuple, set)):
        instruments_iter = list(instruments)
    else:
        instruments_iter = [instruments]

    for item in instruments_iter:
        normalized = _normalize_term(item)
        if not normalized:
            continue
        if normalized not in available_items_tokens:
            reason = f"Required item '{_display_term(item)}' is not available here."
            return False, reason

    # Check target presence
    direct_objects = intent.get("direct_object")
    if direct_objects is None:
        direct_iter: List[Any] = []
    elif isinstance(direct_objects, (list, tuple, set)):
        direct_iter = list(direct_objects)
    else:
        direct_iter = [direct_objects]

    for target in direct_iter:
        normalized = _normalize_term(target)
        display_normalized = _normalize_term(_display_term(target))

        if (
            (normalized and normalized in LOCATION_REFERENCE_KEYWORDS)
            or (display_normalized and display_normalized in LOCATION_REFERENCE_KEYWORDS)
        ):
            if loc_name:
                continue
            return False, "No current location recorded."

        if not normalized:
            continue
        if (
            normalized not in present_entities_tokens
            and normalized not in available_items_tokens
            and normalized not in scene_tokens
        ):
            reason = f"{_display_term(target)} isn't here right now."
            return False, reason

    # Check ability requirements based on categories
    if "spellcasting" in categories and context.get("magic_system") == "none":
        return False, "Spellcasting isn't possible in this setting."
    if "hacking" in categories and context.get("technology_level") in ["primitive", "medieval"]:
        return False, "Hacking isn't possible with the current technology level."

    return True, None

async def _full_dynamic_assessment(
    nyx_ctx: NyxContext,
    user_input: str,
    intents: List[Dict],
    setting_context: Dict
) -> Dict[str, Any]:
    """Full AI-powered assessment for nuanced cases"""
    
    # Build comprehensive assessment context
    assessment_context = {
        "user_input": user_input,
        "intents": intents,
        "setting": {
            "name": setting_context.get("setting_name"),
            "kind": setting_context.get("kind"),
            "capabilities": setting_context.get("capabilities"),
            "reality_level": setting_context.get("reality_context"),
            "environment": setting_context.get("environment_desc", "")[:500],
            "hard_rules": setting_context.get("hard_rules"),
            "soft_rules": setting_context.get("soft_rules")
        },
        "context": {
            "location": setting_context.get("location"),
            "present_npcs": setting_context.get("present_entities"),
            "available_items": setting_context.get("available_items"),
            "player_state": setting_context.get("character_state"),
            "recent_narrative": setting_context.get("narrative_history", [])[-3:]
        },
        "history": {
            "established_impossibilities": setting_context.get("established_impossibilities", [])[-10:],
            "established_possibilities": setting_context.get("established_possibilities", [])[-10:]
        }
    }
    
    # Run full assessment
    run = await Runner.run(FEASIBILITY_AGENT, json.dumps(assessment_context))
    
    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        
        # Enhance denied intents with dynamic rejections
        for i, intent_result in enumerate(result.get("per_intent", [])):
            if not intent_result.get("feasible"):
                # Generate unique rejection
                rejection = await generate_dynamic_rejection(
                    setting_context,
                    {**intents[i], "raw_text": user_input, "violations": intent_result.get("violations", [])},
                    nyx_ctx
                )
                
                intent_result.update(rejection)
        
        return result
    except Exception as e:
        logger.error(f"Dynamic assessment failed: {e}")
        return _default_feasibility_response(intents)

async def _load_rejection_history(nyx_ctx: NyxContext) -> List[Dict]:
    """Load recent rejection narratives to avoid repetition"""
    async with get_db_connection_context() as conn:
        history = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='RejectionHistory'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        return json.loads(history) if history else []

async def _store_rejection(nyx_ctx: NyxContext, rejection: Dict):
    """Store rejection for future reference"""
    history = await _load_rejection_history(nyx_ctx)
    
    # Add timestamp and context
    rejection["timestamp"] = datetime.now().isoformat()
    rejection["context_hash"] = hash(json.dumps(rejection, sort_keys=True))
    
    # Add to history and limit size
    history.append(rejection)
    history = history[-30:]  # Keep last 30 rejections
    
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'RejectionHistory', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
        """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(history))

async def _load_current_scene(nyx_ctx: NyxContext) -> Dict:
    """Load comprehensive current scene data"""
    scene = {
        "npcs": [],
        "items": [],
        "location_features": [],
        "recent_narrative": [],
        "time_phase": "unknown"
    }
    
    async with get_db_connection_context() as conn:
        # Get scene state
        scene_data = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentScene'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if scene_data:
            parsed = json.loads(scene_data)
            scene.update(parsed)
        
        # Get current time phase
        time_data = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentTime'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if time_data:
            import re
            time_match = re.search(r'(Morning|Afternoon|Evening|Night)', time_data)
            if time_match:
                scene["time_phase"] = time_match.group(1).lower()
        
        # Get location details
        location_raw = scene.get("location")
        location_name = None

        if isinstance(location_raw, dict):
            for key in ("name", "location_name", "display_name", "title", "label"):
                value = location_raw.get(key)
                if isinstance(value, str) and value.strip():
                    location_name = value
                    break
        elif isinstance(location_raw, str):
            location_name = location_raw

        if not location_name:
            location_name = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentLocation'
            """, nyx_ctx.user_id, nyx_ctx.conversation_id)

        if location_name:
            location_row = await conn.fetchrow(
                """
                SELECT *
                FROM Locations
                WHERE user_id=$1 AND conversation_id=$2 AND location_name=$3
                """,
                nyx_ctx.user_id,
                nyx_ctx.conversation_id,
                location_name,
            )

            if location_row:
                location_obj = Location.from_record(
                    location_row,
                    user_id=nyx_ctx.user_id,
                    conversation_id=nyx_ctx.conversation_id,
                    location_name=location_name,
                )
                scene["location_features"] = list(location_obj.notable_features)
                scene["location_description"] = location_obj.description or ""
                scene["location_record"] = location_obj.to_dict()

                enriched_details = {
                    "room": location_obj.room,
                    "building": location_obj.building,
                    "district": location_obj.district,
                    "district_type": location_obj.district_type,
                    "city": location_obj.city,
                    "region": location_obj.region,
                    "country": location_obj.country,
                    "planet": location_obj.planet,
                    "galaxy": location_obj.galaxy,
                    "realm": location_obj.realm,
                    "lat": location_obj.lat,
                    "lon": location_obj.lon,
                    "is_fictional": location_obj.is_fictional,
                    "cultural_significance": location_obj.cultural_significance,
                    "economic_importance": location_obj.economic_importance,
                    "strategic_value": location_obj.strategic_value,
                    "population_density": location_obj.population_density,
                    "access_restrictions": list(location_obj.access_restrictions),
                    "local_customs": list(location_obj.local_customs),
                    "hidden_aspects": list(location_obj.hidden_aspects),
                    "controlling_faction": location_obj.controlling_faction,
                }
                enriched_details = {k: v for k, v in enriched_details.items() if v not in (None, [])}

                current_details = scene.get("location_details")
                if isinstance(current_details, dict):
                    current_details.update(enriched_details)
                else:
                    scene["location_details"] = enriched_details

        # Get recent narrative
        recent = await conn.fetch("""
            SELECT content FROM messages
            WHERE conversation_id=$1 AND sender='Nyx'
            ORDER BY created_at DESC LIMIT 3
        """, nyx_ctx.conversation_id)
        
        scene["recent_narrative"] = [r["content"][:200] for r in recent]
    
    return scene

async def _generate_fallback_rejection(
    setting_context: Dict, 
    intent: Dict,
    current_scene: Dict
) -> Dict[str, Any]:
    """Generate dynamic fallback rejection when AI fails"""
    
    # Build contextual response based on setting
    setting_kind = setting_context.get("kind", "realistic")
    
    reality_responses = {
        "realistic": [
            f"The world remains bound by familiar laws",
            f"Reality offers no exception here",
            f"The universe maintains its steady rhythm"
        ],
        "fantasy": [
            f"Even magic has boundaries it cannot cross",
            f"The weave resists this particular thread",
            f"Ancient laws hold firm against your will"
        ],
        "scifi": [
            f"The system parameters reject this input",
            f"Quantum mechanics forbid this outcome",
            f"The simulation constraints remain absolute"
        ]
    }
    
    base_type = "realistic"
    if "fantasy" in setting_kind or "magic" in setting_kind:
        base_type = "fantasy"
    elif "sci" in setting_kind or "cyber" in setting_kind or "tech" in setting_kind:
        base_type = "scifi"
    
    return {
        "reality_response": random.choice(reality_responses.get(base_type, reality_responses["realistic"])),
        "narrator_guidance": f"What you attempt slips beyond reach, the world's fabric unchanged by will alone.",
        "suggested_alternatives": await _generate_scene_based_alternatives(current_scene, setting_context),
        "metaphor": "like trying to paint with shadows on water"
    }

async def record_impossibility(nyx_ctx: NyxContext, action: str, reason: str):
    """Record that something has been established as impossible in this setting"""
    
    async with get_db_connection_context() as conn:
        # Get existing impossibilities
        current = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedImpossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        impossibilities = json.loads(current) if current else []
        
        # Extract categories
        categories = await _extract_action_categories(action)
        
        # Check for duplicates with fuzzy matching
        is_duplicate = False
        for imp in impossibilities:
            if _similar_impossibility(imp, action, categories):
                is_duplicate = True
                break
        
        if not is_duplicate:
            impossibilities.append({
                "action": action,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "categories": categories,
                "hash": hash(action + reason)
            })
            
            # Keep limited history
            impossibilities = impossibilities[-50:]
            
            # Store updated list
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'EstablishedImpossibilities', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(impossibilities))

async def record_possibility(nyx_ctx: NyxContext, action: str, categories: List[str]):
    """Record that something has been established as possible in this setting"""
    
    async with get_db_connection_context() as conn:
        current = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EstablishedPossibilities'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        possibilities = json.loads(current) if current else []
        
        # Check for duplicates
        is_duplicate = any(
            p.get("hash") == hash(action) for p in possibilities
        )
        
        if not is_duplicate:
            possibilities.append({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "categories": categories,
                "hash": hash(action)
            })
            
            possibilities = possibilities[-50:]
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'EstablishedPossibilities', $3)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, json.dumps(possibilities))

async def detect_setting_type(nyx_ctx: NyxContext) -> Dict[str, Any]:
    """Intelligently detect what kind of setting this is based on established elements"""
    
    # Gather comprehensive context
    context = {"narrative": [], "elements": {}, "npcs": [], "locations": [], "items": []}
    
    async with get_db_connection_context() as conn:
        # Get recent narrative
        recent_msgs = await conn.fetch("""
            SELECT content FROM messages
            WHERE conversation_id=$1 
            ORDER BY created_at DESC LIMIT 10
        """, nyx_ctx.conversation_id)
        
        context["narrative"] = [msg["content"][:500] for msg in recent_msgs if msg["content"]]
        
        # Get roleplay elements
        elements = await conn.fetch("""
            SELECT key, value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 
            AND key IN ('EnvironmentDesc', 'EnvironmentHistory', 'CurrentSetting', 'ScenarioName')
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        for elem in elements:
            context["elements"][elem["key"]] = elem["value"]
        
        # Sample NPCs
        npcs = await conn.fetch("""
            SELECT npc_name, role, archetypes FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 5
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["npcs"] = [{"name": n["npc_name"], "role": n["role"]} for n in npcs]
        
        # Sample locations
        locations = await conn.fetch("""
            SELECT location_name, location_type FROM Locations
            WHERE user_id=$1 AND conversation_id=$2
            LIMIT 5
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        context["locations"] = [{"name": l["location_name"], "type": l["location_type"]} for l in locations]
    
    try:
        run = await asyncio.wait_for(
            Runner.run(
                SETTING_DETECTIVE_AGENT,
                json.dumps(_context_payload_for_agent(context)),
            ),
            timeout=SETTING_DETECTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Setting detection timed out after %ss; using heuristic fallback",
            SETTING_DETECTION_TIMEOUT_SECONDS,
        )
        result = _heuristic_setting_detection(context, confidence=0.35)
        await _persist_detected_setting(nyx_ctx, result)
        return result
    except Exception as exc:
        logger.warning("Setting detection failed: %s", exc)
        result = _heuristic_setting_detection(context)
        await _persist_detected_setting(nyx_ctx, result)
        return result

    try:
        result = json.loads(getattr(run, "final_output", "{}"))
        if not isinstance(result, dict):
            raise ValueError("unexpected response type")
        if "setting_type" not in result:
            raise ValueError("missing setting_type in response")
    except Exception as exc:
        logger.warning("Invalid setting detection response: %s", exc)
        result = _heuristic_setting_detection(context)

    await _persist_detected_setting(nyx_ctx, result)
    return result


async def _persist_detected_setting(nyx_ctx: NyxContext, result: Dict[str, Any]) -> None:
    """Persist detected setting information so subsequent runs can reuse it."""

    setting_type = result.get("setting_type", "realistic_modern")
    setting_kind = result.get("setting_kind")
    capabilities = result.get("capabilities", {})

    try:
        capabilities_json = json.dumps(capabilities)
    except TypeError:
        capabilities_json = json.dumps({})

    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'SettingType', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
            """,
            nyx_ctx.user_id,
            nyx_ctx.conversation_id,
            setting_type,
        )

        if setting_kind:
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'SettingKind', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                nyx_ctx.user_id,
                nyx_ctx.conversation_id,
                setting_kind,
            )

        await conn.execute(
            """
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'DetectedCapabilities', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
            """,
            nyx_ctx.user_id,
            nyx_ctx.conversation_id,
            capabilities_json,
        )


def _heuristic_setting_detection(
    context: Dict[str, Any],
    confidence: float = 0.3,
) -> Dict[str, Any]:
    """Fallback setting detection based on simple keyword heuristics."""

    text_fragments: List[str] = []
    text_fragments.extend(context.get("narrative", []))
    text_fragments.extend(context.get("elements", {}).values())
    text_fragments.extend(npc.get("role", "") for npc in context.get("npcs", []))
    text_fragments.extend(loc.get("type", "") for loc in context.get("locations", []))

    text = " ".join(fragment for fragment in text_fragments if fragment).lower()

    result = {
        "setting_type": "realistic_modern",
        "setting_kind": "modern_realistic",
        "confidence": confidence,
        "capabilities": {
            "magic": "none",
            "technology": "modern",
            "physics": "realistic",
            "supernatural": "none",
        },
    }

    heuristic_checks = [
        (
            {"dragon", "spell", "wizard", "sorcery", "enchanted", "mana", "mage"},
            {
                "setting_type": "high_fantasy",
                "setting_kind": "fantasy_epic",
                "confidence": max(confidence, 0.55),
                "capabilities": {
                    "magic": "common",
                    "technology": "medieval",
                    "physics": "flexible",
                    "supernatural": "known",
                },
            },
        ),
        (
            {"spaceship", "quantum", "laser", "android", "starfleet", "hyperspace", "plasma"},
            {
                "setting_type": "sci_fi_futuristic",
                "setting_kind": "science_fiction",
                "confidence": max(confidence, 0.5),
                "capabilities": {
                    "magic": "none",
                    "technology": "futuristic",
                    "physics": "flexible",
                    "supernatural": "none",
                },
            },
        ),
        (
            {"cyber", "augment", "neon", "megacorp", "matrix", "netrunner"},
            {
                "setting_type": "cyberpunk",
                "setting_kind": "science_fiction",
                "confidence": max(confidence, 0.45),
                "capabilities": {
                    "magic": "none",
                    "technology": "advanced",
                    "physics": "realistic",
                    "supernatural": "none",
                },
            },
        ),
        (
            {"wasteland", "mutant", "radiation", "ruins", "apocalypse", "fallout"},
            {
                "setting_type": "post_apocalyptic",
                "setting_kind": "dystopian",
                "confidence": max(confidence, 0.45),
                "capabilities": {
                    "magic": "none",
                    "technology": "scavenged",
                    "physics": "realistic",
                    "supernatural": "hidden",
                },
            },
        ),
        (
            {"haunted", "ghost", "vampire", "werewolf", "eldritch", "occult"},
            {
                "setting_type": "supernatural_modern",
                "setting_kind": "modern_supernatural",
                "confidence": max(confidence, 0.5),
                "capabilities": {
                    "magic": "limited",
                    "technology": "modern",
                    "physics": "flexible",
                    "supernatural": "known",
                },
            },
        ),
    ]

    for keywords, overrides in heuristic_checks:
        if any(keyword in text for keyword in keywords):
            result.update(overrides)
            break

    return result


async def _extract_action_categories(action: str) -> List[str]:
    """Extract categories from an action string"""
    try:
        intents = await parse_action_intents(action)
        categories = set()
        for intent in intents:
            categories.update(intent.get("categories", []))
        return list(categories)
    except Exception:
        return []

def _similar_impossibility(existing: Dict, new_action: str, new_categories: List[str]) -> bool:
    """Check if an impossibility is similar to an existing one"""
    # Check hash first
    if existing.get("hash") == hash(new_action + existing.get("reason", "")):
        return True
    
    # Check category overlap
    existing_cats = set(existing.get("categories", []))
    new_cats = set(new_categories)
    
    if existing_cats and new_cats:
        overlap = len(existing_cats & new_cats)
        if overlap >= max(1, min(len(existing_cats), len(new_cats)) * 0.7):
            return True
    
    # Check action similarity
    if "action" in existing:
        existing_action = existing["action"].lower()
        new_action_lower = new_action.lower()
        
        # Direct substring match
        if len(existing_action) > 20 and len(new_action_lower) > 20:
            if existing_action in new_action_lower or new_action_lower in existing_action:
                return True
    
    return False

def _default_feasibility_response(intents: List[Dict]) -> Dict[str, Any]:
    """Default response when feasibility check fails"""
    
    # Allow only clearly mundane actions by default
    mundane_categories = {"movement", "dialogue", "observation", "mundane_action", "interaction", "trade"}
    
    per_intent = []
    all_feasible = True
    
    for intent in intents:
        intent_categories = set(intent.get("categories", []))
        is_mundane = bool(intent_categories & mundane_categories) or not intent_categories
        
        per_intent.append({
            "feasible": is_mundane,
            "strategy": "allow" if is_mundane else "deny",
            "violations": [] if is_mundane else [{"rule": "unknown", "reason": "Cannot verify feasibility"}],
            "categories": list(intent_categories)
        })
        
        if not is_mundane:
            all_feasible = False
    
    return {
        "overall": {
            "feasible": all_feasible,
            "strategy": "allow" if all_feasible else "deny"
        },
        "per_intent": per_intent
    }



def _is_ooc_request(text: Optional[str]) -> bool:
    """Detect classic out-of-character requests or real-world pivots."""

    if not text:
        return False

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    stripped = normalized.strip()
    if not stripped:
        return False

    for prefix in OOC_PREFIX_MARKERS:
        if stripped.startswith(prefix):
            return True

    if any(phrase in normalized for phrase in OOC_KEYPHRASES):
        return True

    for term in BRAND_TERMS:
        if " " in term:
            if term in normalized:
                return True
        else:
            if re.search(rf"\\b{re.escape(term)}\\b", normalized):
                return True

    return False


def _policy_roleplay_only_enabled(*policy_sources: Any, default: bool = True) -> bool:
    """Inspect nested policy metadata to determine if roleplay-only mode is active."""

    false_tokens = {"0", "false", "off", "disabled", "relaxed", "allow", "allowed", "open", "no"}

    def _coerce(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            norm = unicodedata.normalize("NFKC", value).strip().lower()
            if not norm:
                return None
            if norm in ROLEPLAY_ONLY_DEFAULT:
                return True
            if norm in false_tokens:
                return False
        return None

    def _extract_flag(candidate: Any, depth: int = 0) -> Optional[bool]:
        if candidate is None or depth > 5:
            return None

        coerced = _coerce(candidate)
        if coerced is not None:
            return coerced

        if isinstance(candidate, dict):
            keys = (
                "roleplay_only",
                "roleplayOnly",
                "ROLEPLAY_ONLY",
                "roleplay-mode",
                "roleplay_mode",
                "nyx_roleplay_only",
            )
            for key in keys:
                if key in candidate:
                    flag = _extract_flag(candidate[key], depth + 1)
                    if flag is not None:
                        return flag

            nested_keys = (
                "policy_flags",
                "policies",
                "policy",
                "flags",
                "feature_flags",
                "settings",
                "meta",
                "governance",
            )
            for nested_key in nested_keys:
                if nested_key in candidate:
                    flag = _extract_flag(candidate[nested_key], depth + 1)
                    if flag is not None:
                        return flag

        elif isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                flag = _extract_flag(item, depth + 1)
                if flag is not None:
                    return flag

        return None

    for source in policy_sources:
        flag = _extract_flag(source)
        if flag is not None:
            return flag

    return default


def _nyx_meta_decline_payload(reason: str = "roleplay_only") -> Dict[str, Any]:
    """Return a canonical Nyx meta decline payload for roleplay-only enforcement."""

    violation_reason = {
        "ooc_request": "Nyx stays firmly in character and won't break immersion.",
        "ooc_output": "Nyx refuses to respond to out-of-character prompts.",
        "irl_reference": "Nyx ignores real-world chatter and keeps the scene sealed.",
        "roleplay_only": "Nyx enforces in-character play only.",
    }.get(reason, "Nyx enforces in-character play only.")

    narrator_guidance = (
        "Nyx curls a finger, reminding you this stays in-scene. Answer her as the character you're playing."
    )

    return {
        "overall": {"feasible": False, "strategy": "deny"},
        "per_intent": [
            {
                "feasible": False,
                "strategy": "deny",
                "violations": [
                    {
                        "rule": "policy:roleplay_only",
                        "reason": violation_reason,
                    }
                ],
                "narrator_guidance": narrator_guidance,
                "suggested_alternatives": [
                    "Describe what your character does next in the scene.",
                    "React to Nyx from your character's perspective.",
                ],
                "categories": [],
            }
        ],
        "meta": {
            "kind": "nyx_meta_decline",
            "reason": reason,
        },
    }


def _harden_output_against_ooc_leakage(
    result: Any,
    *,
    request_text: Optional[str] = None,
    policy_sources: Optional[Any] = None,
) -> Dict[str, Any]:
    """Inspect agent output and swap in a Nyx meta decline when OOC leakage is detected."""

    if isinstance(policy_sources, dict) or not isinstance(policy_sources, Iterable):
        sources: List[Any] = [policy_sources] if policy_sources is not None else []
    else:
        sources = list(policy_sources)

    policy_enabled = _policy_roleplay_only_enabled(*sources)
    if not policy_enabled:
        return {"declined": False, "payload": result, "reason": None}

    decline_reason: Optional[str] = None

    if request_text and _is_ooc_request(request_text):
        decline_reason = "ooc_request"

    seen: Set[int] = set()
    fragments: List[str] = []

    def _collect(value: Any, depth: int = 0) -> None:
        if value is None or depth > 5:
            return
        identifier = id(value)
        if identifier in seen:
            return
        seen.add(identifier)

        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                fragments.append(stripped)
            return

        if isinstance(value, dict):
            for candidate in value.values():
                _collect(candidate, depth + 1)
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                _collect(item, depth + 1)
            return

        for attr in ("final_output", "output_text", "content", "text"):
            if hasattr(value, attr):
                _collect(getattr(value, attr), depth + 1)

        for attr in ("messages", "history", "events"):
            if hasattr(value, attr):
                _collect(getattr(value, attr), depth + 1)

    _collect(result)

    for fragment in fragments:
        if _is_ooc_request(fragment):
            decline_reason = decline_reason or "ooc_output"
            break
        normalized_fragment = unicodedata.normalize("NFKC", fragment).lower()
        for term in BRAND_TERMS:
            if (" " in term and term in normalized_fragment) or (
                " " not in term and re.search(rf"\\b{re.escape(term)}\\b", normalized_fragment)
            ):
                decline_reason = decline_reason or "irl_reference"
                break
        if decline_reason:
            break

    if decline_reason:
        payload = _nyx_meta_decline_payload(decline_reason)
        return {"declined": True, "payload": payload, "reason": decline_reason}

    return {"declined": False, "payload": result, "reason": None}



def _normalize_bool(v: Any) -> bool:
    if isinstance(v, bool): return v
    if v is None: return False
    s = str(v).strip().lower()
    return s in {"1","true","yes","y","on","allowed","enable","enabled"}

def _safe_json_loads(s: Optional[str], default):
    try:
        return json.loads(s or "") if s else default
    except Exception:
        return default

INHERENT_INSTRUMENT_TOKENS: Set[str] = {
    "hand",
    "hands",
    "palm",
    "palms",
    "fist",
    "fists",
    "finger",
    "fingers",
    "thumb",
    "thumbs",
    "arm",
    "arms",
    "mouth",
    "teeth",
    "tongue",
    "voice",
    "voices",
    "breath",
}


AMBIENT_DEBRIS_KEYWORDS: Set[str] = {
    "coin",
    "coins",
    "penny",
    "pennies",
    "rock",
    "rocks",
    "stone",
    "stones",
    "pebble",
    "pebbles",
    "stick",
    "sticks",
    "twig",
    "twigs",
    "branch",
    "branches",
}

AMBIENT_DEBRIS_CANONICALS: Set[str] = {"coin", "rock", "pebble"}

STERILE_ENVIRONMENT_HINTS: Set[str] = {
    "lab",
    "laboratory",
    "cleanroom",
    "clean room",
    "medbay",
    "med bay",
    "medical bay",
    "medical lab",
    "clinic",
    "hospital",
    "infirmary",
    "sickbay",
    "sterile",
    "operating room",
    "surgical wing",
}


async def assess_action_feasibility_fast(user_id: int, conversation_id: int, text: str) -> Dict[str, Any]:
    """
    Conversation/scene-aware quick feasibility gate with LOUD logging and dynamic judgments.
    - Uses per-conversation GameRules + CurrentRoleplay to decide.
    - Explicit rules/scene bans/EstablishedImpossibilities can deny.
    - Capability mismatches only downgrade to ASK (soft block) rather than hard DENY.
    - If context is missing, prefer ALLOW (keep world flexible).
    """
    logger.info(f"[FEASIBILITY] Checking: {text[:160]!r}")
    text_l = (text or "").lower()

    # ---- 1) Parse intents (never hard-block on parse errors) -----------------
    parse_error: Optional[str] = None
    try:
        intents = await parse_action_intents(text or "")
        logger.info(f"[FEASIBILITY] Parsed {len(intents)} intents "
                    f"-> cats: {[i.get('categories', []) for i in intents]}")
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
        logger.error(f"[FEASIBILITY] Intent parsing FAILED: {parse_error}", exc_info=True)
        intents = []

    # Fallback single-pass intent so we still produce a decision
    if not intents:
        intents = [{"categories": list(_infer_categories_from_text(text_l)) or []}]

    # ---- 2) Load dynamic context ------------------------------------------------
    setting_kind = "modern_realistic"
    setting_type = "modern_realistic"
    reality_context = "normal"
    physics_model = "realistic"

    capabilities: Dict[str, Any] = {}
    scene: Dict[str, Any] = {}
    location_name: Optional[str] = None
    established_impossibilities: List[Dict[str, Any]] = []
    rules: List[Dict[str, Any]] = []
    known_location_names: List[str] = []
    caps_loaded_flag = False
    world_type: Optional[str] = None
    infra_flags: Dict[str, Any] = {}
    economy_flags: Dict[str, Any] = {}
    physics_caps_local: Dict[str, Any] = {}
    technology_level: Optional[str] = None
    setting_era: Optional[str] = None
    world_model_context: Dict[str, Any] = _normalize_world_model_metadata(
        {},
        {
            "kind": setting_kind,
            "type": setting_type,
            "reality_context": reality_context,
        },
    )

    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                """
                SELECT key, value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2
                  AND key = ANY($3)
                """,
                user_id,
                conversation_id,
                [
                    "WorldType",
                    "SettingType",
                    "SettingKind",
                    "RealityContext",
                    "PhysicsModel",
                    "PhysicsCaps",
                    "SettingCapabilities",
                    "CurrentScene",
                    "CurrentLocation",
                    "EstablishedImpossibilities",
                    "InfrastructureFlags",
                    "EconomyFlags",
                    "TechnologyLevel",
                    "SettingEra",
                    "WorldModel",
                ],
            )
            kv = {r["key"]: r["value"] for r in rows}

            world_type = kv.get("WorldType") or world_type
            setting_type = world_type or kv.get("SettingType") or setting_type
            setting_kind = kv.get("SettingKind") or setting_kind
            reality_context = kv.get("RealityContext") or reality_context
            physics_model = kv.get("PhysicsModel") or physics_model

            raw_capabilities = _safe_json_loads(kv.get("SettingCapabilities"), {}) or {}
            capabilities = dict(raw_capabilities)
            physics_caps_local = _safe_json_loads(kv.get("PhysicsCaps"), {}) or {}
            infra_flags = _safe_json_loads(kv.get("InfrastructureFlags"), {}) or {}
            economy_flags = _safe_json_loads(kv.get("EconomyFlags"), {}) or {}
            technology_level = (
                kv.get("TechnologyLevel")
                or raw_capabilities.get("technology")
                or technology_level
            )
            setting_era = kv.get("SettingEra") or raw_capabilities.get("era") or setting_era
            world_model_raw = _safe_json_loads(kv.get("WorldModel"), {}) or {}
            world_model_context = _normalize_world_model_metadata(
                world_model_raw,
                {
                    "kind": setting_kind,
                    "type": setting_type,
                    "reality_context": reality_context,
                },
            )

            if technology_level and "technology" not in capabilities:
                capabilities["technology"] = technology_level
            if setting_era and "era" not in capabilities:
                capabilities.setdefault("era", setting_era)

            capabilities.update(
                _derive_feasibility_caps(
                    infra_flags,
                    physics_caps_local,
                    economy_flags,
                    setting_era,
                )
            )

            caps_loaded_flag = bool(raw_capabilities) or bool(infra_flags) or bool(physics_caps_local) or bool(economy_flags) or bool(technology_level) or bool(setting_era)
            scene = _safe_json_loads(kv.get("CurrentScene"), {}) or {}
            location_name = kv.get("CurrentLocation") or (scene.get("location") if isinstance(scene, dict) else None)
            established_impossibilities = _safe_json_loads(kv.get("EstablishedImpossibilities"), []) or []

            rules = await conn.fetch(
                """
                SELECT condition, effect
                FROM GameRules
                WHERE user_id=$1 AND conversation_id=$2 AND enabled=TRUE
                """,
                user_id,
                conversation_id,
            )

            known_location_rows = await conn.fetch(
                """
                SELECT location_name FROM Locations
                WHERE user_id=$1 AND conversation_id=$2
                """,
                user_id,
                conversation_id,
            )
            known_location_names = [
                row["location_name"]
                for row in known_location_rows
                if row["location_name"]
            ]
    except Exception as e:
        logger.error(f"[FEASIBILITY] DB read failed (soft): {e}", exc_info=True)
        # Keep defaults; remain permissive

    setting_type = setting_type or "modern_realistic"
    logger.info("[FEASIBILITY] Setting "
                f"type={setting_type} kind={setting_kind} reality={reality_context} physics={physics_model}")
    _log_caps_snapshot(capabilities)
    logger.debug(f"[FEASIBILITY] Capabilities: {capabilities}")
    logger.debug(f"[FEASIBILITY] Scene keys: {list(scene.keys()) if isinstance(scene, dict) else 'n/a'}")

    fail_open = _fail_open_missing_caps(intents, capabilities if caps_loaded_flag else {})
    if fail_open:
        return fail_open

    # ---- 3) Build dynamic allow/deny sets from conversation rules/scene --------
    hard_deny_cats: Set[str] = set()
    allow_cats: Set[str] = set()

    for r in (rules or []):
        cond = (r.get("condition") or "").strip().lower()
        eff = (r.get("effect") or "").strip().lower()
        if cond.startswith("category:"):
            name = cond.split(":", 1)[1].strip()
            # explicit allows/denies ONLY from DB rules (dynamic)
            if any(k in eff for k in ("prohibit", "forbid", "cannot", "not allowed", "disallow", "deny")):
                hard_deny_cats.add(name)
            if any(k in eff for k in ("allow", "permitted", "can", "enabled", "allowed")):
                allow_cats.add(name)

    scene_banned = set((scene.get("banned_categories") or []) if isinstance(scene, dict) else [])
    scene_allowed = set((scene.get("allowed_categories") or []) if isinstance(scene, dict) else [])

    # ---- 4) Soft constraints from capabilities/physics (NO hard rules) ---------
    # We compute "mismatch" categories to nudge toward ASK rather than DENY.
    # You can define your own mapping from capabilities to category clusters in CurrentRoleplay.SettingCapabilities.
    # Example capabilities you might store: { "magic":"none|limited|common", "physics":"realistic|flexible|surreal",
    #   "technology":"primitive|medieval|modern|advanced|futuristic", "supernatural":"none|hidden|known|common" }
    caps_magic = str((capabilities.get("magic") or "")).lower()
    caps_physics = str((capabilities.get("physics") or physics_model or "")).lower()
    caps_tech = str((capabilities.get("technology") or "")).lower()
    caps_supernatural = str((capabilities.get("supernatural") or "")).lower()

    # Dynamic feature flags that creators can set per conversation:
    # e.g. {"feature_flags": {"public_magic_ok": true, "sci_fi_elements_ok": false}}
    feature_flags = (capabilities.get("feature_flags") or {}) if isinstance(capabilities, dict) else {}

    # Category groups to *soft*-warn on when mismatch:
    soft_map: List[Tuple[bool, Set[str], str]] = []

    # Magic-sensitive categories
    if caps_magic in {"none", ""} and not _normalize_bool(feature_flags.get("magic_ok")):
        soft_map.append((
            True,
            {"spellcasting", "ritual_magic", "summoning", "necromancy", "ex_nihilo_conjuration", "psionics", "public_magic"},
            "magic_limited_by_setting"
        ))

    # Physics-sensitive categories
    if caps_physics in {"realistic", ""} and not _normalize_bool(feature_flags.get("loose_physics_ok")):
        soft_map.append((
            True,
            {"physics_violation", "reality_warping", "unaided_flight", "time_travel", "teleportation", "ex_nihilo_conjuration"},
            "physics_constrained"
        ))

    # Tech/scifi-sensitive categories
    if caps_tech in {"primitive", "medieval", "modern", ""} and not _normalize_bool(feature_flags.get("sci_fi_elements_ok")):
        soft_map.append((
            True,
            {"vehicle_operation_space", "ai_system_access", "drone_control", "scifi_setpiece", "vacuum_exposure", "spacewalk"},
            "tech_level_constrained"
        ))

    # Supernatural-sensitive
    if caps_supernatural in {"none", ""} and not _normalize_bool(feature_flags.get("supernatural_ok")):
        soft_map.append((
            True,
            {"undead_control", "spirit_binding", "demon_summoning", "necromancy"},
            "supernatural_constrained"
        ))

    soft_constraints_map: Dict[str, str] = {}
    for _active, cats, tag in soft_map:
        if _active:
            for c in cats:
                # only soft when *not* explicitly allowed
                if c not in allow_cats and c not in scene_allowed:
                    soft_constraints_map[c] = tag

    # ---- 5) Per-intent evaluation ---------------------------------------------
    per_intent: List[Dict[str, Any]] = []
    any_hard_block = False
    any_defer = False
    any_ask = False

    # Quick scene affordances for alternatives
    scene_npcs = (scene.get("npcs") or scene.get("present_npcs") or []) if isinstance(scene, dict) else []
    scene_items = (scene.get("items") or scene.get("available_items") or []) if isinstance(scene, dict) else []
    location_features = (scene.get("location_features") or []) if isinstance(scene, dict) else []
    if isinstance(location_features, str):
        location_features = [location_features]
    time_phase = (scene.get("time_phase") or scene.get("time_of_day") or "day") if isinstance(scene, dict) else "day"

    setting_context = {
        "kind": setting_kind,
        "type": setting_type,
        "setting_kind": setting_kind,
        "setting_type": setting_type,
        "reality_context": reality_context,
        "technology_level": technology_level,
        "setting_era": setting_era,
        "scene": scene,
        "location": {"name": location_name} if location_name else {},
        "location_features": location_features,
        "known_location_names": known_location_names,
        "world_model": world_model_context,
    }

    scene_npc_tokens = _tokenize_scene_values(scene_npcs)
    scene_item_tokens = _tokenize_scene_values(scene_items)
    location_token = str(location_name).strip().lower() if location_name else ""
    location_aliases = _location_reference_aliases(location_token, scene)
    location_context_tokens = _collect_location_context_tokens(
        scene,
        location_features,
        location_token,
    )
    known_location_tokens = _build_known_location_tokens(
        location_aliases,
        location_context_tokens,
        known_location_names,
        scene,
    )
    sterile_environment = _looks_like_sterile_environment(
        location_token, location_context_tokens
    )

    # Use only the last few impossibilities (most recent canon)
    last_imps = (established_impossibilities or [])[-12:]

    impossible_categories = set(hard_deny_cats) | scene_banned
    for imp in last_imps:
        try:
            imp_cats = set((imp or {}).get("categories", []) or [])
        except Exception:
            imp_cats = set()
        impossible_categories |= {str(cat) for cat in imp_cats if cat}
    logger.info(
        "[FEASIBILITY] impossible_categories=%s",
        sorted(impossible_categories),
    )

    def reasons_for(category_hits: Set[str]) -> List[Dict[str, str]]:
        reasons: List[Dict[str, str]] = []
        for c in sorted(category_hits):
            if c in hard_deny_cats:
                reasons.append({"rule": f"category:{c}", "reason": "Prohibited by world rule"})
            elif c in scene_banned:
                reasons.append({"rule": f"scene:{c}", "reason": "Not available in this scene"})
            else:
                # established impossibility covered elsewhere; default:
                reasons.append({"rule": f"unavailable:{c}", "reason": "Unavailable here"})
        return reasons

    for intent in intents or [{}]:
        cats = set(intent.get("categories") or [])
        if not cats:
            inferred = _infer_categories_from_text(text_l)
            if inferred:
                cats = inferred

        missing_location_tokens = await _find_unresolved_location_targets(
            intent,
            text_l,
            location_aliases,
            location_context_tokens,
            known_location_tokens,
            scene_npc_tokens,
            scene_item_tokens,
            setting_context,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        
        # --- BEGIN: Updated location resolver integration ---
        if missing_location_tokens:
            # Check if this is a place query (venue request or real-world toponym)
            is_place_query = any(
                _matches_generic_venue_request(
                    token,
                    _normalize_location_phrase(token),
                    setting_context,
                    location_context_tokens,
                    known_location_tokens,
                )
                or _looks_like_real_world_toponym(
                    token,
                    _normalize_location_phrase(token),
                    setting_context,
                )
                for token in missing_location_tokens
            )

            if is_place_query:
                try:
                    from nyx.location.router import resolve_place_or_travel  # type: ignore
                except Exception:
                    resolve_place_or_travel = None  # type: ignore

                if resolve_place_or_travel:
                    query_text = missing_location_tokens[0]
                    try:
                        try:
                            result = await resolve_place_or_travel(
                                query_text,
                                setting_context,
                                None,
                                str(user_id),
                                str(conversation_id),
                            )
                        except TypeError:
                            # Older signatures may omit store argument; fall back gracefully.
                            result = await resolve_place_or_travel(
                                query_text,
                                setting_context,
                                None,
                                str(user_id),
                                str(conversation_id),
                            )

                        result_obj = _coerce_resolve_result(result)
                        status = result_obj.status if result_obj else (
                            getattr(result, "status", None)
                            if not isinstance(result, dict)
                            else result.get("status")
                        )

                        if status in {STATUS_EXACT, STATUS_MULTIPLE, STATUS_ASK, STATUS_TRAVEL_PLAN}:
                            message = result_obj.message if result_obj else (
                                getattr(result, "message", None)
                                if not isinstance(result, dict)
                                else result.get("message")
                            )
                            operations = list(result_obj.operations if result_obj else (
                                getattr(result, "operations", None)
                                or getattr(result, "canonical_ops", None)
                                or (result.get("operations") if isinstance(result, dict) else [])
                                or (result.get("canonical_ops") if isinstance(result, dict) else [])
                            ) or [])
                            choices = list(result_obj.choices if result_obj else (
                                getattr(result, "choices", None)
                                if not isinstance(result, dict)
                                else result.get("choices")
                            ) or [])

                            candidates = list(result_obj.candidates) if result_obj else []
                            anchor = result_obj.anchor if result_obj else None
                            scope = result_obj.scope if result_obj and result_obj.scope else (
                                anchor.scope if isinstance(anchor, Anchor) and anchor.scope else "real"
                            )

                            policy_meta = resolver_policy_for_context(setting_context, anchor=anchor)
                            mint_policy = policy_meta.get("mint_policy")
                            default_planet = policy_meta.get("default_planet")

                            hierarchy_payload: Optional[Dict[str, Any]] = None
                            chosen_candidate = candidates[0] if candidates else None
                            if chosen_candidate:
                                try:
                                    async with get_db_connection_context() as conn:
                                        hierarchy_payload = await assign_hierarchy(
                                            conn,
                                            chosen_candidate,
                                            scope=scope or "real",
                                            anchor=anchor,
                                            mint_policy=mint_policy,
                                            default_planet=default_planet,
                                        )
                                except Exception:
                                    logger.exception(
                                        "[FEASIBILITY] Failed to assign hierarchy for resolver candidate",
                                        extra={"query": query_text, "status": status},
                                    )

                            candidate_previews = [
                                preview
                                for preview in (
                                    _resolver_candidate_preview(candidate)
                                    for candidate in candidates
                                )
                                if preview
                            ]

                            travel_payload = operations if status == STATUS_TRAVEL_PLAN else []

                            logger.info(
                                "[FEASIBILITY] Location resolver found results for %r -> status=%s",
                                query_text,
                                status,
                            )

                            if status == STATUS_EXACT:
                                payload = {
                                    "feasible": True,
                                    "strategy": "allow",
                                    "categories": sorted(cats),
                                    "location_resolved": True,
                                    "resolver_result": operations,
                                }
                                if hierarchy_payload:
                                    leaf = hierarchy_payload.get("leaf") or {}
                                    payload["resolver_hierarchy"] = hierarchy_payload
                                    payload["resolver_place_id"] = leaf.get("id")
                                    payload["resolver_place_key"] = leaf.get("place_key")
                                    payload["resolver_world_name"] = hierarchy_payload.get("world_name")
                                    payload["resolver_mint_policy"] = (
                                        hierarchy_payload.get("mint_policy") or mint_policy
                                    )
                                if travel_payload:
                                    payload["resolver_travel_plan"] = travel_payload
                                if candidate_previews:
                                    payload["resolver_candidates"] = candidate_previews
                                per_intent.append(payload)
                                continue

                            lead_candidates = _scene_alternatives(
                                _display_scene_values(scene_npcs),
                                _display_scene_values(scene_items),
                                _display_scene_values(location_features),
                                time_phase,
                            )

                            suggested: List[str] = []
                            for preview in candidate_previews[:3]:
                                label = preview["name"]
                                admin_path = preview.get("admin_path")
                                if admin_path:
                                    label = f"{label} — {admin_path}"
                                suggested.append(label)

                            fallback_choices = choices or lead_candidates
                            if len(suggested) < 3:
                                for option in fallback_choices:
                                    if option not in suggested:
                                        suggested.append(option)
                                    if len(suggested) >= 3:
                                        break

                            ask_payload = {
                                "feasible": False,
                                "strategy": "ask",
                                "violations": [],
                                "narrator_guidance": message or "Which one do you mean?",
                                "suggested_alternatives": suggested[:3] if suggested else fallback_choices[:3],
                                "categories": sorted(cats),
                                "location_resolved": True,
                            }
                            if hierarchy_payload:
                                leaf = hierarchy_payload.get("leaf") or {}
                                ask_payload["resolver_hierarchy"] = hierarchy_payload
                                ask_payload["resolver_place_id"] = leaf.get("id")
                                ask_payload["resolver_place_key"] = leaf.get("place_key")
                                ask_payload["resolver_world_name"] = hierarchy_payload.get("world_name")
                                ask_payload["resolver_mint_policy"] = (
                                    hierarchy_payload.get("mint_policy") or mint_policy
                                )
                            if travel_payload:
                                ask_payload["resolver_travel_plan"] = travel_payload
                            if candidate_previews:
                                ask_payload["resolver_candidates"] = candidate_previews

                            per_intent.append(ask_payload)
                            any_ask = True
                            continue
                    except Exception as e:
                        logger.debug(f"[FEASIBILITY] Location resolver failed softly: {e}")
            
            # Fall through to original resolver cache logic if resolver didn't handle it
            missing_location_phrase = _format_missing_names(missing_location_tokens)
            resolver_cache = (
                setting_context.get("location_resolver_cache", {})
                if isinstance(setting_context, dict)
                else {}
            )
            resolver_decisions = [
                _resolver_feedback_for_token(token, resolver_cache)
                for token in missing_location_tokens
            ]
            deny_decision = next(
                (d for d in resolver_decisions if d and d.get("decision") == "deny"),
                None,
            )
            ask_decision = next(
                (d for d in resolver_decisions if d and d.get("decision") == "ask"),
                None,
            )
            lead_candidates = _scene_alternatives(
                _display_scene_values(scene_npcs),
                _display_scene_values(scene_items),
                _display_scene_values(location_features),
                time_phase,
            )
            if deny_decision:
                reason_text = (
                    deny_decision.get("reason")
                    or f"{missing_location_phrase} isn't an established location right now."
                )
                logger.info(
                    "[FEASIBILITY] Hard deny - fabricated location -> %s",
                    missing_location_tokens,
                )
                per_intent.append(
                    {
                        "feasible": False,
                        "strategy": "deny",
                        "violations": [
                            {
                                "rule": "location_resolver:deny",
                                "reason": reason_text,
                            }
                        ],
                        "narrator_guidance": (
                            f"{reason_text} Stick to known locations or introduce it in-scene first."
                        ),
                        "suggested_alternatives": lead_candidates,
                        "leads": lead_candidates,
                        "categories": sorted(cats),
                    }
                )
                any_hard_block = True
            elif ask_decision:
                reason_text = ask_decision.get("reason") or (
                    f"{missing_location_phrase} needs a quick description before I can add it."
                )
                logger.info(
                    "[FEASIBILITY] Resolver ASK for location -> %s", missing_location_tokens
                )
                per_intent.append(
                    {
                        "feasible": False,
                        "strategy": "ask",
                        "violations": [
                            {
                                "rule": "location_resolver:ask",
                                "reason": reason_text,
                            }
                        ],
                        "narrator_guidance": (
                            f"{reason_text} Give me a quick sense of the place or pick one of the known options."
                        ),
                        "suggested_alternatives": lead_candidates,
                        "leads": lead_candidates,
                        "categories": sorted(cats),
                    }
                )
                any_ask = True
            else:
                reason_text = (
                    f"{missing_location_phrase} isn't an established location right now."
                )
                logger.info(
                    "[FEASIBILITY] Hard deny - fabricated location -> %s",
                    missing_location_tokens,
                )
                per_intent.append(
                    {
                        "feasible": False,
                        "strategy": "deny",
                        "violations": [
                            {
                                "rule": "location_resolver:deny",
                                "reason": reason_text,
                            }
                        ],
                        "narrator_guidance": (
                            f"{reason_text} Stick to known locations or introduce it in-scene first."
                        ),
                        "suggested_alternatives": lead_candidates,
                        "leads": lead_candidates,
                        "categories": sorted(cats),
                    }
                )
                any_hard_block = True
            continue
        # --- END: Updated location resolver integration ---

        referenced_targets = _tokenize_scene_values(intent.get("direct_object"))
        referenced_items = _tokenize_scene_values(intent.get("instruments"))
        referenced_items = {
            token
            for token in referenced_items
            if token and token not in INHERENT_INSTRUMENT_TOKENS
        }
        missing_target_tokens: List[str] = []
        for token in referenced_targets:
            if not token:
                continue
            if _is_self_reference_token(token):
                continue
            if token in scene_npc_tokens:
                continue
            if token in scene_item_tokens:
                continue
            if _is_location_reference_token(token, location_aliases):
                continue
            missing_target_tokens.append(token)

        stripped_missing_target_tokens: List[str] = []
        for token in missing_target_tokens:
            if _canonicalize_self_reference_token(token):
                continue
            normalized = _normalize_location_phrase(token)
            normalized = normalized.strip(SELF_REFERENCE_STRIP_CHARS) if normalized else ""
            if _canonicalize_self_reference_token(normalized):
                continue
            if normalized and normalized in location_aliases:
                continue
            if normalized and _is_location_reference_token(normalized, location_aliases):
                continue
            stripped_missing_target_tokens.append(token)

        missing_target_tokens = stripped_missing_target_tokens
        missing_item_tokens: List[str] = []
        for token in referenced_items:
            if not token or token in scene_item_tokens:
                continue
            if _is_location_reference_token(token, location_aliases):
                continue
            if _is_plausible_mundane_search_target(
                token,
                text_l,
                cats,
                location_token,
                location_context_tokens,
            ):
                continue
            missing_item_tokens.append(token)
        if missing_item_tokens and not sterile_environment:
            filtered_tokens = [
                token
                for token in missing_item_tokens
                if not _matches_ambient_debris_token(token)
            ]
            missing_item_tokens = filtered_tokens
        if missing_target_tokens or missing_item_tokens:
            violations: List[Dict[str, str]] = []
            missing_target_phrase = _format_missing_names(missing_target_tokens)
            missing_item_phrase = _format_missing_names(missing_item_tokens)
            if missing_target_tokens:
                violations.append({
                    "rule": "npc_absent",
                    "reason": (
                        f"No sign of {missing_target_phrase} anywhere in this scene—"
                        "I'm genuinely baffled about who you're trying to engage."
                    ),
                })
            if missing_item_tokens:
                violations.append({
                    "rule": "item_absent",
                    "reason": (
                        f"No sign of {missing_item_phrase} anywhere in this scene—"
                        "I'm genuinely baffled about what stash you're imagining."
                    ),
                })

            if missing_target_phrase and missing_item_phrase:
                narrator_guidance = (
                    f"There's no {missing_target_phrase} anywhere in sight, and no "
                    f"{missing_item_phrase} to grab—what fantasy are you chasing? "
                    "Work with the NPCs or items that are actually present."
                )
            elif missing_target_phrase:
                narrator_guidance = (
                    f"There's no {missing_target_phrase} anywhere in sight—what "
                    "fantasy are you chasing? Work with the NPCs that are actually present."
                )
            elif missing_item_phrase:
                narrator_guidance = (
                    f"There's no {missing_item_phrase} to grab—what fantasy are you "
                    "chasing? Work with the items that are actually here."
                )
            else:
                narrator_guidance = (
                    "What fantasy are you chasing? Work with the NPCs or items that are "
                    "actually present."
                )

            lead_candidates = _scene_alternatives(
                _display_scene_values(scene_npcs),
                _display_scene_values(scene_items),
                _display_scene_values(location_features),
                time_phase,
            )

            per_intent.append({
                "feasible": False,
                "strategy": "deny",
                "violations": violations,
                "narrator_guidance": narrator_guidance,
                "leads": lead_candidates,
                "suggested_alternatives": lead_candidates,
                "categories": sorted(cats),
            })
            any_hard_block = True
            continue

        # (A) Established Impossibilities (hard deny if categories overlap)
        hit_imposs = []
        if last_imps and cats:
            for imp in last_imps:
                imp_cats = set((imp or {}).get("categories", []) or [])
                if imp_cats & cats:
                    hit_imposs.append(imp)
        if hit_imposs:
            logger.info(f"[FEASIBILITY] Hard deny by EstablishedImpossibilities -> {cats}")
            per_intent.append({
                "feasible": False,
                "strategy": "deny",
                "violations": [{"rule": "established_impossibility", "reason": (hit_imposs[-1].get("reason") or "Previously established as impossible")}],
                "narrator_guidance": _compose_guidance(setting_kind, location_name, cats),
                "suggested_alternatives": _scene_alternatives(scene_npcs, scene_items, location_features, time_phase),
                "categories": sorted(cats),
            })
            any_hard_block = True
            continue

        # (B) Explicit world/scene rule bans (hard deny)
        hard_bans = (cats & (hard_deny_cats | scene_banned)) - (allow_cats | scene_allowed)
        if hard_bans:
            logger.info(f"[FEASIBILITY] Hard deny by explicit rule/scene -> {hard_bans}")
            per_intent.append({
                "feasible": False,
                "strategy": "deny",
                "violations": reasons_for(hard_bans),
                "narrator_guidance": _compose_guidance(setting_kind, location_name, hard_bans),
                "suggested_alternatives": _scene_alternatives(scene_npcs, scene_items, location_features, time_phase),
                "categories": sorted(cats),
            })
            any_hard_block = True
            continue

        # (C) Soft constraints (ASK for clarification or propose grounded rewrite)
        soft_hits = {c for c in cats if c in soft_constraints_map}
        if soft_hits:
            tag = soft_constraints_map[next(iter(soft_hits))]
            logger.info(f"[FEASIBILITY] Soft constraint (ASK) -> cats={soft_hits} tag={tag}")
            per_intent.append({
                "feasible": False,
                "strategy": "ask",
                "violations": [{"rule": tag, "reason": "May not be supported here without setup"}],
                "narrator_guidance": (
                    "That might stretch this setting. Want to adapt it to what's already present, "
                    "or describe how your character attempts it within realistic bounds?"
                ),
                "suggested_alternatives": _scene_alternatives(scene_npcs, scene_items, location_features, time_phase),
                "categories": sorted(cats),
            })
            # ASK is not a hard block; we won't set any_hard_block = True
            continue

        # (D) No issues => allow
        per_intent.append({
            "feasible": True,
            "strategy": "allow",
            "categories": sorted(cats),
        })

    if any_hard_block:
        overall = {"feasible": False, "strategy": "deny"}
    elif any_ask:
        overall = {"feasible": False, "strategy": "ask"}
    elif any_defer:
        overall = {"feasible": False, "strategy": "defer"}
    else:
        overall = {"feasible": True, "strategy": "allow"}

    # If parse failed and we had literally no signal, prefer ASK rather than deny
    if parse_error and all((not i.get("categories") for i in intents)):
        logger.info("[FEASIBILITY] Parse failed & no categories inferred -> soft ASK")
        overall = {"feasible": False, "strategy": "ask"}
        per_intent = [{
            "feasible": False,
            "strategy": "ask",
            "violations": [{"rule": "parse_error", "reason": "Unclear intent"}],
            "narrator_guidance": "I didn't quite follow that. Say it as a single, concrete action or break it into steps.",
            "suggested_alternatives": ["Describe one action you take", "Name an object in the scene you use"],
            "categories": []
        }]

    logger.info(f"[FEASIBILITY] overall={overall}")
    return {
        "overall": overall,
        "per_intent": per_intent
    }


def _infer_categories_from_text(text_l: str) -> Set[str]:
    """
    Very light, capability-gated hints — only used when the parser gives us nothing.
    We map obvious phrases to canonical categories the rest of the system understands.
    """
    mapping = [
        ({"cast", "spell", "ritual", "incantation"}, "spellcasting"),
        ({"teleport", "blink", "warp"}, "teleportation"),
        ({"time travel", "go back in time", "rewind"}, "time_travel"),
        ({"fly unaided", "take off myself", "levitate"}, "unaided_flight"),
        ({"spaceship", "rocket", "shuttle", "orbit", "space", "plasma", "laser"}, "spaceflight"),
        ({"summon", "conjure", "from nothing"}, "ex_nihilo_conjuration"),
        ({"mind control", "telepathy", "psychic"}, "psionics"),
        ({"reality warp", "bend physics"}, "physics_violation"),
    ]
    cats: Set[str] = set()
    for keys, cat in mapping:
        if any(k in text_l for k in keys):
            cats.add(cat)
    return cats


def _scene_alternatives(npcs: List[str], items: List[str], features: List[str], time_phase: str) -> List[str]:
    alts: List[str] = []
    if npcs:
        alts.append(f"approach {npcs[0]} for help")
    if items:
        alts.append(f"examine the {items[0]} more closely")
    if features:
        alts.append(f"investigate the {features[0]}")
    if time_phase == "night":
        alts.append("wait until dawn for better visibility")
    elif time_phase == "day":
        alts.append("survey the area for a grounded advantage")
    # keep it tight
    return alts[:3]


def _compose_guidance(setting_kind: str, location_name: Optional[str], blocking: Set[str]) -> str:
    loc = f" in {location_name}" if location_name else ""
    # Pick one dominant block to speak to
    if {"spaceflight", "orbital_travel", "spaceship_piloting"} & blocking:
        return f"This isn’t a spacefaring world{loc}; the sky stays near and no engines like that exist here."
    if {"spellcasting", "ritual_magic", "conjuration", "summoning"} & blocking:
        return f"Whatever power hums here, it isn’t magic you can wield{loc}."
    if {"physics_violation", "reality_warping", "ex_nihilo_conjuration"} & blocking:
        return f"The world keeps its seams tight{loc}; physics don’t bend that way."
    if {"unaided_flight"} & blocking:
        return f"Gravity still owns the air{loc}; you can’t take wing without help."
    if {"time_travel", "teleportation"} & blocking:
        return f"Time and distance refuse shortcuts{loc}."
    return f"This setting{loc} doesn’t support that move; try a grounded approach."

# ---------- Helper/guard additions (safe to paste once) ----------
if "_normalize_location_phrase" not in globals():
    def _normalize_location_phrase(s: Optional[str]) -> str:
        return (s or "").strip().lower()

if "_is_modern_or_realistic_setting" not in globals():
    def _is_modern_or_realistic_setting(ctx: Dict[str, Any]) -> bool:
        kind = str(ctx.get("kind") or ctx.get("setting_kind") or "").lower()
        stype = str(ctx.get("type") or ctx.get("setting_type") or "").lower()
        caps = ctx.get("capabilities") or {}
        tech = str(caps.get("technology") or "").lower()
        physics = str(caps.get("physics") or "").lower()
        # "realistic" / "modern" in kind/type OR modernish tech with realistic/flexible physics
        return any([
            "modern" in kind or "realistic" in kind,
            "realistic" in stype or "modern" in stype,
            tech in {"modern", "advanced", "futuristic"} and physics in {"realistic", "flexible"},
        ])

if "_matches_generic_venue_request" not in globals():
    def _matches_generic_venue_request(token: str, normalized: str,
                                       setting_context: Dict[str, Any],
                                       location_context_tokens: Set[str],
                                       known_location_tokens: Set[str]) -> bool:
        if not normalized:
            return False
        if normalized in (known_location_tokens or set()):
            return False
        generic = {
            "cafe","coffee","coffee shop","restaurant","diner","bar","pub","tavern","inn",
            "pharmacy","drugstore","apothecary","hospital","clinic","urgent care",
            "store","shop","market","bazaar","kiosk","booth","stall",
            "bookstore","library",
            "bank","atm",
            "park","plaza","square","station","airport","harbor","harbour","port","dock",
            "gas","gas station","petrol","fuel","charging","ev charger","charger",
            "hotel","motel","hostel",
            "police","police station","precinct","fire station","post office",
            "school","university","college",
            "gym","pool",
        }
        brands = {
            "starbucks","dunkin","mcdonalds","subway","kfc","burger king","taco bell",
            "walmart","target","costco","tesco","sainsbury","aldi","lidl",
            "walgreens","cvs","rite aid","boots",
            "ikea","apple store","best buy","home depot","lowe's","lowes",
            "shell","bp","chevron","exxon","exxonmobil","7-eleven","7 eleven","7‑eleven",
        }
        n = normalized
        return (n in brands) or (n in generic) or any(kw in n for kw in (brands | generic))

if "_looks_like_real_world_toponym" not in globals():
    def _looks_like_real_world_toponym(token: str, normalized: str, setting_context: Dict[str, Any]) -> bool:
        if not normalized:
            return False
        t = normalized
        street_suffixes = {" st", " ave", " ave.", " rd", " rd.", " road", " street", " boulevard", " blvd", " blvd.",
                           " lane", " ln", " ln.", " drive", " dr", " dr.", " court", " ct", " ct.",
                           " way", " square", " sq", " sq."}
        place_suffixes = {" park", " plaza", " mall", " center", " centre", " station",
                          " airport", " university", " college", " hospital", " museum"}
        if any(t.endswith(sfx) for sfx in (street_suffixes | place_suffixes)):
            return True
        big_cities = {
            "new york","los angeles","san francisco","seattle","chicago","boston","miami","houston","dallas","atlanta",
            "london","paris","berlin","rome","madrid","barcelona","amsterdam","brussels","vienna","prague",
            "tokyo","osaka","kyoto","seoul","beijing","shanghai","shenzhen","hong kong","singapore",
            "sydney","melbourne","auckland",
            "toronto","vancouver","montreal",
            "mumbai","delhi","bangalore","bengaluru","kolkata","chennai","karachi",
            "rio de janeiro","são paulo","sao paulo","mexico city","buenos aires",
        }
        if t in big_cities:
            return True
        if " city" in t or " town" in t or " village" in t or "," in t:
            return True
        # loose postal/zip-ish look (letters+digits+separator)
        if any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t) and any(x in t for x in (" ", "-", ",")):
            return True
        return False

if "LOCATION_REFERENCE_KEYWORDS" not in globals():
    LOCATION_REFERENCE_KEYWORDS = {"here","there","nearby","around","this place","that place","the area"}

if "SELF_REFERENCE_STRIP_CHARS" not in globals():
    SELF_REFERENCE_STRIP_CHARS = "\"'.,:;!?`"

if "_looks_like_sterile_environment" not in globals():
    def _looks_like_sterile_environment(location_token: str, context_tokens: Set[str]) -> bool:
        hints = {h.lower() for h in (STERILE_ENVIRONMENT_HINTS or set())}
        token_hit = location_token in hints if location_token else False
        return token_hit or bool((context_tokens or set()) & hints)

if "_matches_ambient_debris_token" not in globals():
    def _matches_ambient_debris_token(token: str) -> bool:
        t = (token or "").strip().lower()
        if not t:
            return False
        if t in AMBIENT_DEBRIS_KEYWORDS:
            return True
        if t.endswith("s") and t[:-1] in AMBIENT_DEBRIS_CANONICALS:
            return True
        return False

if "_display_scene_values" not in globals():
    def _display_scene_values(values: Any) -> List[str]:
        if not values:
            return []
        if isinstance(values, (list, tuple, set)):
            out: List[str] = []
            for v in values:
                if isinstance(v, dict):
                    for k in ("name","display_name","label","title","item_name","npc_name"):
                        if v.get(k):
                            out.append(str(v.get(k)))
                            break
                    else:
                        out.append(str(v))
                else:
                    out.append(str(v))
            return out
        return [str(values)]
