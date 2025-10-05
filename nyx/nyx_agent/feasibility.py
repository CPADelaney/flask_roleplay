# nyx/nyx_agent/feasibility.py
"""
Dynamic feasibility system that learns what's possible/impossible in each unique setting.
Maintains reality consistency without hard-coded rules or repetitive responses.
"""

import json
import random
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from agents import Agent, Runner
from nyx.nyx_agent.context import NyxContext
from db.connection import get_db_connection_context
from logic.action_parser import parse_action_intents
import asyncio

from nyx.feas.actions.mundane import evaluate_mundane
from nyx.feas.archetypes.modern_baseline import ModernBaseline
from nyx.feas.archetypes.roman_empire import RomanEmpire
from nyx.feas.archetypes.underwater_scifi import UnderwaterSciFi
from nyx.feas.capabilities import merge_caps
from nyx.feas.context import build_affordance_index

import logging
logger = logging.getLogger(__name__)


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
SAFE_BASELINE: Set[str] = {"mundane_action", "dialogue", "movement", "social", "trade"}


SETTING_DETECTION_TIMEOUT_SECONDS = 8


ARCHETYPE_REGISTRY = {
    "modern_baseline": ModernBaseline,
    "roman_empire": RomanEmpire,
    "underwater_scifi": UnderwaterSciFi,
}


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
    """Evaluate mundane intents using the archetype-driven affordance engine."""

    indices: List[int] = []
    for idx, intent in enumerate(intents):
        categories = {str(cat).lower() for cat in intent.get("categories", [])}
        if "mundane_action" in categories or "trade" in categories:
            indices.append(idx)

    if not indices:
        return None

    current_scene = await _load_current_scene(nyx_ctx)
    scene_snapshot = _build_scene_snapshot(setting_context, current_scene)
    player_snapshot = _build_player_snapshot(setting_context)
    archetypes = _infer_world_archetypes(setting_context)
    world_caps = load_world_caps({"archetypes": archetypes})
    affordance_index = build_affordance_index(world_caps, scene_snapshot, player_snapshot)

    overrides: Dict[int, Dict[str, Any]] = {}
    for idx in indices:
        evaluation = evaluate_mundane(intents[idx], world_caps, affordance_index, scene_snapshot, player_snapshot)
        overrides[idx] = _mundane_result_to_intent_entry(intents[idx], evaluation)

    return {
        "overrides": overrides,
        "overall": _combine_overall(list(overrides.values())),
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
    # Parse the intended actions
    intents = await parse_action_intents(user_input)
    
    # Load comprehensive setting context
    setting_context = await _load_comprehensive_context(nyx_ctx)
    _log_caps_snapshot(setting_context.get("capabilities"))

    caps_loaded = bool(setting_context.get("caps_loaded"))
    fail_open = _fail_open_missing_caps(
        intents,
        setting_context.get("capabilities") if caps_loaded else {},
    )
    if fail_open:
        return fail_open

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

        return {"overall": {"feasible": False, "strategy": "defer"}, "per_intent": per_intent}

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
        
        return {
            "overall": {"feasible": False, "strategy": "deny"},
            "per_intent": per_intent
        }

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

        return {
            "overall": mundane_eval.get("overall", _combine_overall(per_intent)),
            "per_intent": per_intent,
        }

    # Full AI-powered assessment for nuanced cases
    full_result = await _full_dynamic_assessment(nyx_ctx, user_input, intents, setting_context)

    if mundane_eval and mundane_eval.get("overrides"):
        full_result = _apply_mundane_overrides(full_result, intents, mundane_eval["overrides"])

    return full_result

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
        "established_impossibilities": [],
        "established_possibilities": [],
        "narrative_history": [],
        "environment_desc": "",
        "setting_name": "",
        "stat_modifiers": {}
    }
    
    async with get_db_connection_context() as conn:
        # Get comprehensive setting information from new_game_agent storage
        setting_keys = [
            'WorldType', 'SettingType', 'SettingKind', 'SettingCapabilities',
            'RealityContext', 'PhysicsModel', 'PhysicsCaps', 'EnvironmentDesc',
            'CurrentSetting', 'SettingStatModifiers', 'EnvironmentHistory',
            'ScenarioName', 'CurrentLocation', 'CurrentTime', 'InfrastructureFlags',
            'EconomyFlags', 'TechnologyLevel', 'SettingEra',
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
        technology_level = context.get("technology_level")
        setting_era = context.get("setting_era")
        technology_level_from_db = False
        setting_era_from_db = False

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

        if world_type_value:
            context["type"] = world_type_value
        context["technology_level"] = technology_level or context.get("technology_level")
        context["setting_era"] = setting_era or context.get("setting_era")
        context["infrastructure_flags"] = infra_flags
        context["economy_flags"] = economy_flags
        context["physics_caps"] = physics_caps_local

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
            context["location"].update(scene_data.get("location", {}))
            context["available_items"] = scene_data.get("items", [])
            context["present_entities"] = scene_data.get("npcs", [])
            
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
    
    run = await Runner.run(ALTERNATIVE_GENERATOR_AGENT, json.dumps(context))
    
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
        location_name = scene.get("location") or await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentLocation'
        """, nyx_ctx.user_id, nyx_ctx.conversation_id)
        
        if location_name:
            location = await conn.fetchrow("""
                SELECT notable_features, hidden_aspects, description
                FROM Locations
                WHERE user_id=$1 AND conversation_id=$2 AND location_name=$3
            """, nyx_ctx.user_id, nyx_ctx.conversation_id, location_name)
            
            if location:
                scene["location_features"] = location.get("notable_features", []) or []
                scene["location_description"] = location.get("description", "")
        
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
            Runner.run(SETTING_DETECTIVE_AGENT, json.dumps(context)),
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
    caps_loaded_flag = False
    world_type: Optional[str] = None
    infra_flags: Dict[str, Any] = {}
    economy_flags: Dict[str, Any] = {}
    physics_caps_local: Dict[str, Any] = {}
    technology_level: Optional[str] = None
    setting_era: Optional[str] = None

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

    # Quick scene affordances for alternatives
    scene_npcs = (scene.get("npcs") or scene.get("present_npcs") or []) if isinstance(scene, dict) else []
    scene_items = (scene.get("items") or scene.get("available_items") or []) if isinstance(scene, dict) else []
    location_features = (scene.get("location_features") or []) if isinstance(scene, dict) else []
    if isinstance(location_features, str):
        location_features = [location_features]
    time_phase = (scene.get("time_phase") or scene.get("time_of_day") or "day") if isinstance(scene, dict) else "day"

    scene_npc_tokens = _tokenize_scene_values(scene_npcs)
    scene_item_tokens = _tokenize_scene_values(scene_items)
    location_token = str(location_name).strip().lower() if location_name else ""

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

        referenced_targets = _tokenize_scene_values(intent.get("direct_object"))
        referenced_items = _tokenize_scene_values(intent.get("instruments"))
        missing_target_tokens = [
            token
            for token in referenced_targets
            if token and token not in scene_npc_tokens and token not in scene_item_tokens and token != location_token
        ]
        missing_item_tokens = [
            token
            for token in referenced_items
            if token and token not in scene_item_tokens
        ]
        if missing_target_tokens or missing_item_tokens:
            violations: List[Dict[str, str]] = []
            if missing_target_tokens:
                violations.append({
                    "rule": "npc_absent",
                    "reason": f"{', '.join(sorted(set(missing_target_tokens)))} not present in the scene",
                })
            if missing_item_tokens:
                violations.append({
                    "rule": "item_absent",
                    "reason": f"{', '.join(sorted(set(missing_item_tokens)))} not available here",
                })

            lead_candidates = _scene_alternatives(
                _display_scene_values(scene_npcs),
                _display_scene_values(scene_items),
                _display_scene_values(location_features),
                time_phase,
            )

            per_intent.append({
                "feasible": False,
                "strategy": "defer",
                "violations": violations,
                "narrator_guidance": "Those targets aren't here. Try acting with the NPCs or items that are present.",
                "leads": lead_candidates,
                "suggested_alternatives": lead_candidates,
                "categories": sorted(cats),
            })
            any_defer = True
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
            # ASK is not a hard block; we won’t set any_hard_block = True
            continue

        # (D) No issues => allow
        per_intent.append({
            "feasible": True,
            "strategy": "allow",
            "categories": sorted(cats),
        })

    if any_hard_block:
        overall = {"feasible": False, "strategy": "deny"}
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
            "narrator_guidance": "I didn’t quite follow that. Say it as a single, concrete action or break it into steps.",
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
