# logic/rule_enforcement.py
"""
Refactored Punishment Enforcement with Nyx-style gating and enhanced robustness.

Features:
- Bypasses unrecognized rule conditions gracefully to prevent crashes.
- Persistent, per-conversation cooldown + de-duplication.
- Feasibility/scene/stimuli-aware tiering ('ambient' | 'soft' | 'major').
- Deterministic RNG per (user_id, conversation_id).
- Canonical writes via LoreSystem when possible (guarded import).
- Safe fallbacks if LoreSystem or extended columns are unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import asyncpg
from quart import Blueprint, jsonify, request

from db.connection import get_db_connection_context

# Optional LoreSystem for canonical writes (guarded import)
try:
    from lore.core.lore_system import LoreSystem  # type: ignore
except ImportError:
    LoreSystem = None  # type: ignore

rule_enforcement_bp = Blueprint("rule_enforcement_bp", __name__)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Persistent Gate (per conversation)
# -------------------------------------------------------------------

@dataclass
class _PersistentGate:
    last_any_turn: int = -1_000_000
    last_any_ts: float = 0.0
    seen_effect_hashes: Dict[str, float] = field(default_factory=dict)

_GATE_STATE: Dict[Tuple[int, int], _PersistentGate] = {}
_GATE_LOCKS: Dict[Tuple[int, int], asyncio.Lock] = {}
_GATE_TOUCH: Dict[Tuple[int, int], float] = {}
_GATE_TTL_SECONDS = 60 * 60 * 12  # 12h idle GC

def _gate_for(uid: int, cid: int) -> Tuple[_PersistentGate, asyncio.Lock]:
    key = (uid, cid)
    st = _GATE_STATE.setdefault(key, _PersistentGate())
    lk = _GATE_LOCKS.setdefault(key, asyncio.Lock())
    _GATE_TOUCH[key] = time.time()
    # Opportunistic GC to clean up old conversation data
    cutoff = time.time() - _GATE_TTL_SECONDS
    for k, ts in list(_GATE_TOUCH.items()):
        if ts < cutoff:
            _GATE_TOUCH.pop(k, None)
            _GATE_STATE.pop(k, None)
            _GATE_LOCKS.pop(k, None)
    return st, lk

def purge_punishment_gate_state(user_id: int, conversation_id: int):
    """Explicitly purge gate state for a conversation."""
    key = (user_id, conversation_id)
    for d in (_GATE_STATE, _GATE_LOCKS, _GATE_TOUCH):
        d.pop(key, None)

# -------------------------------------------------------------------
# Trigger Configuration
# -------------------------------------------------------------------

@dataclass
class PunishmentTriggerConfig:
    allowed_scene_tags: Set[str] = field(default_factory=lambda: {
        "discipline", "punishment", "humiliation", "tease", "aftercare",
        "fetish", "kink", "dominance", "submissive", "intimate"
    })
    intent_markers: Set[str] = field(default_factory=lambda: {
        "punishment", "discipline", "humiliation", "dominance", "kink"
    })
    min_turn_gap: int = 2
    min_seconds_between: float = 45.0
    soft_prob_base: float = 0.10
    soft_prob_high: float = 0.22
    stimuli_affinity: Dict[str, Set[str]] = field(default_factory=lambda: {
        "humiliation": {"snicker", "laugh", "eye_roll", "dismissive"},
        "obedience": {"order", "command", "kneel", "obedience"},
        "implements": {"paddle", "cane", "collar", "leash"},
    })
    severity_threshold_level: int = 3

# -------------------------------------------------------------------
# Gate / Context
# -------------------------------------------------------------------

class PunishmentContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = int(user_id)
        self.conversation_id = int(conversation_id)
        self.cfg = PunishmentTriggerConfig()
        self._pg, self._lock = _gate_for(self.user_id, self.conversation_id)
        seed = hashlib.sha1(f"{self.user_id}:{self.conversation_id}".encode()).hexdigest()[:16]
        self._rng = random.Random(int(seed, 16))

    def _scene_allows(self, scene_tags: Iterable[str]) -> bool:
        return bool(set(scene_tags or []) & self.cfg.allowed_scene_tags)

    def _intents_allow(self, feas: Optional[dict]) -> bool:
        if not isinstance(feas, dict): return False
        per_intent = feas.get("per_intent") or []
        for intent in per_intent:
            if set((intent or {}).get("tags", [])) & self.cfg.intent_markers:
                return True
        overall = feas.get("overall") or {}
        return bool(set(overall.get("tags", [])) & self.cfg.intent_markers)

    def _cooldowns_ok(self, turn_idx: int, now: float) -> bool:
        if (turn_idx - self._pg.last_any_turn) < self.cfg.min_turn_gap: return False
        if (now - self._pg.last_any_ts) < self.cfg.min_seconds_between: return False
        return True

    async def _mark_emit(self, turn_idx: int):
        now = time.time()
        async with self._lock:
            self._pg.last_any_turn = turn_idx
            self._pg.last_any_ts = now

    def decide_tier(self, meta: Dict[str, Any], violations_count: int, severity_hint_level: int) -> Optional[str]:
        turn_idx = int(meta.get("turn_index", 0))
        scene_tags = ((meta.get("scene") or {}).get("tags")) or meta.get("scene_tags") or []
        feas = meta.get("feasibility") or {}
        stimuli = set(meta.get("stimuli", []))
        now = time.time()

        if not self._cooldowns_ok(turn_idx, now): return None
        relevant = self._scene_allows(scene_tags) or self._intents_allow(feas)

        if violations_count <= 0:
            if relevant:
                prob = self.cfg.soft_prob_base if severity_hint_level < self.cfg.severity_threshold_level else self.cfg.soft_prob_high
                return "soft" if self._rng.random() < prob else None
            return None

        if relevant:
            return "major" if severity_hint_level >= self.cfg.severity_threshold_level else "soft"
        
        for vocab in self.cfg.stimuli_affinity.values():
            if stimuli & vocab: return "soft"

        return None

# -------------------------------------------------------------------
# Helpers: Player/NPC stats with graceful fallbacks
# -------------------------------------------------------------------

async def get_player_stats(player_name: str = "Chase", user_id: Optional[int] = None, conversation_id: Optional[int] = None) -> Dict[str, int]:
    try:
        async with get_db_connection_context() as conn:
            row = None
            if user_id is not None and conversation_id is not None:
                try:
                    row = await conn.fetchrow("SELECT corruption, confidence, willpower, obedience, dependency, lust, mental_resilience, physical_endurance FROM PlayerStats WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3", int(user_id), int(conversation_id), player_name)
                except Exception: row = None
            if not row:
                row = await conn.fetchrow("SELECT corruption, confidence, willpower, obedience, dependency, lust, mental_resilience, physical_endurance FROM PlayerStats WHERE player_name=$1", player_name)
        if not row: return {}
        return {
            "Corruption": row["corruption"], "Confidence": row["confidence"], "Willpower": row["willpower"], 
            "Obedience": row["obedience"], "Dependency": row["dependency"], "Lust": row["lust"], 
            "Mental Resilience": row["mental_resilience"], "Physical Endurance": row["physical_endurance"],
        }
    except Exception as e:
        logger.error(f"Error fetching player stats: {e}", exc_info=True)
        return {}

async def get_npc_stats(npc_name: Optional[str] = None, npc_id: Optional[int] = None, user_id: Optional[int] = None, conversation_id: Optional[int] = None) -> Dict[str, Any]:
    try:
        async with get_db_connection_context() as conn:
            row = None
            if npc_id is not None and user_id is not None and conversation_id is not None:
                try:
                    row = await conn.fetchrow("SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3", int(user_id), int(conversation_id), int(npc_id))
                except Exception: row = None
            if not row:
                if npc_id is not None: row = await conn.fetchrow("SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity FROM NPCStats WHERE npc_id=$1", int(npc_id))
                elif npc_name: row = await conn.fetchrow("SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity FROM NPCStats WHERE npc_name=$1", npc_name)
        if not row: return {}
        return {
            "NPCName": row["npc_name"], "Dominance": row["dominance"], "Cruelty": row["cruelty"],
            "Closeness": row["closeness"], "Trust": row["trust"], "Respect": row["respect"],
            "Intensity": row["intensity"]
        }
    except Exception as e:
        logger.error(f"Error fetching NPC stats: {e}", exc_info=True)
        return {}

# -------------------------------------------------------------------
# Condition parsing/eval (REFACTORED SECTION)
# -------------------------------------------------------------------

def parse_condition(condition_str: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parses a condition string into a structured format.
    Refactored to gracefully handle and flag unrecognized parts.
    """
    cond = (condition_str or "").strip().lower()
    if " and " in cond:
        logic_op, parts = "AND", [p.strip() for p in cond.split(" and ")]
    elif " or " in cond:
        logic_op, parts = "OR", [p.strip() for p in cond.split(" or ")]
    else:
        logic_op, parts = "SINGLE", [cond]

    parsed_list: List[Dict[str, Any]] = []
    for part in parts:
        if not part:
            continue

        # Pattern 1: Check for token bundle syntax (e.g., "category:theft | hazard:social")
        if "|" in part or ":" in part:
            bundle_tokens: List[Dict[str, str]] = []
            is_valid_bundle = True
            for token in part.split("|"):
                token = token.strip()
                if ":" in token:
                    prefix, value = token.split(":", 1)
                    prefix, value = prefix.strip(), value.strip()
                    if prefix and value:
                        bundle_tokens.append({"kind": prefix, "value": value})
                    else: # Malformed token like "category:"
                        is_valid_bundle = False; break
                else: # Token without a colon
                    is_valid_bundle = False; break
            
            if is_valid_bundle and bundle_tokens:
                parsed_list.append({"type": "token_bundle", "tokens": bundle_tokens})
                continue

        # Pattern 2: Check for numeric comparison syntax (e.g., "willpower < 30")
        tokens = part.split()
        if len(tokens) == 3 and tokens[1] in (">", ">=", "<", "<=", "=="):
            stat_name, operator, threshold_str = tokens
            try:
                threshold = int(threshold_str)
                parsed_list.append({
                    "type": "numeric",
                    "stat": stat_name.title(), # Normalize to TitleCase to match stats dict
                    "operator": operator,
                    "threshold": threshold,
                })
                continue
            except (ValueError, TypeError):
                # Falls through to unrecognized if threshold is not a number
                pass

        # Fallback: If it matches no known patterns, it's unrecognized.
        logger.warning(f"Unrecognized condition part: '{part}'. This rule will be bypassed.")
        parsed_list.append({"type": "unrecognized", "original_text": part})
        
    return (logic_op, parsed_list)

def evaluate_condition(
    logic_op: str,
    parsed_conditions: List[Dict[str, Any]],
    stats_dict: Dict[str, int],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Evaluates parsed conditions.
    Refactored to safely ignore any parts marked as "unrecognized".
    """
    if not parsed_conditions:
        return False

    results: List[bool] = []
    token_sets = _extract_feasibility_sets(metadata)

    for condition in parsed_conditions:
        ctype = condition.get("type")

        if ctype == "unrecognized":
            results.append(False)  # Safely bypass by evaluating to False
            continue

        elif ctype == "numeric":
            stat_name = condition.get("stat", "")
            operator = condition.get("operator", "")
            threshold = int(condition.get("threshold", 0))
            actual_value = stats_dict.get(stat_name, 0)
            
            outcome = False
            if operator == ">": outcome = actual_value > threshold
            elif operator == ">=": outcome = actual_value >= threshold
            elif operator == "<": outcome = actual_value < threshold
            elif operator == "<=": outcome = actual_value <= threshold
            elif operator == "==": outcome = actual_value == threshold
            results.append(outcome)

        elif ctype == "token_bundle":
            tokens = condition.get("tokens") or []
            outcome = _evaluate_token_bundle(tokens, token_sets)
            results.append(outcome)
        
        else: # Safeguard for any other unexpected type
            results.append(False)

    if not results: return False
    if logic_op == "AND": return all(results)
    if logic_op == "OR": return any(results)
    return results[0] # SINGLE case

# ... (The rest of the file, including _normalize_token_values, _extract_feasibility_sets,
# _evaluate_token_bundle, apply_effect, and all other functions, remains unchanged)
# I will include them here for completeness.

def _normalize_token_values(value: Any) -> Set[str]:
    normalized: Set[str] = set()
    if isinstance(value, str):
        val = value.strip().lower()
        if val: normalized.add(val)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for item in value:
            normalized.update(_normalize_token_values(item))
    return normalized

def _extract_feasibility_sets(metadata: Optional[Dict[str, Any]]) -> Dict[str, Set[str]]:
    sets: Dict[str, Set[str]] = {"category": set(), "hazard": set(), "magic": set()}
    if not isinstance(metadata, dict): return sets
    feasibility = metadata.get("feasibility")
    if not isinstance(feasibility, dict): return sets
    def add_from_violations(violations: Any):
        if not isinstance(violations, Iterable) or isinstance(violations, (str, bytes, dict)): return
        for entry in violations:
            if not isinstance(entry, dict): continue
            rule = str(entry.get("rule", "")).strip().lower()
            if not rule or ":" not in rule: continue
            prefix, value = rule.split(":", 1)
            prefix, value = prefix.strip(), value.strip()
            if prefix in sets and value: sets[prefix].add(value)
    def add_from_entry(entry: Dict[str, Any]):
        sets["category"].update(_normalize_token_values(entry.get("categories")))
        sets["hazard"].update(_normalize_token_values(entry.get("hazards")))
        sets["magic"].update(_normalize_token_values(entry.get("magic")))
        sets["magic"].update(_normalize_token_values(entry.get("magic_requirements")))
        add_from_violations(entry.get("violations"))
    per_intent = feasibility.get("per_intent")
    if isinstance(per_intent, Iterable) and not isinstance(per_intent, (str, bytes, dict)):
        for entry in per_intent:
            if isinstance(entry, dict): add_from_entry(entry)
    add_from_violations(feasibility.get("violations"))
    overall = feasibility.get("overall")
    if isinstance(overall, dict): add_from_entry(overall)
    capabilities = feasibility.get("capabilities")
    if isinstance(capabilities, dict):
        sets["category"].update(_normalize_token_values(capabilities.get("categories")))
        sets["hazard"].update(_normalize_token_values(capabilities.get("hazards")))
        sets["magic"].update(_normalize_token_values(capabilities.get("magic")))
        sets["magic"].update(_normalize_token_values(capabilities.get("magic_flags")))
    return sets

def _evaluate_token_bundle(tokens: List[Dict[str, str]], token_sets: Dict[str, Set[str]]) -> bool:
    if not tokens: return False
    category_hits, hazard_hits, magic_hits = token_sets.get("category", set()), token_sets.get("hazard", set()), token_sets.get("magic", set())
    for token in tokens:
        kind, value = str(token.get("kind", "")).strip().lower(), str(token.get("value", "")).strip().lower()
        if not kind or not value: continue
        if kind == "category" and value in category_hits: return True
        if kind == "hazard" and (value in hazard_hits or value in category_hits): return True
        if kind == "magic" and (value in magic_hits or value in category_hits): return True
    return False




def evaluate_condition(
    logic_op: str,
    parsed_conditions: List[Dict[str, Any]],
    stats_dict: Dict[str, int],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    results: List[bool] = []
    token_sets = _extract_feasibility_sets(metadata)

    for condition in parsed_conditions:
        ctype = condition.get("type")
        if ctype == "numeric":
            stat_name = condition.get("stat", "")
            operator = condition.get("operator", "")
            threshold = int(condition.get("threshold", 0))
            actual_value = stats_dict.get(stat_name, 0)
            if operator == ">":
                outcome = actual_value > threshold
            elif operator == ">=":
                outcome = actual_value >= threshold
            elif operator == "<":
                outcome = actual_value < threshold
            elif operator == "<=":
                outcome = actual_value <= threshold
            elif operator == "==":
                outcome = actual_value == threshold
            else:
                outcome = False
            results.append(outcome)
        elif ctype == "token_bundle":
            tokens = condition.get("tokens") or []
            outcome = _evaluate_token_bundle(tokens, token_sets)
            results.append(outcome)
        else:
            results.append(False)

    if logic_op == "AND":
        return all(results)
    if logic_op == "OR":
        return any(results)
    if logic_op == "SINGLE":
        return results[0] if results else False
    return False

# -------------------------------------------------------------------
# Effect application (uses LoreSystem if available)
# -------------------------------------------------------------------

async def _upsert_player_obedience_min(
    player_name: str, min_value: int,
    user_id: Optional[int], conversation_id: Optional[int]
) -> Optional[int]:
    try:
        if LoreSystem:
            ls = await LoreSystem.get_instance(int(user_id or 0), int(conversation_id or 0))
            await ls.propose_and_enact_change(
                ctx=None,
                entity_type="PlayerStats",
                entity_identifier={
                    "player_name": player_name,
                    "user_id": user_id,
                    "conversation_id": conversation_id
                },
                updates={"obedience": f"GREATEST(obedience, {int(min_value)})"},
                reason="Punishment enforcement: raise obedience minimum",
            )
            return None
        else:
            async with get_db_connection_context() as conn:
                q = "UPDATE PlayerStats SET obedience = GREATEST(obedience, $2) WHERE player_name=$1 RETURNING obedience"
                if user_id is not None and conversation_id is not None:
                    try:
                        q = """
                            UPDATE PlayerStats
                            SET obedience = GREATEST(obedience, $4)
                            WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
                            RETURNING obedience
                        """
                        return await conn.fetchval(q, int(user_id), int(conversation_id), player_name, int(min_value))
                    except Exception:
                        pass
                return await conn.fetchval(q, player_name, int(min_value))
    except Exception as e:
        logger.error(f"obedience min upsert failed: {e}", exc_info=True)
        return None

async def _set_player_obedience(
    player_name: str, value: int,
    user_id: Optional[int], conversation_id: Optional[int]
) -> Optional[int]:
    try:
        if LoreSystem:
            ls = await LoreSystem.get_instance(int(user_id or 0), int(conversation_id or 0))
            await ls.propose_and_enact_change(
                ctx=None,
                entity_type="PlayerStats",
                entity_identifier={
                    "player_name": player_name,
                    "user_id": user_id,
                    "conversation_id": conversation_id
                },
                updates={"obedience": f"{int(value)}"},
                reason="Punishment enforcement: force obedience",
            )
            return None
        else:
            async with get_db_connection_context() as conn:
                q = "UPDATE PlayerStats SET obedience=$2 WHERE player_name=$1 RETURNING obedience"
                if user_id is not None and conversation_id is not None:
                    try:
                        q = """
                            UPDATE PlayerStats
                            SET obedience=$4
                            WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
                            RETURNING obedience
                        """
                        return await conn.fetchval(q, int(user_id), int(conversation_id), player_name, int(value))
                    except Exception:
                        pass
                return await conn.fetchval(q, player_name, int(value))
    except Exception as e:
        logger.error(f"set obedience failed: {e}", exc_info=True)
        return None

async def _bump_npc_cruelty(
    npc_id: int, delta: int,
    user_id: Optional[int], conversation_id: Optional[int]
) -> Optional[int]:
    try:
        if LoreSystem:
            ls = await LoreSystem.get_instance(int(user_id or 0), int(conversation_id or 0))
            await ls.propose_and_enact_change(
                ctx=None,
                entity_type="NPCStats",
                entity_identifier={"npc_id": int(npc_id), "user_id": user_id, "conversation_id": conversation_id},
                updates={"cruelty": f"LEAST(cruelty + {int(delta)}, 100)"},
                reason="Punishment enforcement: cruelty intensifies",
            )
            return None
        else:
            async with get_db_connection_context() as conn:
                q = "UPDATE NPCStats SET cruelty=LEAST(cruelty+$2,100) WHERE npc_id=$1 RETURNING cruelty"
                if user_id is not None and conversation_id is not None:
                    try:
                        q = """
                            UPDATE NPCStats
                            SET cruelty=LEAST(cruelty+$4,100)
                            WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                            RETURNING cruelty
                        """
                        return await conn.fetchval(q, int(user_id), int(conversation_id), int(npc_id), int(delta))
                    except Exception:
                        pass
                return await conn.fetchval(q, int(npc_id), int(delta))
    except Exception as e:
        logger.error(f"npc cruelty bump failed: {e}", exc_info=True)
        return None

# -------------------------------------------------------------------
# Effect application (uses LoreSystem if available)
# -------------------------------------------------------------------

async def apply_effect(
    effect_str: str,
    player_name: str,
    npc_id: Optional[int] = None,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parse and apply an effect. Returns telemetry + optional generated scenario.
    """
    result = {
        "message": "",
        "statUpdates": {},
        "effectApplied": effect_str,
        "npcUsed": npc_id
    }
    effect_lower = (effect_str or "").lower()

    try:
        # Basic canonical updates by keyword
        if effect_lower.startswith("locks independent choices"):
            val = await _upsert_player_obedience_min(player_name, 80, user_id, conversation_id)
            if val is not None:
                result["statUpdates"]["obedience"] = val

        elif effect_lower.startswith("total compliance"):
            val = await _set_player_obedience(player_name, 100, user_id, conversation_id)
            if val is not None:
                result["statUpdates"]["obedience"] = val

        elif effect_lower.startswith("npc cruelty intensifies") and npc_id is not None:
            val = await _bump_npc_cruelty(int(npc_id), 10, user_id, conversation_id)
            if val is not None:
                result["statUpdates"]["npc_cruelty"] = val

        # Intensity tier (NPC-first, fallback to player)
        npc_stats = await get_npc_stats(npc_id=npc_id, user_id=user_id, conversation_id=conversation_id) if npc_id else {}
        npc_intensity = npc_stats.get("Intensity")
        result["npcName"] = npc_stats.get("NPCName")
        result["npcIntensity"] = npc_intensity

        intensity_range = (0, 30)
        if npc_intensity is not None:
            if npc_intensity >= 90:
                intensity_range = (90, 100)
            elif npc_intensity >= 60:
                intensity_range = (60, 90)
            elif npc_intensity >= 30:
                intensity_range = (30, 60)
        else:
            pstats = await get_player_stats(player_name, user_id, conversation_id)
            corr, obed = pstats.get("Corruption", 0), pstats.get("Obedience", 0)
            if corr >= 90 or obed >= 90:
                intensity_range = (90, 100)
            elif corr >= 60 or obed >= 60:
                intensity_range = (60, 90)
            elif corr >= 30 or obed >= 30:
                intensity_range = (30, 60)

        # Pull tier row (optional)
        chosen_example = None
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT tier_name, key_features, activity_examples, permanent_effects
                FROM IntensityTiers
                WHERE range_min=$1 AND range_max=$2
            """, intensity_range[0], intensity_range[1])
        if row:
            result["intensityTier"] = row["tier_name"]
            try:
                examples = json.loads(row["activity_examples"] or "[]")
            except Exception:
                examples = []
            if examples:
                chosen_example = random.choice(examples)

        # Optional LLM scenario for punishments
        if "punishment" in effect_lower:
            try:
                from logic.chatgpt_integration import get_openai_client
                client = get_openai_client()
                sys = (
                    "You are a punishment scenario generator for a roleplay game.\n"
                    f"Player intensity tier: {result.get('intensityTier','unknown')}.\n"
                    f"Effect cue: '{effect_str}'. Keep output short."
                )
                gpt_resp = await client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": "Generate a concise punishment setup."}
                    ],
                )
                scenario = gpt_resp.choices[0].message.content.strip()
                result["punishmentScenario"] = scenario
            except Exception as e:
                if chosen_example:
                    result["punishmentScenario"] = f"(From IntensityTier) {chosen_example}"
                else:
                    result["punishmentScenario"] = f"(Generator unavailable: {e})"
        elif chosen_example and "punishmentScenario" not in result:
            result["punishmentScenario"] = f"(From IntensityTier) {chosen_example}"

        # Memory / telemetry text
        result["memoryLog"] = f"Effect triggered: {effect_str}. (Tier: {result.get('intensityTier','unknown')})"
        return result

    except Exception as e:
        logger.error(f"Error applying effect: {e}", exc_info=True)
        return result

# -------------------------------------------------------------------
# Putting it together with gating
# -------------------------------------------------------------------

async def enforce_all_rules_on_player(
    player_name: str = "Chase",
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate rules, decide punishment tier with gating, and optionally apply.
    Returns: {"tier": "ambient|soft|major|none", "triggered": [...], "telemetry": {...}}
    """
    meta = dict(metadata or {})
    uid = int(user_id) if user_id is not None else 0
    cid = int(conversation_id) if conversation_id is not None else 0
    ctx = PunishmentContext(uid, cid)

    try:
        # 1) Load player stats and rules
        pstats = await get_player_stats(player_name, user_id, conversation_id)
        async with get_db_connection_context() as conn:
            rows: List[asyncpg.Record]
            if user_id is not None and conversation_id is not None:
                scoped_queries = [
                    (
                        """
                        SELECT condition, effect
                          FROM GameRules
                         WHERE enabled IS DISTINCT FROM FALSE
                           AND (
                                (user_id=$1 AND conversation_id=$2)
                                OR (user_id IS NULL AND conversation_id IS NULL)
                           )
                        """,
                        (uid, cid),
                    ),
                    (
                        """
                        SELECT condition, effect
                          FROM GameRules
                         WHERE (user_id=$1 AND conversation_id=$2)
                            OR (user_id IS NULL AND conversation_id IS NULL)
                        """,
                        (uid, cid),
                    ),
                    ("SELECT condition, effect FROM GameRules", ()),
                ]
            else:
                scoped_queries = [
                    (
                        "SELECT condition, effect FROM GameRules WHERE enabled IS DISTINCT FROM FALSE",
                        (),
                    ),
                    ("SELECT condition, effect FROM GameRules", ()),
                ]

            rows = []
            for query, params in scoped_queries:
                try:
                    if params:
                        rows = await conn.fetch(query, *params)
                    else:
                        rows = await conn.fetch(query)
                    break
                except asyncpg.UndefinedColumnError:
                    continue

        # 2) Evaluate all rules
        matches: List[Tuple[str, str]] = []
        for r in rows:
            condition_str, effect_str = r["condition"], r["effect"]
            logic_op, parsed = parse_condition(condition_str)
            if evaluate_condition(logic_op, parsed, pstats, metadata=meta):
                matches.append((condition_str, effect_str))

        violations_count = len(matches)

        # 3) Compute a severity hint (0..4) from player context
        sev = 0
        probe = max(pstats.get("Obedience", 0), pstats.get("Corruption", 0), pstats.get("Dependency", 0))
        if probe >= 90:
            sev = 4
        elif probe >= 70:
            sev = 3
        elif probe >= 50:
            sev = 2
        elif probe >= 30:
            sev = 1

        # 4) Decide tier
        tier = ctx.decide_tier(meta, violations_count=violations_count, severity_hint_level=sev)
        if not tier:
            return {"tier": "none", "triggered": [], "telemetry": {"violations": violations_count, "severity": sev}}

        # Mark emission now that we act
        await ctx._mark_emit(int(meta.get("turn_index", 0)))

        # 5) Apply according to tier
        triggered: List[Dict[str, Any]] = []
        if tier == "ambient":
            if matches:
                first = matches[0]
                triggered.append({
                    "condition": first[0],
                    "effect": first[1],
                    "outcome": {"hint": "A warning presence lingers; consequences may follow."}
                })
        elif tier == "soft":
            for condition_str, effect_str in matches[:2]:
                out = await apply_effect(effect_str, player_name, user_id=user_id, conversation_id=conversation_id)
                if isinstance(out.get("punishmentScenario"), str):
                    out["punishmentScenario"] = out["punishmentScenario"][:240]
                triggered.append({"condition": condition_str, "effect": effect_str, "outcome": out})
        else: # major
            for condition_str, effect_str in matches:
                out = await apply_effect(effect_str, player_name, user_id=user_id, conversation_id=conversation_id)
                triggered.append({"condition": condition_str, "effect": effect_str, "outcome": out})

        return {
            "tier": tier,
            "triggered": triggered,
            "telemetry": {
                "violations": violations_count,
                "severity": sev,
                "scene_tags": meta.get("scene_tags", []),
                "stimuli": meta.get("stimuli", []),
            }
        }

    except Exception as e:
        logger.error(f"Error enforcing rules: {e}", exc_info=True)
        return {"tier": "none", "triggered": [], "error": str(e)}

# -------------------------------------------------------------------
# Blueprint Route
# -------------------------------------------------------------------

@rule_enforcement_bp.route("/enforce_rules", methods=["POST"])
async def enforce_rules_route():
    data = await request.get_json() or {}
    player_name = data.get("player_name", "Chase")
    user_id = data.get("user_id")
    conversation_id = data.get("conversation_id")
    metadata = data.get("metadata", {})

    result = await enforce_all_rules_on_player(
        player_name=player_name,
        user_id=user_id,
        conversation_id=conversation_id,
        metadata=metadata
    )
    return jsonify(result), 200

# Public API
__all__ = [
    "rule_enforcement_bp", "enforce_all_rules_on_player",
    "parse_condition", "evaluate_condition",
    "get_player_stats", "get_npc_stats",
    "apply_effect", "purge_punishment_gate_state",
]
