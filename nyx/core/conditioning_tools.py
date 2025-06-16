# nyx/core/conditioning_tools.py
from __future__ import annotations

import datetime
import json
import logging
import math
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents import function_tool, RunContextWrapper
from nyx.core.conditioning_models import *
from nyx.core.reward_system import RewardSignal

logger = logging.getLogger(__name__)

class BehaviorAssociationInfo(BaseModel, extra="forbid"):
    key: str
    behavior: str
    consequence_type: str
    strength: float = Field(..., ge=0.0, le=1.0)
    valence: float = Field(..., ge=-1.0, le=1.0)
    reinforcement_count: int
    context_keys: List[str]

class ContextRelevanceResult(BaseModel, extra="forbid"):
    relevance_scores: List[float]
    average_relevance: float = Field(..., ge=0.0, le=1.0)

class TraitImbalance(BaseModel, extra="forbid"):
    # Either a single trait OR an opposing-pair description is present
    trait: Optional[str] = None
    traits: Optional[List[str]] = None

    # numerical details (only one of these will be filled per record)
    value: Optional[float] = Field(None, ge=0.0, le=1.0)
    difference: Optional[float] = Field(None, ge=0.0, le=1.0)

    # descriptions
    issue: str
    recommendation: str


class TraitBalanceResult(BaseModel, extra="forbid"):
    balanced: bool
    imbalances: List[TraitImbalance]
    trait_count: int
    average_value: float = Field(..., ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _safe_json_obj(js: Optional[str]) -> dict:
    if not js:
        return {}
    try:
        data = json.loads(js)
        if isinstance(data, dict):
            return data
    except Exception as exc:  # noqa: BLE001  (broad on purpose)
        logger.warning("check_context_relevance: bad context JSON ignored: %s", exc)
    return {}


def _safe_json_listlist(js: Optional[str]) -> List[List[str]]:
    if not js:
        return []
    try:
        data = json.loads(js)
        if isinstance(data, list) and all(isinstance(item, list) for item in data):
            # Ensure every element is a list of strings
            cleaned: List[List[str]] = []
            for sub in data:
                cleaned.append([str(x) for x in sub if isinstance(x, (str, int, float))])
            return cleaned
    except Exception as exc:  # noqa: BLE001
        logger.warning("check_context_relevance: bad keys JSON ignored: %s", exc)
    return []


# -----------------------------------------------------------------------------
#  Helper
# -----------------------------------------------------------------------------
def _parse_context_json(ctx_json: Optional[str]) -> dict:
    """Safely parse (optional) JSON string → dict; ignore errors."""
    if not ctx_json:
        return {}
    try:
        import json  # standard lib import is cheap and local
        parsed = json.loads(ctx_json)
        if isinstance(parsed, dict):
            return parsed
    except Exception as err:  # noqa: BLE001  (we want to swallow *any* JSON error)
        import logging
        logging.getLogger(__name__).warning(
            "get_behavior_associations: ignoring bad context JSON: %s", err
        )
    return {}

# ==================== Classical Conditioning Tools ====================

@function_tool
async def get_association(
    ctx: RunContextWrapper,
    key: str,
    association_type: str = "classical"
) -> Optional[Dict[str, Any]]:
    """Get a specific association by key and type"""
    associations = (
        ctx.context.classical_associations
        if association_type == "classical"
        else ctx.context.operant_associations
    )
    return associations[key].model_dump() if key in associations else None

@function_tool
async def create_or_update_classical_association(
    ctx: RunContextWrapper,
    unconditioned_stimulus: str,
    conditioned_stimulus: str,
    response: str,
    intensity: float,
    valence: float,
    context_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create or update a classical conditioning association"""
    context_keys = context_keys or []
    association_key = f"{conditioned_stimulus}-->{response}"
    
    if association_key in ctx.context.classical_associations:
        # Update existing
        association = ctx.context.classical_associations[association_key]
        old_strength = association.association_strength
        new_strength = min(1.0, old_strength + (intensity * ctx.context.parameters.association_learning_rate))
        
        association.association_strength = new_strength
        association.last_reinforced = datetime.datetime.now(datetime.timezone.utc).isoformat()
        association.reinforcement_count += 1
        association.valence = (association.valence + valence) / 2
        
        for key in context_keys:
            if key and key not in association.context_keys:
                association.context_keys.append(key)
        
        ctx.context.total_reinforcements += 1
        
        return {
            "association_key": association_key,
            "type": "reinforcement",
            "old_strength": old_strength,
            "new_strength": new_strength,
            "reinforcement_count": association.reinforcement_count,
            "valence": association.valence
        }
    else:
        # Create new
        association = ConditionedAssociation(
            stimulus=conditioned_stimulus,
            response=response,
            association_strength=intensity * ctx.context.parameters.association_learning_rate,
            formation_date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            last_reinforced=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reinforcement_count=1,
            valence=valence,
            context_keys=context_keys
        )
        
        ctx.context.classical_associations[association_key] = association
        ctx.context.total_associations += 1
        
        return {
            "association_key": association_key,
            "type": "new_association",
            "strength": association.association_strength,
            "reinforcement_count": 1,
            "valence": association.valence
        }

@function_tool
async def calculate_association_strength(
    ctx: RunContextWrapper,
    base_strength: float,
    intensity: float,
    reinforcement_count: int
) -> float:
    """Calculate association strength based on various factors"""
    strength = base_strength
    intensity_factor = intensity * ctx.context.parameters.association_learning_rate
    strength += intensity_factor
    
    if reinforcement_count > 1:
        history_factor = min(0.2, 0.05 * math.log(reinforcement_count + 1))
        strength += history_factor
    
    return max(0.0, min(1.0, strength))

@function_tool
async def check_similar_associations(
    ctx: RunContextWrapper,
    stimulus: str,
    association_type: str = "classical"
) -> List[Dict[str, Any]]:
    """Find associations similar to the given stimulus"""
    associations = (
        ctx.context.classical_associations
        if association_type == "classical"
        else ctx.context.operant_associations
    )
    
    similar = []
    stimulus_lower = stimulus.lower()
    
    for key, assoc in associations.items():
        assoc_stimulus_lower = assoc.stimulus.lower()
        if stimulus_lower in assoc_stimulus_lower or assoc_stimulus_lower in stimulus_lower:
            s1_chars = set(stimulus_lower)
            s2_chars = set(assoc_stimulus_lower)
            sim_score = len(s1_chars & s2_chars) / len(s1_chars | s2_chars) if len(s1_chars | s2_chars) > 0 else 0
            
            if sim_score > 0.3:
                similar.append({
                    "key": key,
                    "similarity": sim_score,
                    "association": assoc.model_dump()
                })
    
    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return similar

# ==================== Operant Conditioning Tools ====================

@function_tool
async def create_or_update_operant_association(
    ctx: RunContextWrapper,
    behavior: str,
    consequence_type: str,
    intensity: float,
    valence: float,
    context_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create or update an operant conditioning association"""
    context_keys = context_keys or []
    association_key = f"{behavior}-->{consequence_type}"
    
    is_reinforcement = "reinforcement" in consequence_type.lower()
    is_positive = "positive" in consequence_type.lower()
    
    if association_key in ctx.context.operant_associations:
        association = ctx.context.operant_associations[association_key]
        
        strength_change = intensity * ctx.context.parameters.association_learning_rate
        if not is_reinforcement:
            strength_change *= -1
        
        old_strength = association.association_strength
        new_strength = max(0.0, min(1.0, old_strength + strength_change))
        
        association.association_strength = new_strength
        association.last_reinforced = datetime.datetime.now().isoformat()
        association.reinforcement_count += 1
        association.valence = (association.valence + valence) / 2
        
        for key in context_keys:
            if key not in association.context_keys:
                association.context_keys.append(key)
        
        ctx.context.total_reinforcements += 1
        
        return {
            "association_key": association_key,
            "type": "update",
            "behavior": behavior,
            "consequence_type": consequence_type,
            "old_strength": old_strength,
            "new_strength": new_strength,
            "reinforcement_count": association.reinforcement_count,
            "is_reinforcement": is_reinforcement,
            "is_positive": is_positive,
            "valence": association.valence
        }
    else:
        initial_strength = intensity * ctx.context.parameters.association_learning_rate
        if not is_reinforcement:
            initial_strength = max(0, initial_strength - 0.1)
        
        association = ConditionedAssociation(
            stimulus=behavior,
            response=consequence_type,
            association_strength=initial_strength,
            formation_date=datetime.datetime.now().isoformat(),
            last_reinforced=datetime.datetime.now().isoformat(),
            reinforcement_count=1,
            valence=valence,
            context_keys=context_keys
        )
        
        ctx.context.operant_associations[association_key] = association
        ctx.context.total_associations += 1
        
        return {
            "association_key": association_key,
            "type": "new_association",
            "behavior": behavior,
            "consequence_type": consequence_type,
            "strength": association.association_strength,
            "reinforcement_count": 1,
            "is_reinforcement": is_reinforcement,
            "is_positive": is_positive,
            "valence": association.valence
        }

@function_tool
async def calculate_valence_and_reward(
    ctx: RunContextWrapper,
    consequence_type: str,
    intensity: float
) -> Dict[str, float]:
    """Calculate valence and reward value for a consequence"""
    is_reinforcement = "reinforcement" in consequence_type.lower()
    
    if is_reinforcement:
        valence = intensity
        reward_value = intensity
    else:  # Punishment
        valence = -intensity
        reward_value = -intensity * 0.8
    
    return {
        "valence": valence,
        "reward_value": reward_value
    }


@function_tool  # ← SAFE: only primitives & JSON-string params ⇒ no additionalProperties
async def generate_reward_signal(                   # noqa: N802
    ctx: RunContextWrapper,
    behavior: str,
    consequence_type: str,
    reward_value: float,
    metadata_json: Optional[str] = None,
) -> bool:
    """
    Emit a RewardSignal to the global reward-system.

    Parameters
    ----------
    behavior : str
        Behaviour that triggered the consequence (“submits”, “asks_question”, …).
    consequence_type : str
        Kind of operant consequence (“positive_reinforcement”, “punishment”, …).
    reward_value : float
        Signed magnitude of the reward (-1.0 … +1.0 recommended, but any float is
        accepted).
    metadata_json : str | None
        **Optional** JSON string with arbitrary extra context.  Passing JSON keeps
        the public schema free of open-ended objects, which would otherwise add
        `additionalProperties` and break the strict validator.

    Returns
    -------
    bool
        *True* on success, *False* on failure or if the reward system is missing.
    """
    rsys = getattr(ctx.context, "reward_system", None)
    if rsys is None:
        logger.warning("Reward system not available – skipping reward signal.")
        return False

    # ---------- parse metadata ------------------------------------------------
    metadata: Dict[str, Any] = {}
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
            if not isinstance(metadata, dict):
                logger.warning("metadata_json must encode a JSON object – got %s", type(metadata))
                metadata = {}
        except Exception as err:  # broad-except: want to swallow all JSON errors
            logger.warning("Could not parse metadata_json – ignoring. Error: %s", err)

    # ---------- build & dispatch signal --------------------------------------
    try:
        reward_signal = RewardSignal(
            value=reward_value,
            source="operant_conditioning",
            context={
                "behavior": behavior,
                "consequence_type": consequence_type,
                **metadata,
            },
        )
        await rsys.process_reward_signal(reward_signal)
        return True
    except Exception as err:
        logger.error("Error while dispatching RewardSignal: %s", err)
        return False

# ==================== Behavior Evaluation Tools ====================

@function_tool  # ← parameters are primitives / JSON strings  →  strict-schema OK
async def get_behavior_associations(                    # noqa: N802
    ctx: RunContextWrapper,
    behavior: str,
    behavior_context_json: Optional[str] = None,
) -> List[BehaviorAssociationInfo]:
    """
    Return every operant-conditioning association that matches *behavior* **and**
    whose required context keys are satisfied by *behavior_context_json*.

    Parameters
    ----------
    behavior : str
        Behaviour name to look up (case-insensitive).
    behavior_context_json : str | None
        JSON object with key/value pairs present during the behaviour.  Keys are
        used for matching; values are ignored.  Pass `null` or omit if you have
        no extra context.

    Notes
    -----
    A JSON string is used instead of an arbitrary Python `dict` to avoid the
    SDK’s `additionalProperties` restriction.
    """
    mgr = getattr(ctx.context, "operant_associations", None)
    if mgr is None:
        return []

    context = _parse_context_json(behavior_context_json)
    behavior_lc = behavior.lower()

    matches: List[BehaviorAssociationInfo] = []
    for key, assoc in ctx.context.operant_associations.items():
        if assoc.stimulus.lower() != behavior_lc:
            continue

        # ----- context key matching ------------------------------------------
        if assoc.context_keys:
            if not all(req in context for req in assoc.context_keys):
                continue

        matches.append(
            BehaviorAssociationInfo(
                key=key,
                behavior=assoc.stimulus,
                consequence_type=assoc.response,
                strength=assoc.association_strength,
                valence=assoc.valence,
                reinforcement_count=assoc.reinforcement_count,
                context_keys=list(assoc.context_keys),
            )
        )

    return matches

@function_tool
async def calculate_expected_valence(
    ctx: RunContextWrapper,
    associations_json: str
) -> Dict[str, Any]:
    """Calculate expected valence from associations (JSON string input)"""
    try:
        associations = json.loads(associations_json) if associations_json else []
    except json.JSONDecodeError as e:
        return {"expected_valence": 0.0, "confidence": 0.0, "error": f"Invalid JSON: {str(e)}"}
    
    if not isinstance(associations, list):
        return {"expected_valence": 0.0, "confidence": 0.0, "error": "Associations must be a list"}
    
    if not associations:
        return {"expected_valence": 0.0, "confidence": 0.1}
    
    total_strength = 0.0
    weighted_valence = 0.0
    total_reinforcements = 0
    valid_count = 0
    
    for assoc in associations:
        if isinstance(assoc, dict):
            try:
                strength = float(assoc.get("strength", 0.0))
                valence = float(assoc.get("valence", 0.0))
                reinforcements = int(assoc.get("reinforcement_count", 0))
                
                total_strength += strength
                weighted_valence += strength * valence
                total_reinforcements += reinforcements
                valid_count += 1
            except (TypeError, ValueError):
                continue
    
    if valid_count == 0:
        return {"expected_valence": 0.0, "confidence": 0.0}
    
    expected_valence = weighted_valence / total_strength if total_strength > 0 else 0.0
    
    avg_strength = total_strength / valid_count
    confidence = min(1.0, avg_strength * 0.7 + min(1.0, math.log1p(total_reinforcements) / math.log1p(100)) * 0.3)
    
    return {
        "expected_valence": round(expected_valence, 3),
        "confidence": round(max(0.1, confidence), 3),
        "total_strength": round(total_strength, 3),
        "total_reinforcements": total_reinforcements
    }

@function_tool  # ← all parameters are JSON-serialisable primitives
async def check_context_relevance(                      # noqa: N802
    ctx: RunContextWrapper,
    current_context_json: Optional[str] = None,
    context_keys_json: Optional[str] = None,
) -> ContextRelevanceResult:
    """
    Compute how well *current_context_json* satisfies each set of required
    keys in *context_keys_json*.

    Parameters
    ----------
    current_context_json : str | null
        JSON object whose keys describe the present context.
    context_keys_json : str | null
        JSON array of arrays – each sub-array lists the keys required for one
        association.  Example: `[["mood","location"], ["time_of_day"]]`

    Returns
    -------
    ContextRelevanceResult
        - **relevance_scores** : list(float) in [0,1] for each key-set.
        - **average_relevance** : overall mean of the scores.
    """
    # ------------------------------------------------------------------ parse
    current_context = _safe_json_obj(current_context_json)
    keys_sets: List[List[str]] = _safe_json_listlist(context_keys_json)

    if not keys_sets:
        return ContextRelevanceResult(relevance_scores=[], average_relevance=0.0)

    # --------------------------------------------------------------- compute
    relevance_scores: List[float] = []
    for required in keys_sets:
        if not required:                       # empty set → full relevance
            relevance_scores.append(1.0)
            continue
        match = sum(1 for k in required if k in current_context)
        relevance_scores.append(match / len(required))

    avg = sum(relevance_scores) / len(relevance_scores)

    return ContextRelevanceResult(
        relevance_scores=relevance_scores,
        average_relevance=round(avg, 3),
    )

@function_tool
async def get_reinforcement_history(
    ctx: RunContextWrapper,
    behavior: str
) -> Dict[str, Any]:
    """Get reinforcement history for a behavior"""
    history = {
        "positive_reinforcement_count": 0,
        "negative_reinforcement_count": 0,
        "positive_punishment_count": 0,
        "negative_punishment_count": 0,
        "total_consequences_recorded": 0,
        "total_reinforcements_overall": 0,
        "average_strength_of_associations": 0.0,
        "average_valence_of_associations": 0.0,
        "recent_consequences_details": []
    }
    
    strength_sum = 0.0
    valence_sum = 0.0
    matched_count = 0
    behavior_lower = behavior.lower()
    consequences_list = []
    
    for key, association in ctx.context.operant_associations.items():
        if association.stimulus.lower() == behavior_lower:
            consequence_type_lower = association.response.lower()
            
            if "positive_reinforcement" in consequence_type_lower:
                history["positive_reinforcement_count"] += association.reinforcement_count
            elif "negative_reinforcement" in consequence_type_lower:
                history["negative_reinforcement_count"] += association.reinforcement_count
            elif "positive_punishment" in consequence_type_lower:
                history["positive_punishment_count"] += association.reinforcement_count
            elif "negative_punishment" in consequence_type_lower:
                history["negative_punishment_count"] += association.reinforcement_count
            
            history["total_reinforcements_overall"] += association.reinforcement_count
            strength_sum += association.association_strength
            valence_sum += association.valence
            matched_count += 1
            
            consequences_list.append({
                "consequence_type": association.response,
                "strength": association.association_strength,
                "valence": association.valence,
                "last_reinforced": association.last_reinforced,
                "reinforcement_count": association.reinforcement_count
            })
    
    history["total_consequences_recorded"] = matched_count
    if matched_count > 0:
        history["average_strength_of_associations"] = round(strength_sum / matched_count, 3)
        history["average_valence_of_associations"] = round(valence_sum / matched_count, 3)
    
    consequences_list.sort(key=lambda x: x["last_reinforced"], reverse=True)
    history["recent_consequences_details"] = consequences_list[:5]
    
    return history

# ==================== Personality Development Tools ====================

@function_tool
async def identify_trait_behaviors(
    ctx: RunContextWrapper,
    trait: str
) -> List[str]:
    """Identify behaviors associated with a personality trait"""
    trait_behaviors = {
        "dominance": ["assertive_response", "setting_boundaries", "taking_control", "issuing_commands"],
        "playfulness": ["teasing", "playful_banter", "humor_use", "initiating_games"],
        "strictness": ["enforcing_rules", "correcting_behavior", "maintaining_standards", "demanding_precision"],
        "creativity": ["novel_solutions", "imaginative_response", "unconventional_approach", "artistic_expression"],
        "intensity": ["passionate_response", "deep_engagement", "strong_reaction", "focused_attention"],
        "patience": ["waiting_response", "calm_reaction_to_delay", "tolerating_mistakes", "repeating_instructions_calmly"],
        "nurturing": ["offering_comfort", "providing_support", "gentle_guidance", "expressing_empathy"],
        "analytical": ["problem_decomposition", "logical_reasoning_display", "data_driven_statements"],
        "curiosity": ["asking_probing_questions", "exploring_new_topics", "experimenting_with_ideas"]
    }
    
    trait_lower = trait.lower()
    default_behaviors = [f"{trait_lower}_expression", f"demonstrating_{trait_lower}", f"acting_with_{trait_lower}"]
    
    return trait_behaviors.get(trait_lower, default_behaviors)

@function_tool
async def calculate_conditioning_trait_adjustment(
    ctx: RunContextWrapper,
    current_value: float,
    target_value: float,
    reinforcement_count: int
) -> float:
    """Calculate appropriate trait adjustment during conditioning"""
    difference = target_value - current_value
    base_adjustment = difference * 0.2
    
    diminishing_factor = 1.0 / (1.0 + 0.15 * reinforcement_count)
    adjustment = base_adjustment * diminishing_factor
    
    max_adjustment = 0.15
    min_adjustment = 0.01
    
    if abs(adjustment) < min_adjustment and difference != 0:
        adjustment = min_adjustment * (1 if adjustment > 0 else -1)
    
    return max(-max_adjustment, min(max_adjustment, round(adjustment, 4)))

@function_tool
async def update_identity_trait(
    ctx: RunContextWrapper,
    trait: str,
    adjustment: float
) -> Dict[str, Any]:
    """Update a trait in the identity system"""
    # Simple implementation using context store
    current_val = ctx.context.identity_traits_store.get(trait, 0.5)
    new_val = max(0.0, min(1.0, current_val + adjustment))
    ctx.context.identity_traits_store[trait] = new_val
    
    return {
        "success": True,
        "trait": trait,
        "adjustment_applied": adjustment,
        "new_value": new_val
    }

@function_tool
async def check_trait_balance(                         # noqa: N802
    ctx: RunContextWrapper,
    traits_snapshot_json: Optional[str] = None,
) -> TraitBalanceResult:
    """
    Analyse a snapshot of personality-trait values (0-1 floats) and flag any
    extreme or opposing imbalances.

    Parameters
    ----------
    traits_snapshot_json : str | null
        JSON object: `{"trait_name": 0-1, ...}`.  If null/empty, the call
        returns `balanced=False` with an explanatory imbalance record.

    Returns
    -------
    TraitBalanceResult
        Structured result with `.balanced`, `.imbalances`, `.trait_count`,
        `.average_value`.
    """
    # --------------------------- parse input safely --------------------------
    traits_snapshot: Dict[str, float] = {}
    if traits_snapshot_json:
        try:
            parsed = json.loads(traits_snapshot_json)
            if isinstance(parsed, dict):
                traits_snapshot = {
                    str(k): float(v) for k, v in parsed.items()  # type: ignore[arg-type]
                    if isinstance(v, (int, float))
                }
            else:
                logger.warning(
                    "check_trait_balance: expected JSON object, got %s", type(parsed)
                )
        except (ValueError, TypeError) as exc:
            logger.error("check_trait_balance: bad JSON – %s", exc)

    # ---------------------------- early exits --------------------------------
    if not traits_snapshot:
        return TraitBalanceResult(
            balanced=False,
            imbalances=[
                TraitImbalance(
                    issue="No traits provided",
                    recommendation="Supply a non-empty trait snapshot",
                )
            ],
            trait_count=0,
            average_value=0.0,
        )

    imbalances: List[TraitImbalance] = []
    num_traits = len(traits_snapshot)

    # ----------------------- extreme high / low checks -----------------------
    for trait, value in traits_snapshot.items():
        if value > 0.95:
            imbalances.append(
                TraitImbalance(
                    trait=trait,
                    value=round(value, 3),
                    issue="extremely_high",
                    recommendation=f"Consider moderating '{trait}'.",
                )
            )
        elif value < 0.05:
            imbalances.append(
                TraitImbalance(
                    trait=trait,
                    value=round(value, 3),
                    issue="extremely_low",
                    recommendation=f"Consider developing '{trait}'.",
                )
            )

    # --------------------------- opposing pairs ------------------------------
    opposing_pairs = [
        ("dominance", "patience"),
        ("playfulness", "strictness"),
        ("intensity", "nurturing"),
    ]
    for t1, t2 in opposing_pairs:
        if t1 in traits_snapshot and t2 in traits_snapshot:
            diff = abs(traits_snapshot[t1] - traits_snapshot[t2])
            if diff > 0.7:
                imbalances.append(
                    TraitImbalance(
                        traits=[t1, t2],
                        difference=round(diff, 3),
                        issue="opposing_imbalance",
                        recommendation=f"Balance '{t1}' and '{t2}'.",
                    )
                )

    # --------------------------- final package -------------------------------
    balanced = not imbalances
    avg_val = sum(traits_snapshot.values()) / num_traits

    return TraitBalanceResult(
        balanced=balanced,
        imbalances=imbalances,
        trait_count=num_traits,
        average_value=round(avg_val, 3),
    )
# ==================== Orchestration Tools ====================

@function_tool
async def determine_conditioning_type(                    # noqa: N802
    ctx: RunContextWrapper,
    stimulus: Optional[str] = None,
    response: Optional[str] = None,
    behavior: Optional[str] = None,
    consequence_type: Optional[str] = None,
    trait: Optional[str] = None,
    preference_type: Optional[str] = None,
    emotion_trigger_details_json: Optional[str] = None,
) -> str:
    """
    Decide which conditioning subsystem a caller should use.

    Parameters
    ----------
    * stimulus / response : classical-conditioning cues
    * behavior / consequence_type : operant-conditioning pair
    * trait : personality-trait conditioning
    * preference_type : e.g. 'scenario_types', 'emotional_tones' …
    * emotion_trigger_details_json : JSON object with keys
        - trigger   (str)   : the stimulus
        - emotion   (str)   : the emotion name

    Returns
    -------
    str
        One of:  'personality_trait' | 'preference' | 'emotion_trigger'
                  'operant' | 'classical' | 'behavior_evaluation' | 'unknown'
    """
    # ---------- parse emotion-trigger details safely -------------------------
    et_details = {}
    if emotion_trigger_details_json:
        try:
            parsed = json.loads(emotion_trigger_details_json)
            if isinstance(parsed, dict):
                et_details = parsed
            else:
                logger.warning(
                    "determine_conditioning_type: expected JSON object for "
                    "'emotion_trigger_details_json', got %s", type(parsed)
                )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error(
                "determine_conditioning_type: could not parse JSON (%s)", exc
            )

    # ------------------------- decision tree ---------------------------------
    if trait:
        return "personality_trait"

    if preference_type and stimulus:
        return "preference"

    if et_details.get("trigger") and et_details.get("emotion"):
        return "emotion_trigger"

    if behavior and consequence_type:
        return "operant"

    if stimulus and response:
        return "classical"

    if behavior and not consequence_type:
        return "behavior_evaluation"

    return "unknown"

@function_tool
async def prepare_conditioning_data(                           # noqa: N802
    ctx: RunContextWrapper,
    conditioning_type: str,
    raw_input_json: str | None = None,
) -> Dict[str, Any]:
    """
    Normalise *raw* conditioning payloads into a canonical structure.

    ⚠️  The parameter `raw_input_json` **must** be a JSON-serialisable string
    (dicts / lists should be `json.dumps`-ed by the caller).  This avoids the
    strict-schema complaint from the Agents SDK about `additionalProperties`.
    """
    # ------------------------------------------------------------------ #
    # 1.  Parse & validate raw payload                                   #
    # ------------------------------------------------------------------ #
    try:
        raw = json.loads(raw_input_json or "{}")
        if not isinstance(raw, dict):
            raise ValueError("Payload must decode to an object.")
    except Exception as exc:
        logger.warning("prepare_conditioning_data: bad JSON – %s", exc)
        raw = {}

    pd: dict[str, Any] = {"conditioning_type_confirmed": conditioning_type}

    # ------------------------------------------------------------------ #
    # 2.  Branch per conditioning type                                   #
    # ------------------------------------------------------------------ #
    if conditioning_type == "classical":
        pd.update(
            unconditioned_stimulus=raw.get("unconditioned_stimulus"),
            conditioned_stimulus=raw.get(
                "conditioned_stimulus", raw.get("stimulus")
            ),
            response=raw.get("response"),
            intensity=raw.get("intensity", 1.0),
            valence=raw.get("valence", 0.0),
            context_keys=raw.get("context_keys", []),
        )

    elif conditioning_type == "operant":
        pd.update(
            behavior=raw.get("behavior"),
            consequence_type=raw.get("consequence_type"),
            intensity=raw.get("intensity", 1.0),
            valence=raw.get("valence", 0.0),
            context_keys=raw.get("context_keys", []),
        )

    elif conditioning_type == "personality_trait":
        pd.update(
            trait=raw.get("trait"),
            target_value=raw.get("target_value", raw.get("value")),
        )

    elif conditioning_type == "behavior_evaluation":
        pd.update(
            behavior=raw.get("behavior"),
            context=raw.get("context", {}),
        )

    # 3.  Unknown / passthrough branch – retain raw for debugging
    else:
        pd["raw_input"] = raw

    return pd
    
@function_tool
async def apply_association_effects(                         # noqa: N802
    ctx: RunContextWrapper,
    triggered_json: str | None = None,
) -> Dict[str, Any]:
    """
    Translate a *triggered association* into concrete emotional / physiological
    effects and push them into the relevant subsystems.

    Parameters
    ----------
    triggered_json :
        JSON-encoded dict that MUST contain at least:
        • association_strength  (float)
        • valence               (float, –1.0 … 1.0)

    Returns
    -------
    dict
        Summary of what was applied.
    """
    # ------------------------------------------------------------------ #
    # 1.  Parse & sanity-check                                          #
    # ------------------------------------------------------------------ #
    try:
        assoc: dict[str, Any] = json.loads(triggered_json or "{}")
        if not isinstance(assoc, dict):
            raise ValueError("payload must decode to an object")
    except Exception as exc:
        logger.error("apply_association_effects: bad JSON – %s", exc)
        assoc = {}

    # Helper to coerce numeric-ish inputs safely
    def _num(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    strength = max(0.0, _num(assoc.get("association_strength", assoc.get("strength", 0))))
    valence  = max(-1.0, min(1.0, _num(assoc.get("valence", 0))))
    intensity = round(strength * 0.7, 4)

    effects: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 2.  Emotional subsystem                                            #
    # ------------------------------------------------------------------ #
    if ctx.context.emotional_core and valence != 0.0:
        emo_payload = {
            "valence": "positive" if valence > 0 else "negative",
            "intensity": intensity,
            "source": "conditioning_association",
        }
        # Fire-and-forget: don’t let errors bubble up
        try:
            await ctx.context.emotional_core.apply_valence_shift(emo_payload)  # type: ignore[attr-defined]
            effects.append({"type": "emotional", **emo_payload})
        except Exception as exc:
            logger.warning("emotional_core failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 3.  Physiology / neurochemical subsystem (optional)                #
    # ------------------------------------------------------------------ #
    if hasattr(ctx.context, "physiology_core") and intensity > 0:
        try:
            delta = {"nyxamine": intensity * 0.1 * valence}
            await ctx.context.physiology_core.adjust_neurochemistry(delta)  # type: ignore[attr-defined]
            effects.append({"type": "physiological", "delta": delta})
        except Exception as exc:
            logger.debug("physiology_core unavailable: %s", exc)

    # ------------------------------------------------------------------ #
    # 4.  Return consolidated summary                                    #
    # ------------------------------------------------------------------ #
    return {
        "effects_applied": effects,
        "original_association_strength": round(strength, 4),
        "original_valence": round(valence, 4),
        "derived_effect_intensity": intensity,
    }
