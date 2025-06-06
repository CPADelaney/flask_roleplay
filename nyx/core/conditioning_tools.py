# nyx/core/conditioning_tools.py

import datetime
import json
import logging
import math
from typing import Dict, List, Any, Optional, Union

from agents import function_tool, RunContextWrapper
from nyx.core.conditioning_models import *
from nyx.core.reward_system import RewardSignal

logger = logging.getLogger(__name__)

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

@function_tool
async def generate_reward_signal(
    ctx: RunContextWrapper,
    behavior: str,
    consequence_type: str,
    reward_value: float,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Generate a reward signal for the reward system"""
    if not ctx.context.reward_system:
        logger.warning("Reward system not available")
        return False
    
    try:
        reward_signal = RewardSignal(
            value=reward_value,
            source="operant_conditioning",
            context={
                "behavior": behavior,
                "consequence_type": consequence_type,
                **(metadata or {})
            }
        )
        
        await ctx.context.reward_system.process_reward_signal(reward_signal)
        return True
    except Exception as e:
        logger.error(f"Error generating reward signal: {e}")
        return False

# ==================== Behavior Evaluation Tools ====================

@function_tool
async def get_behavior_associations(
    ctx: RunContextWrapper,
    behavior: str,
    behavior_context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Get all associations for a specific behavior"""
    behavior_context = behavior_context or {}
    result = []
    behavior_lower = behavior.lower()
    
    for key, assoc in ctx.context.operant_associations.items():
        if assoc.stimulus.lower() == behavior_lower:
            context_match = True
            if assoc.context_keys:
                for req_key in assoc.context_keys:
                    if req_key not in behavior_context:
                        context_match = False
                        break
            
            if context_match:
                result.append({
                    "key": key,
                    "behavior": assoc.stimulus,
                    "consequence_type": assoc.response,
                    "strength": assoc.association_strength,
                    "valence": assoc.valence,
                    "reinforcement_count": assoc.reinforcement_count,
                    "context_keys": assoc.context_keys
                })
    
    return result

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

@function_tool
async def check_context_relevance(
    ctx: RunContextWrapper,
    current_context: Optional[Dict[str, Any]] = None,
    context_keys_list: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """Check relevance of context to association requirements"""
    current_context = current_context or {}
    context_keys_list = context_keys_list or []
    
    relevance_scores = []
    
    for required_keys in context_keys_list:
        if not required_keys:
            relevance_scores.append(1.0)
        else:
            matching = sum(1 for key in required_keys if key in current_context)
            relevance = matching / len(required_keys)
            relevance_scores.append(relevance)
    
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    return {
        "relevance_scores": relevance_scores,
        "average_relevance": round(avg_relevance, 3)
    }

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
async def check_trait_balance(
    ctx: RunContextWrapper,
    traits_snapshot: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Check balance of personality traits"""
    if traits_snapshot is None:
        return {
            "balanced": False,
            "imbalances": [{"issue": "No traits provided"}],
            "trait_count": 0,
            "average_value": 0.0
        }
    
    imbalances = []
    num_traits = len(traits_snapshot)
    
    if num_traits == 0:
        return {
            "balanced": True,
            "imbalances": [],
            "trait_count": 0,
            "average_value": 0.0
        }
    
    # Check extreme values
    for trait, value in traits_snapshot.items():
        if value > 0.95:
            imbalances.append({
                "trait": trait,
                "value": round(value, 3),
                "issue": "extremely_high",
                "recommendation": f"Consider moderating '{trait}'"
            })
        elif value < 0.05:
            imbalances.append({
                "trait": trait,
                "value": round(value, 3),
                "issue": "extremely_low",
                "recommendation": f"Consider developing '{trait}'"
            })
    
    # Check opposing pairs
    opposing_pairs = [
        ("dominance", "patience"),
        ("playfulness", "strictness"),
        ("intensity", "nurturing")
    ]
    
    for t1, t2 in opposing_pairs:
        if t1 in traits_snapshot and t2 in traits_snapshot:
            diff = abs(traits_snapshot[t1] - traits_snapshot[t2])
            if diff > 0.7:
                imbalances.append({
                    "traits": [t1, t2],
                    "difference": round(diff, 3),
                    "issue": "opposing_imbalance",
                    "recommendation": f"Balance {t1} and {t2}"
                })
    
    return {
        "balanced": len(imbalances) == 0,
        "imbalances": imbalances,
        "trait_count": num_traits,
        "average_value": round(sum(traits_snapshot.values()) / num_traits, 3) if num_traits > 0 else 0.0
    }

# ==================== Orchestration Tools ====================

@function_tool
async def determine_conditioning_type(
    ctx: RunContextWrapper,
    stimulus: Optional[str] = None,
    response: Optional[str] = None,
    behavior: Optional[str] = None,
    consequence_type: Optional[str] = None,
    trait: Optional[str] = None,
    preference_type: Optional[str] = None,
    emotion_trigger_details: Optional[Dict[str, Any]] = None
) -> str:
    """Determine the type of conditioning based on input parameters"""
    if trait:
        return "personality_trait"
    if preference_type and stimulus:
        return "preference"
    if emotion_trigger_details and emotion_trigger_details.get("trigger") and emotion_trigger_details.get("emotion"):
        return "emotion_trigger"
    if behavior and consequence_type:
        return "operant"
    if stimulus and response:
        return "classical"
    if behavior and not consequence_type:
        return "behavior_evaluation"
    
    return "unknown"

@function_tool
async def prepare_conditioning_data(
    ctx: RunContextWrapper,
    conditioning_type: str,
    raw_input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare data for specific conditioning type"""
    prepared_data = {"conditioning_type_confirmed": conditioning_type}
    
    if conditioning_type == "classical":
        prepared_data.update({
            "unconditioned_stimulus": raw_input_data.get("unconditioned_stimulus"),
            "conditioned_stimulus": raw_input_data.get("conditioned_stimulus", raw_input_data.get("stimulus")),
            "response": raw_input_data.get("response"),
            "intensity": raw_input_data.get("intensity", 1.0),
            "valence": raw_input_data.get("valence", 0.0),
            "context_keys": raw_input_data.get("context_keys", [])
        })
    
    elif conditioning_type == "operant":
        prepared_data.update({
            "behavior": raw_input_data.get("behavior"),
            "consequence_type": raw_input_data.get("consequence_type"),
            "intensity": raw_input_data.get("intensity", 1.0),
            "valence": raw_input_data.get("valence", 0.0),
            "context_keys": raw_input_data.get("context_keys", [])
        })
    
    elif conditioning_type == "personality_trait":
        prepared_data.update({
            "trait": raw_input_data.get("trait"),
            "target_value": raw_input_data.get("target_value", raw_input_data.get("value"))
        })
    
    elif conditioning_type == "behavior_evaluation":
        prepared_data.update({
            "behavior": raw_input_data.get("behavior"),
            "context": raw_input_data.get("context", {})
        })
    
    return prepared_data

@function_tool
async def apply_association_effects(
    ctx: RunContextWrapper,
    triggered_association: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply physiological/emotional effects from a triggered association"""
    strength = float(triggered_association.get("association_strength", triggered_association.get("strength", 0.0)))
    valence = float(triggered_association.get("valence", 0.0))
    intensity = strength * 0.7
    
    effects_applied = []
    
    # Emotional effects (simplified)
    if ctx.context.emotional_core and valence != 0.0:
        try:
            if valence > 0.1:
                effects_applied.append({
                    "type": "emotional",
                    "details": "Positive emotional response",
                    "valence": "positive",
                    "intensity": round(intensity, 3)
                })
            elif valence < -0.1:
                effects_applied.append({
                    "type": "emotional",
                    "details": "Negative emotional response",
                    "valence": "negative",
                    "intensity": round(intensity, 3)
                })
        except Exception as e:
            logger.error(f"Error applying emotional effects: {e}")
    
    return {
        "effects_applied": effects_applied,
        "original_association_strength": round(strength, 3),
        "original_valence": round(valence, 3),
        "derived_effect_intensity": round(intensity, 3)
    }
