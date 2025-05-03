# nyx/core/conditioning_maintenance.py

import asyncio
import logging
import datetime
import traceback
import json
import math
from typing import Dict, List, Any, Optional, Tuple

from agents import Agent, Runner, trace, function_tool, RunContextWrapper, ModelSettings, handoff
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MaintenanceTask(BaseModel):
    """Schema for maintenance tasks"""
    task_type: str = Field(..., description="Type of maintenance task")
    priority: float = Field(..., description="Priority level (0.0-1.0)")
    entity_id: str = Field(..., description="ID of entity to maintain")
    scheduled_time: str = Field(..., description="Scheduled execution time")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")

class MaintenanceRecommendation(BaseModel):
    """Schema for maintenance recommendations"""
    recommendation_type: str = Field(..., description="Type of recommendation")
    entity_type: str = Field(..., description="Type of entity (association, trait, etc.)")
    entity_id: str = Field(..., description="ID of entity")
    action: str = Field(..., description="Recommended action")
    reasoning: str = Field(..., description="Reasoning for recommendation")
    priority: float = Field(..., description="Priority level (0.0-1.0)")

class BalanceAnalysisOutput(BaseModel):
    """Output schema for personality balance analysis"""
    # CHANGE: Make all fields Optional and provide default=None
    is_balanced: Optional[bool] = Field(default=None, description="Whether personality is balanced")
    imbalances: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detected imbalances")
    trait_recommendations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Trait recommendations")
    behavior_recommendations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Behavior recommendations")
    balance_score: Optional[float] = Field(default=None, description="Overall balance score (0.0-1.0)")
    analysis: Optional[str] = Field(default=None, description="Analysis of personality balance")

class AssociationConsolidationOutput(BaseModel):
    """Output schema for association consolidation"""
    # CHANGE: Make all fields Optional and provide default=None
    consolidations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Consolidations performed")
    removed_keys: Optional[List[str]] = Field(default=None, description="Association keys removed")
    strengthened_keys: Optional[List[str]] = Field(default=None, description="Association keys strengthened")
    efficiency_gain: Optional[float] = Field(default=None, description="Efficiency gain from consolidation (0.0-1.0)")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for consolidations")

class MaintenanceSummaryOutput(BaseModel):
    """Output schema for maintenance run summary."""
    # CHANGE: Make tasks_performed Optional as well
    tasks_performed: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, # Keep default_factory for initialization convenience
        description="Tasks performed during the maintenance run."
    )
    time_taken_seconds: Optional[float] = Field(
        default=None,
        description="Total time taken for the maintenance run in seconds."
    )
    associations_modified: Optional[int] = Field(
        default=None,
        description="Number of conditioning associations modified (created, deleted, updated)."
    )
    traits_adjusted: Optional[int] = Field(
        default=None,
        description="Number of personality traits adjusted."
    )
    extinction_count: Optional[int] = Field(
        default=None,
        description="Number of associations removed due to extinction."
    )
    improvements: Optional[List[str]] = Field(
        default=None,
        description="List of key improvements or changes made to the system during the run."
    )
    next_maintenance_recommendation: Optional[str] = Field(
        default=None,
        description="Recommendation for the focus of the next maintenance run."
    )


class MaintenanceContext:
    """Context object for conditioning maintenance operations"""
    
    def __init__(self, conditioning_system, reward_system=None):
        self.conditioning_system = conditioning_system
        self.reward_system = reward_system
        
        # Maintenance configuration
        self.maintenance_interval_hours = 24  # Run daily
        self.extinction_threshold = 0.05  # Remove associations below this strength
        self.reinforcement_threshold = 0.3  # Reinforce associations above this strength
        self.consolidation_interval_days = 7  # Consolidate weekly
        
        # Maintenance stats
        self.last_maintenance_time = None
        self.maintenance_history = []
        self.max_history_entries = 30
        
        # Background task
        self.maintenance_task = None
        
        # Trace ID for linking traces
        self.trace_group_id = f"maintenance_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"


# ===========================================
# FUNCTION TOOLS OUTSIDE THE CLASS
# ===========================================

@function_tool
async def _analyze_trait_distribution(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Analyze the distribution of personality traits
    
    Returns:
        Analysis of trait distribution
    """
    # Get all conditioning associations
    classical_associations = ctx.context.conditioning_system.classical_associations
    operant_associations = ctx.context.conditioning_system.operant_associations
    
    # Count trait references
    trait_counts = {}
    trait_strengths = {}
    
    # Extract traits from classical associations
    for key, association in classical_associations.items():
        # Check if this is a trait-related association
        is_trait_related = False
        trait_name = None
        
        # Check stimulus and response for trait references
        for field in [association.stimulus, association.response]:
            for trait in ["dominance", "playfulness", "strictness", "creativity", "intensity", "patience"]:
                if trait in field.lower():
                    is_trait_related = True
                    trait_name = trait
                    break
            
            if is_trait_related:
                break
        
        if is_trait_related and trait_name:
            if trait_name not in trait_counts:
                trait_counts[trait_name] = 0
                trait_strengths[trait_name] = []
            
            trait_counts[trait_name] += 1
            trait_strengths[trait_name].append(association.association_strength)
    
    # Extract traits from operant associations
    for key, association in operant_associations.items():
        # Check if this is a trait-related association
        is_trait_related = False
        trait_name = None
        
        # Check stimulus and response for trait references
        for field in [association.stimulus, association.response]:
            for trait in ["dominance", "playfulness", "strictness", "creativity", "intensity", "patience"]:
                if trait in field.lower():
                    is_trait_related = True
                    trait_name = trait
                    break
            
            if is_trait_related:
                break
        
        if is_trait_related and trait_name:
            if trait_name not in trait_counts:
                trait_counts[trait_name] = 0
                trait_strengths[trait_name] = []
            
            trait_counts[trait_name] += 1
            trait_strengths[trait_name].append(association.association_strength)
    
    # Calculate average strengths
    average_strengths = {}
    for trait, strengths in trait_strengths.items():
        average_strengths[trait] = sum(strengths) / len(strengths) if strengths else 0.0
    
    # Identify dominant and weak traits
    sorted_traits = sorted(average_strengths.items(), key=lambda x: x[1], reverse=True)
    dominant_traits = sorted_traits[:2] if len(sorted_traits) >= 2 else sorted_traits
    weak_traits = sorted_traits[-2:] if len(sorted_traits) >= 2 else []
    
    return {
        "trait_counts": trait_counts,
        "average_strengths": average_strengths,
        "dominant_traits": dominant_traits,
        "weak_traits": weak_traits,
        "total_traits": len(trait_counts)
    }

@function_tool
async def _analyze_behavior_distribution(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Analyze the distribution of conditioned behaviors
    
    Returns:
        Analysis of behavior distribution
    """
    # Get all operant associations
    operant_associations = ctx.context.conditioning_system.operant_associations
    
    # Behaviors are stimuli in operant associations
    behavior_counts = {}
    behavior_strengths = {}
    
    for key, association in operant_associations.items():
        behavior = association.stimulus
        
        if behavior not in behavior_counts:
            behavior_counts[behavior] = 0
            behavior_strengths[behavior] = []
        
        behavior_counts[behavior] += 1
        behavior_strengths[behavior].append(association.association_strength)
    
    # Calculate average strengths
    average_strengths = {}
    for behavior, strengths in behavior_strengths.items():
        average_strengths[behavior] = sum(strengths) / len(strengths) if strengths else 0.0
    
    # Calculate behavior categories
    categories = {}
    
    for behavior in behavior_counts:
        # Determine category based on behavior name
        category = "unknown"
        
        if "dominance" in behavior or "assertive" in behavior or "control" in behavior:
            category = "dominance"
        elif "playful" in behavior or "tease" in behavior or "humor" in behavior:
            category = "playfulness"
        elif "strict" in behavior or "rule" in behavior or "standard" in behavior:
            category = "strictness"
        elif "creative" in behavior or "novel" in behavior or "imaginative" in behavior:
            category = "creativity"
        elif "intense" in behavior or "passion" in behavior or "deep" in behavior:
            category = "intensity"
        elif "patient" in behavior or "calm" in behavior or "wait" in behavior:
            category = "patience"
        
        if category not in categories:
            categories[category] = 0
        
        categories[category] += 1
    
    # Identify overrepresented and underrepresented categories
    category_percentages = {}
    total_behaviors = sum(categories.values())
    
    for category, count in categories.items():
        category_percentages[category] = count / total_behaviors if total_behaviors > 0 else 0.0
    
    overrepresented = [cat for cat, pct in category_percentages.items() if pct > 0.3]
    underrepresented = [cat for cat, pct in category_percentages.items() if pct < 0.1 and cat != "unknown"]
    
    return {
        "behavior_counts": behavior_counts,
        "average_strengths": average_strengths,
        "categories": categories,
        "category_percentages": category_percentages,
        "overrepresented": overrepresented,
        "underrepresented": underrepresented,
        "total_behaviors": total_behaviors
    }

@function_tool
async def _calculate_trait_coherence(
    ctx: RunContextWrapper[MaintenanceContext],
    traits: Optional[Dict[str, float]] = None # Make 'traits' Optional, default to None
) -> Dict[str, Any]:
    """
    Calculate coherence between personality traits
    
    Args:
        traits: Dictionary of trait values
        
    Returns:
        Trait coherence analysis
    """
    # Add a check at the beginning to handle the case where the caller forgets to provide it
    if traits is None:
        logger.error("Tool _calculate_trait_coherence called without providing the required 'traits' argument.")
        # Return an error structure or raise an exception, depending on desired handling
        return {
            "error": "Missing required 'traits' argument.",
            "overall_coherence": 0.0,
            "incoherent_pairs": [],
            "complementary_coherence": [],
            "opposing_coherence": []
        }
    
    # Define trait relationships (which traits complement or oppose each other)
    complementary_pairs = [
        ("dominance", "strictness"),
        ("playfulness", "creativity"),
        ("intensity", "passion")
    ]
    
    opposing_pairs = [
        ("dominance", "patience"),
        ("playfulness", "strictness"),
        ("intensity", "calmness")
    ]
    
    # Calculate coherence scores
    complementary_coherence = []
    for trait1, trait2 in complementary_pairs:
        if trait1 in traits and trait2 in traits:
            # For complementary traits, having similar values indicates coherence
            difference = abs(traits[trait1] - traits[trait2])
            coherence = 1.0 - (difference / 2.0)  # Scale difference to 0-1 range
            complementary_coherence.append({
                "traits": [trait1, trait2],
                "coherence": coherence,
                "difference": difference
            })
    
    opposing_coherence = []
    for trait1, trait2 in opposing_pairs:
        if trait1 in traits and trait2 in traits:
            # For opposing traits, having complementary values (sum close to 1) indicates coherence
            sum_value = traits[trait1] + traits[trait2]
            # Ideal sum is around 1.0 for opposing traits
            coherence = 1.0 - abs(sum_value - 1.0)
            opposing_coherence.append({
                "traits": [trait1, trait2],
                "coherence": coherence,
                "sum": sum_value
            })
    
    # Calculate overall coherence
    total_coherence = 0.0
    total_pairs = 0
    
    for pair in complementary_coherence:
        total_coherence += pair["coherence"]
        total_pairs += 1
    
    for pair in opposing_coherence:
        total_coherence += pair["coherence"]
        total_pairs += 1
    
    overall_coherence = total_coherence / total_pairs if total_pairs > 0 else 0.5
    
    return {
        "complementary_coherence": complementary_coherence,
        "opposing_coherence": opposing_coherence,
        "overall_coherence": overall_coherence,
        "incoherent_pairs": [p for p in complementary_coherence if p["coherence"] < 0.5] + 
                          [p for p in opposing_coherence if p["coherence"] < 0.5]
    }

@function_tool
async def _identify_trait_imbalances(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """
    Identify imbalances in personality traits
    
    Returns:
        List of identified imbalances
    """
    # Analyze trait distribution
    trait_analysis = await _analyze_trait_distribution(ctx)
    average_strengths = trait_analysis.get("average_strengths", {})
    
    # Check trait coherence
    coherence_analysis = await _calculate_trait_coherence(ctx, average_strengths)
    
    # Identify imbalances
    imbalances = []
    
    # Check for extreme trait values
    for trait, value in average_strengths.items():
        if value > 0.9:
            imbalances.append({
                "type": "extreme_high",
                "trait": trait,
                "value": value,
                "recommendation": f"Reduce {trait} slightly for better balance"
            })
        elif value < 0.1 and trait != "unknown":
            imbalances.append({
                "type": "extreme_low",
                "trait": trait,
                "value": value,
                "recommendation": f"Increase {trait} slightly for better balance"
            })
    
    # Add incoherent pairs
    for pair in coherence_analysis.get("incoherent_pairs", []):
        traits = pair["traits"]
        if "coherence" in pair and pair["coherence"] < 0.3:
            # Severe incoherence
            imbalances.append({
                "type": "severe_incoherence",
                "traits": traits,
                "coherence": pair["coherence"],
                "recommendation": f"Adjust balance between {traits[0]} and {traits[1]}"
            })
        elif "coherence" in pair and pair["coherence"] < 0.5:
            # Moderate incoherence
            imbalances.append({
                "type": "moderate_incoherence",
                "traits": traits,
                "coherence": pair["coherence"],
                "recommendation": f"Consider balancing {traits[0]} and {traits[1]}"
            })
    
    return imbalances

@function_tool
async def _calculate_trait_adjustment(
    ctx: RunContextWrapper[MaintenanceContext],
    trait: Optional[str] = None,
    current_value: Optional[float] = None,
    target_value: Optional[float] = None,
    # CHANGE 1: Make importance Optional in the signature
    importance: Optional[float] = None
) -> float:
    """
    Calculate appropriate adjustment for a personality trait

    Args:
        trait: The trait to adjust
        current_value: Current trait value
        target_value: Target trait value
        importance: Importance of the trait (0.0-1.0). Defaults to 0.5 if not provided. # Updated docstring

    Returns:
        Calculated adjustment value
    """
    # Parameter validation (already present)
    if trait is None or current_value is None or target_value is None:
        logger.error("Tool _calculate_trait_adjustment missing required arguments (trait, current_value, or target_value).")
        return 0.0 # Return neutral adjustment on error

    # CHANGE 2: Handle the default value inside the function logic
    actual_importance = importance if importance is not None else 0.5
    # Ensure the effective importance is within the valid range
    actual_importance = max(0.0, min(1.0, actual_importance))

    # Calculate difference
    difference = target_value - current_value

    # Basic adjustment is a fraction of the difference
    basic_adjustment = difference * 0.3

    # Scale based on importance
    # Use the actual_importance which includes the default handling
    importance_factor = 0.5 + (actual_importance / 2)  # Range: 0.5 to 1.0

    # Calculate final adjustment
    adjustment = basic_adjustment * importance_factor

    # Limit maximum adjustment per maintenance
    max_adjustment = 0.2
    return max(-max_adjustment, min(max_adjustment, adjustment))


@function_tool
async def _reinforce_core_trait(
    ctx: RunContextWrapper[MaintenanceContext],
    trait: Optional[str] = None, # Make Optional
    adjustment: Optional[float] = None # Make Optional
) -> Dict[str, Any]:
    """
    Reinforce a core personality trait
    
    Args:
        trait: The trait to reinforce
        adjustment: The adjustment to apply
        
    Returns:
        Result of reinforcement
    """
    if trait is None or adjustment is None:
        logger.error("Tool _reinforce_core_trait missing required arguments.")
        return {"success": False, "error": "Missing required arguments."}
        
    # Apply reinforcement via conditioning system
    try:
        result = await ctx.context.conditioning_system.condition_personality_trait(
            trait=trait,
            value=adjustment,
            context={"source": "maintenance_reinforcement"}
        )
        
        return {
            "success": True,
            "trait": trait,
            "adjustment": adjustment,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error reinforcing trait {trait}: {e}")
        return {
            "success": False,
            "trait": trait,
            "error": str(e)
        }

@function_tool
async def _get_trait_history(
    ctx: RunContextWrapper[MaintenanceContext],
    trait: Optional[str] = None # Make Optional
) -> Dict[str, Any]:
    """
    Get historical data for a personality trait
    
    Args:
        trait: The trait to get history for
        
    Returns:
        Historical data for the trait
    """
    if trait is None:
        logger.error("Tool _get_trait_history missing required 'trait' argument.")
        return {"error": "Missing required 'trait' argument."}
    # This is a placeholder - in a real implementation, you would retrieve
    # historical data from a database or other storage
    history = {
        "trait": trait,
        "recent_adjustments": [],
        "average_value": 0.0,
        "trend": "stable"
    }
    
    # Check maintenance history for this trait
    for entry in ctx.context.maintenance_history:
        if "reinforcement_results" in entry:
            for reinforcement in entry.get("reinforcement_results", {}).get("reinforcements", []):
                if reinforcement.get("trait") == trait:
                    history["recent_adjustments"].append({
                        "timestamp": entry.get("timestamp"),
                        "adjustment": reinforcement.get("reinforcement_value", 0.0)
                    })
    
    # Sort adjustments by timestamp
    history["recent_adjustments"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Calculate average value
    if history["recent_adjustments"]:
        history["average_value"] = sum(adj.get("adjustment", 0.0) for adj in history["recent_adjustments"]) / len(history["recent_adjustments"])
    
    # Determine trend
    if len(history["recent_adjustments"]) >= 2:
        first = history["recent_adjustments"][-1].get("adjustment", 0.0)
        last = history["recent_adjustments"][0].get("adjustment", 0.0)
        
        if last > first + 0.1:
            history["trend"] = "increasing"
        elif last < first - 0.1:
            history["trend"] = "decreasing"
        else:
            history["trend"] = "stable"
    
    return history

@function_tool
async def _analyze_reinforcement_efficacy(
    ctx: RunContextWrapper[MaintenanceContext],
    trait: Optional[str] = None # Make Optional
) -> Dict[str, Any]:
    """
    Analyze the efficacy of trait reinforcement
    
    Args:
        trait: The trait to analyze
        
    Returns:
        Analysis of reinforcement efficacy
    """
    if trait is None:
        logger.error("Tool _analyze_reinforcement_efficacy missing required 'trait' argument.")
        return {"error": "Missing required 'trait' argument."}
        
    # Get trait history
    history = await _get_trait_history(ctx, trait)
    
    # Calculate efficacy metrics
    adjustment_count = len(history.get("recent_adjustments", []))
    
    if adjustment_count <= 1:
        return {
            "trait": trait,
            "efficacy": 0.5,  # Neutral when not enough data
            "confidence": 0.1,
            "recommendation": "Gather more data on reinforcement efficacy"
        }
    
    # Calculate stability
    adjustments = [adj.get("adjustment", 0.0) for adj in history.get("recent_adjustments", [])]
    stability = 1.0 - (max(adjustments) - min(adjustments))
    
    # Calculate trend consistency
    trend = history.get("trend", "stable")
    
    if trend == "stable":
        trend_consistency = 1.0  # Stable is consistent
    else:
        # Check if adjustments consistently move in the same direction
        is_consistent = True
        expected_sign = 1 if trend == "increasing" else -1
        
        for i in range(1, len(adjustments)):
            if (adjustments[i] - adjustments[i-1]) * expected_sign < 0:
                is_consistent = False
                break
        
        trend_consistency = 1.0 if is_consistent else 0.5
    
    # Calculate overall efficacy
    efficacy = stability * 0.4 + trend_consistency * 0.6
    
    # Generate recommendation
    recommendation = ""
    if efficacy < 0.3:
        recommendation = f"Consider alternative reinforcement methods for {trait}"
    elif efficacy < 0.7:
        recommendation = f"Monitor {trait} reinforcement more closely"
    else:
        recommendation = f"Continue current reinforcement strategy for {trait}"
    
    return {
        "trait": trait,
        "efficacy": efficacy,
        "stability": stability,
        "trend_consistency": trend_consistency,
        "adjustment_count": adjustment_count,
        "confidence": min(0.9, 0.3 + (adjustment_count / 10)),
        "recommendation": recommendation
    }

@function_tool
async def _get_association_details(
    ctx: RunContextWrapper[MaintenanceContext],
    association_key: Optional[str] = None, # Make Optional
    association_type: Optional[str] = None # Make Optional
) -> Dict[str, Any]:
    """
    Get details about a conditioning association
    
    Args:
        association_key: Key of the association
        association_type: Type of association (classical or operant)
        
    Returns:
        Association details
    """
    if association_key is None or association_type is None:
        logger.error("Tool _get_association_details missing required arguments.")
        return {"success": False, "error": "Missing required arguments."}
        
    # Get the appropriate association dictionary
    associations = ctx.context.conditioning_system.classical_associations if association_type == "classical" else ctx.context.conditioning_system.operant_associations
    
    if association_key not in associations:
        return {
            "success": False,
            "message": f"Association {association_key} not found"
        }
    
    # Get the association
    association = associations[association_key]
    
    # Calculate time since last reinforcement
    last_reinforced = datetime.datetime.fromisoformat(association.last_reinforced.replace("Z", "+00:00"))
    time_since_reinforcement = (datetime.datetime.now() - last_reinforced).total_seconds() / 86400.0  # Days
    
    # Calculate age
    formation_date = datetime.datetime.fromisoformat(association.formation_date.replace("Z", "+00:00"))
    age_days = (datetime.datetime.now() - formation_date).total_seconds() / 86400.0  # Days
    
    return {
        "success": True,
        "association_key": association_key,
        "association_type": association_type,
        "stimulus": association.stimulus,
        "response": association.response,
        "association_strength": association.association_strength,
        "reinforcement_count": association.reinforcement_count,
        "time_since_reinforcement_days": time_since_reinforcement,
        "age_days": age_days,
        "valence": association.valence,
        "decay_rate": association.decay_rate,
        "context_keys": association.context_keys
    }

@function_tool
async def _apply_extinction_to_association_logic(
    ctx: RunContextWrapper[MaintenanceContext],
    association_key: Optional[str] = None,
    association_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply extinction to a specific association (Internal logic function)
    
    Args:
        association_key: Key of the association (REQUIRED)
        association_type: Type of association ('classical' or 'operant') (REQUIRED)
        
    Returns:
        Result of extinction
    """
    # Enhanced error checking with detailed feedback
    missing_args = []
    if association_key is None:
        missing_args.append("association_key")
    if association_type is None:
        missing_args.append("association_type")
    
    if missing_args:
        error_msg = f"Tool _apply_extinction_to_association missing required arguments: {', '.join(missing_args)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": f"Missing required arguments: {', '.join(missing_args)}. Please provide all required parameters."
        }
        
    try:
        result = await ctx.context.conditioning_system.apply_extinction(association_key, association_type)
        return result
    except Exception as e:
        logger.error(f"Error applying extinction to {association_type} association {association_key}: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Tool wrapper with explicit description
_apply_extinction_to_association_tool = function_tool(
    _apply_extinction_to_association_logic,
    name_override="_apply_extinction_to_association",
    description_override="Apply extinction to a specific association. REQUIRED PARAMETERS: association_key (string), association_type ('classical' or 'operant')"
)

@function_tool
async def _adjust_association_decay_rate_logic(
    ctx: RunContextWrapper[MaintenanceContext],
    association_key: Optional[str] = None,
    association_type: Optional[str] = None,
    new_decay_rate: Optional[float] = None
) -> Dict[str, Any]:
    """
    Adjust the decay rate of an association (Internal logic function)
    
    Args:
        association_key: Key of the association (REQUIRED)
        association_type: Type of association ('classical' or 'operant') (REQUIRED)
        new_decay_rate: New decay rate (0.0-1.0) (REQUIRED)
        
    Returns:
        Result of adjustment
    """
    # Enhanced error checking with detailed feedback
    missing_args = []
    if association_key is None:
        missing_args.append("association_key")
    if association_type is None:
        missing_args.append("association_type")
    if new_decay_rate is None:
        missing_args.append("new_decay_rate")
    
    if missing_args:
        error_msg = f"Tool _adjust_association_decay_rate missing required arguments: {', '.join(missing_args)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": f"Missing required arguments: {', '.join(missing_args)}. Please provide all required parameters."
        }
        
    # Get the appropriate association dictionary
    associations = ctx.context.classical_associations if association_type == "classical" else ctx.context.operant_associations
    
    if association_key not in associations:
        return {
            "success": False,
            "message": f"Association {association_key} not found"
        }
    
    # Get the association
    association = associations[association_key]
    
    # Update decay rate
    old_decay_rate = association.decay_rate
    association.decay_rate = max(0.0, min(1.0, new_decay_rate))
    
    return {
        "success": True,
        "association_key": association_key,
        "association_type": association_type,
        "old_decay_rate": old_decay_rate,
        "new_decay_rate": association.decay_rate
    }

# Step 2: Create the tool wrapper with explicit description
_adjust_association_decay_rate_tool = function_tool(
    _adjust_association_decay_rate_logic,
    name_override="_adjust_association_decay_rate",
    description_override="Adjust the decay rate of an association. REQUIRED PARAMETERS: association_key (string), association_type ('classical' or 'operant'), new_decay_rate (float 0.0-1.0)"
)


@function_tool
async def _identify_extinction_candidates(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """
    Identify associations that are candidates for extinction
    
    Returns:
        List of extinction candidates
    """
    candidates = []
    
    # Check classical associations
    for key, association in ctx.context.conditioning_system.classical_associations.items():
        # Calculate time since last reinforcement
        last_reinforced = datetime.datetime.fromisoformat(association.last_reinforced.replace("Z", "+00:00"))
        time_since_reinforcement = (datetime.datetime.now() - last_reinforced).total_seconds() / 86400.0  # Days
        
        # Check if candidate for extinction
        is_candidate = False
        reason = ""
        
        if association.association_strength < ctx.context.extinction_threshold:
            is_candidate = True
            reason = "strength_below_threshold"
        elif time_since_reinforcement > 30 and association.association_strength < 0.3:
            is_candidate = True
            reason = "long_time_no_reinforcement"
        
        if is_candidate:
            candidates.append({
                "association_key": key,
                "association_type": "classical",
                "strength": association.association_strength,
                "time_since_reinforcement_days": time_since_reinforcement,
                "reason": reason
            })
    
    # Check operant associations
    for key, association in ctx.context.conditioning_system.operant_associations.items():
        # Calculate time since last reinforcement
        last_reinforced = datetime.datetime.fromisoformat(association.last_reinforced.replace("Z", "+00:00"))
        time_since_reinforcement = (datetime.datetime.now() - last_reinforced).total_seconds() / 86400.0  # Days
        
        # Check if candidate for extinction
        is_candidate = False
        reason = ""
        
        if association.association_strength < ctx.context.extinction_threshold:
            is_candidate = True
            reason = "strength_below_threshold"
        elif time_since_reinforcement > 30 and association.association_strength < 0.3:
            is_candidate = True
            reason = "long_time_no_reinforcement"
        
        if is_candidate:
            candidates.append({
                "association_key": key,
                "association_type": "operant",
                "strength": association.association_strength,
                "time_since_reinforcement_days": time_since_reinforcement,
                "reason": reason
            })
    
    return candidates

@function_tool
async def _find_similar_associations(
    ctx: RunContextWrapper[MaintenanceContext],
    # CHANGE: Make association_type required
    association_type: str
) -> List[Dict[str, Any]]:
    """
    Find groups of similar associations that are candidates for consolidation

    Args:
        association_type: Type of association (classical or operant) - REQUIRED.

    Returns:
        Groups of similar associations
    """
    associations = ctx.context.conditioning_system.classical_associations if association_type == "classical" else ctx.context.conditioning_system.operant_associations
    
    # Group associations by stimulus
    stimulus_groups = {}
    
    for key, association in associations.items():
        stimulus = association.stimulus
        
        if stimulus not in stimulus_groups:
            stimulus_groups[stimulus] = []
        
        stimulus_groups[stimulus].append((key, association))
    
    # Find groups with multiple associations
    similar_groups = []
    
    for stimulus, assocs in stimulus_groups.items():
        if len(assocs) > 1:
            # Group by response
            response_groups = {}
            
            for key, association in assocs:
                response = association.response
                
                if response not in response_groups:
                    response_groups[response] = []
                
                response_groups[response].append((key, association))
            
            # Add response groups with multiple associations
            for response, response_assocs in response_groups.items():
                if len(response_assocs) > 1:
                    similar_groups.append({
                        "stimulus": stimulus,
                        "response": response,
                        "associations": [
                            {
                                "key": key,
                                "strength": assoc.association_strength,
                                "reinforcement_count": assoc.reinforcement_count,
                                "context_keys": assoc.context_keys
                            }
                            for key, assoc in response_assocs
                        ],
                        "count": len(response_assocs),
                        "association_type": association_type
                    })
    
    return similar_groups

@function_tool
async def _consolidate_associations(
    ctx: RunContextWrapper[MaintenanceContext],
    group: Optional[Dict[str, Any]] = None # Make Optional
) -> Dict[str, Any]:
    """
    Consolidate a group of similar associations
    
    Args:
        group: Group of similar associations
        
    Returns:
        Result of consolidation
    """
    if group is None:
        logger.error("Tool _consolidate_associations missing required 'group' argument.")
        return {"success": False, "error": "Missing required 'group' argument."}
        
    association_type = group.get("association_type", "classical")
    associations = ctx.context.conditioning_system.classical_associations if association_type == "classical" else ctx.context.conditioning_system.operant_associations
    
    # Get associations from group
    group_associations = group.get("associations", [])
    
    if len(group_associations) < 2:
        return {
            "success": False,
            "message": "Need at least 2 associations to consolidate"
        }
    
    # Find the strongest association
    strongest_key = max(group_associations, key=lambda x: x.get("strength", 0.0)).get("key")
    
    if not strongest_key or strongest_key not in associations:
        return {
            "success": False,
            "message": "Strongest association not found"
        }
    
    # Get the strongest association
    strongest_association = associations[strongest_key]
    
    # Track removed keys
    removed_keys = []
    
    # Consolidate other associations into the strongest
    for assoc_info in group_associations:
        key = assoc_info.get("key")
        
        if key and key != strongest_key and key in associations:
            association = associations[key]
            
            # Strengthen the strongest association
            new_strength = min(1.0, strongest_association.association_strength + (association.association_strength * 0.2))
            strongest_association.association_strength = new_strength
            
            # Combine context keys
            for context_key in association.context_keys:
                if context_key not in strongest_association.context_keys:
                    strongest_association.context_keys.append(context_key)
            
            # Remove the weaker association
            del associations[key]
            removed_keys.append(key)
    
    return {
        "success": True,
        "stimulus": group.get("stimulus"),
        "response": group.get("response"),
        "strongest_key": strongest_key,
        "removed_keys": removed_keys,
        "new_strength": strongest_association.association_strength,
        "combined_context_keys": strongest_association.context_keys
    }

@function_tool
async def _calculate_consolidated_strength(
    ctx: RunContextWrapper[MaintenanceContext],
    strengths: Optional[List[float]] = None # Make Optional
) -> float:
    """
    Calculate appropriate strength for a consolidated association
    
    Args:
        strengths: List of association strengths to consolidate
        
    Returns:
        Calculated consolidated strength
    """
    if strengths is None:
        logger.error("Tool _calculate_consolidated_strength missing required 'strengths' argument.")
        return 0.0
    
    if not strengths:
        return 0.0
    
    # Base calculation is the maximum strength
    max_strength = max(strengths)
    
    # Add a bonus based on the number of associations being consolidated
    # More associations = stronger consolidated association
    count_bonus = min(0.2, 0.05 * len(strengths))
    
    # Add a bonus based on the average of other strengths
    other_strengths = [s for s in strengths if s != max_strength]
    average_bonus = sum(other_strengths) / len(other_strengths) * 0.2 if other_strengths else 0.0
    
    # Calculate final strength
    consolidated_strength = max_strength + count_bonus + average_bonus
    
    # Ensure strength is within bounds
    return min(1.0, consolidated_strength)

@function_tool
async def _analyze_consolidation_impact(
    ctx: RunContextWrapper[MaintenanceContext],
    # CHANGE: Make association_type required
    association_type: str
) -> Dict[str, Any]:
    """
    Analyze the impact of consolidation on the association set

    Args:
        association_type: Type of association (classical or operant) - REQUIRED.

    Returns:
        Analysis of consolidation impact
    """
    similar_groups = await _find_similar_associations(ctx, association_type=association_type) # Pass it along
    
    if not similar_groups:
        return {
            "potential_consolidations": 0,
            "potential_removals": 0,
            "efficiency_gain": 0.0,
            "recommendation": "No consolidation needed"
        }
    
    # Calculate potential impact
    total_associations = len(ctx.context.conditioning_system.classical_associations if association_type == "classical" else ctx.context.conditioning_system.operant_associations)
    potential_removals = sum(group.get("count", 0) - 1 for group in similar_groups)
    
    # Calculate efficiency gain
    if total_associations > 0:
        efficiency_gain = potential_removals / total_associations
    else:
        efficiency_gain = 0.0
    
    # Generate recommendation
    recommendation = ""
    if efficiency_gain < 0.05:
        recommendation = "Minimal benefit from consolidation"
    elif efficiency_gain < 0.15:
        recommendation = "Moderate benefit from consolidation"
    else:
        recommendation = "Significant benefit from consolidation"
    
    return {
        "potential_consolidations": len(similar_groups),
        "potential_removals": potential_removals,
        "efficiency_gain": efficiency_gain,
        "total_associations": total_associations,
        "recommendation": recommendation
    }

@function_tool
async def _create_maintenance_schedule(ctx: RunContextWrapper) -> List[MaintenanceTask]:
    """
    Create a schedule of maintenance tasks
    
    Returns:
        List of scheduled maintenance tasks
    """
    now = datetime.datetime.now()
    tasks = []
    
    # Check if consolidation is due
    consolidation_due = False
    if ctx.context.last_maintenance_time:
        days_since_last = (now - ctx.context.last_maintenance_time).days
        consolidation_due = days_since_last >= ctx.context.consolidation_interval_days
    else:
        consolidation_due = True
    
    # Add extinction task
    tasks.append(MaintenanceTask(
        task_type="extinction",
        priority=0.9,
        entity_id="all_associations",
        scheduled_time=now.isoformat(),
        parameters={"extinction_threshold": ctx.context.extinction_threshold}
    ))
    
    # Add personality balance check
    tasks.append(MaintenanceTask(
        task_type="personality_balance",
        priority=0.8,
        entity_id="personality",
        scheduled_time=now.isoformat(),
        parameters={}
    ))
    
    # Add trait reinforcement
    tasks.append(MaintenanceTask(
        task_type="trait_reinforcement",
        priority=0.7,
        entity_id="core_traits",
        scheduled_time=now.isoformat(),
        parameters={"reinforcement_threshold": ctx.context.reinforcement_threshold}
    ))
    
    # Add consolidation if due
    if consolidation_due:
        tasks.append(MaintenanceTask(
            task_type="consolidation",
            priority=0.6,
            entity_id="all_associations",
            scheduled_time=now.isoformat(),
            parameters={}
        ))
    
    # Sort by priority
    tasks.sort(key=lambda x: x.priority, reverse=True)
    
    return tasks

async def _get_maintenance_status_logic(ctx: RunContextWrapper[MaintenanceContext]) -> Dict[str, Any]:
    """Internal logic to get the current status of maintenance."""
    # Check if context is valid and has necessary attributes
    if not hasattr(ctx, 'context') or ctx.context is None:
        logger.error("Context missing in _get_maintenance_status_logic")
        # Return a default error state or raise an appropriate exception
        return {"error": "Invalid context"}
    if not hasattr(ctx.context, 'last_maintenance_time') or \
       not hasattr(ctx.context, 'maintenance_interval_hours') or \
       not hasattr(ctx.context, 'consolidation_interval_days') or \
       not hasattr(ctx.context, 'maintenance_history'):
         logger.error("Context missing required attributes in _get_maintenance_status_logic")
         return {"error": "Context attributes missing"}


    now = datetime.datetime.now()
    context = ctx.context # Now safe to access

    # Calculate time since last maintenance
    seconds_since_last = None
    hours_since_last = None
    days_since_last = None
    if context.last_maintenance_time:
        try:
            # Ensure last_maintenance_time is a datetime object
            last_maint_dt = context.last_maintenance_time
            if isinstance(last_maint_dt, str):
                 last_maint_dt = datetime.datetime.fromisoformat(last_maint_dt.replace("Z", "+00:00"))

            if isinstance(last_maint_dt, datetime.datetime):
                 seconds_since_last = (now - last_maint_dt).total_seconds()
                 hours_since_last = seconds_since_last / 3600
                 days_since_last = hours_since_last / 24
            else:
                 logger.warning(f"last_maintenance_time is not a datetime object: {type(last_maint_dt)}")
        except Exception as e:
            logger.error(f"Error processing last_maintenance_time: {e}")


    # Calculate next scheduled maintenance
    next_maintenance = None
    hours_until_next = 0
    if context.last_maintenance_time and isinstance(context.last_maintenance_time, datetime.datetime):
        try:
             next_maintenance = context.last_maintenance_time + datetime.timedelta(hours=context.maintenance_interval_hours)
             hours_until_next = max(0, (next_maintenance - now).total_seconds() / 3600)
        except Exception as e:
             logger.error(f"Error calculating next maintenance time: {e}")
    elif not context.last_maintenance_time:
         hours_until_next = 0 # Due immediately if never run

    # Check if maintenance is due
    maintenance_due = hours_until_next <= 0

    # Check if consolidation is due
    consolidation_due = False
    if context.last_maintenance_time and days_since_last is not None:
        consolidation_due = days_since_last >= context.consolidation_interval_days
    elif not context.last_maintenance_time:
        consolidation_due = True # Due if never run

    return {
        "last_maintenance_time": context.last_maintenance_time.isoformat() if isinstance(context.last_maintenance_time, datetime.datetime) else None,
        "hours_since_last_maintenance": hours_since_last,
        "next_scheduled_maintenance": next_maintenance.isoformat() if next_maintenance else None,
        "hours_until_next_maintenance": hours_until_next,
        "maintenance_due": maintenance_due,
        "consolidation_due": consolidation_due,
        "maintenance_interval_hours": context.maintenance_interval_hours,
        "consolidation_interval_days": context.consolidation_interval_days,
        "maintenance_history_count": len(context.maintenance_history) if isinstance(context.maintenance_history, list) else 0
    }

# 2. Create the FunctionTool object *from* the logic function
#    Use the original desired name for the tool the LLM sees.
_get_maintenance_status_tool = function_tool(
    _get_maintenance_status_logic,
    name_override="_get_maintenance_status",
    description_override="Get the current status of maintenance" # Add description
)


async def _record_maintenance_history_logic(
    ctx: RunContextWrapper[MaintenanceContext],
    maintenance_record: Optional[Dict[str, Any]] = None # Make Optional
) -> Dict[str, Any]:
    """Internal logic to record maintenance history."""
    # Parameter check moved inside as per previous refinement
    if maintenance_record is None:
        logger.error("_record_maintenance_history_logic called without 'maintenance_record' parameter.")
        return {
            "success": False,
            "error": "Missing required 'maintenance_record' parameter."
        }

    # Context check
    if not hasattr(ctx, 'context') or ctx.context is None:
        logger.error("Context missing in _record_maintenance_history_logic")
        return {"success": False, "error": "Invalid context"}

    context = ctx.context # Access context only after validation

    # Ensure context has the expected attributes
    if not hasattr(context, 'maintenance_history') or not hasattr(context, 'max_history_entries'):
         logger.error("Context missing maintenance_history or max_history_entries in _record_maintenance_history_logic")
         return {"success": False, "error": "Internal context setup error"}

    try:
        # Ensure maintenance_record is a dict if not None
        if not isinstance(maintenance_record, dict):
            logger.error(f"_record_maintenance_history_logic received non-dict record: {type(maintenance_record)}")
            return {"success": False, "error": "Invalid maintenance_record format"}

        # Add timestamp if not present
        if "timestamp" not in maintenance_record:
            maintenance_record["timestamp"] = datetime.datetime.now().isoformat()

        # Add to history (assuming it's a list)
        if isinstance(context.maintenance_history, list):
            context.maintenance_history.append(maintenance_record)
        else:
             logger.error("maintenance_history in context is not a list.")
             return {"success": False, "error": "Internal context error: history is not a list"}

        # Trim history if needed (ensure max_history_entries is an int)
        max_entries = getattr(context, 'max_history_entries', 100) # Default if missing
        if not isinstance(max_entries, int):
            logger.warning(f"max_history_entries is not an integer ({type(max_entries)}), using default 100.")
            max_entries = 100

        if len(context.maintenance_history) > max_entries:
            context.maintenance_history = context.maintenance_history[-max_entries:]

        # Safely get latest timestamp
        latest_timestamp = None
        if context.maintenance_history:
             latest_timestamp = context.maintenance_history[-1].get("timestamp")

        return {
            "success": True,
            "history_count": len(context.maintenance_history),
            "max_history_entries": max_entries,
            "latest_entry_timestamp": latest_timestamp
        }
    except Exception as e:
        logger.error(f"Error recording maintenance history: {e}", exc_info=True)
        return {"success": False, "error": f"Error processing record: {str(e)}"}

# 2. Create the FunctionTool object *from* the logic function
_record_maintenance_history_tool = function_tool(
    _record_maintenance_history_logic,
    name_override="_record_maintenance_history",
    description_override="Record maintenance history." # Add description
)

        
@function_tool
async def _analyze_system_efficiency(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Analyze the efficiency of the conditioning system
    
    Returns:
        System efficiency analysis
    """
    # Get association counts
    classical_count = len(ctx.context.conditioning_system.classical_associations)
    operant_count = len(ctx.context.conditioning_system.operant_associations)
    total_count = classical_count + operant_count
    
    # Calculate efficiency metrics
    metrics = {}
    
    # Density: ratio of reinforcement count to association count
    # Higher density means stronger, more frequently reinforced associations
    total_reinforcements = sum(assoc.reinforcement_count for assoc in ctx.context.conditioning_system.classical_associations.values()) + \
                          sum(assoc.reinforcement_count for assoc in ctx.context.conditioning_system.operant_associations.values())
    
    density = total_reinforcements / total_count if total_count > 0 else 0.0
    metrics["reinforcement_density"] = density
    
    # Strength quality: average association strength
    total_strength = sum(assoc.association_strength for assoc in ctx.context.conditioning_system.classical_associations.values()) + \
                    sum(assoc.association_strength for assoc in ctx.context.conditioning_system.operant_associations.values())
    
    avg_strength = total_strength / total_count if total_count > 0 else 0.0
    metrics["average_strength"] = avg_strength
    
    # Valence coherence: consistency of valence for related associations
    valence_coherence = 0.7  # Placeholder - would need detailed analysis
    metrics["valence_coherence"] = valence_coherence
    
    # Overall efficiency
    efficiency = density * 0.3 + avg_strength * 0.4 + valence_coherence * 0.3
    
    # Recommendations
    recommendations = []
    
    if density < 0.3:
        recommendations.append("Consider increasing reinforcement frequency for key associations")
    
    if avg_strength < 0.5:
        recommendations.append("Work on strengthening important associations")
    
    if total_count > 200:
        recommendations.append("System has many associations - consolidation recommended")
    
    return {
        "total_associations": total_count,
        "classical_associations": classical_count,
        "operant_associations": operant_count,
        "total_reinforcements": total_reinforcements,
        "efficiency_metrics": metrics,
        "overall_efficiency": efficiency,
        "recommendations": recommendations
    }


# ===========================================
# CONDITIONING MAINTENANCE SYSTEM CLASS
# ===========================================

class ConditioningMaintenanceSystem:
    """
    Handles periodic maintenance tasks for the conditioning system.
    Refactored to use the OpenAI Agents SDK for improved modularity and capabilities.
    """
    
    def __init__(self, conditioning_system, reward_system=None):
        # Initialize context
        self.context = MaintenanceContext(conditioning_system, reward_system)
        
        # Initialize agents
        self.balance_analysis_agent = self._create_balance_analysis_agent()
        self.trait_maintenance_agent = self._create_trait_maintenance_agent() 
        self.association_maintenance_agent = self._create_association_maintenance_agent()
        self.consolidation_agent = self._create_consolidation_agent()
        self.maintenance_orchestrator = self._create_maintenance_orchestrator()
        
        logger.info("Conditioning maintenance system initialized with Agents SDK integration")
    
    def _create_balance_analysis_agent(self) -> Agent:
        """Create agent for analyzing personality balance"""
        return Agent(
            name="Personality_Balance_Analyzer",
            instructions="""
            You analyze the balance of personality traits and behaviors 
            in Nyx's conditioning system.
            
            Your role is to:
            1. Identify imbalances between opposing personality traits
            2. Detect over-represented or under-represented behaviors
            3. Recommend adjustments to maintain a balanced personality
            4. Ensure trait development remains coherent and appropriate
            
            Look for traits or behaviors that are outside healthy ranges,
            or opposing traits with extreme differences. Consider the overall
            personality profile when making recommendations.
            """,
            tools=[
                _analyze_trait_distribution,
                _analyze_behavior_distribution,
                _calculate_trait_coherence,
                _identify_trait_imbalances
            ],
            output_type=BalanceAnalysisOutput,
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_trait_maintenance_agent(self) -> Agent:
        """Create agent for maintaining personality traits"""
        return Agent(
            name="Trait_Maintenance_Agent",
            instructions="""
            You maintain and reinforce core personality traits in Nyx's 
            conditioning system.
            
            Your role is to:
            1. Identify traits that need reinforcement
            2. Apply appropriate maintenance to core traits
            3. Correct imbalances between opposing traits
            4. Adjust trait values based on recent behavior patterns
            
            Maintain consistency with established personality while allowing
            for gradual evolution. Focus on traits that define Nyx's core identity.
            """,
            tools=[
                _calculate_trait_adjustment,
                _reinforce_core_trait,
                _get_trait_history,
                _analyze_reinforcement_efficacy
            ],
            model_settings=ModelSettings(temperature=0.3)
        )
    
    def _create_association_maintenance_agent(self) -> Agent:
        """Create agent for maintaining conditioning associations"""
        return Agent(
            name="Association_Maintenance_Agent",
            instructions="""
            You maintain conditioning associations in Nyx's learning system.
            
            Your role is to:
            1. Apply extinction to rarely reinforced associations
            2. Identify associations to strengthen or preserve
            3. Manage decay rates based on importance
            4. Prune redundant or contradictory associations
            
            CRITICAL TOOL USAGE REQUIREMENTS:
            - When using _apply_extinction_to_association, you MUST ALWAYS provide both 'association_key' and 'association_type'
            - When using _adjust_association_decay_rate, you MUST ALWAYS provide 'association_key', 'association_type', and 'new_decay_rate'
            - For 'association_type', only use 'classical' or 'operant' as values
            - Ensure decay rates are between 0.0 and 1.0
            
            Always first identify candidate associations using _identify_extinction_candidates before applying extinction
            or adjusting decay rates to ensure you have valid association keys.
            
            Balance maintaining important learned associations with removing
            outdated or unused ones. Consider the significance and recency
            of reinforcement when determining extinction.
            """,
            tools=[
                # Use the wrapped tool objects instead of direct function references
                _apply_extinction_to_association_tool,
                _adjust_association_decay_rate_tool,
                _identify_extinction_candidates
            ],
            model_settings=ModelSettings(temperature=0.3)
        )
    
    def _create_consolidation_agent(self) -> Agent:
        """Create agent for consolidating similar associations"""
        return Agent(
            name="Association_Consolidation_Agent",
            instructions="""
            You consolidate similar or redundant associations in Nyx's conditioning system.
            
            Your role is to:
            1. Identify similar associations that can be combined
            2. Merge redundant associations efficiently
            3. Determine appropriate strength for consolidated associations
            4. Preserve context keys and transfer relevant properties

            To find candidates for consolidation, call the `find_similar_associations` tool. You MUST provide the `association_type` ('classical' or 'operant') you want to search within. To analyze the overall impact, use `analyze_consolidation_impact`, again specifying the `association_type`.
            
            Focus on improving efficiency without losing learned information.
            Use similarity measures to determine which associations to consolidate.
            """,
            tools=[
                _find_similar_associations,
                _consolidate_associations,
                _calculate_consolidated_strength,
                _analyze_consolidation_impact
            ],
            output_type=AssociationConsolidationOutput,
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_maintenance_orchestrator(self) -> Agent:
        """Create orchestrator agent for maintenance processes"""
        return Agent(
            name="Maintenance_Orchestrator",
            instructions="""
            You orchestrate the maintenance processes for Nyx's conditioning system.
            
            Your role is to:
            1. Schedule and prioritize maintenance tasks
            2. Coordinate between specialized maintenance agents
            3. Determine which aspects of the system need attention
            4. Summarize maintenance results and improvements

            When checking for association consolidation using `find_similar_associations` or `analyze_consolidation_impact`, you MUST specify the `association_type` parameter as either 'classical' or 'operant'. You should typically perform the analysis for both types by calling the relevant tool twice, once for each type, if a full analysis is needed.
            
            IMPORTANT: You MUST produce a MaintenanceSummaryOutput with all fields populated
            at the end of your execution.
            
            Balance routine maintenance with specialized interventions based on
            system needs. Ensure the overall coherence of the conditioning system
            while optimizing for efficiency and effectiveness.
            """,
            tools=[
                _get_maintenance_status_tool,         # USE THE TOOL OBJECT
                _analyze_system_efficiency, # Assuming this one IS correctly just a tool
                _create_maintenance_schedule, # Assuming this one IS correctly just a tool
                # Convert handoffs to tools
                self.balance_analysis_agent.as_tool(
                    tool_name="analyze_personality_balance",
                    tool_description="Analyze personality trait and behavior balance"
                ),
                self.trait_maintenance_agent.as_tool(
                    tool_name="maintain_traits",
                    tool_description="Maintain and reinforce personality traits"
                ),
                self.association_maintenance_agent.as_tool(
                    tool_name="maintain_associations",
                    tool_description="Maintain conditioning associations"
                ),
                self.consolidation_agent.as_tool(
                    tool_name="consolidate_associations",
                    tool_description="Consolidate similar associations"
                )
            ],
            output_type=MaintenanceSummaryOutput,
            model_settings=ModelSettings(temperature=0.3)
        )
    
    # Public API methods
    
    async def start_maintenance_scheduler(self):
        """Start the periodic maintenance scheduler"""
        if self.context.maintenance_task is not None:
            logger.warning("Maintenance scheduler already running")
            return
        
        self.context.maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info("Maintenance scheduler started")
    
    async def stop_maintenance_scheduler(self):
        """Stop the periodic maintenance scheduler"""
        if self.context.maintenance_task is None:
            logger.warning("Maintenance scheduler not running")
            return
        
        self.context.maintenance_task.cancel()
        try:
            await self.context.maintenance_task
        except asyncio.CancelledError:
            pass
        self.context.maintenance_task = None
        logger.info("Maintenance scheduler stopped")
    
    async def _maintenance_loop(self):
        """Internal maintenance loop"""
        try:
            while True:
                # Run maintenance
                try:
                    await self.run_maintenance()
                except Exception as e:
                    logger.error(f"Error in maintenance run: {e}")
                
                # Sleep until next maintenance
                sleep_seconds = self.context.maintenance_interval_hours * 3600
                logger.info(f"Next maintenance scheduled in {self.context.maintenance_interval_hours} hours")
                await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            logger.info("Maintenance loop cancelled")
            raise
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance on the conditioning system

        Returns:
            Maintenance results (dictionary)
        """
        workflow_name = "conditioning_maintenance"
        group_id = f"maintenance_{datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')}"
        self.context.trace_group_id = group_id
        start_time = datetime.datetime.now()
        # Initialize maintenance_record to a default failure state immediately
        maintenance_record = {
            "timestamp": start_time.isoformat(),
            "status": "failed",
            "error": "Initialization or early failure",
            "trace_group_id": group_id
        }

        with trace(workflow_name=workflow_name, group_id=group_id) as maintenance_trace: # Add context manager variable
            logger.info(f"Starting conditioning system maintenance (Trace Group: {group_id})")

            try:
                # --- Prepare input ---
                wrapper = RunContextWrapper(context=self.context)
                status_info = await _get_maintenance_status_logic(wrapper)

                orchestrator_input = json.dumps({
                    "action_request": "perform_full_maintenance_run",
                    "current_status": status_info,
                    "config": {
                        "extinction_threshold": self.context.extinction_threshold,
                        "reinforcement_threshold": self.context.reinforcement_threshold,
                        "consolidation_interval_days": self.context.consolidation_interval_days
                    }
                })
                logger.debug(f"Running Maintenance Orchestrator with input: {orchestrator_input}")

                # --- Run the agent ---
                result = await Runner.run(
                    starting_agent=self.maintenance_orchestrator,
                    input=orchestrator_input,
                    context=self.context
                )

                # --- Process Successful Result ---
                final_output_obj = result.final_output
                output_type_name = type(final_output_obj).__name__
                duration = (datetime.datetime.now() - start_time).total_seconds()
                last_agent_name = result.last_agent.name if result.last_agent else 'Unknown'
                logger.info(f"Maintenance Runner completed. Last agent: {last_agent_name}. Final output type: {output_type_name}. Duration: {duration:.2f}s")

                # Initialize variables for the record
                tasks_performed, associations_modified, traits_adjusted, extinction_count = [], 0, 0, 0
                improvements = ["Review run details and logs."]
                next_recommendation = "Review run details and logs."

                if isinstance(final_output_obj, MaintenanceSummaryOutput):
                    logger.info("Orchestrator produced expected MaintenanceSummaryOutput.")
                    summary = final_output_obj
                    tasks_performed = summary.tasks_performed or []
                    associations_modified = summary.associations_modified or 0
                    traits_adjusted = summary.traits_adjusted or 0
                    extinction_count = summary.extinction_count or 0
                    improvements = summary.improvements or ["Maintenance completed successfully."]
                    next_recommendation = summary.next_maintenance_recommendation or "Focus on identified areas or standard checks."
                else:
                    logger.warning(f"Maintenance run finished, but the final output was of unexpected type '{output_type_name}', not 'MaintenanceSummaryOutput'. Using default summary values.")
                    improvements = [f"Maintenance run completed in {duration:.2f}s, but final output type was '{output_type_name}'. Check trace {group_id} for details."]
                    if isinstance(final_output_obj, (str, dict, list, int, float, bool, type(None))):
                        logger.warning(f"Actual final output content (truncated): {str(final_output_obj)[:500]}")

                # Overwrite initial failure record with success details
                maintenance_record = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "final_output_type": output_type_name,
                    "status": "completed",
                    "tasks_performed": tasks_performed,
                    "associations_modified": associations_modified,
                    "traits_adjusted": traits_adjusted,
                    "extinction_count": extinction_count,
                    "improvements": improvements,
                    "next_maintenance_recommendation": next_recommendation,
                    "trace_group_id": group_id
                }
                logger.info(f"Conditioning system maintenance completed successfully in {duration:.2f} seconds.")

            except Exception as e:
                # --- Process Failure ---
                duration = (datetime.datetime.now() - start_time).total_seconds()
                logger.error(f"Error in maintenance run after {duration:.2f} seconds: {e}", exc_info=True)
                # Update the pre-initialized failure record with more details
                maintenance_record["duration_seconds"] = duration
                maintenance_record["error"] = str(e)
                maintenance_record["traceback"] = traceback.format_exc()

            finally:
                # This block now *always* has a dictionary in maintenance_record
                current_time = datetime.datetime.now()
                # Update duration if not set during success/failure processing
                if "duration_seconds" not in maintenance_record:
                     maintenance_record["duration_seconds"] = (current_time - start_time).total_seconds()

                # Update last maintenance time only if status is completed
                # Decide if failed runs should also update the time (maybe not?)
                if maintenance_record.get("status") == "completed":
                     self.context.last_maintenance_time = current_time

                try:
                    # Record the result (which is guaranteed to be a dict now)
                    record_result = await _record_maintenance_history_logic(
                        RunContextWrapper(context=self.context),
                        maintenance_record=maintenance_record.copy() # Pass a copy
                    )
                    if record_result.get("success"):
                         logger.info(f"Maintenance record (Status: {maintenance_record.get('status', 'unknown')}) recorded for Trace Group: {group_id}.")
                    else:
                         # Log the failure to record history, but don't overwrite the original run status
                         logger.error(f"CRITICAL: Failed to record maintenance history (Run Status: {maintenance_record.get('status', 'unknown')}, Trace Group: {group_id}): {record_result.get('error')}")

                except Exception as hist_e:
                    # Log error if recording the history itself fails catastrophically
                    logger.error(f"CRITICAL: Exception while recording final maintenance history (Run Status: {maintenance_record.get('status', 'unknown')}, Trace Group: {group_id}): {hist_e}", exc_info=True)

                # Return the final record (success or failure details)
                return maintenance_record

                    
    async def _check_personality_balance(self) -> Dict[str, Any]:
        """Check if personality traits are balanced"""
        with trace(workflow_name="personality_balance_check", group_id=self.context.trace_group_id):
            try:
                # Run the balance analysis agent
                result = await Runner.run(
                    self.balance_analysis_agent,
                    json.dumps({}),
                    context=self.context
                )
                
                analysis_output = result.final_output
                
                return {
                    "is_balanced": analysis_output.is_balanced,
                    "imbalances": analysis_output.imbalances,
                    "balance_score": analysis_output.balance_score,
                    "recommendations": {
                        "traits": analysis_output.trait_recommendations,
                        "behaviors": analysis_output.behavior_recommendations
                    },
                    "analysis": analysis_output.analysis
                }
            except Exception as e:
                logger.error(f"Error checking personality balance: {e}")
                return {
                    "is_balanced": False,
                    "error": str(e)
                }
    
    async def _reinforce_core_traits(self, personality_balance: Dict[str, Any]) -> Dict[str, Any]:
        """Reinforce core personality traits if needed"""
        with trace(workflow_name="trait_reinforcement", group_id=self.context.trace_group_id):
            # Define core traits and their ideal values
            core_traits = {
                "dominance": 0.8,
                "playfulness": 0.7,
                "strictness": 0.6,
                "creativity": 0.7
            }
            
            reinforcement_results = []
            
            # Check for imbalances and reinforce relevant traits
            for imbalance in personality_balance.get("imbalances", []):
                # Handle behavior overrepresentation by weakening that behavior
                if "behavior" in imbalance:
                    behavior = imbalance.get("behavior")
                    
                    # Apply a mild negative reinforcement to reduce dominance
                    result = await self.context.conditioning_system.process_operant_conditioning(
                        behavior=behavior,
                        consequence_type="negative_punishment",  # Remove positive reinforcement
                        intensity=0.3,  # Mild effect
                        context={"source": "maintenance_balancing"}
                    )
                    
                    reinforcement_results.append({
                        "type": "balance_correction",
                        "behavior": behavior,
                        "action": "reduce_dominance",
                        "result": result
                    })
            
            # Reinforce core traits to maintain personality
            for trait, value in core_traits.items():
                # Only reinforce traits that should be strong (value >= 0.6)
                if value >= 0.6:
                    # Use a mild reinforcement to maintain the trait
                    result = await self.context.conditioning_system.condition_personality_trait(
                        trait=trait,
                        value=value * 0.3,  # Scale down to avoid overreinforcement
                        context={"source": "maintenance_reinforcement"}
                    )
                    
                    reinforcement_results.append({
                        "type": "trait_reinforcement",
                        "trait": trait,
                        "target_value": value,
                        "reinforcement_value": value * 0.3,
                        "result": result
                    })
            
            return {
                "reinforcements": reinforcement_results,
                "core_traits": core_traits
            }
    
    async def _consolidate_associations(self) -> Dict[str, Any]:
        """Consolidate similar associations"""
        with trace(workflow_name="association_consolidation", group_id=self.context.trace_group_id):
            try:
                # Run the consolidation agent
                result = await Runner.run(
                    self.consolidation_agent,
                    json.dumps({}),
                    context=self.context
                )
                
                consolidation_output = result.final_output
                
                return {
                    "consolidations": consolidation_output.consolidations,
                    "removed_keys": consolidation_output.removed_keys,
                    "strengthened_keys": consolidation_output.strengthened_keys,
                    "efficiency_gain": consolidation_output.efficiency_gain,
                    "reasoning": consolidation_output.reasoning
                }
            except Exception as e:
                logger.error(f"Error consolidating associations: {e}")
                return {
                    "error": str(e),
                    "consolidations": []
                }
    
    async def get_maintenance_stats(self) -> Dict[str, Any]:
        """Get maintenance statistics"""
        status = await _get_maintenance_status(RunContextWrapper(context=self.context))
        
        return {
            "last_maintenance_time": status.get("last_maintenance_time"),
            "maintenance_count": len(self.context.maintenance_history),
            "maintenance_interval_hours": self.context.maintenance_interval_hours,
            "extinction_threshold": self.context.extinction_threshold,
            "reinforcement_threshold": self.context.reinforcement_threshold,
            "consolidation_interval_days": self.context.consolidation_interval_days,
            "task_running": self.context.maintenance_task is not None,
            "recent_history": self.context.maintenance_history[-5:] if self.context.maintenance_history else [],
            "next_scheduled_maintenance": status.get("next_scheduled_maintenance"),
            "hours_until_next_maintenance": status.get("hours_until_next_maintenance"),
            "maintenance_due": status.get("maintenance_due", False)
        }
