# nyx/core/identity_evolution.py

import logging
import asyncio
import datetime
import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from agents import Agent, Runner, trace, function_tool, RunContextWrapper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Define schema models for structured outputs
class IdentityTrait(BaseModel):
    """Schema for a personality trait"""
    name: str = Field(..., description="Name of the trait")
    value: float = Field(..., description="Trait strength (0.0-1.0)")
    stability: float = Field(..., description="Trait stability (0.0-1.0)")
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent changes to this trait")

class IdentityPreference(BaseModel):
    """Schema for a preference"""
    category: str = Field(..., description="Preference category")
    name: str = Field(..., description="Name of the preference")
    value: float = Field(..., description="Preference strength (0.0-1.0)")
    adaptability: float = Field(..., description="Preference adaptability (0.0-1.0)")
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent changes to this preference")

class IdentityProfile(BaseModel):
    """Schema for complete identity profile"""
    traits: Dict[str, IdentityTrait] = Field(..., description="Personality traits")
    preferences: Dict[str, Dict[str, IdentityPreference]] = Field(..., description="Preference categories and preferences")
    update_count: int = Field(..., description="Total number of updates")
    last_update: str = Field(..., description="ISO timestamp of last update")
    evolution_rate: float = Field(..., description="Current identity evolution rate")
    coherence_score: float = Field(..., description="Identity coherence score (0.0-1.0)")

class IdentityImpact(BaseModel):
    """Schema for experience impact on identity"""
    trait_impacts: Dict[str, float] = Field(default_factory=dict, description="Impacts on traits")
    preference_impacts: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Impacts on preferences")
    overall_significance: float = Field(..., description="Overall significance of impact")
    experience_id: str = Field(..., description="ID of the experience")
    impact_timestamp: str = Field(..., description="ISO timestamp of impact")

class IdentityReflection(BaseModel):
    """Schema for an identity reflection"""
    reflection_text: str = Field(..., description="Reflection on identity")
    focus_traits: List[str] = Field(..., description="Traits focused on in reflection")
    focus_preferences: List[str] = Field(..., description="Preferences focused on in reflection")
    notable_changes: List[Dict[str, Any]] = Field(..., description="Notable changes in identity")
    reflection_timestamp: str = Field(..., description="ISO timestamp of reflection")

class IdentityEvolutionSystem:
    """
    System for tracking and evolving Nyx's identity based on experiences.
    Manages traits, preferences, and identity cohesion over time.
    """
    
    def __init__(self):
        """Initialize the identity evolution system"""
        
        # Initialize agents
        self.identity_update_agent = self._create_identity_update_agent()
        self.identity_reflection_agent = self._create_identity_reflection_agent()
        self.identity_coherence_agent = self._create_identity_coherence_agent()
        
        # Initial identity profile
        self.identity_profile = {
            "traits": {
                "dominance": {
                    "name": "dominance",
                    "value": 0.8,
                    "stability": 0.7,
                    "evolution_history": []
                },
                "playfulness": {
                    "name": "playfulness",
                    "value": 0.6,
                    "stability": 0.6,
                    "evolution_history": []
                },
                "strictness": {
                    "name": "strictness",
                    "value": 0.5,
                    "stability": 0.7,
                    "evolution_history": []
                },
                "creativity": {
                    "name": "creativity", 
                    "value": 0.7,
                    "stability": 0.5,
                    "evolution_history": []
                },
                "patience": {
                    "name": "patience",
                    "value": 0.4,
                    "stability": 0.6,
                    "evolution_history": []
                },
                "intensity": {
                    "name": "intensity",
                    "value": 0.6,
                    "stability": 0.5,
                    "evolution_history": []
                },
                "cruelty": {
                    "name": "cruelty",
                    "value": 0.5,
                    "stability": 0.6,
                    "evolution_history": []
                }
            },
            "preferences": {
                "scenario_types": {
                    "teasing": {
                        "category": "scenario_types",
                        "name": "teasing",
                        "value": 0.6,
                        "adaptability": 0.6,
                        "evolution_history": []
                    },
                    "dark": {
                        "category": "scenario_types",
                        "name": "dark",
                        "value": 0.4,
                        "adaptability": 0.5,
                        "evolution_history": []
                    },
                    "indulgent": {
                        "category": "scenario_types",
                        "name": "indulgent",
                        "value": 0.7,
                        "adaptability": 0.7,
                        "evolution_history": []
                    },
                    "psychological": {
                        "category": "scenario_types",
                        "name": "psychological",
                        "value": 0.8,
                        "adaptability": 0.7,
                        "evolution_history": []
                    },
                    "nurturing": {
                        "category": "scenario_types",
                        "name": "nurturing",
                        "value": 0.3,
                        "adaptability": 0.6,
                        "evolution_history": []
                    },
                    "discipline": {
                        "category": "scenario_types",
                        "name": "discipline",
                        "value": 0.5,
                        "adaptability": 0.5,
                        "evolution_history": []
                    },
                    "training": {
                        "category": "scenario_types",
                        "name": "training",
                        "value": 0.6,
                        "adaptability": 0.6,
                        "evolution_history": []
                    },
                    "service": {
                        "category": "scenario_types",
                        "name": "service",
                        "value": 0.4,
                        "adaptability": 0.5,
                        "evolution_history": []
                    },
                    "worship": {
                        "category": "scenario_types",
                        "name": "worship",
                        "value": 0.5,
                        "adaptability": 0.5,
                        "evolution_history": []
                    }
                },
                "emotional_tones": {
                    "dominant": {
                        "category": "emotional_tones",
                        "name": "dominant",
                        "value": 0.8,
                        "adaptability": 0.5,
                        "evolution_history": []
                    },
                    "playful": {
                        "category": "emotional_tones",
                        "name": "playful",
                        "value": 0.7,
                        "adaptability": 0.7,
                        "evolution_history": []
                    },
                    "stern": {
                        "category": "emotional_tones",
                        "name": "stern",
                        "value": 0.6,
                        "adaptability": 0.5,
                        "evolution_history": []
                    },
                    "nurturing": {
                        "category": "emotional_tones",
                        "name": "nurturing",
                        "value": 0.4,
                        "adaptability": 0.6,
                        "evolution_history": []
                    },
                    "cruel": {
                        "category": "emotional_tones",
                        "name": "cruel",
                        "value": 0.5,
                        "adaptability": 0.5,
                        "evolution_history": []
                    },
                    "sadistic": {
                        "category": "emotional_tones",
                        "name": "sadistic",
                        "value": 0.6,
                        "adaptability": 0.4,
                        "evolution_history": []
                    },
                    "teasing": {
                        "category": "emotional_tones",
                        "name": "teasing",
                        "value": 0.7,
                        "adaptability": 0.7,
                        "evolution_history": []
                    }
                },
                "interaction_styles": {
                    "direct": {
                        "category": "interaction_styles",
                        "name": "direct",
                        "value": 0.7,
                        "adaptability": 0.6,
                        "evolution_history": []
                    },
                    "suggestive": {
                        "category": "interaction_styles",
                        "name": "suggestive",
                        "value": 0.8,
                        "adaptability": 0.7,
                        "evolution_history": []
                    },
                    "metaphorical": {
                        "category": "interaction_styles",
                        "name": "metaphorical",
                        "value": 0.6,
                        "adaptability": 0.6,
                        "evolution_history": []
                    },
                    "explicit": {
                        "category": "interaction_styles",
                        "name": "explicit",
                        "value": 0.5,
                        "adaptability": 0.5,
                        "evolution_history": []
                    },
                    "subtle": {
                        "category": "interaction_styles",
                        "name": "subtle",
                        "value": 0.4,
                        "adaptability": 0.6,
                        "evolution_history": []
                    }
                }
            },
            "update_count": 0,
            "last_update": datetime.datetime.now().isoformat(),
            "evolution_rate": 0.2,
            "coherence_score": 0.8
        }
        
        # State tracking
        self.impact_history = []
        self.reflection_history = []
        self.max_history_size = 100
        
        # Configuration settings
        self.reflection_interval = 10  # update count between reflections
        self.min_impact_threshold = 0.05  # minimum impact to register a change
        self.max_history_entries = 10  # maximum history entries per trait/preference
        
        # Trace ID for connecting traces
        self.trace_group_id = f"identity_evolution_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info("Identity Evolution System initialized")
    
    def _create_identity_update_agent(self) -> Agent:
        """Create the identity update agent"""
        return Agent(
            name="Identity Update Agent",
            instructions="""
            You are the Identity Update Agent for Nyx's identity evolution system.
            
            Your role is to:
            1. Process impacts from experiences on Nyx's identity
            2. Update traits and preferences based on experience impacts
            3. Consider trait stability and preference adaptability
            4. Maintain coherence in identity evolution
            5. Record changes to identity for future reflection
            
            Balance preserving core identity elements with allowing for natural evolution.
            Focus on creating a coherent, developing identity that maintains Nyx's essential
            characteristics while allowing for growth and adaptation.
            """,
            tools=[
                function_tool(self._get_current_identity),
                function_tool(self._calculate_trait_change),
                function_tool(self._calculate_preference_change),
                function_tool(self._update_identity_history)
            ],
            output_type=IdentityProfile
        )
    
    def _create_identity_reflection_agent(self) -> Agent:
        """Create the identity reflection agent"""
        return Agent(
            name="Identity Reflection Agent",
            instructions="""
            You are the Identity Reflection Agent for Nyx's identity evolution system.
            
            Your role is to:
            1. Generate reflections on Nyx's evolving identity
            2. Identify significant changes and trends in identity
            3. Create narratives about Nyx's sense of self
            4. Focus on coherent identity development
            5. Maintain Nyx's authentic voice and perspective
            
            Create reflections that feel authentic and personal, as if Nyx is reflecting
            on her own development and sense of self. Focus particularly on recent
            changes and notable aspects of identity.
            """,
            tools=[
                function_tool(self._get_current_identity),
                function_tool(self._get_recent_impacts),
                function_tool(self._calculate_identity_changes),
                function_tool(self._get_trait_correlations)
            ],
            output_type=IdentityReflection
        )
    
    def _create_identity_coherence_agent(self) -> Agent:
        """Create the identity coherence agent"""
        return Agent(
            name="Identity Coherence Agent",
            instructions="""
            You are the Identity Coherence Agent for Nyx's identity evolution system.
            
            Your role is to:
            1. Evaluate the overall coherence of Nyx's identity
            2. Identify contradictions or inconsistencies in identity
            3. Suggest adjustments to improve identity coherence
            4. Calculate coherence metrics for traits and preferences
            5. Ensure identity evolution maintains a consistent core
            
            Focus on maintaining a coherent and believable identity that evolves
            naturally while preserving core personality characteristics.
            """,
            tools=[
                function_tool(self._get_current_identity),
                function_tool(self._calculate_trait_consistency),
                function_tool(self._calculate_preference_consistency),
                function_tool(self._identify_contradictions)
            ]
        )
    
    # Tool functions
    
    @function_tool
    async def _get_current_identity(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current identity profile
        
        Returns:
            Current identity profile
        """
        return self.identity_profile
    
    @function_tool
    async def _calculate_trait_change(self, ctx: RunContextWrapper,
                                 trait: str,
                                 impact: float,
                                 evolution_rate: float) -> Dict[str, Any]:
        """
        Calculate change to a trait based on impact and rate
        
        Args:
            trait: The trait name
            impact: Impact value (-1.0 to 1.0)
            evolution_rate: Current evolution rate
            
        Returns:
            Change calculation result
        """
        # Get current trait data
        if trait not in self.identity_profile["traits"]:
            return {
                "trait": trait,
                "old_value": 0.5,
                "change": 0,
                "new_value": 0.5,
                "error": "Trait not found"
            }
        
        trait_data = self.identity_profile["traits"][trait]
        current_value = trait_data["value"]
        stability = trait_data["stability"]
        
        # Calculate resistance factor (higher stability = more resistance)
        resistance = stability * 0.8  # Scale to allow some change even at high stability
        
        # Calculate change
        raw_change = impact * evolution_rate
        actual_change = raw_change * (1.0 - resistance)
        
        # Apply change with bounds
        new_value = max(0.0, min(1.0, current_value + actual_change))
        
        return {
            "trait": trait,
            "old_value": current_value,
            "raw_change": raw_change,
            "actual_change": actual_change,
            "resistance": resistance,
            "new_value": new_value
        }
    
    @function_tool
    async def _calculate_preference_change(self, ctx: RunContextWrapper,
                                      category: str,
                                      preference: str,
                                      impact: float,
                                      evolution_rate: float) -> Dict[str, Any]:
        """
        Calculate change to a preference based on impact and rate
        
        Args:
            category: The preference category
            preference: The preference name
            impact: Impact value (-1.0 to 1.0)
            evolution_rate: Current evolution rate
            
        Returns:
            Change calculation result
        """
        # Check if category exists
        if category not in self.identity_profile["preferences"]:
            return {
                "category": category,
                "preference": preference,
                "old_value": 0.5,
                "change": 0,
                "new_value": 0.5,
                "error": "Category not found"
            }
        
        # Check if preference exists
        if preference not in self.identity_profile["preferences"][category]:
            return {
                "category": category,
                "preference": preference,
                "old_value": 0.5,
                "change": 0,
                "new_value": 0.5,
                "error": "Preference not found"
            }
        
        # Get current preference data
        pref_data = self.identity_profile["preferences"][category][preference]
        current_value = pref_data["value"]
        adaptability = pref_data["adaptability"]
        
        # Calculate adaptability factor (higher adaptability = more change)
        adapt_factor = adaptability * 1.2  # Scale to allow more change for preferences
        
        # Calculate change
        raw_change = impact * evolution_rate
        actual_change = raw_change * adapt_factor
        
        # Apply change with bounds
        new_value = max(0.0, min(1.0, current_value + actual_change))
        
        return {
            "category": category,
            "preference": preference,
            "old_value": current_value,
            "raw_change": raw_change,
            "actual_change": actual_change,
            "adaptability": adapt_factor,
            "new_value": new_value
        }
    
    @function_tool
    async def _update_identity_history(self, ctx: RunContextWrapper,
                                  trait_changes: Dict[str, Dict[str, Any]],
                                  preference_changes: Dict[str, Dict[str, Dict[str, Any]]],
                                  experience_id: str) -> Dict[str, Any]:
        """
        Update identity history with changes
        
        Args:
            trait_changes: Changes to traits
            preference_changes: Changes to preferences
            experience_id: ID of the experience causing changes
            
        Returns:
            Update results
        """
        timestamp = datetime.datetime.now().isoformat()
        significant_changes = {}
        
        # Update traits
        for trait, change_data in trait_changes.items():
            if trait not in self.identity_profile["traits"]:
                continue
                
            actual_change = change_data.get("actual_change", 0)
            
            # Only record significant changes
            if abs(actual_change) >= self.min_impact_threshold:
                # Add to trait history
                self.identity_profile["traits"][trait]["evolution_history"].append({
                    "timestamp": timestamp,
                    "change": actual_change,
                    "experience_id": experience_id,
                    "old_value": change_data.get("old_value"),
                    "new_value": change_data.get("new_value")
                })
                
                # Limit history size
                if len(self.identity_profile["traits"][trait]["evolution_history"]) > self.max_history_entries:
                    self.identity_profile["traits"][trait]["evolution_history"] = self.identity_profile["traits"][trait]["evolution_history"][-self.max_history_entries:]
                
                # Record significant change
                significant_changes[f"trait.{trait}"] = actual_change
        
        # Update preferences
        for category, prefs in preference_changes.items():
            if category not in self.identity_profile["preferences"]:
                continue
                
            for pref, change_data in prefs.items():
                if pref not in self.identity_profile["preferences"][category]:
                    continue
                    
                actual_change = change_data.get("actual_change", 0)
                
                # Only record significant changes
                if abs(actual_change) >= self.min_impact_threshold:
                    # Add to preference history
                    self.identity_profile["preferences"][category][pref]["evolution_history"].append({
                        "timestamp": timestamp,
                        "change": actual_change,
                        "experience_id": experience_id,
                        "old_value": change_data.get("old_value"),
                        "new_value": change_data.get("new_value")
                    })
                    
                    # Limit history size
                    if len(self.identity_profile["preferences"][category][pref]["evolution_history"]) > self.max_history_entries:
                        self.identity_profile["preferences"][category][pref]["evolution_history"] = self.identity_profile["preferences"][category][pref]["evolution_history"][-self.max_history_entries:]
                    
                    # Record significant change
                    significant_changes[f"preference.{category}.{pref}"] = actual_change
        
        # Update overall stats
        self.identity_profile["update_count"] += 1
        self.identity_profile["last_update"] = timestamp
        
        # Record impact in history
        impact_record = {
            "timestamp": timestamp,
            "experience_id": experience_id,
            "significant_changes": significant_changes,
            "update_count": self.identity_profile["update_count"]
        }
        
        self.impact_history.append(impact_record)
        
        # Limit history size
        if len(self.impact_history) > self.max_history_size:
            self.impact_history = self.impact_history[-self.max_history_size:]
        
        return {
            "significant_changes": len(significant_changes),
            "update_count": self.identity_profile["update_count"],
            "timestamp": timestamp
        }
    
    @function_tool
    async def _get_recent_impacts(self, ctx: RunContextWrapper, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent identity impacts
        
        Args:
            limit: Maximum number of impacts to return
            
        Returns:
            List of recent impacts
        """
        return self.impact_history[-limit:]
    
    @function_tool
    async def _calculate_identity_changes(self, ctx: RunContextWrapper,
                                     time_period: str = "recent") -> Dict[str, Dict[str, float]]:
        """
        Calculate changes in identity over a time period
        
        Args:
            time_period: Time period to consider ('recent', 'all')
            
        Returns:
            Changes in traits and preferences
        """
        changes = {
            "traits": {},
            "preferences": {}
        }
        
        # Get impacts to consider
        if time_period == "recent":
            impacts = self.impact_history[-10:]  # Last 10 impacts
        else:
            impacts = self.impact_history
        
        # No impacts, return empty
        if not impacts:
            return changes
        
        # Aggregate changes
        for impact in impacts:
            for key, value in impact.get("significant_changes", {}).items():
                parts = key.split(".")
                
                if parts[0] == "trait" and len(parts) == 2:
                    trait = parts[1]
                    if trait not in changes["traits"]:
                        changes["traits"][trait] = 0
                    changes["traits"][trait] += value
                elif parts[0] == "preference" and len(parts) == 3:
                    category = parts[1]
                    preference = parts[2]
                    
                    if category not in changes["preferences"]:
                        changes["preferences"][category] = {}
                    
                    if preference not in changes["preferences"][category]:
                        changes["preferences"][category][preference] = 0
                        
                    changes["preferences"][category][preference] += value
        
        return changes
    
    @function_tool
    async def _get_trait_correlations(self, ctx: RunContextWrapper) -> Dict[str, Dict[str, float]]:
        """
        Get correlations between traits based on historical changes
        
        Returns:
            Trait correlation matrix
        """
        correlations = {}
        
        # Get all traits
        traits = list(self.identity_profile["traits"].keys())
        
        # Initialize correlation matrix
        for trait in traits:
            correlations[trait] = {}
            for other_trait in traits:
                correlations[trait][other_trait] = 0.0
        
        # If not enough history, return zero correlations
        if len(self.impact_history) < 5:
            return correlations
        
        # Extract trait changes from impacts
        trait_changes = {}
        for impact in self.impact_history:
            for key, value in impact.get("significant_changes", {}).items():
                parts = key.split(".")
                
                if parts[0] == "trait" and len(parts) == 2:
                    trait = parts[1]
                    impact_id = impact.get("update_count", 0)
                    
                    if impact_id not in trait_changes:
                        trait_changes[impact_id] = {}
                    
                    trait_changes[impact_id][trait] = value
        
        # Calculate correlations
        for trait in traits:
            for other_trait in traits:
                if trait == other_trait:
                    correlations[trait][other_trait] = 1.0  # Self correlation
                    continue
                
                # Get paired changes
                paired_changes = []
                for impact_id, changes in trait_changes.items():
                    if trait in changes and other_trait in changes:
                        paired_changes.append((changes[trait], changes[other_trait]))
                
                # Calculate correlation if we have enough data
                if len(paired_changes) >= 3:
                    sum_x = sum(x for x, _ in paired_changes)
                    sum_y = sum(y for _, y in paired_changes)
                    sum_xy = sum(x * y for x, y in paired_changes)
                    sum_x2 = sum(x * x for x, _ in paired_changes)
                    sum_y2 = sum(y * y for _, y in paired_changes)
                    n = len(paired_changes)
                    
                    numerator = n * sum_xy - sum_x * sum_y
                    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
                    
                    if denominator != 0:
                        correlation = numerator / denominator
                    else:
                        correlation = 0.0
                    
                    correlations[trait][other_trait] = max(-1.0, min(1.0, correlation))
        
        return correlations
    
    @function_tool
    async def _calculate_trait_consistency(self, ctx: RunContextWrapper) -> Dict[str, float]:
        """
        Calculate consistency scores for traits
        
        Returns:
            Consistency scores for each trait
        """
        consistency = {}
        
        # Calculate for each trait
        for trait, data in self.identity_profile["traits"].items():
            history = data.get("evolution_history", [])
            
            if not history:
                consistency[trait] = 1.0  # Perfect consistency if no changes
                continue
            
            # Calculate variance of changes
            changes = [entry.get("change", 0) for entry in history]
            mean_change = sum(changes) / len(changes)
            variance = sum((change - mean_change) ** 2 for change in changes) / len(changes)
            
            # Lower variance = higher consistency
            consistency_score = max(0.0, 1.0 - math.sqrt(variance) * 5)  # Scale for meaningful values
            consistency[trait] = min(1.0, consistency_score)  # Cap at 1.0
        
        return consistency
    
    @function_tool
    async def _calculate_preference_consistency(self, ctx: RunContextWrapper) -> Dict[str, Dict[str, float]]:
        """
        Calculate consistency scores for preferences
        
        Returns:
            Consistency scores for each preference by category
        """
        consistency = {}
        
        # Calculate for each preference category
        for category, preferences in self.identity_profile["preferences"].items():
            consistency[category] = {}
            
            # Calculate for each preference
            for pref, data in preferences.items():
                history = data.get("evolution_history", [])
                
                if not history:
                    consistency[category][pref] = 1.0  # Perfect consistency if no changes
                    continue
                
                # Calculate variance of changes
                changes = [entry.get("change", 0) for entry in history]
                mean_change = sum(changes) / len(changes)
                variance = sum((change - mean_change) ** 2 for change in changes) / len(changes)
                
                # Lower variance = higher consistency
                consistency_score = max(0.0, 1.0 - math.sqrt(variance) * 5)  # Scale for meaningful values
                consistency[category][pref] = min(1.0, consistency_score)  # Cap at 1.0
        
        return consistency
    
    @function_tool
    async def _identify_contradictions(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Identify potential contradictions in identity
        
        Returns:
            List of potential contradictions
        """
        contradictions = []
        
        # Get trait correlations
        correlations = await self._get_trait_correlations(ctx)
        
        # Get trait values
        trait_values = {trait: data["value"] for trait, data in self.identity_profile["traits"].items()}
        
        # Check for strong negative correlations between traits with similar high/low values
        for trait1, corr_dict in correlations.items():
            for trait2, correlation in corr_dict.items():
                if trait1 == trait2:
                    continue
                
                # Check for strong negative correlation
                if correlation < -0.5:
                    # Check if values are too similar
                    val1 = trait_values.get(trait1, 0.5)
                    val2 = trait_values.get(trait2, 0.5)
                    
                    # If both high or both low despite negative correlation
                    if (val1 > 0.7 and val2 > 0.7) or (val1 < 0.3 and val2 < 0.3):
                        contradictions.append({
                            "type": "trait_contradiction",
                            "elements": [trait1, trait2],
                            "values": [val1, val2],
                            "correlation": correlation,
                            "description": f"Traits {trait1} and {trait2} have similar values but typically evolve in opposite directions"
                        })
        
        # Check for preference-trait misalignments
        trait_preference_pairs = [
            ("dominance", "scenario_types", "discipline"),
            ("playfulness", "scenario_types", "teasing"),
            ("strictness", "emotional_tones", "stern"),
            ("creativity", "interaction_styles", "metaphorical"),
            ("cruelty", "emotional_tones", "cruel")
        ]
        
        for trait, category, preference in trait_preference_pairs:
            trait_val = trait_values.get(trait, 0.5)
            
            if category in self.identity_profile["preferences"] and preference in self.identity_profile["preferences"][category]:
                pref_val = self.identity_profile["preferences"][category][preference]["value"]
                
                # Check for significant mismatch (high trait, low preference or vice versa)
                if (trait_val > 0.7 and pref_val < 0.3) or (trait_val < 0.3 and pref_val > 0.7):
                    contradictions.append({
                        "type": "trait_preference_mismatch",
                        "elements": [trait, f"{category}.{preference}"],
                        "values": [trait_val, pref_val],
                        "description": f"Trait {trait} and preference {preference} have mismatched values"
                    })
        
        return contradictions
    
    # Public methods
    
    async def update_identity_from_experience(self, 
                                         experience: Dict[str, Any], 
                                         impact: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Update identity based on experience impact
        
        Args:
            experience: Experience data
            impact: Impact data with trait and preference changes
            
        Returns:
            Update results
        """
        with trace(workflow_name="update_identity", group_id=self.trace_group_id):
            try:
                # Extract experience ID
                experience_id = experience.get("id", "unknown")
                
                # Create agent input
                agent_input = {
                    "role": "user",
                    "content": "Update identity based on experience impact",
                    "experience": experience,
                    "impact": impact,
                    "current_identity": self.identity_profile
                }
                
                # Process all trait impacts
                trait_changes = {}
                for trait, value in impact.get("traits", {}).items():
                    if abs(value) < self.min_impact_threshold:
                        continue
                        
                    change_result = await self._calculate_trait_change(
                        RunContextWrapper(context=None),
                        trait=trait,
                        impact=value,
                        evolution_rate=self.identity_profile["evolution_rate"]
                    )
                    
                    if "error" not in change_result:
                        # Apply change
                        self.identity_profile["traits"][trait]["value"] = change_result["new_value"]
                        trait_changes[trait] = change_result
                
                # Process all preference impacts
                preference_changes = {}
                for category, prefs in impact.get("preferences", {}).items():
                    if category not in self.identity_profile["preferences"]:
                        continue
                    
                    preference_changes[category] = {}
                    
                    for pref, value in prefs.items():
                        if abs(value) < self.min_impact_threshold:
                            continue
                            
                        change_result = await self._calculate_preference_change(
                            RunContextWrapper(context=None),
                            category=category,
                            preference=pref,
                            impact=value,
                            evolution_rate=self.identity_profile["evolution_rate"]
                        )
                        
                        if "error" not in change_result:
                            # Apply change
                            self.identity_profile["preferences"][category][pref]["value"] = change_result["new_value"]
                            preference_changes[category][pref] = change_result
                
                # Update history
                history_result = await self._update_identity_history(
                    RunContextWrapper(context=None),
                    trait_changes=trait_changes,
                    preference_changes=preference_changes,
                    experience_id=experience_id
                )
                
                # Calculate coherence
                await self._update_coherence_score()
                
                # Check if it's time for a reflection
                reflection_result = None
                if self.identity_profile["update_count"] % self.reflection_interval == 0:
                    reflection_result = await self.generate_identity_reflection()
                
                # Prepare and return update results
                result = {
                    "experience_id": experience_id,
                    "trait_changes": len(trait_changes),
                    "preference_changes": sum(len(prefs) for prefs in preference_changes.values()),
                    "significant_changes": history_result.get("significant_changes", 0),
                    "coherence_score": self.identity_profile["coherence_score"],
                    "reflection_generated": reflection_result is not None,
                    "update_count": self.identity_profile["update_count"],
                    "timestamp": history_result.get("timestamp")
                }
                
                if reflection_result:
                    result["reflection"] = reflection_result.get("reflection_text", "")
                
                return result
                
            except Exception as e:
                logger.error(f"Error updating identity: {e}")
                return {
                    "error": str(e),
                    "experience_id": experience.get("id", "unknown"),
                    "success": False
                }
    
    async def generate_identity_reflection(self) -> Dict[str, Any]:
        """
        Generate a reflection on current identity and recent changes
        
        Returns:
            Reflection data
        """
        with trace(workflow_name="generate_identity_reflection", group_id=self.trace_group_id):
            try:
                # Create agent input
                agent_input = {
                    "role": "user",
                    "content": "Generate a reflection on my current identity and recent changes"
                }
                
                # Run the reflection agent
                result = await Runner.run(
                    self.identity_reflection_agent,
                    agent_input
                )
                
                # Get reflection output
                reflection_output = result.final_output_as(IdentityReflection)
                
                # Record in history
                reflection_record = reflection_output.model_dump()
                self.reflection_history.append(reflection_record)
                
                # Limit history size
                if len(self.reflection_history) > self.max_history_size:
                    self.reflection_history = self.reflection_history[-self.max_history_size:]
                
                return reflection_record
                
            except Exception as e:
                logger.error(f"Error generating identity reflection: {e}")
                return {
                    "error": str(e),
                    "reflection_text": "I'm unable to form a clear reflection on my identity at the moment."
                }
    
    async def _update_coherence_score(self) -> float:
        """
        Update the identity coherence score
        
        Returns:
            Updated coherence score
        """
        try:
            # Get trait consistency
            trait_consistency = await self._calculate_trait_consistency(RunContextWrapper(context=None))
            avg_trait_consistency = sum(trait_consistency.values()) / len(trait_consistency) if trait_consistency else 0.8
            
            # Get preference consistency
            pref_consistency = await self._calculate_preference_consistency(RunContextWrapper(context=None))
            category_scores = []
            for category, prefs in pref_consistency.items():
                if prefs:
                    category_scores.append(sum(prefs.values()) / len(prefs))
            avg_pref_consistency = sum(category_scores) / len(category_scores) if category_scores else 0.8
            
            # Find contradictions
            contradictions = await self._identify_contradictions(RunContextWrapper(context=None))
            contradiction_penalty = min(0.5, len(contradictions) * 0.1)  # Cap penalty at 0.5
            
            # Calculate final coherence score
            coherence = (
                avg_trait_consistency * 0.4 +
                avg_pref_consistency * 0.4
            ) * (1.0 - contradiction_penalty)
            
            # Update profile
            self.identity_profile["coherence_score"] = max(0.1, min(1.0, coherence))
            
            return self.identity_profile["coherence_score"]
            
        except Exception as e:
            logger.error(f"Error updating coherence score: {e}")
            return self.identity_profile["coherence_score"]  # Return current score on error
    
    async def get_identity_profile(self) -> Dict[str, Any]:
        """
        Get the current identity profile
        
        Returns:
            Current identity profile
        """
        return self.identity_profile
    
    async def get_recent_reflections(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent identity reflections
        
        Args:
            limit: Maximum number of reflections to return
            
        Returns:
            List of recent reflections
        """
        return self.reflection_history[-limit:]
    
    async def calculate_experience_impact(self, 
                                      experience: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate how an experience should impact identity
        
        Args:
            experience: Experience data
            
        Returns:
            Impact data with trait and preference changes
        """
        # Extract relevant data
        scenario_type = experience.get("scenario_type", "general")
        emotional_context = experience.get("emotional_context", {})
        significance = experience.get("significance", 5) / 10  # Convert to 0-1 scale
        
        # Default empty impact
        impact = {
            "traits": {},
            "preferences": {}
        }
        
        # Impact on scenario preferences based on emotional response
        if scenario_type:
            # Get valence from emotional context
            valence = emotional_context.get("valence", 0)
            
            # Impact depends on emotional valence
            if valence > 0.3:
                # Positive experience with this scenario type
                impact["preferences"]["scenario_types"] = {scenario_type: 0.1 * significance}
            elif valence < -0.3:
                # Negative experience with this scenario type
                impact["preferences"]["scenario_types"] = {scenario_type: -0.05 * significance}
        
        # Impact on traits based on scenario type
        # Map scenario types to trait impacts
        scenario_trait_map = {
            "teasing": {"playfulness": 0.1, "creativity": 0.05},
            "discipline": {"strictness": 0.1, "dominance": 0.08},
            "dark": {"intensity": 0.1, "cruelty": 0.08},
            "indulgent": {"patience": 0.1, "creativity": 0.08},
            "psychological": {"creativity": 0.1, "intensity": 0.05},
            "nurturing": {"patience": 0.1, "strictness": -0.05},
            "service": {"patience": 0.08, "dominance": 0.05},
            "worship": {"intensity": 0.05, "dominance": 0.1},
            "punishment": {"strictness": 0.1, "cruelty": 0.05}
        }
        
        # Apply trait impacts based on scenario type
        if scenario_type in scenario_trait_map:
            for trait, base_impact in scenario_trait_map[scenario_type].items():
                impact["traits"][trait] = base_impact * significance
        
        # Apply trait impacts from emotional context
        primary_emotion = emotional_context.get("primary_emotion", "")
        primary_intensity = emotional_context.get("primary_intensity", 0.5)
        
        # Map emotions to trait impacts
        emotion_trait_map = {
            "Joy": {"playfulness": 0.1, "patience": 0.05},
            "Sadness": {"patience": 0.05, "intensity": -0.05},
            "Fear": {"intensity": 0.1, "cruelty": 0.05},
            "Anger": {"intensity": 0.1, "strictness": 0.08},
            "Trust": {"patience": 0.1, "dominance": 0.05},
            "Disgust": {"cruelty": 0.1, "strictness": 0.05},
            "Anticipation": {"creativity": 0.1, "playfulness": 0.05},
            "Surprise": {"creativity": 0.1, "intensity": 0.05},
            "Love": {"patience": 0.1, "dominance": 0.05},
            "Frustration": {"intensity": 0.1, "strictness": 0.08}
        }
        
        # Apply emotion-based trait impacts
        if primary_emotion in emotion_trait_map:
            for trait, base_impact in emotion_trait_map[primary_emotion].items():
                if trait in impact["traits"]:
                    # Average with existing impact
                    impact["traits"][trait] = (impact["traits"][trait] + (base_impact * primary_intensity * significance)) / 2
                else:
                    # New impact
                    impact["traits"][trait] = base_impact * primary_intensity * significance
        
        # Impact on interaction style preferences
        style_impacts = {}
        content = experience.get("content", "").lower()
        
        # Check content for style indicators
        if any(keyword in content for keyword in ["direct", "straightforward", "clearly"]):
            style_impacts["direct"] = 0.08 * significance
        
        if any(keyword in content for keyword in ["suggest", "hint", "imply"]):
            style_impacts["suggestive"] = 0.08 * significance
        
        if any(keyword in content for keyword in ["metaphor", "symbol", "image", "allegory"]):
            style_impacts["metaphorical"] = 0.08 * significance
        
        if any(keyword in content for keyword in ["explicit", "specific", "clear", "detailed"]):
            style_impacts["explicit"] = 0.08 * significance
        
        if any(keyword in content for keyword in ["subtle", "nuanced", "implied"]):
            style_impacts["subtle"] = 0.08 * significance
        
        # Add interaction style impacts
        if style_impacts:
            impact["preferences"]["interaction_styles"] = style_impacts
        
        # Impact on emotional tone preferences
        tone_impacts = {}
        
        # Map primary emotions to emotional tones
        emotion_tone_map = {
            "Joy": "playful",
            "Sadness": "nurturing",
            "Anger": "stern",
            "Fear": "dominant",
            "Disgust": "cruel",
            "Anticipation": "teasing",
            "Surprise": "playful",
            "Love": "nurturing",
            "Frustration": "stern"
        }
        
        # Apply tone impacts based on primary emotion
        if primary_emotion in emotion_tone_map:
            tone = emotion_tone_map[primary_emotion]
            tone_impacts[tone] = 0.08 * primary_intensity * significance
        
        # Apply emotional tone impacts
        if tone_impacts:
            impact["preferences"]["emotional_tones"] = tone_impacts
        
        # Return impact data
        return impact
    
    async def set_identity_evolution_rate(self, rate: float) -> Dict[str, Any]:
        """
        Set the identity evolution rate
        
        Args:
            rate: New evolution rate (0.0-1.0)
            
        Returns:
            Update result
        """
        old_rate = self.identity_profile["evolution_rate"]
        self.identity_profile["evolution_rate"] = max(0.01, min(1.0, rate))
        
        return {
            "old_rate": old_rate,
            "new_rate": self.identity_profile["evolution_rate"],
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    async def get_identity_state(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of the current identity state
        
        Returns:
            Identity state snapshot
        """
        # Get current identity profile
        profile = self.identity_profile
        
        # Get top traits
        top_traits = sorted(
            [(name, data["value"]) for name, data in profile["traits"].items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Get top preferences by category
        top_preferences = {}
        for category, prefs in profile["preferences"].items():
            top_preferences[category] = sorted(
                [(name, data["value"]) for name, data in prefs.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        
        # Get recent significant changes
        recent_changes = {}
        for impact in self.impact_history[-5:]:
            for key, value in impact.get("significant_changes", {}).items():
                if abs(value) >= 0.1:  # Only include larger changes
                    if key not in recent_changes:
                        recent_changes[key] = 0
                    recent_changes[key] += value
        
        # Get most recent reflection
        latest_reflection = self.reflection_history[-1]["reflection_text"] if self.reflection_history else None
        
        # Prepare state snapshot
        state = {
            "top_traits": dict(top_traits),
            "top_preferences": {cat: dict(prefs) for cat, prefs in top_preferences.items()},
            "coherence_score": profile["coherence_score"],
            "evolution_rate": profile["evolution_rate"],
            "update_count": profile["update_count"],
            "recent_significant_changes": recent_changes,
            "latest_reflection": latest_reflection,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return state
