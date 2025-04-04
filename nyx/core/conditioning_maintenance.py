# nyx/core/conditioning_maintenance.py

import asyncio
import logging
import datetime
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
    is_balanced: bool = Field(..., description="Whether personality is balanced")
    imbalances: List[Dict[str, Any]] = Field(..., description="Detected imbalances")
    trait_recommendations: List[Dict[str, Any]] = Field(..., description="Trait recommendations")
    behavior_recommendations: List[Dict[str, Any]] = Field(..., description="Behavior recommendations")
    balance_score: float = Field(..., description="Overall balance score (0.0-1.0)")
    analysis: str = Field(..., description="Analysis of personality balance")

class AssociationConsolidationOutput(BaseModel):
    """Output schema for association consolidation"""
    consolidations: List[Dict[str, Any]] = Field(..., description="Consolidations performed")
    removed_keys: List[str] = Field(..., description="Association keys removed")
    strengthened_keys: List[str] = Field(..., description="Association keys strengthened")
    efficiency_gain: float = Field(..., description="Efficiency gain from consolidation (0.0-1.0)")
    reasoning: str = Field(..., description="Reasoning for consolidations")

class MaintenanceSummaryOutput(BaseModel):
    """Output schema for maintenance run summary"""
    tasks_performed: List[Dict[str, Any]] = Field(..., description="Tasks performed")
    time_taken_seconds: float = Field(..., description="Time taken in seconds")
    associations_modified: int = Field(..., description="Number of associations modified")
    traits_adjusted: int = Field(..., description="Number of traits adjusted")
    extinction_count: int = Field(..., description="Number of extinctions applied")
    improvements: List[str] = Field(..., description="Improvements made to the system")
    next_maintenance_recommendation: str = Field(..., description="Recommendation for next maintenance")

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
                function_tool(self._analyze_trait_distribution),
                function_tool(self._analyze_behavior_distribution),
                function_tool(self._calculate_trait_coherence),
                function_tool(self._identify_trait_imbalances)
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
                function_tool(self._calculate_trait_adjustment),
                function_tool(self._reinforce_core_trait),
                function_tool(self._get_trait_history),
                function_tool(self._analyze_reinforcement_efficacy)
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
            
            Balance maintaining important learned associations with removing
            outdated or unused ones. Consider the significance and recency
            of reinforcement when determining extinction.
            """,
            tools=[
                function_tool(self._get_association_details),
                function_tool(self._apply_extinction_to_association),
                function_tool(self._adjust_association_decay_rate),
                function_tool(self._identify_extinction_candidates)
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
            
            Focus on improving efficiency without losing learned information.
            Use similarity measures to determine which associations to consolidate.
            """,
            tools=[
                function_tool(self._find_similar_associations),
                function_tool(self._consolidate_associations),
                function_tool(self._calculate_consolidated_strength),
                function_tool(self._analyze_consolidation_impact)
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
            
            Balance routine maintenance with specialized interventions based on
            system needs. Ensure the overall coherence of the conditioning system
            while optimizing for efficiency and effectiveness.
            """,
            handoffs=[
                handoff(self.balance_analysis_agent, 
                       tool_name_override="analyze_personality_balance",
                       tool_description_override="Analyze personality trait and behavior balance"),
                
                handoff(self.trait_maintenance_agent, 
                       tool_name_override="maintain_traits",
                       tool_description_override="Maintain and reinforce personality traits"),
                
                handoff(self.association_maintenance_agent,
                       tool_name_override="maintain_associations",
                       tool_description_override="Maintain conditioning associations"),
                
                handoff(self.consolidation_agent,
                       tool_name_override="consolidate_associations",
                       tool_description_override="Consolidate similar associations")
            ],
            tools=[
                function_tool(self._create_maintenance_schedule),
                function_tool(self._get_maintenance_status),
                function_tool(self._record_maintenance_history),
                function_tool(self._analyze_system_efficiency)
            ],
            output_type=MaintenanceSummaryOutput,
            model_settings=ModelSettings(temperature=0.3)
        )
    
    # Tool functions for agents
    
    @function_tool
    async def _analyze_trait_distribution(self, ctx: RunContextWrapper) -> Dict[str, Any]:
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
    async def _analyze_behavior_distribution(self, ctx: RunContextWrapper) -> Dict[str, Any]:
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
    async def _calculate_trait_coherence(self, ctx: RunContextWrapper, 
                                 traits: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate coherence between personality traits
        
        Args:
            traits: Dictionary of trait values
            
        Returns:
            Trait coherence analysis
        """
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
    async def _identify_trait_imbalances(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Identify imbalances in personality traits
        
        Returns:
            List of identified imbalances
        """
        # Analyze trait distribution
        trait_analysis = await self._analyze_trait_distribution(ctx)
        average_strengths = trait_analysis.get("average_strengths", {})
        
        # Check trait coherence
        coherence_analysis = await self._calculate_trait_coherence(ctx, average_strengths)
        
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
    async def _calculate_trait_adjustment(self, ctx: RunContextWrapper,
                                    trait: str,
                                    current_value: float,
                                    target_value: float,
                                    importance: float = 0.5) -> float:
        """
        Calculate appropriate adjustment for a personality trait
        
        Args:
            trait: The trait to adjust
            current_value: Current trait value
            target_value: Target trait value
            importance: Importance of the trait (0.0-1.0)
            
        Returns:
            Calculated adjustment value
        """
        # Calculate difference
        difference = target_value - current_value
        
        # Basic adjustment is a fraction of the difference
        basic_adjustment = difference * 0.3
        
        # Scale based on importance
        importance_factor = 0.5 + (importance / 2)  # Range: 0.5 to 1.0
        
        # Calculate final adjustment
        adjustment = basic_adjustment * importance_factor
        
        # Limit maximum adjustment per maintenance
        max_adjustment = 0.2
        return max(-max_adjustment, min(max_adjustment, adjustment))
    
    @function_tool
    async def _reinforce_core_trait(self, ctx: RunContextWrapper,
                              trait: str,
                              adjustment: float) -> Dict[str, Any]:
        """
        Reinforce a core personality trait
        
        Args:
            trait: The trait to reinforce
            adjustment: The adjustment to apply
            
        Returns:
            Result of reinforcement
        """
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
    async def _get_trait_history(self, ctx: RunContextWrapper, trait: str) -> Dict[str, Any]:
        """
        Get historical data for a personality trait
        
        Args:
            trait: The trait to get history for
            
        Returns:
            Historical data for the trait
        """
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
    async def _analyze_reinforcement_efficacy(self, ctx: RunContextWrapper,
                                        trait: str) -> Dict[str, Any]:
        """
        Analyze the efficacy of trait reinforcement
        
        Args:
            trait: The trait to analyze
            
        Returns:
            Analysis of reinforcement efficacy
        """
        # Get trait history
        history = await self._get_trait_history(ctx, trait)
        
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
    async def _get_association_details(self, ctx: RunContextWrapper,
                                 association_key: str,
                                 association_type: str) -> Dict[str, Any]:
        """
        Get details about a conditioning association
        
        Args:
            association_key: Key of the association
            association_type: Type of association (classical or operant)
            
        Returns:
            Association details
        """
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
    async def _apply_extinction_to_association(self, ctx: RunContextWrapper,
                                         association_key: str,
                                         association_type: str) -> Dict[str, Any]:
        """
        Apply extinction to a specific association
        
        Args:
            association_key: Key of the association
            association_type: Type of association (classical or operant)
            
        Returns:
            Result of extinction
        """
        try:
            result = await ctx.context.conditioning_system.apply_extinction(association_key, association_type)
            return result
        except Exception as e:
            logger.error(f"Error applying extinction to {association_type} association {association_key}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @function_tool
    async def _adjust_association_decay_rate(self, ctx: RunContextWrapper,
                                       association_key: str,
                                       association_type: str,
                                       new_decay_rate: float) -> Dict[str, Any]:
        """
        Adjust the decay rate of an association
        
        Args:
            association_key: Key of the association
            association_type: Type of association (classical or operant)
            new_decay_rate: New decay rate (0.0-1.0)
            
        Returns:
            Result of adjustment
        """
        # Get the appropriate association dictionary
        associations = ctx.context.conditioning_system.classical_associations if association_type == "classical" else ctx.context.conditioning_system.operant_associations
        
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
    
    @function_tool
    async def _identify_extinction_candidates(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
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
    async def _find_similar_associations(self, ctx: RunContextWrapper,
                                   association_type: str) -> List[Dict[str, Any]]:
        """
        Find groups of similar associations that are candidates for consolidation
        
        Args:
            association_type: Type of association (classical or operant)
            
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
    async def _consolidate_associations(self, ctx: RunContextWrapper,
                                  group: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate a group of similar associations
        
        Args:
            group: Group of similar associations
            
        Returns:
            Result of consolidation
        """
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
    async def _calculate_consolidated_strength(self, ctx: RunContextWrapper,
                                         strengths: List[float]) -> float:
        """
        Calculate appropriate strength for a consolidated association
        
        Args:
            strengths: List of association strengths to consolidate
            
        Returns:
            Calculated consolidated strength
        """
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
    async def _analyze_consolidation_impact(self, ctx: RunContextWrapper,
                                      association_type: str) -> Dict[str, Any]:
        """
        Analyze the impact of consolidation on the association set
        
        Args:
            association_type: Type of association (classical or operant)
            
        Returns:
            Analysis of consolidation impact
        """
        # Find similar associations
        similar_groups = await self._find_similar_associations(ctx, association_type)
        
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
    async def _create_maintenance_schedule(self, ctx: RunContextWrapper) -> List[MaintenanceTask]:
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
    
    @function_tool
    async def _get_maintenance_status(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current status of maintenance
        
        Returns:
            Maintenance status
        """
        now = datetime.datetime.now()
        
        # Calculate time since last maintenance
        if ctx.context.last_maintenance_time:
            seconds_since_last = (now - ctx.context.last_maintenance_time).total_seconds()
            hours_since_last = seconds_since_last / 3600
            days_since_last = hours_since_last / 24
        else:
            seconds_since_last = None
            hours_since_last = None
            days_since_last = None
        
        # Calculate next scheduled maintenance
        if ctx.context.last_maintenance_time:
            next_maintenance = ctx.context.last_maintenance_time + datetime.timedelta(hours=ctx.context.maintenance_interval_hours)
            hours_until_next = max(0, (next_maintenance - now).total_seconds() / 3600)
        else:
            next_maintenance = None
            hours_until_next = 0
        
        # Check if maintenance is due
        maintenance_due = hours_until_next <= 0
        
        # Check if consolidation is due
        consolidation_due = False
        if ctx.context.last_maintenance_time:
            consolidation_due = days_since_last is not None and days_since_last >= ctx.context.consolidation_interval_days
        else:
            consolidation_due = True
        
        return {
            "last_maintenance_time": ctx.context.last_maintenance_time.isoformat() if ctx.context.last_maintenance_time else None,
            "hours_since_last_maintenance": hours_since_last,
            "next_scheduled_maintenance": next_maintenance.isoformat() if next_maintenance else None,
            "hours_until_next_maintenance": hours_until_next,
            "maintenance_due": maintenance_due,
            "consolidation_due": consolidation_due,
            "maintenance_interval_hours": ctx.context.maintenance_interval_hours,
            "consolidation_interval_days": ctx.context.consolidation_interval_days,
            "maintenance_history_count": len(ctx.context.maintenance_history)
        }
    
    @function_tool
    async def _record_maintenance_history(self, ctx: RunContextWrapper,
                                    maintenance_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record maintenance history
        
        Args:
            maintenance_record: Maintenance record to add
            
        Returns:
            Updated history info
        """
        # Add timestamp if not present
        if "timestamp" not in maintenance_record:
            maintenance_record["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add to history
        ctx.context.maintenance_history.append(maintenance_record)
        
        # Trim history if needed
        if len(ctx.context.maintenance_history) > ctx.context.max_history_entries:
            ctx.context.maintenance_history = ctx.context.maintenance_history[-ctx.context.max_history_entries:]
        
        return {
            "success": True,
            "history_count": len(ctx.context.maintenance_history),
            "max_history_entries": ctx.context.max_history_entries,
            "latest_entry_timestamp": ctx.context.maintenance_history[-1].get("timestamp") if ctx.context.maintenance_history else None
        }
    
    @function_tool
    async def _analyze_system_efficiency(self, ctx: RunContextWrapper) -> Dict[str, Any]:
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
            Maintenance results
        """
        with trace(workflow_name="conditioning_maintenance", group_id=self.context.trace_group_id):
            logger.info("Starting conditioning system maintenance")
            start_time = datetime.datetime.now()
            
            try:
                # Run the maintenance orchestrator
                result = await Runner.run(
                    self.maintenance_orchestrator,
                    json.dumps({
                        "last_maintenance_time": self.context.last_maintenance_time.isoformat() if self.context.last_maintenance_time else None,
                        "interval_hours": self.context.maintenance_interval_hours,
                        "consolidation_interval_days": self.context.consolidation_interval_days,
                        "extinction_threshold": self.context.extinction_threshold,
                        "reinforcement_threshold": self.context.reinforcement_threshold
                    }),
                    context=self.context
                )
                
                # Get summary output
                summary_output = result.final_output
                
                # Record completion time
                duration = (datetime.datetime.now() - start_time).total_seconds()
                
                # Update last maintenance time
                self.context.last_maintenance_time = datetime.datetime.now()
                
                # Create maintenance record
                maintenance_record = {
                    "timestamp": self.context.last_maintenance_time.isoformat(),
                    "duration_seconds": duration,
                    "tasks_performed": summary_output.tasks_performed,
                    "associations_modified": summary_output.associations_modified,
                    "traits_adjusted": summary_output.traits_adjusted,
                    "extinction_count": summary_output.extinction_count,
                    "improvements": summary_output.improvements,
                    "next_maintenance_recommendation": summary_output.next_maintenance_recommendation
                }
                
                # Update maintenance history
                await self._record_maintenance_history(RunContextWrapper(context=self.context), maintenance_record)
                
                logger.info(f"Conditioning system maintenance completed in {duration:.2f} seconds")
                return maintenance_record
                
            except Exception as e:
                duration = (datetime.datetime.now() - start_time).total_seconds()
                logger.error(f"Error in maintenance run after {duration:.2f} seconds: {e}")
                return {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "error": str(e),
                    "status": "failed"
                }
    
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
        status = await self._get_maintenance_status(RunContextWrapper(context=self.context))
        
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
