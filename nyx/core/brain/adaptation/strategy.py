# nyx/core/brain/adaptation/strategy.py
import logging
import random
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class StrategySelector:
    """Selects adaptation strategies based on context changes"""
    
    def __init__(self, brain):
        self.brain = brain
        self.strategies = {
            "cross_user_sharing": {
                "name": "Cross-user experience sharing adaptation",
                "description": "Adapts cross-user experience sharing based on context",
                "parameters": {
                    "sharing_threshold": {"min": 0.3, "max": 0.9, "default": 0.7},
                    "enablement": {"min": 0, "max": 1, "default": 1}
                },
                "applicable_changes": ["cross_user_transition", "emotional_change"],
                "success_history": []
            },
            "emotional_expression": {
                "name": "Emotional expression adaptation",
                "description": "Adapts emotional expression based on context",
                "parameters": {
                    "expression_threshold": {"min": 0.4, "max": 0.9, "default": 0.7},
                    "expression_intensity": {"min": 0.1, "max": 1.0, "default": 0.6}
                },
                "applicable_changes": ["emotional_change"],
                "success_history": []
            },
            "memory_prioritization": {
                "name": "Memory retrieval prioritization",
                "description": "Adapts memory retrieval priorities based on context",
                "parameters": {
                    "recency_weight": {"min": 0.3, "max": 0.9, "default": 0.7},
                    "relevance_weight": {"min": 0.4, "max": 0.95, "default": 0.8},
                    "significance_threshold": {"min": 1, "max": 8, "default": 3}
                },
                "applicable_changes": ["scenario_change", "emotional_change"],
                "success_history": []
            },
            "reasoning_depth": {
                "name": "Reasoning depth adaptation",
                "description": "Adapts reasoning depth based on context",
                "parameters": {
                    "reasoning_depth": {"min": 1, "max": 4, "default": 2},
                    "thinking_frequency": {"min": 0.1, "max": 0.8, "default": 0.4}
                },
                "applicable_changes": ["scenario_change"],
                "success_history": []
            }
        }
        
        # Track strategy selection history
        self.selection_history = []
    
    async def select_strategy(self, 
                           context_change: Dict[str, Any],
                           current_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Select an adaptation strategy based on context change
        
        Args:
            context_change: Context change detection results
            current_performance: Current performance metrics
            
        Returns:
            Selected strategy details
        """
        # If no significant change, no strategy needed
        if not context_change.get("significant_change", False):
            return {
                "strategy_selected": False,
                "reason": "No significant context change detected"
            }
        
        # Get the change type
        change_type = context_change.get("change_type", "")
        
        # Find applicable strategies
        applicable_strategies = []
        for strategy_id, strategy in self.strategies.items():
            if change_type in strategy["applicable_changes"]:
                # Calculate selection score
                selection_score = self._calculate_selection_score(
                    strategy_id, strategy, change_type, context_change, current_performance
                )
                
                applicable_strategies.append({
                    "id": strategy_id,
                    "strategy": strategy,
                    "selection_score": selection_score
                })
        
        # If no applicable strategies, return empty result
        if not applicable_strategies:
            return {
                "strategy_selected": False,
                "reason": f"No applicable strategies for change type: {change_type}"
            }
        
        # Sort by selection score
        applicable_strategies.sort(key=lambda x: x["selection_score"], reverse=True)
        
        # Select top strategy
        top_strategy = applicable_strategies[0]
        
        # Determine parameter values for the strategy
        strategy_parameters = await self._determine_strategy_parameters(
            top_strategy["id"], 
            top_strategy["strategy"],
            change_type,
            context_change
        )
        
        # Record the selection
        self.selection_history.append({
            "strategy_id": top_strategy["id"],
            "change_type": change_type,
            "selection_score": top_strategy["selection_score"],
            "parameters": strategy_parameters
        })
        
        # Build result
        result = {
            "strategy_selected": True,
            "strategy_id": top_strategy["id"],
            "strategy_name": top_strategy["strategy"]["name"],
            "confidence": top_strategy["selection_score"],
            "selected_strategy": {
                "id": top_strategy["id"],
                "name": top_strategy["strategy"]["name"],
                "description": top_strategy["strategy"]["description"],
                "parameters": strategy_parameters
            },
            "change_type": change_type,
            "alternative_strategies": [s["id"] for s in applicable_strategies[1:]]
        }
        
        return result
    
    def _calculate_selection_score(self,
                                strategy_id: str,
                                strategy: Dict[str, Any],
                                change_type: str,
                                context_change: Dict[str, Any],
                                current_performance: Dict[str, float]) -> float:
        """
        Calculate selection score for a strategy
        
        Args:
            strategy_id: Strategy ID
            strategy: Strategy details
            change_type: Type of context change
            context_change: Context change details
            current_performance: Current performance metrics
            
        Returns:
            Selection score (higher is better)
        """
        # Start with base score
        score = 0.0
        
        # Higher score for strategies that directly address the change type
        if change_type in strategy["applicable_changes"]:
            score += 0.6
            
            # Even higher score if it's the first applicable change type
            if strategy["applicable_changes"][0] == change_type:
                score += 0.2
        
        # Higher score for strategies with successful history
        success_rate = self._get_strategy_success_rate(strategy_id)
        score += success_rate * 0.3
        
        # Higher score for strategies that address performance issues
        if "success_rate" in current_performance and current_performance["success_rate"] < 0.7:
            # If performance is poor, prioritize strategies that might improve it
            if strategy_id in ["memory_prioritization", "reasoning_depth"]:
                score += 0.2
        
        # Factor in context change confidence
        change_confidence = context_change.get("confidence", 0.5)
        score *= change_confidence
        
        # Add small random factor to prevent always picking the same strategy
        score += random.uniform(0, 0.05)
        
        return min(1.0, score)  # Cap at 1.0
    
    def _get_strategy_success_rate(self, strategy_id: str) -> float:
        """
        Get success rate for a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Success rate (0.0-1.0)
        """
        # If strategy not found, return default
        if strategy_id not in self.strategies:
            return 0.5
        
        # If no history, return default
        success_history = self.strategies[strategy_id]["success_history"]
        if not success_history:
            return 0.5
        
        # Calculate success rate
        return sum(success_history) / len(success_history)
    
    async def _determine_strategy_parameters(self,
                                         strategy_id: str,
                                         strategy: Dict[str, Any],
                                         change_type: str,
                                         context_change: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine parameter values for a strategy
        
        Args:
            strategy_id: Strategy ID
            strategy: Strategy details
            change_type: Type of context change
            context_change: Context change details
            
        Returns:
            Strategy parameters
        """
        # Get parameter definitions
        params = strategy["parameters"]
        
        # Initialize with defaults
        result = {}
        for param_name, param_details in params.items():
            result[param_name] = param_details["default"]
        
        # Adjust based on change type and context
        if strategy_id == "cross_user_sharing" and change_type == "cross_user_transition":
            # Determine direction of transition
            if context_change.get("direction") == "to_cross_user":
                # Transitioning to cross-user, lower threshold
                result["sharing_threshold"] = max(params["sharing_threshold"]["min"], 
                                                params["sharing_threshold"]["default"] - 0.1)
                result["enablement"] = 1  # Enable
            else:
                # Transitioning from cross-user, raise threshold
                result["sharing_threshold"] = min(params["sharing_threshold"]["max"], 
                                                params["sharing_threshold"]["default"] + 0.1)
        
        elif strategy_id == "emotional_expression" and change_type == "emotional_change":
            # Check valence and arousal change
            valence_change = context_change.get("valence_change", 0)
            arousal_change = context_change.get("arousal_change", 0)
            
            if valence_change > 0.2 or arousal_change > 0.2:
                # Significant emotional change, lower expression threshold
                result["expression_threshold"] = max(params["expression_threshold"]["min"], 
                                                  params["expression_threshold"]["default"] - 0.1)
                
                # Increase expression intensity
                result["expression_intensity"] = min(params["expression_intensity"]["max"], 
                                                  params["expression_intensity"]["default"] + 0.1)
        
        elif strategy_id == "memory_prioritization" and change_type == "scenario_change":
            # Scenario change, adjust memory weights
            result["recency_weight"] = min(params["recency_weight"]["max"], 
                                         params["recency_weight"]["default"] + 0.1)
            result["relevance_weight"] = max(params["relevance_weight"]["min"], 
                                           params["relevance_weight"]["default"] - 0.1)
            
        elif strategy_id == "reasoning_depth" and change_type == "scenario_change":
            # Scenario change, adjust reasoning depth
            current_scenario = context_change.get("current_scenario", "")
            
            # Increase depth for complex scenarios
            if current_scenario in ["complex", "philosophical", "technical"]:
                result["reasoning_depth"] = min(params["reasoning_depth"]["max"], 
                                              params["reasoning_depth"]["default"] + 1)
                result["thinking_frequency"] = min(params["thinking_frequency"]["max"], 
                                                 params["thinking_frequency"]["default"] + 0.1)
            else:
                # Decrease depth for simple scenarios
                result["reasoning_depth"] = max(params["reasoning_depth"]["min"], 
                                              params["reasoning_depth"]["default"] - 1)
                result["thinking_frequency"] = max(params["thinking_frequency"]["min"], 
                                                 params["thinking_frequency"]["default"] - 0.1)
        
        return result
    
    async def record_strategy_result(self, 
                                 strategy_id: str, 
                                 success: bool, 
                                 performance_impact: Dict[str, float]) -> Dict[str, Any]:
        """
        Record the result of a strategy application
        
        Args:
            strategy_id: Strategy ID
            success: Whether the strategy was successful
            performance_impact: Impact on performance metrics
            
        Returns:
            Recording result
        """
        # If strategy not found, return error
        if strategy_id not in self.strategies:
            return {"error": f"Strategy not found: {strategy_id}"}
        
        # Add to success history
        self.strategies[strategy_id]["success_history"].append(1 if success else 0)
        
        # Keep history manageable
        if len(self.strategies[strategy_id]["success_history"]) > 20:
            self.strategies[strategy_id]["success_history"] = self.strategies[strategy_id]["success_history"][-20:]
        
        # Calculate new success rate
        success_rate = self._get_strategy_success_rate(strategy_id)
        
        return {
            "strategy_id": strategy_id,
            "success": success,
            "success_rate": success_rate,
            "performance_impact": performance_impact
        }
