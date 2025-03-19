# nyx/eternal/dynamic_adaptation_system.py

import asyncio
import json
import logging
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import math
import random

logger = logging.getLogger(__name__)

class DynamicAdaptationSystem:
    """
    System for dynamically adapting to changing contexts and selecting optimal strategies.
    Detects context changes and adapts behavior accordingly.
    """
    
    def __init__(self):
        self.context_history = []
        self.max_history_size = 20
        self.strategies = {}
        self.strategy_history = []
        self.context_change_threshold = 0.3
        self.performance_history = []
        
        # Initialize with default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize system with default strategies"""
        self.register_strategy({
            "id": "balanced",
            "name": "Balanced Approach",
            "description": "A balanced approach with moderate exploration and adaptation",
            "parameters": {
                "exploration_rate": 0.2,
                "adaptation_rate": 0.15,
                "risk_tolerance": 0.5,
                "innovation_level": 0.5,
                "precision_focus": 0.5
            }
        })
        
        self.register_strategy({
            "id": "exploratory",
            "name": "Exploratory Strategy",
            "description": "High exploration rate with focus on discovering new patterns",
            "parameters": {
                "exploration_rate": 0.4,
                "adaptation_rate": 0.2,
                "risk_tolerance": 0.7,
                "innovation_level": 0.8,
                "precision_focus": 0.3
            }
        })
        
        self.register_strategy({
            "id": "conservative",
            "name": "Conservative Strategy",
            "description": "Low risk with high precision focus",
            "parameters": {
                "exploration_rate": 0.1,
                "adaptation_rate": 0.1,
                "risk_tolerance": 0.2,
                "innovation_level": 0.3,
                "precision_focus": 0.8
            }
        })
        
        self.register_strategy({
            "id": "adaptive",
            "name": "Highly Adaptive Strategy",
            "description": "Focuses on quick adaptation to changes",
            "parameters": {
                "exploration_rate": 0.3,
                "adaptation_rate": 0.3,
                "risk_tolerance": 0.6,
                "innovation_level": 0.6,
                "precision_focus": 0.4
            }
        })
    
    def register_strategy(self, strategy: Dict[str, Any]) -> None:
        """
        Register a new strategy in the system.
        
        Args:
            strategy: Strategy definition to register
        """
        if "id" not in strategy:
            strategy["id"] = f"strategy_{len(self.strategies) + 1}"
            
        self.strategies[strategy["id"]] = strategy
    
    async def detect_context_change(self, 
                              context: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Detect if there has been a significant change in context.
        
        Args:
            context: Current context information
            
        Returns:
            Tuple containing:
            - Whether a significant change was detected
            - Magnitude of the change (0.0-1.0)
            - Description of the change
        """
        # Add current context to history
        self.context_history.append(context)
        if len(self.context_history) > self.max_history_size:
            self.context_history.pop(0)
        
        # If we don't have enough history, no change detected
        if len(self.context_history) < 2:
            return (False, 0.0, "Insufficient context history")
        
        # Compare current context with previous
        current = context
        previous = self.context_history[-2]
        
        # Extract relevant features for comparison
        change_magnitude = self._calculate_context_difference(current, previous)
        
        # Determine if change is significant
        significant_change = change_magnitude > self.context_change_threshold
        
        # Generate change description
        if significant_change:
            description = self._generate_change_description(current, previous, change_magnitude)
        else:
            description = "No significant context change detected"
        
        return (significant_change, change_magnitude, description)
    
    async def monitor_performance(self, 
                            metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitor performance metrics and detect trends.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Performance analysis
        """
        # Add metrics to history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Trim history if needed
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
        # Calculate trends
        trends = self._calculate_performance_trends(metrics)
        
        # Generate insights
        insights = self._generate_performance_insights(metrics, trends)
        
        return {
            "current": metrics,
            "trends": trends,
            "insights": insights,
            "history_points": len(self.performance_history)
        }
    
    async def select_strategy(self, 
                       context: Dict[str, Any], 
                       performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the optimal strategy for the current context.
        
        Args:
            context: Current context information
            performance: Current performance metrics
            
        Returns:
            Selected strategy
        """
        # Extract context features
        context_features = self._extract_context_features(context)
        
        # Calculate context complexity
        complexity = self._calculate_context_complexity(context)
        
        # Calculate volatility
        volatility = self._calculate_context_volatility()
        
        # Calculate strategy scores
        strategy_scores = {}
        for strategy_id, strategy in self.strategies.items():
            strategy_scores[strategy_id] = self._calculate_strategy_score(
                strategy, context_features, performance, complexity, volatility
            )
        
        # Select best strategy
        if not strategy_scores:
            # If no strategies, return balanced
            selected_id = "balanced"
            if selected_id not in self.strategies:
                return {
                    "id": "default",
                    "name": "Default Strategy",
                    "parameters": {
                        "exploration_rate": 0.2,
                        "adaptation_rate": 0.15,
                        "risk_tolerance": 0.5,
                        "innovation_level": 0.5,
                        "precision_focus": 0.5
                    }
                }
        else:
            selected_id = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # Get the selected strategy
        selected_strategy = self.strategies[selected_id]
        
        # Record strategy selection
        self.strategy_history.append({
            "timestamp": datetime.now().isoformat(),
            "strategy_id": selected_id,
            "context_summary": self._summarize_context(context),
            "performance_summary": self._summarize_performance(performance),
            "score": strategy_scores[selected_id]
        })
        
        return selected_strategy
    
    def _calculate_context_difference(self, 
                                    current: Dict[str, Any], 
                                    previous: Dict[str, Any]) -> float:
        """Calculate the difference between two contexts"""
        # Focus on key elements common to both contexts
        common_keys = set(current.keys()) & set(previous.keys())
        if not common_keys:
            return 1.0  # Maximum difference if no common keys
        
        differences = []
        for key in common_keys:
            # Skip complex nested structures, consider only scalar values
            if isinstance(current[key], (str, int, float, bool)) and isinstance(previous[key], (str, int, float, bool)):
                if isinstance(current[key], bool) or isinstance(previous[key], bool):
                    # For boolean values, difference is either 0 or 1
                    diff = 0.0 if current[key] == previous[key] else 1.0
                elif isinstance(current[key], str) or isinstance(previous[key], str):
                    # For string values, difference is either 0 or 1
                    diff = 0.0 if str(current[key]) == str(previous[key]) else 1.0
                else:
                    # For numeric values, calculate normalized difference
                    max_val = max(abs(float(current[key])), abs(float(previous[key])))
                    if max_val > 0:
                        diff = abs(float(current[key]) - float(previous[key])) / max_val
                    else:
                        diff = 0.0
                differences.append(diff)
        
        if not differences:
            return 0.5  # Middle value if no comparable elements
            
        # Return average difference
        return sum(differences) / len(differences)
    
    def _generate_change_description(self, 
                                   current: Dict[str, Any], 
                                   previous: Dict[str, Any], 
                                   magnitude: float) -> str:
        """Generate a description of the context change"""
        changes = []
        
        # Check for new or modified keys
        for key in current:
            if key in previous:
                if current[key] != previous[key] and isinstance(current[key], (str, int, float, bool)):
                    changes.append(f"{key} changed from {previous[key]} to {current[key]}")
            else:
                changes.append(f"New element: {key}")
        
        # Check for removed keys
        for key in previous:
            if key not in current:
                changes.append(f"Removed element: {key}")
        
        if not changes:
            return f"Context changed with magnitude {magnitude:.2f}"
            
        change_desc = ", ".join(changes[:3])  # Limit to first 3 changes
        if len(changes) > 3:
            change_desc += f", and {len(changes) - 3} more changes"
            
        return f"Context changed ({magnitude:.2f}): {change_desc}"
    
    def _calculate_performance_trends(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate trends in performance metrics"""
        trends = {}
        
        if len(self.performance_history) < 2:
            # Not enough history for trends
            for metric, value in metrics.items():
                trends[metric] = {
                    "direction": "stable",
                    "magnitude": 0.0
                }
            return trends
        
        # Calculate trends for each metric
        for metric, current_value in metrics.items():
            # Find previous values for this metric
            previous_values = []
            for history_point in self.performance_history[:-1]:  # Skip current point
                if metric in history_point["metrics"]:
                    previous_values.append(history_point["metrics"][metric])
            
            if not previous_values:
                trends[metric] = {
                    "direction": "stable",
                    "magnitude": 0.0
                }
                continue
                
            # Calculate average of previous values
            avg_previous = sum(previous_values) / len(previous_values)
            
            # Calculate difference
            diff = current_value - avg_previous
            
            # Determine direction and magnitude
            if abs(diff) < 0.05:  # Small threshold for stability
                direction = "stable"
                magnitude = 0.0
            else:
                direction = "improving" if diff > 0 else "declining"
                magnitude = min(1.0, abs(diff))
                
            trends[metric] = {
                "direction": direction,
                "magnitude": magnitude,
                "diff_from_avg": diff
            }
        
        return trends
    
    def _generate_performance_insights(self, 
                                     metrics: Dict[str, float], 
                                     trends: Dict[str, Any]) -> List[str]:
        """Generate insights based on performance metrics and trends"""
        insights = []
        
        # Check for significant improvements
        improvements = [metric for metric, trend in trends.items() 
                       if trend["direction"] == "improving" and trend["magnitude"] > 0.1]
        if improvements:
            metrics_list = ", ".join(improvements)
            insights.append(f"Significant improvement in {metrics_list}")
        
        # Check for significant declines
        declines = [metric for metric, trend in trends.items() 
                   if trend["direction"] == "declining" and trend["magnitude"] > 0.1]
        if declines:
            metrics_list = ", ".join(declines)
            insights.append(f"Significant decline in {metrics_list}")
        
        # Check for overall performance
        avg_performance = sum(metrics.values()) / len(metrics) if metrics else 0.5
        if avg_performance > 0.8:
            insights.append("Overall performance is excellent")
        elif avg_performance < 0.4:
            insights.append("Overall performance is concerning")
        
        # Check for volatility
        volatility = self._calculate_performance_volatility()
        if volatility > 0.2:
            insights.append("Performance metrics show high volatility")
        
        return insights
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from context for strategy selection"""
        features = {}
        
        # Extract basic scalars
        for key, value in context.items():
            if isinstance(value, (int, float, bool)):
                if isinstance(value, bool):
                    features[key] = 1.0 if value else 0.0
                else:
                    features[key] = float(value)
        
        # Extract feature from user input if present
        if "user_input" in context and isinstance(context["user_input"], str):
            features["input_length"] = min(1.0, len(context["user_input"]) / 500.0)
            features["input_complexity"] = min(1.0, len(set(context["user_input"].split())) / 100.0)
        
        # Calculate volatility feature
        features["context_volatility"] = self._calculate_context_volatility()
        
        return features
    
    def _calculate_context_complexity(self, context: Dict[str, Any]) -> float:
        """Calculate the complexity of the current context"""
        # Count the number of nested elements and total elements
        total_elements = 0
        nested_elements = 0
        max_depth = 0
        
        def count_elements(obj, depth=0):
            nonlocal total_elements, nested_elements, max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(obj, dict):
                total_elements += len(obj)
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        nested_elements += 1
                        count_elements(value, depth + 1)
            elif isinstance(obj, list):
                total_elements += len(obj)
                for item in obj:
                    if isinstance(item, (dict, list)):
                        nested_elements += 1
                        count_elements(item, depth + 1)
        
        count_elements(context)
        
        # Calculate complexity factors
        size_factor = min(1.0, total_elements / 50.0)  # Normalize by expecting max 50 elements
        nesting_factor = min(1.0, nested_elements / 10.0)  # Normalize by expecting max 10 nested elements
        depth_factor = min(1.0, max_depth / 5.0)  # Normalize by expecting max depth of 5
        
        # Combine factors with weights
        complexity = (
            size_factor * 0.4 +
            nesting_factor * 0.3 +
            depth_factor * 0.3
        )
        
        return complexity
    
    def _calculate_context_volatility(self) -> float:
        """Calculate the volatility of the context over time"""
        if len(self.context_history) < 3:
            return 0.0  # Not enough history to calculate volatility
        
        # Calculate pairwise differences between consecutive contexts
        differences = []
        for i in range(1, len(self.context_history)):
            diff = self._calculate_context_difference(
                self.context_history[i], 
                self.context_history[i-1]
            )
            differences.append(diff)
        
        # Calculate variance of differences
        mean_diff = sum(differences) / len(differences)
        variance = sum((diff - mean_diff) ** 2 for diff in differences) / len(differences)
        
        # Normalize to [0,1]
        volatility = min(1.0, math.sqrt(variance) * 3.0)  # Scale to make values more meaningful
        
        return volatility
    
    def _calculate_strategy_score(self,
                                strategy: Dict[str, Any], 
                                context_features: Dict[str, float],
                                performance: Dict[str, Any],
                                complexity: float,
                                volatility: float) -> float:
        """Calculate a score for how well a strategy matches the current context"""
        params = strategy["parameters"]
        
        # Base score starts at 0.5
        score = 0.5
        
        # Adjust based on complexity
        # Higher complexity prefers higher adaptation rate
        complexity_match = 1.0 - abs(complexity - params["adaptation_rate"])
        score += complexity_match * 0.1
        
        # Adjust based on volatility
        # Higher volatility prefers higher exploration rate
        volatility_match = 1.0 - abs(volatility - params["exploration_rate"])
        score += volatility_match * 0.1
        
        # Adjust based on performance trends
        trends = performance.get("trends", {})
        
        # If performance is declining, prefer more exploratory strategies
        declining_metrics = sum(1 for t in trends.values() if t.get("direction") == "declining")
        if declining_metrics > 0:
            exploration_bonus = params["exploration_rate"] * 0.1 * declining_metrics
            score += exploration_bonus
        
        # If performance is good and stable, prefer more conservative strategies
        stable_good_metrics = sum(1 for m, t in zip(performance.get("current", {}).values(), trends.values()) 
                                if m > 0.7 and t.get("direction") in ["stable", "improving"])
        if stable_good_metrics > 0:
            precision_bonus = params["precision_focus"] * 0.1 * stable_good_metrics
            score += precision_bonus
        
        # Adjust based on history - avoid using the same strategy too many times in a row
        recency_penalty = 0.0
        for i, history_item in enumerate(reversed(self.strategy_history)):
            if history_item["strategy_id"] == strategy["id"]:
                recency_penalty += 0.05 * (0.8 ** i)  # Exponential decay with distance
        
        score -= min(0.2, recency_penalty)  # Cap penalty
        
        # Ensure score is in [0,1] range
        return min(1.0, max(0.0, score))
    
    def _calculate_performance_volatility(self) -> float:
        """Calculate the volatility of performance metrics over time"""
        if len(self.performance_history) < 3:
            return 0.0  # Not enough history
        
        # Extract all metric values
        metric_values = {}
        
        for history_point in self.performance_history:
            for metric, value in history_point["metrics"].items():
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)
        
        # Calculate standard deviation for each metric
        std_devs = []
        for values in metric_values.values():
            if len(values) >= 3:  # Need at least 3 points
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std_devs.append(math.sqrt(variance))
        
        if not std_devs:
            return 0.0
            
        # Average standard deviation across metrics
        avg_std_dev = sum(std_devs) / len(std_devs)
        
        # Normalize to [0,1] with reasonable scaling
        volatility = min(1.0, avg_std_dev * 3.0)
        
        return volatility
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the context for history records"""
        summary = {}
        
        # Include basic scalar values
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
        
        # Add derived measures
        summary["complexity"] = self._calculate_context_complexity(context)
        summary["volatility"] = self._calculate_context_volatility()
        
        return summary
    
    def _summarize_performance(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of performance for history records"""
        summary = {}
        
        # Include current metrics
        current = performance.get("current", {})
        summary["metrics"] = current
        
        # Include average
        if current:
            summary["average"] = sum(current.values()) / len(current)
        
        # Include volatility
        summary["volatility"] = self._calculate_performance_volatility()
        
        return summary
