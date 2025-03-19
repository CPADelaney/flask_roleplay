# nyx/eternal/meta_learning_system.py

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

class MetaLearningSystem:
    """
    System for meta-learning from interactions to improve over time.
    Handles feature importance learning and algorithm selection.
    """
    
    def __init__(self):
        self.feature_importance = {}
        self.algorithm_performance = {}
        self.learning_cycles = 0
        self.feature_history = []
        self.convergence_threshold = 0.05
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.min_samples_required = 5
        
    async def learn_feature_importance(self, 
                                 features: Dict[str, Any], 
                                 success_score: float) -> Dict[str, float]:
        """
        Learn the importance of different features based on success.
        
        Args:
            features: Feature values for the interaction
            success_score: How successful the interaction was (0.0-1.0)
            
        Returns:
            Updated feature importance scores
        """
        # Initialize importance for new features
        for feature in features:
            if feature not in self.feature_importance:
                self.feature_importance[feature] = 0.5  # Start with neutral importance
        
        # Track feature history for convergence calculation
        old_importance = {k: v for k, v in self.feature_importance.items()}
        self.feature_history.append(old_importance)
        if len(self.feature_history) > 10:  # Keep only recent history
            self.feature_history.pop(0)
        
        # Update importance based on success
        for feature, value in features.items():
            # Convert feature value to normalized float if possible
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                # Normalize to [0,1] based on typical ranges
                value = min(1.0, max(0.0, float(value) / 100.0))
            else:
                # For string or other values, just use presence (1.0)
                value = 1.0
                
            # Calculate correlation with success
            correlation = value * success_score
            
            # Update importance with learning rate
            current_importance = self.feature_importance[feature]
            self.feature_importance[feature] = current_importance * (1 - self.learning_rate) + correlation * self.learning_rate
            
            # Ensure importance stays in [0,1] range
            self.feature_importance[feature] = min(1.0, max(0.0, self.feature_importance[feature]))
        
        # Check for convergence
        if len(self.feature_history) >= self.min_samples_required:
            converged = self._check_convergence()
            if converged:
                self.learning_rate *= 0.9  # Reduce learning rate as we converge
        
        self.learning_cycles += 1
        return self.feature_importance
    
    async def select_best_algorithm(self, context: Dict[str, Any]) -> str:
        """
        Select the best algorithm for the given context.
        
        Args:
            context: Context information for algorithm selection
            
        Returns:
            Name of the selected algorithm
        """
        available_algorithms = [
            "gradient_boosting",
            "neural_network",
            "random_forest", 
            "reinforcement_learning",
            "bayesian_inference"
        ]
        
        # Initialize performance tracking if needed
        for algo in available_algorithms:
            if algo not in self.algorithm_performance:
                self.algorithm_performance[algo] = {
                    "success_rate": 0.5,
                    "samples": 0,
                    "last_used": None
                }
        
        # Decide between exploration and exploitation
        if random.random() < self._calculate_exploration_rate():
            # Exploration: Pick a less-used algorithm
            samples = [self.algorithm_performance[algo]["samples"] for algo in available_algorithms]
            min_samples = min(max(1, s) for s in samples)
            exploration_weights = [
                max(0.1, min_samples / max(1, self.algorithm_performance[algo]["samples"]))
                for algo in available_algorithms
            ]
            total_weight = sum(exploration_weights)
            probabilities = [w / total_weight for w in exploration_weights]
            selected_algorithm = np.random.choice(available_algorithms, p=probabilities)
        else:
            # Exploitation: Pick the best performing algorithm
            context_type = context.get("type", "general")
            complexity = context.get("complexity", 0.5)
            
            # Select algorithms based on context
            if context_type == "classification":
                if complexity > 0.7:
                    candidates = ["neural_network", "gradient_boosting"]
                else:
                    candidates = ["random_forest", "gradient_boosting"]
            elif context_type == "regression":
                if complexity > 0.7:
                    candidates = ["neural_network", "gradient_boosting"]
                else:
                    candidates = ["bayesian_inference", "random_forest"]
            elif context_type == "reinforcement":
                candidates = ["reinforcement_learning"]
            else:  # general
                candidates = available_algorithms
            
            # Find the best performing algorithm among candidates
            best_score = -1
            selected_algorithm = candidates[0]
            for algo in candidates:
                score = self.algorithm_performance[algo]["success_rate"]
                if score > best_score:
                    best_score = score
                    selected_algorithm = algo
        
        # Update usage statistics
        self.algorithm_performance[selected_algorithm]["last_used"] = datetime.now().isoformat()
        self.algorithm_performance[selected_algorithm]["samples"] += 1
        
        return selected_algorithm
    
    async def update_algorithm_performance(self, 
                                     algorithm: str, 
                                     success_rate: float) -> None:
        """
        Update the performance metrics for an algorithm.
        
        Args:
            algorithm: Name of the algorithm
            success_rate: Success rate of the algorithm (0.0-1.0)
        """
        if algorithm not in self.algorithm_performance:
            self.algorithm_performance[algorithm] = {
                "success_rate": success_rate,
                "samples": 1,
                "last_used": datetime.now().isoformat()
            }
            return
            
        # Update with exponential moving average
        current = self.algorithm_performance[algorithm]["success_rate"]
        samples = self.algorithm_performance[algorithm]["samples"]
        alpha = 2 / (samples + 1)  # EMA adjustment factor
        
        new_rate = current * (1 - alpha) + success_rate * alpha
        self.algorithm_performance[algorithm]["success_rate"] = new_rate
        self.algorithm_performance[algorithm]["samples"] += 1
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the learning process.
        
        Returns:
            Statistics about learning
        """
        return {
            "feature_importance": self.feature_importance,
            "algorithm_performance": self.algorithm_performance,
            "learning_cycles": self.learning_cycles,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "convergence": self._check_convergence() if len(self.feature_history) >= self.min_samples_required else False
        }
    
    def _check_convergence(self) -> bool:
        """Check if feature importance has converged"""
        if len(self.feature_history) < 2:
            return False
            
        # Compare most recent with 3 steps back (if available)
        steps_back = min(3, len(self.feature_history) - 1)
        recent = self.feature_history[-1]
        older = self.feature_history[-1 - steps_back]
        
        # Calculate max difference in any feature
        max_diff = 0.0
        for feature in recent:
            if feature in older:
                diff = abs(recent[feature] - older[feature])
                max_diff = max(max_diff, diff)
        
        return max_diff < self.convergence_threshold
    
    def _calculate_exploration_rate(self) -> float:
        """Calculate current exploration rate with decay"""
        min_rate = 0.05  # Minimum exploration rate
        decay_factor = 0.99  # Decay per learning cycle
        current_rate = self.exploration_rate * (decay_factor ** self.learning_cycles)
        return max(min_rate, current_rate)
