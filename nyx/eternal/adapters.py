# nyx/eternal/adapters.py

from typing import Dict, Any, List, Optional
import asyncio
import logging

# Standalone imports to prevent circular dependencies
from nyx.eternal.meta_learning_system import MetaLearningSystem
from nyx.eternal.dynamic_adaptation_system import DynamicAdaptationSystem
from nyx.eternal.internal_feedback_system import InternalFeedbackSystem

logger = logging.getLogger(__name__)

class MetaLearningAdapter:
    """Adapter for meta-learning system to avoid circular imports"""
    
    def __init__(self):
        self.system = MetaLearningSystem()
    
    async def learn_from_interaction(self, 
                                user_input: str, 
                                response: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from an interaction without direct integration"""
        # Extract features
        features = self._extract_features(user_input, response, context)
        
        # Learn feature importance
        feature_importance = await self.system.learn_feature_importance(features, context.get("success_score", 0.5))
        
        # Select best algorithm
        algorithm = await self.system.select_best_algorithm({
            "type": context.get("interaction_type", "general"),
            "complexity": context.get("complexity", 0.5)
        })
        
        return {
            "feature_importance": feature_importance,
            "algorithm": algorithm
        }
    
    def _extract_features(self, 
                        user_input: str, 
                        response: Dict[str, Any], 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for learning"""
        features = {
            "input_length": len(user_input),
            "response_length": len(str(response.get("content", ""))),
            "response_time": context.get("response_time", 0),
        }
        
        # Add safe context features
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                features[f"context_{key}"] = value
        
        return features

class DynamicAdaptationAdapter:
    """Adapter for dynamic adaptation system to avoid circular imports"""
    
    def __init__(self):
        self.system = DynamicAdaptationSystem()
    
    async def adapt_to_interaction(self, 
                             user_input: str, 
                             response: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt to interaction without direct integration"""
        # Create adaptable context
        adaptable_context = {
            "user_input": user_input,
            "response": str(response.get("content", "")),
            "interaction_type": context.get("interaction_type", "general"),
        }
        
        # Add safe context elements
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                adaptable_context[key] = value
        
        # Detect context change
        change_result = await self.system.detect_context_change(adaptable_context)
        
        # Monitor performance
        performance = await self.system.monitor_performance({
            "success_rate": context.get("success_rate", 0.5),
            "user_satisfaction": context.get("user_satisfaction", 0.5),
            "efficiency": context.get("efficiency", 0.5),
            "response_quality": context.get("response_quality", 0.5)
        })
        
        # Select strategy if significant change
        strategy = None
        if change_result[0]:  # significant change
            strategy = await self.system.select_strategy(adaptable_context, performance)
        
        return {
            "context_change": change_result,
            "performance": performance,
            "strategy": strategy
        }

class InternalFeedbackAdapter:
    """Adapter for internal feedback system to avoid circular imports"""
    
    def __init__(self):
        self.system = InternalFeedbackSystem()
    
    async def provide_feedback(self, 
                         user_input: str, 
                         response: Dict[str, Any], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide feedback without direct integration"""
        # Track performance metrics
        metrics = {
            "response_quality": context.get("response_quality", 0.5),
            "user_satisfaction": context.get("user_satisfaction", 0.5)
        }
        
        quality_stats = {}
        for metric, value in metrics.items():
            quality_stats[metric] = await self.system.track_performance(metric, value)
        
        # Evaluate confidence
        confidence_eval = await self.system.evaluate_confidence(
            context.get("confidence", 0.7),
            context.get("success", True)
        )
        
        # Create evaluable content
        evaluable_content = {
            "text": str(response.get("content", "")),
            "type": context.get("response_type", "general"),
            "metrics": metrics
        }
        
        # Critic evaluation
        critic_evals = {}
        for aspect in ["consistency", "effectiveness", "efficiency"]:
            critic_evals[aspect] = await self.system.critic_evaluate(
                aspect, evaluable_content, context
            )
        
        return {
            "quality_stats": quality_stats,
            "confidence_eval": confidence_eval,
            "critic_evals": critic_evals
        }
