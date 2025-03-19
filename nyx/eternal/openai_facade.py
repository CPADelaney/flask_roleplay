# nyx/eternal/openai_facade.py

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Awaitable

# Import from OpenAI Agents SDK implementation
from agents import Agent, Runner, set_default_openai_key

# Import core systems as standalone classes (without importing from your existing modules)
from nyx.integrations.openai_agents_nyx.meta_learning_system import MetaLearningSystem
from nyx.integrations.openai_agents_nyx.dynamic_adaptation_system import DynamicAdaptationSystem
from nyx.integrations.openai_agents_nyx.internal_feedback_system import InternalFeedbackSystem

logger = logging.getLogger(__name__)

class NyxAgentsFacade:
    """
    Facade connecting OpenAI Agents SDK implementation with existing Nyx systems.
    This avoids circular imports by not importing directly from existing Nyx modules.
    """
    
    def __init__(self, user_id: int, conversation_id: int, api_key: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.api_key = api_key
        
        # Initialize systems
        self.meta_learning = MetaLearningSystem()
        self.dynamic_adaptation = DynamicAdaptationSystem()
        self.internal_feedback = InternalFeedbackSystem()
        
        # State tracking
        self.initialized = False
        self.strategy_history = []
        self.context_cache = {}
        
        if api_key:
            set_default_openai_key(api_key)
    
    async def initialize(self):
        """Initialize systems"""
        if self.initialized:
            return
            
        # Initialize default strategies
        self._initialize_default_strategies()
        
        self.initialized = True
        logger.info(f"OpenAI Agents facade initialized for user {self.user_id}")
    
    def _initialize_default_strategies(self):
        """Initialize default strategies for dynamic adaptation"""
        # Register strategies
        self.dynamic_adaptation.register_strategy({
            "id": "default",
            "name": "Default Strategy",
            "description": "Balanced approach for general interactions",
            "parameters": {
                "exploration_rate": 0.2,
                "learning_rate": 0.1,
                "responsiveness": 0.5,
                "creativity": 0.5,
                "persistence": 0.5
            }
        })
        
        # Add more strategies
        self.dynamic_adaptation.register_strategy({
            "id": "creative",
            "name": "Creative Strategy",
            "description": "Enhanced creativity for generative tasks",
            "parameters": {
                "exploration_rate": 0.4,
                "learning_rate": 0.15,
                "responsiveness": 0.7,
                "creativity": 0.9,
                "persistence": 0.3
            }
        })
    
    async def enhance_processing(self, 
                           user_input: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance processing with OpenAI Agents capabilities without direct integration
        
        Args:
            user_input: User's input message
            context: Existing context information
            
        Returns:
            Enhanced context with strategy and feature importance
        """
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
        
        # Extract features for meta-learning
        features = self._extract_features(user_input, context)
        
        # Learn feature importance
        feature_importance = await self.meta_learning.learn_feature_importance(features, 0.5)
        
        # Detect context changes
        context_info = {
            "user_input": user_input,
            "features": features,
            **{k: v for k, v in context.items() if isinstance(v, (str, int, float, bool))}
        }
        
        context_change = await self.dynamic_adaptation.detect_context_change(context_info)
        
        # Select optimal strategy if context changed significantly
        selected_strategy = None
        if context_change[0]:  # significant change
            performance = await self.dynamic_adaptation.monitor_performance({
                "success_rate": 0.5,
                "user_satisfaction": 0.5,
                "efficiency": 0.5,
                "response_quality": 0.5
            })
            
            selected_strategy = await self.dynamic_adaptation.select_strategy(
                context_info, 
                performance
            )
            
            if selected_strategy:
                self.strategy_history.append({
                    "strategy": selected_strategy,
                    "context": context_info
                })
        
        # Return enhanced information
        return {
            "feature_importance": feature_importance,
            "context_change": context_change,
            "selected_strategy": selected_strategy
        }
    
    async def evaluate_response(self, 
                          response: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate response quality using internal feedback
        
        Args:
            response: Response generated by Nyx
            context: Context used for generation
            
        Returns:
            Evaluation results
        """
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
        
        # Track performance metrics
        metrics = {
            "response_quality": context.get("response_quality", 0.5),
            "user_satisfaction": context.get("user_satisfaction", 0.5)
        }
        
        quality_stats = {}
        for metric, value in metrics.items():
            quality_stats[metric] = await self.internal_feedback.track_performance(metric, value)
        
        # Evaluate confidence
        confidence_eval = await self.internal_feedback.evaluate_confidence(
            context.get("confidence", 0.7),
            context.get("success", True)
        )
        
        # Critic evaluation for various aspects
        critic_evals = {}
        for aspect in ["consistency", "effectiveness", "efficiency"]:
            critic_evals[aspect] = await self.internal_feedback.critic_evaluate(
                aspect, response, context
            )
        
        return {
            "quality_stats": quality_stats,
            "confidence_eval": confidence_eval,
            "critic_evals": critic_evals
        }
    
    def _extract_features(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for meta-learning without circular dependencies"""
        # Extract basic features from input
        features = {
            "input_length": len(user_input),
            "word_count": len(user_input.split()),
            "has_question": "?" in user_input,
            "has_command": any(cmd in user_input.lower() for cmd in ["please", "can you", "could you"]),
            "sentiment": self._estimate_sentiment(user_input)
        }
        
        # Add safe context features (avoid complex objects that might cause circular refs)
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                features[f"context_{key}"] = value
        
        return features
    
    def _estimate_sentiment(self, text: str) -> float:
        """Estimate sentiment on scale of -1 to 1 (negative to positive)"""
        # Simple keyword-based sentiment estimation
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "like", "love", "happy", "thanks", "thank"]
        negative_words = ["bad", "terrible", "awful", "horrible", "hate", "dislike", "sad", "angry", "disappointed"]
        
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
