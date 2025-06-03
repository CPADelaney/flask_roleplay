# nyx/core/brain/processing/mode_selector.py
import logging
import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ModeSelector:
    """Selects optimal processing mode based on input characteristics"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.selection_history = []
        self.mode_metrics = {
            "serial": {"success_rate": 0.95, "avg_time": 0.0, "usage_count": 0},
            "parallel": {"success_rate": 0.92, "avg_time": 0.0, "usage_count": 0},
            "distributed": {"success_rate": 0.90, "avg_time": 0.0, "usage_count": 0},
            "reflexive": {"success_rate": 0.85, "avg_time": 0.0, "usage_count": 0},
            "agent": {"success_rate": 0.88, "avg_time": 0.0, "usage_count": 0},
            "integrated": {"success_rate": 0.93, "avg_time": 0.0, "usage_count": 0}
        }
        
    async def determine_processing_mode(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Determine optimal processing mode"""
        context = context or {}
        
        # Check for explicit mode request
        if "requested_mode" in context:
            return context["requested_mode"]
        
        # Quick reflexive check
        if self._is_reflexive_pattern(user_input):
            return "reflexive"
        
        # Check for agent suitability
        agent_score = self._calculate_agent_suitability(user_input, context)
        if agent_score > 0.8:
            return "agent" if not context.get("integration_mode") else "integrated"
        
        # Calculate complexity
        complexity = self._calculate_complexity(user_input, context)
        
        # Determine mode based on complexity
        if complexity < 0.3:
            return "reflexive"
        elif complexity < 0.6:
            return "serial"
        elif complexity < 0.8:
            return "parallel"
        else:
            return "distributed"
    
    def _is_reflexive_pattern(self, user_input: str) -> bool:
        """Check if input matches reflexive patterns"""
        input_lower = user_input.lower().strip()
        
        reflexive_patterns = [
            # Greetings
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            # Farewells
            "bye", "goodbye", "see you", "take care",
            # Thanks
            "thanks", "thank you", "appreciate it",
            # Simple acknowledgments
            "ok", "okay", "yes", "no", "sure"
        ]
        
        return input_lower in reflexive_patterns or len(input_lower.split()) < 3
    
    def _calculate_agent_suitability(self, user_input: str, context: Dict[str, Any]) -> float:
        """Calculate suitability for agent processing"""
        agent_indicators = [
            "roleplay", "story", "imagine", "create", "generate",
            "tell me about", "describe", "narrative", "character"
        ]
        
        score = 0.0
        input_lower = user_input.lower()
        
        for indicator in agent_indicators:
            if indicator in input_lower:
                score += 0.2
        
        # Check context for agent hints
        if context.get("prefer_agent"):
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_complexity(self, user_input: str, context: Dict[str, Any]) -> float:
        """Calculate input complexity"""
        # Length factor
        length_score = min(1.0, len(user_input) / 500)
        
        # Word complexity
        words = user_input.split()
        unique_words = len(set(words))
        word_score = min(1.0, unique_words / 50)
        
        # Question complexity
        question_score = 0.0
        if "?" in user_input:
            question_count = user_input.count("?")
            question_score = min(1.0, question_count * 0.3)
        
        # Context complexity
        context_score = 0.0
        if context:
            context_score = min(1.0, len(str(context)) / 1000)
        
        # Combine scores
        complexity = (
            length_score * 0.3 +
            word_score * 0.3 +
            question_score * 0.2 +
            context_score * 0.2
        )
        
        return complexity
    
    def update_mode_metrics(self, mode: str, success: bool, response_time: float):
        """Update metrics for a mode"""
        if mode in self.mode_metrics:
            metrics = self.mode_metrics[mode]
            
            # Update success rate with decay
            decay = 0.95
            success_value = 1.0 if success else 0.0
            metrics["success_rate"] = metrics["success_rate"] * decay + success_value * (1 - decay)
            
            # Update average time
            if metrics["usage_count"] > 0:
                metrics["avg_time"] = (metrics["avg_time"] * metrics["usage_count"] + response_time) / (metrics["usage_count"] + 1)
            else:
                metrics["avg_time"] = response_time
            
            metrics["usage_count"] += 1
    
    async def analyze_mode_usage(self) -> Dict[str, Any]:
        """Analyze mode usage patterns"""
        total_usage = sum(m["usage_count"] for m in self.mode_metrics.values())
        
        if total_usage == 0:
            return {"status": "no_usage_data"}
        
        usage_distribution = {
            mode: metrics["usage_count"] / total_usage
            for mode, metrics in self.mode_metrics.items()
        }
        
        performance_summary = {
            mode: {
                "success_rate": metrics["success_rate"],
                "avg_time": metrics["avg_time"]
            }
            for mode, metrics in self.mode_metrics.items()
        }
        
        return {
            "total_usage": total_usage,
            "usage_distribution": usage_distribution,
            "performance_summary": performance_summary
        }
