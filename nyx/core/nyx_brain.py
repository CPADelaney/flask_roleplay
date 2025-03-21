import logging
import asyncio
import json
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.emotional_core import EmotionalCore
from nyx.core.memory_core import MemoryCore
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.experience_interface import ExperienceInterface
from nyx.core.internal_feedback_system import InternalFeedbackSystem

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

class NyxBrain:
    """
    Central integration point for all Nyx systems.
    Handles cross-component communication and provides a unified API.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core components
        self.emotional_core = EmotionalCore()
        self.memory_core = None  # Initialized in initialize()
        self.reflection_engine = ReflectionEngine()
        self.experience_interface = None  # Initialized in initialize()
        self.internal_feedback = None  # Initialized in initialize()
        self.dynamic_adaptation = None  # Initialized in initialize()
        
        # State tracking
        self.initialized = False
        self.last_interaction = datetime.now()
        self.interaction_count = 0
        
        # Bidirectional influence settings
        self.memory_to_emotion_influence = 0.3  # How much memories influence emotions
        self.emotion_to_memory_influence = 0.4  # How much emotions influence memory retrieval
        
        # Performance monitoring
        self.performance_metrics = {
            "memory_operations": 0,
            "emotion_updates": 0,
            "reflections_generated": 0,
            "experiences_shared": 0,
            "response_times": []
        }
        
        # Caching
        self.context_cache = {}
        
        # Singleton registry
        self._instance_count = 0
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'NyxBrain':
        """Get or create a singleton instance for the specified user and conversation"""
        # Use a key for the specific user/conversation
        key = f"brain_{user_id}_{conversation_id}"
        
        # Check if instance exists in a global registry
        if not hasattr(cls, '_instances'):
            cls._instances = {}
            
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.initialize()
            cls._instances[key] = instance
        
        return cls._instances[key]
    
    async def initialize(self):
        """Initialize all subsystems"""
        if self.initialized:
            return
        
        # Increment initialization counter
        self._instance_count += 1
        logger.info(f"Initializing NyxBrain {self._instance_count} for user {self.user_id}")
        
        # Initialize memory system
        self.memory_core = MemoryCore(self.user_id, self.conversation_id)
        await self.memory_core.initialize()
        
        # Initialize experience interface with memory core and emotional core
        self.experience_interface = ExperienceInterface(self.memory_core, self.emotional_core)
        
        # Initialize internal feedback system
        self.internal_feedback = InternalFeedbackSystem()
        
        # Initialize dynamic adaptation system
        self.dynamic_adaptation = DynamicAdaptationSystem()
        
        self.initialized = True
        logger.info(f"NyxBrain initialized for user {self.user_id}")
    
    async def process_input(self, 
                          user_input: str, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input and update all systems.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with relevant memories, emotional state, etc.
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        # Update interaction tracking
        self.last_interaction = datetime.now()
        self.interaction_count += 1
        
        # Initialize context
        context = context or {}
        
        # Process emotional impact of input
        emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
        emotional_state = self.emotional_core.update_from_stimuli(emotional_stimuli)
        
        # Add emotional state to context for memory retrieval
        context["emotional_state"] = emotional_state
        
        # Retrieve relevant memories
        memories = await self.memory_core.retrieve_memories(
            query=user_input,
            context=context
        )
        self.performance_metrics["memory_operations"] += 1
        
        # Update emotional state based on retrieved memories
        if memories:
            memory_emotional_impact = await self._calculate_memory_emotional_impact(memories)
            # Apply memory-to-emotion influence
            for emotion, value in memory_emotional_impact.items():
                self.emotional_core.update_emotion(emotion, value * self.memory_to_emotion_influence)
            
            # Get updated emotional state
            emotional_state = self.emotional_core.get_emotional_state()
        
        self.performance_metrics["emotion_updates"] += 1
        
        # Check if experience sharing is requested
        should_share_experience = self._should_share_experience(user_input, context)
        experience_result = None
        
        if should_share_experience:
            # Retrieve and format experience
            experience_result = await self.experience_interface.handle_experience_sharing_request(
                user_query=user_input,
                context_data=context
            )
            self.performance_metrics["experiences_shared"] += 1
        
        # Add memory of this interaction
        memory_text = f"User said: {user_input}"
        
        memory_id = await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="observation",
            memory_scope="game",
            significance=5,
            tags=["interaction", "user_input"],
            metadata={
                "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Check for context change
        context_change_result = None
        if self.dynamic_adaptation:
            # Prepare context for change detection
            context_for_adaptation = {
                "user_input": user_input,
                "emotional_state": emotional_state,
                "memories_retrieved": len(memories),
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "interaction_count": self.interaction_count
            }
            
            # Detect context change
            context_change_result = await self.dynamic_adaptation.detect_context_change(context_for_adaptation)
        
        # Calculate response time
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        self.performance_metrics["response_times"].append(response_time)
        
        # Return processing results
        return {
            "user_input": user_input,
            "emotional_state": emotional_state,
            "memories": memories,
            "memory_count": len(memories),
            "has_experience": experience_result["has_experience"] if experience_result else False,
            "experience_response": experience_result["response_text"] if experience_result and experience_result["has_experience"] else None,
            "memory_id": memory_id,
            "response_time": response_time,
            "context_change": context_change_result
        }
    
    async def generate_response(self, 
                             user_input: str, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete response to user input.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Response data including main message and supporting information
        """
        # Process the input first
        processing_result = await self.process_input(user_input, context)
        
        # Determine if experience response should be used
        if processing_result["has_experience"]:
            main_response = processing_result["experience_response"]
            response_type = "experience"
        else:
            # No specific experience to share, generate standard response
            # In a real implementation, this would call an LLM or other response generation system
            main_response = "I acknowledge your message."
            response_type = "standard"
        
        # Determine if emotion should be expressed
        should_express_emotion = self.emotional_core.should_express_emotion()
        emotional_expression = None
        
        if should_express_emotion:
            emotional_expression = self.emotional_core.get_expression_for_emotion()
        
        # Package the response
        response_data = {
            "message": main_response,
            "response_type": response_type,
            "emotional_state": processing_result["emotional_state"],
            "emotional_expression": emotional_expression,
            "memories_used": [m["id"] for m in processing_result["memories"]],
            "memory_count": processing_result["memory_count"]
        }
        
        # Add memory of this response
        await self.memory_core.add_memory(
            memory_text=f"I responded: {main_response}",
            memory_type="observation",
            memory_scope="game",
            significance=5,
            tags=["interaction", "nyx_response"],
            metadata={
                "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                "timestamp": datetime.now().isoformat(),
                "response_type": response_type
            }
        )
        
        # Evaluate the response if internal feedback system is available
        if self.internal_feedback:
            evaluation = await self.internal_feedback.critic_evaluate(
                aspect="effectiveness",
                content={"text": main_response, "type": response_type},
                context={"user_input": user_input}
            )
            
            # Add evaluation to response data
            response_data["evaluation"] = evaluation
        
        return response_data
    
    async def create_reflection(self, 
                             topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a reflection on memories.
        
        Args:
            topic: Optional topic to focus reflection on
            
        Returns:
            Reflection data
        """
        if not self.initialized:
            await self.initialize()
        
        # Use memory core's reflection creation
        reflection_result = await self.memory_core.create_reflection_from_memories(topic=topic)
        self.performance_metrics["reflections_generated"] += 1
        
        return reflection_result
    
    async def create_abstraction(self,
                              memory_ids: List[str],
                              pattern_type: str = "behavior") -> Dict[str, Any]:
        """
        Create a higher-level abstraction from specific memories.
        
        Args:
            memory_ids: IDs of memories to abstract from
            pattern_type: Type of pattern to identify
            
        Returns:
            Abstraction data
        """
        if not self.initialized:
            await self.initialize()
        
        # Use memory core's abstraction creation
        return await self.memory_core.create_abstraction_from_memories(
            memory_ids=memory_ids,
            pattern_type=pattern_type
        )
    
    async def retrieve_experiences(self,
                                query: str,
                                scenario_type: Optional[str] = None,
                                limit: int = 3) -> Dict[str, Any]:
        """
        Retrieve experiences relevant to a query.
        
        Args:
            query: Search query
            scenario_type: Optional scenario type to filter by
            limit: Maximum number of experiences to return
            
        Returns:
            Experience retrieval results
        """
        if not self.initialized:
            await self.initialize()
        
        # Use experience interface for retrieving experiences
        experiences = await self.experience_interface.retrieve_experiences_enhanced(
            query=query,
            scenario_type=scenario_type,
            limit=limit
        )
        
        return {
            "experiences": experiences,
            "count": len(experiences),
            "query": query,
            "scenario_type": scenario_type
        }
    
    async def construct_narrative(self,
                               topic: str,
                               chronological: bool = True,
                               limit: int = 5) -> Dict[str, Any]:
        """
        Construct a narrative from memories about a topic.
        
        Args:
            topic: Topic for narrative
            chronological: Whether to maintain chronological order
            limit: Maximum number of memories to include
            
        Returns:
            Narrative data
        """
        if not self.initialized:
            await self.initialize()
        
        # Use memory core's narrative construction
        return await self.memory_core.construct_narrative_from_memories(
            topic=topic,
            chronological=chronological,
            limit=limit
        )
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance on all subsystems"""
        if not self.initialized:
            await self.initialize()
        
        # Run memory maintenance
        memory_result = await self.memory_core.run_maintenance()
        
        # Perform additional maintenance as needed
        # (other components don't need routine maintenance)
        
        return {
            "memory_maintenance": memory_result,
            "maintenance_time": datetime.now().isoformat()
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about all systems"""
        if not self.initialized:
            await self.initialize()
        
        # Get memory stats
        memory_stats = await self.memory_core.get_memory_stats()
        
        # Get emotional state
        emotional_state = self.emotional_core.get_emotional_state()
        dominant_emotion, dominant_value = self.emotional_core.get_dominant_emotion()
        
        # Get performance metrics
        avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
        
        # Generate introspection text
        introspection = await self.reflection_engine.generate_introspection(
            memory_stats=memory_stats,
            player_model=None  # Player model would be provided in real implementation
        )
        
        return {
            "memory_stats": memory_stats,
            "emotional_state": {
                "emotions": emotional_state,
                "dominant_emotion": dominant_emotion,
                "dominant_value": dominant_value,
                "valence": self.emotional_core.get_emotional_valence(),
                "arousal": self.emotional_core.get_emotional_arousal()
            },
            "interaction_stats": {
                "interaction_count": self.interaction_count,
                "last_interaction": self.last_interaction.isoformat()
            },
            "performance_metrics": {
                "memory_operations": self.performance_metrics["memory_operations"],
                "emotion_updates": self.performance_metrics["emotion_updates"],
                "reflections_generated": self.performance_metrics["reflections_generated"],
                "experiences_shared": self.performance_metrics["experiences_shared"],
                "avg_response_time": avg_response_time
            },
            "introspection": introspection
        }
    
    def _should_share_experience(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Determine if we should share an experience based on input and context"""
        # Check for explicit experience requests
        explicit_request = any(phrase in user_input.lower() for phrase in 
                             ["remember", "recall", "tell me about", "have you done", 
                              "previous", "before", "past", "experience"])
        
        if explicit_request:
            return True
        
        # Check if it's a question that could benefit from experience sharing
        is_question = user_input.endswith("?") or user_input.lower().startswith(("what", "how", "when", "where", "why", "who", "can", "could", "do", "did"))
        
        if is_question and "share_experiences" in context and context["share_experiences"]:
            return True
        
        # Default to not sharing experiences unless explicitly requested
        return False
    
    async def _calculate_memory_emotional_impact(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate emotional impact from relevant memories"""
        impact = {}
        
        for memory in memories:
            # Extract emotional context
            emotional_context = memory.get("metadata", {}).get("emotional_context", {})
            
            if not emotional_context:
                continue
                
            # Get primary emotion
            primary_emotion = emotional_context.get("primary_emotion")
            primary_intensity = emotional_context.get("primary_intensity", 0.5)
            
            if primary_emotion:
                # Calculate impact based on relevance and recency
                relevance = memory.get("relevance", 0.5)
                
                # Get timestamp if available
                timestamp_str = memory.get("metadata", {}).get("timestamp")
                recency_factor = 1.0
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    days_old = (datetime.now() - timestamp).days
                    recency_factor = max(0.5, 1.0 - (days_old / 30))  # Decay over 30 days, minimum 0.5
                
                # Calculate final impact value
                impact_value = primary_intensity * relevance * recency_factor * 0.1
                
                # Add to impact dict
                if primary_emotion not in impact:
                    impact[primary_emotion] = 0
                impact[primary_emotion] += impact_value
        
        return impact
    
    # Integration with Dynamic Adaptation System
    
    async def adapt_to_context(self, 
                           user_input: str, 
                           context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adapt system behavior based on context changes.
        
        Args:
            user_input: User's input text
            context_data: Additional context information
            
        Returns:
            Adaptation results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.dynamic_adaptation:
            return {"error": "Dynamic adaptation system not available"}
        
        # Create adaptable context
        adaptable_context = {
            "user_input": user_input,
            **({} if context_data is None else context_data)
        }
        
        # Add emotional state if available
        if hasattr(self, 'emotional_core'):
            adaptable_context["emotional_state"] = self.emotional_core.get_formatted_emotional_state()
        
        # Detect context change
        change_result = await self.dynamic_adaptation.detect_context_change(adaptable_context)
        
        # Monitor performance
        performance = await self.dynamic_adaptation.monitor_performance({
            "success_rate": context_data.get("success_rate", 0.5) if context_data else 0.5,
            "user_satisfaction": context_data.get("user_satisfaction", 0.5) if context_data else 0.5,
            "efficiency": context_data.get("efficiency", 0.5) if context_data else 0.5,
            "response_quality": context_data.get("response_quality", 0.5) if context_data else 0.5
        })
        
        # Select strategy if significant change
        strategy = None
        if change_result[0]:  # significant change
            strategy = await self.dynamic_adaptation.select_strategy(adaptable_context, performance)
        
        return {
            "context_change": change_result,
            "performance": performance,
            "strategy": strategy
        }
    
    async def evaluate_response(self, 
                             response: str, 
                             context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a response using internal feedback system.
        
        Args:
            response: The response to evaluate
            context_data: Additional context information
            
        Returns:
            Evaluation results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.internal_feedback:
            return {"error": "Internal feedback system not available"}
        
        # Track performance metrics
        metrics = {
            "response_quality": context_data.get("response_quality", 0.5) if context_data else 0.5,
            "user_satisfaction": context_data.get("user_satisfaction", 0.5) if context_data else 0.5
        }
        
        quality_stats = {}
        for metric, value in metrics.items():
            quality_stats[metric] = await self.internal_feedback.track_performance(metric, value)
        
        # Evaluate confidence
        confidence_eval = await self.internal_feedback.evaluate_confidence(
            context_data.get("confidence", 0.7) if context_data else 0.7,
            context_data.get("success", True) if context_data else True
        )
        
        # Create evaluable content
        evaluable_content = {
            "text": response,
            "type": context_data.get("response_type", "general") if context_data else "general",
            "metrics": metrics
        }
        
        # Critic evaluation
        critic_evals = {}
        for aspect in ["consistency", "effectiveness", "efficiency"]:
            critic_evals[aspect] = await self.internal_feedback.critic_evaluate(
                aspect, evaluable_content, context_data or {}
            )
        
        return {
            "quality_stats": quality_stats,
            "confidence_eval": confidence_eval,
            "critic_evals": critic_evals
        }
    
    # Enhanced system function implementations
    
    async def process_user_input_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced processing of user input with comprehensive results.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Comprehensive processing results
        """
        # Use existing process_input but enhance the result
        result = await self.process_input(user_input, context)
        
        # Add additional processing information
        system_stats = await self.get_system_stats()
        
        # Return enhanced result
        return {
            "input": user_input,
            "emotional_state": result["emotional_state"],
            "memories": result["memories"],
            "memory_count": result["memory_count"],
            "has_experience": result["has_experience"],
            "experience_response": result["experience_response"],
            "memory_id": result["memory_id"],
            "response_time": result["response_time"],
            "system_stats": {
                "memory_stats": system_stats["memory_stats"],
                "emotional_state": system_stats["emotional_state"],
                "performance_metrics": system_stats["performance_metrics"]
            }
        }
    
    async def generate_enhanced_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an enhanced response to user input with adaptation.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Enhanced response data with adaptation
        """
        # Generate standard response
        response_data = await self.generate_response(user_input, context)
        
        # Add adaptive behavior from dynamic adaptation system
        if self.dynamic_adaptation:
            # Create adaptable context
            adaptable_context = {
                "user_input": user_input,
                "response": response_data["message"],
                "interaction_type": context.get("interaction_type", "general") if context else "general",
            }
            
            # Detect context change
            change_result = await self.dynamic_adaptation.detect_context_change(adaptable_context)
            
            # Monitor performance
            performance = await self.dynamic_adaptation.monitor_performance({
                "success_rate": context.get("success_rate", 0.5) if context else 0.5,
                "user_satisfaction": context.get("user_satisfaction", 0.5) if context else 0.5,
                "efficiency": context.get("efficiency", 0.5) if context else 0.5,
                "response_quality": context.get("response_quality", 0.5) if context else 0.5
            })
            
            # Add adaptation data to response
            response_data["adaptation"] = {
                "context_change": change_result,
                "performance": performance
            }
            
            # If significant change, select strategy
            if change_result[0]:  # significant change
                strategy = await self.dynamic_adaptation.select_strategy(adaptable_context, performance)
                response_data["adaptation"]["strategy"] = strategy
        
        return response_data
