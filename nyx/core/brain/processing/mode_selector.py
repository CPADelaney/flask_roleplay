# nyx/core/brain/processing/mode_selector.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple
import random
import re

logger = logging.getLogger(__name__)

class ModeSelector:
    """Advanced mode selection logic for brain processing"""
    
    def __init__(self, brain=None):
        """
        Initialize the mode selector
        
        Args:
            brain: Reference to the NyxBrain instance
        """
        self.brain = brain
        
        # Mode selection history
        self.selection_history = []
        
        # Mode performance metrics
        self.mode_metrics = {
            "serial": {"success_rate": 0.95, "avg_time": 0.0, "usage_count": 0},
            "parallel": {"success_rate": 0.92, "avg_time": 0.0, "usage_count": 0},
            "distributed": {"success_rate": 0.90, "avg_time": 0.0, "usage_count": 0},
            "reflexive": {"success_rate": 0.85, "avg_time": 0.0, "usage_count": 0},
            "agent": {"success_rate": 0.88, "avg_time": 0.0, "usage_count": 0},
            "integrated": {"success_rate": 0.93, "avg_time": 0.0, "usage_count": 0}
        }
        
        # Mode complexity thresholds
        self.complexity_thresholds = {
            "parallel": 0.6,   # Switch to parallel at this complexity
            "distributed": 0.8  # Switch to distributed at this complexity
        }
        
        # Task type indicators
        self.task_type_indicators = {
            "reasoning": [
                "why", "explain", "analyze", "reason", "think through", 
                "consider", "evaluate", "compare", "contrast", "deduce"
            ],
            "creative": [
                "imagine", "create", "generate", "design", "write", 
                "story", "fiction", "narrative", "creative", "novel"
            ],
            "factual": [
                "what is", "when did", "where is", "definition", "fact", 
                "information", "history", "data", "tell me about", "describe"
            ],
            "emotional": [
                "feel", "emotion", "sad", "happy", "angry", "worried", 
                "anxious", "excited", "depressed", "stressed", "mood"
            ],
            "procedural": [
                "how to", "steps", "guide", "tutorial", "procedure", 
                "method", "process", "instructions", "walkthrough", "implement"
            ],
            "urgent": [
                "urgent", "immediately", "asap", "now", "emergency", 
                "critical", "quick", "right away", "hurry", "fast"
            ],
            "multi_step": [
                "first", "then", "next", "finally", "steps", 
                "sequence", "order", "stage", "phase", "multi"
            ]
        }
        
        # User preference learning
        self.user_preferences = {}
        
        # Reflection insights about mode selection
        self.selection_insights = {
            "learned_patterns": [],
            "successful_overrides": [],
            "mode_switching_triggers": {}
        }
        
        # Callback registry for mode selection events
        self.selection_callbacks = []
        
        logger.info("Mode selector initialized")
    
    async def determine_processing_mode(self, 
                                     user_input: str, 
                                     context: Dict[str, Any] = None) -> str:
        """
        Determine optimal processing mode based on input complexity and context
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Selected processing mode
        """
        context = context or {}
        
        # Create a selection record for tracking
        selection_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input_length": len(user_input),
            "input_preview": user_input[:50] + "..." if len(user_input) > 50 else user_input,
            "context_keys": list(context.keys()) if context else []
        }
        
        # 0. Check for explicit mode request in context
        if "requested_mode" in context and context["requested_mode"] in self.mode_metrics:
            selected_mode = context["requested_mode"]
            selection_record["mode"] = selected_mode
            selection_record["reason"] = "Explicit mode request in context"
            selection_record["complexity_score"] = None
            
            # Update history and metrics
            self._update_selection_history(selection_record)
            self._increment_usage_count(selected_mode)
            
            return selected_mode
        
        # 1. Check for reflexive pattern match if available
        if hasattr(self.brain, "reflexive_system") and self.brain.reflexive_system:
            should_use_reflex, confidence = await self.brain.reflexive_system.should_use_reflex(
                {"text": user_input}, context, None
            )
            
            if should_use_reflex and confidence > 0.7:
                selection_record["mode"] = "reflexive"
                selection_record["reason"] = f"Reflexive pattern match (confidence: {confidence:.2f})"
                selection_record["confidence"] = confidence
                selection_record["complexity_score"] = None
                
                # Update history and metrics
                self._update_selection_history(selection_record)
                self._increment_usage_count("reflexive")
                
                return "reflexive"
        
        # 2. Check for agent indicators if agent capabilities initialized
        if (hasattr(self.brain, "agent_capabilities_initialized") and 
            self.brain.agent_capabilities_initialized):
            
            agent_score = self._calculate_agent_suitability(user_input, context)
            
            if agent_score > 0.8:
                # High agent suitability, use agent or integrated mode
                if hasattr(self.brain, "processing_manager") and "integrated" in self.brain.processing_manager.processors:
                    selected_mode = "integrated"
                else:
                    selected_mode = "agent"
                
                selection_record["mode"] = selected_mode
                selection_record["reason"] = f"High agent suitability (score: {agent_score:.2f})"
                selection_record["agent_score"] = agent_score
                selection_record["complexity_score"] = None
                
                # Update history and metrics
                self._update_selection_history(selection_record)
                self._increment_usage_count(selected_mode)
                
                return selected_mode
        
        # 3. Calculate complexity score for standard modes
        complexity_score = self._calculate_complexity_score(user_input, context)
        selection_record["complexity_score"] = complexity_score
        
        # 4. Determine task type for better mode matching
        task_types = self._identify_task_types(user_input)
        selection_record["identified_task_types"] = task_types
        
        # 5. Apply task type modifiers to complexity score
        adjusted_score = self._apply_task_type_modifiers(complexity_score, task_types, context)
        selection_record["adjusted_complexity"] = adjusted_score
        
        # 6. Apply user preference learning if available
        if context.get("user_id") in self.user_preferences:
            user_id = context["user_id"]
            preferred_modes = self.user_preferences[user_id].get("preferred_modes", {})
            
            # Check if any identified task types have preferred modes
            for task_type in task_types:
                if task_type in preferred_modes and preferred_modes[task_type]["confidence"] > 0.7:
                    preferred_mode = preferred_modes[task_type]["mode"]
                    
                    selection_record["mode"] = preferred_mode
                    selection_record["reason"] = f"User preference for {task_type} tasks"
                    selection_record["user_preference_applied"] = True
                    
                    # Update history and metrics
                    self._update_selection_history(selection_record)
                    self._increment_usage_count(preferred_mode)
                    
                    return preferred_mode
        
        # 7. Check for temporal context influence
        temporal_influence = self._check_temporal_influence(context)
        
        if temporal_influence:
            adjusted_score = temporal_influence["adjusted_score"]
            selection_record["temporal_influence"] = temporal_influence["reason"]
            selection_record["temporally_adjusted_complexity"] = adjusted_score
        
        # 8. Select mode based on adjusted complexity score
        if adjusted_score < self.complexity_thresholds["parallel"]:
            # Low complexity, use serial processing
            selected_mode = "serial"
            selection_record["reason"] = f"Low complexity score ({adjusted_score:.2f})"
        elif adjusted_score < self.complexity_thresholds["distributed"]:
            # Medium complexity, use parallel processing
            selected_mode = "parallel"
            selection_record["reason"] = f"Medium complexity score ({adjusted_score:.2f})"
        else:
            # High complexity, use distributed processing
            selected_mode = "distributed"
            selection_record["reason"] = f"High complexity score ({adjusted_score:.2f})"
        
        # Special case for task types that benefit from specific modes
        if "reasoning" in task_types and task_types["reasoning"] > 0.7:
            if "distributed" in self.mode_metrics and self.mode_metrics["distributed"]["success_rate"] > 0.85:
                selected_mode = "distributed"
                selection_record["reason"] = "Reasoning task type, using distributed mode"
        
        elif "urgent" in task_types and task_types["urgent"] > 0.7:
            if "parallel" in self.mode_metrics and self.mode_metrics["parallel"]["success_rate"] > 0.85:
                selected_mode = "parallel"
                selection_record["reason"] = "Urgent task type, using parallel mode"
        
        elif "creative" in task_types and task_types["creative"] > 0.7:
            if "agent" in self.mode_metrics and "agent" in self.mode_metrics:
                selected_mode = "agent"
                selection_record["reason"] = "Creative task type, using agent mode"
        
        # Apply performance-based adjustments
        performance_adjusted_mode = self._apply_performance_adjustments(selected_mode, context)
        
        if performance_adjusted_mode != selected_mode:
            selection_record["performance_adjustment"] = f"Changed from {selected_mode} to {performance_adjusted_mode}"
            selected_mode = performance_adjusted_mode
        
        selection_record["mode"] = selected_mode
        
        # Update history and metrics
        self._update_selection_history(selection_record)
        self._increment_usage_count(selected_mode)
        
        # Call any registered callbacks
        for callback in self.selection_callbacks:
            try:
                asyncio.create_task(callback(selection_record))
            except Exception as e:
                logger.error(f"Error in mode selection callback: {str(e)}")
        
        return selected_mode
    
    def _calculate_complexity_score(self, user_input: str, context: Dict[str, Any]) -> float:
        """Calculate complexity score based on input and context"""
        # 1. Input length
        input_length_factor = min(1.0, len(user_input) / 500.0)  # Normalize to [0,1]
        
        # 2. Content complexity
        words = user_input.lower().split()
        unique_words = len(set(words))
        word_complexity = min(1.0, unique_words / 50.0)  # Normalize to [0,1]
        
        punctuation_count = sum(1 for c in user_input if c in "?!.,;:()[]{}\"'")
        punctuation_complexity = min(1.0, punctuation_count / 20.0)  # Normalize to [0,1]
        
        content_complexity = (word_complexity * 0.7 + punctuation_complexity * 0.3)
        
        # 3. Context complexity
        context_complexity = 0.0
        if context:
            context_size = min(1.0, len(str(context)) / 1000.0)
            context_keys = min(1.0, len(context) / 10.0)
            context_complexity = (context_size * 0.7 + context_keys * 0.3)
            
            # Additional factor for special context keys that imply complexity
            complex_keys = ["memory_limit", "memory_types", "reasoning_depth", "thinking_mode"]
            if any(key in context for key in complex_keys):
                context_complexity += 0.1
        
        # 4. Intent complexity (based on question words and sentence structure)
        intent_complexity = 0.0
        
        # Check for multiple questions
        question_count = user_input.count("?")
        if question_count > 1:
            intent_complexity += min(1.0, question_count / 5.0) * 0.3
        
        # Check for complex question prefixes
        complex_prefixes = ["explain how", "why would", "what are the implications", 
                         "in what way", "to what extent", "how would", "analyze"]
        
        if any(user_input.lower().startswith(prefix) for prefix in complex_prefixes):
            intent_complexity += 0.3
        
        # Check for comparisons
        comparison_terms = ["compare", "contrast", "versus", "vs", "difference", "similarities", 
                         "advantages", "disadvantages", "pros and cons"]
        
        if any(term in user_input.lower() for term in comparison_terms):
            intent_complexity += 0.2
        
        # 5. History/state complexity
        history_complexity = 0.0
        if hasattr(self.brain, "interaction_count"):
            history_complexity = min(1.0, self.brain.interaction_count / 50.0)
        
        # 6. Required memory complexity
        memory_complexity = 0.0
        if "memory_types" in context or "memory_limit" in context:
            memory_complexity = 0.2
            if "memory_limit" in context and context["memory_limit"] > 10:
                memory_complexity += 0.1
        
        # Calculate weighted final score
        complexity_score = (
            input_length_factor * 0.2 +
            content_complexity * 0.25 +
            context_complexity * 0.15 +
            intent_complexity * 0.25 +
            history_complexity * 0.1 +
            memory_complexity * 0.05
        )
        
        return complexity_score
    
    def _identify_task_types(self, user_input: str) -> Dict[str, float]:
        """Identify task types from user input"""
        input_lower = user_input.lower()
        
        # Calculate match score for each task type
        task_types = {}
        for task_type, indicators in self.task_type_indicators.items():
            indicator_matches = [ind for ind in indicators if ind in input_lower]
            match_score = min(1.0, len(indicator_matches) / (len(indicators) * 0.3))
            
            if match_score > 0.3:  # Only include significant matches
                task_types[task_type] = match_score
        
        return task_types
    
    def _apply_task_type_modifiers(self, 
                                complexity_score: float, 
                                task_types: Dict[str, float],
                                context: Dict[str, Any]) -> float:
        """Apply task type-specific modifiers to complexity score"""
        modified_score = complexity_score
        
        # Reasoning tasks tend to be more complex
        if "reasoning" in task_types:
            reasoning_modifier = 0.1 * task_types["reasoning"]
            modified_score += reasoning_modifier
        
        # Creative tasks benefit from certain modes
        if "creative" in task_types:
            creative_modifier = 0.05 * task_types["creative"]
            modified_score += creative_modifier
        
        # Urgent tasks might need faster processing
        if "urgent" in task_types:
            urgent_modifier = -0.1 * task_types["urgent"]  # Reduce complexity to favor faster modes
            modified_score += urgent_modifier
        
        # Multi-step tasks are more complex
        if "multi_step" in task_types:
            multi_step_modifier = 0.15 * task_types["multi_step"]
            modified_score += multi_step_modifier
        
        # Procedural tasks might benefit from specific modes
        if "procedural" in task_types and hasattr(self.brain, "procedural_memory"):
            # Check if we have procedures for this
            procedural_modifier = 0.0
            if "procedural_check" in context and context["procedural_check"]:
                procedural_modifier = -0.1  # Reduce complexity if procedures are available
            modified_score += procedural_modifier
        
        # Emotional tasks might need specific handling
        if "emotional" in task_types and hasattr(self.brain, "emotional_core"):
            emotional_modifier = 0.05 * task_types["emotional"]
            modified_score += emotional_modifier
        
        # Ensure score remains in valid range
        return max(0.0, min(1.0, modified_score))
    
    def _calculate_agent_suitability(self, user_input: str, context: Dict[str, Any]) -> float:
        """Calculate suitability score for agent-based processing"""
        agent_indicators = [
            "roleplay", "role play", "acting", "pretend", "scenario",
            "imagine", "fantasy", "act as", "play as", "in-character",
            "story", "scene", "setting", "character", "plot",
            "describe", "tell me about", "what happens",
            "picture", "image", "draw", "show me", "visualize"
        ]
        
        input_lower = user_input.lower()
        
        # Check for direct agent indicators
        indicator_matches = [ind for ind in agent_indicators if ind in input_lower]
        direct_indicator_score = min(1.0, len(indicator_matches) / 3.0)
        
        # Check for context indicators
        context_score = 0.0
        if context:
            agent_context_keys = ["role", "character", "scenario", "scene", "narrative"]
            matching_keys = [key for key in agent_context_keys if key in context]
            context_score = min(1.0, len(matching_keys) / len(agent_context_keys))
        
        # Check for creativity requirements
        creativity_score = 0.0
        creativity_indicators = ["creative", "novel", "original", "unique", "imagination"]
        if any(ind in input_lower for ind in creativity_indicators):
            creativity_score = 0.3
        
        # Check for storytelling requirements
        storytelling_score = 0.0
        if "story" in input_lower or "narrative" in input_lower or "plot" in input_lower:
            storytelling_score = 0.4
        
        # Combine scores with appropriate weights
        agent_suitability = (
            direct_indicator_score * 0.5 +
            context_score * 0.2 +
            creativity_score * 0.15 +
            storytelling_score * 0.15
        )
        
        return agent_suitability
    
    def _apply_performance_adjustments(self, 
                                    selected_mode: str, 
                                    context: Dict[str, Any]) -> str:
        """Apply performance-based adjustments to mode selection"""
        # If success rate for selected mode is lower than another mode,
        # consider switching based on relative performance
        selected_success_rate = self.mode_metrics[selected_mode]["success_rate"]
        
        # Find modes with better success rates
        better_modes = []
        for mode, metrics in self.mode_metrics.items():
            if metrics["success_rate"] > selected_success_rate + 0.05:  # At least 5% better
                better_modes.append((mode, metrics["success_rate"]))
        
        # If better modes are available, consider switching
        if better_modes:
            # Sort by success rate, highest first
            better_modes.sort(key=lambda x: x[1], reverse=True)
            best_mode, best_rate = better_modes[0]
            
            # Make a probabilistic decision based on the performance difference
            diff = best_rate - selected_success_rate
            switch_probability = diff * 1.5  # More likely to switch with bigger differences
            
            if random.random() < switch_probability:
                # Add to successful overrides if this turns out well
                self.selection_insights["successful_overrides"].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "from_mode": selected_mode,
                    "to_mode": best_mode,
                    "success_rate_diff": diff
                })
                
                return best_mode
        
        # No adjustment needed
        return selected_mode
    
    def _check_temporal_influence(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for temporal context influence on mode selection"""
        if not hasattr(self.brain, "temporal_perception"):
            return None
        
        # Check if temporal context is available
        if not context or "temporal_context" not in context:
            return None
        
        temporal_context = context["temporal_context"]
        
        # Initialize result
        result = {"adjusted_score": None, "reason": None}
        
        # Apply temporal effects
        # 1. Time of day effect
        if "time_of_day" in temporal_context:
            time_of_day = temporal_context["time_of_day"]
            
            # Late night - favor simpler processing
            if time_of_day > 0.9 or time_of_day < 0.1:  # Very late or very early
                time_factor = -0.1  # Reduce complexity
                result["reason"] = "Late night temporal context"
            # Morning - balanced
            elif 0.25 <= time_of_day <= 0.5:
                time_factor = 0.0  # No change
                result["reason"] = "Morning temporal context"
            # Afternoon/evening - favor richer processing
            else:
                time_factor = 0.05  # Slightly increase complexity
                result["reason"] = "Afternoon/evening temporal context"
            
            if "complexity_score" in context:
                result["adjusted_score"] = max(0.0, min(1.0, context["complexity_score"] + time_factor))
        
        # 2. Session duration effect
        if "session_duration" in temporal_context:
            duration = temporal_context["session_duration"]
            
            # Long sessions might benefit from more complex processing
            if duration > 0.6:  # Over 60% of reference duration
                duration_factor = 0.1
                result["reason"] = "Long session duration"
                
                if "adjusted_score" in result and result["adjusted_score"] is not None:
                    result["adjusted_score"] = max(0.0, min(1.0, result["adjusted_score"] + duration_factor))
        
        # Return None if no adjustment was made
        if result["adjusted_score"] is None:
            return None
            
        return result
    
    def _update_selection_history(self, selection_record: Dict[str, Any]) -> None:
        """Update selection history and trim if needed"""
        self.selection_history.append(selection_record)
        
        # Keep history to a reasonable size
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
    
    def _increment_usage_count(self, mode: str) -> None:
        """Increment usage count for a mode"""
        if mode in self.mode_metrics:
            self.mode_metrics[mode]["usage_count"] += 1
    
    def update_mode_metrics(self, 
                          mode: str, 
                          success: bool, 
                          response_time: float) -> None:
        """
        Update performance metrics for a mode
        
        Args:
            mode: Processing mode
            success: Whether processing was successful
            response_time: Time taken to process
        """
        if mode not in self.mode_metrics:
            return
        
        metrics = self.mode_metrics[mode]
        
        # Update success rate with decay
        current_success_rate = metrics["success_rate"]
        decay_factor = 0.95  # How much to retain previous value
        
        if metrics["usage_count"] > 0:
            new_success_rate = (current_success_rate * decay_factor * metrics["usage_count"] + 
                              (1.0 if success else 0.0)) / (metrics["usage_count"] * decay_factor + 1)
        else:
            new_success_rate = 1.0 if success else 0.0
        
        metrics["success_rate"] = new_success_rate
        
        # Update average time
        current_avg_time = metrics["avg_time"]
        if metrics["usage_count"] > 0:
            new_avg_time = (current_avg_time * metrics["usage_count"] + response_time) / (metrics["usage_count"] + 1)
        else:
            new_avg_time = response_time
            
        metrics["avg_time"] = new_avg_time
    
    def update_complexity_thresholds(self) -> None:
        """Dynamically update complexity thresholds based on performance metrics"""
        if not self.selection_history:
            return
        
        # Get recent selections
        recent_selections = self.selection_history[-30:] if len(self.selection_history) >= 30 else self.selection_history
        
        # Extract complexity scores and modes
        complexity_modes = [(s["complexity_score"], s["mode"]) for s in recent_selections 
                          if "complexity_score" in s and s["complexity_score"] is not None]
        
        if not complexity_modes:
            return
            
        # Group by mode
        mode_complexities = {}
        for score, mode in complexity_modes:
            if mode not in mode_complexities:
                mode_complexities[mode] = []
            mode_complexities[mode].append(score)
        
        # Calculate average complexity for each mode
        avg_complexities = {}
        for mode, scores in mode_complexities.items():
            if scores:
                avg_complexities[mode] = sum(scores) / len(scores)
        
        # Check if we have data for both serial and parallel
        if "serial" in avg_complexities and "parallel" in avg_complexities:
            # Find the midpoint between serial and parallel average complexities
            serial_avg = avg_complexities["serial"]
            parallel_avg = avg_complexities["parallel"]
            
            if serial_avg < parallel_avg:
                # Update parallel threshold to be between serial and parallel averages
                new_parallel_threshold = (serial_avg + parallel_avg) / 2
                self.complexity_thresholds["parallel"] = new_parallel_threshold
        
        # Check if we have data for both parallel and distributed
        if "parallel" in avg_complexities and "distributed" in avg_complexities:
            # Find the midpoint between parallel and distributed average complexities
            parallel_avg = avg_complexities["parallel"]
            distributed_avg = avg_complexities["distributed"]
            
            if parallel_avg < distributed_avg:
                # Update distributed threshold to be between parallel and distributed averages
                new_distributed_threshold = (parallel_avg + distributed_avg) / 2
                self.complexity_thresholds["distributed"] = new_distributed_threshold
    
    def learn_from_user_feedback(self, 
                                user_id: str, 
                                feedback: Dict[str, Any],
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update user preferences based on feedback
        
        Args:
            user_id: ID of the user
            feedback: Feedback data
            context: Original processing context
            
        Returns:
            Learning results
        """
        # Initialize user preferences if not exists
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_modes": {},
                "task_type_preferences": {},
                "feedback_history": []
            }
        
        user_prefs = self.user_preferences[user_id]
        
        # Add to feedback history
        feedback_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode_used": feedback.get("mode_used"),
            "satisfaction": feedback.get("satisfaction", 0),
            "preferred_mode": feedback.get("preferred_mode"),
            "task_types": feedback.get("task_types", [])
        }
        user_prefs["feedback_history"].append(feedback_record)
        
        # If user specified a preferred mode for a task type
        if (feedback.get("preferred_mode") and feedback.get("task_types") and 
            feedback.get("preferred_mode") in self.mode_metrics):
            
            preferred_mode = feedback["preferred_mode"]
            
            for task_type in feedback["task_types"]:
                # Check if we already have a preference for this task type
                if task_type in user_prefs["preferred_modes"]:
                    # Update existing preference
                    current_pref = user_prefs["preferred_modes"][task_type]
                    
                    # If same mode, increase confidence
                    if current_pref["mode"] == preferred_mode:
                        current_pref["confidence"] = min(1.0, current_pref["confidence"] + 0.1)
                        current_pref["count"] += 1
                    else:
                        # Different mode, adjust preference
                        if current_pref["confidence"] > 0.2:
                            # Reduce confidence in current preference
                            current_pref["confidence"] -= 0.2
                        else:
                            # Switch to new preference
                            user_prefs["preferred_modes"][task_type] = {
                                "mode": preferred_mode,
                                "confidence": 0.6,
                                "count": 1
                            }
                else:
                    # Add new preference
                    user_prefs["preferred_modes"][task_type] = {
                        "mode": preferred_mode,
                        "confidence": 0.6,  # Initial confidence
                        "count": 1
                    }
        
        # Learn from satisfaction rating
        if "satisfaction" in feedback and feedback.get("mode_used") in self.mode_metrics:
            satisfaction = feedback["satisfaction"]  # Assuming 0-1 scale
            mode_used = feedback["mode_used"]
            
            # If high satisfaction, slightly boost success rate
            if satisfaction > 0.8:
                current_rate = self.mode_metrics[mode_used]["success_rate"]
                self.mode_metrics[mode_used]["success_rate"] = min(0.99, current_rate + 0.01)
            
            # If low satisfaction, slightly reduce success rate
            elif satisfaction < 0.3:
                current_rate = self.mode_metrics[mode_used]["success_rate"]
                self.mode_metrics[mode_used]["success_rate"] = max(0.5, current_rate - 0.02)
        
        return {
            "user_id": user_id,
            "preferences_updated": True,
            "current_preferences": user_prefs["preferred_modes"]
        }
    
    def register_selection_callback(self, callback: callable) -> None:
        """
        Register a callback for mode selection events
        
        Args:
            callback: Async function to call when a mode is selected
        """
        self.selection_callbacks.append(callback)
    
    async def analyze_mode_usage(self) -> Dict[str, Any]:
        """
        Analyze mode usage patterns
        
        Returns:
            Analysis results
        """
        # Check if we have enough history
        if len(self.selection_history) < 5:
            return {
                "status": "insufficient_data",
                "message": "Not enough mode selection history for analysis"
            }
        
        # Calculate usage distribution
        usage_counts = {}
        for mode in self.mode_metrics:
            usage_counts[mode] = self.mode_metrics[mode]["usage_count"]
        
        total_usage = sum(usage_counts.values())
        usage_distribution = {mode: count/total_usage if total_usage > 0 else 0 
                            for mode, count in usage_counts.items()}
        
        # Analyze recent selections
        recent_selections = self.selection_history[-20:]
        recent_modes = [s["mode"] for s in recent_selections]
        
        # Calculate transition probabilities
        transitions = {}
        for i in range(len(recent_modes) - 1):
            from_mode = recent_modes[i]
            to_mode = recent_modes[i + 1]
            
            if from_mode not in transitions:
                transitions[from_mode] = {}
            
            if to_mode not in transitions[from_mode]:
                transitions[from_mode][to_mode] = 0
                
            transitions[from_mode][to_mode] += 1
        
        # Convert counts to probabilities
        transition_probabilities = {}
        for from_mode, to_modes in transitions.items():
            total = sum(to_modes.values())
            transition_probabilities[from_mode] = {to: count/total for to, count in to_modes.items()}
        
        # Analyze task type patterns
        task_type_modes = {}
        for selection in recent_selections:
            if "identified_task_types" in selection:
                for task_type, score in selection["identified_task_types"].items():
                    if task_type not in task_type_modes:
                        task_type_modes[task_type] = {}
                    
                    mode = selection["mode"]
                    if mode not in task_type_modes[task_type]:
                        task_type_modes[task_type][mode] = 0
                        
                    task_type_modes[task_type][mode] += 1
        
        # Get top mode for each task type
        task_type_preferred_modes = {}
        for task_type, modes in task_type_modes.items():
            top_mode = max(modes.items(), key=lambda x: x[1])
            task_type_preferred_modes[task_type] = {
                "mode": top_mode[0],
                "count": top_mode[1],
                "percentage": top_mode[1] / sum(modes.values())
            }
        
        # Return analysis results
        return {
            "status": "success",
            "total_selections": len(self.selection_history),
            "usage_distribution": usage_distribution,
            "transition_probabilities": transition_probabilities,
            "task_type_patterns": task_type_preferred_modes,
            "complexity_thresholds": self.complexity_thresholds,
            "performance_metrics": {
                mode: {
                    "success_rate": metrics["success_rate"],
                    "avg_time": metrics["avg_time"]
                } for mode, metrics in self.mode_metrics.items()
            }
        }
