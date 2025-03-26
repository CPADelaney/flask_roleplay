# nyx/core/brain/adaptation/self_config.py
import logging
import asyncio
import datetime
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SelfConfigManager:
    """Manages self-configuration capabilities of the brain"""
    
    def __init__(self, brain):
        self.brain = brain
        self.self_config_enabled = False
        self.config_change_history = []
        self.param_performance_impact = {}
        self.config_update_interval = 50  # Interactions between updates
        self.last_config_update = 0
        self.confidence_thresholds = {
            "low": 0.4,
            "medium": 0.7,
            "high": 0.9
        }
        
        # Define adaptation strategies
        self.adaptation_strategies = {
            "conservative": {
                "description": "Make small, cautious adjustments with high confidence requirements",
                "confidence_multiplier": 1.2,
                "step_multiplier": 0.7,
                "evaluation_frequency": 1.5  # Longer evaluation periods
            },
            "balanced": {
                "description": "Make moderate adjustments with balanced confidence requirements",
                "confidence_multiplier": 1.0,
                "step_multiplier": 1.0,
                "evaluation_frequency": 1.0
            },
            "exploratory": {
                "description": "Make larger adjustments with lower confidence requirements",
                "confidence_multiplier": 0.8,
                "step_multiplier": 1.3,
                "evaluation_frequency": 0.7  # Shorter evaluation periods
            }
        }
        
        # Current strategy - can be adjusted based on context
        self.current_adaptation_strategy = "balanced"
        
        # Define parameter categories for organization
        self.parameter_categories = {
            "core": "Core operational parameters",
            "memory": "Memory system parameters",
            "emotion": "Emotional system parameters",
            "reasoning": "Reasoning system parameters",
            "social": "Cross-user and social parameters",
            "attention": "Attentional system parameters",
            "temporal": "Temporal processing parameters",
            "hormonal": "Hormone system parameters",
            "reflection": "Meta-cognitive reflection parameters",
            "procedural": "Procedural memory parameters",
            "reflexive": "Reflexive system parameters",
            "performance": "System performance parameters"
        }
        
        # Parameter evaluation priority (some parameters are more important to evaluate)
        self.parameter_evaluation_priority = {}
        
        # User feedback impact tracking
        self.user_feedback_impact = {
            "positive_feedback": {},
            "negative_feedback": {},
            "parameter_adjustments": {}
        }
        
        logger.info("Self-configuration manager initialized")
    
    async def enable(self) -> Dict[str, Any]:
        """Enable self-configuration capabilities"""
        if self.self_config_enabled:
            return {"status": "already_enabled"}
        
        # Initialize parameter interdependence graph
        self.parameter_dependencies = {}
        
        # Define adjustable parameters with safe ranges, defaults, and categories
        self.adjustable_parameters = {
            # Core parameters
            "cross_user_sharing_threshold": {
                "current": getattr(self.brain, "cross_user_sharing_threshold", 0.7),
                "min": 0.3,
                "max": 0.95,
                "default": 0.7,
                "step": 0.05,
                "description": "Threshold for sharing experiences across users",
                "category": "social",
                "related_to": ["cross_user_enabled"]
            },
            "memory_to_emotion_influence": {
                "current": getattr(self.brain, "memory_to_emotion_influence", 0.3),
                "min": 0.1,
                "max": 0.8,
                "default": 0.3,
                "step": 0.05,
                "description": "How much memories influence emotions",
                "category": "emotion",
                "related_to": ["emotion_to_memory_influence"]
            },
            "emotion_to_memory_influence": {
                "current": getattr(self.brain, "emotion_to_memory_influence", 0.4),
                "min": 0.1,
                "max": 0.8,
                "default": 0.4,
                "step": 0.05,
                "description": "How much emotions influence memory retrieval",
                "category": "memory",
                "related_to": ["memory_to_emotion_influence"]
            },
            "experience_to_identity_influence": {
                "current": getattr(self.brain, "experience_to_identity_influence", 0.2),
                "min": 0.05,
                "max": 0.6,
                "default": 0.2,
                "step": 0.05,
                "description": "How much experiences influence identity",
                "category": "core",
                "related_to": []
            },
            "cross_user_enabled": {
                "current": 1 if getattr(self.brain, "cross_user_enabled", True) else 0,
                "min": 0,  # Boolean as 0/1
                "max": 1,
                "default": 1,
                "step": 1,
                "description": "Whether cross-user experiences are enabled",
                "category": "social",
                "related_to": ["cross_user_sharing_threshold"]
            },
            "consolidation_interval": {
                "current": getattr(self.brain, "consolidation_interval", 24),
                "min": 6,
                "max": 72,
                "default": 24,
                "step": 6,
                "description": "Hours between experience consolidations",
                "category": "memory",
                "related_to": []
            },
            "identity_reflection_interval": {
                "current": getattr(self.brain, "identity_reflection_interval", 10),
                "min": 5,
                "max": 50,
                "default": 10,
                "step": 5,
                "description": "Interactions between identity reflections",
                "category": "reflection",
                "related_to": []
            },
            
            # Memory system parameters
            "memory_recency_weight": {
                "current": 0.7 if hasattr(self.brain, "memory_orchestrator") and hasattr(self.brain.memory_orchestrator, "recency_weight") else 0.7,
                "min": 0.3,
                "max": 0.9,
                "default": 0.7,
                "step": 0.05,
                "description": "Weight given to recency in memory retrieval",
                "category": "memory",
                "related_to": ["memory_relevance_weight"]
            },
            "memory_relevance_weight": {
                "current": 0.8 if hasattr(self.brain, "memory_orchestrator") and hasattr(self.brain.memory_orchestrator, "relevance_weight") else 0.8,
                "min": 0.4,
                "max": 0.95,
                "default": 0.8,
                "step": 0.05,
                "description": "Weight given to relevance in memory retrieval",
                "category": "memory",
                "related_to": ["memory_recency_weight"]
            },
            "memory_significance_threshold": {
                "current": 3 if hasattr(self.brain, "memory_core") and hasattr(self.brain.memory_core, "significance_threshold") else 3,
                "min": 1,
                "max": 8,
                "default": 3,
                "step": 1,
                "description": "Minimum significance for memories to be retrieved",
                "category": "memory",
                "related_to": []
            },
            
            # Emotional system parameters
            "emotional_decay_rate": {
                "current": 0.05 if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "decay_rate") else 0.05,
                "min": 0.01,
                "max": 0.3,
                "default": 0.05,
                "step": 0.01,
                "description": "Rate at which emotions decay over time",
                "category": "emotion",
                "related_to": []
            },
            "emotional_expression_threshold": {
                "current": 0.7 if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "expression_threshold") else 0.7,
                "min": 0.4,
                "max": 0.9,
                "default": 0.7,
                "step": 0.05,
                "description": "Threshold for expressing emotions",
                "category": "emotion",
                "related_to": []
            },
            "emotional_complexity": {
                "current": 0.6 if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "complexity") else 0.6,
                "min": 0.1,
                "max": 1.0,
                "default": 0.6,
                "step": 0.1,
                "description": "Complexity of emotional expressions",
                "category": "emotion",
                "related_to": []
            },
            
            # System performance parameters
            "parallel_processing_threshold": {
                "current": 0.6,  # Complexity threshold for using parallel processing
                "min": 0.3,
                "max": 0.9,
                "default": 0.6,
                "step": 0.1,
                "description": "Complexity threshold for switching to parallel processing",
                "category": "performance",
                "related_to": []
            },
            "distributed_processing_threshold": {
                "current": 0.8,  # Complexity threshold for using distributed processing
                "min": 0.5,
                "max": 0.95,
                "default": 0.8,
                "step": 0.05,
                "description": "Complexity threshold for switching to distributed processing",
                "category": "performance",
                "related_to": ["parallel_processing_threshold"]
            }
        }
        
        # Add attention system parameters if available
        if hasattr(self.brain, "attentional_controller"):
            self.adjustable_parameters.update({
                "attentional_focus_duration": {
                    "current": getattr(self.brain.attentional_controller, "focus_duration", 5),
                    "min": 1,
                    "max": 20,
                    "default": 5,
                    "step": 1,
                    "description": "Duration of attentional focus (in interactions)",
                    "category": "attention",
                    "related_to": []
                },
                "novelty_weight": {
                    "current": getattr(self.brain.attentional_controller, "novelty_weight", 0.7),
                    "min": 0.3,
                    "max": 0.9,
                    "default": 0.7,
                    "step": 0.05,
                    "description": "Weight given to novelty in attention",
                    "category": "attention",
                    "related_to": []
                }
            })
        
        # Add hormone system parameters if available
        if hasattr(self.brain, "hormone_system"):
            self.adjustable_parameters.update({
                "hormone_influence_factor": {
                    "current": getattr(self.brain.hormone_system, "influence_factor", 0.6),
                    "min": 0.2,
                    "max": 1.0,
                    "default": 0.6,
                    "step": 0.1,
                    "description": "Overall influence of hormones on other systems",
                    "category": "hormonal",
                    "related_to": []
                },
                "hormonal_cycle_speed": {
                    "current": getattr(self.brain.hormone_system, "cycle_speed", 1.0),
                    "min": 0.5,
                    "max": 2.0,
                    "default": 1.0,
                    "step": 0.1,
                    "description": "Speed of hormonal cycles (multiplier)",
                    "category": "hormonal",
                    "related_to": []
                }
            })
        
        # Add reasoning system parameters if available
        if hasattr(self.brain, "reasoning_core"):
            self.adjustable_parameters.update({
                "reasoning_depth": {
                    "current": getattr(self.brain.reasoning_core, "reasoning_depth", 2),
                    "min": 1,
                    "max": 4,
                    "default": 2,
                    "step": 1,
                    "description": "Depth of reasoning processes",
                    "category": "reasoning",
                    "related_to": []
                }
            })
        
        # Add thinking system parameters if available
        if hasattr(self.brain, "thinking_config"):
            self.adjustable_parameters.update({
                "thinking_frequency": {
                    "current": 0.4 if self.brain.thinking_config.get("thinking_enabled", False) else 0.4,
                    "min": 0.1,
                    "max": 0.8,
                    "default": 0.4,
                    "step": 0.1,
                    "description": "Frequency of using extended thinking",
                    "category": "reasoning",
                    "related_to": []
                }
            })
        
        # Add reflexive system parameters if available
        if hasattr(self.brain, "reflexive_system"):
            self.adjustable_parameters.update({
                "reflex_threshold": {
                    "current": getattr(self.brain.reflexive_system, "default_threshold", 0.6),
                    "min": 0.4,
                    "max": 0.9,
                    "default": 0.6,
                    "step": 0.05,
                    "description": "Threshold for triggering reflexes",
                    "category": "reflexive",
                    "related_to": []
                }
            })
        
        # Add procedural memory parameters if available
        if hasattr(self.brain, "agent_enhanced_memory"):
            self.adjustable_parameters.update({
                "procedural_confidence_threshold": {
                    "current": getattr(self.brain.agent_enhanced_memory, "confidence_threshold", 0.7),
                    "min": 0.5,
                    "max": 0.95,
                    "default": 0.7,
                    "step": 0.05,
                    "description": "Confidence threshold for procedural memory execution",
                    "category": "procedural",
                    "related_to": []
                }
            })
        
        # Build parameter dependency graph
        for param_name, param_config in self.adjustable_parameters.items():
            self.parameter_dependencies[param_name] = {
                "affects": [],
                "affected_by": param_config["related_to"]
            }
            
        # Complete bidirectional dependencies
        for param_name, dependencies in self.parameter_dependencies.items():
            for related_param in dependencies["affected_by"]:
                if related_param in self.parameter_dependencies:
                    if param_name not in self.parameter_dependencies[related_param]["affects"]:
                        self.parameter_dependencies[related_param]["affects"].append(param_name)
        
        # Performance metrics to track for adaptation
        self.parameter_metrics_map = {
            # Core parameters
            "cross_user_sharing_threshold": ["experiences_shared", "cross_user_experiences_shared"],
            "memory_to_emotion_influence": ["emotion_updates"],
            "emotion_to_memory_influence": ["memory_operations"],
            "experience_to_identity_influence": ["experiences_shared"],
            "cross_user_enabled": ["cross_user_experiences_shared"],
            "consolidation_interval": ["experience_consolidations"],
            "identity_reflection_interval": ["reflections_generated"],
            
            # Memory system parameters
            "memory_recency_weight": ["memory_operations", "avg_response_time"],
            "memory_relevance_weight": ["memory_operations", "experiences_shared"],
            "memory_significance_threshold": ["memory_operations", "memory_count"],
            
            # Emotional system parameters
            "emotional_decay_rate": ["emotion_updates"],
            "emotional_expression_threshold": ["emotion_updates"],
            "emotional_complexity": ["emotion_updates"],
            
            # Attention system parameters
            "attentional_focus_duration": ["memory_operations", "experiences_shared"],
            "novelty_weight": ["memory_operations", "experiences_shared"],
            
            # Hormone system parameters
            "hormone_influence_factor": ["emotion_updates", "experiences_shared"],
            "hormonal_cycle_speed": ["emotion_updates"],
            
            # Reasoning system parameters
            "reasoning_depth": ["avg_response_time"],
            "thinking_frequency": ["avg_response_time"],
            
            # Reflexive system parameters
            "reflex_threshold": ["avg_response_time"],
            "procedural_confidence_threshold": ["avg_response_time"],
            
            # System performance parameters
            "parallel_processing_threshold": ["avg_response_time"],
            "distributed_processing_threshold": ["avg_response_time"]
        }
        
        # Meta-learning: track which parameters have the most impact
        self.parameter_impact_ranking = {}
        
        self.self_config_enabled = True
        
        # Schedule periodic parameter evaluation
        asyncio.create_task(self._parameter_evaluation_loop())
        
        logger.info("Self-configuration system enabled")
        
        return {
            "enabled": self.self_config_enabled,
            "adjustable_parameters_count": len(self.adjustable_parameters),
            "categories": {k: v for k, v in self.parameter_categories.items()},
            "available_strategies": {k: v["description"] for k, v in self.adaptation_strategies.items()},
            "current_strategy": self.current_adaptation_strategy,
            "update_interval": self.config_update_interval,
            "last_update": self.last_config_update
        }
    
    async def _parameter_evaluation_loop(self):
        """Background task to periodically evaluate and adjust parameters"""
        while self.self_config_enabled:
            try:
                # Only evaluate after certain number of interactions
                if hasattr(self.brain, "interaction_count") and self.brain.interaction_count - self.last_config_update >= self.config_update_interval:
                    await self.evaluate_and_adjust_parameters()
                    self.last_config_update = self.brain.interaction_count
                    
                    # Also update meta-learning
                    await self._update_parameter_meta_learning()
            except Exception as e:
                logger.error(f"Error in parameter evaluation loop: {str(e)}")
                
            # Wait before checking again
            await asyncio.sleep(60)  # Check every minute
    
    async def _update_parameter_meta_learning(self):
        """Update meta-learning about which parameters are most impactful"""
        try:
            # Calculate impact scores for each parameter
            impact_scores = {}
            
            for param_name, param_data in self.param_performance_impact.items():
                if param_data["history"]:
                    # Calculate normalized impact
                    impacts = [abs(entry["impact"]) for entry in param_data["history"]]
                    avg_impact = sum(impacts) / len(impacts)
                    
                    # Consider number of samples for confidence
                    sample_confidence = min(1.0, len(impacts) / 10)
                    
                    # Calculate final score
                    impact_scores[param_name] = avg_impact * sample_confidence
            
            # Rank parameters by impact score
            self.parameter_impact_ranking = {
                k: {"score": v, "rank": i+1} 
                for i, (k, v) in enumerate(sorted(impact_scores.items(), key=lambda x: x[1], reverse=True))
            }
            
            # Adjust evaluation priorities
            high_impact_params = [k for k, v in self.parameter_impact_ranking.items() 
                                if v["rank"] <= 5]  # Top 5 parameters
            
            for param_name in high_impact_params:
                # Increase evaluation frequency for high-impact parameters
                self.parameter_evaluation_priority[param_name] = 1.5  # 50% more likely to be evaluated
                
            logger.info(f"Updated parameter meta-learning - top impacts: {high_impact_params}")
        except Exception as e:
            logger.error(f"Error updating parameter meta-learning: {str(e)}")
    
    async def evaluate_and_adjust_parameters(self) -> Dict[str, Any]:
        """
        Evaluate current performance metrics and adjust parameters if needed
        
        Returns:
            Results of parameter adjustments
        """
        if not self.self_config_enabled:
            return {"status": "disabled"}
        
        results = {
            "evaluated": [],
            "adjusted": [],
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy": self.current_adaptation_strategy
        }
        
        try:
            # Get current strategy configuration
            strategy = self.adaptation_strategies[self.current_adaptation_strategy]
            confidence_multiplier = strategy["confidence_multiplier"]
            step_multiplier = strategy["step_multiplier"]
            evaluation_frequency = strategy["evaluation_frequency"]
            
            # Get current performance metrics
            stats = await self.brain.get_system_stats()
            performance = stats["performance_metrics"]
            
            # Check current emotional state to influence strategy
            emotional_state = stats["emotional_state"]
            dominant_emotion = emotional_state.get("dominant_emotion")
            
            # Adjust strategy based on emotion if needed
            adjusted_step_multiplier = step_multiplier
            if dominant_emotion in ["Joy", "Trust", "Anticipation"]:
                # More positive emotions -> slightly more exploratory
                adjusted_step_multiplier *= 1.1
            elif dominant_emotion in ["Fear", "Anger", "Disgust"]:
                # More negative emotions -> slightly more conservative
                adjusted_step_multiplier *= 0.9
            
            # Check if brain is in an abnormal state to prioritize stability
            abnormal_state = False
            if performance.get("avg_response_time", 0) > 1.5:  # High response time
                abnormal_state = True
            
            # Get highest priority parameters to evaluate
            eval_params = []
            
            # During abnormal state, prioritize performance parameters
            if abnormal_state:
                eval_params.extend([p for p, c in self.adjustable_parameters.items() 
                                  if c["category"] == "performance"])
            
            # Add meta-learned high-impact parameters
            if self.parameter_impact_ranking:
                high_impact = [k for k, v in self.parameter_impact_ranking.items() 
                              if v["rank"] <= 3]  # Top 3
                eval_params.extend(high_impact)
            
            # Add parameters with high evaluation priority
            priority_params = [p for p, priority in self.parameter_evaluation_priority.items() 
                             if priority > 1.0]
            eval_params.extend(priority_params)
            
            # Add random parameters to evaluate
            remaining_params = [p for p in self.adjustable_parameters.keys() 
                              if p not in eval_params]
            
            # Determine how many to evaluate based on strategy's evaluation frequency
            eval_count = max(3, int(len(remaining_params) * 0.3 * evaluation_frequency))
            try:
                random_params = random.sample(remaining_params, min(eval_count, len(remaining_params)))
                eval_params.extend(random_params)
            except ValueError:
                # Handle case when remaining_params is too small
                eval_params.extend(remaining_params)
            
            # Ensure no duplicates
            eval_params = list(set(eval_params))
            
            # For each parameter to evaluate
            for param_name in eval_params:
                if param_name not in self.adjustable_parameters:
                    continue
                    
                param_config = self.adjustable_parameters[param_name]
                results["evaluated"].append(param_name)
                
                # Get relevant metrics for this parameter
                relevant_metrics = self.parameter_metrics_map.get(param_name, [])
                if not relevant_metrics:
                    continue
                    
                # Calculate current performance score for these metrics
                current_score = sum(performance.get(metric, 0) for metric in relevant_metrics)
                
                # Analyze historical performance for this parameter
                should_adjust, direction, confidence = await self._analyze_parameter_performance(
                    param_name, 
                    current_score,
                    relevant_metrics
                )
                
                # Apply strategy confidence modifier
                adjusted_confidence = confidence * confidence_multiplier
                
                if should_adjust and adjusted_confidence >= self.confidence_thresholds["medium"]:
                    # Calculate step size based on confidence and strategy
                    confidence_factor = min(1.0, adjusted_confidence / self.confidence_thresholds["high"])
                    step_size = param_config["step"] * direction * adjusted_step_multiplier
                    
                    # Check for parameter dependencies
                    dependency_adjusted_step = await self._adjust_for_dependencies(
                        param_name, step_size, confidence_factor
                    )
                    
                    # Calculate new value using the adjusted step
                    current = param_config["current"]
                    new_value = current + dependency_adjusted_step
                    
                    # Clamp to safe range
                    new_value = max(param_config["min"], min(param_config["max"], new_value))
                    
                    # Special case for boolean parameters
                    if param_config["min"] == 0 and param_config["max"] == 1 and param_config["step"] == 1:
                        new_value = round(new_value)
                    
                    # Only adjust if the value actually changed
                    if new_value != current:
                        # Update the parameter
                        await self._update_parameter(param_name, new_value)
                        
                        results["adjusted"].append({
                            "parameter": param_name,
                            "old_value": current,
                            "new_value": new_value,
                            "direction": "increase" if direction > 0 else "decrease",
                            "confidence": adjusted_confidence,
                            "relevant_metrics": relevant_metrics,
                            "category": param_config["category"]
                        })
                        
                        # If we changed an important parameter in abnormal state, stop
                        # to observe its effects before making more changes
                        if abnormal_state and param_config["category"] == "performance":
                            break
            
            # If reflection engine is available, generate a reflection on the changes
            if hasattr(self.brain, "reflection_engine") and self.brain.reflection_engine and results["adjusted"]:
                try:
                    reflection = await self.brain.reflection_engine.generate_reflection(
                        topic=f"parameter adjustment ({len(results['adjusted'])} changes)",
                        context={
                            "adjustments": results["adjusted"],
                            "performance": performance,
                            "strategy": self.current_adaptation_strategy
                        }
                    )
                    results["reflection"] = reflection
                    
                    # Also check if we should change adaptation strategy
                    if len(results["adjusted"]) > 5:
                        # Many parameters changed, consider more conservative approach
                        if self.current_adaptation_strategy == "exploratory":
                            results["strategy_recommendation"] = "Consider switching to balanced strategy after many parameters changed"
                    elif len(results["adjusted"]) == 0 and len(results["evaluated"]) > 10:
                        # Many evaluations but no changes, consider more exploratory approach
                        if self.current_adaptation_strategy == "conservative":
                            results["strategy_recommendation"] = "Consider switching to balanced strategy after few parameter changes"
                except Exception as e:
                    logger.error(f"Error generating reflection on parameter adjustments: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error evaluating and adjusting parameters: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    async def _adjust_for_dependencies(self, param_name, step_size, confidence_factor):
        """
        Adjust step size based on parameter dependencies
        
        Args:
            param_name: Parameter being adjusted
            step_size: Proposed step size
            confidence_factor: Confidence in the adjustment (0-1)
            
        Returns:
            Adjusted step size
        """
        # Get dependencies
        dependencies = self.parameter_dependencies.get(param_name, {"affects": [], "affected_by": []})
        
        # If no dependencies, return original step
        if not dependencies["affected_by"] and not dependencies["affects"]:
            return step_size
        
        # Check for conflicts with affected parameters
        for affected_param in dependencies["affects"]:
            if affected_param not in self.adjustable_parameters:
                continue
                
            # Get affected parameter recent history
            if affected_param in self.param_performance_impact:
                history = self.param_performance_impact[affected_param]["history"]
                if not history:
                    continue
                    
                # Check if affected parameter is already improving
                recent_entries = history[-3:] if len(history) >= 3 else history
                if len(recent_entries) < 2:
                    continue
                    
                avg_impact = sum(entry["impact"] for entry in recent_entries) / len(recent_entries)
                
                # If affected parameter is already improving well, reduce our step
                if avg_impact > 0.1:
                    step_size *= 0.8  # Reduce step size
        
        # Check if we're affected by other parameters
        for affecting_param in dependencies["affected_by"]:
            if affecting_param not in self.adjustable_parameters:
                continue
                
            # Check if affecting parameter recently changed
            if self.config_change_history:
                recent_changes = [change for change in self.config_change_history[-5:] 
                                 if change["parameter"] == affecting_param]
                
                if recent_changes:
                    # Recent change in a parameter that affects us
                    # Reduce our step size to avoid interference
                    step_size *= 0.7
        
        # Adjust based on confidence
        final_step = step_size * (0.7 + (0.3 * confidence_factor))
        
        return final_step
    
    async def _analyze_parameter_performance(self, 
                                          param_name: str, 
                                          current_score: float,
                                          relevant_metrics: List[str]) -> Tuple[bool, float, float]:
        """
        Analyze whether a parameter should be adjusted and in which direction
        
        Args:
            param_name: Name of parameter to analyze
            current_score: Current performance score
            relevant_metrics: List of relevant metric names
            
        Returns:
            Tuple of (should_adjust, direction, confidence)
        """
        # Initialize if not in history
        if param_name not in self.param_performance_impact:
            self.param_performance_impact[param_name] = {
                "history": [],
                "baseline": current_score
            }
        
        param_data = self.param_performance_impact[param_name]
        history = param_data["history"]
        
        # If no history yet, establish baseline and make exploratory change
        if not history:
            # Direction based on user feedback if available
            direction = 0.0
            
            if param_name in self.user_feedback_impact.get("parameter_adjustments", {}):
                feedback_data = self.user_feedback_impact["parameter_adjustments"][param_name]
                if feedback_data.get("recommended_direction") is not None:
                    direction = feedback_data["recommended_direction"]
            
            # Default to increase if no feedback
            if direction == 0.0:
                direction = 1.0
                
            return True, direction, 0.5  # Exploratory with medium confidence
        
        # Calculate average performance change per direction
        increases = [entry for entry in history if entry["direction"] > 0]
        decreases = [entry for entry in history if entry["direction"] < 0]
