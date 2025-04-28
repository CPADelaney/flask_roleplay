# nyx/core/brain/adaptation/self_config.py
import logging
import asyncio
import datetime
import json
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import RunContextWrapper

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
        
        return result
    
    async def _process_specific_feedback(self, 
                                     feedback_text: str, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process specific feedback"""
        result = {"parameters_affected": []}
        
        # Look for specific parameter mentions
        param_keywords = {
            "memory_recency_weight": ["recency", "recent", "memory recency"],
            "memory_relevance_weight": ["relevance", "relevant", "memory relevance"],
            "memory_significance_threshold": ["significance", "important", "memory significance"],
            "emotional_decay_rate": ["emotional decay", "emotion fade", "decay rate"],
            "emotional_expression_threshold": ["emotional expression", "express emotions", "expression threshold"],
            "emotional_complexity": ["emotional complexity", "emotion complexity", "complex emotions"],
            "cross_user_enabled": ["cross user", "shared experiences", "share experiences"],
            "cross_user_sharing_threshold": ["sharing threshold", "share threshold", "experience sharing"],
            "reasoning_depth": ["reasoning depth", "think depth", "deep reasoning"],
            "thinking_frequency": ["thinking frequency", "think frequency", "think more"],
            "experience_to_identity_influence": ["identity influence", "experience identity", "identity impact"]
        }
        
        # Check for parameter mentions
        mentioned_params = []
        for param, keywords in param_keywords.items():
            if any(keyword in feedback_text for keyword in keywords):
                mentioned_params.append(param)
        
        # Check for direction indicators
        increase_indicators = ["more", "higher", "increase", "stronger", "boost", "enhance"]
        decrease_indicators = ["less", "lower", "decrease", "weaker", "reduce", "limit"]
        
        has_increase = any(indicator in feedback_text for indicator in increase_indicators)
        has_decrease = any(indicator in feedback_text for indicator in decrease_indicators)
        
        # Process mentioned parameters
        for param_name in mentioned_params:
            if param_name in self.adjustable_parameters:
                param_data = self.adjustable_parameters[param_name]
                current = param_data["current"]
                step = param_data["step"]
                
                # Determine direction based on feedback
                direction = None
                if has_increase and not has_decrease:
                    direction = 1
                elif has_decrease and not has_increase:
                    direction = -1
                
                # If direction is clear, adjust parameter
                if direction is not None:
                    new_value = current + (step * direction)
                    
                    update_result = await self.update_parameter(
                        param_name, 
                        new_value, 
                        "specific_feedback_adjustment"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": param_name,
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": f"Adjusted based on specific feedback ({direction})"
                        })
        
        # If user_id is available, update user-specific parameters
        user_id = context.get("user_id", "unknown")
        if user_id != "unknown" and mentioned_params:
            for param_name in mentioned_params:
                if param_name in self.adjustable_parameters:
                    # Update user-specific preference
                    self.user_specific_parameters[user_id][param_name] = {
                        "direction": 1 if has_increase else (-1 if has_decrease else 0),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "feedback_text": feedback_text
                    }
                    
                    result["user_specific_updates"].append({
                        "param_name": param_name,
                        "preferred_direction": 1 if has_increase else (-1 if has_decrease else 0),
                        "user_id": user_id
                    })
        
        return result
    
    async def get_parameter_health_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive health report for all parameters
        
        Returns:
            Health report data
        """
        if not self.parameter_health:
            return {"status": "no_data", "message": "No parameter health data available"}
        
        # Calculate overall system health
        overall_scores = [data["health_score"] for data in self.parameter_health.values()]
        overall_health = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        # Get parameters with health issues
        unhealthy_params = []
        for param_name, health_data in self.parameter_health.items():
            if health_data["health_score"] < 0.7:
                unhealthy_params.append({
                    "param_name": param_name,
                    "health_score": health_data["health_score"],
                    "stability": health_data["stability"],
                    "exploration_ratio": health_data["exploration_ratio"],
                    "drift_from_default": health_data["drift_from_default"],
                    "boundary_violations": health_data["boundary_violations"]
                })
        
        # Sort by health score, ascending (worst first)
        unhealthy_params.sort(key=lambda x: x["health_score"])
        
        # Get healthiest parameters
        healthiest_params = []
        for param_name, health_data in sorted(self.parameter_health.items(), 
                                           key=lambda x: x[1]["health_score"], 
                                           reverse=True)[:5]:
            healthiest_params.append({
                "param_name": param_name,
                "health_score": health_data["health_score"],
                "stability": health_data["stability"],
                "exploration_ratio": health_data["exploration_ratio"]
            })
        
        # Get most stable parameters
        most_stable = []
        for param_name, health_data in sorted(self.parameter_health.items(), 
                                           key=lambda x: x[1]["stability"], 
                                           reverse=True)[:5]:
            most_stable.append({
                "param_name": param_name,
                "stability": health_data["stability"],
                "health_score": health_data["health_score"]
            })
        
        # Get most explored parameters
        most_explored = []
        for param_name, health_data in sorted(self.parameter_health.items(), 
                                           key=lambda x: x[1]["exploration_ratio"], 
                                           reverse=True)[:5]:
            most_explored.append({
                "param_name": param_name,
                "exploration_ratio": health_data["exploration_ratio"],
                "health_score": health_data["health_score"]
            })
        
        # Get parameters with boundary violations
        boundary_violators = []
        for param_name, health_data in sorted(self.parameter_health.items(), 
                                           key=lambda x: x[1]["boundary_violations"], 
                                           reverse=True):
            if health_data["boundary_violations"] > 0:
                boundary_violators.append({
                    "param_name": param_name,
                    "boundary_violations": health_data["boundary_violations"],
                    "health_score": health_data["health_score"]
                })
        
        # Get parameters with high drift
        high_drift = []
        for param_name, health_data in sorted(self.parameter_health.items(), 
                                           key=lambda x: x[1]["drift_from_default"], 
                                           reverse=True)[:5]:
            high_drift.append({
                "param_name": param_name,
                "drift_from_default": health_data["drift_from_default"],
                "health_score": health_data["health_score"]
            })
        
        # Get categories with health issues
        category_health = {}
        for param_name, health_data in self.parameter_health.items():
            if param_name in self.adjustable_parameters:
                category = self.adjustable_parameters[param_name].get("category", "unknown")
                
                if category not in category_health:
                    category_health[category] = {
                        "parameters": 0,
                        "total_health": 0.0,
                        "avg_health": 0.0
                    }
                
                category_health[category]["parameters"] += 1
                category_health[category]["total_health"] += health_data["health_score"]
        
        # Calculate average health by category
        for category, data in category_health.items():
            if data["parameters"] > 0:
                data["avg_health"] = data["total_health"] / data["parameters"]
        
        # Sort categories by average health
        sorted_categories = sorted(category_health.items(), key=lambda x: x[1]["avg_health"])
        
        return {
            "overall_health": overall_health,
            "total_parameters": len(self.parameter_health),
            "unhealthy_parameters": unhealthy_params[:5],  # Top 5 unhealthy
            "healthiest_parameters": healthiest_params,
            "most_stable_parameters": most_stable,
            "most_explored_parameters": most_explored,
            "boundary_violators": boundary_violators,
            "high_drift_parameters": high_drift,
            "category_health": {
                "worst": sorted_categories[:3] if len(sorted_categories) >= 3 else sorted_categories,
                "best": sorted_categories[-3:] if len(sorted_categories) >= 3 else []
            },
            "system_state": await self._get_system_state_summary()
        }
    
    async def _get_system_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the overall system state"""
        # Run all validators
        validation_results = {}
        for validator_name, validator_func in self.state_validators.items():
            try:
                validation_result = await validator_func()
                validation_results[validator_name] = validation_result
            except Exception as e:
                logger.error(f"Error in validator {validator_name}: {str(e)}")
                validation_results[validator_name] = {"error": str(e)}
        
        # Count issues
        bound_violations = sum(1 for result in validation_results.values() 
                             if isinstance(result, dict) and "violations" in result 
                             for violation in result["violations"])
        
        coherence_issues = sum(1 for result in validation_results.values() 
                             if isinstance(result, dict) and "incoherencies" in result 
                             for issue in result["incoherencies"])
        
        consistency_issues = sum(1 for result in validation_results.values() 
                               if isinstance(result, dict) and "inconsistencies" in result 
                               for issue in result["inconsistencies"])
        
        # Overall status assessment
        total_issues = bound_violations + coherence_issues + consistency_issues
        
        if total_issues == 0:
            status = "healthy"
        elif total_issues < 3:
            status = "minor_issues"
        elif total_issues < 7:
            status = "moderate_issues"
        else:
            status = "significant_issues"
        
        return {
            "status": status,
            "bound_violations": bound_violations,
            "coherence_issues": coherence_issues,
            "consistency_issues": consistency_issues,
            "total_issues": total_issues,
            "validation_results": validation_results
        }
    
    async def run_self_optimization(self) -> Dict[str, Any]:
        """
        Run an automatic self-optimization process to improve system health
        
        Returns:
            Optimization results
        """
        # Get current health report
        health_report = await self.get_parameter_health_report()
        
        # Create optimization results
        optimization_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "initial_health": health_report["overall_health"],
            "parameters_adjusted": [],
            "health_improvements": {}
        }
        
        # Focus on unhealthy parameters
        for param_info in health_report["unhealthy_parameters"]:
            param_name = param_info["param_name"]
            
            if param_name not in self.adjustable_parameters:
                continue
                
            param_data = self.adjustable_parameters[param_name]
            health_data = self.parameter_health[param_name]
            
            # Determine optimization strategy based on health issues
            if health_data["stability"] < 0.6:
                # Low stability - move toward default to stabilize
                current = param_data["current"]
                default = param_data["default"]
                step = param_data["step"] * 0.5  # Smaller step for stability
                
                # Move toward default
                direction = 1 if default > current else -1
                new_value = current + (step * direction)
                
                update_result = await self.update_parameter(
                    param_name, 
                    new_value, 
                    "self_optimization_stability"
                )
                
                if update_result["changed"]:
                    optimization_results["parameters_adjusted"].append({
                        "param_name": param_name,
                        "old_value": update_result["old_value"],
                        "new_value": update_result["new_value"],
                        "reason": "Improved stability by moving toward default",
                        "health_before": health_data["health_score"]
                    })
            
            elif health_data["exploration_ratio"] < 0.3:
                # Under-explored - try a new area of the parameter space
                current = param_data["current"]
                min_val = param_data["min"]
                max_val = param_data["max"]
                step = param_data["step"]
                
                # Get exploration data
                exploration_data = self.enhanced_metrics["parameter_exploration"].get(param_name, {})
                bin_counts = exploration_data.get("visits", {}).get("counts", [])
                
                if bin_counts:
                    # Find the least explored bin
                    bins = len(bin_counts)
                    bin_size = (max_val - min_val) / bins
                    
                    least_explored = min(range(len(bin_counts)), key=lambda i: bin_counts[i])
                    
                    # Calculate a value in that bin
                    target_value = min_val + (least_explored + 0.5) * bin_size
                    
                    # Move toward the target
                    direction = 1 if target_value > current else -1
                    new_value = current + (step * direction)
                    
                    update_result = await self.update_parameter(
                        param_name, 
                        new_value, 
                        "self_optimization_exploration"
                    )
                    
                    if update_result["changed"]:
                        optimization_results["parameters_adjusted"].append({
                            "param_name": param_name,
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Improved exploration by targeting under-explored region",
                            "health_before": health_data["health_score"]
                        })
            
            elif health_data["drift_from_default"] > 0.7:
                # High drift - move back toward default
                current = param_data["current"]
                default = param_data["default"]
                step = param_data["step"]
                
                # Move toward default with larger step
                direction = 1 if default > current else -1
                new_value = current + (step * direction)
                
                update_result = await self.update_parameter(
                    param_name, 
                    new_value, 
                    "self_optimization_drift"
                )
                
                if update_result["changed"]:
                    optimization_results["parameters_adjusted"].append({
                        "param_name": param_name,
                        "old_value": update_result["old_value"],
                        "new_value": update_result["new_value"],
                        "reason": "Reduced drift by moving toward default",
                        "health_before": health_data["health_score"]
                    })
            
            # If we've already adjusted this parameter, skip remaining checks
            if any(p["param_name"] == param_name for p in optimization_results["parameters_adjusted"]):
                continue
            
            # Check for boundary violations
            if health_data["boundary_violations"] > 0:
                # Move away from boundaries
                current = param_data["current"]
                min_val = param_data["min"]
                max_val = param_data["max"]
                step = param_data["step"] * 0.5  # Smaller step
                
                # Check which boundary we're closer to
                if abs(current - min_val) < abs(current - max_val):
                    # Close to min, move up
                    new_value = current + step
                else:
                    # Close to max, move down
                    new_value = current - step
                
                update_result = await self.update_parameter(
                    param_name, 
                    new_value, 
                    "self_optimization_boundaries"
                )
                
                if update_result["changed"]:
                    optimization_results["parameters_adjusted"].append({
                        "param_name": param_name,
                        "old_value": update_result["old_value"],
                        "new_value": update_result["new_value"],
                        "reason": "Moved away from parameter boundaries",
                        "health_before": health_data["health_score"]
                    })
        
        # Check for parameter coherence issues and fix them
        coherence_result = await self._validate_parameter_coherence()
        if coherence_result.get("incoherencies"):
            for issue in coherence_result["incoherencies"]:
                if issue["type"] == "bidirectional_influence":
                    # Fix bidirectional influence balance
                    params = issue["parameters"]
                    values = issue["values"]
                    
                    # Reduce both proportionally
                    reduction_factor = 0.9  # Reduce by 10%
                    
                    for i, param_name in enumerate(params):
                        if param_name in self.adjustable_parameters:
                            new_value = values[i] * reduction_factor
                            
                            update_result = await self.update_parameter(
                                param_name, 
                                new_value, 
                                "self_optimization_coherence"
                            )
                            
                            if update_result["changed"]:
                                optimization_results["parameters_adjusted"].append({
                                    "param_name": param_name,
                                    "old_value": update_result["old_value"],
                                    "new_value": update_result["new_value"],
                                    "reason": "Fixed parameter coherence: " + issue["issue"],
                                    "health_before": self.parameter_health[param_name]["health_score"]
                                })
                
                elif issue["type"] == "contradictory_settings":
                    # Fix contradictory settings
                    params = issue["parameters"]
                    
                    # For contradictory cross-user settings, enable sharing but increase threshold
                    if "cross_user_enabled" in params and "cross_user_sharing_threshold" in params:
                        # Update enabled to 1
                        update_result = await self.update_parameter(
                            "cross_user_enabled", 
                            1, 
                            "self_optimization_coherence"
                        )
                        
                        if update_result["changed"]:
                            optimization_results["parameters_adjusted"].append({
                                "param_name": "cross_user_enabled",
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Fixed parameter coherence: " + issue["issue"],
                                "health_before": self.parameter_health["cross_user_enabled"]["health_score"]
                            })
                        
                        # Update threshold to be higher
                        update_result = await self.update_parameter(
                            "cross_user_sharing_threshold", 
                            0.7, 
                            "self_optimization_coherence"
                        )
                        
                        if update_result["changed"]:
                            optimization_results["parameters_adjusted"].append({
                                "param_name": "cross_user_sharing_threshold",
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Fixed parameter coherence: " + issue["issue"],
                                "health_before": self.parameter_health["cross_user_sharing_threshold"]["health_score"]
                            })
                
                elif issue["type"] == "threshold_ordering":
                    # Fix threshold ordering
                    params = issue["parameters"]
                    values = issue["values"]
                    
                    # Increase the distributed threshold
                    if "distributed_processing_threshold" in params:
                        parallel_idx = params.index("parallel_processing_threshold")
                        distributed_idx = params.index("distributed_processing_threshold")
                        
                        parallel_val = values[parallel_idx]
                        
                        # Set distributed threshold higher than parallel
                        new_value = parallel_val + 0.1
                        
                        update_result = await self.update_parameter(
                            "distributed_processing_threshold", 
                            new_value, 
                            "self_optimization_coherence"
                        )
                        
                        if update_result["changed"]:
                            optimization_results["parameters_adjusted"].append({
                                "param_name": "distributed_processing_threshold",
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Fixed parameter coherence: " + issue["issue"],
                                "health_before": self.parameter_health["distributed_processing_threshold"]["health_score"]
                            })
        
        # Check for cross-module consistency issues and fix them
        consistency_result = await self._validate_cross_module_consistency()
        if consistency_result.get("inconsistencies"):
            for issue in consistency_result["inconsistencies"]:
                param_name = issue["parameter"]
                
                if param_name in self.adjustable_parameters:
                    # For brain attributes, sync with the brain
                    if "brain_value" in issue:
                        update_result = await self.update_parameter(
                            param_name, 
                            issue["brain_value"], 
                            "self_optimization_consistency"
                        )
                        
                        if update_result["changed"]:
                            optimization_results["parameters_adjusted"].append({
                                "param_name": param_name,
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Fixed cross-module consistency",
                                "health_before": self.parameter_health[param_name]["health_score"]
                            })
                    
                    # For module attributes, update the module value
                    elif "module_value" in issue:
                        config_value = issue["config_value"]
                        module = issue["module"]
                        
                        # Update the parameter first (in case this triggers updates elsewhere)
                        update_result = await self.update_parameter(
                            param_name, 
                            config_value, 
                            "self_optimization_consistency"
                        )
                        
                        if update_result["changed"]:
                            optimization_results["parameters_adjusted"].append({
                                "param_name": param_name,
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": f"Fixed cross-module consistency with {module}",
                                "health_before": self.parameter_health[param_name]["health_score"]
                            })
        
        # Get updated health report
        new_health_report = await self.get_parameter_health_report()
        optimization_results["final_health"] = new_health_report["overall_health"]
        
        # Calculate health improvements for adjusted parameters
        for adjustment in optimization_results["parameters_adjusted"]:
            param_name = adjustment["param_name"]
            health_before = adjustment["health_before"]
            
            if param_name in self.parameter_health:
                health_after = self.parameter_health[param_name]["health_score"]
                health_change = health_after - health_before
                
                optimization_results["health_improvements"][param_name] = {
                    "before": health_before,
                    "after": health_after,
                    "change": health_change
                }
        
        # Calculate overall health improvement
        optimization_results["overall_health_change"] = (
            optimization_results["final_health"] - optimization_results["initial_health"]
        )
        
        # Create a snapshot of the optimized state
        await self._create_config_snapshot("self_optimization_completed")
        
        return optimization_results
    
    async def experiment_with_parameter(self, 
                                    param_name: str, 
                                    value_range: List[float] = None,
                                    duration: int = 5) -> Dict[str, Any]:
        """
        Run an experiment with a parameter
        
        Args:
            param_name: Parameter to experiment with
            value_range: Range of values to try, if None, use default range
            duration: Number of interactions to test each value for
            
        Returns:
            Experiment setup results
        """
        if param_name not in self.adjustable_parameters:
            return {"success": False, "error": f"Parameter '{param_name}' not found"}
        
        param_data = self.adjustable_parameters[param_name]
        
        # Generate values to test if not provided
        if value_range is None:
            min_val = param_data["min"]
            max_val = param_data["max"]
            step = param_data["step"]
            
            # Generate 3 values: min, middle, max
            middle = (min_val + max_val) / 2
            value_range = [min_val, middle, max_val]
        
        # Setup experiment
        experiment_id = f"exp_{param_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        experiment = {
            "id": experiment_id,
            "param_name": param_name,
            "values": value_range,
            "duration": duration,
            "current_value_index": 0,
            "current_value": value_range[0],
            "interactions_with_current": 0,
            "original_value": param_data["current"],
            "results": {val: {"metrics": {}, "health_score": None} for val in value_range},
            "start_time": datetime.datetime.now().isoformat(),
            "status": "running"
        }
        
        # Store experiment
        self.experiments.append(experiment)
        self.active_experiments[experiment_id] = experiment
        
        # Set the first experiment value
        update_result = await self.update_parameter(
            param_name, 
            value_range[0], 
            f"experiment_{experiment_id}"
        )
        
        # Return experiment details
        return {
            "success": True,
            "experiment_id": experiment_id,
            "param_name": param_name,
            "values": value_range,
            "duration": duration,
            "current_value": value_range[0],
            "original_value": param_data["current"]
        }
    
    async def update_experiment(self, 
                            experiment_id: str, 
                            metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Update an experiment with new metrics and possibly advance to next value
        
        Args:
            experiment_id: ID of the experiment
            metrics: Current performance metrics
            
        Returns:
            Update results
        """
        if experiment_id not in self.active_experiments:
            return {"success": False, "error": f"Experiment '{experiment_id}' not found or not active"}
        
        experiment = self.active_experiments[experiment_id]
        param_name = experiment["param_name"]
        
        # Update metrics for current value
        current_value = experiment["current_value"]
        
        if metrics:
            if current_value not in experiment["results"]:
                experiment["results"][current_value] = {"metrics": {}, "health_score": None}
                
            # Update metrics
            for metric_name, metric_value in metrics.items():
                if metric_name not in experiment["results"][current_value]["metrics"]:
                    experiment["results"][current_value]["metrics"][metric_name] = []
                    
                experiment["results"][current_value]["metrics"][metric_name].append(metric_value)
        
        # Update health score for current value
        if param_name in self.parameter_health:
            experiment["results"][current_value]["health_score"] = self.parameter_health[param_name]["health_score"]
        
        # Increment interactions with current value
        experiment["interactions_with_current"] += 1
        
        # Check if we should advance to next value
        if experiment["interactions_with_current"] >= experiment["duration"]:
            # Move to next value
            experiment["current_value_index"] += 1
            
            # Check if experiment is complete
            if experiment["current_value_index"] >= len(experiment["values"]):
                # Experiment complete, restore original value
                await self.update_parameter(
                    param_name, 
                    experiment["original_value"], 
                    f"experiment_{experiment_id}_complete"
                )
                
                # Mark as complete
                experiment["status"] = "complete"
                experiment["end_time"] = datetime.datetime.now().isoformat()
                
                # Remove from active experiments
                del self.active_experiments[experiment_id]
                
                # Analyze results
                results = await self._analyze_experiment_results(experiment)
                
                return {
                    "success": True,
                    "status": "complete",
                    "best_value": results["best_value"],
                    "best_metrics": results["best_metrics"],
                    "experiment": experiment
                }
            else:
                # Move to next value
                next_value = experiment["values"][experiment["current_value_index"]]
                experiment["current_value"] = next_value
                experiment["interactions_with_current"] = 0
                
                # Update parameter
                await self.update_parameter(
                    param_name, 
                    next_value, 
                    f"experiment_{experiment_id}_next_value"
                )
                
                return {
                    "success": True,
                    "status": "advanced",
                    "current_value_index": experiment["current_value_index"],
                    "current_value": next_value,
                    "values_remaining": len(experiment["values"]) - experiment["current_value_index"]
                }
        
        # Continue with current value
        return {
            "success": True,
            "status": "continuing",
            "current_value": current_value,
            "interactions": experiment["interactions_with_current"],
            "remaining": experiment["duration"] - experiment["interactions_with_current"]
        }
    
    async def _analyze_experiment_results(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results to determine the best value"""
        results = experiment["results"]
        param_name = experiment["param_name"]
        
        # Calculate average metrics for each value
        avg_metrics = {}
        for value, value_results in results.items():
            avg_metrics[value] = {}
            
            for metric_name, metric_values in value_results["metrics"].items():
                if metric_values:
                    avg_metrics[value][metric_name] = sum(metric_values) / len(metric_values)
        
        # Normalize metrics (higher is better)
        normalized_scores = {}
        for value in avg_metrics:
            score = 0.0
            
            # Add health score if available
            if results[value]["health_score"] is not None:
                score += results[value]["health_score"]
            
            # Add normalized metrics
            for metric_name, metric_value in avg_metrics[value].items():
                # Some metrics are better when higher, some when lower
                if metric_name in ["success_rate", "experiences_shared"]:
                    # Higher is better
                    score += metric_value
                elif metric_name in ["avg_response_time", "error_rate"]:
                    # Lower is better, invert
                    score += (1.0 - metric_value) if 0.0 <= metric_value <= 1.0 else 0.0
            
            normalized_scores[value] = score
        
        # Find best value
        best_value = max(normalized_scores.items(), key=lambda x: x[1])
        
        # Format results
        return {
            "best_value": best_value[0],
            "best_score": best_value[1],
            "all_scores": normalized_scores,
            "avg_metrics": avg_metrics
        }
    
    async def recommend_parameters_for_user(self, user_id: str) -> Dict[str, Any]:
        """
        Generate parameter recommendations for a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            Parameter recommendations
        """
        # Check if we have user-specific parameters
        if user_id not in self.user_specific_parameters:
            return {"success": False, "message": "No user-specific parameters available"}
        
        user_params = self.user_specific_parameters[user_id]
        
        # Check if we have user feedback history
        user_feedback = {}
        if hasattr(self, "user_feedback_impact"):
            for feedback_type, feedback_list in self.user_feedback_impact.items():
                user_feedback[feedback_type] = [f for f in feedback_list if f.get("user_id") == user_id]
        
        # Generate recommendations
        recommendations = []
        
        # 1. Start with parameters the user has directly provided feedback on
        for param_name, param_data in user_params.items():
            if param_name in self.adjustable_parameters:
                direction = param_data.get("direction", 0)
                
                if direction != 0:
                    # User has expressed a preference for this parameter
                    current = self.adjustable_parameters[param_name]["current"]
                    step = self.adjustable_parameters[param_name]["step"] * 0.5  # Smaller step
                    
                    # Calculate recommended value
                    new_value = current + (step * direction)
                    
                    recommendations.append({
                        "param_name": param_name,
                        "current_value": current,
                        "recommended_value": new_value,
                        "confidence": 0.8,
                        "reason": "User has expressed a preference for this parameter",
                        "feedback_source": param_data.get("feedback_text", "")
                    })
        
        # 2. Add parameters from the same category as those the user has given feedback on
        categories_of_interest = set()
        
        for param_name in user_params:
            if param_name in self.adjustable_parameters:
                category = self.adjustable_parameters[param_name].get("category", "unknown")
                categories_of_interest.add(category)
        
        for category in categories_of_interest:
            # Find parameters in the same category that haven't been recommended yet
            params_in_category = [p for p in self.adjustable_parameters.keys() 
                                if self.adjustable_parameters[p].get("category") == category and
                                p not in [rec["param_name"] for rec in recommendations]]
            
            for param_name in params_in_category:
                # Check health
                if param_name in self.parameter_health and self.parameter_health[param_name]["health_score"] < 0.7:
                    # Parameter needs improvement
                    current = self.adjustable_parameters[param_name]["current"]
                    default = self.adjustable_parameters[param_name]["default"]
                    
                    # Move toward default
                    step = self.adjustable_parameters[param_name]["step"] * 0.3  # Small step
                    direction = 1 if default > current else -1
                    new_value = current + (step * direction)
                    
                    recommendations.append({
                        "param_name": param_name,
                        "current_value": current,
                        "recommended_value": new_value,
                        "confidence": 0.6,
                        "reason": f"Parameter in {category} category needs improvement",
                        "health_score": self.parameter_health[param_name]["health_score"]
                    })
        
        # Sort recommendations by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "user_id": user_id,
            "recommendations": recommendations,
            "user_feedback_count": sum(len(feedback_list) for feedback_list in user_feedback.values())
        }





class EnhancedConfigManager:
    """
    Enhanced self-configuration management for the Nyx brain
    Designed to work alongside the existing SelfConfigManager
    """
    
    def __init__(self, brain, base_config_manager=None):
        """
        Initialize the enhanced configuration manager
        
        Args:
            brain: Reference to the NyxBrain instance
            base_config_manager: Reference to the existing SelfConfigManager
        """
        self.brain = brain
        self.base_config_manager = base_config_manager
        
        # Parameter interdependency graph enhancements
        self.parameter_relationships = {}
        self.parameter_clusters = {}
        
        # Parameter health tracking
        self.parameter_health = {}
        
        # State validation registry
        self.state_validators = {}
        
        # Enhanced learning metrics
        self.enhanced_metrics = {
            "parameter_stability": {},
            "parameter_responsiveness": {},
            "parameter_exploration": {},
            "parameter_boundaries": {}
        }
        
        # Multi-user parameter adaptation
        self.user_specific_parameters = {}
        
        # Experiment tracking
        self.experiments = []
        self.active_experiments = {}
        
        # Meta-level patterns
        self.meta_patterns = []
        
        # Configuration states history
        self.config_states = []
        
        # Cross-module validation results
        self.validation_results = {}
        
        logger.info("Enhanced configuration manager initialized")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the enhanced configuration system
        
        Returns:
            Initialization results
        """
        # Initialize parameter relationships from base config manager
        if self.base_config_manager:
            await self._initialize_from_base_manager()
        
        # Build enhanced parameter relationships
        await self._build_parameter_relationships()
        
        # Initialize health metrics for all parameters
        await self._initialize_parameter_health()
        
        # Register state validators
        self._register_default_validators()
        
        # Create initial config state snapshot
        await self._create_config_snapshot("initialization")
        
        return {
            "parameters_tracked": len(self.parameter_health),
            "relationships_mapped": len(self.parameter_relationships),
            "validators_registered": len(self.state_validators),
            "clusters_identified": len(self.parameter_clusters)
        }
    
    async def _initialize_from_base_manager(self) -> None:
        """Initialize data from the base configuration manager"""
        if not self.base_config_manager:
            return
        
        # Copy adjustable parameters
        if hasattr(self.base_config_manager, "adjustable_parameters"):
            self.adjustable_parameters = self.base_config_manager.adjustable_parameters.copy()
        
        # Copy parameter dependencies
        if hasattr(self.base_config_manager, "parameter_dependencies"):
            # Deep copy dependencies
            for param_name, dependencies in self.base_config_manager.parameter_dependencies.items():
                if param_name not in self.parameter_relationships:
                    self.parameter_relationships[param_name] = {
                        "affects": dependencies.get("affects", []).copy(),
                        "affected_by": dependencies.get("affected_by", []).copy(),
                        "strength": {},  # Will populate with relationship strengths
                        "correlation": {}  # Will track correlation between parameters
                    }
        
        # Copy parameter performance impact data
        if hasattr(self.base_config_manager, "param_performance_impact"):
            for param_name, impact_data in self.base_config_manager.param_performance_impact.items():
                self.enhanced_metrics["parameter_responsiveness"][param_name] = {
                    "last_changes": [],
                    "impact_history": impact_data.get("history", []).copy() if "history" in impact_data else []
                }
        
        # Copy user feedback impact
        if hasattr(self.base_config_manager, "user_feedback_impact"):
            self.user_feedback_impact = self.base_config_manager.user_feedback_impact.copy()
        
        # Copy adaptation strategies
        if hasattr(self.base_config_manager, "adaptation_strategies"):
            self.adaptation_strategies = self.base_config_manager.adaptation_strategies.copy()
            self.current_adaptation_strategy = getattr(
                self.base_config_manager, "current_adaptation_strategy", "balanced"
            )
        
        logger.info("Initialized from base configuration manager")
    
    async def _build_parameter_relationships(self) -> None:
        """Build enhanced parameter relationships"""
        if not hasattr(self, "adjustable_parameters"):
            return
        
        # Create relationship strengths based on declared relationships
        for param_name, param_data in self.parameter_relationships.items():
            # Affected parameters - initialize strength
            for affected_param in param_data["affects"]:
                if affected_param in self.adjustable_parameters:
                    param_data["strength"][affected_param] = 0.5  # Default medium strength
            
            # Parameters that affect this one - initialize strength
            for affecting_param in param_data["affected_by"]:
                if affecting_param in self.adjustable_parameters:
                    if affecting_param not in self.parameter_relationships:
                        self.parameter_relationships[affecting_param] = {
                            "affects": [param_name],
                            "affected_by": [],
                            "strength": {param_name: 0.5},
                            "correlation": {}
                        }
                    elif param_name not in self.parameter_relationships[affecting_param]["affects"]:
                        self.parameter_relationships[affecting_param]["affects"].append(param_name)
                        self.parameter_relationships[affecting_param]["strength"][param_name] = 0.5
        
        # Discover parameter clusters through graph analysis
        await self._identify_parameter_clusters()
        
        logger.info(f"Built parameter relationships for {len(self.parameter_relationships)} parameters")
    
    async def _identify_parameter_clusters(self) -> None:
        """Identify clusters of related parameters using graph analysis"""
        # Simple connected components algorithm
        visited = set()
        clusters = []
        
        for param_name in self.parameter_relationships:
            if param_name in visited:
                continue
                
            # Start a new cluster
            cluster = []
            queue = [param_name]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                    
                visited.add(current)
                cluster.append(current)
                
                # Add neighbors
                if current in self.parameter_relationships:
                    relations = self.parameter_relationships[current]
                    neighbors = relations["affects"] + relations["affected_by"]
                    
                    for neighbor in neighbors:
                        if neighbor not in visited and neighbor in self.parameter_relationships:
                            queue.append(neighbor)
            
            if cluster:
                clusters.append(cluster)
        
        # Create dictionary of named clusters
        for i, cluster in enumerate(clusters):
            cluster_name = f"cluster_{i+1}"
            
            # Try to generate a meaningful name based on categories
            category_counts = {}
            for param in cluster:
                if param in self.adjustable_parameters:
                    category = self.adjustable_parameters[param].get("category", "unknown")
                    if category not in category_counts:
                        category_counts[category] = 0
                    category_counts[category] += 1
            
            if category_counts:
                # Use the most common category as the cluster name
                most_common = max(category_counts.items(), key=lambda x: x[1])
                cluster_name = f"{most_common[0]}_cluster_{i+1}"
            
            self.parameter_clusters[cluster_name] = {
                "parameters": cluster,
                "size": len(cluster),
                "dominant_category": most_common[0] if category_counts else "mixed"
            }
        
        logger.info(f"Identified {len(self.parameter_clusters)} parameter clusters")
    
    async def _initialize_parameter_health(self) -> None:
        """Initialize health metrics for all parameters"""
        if not hasattr(self, "adjustable_parameters"):
            return
        
        for param_name, param_data in self.adjustable_parameters.items():
            # Initialize health data for each parameter
            self.parameter_health[param_name] = {
                "stability": 1.0,  # Start with perfect stability
                "exploration_ratio": 0.0,  # No exploration yet
                "change_frequency": 0.0,  # No changes yet
                "drift_from_default": abs(param_data["current"] - param_data["default"]) / 
                                   (param_data["max"] - param_data["min"]) if 
                                   (param_data["max"] - param_data["min"]) > 0 else 0.0,
                "boundary_violations": 0,  # No violations yet
                "last_updated": datetime.datetime.now().isoformat(),
                "health_score": 1.0  # Start with perfect health
            }
            
            # Initialize stability tracking
            self.enhanced_metrics["parameter_stability"][param_name] = {
                "values": [param_data["current"]],
                "timestamps": [datetime.datetime.now().isoformat()],
                "variance": 0.0
            }
            
            # Initialize exploration metrics
            self.enhanced_metrics["parameter_exploration"][param_name] = {
                "exploration_range": [param_data["current"], param_data["current"]],
                "range_coverage": 0.0,  # Percentage of possible range explored
                "visits": {  # Divide range into bins and track visits
                    "bins": 10,
                    "counts": [0] * 10
                }
            }
            
            # Initialize boundary tracking
            self.enhanced_metrics["parameter_boundaries"][param_name] = {
                "min_tested": param_data["current"],
                "max_tested": param_data["current"],
                "boundary_approaches": 0,  # Times approached boundary
                "min_violations": 0,
                "max_violations": 0
            }
            
            # Add to parameter responsiveness if not already there
            if param_name not in self.enhanced_metrics["parameter_responsiveness"]:
                self.enhanced_metrics["parameter_responsiveness"][param_name] = {
                    "last_changes": [],
                    "impact_history": []
                }
            
            # Update exploration bins
            self._update_exploration_bins(param_name, param_data["current"])
        
        logger.info(f"Initialized health metrics for {len(self.parameter_health)} parameters")
    
    def _register_default_validators(self) -> None:
        """Register default state validators"""
        # Register validators for different categories
        self.register_state_validator("parameter_bounds", self._validate_parameter_bounds)
        self.register_state_validator("parameter_coherence", self._validate_parameter_coherence)
        self.register_state_validator("cross_module_consistency", self._validate_cross_module_consistency)
        
        logger.info("Registered default state validators")
    
    def register_state_validator(self, name: str, validator_func: callable) -> None:
        """
        Register a state validator function
        
        Args:
            name: Name of the validator
            validator_func: Validation function that returns Dict[str, Any]
        """
        self.state_validators[name] = validator_func
        logger.info(f"Registered state validator: {name}")
    
    async def _create_config_snapshot(self, reason: str) -> Dict[str, Any]:
        """
        Create a snapshot of the current configuration state
        
        Args:
            reason: Reason for creating the snapshot
            
        Returns:
            Snapshot data
        """
        if not hasattr(self, "adjustable_parameters"):
            return {"error": "No adjustable parameters available"}
        
        # Create parameter snapshot
        parameter_values = {}
        for param_name, param_data in self.adjustable_parameters.items():
            parameter_values[param_name] = param_data["current"]
        
        # Create snapshot record
        snapshot = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason,
            "parameters": parameter_values,
            "strategy": getattr(self, "current_adaptation_strategy", "unknown"),
            "validation_results": {}
        }
        
        # Run validators
        for validator_name, validator_func in self.state_validators.items():
            try:
                validation_result = await validator_func()
                snapshot["validation_results"][validator_name] = validation_result
                self.validation_results[validator_name] = validation_result
            except Exception as e:
                logger.error(f"Error in validator {validator_name}: {str(e)}")
                snapshot["validation_results"][validator_name] = {"error": str(e)}
        
        # Add to history
        self.config_states.append(snapshot)
        
        # Keep history to a reasonable size
        if len(self.config_states) > 100:
            self.config_states = self.config_states[-100:]
        
        return snapshot
    
    async def update_parameter(self, 
                           param_name: str, 
                           new_value: float, 
                           reason: str = None) -> Dict[str, Any]:
        """
        Update a parameter with enhanced tracking
        
        Args:
            param_name: Name of the parameter to update
            new_value: New value for the parameter
            reason: Reason for the update
            
        Returns:
            Update result
        """
        # Check if parameter exists
        if not hasattr(self, "adjustable_parameters") or param_name not in self.adjustable_parameters:
            return {"success": False, "error": f"Parameter '{param_name}' not found"}
        
        param_data = self.adjustable_parameters[param_name]
        old_value = param_data["current"]
        
        # Ensure value is within bounds
        min_val = param_data["min"]
        max_val = param_data["max"]
        
        if new_value < min_val:
            # Track boundary violation
            if param_name in self.enhanced_metrics["parameter_boundaries"]:
                self.enhanced_metrics["parameter_boundaries"][param_name]["min_violations"] += 1
            
            new_value = min_val
        elif new_value > max_val:
            # Track boundary violation
            if param_name in self.enhanced_metrics["parameter_boundaries"]:
                self.enhanced_metrics["parameter_boundaries"][param_name]["max_violations"] += 1
            
            new_value = max_val
        
        # Only proceed if the value actually changes
        if new_value == old_value:
            return {
                "success": True, 
                "param_name": param_name, 
                "value": new_value,
                "changed": False,
                "reason": "Value unchanged"
            }
        
        # Update the parameter
        param_data["current"] = new_value
        
        # Track this change
        timestamp = datetime.datetime.now()
        change_record = {
            "timestamp": timestamp.isoformat(),
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason or "unspecified"
        }
        
        # Update enhanced metrics
        await self._update_enhanced_metrics(param_name, old_value, new_value, timestamp, reason)
        
        # Update parameter health
        await self._update_parameter_health(param_name)
        
        # Update base config manager if available
        if self.base_config_manager and hasattr(self.base_config_manager, "adjustable_parameters"):
            if param_name in self.base_config_manager.adjustable_parameters:
                self.base_config_manager.adjustable_parameters[param_name]["current"] = new_value
        
        # Propagate to brain if available
        if self.brain and hasattr(self.brain, param_name):
            setattr(self.brain, param_name, new_value)
            
            # If it's a boolean parameter, convert 0/1 to False/True
            if param_data["min"] == 0 and param_data["max"] == 1 and param_data["step"] == 1:
                setattr(self.brain, param_name, bool(new_value))
        
        # Create config snapshot
        await self._create_config_snapshot(f"parameter_update_{param_name}")
        
        return {
            "success": True,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "changed": True,
            "reason": reason or "unspecified",
            "health": self.parameter_health.get(param_name, {})
        }
    
    async def _update_enhanced_metrics(self, 
                                   param_name: str, 
                                   old_value: float, 
                                   new_value: float,
                                   timestamp: datetime.datetime,
                                   reason: str) -> None:
        """Update enhanced metrics for a parameter change"""
        # Update stability metrics
        if param_name in self.enhanced_metrics["parameter_stability"]:
            stability_data = self.enhanced_metrics["parameter_stability"][param_name]
            
            # Add new value and timestamp
            stability_data["values"].append(new_value)
            stability_data["timestamps"].append(timestamp.isoformat())
            
            # Keep a reasonable history
            if len(stability_data["values"]) > 20:
                stability_data["values"] = stability_data["values"][-20:]
                stability_data["timestamps"] = stability_data["timestamps"][-20:]
            
            # Update variance
            if len(stability_data["values"]) >= 2:
                stability_data["variance"] = statistics.variance(stability_data["values"])
        
        # Update exploration metrics
        if param_name in self.enhanced_metrics["parameter_exploration"]:
            exploration_data = self.enhanced_metrics["parameter_exploration"][param_name]
            
            # Update exploration range
            min_seen = exploration_data["exploration_range"][0]
            max_seen = exploration_data["exploration_range"][1]
            
            exploration_data["exploration_range"][0] = min(min_seen, new_value)
            exploration_data["exploration_range"][1] = max(max_seen, new_value)
            
            # Update exploration bins
            self._update_exploration_bins(param_name, new_value)
            
            # Update range coverage
            if param_name in self.adjustable_parameters:
                param_data = self.adjustable_parameters[param_name]
                possible_range = param_data["max"] - param_data["min"]
                explored_range = exploration_data["exploration_range"][1] - exploration_data["exploration_range"][0]
                
                if possible_range > 0:
                    exploration_data["range_coverage"] = explored_range / possible_range
        
        # Update boundary tracking
        if param_name in self.enhanced_metrics["parameter_boundaries"]:
            boundary_data = self.enhanced_metrics["parameter_boundaries"][param_name]
            
            # Update min/max tested
            boundary_data["min_tested"] = min(boundary_data["min_tested"], new_value)
            boundary_data["max_tested"] = max(boundary_data["max_tested"], new_value)
            
            # Check if approaching boundaries
            if param_name in self.adjustable_parameters:
                param_data = self.adjustable_parameters[param_name]
                min_val = param_data["min"]
                max_val = param_data["max"]
                
                boundary_threshold = (param_data["max"] - param_data["min"]) * 0.1
                
                if (new_value - min_val) < boundary_threshold or (max_val - new_value) < boundary_threshold:
                    boundary_data["boundary_approaches"] += 1
        
        # Update responsiveness tracking
        if param_name in self.enhanced_metrics["parameter_responsiveness"]:
            responsiveness_data = self.enhanced_metrics["parameter_responsiveness"][param_name]
            
            # Add change to history
            change = {
                "timestamp": timestamp.isoformat(),
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
                "impact_measured": False  # Will be updated when impact is measured
            }
            
            responsiveness_data["last_changes"].append(change)
            
            # Keep a reasonable history
            if len(responsiveness_data["last_changes"]) > 10:
                responsiveness_data["last_changes"] = responsiveness_data["last_changes"][-10:]
    
    def _update_exploration_bins(self, param_name: str, value: float) -> None:
        """Update exploration bins for a parameter"""
        if param_name not in self.enhanced_metrics["parameter_exploration"]:
            return
        
        if param_name not in self.adjustable_parameters:
            return
        
        # Get parameter data
        param_data = self.adjustable_parameters[param_name]
        min_val = param_data["min"]
        max_val = param_data["max"]
        
        # Get exploration data
        exploration_data = self.enhanced_metrics["parameter_exploration"][param_name]
        bins = exploration_data["visits"]["bins"]
        
        # Calculate which bin this value falls into
        if max_val > min_val:
            bin_size = (max_val - min_val) / bins
            bin_index = min(bins - 1, int((value - min_val) / bin_size))
            
            # Increment bin count
            exploration_data["visits"]["counts"][bin_index] += 1
    
    async def _update_parameter_health(self, param_name: str) -> None:
        """Update health metrics for a parameter"""
        if param_name not in self.parameter_health:
            return
        
        health_data = self.parameter_health[param_name]
        
        # Update stability based on variance
        if param_name in self.enhanced_metrics["parameter_stability"]:
            stability_data = self.enhanced_metrics["parameter_stability"][param_name]
            
            if "variance" in stability_data and stability_data["variance"] > 0:
                # Calculate stability as inverse of normalized variance
                if param_name in self.adjustable_parameters:
                    param_data = self.adjustable_parameters[param_name]
                    param_range = param_data["max"] - param_data["min"]
                    
                    if param_range > 0:
                        normalized_variance = stability_data["variance"] / (param_range ** 2)
                        health_data["stability"] = max(0.0, 1.0 - (normalized_variance * 10))
        
        # Update exploration ratio
        if param_name in self.enhanced_metrics["parameter_exploration"]:
            exploration_data = self.enhanced_metrics["parameter_exploration"][param_name]
            
            if "range_coverage" in exploration_data:
                health_data["exploration_ratio"] = exploration_data["range_coverage"]
        
        # Update boundary violations
        if param_name in self.enhanced_metrics["parameter_boundaries"]:
            boundary_data = self.enhanced_metrics["parameter_boundaries"][param_name]
            
            health_data["boundary_violations"] = (
                boundary_data["min_violations"] + boundary_data["max_violations"]
            )
        
        # Update change frequency
        if param_name in self.enhanced_metrics["parameter_responsiveness"]:
            responsiveness_data = self.enhanced_metrics["parameter_responsiveness"][param_name]
            
            # Calculate frequency based on time between changes
            changes = responsiveness_data["last_changes"]
            if len(changes) >= 2:
                # Calculate average time between changes
                timestamps = [datetime.datetime.fromisoformat(change["timestamp"]) for change in changes]
                timestamps.sort()
                
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                
                if time_diffs:
                    avg_time_diff = sum(time_diffs) / len(time_diffs)
                    
                    # Convert to a frequency score (higher is more frequent)
                    # Assume 1 hour as a reference point
                    reference_time = 3600  # seconds in an hour
                    
                    if avg_time_diff > 0:
                        health_data["change_frequency"] = min(1.0, reference_time / avg_time_diff)
                    else:
                        health_data["change_frequency"] = 1.0
        
        # Update drift from default
        if param_name in self.adjustable_parameters:
            param_data = self.adjustable_parameters[param_name]
            
            param_range = param_data["max"] - param_data["min"]
            if param_range > 0:
                health_data["drift_from_default"] = abs(param_data["current"] - param_data["default"]) / param_range
        
        # Calculate overall health score
        # Good health: high stability, moderate exploration, low drift, low violations
        stability_weight = 0.4
        exploration_weight = 0.2
        drift_weight = 0.2
        boundary_weight = 0.2
        
        health_data["health_score"] = (
            health_data["stability"] * stability_weight +
            (0.5 - abs(health_data["exploration_ratio"] - 0.5)) * 2 * exploration_weight +  # Best at 0.5
            (1.0 - health_data["drift_from_default"]) * drift_weight +
            (1.0 - min(1.0, health_data["boundary_violations"] / 5.0)) * boundary_weight
        )
        
        # Update timestamp
        health_data["last_updated"] = datetime.datetime.now().isoformat()
    
    async def _validate_parameter_bounds(self) -> Dict[str, Any]:
        """Validate that all parameters are within their bounds"""
        if not hasattr(self, "adjustable_parameters"):
            return {"valid": False, "error": "No adjustable parameters available"}
        
        violations = []
        
        for param_name, param_data in self.adjustable_parameters.items():
            current = param_data["current"]
            min_val = param_data["min"]
            max_val = param_data["max"]
            
            if current < min_val:
                violations.append({
                    "parameter": param_name,
                    "current": current,
                    "min": min_val,
                    "violation": "below_minimum"
                })
            elif current > max_val:
                violations.append({
                    "parameter": param_name,
                    "current": current,
                    "max": max_val,
                    "violation": "above_maximum"
                })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "violation_count": len(violations)
        }
    
    async def _validate_parameter_coherence(self) -> Dict[str, Any]:
        """Validate coherence between interdependent parameters"""
        if not hasattr(self, "adjustable_parameters") or not hasattr(self, "parameter_relationships"):
            return {"valid": False, "error": "Parameter data not available"}
        
        incoherencies = []
        
        # Check specific coherence rules
        if all(p in self.adjustable_parameters for p in ["memory_to_emotion_influence", "emotion_to_memory_influence"]):
            mem_to_emo = self.adjustable_parameters["memory_to_emotion_influence"]["current"]
            emo_to_mem = self.adjustable_parameters["emotion_to_memory_influence"]["current"]
            
            # Check bidirectional influence balance
            total_influence = mem_to_emo + emo_to_mem
            if total_influence > 1.0:
                incoherencies.append({
                    "type": "bidirectional_influence",
                    "parameters": ["memory_to_emotion_influence", "emotion_to_memory_influence"],
                    "values": [mem_to_emo, emo_to_mem],
                    "issue": "Combined influence exceeds 1.0",
                    "recommendation": "Reduce one or both influences"
                })
        
        # Check cross-user parameters
        if all(p in self.adjustable_parameters for p in ["cross_user_enabled", "cross_user_sharing_threshold"]):
            enabled = self.adjustable_parameters["cross_user_enabled"]["current"]
            threshold = self.adjustable_parameters["cross_user_sharing_threshold"]["current"]
            
            # If disabled but threshold is low (easy sharing)
            if enabled == 0 and threshold < 0.5:
                incoherencies.append({
                    "type": "contradictory_settings",
                    "parameters": ["cross_user_enabled", "cross_user_sharing_threshold"],
                    "values": [enabled, threshold],
                    "issue": "Cross-user sharing disabled but threshold is low",
                    "recommendation": "Either enable sharing or increase threshold"
                })
        
        # Check processing thresholds
        if all(p in self.adjustable_parameters for p in ["parallel_processing_threshold", "distributed_processing_threshold"]):
            parallel = self.adjustable_parameters["parallel_processing_threshold"]["current"]
            distributed = self.adjustable_parameters["distributed_processing_threshold"]["current"]
            
            # Check threshold ordering
            if parallel >= distributed:
                incoherencies.append({
                    "type": "threshold_ordering",
                    "parameters": ["parallel_processing_threshold", "distributed_processing_threshold"],
                    "values": [parallel, distributed],
                    "issue": "Parallel threshold should be lower than distributed threshold",
                    "recommendation": "Increase distributed or decrease parallel threshold"
                })
        
        return {
            "valid": len(incoherencies) == 0,
            "incoherencies": incoherencies,
            "incoherence_count": len(incoherencies)
        }
    
    async def _validate_cross_module_consistency(self) -> Dict[str, Any]:
        """Validate consistency between brain parameters and module state"""
        if not self.brain:
            return {"valid": False, "error": "Brain reference not available"}
        
        inconsistencies = []
        
        # Check brain attribute consistency
        if hasattr(self, "adjustable_parameters"):
            for param_name, param_data in self.adjustable_parameters.items():
                if hasattr(self.brain, param_name):
                    brain_value = getattr(self.brain, param_name)
                    config_value = param_data["current"]
                    
                    # Handle boolean conversions
                    if param_data["min"] == 0 and param_data["max"] == 1 and param_data["step"] == 1:
                        # Convert brain boolean to 0/1
                        brain_value = 1 if brain_value else 0
                    
                    # Check if values match
                    if brain_value != config_value:
                        inconsistencies.append({
                            "parameter": param_name,
                            "brain_value": brain_value,
                            "config_value": config_value,
                            "recommendation": "Synchronize values"
                        })
        
        # Check emotional core parameters
        if hasattr(self.brain, "emotional_core"):
            emotional_core = self.brain.emotional_core
            
            # Check decay rate
            if hasattr(emotional_core, "decay_rate") and "emotional_decay_rate" in self.adjustable_parameters:
                core_value = emotional_core.decay_rate
                config_value = self.adjustable_parameters["emotional_decay_rate"]["current"]
                
                if core_value != config_value:
                    inconsistencies.append({
                        "parameter": "emotional_decay_rate",
                        "module": "emotional_core",
                        "module_value": core_value,
                        "config_value": config_value,
                        "recommendation": "Synchronize values"
                    })
        
        # Check memory parameters
        if hasattr(self.brain, "memory_orchestrator"):
            memory_orchestrator = self.brain.memory_orchestrator
            
            # Check weights
            if hasattr(memory_orchestrator, "recency_weight") and "memory_recency_weight" in self.adjustable_parameters:
                module_value = memory_orchestrator.recency_weight
                config_value = self.adjustable_parameters["memory_recency_weight"]["current"]
                
                if module_value != config_value:
                    inconsistencies.append({
                        "parameter": "memory_recency_weight",
                        "module": "memory_orchestrator",
                        "module_value": module_value,
                        "config_value": config_value,
                        "recommendation": "Synchronize values"
                    })
        
        return {
            "valid": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "inconsistency_count": len(inconsistencies)
        }
    
    async def record_parameter_impact(self,
                                  param_name: str,
                                  metrics: Dict[str, float],
                                  compare_to_baseline: bool = True) -> Dict[str, Any]:
        """
        Record the impact of a parameter change on system metrics
        
        Args:
            param_name: Name of the parameter
            metrics: Performance metrics to record
            compare_to_baseline: Whether to compare to baseline
            
        Returns:
            Impact assessment
        """
        if param_name not in self.enhanced_metrics["parameter_responsiveness"]:
            return {"success": False, "error": f"Parameter '{param_name}' not being tracked"}
        
        responsiveness_data = self.enhanced_metrics["parameter_responsiveness"][param_name]
        
        # Find the most recent change
        last_changes = responsiveness_data["last_changes"]
        if not last_changes:
            return {"success": False, "error": "No recent changes to evaluate impact"}
        
        last_change = last_changes[-1]
        
        # Skip if impact already measured
        if last_change.get("impact_measured", False):
            return {"success": False, "error": "Impact already measured for the most recent change"}
        
        # Calculate impact
        impact_assessment = {
            "change_timestamp": last_change["timestamp"],
            "old_value": last_change["old_value"],
            "new_value": last_change["new_value"],
            "metrics": metrics,
            "impact_scores": {}
        }
        
        # Compare to baseline if requested
        if compare_to_baseline and param_name in self.base_config_manager.param_performance_impact:
            baseline = self.base_config_manager.param_performance_impact[param_name].get("baseline", {})
            
            # Calculate impact for each metric
            for metric_name, metric_value in metrics.items():
                if metric_name in baseline:
                    baseline_value = baseline[metric_name]
                    
                    # Calculate impact as percentage change
                    if baseline_value != 0:
                        impact = (metric_value - baseline_value) / abs(baseline_value)
                    else:
                        impact = 0.0 if metric_value == 0 else 1.0
                    
                    impact_assessment["impact_scores"][metric_name] = impact
        
        # Calculate overall impact
        total_impact = sum(impact_assessment["impact_scores"].values())
        avg_impact = total_impact / len(impact_assessment["impact_scores"]) if impact_assessment["impact_scores"] else 0.0
        
        impact_assessment["overall_impact"] = avg_impact
        
        # Update the change record
        last_change["impact_measured"] = True
        last_change["impact"] = avg_impact
        
        # Add to impact history
        responsiveness_data["impact_history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "change": {
                "old_value": last_change["old_value"],
                "new_value": last_change["new_value"]
            },
            "impact": avg_impact,
            "metrics": metrics
        })
        
        # Keep history to a reasonable size
        if len(responsiveness_data["impact_history"]) > 20:
            responsiveness_data["impact_history"] = responsiveness_data["impact_history"][-20:]
        
        # Sync with base manager if available
        if (self.base_config_manager and hasattr(self.base_config_manager, "param_performance_impact") and
            param_name in self.base_config_manager.param_performance_impact):
            
            if "history" not in self.base_config_manager.param_performance_impact[param_name]:
                self.base_config_manager.param_performance_impact[param_name]["history"] = []
            
            # Add to base manager's history
            self.base_config_manager.param_performance_impact[param_name]["history"].append({
                "direction": 1 if last_change["new_value"] > last_change["old_value"] else -1,
                "impact": avg_impact
            })
            
            # Keep history to a reasonable size
            if len(self.base_config_manager.param_performance_impact[param_name]["history"]) > 20:
                self.base_config_manager.param_performance_impact[param_name]["history"] = (
                    self.base_config_manager.param_performance_impact[param_name]["history"][-20:]
                )
        
        return {
            "success": True,
            "param_name": param_name,
            "impact_assessment": impact_assessment
        }
    
    async def process_user_feedback(self,
                                feedback_type: str,
                                feedback_text: str,
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user feedback to influence configuration
        
        Args:
            feedback_type: Type of feedback ("positive", "negative", "specific")
            feedback_text: Text of user feedback
            context: Additional context information
            
        Returns:
            Processing results
        """
        context = context or {}
        user_id = context.get("user_id", "unknown")
        
        # Initialize result
        result = {
            "feedback_type": feedback_type,
            "parameters_affected": [],
            "user_specific_updates": []
        }
        
        # Convert feedback to lower case for easier matching
        feedback_lower = feedback_text.lower()
        
        # Initialize user-specific parameters if needed
        if user_id not in self.user_specific_parameters:
            self.user_specific_parameters[user_id] = {}
        
        # Process feedback based on type
        if feedback_type == "positive":
            # For positive feedback, slightly amplify the current trajectory
            result.update(await self._process_positive_feedback(feedback_lower, context))
        elif feedback_type == "negative":
            # For negative feedback, reverse recent changes or revert to defaults
            result.update(await self._process_negative_feedback(feedback_lower, context))
        elif feedback_type == "specific":
            # For specific feedback, try to identify targeted parameters
            result.update(await self._process_specific_feedback(feedback_lower, context))
        
        # Add to user feedback impact
        if hasattr(self, "user_feedback_impact"):
            # Record the feedback
            if feedback_type not in self.user_feedback_impact:
                self.user_feedback_impact[feedback_type] = []
            
            self.user_feedback_impact[feedback_type].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "text": feedback_text,
                "parameters_affected": result["parameters_affected"],
                "user_id": user_id
            })
        
        # Create config snapshot
        await self._create_config_snapshot(f"user_feedback_{feedback_type}")
        
        return result
    
    async def _process_positive_feedback(self, 
                                     feedback_text: str, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process positive feedback"""
        result = {"parameters_affected": []}
        
        # Check for specific areas mentioned in the feedback
        mentioned_areas = []
        
        # Memory-related feedback
        memory_terms = ["memory", "remember", "recall", "forget", "retrieval"]
        if any(term in feedback_text for term in memory_terms):
            mentioned_areas.append("memory")
        
        # Emotion-related feedback
        emotion_terms = ["emotion", "feel", "felt", "emotional", "mood"]
        if any(term in feedback_text for term in emotion_terms):
            mentioned_areas.append("emotion")
        
        # Processing-related feedback
        processing_terms = ["speed", "fast", "slow", "quick", "responsive"]
        if any(term in feedback_text for term in processing_terms):
            mentioned_areas.append("performance")
        
        # Intelligence-related feedback
        intelligence_terms = ["smart", "intelligent", "clever", "reasoning", "understand"]
        if any(term in feedback_text for term in intelligence_terms):
            mentioned_areas.append("reasoning")
        
        # Social-related feedback
        social_terms = ["conversation", "talk", "chat", "communicate", "social"]
        if any(term in feedback_text for term in social_terms):
            mentioned_areas.append("social")
        
        # No specific areas mentioned, look at recent changes
        if not mentioned_areas and self.config_states:
            recent_changes = []
            
            # Get parameters changed in the last few states
            recent_states = self.config_states[-3:] if len(self.config_states) >= 3 else self.config_states
            if len(recent_states) >= 2:
                for i in range(len(recent_states) - 1):
                    curr_state = recent_states[i+1]
                    prev_state = recent_states[i]
                    
                    # Find changed parameters
                    for param, value in curr_state["parameters"].items():
                        if param in prev_state["parameters"] and value != prev_state["parameters"][param]:
                            recent_changes.append(param)
            
            # If we have recent changes, focus on those
            if recent_changes:
                # Try to reinforce those changes (continue in same direction)
                for param_name in recent_changes:
                    if param_name in self.adjustable_parameters:
                        param_data = self.adjustable_parameters[param_name]
                        
                        # Get the current value
                        current = param_data["current"]
                        
                        # Find the most recent change direction
                        if len(self.enhanced_metrics["parameter_responsiveness"][param_name]["last_changes"]) > 0:
                            last_change = self.enhanced_metrics["parameter_responsiveness"][param_name]["last_changes"][-1]
                            change_direction = 1 if last_change["new_value"] > last_change["old_value"] else -1
                            
                            # Continue in the same direction
                            step = param_data["step"] * change_direction * 0.5  # 50% of normal step
                            new_value = current + step
                            
                            # Update the parameter
                            update_result = await self.update_parameter(
                                param_name, 
                                new_value, 
                                "positive_feedback_reinforcement"
                            )
                            
                            if update_result["changed"]:
                                result["parameters_affected"].append({
                                    "param_name": param_name,
                                    "old_value": update_result["old_value"],
                                    "new_value": update_result["new_value"],
                                    "reason": "Reinforced recent change due to positive feedback"
                                })
        
        # Process based on mentioned areas
        for area in mentioned_areas:
            if area == "memory":
                memory_params = [p for p in self.adjustable_parameters if "memory" in p and p in self.parameter_health]
                
                # Find the memory parameter with lowest health to improve
                if memory_params:
                    param_health = [(p, self.parameter_health[p]["health_score"]) for p in memory_params]
                    param_health.sort(key=lambda x: x[1])
                    
                    # Take the parameter with lowest health
                    param_name = param_health[0][0]
                    param_data = self.adjustable_parameters[param_name]
                    
                    # Adjust parameter (move toward default if far from it)
                    current = param_data["current"]
                    default = param_data["default"]
                    step = param_data["step"]
                    
                    if abs(current - default) > step:
                        # Move toward default
                        direction = 1 if default > current else -1
                        new_value = current + (step * direction)
                        
                        # Update the parameter
                        update_result = await self.update_parameter(
                            param_name, 
                            new_value, 
                            "positive_feedback_memory"
                        )
                        
                        if update_result["changed"]:
                            result["parameters_affected"].append({
                                "param_name": param_name,
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Adjusted memory parameter due to positive feedback"
                            })
            
            elif area == "emotion":
                # If positive about emotions, slightly increase emotional influence
                if "emotion_to_memory_influence" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["emotion_to_memory_influence"]
                    current = param_data["current"]
                    step = param_data["step"]
                    
                    # Increase emotional influence slightly
                    new_value = min(param_data["max"], current + (step * 0.5))
                    
                    # Update the parameter
                    update_result = await self.update_parameter(
                        "emotion_to_memory_influence", 
                        new_value, 
                        "positive_feedback_emotion"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": "emotion_to_memory_influence",
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Increased emotional influence due to positive feedback"
                        })
            
            elif area == "performance":
                # If positive about performance, optimize for speed
                if "distributed_processing_threshold" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["distributed_processing_threshold"]
                    current = param_data["current"]
                    step = param_data["step"]
                    
                    # Lower distributed threshold (use it more often)
                    new_value = max(param_data["min"], current - (step * 0.5))
                    
                    # Update the parameter
                    update_result = await self.update_parameter(
                        "distributed_processing_threshold", 
                        new_value, 
                        "positive_feedback_performance"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": "distributed_processing_threshold",
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Optimized for performance due to positive feedback"
                        })
            
            elif area == "reasoning":
                # If positive about reasoning, increase reasoning depth
                if "reasoning_depth" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["reasoning_depth"]
                    current = param_data["current"]
                    
                    # Increase reasoning depth if not at max
                    if current < param_data["max"]:
                        new_value = current + param_data["step"]
                        
                        # Update the parameter
                        update_result = await self.update_parameter(
                            "reasoning_depth", 
                            new_value, 
                            "positive_feedback_reasoning"
                        )
                        
                        if update_result["changed"]:
                            result["parameters_affected"].append({
                                "param_name": "reasoning_depth",
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Increased reasoning depth due to positive feedback"
                            })
            
            elif area == "social":
                # If positive about social aspects, enable cross-user experiences
                if "cross_user_enabled" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["cross_user_enabled"]
                    current = param_data["current"]
                    
                    # Enable cross-user experiences if not already
                    if current < 1:
                        # Update the parameter
                        update_result = await self.update_parameter(
                            "cross_user_enabled", 
                            1, 
                            "positive_feedback_social"
                        )
                        
                        if update_result["changed"]:
                            result["parameters_affected"].append({
                                "param_name": "cross_user_enabled",
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Enabled cross-user experiences due to positive feedback"
                            })
        
        # If we haven't affected any parameters yet, make a small general improvement
        if not result["parameters_affected"]:
            # Choose a random parameter to slightly improve
            adjustable_params = list(self.adjustable_parameters.keys())
            if adjustable_params:
                # Select a random parameter with health issues
                unhealthy_params = [p for p in adjustable_params if p in self.parameter_health and
                                  self.parameter_health[p]["health_score"] < 0.8]
                
                param_name = random.choice(unhealthy_params) if unhealthy_params else random.choice(adjustable_params)
                param_data = self.adjustable_parameters[param_name]
                
                # Move toward default
                current = param_data["current"]
                default = param_data["default"]
                step = param_data["step"] * 0.5  # Small step
                
                # Determine direction
                direction = 1 if default > current else -1
                new_value = current + (step * direction)
                
                # Update the parameter
                update_result = await self.update_parameter(
                    param_name, 
                    new_value, 
                    "positive_feedback_general"
                )
                
                if update_result["changed"]:
                    result["parameters_affected"].append({
                        "param_name": param_name,
                        "old_value": update_result["old_value"],
                        "new_value": update_result["new_value"],
                        "reason": "General improvement due to positive feedback"
                    })
        
        return result
    
    async def _process_negative_feedback(self, 
                                     feedback_text: str, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process negative feedback"""
        result = {"parameters_affected": []}
        
        # Check for specific areas mentioned in the feedback
        mentioned_areas = []
        
        # Memory-related feedback
        memory_terms = ["memory", "remember", "recall", "forget", "retrieval"]
        if any(term in feedback_text for term in memory_terms):
            mentioned_areas.append("memory")
        
        # Emotion-related feedback
        emotion_terms = ["emotion", "feel", "felt", "emotional", "mood"]
        if any(term in feedback_text for term in emotion_terms):
            mentioned_areas.append("emotion")
        
        # Processing-related feedback
        processing_terms = ["speed", "fast", "slow", "quick", "responsive"]
        if any(term in feedback_text for term in processing_terms):
            mentioned_areas.append("performance")
        
        # Intelligence-related feedback
        intelligence_terms = ["smart", "intelligent", "clever", "reasoning", "understand"]
        if any(term in feedback_text for term in intelligence_terms):
            mentioned_areas.append("reasoning")
        
        # Social-related feedback
        social_terms = ["conversation", "talk", "chat", "communicate", "social"]
        if any(term in feedback_text for term in social_terms):
            mentioned_areas.append("social")
        
        # Look at recent changes to revert them
        if self.config_states:
            recent_changes = []
            
            # Get parameters changed in the last few states
            recent_states = self.config_states[-3:] if len(self.config_states) >= 3 else self.config_states
            if len(recent_states) >= 2:
                for i in range(len(recent_states) - 1):
                    curr_state = recent_states[i+1]
                    prev_state = recent_states[i]
                    
                    # Find changed parameters
                    for param, value in curr_state["parameters"].items():
                        if param in prev_state["parameters"] and value != prev_state["parameters"][param]:
                            recent_changes.append({
                                "param_name": param,
                                "old_value": prev_state["parameters"][param],
                                "new_value": value
                            })
            
            # If we have recent changes, check if they were in areas of concern
            if recent_changes and mentioned_areas:
                for change in recent_changes:
                    param_name = change["param_name"]
                    
                    # Check if parameter belongs to a mentioned area
                    param_area = None
                    if "memory" in param_name:
                        param_area = "memory"
                    elif "emotion" in param_name:
                        param_area = "emotion"
                    elif "processing" in param_name:
                        param_area = "performance"
                    elif "reasoning" in param_name:
                        param_area = "reasoning"
                    elif "cross_user" in param_name:
                        param_area = "social"
                    
                    if param_area in mentioned_areas:
                        # Revert this change
                        update_result = await self.update_parameter(
                            param_name, 
                            change["old_value"], 
                            f"negative_feedback_revert_{param_area}"
                        )
                        
                        if update_result["changed"]:
                            result["parameters_affected"].append({
                                "param_name": param_name,
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": f"Reverted recent change to {param_area} due to negative feedback"
                            })
        
        # Process based on mentioned areas
        for area in mentioned_areas:
            # If we already fixed parameters in this area, skip
            if any(p.get("reason", "").endswith(f"feedback_revert_{area}") for p in result["parameters_affected"]):
                continue
                
            if area == "memory":
                memory_params = [p for p in self.adjustable_parameters if "memory" in p and p in self.parameter_health]
                
                if "memory_to_emotion_influence" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["memory_to_emotion_influence"]
                    current = param_data["current"]
                    default = param_data["default"]
                    
                    # Reset to default if significantly different
                    if abs(current - default) > param_data["step"]:
                        update_result = await self.update_parameter(
                            "memory_to_emotion_influence", 
                            default, 
                            "negative_feedback_memory_reset"
                        )
                        
                        if update_result["changed"]:
                            result["parameters_affected"].append({
                                "param_name": "memory_to_emotion_influence",
                                "old_value": update_result["old_value"],
                                "new_value": update_result["new_value"],
                                "reason": "Reset memory-emotion influence due to negative feedback"
                            })
            
            elif area == "emotion":
                if "emotional_expression_threshold" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["emotional_expression_threshold"]
                    current = param_data["current"]
                    
                    # Increase threshold (less emotional expression)
                    new_value = min(param_data["max"], current + param_data["step"])
                    
                    update_result = await self.update_parameter(
                        "emotional_expression_threshold", 
                        new_value, 
                        "negative_feedback_emotion"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": "emotional_expression_threshold",
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Decreased emotional expression due to negative feedback"
                        })
            
            elif area == "performance":
                if "parallel_processing_threshold" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["parallel_processing_threshold"]
                    current = param_data["current"]
                    default = param_data["default"]
                    
                    # Reset to default
                    update_result = await self.update_parameter(
                        "parallel_processing_threshold", 
                        default, 
                        "negative_feedback_performance"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": "parallel_processing_threshold",
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Reset processing threshold due to negative feedback"
                        })
            
            elif area == "reasoning":
                if "reasoning_depth" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["reasoning_depth"]
                    default = param_data["default"]
                    
                    # Reset to default
                    update_result = await self.update_parameter(
                        "reasoning_depth", 
                        default, 
                        "negative_feedback_reasoning"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": "reasoning_depth",
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Reset reasoning depth due to negative feedback"
                        })
            
            elif area == "social":
                if "cross_user_sharing_threshold" in self.adjustable_parameters:
                    param_data = self.adjustable_parameters["cross_user_sharing_threshold"]
                    
                    # Make sharing more restrictive
                    new_value = min(param_data["max"], param_data["current"] + param_data["step"])
                    
                    update_result = await self.update_parameter(
                        "cross_user_sharing_threshold", 
                        new_value, 
                        "negative_feedback_social"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": "cross_user_sharing_threshold",
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Made cross-user sharing more restrictive due to negative feedback"
                        })
        
        # If no parameters were affected, try resetting to defaults for unhealthy parameters
        if not result["parameters_affected"]:
            unhealthy_params = [p for p in self.adjustable_parameters if 
                              p in self.parameter_health and
                              self.parameter_health[p]["health_score"] < 0.7]
            
            if unhealthy_params:
                # Take top 2 most unhealthy parameters
                sorted_params = sorted(unhealthy_params, 
                                     key=lambda p: self.parameter_health[p]["health_score"])
                
                for param_name in sorted_params[:2]:
                    param_data = self.adjustable_parameters[param_name]
                    default = param_data["default"]
                    
                    # Reset to default
                    update_result = await self.update_parameter(
                        param_name, 
                        default, 
                        "negative_feedback_reset_unhealthy"
                    )
                    
                    if update_result["changed"]:
                        result["parameters_affected"].append({
                            "param_name": param_name,
                            "old_value": update_result["old_value"],
                            "new_value": update_result["new_value"],
                            "reason": "Reset unhealthy parameter due to negative feedback"
                        })
        
        return result


# Create a unified configuration manager that combines both classes
class UnifiedConfigManager:
    """
    Unified configuration manager that integrates both the base SelfConfigManager 
    and the EnhancedConfigManager functionality
    """
    
    def __init__(self, brain):
        """
        Initialize the unified configuration manager
        
        Args:
            brain: Reference to the NyxBrain instance
        """
        self.brain = brain
        
        # Initialize base manager
        self.base_manager = SelfConfigManager(brain)
        
        # Initialize enhanced manager with reference to base manager
        self.enhanced_manager = EnhancedConfigManager(brain, self.base_manager)
        
        logger.info("Unified configuration manager initialized")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize both configuration managers
        
        Returns:
            Initialization results
        """
        try:
            # Enable base manager
            base_result = await self.base_manager.enable()
            
            # Initialize enhanced manager
            enhanced_result = await self.enhanced_manager.initialize()
            
            return {
                "base_manager": base_result,
                "enhanced_manager": enhanced_result,
                "status": "initialized"
            }
        except Exception as e:
            logger.error(f"Error initializing configuration managers: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def update_parameter(self, 
                           param_name: str, 
                           new_value: float, 
                           reason: str = None) -> Dict[str, Any]:
        """
        Update a parameter with tracking in both managers
        
        Args:
            param_name: Name of the parameter to update
            new_value: New value for the parameter
            reason: Reason for the update
            
        Returns:
            Update result
        """
        try:
            # Update via enhanced manager (which will sync with base manager)
            return await self.enhanced_manager.update_parameter(param_name, new_value, reason)
        except Exception as e:
            logger.error(f"Error updating parameter {param_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "param_name": param_name
            }
    
    async def evaluate_and_adjust_parameters(self) -> Dict[str, Any]:
        """
        Evaluate and adjust parameters using base manager's logic
        
        Returns:
            Results of parameter adjustments
        """
        try:
            return await self.base_manager.evaluate_and_adjust_parameters()
        except Exception as e:
            logger.error(f"Error evaluating and adjusting parameters: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def process_user_feedback(self,
                                feedback_type: str,
                                feedback_text: str,
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user feedback to influence configuration
        
        Args:
            feedback_type: Type of feedback ("positive", "negative", "specific")
            feedback_text: Text of user feedback
            context: Additional context information
            
        Returns:
            Processing results
        """
        try:
            return await self.enhanced_manager.process_user_feedback(feedback_type, feedback_text, context)
        except Exception as e:
            logger.error(f"Error processing user feedback: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_parameter_health_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive health report for all parameters
        
        Returns:
            Health report data
        """
        try:
            return await self.enhanced_manager.get_parameter_health_report()
        except Exception as e:
            logger.error(f"Error getting parameter health report: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_self_optimization(self) -> Dict[str, Any]:
        """
        Run an automatic self-optimization process to improve system health
        
        Returns:
            Optimization results
        """
        try:
            return await self.enhanced_manager.run_self_optimization()
        except Exception as e:
            logger.error(f"Error running self-optimization: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def experiment_with_parameter(self, 
                                    param_name: str, 
                                    value_range: List[float] = None,
                                    duration: int = 5) -> Dict[str, Any]:
        """
        Run an experiment with a parameter
        
        Args:
            param_name: Parameter to experiment with
            value_range: Range of values to try, if None, use default range
            duration: Number of interactions to test each value for
            
        Returns:
            Experiment setup results
        """
        try:
            return await self.enhanced_manager.experiment_with_parameter(param_name, value_range, duration)
        except Exception as e:
            logger.error(f"Error experimenting with parameter {param_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "param_name": param_name
            }
    
    async def update_experiment(self, 
                            experiment_id: str, 
                            metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Update an experiment with new metrics and possibly advance to next value
        
        Args:
            experiment_id: ID of the experiment
            metrics: Current performance metrics
            
        Returns:
            Update results
        """
        try:
            return await self.enhanced_manager.update_experiment(experiment_id, metrics)
        except Exception as e:
            logger.error(f"Error updating experiment {experiment_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "experiment_id": experiment_id
            }
    
    async def recommend_parameters_for_user(self, user_id: str) -> Dict[str, Any]:
        """
        Generate parameter recommendations for a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            Parameter recommendations
        """
        try:
            return await self.enhanced_manager.recommend_parameters_for_user(user_id)
        except Exception as e:
            logger.error(f"Error recommending parameters for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    async def record_parameter_impact(self,
                                 param_name: str,
                                 metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Record the impact of a parameter change on system metrics
        
        Args:
            param_name: Name of the parameter
            metrics: Performance metrics to record
            
        Returns:
            Impact assessment
        """
        try:
            return await self.enhanced_manager.record_parameter_impact(param_name, metrics)
        except Exception as e:
            logger.error(f"Error recording parameter impact for {param_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "param_name": param_name
            }
    
    # Convenience methods to access managers directly
    def get_base_manager(self) -> SelfConfigManager:
        """Get the base configuration manager"""
        return self.base_manager
    
    def get_enhanced_manager(self) -> EnhancedConfigManager:
        """Get the enhanced configuration manager"""
        return self.enhanced_manager
