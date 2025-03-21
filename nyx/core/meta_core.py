# nyx/core/meta_core.py

import asyncio
import datetime
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set

import numpy as np

logger = logging.getLogger(__name__)

class MetaCore:
    """
    Core system for meta-learning and meta-cognition in Nyx.
    
    Integrates functionality from the meta-learning and meta-cognition systems
    directly into the core of Nyx, enabling higher-order cognitive processes
    and systematic self-improvement.
    
    This system provides:
    1. Meta-learning - learning how to learn from interactions
    2. Meta-cognition - monitoring and regulating cognitive processes
    3. Resource optimization - dynamic allocation of cognitive resources
    4. Performance evaluation - systematic assessment of cognitive effectiveness
    5. Strategic adaptation - evolving cognitive strategies based on feedback
    6. Attention management - managing focus on critical systems or processes
    7. Meta-parameter optimization - improving the meta-cognitive process itself
    """
    
    def __init__(self):
        # Meta-learning components
        self.feature_importance = {}
        self.algorithm_performance = {}
        self.learning_cycles = 0
        self.feature_history = []
        
        # Meta-cognitive components
        self.monitored_systems = {
            "memory": {"performance": {}, "parameters": {}, "bottlenecks": []},
            "emotion": {"performance": {}, "parameters": {}, "bottlenecks": []},
            "reasoning": {"performance": {}, "parameters": {}, "bottlenecks": []},
            "reflection": {"performance": {}, "parameters": {}, "bottlenecks": []},
            "adaptation": {"performance": {}, "parameters": {}, "bottlenecks": []}
        }
        
        # Resource allocation
        self.resource_allocation = {
            "memory": 0.2,
            "emotion": 0.2,
            "reasoning": 0.2,
            "reflection": 0.15,
            "adaptation": 0.15,
            "meta": 0.1  # Reserve resources for meta processes themselves
        }
        
        # Performance tracking
        self.performance_history = {system: [] for system in self.monitored_systems}
        self.cognitive_processes = {}
        self.mental_models = {}
        
        # Meta-cognitive insights and reflections
        self.insights = []
        self.reflections = []
        self.improvement_plans = []
        
        # Error logging
        self.error_logs = []
        
        # Attention focus
        self.attention_focus = None
        
        # Configuration
        self.meta_parameters = {
            # Learning parameters
            "learning_rate": 0.1,
            "exploration_rate": 0.2,
            "convergence_threshold": 0.05,
            "min_samples_required": 5,
            
            # Cognitive evaluation parameters
            "reflection_frequency": 10,  # Interactions between reflections
            "evaluation_interval": 5,    # Cycles between evaluations
            "confidence_threshold": 0.7, # Threshold for accepting a new strategy
            "resource_flexibility": 0.3, # How much to vary resource allocation
            "bottleneck_severity_threshold": 0.7, # Threshold for critical bottlenecks
            
            # Parameter optimization
            "parameter_optimization_interval": 50, # Cycles between meta-parameter optimizations
            "parameter_adjustment_factor": 0.2,    # How much to adjust parameters
            
            # Attention management
            "attention_shift_threshold": 0.8,      # Threshold to shift attention 
            "attention_default_duration": 5        # Default cycles for attention focus
        }
        
        # System references (will be set in initialize)
        self.system_references = {}
        
        # Internal state tracking
        self.initialized = False
        self.last_evaluation_time = None
        self.last_reflection_time = None
        self.last_parameter_optimization_cycle = 0
        self.cognitive_cycle_count = 0
        
        # Next ID counters for tracking
        self.next_process_id = 1
        self.next_model_id = 1
        self.next_insight_id = 1
        self.next_reflection_id = 1
        
        # System metrics
        self.system_metrics = {
            "start_time": datetime.datetime.now(),
            "total_runtime": 0.0,
            "cycles_completed": 0,
            "total_processes": 0,
            "resource_usage": {
                "cpu": 0.0,
                "memory": 0.0,
                "io": 0.0
            },
            "average_cycle_time": 0.0,
            "error_rate": 0.0
        }
    
    async def initialize(self, system_references: Dict[str, Any]) -> None:
        """Initialize the meta core with references to other core systems"""
        if self.initialized:
            return
            
        logger.info("Initializing MetaCore")
        
        # Store system references
        self.system_references = system_references
        
        # Extract initial parameters from all systems
        for system_name, system in system_references.items():
            if system_name in self.monitored_systems:
                self.monitored_systems[system_name]["parameters"] = await self._extract_system_parameters(system)
        
        # Register core cognitive processes
        for system_name, system in system_references.items():
            if system_name in self.monitored_systems:
                await self.register_cognitive_process(
                    name=f"{system_name}_process",
                    type=system_name,
                    priority=0.5,
                    resource_allocation=self.resource_allocation.get(system_name, 0.1)
                )
        
        # Create initial mental models for core domains
        await self.create_mental_model("emotional_model", "emotion", confidence=0.6)
        await self.create_mental_model("memory_model", "memory", confidence=0.6)
        await self.create_mental_model("reasoning_model", "reasoning", confidence=0.6)
        await self.create_mental_model("user_model", "user", confidence=0.4)
        
        # Set timestamps for evaluation scheduling
        self.last_evaluation_time = datetime.datetime.now()
        self.last_reflection_time = datetime.datetime.now()
        
        # Conduct initial self-assessment
        await self._conduct_initial_assessment()
        
        self.initialized = True
        logger.info("MetaCore initialized")
    
    #--------------------------------------------------------------------------
    # Core Meta-Cognitive Cycle
    #--------------------------------------------------------------------------
    
    async def cognitive_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete meta-cognitive cycle.
        
        Args:
            context: Context information for the cognitive cycle
            
        Returns:
            Results of the cognitive cycle
        """
        cycle_start = time.time()
        self.cognitive_cycle_count += 1
        
        # 1. Check attention focus
        attention_shift = await self._check_attention_focus(context)
        
        # 2. Collect performance metrics from all systems
        try:
            performance_data = await self._collect_performance_metrics()
        except Exception as e:
            await self.log_cognitive_error("metric_collection_failure", 
                                        f"Failed to collect performance metrics: {str(e)}", 
                                        severity=0.7, 
                                        context={"cycle": self.cognitive_cycle_count})
            performance_data = {}
        
        # 3. Update performance history
        self._update_performance_history(performance_data)
        
        # 4. Check for inefficient dependencies
        dependency_issues = self._identify_inefficient_dependencies()
        if dependency_issues:
            # Log the most critical dependency issue
            most_critical = dependency_issues[0]
            await self.log_cognitive_error("inefficient_dependency",
                                        most_critical["recommendation"],
                                        severity=0.6,
                                        context=most_critical)
        
        # 5. Check if evaluation is needed
        evaluation_results = None
        if self._should_evaluate():
            try:
                evaluation_results = await self.evaluate_cognition()
                self.last_evaluation_time = datetime.datetime.now()
            except Exception as e:
                await self.log_cognitive_error("evaluation_failure", 
                                           f"Failed to evaluate cognition: {str(e)}", 
                                           severity=0.8, 
                                           context={"cycle": self.cognitive_cycle_count})
        
        # 6. Check if reflection is needed
        reflection_results = None
        if self._should_reflect():
            try:
                reflection_results = await self._conduct_reflection()
                self.last_reflection_time = datetime.datetime.now()
            except Exception as e:
                await self.log_cognitive_error("reflection_failure", 
                                           f"Failed to conduct reflection: {str(e)}", 
                                           severity=0.7, 
                                           context={"cycle": self.cognitive_cycle_count})
        
        # 7. Update resource allocation if needed
        resource_changes = {}
        if evaluation_results:
            resource_changes = evaluation_results.get("resource_changes", {})
        
        # 8. Check if meta-parameter optimization is needed
        meta_parameter_changes = None
        if self._should_optimize_parameters():
            try:
                meta_parameter_changes = await self.improve_meta_parameters()
                self.last_parameter_optimization_cycle = self.cognitive_cycle_count
            except Exception as e:
                await self.log_cognitive_error("parameter_optimization_failure", 
                                           f"Failed to optimize meta-parameters: {str(e)}", 
                                           severity=0.6, 
                                           context={"cycle": self.cognitive_cycle_count})
        
        # 9. Update system metrics
        self._update_system_metrics(cycle_start)
        
        # Return cycle results
        return {
            "cycle": self.cognitive_cycle_count,
            "performance_data": performance_data,
            "attention_shift": attention_shift,
            "evaluation_results": evaluation_results,
            "reflection_results": reflection_results,
            "resource_changes": resource_changes,
            "meta_parameter_changes": meta_parameter_changes,
            "dependency_issues": dependency_issues[:3] if dependency_issues else [],  # Return top 3 issues only
            "system_metrics": self.system_metrics
        }
    
    def _should_evaluate(self) -> bool:
        """Determine if it's time to evaluate cognitive systems"""
        if not self.last_evaluation_time:
            return True
            
        time_since_evaluation = (datetime.datetime.now() - self.last_evaluation_time).total_seconds()
        cycles_per_evaluation = self.meta_parameters["evaluation_interval"]
        
        # Time-based evaluation
        if self.cognitive_cycle_count % cycles_per_evaluation == 0:
            return True
            
        # Performance-triggered evaluation
        if self.cognitive_cycle_count > 3:  # Minimum cycles before checking
            performance_drop = self._detect_performance_drop()
            if performance_drop:
                logger.info("Performance drop detected - triggering evaluation")
                return True
        
        return False
    
    def _should_reflect(self) -> bool:
        """Determine if it's time to conduct a reflection"""
        if not self.last_reflection_time:
            return False
            
        time_since_reflection = (datetime.datetime.now() - self.last_reflection_time).total_seconds()
        cycles_per_reflection = self.meta_parameters["reflection_frequency"]
        
        # Time-based reflection
        if self.cognitive_cycle_count % cycles_per_reflection == 0:
            return True
            
        # Performance-triggered reflection
        if self.cognitive_cycle_count > 5:  # Minimum cycles before checking
            severe_issues = self._detect_severe_performance_issues()
            if severe_issues:
                logger.info("Severe performance issues detected - triggering reflection")
                return True
                
        # Attention-triggered reflection
        if self.attention_focus and self.attention_focus.get("priority", 0) > 0.8:
            # High priority attention focus might need reflection
            return True
        
        return False
    
    def _should_optimize_parameters(self) -> bool:
        """Determine if it's time to optimize meta-parameters"""
        # Only optimize parameters occasionally (every 50 cycles by default)
        cycles_since_optimization = self.cognitive_cycle_count - self.last_parameter_optimization_cycle
        
        # Time-based optimization
        if cycles_since_optimization >= self.meta_parameters["parameter_optimization_interval"]:
            return True
        
        # If overall performance is very poor, consider optimizing sooner
        if self.performance_history and self.cognitive_cycle_count > 10:
            overall_effectiveness = self._calculate_overall_effectiveness()
            if overall_effectiveness < 0.3:  # Very poor performance
                # The more cycles since last optimization, the more likely to optimize
                probability = cycles_since_optimization / self.meta_parameters["parameter_optimization_interval"]
                if random.random() < probability:
                    return True
        
        return False
    
    def _detect_performance_drop(self) -> bool:
        """Detect if there's been a significant drop in performance"""
        for system_name, history in self.performance_history.items():
            if len(history) < 3:
                continue
                
            # Get recent performance metrics
            recent = history[-3:]
            
            # Check for performance drops in key metrics
            key_metrics = ['success_rate', 'accuracy', 'effectiveness', 'response_quality']
            for metric in key_metrics:
                values = [entry.get('metrics', {}).get(metric) for entry in recent]
                values = [v for v in values if v is not None]
                
                if len(values) >= 2 and values[0] > 0:
                    # Calculate percentage drop
                    drop_percent = (values[0] - values[-1]) / values[0]
                    if drop_percent > 0.2:  # 20% drop threshold
                        logger.info(f"Performance drop detected in {system_name}.{metric}: {drop_percent:.2f}")
                        return True
        
        return False
    
    def _detect_severe_performance_issues(self) -> bool:
        """Detect severe performance issues requiring immediate attention"""
        for system_name, history in self.performance_history.items():
            if not history:
                continue
                
            latest = history[-1].get('metrics', {})
            
            # Check for critically low performance
            if latest.get('success_rate', 1.0) < 0.3:  # Below 30% success
                return True
            if latest.get('error_rate', 0.0) > 0.5:    # Above 50% errors
                return True
            if latest.get('response_time', 0.0) > 5.0:  # Very slow responses
                return True
            
        return False
    
    #--------------------------------------------------------------------------
    # Meta-Learning Methods
    #--------------------------------------------------------------------------
    
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
        learning_rate = self.meta_parameters["learning_rate"]
        
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
            self.feature_importance[feature] = current_importance * (1 - learning_rate) + correlation * learning_rate
            
            # Ensure importance stays in [0,1] range
            self.feature_importance[feature] = min(1.0, max(0.0, self.feature_importance[feature]))
        
        # Check for convergence
        if len(self.feature_history) >= self.meta_parameters["min_samples_required"]:
            converged = self._check_convergence()
            if converged:
                self.meta_parameters["learning_rate"] *= 0.9  # Reduce learning rate as we converge
        
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
        self.algorithm_performance[selected_algorithm]["last_used"] = datetime.datetime.now().isoformat()
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
                "last_used": datetime.datetime.now().isoformat()
            }
            return
            
        # Update with exponential moving average
        current = self.algorithm_performance[algorithm]["success_rate"]
        samples = self.algorithm_performance[algorithm]["samples"]
        alpha = 2 / (samples + 1)  # EMA adjustment factor
        
        new_rate = current * (1 - alpha) + success_rate * alpha
        self.algorithm_performance[algorithm]["success_rate"] = new_rate
        self.algorithm_performance[algorithm]["samples"] += 1
    
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
        
        return max_diff < self.meta_parameters["convergence_threshold"]
    
    def _calculate_exploration_rate(self) -> float:
        """Calculate current exploration rate with decay"""
        min_rate = 0.05  # Minimum exploration rate
        decay_factor = 0.99  # Decay per learning cycle
        base_rate = self.meta_parameters["exploration_rate"]
        current_rate = base_rate * (decay_factor ** self.learning_cycles)
        return max(min_rate, current_rate)
    
    #--------------------------------------------------------------------------
    # Meta-Cognitive Evaluation Methods
    #--------------------------------------------------------------------------
    
    async def evaluate_cognition(self) -> Dict[str, Any]:
        """
        Evaluate performance of all cognitive systems and reallocate resources.
        
        Returns:
            Evaluation results including bottlenecks, strategies, and resource changes
        """
        cycle_start = time.time()
        
        # Collect performance metrics from all cognitive systems
        performance_data = await self._collect_performance_metrics()
        
        # Identify bottlenecks and underperforming processes
        bottlenecks = self._identify_bottlenecks(performance_data)
        
        # Analyze cognitive strategies effectiveness
        strategy_analysis = self._analyze_cognitive_strategies()
        
        # Reallocate resources based on needs and goals
        resource_changes = self._reallocate_resources(bottlenecks, strategy_analysis)
        
        # Generate meta-cognitive insights
        insights = self._generate_metacognitive_insights()
        
        # Store evaluation results
        evaluation_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cognitive_cycle_count,
            "bottlenecks": bottlenecks,
            "strategy_analysis": strategy_analysis,
            "resource_changes": resource_changes,
            "insights": insights,
            "system_metrics": self.system_metrics
        }
        
        # Update core system parameters if needed
        await self._update_system_parameters(bottlenecks, strategy_analysis)
        
        return evaluation_results
    
    async def _collect_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect performance data from all cognitive systems"""
        performance_data = {}
        
        # Collect from registered processes
        for process_id, process in self.cognitive_processes.items():
            metrics = getattr(process, "performance_metrics", {}).copy()
            
            # Add derived metrics
            if hasattr(process, "total_runtime"):
                metrics["runtime"] = process.total_runtime
                
            # Get process type
            process_type = getattr(process, "type", "unknown")
            
            # Store in performance data
            if hasattr(process, "to_dict"):
                process_dict = process.to_dict()
            else:
                # Create simplified dict if to_dict not available
                process_dict = {
                    "id": process_id,
                    "name": getattr(process, "name", "Unknown"),
                    "type": process_type,
                    "status": getattr(process, "status", "unknown")
                }
                
            performance_data[process_id] = {
                "process": process_dict,
                "metrics": metrics
            }
        
        # Also collect from system references
        for system_name, system in self.system_references.items():
            try:
                # Get metrics using the most appropriate method
                system_metrics = None
                
                if hasattr(system, "get_performance_metrics"):
                    system_metrics = await system.get_performance_metrics()
                elif hasattr(system, "get_metrics"):
                    system_metrics = await system.get_metrics()
                elif hasattr(system, "get_stats"):
                    system_metrics = await system.get_stats()
                    
                if system_metrics:
                    sys_id = f"system_{system_name}"
                    
                    # Store simplified representation
                    performance_data[sys_id] = {
                        "process": {
                            "id": sys_id,
                            "name": system_name,
                            "type": system_name,
                            "status": "active"
                        },
                        "metrics": system_metrics
                    }
            except Exception as e:
                logger.error(f"Error collecting metrics from {system_name}: {str(e)}")
        
        return performance_data
    
    def _update_performance_history(self, performance_data: Dict[str, Dict[str, Any]]) -> None:
        """Update performance history with new metrics"""
        # Group by system type
        grouped_metrics = defaultdict(list)
        
        for process_id, data in performance_data.items():
            process_type = data["process"].get("type", "unknown")
            if process_type in self.monitored_systems:
                grouped_metrics[process_type].append(data)
        
        # Update history for each system type
        for system_name in self.monitored_systems:
            if system_name in grouped_metrics:
                # Aggregate metrics from all processes of this type
                aggregated_metrics = self._aggregate_process_metrics(grouped_metrics[system_name])
                
                # Add timestamped entry
                timestamped_metrics = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "cycle": self.cognitive_cycle_count,
                    "metrics": aggregated_metrics
                }
                
                self.performance_history[system_name].append(timestamped_metrics)
                
                # Keep history to a reasonable size
                if len(self.performance_history[system_name]) > 100:
                    self.performance_history[system_name] = self.performance_history[system_name][-100:]
    
    def _aggregate_process_metrics(self, process_data_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics from multiple processes of the same type"""
        # Find all metrics keys
        all_metrics = set()
        for data in process_data_list:
            all_metrics.update(data["metrics"].keys())
        
        # Aggregate each metric
        aggregated = {}
        for metric in all_metrics:
            values = []
            for data in process_data_list:
                if metric in data["metrics"]:
                    value = data["metrics"][metric]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                # Use mean for most metrics
                aggregated[metric] = sum(values) / len(values)
        
        return aggregated
    
    def _identify_bottlenecks(self, performance_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify bottlenecks and underperforming processes"""
        bottlenecks = []
        
        for process_id, data in performance_data.items():
            process = data["process"]
            metrics = data["metrics"]
            
            # Check for high resource utilization
            if metrics.get("resource_utilization", 0) > 0.9:
                bottlenecks.append({
                    "process_id": process_id,
                    "process_name": process.get("name", "Unknown"),
                    "process_type": process.get("type", "unknown"),
                    "type": "resource_utilization",
                    "severity": 0.8,
                    "description": f"Process {process.get('name', 'Unknown')} has high resource utilization",
                    "metrics": {
                        "resource_utilization": metrics.get("resource_utilization", 0)
                    }
                })
            
            # Check for low efficiency
            if metrics.get("efficiency", 0) < 0.3:
                bottlenecks.append({
                    "process_id": process_id,
                    "process_name": process.get("name", "Unknown"),
                    "process_type": process.get("type", "unknown"),
                    "type": "low_efficiency",
                    "severity": 0.7,
                    "description": f"Process {process.get('name', 'Unknown')} has low efficiency",
                    "metrics": {
                        "efficiency": metrics.get("efficiency", 0)
                    }
                })
            
            # Check for high error rates
            if metrics.get("error_rate", 0) > 0.3:
                bottlenecks.append({
                    "process_id": process_id,
                    "process_name": process.get("name", "Unknown"),
                    "process_type": process.get("type", "unknown"),
                    "type": "high_error_rate",
                    "severity": 0.8,
                    "description": f"Process {process.get('name', 'Unknown')} has high error rate",
                    "metrics": {
                        "error_rate": metrics.get("error_rate", 0)
                    }
                })
            
            # Check for slow response time
            if metrics.get("response_time", 0) > 2.0:
                bottlenecks.append({
                    "process_id": process_id,
                    "process_name": process.get("name", "Unknown"),
                    "process_type": process.get("type", "unknown"),
                    "type": "slow_response",
                    "severity": 0.6,
                    "description": f"Process {process.get('name', 'Unknown')} has slow response time",
                    "metrics": {
                        "response_time": metrics.get("response_time", 0)
                    }
                })
            
            # Check for process-specific bottlenecks
            process_bottlenecks = process.get("bottlenecks", [])
            if process_bottlenecks:
                # Get most recent bottleneck
                recent_bottleneck = process_bottlenecks[-1] if isinstance(process_bottlenecks, list) else process_bottlenecks
                
                bottlenecks.append({
                    "process_id": process_id,
                    "process_name": process.get("name", "Unknown"),
                    "process_type": process.get("type", "unknown"),
                    "type": "process_bottleneck",
                    "severity": recent_bottleneck.get("severity", 0.5),
                    "description": recent_bottleneck.get("description", "Unknown bottleneck"),
                    "resource_type": recent_bottleneck.get("resource_type", "general")
                })
        
        # Sort bottlenecks by severity
        bottlenecks.sort(key=lambda x: x["severity"], reverse=True)
        
        return bottlenecks
    
    def _analyze_cognitive_strategies(self) -> Dict[str, Any]:
        """Analyze effectiveness of current cognitive strategies"""
        analysis = {
            "overall_effectiveness": 0.0,
            "system_evaluations": {},
            "adaptation_rate": 0.0,
            "learning_effectiveness": 0.0,
            "recommended_changes": []
        }
        
        # Group processes by type
        processes_by_type = defaultdict(list)
        for process_id, process in self.cognitive_processes.items():
            process_type = getattr(process, "type", "unknown")
            processes_by_type[process_type].append(process)
        
        # Evaluate effectiveness by system type
        total_score = 0.0
        evaluated_types = 0
        
        for system_name in self.monitored_systems:
            # Get historical performance
            if system_name not in self.performance_history or not self.performance_history[system_name]:
                continue
                
            # Get most recent metrics
            recent_entries = self.performance_history[system_name][-3:] if len(self.performance_history[system_name]) >= 3 else self.performance_history[system_name]
            avg_metrics = {}
            
            # Calculate average of recent metrics
            for metric in ["efficiency", "success_rate", "accuracy", "response_time", "error_rate"]:
                values = []
                for entry in recent_entries:
                    if metric in entry["metrics"]:
                        values.append(entry["metrics"][metric])
                
                if values:
                    avg_metrics[metric] = sum(values) / len(values)
            
            # Calculate effectiveness score
            effectiveness_score = 0.5  # Default score
            
            if "efficiency" in avg_metrics:
                effectiveness_score = avg_metrics["efficiency"]
            elif "success_rate" in avg_metrics:
                effectiveness_score = avg_metrics["success_rate"]
            elif "accuracy" in avg_metrics and "response_time" in avg_metrics:
                # Balance accuracy and speed
                norm_time = min(1.0, 1.0 / (1.0 + avg_metrics["response_time"]))
                effectiveness_score = 0.7 * avg_metrics["accuracy"] + 0.3 * norm_time
            
            # Add to analysis
            analysis["system_evaluations"][system_name] = {
                "effectiveness": effectiveness_score,
                "average_metrics": avg_metrics,
                "process_count": len(processes_by_type[system_name])
            }
            
            # Add to overall score
            total_score += effectiveness_score
            evaluated_types += 1
            
            # Generate recommendations
            if effectiveness_score < 0.4:
                analysis["recommended_changes"].append({
                    "system": system_name,
                    "current_effectiveness": effectiveness_score,
                    "recommendation": f"Improve {system_name} strategy - consider new algorithms or increasing resources"
                })
        
        # Calculate overall effectiveness
        if evaluated_types > 0:
            analysis["overall_effectiveness"] = total_score / evaluated_types
        
        # Calculate adaptation and learning metrics
        if len(self.performance_history) >= 5:
            # Check how metrics have improved over time
            improvement_rates = []
            
            for system_name in self.monitored_systems:
                if system_name in analysis["system_evaluations"]:
                    current_score = analysis["system_evaluations"][system_name]["effectiveness"]
                    history = self.performance_history[system_name]
                    
                    if len(history) >= 5:
                        # Get score from 5 cycles ago
                        old_metrics = history[-5]["metrics"]
                        
                        if "efficiency" in old_metrics:
                            old_effectiveness = old_metrics["efficiency"]
                        elif "success_rate" in old_metrics:
                            old_effectiveness = old_metrics["success_rate"]
                        else:
                            continue
                            
                        # Calculate improvement rate
                        if old_effectiveness > 0:
                            improvement = (current_score - old_effectiveness) / old_effectiveness
                            improvement_rates.append(improvement)
            
            if improvement_rates:
                avg_improvement = sum(improvement_rates) / len(improvement_rates)
                analysis["adaptation_rate"] = max(0.0, min(1.0, avg_improvement + 0.5))  # Normalize to 0-1
                analysis["learning_effectiveness"] = analysis["adaptation_rate"]
        
        return analysis
    
    def _reallocate_resources(self, bottlenecks: List[Dict[str, Any]], 
                             strategy_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Reallocate resources based on bottlenecks and strategy analysis"""
        # Initialize with current allocations
        new_allocations = self.resource_allocation.copy()
        
        # Handle critical bottlenecks first
        critical_bottlenecks = [b for b in bottlenecks 
                              if b["severity"] >= self.meta_parameters["bottleneck_severity_threshold"]]
        
        for bottleneck in critical_bottlenecks:
            process_type = bottleneck["process_type"]
            if process_type in new_allocations:
                current = new_allocations[process_type]
                
                # Increase allocation for resource bottlenecks
                if bottleneck["type"] in ["resource_utilization", "slow_response", "process_bottleneck"]:
                    new_allocations[process_type] = min(0.4, current * 1.5)
        
        # Adjust based on strategy effectiveness
        for system_name, eval_data in strategy_analysis["system_evaluations"].items():
            effectiveness = eval_data["effectiveness"]
            
            if system_name in new_allocations:
                current = new_allocations[system_name]
                
                # Adjust based on effectiveness
                if effectiveness < 0.3:
                    # Low effectiveness - increase resources
                    new_allocations[system_name] = min(0.4, current * 1.3)
                elif effectiveness > 0.8:
                    # High effectiveness - may be able to reduce resources
                    new_allocations[system_name] = max(0.1, current * 0.9)
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(new_allocations.values())
        
        if total_allocation > 0:
            for system_name in new_allocations:
                new_allocations[system_name] /= total_allocation
        
        # Calculate changes from current allocation
        changes = {}
        resource_flexibility = self.meta_parameters["resource_flexibility"]
        
        for system_name, new_allocation in new_allocations.items():
            current = self.resource_allocation.get(system_name, 0)
            change = new_allocation - current
            
            # Only record and apply significant changes
            if abs(change) >= 0.02:  # 2% threshold for significance
                changes[system_name] = change
                
                # Apply change (constrained by flexibility)
                max_change = current * resource_flexibility
                applied_change = max(min(change, max_change), -max_change)
                
                # Update resource allocation
                self.resource_allocation[system_name] = current + applied_change
        
        # Ensure allocations still sum to 1.0 after applying changes
        total = sum(self.resource_allocation.values())
        if total > 0 and abs(total - 1.0) > 0.001:  # If not very close to 1.0
            for system_name in self.resource_allocation:
                self.resource_allocation[system_name] /= total
        
        return changes
    
    def _generate_metacognitive_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about cognitive processes and patterns"""
        insights = []
        
        # Check for recurring bottlenecks
        recurring_bottlenecks = self._identify_recurring_bottlenecks()
        
        for bottleneck in recurring_bottlenecks:
            insight_id = f"insight_{self.next_insight_id}"
            self.next_insight_id += 1
            
            insight = {
                "id": insight_id,
                "type": "recurring_bottleneck",
                "system": bottleneck["system"],
                "description": bottleneck["description"],
                "occurrences": bottleneck["occurrences"],
                "severity": bottleneck["avg_severity"],
                "recommendation": bottleneck["recommendation"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            insights.append(insight)
            self.insights.append(insight)
        
        # Check for resource utilization patterns
        resource_patterns = self._identify_resource_patterns()
        
        for pattern in resource_patterns:
            insight_id = f"insight_{self.next_insight_id}"
            self.next_insight_id += 1
            
            insight = {
                "id": insight_id,
                "type": "resource_pattern",
                "resource_type": pattern["resource_type"],
                "description": pattern["description"],
                "trend": pattern["trend"],
                "impact": pattern["impact"],
                "recommendation": pattern["recommendation"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            insights.append(insight)
            self.insights.append(insight)
        
        # Check for inefficient dependencies
        inefficient_dependencies = self._identify_inefficient_dependencies()
        for dependency in inefficient_dependencies[:3]:  # Add top 3 only
            insight_id = f"insight_{self.next_insight_id}"
            self.next_insight_id += 1
            
            insight = {
                "id": insight_id,
                "type": "inefficient_dependency",
                "description": f"Inefficient dependency from {dependency['source_name']} to {dependency['target_name']}",
                "impact": dependency.get("impact", 0.5),
                "recommendation": dependency.get("recommendation", "Reconsider dependency structure"),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            insights.append(insight)
            self.insights.append(insight)
            
        # Generate insights about learning and adaptation
        if len(self.performance_history["reasoning"]) >= 10:
            learning_insights = self._analyze_learning_trends()
            
            for learning_insight in learning_insights:
                insight_id = f"insight_{self.next_insight_id}"
                self.next_insight_id += 1
                
                insight = {
                    "id": insight_id,
                    "type": "learning_trend",
                    "system": learning_insight.get("system", "learning"),
                    "description": learning_insight["description"],
                    "trend": learning_insight.get("trend", {}),
                    "recommendation": learning_insight.get("recommendation", ""),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                insights.append(insight)
                self.insights.append(insight)
        
        # Limit total stored insights
        if len(self.insights) > 100:
            self.insights = self.insights[-100:]
        
        return insights
    
    def _identify_recurring_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify recurring bottlenecks across evaluation cycles"""
        # Count bottleneck occurrences by system
        system_bottlenecks = defaultdict(lambda: {
            "counts": defaultdict(int),
            "severities": defaultdict(list)
        })
        
        # Process bottlenecks from performance history
        for system_name, history in self.performance_history.items():
            for entry in history:
                # Extract bottlenecks if available
                bottlenecks = []
                
                if "bottlenecks" in entry:
                    bottlenecks = entry["bottlenecks"]
                elif "issues" in entry:
                    bottlenecks = entry["issues"]
                
                for bottleneck in bottlenecks:
                    bottleneck_type = bottleneck.get("type", "unknown")
                    severity = bottleneck.get("severity", 0.5)
                    
                    system_bottlenecks[system_name]["counts"][bottleneck_type] += 1
                    system_bottlenecks[system_name]["severities"][bottleneck_type].append(severity)
        
        # Identify significant recurring bottlenecks
        recurring = []
        threshold = 3  # Minimum occurrences to be considered recurring
        
        for system_name, data in system_bottlenecks.items():
            for bottleneck_type, count in data["counts"].items():
                if count >= threshold:
                    # Calculate average severity
                    severities = data["severities"][bottleneck_type]
                    avg_severity = sum(severities) / len(severities) if severities else 0.5
                    
                    # Generate recommendation
                    recommendation = "Consider allocating more resources"
                    
                    if bottleneck_type == "resource_utilization":
                        recommendation = "Optimize resource usage or increase allocation"
                    elif bottleneck_type == "low_efficiency":
                        recommendation = "Review algorithm efficiency or simplify processing"
                    elif bottleneck_type == "high_error_rate":
                        recommendation = "Implement error correction or validation mechanisms"
                    elif bottleneck_type == "slow_response":
                        recommendation = "Optimize processing or implement caching"
                    
                    recurring.append({
                        "system": system_name,
                        "type": bottleneck_type,
                        "description": f"Recurring {bottleneck_type} bottleneck in {system_name} system",
                        "occurrences": count,
                        "avg_severity": avg_severity,
                        "recommendation": recommendation
                    })
        
        # Sort by severity * occurrences
        recurring.sort(key=lambda x: x["avg_severity"] * x["occurrences"], reverse=True)
        
        return recurring
    
    def _identify_resource_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in resource utilization"""
        patterns = []
        
        # Skip if not enough history
        if len(self.performance_history["memory"]) < 5:
            return patterns
        
        # Analyze resource usage patterns for each system
        for system_name, history in self.performance_history.items():
            if len(history) < 5:
                continue
                
            # Extract resource metrics
            resource_metrics = {}
            
            for metric in ["memory_usage", "cpu_usage", "response_time", "throughput"]:
                values = []
                for entry in history:
                    if metric in entry.get("metrics", {}):
                        values.append(entry["metrics"][metric])
                
                if len(values) >= 5:
                    resource_metrics[metric] = values
            
            # Analyze trends for each resource metric
            for metric, values in resource_metrics.items():
                trend = self._calculate_resource_trend(values)
                
                if trend["direction"] != "stable" and abs(trend["magnitude"]) > 0.1:
                    # Define impact based on metric and direction
                    impact = ""
                    recommendation = ""
                    resource_type = metric.split("_")[0] if "_" in metric else metric
                    
                    if metric in ["memory_usage", "cpu_usage"]:
                        if trend["direction"] == "increasing":
                            impact = f"Increasing {resource_type} usage may lead to resource exhaustion"
                            recommendation = f"Optimize {resource_type} usage or increase allocation"
                        else:
                            impact = f"Decreasing {resource_type} usage indicates optimization is working"
                            recommendation = f"Continue current {resource_type} optimization approach"
                    elif metric == "response_time":
                        if trend["direction"] == "increasing":
                            impact = "Increasing response times may degrade user experience"
                            recommendation = "Identify and optimize slow operations"
                        else:
                            impact = "Decreasing response times improve user experience"
                            recommendation = "Continue current optimization approach"
                    elif metric == "throughput":
                        if trend["direction"] == "increasing":
                            impact = "Increasing throughput indicates improved processing efficiency"
                            recommendation = "Continue current optimization approach"
                        else:
                            impact = "Decreasing throughput may indicate developing bottlenecks"
                            recommendation = "Identify and address processing bottlenecks"
                    
                    patterns.append({
                        "system": system_name,
                        "resource_type": resource_type,
                        "metric": metric,
                        "description": f"{system_name} {metric} is {trend['direction']}",
                        "trend": trend,
                        "impact": impact,
                        "recommendation": recommendation
                    })
        
        # Sort by magnitude
        patterns.sort(key=lambda x: x["trend"]["magnitude"], reverse=True)
        
        return patterns
    
    def _analyze_learning_trends(self) -> List[Dict[str, Any]]:
        """Analyze trends in learning and adaptation"""
        insights = []
        
        # Analyze improvement rates for each system
        for system_name, history in self.performance_history.items():
            if len(history) < 10:
                continue
                
            # Extract efficiency or accuracy values
            efficiency_values = []
            
            for entry in history:
                metrics = entry.get("metrics", {})
                
                # Try different performance metrics in order of preference
                if "efficiency" in metrics:
                    efficiency_values.append(metrics["efficiency"])
                elif "success_rate" in metrics:
                    efficiency_values.append(metrics["success_rate"])
                elif "accuracy" in metrics:
                    efficiency_values.append(metrics["accuracy"])
            
            # Calculate learning trend if we have enough data
            if len(efficiency_values) >= 5:
                trend = self._calculate_resource_trend(efficiency_values)
                
                if trend["direction"] == "increasing" and trend["magnitude"] > 0.05:
                    insights.append({
                        "type": "learning_improvement",
                        "system": system_name,
                        "description": f"{system_name} system is showing learning improvements",
                        "trend": trend,
                        "recommendation": "Continue current learning approach"
                    })
                elif trend["direction"] == "decreasing" and trend["magnitude"] > 0.05:
                    insights.append({
                        "type": "learning_degradation",
                        "system": system_name,
                        "description": f"{system_name} system is showing degrading performance",
                        "trend": trend,
                        "recommendation": "Review learning parameters or implement more exploration"
                    })
                elif trend["magnitude"] < 0.01:
                    insights.append({
                        "type": "learning_plateau",
                        "system": system_name,
                        "description": f"{system_name} system has plateaued in learning",
                        "trend": trend,
                        "recommendation": "Introduce new learning challenges or adjust learning rate"
                    })
        
        return insights
    
    def _identify_inefficient_dependencies(self) -> List[Dict[str, Any]]:
        """Identify inefficient dependencies between processes"""
        inefficient_dependencies = []
        
        # Build dependency graph
        dependency_graph = {}
        for process_id, process in self.cognitive_processes.items():
            for dependency in getattr(process, "dependencies", []):
                dep_id = dependency.get("process_id")
                importance = dependency.get("importance", 0.5)
                
                if process_id not in dependency_graph:
                    dependency_graph[process_id] = []
                
                dependency_graph[process_id].append({
                    "target_id": dep_id,
                    "importance": importance
                })
        
        # Check for inefficiencies
        for source_id, dependencies in dependency_graph.items():
            source_process = self.cognitive_processes.get(source_id)
            if not source_process:
                continue
                
            source_name = getattr(source_process, "name", "Unknown Process")
            source_type = getattr(source_process, "type", "unknown")
            
            for dep in dependencies:
                target_id = dep["target_id"]
                importance = dep["importance"]
                
                target_process = self.cognitive_processes.get(target_id)
                if not target_process:
                    continue
                
                target_name = getattr(target_process, "name", "Unknown Process")
                target_type = getattr(target_process, "type", "unknown")
                
                # Check for performance issues
                inefficiency_score = 0.0
                reasons = []
                
                # Check for status issues (blocking)
                source_status = getattr(source_process, "status", "unknown")
                target_status = getattr(target_process, "status", "unknown")
                
                if source_status == "blocked" and target_status != "completed":
                    inefficiency_score += 0.4
                    reasons.append("Frequent blocking")
                
                # Check for high latency in dependent process
                target_metrics = getattr(target_process, "performance_metrics", {})
                if target_metrics.get("response_time", 0) > 2.0:  # High response time
                    inefficiency_score += 0.3
                    reasons.append("High dependency latency")
                
                # Check for high importance but low performance
                if importance > 0.7:
                    target_efficiency = target_metrics.get("efficiency", 0.5)
                    if target_efficiency < 0.4:  # Low efficiency
                        inefficiency_score += importance * 0.4
                        reasons.append("High importance but low performance")
                
                # Add if significant inefficiency found
                if inefficiency_score > 0.3:
                    recommendation = ""
                    
                    if "Frequent blocking" in reasons:
                        recommendation = "Consider making dependency asynchronous or preemptive"
                    elif "High dependency latency" in reasons:
                        recommendation = "Optimize dependent process or use caching"
                    else:
                        recommendation = "Reconsider dependency structure or improve dependent process"
                    
                    inefficient_dependencies.append({
                        "source_id": source_id,
                        "source_name": source_name,
                        "source_type": source_type,
                        "target_id": target_id,
                        "target_name": target_name,
                        "target_type": target_type,
                        "importance": importance,
                        "inefficiency_score": inefficiency_score,
                        "reasons": reasons,
                        "impact": inefficiency_score * importance,
                        "recommendation": recommendation
                    })
        
        # Sort by impact
        inefficient_dependencies.sort(key=lambda x: x["impact"], reverse=True)
        
        return inefficient_dependencies
    
    def _calculate_resource_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend from a series of resource values"""
        if len(values) < 2:
            return {"direction": "stable", "magnitude": 0.0}
            
        # Calculate linear regression
        n = len(values)
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(values) / n
        
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return {"direction": "stable", "magnitude": 0.0}
            
        slope = numerator / denominator
        
        # Normalize slope based on mean value
        if mean_y != 0:
            normalized_slope = slope / abs(mean_y)
        else:
            normalized_slope = slope
            
        # Determine direction and magnitude
        if abs(normalized_slope) < 0.05:
            direction = "stable"
        elif normalized_slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
            
        return {
            "direction": direction,
            "magnitude": abs(normalized_slope),
            "slope": slope,
            "mean": mean_y
        }
    
    def _update_system_metrics(self, cycle_start: float) -> None:
        """Update system metrics based on current state"""
        now = time.time()
        cycle_time = now - cycle_start
        
        # Update system metrics
        self.system_metrics["cycles_completed"] += 1
        
        # Update total runtime
        runtime = (datetime.datetime.now() - self.system_metrics["start_time"]).total_seconds()
        self.system_metrics["total_runtime"] = runtime
        
        # Update process count
        self.system_metrics["total_processes"] = len(self.cognitive_processes)
        
        # Update average cycle time with exponential moving average
        alpha = 0.2  # Weight for current cycle
        current_avg = self.system_metrics["average_cycle_time"]
        new_avg = (1 - alpha) * current_avg + alpha * cycle_time
        self.system_metrics["average_cycle_time"] = new_avg
        
        # Update resource usage metrics if available
        try:
            import psutil
            process = psutil.Process()
            
            # Update CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.system_metrics["resource_usage"]["cpu"] = cpu_percent
            
            # Update memory usage
            memory_info = process.memory_info()
            self.system_metrics["resource_usage"]["memory"] = memory_info.rss / (1024 * 1024)  # MB
            
            # Update IO usage if available
            if hasattr(process, 'io_counters'):
                io_counters = process.io_counters()
                
                # Calculate IO rate since last update if we have previous counters
                if hasattr(self, "_last_io_counters"):
                    last_io = self._last_io_counters
                    io_rate = ((io_counters.read_bytes + io_counters.write_bytes) - 
                             (last_io.read_bytes + last_io.write_bytes)) / cycle_time
                    self.system_metrics["resource_usage"]["io"] = io_rate
                
                self._last_io_counters = io_counters
        except:
            # Resource usage metrics not available, use fallback values
            pass
        
        # Update error rate
        if hasattr(self, 'error_logs') and self.error_logs:
            total_errors = len(self.error_logs)
            if runtime > 0:
                self.system_metrics["error_rate"] = total_errors / runtime
    
    async def _update_system_parameters(self, bottlenecks: List[Dict[str, Any]], 
                                      strategy_analysis: Dict[str, Any]) -> None:
        """Update parameters in other core systems based on evaluation"""
        # Process critical bottlenecks
        critical_bottlenecks = [b for b in bottlenecks if b["severity"] >= 0.7]
        
        for bottleneck in critical_bottlenecks:
            system_name = bottleneck["process_type"]
            
            if system_name in self.system_references:
                system = self.system_references[system_name]
                
                # Create parameter adjustments based on bottleneck type
                param_adjustments = {}
                
                if bottleneck["type"] == "high_error_rate":
                    param_adjustments = {
                        "error_correction_level": "high",
                        "validation_threshold": 0.8
                    }
                elif bottleneck["type"] == "slow_response":
                    param_adjustments = {
                        "caching_enabled": True,
                        "optimization_level": "aggressive"
                    }
                elif bottleneck["type"] == "resource_utilization":
                    param_adjustments = {
                        "resource_efficiency_mode": "enabled",
                        "batch_processing": True
                    }
                
                # Apply parameter adjustments if system supports it
                if param_adjustments and hasattr(system, "set_parameters"):
                    try:
                        await system.set_parameters(param_adjustments)
                        logger.info(f"Updated parameters for {system_name} system: {param_adjustments}")
                    except Exception as e:
                        logger.error(f"Error updating parameters for {system_name}: {str(e)}")
        
        # Apply strategy improvements
        for recommendation in strategy_analysis.get("recommended_changes", []):
            system_name = recommendation.get("system")
            
            if system_name in self.system_references:
                system = self.system_references[system_name]
                
                # Create strategy adjustments
                strategy_adjustments = {
                    "strategy_improvement": True,
                    "effectiveness_target": recommendation.get("current_effectiveness", 0.5) + 0.2
                }
                
                # Apply strategy adjustments if system supports it
                if hasattr(system, "set_strategy"):
                    try:
                        await system.set_strategy(strategy_adjustments)
                        logger.info(f"Updated strategy for {system_name} system")
                    except Exception as e:
                        logger.error(f"Error updating strategy for {system_name}: {str(e)}")
    
    #--------------------------------------------------------------------------
    # Attention Management Methods
    #--------------------------------------------------------------------------
    
    async def set_attention_focus(self, focus: Dict[str, Any]) -> None:
        """Set the current attention focus"""
        self.attention_focus = {
            "target": focus.get("target"),
            "priority": focus.get("priority", 0.5),
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": focus.get("reason", ""),
            "expiration": focus.get("expiration")
        }
        
        logger.info(f"Attention focus set to {focus.get('target')} with priority {focus.get('priority', 0.5)}")

    async def clear_attention_focus(self) -> None:
        """Clear the current attention focus"""
        old_focus = self.attention_focus
        self.attention_focus = None
        
        if old_focus:
            logger.info(f"Cleared attention focus from {old_focus.get('target')}")

    async def get_attention_focus(self) -> Optional[Dict[str, Any]]:
        """Get the current attention focus"""
        return self.attention_focus
    
    async def _check_attention_focus(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if attention focus needs to shift based on context"""
        if not self.attention_focus:
            # No current focus, check if we should establish one
            attention_shift = self._determine_attention_priority(context)
            if attention_shift:
                await self.set_attention_focus(attention_shift)
                return {"type": "established", "focus": attention_shift}
            return None
        
        # Check if current focus has expired
        if "expiration" in self.attention_focus:
            expiration = self.attention_focus["expiration"]
            if isinstance(expiration, str):
                try:
                    expiration_time = datetime.datetime.fromisoformat(expiration)
                    if datetime.datetime.now() > expiration_time:
                        old_focus = self.attention_focus.copy()
                        await self.clear_attention_focus()
                        return {"type": "expired", "previous_focus": old_focus}
                except ValueError:
                    # Invalid timestamp format
                    pass
            elif isinstance(expiration, int):
                # Expiration in cycles
                if self.cognitive_cycle_count >= expiration:
                    old_focus = self.attention_focus.copy()
                    await self.clear_attention_focus()
                    return {"type": "expired", "previous_focus": old_focus}
        
        # Check if a higher priority focus should take over
        new_priority = self._determine_attention_priority(context)
        if new_priority and new_priority.get("priority", 0) > self.attention_focus.get("priority", 0):
            if (new_priority.get("priority", 0) - self.attention_focus.get("priority", 0) > 
                self.meta_parameters["attention_shift_threshold"]):
                old_focus = self.attention_focus.copy()
                await self.set_attention_focus(new_priority)
                return {"type": "shifted", "previous_focus": old_focus, "new_focus": new_priority}
                
        # No change in attention
        return None
    
    def _determine_attention_priority(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine if any system or process needs priority attention"""
        highest_priority = None
        
        # Check for critical bottlenecks
        if "bottlenecks" in context:
            critical_bottlenecks = [b for b in context["bottlenecks"] 
                                  if b.get("severity", 0) >= self.meta_parameters["bottleneck_severity_threshold"]]
            if critical_bottlenecks:
                top_bottleneck = critical_bottlenecks[0]
                priority = top_bottleneck.get("severity", 0.5)
                process_type = top_bottleneck.get("process_type", "unknown")
                process_name = top_bottleneck.get("process_name", "Unknown")
                
                highest_priority = {
                    "target": process_type,
                    "priority": priority,
                    "reason": f"Critical bottleneck in {process_name}",
                    "expiration": self.cognitive_cycle_count + self.meta_parameters["attention_default_duration"]
                }
        
        # Check for extremely low performance
        for system_name, history in self.performance_history.items():
            if not history:
                continue
                
            latest = history[-1].get("metrics", {})
            
            # Check for critically low performance
            if latest.get("success_rate", 1.0) < 0.2:  # Extremely low success
                priority = 0.9
                reason = f"Critically low success rate in {system_name}"
                
                if not highest_priority or priority > highest_priority.get("priority", 0):
                    highest_priority = {
                        "target": system_name,
                        "priority": priority,
                        "reason": reason,
                        "expiration": self.cognitive_cycle_count + self.meta_parameters["attention_default_duration"]
                    }
        
        # Check context for explicit attention requests
        if "attention_request" in context:
            attention_request = context["attention_request"]
            request_priority = attention_request.get("priority", 0.5)
            
            if not highest_priority or request_priority > highest_priority.get("priority", 0):
                highest_priority = {
                    "target": attention_request.get("target"),
                    "priority": request_priority,
                    "reason": attention_request.get("reason", "Explicit request"),
                    "expiration": attention_request.get("expiration", 
                                                     self.cognitive_cycle_count + self.meta_parameters["attention_default_duration"])
                }
        
        return highest_priority
    
    #--------------------------------------------------------------------------
    # Error Logging and Monitoring Methods
    #--------------------------------------------------------------------------
    
    async def log_cognitive_error(self, error_type: str, message: str, 
                              severity: float = 0.5, context: Dict[str, Any] = None) -> None:
        """Log a cognitive error for later analysis"""
        error_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cognitive_cycle_count,
            "type": error_type,
            "message": message,
            "severity": severity,
            "context": context or {}
        }
        
        self.error_logs.append(error_log)
        
        # Keep error logs to a reasonable size
        if len(self.error_logs) > 100:
            self.error_logs = self.error_logs[-100:]
        
        # For high severity errors, consider shifting attention
        if severity >= 0.8:
            # Create attention request for high-severity errors
            attention_request = {
                "target": error_type,
                "priority": severity,
                "reason": f"Critical error: {message[:50]}...",
                "expiration": self.cognitive_cycle_count + 5  # Short focus for error handling
            }
            
            # Only set focus if no higher priority focus exists
            if (not self.attention_focus or 
                attention_request["priority"] > self.attention_focus.get("priority", 0)):
                await self.set_attention_focus(attention_request)
        
        # Log severe errors
        if severity >= 0.7:
            logger.error(f"Cognitive error: {error_type} - {message}")
        elif severity >= 0.5:
            logger.warning(f"Cognitive warning: {error_type} - {message}")
        else:
            logger.info(f"Cognitive notice: {error_type} - {message}")
    
    async def get_error_logs(self, error_type: Optional[str] = None, 
                        min_severity: float = 0.0,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """Get error logs, optionally filtered by type and severity"""
        filtered_logs = [log for log in self.error_logs 
                      if log["severity"] >= min_severity and 
                         (error_type is None or log["type"] == error_type)]
        
        # Sort by timestamp (most recent first)
        sorted_logs = sorted(filtered_logs, 
                          key=lambda x: x["timestamp"], 
                          reverse=True)
        
        return sorted_logs[:limit]
    
    def _calculate_overall_effectiveness(self) -> float:
        """Calculate overall system effectiveness across all monitored systems"""
        total_score = 0.0
        systems_count = 0
        
        for system_name, history in self.performance_history.items():
            if not history:
                continue
                
            # Get latest metrics
            latest = history[-1].get("metrics", {})
            
            # Calculate effectiveness
            effectiveness = 0.5  # Default
            
            if "effectiveness" in latest:
                effectiveness = latest["effectiveness"]
            elif "success_rate" in latest:
                effectiveness = latest["success_rate"]
            elif "accuracy" in latest and "response_time" in latest:
                # Balance accuracy and response time
                norm_time = min(1.0, 1.0 / (1.0 + latest["response_time"]))
                effectiveness = 0.7 * latest["accuracy"] + 0.3 * norm_time
            elif "error_rate" in latest:
                # Lower error rate means higher effectiveness
                effectiveness = max(0.0, 1.0 - latest["error_rate"] * 2)
            
            total_score += effectiveness
            systems_count += 1
        
        if systems_count == 0:
            return 0.5  # Default when no data available
            
        return total_score / systems_count
    
    #--------------------------------------------------------------------------
    # Meta-Parameter Optimization Methods
    #--------------------------------------------------------------------------
    
    async def improve_meta_parameters(self) -> Dict[str, Any]:
        """Recursively improve the meta-parameters themselves"""
        # History of parameter configurations and their performance
        param_history = self._get_parameter_history()
        
        # Analyze which parameter changes led to improvements
        param_effectiveness = self._analyze_parameter_effectiveness(param_history)
        
        # Generate new parameter configurations to try
        new_params = self._generate_parameter_candidates(param_effectiveness)
        
        # Evaluate and select the most promising configuration
        selected_params = await self._evaluate_parameter_candidates(new_params)
        
        # Apply the selected parameters
        old_params = self.meta_parameters.copy()
        self._apply_meta_parameters(selected_params)
        
        # Record the changes
        changes = {}
        for param, new_value in selected_params.items():
            old_value = old_params.get(param)
            if new_value != old_value:
                changes[param] = {"old": old_value, "new": new_value}
        
        # Return the changes
        return {
            "parameters_changed": changes,
            "effectiveness_analysis": param_effectiveness,
            "cycle": self.cognitive_cycle_count
        }
    
    def _get_parameter_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter configurations and their performance"""
        # Create if not exists
        if not hasattr(self, '_parameter_history'):
            self._parameter_history = []
            
            # Add current configuration as baseline
            self._parameter_history.append({
                "parameters": self.meta_parameters.copy(),
                "timestamp": datetime.datetime.now().isoformat(),
                "cycle": self.cognitive_cycle_count,
                "metrics": {
                    "cycle_time": self.system_metrics["average_cycle_time"],
                    "error_rate": self.system_metrics["error_rate"],
                    "overall_effectiveness": self._calculate_overall_effectiveness()
                }
            })
            
        return self._parameter_history
    
    def _analyze_parameter_effectiveness(self, param_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze which parameter changes led to improvements"""
        if len(param_history) < 2:
            # Not enough history to analyze
            return {param: {"effect": 0.0, "confidence": 0.0} for param in self.meta_parameters}
            
        # Analyze each parameter
        effectiveness = {}
        
        for param in self.meta_parameters:
            # Skip parameters without enough variation
            values = [entry["parameters"].get(param) for entry in param_history]
            unique_values = set(values)
            
            if len(unique_values) < 2:
                effectiveness[param] = {"effect": 0.0, "confidence": 0.0}
                continue
                
            # Calculate correlation with performance
            cycle_times = [entry["metrics"].get("cycle_time", 0) for entry in param_history]
            error_rates = [entry["metrics"].get("error_rate", 0) for entry in param_history]
            effectiveness_scores = [entry["metrics"].get("overall_effectiveness", 0.5) for entry in param_history]
            
            # Correlate parameter with various metrics
            time_correlation = self._calculate_correlation(values, cycle_times)
            error_correlation = self._calculate_correlation(values, error_rates)
            effectiveness_correlation = self._calculate_correlation(values, effectiveness_scores)
            
            # Combine correlations (negative correlation with time and error is good, positive with effectiveness is good)
            effect = -0.3 * time_correlation - 0.3 * error_correlation + 0.4 * effectiveness_correlation
            
            # Determine confidence based on sample size and consistency
            confidence = min(1.0, len(param_history) / 10) * min(1.0, abs(effect) * 2)
            
            effectiveness[param] = {
                "effect": effect,
                "confidence": confidence,
                "effectiveness_correlation": effectiveness_correlation,
                "error_correlation": error_correlation,
                "time_correlation": time_correlation
            }
        
        return effectiveness
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two variables"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
            
        return numerator / (math.sqrt(denominator_x) * math.sqrt(denominator_y))
    
    def _generate_parameter_candidates(self, param_effectiveness: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate new parameter configurations to try"""
        candidates = []
        current_config = self.meta_parameters.copy()
        
        # Generate individual parameter variations first
        for param, effect_data in param_effectiveness.items():
            effect = effect_data.get("effect", 0.0)
            confidence = effect_data.get("confidence", 0.0)
            
            # Skip parameters with low confidence
            if confidence < 0.2:
                continue
                
            # Adjust parameter based on effect direction and confidence
            if abs(effect) > 0.1:  # Only adjust if effect is significant
                candidate = current_config.copy()
                
                # Adjust in the direction of improvement
                adjustment = effect * confidence * self.meta_parameters["parameter_adjustment_factor"]
                
                # Apply adjustment based on parameter type
                if param in ["evaluation_interval", "reflection_frequency", "parameter_optimization_interval"]:
                    # Integer parameter
                    current_value = candidate[param]
                    new_value = max(1, round(current_value * (1 + adjustment)))
                    candidate[param] = new_value
                elif param in ["learning_rate", "exploration_rate"]:
                    # Rate parameter (0-1)
                    current_value = candidate[param]
                    new_value = min(0.9, max(0.01, current_value + adjustment * 0.1))
                    candidate[param] = new_value
                elif param.endswith("threshold"):
                    # Threshold parameter (0-1)
                    current_value = candidate[param]
                    new_value = min(0.95, max(0.05, current_value + adjustment * 0.1))
                    candidate[param] = new_value
                elif param == "resource_flexibility":
                    # Flexibility parameter (0-1)
                    current_value = candidate[param]
                    new_value = min(0.9, max(0.1, current_value + adjustment * 0.1))
                    candidate[param] = new_value
                elif param == "parameter_adjustment_factor":
                    # Adjustment factor (0-1)
                    current_value = candidate[param]
                    new_value = min(0.5, max(0.05, current_value + adjustment * 0.05))
                    candidate[param] = new_value
                else:
                    # Default case - scale by percentage
                    current_value = candidate[param]
                    new_value = current_value * (1 + adjustment)
                    candidate[param] = new_value
                
                candidates.append(candidate)
        
        # Generate combinations of top parameters
        top_params = sorted(param_effectiveness.items(), 
                          key=lambda x: abs(x[1]["effect"]) * x[1]["confidence"], 
                          reverse=True)[:3]
        
        if len(top_params) >= 2:
            combo_candidate = current_config.copy()
            
            for param, effect_data in top_params:
                effect = effect_data.get("effect", 0.0)
                confidence = effect_data.get("confidence", 0.0)
                
                if abs(effect) > 0.1 and confidence > 0.2:
                    adjustment = effect * confidence * 0.05  # Smaller adjustment for combined changes
                    
                    # Apply adjustment (similar logic as above)
                    if param in ["evaluation_interval", "reflection_frequency", "parameter_optimization_interval"]:
                        current_value = combo_candidate[param]
                        new_value = max(1, round(current_value * (1 + adjustment)))
                        combo_candidate[param] = new_value
                    elif param in ["learning_rate", "exploration_rate"]:
                        current_value = combo_candidate[param]
                        new_value = min(0.9, max(0.01, current_value + adjustment * 0.1))
                        combo_candidate[param] = new_value
                    elif param.endswith("threshold"):
                        current_value = combo_candidate[param]
                        new_value = min(0.95, max(0.05, current_value + adjustment * 0.1))
                        combo_candidate[param] = new_value
                    else:
                        current_value = combo_candidate[param]
                        new_value = current_value * (1 + adjustment)
                        combo_candidate[param] = new_value
            
            candidates.append(combo_candidate)
        
        # Add a random exploration candidate
        random_candidate = current_config.copy()
        
        for param in random_candidate:
            # Apply random adjustment
            if param in ["evaluation_interval", "reflection_frequency", "parameter_optimization_interval"]:
                random_candidate[param] = max(1, round(random_candidate[param] * random.uniform(0.8, 1.2)))
            elif param in ["learning_rate", "exploration_rate"]:
                random_candidate[param] = min(0.9, max(0.01, random_candidate[param] + random.uniform(-0.1, 0.1)))
            elif param.endswith("threshold"):
                random_candidate[param] = min(0.95, max(0.05, random_candidate[param] + random.uniform(-0.1, 0.1)))
            else:
                random_candidate[param] = random_candidate[param] * random.uniform(0.8, 1.2)
        
        candidates.append(random_candidate)
        
        # Add current configuration as a safe fallback
        candidates.append(current_config)
        
        return candidates
    
    async def _evaluate_parameter_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate and select the most promising parameter configuration"""
        best_candidate = None
        best_score = float('-inf')
        
        for candidate in candidates:
            # Evaluate candidate
            score = self._evaluate_parameter_candidate(candidate)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        # Log candidate evaluation
        if not hasattr(self, '_parameter_evaluations'):
            self._parameter_evaluations = []
            
        self._parameter_evaluations.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cognitive_cycle_count,
            "candidates": candidates,
            "best_candidate": best_candidate,
            "best_score": best_score
        })
        
        return best_candidate or self.meta_parameters.copy()
    
    def _evaluate_parameter_candidate(self, candidate: Dict[str, Any]) -> float:
        """Evaluate a parameter candidate configuration"""
        # Start with a base score
        score = 0.0
        
        # Current performance metrics for reference
        overall_effectiveness = self._calculate_overall_effectiveness()
        avg_cycle_time = self.system_metrics["average_cycle_time"]
        error_rate = self.system_metrics["error_rate"]
        
        # Check for valid ranges and penalize invalid configurations
        if candidate.get("learning_rate", 0) > candidate.get("exploration_rate", 0) * 2:
            score -= 1.0  # Learning rate should generally be lower than exploration rate
        
        # Check threshold parameters are in valid range
        for param in candidate:
            if param.endswith("threshold") and (candidate[param] < 0 or candidate[param] > 1):
                return float('-inf')  # Invalid threshold
                
        # Reward configurations that might improve overall effectiveness
        potential_improvement = 0.0
        
        # Check learning rate - higher can be good for low effectiveness, lower for high effectiveness
        if overall_effectiveness < 0.5 and candidate.get("learning_rate", 0) > self.meta_parameters.get("learning_rate", 0):
            potential_improvement += 0.3
        elif overall_effectiveness > 0.8 and candidate.get("learning_rate", 0) < self.meta_parameters.get("learning_rate", 0):
            potential_improvement += 0.2
            
        # Check evaluation interval - shorter can be good for high error rates
        if error_rate > 0.3 and candidate.get("evaluation_interval", 100) < self.meta_parameters.get("evaluation_interval", 100):
            potential_improvement += 0.4
            
        # Reflect potential improvement in score
        score += potential_improvement
        
        # Reward balance between parameters
        learning_to_exploration = candidate.get("learning_rate", 0.1) / max(0.05, candidate.get("exploration_rate", 0.2))
        if 0.2 < learning_to_exploration < 1.0:
            score += 0.2  # Good balance
            
        # Reward efficient evaluation intervals
        if 3 <= candidate.get("evaluation_interval", 5) <= 10:
            score += 0.2  # Reasonable interval
            
        # Penalize extremes
        for param, value in candidate.items():
            if param in ["learning_rate", "exploration_rate", "confidence_threshold", "resource_flexibility"]:
                if value < 0.05 or value > 0.95:
                    score -= 0.3  # Extreme value
        
        # Add some noise for exploration
        score += random.uniform(-0.1, 0.1)
        
        return score
    
    def _apply_meta_parameters(self, selected_params: Dict[str, Any]) -> None:
        """Apply the selected parameters"""
        # Record the current state before applying changes
        if not hasattr(self, '_parameter_history'):
            self._parameter_history = []
            
        self._parameter_history.append({
            "parameters": self.meta_parameters.copy(),
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cognitive_cycle_count,
            "metrics": {
                "cycle_time": self.system_metrics["average_cycle_time"],
                "error_rate": self.system_metrics["error_rate"],
                "overall_effectiveness": self._calculate_overall_effectiveness()
            }
        })
        
        # Apply new configuration
        for param, value in selected_params.items():
            if param in self.meta_parameters:
                self.meta_parameters[param] = value
                
        logger.info(f"Applied new meta-parameters at cycle {self.cognitive_cycle_count}")
    
    #--------------------------------------------------------------------------
    # Reflection and Improvement Methods
    #--------------------------------------------------------------------------
    
    async def _conduct_reflection(self) -> Dict[str, Any]:
        """Conduct a comprehensive self-reflection on system performance"""
        # Analyze recent performance
        performance_analysis = self._analyze_recent_performance()
        
        # Generate insights about cognitive patterns
        insights = self._generate_cognitive_insights(performance_analysis)
        
        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(performance_analysis, insights)
        
        # Generate new cognitive strategies
        new_strategies = await self._generate_cognitive_strategies(improvement_areas)
        
        # Create an improvement plan
        improvement_plan = self._create_improvement_plan(improvement_areas, new_strategies)
        self.improvement_plans.append(improvement_plan)
        
        # Create reflection record
        reflection_id = f"reflection_{self.next_reflection_id}"
        self.next_reflection_id += 1
        
        reflection = {
            "id": reflection_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cognitive_cycle_count,
            "performance_analysis": performance_analysis,
            "insights": insights,
            "improvement_areas": improvement_areas,
            "strategies": new_strategies,
            "plan": improvement_plan
        }
        
        # Add to reflection history
        self.reflections.append(reflection)
        
        # Limit reflection history size
        if len(self.reflections) > 20:
            self.reflections = self.reflections[-20:]
        
        return reflection
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance across all systems"""
        analysis = {}
        
        for system_name, history in self.performance_history.items():
            if not history:
                analysis[system_name] = {"status": "insufficient_data"}
                continue
                
            # Calculate trends for key metrics
            system_analysis = {"trends": {}}
            for metric in ['success_rate', 'accuracy', 'effectiveness', 'error_rate', 'response_time']:
                values = []
                for entry in history:
                    metrics = entry.get("metrics", {})
                    if metric in metrics:
                        values.append(metrics[metric])
                
                if len(values) >= 3:
                    trend = self._calculate_resource_trend(values)
                    system_analysis["trends"][metric] = trend
            
            # Determine overall status
            positive_trends = sum(1 for t in system_analysis["trends"].values() 
                               if t.get("direction") == "improving")
            negative_trends = sum(1 for t in system_analysis["trends"].values() 
                               if t.get("direction") == "declining")
            
            if positive_trends > negative_trends * 2:
                status = "excellent"
            elif positive_trends > negative_trends:
                status = "good"
            elif positive_trends == negative_trends:
                status = "stable"
            elif negative_trends > positive_trends * 2:
                status = "critical"
            else:
                status = "concerning"
                
            system_analysis["status"] = status
            
            # Get latest performance
            if history:
                latest = history[-1].get("metrics", {})
                system_analysis["current_metrics"] = latest
            
            analysis[system_name] = system_analysis
        
        return analysis
    
    def _generate_cognitive_insights(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about cognitive patterns and performance"""
        insights = []
        
        # Check for systems with excellent performance
        excellent_systems = [sys for sys, data in performance_analysis.items() 
                           if data.get("status") == "excellent"]
        
        if excellent_systems:
            insights.append({
                "type": "strength",
                "description": f"Systems showing excellent performance: {', '.join(excellent_systems)}",
                "confidence": 0.9
            })
        
        # Check for systems with critical performance
        critical_systems = [sys for sys, data in performance_analysis.items() 
                          if data.get("status") == "critical"]
        
        if critical_systems:
            insights.append({
                "type": "weakness",
                "description": f"Systems showing critical performance issues: {', '.join(critical_systems)}",
                "confidence": 0.9,
                "priority": "high"
            })
            
            # Add specific insights for each critical system
            for system in critical_systems:
                trends = performance_analysis[system].get("trends", {})
                metrics = performance_analysis[system].get("current_metrics", {})
                
                # Identify problematic metrics
                if metrics.get("error_rate", 0) > 0.3:
                    insights.append({
                        "type": "weakness",
                        "system": system,
                        "description": f"High error rate in {system} system: {metrics['error_rate']:.2f}",
                        "confidence": 0.9,
                        "priority": "high"
                    })
                
                if metrics.get("response_time", 0) > 2.0:
                    insights.append({
                        "type": "weakness",
                        "system": system,
                        "description": f"Slow response time in {system} system: {metrics['response_time']:.2f}s",
                        "confidence": 0.9,
                        "priority": "high"
                    })
                
                if metrics.get("success_rate", 1.0) < 0.5:
                    insights.append({
                        "type": "weakness",
                        "system": system,
                        "description": f"Low success rate in {system} system: {metrics['success_rate']:.2f}",
                        "confidence": 0.9,
                        "priority": "high"
                    })
        
        # Check for improving systems
        improving_systems = []
        for system, data in performance_analysis.items():
            trends = data.get("trends", {})
            improving_metrics = [m for m, t in trends.items() 
                              if t.get("direction") == "improving" and t.get("magnitude", 0) > 0.1]
            
            if improving_metrics and len(improving_metrics) >= 2:
                improving_systems.append((system, improving_metrics))
        
        if improving_systems:
            for system, metrics in improving_systems:
                metrics_str = ", ".join(metrics)
                insights.append({
                    "type": "improvement",
                    "system": system,
                    "description": f"{system} system is showing improvements in: {metrics_str}",
                    "confidence": 0.8
                })
        
        # Check for cross-system patterns
        all_statuses = [data.get("status") for data in performance_analysis.values() 
                      if "status" in data and data["status"] != "insufficient_data"]
        
        if all_statuses and all(status in ["excellent", "good"] for status in all_statuses):
            insights.append({
                "type": "synergy",
                "description": "All systems are performing well, indicating good synergy",
                "confidence": 0.8
            })
        
        # Check for resource allocation effectiveness
        for system, allocation in self.resource_allocation.items():
            if system in performance_analysis:
                status = performance_analysis[system].get("status")
                
                if status == "excellent" and allocation < 0.15:
                    insights.append({
                        "type": "efficiency",
                        "system": system,
                        "description": f"{system} system performing excellently with minimal resources ({allocation:.2f})",
                        "confidence": 0.7
                    })
                elif status == "critical" and allocation > 0.25:
                    insights.append({
                        "type": "inefficiency",
                        "system": system,
                        "description": f"{system} system performing poorly despite high resource allocation ({allocation:.2f})",
                        "confidence": 0.7,
                        "priority": "medium"
                    })
        
        return insights
    
    def _identify_improvement_areas(self, 
                                  performance_analysis: Dict[str, Any], 
                                  insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify specific areas for cognitive improvement"""
        improvement_areas = []
        
        # Identify systems needing improvement
        critical_systems = [sys for sys, data in performance_analysis.items() 
                          if data.get("status") in ["critical", "concerning"]]
        
        for system_name in critical_systems:
            analysis = performance_analysis[system_name]
            
            # Identify problematic metrics
            problematic_metrics = []
            for metric, trend in analysis.get("trends", {}).items():
                if trend.get("direction") == "declining" and trend.get("magnitude", 0) > 0.1:
                    problematic_metrics.append(metric)
            
            # Current metrics for reference
            current_metrics = analysis.get("current_metrics", {})
            
            # Determine priority
            priority = 1 if analysis.get("status") == "critical" else 2
            
            improvement_areas.append({
                "system": system_name,
                "priority": priority,
                "metrics_to_improve": problematic_metrics,
                "current_metrics": current_metrics,
                "current_status": analysis.get("status")
            })
        
        # Check for resource allocation improvements
        resource_insights = [i for i in insights if i["type"] in ["efficiency", "inefficiency"]]
        
        if resource_insights:
            improvement_areas.append({
                "system": "resource_allocation",
                "priority": 3,  # Lower priority than system-specific issues
                "description": "Resource allocation needs optimization",
                "details": [i["description"] for i in resource_insights],
                "current_status": "inefficient"
            })
        
        # Sort by priority
        improvement_areas.sort(key=lambda x: x["priority"])
        
        return improvement_areas
    
    async def _generate_cognitive_strategies(self, improvement_areas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate new cognitive strategies for improvement areas"""
        strategies = []
        
        for area in improvement_areas:
            system = area.get("system")
            
            # Generate system-specific strategies
            if system in self.system_references:
                system_ref = self.system_references[system]
                
                # Try to get strategies from the system itself if it has the capability
                system_strategies = []
                if hasattr(system_ref, "generate_improvement_strategies"):
                    try:
                        system_strategies = await system_ref.generate_improvement_strategies()
                    except Exception as e:
                        logger.error(f"Error generating strategies from {system}: {str(e)}")
                
                if system_strategies:
                    for strategy in system_strategies:
                        strategies.append({
                            "name": strategy.get("name", f"Strategy for {system}"),
                            "system": system,
                            "description": strategy.get("description", ""),
                            "implementation": strategy.get("implementation", {}),
                            "expected_impact": strategy.get("expected_impact", {}),
                            "source": "system_generated"
                        })
                else:
                    # Generate generic strategies if the system doesn't provide them
                    generic_strategy = self._generate_generic_strategy(system, area)
                    if generic_strategy:
                        strategies.append(generic_strategy)
            
            # Generate resource allocation strategies
            elif system == "resource_allocation":
                strategy = {
                    "name": "Resource Reallocation",
                    "system": "resource_allocation",
                    "description": "Adjust resource allocation based on system performance",
                    "implementation": {
                        "type": "resource_adjustment",
                        "adjustments": self._generate_resource_adjustments(improvement_areas)
                    },
                    "expected_impact": {
                        "balanced_performance": 0.7,
                        "overall_efficiency": 0.6
                    },
                    "source": "meta_generated"
                }
                strategies.append(strategy)
        
        return strategies
    
    def _generate_generic_strategy(self, system: str, area: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a generic strategy for a system"""
        metrics_to_improve = area.get("metrics_to_improve", [])
        current_metrics = area.get("current_metrics", {})
        
        strategy_type = "optimization"
        strategy_description = f"Optimize {system} system performance"
        strategy_details = {}
        expected_impact = {}
        
        # Customize strategy based on problematic metrics
        if "error_rate" in metrics_to_improve or current_metrics.get("error_rate", 0) > 0.3:
            strategy_type = "error_reduction"
            strategy_description = f"Reduce errors in {system} system"
            strategy_details = {
                "error_correction": True,
                "validation_level": "increased",
                "monitoring": "enhanced"
            }
            expected_impact = {
                "error_rate": -0.2,  # 20% reduction
                "success_rate": 0.15  # 15% improvement
            }
        elif "response_time" in metrics_to_improve or current_metrics.get("response_time", 0) > 2.0:
            strategy_type = "performance_optimization"
            strategy_description = f"Improve response time in {system} system"
            strategy_details = {
                "caching": True,
                "parallel_processing": True,
                "optimization_level": "aggressive"
            }
            expected_impact = {
                "response_time": -0.3,  # 30% reduction
                "throughput": 0.2  # 20% improvement
            }
        elif "success_rate" in metrics_to_improve or current_metrics.get("success_rate", 1.0) < 0.5:
            strategy_type = "success_improvement"
            strategy_description = f"Improve success rate in {system} system"
            strategy_details = {
                "fallback_mechanisms": True,
                "multiple_attempts": True,
                "adaptive_strategy": True
            }
            expected_impact = {
                "success_rate": 0.2,  # 20% improvement
                "error_rate": -0.15  # 15% reduction
            }
        else:
            # General optimization strategy
            strategy_details = {
                "optimization_level": "balanced",
                "resource_efficiency": True,
                "monitoring": "standard"
            }
            expected_impact = {
                "overall_performance": 0.15  # 15% improvement
            }
        
        return {
            "name": f"{strategy_type.capitalize()} for {system}",
            "system": system,
            "type": strategy_type,
            "description": strategy_description,
            "implementation": strategy_details,
            "expected_impact": expected_impact,
            "source": "meta_generated"
        }
    
    def _generate_resource_adjustments(self, improvement_areas: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate resource adjustment recommendations"""
        adjustments = {}
        
        # Reduce resources for inefficient systems
        for area in improvement_areas:
            system = area.get("system")
            if system != "resource_allocation" and area.get("current_status") in ["inefficient", "critical"]:
                current_allocation = self.resource_allocation.get(system, 0.2)
                
                if area.get("current_status") == "inefficient":
                    # Reduce allocation for inefficient systems
                    adjustments[system] = current_allocation * 0.8
                else:
                    # Increase allocation for critical systems
                    adjustments[system] = min(0.4, current_allocation * 1.5)
        
        # Balance adjustments to sum to 1.0
        total_adjusted = sum(adjustments.values())
        remaining = 1.0 - total_adjusted
        
        # Distribute remaining resources to systems not explicitly adjusted
        unadjusted_systems = [system for system in self.resource_allocation 
                             if system not in adjustments]
        
        if unadjusted_systems:
            per_system = remaining / len(unadjusted_systems)
            for system in unadjusted_systems:
                adjustments[system] = per_system
        
        return adjustments
    
    def _create_improvement_plan(self, 
                               improvement_areas: List[Dict[str, Any]], 
                               strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a comprehensive improvement plan"""
        # Map strategies to improvement areas
        area_strategies = {}
        for area in improvement_areas:
            system = area.get("system")
            area_strategies[system] = [s for s in strategies if s.get("system") == system]
        
        # Create plan with phases
        plan = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cognitive_cycle_count,
            "priority_areas": [area["system"] for area in improvement_areas if area.get("priority", 3) == 1],
            "phases": [
                {
                    "name": "Critical Improvements",
                    "duration": 5,  # cycles
                    "targets": [area["system"] for area in improvement_areas if area.get("priority", 3) == 1],
                    "strategies": [s for s in strategies if s.get("system") in 
                                [area["system"] for area in improvement_areas if area.get("priority", 3) == 1]]
                },
                {
                    "name": "Secondary Enhancements",
                    "duration": 10,  # cycles
                    "targets": [area["system"] for area in improvement_areas if area.get("priority", 3) == 2],
                    "strategies": [s for s in strategies if s.get("system") in 
                                [area["system"] for area in improvement_areas if area.get("priority", 3) == 2]]
                },
                {
                    "name": "Optimization",
                    "duration": 15,  # cycles
                    "targets": [area["system"] for area in improvement_areas if area.get("priority", 3) == 3],
                    "strategies": [s for s in strategies if s.get("system") in 
                                [area["system"] for area in improvement_areas if area.get("priority", 3) == 3]]
                }
            ],
            "expected_outcomes": {
                "performance_improvement": 0.3,
                "bottleneck_reduction": 0.5,
                "efficiency_gain": 0.2
            },
            "status": "created"
        }
        
        return plan
    
    async def _conduct_initial_assessment(self) -> None:
        """Conduct initial assessment of cognitive systems"""
        logger.info("Conducting initial assessment")
        
        # Start with baseline self-assessment
        initial_assessment = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": 0,
            "systems": {},
            "overall_state": "initializing",
            "priorities": [],
            "initial_strategies": []
        }
        
        # Check each system for baseline metrics
        for system_name, system in self.system_references.items():
            if system_name in self.monitored_systems:
                try:
                    # Get initial metrics
                    metrics = {}
                    
                    if hasattr(system, "get_performance_metrics"):
                        metrics = await system.get_performance_metrics()
                    elif hasattr(system, "get_metrics"):
                        metrics = await system.get_metrics()
                    elif hasattr(system, "get_stats"):
                        metrics = await system.get_stats()
                    
                    initial_assessment["systems"][system_name] = {
                        "initial_metrics": metrics,
                        "parameters": self.monitored_systems[system_name]["parameters"]
                    }
                    
                    # Identify initial high-priority systems
                    if metrics.get("error_rate", 0) > 0.3:
                        initial_assessment["priorities"].append(system_name)
                    if metrics.get("success_rate", 1.0) < 0.5:
                        initial_assessment["priorities"].append(system_name)
                    
                except Exception as e:
                    logger.error(f"Error in initial assessment of {system_name}: {str(e)}")
                    initial_assessment["systems"][system_name] = {"error": str(e)}
        
        # Generate initial strategies for high-priority systems
        for system_name in set(initial_assessment["priorities"]):
            strategy = {
                "name": f"Initial Optimization for {system_name}",
                "system": system_name,
                "description": f"Initial performance improvement for {system_name}",
                "implementation": {
                    "type": "parameter_tuning",
                    "parameters": {
                        "optimization_level": "moderate",
                        "error_tolerance": "adaptive",
                        "performance_focus": True
                    }
                }
            }
            initial_assessment["initial_strategies"].append(strategy)
        
        # Set initial resource allocation based on priorities
        if initial_assessment["priorities"]:
            priority_count = len(set(initial_assessment["priorities"]))
            priority_allocation = 0.5  # 50% of resources to priority systems
            per_priority = priority_allocation / priority_count
            
            remaining = 1.0 - priority_allocation
            non_priority_count = len(self.resource_allocation) - priority_count
            per_non_priority = remaining / non_priority_count if non_priority_count > 0 else 0
            
            for system in self.resource_allocation:
                if system in initial_assessment["priorities"]:
                    self.resource_allocation[system] = per_priority
                else:
                    self.resource_allocation[system] = per_non_priority
        
        # Create initial reflection
        self.reflections.append(initial_assessment)
        logger.info("Initial assessment completed")
    
    async def _extract_system_parameters(self, system: Any) -> Dict[str, Any]:
        """Extract current parameters from a system"""
        parameters = {}
        
        try:
            if hasattr(system, "get_parameters"):
                parameters = await system.get_parameters()
            elif hasattr(system, "parameters"):
                parameters = system.parameters
            elif hasattr(system, "get_config"):
                parameters = await system.get_config()
            elif hasattr(system, "config"):
                parameters = system.config
        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
        
        return parameters
    
    #--------------------------------------------------------------------------
    # Cognitive Process Management Methods
    #--------------------------------------------------------------------------
    
    async def register_cognitive_process(self, name: str, type: str,
                                      priority: float = 0.5, 
                                      resource_allocation: float = 0.1) -> str:
        """Register a new cognitive process for monitoring"""
        # Generate process ID
        process_id = f"process_{self.next_process_id}"
        self.next_process_id += 1
        
        # Create process object
        process = {
            "id": process_id,
            "name": name,
            "type": type,
            "priority": priority,
            "resource_allocation": resource_allocation,
            "performance_metrics": {},
            "bottlenecks": [],
            "dependencies": [],
            "start_time": datetime.datetime.now(),
            "last_activity": datetime.datetime.now(),
            "total_runtime": 0.0,
            "status": "idle"
        }
        
        # Add to processes
        self.cognitive_processes[process_id] = process
        
        return process_id
    
    async def update_process_status(self, process_id: str, status: str,
                                 metrics: Optional[Dict[str, float]] = None) -> bool:
        """Update status and metrics for a cognitive process"""
        if process_id not in self.cognitive_processes:
            return False
            
        process = self.cognitive_processes[process_id]
        
        # Update status
        process["status"] = status
        
        # Update last activity
        process["last_activity"] = datetime.datetime.now()
        
        # Update total runtime
        runtime = (datetime.datetime.now() - process["start_time"]).total_seconds()
        process["total_runtime"] = runtime
        
        # Update metrics if provided
        if metrics:
            process["performance_metrics"].update(metrics)
        
        return True
    
    async def report_bottleneck(self, process_id: str, description: str, 
                             severity: float = 0.5, 
                             resource_type: str = "general") -> bool:
        """Report a bottleneck for a cognitive process"""
        if process_id not in self.cognitive_processes:
            return False
            
        process = self.cognitive_processes[process_id]
        
        # Add bottleneck
        bottleneck = {
            "description": description,
            "severity": severity,
            "resource_type": resource_type,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if not isinstance(process["bottlenecks"], list):
            process["bottlenecks"] = []
            
        process["bottlenecks"].append(bottleneck)
        
        # Keep only recent bottlenecks
        if len(process["bottlenecks"]) > 10:
            process["bottlenecks"] = process["bottlenecks"][-10:]
        
        return True
    
    async def add_process_dependency(self, process_id: str, 
                                  dependency_id: str,
                                  importance: float = 0.5) -> bool:
        """Add a dependency between cognitive processes"""
        if process_id not in self.cognitive_processes or dependency_id not in self.cognitive_processes:
            return False
            
        process = self.cognitive_processes[process_id]
        
        # Add dependency
        dependency = {
            "process_id": dependency_id,
            "importance": importance
        }
        
        if not isinstance(process["dependencies"], list):
            process["dependencies"] = []
            
        # Check if dependency already exists
        for dep in process["dependencies"]:
            if dep["process_id"] == dependency_id:
                # Update importance
                dep["importance"] = importance
                return True
                
        # Add new dependency
        process["dependencies"].append(dependency)
        
        return True
    
    #--------------------------------------------------------------------------
    # Mental Model Methods
    #--------------------------------------------------------------------------
    
    async def create_mental_model(self, name: str, domain: str,
                               confidence: float = 0.5,
                               complexity: float = 0.5) -> str:
        """Create a new mental model"""
        # Generate model ID
        model_id = f"model_{self.next_model_id}"
        self.next_model_id += 1
        
        # Create model
        model = {
            "id": model_id,
            "name": name,
            "domain": domain,
            "confidence": confidence,
            "complexity": complexity,
            "elements": {},
            "relations": {},
            "last_updated": datetime.datetime.now().isoformat(),
            "last_used": datetime.datetime.now().isoformat(),
            "usage_count": 0,
            "accuracy_history": []
        }
        
        # Add to models
        self.mental_models[model_id] = model
        
        return model_id
    
    async def add_mental_model_element(self, model_id: str, key: str,
                                    description: str,
                                    importance: float = 0.5) -> bool:
        """Add an element to a mental model"""
        if model_id not in self.mental_models:
            return False
            
        model = self.mental_models[model_id]
        
        # Add element
        model["elements"][key] = {
            "description": description,
            "importance": importance,
            "added": datetime.datetime.now().isoformat()
        }
        
        # Update last modified
        model["last_updated"] = datetime.datetime.now().isoformat()
        
        return True
    
    async def add_mental_model_relation(self, model_id: str, 
                                     source: str, target: str,
                                     type: str, strength: float = 0.5) -> bool:
        """Add a relation between elements in a mental model"""
        if model_id not in self.mental_models:
            return False
            
        model = self.mental_models[model_id]
        
        # Check if elements exist
        if source not in model["elements"] or target not in model["elements"]:
            return False
            
        # Add relation
        relation_key = f"{source}_{target}"
        model["relations"][relation_key] = {
            "source": source,
            "target": target,
            "type": type,
            "strength": strength,
            "added": datetime.datetime.now().isoformat()
        }
        
        # Update last modified
        model["last_updated"] = datetime.datetime.now().isoformat()
        
        return True
    
    async def record_mental_model_usage(self, model_id: str, 
                                     accuracy: Optional[float] = None) -> bool:
        """Record usage of a mental model"""
        if model_id not in self.mental_models:
            return False
            
        model = self.mental_models[model_id]
        
        # Update usage stats
        model["last_used"] = datetime.datetime.now().isoformat()
        model["usage_count"] += 1
        
        # Record accuracy if provided
        if accuracy is not None:
            model["accuracy_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "accuracy": accuracy
            })
            
            # Update overall confidence based on accuracy history
            if model["accuracy_history"]:
                avg_accuracy = sum(entry["accuracy"] for entry in model["accuracy_history"]) / len(model["accuracy_history"])
                model["confidence"] = (model["confidence"] * 0.7) + (avg_accuracy * 0.3)
        
        return True
    
    #--------------------------------------------------------------------------
    # System Integration and Utility Methods
    #--------------------------------------------------------------------------
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning process"""
        return {
            "feature_importance": self.feature_importance,
            "algorithm_performance": self.algorithm_performance,
            "learning_cycles": self.learning_cycles,
            "learning_rate": self.meta_parameters["learning_rate"],
            "exploration_rate": self._calculate_exploration_rate(),
            "convergence": self._check_convergence() if len(self.feature_history) >= self.meta_parameters["min_samples_required"] else False
        }
    
    async def get_meta_stats(self) -> Dict[str, Any]:
        """Get statistics about the meta-cognitive system"""
        return {
            "monitored_systems": list(self.monitored_systems.keys()),
            "resource_allocation": self.resource_allocation,
            "cognitive_processes": len(self.cognitive_processes),
            "mental_models": len(self.mental_models),
            "insights": len(self.insights),
            "reflections": len(self.reflections),
            "improvement_plans": len(self.improvement_plans),
            "cognitive_cycle_count": self.cognitive_cycle_count,
            "system_metrics": self.system_metrics,
            "meta_parameters": self.meta_parameters,
            "error_logs": len(self.error_logs),
            "attention_focus": self.attention_focus,
            "overall_effectiveness": self._calculate_overall_effectiveness()
        }
    
    async def get_insights(self, limit: int = 10, system: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get metacognitive insights, optionally filtered by system"""
        filtered_insights = self.insights
        
        if system:
            filtered_insights = [i for i in self.insights if i.get("system") == system]
            
        sorted_insights = sorted(filtered_insights, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return sorted_insights[:limit]
    
    async def get_reflections(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent reflections"""
        sorted_reflections = sorted(self.reflections, key=lambda x: x.get("timestamp", ""), reverse=True)
        return sorted_reflections[:limit]
    
    async def get_improvement_plans(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get improvement plans"""
        sorted_plans = sorted(self.improvement_plans, key=lambda x: x.get("timestamp", ""), reverse=True)
        return sorted_plans[:limit]
    
    async def get_performance_history(self, system: Optional[str] = None, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance history, optionally for a specific system"""
        if system and system in self.performance_history:
            history = self.performance_history[system][-limit:]
            return {system: history}
            
        history = {}
        for sys_name, sys_history in self.performance_history.items():
            history[sys_name] = sys_history[-limit:]
            
        return history
    
    async def get_cognitive_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get all cognitive processes"""
        return self.cognitive_processes
    
    async def get_mental_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all mental models"""
        return self.mental_models
    
    async def save_state(self, file_path: str) -> bool:
        """Save current state to file"""
        try:
            state = {
                "feature_importance": self.feature_importance,
                "algorithm_performance": self.algorithm_performance,
                "learning_cycles": self.learning_cycles,
                "resource_allocation": self.resource_allocation,
                "performance_history": self.performance_history,
                "cognitive_processes": self.cognitive_processes,
                "mental_models": self.mental_models,
                "insights": self.insights,
                "reflections": self.reflections,
                "improvement_plans": self.improvement_plans,
                "meta_parameters": self.meta_parameters,
                "system_metrics": self.system_metrics,
                "cognitive_cycle_count": self.cognitive_cycle_count,
                "next_process_id": self.next_process_id,
                "next_model_id": self.next_model_id,
                "next_insight_id": self.next_insight_id,
                "next_reflection_id": self.next_reflection_id,
                "error_logs": self.error_logs,
                "attention_focus": self.attention_focus,
                "last_evaluation_time": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
                "last_reflection_time": self.last_reflection_time.isoformat() if self.last_reflection_time else None,
                "last_parameter_optimization_cycle": self.last_parameter_optimization_cycle
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"MetaCore state saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving MetaCore state: {str(e)}")
            return False
        
    async def load_state(self, file_path: str) -> bool:
        """Load state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load state components
            self.feature_importance = state["feature_importance"]
            self.algorithm_performance = state["algorithm_performance"]
            self.learning_cycles = state["learning_cycles"]
            self.resource_allocation = state["resource_allocation"]
            self.performance_history = state["performance_history"]
            self.cognitive_processes = state["cognitive_processes"]
            self.mental_models = state["mental_models"]
            self.insights = state["insights"]
            self.reflections = state["reflections"]
            self.improvement_plans = state["improvement_plans"]
            self.meta_parameters = state["meta_parameters"]
            self.system_metrics = state["system_metrics"]
            self.cognitive_cycle_count = state["cognitive_cycle_count"]
            self.next_process_id = state["next_process_id"]
            self.next_model_id = state["next_model_id"]
            self.next_insight_id = state["next_insight_id"]
            self.next_reflection_id = state["next_reflection_id"]
            
            # Load new components
            self.error_logs = state.get("error_logs", [])
            self.attention_focus = state.get("attention_focus")
            self.last_parameter_optimization_cycle = state.get("last_parameter_optimization_cycle", 0)
            
            # Parse datetime objects
            if "last_evaluation_time" in state and state["last_evaluation_time"]:
                self.last_evaluation_time = datetime.datetime.fromisoformat(state["last_evaluation_time"])
            if "last_reflection_time" in state and state["last_reflection_time"]:
                self.last_reflection_time = datetime.datetime.fromisoformat(state["last_reflection_time"])
            
            logger.info(f"MetaCore state loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading MetaCore state: {str(e)}")
            return False
