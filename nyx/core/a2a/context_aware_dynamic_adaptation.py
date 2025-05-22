# nyx/core/a2a/context_aware_dynamic_adaptation.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareDynamicAdaptation(ContextAwareModule):
    """
    Advanced Dynamic Adaptation System with full context distribution capabilities
    """
    
    def __init__(self, original_adaptation_system):
        super().__init__("dynamic_adaptation")
        self.original_system = original_adaptation_system
        self.context_subscriptions = [
            "performance_update", "strategy_change", "goal_progress",
            "emotional_state_update", "memory_retrieval_complete",
            "experience_sharing_update", "identity_evolution_update",
            "prediction_error", "system_bottleneck", "user_feedback"
        ]
        
        # Track adaptation state
        self.current_adaptation_context = {}
        self.performance_trend = {}
        self.strategy_effectiveness = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize adaptation processing for this context"""
        logger.debug(f"DynamicAdaptation received context for user: {context.user_id}")
        
        # Analyze current system state for adaptation needs
        system_state = await self._analyze_system_state(context)
        
        # Determine if adaptation is needed
        adaptation_needed = await self._assess_adaptation_need(context, system_state)
        
        # Get current strategy and performance
        current_strategy = self.original_system.context.current_strategy_id
        current_performance = await self._gather_performance_metrics(context)
        
        # Send initial adaptation context to other modules
        await self.send_context_update(
            update_type="adaptation_context_available",
            data={
                "current_strategy": current_strategy,
                "performance_metrics": current_performance,
                "system_state": system_state,
                "adaptation_needed": adaptation_needed,
                "adaptation_readiness": await self._calculate_adaptation_readiness(context)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect adaptation"""
        
        if update.update_type == "performance_update":
            # Performance changes trigger adaptation evaluation
            performance_data = update.data
            metrics = performance_data.get("metrics", {})
            
            # Update performance tracking
            await self._update_performance_tracking(metrics)
            
            # Check if performance degradation requires immediate adaptation
            if await self._detect_performance_degradation(metrics):
                await self._trigger_urgent_adaptation(metrics)
        
        elif update.update_type == "prediction_error":
            # Prediction errors may require strategy adjustment
            error_data = update.data
            prediction_error = error_data.get("prediction_error", 0.0)
            
            if prediction_error > self.original_system.prediction_error_threshold:
                # Adapt based on prediction error
                await self._adapt_from_prediction_error(error_data)
        
        elif update.update_type == "goal_progress":
            # Goal progress affects adaptation priorities
            goal_data = update.data
            await self._incorporate_goal_progress(goal_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state influences adaptation decisions
            emotional_data = update.data
            await self._incorporate_emotional_context(emotional_data)
        
        elif update.update_type == "experience_sharing_update":
            # Experience sharing effectiveness affects strategy
            experience_data = update.data
            await self._adapt_experience_parameters(experience_data)
        
        elif update.update_type == "identity_evolution_update":
            # Identity changes may require strategy adjustment
            identity_data = update.data
            await self._adapt_identity_parameters(identity_data)
        
        elif update.update_type == "system_bottleneck":
            # System bottlenecks require immediate attention
            bottleneck_data = update.data
            await self._address_system_bottleneck(bottleneck_data)
        
        elif update.update_type == "user_feedback":
            # Direct user feedback influences adaptation
            feedback_data = update.data
            await self._process_user_feedback_for_adaptation(feedback_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with adaptation awareness"""
        # Analyze input for adaptation triggers
        adaptation_triggers = await self._analyze_input_for_triggers(context)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Determine if immediate adaptation is needed
        immediate_adaptation = await self._check_immediate_adaptation_need(
            context, adaptation_triggers, messages
        )
        
        if immediate_adaptation:
            # Run adaptation cycle
            adaptation_result = await self._execute_context_aware_adaptation(context, messages)
            
            # Send adaptation updates
            await self.send_context_update(
                update_type="adaptation_executed",
                data={
                    "trigger": immediate_adaptation,
                    "result": adaptation_result,
                    "new_strategy": adaptation_result.get("selected_strategy", {}).get("id"),
                    "confidence": adaptation_result.get("strategy_confidence", 0.5)
                },
                priority=ContextPriority.HIGH
            )
            
            return {
                "adaptation_processed": True,
                "immediate_adaptation": True,
                "result": adaptation_result,
                "triggers": adaptation_triggers
            }
        
        # Regular processing with adaptation monitoring
        monitoring_result = await self._monitor_system_performance(context, messages)
        
        return {
            "adaptation_processed": True,
            "immediate_adaptation": False,
            "monitoring": monitoring_result,
            "triggers": adaptation_triggers
        }

    async def _calculate_context_volatility(self) -> float:
        """Calculate the volatility of the context over time - PRODUCTION VERSION"""
        if len(self.context.context_history) < 3:
            return 0.0  # Not enough history to calculate volatility
        
        # Calculate comprehensive volatility metrics
        volatility_components = {
            "structural_volatility": 0.0,
            "content_volatility": 0.0,
            "module_volatility": 0.0,
            "emotional_volatility": 0.0,
            "temporal_volatility": 0.0
        }
        
        # Analyze structural changes
        structural_changes = []
        for i in range(1, len(self.context.context_history)):
            curr = self.context.context_history[i]
            prev = self.context.context_history[i-1]
            
            # Compare structure
            curr_keys = set(curr.keys())
            prev_keys = set(prev.keys())
            
            added_keys = curr_keys - prev_keys
            removed_keys = prev_keys - curr_keys
            
            structural_change = (len(added_keys) + len(removed_keys)) / max(1, len(curr_keys.union(prev_keys)))
            structural_changes.append(structural_change)
        
        if structural_changes:
            volatility_components["structural_volatility"] = np.std(structural_changes) * 2
        
        # Analyze content volatility
        content_differences = []
        for i in range(1, len(self.context.context_history)):
            diff = await self._calculate_context_difference(
                RunContextWrapper(context=self.context), 
                self.context.context_history[i], 
                self.context.context_history[i-1]
            )
            content_differences.append(diff)
        
        if content_differences:
            # Calculate variance of differences
            mean_diff = sum(content_differences) / len(content_differences)
            variance = sum((diff - mean_diff) ** 2 for diff in content_differences) / len(content_differences)
            volatility_components["content_volatility"] = min(1.0, math.sqrt(variance) * 3.0)
        
        # Analyze module participation volatility
        module_sets = []
        for ctx in self.context.context_history:
            if "active_modules" in ctx:
                module_sets.append(set(ctx.get("active_modules", [])))
        
        if len(module_sets) >= 2:
            module_changes = []
            for i in range(1, len(module_sets)):
                intersection = module_sets[i].intersection(module_sets[i-1])
                union = module_sets[i].union(module_sets[i-1])
                if union:
                    stability = len(intersection) / len(union)
                    module_changes.append(1.0 - stability)
            
            if module_changes:
                volatility_components["module_volatility"] = sum(module_changes) / len(module_changes)
        
        # Analyze emotional volatility
        emotional_states = []
        for ctx in self.context.context_history:
            if "emotional_state" in ctx and ctx["emotional_state"]:
                emotional_states.append(ctx["emotional_state"])
        
        if len(emotional_states) >= 2:
            emotional_distances = []
            for i in range(1, len(emotional_states)):
                curr_emotions = emotional_states[i]
                prev_emotions = emotional_states[i-1]
                
                # Calculate emotional distance
                all_emotions = set(curr_emotions.keys()).union(set(prev_emotions.keys()))
                if all_emotions:
                    distance = sum(
                        abs(curr_emotions.get(e, 0) - prev_emotions.get(e, 0))
                        for e in all_emotions
                    ) / len(all_emotions)
                    emotional_distances.append(distance)
            
            if emotional_distances:
                volatility_components["emotional_volatility"] = sum(emotional_distances) / len(emotional_distances)
        
        # Analyze temporal volatility (rate of change)
        if len(self.context.context_history) >= 2:
            timestamps = []
            for ctx in self.context.context_history:
                if "timestamp" in ctx:
                    try:
                        timestamps.append(datetime.fromisoformat(ctx["timestamp"]))
                    except:
                        pass
            
            if len(timestamps) >= 2:
                time_gaps = []
                for i in range(1, len(timestamps)):
                    gap = (timestamps[i] - timestamps[i-1]).total_seconds()
                    time_gaps.append(gap)
                
                if time_gaps:
                    # High variance in time gaps = high temporal volatility
                    mean_gap = sum(time_gaps) / len(time_gaps)
                    if mean_gap > 0:
                        gap_variance = sum((gap - mean_gap) ** 2 for gap in time_gaps) / len(time_gaps)
                        normalized_variance = math.sqrt(gap_variance) / mean_gap
                        volatility_components["temporal_volatility"] = min(1.0, normalized_variance)
        
        # Combine volatility components with weights
        weights = {
            "structural_volatility": 0.15,
            "content_volatility": 0.35,
            "module_volatility": 0.20,
            "emotional_volatility": 0.20,
            "temporal_volatility": 0.10
        }
        
        total_volatility = sum(
            volatility_components[component] * weight
            for component, weight in weights.items()
        )
        
        # Apply smoothing based on history length
        history_factor = min(1.0, len(self.context.context_history) / 10)
        smoothed_volatility = total_volatility * (0.7 + 0.3 * history_factor)
        
        return min(1.0, smoothed_volatility)

    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze adaptation needs in current context"""
        # Get comprehensive system state
        system_analysis = await self._comprehensive_system_analysis(context)
        
        # Analyze adaptation patterns
        adaptation_patterns = await self._analyze_adaptation_patterns()
        
        # Evaluate strategy effectiveness
        strategy_evaluation = await self._evaluate_all_strategies(context)
        
        # Analyze cross-module impacts
        messages = await self.get_cross_module_messages()
        cross_module_analysis = await self._analyze_cross_module_impacts(messages)
        
        # Predict future adaptation needs
        future_needs = await self._predict_adaptation_needs(
            system_analysis, adaptation_patterns, cross_module_analysis
        )
        
        return {
            "system_analysis": system_analysis,
            "adaptation_patterns": adaptation_patterns,
            "strategy_evaluation": strategy_evaluation,
            "cross_module_impacts": cross_module_analysis,
            "future_adaptation_needs": future_needs,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize adaptation recommendations for response"""
        messages = await self.get_cross_module_messages()
        
        # Create adaptation synthesis
        adaptation_synthesis = {
            "current_strategy_status": await self._get_strategy_status_summary(),
            "adaptation_recommendations": await self._generate_adaptation_recommendations(context, messages),
            "performance_insights": await self._synthesize_performance_insights(context),
            "system_optimization_suggestions": await self._generate_optimization_suggestions(context),
            "adaptation_confidence": await self._calculate_overall_adaptation_confidence(context)
        }
        
        # Check if proactive adaptation is recommended
        proactive_adaptation = await self._evaluate_proactive_adaptation(adaptation_synthesis)
        
        if proactive_adaptation:
            await self.send_context_update(
                update_type="proactive_adaptation_recommended",
                data={
                    "recommendation": proactive_adaptation,
                    "confidence": adaptation_synthesis["adaptation_confidence"],
                    "rationale": adaptation_synthesis["adaptation_recommendations"]
                },
                priority=ContextPriority.HIGH
            )
        
        return adaptation_synthesis
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_system_state(self, context: SharedContext) -> Dict[str, Any]:
        """Comprehensive analysis of current system state"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "active_modules": list(context.active_modules),
            "processing_stage": context.processing_stage,
            "context_complexity": self._calculate_context_complexity(context)
        }
        
        # Analyze module outputs
        if context.module_outputs:
            state["module_performance"] = {}
            for stage, outputs in context.module_outputs.items():
                state["module_performance"][stage] = {
                    "responding_modules": list(outputs.keys()),
                    "response_rate": len(outputs) / max(1, len(context.active_modules))
                }
        
        # Analyze context updates
        state["context_update_rate"] = len(context.context_updates) / max(1, 
            (datetime.now() - context.created_at).total_seconds())
        
        # Calculate system load
        state["system_load"] = await self._calculate_system_load(context)
        
        return state
    
    async def _assess_adaptation_need(self, context: SharedContext, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess whether adaptation is needed based on context and state"""
        need_factors = {
            "performance_degradation": False,
            "context_change": False,
            "strategy_ineffective": False,
            "system_overload": False,
            "user_dissatisfaction": False
        }
        
        # Check performance trends
        if hasattr(self, 'performance_trend') and self.performance_trend:
            declining_metrics = [m for m, trend in self.performance_trend.items() 
                               if trend.get("direction") == "declining"]
            need_factors["performance_degradation"] = len(declining_metrics) > 2
        
        # Check context complexity change
        complexity = system_state.get("context_complexity", 0.5)
        if hasattr(self, 'last_complexity'):
            complexity_change = abs(complexity - self.last_complexity)
            need_factors["context_change"] = complexity_change > 0.3
        self.last_complexity = complexity
        
        # Check strategy effectiveness
        current_strategy = self.original_system.context.current_strategy_id
        if current_strategy in self.strategy_effectiveness:
            effectiveness = self.strategy_effectiveness[current_strategy]
            need_factors["strategy_ineffective"] = effectiveness < 0.4
        
        # Check system load
        system_load = system_state.get("system_load", 0.5)
        need_factors["system_overload"] = system_load > 0.8
        
        # Calculate overall need
        need_score = sum(1 for factor, needed in need_factors.items() if needed) / len(need_factors)
        
        return {
            "adaptation_needed": need_score > 0.3,
            "need_score": need_score,
            "need_factors": need_factors,
            "urgency": "high" if need_score > 0.6 else "medium" if need_score > 0.3 else "low"
        }
    
    async def _gather_performance_metrics(self, context: SharedContext) -> Dict[str, float]:
        """Gather comprehensive performance metrics from context"""
        metrics = {
            "response_quality": 0.5,
            "processing_speed": 0.5,
            "context_coherence": 0.5,
            "module_coordination": 0.5,
            "adaptation_effectiveness": 0.5
        }
        
        # Calculate response quality from module outputs
        if context.module_outputs:
            successful_outputs = sum(
                1 for outputs in context.module_outputs.values()
                for output in outputs.values()
                if output and not isinstance(output, dict) or not output.get("error")
            )
            total_outputs = sum(len(outputs) for outputs in context.module_outputs.values())
            metrics["response_quality"] = successful_outputs / max(1, total_outputs)
        
        # Calculate processing speed
        processing_time = (datetime.now() - context.created_at).total_seconds()
        expected_time = 2.0  # Expected processing time in seconds
        metrics["processing_speed"] = min(1.0, expected_time / max(0.1, processing_time))
        
        # Calculate context coherence
        metrics["context_coherence"] = await self._calculate_context_coherence(context)
        
        # Calculate module coordination
        if context.module_messages:
            message_count = sum(len(msgs) for msgs in context.module_messages.values())
            active_module_count = len(context.active_modules)
            expected_messages = active_module_count * 2  # Expected cross-module messages
            metrics["module_coordination"] = min(1.0, message_count / max(1, expected_messages))
        
        # Get adaptation effectiveness from history
        if hasattr(self.original_system, 'context') and self.original_system.context.strategy_history:
            recent_strategies = self.original_system.context.strategy_history[-5:]
            if recent_strategies:
                # Calculate how often we've had to change strategies (stability)
                unique_strategies = len(set(s["strategy_id"] for s in recent_strategies))
                metrics["adaptation_effectiveness"] = 1.0 - (unique_strategies - 1) / len(recent_strategies)
        
        return metrics
    
    async def _calculate_adaptation_readiness(self, context: SharedContext) -> float:
        """Calculate how ready the system is for adaptation"""
        readiness_factors = []
        
        # Factor 1: Time since last adaptation
        if hasattr(self.original_system, 'context') and self.original_system.context.strategy_history:
            last_adaptation = self.original_system.context.strategy_history[-1] if self.original_system.context.strategy_history else None
            if last_adaptation and "timestamp" in last_adaptation:
                time_since = (datetime.now() - datetime.fromisoformat(last_adaptation["timestamp"])).total_seconds()
                # Ready after 5 minutes, fully ready after 30 minutes
                time_readiness = min(1.0, time_since / 1800)
                readiness_factors.append(time_readiness)
        else:
            readiness_factors.append(1.0)  # Ready if no history
        
        # Factor 2: System stability
        if hasattr(self, 'performance_trend'):
            stable_metrics = sum(1 for trend in self.performance_trend.values() 
                               if trend.get("direction") == "stable")
            stability_readiness = stable_metrics / max(1, len(self.performance_trend))
            readiness_factors.append(stability_readiness)
        else:
            readiness_factors.append(0.5)  # Neutral if no trend data
        
        # Factor 3: Context completeness
        context_completeness = len(context.active_modules) / 10  # Assume 10 modules is complete
        readiness_factors.append(min(1.0, context_completeness))
        
        return sum(readiness_factors) / len(readiness_factors) if readiness_factors else 0.5
    
    async def _update_performance_tracking(self, metrics: Dict[str, float]):
        """Update performance tracking with new metrics"""
        for metric_name, value in metrics.items():
            if metric_name not in self.performance_trend:
                self.performance_trend[metric_name] = {
                    "values": [],
                    "direction": "stable",
                    "trend_strength": 0.0
                }
            
            # Add new value
            trend_data = self.performance_trend[metric_name]
            trend_data["values"].append(value)
            
            # Keep only recent values (last 10)
            if len(trend_data["values"]) > 10:
                trend_data["values"] = trend_data["values"][-10:]
            
            # Calculate trend if enough data
            if len(trend_data["values"]) >= 3:
                # Simple linear regression for trend
                n = len(trend_data["values"])
                x = list(range(n))
                y = trend_data["values"]
                
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                
                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                
                if denominator > 0:
                    slope = numerator / denominator
                    
                    # Determine direction
                    if abs(slope) < 0.01:
                        trend_data["direction"] = "stable"
                    elif slope > 0:
                        trend_data["direction"] = "improving"
                    else:
                        trend_data["direction"] = "declining"
                    
                    trend_data["trend_strength"] = min(1.0, abs(slope) * 10)
    
    async def _detect_performance_degradation(self, metrics: Dict[str, float]) -> bool:
        """Detect if performance is degrading significantly"""
        degradation_indicators = 0
        
        # Check individual metrics against thresholds
        critical_thresholds = {
            "response_quality": 0.3,
            "processing_speed": 0.2,
            "context_coherence": 0.3,
            "module_coordination": 0.2
        }
        
        for metric, threshold in critical_thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                degradation_indicators += 1
        
        # Check trends
        declining_trends = sum(1 for trend in self.performance_trend.values()
                             if trend.get("direction") == "declining" and trend.get("trend_strength", 0) > 0.5)
        
        if declining_trends >= 2:
            degradation_indicators += 1
        
        return degradation_indicators >= 2
    
    async def _trigger_urgent_adaptation(self, metrics: Dict[str, float]):
        """Trigger urgent adaptation due to performance issues"""
        logger.warning(f"Triggering urgent adaptation due to performance degradation: {metrics}")
        
        # Create urgent adaptation context
        adaptation_context = {
            "trigger": "performance_degradation",
            "urgency": "critical",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Run adaptation with urgency
        try:
            result = await self.original_system.adaptation_cycle(
                adaptation_context,
                metrics
            )
            
            # Send critical update
            await self.send_context_update(
                update_type="urgent_adaptation_completed",
                data={
                    "trigger": "performance_degradation",
                    "result": result,
                    "metrics_before": metrics
                },
                priority=ContextPriority.CRITICAL
            )
            
        except Exception as e:
            logger.error(f"Error in urgent adaptation: {e}")
    
    async def _adapt_from_prediction_error(self, error_data: Dict[str, Any]):
        """Adapt based on prediction error - PRODUCTION VERSION"""
        prediction_error = error_data.get("prediction_error", 0.0)
        error_details = error_data.get("error_details", {})
        prediction_id = error_data.get("prediction_id")
        prediction_type = error_details.get("type", "general")
        
        # Detailed error analysis
        error_analysis = {
            "error_magnitude": prediction_error,
            "error_category": self._categorize_prediction_error(prediction_error),
            "error_pattern": await self._analyze_error_pattern(prediction_id, error_details),
            "adaptive_response": {}
        }
        
        # Categorize error type
        if prediction_error > 0.8:
            error_category = "catastrophic"
        elif prediction_error > 0.6:
            error_category = "significant"
        elif prediction_error > 0.4:
            error_category = "moderate"
        else:
            error_category = "minor"
        
        error_analysis["error_category"] = error_category
        
        # Analyze error patterns
        if not hasattr(self, 'prediction_error_history'):
            self.prediction_error_history = []
        
        self.prediction_error_history.append({
            "timestamp": datetime.now().isoformat(),
            "prediction_id": prediction_id,
            "error": prediction_error,
            "type": prediction_type,
            "details": error_details,
            "current_strategy": self.original_system.context.current_strategy_id
        })
        
        # Keep history limited
        if len(self.prediction_error_history) > 50:
            self.prediction_error_history = self.prediction_error_history[-50:]
        
        # Identify error patterns
        error_pattern = await self._identify_prediction_error_patterns()
        
        # Determine adaptive response based on error analysis
        if error_category == "catastrophic":
            # Immediate major adaptation
            adaptive_response = {
                "action": "major_strategy_shift",
                "urgency": "immediate",
                "parameters": {
                    "increase_exploration": 0.3,
                    "increase_adaptation_rate": 0.2,
                    "reduce_confidence_threshold": 0.2
                }
            }
        elif error_category == "significant" and error_pattern.get("recurring", False):
            # Recurring significant errors need strategy adjustment
            adaptive_response = {
                "action": "strategy_adjustment",
                "urgency": "high",
                "parameters": {
                    "increase_exploration": 0.15,
                    "adjust_risk_tolerance": 0.1,
                    "modify_prediction_horizon": -0.2
                }
            }
        elif error_category == "moderate":
            # Parameter tuning
            adaptive_response = {
                "action": "parameter_tuning",
                "urgency": "normal",
                "parameters": {
                    "fine_tune_exploration": 0.05,
                    "adjust_learning_rate": 0.05
                }
            }
        else:
            # Minor adjustment or monitoring
            adaptive_response = {
                "action": "monitor",
                "urgency": "low",
                "parameters": {}
            }
        
        error_analysis["adaptive_response"] = adaptive_response
        
        # Execute adaptation based on response
        if adaptive_response["action"] != "monitor":
            try:
                # Prepare adaptation context
                adaptation_context = {
                    "trigger": "prediction_error",
                    "error_analysis": error_analysis,
                    "current_performance": await self._gather_performance_metrics(self.current_context),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Run adaptation with error-specific parameters
                result = await self.original_system.adaptation_cycle(
                    adaptation_context,
                    {"prediction_error": prediction_error}
                )
                
                # Track adaptation result
                if hasattr(result, "selected_strategy") and result["selected_strategy"]:
                    error_analysis["adaptation_result"] = {
                        "new_strategy": result["selected_strategy"]["id"],
                        "confidence": result.get("strategy_confidence", 0.5)
                    }
                
                # Send detailed update
                await self.send_context_update(
                    update_type="prediction_error_adaptation_complete",
                    data={
                        "error_data": error_data,
                        "error_analysis": error_analysis,
                        "adaptation_result": result
                    },
                    priority=ContextPriority.HIGH if error_category in ["catastrophic", "significant"] else ContextPriority.NORMAL
                )
                
            except Exception as e:
                logger.error(f"Error in prediction error adaptation: {e}")
                # Fallback to monitoring
                error_analysis["adaptation_result"] = {"error": str(e), "fallback": "monitoring"}
        
        # Store analysis for future reference
        self.last_prediction_error_analysis = error_analysis

    def _categorize_prediction_error(self, error: float) -> str:
        """Categorize prediction error magnitude"""
        if error > 0.8:
            return "catastrophic"
        elif error > 0.6:
            return "significant"
        elif error > 0.4:
            return "moderate"
        elif error > 0.2:
            return "minor"
        else:
            return "negligible"

    async def _analyze_error_pattern(self, prediction_id: str, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in prediction errors"""
        pattern_analysis = {
            "recurring": False,
            "pattern_type": "isolated",
            "frequency": 0,
            "common_factors": []
        }
        
        if not hasattr(self, 'prediction_error_history'):
            return pattern_analysis
        
        # Find similar errors
        error_type = error_details.get("type", "general")
        similar_errors = [
            e for e in self.prediction_error_history
            if e.get("type") == error_type and e.get("error", 0) > 0.4
        ]
        
        if len(similar_errors) >= 3:
            pattern_analysis["recurring"] = True
            pattern_analysis["frequency"] = len(similar_errors)
            
            # Identify common factors
            common_strategies = {}
            common_contexts = {}
            
            for error in similar_errors:
                strategy = error.get("current_strategy")
                if strategy:
                    common_strategies[strategy] = common_strategies.get(strategy, 0) + 1
                
                # Extract context features
                details = error.get("details", {})
                for key, value in details.items():
                    if key not in ["timestamp", "id"]:
                        context_key = f"{key}:{value}"
                        common_contexts[context_key] = common_contexts.get(context_key, 0) + 1
            
            # Find most common factors
            if common_strategies:
                most_common_strategy = max(common_strategies.items(), key=lambda x: x[1])
                if most_common_strategy[1] >= len(similar_errors) * 0.6:
                    pattern_analysis["common_factors"].append(f"strategy:{most_common_strategy[0]}")
                    pattern_analysis["pattern_type"] = "strategy_related"
            
            if common_contexts:
                for context, count in common_contexts.items():
                    if count >= len(similar_errors) * 0.5:
                        pattern_analysis["common_factors"].append(context)
                
                if len(pattern_analysis["common_factors"]) > 2:
                    pattern_analysis["pattern_type"] = "context_related"
        
        return pattern_analysis
    
    async def _incorporate_goal_progress(self, goal_data: Dict[str, Any]):
        """Incorporate goal progress into adaptation decisions"""
        # Extract goal performance
        goals_completed = goal_data.get("goals_completed", 0)
        goals_blocked = goal_data.get("goals_blocked", 0)
        goals_active = goal_data.get("goals_active", 0)
        
        # Calculate goal success rate
        if goals_completed + goals_blocked > 0:
            goal_success_rate = goals_completed / (goals_completed + goals_blocked)
        else:
            goal_success_rate = 0.5  # Neutral if no completed/blocked goals
        
        # Update strategy effectiveness based on goal performance
        current_strategy = self.original_system.context.current_strategy_id
        
        if current_strategy not in self.strategy_effectiveness:
            self.strategy_effectiveness[current_strategy] = []
        
        self.strategy_effectiveness[current_strategy].append(goal_success_rate)
        
        # Keep only recent effectiveness scores
        if len(self.strategy_effectiveness[current_strategy]) > 10:
            self.strategy_effectiveness[current_strategy] = self.strategy_effectiveness[current_strategy][-10:]
    
    async def _incorporate_emotional_context(self, emotional_data: Dict[str, Any]):
        """Incorporate emotional state into adaptation"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, intensity = dominant_emotion
        
        # Adjust adaptation parameters based on emotion
        self.current_adaptation_context["emotional_influence"] = {
            "dominant_emotion": emotion_name,
            "intensity": intensity,
            "timestamp": datetime.now().isoformat()
        }
        
        # High negative emotions may warrant more conservative strategies
        negative_emotions = ["Frustration", "Anger", "Fear", "Anxiety"]
        if emotion_name in negative_emotions and intensity > 0.7:
            self.current_adaptation_context["prefer_conservative"] = True
    
    async def _adapt_experience_parameters(self, experience_data: Dict[str, Any]):
        """Adapt experience sharing parameters based on effectiveness"""
        experience_utility = experience_data.get("experience_utility", 0.5)
        user_engagement = experience_data.get("user_engagement", 0.5)
        
        # Track experience effectiveness
        if "experience_effectiveness" not in self.current_adaptation_context:
            self.current_adaptation_context["experience_effectiveness"] = []
        
        self.current_adaptation_context["experience_effectiveness"].append({
            "utility": experience_utility,
            "engagement": user_engagement,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent data
        if len(self.current_adaptation_context["experience_effectiveness"]) > 20:
            self.current_adaptation_context["experience_effectiveness"] = \
                self.current_adaptation_context["experience_effectiveness"][-20:]
        
        # If consistently low utility, may need to adapt
        recent_utilities = [e["utility"] for e in self.current_adaptation_context["experience_effectiveness"][-5:]]
        if recent_utilities and sum(recent_utilities) / len(recent_utilities) < 0.3:
            # Low utility - consider adaptation
            self.current_adaptation_context["experience_adaptation_needed"] = True
    
    async def _adapt_identity_parameters(self, identity_data: Dict[str, Any]):
        """Adapt identity evolution parameters based on coherence"""
        identity_coherence = identity_data.get("identity_coherence", 0.5)
        evolution_rate = identity_data.get("evolution_rate", 0.2)
        
        # Track identity coherence over time
        if "identity_tracking" not in self.current_adaptation_context:
            self.current_adaptation_context["identity_tracking"] = []
        
        self.current_adaptation_context["identity_tracking"].append({
            "coherence": identity_coherence,
            "evolution_rate": evolution_rate,
            "timestamp": datetime.now().isoformat()
        })
        
        # Detect if identity is becoming incoherent
        recent_coherence = [t["coherence"] for t in self.current_adaptation_context["identity_tracking"][-5:]]
        if recent_coherence and sum(recent_coherence) / len(recent_coherence) < 0.4:
            # Low coherence - may need to slow evolution
            self.current_adaptation_context["identity_adaptation_needed"] = True

    async def _identify_prediction_error_patterns(self) -> Dict[str, Any]:
        """Identify comprehensive patterns in prediction errors"""
        if not hasattr(self, 'prediction_error_history') or len(self.prediction_error_history) < 5:
            return {"patterns_found": False}
        
        patterns = {
            "patterns_found": True,
            "error_trends": {},
            "strategy_performance": {},
            "error_clustering": [],
            "recommendations": []
        }
        
        # Analyze error trends over time
        recent_errors = self.prediction_error_history[-10:]
        error_values = [e["error"] for e in recent_errors]
        
        # Calculate trend
        if len(error_values) >= 5:
            first_half = error_values[:len(error_values)//2]
            second_half = error_values[len(error_values)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg * 1.2:
                patterns["error_trends"]["direction"] = "worsening"
                patterns["recommendations"].append("Prediction accuracy declining - increase exploration")
            elif second_avg < first_avg * 0.8:
                patterns["error_trends"]["direction"] = "improving"
            else:
                patterns["error_trends"]["direction"] = "stable"
        
        # Analyze by strategy
        for error_entry in self.prediction_error_history:
            strategy = error_entry.get("current_strategy", "unknown")
            if strategy not in patterns["strategy_performance"]:
                patterns["strategy_performance"][strategy] = {
                    "errors": [],
                    "avg_error": 0.0,
                    "error_count": 0
                }
            
            patterns["strategy_performance"][strategy]["errors"].append(error_entry["error"])
            patterns["strategy_performance"][strategy]["error_count"] += 1
        
        # Calculate averages and identify problematic strategies
        problematic_strategies = []
        for strategy, perf in patterns["strategy_performance"].items():
            if perf["errors"]:
                perf["avg_error"] = sum(perf["errors"]) / len(perf["errors"])
                
                if perf["avg_error"] > 0.6 and perf["error_count"] >= 3:
                    problematic_strategies.append(strategy)
        
        if problematic_strategies:
            patterns["recommendations"].append(
                f"Avoid strategies: {', '.join(problematic_strategies)} due to high prediction errors"
            )
        
        # Identify error clusters
        # Group errors by time proximity and magnitude
        error_clusters = []
        cluster_threshold = 300  # 5 minutes in seconds
        
        sorted_errors = sorted(self.prediction_error_history, key=lambda x: x["timestamp"])
        current_cluster = []
        
        for i, error in enumerate(sorted_errors):
            if not current_cluster:
                current_cluster.append(error)
            else:
                # Check time proximity
                last_time = datetime.fromisoformat(current_cluster[-1]["timestamp"])
                curr_time = datetime.fromisoformat(error["timestamp"])
                
                if (curr_time - last_time).total_seconds() <= cluster_threshold:
                    current_cluster.append(error)
                else:
                    # Save cluster if significant
                    if len(current_cluster) >= 2:
                        cluster_avg_error = sum(e["error"] for e in current_cluster) / len(current_cluster)
                        if cluster_avg_error > 0.5:
                            error_clusters.append({
                                "size": len(current_cluster),
                                "avg_error": cluster_avg_error,
                                "time_span": (
                                    datetime.fromisoformat(current_cluster[-1]["timestamp"]) -
                                    datetime.fromisoformat(current_cluster[0]["timestamp"])
                                ).total_seconds(),
                                "errors": current_cluster
                            })
                    
                    current_cluster = [error]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            cluster_avg_error = sum(e["error"] for e in current_cluster) / len(current_cluster)
            if cluster_avg_error > 0.5:
                error_clusters.append({
                    "size": len(current_cluster),
                    "avg_error": cluster_avg_error,
                    "errors": current_cluster
                })
        
        patterns["error_clustering"] = error_clusters
        
        if error_clusters:
            patterns["recommendations"].append(
                f"Detected {len(error_clusters)} error clusters - system may be unstable during certain conditions"
            )
        
        return patterns
    
    async def _address_system_bottleneck(self, bottleneck_data: Dict[str, Any]):
        """Address identified system bottlenecks"""
        bottleneck_type = bottleneck_data.get("type", "unknown")
        severity = bottleneck_data.get("severity", 0.5)
        affected_modules = bottleneck_data.get("affected_modules", [])
        
        logger.warning(f"System bottleneck detected: {bottleneck_type} (severity: {severity})")
        
        # Create adaptation to address bottleneck
        if severity > 0.7:
            # Severe bottleneck - immediate adaptation
            adaptation_context = {
                "trigger": "system_bottleneck",
                "bottleneck_type": bottleneck_type,
                "severity": severity,
                "affected_modules": affected_modules
            }
            
            # Prefer strategies that reduce system load
            self.current_adaptation_context["prefer_efficient_strategies"] = True
            
            # Send urgent update
            await self.send_context_update(
                update_type="bottleneck_adaptation_needed",
                data=bottleneck_data,
                priority=ContextPriority.CRITICAL
            )
    
    async def _process_user_feedback_for_adaptation(self, feedback_data: Dict[str, Any]):
        """Process user feedback to influence adaptation"""
        satisfaction = feedback_data.get("satisfaction", 0.5)
        feedback_type = feedback_data.get("type", "general")
        specific_issues = feedback_data.get("issues", [])
        
        # Track user satisfaction
        if "user_satisfaction_history" not in self.current_adaptation_context:
            self.current_adaptation_context["user_satisfaction_history"] = []
        
        self.current_adaptation_context["user_satisfaction_history"].append({
            "satisfaction": satisfaction,
            "type": feedback_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # If satisfaction is consistently low, trigger adaptation
        recent_satisfaction = [
            f["satisfaction"] for f in self.current_adaptation_context["user_satisfaction_history"][-5:]
        ]
        
        if recent_satisfaction and sum(recent_satisfaction) / len(recent_satisfaction) < 0.4:
            # Low satisfaction - adapt
            self.current_adaptation_context["user_driven_adaptation"] = True
            
            # Analyze specific issues to guide adaptation
            if "response_quality" in specific_issues:
                self.current_adaptation_context["focus_on_quality"] = True
            if "response_time" in specific_issues:
                self.current_adaptation_context["focus_on_speed"] = True
    
    async def _analyze_input_for_triggers(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze input for adaptation triggers"""
        triggers = {
            "complexity_spike": False,
            "new_domain": False,
            "performance_request": False,
            "explicit_adaptation": False
        }
        
        user_input = context.user_input.lower()
        
        # Check for complexity indicators
        complexity_words = ["complex", "difficult", "challenging", "intricate", "complicated"]
        triggers["complexity_spike"] = any(word in user_input for word in complexity_words)
        
        # Check for new domain indicators
        domain_words = ["new", "different", "change", "switch", "unfamiliar"]
        triggers["new_domain"] = any(word in user_input for word in domain_words)
        
        # Check for performance requests
        performance_words = ["faster", "better", "improve", "optimize", "enhance"]
        triggers["performance_request"] = any(word in user_input for word in performance_words)
        
        # Check for explicit adaptation requests
        adaptation_words = ["adapt", "adjust", "change strategy", "different approach"]
        triggers["explicit_adaptation"] = any(word in user_input for word in adaptation_words)
        
        return triggers
    
    async def _check_immediate_adaptation_need(self, 
                                             context: SharedContext,
                                             triggers: Dict[str, Any],
                                             messages: Dict[str, List[Dict]]) -> Optional[str]:
        """Check if immediate adaptation is needed"""
        # Check explicit triggers
        if triggers.get("explicit_adaptation"):
            return "explicit_request"
        
        # Check performance triggers
        if triggers.get("performance_request") and hasattr(self, 'performance_trend'):
            declining = sum(1 for trend in self.performance_trend.values() 
                          if trend.get("direction") == "declining")
            if declining >= 2:
                return "performance_improvement_needed"
        
        # Check context-based triggers
        if triggers.get("complexity_spike"):
            current_strategy = self.original_system.context.strategies.get(
                self.original_system.context.current_strategy_id, {}
            )
            if current_strategy.get("name") == "Conservative Strategy":
                return "complexity_requires_different_strategy"
        
        # Check cross-module emergency signals
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg.get("type") == "emergency_adaptation_needed":
                    return f"emergency_from_{module_name}"
        
        return None
    
    async def _execute_context_aware_adaptation(self, 
                                              context: SharedContext,
                                              messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Execute adaptation with full context awareness"""
        # Gather comprehensive context
        adaptation_context = {
            "user_input": context.user_input,
            "active_modules": list(context.active_modules),
            "processing_stage": context.processing_stage,
            "cross_module_context": self._summarize_cross_module_context(messages),
            "current_performance": await self._gather_performance_metrics(context),
            "adaptation_history": self.current_adaptation_context
        }
        
        # Add emotional context if available
        if context.emotional_state:
            adaptation_context["emotional_context"] = context.emotional_state
        
        # Add goal context if available
        if context.goal_context:
            adaptation_context["goal_context"] = context.goal_context
        
        # Run adaptation cycle
        result = await self.original_system.adaptation_cycle(
            adaptation_context,
            adaptation_context["current_performance"]
        )
        
        # Update tracking
        if "selected_strategy" in result:
            new_strategy_id = result["selected_strategy"].get("id")
            if new_strategy_id:
                # Track strategy change
                await self._track_strategy_change(
                    self.original_system.context.current_strategy_id,
                    new_strategy_id,
                    result.get("strategy_confidence", 0.5)
                )
        
        return result
    
    async def _monitor_system_performance(self, 
                                        context: SharedContext,
                                        messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Continuous performance monitoring"""
        # Gather current metrics
        current_metrics = await self._gather_performance_metrics(context)
        
        # Run monitoring
        monitoring_result = await self.original_system.monitor_performance(current_metrics)
        
        # Analyze trends
        if "trends" in monitoring_result:
            for metric, trend in monitoring_result["trends"].items():
                if trend.get("direction") == "declining" and trend.get("magnitude", 0) > 0.2:
                    # Significant decline detected
                    await self.send_context_update(
                        update_type="performance_decline_detected",
                        data={
                            "metric": metric,
                            "trend": trend,
                            "current_value": current_metrics.get(metric, 0)
                        },
                        priority=ContextPriority.HIGH
                    )
        
        return monitoring_result
    
    async def _comprehensive_system_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Perform comprehensive system analysis"""
        analysis = {
            "system_health": await self._analyze_system_health(context),
            "module_performance": await self._analyze_module_performance(context),
            "adaptation_history": await self._analyze_adaptation_history(),
            "bottlenecks": await self._identify_bottlenecks(context),
            "optimization_opportunities": await self._identify_optimization_opportunities(context)
        }
        
        return analysis
    
    async def _analyze_adaptation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in adaptation history"""
        if not hasattr(self.original_system, 'context') or not self.original_system.context.strategy_history:
            return {"patterns": [], "insights": []}
        
        history = self.original_system.context.strategy_history
        patterns = {
            "strategy_frequency": {},
            "strategy_sequences": [],
            "adaptation_triggers": {},
            "time_patterns": []
        }
        
        # Analyze strategy frequency
        for entry in history:
            strategy_id = entry.get("strategy_id")
            if strategy_id:
                patterns["strategy_frequency"][strategy_id] = \
                    patterns["strategy_frequency"].get(strategy_id, 0) + 1
        
        # Analyze strategy sequences
        if len(history) >= 2:
            for i in range(len(history) - 1):
                sequence = (history[i].get("strategy_id"), history[i+1].get("strategy_id"))
                patterns["strategy_sequences"].append(sequence)
        
        # Generate insights
        insights = []
        
        # Most used strategy
        if patterns["strategy_frequency"]:
            most_used = max(patterns["strategy_frequency"].items(), key=lambda x: x[1])
            insights.append(f"Most frequently used strategy: {most_used[0]} ({most_used[1]} times)")
        
        # Strategy cycling detection
        if patterns["strategy_sequences"]:
            # Check for repeated sequences
            sequence_counts = {}
            for seq in patterns["strategy_sequences"]:
                seq_str = f"{seq[0]}->{seq[1]}"
                sequence_counts[seq_str] = sequence_counts.get(seq_str, 0) + 1
            
            repeated_sequences = [seq for seq, count in sequence_counts.items() if count > 1]
            if repeated_sequences:
                insights.append(f"Detected strategy cycling patterns: {', '.join(repeated_sequences)}")
        
        patterns["insights"] = insights
        return patterns
    
    async def _evaluate_all_strategies(self, context: SharedContext) -> Dict[str, Any]:
        """Evaluate effectiveness of all available strategies"""
        evaluations = {}
        
        for strategy_id, strategy in self.original_system.context.strategies.items():
            # Get historical performance for this strategy
            if strategy_id in self.strategy_effectiveness:
                effectiveness_scores = self.strategy_effectiveness[strategy_id]
                if effectiveness_scores:
                    avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
                    
                    evaluations[strategy_id] = {
                        "name": strategy["name"],
                        "average_effectiveness": avg_effectiveness,
                        "sample_size": len(effectiveness_scores),
                        "recent_trend": self._calculate_recent_trend(effectiveness_scores),
                        "suitability_for_context": await self._evaluate_strategy_suitability(
                            strategy, context
                        )
                    }
                else:
                    evaluations[strategy_id] = {
                        "name": strategy["name"],
                        "average_effectiveness": 0.5,  # Unknown
                        "sample_size": 0,
                        "recent_trend": "unknown",
                        "suitability_for_context": await self._evaluate_strategy_suitability(
                            strategy, context
                        )
                    }
        
        return evaluations
    
    async def _analyze_cross_module_impacts(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze how different modules impact adaptation needs"""
        impacts = {
            "module_stress_levels": {},
            "inter_module_conflicts": [],
            "coordination_quality": 0.5,
            "emergent_patterns": []
        }
        
        # Analyze message patterns
        for module_name, module_messages in messages.items():
            # Count stress indicators
            stress_indicators = 0
            for msg in module_messages:
                msg_type = msg.get("type", "")
                if any(indicator in msg_type for indicator in ["error", "failure", "timeout", "overload"]):
                    stress_indicators += 1
                
                # Check for conflict indicators
                if "conflict" in msg_type or "incompatible" in msg_type:
                    impacts["inter_module_conflicts"].append({
                        "module": module_name,
                        "message": msg_type,
                        "timestamp": msg.get("timestamp")
                    })
            
            # Calculate stress level
            if module_messages:
                impacts["module_stress_levels"][module_name] = stress_indicators / len(module_messages)
        
        # Calculate coordination quality
        total_messages = sum(len(msgs) for msgs in messages.values())
        if total_messages > 0:
            # Higher message count indicates better coordination (to a point)
            impacts["coordination_quality"] = min(1.0, total_messages / 50)
        
        # Identify emergent patterns
        if total_messages > 10:
            # Look for message clusters
            message_types = []
            for module_messages in messages.values():
                for msg in module_messages:
                    message_types.append(msg.get("type", "unknown"))
            
            # Find common message types
            type_counts = {}
            for msg_type in message_types:
                type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
            
            # Patterns are message types that appear from multiple modules
            for msg_type, count in type_counts.items():
                if count > 2:  # Appears multiple times
                    impacts["emergent_patterns"].append({
                        "pattern": msg_type,
                        "frequency": count,
                        "percentage": count / total_messages
                    })
        
        return impacts
    
    async def _predict_adaptation_needs(self, 
                                      system_analysis: Dict[str, Any],
                                      adaptation_patterns: Dict[str, Any],
                                      cross_module_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future adaptation needs based on analysis"""
        predictions = {
            "likelihood_of_adaptation": 0.5,
            "predicted_timeframe": "unknown",
            "likely_triggers": [],
            "recommended_preparation": []
        }
        
        # Analyze system health trajectory
        system_health = system_analysis.get("system_health", {})
        if system_health.get("trend") == "declining":
            predictions["likelihood_of_adaptation"] += 0.2
            predictions["likely_triggers"].append("system_health_degradation")
        
        # Analyze adaptation frequency
        if adaptation_patterns.get("strategy_frequency"):
            total_adaptations = sum(adaptation_patterns["strategy_frequency"].values())
            if total_adaptations > 10:
                # Frequent adaptations suggest instability
                predictions["likelihood_of_adaptation"] += 0.1
                predictions["likely_triggers"].append("system_instability")
        
        # Analyze cross-module stress
        stress_levels = cross_module_analysis.get("module_stress_levels", {})
        high_stress_modules = [m for m, level in stress_levels.items() if level > 0.5]
        
        if len(high_stress_modules) > 2:
            predictions["likelihood_of_adaptation"] += 0.15
            predictions["likely_triggers"].append("multi_module_stress")
        
        # Determine timeframe
        if predictions["likelihood_of_adaptation"] > 0.7:
            predictions["predicted_timeframe"] = "imminent"
            predictions["recommended_preparation"].append("Pre-load alternative strategies")
        elif predictions["likelihood_of_adaptation"] > 0.5:
            predictions["predicted_timeframe"] = "soon"
            predictions["recommended_preparation"].append("Monitor key metrics closely")
        else:
            predictions["predicted_timeframe"] = "not_immediate"
        
        # Cap likelihood at 1.0
        predictions["likelihood_of_adaptation"] = min(1.0, predictions["likelihood_of_adaptation"])
        
        return predictions
    
    def _calculate_context_complexity(self, context: SharedContext) -> float:
        """Calculate complexity of current context"""
        complexity = 0.0
        
        # Factor 1: Number of active modules
        module_complexity = len(context.active_modules) / 15  # Assume 15 is max modules
        complexity += module_complexity * 0.3
        
        # Factor 2: Context update rate
        update_rate = len(context.context_updates) / max(1, (datetime.now() - context.created_at).total_seconds())
        complexity += min(1.0, update_rate * 10) * 0.2  # Scale update rate
        
        # Factor 3: Cross-module message density
        message_count = sum(len(msgs) for msgs in context.module_messages.values())
        message_density = message_count / max(1, len(context.active_modules))
        complexity += min(1.0, message_density / 10) * 0.2  # Scale density
        
        # Factor 4: Processing stage
        stage_complexity = {
            "input": 0.2,
            "processing": 0.5,
            "synthesis": 0.8,
            "output": 0.3
        }
        complexity += stage_complexity.get(context.processing_stage, 0.5) * 0.2
        
        # Factor 5: User input complexity (word count as proxy)
        word_count = len(context.user_input.split())
        input_complexity = min(1.0, word_count / 50)  # Cap at 50 words
        complexity += input_complexity * 0.1
        
        return min(1.0, complexity)
    
    async def _calculate_system_load(self, context: SharedContext) -> float:
        """Calculate current system load"""
        load_factors = []
        
        # Active modules vs total
        if hasattr(context, 'active_modules'):
            active_ratio = len(context.active_modules) / 15  # Assume 15 max
            load_factors.append(min(1.0, active_ratio))
        
        # Context updates per second
        if hasattr(context, 'context_updates') and hasattr(context, 'created_at'):
            time_elapsed = (datetime.now() - context.created_at).total_seconds()
            if time_elapsed > 0:
                updates_per_second = len(context.context_updates) / time_elapsed
                # Assume 10 updates/second is high load
                load_factors.append(min(1.0, updates_per_second / 10))
        
        # Module output completeness
        if hasattr(context, 'module_outputs') and hasattr(context, 'active_modules'):
            total_expected_outputs = len(context.active_modules) * len(context.module_outputs)
            actual_outputs = sum(len(outputs) for outputs in context.module_outputs.values())
            if total_expected_outputs > 0:
                output_ratio = actual_outputs / total_expected_outputs
                # Invert - more outputs = higher load
                load_factors.append(output_ratio)
        
        return sum(load_factors) / len(load_factors) if load_factors else 0.5
    
    async def _calculate_context_coherence(self, context: SharedContext) -> float:
        """Calculate how coherent the context is across modules"""
        coherence_factors = []
        
        # Check if modules are providing consistent information
        if context.emotional_state and context.goal_context:
            # Check for emotional-goal alignment
            emotional_valence = context.emotional_state.get("valence", 0)
            goal_progress = context.goal_context.get("progress", 0.5)
            
            # Positive emotions should correlate with good progress
            alignment = 1.0 - abs((emotional_valence + 1) / 2 - goal_progress)
            coherence_factors.append(alignment)
        
        # Check module response consistency
        if context.module_outputs:
            for stage, outputs in context.module_outputs.items():
                # Check if all expected modules responded
                response_rate = len(outputs) / max(1, len(context.active_modules))
                coherence_factors.append(response_rate)
        
        # Check temporal coherence (updates should be reasonably spaced)
        if len(context.context_updates) > 1:
            # Check for update clustering
            update_times = [u.timestamp for u in context.context_updates]
            time_diffs = []
            for i in range(1, len(update_times)):
                diff = (update_times[i] - update_times[i-1]).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                # Calculate variance in time differences
                mean_diff = sum(time_diffs) / len(time_diffs)
                variance = sum((d - mean_diff) ** 2 for d in time_diffs) / len(time_diffs)
                # Lower variance = more regular = more coherent
                coherence_from_timing = 1.0 / (1.0 + variance)
                coherence_factors.append(coherence_from_timing)
        
        return sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.5
    
    def _summarize_cross_module_context(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Summarize cross-module messages for adaptation context"""
        summary = {
            "total_messages": sum(len(msgs) for msgs in messages.values()),
            "active_modules": list(messages.keys()),
            "message_types": {},
            "priority_messages": []
        }
        
        # Count message types
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                msg_type = msg.get("type", "unknown")
                summary["message_types"][msg_type] = summary["message_types"].get(msg_type, 0) + 1
                
                # Extract high priority messages
                if msg.get("priority") in ["HIGH", "CRITICAL"]:
                    summary["priority_messages"].append({
                        "module": module_name,
                        "type": msg_type,
                        "data": msg.get("data", {})
                    })
        
        return summary
    
    async def _track_strategy_change(self, old_strategy: str, new_strategy: str, confidence: float):
        """Track strategy changes for analysis"""
        if not hasattr(self, 'strategy_change_log'):
            self.strategy_change_log = []
        
        self.strategy_change_log.append({
            "timestamp": datetime.now().isoformat(),
            "from_strategy": old_strategy,
            "to_strategy": new_strategy,
            "confidence": confidence,
            "context": self.current_adaptation_context.copy()
        })
        
        # Limit log size
        if len(self.strategy_change_log) > 50:
            self.strategy_change_log = self.strategy_change_log[-50:]
    
    async def _analyze_system_health(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze overall system health"""
        health_indicators = {
            "response_quality": await self._assess_response_quality(context),
            "module_coordination": await self._assess_module_coordination(context),
            "resource_usage": await self._assess_resource_usage(context),
            "error_rate": await self._assess_error_rate(context)
        }
        
        # Calculate overall health score
        health_score = sum(health_indicators.values()) / len(health_indicators)
        
        # Determine trend
        trend = "stable"
        if hasattr(self, 'previous_health_score'):
            if health_score > self.previous_health_score + 0.1:
                trend = "improving"
            elif health_score < self.previous_health_score - 0.1:
                trend = "declining"
        
        self.previous_health_score = health_score
        
        return {
            "score": health_score,
            "trend": trend,
            "indicators": health_indicators,
            "status": "healthy" if health_score > 0.7 else "degraded" if health_score < 0.3 else "moderate"
        }
    
    async def _analyze_module_performance(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze individual module performance"""
        module_performance = {}
        
        for module_name in context.active_modules:
            performance = {
                "response_rate": 0.0,
                "error_rate": 0.0,
                "message_activity": 0.0
            }
            
            # Check module outputs
            output_count = 0
            error_count = 0
            for stage_outputs in context.module_outputs.values():
                if module_name in stage_outputs:
                    output_count += 1
                    if isinstance(stage_outputs[module_name], dict) and "error" in stage_outputs[module_name]:
                        error_count += 1
            
            if len(context.module_outputs) > 0:
                performance["response_rate"] = output_count / len(context.module_outputs)
                performance["error_rate"] = error_count / max(1, output_count)
            
            # Check message activity
            if module_name in context.module_messages:
                performance["message_activity"] = len(context.module_messages[module_name]) / 10  # Normalize
            
            module_performance[module_name] = performance
        
        return module_performance
    
    async def _analyze_adaptation_history(self) -> Dict[str, Any]:
        """Analyze adaptation history for insights"""
        if not hasattr(self, 'strategy_change_log'):
            return {"total_changes": 0, "insights": []}
        
        analysis = {
            "total_changes": len(self.strategy_change_log),
            "change_frequency": 0.0,
            "most_stable_strategy": None,
            "most_transitions": [],
            "insights": []
        }
        
        if self.strategy_change_log:
            # Calculate change frequency (changes per hour)
            first_change = datetime.fromisoformat(self.strategy_change_log[0]["timestamp"])
            last_change = datetime.fromisoformat(self.strategy_change_log[-1]["timestamp"])
            time_span = (last_change - first_change).total_seconds() / 3600  # hours
            
            if time_span > 0:
                analysis["change_frequency"] = len(self.strategy_change_log) / time_span
            
            # Find most stable strategy (longest time without change FROM it)
            strategy_durations = {}
            for i in range(len(self.strategy_change_log) - 1):
                from_strategy = self.strategy_change_log[i]["to_strategy"]
                next_change_time = datetime.fromisoformat(self.strategy_change_log[i + 1]["timestamp"])
                current_time = datetime.fromisoformat(self.strategy_change_log[i]["timestamp"])
                duration = (next_change_time - current_time).total_seconds()
                
                if from_strategy not in strategy_durations:
                    strategy_durations[from_strategy] = []
                strategy_durations[from_strategy].append(duration)
            
            # Calculate average duration for each strategy
            avg_durations = {}
            for strategy, durations in strategy_durations.items():
                avg_durations[strategy] = sum(durations) / len(durations)
            
            if avg_durations:
                analysis["most_stable_strategy"] = max(avg_durations.items(), key=lambda x: x[1])[0]
            
            # Generate insights
            if analysis["change_frequency"] > 10:
                analysis["insights"].append("High adaptation frequency suggests system instability")
            elif analysis["change_frequency"] < 0.5:
                analysis["insights"].append("Low adaptation frequency indicates system stability")
        
        return analysis
    
    async def _identify_bottlenecks(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # Check for slow modules
        if context.module_outputs:
            for module_name in context.active_modules:
                response_count = sum(1 for outputs in context.module_outputs.values() 
                                   if module_name in outputs)
                expected_count = len(context.module_outputs)
                
                if expected_count > 0 and response_count / expected_count < 0.5:
                    bottlenecks.append({
                        "type": "slow_module",
                        "module": module_name,
                        "severity": 1.0 - (response_count / expected_count),
                        "impact": "reduced_responsiveness"
                    })
        
        # Check for message queue buildup
        total_messages = sum(len(msgs) for msgs in context.module_messages.values())
        if total_messages > 100:
            bottlenecks.append({
                "type": "message_overload",
                "severity": min(1.0, total_messages / 200),
                "impact": "processing_delay",
                "message_count": total_messages
            })
        
        # Check for context update overload
        update_rate = len(context.context_updates) / max(1, (datetime.now() - context.created_at).total_seconds())
        if update_rate > 20:  # More than 20 updates per second
            bottlenecks.append({
                "type": "update_overload",
                "severity": min(1.0, update_rate / 50),
                "impact": "context_processing_delay",
                "update_rate": update_rate
            })
        
        return bottlenecks
    
    async def _identify_optimization_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for optimization"""
        opportunities = []
        
        # Check for redundant module activation
        if len(context.active_modules) > 10:
            opportunities.append({
                "type": "module_reduction",
                "description": "Reduce active modules for simpler queries",
                "potential_improvement": "20-30% faster processing",
                "complexity": "low"
            })
        
        # Check for inefficient message patterns
        message_counts = {}
        for module_name, messages in context.module_messages.items():
            message_counts[module_name] = len(messages)
        
        if message_counts:
            avg_messages = sum(message_counts.values()) / len(message_counts)
            high_messagers = [m for m, count in message_counts.items() if count > avg_messages * 2]
            
            if high_messagers:
                opportunities.append({
                    "type": "message_optimization",
                    "description": f"Optimize message patterns for modules: {', '.join(high_messagers)}",
                    "potential_improvement": "15-20% reduction in overhead",
                    "complexity": "medium"
                })
        
        # Check for strategy optimization
        if hasattr(self, 'strategy_effectiveness'):
            underperforming = [s for s, scores in self.strategy_effectiveness.items()
                             if scores and sum(scores) / len(scores) < 0.4]
            
            if underperforming:
                opportunities.append({
                    "type": "strategy_tuning",
                    "description": f"Tune or replace underperforming strategies: {', '.join(underperforming)}",
                    "potential_improvement": "30-40% better adaptation",
                    "complexity": "high"
                })
        
        return opportunities
    
    def _calculate_recent_trend(self, values: List[float]) -> str:
        """Calculate trend from recent values"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Compare recent average to older average
        recent = values[-3:]
        older = values[:-3]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else recent_avg
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    async def _evaluate_strategy_suitability(self, strategy: Dict[str, Any], context: SharedContext) -> float:
        """Evaluate how suitable a strategy is for current context"""
        suitability = 0.5  # Base suitability
        
        params = strategy.get("parameters", {})
        
        # Evaluate based on context complexity
        complexity = self._calculate_context_complexity(context)
        
        # Higher complexity favors adaptive strategies
        if complexity > 0.7:
            suitability += params.get("adaptation_rate", 0.15) * 0.3
            suitability += params.get("exploration_rate", 0.2) * 0.2
        else:
            # Lower complexity favors conservative strategies
            suitability += (1.0 - params.get("risk_tolerance", 0.5)) * 0.3
            suitability += params.get("precision_focus", 0.5) * 0.2
        
        # Evaluate based on current performance
        if hasattr(self, 'performance_trend'):
            declining_count = sum(1 for trend in self.performance_trend.values()
                                if trend.get("direction") == "declining")
            
            if declining_count > 2:
                # Need more exploration
                suitability += params.get("exploration_rate", 0.2) * 0.2
            else:
                # Can focus on exploitation
                suitability += (1.0 - params.get("exploration_rate", 0.2)) * 0.2
        
        return min(1.0, suitability)
    
    async def _get_strategy_status_summary(self) -> Dict[str, Any]:
        """Get summary of current strategy status"""
        current_strategy_id = self.original_system.context.current_strategy_id
        current_strategy = self.original_system.context.strategies.get(current_strategy_id, {})
        
        summary = {
            "current_strategy": {
                "id": current_strategy_id,
                "name": current_strategy.get("name", "Unknown"),
                "parameters": current_strategy.get("parameters", {})
            },
            "time_in_strategy": "unknown",
            "effectiveness": 0.5
        }
        
        # Calculate time in current strategy
        if hasattr(self, 'strategy_change_log') and self.strategy_change_log:
            last_change = self.strategy_change_log[-1]
            if last_change["to_strategy"] == current_strategy_id:
                time_since = datetime.now() - datetime.fromisoformat(last_change["timestamp"])
                summary["time_in_strategy"] = str(time_since)
        
        # Get effectiveness
        if current_strategy_id in self.strategy_effectiveness:
            scores = self.strategy_effectiveness[current_strategy_id]
            if scores:
                summary["effectiveness"] = sum(scores) / len(scores)
        
        return summary
    
    async def _generate_adaptation_recommendations(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate specific adaptation recommendations"""
        recommendations = []
        
        # Check performance trends
        if hasattr(self, 'performance_trend'):
            declining = [m for m, t in self.performance_trend.items() 
                        if t.get("direction") == "declining"]
            
            if len(declining) > 2:
                recommendations.append(
                    f"Consider switching to a more exploratory strategy due to declining {', '.join(declining[:2])}"
                )
        
        # Check context complexity
        complexity = self._calculate_context_complexity(context)
        current_strategy = self.original_system.context.strategies.get(
            self.original_system.context.current_strategy_id, {}
        )
        
        if complexity > 0.7 and current_strategy.get("name") == "Conservative Strategy":
            recommendations.append(
                "High context complexity suggests switching from Conservative to Adaptive strategy"
            )
        
        # Check cross-module stress
        stressed_modules = []
        for module_name, module_messages in messages.items():
            stress_count = sum(1 for msg in module_messages 
                             if "error" in msg.get("type", "") or "failure" in msg.get("type", ""))
            if stress_count > 2:
                stressed_modules.append(module_name)
        
        if stressed_modules:
            recommendations.append(
                f"Modules showing stress: {', '.join(stressed_modules)}. Consider efficiency-focused strategy"
            )
        
        # Experience/Identity specific recommendations
        if "experience_adaptation_needed" in self.current_adaptation_context:
            recommendations.append(
                "Experience sharing showing low utility. Consider Experience-Focused strategy"
            )
        
        if "identity_adaptation_needed" in self.current_adaptation_context:
            recommendations.append(
                "Identity coherence is low. Consider Identity Stability strategy"
            )
        
        if not recommendations:
            recommendations.append("Current strategy appears well-suited to context")
        
        return recommendations
    
    async def _synthesize_performance_insights(self, context: SharedContext) -> List[str]:
        """Synthesize insights about performance"""
        insights = []
        
        # Overall performance level
        current_metrics = await self._gather_performance_metrics(context)
        avg_performance = sum(current_metrics.values()) / len(current_metrics) if current_metrics else 0.5
        
        if avg_performance > 0.8:
            insights.append("System performing excellently across all metrics")
        elif avg_performance < 0.4:
            insights.append("System performance is below acceptable levels")
        else:
            insights.append(f"System performance is moderate (avg: {avg_performance:.2f})")
        
        # Specific metric insights
        if current_metrics:
            best_metric = max(current_metrics.items(), key=lambda x: x[1])
            worst_metric = min(current_metrics.items(), key=lambda x: x[1])
            
            insights.append(f"Strongest area: {best_metric[0]} ({best_metric[1]:.2f})")
            insights.append(f"Needs improvement: {worst_metric[0]} ({worst_metric[1]:.2f})")
        
        # Trend insights
        if hasattr(self, 'performance_trend'):
            improving = sum(1 for t in self.performance_trend.values() 
                          if t.get("direction") == "improving")
            declining = sum(1 for t in self.performance_trend.values() 
                          if t.get("direction") == "declining")
            
            if improving > declining:
                insights.append(f"{improving} metrics improving, {declining} declining - positive trend")
            elif declining > improving:
                insights.append(f"{declining} metrics declining, {improving} improving - concerning trend")
        
        return insights
    
    async def _generate_optimization_suggestions(self, context: SharedContext) -> List[str]:
        """Generate system optimization suggestions"""
        suggestions = []
        
        # Module optimization
        if len(context.active_modules) > 12:
            suggestions.append(
                f"Reduce active modules from {len(context.active_modules)} to improve processing speed"
            )
        
        # Message optimization
        total_messages = sum(len(msgs) for msgs in context.module_messages.values())
        if total_messages > 100:
            suggestions.append(
                f"High message volume ({total_messages}) - consider message batching or filtering"
            )
        
        # Context update optimization
        update_rate = len(context.context_updates) / max(1, (datetime.now() - context.created_at).total_seconds())
        if update_rate > 10:
            suggestions.append(
                f"High context update rate ({update_rate:.1f}/sec) - consider update throttling"
            )
        
        # Strategy parameter tuning
        current_strategy = self.original_system.context.strategies.get(
            self.original_system.context.current_strategy_id, {}
        )
        params = current_strategy.get("parameters", {})
        
        if params.get("exploration_rate", 0) < 0.1 and hasattr(self, 'performance_trend'):
            stagnant = sum(1 for t in self.performance_trend.values() 
                         if t.get("direction") == "stable")
            if stagnant > 3:
                suggestions.append("Consider increasing exploration rate to break out of performance plateau")
        
        if not suggestions:
            suggestions.append("No immediate optimizations recommended")
        
        return suggestions
    
    async def _calculate_overall_adaptation_confidence(self, context: SharedContext) -> float:
        """Calculate confidence in adaptation decisions"""
        confidence_factors = []
        
        # Factor 1: Data sufficiency
        if hasattr(self.original_system, 'context') and self.original_system.context.performance_history:
            data_points = len(self.original_system.context.performance_history)
            data_confidence = min(1.0, data_points / 20)  # Full confidence at 20+ data points
            confidence_factors.append(data_confidence)
        else:
            confidence_factors.append(0.3)  # Low confidence without data
        
        # Factor 2: System stability
        if hasattr(self, 'performance_trend'):
            stable_metrics = sum(1 for t in self.performance_trend.values() 
                               if t.get("trend_strength", 0) < 0.2)
            stability_confidence = stable_metrics / max(1, len(self.performance_trend))
            confidence_factors.append(stability_confidence)
        
        # Factor 3: Strategy effectiveness clarity
        if hasattr(self, 'strategy_effectiveness'):
            # Check if we have clear winners/losers
            effectiveness_values = []
            for scores in self.strategy_effectiveness.values():
                if scores:
                    effectiveness_values.append(sum(scores) / len(scores))
            
            if len(effectiveness_values) >= 2:
                # Higher variance = clearer differences = higher confidence
                mean_eff = sum(effectiveness_values) / len(effectiveness_values)
                variance = sum((v - mean_eff) ** 2 for v in effectiveness_values) / len(effectiveness_values)
                clarity_confidence = min(1.0, variance * 5)  # Scale variance
                confidence_factors.append(clarity_confidence)
        
        # Factor 4: Context coherence
        coherence = await self._calculate_context_coherence(context)
        confidence_factors.append(coherence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    async def _evaluate_proactive_adaptation(self, synthesis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if proactive adaptation should be recommended"""
        confidence = synthesis.get("adaptation_confidence", 0.5)
        
        # Need high confidence for proactive adaptation
        if confidence < 0.7:
            return None
        
        # Check recommendations
        recommendations = synthesis.get("adaptation_recommendations", [])
        if not recommendations or recommendations[0] == "Current strategy appears well-suited to context":
            return None
        
        # Check for specific triggers
        strong_triggers = [
            "declining",
            "stress",
            "low utility",
            "low coherence",
            "overload"
        ]
        
        recommendation_text = " ".join(recommendations).lower()
        trigger_found = any(trigger in recommendation_text for trigger in strong_triggers)
        
        if not trigger_found:
            return None
        
        # Build proactive recommendation
        return {
            "recommended_action": "proactive_adaptation",
            "rationale": recommendations[0],  # Primary recommendation
            "confidence": confidence,
            "suggested_strategy": self._extract_suggested_strategy(recommendations),
            "urgency": "medium"  # Proactive is not urgent by definition
        }
    
    def _extract_suggested_strategy(self, recommendations: List[str]) -> Optional[str]:
        """Extract suggested strategy from recommendations"""
        strategy_keywords = {
            "exploratory": ["exploratory", "exploration"],
            "conservative": ["conservative", "precision"],
            "adaptive": ["adaptive", "adaptation"],
            "balanced": ["balanced", "moderate"],
            "experience_focused": ["experience-focused", "experience sharing"],
            "identity_stable": ["identity stability", "identity coherence"]
        }
        
        recommendation_text = " ".join(recommendations).lower()
        
        for strategy_id, keywords in strategy_keywords.items():
            if any(keyword in recommendation_text for keyword in keywords):
                return strategy_id
        
        return None
    
    async def _assess_response_quality(self, context: SharedContext) -> float:
        """Assess quality of system responses"""
        quality_factors = []
        
        # Check module response completeness
        if context.module_outputs:
            for stage, outputs in context.module_outputs.items():
                response_rate = len(outputs) / max(1, len(context.active_modules))
                quality_factors.append(response_rate)
        
        # Check for errors in outputs
        error_count = 0
        total_outputs = 0
        
        for outputs in context.module_outputs.values():
            for output in outputs.values():
                total_outputs += 1
                if isinstance(output, dict) and "error" in output:
                    error_count += 1
        
        if total_outputs > 0:
            error_rate = error_count / total_outputs
            quality_factors.append(1.0 - error_rate)
        
        # Check synthesis quality
        if context.synthesis_results:
            # Synthesis exists = good
            quality_factors.append(0.8)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
    
    async def _assess_module_coordination(self, context: SharedContext) -> float:
        """Assess how well modules are coordinating"""
        coordination_factors = []
        
        # Check cross-module messaging
        if context.module_messages:
            message_count = sum(len(msgs) for msgs in context.module_messages.values())
            active_modules = len(context.active_modules)
            
            if active_modules > 1:
                # Expect at least 2 messages per module for good coordination
                expected_messages = active_modules * 2
                message_ratio = min(1.0, message_count / expected_messages)
                coordination_factors.append(message_ratio)
        
        # Check context update participation
        if context.context_updates:
            unique_sources = set(update.source_module for update in context.context_updates)
            participation_rate = len(unique_sources) / max(1, len(context.active_modules))
            coordination_factors.append(participation_rate)
        
        # Check for module conflicts (from messages)
        conflict_count = 0
        for messages in context.module_messages.values():
            for msg in messages:
                if "conflict" in msg.get("type", "").lower():
                    conflict_count += 1
        
        # No conflicts = good coordination
        conflict_penalty = min(1.0, conflict_count / 5)  # 5+ conflicts = 0 score
        coordination_factors.append(1.0 - conflict_penalty)
        
        return sum(coordination_factors) / len(coordination_factors) if coordination_factors else 0.5
    
    async def _assess_resource_usage(self, context: SharedContext) -> float:
        """Assess resource usage efficiency"""
        # This is a simplified assessment - in production would check actual resources
        efficiency_factors = []
        
        # Check processing time
        processing_time = (datetime.now() - context.created_at).total_seconds()
        
        # Expect 1-3 seconds for normal processing
        if processing_time < 1:
            time_efficiency = 1.0  # Very fast
        elif processing_time < 3:
            time_efficiency = 0.8  # Good
        elif processing_time < 5:
            time_efficiency = 0.5  # Acceptable
        else:
            time_efficiency = max(0.0, 1.0 - (processing_time - 5) / 10)  # Degrade after 5s
        
        efficiency_factors.append(time_efficiency)
        
        # Check module efficiency (active vs actually used)
        if context.module_outputs:
            modules_that_responded = set()
            for outputs in context.module_outputs.values():
                modules_that_responded.update(outputs.keys())
            
            if context.active_modules:
                utilization = len(modules_that_responded) / len(context.active_modules)
                efficiency_factors.append(utilization)
        
        return sum(efficiency_factors) / len(efficiency_factors) if efficiency_factors else 0.5
    
    async def _assess_error_rate(self, context: SharedContext) -> float:
        """Assess system error rate (inverted - higher score = lower errors)"""
        error_count = 0
        total_operations = 0
        
        # Count errors in module outputs
        for outputs in context.module_outputs.values():
            for output in outputs.values():
                total_operations += 1
                if isinstance(output, dict) and "error" in output:
                    error_count += 1
        
        # Count errors in context updates
        for update in context.context_updates:
            if "error" in update.update_type.lower():
                error_count += 1
            total_operations += 1
        
        if total_operations == 0:
            return 1.0  # No operations = no errors
        
        error_rate = error_count / total_operations
        return 1.0 - error_rate  # Invert so higher score = better
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
