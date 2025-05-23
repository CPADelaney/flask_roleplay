# nyx/core/a2a/context_aware_meta_core.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareMetaCore(ContextAwareModule):
    """
    Advanced MetaCore with full context distribution capabilities for meta-cognitive coordination
    """
    
    def __init__(self, original_meta_core):
        super().__init__("meta_core")
        self.original_core = original_meta_core
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", "goal_progress",
            "needs_state_change", "relationship_milestone", "mode_change",
            "reward_signal", "performance_drop", "system_bottleneck",
            "mood_state_update", "attention_shift", "learning_event"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize meta-cognitive processing for this context"""
        logger.debug(f"MetaCore received context for user: {context.user_id}")
        
        # Analyze context for meta-cognitive implications
        meta_analysis = await self._analyze_context_for_meta_cognition(context)
        
        # Check if this requires immediate meta-cognitive attention
        requires_immediate_analysis = await self._check_immediate_analysis_needed(context, meta_analysis)
        
        # Get current system performance snapshot
        performance_snapshot = await self._get_performance_snapshot()
        
        # Send initial meta-cognitive assessment
        await self.send_context_update(
            update_type="meta_cognitive_assessment",
            data={
                "meta_analysis": meta_analysis,
                "requires_immediate_analysis": requires_immediate_analysis,
                "performance_snapshot": performance_snapshot,
                "cognitive_cycle": self.original_core.context.cognitive_cycle_count,
                "current_strategies": await self._get_active_strategies()
            },
            priority=ContextPriority.HIGH if requires_immediate_analysis else ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules with meta-cognitive analysis"""
        
        if update.update_type == "performance_drop":
            # Critical performance issue detected
            performance_data = update.data
            affected_system = performance_data.get("system")
            severity = performance_data.get("severity", 0.5)
            
            if severity > 0.7:
                # Trigger immediate cognitive cycle
                await self._trigger_emergency_cognitive_cycle(affected_system, performance_data)
                
                # Send high-priority strategy adjustment
                await self.send_context_update(
                    update_type="emergency_strategy_adjustment",
                    data={
                        "affected_system": affected_system,
                        "severity": severity,
                        "recommended_actions": await self._generate_emergency_actions(affected_system)
                    },
                    priority=ContextPriority.CRITICAL
                )
        
        elif update.update_type == "system_bottleneck":
            # System bottleneck detected
            bottleneck_data = update.data
            await self._process_bottleneck_alert(bottleneck_data)
        
        elif update.update_type == "emotional_state_update":
            # Significant emotional change might affect cognitive performance
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion and dominant_emotion[1] > 0.8:  # High intensity
                await self._adjust_cognitive_parameters_for_emotion(dominant_emotion[0], dominant_emotion[1])
        
        elif update.update_type == "mood_state_update":
            # Mood changes affect longer-term cognitive strategies
            mood_data = update.data
            await self._adjust_strategies_for_mood(mood_data)
        
        elif update.update_type == "goal_progress":
            # Goal progress affects resource allocation
            goal_data = update.data
            if goal_data.get("goal_completed") or goal_data.get("goal_failed"):
                await self._update_goal_performance_metrics(goal_data)
        
        elif update.update_type == "learning_event":
            # Significant learning event
            learning_data = update.data
            await self._integrate_learning_insights(learning_data)
        
        elif update.update_type == "attention_shift":
            # Attention focus changed
            attention_data = update.data
            await self._update_attention_allocation(attention_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with meta-cognitive awareness"""
        # Determine if meta-cognitive analysis is needed
        needs_analysis = await self._assess_meta_cognitive_need(context)
        
        # Run cognitive cycle if needed
        cycle_results = None
        if needs_analysis["run_cycle"]:
            # Prepare context for cognitive cycle
            cycle_context = {
                "user_input": context.user_input,
                "trigger": needs_analysis["trigger"],
                "priority_focus": needs_analysis.get("focus_area"),
                "current_performance": await self._get_current_performance_summary()
            }
            
            # Run cognitive cycle
            cycle_results = await self.original_core.cognitive_cycle(cycle_context)
            
            # Extract key insights
            insights = self._extract_cognitive_insights(cycle_results)
            
            # Send insights to other modules
            await self.send_context_update(
                update_type="meta_cognitive_insights",
                data={
                    "insights": insights,
                    "cycle_number": self.original_core.context.cognitive_cycle_count,
                    "improvements_suggested": cycle_results.get("improvements", []),
                    "strategy_changes": cycle_results.get("strategy_changes", {})
                }
            )
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        cross_module_insights = await self._analyze_cross_module_patterns(messages)
        
        return {
            "meta_cognition_active": True,
            "cycle_run": needs_analysis["run_cycle"],
            "cycle_results": cycle_results,
            "cross_module_insights": cross_module_insights,
            "cognitive_health": await self._assess_cognitive_health()
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Advanced meta-cognitive analysis of current state"""
        # Comprehensive system analysis
        system_analysis = await self._comprehensive_system_analysis()
        
        # Pattern detection across modules
        patterns = await self._detect_cognitive_patterns(context)
        
        # Performance trend analysis
        trends = await self._analyze_performance_trends()
        
        # Resource optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities()
        
        # Learning effectiveness
        learning_analysis = await self._analyze_learning_effectiveness()
        
        return {
            "system_analysis": system_analysis,
            "cognitive_patterns": patterns,
            "performance_trends": trends,
            "optimization_opportunities": optimization_opportunities,
            "learning_effectiveness": learning_analysis,
            "overall_cognitive_health": await self._calculate_overall_health(),
            "recommended_adjustments": await self._generate_adjustment_recommendations()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize meta-cognitive guidance for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Generate meta-cognitive influence on response
        meta_influence = {
            "cognitive_confidence": await self._calculate_cognitive_confidence(),
            "suggested_reasoning_depth": await self._determine_reasoning_depth(context),
            "processing_recommendations": await self._generate_processing_recommendations(context),
            "quality_thresholds": await self._determine_quality_thresholds(context),
            "error_prevention_strategies": await self._identify_error_prevention_needs(context)
        }
        
        # Check if any critical adjustments are needed
        critical_adjustments = await self._check_critical_adjustments_needed()
        if critical_adjustments:
            await self.send_context_update(
                update_type="critical_meta_adjustments",
                data=critical_adjustments,
                priority=ContextPriority.CRITICAL
            )
        
        # Generate predictions if appropriate
        if self._should_generate_predictions(context):
            prediction_context = {
                "current_context": context.dict(),
                "cognitive_state": await self._get_cognitive_state_summary(),
                "recent_patterns": patterns
            }
            
            prediction = await self.original_core.generate_prediction(prediction_context)
            meta_influence["prediction"] = prediction
        
        return {
            "meta_influence": meta_influence,
            "synthesis_complete": True,
            "cognitive_guidance_active": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_context_for_meta_cognition(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze context for meta-cognitive implications"""
        analysis = {
            "complexity_score": self._calculate_context_complexity(context),
            "uncertainty_level": self._assess_uncertainty(context),
            "cognitive_load": self._estimate_cognitive_load(context),
            "requires_deep_reasoning": self._check_deep_reasoning_need(context),
            "performance_risk": self._assess_performance_risk(context)
        }
        
        # Check for specific triggers
        user_input_lower = context.user_input.lower()
        
        if any(term in user_input_lower for term in ["optimize", "improve", "better", "performance"]):
            analysis["optimization_requested"] = True
            
        if any(term in user_input_lower for term in ["think", "reason", "analyze", "consider"]):
            analysis["explicit_reasoning_request"] = True
            
        if any(term in user_input_lower for term in ["learn", "adapt", "change", "evolve"]):
            analysis["adaptation_focus"] = True
        
        return analysis
    
    async def _check_immediate_analysis_needed(self, context: SharedContext, meta_analysis: Dict[str, Any]) -> bool:
        """Check if immediate meta-cognitive analysis is needed"""
        # High complexity or uncertainty
        if meta_analysis["complexity_score"] > 0.8 or meta_analysis["uncertainty_level"] > 0.7:
            return True
            
        # High cognitive load
        if meta_analysis["cognitive_load"] > 0.85:
            return True
            
        # Explicit request
        if meta_analysis.get("optimization_requested") or meta_analysis.get("explicit_reasoning_request"):
            return True
            
        # Performance risk
        if meta_analysis["performance_risk"] > 0.7:
            return True
            
        # Check system health
        health = await self._assess_cognitive_health()
        if health["overall_health"] < 0.5:
            return True
            
        return False
    
    async def _get_performance_snapshot(self) -> Dict[str, Any]:
        """Get current performance snapshot across all systems"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # Collect performance from registered systems
        for system_name in self.original_core.context.performance_history:
            system_data = self.original_core.context.performance_history.get(system_name, {})
            
            if isinstance(system_data, dict) and "history" in system_data and system_data["history"]:
                latest = system_data["history"][-1]
                metrics = latest.get("metrics", {})
                
                snapshot["systems"][system_name] = {
                    "success_rate": metrics.get("success_rate", 0.5),
                    "error_rate": metrics.get("error_rate", 0.0),
                    "response_time": metrics.get("response_time", 0.0),
                    "efficiency": metrics.get("efficiency", 0.5),
                    "status": self._determine_system_status(metrics)
                }
        
        # Calculate overall metrics
        if snapshot["systems"]:
            snapshot["overall"] = {
                "average_success_rate": sum(s["success_rate"] for s in snapshot["systems"].values()) / len(snapshot["systems"]),
                "average_error_rate": sum(s["error_rate"] for s in snapshot["systems"].values()) / len(snapshot["systems"]),
                "systems_healthy": sum(1 for s in snapshot["systems"].values() if s["status"] == "healthy"),
                "systems_struggling": sum(1 for s in snapshot["systems"].values() if s["status"] in ["warning", "critical"])
            }
        
        return snapshot
    
    async def _get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get currently active cognitive strategies"""
        strategies = []
        
        # Get from context if available
        if hasattr(self.original_core.context, 'improvement_plans'):
            for plan in self.original_core.context.improvement_plans:
                if plan.get("status") == "active":
                    strategies.append({
                        "name": plan.get("name", "Unknown Strategy"),
                        "target_systems": plan.get("targets", []),
                        "expected_impact": plan.get("expected_outcomes", {}),
                        "cycles_active": self.original_core.context.cognitive_cycle_count - plan.get("start_cycle", 0)
                    })
        
        return strategies
    
    async def _trigger_emergency_cognitive_cycle(self, affected_system: str, performance_data: Dict[str, Any]):
        """Trigger emergency cognitive cycle for critical issues"""
        logger.warning(f"Triggering emergency cognitive cycle for {affected_system}")
        
        # Prepare emergency context
        emergency_context = {
            "emergency": True,
            "affected_system": affected_system,
            "performance_data": performance_data,
            "priority": "critical",
            "timestamp": datetime.now().isoformat()
        }
        
        # Run focused cognitive cycle
        try:
            results = await self.original_core.cognitive_cycle(emergency_context)
            
            # Log results
            logger.info(f"Emergency cognitive cycle completed: {results.get('improvements', [])}")
            
        except Exception as e:
            logger.error(f"Error in emergency cognitive cycle: {e}")
    
    async def _generate_emergency_actions(self, affected_system: str) -> List[Dict[str, Any]]:
        """Generate emergency actions for system recovery"""
        actions = []
        
        # System-specific emergency actions
        system_actions = {
            "memory": [
                {"action": "clear_cache", "description": "Clear memory caches to free resources"},
                {"action": "reduce_retrieval_scope", "description": "Temporarily reduce memory retrieval scope"},
                {"action": "prioritize_recent", "description": "Focus on recent memories only"}
            ],
            "emotion": [
                {"action": "stabilize_valence", "description": "Stabilize emotional valence to neutral"},
                {"action": "reduce_intensity", "description": "Reduce emotional intensity"},
                {"action": "increase_regulation", "description": "Increase emotional regulation"}
            ],
            "reasoning": [
                {"action": "simplify_reasoning", "description": "Use simpler reasoning strategies"},
                {"action": "reduce_depth", "description": "Reduce reasoning depth temporarily"},
                {"action": "increase_timeouts", "description": "Increase reasoning timeouts"}
            ],
            "goal": [
                {"action": "pause_low_priority", "description": "Pause low priority goals"},
                {"action": "consolidate_similar", "description": "Consolidate similar goals"},
                {"action": "extend_deadlines", "description": "Extend goal deadlines"}
            ]
        }
        
        # Get system-specific actions
        if affected_system in system_actions:
            actions.extend(system_actions[affected_system])
        
        # Add general recovery actions
        actions.extend([
            {"action": "increase_monitoring", "description": "Increase monitoring frequency"},
            {"action": "allocate_resources", "description": f"Increase resource allocation to {affected_system}"},
            {"action": "reduce_load", "description": "Reduce overall cognitive load"}
        ])
        
        return actions
    
    async def _process_bottleneck_alert(self, bottleneck_data: Dict[str, Any]):
        """Process system bottleneck alert"""
        bottleneck_type = bottleneck_data.get("type")
        severity = bottleneck_data.get("severity", 0.5)
        affected_processes = bottleneck_data.get("processes", [])
        
        # Update context
        if not hasattr(self.original_core.context, 'active_bottlenecks'):
            self.original_core.context.active_bottlenecks = []
            
        self.original_core.context.active_bottlenecks.append({
            "type": bottleneck_type,
            "severity": severity,
            "processes": affected_processes,
            "detected_at": datetime.now().isoformat()
        })
        
        # Clean old bottlenecks (older than 1 hour)
        current_time = datetime.now()
        self.original_core.context.active_bottlenecks = [
            b for b in self.original_core.context.active_bottlenecks
            if (current_time - datetime.fromisoformat(b["detected_at"])).total_seconds() < 3600
        ]
        
        # If severe, trigger analysis
        if severity > 0.7:
            await self.send_context_update(
                update_type="bottleneck_analysis_needed",
                data={
                    "bottleneck_type": bottleneck_type,
                    "severity": severity,
                    "recommended_action": "immediate_reallocation"
                },
                priority=ContextPriority.HIGH
            )
    
    async def _adjust_cognitive_parameters_for_emotion(self, emotion: str, intensity: float):
        """Adjust cognitive parameters based on emotional state"""
        # Emotion-specific adjustments
        adjustments = {
            "Anxiety": {
                "exploration_rate": -0.2 * intensity,
                "risk_tolerance": -0.3 * intensity,
                "precision_focus": 0.2 * intensity
            },
            "Excitement": {
                "exploration_rate": 0.3 * intensity,
                "risk_tolerance": 0.2 * intensity,
                "adaptation_rate": 0.15 * intensity
            },
            "Frustration": {
                "patience_threshold": -0.3 * intensity,
                "error_tolerance": -0.2 * intensity,
                "strategy_switching_rate": 0.3 * intensity
            },
            "Confidence": {
                "risk_tolerance": 0.2 * intensity,
                "exploration_rate": 0.15 * intensity,
                "goal_ambition": 0.2 * intensity
            }
        }
        
        if emotion in adjustments:
            params = adjustments[emotion]
            
            # Apply adjustments to meta parameters
            for param, adjustment in params.items():
                if param in self.original_core.context.meta_parameters:
                    current = self.original_core.context.meta_parameters[param]
                    new_value = max(0.0, min(1.0, current + adjustment))
                    self.original_core.context.meta_parameters[param] = new_value
                    
            logger.debug(f"Adjusted cognitive parameters for {emotion} (intensity: {intensity})")
    
    async def _adjust_strategies_for_mood(self, mood_data: Dict[str, Any]):
        """Adjust long-term strategies based on mood state"""
        mood = mood_data.get("dominant_mood", "Neutral")
        valence = mood_data.get("valence", 0.0)
        arousal = mood_data.get("arousal", 0.5)
        
        # Strategy adjustments based on mood dimensions
        strategy_adjustments = {}
        
        # Low valence (negative mood) - more conservative
        if valence < -0.3:
            strategy_adjustments["risk_tolerance"] = -0.2
            strategy_adjustments["exploration_rate"] = -0.15
            strategy_adjustments["resource_conservation"] = 0.3
            
        # High valence (positive mood) - more exploratory
        elif valence > 0.3:
            strategy_adjustments["exploration_rate"] = 0.2
            strategy_adjustments["innovation_level"] = 0.25
            strategy_adjustments["goal_ambition"] = 0.15
            
        # Low arousal - focus on efficiency
        if arousal < 0.3:
            strategy_adjustments["processing_depth"] = -0.2
            strategy_adjustments["quick_decisions"] = 0.3
            
        # High arousal - deeper processing
        elif arousal > 0.7:
            strategy_adjustments["processing_depth"] = 0.3
            strategy_adjustments["parallel_processing"] = 0.2
        
        # Send strategy adjustment recommendation
        if strategy_adjustments:
            await self.send_context_update(
                update_type="mood_based_strategy_adjustment",
                data={
                    "mood": mood,
                    "adjustments": strategy_adjustments,
                    "reasoning": f"Mood state ({mood}) suggests strategy adjustments"
                }
            )
    
    async def _update_goal_performance_metrics(self, goal_data: Dict[str, Any]):
        """Update performance metrics based on goal outcomes"""
        if goal_data.get("goal_completed"):
            # Successful completion
            system = goal_data.get("system", "goal")
            
            # Update success metrics
            if system in self.original_core.context.performance_history:
                history = self.original_core.context.performance_history[system]
                if isinstance(history, dict) and "history" in history and history["history"]:
                    latest = history["history"][-1]
                    if "metrics" in latest:
                        latest["metrics"]["success_rate"] = min(1.0, latest["metrics"].get("success_rate", 0.5) + 0.05)
                        
        elif goal_data.get("goal_failed"):
            # Failed goal
            system = goal_data.get("system", "goal")
            failure_reason = goal_data.get("failure_reason", "unknown")
            
            # Analyze failure for insights
            failure_insights = {
                "system": system,
                "reason": failure_reason,
                "timestamp": datetime.now().isoformat(),
                "context": goal_data
            }
            
            # Store for learning
            if not hasattr(self.original_core.context, 'failure_history'):
                self.original_core.context.failure_history = []
            self.original_core.context.failure_history.append(failure_insights)
            
            # Keep only recent failures (last 50)
            if len(self.original_core.context.failure_history) > 50:
                self.original_core.context.failure_history = self.original_core.context.failure_history[-50:]
    
    async def _integrate_learning_insights(self, learning_data: Dict[str, Any]):
        """Integrate insights from learning events"""
        insight_type = learning_data.get("type", "general")
        insight_content = learning_data.get("content", {})
        confidence = learning_data.get("confidence", 0.5)
        
        # Store learning insight
        learning_insight = {
            "type": insight_type,
            "content": insight_content,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "applied": False
        }
        
        self.original_core.context.insights.append(learning_insight)
        
        # If high confidence, consider immediate application
        if confidence > 0.8:
            await self.send_context_update(
                update_type="high_confidence_learning",
                data={
                    "insight": learning_insight,
                    "recommended_application": await self._generate_insight_application(learning_insight)
                },
                priority=ContextPriority.HIGH
            )
    
    async def _update_attention_allocation(self, attention_data: Dict[str, Any]):
        """Update resource allocation based on attention shifts"""
        new_focus = attention_data.get("new_focus")
        priority = attention_data.get("priority", 0.5)
        
        if new_focus and priority > 0.7:
            # High priority attention shift
            current_allocation = self.original_core.context.resource_allocation.copy()
            
            # Increase allocation to focus area
            if new_focus in current_allocation:
                # Boost focus area
                boost = 0.2 * priority
                current_allocation[new_focus] = min(0.5, current_allocation[new_focus] + boost)
                
                # Reduce others proportionally
                total_other = sum(v for k, v in current_allocation.items() if k != new_focus)
                if total_other > 0:
                    reduction_factor = (1.0 - current_allocation[new_focus]) / total_other
                    for k in current_allocation:
                        if k != new_focus:
                            current_allocation[k] *= reduction_factor
                
                # Update allocation
                self.original_core.context.resource_allocation = current_allocation
                
                logger.info(f"Updated resource allocation for attention shift to {new_focus}")
    
    async def _assess_meta_cognitive_need(self, context: SharedContext) -> Dict[str, Any]:
        """Assess if meta-cognitive analysis is needed"""
        need_assessment = {
            "run_cycle": False,
            "trigger": None,
            "focus_area": None
        }
        
        # Check cycle frequency
        cycles_since_last = self.original_core.context.cognitive_cycle_count % self.original_core.context.meta_parameters.get("evaluation_interval", 5)
        
        if cycles_since_last == 0:
            need_assessment["run_cycle"] = True
            need_assessment["trigger"] = "scheduled"
            
        # Check for performance issues
        health = await self._assess_cognitive_health()
        if health["overall_health"] < 0.6:
            need_assessment["run_cycle"] = True
            need_assessment["trigger"] = "performance_issue"
            need_assessment["focus_area"] = health.get("worst_system")
            
        # Check for explicit requests
        if any(term in context.user_input.lower() for term in ["optimize", "improve performance", "think about your thinking"]):
            need_assessment["run_cycle"] = True
            need_assessment["trigger"] = "user_request"
            
        # Check for high complexity
        complexity = self._calculate_context_complexity(context)
        if complexity > 0.85:
            need_assessment["run_cycle"] = True
            need_assessment["trigger"] = "high_complexity"
            
        return need_assessment
    
    async def _get_current_performance_summary(self) -> Dict[str, Any]:
        """Get summary of current performance metrics"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {},
            "system_status": {}
        }
        
        # Aggregate metrics
        total_success = 0.0
        total_error = 0.0
        count = 0
        
        for system_name, data in self.original_core.context.performance_history.items():
            if isinstance(data, dict) and "history" in data and data["history"]:
                latest = data["history"][-1].get("metrics", {})
                
                success_rate = latest.get("success_rate", 0.5)
                error_rate = latest.get("error_rate", 0.0)
                
                total_success += success_rate
                total_error += error_rate
                count += 1
                
                summary["system_status"][system_name] = {
                    "success_rate": success_rate,
                    "error_rate": error_rate,
                    "status": self._determine_system_status(latest)
                }
        
        if count > 0:
            summary["overall_metrics"]["average_success_rate"] = total_success / count
            summary["overall_metrics"]["average_error_rate"] = total_error / count
            summary["overall_metrics"]["health_score"] = (total_success / count) * (1.0 - total_error / count)
        
        return summary
    
    def _extract_cognitive_insights(self, cycle_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key insights from cognitive cycle results"""
        insights = []
        
        # Extract bottlenecks
        if "bottlenecks" in cycle_results:
            for bottleneck in cycle_results["bottlenecks"]:
                insights.append({
                    "type": "bottleneck",
                    "system": bottleneck.get("process_type"),
                    "severity": bottleneck.get("severity", 0.5),
                    "description": bottleneck.get("description", "")
                })
        
        # Extract improvements
        if "improvements" in cycle_results:
            for improvement in cycle_results["improvements"]:
                insights.append({
                    "type": "improvement",
                    "description": improvement.get("description", ""),
                    "expected_impact": improvement.get("expected_impact", {})
                })
        
        # Extract strategy changes
        if "strategies" in cycle_results:
            for strategy in cycle_results["strategies"]:
                insights.append({
                    "type": "strategy",
                    "name": strategy.get("name", ""),
                    "parameters": strategy.get("parameters", {})
                })
        
        return insights
    
    async def _analyze_cross_module_patterns(self, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze patterns in cross-module communication"""
        patterns = {
            "communication_frequency": {},
            "common_themes": [],
            "coordination_issues": [],
            "synergies": []
        }
        
        # Analyze communication frequency
        for module, module_messages in messages.items():
            patterns["communication_frequency"][module] = len(module_messages)
        
        # Detect common themes
        all_messages = []
        for module_messages in messages.values():
            all_messages.extend(module_messages)
        
        # Simple theme detection based on message types
        theme_counts = {}
        for msg in all_messages:
            msg_type = msg.get("type", "unknown")
            theme_counts[msg_type] = theme_counts.get(msg_type, 0) + 1
        
        # Top themes
        patterns["common_themes"] = sorted(
            theme_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Detect coordination issues (modules not communicating when they should)
        expected_pairs = [
            ("emotional_core", "mood_manager"),
            ("goal_manager", "needs_system"),
            ("memory_core", "reasoning_core")
        ]
        
        for module1, module2 in expected_pairs:
            if module1 in messages and module2 in messages:
                # Check if they reference each other
                m1_refs_m2 = any(module2 in str(msg) for msg in messages[module1])
                m2_refs_m1 = any(module1 in str(msg) for msg in messages[module2])
                
                if not (m1_refs_m2 or m2_refs_m1):
                    patterns["coordination_issues"].append({
                        "modules": [module1, module2],
                        "issue": "No cross-referencing detected"
                    })
        
        return patterns
    
    async def _assess_cognitive_health(self) -> Dict[str, Any]:
        """Assess overall cognitive health"""
        health_metrics = {
            "overall_health": 0.0,
            "system_health": {},
            "worst_system": None,
            "best_system": None
        }
        
        worst_score = 1.0
        best_score = 0.0
        total_score = 0.0
        count = 0
        
        for system_name, data in self.original_core.context.performance_history.items():
            if isinstance(data, dict) and "history" in data and data["history"]:
                latest = data["history"][-1].get("metrics", {})
                
                # Calculate health score
                success_rate = latest.get("success_rate", 0.5)
                error_rate = latest.get("error_rate", 0.0)
                efficiency = latest.get("efficiency", 0.5)
                
                health_score = (success_rate * 0.5 + efficiency * 0.3 + (1.0 - error_rate) * 0.2)
                
                health_metrics["system_health"][system_name] = health_score
                total_score += health_score
                count += 1
                
                if health_score < worst_score:
                    worst_score = health_score
                    health_metrics["worst_system"] = system_name
                    
                if health_score > best_score:
                    best_score = health_score
                    health_metrics["best_system"] = system_name
        
        if count > 0:
            health_metrics["overall_health"] = total_score / count
        
        return health_metrics
    
    async def _comprehensive_system_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of all systems"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "systems": {},
            "correlations": [],
            "recommendations": []
        }
        
        # Analyze each system
        for system_name in self.original_core.context.performance_history:
            system_analysis = await self._analyze_single_system(system_name)
            analysis["systems"][system_name] = system_analysis
        
        # Find correlations between systems
        analysis["correlations"] = await self._find_system_correlations()
        
        # Generate recommendations
        analysis["recommendations"] = await self._generate_system_recommendations(analysis["systems"])
        
        return analysis
    
    async def _detect_cognitive_patterns(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Detect patterns in cognitive processing"""
        patterns = []
        
        # Pattern: Emotional influence on reasoning
        if context.emotional_state and context.session_context.get("reasoning_depth"):
            emotion_strength = max(context.emotional_state.values()) if context.emotional_state else 0
            reasoning_depth = context.session_context["reasoning_depth"]
            
            if emotion_strength > 0.7 and reasoning_depth < 0.5:
                patterns.append({
                    "type": "emotion_reasoning_interference",
                    "description": "High emotion reducing reasoning depth",
                    "strength": emotion_strength
                })
        
        # Pattern: Goal-memory alignment
        if context.goal_context and context.memory_context:
            active_goals = context.goal_context.get("active_goals", [])
            retrieved_memories = context.memory_context.get("memory_count", 0)
            
            if len(active_goals) > 5 and retrieved_memories < 3:
                patterns.append({
                    "type": "goal_memory_mismatch",
                    "description": "Many goals but few supporting memories",
                    "goal_count": len(active_goals),
                    "memory_count": retrieved_memories
                })
        
        # Pattern: Mode-emotion coherence
        if context.mode_context and context.emotional_state:
            mode = context.mode_context.get("primary_mode", "")
            dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else None
            
            # Check for mismatches
            mismatches = {
                ("dominant", "Submissive"): "mode_emotion_conflict",
                ("compassionate", "Frustration"): "mode_emotion_conflict",
                ("playful", "Anxiety"): "mode_emotion_conflict"
            }
            
            if (mode, dominant_emotion) in mismatches:
                patterns.append({
                    "type": mismatches[(mode, dominant_emotion)],
                    "description": f"Mode {mode} conflicts with emotion {dominant_emotion}",
                    "mode": mode,
                    "emotion": dominant_emotion
                })
        
        return patterns
    
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across systems"""
        trends = {
            "improving_systems": [],
            "declining_systems": [],
            "stable_systems": [],
            "volatile_systems": []
        }
        
        for system_name, data in self.original_core.context.performance_history.items():
            if isinstance(data, dict) and "history" in data and len(data["history"]) >= 5:
                # Get last 5 entries
                recent_history = data["history"][-5:]
                
                # Extract success rates
                success_rates = [h.get("metrics", {}).get("success_rate", 0.5) for h in recent_history]
                
                # Calculate trend
                if len(success_rates) >= 3:
                    # Simple linear trend
                    first_half_avg = sum(success_rates[:len(success_rates)//2]) / (len(success_rates)//2)
                    second_half_avg = sum(success_rates[len(success_rates)//2:]) / (len(success_rates) - len(success_rates)//2)
                    
                    trend = second_half_avg - first_half_avg
                    
                    # Calculate volatility
                    avg_rate = sum(success_rates) / len(success_rates)
                    volatility = sum(abs(rate - avg_rate) for rate in success_rates) / len(success_rates)
                    
                    # Categorize
                    if volatility > 0.2:
                        trends["volatile_systems"].append({
                            "system": system_name,
                            "volatility": volatility,
                            "average": avg_rate
                        })
                    elif trend > 0.1:
                        trends["improving_systems"].append({
                            "system": system_name,
                            "improvement": trend,
                            "current": success_rates[-1]
                        })
                    elif trend < -0.1:
                        trends["declining_systems"].append({
                            "system": system_name,
                            "decline": abs(trend),
                            "current": success_rates[-1]
                        })
                    else:
                        trends["stable_systems"].append({
                            "system": system_name,
                            "average": avg_rate
                        })
        
        return trends
    
    async def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for system optimization"""
        opportunities = []
        
        # Check resource allocation efficiency
        for system_name, allocation in self.original_core.context.resource_allocation.items():
            if system_name in self.original_core.context.performance_history:
                data = self.original_core.context.performance_history[system_name]
                
                if isinstance(data, dict) and "history" in data and data["history"]:
                    latest = data["history"][-1].get("metrics", {})
                    success_rate = latest.get("success_rate", 0.5)
                    
                    # High allocation but low success
                    if allocation > 0.25 and success_rate < 0.5:
                        opportunities.append({
                            "type": "resource_inefficiency",
                            "system": system_name,
                            "current_allocation": allocation,
                            "success_rate": success_rate,
                            "recommendation": "Reduce allocation or improve algorithms"
                        })
                    
                    # Low allocation but high success (could do more)
                    elif allocation < 0.15 and success_rate > 0.8:
                        opportunities.append({
                            "type": "underutilized_potential",
                            "system": system_name,
                            "current_allocation": allocation,
                            "success_rate": success_rate,
                            "recommendation": "Increase allocation to leverage success"
                        })
        
        # Check for parameter optimization
        if hasattr(self.original_core.context, 'meta_parameters'):
            # Learning rate optimization
            if self.original_core.context.cognitive_cycle_count > 100:
                current_lr = self.original_core.context.meta_parameters.get("learning_rate", 0.1)
                if current_lr > 0.05:
                    opportunities.append({
                        "type": "parameter_optimization",
                        "parameter": "learning_rate",
                        "current_value": current_lr,
                        "recommended_value": current_lr * 0.8,
                        "reasoning": "Reduce learning rate after many cycles for stability"
                    })
        
        return opportunities
    
    async def _analyze_learning_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effectively the system is learning"""
        effectiveness = {
            "learning_rate": 0.0,
            "adaptation_success": 0.0,
            "insight_application_rate": 0.0,
            "failure_learning_rate": 0.0
        }
        
        # Calculate insight application rate
        if hasattr(self.original_core.context, 'insights'):
            total_insights = len(self.original_core.context.insights)
            applied_insights = sum(1 for i in self.original_core.context.insights if i.get("applied", False))
            
            if total_insights > 0:
                effectiveness["insight_application_rate"] = applied_insights / total_insights
        
        # Calculate failure learning rate
        if hasattr(self.original_core.context, 'failure_history'):
            # Group failures by type
            failure_types = {}
            for failure in self.original_core.context.failure_history:
                reason = failure.get("reason", "unknown")
                failure_types[reason] = failure_types.get(reason, 0) + 1
            
            # Check if repeat failures are decreasing
            if len(failure_types) > 0:
                avg_failures = sum(failure_types.values()) / len(failure_types)
                recent_failures = len([f for f in self.original_core.context.failure_history[-10:]])
                
                if avg_failures > 0:
                    effectiveness["failure_learning_rate"] = 1.0 - (recent_failures / min(10, len(self.original_core.context.failure_history)))
        
        # Overall learning effectiveness
        effectiveness["learning_rate"] = sum([
            effectiveness["insight_application_rate"] * 0.5,
            effectiveness["failure_learning_rate"] * 0.5
        ])
        
        return effectiveness
    
    async def _calculate_overall_health(self) -> float:
        """Calculate overall cognitive health score"""
        health_components = []
        
        # System health
        system_health = await self._assess_cognitive_health()
        health_components.append(system_health["overall_health"])
        
        # Learning effectiveness
        learning = await self._analyze_learning_effectiveness()
        health_components.append(learning["learning_rate"])
        
        # Resource efficiency
        resource_efficiency = await self._calculate_resource_efficiency()
        health_components.append(resource_efficiency)
        
        # Error rate (inverted)
        avg_error_rate = 0.0
        count = 0
        for system_name, data in self.original_core.context.performance_history.items():
            if isinstance(data, dict) and "history" in data and data["history"]:
                latest = data["history"][-1].get("metrics", {})
                avg_error_rate += latest.get("error_rate", 0.0)
                count += 1
        
        if count > 0:
            health_components.append(1.0 - (avg_error_rate / count))
        
        # Average all components
        return sum(health_components) / len(health_components) if health_components else 0.5
    
    async def _generate_adjustment_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific adjustment recommendations"""
        recommendations = []
        
        # Get current state
        health = await self._assess_cognitive_health()
        trends = await self._analyze_performance_trends()
        opportunities = await self._identify_optimization_opportunities()
        
        # Priority 1: Address critical systems
        if health["worst_system"] and health["system_health"].get(health["worst_system"], 1.0) < 0.4:
            recommendations.append({
                "priority": 1,
                "type": "critical_system_recovery",
                "target": health["worst_system"],
                "actions": [
                    f"Increase resource allocation to {health['worst_system']}",
                    "Simplify processing strategies",
                    "Increase error tolerance temporarily"
                ]
            })
        
        # Priority 2: Address declining systems
        for system_info in trends["declining_systems"]:
            recommendations.append({
                "priority": 2,
                "type": "arrest_decline",
                "target": system_info["system"],
                "actions": [
                    "Analyze recent changes that caused decline",
                    "Revert problematic parameter changes",
                    "Increase monitoring frequency"
                ]
            })
        
        # Priority 3: Leverage improving systems
        for system_info in trends["improving_systems"]:
            recommendations.append({
                "priority": 3,
                "type": "amplify_success",
                "target": system_info["system"],
                "actions": [
                    "Document successful strategies",
                    "Apply similar improvements to related systems",
                    "Slightly increase resource allocation"
                ]
            })
        
        # Priority 4: Optimization opportunities
        for opportunity in opportunities[:3]:  # Top 3 opportunities
            recommendations.append({
                "priority": 4,
                "type": "optimization",
                "target": opportunity.get("system", opportunity.get("parameter", "general")),
                "actions": [opportunity["recommendation"]]
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"])
        
        return recommendations
    
    async def _calculate_cognitive_confidence(self) -> float:
        """Calculate confidence in cognitive processing"""
        confidence_factors = []
        
        # Factor 1: System health
        health = await self._assess_cognitive_health()
        confidence_factors.append(health["overall_health"])
        
        # Factor 2: Prediction accuracy (if available)
        if hasattr(self.original_core, 'prediction_engine'):
            # Get recent prediction accuracy
            recent_evaluations = [
                i for i in self.original_core.context.insights 
                if i.get("type") == "prediction" and i.get("evaluated", False)
            ][-5:]  # Last 5 evaluations
            
            if recent_evaluations:
                accuracies = []
                for eval_insight in recent_evaluations:
                    if "evaluation" in eval_insight:
                        accuracy = eval_insight["evaluation"].get("accuracy", 0.5)
                        accuracies.append(accuracy)
                
                if accuracies:
                    confidence_factors.append(sum(accuracies) / len(accuracies))
        
        # Factor 3: Low error rate
        avg_error_rate = 0.0
        count = 0
        for system_name, data in self.original_core.context.performance_history.items():
            if isinstance(data, dict) and "history" in data and data["history"]:
                latest = data["history"][-1].get("metrics", {})
                avg_error_rate += latest.get("error_rate", 0.0)
                count += 1
        
        if count > 0:
            confidence_factors.append(1.0 - (avg_error_rate / count))
        
        # Factor 4: Stable performance (low volatility)
        trends = await self._analyze_performance_trends()
        volatility_penalty = len(trends["volatile_systems"]) * 0.1
        confidence_factors.append(max(0.0, 1.0 - volatility_penalty))
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    async def _determine_reasoning_depth(self, context: SharedContext) -> float:
        """Determine appropriate reasoning depth for context"""
        base_depth = 0.5
        
        # Increase for complex contexts
        complexity = self._calculate_context_complexity(context)
        base_depth += complexity * 0.3
        
        # Increase for uncertainty
        uncertainty = self._assess_uncertainty(context)
        base_depth += uncertainty * 0.2
        
        # Decrease if under high load
        load = self._estimate_cognitive_load(context)
        if load > 0.8:
            base_depth -= 0.2
        
        # Adjust based on available resources
        if hasattr(self.original_core.context, 'resource_allocation'):
            reasoning_allocation = self.original_core.context.resource_allocation.get("reasoning", 0.2)
            base_depth *= (0.5 + reasoning_allocation)  # Scale by allocation
        
        return max(0.1, min(1.0, base_depth))
    
    async def _generate_processing_recommendations(self, context: SharedContext) -> List[str]:
        """Generate specific processing recommendations"""
        recommendations = []
        
        # Based on context analysis
        complexity = self._calculate_context_complexity(context)
        uncertainty = self._assess_uncertainty(context)
        
        if complexity > 0.8:
            recommendations.append("Break down into smaller sub-problems")
            recommendations.append("Use systematic step-by-step analysis")
            
        if uncertainty > 0.7:
            recommendations.append("Gather additional information before deciding")
            recommendations.append("Consider multiple hypotheses")
            
        # Based on current state
        health = await self._assess_cognitive_health()
        if health["overall_health"] < 0.6:
            recommendations.append("Prioritize accuracy over speed")
            recommendations.append("Use validated reasoning patterns")
        
        # Based on emotional state
        if context.emotional_state:
            dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else None
            if dominant_emotion == "Anxiety":
                recommendations.append("Take time to calm before responding")
                recommendations.append("Focus on facts over speculation")
        
        return recommendations
    
    async def _determine_quality_thresholds(self, context: SharedContext) -> Dict[str, float]:
        """Determine quality thresholds for response generation"""
        thresholds = {
            "minimum_confidence": 0.6,
            "coherence_threshold": 0.7,
            "completeness_threshold": 0.8,
            "accuracy_requirement": 0.85
        }
        
        # Adjust based on context importance
        if "critical" in context.user_input.lower() or "important" in context.user_input.lower():
            # Increase all thresholds
            for key in thresholds:
                thresholds[key] = min(0.95, thresholds[key] + 0.1)
        
        # Adjust based on mode
        if context.mode_context and context.mode_context.get("primary_mode") == "professional":
            thresholds["accuracy_requirement"] = 0.9
            thresholds["coherence_threshold"] = 0.85
        
        # Adjust based on relationship context
        if context.relationship_context and context.relationship_context.get("trust", 0) > 0.8:
            # Can be slightly more relaxed with trusted users
            for key in thresholds:
                thresholds[key] = max(0.5, thresholds[key] - 0.05)
        
        return thresholds
    
    async def _identify_error_prevention_needs(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify potential errors and prevention strategies"""
        prevention_needs = []
        
        # Check for common error patterns
        if hasattr(self.original_core.context, 'failure_history'):
            recent_failures = self.original_core.context.failure_history[-10:]
            
            # Group by failure type
            failure_types = {}
            for failure in recent_failures:
                reason = failure.get("reason", "unknown")
                failure_types[reason] = failure_types.get(reason, 0) + 1
            
            # Identify repeat failures
            for reason, count in failure_types.items():
                if count >= 2:
                    prevention_needs.append({
                        "error_type": reason,
                        "frequency": count,
                        "prevention_strategy": self._get_prevention_strategy(reason)
                    })
        
        # Context-specific error risks
        complexity = self._calculate_context_complexity(context)
        if complexity > 0.85:
            prevention_needs.append({
                "error_type": "complexity_overload",
                "risk_level": complexity,
                "prevention_strategy": "Decompose into simpler components"
            })
        
        # Emotional interference risks
        if context.emotional_state:
            max_emotion_strength = max(context.emotional_state.values()) if context.emotional_state else 0
            if max_emotion_strength > 0.8:
                prevention_needs.append({
                    "error_type": "emotional_bias",
                    "risk_level": max_emotion_strength,
                    "prevention_strategy": "Apply emotional regulation before processing"
                })
        
        return prevention_needs
    
    async def _check_critical_adjustments_needed(self) -> Optional[Dict[str, Any]]:
        """Check if any critical adjustments are needed immediately"""
        critical_adjustments = None
        
        # Check for system failures
        health = await self._assess_cognitive_health()
        if health["overall_health"] < 0.3:
            critical_adjustments = {
                "type": "emergency_recovery",
                "severity": "critical",
                "affected_systems": [s for s, h in health["system_health"].items() if h < 0.4],
                "immediate_actions": [
                    "Reduce processing complexity",
                    "Focus on core functions only",
                    "Increase error tolerance",
                    "Activate fallback strategies"
                ]
            }
        
        # Check for resource starvation
        elif hasattr(self.original_core.context, 'resource_allocation'):
            starved_systems = [
                s for s, a in self.original_core.context.resource_allocation.items() 
                if a < 0.05 and s in health["system_health"] and health["system_health"][s] < 0.5
            ]
            
            if starved_systems:
                critical_adjustments = {
                    "type": "resource_starvation",
                    "severity": "high",
                    "affected_systems": starved_systems,
                    "immediate_actions": [
                        f"Reallocate resources to {', '.join(starved_systems)}",
                        "Reduce load on over-allocated systems"
                    ]
                }
        
        return critical_adjustments
    
    def _should_generate_predictions(self, context: SharedContext) -> bool:
        """Determine if predictions should be generated"""
        # Generate predictions for forward-looking queries
        predictive_keywords = ["will", "future", "expect", "predict", "forecast", "anticipate", "plan"]
        
        return any(keyword in context.user_input.lower() for keyword in predictive_keywords)
    
    async def _get_cognitive_state_summary(self) -> Dict[str, Any]:
        """Get summary of current cognitive state"""
        return {
            "cycle_count": self.original_core.context.cognitive_cycle_count,
            "resource_allocation": self.original_core.context.resource_allocation,
            "active_strategies": await self._get_active_strategies(),
            "current_focus": self.original_core.context.attention_focus,
            "meta_parameters": self.original_core.context.meta_parameters
        }
    
    def _calculate_context_complexity(self, context: SharedContext) -> float:
        """Calculate complexity score for context"""
        complexity = 0.0
        
        # Input length factor
        input_length = len(context.user_input)
        complexity += min(1.0, input_length / 500) * 0.2
        
        # Vocabulary complexity
        words = context.user_input.split()
        unique_words = len(set(words))
        if len(words) > 0:
            complexity += (unique_words / len(words)) * 0.2
        
        # Active modules factor
        complexity += min(1.0, len(context.active_modules) / 10) * 0.2
        
        # Context updates factor
        complexity += min(1.0, len(context.context_updates) / 20) * 0.2
        
        # Multi-domain factor (multiple types of context active)
        active_contexts = sum(1 for ctx in [
            context.emotional_state,
            context.memory_context,
            context.goal_context,
            context.relationship_context
        ] if ctx)
        complexity += (active_contexts / 4) * 0.2
        
        return min(1.0, complexity)
    
    def _assess_uncertainty(self, context: SharedContext) -> float:
        """Assess uncertainty level in context"""
        uncertainty = 0.0
        
        # Question indicators
        question_words = ["what", "why", "how", "when", "where", "who", "which"]
        question_count = sum(1 for word in question_words if word in context.user_input.lower())
        uncertainty += min(1.0, question_count / 3) * 0.3
        
        # Uncertainty words
        uncertainty_words = ["maybe", "perhaps", "might", "could", "possibly", "unsure", "unclear"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in context.user_input.lower())
        uncertainty += min(1.0, uncertainty_count / 2) * 0.3
        
        # Conditional statements
        if "if" in context.user_input.lower() or "when" in context.user_input.lower():
            uncertainty += 0.2
        
        # Multiple options
        if " or " in context.user_input.lower():
            uncertainty += 0.2
        
        return min(1.0, uncertainty)
    
    def _estimate_cognitive_load(self, context: SharedContext) -> float:
        """Estimate current cognitive load"""
        load = 0.0
        
        # Active modules load
        load += min(1.0, len(context.active_modules) / 15) * 0.3
        
        # Context updates load
        load += min(1.0, len(context.context_updates) / 30) * 0.2
        
        # Module messages load
        total_messages = sum(len(msgs) for msgs in context.module_messages.values())
        load += min(1.0, total_messages / 50) * 0.2
        
        # Processing time factor (if available)
        if hasattr(context, 'processing_time'):
            load += min(1.0, context.processing_time / 5.0) * 0.3  # 5 seconds = full load
        
        return min(1.0, load)
    
    def _check_deep_reasoning_need(self, context: SharedContext) -> bool:
        """Check if deep reasoning is needed"""
        # Complex logical constructs
        logical_indicators = ["therefore", "because", "since", "implies", "follows that", "conclude"]
        
        # Abstract concepts
        abstract_indicators = ["meaning", "purpose", "significance", "essence", "nature of"]
        
        # Multi-step reasoning
        multistep_indicators = ["first", "then", "finally", "step by step", "in order to"]
        
        input_lower = context.user_input.lower()
        
        return (
            any(indicator in input_lower for indicator in logical_indicators) or
            any(indicator in input_lower for indicator in abstract_indicators) or
            any(indicator in input_lower for indicator in multistep_indicators) or
            context.user_input.count(",") > 3  # Complex sentence structure
        )
    
    def _assess_performance_risk(self, context: SharedContext) -> float:
        """Assess risk of poor performance"""
        risk = 0.0
        
        # High complexity
        risk += self._calculate_context_complexity(context) * 0.3
        
        # High uncertainty
        risk += self._assess_uncertainty(context) * 0.3
        
        # High cognitive load
        risk += self._estimate_cognitive_load(context) * 0.2
        
        # System health factor
        # (Would need to be async to properly check, so simplified here)
        if hasattr(self.original_core.context, 'system_metrics'):
            error_rate = self.original_core.context.system_metrics.get("error_rate", 0.0)
            risk += min(1.0, error_rate * 5) * 0.2  # Scale error rate
        
        return min(1.0, risk)
    
    def _determine_system_status(self, metrics: Dict[str, Any]) -> str:
        """Determine system status from metrics"""
        success_rate = metrics.get("success_rate", 0.5)
        error_rate = metrics.get("error_rate", 0.0)
        
        if success_rate > 0.8 and error_rate < 0.1:
            return "healthy"
        elif success_rate > 0.6 and error_rate < 0.2:
            return "good"
        elif success_rate > 0.4 or error_rate < 0.3:
            return "warning"
        else:
            return "critical"
    
    async def _generate_insight_application(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Generate application strategy for an insight"""
        insight_type = insight.get("type", "general")
        content = insight.get("content", {})
        
        application_strategies = {
            "bottleneck": {
                "strategy": "immediate_optimization",
                "actions": ["Reallocate resources", "Simplify processing", "Add caching"]
            },
            "pattern": {
                "strategy": "gradual_adjustment",
                "actions": ["Monitor pattern", "Adjust parameters slowly", "Test impact"]
            },
            "failure": {
                "strategy": "prevention_focus",
                "actions": ["Add validation", "Increase monitoring", "Create fallbacks"]
            }
        }
        
        strategy = application_strategies.get(insight_type, {
            "strategy": "experimental_application",
            "actions": ["Test in limited scope", "Monitor results", "Expand if successful"]
        })
        
        return {
            "insight_id": insight.get("id", "unknown"),
            "application_strategy": strategy["strategy"],
            "recommended_actions": strategy["actions"],
            "expected_timeline": "immediate" if insight_type == "bottleneck" else "gradual"
        }
    
    async def _analyze_single_system(self, system_name: str) -> Dict[str, Any]:
        """Analyze a single system in detail"""
        analysis = {
            "status": "unknown",
            "performance": {},
            "trends": {},
            "issues": [],
            "strengths": []
        }
        
        data = self.original_core.context.performance_history.get(system_name, {})
        
        if isinstance(data, dict) and "history" in data and data["history"]:
            # Get latest metrics
            latest = data["history"][-1].get("metrics", {})
            analysis["performance"] = latest
            
            # Determine status
            analysis["status"] = self._determine_system_status(latest)
            
            # Analyze trends if enough history
            if len(data["history"]) >= 3:
                # Get trend for each metric
                for metric in ["success_rate", "error_rate", "response_time", "efficiency"]:
                    values = [h.get("metrics", {}).get(metric) for h in data["history"][-5:] if metric in h.get("metrics", {})]
                    
                    if len(values) >= 3:
                        # Simple trend calculation
                        if values[-1] > values[0]:
                            trend = "improving"
                        elif values[-1] < values[0]:
                            trend = "declining"
                        else:
                            trend = "stable"
                            
                        analysis["trends"][metric] = {
                            "direction": trend,
                            "current": values[-1],
                            "change": values[-1] - values[0]
                        }
            
            # Identify issues
            if latest.get("error_rate", 0) > 0.2:
                analysis["issues"].append("High error rate")
            if latest.get("response_time", 0) > 2.0:
                analysis["issues"].append("Slow response time")
            if latest.get("success_rate", 1.0) < 0.5:
                analysis["issues"].append("Low success rate")
            
            # Identify strengths
            if latest.get("success_rate", 0) > 0.8:
                analysis["strengths"].append("High success rate")
            if latest.get("efficiency", 0) > 0.8:
                analysis["strengths"].append("High efficiency")
            if latest.get("error_rate", 1.0) < 0.05:
                analysis["strengths"].append("Low error rate")
        
        return analysis
    
    async def _find_system_correlations(self) -> List[Dict[str, Any]]:
        """Find correlations between system performances"""
        correlations = []
        
        # Simple correlation detection between system pairs
        systems = list(self.original_core.context.performance_history.keys())
        
        for i, system1 in enumerate(systems):
            for system2 in systems[i+1:]:
                correlation = await self._calculate_system_correlation(system1, system2)
                
                if abs(correlation) > 0.6:  # Significant correlation
                    correlations.append({
                        "systems": [system1, system2],
                        "correlation": correlation,
                        "type": "positive" if correlation > 0 else "negative",
                        "strength": "strong" if abs(correlation) > 0.8 else "moderate"
                    })
        
        return correlations
    
    async def _calculate_system_correlation(self, system1: str, system2: str) -> float:
        """Calculate correlation between two systems' performance"""
        data1 = self.original_core.context.performance_history.get(system1, {})
        data2 = self.original_core.context.performance_history.get(system2, {})
        
        if not (isinstance(data1, dict) and isinstance(data2, dict) and 
                "history" in data1 and "history" in data2):
            return 0.0
        
        # Get success rates for both systems
        values1 = []
        values2 = []
        
        # Find overlapping timestamps
        for entry1 in data1["history"][-10:]:  # Last 10 entries
            timestamp1 = entry1.get("timestamp")
            success1 = entry1.get("metrics", {}).get("success_rate", 0.5)
            
            # Find matching timestamp in system2
            for entry2 in data2["history"][-10:]:
                if entry2.get("timestamp") == timestamp1:
                    success2 = entry2.get("metrics", {}).get("success_rate", 0.5)
                    values1.append(success1)
                    values2.append(success2)
                    break
        
        # Calculate simple correlation
        if len(values1) < 3:
            return 0.0
        
        # Pearson correlation coefficient (simplified)
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
        denominator1 = sum((v1 - mean1) ** 2 for v1 in values1) ** 0.5
        denominator2 = sum((v2 - mean2) ** 2 for v2 in values2) ** 0.5
        
        if denominator1 * denominator2 == 0:
            return 0.0
        
        return numerator / (denominator1 * denominator2)
    
    async def _generate_system_recommendations(self, system_analyses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on system analyses"""
        recommendations = []
        
        for system_name, analysis in system_analyses.items():
            # Critical systems need immediate attention
            if analysis["status"] == "critical":
                recommendations.append({
                    "system": system_name,
                    "priority": "critical",
                    "recommendation": f"Immediate intervention needed for {system_name}",
                    "actions": [
                        "Diagnose root cause of failures",
                        "Implement emergency recovery procedures",
                        "Increase monitoring and logging"
                    ]
                })
            
            # Systems with declining trends
            elif any(t["direction"] == "declining" for t in analysis["trends"].values()):
                recommendations.append({
                    "system": system_name,
                    "priority": "high",
                    "recommendation": f"Address declining performance in {system_name}",
                    "actions": [
                        "Analyze recent changes",
                        "Review parameter adjustments",
                        "Consider rollback if necessary"
                    ]
                })
            
            # High-performing systems that could be leveraged
            elif len(analysis["strengths"]) >= 2:
                recommendations.append({
                    "system": system_name,
                    "priority": "low",
                    "recommendation": f"Leverage success of {system_name}",
                    "actions": [
                        "Document successful strategies",
                        "Apply learnings to other systems",
                        "Consider increasing responsibilities"
                    ]
                })
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))
        
        return recommendations
    
    async def _calculate_resource_efficiency(self) -> float:
        """Calculate overall resource utilization efficiency"""
        efficiency_scores = []
        
        for system_name, allocation in self.original_core.context.resource_allocation.items():
            if system_name in self.original_core.context.performance_history:
                data = self.original_core.context.performance_history[system_name]
                
                if isinstance(data, dict) and "history" in data and data["history"]:
                    latest = data["history"][-1].get("metrics", {})
                    success_rate = latest.get("success_rate", 0.5)
                    
                    # Efficiency = performance / resource allocation
                    if allocation > 0:
                        efficiency = success_rate / allocation
                        # Normalize (perfect efficiency = 1.0 success with minimal allocation)
                        normalized_efficiency = min(1.0, efficiency / 5.0)  # Cap at 5x return
                        efficiency_scores.append(normalized_efficiency)
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.5
    
    def _get_prevention_strategy(self, error_type: str) -> str:
        """Get prevention strategy for specific error type"""
        strategies = {
            "timeout": "Increase processing time limits and optimize algorithms",
            "memory_error": "Implement better memory management and cleanup",
            "validation_error": "Add comprehensive input validation",
            "reasoning_error": "Simplify reasoning chains and add verification steps",
            "context_error": "Improve context management and validation",
            "unknown": "Increase logging and monitoring to identify root cause"
        }
        
        return strategies.get(error_type, strategies["unknown"])
    
    # Delegate all other methods to the original core
    def __getattr__(self, name):
        """Delegate any missing methods to the original core"""
        return getattr(self.original_core, name)
