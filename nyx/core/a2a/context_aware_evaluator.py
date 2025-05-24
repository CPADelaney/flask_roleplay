# nyx/core/a2a/context_aware_evaluator.py

import logging
from typing import Dict, List, Any, Optional

from nyx.core.brain.integration_layer import ContextAwareModule  
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareAgentEvaluator(ContextAwareModule):
    """
    Enhanced AgentEvaluator with context distribution capabilities
    """
    
    def __init__(self, original_evaluator):
        super().__init__("agent_evaluator")
        self.original_evaluator = original_evaluator
        self.context_subscriptions = [
            "response_generated", "module_performance_data",
            "user_satisfaction_signal", "goal_completion_quality",
            "interaction_quality_metric"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize evaluation processing for this context"""
        logger.debug(f"AgentEvaluator received context for evaluation")
        
        # Prepare evaluation context
        eval_readiness = {
            "evaluator_ready": True,
            "active_modules": list(context.active_modules),
            "evaluation_scope": self._determine_evaluation_scope(context)
        }
        
        # Send initial evaluation context
        await self.send_context_update(
            update_type="evaluator_initialized", 
            data=eval_readiness,
            priority=ContextPriority.LOW
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that trigger evaluation"""
        
        if update.update_type == "response_generated":
            # Evaluate a generated response
            response_data = update.data
            await self._evaluate_response_quality(response_data)
        
        elif update.update_type == "module_performance_data":
            # Evaluate module performance
            perf_data = update.data
            await self._evaluate_module_performance(perf_data)
        
        elif update.update_type == "goal_completion_quality":
            # Evaluate goal completion quality
            goal_data = update.data
            await self._evaluate_goal_achievement(goal_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for evaluation needs"""
        # Check if input requests evaluation
        if self._requests_evaluation(context.user_input):
            eval_type = self._determine_evaluation_type(context.user_input)
            
            return {
                "evaluation_requested": True,
                "evaluation_type": eval_type,
                "evaluation_initiated": True
            }
        
        return {"evaluation_processing": False}
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze system performance in context"""
        messages = await self.get_cross_module_messages()
        
        analysis = {
            "module_coordination_quality": await self._analyze_coordination_quality(messages),
            "context_coherence": await self._analyze_context_coherence(context),
            "response_quality_factors": await self._analyze_response_factors(context, messages),
            "improvement_opportunities": await self._identify_improvements(context, messages)
        }
        
        return analysis
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize evaluation insights"""
        messages = await self.get_cross_module_messages()
        
        synthesis = {
            "performance_summary": await self._synthesize_performance_summary(context, messages),
            "quality_insights": await self._synthesize_quality_insights(context),
            "recommendations": await self._synthesize_recommendations(context, messages)
        }
        
        # Send synthesis to interested modules
        if synthesis["recommendations"]:
            await self.send_context_update(
                update_type="evaluation_recommendations",
                data={
                    "recommendations": synthesis["recommendations"],
                    "priority_improvements": synthesis.get("priority_improvements", [])
                },
                priority=ContextPriority.NORMAL
            )
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    def _determine_evaluation_scope(self, context: SharedContext) -> str:
        """Determine scope of evaluation needed"""
        if len(context.active_modules) > 5:
            return "comprehensive"
        elif len(context.active_modules) > 2:
            return "standard"
        else:
            return "minimal"
    
    def _requests_evaluation(self, user_input: str) -> bool:
        """Check if input requests evaluation"""
        eval_keywords = ["evaluate", "assessment", "performance", "quality", "how well", "review"]
        input_lower = user_input.lower()
        return any(keyword in input_lower for keyword in eval_keywords)
    
    def _determine_evaluation_type(self, user_input: str) -> str:
        """Determine what type of evaluation is requested"""
        input_lower = user_input.lower()
        
        if "response" in input_lower:
            return "response_quality"
        elif "performance" in input_lower:
            return "system_performance"
        elif "module" in input_lower:
            return "module_evaluation"
        else:
            return "general_evaluation"
    
    async def _evaluate_response_quality(self, response_data: Dict[str, Any]):
        """Evaluate quality of a generated response"""
        response_text = response_data.get("response", "")
        source_module = response_data.get("source_module", "unknown")
        
        # Use original evaluator to assess response
        if response_text:
            evaluation = await self.original_evaluator.evaluate_response(
                agent_name=source_module,
                user_input=response_data.get("user_input", ""),
                agent_output=response_text,
                dimensions=["coherence", "relevance", "completeness"]
            )
            
            # Send evaluation results
            await self.send_context_update(
                update_type="response_evaluation_complete",
                data={
                    "module": source_module,
                    "average_score": evaluation.average_score,
                    "metrics": {k: v.dict() for k, v in evaluation.metrics.items()}
                }
            )
    
    async def _evaluate_module_performance(self, perf_data: Dict[str, Any]):
        """Evaluate module performance"""
        module_name = perf_data.get("module_name")
        performance_metrics = perf_data.get("metrics", {})
        
        # Analyze performance
        performance_score = self._calculate_performance_score(performance_metrics)
        
        # Send evaluation
        await self.send_context_update(
            update_type="module_performance_evaluated",
            data={
                "module": module_name,
                "performance_score": performance_score,
                "metrics": performance_metrics
            }
        )
    
    async def _evaluate_goal_achievement(self, goal_data: Dict[str, Any]):
        """Evaluate quality of goal achievement"""
        goal_id = goal_data.get("goal_id")
        completion_quality = goal_data.get("completion_quality", 0.5)
        
        # Evaluate based on goal data
        quality_assessment = {
            "goal_id": goal_id,
            "completion_quality": completion_quality,
            "efficiency": goal_data.get("efficiency", 0.5),
            "side_effects": goal_data.get("side_effects", [])
        }
        
        # Send evaluation
        await self.send_context_update(
            update_type="goal_achievement_evaluated",
            data=quality_assessment
        )
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score from metrics"""
        # Simple weighted average
        weights = {
            "response_time": 0.2,
            "accuracy": 0.3,
            "completeness": 0.3,
            "efficiency": 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.5
    
    async def _analyze_coordination_quality(self, messages: Dict) -> Dict[str, Any]:
        """Analyze quality of module coordination"""
        coordination_metrics = {
            "message_count": sum(len(msgs) for msgs in messages.values()),
            "active_modules": len(messages),
            "message_patterns": {}
        }
        
        # Analyze message patterns
        for module, module_messages in messages.items():
            coordination_metrics["message_patterns"][module] = {
                "sent": len(module_messages),
                "types": list(set(msg.get("type", "unknown") for msg in module_messages))
            }
        
        # Calculate coordination score
        if coordination_metrics["active_modules"] > 1:
            # More modules coordinating = better
            coordination_score = min(1.0, coordination_metrics["active_modules"] / 5.0)
        else:
            coordination_score = 0.3
        
        coordination_metrics["coordination_score"] = coordination_score
        
        return coordination_metrics
    
    async def _analyze_context_coherence(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze coherence of the shared context"""
        coherence_factors = {
            "emotional_coherence": 1.0,
            "goal_coherence": 1.0,
            "memory_coherence": 1.0,
            "spatial_coherence": 1.0
        }
        
        # Check emotional coherence
        if context.emotional_state and context.goal_context:
            # Check if emotions align with goals
            if "Frustration" in context.emotional_state and context.emotional_state["Frustration"] > 0.7:
                if context.goal_context.get("goals_blocked", False):
                    coherence_factors["emotional_coherence"] = 1.0  # Coherent
                else:
                    coherence_factors["emotional_coherence"] = 0.5  # Less coherent
        
        # Calculate overall coherence
        overall_coherence = sum(coherence_factors.values()) / len(coherence_factors)
        
        return {
            "overall_coherence": overall_coherence,
            "factors": coherence_factors,
            "issues": [k for k, v in coherence_factors.items() if v < 0.7]
        }
    
    async def _analyze_response_factors(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze factors affecting response quality"""
        factors = {
            "context_completeness": len(context.context_updates) / 10.0,  # Normalized
            "module_participation": len(context.active_modules) / 10.0,   # Normalized
            "processing_stages_completed": len(context.module_outputs),
            "synthesis_quality": 0.5  # Default
        }
        
        # Check synthesis quality
        if context.synthesis_results:
            if "primary_response" in context.synthesis_results:
                factors["synthesis_quality"] = 0.8
        
        return factors
    
    async def _identify_improvements(self, context: SharedContext, messages: Dict) -> List[Dict[str, Any]]:
        """Identify improvement opportunities"""
        improvements = []
        
        # Check for missing modules
        expected_modules = {"emotional_core", "memory_core", "goal_manager"}
        active_modules = set(context.active_modules)
        missing = expected_modules - active_modules
        
        for module in missing:
            improvements.append({
                "type": "missing_module",
                "module": module,
                "impact": "reduced_context_awareness",
                "recommendation": f"Activate {module} for better results"
            })
        
        # Check for low coordination
        coordination = await self._analyze_coordination_quality(messages)
        if coordination["coordination_score"] < 0.5:
            improvements.append({
                "type": "low_coordination",
                "score": coordination["coordination_score"],
                "impact": "fragmented_responses",
                "recommendation": "Increase cross-module communication"
            })
        
        return improvements
    
    async def _synthesize_performance_summary(self, context: SharedContext, messages: Dict) -> str:
        """Synthesize overall performance summary"""
        # Get metrics
        coordination = await self._analyze_coordination_quality(messages)
        coherence = await self._analyze_context_coherence(context)
        
        # Create summary
        if coordination["coordination_score"] > 0.7 and coherence["overall_coherence"] > 0.7:
            return "System performing well with good coordination and coherence"
        elif coordination["coordination_score"] < 0.5:
            return "System performance limited by low module coordination"
        elif coherence["overall_coherence"] < 0.5:
            return "System showing context coherence issues"
        else:
            return "System performing at moderate levels"
    
    async def _synthesize_quality_insights(self, context: SharedContext) -> List[str]:
        """Synthesize quality insights"""
        insights = []
        
        # Check processing completeness
        if len(context.module_outputs) >= 3:
            insights.append("Multi-stage processing completed successfully")
        
        # Check context richness
        if len(context.context_updates) > 10:
            insights.append("Rich context established through extensive updates")
        
        # Check synthesis
        if context.synthesis_results and "primary_response" in context.synthesis_results:
            insights.append("Response synthesis achieved successfully")
        
        return insights
    
    async def _synthesize_recommendations(self, context: SharedContext, messages: Dict) -> List[str]:
        """Synthesize improvement recommendations"""
        recommendations = []
        
        # Get improvement opportunities
        improvements = await self._identify_improvements(context, messages)
        
        # Convert to recommendations
        for improvement in improvements[:3]:  # Top 3
            recommendations.append(improvement["recommendation"])
        
        return recommendations
    
    # Delegate all other methods to the original evaluator
    def __getattr__(self, name):
        """Delegate any missing methods to the original evaluator"""
        return getattr(self.original_evaluator, name)
