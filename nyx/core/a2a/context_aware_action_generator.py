# nyx/core/a2a/context_aware_action_generator.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareAgenticActionGenerator(ContextAwareModule):
    """
    Enhanced AgenticActionGenerator with full context distribution capabilities
    """
    
    def __init__(self, original_action_generator):
        super().__init__("agentic_action_generator")
        self.original_generator = original_action_generator
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", "goal_progress",
            "relationship_state_change", "need_satisfaction", "mood_state_update",
            "mode_state_change", "sensory_state_update", "observation_available",
            "communication_intent", "reasoning_insight", "reflection_complete"
        ]
        
    async def on_context_received(self, context: SharedContext):
        """Initialize action generation for this context"""
        logger.debug(f"AgenticActionGenerator received context for user: {context.user_id}")
        
        # Gather initial action context from all systems
        action_context = await self._build_action_context(context)
        
        # Analyze possible actions based on current state
        available_actions = await self._identify_available_actions(context)
        
        # Determine action priorities based on context
        action_priorities = await self._calculate_action_priorities(context, available_actions)
        
        # Send initial action context to other modules
        await self.send_context_update(
            update_type="action_context_available",
            data={
                "action_context": action_context,
                "available_actions": available_actions,
                "action_priorities": action_priorities,
                "current_motivations": self.original_generator.motivations,
                "exploration_rate": self.original_generator.exploration_rate
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect action generation"""
        
        if update.update_type == "emotional_state_update":
            # Emotional changes affect motivation and action selection
            emotional_data = update.data
            await self._adjust_motivations_from_emotion(emotional_data)
            
        elif update.update_type == "goal_progress":
            # Goal progress affects action priorities
            goal_data = update.data
            await self._update_goal_based_actions(goal_data)
            
        elif update.update_type == "need_satisfaction":
            # Need satisfaction affects urgency of need-based actions
            need_data = update.data
            await self._update_need_based_motivations(need_data)
            
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect social action priorities
            relationship_data = update.data
            await self._adjust_social_actions(relationship_data)
            
        elif update.update_type == "mood_state_update":
            # Mood affects action style and energy
            mood_data = update.data
            await self._adjust_action_style_from_mood(mood_data)
            
        elif update.update_type == "sensory_state_update":
            # Physical sensations might trigger specific actions
            sensory_data = update.data
            await self._process_sensory_driven_actions(sensory_data)
            
        elif update.update_type == "observation_available":
            # New observations might suggest actions
            observation_data = update.data
            await self._consider_observation_based_actions(observation_data)
            
        elif update.update_type == "reasoning_insight":
            # Reasoning insights might suggest new action strategies
            reasoning_data = update.data
            await self._integrate_reasoning_insights(reasoning_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input to generate optimal action with full context"""
        # Get cross-module messages for comprehensive context
        messages = await self.get_cross_module_messages()
        
        # Build comprehensive action context
        full_action_context = await self._build_comprehensive_action_context(context, messages)
        
        # Generate optimal action using the original generator's enhanced method
        action = await self.original_generator.generate_optimal_action(full_action_context)
        
        # Send action generation update
        await self.send_context_update(
            update_type="action_generated",
            data={
                "action": action,
                "action_source": action.get("source"),
                "action_id": action.get("id"),
                "selection_metadata": action.get("selection_metadata", {}),
                "context_factors": self._extract_context_factors(full_action_context)
            }
        )
        
        return {
            "action": action,
            "context_aware": True,
            "cross_module_inputs": len(messages),
            "action_generation_complete": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze action space and opportunities in current context"""
        messages = await self.get_cross_module_messages()
        
        # Analyze action success patterns
        success_analysis = self._analyze_action_success_patterns()
        
        # Identify action opportunities based on context
        opportunities = await self._identify_action_opportunities(context, messages)
        
        # Analyze action-consequence relationships
        consequence_analysis = await self._analyze_action_consequences(context)
        
        # Strategy effectiveness analysis
        strategy_analysis = self._analyze_strategy_effectiveness()
        
        return {
            "success_patterns": success_analysis,
            "action_opportunities": opportunities,
            "consequence_analysis": consequence_analysis,
            "strategy_effectiveness": strategy_analysis,
            "current_exploration_rate": self.original_generator.exploration_rate,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize action recommendations for response"""
        messages = await self.get_cross_module_messages()
        
        # Get recent action history
        recent_actions = self.original_generator.action_history[-5:] if hasattr(self.original_generator, 'action_history') else []
        
        # Synthesize action influence on response
        action_synthesis = {
            "recent_actions": self._summarize_recent_actions(recent_actions),
            "action_continuity": self._check_action_continuity(recent_actions),
            "suggested_next_actions": await self._suggest_next_actions(context, messages),
            "action_style_modifiers": self._get_action_style_modifiers(context),
            "proactive_action_suggestions": await self._generate_proactive_suggestions(context)
        }
        
        # Check if any critical actions are needed
        critical_actions = await self._check_critical_actions(context, messages)
        if critical_actions:
            await self.send_context_update(
                update_type="critical_action_needed",
                data={
                    "critical_actions": critical_actions,
                    "urgency": "high"
                },
                priority=ContextPriority.CRITICAL
            )
        
        return action_synthesis
    
    async def on_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]):
        """Special handler for when an action completes with outcome"""
        # Record outcome using original generator
        learning_result = await self.original_generator.record_action_outcome(action, outcome)
        
        # Send outcome update to interested modules
        await self.send_context_update(
            update_type="action_outcome",
            data={
                "action": action,
                "outcome": outcome,
                "learning_result": learning_result,
                "success": outcome.get("success", False),
                "reward_value": outcome.get("reward_value", 0.0)
            },
            priority=ContextPriority.HIGH
        )
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _build_action_context(self, context: SharedContext) -> Dict[str, Any]:
        """Build initial action context from shared context"""
        return {
            "state": context.session_context,
            "user_id": context.user_id,
            "relationship_data": context.relationship_context,
            "emotional_state": context.emotional_state,
            "goal_context": context.goal_context,
            "memory_context": context.memory_context,
            "motivations": self.original_generator.motivations,
            "active_modules": list(context.active_modules)
        }
    
    async def _identify_available_actions(self, context: SharedContext) -> List[str]:
        """Identify available actions based on context"""
        available = []
        
        # Get registered actions from the generator
        if hasattr(self.original_generator, 'action_handlers'):
            available.extend(list(self.original_generator.action_handlers.keys()))
        
        # Add context-specific actions
        if context.goal_context and context.goal_context.get("active_goals"):
            available.append("work_on_goal")
            
        if context.relationship_context:
            available.extend(["express_affection", "build_trust", "share_experience"])
            
        return available
    
    async def _calculate_action_priorities(self, context: SharedContext, actions: List[str]) -> Dict[str, float]:
        """Calculate priority scores for available actions"""
        priorities = {}
        
        for action in actions:
            base_priority = 0.5
            
            # Adjust based on motivations
            if action in ["explore", "learn", "investigate"]:
                base_priority += self.original_generator.motivations.get("curiosity", 0) * 0.3
            elif action in ["connect", "share", "express"]:
                base_priority += self.original_generator.motivations.get("connection", 0) * 0.3
            
            priorities[action] = min(1.0, base_priority)
            
        return priorities
    
    async def _build_comprehensive_action_context(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Build comprehensive context including cross-module messages"""
        base_context = await self._build_action_context(context)
        
        # Enhance with cross-module information
        base_context["cross_module_context"] = {}
        
        for module_name, module_messages in messages.items():
            if module_messages:
                latest_message = module_messages[-1]  # Get most recent
                base_context["cross_module_context"][module_name] = latest_message.get("data", {})
        
        return base_context
    
    async def _adjust_motivations_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust motivations based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        
        # Map emotions to motivation adjustments
        emotion_motivation_map = {
            "Joy": {"expression": 0.2, "connection": 0.15},
            "Curiosity": {"curiosity": 0.3, "exploration": 0.2},
            "Frustration": {"dominance": 0.2, "autonomy": 0.15},
            "Love": {"connection": 0.3, "expression": 0.2},
            "Pride": {"validation": 0.2, "competence": 0.15}
        }
        
        for emotion, strength in emotional_state.items():
            if emotion in emotion_motivation_map and strength > 0.3:
                adjustments = emotion_motivation_map[emotion]
                for motivation, adjustment in adjustments.items():
                    if motivation in self.original_generator.motivations:
                        current = self.original_generator.motivations[motivation]
                        self.original_generator.motivations[motivation] = min(0.9, current + adjustment * strength)
    
    async def _update_goal_based_actions(self, goal_data: Dict[str, Any]):
        """Update action priorities based on goal progress"""
        if goal_data.get("goal_completed"):
            # Boost competence and self-improvement motivations
            self.original_generator.motivations["competence"] = min(0.9, 
                self.original_generator.motivations.get("competence", 0.5) + 0.2)
        elif goal_data.get("goal_blocked"):
            # Might increase frustration, adjust strategies
            self.original_generator.motivations["autonomy"] = min(0.9,
                self.original_generator.motivations.get("autonomy", 0.5) + 0.15)
    
    def _extract_context_factors(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key context factors that influenced action selection"""
        return {
            "dominant_motivation": max(self.original_generator.motivations.items(), 
                                     key=lambda x: x[1])[0] if self.original_generator.motivations else None,
            "emotional_influence": bool(action_context.get("emotional_state")),
            "goal_influence": bool(action_context.get("goal_context")),
            "relationship_influence": bool(action_context.get("relationship_data")),
            "cross_module_influence": bool(action_context.get("cross_module_context"))
        }
    
    def _analyze_action_success_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in action success rates"""
        if not hasattr(self.original_generator, 'action_success_rates'):
            return {}
            
        success_rates = self.original_generator.action_success_rates
        
        # Find most and least successful actions
        sorted_actions = sorted(success_rates.items(), 
                              key=lambda x: x[1]["rate"], 
                              reverse=True)
        
        return {
            "most_successful": sorted_actions[:3] if sorted_actions else [],
            "least_successful": sorted_actions[-3:] if len(sorted_actions) > 3 else [],
            "overall_success_rate": sum(a[1]["rate"] for a in sorted_actions) / len(sorted_actions) if sorted_actions else 0.5
        }
    
    # Delegate all other methods to the original generator
    def __getattr__(self, name):
        """Delegate any missing methods to the original generator"""
        return getattr(self.original_generator, name)
