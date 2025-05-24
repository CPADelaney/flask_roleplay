# nyx/core/a2a/context_aware_nyx_sync_daemon.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareNyxSyncDaemon(ContextAwareModule):
    """
    Enhanced NyxSyncDaemon with context distribution capabilities
    """
    
    def __init__(self, original_daemon):
        super().__init__("nyx_sync_daemon")
        self.original_daemon = original_daemon
        self.context_subscriptions = [
            "strategy_activation_request", "user_state_change", 
            "goal_strategy_alignment", "emotional_strategy_trigger",
            "scene_activation_trigger"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize sync daemon processing for this context"""
        logger.debug(f"NyxSyncDaemon received context for sync operations")
        
        # Check if any sync operations are needed
        sync_analysis = await self._analyze_sync_needs(context)
        
        # Send initial sync context
        await self.send_context_update(
            update_type="sync_daemon_initialized",
            data={
                "active_strategies": await self._get_active_strategy_count(),
                "sync_analysis": sync_analysis,
                "daemon_ready": True
            },
            priority=ContextPriority.LOW
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that might trigger sync operations"""
        
        if update.update_type == "strategy_activation_request":
            # Request to activate a specific strategy
            strategy_data = update.data
            await self._activate_strategy(strategy_data)
        
        elif update.update_type == "user_state_change":
            # User state changed, might need strategy adjustment
            state_data = update.data
            await self._adjust_strategies_for_state(state_data)
        
        elif update.update_type == "emotional_strategy_trigger":
            # Emotional state triggers strategy consideration
            emotional_data = update.data
            await self._consider_emotional_strategies(emotional_data)
        
        elif update.update_type == "goal_strategy_alignment":
            # Goal manager requests strategy alignment
            goal_data = update.data
            await self._align_strategies_with_goals(goal_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for sync-related operations"""
        # Check if input mentions strategies or scenes
        if self._mentions_strategy_concepts(context.user_input):
            # Analyze what kind of strategy adjustment might be needed
            strategy_intent = await self._analyze_strategy_intent(context)
            
            if strategy_intent:
                await self.send_context_update(
                    update_type="strategy_intent_detected",
                    data={
                        "intent": strategy_intent,
                        "user_input": context.user_input[:200]
                    }
                )
                
                return {
                    "sync_processing": True,
                    "strategy_intent": strategy_intent
                }
        
        return {"sync_processing": False}
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze sync state and strategy effectiveness"""
        messages = await self.get_cross_module_messages()
        
        analysis = {
            "active_strategy_analysis": await self._analyze_active_strategies(),
            "strategy_effectiveness": await self._analyze_strategy_effectiveness(messages),
            "scene_readiness": await self._analyze_scene_readiness(context),
            "sync_recommendations": await self._generate_sync_recommendations(context, messages)
        }
        
        return analysis
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize sync daemon insights for response"""
        messages = await self.get_cross_module_messages()
        
        synthesis = {
            "strategy_status": await self._synthesize_strategy_status(),
            "sync_insights": await self._synthesize_sync_insights(messages),
            "recommended_adjustments": await self._synthesize_adjustments(context, messages)
        }
        
        # Check if any critical sync operations are needed
        if await self._check_critical_sync_needs(context, messages):
            synthesis["critical_sync_needed"] = True
            
            await self.send_context_update(
                update_type="critical_sync_required",
                data={
                    "reason": "Context analysis indicates critical sync need",
                    "recommended_action": "immediate_strategy_review"
                },
                priority=ContextPriority.CRITICAL
            )
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_sync_needs(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze what sync operations might be needed"""
        needs = {
            "strategy_update_needed": False,
            "scene_activation_possible": False,
            "user_model_sync_needed": False
        }
        
        # Check various factors
        # This is simplified - would have more complex logic
        if context.emotional_state:
            # High emotional intensity might need strategy adjustment
            max_emotion = max(context.emotional_state.values()) if context.emotional_state else 0
            if max_emotion > 0.8:
                needs["strategy_update_needed"] = True
        
        if context.goal_context:
            # Active goals might need strategy support
            active_goals = context.goal_context.get("active_goals", [])
            if len(active_goals) > 3:
                needs["strategy_update_needed"] = True
        
        return needs
    
    async def _get_active_strategy_count(self) -> int:
        """Get count of active strategies"""
        # Would query database through original daemon
        # Simplified version
        return 0
    
    def _mentions_strategy_concepts(self, user_input: str) -> bool:
        """Check if input mentions strategy-related concepts"""
        strategy_keywords = [
            "strategy", "approach", "method", "technique", "style",
            "scene", "scenario", "intensity", "dynamic"
        ]
        
        input_lower = user_input.lower()
        return any(keyword in input_lower for keyword in strategy_keywords)
    
    async def _analyze_strategy_intent(self, context: SharedContext) -> Optional[str]:
        """Analyze what kind of strategy adjustment user might want"""
        input_lower = context.user_input.lower()
        
        if "more intense" in input_lower or "increase" in input_lower:
            return "increase_intensity"
        elif "less intense" in input_lower or "decrease" in input_lower:
            return "decrease_intensity"
        elif "different" in input_lower or "change" in input_lower:
            return "change_strategy"
        elif "scene" in input_lower:
            return "activate_scene"
        
        return None
    
    async def _activate_strategy(self, strategy_data: Dict[str, Any]):
        """Activate a specific strategy"""
        strategy_name = strategy_data.get("strategy_name")
        strategy_type = strategy_data.get("strategy_type", "general")
        
        logger.info(f"Activating strategy: {strategy_name}")
        
        # Would interface with database through original daemon
        # For now, just log and send update
        await self.send_context_update(
            update_type="strategy_activated",
            data={
                "strategy_name": strategy_name,
                "strategy_type": strategy_type,
                "activation_time": datetime.now().isoformat()
            },
            priority=ContextPriority.HIGH
        )
    
    async def _adjust_strategies_for_state(self, state_data: Dict[str, Any]):
        """Adjust strategies based on user state change"""
        # Would analyze state change and adjust strategies accordingly
        new_state = state_data.get("new_state")
        
        if new_state:
            logger.info(f"Adjusting strategies for user state: {new_state}")
    
    async def _consider_emotional_strategies(self, emotional_data: Dict[str, Any]):
        """Consider strategies based on emotional triggers"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if dominant_emotion:
            emotion_name, strength = dominant_emotion
            
            # High-intensity emotions might trigger specific strategies
            if strength > 0.8:
                logger.info(f"Considering strategies for high {emotion_name} (strength: {strength})")
                
                # Would select appropriate strategies based on emotion
                strategy_mapping = {
                    "Excitement": "escalation_strategy",
                    "Anxiety": "comfort_strategy",
                    "Curiosity": "exploration_strategy",
                    "Frustration": "redirection_strategy"
                }
                
                if emotion_name in strategy_mapping:
                    await self._activate_strategy({
                        "strategy_name": strategy_mapping[emotion_name],
                        "strategy_type": "emotional_response",
                        "trigger": f"{emotion_name}_{strength}"
                    })
    
    async def _align_strategies_with_goals(self, goal_data: Dict[str, Any]):
        """Align active strategies with current goals"""
        active_goals = goal_data.get("active_goals", [])
        
        for goal in active_goals:
            goal_type = goal.get("associated_need")
            if goal_type:
                # Would map goals to appropriate strategies
                logger.info(f"Aligning strategies for goal type: {goal_type}")
    
    async def _analyze_active_strategies(self) -> Dict[str, Any]:
        """Analyze currently active strategies"""
        # Would query database for active strategies
        # Simplified version
        return {
            "total_active": 0,
            "strategy_types": [],
            "average_effectiveness": 0.0,
            "oldest_strategy_age": None
        }
    
    async def _analyze_strategy_effectiveness(self, messages: Dict) -> Dict[str, Any]:
        """Analyze effectiveness of current strategies"""
        effectiveness = {
            "overall_score": 0.5,
            "goal_alignment": 0.0,
            "emotional_alignment": 0.0,
            "user_satisfaction": 0.0
        }
        
        # Check goal alignment
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg.get("type") == "goal_progress":
                        # Progress indicates strategy effectiveness
                        progress = msg.get("data", {}).get("execution_result", {})
                        if progress:
                            effectiveness["goal_alignment"] = 0.7
        
        return effectiveness
    
    async def _analyze_scene_readiness(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze readiness for scene activation"""
        readiness = {
            "scene_appropriate": False,
            "user_receptive": True,
            "context_suitable": True
        }
        
        # Check various factors for scene appropriateness
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0)
            if trust > 0.6:
                readiness["scene_appropriate"] = True
        
        return readiness
    
    async def _generate_sync_recommendations(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate recommendations for sync operations"""
        recommendations = []
        
        # Check if strategies need updating
        effectiveness = await self._analyze_strategy_effectiveness(messages)
        if effectiveness["overall_score"] < 0.3:
            recommendations.append("Consider updating active strategies for better effectiveness")
        
        # Check emotional alignment
        if context.emotional_state:
            max_emotion = max(context.emotional_state.values()) if context.emotional_state else 0
            if max_emotion > 0.8:
                recommendations.append("High emotional intensity detected - consider emotional response strategies")
        
        return recommendations
    
    async def _synthesize_strategy_status(self) -> str:
        """Synthesize current strategy status"""
        # Would get actual status from database
        active_count = await self._get_active_strategy_count()
        
        if active_count == 0:
            return "No active strategies"
        elif active_count == 1:
            return "One active strategy in place"
        else:
            return f"{active_count} active strategies coordinating response"
    
    async def _synthesize_sync_insights(self, messages: Dict) -> List[str]:
        """Synthesize insights from sync operations"""
        insights = []
        
        # Check for cross-module patterns
        module_activity = len(messages)
        if module_activity > 5:
            insights.append("High cross-module activity detected")
        
        # Check for strategy triggers
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if "trigger" in msg.get("type", "").lower():
                    insights.append(f"Strategy trigger detected from {module_name}")
        
        return insights[:3]
    
    async def _synthesize_adjustments(self, context: SharedContext, messages: Dict) -> List[str]:
        """Synthesize recommended adjustments"""
        adjustments = []
        
        # Based on context analysis
        sync_needs = await self._analyze_sync_needs(context)
        
        if sync_needs["strategy_update_needed"]:
            adjustments.append("Strategy update recommended based on context")
        
        if sync_needs["scene_activation_possible"]:
            adjustments.append("Scene activation possible if desired")
        
        return adjustments
    
    async def _check_critical_sync_needs(self, context: SharedContext, messages: Dict) -> bool:
        """Check if there are critical sync needs"""
        # Check for critical conditions
        
        # High emotional distress
        if context.emotional_state:
            distress_emotions = ["Anxiety", "Fear", "Frustration"]
            distress_level = sum(context.emotional_state.get(e, 0) for e in distress_emotions)
            if distress_level > 2.0:  # High combined distress
                return True
        
        # Goal crisis
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg.get("type") == "goal_crisis":
                        return True
        
        return False
    
    # Delegate all other methods to the original daemon
    def __getattr__(self, name):
        """Delegate any missing methods to the original daemon"""
        return getattr(self.original_daemon, name)
