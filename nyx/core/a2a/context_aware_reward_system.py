# nyx/core/a2a/context_aware_reward_system.py

class ContextAwareRewardSystem(ContextAwareModule):
    """
    Enhanced RewardSignalProcessor with context distribution
    """
    
    def __init__(self, original_reward_system):
        super().__init__("reward_system")
        self.original_system = original_reward_system
        self.context_subscriptions = [
            "goal_completion", "goal_progress", "dominance_gratification",
            "need_satisfied", "emotional_peak", "relationship_milestone",
            "learning_outcome", "habit_trigger", "identity_update"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize reward processing for this context"""
        logger.debug(f"RewardSystem received context for user: {context.user_id}")
        
        # Get current reward statistics
        stats = await self.original_system.get_reward_statistics()
        
        # Analyze input for potential rewards
        reward_potential = await self._analyze_reward_potential(context)
        
        # Send initial reward context
        await self.send_context_update(
            update_type="reward_context_available",
            data={
                "current_nyxamine": self.original_system.current_nyxamine,
                "reward_statistics": stats,
                "reward_potential": reward_potential,
                "exploration_rate": self.original_system.exploration_rate
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Process updates that generate rewards"""
        reward_signal = None
        
        if update.update_type == "goal_completion":
            # Goal completions generate strong rewards
            goal_data = update.data
            success = goal_data.get("status") == "completed"
            priority = goal_data.get("priority", 0.5)
            
            reward_value = priority * (0.8 if success else -0.4)
            reward_signal = RewardSignal(
                value=reward_value,
                source="goal_completion",
                context={
                    "goal_id": goal_data.get("goal_id"),
                    "goal_description": goal_data.get("description"),
                    "associated_need": goal_data.get("associated_need"),
                    "action": f"complete_goal_{goal_data.get('associated_need', 'general')}"
                }
            )
        
        elif update.update_type == "dominance_gratification":
            # Dominance events generate specialized rewards
            dom_data = update.data
            intensity = dom_data.get("intensity", 0.5)
            success = dom_data.get("success", True)
            
            reward_value = intensity * (0.9 if success else -0.3)
            reward_signal = RewardSignal(
                value=reward_value,
                source="dominance_gratification",
                context={
                    "dominance_type": dom_data.get("type"),
                    "intensity": intensity,
                    "user_response": dom_data.get("user_response"),
                    "action": f"dominance_{dom_data.get('type', 'general')}"
                }
            )
        
        elif update.update_type == "need_satisfied":
            # Need satisfaction generates moderate rewards
            need_data = update.data
            need_name = need_data.get("need_name")
            satisfaction_level = need_data.get("satisfaction_level", 0.5)
            
            reward_value = satisfaction_level * 0.6
            reward_signal = RewardSignal(
                value=reward_value,
                source="need_satisfaction",
                context={
                    "need_name": need_name,
                    "satisfaction_level": satisfaction_level,
                    "action": f"satisfy_need_{need_name}"
                }
            )
        
        elif update.update_type == "emotional_peak":
            # Strong emotions can generate rewards
            emotion_data = update.data
            valence = emotion_data.get("valence", 0)
            arousal = emotion_data.get("arousal", 0.5)
            
            if abs(valence) > 0.7 and arousal > 0.6:
                reward_value = valence * arousal * 0.5
                reward_signal = RewardSignal(
                    value=reward_value,
                    source="emotional_peak",
                    context={
                        "emotion": emotion_data.get("dominant_emotion"),
                        "valence": valence,
                        "arousal": arousal,
                        "action": "emotional_experience"
                    }
                )
        
        # Process the reward signal if generated
        if reward_signal:
            result = await self.original_system.process_reward_signal(reward_signal)
            
            # Notify other modules of reward processing
            await self.send_context_update(
                update_type="reward_processed",
                data={
                    "reward_value": reward_signal.value,
                    "reward_source": reward_signal.source,
                    "nyxamine_change": result.get("nyxamine_change"),
                    "current_nyxamine": result.get("current_nyxamine"),
                    "learning_updates": result.get("learning")
                }
            )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for reward opportunities"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Extract state for action prediction
        current_state = {
            "emotional_state": context.emotional_state,
            "relationship_state": context.relationship_context,
            "goal_state": context.goal_context,
            "user_input": context.user_input
        }
        
        # Get available actions from context
        available_actions = self._extract_available_actions(context, messages)
        
        # Predict best action if we have options
        action_prediction = {}
        if available_actions:
            action_prediction = await self.original_system.predict_best_action(
                current_state, available_actions
            )
        
        # Check for submission/compliance rewards
        submission_check = await self._check_submission_reward(context, messages)
        if submission_check:
            await self.original_system.process_submission_reward(
                submission_check["type"],
                submission_check["compliance_level"],
                context.user_id,
                submission_check.get("novelty", 0.5),
                submission_check.get("was_resistant", False)
            )
        
        return {
            "reward_processing_active": True,
            "action_prediction": action_prediction,
            "submission_processed": bool(submission_check),
            "current_nyxamine": self.original_system.current_nyxamine
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze reward patterns and learning"""
        # Get current statistics
        stats = await self.original_system.get_reward_statistics()
        
        # Analyze reward patterns if enough data
        pattern_analysis = {}
        if stats["reward_memories_count"] > 50:
            pattern_analysis = await self.original_system.analyze_reward_patterns()
        
        # Get messages for context
        messages = await self.get_cross_module_messages()
        
        # Identify learning opportunities
        learning_opportunities = await self._identify_learning_opportunities(context, messages)
        
        return {
            "reward_statistics": stats,
            "pattern_analysis": pattern_analysis,
            "learning_opportunities": learning_opportunities,
            "nyxamine_level": self.original_system.current_nyxamine,
            "exploration_vs_exploitation": {
                "exploration_rate": self.original_system.exploration_rate,
                "learned_actions": stats["learned_action_count"],
                "reliable_actions": self._count_reliable_actions()
            }
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize reward-informed response components"""
        # Get current nyxamine level for mood modulation
        nyxamine = self.original_system.current_nyxamine
        
        # Determine reward-seeking vs satisfaction state
        reward_state = "seeking" if nyxamine < 0.4 else "satisfied" if nyxamine > 0.7 else "neutral"
        
        # Get top performing actions for reference
        stats = await self.original_system.get_reward_statistics()
        top_actions = stats.get("top_performing_actions", [])
        
        # Generate reward-informed suggestions
        response_suggestions = {
            "reward_state": reward_state,
            "suggest_rewarding_actions": reward_state == "seeking",
            "preferred_actions": [action["action"] for action in top_actions[:3]],
            "nyxamine_influence": {
                "motivation_level": nyxamine,
                "seek_stimulation": nyxamine < 0.3,
                "express_satisfaction": nyxamine > 0.8
            }
        }
        
        # Send synthesis update
        await self.send_context_update(
            update_type="reward_synthesis",
            data={
                "response_suggestions": response_suggestions,
                "current_nyxamine": nyxamine,
                "reward_state": reward_state
            }
        )
        
        return {
            "reward_synthesis": response_suggestions,
            "synthesis_complete": True
        }
    
    # Helper methods
    async def _analyze_reward_potential(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze potential rewards in current context"""
        potential = {
            "goal_related": bool(context.goal_context),
            "emotional_intensity": max(context.emotional_state.values()) if context.emotional_state else 0,
            "relationship_depth": context.relationship_context.get("intimacy", 0) if context.relationship_context else 0,
            "dominance_opportunity": "submit" in context.user_input.lower() or "obey" in context.user_input.lower()
        }
        
        # Calculate overall reward potential
        potential["overall"] = sum([
            potential["goal_related"] * 0.3,
            potential["emotional_intensity"] * 0.2,
            potential["relationship_depth"] * 0.2,
            potential["dominance_opportunity"] * 0.3
        ])
        
        return potential
    
    def _count_reliable_actions(self) -> int:
        """Count actions with reliable Q-values"""
        count = 0
        for state_actions in self.original_system.action_values.values():
            for action_value in state_actions.values():
                if action_value.is_reliable:
                    count += 1
        return count
    
    def __getattr__(self, name):
        return getattr(self.original_system, name)
