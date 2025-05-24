# nyx/core/a2a/context_aware_relationship_manager.py
from typing import Dict, Any
from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

class ContextAwareRelationshipManager(ContextAwareModule):
    """
    Enhanced RelationshipManager with full context distribution capabilities
    """
    
    def __init__(self, original_relationship_manager):
        super().__init__("relationship_manager")
        self.original_manager = original_relationship_manager
        self.context_subscriptions = [
            "emotional_state_update", "goal_completion", "goal_progress",
            "memory_retrieval_complete", "dominance_gratification", 
            "need_satisfied", "reward_signal", "temporal_milestone",
            "identity_update", "conditioning_event"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize relationship processing for this context"""
        logger.debug(f"RelationshipManager received context for user: {context.user_id}")
        
        # Get or create relationship for this user
        relationship_state = await self.original_manager.get_or_create_relationship_internal(context.user_id)
        
        # Analyze user input for relationship implications
        relationship_implications = await self._analyze_input_for_relationship(context.user_input)
        
        # Get interaction history for context
        interaction_history = await self.original_manager.get_interaction_history_internal(
            context.user_id, limit=5
        )
        
        # Send initial relationship context to other modules
        await self.send_context_update(
            update_type="relationship_context_available",
            data={
                "relationship_state": relationship_state,
                "interaction_history": interaction_history,
                "relationship_implications": relationship_implications,
                "trust": relationship_state.get("trust", 0.5),
                "intimacy": relationship_state.get("intimacy", 0.1),
                "familiarity": relationship_state.get("familiarity", 0.1),
                "dominance_balance": relationship_state.get("dominance_balance", 0.0)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect relationships"""
        user_id = update.data.get("user_id") or getattr(self._context_system.current_context, "user_id", None)
        
        if update.update_type == "emotional_state_update":
            # Emotional states affect relationship dynamics
            emotional_data = update.data
            await self._process_emotional_impact(user_id, emotional_data)
        
        elif update.update_type == "goal_completion":
            # Goal completions can strengthen relationships
            goal_data = update.data
            if goal_data.get("associated_need") in ["connection", "intimacy"]:
                await self._process_relationship_goal_completion(user_id, goal_data)
        
        elif update.update_type == "dominance_gratification":
            # Dominance events significantly impact relationship dynamics
            dominance_data = update.data
            await self._process_dominance_event(user_id, dominance_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Shared memories affect relationship depth
            memory_data = update.data
            if memory_data.get("memory_type") == "shared_experience":
                await self._process_shared_memory(user_id, memory_data)
        
        elif update.update_type == "temporal_milestone":
            # Time milestones are relationship events
            milestone_data = update.data
            await self._process_temporal_milestone(user_id, milestone_data)
        
        elif update.update_type == "reward_signal":
            # Significant rewards affect relationship
            reward_data = update.data
            if abs(reward_data.get("value", 0)) > 0.7:
                await self._process_reward_impact(user_id, reward_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with relationship awareness"""
        user_id = context.user_id
        
        # Get cross-module messages for relationship context
        messages = await self.get_cross_module_messages()
        
        # Analyze input for relationship signals
        input_analysis = await self._analyze_relationship_signals(context.user_input, messages)
        
        # Check if this is a dominance interaction
        is_dominance = await self._check_dominance_context(context, messages)
        
        # Prepare interaction data
        interaction_data = {
            "emotional_context": context.emotional_state,
            "goal_context": context.goal_context,
            "memory_context": context.memory_context,
            "interaction_type": input_analysis.get("interaction_type", "conversation"),
            "user_input_style": input_analysis.get("style", "neutral"),
            "dominance": is_dominance
        }
        
        # Update relationship based on interaction
        update_result = await self.original_manager.update_relationship_on_interaction(
            user_id, interaction_data
        )
        
        # Send relationship update to other modules
        if update_result.get("status") == "success":
            await self.send_context_update(
                update_type="relationship_state_change",
                data={
                    "user_id": user_id,
                    "relationship_context": update_result,
                    "trust_change": update_result.get("trust_impact", 0),
                    "intimacy_change": update_result.get("intimacy_impact", 0),
                    "dominance_change": update_result.get("dominance_impact", 0)
                }
            )
        
        return {
            "relationship_processed": True,
            "input_analysis": input_analysis,
            "update_result": update_result,
            "is_dominance_context": is_dominance
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze relationship state in context"""
        user_id = context.user_id
        
        # Get current relationship state
        relationship = await self.original_manager.get_relationship_state_internal(user_id)
        
        # Get messages from other modules
        messages = await self.get_cross_module_messages()
        
        # Analyze relationship trajectory
        trajectory = await self._analyze_relationship_trajectory(relationship, messages)
        
        # Get dominance recommendations if applicable
        dominance_recs = {}
        if relationship.get("dominance_balance", 0) != 0:
            dominance_recs = await self.original_manager.get_dominance_recommendations(user_id)
        
        # Identify relationship opportunities
        opportunities = await self._identify_relationship_opportunities(relationship, context)
        
        return {
            "current_relationship": relationship,
            "trajectory_analysis": trajectory,
            "dominance_recommendations": dominance_recs,
            "relationship_opportunities": opportunities,
            "cross_module_insights": len(messages)
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize relationship-aware response components"""
        user_id = context.user_id
        
        # Get relationship state
        relationship = await self.original_manager.get_relationship_state_internal(user_id)
        messages = await self.get_cross_module_messages()
        
        # Generate relationship-appropriate response modulation
        response_modulation = await self._generate_response_modulation(relationship, context)
        
        # Check for relationship milestones to announce
        milestone_check = await self._check_relationship_milestones(relationship)
        
        # Generate relationship summary for context
        summary = await self.original_manager.get_relationship_summary(user_id)
        
        # Send synthesis data
        await self.send_context_update(
            update_type="relationship_synthesis",
            data={
                "response_modulation": response_modulation,
                "relationship_summary": summary,
                "milestone_announcement": milestone_check,
                "trust_level": relationship.get("trust", 0.5),
                "intimacy_level": relationship.get("intimacy", 0.1)
            }
        )
        
        return {
            "relationship_influence": response_modulation,
            "synthesis_complete": True
        }
    
    # Helper methods
    async def _analyze_input_for_relationship(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for relationship implications"""
        input_lower = user_input.lower()
        
        implications = {
            "expresses_affection": any(kw in input_lower for kw in ["love", "care", "miss", "like you"]),
            "expresses_trust": any(kw in input_lower for kw in ["trust", "rely", "depend", "confide"]),
            "shares_vulnerability": any(kw in input_lower for kw in ["afraid", "worried", "scared", "vulnerable"]),
            "seeks_connection": any(kw in input_lower for kw in ["close", "together", "bond", "connect"]),
            "shows_dominance": any(kw in input_lower for kw in ["obey", "submit", "control", "command"]),
            "shows_submission": any(kw in input_lower for kw in ["yes mistress", "i obey", "as you wish"])
        }
        
        # Determine interaction type
        if implications["shows_submission"]:
            implications["interaction_type"] = "submission"
        elif implications["shows_dominance"]:
            implications["interaction_type"] = "dominance_challenge"
        elif implications["shares_vulnerability"]:
            implications["interaction_type"] = "vulnerable_sharing"
        elif implications["expresses_affection"]:
            implications["interaction_type"] = "affectionate"
        else:
            implications["interaction_type"] = "conversation"
        
        return implications
    
    async def _process_emotional_impact(self, user_id: str, emotional_data: Dict[str, Any]):
        """Process how emotional states impact relationship"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Map emotions to relationship impacts
        emotion_impacts = {
            "Joy": {"trust": 0.02, "positive_interaction_score": 0.1},
            "Trust": {"trust": 0.05, "intimacy": 0.02},
            "Love": {"intimacy": 0.05, "trust": 0.03},
            "Fear": {"trust": -0.02, "conflict": 0.02},
            "Anger": {"conflict": 0.05, "trust": -0.03}
        }
        
        impacts = emotion_impacts.get(emotion_name, {})
        if impacts and strength > 0.5:
            # Apply impacts to relationship
            interaction_data = {
                "emotional_context": emotional_data,
                "emotion_impacts": impacts
            }
            await self.original_manager.update_relationship_on_interaction(user_id, interaction_data)
    
    async def _generate_response_modulation(self, relationship: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Generate how response should be modulated based on relationship"""
        trust = relationship.get("trust", 0.5)
        intimacy = relationship.get("intimacy", 0.1)
        familiarity = relationship.get("familiarity", 0.1)
        dominance = relationship.get("dominance_balance", 0.0)
        
        modulation = {
            "formality_level": 1.0 - (familiarity * 0.7),  # Less formal with familiarity
            "emotional_openness": intimacy * 0.8 + trust * 0.2,  # More open with intimacy/trust
            "vulnerability_expression": intimacy * 0.6,
            "dominance_intensity": max(0, dominance * 0.8) if dominance > 0 else 0,
            "playfulness": familiarity * 0.5 + trust * 0.3
        }
        
        # Add specific suggestions
        suggestions = []
        if trust > 0.7:
            suggestions.append("express_deeper_thoughts")
        if intimacy > 0.6:
            suggestions.append("share_vulnerabilities")
        if familiarity > 0.6:
            suggestions.append("use_casual_language")
        if dominance > 0.5:
            suggestions.append("assert_dominance")
        
        modulation["suggestions"] = suggestions
        return modulation
    
    # Delegate other methods to original manager
    def __getattr__(self, name):
        return getattr(self.original_manager, name)
