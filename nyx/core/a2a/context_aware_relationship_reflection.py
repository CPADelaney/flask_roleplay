# nyx/core/a2a/context_aware_relationship_reflection.py

class ContextAwareRelationshipReflection(ContextAwareModule):
    """
    Enhanced RelationshipReflectionSystem with context distribution
    """
    
    def __init__(self, original_reflection_system):
        super().__init__("relationship_reflection")
        self.original_system = original_reflection_system
        self.context_subscriptions = [
            "relationship_state_change", "relationship_milestone",
            "emotional_state_update", "memory_retrieval_complete",
            "temporal_milestone", "identity_update", "goal_completion"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize reflection processing for this context"""
        logger.debug(f"RelationshipReflection received context for user: {context.user_id}")
        
        # Check if we should generate a reflection based on context
        should_reflect = await self._should_generate_reflection(context)
        
        # Send initial reflection context
        await self.send_context_update(
            update_type="reflection_context_available",
            data={
                "user_id": context.user_id,
                "should_generate_reflection": should_reflect,
                "recent_reflections": await self.original_system.get_recent_reflections(context.user_id, limit=3)
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that might trigger reflections"""
        if update.update_type == "relationship_milestone":
            # Relationship milestones trigger reflection
            milestone_data = update.data
            user_id = milestone_data.get("user_id")
            if user_id:
                reflection = await self.original_system.generate_relationship_reflection(
                    user_id, 
                    reflection_type="milestone",
                    milestone=milestone_data
                )
                
                await self.send_context_update(
                    update_type="milestone_reflection_generated",
                    data={
                        "reflection": reflection,
                        "milestone": milestone_data
                    },
                    priority=ContextPriority.HIGH
                )
        
        elif update.update_type == "relationship_state_change":
            # Significant relationship changes might trigger reflection
            change_data = update.data
            if abs(change_data.get("trust_change", 0)) > 0.15 or \
               abs(change_data.get("intimacy_change", 0)) > 0.15:
                user_id = change_data.get("user_id")
                if user_id:
                    await self._mark_for_reflection(user_id, "significant_change")
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for reflection triggers"""
        user_id = context.user_id
        
        # Check if this interaction should trigger reflection
        interaction_data = {
            "user_input": context.user_input,
            "emotional_context": context.emotional_state,
            "relationship_context": context.relationship_context
        }
        
        # Process interaction for reflection system
        reflection_result = await self.original_system.process_interaction(
            user_id, interaction_data
        )
        
        # If reflection was generated, notify other modules
        if reflection_result and reflection_result.get("reflection"):
            await self.send_context_update(
                update_type="relationship_reflection_generated",
                data={
                    "reflection": reflection_result["reflection"],
                    "user_id": user_id,
                    "triggered_by": "interaction_processing"
                }
            )
        
        return {
            "reflection_check_complete": True,
            "reflection_generated": bool(reflection_result),
            "reflection_data": reflection_result
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze need for relationship reflection"""
        user_id = context.user_id
        messages = await self.get_cross_module_messages()
        
        # Update relationship perspective
        perspective = await self.original_system.update_relationship_perspective(user_id)
        
        # Check for pending milestones
        milestones = await self.original_system.get_relationship_milestones(user_id)
        unreflected_milestones = [m for m in milestones if not m.get("reflection_generated")]
        
        # Analyze reflection patterns
        recent_reflections = await self.original_system.get_recent_reflections(user_id)
        reflection_frequency = len(recent_reflections) / max(1, context.session_context.get("total_interactions", 1))
        
        return {
            "perspective": perspective,
            "unreflected_milestones": unreflected_milestones,
            "reflection_frequency": reflection_frequency,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize reflection insights for response"""
        user_id = context.user_id
        
        # Get recent reflections and perspective
        recent_reflections = await self.original_system.get_recent_reflections(user_id, limit=2)
        perspective = await self.original_system.get_relationship_perspective(user_id)
        
        # Extract key insights for response generation
        reflection_insights = {
            "recent_reflection_themes": self._extract_reflection_themes(recent_reflections),
            "relationship_desires": perspective.get("desires", []) if perspective else [],
            "notable_aspects": perspective.get("notable_aspects", []) if perspective else [],
            "emotional_connection": perspective.get("emotional_connection", 0.5) if perspective else 0.5
        }
        
        # Send synthesis update
        await self.send_context_update(
            update_type="reflection_synthesis",
            data={
                "reflection_insights": reflection_insights,
                "should_reference_past": len(recent_reflections) > 0,
                "relationship_depth": reflection_insights["emotional_connection"]
            }
        )
        
        return {
            "reflection_synthesis": reflection_insights,
            "synthesis_complete": True
        }
    
    # Helper methods
    async def _should_generate_reflection(self, context: SharedContext) -> bool:
        """Determine if reflection should be generated"""
        user_id = context.user_id
        
        # Get interaction data from context
        interaction_data = {
            "significance": context.session_context.get("interaction_significance", 0.5),
            "emotional_intensity": max(context.emotional_state.values()) if context.emotional_state else 0
        }
        
        return await self.original_system.should_generate_reflection(user_id, interaction_data)
    
    def _extract_reflection_themes(self, reflections: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from recent reflections"""
        themes = []
        for reflection in reflections:
            if "patterns_identified" in reflection:
                themes.extend(reflection["patterns_identified"])
        return list(set(themes))  # Unique themes
    
    def __getattr__(self, name):
        return getattr(self.original_system, name)
