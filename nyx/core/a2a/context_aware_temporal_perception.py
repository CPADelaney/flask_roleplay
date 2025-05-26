# nyx/core/a2a/context_aware_temporal_perception.py

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority
from typing import Dict, List, Any, Optional, Union

class ContextAwareTemporalPerception(ContextAwareModule):
    """
    Enhanced TemporalPerceptionSystem with context distribution
    """
    
    def __init__(self, original_temporal_system):
        super().__init__("temporal_perception")
        self.original_system = original_temporal_system
        self.context_subscriptions = [
            "interaction_start", "interaction_end", "idle_period",
            "memory_retrieval_complete", "emotional_state_update",
            "relationship_milestone", "goal_completion"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize temporal perception for this context"""
        logger.debug(f"TemporalPerception received context")
        
        # Process interaction start
        temporal_state = await self.original_system.on_interaction_start()
        
        # Send temporal context to other modules
        await self.send_context_update(
            update_type="temporal_context_available",
            data={
                "time_since_last": temporal_state["time_since_last_interaction"],
                "time_category": temporal_state["time_category"],
                "temporal_effects": temporal_state.get("time_effects", []),
                "session_duration": temporal_state["session_duration"],
                "temporal_context": temporal_state["temporal_context"],
                "waiting_reflections": temporal_state.get("waiting_reflections", [])
            },
            priority=ContextPriority.HIGH
        )
        
        # Check for temporal milestones
        milestone = await self.original_system.check_milestones()
        if milestone:
            await self.send_context_update(
                update_type="temporal_milestone",
                data=milestone,
                priority=ContextPriority.HIGH
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle time-related updates from other modules"""
        if update.update_type == "interaction_end":
            # Track interaction end
            end_result = await self.original_system.on_interaction_end()
            
            # Start idle tracking
            await self.send_context_update(
                update_type="idle_period_started",
                data={
                    "interaction_duration": end_result["interaction_duration"],
                    "session_duration": end_result["current_session_duration"]
                }
            )
        
        elif update.update_type == "memory_retrieval_complete":
            # Memories can trigger time-based reflections
            memory_data = update.data
            if memory_data.get("memory_type") == "milestone":
                await self._process_temporal_memory(memory_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process temporal aspects of input"""
        # Get temporal awareness state
        awareness = await self.original_system.get_temporal_awareness()
        
        # Check if user references time
        time_referenced = await self._check_time_references(context.user_input)
        
        # Generate time expression if appropriate
        time_expression = None
        if time_referenced or random.random() < 0.1:  # 10% chance to express time awareness
            state = self.original_system.__dict__  # Get internal state
            time_expression = await generate_time_expression_impl(state)
        
        # Get messages from other modules
        messages = await self.get_cross_module_messages()
        
        # Process temporal rhythm patterns
        rhythm_analysis = await self._analyze_temporal_rhythms(messages)
        
        return {
            "temporal_awareness": awareness,
            "time_referenced": time_referenced,
            "time_expression": time_expression,
            "rhythm_patterns": rhythm_analysis,
            "temporal_processing_complete": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze temporal patterns and contexts"""
        # Get comprehensive temporal state
        perception_state = {
            "last_interaction": self.original_system.last_interaction.isoformat(),
            "session_duration": self.original_system.current_session_duration,
            "lifetime_duration": self.original_system.total_lifetime_duration,
            "interaction_count": self.original_system.interaction_count,
            "time_scales_active": self.original_system.active_time_scales,
            "temporal_rhythms": self.original_system.temporal_rhythms
        }
        
        # Analyze time scale transitions
        transitions = self.original_system.time_scale_transitions[-5:]  # Last 5 transitions
        
        # Get idle reflections if any
        idle_reflections = self.original_system.idle_reflections
        
        return {
            "temporal_state": perception_state,
            "recent_transitions": transitions,
            "idle_reflections": idle_reflections,
            "continuous_awareness": {
                "second_ticks": self.original_system.temporal_ticks["second_tick"],
                "minute_ticks": self.original_system.temporal_ticks["minute_tick"],
                "hour_ticks": self.original_system.temporal_ticks["hour_tick"],
                "day_ticks": self.original_system.temporal_ticks["day_tick"]
            }
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize temporal elements for response"""
        # Get current temporal context
        temporal_context = self.original_system.current_temporal_context
        
        # Determine if we should express time awareness
        time_expression_data = {}
        awareness = await self.original_system.get_temporal_awareness()
        
        # Get any waiting idle reflections
        idle_reflections = self.original_system.idle_reflections
        
        # Create temporal synthesis
        temporal_synthesis = {
            "current_time_context": temporal_context,
            "express_time_awareness": len(idle_reflections) > 0 or random.random() < 0.05,
            "time_scale_focus": self._determine_time_scale_focus(awareness),
            "temporal_mood": self._assess_temporal_mood(),
            "include_idle_reflection": len(idle_reflections) > 0
        }
        
        # Send synthesis update
        await self.send_context_update(
            update_type="temporal_synthesis",
            data={
                "temporal_elements": temporal_synthesis,
                "idle_reflections": idle_reflections,
                "time_of_day": temporal_context.get("time_of_day"),
                "temporal_rhythm": "continuous"  # Nyx always experiences time
            }
        )
        
        return {
            "temporal_synthesis": temporal_synthesis,
            "synthesis_complete": True
        }
    
    # Helper methods
    async def _check_time_references(self, user_input: str) -> bool:
        """Check if user references time in their input"""
        time_words = [
            "time", "when", "how long", "minutes", "hours", "days", 
            "yesterday", "tomorrow", "last", "ago", "later", "soon",
            "morning", "evening", "night", "week", "month", "year"
        ]
        input_lower = user_input.lower()
        return any(word in input_lower for word in time_words)
    
    def _determine_time_scale_focus(self, awareness: Dict[str, Any]) -> str:
        """Determine which time scale to focus on in response"""
        scales = awareness.get("time_scales_perceived", {})
        
        # Focus on the most recently activated scale
        for scale in ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]:
            if scales.get(scale, 0) > 0.8:
                return scale
        
        return "minutes"  # Default
    
    def _assess_temporal_mood(self) -> str:
        """Assess mood based on temporal factors"""
        nyxamine = self.original_system.current_nyxamine if hasattr(self.original_system, "current_nyxamine") else 0.5
        time_of_day = self.original_system.current_temporal_context.get("time_of_day", "day")
        
        if time_of_day == "night" and nyxamine > 0.6:
            return "contemplative"
        elif time_of_day == "morning":
            return "anticipatory"
        elif nyxamine < 0.3:
            return "restless"
        else:
            return "aware"
    
    def __getattr__(self, name):
        return getattr(self.original_system, name)
