# nyx/core/a2a/context_aware_hormone_system.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareHormoneSystem(ContextAwareModule):
    """
    Context-aware wrapper for HormoneSystem with full A2A integration
    """
    
    def __init__(self, original_hormone_system):
        super().__init__("hormone_system")
        self.original_system = original_hormone_system
        self.context_subscriptions = [
            "emotional_state_available", "emotional_processing_complete",
            "relationship_milestone", "intimacy_event", "dominance_event",
            "goal_completion_announcement", "stress_event",
            "circadian_update", "session_duration_update",
            "physical_interaction", "gratification_event"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize hormone processing for this context"""
        logger.debug(f"HormoneSystem received context for user: {context.user_id}")
        
        # Update environmental factors based on context
        await self._update_environmental_factors(context)
        
        # Check and update hormone cycles
        cycle_update = await self.original_system.update_hormone_cycles(
            self._create_hormone_context()
        )
        
        # Get current hormone levels
        hormone_levels = self.original_system.get_hormone_levels()
        
        # Analyze circadian alignment
        circadian_info = await self.original_system.get_hormone_circadian_info(
            self._create_hormone_context()
        )
        
        # Send hormone context to other modules
        await self.send_context_update(
            update_type="hormone_state_available",
            data={
                "hormone_levels": hormone_levels,
                "circadian_info": circadian_info,
                "environmental_factors": self.original_system.environmental_factors,
                "active_influences": self._get_active_influences(),
                "hormone_phase": self._summarize_hormone_phases(hormone_levels)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        ctx = self._create_hormone_context()
        
        if update.update_type == "emotional_state_available":
            # Emotions can influence hormones
            emotional_data = update.data
            await self._process_emotional_influence(emotional_data, ctx)
            
        elif update.update_type == "relationship_milestone":
            # Relationship milestones affect bonding hormones
            relationship_data = update.data
            await self._process_relationship_milestone(relationship_data, ctx)
            
        elif update.update_type == "intimacy_event":
            # Intimacy affects multiple hormones
            intimacy_data = update.data
            await self._process_intimacy_event(intimacy_data, ctx)
            
        elif update.update_type == "dominance_event":
            # Dominance events affect testosterone analog
            dominance_data = update.data
            await self._process_dominance_event(dominance_data, ctx)
            
        elif update.update_type == "gratification_event":
            # Trigger post-gratification response
            gratification_data = update.data
            await self._process_gratification_event(gratification_data, ctx)
            
        elif update.update_type == "stress_event":
            # Stress affects multiple hormones
            stress_data = update.data
            await self._process_stress_event(stress_data, ctx)
            
        elif update.update_type == "session_duration_update":
            # Long sessions affect hormones
            duration_data = update.data
            self.original_system.environmental_factors["session_duration"] = min(
                1.0, duration_data.get("duration_minutes", 0) / 120.0
            )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with hormone awareness"""
        ctx = self._create_hormone_context()
        
        # Update hormone cycles
        cycle_result = await self.original_system.update_hormone_cycles(ctx)
        
        # Get messages from other modules
        messages = await self.get_cross_module_messages()
        
        # Process contextual hormone adjustments
        adjustments = await self._process_contextual_adjustments(context, messages)
        
        # Apply hormone influences to neurochemicals
        influence_result = await self.original_system._update_hormone_influences(ctx)
        
        # Send hormone update
        await self.send_context_update(
            update_type="hormone_update_complete",
            data={
                "cycle_updates": cycle_result,
                "contextual_adjustments": adjustments,
                "neurochemical_influences": influence_result,
                "current_levels": self.original_system.get_hormone_levels()
            }
        )
        
        return {
            "hormones_processed": True,
            "cycle_result": cycle_result,
            "influences_applied": influence_result
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze hormone patterns and stability"""
        ctx = self._create_hormone_context()
        
        # Get hormone stability
        stability = await self.original_system.get_hormone_stability(ctx)
        
        # Analyze hormone influences
        influences = await self.original_system.analyze_hormone_influences(ctx)
        
        # Check phase relationships
        circadian = await self.original_system.get_hormone_circadian_info(ctx)
        
        # Analyze interaction with emotional state
        emotional_coherence = await self._analyze_hormone_emotion_coherence(context)
        
        return {
            "hormone_stability": stability,
            "active_influences": influences,
            "circadian_alignment": circadian,
            "emotional_coherence": emotional_coherence,
            "recommendations": self._generate_hormone_recommendations(stability, influences)
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize hormone influence for response"""
        messages = await self.get_cross_module_messages()
        
        # Determine hormone-based response modulation
        modulation = {
            "energy_level": self._calculate_energy_level(),
            "mood_baseline": self._calculate_mood_baseline(),
            "social_openness": self._calculate_social_openness(),
            "assertiveness": self._calculate_assertiveness(),
            "sensitivity": self._calculate_emotional_sensitivity()
        }
        
        # Check for hormone-driven needs
        hormone_needs = self._analyze_hormone_driven_needs()
        
        if hormone_needs:
            await self.send_context_update(
                update_type="hormone_needs_detected",
                data={
                    "needs": hormone_needs,
                    "urgency": self._calculate_hormone_urgency(hormone_needs)
                },
                priority=ContextPriority.NORMAL
            )
        
        return {
            "hormone_modulation": modulation,
            "synthesis_complete": True,
            "hormone_driven_needs": hormone_needs
        }
    
    # Helper methods
    def _create_hormone_context(self):
        """Create a context wrapper for hormone tools"""
        from nyx.core.emotions.context import EmotionalContext
        from agents import RunContextWrapper
        
        ctx = EmotionalContext()
        return RunContextWrapper(context=ctx)
    
    async def _update_environmental_factors(self, context: SharedContext):
        """Update environmental factors from context"""
        # Time of day is auto-updated
        
        # User familiarity from relationship context
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            familiarity = context.relationship_context.get("familiarity", 0.5)
            self.original_system.environmental_factors["user_familiarity"] = (trust + familiarity) / 2
        
        # Interaction quality from emotional valence
        if context.emotional_state:
            valence = context.emotional_state.get("valence", 0.0)
            # Convert -1 to 1 valence to 0 to 1 quality
            self.original_system.environmental_factors["interaction_quality"] = (valence + 1) / 2
    
    async def _process_emotional_influence(self, emotional_data: Dict, ctx):
        """Process how emotions influence hormones"""
        dominant_emotion = emotional_data.get("dominant_emotion", "Neutral")
        intensity = emotional_data.get("intensity", 0.5)
        
        # Map emotions to hormone effects
        emotion_hormone_map = {
            "Joy": {
                "endoryx": 0.15 * intensity,
                "testoryx": 0.05 * intensity
            },
            "Love": {
                "oxytonyx": 0.2 * intensity,
                "endoryx": 0.1 * intensity
            },
            "Trust": {
                "oxytonyx": 0.15 * intensity
            },
            "Fear": {
                "testoryx": -0.1 * intensity,
                "melatonyx": 0.1 * intensity
            },
            "Anger": {
                "testoryx": 0.15 * intensity,
                "endoryx": -0.1 * intensity
            },
            "Sadness": {
                "melatonyx": 0.15 * intensity,
                "testoryx": -0.15 * intensity
            }
        }
        
        if dominant_emotion in emotion_hormone_map:
            for hormone, change in emotion_hormone_map[dominant_emotion].items():
                await self.original_system.update_hormone(ctx, hormone, change, f"emotion_{dominant_emotion}")
    
    async def _process_gratification_event(self, gratification_data: Dict, ctx):
        """Process gratification events"""
        intensity = gratification_data.get("intensity", 1.0)
        gratification_type = gratification_data.get("type", "general")
        
        await self.original_system.trigger_post_gratification_response(
            ctx, intensity, gratification_type
        )
        
        # Send hormone influence update
        await self.send_context_update(
            update_type="hormone_influence_update",
            data={
                "post_gratification": True,
                "serenity_active": True,
                "refractory_period": True
            }
        )
    
    def _calculate_energy_level(self) -> float:
        """Calculate energy level from hormones"""
        testoryx = self.original_system.hormones["testoryx"]["value"]
        endoryx = self.original_system.hormones["endoryx"]["value"]
        melatonyx = self.original_system.hormones["melatonyx"]["value"]
        
        # High testosterone and endorphins increase energy, melatonin decreases
        return (testoryx * 0.4 + endoryx * 0.4 - melatonyx * 0.2)
    
    def _calculate_social_openness(self) -> float:
        """Calculate social openness from hormones"""
        oxytonyx = self.original_system.hormones["oxytonyx"]["value"]
        estradyx = self.original_system.hormones["estradyx"]["value"]
        
        return (oxytonyx * 0.6 + estradyx * 0.4)
    
    # Delegate missing methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
