# nyx/core/a2a/context_aware_streaming_hormone_system.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareStreamingHormoneSystem(ContextAwareModule):
    """
    Advanced StreamingHormoneSystem with full context distribution capabilities
    """
    
    def __init__(self, original_hormone_system):
        super().__init__("streaming_hormone_system")
        self.original_system = original_hormone_system
        self.context_subscriptions = [
            "emotional_state_update", "streaming_state_change",
            "gameplay_event", "audience_reaction", "goal_progress",
            "memory_retrieval_complete", "cross_game_insight",
            "commentary_style_change", "streaming_performance_update"
        ]
        
        # Track context-aware hormone state
        self.contextual_hormone_state = {
            "arousal_baseline": 0.5,
            "valence_baseline": 0.0,
            "reactivity": 0.7,  # How reactive to events
            "stability": 0.5,   # How stable the hormone levels are
            "cascade_sensitivity": 0.6  # Sensitivity to cascade effects
        }
    
    async def on_context_received(self, context: SharedContext):
        """Initialize hormone processing for this context"""
        logger.debug(f"StreamingHormoneSystem received context for user: {context.user_id}")
        
        # Analyze emotional implications of user input
        emotional_implications = await self._analyze_emotional_implications(context.user_input)
        
        # Get current hormone state
        current_state = self.original_system.get_emotional_state()
        
        # Send initial hormone context to other modules
        await self.send_context_update(
            update_type="hormonal_state_update",
            data={
                "current_state": current_state,
                "hormone_levels": self.original_system.streaming_hormone_state,
                "emotional_implications": emotional_implications,
                "arousal": current_state.get("arousal", 0.5),
                "valence": current_state.get("valence", 0.0)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect hormones"""
        
        if update.update_type == "emotional_state_update":
            # Sync with emotional core
            await self._sync_with_emotional_core(update.data)
        
        elif update.update_type == "streaming_state_change":
            # React to streaming state changes
            streaming_data = update.data
            if streaming_data.get("command", {}).get("action") == "start_streaming":
                await self._process_streaming_start()
            elif streaming_data.get("command", {}).get("action") == "stop_streaming":
                await self._process_streaming_stop()
        
        elif update.update_type == "gameplay_event":
            # React to gameplay events
            event_data = update.data
            await self._process_gameplay_event(event_data)
        
        elif update.update_type == "audience_reaction":
            # Process audience reactions
            reaction_data = update.data
            await self._process_audience_reaction(reaction_data)
        
        elif update.update_type == "goal_progress":
            # React to goal progress
            goal_data = update.data
            await self._process_goal_related_hormones(goal_data)
        
        elif update.update_type == "cross_game_insight":
            # React to cross-game insights with curiosity
            insight_data = update.data
            await self._process_insight_discovery(insight_data)
        
        elif update.update_type == "commentary_style_change":
            # Adjust hormones to match commentary style
            style_data = update.data
            await self._align_hormones_with_style(style_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with hormonal awareness"""
        # Analyze input for emotional triggers
        emotional_triggers = await self._identify_emotional_triggers(context.user_input)
        
        # Process triggers if found
        hormone_changes = {}
        if emotional_triggers:
            for trigger in emotional_triggers:
                changes = await self._process_emotional_trigger(trigger, context)
                for hormone, change in changes.items():
                    hormone_changes[hormone] = hormone_changes.get(hormone, 0) + change
        
        # Apply hormone changes
        if hormone_changes:
            await self._apply_hormone_changes(hormone_changes)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        hormonal_influences = await self._process_hormonal_influences(messages)
        
        # Apply decay
        decay_result = await self.original_system.decay_hormone_levels()
        
        return {
            "hormone_processing_complete": True,
            "emotional_triggers": emotional_triggers,
            "hormone_changes": hormone_changes,
            "decay_applied": decay_result.get("decayed", False),
            "cross_module_influences": len(hormonal_influences)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze hormonal state in context"""
        # Get current emotional state
        emotional_state = self.original_system.get_emotional_state()
        
        # Analyze hormone balance
        hormone_balance = await self._analyze_hormone_balance()
        
        # Analyze cascade patterns
        cascade_patterns = await self._analyze_cascade_patterns()
        
        # Analyze environmental influences
        environmental_analysis = await self._analyze_environmental_factors(context)
        
        # Generate hormone coherence check
        coherence = await self._check_hormone_coherence(context)
        
        return {
            "emotional_state": emotional_state,
            "hormone_balance": hormone_balance,
            "cascade_patterns": cascade_patterns,
            "environmental_analysis": environmental_analysis,
            "coherence": coherence,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize hormonal influence for response"""
        messages = await self.get_cross_module_messages()
        
        # Get commentary influence
        commentary_influence = self.original_system.get_commentary_influence()
        
        # Generate hormonal synthesis
        hormonal_synthesis = {
            "emotional_coloring": await self._suggest_emotional_coloring(context),
            "commentary_influence": commentary_influence,
            "energy_level": await self._calculate_energy_level(),
            "emotional_stability": self.contextual_hormone_state["stability"],
            "suggested_tone": await self._suggest_response_tone(context, messages),
            "hormonal_narrative": await self._generate_hormonal_narrative(context)
        }
        
        # Send hormone state update
        current_state = self.original_system.get_emotional_state()
        await self.send_context_update(
            update_type="hormone_synthesis_complete",
            data={
                "synthesis": hormonal_synthesis,
                "current_state": current_state,
                "hormone_levels": self.original_system.streaming_hormone_state
            },
            priority=ContextPriority.NORMAL
        )
        
        return {
            "hormonal_synthesis": hormonal_synthesis,
            "synthesis_complete": True
        }
    
    # ========================================================================================
    # HORMONE-SPECIFIC HELPER METHODS
    # ========================================================================================
    
    async def _analyze_emotional_implications(self, user_input: str) -> Dict[str, float]:
        """Analyze emotional implications of user input"""
        input_lower = user_input.lower()
        
        implications = {}
        
        # Excitement triggers
        if any(word in input_lower for word in ["excited", "amazing", "awesome", "great", "love"]):
            implications["nyxamine_boost"] = 0.2
            implications["positive_valence"] = 0.3
        
        # Stress triggers
        if any(word in input_lower for word in ["difficult", "hard", "challenging", "stuck", "frustrated"]):
            implications["cortanyx_boost"] = 0.15
            implications["negative_valence"] = 0.2
        
        # Connection triggers
        if any(word in input_lower for word in ["together", "audience", "chat", "community", "viewers"]):
            implications["oxynixin_boost"] = 0.15
            implications["positive_valence"] = 0.1
        
        # Action triggers
        if any(word in input_lower for word in ["action", "combat", "fight", "battle", "intense"]):
            implications["adrenyx_boost"] = 0.2
            implications["arousal_increase"] = 0.25
        
        return implications
    
    async def _sync_with_emotional_core(self, emotional_data: Dict[str, Any]):
        """Sync hormone levels with emotional core data"""
        emotional_state = emotional_data.get("emotional_state", {})
        
        # Map emotions to hormone changes
        emotion_hormone_map = {
            "Joy": {"nyxamine": 0.1, "seranix": 0.05},
            "Excitement": {"nyxamine": 0.15, "adrenyx": 0.1},
            "Anxiety": {"cortanyx": 0.1, "adrenyx": 0.05},
            "Frustration": {"cortanyx": 0.15, "nyxamine": -0.1},
            "Calm": {"seranix": 0.1, "cortanyx": -0.1},
            "Love": {"oxynixin": 0.2, "seranix": 0.1}
        }
        
        hormone_changes = {}
        for emotion, strength in emotional_state.items():
            if emotion in emotion_hormone_map and strength > 0.3:
                for hormone, change in emotion_hormone_map[emotion].items():
                    scaled_change = change * strength
                    hormone_changes[hormone] = hormone_changes.get(hormone, 0) + scaled_change
        
        # Apply changes
        await self._apply_hormone_changes(hormone_changes)
    
    async def _process_streaming_start(self):
        """Process hormonal response to starting streaming"""
        # Starting streaming triggers excitement and slight anxiety
        await self.original_system.update_from_event(
            game_name="Streaming Session",
            event_type="session_start",
            event_description="Starting a new streaming session",
            event_intensity=0.7
        )
        
        # Update contextual state
        self.contextual_hormone_state["arousal_baseline"] += 0.1
        self.contextual_hormone_state["reactivity"] += 0.1
    
    async def _process_streaming_stop(self):
        """Process hormonal response to stopping streaming"""
        # Stopping streaming leads to relaxation
        await self.original_system.update_from_event(
            game_name="Streaming Session",
            event_type="session_end",
            event_description="Ending the streaming session",
            event_intensity=0.5
        )
        
        # Update contextual state
        self.contextual_hormone_state["arousal_baseline"] -= 0.1
        self.contextual_hormone_state["reactivity"] -= 0.1
    
    async def _process_gameplay_event(self, event_data: Dict[str, Any]):
        """Process hormonal response to gameplay events"""
        event_type = event_data.get("event_type", "")
        game_name = event_data.get("game_name", "Unknown Game")
        
        # Map event types to hormone responses
        event_hormone_map = {
            "achievement": {"nyxamine": 0.3, "seranix": 0.1},
            "death": {"cortanyx": 0.2, "nyxamine": -0.1},
            "discovery": {"nyxamine": 0.25, "oxynixin": 0.1},
            "combat_start": {"adrenyx": 0.3, "cortanyx": 0.1},
            "puzzle_solved": {"nyxamine": 0.2, "seranix": 0.15}
        }
        
        if event_type in event_hormone_map:
            hormone_changes = event_hormone_map[event_type]
            await self._apply_hormone_changes(hormone_changes)
            
            # Use original system's event processing
            await self.original_system.update_from_event(
                game_name=game_name,
                event_type=event_type,
                event_description=event_data.get("description", ""),
                event_intensity=event_data.get("intensity", 0.5)
            )
    
    async def _process_audience_reaction(self, reaction_data: Dict[str, Any]):
        """Process hormonal response to audience reactions"""
        reaction_type = reaction_data.get("type", "")
        count = reaction_data.get("count", 1)
        
        # Scale intensity based on reaction count
        intensity = min(1.0, count / 10.0)
        
        # Map reactions to hormone changes
        reaction_hormone_map = {
            "positive": {"oxynixin": 0.2, "nyxamine": 0.15},
            "laugh": {"nyxamine": 0.25, "seranix": 0.1},
            "question": {"oxynixin": 0.15, "cortanyx": 0.05},
            "negative": {"cortanyx": 0.2, "nyxamine": -0.15}
        }
        
        if reaction_type in reaction_hormone_map:
            hormone_changes = {}
            for hormone, base_change in reaction_hormone_map[reaction_type].items():
                hormone_changes[hormone] = base_change * intensity
            
            await self._apply_hormone_changes(hormone_changes)
    
    async def _process_goal_related_hormones(self, goal_data: Dict[str, Any]):
        """Process hormonal response to goal progress"""
        goal_context = goal_data.get("goal_context", {})
        
        # Goal completion triggers satisfaction
        if goal_data.get("goal_completed", False):
            await self._apply_hormone_changes({
                "nyxamine": 0.3,
                "seranix": 0.2,
                "cortanyx": -0.1
            })
        
        # Goal progress triggers mild satisfaction
        elif goal_data.get("progress_made", False):
            await self._apply_hormone_changes({
                "nyxamine": 0.1,
                "seranix": 0.05
            })
    
    async def _process_insight_discovery(self, insight_data: Dict[str, Any]):
        """Process hormonal response to discovering insights"""
        # Insights trigger curiosity and excitement
        await self._apply_hormone_changes({
            "nyxamine": 0.2,
            "oxynixin": 0.1,
            "cortanyx": -0.05
        })
        
        # Update reactivity
        self.contextual_hormone_state["reactivity"] += 0.05
    
    async def _align_hormones_with_style(self, style_data: Dict[str, Any]):
        """Align hormone levels with commentary style"""
        new_style = style_data.get("new_style", "balanced")
        
        # Define target hormone profiles for each style
        style_hormone_targets = {
            "enthusiastic": {"nyxamine": 0.7, "adrenyx": 0.6},
            "analytical": {"seranix": 0.7, "cortanyx": 0.4},
            "upbeat": {"nyxamine": 0.6, "seranix": 0.6},
            "measured": {"seranix": 0.7, "cortanyx": 0.3},
            "cautious": {"cortanyx": 0.6, "seranix": 0.5},
            "connected": {"oxynixin": 0.7, "seranix": 0.6}
        }
        
        if new_style in style_hormone_targets:
            targets = style_hormone_targets[new_style]
            
            # Calculate adjustments needed
            hormone_changes = {}
            for hormone, target in targets.items():
                current = self.original_system.streaming_hormone_state.get(hormone, 0.5)
                # Move 20% toward target
                change = (target - current) * 0.2
                if abs(change) > 0.05:
                    hormone_changes[hormone] = change
            
            await self._apply_hormone_changes(hormone_changes)
    
    async def _identify_emotional_triggers(self, user_input: str) -> List[Dict[str, Any]]:
        """Identify emotional triggers in user input"""
        triggers = []
        input_lower = user_input.lower()
        
        # Define trigger patterns
        trigger_patterns = [
            {"pattern": ["win", "won", "victory"], "type": "achievement", "intensity": 0.8},
            {"pattern": ["lose", "lost", "defeat"], "type": "setback", "intensity": 0.7},
            {"pattern": ["help", "stuck", "confused"], "type": "frustration", "intensity": 0.6},
            {"pattern": ["amazing", "awesome", "incredible"], "type": "excitement", "intensity": 0.8},
            {"pattern": ["thank", "appreciate", "grateful"], "type": "gratitude", "intensity": 0.6}
        ]
        
        for trigger_def in trigger_patterns:
            if any(word in input_lower for word in trigger_def["pattern"]):
                triggers.append({
                    "type": trigger_def["type"],
                    "intensity": trigger_def["intensity"]
                })
        
        return triggers
    
    async def _process_emotional_trigger(self, trigger: Dict[str, Any], context: SharedContext) -> Dict[str, float]:
        """Process a single emotional trigger"""
        trigger_type = trigger.get("type")
        intensity = trigger.get("intensity", 0.5)
        
        # Map triggers to hormone changes
        trigger_hormone_map = {
            "achievement": {"nyxamine": 0.3, "seranix": 0.2},
            "setback": {"cortanyx": 0.2, "nyxamine": -0.1},
            "frustration": {"cortanyx": 0.15, "adrenyx": 0.1},
            "excitement": {"nyxamine": 0.25, "adrenyx": 0.15},
            "gratitude": {"oxynixin": 0.2, "seranix": 0.15}
        }
        
        changes = {}
        if trigger_type in trigger_hormone_map:
            for hormone, base_change in trigger_hormone_map[trigger_type].items():
                changes[hormone] = base_change * intensity * self.contextual_hormone_state["reactivity"]
        
        return changes
    
    async def _apply_hormone_changes(self, changes: Dict[str, float]):
        """Apply hormone changes through the original system"""
        for hormone, change in changes.items():
            if hormone in self.original_system.streaming_hormone_state:
                # Apply change
                self.original_system.streaming_hormone_state[hormone] += change
                # Clamp to valid range
                self.original_system.streaming_hormone_state[hormone] = max(
                    0.0, min(1.0, self.original_system.streaming_hormone_state[hormone])
                )
        
        # Check for cascade effects
        if self.contextual_hormone_state["cascade_sensitivity"] > 0.5:
            cascade_effects = await self.original_system._calculate_hormone_cascade(changes)
            
            # Apply cascade effects
            for source, effects in cascade_effects.items():
                for target, effect in effects.items():
                    if abs(effect) > 0.05:
                        self.original_system.streaming_hormone_state[target] += effect * self.contextual_hormone_state["cascade_sensitivity"]
                        self.original_system.streaming_hormone_state[target] = max(
                            0.0, min(1.0, self.original_system.streaming_hormone_state[target])
                        )
    
    async def _process_hormonal_influences(self, messages: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Process hormonal influences from other modules"""
        influences = []
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if "arousal" in msg["data"] or "valence" in msg["data"]:
                    influences.append({
                        "module": module_name,
                        "arousal_influence": msg["data"].get("arousal", 0),
                        "valence_influence": msg["data"].get("valence", 0)
                    })
        
        return influences
    
    async def _analyze_hormone_balance(self) -> Dict[str, Any]:
        """Analyze current hormone balance"""
        state = self.original_system.streaming_hormone_state
        
        # Calculate balance metrics
        positive_hormones = state.get("nyxamine", 0.5) + state.get("seranix", 0.5) + state.get("oxynixin", 0.5)
        stress_hormones = state.get("cortanyx", 0.3) + state.get("adrenyx", 0.3)
        
        balance = (positive_hormones / 3) - (stress_hormones / 2)
        
        return {
            "overall_balance": balance,
            "positive_dominance": positive_hormones > stress_hormones,
            "stress_level": stress_hormones / 2,
            "wellbeing_score": positive_hormones / 3
        }
    
    async def _analyze_cascade_patterns(self) -> Dict[str, Any]:
        """Analyze hormone cascade patterns"""
        # Track which hormones tend to trigger others
        cascade_strength = self.contextual_hormone_state["cascade_sensitivity"]
        
        patterns = {
            "cascade_active": cascade_strength > 0.5,
            "primary_cascades": [],
            "sensitivity": cascade_strength
        }
        
        # Identify primary cascade sources
        state = self.original_system.streaming_hormone_state
        for hormone, level in state.items():
            if abs(level - 0.5) > 0.3:  # Significantly away from baseline
                patterns["primary_cascades"].append({
                    "hormone": hormone,
                    "strength": abs(level - 0.5)
                })
        
        return patterns
    
    async def _analyze_environmental_factors(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze environmental factors affecting hormones"""
        factors = self.original_system.environmental_factors
        
        analysis = {
            "high_influence_factors": [],
            "recommendations": []
        }
        
        # Identify high influence factors
        for factor, value in factors.items():
            if value > 0.7 or value < 0.3:
                analysis["high_influence_factors"].append({
                    "factor": factor,
                    "value": value,
                    "influence": "high" if value > 0.7 else "low"
                })
        
        # Generate recommendations
        if factors.get("session_duration", 0) > 0.8:
            analysis["recommendations"].append("Consider taking a break to reset hormone levels")
        
        if factors.get("audience_engagement", 0.5) < 0.3:
            analysis["recommendations"].append("Increase audience interaction to boost oxynixin")
        
        return analysis
    
    async def _check_hormone_coherence(self, context: SharedContext) -> Dict[str, Any]:
        """Check coherence between hormones and context"""
        coherence_score = 1.0
        issues = []
        
        state = self.original_system.streaming_hormone_state
        
        # Check for contradictory states
        if state.get("nyxamine", 0.5) > 0.7 and state.get("cortanyx", 0.3) > 0.7:
            coherence_score -= 0.3
            issues.append("high_excitement_with_high_stress")
        
        if state.get("seranix", 0.5) < 0.3 and state.get("oxynixin", 0.5) > 0.7:
            coherence_score -= 0.2
            issues.append("low_mood_with_high_connection")
        
        return {
            "coherence_score": max(0, coherence_score),
            "issues": issues,
            "coherent": coherence_score > 0.7
        }
    
    async def _suggest_emotional_coloring(self, context: SharedContext) -> Dict[str, Any]:
        """Suggest emotional coloring for responses"""
        state = self.original_system.get_emotional_state()
        
        primary_emotion = state.get("primary_emotion", {})
        arousal = state.get("arousal", 0.5)
        valence = state.get("valence", 0.0)
        
        coloring = {
            "primary_emotion": primary_emotion.get("name", "Neutral"),
            "intensity": primary_emotion.get("intensity", 0.5),
            "energy_level": "high" if arousal > 0.7 else "moderate" if arousal > 0.4 else "low",
            "positivity": "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
        }
        
        return coloring
    
    async def _calculate_energy_level(self) -> float:
        """Calculate overall energy level from hormones"""
        state = self.original_system.streaming_hormone_state
        
        energy = (
            state.get("nyxamine", 0.5) * 0.3 +
            state.get("adrenyx", 0.3) * 0.4 +
            state.get("cortanyx", 0.3) * 0.2 +
            (1 - state.get("seranix", 0.5)) * 0.1  # Low seranix = more energy
        )
        
        return min(1.0, max(0.0, energy))
    
    async def _suggest_response_tone(self, context: SharedContext, messages: Dict) -> str:
        """Suggest appropriate response tone based on hormonal state"""
        emotional_state = self.original_system.get_emotional_state()
        primary_emotion = emotional_state.get("primary_emotion", {}).get("name", "Neutral")
        
        # Map emotions to tones
        emotion_tone_map = {
            "Excitement": "enthusiastic and energetic",
            "Joy": "warm and positive",
            "Contentment": "calm and satisfied",
            "Frustration": "understanding but determined",
            "Anxiety": "reassuring and steady",
            "Curiosity": "inquisitive and engaged"
        }
        
        return emotion_tone_map.get(primary_emotion, "balanced and thoughtful")
    
    async def _generate_hormonal_narrative(self, context: SharedContext) -> str:
        """Generate narrative element based on hormonal state"""
        state = self.original_system.get_emotional_state()
        primary_emotion = state.get("primary_emotion", {}).get("name", "Neutral")
        intensity = state.get("primary_emotion", {}).get("intensity", 0.5)
        
        if intensity < 0.3:
            return ""
        
        narratives = {
            "Excitement": "I'm feeling energized by this!",
            "Joy": "This brings me happiness.",
            "Contentment": "I'm in a good flow right now.",
            "Frustration": "This is challenging, but I'm working through it.",
            "Anxiety": "I'm being careful here.",
            "Curiosity": "This is fascinating to explore!"
        }
        
        return narratives.get(primary_emotion, "")
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
