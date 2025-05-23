# nyx/core/a2a/context_aware_mood_manager.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareMoodManager(ContextAwareModule):
    """
    Advanced MoodManager with full context distribution capabilities for mood state management
    """
    
    def __init__(self, original_mood_manager):
        super().__init__("mood_manager")
        self.original_manager = original_mood_manager
        self.context_subscriptions = [
            "emotional_state_update", "needs_state_change", "goal_progress",
            "reward_signal", "hormone_level_update", "relationship_milestone",
            "dominance_gratification", "social_interaction", "memory_retrieval_complete",
            "stress_indicator", "pleasure_signal", "achievement_unlocked"
        ]
        
        # Track mood influences over time
        self.mood_influence_history = []
        self.max_history_size = 100
    
    async def on_context_received(self, context: SharedContext):
        """Initialize mood processing for this context"""
        logger.debug(f"MoodManager received context for user: {context.user_id}")
        
        # Get current mood state
        current_mood = await self._get_current_mood_state()
        
        # Analyze context for mood influences
        mood_influences = await self._analyze_context_for_mood(context)
        
        # Calculate mood trajectory
        mood_trajectory = await self._calculate_mood_trajectory(current_mood, mood_influences)
        
        # Send initial mood context
        await self.send_context_update(
            update_type="mood_context_initialized",
            data={
                "current_mood": current_mood,
                "mood_influences": mood_influences,
                "mood_trajectory": mood_trajectory,
                "mood_stability": await self._calculate_mood_stability(),
                "dominant_mood": current_mood.get("dominant_mood", "Neutral")
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules affecting mood"""
        
        # Store influence for history tracking
        influence_record = {
            "timestamp": datetime.now(),
            "update_type": update.update_type,
            "influence_strength": 0.0,
            "mood_before": await self._get_current_mood_state()
        }
        
        if update.update_type == "emotional_state_update":
            # Emotions have strong influence on mood
            emotional_data = update.data
            influence = await self._process_emotional_influence(emotional_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "needs_state_change":
            # Unmet needs negatively affect mood
            needs_data = update.data
            influence = await self._process_needs_influence(needs_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "goal_progress":
            # Goal achievement positively affects mood
            goal_data = update.data
            influence = await self._process_goal_influence(goal_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "reward_signal":
            # Rewards boost mood
            reward_data = update.data
            influence = await self._process_reward_influence(reward_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "hormone_level_update":
            # Hormones directly influence mood
            hormone_data = update.data
            influence = await self._process_hormone_influence(hormone_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "dominance_gratification":
            # Dominance success affects mood positively
            dominance_data = update.data
            influence = await self._process_dominance_influence(dominance_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "stress_indicator":
            # Stress negatively affects mood
            stress_data = update.data
            influence = await self._process_stress_influence(stress_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "social_interaction":
            # Social interactions affect mood based on quality
            social_data = update.data
            influence = await self._process_social_influence(social_data)
            influence_record["influence_strength"] = influence
            
        elif update.update_type == "achievement_unlocked":
            # Achievements boost mood
            achievement_data = update.data
            influence = await self._process_achievement_influence(achievement_data)
            influence_record["influence_strength"] = influence
        
        # Record mood after processing
        influence_record["mood_after"] = await self._get_current_mood_state()
        
        # Add to history
        self._add_to_influence_history(influence_record)
        
        # Check if mood changed significantly
        if abs(influence_record["influence_strength"]) > 0.1:
            await self._check_and_send_mood_update()
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with mood awareness"""
        # Update mood based on input context
        mood_update_result = await self.original_manager.update_mood({
            "user_input": context.user_input,
            "emotional_state": context.emotional_state,
            "session_context": context.session_context
        })
        
        # Get current mood state after update
        current_mood = await self._get_current_mood_state()
        
        # Analyze mood impact on processing
        mood_impact = await self._analyze_mood_impact(current_mood)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        mood_coherence = await self._analyze_mood_coherence(current_mood, messages)
        
        # Send mood state if changed
        if mood_update_result.get("mood_changed", False):
            await self.send_context_update(
                update_type="mood_state_update",
                data={
                    "dominant_mood": current_mood["dominant_mood"],
                    "valence": current_mood["valence"],
                    "arousal": current_mood["arousal"],
                    "control": current_mood["control"],
                    "mood_vector": current_mood["mood_vector"],
                    "change_magnitude": mood_update_result.get("change_magnitude", 0.0)
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "mood_processing_complete": True,
            "current_mood": current_mood,
            "mood_impact": mood_impact,
            "mood_coherence": mood_coherence,
            "mood_update_result": mood_update_result
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze mood patterns and stability"""
        # Get current mood state
        current_mood = await self._get_current_mood_state()
        
        # Analyze mood stability over time
        stability_analysis = await self._analyze_mood_stability_detailed()
        
        # Analyze mood patterns
        mood_patterns = await self._analyze_mood_patterns()
        
        # Analyze mood-behavior correlations
        behavior_correlations = await self._analyze_mood_behavior_correlations(context)
        
        # Generate mood insights
        mood_insights = await self._generate_mood_insights(
            current_mood, stability_analysis, mood_patterns
        )
        
        return {
            "current_mood_analysis": current_mood,
            "stability_analysis": stability_analysis,
            "mood_patterns": mood_patterns,
            "behavior_correlations": behavior_correlations,
            "mood_insights": mood_insights,
            "mood_health": await self._assess_mood_health()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize mood influence on response generation"""
        # Get current mood state
        current_mood = await self._get_current_mood_state()
        
        # Calculate mood-based response modulations
        response_modulations = await self._calculate_response_modulations(current_mood)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Generate mood-aware guidance
        mood_guidance = {
            "energy_level": self._calculate_energy_from_mood(current_mood),
            "openness_level": self._calculate_openness_from_mood(current_mood),
            "emotional_depth": self._calculate_emotional_depth_from_mood(current_mood),
            "response_style": await self._determine_response_style_from_mood(current_mood),
            "interaction_preferences": await self._get_mood_based_preferences(current_mood)
        }
        
        # Check for mood-based constraints
        constraints = await self._get_mood_constraints(current_mood)
        
        # Send synthesis results
        await self.send_context_update(
            update_type="mood_synthesis_complete",
            data={
                "mood_guidance": mood_guidance,
                "response_modulations": response_modulations,
                "mood_constraints": constraints,
                "synthesis_confidence": await self._calculate_mood_confidence()
            }
        )
        
        return {
            "mood_synthesis": mood_guidance,
            "modulations": response_modulations,
            "constraints": constraints,
            "synthesis_complete": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _get_current_mood_state(self) -> Dict[str, Any]:
        """Get comprehensive current mood state"""
        mood_context = self.original_manager.mood_context
        current_mood = mood_context.current_mood
        
        # Convert to comprehensive state
        mood_state = {
            "dominant_mood": current_mood.name,
            "valence": current_mood.valence,
            "arousal": current_mood.arousal,
            "control": current_mood.dominance,  # Some implementations use 'dominance' instead of 'control'
            "mood_vector": current_mood.vector.tolist() if hasattr(current_mood.vector, 'tolist') else list(current_mood.vector),
            "intensity": np.linalg.norm(current_mood.vector) if hasattr(np, 'linalg') else 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add stability indicator
        mood_state["stability"] = await self._calculate_mood_stability()
        
        return mood_state
    
    async def _analyze_context_for_mood(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze context for mood influences"""
        influences = {
            "positive_influences": [],
            "negative_influences": [],
            "arousal_influences": [],
            "control_influences": [],
            "net_influence": 0.0
        }
        
        # Analyze user input sentiment
        user_input_lower = context.user_input.lower()
        
        # Positive indicators
        positive_words = ["happy", "great", "wonderful", "excellent", "love", "amazing", "fantastic"]
        positive_count = sum(1 for word in positive_words if word in user_input_lower)
        if positive_count > 0:
            influences["positive_influences"].append({
                "source": "user_input_sentiment",
                "strength": min(0.3, positive_count * 0.1)
            })
        
        # Negative indicators
        negative_words = ["sad", "angry", "frustrated", "terrible", "hate", "awful", "horrible"]
        negative_count = sum(1 for word in negative_words if word in user_input_lower)
        if negative_count > 0:
            influences["negative_influences"].append({
                "source": "user_input_sentiment",
                "strength": min(0.3, negative_count * 0.1)
            })
        
        # High energy indicators
        if "!" in context.user_input:
            influences["arousal_influences"].append({
                "source": "punctuation_energy",
                "strength": min(0.2, context.user_input.count("!") * 0.05)
            })
        
        # Control indicators
        if any(word in user_input_lower for word in ["please", "help", "need"]):
            influences["control_influences"].append({
                "source": "user_dependency",
                "strength": 0.15
            })
        elif any(word in user_input_lower for word in ["command", "order", "demand"]):
            influences["control_influences"].append({
                "source": "user_assertion",
                "strength": -0.15
            })
        
        # Calculate net influence
        positive_total = sum(inf["strength"] for inf in influences["positive_influences"])
        negative_total = sum(inf["strength"] for inf in influences["negative_influences"])
        influences["net_influence"] = positive_total - negative_total
        
        return influences
    
    async def _calculate_mood_trajectory(self, current_mood: Dict[str, Any], influences: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate predicted mood trajectory"""
        trajectory = {
            "predicted_valence_change": 0.0,
            "predicted_arousal_change": 0.0,
            "predicted_control_change": 0.0,
            "time_horizon": "next_hour",
            "confidence": 0.7
        }
        
        # Base trajectory on current influences
        net_influence = influences.get("net_influence", 0.0)
        
        # Valence trajectory
        trajectory["predicted_valence_change"] = net_influence * 0.3
        
        # Arousal trajectory
        arousal_influence = sum(inf["strength"] for inf in influences.get("arousal_influences", []))
        trajectory["predicted_arousal_change"] = arousal_influence * 0.2
        
        # Control trajectory
        control_influence = sum(inf["strength"] for inf in influences.get("control_influences", []))
        trajectory["predicted_control_change"] = control_influence * 0.2
        
        # Add momentum from current mood
        current_valence = current_mood.get("valence", 0.0)
        if abs(current_valence) > 0.5:
            # Strong moods tend to persist
            trajectory["predicted_valence_change"] += current_valence * 0.1
        
        return trajectory
    
    async def _calculate_mood_stability(self) -> float:
        """Calculate mood stability score"""
        if len(self.mood_influence_history) < 5:
            return 0.8  # Default to relatively stable
        
        # Get recent mood changes
        recent_changes = []
        for i in range(1, min(11, len(self.mood_influence_history))):
            record = self.mood_influence_history[-i]
            if record["mood_before"] and record["mood_after"]:
                # Calculate change magnitude
                before_vector = record["mood_before"].get("mood_vector", [0, 0, 0])
                after_vector = record["mood_after"].get("mood_vector", [0, 0, 0])
                
                if len(before_vector) >= 3 and len(after_vector) >= 3:
                    change = sum((a - b) ** 2 for a, b in zip(after_vector[:3], before_vector[:3])) ** 0.5
                    recent_changes.append(change)
        
        if not recent_changes:
            return 0.8
        
        # Calculate stability (inverse of average change)
        avg_change = sum(recent_changes) / len(recent_changes)
        stability = max(0.0, min(1.0, 1.0 - avg_change))
        
        return stability
    
    async def _process_emotional_influence(self, emotional_data: Dict[str, Any]) -> float:
        """Process emotional state influence on mood"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return 0.0
        
        emotion_name, intensity = dominant_emotion
        
        # Define emotion to mood mappings
        emotion_mood_map = {
            "Joy": {"valence": 0.8, "arousal": 0.6, "control": 0.2},
            "Sadness": {"valence": -0.7, "arousal": -0.3, "control": -0.2},
            "Anger": {"valence": -0.6, "arousal": 0.8, "control": 0.3},
            "Fear": {"valence": -0.5, "arousal": 0.7, "control": -0.6},
            "Love": {"valence": 0.9, "arousal": 0.4, "control": 0.1},
            "Excitement": {"valence": 0.7, "arousal": 0.9, "control": 0.3},
            "Frustration": {"valence": -0.4, "arousal": 0.5, "control": -0.3},
            "Pride": {"valence": 0.6, "arousal": 0.3, "control": 0.7},
            "Curiosity": {"valence": 0.3, "arousal": 0.6, "control": 0.2}
        }
        
        if emotion_name in emotion_mood_map:
            mood_influence = emotion_mood_map[emotion_name]
            
            # Apply influence scaled by intensity
            mood_update = {
                "valence": mood_influence["valence"] * intensity * 0.4,
                "arousal": mood_influence["arousal"] * intensity * 0.4,
                "control": mood_influence["control"] * intensity * 0.3
            }
            
            # Update mood through original manager
            await self.original_manager._apply_mood_influence(mood_update, f"emotion_{emotion_name}")
            
            # Return influence magnitude
            return (abs(mood_update["valence"]) + abs(mood_update["arousal"]) + abs(mood_update["control"])) / 3
        
        return 0.0
    
    async def _process_needs_influence(self, needs_data: Dict[str, Any]) -> float:
        """Process needs state influence on mood"""
        unmet_needs = needs_data.get("unmet_needs", [])
        high_priority_needs = needs_data.get("high_priority_needs", [])
        needs_satisfaction = needs_data.get("overall_satisfaction", 0.5)
        
        influence_strength = 0.0
        
        # Low satisfaction negatively affects mood
        if needs_satisfaction < 0.3:
            mood_update = {
                "valence": -0.3 * (1 - needs_satisfaction),
                "arousal": 0.1,  # Slight increase in arousal from urgency
                "control": -0.2 * (1 - needs_satisfaction)
            }
            
            await self.original_manager._apply_mood_influence(mood_update, "low_needs_satisfaction")
            influence_strength = 0.3 * (1 - needs_satisfaction)
        
        # High priority unmet needs create frustration
        if high_priority_needs:
            frustration_level = min(0.4, len(high_priority_needs) * 0.1)
            mood_update = {
                "valence": -frustration_level,
                "arousal": frustration_level * 0.5,
                "control": -frustration_level * 0.7
            }
            
            await self.original_manager._apply_mood_influence(mood_update, "unmet_high_priority_needs")
            influence_strength = max(influence_strength, frustration_level)
        
        return influence_strength
    
    async def _process_goal_influence(self, goal_data: Dict[str, Any]) -> float:
        """Process goal progress influence on mood"""
        goal_completed = goal_data.get("goal_completed", False)
        goal_failed = goal_data.get("goal_failed", False)
        progress_made = goal_data.get("progress_made", False)
        goal_importance = goal_data.get("importance", 0.5)
        
        influence_strength = 0.0
        
        if goal_completed:
            # Goal completion boosts mood
            boost = 0.3 + (goal_importance * 0.3)
            mood_update = {
                "valence": boost,
                "arousal": boost * 0.5,
                "control": boost * 0.7
            }
            
            await self.original_manager._apply_mood_influence(mood_update, "goal_completed")
            influence_strength = boost
            
        elif goal_failed:
            # Goal failure decreases mood
            decrease = -0.2 - (goal_importance * 0.3)
            mood_update = {
                "valence": decrease,
                "arousal": abs(decrease) * 0.3,
                "control": decrease * 0.8
            }
            
            await self.original_manager._apply_mood_influence(mood_update, "goal_failed")
            influence_strength = abs(decrease)
            
        elif progress_made:
            # Progress provides small boost
            boost = 0.1 + (goal_importance * 0.1)
            mood_update = {
                "valence": boost,
                "arousal": 0.0,
                "control": boost * 0.5
            }
            
            await self.original_manager._apply_mood_influence(mood_update, "goal_progress")
            influence_strength = boost
        
        return influence_strength
    
    async def _process_reward_influence(self, reward_data: Dict[str, Any]) -> float:
        """Process reward signal influence on mood"""
        reward_value = reward_data.get("reward_value", 0.0)
        reward_type = reward_data.get("reward_type", "general")
        
        if reward_value > 0:
            # Positive reward improves mood
            mood_boost = min(0.5, reward_value * 0.3)
            
            mood_update = {
                "valence": mood_boost,
                "arousal": mood_boost * 0.4,
                "control": mood_boost * 0.3
            }
            
            # Special handling for dominance rewards
            if reward_type == "dominance":
                mood_update["control"] = mood_boost * 0.8
            
            await self.original_manager._apply_mood_influence(mood_update, f"reward_{reward_type}")
            
            return mood_boost
        
        elif reward_value < 0:
            # Negative reward (punishment) decreases mood
            mood_decrease = max(-0.5, reward_value * 0.3)
            
            mood_update = {
                "valence": mood_decrease,
                "arousal": abs(mood_decrease) * 0.3,
                "control": mood_decrease * 0.5
            }
            
            await self.original_manager._apply_mood_influence(mood_update, f"punishment_{reward_type}")
            
            return abs(mood_decrease)
        
        return 0.0
    
    async def _process_hormone_influence(self, hormone_data: Dict[str, Any]) -> float:
        """Process hormone level influence on mood"""
        hormone_levels = hormone_data.get("levels", {})
        
        influence_strength = 0.0
        mood_update = {"valence": 0.0, "arousal": 0.0, "control": 0.0}
        
        # Serotonin affects valence
        serotonin = hormone_levels.get("serotonin", 0.5)
        mood_update["valence"] += (serotonin - 0.5) * 0.4
        
        # Dopamine affects both valence and arousal
        dopamine = hormone_levels.get("dopamine", 0.5)
        mood_update["valence"] += (dopamine - 0.5) * 0.3
        mood_update["arousal"] += (dopamine - 0.5) * 0.4
        
        # Oxytocin affects valence positively
        oxytocin = hormone_levels.get("oxytocin", 0.5)
        mood_update["valence"] += (oxytocin - 0.5) * 0.3
        
        # Cortisol affects mood negatively
        cortisol = hormone_levels.get("cortisol", 0.5)
        mood_update["valence"] -= (cortisol - 0.5) * 0.4
        mood_update["arousal"] += (cortisol - 0.5) * 0.3
        
        # Testosterone affects control/dominance
        testosterone = hormone_levels.get("testosterone", 0.5)
        mood_update["control"] += (testosterone - 0.5) * 0.5
        
        # Apply hormone influence
        if any(abs(v) > 0.01 for v in mood_update.values()):
            await self.original_manager._apply_mood_influence(mood_update, "hormone_levels")
            influence_strength = sum(abs(v) for v in mood_update.values()) / 3
        
        return influence_strength
    
    async def _process_dominance_influence(self, dominance_data: Dict[str, Any]) -> float:
        """Process dominance gratification influence on mood"""
        gratification_type = dominance_data.get("gratification_type", "general")
        intensity = dominance_data.get("intensity", 0.5)
        success = dominance_data.get("success", True)
        
        if success:
            # Successful dominance significantly boosts mood
            mood_boost = 0.3 + (intensity * 0.4)
            
            mood_update = {
                "valence": mood_boost * 0.8,
                "arousal": mood_boost * 0.6,
                "control": mood_boost  # Full boost to control
            }
            
            await self.original_manager._apply_mood_influence(mood_update, f"dominance_success_{gratification_type}")
            
            return mood_boost
        else:
            # Failed dominance attempt frustrates
            mood_decrease = -0.2 - (intensity * 0.2)
            
            mood_update = {
                "valence": mood_decrease,
                "arousal": abs(mood_decrease) * 0.4,
                "control": mood_decrease * 1.2  # Extra hit to control
            }
            
            await self.original_manager._apply_mood_influence(mood_update, "dominance_failure")
            
            return abs(mood_decrease)
    
    async def _process_stress_influence(self, stress_data: Dict[str, Any]) -> float:
        """Process stress indicator influence on mood"""
        stress_level = stress_data.get("stress_level", 0.5)
        stress_source = stress_data.get("source", "general")
        
        if stress_level > 0.5:
            # High stress negatively affects mood
            stress_impact = (stress_level - 0.5) * 0.6
            
            mood_update = {
                "valence": -stress_impact,
                "arousal": stress_impact * 0.8,  # Stress increases arousal
                "control": -stress_impact * 0.7
            }
            
            await self.original_manager._apply_mood_influence(mood_update, f"stress_{stress_source}")
            
            return stress_impact
        
        return 0.0
    
    async def _process_social_influence(self, social_data: Dict[str, Any]) -> float:
        """Process social interaction influence on mood"""
        interaction_quality = social_data.get("quality", 0.5)
        interaction_type = social_data.get("type", "general")
        
        # Quality determines mood impact
        if interaction_quality > 0.6:
            # Positive interaction
            mood_boost = (interaction_quality - 0.5) * 0.4
            
            mood_update = {
                "valence": mood_boost,
                "arousal": mood_boost * 0.3,
                "control": mood_boost * 0.2
            }
            
            await self.original_manager._apply_mood_influence(mood_update, f"positive_{interaction_type}_interaction")
            
            return mood_boost
            
        elif interaction_quality < 0.4:
            # Negative interaction
            mood_decrease = (0.5 - interaction_quality) * 0.4
            
            mood_update = {
                "valence": -mood_decrease,
                "arousal": mood_decrease * 0.4,
                "control": -mood_decrease * 0.3
            }
            
            await self.original_manager._apply_mood_influence(mood_update, f"negative_{interaction_type}_interaction")
            
            return mood_decrease
        
        return 0.0
    
    async def _process_achievement_influence(self, achievement_data: Dict[str, Any]) -> float:
        """Process achievement influence on mood"""
        achievement_type = achievement_data.get("type", "general")
        achievement_value = achievement_data.get("value", 0.5)
        
        # Achievements always boost mood
        mood_boost = 0.2 + (achievement_value * 0.3)
        
        mood_update = {
            "valence": mood_boost,
            "arousal": mood_boost * 0.4,
            "control": mood_boost * 0.6
        }
        
        await self.original_manager._apply_mood_influence(mood_update, f"achievement_{achievement_type}")
        
        return mood_boost
    
    def _add_to_influence_history(self, record: Dict[str, Any]):
        """Add influence record to history"""
        self.mood_influence_history.append(record)
        
        # Maintain max history size
        if len(self.mood_influence_history) > self.max_history_size:
            self.mood_influence_history = self.mood_influence_history[-self.max_history_size:]
    
    async def _check_and_send_mood_update(self):
        """Check if mood changed significantly and send update"""
        if len(self.mood_influence_history) < 1:
            return
        
        latest_record = self.mood_influence_history[-1]
        
        if latest_record["mood_before"] and latest_record["mood_after"]:
            # Check change magnitude
            before = latest_record["mood_before"]
            after = latest_record["mood_after"]
            
            valence_change = abs(after.get("valence", 0) - before.get("valence", 0))
            arousal_change = abs(after.get("arousal", 0) - before.get("arousal", 0))
            control_change = abs(after.get("control", 0) - before.get("control", 0))
            
            total_change = valence_change + arousal_change + control_change
            
            if total_change > 0.15:  # Significant change threshold
                await self.send_context_update(
                    update_type="mood_state_update",
                    data={
                        "dominant_mood": after["dominant_mood"],
                        "valence": after["valence"],
                        "arousal": after["arousal"],
                        "control": after["control"],
                        "mood_vector": after["mood_vector"],
                        "change_magnitude": total_change,
                        "change_source": latest_record["update_type"]
                    },
                    priority=ContextPriority.HIGH if total_change > 0.3 else ContextPriority.NORMAL
                )
    
    async def _analyze_mood_impact(self, current_mood: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how current mood impacts processing"""
        impact = {
            "cognitive_impact": {},
            "behavioral_tendencies": {},
            "interaction_style": {},
            "processing_biases": []
        }
        
        valence = current_mood.get("valence", 0.0)
        arousal = current_mood.get("arousal", 0.5)
        control = current_mood.get("control", 0.0)
        
        # Cognitive impact
        if valence > 0.3:
            impact["cognitive_impact"]["creativity"] = 0.2
            impact["cognitive_impact"]["openness"] = 0.3
            impact["processing_biases"].append("positive_attribution_bias")
        elif valence < -0.3:
            impact["cognitive_impact"]["critical_thinking"] = 0.2
            impact["cognitive_impact"]["caution"] = 0.3
            impact["processing_biases"].append("threat_detection_bias")
        
        if arousal > 0.7:
            impact["cognitive_impact"]["processing_speed"] = 0.3
            impact["cognitive_impact"]["focus_breadth"] = -0.2
        elif arousal < 0.3:
            impact["cognitive_impact"]["deep_processing"] = 0.3
            impact["cognitive_impact"]["patience"] = 0.4
        
        # Behavioral tendencies
        if control > 0.3:
            impact["behavioral_tendencies"]["assertiveness"] = 0.4
            impact["behavioral_tendencies"]["initiative_taking"] = 0.3
        elif control < -0.3:
            impact["behavioral_tendencies"]["help_seeking"] = 0.3
            impact["behavioral_tendencies"]["compliance"] = 0.2
        
        # Interaction style
        dominant_mood = current_mood.get("dominant_mood", "Neutral")
        mood_styles = {
            "Happy": {"warmth": 0.4, "enthusiasm": 0.3},
            "Sad": {"empathy": 0.3, "gentleness": 0.4},
            "Energetic": {"dynamism": 0.4, "expressiveness": 0.3},
            "Calm": {"patience": 0.4, "thoughtfulness": 0.3},
            "Confident": {"directness": 0.3, "leadership": 0.4}
        }
        
        if dominant_mood in mood_styles:
            impact["interaction_style"] = mood_styles[dominant_mood]
        
        return impact
    
    async def _analyze_mood_coherence(self, current_mood: Dict[str, Any], messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze coherence between mood and other system states"""
        coherence = {
            "overall_coherence": 0.8,
            "coherence_issues": [],
            "synergies": []
        }
        
        # Check emotional coherence
        for module_messages in messages.values():
            for msg in module_messages:
                if msg["type"] == "emotional_state_update":
                    emotion_data = msg["data"]
                    emotion_coherence = await self._check_emotion_mood_coherence(
                        current_mood, emotion_data
                    )
                    coherence["emotion_coherence"] = emotion_coherence
                    
                elif msg["type"] == "mode_context_initialized":
                    mode_data = msg["data"]
                    mode_coherence = await self._check_mode_mood_coherence(
                        current_mood, mode_data
                    )
                    coherence["mode_coherence"] = mode_coherence
        
        # Calculate overall coherence
        coherence_scores = []
        if "emotion_coherence" in coherence:
            coherence_scores.append(coherence["emotion_coherence"]["score"])
        if "mode_coherence" in coherence:
            coherence_scores.append(coherence["mode_coherence"]["score"])
            
        if coherence_scores:
            coherence["overall_coherence"] = sum(coherence_scores) / len(coherence_scores)
        
        return coherence
    
    async def _check_emotion_mood_coherence(self, mood: Dict[str, Any], emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check coherence between mood and emotional state"""
        coherence = {
            "score": 0.7,
            "issues": []
        }
        
        dominant_emotion = emotion_data.get("dominant_emotion")
        if dominant_emotion:
            emotion_name, intensity = dominant_emotion
            mood_valence = mood.get("valence", 0.0)
            
            # Check for conflicts
            if emotion_name in ["Joy", "Love", "Excitement"] and mood_valence < -0.3:
                coherence["issues"].append("Positive emotion but negative mood")
                coherence["score"] -= 0.3
                
            elif emotion_name in ["Sadness", "Anger", "Fear"] and mood_valence > 0.3:
                coherence["issues"].append("Negative emotion but positive mood")
                coherence["score"] -= 0.3
            else:
                # Good alignment
                coherence["score"] = min(1.0, coherence["score"] + 0.2)
        
        return coherence
    
    async def _check_mode_mood_coherence(self, mood: Dict[str, Any], mode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check coherence between mood and interaction modes"""
        coherence = {
            "score": 0.7,
            "issues": []
        }
        
        primary_mode = mode_data.get("primary_mode")
        mood_name = mood.get("dominant_mood", "Neutral")
        
        # Check for conflicts
        conflicting_combinations = [
            ("playful", "Sad"),
            ("dominant", "Anxious"),
            ("professional", "Excited")
        ]
        
        for mode, mood_conflict in conflicting_combinations:
            if primary_mode == mode and mood_name == mood_conflict:
                coherence["issues"].append(f"{mode} mode conflicts with {mood_conflict} mood")
                coherence["score"] -= 0.2
        
        # Check for synergies
        synergistic_combinations = [
            ("playful", "Happy"),
            ("compassionate", "Loving"),
            ("intellectual", "Curious")
        ]
        
        for mode, mood_synergy in synergistic_combinations:
            if primary_mode == mode and mood_name == mood_synergy:
                coherence["score"] = min(1.0, coherence["score"] + 0.2)
        
        return coherence
    
    async def _analyze_mood_stability_detailed(self) -> Dict[str, Any]:
        """Detailed analysis of mood stability"""
        stability = {
            "overall_stability": await self._calculate_mood_stability(),
            "valence_stability": 0.8,
            "arousal_stability": 0.8,
            "control_stability": 0.8,
            "volatility_periods": []
        }
        
        if len(self.mood_influence_history) < 10:
            return stability
        
        # Analyze each dimension
        dimensions = ["valence", "arousal", "control"]
        
        for dimension in dimensions:
            values = []
            for record in self.mood_influence_history[-20:]:
                if record["mood_after"]:
                    values.append(record["mood_after"].get(dimension, 0.0))
            
            if len(values) >= 3:
                # Calculate variance
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                
                # Convert to stability (inverse of variance)
                stability[f"{dimension}_stability"] = max(0.0, min(1.0, 1.0 - (variance * 2)))
        
        # Identify volatility periods
        for i in range(1, min(11, len(self.mood_influence_history))):
            record = self.mood_influence_history[-i]
            if record["influence_strength"] > 0.3:
                stability["volatility_periods"].append({
                    "timestamp": record["timestamp"].isoformat(),
                    "trigger": record["update_type"],
                    "strength": record["influence_strength"]
                })
        
        return stability
    
    async def _analyze_mood_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in mood changes"""
        patterns = {
            "cyclic_patterns": [],
            "trigger_patterns": {},
            "mood_preferences": {},
            "common_transitions": []
        }
        
        if len(self.mood_influence_history) < 20:
            return patterns
        
        # Analyze trigger patterns
        trigger_impacts = {}
        for record in self.mood_influence_history:
            trigger = record["update_type"]
            impact = record["influence_strength"]
            
            if trigger not in trigger_impacts:
                trigger_impacts[trigger] = []
            trigger_impacts[trigger].append(impact)
        
        # Calculate average impact per trigger
        for trigger, impacts in trigger_impacts.items():
            if impacts:
                patterns["trigger_patterns"][trigger] = {
                    "average_impact": sum(impacts) / len(impacts),
                    "frequency": len(impacts),
                    "max_impact": max(impacts)
                }
        
        # Analyze mood preferences (which moods appear most often)
        mood_counts = {}
        for record in self.mood_influence_history:
            if record["mood_after"]:
                mood = record["mood_after"].get("dominant_mood", "Neutral")
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        total_counts = sum(mood_counts.values())
        if total_counts > 0:
            patterns["mood_preferences"] = {
                mood: count / total_counts 
                for mood, count in mood_counts.items()
            }
        
        # Analyze common transitions
        transitions = {}
        for i in range(1, len(self.mood_influence_history)):
            prev_mood = self.mood_influence_history[i-1]["mood_after"]
            curr_mood = self.mood_influence_history[i]["mood_after"]
            
            if prev_mood and curr_mood:
                prev_name = prev_mood.get("dominant_mood", "Neutral")
                curr_name = curr_mood.get("dominant_mood", "Neutral")
                
                if prev_name != curr_name:
                    transition = f"{prev_name}->{curr_name}"
                    transitions[transition] = transitions.get(transition, 0) + 1
        
        # Get top transitions
        if transitions:
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            patterns["common_transitions"] = [
                {"transition": t, "frequency": f} 
                for t, f in sorted_transitions[:5]
            ]
        
        return patterns
    
    async def _analyze_mood_behavior_correlations(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze correlations between mood and behavior"""
        correlations = {
            "mood_response_correlations": {},
            "mood_mode_correlations": {},
            "mood_effectiveness": {}
        }
        
        # This would require tracking response success rates per mood
        # For now, return theoretical correlations
        
        current_mood = await self._get_current_mood_state()
        dominant_mood = current_mood.get("dominant_mood", "Neutral")
        
        # Theoretical mood-behavior correlations
        mood_behaviors = {
            "Happy": {
                "response_length": 1.2,  # Tend to be more verbose
                "creativity": 1.3,
                "helpfulness": 1.4
            },
            "Sad": {
                "response_length": 0.8,
                "empathy": 1.5,
                "processing_speed": 0.7
            },
            "Energetic": {
                "response_speed": 1.4,
                "initiative": 1.3,
                "focus": 0.8
            },
            "Calm": {
                "thoughtfulness": 1.3,
                "accuracy": 1.2,
                "patience": 1.5
            }
        }
        
        if dominant_mood in mood_behaviors:
            correlations["mood_response_correlations"] = mood_behaviors[dominant_mood]
        
        return correlations
    
    async def _generate_mood_insights(self, current_mood: Dict[str, Any], 
                                    stability: Dict[str, Any], 
                                    patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about mood state and patterns"""
        insights = []
        
        # Stability insights
        if stability["overall_stability"] < 0.5:
            insights.append({
                "type": "stability_warning",
                "description": "Mood has been volatile recently",
                "recommendation": "Consider mood stabilization strategies"
            })
        elif stability["overall_stability"] > 0.8:
            insights.append({
                "type": "stability_strength",
                "description": "Mood has been very stable",
                "implication": "Good emotional regulation"
            })
        
        # Pattern insights
        if patterns["trigger_patterns"]:
            # Find most impactful trigger
            most_impactful = max(
                patterns["trigger_patterns"].items(),
                key=lambda x: x[1]["average_impact"]
            )
            insights.append({
                "type": "trigger_insight",
                "description": f"{most_impactful[0]} has strongest mood impact",
                "average_impact": most_impactful[1]["average_impact"]
            })
        
        # Mood preference insights
        if patterns["mood_preferences"]:
            dominant_preference = max(
                patterns["mood_preferences"].items(),
                key=lambda x: x[1]
            )
            if dominant_preference[1] > 0.4:
                insights.append({
                    "type": "mood_preference",
                    "description": f"Tendency toward {dominant_preference[0]} mood",
                    "frequency": dominant_preference[1]
                })
        
        # Current state insights
        valence = current_mood.get("valence", 0.0)
        if abs(valence) > 0.7:
            insights.append({
                "type": "extreme_valence",
                "description": f"Currently in {'very positive' if valence > 0 else 'very negative'} mood state",
                "recommendation": "May affect decision-making and interactions"
            })
        
        return insights
    
    async def _assess_mood_health(self) -> Dict[str, Any]:
        """Assess overall mood system health"""
        health = {
            "health_score": 0.7,
            "indicators": {},
            "issues": [],
            "strengths": []
        }
        
        # Stability indicator
        stability = await self._calculate_mood_stability()
        health["indicators"]["stability"] = stability
        
        if stability < 0.4:
            health["issues"].append("Low mood stability")
            health["health_score"] -= 0.2
        elif stability > 0.7:
            health["strengths"].append("Good mood stability")
            health["health_score"] += 0.1
        
        # Responsiveness indicator
        if len(self.mood_influence_history) >= 10:
            recent_influences = [
                record["influence_strength"] 
                for record in self.mood_influence_history[-10:]
            ]
            avg_responsiveness = sum(recent_influences) / len(recent_influences)
            
            health["indicators"]["responsiveness"] = avg_responsiveness
            
            if avg_responsiveness < 0.1:
                health["issues"].append("Low mood responsiveness")
                health["health_score"] -= 0.1
            elif avg_responsiveness > 0.4:
                health["issues"].append("Hyper-responsive mood system")
                health["health_score"] -= 0.15
        
        # Balance indicator
        current_mood = await self._get_current_mood_state()
        valence = current_mood.get("valence", 0.0)
        
        if abs(valence) < 0.8:
            health["strengths"].append("Balanced mood state")
            health["health_score"] += 0.1
        else:
            health["issues"].append("Extreme mood state")
            health["health_score"] -= 0.1
        
        health["health_score"] = max(0.0, min(1.0, health["health_score"]))
        
        return health
    
    async def _calculate_response_modulations(self, current_mood: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how mood should modulate responses"""
        modulations = {
            "tone_shift": 0.0,
            "energy_adjustment": 0.0,
            "verbosity_modifier": 1.0,
            "formality_adjustment": 0.0,
            "creativity_boost": 0.0
        }
        
        valence = current_mood.get("valence", 0.0)
        arousal = current_mood.get("arousal", 0.5)
        control = current_mood.get("control", 0.0)
        
        # Valence affects tone
        modulations["tone_shift"] = valence * 0.3
        
        # Arousal affects energy and verbosity
        modulations["energy_adjustment"] = (arousal - 0.5) * 0.4
        if arousal > 0.7:
            modulations["verbosity_modifier"] = 1.2
        elif arousal < 0.3:
            modulations["verbosity_modifier"] = 0.8
        
        # Control affects formality
        if control > 0.3:
            modulations["formality_adjustment"] = -0.2  # Less formal when feeling in control
        elif control < -0.3:
            modulations["formality_adjustment"] = 0.2  # More formal when feeling less control
        
        # Positive mood boosts creativity
        if valence > 0.3:
            modulations["creativity_boost"] = valence * 0.3
        
        return modulations
    
    def _calculate_energy_from_mood(self, mood: Dict[str, Any]) -> float:
        """Calculate energy level from mood state"""
        arousal = mood.get("arousal", 0.5)
        valence = mood.get("valence", 0.0)
        
        # Base energy on arousal
        energy = arousal
        
        # Positive valence adds some energy
        if valence > 0:
            energy += valence * 0.2
        
        return max(0.0, min(1.0, energy))
    
    def _calculate_openness_from_mood(self, mood: Dict[str, Any]) -> float:
        """Calculate openness level from mood state"""
        valence = mood.get("valence", 0.0)
        control = mood.get("control", 0.0)
        
        # Positive mood increases openness
        openness = 0.5 + (valence * 0.3)
        
        # Feeling in control also increases openness
        openness += control * 0.2
        
        return max(0.0, min(1.0, openness))
    
    def _calculate_emotional_depth_from_mood(self, mood: Dict[str, Any]) -> float:
        """Calculate emotional depth from mood state"""
        intensity = mood.get("intensity", 0.5)
        valence = abs(mood.get("valence", 0.0))
        
        # Strong moods (positive or negative) increase emotional depth
        depth = 0.3 + (intensity * 0.4) + (valence * 0.3)
        
        return max(0.0, min(1.0, depth))
    
    async def _determine_response_style_from_mood(self, mood: Dict[str, Any]) -> Dict[str, Any]:
        """Determine response style based on mood"""
        style = {
            "approach": "balanced",
            "tone": "neutral",
            "pace": "moderate"
        }
        
        dominant_mood = mood.get("dominant_mood", "Neutral")
        valence = mood.get("valence", 0.0)
        arousal = mood.get("arousal", 0.5)
        
        # Mood-specific styles
        mood_styles = {
            "Happy": {"approach": "enthusiastic", "tone": "warm", "pace": "lively"},
            "Sad": {"approach": "gentle", "tone": "empathetic", "pace": "slow"},
            "Excited": {"approach": "energetic", "tone": "bright", "pace": "quick"},
            "Calm": {"approach": "thoughtful", "tone": "serene", "pace": "measured"},
            "Confident": {"approach": "assertive", "tone": "assured", "pace": "steady"},
            "Anxious": {"approach": "cautious", "tone": "tentative", "pace": "variable"}
        }
        
        if dominant_mood in mood_styles:
            style.update(mood_styles[dominant_mood])
        else:
            # Derive from dimensions
            if valence > 0.3:
                style["tone"] = "positive"
            elif valence < -0.3:
                style["tone"] = "understanding"
                
            if arousal > 0.7:
                style["pace"] = "dynamic"
            elif arousal < 0.3:
                style["pace"] = "relaxed"
        
        return style
    
    async def _get_mood_based_preferences(self, mood: Dict[str, Any]) -> Dict[str, Any]:
        """Get interaction preferences based on mood"""
        preferences = {
            "interaction_distance": "moderate",
            "information_processing": "balanced",
            "decision_style": "considered",
            "communication_preference": "direct"
        }
        
        valence = mood.get("valence", 0.0)
        arousal = mood.get("arousal", 0.5)
        control = mood.get("control", 0.0)
        
        # High positive mood - more social
        if valence > 0.5:
            preferences["interaction_distance"] = "close"
            preferences["communication_preference"] = "expressive"
        # Negative mood - more reserved
        elif valence < -0.3:
            preferences["interaction_distance"] = "distant"
            preferences["communication_preference"] = "careful"
        
        # High arousal - quick processing
        if arousal > 0.7:
            preferences["information_processing"] = "rapid"
            preferences["decision_style"] = "intuitive"
        # Low arousal - thorough processing
        elif arousal < 0.3:
            preferences["information_processing"] = "thorough"
            preferences["decision_style"] = "analytical"
        
        # High control - more decisive
        if control > 0.5:
            preferences["decision_style"] = "decisive"
            preferences["communication_preference"] = "directive"
        
        return preferences
    
    async def _get_mood_constraints(self, mood: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get constraints based on current mood"""
        constraints = []
        
        valence = mood.get("valence", 0.0)
        arousal = mood.get("arousal", 0.5)
        stability = mood.get("stability", 0.8)
        
        # Very negative mood - avoid certain topics
        if valence < -0.7:
            constraints.append({
                "type": "topic_avoidance",
                "description": "Avoid overly cheerful or dismissive responses",
                "severity": "high"
            })
        
        # Very high arousal - avoid overload
        if arousal > 0.85:
            constraints.append({
                "type": "complexity_limit",
                "description": "Keep responses focused and clear",
                "severity": "medium"
            })
        
        # Low stability - be consistent
        if stability < 0.3:
            constraints.append({
                "type": "consistency_requirement",
                "description": "Maintain steady interaction style",
                "severity": "high"
            })
        
        return constraints
    
    async def _calculate_mood_confidence(self) -> float:
        """Calculate confidence in mood assessment"""
        confidence = 0.8  # Base confidence
        
        # More history increases confidence
        history_size = len(self.mood_influence_history)
        if history_size < 5:
            confidence -= 0.2
        elif history_size > 20:
            confidence += 0.1
        
        # Stable mood increases confidence
        stability = await self._calculate_mood_stability()
        if stability > 0.7:
            confidence += 0.1
        elif stability < 0.3:
            confidence -= 0.2
        
        # Recent strong influences decrease confidence
        if self.mood_influence_history:
            recent_influence = self.mood_influence_history[-1]["influence_strength"]
            if recent_influence > 0.5:
                confidence -= 0.15
        
        return max(0.3, min(1.0, confidence))
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original manager"""
        return getattr(self.original_manager, name)
