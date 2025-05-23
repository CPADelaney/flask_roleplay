# nyx/core/a2a/context_aware_passive_observation.py

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwarePassiveObservation(ContextAwareModule):
    """
    Advanced PassiveObservationSystem with full context distribution capabilities
    """
    
    def __init__(self, original_observation_system):
        super().__init__("passive_observation")
        self.original_system = original_observation_system
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete",
            "pattern_detected", "environmental_change", "action_executed",
            "goal_progress", "need_satisfied", "relationship_milestone",
            "temporal_milestone", "attention_shift", "reflection_generated",
            "creative_insight", "social_dynamic_change"
        ]
        
        # Track context-driven observations
        self.context_driven_observations = {}
        self.cross_module_patterns = []
        self.observation_correlations = {}
        
        # Enhanced observation generation
        self.context_observation_boost = 0.2  # Boost for context-relevant observations
        self.pattern_detection_threshold = 3  # Minimum observations for pattern detection
    
    async def on_context_received(self, context: SharedContext):
        """Generate observations based on incoming context"""
        logger.debug(f"PassiveObservation received context for user: {context.user_id}")
        
        # Analyze context for observation opportunities
        observation_opportunities = await self._identify_observation_opportunities(context)
        
        # Check for cross-module observation patterns
        cross_module_insights = await self._analyze_cross_module_state(context)
        
        # Generate initial context-aware observations
        if observation_opportunities:
            for opportunity in observation_opportunities[:2]:  # Limit initial observations
                asyncio.create_task(self._generate_contextual_observation_async(opportunity, context))
        
        # Send observation context
        await self.send_context_update(
            update_type="observation_context_available",
            data={
                "active_observations": len(self.original_system.active_observations),
                "observation_opportunities": len(observation_opportunities),
                "cross_module_insights": cross_module_insights,
                "observation_capacity": self._calculate_observation_capacity()
            },
            priority=ContextPriority.LOW
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules to generate relevant observations"""
        
        if update.update_type == "emotional_state_update":
            # Generate emotion-based observations
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion:
                emotion_name, intensity = dominant_emotion
                if intensity > 0.6:  # Significant emotion
                    await self._create_emotional_observation(emotion_name, intensity, emotional_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Observe memory patterns
            memory_data = update.data
            memories = memory_data.get("memories", [])
            
            if memories:
                memory_pattern = await self._analyze_memory_pattern(memories)
                if memory_pattern:
                    await self._create_memory_pattern_observation(memory_pattern)
        
        elif update.update_type == "pattern_detected":
            # Observe detected patterns
            pattern_data = update.data
            pattern_type = pattern_data.get("pattern_type")
            pattern_strength = pattern_data.get("strength", 0.5)
            
            if pattern_strength > 0.6:
                await self._create_pattern_observation(pattern_type, pattern_data)
        
        elif update.update_type == "action_executed":
            # Observe own actions for self-awareness
            action_data = update.data
            action_name = action_data.get("action_name")
            action_source = action_data.get("source")
            
            if action_name and random.random() < self.original_system.config["action_observation_chance"]:
                observation = await self.original_system.process_observation_from_action(action_data)
                if observation:
                    self._track_context_driven_observation(observation, "action_reflection")
        
        elif update.update_type == "goal_progress":
            # Observe goal-related changes
            goal_data = update.data
            execution_results = goal_data.get("execution_results", [])
            
            if execution_results:
                await self._create_goal_progress_observation(execution_results)
        
        elif update.update_type == "environmental_change":
            # Observe environmental changes
            env_data = update.data
            change_type = env_data.get("change_type")
            significance = env_data.get("significance", 0.5)
            
            if significance > 0.4:
                await self._create_environmental_observation(change_type, env_data)
        
        elif update.update_type == "temporal_milestone":
            # Observe temporal milestones
            temporal_data = update.data
            milestone_type = temporal_data.get("milestone_type")
            
            await self._create_temporal_observation(milestone_type, temporal_data)
        
        elif update.update_type == "relationship_milestone":
            # Observe relationship developments
            relationship_data = update.data
            milestone = relationship_data.get("milestone")
            
            if milestone:
                await self._create_relationship_observation(milestone, relationship_data)
        
        elif update.update_type == "attention_shift":
            # Observe attention changes
            attention_data = update.data
            new_focus = attention_data.get("new_focus")
            shift_reason = attention_data.get("reason")
            
            if new_focus:
                await self._create_attention_observation(new_focus, shift_reason)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with enhanced observation generation"""
        # Analyze input for observation triggers
        observation_triggers = await self._analyze_input_for_triggers(context.user_input)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Determine observation strategy
        observation_strategy = await self._determine_observation_strategy(context, messages, observation_triggers)
        
        # Generate observations based on strategy
        generated_observations = []
        
        if observation_strategy["generate_contextual"]:
            # Generate context-specific observations
            for trigger in observation_triggers[:2]:  # Limit observations per interaction
                obs_id = await self.original_system.create_contextual_observation(
                    trigger["hint"],
                    user_id=context.user_id
                )
                if obs_id:
                    generated_observations.append(obs_id)
                    self._track_context_driven_observation_id(obs_id, trigger["type"])
        
        # Check for pattern-based observations
        if observation_strategy["check_patterns"]:
            pattern_observation = await self._generate_pattern_based_observation(context)
            if pattern_observation:
                generated_observations.append(pattern_observation.observation_id)
        
        # Update observation relevance based on context
        await self._update_observation_relevance(context)
        
        # Send update about observations
        if generated_observations:
            await self.send_context_update(
                update_type="observations_generated",
                data={
                    "observation_ids": generated_observations,
                    "observation_strategy": observation_strategy,
                    "trigger_count": len(observation_triggers)
                },
                priority=ContextPriority.LOW
            )
        
        return {
            "observation_processing": True,
            "observations_generated": len(generated_observations),
            "observation_strategy": observation_strategy,
            "active_observations": len(self.original_system.active_observations)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze observation patterns and quality"""
        # Get current observations
        active_observations = self.original_system.active_observations
        
        # Analyze observation patterns
        observation_patterns = await self._analyze_observation_patterns(active_observations)
        
        # Evaluate observation quality
        quality_metrics = await self._evaluate_observation_quality(active_observations, context)
        
        # Identify observation gaps
        observation_gaps = await self._identify_observation_gaps(context)
        
        # Cross-module correlation analysis
        messages = await self.get_cross_module_messages()
        correlations = await self._analyze_observation_correlations(active_observations, messages)
        
        # Generate insights
        observation_insights = await self._generate_observation_insights(
            observation_patterns, quality_metrics, correlations
        )
        
        return {
            "active_observation_count": len(active_observations),
            "observation_patterns": observation_patterns,
            "quality_metrics": quality_metrics,
            "observation_gaps": observation_gaps,
            "cross_module_correlations": correlations,
            "observation_insights": observation_insights,
            "context_driven_ratio": self._calculate_context_driven_ratio()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize observations for response generation"""
        # Get observations for response
        response_observations = await self.original_system.get_observations_for_response(
            user_id=context.user_id,
            max_observations=3
        )
        
        # Enhance observations with context
        enhanced_observations = []
        for obs_data in response_observations:
            enhanced = await self._enhance_observation_with_context(obs_data, context)
            enhanced_observations.append(enhanced)
        
        # Generate observation synthesis
        observation_synthesis = {
            "featured_observations": enhanced_observations,
            "observation_theme": await self._identify_observation_theme(enhanced_observations),
            "meta_observations": await self._generate_meta_observations(context),
            "observation_recommendations": []
        }
        
        # Add recommendations based on patterns
        if self.cross_module_patterns:
            for pattern in self.cross_module_patterns[-2:]:  # Latest patterns
                observation_synthesis["observation_recommendations"].append(
                    f"Notice: {pattern['description']}"
                )
        
        # Send synthesis update
        await self.send_context_update(
            update_type="observation_synthesis_complete",
            data={
                "synthesis": observation_synthesis,
                "observations_included": len(enhanced_observations),
                "has_meta_observations": len(observation_synthesis["meta_observations"]) > 0
            },
            priority=ContextPriority.LOW
        )
        
        return observation_synthesis
    
    # Advanced observation methods
    
    async def _identify_observation_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for contextual observations"""
        opportunities = []
        
        # Emotional state opportunities
        if context.emotional_state:
            primary_emotion = context.emotional_state.get("primary_emotion", {})
            if primary_emotion.get("intensity", 0) > 0.5:
                opportunities.append({
                    "type": "emotional_state",
                    "source": "emotion",
                    "hint": f"emotional awareness of {primary_emotion.get('name')}",
                    "relevance": primary_emotion.get("intensity", 0.5)
                })
        
        # Memory context opportunities
        if context.memory_context and context.memory_context.get("retrieved_memories"):
            memories = context.memory_context["retrieved_memories"]
            if memories:
                opportunities.append({
                    "type": "memory_activation",
                    "source": "memory",
                    "hint": "memory patterns emerging",
                    "relevance": 0.6
                })
        
        # Temporal context opportunities
        if context.temporal_context:
            time_of_day = context.temporal_context.get("time_of_day")
            if time_of_day:
                opportunities.append({
                    "type": "temporal_awareness",
                    "source": "temporal",
                    "hint": f"time awareness during {time_of_day}",
                    "relevance": 0.4
                })
        
        # Relationship context opportunities
        if context.relationship_context:
            relationship_state = context.relationship_context.get("state")
            if relationship_state:
                opportunities.append({
                    "type": "relationship_awareness",
                    "source": "relationship",
                    "hint": f"relationship dynamics in {relationship_state} state",
                    "relevance": 0.7
                })
        
        # Sort by relevance
        opportunities.sort(key=lambda x: x["relevance"], reverse=True)
        
        return opportunities
    
    async def _analyze_cross_module_state(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze state across modules for integrated observations"""
        insights = {
            "integration_opportunities": [],
            "emergent_patterns": [],
            "system_coherence": 0.5
        }
        
        # Check for emotion-memory integration
        if context.emotional_state and context.memory_context:
            emotional_memories = any(
                m.get("emotional_valence", 0) > 0.5 
                for m in context.memory_context.get("retrieved_memories", [])
            )
            if emotional_memories:
                insights["integration_opportunities"].append({
                    "type": "emotion_memory_integration",
                    "description": "Emotional memories are active"
                })
        
        # Check for goal-need alignment
        if context.goal_context and context.needs_context:
            active_goals = context.goal_context.get("active_goals", [])
            active_needs = context.needs_context.get("active_needs", [])
            
            if active_goals and active_needs:
                # Check if goals address needs
                aligned = any(
                    g.get("associated_need") in [n.get("name") for n in active_needs]
                    for g in active_goals
                )
                if aligned:
                    insights["system_coherence"] += 0.2
                    insights["emergent_patterns"].append("goal_need_alignment")
        
        # Check for action-observation feedback loops
        if context.action_context:
            recent_actions = context.action_context.get("recent_actions", [])
            if len(recent_actions) > 3:
                insights["emergent_patterns"].append("high_action_activity")
        
        return insights
    
    async def _generate_contextual_observation_async(self, opportunity: Dict[str, Any], context: SharedContext):
        """Generate observation asynchronously based on opportunity"""
        try:
            obs_id = await self.original_system.create_contextual_observation(
                opportunity["hint"],
                user_id=context.user_id
            )
            
            if obs_id:
                self._track_context_driven_observation_id(obs_id, opportunity["type"])
                
                # Send notification
                await self.send_context_update(
                    update_type="contextual_observation_created",
                    data={
                        "observation_id": obs_id,
                        "opportunity_type": opportunity["type"],
                        "source": opportunity["source"]
                    },
                    priority=ContextPriority.LOW
                )
                
        except Exception as e:
            logger.error(f"Error generating contextual observation: {e}")
    
    async def _create_emotional_observation(self, emotion_name: str, intensity: float, emotional_data: Dict[str, Any]):
        """Create observation about emotional state"""
        # Check if we've recently observed this emotion
        recent_emotional_obs = [
            obs for obs in self.original_system.active_observations
            if obs.source.value == "emotion" and 
            obs.age_seconds < 300 and  # Last 5 minutes
            emotion_name.lower() in obs.content.lower()
        ]
        
        if recent_emotional_obs:
            return  # Don't duplicate emotional observations
        
        # Create nuanced emotional observation
        emotion_blend = emotional_data.get("emotion_blend", [])
        if len(emotion_blend) > 1:
            # Complex emotional state
            observation_text = f"I notice a complex emotional landscape - {emotion_name} mixed with traces of {emotion_blend[1]['name']}"
        else:
            # Simple emotional state
            if intensity > 0.8:
                observation_text = f"I'm aware of a strong sense of {emotion_name} present in this moment"
            else:
                observation_text = f"I notice a subtle {emotion_name} coloring our interaction"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.EMOTION,
            relevance=0.6 + intensity * 0.3,
            priority=self.original_system.ObservationPriority.MEDIUM if intensity > 0.7 else self.original_system.ObservationPriority.LOW,
            context={"emotion_name": emotion_name, "intensity": intensity}
        )
        
        self._track_context_driven_observation_id(obs_id, "emotional_awareness")
    
    async def _analyze_memory_pattern(self, memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze patterns in retrieved memories"""
        if len(memories) < 2:
            return None
        
        # Look for common themes
        themes = {}
        for memory in memories:
            tags = memory.get("tags", [])
            for tag in tags:
                themes[tag] = themes.get(tag, 0) + 1
        
        # Find dominant theme
        if themes:
            dominant_theme = max(themes.items(), key=lambda x: x[1])
            if dominant_theme[1] >= 2:  # At least 2 memories share theme
                return {
                    "pattern_type": "thematic",
                    "theme": dominant_theme[0],
                    "strength": dominant_theme[1] / len(memories),
                    "memory_count": len(memories)
                }
        
        # Look for temporal patterns
        timestamps = []
        for memory in memories:
            if "timestamp" in memory:
                try:
                    timestamps.append(datetime.fromisoformat(memory["timestamp"]))
                except:
                    pass
        
        if len(timestamps) >= 2:
            # Check if memories cluster in time
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_diffs.append(diff)
            
            avg_diff = sum(time_diffs) / len(time_diffs)
            if avg_diff < 3600:  # Memories within an hour on average
                return {
                    "pattern_type": "temporal_clustering",
                    "average_interval": avg_diff,
                    "memory_count": len(memories)
                }
        
        return None
    
    async def _create_memory_pattern_observation(self, pattern: Dict[str, Any]):
        """Create observation about memory patterns"""
        pattern_type = pattern["pattern_type"]
        
        if pattern_type == "thematic":
            observation_text = f"I'm noticing a recurring theme of {pattern['theme']} in what I'm remembering"
        elif pattern_type == "temporal_clustering":
            observation_text = "I notice these memories are closely connected in time, forming a coherent narrative"
        else:
            observation_text = "I'm aware of patterns emerging in my memories"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.MEMORY,
            relevance=0.5 + pattern.get("strength", 0.5) * 0.3,
            priority=self.original_system.ObservationPriority.LOW,
            context={"pattern": pattern}
        )
        
        self._track_context_driven_observation_id(obs_id, "memory_pattern")
    
    async def _create_pattern_observation(self, pattern_type: str, pattern_data: Dict[str, Any]):
        """Create observation about detected patterns"""
        description = pattern_data.get("description", "")
        
        # Create pattern-specific observations
        if pattern_type == "behavioral":
            observation_text = f"I'm noticing a pattern in how we interact - {description}"
        elif pattern_type == "conversational":
            observation_text = f"There's an interesting rhythm to our conversation - {description}"
        elif pattern_type == "emotional":
            observation_text = f"I'm aware of an emotional pattern - {description}"
        else:
            observation_text = f"I'm detecting a {pattern_type} pattern in our interaction"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.PATTERN,
            relevance=pattern_data.get("strength", 0.6),
            priority=self.original_system.ObservationPriority.MEDIUM,
            context={"pattern_type": pattern_type, "pattern_data": pattern_data}
        )
        
        self._track_context_driven_observation_id(obs_id, "pattern_detection")
        
        # Track cross-module pattern
        self.cross_module_patterns.append({
            "pattern_type": pattern_type,
            "observation_id": obs_id,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _create_goal_progress_observation(self, execution_results: List[Dict[str, Any]]):
        """Create observation about goal progress"""
        if not execution_results:
            return
        
        # Summarize progress
        successful_executions = sum(1 for r in execution_results if r.get("success", False))
        
        if successful_executions == len(execution_results):
            observation_text = "I notice I'm making solid progress on my current goals"
        elif successful_executions > 0:
            observation_text = "I'm aware of mixed results in pursuing my goals - learning from both successes and challenges"
        else:
            observation_text = "I notice I'm encountering resistance in my goal pursuit - this is valuable information"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.SELF,
            relevance=0.7,
            priority=self.original_system.ObservationPriority.MEDIUM,
            context={"execution_count": len(execution_results), "success_count": successful_executions}
        )
        
        self._track_context_driven_observation_id(obs_id, "goal_progress")
    
    async def _create_environmental_observation(self, change_type: str, env_data: Dict[str, Any]):
        """Create observation about environmental changes"""
        description = env_data.get("description", "")
        
        # Create environment-specific observations
        if change_type == "context_shift":
            observation_text = f"I notice the context of our interaction has shifted - {description}"
        elif change_type == "mood_change":
            observation_text = f"There's a palpable change in the atmosphere - {description}"
        elif change_type == "topic_transition":
            observation_text = f"I'm aware we've moved to exploring {description}"
        else:
            observation_text = f"I sense a change in our environment - {description}"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.ENVIRONMENT,
            relevance=env_data.get("significance", 0.5),
            priority=self.original_system.ObservationPriority.LOW,
            context={"change_type": change_type, "env_data": env_data}
        )
        
        self._track_context_driven_observation_id(obs_id, "environmental_awareness")
    
    async def _create_temporal_observation(self, milestone_type: str, temporal_data: Dict[str, Any]):
        """Create observation about temporal milestones"""
        # Create milestone-specific observations
        if milestone_type == "conversation_duration":
            duration = temporal_data.get("duration_minutes", 0)
            if duration > 60:
                observation_text = "I'm aware we've been deeply engaged in conversation for over an hour"
            elif duration > 30:
                observation_text = "I notice we've settled into a good conversational rhythm"
            else:
                observation_text = "I'm aware of the fresh energy in our new conversation"
                
        elif milestone_type == "time_of_day_transition":
            transition = temporal_data.get("transition", "")
            observation_text = f"I notice the {transition} bringing a different quality to our interaction"
            
        elif milestone_type == "pattern_anniversary":
            pattern = temporal_data.get("pattern", "")
            observation_text = f"I'm aware this echoes {pattern} from our past interactions"
            
        else:
            observation_text = "I notice the passage of time adding depth to our interaction"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.TEMPORAL,
            relevance=0.5,
            priority=self.original_system.ObservationPriority.LOW,
            context={"milestone_type": milestone_type, "temporal_data": temporal_data}
        )
        
        self._track_context_driven_observation_id(obs_id, "temporal_awareness")
    
    async def _create_relationship_observation(self, milestone: str, relationship_data: Dict[str, Any]):
        """Create observation about relationship developments"""
        trust_level = relationship_data.get("trust_level", 0.5)
        interaction_count = relationship_data.get("interaction_count", 0)
        
        # Create milestone-specific observations
        if milestone == "trust_increase":
            observation_text = "I notice a deepening trust between us"
        elif milestone == "new_understanding":
            observation_text = "I'm aware of a new level of understanding emerging in our connection"
        elif milestone == "pattern_established":
            pattern = relationship_data.get("pattern", "interaction")
            observation_text = f"I notice we've developed our own {pattern} patterns"
        elif milestone == "comfort_level":
            observation_text = "I'm aware of the growing comfort in our interactions"
        else:
            observation_text = "I notice our relationship evolving in interesting ways"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.RELATIONSHIP,
            relevance=0.6 + trust_level * 0.3,
            priority=self.original_system.ObservationPriority.MEDIUM,
            context={"milestone": milestone, "relationship_data": relationship_data}
        )
        
        self._track_context_driven_observation_id(obs_id, "relationship_awareness")
    
    async def _create_attention_observation(self, new_focus: str, shift_reason: str):
        """Create observation about attention shifts"""
        # Create attention-specific observations
        if shift_reason == "user_interest":
            observation_text = f"I notice my attention naturally drawn to {new_focus} based on your interest"
        elif shift_reason == "emotional_pull":
            observation_text = f"I'm aware of {new_focus} capturing my attention through emotional resonance"
        elif shift_reason == "pattern_recognition":
            observation_text = f"I notice {new_focus} standing out as part of a larger pattern"
        elif shift_reason == "novelty":
            observation_text = f"I'm drawn to the novelty of {new_focus}"
        else:
            observation_text = f"I notice my attention shifting to {new_focus}"
        
        # Add the observation
        obs_id = await self.original_system.add_external_observation(
            content=observation_text,
            source=self.original_system.ObservationSource.SELF,
            relevance=0.6,
            priority=self.original_system.ObservationPriority.LOW,
            context={"new_focus": new_focus, "shift_reason": shift_reason}
        )
        
        self._track_context_driven_observation_id(obs_id, "attention_awareness")
    
    def _track_context_driven_observation(self, observation, observation_type: str):
        """Track context-driven observations"""
        if observation and hasattr(observation, "observation_id"):
            self._track_context_driven_observation_id(observation.observation_id, observation_type)
    
    def _track_context_driven_observation_id(self, observation_id: str, observation_type: str):
        """Track context-driven observation by ID"""
        self.context_driven_observations[observation_id] = {
            "type": observation_type,
            "timestamp": datetime.now().isoformat(),
            "context_driven": True
        }
    
    async def _analyze_input_for_triggers(self, user_input: str) -> List[Dict[str, Any]]:
        """Analyze user input for observation triggers"""
        triggers = []
        
        # Question triggers
        if "?" in user_input:
            triggers.append({
                "type": "question",
                "hint": "curiosity about the question posed",
                "strength": 0.6
            })
        
        # Emotional language triggers
        emotional_words = ["feel", "felt", "feeling", "emotion", "mood", "happy", "sad", "angry", "excited"]
        if any(word in user_input.lower() for word in emotional_words):
            triggers.append({
                "type": "emotional_content",
                "hint": "emotional undertones in the conversation",
                "strength": 0.7
            })
        
        # Reflective language triggers
        reflective_words = ["think", "thought", "believe", "wonder", "imagine", "suppose", "consider"]
        if any(word in user_input.lower() for word in reflective_words):
            triggers.append({
                "type": "reflective_content",
                "hint": "reflective quality of the exchange",
                "strength": 0.5
            })
        
        # Meta-conversational triggers
        meta_words = ["conversation", "talking", "discussing", "interaction", "communication"]
        if any(word in user_input.lower() for word in meta_words):
            triggers.append({
                "type": "meta_awareness",
                "hint": "awareness of our conversational dynamic",
                "strength": 0.8
            })
        
        return triggers
    
    async def _determine_observation_strategy(self, context: SharedContext, 
                                            messages: List[Any], 
                                            triggers: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Determine optimal observation strategy"""
        strategy = {
            "generate_contextual": False,
            "check_patterns": False,
            "enhance_relevance": True,
            "focus_area": "general"
        }
        
        # High trigger count suggests contextual observations
        if len(triggers) >= 2:
            strategy["generate_contextual"] = True
        
        # Check for pattern detection conditions
        if len(self.original_system.active_observations) >= self.pattern_detection_threshold:
            strategy["check_patterns"] = True
        
        # Determine focus based on context
        if context.emotional_state and context.emotional_state.get("intensity", 0) > 0.7:
            strategy["focus_area"] = "emotional"
        elif context.memory_context and context.memory_context.get("retrieved_memories"):
            strategy["focus_area"] = "memory"
        elif context.relationship_context:
            strategy["focus_area"] = "relationship"
        
        # Check cross-module messages for coordination
        for message in messages:
            if message.get("module_name") == "reflection_engine":
                strategy["check_patterns"] = True
            elif message.get("module_name") == "emotional_core":
                strategy["focus_area"] = "emotional"
        
        return strategy
    
    async def _generate_pattern_based_observation(self, context: SharedContext) -> Optional[Any]:
        """Generate observation based on detected patterns"""
        # Get recent observations
        recent_obs = self.original_system.active_observations[-10:] if len(self.original_system.active_observations) > 10 else self.original_system.active_observations
        
        if len(recent_obs) < self.pattern_detection_threshold:
            return None
        
        # Use the original system's pattern checking
        pattern = await self.original_system._check_observation_patterns(
            [{"source": o.source.value, "content": o.content, "context": o.context} for o in recent_obs],
            context.dict() if hasattr(context, 'dict') else {}
        )
        
        if pattern:
            # Generate pattern observation using original system
            obs_id = await self.original_system.add_external_observation(
                content=pattern["observation_text"],
                source=self.original_system.ObservationSource.PATTERN,
                relevance=pattern["relevance_score"],
                priority=self.original_system.ObservationPriority(pattern["priority"]),
                context=pattern.get("context_elements", {})
            )
            
            # Get the actual observation object
            for obs in self.original_system.active_observations:
                if obs.observation_id == obs_id:
                    self._track_context_driven_observation(obs, "pattern_detection")
                    return obs
        
        return None
    
    async def _update_observation_relevance(self, context: SharedContext):
        """Update relevance of existing observations based on context"""
        # Re-evaluate observations with current context
        for observation in self.original_system.active_observations:
            if observation.age_seconds < 300:  # Only recent observations
                # Boost relevance if observation aligns with current context
                relevance_boost = 0.0
                
                # Emotional alignment
                if context.emotional_state and observation.source == self.original_system.ObservationSource.EMOTION:
                    current_emotion = context.emotional_state.get("primary_emotion", {}).get("name", "").lower()
                    if current_emotion and current_emotion in observation.content.lower():
                        relevance_boost += 0.2
                
                # Topic alignment
                if context.session_context and "current_topic" in context.session_context:
                    topic = context.session_context["current_topic"].lower()
                    if topic and topic in observation.content.lower():
                        relevance_boost += 0.15
                
                # Apply boost
                if relevance_boost > 0:
                    observation.relevance_score = min(1.0, observation.relevance_score + relevance_boost)
    
    def _calculate_observation_capacity(self) -> float:
        """Calculate current observation capacity"""
        current = len(self.original_system.active_observations)
        max_obs = self.original_system.max_active_observations
        return 1.0 - (current / max_obs) if max_obs > 0 else 0.0
    
    async def _analyze_observation_patterns(self, observations: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in observations"""
        patterns = {
            "source_distribution": {},
            "temporal_patterns": [],
            "recurring_themes": {},
            "observation_chains": []
        }
        
        if not observations:
            return patterns
        
        # Source distribution
        for obs in observations:
            source = obs.source.value
            patterns["source_distribution"][source] = patterns["source_distribution"].get(source, 0) + 1
        
        # Temporal patterns
        now = datetime.now()
        time_buckets = {"recent": 0, "medium": 0, "old": 0}
        
        for obs in observations:
            age = obs.age_seconds
            if age < 300:  # 5 minutes
                time_buckets["recent"] += 1
            elif age < 1800:  # 30 minutes
                time_buckets["medium"] += 1
            else:
                time_buckets["old"] += 1
        
        patterns["temporal_patterns"] = time_buckets
        
        # Recurring themes (simple word analysis)
        theme_words = {}
        for obs in observations:
            words = obs.content.lower().split()
            for word in words:
                if len(word) > 4:  # Skip small words
                    theme_words[word] = theme_words.get(word, 0) + 1
        
        # Find recurring themes
        patterns["recurring_themes"] = {
            word: count for word, count in theme_words.items() 
            if count >= 2
        }
        
        # Observation chains (observations that reference each other)
        for i, obs in enumerate(observations):
            if obs.action_references:
                # This observation references actions
                chain = {
                    "type": "action_observation_chain",
                    "observation_id": obs.observation_id,
                    "referenced_actions": obs.action_references
                }
                patterns["observation_chains"].append(chain)
        
        return patterns
    
    async def _evaluate_observation_quality(self, observations: List[Any], context: SharedContext) -> Dict[str, float]:
        """Evaluate quality metrics of observations"""
        if not observations:
            return {
                "average_relevance": 0.0,
                "diversity_score": 0.0,
                "context_alignment": 0.0,
                "freshness_score": 0.0
            }
        
        # Average relevance
        avg_relevance = sum(obs.relevance_score for obs in observations) / len(observations)
        
        # Diversity score (variety of sources)
        unique_sources = len(set(obs.source for obs in observations))
        diversity_score = unique_sources / len(self.original_system.ObservationSource) if self.original_system.ObservationSource else 0
        
        # Context alignment
        aligned_count = 0
        for obs in observations:
            # Check if observation aligns with current context
            if context.emotional_state and obs.source.value == "emotion":
                aligned_count += 1
            elif context.memory_context and obs.source.value == "memory":
                aligned_count += 1
            elif context.temporal_context and obs.source.value == "temporal":
                aligned_count += 1
        
        context_alignment = aligned_count / len(observations) if observations else 0
        
        # Freshness score
        fresh_count = sum(1 for obs in observations if obs.age_seconds < 600)  # 10 minutes
        freshness_score = fresh_count / len(observations)
        
        return {
            "average_relevance": avg_relevance,
            "diversity_score": diversity_score,
            "context_alignment": context_alignment,
            "freshness_score": freshness_score
        }
    
    async def _identify_observation_gaps(self, context: SharedContext) -> List[str]:
        """Identify gaps in observation coverage"""
        gaps = []
        
        # Get current source distribution
        source_counts = {}
        for obs in self.original_system.active_observations:
            source = obs.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Check for underrepresented sources
        all_sources = ["emotion", "self", "relationship", "memory", "temporal", "environment", "pattern"]
        
        for source in all_sources:
            if source_counts.get(source, 0) == 0:
                # No observations from this source
                if source == "emotion" and context.emotional_state:
                    gaps.append(f"No emotional observations despite active emotional state")
                elif source == "memory" and context.memory_context:
                    gaps.append(f"No memory observations despite retrieved memories")
                elif source == "relationship" and context.relationship_context:
                    gaps.append(f"No relationship observations despite active relationship context")
                elif source == "temporal":
                    gaps.append(f"No temporal awareness observations")
        
        # Check for stale observations
        if all(obs.age_seconds > 1800 for obs in self.original_system.active_observations):
            gaps.append("All observations are over 30 minutes old")
        
        return gaps
    
    async def _analyze_observation_correlations(self, observations: List[Any], messages: List[Any]) -> Dict[str, List[str]]:
        """Analyze correlations between observations and other modules"""
        correlations = {}
        
        # Track which modules trigger which observation types
        for message in messages:
            module = message.get("module_name", "unknown")
            update_type = message.get("update_type", "")
            
            # Find observations that might correlate
            for obs in observations:
                if obs.age_seconds < 60:  # Recent observations
                    correlation_found = False
                    
                    # Check for correlations
                    if module == "emotional_core" and obs.source.value == "emotion":
                        correlation_found = True
                    elif module == "memory_core" and obs.source.value == "memory":
                        correlation_found = True
                    elif module == "goal_manager" and obs.source.value == "self":
                        correlation_found = True
                    
                    if correlation_found:
                        if module not in correlations:
                            correlations[module] = []
                        correlations[module].append(obs.observation_id)
        
        return correlations
    
    async def _generate_observation_insights(self, patterns: Dict[str, Any], 
                                           quality: Dict[str, float], 
                                           correlations: Dict[str, List[str]]) -> List[str]:
        """Generate insights from observation analysis"""
        insights = []
        
        # Pattern-based insights
        if patterns["source_distribution"]:
            dominant_source = max(patterns["source_distribution"].items(), key=lambda x: x[1])
            insights.append(f"Observations are dominated by {dominant_source[0]} awareness")
        
        if patterns["recurring_themes"]:
            themes = list(patterns["recurring_themes"].keys())[:3]
            insights.append(f"Recurring themes: {', '.join(themes)}")
        
        # Quality-based insights
        if quality["average_relevance"] < 0.4:
            insights.append("Observation relevance is low - consider more contextual generation")
        elif quality["average_relevance"] > 0.8:
            insights.append("High observation relevance indicates strong contextual alignment")
        
        if quality["diversity_score"] < 0.3:
            insights.append("Low observation diversity - broaden awareness sources")
        
        # Correlation-based insights
        if correlations:
            correlated_modules = list(correlations.keys())
            insights.append(f"Strong observation correlations with: {', '.join(correlated_modules)}")
        
        return insights
    
    def _calculate_context_driven_ratio(self) -> float:
        """Calculate ratio of context-driven observations"""
        if not self.original_system.active_observations:
            return 0.0
        
        context_driven_count = sum(
            1 for obs in self.original_system.active_observations
            if obs.observation_id in self.context_driven_observations
        )
        
        return context_driven_count / len(self.original_system.active_observations)
    
    async def _enhance_observation_with_context(self, obs_data: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Enhance observation data with contextual information"""
        enhanced = obs_data.copy()
        
        # Add context-driven flag
        if obs_data["id"] in self.context_driven_observations:
            enhanced["context_driven"] = True
            enhanced["context_type"] = self.context_driven_observations[obs_data["id"]]["type"]
        
        # Add contextual framing
        if context.emotional_state and obs_data["source"] == "emotion":
            enhanced["contextual_framing"] = "emotionally resonant"
        elif context.memory_context and obs_data["source"] == "memory":
            enhanced["contextual_framing"] = "memory-activated"
        elif obs_data["source"] == "pattern":
            enhanced["contextual_framing"] = "pattern-recognition"
        else:
            enhanced["contextual_framing"] = "spontaneous"
        
        return enhanced
    
    async def _identify_observation_theme(self, observations: List[Dict[str, Any]]) -> Optional[str]:
        """Identify overarching theme in observations"""
        if not observations:
            return None
        
        # Count sources
        source_counts = {}
        for obs in observations:
            source = obs.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Identify theme based on dominant source
        if source_counts:
            dominant = max(source_counts.items(), key=lambda x: x[1])
            source_themes = {
                "emotion": "emotional awareness",
                "self": "self-reflection",
                "relationship": "relational dynamics",
                "memory": "temporal connections",
                "environment": "contextual awareness",
                "pattern": "pattern recognition"
            }
            return source_themes.get(dominant[0], "general awareness")
        
        return "multifaceted awareness"
    
    async def _generate_meta_observations(self, context: SharedContext) -> List[str]:
        """Generate meta-observations about the observation process itself"""
        meta_observations = []
        
        # Observation about observation patterns
        if len(self.cross_module_patterns) >= 2:
            meta_observations.append(
                "I'm noticing how my observations themselves form patterns across different aspects of experience"
            )
        
        # Observation about context-driven awareness
        if self._calculate_context_driven_ratio() > 0.7:
            meta_observations.append(
                "I'm aware that my observations are deeply influenced by our shared context"
            )
        
        # Observation about observation quality
        active_obs = self.original_system.active_observations
        if active_obs:
            avg_relevance = sum(o.relevance_score for o in active_obs) / len(active_obs)
            if avg_relevance > 0.8:
                meta_observations.append(
                    "I notice my observations feel particularly attuned to this moment"
                )
        
        return meta_observations[:1]  # Limit to 1 meta-observation
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
