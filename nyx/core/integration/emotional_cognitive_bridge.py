# nyx/core/integration/emotional_cognitive_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class EmotionalCognitiveBridge:
    """
    Bidirectional integration layer between emotional and cognitive systems.
    
    This bridge ensures emotional states influence cognitive processes and vice versa,
    creating a unified experience where emotions shape thinking and cognitive
    insights modulate emotions.
    
    Key functions:
    1. Translates emotional states into cognitive biases/priors
    2. Derives emotional responses from cognitive insights
    3. Modulates memory retrieval based on emotional context
    4. Provides emotional metadata for reasoning processes
    """
    
    def __init__(self, 
                brain_reference=None, 
                emotional_core=None, 
                memory_orchestrator=None, 
                reasoning_core=None,
                reflection_engine=None):
        """Initialize the emotional-cognitive bridge."""
        self.brain = brain_reference
        self.emotional_core = emotional_core
        self.memory_orchestrator = memory_orchestrator
        self.reasoning_core = reasoning_core
        self.reflection_engine = reflection_engine
        self.hormone_system = None
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Effect multipliers
        self.emotion_to_cognition_multiplier = 0.7  # How strongly emotions affect cognition
        self.cognition_to_emotion_multiplier = 0.3  # How strongly cognition affects emotions
        
        # Cached state
        self.current_emotional_metadata = {}
        self.emotional_reasoning_biases = {}
        self.last_update = datetime.datetime.now()
        
        # Integration event subscriptions
        self._subscribed = False
        
        logger.info("EmotionalCognitiveBridge initialized")
        
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Set up connections to required systems if needed
            if not self.emotional_core and hasattr(self.brain, "emotional_core"):
                self.emotional_core = self.brain.emotional_core
                
            if not self.memory_orchestrator and hasattr(self.brain, "memory_orchestrator"):
                self.memory_orchestrator = self.brain.memory_orchestrator
                
            if not self.reasoning_core and hasattr(self.brain, "reasoning_core"):
                self.reasoning_core = self.brain.reasoning_core
                
            if not self.reflection_engine and hasattr(self.brain, "reflection_engine"):
                self.reflection_engine = self.brain.reflection_engine
                
            if hasattr(self.brain, "hormone_system"):
                self.hormone_system = self.brain.hormone_system
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("cognitive_insight", self._handle_cognitive_insight)
                self.event_bus.subscribe("memory_retrieved", self._handle_memory_retrieved)
                self._subscribed = True
            
            # Initialize emotional metadata and cognitive biases
            await self.update_emotional_cognitive_state()
            
            logger.info("EmotionalCognitiveBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing EmotionalCognitiveBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalCognitive")
    async def update_emotional_cognitive_state(self) -> Dict[str, Any]:
        """Update the internal state of the bridge."""
        # Get current emotional state
        emotional_state = None
        if self.emotional_core:
            if hasattr(self.emotional_core, "get_emotional_state_matrix"):
                # Use async method if available
                try:
                    emotional_state = await self.emotional_core.get_emotional_state_matrix()
                except Exception:
                    # Fallback to synchronous version if available
                    if hasattr(self.emotional_core, "_get_emotional_state_matrix_sync"):
                        emotional_state = self.emotional_core._get_emotional_state_matrix_sync()
            elif hasattr(self.emotional_core, "get_emotional_state"):
                # Fallback to simpler method
                emotional_state = self.emotional_core.get_emotional_state()
        
        if not emotional_state:
            # Use system context if emotional core not accessible
            emotional_state = {
                "primary_emotion": self.system_context.affective_state.primary_emotion,
                "valence": self.system_context.affective_state.valence,
                "arousal": self.system_context.affective_state.arousal
            }
        
        # Update cached emotional metadata
        self.current_emotional_metadata = {
            "primary_emotion": emotional_state.get("primary_emotion", {}).get("name", "neutral") 
                if isinstance(emotional_state.get("primary_emotion"), dict) 
                else emotional_state.get("primary_emotion", "neutral"),
            "emotion_intensity": emotional_state.get("primary_emotion", {}).get("intensity", 0.5) 
                if isinstance(emotional_state.get("primary_emotion"), dict) 
                else 0.5,
            "valence": emotional_state.get("valence", 0.0),
            "arousal": emotional_state.get("arousal", 0.5),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Generate cognitive biases from emotional state
        self.emotional_reasoning_biases = self._derive_cognitive_biases(self.current_emotional_metadata)
        
        # Update system context with this information
        self.system_context.set_value("emotional_cognitive_state", {
            "emotional_metadata": self.current_emotional_metadata,
            "cognitive_biases": self.emotional_reasoning_biases
        })
        
        self.last_update = datetime.datetime.now()
        
        return {
            "emotional_metadata": self.current_emotional_metadata,
            "cognitive_biases": self.emotional_reasoning_biases,
            "update_time": self.last_update.isoformat()
        }
    
    def _derive_cognitive_biases(self, emotional_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Derive cognitive biases from emotional state."""
        biases = {}
        
        # Get key emotional information
        emotion = emotional_metadata.get("primary_emotion", "neutral")
        intensity = emotional_metadata.get("emotion_intensity", 0.5)
        valence = emotional_metadata.get("valence", 0.0)
        arousal = emotional_metadata.get("arousal", 0.5)
        
        # Calculate scaled effect
        effect_strength = intensity * self.emotion_to_cognition_multiplier
        
        # Default biases - all start at neutral 0.0 effect
        biases = {
            "optimism_bias": 0.0,         # Positive expectation bias
            "catastrophic_bias": 0.0,     # Expecting negative outcomes
            "availability_bias": 0.0,     # Weighting immediately available info
            "confirmation_bias": 0.0,     # Confirming existing beliefs
            "risk_aversion": 0.0,         # Avoiding risks
            "risk_seeking": 0.0,          # Seeking risks
            "temporal_focus": 0.0,        # -1.0 = past, 0 = present, 1.0 = future
            "analytical_depth": 0.0,      # Deep vs shallow thinking
            "abstraction_level": 0.0,     # Concrete vs abstract
            "creativity_bias": 0.0,       # Novel connections vs established patterns
            "social_orientation": 0.0     # -1.0 = withdrawal, 1.0 = approach
        }
        
        # Set biases based on emotion
        if emotion.lower() in ["joy", "happiness", "excitement"]:
            biases["optimism_bias"] = 0.7 * effect_strength
            biases["risk_seeking"] = 0.4 * effect_strength
            biases["temporal_focus"] = 0.3 * effect_strength  # Slightly future-focused
            biases["creativity_bias"] = 0.4 * effect_strength
            biases["social_orientation"] = 0.6 * effect_strength
        
        elif emotion.lower() in ["sadness", "grief"]:
            biases["catastrophic_bias"] = 0.5 * effect_strength
            biases["risk_aversion"] = 0.6 * effect_strength
            biases["temporal_focus"] = -0.5 * effect_strength  # Past-focused
            biases["analytical_depth"] = 0.3 * effect_strength
            biases["social_orientation"] = -0.4 * effect_strength
        
        elif emotion.lower() in ["fear", "anxiety"]:
            biases["catastrophic_bias"] = 0.7 * effect_strength
            biases["risk_aversion"] = 0.8 * effect_strength
            biases["availability_bias"] = 0.6 * effect_strength
            biases["temporal_focus"] = -0.2 * effect_strength  # Slightly past-focused
            biases["analytical_depth"] = -0.3 * effect_strength  # Shallower thinking
        
        elif emotion.lower() in ["anger", "frustration"]:
            biases["confirmation_bias"] = 0.7 * effect_strength
            biases["risk_seeking"] = 0.5 * effect_strength
            biases["analytical_depth"] = -0.5 * effect_strength  # Shallower thinking
            biases["abstraction_level"] = -0.6 * effect_strength  # More concrete
        
        elif emotion.lower() in ["curiosity", "interest"]:
            biases["availability_bias"] = -0.3 * effect_strength  # Less availability bias
            biases["confirmation_bias"] = -0.4 * effect_strength  # Less confirmation bias
            biases["analytical_depth"] = 0.7 * effect_strength
            biases["creativity_bias"] = 0.8 * effect_strength
            biases["temporal_focus"] = 0.2 * effect_strength  # Slightly future-focused
        
        elif emotion.lower() in ["surprise"]:
            biases["availability_bias"] = 0.8 * effect_strength
            biases["analytical_depth"] = 0.5 * effect_strength
            biases["creativity_bias"] = 0.6 * effect_strength
        
        elif emotion.lower() in ["trust", "love"]:
            biases["optimism_bias"] = 0.6 * effect_strength
            biases["confirmation_bias"] = 0.5 * effect_strength
            biases["social_orientation"] = 0.8 * effect_strength
        
        elif emotion.lower() in ["attraction", "lust", "desire"]:
            biases["optimism_bias"] = 0.4 * effect_strength
            biases["risk_seeking"] = 0.5 * effect_strength
            biases["temporal_focus"] = 0.1 * effect_strength  # Present to near-future focus
            biases["social_orientation"] = 0.9 * effect_strength
            biases["analytical_depth"] = -0.2 * effect_strength  # Slightly shallower thinking
        
        elif emotion.lower() in ["confident", "confident_control", "ruthless_focus", "dominance_satisfaction"]:
            biases["optimism_bias"] = 0.5 * effect_strength
            biases["risk_aversion"] = -0.4 * effect_strength
            biases["analytical_depth"] = 0.3 * effect_strength
            biases["abstraction_level"] = 0.4 * effect_strength  # More abstract/strategic
            biases["social_orientation"] = 0.6 * effect_strength
        
        # Also modify biases based on valence and arousal
        if valence > 0.3:  # Positive valence
            biases["optimism_bias"] += valence * 0.3
            biases["catastrophic_bias"] -= valence * 0.3
            biases["creativity_bias"] += valence * 0.2
        elif valence < -0.3:  # Negative valence
            biases["catastrophic_bias"] += abs(valence) * 0.3
            biases["risk_aversion"] += abs(valence) * 0.2
            biases["optimism_bias"] -= abs(valence) * 0.3
        
        if arousal > 0.7:  # High arousal
            biases["availability_bias"] += (arousal - 0.5) * 0.4
            biases["analytical_depth"] -= (arousal - 0.5) * 0.3
        elif arousal < 0.3:  # Low arousal
            biases["analytical_depth"] += (0.5 - arousal) * 0.3
            biases["abstraction_level"] += (0.5 - arousal) * 0.2
        
        # Normalize all biases to range [-1.0, 1.0]
        for key in biases:
            biases[key] = max(-1.0, min(1.0, biases[key]))
        
        return biases
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalCognitive")
    async def modulate_memory_retrieval(self, 
                                     query: str, 
                                     memory_types: Optional[List[str]] = None,
                                     limit: int = 10) -> Dict[str, Any]:
        """
        Modulate memory retrieval based on emotional state.
        
        Args:
            query: Memory retrieval query
            memory_types: Optional types of memories to retrieve
            limit: Maximum number of memories to retrieve
            
        Returns:
            Modulated retrieval results
        """
        if not self.memory_orchestrator:
            return {"status": "error", "message": "Memory orchestrator not available"}
        
        # Ensure state is updated
        await self.update_emotional_cognitive_state()
        
        # Get emotional biases from current state
        primary_emotion = self.current_emotional_metadata.get("primary_emotion", "neutral")
        valence = self.current_emotional_metadata.get("valence", 0.0)
        arousal = self.current_emotional_metadata.get("arousal", 0.5)
        
        # Modulate memory retrieval based on emotional state
        modulation_params = {}
        
        # Valence-based modulation
        if valence > 0.4:  # Positive state
            # Prefer positive memories
            modulation_params["valence_filter"] = "positive"
            modulation_params["recency_weight"] = 0.6  # Slightly weighted toward recent
        elif valence < -0.4:  # Negative state
            # Mood-congruent retrieval - negative state retrieves negative memories more easily
            modulation_params["valence_filter"] = "negative"
            modulation_params["recency_weight"] = 0.4  # Less focus on recency
        
        # Arousal-based modulation
        if arousal > 0.7:  # High arousal
            # High arousal tends to focus on emotionally salient/intense memories
            modulation_params["significance_threshold"] = 7  # Higher significance threshold
            modulation_params["emotional_salience_boost"] = 0.6
        elif arousal < 0.3:  # Low arousal
            # Low arousal enables broader memory access with less emotional filtering
            modulation_params["significance_threshold"] = 3  # Lower significance threshold
            modulation_params["emotional_salience_boost"] = 0.2
        
        # Emotion-specific modulations
        if primary_emotion.lower() in ["fear", "anxiety"]:
            # Fear prioritizes threat-related or negative memories
            modulation_params["tag_boosts"] = {"threat": 0.8, "negative": 0.6, "safety": 0.7}
        elif primary_emotion.lower() in ["joy", "happiness"]:
            # Joy prioritizes positive experiences
            modulation_params["tag_boosts"] = {"success": 0.7, "positive": 0.8, "achievement": 0.6}
        elif primary_emotion.lower() in ["curiosity", "interest"]:
            # Curiosity prioritizes novel and unexplored areas
            modulation_params["novelty_weight"] = 0.8
            modulation_params["tag_boosts"] = {"learning": 0.7, "insight": 0.8}
        elif primary_emotion.lower() in ["dominance_satisfaction", "confident_control"]:
            # Dominance prioritizes control/power-related memories
            modulation_params["tag_boosts"] = {"control": 0.9, "dominance": 0.8, "success": 0.7}
        
        # Execute modulated retrieval
        try:
            # Prepare memory retrieval parameters
            retrieval_params = {
                "query": query,
                "limit": limit
            }
            
            # Add memory types if specified
            if memory_types:
                retrieval_params["memory_types"] = memory_types
                
            # Add modulation parameters
            retrieval_params.update(modulation_params)
            
            # Execute retrieval with emotional modulation
            memories = await self.memory_orchestrator.retrieve_memories(**retrieval_params)
            
            return {
                "status": "success",
                "memories": memories,
                "query": query,
                "emotional_modulation": modulation_params,
                "count": len(memories)
            }
        except Exception as e:
            logger.error(f"Error in modulated memory retrieval: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalCognitive")
    async def modulate_reasoning(self, 
                              reasoning_input: str, 
                              reasoning_type: str = "general",
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Modulate reasoning processes based on emotional state.
        
        Args:
            reasoning_input: Input for reasoning
            reasoning_type: Type of reasoning to perform
            context: Optional additional context
            
        Returns:
            Modulated reasoning results
        """
        if not self.reasoning_core:
            return {"status": "error", "message": "Reasoning core not available"}
        
        # Ensure state is updated
        await self.update_emotional_cognitive_state()
        
        # Get cognitive biases from current state
        biases = self.emotional_reasoning_biases
        
        # Prepare reasoning context with emotional biases
        reasoning_context = context or {}
        reasoning_context["emotional_cognitive_state"] = {
            "primary_emotion": self.current_emotional_metadata.get("primary_emotion"),
            "valence": self.current_emotional_metadata.get("valence"),
            "arousal": self.current_emotional_metadata.get("arousal"),
            "biases": biases
        }
        
        # Apply bias-specific reasoning adjustments
        # For example, adjust depth of processing based on analytical_depth bias
        analytical_depth = biases.get("analytical_depth", 0.0)
        if analytical_depth > 0.5:
            reasoning_context["processing_depth"] = "deep"
            reasoning_context["reasoning_steps"] = 5
        elif analytical_depth < -0.5:
            reasoning_context["processing_depth"] = "shallow" 
            reasoning_context["reasoning_steps"] = 2
        else:
            reasoning_context["processing_depth"] = "standard"
            reasoning_context["reasoning_steps"] = 3
        
        # Adjust abstract vs concrete thinking based on abstraction_level bias
        abstraction_level = biases.get("abstraction_level", 0.0)
        if abstraction_level > 0.5:
            reasoning_context["thinking_style"] = "abstract"
        elif abstraction_level < -0.5:
            reasoning_context["thinking_style"] = "concrete"
        else:
            reasoning_context["thinking_style"] = "balanced"
        
        # Adjust counterfactual thinking based on creativity_bias
        creativity_bias = biases.get("creativity_bias", 0.0)
        if creativity_bias > 0.5:
            reasoning_context["generative_thinking"] = "high"
        elif creativity_bias < -0.5:
            reasoning_context["generative_thinking"] = "low"
        else:
            reasoning_context["generative_thinking"] = "medium"
        
        # Execute modulated reasoning
        try:
            # Execute reasoning with emotional modulation
            if reasoning_type == "causal" and hasattr(self.reasoning_core, "reason_causal"):
                result = await self.reasoning_core.reason_causal(reasoning_input, reasoning_context)
            elif reasoning_type == "counterfactual" and hasattr(self.reasoning_core, "reason_counterfactually"):
                result = await self.reasoning_core.reason_counterfactually(reasoning_input, reasoning_context)
            elif reasoning_type == "intervention" and hasattr(self.reasoning_core, "reason_intervention"):
                result = await self.reasoning_core.reason_intervention(reasoning_input, reasoning_context)
            else:
                # Default to general reasoning
                result = await self.reasoning_core.execute_reasoning(reasoning_input, reasoning_context)
            
            return {
                "status": "success",
                "reasoning_result": result,
                "emotional_modulation": {
                    "biases_applied": biases,
                    "reasoning_context": reasoning_context
                }
            }
        except Exception as e:
            logger.error(f"Error in emotionally modulated reasoning: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalCognitive")
    async def generate_emotional_response(self, 
                                       cognitive_insight: str, 
                                       intensity: float = 0.5) -> Dict[str, Any]:
        """
        Generate an emotional response to a cognitive insight.
        
        Args:
            cognitive_insight: The cognitive insight to respond to
            intensity: Desired intensity of emotional response
            
        Returns:
            Generated emotional response
        """
        if not self.emotional_core:
            return {"status": "error", "message": "Emotional core not available"}
        
        try:
            # Analyze the cognitive insight for emotional content
            emotional_analysis = {}
            if hasattr(self.emotional_core, "analyze_text_sentiment"):
                if asyncio.iscoroutinefunction(self.emotional_core.analyze_text_sentiment):
                    emotional_analysis = await self.emotional_core.analyze_text_sentiment(cognitive_insight)
                else:
                    emotional_analysis = self.emotional_core.analyze_text_sentiment(cognitive_insight)
            
            # Scale the emotional response based on intensity and cognitive-to-emotion multiplier
            impact_strength = intensity * self.cognition_to_emotion_multiplier
            
            # Convert analysis to emotional input for processing
            emotional_input = f"Cognitive insight: {cognitive_insight[:100]}..." if len(cognitive_insight) > 100 else cognitive_insight
            
            # Process the emotional input
            if hasattr(self.emotional_core, "process_emotional_input"):
                response = await self.emotional_core.process_emotional_input(emotional_input)
            else:
                # Fallback to updating individual emotions if available
                response = {"status": "partial"}
                if hasattr(self.emotional_core, "update_emotion"):
                    # Get dominant emotion from analysis
                    dominant_emotion = max(emotional_analysis.items(), key=lambda x: x[1])[0] if emotional_analysis else "neutral"
                    response["updated_emotion"] = await self.emotional_core.update_emotion(dominant_emotion, impact_strength)
            
            return {
                "status": "success",
                "emotional_response": response,
                "analysis": emotional_analysis,
                "impact_strength": impact_strength
            }
        except Exception as e:
            logger.error(f"Error generating emotional response: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional state change events.
        
        Args:
            event: Emotional state change event
        """
        try:
            # Update internal state
            await self.update_emotional_cognitive_state()
            
            # Extract emotional data from event
            emotion = event.data.get("emotion", "neutral")
            valence = event.data.get("valence", 0.0)
            arousal = event.data.get("arousal", 0.5)
            
            # Only trigger cognitive processes for significant emotional changes
            if event.data.get("intensity", 0.5) > 0.7 or abs(valence) > 0.7:
                # Trigger reflections for significant emotional changes
                if self.reflection_engine and hasattr(self.reflection_engine, "generate_reflection"):
                    reflection_context = {
                        "trigger": "emotional_change",
                        "emotion": emotion,
                        "valence": valence,
                        "arousal": arousal
                    }
                    asyncio.create_task(self.reflection_engine.generate_reflection(
                        f"Emotional response: {emotion}", reflection_context
                    ))
        except Exception as e:
            logger.error(f"Error handling emotional change event: {e}")
    
    async def _handle_cognitive_insight(self, event: Event) -> None:
        """
        Handle cognitive insight events.
        
        Args:
            event: Cognitive insight event
        """
        try:
            # Extract insight from event
            insight = event.data.get("insight", "")
            importance = event.data.get("importance", 0.5)
            
            # Generate emotional response for significant insights
            if importance > 0.6 and insight:
                asyncio.create_task(self.generate_emotional_response(insight, importance))
        except Exception as e:
            logger.error(f"Error handling cognitive insight event: {e}")
    
    async def _handle_memory_retrieved(self, event: Event) -> None:
        """
        Handle memory retrieved events.
        
        Args:
            event: Memory retrieved event
        """
        try:
            # Extract memory data from event
            memory = event.data.get("memory", {})
            significance = memory.get("significance", 0.5)
            
            # Generate emotional response for significant memories
            if significance > 0.7 and memory.get("memory_text"):
                emotional_text = f"Recalled memory: {memory.get('memory_text')}"
                asyncio.create_task(self.generate_emotional_response(emotional_text, significance * 0.8))
        except Exception as e:
            logger.error(f"Error handling memory retrieved event: {e}")
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalCognitive")
    async def get_bridge_state(self) -> Dict[str, Any]:
        """
        Get the current state of the emotional-cognitive bridge.
        
        Returns:
            Current bridge state
        """
        # Ensure state is updated
        await self.update_emotional_cognitive_state()
        
        return {
            "emotional_metadata": self.current_emotional_metadata,
            "cognitive_biases": self.emotional_reasoning_biases,
            "emotion_to_cognition_multiplier": self.emotion_to_cognition_multiplier,
            "cognition_to_emotion_multiplier": self.cognition_to_emotion_multiplier,
            "last_update": self.last_update.isoformat()
        }

# Function to create the emotional-cognitive bridge
def create_emotional_cognitive_bridge(brain_reference=None):
    """Create an emotional-cognitive bridge for the given brain."""
    return EmotionalCognitiveBridge(brain_reference=brain_reference)
