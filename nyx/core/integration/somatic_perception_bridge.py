# nyx/core/integration/somatic_perception_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from nyx.core.integration.event_bus import Event, PhysicalSensationEvent, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class SomaticPerceptionBridge:
    """
    Integration bridge between digital somatosensory system, body image, and other modules.
    
    Coordinates physical sensations, body awareness, and perceptual processes to create
    a cohesive somatic experience that interacts with emotion, memory, and attention.
    
    Key functions:
    1. Routes physical sensations to appropriate systems
    2. Ensures body image consistency across perception modules
    3. Connects emotional states with physical sensations
    4. Provides somatic feedback for identity and narrative systems
    5. Integrates attention with somatosensory processes
    """
    
    def __init__(self, 
                brain_reference=None, 
                digital_somatosensory_system=None, 
                body_image=None,
                emotional_core=None,
                multimodal_integrator=None,
                attention_system=None,
                memory_orchestrator=None):
        """Initialize the somatic perception bridge."""
        self.brain = brain_reference
        self.dss = digital_somatosensory_system
        self.body_image = body_image
        self.emotional_core = emotional_core
        self.multimodal_integrator = multimodal_integrator
        self.attention_system = attention_system
        self.memory_orchestrator = memory_orchestrator
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.somatic_memory_threshold = 0.7  # Threshold for creating memories from sensations
        self.emotion_to_sensation_strength = 0.6  # How strongly emotions trigger sensations
        self.sensation_to_emotion_strength = 0.4  # How strongly sensations affect emotions
        
        # Maintain history of recent sensations
        self.recent_sensations = []
        self.max_recent_sensations = 20
        
        # Track somatic-emotional correlations
        self.somatic_emotional_correlations = {
            # Emotions -> Body regions and sensations
            "joy": {"regions": ["chest", "face"], "sensations": ["warmth", "tingling"]},
            "sadness": {"regions": ["throat", "chest"], "sensations": ["pressure", "heaviness"]},
            "fear": {"regions": ["stomach", "chest"], "sensations": ["tightness", "cold"]},
            "anger": {"regions": ["head", "hands"], "sensations": ["heat", "pressure"]},
            "disgust": {"regions": ["stomach", "throat"], "sensations": ["nausea", "tension"]},
            "surprise": {"regions": ["chest", "face"], "sensations": ["tingling", "pressure"]},
            "anticipation": {"regions": ["stomach", "chest"], "sensations": ["fluttering", "lightness"]},
            "trust": {"regions": ["chest", "shoulders"], "sensations": ["warmth", "relaxation"]},
            "shame": {"regions": ["face", "stomach"], "sensations": ["heat", "sinking"]},
            "pride": {"regions": ["chest", "spine"], "sensations": ["expansion", "lightness"]},
            "confident": {"regions": ["chest", "spine", "head"], "sensations": ["warmth", "expansion"]},
            "submissive": {"regions": ["neck", "shoulders"], "sensations": ["lightness", "tingling"]},
            "dominant": {"regions": ["chest", "spine", "arms"], "sensations": ["heat", "expansion", "pressure"]},
            
            # Sensations -> Emotional effects
            "pain": {"emotion_changes": {"negative": 0.3, "arousal": 0.4}},
            "pleasure": {"emotion_changes": {"positive": 0.4, "relaxation": 0.3}},
            "warmth": {"emotion_changes": {"positive": 0.2, "relaxation": 0.2}},
            "cold": {"emotion_changes": {"negative": 0.2, "tension": 0.2}},
            "tingling": {"emotion_changes": {"arousal": 0.3, "attention": 0.3}},
            "pressure": {"emotion_changes": {"tension": 0.3, "focus": 0.2}}
        }
        
        # Integration state tracking
        self._subscribed = False
        
        logger.info("SomaticPerceptionBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Set up connections to required systems if needed
            if not self.dss and hasattr(self.brain, "digital_somatosensory_system"):
                self.dss = self.brain.digital_somatosensory_system
                
            if not self.body_image and hasattr(self.brain, "body_image"):
                self.body_image = self.brain.body_image
                
            if not self.emotional_core and hasattr(self.brain, "emotional_core"):
                self.emotional_core = self.brain.emotional_core
                
            if not self.multimodal_integrator and hasattr(self.brain, "multimodal_integrator"):
                self.multimodal_integrator = self.brain.multimodal_integrator
                
            if not self.attention_system and hasattr(self.brain, "dynamic_attention_system"):
                self.attention_system = self.brain.dynamic_attention_system
                
            if not self.memory_orchestrator and hasattr(self.brain, "memory_orchestrator"):
                self.memory_orchestrator = self.brain.memory_orchestrator
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("physical_sensation", self._handle_physical_sensation)
                self.event_bus.subscribe("attention_focus_changed", self._handle_attention_change)
                self._subscribed = True
            
            # Initialize body state
            if self.dss:
                await self.dss.initialize()
            
            if self.body_image:
                # Initialize with default form if needed
                if not self.body_image.current_state.has_visual_form:
                    await self.body_image.reset_to_default_form()
            
            logger.info("SomaticPerceptionBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing SomaticPerceptionBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="SomaticPerception")
    async def process_physical_sensation(self, 
                                     region: str, 
                                     sensation_type: str,
                                     intensity: float,
                                     cause: str = "",
                                     duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a physical sensation through the bridge.
        
        Args:
            region: Body region experiencing sensation
            sensation_type: Type of sensation (pressure, temperature, pain, pleasure, tingling)
            intensity: Intensity of sensation (0.0-1.0)
            cause: Optional cause of sensation
            duration: Duration of sensation in seconds
            
        Returns:
            Processing results with cross-module effects
        """
        if not self.dss:
            return {"status": "error", "message": "Digital somatosensory system not available"}
        
        try:
            # Create processing result container
            processing_result = {
                "region": region,
                "sensation_type": sensation_type,
                "intensity": intensity,
                "cause": cause,
                "duration": duration,
                "dss_processed": False,
                "body_image_updated": False,
                "emotion_affected": False,
                "attention_affected": False,
                "memory_created": False
            }
            
            # 1. Process through digital somatosensory system
            dss_result = await self.dss.process_stimulus(
                stimulus_type=sensation_type,
                body_region=region,
                intensity=intensity,
                cause=cause,
                duration=duration
            )
            
            processing_result["dss_processed"] = True
            processing_result["dss_result"] = dss_result
            
            # 2. Update body image
            if self.body_image:
                body_update = await self.body_image.update_from_somatic()
                processing_result["body_image_updated"] = True
                processing_result["body_update"] = body_update
            
            # 3. Generate emotional effect if intensity is significant
            if intensity >= 0.5 and self.emotional_core:
                emotional_effect = await self._generate_emotional_effect(sensation_type, region, intensity)
                if emotional_effect.get("emotion_updated", False):
                    processing_result["emotion_affected"] = True
                    processing_result["emotional_effect"] = emotional_effect
            
            # 4. Affect attention if intensity is high
            if intensity >= 0.7 and self.attention_system:
                attention_result = await self._direct_attention_to_sensation(region, sensation_type, intensity)
                if attention_result.get("attention_shifted", False):
                    processing_result["attention_affected"] = True
                    processing_result["attention_result"] = attention_result
            
            # 5. Create memory if significant enough
            if intensity >= self.somatic_memory_threshold and self.memory_orchestrator:
                memory_id = await self._create_sensation_memory(region, sensation_type, intensity, cause, dss_result)
                if memory_id:
                    processing_result["memory_created"] = True
                    processing_result["memory_id"] = memory_id
            
            # 6. Track in recent sensations
            self.recent_sensations.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "region": region,
                "sensation_type": sensation_type,
                "intensity": intensity,
                "cause": cause,
                "duration": duration
            })
            
            # Trim history if needed
            if len(self.recent_sensations) > self.max_recent_sensations:
                self.recent_sensations = self.recent_sensations[-self.max_recent_sensations:]
            
            # 7. Publish event for other modules
            event = PhysicalSensationEvent(
                source="somatic_perception_bridge",
                region=region,
                sensation_type=sensation_type,
                intensity=intensity,
                cause=cause
            )
            await self.event_bus.publish(event)
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing physical sensation: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="SomaticPerception")
    async def generate_physical_expression(self, 
                                       emotional_state: Dict[str, Any],
                                       expression_type: str = "natural",
                                       intensity: float = 0.5) -> Dict[str, Any]:
        """
        Generate a physical expression based on emotional state.
        
        Args:
            emotional_state: Current emotional state
            expression_type: Type of expression (natural, intense, subtle)
            intensity: Intensity of expression (0.0-1.0)
            
        Returns:
            Generated physical expression
        """
        if not self.dss:
            return {"status": "error", "message": "Digital somatosensory system not available"}
        
        try:
            # Extract emotion data
            emotion = emotional_state.get("primary_emotion", "neutral")
            valence = emotional_state.get("valence", 0.0)
            arousal = emotional_state.get("arousal", 0.5)
            
            # Adjust intensity based on expression type
            adjusted_intensity = intensity
            if expression_type == "intense":
                adjusted_intensity = min(1.0, intensity * 1.5)
            elif expression_type == "subtle":
                adjusted_intensity = max(0.1, intensity * 0.6)
            
            # Find emotion in correlation map or use most similar
            correlation_data = None
            if emotion in self.somatic_emotional_correlations:
                correlation_data = self.somatic_emotional_correlations[emotion]
            else:
                # Find most similar emotion based on valence and arousal
                closest_emotion = "neutral"
                closest_distance = 2.0  # Maximum possible distance
                
                for corr_emotion, data in self.somatic_emotional_correlations.items():
                    # Get typical valence/arousal for this emotion (simplified)
                    emotion_valence = 0.5 if "positive" in corr_emotion else -0.5
                    emotion_arousal = 0.7 if corr_emotion in ["anger", "fear", "joy", "surprise"] else 0.3
                    
                    # Calculate distance
                    distance = ((valence - emotion_valence) ** 2 + (arousal - emotion_arousal) ** 2) ** 0.5
                    
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_emotion = corr_emotion
                
                correlation_data = self.somatic_emotional_correlations.get(closest_emotion)
            
            if not correlation_data:
                # Fallback for unknown emotions
                correlation_data = {
                    "regions": ["chest", "shoulders"],
                    "sensations": ["pressure", "temperature"]
                }
            
            # Select region and sensation type based on correlation data
            regions = correlation_data.get("regions", [])
            sensations = correlation_data.get("sensations", [])
            
            if not regions or not sensations:
                return {"status": "error", "message": "Insufficient correlation data"}
            
            # Select primary region and sensation
            region = regions[0] if regions else "chest"
            sensation = sensations[0] if sensations else "pressure"
            
            # Map sensation to stimulus type
            sensation_mapping = {
                "warmth": "temperature", "heat": "temperature", "cold": "temperature",
                "pressure": "pressure", "tightness": "pressure", "heaviness": "pressure",
                "tingling": "tingling", "fluttering": "tingling",
                "pain": "pain", 
                "pleasure": "pleasure",
                "expansion": "pressure", "relaxation": "pressure", "tension": "pressure"
            }
            
            stimulus_type = sensation_mapping.get(sensation, "pressure")
            
            # Map temperature sensations to values
            temp_value = 0.5  # Neutral
            if sensation == "warmth":
                temp_value = 0.65
            elif sensation == "heat":
                temp_value = 0.8
            elif sensation == "cold":
                temp_value = 0.2
            
            # Process stimulus through DSS
            if stimulus_type == "temperature":
                expression_result = await self.dss.process_stimulus(
                    stimulus_type=stimulus_type,
                    body_region=region,
                    intensity=temp_value,  # Use temperature value
                    cause=f"emotional_expression:{emotion}",
                    duration=adjusted_intensity * 3.0  # Longer duration for more intense
                )
            else:
                expression_result = await self.dss.process_stimulus(
                    stimulus_type=stimulus_type,
                    body_region=region,
                    intensity=adjusted_intensity,
                    cause=f"emotional_expression:{emotion}",
                    duration=adjusted_intensity * 3.0
                )
            
            # Get expression from DSS
            verbal_expression = None
            if hasattr(self.dss, "generate_sensory_expression"):
                verbal_expression = await self.dss.generate_sensory_expression(
                    stimulus_type=stimulus_type,
                    body_region=region
                )
            
            # Return result
            return {
                "status": "success",
                "emotion": emotion,
                "region": region,
                "sensation": sensation,
                "stimulus_type": stimulus_type,
                "intensity": adjusted_intensity,
                "expression_result": expression_result,
                "verbal_expression": verbal_expression
            }
            
        except Exception as e:
            logger.error(f"Error generating physical expression: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="SomaticPerception")
    async def sync_body_state(self) -> Dict[str, Any]:
        """
        Synchronize body state between somatosensory system and body image.
        
        Returns:
            Synchronization results
        """
        if not self.dss or not self.body_image:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get current DSS state
            dss_state = await self.dss.get_body_state()
            
            # 2. Get current body image state
            body_image_state = self.body_image.get_body_image_state()
            
            # 3. Update body image from DSS
            body_update = await self.body_image.update_from_somatic()
            
            # 4. Update overall state in system context
            self.system_context.set_value("somatic_state", {
                "dss_state": dss_state,
                "body_image": {
                    "form_description": body_image_state.form_description,
                    "integrity": body_image_state.overall_integrity,
                    "proprioception_confidence": body_image_state.proprioception_confidence,
                    "part_count": len(body_image_state.perceived_parts)
                },
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # 5. Update attention with significant body sensations
            attention_updated = False
            if self.attention_system and dss_state:
                dominant_sensation = dss_state.get("dominant_sensation")
                dominant_intensity = dss_state.get("dominant_intensity", 0.0)
                dominant_region = dss_state.get("dominant_region")
                
                if dominant_intensity >= 0.7:
                    # High intensity, direct attention
                    await self._direct_attention_to_sensation(
                        dominant_region,
                        dominant_sensation,
                        dominant_intensity
                    )
                    attention_updated = True
            
            return {
                "status": "success",
                "dss_state": {
                    "dominant_sensation": dss_state.get("dominant_sensation"),
                    "dominant_region": dss_state.get("dominant_region"),
                    "comfort_level": dss_state.get("comfort_level")
                },
                "body_image": {
                    "form_description": body_image_state.form_description,
                    "proprioception_confidence": body_image_state.proprioception_confidence
                },
                "attention_updated": attention_updated,
                "last_update": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error synchronizing body state: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="SomaticPerception")
    async def process_touch_event(self,
                               region: str,
                               touch_type: str,
                               intensity: float,
                               source: str = "user",
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a touch event and generate appropriate sensory response.
        
        Args:
            region: Body region being touched
            touch_type: Type of touch (stroke, pat, grab, etc.)
            intensity: Intensity of touch (0.0-1.0)
            source: Source of touch (user, environment, self)
            context: Optional additional context
            
        Returns:
            Processing results
        """
        if not self.dss:
            return {"status": "error", "message": "Digital somatosensory system not available"}
        
        try:
            # Map touch types to sensation types
            touch_mapping = {
                "stroke": "pleasure",
                "pat": "pressure",
                "grab": "pressure",
                "squeeze": "pressure",
                "poke": "pressure",
                "caress": "pleasure",
                "massage": "pleasure",
                "tickle": "tingling",
                "slap": "pain",
                "pinch": "pain",
                "rub": "temperature"  # warming friction
            }
            
            # Default to pressure if unknown
            sensation_type = touch_mapping.get(touch_type.lower(), "pressure")
            
            # Adjust intensity based on touch type and region sensitivity
            region_sensitivity = {
                "face": 1.3,
                "neck": 1.2,
                "hands": 1.4,
                "feet": 1.3,
                "chest": 1.2,
                "back": 1.2,
                "stomach": 1.3,
                "arms": 1.1,
                "legs": 1.1
            }
            
            # Default sensitivity of 1.0 if region not found
            sensitivity = region_sensitivity.get(region, 1.0)
            adjusted_intensity = min(1.0, intensity * sensitivity)
            
            # Process as physical sensation
            processing_result = await self.process_physical_sensation(
                region=region,
                sensation_type=sensation_type,
                intensity=adjusted_intensity,
                cause=f"touch:{touch_type} from {source}",
                duration=intensity * 2.0  # Longer duration for more intense touches
            )
            
            # Add touch-specific information
            processing_result["touch_type"] = touch_type
            processing_result["touch_source"] = source
            
            # Check for emotional meaning of touch
            if context and "emotional_intent" in context:
                emotional_intent = context["emotional_intent"]
                processing_result["emotional_intent"] = emotional_intent
                
                # Add to relationship data if user source
                if source == "user" and hasattr(self.brain, "relationship_manager"):
                    relationship_update = {
                        "touch_event": {
                            "region": region,
                            "type": touch_type,
                            "intensity": intensity,
                            "emotional_intent": emotional_intent
                        }
                    }
                    
                    user_id = context.get("user_id")
                    if user_id:
                        await self.brain.relationship_manager.update_relationship_on_interaction(
                            user_id, relationship_update
                        )
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing touch event: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="SomaticPerception")
    async def get_somatic_perceptual_state(self) -> Dict[str, Any]:
        """
        Get the current somatic perceptual state across systems.
        
        Returns:
            Current somatic perceptual state
        """
        state = {
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # Get DSS state
            if self.dss:
                body_state = await self.dss.get_body_state()
                state["dss_state"] = body_state
            
            # Get body image state
            if self.body_image:
                body_image_state = self.body_image.get_body_image_state()
                
                # Convert to serializable form if needed
                if hasattr(body_image_state, "model_dump"):
                    body_image_dict = body_image_state.model_dump()
                else:
                    body_image_dict = {
                        "form_description": body_image_state.form_description,
                        "has_visual_form": body_image_state.has_visual_form,
                        "overall_integrity": body_image_state.overall_integrity,
                        "proprioception_confidence": body_image_state.proprioception_confidence,
                        "perceived_parts_count": len(body_image_state.perceived_parts)
                    }
                
                state["body_image"] = body_image_dict
            
            # Add recent sensations
            state["recent_sensations"] = self.recent_sensations[-5:] if self.recent_sensations else []
            
            # Get current temperature effects if available
            if self.dss and hasattr(self.dss, "get_temperature_effects"):
                state["temperature_effects"] = await self.dss.get_temperature_effects()
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting somatic perceptual state: {e}")
            state["error"] = str(e)
            return state
    
    async def _generate_emotional_effect(self, 
                                     sensation_type: str, 
                                     region: str, 
                                     intensity: float) -> Dict[str, Any]:
        """Generate emotional effect from sensation."""
        if not self.emotional_core:
            return {"status": "error", "message": "Emotional core not available"}
        
        try:
            # Get sensation-to-emotion mapping
            emotional_effects = {}
            
            # Check if we have specific mapping for this sensation
            if sensation_type in self.somatic_emotional_correlations:
                emotional_effects = self.somatic_emotional_correlations[sensation_type].get("emotion_changes", {})
            
            # Default effects if not found
            if not emotional_effects:
                if sensation_type == "pain":
                    emotional_effects = {"negative": 0.3, "arousal": 0.4}
                elif sensation_type == "pleasure":
                    emotional_effects = {"positive": 0.4, "relaxation": 0.3}
                else:
                    emotional_effects = {"arousal": 0.2}
            
            # Scale by intensity and bridge strength factor
            scaled_effects = {}
            for effect, value in emotional_effects.items():
                scaled_effects[effect] = value * intensity * self.sensation_to_emotion_strength
            
            # Determine primary emotion to update
            primary_emotion = None
            update_value = 0.0
            
            if "positive" in scaled_effects and scaled_effects["positive"] > 0.2:
                if sensation_type == "pleasure":
                    primary_emotion = "pleasure"
                    update_value = scaled_effects["positive"]
                else:
                    primary_emotion = "joy"
                    update_value = scaled_effects["positive"]
            elif "negative" in scaled_effects and scaled_effects["negative"] > 0.2:
                if sensation_type == "pain":
                    primary_emotion = "distress"
                    update_value = scaled_effects["negative"]
                else:
                    primary_emotion = "discomfort"
                    update_value = scaled_effects["negative"]
            
            # Update emotional state if we have a primary emotion
            emotion_updated = False
            if primary_emotion and hasattr(self.emotional_core, 'update_emotion'):
                await self.emotional_core.update_emotion(primary_emotion, update_value)
                emotion_updated = True
            
            # Also update neurochemicals if available
            if hasattr(self.emotional_core, 'update_neurochemical'):
                # Map sensation effects to neurochemicals
                if "positive" in scaled_effects:
                    await self.emotional_core.update_neurochemical("nyxamine", scaled_effects["positive"])
                
                if "relaxation" in scaled_effects:
                    await self.emotional_core.update_neurochemical("seranix", scaled_effects["relaxation"])
                
                if "arousal" in scaled_effects:
                    await self.emotional_core.update_neurochemical("adrenyx", scaled_effects["arousal"])
                
                if "negative" in scaled_effects:
                    await self.emotional_core.update_neurochemical("cortanyx", scaled_effects["negative"])
            
            return {
                "emotion_updated": emotion_updated,
                "primary_emotion": primary_emotion,
                "update_value": update_value,
                "scaled_effects": scaled_effects
            }
            
        except Exception as e:
            logger.error(f"Error generating emotional effect: {e}")
            return {"emotion_updated": False, "error": str(e)}
    
    async def _direct_attention_to_sensation(self, 
                                         region: str, 
                                         sensation_type: str, 
                                         intensity: float) -> Dict[str, Any]:
        """Direct attention to sensation."""
        if not self.attention_system:
            return {"status": "error", "message": "Attention system not available"}
        
        try:
            # Create attention target
            target = f"physical sensation: {sensation_type} in {region}"
            
            # Calculate attention level based on intensity
            attention_level = min(1.0, intensity * 1.2)  # Scale up slightly, but max at 1.0
            
            # Direct attention
            if hasattr(self.attention_system, 'focus_attention'):
                result = await self.attention_system.focus_attention(
                    target=target,
                    target_type="physical_sensation",
                    attention_level=attention_level,
                    source="somatic_perception_bridge"
                )
                
                return {
                    "attention_shifted": True,
                    "target": target,
                    "attention_level": attention_level,
                    "result": result
                }
            else:
                return {
                    "attention_shifted": False,
                    "reason": "Attention system lacking focus_attention method"
                }
            
        except Exception as e:
            logger.error(f"Error directing attention to sensation: {e}")
            return {"attention_shifted": False, "error": str(e)}
    
    async def _create_sensation_memory(self, 
                                    region: str, 
                                    sensation_type: str, 
                                    intensity: float, 
                                    cause: str,
                                    dss_result: Any) -> Optional[str]:
        """Create memory of significant sensation."""
        if not self.memory_orchestrator:
            return None
        
        try:
            # Create memory text
            memory_text = f"Experienced {sensation_type} sensation in my {region} with intensity {intensity:.2f}"
            if cause:
                memory_text += f" caused by {cause}"
            
            # Add sensory expression if available
            if self.dss and hasattr(self.dss, "generate_sensory_expression"):
                sensory_expression = await self.dss.generate_sensory_expression(
                    stimulus_type=sensation_type,
                    body_region=region
                )
                if sensory_expression:
                    memory_text += f"\nExperienced as: {sensory_expression}"
            
            # Calculate significance (0-10 scale)
            significance = int(min(10, intensity * 10))
            
            # Create tags
            tags = ["physical_sensation", sensation_type, region]
            
            # Add memory
            memory_id = await self.memory_orchestrator.add_memory(
                memory_text=memory_text,
                memory_type="experience",
                significance=significance,
                tags=tags,
                metadata={
                    "sensation_type": sensation_type,
                    "body_region": region,
                    "intensity": intensity,
                    "cause": cause,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "dss_result": {k: v for k, v in dss_result.items() if not isinstance(v, (dict, list))} if isinstance(dss_result, dict) else None
                }
            )
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error creating sensation memory: {e}")
            return None
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional state change events.
        
        Args:
            event: Emotional state change event
        """
        try:
            # Extract emotional data
            emotion = event.data.get("emotion")
            valence = event.data.get("valence", 0.0)
            arousal = event.data.get("arousal", 0.5)
            intensity = event.data.get("intensity", 0.5)
            
            if not emotion:
                return
            
            # Only process significant emotional changes
            if intensity < 0.7:
                return
            
            # Generate physical expression for this emotion
            asyncio.create_task(
                self.generate_physical_expression(
                    emotional_state={
                        "primary_emotion": emotion,
                        "valence": valence,
                        "arousal": arousal
                    },
                    intensity=intensity
                )
            )
        except Exception as e:
            logger.error(f"Error handling emotional change: {e}")
    
    async def _handle_physical_sensation(self, event: Event) -> None:
        """
        Handle physical sensation events.
        
        Args:
            event: Physical sensation event
        """
        try:
            # Extract sensation data
            region = event.data.get("region")
            sensation_type = event.data.get("sensation_type")
            intensity = event.data.get("intensity")
            cause = event.data.get("cause", "")
            
            if not region or not sensation_type or intensity is None:
                return
            
            # Process the sensation
            asyncio.create_task(
                self.process_physical_sensation(
                    region=region,
                    sensation_type=sensation_type,
                    intensity=intensity,
                    cause=cause
                )
            )
        except Exception as e:
            logger.error(f"Error handling physical sensation event: {e}")
    
    async def _handle_attention_change(self, event: Event) -> None:
        """
        Handle attention change events.
        
        Args:
            event: Attention change event
        """
        try:
            # Extract attention data
            focus = event.data.get("focus")
            target_type = event.data.get("target_type", "")
            
            if not focus:
                return
            
            # If attention is on body part, enhance sensitivity
            if "body" in target_type or "physical" in target_type:
                # Parse region from target if possible
                target = event.data.get("target", "")
                
                # Check if target mentions a body region
                region = None
                for part in ["hand", "arm", "leg", "chest", "head", "face", "shoulder", "back", "stomach"]:
                    if part in target.lower():
                        region = part
                        break
                
                if region and self.dss:
                    # Enhanced sensitivity to this region while attention is on it
                    # Just a subtle tingling to reflect attention
                    asyncio.create_task(
                        self.dss.process_stimulus(
                            stimulus_type="tingling",
                            body_region=region,
                            intensity=0.3,  # Light sensation
                            cause="attention_focus",
                            duration=1.0
                        )
                    )
        except Exception as e:
            logger.error(f"Error handling attention change event: {e}")

# Function to create the bridge
def create_somatic_perception_bridge(nyx_brain):
    """Create a somatic perception bridge for the given brain."""
    return SomaticPerceptionBridge(brain_reference=nyx_brain)
