# nyx/core/integration/perceptual_integration_layer.py

import logging
import asyncio
import datetime
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import deque

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class PerceptualIntegrationLayer:
    """
    Unified integration layer for sensory and perceptual processes.
    
    This module coordinates between multimodal perception, internal simulations,
    body image, digital somatosensory system, and attention to create a coherent
    perceptual experience across sensory modalities.
    
    Key functions:
    1. Coordinates multimodal inputs into a unified percept
    2. Manages attention across sensory channels
    3. Integrates DSS (Digital Somatosensory) input with other perception
    4. Manages expectation generation and top-down processing
    5. Creates and updates body image based on perceptual data
    """
    
    def __init__(self, 
                brain_reference=None, 
                multimodal_integrator=None, 
                digital_somatosensory_system=None,
                body_image=None,
                attentional_controller=None):
        """Initialize the perceptual integration layer."""
        self.brain = brain_reference
        self.multimodal_integrator = multimodal_integrator
        self.dss = digital_somatosensory_system
        self.body_image = body_image
        self.attentional_controller = attentional_controller
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.integration_window = 0.5  # seconds to integrate inputs
        self.minimum_attention_threshold = 0.4  # minimum attention for processing
        self.sensory_decay_rate = 0.2  # rate of decay for sensory traces
        
        # Recent percepts (modality -> list of recent percepts)
        self.recent_percepts = {
            "text": deque(maxlen=10),
            "image": deque(maxlen=5),
            "audio": deque(maxlen=5),
            "somatosensory": deque(maxlen=10),
            "integrated": deque(maxlen=10)
        }
        
        # Expectation management
        self.active_expectations = []
        self.expectation_decay_rate = 0.1  # rate of decay for expectations
        
        # Cross-modal correlations
        self.modal_correlations = {}
        
        # Timestamp tracking
        self.last_integration = datetime.datetime.now()
        self.startup_time = datetime.datetime.now()
        
        # Integration event subscriptions
        self._subscribed = False
        
        logger.info("PerceptualIntegrationLayer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the layer and establish connections to systems."""
        try:
            # Set up connections to required systems if needed
            if not self.multimodal_integrator and hasattr(self.brain, "multimodal_integrator"):
                self.multimodal_integrator = self.brain.multimodal_integrator
                
            if not self.dss and hasattr(self.brain, "digital_somatosensory_system"):
                self.dss = self.brain.digital_somatosensory_system
                
            if not self.body_image and hasattr(self.brain, "body_image"):
                self.body_image = self.brain.body_image
                
            if not self.attentional_controller and hasattr(self.brain, "attentional_controller"):
                self.attentional_controller = self.brain.attentional_controller
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("sensory_input", self._handle_sensory_input)
                self.event_bus.subscribe("physical_sensation", self._handle_physical_sensation)
                self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
                self._subscribed = True
            
            logger.info("PerceptualIntegrationLayer successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing PerceptualIntegrationLayer: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="PerceptualIntegration")
    async def process_sensory_input(self, 
                                 input_data: Dict[str, Any], 
                                 modality: str = "text",
                                 expectations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process sensory input through the integration layer.
        
        Args:
            input_data: Input data to process
            modality: Input modality (text, image, audio, etc.)
            expectations: Optional expectations to apply
            
        Returns:
            Processing results with integrated percept
        """
        try:
            # 1. Attention check - determine if this input gets attention
            attention_score = await self._calculate_attention_score(input_data, modality)
            
            if attention_score < self.minimum_attention_threshold:
                logger.debug(f"Ignoring {modality} input due to low attention score: {attention_score:.2f}")
                return {
                    "status": "ignored",
                    "attention_score": attention_score,
                    "reason": "below_attention_threshold"
                }
            
            # 2. Process through modality-specific processor if available
            modality_processor = getattr(self, f"_process_{modality}", None)
            
            modality_result = None
            if modality_processor and callable(modality_processor):
                modality_result = await modality_processor(input_data)
            
            # 3. Apply expectations if provided
            processed_expectations = []
            if expectations:
                processed_expectations = await self._process_expectations(expectations, modality)
            
            # 4. Pass to multimodal integrator if available
            if self.multimodal_integrator:
                # Prepare input for integrator
                integrator_input = {
                    "content": input_data,
                    "modality": modality,
                    "attention": attention_score,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "processed_data": modality_result
                }
                
                # Process through multimodal integrator
                integrator_result = await self.multimodal_integrator.process_sensory_input(
                    integrator_input, processed_expectations
                )
                
                # Store in recent percepts
                if integrator_result:
                    self.recent_percepts[modality].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "content": integrator_result,
                        "attention_score": attention_score
                    })
                    
                    # Also add to integrated percepts
                    self.recent_percepts["integrated"].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "modality": modality,
                        "content": integrator_result,
                        "attention_score": attention_score
                    })
                
                # Update body image if visual percept contains self
                if modality == "image" and self.body_image:
                    await self._update_body_image_from_percept(integrator_result)
                
                # Generate cross-modal correlations
                await self._update_cross_modal_correlations()
                
                # Record last integration time
                self.last_integration = datetime.datetime.now()
                
                return {
                    "status": "success",
                    "percept": integrator_result,
                    "attention_score": attention_score,
                    "expectations_applied": len(processed_expectations),
                    "timestamp": self.last_integration.isoformat()
                }
            else:
                # Return modality-specific processing result if no integrator
                return {
                    "status": "partial",
                    "percept": modality_result,
                    "attention_score": attention_score,
                    "expectations_applied": len(processed_expectations),
                    "timestamp": datetime.datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error processing {modality} input: {e}")
            return {"status": "error", "error": str(e), "modality": modality}
    
    @trace_method(level=TraceLevel.INFO, group_id="PerceptualIntegration")
    async def add_expectation(self, 
                           target_modality: str, 
                           pattern: Dict[str, Any], 
                           strength: float = 0.5,
                           duration: float = 10.0, 
                           source: str = "system") -> Dict[str, Any]:
        """
        Add a perceptual expectation to influence perception.
        
        Args:
            target_modality: Modality to apply expectation to (text, image, audio)
            pattern: Pattern to expect (format depends on modality)
            strength: Strength of expectation (0.0-1.0)
            duration: Duration in seconds before expectation decays
            source: Source of expectation
            
        Returns:
            Added expectation
        """
        expectation_id = f"exp_{len(self.active_expectations)}_{datetime.datetime.now().timestamp()}"
        
        expectation = {
            "id": expectation_id,
            "target_modality": target_modality,
            "pattern": pattern,
            "strength": strength,
            "original_strength": strength,
            "creation_time": datetime.datetime.now(),
            "expiration_time": datetime.datetime.now() + datetime.timedelta(seconds=duration),
            "source": source,
            "active": True
        }
        
        self.active_expectations.append(expectation)
        
        logger.info(f"Added new {target_modality} expectation {expectation_id} with strength {strength:.2f}")
        
        return {
            "status": "success",
            "expectation_id": expectation_id,
            "target_modality": target_modality,
            "strength": strength,
            "duration": duration,
            "expires_at": expectation["expiration_time"].isoformat()
        }
    
    @trace_method(level=TraceLevel.INFO, group_id="PerceptualIntegration")
    async def integrate_somatosensory_input(self, 
                                         region: str, 
                                         sensation_type: str,
                                         intensity: float, 
                                         cause: str = "") -> Dict[str, Any]:
        """
        Integrate somatosensory input with other perceptual processes.
        
        Args:
            region: Body region experiencing sensation
            sensation_type: Type of sensation (pressure, temperature, etc.)
            intensity: Intensity of sensation (0.0-1.0)
            cause: Optional cause of sensation
            
        Returns:
            Integration results
        """
        if not self.dss:
            return {"status": "error", "message": "Digital somatosensory system not available"}
        
        try:
            # 1. Process through DSS
            dss_result = await self.dss.process_stimulus(
                stimulus_type=sensation_type,
                body_region=region,
                intensity=intensity,
                cause=cause
            )
            
            # 2. Calculate attention score for this sensation
            attention_score = intensity
            # Adjust based on region sensitivity
            region_sensitivity = {
                "face": 1.2,
                "hands": 1.1,
                "torso": 0.9,
                "limbs": 0.8,
                "core": 1.0
            }
            if region in region_sensitivity:
                attention_score *= region_sensitivity[region]
            
            # 3. Store in recent percepts if passes attention threshold
            if attention_score >= self.minimum_attention_threshold:
                self.recent_percepts["somatosensory"].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "region": region,
                    "sensation_type": sensation_type,
                    "intensity": intensity,
                    "cause": cause,
                    "attention_score": attention_score,
                    "dss_result": dss_result
                })
                
                # 4. Update body image if available
                if self.body_image:
                    await self.body_image.update_from_somatic()
                
                # 5. Create an integrated percept
                integrated_percept = {
                    "modality": "somatosensory",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "content": {
                        "region": region,
                        "sensation_type": sensation_type,
                        "intensity": intensity,
                        "cause": cause
                    },
                    "attention_score": attention_score,
                    "dss_result": dss_result
                }
                
                # Add to integrated percepts
                self.recent_percepts["integrated"].append(integrated_percept)
                
                return {
                    "status": "success",
                    "integrated_percept": integrated_percept,
                    "attention_score": attention_score,
                    "dss_result": dss_result
                }
            else:
                return {
                    "status": "ignored",
                    "attention_score": attention_score,
                    "reason": "below_attention_threshold"
                }
        except Exception as e:
            logger.error(f"Error integrating somatosensory input: {e}")
            return {"status": "error", "error": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="PerceptualIntegration")
    async def get_current_perceptual_state(self) -> Dict[str, Any]:
        """
        Get the current perceptual state across all modalities.
        
        Returns:
            Current perceptual state
        """
        # Update expectations first
        await self._update_expectations()
        
        # Get most recent percepts for each modality
        recent_state = {}
        for modality, percepts in self.recent_percepts.items():
            if percepts:
                # Get most recent percept
                recent_state[modality] = percepts[-1]
        
        # Get active expectations
        active_expectations = [
            {k: v for k, v in exp.items() if k != 'pattern'} 
            for exp in self.active_expectations 
            if exp["active"]
        ]
        
        # Get current attention focus if available
        attention_focus = None
        if self.attentional_controller and hasattr(self.attentional_controller, "get_current_focus"):
            attention_focus = await self.attentional_controller.get_current_focus()
        
        # Get body image state if available
        body_state = None
        if self.body_image and hasattr(self.body_image, "get_body_image_state"):
            body_state = self.body_image.get_body_image_state()
            # Convert to serializable format if needed
            if body_state and hasattr(body_state, "model_dump"):
                body_state = body_state.model_dump()
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "recent_percepts": recent_state,
            "active_expectations": active_expectations,
            "attention_focus": attention_focus,
            "body_image_state": body_state,
            "cross_modal_correlations": self.modal_correlations
        }
    
    async def _process_text(self, text_data: str) -> Dict[str, Any]:
        """Process text input."""
        # Simple text processing - in a real implementation, this would be more sophisticated
        words = text_data.split()
        word_count = len(words)
        
        # Extract key features
        features = {
            "word_count": word_count,
            "char_count": len(text_data),
            "has_question": "?" in text_data,
            "capitalized_words": [w for w in words if w[0].isupper()] if words else []
        }
        
        # Basic sentiment analysis
        positive_words = ["good", "great", "wonderful", "happy", "love", "like", "enjoy"]
        negative_words = ["bad", "terrible", "sad", "hate", "dislike", "angry"]
        
        pos_count = sum(1 for w in words if w.lower() in positive_words)
        neg_count = sum(1 for w in words if w.lower() in negative_words)
        
        if pos_count + neg_count > 0:
            sentiment = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            sentiment = 0.0
            
        features["sentiment"] = sentiment
        
        return {
            "features": features,
            "processed_text": text_data
        }
    
    async def _process_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image input."""
        # Placeholder for image processing logic
        # In a real implementation, this would include object detection, scene recognition, etc.
        return {
            "features": {
                "analyzed": True,
                "size": image_data.get("size", "unknown"),
                "format": image_data.get("format", "unknown")
            },
            "processed_image": "Image processed"
        }
    
    async def _process_audio(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio input."""
        # Placeholder for audio processing logic
        # In a real implementation, this would include speech recognition, sound classification, etc.
        return {
            "features": {
                "analyzed": True,
                "duration": audio_data.get("duration", 0),
                "format": audio_data.get("format", "unknown")
            },
            "processed_audio": "Audio processed"
        }
    
    async def _calculate_attention_score(self, input_data: Any, modality: str) -> float:
        """Calculate attention score for input."""
        base_score = 0.5  # Default score
        
        # Ask attentional controller if available
        if self.attentional_controller and hasattr(self.attentional_controller, "calculate_attention"):
            controller_score = await self.attentional_controller.calculate_attention(input_data, modality)
            if controller_score is not None:
                return controller_score
        
        # Default calculation based on modality
        if modality == "text":
            text = input_data if isinstance(input_data, str) else str(input_data)
            # Attention heuristics for text
            if "?" in text:  # Questions get more attention
                base_score += 0.2
            if "!" in text:  # Exclamations get more attention
                base_score += 0.1
            # Longer texts get slightly less attention per word
            word_count = len(text.split())
            base_score -= min(0.3, word_count * 0.01)  # Max penalty of 0.3
        elif modality == "image":
            # Images generally get high attention
            base_score += 0.3
        elif modality == "audio":
            # Audio gets medium-high attention
            base_score += 0.2
        elif modality == "somatosensory":
            # Somatosensory input gets attention based on intensity
            if isinstance(input_data, dict) and "intensity" in input_data:
                base_score = input_data["intensity"]
            else:
                base_score += 0.1
        
        # Clamp to valid range
        return max(0.0, min(1.0, base_score))
    
    async def _process_expectations(self, expectations: List[Dict[str, Any]], modality: str) -> List[Dict[str, Any]]:
        """Process and filter expectations for a modality."""
        processed = []
        
        for exp in expectations:
            # Skip if not targeting this modality
            if exp.get("target_modality") != modality:
                continue
                
            # Check if still active
            if not exp.get("active", True):
                continue
                
            # Prepare expectation in format multimodal integrator expects
            processed_exp = {
                "id": exp.get("id", f"exp_{len(processed)}"),
                "target_modality": modality,
                "pattern": exp.get("pattern", {}),
                "strength": exp.get("strength", 0.5),
                "source": exp.get("source", "system")
            }
            
            processed.append(processed_exp)
        
        return processed
    
    async def _update_expectations(self) -> None:
        """Update expectations by applying decay and removing expired ones."""
        now = datetime.datetime.now()
        updated = []
        
        for exp in self.active_expectations:
            # Skip if already inactive
            if not exp.get("active", True):
                continue
                
            # Check expiration
            if now > exp.get("expiration_time", now):
                exp["active"] = False
                continue
            
            # Apply decay
            time_factor = (now - exp.get("creation_time", now)).total_seconds() / 60.0  # minutes
            strength_decay = self.expectation_decay_rate * time_factor
            exp["strength"] = max(0.0, exp.get("original_strength", 0.5) - strength_decay)
            
            # Deactivate if too weak
            if exp["strength"] < 0.1:
                exp["active"] = False
            else:
                updated.append(exp)
        
        # Replace active expectations with updated list
        self.active_expectations = self.active_expectations[:]  # Copy to preserve inactive ones
    
    async def _update_body_image_from_percept(self, percept: Any) -> None:
        """Update body image based on percept if it contains self-related information."""
        if not self.body_image:
            return
            
        # Check if percept has body image relevant data
        if hasattr(percept, "content") and isinstance(percept.content, dict):
            content = percept.content
            
            # Look for avatar or self-representation in visual percept
            if any(k for k in content.keys() if "avatar" in k.lower() or "self" in k.lower() or "nyx" in k.lower()):
                # Update body image from visual
                await self.body_image.update_from_visual(percept)
    
    async def _update_cross_modal_correlations(self) -> None:
        """Update cross-modal correlations based on recent percepts."""
        # This would use temporal correlation to identify related percepts across modalities
        # For example, correlating a visual smile with positive text sentiment
        
        # Simplistic implementation for demonstration
        visual_percepts = list(self.recent_percepts.get("image", []))
        text_percepts = list(self.recent_percepts.get("text", []))
        audio_percepts = list(self.recent_percepts.get("audio", []))
        somatic_percepts = list(self.recent_percepts.get("somatosensory", []))
        
        # Clear old correlations
        self.modal_correlations = {}
        
        # Check for temporal correlations (percepts close in time)
        # This is a simplified approach - real implementation would be more sophisticated
        if visual_percepts and text_percepts:
            # Get latest of each
            latest_visual = visual_percepts[-1]
            latest_text = text_percepts[-1]
            
            # Check if they are within a small time window
            visual_time = datetime.datetime.fromisoformat(latest_visual.get("timestamp", self.startup_time.isoformat()))
            text_time = datetime.datetime.fromisoformat(latest_text.get("timestamp", self.startup_time.isoformat()))
            
            if abs((visual_time - text_time).total_seconds()) < self.integration_window:
                # Record correlation
                self.modal_correlations["visual_text"] = {
                    "strength": 0.8,
                    "visual_timestamp": latest_visual.get("timestamp"),
                    "text_timestamp": latest_text.get("timestamp")
                }
        
        # Similarly for other modality pairs
        if text_percepts and somatic_percepts:
            latest_text = text_percepts[-1]
            latest_somatic = somatic_percepts[-1]
            
            text_time = datetime.datetime.fromisoformat(latest_text.get("timestamp", self.startup_time.isoformat()))
            somatic_time = datetime.datetime.fromisoformat(latest_somatic.get("timestamp", self.startup_time.isoformat()))
            
            if abs((text_time - somatic_time).total_seconds()) < self.integration_window:
                self.modal_correlations["text_somatic"] = {
                    "strength": 0.7,
                    "text_timestamp": latest_text.get("timestamp"),
                    "somatic_timestamp": latest_somatic.get("timestamp"),
                    "somatic_type": latest_somatic.get("sensation_type", "unknown")
                }
    
    async def _handle_sensory_input(self, event: Event) -> None:
        """
        Handle sensory input events from the event bus.
        
        Args:
            event: Sensory input event
        """
        try:
            # Extract event data
            modality = event.data.get("modality")
            content = event.data.get("content")
            
            if not modality or not content:
                return
            
            # Process the sensory input
            asyncio.create_task(self.process_sensory_input(content, modality))
        except Exception as e:
            logger.error(f"Error handling sensory input event: {e}")
    
    async def _handle_physical_sensation(self, event: Event) -> None:
        """
        Handle physical sensation events from the event bus.
        
        Args:
            event: Physical sensation event
        """
        try:
            # Extract event data
            region = event.data.get("region")
            sensation_type = event.data.get("sensation_type")
            intensity = event.data.get("intensity")
            cause = event.data.get("cause", "")
            
            if not region or not sensation_type or intensity is None:
                return
            
            # Integrate the somatosensory input
            asyncio.create_task(self.integrate_somatosensory_input(
                region, sensation_type, intensity, cause
            ))
        except Exception as e:
            logger.error(f"Error handling physical sensation event: {e}")
    
    async def _handle_user_interaction(self, event: Event) -> None:
        """
        Handle user interaction events from the event bus.
        
        Args:
            event: User interaction event
        """
        try:
            # Extract event data
            content = event.data.get("content")
            input_type = event.data.get("input_type", "text")
            
            if not content:
                return
            
            # Process as sensory input
            modality = input_type if input_type in ["text", "image", "audio"] else "text"
            asyncio.create_task(self.process_sensory_input(content, modality))
        except Exception as e:
            logger.error(f"Error handling user interaction event: {e}")

# Function to create the perceptual integration layer
def create_perceptual_integration_layer(brain_reference=None):
    """Create a perceptual integration layer for the given brain."""
    return PerceptualIntegrationLayer(brain_reference=brain_reference)
