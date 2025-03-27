# nyx/core/multimodal_integrator.py

import logging
import numpy as np
import datetime # Added for timestamp generation
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from pydantic import BaseModel, Field
import asyncio # Added for placeholder async functions

from agents import Agent, Runner, function_tool, RunContextWrapper

# --- Schemas (Keep existing SensoryInput, ExpectationSignal, IntegratedPercept) ---

class SensoryInput(BaseModel):
    """Schema for raw sensory input"""
    modality: str = Field(..., description="Input modality (text, image, video, audio_music, audio_speech, system_screen, system_audio)")
    data: Any = Field(..., description="Raw input data (e.g., text string, image bytes, audio chunk path/bytes, video frame path/bytes/stream)")
    confidence: float = Field(1.0, description="Input confidence (0.0-1.0)", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="Input timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., filename, source URL, camera ID, mic ID)")

class ExpectationSignal(BaseModel):
    """Schema for top-down expectation signal"""
    target_modality: str = Field(..., description="Target modality to influence")
    pattern: Any = Field(..., description="Expected pattern or feature (e.g., object name, specific sound, text content)")
    strength: float = Field(0.5, description="Signal strength (0.0-1.0)", ge=0.0, le=1.0)
    source: str = Field(..., description="Source of expectation (reasoning, memory, etc.)")
    priority: float = Field(0.5, description="Priority level (0.0-1.0)", ge=0.0, le=1.0)

class IntegratedPercept(BaseModel):
    """Schema for integrated percept after bottom-up and top-down processing"""
    modality: str = Field(..., description="Percept modality")
    content: Any = Field(..., description="Processed content (e.g., text summary, image description, audio features, recognized actions)")
    bottom_up_confidence: float = Field(..., description="Confidence from bottom-up processing")
    top_down_influence: float = Field(..., description="Degree of top-down influence")
    attention_weight: float = Field(..., description="Attentional weight applied")
    timestamp: str = Field(..., description="Processing timestamp")
    raw_features: Optional[Dict[str, Any]] = Field(None, description="Detailed features extracted during bottom-up processing") # Added for detail


# --- Feature Schemas (Examples - Define more detailed ones as needed) ---

class ImageFeatures(BaseModel):
    description: str = "No description available."
    objects: List[str] = []
    text_content: Optional[str] = None # OCR
    dominant_colors: List[Tuple[int, int, int]] = []
    estimated_mood: Optional[str] = None
    is_screenshot: bool = False

class AudioFeatures(BaseModel):
    type: str = "unknown" # music, speech, ambient, sound_effect
    transcription: Optional[str] = None # For speech
    speaker_id: Optional[str] = None # For speech
    mood: Optional[str] = None # For music/speech
    genre: Optional[str] = None # For music
    tempo_bpm: Optional[float] = None # For music
    key: Optional[str] = None # For music
    sound_events: List[str] = [] # For ambient/effects

class VideoFeatures(BaseModel):
    summary: str = "No summary available."
    key_actions: List[str] = []
    tracked_objects: Dict[str, List[Tuple[float, float]]] = {} # Object name -> list of (time, confidence)
    scene_changes: List[float] = [] # Timestamps of scene changes
    audio_features: Optional[AudioFeatures] = None # Embedded audio analysis
    estimated_mood_progression: List[Tuple[float, str]] = [] # List of (time, mood)


# --- Constants for Modalities ---
MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_VIDEO = "video"
MODALITY_AUDIO_MUSIC = "audio_music"
MODALITY_AUDIO_SPEECH = "audio_speech"
MODALITY_SYSTEM_SCREEN = "system_screen"
MODALITY_SYSTEM_AUDIO = "system_audio"


class EnhancedMultiModalIntegrator:
    """
    Processes sensory inputs using both bottom-up and top-down pathways.
    Handles text, image, video, audio, and system inputs.
    """

    def __init__(self, reasoning_core=None, attentional_controller=None):
        self.reasoning_core = reasoning_core
        self.attentional_controller = attentional_controller

        # Processing stages
        self.feature_extractors: Dict[str, callable] = {}
        self.expectation_modulators: Dict[str, callable] = {}
        self.integration_strategies: Dict[str, callable] = {}

        # Buffer for recent perceptions
        self.perception_buffer: List[IntegratedPercept] = []
        self.max_buffer_size = 100 # Increased buffer size

        # Current active expectations
        self.active_expectations: List[ExpectationSignal] = []

        self.logger = logging.getLogger(__name__)

        # --- Register Handlers for All Modalities ---
        self._register_handlers()

    def _register_handlers(self):
        """Register placeholder handlers for different modalities."""
        self.logger.info("Registering multimodal handlers...")

        # Text (Already partially handled, ensure registration)
        self.register_feature_extractor(MODALITY_TEXT, self._extract_text_features)
        self.register_expectation_modulator(MODALITY_TEXT, self._modulate_text_perception)
        self.register_integration_strategy(MODALITY_TEXT, self._integrate_text_pathways)

        # Image
        self.register_feature_extractor(MODALITY_IMAGE, self._extract_image_features)
        self.register_expectation_modulator(MODALITY_IMAGE, self._modulate_generic_perception) # Use generic for now
        self.register_integration_strategy(MODALITY_IMAGE, self._integrate_generic_pathways) # Use generic for now

        # Video
        self.register_feature_extractor(MODALITY_VIDEO, self._extract_video_features)
        self.register_expectation_modulator(MODALITY_VIDEO, self._modulate_generic_perception)
        self.register_integration_strategy(MODALITY_VIDEO, self._integrate_generic_pathways)

        # Audio (Music)
        self.register_feature_extractor(MODALITY_AUDIO_MUSIC, self._extract_audio_features)
        self.register_expectation_modulator(MODALITY_AUDIO_MUSIC, self._modulate_generic_perception)
        self.register_integration_strategy(MODALITY_AUDIO_MUSIC, self._integrate_generic_pathways)

        # Audio (Speech)
        self.register_feature_extractor(MODALITY_AUDIO_SPEECH, self._extract_audio_features)
        # Potentially use text modulator for transcribed speech? Or a dedicated one.
        self.register_expectation_modulator(MODALITY_AUDIO_SPEECH, self._modulate_speech_perception)
        self.register_integration_strategy(MODALITY_AUDIO_SPEECH, self._integrate_speech_pathways)

        # System Screen (Treat as Image/Video initially)
        self.register_feature_extractor(MODALITY_SYSTEM_SCREEN, self._extract_image_features) # Start with image features
        self.register_expectation_modulator(MODALITY_SYSTEM_SCREEN, self._modulate_generic_perception)
        self.register_integration_strategy(MODALITY_SYSTEM_SCREEN, self._integrate_generic_pathways)

        # System Audio (Treat as Speech/Ambient initially)
        self.register_feature_extractor(MODALITY_SYSTEM_AUDIO, self._extract_audio_features)
        self.register_expectation_modulator(MODALITY_SYSTEM_AUDIO, self._modulate_speech_perception) # Speech is often key
        self.register_integration_strategy(MODALITY_SYSTEM_AUDIO, self._integrate_speech_pathways)

        self.logger.info("Multimodal handlers registered.")


    # --- Registration Methods (Unchanged) ---
    async def register_feature_extractor(self, modality: str, extractor_function):
        """Register a feature extraction function for a specific modality"""
        self.feature_extractors[modality] = extractor_function

    async def register_expectation_modulator(self, modality: str, modulator_function):
        """Register a function that applies top-down expectations to a modality"""
        self.expectation_modulators[modality] = modulator_function

    async def register_integration_strategy(self, modality: str, integration_function):
        """Register a function that integrates bottom-up and top-down processing"""
        self.integration_strategies[modality] = integration_function

    # --- Core Processing Logic (Small modifications for logging/clarity) ---
    async def process_sensory_input(self,
                                   input_data: SensoryInput,
                                   expectations: List[ExpectationSignal] = None) -> IntegratedPercept:
        """
        Process sensory input using both bottom-up and top-down pathways
        Handles various modalities including vision and audio.

        Args:
            input_data: Raw sensory input object.
            expectations: Optional list of top-down expectations.

        Returns:
            Integrated percept combining bottom-up and top-down processing.
        """
        modality = input_data.modality
        timestamp = input_data.timestamp or datetime.datetime.now().isoformat() # Ensure timestamp exists
        self.logger.info(f"Processing sensory input: Modality={modality}, Timestamp={timestamp}")

        # 1. Bottom-up processing (data-driven)
        self.logger.debug(f"Performing bottom-up processing for {modality}...")
        bottom_up_result = await self._perform_bottom_up_processing(input_data)
        self.logger.debug(f"Bottom-up features for {modality}: {str(bottom_up_result['features'])[:200]}...") # Log snippet

        # 2. Get or use provided top-down expectations
        if expectations is None:
            self.logger.debug(f"Fetching active expectations for {modality}...")
            expectations = await self._get_active_expectations(modality)
            self.logger.debug(f"Found {len(expectations)} active expectations for {modality}.")
        else:
            self.logger.debug(f"Using {len(expectations)} provided expectations for {modality}.")


        # 3. Apply top-down modulation
        self.logger.debug(f"Applying top-down modulation for {modality}...")
        modulated_result = await self._apply_top_down_modulation(bottom_up_result, expectations, modality)
        self.logger.debug(f"Modulation influence for {modality}: {modulated_result['influence_strength']:.2f}")

        # 4. Integrate bottom-up and top-down pathways
        self.logger.debug(f"Integrating pathways for {modality}...")
        integrated_result = await self._integrate_pathways(bottom_up_result, modulated_result, modality)
        self.logger.debug(f"Integrated content for {modality}: {str(integrated_result['content'])[:200]}...")

        # 5. Apply attentional filtering if available
        attentional_weight = 1.0 # Default
        if self.attentional_controller:
            try:
                self.logger.debug(f"Calculating attention weight for {modality}...")
                # Ensure integrated_result provides necessary info for attention
                attention_input = {
                     "modality": modality,
                     "content_summary": str(integrated_result.get('content', ''))[:100], # Pass summary
                     "confidence": bottom_up_result.get("confidence", 1.0),
                     "metadata": input_data.metadata
                 }
                attentional_weight = await self.attentional_controller.calculate_attention_weight(
                    attention_input, expectations # Pass attention_input and expectations
                )
                self.logger.debug(f"Calculated attention weight for {modality}: {attentional_weight:.2f}")
            except Exception as e:
                 self.logger.error(f"Error calculating attention weight for {modality}: {e}")
                 attentional_weight = 0.5 # Fallback attention if error


        # 6. Create final percept
        percept = IntegratedPercept(
            modality=modality,
            content=integrated_result.get("content"), # Use .get for safety
            bottom_up_confidence=bottom_up_result.get("confidence", 0.0),
            top_down_influence=modulated_result.get("influence_strength", 0.0),
            attention_weight=attentional_weight,
            timestamp=timestamp,
            raw_features=bottom_up_result.get("features") # Store raw features for potential downstream use
        )

        # 7. Add to perception buffer
        self._add_to_perception_buffer(percept)

        # 8. Update reasoning core with new perception (if significant attention)
        if self.reasoning_core and attentional_weight > 0.3: # Lower threshold slightly
             try:
                 self.logger.debug(f"Updating reasoning core with significant percept ({modality}, weight={attentional_weight:.2f}).")
                 # Assume reasoning_core has an update method
                 await self.reasoning_core.update_with_perception(percept)
             except AttributeError:
                 self.logger.warning("Reasoning core does not have 'update_with_perception' method.")
             except Exception as e:
                 self.logger.error(f"Error updating reasoning core: {e}")


        self.logger.info(f"Finished processing sensory input for {modality}.")
        return percept

    # --- Bottom-Up Processing (Feature Extraction) ---

    async def _perform_bottom_up_processing(self, input_data: SensoryInput) -> Dict[str, Any]:
        """Extract features from raw sensory input using registered extractors."""
        modality = input_data.modality

        if modality in self.feature_extractors:
            try:
                extractor = self.feature_extractors[modality]
                features = await extractor(input_data.data, input_data.metadata) # Pass data and metadata
                return {
                    "modality": modality,
                    "features": features, # Should match modality-specific Feature Schema ideally
                    "confidence": input_data.confidence,
                    "metadata": input_data.metadata
                }
            except Exception as e:
                self.logger.exception(f"Error in bottom-up feature extraction for {modality}: {e}") # Use exception for traceback
                # Fallback with error info
                return {
                    "modality": modality,
                    "features": {"error": f"Feature extraction failed: {e}"},
                    "confidence": input_data.confidence * 0.5, # Lower confidence on error
                    "metadata": input_data.metadata
                }
        else:
            self.logger.warning(f"No feature extractor registered for modality: {modality}. Passing data through.")
            return {
                "modality": modality,
                "features": input_data.data, # Pass through raw data
                "confidence": input_data.confidence * 0.8, # Slightly lower confidence if no extractor
                "metadata": input_data.metadata
            }

    # --- Placeholder Feature Extractors ---
    # NOTE: These are placeholders. Real implementations would call external models/APIs.

    async def _extract_text_features(self, data: str, metadata: Dict) -> Dict:
         """ Placeholder: Extracts basic text features. """
         self.logger.debug(f"Placeholder: Extracting text features for: {data[:50]}...")
         # In reality: Use NLP library or model for sentiment, entities, keywords etc.
         word_count = len(data.split())
         char_count = len(data)
         return {"word_count": word_count, "char_count": char_count, "preview": data[:100]}

    async def _extract_image_features(self, data: Any, metadata: Dict) -> ImageFeatures:
        """ Placeholder: Extracts image features. Assumes data is image bytes or path. """
        self.logger.debug(f"Placeholder: Extracting image features (data type: {type(data)})...")
        # In reality: Use a vision model (e.g., GPT-4V API, local ViT/CLIP)
        # Input `data` could be bytes, file path, URL. Need handling.
        await asyncio.sleep(0.1) # Simulate processing time
        is_screenshot = MODALITY_SYSTEM_SCREEN in metadata.get("source_modality", "")
        return ImageFeatures(
            description="Placeholder image description.",
            objects=["object1", "placeholder"],
            text_content="OCR text placeholder" if is_screenshot else None,
            is_screenshot=is_screenshot
        )

    async def _extract_video_features(self, data: Any, metadata: Dict) -> VideoFeatures:
        """ Placeholder: Extracts video features. Assumes data is path, URL, or stream. """
        self.logger.debug(f"Placeholder: Extracting video features (data type: {type(data)})...")
        # In reality: Use video analysis model/API (action recognition, object tracking, summarization)
        # This is complex, might involve processing frames/segments.
        await asyncio.sleep(0.3) # Simulate longer processing time
        return VideoFeatures(
            summary="Placeholder video summary.",
            key_actions=["action1", "placeholder_event"],
            scene_changes=[5.2, 15.8]
        )

    async def _extract_audio_features(self, data: Any, metadata: Dict) -> AudioFeatures:
        """ Placeholder: Extracts audio features. Assumes data is path, bytes, or stream. """
        self.logger.debug(f"Placeholder: Extracting audio features (data type: {type(data)})...")
        # In reality: Use audio processing libs (librosa) and models (Whisper for STT, genre/mood classifiers)
        await asyncio.sleep(0.15) # Simulate processing time
        modality = metadata.get("source_modality", MODALITY_AUDIO_MUSIC) # Guess based on metadata if possible

        if modality == MODALITY_AUDIO_SPEECH or modality == MODALITY_SYSTEM_AUDIO:
             return AudioFeatures(
                 type="speech",
                 transcription="Placeholder speech transcription.",
                 mood="neutral"
             )
        elif modality == MODALITY_AUDIO_MUSIC:
             return AudioFeatures(
                 type="music",
                 mood="calm",
                 genre="ambient",
                 tempo_bpm=80.0
             )
        else: # Ambient/Effects
             return AudioFeatures(
                 type="ambient",
                 sound_events=["low_hum", "distant_chatter"]
             )

    # --- Top-Down Modulation ---

    async def _get_active_expectations(self, modality: str) -> List[ExpectationSignal]:
        """Get current active expectations, potentially querying reasoning core."""
        relevant_expectations = [exp for exp in self.active_expectations
                               if exp.target_modality == modality]

        # Optional: Query reasoning core for dynamic expectations (if reasoning core is sophisticated enough)
        # if self.reasoning_core and hasattr(self.reasoning_core, 'generate_perceptual_expectations'):
        #     try:
        #         self.logger.debug(f"Querying reasoning core for expectations ({modality})...")
        #         # This requires reasoning_core to have this method implemented
        #         new_expectations = await self.reasoning_core.generate_perceptual_expectations(modality)
        #         # ... (Add new expectations, prune old ones as before) ...
        #         self.logger.debug(f"Received {len(new_expectations)} new expectations.")
        #         relevant_expectations.extend([exp for exp in new_expectations if exp.target_modality == modality])
        #     except Exception as e:
        #         self.logger.error(f"Error getting expectations from reasoning core: {e}")

        # Sort by priority before returning
        relevant_expectations.sort(key=lambda x: x.priority, reverse=True)
        return relevant_expectations


    async def _apply_top_down_modulation(self,
                                       bottom_up_result: Dict[str, Any],
                                       expectations: List[ExpectationSignal],
                                       modality: str) -> Dict[str, Any]:
        """Apply top-down expectations using registered modulators."""
        if not expectations or modality not in self.expectation_modulators:
            # Return default if no expectations or no modulator
            return {
                "modality": modality,
                "features": bottom_up_result.get("features"),
                "influence_strength": 0.0,
                "influenced_by": []
            }

        try:
            modulator = self.expectation_modulators[modality]
            # Pass features extracted by bottom-up processing
            modulation_result = await modulator(bottom_up_result.get("features"), expectations)

            # Ensure result has expected keys
            return {
                "modality": modality,
                "features": modulation_result.get("features", bottom_up_result.get("features")),
                "influence_strength": modulation_result.get("influence_strength", 0.0),
                "influenced_by": modulation_result.get("influenced_by", [])
            }
        except Exception as e:
            self.logger.exception(f"Error in top-down modulation for {modality}: {e}")
            # Fallback to unmodulated result on error
            return {
                "modality": modality,
                "features": bottom_up_result.get("features"),
                "influence_strength": 0.0,
                "influenced_by": ["modulation_error"]
            }

    # --- Placeholder Modulators ---
    # NOTE: These are placeholders. Real ones would apply biases based on expectations.

    async def _modulate_text_perception(self, features: Dict, expectations: List[ExpectationSignal]) -> Dict:
         """ Placeholder: Applies text-specific expectations. """
         self.logger.debug("Placeholder: Modulating text perception...")
         influence_strength = 0.0
         influenced_by = []
         # Example: If expecting specific keywords, slightly boost confidence if found
         for exp in expectations:
             if isinstance(exp.pattern, str) and exp.pattern.lower() in features.get("preview","").lower():
                 influence_strength = max(influence_strength, exp.strength * 0.2) # Small influence
                 influenced_by.append(f"keyword_{exp.pattern[:10]}")

         return {"features": features, "influence_strength": min(influence_strength, 0.8), "influenced_by": influenced_by}

    async def _modulate_generic_perception(self, features: Any, expectations: List[ExpectationSignal]) -> Dict:
         """ Placeholder: Generic modulator, maybe adjusting overall confidence. """
         self.logger.debug("Placeholder: Modulating generic perception...")
         influence_strength = 0.0
         influenced_by = []
         # Example: Strong expectation might slightly increase confidence if features seem related
         if isinstance(features, dict) and expectations:
              # Very basic check if expectation pattern relates to description/objects
              for exp in expectations:
                   pattern_str = str(exp.pattern).lower()
                   desc = features.get("description", "").lower()
                   objs = features.get("objects", [])
                   if pattern_str in desc or any(pattern_str in obj.lower() for obj in objs):
                        influence_strength = max(influence_strength, exp.strength * 0.1)
                        influenced_by.append(f"pattern_match_{exp.pattern[:10]}")

         # Could modify features here based on expectations (e.g., bias description)
         modified_features = features
         return {"features": modified_features, "influence_strength": min(influence_strength, 0.7), "influenced_by": influenced_by}

    async def _modulate_speech_perception(self, features: AudioFeatures, expectations: List[ExpectationSignal]) -> Dict:
        """ Placeholder: Modulates speech, potentially biasing transcription. """
        self.logger.debug("Placeholder: Modulating speech perception...")
        influence_strength = 0.0
        influenced_by = []
        modified_features = features

        # Example: If expecting a certain speaker, bias speaker_id or confidence
        # Example: If expecting certain keywords, could inform the STT model (if supported)
        for exp in expectations:
            if "speaker" in str(exp.pattern).lower():
                 # Hypothetical: bias speaker ID if model supports it
                 influence_strength = max(influence_strength, exp.strength * 0.3)
                 influenced_by.append(f"speaker_expectation")
            elif isinstance(exp.pattern, str):
                 # Check if expected word is in transcription (if available)
                 if features.transcription and exp.pattern.lower() in features.transcription.lower():
                      influence_strength = max(influence_strength, exp.strength * 0.1)
                      influenced_by.append(f"keyword_{exp.pattern[:10]}")

        return {"features": modified_features, "influence_strength": min(influence_strength, 0.7), "influenced_by": influenced_by}

    # --- Pathway Integration ---

    async def _integrate_pathways(self,
                                bottom_up_result: Dict[str, Any],
                                top_down_result: Dict[str, Any],
                                modality: str) -> Dict[str, Any]:
        """Integrate pathways using registered strategies or default logic."""
        if modality in self.integration_strategies:
            try:
                integration_func = self.integration_strategies[modality]
                # Pass both full results to the integration function
                integrated_data = await integration_func(bottom_up_result, top_down_result)
                # Ensure the result is a dict with at least 'content'
                if isinstance(integrated_data, dict) and 'content' in integrated_data:
                    return integrated_data
                else:
                    self.logger.warning(f"Integration strategy for {modality} returned unexpected format. Using default.")
            except Exception as e:
                self.logger.exception(f"Error in pathway integration strategy for {modality}: {e}")
                # Fallback to default on error

        # --- Default Integration Logic ---
        # Use modulated features but retain original confidence; calculate ratios
        bottom_up_conf = bottom_up_result.get("confidence", 0.0)
        top_down_infl = top_down_result.get("influence_strength", 0.0)

        # Blend confidence slightly based on influence
        integrated_confidence = bottom_up_conf * (1.0 - top_down_infl * 0.5) + 0.5 * (top_down_infl * 0.5)

        # Use the *modulated* features as the primary content
        integrated_content = top_down_result.get("features")

        return {
            "content": integrated_content,
            "integrated_confidence": integrated_confidence,
            "bottom_up_features": bottom_up_result.get("features"), # Keep original bottom-up
            "top_down_features": top_down_result.get("features")   # Keep modulated features
        }

    # --- Placeholder Integration Strategies ---

    async def _integrate_text_pathways(self, bottom_up: Dict, top_down: Dict) -> Dict:
         """ Placeholder: Simple text integration (uses modulated features). """
         self.logger.debug("Placeholder: Integrating text pathways...")
         # Use modulated features, maybe adjust confidence based on influence
         confidence = bottom_up.get("confidence", 1.0)
         influence = top_down.get("influence_strength", 0.0)
         integrated_conf = confidence * (1.0 - influence*0.3) # Slightly reduce conf if heavily modulated

         return {
             "content": top_down.get("features", bottom_up.get("features")), # Prioritize top-down if available
             "integrated_confidence": integrated_conf,
             "bottom_up_features": bottom_up.get("features"),
             "top_down_features": top_down.get("features"),
         }

    async def _integrate_generic_pathways(self, bottom_up: Dict, top_down: Dict) -> Dict:
         """ Placeholder: Generic integration (uses modulated features). """
         self.logger.debug("Placeholder: Integrating generic pathways...")
         confidence = bottom_up.get("confidence", 1.0)
         influence = top_down.get("influence_strength", 0.0)
         integrated_conf = confidence * (1.0 - influence*0.2)

         # For complex features (like ImageFeatures), the modulated features *are* the integrated content
         return {
             "content": top_down.get("features", bottom_up.get("features")),
             "integrated_confidence": integrated_conf,
             "bottom_up_features": bottom_up.get("features"),
             "top_down_features": top_down.get("features"),
         }

    async def _integrate_speech_pathways(self, bottom_up: Dict, top_down: Dict) -> Dict:
        """ Placeholder: Integrates speech, perhaps combining biased transcription. """
        self.logger.debug("Placeholder: Integrating speech pathways...")
        confidence = bottom_up.get("confidence", 1.0)
        influence = top_down.get("influence_strength", 0.0)
        integrated_conf = confidence * (1.0 - influence*0.2)

        # If top-down modified the transcription, use that one.
        # If not, use bottom-up. Combine other features potentially.
        bu_features = bottom_up.get("features")
        td_features = top_down.get("features")

        integrated_content = td_features if td_features else bu_features

        return {
            "content": integrated_content, # Usually an AudioFeatures object
            "integrated_confidence": integrated_conf,
            "bottom_up_features": bu_features,
            "top_down_features": td_features,
        }

    # --- Buffer and Reasoning Update Methods (Unchanged) ---
    def _add_to_perception_buffer(self, percept: IntegratedPercept):
        """Add percept to buffer, removing oldest if full"""
        self.perception_buffer.append(percept)

        if len(self.perception_buffer) > self.max_buffer_size:
            self.perception_buffer.pop(0)

    async def _update_reasoning_with_perception(self, percept: IntegratedPercept):
        """Update reasoning core with significant perception"""
        # Added check for method existence
        if self.reasoning_core and hasattr(self.reasoning_core, 'update_with_perception'):
            try:
                await self.reasoning_core.update_with_perception(percept)
            except Exception as e:
                self.logger.error(f"Error updating reasoning with perception: {str(e)}")
        elif self.reasoning_core:
             self.logger.warning("Reasoning core does not have 'update_with_perception' method.")


    # --- Expectation Management (Unchanged) ---
    async def add_expectation(self, expectation: ExpectationSignal):
        """Add a new top-down expectation"""
        # Add basic validation
        if not isinstance(expectation, ExpectationSignal):
             self.logger.warning("Attempted to add invalid expectation type.")
             return
        self.active_expectations.append(expectation)
        # Prune if too many expectations
        if len(self.active_expectations) > 50: # Limit active expectations
            self.active_expectations.sort(key=lambda x: x.priority, reverse=True)
            self.active_expectations = self.active_expectations[:40] # Keep top 40


    async def clear_expectations(self, modality: str = None):
        """Clear active expectations, optionally for a specific modality"""
        if modality:
            self.active_expectations = [exp for exp in self.active_expectations
                                      if exp.target_modality != modality]
            self.logger.info(f"Cleared expectations for modality: {modality}")
        else:
            self.active_expectations = []
            self.logger.info("Cleared all active expectations.")

    # --- Retrieval Method (Unchanged) ---
    async def get_recent_percepts(self, modality: str = None, limit: int = 10) -> List[IntegratedPercept]:
        """Get recent percepts, optionally filtered by modality"""
        buffer = self.perception_buffer
        if modality:
            filtered = [p for p in buffer if p.modality == modality]
            return filtered[-limit:]
        else:
            return buffer[-limit:]

# --- How NyxBrain would use it (Conceptual Example) ---
# class NyxBrain:
#     # ... (initialization of multimodal_integrator) ...

#     async def process_image_input(self, image_bytes: bytes, metadata: Dict = None):
#         input_obj = SensoryInput(
#             modality=MODALITY_IMAGE,
#             data=image_bytes, # Pass raw bytes
#             timestamp=datetime.datetime.now().isoformat(),
#             metadata=metadata or {}
#         )
#         percept = await self.multimodal_integrator.process_sensory_input(input_obj)
#         self.logger.info(f"Processed image, percept content: {percept.content}")
#         # Further processing based on percept...
#         await self.emotional_core.update_from_stimuli(...) # Update based on image mood?
#         await self.memory_core.add_memory(memory_text=f"Saw an image: {percept.content.description}", ...)

#     async def process_audio_chunk(self, audio_bytes: bytes, metadata: Dict = None):
#         # Determine if speech or music (maybe based on metadata or a quick classifier)
#         modality = MODALITY_AUDIO_SPEECH # Default to speech
#         if metadata and metadata.get("content_type") == "music":
#             modality = MODALITY_AUDIO_MUSIC
#         elif metadata and metadata.get("source") == "system_mic":
#             modality = MODALITY_SYSTEM_AUDIO

#         input_obj = SensoryInput(
#             modality=modality,
#             data=audio_bytes,
#             timestamp=datetime.datetime.now().isoformat(),
#             metadata=metadata or {}
#         )
#         percept = await self.multimodal_integrator.process_sensory_input(input_obj)
#         self.logger.info(f"Processed audio ({modality}), percept content: {percept.content}")
#         # If speech, maybe process transcription as text input?
#         if percept.content and percept.content.transcription:
#             await self.process_input(percept.content.transcription, context={"source": "audio_transcription"})

#     async def capture_and_process_screen(self):
#         # 1. Use an OS-specific library to capture the screen (e.g., pyautogui, mss)
#         # screen_bytes = capture_screen_function()
#         screen_bytes = b"dummy_screen_bytes" # Placeholder
#         if not screen_bytes: return

#         # 2. Feed into integrator
#         input_obj = SensoryInput(
#             modality=MODALITY_SYSTEM_SCREEN,
#             data=screen_bytes,
#             timestamp=datetime.datetime.now().isoformat(),
#             metadata={"source_modality": MODALITY_SYSTEM_SCREEN} # Add metadata
#         )
#         percept = await self.multimodal_integrator.process_sensory_input(input_obj)
#         self.logger.info(f"Processed system screen, description: {percept.content.description}")
#         # React based on screen content (e.g., identify application, read text)
