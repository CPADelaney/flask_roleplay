# nyx/core/multimodal_integrator.py

import logging
import numpy as np
import datetime
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

# --- Constants for Modalities ---
class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO_MUSIC = "audio_music"
    AUDIO_SPEECH = "audio_speech"
    SYSTEM_SCREEN = "system_screen"
    SYSTEM_AUDIO = "system_audio"
    TOUCH_EVENT = "touch_event"
    TASTE = "taste"
    SMELL = "smell"

# --- Base Schemas ---

class SensoryInput(BaseModel):
    """Schema for raw sensory input"""
    modality: Modality
    data: Any = Field(..., description="Raw input data (e.g., text string, image bytes, audio data)")
    confidence: float = Field(1.0, description="Input confidence (0.0-1.0)", ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ExpectationSignal(BaseModel):
    """Schema for top-down expectation signal"""
    target_modality: Modality
    pattern: Any = Field(..., description="Expected pattern or feature (e.g., object name, specific sound, text content)")
    strength: float = Field(0.5, description="Signal strength (0.0-1.0)", ge=0.0, le=1.0)
    source: str = Field(..., description="Source of expectation (reasoning, memory, etc.)")
    priority: float = Field(0.5, description="Priority level (0.0-1.0)", ge=0.0, le=1.0)

class IntegratedPercept(BaseModel):
    """Schema for integrated percept after bottom-up and top-down processing"""
    modality: Modality
    content: Any = Field(..., description="Processed content (features, descriptions, etc.)")
    bottom_up_confidence: float = Field(..., description="Confidence from bottom-up processing")
    top_down_influence: float = Field(..., description="Degree of top-down influence")
    attention_weight: float = Field(..., description="Attentional weight applied")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    raw_features: Optional[Dict[str, Any]] = Field(None, description="Detailed features from bottom-up processing")

# --- Feature Schemas for Different Modalities ---

class ImageFeatures(BaseModel):
    """Features extracted from image data."""
    description: str = Field(default="No description available.")
    objects: List[str] = Field(default_factory=list)
    text_content: Optional[str] = None  # OCR
    dominant_colors: List[Tuple[int, int, int]] = Field(default_factory=list)
    estimated_mood: Optional[str] = None
    is_screenshot: bool = False
    spatial_layout: Optional[Dict[str, Any]] = None  # Positions of detected objects

class AudioFeatures(BaseModel):
    """Features extracted from audio data."""
    type: str = "unknown"  # music, speech, ambient, sound_effect
    transcription: Optional[str] = None  # For speech
    speaker_id: Optional[str] = None  # For speech
    mood: Optional[str] = None  # For music/speech
    genre: Optional[str] = None  # For music
    tempo_bpm: Optional[float] = None  # For music
    key: Optional[str] = None  # For music
    sound_events: List[str] = Field(default_factory=list)  # For ambient/effects
    noise_level: Optional[float] = None  # Estimated background noise

class VideoFeatures(BaseModel):
    """Features extracted from video data."""
    summary: str = "No summary available."
    key_actions: List[str] = Field(default_factory=list)
    tracked_objects: Dict[str, List[Tuple[float, float]]] = Field(default_factory=dict)  # Object name -> list of (time, confidence)
    scene_changes: List[float] = Field(default_factory=list)  # Timestamps of scene changes
    audio_features: Optional[AudioFeatures] = None  # Embedded audio analysis
    estimated_mood_progression: List[Tuple[float, str]] = Field(default_factory=list)  # List of (time, mood)
    significant_frames: List[Dict[str, Any]] = Field(default_factory=list)  # Key moments

class TouchEventFeatures(BaseModel):
    """Features extracted from touch event data."""
    region: str
    texture: Optional[str] = None  # e.g., smooth, rough, soft, sticky, wet
    temperature: Optional[str] = None  # e.g., warm, cool, hot, cold
    pressure_level: Optional[float] = None  # 0.0-1.0
    hardness: Optional[str] = None  # e.g., soft, firm, hard
    shape: Optional[str] = None  # e.g., flat, round, sharp
    object_description: Optional[str] = None  # What was touched

class TasteFeatures(BaseModel):
    """Features extracted from taste data."""
    profiles: List[str] = Field(default_factory=list)  # e.g., sweet, sour, bitter, salty, umami, fatty, metallic, fruity, spicy
    intensity: float = 0.5  # 0.0-1.0
    texture: Optional[str] = None  # e.g., creamy, crunchy, chewy, liquid
    temperature: Optional[str] = None  # e.g., hot, cold, room
    source_description: Optional[str] = None

class SmellFeatures(BaseModel):
    """Features extracted from smell data."""
    profiles: List[str] = Field(default_factory=list)  # e.g., floral, fruity, citrus, woody, spicy, fresh, pungent, earthy, chemical, sweet, rotten
    intensity: float = 0.5  # 0.0-1.0
    pleasantness: Optional[float] = None  # Estimated pleasantness -1.0 to 1.0
    source_description: Optional[str] = None

# --- Constants for Feature Classification ---
POSITIVE_TASTES = {"sweet", "umami", "fatty", "savory"}
NEGATIVE_TASTES = {"bitter", "sour", "metallic", "spoiled"}
POSITIVE_SMELLS = {"floral", "fruity", "sweet", "fresh", "baked", "woody", "earthy"}
NEGATIVE_SMELLS = {"pungent", "chemical", "rotten", "sour", "fishy", "burnt"}

class MultimodalIntegrator:
    """
    Processes sensory inputs using both bottom-up and top-down pathways.
    Handles text, image, video, audio, and other sensory modalities.
    """

    def __init__(self, reasoning_core=None, attentional_controller=None, vision_model=None, audio_processor=None):
        """
        Initialize the multimodal integrator.
        
        Args:
            reasoning_core: Optional reference to the reasoning core for top-down processing
            attentional_controller: Optional reference to the attentional controller for filtering
            vision_model: Optional reference to vision model for image/video processing
            audio_processor: Optional reference to audio processor for audio processing
        """
        self.reasoning_core = reasoning_core
        self.attentional_controller = attentional_controller
        self.vision_model = vision_model
        self.audio_processor = audio_processor
        
        # Integration agent (for analyzing percepts)
        self.integration_agent = self._create_integration_agent()

        # Processing stages by modality
        self.feature_extractors: Dict[Modality, callable] = {}
        self.expectation_modulators: Dict[Modality, callable] = {}
        self.integration_strategies: Dict[Modality, callable] = {}

        # Buffer for recent perceptions
        self.perception_buffer: List[IntegratedPercept] = []
        self.max_buffer_size = 100
        
        # Track perception stats
        self.perception_stats: Dict[Modality, Dict[str, int]] = {
            modality: {"count": 0, "attention_filtered": 0} for modality in Modality
        }

        # Current active expectations
        self.active_expectations: List[ExpectationSignal] = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Set up handlers for all modalities
        self._register_handlers()
        
        logger.info("MultimodalIntegrator initialized")

    def _create_integration_agent(self) -> Optional[Agent]:
        """Creates an agent for analyzing and integrating multimodal percepts."""
        try:
            return Agent(
                name="Multimodal Integration Agent",
                instructions="""You analyze sensory percepts and integrate information across modalities.
                
                Your tasks include:
                1. Analyzing features extracted from different sensory modalities (vision, audio, touch, etc.)
                2. Identifying patterns and connections between sensory inputs
                3. Creating coherent descriptions that integrate information from multiple sources
                4. Resolving conflicts or ambiguities between different modalities
                5. Translating sensory features into semantically meaningful information
                
                For different modalities, focus on:
                - Vision: Identify objects, relationships, scene context, and visual properties
                - Audio: Understand speech content, emotional tone, music characteristics, or ambient sounds
                - Touch: Interpret tactile properties like texture, temperature, and pressure
                - Other senses: Extract meaningful patterns and properties
                
                Respond with structured JSON that enhances the raw features with meaningful interpretations.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(
                    temperature=0.2,
                    response_format={"type": "json_object"}
                ),
                output_type=Dict[str, Any]
            )
        except Exception as e:
            logger.error(f"Error creating integration agent: {e}")
            return None

    def _register_handlers(self):
        """Register processing functions for all modalities."""
        logger.info("Registering multimodal handlers")

        # Text modality
        self.feature_extractors[Modality.TEXT] = self._extract_text_features
        self.expectation_modulators[Modality.TEXT] = self._modulate_text_perception
        self.integration_strategies[Modality.TEXT] = self._integrate_text_pathways

        # Image modality
        self.feature_extractors[Modality.IMAGE] = self._extract_image_features
        self.expectation_modulators[Modality.IMAGE] = self._modulate_generic_perception
        self.integration_strategies[Modality.IMAGE] = self._integrate_generic_pathways

        # Video modality
        self.feature_extractors[Modality.VIDEO] = self._extract_video_features
        self.expectation_modulators[Modality.VIDEO] = self._modulate_generic_perception
        self.integration_strategies[Modality.VIDEO] = self._integrate_generic_pathways

        # Audio (Music) modality
        self.feature_extractors[Modality.AUDIO_MUSIC] = self._extract_audio_features
        self.expectation_modulators[Modality.AUDIO_MUSIC] = self._modulate_generic_perception
        self.integration_strategies[Modality.AUDIO_MUSIC] = self._integrate_generic_pathways

        # Audio (Speech) modality
        self.feature_extractors[Modality.AUDIO_SPEECH] = self._extract_audio_features
        self.expectation_modulators[Modality.AUDIO_SPEECH] = self._modulate_speech_perception
        self.integration_strategies[Modality.AUDIO_SPEECH] = self._integrate_speech_pathways

        # System Screen modality
        self.feature_extractors[Modality.SYSTEM_SCREEN] = self._extract_image_features
        self.expectation_modulators[Modality.SYSTEM_SCREEN] = self._modulate_generic_perception
        self.integration_strategies[Modality.SYSTEM_SCREEN] = self._integrate_generic_pathways

        # System Audio modality
        self.feature_extractors[Modality.SYSTEM_AUDIO] = self._extract_audio_features
        self.expectation_modulators[Modality.SYSTEM_AUDIO] = self._modulate_speech_perception
        self.integration_strategies[Modality.SYSTEM_AUDIO] = self._integrate_speech_pathways

        # Touch Event modality
        self.feature_extractors[Modality.TOUCH_EVENT] = self._extract_touch_event_features
        self.expectation_modulators[Modality.TOUCH_EVENT] = self._modulate_generic_perception
        self.integration_strategies[Modality.TOUCH_EVENT] = self._integrate_generic_pathways

        # Taste modality
        self.feature_extractors[Modality.TASTE] = self._extract_taste_features
        self.expectation_modulators[Modality.TASTE] = self._modulate_generic_perception
        self.integration_strategies[Modality.TASTE] = self._integrate_generic_pathways

        # Smell modality
        self.feature_extractors[Modality.SMELL] = self._extract_smell_features
        self.expectation_modulators[Modality.SMELL] = self._modulate_generic_perception
        self.integration_strategies[Modality.SMELL] = self._integrate_generic_pathways

        logger.info("All multimodal handlers registered")

    async def process_sensory_input(self,
                                   input_data: SensoryInput,
                                   expectations: Optional[List[ExpectationSignal]] = None) -> IntegratedPercept:
        """
        Process sensory input using both bottom-up and top-down pathways.
        Handles various modalities including vision and audio.

        Args:
            input_data: Raw sensory input object.
            expectations: Optional list of top-down expectations.

        Returns:
            Integrated percept combining bottom-up and top-down processing.
        """
        modality = input_data.modality
        timestamp = input_data.timestamp or datetime.datetime.now().isoformat()
        logger.info(f"Processing sensory input: Modality={modality}, Timestamp={timestamp}")
        
        # Update stats
        async with self._lock:
            self.perception_stats[modality]["count"] += 1

        # 1. Bottom-up processing (data-driven)
        logger.debug(f"Performing bottom-up processing for {modality}")
        bottom_up_result = await self._perform_bottom_up_processing(input_data)
        logger.debug(f"Bottom-up features for {modality} extracted")

        # 2. Get or use provided top-down expectations
        if expectations is None:
            logger.debug(f"Fetching active expectations for {modality}")
            expectations = await self._get_active_expectations(modality)
            logger.debug(f"Found {len(expectations)} active expectations for {modality}")
        else:
            logger.debug(f"Using {len(expectations)} provided expectations for {modality}")

        # 3. Apply top-down modulation
        logger.debug(f"Applying top-down modulation for {modality}")
        modulated_result = await self._apply_top_down_modulation(bottom_up_result, expectations, modality)
        logger.debug(f"Modulation influence for {modality}: {modulated_result['influence_strength']:.2f}")

        # 4. Integrate bottom-up and top-down pathways
        logger.debug(f"Integrating pathways for {modality}")
        integrated_result = await self._integrate_pathways(bottom_up_result, modulated_result, modality)
        logger.debug(f"Integrated content for {modality}")

        # 5. Apply attentional filtering if available
        attentional_weight = 1.0  # Default
        if self.attentional_controller:
            try:
                logger.debug(f"Calculating attention weight for {modality}")
                # Prepare attention input
                attention_input = {
                    "modality": modality,
                    "content_summary": self._get_content_summary(integrated_result.get('content', {})),
                    "confidence": bottom_up_result.get("confidence", 1.0),
                    "metadata": input_data.metadata
                }
                
                attentional_weight = await self.attentional_controller.calculate_attention_weight(
                    attention_input, expectations
                )
                logger.debug(f"Calculated attention weight for {modality}: {attentional_weight:.2f}")
                
                # Update stats for filtered percepts
                if attentional_weight < 0.2:  # Threshold for considering "filtered"
                    async with self._lock:
                        self.perception_stats[modality]["attention_filtered"] += 1
                    
            except Exception as e:
                logger.error(f"Error calculating attention weight for {modality}: {e}")
                attentional_weight = 0.5  # Fallback attention if error

        # 6. Create final percept
        percept = IntegratedPercept(
            modality=modality,
            content=integrated_result.get("content"),
            bottom_up_confidence=bottom_up_result.get("confidence", 0.0),
            top_down_influence=modulated_result.get("influence_strength", 0.0),
            attention_weight=attentional_weight,
            timestamp=timestamp,
            raw_features=bottom_up_result.get("features")
        )

        # 7. Add to perception buffer
        await self._add_to_perception_buffer(percept)

        # 8. Update reasoning core with new perception (if significant attention)
        if self.reasoning_core and attentional_weight > 0.3:
            try:
                logger.debug(f"Updating reasoning core with significant percept ({modality}, weight={attentional_weight:.2f})")
                # Check if reasoning core has update method
                if hasattr(self.reasoning_core, 'update_with_perception'):
                    await self.reasoning_core.update_with_perception(percept)
            except Exception as e:
                logger.error(f"Error updating reasoning core: {e}")

        logger.info(f"Finished processing sensory input for {modality}")
        return percept
    
    def _get_content_summary(self, content: Any) -> str:
        """Get a short summary string of content for attention calculations."""
        if isinstance(content, str):
            return content[:100]
        elif isinstance(content, dict):
            if "description" in content:
                return content["description"][:100]
            elif "summary" in content:
                return content["summary"][:100]
            else:
                return str(list(content.keys()))[:100]
        elif isinstance(content, (list, tuple)):
            return str(content[:3])[:100]
        else:
            return str(content)[:100]

    async def _add_to_perception_buffer(self, percept: IntegratedPercept):
        """Add percept to buffer, maintaining thread safety."""
        async with self._lock:
            self.perception_buffer.append(percept)
            if len(self.perception_buffer) > self.max_buffer_size:
                self.perception_buffer.pop(0)

    async def _perform_bottom_up_processing(self, input_data: SensoryInput) -> Dict[str, Any]:
        """Extract features from raw sensory input using registered extractors."""
        modality = input_data.modality

        if modality in self.feature_extractors:
            try:
                extractor = self.feature_extractors[modality]
                features = await extractor(input_data.data, input_data.metadata)
                return {
                    "modality": modality,
                    "features": features,
                    "confidence": input_data.confidence,
                    "metadata": input_data.metadata
                }
            except Exception as e:
                logger.exception(f"Error in bottom-up feature extraction for {modality}: {e}")
                # Fallback with error info
                return {
                    "modality": modality,
                    "features": {"error": f"Feature extraction failed: {e}"},
                    "confidence": input_data.confidence * 0.5,  # Lower confidence on error
                    "metadata": input_data.metadata
                }
        else:
            logger.warning(f"No feature extractor registered for modality: {modality}. Passing data through.")
            return {
                "modality": modality,
                "features": input_data.data,  # Pass through raw data
                "confidence": input_data.confidence * 0.8,  # Slightly lower confidence if no extractor
                "metadata": input_data.metadata
            }

    # --- Feature Extraction Methods ---

    async def _extract_text_features(self, data: str, metadata: Dict) -> Dict:
        """Extract features from text input."""
        logger.debug(f"Extracting text features: {data[:50]}...")
        
        # If using vision model and it supports text, use that
        if self.vision_model and hasattr(self.vision_model, 'analyze_text'):
            try:
                vision_results = await self.vision_model.analyze_text(data)
                return vision_results
            except Exception as e:
                logger.error(f"Error using vision model for text analysis: {e}")
        
        # Basic text analysis
        word_count = len(data.split())
        char_count = len(data)
        
        # Simple sentiment analysis (basic keyword search)
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'happy', 'joy', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'hate', 'dislike']
        
        text_lower = data.lower()
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        sentiment = 0.0
        if positive_count + negative_count > 0:
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        
        # Check for questions
        is_question = '?' in data
        
        return {
            "word_count": word_count, 
            "char_count": char_count,
            "preview": data[:100],
            "sentiment": sentiment,
            "is_question": is_question
        }

    async def _extract_image_features(self, data: Any, metadata: Dict) -> ImageFeatures:
        """Extract features from image data."""
        logger.debug(f"Extracting image features (data type: {type(data)})...")
        
        # Use vision model if available
        if self.vision_model and hasattr(self.vision_model, 'analyze_image'):
            try:
                vision_results = await self.vision_model.analyze_image(data)
                if isinstance(vision_results, dict):
                    # Convert to ImageFeatures
                    is_screenshot = metadata.get("source_modality") == Modality.SYSTEM_SCREEN
                    
                    # Extract text content from OCR if available
                    text_content = None
                    if "text_content" in vision_results:
                        text_content = vision_results["text_content"]
                    elif "ocr_text" in vision_results:
                        text_content = vision_results["ocr_text"]
                    
                    return ImageFeatures(
                        description=vision_results.get("description", "No description available"),
                        objects=vision_results.get("objects", []),
                        text_content=text_content,
                        dominant_colors=vision_results.get("dominant_colors", []),
                        estimated_mood=vision_results.get("mood", None),
                        is_screenshot=is_screenshot,
                        spatial_layout=vision_results.get("spatial_layout", None)
                    )
                else:
                    return vision_results  # If already an ImageFeatures object
            except Exception as e:
                logger.error(f"Error using vision model: {e}")
        
        # Fallback to basic processing
        await asyncio.sleep(0.1)  # Simulate processing time
        is_screenshot = metadata.get("source_modality") == Modality.SYSTEM_SCREEN
        
        return ImageFeatures(
            description="[Placeholder image description - vision model unavailable]",
            objects=["placeholder_object"],
            text_content="[OCR unavailable]" if is_screenshot else None,
            is_screenshot=is_screenshot
        )

    async def _extract_video_features(self, data: Any, metadata: Dict) -> VideoFeatures:
        """Extract features from video data."""
        logger.debug(f"Extracting video features (data type: {type(data)})...")
        
        # Use vision model if available for key frames
        if self.vision_model and hasattr(self.vision_model, 'analyze_video'):
            try:
                vision_results = await self.vision_model.analyze_video(data)
                return vision_results  # Assuming it returns a VideoFeatures object
            except Exception as e:
                logger.error(f"Error using vision model for video: {e}")
        
        # Use audio processor for audio track if available
        audio_features = None
        if self.audio_processor and hasattr(self.audio_processor, 'extract_audio_from_video'):
            try:
                audio_data = await self.audio_processor.extract_audio_from_video(data)
                audio_features = await self._extract_audio_features(audio_data, metadata)
            except Exception as e:
                logger.error(f"Error extracting audio from video: {e}")
        
        # Fallback placeholder
        await asyncio.sleep(0.3)  # Simulate longer processing time
        return VideoFeatures(
            summary="[Placeholder video summary - video processing unavailable]",
            key_actions=["placeholder_action"],
            scene_changes=[5.2, 15.8],
            audio_features=audio_features
        )

    async def _extract_audio_features(self, data: Any, metadata: Dict) -> AudioFeatures:
        """Extract features from audio data."""
        logger.debug(f"Extracting audio features (data type: {type(data)})...")
        
        # Use audio processor if available
        if self.audio_processor:
            try:
                if hasattr(self.audio_processor, 'analyze_speech') and metadata.get("source_modality") in [Modality.AUDIO_SPEECH, Modality.SYSTEM_AUDIO]:
                    return await self.audio_processor.analyze_speech(data)
                elif hasattr(self.audio_processor, 'analyze_music') and metadata.get("source_modality") == Modality.AUDIO_MUSIC:
                    return await self.audio_processor.analyze_music(data)
                elif hasattr(self.audio_processor, 'analyze_audio'):
                    return await self.audio_processor.analyze_audio(data)
            except Exception as e:
                logger.error(f"Error using audio processor: {e}")
        
        # Fallback placeholder
        await asyncio.sleep(0.15)  # Simulate processing time
        modality = metadata.get("source_modality", Modality.AUDIO_MUSIC)
        
        if modality in [Modality.AUDIO_SPEECH, Modality.SYSTEM_AUDIO]:
            return AudioFeatures(
                type="speech",
                transcription="[Placeholder speech transcription - audio processing unavailable]",
                mood="neutral"
            )
        elif modality == Modality.AUDIO_MUSIC:
            return AudioFeatures(
                type="music",
                mood="calm",
                genre="ambient",
                tempo_bpm=80.0
            )
        else:  # Ambient/Effects
            return AudioFeatures(
                type="ambient",
                sound_events=["background_noise", "indistinct_sounds"]
            )

    async def _extract_touch_event_features(self, data: Dict, metadata: Dict) -> TouchEventFeatures:
        """Extract features from touch event data."""
        logger.debug(f"Extracting touch event features: {data}")
        
        # Validate input data
        if not isinstance(data, dict):
            logger.warning("Touch event data is not a dict, cannot parse features.")
            return TouchEventFeatures(region="unknown", object_description="Parsing error")
        
        # Extract basic touch properties
        try:
            # Convert dictionary to TouchEventFeatures model
            return TouchEventFeatures(**data)
        except Exception as e:
            logger.error(f"Error parsing touch event data: {e}")
            # Provide minimal fallback
            return TouchEventFeatures(
                region=data.get("region", "unknown"),
                object_description=data.get("object_description", "Error parsing touch data")
            )

    async def _extract_taste_features(self, data: Dict, metadata: Dict) -> TasteFeatures:
        """Extract features from taste data."""
        logger.debug(f"Extracting taste features: {data}")
        
        # Validate input data
        if not isinstance(data, dict):
            logger.warning("Taste data is not a dict, cannot parse features.")
            return TasteFeatures(profiles=["unknown"], source_description="Parsing error")
        
        try:
            # Convert dictionary to TasteFeatures model
            return TasteFeatures(**data)
        except Exception as e:
            logger.error(f"Error parsing taste data: {e}")
            # Provide minimal fallback
            return TasteFeatures(
                profiles=data.get("profiles", ["unknown"]),
                intensity=data.get("intensity", 0.5),
                source_description=data.get("source_description", "Error parsing taste data")
            )

    async def _extract_smell_features(self, data: Dict, metadata: Dict) -> SmellFeatures:
        """Extract features from smell data."""
        logger.debug(f"Extracting smell features: {data}")
        
        # Validate input data
        if not isinstance(data, dict):
            logger.warning("Smell data is not a dict, cannot parse features.")
            return SmellFeatures(profiles=["unknown"], source_description="Parsing error")
        
        try:
            # Convert dictionary to SmellFeatures
            features = SmellFeatures(**data)
            
            # Add pleasantness estimation if not provided
            if features.pleasantness is None:
                pos_count = sum(1 for p in features.profiles if p in POSITIVE_SMELLS)
                neg_count = sum(1 for p in features.profiles if p in NEGATIVE_SMELLS)
                if pos_count + neg_count > 0:
                    features.pleasantness = (pos_count - neg_count) / (pos_count + neg_count)
                else:
                    features.pleasantness = 0.0  # Neutral if no known profiles
            
            return features
        except Exception as e:
            logger.error(f"Error parsing smell data: {e}")
            # Provide minimal fallback
            return SmellFeatures(
                profiles=data.get("profiles", ["unknown"]),
                intensity=data.get("intensity", 0.5),
                source_description=data.get("source_description", "Error parsing smell data")
            )

    # --- Top-Down Modulation Methods ---

    async def _get_active_expectations(self, modality: Modality) -> List[ExpectationSignal]:
        """Get current active expectations for a specific modality."""
        # Filter expectations relevant to this modality
        relevant_expectations = [exp for exp in self.active_expectations
                              if exp.target_modality == modality]
        
        # If reasoning core can provide expectations, query it
        if self.reasoning_core and hasattr(self.reasoning_core, 'generate_perceptual_expectations'):
            try:
                logger.debug(f"Querying reasoning core for expectations ({modality})")
                new_expectations = await self.reasoning_core.generate_perceptual_expectations(modality)
                logger.debug(f"Received {len(new_expectations)} new expectations")
                
                # Add new expectations to the list
                relevant_expectations.extend(new_expectations)
                
                # Add new expectations to active list for future use
                self.active_expectations.extend(new_expectations)
                
                # Prune old expectations if we have too many
                if len(self.active_expectations) > 50:
                    self.active_expectations.sort(key=lambda x: x.priority, reverse=True)
                    self.active_expectations = self.active_expectations[:40]  # Keep top 40
                    
            except Exception as e:
                logger.error(f"Error getting expectations from reasoning core: {e}")
        
        # Sort by priority
        if relevant_expectations:
            relevant_expectations.sort(key=lambda x: x.priority, reverse=True)
            
        return relevant_expectations

    async def _apply_top_down_modulation(self, bottom_up_result: Dict[str, Any],
                                       expectations: List[ExpectationSignal],
                                       modality: Modality) -> Dict[str, Any]:
        """Apply top-down expectations to modulate perceptual processing."""
        # If no expectations or no modulator for this modality, return unmodified
        if not expectations or modality not in self.expectation_modulators:
            return {
                "modality": modality,
                "features": bottom_up_result.get("features"),
                "influence_strength": 0.0,
                "influenced_by": []
            }
        
        try:
            # Get the appropriate modulator for this modality
            modulator = self.expectation_modulators[modality]
            
            # Apply modulation
            modulation_result = await modulator(bottom_up_result.get("features"), expectations)
            
            # Ensure the result has expected keys
            return {
                "modality": modality,
                "features": modulation_result.get("features", bottom_up_result.get("features")),
                "influence_strength": modulation_result.get("influence_strength", 0.0),
                "influenced_by": modulation_result.get("influenced_by", [])
            }
        except Exception as e:
            logger.exception(f"Error in top-down modulation for {modality}: {e}")
            # Return unmodified on error
            return {
                "modality": modality,
                "features": bottom_up_result.get("features"),
                "influence_strength": 0.0,
                "influenced_by": ["modulation_error"]
            }

    # --- Modulation Methods ---

    async def _modulate_text_perception(self, features: Dict, expectations: List[ExpectationSignal]) -> Dict:
        """Apply top-down expectations to text features."""
        logger.debug("Modulating text perception...")
        influence_strength = 0.0
        influenced_by = []
        modified_features = features.copy()
        
        # For each expectation, see if it can influence text processing
        for exp in expectations:
            # If expecting specific keywords in text
            if isinstance(exp.pattern, str) and "preview" in features:
                preview_text = features["preview"].lower()
                pattern_text = exp.pattern.lower()
                
                # If the keyword is found, increase influence
                if pattern_text in preview_text:
                    influence_strength = max(influence_strength, exp.strength * 0.3)
                    influenced_by.append(f"keyword_match:{exp.pattern[:10]}")
                    
                    # Could potentially highlight or emphasize text that matches expectation
                    # This is a placeholder for more sophisticated text biasing
            
            # Could add more sophisticated text biasing here
        
        return {
            "features": modified_features,
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }

    async def _modulate_speech_perception(self, features: AudioFeatures, expectations: List[ExpectationSignal]) -> Dict:
        """Apply top-down expectations to speech features."""
        logger.debug("Modulating speech perception...")
        influence_strength = 0.0
        influenced_by = []
        
        # Create a mutable copy of features (Pydantic model to dict)
        modified_features = features.dict()
        
        for exp in expectations:
            # If expecting specific content in speech
            if isinstance(exp.pattern, str) and features.transcription:
                transcription = features.transcription.lower()
                pattern = exp.pattern.lower()
                
                # If expected content is present, increase confidence
                if pattern in transcription:
                    influence_strength = max(influence_strength, exp.strength * 0.2)
                    influenced_by.append(f"content_match:{exp.pattern[:10]}")
                    
            # If expecting specific speaker
            elif hasattr(exp.pattern, "speaker_id") and features.speaker_id:
                if exp.pattern.speaker_id == features.speaker_id:
                    influence_strength = max(influence_strength, exp.strength * 0.4)
                    influenced_by.append(f"speaker_match:{features.speaker_id}")
        
        # Convert dict back to AudioFeatures
        return {
            "features": AudioFeatures(**modified_features),
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }

    async def _modulate_generic_perception(self, features: Any, expectations: List[ExpectationSignal]) -> Dict:
        """Generic modulation for sensory modalities without specialized modulators."""
        logger.debug("Applying generic expectation modulation...")
        influence_strength = 0.0
        influenced_by = []
        
        # Handle different types of feature data
        if isinstance(features, dict):
            # For dictionary features (like image or video features)
            for exp in expectations:
                pattern_str = str(exp.pattern).lower()
                
                # Check description fields
                for field in ["description", "summary"]:
                    if field in features and isinstance(features[field], str):
                        if pattern_str in features[field].lower():
                            influence_strength = max(influence_strength, exp.strength * 0.2)
                            influenced_by.append(f"description_match:{exp.pattern}")
                
                # Check object lists
                if "objects" in features and isinstance(features["objects"], list):
                    if any(pattern_str in obj.lower() for obj in features["objects"] if isinstance(obj, str)):
                        influence_strength = max(influence_strength, exp.strength * 0.3)
                        influenced_by.append(f"object_match:{exp.pattern}")
        
        elif isinstance(features, BaseModel):
            # For Pydantic models (like sensory feature models)
            model_dict = features.dict()
            
            for exp in expectations:
                pattern_str = str(exp.pattern).lower()
                
                # Check string fields in the model
                for field, value in model_dict.items():
                    if isinstance(value, str) and pattern_str in value.lower():
                        influence_strength = max(influence_strength, exp.strength * 0.2)
                        influenced_by.append(f"field_match:{field}:{exp.pattern}")
                    
                    # Check lists of strings
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        if any(pattern_str in item.lower() for item in value):
                            influence_strength = max(influence_strength, exp.strength * 0.3)
                            influenced_by.append(f"list_match:{field}:{exp.pattern}")
        
        # Return original features with modulation metadata
        return {
            "features": features,  # For generic modulation, often return unmodified features
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }

    # --- Integration Methods ---

    async def _integrate_pathways(self, bottom_up_result: Dict[str, Any],
                                top_down_result: Dict[str, Any],
                                modality: Modality) -> Dict[str, Any]:
        """Integrate bottom-up and top-down processing pathways."""
        # Use modality-specific integration strategy if available
        if modality in self.integration_strategies:
            try:
                integration_func = self.integration_strategies[modality]
                result = await integration_func(bottom_up_result, top_down_result)
                
                # Ensure result has content key
                if "content" not in result:
                    logger.warning(f"Integration result for {modality} missing 'content' key")
                    result["content"] = top_down_result.get("features", bottom_up_result.get("features"))
                
                return result
            except Exception as e:
                logger.exception(f"Error in pathway integration for {modality}: {e}")
                # Fall back to default integration on error
        
        # Default integration - blend bottom-up and top-down processing
        bottom_up_conf = bottom_up_result.get("confidence", 0.0)
        top_down_infl = top_down_result.get("influence_strength", 0.0)
        
        # Integrate confidence values
        integrated_conf = bottom_up_conf * (1.0 - top_down_infl * 0.3)
        
        # Use top-down features as primary content, falling back to bottom-up if needed
        integrated_content = top_down_result.get("features", bottom_up_result.get("features"))
        
        return {
            "content": integrated_content,
            "integrated_confidence": integrated_conf,
            "bottom_up_features": bottom_up_result.get("features"),
            "top_down_features": top_down_result.get("features")
        }

    async def _integrate_text_pathways(self, bottom_up: Dict, top_down: Dict) -> Dict:
        """Integrate pathways for text processing."""
        logger.debug("Integrating text processing pathways...")
        
        # Blend confidence based on top-down influence
        bu_conf = bottom_up.get("confidence", 0.0)
        td_infl = top_down.get("influence_strength", 0.0)
        integrated_conf = bu_conf * (1.0 - td_infl * 0.2)
        
        # Use modulated features for content
        features = top_down.get("features", bottom_up.get("features", {}))
        
        # For text, we can create an enriched content representation
        content = {
            "text": features.get("preview", ""),
            "sentiment": features.get("sentiment", 0.0),
            "is_question": features.get("is_question", False),
            "word_count": features.get("word_count", 0)
        }
        
        # Add any modulation influences to content
        if top_down.get("influenced_by"):
            content["influences"] = top_down.get("influenced_by")
        
        return {
            "content": content,
            "integrated_confidence": integrated_conf,
            "bottom_up_features": bottom_up.get("features"),
            "top_down_features": top_down.get("features")
        }

    async def _integrate_generic_pathways(self, bottom_up: Dict, top_down: Dict) -> Dict:
        """Generic integration for modalities without specialized integrators."""
        logger.debug("Applying generic pathway integration...")
        
        # Blend confidence based on top-down influence
        bu_conf = bottom_up.get("confidence", 0.0)
        td_infl = top_down.get("influence_strength", 0.0)
        integrated_conf = bu_conf * (1.0 - td_infl * 0.2)
        
        # Use top-down features, falling back to bottom-up if not available
        content = top_down.get("features", bottom_up.get("features"))
        
        # For structured content (BaseModel instances), add influence information if available
        if isinstance(content, BaseModel) and top_down.get("influenced_by"):
            # Create a mutable copy
            content_dict = content.dict()
            # Add influence information if not already present
            if "influenced_by" not in content_dict:
                content_dict["influenced_by"] = top_down.get("influenced_by")
            # Convert back to the original type
            content = type(content)(**content_dict)
        
        return {
            "content": content,
            "integrated_confidence": integrated_conf,
            "bottom_up_features": bottom_up.get("features"),
            "top_down_features": top_down.get("features")
        }

    async def _integrate_speech_pathways(self, bottom_up: Dict, top_down: Dict) -> Dict:
        """Integrate pathways for speech processing."""
        logger.debug("Integrating speech processing pathways...")
        
        # Blend confidence based on top-down influence
        bu_conf = bottom_up.get("confidence", 0.0)
        td_infl = top_down.get("influence_strength", 0.0)
        integrated_conf = bu_conf * (1.0 - td_infl * 0.25)
        
        # Get features from results
        bu_features = bottom_up.get("features")
        td_features = top_down.get("features")
        
        # For speech, we want to use the top-down modulated features if available
        # This might include bias toward expected words or speakers
        content = td_features if td_features else bu_features
        
        # Create enriched content if we have a transcription
        if isinstance(content, AudioFeatures) and content.transcription:
            enriched_content = {
                "transcription": content.transcription,
                "speaker_id": content.speaker_id,
                "mood": content.mood,
                "type": content.type
            }
            
            # Add influence info
            if top_down.get("influenced_by"):
                enriched_content["influences"] = top_down.get("influenced_by")
            
            # Use enriched dict as content
            return {
                "content": enriched_content,
                "integrated_confidence": integrated_conf,
                "bottom_up_features": bu_features,
                "top_down_features": td_features
            }
        
        # Otherwise return the features as content
        return {
            "content": content,
            "integrated_confidence": integrated_conf,
            "bottom_up_features": bu_features,
            "top_down_features": td_features
        }

    # --- Public API Methods ---

    async def register_feature_extractor(self, modality: Modality, extractor_function):
        """Register a feature extraction function for a specific modality"""
        self.feature_extractors[modality] = extractor_function
        logger.info(f"Registered feature extractor for {modality}")

    async def register_expectation_modulator(self, modality: Modality, modulator_function):
        """Register a function that applies top-down expectations to a modality"""
        self.expectation_modulators[modality] = modulator_function
        logger.info(f"Registered expectation modulator for {modality}")

    async def register_integration_strategy(self, modality: Modality, integration_function):
        """Register a function that integrates bottom-up and top-down processing"""
        self.integration_strategies[modality] = integration_function
        logger.info(f"Registered integration strategy for {modality}")

    async def add_expectation(self, expectation: ExpectationSignal):
        """Add a new top-down expectation signal"""
        self.active_expectations.append(expectation)
        logger.debug(f"Added expectation for {expectation.target_modality}: {expectation.pattern}")
        
        # Prune if we have too many expectations
        if len(self.active_expectations) > 50:
            self.active_expectations.sort(key=lambda x: x.priority, reverse=True)
            self.active_expectations = self.active_expectations[:40]  # Keep top 40

    async def clear_expectations(self, modality: Optional[Modality] = None):
        """Clear active expectations, optionally for a specific modality"""
        if modality:
            old_count = len(self.active_expectations)
            self.active_expectations = [exp for exp in self.active_expectations 
                                      if exp.target_modality != modality]
            new_count = len(self.active_expectations)
            logger.info(f"Cleared {old_count - new_count} expectations for {modality}")
        else:
            old_count = len(self.active_expectations)
            self.active_expectations = []
            logger.info(f"Cleared all {old_count} active expectations")

    async def get_recent_percepts(self, modality: Optional[Modality] = None, 
                                limit: int = 10) -> List[IntegratedPercept]:
        """Get recent percepts, optionally filtered by modality"""
        if modality:
            filtered = [p for p in self.perception_buffer if p.modality == modality]
            return filtered[-limit:]
        else:
            return self.perception_buffer[-limit:]

    # --- Utils for OpenAI Agents SDK Integration ---

    @function_tool
    async def process_text(self, ctx: RunContextWrapper, text: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a text input through the multimodal integration system.
        
        Args:
            text: The text input to process
            metadata: Optional metadata about the text
            
        Returns:
            Processing results including features and integration data
        """
        input_data = SensoryInput(
            modality=Modality.TEXT,
            data=text,
            timestamp=datetime.datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        percept = await self.process_sensory_input(input_data)
        
        return {
            "modality": percept.modality,
            "content": percept.content,
            "confidence": percept.bottom_up_confidence,
            "influence": percept.top_down_influence,
            "attention": percept.attention_weight,
            "features": percept.raw_features
        }
    
    @function_tool
    async def process_image(self, ctx: RunContextWrapper, image_data: Any,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image through the multimodal integration system.
        
        Args:
            image_data: Raw image data (bytes, path, or base64 encoded string)
            metadata: Optional metadata about the image
            
        Returns:
            Processing results including image features and integration data
        """
        input_data = SensoryInput(
            modality=Modality.IMAGE,
            data=image_data,
            timestamp=datetime.datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        percept = await self.process_sensory_input(input_data)
        
        return {
            "modality": percept.modality,
            "content": percept.content,
            "confidence": percept.bottom_up_confidence,
            "influence": percept.top_down_influence,
            "attention": percept.attention_weight,
            "features": percept.raw_features
        }
    
    @function_tool
    async def add_perceptual_expectation(self, ctx: RunContextWrapper, 
                                       target_modality: str,
                                       pattern: Any,
                                       strength: float = 0.5,
                                       priority: float = 0.5,
                                       source: str = "agent") -> Dict[str, Any]:
        """
        Add a top-down expectation signal to guide perception.
        
        Args:
            target_modality: Which modality to apply the expectation to
            pattern: What to expect (e.g., object name, content, feature)
            strength: How strongly to apply the expectation (0.0-1.0)
            priority: Priority relative to other expectations (0.0-1.0)
            source: Source of the expectation (e.g., "memory", "reasoning")
            
        Returns:
            Result of adding the expectation
        """
        try:
            modality = Modality(target_modality)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid modality: {target_modality}. Valid options are: {[m.value for m in Modality]}"
            }
            
        expectation = ExpectationSignal(
            target_modality=modality,
            pattern=pattern,
            strength=min(1.0, max(0.0, strength)),
            priority=min(1.0, max(0.0, priority)),
            source=source
        )
        
        await self.add_expectation(expectation)
        
        return {
            "success": True,
            "expectation_added": {
                "modality": modality,
                "pattern": str(pattern),
                "strength": strength,
                "priority": priority
            },
            "active_expectations_count": len(self.active_expectations)
        }
    
    @function_tool
    async def get_perception_stats(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get statistics about processed percepts.
        
        Returns:
            Statistics by modality including counts and filtering rates
        """
        stats = {}
        for modality in Modality:
            modality_stats = self.perception_stats[modality]
            total = modality_stats["count"]
            filtered = modality_stats["attention_filtered"]
            
            filter_rate = filtered / total if total > 0 else 0.0
            
            stats[modality] = {
                "total_processed": total,
                "attention_filtered": filtered,
                "filter_rate": filter_rate
            }
            
        return {
            "stats_by_modality": stats,
            "total_percepts": len(self.perception_buffer),
            "active_expectations": len(self.active_expectations)
        }
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
