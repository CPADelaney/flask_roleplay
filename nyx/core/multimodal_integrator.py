# nyx/core/multimodal_integrator.py

import logging
import numpy as np
import datetime
import base64
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator, Set, TypedDict
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
# Define a concrete type for metadata instead of using forward references
class MetadataValue(BaseModel):
    """Wrapper for metadata values to ensure JSON compatibility"""
    value: Union[str, int, float, bool, None, List["MetadataValue"], Dict[str, "MetadataValue"]]

class SensoryInput(BaseModel):
    """
    Raw sensory payload passed into the process_sensory_input() tool.

    • `data` is now a *string* (text or base-64 for binary).  
    • `metadata` is an object/dict (recursive JSON).  
      → If you need to pass non-JSON, JSON-encode it first.
    """
    modality: Modality

    # If you need to send binary (image bytes, etc.), base-64 encode it
    data: str = Field(..., description="Raw payload – text or base-64 string")

    confidence: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Source-estimated reliability of the payload"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat(),
        description="ISO-8601 timestamp when the data was captured"
    )

    # Use explicit model instead of Dict with forward reference
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque metadata blob (must be JSON-serialisable)"
    )

class ExpectationSignal(BaseModel):
    """
    Top-down bias sent to the integrator.

    If you must transmit complex patterns (e.g. an array or dict),
    JSON-encode them first and put the resulting string here.
    """
    target_modality: Modality
    pattern: str = Field(
        ...,
        description="Expected pattern/feature. "
                    "Encode complex data with json.dumps(...)"
    )
    strength: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field(..., description="Subsystem that emitted the expectation")
    priority: float = Field(0.5, ge=0.0, le=1.0)

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
    text_content: Optional[str] = None
    dominant_colors: List[Tuple[int, int, int]] = Field(default_factory=list)
    estimated_mood: Optional[str] = None
    is_screenshot: bool = False
    spatial_layout: Any = None             # ← Dict → Any

class AudioFeatures(BaseModel):            # unchanged – shown for fwd-ref clarity
    type: str = "unknown"
    transcription: Optional[str] = None
    speaker_id: Optional[str] = None
    mood: Optional[str] = None
    genre: Optional[str] = None
    tempo_bpm: Optional[float] = None
    key: Optional[str] = None
    sound_events: List[str] = Field(default_factory=list)
    noise_level: Optional[float] = None

class VideoFeatures(BaseModel):
    """Features extracted from video data."""
    summary: str = "No summary available."
    key_actions: List[str] = Field(default_factory=list)
    tracked_objects: Any = None            # ← Dict → Any
    scene_changes: List[float] = Field(default_factory=list)
    audio_features: Optional[AudioFeatures] = None
    estimated_mood_progression: List[Tuple[float, str]] = Field(default_factory=list)
    significant_frames: Any = None         # ← List[Dict] → Any

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

class FeatureExtractionResult(BaseModel):
    modality: Modality
    features: Any
    confidence: float
    metadata: Any

class ExpectationResult(BaseModel):
    """Result of applying top-down expectations to features."""
    features: Any
    influence_strength: float
    influenced_by: List[str]

class IntegrationResult(BaseModel):
    """Result of integrating bottom-up and top-down processing."""
    content: Any
    integrated_confidence: float
    bottom_up_features: Optional[Any] = None
    top_down_features: Optional[Any] = None

class SensoryInputDict(TypedDict, total=False):
    """Type definition for sensory input data"""
    modality: str
    data: Any
    confidence: float
    timestamp: str
    metadata: Dict[str, Any]

class ExpectationDict(TypedDict, total=False):
    """Type definition for expectation data"""
    target_modality: str
    pattern: Any
    strength: float
    source: str
    priority: float
    

# --- Constants for Feature Classification ---
POSITIVE_TASTES = {"sweet", "umami", "fatty", "savory"}
NEGATIVE_TASTES = {"bitter", "sour", "metallic", "spoiled"}
POSITIVE_SMELLS = {"floral", "fruity", "sweet", "fresh", "baked", "woody", "earthy"}
NEGATIVE_SMELLS = {"pungent", "chemical", "rotten", "sour", "fishy", "burnt"}

class ProcessSensoryInputParams(BaseModel):
    input_data: SensoryInput
    expectations: Optional[List[ExpectationSignal]] = None


def make_sensory_input(
    modality: Modality,
    data: Union[str, bytes],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
) -> SensoryInput:
    """
    Convenience wrapper that:
    • base-64 encodes bytes
    • injects a timestamp automatically
    """
    if isinstance(data, bytes):
        data = base64.b64encode(data).decode("ascii")
    return SensoryInput(
        modality=modality,
        data=data,
        confidence=confidence,
        metadata=metadata or {},
    )


# Context class to hold the state and dependencies
class MultimodalIntegratorContext:
    """Context object for the MultimodalIntegrator agent operations."""
    
    def __init__(self, reasoning_core=None, attentional_controller=None, vision_model=None, audio_processor=None):
        """Initialize the context with dependencies."""
        self.reasoning_core = reasoning_core
        self.attentional_controller = attentional_controller
        self.vision_model = vision_model
        self.audio_processor = audio_processor
        
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
        
        # Embedding model and cache
        self.embedding_model = None
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("MultimodalIntegratorContext initialized")


# --- Function Tools for Agent SDK ---

# Create parameter models for function tools that need them
class AddExpectationParams(BaseModel):
    target_modality: str
    pattern: Any
    strength: float = 0.5
    priority: float = 0.5
    source: str = "agent"

class ClearExpectationsParams(BaseModel):
    modality: Optional[str] = None

class GetRecentPerceptsParams(BaseModel):
    modality: Optional[str] = None
    limit: int = 10

class ProcessTextParams(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class ProcessImageParams(BaseModel):
    image_data: Any
    metadata: Optional[Dict[str, Any]] = None

class RegisterFeatureExtractorParams(BaseModel):
    modality: str
    extractor_name: str

@function_tool
async def process_sensory_input(
    ctx: RunContextWrapper[MultimodalIntegratorContext],
    params: ProcessSensoryInputParams,                     # ← single Pydantic arg
) -> Dict[str, Any]:
    """
    Bottom-up + top-down sensory processing. Returns IntegratedPercept as dict.
    """
    input_data = params.input_data
    expectations = params.expectations
    integrator_ctx = ctx.context
    
    # Convert input_data to SensoryInput if it's a dict
    if isinstance(input_data, dict):
        try:
            input_data = SensoryInput(**input_data)
        except Exception as e:
            logger.error(f"Error converting input data to SensoryInput: {e}")
            return {"error": f"Invalid input data: {e}"}
    
    # Convert expectations to ExpectationSignal objects if they're dicts
    expectation_objects = None
    if expectations:
        expectation_objects = []
        for exp in expectations:
            try:
                expectation_objects.append(ExpectationSignal(**exp))
            except Exception as e:
                logger.error(f"Error converting expectation to ExpectationSignal: {e}")
    
    modality = input_data.modality
    timestamp = input_data.timestamp or datetime.datetime.now().isoformat()
    logger.info(f"Processing sensory input: Modality={modality}, Timestamp={timestamp}")
    
    # Update stats
    async with integrator_ctx._lock:
        integrator_ctx.perception_stats[modality]["count"] += 1

    # 1. Bottom-up processing (data-driven)
    logger.debug(f"Performing bottom-up processing for {modality}")
    bottom_up_result = await _perform_bottom_up_processing(integrator_ctx, input_data)
    logger.debug(f"Bottom-up features for {modality} extracted")

    # 2. Get or use provided top-down expectations
    if expectation_objects is None:
        logger.debug(f"Fetching active expectations for {modality}")
        expectation_objects = await _get_active_expectations(integrator_ctx, modality)
        logger.debug(f"Found {len(expectation_objects)} active expectations for {modality}")
    else:
        logger.debug(f"Using {len(expectation_objects)} provided expectations for {modality}")

    # 3. Apply top-down modulation
    logger.debug(f"Applying top-down modulation for {modality}")
    modulated_result = await _apply_top_down_modulation(integrator_ctx, bottom_up_result, expectation_objects, modality)
    logger.debug(f"Modulation influence for {modality}: {modulated_result.get('influence_strength', 0):.2f}")

    # 4. Integrate bottom-up and top-down pathways
    logger.debug(f"Integrating pathways for {modality}")
    integrated_result = await _integrate_pathways(integrator_ctx, bottom_up_result, modulated_result, modality)
    logger.debug(f"Integrated content for {modality}")

    # 5. Apply attentional filtering if available
    attentional_weight = 1.0  # Default
    if integrator_ctx.attentional_controller:
        try:
            logger.debug(f"Calculating attention weight for {modality}")
            # Prepare attention input
            attention_input = {
                "modality": modality,
                "content_summary": _get_content_summary(integrated_result.get('content', {})),
                "confidence": bottom_up_result.get("confidence", 1.0),
                "metadata": input_data.metadata
            }
            
            attentional_weight = await integrator_ctx.attentional_controller.calculate_attention_weight(
                attention_input, expectation_objects
            )
            logger.debug(f"Calculated attention weight for {modality}: {attentional_weight:.2f}")
            
            # Update stats for filtered percepts
            if attentional_weight < 0.2:  # Threshold for considering "filtered"
                async with integrator_ctx._lock:
                    integrator_ctx.perception_stats[modality]["attention_filtered"] += 1
                
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
    await _add_to_perception_buffer(integrator_ctx, percept)

    # 8. Update reasoning core with new perception (if significant attention)
    if integrator_ctx.reasoning_core and attentional_weight > 0.3:
        try:
            logger.debug(f"Updating reasoning core with significant percept ({modality}, weight={attentional_weight:.2f})")
            # Check if reasoning core has update method
            if hasattr(integrator_ctx.reasoning_core, 'update_with_perception'):
                await integrator_ctx.reasoning_core.update_with_perception(percept)
        except Exception as e:
            logger.error(f"Error updating reasoning core: {e}")

    logger.info(f"Finished processing sensory input for {modality}")
    
    # Return the percept as a dict
    return percept.dict()

@function_tool
async def add_expectation(
    ctx: RunContextWrapper[MultimodalIntegratorContext],
    params: AddExpectationParams
) -> Dict[str, Any]:
    """
    Add a top-down expectation signal to guide perception.
    
    Args:
        params: Parameters for adding expectation
        
    Returns:
        Result of adding the expectation
    """
    integrator_ctx = ctx.context
    
    try:
        modality = Modality(params.target_modality)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid modality: {params.target_modality}. Valid options are: {[m.value for m in Modality]}"
        }
        
    pattern = params.pattern if isinstance(params.pattern, str) else json.dumps(params.pattern)
    expectation = ExpectationSignal(
        target_modality=modality,
        pattern=pattern,
        strength=min(1.0, max(0.0, params.strength)),
        priority=min(1.0, max(0.0, params.priority)),
        source=params.source
    )
    
    async with integrator_ctx._lock:
        integrator_ctx.active_expectations.append(expectation)
        
        # Prune if we have too many expectations
        if len(integrator_ctx.active_expectations) > 50:
            integrator_ctx.active_expectations.sort(key=lambda x: x.priority, reverse=True)
            integrator_ctx.active_expectations = integrator_ctx.active_expectations[:40]  # Keep top 40
    
    logger.debug(f"Added expectation for {modality}: {pattern}")
    
    return {
        "success": True,
        "expectation_added": {
            "modality": modality,
            "pattern": str(pattern),
            "strength": params.strength,
            "priority": params.priority
        },
        "active_expectations_count": len(integrator_ctx.active_expectations)
    }

@function_tool
async def clear_expectations(
    ctx: RunContextWrapper[MultimodalIntegratorContext],
    params: ClearExpectationsParams
) -> Dict[str, Any]:
    """
    Clear active expectations, optionally for a specific modality.
    
    Args:
        params: Parameters for clearing expectations
        
    Returns:
        Result of clearing expectations
    """
    integrator_ctx = ctx.context
    
    async with integrator_ctx._lock:
        old_count = len(integrator_ctx.active_expectations)
        
        if params.modality:
            try:
                mod = Modality(params.modality)
                integrator_ctx.active_expectations = [
                    exp for exp in integrator_ctx.active_expectations 
                    if exp.target_modality != mod
                ]
                new_count = len(integrator_ctx.active_expectations)
                logger.info(f"Cleared {old_count - new_count} expectations for {params.modality}")
                
                return {
                    "success": True,
                    "cleared_count": old_count - new_count,
                    "remaining_count": new_count,
                    "modality": params.modality
                }
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid modality: {params.modality}"
                }
        else:
            integrator_ctx.active_expectations = []
            logger.info(f"Cleared all {old_count} active expectations")
            
            return {
                "success": True,
                "cleared_count": old_count,
                "remaining_count": 0
            }

@function_tool
async def get_recent_percepts(
    ctx: RunContextWrapper[MultimodalIntegratorContext],
    params: GetRecentPerceptsParams
) -> List[Dict[str, Any]]:
    """
    Get recent percepts, optionally filtered by modality.
    
    Args:
        params: Parameters for getting recent percepts
        
    Returns:
        List of recent percepts
    """
    integrator_ctx = ctx.context
    
    if params.modality:
        try:
            mod = Modality(params.modality)
            filtered = [p for p in integrator_ctx.perception_buffer if p.modality == mod]
            return [p.dict() for p in filtered[-params.limit:]]
        except ValueError:
            return []
    else:
        return [p.dict() for p in integrator_ctx.perception_buffer[-params.limit:]]

@function_tool
async def get_perception_stats(
    ctx: RunContextWrapper[MultimodalIntegratorContext]
) -> Dict[str, Any]:
    """
    Get statistics about processed percepts.
    
    Returns:
        Statistics by modality including counts and filtering rates
    """
    integrator_ctx = ctx.context
    
    stats = {}
    for modality in Modality:
        modality_stats = integrator_ctx.perception_stats[modality]
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
        "total_percepts": len(integrator_ctx.perception_buffer),
        "active_expectations": len(integrator_ctx.active_expectations)
    }

@function_tool
async def process_text(
    ctx: RunContextWrapper[MultimodalIntegratorContext],
    params: ProcessTextParams
) -> Dict[str, Any]:
    input_data = SensoryInput(
        modality=Modality.TEXT,
        data=params.text,
        metadata=params.metadata or {}
    )
    return await process_sensory_input(ctx, ProcessSensoryInputParams(input_data=input_data))

@function_tool
async def process_image(
    ctx: RunContextWrapper[MultimodalIntegratorContext],
    params: ProcessImageParams
) -> Dict[str, Any]:
    if isinstance(params.image_data, bytes):
        image_data = base64.b64encode(params.image_data).decode("ascii")
    else:
        image_data = params.image_data
    
    input_data = make_sensory_input(
        Modality.IMAGE,
        image_data,
        metadata=params.metadata or {}
    )
    return await process_sensory_input(ctx, ProcessSensoryInputParams(input_data=input_data))

@function_tool
async def register_feature_extractor(
    ctx: RunContextWrapper[MultimodalIntegratorContext],
    params: RegisterFeatureExtractorParams
) -> Dict[str, Any]:
    """
    Register a named feature extraction function for a specific modality.
    
    Args:
        params: Parameters for registering feature extractor
        
    Returns:
        Result of the registration
    """
    integrator_ctx = ctx.context
    
    try:
        mod = Modality(params.modality)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid modality: {params.modality}"
        }
    
    # Map extractor names to functions
    extractors = {
        "text": _extract_text_features,
        "image": _extract_image_features,
        "video": _extract_video_features,
        "audio": _extract_audio_features,
        "touch": _extract_touch_event_features,
        "taste": _extract_taste_features,
        "smell": _extract_smell_features
    }
    
    if params.extractor_name not in extractors:
        return {
            "success": False,
            "error": f"Unknown extractor: {params.extractor_name}. Available extractors: {list(extractors.keys())}"
        }
    
    integrator_ctx.feature_extractors[mod] = extractors[params.extractor_name]
    logger.info(f"Registered {params.extractor_name} feature extractor for {params.modality}")
    
    return {
        "success": True,
        "modality": params.modality,
        "extractor": params.extractor_name
    }

# --- Helper Functions ---

def _get_content_summary(content: Any) -> str:
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

async def _add_to_perception_buffer(ctx: MultimodalIntegratorContext, percept: IntegratedPercept):
    """Add percept to buffer, maintaining thread safety."""
    async with ctx._lock:
        ctx.perception_buffer.append(percept)
        if len(ctx.perception_buffer) > ctx.max_buffer_size:
            ctx.perception_buffer.pop(0)

async def _perform_bottom_up_processing(ctx: MultimodalIntegratorContext, input_data: SensoryInput) -> Dict[str, Any]:
    """Extract features from raw sensory input using registered extractors."""
    modality = input_data.modality

    if modality in ctx.feature_extractors:
        try:
            extractor = ctx.feature_extractors[modality]
            features = await extractor(ctx, input_data.data, input_data.metadata)
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

async def _get_active_expectations(ctx: MultimodalIntegratorContext, modality: Modality) -> List[ExpectationSignal]:
    """Get current active expectations for a specific modality."""
    # Filter expectations relevant to this modality
    relevant_expectations = [exp for exp in ctx.active_expectations
                          if exp.target_modality == modality]
    
    # If reasoning core can provide expectations, query it
    if ctx.reasoning_core and hasattr(ctx.reasoning_core, 'generate_perceptual_expectations'):
        try:
            logger.debug(f"Querying reasoning core for expectations ({modality})")
            new_expectations = await ctx.reasoning_core.generate_perceptual_expectations(modality)
            logger.debug(f"Received {len(new_expectations)} new expectations")
            
            # Add new expectations to the list
            relevant_expectations.extend(new_expectations)
            
            # Add new expectations to active list for future use
            ctx.active_expectations.extend(new_expectations)
            
            # Prune old expectations if we have too many
            if len(ctx.active_expectations) > 50:
                ctx.active_expectations.sort(key=lambda x: x.priority, reverse=True)
                ctx.active_expectations = ctx.active_expectations[:40]  # Keep top 40
                
        except Exception as e:
            logger.error(f"Error getting expectations from reasoning core: {e}")
    
    # Sort by priority
    if relevant_expectations:
        relevant_expectations.sort(key=lambda x: x.priority, reverse=True)
        
    return relevant_expectations

async def _apply_top_down_modulation(
    ctx: MultimodalIntegratorContext,
    bottom_up_result: Dict[str, Any],
    expectations: List[ExpectationSignal],
    modality: Modality
) -> Dict[str, Any]:
    """Apply top-down expectations to modulate perceptual processing."""
    # If no expectations or no modulator for this modality, return unmodified
    if not expectations or modality not in ctx.expectation_modulators:
        return {
            "modality": modality,
            "features": bottom_up_result.get("features"),
            "influence_strength": 0.0,
            "influenced_by": []
        }
    
    try:
        # Get the appropriate modulator for this modality
        modulator = ctx.expectation_modulators[modality]
        
        # Apply modulation
        modulation_result = await modulator(ctx, bottom_up_result.get("features"), expectations)
        
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

async def _integrate_pathways(
    ctx: MultimodalIntegratorContext,
    bottom_up_result: Dict[str, Any],
    top_down_result: Dict[str, Any],
    modality: Modality
) -> Dict[str, Any]:
    """Integrate bottom-up and top-down processing pathways."""
    # Use modality-specific integration strategy if available
    if modality in ctx.integration_strategies:
        try:
            integration_func = ctx.integration_strategies[modality]
            result = await integration_func(ctx, bottom_up_result, top_down_result)
            
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

# --- Feature Extraction Methods ---

async def _extract_text_features(ctx: MultimodalIntegratorContext, data: str, metadata: Dict) -> Dict:
    """Extract features from text input."""
    logger.debug(f"Extracting text features: {data[:50]}...")
    
    # If using vision model and it supports text, use that
    if ctx.vision_model and hasattr(ctx.vision_model, 'analyze_text'):
        try:
            vision_results = await ctx.vision_model.analyze_text(data)
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

async def _extract_image_features(ctx: MultimodalIntegratorContext, data: Any, metadata: Dict) -> ImageFeatures:
    """Extract features from image data."""
    logger.debug(f"Extracting image features (data type: {type(data)})...")
    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            logger.warning("Invalid base-64 payload")
    
    # Use vision model if available
    if ctx.vision_model and hasattr(ctx.vision_model, 'analyze_image'):
        try:
            vision_results = await ctx.vision_model.analyze_image(data)
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

async def _extract_video_features(ctx: MultimodalIntegratorContext, data: Any, metadata: Dict) -> VideoFeatures:
    """Extract features from video data."""
    logger.debug(f"Extracting video features (data type: {type(data)})...")
    
    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            logger.warning("Invalid base-64 payload")
    
    # Use vision model if available for key frames
    if ctx.vision_model and hasattr(ctx.vision_model, 'analyze_video'):
        try:
            vision_results = await ctx.vision_model.analyze_video(data)
            return vision_results  # Assuming it returns a VideoFeatures object
        except Exception as e:
            logger.error(f"Error using vision model for video: {e}")
    
    # Use audio processor for audio track if available
    audio_features = None
    if ctx.audio_processor and hasattr(ctx.audio_processor, 'extract_audio_from_video'):
        try:
            audio_data = await ctx.audio_processor.extract_audio_from_video(data)
            audio_features = await _extract_audio_features(ctx, audio_data, metadata)
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

async def _extract_audio_features(ctx: MultimodalIntegratorContext, data: Any, metadata: Dict) -> AudioFeatures:
    """Extract features from audio data."""
    logger.debug(f"Extracting audio features (data type: {type(data)})...")

    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            logger.warning("Invalid base-64 payload")
    
    # Use audio processor if available
    if ctx.audio_processor:
        try:
            if hasattr(ctx.audio_processor, 'analyze_speech') and metadata.get("source_modality") in [Modality.AUDIO_SPEECH, Modality.SYSTEM_AUDIO]:
                return await ctx.audio_processor.analyze_speech(data)
            elif hasattr(ctx.audio_processor, 'analyze_music') and metadata.get("source_modality") == Modality.AUDIO_MUSIC:
                return await ctx.audio_processor.analyze_music(data)
            elif hasattr(ctx.audio_processor, 'analyze_audio'):
                return await ctx.audio_processor.analyze_audio(data)
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

async def _extract_touch_event_features(ctx: MultimodalIntegratorContext, data: Dict, metadata: Dict) -> TouchEventFeatures:
    """Extract features from touch event data."""
    logger.debug(f"Extracting touch event features: {data}")

    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            logger.warning("Invalid base-64 payload")
    
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

async def _extract_taste_features(ctx: MultimodalIntegratorContext, data: Dict, metadata: Dict) -> TasteFeatures:
    """Extract features from taste data."""
    logger.debug(f"Extracting taste features: {data}")

    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            logger.warning("Invalid base-64 payload")
    
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

async def _extract_smell_features(ctx: MultimodalIntegratorContext, data: Dict, metadata: Dict) -> SmellFeatures:
    """Extract features from smell data."""
    logger.debug(f"Extracting smell features: {data}")

    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            logger.warning("Invalid base-64 payload")
    
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

# --- Create Agent ---
multimodal_agent = Agent(
    name="Multimodal Integration Agent",
    instructions="""You analyze sensory percepts and integrate information across modalities.
    
    As a multimodal integration agent, your responsibilities include:
    1. Processing raw sensory inputs across different modalities (text, image, audio, etc.)
    2. Extracting relevant features from each modality
    3. Applying top-down expectations to guide and influence perception
    4. Integrating bottom-up and top-down pathways into coherent percepts
    5. Managing attention to filter less important percepts
    6. Maintaining perception history and connections between modalities
    
    Use the appropriate tools for each type of sensory input, and aim to create 
    rich, contextual understanding of multisensory information.
    """,
    tools=[
        process_sensory_input,
        add_expectation,
        clear_expectations,
        get_recent_percepts,
        get_perception_stats,
        process_text,
        process_image,
        register_feature_extractor
    ],
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.2)
)

class MultimodalIntegrator:
    """
    Processes sensory inputs using both bottom-up and top-down pathways.
    Handles text, image, video, audio, and other sensory modalities.
    Uses Agent SDK for integration management.
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
        self.context = MultimodalIntegratorContext(
            reasoning_core=reasoning_core,
            attentional_controller=attentional_controller,
            vision_model=vision_model,
            audio_processor=audio_processor
        )
        
        # Agent for integration
        self.agent = multimodal_agent
        
        # Register default handlers for each modality
        self._register_default_handlers()
        
        logger.info("MultimodalIntegrator initialized with Agent SDK")

    def _register_default_handlers(self):
        """Register default handlers for all modalities."""
        logger.info("Registering default multimodal handlers")

        # Text modality
        self.context.feature_extractors[Modality.TEXT] = _extract_text_features
        self.context.expectation_modulators[Modality.TEXT] = self._modulate_text_perception
        self.context.integration_strategies[Modality.TEXT] = self._integrate_text_pathways

        # Image modality
        self.context.feature_extractors[Modality.IMAGE] = _extract_image_features
        self.context.expectation_modulators[Modality.IMAGE] = self._modulate_generic_perception
        self.context.integration_strategies[Modality.IMAGE] = self._integrate_generic_pathways

        # Video modality
        self.context.feature_extractors[Modality.VIDEO] = _extract_video_features
        self.context.expectation_modulators[Modality.VIDEO] = self._modulate_generic_perception
        self.context.integration_strategies[Modality.VIDEO] = self._integrate_generic_pathways

        # Audio (Music) modality
        self.context.feature_extractors[Modality.AUDIO_MUSIC] = _extract_audio_features
        self.context.expectation_modulators[Modality.AUDIO_MUSIC] = self._modulate_generic_perception
        self.context.integration_strategies[Modality.AUDIO_MUSIC] = self._integrate_generic_pathways

        # Audio (Speech) modality
        self.context.feature_extractors[Modality.AUDIO_SPEECH] = _extract_audio_features
        self.context.expectation_modulators[Modality.AUDIO_SPEECH] = self._modulate_speech_perception
        self.context.integration_strategies[Modality.AUDIO_SPEECH] = self._integrate_speech_pathways

        # System Screen modality
        self.context.feature_extractors[Modality.SYSTEM_SCREEN] = _extract_image_features
        self.context.expectation_modulators[Modality.SYSTEM_SCREEN] = self._modulate_generic_perception
        self.context.integration_strategies[Modality.SYSTEM_SCREEN] = self._integrate_generic_pathways

        # System Audio modality
        self.context.feature_extractors[Modality.SYSTEM_AUDIO] = _extract_audio_features
        self.context.expectation_modulators[Modality.SYSTEM_AUDIO] = self._modulate_speech_perception
        self.context.integration_strategies[Modality.SYSTEM_AUDIO] = self._integrate_speech_pathways

        # Touch Event modality
        self.context.feature_extractors[Modality.TOUCH_EVENT] = _extract_touch_event_features
        self.context.expectation_modulators[Modality.TOUCH_EVENT] = self._modulate_generic_perception
        self.context.integration_strategies[Modality.TOUCH_EVENT] = self._integrate_generic_pathways

        # Taste modality
        self.context.feature_extractors[Modality.TASTE] = _extract_taste_features
        self.context.expectation_modulators[Modality.TASTE] = self._modulate_generic_perception
        self.context.integration_strategies[Modality.TASTE] = self._integrate_generic_pathways

        # Smell modality
        self.context.feature_extractors[Modality.SMELL] = _extract_smell_features
        self.context.expectation_modulators[Modality.SMELL] = self._modulate_generic_perception
        self.context.integration_strategies[Modality.SMELL] = self._integrate_generic_pathways

        logger.info("All default multimodal handlers registered")

    # --- Modulation Methods ---

    async def _modulate_text_perception(self, ctx: MultimodalIntegratorContext, features: Dict, expectations: List[ExpectationSignal]) -> Dict:
        """Apply top-down expectations to text features."""
        logger.debug("Modulating text perception...")
        influence_strength = 0.0
        influenced_by = []
        modified_features = features.copy() if isinstance(features, dict) else {}
        
        # For each expectation, see if it can influence text processing
        for exp in expectations:
            # If expecting specific keywords in text
            if isinstance(exp.pattern, str) and isinstance(features, dict) and "preview" in features:
                preview_text = features["preview"].lower()
                pattern_text = exp.pattern.lower()
                
                # If the keyword is found, increase influence
                if pattern_text in preview_text:
                    influence_strength = max(influence_strength, exp.strength * 0.3)
                    influenced_by.append(f"keyword_match:{exp.pattern[:10]}")
                    
                    # Could potentially highlight or emphasize text that matches expectation
            
            # Could add more sophisticated text biasing here
        
        return {
            "features": modified_features,
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }

    async def _modulate_speech_perception(self, ctx: MultimodalIntegratorContext, features: Any, expectations: List[ExpectationSignal]) -> Dict:
        """Apply top-down expectations to speech features."""
        logger.debug("Modulating speech perception...")
        influence_strength = 0.0
        influenced_by = []
        
        # Handle AudioFeatures or dict
        if isinstance(features, AudioFeatures):
            modified_features = features.copy()
        elif isinstance(features, dict):
            modified_features = features.copy()
        else:
            modified_features = {}
            
        for exp in expectations:
            # If expecting specific content in speech
            if isinstance(exp.pattern, str):
                # Handle AudioFeatures
                if isinstance(features, AudioFeatures) and features.transcription:
                    transcription = features.transcription.lower()
                    pattern = exp.pattern.lower()
                    
                    # If expected content is present, increase confidence
                    if pattern in transcription:
                        influence_strength = max(influence_strength, exp.strength * 0.2)
                        influenced_by.append(f"content_match:{exp.pattern[:10]}")
                
                # Handle dict
                elif isinstance(features, dict) and "transcription" in features:
                    transcription = features["transcription"].lower()
                    pattern = exp.pattern.lower()
                    
                    if pattern in transcription:
                        influence_strength = max(influence_strength, exp.strength * 0.2)
                        influenced_by.append(f"content_match:{exp.pattern[:10]}")
                    
            # If expecting specific speaker
            elif hasattr(exp.pattern, "speaker_id"):
                # Handle AudioFeatures
                if isinstance(features, AudioFeatures) and features.speaker_id:
                    if exp.pattern.speaker_id == features.speaker_id:
                        influence_strength = max(influence_strength, exp.strength * 0.4)
                        influenced_by.append(f"speaker_match:{features.speaker_id}")
                
                # Handle dict
                elif isinstance(features, dict) and "speaker_id" in features:
                    if exp.pattern.speaker_id == features["speaker_id"]:
                        influence_strength = max(influence_strength, exp.strength * 0.4)
                        influenced_by.append(f"speaker_match:{features['speaker_id']}")
        
        return {
            "features": modified_features,
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }

    async def _modulate_generic_perception(self, ctx: MultimodalIntegratorContext, features: Any, expectations: List[ExpectationSignal]) -> Dict:
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

    async def _integrate_text_pathways(self, ctx: MultimodalIntegratorContext, bottom_up: Dict, top_down: Dict) -> Dict:
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

    async def _integrate_generic_pathways(self, ctx: MultimodalIntegratorContext, bottom_up: Dict, top_down: Dict) -> Dict:
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

    async def _integrate_speech_pathways(self, ctx: MultimodalIntegratorContext, bottom_up: Dict, top_down: Dict) -> Dict:
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
        with trace(workflow_name="sensory_processing"):
            # Prepare prompt for agent
            if expectations:
                prompt = f"Process {input_data.modality} input with provided expectations"
            else:
                prompt = f"Process {input_data.modality} input with default expectations"
            
            # Call the agent to process the input
            input_dict = input_data.dict()
            expectations_dict = [exp.dict() for exp in expectations] if expectations else None
            
            result = await Runner.run(
                self.agent,
                input=prompt,
                context=self.context,
                model_settings=ModelSettings(response_format={"type": "json_object"})
            )
            
            # Convert the result to IntegratedPercept
            if isinstance(result.final_output, dict):
                try:
                    return IntegratedPercept(**result.final_output)
                except Exception as e:
                    logger.error(f"Error converting agent output to IntegratedPercept: {e}")
                    
                    # Create fallback percept
                    return IntegratedPercept(
                        modality=input_data.modality,
                        content={"error": "Error processing input", "details": str(e)},
                        bottom_up_confidence=0.5,
                        top_down_influence=0.0,
                        attention_weight=0.1,
                        timestamp=datetime.datetime.now().isoformat(),
                        raw_features=None
                    )
            else:
                logger.error(f"Unexpected result format: {type(result.final_output)}")
                
                # Create fallback percept
                return IntegratedPercept(
                    modality=input_data.modality,
                    content={"error": "Unexpected agent output format"},
                    bottom_up_confidence=0.5,
                    top_down_influence=0.0,
                    attention_weight=0.1,
                    timestamp=datetime.datetime.now().isoformat(),
                    raw_features=None
                )

    async def add_expectation(self, expectation: ExpectationSignal) -> Dict[str, Any]:
        """Add a new top-down expectation signal"""
        prompt = f"Add expectation for {expectation.target_modality} with pattern '{expectation.pattern}', " \
                 f"strength {expectation.strength}, priority {expectation.priority}, source '{expectation.source}'"
        
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        return result.final_output

    async def clear_expectations(self, modality: Optional[Modality] = None) -> Dict[str, Any]:
        """Clear active expectations, optionally for a specific modality"""
        if modality:
            prompt = f"Clear all expectations for {modality}"
        else:
            prompt = "Clear all active expectations"
            
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        return result.final_output

    async def get_recent_percepts(self, modality: Optional[Modality] = None, 
                                limit: int = 10) -> List[IntegratedPercept]:
        """Get recent percepts, optionally filtered by modality"""
        if modality:
            prompt = f"Get {limit} most recent percepts for {modality}"
        else:
            prompt = f"Get {limit} most recent percepts across all modalities"
            
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        # Convert the result to IntegratedPercept objects
        if isinstance(result.final_output, list):
            percepts = []
            for item in result.final_output:
                try:
                    percepts.append(IntegratedPercept(**item))
                except Exception as e:
                    logger.error(f"Error converting percept: {e}")
            return percepts
        else:
            logger.error(f"Unexpected result format for get_recent_percepts: {type(result.final_output)}")
            return []

    async def get_perception_stats(self) -> Dict[str, Any]:
        """Get statistics about processed percepts"""
        result = await Runner.run(
            self.agent,
            "Get perception statistics across all modalities",
            context=self.context
        )
        
        return result.final_output

    async def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> IntegratedPercept:
        """Process a text input through the multimodal integration system"""
        prompt = f"Process this text: {text[:50]}..." + (f" with metadata {json.dumps(metadata)}" if metadata else "")
        
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        # Convert result to IntegratedPercept
        if isinstance(result.final_output, dict):
            try:
                return IntegratedPercept(**result.final_output)
            except Exception as e:
                logger.error(f"Error converting text processing result: {e}")
        
        # Create fallback
        return IntegratedPercept(
            modality=Modality.TEXT,
            content={"text": text[:100], "error": "Processing error"},
            bottom_up_confidence=0.5,
            top_down_influence=0.0,
            attention_weight=0.3,
            timestamp=datetime.datetime.now().isoformat(),
            raw_features=None
        )

    async def process_image(self, image_data: Any, 
                           metadata: Optional[Dict[str, Any]] = None) -> IntegratedPercept:
        """Process an image through the multimodal integration system"""
        prompt = "Process image data" + (f" with metadata {json.dumps(metadata)}" if metadata else "")
        
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        # Convert result to IntegratedPercept
        if isinstance(result.final_output, dict):
            try:
                return IntegratedPercept(**result.final_output)
            except Exception as e:
                logger.error(f"Error converting image processing result: {e}")
        
        # Create fallback
        return IntegratedPercept(
            modality=Modality.IMAGE,
            content={"error": "Image processing error"},
            bottom_up_confidence=0.5,
            top_down_influence=0.0,
            attention_weight=0.3,
            timestamp=datetime.datetime.now().isoformat(),
            raw_features=None
        )

    async def register_feature_extractor(self, modality: Modality, extractor_function) -> bool:
        """Register a feature extraction function for a specific modality"""
        self.context.feature_extractors[modality] = extractor_function
        logger.info(f"Registered feature extractor for {modality}")
        return True

    async def register_expectation_modulator(self, modality: Modality, modulator_function) -> bool:
        """Register a function that applies top-down expectations to a modality"""
        self.context.expectation_modulators[modality] = modulator_function
        logger.info(f"Registered expectation modulator for {modality}")
        return True

    async def register_integration_strategy(self, modality: Modality, integration_function) -> bool:
        """Register a function that integrates bottom-up and top-down processing"""
        self.context.integration_strategies[modality] = integration_function
        logger.info(f"Registered integration strategy for {modality}")
        return True
        
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
