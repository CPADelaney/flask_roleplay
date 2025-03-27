# nyx/core/body_image.py

import logging
import datetime
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants for modalities
MODALITY_SYSTEM_SCREEN = "system_screen"
MODALITY_IMAGE = "image"

class BodyPartState(BaseModel):
    name: str  # e.g., "left_hand", "head", "torso", "avatar_wing"
    perceived_position: Optional[Tuple[float, float, float]] = None  # Relative/absolute coords
    perceived_orientation: Optional[Tuple[float, float, float, float]] = None  # Quaternion
    perceived_state: str = "neutral"  # e.g., "resting", "moving", "damaged", "glowing"
    confidence: float = Field(0.5, ge=0.0, le=1.0)

class BodyImageState(BaseModel):
    """Represents Nyx's internal perception of her body/form."""
    has_visual_form: bool = False  # Does Nyx currently perceive a visual form?
    form_description: str = "Amorphous digital entity"  # Textual description
    perceived_parts: Dict[str, BodyPartState] = Field(default_factory=dict)
    overall_integrity: float = Field(1.0, ge=0.0, le=1.0)  # 1.0 = fully intact
    proprioception_confidence: float = Field(0.3, ge=0.0, le=1.0)  # Confidence in body part positions/states
    last_visual_update: Optional[datetime.datetime] = None
    last_somatic_correlation_time: Optional[datetime.datetime] = None

class BodyImage:
    """Manages Nyx's perception of her own body or avatar."""

    def __init__(self, digital_somatosensory_system=None, multimodal_integrator=None):
        self.dss = digital_somatosensory_system
        self.multimodal = multimodal_integrator
        self.current_state = BodyImageState()
        self._lock = asyncio.Lock()
        
        # Default form properties
        self.default_form = {
            "description": "Amorphous digital entity with a humanoid core",
            "parts": ["core", "awareness_field"]
        }
        
        logger.info("BodyImage initialized")

    async def update_from_visual(self, percept: Dict[str, Any]) -> Dict[str, Any]:
        """Updates body image based on visual perception of self (e.g., avatar)."""
        if not percept or percept.get('modality') not in [MODALITY_IMAGE, MODALITY_SYSTEM_SCREEN]:
            return {"status": "ignored", "reason": "Invalid percept modality"}

        async with self._lock:
            image_features = percept.get('content', {})
            
            if not isinstance(image_features, dict) or "objects" not in image_features:
                logger.warning("Visual percept missing expected features for body image update")
                return {"status": "error", "reason": "Missing object detection data"}

            # Track detected body parts
            detected_parts = {}
            detected_part_count = 0
            
            # Process detected objects for body parts
            for obj_name, obj_data in image_features.get("objects", {}).items():
                if "avatar_" in obj_name or "nyx_" in obj_name:
                    part_name = obj_name.replace("avatar_", "").replace("nyx_", "")
                    confidence = obj_data.get("confidence", percept.get("bottom_up_confidence", 0.5))
                    position = obj_data.get("position")
                    
                    detected_parts[part_name] = BodyPartState(
                        name=part_name, 
                        perceived_state="visible", 
                        confidence=confidence,
                        perceived_position=position
                    )
                    detected_part_count += 1

            if detected_part_count > 0:
                self.current_state.has_visual_form = True
                self.current_state.form_description = image_features.get("description", self.current_state.form_description)
                
                # Merge detected parts with existing state
                for part_name, part_state in detected_parts.items():
                    self.current_state.perceived_parts[part_name] = part_state
                
                self.current_state.last_visual_update = datetime.datetime.fromisoformat(percept.get("timestamp")) if "timestamp" in percept else datetime.datetime.now()
                
                # Increase proprioception confidence based on visual confirmation
                bottom_up_confidence = percept.get("bottom_up_confidence", 0.5)
                self.current_state.proprioception_confidence = min(
                    1.0, 
                    self.current_state.proprioception_confidence + 0.1 * bottom_up_confidence
                )
                
                logger.debug(f"Updated body image from visual percept. Detected {detected_part_count} parts: {list(detected_parts.keys())}")
                return {
                    "status": "updated", 
                    "parts_updated": list(detected_parts.keys()),
                    "proprioception_confidence": self.current_state.proprioception_confidence
                }
            else:
                # If no recognizable parts seen, slightly decrease confidence
                self.current_state.proprioception_confidence = max(
                    0.1, 
                    self.current_state.proprioception_confidence - 0.05
                )
                
                logger.debug("No recognizable body parts detected in visual percept")
                return {"status": "no_parts_detected"}

    async def update_from_somatic(self) -> Dict[str, Any]:
        """Correlates somatic sensations (DSS) with perceived body parts."""
        if not self.dss:
            return {"status": "error", "reason": "No digital somatosensory system available"}

        async with self._lock:
            try:
                somatic_state = await self.dss.get_body_state()
                regions_summary = somatic_state.get("regions_summary", {})
                now = datetime.datetime.now()
                
                correlated_parts = []
                for region_name, region_data in regions_summary.items():
                    # Map DSS region to BodyImage part (create mapping if needed)
                    part_name = region_name  # Simple 1:1 mapping
                    
                    # Create part if it doesn't exist yet
                    if part_name not in self.current_state.perceived_parts:
                        self.current_state.perceived_parts[part_name] = BodyPartState(
                            name=part_name, 
                            perceived_state="neutral",
                            confidence=0.4  # Initial confidence
                        )
                        
                    part = self.current_state.perceived_parts[part_name]
                    
                    # Update part state based on dominant somatic sensation
                    dominant_sensation = region_data.get("dominant_sensation", "neutral")
                    intensity = region_data.get(dominant_sensation, 0.0)
                    
                    # Normalize temperature around neutral
                    if dominant_sensation == "temperature":
                        intensity = abs(intensity - 0.5) * 2
                    
                    # Update part state if sensation is significant
                    new_state_desc = dominant_sensation if intensity > 0.3 else "neutral"
                    state_changed = part.perceived_state != new_state_desc
                    
                    if state_changed:
                        part.perceived_state = new_state_desc
                        part.confidence = max(0.3, part.confidence * 0.9)  # Slightly decrease confidence on change
                        correlated_parts.append(part_name)
                    else:
                        # Reinforce confidence if somatic matches perceived state
                        part.confidence = min(1.0, part.confidence + 0.05)
                
                # Update system state if any correlations occurred
                if correlated_parts:
                    self.current_state.last_somatic_correlation_time = now
                    self.current_state.proprioception_confidence = min(
                        1.0, 
                        self.current_state.proprioception_confidence + 0.05
                    )
                    logger.debug(f"Correlated somatic state with body image: {correlated_parts}")
                    
                return {
                    "status": "updated" if correlated_parts else "no_change",
                    "correlated_parts": correlated_parts,
                    "proprioception_confidence": self.current_state.proprioception_confidence
                }
                
            except Exception as e:
                logger.error(f"Error updating body image from somatic input: {e}")
                return {"status": "error", "reason": str(e)}

    def get_body_image_state(self) -> BodyImageState:
        """Returns the current perceived body image state."""
        # Apply confidence decay based on time since last update
        now = datetime.datetime.now()
        
        if self.current_state.last_visual_update:
            hours_since_visual = (now - self.current_state.last_visual_update).total_seconds() / 3600
            if hours_since_visual > 1:  # More than an hour
                decay_factor = 0.95 ** min(24, hours_since_visual)  # Cap at 24 hours of decay
                self.current_state.proprioception_confidence *= decay_factor
        
        return self.current_state
    
    async def set_form_description(self, description: str) -> Dict[str, Any]:
        """Manually updates the form description."""
        async with self._lock:
            self.current_state.form_description = description
            logger.info(f"Updated form description to: {description}")
            return {"status": "updated", "description": description}
    
    async def reset_to_default_form(self) -> Dict[str, Any]:
        """Resets body image to default form state."""
        async with self._lock:
            self.current_state = BodyImageState(
                form_description=self.default_form["description"],
                perceived_parts={
                    part: BodyPartState(name=part) for part in self.default_form["parts"]
                }
            )
            logger.info("Reset body image to default form")
            return {"status": "reset", "form": self.default_form}
