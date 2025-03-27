# nyx/core/body_image.py

import logging
import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import asyncio

# Assume MultimodalIntegrator and DSS are importable
# from nyx.core.multimodal_integrator import MODALITY_SYSTEM_SCREEN, MODALITY_IMAGE
# from nyx.core.digital_somatosensory_system import DigitalSomatosensorySystem

logger = logging.getLogger(__name__)

class BodyPartState(BaseModel):
    name: str # e.g., "left_hand", "head", "torso", "avatar_wing"
    perceived_position: Optional[Tuple[float, float, float]] = None # Relative or absolute coords
    perceived_orientation: Optional[Tuple[float, float, float, float]] = None # Quaternion
    perceived_state: str = "neutral" # e.g., "resting", "moving", "damaged", "glowing"
    confidence: float = 0.5

class BodyImageState(BaseModel):
    """Represents Nyx's internal perception of her body/form."""
    has_visual_form: bool = False # Does Nyx currently perceive a visual form?
    form_description: str = "Amorphous digital entity" # Textual description
    perceived_parts: Dict[str, BodyPartState] = Field(default_factory=dict)
    overall_integrity: float = Field(1.0, ge=0.0, le=1.0) # 1.0 = fully intact
    proprioception_confidence: float = Field(0.3, ge=0.0, le=1.0) # Confidence in body part positions/states
    last_visual_update: Optional[datetime.datetime] = None
    last_somatic_correlation_time: Optional[datetime.datetime] = None

class BodyImage:
    """Manages Nyx's perception of her own body or avatar."""

    def __init__(self, digital_somatosensory_system=None, multimodal_integrator=None):
        self.dss = digital_somatosensory_system
        self.multimodal = multimodal_integrator
        self.current_state = BodyImageState()
        self._lock = asyncio.Lock()

        logger.info("BodyImage initialized.")

    async def update_from_visual(self, percept: Any): # Percept type from MultimodalIntegrator
        """Updates body image based on visual perception of self (e.g., avatar)."""
        if not percept or percept.modality not in [MODALITY_IMAGE, MODALITY_SYSTEM_SCREEN]:
            return

        async with self._lock:
            # Placeholder: In reality, use CV/avatar analysis to identify body parts
            # from the image features (percept.content which should be ImageFeatures)
            image_features = percept.content
            if not isinstance(image_features, dict) or "objects" not in image_features:
                 logger.warning("Visual percept content missing expected features for body image update.")
                 return

            # Example: If 'avatar_hand' is detected in objects
            detected_parts = {}
            if "avatar_hand_left" in image_features["objects"]:
                 detected_parts["left_hand"] = BodyPartState(name="left_hand", perceived_state="visible", confidence=percept.bottom_up_confidence)
            if "avatar_head" in image_features["objects"]:
                 detected_parts["head"] = BodyPartState(name="head", perceived_state="visible", confidence=percept.bottom_up_confidence)
            # ... add more part detection logic ...

            if detected_parts:
                self.current_state.has_visual_form = True
                self.current_state.form_description = image_features.get("description", self.current_state.form_description)
                # Merge detected parts with existing state, updating confidence/position
                for part_name, part_state in detected_parts.items():
                     self.current_state.perceived_parts[part_name] = part_state # Simple overwrite for now
                self.current_state.last_visual_update = datetime.datetime.fromisoformat(percept.timestamp)
                # Increase proprioception confidence slightly based on visual confirmation
                self.current_state.proprioception_confidence = min(1.0, self.current_state.proprioception_confidence + 0.1 * percept.bottom_up_confidence)
                logger.debug(f"Updated body image from visual percept. Parts: {list(detected_parts.keys())}")
            else:
                 # If no recognizable parts seen, slightly decrease confidence
                 self.current_state.proprioception_confidence = max(0.1, self.current_state.proprioception_confidence - 0.05)


    async def update_from_somatic(self):
        """Correlates somatic sensations (DSS) with perceived body parts."""
        if not self.dss: return

        async with self._lock:
            somatic_state = await self.dss.get_body_state() # Get current DSS state
            regions_summary = somatic_state.get("regions_summary", {})
            now = datetime.datetime.now()

            correlated = False
            for region_name, region_data in regions_summary.items():
                 # Map DSS region to BodyImage part (may need explicit mapping)
                 part_name = region_name # Simple 1:1 mapping for now
                 if part_name in self.current_state.perceived_parts:
                      part = self.current_state.perceived_parts[part_name]
                      # Update part state based on dominant somatic sensation
                      dominant_sensation = region_data.get("dominant_sensation", "neutral")
                      intensity = 0.0
                      if dominant_sensation != "neutral":
                           # Find intensity (max value for that region in DSS)
                           intensity = region_data.get(dominant_sensation, 0.0)
                           if dominant_sensation == "temperature": intensity = abs(intensity - 0.5) * 2 # Deviation

                      new_state_desc = dominant_sensation if intensity > 0.3 else "neutral"

                      if part.perceived_state != new_state_desc:
                           part.perceived_state = new_state_desc
                           part.confidence = max(0.3, part.confidence * 0.9) # Slightly decrease confidence on state change via somatic
                           correlated = True
                      else:
                           # Reinforce confidence if somatic matches perceived state (even if neutral)
                           part.confidence = min(1.0, part.confidence + 0.05)

            if correlated:
                self.current_state.last_somatic_correlation_time = now
                # Increase proprioception confidence slightly when somatic data is correlated
                self.current_state.proprioception_confidence = min(1.0, self.current_state.proprioception_confidence + 0.05)
                logger.debug("Correlated somatic state with body image.")


    def get_body_image_state(self) -> BodyImageState:
        """Returns the current perceived body image state."""
        # Could add decay to proprioception confidence here if desired
        return self.current_state
