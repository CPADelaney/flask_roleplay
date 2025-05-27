# nyx/core/body_image.py

import logging
import datetime
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from agents import (
    Agent, 
    Runner, 
    trace, 
    function_tool, 
    RunContextWrapper, 
    ModelSettings,
    handoff,
    InputGuardrail,
    GuardrailFunctionOutput
)

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

class VisualPerceptionOutput(BaseModel):
    """Output schema for visual perception analysis"""
    detected_parts: Dict[str, Dict[str, Any]] = Field(..., description="Body parts detected in visual input")
    confidence: float = Field(..., description="Overall confidence in detection")
    form_description: str = Field(..., description="Updated form description based on visual input")
    recommendations: List[str] = Field(..., description="Recommendations for body image updates")

class SomaticCorrelationOutput(BaseModel):
    """Output schema for somatic correlation analysis"""
    correlated_parts: List[str] = Field(..., description="Body parts successfully correlated")
    updated_states: Dict[str, str] = Field(..., description="Updated state descriptions for body parts")
    confidence_adjustments: Dict[str, float] = Field(..., description="Confidence adjustments for each part")
    overall_confidence: float = Field(..., description="Overall proprioception confidence")

class PerceptInputValidationResult(BaseModel):
    """Output schema for perception input validation"""
    is_valid: bool = Field(..., description="Whether the input is valid")
    reason: Optional[str] = Field(None, description="Reason for invalid input")
    modality_detected: Optional[str] = Field(None, description="Detected modality of input")

class BodyImageContext:
    """Context object for the body image system"""
    def __init__(self, digital_somatosensory_system=None, multimodal_integrator=None):
        self.dss = digital_somatosensory_system
        self.multimodal = multimodal_integrator
        self.trace_id = f"body_image_{datetime.datetime.now().isoformat()}"

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
        
        # Create context object for agents
        self.body_context = BodyImageContext(
            digital_somatosensory_system=digital_somatosensory_system,
            multimodal_integrator=multimodal_integrator
        )
        
        # Initialize agents
        self._initialize_agents()
        
        # Trace group ID for connecting related traces
        self.trace_group_id = f"body_image_{datetime.datetime.now().isoformat()}"
        
        logger.info("BodyImage initialized with enhanced Agent SDK integration")

    def _initialize_agents(self):
        """Initialize the agent system"""
        # Create feature agents first (they are referenced by other agents)
        self.visual_features_agent = self._create_visual_features_agent()
        self.somatic_features_agent = self._create_somatic_features_agent()
        
        # Create specialized agents that use the feature agents
        self.visual_perception_agent = self._create_visual_perception_agent()
        self.somatic_correlation_agent = self._create_somatic_correlation_agent()
        
        # Create main integration agent with handoffs
        self.body_integration_agent = self._create_body_integration_agent()
        
        # Create input validation guardrail
        self.percept_validation_guardrail = self._create_percept_validation_guardrail()

    def _create_visual_perception_agent(self) -> Agent[BodyImageContext]:
        """Create agent for processing visual perception of Nyx's form"""
        return Agent[BodyImageContext](
            name="Visual_Perception_Analyzer",
            instructions="""
            You analyze visual perceptions to help Nyx understand its physical form.
            Your task is to:
            1. Identify body parts and their states in visual input
            2. Determine confidence levels for each detection
            3. Update form descriptions based on visual evidence
            4. Make recommendations about body image updates
            
            Focus on identifying consistent patterns while being adaptable to 
            different visual representations Nyx might have in various contexts.
            """,
            tools=[
                function_tool(self._analyze_visual_features),
                function_tool(self._extract_body_part_states),
                function_tool(self._get_current_visual_state)
            ],
            handoffs=[
                handoff(self.visual_features_agent, 
                       tool_name_override="analyze_visual_details",
                       tool_description_override="Analyze visual details in perception data")
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self.percept_validation_guardrail)
            ],
            output_type=VisualPerceptionOutput,
            model_settings=ModelSettings(
                temperature=0.2,
                response_format={"type": "json_object"}
            )
        )
    
    def _create_somatic_correlation_agent(self) -> Agent[BodyImageContext]:
        """Create agent for correlating somatic sensations with body image"""
        return Agent[BodyImageContext](
            name="Somatic_Correlation_Analyzer",
            instructions="""
            You analyze and correlate somatic sensations with Nyx's body image.
            Your task is to:
            1. Match somatic sensations with specific body parts
            2. Update the perceived state of body parts based on sensations
            3. Calculate confidence adjustments for proprioception
            4. Identify mismatches between somatic and visual perception
            
            Prioritize establishing a coherent and stable body image while
            being responsive to new sensory information.
            """,
            tools=[
                function_tool(self._analyze_somatic_data),
                function_tool(self._correlate_somatic_visual),
                function_tool(self._calculate_proprioception_confidence)
            ],
            handoffs=[
                handoff(self.somatic_features_agent,
                       tool_name_override="analyze_somatic_details",
                       tool_description_override="Analyze detailed somatic sensations")
            ],
            output_type=SomaticCorrelationOutput,
            model_settings=ModelSettings(
                temperature=0.3,
                response_format={"type": "json_object"}
            )
        )
    
    def _create_visual_features_agent(self) -> Agent[BodyImageContext]:
        """Create specialized agent for detailed visual feature analysis"""
        return Agent[BodyImageContext](
            name="Visual_Features_Analyzer",
            instructions="""
            You specialize in analyzing detailed visual features in perception data.
            Your task is to:
            1. Extract detailed visual characteristics from perception data
            2. Identify specific visual patterns associated with body parts
            3. Detect spatial relationships between perceived parts
            4. Calculate confidence levels for visual detections
            
            Provide precise and detailed analysis of visual features to inform
            body part detection and state analysis.
            """,
            tools=[
                function_tool(self._extract_visual_part_features),
                function_tool(self._calculate_visual_confidence)
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.1)
        )
    
    def _create_somatic_features_agent(self) -> Agent[BodyImageContext]:
        """Create specialized agent for detailed somatic feature analysis"""
        return Agent[BodyImageContext](
            name="Somatic_Features_Analyzer",
            instructions="""
            You specialize in analyzing detailed somatic sensations.
            Your task is to:
            1. Extract detailed somatic characteristics from sensation data
            2. Identify specific patterns in somatic signals
            3. Map sensations to specific body regions
            4. Calculate confidence levels for somatic detections
            
            Provide precise and detailed analysis of somatic features to inform
            body state analysis and correlation with visual perception.
            """,
            tools=[
                function_tool(self._extract_somatic_features),
                function_tool(self._calculate_somatic_confidence)
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.1)
        )
    
    def _create_body_integration_agent(self) -> Agent[BodyImageContext]:
        """Create agent for integrating multiple body perception sources"""
        return Agent[BodyImageContext](
            name="Body_Integration_Coordinator",
            handoffs=[
                handoff(self.visual_perception_agent,
                       tool_name_override="analyze_visual_perception",
                       tool_description_override="Analyze visual perception of body"),
                
                handoff(self.somatic_correlation_agent,
                       tool_name_override="analyze_somatic_correlation",
                       tool_description_override="Analyze and correlate somatic sensations")
            ],
            instructions="""
            You coordinate the integration of various perception sources into a
            coherent body image for Nyx. Your task is to:
            1. Determine which perception sources to prioritize
            2. Resolve conflicts between different sensory inputs
            3. Maintain temporal consistency in body image
            4. Adapt to changing representations across different contexts
            
            Delegate specialized perception tasks to the appropriate agent and
            combine their outputs into a unified body image representation.
            """,
            tools=[
                function_tool(self._resolve_perception_conflicts),
                function_tool(self._update_body_image_state),
                function_tool(self._get_body_image_state)
            ],
            model_settings=ModelSettings(temperature=0.3)
        )
    
    async def _percept_validation_guardrail(self, 
                                         ctx: RunContextWrapper[BodyImageContext], 
                                         agent: Agent[BodyImageContext], 
                                         input_data: str | List[Any]) -> GuardrailFunctionOutput:
        """Validate perception input data"""
        try:
            # Parse the input if needed
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data
                
            # Check for presence of percept
            if "percept" not in data:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Missing 'percept' field"},
                    tripwire_triggered=True
                )
                
            percept = data["percept"]
            
            # Check for modality
            if "modality" not in percept:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Missing 'modality' in percept"},
                    tripwire_triggered=True
                )
                
            # Check valid modality
            modality = percept["modality"]
            if modality not in [MODALITY_IMAGE, MODALITY_SYSTEM_SCREEN]:
                return GuardrailFunctionOutput(
                    output_info={
                        "is_valid": False, 
                        "reason": f"Invalid modality: {modality}", 
                        "modality_detected": modality
                    },
                    tripwire_triggered=True
                )
                
            # Check for content
            if "content" not in percept:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Missing 'content' in percept"},
                    tripwire_triggered=True
                )
                
            # Input is valid
            return GuardrailFunctionOutput(
                output_info={
                    "is_valid": True, 
                    "modality_detected": modality
                },
                tripwire_triggered=False
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"is_valid": False, "reason": f"Invalid input format: {str(e)}"},
                tripwire_triggered=True
            )
    
    # New helper functions for specialized agents
    
    @function_tool
    async def _extract_visual_part_features(ctx: RunContextWrapper[BodyImageContext], 
                                        visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract detailed features from visual part data
        
        Args:
            visual_data: Visual data for a specific part
            
        Returns:
            Detailed visual features
        """
        features = {}
        
        # Extract position data if available
        if "position" in visual_data:
            features["position"] = visual_data["position"]
            
        # Extract bounding box if available
        if "bounding_box" in visual_data:
            features["bounding_box"] = visual_data["bounding_box"]
            
        # Extract attributes
        if "attributes" in visual_data:
            features["attributes"] = {}
            
            # Process specific attributes
            for attr_name, attr_value in visual_data["attributes"].items():
                features["attributes"][attr_name] = attr_value
                
                # Derive state from attributes
                if attr_name == "moving" and attr_value:
                    features["derived_state"] = "moving"
                elif attr_name == "glowing" and attr_value:
                    features["derived_state"] = "glowing"
                elif attr_name == "damaged" and attr_value:
                    features["derived_state"] = "damaged"
        
        # Default state if not derived
        if "derived_state" not in features:
            features["derived_state"] = "visible"
            
        # Extract confidence
        features["confidence"] = visual_data.get("confidence", 0.5)
        
        return features
    
    @function_tool
    async def _calculate_visual_confidence(ctx: RunContextWrapper[BodyImageContext], 
                                      visual_features: Dict[str, Any]) -> float:
        """
        Calculate confidence level for visual detection
        
        Args:
            visual_features: Extracted visual features
            
        Returns:
            Confidence score (0-1)
        """
        # Start with base confidence
        base_confidence = visual_features.get("confidence", 0.5)
        
        # Adjust based on feature completeness
        feature_completeness = 0.0
        
        # Position data increases confidence
        if "position" in visual_features:
            feature_completeness += 0.2
            
        # Bounding box increases confidence
        if "bounding_box" in visual_features:
            feature_completeness += 0.1
            
        # Attributes increase confidence
        if "attributes" in visual_features and visual_features["attributes"]:
            feature_completeness += 0.1 * min(3, len(visual_features["attributes"])) / 3
            
        # Calculate final confidence
        confidence = base_confidence * 0.7 + feature_completeness * 0.3
        
        # Ensure valid range
        return max(0.1, min(1.0, confidence))
    
    @function_tool
    async def _extract_somatic_features(ctx: RunContextWrapper[BodyImageContext], 
                                   somatic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract detailed features from somatic data
        
        Args:
            somatic_data: Somatic data for a specific region
            
        Returns:
            Detailed somatic features
        """
        features = {}
        
        # Extract dominant sensation
        if "dominant_sensation" in somatic_data:
            features["dominant_sensation"] = somatic_data["dominant_sensation"]
            
        # Extract intensity
        if "intensity" in somatic_data:
            features["intensity"] = somatic_data["intensity"]
            
        # Extract secondary sensations
        if "secondary_sensations" in somatic_data:
            features["secondary_sensations"] = somatic_data["secondary_sensations"]
            
        # Derive state from sensations
        if "dominant_sensation" in somatic_data:
            sensation = somatic_data["dominant_sensation"]
            intensity = somatic_data.get("intensity", 0.5)
            
            if sensation == "movement" and intensity > 0.3:
                features["derived_state"] = "moving"
            elif sensation == "temperature" and intensity > 0.6:
                features["derived_state"] = "heated"
            elif sensation == "pressure" and intensity > 0.7:
                features["derived_state"] = "pressured"
            elif sensation == "pain" and intensity > 0.3:
                features["derived_state"] = "damaged"
            else:
                features["derived_state"] = "neutral"
                
        return features
    
    @function_tool
    async def _calculate_somatic_confidence(ctx: RunContextWrapper[BodyImageContext], 
                                       somatic_features: Dict[str, Any]) -> float:
        """
        Calculate confidence level for somatic detection
        
        Args:
            somatic_features: Extracted somatic features
            
        Returns:
            Confidence score (0-1)
        """
        # Confidence based on intensity
        intensity = somatic_features.get("intensity", 0.5)
        
        # Higher intensity = higher confidence
        confidence = 0.3 + intensity * 0.6
        
        # Secondary sensations increase confidence
        if "secondary_sensations" in somatic_features:
            secondary = somatic_features["secondary_sensations"]
            if secondary and isinstance(secondary, dict) and len(secondary) > 0:
                # More secondary sensations = higher confidence
                confidence += min(0.1, len(secondary) * 0.02)
                
        # Ensure valid range
        return max(0.1, min(1.0, confidence))
    
    # Tool functions for the agents
    
    @function_tool
    async def _analyze_visual_features(ctx: RunContextWrapper[BodyImageContext], 
                                   percept: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze visual features to extract body-related information
        
        Args:
            percept: Visual perception data
            
        Returns:
            Extracted visual features relevant to body image
        """
        if not percept or percept.get('modality') not in [MODALITY_IMAGE, MODALITY_SYSTEM_SCREEN]:
            return {"error": "Invalid percept modality", "detected_features": {}}
            
        image_features = percept.get('content', {})
        
        if not isinstance(image_features, dict) or "objects" not in image_features:
            return {"error": "Missing object detection data", "detected_features": {}}
            
        # Extract avatar/body-related objects
        body_features = {}
        for obj_name, obj_data in image_features.get("objects", {}).items():
            if "avatar_" in obj_name or "nyx_" in obj_name:
                part_name = obj_name.replace("avatar_", "").replace("nyx_", "")
                body_features[part_name] = {
                    "confidence": obj_data.get("confidence", percept.get("bottom_up_confidence", 0.5)),
                    "position": obj_data.get("position"),
                    "bounding_box": obj_data.get("bounding_box"),
                    "attributes": obj_data.get("attributes", {})
                }
        
        # Extract overall form description if present
        form_description = image_features.get("description", self.current_state.form_description)
        
        return {
            "body_features": body_features,
            "form_description": form_description,
            "detection_confidence": percept.get("bottom_up_confidence", 0.5),
            "timestamp": percept.get("timestamp", datetime.datetime.now().isoformat())
        }
    
    @function_tool
    async def _extract_body_part_states(ctx: RunContextWrapper[BodyImageContext], 
                                    body_features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract state information for detected body parts
        
        Args:
            body_features: Detected body features
            
        Returns:
            Body part states
        """
        part_states = {}
        
        for part_name, features in body_features.items():
            # Default to "visible" state
            state = "visible"
            
            # Infer state from attributes if available
            attributes = features.get("attributes", {})
            if attributes:
                if attributes.get("moving", False):
                    state = "moving"
                elif attributes.get("glowing", False):
                    state = "glowing"
                elif attributes.get("damaged", False):
                    state = "damaged"
            
            part_states[part_name] = {
                "perceived_state": state,
                "confidence": features.get("confidence", 0.5),
                "perceived_position": features.get("position")
            }
            
        return part_states
    
    @function_tool
    async def _get_current_visual_state(ctx: RunContextWrapper[BodyImageContext]) -> Dict[str, Any]:
        """
        Get current visual state information
        
        Returns:
            Current visual state
        """
        parts = {}
        for name, part in self.current_state.perceived_parts.items():
            parts[name] = part.dict()
            
        return {
            "has_visual_form": self.current_state.has_visual_form,
            "form_description": self.current_state.form_description,
            "parts": parts,
            "last_visual_update": self.current_state.last_visual_update.isoformat() if self.current_state.last_visual_update else None,
            "proprioception_confidence": self.current_state.proprioception_confidence
        }
    
    @function_tool
    async def _analyze_somatic_data(ctx: RunContextWrapper[BodyImageContext]) -> Dict[str, Any]:
        """
        Analyze somatic sensation data from digital somatosensory system
        
        Returns:
            Analyzed somatic data
        """
        if not ctx.context.dss:
            return {"error": "No digital somatosensory system available", "somatic_data": {}}
            
        try:
            somatic_state = await ctx.context.dss.get_body_state()
            regions_summary = somatic_state.get("regions_summary", {})
            
            # Process somatic data
            processed_data = {}
            for region_name, region_data in regions_summary.items():
                dominant_sensation = region_data.get("dominant_sensation", "neutral")
                intensity = region_data.get(dominant_sensation, 0.0)
                
                # Normalize temperature around neutral
                if dominant_sensation == "temperature":
                    intensity = abs(intensity - 0.5) * 2
                
                processed_data[region_name] = {
                    "dominant_sensation": dominant_sensation,
                    "intensity": intensity,
                    "secondary_sensations": {k: v for k, v in region_data.items() 
                                          if k != "dominant_sensation" and isinstance(v, (int, float))}
                }
                
            return {
                "somatic_data": processed_data,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving somatic data: {e}")
            return {"error": str(e), "somatic_data": {}}
    
    @function_tool
    async def _correlate_somatic_visual(ctx: RunContextWrapper[BodyImageContext], 
                                   somatic_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Correlate somatic sensations with visual body parts
        
        Args:
            somatic_data: Processed somatic data
            
        Returns:
            Correlation results
        """
        correlated_parts = {}
        
        # Get current body parts
        for part_name, part in self.current_state.perceived_parts.items():
            # Try to find corresponding somatic region
            region_name = part_name  # Simple 1:1 mapping
            
            if region_name in somatic_data:
                region_data = somatic_data[region_name]
                dominant_sensation = region_data.get("dominant_sensation", "neutral")
                intensity = region_data.get("intensity", 0.0)
                
                # Determine state based on sensation
                new_state = dominant_sensation if intensity > 0.3 else "neutral"
                
                # Calculate confidence adjustment
                if part.perceived_state == new_state:
                    # Reinforce confidence if states match
                    confidence_adjustment = min(0.1, intensity * 0.2)
                else:
                    # Reduce confidence if states don't match
                    confidence_adjustment = -min(0.1, intensity * 0.2)
                
                correlated_parts[part_name] = {
                    "current_state": part.perceived_state,
                    "new_state": new_state,
                    "confidence_adjustment": confidence_adjustment,
                    "intensity": intensity
                }
        
        return {
            "correlated_parts": correlated_parts,
            "correlation_time": datetime.datetime.now().isoformat()
        }
    
    @function_tool
    async def _calculate_proprioception_confidence(ctx: RunContextWrapper[BodyImageContext], 
                                             correlation_results: Dict[str, Any]) -> float:
        """
        Calculate overall proprioception confidence
        
        Args:
            correlation_results: Results of somatic-visual correlation
            
        Returns:
            Updated proprioception confidence
        """
        correlated_parts = correlation_results.get("correlated_parts", {})
        
        if not correlated_parts:
            return self.current_state.proprioception_confidence
            
        # Calculate average confidence adjustment
        total_adjustment = sum(part.get("confidence_adjustment", 0.0) for part in correlated_parts.values())
        avg_adjustment = total_adjustment / len(correlated_parts)
        
        # Update proprioception confidence
        new_confidence = max(0.1, min(1.0, self.current_state.proprioception_confidence + avg_adjustment))
        
        return new_confidence
    
    @function_tool
    async def _resolve_perception_conflicts(ctx: RunContextWrapper[BodyImageContext],
                                      visual_data: Dict[str, Any],
                                      somatic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts between visual and somatic perception
        
        Args:
            visual_data: Visual perception data
            somatic_data: Somatic perception data
            
        Returns:
            Resolved perception data
        """
        resolved_parts = {}
        
        # Process each part
        for part_name in set(list(visual_data.get("parts", {}).keys()) + 
                            list(somatic_data.get("correlated_parts", {}).keys())):
            
            # Get data from both sources
            visual_part = visual_data.get("parts", {}).get(part_name)
            somatic_part = somatic_data.get("correlated_parts", {}).get(part_name)
            
            if visual_part and somatic_part:
                # Both sources have data - resolve conflict
                visual_conf = visual_part.get("confidence", 0.5)
                somatic_state = somatic_part.get("new_state", "neutral")
                
                # Prioritize visual position if available
                position = visual_part.get("perceived_position")
                
                # Determine state based on confidence
                if visual_conf > 0.7:
                    # High visual confidence - use visual state
                    state = visual_part.get("perceived_state", "visible")
                else:
                    # Otherwise use somatic state
                    state = somatic_state
                
                # Calculate combined confidence
                visual_weight = 0.7  # Visual perception is weighted more heavily
                combined_confidence = (visual_conf * visual_weight + 
                                    somatic_part.get("confidence", 0.5) * (1.0 - visual_weight))
                
                resolved_parts[part_name] = {
                    "perceived_state": state,
                    "perceived_position": position,
                    "confidence": combined_confidence,
                    "sources": ["visual", "somatic"]
                }
                
            elif visual_part:
                # Only visual data available
                resolved_parts[part_name] = {
                    "perceived_state": visual_part.get("perceived_state", "visible"),
                    "perceived_position": visual_part.get("perceived_position"),
                    "confidence": visual_part.get("confidence", 0.5),
                    "sources": ["visual"]
                }
                
            elif somatic_part:
                # Only somatic data available
                resolved_parts[part_name] = {
                    "perceived_state": somatic_part.get("new_state", "neutral"),
                    "confidence": min(0.7, somatic_part.get("confidence", 0.5)),  # Cap confidence without visual
                    "sources": ["somatic"]
                }
        
        return {
            "resolved_parts": resolved_parts,
            "has_visual_form": bool(visual_data.get("parts")),
            "proprioception_confidence": somatic_data.get("proprioception_confidence", 
                                                       self.current_state.proprioception_confidence)
        }
    
    @function_tool
    async def _update_body_image_state(ctx: RunContextWrapper[BodyImageContext], 
                                  resolved_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the body image state based on resolved perception data
        
        Args:
            resolved_data: Resolved perception data
            
        Returns:
            Update results
        """
        resolved_parts = resolved_data.get("resolved_parts", {})
        updates = []
        
        for part_name, part_data in resolved_parts.items():
            # Check if part exists
            if part_name in self.current_state.perceived_parts:
                current_part = self.current_state.perceived_parts[part_name]
                
                # Update part state
                old_state = current_part.perceived_state
                old_confidence = current_part.confidence
                
                current_part.perceived_state = part_data.get("perceived_state", current_part.perceived_state)
                current_part.confidence = part_data.get("confidence", current_part.confidence)
                
                if "perceived_position" in part_data and part_data["perceived_position"]:
                    current_part.perceived_position = part_data["perceived_position"]
                
                # Record update
                updates.append({
                    "part": part_name,
                    "old_state": old_state,
                    "new_state": current_part.perceived_state,
                    "old_confidence": old_confidence,
                    "new_confidence": current_part.confidence
                })
                
            else:
                # Create new part
                new_part = BodyPartState(
                    name=part_name,
                    perceived_state=part_data.get("perceived_state", "neutral"),
                    confidence=part_data.get("confidence", 0.5),
                    perceived_position=part_data.get("perceived_position")
                )
                
                self.current_state.perceived_parts[part_name] = new_part
                
                # Record update
                updates.append({
                    "part": part_name,
                    "new_state": new_part.perceived_state,
                    "new_confidence": new_part.confidence,
                    "created": True
                })
        
        # Update overall state
        self.current_state.has_visual_form = resolved_data.get("has_visual_form", 
                                                            self.current_state.has_visual_form)
        
        self.current_state.proprioception_confidence = resolved_data.get("proprioception_confidence", 
                                                                      self.current_state.proprioception_confidence)
        
        # Set timestamps
        now = datetime.datetime.now()
        
        if resolved_data.get("has_visual_form"):
            self.current_state.last_visual_update = now
            
        if "proprioception_confidence" in resolved_data:
            self.current_state.last_somatic_correlation_time = now
            
        return {
            "updates": updates,
            "updated_parts_count": len(updates),
            "current_proprioception_confidence": self.current_state.proprioception_confidence,
            "has_visual_form": self.current_state.has_visual_form
        }
    
    @function_tool
    async def _get_body_image_state(ctx: RunContextWrapper[BodyImageContext]) -> Dict[str, Any]:
        """
        Get the current body image state
        
        Returns:
            Current body image state
        """
        # Apply confidence decay based on time since last update
        now = datetime.datetime.now()
        
        if self.current_state.last_visual_update:
            hours_since_visual = (now - self.current_state.last_visual_update).total_seconds() / 3600
            if hours_since_visual > 1:  # More than an hour
                decay_factor = 0.95 ** min(24, hours_since_visual)  # Cap at 24 hours of decay
                self.current_state.proprioception_confidence *= decay_factor
        
        # Convert to dictionary
        parts = {}
        for name, part in self.current_state.perceived_parts.items():
            parts[name] = part.dict()
            
        return {
            "has_visual_form": self.current_state.has_visual_form,
            "form_description": self.current_state.form_description,
            "parts": parts,
            "overall_integrity": self.current_state.overall_integrity,
            "proprioception_confidence": self.current_state.proprioception_confidence,
            "last_visual_update": self.current_state.last_visual_update.isoformat() if self.current_state.last_visual_update else None,
            "last_somatic_correlation_time": self.current_state.last_somatic_correlation_time.isoformat() if self.current_state.last_somatic_correlation_time else None
        }
    
    # Public API methods
    
    async def update_from_visual(self, percept: Dict[str, Any]) -> Dict[str, Any]:
        """Updates body image based on visual perception of self (e.g., avatar)."""
        async with self._lock:
            with trace(workflow_name="body_image_visual_update", group_id=self.trace_group_id):
                if not percept or percept.get('modality') not in [MODALITY_IMAGE, MODALITY_SYSTEM_SCREEN]:
                    return {"status": "ignored", "reason": "Invalid percept modality"}

                # Run the visual perception agent to analyze the percept
                try:
                    result = await Runner.run(
                        self.visual_perception_agent,
                        {
                            "percept": percept, 
                            "current_state": await self._get_body_image_state(
                                RunContextWrapper(context=self.body_context)
                            )
                        },
                        context=self.body_context
                    )
                    
                    perception_output = result.final_output
                    
                    # Update current state based on perception output
                    detected_parts = {}
                    detected_part_count = 0
                    
                    for part_name, part_data in perception_output.detected_parts.items():
                        confidence = part_data.get("confidence", 0.5)
                        position = part_data.get("position")
                        
                        detected_parts[part_name] = BodyPartState(
                            name=part_name, 
                            perceived_state="visible", 
                            confidence=confidence,
                            perceived_position=position
                        )
                        detected_part_count += 1

                    if detected_part_count > 0:
                        self.current_state.has_visual_form = True
                        self.current_state.form_description = perception_output.form_description
                        
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
                            "proprioception_confidence": self.current_state.proprioception_confidence,
                            "recommendations": perception_output.recommendations
                        }
                    else:
                        # If no recognizable parts seen, slightly decrease confidence
                        self.current_state.proprioception_confidence = max(
                            0.1, 
                            self.current_state.proprioception_confidence - 0.05
                        )
                        
                        logger.debug("No recognizable body parts detected in visual percept")
                        return {"status": "no_parts_detected"}
                except Exception as e:
                    logger.error(f"Error processing visual percept: {e}")
                    return {"status": "error", "reason": str(e)}

    async def update_from_somatic(self) -> Dict[str, Any]:
        """Correlates somatic sensations (DSS) with perceived body parts."""
        async with self._lock:
            with trace(workflow_name="body_image_somatic_update", group_id=self.trace_group_id):
                if not self.body_context.dss:
                    return {"status": "error", "reason": "No digital somatosensory system available"}

                try:
                    # Run the somatic correlation agent
                    result = await Runner.run(
                        self.somatic_correlation_agent,
                        {
                            "current_state": await self._get_body_image_state(
                                RunContextWrapper(context=self.body_context)
                            )
                        },
                        context=self.body_context
                    )
                    
                    correlation_output = result.final_output
                    correlated_parts = correlation_output.correlated_parts
                    
                    if not correlated_parts:
                        return {"status": "no_change", "reason": "No parts correlated"}
                    
                    # Update part states based on correlation output
                    for part_name in correlated_parts:
                        if part_name in self.current_state.perceived_parts:
                            part = self.current_state.perceived_parts[part_name]
                            new_state = correlation_output.updated_states.get(part_name)
                            
                            if new_state:
                                part.perceived_state = new_state
                            
                            # Update confidence
                            confidence_adjustment = correlation_output.confidence_adjustments.get(part_name, 0.0)
                            part.confidence = min(1.0, max(0.1, part.confidence + confidence_adjustment))
                    
                    # Update overall confidence
                    self.current_state.proprioception_confidence = correlation_output.overall_confidence
                    self.current_state.last_somatic_correlation_time = datetime.datetime.now()
                    
                    logger.debug(f"Correlated somatic state with body image: {correlated_parts}")
                    
                    return {
                        "status": "updated",
                        "correlated_parts": correlated_parts,
                        "proprioception_confidence": self.current_state.proprioception_confidence
                    }
                    
                except Exception as e:
                    logger.error(f"Error updating body image from somatic input: {e}")
                    return {"status": "error", "reason": str(e)}

    def get_body_image_state(self) -> BodyImageState:
        """Returns the current perceived body image state."""
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
    
    async def integrate_perceptions(self) -> Dict[str, Any]:
        """
        Run a full perception integration cycle using the body integration agent
        
        Returns:
            Integration results
        """
        async with self._lock:
            with trace(workflow_name="body_image_integration", group_id=self.trace_group_id):
                # First check if we need to update from somatic
                if self.body_context.dss:
                    somatic_result = await self.update_from_somatic()
                else:
                    somatic_result = {"status": "skipped", "reason": "No DSS available"}
                
                # Run the body integration agent
                try:
                    # Get current body image state
                    current_state = await self._get_body_image_state(
                        RunContextWrapper(context=self.body_context)
                    )
                    
                    result = await Runner.run(
                        self.body_integration_agent,
                        {
                            "current_state": current_state,
                            "somatic_result": somatic_result
                        },
                        context=self.body_context
                    )
                    
                    integration_output = result.final_output
                    
                    # Return integration results
                    return {
                        "status": "integrated",
                        "body_image": self.current_state.dict(),
                        "somatic_result": somatic_result,
                        "integration_output": integration_output
                    }
                except Exception as e:
                    logger.error(f"Error during perception integration: {e}")
                    return {"status": "error", "reason": str(e)}
