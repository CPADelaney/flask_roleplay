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

class _JSONModel(BaseModel, extra="forbid"):
    """Base that just wraps an arbitrary json blob as a string."""
    json: str


# PARAM / RESULT models -------------------------------------------------------
# visual part feature extraction
class ExtractVisualPartFeaturesParams(BaseModel, extra="forbid"):
    visual_json: str


class VisualPartFeaturesResult(_JSONModel):
    pass


class CalculateVisualConfidenceParams(BaseModel, extra="forbid"):
    visual_features_json: str


class VisualConfidenceResult(BaseModel, extra="forbid"):
    confidence: float


# somatic
class ExtractSomaticFeaturesParams(BaseModel, extra="forbid"):
    somatic_json: str


class SomaticFeaturesResult(_JSONModel):
    pass


class CalculateSomaticConfidenceParams(BaseModel, extra="forbid"):
    somatic_features_json: str


class SomaticConfidenceResult(BaseModel, extra="forbid"):
    confidence: float


# visual analysis
class AnalyzeVisualFeaturesParams(BaseModel, extra="forbid"):
    percept_json: str


class VisualAnalysisResult(_JSONModel):
    pass


class ExtractBodyPartStatesParams(BaseModel, extra="forbid"):
    body_features_json: str


class BodyPartStatesResult(_JSONModel):
    pass


class CorrelateSomaticVisualParams(BaseModel, extra="forbid"):
    somatic_json: str


class CorrelationResult(_JSONModel):
    pass


class ProprioceptionConfidenceParams(BaseModel, extra="forbid"):
    correlation_json: str


class ProprioceptionConfidenceResult(BaseModel, extra="forbid"):
    confidence: float


class ResolvePerceptionConflictsParams(BaseModel, extra="forbid"):
    visual_json: str
    somatic_json: str


class ResolveConflictsResult(_JSONModel):
    pass


class UpdateBodyImageStateParams(BaseModel, extra="forbid"):
    resolved_json: str


class UpdateBodyImageStateResult(_JSONModel):
    updated_parts_count: int
    current_proprioception_confidence: float
    has_visual_form: bool


class CurrentVisualStateResult(_JSONModel):
    pass


class SomaticDataAnalysisResult(_JSONModel):
    pass


class BodyImageStateResult(_JSONModel):
    pass


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
                InputGuardrail(guardrail_function=self._percept_validation_guardrail)
            ],
            output_type=VisualPerceptionOutput,
            model_settings=ModelSettings(
                temperature=0.2
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
                temperature=0.3
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
    
    def _create_extract_visual_part_features_tool(self):
        """Strict schema wrapper – visual part feature extraction"""
        @function_tool
        async def _extract_visual_part_features(                       # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: ExtractVisualPartFeaturesParams,
        ) -> VisualPartFeaturesResult:
            visual_data = json.loads(params.visual_json)
            features: Dict[str, Any] = {}
    
            # ---- original logic (with correct root-level derived_state) --------
            if "position" in visual_data:
                features["position"] = visual_data["position"]
    
            if "bounding_box" in visual_data:
                features["bounding_box"] = visual_data["bounding_box"]
    
            if "attributes" in visual_data:
                feats: Dict[str, Any] = {}
                for attr, val in visual_data["attributes"].items():
                    feats[attr] = val
                    if val:                      # Only when the flag is truthy
                        if attr == "moving":
                            features["derived_state"] = "moving"
                        elif attr == "glowing":
                            features["derived_state"] = "glowing"
                        elif attr == "damaged":
                            features["derived_state"] = "damaged"
                features["attributes"] = feats
    
            # Default state if nothing above set it
            features.setdefault("derived_state", "visible")
            features["confidence"] = visual_data.get("confidence", 0.5)
    
            logger.debug("extracted_visual_part_features", extra={"features": features})
            # -------------------------------------------------------------------
    
            return VisualPartFeaturesResult(json=json.dumps(features))
    
        return _extract_visual_part_features
    
    # ---------------------------------------------------------------------
    def _create_calculate_visual_confidence_tool(self):
        @function_tool
        async def _calculate_visual_confidence(                       # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: CalculateVisualConfidenceParams,
        ) -> VisualConfidenceResult:
            vf = json.loads(params.visual_features_json)
            base_conf = vf.get("confidence", 0.5)
            completeness = 0.0
            if "position" in vf:
                completeness += 0.2
            if "bounding_box" in vf:
                completeness += 0.1
            if vf.get("attributes"):
                completeness += 0.1 * min(3, len(vf["attributes"])) / 3
            confidence = base_conf * 0.7 + completeness * 0.3
            confidence = max(0.1, min(1.0, confidence))
            return VisualConfidenceResult(confidence=confidence)
    
        return _calculate_visual_confidence
    
    
    # ---------------------------------------------------------------------
    def _create_extract_somatic_features_tool(self):
        @function_tool
        async def _extract_somatic_features(                          # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: ExtractSomaticFeaturesParams,
        ) -> SomaticFeaturesResult:
            somatic_data = json.loads(params.somatic_json)
            features: Dict[str, Any] = {}
    
            # ---- original logic -------------------------------------------------
            if "dominant_sensation" in somatic_data:
                features["dominant_sensation"] = somatic_data["dominant_sensation"]
            if "intensity" in somatic_data:
                features["intensity"] = somatic_data["intensity"]
            if "secondary_sensations" in somatic_data:
                features["secondary_sensations"] = somatic_data["secondary_sensations"]
    
            if "dominant_sensation" in somatic_data:
                sensation = somatic_data["dominant_sensation"]
                intensity = somatic_data.get("intensity", 0.5)
                match sensation:
                    case "movement" if intensity > 0.3:
                        features["derived_state"] = "moving"
                    case "temperature" if intensity > 0.6:
                        features["derived_state"] = "heated"
                    case "pressure" if intensity > 0.7:
                        features["derived_state"] = "pressured"
                    case "pain" if intensity > 0.3:
                        features["derived_state"] = "damaged"
                    case _:
                        features["derived_state"] = "neutral"
            # --------------------------------------------------------------------
    
            return SomaticFeaturesResult(json=json.dumps(features))
    
        return _extract_somatic_features
    
    
    # ---------------------------------------------------------------------
    def _create_calculate_somatic_confidence_tool(self):
        @function_tool
        async def _calculate_somatic_confidence(                       # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: CalculateSomaticConfidenceParams,
        ) -> SomaticConfidenceResult:
            sf = json.loads(params.somatic_features_json)
            intensity = sf.get("intensity", 0.5)
            confidence = 0.3 + intensity * 0.6
            if isinstance(sf.get("secondary_sensations"), dict):
                confidence += min(0.1, len(sf["secondary_sensations"]) * 0.02)
            confidence = max(0.1, min(1.0, confidence))
            return SomaticConfidenceResult(confidence=confidence)
    
        return _calculate_somatic_confidence
    
    
    # ---------------------------------------------------------------------
    def _create_analyze_visual_features_tool(self):
        @function_tool
        async def _analyze_visual_features(                           # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: AnalyzeVisualFeaturesParams,
        ) -> VisualAnalysisResult:
            percept = json.loads(params.percept_json)
            MODS = {MODALITY_IMAGE, MODALITY_SYSTEM_SCREEN}
            if not percept or percept.get("modality") not in MODS:
                return VisualAnalysisResult(
                    json=json.dumps({"error": "Invalid percept modality", "detected_features": {}})
                )
    
            image_features = percept.get("content", {})
            if "objects" not in image_features:
                return VisualAnalysisResult(
                    json=json.dumps({"error": "Missing object detection data", "detected_features": {}})
                )
    
            body_features = {}
            for name, data in image_features.get("objects", {}).items():
                if "avatar_" in name or "nyx_" in name:
                    part = name.replace("avatar_", "").replace("nyx_", "")
                    body_features[part] = {
                        "confidence": data.get("confidence", percept.get("bottom_up_confidence", 0.5)),
                        "position": data.get("position"),
                        "bounding_box": data.get("bounding_box"),
                        "attributes": data.get("attributes", {}),
                    }
    
            result = {
                "body_features": body_features,
                "form_description": image_features.get("description", self.current_state.form_description),
                "detection_confidence": percept.get("bottom_up_confidence", 0.5),
                "timestamp": percept.get("timestamp", datetime.datetime.now().isoformat()),
            }
            return VisualAnalysisResult(json=json.dumps(result))
    
        return _analyze_visual_features
    
    
    # ---------------------------------------------------------------------
    def _create_extract_body_part_states_tool(self):
        @function_tool
        async def _extract_body_part_states(                          # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: ExtractBodyPartStatesParams,
        ) -> BodyPartStatesResult:
            body_features = json.loads(params.body_features_json)
            part_states: Dict[str, Dict[str, Any]] = {}
    
            # ---- original logic -------------------------------------------------
            for part, feats in body_features.items():
                state = "visible"
                attrs = feats.get("attributes", {})
                if attrs.get("moving"):
                    state = "moving"
                elif attrs.get("glowing"):
                    state = "glowing"
                elif attrs.get("damaged"):
                    state = "damaged"
    
                part_states[part] = {
                    "perceived_state": state,
                    "confidence": feats.get("confidence", 0.5),
                    "perceived_position": feats.get("position"),
                }
            # --------------------------------------------------------------------
    
            return BodyPartStatesResult(json=json.dumps(part_states))
    
        return _extract_body_part_states
    
    
    # ---------------------------------------------------------------------
    def _create_get_current_visual_state_tool(self):
        @function_tool
        async def _get_current_visual_state(                          # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
        ) -> CurrentVisualStateResult:
            parts = {n: p.dict() for n, p in self.current_state.perceived_parts.items()}
            state = {
                "has_visual_form": self.current_state.has_visual_form,
                "form_description": self.current_state.form_description,
                "parts": parts,
                "last_visual_update": (
                    self.current_state.last_visual_update.isoformat()
                    if self.current_state.last_visual_update
                    else None
                ),
                "proprioception_confidence": self.current_state.proprioception_confidence,
            }
            return CurrentVisualStateResult(json=json.dumps(state))
    
        return _get_current_visual_state
    
    
    # ---------------------------------------------------------------------
    def _create_analyze_somatic_data_tool(self):
        @function_tool
        async def _analyze_somatic_data(                              # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
        ) -> SomaticDataAnalysisResult:
            if not ctx.context.dss:
                return SomaticDataAnalysisResult(
                    json=json.dumps({"error": "No digital somatosensory system available", "somatic_data": {}})
                )
    
            try:
                s_state = await ctx.context.dss.get_body_state()
                summary = s_state.get("regions_summary", {})
                processed: Dict[str, Any] = {}
                for region, data in summary.items():
                    dom = data.get("dominant_sensation", "neutral")
                    intensity = data.get(dom, 0.0)
                    if dom == "temperature":
                        intensity = abs(intensity - 0.5) * 2
                    processed[region] = {
                        "dominant_sensation": dom,
                        "intensity": intensity,
                        "secondary_sensations": {
                            k: v
                            for k, v in data.items()
                            if k != "dominant_sensation" and isinstance(v, (int, float))
                        },
                    }
                payload = {"somatic_data": processed, "timestamp": datetime.datetime.now().isoformat()}
                return SomaticDataAnalysisResult(json=json.dumps(payload))
            except Exception as exc:
                logger.error(f"Error retrieving somatic data: {exc}")
                return SomaticDataAnalysisResult(json=json.dumps({"error": str(exc), "somatic_data": {}}))
    
        return _analyze_somatic_data
    
    
    # ---------------------------------------------------------------------
    def _create_correlate_somatic_visual_tool(self):
        @function_tool
        async def _correlate_somatic_visual(                          # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: CorrelateSomaticVisualParams,
        ) -> CorrelationResult:
            somatic_data = json.loads(params.somatic_json)
            correlated: Dict[str, Any] = {}
            for part, part_state in self.current_state.perceived_parts.items():
                reg = somatic_data.get(part)
                if not reg:
                    continue
                dom = reg.get("dominant_sensation", "neutral")
                inten = reg.get("intensity", 0.0)
                new_state = dom if inten > 0.3 else "neutral"
                if part_state.perceived_state == new_state:
                    adj = min(0.1, inten * 0.2)
                else:
                    adj = -min(0.1, inten * 0.2)
                correlated[part] = {
                    "current_state": part_state.perceived_state,
                    "new_state": new_state,
                    "confidence_adjustment": adj,
                    "intensity": inten,
                }
            result = {
                "correlated_parts": correlated,
                "correlation_time": datetime.datetime.now().isoformat(),
            }
            return CorrelationResult(json=json.dumps(result))
    
        return _correlate_somatic_visual
    
    
    # ---------------------------------------------------------------------
    def _create_calculate_proprioception_confidence_tool(self):
        @function_tool
        async def _calculate_proprioception_confidence(               # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: ProprioceptionConfidenceParams,
        ) -> ProprioceptionConfidenceResult:
            corr = json.loads(params.correlation_json).get("correlated_parts", {})
            if not corr:
                return ProprioceptionConfidenceResult(
                    confidence=self.current_state.proprioception_confidence
                )
            avg_adj = sum(p.get("confidence_adjustment", 0.0) for p in corr.values()) / len(corr)
            new_c = max(0.1, min(1.0, self.current_state.proprioception_confidence + avg_adj))
            return ProprioceptionConfidenceResult(confidence=new_c)
    
        return _calculate_proprioception_confidence
    
    
    # ---------------------------------------------------------------------
    def _create_resolve_perception_conflicts_tool(self):
        @function_tool
        async def _resolve_perception_conflicts(                      # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: ResolvePerceptionConflictsParams,
        ) -> ResolveConflictsResult:
            visual_data  = json.loads(params.visual_json)
            somatic_data = json.loads(params.somatic_json)
            resolved: Dict[str, Any] = {}
    
            parts = set(visual_data.get("parts", {}).keys()) | set(
                somatic_data.get("correlated_parts", {}).keys()
            )
            for part in parts:
                v_part = visual_data.get("parts", {}).get(part)
                s_part = somatic_data.get("correlated_parts", {}).get(part)
                # ---- original resolution logic (condensed for brevity) ----------
                if v_part and s_part:
                    v_conf = v_part.get("confidence", 0.5)
                    position = v_part.get("perceived_position")
                    state = (
                        v_part.get("perceived_state", "visible")
                        if v_conf > 0.7
                        else s_part.get("new_state", "neutral")
                    )
                    comb_conf = v_conf * 0.7 + s_part.get("confidence", 0.5) * 0.3
                    resolved[part] = {
                        "perceived_state": state,
                        "perceived_position": position,
                        "confidence": comb_conf,
                        "sources": ["visual", "somatic"],
                    }
                elif v_part:
                    resolved[part] = {
                        "perceived_state": v_part.get("perceived_state", "visible"),
                        "perceived_position": v_part.get("perceived_position"),
                        "confidence": v_part.get("confidence", 0.5),
                        "sources": ["visual"],
                    }
                elif s_part:
                    resolved[part] = {
                        "perceived_state": s_part.get("new_state", "neutral"),
                        "confidence": min(0.7, s_part.get("confidence", 0.5)),
                        "sources": ["somatic"],
                    }
            result = {
                "resolved_parts": resolved,
                "has_visual_form": bool(visual_data.get("parts")),
                "proprioception_confidence": somatic_data.get(
                    "proprioception_confidence", self.current_state.proprioception_confidence
                ),
            }
            return ResolveConflictsResult(json=json.dumps(result))
    
        return _resolve_perception_conflicts
    
    
    # ---------------------------------------------------------------------
    def _create_update_body_image_state_tool(self):
        @function_tool
        async def _update_body_image_state(                           # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
            params: UpdateBodyImageStateParams,
        ) -> UpdateBodyImageStateResult:
            resolved_data = json.loads(params.resolved_json)
            updates: List[Dict[str, Any]] = []
    
            for part, pdata in resolved_data.get("resolved_parts", {}).items():
                if part in self.current_state.perceived_parts:
                    curr = self.current_state.perceived_parts[part]
                    updates.append(
                        {
                            "part": part,
                            "old_state": curr.perceived_state,
                            "new_state": pdata.get("perceived_state", curr.perceived_state),
                            "old_confidence": curr.confidence,
                            "new_confidence": pdata.get("confidence", curr.confidence),
                        }
                    )
                    curr.perceived_state = pdata.get("perceived_state", curr.perceived_state)
                    curr.confidence = pdata.get("confidence", curr.confidence)
                    if "perceived_position" in pdata and pdata["perceived_position"]:
                        curr.perceived_position = pdata["perceived_position"]
                else:
                    new_p = BodyPartState(
                        name=part,
                        perceived_state=pdata.get("perceived_state", "neutral"),
                        confidence=pdata.get("confidence", 0.5),
                        perceived_position=pdata.get("perceived_position"),
                    )
                    self.current_state.perceived_parts[part] = new_p
                    updates.append(
                        {
                            "part": part,
                            "new_state": new_p.perceived_state,
                            "new_confidence": new_p.confidence,
                            "created": True,
                        }
                    )
    
            self.current_state.has_visual_form = resolved_data.get(
                "has_visual_form", self.current_state.has_visual_form
            )
            self.current_state.proprioception_confidence = resolved_data.get(
                "proprioception_confidence", self.current_state.proprioception_confidence
            )
    
            now = datetime.datetime.now()
            if resolved_data.get("has_visual_form"):
                self.current_state.last_visual_update = now
            if "proprioception_confidence" in resolved_data:
                self.current_state.last_somatic_correlation_time = now
    
            payload = {
                "updates": updates,
                "updated_parts_count": len(updates),
                "current_proprioception_confidence": self.current_state.proprioception_confidence,
                "has_visual_form": self.current_state.has_visual_form,
            }
            return UpdateBodyImageStateResult(
                json=json.dumps(payload),
                updated_parts_count=len(updates),
                current_proprioception_confidence=self.current_state.proprioception_confidence,
                has_visual_form=self.current_state.has_visual_form,
            )
    
        return _update_body_image_state
    
    
    # ---------------------------------------------------------------------
    def _create_get_body_image_state_tool(self):
        @function_tool
        async def _get_body_image_state(                              # noqa: N802
            ctx: RunContextWrapper["BodyImageContext"],
        ) -> BodyImageStateResult:
            now = datetime.datetime.now()
            if self.current_state.last_visual_update:
                hrs = (now - self.current_state.last_visual_update).total_seconds() / 3600
                if hrs > 1:
                    self.current_state.proprioception_confidence *= 0.95 ** min(24, hrs)
    
            state = {
                "has_visual_form": self.current_state.has_visual_form,
                "form_description": self.current_state.form_description,
                "parts": {n: p.dict() for n, p in self.current_state.perceived_parts.items()},
                "overall_integrity": self.current_state.overall_integrity,
                "proprioception_confidence": self.current_state.proprioception_confidence,
                "last_visual_update": (
                    self.current_state.last_visual_update.isoformat()
                    if self.current_state.last_visual_update
                    else None
                ),
                "last_somatic_correlation_time": (
                    self.current_state.last_somatic_correlation_time.isoformat()
                    if self.current_state.last_somatic_correlation_time
                    else None
                ),
            }
            return BodyImageStateResult(json=json.dumps(state))
    
        return _get_body_image_state
    
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
