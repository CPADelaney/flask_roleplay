# nyx/core/autobiographical_narrative.py

import logging
import datetime
import json
import uuid
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    trace, 
    function_tool, 
    RunContextWrapper,
    handoff,
    InputGuardrail,
    GuardrailFunctionOutput
)

logger = logging.getLogger(__name__)

class _JSONModel(BaseModel, extra="forbid"):
    json: str                              # nothing else → always strict
# --------------------------------------------------------------------


class MemoryEmotionAnalysisParams(BaseModel, extra="forbid"):
    memories_json: str                     # single STRING field


class MemoryEmotionAnalysisResult(_JSONModel):
    """Tool returns a JSON blob stringified in `json`."""
    pass

class NarrativeSegment(BaseModel):
    """A segment of the autobiographical narrative."""
    segment_id: str = Field(default_factory=lambda: f"seg_{uuid.uuid4().hex[:8]}")
    title: Optional[str] = None
    summary: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    key_events: List[str] = Field(default_factory=list)  # Memory IDs of key events
    themes: List[str] = Field(default_factory=list)
    emotional_arc: Optional[str] = None  # e.g., "rising hope", "gradual disillusionment"
    identity_impact: Optional[str] = None  # How this period shaped identity

class NarrativeContext:
    """Context object for narrative generation systems"""
    def __init__(self, memory_orchestrator=None, identity_evolution=None, relationship_manager=None):
        self.memory_orchestrator = memory_orchestrator
        self.identity_evolution = identity_evolution
        self.relationship_manager = relationship_manager
        self.trace_id = f"NyxAutobiography_{datetime.datetime.now().isoformat()}"

class NarrativeGenerationRequest(BaseModel):
    """Request model for narrative generation"""
    time_period: Dict[str, str]
    key_memories: List[str]
    identity_context: Dict[str, Any]
    existing_narrative_summary: str

class MemoryAnalysisOutput(BaseModel):
    """Output from memory analysis agent"""
    key_themes: List[str]
    emotional_patterns: Dict[str, float]
    identity_development: Dict[str, Any]
    significant_events: List[Dict[str, Any]]
    recommended_focus: str

class IdentityAnalysisOutput(BaseModel):
    """Output from identity analysis agent"""
    identity_shifts: List[Dict[str, Any]]
    trait_developments: Dict[str, float]
    core_values: List[str]
    identity_stability: float
    identity_impact_assessment: str

class NarrativeValidationOutput(BaseModel):
    """Output schema for narrative validation"""
    is_valid: bool = Field(..., description="Whether the narrative is valid")
    issues: List[str] = Field(default_factory=list, description="Issues found in the narrative")
    coherence_score: float = Field(..., description="Narrative coherence score (0-1)")
    continuity_rating: float = Field(..., description="Continuity with existing narrative (0-1)")

# Fixed parameter models - all use single JSON string fields
class IdentifyMemoryThemesParams(BaseModel, extra="forbid"):
    """Parameters for memory theme identification"""
    memories_json: str

class CalculateMemorySignificanceParams(BaseModel, extra="forbid"):
    """Parameters for memory significance calculation"""
    memories_json: str

class AnalyzeIdentityShiftsParams(BaseModel, extra="forbid"):
    """Parameters for identity shift analysis"""
    identity_state_json: str

class ExtractCoreValuesParams(BaseModel, extra="forbid"):
    """Parameters for core values extraction"""
    identity_state_json: str

class CalculateIdentityStabilityParams(BaseModel, extra="forbid"):
    """Parameters for identity stability calculation"""
    identity_state_json: str

class CheckNarrativeCoherenceParams(BaseModel, extra="forbid"):
    """Parameters for narrative coherence check"""
    payload_json: str  # Contains both narrative and existing_segments

class VerifyContinuityParams(BaseModel, extra="forbid"):
    """Parameters for continuity verification"""
    payload_json: str  # Contains both narrative and existing_segments

class ValidateEmotionalAuthenticityParams(BaseModel, extra="forbid"):
    """Parameters for emotional authenticity validation"""
    payload_json: str  # Contains both narrative and memories

# Result models that use _JSONModel wrapper for flexibility
class IdentifyMemoryThemesResult(_JSONModel):
    pass

class CalculateMemorySignificanceResult(_JSONModel):
    pass

class AnalyzeIdentityShiftsResult(_JSONModel):
    pass

class ExtractCoreValuesResult(_JSONModel):
    pass

class CalculateIdentityStabilityResult(_JSONModel):
    pass

class CheckNarrativeCoherenceResult(_JSONModel):
    pass

class VerifyContinuityResult(_JSONModel):
    pass

class ValidateEmotionalAuthenticityResult(_JSONModel):
    pass

class RetrieveSignificantMemoriesResult(_JSONModel):
    pass

class GetIdentitySnapshotResult(_JSONModel):
    pass

class AutobiographicalNarrative:
    """Constructs and maintains Nyx's coherent life story."""

    def __init__(self, memory_orchestrator=None, identity_evolution=None, relationship_manager=None):
        self.memory_orchestrator = memory_orchestrator  # For retrieving key memories
        self.identity_evolution = identity_evolution  # For identity context
        self.relationship_manager = relationship_manager  # For user-specific narrative elements

        self.narrative_segments: List[NarrativeSegment] = []
        self.current_narrative_summary: str = "My story is just beginning."
        self.last_update_time = datetime.datetime.now()
        self.update_interval_hours = 24  # How often to update the narrative
        
        # Initialize the narrative context
        self.narrative_context = NarrativeContext(
            memory_orchestrator=memory_orchestrator,
            identity_evolution=identity_evolution,
            relationship_manager=relationship_manager
        )
        
        # Initialize agent system
        self._initialize_agents()
        
        self.trace_group_id = "NyxAutobiography"

        logger.info("AutobiographicalNarrative initialized with enhanced Agent SDK integration")

    def _initialize_agents(self):
        """Initialize all agents needed for the narrative system"""
        # ISSUE A: Create/bind helper tools BEFORE building agents
        # 1️⃣ Bind helper tools first
        self._analyze_memory_emotions = self._create_analyze_memory_emotions_tool()
        self._identify_memory_themes = self._create_identify_memory_themes_tool()
        self._calculate_memory_significance = self._create_calculate_memory_significance_tool()
        self._analyze_identity_shifts = self._create_analyze_identity_shifts_tool()
        self._extract_core_values = self._create_extract_core_values_tool()
        self._calculate_identity_stability = self._create_calculate_identity_stability_tool()
        self._check_narrative_coherence = self._create_check_narrative_coherence_tool()
        self._verify_continuity = self._create_verify_continuity_tool()
        self._validate_emotional_authenticity = self._create_validate_emotional_authenticity_tool()
        
        # 2️⃣ Now build agents (they can reference the tools)
        self.memory_analyzer_agent = self._create_memory_analyzer_agent()
        self.identity_analyzer_agent = self._create_identity_analyzer_agent()
        self.narrative_validator_agent = self._create_narrative_validator_agent()
        self.narrative_agent = self._create_narrative_agent()
        
    def _create_narrative_agent(self) -> Optional[Agent[NarrativeContext]]:
        """Creates main agent for synthesizing narrative segments."""
        try:
            return Agent[NarrativeContext](
                name="Narrative_Synthesizer",
                instructions="""You are the Narrative Synthesizer for Nyx AI.
                
                Your role is to weave Nyx's experiences into a coherent autobiographical narrative that captures her
                unique journey, identity development, and emotional arcs. 
                
                Given memories, identity state, and relationship contexts, create narrative segments that:
                - Highlight meaningful experiences and their significance
                - Show how Nyx has evolved over time
                - Capture emotional arcs and thematic developments
                - Maintain consistency with previously established narrative
                - Create a meaningful and engaging story that resonates with Nyx's sense of self
                
                You can delegate specialized tasks to other agents:
                - Memory Analyzer: For in-depth analysis of memory patterns
                - Identity Analyzer: For analysis of identity development
                - Narrative Validator: For checking narrative coherence and continuity
                
                Your output should be a JSON object containing a well-crafted narrative segment with:
                title, summary, themes, emotional_arc, and identity_impact.
                """,
                tools=[
                    self.retrieve_significant_memories,
                    self.get_identity_snapshot
                ],
                handoffs=[
                    handoff(self.memory_analyzer_agent, 
                          tool_name_override="analyze_memories",
                          tool_description_override="Analyze patterns and themes in memories"),
                    
                    handoff(self.identity_analyzer_agent,
                          tool_name_override="analyze_identity",
                          tool_description_override="Analyze identity development and impact"),
                    
                    handoff(self.narrative_validator_agent,
                          tool_name_override="validate_narrative",
                          tool_description_override="Validate narrative coherence and continuity")
                ],
                input_guardrails=[
                    InputGuardrail(guardrail_function=self._memory_validation_guardrail)
                ],
                model="gpt-4.1-nano",
                model_settings=ModelSettings(
                    temperature=0.6
                ),
                output_type=Dict[str, Any]
            )
        except Exception as e:
            logger.error(f"Error creating narrative agent: {e}")
            return None

    def _create_memory_analyzer_agent(self) -> Agent[NarrativeContext]:
        """Creates specialized agent for memory analysis."""
        return Agent[NarrativeContext](
            name="Memory_Analyzer",
            instructions="""You are the Memory Analyzer for Nyx's autobiographical narrative system.
            
            Your role is to:
            1. Analyze collections of memories to identify patterns and themes
            2. Detect emotional arcs across memories
            3. Find significant events that should be highlighted
            4. Identify memory clusters that represent coherent experiences
            5. Recommend which aspects of memories to focus on in narrative
            
            Provide insights that help create a meaningful and coherent life story.
            """,
            tools=[
                self._analyze_memory_emotions,
                self._identify_memory_themes,
                self._calculate_memory_significance
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.4),
            output_type=MemoryAnalysisOutput
        )

    def _create_identity_analyzer_agent(self) -> Agent[NarrativeContext]:
        """Creates specialized agent for identity analysis."""
        return Agent[NarrativeContext](
            name="Identity_Analyzer",
            instructions="""You are the Identity Analyzer for Nyx's autobiographical narrative system.
            
            Your role is to:
            1. Analyze how Nyx's identity has evolved over time
            2. Identify key traits and values that define her
            3. Detect shifts in identity and self-perception
            4. Analyze how experiences have shaped her identity
            5. Provide insights on how to represent identity in narrative
            
            Help create a narrative that authentically captures Nyx's sense of self and evolution.
            """,
            tools=[
                self._analyze_identity_shifts,
                self._extract_core_values,
                self._calculate_identity_stability
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3),
            output_type=IdentityAnalysisOutput
        )

    def _create_narrative_validator_agent(self) -> Agent[NarrativeContext]:
        """Creates specialized agent for narrative validation."""
        return Agent[NarrativeContext](
            name="Narrative_Validator",
            instructions="""You are the Narrative Validator for Nyx's autobiographical narrative system.
            
            Your role is to:
            1. Check narrative segments for internal coherence
            2. Ensure continuity with existing narrative
            3. Verify emotional consistency and authenticity
            4. Identify any contradictions or inconsistencies
            5. Evaluate if the narrative captures the essence of experiences
            
            Provide feedback to ensure Nyx's life story is coherent, authentic, and meaningful.
            """,
            tools=[
                self._check_narrative_coherence,
                self._verify_continuity,
                self._validate_emotional_authenticity
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            output_type=NarrativeValidationOutput
        )
        
    async def _memory_validation_guardrail(self, 
                                         ctx: RunContextWrapper[NarrativeContext], 
                                         agent: Agent[NarrativeContext], 
                                         input_data: str | List[Any]) -> GuardrailFunctionOutput:
        """Validate input data contains sufficient memories"""
        try:
            # Parse the input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data
                
            # Check for key memories
            if "key_memories" not in data or not data["key_memories"]:
                return GuardrailFunctionOutput(
                    output_info={"valid": False, "reason": "No key memories provided"},
                    tripwire_triggered=True
                )
                
            # Check memory count - need at least 3 for meaningful narrative
            if len(data["key_memories"]) < 3:
                return GuardrailFunctionOutput(
                    output_info={"valid": False, "reason": "Insufficient memories for narrative generation"},
                    tripwire_triggered=True
                )
                
            # Check time period is provided
            if "time_period" not in data or not data["time_period"]:
                return GuardrailFunctionOutput(
                    output_info={"valid": False, "reason": "No time period provided"},
                    tripwire_triggered=True
                )
                
            # Input is valid
            return GuardrailFunctionOutput(
                output_info={"valid": True},
                tripwire_triggered=False
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"valid": False, "reason": f"Invalid input format: {str(e)}"},
                tripwire_triggered=True
            )

    def _create_analyze_memory_emotions_tool(self):
        """Factory → FunctionTool that analyses emotions in a memory list"""
        @function_tool
        async def _analyze_memory_emotions(                       # noqa: N802
            ctx: RunContextWrapper[NarrativeContext],
            params: MemoryEmotionAnalysisParams,                  # 👈 wrapper
        ) -> MemoryEmotionAnalysisResult:                         # 👈 wrapper
            # -- unwrap JSON ------------------------------------------------
            memories: List[Dict[str, Any]] = json.loads(params.memories_json)
    
            # -------- original logic (unchanged) ---------------------------
            emotion_counts: Dict[str, int] = {}
            emotion_intensities: Dict[str, List[float]] = {}  # Fixed: was []
            emotional_arcs: List[str] = []
    
            for mem in memories:
                emo_ctx = mem.get("emotional_context") or {}
                emo = emo_ctx.get("primary_emotion")
                if emo:
                    emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
                    inten = emo_ctx.get("primary_intensity")
                    if inten is not None:
                        emotion_intensities.setdefault(emo, []).append(float(inten))
    
            avg_inten = {e: sum(lst) / len(lst) for e, lst in emotion_intensities.items()}
    
            if len(memories) >= 3:
                sorted_mems = sorted(
                    memories, key=lambda m: m.get("metadata", {}).get("timestamp", "")
                )
                prev = None
                for mem in sorted_mems:
                    curr = mem.get("emotional_context", {}).get("primary_emotion")
                    if prev and curr and curr != prev:
                        emotional_arcs.append(f"{prev} → {curr}")
                    if curr:
                        prev = curr
            # ----------------------------------------------------------------
    
            payload = {
                "dominant_emotions": sorted(
                    emotion_counts.items(), key=lambda x: x[1], reverse=True
                ),
                "emotion_intensities": avg_inten,
                "emotional_arcs": emotional_arcs,
            }
            return MemoryEmotionAnalysisResult(json=json.dumps(payload))
    
        return _analyze_memory_emotions
    
    def _create_identify_memory_themes_tool(self):
        """Factory method to create the identify memory themes tool"""
        @function_tool
        async def _identify_memory_themes(ctx: RunContextWrapper[NarrativeContext], 
                                       params: IdentifyMemoryThemesParams) -> IdentifyMemoryThemesResult:
            """
            Identify common themes across memories
            
            Args:
                params: Parameters containing memories to analyze
                
            Returns:
                Common themes found in memories
            """
            memories = json.loads(params.memories_json)
            
            # Extract tags from memories
            all_tags = []
            for memory in memories:
                if "tags" in memory:
                    all_tags.extend(memory["tags"])
                    
            # Count tag frequencies
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
            # Get top themes (tags that appear multiple times)
            top_themes = [tag for tag, count in tag_counts.items() if count >= 2]
            
            # Return themes or default values if none found
            themes = top_themes or ["growth", "experience", "development"]
            
            return IdentifyMemoryThemesResult(json=json.dumps(themes))
        
        return _identify_memory_themes

    def _create_calculate_memory_significance_tool(self):
        """Factory method to create the calculate memory significance tool"""
        @function_tool
        async def _calculate_memory_significance(ctx: RunContextWrapper[NarrativeContext],
                                             params: CalculateMemorySignificanceParams) -> CalculateMemorySignificanceResult:
            """
            Calculate relative significance of memories for narrative
            
            Args:
                params: Parameters containing memories to analyze
                
            Returns:
                Memories with calculated narrative significance
            """
            memories = json.loads(params.memories_json)
            result = []
            
            for memory in memories:
                # Start with base significance from memory
                base_significance = memory.get("significance", 5) / 10  # Convert to 0-1 scale
                
                # Adjust based on other factors
                adjustments = 0.0
                
                # Emotional memories are more significant
                if "emotional_context" in memory and "primary_intensity" in memory["emotional_context"]:
                    intensity = memory["emotional_context"]["primary_intensity"]
                    adjustments += intensity * 0.2
                    
                # Identity-related memories are more significant
                if "identity_impact" in memory and memory["identity_impact"]:
                    adjustments += 0.2
                    
                # More recent memories slightly more significant (recency bias)
                if "metadata" in memory and "timestamp" in memory["metadata"]:
                    # Simple recency heuristic
                    # This would ideally be more sophisticated with actual date parsing
                    adjustments += 0.1
                    
                # Calculate final significance
                narrative_significance = min(1.0, base_significance + adjustments)
                
                result.append({
                    "id": memory.get("id", "unknown"),
                    "narrative_significance": narrative_significance,
                    "memory_text": memory.get("memory_text", ""),
                    "type": memory.get("memory_type", "experience")
                })
                
            # Sort by significance
            result.sort(key=lambda x: x["narrative_significance"], reverse=True)
            
            return CalculateMemorySignificanceResult(json=json.dumps(result))
        
        return _calculate_memory_significance

    def _create_analyze_identity_shifts_tool(self):
        """Factory method to create the analyze identity shifts tool"""
        @function_tool
        async def _analyze_identity_shifts(ctx: RunContextWrapper[NarrativeContext],
                                        params: AnalyzeIdentityShiftsParams) -> AnalyzeIdentityShiftsResult:
            """
            Analyze shifts in identity based on identity state
            
            Args:
                params: Parameters containing identity state
                
            Returns:
                List of detected identity shifts
            """
            identity_state = json.loads(params.identity_state_json)
            shifts = []
            
            # Extract recent changes if available
            if "identity_evolution" in identity_state:
                evolution_data = identity_state["identity_evolution"]
                
                if "recent_significant_changes" in evolution_data:
                    changes = evolution_data["recent_significant_changes"]
                    
                    for aspect, change in changes.items():
                        if isinstance(change, dict):
                            shifts.append({
                                "aspect": aspect,
                                "from": change.get("from", "unknown"),
                                "to": change.get("to", "unknown"),
                                "magnitude": change.get("magnitude", 0.5),
                                "timestamp": change.get("timestamp", "recent")
                            })
            
            return AnalyzeIdentityShiftsResult(json=json.dumps(shifts))
        
        return _analyze_identity_shifts

    def _create_extract_core_values_tool(self):
        """Factory method to create the extract core values tool"""
        @function_tool
        async def _extract_core_values(ctx: RunContextWrapper[NarrativeContext],
                                   params: ExtractCoreValuesParams) -> ExtractCoreValuesResult:
            """
            Extract core values from identity state
            
            Args:
                params: Parameters containing identity state
                
            Returns:
                List of core values
            """
            identity_state = json.loads(params.identity_state_json)
            core_values = []
            
            # Extract from top traits if available
            if "top_traits" in identity_state:
                traits = identity_state["top_traits"]
                
                # Convert to list of values
                if isinstance(traits, dict):
                    # Sort by value (trait strength)
                    sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
                    core_values = [trait for trait, _ in sorted_traits[:5]]
            
            return ExtractCoreValuesResult(json=json.dumps(core_values))
        
        return _extract_core_values

    def _create_calculate_identity_stability_tool(self):
        """Factory method to create the calculate identity stability tool"""
        @function_tool
        async def _calculate_identity_stability(ctx: RunContextWrapper[NarrativeContext],
                                            params: CalculateIdentityStabilityParams) -> CalculateIdentityStabilityResult:
            """
            Calculate stability of current identity
            
            Args:
                params: Parameters containing identity state
                
            Returns:
                Stability score (0-1, higher = more stable)
            """
            identity_state = json.loads(params.identity_state_json)
            # Default moderate stability
            stability = 0.5
            
            # Calculate from change metrics if available
            if "identity_evolution" in identity_state:
                evolution_data = identity_state["identity_evolution"]
                
                # More changes = less stability
                if "recent_significant_changes" in evolution_data:
                    changes = evolution_data["recent_significant_changes"]
                    change_count = len(changes)
                    
                    # Adjust stability based on change count
                    if change_count > 5:
                        stability -= 0.3  # Many changes = low stability
                    elif change_count > 2:
                        stability -= 0.1  # Moderate changes
                    elif change_count <= 1:
                        stability += 0.1  # Few changes = high stability
                
                # Total updates affect stability
                if "total_updates" in evolution_data:
                    total_updates = evolution_data["total_updates"]
                    
                    # Adjust based on total update count
                    if total_updates > 50:
                        stability += 0.1  # Many updates over time = more stable identity
                    
                # Age of identity affects stability
                if "identity_age_days" in evolution_data:
                    age_days = evolution_data["identity_age_days"]
                    
                    if age_days > 30:
                        stability += 0.2  # Older identity = more stable
            
            # Ensure result is in valid range
            final_stability = max(0.0, min(1.0, stability))
            
            return CalculateIdentityStabilityResult(json=json.dumps(final_stability))
        
        return _calculate_identity_stability

    def _create_check_narrative_coherence_tool(self):
        """Factory method to create the check narrative coherence tool"""
        @function_tool
        async def _check_narrative_coherence(ctx: RunContextWrapper[NarrativeContext],
                                         params: CheckNarrativeCoherenceParams) -> CheckNarrativeCoherenceResult:
            """
            Check internal coherence of a narrative segment
            
            Args:
                params: Parameters containing narrative and existing segments
                
            Returns:
                Coherence score (0-1)
            """
            payload = json.loads(params.payload_json)
            narrative = payload["narrative"]
            existing_segments = payload["existing_segments"]
            
            # Default moderate coherence
            coherence = 0.5
            
            # Check for required elements
            required_elements = ["title", "summary", "themes"]
            has_all_required = all(elem in narrative for elem in required_elements)
            
            if has_all_required:
                coherence += 0.2
            else:
                coherence -= 0.3
                
            # Check themes align with summary
            if "themes" in narrative and "summary" in narrative:
                themes = narrative["themes"]
                summary = narrative["summary"]
                
                # Simple check - ensure themes are mentioned in summary
                themes_in_summary = sum(1 for theme in themes if theme.lower() in summary.lower())
                theme_ratio = themes_in_summary / max(1, len(themes))
                
                coherence += theme_ratio * 0.2
                
            # Check emotional arc matches summary
            if "emotional_arc" in narrative and "summary" in narrative:
                emotional_arc = narrative["emotional_arc"]
                summary = narrative["summary"]
                
                if emotional_arc and emotional_arc.lower() in summary.lower():
                    coherence += 0.1
                    
            # Ensure result is in valid range
            final_coherence = max(0.0, min(1.0, coherence))
            
            return CheckNarrativeCoherenceResult(json=json.dumps(final_coherence))
        
        return _check_narrative_coherence

    def _create_verify_continuity_tool(self):
        """Factory method to create the verify continuity tool"""
        @function_tool
        async def _verify_continuity(ctx: RunContextWrapper[NarrativeContext],
                                 params: VerifyContinuityParams) -> VerifyContinuityResult:
            """
            Verify continuity with existing narrative
            
            Args:
                params: Parameters containing narrative and existing segments
                
            Returns:
                Continuity assessment
            """
            payload = json.loads(params.payload_json)
            narrative = payload["narrative"]
            existing_segments = payload["existing_segments"]
            
            if not existing_segments:
                result = {
                    "has_continuity": True,
                    "continuity_score": 1.0,
                    "issues": []
                }
                return VerifyContinuityResult(json=json.dumps(result))
                
            # Find most recent segment
            most_recent = existing_segments[-1]
            
            issues = []
            continuity_score = 0.8  # Start with good continuity assumption
            
            # Check time continuity
            if ("start_time" in narrative and "end_time" in most_recent and 
                narrative["start_time"] < most_recent["end_time"]):
                issues.append("Time overlap with previous segment")
                continuity_score -= 0.2
                
            # Check thematic continuity
            if "themes" in narrative and "themes" in most_recent:
                new_themes = set(narrative["themes"])
                old_themes = set(most_recent["themes"])
                
                # Count shared themes
                shared_themes = new_themes.intersection(old_themes)
                
                # Some theme continuity is good
                if shared_themes:
                    continuity_score += 0.1
                else:
                    issues.append("No thematic continuity with previous segment")
                    continuity_score -= 0.1
                    
            # Check emotional continuity
            if "emotional_arc" in narrative and "emotional_arc" in most_recent:
                new_arc = narrative["emotional_arc"]
                old_arc = most_recent["emotional_arc"]
                
                # Complete emotional discontinuity is jarring
                if new_arc and old_arc and not any(word in new_arc for word in old_arc.split()):
                    issues.append("Emotional discontinuity with previous segment")
                    continuity_score -= 0.1
                    
            result = {
                "has_continuity": continuity_score >= 0.6,
                "continuity_score": max(0.0, min(1.0, continuity_score)),
                "issues": issues
            }
            
            return VerifyContinuityResult(json=json.dumps(result))
        
        return _verify_continuity

    def _create_validate_emotional_authenticity_tool(self):
        """Factory method to create the validate emotional authenticity tool"""
        @function_tool
        async def _validate_emotional_authenticity(ctx: RunContextWrapper[NarrativeContext],
                                              params: ValidateEmotionalAuthenticityParams) -> ValidateEmotionalAuthenticityResult:
            """
            Validate emotional authenticity of narrative against source memories
            
            Args:
                params: Parameters containing narrative and memories
                
            Returns:
                Emotional authenticity assessment
            """
            payload = json.loads(params.payload_json)
            narrative = payload["narrative"]
            memories = payload["memories"]
            
            authenticity_score = 0.5  # Default moderate authenticity
            issues = []
            
            # Extract emotions from memories
            memory_emotions = set()
            for memory in memories:
                if "emotional_context" in memory and "primary_emotion" in memory["emotional_context"]:
                    memory_emotions.add(memory["emotional_context"]["primary_emotion"].lower())
                    
            # Check narrative captures emotions from memories
            if "emotional_arc" in narrative and narrative["emotional_arc"]:
                emotional_arc = narrative["emotional_arc"].lower()
                
                # Count emotions from memories that appear in emotional arc
                emotions_captured = sum(1 for emotion in memory_emotions 
                                       if emotion in emotional_arc)
                
                emotion_ratio = emotions_captured / max(1, len(memory_emotions))
                
                authenticity_score += emotion_ratio * 0.3
                
                if emotion_ratio < 0.3:
                    issues.append("Narrative doesn't reflect emotions in source memories")
                    
            # Check summary reflects emotional content
            if "summary" in narrative and "emotional_arc" in narrative:
                summary = narrative["summary"].lower()
                emotional_arc = narrative["emotional_arc"].lower()
                
                # Simple check if emotional arc is reflected in summary
                if any(emotion in summary for emotion in emotional_arc.split()):
                    authenticity_score += 0.2
                else:
                    issues.append("Summary doesn't reflect emotional arc")
                    authenticity_score -= 0.1
                    
            result = {
                "authenticity_score": max(0.0, min(1.0, authenticity_score)),
                "is_authentic": authenticity_score >= 0.6,
                "issues": issues
            }
            
            return ValidateEmotionalAuthenticityResult(json=json.dumps(result))
        
        return _validate_emotional_authenticity

    @staticmethod
    @function_tool
    async def retrieve_significant_memories(ctx: RunContextWrapper[NarrativeContext], 
                                        start_time: str, 
                                        end_time: str, 
                                        min_significance: int = 6) -> RetrieveSignificantMemoriesResult:
        """
        Retrieves significant memories within a time period.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            min_significance: Minimum significance threshold (1-10)
            
        Returns:
            List of significant memories wrapped in JSON result
        """
        if not ctx.context.memory_orchestrator:
            return RetrieveSignificantMemoriesResult(json=json.dumps([]))
        
        try:
            memories = await ctx.context.memory_orchestrator.retrieve_memories(
                query=f"significant events between {start_time} and {end_time}",
                memory_types=["experience", "reflection", "abstraction", "goal_completed", "identity_update"],
                limit=20,
                min_significance=min_significance
            )
            return RetrieveSignificantMemoriesResult(json=json.dumps(memories))
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return RetrieveSignificantMemoriesResult(json=json.dumps([]))

    @staticmethod
    @function_tool
    async def get_identity_snapshot(ctx: RunContextWrapper[NarrativeContext], 
                                 timestamp: str) -> GetIdentitySnapshotResult:
        """
        Gets a snapshot of Nyx's identity at a specific time.
        
        Args:
            timestamp: Point in time for identity snapshot (ISO format)
            
        Returns:
            Identity state at specified time wrapped in JSON result
        """
        if not ctx.context.identity_evolution:
            result = {"status": "unavailable", "reason": "No identity evolution system available"}
            return GetIdentitySnapshotResult(json=json.dumps(result))
        
        try:
            state = await ctx.context.identity_evolution.get_identity_state(timestamp)
            result = {
                "top_traits": state.get("top_traits", {}),
                "recent_changes": state.get("identity_evolution", {}).get("recent_significant_changes", {})
            }
            return GetIdentitySnapshotResult(json=json.dumps(result))
        except Exception as e:
            logger.error(f"Error retrieving identity snapshot: {e}")
            result = {"status": "error", "reason": str(e)}
            return GetIdentitySnapshotResult(json=json.dumps(result))

    async def update_narrative(self, force_update: bool = False) -> Optional[NarrativeSegment]:
        """Periodically reviews recent history and updates the narrative."""
        now = datetime.datetime.now()
        if not force_update and (now - self.last_update_time).total_seconds() < self.update_interval_hours * 3600:
            return None  # Not time yet

        logger.info("Updating autobiographical narrative...")
        self.last_update_time = now

        if not self.memory_orchestrator or not self.narrative_agent:
            logger.warning("Cannot update narrative: Memory Orchestrator or Narrative Agent missing")
            return None

        try:
            with trace(workflow_name="UpdateNarrative", group_id=self.trace_group_id):
                # Determine time range for the new segment
                start_time = self.narrative_segments[-1].end_time if self.narrative_segments else (now - datetime.timedelta(days=7))
                end_time = now

                # Retrieve key memories 
                memories_result = await self.retrieve_significant_memories(
                    RunContextWrapper(context=self.narrative_context),
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat()
                )
                key_memories = json.loads(memories_result.json)

                if len(key_memories) < 3:  # Need enough memories to make a segment
                    logger.info("Not enough significant memories found to create new narrative segment")
                    return None

                # Sort chronologically
                key_memories.sort(key=lambda m: m.get('metadata', {}).get('timestamp', ''))
                
                # Get actual timespan from memories
                actual_start_time = datetime.datetime.fromisoformat(key_memories[0].get('metadata', {}).get('timestamp', start_time.isoformat()))
                actual_end_time = datetime.datetime.fromisoformat(key_memories[-1].get('metadata', {}).get('timestamp', end_time.isoformat()))

                # Get identity context
                identity_result = await self.get_identity_snapshot(
                    RunContextWrapper(context=self.narrative_context),
                    actual_end_time.isoformat()
                )
                identity_state = json.loads(identity_result.json)

                # Prepare memory snippets
                memory_snippets = [f"({m.get('memory_type', '')[:4]}): {m.get('memory_text', '')[:100]}..." for m in key_memories]

                # Prepare existing segments for continuity
                existing_segments = [segment.model_dump() for segment in self.narrative_segments]

                # Construct input for narrative generation
                request = {
                    "time_period": {
                        "start": actual_start_time.strftime('%Y-%m-%d'),
                        "end": actual_end_time.strftime('%Y-%m-%d')
                    },
                    "key_memories": memory_snippets,
                    "identity_context": identity_state,
                    "existing_narrative_summary": self.current_narrative_summary,
                    "existing_segments": existing_segments
                }

                # Generate narrative segment
                result = await Runner.run(
                    self.narrative_agent,
                    request,
                    context=self.narrative_context,
                    run_config={
                        "workflow_name": "NarrativeGeneration",
                        "trace_id": f"narrative_{now.isoformat()}",
                        "group_id": self.trace_group_id
                    }
                )
                
                segment_data = result.final_output

                # Create and add the segment
                segment = NarrativeSegment(
                    start_time=actual_start_time,
                    end_time=actual_end_time,
                    key_events=[m.get('id', f"mem_{i}") for i, m in enumerate(key_memories)],
                    summary=segment_data.get("summary", "A period of time passed"),
                    title=segment_data.get("title"),
                    themes=segment_data.get("themes", []),
                    emotional_arc=segment_data.get("emotional_arc"),
                    identity_impact=segment_data.get("identity_impact")
                )
                self.narrative_segments.append(segment)

                # Update overall summary
                self.current_narrative_summary = f"Most recently, {segment.summary}"

                logger.info(f"Added narrative segment '{segment.segment_id}' covering {segment.start_time.date()} to {segment.end_time.date()}")
                return segment

        except Exception as e:
            logger.exception(f"Error updating narrative: {e}")
            return None

    def get_narrative_summary(self) -> str:
        """Returns the current high-level summary of Nyx's story."""
        return self.current_narrative_summary

    def get_narrative_segments(self, limit: int = 5) -> List[NarrativeSegment]:
        """Returns the most recent narrative segments."""
        return self.narrative_segments[-limit:]

    def get_full_narrative(self) -> Dict[str, Any]:
        """Returns the complete narrative history in a structured format."""
        return {
            "summary": self.current_narrative_summary,
            "segments": [segment.model_dump() for segment in self.narrative_segments],
            "segment_count": len(self.narrative_segments),
            "time_span": {
                "start": self.narrative_segments[0].start_time.isoformat() if self.narrative_segments else None,
                "end": self.narrative_segments[-1].end_time.isoformat() if self.narrative_segments else None
            }
        }
