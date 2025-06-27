# nyx/core/experience_consolidation.py

import logging
import asyncio
import random
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import os
from pydantic import BaseModel, Field

# Update imports to use OpenAI Agents SDK
from agents import (
    Agent, Runner, trace, function_tool, handoff, RunContextWrapper, 
    InputGuardrail, GuardrailFunctionOutput, RunConfig, RunHooks,
    ModelSettings, set_default_openai_key
)

from nyx.core.memory_core import MemoryCoreAgents, MemoryCreateParams

# Configure OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    # Replace with your actual key if not using environment variable
    set_default_openai_key("sk-your-key-here")

logger = logging.getLogger(__name__)

# Define schema models for structured outputs
class ConsolidationCandidate(BaseModel):
    """Schema for consolidation candidate group"""
    source_ids: List[str] = Field(..., description="Source experience IDs")
    similarity_score: float = Field(..., description="Average similarity score between experiences")
    scenario_type: str = Field(..., description="Primary scenario type")
    theme: str = Field(..., description="Common theme")
    user_ids: List[str] = Field(default_factory=list, description="Source user IDs")
    consolidation_type: str = Field(default="pattern", description="Type of consolidation")

class ConsolidationOutput(BaseModel):
    """Schema for consolidated experience output"""
    consolidation_text: str = Field(..., description="Text of the consolidated experience")
    source_count: int = Field(..., description="Number of source experiences")
    source_ids: List[str] = Field(..., description="Source experience IDs")
    scenario_type: str = Field(..., description="Primary scenario type")
    coherence_score: float = Field(..., description="Coherence score (0.0-1.0)")
    significance: float = Field(..., description="Significance score (0.0-10.0)")
    tags: List[str] = Field(..., description="Tags for the consolidated experience")

class ConsolidationEvaluation(BaseModel):
    """Schema for consolidation evaluation"""
    overall_quality: float = Field(..., description="Overall quality score (0.0-1.0)")
    coverage: float = Field(..., description="Coverage of source experiences (0.0-1.0)")
    coherence: float = Field(..., description="Coherence of consolidation (0.0-1.0)")
    information_gain: float = Field(..., description="Information gain from consolidation (0.0-1.0)")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")

class ConsolidationRequest(BaseModel):
    """Schema for consolidation request"""
    experience_ids: List[str] = Field(..., description="IDs of experiences to consider for consolidation")
    min_similarity: float = Field(default=0.7, description="Minimum similarity threshold")
    consolidation_type: Optional[str] = Field(default=None, description="Type of consolidation to perform")

class SimilarExperiencesParams(BaseModel):
    """Parameters for finding similar experiences"""
    experience_id: str = Field(..., description="ID of the experience to find similar ones for")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    max_similar: int = Field(default=5, description="Maximum number of similar experiences to return")

# Additional Pydantic models for function tools
class ExperienceDetails(BaseModel, extra="forbid"):
    """Details of an experience"""
    id: str
    content: str
    scenario_type: str = "general"
    timestamp: str = ""
    significance: float = 5
    tags: List[str] = Field(default_factory=list)
    emotional_context: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = "unknown"
    is_consolidated: bool = False
    error: Optional[str] = None

class SimilarityScore(BaseModel, extra="forbid"):
    """Result of similarity calculation"""
    score: float

class CommonTheme(BaseModel, extra="forbid"):
    """Common theme across experiences"""
    theme: str

class SortedGroups(BaseModel, extra="forbid"):
    """Sorted candidate groups"""
    groups: List[Dict[str, Any]]

class EmotionalContextSummary(BaseModel, extra="forbid"):
    """Common emotional context"""
    primary_emotion: Optional[str] = None
    primary_intensity: Optional[float] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None

class ConsolidationType(BaseModel, extra="forbid"):
    """Consolidation type determination"""
    type: str  # "pattern", "abstraction", "trend"

class SignificanceScore(BaseModel, extra="forbid"):
    """Significance score for consolidation"""
    score: float

class CoverageScore(BaseModel, extra="forbid"):
    """Coverage score for consolidation"""
    score: float

class CoherenceScore(BaseModel, extra="forbid"):
    """Coherence score for consolidation"""
    score: float

class InformationGain(BaseModel, extra="forbid"):
    """Information gain score"""
    score: float

class UpdateHistoryResult(BaseModel, extra="forbid"):
    """Result of updating consolidation history"""
    success: bool

class ConsolidationStatistics(BaseModel, extra="forbid"):
    """Consolidation statistics"""
    total_consolidations: int
    avg_quality: float
    type_distribution: Dict[str, int] = Field(default_factory=dict)
    ready_for_next: bool
    hours_until_next: float = 0.0

class FindSimilarResult(BaseModel, extra="forbid"):
    """Result of finding similar experiences"""
    experiences: List[Dict[str, Any]]

class RunCycleResult(BaseModel, extra="forbid"):
    """Result of running consolidation cycle"""
    status: str
    consolidations_created: int = 0
    source_memories_processed: int = 0
    reason: Optional[str] = None
    error: Optional[str] = None

class ConsolidationInsights(BaseModel, extra="forbid"):
    """Insights about consolidation activities"""
    total_consolidations: int
    last_consolidation: str
    consolidation_types: Dict[str, int] = Field(default_factory=dict)
    unique_users_consolidated: int
    user_coverage: List[str] = Field(default_factory=list)
    hours_until_next_consolidation: float
    ready_for_consolidation: bool

class ConsolidationContext:
    """Context object for sharing state between agents and tools"""
    def __init__(self):
        self.candidate_groups = []
        self.current_evaluation = None
        self.consolidation_history = []
        self.max_history_size = 100
        self.cycle_count = 0
        # Add reference to the parent system
        self.parent_system = None

class ConsolidationRunHooks(RunHooks):
    """Hooks for monitoring consolidation agent execution"""
    
    async def on_agent_start(self, context: RunContextWrapper, agent: Agent):
        """Called when an agent starts running"""
        logger.info(f"Starting agent: {agent.name}")
        
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any):
        """Called when an agent finishes running"""
        logger.info(f"Agent {agent.name} completed with output type: {type(output)}")
        
    async def on_handoff(self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent):
        """Called when a handoff occurs between agents"""
        logger.info(f"Handoff from {from_agent.name} to {to_agent.name}")
        
    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Any):
        """Called before a tool is invoked"""
        # Handle tools that might not have a name attribute
        tool_name = getattr(tool, 'name', None) or getattr(tool, '__name__', None) or str(tool)
        logger.info(f"Starting tool: {tool_name} from agent: {agent.name}")
        
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Any, result: str):
        """Called after a tool is invoked"""
        # Handle tools that might not have a name attribute
        tool_name = getattr(tool, 'name', None) or getattr(tool, '__name__', None) or str(tool)
        logger.info(f"Tool {tool_name} completed with result length: {len(str(result))}")

class ExperienceConsolidationSystem:
    """
    System for consolidating similar experiences into higher-level abstractions.
    Identifies patterns, trends, and recurring themes across experiences.
    """
    
    def __init__(self, memory_core: MemoryCoreAgents, experience_interface=None):
        """
        Initialize the experience consolidation system.
        
        Args:
            memory_core: Memory core for retrieving and storing experiences
            experience_interface: Experience interface for experience processing
        """
        self.memory_core = memory_core
        self.experience_interface = experience_interface
        
        # Create context object for sharing state between agents
        self.context = ConsolidationContext()
        # Set parent reference so tools can access the system
        self.context.parent_system = self
        
        # Create run hooks for monitoring
        self.run_hooks = ConsolidationRunHooks()
        
        # Initialize agents with properly defined handoffs
        self.candidate_finder_agent = self._create_candidate_finder_agent()
        self.consolidation_agent = self._create_consolidation_agent()
        self.evaluation_agent = self._create_evaluation_agent()
        self.orchestrator_agent = self._create_orchestrator_agent()
        
        # Configuration settings
        self.similarity_threshold = 0.7
        self.consolidation_interval = 24  # hours
        self.max_group_size = 5
        self.min_group_size = 2
        self.quality_threshold = 0.6
        
        # State tracking
        self.last_consolidation = datetime.now() - timedelta(hours=25)  # Start ready for consolidation
        self.consolidation_history = []
        self.max_history_size = 100
        
        # Trace ID for connecting traces
        self.trace_group_id = f"exp_consolidation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info("Experience Consolidation System initialized")
    
    def _create_orchestrator_agent(self) -> Agent:
        """Create the main orchestrator agent for coordination"""
        return Agent(
            name="Consolidation Orchestrator",
            instructions="""
            You are the Consolidation Orchestrator agent for Nyx's experience consolidation system.
            
            Your role is to coordinate the entire consolidation process:
            1. First, find candidate groups of similar experiences using the Consolidation Candidate Finder
            2. For each candidate group, create consolidated experiences using the Experience Consolidator
            3. Evaluate the quality of each consolidation using the Consolidation Evaluator
            4. Return a summary of the consolidation process
            
            Manage the overall process and ensure quality results.
            """,
            handoffs=[
                handoff(
                    self.candidate_finder_agent,
                    tool_name_override="find_consolidation_candidates",
                    tool_description_override="Find groups of similar experiences that can be consolidated"
                ),
                handoff(
                    self.consolidation_agent,
                    tool_name_override="create_consolidated_experience",
                    tool_description_override="Create a consolidated experience from similar experiences"
                ),
                handoff(
                    self.evaluation_agent,
                    tool_name_override="evaluate_consolidation",
                    tool_description_override="Evaluate the quality of a consolidated experience"
                )
            ],
            tools=[
                self._update_consolidation_history,
                self._get_consolidation_statistics
            ],
            model="gpt-4.1-nano",
            input_guardrails=[
                InputGuardrail(guardrail_function=self._consolidation_request_guardrail)
            ],
            output_type=dict
        )
    
    def _create_candidate_finder_agent(self) -> Agent:
        """Create the candidate finder agent"""
        return Agent(
            name="Consolidation Candidate Finder",
            handoff_description="Specialist agent for finding groups of similar experiences that can be consolidated",
            instructions="""
            You are the Consolidation Candidate Finder agent for Nyx's experience consolidation system.
            
            Your role is to:
            1. Analyze a set of experiences to identify groups of similar experiences
            2. Determine thematic connections and patterns across experiences
            3. Evaluate the similarity between experiences using vector embeddings
            4. Group experiences that could be meaningfully consolidated
            5. Prioritize groups based on similarity scores and coherence
            
            Focus on finding meaningful patterns and connections rather than superficial similarities.
            Consider scenario types, emotional context, and common themes when grouping experiences.
            """,
            tools=[
                self._get_experience_details,
                self._calculate_similarity_score,
                self._find_common_theme,
                self._sort_candidate_groups
            ],
            model="gpt-4.1-nano",
            output_type=List[ConsolidationCandidate]
        )
    
    def _create_consolidation_agent(self) -> Agent:
        """Create the consolidation agent"""
        return Agent(
            name="Experience Consolidator",
            handoff_description="Specialist agent for creating consolidated experiences from similar experiences",
            instructions="""
            You are the Experience Consolidator agent for Nyx's experience consolidation system.
            
            Your role is to:
            1. Create a consolidated experience from multiple similar experiences
            2. Extract common patterns, insights, and themes
            3. Generate meaningful abstractions that capture the essence of the experiences
            4. Ensure the consolidation adds value beyond the individual experiences
            5. Maintain coherence and emotional context in the consolidated experience
            
            Create consolidations that provide genuine insights and patterns, not just summaries.
            Focus on revealing underlying principles, patterns, or insights across experiences.
            """,
            tools=[
                self._get_experience_details,
                self._extract_common_emotional_context,
                self._generate_consolidation_type,
                self._calculate_significance_score
            ],
            model="gpt-4.1-nano",
            output_type=ConsolidationOutput
        )
    
    def _create_evaluation_agent(self) -> Agent:
        """Create the evaluation agent"""
        return Agent(
            name="Consolidation Evaluator",
            handoff_description="Specialist agent for evaluating the quality of consolidated experiences",
            instructions="""
            You are the Consolidation Evaluator agent for Nyx's experience consolidation system.
            
            Your role is to:
            1. Evaluate the quality of a consolidated experience
            2. Assess how well it covers the source experiences
            3. Measure the coherence and clarity of the consolidation
            4. Determine the information gain provided by the consolidation
            5. Provide recommendations for improving consolidation quality
            
            Be critical and thorough in your evaluation, focusing on whether the consolidation
            provides genuine value beyond the individual experiences.
            """,
            tools=[
                self._get_experience_details,
                self._calculate_coverage_score,
                self._calculate_coherence_score,
                self._calculate_information_gain
            ],
            model="gpt-4.1-nano",
            output_type=ConsolidationEvaluation
        )
    
    
    # Guardrail functions
    
    async def _consolidation_request_guardrail(self, ctx, agent, input_data):
        """Guardrail to validate consolidation requests"""
        # Check for valid experience IDs
        if isinstance(input_data, dict) and "experience_ids" in input_data:
            experience_ids = input_data["experience_ids"]
            if not experience_ids or len(experience_ids) < self.min_group_size:
                return GuardrailFunctionOutput(
                    output_info={"error": f"Need at least {self.min_group_size} experiences for consolidation"},
                    tripwire_triggered=True
                )
        
        # Check for valid similarity threshold
        if isinstance(input_data, dict) and "min_similarity" in input_data:
            min_similarity = input_data["min_similarity"]
            if min_similarity < 0.5:
                return GuardrailFunctionOutput(
                    output_info={"error": "Similarity threshold must be at least 0.5"},
                    tripwire_triggered=True
                )
        
        return GuardrailFunctionOutput(
            output_info={"valid": True},
            tripwire_triggered=False
        )

    # Tool functions with pydantic models for parameters
    @staticmethod  
    @function_tool
    async def _get_experience_details(ctx: RunContextWrapper, experience_id: str) -> ExperienceDetails:
        """
        Get details for a specific experience
        
        Args:
            experience_id: ID of the experience
            
        Returns:
            Experience details
        """
        # Get parent system from context
        parent_system = ctx.context.parent_system
        
        if not parent_system.memory_core:
            return ExperienceDetails(
                id=experience_id,
                content="",
                error="Memory core not available"
            )
        
        try:
            experience = await parent_system.memory_core.get_memory_by_id(experience_id)
            
            if not experience:
                return ExperienceDetails(
                    id=experience_id,
                    content="",
                    error=f"Experience {experience_id} not found"
                )
            
            # Extract key details
            return ExperienceDetails(
                id=experience_id,
                content=experience.get("memory_text", ""),
                scenario_type=experience.get("metadata", {}).get("scenario_type", "general"),
                timestamp=experience.get("timestamp", ""),
                significance=experience.get("significance", 5),
                tags=experience.get("tags", []),
                emotional_context=experience.get("metadata", {}).get("emotional_context", {}),
                user_id=experience.get("metadata", {}).get("user_id", "unknown"),
                is_consolidated=experience.get("metadata", {}).get("is_consolidation", False)
            )
            
        except Exception as e:
            logger.error(f"Error getting experience details: {e}")
            return ExperienceDetails(
                id=experience_id,
                content="",
                error=str(e)
            )

    @staticmethod  
    @function_tool
    async def _calculate_similarity_score(ctx: RunContextWrapper, 
                                     experience_id1: str, 
                                     experience_id2: str) -> SimilarityScore:
        """
        Calculate similarity score between two experiences
        
        Args:
            experience_id1: First experience ID
            experience_id2: Second experience ID
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Get parent system from context
        parent_system = ctx.context.parent_system
        
        if not parent_system.experience_interface:
            return SimilarityScore(score=0.0)
        
        try:
            # Get experience details
            exp1 = await parent_system._get_experience_details(ctx, experience_id1)
            exp2 = await parent_system._get_experience_details(ctx, experience_id2)
            
            if exp1.error or exp2.error:
                return SimilarityScore(score=0.0)
            
            # Use vector similarity if available
            if hasattr(parent_system.experience_interface, "_calculate_cosine_similarity"):
                # Generate vectors using experience interface
                vec1 = await parent_system.experience_interface._generate_experience_vector(ctx, parent_system.experience_interface, exp1.content)
                vec2 = await parent_system.experience_interface._generate_experience_vector(ctx, parent_system.experience_interface, exp2.content)
                
                # Calculate cosine similarity
                score = parent_system.experience_interface._calculate_cosine_similarity(vec1, vec2)
                return SimilarityScore(score=score)
            
            # Fallback similarity calculation
            similarity = 0.3
            if exp1.scenario_type == exp2.scenario_type:
                similarity += 0.2
            
            common_tags = set(exp1.tags) & set(exp2.tags)
            similarity += min(0.3, len(common_tags) * 0.05)
            
            if exp1.emotional_context and exp2.emotional_context:
                primary1 = exp1.emotional_context.get("primary_emotion", "")
                primary2 = exp2.emotional_context.get("primary_emotion", "")
                
                if primary1 and primary2 and primary1 == primary2:
                    similarity += 0.2
            
            return SimilarityScore(score=min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return SimilarityScore(score=0.0)

    @staticmethod  
    @function_tool
    async def _find_common_theme(ctx: RunContextWrapper, experience_ids: List[str]) -> CommonTheme:
        """
        Find common theme across multiple experiences
        
        Args:
            experience_ids: List of experience IDs
            
        Returns:
            Common theme description
        """
        # Get parent system from context
        parent_system = ctx.context.parent_system
        
        if not parent_system.memory_core:
            return CommonTheme(theme="Unknown theme")
        
        try:
            # Get all experiences
            experiences = []
            for exp_id in experience_ids:
                exp = await parent_system._get_experience_details(ctx, exp_id)
                if not exp.error:
                    experiences.append(exp)
            
            if not experiences:
                return CommonTheme(theme="Unknown theme")
            
            # Count scenario types
            scenario_counts = {}
            for exp in experiences:
                scenario = exp.scenario_type
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            
            # Find most common scenario
            common_scenario = max(scenario_counts.items(), key=lambda x: x[1])[0] if scenario_counts else "general"
            
            # Count emotional context
            emotion_counts = {}
            for exp in experiences:
                emotion = exp.emotional_context.get("primary_emotion", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
            
            # Count tags
            tag_counts = {}
            for exp in experiences:
                for tag in exp.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            common_tags = [tag for tag, count in tag_counts.items() 
                         if count >= len(experiences) / 2 and tag not in ["experience", "consolidated", common_scenario]]
            
            # Generate theme description
            if common_scenario != "general":
                if common_emotion != "neutral":
                    theme = f"{common_emotion} experiences in {common_scenario} scenarios"
                else:
                    theme = f"Experiences in {common_scenario} scenarios"
            else:
                if common_emotion != "neutral":
                    theme = f"{common_emotion} experiences"
                else:
                    theme = "General experiences"
            
            # Add common tags
            if common_tags:
                tags_str = ", ".join(common_tags[:2])
                theme += f" involving {tags_str}"
            
            return CommonTheme(theme=theme)
            
        except Exception as e:
            logger.error(f"Error finding common theme: {e}")
            return CommonTheme(theme="Unknown theme")

    @staticmethod
    @function_tool
    async def _sort_candidate_groups(
        ctx: RunContextWrapper,
        groups: Any                       # <— was List[Dict[str, Any]]
    ) -> SortedGroups:
        """
        Rank and sort candidate groups for consolidation.
    
        • Uses similarity, size, consolidation-type, and user diversity.
        • Returns a SortedGroups model, ready for the next agent step.
        """
        parent_system = ctx.context.parent_system
        # sensible default in case the attribute isn't present
        max_size = getattr(parent_system, "max_group_size", 10)
    
        def _score(group: Dict[str, Any]) -> float:
            sim  = group.get("similarity_score", 0.0)
            size = len(group.get("source_ids", []))
            size_factor = min(1.0, size / max_size)
    
            score = sim * 0.7 + size_factor * 0.3
    
            ctype = group.get("consolidation_type", "pattern")
            if ctype == "pattern":
                score *= 1.2
            elif ctype == "abstraction":
                score *= 1.1
    
            user_ids = group.get("user_ids", [])
            uniq = len(set(user_ids))
            if uniq > 1:
                score *= 1.0 + min(0.3, (uniq - 1) * 0.1)
    
            return score
    
        # Pair each group with its score, sort descending, strip scores.
        ranked = sorted(((g, _score(g)) for g in (groups or [])),
                        key=lambda t: t[1], reverse=True)
        sorted_groups = [g for g, _ in ranked]
    
        return SortedGroups(groups=sorted_groups)

    @staticmethod  
    @function_tool
    async def _extract_common_emotional_context(ctx: RunContextWrapper, 
                                          experience_ids: List[str]) -> EmotionalContextSummary:
        """
        Extract common emotional context from multiple experiences
        
        Args:
            experience_ids: List of experience IDs
            
        Returns:
            Common emotional context
        """
        # Get parent system from context
        parent_system = ctx.context.parent_system
        
        if not parent_system.memory_core:
            return EmotionalContextSummary()
        
        try:
            # Get all experiences
            emotional_contexts = []
            for exp_id in experience_ids:
                exp = await parent_system._get_experience_details(ctx, exp_id)
                if not exp.error and exp.emotional_context:
                    emotional_contexts.append(exp.emotional_context)
            
            if not emotional_contexts:
                return EmotionalContextSummary()
            
            # Calculate average emotional context
            common_context = EmotionalContextSummary()
            
            # Find most common primary emotion
            primary_emotions = {}
            for ec in emotional_contexts:
                emotion = ec.get("primary_emotion", "neutral")
                primary_emotions[emotion] = primary_emotions.get(emotion, 0) + 1
            
            if primary_emotions:
                common_primary = max(primary_emotions.items(), key=lambda x: x[1])[0]
                common_context.primary_emotion = common_primary
                
                # Calculate average intensity
                intensities = [ec.get("primary_intensity", 0.5) for ec in emotional_contexts 
                             if ec.get("primary_emotion", "") == common_primary]
                common_context.primary_intensity = sum(intensities) / len(intensities) if intensities else 0.5
            
            # Calculate average valence
            valences = [ec.get("valence", 0.0) for ec in emotional_contexts if "valence" in ec]
            if valences:
                common_context.valence = sum(valences) / len(valences)
            
            # Calculate average arousal
            arousals = [ec.get("arousal", 0.5) for ec in emotional_contexts if "arousal" in ec]
            if arousals:
                common_context.arousal = sum(arousals) / len(arousals)
            
            return common_context
            
        except Exception as e:
            logger.error(f"Error extracting emotional context: {e}")
            return EmotionalContextSummary()

    @staticmethod
    @function_tool
    async def _generate_consolidation_type(
        ctx: RunContextWrapper,
        experiences: Any                      # <— was List[ExperienceDetails]
    ) -> ConsolidationType:
        """
        Decide which consolidation strategy ("pattern", "abstraction", "trend")
        best fits the supplied experiences.
        """
        if not experiences:
            return ConsolidationType(type="pattern")
    
        # --- helper to safely unwrap attrs whether they're Pydantic objects
        #     or plain dicts -------------------------------------------------
        def _get(obj, key, default=None):
            return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)
    
        consolidated_cnt = sum(1 for e in experiences if _get(e, "is_consolidated", False))
    
        if consolidated_cnt >= len(experiences) / 2:
            return ConsolidationType(type="abstraction")
    
        scenario_set = {_get(e, "scenario_type", "general") for e in experiences}
        if len(scenario_set) > 1:
            return ConsolidationType(type="abstraction")
    
        # temporal spread check
        ts_list = []
        for e in experiences:
            ts = _get(e, "timestamp", "")
            if ts:
                try:
                    ts_list.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except ValueError:
                    continue
    
        if ts_list and (max(ts_list) - min(ts_list)) > timedelta(days=7):
            return ConsolidationType(type="trend")
    
        return ConsolidationType(type="pattern")

    @staticmethod
    @function_tool
    async def _calculate_significance_score(
        ctx: RunContextWrapper,
        experiences: Any                           # <— was List[ExperienceDetails]
    ) -> SignificanceScore:
        """
        Compute a 0-10 "significance" rating for a prospective consolidation.
        """
    
        # helper – works for either Pydantic objects or dicts
        def _get(obj, key, default=None):
            return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)
    
        if not experiences:
            return SignificanceScore(score=5.0)    # neutral fallback
    
        # ---------- core formula --------------------------------------------------
        base = sum(_get(e, "significance", 5.0) for e in experiences) / len(experiences)
    
        size_factor = min(1.5, 1.0 + (len(experiences) - 1) * 0.1)
    
        unique_users = len({ _get(e, "user_id", "unknown") for e in experiences })
        diversity_factor = 1.0 + min(0.5, (unique_users - 1) * 0.1)
    
        score = base * size_factor * diversity_factor
        # -------------------------------------------------------------------------
    
        return SignificanceScore(score=min(10.0, score))

    @staticmethod
    @function_tool
    async def _calculate_coverage_score(
        ctx: RunContextWrapper,
        consolidated: Any,                    # ExperienceDetails *or* dict
        source_experiences: List[Any]         # ditto
    ) -> CoverageScore:
        """
        Measure how thoroughly the consolidated memory represents its sources.
    
        Coverage for each source = |common_words| / |source_words|
        (scaled ×2, capped at 1).  Final score = mean of per-source coverages.
        """
        def _get(e, k, default=""):
            return getattr(e, k, e.get(k, default) if isinstance(e, dict) else default)
    
        if not source_experiences:
            return CoverageScore(score=0.0)
    
        consolidated_text = _get(consolidated, "content", "")
        if not consolidated_text:
            return CoverageScore(score=0.0)
    
        cons_words = set(consolidated_text.lower().split())
        scores: List[float] = []
    
        for src in source_experiences:
            src_text = _get(src, "content", "")
            if not src_text:
                continue
            src_words = set(src_text.lower().split())
            common    = src_words & cons_words
            coverage  = len(common) / max(1, len(src_words))
            scores.append(min(1.0, coverage * 2.0))   # scale but cap
    
        if not scores:
            return CoverageScore(score=0.0)
    
        return CoverageScore(score=sum(scores) / len(scores))

    @staticmethod
    @function_tool
    async def _calculate_coherence_score(
        ctx: RunContextWrapper,
        consolidated: Any                     # ExperienceDetails OR dict
    ) -> CoherenceScore:
        """
        Compute a robust 0-1 coherence score for a consolidated memory.
    
        ─────────────────────────────────────────────────────────────────────
        COMPONENTS & WEIGHTS
        ╭──────────────────────────────────────────────────────────────────╮
        │   0.40  ➜  Lexical / Semantic cohesion between adjacent          │
        │             sentences (spaCy vector cosine or Jaccard)           │
        │   0.40  ➜  Readability (Flesch Reading-Ease, 0-100 ↦ 0-1)        │
        │   0.20  ➜  Heuristic structure score (keyword presence)          │
        ╰──────────────────────────────────────────────────────────────────╯
        All heavy-weight libs are optional; graceful degradation ensures the
        function never crashes in production.
        """
    
        # -------- helpers ---------------------------------------------------------
        def _safe_get(obj, attr, default=""):
            return getattr(obj, attr, obj.get(attr, default) if isinstance(obj, dict) else default)
    
        def _keyword_structure_score(text: str) -> float:
            keys = ["pattern", "common", "across", "multiple", "experiences",
                    "trend", "insight", "connection"]
            return sum(k in text for k in keys) / len(keys)
    
        # Approx. Flesch Reading-Ease if textstat unavailable
        def _flesch_fallback(words: List[str], sentences: List[str]) -> float:
            # Very rough syllable estimate:   syllables ≈ vowels groups
            vowels = "aeiouy"
            syllables = 0
            for w in words:
                w = w.lower()
                v_groups = 0
                prev_is_vowel = False
                for ch in w:
                    is_vowel = ch in vowels
                    if is_vowel and not prev_is_vowel:
                        v_groups += 1
                    prev_is_vowel = is_vowel
                syllables += max(1, v_groups)
            W, S = len(words), max(1, len(sentences))
            # Flesch formula
            return 206.835 - 1.015 * (W / S) - 84.6 * (syllables / W)
    
        # -------- extract text ----------------------------------------------------
        text = _safe_get(consolidated, "content", "")
        if not text.strip():
            return CoherenceScore(score=0.0)
    
        low_text = text.lower()
    
        # -------- readability -----------------------------------------------------
        try:
            import textstat                  # optional dependency
            flesch = textstat.flesch_reading_ease(text)
        except Exception:
            # lightweight fallback
            tokens = text.split()
            sents  = [s for s in text.replace(";", ".").split(".") if s.strip()]
            flesch = _flesch_fallback(tokens, sents)
    
        # Map roughly 0-100 to 0-1 where ≥ 60 is "good"
        readability = min(100.0, max(0.0, flesch)) / 100.0
    
        # -------- cohesion (lexical / semantic) -----------------------------------
        cohesion_scores: List[float] = []
    
        # prefer spaCy semantic vectors if available
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_md")
            except Exception:
                nlp = spacy.load("en_core_web_sm")     # has no vectors – will fall through
            doc = nlp(text)
            has_vectors = doc[0].vector_norm != 0
            sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
            for sent_a, sent_b in zip(sentences, sentences[1:]):
                if has_vectors:
                    sim = sent_a.similarity(sent_b)
                else:
                    # fall back to Jaccard
                    a_words = {t.lemma_.lower() for t in sent_a
                               if not t.is_stop and t.is_alpha}
                    b_words = {t.lemma_.lower() for t in sent_b
                               if not t.is_stop and t.is_alpha}
                    if not a_words or not b_words:
                        continue
                    sim = len(a_words & b_words) / len(a_words | b_words)
                cohesion_scores.append(sim)
        except Exception:
            # Minimal tokenisation fallback
            raw_sents = [s.strip() for s in text.replace(";", ".").split(".") if s.strip()]
            for s1, s2 in zip(raw_sents, raw_sents[1:]):
                w1 = set(s1.lower().split())
                w2 = set(s2.lower().split())
                if not w1 or not w2:
                    continue
                cohesion_scores.append(len(w1 & w2) / len(w1 | w2))
    
        lexical_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0.0
        lexical_cohesion = min(1.0, max(0.0, lexical_cohesion))
    
        # -------- structure heuristic --------------------------------------------
        structure = _keyword_structure_score(low_text)
    
        # -------- weighted aggregate ---------------------------------------------
        coherence_score = (
            lexical_cohesion * 0.40 +
            readability     * 0.40 +
            structure       * 0.20
        )
    
        return CoherenceScore(score=round(coherence_score, 4))

    @staticmethod
    @function_tool
    async def _calculate_information_gain(
        ctx: RunContextWrapper,
        consolidated: Any,                    # ExperienceDetails or dict
        source_experiences: List[Any]
    ) -> InformationGain:
        """
        Estimate how much new / abstracted value the consolidation adds.
    
        Components (weights):
          • 0.40 insight markers          ("pattern", "trend", …)
          • 0.40 compression_score        (shorter than sources)
          • 0.20 abstraction_score        (generalising adverbs)
        """
        def _get(e, k, default=""):
            return getattr(e, k, e.get(k, default) if isinstance(e, dict) else default)
    
        if not source_experiences:
            return InformationGain(score=0.0)
    
        text = _get(consolidated, "content", "")
        if not text:
            return InformationGain(score=0.0)
    
        low = text.lower()
        insight_markers = any(m in low for m in
                              ["pattern", "trend", "insight", "connection", "common", "across"])
        insight_score   = 1.0 if insight_markers else 0.0
    
        total_source_len = sum(len(_get(s, "content", "")) for s in source_experiences)
        compression_ratio = 1.0 - (len(text) / max(1, total_source_len))
        compression_score = min(1.0, max(0.0, compression_ratio * 2.0))
    
        abstraction_words  = ["generally", "typically", "tend to", "often",
                              "usually", "pattern"]
        abstractions_found = sum(1 for w in abstraction_words if w in low)
        abstraction_score  = min(1.0, abstractions_found / 3.0)
    
        gain = (insight_score * 0.4 +
                compression_score * 0.4 +
                abstraction_score * 0.2)
    
        return InformationGain(score=gain)
    
    # New helper functions for orchestration

    @staticmethod
    @function_tool
    async def _update_consolidation_history(                     # noqa: N802
        ctx: RunContextWrapper,
        consolidated_id: str,
        source_ids: List[str],
        quality_score: float,
        consolidation_type: str
    ) -> UpdateHistoryResult:
        """
        Record a new consolidation entry.
    
        • De-duplicates identical entries (same consolidated_id).  
        • Maintains `parent_system.consolidation_history` no longer than
          `parent_system.max_history_size`.
        """
        ps = ctx.context.parent_system       # ↟ convenience alias
        if not ps:                             # should never happen
            return UpdateHistoryResult(success=False)
    
        # --- de-dup: if we already have this consolidated_id, update quality/type
        for entry in ps.consolidation_history:
            if entry["consolidated_id"] == consolidated_id:
                entry.update(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    quality_score=quality_score,
                    consolidation_type=consolidation_type,
                    source_ids=list(set(entry["source_ids"]) | set(source_ids)),
                    source_count=len(entry["source_ids"]),
                )
                break
        else:
            # new record
            ps.consolidation_history.append(
                {
                    "timestamp":       datetime.now(timezone.utc).isoformat(),
                    "consolidated_id": consolidated_id,
                    "source_ids":      source_ids,
                    "source_count":    len(source_ids),
                    "quality_score":   quality_score,
                    "consolidation_type": consolidation_type,
                }
            )
    
        # trim history ∵ bounded memory
        if len(ps.consolidation_history) > ps.max_history_size:
            ps.consolidation_history = ps.consolidation_history[-ps.max_history_size :]
    
        # keep convenience pointer to "last consolidation"
        ps.last_consolidation = datetime.now(timezone.utc)
    
        return UpdateHistoryResult(success=True)

    @staticmethod
    @function_tool
    async def _get_consolidation_statistics(                     # noqa: N802
        ctx: RunContextWrapper
    ) -> ConsolidationStatistics:
        """
        Return snapshot statistics with graceful handling of first-run state.
        """
        ps = ctx.context.parent_system
        if not ps or not getattr(ps, "consolidation_history", None):
            return ConsolidationStatistics(
                total_consolidations=0,
                avg_quality=0.0,
                type_distribution={},
                ready_for_next=True,
                hours_until_next=0.0,
            )
    
        hist = ps.consolidation_history
        total = len(hist)
        avg_quality = sum(e.get("quality_score", 0.0) for e in hist) / total if total else 0.0
    
        type_dist: Dict[str, int] = {}
        for e in hist:
            ctype = e.get("consolidation_type", "unknown")
            type_dist[ctype] = type_dist.get(ctype, 0) + 1
    
        # readiness for next consolidation
        last_ts = getattr(ps, "last_consolidation", datetime.now(timezone.utc))
        elapsed_hrs = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
        interval = getattr(ps, "consolidation_interval", 6)  # default 6 h
        ready = elapsed_hrs >= interval
    
        return ConsolidationStatistics(
            total_consolidations=total,
            avg_quality=round(avg_quality, 4),
            type_distribution=type_dist,
            ready_for_next=ready,
            hours_until_next=max(0.0, interval - elapsed_hrs),
        )

    # Helper – very fast lexical Jaccard as a fallback
    def _quick_jaccard(self, a: str, b: str) -> float:
        a_set, b_set = set(a.lower().split()), set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)
    
    @staticmethod
    @function_tool
    async def _find_similar_experiences(                        # noqa: N802
        ctx: RunContextWrapper,
        params: SimilarExperiencesParams
    ) -> FindSimilarResult:
        """
        Concurrent, vector-based (if available) *or* lexical fallback similarity.
        """
        ps = ctx.context.parent_system
        if not ps:
            return FindSimilarResult(experiences=[])
    
        mem_core   = ps.memory_core
        interface  = getattr(ps, "experience_interface", None)
        vectors_ok = (
            interface
            and hasattr(interface, "_generate_experience_vector")
            and hasattr(interface, "experience_vectors")
        )
    
        # ── fetch target experience ────────────────────────────────────────────
        target = await mem_core.get_memory_by_id(params.experience_id)
        if not target:
            return FindSimilarResult(experiences=[])
        target_text = target.get("memory_text", "")
        if not target_text:
            return FindSimilarResult(experiences=[])
    
        # ── obtain / compute vector ────────────────────────────────────────────
        if vectors_ok:
            if params.experience_id not in interface.experience_vectors:
                vec = await interface._generate_experience_vector(  # pylint: disable=protected-access
                    ctx, interface, target_text
                )
                interface.experience_vectors[params.experience_id] = {
                    "experience_id": params.experience_id,
                    "vector": vec,
                    "metadata": {
                        "user_id": target.get("metadata", {}).get("user_id", "unknown"),
                        "timestamp": target.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    },
                }
            target_vec = interface.experience_vectors[params.experience_id]["vector"]
        else:
            target_vec = None  # lexical fallback
    
        # ── gather candidate ids quickly (reuse cached vectors if present) ─────
        candidate_ids = (
            list(interface.experience_vectors.keys()) if vectors_ok
            else await mem_core.get_all_memory_ids()
        )
        candidate_ids = [cid for cid in candidate_ids if cid != params.experience_id]
    
        # ── concurrency: fetch & score ─────────────────────────────────────────
        semaphore = asyncio.Semaphore(50)  # limit I/O concurrency
    
        async def _score(cid: str):
            async with semaphore:
                # fetch text (cached vectors may carry metadata already)
                if vectors_ok and cid in interface.experience_vectors:
                    cand_vec = interface.experience_vectors[cid]["vector"]
                    cand_mem = None
                else:
                    cand_mem = await mem_core.get_memory_by_id(cid)
                    if not cand_mem:
                        return None
                    cand_vec = None
                cand_text = (
                    cand_mem["memory_text"]
                    if cand_mem
                    else await mem_core.get_memory_by_id(cid)  # ensure we have text
                ).get("memory_text", "")
    
                if not cand_text:
                    return None
    
                # similarity
                if target_vec is not None and cand_vec is not None:
                    sim = interface._calculate_cosine_similarity(target_vec, cand_vec)  # pylint: disable=protected-access
                else:
                    sim = ps._quick_jaccard(target_text, cand_text)
    
                if sim < params.similarity_threshold:
                    return None
    
                # ensure we expose same dict shape as upstream code expected
                if not cand_mem:
                    cand_mem = await mem_core.get_memory_by_id(cid)
                cand_mem = dict(cand_mem)  # shallow copy
                cand_mem["similarity"] = sim
                return cand_mem
    
        tasks = [_score(cid) for cid in candidate_ids]
        scored = [res for res in await asyncio.gather(*tasks) if res]
    
        # ── sort & truncate ────────────────────────────────────────────────────
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return FindSimilarResult(experiences=scored[: params.max_similar])

    # Public methods with enhanced implementation
    
    async def find_consolidation_candidates(self, experience_ids: List[str]) -> List[ConsolidationCandidate]:
        """
        Find candidate groups of experiences for consolidation using the candidate finder agent
        
        Args:
            experience_ids: List of experience IDs to consider
            
        Returns:
            List of candidate groups
        """
        with trace(
            workflow_name="find_consolidation_candidates", 
            group_id=self.trace_group_id,
            trace_metadata={"experience_count": len(experience_ids)}
        ):
            try:
                # Use the candidate finder agent
                result = await Runner.run(
                    self.candidate_finder_agent,
                    {
                        "experience_ids": experience_ids,
                        "similarity_threshold": self.similarity_threshold,
                        "max_group_size": self.max_group_size,
                        "min_group_size": self.min_group_size
                    },
                    context=self.context,
                    hooks=self.run_hooks,
                    run_config=RunConfig(
                        workflow_name="ConsolidationCandidateFinder",
                        trace_metadata={"experience_count": len(experience_ids)}
                    )
                )
                
                # Parse and return candidates
                candidates = result.final_output
                    
                return candidates
                
            except Exception as e:
                logger.error(f"Error finding consolidation candidates: {e}")
                return []
                
    async def create_consolidated_experience(self, 
                                         candidate: ConsolidationCandidate) -> Optional[ConsolidationOutput]:
        """
        Create a consolidated experience from a candidate group using the consolidation agent
        
        Args:
            candidate: Consolidation candidate group
            
        Returns:
            Consolidated experience or None if creation fails
        """
        with trace(
            workflow_name="create_consolidated_experience", 
            group_id=self.trace_group_id,
            trace_metadata={
                "candidate_type": candidate.consolidation_type,
                "source_count": len(candidate.source_ids)
            }
        ):
            try:
                # Use the consolidation agent
                result = await Runner.run(
                    self.consolidation_agent,
                    {
                        "source_ids": candidate.source_ids,
                        "consolidation_type": candidate.consolidation_type,
                        "theme": candidate.theme,
                        "scenario_type": candidate.scenario_type,
                        "similarity_score": candidate.similarity_score
                    },
                    context=self.context,
                    hooks=self.run_hooks,
                    run_config=RunConfig(
                        workflow_name="ConsolidationCreation",
                        trace_metadata={
                            "consolidation_type": candidate.consolidation_type,
                            "source_count": len(candidate.source_ids)
                        }
                    )
                )
                
                # Parse the output
                consolidation_output = result.final_output
                
                # Store the consolidated experience in memory core
                if self.memory_core:
                    try:
                        # Create metadata
                        metadata = {
                            "is_consolidation": True,
                            "consolidation_type": candidate.consolidation_type,
                            "source_experience_ids": candidate.source_ids,
                            "source_count": len(candidate.source_ids),
                            "similarity_score": candidate.similarity_score,
                            "theme": candidate.theme,
                            "scenario_type": candidate.scenario_type,
                            "emotional_context": await self._extract_common_emotional_context(
                                RunContextWrapper(context=self.context),
                                candidate.source_ids
                            ),
                            "user_ids": candidate.user_ids,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Store in memory
                        memory_id = await self.memory_core.add_memory(
                            memory_text=consolidation_output.consolidation_text,
                            memory_type="consolidated",
                            memory_scope="game",
                            significance=consolidation_output.significance,
                            tags=consolidation_output.tags,
                            metadata=metadata
                        )
                        
                        # Update output with ID
                        consolidation_data = consolidation_output.model_dump()
                        consolidation_data["id"] = memory_id
                        
                        # Add to vector embeddings if experience interface available
                        if (self.experience_interface and 
                            hasattr(self.experience_interface, "_generate_experience_vector")):
                            vector = await self.experience_interface._generate_experience_vector(
                                RunContextWrapper(context=self.context),
                                self.experience_interface,
                                consolidation_output.consolidation_text
                            )
                            
                            # Store vector
                            self.experience_interface.experience_vectors[memory_id] = {
                                "experience_id": memory_id,
                                "vector": vector,
                                "metadata": {
                                    "is_consolidation": True,
                                    "source_ids": candidate.source_ids,
                                    "timestamp": datetime.now().isoformat()
                                }
                            }
                        
                        # Return with ID
                        return ConsolidationOutput(**consolidation_data)
                    
                    except Exception as e:
                        logger.error(f"Error storing consolidated experience: {e}")
                
                return consolidation_output
                
            except Exception as e:
                logger.error(f"Error creating consolidated experience: {e}")
                return None
    
    async def evaluate_consolidation(self, 
                                 consolidated_id: str, 
                                 source_ids: List[str]) -> Optional[ConsolidationEvaluation]:
        """
        Evaluate the quality of a consolidated experience using the evaluation agent
        
        Args:
            consolidated_id: ID of consolidated experience
            source_ids: IDs of source experiences
            
        Returns:
            Evaluation results or None if evaluation fails
        """
        with trace(
            workflow_name="evaluate_consolidation", 
            group_id=self.trace_group_id,
            trace_metadata={
                "consolidated_id": consolidated_id,
                "source_count": len(source_ids)
            }
        ):
            try:
                # Use the evaluation agent
                result = await Runner.run(
                    self.evaluation_agent,
                    {
                        "consolidated_id": consolidated_id,
                        "source_ids": source_ids
                    },
                    context=self.context,
                    hooks=self.run_hooks,
                    run_config=RunConfig(
                        workflow_name="ConsolidationEvaluation",
                        trace_metadata={
                            "consolidated_id": consolidated_id,
                            "source_count": len(source_ids)
                        }
                    )
                )
                
                # Parse the output
                evaluation_output = result.final_output
                
                return evaluation_output
                
            except Exception as e:
                logger.error(f"Error evaluating consolidation: {e}")
                return None
    
    async def run_consolidation_cycle(self, experience_ids: Optional[List[str]] = None) -> RunCycleResult:
        """
        Run a complete consolidation cycle using direct logic (not orchestrator).
        Modified to correctly store consolidated memories with hierarchical data.
        """
        # --- Time Check (Keep this) ---
        now = datetime.now()
        time_since_last = (now - self.last_consolidation).total_seconds() / 3600
        if time_since_last < self.consolidation_interval:
            logger.info(f"Skipping consolidation cycle: Only {time_since_last:.1f} hours passed ({self.consolidation_interval} required).")
            return RunCycleResult(
                status="skipped",
                reason="Interval not met"
            )

        logger.info("Starting experience consolidation cycle...")
        with trace(workflow_name="consolidation_cycle", group_id=self.trace_group_id):
            consolidations_created = 0
            total_memories_affected = 0

            try:
                # 1. Find candidate groups
                result = await Runner.run(
                    self.candidate_finder_agent,
                    {"experience_ids": experience_ids or [], 
                     "similarity_threshold": self.similarity_threshold,
                     "max_group_size": self.max_group_size,
                     "min_group_size": self.min_group_size},
                    context=self.context,
                    run_config=RunConfig(workflow_name="CandidateFinder")
                )
                raw = result.final_output or []
                candidate_groups = [ConsolidationCandidate(**c) for c in raw] if isinstance(raw, list) else raw

                if not candidate_groups:
                    logger.info("No candidate groups found.")
                    self.last_consolidation = now
                    return RunCycleResult(
                        status="completed",
                        consolidations_created=0,
                        source_memories_processed=0
                    )

                # 2. Loop over each group
                for cand in candidate_groups:
                    cluster = cand.source_ids
                    if len(cluster) < self.min_group_size:
                        continue

                    # 2a. Retrieve full memory details
                    try:
                        retrieved = await self.memory_core.retrieve_memories(
                            query=f"ids:{','.join(cluster)}",
                            limit=len(cluster),
                            retrieval_level='detail'
                        )
                        details = {m['id']: m for m in retrieved}
                        source_details = [details[i] for i in cluster if i in details]
                        if len(source_details) < self.min_group_size:
                            logger.warning(f"Incomplete details for {cluster}, skipping.")
                            continue
                    except Exception as e:
                        logger.error(f"Retrieval error for {cluster}: {e}")
                        continue

                    # 2b. Generate consolidated text
                    try:
                        res = await Runner.run(
                            self.consolidation_agent,
                            {"source_ids": cluster,
                             "consolidation_type": cand.consolidation_type,
                             "theme": cand.theme,
                             "scenario_type": cand.scenario_type,
                             "similarity_score": cand.similarity_score},
                            context=self.context,
                            run_config=RunConfig(workflow_name="Consolidator")
                        )
                        out: ConsolidationOutput = res.final_output
                        text = out.consolidation_text
                        if not text:
                            raise ValueError("Empty consolidation text")
                    except Exception as e:
                        logger.error(f"Consolidator failed for {cluster}: {e}")
                        continue

                    # 2c. Compute metadata
                    significance = out.significance
                    avg_fidelity = sum(m.get('metadata', {}).get('fidelity', 1.0) for m in source_details) / len(source_details)
                    fidelity = max(0.1, avg_fidelity * 0.9)
                    level = 'abstraction' if any(w in text.lower() for w in ('pattern','abstract')) else 'summary'
                    tags = out.tags.copy()
                    for t in (level, 'consolidated_experience'):
                        if t not in tags:
                            tags.append(t)
                    scopes = {m.get('memory_scope','game') for m in source_details}
                    scope = 'user' if scopes=={'user'} else 'game'
                    summary_desc = f"{level.capitalize()} of {len(cluster)} experiences on '{cand.theme}'"

                    # 2d. Store the consolidated memory
                    try:
                        params = MemoryCreateParams(
                            memory_text=text,
                            memory_type="consolidated_experience",
                            memory_level=level,
                            source_memory_ids=cluster,
                            fidelity=fidelity,
                            summary_of=summary_desc,
                            memory_scope=scope,
                            significance=int(significance),
                            tags=tags,
                            metadata={}  # or pull any emotional context here
                        )
                        new_id = await self.memory_core.add_memory(**params.model_dump())
                        if new_id:
                            consolidations_created += 1
                            total_memories_affected += len(cluster)
                            logger.info(f"Stored consolidated memory {new_id} (level={level})")

                            # 2e. Mark each source
                            for sid in cluster:
                                meta = details[sid].get('metadata', {})
                                meta.update({
                                    "consolidated_into": new_id,
                                    "consolidation_date": datetime.now().isoformat()
                                })
                                await self.memory_core.update_memory(
                                    memory_id=sid,
                                    updates={"is_consolidated": True, "metadata": meta}
                                )
                        else:
                            logger.error(f"Failed to store consolidation for {cluster}")
                    except Exception as e:
                        logger.error(f"Error storing consolidation for {cluster}: {e}", exc_info=True)

                # 3. Wrap up
                self.last_consolidation = now
                logger.info(f"Cycle done: created={consolidations_created}, affected={total_memories_affected}")
                return RunCycleResult(
                    status="completed",
                    consolidations_created=consolidations_created,
                    source_memories_processed=total_memories_affected
                )

            except Exception as e:
                logger.error(f"Unexpected error in consolidation cycle: {e}", exc_info=True)
                return RunCycleResult(
                    status="error",
                    error=str(e)
                )

    
    async def get_consolidation_insights(self) -> ConsolidationInsights:
        """
        Get insights about consolidation activities
        
        Returns:
            Consolidation insights
        """
        with trace(workflow_name="get_consolidation_insights", group_id=self.trace_group_id):
            # Use the _get_consolidation_statistics tool
            stats = await self._get_consolidation_statistics(RunContextWrapper(context=self.context))
            
            # Add additional insights
            insights = ConsolidationInsights(
                total_consolidations=stats.total_consolidations,
                last_consolidation=self.last_consolidation.isoformat(),
                consolidation_types=stats.type_distribution,
                unique_users_consolidated=0,
                user_coverage=[],
                hours_until_next_consolidation=stats.hours_until_next,
                ready_for_consolidation=stats.ready_for_next
            )
            
            # Count consolidation types and users
            user_coverage_set = set()
            for entry in self.consolidation_history:
                # Track unique users
                user_ids = entry.get("user_ids", [])
                for user_id in user_ids:
                    user_coverage_set.add(user_id)
            
            # Convert set to count
            insights.unique_users_consolidated = len(user_coverage_set)
            insights.user_coverage = list(user_coverage_set)
            
            return insights
