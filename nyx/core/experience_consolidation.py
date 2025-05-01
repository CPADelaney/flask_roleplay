# nyx/core/experience_consolidation.py

import logging
import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import os

# Update imports to use OpenAI Agents SDK
from agents import (
    Agent, Runner, trace, function_tool, handoff, RunContextWrapper, 
    InputGuardrail, GuardrailFunctionOutput, RunConfig, RunHooks,
    ModelSettings, set_default_openai_key
)
from pydantic import BaseModel, Field

from nyx.core.memory_core import MemoryCoreAgents

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

class ConsolidationContext:
    """Context object for sharing state between agents and tools"""
    def __init__(self):
        self.candidate_groups = []
        self.current_evaluation = None
        self.consolidation_history = []
        self.max_history_size = 100
        self.cycle_count = 0

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
        logger.info(f"Starting tool: {tool.name} from agent: {agent.name}")
        
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Any, result: str):
        """Called after a tool is invoked"""
        logger.info(f"Tool {tool.name} completed with result length: {len(str(result))}")

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
                function_tool(self._update_consolidation_history),
                function_tool(self._get_consolidation_statistics)
            ],
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
                function_tool(self._get_experience_details),
                function_tool(self._calculate_similarity_score),
                function_tool(self._find_common_theme),
                function_tool(self._sort_candidate_groups)
            ],
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
                function_tool(self._get_experience_details),
                function_tool(self._extract_common_emotional_context),
                function_tool(self._generate_consolidation_type),
                function_tool(self._calculate_significance_score)
            ],
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
                function_tool(self._get_experience_details),
                function_tool(self._calculate_coverage_score),
                function_tool(self._calculate_coherence_score),
                function_tool(self._calculate_information_gain)
            ],
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
    async def _get_experience_details(ctx: RunContextWrapper, experience_id: str) -> Dict[str, Any]:
        """
        Get details for a specific experience
        
        Args:
            experience_id: ID of the experience
            
        Returns:
            Experience details
        """
        if not self.memory_core:
            return {"error": "Memory core not available"}
        
        try:
            experience = await self.memory_core.get_memory_by_id(experience_id)
            
            if not experience:
                return {"error": f"Experience {experience_id} not found"}
            
            # Extract key details
            return {
                "id": experience_id,
                "content": experience.get("memory_text", ""),
                "scenario_type": experience.get("metadata", {}).get("scenario_type", "general"),
                "timestamp": experience.get("timestamp", ""),
                "significance": experience.get("significance", 5),
                "tags": experience.get("tags", []),
                "emotional_context": experience.get("metadata", {}).get("emotional_context", {}),
                "user_id": experience.get("metadata", {}).get("user_id", "unknown"),
                "is_consolidated": experience.get("metadata", {}).get("is_consolidation", False)
            }
            
        except Exception as e:
            logger.error(f"Error getting experience details: {e}")
            return {"error": str(e)}

    @staticmethod  
    @function_tool
    async def _calculate_similarity_score(ctx: RunContextWrapper, 
                                     experience_id1: str, 
                                     experience_id2: str) -> float:
        """
        Calculate similarity score between two experiences
        
        Args:
            experience_id1: First experience ID
            experience_id2: Second experience ID
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if not self.experience_interface:
            return 0.0
        
        try:
            # Get experience details
            exp1 = await self._get_experience_details(ctx, experience_id1)
            exp2 = await self._get_experience_details(ctx, experience_id2)
            
            if "error" in exp1 or "error" in exp2:
                return 0.0
            
            # Use vector similarity if available
            if hasattr(self.experience_interface, "_calculate_cosine_similarity"):
                # Generate vectors using experience interface
                vec1 = await self.experience_interface._generate_experience_vector(ctx, exp1["content"])
                vec2 = await self.experience_interface._generate_experience_vector(ctx, exp2["content"])
                
                # Calculate cosine similarity
                return self.experience_interface._calculate_cosine_similarity(vec1, vec2)
            
            # Same logic as before for fallback
            similarity = 0.3
            if exp1["scenario_type"] == exp2["scenario_type"]:
                similarity += 0.2
            
            common_tags = set(exp1["tags"]) & set(exp2["tags"])
            similarity += min(0.3, len(common_tags) * 0.05)
            
            if exp1["emotional_context"] and exp2["emotional_context"]:
                primary1 = exp1["emotional_context"].get("primary_emotion", "")
                primary2 = exp2["emotional_context"].get("primary_emotion", "")
                
                if primary1 and primary2 and primary1 == primary2:
                    similarity += 0.2
            
            return min(1.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    @staticmethod  
    @function_tool
    async def _find_common_theme(ctx: RunContextWrapper, experience_ids: List[str]) -> str:
        """
        Find common theme across multiple experiences
        
        Args:
            experience_ids: List of experience IDs
            
        Returns:
            Common theme description
        """
        # Same implementation as before
        if not self.memory_core:
            return "Unknown theme"
        
        try:
            # Get all experiences
            experiences = []
            for exp_id in experience_ids:
                exp = await self._get_experience_details(ctx, exp_id)
                if not "error" in exp:
                    experiences.append(exp)
            
            if not experiences:
                return "Unknown theme"
            
            # Count scenario types
            scenario_counts = {}
            for exp in experiences:
                scenario = exp["scenario_type"]
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            
            # Rest of the implementation remains the same
            common_scenario = max(scenario_counts.items(), key=lambda x: x[1])[0] if scenario_counts else "general"
            
            # Count emotional context
            emotion_counts = {}
            for exp in experiences:
                emotion = exp.get("emotional_context", {}).get("primary_emotion", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
            
            # Count tags
            tag_counts = {}
            for exp in experiences:
                for tag in exp["tags"]:
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
            
            return theme
            
        except Exception as e:
            logger.error(f"Error finding common theme: {e}")
            return "Unknown theme"

    @staticmethod  
    @function_tool
    async def _sort_candidate_groups(ctx: RunContextWrapper, 
                                groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort candidate groups by quality for consolidation
        
        Args:
            groups: List of candidate groups
            
        Returns:
            Sorted groups
        """
        # Define scoring function
        def score_group(group):
            similarity = group.get("similarity_score", 0)
            size = len(group.get("source_ids", []))
            size_factor = min(1.0, size / self.max_group_size)
            
            # Basic score is similarity * size factor
            score = similarity * 0.7 + size_factor * 0.3
            
            # Adjust score based on consolidation type
            consolidation_type = group.get("consolidation_type", "pattern")
            if consolidation_type == "pattern":
                score *= 1.2  # Favor pattern consolidations
            elif consolidation_type == "abstraction":
                score *= 1.1  # Slightly favor abstractions
            
            # Adjust score based on user diversity
            user_ids = group.get("user_ids", [])
            unique_users = len(set(user_ids))
            if unique_users > 1:
                # Favor groups with experiences from multiple users
                score *= 1.0 + min(0.3, (unique_users - 1) * 0.1)
            
            return score
        
        # Sort groups by score
        scored_groups = [(group, score_group(group)) for group in groups]
        sorted_groups = [group for group, _ in sorted(scored_groups, key=lambda x: x[1], reverse=True)]
        
        return sorted_groups

    @staticmethod  
    @function_tool
    async def _extract_common_emotional_context(ctx: RunContextWrapper, 
                                          experience_ids: List[str]) -> Dict[str, Any]:
        """
        Extract common emotional context from multiple experiences
        
        Args:
            experience_ids: List of experience IDs
            
        Returns:
            Common emotional context
        """
        # Same implementation as before
        if not self.memory_core:
            return {}
        
        try:
            # Get all experiences
            emotional_contexts = []
            for exp_id in experience_ids:
                exp = await self._get_experience_details(ctx, exp_id)
                if not "error" in exp and "emotional_context" in exp:
                    emotional_contexts.append(exp["emotional_context"])
            
            if not emotional_contexts:
                return {}
            
            # Calculate average emotional context
            common_context = {}
            
            # Rest of implementation remains the same
            # Find most common primary emotion
            primary_emotions = {}
            for ec in emotional_contexts:
                emotion = ec.get("primary_emotion", "neutral")
                primary_emotions[emotion] = primary_emotions.get(emotion, 0) + 1
            
            if primary_emotions:
                common_primary = max(primary_emotions.items(), key=lambda x: x[1])[0]
                common_context["primary_emotion"] = common_primary
                
                # Calculate average intensity
                intensities = [ec.get("primary_intensity", 0.5) for ec in emotional_contexts 
                             if ec.get("primary_emotion", "") == common_primary]
                common_context["primary_intensity"] = sum(intensities) / len(intensities) if intensities else 0.5
            
            # Calculate average valence
            valences = [ec.get("valence", 0.0) for ec in emotional_contexts if "valence" in ec]
            if valences:
                common_context["valence"] = sum(valences) / len(valences)
            
            # Calculate average arousal
            arousals = [ec.get("arousal", 0.5) for ec in emotional_contexts if "arousal" in ec]
            if arousals:
                common_context["arousal"] = sum(arousals) / len(arousals)
            
            return common_context
            
        except Exception as e:
            logger.error(f"Error extracting emotional context: {e}")
            return {}

    @staticmethod  
    @function_tool
    async def _generate_consolidation_type(ctx: RunContextWrapper, experiences: List[Dict[str, Any]]) -> str:
        """
        Determine appropriate consolidation type based on experiences
        
        Args:
            experiences: List of experiences
            
        Returns:
            Consolidation type
        """
        # Same implementation as before
        if not experiences:
            return "pattern"
        
        # Count consolidated experiences
        consolidated_count = sum(1 for e in experiences if e.get("is_consolidated", False))
        
        # If many experiences are already consolidations, use abstraction
        if consolidated_count >= len(experiences) / 2:
            return "abstraction"
        
        # Check for scenario diversity
        scenarios = [e.get("scenario_type", "general") for e in experiences]
        unique_scenarios = len(set(scenarios))
        
        # If multiple scenarios, use abstraction
        if unique_scenarios > 1:
            return "abstraction"
        
        # Check for temporal relationships
        timestamps = []
        for e in experiences:
            if "timestamp" in e:
                try:
                    ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
                    timestamps.append(ts)
                except:
                    pass
        
        # If experiences span a significant time period, use trend
        if timestamps and max(timestamps) - min(timestamps) > timedelta(days=7):
            return "trend"
        
        # Default to pattern
        return "pattern"

    @staticmethod  
    @function_tool
    async def _calculate_significance_score(ctx: RunContextWrapper, 
                                      experiences: List[Dict[str, Any]]) -> float:
        """
        Calculate significance score for consolidated experience
        
        Args:
            experiences: List of experiences
            
        Returns:
            Significance score (0.0-10.0)
        """
        # Same implementation as before
        if not experiences:
            return 5.0
        
        # Base significance is average of source significances
        base_significance = sum(e.get("significance", 5.0) for e in experiences) / len(experiences)
        
        # Adjust based on number of experiences
        size_factor = min(1.5, 1.0 + (len(experiences) - 1) * 0.1)
        
        # Adjust based on diversity
        user_ids = [e.get("user_id", "unknown") for e in experiences]
        unique_users = len(set(user_ids))
        diversity_factor = 1.0 + min(0.5, (unique_users - 1) * 0.1)
        
        # Calculate final significance
        significance = base_significance * size_factor * diversity_factor
        
        # Cap at 10.0
        return min(10.0, significance)

    @staticmethod  
    @function_tool
    async def _calculate_coverage_score(ctx: RunContextWrapper,
                                  consolidated: Dict[str, Any],
                                  source_experiences: List[Dict[str, Any]]) -> float:
        """
        Calculate how well the consolidation covers the source experiences
        
        Args:
            consolidated: Consolidated experience
            source_experiences: Source experiences
            
        Returns:
            Coverage score (0.0-1.0)
        """
        # Same implementation as before
        if not source_experiences:
            return 0.0
        
        consolidated_text = consolidated.get("content", "")
        if not consolidated_text:
            return 0.0
        
        # For each source experience, check if key elements are represented in the consolidation
        coverage_scores = []
        
        for source in source_experiences:
            source_text = source.get("content", "")
            if not source_text:
                continue
            
            # Extract key elements from source (simple approach: key nouns and verbs)
            source_words = set(source_text.lower().split())
            
            # Compare with consolidated text
            consolidated_words = set(consolidated_text.lower().split())
            common_words = source_words.intersection(consolidated_words)
            
            # Calculate coverage for this source
            source_coverage = len(common_words) / max(1, len(source_words))
            coverage_scores.append(min(1.0, source_coverage * 2))  # Scale up but cap at 1.0
        
        if not coverage_scores:
            return 0.0
        
        # Overall coverage is average of individual coverages
        return sum(coverage_scores) / len(coverage_scores)

    @staticmethod  
    @function_tool
    async def _calculate_coherence_score(ctx: RunContextWrapper, consolidated: Dict[str, Any]) -> float:
        """
        Calculate coherence of the consolidated experience
        
        Args:
            consolidated: Consolidated experience
            
        Returns:
            Coherence score (0.0-1.0)
        """
        # Same implementation as before
        consolidated_text = consolidated.get("content", "")
        if not consolidated_text:
            return 0.0
        
        # Simple measure based on text structure and clarity
        # In a full implementation, this would use more sophisticated NLP
        
        # Check for structured explanation
        has_pattern = "pattern" in consolidated_text.lower()
        has_common = "common" in consolidated_text.lower()
        has_across = "across" in consolidated_text.lower()
        has_multiple = "multiple" in consolidated_text.lower()
        has_experiences = "experiences" in consolidated_text.lower()
        
        structure_score = sum([has_pattern, has_common, has_across, has_multiple, has_experiences]) / 5
        
        # Check for clear articulation
        sentences = consolidated_text.split(".")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        length_score = min(1.0, 20 / max(1, avg_sentence_length))  # Prefer concise sentences
        
        # Combined score with weights
        coherence = structure_score * 0.7 + length_score * 0.3
        
        return coherence

    @staticmethod  
    @function_tool
    async def _calculate_information_gain(ctx: RunContextWrapper,
                                    consolidated: Dict[str, Any],
                                    source_experiences: List[Dict[str, Any]]) -> float:
        """
        Calculate information gain from consolidation
        
        Args:
            consolidated: Consolidated experience
            source_experiences: Source experiences
            
        Returns:
            Information gain score (0.0-1.0)
        """
        # Same implementation as before
        if not source_experiences:
            return 0.0
        
        consolidated_text = consolidated.get("content", "")
        if not consolidated_text:
            return 0.0
        
        # Check for higher-level insights
        has_insight_markers = any(marker in consolidated_text.lower() for marker in 
                              ["pattern", "trend", "insight", "connection", "common", "across"])
        
        # Check if consolidation is shorter than sum of sources
        source_length = sum(len(s.get("content", "")) for s in source_experiences)
        consolidated_length = len(consolidated_text)
        compression_ratio = 1.0 - (consolidated_length / max(1, source_length))
        compression_score = min(1.0, max(0.0, compression_ratio * 2))  # Scale up but cap at 1.0
        
        # Check if consolidation uses abstraction terms
        abstraction_markers = ["generally", "typically", "tend to", "often", "usually", "pattern"]
        has_abstractions = sum(1 for marker in abstraction_markers if marker in consolidated_text.lower())
        abstraction_score = min(1.0, has_abstractions / 3)  # Scale but cap at 1.0
        
        # Combined score with weights
        information_gain = (
            (has_insight_markers * 0.4) +
            (compression_score * 0.4) +
            (abstraction_score * 0.2)
        )
        
        return information_gain
    
    # New helper functions for orchestration

    @staticmethod  
    @function_tool
    async def _update_consolidation_history(ctx: RunContextWrapper,
                                       consolidated_id: str,
                                       source_ids: List[str],
                                       quality_score: float,
                                       consolidation_type: str) -> bool:
        """
        Update consolidation history with new entry
        
        Args:
            consolidated_id: ID of the consolidated experience
            source_ids: IDs of source experiences
            quality_score: Quality evaluation score
            consolidation_type: Type of consolidation performed
            
        Returns:
            Success status
        """
        # Add to history
        self.consolidation_history.append({
            "timestamp": datetime.now().isoformat(),
            "consolidated_id": consolidated_id,
            "source_ids": source_ids,
            "source_count": len(source_ids),
            "quality_score": quality_score,
            "consolidation_type": consolidation_type
        })
        
        # Limit history size
        if len(self.consolidation_history) > self.max_history_size:
            self.consolidation_history = self.consolidation_history[-self.max_history_size:]
        
        return True

    @staticmethod  
    @function_tool
    async def _get_consolidation_statistics(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get statistics about consolidation activities
        
        Returns:
            Consolidation statistics
        """
        if not self.consolidation_history:
            return {
                "total_consolidations": 0,
                "avg_quality": 0.0,
                "type_distribution": {},
                "ready_for_next": True
            }
        
        # Calculate statistics
        total = len(self.consolidation_history)
        avg_quality = sum(entry.get("quality_score", 0.0) for entry in self.consolidation_history) / total
        
        # Count consolidation types
        type_counts = {}
        for entry in self.consolidation_history:
            c_type = entry.get("consolidation_type", "unknown")
            type_counts[c_type] = type_counts.get(c_type, 0) + 1
        
        # Check if ready for next consolidation
        now = datetime.now()
        time_since_last = (now - self.last_consolidation).total_seconds() / 3600  # hours
        ready_for_next = time_since_last >= self.consolidation_interval
        
        return {
            "total_consolidations": total,
            "avg_quality": avg_quality,
            "type_distribution": type_counts,
            "ready_for_next": ready_for_next,
            "hours_until_next": max(0, self.consolidation_interval - time_since_last)
        }
    
    # New implementation of find_similar_experiences with Pydantic model

    @staticmethod  
    @function_tool
    async def _find_similar_experiences(ctx: RunContextWrapper,
                                   params: SimilarExperiencesParams) -> List[Dict[str, Any]]:
        """
        Find experiences similar to the given one
        
        Args:
            params: Parameters for finding similar experiences
            
        Returns:
            List of similar experiences
        """
        # Extract parameters
        experience_id = params.experience_id
        similarity_threshold = params.similarity_threshold
        max_similar = params.max_similar
        
        # Get the experience
        try:
            experience = await self.memory_core.get_memory_by_id(experience_id)
            
            if not experience:
                return []
            
            # Get the experience text
            experience_text = experience.get("memory_text", "")
            
            if not experience_text:
                return []
            
            # Get vector for this experience
            if hasattr(self.experience_interface, "_generate_experience_vector") and hasattr(self.experience_interface, "experience_vectors"):
                if experience_id in self.experience_interface.experience_vectors:
                    exp_vector = self.experience_interface.experience_vectors[experience_id].get("vector", [])
                else:
                    # Generate vector
                    exp_vector = await self.experience_interface._generate_experience_vector(ctx, experience_text)
                    
                    # Store for future use
                    self.experience_interface.experience_vectors[experience_id] = {
                        "experience_id": experience_id,
                        "vector": exp_vector,
                        "metadata": {
                            "user_id": experience.get("metadata", {}).get("user_id", "unknown"),
                            "timestamp": experience.get("timestamp", datetime.now().isoformat())
                        }
                    }
                
                # Find similar experiences
                similar_experiences = []
                
                for other_id, other_vector_data in self.experience_interface.experience_vectors.items():
                    if other_id != experience_id:
                        other_vector = other_vector_data.get("vector", [])
                        
                        # Calculate similarity
                        similarity = self.experience_interface._calculate_cosine_similarity(exp_vector, other_vector)
                        
                        if similarity >= similarity_threshold:
                            # Get the experience
                            other_exp = await self.memory_core.get_memory_by_id(other_id)
                            
                            if other_exp:
                                other_exp["similarity"] = similarity
                                similar_experiences.append(other_exp)
                
                # Sort by similarity
                similar_experiences.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                
                # Return top matches
                return similar_experiences[:max_similar]
            else:
                # Fallback method if vector search not available
                logger.warning("Vector search not available, using fallback method")
                return []
                
        except Exception as e:
            logger.error(f"Error finding similar experiences: {e}")
            return []
    
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
    
    async def run_consolidation_cycle(self, experience_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a complete consolidation cycle using direct logic (not orchestrator).
        Modified to correctly store consolidated memories with hierarchical data.
        """
        # --- Time Check (Keep this) ---
        now = datetime.now()
        time_since_last = (now - self.last_consolidation).total_seconds() / 3600
        if time_since_last < self.consolidation_interval:
            logger.info(f"Skipping consolidation cycle: Only {time_since_last:.1f} hours passed ({self.consolidation_interval} required).")
            return {"status": "skipped", "reason": "Interval not met"}

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
                candidate_groups = [ConsolidationCandidate(**c) for c in raw]

                if not candidate_groups:
                    logger.info("No candidate groups found.")
                    self.last_consolidation = now
                    return {"status": "completed", "consolidations_created": 0, "source_memories_processed": 0}

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
                return {
                    "status": "completed",
                    "consolidations_created": consolidations_created,
                    "source_memories_processed": total_memories_affected
                }

            except Exception as e:
                logger.error(f"Unexpected error in consolidation cycle: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

    
    async def get_consolidation_insights(self) -> Dict[str, Any]:
        """
        Get insights about consolidation activities
        
        Returns:
            Consolidation insights
        """
        with trace(workflow_name="get_consolidation_insights", group_id=self.trace_group_id):
            # Use the _get_consolidation_statistics tool
            stats = await self._get_consolidation_statistics(RunContextWrapper(context=self.context))
            
            # Add additional insights
            insights = {
                "total_consolidations": stats["total_consolidations"],
                "last_consolidation": self.last_consolidation.isoformat(),
                "consolidation_types": stats["type_distribution"],
                "user_coverage": set()
            }
            
            # Count consolidation types
            for entry in self.consolidation_history:
                # Track unique users
                user_ids = entry.get("user_ids", [])
                for user_id in user_ids:
                    insights["user_coverage"].add(user_id)
            
            # Convert set to count
            insights["unique_users_consolidated"] = len(insights["user_coverage"])
            insights["user_coverage"] = list(insights["user_coverage"])
            
            # Add time until next consolidation
            now = datetime.now()
            time_since_last = (now - self.last_consolidation).total_seconds() / 3600  # hours
            time_until_next = max(0, self.consolidation_interval - time_since_last)
            
            insights["hours_until_next_consolidation"] = time_until_next
            insights["ready_for_consolidation"] = time_until_next <= 0
            
            return insights
