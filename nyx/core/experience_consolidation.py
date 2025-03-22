# nyx/core/experience_consolidation.py

import logging
import asyncio
import random
import math
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from agents import Agent, Runner, trace, function_tool, RunContextWrapper
from pydantic import BaseModel, Field

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

class ExperienceConsolidationSystem:
    """
    System for consolidating similar experiences into higher-level abstractions.
    Identifies patterns, trends, and recurring themes across experiences.
    """
    
    def __init__(self, memory_core=None, experience_interface=None):
        """
        Initialize the experience consolidation system.
        
        Args:
            memory_core: Memory core for retrieving and storing experiences
            experience_interface: Experience interface for experience processing
        """
        self.memory_core = memory_core
        self.experience_interface = experience_interface
        
        # Initialize agents
        self.candidate_finder_agent = self._create_candidate_finder_agent()
        self.consolidation_agent = self._create_consolidation_agent()
        self.evaluation_agent = self._create_evaluation_agent()
        
        # Configuration settings
        self.similarity_threshold = 0.7
        self.consolidation_interval = 24  # hours
        self.max_group_size = 5
        self.min_group_size = 2
        self.quality_threshold = 0.6
        
        # State tracking
        self.last_consolidation = datetime.datetime.now() - datetime.timedelta(hours=25)  # Start ready for consolidation
        self.consolidation_history = []
        self.max_history_size = 100
        
        # Trace ID for connecting traces
        self.trace_group_id = f"exp_consolidation_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info("Experience Consolidation System initialized")
    
    def _create_candidate_finder_agent(self) -> Agent:
        """Create the candidate finder agent"""
        return Agent(
            name="Consolidation Candidate Finder",
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
    
    # Tool functions
    
    @function_tool
    async def _get_experience_details(self, ctx: RunContextWrapper, experience_id: str) -> Dict[str, Any]:
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
    
    @function_tool
    async def _calculate_similarity_score(self, ctx: RunContextWrapper, 
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
            
            # Fallback: simple similarity calculation
            # Same scenario type adds similarity
            similarity = 0.3
            if exp1["scenario_type"] == exp2["scenario_type"]:
                similarity += 0.2
            
            # Common tags add similarity
            common_tags = set(exp1["tags"]) & set(exp2["tags"])
            similarity += min(0.3, len(common_tags) * 0.05)
            
            # Similar emotional context adds similarity
            if exp1["emotional_context"] and exp2["emotional_context"]:
                primary1 = exp1["emotional_context"].get("primary_emotion", "")
                primary2 = exp2["emotional_context"].get("primary_emotion", "")
                
                if primary1 and primary2 and primary1 == primary2:
                    similarity += 0.2
            
            return min(1.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    @function_tool
    async def _find_common_theme(self, ctx: RunContextWrapper, experience_ids: List[str]) -> str:
        """
        Find common theme across multiple experiences
        
        Args:
            experience_ids: List of experience IDs
            
        Returns:
            Common theme description
        """
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
            
            # Get most common scenario type
            common_scenario = max(scenario_counts.items(), key=lambda x: x[1])[0] if scenario_counts else "general"
            
            # Count emotional context
            emotion_counts = {}
            for exp in experiences:
                emotion = exp.get("emotional_context", {}).get("primary_emotion", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Get most common emotion
            common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
            
            # Count tags
            tag_counts = {}
            for exp in experiences:
                for tag in exp["tags"]:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Get common tags
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
    
    @function_tool
    async def _sort_candidate_groups(self, ctx: RunContextWrapper, 
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
    
    @function_tool
    async def _extract_common_emotional_context(self, ctx: RunContextWrapper, 
                                           experience_ids: List[str]) -> Dict[str, Any]:
        """
        Extract common emotional context from multiple experiences
        
        Args:
            experience_ids: List of experience IDs
            
        Returns:
            Common emotional context
        """
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
    
    @function_tool
    async def _generate_consolidation_type(self, ctx: RunContextWrapper, experiences: List[Dict[str, Any]]) -> str:
        """
        Determine appropriate consolidation type based on experiences
        
        Args:
            experiences: List of experiences
            
        Returns:
            Consolidation type
        """
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
                    ts = datetime.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
                    timestamps.append(ts)
                except:
                    pass
        
        # If experiences span a significant time period, use trend
        if timestamps and max(timestamps) - min(timestamps) > datetime.timedelta(days=7):
            return "trend"
        
        # Default to pattern
        return "pattern"
    
    @function_tool
    async def _calculate_significance_score(self, ctx: RunContextWrapper, 
                                       experiences: List[Dict[str, Any]]) -> float:
        """
        Calculate significance score for consolidated experience
        
        Args:
            experiences: List of experiences
            
        Returns:
            Significance score (0.0-10.0)
        """
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
    
    @function_tool
    async def _calculate_coverage_score(self, ctx: RunContextWrapper,
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
    
    @function_tool
    async def _calculate_coherence_score(self, ctx: RunContextWrapper, consolidated: Dict[str, Any]) -> float:
        """
        Calculate coherence of the consolidated experience
        
        Args:
            consolidated: Consolidated experience
            
        Returns:
            Coherence score (0.0-1.0)
        """
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
    
    @function_tool
    async def _calculate_information_gain(self, ctx: RunContextWrapper,
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
    
    # Public methods
    
    async def find_consolidation_candidates(self, experience_ids: List[str]) -> List[ConsolidationCandidate]:
        """
        Find candidate groups of experiences for consolidation
        
        Args:
            experience_ids: List of experience IDs to consider
            
        Returns:
            List of candidate groups
        """
        with trace(workflow_name="find_consolidation_candidates", group_id=self.trace_group_id):
            # Get experience details
            experiences = []
            for exp_id in experience_ids:
                exp = await self._get_experience_details(RunContextWrapper(context=None), exp_id)
                if not "error" in exp:
                    experiences.append(exp)
            
            if not experiences:
                return []
            
            # Build similarity matrix
            similarity_matrix = {}
            for i, exp1 in enumerate(experiences):
                exp_id1 = exp1["id"]
                similarity_matrix[exp_id1] = {}
                
                for j, exp2 in enumerate(experiences):
                    if i == j:
                        continue
                        
                    exp_id2 = exp2["id"]
                    
                    # Calculate similarity
                    similarity = await self._calculate_similarity_score(
                        RunContextWrapper(context=None),
                        exp_id1,
                        exp_id2
                    )
                    
                    if similarity >= self.similarity_threshold:
                        similarity_matrix[exp_id1][exp_id2] = similarity
            
            # Find candidate groups
            candidate_groups = []
            
            # Start with highly similar pairs
            for exp_id1, similarities in similarity_matrix.items():
                if not similarities:
                    continue
                    
                # Sort similarities
                sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                
                for exp_id2, similarity in sorted_sims:
                    # Check if either experience is already in a group
                    already_grouped = False
                    for group in candidate_groups:
                        if exp_id1 in group["source_ids"] or exp_id2 in group["source_ids"]:
                            already_grouped = True
                            break
                    
                    if already_grouped:
                        continue
                    
                    # Start a new group with this pair
                    # Get experiences
                    exp1 = next((e for e in experiences if e["id"] == exp_id1), None)
                    exp2 = next((e for e in experiences if e["id"] == exp_id2), None)
                    
                    if not exp1 or not exp2:
                        continue
                    
                    # Find common theme
                    theme = await self._find_common_theme(
                        RunContextWrapper(context=None),
                        [exp_id1, exp_id2]
                    )
                    
                    # Determine scenario type
                    if exp1["scenario_type"] == exp2["scenario_type"]:
                        scenario_type = exp1["scenario_type"]
                    else:
                        scenario_type = "mixed"
                    
                    # Create group
                    group = {
                        "source_ids": [exp_id1, exp_id2],
                        "similarity_score": similarity,
                        "scenario_type": scenario_type,
                        "theme": theme,
                        "user_ids": [exp1["user_id"], exp2["user_id"]],
                        "consolidation_type": "pattern"
                    }
                    
                    candidate_groups.append(group)
            
            # Expand groups to include more experiences
            expanded_groups = []
            
            for group in candidate_groups:
                # Skip if group is already at maximum size
                if len(group["source_ids"]) >= self.max_group_size:
                    expanded_groups.append(group)
                    continue
                
                # Find potential new members
                potential_members = set()
                for exp_id in group["source_ids"]:
                    for other_id, similarity in similarity_matrix.get(exp_id, {}).items():
                        if other_id not in group["source_ids"] and similarity >= self.similarity_threshold:
                            potential_members.add(other_id)
                
                # Sort potential members by average similarity to current group
                scored_members = []
                for potential_id in potential_members:
                    # Calculate average similarity to current group
                    similarities = [
                        similarity_matrix.get(member_id, {}).get(potential_id, 0)
                        for member_id in group["source_ids"]
                    ]
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                    
                    scored_members.append((potential_id, avg_similarity))
                
                # Sort by similarity
                sorted_members = sorted(scored_members, key=lambda x: x[1], reverse=True)
                
                # Add members until max size
                expanded_group = group.copy()
                for potential_id, sim in sorted_members:
                    if len(expanded_group["source_ids"]) >= self.max_group_size:
                        break
                        
                    if sim >= self.similarity_threshold:
                        # Add to group
                        expanded_group["source_ids"].append(potential_id)
                        
                        # Update user IDs
                        exp = next((e for e in experiences if e["id"] == potential_id), None)
                        if exp:
                            expanded_group["user_ids"].append(exp["user_id"])
                
                # Recalculate group properties
                if len(expanded_group["source_ids"]) > len(group["source_ids"]):
                    # Recalculate similarity score (average pairwise similarity)
                    similarities = []
                    for i, id1 in enumerate(expanded_group["source_ids"]):
                        for j, id2 in enumerate(expanded_group["source_ids"]):
                            if i < j:
                                sim = similarity_matrix.get(id1, {}).get(id2, 0)
                                similarities.append(sim)
                    
                    expanded_group["similarity_score"] = sum(similarities) / len(similarities) if similarities else 0
                    
                    # Recalculate theme
                    expanded_group["theme"] = await self._find_common_theme(
                        RunContextWrapper(context=None),
                        expanded_group["source_ids"]
                    )
                    
                    # Determine consolidation type
                    expanded_group["consolidation_type"] = "pattern"
                    if len(set(expanded_group["user_ids"])) > 1:
                        expanded_group["consolidation_type"] = "abstraction"
                    
                    # Determine scenario type
                    scenario_types = []
                    for exp_id in expanded_group["source_ids"]:
                        exp = next((e for e in experiences if e["id"] == exp_id), None)
                        if exp:
                            scenario_types.append(exp["scenario_type"])
                    
                    if scenario_types:
                        if len(set(scenario_types)) == 1:
                            expanded_group["scenario_type"] = scenario_types[0]
                        else:
                            expanded_group["scenario_type"] = "mixed"
                
                expanded_groups.append(expanded_group)
            
            # Filter groups to ensure minimum size and similarity
            valid_groups = [
                group for group in expanded_groups
                if len(group["source_ids"]) >= self.min_group_size
                and group["similarity_score"] >= self.similarity_threshold
            ]
            
            # Sort groups by quality
            sorted_groups = await self._sort_candidate_groups(
                RunContextWrapper(context=None),
                valid_groups
            )
            
            # Convert to pydantic models
            candidates = []
            for group in sorted_groups:
                try:
                    candidate = ConsolidationCandidate(
                        source_ids=group["source_ids"],
                        similarity_score=group["similarity_score"],
                        scenario_type=group["scenario_type"],
                        theme=group["theme"],
                        user_ids=group["user_ids"],
                        consolidation_type=group["consolidation_type"]
                    )
                    candidates.append(candidate)
                except Exception as e:
                    logger.error(f"Error creating candidate: {e}")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding consolidation candidates: {e}")
            return []
    
    async def create_consolidated_experience(self, 
                                         candidate: ConsolidationCandidate) -> Optional[ConsolidationOutput]:
        """
        Create a consolidated experience from a candidate group
        
        Args:
            candidate: Consolidation candidate group
            
        Returns:
            Consolidated experience or None if creation fails
        """
        with trace(workflow_name="create_consolidated_experience", group_id=self.trace_group_id):
            try:
                # Get source experiences
                source_experiences = []
                for exp_id in candidate.source_ids:
                    exp = await self._get_experience_details(RunContextWrapper(context=None), exp_id)
                    if not "error" in exp:
                        source_experiences.append(exp)
                
                if not source_experiences:
                    logger.error("No valid source experiences found")
                    return None
                
                # Create agent input
                agent_input = {
                    "role": "user",
                    "content": f"Create a consolidated experience from these {len(source_experiences)} related experiences. Theme: {candidate.theme}",
                    "experiences": source_experiences,
                    "consolidation_type": candidate.consolidation_type
                }
                
                # Run the consolidation agent
                result = await Runner.run(
                    self.consolidation_agent,
                    agent_input
                )
                
                # Parse the output
                consolidation_output = result.final_output_as(ConsolidationOutput)
                
                # Store the consolidated experience
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
                                RunContextWrapper(context=None),
                                candidate.source_ids
                            ),
                            "user_ids": candidate.user_ids,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        # Create tags
                        tags = ["consolidated", candidate.consolidation_type, candidate.scenario_type]
                        if "tags" in consolidation_output.model_dump():
                            tags.extend([t for t in consolidation_output.tags if t not in tags])
                        
                        # Store in memory
                        memory_id = await self.memory_core.add_memory(
                            memory_text=consolidation_output.consolidation_text,
                            memory_type="consolidated",
                            memory_scope="game",
                            significance=consolidation_output.significance,
                            tags=tags,
                            metadata=metadata
                        )
                        
                        # Add consolidated ID to output
                        consolidation_data = consolidation_output.model_dump()
                        consolidation_data["id"] = memory_id
                        
                        # Add to vector embeddings if experience interface available
                        if self.experience_interface and hasattr(self.experience_interface, "_generate_experience_vector"):
                            vector = await self.experience_interface._generate_experience_vector(
                                RunContextWrapper(context=None),
                                consolidation_output.consolidation_text
                            )
                            
                            # Store vector
                            self.experience_interface.experience_vectors[memory_id] = {
                                "experience_id": memory_id,
                                "vector": vector,
                                "metadata": {
                                    "is_consolidation": True,
                                    "source_ids": candidate.source_ids,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                            }
                        
                        # Record in history
                        self.consolidation_history.append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "consolidation_id": memory_id,
                            "source_ids": candidate.source_ids,
                            "consolidation_type": candidate.consolidation_type,
                            "theme": candidate.theme
                        })
                        
                        # Limit history size
                        if len(self.consolidation_history) > self.max_history_size:
                            self.consolidation_history = self.consolidation_history[-self.max_history_size:]
                        
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
        Evaluate the quality of a consolidated experience
        
        Args:
            consolidated_id: ID of consolidated experience
            source_ids: IDs of source experiences
            
        Returns:
            Evaluation results or None if evaluation fails
        """
        with trace(workflow_name="evaluate_consolidation", group_id=self.trace_group_id):
            try:
                # Get consolidated experience
                consolidated = await self._get_experience_details(RunContextWrapper(context=None), consolidated_id)
                if "error" in consolidated:
                    logger.error(f"Consolidated experience not found: {consolidated_id}")
                    return None
                
                # Get source experiences
                source_experiences = []
                for exp_id in source_ids:
                    exp = await self._get_experience_details(RunContextWrapper(context=None), exp_id)
                    if not "error" in exp:
                        source_experiences.append(exp)
                
                if not source_experiences:
                    logger.error("No valid source experiences found")
                    return None
                
                # Create agent input
                agent_input = {
                    "role": "user",
                    "content": "Evaluate the quality of this consolidated experience",
                    "consolidated": consolidated,
                    "source_experiences": source_experiences
                }
                
                # Run the evaluation agent
                result = await Runner.run(
                    self.evaluation_agent,
                    agent_input
                )
                
                # Parse the output
                evaluation_output = result.final_output_as(ConsolidationEvaluation)
                
                return evaluation_output
                
            except Exception as e:
                logger.error(f"Error evaluating consolidation: {e}")
                return None
    
    async def run_consolidation_cycle(self, experience_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a complete consolidation cycle to find and create consolidations
        
        Args:
            experience_ids: Optional list of experience IDs to consider (if None, uses all available)
            
        Returns:
            Results of the consolidation cycle
        """
        with trace(workflow_name="consolidation_cycle", group_id=self.trace_group_id):
            # Check if enough time has passed since last consolidation
            now = datetime.datetime.now()
            time_since_last = (now - self.last_consolidation).total_seconds() / 3600  # hours
            
            if time_since_last < self.consolidation_interval:
                return {
                    "status": "skipped",
                    "reason": f"Not enough time elapsed since last consolidation ({time_since_last:.1f} hours of {self.consolidation_interval} required)"
                }
            
            try:
                # Get experience IDs if not provided
                if not experience_ids and self.memory_core:
                    # Get experiences from memory core
                    # Exclude already consolidated experiences to avoid double consolidation
                    memories = await self.memory_core.search_memories(
                        query="",
                        memory_types=["experience"],
                        exclude_tags=["consolidated"],
                        limit=100
                    )
                    
                    experience_ids = [memory["id"] for memory in memories]
                
                if not experience_ids:
                    return {
                        "status": "skipped",
                        "reason": "No experiences available for consolidation"
                    }
                
                # Find consolidation candidates
                candidates = await self.find_consolidation_candidates(experience_ids)
                
                if not candidates:
                    # Update last consolidation time even if no candidates found
                    self.last_consolidation = now
                    
                    return {
                        "status": "completed",
                        "candidates_found": 0,
                        "consolidations_created": 0,
                        "message": "No suitable consolidation candidates found"
                    }
                
                # Create consolidations for top candidates
                consolidations = []
                for candidate in candidates[:5]:  # Limit to 5 consolidations per cycle
                    consolidated = await self.create_consolidated_experience(candidate)
                    
                    if consolidated:
                        # Evaluate the consolidation
                        evaluation = await self.evaluate_consolidation(
                            consolidated.id if hasattr(consolidated, "id") else "unknown",
                            candidate.source_ids
                        )
                        
                        # Only keep if quality is above threshold
                        if evaluation and evaluation.overall_quality >= self.quality_threshold:
                            consolidations.append({
                                "id": consolidated.id if hasattr(consolidated, "id") else "unknown",
                                "text": consolidated.consolidation_text,
                                "source_count": len(candidate.source_ids),
                                "source_ids": candidate.source_ids,
                                "theme": candidate.theme,
                                "type": candidate.consolidation_type,
                                "quality": evaluation.overall_quality
                            })
                
                # Update last consolidation time
                self.last_consolidation = now
                
                return {
                    "status": "completed",
                    "candidates_found": len(candidates),
                    "consolidations_created": len(consolidations),
                    "consolidations": consolidations,
                    "timestamp": now.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error in consolidation cycle: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    async def get_consolidation_insights(self) -> Dict[str, Any]:
        """
        Get insights about consolidation activities
        
        Returns:
            Consolidation insights
        """
        insights = {
            "total_consolidations": len(self.consolidation_history),
            "last_consolidation": self.last_consolidation.isoformat(),
            "consolidation_types": {},
            "user_coverage": set()
        }
        
        # Count consolidation types
        for entry in self.consolidation_history:
            c_type = entry.get("consolidation_type", "unknown")
            insights["consolidation_types"][c_type] = insights["consolidation_types"].get(c_type, 0) + 1
            
            # Track unique users
            user_ids = entry.get("user_ids", [])
            for user_id in user_ids:
                insights["user_coverage"].add(user_id)
        
        # Convert set to count
        insights["unique_users_consolidated"] = len(insights["user_coverage"])
        insights["user_coverage"] = list(insights["user_coverage"])
        
        # Add time until next consolidation
        now = datetime.datetime.now()
        time_since_last = (now - self.last_consolidation).total_seconds() / 3600  # hours
        time_until_next = max(0, self.consolidation_interval - time_since_last)
        
        insights["hours_until_next_consolidation"] = time_until_next
        insights["ready_for_consolidation"] = time_until_next <= 0
        
        return insights
