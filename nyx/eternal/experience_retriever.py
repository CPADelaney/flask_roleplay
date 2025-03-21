# nyx/eternal/experience_retriever.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from memory.core import Memory, MemoryType, MemorySignificance
from memory.emotional import EmotionalMemoryManager
from memory.integrated import IntegratedMemorySystem

logger = logging.getLogger("experience_retriever")

class ExperienceRetriever:
    """
    Specialized component for retrieving relevant past experiences.
    Focuses on finding experiences that relate to current conversation context.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None
        self.emotional_system = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the experience retriever."""
        if self.initialized:
            return
            
        # Get instances of required systems
        self.memory_system = await IntegratedMemorySystem.get_instance(
            self.user_id, self.conversation_id
        )
        
        self.emotional_system = await EmotionalMemoryManager(
            self.user_id, self.conversation_id
        )
        
        self.initialized = True
        logger.info(f"Experience retriever initialized for user {self.user_id}")
    
    async def retrieve_relevant_experiences(self, 
                                          current_context: Dict[str, Any],
                                          limit: int = 3,
                                          min_relevance: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve experiences relevant to the current conversation context.
        
        Args:
            current_context: Current conversation context including:
                - query: Search query or current topic
                - scenario_type: Type of scenario (e.g., "teasing", "dark")
                - emotional_state: Current emotional state
                - entities: Entities involved in current context
            limit: Maximum number of experiences to return
            min_relevance: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of relevant experiences with metadata
        """
        if not self.initialized:
            await self.initialize()
        
        # Extract key information from context
        query = current_context.get("query", "")
        scenario_type = current_context.get("scenario_type", "")
        emotional_state = current_context.get("emotional_state", {})
        entities = current_context.get("entities", [])
        
        # Get base memories from memory system
        base_memories = await self.memory_system.retrieve_memories(
            query=query,
            memory_types=["observation", "reflection", "episodic"],
            limit=limit*3,  # Get more to filter later
            min_significance=MemorySignificance.MEDIUM
        )
        
        if not base_memories:
            logger.info("No base memories found for experience retrieval")
            return []
        
        # Score memories for relevance
        scored_memories = []
        for memory in base_memories:
            # Calculate relevance score
            relevance_score = await self._calculate_relevance_score(
                memory, current_context
            )
            
            # Skip low-relevance memories
            if relevance_score < min_relevance:
                continue
            
            # Get emotional context for this memory
            emotional_context = await self._get_memory_emotional_context(memory)
            
            # Calculate experiential richness (how much detail/emotion it has)
            experiential_richness = self._calculate_experiential_richness(
                memory, emotional_context
            )
            
            # Add to scored memories
            scored_memories.append({
                "memory": memory,
                "relevance_score": relevance_score,
                "emotional_context": emotional_context,
                "experiential_richness": experiential_richness,
                "final_score": relevance_score * 0.7 + experiential_richness * 0.3
            })
        
        # Sort by final score
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Process top memories into experiences
        experiences = []
        for item in scored_memories[:limit]:
            experience = await self._convert_memory_to_experience(
                item["memory"],
                item["emotional_context"],
                item["relevance_score"],
                item["experiential_richness"]
            )
            experiences.append(experience)
        
        return experiences
    
    async def _calculate_relevance_score(self, 
                                      memory: Dict[str, Any], 
                                      context: Dict[str, Any]) -> float:
        """
        Calculate how relevant a memory is to the current context.
        
        Args:
            memory: Memory to evaluate
            context: Current conversation context
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Extract memory attributes
        memory_text = memory.get("memory_text", "")
        memory_tags = memory.get("tags", [])
        memory_metadata = memory.get("metadata", {})
        
        # Initialize score components
        semantic_score = 0.0
        tag_score = 0.0
        temporal_score = 0.0
        emotional_score = 0.0
        entity_score = 0.0
        
        # Semantic relevance (via embedding similarity)
        # This would ideally use the embedding similarity from your memory system
        if "relevance" in memory:
            semantic_score = memory.get("relevance", 0.0)
        elif "embedding" in memory and "query_embedding" in context:
            # Calculate cosine similarity if both embeddings available
            memory_embedding = np.array(memory["embedding"])
            query_embedding = np.array(context["query_embedding"])
            
            # Cosine similarity
            dot_product = np.dot(memory_embedding, query_embedding)
            norm_product = np.linalg.norm(memory_embedding) * np.linalg.norm(query_embedding)
            semantic_score = dot_product / norm_product if norm_product > 0 else 0.0
        
        # Tag matching
        scenario_type = context.get("scenario_type", "").lower()
        context_tags = context.get("tags", [])
        
        # Count matching tags
        matching_tags = 0
        for tag in memory_tags:
            if tag.lower() in scenario_type or tag in context_tags:
                matching_tags += 1
        
        tag_score = min(1.0, matching_tags / 3) if memory_tags else 0.0
        
        # Temporal relevance (newer memories score higher)
        if "timestamp" in memory:
            memory_time = datetime.fromisoformat(memory["timestamp"]) if isinstance(memory["timestamp"], str) else memory["timestamp"]
            age_in_days = (datetime.now() - memory_time).total_seconds() / 86400
            # Newer memories get higher scores, but not too dominant
            temporal_score = max(0.0, 1.0 - (age_in_days / 180))  # 6 month scale
        
        # Emotional relevance
        memory_emotions = memory_metadata.get("emotions", {})
        context_emotions = context.get("emotional_state", {})
        
        if memory_emotions and context_emotions:
            # Compare primary emotions
            memory_primary = memory_emotions.get("primary", {}).get("name", "neutral")
            context_primary = context_emotions.get("primary_emotion", "neutral")
            
            # Emotion match bonus
            if memory_primary == context_primary:
                emotional_score += 0.5
                
            # Emotional intensity comparison
            memory_intensity = memory_emotions.get("primary", {}).get("intensity", 0.5)
            context_intensity = context_emotions.get("intensity", 0.5)
            
            # Similar intensity bonus
            emotional_score += 0.5 * (1.0 - abs(memory_intensity - context_intensity))
        
        # Entity relevance
        memory_entities = memory_metadata.get("entities", [])
        context_entities = context.get("entities", [])
        
        if memory_entities and context_entities:
            matching_entities = len(set(memory_entities).intersection(set(context_entities)))
            entity_score = min(1.0, matching_entities / len(context_entities)) if context_entities else 0.0
        
        # Combine scores with weights
        final_score = (
            semantic_score * 0.35 +
            tag_score * 0.20 +
            temporal_score * 0.10 +
            emotional_score * 0.20 +
            entity_score * 0.15
        )
        
        return min(1.0, max(0.0, final_score))
    
    async def _get_memory_emotional_context(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or infer emotional context for a memory.
        
        Args:
            memory: Memory to analyze
            
        Returns:
            Emotional context information
        """
        # Check if memory already has emotional data
        metadata = memory.get("metadata", {})
        if "emotions" in metadata:
            return metadata["emotions"]
        
        # If memory has emotional intensity but no detailed emotions, infer them
        emotional_intensity = memory.get("emotional_intensity", 0)
        if emotional_intensity > 0:
            # Get memory tags for context
            tags = memory.get("tags", [])
            
            # For best results, analyze the text directly
            # This uses your emotional system's analysis capabilities
            try:
                analysis = await self.emotional_system.analyze_emotional_content(
                    memory.get("memory_text", "")
                )
                return {
                    "primary": {
                        "name": analysis.get("primary_emotion", "neutral"),
                        "intensity": analysis.get("intensity", 0.5)
                    },
                    "secondary": analysis.get("secondary_emotions", {}),
                    "valence": analysis.get("valence", 0.0),
                    "arousal": analysis.get("arousal", 0.0)
                }
            except Exception as e:
                logger.error(f"Error analyzing emotional content: {e}")
                
                # Fallback: infer from intensity and tags
                primary_emotion = "neutral"
                for tag in tags:
                    if tag in ["joy", "sadness", "anger", "fear", "disgust", 
                              "surprise", "anticipation", "trust"]:
                        primary_emotion = tag
                        break
                
                return {
                    "primary": {
                        "name": primary_emotion,
                        "intensity": emotional_intensity / 100  # Convert to 0-1 scale
                    },
                    "secondary": {},
                    "valence": 0.5 if primary_emotion in ["joy", "anticipation", "trust"] else -0.5,
                    "arousal": 0.7 if emotional_intensity > 50 else 0.3
                }
        
        # Default emotional context if nothing else available
        return {
            "primary": {
                "name": "neutral",
                "intensity": 0.3
            },
            "secondary": {},
            "valence": 0.0,
            "arousal": 0.2
        }
    
    def _calculate_experiential_richness(self, 
                                       memory: Dict[str, Any], 
                                       emotional_context: Dict[str, Any]) -> float:
        """
        Calculate how rich and detailed the experience is.
        Higher scores mean the memory has more emotional and sensory detail.
        
        Args:
            memory: Memory to evaluate
            emotional_context: Emotional context of the memory
            
        Returns:
            Experiential richness score (0.0-1.0)
        """
        # Extract memory attributes
        memory_text = memory.get("memory_text", "")
        memory_tags = memory.get("tags", [])
        significance = memory.get("significance", 3)
        
        # Initialize richness factors
        detail_score = 0.0
        emotional_depth = 0.0
        sensory_richness = 0.0
        significance_score = 0.0
        
        # Text length as a proxy for detail (longer memories might have more detail)
        # Capped at reasonable limits to avoid overly long memories dominating
        word_count = len(memory_text.split())
        detail_score = min(1.0, word_count / 100)  # Cap at 100 words
        
        # Emotional depth from context
        if emotional_context:
            # Primary emotion intensity
            primary_intensity = emotional_context.get("primary", {}).get("intensity", 0.0)
            
            # Count secondary emotions
            secondary_count = len(emotional_context.get("secondary", {}))
            
            # Combine for emotional depth
            emotional_depth = 0.7 * primary_intensity + 0.3 * min(1.0, secondary_count / 3)
        
        # Sensory richness via presence of sensory words
        sensory_words = ["see", "saw", "look", "hear", "heard", "sound", 
                        "feel", "felt", "touch", "smell", "scent", "taste"]
        
        sensory_count = sum(1 for word in sensory_words if word in memory_text.lower())
        sensory_richness = min(1.0, sensory_count / 5)  # Cap at 5 sensory words
        
        # Significance as a direct factor
        significance_score = significance / 10.0  # Convert to 0-1 scale
        
        # Combine scores with weights
        richness_score = (
            detail_score * 0.3 +
            emotional_depth * 0.4 +
            sensory_richness * 0.2 +
            significance_score * 0.1
        )
        
        return min(1.0, max(0.0, richness_score))
    
    async def _convert_memory_to_experience(self,
                                         memory: Dict[str, Any],
                                         emotional_context: Dict[str, Any],
                                         relevance_score: float,
                                         experiential_richness: float) -> Dict[str, Any]:
        """
        Convert a raw memory into a rich experience format.
        
        Args:
            memory: The base memory
            emotional_context: Emotional context information
            relevance_score: How relevant this memory is
            experiential_richness: How rich and detailed the experience is
            
        Returns:
            Formatted experience
        """
        # Get memory participants/entities
        metadata = memory.get("metadata", {})
        entities = metadata.get("entities", [])
        
        # Get scenario type from tags
        tags = memory.get("tags", [])
        scenario_types = [tag for tag in tags if tag in [
            "teasing", "dark", "indulgent", "psychological", "nurturing",
            "training", "discipline", "service", "worship", "punishment"
        ]]
        
        scenario_type = scenario_types[0] if scenario_types else "general"
        
        # Format the experience
        experience = {
            "id": memory.get("id"),
            "content": memory.get("memory_text", ""),
            "emotional_context": emotional_context,
            "scenario_type": scenario_type,
            "entities": entities,
            "timestamp": memory.get("timestamp"),
            "relevance_score": relevance_score,
            "experiential_richness": experiential_richness,
            "tags": tags,
            "significance": memory.get("significance", 3)
        }
        
        return experience

                                   # memory/experience_abstractor.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

from nyx.llm_integration import generate_text_completion

logger = logging.getLogger("experience_abstractor")

class ExperienceAbstractor:
    """
    Transforms memories into natural conversational summaries for Nyx to share.
    Makes experience recall feel natural rather than robotic.
    """
    
    def __init__(self):
        # Template patterns for different types of experience sharing
        self.recall_templates = {
            # Basic recall templates
            "standard": [
                "That reminds me of {timeframe} when {brief_summary}... {detail}",
                "I recall {timeframe} when {brief_summary}. {reflection}",
                "This is similar to {timeframe} when {brief_summary}... {detail}"
            ],
            
            # For emotionally positive experiences
            "positive": [
                "Mmm, I remember {timeframe} when {brief_summary}. {reflection}",
                "That brings back a delicious memory of {timeframe} when {brief_summary}... {detail}",
                "I quite enjoyed {timeframe} when {brief_summary}. {reflection}"
            ],
            
            # For emotionally negative experiences
            "negative": [
                "I recall {timeframe} dealing with someone who {brief_summary}. {reflection}",
                "This reminds me of a frustrating time when {brief_summary}... {detail}",
                "I once had to handle someone who {brief_summary}. {reflection}"
            ],
            
            # For intense experiences
            "intense": [
                "Mmm, that reminds me of an *intense* experience where {brief_summary}... {detail}",
                "I vividly remember when {brief_summary}. {reflection}",
                "I'll never forget when {brief_summary}... {detail}"
            ],
            
            # For teasing experiences
            "teasing": [
                "Oh, this reminds me of {timeframe} when I teased someone until {brief_summary}... {reflection}",
                "I once had such fun teasing someone who {brief_summary}. {reflection}",
                "There was this delicious time when I {brief_summary}... {detail}"
            ],
            
            # For disciplinary experiences
            "disciplinary": [
                "I remember having to discipline someone who {brief_summary}. {reflection}",
                "This reminds me of {timeframe} when I had to correct someone who {brief_summary}... {detail}",
                "I once dealt with someone who needed strict handling when they {brief_summary}. {reflection}"
            ]
        }
    
    def _get_timeframe_text(self, timestamp: Optional[str]) -> str:
        """
        Get conversational timeframe text from timestamp.
        
        Args:
            timestamp: ISO timestamp string
            
        Returns:
            Natural language timeframe like "last week" or "a while back"
        """
        if not timestamp:
            return "a while back"
            
        try:
            if isinstance(timestamp, str):
                memory_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                memory_time = timestamp
                
            days_ago = (datetime.now() - memory_time).days
            
            if days_ago < 1:
                return "earlier today"
            elif days_ago < 2:
                return "yesterday"
            elif days_ago < 7:
                return f"{days_ago} days ago"
            elif days_ago < 14:
                return "last week"
            elif days_ago < 30:
                return "a couple weeks ago"
            elif days_ago < 60:
                return "a month ago"
            elif days_ago < 365:
                return f"{days_ago // 30} months ago"
            else:
                return "a while back"
                
        except Exception as e:
            logger.error(f"Error processing timestamp: {e}")
            return "a while back"
    
    def _get_emotional_tone(self, emotional_context: Dict[str, Any]) -> str:
        """
        Determine the emotional tone for recall based on the experience's emotions.
        
        Args:
            emotional_context: Emotional context of the experience
            
        Returns:
            Tone category like "positive", "negative", or "intense"
        """
        if not emotional_context:
            return "standard"
            
        primary = emotional_context.get("primary", {})
        emotion_name = primary.get("name", "neutral")
        intensity = primary.get("intensity", 0.5)
        valence = emotional_context.get("valence", 0.0)
        
        # High intensity experiences
        if intensity > 0.8:
            return "intense"
            
        # Positive emotions
        if valence > 0.3 or emotion_name in ["joy", "anticipation", "trust"]:
            return "positive"
            
        # Negative emotions
        if valence < -0.3 or emotion_name in ["anger", "fear", "disgust", "sadness"]:
            return "negative"
            
        # Default to standard
        return "standard"
    
    def _get_scenario_tone(self, scenario_type: str) -> str:
        """
        Get tone based on scenario type.
        
        Args:
            scenario_type: Type of scenario
            
        Returns:
            Tone category
        """
        scenario_type = scenario_type.lower()
        
        if scenario_type in ["teasing", "indulgent"]:
            return "teasing"
        elif scenario_type in ["discipline", "punishment", "training"]:
            return "disciplinary"
        
        # Default to emotional tone
        return None
    
    async def generate_conversational_recall(self, 
                                          experience: Dict[str, Any],
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a natural, conversational recall of an experience.
        
        Args:
            experience: The experience to recall
            context: Current conversation context
            
        Returns:
            Conversational recall with reflection
        """
        # Extract experience data
        content = experience.get("content", "")
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        timestamp = experience.get("timestamp")
        
        # Get timeframe text
        timeframe = self._get_timeframe_text(timestamp)
        
        # Determine tone for recall
        emotional_tone = self._get_emotional_tone(emotional_context)
        scenario_tone = self._get_scenario_tone(scenario_type)
        
        # Select tone (prioritize scenario tone if available)
        tone = scenario_tone or emotional_tone
        
        # Get templates for this tone
        templates = self.recall_templates.get(tone, self.recall_templates["standard"])
        
        # Select a random template
        template = random.choice(templates)
        
        # Generate summary components using LLM
        components = await self._generate_summary_components(experience, context)
        
        brief_summary = components.get("brief_summary", "")
        detail = components.get("detail", "")
        reflection = components.get("reflection", "")
        
        # Fill in the template
        recall_text = template.format(
            timeframe=timeframe,
            brief_summary=brief_summary,
            detail=detail,
            reflection=reflection
        )
        
        return {
            "recall_text": recall_text,
            "summary_components": components,
            "template_used": template,
            "tone": tone
        }
    
    async def _generate_summary_components(self,
                                        experience: Dict[str, Any],
                                        context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate summary components for experience recall using LLM.
        
        Args:
            experience: The experience to summarize
            context: Current conversation context
            
        Returns:
            Dictionary with summary components
        """
        # Extract experience data
        content = experience.get("content", "")
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        
        # Get emotional information for context
        primary_emotion = emotional_context.get("primary", {}).get("name", "neutral")
        emotional_intensity = emotional_context.get("primary", {}).get("intensity", 0.5)
        
        # Current conversation context
        context_query = context.get("query", "") if context else ""
        
        # LLM prompt for summary components
        prompt = f"""
        Create narrative components for Nyx to recall this experience in a natural, conversational way:
        
        Experience: "{content}"
        
        Primary emotion during experience: {primary_emotion} (intensity: {emotional_intensity:.1f})
        Scenario type: {scenario_type}
        Current topic: {context_query}
        
        Generate these three components:
        1. Brief summary (10-15 words) - The core of what happened
        2. Detail (15-20 words) - A specific detail that makes it vivid
        3. Reflection (15-20 words) - Nyx's personal thoughts/feelings about it now
        
        Format your response as JSON with keys: "brief_summary", "detail", and "reflection".
        Ensure the tone matches Nyx's dominant personality.
        """
        
        try:
            # Use your LLM integration to generate components
            response = await generate_text_completion(
                system_prompt="You help Nyx recall experiences in a natural, conversational way.",
                user_prompt=prompt,
                temperature=0.7,
                task_type="reflection"
            )
            
            # Parse JSON response
            try:
                components = json.loads(response)
                return components
            except json.JSONDecodeError:
                # Fallback parsing if not valid JSON
                components = {}
                
                if "brief_summary" in response:
                    brief_start = response.find("brief_summary") + 15
                    brief_end = response.find('"', brief_start + 2)
                    components["brief_summary"] = response[brief_start:brief_end].strip('":\r\n ')
                    
                if "detail" in response:
                    detail_start = response.find("detail") + 8
                    detail_end = response.find('"', detail_start + 2)
                    components["detail"] = response[detail_start:detail_end].strip('":\r\n ')
                    
                if "reflection" in response:
                    reflect_start = response.find("reflection") + 12
                    reflect_end = response.find('"', reflect_start + 2)
                    components["reflection"] = response[reflect_start:reflect_end].strip('":\r\n ')
                
                return components
                
        except Exception as e:
            logger.error(f"Error generating summary components: {e}")
            
            # Fallback components if LLM fails
            return {
                "brief_summary": self._extract_brief_summary(content),
                "detail": self._extract_fallback_detail(content),
                "reflection": self._generate_fallback_reflection(emotional_context, scenario_type)
            }
    
    def _extract_brief_summary(self, content: str) -> str:
        """Generate a fallback brief summary from content."""
        # Simple extractive summary (first sentence, up to 15 words)
        first_sentence = content.split('.')[0]
        words = first_sentence.split()[:15]
        return ' '.join(words)
    
    def _extract_fallback_detail(self, content: str) -> str:
        """Extract a fallback detail from content."""
        # Try to get a detail from later in the content
        sentences = content.split('.')
        if len(sentences) > 1:
            detail_sentence = sentences[1]
            words = detail_sentence.split()[:20]
            return ' '.join(words)
        return ""
    
    def _generate_fallback_reflection(self, 
                                   emotional_context: Dict[str, Any],
                                   scenario_type: str) -> str:
        """Generate a fallback reflection based on emotion and scenario."""
        # Get emotional information
        primary_emotion = emotional_context.get("primary", {}).get("name", "neutral")
        
        # Map emotions to reflections
        emotion_reflections = {
            "joy": "I found it quite satisfying",
            "sadness": "It was somewhat disappointing",
            "anger": "It was rather frustrating",
            "fear": "It was rather unnerving",
            "disgust": "It was rather distasteful",
            "surprise": "It was quite unexpected",
            "anticipation": "I found myself looking forward to the outcome",
            "trust": "It reinforced my expectations",
            "neutral": "It was an interesting experience"
        }
        
        # Map scenario types to reflections
        scenario_reflections = {
            "teasing": "It was deliciously fun to watch them squirm",
            "dark": "The darkness of it was quite satisfying",
            "indulgent": "It was a delightful indulgence",
            "psychological": "The mind games were particularly satisfying",
            "nurturing": "I enjoyed guiding them through it",
            "training": "The training paid off in the end",
            "discipline": "The discipline was necessary and effective",
            "service": "Their service was noted and appreciated",
            "worship": "Their devotion was properly rewarded",
            "punishment": "The punishment fit the transgression perfectly"
        }
        
        # Prioritize scenario reflection if available
        if scenario_type in scenario_reflections:
            return scenario_reflections[scenario_type]
        
        # Otherwise use emotion-based reflection
        return emotion_reflections.get(primary_emotion, "It was an interesting experience")

import logging
import asyncio
from typing import Dict, List, Any, Optional
import json
import random

from nyx.llm_integration import generate_text_completion

logger = logging.getLogger("reflection_generator")

class ReflectionGenerator:
    """
    Generates personality-driven reflections for Nyx based on experiences.
    Helps Nyx form "opinions" about past experiences.
    """
    
    def __init__(self):
        # Reflection patterns based on scenario types
        self.reflection_patterns = {
            "teasing": {
                "positive": [
                    "I particularly enjoy when a {subject} {response_verb} to teasing. It's {reaction_adj}.",
                    "There's something {quality_adj} about watching a {subject} {response_verb} when teased just right.",
                    "The way some {subject}s {response_verb} to teasing... mmm, {reaction_adj}."
                ],
                "negative": [
                    "Some {subject}s just don't {response_verb} well to teasing. It's rather {reaction_adj}.",
                    "I find it {quality_adj} when a {subject} {response_verb} incorrectly to teasing.",
                    "Not everyone can {response_verb} properly to teasing. Some are just {reaction_adj}."
                ]
            },
            "discipline": {
                "positive": [
                    "When a {subject} {response_verb} to discipline, it's {reaction_adj}.",
                    "There's something {quality_adj} about a {subject} who {response_verb} correctly to correction.",
                    "I appreciate those who {response_verb} properly when disciplined. It's {reaction_adj}."
                ],
                "negative": [
                    "Some {subject}s simply don't {response_verb} properly to discipline. Rather {reaction_adj}.",
                    "It's {quality_adj} when a {subject} fails to {response_verb} to proper discipline.",
                    "Those who can't {response_verb} to discipline are {reaction_adj}."
                ]
            },
            "service": {
                "positive": [
                    "I value a {subject} who {response_verb} eagerly in service. It's {reaction_adj}.",
                    "There's something {quality_adj} about watching a {subject} {response_verb} in service.",
                    "When a {subject} {response_verb} properly in service, it's {reaction_adj}."
                ],
                "negative": [
                    "A {subject} who doesn't {response_verb} correctly in service is {reaction_adj}.",
                    "It's {quality_adj} when a {subject} fails to {response_verb} properly in service.",
                    "Those who cannot {response_verb} adequately in service are {reaction_adj}."
                ]
            },
            "general": {
                "positive": [
                    "I find it {quality_adj} when a {subject} {response_verb} that way.",
                    "There's something {reaction_adj} about how some {subject}s {response_verb}.",
                    "A {subject} who can {response_verb} properly is {reaction_adj}."
                ],
                "negative": [
                    "It's rather {quality_adj} when a {subject} {response_verb} that way.",
                    "Some {subject}s who {response_verb} like that are {reaction_adj}.",
                    "I find it {reaction_adj} when a {subject} tries to {response_verb} incorrectly."
                ]
            }
        }
    
    async def generate_personality_reflection(self,
                                           experiences: List[Dict[str, Any]],
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a personality-driven reflection based on experiences.
        
        Args:
            experiences: List of experiences to reflect on
            context: Current conversation context
            
        Returns:
            Personality-driven reflection
        """
        if not experiences:
            return {
                "reflection": "I don't have specific experiences to reflect on yet.",
                "confidence": 0.3
            }
        
        # Extract key information from top experience
        top_experience = experiences[0]
        
        # For more complex reflections with multiple experiences, use LLM
        if len(experiences) > 1:
            return await self._generate_complex_reflection(experiences, context)
        
        # For simpler reflections based on a single experience
        scenario_type = top_experience.get("scenario_type", "general")
        emotional_context = top_experience.get("emotional_context", {})
        
        # Determine if positive or negative experience
        valence = emotional_context.get("valence", 0.0)
        sentiment = "positive" if valence >= 0 else "negative"
        
        # Get reflection templates for this scenario and sentiment
        scenario_patterns = self.reflection_patterns.get(
            scenario_type, self.reflection_patterns["general"]
        )
        
        templates = scenario_patterns.get(sentiment, scenario_patterns["positive"])
        
        # Select a random template
        template = random.choice(templates)
        
        # Generate template variables
        template_vars = await self._generate_template_variables(top_experience)
        
        # Fill in the template
        reflection_text = template.format(**template_vars)
        
        return {
            "reflection": reflection_text,
            "confidence": 0.7,
            "experience_id": top_experience.get("id"),
            "scenario_type": scenario_type,
            "sentiment": sentiment
        }
    
    async def _generate_template_variables(self, 
                                        experience: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate variables to fill in reflection templates.
        
        Args:
            experience: Experience to reflect on
            
        Returns:
            Template variables
        """
        # Extract information to help generate variables
        content = experience.get("content", "")
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        
        # Subject variables
        subject_options = ["subject", "pet", "submissive", "plaything", "toy"]
        subject = random.choice(subject_options)
        
        # Response verb options based on scenario
        response_verbs = {
            "teasing": ["responds", "reacts", "squirms", "blushes", "moans"],
            "discipline": ["submits", "yields", "accepts", "responds", "behaves"],
            "service": ["performs", "serves", "attends", "kneels", "obeys"],
            "general": ["responds", "reacts", "behaves", "performs", "acts"]
        }
        
        scenario_verbs = response_verbs.get(scenario_type, response_verbs["general"])
        response_verb = random.choice(scenario_verbs)
        
        # Quality adjective options based on emotional context
        valence = emotional_context.get("valence", 0.0)
        
        if valence >= 0.3:
            quality_adj_options = ["delightful", "satisfying", "enjoyable", "pleasing", "gratifying"]
        elif valence <= -0.3:
            quality_adj_options = ["disappointing", "frustrating", "tedious", "displeasing", "unsatisfying"]
        else:
            quality_adj_options = ["interesting", "curious", "notable", "peculiar", "unusual"]
            
        quality_adj = random.choice(quality_adj_options)
        
        # Reaction adjectives
        if valence >= 0.3:
            reaction_adj_options = ["quite satisfying", "delicious to witness", "rather enjoyable", "truly gratifying"]
        elif valence <= -0.3:
            reaction_adj_options = ["rather disappointing", "somewhat irritating", "quite vexing", "hardly worth my time"]
        else:
            reaction_adj_options = ["somewhat interesting", "moderately entertaining", "passably amusing"]
            
        reaction_adj = random.choice(reaction_adj_options)
        
        return {
            "subject": subject,
            "response_verb": response_verb,
            "quality_adj": quality_adj,
            "reaction_adj": reaction_adj
        }
    
    async def _generate_complex_reflection(self,
                                        experiences: List[Dict[str, Any]],
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a more complex reflection based on multiple experiences.
        Uses LLM for more sophisticated pattern recognition.
        
        Args:
            experiences: List of experiences to reflect on
            context: Current conversation context
            
        Returns:
            Complex reflection
        """
        # Format experiences for prompt
        formatted_experiences = []
        for i, exp in enumerate(experiences[:3]):  # Limit to top 3
            content = exp.get("content", "")
            scenario = exp.get("scenario_type", "general")
            formatted_experiences.append(f"Experience {i+1} ({scenario}): {content}")
        
        experiences_text = "\n".join(formatted_experiences)
        
        # Current conversation context
        context_query = context.get("query", "") if context else ""
        
        # LLM prompt for complex reflection
        prompt = f"""
        Generate a personality-driven reflection for Nyx based on these past experiences:
        
        {experiences_text}
        
        Current context/query: {context_query}
        
        The reflection should:
        1. Identify patterns or preferences based on these experiences
        2. Express Nyx's personal "opinion" about these patterns
        3. Match Nyx's dominant, confident personality
        4. Be approximately 2-3 sentences (50-75 words)
        5. Speak in first-person from Nyx's perspective
        
        Make the reflection feel like Nyx is drawing on lived experiences to form her own viewpoint.
        """
        
        try:
            # Use LLM to generate reflection
            response = await generate_text_completion(
                system_prompt="You help generate Nyx's personality-driven reflections on past experiences.",
                user_prompt=prompt,
                temperature=0.7,
                task_type="reflection"
            )
            
            # Determine confidence based on experience count and relevance
            relevance_scores = [exp.get("relevance_score", 0.5) for exp in experiences]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
            
            # More relevant experiences = higher confidence
            confidence = 0.5 + (avg_relevance * 0.3) + (min(len(experiences), 3) * 0.1 / 3)
            
            return {
                "reflection": response.strip(),
                "confidence": min(1.0, confidence),
                "experience_count": len(experiences),
                "experience_ids": [exp.get("id") for exp in experiences]
            }
                
        except Exception as e:
            logger.error(f"Error generating complex reflection: {e}")
            
            # Fallback: use the simple reflection for the top experience
            return await self.generate_personality_reflection([experiences[0]], context)


import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from memory.core import Memory, MemoryType, MemorySignificance

logger = logging.getLogger("experience_manager")

class ExperienceManager:
    """
    Manager for experience-based recall, integrating retrieval, abstraction,
    and reflection components into a single interface.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.retriever = ExperienceRetriever(user_id, conversation_id)
        self.abstractor = ExperienceAbstractor()
        self.reflector = ReflectionGenerator()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the experience manager."""
        if self.initialized:
            return
            
        # Initialize the retriever (which initializes required subsystems)
        await self.retriever.initialize()
        
        self.initialized = True
        logger.info(f"Experience manager initialized for user {self.user_id}")
    
    async def retrieve_and_share_experience(self,
                                         context: Dict[str, Any],
                                         limit: int = 3,
                                         min_relevance: float = 0.6) -> Dict[str, Any]:
        """
        Main method to retrieve and format experiences for sharing.
        
        Args:
            context: Current conversation context
            limit: Maximum number of experiences to retrieve
            min_relevance: Minimum relevance score for experiences
            
        Returns:
            Experience sharing results including natural recall text
        """
        if not self.initialized:
            await self.initialize()
        
        # Retrieve relevant experiences
        experiences = await self.retriever.retrieve_relevant_experiences(
            context, limit=limit, min_relevance=min_relevance
        )
        
        if not experiences:
            return {
                "experiences_found": False,
                "message": "No relevant experiences found."
            }
        
        # Generate conversational recall for top experience
        recall_result = await self.abstractor.generate_conversational_recall(
            experiences[0], context
        )
        
        # Generate personality-driven reflection if multiple experiences
        if len(experiences) > 1:
            reflection_result = await self.reflector.generate_personality_reflection(
                experiences, context
            )
            
            # Combine recall and reflection for rich response
            combined_text = f"{recall_result['recall_text']} {reflection_result['reflection']}"
            
            return {
                "experiences_found": True,
                "experience_count": len(experiences),
                "recall_text": recall_result["recall_text"],
                "reflection_text": reflection_result["reflection"],
                "combined_text": combined_text,
                "top_experience": experiences[0],
                "all_experiences": experiences,
                "confidence": reflection_result["confidence"]
            }
        else:
            # Simple response with just recall
            return {
                "experiences_found": True,
                "experience_count": 1,
                "recall_text": recall_result["recall_text"],
                "reflection_text": None,
                "combined_text": recall_result["recall_text"],
                "top_experience": experiences[0],
                "all_experiences": experiences,
                "confidence": 0.6  # Lower confidence with single experience
            }
    
    async def handle_experience_sharing_request(self,
                                             user_query: str,
                                             context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user request to share experiences.
        
        Args:
            user_query: User's query text
            context_data: Additional context data
            
        Returns:
            Experience sharing response
        """
        if not self.initialized:
            await self.initialize()
        
        # Prepare context
        context = context_data or {}
        context["query"] = user_query
        
        # Set default scenario type if not provided
        if "scenario_type" not in context:
            # Try to infer from query
            scenario_keywords = {
                "teasing": ["tease", "teasing", "playful", "joking"],
                "dark": ["dark", "intense", "scary", "fear"],
                "indulgent": ["indulge", "pleasure", "pamper", "spoil"],
                "psychological": ["mind", "psychological", "mental", "think"],
                "nurturing": ["nurture", "guide", "help", "support"],
                "training": ["train", "teach", "learn", "practice"],
                "discipline": ["discipline", "punish", "behave", "correct"],
                "service": ["serve", "service", "attend", "assist"],
                "worship": ["worship", "adore", "praise", "admire"],
                "punishment": ["punish", "penalty", "consequence", "retribution"]
            }
            
            # Check for keywords in query
            query_lower = user_query.lower()
            matched_scenarios = []
            
            for scenario, keywords in scenario_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    matched_scenarios.append(scenario)
            
            # Set first matched scenario or default to general
            context["scenario_type"] = matched_scenarios[0] if matched_scenarios else "general"
        
        # Determine min_relevance based on how explicit the request is
        explicit_request = any(phrase in user_query.lower() for phrase in 
                             ["remember", "recall", "tell me about", "have you done", 
                              "previous", "before", "past", "experience"])
        
        min_relevance = 0.5 if explicit_request else 0.7
        
        # Process the request
        result = await self.retrieve_and_share_experience(
            context=context,
            min_relevance=min_relevance
        )
        
        # Format response based on whether experiences were found
        if result["experiences_found"]:
            return {
                "success": True,
                "has_experience": True,
                "response_text": result["combined_text"],
                "experience_details": {
                    "count": result["experience_count"],
                    "top_experience": result["top_experience"],
                    "confidence": result["confidence"]
                }
            }
        else:
            # No experiences found
            return {
                "success": True,
                "has_experience": False,
                "response_text": "I don't have specific experiences to share about that yet."
            }
    
    async def store_experience(self,
                            memory_text: str,
                            scenario_type: str = "general",
                            entities: List[str] = None,
                            emotional_context: Dict[str, Any] = None,
                            significance: int = MemorySignificance.MEDIUM,
                            tags: List[str] = None) -> Dict[str, Any]:
        """
        Store a new experience in the memory system.
        
        Args:
            memory_text: The memory text
            scenario_type: Type of scenario
            entities: List of entity IDs involved
            emotional_context: Emotional context data
            significance: Memory significance
            tags: Additional tags
            
        Returns:
            Stored experience information
        """
        if not self.initialized:
            await self.initialize()
        
        # Set default tags if not provided
        tags = tags or []
        
        # Add scenario type to tags if not already present
        if scenario_type not in tags:
            tags.append(scenario_type)
        
        # Add experience tag
        if "experience" not in tags:
            tags.append("experience")
            
        # Prepare metadata
        metadata = {
            "scenario_type": scenario_type,
            "entities": entities or [],
            "is_experience": True
        }
        
        # Add emotional context to metadata if provided
        if emotional_context:
            metadata["emotions"] = emotional_context
        
        # Create memory using memory system
        memory_system = self.retriever.memory_system
        
        memory = Memory(
            text=memory_text,
            memory_type=MemoryType.OBSERVATION,
            significance=significance,
            emotional_intensity=emotional_context.get("primary", {}).get("intensity", 0.5) * 100 if emotional_context else 50,
            tags=tags,
            metadata=metadata,
            timestamp=datetime.now()
        )
        
        # Store memory
        memory_id = await memory_system.add_memory(memory)
        
        return {
            "memory_id": memory_id,
            "memory_text": memory_text,
            "scenario_type": scenario_type,
            "tags": tags,
            "significance": significance
        }
