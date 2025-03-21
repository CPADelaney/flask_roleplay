# nyx/core/experience_engine.py

import logging
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ExperienceEngine:
    """
    Unified experience retrieval and sharing system.
    Handles experience relevance scoring, abstraction, narrative generation,
    and conversational recall formatting.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
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
        
        # Confidence marker mapping for relevance scores
        self.confidence_markers = {
            (0.8, 1.0): "vividly recall",
            (0.6, 0.8): "clearly remember",
            (0.4, 0.6): "remember",
            (0.2, 0.4): "think I recall",
            (0.0, 0.2): "vaguely remember"
        }
        
        # Experiential richness factors
        self.richness_factors = {
            "detail_weight": 0.3,       # Text length and detail level
            "emotional_weight": 0.4,     # Emotional depth and intensity
            "sensory_weight": 0.2,       # Sensory details included
            "significance_weight": 0.1   # Memory significance
        }
        
        # Initialize experiential cache for performance
        self.experience_cache = {}
        self.narrative_cache = {}
        
        # LLM client would be initialized here in a real implementation
        self.llm_client = None
    
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
        # Extract key information from context
        query = current_context.get("query", "")
        scenario_type = current_context.get("scenario_type", "")
        emotional_state = current_context.get("emotional_state", {})
        entities = current_context.get("entities", [])
        
        # Check cache for identical request
        cache_key = f"{query}_{scenario_type}_{limit}_{min_relevance}"
        if cache_key in self.experience_cache:
            cache_time, cache_results = self.experience_cache[cache_key]
            # Use cache if less than 5 minutes old
            if (datetime.now() - cache_time).total_seconds() < 300:
                return cache_results
        
        # This would fetch candidate experiences from the memory system
        # For this implementation, we'll generate placeholder experiences
        # In a real implementation, this would call memory_core.retrieve_memories
        
        # Placeholder for experience retrieval
        experiences = await self._retrieve_candidate_experiences(query, scenario_type, entities)
        
        # Score experiences for relevance
        scored_experiences = []
        for experience in experiences:
            # Calculate relevance score
            relevance_score = await self._calculate_relevance_score(
                experience, current_context
            )
            
            # Skip low-relevance experiences
            if relevance_score < min_relevance:
                continue
            
            # Get emotional context for this experience
            emotional_context = await self._get_experience_emotional_context(experience)
            
            # Calculate experiential richness (how much detail/emotion it has)
            experiential_richness = self._calculate_experiential_richness(
                experience, emotional_context
            )
            
            # Add to scored experiences
            scored_experiences.append({
                "experience": experience,
                "relevance_score": relevance_score,
                "emotional_context": emotional_context,
                "experiential_richness": experiential_richness,
                "final_score": relevance_score * 0.7 + experiential_richness * 0.3
            })
        
        # Sort by final score
        scored_experiences.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Process top experiences
        results = []
        for item in scored_experiences[:limit]:
            experience = item["experience"]
            experience["relevance_score"] = item["relevance_score"]
            experience["emotional_context"] = item["emotional_context"]
            experience["experiential_richness"] = item["experiential_richness"]
            
            # Add confidence marker based on relevance
            experience["confidence_marker"] = self._get_confidence_marker(item["relevance_score"])
            
            results.append(experience)
        
        # Cache results
        self.experience_cache[cache_key] = (datetime.now(), results)
        
        return results
    
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
        content = experience.get("content", experience.get("memory_text", ""))
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        timestamp = experience.get("timestamp", experience.get("metadata", {}).get("timestamp"))
        
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
        
        # Generate summary components
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
        
        # For more complex reflections with multiple experiences, generate with summary components
        if len(experiences) > 1:
            return await self._generate_complex_reflection(experiences, context)
        
        # For simpler reflections based on a single experience
        scenario_type = top_experience.get("scenario_type", "general")
        emotional_context = top_experience.get("emotional_context", {})
        
        # Generate summary components for reflection
        components = await self._generate_summary_components(top_experience, context)
        reflection = components.get("reflection", "")
        
        # Determine confidence based on relevance and richness
        relevance = top_experience.get("relevance_score", 0.5)
        richness = top_experience.get("experiential_richness", 0.5)
        confidence = 0.5 + (relevance * 0.25) + (richness * 0.25)
        
        return {
            "reflection": reflection,
            "confidence": confidence,
            "experience_id": top_experience.get("id"),
            "scenario_type": scenario_type,
            "emotional_context": emotional_context
        }
    
    async def construct_narrative(self,
                               experiences: List[Dict[str, Any]],
                               topic: str,
                               chronological: bool = True) -> Dict[str, Any]:
        """
        Construct a coherent narrative from multiple experiences.
        
        Args:
            experiences: List of experiences to include in narrative
            topic: Topic of the narrative
            chronological: Whether to maintain chronological order
            
        Returns:
            Narrative data
        """
        if not experiences:
            return {
                "narrative": f"I don't have any significant experiences about {topic}.",
                "confidence": 0.2
            }
        
        # Sort chronologically if required
        if chronological:
            # Sort by timestamp if available
            sorted_experiences = sorted(
                experiences,
                key=lambda e: datetime.fromisoformat(
                    e.get("timestamp", e.get("metadata", {}).get("timestamp", datetime.now().isoformat()))
                    .replace("Z", "+00:00")
                )
            )
        else:
            # Otherwise sort by relevance
            sorted_experiences = sorted(
                experiences,
                key=lambda e: e.get("relevance_score", 0.5),
                reverse=True
            )
        
        # Extract experience data
        experience_texts = [e.get("content", e.get("memory_text", "")) for e in sorted_experiences]
        experience_ids = [e.get("id") for e in sorted_experiences]
        
        # Calculate base confidence from average relevance and richness
        avg_relevance = sum(e.get("relevance_score", 0.5) for e in experiences) / len(experiences)
        avg_richness = sum(e.get("experiential_richness", 0.5) for e in experiences) / len(experiences)
        base_confidence = (avg_relevance * 0.6) + (avg_richness * 0.4)
        
        # Generate narrative text
        # In a real implementation, this would use an LLM
        narrative_text = await self._generate_narrative_text(sorted_experiences, topic, base_confidence)
        
        return {
            "narrative": narrative_text,
            "experiences": experience_ids,
            "confidence": base_confidence,
            "chronological": chronological
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
        # Prepare context
        context = context_data or {}
        context["query"] = user_query
        
        # Set default scenario type if not provided
        if "scenario_type" not in context:
            # Try to infer from query
            scenario_keywords = {
                "teasing": ["tease", "teasing", "playful", "joke"],
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
        experiences = await self.retrieve_relevant_experiences(
            current_context=context,
            min_relevance=min_relevance
        )
        
        # Format response based on whether experiences were found
        if experiences:
            # Generate conversational recall for top experience
            recall_result = await self.generate_conversational_recall(
                experiences[0], context
            )
            
            # Generate personality-driven reflection if multiple experiences
            if len(experiences) > 1:
                reflection_result = await self.generate_personality_reflection(
                    experiences, context
                )
                
                # Combine recall and reflection for rich response
                combined_text = f"{recall_result['recall_text']} {reflection_result['reflection']}"
                
                return {
                    "success": True,
                    "has_experience": True,
                    "response_text": combined_text,
                    "recall_text": recall_result["recall_text"],
                    "reflection_text": reflection_result["reflection"],
                    "experience_count": len(experiences),
                    "top_experience": experiences[0],
                    "all_experiences": experiences,
                    "confidence": reflection_result["confidence"]
                }
            else:
                # Simple response with just recall
                return {
                    "success": True,
                    "has_experience": True,
                    "response_text": recall_result["recall_text"],
                    "recall_text": recall_result["recall_text"],
                    "reflection_text": None,
                    "experience_count": 1,
                    "top_experience": experiences[0],
                    "all_experiences": experiences,
                    "confidence": 0.6  # Lower confidence with single experience
                }
        else:
            # No experiences found
            return {
                "success": True,
                "has_experience": False,
                "response_text": "I don't have specific experiences to share about that yet."
            }
    
    def _get_confidence_marker(self, relevance: float) -> str:
        """Get confidence marker text based on relevance score"""
        for (min_val, max_val), marker in self.confidence_markers.items():
            if min_val <= relevance < max_val:
                return marker
        return "remember"  # Default
    
    def _get_timeframe_text(self, timestamp: Optional[str]) -> str:
        """Get conversational timeframe text from timestamp"""
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
        """Determine the emotional tone for recall based on the experience's emotions"""
        if not emotional_context:
            return "standard"
            
        primary = emotional_context.get("primary_emotion", "neutral")
        intensity = emotional_context.get("primary_intensity", 0.5)
        valence = emotional_context.get("valence", 0.0)
        
        # High intensity experiences
        if intensity > 0.8:
            return "intense"
            
        # Positive emotions
        if valence > 0.3 or primary in ["Joy", "Anticipation", "Trust", "Love"]:
            return "positive"
            
        # Negative emotions
        if valence < -0.3 or primary in ["Anger", "Fear", "Disgust", "Sadness", "Frustration"]:
            return "negative"
            
        # Default to standard
        return "standard"
    
    def _get_scenario_tone(self, scenario_type: str) -> Optional[str]:
        """Get tone based on scenario type"""
        scenario_type = scenario_type.lower()
        
        if scenario_type in ["teasing", "indulgent"]:
            return "teasing"
        elif scenario_type in ["discipline", "punishment", "training"]:
            return "disciplinary"
        elif scenario_type in ["dark", "fear"]:
            return "intense"
        
        # No specific tone for this scenario type
        return None
    
    def _calculate_experiential_richness(self, 
                                       experience: Dict[str, Any], 
                                       emotional_context: Dict[str, Any]) -> float:
        """
        Calculate how rich and detailed the experience is.
        Higher scores mean the experience has more emotional and sensory detail.
        """
        # Extract experience attributes
        experience_text = experience.get("content", experience.get("memory_text", ""))
        tags = experience.get("tags", [])
        significance = experience.get("significance", 5)
        
        # Initialize richness factors
        detail_score = 0.0
        emotional_depth = 0.0
        sensory_richness = 0.0
        significance_score = 0.0
        
        # Text length as a proxy for detail (longer texts might have more detail)
        word_count = len(experience_text.split())
        detail_score = min(1.0, word_count / 100)  # Cap at 100 words
        
        # Emotional depth from context
        if emotional_context:
            # Primary emotion intensity
            primary_intensity = emotional_context.get("primary_intensity", 0.0)
            
            # Count secondary emotions
            secondary_count = len(emotional_context.get("secondary_emotions", {}))
            
            # Combine for emotional depth
            emotional_depth = 0.7 * primary_intensity + 0.3 * min(1.0, secondary_count / 3)
        
        # Sensory richness via presence of sensory words
        sensory_words = ["see", "saw", "look", "hear", "heard", "sound", 
                        "feel", "felt", "touch", "smell", "scent", "taste"]
        
        sensory_count = sum(1 for word in sensory_words if word in experience_text.lower())
        sensory_richness = min(1.0, sensory_count / 5)  # Cap at 5 sensory words
        
        # Significance as a direct factor
        significance_score = significance / 10.0  # Convert to 0-1 scale
        
        # Combine scores with weights
        richness_score = (
            detail_score * self.richness_factors["detail_weight"] +
            emotional_depth * self.richness_factors["emotional_weight"] +
            sensory_richness * self.richness_factors["sensory_weight"] +
            significance_score * self.richness_factors["significance_weight"]
        )
        
        return min(1.0, max(0.0, richness_score))
    
    async def _retrieve_candidate_experiences(self, 
                                          query: str, 
                                          scenario_type: str, 
                                          entities: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve candidate experiences from memory system
        
        This is a placeholder for the actual retrieval from memory_core
        """
        # In a real implementation, this would call memory_core.retrieve_memories
        # with appropriate filters for experience-type memories
        
        # For demo purposes, return empty list
        # In actual implementation this would return real memories
        return []
    
    async def _calculate_relevance_score(self, 
                                      experience: Dict[str, Any], 
                                      context: Dict[str, Any]) -> float:
        """
        Calculate how relevant an experience is to the current context.
        
        Args:
            experience: Experience to evaluate
            context: Current conversation context
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Extract experience attributes
        experience_text = experience.get("content", experience.get("memory_text", ""))
        experience_tags = experience.get("tags", [])
        experience_metadata = experience.get("metadata", {})
        
        # Extract context attributes
        query = context.get("query", "")
        scenario_type = context.get("scenario_type", "")
        emotional_state = context.get("emotional_state", {})
        entities = context.get("entities", [])
        
        # Initialize score components
        semantic_score = 0.0
        tag_score = 0.0
        temporal_score = 0.0
        emotional_score = 0.0
        entity_score = 0.0
        
        # If experience already has a relevance score, use it as semantic score
        if "relevance" in experience:
            semantic_score = experience.get("relevance", 0.0)
        # Otherwise calculate semantic relevance (normally via embedding similarity)
        else:
            # This would normally use embedding similarity
            # For demo, use simple keyword matching
            query_words = set(query.lower().split())
            text_words = set(experience_text.lower().split())
            if query_words and text_words:
                matching_words = query_words.intersection(text_words)
                semantic_score = len(matching_words) / len(query_words) if query_words else 0.0
            else:
                semantic_score = 0.0
        
        # Tag matching
        context_tags = context.get("tags", [])
        
        if scenario_type and experience_tags:
            # Check if scenario type is in tags
            scenario_match = any(tag.lower() == scenario_type.lower() for tag in experience_tags)
            if scenario_match:
                tag_score += 0.5
            
            # Count other matching tags
            matching_tags = sum(1 for tag in experience_tags if tag in context_tags)
            tag_score += min(0.5, matching_tags / max(1, len(context_tags)))
        
        # Temporal relevance (newer experiences score higher)
        if "timestamp" in experience or "timestamp" in experience.get("metadata", {}):
            timestamp_str = experience.get("timestamp", experience.get("metadata", {}).get("timestamp"))
            if timestamp_str:
                experience_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                age_in_days = (datetime.now() - experience_time).total_seconds() / 86400
                # Newer experiences get higher scores, but not too dominant
                temporal_score = max(0.0, 1.0 - (age_in_days / 180))  # 6 month scale
        
        # Emotional relevance
        experience_emotions = experience.get("emotional_context", {})
        
        if experience_emotions and emotional_state:
            # Compare primary emotions
            experience_primary = experience_emotions.get("primary_emotion", "neutral")
            context_primary = emotional_state.get("primary_emotion", "neutral")
            
            # Emotion match bonus
            if experience_primary == context_primary:
                emotional_score += 0.5
                
            # Valence comparison (positive/negative alignment)
            experience_valence = experience_emotions.get("valence", 0.0)
            context_valence = emotional_state.get("valence", 0.0)
            
            # Similar valence bonus
            emotional_score += 0.5 * (1.0 - min(1.0, abs(experience_valence - context_valence)))
        
        # Entity relevance
        experience_entities = experience.get("entities", experience_metadata.get("entities", []))
        
        if experience_entities and entities:
            matching_entities = len(set(experience_entities).intersection(set(entities)))
            entity_score = min(1.0, matching_entities / max(1, len(entities)))
        
        # Combine scores with weights
        final_score = (
            semantic_score * 0.35 +
            tag_score * 0.20 +
            temporal_score * 0.10 +
            emotional_score * 0.20 +
            entity_score * 0.15
        )
        
        return min(1.0, max(0.0, final_score))
    
    async def _get_experience_emotional_context(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or infer emotional context for an experience.
        
        Args:
            experience: Experience to analyze
            
        Returns:
            Emotional context information
        """
        # Check if experience already has emotional context
        if "emotional_context" in experience:
            return experience["emotional_context"]
        
        # Check metadata for emotional context
        metadata = experience.get("metadata", {})
        if "emotional_context" in metadata:
            return metadata["emotional_context"]
        
        # If no emotional context information is available, infer a basic one
        return {
            "primary_emotion": "neutral",
            "primary_intensity": 0.5,
            "secondary_emotions": {},
            "valence": 0.0,
            "arousal": 0.3
        }
    
    async def _generate_summary_components(self,
                                        experience: Dict[str, Any],
                                        context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate summary components for experience recall.
        
        Args:
            experience: The experience to summarize
            context: Current conversation context
            
        Returns:
            Dictionary with summary components
        """
        # Extract experience data
        content = experience.get("content", experience.get("memory_text", ""))
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        
        # Get emotional information for context
        primary_emotion = emotional_context.get("primary_emotion", "neutral")
        emotional_intensity = emotional_context.get("primary_intensity", 0.5)
        
        # In a real implementation, this would use an LLM
        # For this implementation, we'll generate placeholder components
        
        # Extract a brief summary (first sentence or phrase)
        sentences = content.split(".")
        brief_summary = sentences[0] if sentences else "something happened"
        if len(brief_summary) > 50:
            brief_summary = brief_summary[:47] + "..."
        
        # Extract a detail from later in the content
        detail = ""
        if len(sentences) > 1:
            detail = sentences[1]
            if len(detail) > 60:
                detail = detail[:57] + "..."
        else:
            detail = "It was quite memorable."
        
        # Generate a reflection based on scenario type and emotion
        reflection = self._generate_fallback_reflection(emotional_context, scenario_type)
        
        return {
            "brief_summary": brief_summary,
            "detail": detail,
            "reflection": reflection
        }
    
    def _generate_fallback_reflection(self, 
                                   emotional_context: Dict[str, Any],
                                   scenario_type: str) -> str:
        """Generate a fallback reflection based on emotion and scenario"""
        # Get primary emotion
        primary_emotion = emotional_context.get("primary_emotion", "neutral")
        
        # Map emotions to reflections
        emotion_reflections = {
            "Joy": "I found it quite satisfying",
            "Sadness": "It was somewhat disappointing",
            "Anger": "It was rather frustrating",
            "Fear": "It was rather unnerving",
            "Disgust": "It was rather distasteful",
            "Surprise": "It was quite unexpected",
            "Anticipation": "I found myself looking forward to the outcome",
            "Trust": "It reinforced my expectations",
            "Love": "I felt a strong connection during that moment",
            "Frustration": "It left me somewhat dissatisfied",
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
        if scenario_type.lower() in scenario_reflections:
            return scenario_reflections[scenario_type.lower()]
        
        # Otherwise use emotion-based reflection
        return emotion_reflections.get(primary_emotion, "It was an interesting experience")
    
    async def _generate_complex_reflection(self,
                                        experiences: List[Dict[str, Any]],
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a more complex reflection based on multiple experiences.
        
        Args:
            experiences: List of experiences to reflect on
            context: Current conversation context
            
        Returns:
            Complex reflection
        """
        # Extract key data from experiences
        primary_emotions = [exp.get("emotional_context", {}).get("primary_emotion", "neutral") for exp in experiences]
        scenario_types = [exp.get("scenario_type", "general") for exp in experiences]
        
        # Determine dominant emotion and scenario type
        dominant_emotion = max(set(primary_emotions), key=primary_emotions.count) if primary_emotions else "neutral"
        dominant_scenario = max(set(scenario_types), key=scenario_types.count) if scenario_types else "general"
        
        # In a real implementation, this would use an LLM
        # For this implementation, we'll use templates
        
        reflection_templates = [
            "I've noticed a pattern in these situations: {pattern}. It's something I find {reaction}.",
            "There's a common thread in these experiences: {pattern}. I find it {reaction} when this happens.",
            "I've observed that {pattern} tends to occur in these scenarios. It's typically {reaction}."
        ]
        
        pattern_templates = {
            "teasing": [
                "people respond most intensely when teased about their {aspect}",
                "teasing works best when it's balanced with {aspect}",
                "the right amount of {aspect} makes teasing much more effective"
            ],
            "discipline": [
                "clear consequences for {aspect} lead to better results",
                "consistency in handling {aspect} is crucial",
                "addressing {aspect} promptly prevents escalation"
            ],
            "service": [
                "acknowledging good {aspect} reinforces desired behavior",
                "setting clear expectations for {aspect} leads to better service",
                "recognizing effort in {aspect} encourages continued dedication"
            ],
            "general": [
                "paying attention to {aspect} yields better results",
                "being mindful of {aspect} creates better experiences",
                "acknowledging {aspect} leads to more satisfying interactions"
            ]
        }
        
        reaction_templates = {
            "Joy": ["quite satisfying", "delightful", "enjoyable"],
            "Sadness": ["somewhat disappointing", "melancholic", "bittersweet"],
            "Anger": ["rather frustrating", "irritating", "vexing"],
            "Fear": ["rather unnerving", "concerning", "unsettling"],
            "Disgust": ["rather distasteful", "off-putting", "unpleasant"],
            "Surprise": ["quite unexpected", "surprising", "remarkable"],
            "Anticipation": ["exciting", "promising", "intriguing"],
            "Trust": ["reassuring", "comforting", "validating"],
            "Love": ["deeply fulfilling", "heartwarming", "touching"],
            "Frustration": ["somewhat challenging", "testing", "trying"],
            "neutral": ["interesting", "noteworthy", "thought-provoking"]
        }
        
        # Select templates
        reflection_template = random.choice(reflection_templates)
        scenario_key = dominant_scenario if dominant_scenario in pattern_templates else "general"
        pattern_template = random.choice(pattern_templates[scenario_key])
        reaction_key = dominant_emotion if dominant_emotion in reaction_templates else "neutral"
        reaction = random.choice(reaction_templates[reaction_key])
        
        # Generate aspect
        aspects = ["vulnerabilities", "desires", "insecurities", "aspirations", 
                  "behavior", "reactions", "responses", "boundaries", 
                  "preferences", "tendencies", "hesitations", "enthusiasms"]
        aspect = random.choice(aspects)
        
        # Fill in templates
        pattern = pattern_template.format(aspect=aspect)
        reflection_text = reflection_template.format(pattern=pattern, reaction=reaction)
        
        # Calculate confidence based on experience count and relevance
        relevance_scores = [exp.get("relevance_score", 0.5) for exp in experiences]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        
        # More relevant experiences = higher confidence
        confidence = 0.5 + (avg_relevance * 0.3) + (min(len(experiences), 3) * 0.1 / 3)
        
        return {
            "reflection": reflection_text,
            "confidence": min(1.0, confidence),
            "experience_count": len(experiences),
            "experience_ids": [exp.get("id") for exp in experiences],
            "dominant_emotion": dominant_emotion,
            "dominant_scenario": dominant_scenario
        }
    
    async def _generate_narrative_text(self,
                                    experiences: List[Dict[str, Any]],
                                    topic: str,
                                    confidence: float) -> str:
        """
        Generate narrative text from experiences
        
        This would typically use an LLM in a real implementation
        """
        # In a real implementation, this would use an LLM
        # For this implementation, we'll use a template approach
        
        # Extract experience texts
        experience_texts = [exp.get("content", exp.get("memory_text", "")) for exp in experiences]
        
        # Get confidence marker
        confidence_marker = self._get_confidence_marker(confidence)
        
        # Generate a simple narrative
        if len(experiences) == 1:
            return f"I {confidence_marker} {experience_texts[0]}"
        else:
            # Combine first two experiences for brevity
            narrative = f"I {confidence_marker} a series of experiences about {topic}. "
            narrative += f"First, {experience_texts[0]} "
            narrative += f"Later, {experience_texts[1]}"
            
            if len(experiences) > 2:
                narrative += f" There were {len(experiences)-2} more related experiences after these."
            
            return narrative
    # Add these methods to the ExperienceEngine class
    
    async def retrieve_experiences_enhanced(self, query: str, scenario_type: Optional[str] = None, 
                                           limit: int = 3) -> List[Dict[str, Any]]:
        """
        Enhanced method for retrieving experiences with improved formatting.
        
        Args:
            query: Search query
            scenario_type: Optional scenario type
            limit: Maximum number of results
        
        Returns:
            List of formatted experiences
        """
        # Prepare context for experience retrieval
        context = {
            "query": query,
            "scenario_type": scenario_type
        }
        
        # Retrieve experiences
        experiences = await self.retrieve_relevant_experiences(
            current_context=context,
            limit=limit
        )
        
        # Format experiences for return
        formatted_experiences = []
        for exp in experiences:
            formatted = {
                "content": exp.get("content", ""),
                "scenario_type": exp.get("scenario_type", ""),
                "confidence_marker": exp.get("confidence_marker", ""),
                "relevance_score": exp.get("relevance_score", 0.5),
                "experiential_richness": exp.get("experiential_richness", 0.5)
            }
            formatted_experiences.append(formatted)
        
        return formatted_experiences
    
    async def share_experience_enhanced(self, query: str, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced method for sharing experiences with better context handling.
        
        Args:
            query: User's query text
            context_data: Additional context data
        
        Returns:
            Formatted experience sharing result
        """
        result = await self.handle_experience_sharing_request(
            user_query=query,
            context_data=context_data
        )
        
        return {
            "success": result["success"],
            "has_experience": result.get("has_experience", False),
            "response_text": result.get("response_text", ""),
            "experience_count": result.get("experience_count", 0)
        }
