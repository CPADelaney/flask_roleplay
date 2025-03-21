# nyx/core/experience_interface.py

import asyncio
import datetime
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ExperienceInterface:
    """
    Interface for experience retrieval and narrative generation that works with MemoryCore.
    Provides experience-specific formatting and interaction patterns.
    """
    
    def __init__(self, memory_core):
        self.memory_core = memory_core
        
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
        
        # Retrieve experiences using the memory core
        memories = await self.memory_core.retrieve_memories(
            query=query,
            memory_types=["experience"],
            limit=limit,
            context=context
        )
        
        # Format experiences for return
        formatted_experiences = []
        for exp in memories:
            formatted = {
                "content": exp.get("memory_text", ""),
                "scenario_type": exp.get("tags", ["general"])[0] if exp.get("tags") else "general",
                "confidence_marker": exp.get("confidence_marker", ""),
                "relevance_score": exp.get("relevance", 0.5),
                "experiential_richness": min(1.0, exp.get("significance", 5) / 10.0)
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
        
        # Process the request by retrieving experiences via Memory Core
        experiences = await self.memory_core.retrieve_relevant_experiences(
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
        # Get the recall text directly from Memory Core
        recall_result = await self.memory_core.generate_conversational_recall(experience, context)
        return recall_result
    
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
        
        # For more complex reflections with multiple experiences
        if len(experiences) > 1:
            reflection = await self._generate_complex_reflection(experiences, context)
            return reflection
        
        # For simpler reflections based on a single experience
        scenario_type = top_experience.get("scenario_type", "general")
        emotional_context = top_experience.get("emotional_context", {})
        
        # Generate simple reflection
        reflection = "Reflecting on this experience, I've noticed patterns in how I approach similar situations."
        
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
        
        # Generate a reflection about patterns across experiences
        reflection_text = "I've noticed an interesting pattern across these experiences that highlights my approach to similar situations."
        
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
        return await self.memory_core.construct_narrative_from_memories(
            topic=topic,
            chronological=chronological,
            limit=len(experiences)
        )
