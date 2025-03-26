# nyx/core/brain/processing/base_processor.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional

from agents import trace

logger = logging.getLogger(__name__)

class BaseProcessor:
    """Base processor that defines the common interface and shared functionality"""
    
    def __init__(self, brain):
        self.brain = brain
    
    async def initialize(self):
        """Initialize the processor - implemented by subclasses"""
        logger.info(f"{self.__class__.__name__} initialized")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_input")
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_response")
    
    async def _calculate_memory_emotional_impact(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate emotional impact from relevant memories"""
        impact = {}
        
        for memory in memories:
            # Extract emotional context
            emotional_context = memory.get("metadata", {}).get("emotional_context", {})
            
            if not emotional_context:
                continue
                
            # Get primary emotion
            primary_emotion = emotional_context.get("primary_emotion")
            primary_intensity = emotional_context.get("primary_intensity", 0.5)
            
            if primary_emotion:
                # Calculate impact based on relevance and recency
                relevance = memory.get("relevance", 0.5)
                
                # Get timestamp if available
                timestamp_str = memory.get("metadata", {}).get("timestamp")
                recency_factor = 1.0
                if timestamp_str:
                    try:
                        timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        days_old = (datetime.datetime.now() - timestamp).days
                        recency_factor = max(0.5, 1.0 - (days_old / 30))  # Decay over 30 days, minimum 0.5
                    except (ValueError, TypeError):
                        # If timestamp can't be parsed, use default
                        pass
                
                # Calculate final impact value
                impact_value = primary_intensity * relevance * recency_factor * 0.1
                
                # Add to impact dict
                if primary_emotion not in impact:
                    impact[primary_emotion] = 0
                impact[primary_emotion] += impact_value
        
        return impact
    
    def _should_share_experience(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Determine if experience sharing is appropriate"""
        # Check for explicit experience requests
        explicit_request = any(phrase in user_input.lower() for phrase in 
                             ["remember", "recall", "tell me about", "have you done", 
                              "previous", "before", "past", "experience", "what happened",
                              "have you ever", "did you ever", "similar", "others"])
        
        if explicit_request:
            return True
        
        # Check if it's a question that could benefit from experience sharing
        is_question = user_input.endswith("?") or user_input.lower().startswith(("what", "how", "when", "where", "why", "who", "can", "could", "do", "did"))
        
        if is_question and context and "share_experiences" in context and context["share_experiences"]:
            return True
        
        # Check for personal references that might trigger experience sharing
        personal_references = any(phrase in user_input.lower() for phrase in 
                               ["your experience", "you like", "you prefer", "you enjoy",
                                "you think", "you feel", "your opinion", "your view"])
        
        if personal_references:
            return True
            
        # Get user preference for experience sharing if available
        if hasattr(self.brain, "user_id") and hasattr(self.brain, "experience_interface"):
            user_id = str(self.brain.user_id)
            if hasattr(self.brain.experience_interface, "_get_user_preference_profile"):
                try:
                    profile = self.brain.experience_interface._get_user_preference_profile(user_id)
                    sharing_preference = profile.get("experience_sharing_preference", 0.5)
                    
                    # Higher preference means more likely to share experiences
                    import random
                    random_factor = random.random()
                    if random_factor < sharing_preference * 0.5:
                        return True
                except Exception:
                    pass
        
        # Default to not sharing experiences unless explicitly requested
        return False
        
    def _is_reasoning_query(self, user_input: str) -> bool:
        """Determine if a query is likely to need reasoning capabilities"""
        reasoning_indicators = [
            "why", "how come", "explain", "what if", "cause", "reason", "logic", 
            "analyze", "understand", "think through", "consider", "would happen",
            "hypothetical", "scenario", "reasoning", "connect", "relationship",
            "causality", "counterfactual", "consequence", "impact", "effect"
        ]
        
        return any(indicator in user_input.lower() for indicator in reasoning_indicators)
        
    async def _handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during processing"""
        logger.error(f"Error in {self.__class__.__name__}: {str(error)}")
        
        # Register with issue tracker if available
        if hasattr(self.brain, "issue_tracker"):
            issue_data = {
                "title": f"Error in {self.__class__.__name__}",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "component": self.__class__.__name__,
                "category": "PROCESSING",
                "severity": "MEDIUM",
                "context": context
            }
            return await self.brain.issue_tracker.register_issue(issue_data)
        
        return {"registered": False, "error": str(error)}
