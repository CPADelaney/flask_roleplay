# nyx/core/brain/processing/base_processor.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional

from agents import trace

logger = logging.getLogger(__name__)

class BaseProcessor:
    def __init__(self, brain):
        self.brain = brain
        self._initialized = False
    
    async def initialize(self):
        """Initialize processor after brain is fully initialized"""
        self._initialized = True
        logger.info(f"{self.__class__.__name__} initialized")
    
    async def _ensure_initialized(self):
        """Ensure processor is initialized before use"""
        if not self._initialized:
            await self.initialize()
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_input")
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_response")
    
    async def _process_emotional_impact(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotional impact of user input"""
        # Process emotional impact
        if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
            emotional_stimuli = self.brain.emotional_core.analyze_text_sentiment(user_input)
            emotional_state = self.brain.emotional_core.update_from_stimuli(emotional_stimuli)
            
            # Update performance counter
            if hasattr(self.brain, "performance_metrics"):
                self.brain.performance_metrics["emotion_updates"] = self.brain.performance_metrics.get("emotion_updates", 0) + 1
            
            return {
                "emotional_state": emotional_state,
                "stimuli": emotional_stimuli
            }
        
        return {"emotional_state": {}, "stimuli": {}}
    
    async def _retrieve_memories_with_emotion(self, 
                                          user_input: str, 
                                          context: Dict[str, Any],
                                          emotional_state: Dict[str, float]) -> List[Dict[str, Any]]:
        """Retrieve relevant memories with emotional influence"""
        if not hasattr(self.brain, "memory_orchestrator") or not self.brain.memory_orchestrator:
            return []
        
        # Create emotional prioritization for memory types
        # Based on current emotional state, prioritize different memory types
        
        # Check if emotional valence is available
        valence = 0
        arousal = 0.5
        if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
            valence = self.brain.emotional_core.get_emotional_valence()
            arousal = self.brain.emotional_core.get_emotional_arousal()
        
        # Prioritize experiences and reflections for high emotional states
        if abs(valence) > 0.6 or arousal > 0.7:
            prioritization = {
                "experience": 0.5,
                "reflection": 0.3,
                "abstraction": 0.1,
                "observation": 0.1
            }
        # Prioritize abstractions and reflections for low emotional states
        elif arousal < 0.3:
            prioritization = {
                "abstraction": 0.4,
                "reflection": 0.3,
                "experience": 0.2,
                "observation": 0.1
            }
        # Balanced prioritization for moderate emotional states
        else:
            prioritization = {
                "experience": 0.3,
                "reflection": 0.3,
                "abstraction": 0.2,
                "observation": 0.2
            }
        
        # Adjust prioritization based on emotion-to-memory influence
        influence = getattr(self.brain, "emotion_to_memory_influence", 0.4)
        for memory_type, priority in prioritization.items():
            prioritization[memory_type] = priority * (1 + influence)
        
        # Use prioritized retrieval if available
        if hasattr(self.brain.memory_orchestrator, "retrieve_memories_with_prioritization"):
            memories = await self.brain.memory_orchestrator.retrieve_memories_with_prioritization(
                query=user_input,
                memory_types=context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]),
                prioritization=prioritization,
                limit=context.get("memory_limit", 5)
            )
        else:
            # Fallback to regular retrieval
            memories = await self.brain.memory_orchestrator.retrieve_memories(
                query=user_input,
                memory_types=context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]), 
                limit=context.get("memory_limit", 5)
            )
        
        # Update performance counter
        if hasattr(self.brain, "performance_metrics"):
            self.brain.performance_metrics["memory_operations"] = self.brain.performance_metrics.get("memory_operations", 0) + 1
        
        return memories
    
    async def _share_experience(self, 
                            user_input: str, 
                            context: Dict[str, Any], 
                            emotional_state: Dict[str, float]) -> Dict[str, Any]:
        """Share experience based on user input"""
        if not hasattr(self.brain, "experience_interface") or not self.brain.experience_interface:
            return {"has_experience": False}
        
        # Enhanced experience sharing with cross-user support and adaptation
        cross_user_enabled = getattr(self.brain, "cross_user_enabled", True)
        cross_user_sharing_threshold = getattr(self.brain, "cross_user_sharing_threshold", 0.7)
        
        experience_result = await self.brain.experience_interface.share_experience_enhanced(
            query=user_input,
            context_data={
                "user_id": str(self.brain.user_id),
                "emotional_state": emotional_state,
                "include_cross_user": cross_user_enabled and context.get("include_cross_user", True),
                "cross_user_threshold": cross_user_sharing_threshold,
                "scenario_type": context.get("scenario_type", ""),
                "conversation_id": self.brain.conversation_id
            }
        )
        
        # Update performance metrics
        if hasattr(self.brain, "performance_metrics") and experience_result.get("has_experience", False):
            self.brain.performance_metrics["experiences_shared"] = self.brain.performance_metrics.get("experiences_shared", 0) + 1
            
            # Track cross-user experiences
            if experience_result.get("cross_user", False):
                self.brain.performance_metrics["cross_user_experiences_shared"] = self.brain.performance_metrics.get("cross_user_experiences_shared", 0) + 1
        
        return experience_result
    
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
    
    async def _generate_emotional_expression(self, emotional_state: Dict[str, float]) -> Dict[str, Any]:
        """Generate emotional expression based on emotional state"""
        # Determine if emotion should be expressed
        if not hasattr(self.brain, "emotional_core") or not self.brain.emotional_core:
            return {"expression": None, "should_express": False}
        
        should_express_emotion = False
        if hasattr(self.brain.emotional_core, "should_express_emotion"):
            should_express_emotion = self.brain.emotional_core.should_express_emotion()
        
        emotional_expression = None
        
        if should_express_emotion:
            try:
                if hasattr(self.brain.emotional_core, "generate_emotional_expression"):
                    expression_result = await self.brain.emotional_core.generate_emotional_expression(force=False)
                    if expression_result.get("expressed", False):
                        emotional_expression = expression_result.get("expression", "")
                elif hasattr(self.brain.emotional_core, "get_expression_for_emotion"):
                    emotional_expression = self.brain.emotional_core.get_expression_for_emotion()
            except Exception as e:
                logger.error(f"Error generating emotional expression: {str(e)}")
                if hasattr(self.brain.emotional_core, "get_expression_for_emotion"):
                    emotional_expression = self.brain.emotional_core.get_expression_for_emotion()
        
        return {
            "expression": emotional_expression,
            "should_express": should_express_emotion
        }
    
    async def _generate_reasoning_response(self, user_input: str) -> str:
        """Generate a response using reasoning capabilities"""
        try:
            if hasattr(self.brain, "reasoning_triage_agent") and hasattr(self.brain, "Runner"):
                reasoning_result = await self.brain.Runner.run(
                    self.brain.reasoning_triage_agent,
                    user_input
                )
                return reasoning_result.final_output if hasattr(reasoning_result, "final_output") else str(reasoning_result)
            else:
                return "I understand your question and would like to reason through it with you."
        except Exception as e:
            logger.error(f"Error in reasoning response: {str(e)}")
            return "I understand your question and would like to reason through it with you."
    
    async def _determine_main_response(self, 
                                   user_input: str, 
                                   processing_result: Dict[str, Any],
                                   context: Dict[str, Any]) -> Dict[str, str]:
        """Determine the main response content based on processing results"""
        # Determine if experience response should be used
        if processing_result["has_experience"]:
            main_response = processing_result["experience_response"]
            response_type = "experience"
            
            # If it's a cross-user experience, mark it
            if processing_result.get("cross_user_experience", False):
                response_type = "cross_user_experience"
        else:
            # Check if procedural knowledge can be used
            procedural_knowledge = processing_result.get("procedural_knowledge", None)
            if procedural_knowledge and procedural_knowledge.get("can_execute", False) and hasattr(self.brain, "agent_enhanced_memory"):
                try:
                    # Get the most relevant procedure
                    top_procedure = procedural_knowledge["relevant_procedures"][0]
                    
                    # Execute the procedure
                    procedure_result = await self.brain.agent_enhanced_memory.execute_procedure(
                        top_procedure["name"],
                        context={"user_input": user_input, **(context or {})}
                    )
                    
                    # If successful, use the procedure's response
                    if procedure_result.get("success", False) and "output" in procedure_result:
                        main_response = procedure_result["output"]
                        response_type = "procedural"
                    else:
                        # For reasoning-related queries, use the reasoning agents
                        if self._is_reasoning_query(user_input):
                            main_response = await self._generate_reasoning_response(user_input)
                            response_type = "reasoning"
                        else:
                            # No specific type, standard response
                            main_response = "I understand your input and have processed it."
                            response_type = "standard"
                except Exception as e:
                    logger.error(f"Error executing procedure: {str(e)}")
                    
                    # For reasoning-related queries, use the reasoning agents
                    if self._is_reasoning_query(user_input):
                        main_response = await self._generate_reasoning_response(user_input)
                        response_type = "reasoning"
                    else:
                        # No specific type, standard response
                        main_response = "I understand your input and have processed it."
                        response_type = "standard"
            else:
                # For reasoning-related queries, use the reasoning agents
                if self._is_reasoning_query(user_input):
                    main_response = await self._generate_reasoning_response(user_input)
                    response_type = "reasoning"
                else:
                    # No specific type, standard response
                    main_response = "I understand your input and have processed it."
                    response_type = "standard"
        
        return {
            "message": main_response,
            "response_type": response_type
        }
    
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
