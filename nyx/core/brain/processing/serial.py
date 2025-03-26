# nyx/core/brain/processing/serial.py
import logging
import asyncio
import datetime
import random
from typing import Dict, List, Any, Optional

from agents import trace, Runner
from nyx.core.brain.models import SensoryInput

logger = logging.getLogger(__name__)

class SerialProcessor:
    """Handles serial processing of inputs"""
    
    def __init__(self, brain):
        self.brain = brain
    
    async def initialize(self):
        """Initialize the processor"""
        logger.info("Serial processor initialized")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input using serial processing path
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results
        """
        if not self.brain.initialized:
            await self.brain.initialize()
        
        with trace(workflow_name="process_input_serial", group_id=self.brain.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Process temporal effects if available
            temporal_effects = None
            if hasattr(self.brain, "temporal_perception"):
                temporal_effects = await self.brain.temporal_perception.on_interaction_start()
            
            # Initialize context if needed
            context = context or {}
            
            # Add temporal context if available
            if temporal_effects:
                context["temporal_context"] = temporal_effects
            
            # Add user_id to context if not present
            if "user_id" not in context:
                context["user_id"] = str(self.brain.user_id)
            
            # Create sensory input if multimodal processing is available
            if hasattr(self.brain, "multimodal_integrator"):
                sensory_input = SensoryInput(
                    modality="text",
                    data=user_input,
                    confidence=1.0,
                    timestamp=datetime.datetime.now().isoformat(),
                    metadata=context
                )
                
                # Update attention with this new input
                if hasattr(self.brain, "attentional_controller"):
                    salient_items = [{
                        "target": "text_input",
                        "novelty": 0.8,  # Assume new input is novel
                        "intensity": min(1.0, len(user_input) / 500),  # Longer inputs have higher intensity
                        "emotional_impact": 0.5,  # Default moderate emotional impact
                        "goal_relevance": 0.7  # Assume user input is relevant to goals
                    }]
                    
                    await self.brain.attentional_controller.update_attention(salient_items=salient_items)
                
                # Process through multimodal integrator
                percept = await self.brain.multimodal_integrator.process_sensory_input(sensory_input)
                
                # Update reasoning with the integrated percept if necessary
                if hasattr(self.brain, "reasoning_core") and percept.attention_weight > 0.5:
                    await self.brain.reasoning_core.update_with_perception(percept)
            
            # Update hormone system environmental factors if available
            if hasattr(self.brain, "hormone_system") and self.brain.hormone_system:
                # Update time of day
                current_hour = datetime.datetime.now().hour
                self.brain.hormone_system.environmental_factors["time_of_day"] = (current_hour % 24) / 24.0
                
                # Update session duration if last_interaction is available
                if hasattr(self.brain, "last_interaction"):
                    time_in_session = (datetime.datetime.now() - self.brain.last_interaction).total_seconds() / 3600  # hours
                    self.brain.hormone_system.environmental_factors["session_duration"] = min(1.0, time_in_session / 8.0)
                
                # Update user familiarity
                if hasattr(self.brain, "interaction_count"):
                    self.brain.hormone_system.environmental_factors["user_familiarity"] = min(1.0, self.brain.interaction_count / 100)
            
            # Run meta-cognitive cycle if available
            meta_result = {}
            if hasattr(self.brain, "meta_core") and self.brain.meta_core:
                try:
                    meta_result = await self.brain.meta_core.cognitive_cycle(context)
                except Exception as e:
                    logger.error(f"Error in meta-cognitive cycle: {str(e)}")
                    meta_result = {"error": str(e)}
            
            # Process emotional impact of input
            if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
                emotional_stimuli = self.brain.emotional_core.analyze_text_sentiment(user_input)
                emotional_state = self.brain.emotional_core.update_from_stimuli(emotional_stimuli)
                
                # Add to performance metrics if they exist
                if hasattr(self.brain, "performance_metrics"):
                    self.brain.performance_metrics["emotion_updates"] = self.brain.performance_metrics.get("emotion_updates", 0) + 1
                
                # Add emotional state to context for memory retrieval
                context["emotional_state"] = emotional_state
            else:
                emotional_state = {}
            
            # Check for procedural knowledge
            procedural_knowledge = None
            if hasattr(self.brain, "agent_enhanced_memory"):
                try:
                    # Find relevant procedures for this input
                    relevant_procedures = await self.brain.agent_enhanced_memory.find_similar_procedures(user_input)
                    if relevant_procedures:
                        procedural_knowledge = {
                            "relevant_procedures": relevant_procedures,
                            "can_execute": len(relevant_procedures) > 0
                        }
                except Exception as e:
                    logger.error(f"Error checking procedural knowledge: {str(e)}")
            
            # Update hormone system interaction quality based on emotional valence if available
            if hasattr(self.brain, "hormone_system") and hasattr(self.brain, "emotional_core"):
                valence = self.brain.emotional_core.get_emotional_valence()
                interaction_quality = (valence + 1.0) / 2.0  # Convert from -1:1 to 0:1 range
                self.brain.hormone_system.environmental_factors["interaction_quality"] = interaction_quality
            
            # Retrieve relevant memories using memory orchestrator
            memories = []
            if hasattr(self.brain, "memory_orchestrator"):
                memories = await self.brain.memory_orchestrator.retrieve_memories(
                    query=user_input,
                    memory_types=context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]), 
                    limit=context.get("memory_limit", 5)
                )
                
                # Add to performance metrics if they exist
                if hasattr(self.brain, "performance_metrics"):
                    self.brain.performance_metrics["memory_operations"] = self.brain.performance_metrics.get("memory_operations", 0) + 1
            
            # Update emotional state based on retrieved memories
            if memories and hasattr(self.brain, "emotional_core"):
                # Calculate memory emotional impact
                memory_emotional_impact = await self._calculate_memory_emotional_impact(memories)
                
                # Apply memory-to-emotion influence
                for emotion, value in memory_emotional_impact.items():
                    influence = getattr(self.brain, "memory_to_emotion_influence", 0.3)
                    self.brain.emotional_core.update_emotion(emotion, value * influence)
                
                # Get updated emotional state
                emotional_state = self.brain.emotional_core.get_emotional_state()
            
            # Check if experience sharing is appropriate
            should_share_experience = self._should_share_experience(user_input, context)
            experience_result = None
            identity_impact = None
            
            if should_share_experience and hasattr(self.brain, "experience_interface"):
                # Enhanced experience sharing with cross-user support
                cross_user_enabled = getattr(self.brain, "cross_user_enabled", False)
                experience_result = await self.brain.experience_interface.share_experience_enhanced(
                    query=user_input,
                    context_data={
                        "user_id": str(self.brain.user_id),
                        "emotional_state": emotional_state,
                        "include_cross_user": cross_user_enabled and context.get("include_cross_user", True),
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
                
                # Calculate identity impact if experience was found and identity evolution exists
                if experience_result.get("has_experience", False) and hasattr(self.brain, "identity_evolution"):
                    experience = experience_result.get("experience", {})
                    if experience:
                        try:
                            # Calculate impact on identity
                            identity_impact = await self.brain.identity_evolution.calculate_experience_impact(experience)
                            
                            # Update identity based on experience
                            await self.brain.identity_evolution.update_identity_from_experience(
                                experience=experience,
                                impact=identity_impact
                            )
                        except Exception as e:
                            logger.error(f"Error updating identity from experience: {str(e)}")
            
            # Add memory of this interaction
            memory_id = None
            if hasattr(self.brain, "memory_core"):
                memory_text = f"User said: {user_input}"
                
                # Get emotional context formatting if available
                emotional_context = {}
                if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_formatted_emotional_state"):
                    emotional_context = self.brain.emotional_core.get_formatted_emotional_state()
                
                memory_id = await self.brain.memory_core.add_memory(
                    memory_text=memory_text,
                    memory_type="observation",
                    significance=5,
                    tags=["interaction", "user_input"],
                    metadata={
                        "emotional_context": emotional_context,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_id": str(self.brain.user_id)
                    }
                )
            
            # Check for context change using dynamic adaptation
            context_change_result = None
            adaptation_result = None
            if hasattr(self.brain, "dynamic_adaptation"):
                # Prepare context for change detection
                context_for_adaptation = {
                    "user_input": user_input,
                    "emotional_state": emotional_state,
                    "memories_retrieved": len(memories),
                    "has_experience": experience_result["has_experience"] if experience_result else False,
                    "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                    "interaction_count": getattr(self.brain, "interaction_count", 0),
                    "identity_impact": True if identity_impact else False
                }
                
                try:
                    # Detect context change
                    context_change_result = await self.brain.dynamic_adaptation.detect_context_change(context_for_adaptation)
                    
                    # If significant change, run adaptation cycle
                    if hasattr(context_change_result, "significant_change") and context_change_result.significant_change:
                        # Measure current performance
                        current_performance = {
                            "success_rate": context.get("success_rate", 0.7),
                            "error_rate": context.get("error_rate", 0.1),
                            "efficiency": context.get("efficiency", 0.8),
                            "response_time": (datetime.datetime.now() - start_time).total_seconds()
                        }
                        
                        # Run adaptation cycle
                        adaptation_result = await self.brain.dynamic_adaptation.adaptation_cycle(
                            context_for_adaptation, current_performance
                        )
                except Exception as e:
                    logger.error(f"Error in adaptation: {str(e)}")
            
            # Update interaction tracking
            if hasattr(self.brain, "last_interaction"):
                self.brain.last_interaction = datetime.datetime.now()
            
            if hasattr(self.brain, "interaction_count"):
                self.brain.interaction_count += 1
            
            # Add perceptual processing info if available
            perceptual_processing = None
            if 'percept' in locals() and percept:
                perceptual_processing = {
                    "modality": percept.modality,
                    "attention_weight": percept.attention_weight,
                    "bottom_up_confidence": percept.bottom_up_confidence,
                    "top_down_influence": percept.top_down_influence
                }
            
            # Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Track response time if performance metrics available
            if hasattr(self.brain, "performance_metrics") and "response_times" in self.brain.performance_metrics:
                self.brain.performance_metrics["response_times"].append(response_time)
            
            # Prepare result
            result = {
                "user_input": user_input,
                "emotional_state": emotional_state,
                "memories": memories,
                "memory_count": len(memories),
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "experience_response": experience_result["response_text"] if experience_result and experience_result.get("has_experience", False) else None,
                "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                "memory_id": memory_id,
                "response_time": response_time,
                "context_change": context_change_result,
                "adaptation_result": adaptation_result,
                "identity_impact": identity_impact,
                "meta_result": meta_result,
                "procedural_knowledge": procedural_knowledge
            }
            
            # Add temporal context if available
            if temporal_effects:
                result["temporal_context"] = temporal_effects
            
            # Add perceptual processing if available
            if perceptual_processing:
                result["perceptual_processing"] = perceptual_processing
            
            # Add hormone info if available
            if hasattr(self.brain, "hormone_system") and self.brain.hormone_system:
                try:
                    hormone_levels = {name: data["value"] for name, data in self.brain.hormone_system.hormones.items()}
                    result["hormone_levels"] = hormone_levels
                except Exception as e:
                    logger.error(f"Error getting hormone levels: {str(e)}")
            
            # Add reward result if available in context
            if context and "reward_outcome" in context and hasattr(self.brain, "process_reward"):
                try:
                    reward_result = await self.brain.process_reward(
                        context=context,
                        outcome=context["reward_outcome"],
                        success_level=context.get("success_level", 0.5)
                    )
                    result["reward_processing"] = reward_result
                except Exception as e:
                    logger.error(f"Error processing reward: {str(e)}")
            
            return result
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response based on processing results
        
        Args:
            user_input: Original user input
            processing_result: Results from process_input
            context: Additional context
            
        Returns:
            Response data
        """
        context = context or {}
        
        with trace(workflow_name="generate_response_serial", group_id=self.brain.trace_group_id):
            # Track if experience sharing was adapted
            experience_sharing_adapted = processing_result.get("context_change") is not None and \
                                       processing_result.get("adaptation_result") is not None
            
            # Determine if experience response should be used
            if processing_result["has_experience"]:
                main_response = processing_result["experience_response"]
                response_type = "experience"
                
                # If it's a cross-user experience, mark it
                if processing_result.get("cross_user_experience", False):
                    response_type = "cross_user_experience"
            else:
                # For reasoning-related queries, use the reasoning agents
                if self._is_reasoning_query(user_input) and hasattr(self.brain, "reasoning_core"):
                    try:
                        reasoning_result = await Runner.run(
                            self.brain.reasoning_triage_agent,
                            user_input
                        )
                        main_response = reasoning_result.final_output
                        response_type = "reasoning"
                    except Exception as e:
                        logger.error(f"Error in reasoning response: {str(e)}")
                        # Fallback to standard response
                        main_response = "I understand your question and would like to reason through it with you."
                        response_type = "standard"
                else:
                    # Check if procedural knowledge can be used
                    procedural_knowledge = processing_result.get("procedural_knowledge", None)
                    if procedural_knowledge and procedural_knowledge.get("can_execute", False) and len(procedural_knowledge.get("relevant_procedures", [])) > 0:
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
                                # Standard response
                                main_response = "I understand your input and have processed it."
                                response_type = "standard"
                        except Exception as e:
                            logger.error(f"Error executing procedure: {str(e)}")
                            main_response = "I understand your input and have processed it."
                            response_type = "standard"
                    else:
                        # No specific type, standard response
                        main_response = "I understand your input and have processed it."
                        response_type = "standard"
            
            # Determine if emotion should be expressed
            emotional_expression = None
            if hasattr(self.brain, "emotional_core"):
                should_express_emotion = self.brain.emotional_core.should_express_emotion()
                
                if should_express_emotion:
                    try:
                        expression_result = await self.brain.emotional_core.generate_emotional_expression(force=False)
                        if expression_result.get("expressed", False):
                            emotional_expression = expression_result.get("expression", "")
                    except Exception as e:
                        logger.error(f"Error generating emotional expression: {str(e)}")
                        if hasattr(self.brain.emotional_core, "get_expression_for_emotion"):
                            emotional_expression = self.brain.emotional_core.get_expression_for_emotion()
            
            # Get emotional state from processing result or update if needed
            emotional_state = processing_result.get("emotional_state", {})
            
            # Package the response
            response_data = {
                "message": main_response,
                "response_type": response_type,
                "emotional_state": emotional_state,
                "emotional_expression": emotional_expression,
                "memories_used": [m["id"] for m in processing_result["memories"]] if "memories" in processing_result else [],
                "memory_count": processing_result.get("memory_count", 0),
                "experience_sharing_adapted": experience_sharing_adapted,
                "identity_impact": processing_result.get("identity_impact")
            }
            
            # Add memory of this response
            if hasattr(self.brain, "memory_core"):
                try:
                    # Get emotional context formatting if available
                    emotional_context = {}
                    if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_formatted_emotional_state"):
                        emotional_context = self.brain.emotional_core.get_formatted_emotional_state()
                    
                    await self.brain.memory_core.add_memory(
                        memory_text=f"I responded: {main_response}",
                        memory_type="observation",
                        significance=5,
                        tags=["interaction", "nyx_response", response_type],
                        metadata={
                            "emotional_context": emotional_context,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "response_type": response_type,
                            "user_id": str(self.brain.user_id)
                        }
                    )
                except Exception as e:
                    logger.error(f"Error adding response memory: {str(e)}")
            
            # Evaluate the response if internal feedback system is available
            if hasattr(self.brain, "internal_feedback"):
                try:
                    evaluation = await self.brain.internal_feedback.critic_evaluate(
                        aspect="effectiveness",
                        content={"text": main_response, "type": response_type},
                        context={"user_input": user_input}
                    )
                    
                    # Add evaluation to response data
                    response_data["evaluation"] = evaluation
                except Exception as e:
                    logger.error(f"Error evaluating response: {str(e)}")
            
            # Check if it's time for experience consolidation
            self._check_and_run_consolidation()
            
            # Check if it's time for identity reflection
            identity_reflection_interval = getattr(self.brain, "identity_reflection_interval", 10)
            if hasattr(self.brain, "interaction_count") and self.brain.interaction_count % identity_reflection_interval == 0:
                try:
                    if hasattr(self.brain, "get_identity_state"):
                        identity_state = await self.brain.get_identity_state()
                        response_data["identity_reflection"] = identity_state
                except Exception as e:
                    logger.error(f"Error generating identity reflection: {str(e)}")
            
            # Add temporal expressions if appropriate
            if hasattr(self.brain, "temporal_perception") and (random.random() < 0.2 or context.get("include_time_expression", False)):
                try:
                    time_expression = await self.brain.temporal_perception.generate_temporal_expression()
                    if time_expression:
                        # Prepend or append the time expression to the response
                        if random.random() < 0.5 and not response_data["message"].startswith(time_expression["expression"]):
                            response_data["message"] = f"{time_expression['expression']} {response_data['message']}"
                        elif not response_data["message"].endswith(time_expression["expression"]):
                            response_data["message"] = f"{response_data['message']} {time_expression['expression']}"
                        
                        response_data["time_expression"] = time_expression
                except Exception as e:
                    logger.error(f"Error generating time expression: {str(e)}")
            
            # Process end of interaction for temporal tracking
            if hasattr(self.brain, "temporal_perception"):
                await self.brain.temporal_perception.on_interaction_end()
            
            return response_data
    
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
        """Determine if we should share an experience based on input and context"""
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
        user_id = str(self.brain.user_id)
        if hasattr(self.brain, "experience_interface") and hasattr(self.brain.experience_interface, "_get_user_preference_profile"):
            try:
                profile = self.brain.experience_interface._get_user_preference_profile(user_id)
                sharing_preference = profile.get("experience_sharing_preference", 0.5)
                
                # Higher preference means more likely to share experiences even without explicit request
                random_factor = random.random()
                if random_factor < sharing_preference * 0.5:  # Scale down to make this path less common
                    return True
            except:
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
    
    def _check_and_run_consolidation(self) -> None:
        """Check if it's time for consolidation and run if needed"""
        if not hasattr(self.brain, "experience_interface"):
            return
        
        # Check time since last consolidation
        now = datetime.datetime.now()
        
        if hasattr(self.brain, "last_consolidation") and hasattr(self.brain, "consolidation_interval"):
            time_since_last = (now - self.brain.last_consolidation).total_seconds() / 3600  # hours
            
            if time_since_last >= self.brain.consolidation_interval:
                try:
                    # Run consolidation in background
                    if hasattr(self.brain, "run_experience_consolidation"):
                        asyncio.create_task(self.brain.run_experience_consolidation())
                    
                    # Update last consolidation time
                    self.brain.last_consolidation = now
                except Exception as e:
                    logger.error(f"Error scheduling consolidation: {str(e)}")
