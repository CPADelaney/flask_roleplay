# nyx/core/brain/processing/parallel.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional

from agents import trace

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Handles parallel processing of inputs using multiple tasks"""
    
    def __init__(self, brain):
        self.brain = brain
    
    async def initialize(self):
        """Initialize the processor"""
        logger.info("Parallel processor initialized")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input with parallel operations for improved performance
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with relevant memories, emotional state, etc.
        """
        if not self.brain.initialized:
            await self.brain.initialize()
        
        with trace(workflow_name="process_input_parallel", group_id=self.brain.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context
            context = context or {}
            
            # Add user_id to context if not present
            if "user_id" not in context:
                context["user_id"] = str(self.brain.user_id)
            
            # Create tasks for parallel processing
            tasks = {}
            
            # Task 1: Process emotional impact
            if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
                tasks["emotional"] = asyncio.create_task(
                    self._process_emotional_impact(user_input, context)
                )
            
            # Task 2: Run meta-cognitive cycle
            if hasattr(self.brain, "meta_core") and self.brain.meta_core:
                meta_context = context.copy()
                meta_context["user_input"] = user_input
                tasks["meta"] = asyncio.create_task(
                    self.brain.meta_core.cognitive_cycle(meta_context)
                )
            
            # Task 3: Check for experience sharing opportunity
            if hasattr(self.brain, "experience_interface") and self.brain.experience_interface:
                tasks["experience_check"] = asyncio.create_task(
                    self._check_experience_sharing(user_input, context)
                )
            
            # Wait for emotional processing to complete
            emotional_state = {}
            try:
                if "emotional" in tasks:
                    emotional_result = await tasks["emotional"]
                    emotional_state = emotional_result["emotional_state"]
                    
                    # Add emotional state to context for memory retrieval
                    context["emotional_state"] = emotional_state
            except Exception as e:
                logger.error(f"Error in emotional processing: {str(e)}")
            
            # Start memory retrieval with emotional context
            if hasattr(self.brain, "memory_orchestrator") and self.brain.memory_orchestrator:
                tasks["memory"] = asyncio.create_task(
                    self._retrieve_memories_with_emotion(user_input, context, emotional_state)
                )
            
            # Wait for experience sharing check to complete
            should_share_experience = False
            try:
                if "experience_check" in tasks:
                    should_share_experience = await tasks["experience_check"]
            except Exception as e:
                logger.error(f"Error checking experience sharing: {str(e)}")
            
            # Start experience sharing task if needed
            experience_result = None
            if should_share_experience and hasattr(self.brain, "experience_interface"):
                tasks["experience"] = asyncio.create_task(
                    self._share_experience(user_input, context, emotional_state)
                )
            
            # Wait for memory retrieval to complete
            memories = []
            try:
                if "memory" in tasks:
                    memories = await tasks["memory"]
            except Exception as e:
                logger.error(f"Error retrieving memories: {str(e)}")
            
            # Update emotional state based on retrieved memories
            if memories and hasattr(self.brain, "emotional_core"):
                try:
                    memory_emotional_impact = await self._calculate_memory_emotional_impact(memories)
                    
                    # Apply memory-to-emotion influence
                    influence = getattr(self.brain, "memory_to_emotion_influence", 0.3)
                    for emotion, value in memory_emotional_impact.items():
                        self.brain.emotional_core.update_emotion(emotion, value * influence)
                    
                    # Get updated emotional state
                    emotional_state = self.brain.emotional_core.get_emotional_state()
                except Exception as e:
                    logger.error(f"Error updating emotion from memories: {str(e)}")
            
            # Wait for experience sharing to complete if started
            identity_impact = None
            if "experience" in tasks:
                try:
                    experience_result = await tasks["experience"]
                    
                    # Calculate potential identity impact if experience found
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
                except Exception as e:
                    logger.error(f"Error sharing experience: {str(e)}")
                    experience_result = {"has_experience": False}
            
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
            
            if hasattr(self.brain, "dynamic_adaptation") and self.brain.dynamic_adaptation:
                # Process adaptation in parallel
                tasks["adaptation"] = asyncio.create_task(
                    self._process_adaptation(user_input, context, emotional_state, 
                                          experience_result, identity_impact)
                )
                
                try:
                    adaptation_results = await tasks["adaptation"]
                    context_change_result = adaptation_results.get("context_change")
                    adaptation_result = adaptation_results.get("adaptation_result")
                except Exception as e:
                    logger.error(f"Error in adaptation: {str(e)}")
            
            # Wait for meta cognitive cycle to complete
            meta_result = {}
            if "meta" in tasks:
                try:
                    meta_result = await tasks["meta"]
                except Exception as e:
                    logger.error(f"Error in meta-cognitive cycle: {str(e)}")
                    meta_result = {"error": str(e)}
            
            # Check procedural knowledge
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
            
            # Update interaction tracking
            if hasattr(self.brain, "last_interaction"):
                self.brain.last_interaction = datetime.datetime.now()
            
            if hasattr(self.brain, "interaction_count"):
                self.brain.interaction_count += 1
            
            # Update performance metrics
            if hasattr(self.brain, "performance_metrics"):
                performance_metrics = self.brain.performance_metrics
                
                performance_metrics["memory_operations"] = performance_metrics.get("memory_operations", 0) + 1
                performance_metrics["emotion_updates"] = performance_metrics.get("emotion_updates", 0) + 1
                
                if experience_result and experience_result.get("has_experience", False):
                    performance_metrics["experiences_shared"] = performance_metrics.get("experiences_shared", 0) + 1
                    if experience_result.get("cross_user", False):
                        performance_metrics["cross_user_experiences_shared"] = performance_metrics.get("cross_user_experiences_shared", 0) + 1
            
            # Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if hasattr(self.brain, "performance_metrics") and "response_times" in self.brain.performance_metrics:
                self.brain.performance_metrics["response_times"].append(response_time)
            
            # Return integrated processing results
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
                "procedural_knowledge": procedural_knowledge,
                "parallel_processing": True  # Flag to indicate parallel processing was used
            }
            
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
        
        with trace(workflow_name="generate_response_parallel", group_id=self.brain.trace_group_id):
            # Start response generation tasks in parallel
            tasks = {}
            
            # Task 1: Determine main response content
            tasks["main_response"] = asyncio.create_task(
                self._determine_main_response(user_input, processing_result, context)
            )
            
            # Task 2: Generate emotional expression
            if hasattr(self.brain, "emotional_core"):
                tasks["emotional_expression"] = asyncio.create_task(
                    self._generate_emotional_expression(processing_result["emotional_state"])
                )
            
            # Wait for main response content
            main_response_result = await tasks["main_response"]
            main_response = main_response_result["message"]
            response_type = main_response_result["response_type"]
            
            # Wait for emotional expression
            emotional_expression = None
            try:
                if "emotional_expression" in tasks:
                    emotional_expression_result = await tasks["emotional_expression"]
                    emotional_expression = emotional_expression_result["expression"]
            except Exception as e:
                logger.error(f"Error generating emotional expression: {str(e)}")
            
            # Add memory of this response
            memory_id = None
            if hasattr(self.brain, "memory_core"):
                memory_text = f"I responded: {main_response}"
                
                # Get emotional context formatting if available
                emotional_context = {}
                if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_formatted_emotional_state"):
                    emotional_context = self.brain.emotional_core.get_formatted_emotional_state()
                
                memory_id = await self.brain.memory_core.add_memory(
                    memory_text=memory_text,
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
            
            # Start evaluation in parallel if internal feedback system is available
            evaluation = None
            if hasattr(self.brain, "internal_feedback") and self.brain.internal_feedback:
                tasks["evaluation"] = asyncio.create_task(
                    self.brain.internal_feedback.critic_evaluate(
                        aspect="effectiveness",
                        content={"text": main_response, "type": response_type},
                        context={"user_input": user_input}
                    )
                )
                
                try:
                    evaluation = await tasks["evaluation"]
                except Exception as e:
                    logger.error(f"Error evaluating response: {str(e)}")
            
            # Check if it's time for experience consolidation in parallel
            if hasattr(self.brain, "last_consolidation") and hasattr(self.brain, "consolidation_interval"):
                now = datetime.datetime.now()
                time_since_last = (now - self.brain.last_consolidation).total_seconds() / 3600  # hours
                
                if time_since_last >= self.brain.consolidation_interval and hasattr(self.brain, "run_experience_consolidation"):
                    # Create background task that won't block response
                    asyncio.create_task(self.brain.run_experience_consolidation())
                    self.brain.last_consolidation = now
            
            # Check if it's time for identity reflection in parallel
            identity_reflection = None
            if hasattr(self.brain, "interaction_count") and hasattr(self.brain, "identity_reflection_interval"):
                if self.brain.interaction_count % self.brain.identity_reflection_interval == 0:
                    try:
                        # Create task for identity reflection
                        if hasattr(self.brain, "get_identity_state"):
                            tasks["identity"] = asyncio.create_task(
                                self.brain.get_identity_state()
                            )
                            
                            identity_reflection = await tasks["identity"]
                    except Exception as e:
                        logger.error(f"Error generating identity reflection: {str(e)}")
            
            # Package the response
            response_data = {
                "message": main_response,
                "response_type": response_type,
                "emotional_state": processing_result["emotional_state"],
                "emotional_expression": emotional_expression,
                "memories_used": [m["id"] for m in processing_result["memories"]] if "memories" in processing_result and isinstance(processing_result["memories"], list) else [],
                "memory_count": processing_result.get("memory_count", 0),
                "evaluation": evaluation,
                "experience_sharing_adapted": processing_result.get("context_change") is not None and processing_result.get("adaptation_result") is not None,
                "identity_impact": processing_result.get("identity_impact"),
                "identity_reflection": identity_reflection,
                "response_memory_id": memory_id,
                "parallel_processing": True  # Flag to indicate parallel processing was used
            }
            
            return response_data
    
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
    
    async def _check_experience_sharing(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Check if experience sharing should be used"""
        if hasattr(self.brain, "_should_share_experience"):
            return self.brain._should_share_experience(user_input, context)
        
        # Default implementation if brain doesn't have the method
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
        
        return False
    
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
    
    async def _process_adaptation(self,
                               user_input: str,
                               context: Dict[str, Any],
                               emotional_state: Dict[str, float],
                               experience_result: Optional[Dict[str, Any]],
                               identity_impact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process adaptation based on context change"""
        if not hasattr(self.brain, "dynamic_adaptation") or not self.brain.dynamic_adaptation:
            return {"context_change": None, "adaptation_result": None}
        
        # Prepare context for change detection
        context_for_adaptation = {
            "user_input": user_input,
            "emotional_state": emotional_state,
            "memories_retrieved": context.get("memory_count", 0),
            "has_experience": experience_result["has_experience"] if experience_result else False,
            "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
            "interaction_count": getattr(self.brain, "interaction_count", 0),
            "identity_impact": True if identity_impact else False
        }
        
        # Detect context change
        context_change_result = await self.brain.dynamic_adaptation.detect_context_change(context_for_adaptation)
        
        adaptation_result = None
        
        # If significant change, run adaptation cycle
        if hasattr(context_change_result, "significant_change") and context_change_result.significant_change:
            # Measure current performance
            current_performance = {
                "success_rate": context.get("success_rate", 0.7),
                "error_rate": context.get("error_rate", 0.1),
                "efficiency": context.get("efficiency", 0.8),
                "response_time": context.get("response_time", 0.5)
            }
            
            # Run adaptation cycle
            adaptation_result = await self.brain.dynamic_adaptation.adaptation_cycle(
                context_for_adaptation, current_performance
            )
        
        return {
            "context_change": context_change_result,
            "adaptation_result": adaptation_result
        }
    
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
            if procedural_knowledge and procedural_knowledge.get("can_execute", False) and len(procedural_knowledge.get("relevant_procedures", [])) > 0:
                try:
                    # Get the most relevant procedure
                    top_procedure = procedural_knowledge["relevant_procedures"][0]
                    
                    # Execute the procedure
                    if hasattr(self.brain, "agent_enhanced_memory"):
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
                            if hasattr(self.brain, "_is_reasoning_query") and self.brain._is_reasoning_query(user_input):
                                main_response = await self._generate_reasoning_response(user_input)
                                response_type = "reasoning"
                            else:
                                # Fallback
                                main_response = "I understand your input and have processed it."
                                response_type = "standard"
                    else:
                        # For reasoning-related queries, use the reasoning agents
                        if hasattr(self.brain, "_is_reasoning_query") and self.brain._is_reasoning_query(user_input):
                            main_response = await self._generate_reasoning_response(user_input)
                            response_type = "reasoning"
                        else:
                            # Fallback
                            main_response = "I understand your input and have processed it."
                            response_type = "standard"
                except Exception as e:
                    logger.error(f"Error executing procedure: {str(e)}")
                    
                    # For reasoning-related queries, use the reasoning agents
                    if hasattr(self.brain, "_is_reasoning_query") and self.brain._is_reasoning_query(user_input):
                        main_response = await self._generate_reasoning_response(user_input)
                        response_type = "reasoning"
                    else:
                        # Fallback
                        main_response = "I understand your input and have processed it."
                        response_type = "standard"
            else:
                # For reasoning-related queries, use the reasoning agents
                if hasattr(self.brain, "_is_reasoning_query") and self.brain._is_reasoning_query(user_input):
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
