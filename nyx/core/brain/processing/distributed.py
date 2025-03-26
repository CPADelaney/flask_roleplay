# nyx/core/brain/processing/distributed.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional

from agents import trace

logger = logging.getLogger(__name__)

class DistributedProcessor:
    """Handles distributed processing across multiple subsystems"""
    
    def __init__(self, brain):
        self.brain = brain
    
    async def initialize(self):
        """Initialize the processor"""
        logger.info("Distributed processor initialized")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input using fully distributed processing
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results
        """
        if not self.brain.initialized:
            await self.brain.initialize()
        
        with trace(workflow_name="process_input_distributed", group_id=self.brain.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context
            context = context or {}
            
            # Add user_id to context if not present
            if "user_id" not in context:
                context["user_id"] = str(self.brain.user_id)
            
            # Initialize distributed processing manager
            distributed_processing = getattr(self.brain, "distributed_processing", None)
            
            # If no distributed processing manager, use parallel processing as fallback
            if not distributed_processing:
                logger.warning("Distributed processing manager not available, falling back to parallel processing")
                if hasattr(self.brain, "processing_manager") and "parallel" in self.brain.processing_manager.processors:
                    return await self.brain.processing_manager.processors["parallel"].process_input(user_input, context)
                else:
                    # Create a simple task manager
                    task_manager = TaskManager()
            else:
                task_manager = distributed_processing
            
            # 1. Register emotional processing task (no dependencies)
            task_manager.register_task(
                task_id="emotional_processing",
                coroutine=self._process_emotional_impact(user_input, context),
                priority=3,
                group="emotion"
            )
            
            # 2. Register meta-cognitive cycle task (no dependencies)
            if hasattr(self.brain, "meta_core") and self.brain.meta_core:
                meta_context = context.copy()
                meta_context["user_input"] = user_input
                
                task_manager.register_task(
                    task_id="meta_cycle",
                    coroutine=self.brain.meta_core.cognitive_cycle(meta_context),
                    priority=1,
                    group="meta"
                )
            
            # 3. Register prediction task if available
            if hasattr(self.brain, "prediction_engine") and self.brain.prediction_engine:
                prediction_input = {
                    "context": context.copy(),
                    "user_input": user_input,
                    "cycle": getattr(self.brain, "interaction_count", 0)
                }
                
                task_manager.register_task(
                    task_id="prediction",
                    coroutine=self.brain.prediction_engine.generate_prediction(prediction_input),
                    priority=2,
                    group="meta"
                )
            
            # 4. Register memory retrieval task (depends on emotional processing)
            task_manager.register_task(
                task_id="memory_retrieval",
                coroutine=self._retrieve_memories_placeholder(user_input, context),
                dependencies=["emotional_processing"],
                priority=3,
                group="memory"
            )
            
            # 5. Register experience check task (no dependencies)
            task_manager.register_task(
                task_id="experience_check",
                coroutine=self._check_experience_sharing(user_input, context),
                priority=2,
                group="memory"
            )
            
            # 6. Register experience sharing task (depends on experience check and emotional processing)
            task_manager.register_task(
                task_id="experience_sharing",
                coroutine=self._share_experience_placeholder(user_input, context),
                dependencies=["experience_check", "emotional_processing"],
                priority=2,
                group="memory"
            )
            
            # 7. Register memory storage task (no dependencies)
            if hasattr(self.brain, "memory_core") and self.brain.memory_core:
                memory_text = f"User said: {user_input}"
                task_manager.register_task(
                    task_id="memory_storage",
                    coroutine=self.brain.memory_core.add_memory(
                        memory_text=memory_text,
                        memory_type="observation",
                        significance=5,
                        tags=["interaction", "user_input"],
                        metadata={
                            "timestamp": datetime.datetime.now().isoformat(),
                            "user_id": str(self.brain.user_id)
                        }
                    ),
                    priority=1,
                    group="memory"
                )
            
            # 8. Register adaptation task (depends on multiple tasks)
            if hasattr(self.brain, "dynamic_adaptation") and self.brain.dynamic_adaptation:
                task_manager.register_task(
                    task_id="adaptation",
                    coroutine=self._process_adaptation_placeholder(user_input, context),
                    dependencies=["emotional_processing", "experience_sharing", "memory_retrieval"],
                    priority=1,
                    group="adaptation"
                )
            
            # 9. Register identity impact task (depends on experience sharing)
            if hasattr(self.brain, "identity_evolution") and self.brain.identity_evolution:
                task_manager.register_task(
                    task_id="identity_impact",
                    coroutine=self._process_identity_impact_placeholder(user_input, context),
                    dependencies=["experience_sharing"],
                    priority=1,
                    group="reflection"
                )
            
            # 10. Register procedural memory task (no dependencies)
            if hasattr(self.brain, "agent_enhanced_memory") and self.brain.agent_enhanced_memory:
                task_manager.register_task(
                    task_id="procedural_memory",
                    coroutine=self.brain.agent_enhanced_memory.find_similar_procedures(user_input),
                    priority=2,
                    group="procedural"
                )
            
            # Execute all tasks with dependency resolution
            results = await task_manager.execute_tasks()
            
            # Process results from each task
            emotional_state = results.get("emotional_processing", {}).get("emotional_state", {})
            memories = results.get("memory_retrieval", [])
            memory_id = results.get("memory_storage", "")
            experience_result = results.get("experience_sharing", {"has_experience": False})
            adaptation_result = results.get("adaptation", {})
            identity_impact = results.get("identity_impact", None)
            meta_result = results.get("meta_cycle", {})
            prediction_result = results.get("prediction", {})
            procedural_knowledge = None
            
            if "procedural_memory" in results and results["procedural_memory"]:
                relevant_procedures = results["procedural_memory"]
                procedural_knowledge = {
                    "relevant_procedures": relevant_procedures,
                    "can_execute": len(relevant_procedures) > 0
                }
            
            # Update context change info
            context_change_result = adaptation_result.get("context_change")
            adaptation_cycle_result = adaptation_result.get("adaptation_result")
            
            # Update interaction tracking
            if hasattr(self.brain, "last_interaction"):
                self.brain.last_interaction = datetime.datetime.now()
            
            if hasattr(self.brain, "interaction_count"):
                self.brain.interaction_count += 1
            
            # Update performance metrics
            if hasattr(self.brain, "performance_metrics"):
                performance_metrics = self.brain.performance_metrics
                
                # Update memory operations count
                performance_metrics["memory_operations"] = performance_metrics.get("memory_operations", 0) + 1
                
                # Update emotion updates count
                performance_metrics["emotion_updates"] = performance_metrics.get("emotion_updates", 0) + 1
                
                # Update experiences shared count
                if experience_result and experience_result.get("has_experience", False):
                    performance_metrics["experiences_shared"] = performance_metrics.get("experiences_shared", 0) + 1
                    
                    # Track cross-user experiences
                    if experience_result.get("cross_user", False):
                        performance_metrics["cross_user_experiences_shared"] = performance_metrics.get("cross_user_experiences_shared", 0) + 1
            
            # Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if hasattr(self.brain, "performance_metrics") and "response_times" in self.brain.performance_metrics:
                self.brain.performance_metrics["response_times"].append(response_time)
            
            # Performance metrics from distributed processing
            performance_metrics = results.get("_performance", {})
            
            # Return processing results in a structured format
            result = {
                "user_input": user_input,
                "emotional_state": emotional_state,
                "memories": memories,
                "memory_count": len(memories) if isinstance(memories, list) else 0,
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "experience_response": experience_result.get("response_text", None) if experience_result else None,
                "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                "memory_id": memory_id,
                "response_time": response_time,
                "context_change": context_change_result,
                "adaptation_result": adaptation_cycle_result,
                "identity_impact": identity_impact,
                "meta_result": meta_result,
                "prediction": prediction_result,
                "procedural_knowledge": procedural_knowledge,
                "distributed_processing": True,
                "performance_metrics": performance_metrics
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
        # Use the parallel processor's generation method since it works well for distributed too
        if hasattr(self.brain, "processing_manager") and "parallel" in self.brain.processing_manager.processors:
            parallel_processor = self.brain.processing_manager.processors["parallel"]
            return await parallel_processor.generate_response(user_input, processing_result, context)
        
        # Fallback implementation
        with trace(workflow_name="generate_response_distributed", group_id=self.brain.trace_group_id):
            # Determine if experience response should be used
            if processing_result["has_experience"]:
                main_response = processing_result["experience_response"]
                response_type = "experience"
                
                # If it's a cross-user experience, mark it
                if processing_result.get("cross_user_experience", False):
                    response_type = "cross_user_experience"
            else:
                # Check if procedural knowledge can be used
                procedural_knowledge = processing_result.get("procedural_knowledge")
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
                            if hasattr(self.brain, "_is_reasoning_query") and self.brain._is_reasoning_query(user_input):
                                try:
                                    if hasattr(self.brain, "reasoning_triage_agent") and hasattr(self.brain, "Runner"):
                                        reasoning_result = await self.brain.Runner.run(
                                            self.brain.reasoning_triage_agent,
                                            user_input
                                        )
                                        main_response = reasoning_result.final_output if hasattr(reasoning_result, "final_output") else str(reasoning_result)
                                    else:
                                        main_response = "I understand your question and would like to reason through it with you."
                                    response_type = "reasoning"
                                except Exception as e:
                                    logger.error(f"Error in reasoning response: {str(e)}")
                                    main_response = "I understand your question and would like to reason through it with you."
                                    response_type = "standard"
                            else:
                                # No specific type, standard response
                                main_response = "I understand your input and have processed it."
                                response_type = "standard"
                    except Exception as e:
                        logger.error(f"Error executing procedure: {str(e)}")
                        main_response = "I understand your input and have processed it."
                        response_type = "standard"
                else:
                    # For reasoning-related queries, use the reasoning agents
                    if hasattr(self.brain, "_is_reasoning_query") and self.brain._is_reasoning_query(user_input):
                        try:
                            if hasattr(self.brain, "reasoning_triage_agent") and hasattr(self.brain, "Runner"):
                                reasoning_result = await self.brain.Runner.run(
                                    self.brain.reasoning_triage_agent,
                                    user_input
                                )
                                main_response = reasoning_result.final_output if hasattr(reasoning_result, "final_output") else str(reasoning_result)
                            else:
                                main_response = "I understand your question and would like to reason through it with you."
                            response_type = "reasoning"
                        except Exception as e:
                            logger.error(f"Error in reasoning response: {str(e)}")
                            main_response = "I understand your question and would like to reason through it with you."
                            response_type = "standard"
                    else:
                        # No specific type, standard response
                        main_response = "I understand your input and have processed it."
                        response_type = "standard"
            
            # Determine if emotion should be expressed
            emotional_expression = None
            if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
                try:
                    should_express_emotion = self.brain.emotional_core.should_express_emotion()
                    
                    if should_express_emotion:
                        if hasattr(self.brain.emotional_core, "generate_emotional_expression"):
                            expression_result = await self.brain.emotional_core.generate_emotional_expression(force=False)
                            if expression_result.get("expressed", False):
                                emotional_expression = expression_result.get("expression", "")
                        elif hasattr(self.brain.emotional_core, "get_expression_for_emotion"):
                            emotional_expression = self.brain.emotional_core.get_expression_for_emotion()
                except Exception as e:
                    logger.error(f"Error generating emotional expression: {str(e)}")
            
            # Add memory of this response
            response_memory_id = None
            if hasattr(self.brain, "memory_core") and self.brain.memory_core:
                try:
                    # Get emotional context formatting if available
                    emotional_context = {}
                    if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_formatted_emotional_state"):
                        emotional_context = self.brain.emotional_core.get_formatted_emotional_state()
                    
                    response_memory_id = await self.brain.memory_core.add_memory(
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
            
            # Package the response
            response_data = {
                "message": main_response,
                "response_type": response_type,
                "emotional_state": processing_result.get("emotional_state", {}),
                "emotional_expression": emotional_expression,
                "memories_used": [m["id"] for m in processing_result["memories"]] if "memories" in processing_result and isinstance(processing_result["memories"], list) else [],
                "memory_count": processing_result.get("memory_count", 0),
                "distributed_processing": True,
                "response_memory_id": response_memory_id
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
    
    async def _retrieve_memories_placeholder(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Placeholder method for memory retrieval that will be updated with emotional context"""
        # This would normally wait for emotional processing, but in distributed processing,
        # we handle dependencies at the manager level, so here we can just retrieve
        # the emotional state directly from the core
        emotional_state = {}
        if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
            emotional_state = self.brain.emotional_core.get_emotional_state()
        
        # Now call the regular method
        if hasattr(self.brain, "processing_manager") and "parallel" in self.brain.processing_manager.processors:
            parallel_processor = self.brain.processing_manager.processors["parallel"]
            return await parallel_processor._retrieve_memories_with_emotion(user_input, context, emotional_state)
        
        # Fallback to direct retrieval
        if hasattr(self.brain, "memory_orchestrator") and self.brain.memory_orchestrator:
            return await self.brain.memory_orchestrator.retrieve_memories(
                query=user_input,
                memory_types=context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]), 
                limit=context.get("memory_limit", 5)
            )
        
        return []
    
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
    
    async def _share_experience_placeholder(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder method for experience sharing that will be updated with emotional context"""
        # Get the latest emotional state
        emotional_state = {}
        if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
            emotional_state = self.brain.emotional_core.get_emotional_state()
        
        # Check if we should share experience
        should_share = await self._check_experience_sharing(user_input, context)
        
        if not should_share:
            return {"has_experience": False}
        
        # Call the appropriate sharing method
        if hasattr(self.brain, "processing_manager") and "parallel" in self.brain.processing_manager.processors:
            parallel_processor = self.brain.processing_manager.processors["parallel"]
            return await parallel_processor._share_experience(user_input, context, emotional_state)
        
        # Fallback to direct experience sharing
        if hasattr(self.brain, "experience_interface") and self.brain.experience_interface:
            cross_user_enabled = getattr(self.brain, "cross_user_enabled", True)
            cross_user_sharing_threshold = getattr(self.brain, "cross_user_sharing_threshold", 0.7)
            
            return await self.brain.experience_interface.share_experience_enhanced(
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
        
        return {"has_experience": False}
    
    async def _process_adaptation_placeholder(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder method for adaptation processing"""
        # Get the latest state from other cores
        emotional_state = {}
        if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
            emotional_state = self.brain.emotional_core.get_emotional_state()
        
        # Try to get experience information
        experience_result = None
        try:
            if hasattr(self.brain, "experience_interface") and self.brain.experience_interface:
                # Check if experience is available
                experiences = await self.brain.experience_interface.retrieve_experiences_enhanced(
                    query=user_input,
                    limit=1,
                    user_id=str(self.brain.user_id)
                )
                
                if experiences:
                    experience_result = {
                        "has_experience": True,
                        "cross_user": False  # Default
                    }
                    
                    # Check if it's a cross-user experience
                    if "cross_user" in experiences[0]:
                        experience_result["cross_user"] = experiences[0]["cross_user"]
        except Exception as e:
            logger.error(f"Error checking experience in adaptation placeholder: {str(e)}")
            experience_result = {"has_experience": False}
        
        # Call the regular adaptation method
        if hasattr(self.brain, "processing_manager") and "parallel" in self.brain.processing_manager.processors:
            parallel_processor = self.brain.processing_manager.processors["parallel"]
            return await parallel_processor._process_adaptation(
                user_input, 
                context, 
                emotional_state, 
                experience_result, 
                None  # No identity impact yet
            )
        
        # Fallback if no adaptation processing available
        return {"context_change": None, "adaptation_result": None}
    
    async def _process_identity_impact_placeholder(self, user_input: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
