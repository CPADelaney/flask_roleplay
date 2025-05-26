# nyx/core/brain/processing/serial.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional

from agents import trace, Runner
from nyx.core.brain.models import SensoryInput
from nyx.core.brain.processing.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class SerialProcessor(BaseProcessor):
    """Handles serial processing of inputs"""
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input using serial processing path
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with relevant memories, emotional state, etc.
        """
        if not self.brain.initialized:
            await self.brain.initialize()
        
        with trace(workflow_name="process_input_serial", group_id=self.brain.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Process temporal effects if available
            temporal_effects = None
            if hasattr(self.brain, "temporal_perception") and self.brain.temporal_perception:
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
            
            # Process emotional impact of input - using base class method
            emotional_result = await self._process_emotional_impact(user_input, context)
            emotional_state = emotional_result["emotional_state"]
            
            # Add emotional state to context for memory retrieval
            context["emotional_state"] = emotional_state
            
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
            
            # Retrieve relevant memories using memory orchestrator - using base class method
            memories = await self._retrieve_memories_with_emotion(user_input, context, emotional_state)
            
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
            
            # Check if experience sharing is appropriate - using base class method
            should_share_experience = self._should_share_experience(user_input, context)
            experience_result = None
            identity_impact = None
            
            if should_share_experience:
                # Share experience - using base class method
                experience_result = await self._share_experience(user_input, context, emotional_state)
                
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
            else:
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
            if 'percept' in locals() and percept:
                result["perceptual_processing"] = {
                    "modality": percept.modality,
                    "attention_weight": percept.attention_weight,
                    "bottom_up_confidence": percept.bottom_up_confidence,
                    "top_down_influence": percept.top_down_influence
                }
            
            # Add hormone info if available
            if hasattr(self.brain, "hormone_system") and self.brain.hormone_system:
                try:
                    hormone_levels = {name: data["value"] for name, data in self.brain.hormone_system.hormones.items()}
                    result["hormone_levels"] = hormone_levels
                except Exception as e:
                    logger.error(f"Error getting hormone levels: {str(e)}")
            
            # Publish event for processing completion if event system exists
            if hasattr(self.brain, "event_system"):
                await self.brain.event_system.publish("input_processed", {
                    "processor": "serial",
                    "user_input": user_input,
                    "result": result
                })
            
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
            
            # Determine main response using base class method
            main_response_result = await self._determine_main_response(user_input, processing_result, context)
            main_response = main_response_result["message"]
            response_type = main_response_result["response_type"]
            
            # Generate emotional expression using base class method
            emotional_expression_result = await self._generate_emotional_expression(processing_result["emotional_state"])
            emotional_expression = emotional_expression_result["expression"]
            
            # Package the response
            response_data = {
                "message": main_response,
                "response_type": response_type,
                "emotional_state": processing_result["emotional_state"],
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
            
            # Publish event for response generated if event system exists
            if hasattr(self.brain, "event_system"):
                await self.brain.event_system.publish("response_generated", {
                    "processor": "serial",
                    "response_type": response_type,
                    "response_data": response_data
                })
            
            return response_data
