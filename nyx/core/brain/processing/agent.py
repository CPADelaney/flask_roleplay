# nyx/core/brain/processing/agent.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional

from agents import trace, Runner

logger = logging.getLogger(__name__)

class AgentProcessor:
    """Handles agent-based processing of inputs"""
    
    def __init__(self, brain, integration_mode=False):
        self.brain = brain
        self.integration_mode = integration_mode
    
    async def initialize(self):
        """Initialize the processor"""
        # Ensure agent capabilities are initialized
        if hasattr(self.brain, "initialize_agent_capabilities"):
            await self.brain.initialize_agent_capabilities()
        
        logger.info(f"Agent processor initialized (integration_mode={self.integration_mode})")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input using agent capabilities
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results
        """
        if not self.brain.initialized:
            await self.brain.initialize()
        
        with trace(workflow_name="process_input_agent", group_id=self.brain.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context if needed
            context = context or {}
            
            if self.integration_mode:
                # Process with both systems in parallel
                brain_task = asyncio.create_task(self._process_with_brain(user_input, context))
                agent_task = asyncio.create_task(self._process_with_agent(user_input, context))
                
                # Wait for both to complete
                brain_result, agent_result = await asyncio.gather(brain_task, agent_task)
                
                # Create integrated result
                result = {
                    "brain_processing": brain_result,
                    "agent_processing": agent_result,
                    "integrated": True,
                    "has_experience": brain_result.get("has_experience", False) or agent_result.get("has_experience", False),
                    "user_input": user_input,
                    "response_time": (datetime.datetime.now() - start_time).total_seconds()
                }
                
                # Prefer brain experiences if available
                if brain_result.get("has_experience", False):
                    result["experience_response"] = brain_result.get("experience_response")
                    result["cross_user_experience"] = brain_result.get("cross_user_experience", False)
                
                # Add image generation if agent suggests it
                if agent_result.get("generate_image", False):
                    result["generate_image"] = True
                    result["image_prompt"] = agent_result.get("image_prompt")
                
                return result
            else:
                # Process just with agent
                return await self._process_with_agent(user_input, context)
    
    async def _process_with_brain(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with brain systems"""
        # Default to serial processor if available
        if hasattr(self.brain, "processing_manager") and "serial" in self.brain.processing_manager.processors:
            serial_processor = self.brain.processing_manager.processors["serial"]
            return await serial_processor.process_input(user_input, context)
        
        # Fallback to direct processing
        if hasattr(self.brain, "_process_input_serial"):
            return await self.brain._process_input_serial(user_input, context)
        
        return {"error": "Brain processing not available", "user_input": user_input}
    
    async def _process_with_agent(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with agent capabilities"""
        if not hasattr(self.brain, "agent_capabilities_initialized") or not self.brain.agent_capabilities_initialized:
            if hasattr(self.brain, "initialize_agent_capabilities"):
                await self.brain.initialize_agent_capabilities()
            else:
                return {"error": "Agent capabilities not available", "user_input": user_input}
        
        try:
            # Get memories to enhance context
            enhanced_context = context.copy() if context else {}
            
            if hasattr(self.brain, "agent_context") and hasattr(self.brain, "retrieve_memories"):
                memories = await self.brain.retrieve_memories(self.brain.agent_context, user_input)
                enhanced_context["relevant_memories"] = memories
            
            # Get user model guidance if available
            if hasattr(self.brain, "get_user_model_guidance") and hasattr(self.brain, "agent_context"):
                user_guidance = await self.brain.get_user_model_guidance(self.brain.agent_context)
                enhanced_context["user_guidance"] = user_guidance
            
            # Handle emotional state
            emotional_state = {}
            if hasattr(self.brain, "emotional_core"):
                emotional_state = self.brain.emotional_core.get_emotional_state()
            enhanced_context["emotional_state"] = emotional_state
            
            # Generate response using the main agent
            if hasattr(self.brain, "nyx_main_agent") and hasattr(self.brain, "Runner") and hasattr(self.brain, "agent_context"):
                # Run the main agent
                result = await Runner.run(
                    self.brain.nyx_main_agent,
                    user_input,
                    context=self.brain.agent_context,
                    run_context=enhanced_context
                )
                
                # Get structured output
                if hasattr(result, "final_output_as") and hasattr(self.brain, "NarrativeResponse"):
                    narrative_response = result.final_output_as(self.brain.NarrativeResponse)
                    
                    # Filter and enhance response if available
                    if hasattr(self.brain, "response_filter"):
                        filtered_response = await self.brain.response_filter.filter_response(
                            narrative_response.narrative,
                            enhanced_context
                        )
                        
                        # Update response with filtered version
                        narrative_response.narrative = filtered_response
                    
                    # Add memory of this interaction
                    if hasattr(self.brain, "add_memory") and hasattr(self.brain, "agent_context"):
                        await self.brain.add_memory(
                            self.brain.agent_context,
                            f"User said: {user_input}\nI responded with: {narrative_response.narrative}",
                            "observation",
                            7
                        )
                    
                    # Convert to dictionary
                    if hasattr(narrative_response, "dict"):
                        response_dict = narrative_response.dict()
                    else:
                        response_dict = {
                            "narrative": narrative_response.narrative,
                            "generate_image": getattr(narrative_response, "generate_image", False),
                            "image_prompt": getattr(narrative_response, "image_prompt", None)
                        }
                    
                    # Create result
                    agent_result = {
                        "success": True,
                        "message": response_dict.get("narrative", ""),
                        "response": response_dict,
                        "has_experience": True,
                        "memory_id": None,
                        "emotional_state": emotional_state,
                        "memories_used": memories if "memories" in locals() else [],
                        "generate_image": response_dict.get("generate_image", False),
                        "image_prompt": response_dict.get("image_prompt"),
                        "user_input": user_input,
                        "response_time": (datetime.datetime.now() - start_time).total_seconds() if "start_time" in locals() else 0
                    }
                    
                    return agent_result
            
            # Fallback to simple response
            return {
                "success": True,
                "message": f"I've processed your message: {user_input}",
                "has_experience": False,
                "user_input": user_input,
                "emotional_state": emotional_state,
                "response_time": (datetime.datetime.now() - start_time).total_seconds() if "start_time" in locals() else 0
            }
                
        except Exception as e:
            logger.error(f"Error processing with agent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "I apologize, but I encountered an error while processing your input.",
                "user_input": user_input
            }
    
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
        
        with trace(workflow_name="generate_response_agent", group_id=self.brain.trace_group_id):
            if self.integration_mode and "brain_processing" in processing_result and "agent_processing" in processing_result:
                # Use integrated processing result
                
                # Determine which message to use based on available experiences
                has_brain_experience = (processing_result["brain_processing"].get("has_experience", False) and 
                                      processing_result["brain_processing"].get("experience_response"))
                agent_has_image = processing_result["agent_processing"].get("generate_image", False)
                agent_message = processing_result["agent_processing"].get("message", "")
                
                # Use brain experience if available, otherwise use agent response
                if has_brain_experience:
                    main_response = processing_result["brain_processing"]["experience_response"]
                    response_type = "experience"
                    
                    # If it's a cross-user experience, mark it
                    if processing_result["brain_processing"].get("cross_user_experience", False):
                        response_type = "cross_user_experience"
                else:
                    main_response = agent_message or "I've processed your message."
                    response_type = "agent"
                
                # Package the response
                response_data = {
                    "message": main_response,
                    "response_type": response_type,
                    "emotional_state": processing_result["brain_processing"].get("emotional_state", {}),
                    "emotional_expression": None,  # Could be generated if needed
                    "memories_used": [],  # Could be populated if needed
                    "memory_count": 0,
                    "experience_sharing_adapted": False
                }
                
                # Add image generation if agent suggests it
                if agent_has_image:
                    response_data["generate_image"] = True
                    response_data["image_prompt"] = processing_result["agent_processing"].get("image_prompt")
                
                return response_data
            
            elif "message" in processing_result:
                # Direct agent processing result
                return {
                    "message": processing_result["message"],
                    "response_type": "agent",
                    "emotional_state": processing_result.get("emotional_state", {}),
                    "emotional_expression": None,
                    "generate_image": processing_result.get("generate_image", False),
                    "image_prompt": processing_result.get("image_prompt")
                }
            
            # Fallback to simple response
            return {
                "message": "I've processed your message.",
                "response_type": "agent_fallback",
                "emotional_state": {},
                "emotional_expression": None
            }
