# nyx/core/brain/processing/reflexive.py
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional

from agents import trace

logger = logging.getLogger(__name__)

class ReflexiveProcessor:
    """Handles reflexive processing of inputs using the reflexive system"""
    
    def __init__(self, brain):
        self.brain = brain
    
    async def initialize(self):
        """Initialize the processor"""
        logger.info("Reflexive processor initialized")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input using the reflexive system for fast reactions
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results
        """
        if not self.brain.initialized:
            await self.brain.initialize()
        
        with trace(workflow_name="process_input_reflexive", group_id=self.brain.trace_group_id):
            start_time = time.time()
            
            # Initialize context
            context = context or {}
            
            # Check for reflexive system
            if not hasattr(self.brain, "reflexive_system") or not self.brain.reflexive_system:
                logger.warning("Reflexive system not available, falling back to serial processing")
                if hasattr(self.brain, "processing_manager") and "serial" in self.brain.processing_manager.processors:
                    return await self.brain.processing_manager.processors["serial"].process_input(user_input, context)
                else:
                    return {"error": "Reflexive system not available", "user_input": user_input}
            
            # Create stimulus from user input
            stimulus = {"text": user_input}
            
            # Extract relevant context features
            if context:
                if "domain" in context:
                    stimulus["domain"] = context["domain"]
                if "urgency" in context:
                    stimulus["urgency"] = context["urgency"]
                if "scenario_type" in context:
                    stimulus["scenario_type"] = context["scenario_type"]
            
            # Start procedural memory lookup in parallel
            procedural_task = None
            if hasattr(self.brain, "agent_enhanced_memory"):
                procedural_task = asyncio.create_task(
                    self.brain.agent_enhanced_memory.find_similar_procedures(user_input)
                )
            
            # Process with reflexes
            reflex_result = await self.brain.reflexive_system.process_stimulus_fast(stimulus)
            
            # Wait for procedural memory lookup to complete if started
            procedural_knowledge = None
            if procedural_task:
                try:
                    relevant_procedures = await procedural_task
                    if relevant_procedures:
                        procedural_knowledge = {
                            "relevant_procedures": relevant_procedures,
                            "can_execute": len(relevant_procedures) > 0
                        }
                except Exception as e:
                    logger.error(f"Error in procedural memory lookup: {str(e)}")
            
            # Track emotional state
            emotional_state = {}
            if hasattr(self.brain, "emotional_core") and self.brain.emotional_core:
                # Quick update to emotional state based on reflex
                if reflex_result["success"]:
                    # Successful reflex might trigger satisfaction
                    self.brain.emotional_core.update_emotion("Satisfaction", 0.1)
                
                emotional_state = self.brain.emotional_core.get_emotional_state()
            
            # Add memory of this interaction if reflexive processing was successful
            memory_id = None
            if reflex_result["success"] and hasattr(self.brain, "memory_core"):
                pattern_name = reflex_result.get("pattern_name", "unknown pattern")
                memory_text = f"User input '{user_input}' triggered reflexive response for pattern '{pattern_name}'"
                
                # Get emotional context formatting if available
                emotional_context = {}
                if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_formatted_emotional_state"):
                    emotional_context = self.brain.emotional_core.get_formatted_emotional_state()
                
                memory_id = await self.brain.memory_core.add_memory(
                    memory_text=memory_text,
                    memory_type="observation",
                    significance=4,  # Lower significance for reflexive responses
                    tags=["interaction", "user_input", "reflexive_response", pattern_name],
                    metadata={
                        "emotional_context": emotional_context,
                        "timestamp": time.time(),
                        "user_id": str(self.brain.user_id),
                        "reflex_pattern": pattern_name,
                        "reaction_time_ms": reflex_result.get("reaction_time_ms", 0)
                    }
                )
            
            # Build result
            result = {
                "user_input": user_input,
                "response_type": "reflexive",
                "reaction_time_ms": reflex_result.get("reaction_time_ms", 0),
                "reflex_pattern": reflex_result.get("pattern_name", "unknown"),
                "success": reflex_result.get("success", False),
                "reflex_result": reflex_result,
                "procedural_knowledge": procedural_knowledge,
                "emotional_state": emotional_state,
                "memory_id": memory_id,
                "total_processing_time_ms": (time.time() - start_time) * 1000
            }
            
            # Update decision system if available
            if hasattr(self.brain.reflexive_system, "decision_system"):
                self.brain.reflexive_system.decision_system.update_from_result(
                    stimulus, True, reflex_result.get("success", False)
                )
            
            # Update interaction tracking
            if hasattr(self.brain, "last_interaction"):
                self.brain.last_interaction = time.time()
            
            if hasattr(self.brain, "interaction_count"):
                self.brain.interaction_count += 1
            
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
        
        with trace(workflow_name="generate_response_reflexive", group_id=self.brain.trace_group_id):
            # Extract the reflex result
            reflex_result = processing_result.get("reflex_result", {})
            
            # Check if reflex was successful
            if processing_result.get("success", False):
                # Check if the reflex result includes a direct response
                if "response" in reflex_result:
                    message = reflex_result["response"]
                elif "output" in reflex_result and isinstance(reflex_result["output"], dict) and "response" in reflex_result["output"]:
                    message = reflex_result["output"]["response"]
                else:
                    # Use a generic response indicating a reflexive action
                    message = f"I've reacted to your input reflexively."
                
                response_type = "reflexive"
            else:
                # Fallback to procedural or standard response if reflex failed
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
                            message = procedure_result["output"]
                            response_type = "procedural"
                        else:
                            # Fallback to standard response
                            message = "I understand your input and have processed it, but couldn't react reflexively."
                            response_type = "standard"
                    except Exception as e:
                        logger.error(f"Error executing procedure: {str(e)}")
                        message = "I understand your input but couldn't process it reflexively."
                        response_type = "standard"
                else:
                    # Standard fallback
                    message = "I understand your input but couldn't process it reflexively."
                    response_type = "standard"
            
            # Add memory of this response
            response_memory_id = None
            if hasattr(self.brain, "memory_core"):
                memory_text = f"I responded reflexively: {message}"
                
                # Get emotional context formatting if available
                emotional_context = {}
                if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_formatted_emotional_state"):
                    emotional_context = self.brain.emotional_core.get_formatted_emotional_state()
                
                response_memory_id = await self.brain.memory_core.add_memory(
                    memory_text=memory_text,
                    memory_type="observation",
                    significance=4,  # Lower significance for reflexive responses
                    tags=["interaction", "nyx_response", response_type],
                    metadata={
                        "emotional_context": emotional_context,
                        "timestamp": time.time(),
                        "response_type": response_type,
                        "user_id": str(self.brain.user_id),
                        "reaction_time_ms": processing_result.get("reaction_time_ms", 0)
                    }
                )
            
            # Package the response
            response_data = {
                "message": message,
                "response_type": response_type,
                "emotional_state": processing_result.get("emotional_state", {}),
                "emotional_expression": None,  # Reflexive responses typically don't have emotional expressions
                "reflex_pattern": processing_result.get("reflex_pattern"),
                "reaction_time_ms": processing_result.get("reaction_time_ms", 0),
                "reflexive_processing": True,
                "response_memory_id": response_memory_id
            }
            
            return response_data
