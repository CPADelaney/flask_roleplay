# nyx/streamer/integration.py

import asyncio
import logging
from typing import Dict, Any, Optional

from nyx.core.nyx_brain import NyxBrain
from nyx.streamer.nyx_streaming_core import integrate_with_nyx_brain, StreamingCore
from nyx.streamer.streaming_hormone_system import StreamingHormoneIntegration
from nyx.streamer.streaming_reflection import StreamingIntegration

logger = logging.getLogger("nyx_streaming_integration")

async def setup_integrated_streaming(brain: NyxBrain, 
                                  video_source=0, 
                                  audio_source=None,
                                  enable_experience_integration: bool = True,
                                  enable_reasoning_integration: bool = True) -> StreamingCore:
    """
    Set up fully integrated streaming system for Nyx
    
    This function creates a streaming system deeply integrated with all of Nyx's
    core capabilities including memory, reflection, emotion, reasoning, and experience.
    
    Args:
        brain: Instance of NyxBrain
        video_source: Video capture source
        audio_source: Audio capture source
        enable_experience_integration: Whether to enable experience system integration
        enable_reasoning_integration: Whether to enable reasoning system integration
        
    Returns:
        Integrated StreamingCore instance
    """
    logger.info(f"Setting up integrated streaming for user {brain.user_id}")
    
    # 1. Base integration with Nyx brain
    streaming_core = await integrate_with_nyx_brain(brain, video_source, audio_source)
    
    # 2. Add hormone system integration
    hormone_result = await StreamingHormoneIntegration.integrate(brain, streaming_core)
    logger.info(f"Hormone integration status: {hormone_result['status']}")
    
    # 3. Add reflection engine integration
    reflection_result = await StreamingIntegration.integrate(brain, streaming_core)
    logger.info(f"Reflection integration status: {reflection_result['status']}")
    
    # 4. Add experience system integration
    if enable_experience_integration and hasattr(brain, "experience_interface"):
        experience_result = await _integrate_experience_system(brain, streaming_core)
        logger.info(f"Experience integration status: {experience_result['status']}")
    
    # 5. Add reasoning system integration
    if enable_reasoning_integration and hasattr(brain, "reasoning_core"):
        reasoning_result = await _integrate_reasoning_system(brain, streaming_core)
        logger.info(f"Reasoning integration status: {reasoning_result['status']}")
    
    # 6. Enable periodic tasks
    task = asyncio.create_task(_run_streaming_periodic_tasks(brain, streaming_core))
    streaming_core._periodic_task = task
    logger.info("Periodic streaming tasks enabled")
    
    # 7. Register necessary functions in brain
    _register_brain_functions(brain, streaming_core)
    
    logger.info(f"Integrated streaming setup complete for user {brain.user_id}")
    
    return streaming_core

async def _integrate_experience_system(brain: NyxBrain, streaming_core: StreamingCore) -> Dict[str, Any]:
    """
    Integrate streaming with Nyx's experience system
    
    Args:
        brain: NyxBrain instance
        streaming_core: StreamingCore instance
        
    Returns:
        Integration status
    """
    if not hasattr(brain, "experience_interface"):
        return {"status": "experience_interface_unavailable"}
    
    # Create an enhanced store_streaming_experience method
    async def enhanced_store_streaming_experience(
        game_name: str,
        moment_data: Dict[str, Any],
        emotional_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # Create experience memory
        memory_text = f"While streaming {game_name}, I experienced: {moment_data.get('description', '')}"
        
        # Prepare metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "game_name": game_name,
            "moment_data": moment_data,
            "streaming": True,
            "scenario_type": "gaming"
        }
        
        # Add emotional context if provided
        if emotional_context:
            metadata["emotional_context"] = emotional_context
        
        # Prepare tags
        tags = ["streaming", "experience", game_name]
        if "tags" in moment_data:
            tags.extend(moment_data["tags"])
        
        # First store in memory system
        memory_id = await brain.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="experience",
            memory_scope="game",
            significance=8.0,  # Higher significance for experiences
            tags=tags,
            metadata=metadata
        )
        
        # Then store in experience system for richer access
        try:
            exp_result = await brain.experience_interface.store_experience(
                text=memory_text,
                scenario_type="gaming",
                entities=[game_name],
                emotional_context=emotional_context or {},
                significance=8.0,
                tags=tags,
                user_id=str(brain.user_id)
            )
            
            return {
                "memory_id": memory_id,
                "experience_id": exp_result.get("id"),
                "stored": True
            }
        except Exception as e:
            logger.error(f"Error storing in experience interface: {e}")
            return {
                "memory_id": memory_id,
                "stored": True,
                "experience_error": str(e)
            }
    
    # Create an enhanced retrieve_streaming_experience method
    async def enhanced_retrieve_streaming_experience(
        query: str,
        game_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # Prepare full query
        full_query = f"streaming {query}"
        if game_name:
            full_query = f"streaming {game_name} {query}"
        
        # Try to retrieve from experience interface first
        try:
            exp_result = await brain.experience_interface.recall_experience(
                query=full_query,
                scenario_type="gaming",
                confidence_threshold=0.6
            )
            
            if exp_result and exp_result.get("has_experience", False):
                return {
                    "text": exp_result.get("text", ""),
                    "confidence": exp_result.get("confidence", 0.0),
                    "source": "experience_interface"
                }
        except Exception as e:
            logger.error(f"Error retrieving from experience interface: {e}")
        
        # Fall back to memory system if needed
        try:
            memories = await brain.memory_core.retrieve_memories(
                query=full_query,
                memory_types=["experience"],
                limit=1,
                min_significance=5
            )
            
            if memories:
                return {
                    "text": memories[0]["memory_text"],
                    "confidence": memories[0].get("metadata", {}).get("confidence", 0.7),
                    "source": "memory_core"
                }
        except Exception as e:
            logger.error(f"Error retrieving from memory core: {e}")
        
        return {
            "text": "",
            "confidence": 0.0,
            "has_experience": False
        }
    
    # Add methods to the streaming core
    streaming_core.store_streaming_experience = enhanced_store_streaming_experience
    streaming_core.retrieve_streaming_experience = enhanced_retrieve_streaming_experience
    
    # Enhanced process_significant_moment to use experience system
    original_process = streaming_core.process_significant_moment
    
    async def enhanced_process_significant_moment(
        self,
        game_name: str,
        event_type: str,
        event_data: Dict[str, Any],
        significance: float = 7.0
    ) -> Dict[str, Any]:
        # Run original processing
        results = await original_process(
            self,
            game_name=game_name,
            event_type=event_type,
            event_data=event_data,
            significance=significance
        )
        
        # Additional experience system integration for highly significant moments
        if significance >= 8.0 and hasattr(brain, "experience_interface"):
            # Get emotional context if available
            emotional_context = {}
            if hasattr(self, "hormone_system"):
                emotional_context = self.hormone_system.get_emotional_state()
            
            # Store in experience system
            experience_result = await enhanced_store_streaming_experience(
                game_name=game_name,
                moment_data={
                    "type": event_type,
                    "description": event_data.get("description", ""),
                    "details": event_data
                },
                emotional_context=emotional_context
            )
            
            results["experience_system_stored"] = True
            results["experience_result"] = experience_result
        
        return results
    
    # Replace method with enhanced version
    streaming_core.process_significant_moment = types.MethodType(
        enhanced_process_significant_moment, streaming_core)
    
    return {
        "status": "integrated",
        "components": {
            "store_experience": True,
            "retrieve_experience": True,
            "enhanced_processing": True
        }
    }

async def _integrate_reasoning_system(brain: NyxBrain, streaming_core: StreamingCore) -> Dict[str, Any]:
    """
    Integrate streaming with Nyx's reasoning system
    
    Args:
        brain: NyxBrain instance
        streaming_core: StreamingCore instance
        
    Returns:
        Integration status
    """
    if not hasattr(brain, "reasoning_core"):
        return {"status": "reasoning_core_unavailable"}
    
    # Add method to apply reasoning to streaming events
    async def reason_about_streaming_event(
        game_name: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Prepare reasoning context
            context = {
                "domain": "streaming",
                "game_name": game_name,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Apply reasoning
            reasoning_result = await brain.reasoning_core.analyze_event(
                event_data=event_data,
                context=context
            )
            
            # Store reasoning as memory if significant
            if reasoning_result and reasoning_result.get("significance", 0) >= 6:
                memory_text = f"While streaming {game_name}, I reasoned about {event_type}: {reasoning_result.get('conclusion', '')}"
                
                memory_id = await brain.memory_core.add_memory(
                    memory_text=memory_text,
                    memory_type="reflection",
                    memory_scope="game",
                    significance=reasoning_result.get("significance", 6),
                    tags=["streaming", "reasoning", game_name, event_type],
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "game_name": game_name,
                        "event_type": event_type,
                        "reasoning_process": reasoning_result.get("process", {}),
                        "streaming": True
                    }
                )
                
                reasoning_result["memory_id"] = memory_id
            
            return reasoning_result
        
        except Exception as e:
            logger.error(f"Error applying reasoning to streaming event: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    # Add method to streaming core
    streaming_core.reason_about_streaming_event = reason_about_streaming_event
    
    # Enhance process_significant_moment to use reasoning system
    original_process = streaming_core.process_significant_moment
    
    async def enhanced_process_significant_moment(
        self,
        game_name: str,
        event_type: str,
        event_data: Dict[str, Any],
        significance: float = 7.0
    ) -> Dict[str, Any]:
        # Run original processing
        results = await original_process(
            self,
            game_name=game_name,
            event_type=event_type,
            event_data=event_data,
            significance=significance
        )
        
        # Apply reasoning for significant events that might need deeper analysis
        if significance >= 7.0 or event_type in ["unexpected_event", "decision_point", "complex_situation"]:
            reasoning_result = await reason_about_streaming_event(
                game_name=game_name,
                event_type=event_type,
                event_data=event_data
            )
            
            results["reasoning_applied"] = True
            results["reasoning_result"] = reasoning_result
        
        return results
    
    # Replace method with enhanced version
    streaming_core.process_significant_moment = types.MethodType(
        enhanced_process_significant_moment, streaming_core)
    
    return {
        "status": "integrated",
        "components": {
            "event_reasoning": True,
            "enhanced_processing": True
        }
    }

def _register_brain_functions(brain: NyxBrain, streaming_core: StreamingCore) -> None:
    """
    Register streaming functions in the brain for easy access
    
    Args:
        brain: NyxBrain instance
        streaming_core: StreamingCore instance
    """
    # Register streaming capabilities
    if not hasattr(brain, "stream"):
        brain.stream = streaming_core.start_streaming
    
    if not hasattr(brain, "stop_stream"):
        brain.stop_stream = streaming_core.stop_streaming
    
    if not hasattr(brain, "get_stream_stats"):
        brain.get_stream_stats = streaming_core.get_streaming_stats
    
    # Register experience access
    if not hasattr(brain, "retrieve_streaming_experience"):
        brain.retrieve_streaming_experience = streaming_core.retrieve_streaming_experience
    
    # Register memory creation
    if not hasattr(brain, "create_streaming_memory"):
        brain.create_streaming_memory = streaming_core.memory_mapper.store_gameplay_memory
    
    # Register reflection creation
    if not hasattr(brain, "create_streaming_reflection"):
        brain.create_streaming_reflection = streaming_core.memory_mapper.create_streaming_reflection
    
    # Register reasoning if available
    if hasattr(streaming_core, "reason_about_streaming_event"):
        brain.reason_about_streaming_event = streaming_core.reason_about_streaming_event

async def _run_streaming_periodic_tasks(brain: NyxBrain, streaming_core: StreamingCore):
    """
    Run periodic tasks for fully integrated streaming
    
    Args:
        brain: NyxBrain instance
        streaming_core: StreamingCore instance
    """
    try:
        while True:
            # Only run tasks if a streaming session is active
            if streaming_core.session_start_time is not None:
                logger.info("Running streaming periodic tasks")
                
                # 1. Run streaming reflection
                if hasattr(streaming_core, "reflection_engine"):
                    reflection_result = await streaming_core.reflection_engine.run_periodic_reflection()
                    if reflection_result:
                        logger.info(f"Created streaming reflection: {reflection_result.get('reflection_id', 'unknown')}")
                
                # 2. Sync hormone system with brain
                if hasattr(streaming_core, "hormone_system"):
                    hormone_result = streaming_core.hormone_system.sync_with_brain_hormone_system()
                    if hormone_result.get("synced", False):
                        logger.info(f"Synchronized hormone states: {len(hormone_result.get('differences', {}))} changes")
                
                # 3. Run experience consolidation
                if (hasattr(streaming_core, "reflection_engine") and 
                    hasattr(streaming_core.reflection_engine, "consolidate_streaming_experiences")):
                    consolidation = await streaming_core.reflection_engine.consolidate_streaming_experiences()
                    if consolidation.get("status") == "success":
                        logger.info(f"Consolidated streaming experiences: {consolidation.get('abstractions_created', 0)} abstractions")
                
                # 4. Run meta-cognitive cycle if available
                if hasattr(brain, "meta_core") and brain.meta_core:
                    meta_result = await brain.meta_core.cognitive_cycle({
                        "streaming": True,
                        "current_game": streaming_core.streaming_system.game_state.game_name
                    })
                    logger.info("Ran meta-cognitive cycle for streaming")
                
                # 5. Apply reasoning to recent significant events
                if hasattr(streaming_core, "reason_about_streaming_event"):
                    recent_events = streaming_core.streaming_system.game_state.recent_events
                    significant_events = [
                        event for event in recent_events
                        if event.get("data", {}).get("significance", 0) >= 7.0
                        and "reasoning_applied" not in event.get("data", {})
                    ]
                    
                    if significant_events:
                        event = significant_events[0]  # Process one event per cycle
                        game_name = streaming_core.streaming_system.game_state.game_name
                        
                        await streaming_core.reason_about_streaming_event(
                            game_name=game_name,
                            event_type=event["type"],
                            event_data=event["data"]
                        )
                        
                        # Mark as processed
                        event["data"]["reasoning_applied"] = True
                        logger.info(f"Applied reasoning to significant event: {event['type']}")
            
            # Run every 15 minutes
            await asyncio.sleep(15 * 60)
            
    except asyncio.CancelledError:
        logger.info("Streaming periodic tasks cancelled")
    except Exception as e:
        logger.error(f"Error in streaming periodic tasks: {e}")

async def _run_streaming_periodic_tasks(brain: NyxBrain, streaming_core: StreamingCore):
    """
    Run periodic tasks for fully integrated streaming
    
    Args:
        brain: NyxBrain instance
        streaming_core: StreamingCore instance
    """
    try:
        while True:
            # Only run tasks if a streaming session is active
            if streaming_core.session_start_time is not None:
                logger.info("Running streaming periodic tasks")
                
                # 1. Run streaming reflection
                if hasattr(streaming_core, "reflection_engine"):
                    reflection_result = await streaming_core.reflection_engine.run_periodic_reflection()
                    if reflection_result:
                        logger.info(f"Created streaming reflection: {reflection_result.get('reflection_id', 'unknown')}")
                
                # 2. Sync hormone system with brain
                if hasattr(streaming_core, "hormone_system"):
                    hormone_result = streaming_core.hormone_system.sync_with_brain_hormone_system()
                    if hormone_result.get("synced", False):
                        logger.info(f"Synchronized hormone states: {len(hormone_result.get('differences', {}))} changes")
                
                # 3. Run experience consolidation
                if (hasattr(streaming_core, "reflection_engine") and 
                    hasattr(streaming_core.reflection_engine, "consolidate_streaming_experiences")):
                    consolidation = await streaming_core.reflection_engine.consolidate_streaming_experiences()
                    if consolidation.get("status") == "success":
                        logger.info(f"Consolidated streaming experiences: {consolidation.get('abstractions_created', 0)} abstractions")
                
                # 4. Run meta-cognitive cycle if available
                if hasattr(brain, "meta_core") and brain.meta_core:
                    meta_result = await brain.meta_core.cognitive_cycle({
                        "streaming": True,
                        "current_game": streaming_core.streaming_system.game_state.game_name
                    })
                    logger.info("Ran meta-cognitive cycle for streaming")
            
            # Run every 15 minutes
            await asyncio.sleep(15 * 60)
            
    except asyncio.CancelledError:
        logger.info("Streaming periodic tasks cancelled")
    except Exception as e:
        logger.error(f"Error in streaming periodic tasks: {e}")
