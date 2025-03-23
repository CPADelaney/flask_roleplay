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
                                  enable_periodic_tasks: bool = True) -> StreamingCore:
    """
    Set up fully integrated streaming system for Nyx
    
    This function creates a streaming system deeply integrated with all of Nyx's
    core capabilities including memory, reflection, emotion, reasoning, and experience.
    
    Args:
        brain: Instance of NyxBrain
        video_source: Video capture source
        audio_source: Audio capture source
        enable_periodic_tasks: Whether to enable periodic tasks
        
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
    
    # 4. Enable periodic tasks if requested
    if enable_periodic_tasks:
        task = asyncio.create_task(_run_streaming_periodic_tasks(brain, streaming_core))
        streaming_core._periodic_task = task
        logger.info("Periodic streaming tasks enabled")
    
    # 5. Register streaming experience retrieval in brain
    if not hasattr(brain, "get_streaming_experiences"):
        brain.get_streaming_experiences = streaming_core.memory_mapper.retrieve_game_experiences
    
    # 6. Register streaming reflection in brain
    if not hasattr(brain, "create_streaming_reflection"):
        brain.create_streaming_reflection = streaming_core.memory_mapper.create_streaming_reflection
    
    # 7. Add methods for processing significant moments
    if not hasattr(streaming_core, "process_significant_moment"):
        streaming_core.process_significant_moment = streaming_core.__class__.process_significant_moment
    
    logger.info(f"Integrated streaming setup complete for user {brain.user_id}")
    
    return streaming_core

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
