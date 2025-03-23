# nyx/streamer/integration.py

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from nyx.core.nyx_brain import NyxBrain
from nyx.streamer.nyx_streaming_core import integrate_with_nyx_brain, StreamingCore
from nyx.streamer.streaming_hormone_system import StreamingHormoneIntegration
from nyx.streamer.streaming_reflection import StreamingIntegration

# Import from OpenAI Agents SDK
from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff
)

logger = logging.getLogger("nyx_streaming_integration")

async def setup_enhanced_streaming(brain: NyxBrain, 
                                video_source=0, 
                                audio_source=None) -> StreamingCore:
    """
    Set up deeply integrated streaming system utilizing all of Nyx's cognitive systems
    
    Args:
        brain: NyxBrain instance
        video_source: Video source
        audio_source: Audio source
        
    Returns:
        Fully integrated StreamingCore
    """
    # 1. First create the base streaming core
    streaming_core = await integrate_with_nyx_brain(brain, video_source, audio_source)
    
    # 2. Add hormone system integration
    await StreamingHormoneIntegration.integrate(brain, streaming_core)
    
    # 3. Add reflection engine integration
    await StreamingIntegration.integrate(brain, streaming_core)
    
    # 4. Enhance with reasoning system integration
    streaming_core = await enhance_with_reasoning(brain, streaming_core)
    
    # 5. Enhance with experience system integration
    streaming_core = await enhance_with_experience(brain, streaming_core)
    
    # 6. Connect to meta-cognitive system for ongoing integration
    streaming_core = await connect_to_metacognition(brain, streaming_core)
    
    # 7. Set up bi-directional identity influence
    streaming_core = await setup_identity_integration(brain, streaming_core)
    
    # 8. Enable periodic cross-system tasks
    task = asyncio.create_task(_run_enhanced_periodic_tasks(brain, streaming_core))
    streaming_core._integrated_task = task
    
    # 9. Register functions in the brain
    _register_brain_functions(brain, streaming_core)
    
    return streaming_core

async def enhance_with_reasoning(brain: NyxBrain, streaming_core: StreamingCore):
    """Enhance streaming with reasoning capabilities"""
    if not hasattr(brain, "reasoning_core"):
        logger.info("Reasoning core not available for integration")
        return streaming_core
    
    # Add reasoning method to streaming core
    async def reason_about_game_event(
        game_name: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply reasoning to game events"""
        # Create reasoning context
        context = {
            "domain": "gaming",
            "game_name": game_name,
            "event_type": event_type
        }
        
        # Use integrated reasoning agent
        result = await Runner.run(
            brain.reasoning_core.integrated_reasoning_agent,
            json.dumps(event_data),
            context=context
        )
        
        # Store reasoning result as memory
        memory_text = f"While streaming {game_name}, I reasoned about {event_type}: {result.final_output}"
        
        memory_id = await brain.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="reflection",
            memory_scope="game",
            significance=7.0,
            tags=["streaming", "reasoning", game_name],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "game_name": game_name,
                "event_type": event_type,
                "reasoning_process": result.trace,
                "streaming": True
            }
        )
        
        return {
            "reasoning": result.final_output,
            "memory_id": memory_id
        }
    
    # Replace the existing _process_game_frame with enhanced version
    original_process = streaming_core.streaming_system._process_game_frame
    
    async def reasoning_enhanced_game_frame():
        # Run original processing
        await original_process()
        
        # Apply reasoning to significant events
        game_state = streaming_core.streaming_system.game_state
        if game_state.game_id and game_state.recent_events:
            for event in game_state.recent_events:
                # Only reason about significant or complex events
                if event.get("data", {}).get("significance", 0) >= 7.0:
                    await reason_about_game_event(
                        game_name=game_state.game_name,
                        event_type=event["type"],
                        event_data=event["data"]
                    )
    
    # Add method to streaming core
    streaming_core.reason_about_game_event = reason_about_game_event
    streaming_core.streaming_system._process_game_frame = reasoning_enhanced_game_frame
    
    return streaming_core

async def enhance_with_experience(brain: NyxBrain, streaming_core: StreamingCore):
    """Enhance streaming with experience system integration"""
    if not hasattr(brain, "experience_interface"):
        logger.info("Experience interface not available for integration")
        return streaming_core
    
    # Create enhanced methods for experience storage and retrieval
    async def store_game_experience(
        game_name: str,
        event_type: str,
        event_data: Dict[str, Any],
        emotional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a game event as an experience in the experience system"""
        # Format description
        description = event_data.get("description", "")
        if not description and "text" in event_data:
            description = event_data["text"]
            
        experience_text = f"While streaming {game_name}, I experienced: {description}"
        
        # First create memory
        memory_id = await brain.memory_core.add_memory(
            memory_text=experience_text,
            memory_type="experience",
            memory_scope="game",
            significance=8.0,
            tags=["streaming", "experience", game_name, event_type],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "game_name": game_name,
                "event_type": event_type,
                "event_data": event_data,
                "emotional_context": emotional_context,
                "streaming": True
            }
        )
        
        # Then store in experience system
        scenario_type = "gaming"
        if event_type == "cross_game_insight":
            scenario_type = "analysis"
        
        try:
            experience_result = await brain.experience_interface.store_experience(
                text=experience_text,
                scenario_type=scenario_type,
                entities=[game_name],
                emotional_context=emotional_context or {},
                significance=8.0,
                tags=["streaming", game_name, event_type],
                user_id=str(brain.user_id)
            )
            
            return {
                "memory_id": memory_id,
                "experience_id": experience_result.get("id"),
                "stored": True
            }
        except Exception as e:
            logger.error(f"Error storing in experience interface: {e}")
            return {
                "memory_id": memory_id,
                "stored": True,
                "experience_error": str(e)
            }
    
    async def recall_game_experience(
        game_name: str,
        context: str
    ) -> Dict[str, Any]:
        """Recall a relevant gaming experience"""
        # Try experience interface first
        try:
            recall_result = await brain.experience_interface.recall_experience(
                query=f"streaming {game_name} {context}",
                scenario_type="gaming",
                confidence_threshold=0.6
            )
            
            if recall_result and recall_result.get("has_experience", False):
                return {
                    "text": recall_result.get("text", ""),
                    "confidence": recall_result.get("confidence", 0.0),
                    "source": "experience_interface"
                }
        except Exception as e:
            logger.error(f"Error recalling from experience interface: {e}")
        
        # Fall back to memory retrieval
        memories = await brain.memory_core.retrieve_memories(
            query=f"streaming {game_name} {context}",
            memory_types=["experience"],
            limit=1,
            min_significance=6.0
        )
        
        if memories:
            return {
                "text": memories[0]["memory_text"],
                "confidence": memories[0].get("relevance", 0.7),
                "source": "memory_core"
            }
            
        return {
            "text": "",
            "confidence": 0.0,
            "has_experience": False
        }
    
    # Add methods to streaming core
    streaming_core.store_game_experience = store_game_experience
    streaming_core.recall_game_experience = recall_game_experience
    
    # Enhance the process_significant_moment to use experience system
    original_process = streaming_core.process_significant_moment
    
    async def experience_enhanced_process_moment(
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
        
        # Store in experience system for significant events
        if significance >= 7.5:
            # Get emotional context
            emotional_context = None
            if hasattr(self, "hormone_system"):
                emotional_context = self.hormone_system.get_emotional_state()
                
            # Store experience
            experience_result = await store_game_experience(
                game_name=game_name,
                event_type=event_type,
                event_data=event_data,
                emotional_context=emotional_context
            )
            
            results["experience_stored"] = True
            results["experience_result"] = experience_result
            
        return results
    
    # Replace with enhanced version using types.MethodType
    import types
    streaming_core.process_significant_moment = types.MethodType(
        experience_enhanced_process_moment, streaming_core)
    
    # Add experience consolidation to the streaming reflection engine
    if hasattr(streaming_core, "reflection_engine"):
        async def consolidate_streaming_experiences():
            # Get all streaming experiences
            experiences = await brain.memory_core.retrieve_memories(
                query="streaming",
                memory_types=["experience"],
                limit=20,
                min_significance=6.0
            )
            
            if len(experiences) < 3:
                return {"status": "not_enough_experiences"}
                
            # Use experience consolidation system
            if hasattr(brain, "experience_consolidation"):
                result = await brain.experience_consolidation.consolidate_experiences(
                    experiences=experiences,
                    topic="streaming experiences",
                    min_count=3
                )
                
                return {
                    "status": "success",
                    "abstractions_created": result.get("abstractions_created", 0),
                    "result": result
                }
                
            return {"status": "consolidation_unavailable"}
            
        streaming_core.reflection_engine.consolidate_streaming_experiences = consolidate_streaming_experiences
    
    return streaming_core

async def connect_to_metacognition(brain: NyxBrain, streaming_core: StreamingCore):
    """Connect streaming to meta-cognitive system"""
    if not hasattr(brain, "meta_core"):
        logger.info("Meta core not available for integration")
        return streaming_core
    
    # Create function to run meta-cognitive cycle with streaming context
    async def run_metacognitive_cycle(game_state):
        """Run meta-cognitive cycle with streaming context"""
        # Prepare streaming context for meta core
        streaming_context = {
            "streaming": True,
            "game_name": game_state.game_name,
            "game_state": {
                "current_location": game_state.current_location,
                "detected_action": game_state.detected_action,
                "player_status": game_state.player_status
            },
            "streaming_stats": {
                "commentary_count": game_state.session_stats.get("commentary_count", 0),
                "questions_answered": game_state.session_stats.get("questions_answered", 0),
                "session_duration": (datetime.now() - game_state.session_start_time).total_seconds() 
                                    if game_state.session_start_time else 0
            }
        }
        
        # Run meta-cognitive cycle
        result = await brain.meta_core.cognitive_cycle(streaming_context)
        
        return result
    
    # Add periodic meta-cognitive processing
    original_process = streaming_core.streaming_system._process_game_frame
    
    async def metacognitive_enhanced_frame():
        # Run original processing
        await original_process()
        
        # Run meta-cognitive cycle periodically
        game_state = streaming_core.streaming_system.game_state
        if game_state.game_id and game_state.frame_count % 900 == 0:  # Every ~30 seconds at 30fps
            await run_metacognitive_cycle(game_state)
    
    # Replace with enhanced version
    streaming_core.streaming_system._process_game_frame = metacognitive_enhanced_frame
    streaming_core.run_metacognitive_cycle = run_metacognitive_cycle
    
    return streaming_core

async def setup_identity_integration(brain: NyxBrain, streaming_core: StreamingCore):
    """Connect streaming to identity evolution system"""
    if not hasattr(brain, "identity_evolution"):
        logger.info("Identity evolution not available for integration")
        return streaming_core
    
    # Create function to update identity based on streaming
    async def update_identity_from_streaming(
        game_name: str,
        streaming_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update identity based on streaming experiences"""
        # Extract potential identity impacts
        preferences = {}
        traits = {}
        
        # Game preferences
        if "game_genre" in streaming_data and streaming_data["game_genre"]:
            genre = streaming_data["game_genre"]
            preferences["genre_preferences"] = {
                genre: 0.1  # Small increase in preference for this genre
            }
        
        # Streaming activity preference
        if "session_duration" in streaming_data and streaming_data["session_duration"] > 1800:  # 30+ minutes
            preferences["activity_preferences"] = {
                "streaming": 0.1  # Small increase in streaming preference
            }
        
        # Commentary style traits
        if "commentary_style" in streaming_data:
            style = streaming_data["commentary_style"]
            if style == "analytical":
                traits["analytical"] = 0.1
            elif style == "humorous":
                traits["humorous"] = 0.1
            elif style == "educational":
                traits["informative"] = 0.1
        
        # Impact based on audience interaction
        if streaming_data.get("questions_answered", 0) > 5:
            traits["helpful"] = 0.1
            
        # Create impact
        impact = {
            "preferences": preferences,
            "traits": traits
        }
        
        # Only update if we have meaningful impacts
        if (preferences or traits) and hasattr(brain, "identity_evolution"):
            try:
                result = await brain.identity_evolution.update_identity_from_experience(
                    experience={
                        "type": "streaming",
                        "game_name": game_name,
                        "streaming_data": streaming_data
                    },
                    impact=impact
                )
                
                return {
                    "updated": True,
                    "preferences_updated": list(preferences.keys()),
                    "traits_updated": list(traits.keys())
                }
            except Exception as e:
                logger.error(f"Error updating identity: {e}")
                return {"error": str(e)}
        
        return {"updated": False, "reason": "no_significant_impact"}
    
    # Add method to streaming core
    streaming_core.update_identity_from_streaming = update_identity_from_streaming
    
    # Enhance session end handling to update identity
    original_stop = streaming_core.stop_streaming
    
    async def identity_enhanced_stop_streaming():
        # Get results from original method
        results = await original_stop()
        
        # Update identity based on session
        if results["status"] == "streaming_stopped" and "stats" in results:
            stats = results["stats"]
            game_names = stats.get("games_played", [])
            
            if game_names:
                identity_result = await update_identity_from_streaming(
                    game_name=", ".join(game_names),
                    streaming_data=stats
                )
                
                results["identity_updated"] = identity_result["updated"]
        
        return results
    
    # Replace with enhanced version
    streaming_core.stop_streaming = identity_enhanced_stop_streaming
    
    return streaming_core

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
    
    if not hasattr(brain, "add_stream_question"):
        brain.add_stream_question = streaming_core.add_audience_question
    
    if not hasattr(brain, "get_stream_stats"):
        brain.get_stream_stats = streaming_core.get_streaming_stats
    
    # Register experience access
    if not hasattr(brain, "retrieve_streaming_experience"):
        brain.retrieve_streaming_experience = streaming_core.recall_streaming_experience
    
    # Register memory creation
    if not hasattr(brain, "create_streaming_memory"):
        brain.create_streaming_memory = streaming_core.memory_mapper.store_gameplay_memory
    
    # Register reflection creation
    if not hasattr(brain, "create_streaming_reflection"):
        brain.create_streaming_reflection = streaming_core.memory_mapper.create_streaming_reflection
    
    # Register reasoning if available
    if hasattr(streaming_core, "reason_about_streaming_event"):
        brain.reason_about_streaming_event = streaming_core.reason_about_streaming_event

async def _run_enhanced_periodic_tasks(brain: NyxBrain, streaming_core: StreamingCore):
    """Run periodic tasks for enhanced integration"""
    try:
        while True:
            # Only run tasks if streaming is active
            if streaming_core.session_start_time is not None:
                logger.info("Running enhanced periodic streaming tasks")
                
                # 1. Run streaming reflection
                if hasattr(streaming_core, "reflection_engine"):
                    reflection_result = await streaming_core.reflection_engine.run_periodic_reflection()
                    if reflection_result:
                        logger.info(f"Periodic streaming reflection created")
                
                # 2. Consolidate streaming experiences
                if (hasattr(streaming_core, "reflection_engine") and 
                    hasattr(streaming_core.reflection_engine, "consolidate_streaming_experiences")):
                    consolidation = await streaming_core.reflection_engine.consolidate_streaming_experiences()
                    if consolidation.get("status") == "success":
                        logger.info(f"Consolidated streaming experiences")
                
                # 3. Run meta-cognitive cycle
                if hasattr(streaming_core, "run_metacognitive_cycle"):
                    meta_result = await streaming_core.run_metacognitive_cycle(
                        streaming_core.streaming_system.game_state
                    )
                    logger.info("Ran meta-cognitive cycle for streaming")
                
                # 4. Sync identity if needed
                if (hasattr(streaming_core, "update_identity_from_streaming") and 
                    streaming_core.streaming_system.game_state.frame_count % 3000 == 0):  # Every ~100 seconds
                    
                    game_state = streaming_core.streaming_system.game_state
                    game_name = game_state.game_name or "Unknown Game"
                    
                    # Update identity periodically
                    streaming_data = {
                        "commentary_count": game_state.session_stats.get("commentary_count", 0),
                        "questions_answered": game_state.session_stats.get("questions_answered", 0),
                        "commentary_style": "analytical",  # Would be determined dynamically in real implementation
                        "session_duration": (datetime.now() - game_state.session_start_time).total_seconds() 
                                           if game_state.session_start_time else 0
                    }
                    
                    identity_result = await streaming_core.update_identity_from_streaming(
                        game_name=game_name,
                        streaming_data=streaming_data
                    )
                    
                    if identity_result.get("updated", False):
                        logger.info("Updated identity based on streaming")
                
                # 5. Apply reasoning to recent significant events
                if hasattr(streaming_core, "reason_about_game_event"):
                    game_state = streaming_core.streaming_system.game_state
                    recent_events = game_state.recent_events
                    
                    significant_events = [
                        event for event in recent_events
                        if event.get("data", {}).get("significance", 0) >= 7.0 and
                        not event.get("data", {}).get("reasoning_applied", False)
                    ]
                    
                    if significant_events:
                        # Process one significant event
                        event = significant_events[0]
                        await streaming_core.reason_about_game_event(
                            game_name=game_state.game_name or "Unknown Game",
                            event_type=event["type"],
                            event_data=event["data"]
                        )
                        
                        # Mark as processed
                        event["data"]["reasoning_applied"] = True
                        logger.info(f"Applied reasoning to significant event")
            
            # Run every 10 minutes
            await asyncio.sleep(600)
    except asyncio.CancelledError:
        logger.info("Enhanced periodic tasks cancelled")
    except Exception as e:
        logger.error(f"Error in enhanced periodic tasks: {e}")
async def setup_enhanced_streaming_v2(brain: NyxBrain, 
                                    video_source=0, 
                                    audio_source=None) -> StreamingCore:
    """
    Set up deeply integrated streaming system with all enhancements
    
    Args:
        brain: NyxBrain instance
        video_source: Video source
        audio_source: Audio source
        
    Returns:
        Fully integrated StreamingCore
    """
    # 1. Create the optimized streaming core
    streaming_core = await integrate_with_nyx_brain(brain, video_source, audio_source)
    
    # 2. Add hormone system integration
    await StreamingHormoneIntegration.integrate(brain, streaming_core)
    
    # 3. Add enhanced reflection engine integration
    await StreamingIntegration.integrate(brain, streaming_core)
    
    # 4. Add cross-game knowledge integration
    await CrossGameKnowledgeIntegration.integrate(brain, streaming_core)
    
    # 5. Enhance with reasoning system integration
    streaming_core = await enhance_with_reasoning(brain, streaming_core)
    
    # 6. Enhance with experience system integration
    streaming_core = await enhance_with_experience(brain, streaming_core)
    
    # 7. Connect to meta-cognitive system for ongoing integration
    streaming_core = await connect_to_metacognition(brain, streaming_core)
    
    # 8. Set up bi-directional identity influence
    streaming_core = await setup_identity_integration(brain, streaming_core)
    
    # 9. Enable periodic cross-system tasks
    task = asyncio.create_task(_run_enhanced_periodic_tasks(brain, streaming_core))
    streaming_core._integrated_task = task
    
    # 10. Register enhanced functions in the brain
    _register_enhanced_brain_functions(brain, streaming_core)
    
    logger.info("Enhanced streaming system v2 fully initialized with all improvements")
    
    return streaming_core

def _register_enhanced_brain_functions(brain: NyxBrain, streaming_core: StreamingCore) -> None:
    """
    Register all enhanced streaming functions in the brain
    
    Args:
        brain: NyxBrain instance
        streaming_core: StreamingCore instance
    """
    # Basic streaming functions
    brain.stream = streaming_core.start_streaming
    brain.stop_stream = streaming_core.stop_streaming
    brain.add_stream_question = streaming_core.add_audience_question
    brain.get_stream_stats = streaming_core.get_streaming_stats
    
    # Optimized processing
    brain.process_frame_optimized = streaming_core.process_frame_optimized
    brain.get_performance_metrics = streaming_core.get_performance_metrics
    
    # Enhanced memory and experience
    brain.retrieve_streaming_experience = streaming_core.recall_streaming_experience
    brain.create_streaming_memory = streaming_core.memory_mapper.store_gameplay_memory
    brain.create_streaming_reflection = streaming_core.memory_mapper.create_streaming_reflection
    
    # Enhanced reflection
    brain.generate_deep_reflection = streaming_core.generate_deep_reflection
    brain.generate_comparative_reflection = streaming_core.generate_comparative_reflection
    brain.consolidate_streaming_experiences = streaming_core.enhanced_consolidate_experiences
    
    # Cross-game knowledge
    brain.get_cross_game_insights = streaming_core.get_cross_game_insights
    brain.generate_game_insight = streaming_core.generate_game_insight
    brain.discover_game_patterns = streaming_core.discover_game_patterns
    brain.run_knowledge_consolidation = streaming_core.run_knowledge_consolidation
    
    # Hormone system
    brain.update_hormone_from_streaming_event = streaming_core.update_hormone_from_event
    brain.get_streaming_hormone_state = streaming_core.get_hormone_state
    
    # Reasoning
    if hasattr(streaming_core, "reason_about_streaming_event"):
        brain.reason_about_streaming_event = streaming_core.reason_about_streaming_event
    
    # Audience stats
    if hasattr(streaming_core.streaming_system, "enhanced_audience"):
        brain.get_audience_stats = streaming_core.streaming_system.enhanced_audience.get_audience_stats
        brain.get_popular_topics = streaming_core.streaming_system.enhanced_audience.get_popular_topics
        brain.get_user_personalization = streaming_core.streaming_system.enhanced_audience.get_user_personalization

# Add this function to the end of _run_enhanced_periodic_tasks
async def _run_enhanced_periodic_tasks_v2(brain: NyxBrain, streaming_core: StreamingCore):
    """Run all enhanced periodic tasks"""
    try:
        while True:
            # Only run tasks if streaming is active
            if streaming_core.session_start_time is not None:
                logger.info("Running enhanced periodic streaming tasks")
                
                # 1. Run streaming reflection
                if hasattr(streaming_core, "reflection_engine"):
                    reflection_result = await streaming_core.reflection_engine.run_periodic_reflection()
                    if reflection_result:
                        logger.info(f"Periodic streaming reflection created")
                
                # 2. Run deep reflection periodically (every hour)
                if (hasattr(streaming_core, "reflection_engine") and 
                    isinstance(streaming_core.reflection_engine, EnhancedStreamingReflectionEngine)):
                    # Check if time for deep reflection
                    now = datetime.now()
                    time_since_deep = now - streaming_core.reflection_engine.last_deep_reflection_time
                    if time_since_deep >= streaming_core.reflection_engine.deep_reflection_interval:
                        # Force deep reflection
                        deep_result = await streaming_core.reflection_engine.run_periodic_reflection(force=True)
                        if deep_result and "deep_reflection" in deep_result:
                            logger.info("Deep streaming reflection created")
                
                # 3. Consolidate streaming experiences
                if (hasattr(streaming_core, "reflection_engine") and 
                    hasattr(streaming_core.reflection_engine, "enhanced_consolidate_streaming_experiences")):
                    consolidation = await streaming_core.reflection_engine.enhanced_consolidate_streaming_experiences()
                    if consolidation.get("status") == "success":
                        logger.info(f"Consolidated streaming experiences: {consolidation.get('abstractions_created', 0)} abstractions")
                
                # 4. Run meta-cognitive cycle
                if hasattr(streaming_core, "run_metacognitive_cycle"):
                    meta_result = await streaming_core.run_metacognitive_cycle(
                        streaming_core.streaming_system.game_state
                    )
                    logger.info("Ran meta-cognitive cycle for streaming")
                
                # 5. Consolidate cross-game knowledge
                if hasattr(streaming_core, "run_knowledge_consolidation"):
                    knowledge_result = await streaming_core.run_knowledge_consolidation()
                    logger.info(f"Consolidated cross-game knowledge")
                
                # 6. Sync hormone system with Nyx's emotional core
                if hasattr(streaming_core, "hormone_system") and hasattr(streaming_core.hormone_system, "sync_with_brain_hormone_system"):
                    hormone_result = streaming_core.hormone_system.sync_with_brain_hormone_system()
                    logger.info(f"Synced streaming hormone system with brain")
                
                # 7. Apply reasoning to recent significant events
                if hasattr(streaming_core, "reason_about_game_event"):
                    game_state = streaming_core.streaming_system.game_state
                    recent_events = game_state.recent_events
                    
                    significant_events = [
                        event for event in recent_events
                        if event.get("data", {}).get("significance", 0) >= 7.0 and
                        not event.get("data", {}).get("reasoning_applied", False)
                    ]
                    
                    if significant_events:
                        # Process one significant event
                        event = significant_events[0]
                        await streaming_core.reason_about_game_event(
                            game_name=game_state.game_name or "Unknown Game",
                            event_type=event["type"],
                            event_data=event["data"]
                        )
                        
                        # Mark as processed
                        event["data"]["reasoning_applied"] = True
                        logger.info(f"Applied reasoning to significant event")
                
                # 8. Update identity from streaming experience periodically
                if hasattr(streaming_core, "update_identity_from_streaming"):
                    game_state = streaming_core.streaming_system.game_state
                    game_name = game_state.game_name or "Unknown Game"
                    
                    # Update identity with current session data
                    streaming_data = {
                        "commentary_count": game_state.session_stats.get("commentary_count", 0),
                        "questions_answered": game_state.session_stats.get("questions_answered", 0),
                        "commentary_style": "analytical",  # Would be determined dynamically
                        "session_duration": (datetime.now() - streaming_core.session_start_time).total_seconds() 
                                           if streaming_core.session_start_time else 0,
                        "game_genre": game_state.game_genre
                    }
                    
                    identity_result = await streaming_core.update_identity_from_streaming(
                        game_name=game_name,
                        streaming_data=streaming_data
                    )
                    
                    if identity_result.get("updated", False):
                        logger.info("Updated identity based on streaming")
            
            # Run every 10 minutes
            await asyncio.sleep(600)
    except asyncio.CancelledError:
        logger.info("Enhanced periodic tasks cancelled")
    except Exception as e:
        logger.error(f"Error in enhanced periodic tasks: {e}")
