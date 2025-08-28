# logic/nyx_enhancements_integration.py
"""
Enhanced Nyx Integration Module - Refactored for New SDK Architecture

This module integrates the Nyx memory system with the new modular agent SDK,
providing background processing, memory management, and coordination between
all Nyx subsystems through the centralized SDK.
"""

import logging
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from celery_config import celery_app
from quart import Blueprint, current_app, jsonify

# New SDK imports - use the 'final stop'
from nyx.nyx_agent_sdk import (
    NyxAgentSDK,
    NyxSDKConfig,
    NyxResponse as SDKResponse,
    process_user_input as sdk_process_user_input
)

# Direct context/orchestrator imports for specialized operations
from nyx.nyx_agent.context import NyxContext, ContextBroker, SceneScope
from nyx.nyx_agent.orchestrator import (
    generate_reflection,
    manage_scenario,
    manage_relationships,
    store_messages
)

# Memory operations through the new orchestrator
from memory.memory_orchestrator import (
    MemoryOrchestrator,
    EntityType,
    get_memory_orchestrator
)

# Conflict system
from logic.conflict_system.conflict_synthesizer import (
    get_synthesizer as get_conflict_synthesizer,
    ConflictSynthesizer
)
from logic.conflict_system.background_processor import get_conflict_scheduler

# User model through SDK context
from nyx.user_model_sdk import UserModelManager

# NPCs through orchestrator
from npcs.npc_orchestrator import NPCOrchestrator

# DB connection
from db.connection import get_db_connection_context

# Performance monitoring
from utils.performance import PerformanceTracker, timed_function

# Configure logging
logger = logging.getLogger(__name__)

# Global SDK instance
_sdk_instance: Optional[NyxAgentSDK] = None
_sdk_config = NyxSDKConfig(
    pre_moderate_input=False,  # Set based on your needs
    post_moderate_output=False,
    enable_telemetry=True,
    streaming_chunk_size=320,
    request_timeout_seconds=45.0,
    retry_on_failure=True,
    result_cache_ttl_seconds=10
)

def get_sdk() -> NyxAgentSDK:
    """Get or create the global SDK instance"""
    global _sdk_instance
    if _sdk_instance is None:
        _sdk_instance = NyxAgentSDK(_sdk_config)
        logger.info("NyxAgentSDK initialized for background tasks")
    return _sdk_instance

# -----------------------------------------------------------
# Celery Tasks for Background Processing
# -----------------------------------------------------------

@celery_app.task
def nyx_memory_maintenance_task():
    """
    Celery task to perform regular maintenance on Nyx's memory system.
    Should be scheduled to run daily.
    """
    import asyncio
    
    async def process_all_conversations():
        try:
            async with get_db_connection_context() as conn:
                # Get active conversations
                rows = await conn.fetch("""
                    SELECT DISTINCT user_id, conversation_id
                    FROM messages
                    WHERE created_at > NOW() - INTERVAL '30 days'
                    GROUP BY user_id, conversation_id
                    HAVING COUNT(*) > 10
                """)
                
                for row in rows:
                    user_id = row["user_id"]
                    conversation_id = row["conversation_id"]
                    
                    try:
                        # Create context and run maintenance through orchestrator
                        memory_orchestrator = await get_memory_orchestrator(
                            user_id, conversation_id
                        )
                        
                        # Consolidate memories
                        await memory_orchestrator.consolidate_memories(
                            entity_type=EntityType.PLAYER,
                            entity_id=user_id,
                            force=False
                        )
                        
                        # Clean up SDK cache for this conversation
                        sdk = get_sdk()
                        await sdk.cleanup_conversation(str(conversation_id))
                        
                        logger.info(f"Memory maintenance completed for user_id={user_id}, "
                                   f"conversation_id={conversation_id}")
                                   
                    except Exception as e:
                        logger.error(f"Error in memory maintenance for user_id={user_id}, "
                                    f"conversation_id={conversation_id}: {str(e)}")
                    
                    # Brief pause between processing
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error in nyx_memory_maintenance_task: {str(e)}")
    
    asyncio.run(process_all_conversations())
    return {"status": "Memory maintenance completed"}

# -----------------------------------------------------------
# Enhanced Background Chat Task - Using SDK
# -----------------------------------------------------------

@timed_function(name="enhanced_background_chat_task")
async def enhanced_background_chat_task(
    conversation_id: int,
    user_input: str,
    universal_update: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None
):
    """
    Enhanced background chat task using the new SDK architecture.
    Simplified to leverage SDK's built-in capabilities.
    """
    performance_tracker = PerformanceTracker("enhanced_background_chat_task")
    
    try:
        performance_tracker.start_phase("initialization")
        
        # Get user_id if not provided
        if user_id is None:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    "SELECT user_id FROM conversations WHERE id=$1", 
                    conversation_id
                )
                if not row:
                    logger.error(f"No conversation found with id {conversation_id}")
                    return
                user_id = row["user_id"]
        
        performance_tracker.end_phase()
        
        # Build context metadata
        performance_tracker.start_phase("context_preparation")
        
        metadata = {
            "background_task": True,
            "timestamp": time.time()
        }
        
        # Add universal update if provided
        if universal_update:
            metadata["universal_update"] = universal_update
            metadata["apply_universal_updates"] = True
        
        # Get current scene context if available
        try:
            async with get_db_connection_context() as conn:
                # Get current location
                location_row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay 
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'location'
                """, user_id, conversation_id)
                
                if location_row and location_row["value"]:
                    metadata["location"] = location_row["value"]
                
                # Get active NPCs
                npcs_row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay 
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'introduced_npcs'
                """, user_id, conversation_id)
                
                if npcs_row and npcs_row["value"]:
                    try:
                        npcs_data = json.loads(npcs_row["value"])
                        metadata["npc_present"] = [npc.get("npc_id") for npc in npcs_data if npc.get("npc_id")]
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Could not fetch scene context: {e}")
        
        performance_tracker.end_phase()
        
        # Process through SDK
        performance_tracker.start_phase("sdk_processing")
        
        sdk = get_sdk()
        
        # Optional: Warm cache if we have location
        if metadata.get("location"):
            await sdk.warmup_cache(str(conversation_id), metadata["location"])
        
        # Main SDK call
        sdk_response: SDKResponse = await sdk.process_user_input(
            message=user_input,
            conversation_id=str(conversation_id),
            user_id=str(user_id),
            metadata=metadata
        )
        
        performance_tracker.end_phase()
        
        # Extract response data
        ai_response = sdk_response.narrative
        response_metadata = sdk_response.metadata
        
        # Store messages (SDK may already do this, but ensure it happens)
        performance_tracker.start_phase("store_messages")
        await store_messages(user_id, conversation_id, user_input, ai_response)
        logger.info(f"Stored messages for conversation {conversation_id}")
        performance_tracker.end_phase()
        
        # Emit response to client via SocketIO
        performance_tracker.start_phase("emit_response")
        try:
            # Stream the response token by token
            for i in range(0, len(ai_response), 3):
                token = ai_response[i:i+3]
                await current_app.socketio.emit('new_token', {'token': token}, room=conversation_id)
                await asyncio.sleep(0.05)
            
            # Signal completion with metadata
            await current_app.socketio.emit('done', {
                'full_text': ai_response,
                'world_state': sdk_response.world_state,
                'choices': sdk_response.choices,
                'metadata': response_metadata
            }, room=conversation_id)
            
            logger.info(f"Completed streaming response for conversation {conversation_id}")
        except Exception as socket_err:
            logger.error(f"Error emitting response: {socket_err}")
            try:
                await current_app.socketio.emit('error', {'error': str(socket_err)}, room=conversation_id)
            except:
                pass
        performance_tracker.end_phase()
        
        # Handle image generation if indicated
        if sdk_response.image and sdk_response.image.get("should_generate"):
            performance_tracker.start_phase("image_generation")
            await handle_image_generation_through_sdk(
                user_id, 
                conversation_id, 
                sdk_response.image
            )
            performance_tracker.end_phase()
        
        # Handle narrative arc updates
        if response_metadata.get("narrative_arc_update"):
            performance_tracker.start_phase("narrative_arc_update")
            await update_narrative_arcs_from_sdk_response(
                user_id, 
                conversation_id, 
                response_metadata["narrative_arc_update"]
            )
            performance_tracker.end_phase()
        
        # Get final metrics
        performance_metrics = performance_tracker.get_metrics()
        logger.info(f"Enhanced background chat task completed in {performance_metrics['total_time']:.3f}s")
        
        # Store performance metrics
        await store_performance_metrics(user_id, conversation_id, performance_metrics)
        
    except Exception as e:
        logger.error(f"Critical error in enhanced_background_chat_task: {str(e)}", exc_info=True)
        try:
            await current_app.socketio.emit('error', {'error': f"Server error: {str(e)}"}, room=conversation_id)
        except Exception as notify_err:
            logger.error(f"Failed to send error notification: {notify_err}")

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------

async def handle_image_generation_through_sdk(
    user_id: int,
    conversation_id: int,
    image_data: Dict[str, Any]
):
    """Handle image generation using SDK response data"""
    try:
        from routes.ai_image_generator import generate_roleplay_image_from_gpt
        
        image_result = await generate_roleplay_image_from_gpt(
            {
                "narrative": image_data.get("context", ""),
                "image_generation": {
                    "generate": True,
                    "priority": image_data.get("priority", "medium"),
                    "focus": "balanced",
                    "framing": "medium_shot",
                    "reason": image_data.get("reason", "Narrative moment"),
                    "prompt": image_data.get("prompt", "")
                }
            },
            user_id,
            conversation_id
        )
        
        # Emit image to the client via SocketIO
        if image_result and "image_urls" in image_result and image_result["image_urls"]:
            await current_app.socketio.emit('image', {
                'image_url': image_result["image_urls"][0],
                'prompt_used': image_result.get('prompt_used', ''),
                'reason': image_data.get("reason", "Narrative moment")
            }, room=conversation_id)
            logger.info(f"Image emitted to client for conversation {conversation_id}")
            
    except Exception as img_err:
        logger.error(f"Error generating image: {img_err}")

async def update_narrative_arcs_from_sdk_response(
    user_id: int,
    conversation_id: int,
    arc_update: Dict[str, Any]
):
    """Update narrative arcs based on SDK response metadata"""
    try:
        async with get_db_connection_context() as conn:
            # Get current narrative arcs
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
            """, user_id, conversation_id)
            
            if row and row["value"]:
                narrative_arcs = json.loads(row["value"])
            else:
                narrative_arcs = {
                    "active_arcs": [],
                    "planned_arcs": [],
                    "completed_arcs": [],
                    "narrative_adaption_history": []
                }
            
            # Apply updates from SDK
            if arc_update.get("progress_updates"):
                for arc_id, progress in arc_update["progress_updates"].items():
                    for arc in narrative_arcs.get("active_arcs", []):
                        if arc.get("id") == arc_id:
                            arc["progress"] = progress
                            if progress >= 100:
                                arc["status"] = "completed"
                                arc["completion_date"] = datetime.now(timezone.utc).isoformat()
            
            # Add new arcs if specified
            if arc_update.get("new_arcs"):
                narrative_arcs["planned_arcs"].extend(arc_update["new_arcs"])
            
            # Save updated narrative arcs
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = $4
            """, user_id, conversation_id, "NyxNarrativeArcs", json.dumps(narrative_arcs))
            
    except Exception as e:
        logger.error(f"Error updating narrative arcs: {e}")

async def store_performance_metrics(
    user_id: int,
    conversation_id: int,
    metrics: Dict[str, Any]
):
    """Store performance metrics for analysis"""
    try:
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO performance_logs (user_id, conversation_id, metrics, created_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            """, user_id, conversation_id, json.dumps(metrics))
    except Exception as e:
        logger.debug(f"Could not store performance metrics: {e}")

# -----------------------------------------------------------
# Database Migration Function
# -----------------------------------------------------------

async def migrate_nyx_memory_system():
    """
    Create or update the necessary database tables for the enhanced memory system.
    Run this once when deploying the new system.
    """
    try:
        async with get_db_connection_context() as conn:
            # Create performance_logs table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    metrics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_perf_user_conv ON performance_logs(user_id, conversation_id);
                CREATE INDEX IF NOT EXISTS idx_perf_created ON performance_logs(created_at);
            """)
            
            logger.info("Successfully migrated database for enhanced Nyx system")
            
    except Exception as e:
        logger.error(f"Database migration error: {str(e)}")
        raise

# -----------------------------------------------------------
# Integration Endpoints
# -----------------------------------------------------------

async def nyx_introspection_endpoint(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Handler for an API endpoint to get Nyx's introspection.
    Now uses the SDK and orchestrator.
    """
    try:
        # Generate introspective reflection using orchestrator
        reflection_result = await generate_reflection(
            user_id=user_id,
            conversation_id=conversation_id,
            topic="self_awareness and current state"
        )
        
        # Get emotional state from context
        ctx = NyxContext(user_id, conversation_id)
        await ctx.initialize()
        
        emotional_state = {}
        if hasattr(ctx, 'emotional_core') and ctx.emotional_core:
            emotional_state = ctx.emotional_core.get_current_state()
        
        return {
            "status": "success",
            "introspection": reflection_result.get("reflection", ""),
            "confidence": reflection_result.get("confidence", 0.5),
            "emotional_state": emotional_state,
            "topic": reflection_result.get("topic", "self_awareness")
        }
        
    except Exception as e:
        logger.error(f"Error in nyx_introspection_endpoint: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# -----------------------------------------------------------
# Background Task Runner
# -----------------------------------------------------------

async def run_background_maintenance(user_id: int, conversation_id: int):
    """
    Run background maintenance tasks for a conversation.
    Can be called periodically or triggered by events.
    """
    try:
        # Memory consolidation
        memory_orchestrator = await get_memory_orchestrator(user_id, conversation_id)
        await memory_orchestrator.consolidate_memories(
            entity_type=EntityType.PLAYER,
            entity_id=user_id,
            force=False
        )
        
        # Conflict tension calculation
        conflict_synthesizer = await get_conflict_synthesizer(user_id, conversation_id)
        if conflict_synthesizer:
            tensions = await conflict_synthesizer.calculate_tensions()
            logger.info(f"Updated conflict tensions: {tensions}")
        
        # Clean SDK cache
        sdk = get_sdk()
        await sdk.cleanup_conversation(str(conversation_id))
        
        logger.info(f"Background maintenance completed for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Error in background maintenance: {e}")

# -----------------------------------------------------------
# Initialization Function
# -----------------------------------------------------------

async def initialize_nyx_memory_system():
    """
    Initialize the Nyx memory system by setting up database tables
    and performing any necessary data migrations.
    Call this during application startup.
    """
    try:
        await migrate_nyx_memory_system()
        
        # Initialize SDK
        sdk = get_sdk()
        await sdk.initialize_agent()
        
        logger.info("Nyx memory system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Nyx memory system: {str(e)}")
        # Don't fail startup, but log the error

# -----------------------------------------------------------
# Export list
# -----------------------------------------------------------

__all__ = [
    'enhanced_background_chat_task',
    'nyx_memory_maintenance_task',
    'nyx_introspection_endpoint',
    'run_background_maintenance',
    'initialize_nyx_memory_system',
    'get_sdk'
]
