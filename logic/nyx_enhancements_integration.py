# Integration of Enhanced Nyx Memory System into existing codebase

import logging
import asyncio
from datetime import datetime
import json
import os
from celery_config import celery_app
from quart import Blueprint, current_app, jsonify

# Updated imports using newer modules
from nyx.nyx_agent_sdk import (
    process_user_input,
    process_user_input_with_openai,
    determine_image_generation, 
    get_emotional_state,
    update_emotional_state
)
from memory.memory_nyx_integration import (
    MemoryNyxBridge,
    get_memory_nyx_bridge,
    run_maintenance_through_nyx
)
from nyx.nyx_enhanced_system import NyxEnhancedSystem, NyxGoal
from nyx.user_model_sdk import (
    UserModelManager,
    process_user_input_for_model,
    get_response_guidance_for_user
)

# DB connection
from db.connection import get_db_connection_context
import asyncpg

# Keep original aggregator import for context building
from logic.aggregator_sdk import get_aggregated_roleplay_context, build_aggregator_text

# Keep the original universal updater integration
from logic.universal_updater_agent import apply_universal_updates_async

# Keep NPCLearningManager for NPC adaptation
from npcs.npc_learning_adaptation import NPCLearningManager

# Performance monitoring
from utils.performance import PerformanceTracker, timed_function

# Configure logging
logger = logging.getLogger(__name__)

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
                    FROM NyxMemories
                    WHERE is_archived = FALSE
                    AND timestamp > NOW() - INTERVAL '30 days'
                """)
                
                for row in rows:
                    user_id = row["user_id"]
                    conversation_id = row["conversation_id"]
                    
                    try:
                        # Updated to use the new maintenance function
                        await run_maintenance_through_nyx(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            entity_type="nyx",
                            entity_id=user_id
                        )
                        logger.info(f"Memory maintenance completed for user_id={user_id}, conversation_id={conversation_id}")
                    except Exception as e:
                        logger.error(f"Error in memory maintenance for user_id={user_id}, conversation_id={conversation_id}: {str(e)}")
                        
                    # Brief pause between processing to avoid overloading the database
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error in nyx_memory_maintenance_task: {str(e)}")
                
    asyncio.run(process_all_conversations())
    return {"status": "Memory maintenance completed"}

# -----------------------------------------------------------
# Integration with Game Processing
# -----------------------------------------------------------

# Update to background_chat_task in main.py
@timed_function(name="enhanced_background_chat_task")
async def enhanced_background_chat_task(conversation_id, user_input, universal_update=None, user_id=None):
    """
    Enhanced background chat task that leverages the Nyx agent for responses.
    
    Args:
        conversation_id: The conversation ID
        user_input: The user's input message
        universal_update: Optional universal update data
        user_id: Optional user ID (if not provided, will be fetched from DB)
    """
    performance_tracker = PerformanceTracker("enhanced_background_chat_task")
    
    try:
        performance_tracker.start_phase("initialization")
        
        # Get user_id if not provided
        if user_id is None:
            try:
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow(
                        "SELECT user_id FROM conversations WHERE id=$1", 
                        conversation_id
                    )
                    if not row:
                        logging.error(f"No conversation found with id {conversation_id}")
                        return
                    user_id = row["user_id"]
            except Exception as e:
                logging.error(f"Error fetching user_id for conversation {conversation_id}: {e}")
                return
        
        # Initialize the updated components
        nyx_enhanced_system = NyxEnhancedSystem(user_id, conversation_id)
        await nyx_enhanced_system.initialize()
        
        # Initialize NPCLearningManager (keeping original functionality)
        learning_manager = NPCLearningManager(user_id, conversation_id)
        try:
            await learning_manager.initialize()
        except Exception as e:
            logging.error(f"Failed to initialize learning manager: {e}")
            # Continue without learning - don't block the chat process
        
        # User model manager
        user_model_manager = UserModelManager(user_id, conversation_id)
        
        # Get memory bridge
        memory_bridge = await get_memory_nyx_bridge(user_id, conversation_id)
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("get_context")
        
        # Get aggregated context - using the original function
        player_name = "Chase"  # Default player name
        aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, player_name)
        
        # Get user model guidance
        user_guidance = await get_response_guidance_for_user(user_id, conversation_id)
        
        # Prepare context data
        context_data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "user_guidance": user_guidance,
            "aggregator_data": aggregator_data,
            "player_name": player_name,
            "location": aggregator_data.get("current_location", "Unknown"),
            "time_of_day": aggregator_data.get("time_of_day", "Morning"),
            "emotional_state": await get_emotional_state({
                "context": {"user_id": user_id, "conversation_id": conversation_id}
            })
        }
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("process_universal_updates")
        
        # Process universal updates if provided
        if universal_update:
            universal_update["user_id"] = user_id
            universal_update["conversation_id"] = conversation_id
            
            try:
                # Apply updates with proper async context
                async with get_db_connection_context() as conn:
                    # Keep the original apply_universal_updates_async function
                    await apply_universal_updates_async(
                        user_id, 
                        conversation_id, 
                        universal_update,
                        conn
                    )
                
                # Refresh context after updates
                aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
                context_data["aggregator_data"] = aggregator_data
                context_data["location"] = aggregator_data.get("current_location", "Unknown")
                context_data["time_of_day"] = aggregator_data.get("time_of_day", "Morning")
                
            except Exception as update_err:
                logging.error(f"Error applying universal updates: {update_err}", exc_info=True)
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("extract_npcs")
        
        # Get list of NPCs in the scene for learning
        npcs_in_scene = []
        if "introduced_npcs" in aggregator_data:
            npcs_in_scene = aggregator_data["introduced_npcs"]
        
        npc_ids = []
        for npc in npcs_in_scene:
            if "npc_id" in npc:
                npc_ids.append(npc["npc_id"])
        
        context_data["npcsPresent"] = npcs_in_scene
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("enhance_context")
        
        # Enhance context with memories
        memory_enhancement = await enhance_context_with_memories(
            user_id, conversation_id, user_input, context_data, memory_bridge
        )
        
        # Add memory context to the main context
        if memory_enhancement.get("text"):
            context_data["memory_context"] = memory_enhancement["text"]
            context_data["referenced_memory_ids"] = memory_enhancement.get("referenced_memory_ids", [])
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("process_input")
        
        # Process the user input using the updated function
        response_data = await process_user_input(
            user_id=user_id,
            conversation_id=conversation_id,
            user_input=user_input,
            context_data=context_data
        )
        
        if not response_data.get("success", False):
            # Fall back to OpenAI integration if the standard processing fails
            response_data = await process_user_input_with_openai(
                user_id=user_id,
                conversation_id=conversation_id,
                user_input=user_input,
                context_data=context_data
            )
        
        # Extract the response text
        ai_response = response_data.get("response", {}).get("narrative", "")
        if not ai_response and "message" in response_data:
            ai_response = response_data["message"]
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("update_user_model")
        
        # Update user model based on interaction
        await process_user_input_for_model(
            user_id=user_id,
            conversation_id=conversation_id,
            user_input=user_input,
            nyx_response=ai_response,
            context_data=context_data
        )
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("npc_learning")
        
        # Process the interaction for NPC learning if NPCs are present
        if npc_ids:
            try:
                learning_result = await learning_manager.process_event_for_learning(
                    event_text=user_input,
                    event_type="player_conversation",
                    npc_ids=npc_ids,
                    player_response={
                        "summary": "Player initiated conversation",
                        "content": user_input
                    }
                )
                logging.info(f"Processed learning for {len(npc_ids)} NPCs")
            except Exception as learn_err:
                logging.error(f"Error in NPC learning processing: {learn_err}", exc_info=True)
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("store_response")
        
        # Store Nyx response in database with proper error handling
        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
                    conversation_id, "Nyx", ai_response
                )
                logging.info(f"Stored Nyx response in database for conversation {conversation_id}")
        except Exception as db_error:
            logging.error(f"Database error storing Nyx response: {str(db_error)}", exc_info=True)
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("emit_response")
        
        # Emit response to client via SocketIO with proper error handling
        try:
            # Stream the response token by token
            for i in range(0, len(ai_response), 3):
                token = ai_response[i:i+3]
                await current_app.socketio.emit('new_token', {'token': token}, room=conversation_id)
                await asyncio.sleep(0.05)  # Use asyncio.sleep instead of socketio.sleep
                
            # Signal completion
            await current_app.socketio.emit('done', {'full_text': ai_response}, room=conversation_id)
            logging.info(f"Completed streaming response for conversation {conversation_id}")
        except Exception as socket_err:
            logging.error(f"Error emitting response: {socket_err}", exc_info=True)
            try:
                await current_app.socketio.emit('error', {'error': str(socket_err)}, room=conversation_id)
            except:
                pass
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("image_generation")
        
        # Check if we should generate an image using the updated function
        image_result = await determine_image_generation(
            {"context": {"user_id": user_id, "conversation_id": conversation_id}},
            ai_response
        )
        image_data = json.loads(image_result)
        should_generate = image_data.get("should_generate", False)
        
        # Image generation with proper error handling
        if should_generate:
            try:
                # Generate image based on the response
                from routes.ai_image_generator import generate_roleplay_image_from_gpt
                
                image_prompt = image_data.get("image_prompt", ai_response[:200])
                
                image_result = await generate_roleplay_image_from_gpt(
                    {
                        "narrative": ai_response,
                        "image_generation": {
                            "generate": True,
                            "priority": "medium",
                            "focus": "balanced",
                            "framing": "medium_shot",
                            "reason": "Narrative moment",
                            "prompt": image_prompt
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
                        'reason': "Narrative moment"
                    }, room=conversation_id)
                    logging.info(f"Image emitted to client for conversation {conversation_id}")
            except Exception as img_err:
                logging.error(f"Error generating image: {img_err}", exc_info=True)
                try:
                    await current_app.socketio.emit('error', {'error': f"Image generation failed: {str(img_err)}"}, room=conversation_id)
                except:
                    pass
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("memory_recording")
        
        # Store memory of the response (asynchronously)
        try:
            memory_task = asyncio.create_task(
                memory_bridge.remember(
                    entity_type="nyx",
                    entity_id=user_id,
                    memory_text=f"I responded: {ai_response[:200]}..." if len(ai_response) > 200 else f"I responded: {ai_response}",
                    importance="medium",
                    emotional=True,
                    tags=["nyx_response"],
                    related_entities={"player": context_data.get("player_name", "User")}
                )
            )
            
            # Don't wait for completion - let it run in background
            logging.debug("Memory recording task created")
        except Exception as mem_err:
            logging.error(f"Error setting up memory recording: {mem_err}", exc_info=True)
            # Non-critical, continue without failing the main task
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("update_narrative_arcs")
        
        # Update narrative arcs based on the interaction
        try:
            await update_narrative_arcs_for_interaction(
                user_id, conversation_id, user_input, ai_response
            )
        except Exception as arc_err:
            logging.error(f"Error updating narrative arcs: {arc_err}", exc_info=True)
        
        performance_tracker.end_phase()
        performance_tracker.start_phase("update_emotional_state")
        
        # Update emotional state based on the interaction
        try:
            # Get current emotional state
            current_emotional_state = json.loads(await get_emotional_state(
                {"context": {"user_id": user_id, "conversation_id": conversation_id}}
            ))
            
            # Process emotional updates based on the interaction
            event = {
                "type": "interaction",
                "content": f"User said: {user_input}\nNyx responded: {ai_response[:100]}..."
            }
            
            processed_event = await nyx_enhanced_system.process_event(event)
            emotional_state = processed_event.get("emotional_elements", {})
            
            # Update the emotional state
            await update_emotional_state(
                {"context": {"user_id": user_id, "conversation_id": conversation_id}},
                emotional_state
            )
        except Exception as emotional_err:
            logging.error(f"Error updating emotional state: {emotional_err}", exc_info=True)
        
        # Get final metrics
        performance_metrics = performance_tracker.get_metrics()
        logging.info(f"Enhanced background chat task completed in {performance_metrics['total_time']:.3f}s")
            
    except Exception as e:
        logging.error(f"Critical error in enhanced_background_chat_task: {str(e)}", exc_info=True)
        # Attempt to notify the client about the error
        try:
            await current_app.socketio.emit('error', {'error': f"Server error: {str(e)}"}, room=conversation_id)
        except Exception as notify_err:
            logging.error(f"Failed to send error notification: {notify_err}")

async def enhance_context_with_memories(
    user_id, conversation_id, user_input, context, memory_bridge
):
    """
    Enhance the context with relevant memories and Nyx's metacognition.
    This makes responses more consistent and personalized.
    """
    # Retrieve relevant memories based on the user input
    memories = await memory_bridge.recall(
        entity_type="nyx",
        entity_id=user_id,
        query=user_input,
        context=context,
        limit=5
    )
    
    # Format the enhancement
    enhancement = {
        "memory_context": "",
        "referenced_memory_ids": []
    }
    
    # Extract memory texts and IDs
    if memories and "memories" in memories:
        memory_texts = [m["text"] for m in memories["memories"]]
        memory_ids = [m["id"] for m in memories["memories"]]
        
        # Format the memory context
        enhancement["memory_context"] = "\n\n### Nyx's Relevant Memories ###\n" + \
                                      "\n".join([f"- {text}" for text in memory_texts]) if memory_texts else ""
        enhancement["referenced_memory_ids"] = memory_ids
    
    # Generate introspection about Nyx's understanding
    try:
        # Create the enhanced system for introspection
        nyx_enhanced = NyxEnhancedSystem(user_id, conversation_id)
        await nyx_enhanced.initialize()
        
        # Generate introspection
        event = {
            "type": "introspection",
            "content": user_input
        }
        introspection_result = await nyx_enhanced.process_event(event)
        
        if introspection_result:
            enhancement["introspection_context"] = f"\n\n### Nyx's Self-Reflection ###\n{introspection_result.get('content', '')}"
    except Exception as introspection_err:
        logging.error(f"Error generating introspection: {introspection_err}", exc_info=True)
    
    # Combine all enhancements
    combined_text = ""
    if enhancement["memory_context"]:
        combined_text += enhancement["memory_context"]
    if enhancement.get("introspection_context"):
        combined_text += enhancement["introspection_context"]
    
    # Also include the enhancement object for additional processing
    enhancement["text"] = combined_text
    
    return enhancement

async def update_narrative_arcs_for_interaction(
    user_id, conversation_id, user_input, ai_response
):
    """
    Update narrative arcs based on the player interaction.
    This helps Nyx maintain coherent storylines.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get current narrative arcs
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
            """, user_id, conversation_id)
            
            if not row or not row["value"]:
                return  # No narrative arcs defined
            
            narrative_arcs = json.loads(row["value"])
            
            # Check for progression in active arcs
            for arc in narrative_arcs.get("active_arcs", []):
                # Simple keyword matching to detect progression
                arc_keywords = arc.get("keywords", [])
                progression_detected = False
                
                # Check user input and AI response for keywords
                combined_text = f"{user_input} {ai_response}".lower()
                for keyword in arc_keywords:
                    if keyword.lower() in combined_text:
                        progression_detected = True
                        break
                
                if progression_detected:
                    # Update arc progress
                    if "progress" not in arc:
                        arc["progress"] = 0
                    
                    # Increment progress (small increment for keyword matches)
                    arc["progress"] = min(100, arc["progress"] + 5)
                    
                    # Record the interaction
                    if "interactions" not in arc:
                        arc["interactions"] = []
                    
                    arc["interactions"].append({
                        "timestamp": datetime.now().isoformat(),
                        "progression_amount": 5,
                        "notes": f"Keyword match in interaction"
                    })
                    
                    # Check for completion
                    if arc["progress"] >= 100 and arc.get("status") != "completed":
                        arc["status"] = "completed"
                        arc["completion_date"] = datetime.now().isoformat()
                        
                        # Move from active to completed
                        if arc in narrative_arcs["active_arcs"]:
                            narrative_arcs["active_arcs"].remove(arc)
                            if "completed_arcs" not in narrative_arcs:
                                narrative_arcs["completed_arcs"] = []
                            narrative_arcs["completed_arcs"].append(arc)
                        
                        # Add record of completion
                        if "narrative_adaption_history" not in narrative_arcs:
                            narrative_arcs["narrative_adaption_history"] = []
                        
                        narrative_arcs["narrative_adaption_history"].append({
                            "event": f"Arc completed: {arc.get('name', 'Unnamed Arc')}",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Activate a new arc if available
                        if narrative_arcs.get("planned_arcs", []):
                            new_arc = narrative_arcs["planned_arcs"].pop(0)
                            new_arc["status"] = "active"
                            new_arc["start_date"] = datetime.now().isoformat()
                            narrative_arcs["active_arcs"].append(new_arc)
                            
                            narrative_arcs["narrative_adaption_history"].append({
                                "event": f"New arc activated: {new_arc.get('name', 'Unnamed Arc')}",
                                "timestamp": datetime.now().isoformat()
                            })
            
            # Save updated narrative arcs
            await conn.execute("""
                UPDATE CurrentRoleplay
                SET value = $1
                WHERE user_id = $2 AND conversation_id = $3 AND key = 'NyxNarrativeArcs'
            """, json.dumps(narrative_arcs), user_id, conversation_id)
    except Exception as e:
        logging.error(f"Error updating narrative arcs: {e}", exc_info=True)

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
            # Create the enhanced NyxMemories table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS NyxMemories (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    memory_text TEXT NOT NULL,
                    memory_type TEXT DEFAULT 'observation',
                    significance FLOAT DEFAULT 3.0,
                    embedding VECTOR(1536),
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT[],
                    times_recalled INTEGER DEFAULT 0,
                    last_recalled TIMESTAMP,
                    is_archived BOOLEAN DEFAULT FALSE,
                    is_consolidated BOOLEAN DEFAULT FALSE,
                    metadata JSONB,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                
                -- Create indexes for efficient retrieval
                CREATE INDEX IF NOT EXISTS idx_nyxmem_user_conv ON NyxMemories(user_id, conversation_id);
                CREATE INDEX IF NOT EXISTS idx_nyxmem_type ON NyxMemories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_nyxmem_archived ON NyxMemories(is_archived);
                CREATE INDEX IF NOT EXISTS idx_nyxmem_timestamp ON NyxMemories(timestamp);
            """)
            
            # Create NyxAgentState table for metacognition
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS NyxAgentState (
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    current_goals JSONB,
                    predicted_futures JSONB,
                    reflection_notes TEXT,
                    emotional_state JSONB,
                    narrative_assessment JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    PRIMARY KEY (user_id, conversation_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
            """)
            
            # Create user model table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS UserModels (
                    user_id INTEGER PRIMARY KEY,
                    model_data JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
            
            # Add any columns that might be missing in existing tables
            try:
                await conn.execute("ALTER TABLE NyxMemories ADD COLUMN IF NOT EXISTS is_consolidated BOOLEAN DEFAULT FALSE;")
                await conn.execute("ALTER TABLE NyxMemories ADD COLUMN IF NOT EXISTS metadata JSONB;")
                await conn.execute("ALTER TABLE NyxAgentState ADD COLUMN IF NOT EXISTS emotional_state JSONB;")
                await conn.execute("ALTER TABLE NyxAgentState ADD COLUMN IF NOT EXISTS narrative_assessment JSONB;")
            except Exception as column_error:
                logging.error(f"Error adding columns: {str(column_error)}")
                
            logging.info("Successfully migrated database for enhanced Nyx memory system")
    except Exception as e:
        logging.error(f"Database migration error: {str(e)}")
        raise

# -----------------------------------------------------------
# Integration with routes
# -----------------------------------------------------------

async def nyx_introspection_endpoint(user_id, conversation_id):
    """
    Handler for an API endpoint to get Nyx's introspection.
    Can be integrated into your API routes.
    """
    try:
        # Create enhanced system
        nyx_enhanced = NyxEnhancedSystem(user_id, conversation_id)
        await nyx_enhanced.initialize()
        
        # Generate introspection
        event = {
            "type": "introspection",
            "content": "Generate introspection"
        }
        introspection_result = await nyx_enhanced.process_event(event)
        
        # Get memory statistics
        memory_bridge = await get_memory_nyx_bridge(user_id, conversation_id)
        
        # Combine results
        return {
            "status": "success",
            "introspection": introspection_result.get("content", ""),
            "emotional_state": introspection_result.get("emotional_elements", {}),
            "confidence": introspection_result.get("style", {}).get("dominance", 0.7)
        }
    except Exception as e:
        logging.error(f"Error in nyx_introspection_endpoint: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Example of how to add this to your Flask routes:
"""
@app.route("/api/nyx/introspection", methods=["GET"])
async def get_nyx_introspection():
    user_id = request.args.get("user_id", type=int)
    conversation_id = request.args.get("conversation_id", type=int)
    
    if not user_id or not conversation_id:
        return jsonify({"status": "error", "error": "Missing parameters"}), 400
        
    result = await nyx_introspection_endpoint(user_id, conversation_id)
    return jsonify(result)
"""

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
        logging.info("Nyx memory system initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Nyx memory system: {str(e)}")
        # Don't fail startup, but log the error
