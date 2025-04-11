# Integration of Enhanced Nyx Memory System into existing codebase

import logging
import asyncio
from datetime import datetime
import json
import os
from celery_config import celery_app
from nyx.nyx_agent_sdk import NyxAgent
from logic.aggregator_sdk import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text
from logic.gpt_image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from logic.universal_updater_agent import apply_universal_updates_async
from memory.memory_nyx_integration import MemoryNyxBridge, run_maintenance
from npcs.npc_learning_adaptation import NPCLearningManager
from db.connection import get_db_connection_context
import asyncpg

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
                        await run_maintenance(user_id, conversation_id)
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
async def enhanced_background_chat_task(conversation_id, user_input, universal_update=None, user_id=None):
    """
    Enhanced background chat task that leverages the Nyx agent for responses.
    
    Args:
        conversation_id: The conversation ID
        user_input: The user's input message
        universal_update: Optional universal update data
        user_id: Optional user ID (if not provided, will be fetched from DB)
    """
    try:
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
        
        # Initialize Nyx agent and NPCLearningManager
        nyx_agent = NyxAgent(user_id, conversation_id)
        learning_manager = NPCLearningManager(user_id, conversation_id)
        try:
            await learning_manager.initialize()
        except Exception as e:
            logging.error(f"Failed to initialize learning manager: {e}")
            # Continue without learning - don't block the chat process
        
        # Get aggregated context
        player_name = "Chase"  # Default player name
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
        
        # Process universal updates if provided
        if universal_update:
            universal_update["user_id"] = user_id
            universal_update["conversation_id"] = conversation_id
            
            try:
                # Apply updates with proper async context
                async with get_db_connection_context() as conn:
                    await apply_universal_updates_async(
                        user_id, 
                        conversation_id, 
                        universal_update,
                        conn
                    )
                
                # Refresh context after updates
                aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
            except Exception as update_err:
                logging.error(f"Error applying universal updates: {update_err}", exc_info=True)
        
        # Get list of NPCs in the scene for learning
        npcs_in_scene = aggregator_data.get("npcsPresent", [])
        npc_ids = []
        for npc in npcs_in_scene:
            if "id" in npc:
                npc_ids.append(npc["id"])
        
        # Build context for Nyx
        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
            "player_name": player_name,
            "npc_present": npcs_in_scene,
            "aggregator_data": aggregator_data
        }
        
        # Get response from Nyx agent
        response_data = await nyx_agent.process_input(
            user_input,
            context=context
        )
        
        ai_response = response_data.get("text", "")
        
        # Process the interaction for NPC learning if NPCs are present
        if npc_ids:
            try:
                learning_result = await learning_manager.process_event_for_learning(
                    event_text=user_input,
                    event_type="player_conversation",
                    npc_ids=npc_ids,
                    player_response={
                        "summary": "Player initiated conversation"
                    }
                )
                logging.info(f"Processed learning for {len(npc_ids)} NPCs")
            except Exception as learn_err:
                logging.error(f"Error in NPC learning processing: {learn_err}", exc_info=True)
        
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
        
        # Emit response to client via SocketIO with proper error handling
        from flask_socketio import emit
        try:
            # Stream the response token by token
            for i in range(0, len(ai_response), 3):
                token = ai_response[i:i+3]
                emit('new_token', {'token': token}, room=conversation_id)
                socketio.sleep(0.05)
                
            # Signal completion
            emit('done', {'full_text': ai_response}, room=conversation_id)
            logging.info(f"Completed streaming response for conversation {conversation_id}")
        except Exception as socket_err:
            logging.error(f"Error emitting response: {socket_err}", exc_info=True)
            try:
                emit('error', {'error': str(socket_err)}, room=conversation_id)
            except:
                pass
        
        # Check if we should generate an image
        should_generate = response_data.get("generate_image", False)
        
        # Image generation with proper error handling
        if should_generate:
            try:
                # Generate image based on the response
                image_result = await generate_roleplay_image_from_gpt(
                    {
                        "narrative": ai_response,
                        "image_generation": {
                            "generate": True,
                            "priority": "medium",
                            "focus": "balanced",
                            "framing": "medium_shot",
                            "reason": "Narrative moment"
                        }
                    },
                    user_id,
                    conversation_id
                )
                
                # Emit image to the client via SocketIO
                if image_result and "image_urls" in image_result and image_result["image_urls"]:
                    emit('image', {
                        'image_url': image_result["image_urls"][0],
                        'prompt_used': image_result.get('prompt_used', ''),
                        'reason': "Narrative moment"
                    }, room=conversation_id)
                    logging.info(f"Image emitted to client for conversation {conversation_id}")
            except Exception as img_err:
                logging.error(f"Error generating image: {img_err}", exc_info=True)
                try:
                    emit('error', {'error': f"Image generation failed: {str(img_err)}"}, room=conversation_id)
                except:
                    pass
                
        # Store memory of the response (asynchronously)
        try:
            nyx_memory = MemoryNyxBridge(user_id, conversation_id)
            
            memory_task = asyncio.create_task(
                nyx_memory.add_memory(
                    memory_text=f"I responded: {ai_response[:200]}..." if len(ai_response) > 200 else f"I responded: {ai_response}",
                    memory_type="observation",
                    significance=4,
                    tags=["nyx_response"],
                    related_entities={"player": player_name},
                    context=context
                )
            )
            
            # Don't wait for completion - let it run in background
            logging.debug("Memory recording task created")
        except Exception as mem_err:
            logging.error(f"Error setting up memory recording: {mem_err}", exc_info=True)
            # Non-critical, continue without failing the main task
            
    except Exception as e:
        logging.error(f"Critical error in enhanced_background_chat_task: {str(e)}", exc_info=True)
        # Attempt to notify the client about the error
        try:
            from flask_socketio import emit
            emit('error', {'error': f"Server error: {str(e)}"}, room=conversation_id)
        except Exception as notify_err:
            logging.error(f"Failed to send error notification: {notify_err}")

async def enhance_context_with_memories(
    user_id, conversation_id, user_input, context, nyx_memory
):
    """
    Enhance the context with relevant memories and Nyx's metacognition.
    This makes responses more consistent and personalized.
    """
    # Retrieve relevant memories based on the user input
    memories = await nyx_memory.retrieve_memories(
        query=user_input,
        memory_types=["observation", "semantic", "reflection"],
        limit=5,
        min_significance=3,
        context=context
    )
    
    # Extract memory texts and IDs
    memory_texts = [m["memory_text"] for m in memories]
    memory_ids = [m["id"] for m in memories]
    
    # Generate a narrative about the topic if we have memories
    narrative = None
    if memories:
        narrative_result = await nyx_memory.construct_narrative(
            topic=user_input,
            context=context,
            limit=5
        )
        narrative = narrative_result["narrative"]
    
    # Generate introspection about Nyx's understanding
    introspection = await nyx_memory.generate_introspection()
    
    # Format the enhancement
    enhancement = {
        "memory_context": "\n\n### Nyx's Relevant Memories ###\n" + 
                         "\n".join([f"- {text}" for text in memory_texts]) if memory_texts else "",
        "narrative_context": f"\n\n### Nyx's Narrative Understanding ###\n{narrative}" if narrative else "",
        "introspection_context": f"\n\n### Nyx's Self-Reflection ###\n{introspection['introspection']}" 
                               if introspection and "introspection" in introspection else "",
        "referenced_memory_ids": memory_ids
    }
    
    # Combine all enhancements
    combined_text = ""
    if enhancement["memory_context"]:
        combined_text += enhancement["memory_context"]
    if enhancement["narrative_context"]:
        combined_text += enhancement["narrative_context"]
    if enhancement["introspection_context"]:
        combined_text += enhancement["introspection_context"]
    
    # Also include the enhancement object for additional processing
    enhancement["text"] = combined_text
    
    return enhancement

async def update_narrative_arcs_for_interaction(
    user_id, conversation_id, user_input, ai_response, conn
):
    """
    Update narrative arcs based on the player interaction.
    This helps Nyx maintain coherent storylines.
    """
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
        nyx_memory = MemoryNyxBridge(user_id, conversation_id)
        introspection = await nyx_memory.generate_introspection()
        
        return {
            "status": "success",
            "introspection": introspection.get("introspection", ""),
            "memory_stats": introspection.get("memory_stats", {}),
            "confidence": introspection.get("confidence", 0)
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
