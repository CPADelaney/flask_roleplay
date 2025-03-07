# Integration of Enhanced Nyx Memory System into existing codebase

import logging
import asyncio
from datetime import datetime
import json
import os
from celery_config import celery_app
from logic.nyx_memory import NyxMemoryManager, perform_memory_maintenance

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
    import asyncpg
    
    async def process_all_conversations():
        conn = None
        try:
            dsn = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb")
            conn = await asyncpg.connect(dsn)
            
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
                    await perform_memory_maintenance(user_id, conversation_id)
                    logger.info(f"Memory maintenance completed for user_id={user_id}, conversation_id={conversation_id}")
                except Exception as e:
                    logger.error(f"Error in memory maintenance for user_id={user_id}, conversation_id={conversation_id}: {str(e)}")
                    
                # Brief pause between processing to avoid overloading the database
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in nyx_memory_maintenance_task: {str(e)}")
        finally:
            if conn:
                await conn.close()
                
    asyncio.run(process_all_conversations())
    return {"status": "Memory maintenance completed"}

# -----------------------------------------------------------
# Integration with Game Processing
# -----------------------------------------------------------

# Update to background_chat_task in main.py
def enhanced_background_chat_task(conversation_id, user_input, universal_update):
    """
    Enhanced version of background_chat_task with Nyx memory system integration
    while preserving all existing functionality.
    """
    try:
        logging.info(f"Starting enhanced GPT background chat task for conversation {conversation_id}")
        
        # Retrieve the user_id for this conversation
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conversation_id,))
                result = cur.fetchone()
                if not result:
                    logging.error(f"Conversation {conversation_id} not found")
                    socketio.emit('error', {'error': f"Conversation not found"}, room=conversation_id)
                    return
                user_id = result[0]
            
        # Get player name (defaulting to "Chase")
        player_name = "Chase"  # Default value
        
        # Get current environmental context for the memory system
        context = {}
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT key, value FROM CurrentRoleplay 
                    WHERE user_id = %s AND conversation_id = %s 
                    AND key IN ('CurrentLocation', 'TimeOfDay', 'CurrentYear', 'CurrentMonth', 'CurrentDay')
                """, (user_id, conversation_id))
                for row in cur.fetchall():
                    context[row[0]] = row[1]
        
        # NEW: Initialize Nyx memory manager
        import asyncio
        from logic.nyx_memory_manager import NyxMemoryManager
        
        async def memory_operations():
            # Create NyxMemoryManager instance
            nyx_memory = NyxMemoryManager(user_id, conversation_id)
            
            # Record user input as a memory
            memory_id = await nyx_memory.add_memory(
                memory_text=f"Chase said: {user_input}",
                memory_type="observation",
                significance=4,  # Moderate significance by default
                tags=["player_input"],
                related_entities={"player": player_name},
                context=context
            )
            
            # Get memory enhancement
            from logic.nyx_enhancements_integration import enhance_context_with_memories
            memory_enhancement = await enhance_context_with_memories(
                user_id, conversation_id, user_input, context, nyx_memory
            )
            
            return nyx_memory, memory_enhancement
        
        # Run memory operations asynchronously
        nyx_memory, memory_enhancement = asyncio.run(memory_operations())
        
        # Use the advanced context generator from story_routes.py (EXISTING CODE)
        # First, get the aggregated context using the aggregator
        from logic.aggregator import get_aggregated_roleplay_context
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
        
        # Then, use build_aggregator_text to convert it to a text representation
        from routes.story_routes import build_aggregator_text, gather_rule_knowledge
        
        # Optionally include rule knowledge for deeper context
        rule_knowledge = gather_rule_knowledge()
        
        # Build the full context with rules
        aggregator_text = build_aggregator_text(aggregator_data, rule_knowledge)
        logging.info(f"Built advanced context for conversation {conversation_id}")
        
        # NEW: Combine standard context with memory-enhanced context
        enhanced_context = f"{aggregator_text}\n\n{memory_enhancement['text']}"
        
        # Call the GPT integration with the enhanced context
        response_data = get_chatgpt_response(conversation_id, enhanced_context, user_input)
        logging.info("Received GPT response from ChatGPT integration")
        
        # Extract the narrative from the response
        if response_data["type"] == "function_call":
            # When the response is a function call, extract the narrative field
            ai_response = response_data["function_args"].get("narrative", "")
            
            # Process any universal updates received
            try:
                # Only apply updates if we received function call with args
                if response_data["function_args"]:
                    import asyncio
                    import asyncpg
                    from logic.universal_updater import apply_universal_updates_async
                    
                    async def apply_updates():
                        dsn = os.getenv("DB_DSN")
                        conn = await asyncpg.connect(dsn=dsn, statement_cache_size=0)
                        try:
                            await apply_universal_updates_async(
                                user_id, 
                                conversation_id, 
                                response_data["function_args"], 
                                conn
                            )
                        finally:
                            await conn.close()
                    
                    # Run the async function
                    asyncio.run(apply_updates())
                    logging.info(f"Applied universal updates for conversation {conversation_id}")
            except Exception as update_error:
                logging.error(f"Error applying universal updates: {str(update_error)}")
        else:
            ai_response = response_data.get("response", "")
        
        # NEW: Store Nyx's response in memory
        async def record_response_in_memory():
            await nyx_memory.add_memory(
                memory_text=f"I responded: {ai_response[:200]}...",  # Truncate long responses
                memory_type="observation",
                significance=4,
                tags=["nyx_response"],
                related_entities={"player": player_name},
                context=context
            )
            
            # Perform memory reconsolidation for memories referenced in this interaction
            if memory_enhancement.get("referenced_memory_ids", []):
                for mem_id in memory_enhancement["referenced_memory_ids"]:
                    await nyx_memory.reconsolidate_memory(mem_id, context)
                    
            # Update narrative arcs if they exist
            from logic.nyx_enhancements_integration import update_narrative_arcs_for_interaction
            async_conn = await asyncpg.connect(dsn=os.getenv("DB_DSN"))
            try:
                await update_narrative_arcs_for_interaction(
                    user_id, conversation_id, user_input, ai_response, async_conn
                )
            finally:
                await async_conn.close()
                
        # Record the response asynchronously (don't wait for completion)
        asyncio.create_task(record_response_in_memory())
        
        # Stream the response token by token (ORIGINAL CODE)
        for i in range(0, len(ai_response), 3):
            token = ai_response[i:i+3]
            socketio.emit('new_token', {'token': token}, room=conversation_id)
            socketio.sleep(0.05)
        
        # Store the complete GPT response in the database (ORIGINAL CODE)
        try:
            conn = get_db_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                        (conversation_id, "Nyx", ai_response)
                    )
                    conn.commit()
            logging.info(f"GPT response stored in database for conversation {conversation_id}")
        except Exception as db_error:
            logging.error(f"Database error storing GPT response: {str(db_error)}")

        # Check if we should generate an image (ORIGINAL CODE)
        should_generate, reason = should_generate_image_for_response(
            user_id, 
            conversation_id, 
            response_data["function_args"] if response_data["type"] == "function_call" else {}
        )
        
        # Image generation code (ORIGINAL CODE)
        image_result = None
        if should_generate:
            logging.info(f"Generating image for scene: {reason}")
            from routes.ai_image_generator import generate_roleplay_image_from_gpt
            image_result = generate_roleplay_image_from_gpt(
                response_data["function_args"] if response_data["type"] == "function_call" else {}, 
                user_id, 
                conversation_id
            )
            
            # Emit image to the client
            if image_result and "image_urls" in image_result and image_result["image_urls"]:
                socketio.emit('image', {
                    'image_url': image_result["image_urls"][0],
                    'prompt_used': image_result.get('prompt_used', ''),
                    'reason': reason
                }, room=conversation_id)
                logging.info(f"Image emitted to client: {image_result['image_urls'][0]}")
        
        # Stream the response token by token again (ORIGINAL CODE)
        for i in range(0, len(ai_response), 3):
            token = ai_response[i:i+3]
            socketio.emit('new_token', {'token': token}, room=conversation_id)
            socketio.sleep(0.05)
        
        # Emit the final 'done' event with the full text
        socketio.emit('done', {'full_text': ai_response}, room=conversation_id)
        logging.info(f"Completed streaming GPT response for conversation {conversation_id}")
    
    except Exception as e:
        logging.error(f"Error in enhanced_background_chat_task: {str(e)}")
        socketio.emit('error', {'error': f"Server error: {str(e)}"}, room=conversation_id)

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
    import asyncpg
    
    dsn = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb") 
    conn = await asyncpg.connect(dsn)
    
    try:
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
    finally:
        await conn.close()

# -----------------------------------------------------------
# Integration with routes
# -----------------------------------------------------------

async def nyx_introspection_endpoint(user_id, conversation_id):
    """
    Handler for an API endpoint to get Nyx's introspection.
    Can be integrated into your API routes.
    """
    try:
        nyx_memory = NyxMemoryManager(user_id, conversation_id)
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
