# routes/chatgpt_routes.py

import json
import time
from quart import Blueprint, request, jsonify, session
from logic.chatgpt_integration import get_chatgpt_response
from logic.gpt_image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from db.connection import get_db_connection_context
from logic.gpt_image_prompting import get_system_prompt_with_image_guidance, format_user_prompt_for_image_awareness
from lore.core.lore_system import LoreSystem
from lore.core import canon

chatgpt_bp = Blueprint('chatgpt_bp', __name__)

@chatgpt_bp.route('/chat', methods=['POST'])
async def chat():
    """Process user message, get GPT response, and handle image generation."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    user_message = data.get('message')
    conversation_id = data.get('conversation_id')
    
    if not user_message or not conversation_id:
        return jsonify({"error": "Missing message or conversation_id"}), 400
    
    # Get conversation context
    conversation_context = await get_conversation_context(user_id, conversation_id)
    
    # Prepare prompts with image awareness
    system_prompt = await get_system_prompt_with_image_guidance(user_id, conversation_id)
    formatted_user_prompt = await format_user_prompt_for_image_awareness(
        user_message, 
        conversation_context
    )
    
    # Get response from GPT
    gpt_response_json = await get_gpt_response(
        system_prompt=system_prompt,
        user_message=formatted_user_prompt,
        conversation_history=conversation_context.get('messages', [])
    )
    
    try:
        # Parse the response
        if isinstance(gpt_response_json, str):
            gpt_response = json.loads(gpt_response_json)
        else:
            gpt_response = gpt_response_json
        
        # Extract the narrative text to show the user
        response_text = gpt_response.get('response_text', '')
        
        # Check if we should generate an image
        should_generate, reason = await should_generate_image_for_response(
            user_id, 
            conversation_id, 
            gpt_response
        )
        
        # Process image generation if needed
        image_result = None
        if should_generate:
            print(f"Generating image for scene: {reason}")
            image_result = await generate_roleplay_image_from_gpt(
                gpt_response, 
                user_id, 
                conversation_id
            )
            
            # Save information about this image generation
            await save_image_generation_info(
                user_id,
                conversation_id,
                reason,
                image_result
            )
        
        # Save the GPT response to conversation history
        await save_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=response_text,
            structured_content=gpt_response
        )
        
        # Process state updates from the response
        if 'state_updates' in gpt_response:
            await process_state_updates(
                user_id, 
                conversation_id, 
                gpt_response['state_updates']
            )
        
        # Return complete response to frontend
        return jsonify({
            "message": response_text,
            "image": image_result,
            "image_generation_reason": reason if should_generate else None,
            "structured_data": gpt_response
        })
        
    except Exception as e:
        print(f"Error processing GPT response: {e}")
        return jsonify({
            "error": "Failed to process response",
            "message": "Something went wrong processing the response."
        }), 500

async def get_conversation_context(user_id, conversation_id):
    """Get the conversation context including message history."""
    async with get_db_connection_context() as conn:
        # Get the last N messages from this conversation
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT sender, content, structured_content, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT 10
            """, (conversation_id,))
            
            rows = await cursor.fetchall()
        
        messages = []
        for row in rows:
            messages.append({
                "role": "user" if row[0] == "user" else "assistant",
                "content": row[1],
                "structured_content": row[2],
                "timestamp": row[3].isoformat() if row[3] else None
            })
        
        # Reverse to get chronological order
        messages.reverse()
        
        # Get information about the last generated image
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT 
                    image_path, 
                    generation_reason, 
                    generated_at
                FROM ImageGenerations
                WHERE user_id = %s AND conversation_id = %s
                ORDER BY generated_at DESC
                LIMIT 1
            """, (user_id, conversation_id))
            
            last_image_row = await cursor.fetchone()
        
        last_image_info = None
        last_image_timestamp = None
        
        if last_image_row:
            last_image_info = last_image_row[1]  # The reason for generation
            last_image_timestamp = last_image_row[2].timestamp() if last_image_row[2] else None
    
    return {
        "messages": messages,
        "last_image_info": last_image_info,
        "last_image_timestamp": last_image_timestamp
    }

async def save_message(conversation_id, sender, content, structured_content=None):
    """Save a message to the conversation history."""
    async with get_db_connection_context() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO messages
                (conversation_id, sender, content, structured_content)
                VALUES (%s, %s, %s, %s)
            """, (
                conversation_id, 
                sender, 
                content, 
                json.dumps(structured_content) if structured_content else None
            ))
        await conn.commit()

async def save_image_generation_info(user_id, conversation_id, reason, image_result):
    """Save information about an image generation."""
    # Create the table if it doesn't exist
    async with get_db_connection_context() as conn:
        # Create table if needed
        async with conn.cursor() as cursor:
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS ImageGenerations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    prompt_used TEXT,
                    generation_reason TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
        
        # Insert the record
        if image_result and 'image_urls' in image_result and image_result['image_urls']:
            image_path = image_result['image_urls'][0]
            prompt_used = image_result.get('prompt_used', '')
            
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO ImageGenerations
                    (user_id, conversation_id, image_path, prompt_used, generation_reason)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    user_id,
                    conversation_id,
                    image_path,
                    prompt_used,
                    reason
                ))
        
        await conn.commit()
    
    # Also update session for rate limiting
    recent_generations = session.get('recent_image_generations', [])
    recent_generations.append(time.time())
    session['recent_image_generations'] = recent_generations

async def process_state_updates(user_id, conversation_id, state_updates):
    """Process state updates from GPT response using canonical APIs."""
    if not state_updates:
        return

    class UpdateContext:
        def __init__(self, user_id, conversation_id):
            self.user_id = user_id
            self.conversation_id = conversation_id

    ctx = UpdateContext(user_id, conversation_id)

    async with get_db_connection_context() as conn:
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)

        if 'NPCStats' in state_updates:
            for npc_name, updates in state_updates['NPCStats'].items():
                npc_row = await conn.fetchrow(
                    """
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_name = $3
                    """,
                    user_id, conversation_id, npc_name
                )

                if npc_row:
                    npc_id = npc_row['npc_id']
                else:
                    npc_id = await canon.find_or_create_npc(ctx, conn, npc_name)

                await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCStats",
                    entity_identifier={"id": npc_id},
                    updates=updates,
                    reason="GPT state update"
                )

        if 'PlayerStats' in state_updates:
            for player_name, updates in state_updates['PlayerStats'].items():
                player_row = await conn.fetchrow(
                    """
                        SELECT id FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """,
                    user_id, conversation_id, player_name
                )

                if not player_row:
                    await canon.find_or_create_player_stats(ctx, conn, player_name)
                    player_row = await conn.fetchrow(
                        """
                            SELECT id FROM PlayerStats
                            WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                        """,
                        user_id, conversation_id, player_name
                    )

                for stat_name, change in updates.items():
                    current_value = await conn.fetchval(
                        f"SELECT {stat_name} FROM PlayerStats WHERE id = $1",
                        player_row['id']
                    )

                    if isinstance(change, str) and (change.startswith('+') or change.startswith('-')):
                        new_value = (current_value or 0) + int(change)
                    else:
                        new_value = int(change)

                    new_value = max(0, min(100, new_value))

                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, stat_name, new_value, "GPT Response"
                    )

def init_app(app):
    """Initialize the chatgpt routes with the Flask app."""
    app.register_blueprint(chatgpt_bp, url_prefix='/api')
