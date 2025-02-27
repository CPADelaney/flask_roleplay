# routes/chatgpt_routes.py

import json
import time
from flask import Blueprint, request, jsonify, session
from logic.chatgpt_integration import get_gpt_response
from logic.image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from db.connection import get_db_connection
from logic.gpt_prompting import (
    get_system_prompt_with_image_guidance,
    format_user_prompt_for_image_awareness
)

chatgpt_bp = Blueprint('chatgpt_bp', __name__)

@chatgpt_bp.route('/chat', methods=['POST'])
def chat():
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
    conversation_context = get_conversation_context(user_id, conversation_id)
    
    # Prepare prompts with image awareness
    system_prompt = get_system_prompt_with_image_guidance(user_id, conversation_id)
    formatted_user_prompt = format_user_prompt_for_image_awareness(
        user_message, 
        conversation_context
    )
    
    # Get response from GPT
    gpt_response_json = get_gpt_response(
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
        should_generate, reason = should_generate_image_for_response(
            user_id, 
            conversation_id, 
            gpt_response
        )
        
        # Process image generation if needed
        image_result = None
        if should_generate:
            print(f"Generating image for scene: {reason}")
            image_result = generate_roleplay_image_from_gpt(
                gpt_response, 
                user_id, 
                conversation_id
            )
            
            # Save information about this image generation
            save_image_generation_info(
                user_id,
                conversation_id,
                reason,
                image_result
            )
        
        # Save the GPT response to conversation history
        save_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=response_text,
            structured_content=gpt_response
        )
        
        # Process state updates from the response
        if 'state_updates' in gpt_response:
            process_state_updates(
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

def get_conversation_context(user_id, conversation_id):
    """Get the conversation context including message history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get the last N messages from this conversation
    cursor.execute("""
        SELECT sender, content, structured_content, created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY created_at DESC
        LIMIT 10
    """, (conversation_id,))
    
    messages = []
    for row in cursor.fetchall():
        messages.append({
            "role": "user" if row[0] == "user" else "assistant",
            "content": row[1],
            "structured_content": row[2],
            "timestamp": row[3].isoformat() if row[3] else None
        })
    
    # Reverse to get chronological order
    messages.reverse()
    
    # Get information about the last generated image
    cursor.execute("""
        SELECT 
            image_path, 
            generation_reason, 
            generated_at
        FROM ImageGenerations
        WHERE user_id = %s AND conversation_id = %s
        ORDER BY generated_at DESC
        LIMIT 1
    """, (user_id, conversation_id))
    
    last_image_row = cursor.fetchone()
    last_image_info = None
    last_image_timestamp = None
    
    if last_image_row:
        last_image_info = last_image_row[1]  # The reason for generation
        last_image_timestamp = last_image_row[2].timestamp() if last_image_row[2] else None
    
    cursor.close()
    conn.close()
    
    return {
        "messages": messages,
        "last_image_info": last_image_info,
        "last_image_timestamp": last_image_timestamp
    }

def save_message(conversation_id, sender, content, structured_content=None):
    """Save a message to the conversation history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO messages
        (conversation_id, sender, content, structured_content)
        VALUES (%s, %s, %s, %s)
    """, (
        conversation_id, 
        sender, 
        content, 
        json.dumps(structured_content) if structured_content else None
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def save_image_generation_info(user_id, conversation_id, reason, image_result):
    """Save information about an image generation."""
    # Create the table if it doesn't exist
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create table if needed
    cursor.execute("""
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
        
        cursor.execute("""
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
    
    conn.commit()
    cursor.close()
    conn.close()
    
    # Also update session for rate limiting
    recent_generations = session.get('recent_image_generations', [])
    recent_generations.append(time.time())
    session['recent_image_generations'] = recent_generations

def process_state_updates(user_id, conversation_id, state_updates):
    """Process state updates from GPT response."""
    if not state_updates:
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Process NPC stat updates
    if 'NPCStats' in state_updates:
        for npc_name, updates in state_updates['NPCStats'].items():
            # Get the NPC ID
            cursor.execute("""
                SELECT npc_id FROM NPCStats
                WHERE user_id = %s AND conversation_id = %s AND npc_name = %s
            """, (user_id, conversation_id, npc_name))
            
            npc_row = cursor.fetchone()
            if npc_row:
                npc_id = npc_row[0]
                
                # Build the update query
                update_fields = []
                update_values = []
                
                for key, value in updates.items():
                    update_fields.append(f"{key} = %s")
                    update_values.append(value)
                
                if update_fields:
                    update_query = f"""
                        UPDATE NPCStats
                        SET {', '.join(update_fields)}
                        WHERE npc_id = %s
                    """
                    cursor.execute(update_query, update_values + [npc_id])
            else:
                # NPC doesn't exist yet, create it
                if 'introduced' not in updates:
                    updates['introduced'] = True
                
                fields = ['user_id', 'conversation_id', 'npc_name'] + list(updates.keys())
                placeholders = ['%s', '%s', '%s'] + ['%s'] * len(updates)
                values = [user_id, conversation_id, npc_name] + list(updates.values())
                
                insert_query = f"""
                    INSERT INTO NPCStats
                    ({', '.join(fields)})
                    VALUES ({', '.join(placeholders)})
                """
                cursor.execute(insert_query, values)
    
    # Process Player stat updates
    if 'PlayerStats' in state_updates:
        for player_name, updates in state_updates['PlayerStats'].items():
            # Get player record
            cursor.execute("""
                SELECT id FROM PlayerStats
                WHERE user_id = %s AND conversation_id = %s AND player_name = %s
            """, (user_id, conversation_id, player_name))
            
            player_row = cursor.fetchone()
            
            if player_row:
                # Player exists, update stats
                player_id = player_row[0]
                
                for stat_name, change in updates.items():
                    # Handle relative changes (e.g., +5, -3)
                    if isinstance(change, str) and (change.startswith('+') or change.startswith('-')):
                        # Get current value
                        cursor.execute(f"""
                            SELECT {stat_name} FROM PlayerStats
                            WHERE id = %s
                        """, (player_id,))
                        
                        current_value_row = cursor.fetchone()
                        if current_value_row and current_value_row[0] is not None:
                            current_value = current_value_row[0]
                            change_value = int(change)
                            new_value = current_value + change_value
                            
                            # Update with bounds checking
                            cursor.execute(f"""
                                UPDATE PlayerStats
                                SET {stat_name} = GREATEST(0, LEAST(100, %s))
                                WHERE id = %s
                            """, (new_value, player_id))
                            
                            # Log significant stat changes
                            if abs(change_value) >= 5:
                                cursor.execute("""
                                    INSERT INTO StatsHistory
                                    (user_id, conversation_id, player_name, stat_name, old_value, new_value, cause)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    user_id, 
                                    conversation_id, 
                                    player_name, 
                                    stat_name, 
                                    current_value, 
                                    new_value,
                                    "GPT Response"
                                ))
                    else:
                        # Absolute value
                        cursor.execute(f"""
                            UPDATE PlayerStats
                            SET {stat_name} = GREATEST(0, LEAST(100, %s))
                            WHERE id = %s
                        """, (change, player_id))
    
    conn.commit()
    cursor.close()
    conn.close()

def init_app(app):
    """Initialize the chatgpt routes with the Flask app."""
    app.register_blueprint(chatgpt_bp, url_prefix='/api')
