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
import logging

chatgpt_bp = Blueprint('chatgpt_bp', __name__)

@chatgpt_bp.route('/chat', methods=['POST'])
async def chat():
    """Process user message, get GPT response, and handle image generation."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json()
    user_message = data.get('message')
    conversation_id = data.get('conversation_id')
    
    if not user_message or not conversation_id:
        return jsonify({"error": "Missing message or conversation_id"}), 400
    
    try:
        # Get conversation context
        conversation_context = await get_conversation_context(user_id, conversation_id)
        
        # Try to get image-aware prompts, but fall back to regular flow if not available
        formatted_user_prompt = user_message  # Default to plain user message
        try:
            # Only try to use image-aware formatting if the functions exist and work
            system_prompt = get_system_prompt_with_image_guidance(user_id, conversation_id)
            formatted_user_prompt = format_user_prompt_for_image_awareness(
                user_message, 
                conversation_context
            )
            logging.info("Using image-aware prompts")
        except Exception as e:
            logging.info(f"Image prompting not available, using regular flow: {e}")
            # Fall back to regular system prompt (which is handled internally by get_chatgpt_response)
        
        # Get response from GPT using the existing function signature
        try:
            # Always use unified pipeline
            gpt_response_data = await get_chatgpt_response(
                conversation_id=conversation_id,
                aggregator_text="",
                user_input=user_message,
                reflection_enabled=data.get('reflection_enabled', False),
                use_nyx_integration=True  # Force Nyx integration
            )
        
        # Handle the response based on type
        if gpt_response_data['type'] == 'function_call':
            # Extract the narrative from function args
            response_text = gpt_response_data['function_args'].get('narrative', '')
            structured_response = gpt_response_data['function_args']
        else:
            # Plain text response
            response_text = gpt_response_data.get('response', '')
            structured_response = {'response_text': response_text}
        
        # Check if we should generate an image (only if image features are available)
        image_result = None
        image_generation_reason = None
        try:
            # Check if image generation modules are available
            if 'should_generate_image_for_response' in globals() and 'generate_roleplay_image_from_gpt' in globals():
                should_generate, reason = await should_generate_image_for_response(
                    user_id, 
                    conversation_id, 
                    structured_response
                )
                
                # Process image generation if needed
                if should_generate:
                    print(f"Generating image for scene: {reason}")
                    image_result = await generate_roleplay_image_from_gpt(
                        structured_response, 
                        user_id, 
                        conversation_id
                    )
                    image_generation_reason = reason
                    
                    # Save information about this image generation
                    await save_image_generation_info(
                        user_id,
                        conversation_id,
                        reason,
                        image_result
                    )
        except Exception as e:
            print(f"Image generation not available: {e}")
            # Continue without image generation
        
        # Save the user message to conversation history
        await save_message(
            conversation_id=conversation_id,
            sender="user",
            content=user_message,
            structured_content=None
        )
        
        # Save the assistant response to conversation history
        await save_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=response_text,
            structured_content=structured_response
        )
        
        # Process state updates from the response if it was a function call
        if gpt_response_data['type'] == 'function_call' and 'function_args' in gpt_response_data:
            # Use the structured response which contains all the state updates
            await process_state_updates(
                user_id, 
                conversation_id, 
                structured_response  # This contains all the update fields
            )
        
        # Return complete response to frontend
        return jsonify({
            "message": response_text,
            "image": image_result,
            "image_generation_reason": image_generation_reason,
            "structured_data": structured_response,
            "tokens_used": gpt_response_data.get('tokens_used', 0)
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
        rows = await conn.fetch("""
            SELECT sender, content, structured_content, created_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT 10
        """, conversation_id)
        
        messages = []
        for row in rows:
            messages.append({
                "role": "user" if row['sender'] == "user" else "assistant",
                "content": row['content'],
                "structured_content": json.loads(row['structured_content']) if row['structured_content'] else None,
                "timestamp": row['created_at'].isoformat() if row['created_at'] else None
            })
        
        # Reverse to get chronological order
        messages.reverse()
        
        # Get information about the last generated image (if the table exists)
        last_image_info = None
        last_image_timestamp = None
        
        try:
            last_image_row = await conn.fetchrow("""
                SELECT 
                    image_path, 
                    generation_reason, 
                    generated_at
                FROM ImageGenerations
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY generated_at DESC
                LIMIT 1
            """, user_id, conversation_id)
            
            if last_image_row:
                last_image_info = last_image_row['generation_reason']
                last_image_timestamp = last_image_row['generated_at'].timestamp() if last_image_row['generated_at'] else None
        except Exception as e:
            logging.debug(f"ImageGenerations table not available: {e}")
    
    return {
        "messages": messages,
        "last_image_info": last_image_info,
        "last_image_timestamp": last_image_timestamp
    }

async def save_message(conversation_id, sender, content, structured_content=None):
    """Save a message to the conversation history."""
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO messages
            (conversation_id, sender, content, structured_content)
            VALUES ($1, $2, $3, $4)
        """, 
        conversation_id, 
        sender, 
        content, 
        json.dumps(structured_content) if structured_content else None
        )

async def save_image_generation_info(user_id, conversation_id, reason, image_result):
    """Save information about an image generation."""
    # Create the table if it doesn't exist
    async with get_db_connection_context() as conn:
        # Create table if needed
        await conn.execute("""
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
            
            await conn.execute("""
                INSERT INTO ImageGenerations
                (user_id, conversation_id, image_path, prompt_used, generation_reason)
                VALUES ($1, $2, $3, $4, $5)
            """, 
            user_id,
            conversation_id,
            image_path,
            prompt_used,
            reason
            )
    
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

        # Handle NPC updates
        if 'npc_updates' in state_updates:
            for npc_update in state_updates['npc_updates']:
                if 'npc_id' in npc_update:
                    npc_id = npc_update['npc_id']
                    # Use the LoreSystem to propose and enact changes
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="NPCStats",
                        entity_identifier={"id": npc_id},
                        updates=npc_update,
                        reason="GPT state update"
                    )

        # Handle NPC creations
        if 'npc_creations' in state_updates:
            for npc_creation in state_updates['npc_creations']:
                npc_name = npc_creation.get('npc_name')
                if npc_name:
                    # Find or create the NPC
                    npc_id = await canon.find_or_create_npc(ctx, conn, npc_name)
                    # Update with creation data
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="NPCStats",
                        entity_identifier={"id": npc_id},
                        updates=npc_creation,
                        reason="GPT NPC creation"
                    )

        # Handle character stat updates (PlayerStats)
        if 'character_stat_updates' in state_updates:
            stat_updates = state_updates['character_stat_updates']
            player_name = stat_updates.get('player_name', 'Chase')
            stats = stat_updates.get('stats', {})
            
            # Ensure player stats exist
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
            
            # Update each stat
            for stat_name, change in stats.items():
                current_value = await conn.fetchval(
                    f"SELECT {stat_name} FROM PlayerStats WHERE id = $1",
                    player_row['id']
                )
                
                if isinstance(change, str) and (change.startswith('+') or change.startswith('-')):
                    new_value = (current_value or 0) + int(change)
                else:
                    new_value = int(change)
                
                # Clamp value between 0 and 100
                new_value = max(0, min(100, new_value))
                
                await canon.update_player_stat_canonically(
                    ctx, conn, player_name, stat_name, new_value, "GPT Response"
                )

        # Handle relationship updates
        if 'relationship_updates' in state_updates:
            for rel_update in state_updates['relationship_updates']:
                if 'npc_id' in rel_update and 'affiliations' in rel_update:
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="NPCStats",
                        entity_identifier={"id": rel_update['npc_id']},
                        updates={"affiliations": rel_update['affiliations']},
                        reason="GPT relationship update"
                    )

        # Handle location creations
        if 'location_creations' in state_updates:
            for location in state_updates['location_creations']:
                # Create location entries
                location_name = location.get('location_name')
                if location_name:
                    await conn.execute("""
                        INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (user_id, conversation_id, location_name) 
                        DO UPDATE SET 
                            description = EXCLUDED.description,
                            open_hours = EXCLUDED.open_hours
                    """, 
                    user_id, 
                    conversation_id,
                    location_name,
                    location.get('description', ''),
                    json.dumps(location.get('open_hours', []))
                    )

        # Handle event list updates
        if 'event_list_updates' in state_updates:
            for event in state_updates['event_list_updates']:
                # Create event entries
                await conn.execute("""
                    INSERT INTO Events (
                        user_id, conversation_id, event_name, description, 
                        start_time, end_time, location, npc_id, year, month, 
                        day, time_of_day, override_location, fantasy_level
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                user_id,
                conversation_id,
                event.get('event_name'),
                event.get('description'),
                event.get('start_time'),
                event.get('end_time'),
                event.get('location'),
                event.get('npc_id'),
                event.get('year'),
                event.get('month'),
                event.get('day'),
                event.get('time_of_day'),
                event.get('override_location'),
                event.get('fantasy_level', 'realistic')
                )

        # Handle inventory updates
        if 'inventory_updates' in state_updates:
            inv_updates = state_updates['inventory_updates']
            player_name = inv_updates.get('player_name', 'Chase')
            
            # Add items
            if 'added_items' in inv_updates:
                for item in inv_updates['added_items']:
                    if isinstance(item, str):
                        item_name = item
                        item_desc = ""
                        item_effect = ""
                        category = "general"
                    else:
                        item_name = item.get('item_name', '')
                        item_desc = item.get('item_description', '')
                        item_effect = item.get('item_effect', '')
                        category = item.get('category', 'general')
                    
                    if item_name:
                        await conn.execute("""
                            INSERT INTO PlayerInventory (
                                user_id, conversation_id, player_name, 
                                item_name, item_description, item_effect, category
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        user_id, conversation_id, player_name,
                        item_name, item_desc, item_effect, category
                        )
            
            # Remove items
            if 'removed_items' in inv_updates:
                for item in inv_updates['removed_items']:
                    if isinstance(item, str):
                        item_name = item
                    else:
                        item_name = item.get('item_name', '')
                    
                    if item_name:
                        await conn.execute("""
                            DELETE FROM PlayerInventory
                            WHERE user_id = $1 AND conversation_id = $2 
                            AND player_name = $3 AND item_name = $4
                        """,
                        user_id, conversation_id, player_name, item_name
                        )

        # Handle quest updates
        if 'quest_updates' in state_updates:
            for quest in state_updates['quest_updates']:
                await conn.execute("""
                    INSERT INTO Quests (
                        user_id, conversation_id, quest_id, quest_name, 
                        status, progress_detail, quest_giver, reward
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (user_id, conversation_id, quest_id)
                    DO UPDATE SET
                        quest_name = EXCLUDED.quest_name,
                        status = EXCLUDED.status,
                        progress_detail = EXCLUDED.progress_detail,
                        quest_giver = EXCLUDED.quest_giver,
                        reward = EXCLUDED.reward
                """,
                user_id, conversation_id,
                quest.get('quest_id'),
                quest.get('quest_name'),
                quest.get('status'),
                quest.get('progress_detail'),
                quest.get('quest_giver'),
                quest.get('reward')
                )

        # Handle social links
        if 'social_links' in state_updates:
            for link in state_updates['social_links']:
                # Process social link updates
                pass  # Implement based on your social links system

        # Handle perk unlocks
        if 'perk_unlocks' in state_updates:
            for perk in state_updates['perk_unlocks']:
                player_name = perk.get('player_name', 'Chase')
                await conn.execute("""
                    INSERT INTO PlayerPerks (
                        user_id, conversation_id, player_name,
                        perk_name, perk_description, perk_effect
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                """,
                user_id, conversation_id, player_name,
                perk.get('perk_name'),
                perk.get('perk_description'),
                perk.get('perk_effect')
                )

        # Handle activity updates
        if 'activity_updates' in state_updates:
            for activity in state_updates['activity_updates']:
                # Process activity updates
                pass  # Implement based on your activities system

        # Handle journal updates
        if 'journal_updates' in state_updates:
            for entry in state_updates['journal_updates']:
                await conn.execute("""
                    INSERT INTO PlayerJournal (
                        user_id, conversation_id, entry_type, 
                        entry_text, fantasy_flag, intensity_level
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id, conversation_id,
                entry.get('entry_type'),
                entry.get('entry_text'),
                entry.get('fantasy_flag', False),
                entry.get('intensity_level', 0)
                )

        # Handle roleplay updates (CurrentRoleplay)
        if 'roleplay_updates' in state_updates:
            for key, value in state_updates['roleplay_updates'].items():
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                """,
                user_id, conversation_id, key, json.dumps(value)
                )

        # Handle Chase schedule updates
        if 'ChaseSchedule' in state_updates:
            # Store Chase's schedule
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'ChaseSchedule', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """,
            user_id, conversation_id, json.dumps(state_updates['ChaseSchedule'])
            )

        # Handle main quest update
        if 'MainQuest' in state_updates:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'MainQuest', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """,
            user_id, conversation_id, json.dumps(state_updates['MainQuest'])
            )

        # Handle player role update
        if 'PlayerRole' in state_updates:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'PlayerRole', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """,
            user_id, conversation_id, json.dumps(state_updates['PlayerRole'])
            )

def init_app(app):
    """Initialize the chatgpt routes with the Flask app."""
    app.register_blueprint(chatgpt_bp, url_prefix='/api')
