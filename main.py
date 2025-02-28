# main.py

import os
import logging
import time
from flask import Flask, render_template, session, request, jsonify, redirect
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi

# Blueprint imports
from routes.new_game import new_game_bp
from routes.player_input import player_input_bp, player_input_root_bp
from routes.settings_routes import settings_bp
from routes.knowledge_routes import knowledge_bp
from routes.story_routes import story_bp
from logic.memory_logic import memory_bp
from logic.rule_enforcement import rule_enforcement_bp
from db.admin import admin_bp
from routes.debug import debug_bp
from routes.universal_update import universal_bp
from routes.multiuser_routes import multiuser_bp
from logic.chatgpt_integration import build_message_history
from routes.ai_image_generator import init_app as init_image_routes
from routes.chatgpt_routes import init_app as init_chat_routes
from logic.gpt_image_decision import should_generate_image_for_response
from logic.gpt_image_prompting import get_system_prompt_with_image_guidance
from routes.ai_image_generator import generate_roleplay_image_from_gpt

# DB connection helper
from db.connection import get_db_connection

# Global socketio instance
socketio = None

from logic.chatgpt_integration import get_chatgpt_response
from db.connection import get_db_connection

def background_chat_task(conversation_id, user_input, universal_update):
    """
    Process chat messages in a background task using ChatGPT.
    Uses the sophisticated context generator from story_routes.py.
    """
    try:
        logging.info(f"Starting GPT background chat task for conversation {conversation_id}")
        
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
        
        # Use the advanced context generator from story_routes.py
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
        
        # Call the GPT integration with the enhanced context
        response_data = get_chatgpt_response(conversation_id, aggregator_text, user_input)
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
        
        # Stream the response token by token
        for i in range(0, len(ai_response), 3):
            token = ai_response[i:i+3]
            socketio.emit('new_token', {'token': token}, room=conversation_id)
            socketio.sleep(0.05)
        
        # Store the complete GPT response in the database
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

        # Check if we should generate an image
        should_generate, reason = should_generate_image_for_response(
            user_id, 
            conversation_id, 
            response_data["function_args"]
        )
        
        # Image generation code
        image_result = None
        if should_generate:
            logging.info(f"Generating image for scene: {reason}")
            from routes.ai_image_generator import generate_roleplay_image_from_gpt
            image_result = generate_roleplay_image_from_gpt(
                response_data["function_args"], 
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
        
        # Stream the response token by token
        for i in range(0, len(ai_response), 3):
            token = ai_response[i:i+3]
            socketio.emit('new_token', {'token': token}, room=conversation_id)
            socketio.sleep(0.05)
        
        # Emit the final 'done' event with the full text
        socketio.emit('done', {'full_text': ai_response}, room=conversation_id)
        logging.info(f"Completed streaming GPT response for conversation {conversation_id}")
    
    except Exception as e:
        logging.error(f"Error in background_chat_task: {str(e)}")
        socketio.emit('error', {'error': f"Server error: {str(e)}"}, room=conversation_id)

def create_flask_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback_dev_key")
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(new_game_bp)
    app.register_blueprint(player_input_bp, url_prefix="/player")
    app.register_blueprint(player_input_root_bp)
    app.register_blueprint(settings_bp, url_prefix="/settings")
    app.register_blueprint(memory_bp, url_prefix="/memory")
    app.register_blueprint(rule_enforcement_bp, url_prefix="/rules")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(knowledge_bp, url_prefix="/knowledge")
    app.register_blueprint(story_bp, url_prefix="/story")
    app.register_blueprint(debug_bp, url_prefix="/debug")
    app.register_blueprint(universal_bp, url_prefix="/universal")
    app.register_blueprint(multiuser_bp, url_prefix="/multiuser")
    
    init_image_routes(app)
    init_chat_routes(app)
    
    # HTTP routes
    @app.route("/chat")
    def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        return render_template("chat.html")
    
    @app.route("/start_chat", methods=["POST"])
    def start_chat():
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
    
        data = request.get_json()
        user_input = data.get("user_input")
        conversation_id = data.get("conversation_id")
        universal_update = data.get("universal_update", {})
    
        # Store user message in the database
        try:
            conn = get_db_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                        (conversation_id, "user", user_input)
                    )
                    conn.commit()
        except Exception as e:
            logging.error(f"Database error: {str(e)}")
            return jsonify({"error": "Database error"}), 500
    
        # Start the background chat task (now using enhanced context)
        socketio.start_background_task(background_chat_task, conversation_id, user_input, universal_update)
    
        return jsonify({"status": "success", "message": "Chat started"})
    
    # Authentication routes
    @app.route("/login_page", methods=["GET"])
    def login_page():
        return render_template("login.html")
    
    @app.route("/login", methods=["POST"])
    def login():
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        try:
            conn = get_db_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
                    row = cur.fetchone()
            
            if not row:
                return jsonify({"error": "Invalid username"}), 401
            
            user_id, _ = row
            session["user_id"] = user_id
            return jsonify({"message": "Logged in", "user_id": user_id})
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            return jsonify({"error": "Server error"}), 500
    
    @app.route("/whoami", methods=["GET"])
    def whoami():
        user_id = session.get("user_id")
        if user_id:
            return jsonify({"logged_in": True, "user_id": user_id}), 200
        return jsonify({"logged_in": False}), 200
    
    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return jsonify({"message": "Logged out"}), 200
    
    @app.route("/register", methods=["POST"])
    def register():
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        try:
            conn = get_db_connection()
            with conn:
                with conn.cursor() as cur:
                    # Check if username exists
                    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                    if cur.fetchone():
                        return jsonify({"error": "Username already taken"}), 400
                    
                    # Create new user
                    cur.execute(
                        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
                        (username, password)  # Note: In production, hash the password!
                    )
                    new_user_id = cur.fetchone()[0]
                    conn.commit()
            
            session["user_id"] = new_user_id
            return jsonify({"message": "User registered successfully", "user_id": new_user_id})
        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            return jsonify({"error": "Server error"}), 500
    
    return app

def create_socketio(app):
    """
    Create and configure the SocketIO instance
    """
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', 
                       logger=True, engineio_logger=True, ping_timeout=60)
    
    @socketio.on('connect')
    def handle_connect():
        logging.info("SocketIO: Client connected")
        emit('response', {'data': 'Connected to SocketIO server!'})

    @socketio.on('disconnect')
    def handle_disconnect():
        logging.info("SocketIO: Client disconnected")
    
    @socketio.on('join')
    def handle_join(data):
        conversation_id = data.get('conversation_id')
        if conversation_id:
            join_room(conversation_id)
            logging.info(f"SocketIO: Client joined room {conversation_id}")
            emit('joined', {'room': conversation_id})
        else:
            logging.warning("Client attempted to join room without conversation_id")
            emit('error', {'error': 'Missing conversation_id'})
    
    @socketio.on('message')
    def handle_message(data):
        try:
            logging.info(f"SocketIO: Received message: {data}")
            user_input = data.get('user_input')
            conversation_id = data.get('conversation_id')
            universal_update = data.get('universal_update', {})
            
            if not user_input or not conversation_id:
                emit('error', {'error': 'Missing required fields'})
                return
            
            # Join the room for this conversation
            join_room(conversation_id)
            
            # Store user message in database
            try:
                conn = get_db_connection()
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                            (conversation_id, "user", user_input)
                        )
                        conn.commit()
                logging.info("User message stored in database")
            except Exception as db_error:
                logging.error(f"Database error: {str(db_error)}")
                emit('error', {'error': 'Database error'}, room=conversation_id)
                return
            
            # Start the background task for message processing
            # This now uses the enhanced context generator from story_routes.py
            socketio.start_background_task(
                background_chat_task, 
                conversation_id, 
                user_input, 
                universal_update
            )
            
        except Exception as e:
            logging.error(f"Error in handle_message: {str(e)}")
            emit('error', {'error': f'Server error: {str(e)}'}, room=conversation_id)
    
    # Move the storybeat handler here
    @socketio.on('storybeat')
    def handle_storybeat(data):
        """
        Socket.IO event handler for advanced storybeat processing.
        Incorporates the functionality from next_storybeat in story_routes.py
        but uses Socket.IO for real-time streaming of responses.
        """
        try:
            # Get user session data
            user_id = session.get("user_id")
            if not user_id:
                emit('error', {'error': 'Not authenticated'})
                return
    
            # Extract data from the message
            user_input = data.get("user_input", "").strip()
            conversation_id = data.get("conversation_id")
            player_name = data.get("player_name", "Chase")
            advance_time = data.get("advance_time", False)
            universal_update = data.get("universal_update", {})
    
            if not user_input or not conversation_id:
                emit('error', {'error': 'Missing required fields'})
                return
    
            # Join the room for this conversation
            join_room(conversation_id)
            
            # Emit an acknowledgment that processing has begun
            emit('processing', {'message': 'Processing your request...'})
            
            # Function to do the heavy processing in background
            def process_message_background():
                conn = None
                try:
                    # Create database connection
                    conn = get_db_connection()
                    
                    # Validate conversation ownership
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conversation_id,))
                            row = cur.fetchone()
                            if not row:
                                socketio.emit('error', {'error': f"Conversation {conversation_id} not found"}, room=conversation_id)
                                return
                            if row[0] != user_id:
                                socketio.emit('error', {'error': f"Conversation {conversation_id} not owned by this user"}, room=conversation_id)
                                return
                            
                            # Store user message
                            cur.execute(
                                "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                                (conversation_id, "user", user_input)
                            )
                            conn.commit()
                    
                    # Process universal updates if provided
                    if universal_update:
                        def apply_universal_updates_thread():
                            import asyncio
                            import os
                            import asyncpg
                            from logic.universal_updater import apply_universal_updates_async
                            
                            universal_update["user_id"] = user_id
                            universal_update["conversation_id"] = conversation_id
                            
                            async def run_updates():
                                dsn = os.getenv("DB_DSN")
                                async_conn = await asyncpg.connect(dsn=dsn)
                                try:
                                    return await apply_universal_updates_async(
                                        user_id, conversation_id, universal_update, async_conn
                                    )
                                finally:
                                    await async_conn.close()
                            
                            result = asyncio.run(run_updates())
                            logging.info(f"Universal update result: {result}")
                        
                        # Apply updates before generating response
                        apply_universal_updates_thread()
                    
                    # Check and spawn NPCs if needed
                    from logic.aggregator import get_aggregated_roleplay_context
                    from logic.npc_creation import spawn_multiple_npcs
                    import asyncio
                    
                    # Check NPC count
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT COUNT(*) FROM NPCStats
                                WHERE user_id=%s AND conversation_id=%s AND introduced=FALSE
                            """, (user_id, conversation_id))
                            count = cur.fetchone()[0]
                    
                    # If too few NPCs, spawn more
                    if count < 2:
                        logging.info("Only %d unintroduced NPC(s) found; generating 3 more.", count)
                        # Get aggregator data for environment context
                        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
                        env_desc = aggregator_data.get("currentRoleplay", {}).get("EnvironmentDesc", 
                                                                               "A default environment description.")
                        calendar = aggregator_data.get("calendar", {})
                        day_names = calendar.get("days", 
                                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                        
                        # Spawn NPCs using asyncio
                        async def spawn_npcs_async():
                            return await spawn_multiple_npcs(user_id, conversation_id, env_desc, day_names, count=3)
                        
                        npc_ids = asyncio.run(spawn_npcs_async())
                        logging.info(f"Generated new NPCs: {npc_ids}")
                    
                    # Advance time if requested
                    from logic.time_cycle import advance_time_and_update
                    
                    if advance_time:
                        new_year, new_month, new_day, new_phase = advance_time_and_update(
                            user_id, conversation_id, increment=1
                        )
                        logging.info(f"Advanced time to Y{new_year} M{new_month} D{new_day} {new_phase}")
                    
                    # Get the latest aggregated context (after time advance and NPC creation)
                    aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
                    
                    # Append additional NPC context
                    npc_context_summary = ""
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT npc_name, memory, archetype_extras_summary
                                FROM NPCStats
                                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                            """, (user_id, conversation_id))
                            
                            for row in cur.fetchall():
                                npc_name, memory_json, extras_summary = row
                                mem_text = ""
                                if memory_json:
                                    # Handle the case where memory_json might already be a list
                                    if isinstance(memory_json, list):
                                        mem_list = memory_json
                                    else:
                                        try:
                                            mem_list = json.loads(memory_json)
                                        except Exception as e:
                                            logging.warning(f"Error parsing memory JSON for NPC {npc_name}: {e}")
                                            mem_list = []
                                    
                                    # Convert the memory list to a string
                                    if isinstance(mem_list, list):
                                        mem_text = " ".join(str(item) for item in mem_list)
                                    else:
                                        mem_text = str(mem_list)
                                
                                extra_text = extras_summary if extras_summary else ""
                                npc_context_summary += f"{npc_name}: {mem_text} {extra_text}\n"
                    
                    # Build the complete context
                    from routes.story_routes import build_aggregator_text, gather_rule_knowledge
                    rule_knowledge = gather_rule_knowledge()
                    aggregator_text = build_aggregator_text(aggregator_data, rule_knowledge)
                    
                    if npc_context_summary:
                        aggregator_text += "\nNPC Context:\n" + npc_context_summary
                    
                    # Get response from GPT
                    from logic.chatgpt_integration import get_chatgpt_response
                    response_data = get_chatgpt_response(conversation_id, aggregator_text, user_input)
                    
                    # Extract narrative
                    if response_data["type"] == "function_call":
                        ai_response = response_data["function_args"].get("narrative", "")
                        
                        # Process universal updates from GPT response
                        if response_data["function_args"]:
                            try:
                                import asyncio
                                import asyncpg
                                from logic.universal_updater import apply_universal_updates_async
                                
                                async def apply_updates():
                                    dsn = os.getenv("DB_DSN")
                                    conn = await asyncpg.connect(dsn=dsn, statement_cache_size=0)
                                    try:
                                        return await apply_universal_updates_async(
                                            user_id, 
                                            conversation_id, 
                                            response_data["function_args"],
                                            conn
                                        )
                                    finally:
                                        await conn.close()
                                
                                update_result = asyncio.run(apply_updates())
                                logging.info(f"Applied universal updates from GPT: {update_result}")
                            except Exception as update_error:
                                logging.error(f"Error applying universal updates from GPT: {str(update_error)}")
                                
                        should_generate, reason = should_generate_image_for_response(
                            user_id, 
                            conversation_id, 
                            response_data["function_args"]
                        )
                        
                        # Process image generation if needed
                        if should_generate:
                            logging.info(f"Generating image for scene: {reason}")
                            image_result = generate_roleplay_image_from_gpt(
                                response_data["function_args"], 
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
                                
                    else:
                        ai_response = response_data.get("response", "")
                    
                    # Stream the response token by token
                    for i in range(0, len(ai_response), 3):
                        token = ai_response[i:i+3]
                        socketio.emit('new_token', {'token': token}, room=conversation_id)
                        socketio.sleep(0.05)
                    
                    # Store the complete GPT response in the database
                    try:
                        with conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                                    (conversation_id, "Nyx", ai_response)
                                )
                                conn.commit()
                    except Exception as db_error:
                        logging.error(f"Database error storing GPT response: {str(db_error)}")
                    
                    # Emit the final 'done' event with the full text
                    socketio.emit('done', {'full_text': ai_response}, room=conversation_id)
                    
                except Exception as e:
                    logging.error(f"Error in background processing: {str(e)}", exc_info=True)
                    socketio.emit('error', {'error': f"Server error: {str(e)}"}, room=conversation_id)
                finally:
                    if conn:
                        conn.close()
            
            # Start the background processing task
            socketio.start_background_task(process_message_background)
                    
        except Exception as e:
            logging.error(f"Unexpected error in handle_storybeat: {str(e)}", exc_info=True)
            emit('error', {'error': f"Unexpected server error: {str(e)}"})
    
    return socketio
        
# Optional ASGI wrapper for ASGI servers
asgi_app = None

def init_asgi():
    global asgi_app
    app = create_flask_app()
    socketio = create_socketio(app)
    asgi_app = WsgiToAsgi(app)
    return asgi_app

if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    port = int(os.getenv("PORT", 5000))
    app = create_flask_app()
    socketio = create_socketio(app)
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
