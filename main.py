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

# DB connection helper
from db.connection import get_db_connection

# Global socketio instance
socketio = None

def background_chat_task(conversation_id, user_input, universal_update):
    """
    Process chat messages in a background task using ChatGPT,
    stream the generated response token by token, store it in the DB,
    and emit a 'done' event when finished.
    """
    try:
        logging.info(f"Starting GPT background chat task for conversation {conversation_id}")
        
        # 1. Build the conversation history for context.
        # You might pull messages from your DB and use build_message_history to format them.
        history = build_message_history(conversation_id)
        # Append the latest user message
        history.append({"role": "user", "content": user_input})
        logging.info(f"Built conversation history with {len(history)} messages")
        
        # 2. Get the GPT response.
        # get_chatgpt_response should take the conversation history and return a full response.
        ai_response = get_chatgpt_response(history)
        logging.info("Received GPT response")
        
        # 3. Stream the response token by token.
        # Here we simulate token streaming by splitting the response text.
        for i in range(0, len(ai_response), 3):
            token = ai_response[i:i+3]
            socketio.emit('new_token', {'token': token}, room=conversation_id)
            # Use a small sleep to simulate a natural token delay.
            socketio.sleep(0.05)
        
        # 4. Store the complete GPT response in the database.
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
        
        # 5. Emit the final 'done' event with the full GPT response.
        socketio.emit('done', {'full_text': ai_response}, room=conversation_id)
        logging.info(f"Completed streaming GPT response for conversation {conversation_id}")
        
    except Exception as e:
        logging.error(f"Error in background_chat_task: {str(e)}")
        socketio.emit('error', {'error': f'Server error: {str(e)}'}, room=conversation_id)

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
    
        # Start the background chat task
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
            socketio.start_background_task(
                background_chat_task, 
                conversation_id, 
                user_input, 
                universal_update
            )
            
        except Exception as e:
            logging.error(f"Error in handle_message: {str(e)}")
            emit('error', {'error': f'Server error: {str(e)}'}, room=conversation_id)
    
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
