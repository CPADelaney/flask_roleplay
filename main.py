# main.py

import os
import logging
from flask import Flask, render_template, session, request, jsonify, redirect
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi

# Import your blueprint modules
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

# Import your database connection helper
from db.connection import get_db_connection

def create_flask_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback_dev_key")
    
    # Enable CORS for all routes.
    CORS(app)
    
    # Register blueprint modules.
    app.register_blueprint(new_game_bp)
    app.register_blueprint(player_input_bp, url_prefix="/player")
    app.register_blueprint(player_input_root_bp)
    app.register_blueprint(settings_bp, url_prefix="/settings")
    app.register_blueprint(memory_bp, url_prefix="/memory")
    app.register_blueprint(rule_enforcement_bp, url_prefix="/rules")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(knowledge_bp, url_prefix="/knowledge")
    app.register_blueprint(story_bp, url_prefix="/story")
    app.register_blueprint(debug_bp, url_prefix='/debug')
    app.register_blueprint(universal_bp, url_prefix="/universal")
    app.register_blueprint(multiuser_bp, url_prefix="/multiuser")
    
    # Define HTTP routes.
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
        
        # Store the user's message in the database.
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
            (conversation_id, "user", user_input)
        )
        conn.commit()
        cur.close()
        conn.close()
        
        # Emit an event to notify the Socket.IO connection that a new chat has started.
        socketio.emit(
            'chat_started',
            {
                'conversation_id': conversation_id,
                'user_input': user_input,
                'universal_update': universal_update
            },
            room=conversation_id
        )
        return jsonify({"status": "success", "message": "Chat started"})

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
    
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
    
        if not row:
            return jsonify({"error": "Invalid username"}), 401
    
        user_id, _ = row
        session["user_id"] = user_id
        return jsonify({"message": "Logged in", "user_id": user_id})
    
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
    
        conn = get_db_connection()
        cur = conn.cursor()
        # Check if username already exists.
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({"error": "Username already taken"}), 400
    
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (username, password)
        )
        new_user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
    
        session["user_id"] = new_user_id
        return jsonify({"message": "User registered successfully", "user_id": new_user_id})
    
    return app

# Create the Flask app.
app = create_app()
create_flask_app = create_app

# Create the Socket.IO instance (using eventlet for async).
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)

# Socket.IO Event Handlers.
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

@socketio.on('message')
def handle_message(data):
    logging.info(f"SocketIO: Received message: {data}")
    user_input = data.get('user_input')
    conversation_id = data.get('conversation_id')
    
    if not user_input or not conversation_id:
        emit('error', {'error': 'Missing required fields'})
        return
    
    join_room(conversation_id)
    
    # Simulate a response (replace this with your actual GPT integration logic).
    response = f"Echo: {user_input}"
    
    # Stream the response token by token.
    for i in range(0, len(response), 2):
        token = response[i:i+2]
        emit('new_token', {'token': token}, room=conversation_id)
        socketio.sleep(0.1)
    
    # Send the "done" event when complete.
    emit('done', {'full_text': response}, room=conversation_id)

@socketio.on('chat_started')
def handle_chat_started(data):
    conversation_id = data.get('conversation_id')
    user_input = data.get('user_input')
    universal_update = data.get('universal_update', {})
    
    # Start a background task to process the chat response.
    socketio.start_background_task(background_chat_task, conversation_id, user_input, universal_update)

def background_chat_task(conversation_id, user_input, universal_update):
    # For demonstration purposes, we simulate a GPT response.
    ai_response = "This is a sample response from Nyx."
    
    # Stream the response token by token.
    for i in range(0, len(ai_response), 3):
        token = ai_response[i:i+3]
        socketio.emit('new_token', {'token': token}, room=conversation_id)
        socketio.sleep(0.1)
    
    # Store the AI message in the database.
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
        (conversation_id, "Nyx", ai_response)
    )
    conn.commit()
    cur.close()
    conn.close()
    
    # Emit the "done" event when complete.
    socketio.emit('done', {'full_text': ai_response}, room=conversation_id)

# Optional ASGI wrapper (if you need to run in an ASGI server).
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
