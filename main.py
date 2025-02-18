import os
import logging
from flask import Flask, render_template, request, session, jsonify, redirect
from flask_cors import CORS
from celery import Celery

# Blueprint imports
from routes.new_game import new_game_bp
from routes.player_input import player_input_bp
from routes.settings_routes import settings_bp
from routes.knowledge_routes import knowledge_bp
from routes.story_routes import story_bp
from logic.memory_logic import memory_bp
from logic.rule_enforcement import rule_enforcement_bp
from db.admin import admin_bp
from routes.debug import debug_bp
from routes.universal_update import universal_bp
from routes.multiuser_routes import multiuser_bp
from db.connection import get_db_connection

# --- Updated Import for RabbitMQ-based Manager ---
from flask_socketio import SocketIO, join_room
from socketio import KombuManager  # Use KombuManager instead of AMQPManager

def create_celery_app():
    """
    Create and configure a single Celery app instance.
    """
    # Retrieve the RabbitMQ URL from environment variables:
    RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")
    
    celery_app = Celery("my_celery_app")
    celery_app.conf.broker_url = RABBITMQ_URL
    celery_app.conf.result_backend = "rpc://"
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        worker_log_format="%(levelname)s:%(name)s:%(message)s",
        worker_redirect_stdouts_level='INFO'
    )
    return celery_app

# Global SocketIO instance (will be created later)
socketio = None

def create_flask_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback_dev_key")
    CORS(app)

    # Register blueprint modules
    app.register_blueprint(new_game_bp)
    app.register_blueprint(player_input_bp, url_prefix="/player")
    app.register_blueprint(settings_bp, url_prefix="/settings")
    app.register_blueprint(memory_bp, url_prefix="/memory")
    app.register_blueprint(rule_enforcement_bp, url_prefix="/rules")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(knowledge_bp, url_prefix="/knowledge")
    app.register_blueprint(story_bp, url_prefix="/story")
    app.register_blueprint(debug_bp, url_prefix='/debug')
    app.register_blueprint(universal_bp, url_prefix="/universal")
    app.register_blueprint(multiuser_bp, url_prefix="/multiuser")

    # Example Routes
    @app.route("/chat")
    def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        return render_template("chat.html")

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
        return jsonify({"logged_in": bool(user_id), "user_id": user_id}), 200

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
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({"error": "Username already taken"}), 400
        cur.execute("""
            INSERT INTO users (username, password_hash)
            VALUES (%s, %s)
            RETURNING id
        """, (username, password))
        new_user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        session["user_id"] = new_user_id
        return jsonify({"message": "User registered successfully", "user_id": new_user_id})

    # New route: /start_chat for enqueuing the streaming task
    @app.route("/start_chat", methods=["POST"])
    def start_chat():
        if "user_id" not in session:
            return jsonify({"error": "Not logged in"}), 401
        data = request.get_json(force=True)
        user_input = data.get("user_input", "")
        conversation_id = data.get("conversation_id", "default_convo")
        from tasks import stream_openai_tokens_task
        stream_openai_tokens_task.delay(user_input, conversation_id)
        return jsonify({
            "status": "queued",
            "conversation_id": conversation_id
        })

    return app

# Instantiate Celery and Flask
celery_app = create_celery_app()
flask_app = create_flask_app()

# Create a KombuManager for Socket.IO using RabbitMQ
RABBIT_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")
kombu_manager = KombuManager(RABBIT_URL)
socketio = SocketIO(flask_app, cors_allowed_origins="*", client_manager=kombu_manager)

@socketio.on("join")
def on_join(data):
    convo_id = data.get("conversation_id")
    join_room(convo_id)
    print(f"Socket client joined room: {convo_id}")

app = flask_app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    socketio.run(flask_app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
