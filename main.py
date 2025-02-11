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

# DB connection helper
from db.connection import get_db_connection


def create_celery_app():
    """Create and return a Celery application instance."""
    celery_app = Celery("my_celery_app")

    # Example settings—adjust to your environment:
    celery_app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # If you have a separate config file or want to discover tasks from multiple modules:
    # celery_app.config_from_object('celeryconfig')
    # celery_app.autodiscover_tasks(['some_package'])
    return celery_app


def create_flask_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback_dev_key")  # fallback for local dev

    # Enable CORS for all routes
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

    # -----------------
    # Example Routes
    # -----------------

    @app.route("/test_task", methods=["GET"])
    def run_test_task():
        """
        Example: triggers our Celery task asynchronously.
        """
        # Instead of importing tasks from tasks.py, we'll call the below inlined @celery_app.task
        result = test_task.delay()
        return jsonify({"job_id": result.id})

    @app.route("/chat")
    def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        return render_template("chat.html")

    @app.route("/login_page", methods=["GET"])
    def login_page():
        return render_template("login.html")  # Create a templates/login.html file

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

        user_id, password_hash = row
        # NOTE: For real production, use a proper password hashing library.

        session["user_id"] = user_id
        return jsonify({"message": "Logged in", "user_id": user_id})

    @app.route("/whoami", methods=["GET"])
    def whoami():
        user_id = session.get("user_id")
        if user_id:
            return jsonify({"logged_in": True, "user_id": user_id}), 200
        else:
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

        # Check if username already exists
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        existing = cur.fetchone()
        if existing:
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

    return app


#
# Instantiate Celery and Flask
#
celery_app = create_celery_app()
app = create_flask_app()


#
# Example Celery task
#
@celery_app.task
def test_task():
    """
    Example Celery task: sleeps a bit, returns a simple string.
    You can add more tasks as needed or in a separate tasks.py.
    """
    import time
    time.sleep(2)
    return "Task finished successfully!"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Start Flask normally
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
