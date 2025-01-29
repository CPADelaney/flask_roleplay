# main.py
import os
from flask import Flask, render_template, request, session, jsonify
from flask_cors import CORS

# Import your route blueprints
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

# If you have a helper function to get DB connections:
from db.connection import get_db_connection

def create_app():
    app = Flask(__name__)

    # Use SECRET_KEY from environment or a fallback for local dev
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

    CORS(app)  # Allow cross-origin requests globally

    # Register your blueprint modules
    app.register_blueprint(new_game_bp, url_prefix="/new_game")
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

    @app.route("/chat")
    def chat_page():
        return render_template("chat.html")

    @app.route("/login", methods=["POST"])
    def login():
        """
        Minimal login route that looks up 'username' & 'password' in the DB.
        If valid, stores user_id in session for subsequent requests.
        """
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

        # Get a DB connection and run a SELECT
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        
        cur.close()
        conn.close()

        if not row:
            return jsonify({"error": "Invalid username"}), 401

        user_id, password_hash = row
        # TODO: Check if 'password' matches 'password_hash' for real security

        # If valid, store user_id in session
        session["user_id"] = user_id

        return jsonify({"message": "Logged in", "user_id": user_id})

    return app


# Create the Flask app instance
app = create_app()

if __name__ == "__main__":
    # Run in debug mode on port 5000, or as specified by environment variables
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
