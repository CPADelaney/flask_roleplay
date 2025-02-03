import os
from flask import Flask, render_template, request, session, jsonify, redirect
from flask_cors import CORS
import logging

# Your blueprint imports
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


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback_dev_key")  # fallback for local dev

    CORS(app)

    # Register your blueprint modules
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
    # ROUTES
    # -----------------

    # A simple route to serve the chat page
    @app.route("/chat")
    def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        return render_template("chat.html")

    # Optional: a dedicated login_page that serves login.html
    @app.route("/login_page", methods=["GET"])
    def login_page():
        return render_template("login.html")  # Create templates/login.html

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
        
        # TODO: For real security, verify the password with a hashing library (bcrypt or similar).
        # if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
        #     return jsonify({"error": "Invalid password"}), 401

        # If valid, store user_id in session
        session["user_id"] = user_id

        return jsonify({"message": "Logged in", "user_id": user_id})\
    
    @app.route("/whoami", methods=["GET"])
    def whoami():
        user_id = session.get("user_id")
        if user_id:
            return jsonify({"logged_in": True, "user_id": user_id}), 200
        else:
            return jsonify({"logged_in": False}), 200

    @app.route("/logout", methods=["POST"])
    def logout():
        # Clear the session so the user is "logged out"
        session.clear()
        return jsonify({"message": "Logged out"}), 200       
    
    @app.route("/register", methods=["POST"])
    def register():
        """
        A minimal endpoint to create a new user in the 'users' table.
        """
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

        # (Optional) Check password complexity, length, etc.
        # For real production, also store a hashed password. Example with bcrypt:
        #
        # import bcrypt
        # hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')     
        
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if username already exists
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        existing = cur.fetchone()
        if existing:
            cur.close()
            conn.close()
            return jsonify({"error": "Username already taken"}), 400

        # Insert new user
        # For demonstration, storing password in plaintext (NOT recommended in production)
        cur.execute("""
            INSERT INTO users (username, password_hash)
            VALUES (%s, %s)
            RETURNING id
        """, (username, password))
        new_user_id = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        # Optionally log them in right away:
        session["user_id"] = new_user_id

        return jsonify({"message": "User registered successfully", "user_id": new_user_id})

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
