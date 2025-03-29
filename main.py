# main.py

import os
import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional

# Flask and related imports
from flask import Flask, render_template, session, request, jsonify, redirect
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
# Removed WsgiToAsgi as we use eventlet
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from prometheus_flask_exporter import PrometheusMetrics
from flasgger import Swagger
from datetime import timedelta

# Security
import bcrypt
import secrets
import atexit

# External services
import asyncpg # Use asyncpg directly where needed
from redis import Redis # Keep Redis sync for now, unless heavy usage requires aioredis
from celery import Celery # Keep Celery object import

# Blueprint imports (ensure these use asyncpg in their async routes)
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
from routes.nyx_agent_routes import nyx_agent_bp
from routes.conflict_routes import conflict_bp
from routes.npc_learning_routes import npc_learning_bp

# MCP Orchestrator
from mcp_orchestrator import MCPOrchestrator

# NPC creation / learning
from npcs.new_npc_creation import NPCCreationHandler, RunContextWrapper
from npcs.npc_learning_adaptation import NPCLearningManager

# OpenAI and image generation
from logic.chatgpt_integration import build_message_history
from routes.ai_image_generator import init_app as init_image_routes, generate_roleplay_image_from_gpt
from routes.chatgpt_routes import init_app as init_chat_routes
from logic.gpt_image_decision import should_generate_image_for_response
from logic.gpt_image_prompting import get_system_prompt_with_image_guidance

# Nyx integration
from logic.nyx_enhancements_integration import initialize_nyx_memory_system # Keep this async
from nyx.integrate import get_central_governance
from logic.conflict_system.conflict_integration import register_enhanced_integration

# DB connection helper - CRITICAL: Ensure these work with asyncpg pool
from db.connection import (
    # get_db_connection, # REMOVE sync connection getter if possible
    initialize_connection_pool, # Should init asyncpg pool
    close_connection_pool, # Should close asyncpg pool
    get_db_connection_context # Use this async context manager
)

# Middleware
from middleware.rate_limiting import rate_limit, ip_block_list
from middleware.validation import validate_request

# Config and utilities
from .config.settings import config
from .utils.health_check import HealthCheck

logger = logging.getLogger(__name__)

# Global placeholder for SocketIO instance
socketio = None

# Database DSN
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.critical("DB_DSN environment variable not set!")
    # Potentially exit or raise an error here depending on requirements

# Removed ConnectivityManager for brevity, can be added back if needed, ensure it uses async checks

###############################################################################
# BACKGROUND TASKS (Called via SocketIO or Celery)
###############################################################################

# Ensure this task ONLY uses asyncpg for DB access
async def background_chat_task(conversation_id, user_input, user_id, universal_update=None):
    """
    Background task for processing chat messages using Nyx agent with OpenAI integration.
    Uses asyncpg for database operations.
    """
    global socketio # Need access to socketio instance to emit

    try:
        logger.info(f"Starting background_chat_task for conv_id={conversation_id}, user_id={user_id}")

        # Get aggregator context (ensure this function is async or thread-safe if it hits DB)
        from logic.aggregator import get_aggregated_roleplay_context # Assuming sync for now
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase") # Adjust player name if needed

        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
            "player_name": aggregator_data.get("playerName", "Chase"),
            "npc_present": aggregator_data.get("npcsPresent", []),
            "aggregator_data": aggregator_data
        }

        # Apply universal update if provided (ensure this uses asyncpg)
        if universal_update:
            context["universal_update"] = universal_update
            try:
                from logic.universal_updater import apply_universal_updates_async # Needs to be async
                async with get_db_connection_context() as conn:
                    await apply_universal_updates_async(
                        user_id,
                        conversation_id,
                        universal_update,
                        conn # Pass the connection
                    )
                logger.info(f"Applied universal updates for conv_id={conversation_id}")
                # Refresh aggregator data post-update
                aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
                context["aggregator_data"] = aggregator_data
            except Exception as update_err:
                 logger.error(f"Error applying universal updates in background task: {update_err}", exc_info=True)
                 # Decide if to continue or emit error

        # Process the user_input with OpenAI-enhanced Nyx agent
        # Ensure this function is async
        from nyx.nyx_agent_sdk import process_user_input_with_openai
        response = await process_user_input_with_openai(user_id, conversation_id, user_input, context)

        if not response or not response.get("success", False):
            error_msg = response.get("error", "Unknown error from Nyx agent") if response else "Empty response from Nyx agent"
            logger.error(f"Nyx agent failed for conv_id={conversation_id}: {error_msg}")
            if socketio:
                 socketio.emit('error', {'error': error_msg}, room=str(conversation_id))
            return

        # Extract the message content
        message_content = response.get("message", "")
        if not message_content and "function_args" in response:
            message_content = response["function_args"].get("narrative", "")
        logger.debug(f"Nyx response for conv_id={conversation_id}: {message_content[:100]}...")

        # Store the Nyx response in DB using asyncpg
        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """INSERT INTO messages (conversation_id, sender, content, created_at)
                       VALUES ($1, $2, $3, NOW())""",
                    conversation_id, "Nyx", message_content
                )
            logger.info(f"Stored Nyx response for conv_id={conversation_id}")
        except Exception as db_err:
            logger.error(f"DB Error storing Nyx response for conv_id={conversation_id}: {db_err}", exc_info=True)
            # Continue but log the error

        # Check if we should generate an image
        should_generate = response.get("generate_image", False)
        if "function_args" in response and "image_generation" in response["function_args"]:
            img_settings = response["function_args"]["image_generation"]
            should_generate = should_generate or img_settings.get("generate", False)

        # Generate image if needed (ensure this is async)
        if should_generate:
            logger.info(f"Image generation triggered for conv_id={conversation_id}")
            try:
                img_data = {
                    "narrative": message_content,
                    "image_generation": {
                        "generate": True,
                        "priority": "medium",
                        "focus": "balanced",
                        "framing": "medium_shot",
                        "reason": "Narrative moment" # Default reason
                    }
                }
                # Update with specific settings if provided
                if "function_args" in response and "image_generation" in response["function_args"]:
                    img_data["image_generation"].update(response["function_args"]["image_generation"])

                # Ensure generate_roleplay_image_from_gpt is async
                res = await generate_roleplay_image_from_gpt(img_data, user_id, conversation_id)
                if res and "image_urls" in res and res["image_urls"]:
                    image_url = res["image_urls"][0]
                    prompt_used = res.get('prompt_used', '')
                    reason = img_data["image_generation"].get("reason", "Narrative moment")
                    logger.info(f"Image generated successfully for conv_id={conversation_id}: {image_url}")
                    if socketio:
                        socketio.emit('image', {
                            'image_url': image_url,
                            'prompt_used': prompt_used,
                            'reason': reason
                        }, room=str(conversation_id))
                else:
                    logger.warning(f"Image generation task ran but produced no valid URLs for conv_id={conversation_id}. Response: {res}")
            except Exception as img_err:
                logger.error(f"Error generating image for conv_id={conversation_id}: {img_err}", exc_info=True)

        # Stream the text tokens
        if message_content and socketio:
            logger.debug(f"Streaming tokens for conv_id={conversation_id}")
            # Ensure streaming doesn't block the event loop for too long
            # Consider using socketio.sleep(0) occasionally if message_content is huge
            for i in range(0, len(message_content), 5): # Send slightly larger chunks
                token = message_content[i:i+5]
                socketio.emit('new_token', {'token': token}, room=str(conversation_id))
                await asyncio.sleep(0.01) # Use asyncio.sleep in async task

            socketio.emit('done', {'full_text': message_content}, room=str(conversation_id))
            logger.info(f"Finished streaming response for conv_id={conversation_id}")
        elif not message_content:
             logger.warning(f"No message content to stream for conv_id={conversation_id}")
             if socketio: # Emit done even if no content to signal completion
                  socketio.emit('done', {'full_text': ''}, room=str(conversation_id))

    except Exception as e:
        logger.error(f"Critical Error in background_chat_task for conv_id={conversation_id}: {str(e)}", exc_info=True)
        if socketio:
             socketio.emit('error', {'error': f"Server error processing message: {str(e)}"}, room=str(conversation_id))


async def initialize_systems(app):
    """Initialize systems required for the application AFTER app creation."""
    logger.info("Starting asynchronous system initializations...")
    try:
        # --- Database Schema/Seed ---
        # !! IMPORTANT !!: Schema creation/migration and seeding should ideally
        # be done via separate CLI commands (e.g., using Flask-Migrate, Alembic, or your init_db_script.py)
        # BEFORE starting the application server, not during runtime initialization.
        # Doing it here is risky and slows down startup.
        # Commenting out the direct calls:
        # from db.schema_and_seed import create_all_tables, seed_initial_data
        # from db.schema_migrations import ensure_schema_version
        # ensure_schema_version() # Run migrations separately!
        # logger.info("Database migrations check completed (ensure this was run beforehand!).")
        # create_all_tables() # Create tables separately!
        # seed_initial_data() # Seed data separately!
        # logger.info("Database tables initialization check completed (ensure this was done beforehand!).")
        logger.warning("Skipping DB schema/seed/migration checks in app startup. Run these manually/via deployment script.")

        # --- Initialize DB Pool ---
        # This *should* be done here or early in create_flask_app
        if not initialize_connection_pool(): # Ensure this initializes asyncpg pool
            logger.critical("Failed to initialize database connection pool! Exiting.")
            # Handle failure appropriately, maybe raise exception
            raise RuntimeError("Database pool initialization failed")
        else:
            logger.info("Database connection pool initialized successfully.")
        # Register pool cleanup
        atexit.register(close_connection_pool) # Ensure this closes asyncpg pool

        # --- Initialize Nyx Memory System ---
        # Pass DSN or ensure it's configured globally for initialize_nyx_memory_system
        await initialize_nyx_memory_system() # Ensure this uses asyncpg/DB_DSN if needed
        logger.info("Nyx memory system initialized successfully.")

        # --- Initialize OpenAI Integration ---
        # Make initialize_openai_integration async if it involves I/O
        async def initialize_openai_integration():
            """Initialize the OpenAI integration system."""
            try:
                from nyx.eternal.openai_integration import initialize
                from nyx.nyx_agent_sdk import process_user_input

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set. OpenAI integration might fail.")
                # Assuming 'initialize' is synchronous or handles its own async loop
                initialize(
                    api_key=api_key,
                    original_processor=process_user_input # Pass the function itself
                )
                logger.info("OpenAI integration system initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Error initializing OpenAI integration: {e}", exc_info=True)
                return False
        await initialize_openai_integration()

        # --- Initialize Global NyxBrain Instance ---
        # Ensure NyxBrain.get_instance is async and uses asyncpg if needed
        try:
            from nyx.core.brain.base import NyxBrain
            system_user_id = 0
            system_conversation_id = 0
            # Pass DSN or ensure NyxBrain uses the configured pool
            app.nyx_brain = await NyxBrain.get_instance(system_user_id, system_conversation_id)
            logger.info("Global NyxBrain instance initialized successfully")

            # Register response processors (ensure these handlers are async)
            from nyx.nyx_agent_sdk import process_user_input, process_user_input_with_openai
            # Assuming enhanced_background_chat_task is defined elsewhere and is async
            # from logic.nyx_enhancements_integration import enhanced_background_chat_task # Moved definition earlier

            app.nyx_brain.response_processors = {
                "default": enhanced_background_chat_task, # Needs to be async
                "openai": process_user_input_with_openai, # Needs to be async
                "base": process_user_input # Needs to be async
            }
            logger.info("Response processors registered with NyxBrain")
        except ImportError as e:
             logger.error(f"Could not import NyxBrain or its dependencies: {e}. Skipping NyxBrain init.")
        except Exception as e:
             logger.error(f"Error initializing NyxBrain: {e}", exc_info=True)


        # --- Initialize MCP Orchestrator ---
        # Ensure initialize is async and uses asyncpg/pool if needed
        try:
            app.mcp_orchestrator = MCPOrchestrator()
            await app.mcp_orchestrator.initialize()
            logger.info("MCP orchestrator initialized successfully.")
        except Exception as e:
             logger.error(f"Error initializing MCP Orchestrator: {e}", exc_info=True)


        # --- Register Conflict System ---
        # Ensure register_enhanced_integration is async and uses asyncpg/pool
        try:
            # Use dummy IDs or configure appropriately for system-wide registration if needed
            system_user_id = 1 # Or a dedicated system user ID
            system_conversation_id = 1 # Or a dedicated system conversation ID
            res = await register_enhanced_integration(system_user_id, system_conversation_id)
            if res.get("success"):
                logger.info("Conflict system successfully registered with Nyx governance")
            else:
                logger.error(f"Failed to register conflict system: {res.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error during conflict system registration: {str(e)}", exc_info=True)


        # --- Initialize NPC Learning ---
        # Ensure initialize_system is async and uses asyncpg/pool
        try:
            await NPCLearningManager.initialize_system()
            logger.info("NPC learning system initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing NPC Learning Manager: {e}", exc_info=True)

        # --- Initialize Universal Updater ---
        # Ensure initialize_universal_updater is async and uses asyncpg/pool
        try:
            from logic.universal_updater import initialize_universal_updater
            await initialize_universal_updater()
            logger.info("Universal updater initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Universal Updater: {e}", exc_info=True)

        # Configure admin access (Consider moving to config file)
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "1")
        try:
            app.config['ADMIN_USER_IDS'] = [int(uid.strip()) for uid in admin_ids_str.split(',')]
            logger.info(f"Admin User IDs configured: {app.config['ADMIN_USER_IDS']}")
        except ValueError:
            logger.error(f"Invalid ADMIN_USER_IDS format: '{admin_ids_str}'. Defaulting to [1]. Should be comma-separated integers.")
            app.config['ADMIN_USER_IDS'] = [1]

        logger.info("All asynchronous system initializations completed.")

    except Exception as e:
        logger.critical(f"Fatal error during system initialization: {str(e)}", exc_info=True)
        # Depending on the severity, might want to exit or prevent app from serving requests
        raise


###############################################################################
# FLASK APP CREATION
###############################################################################

def create_flask_app():
    """Create and configure a Flask application."""
    app = Flask(__name__, static_folder='static', template_folder='templates')

    # --- Basic Config ---
    try:
        # Use environment variables with defaults for security keys
        app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-insecure-secret-key-please-change')
        if app.config['SECRET_KEY'] == 'default-insecure-secret-key-please-change':
             logger.warning("FLASK_SECRET_KEY is not set or is using the insecure default!")

        # Secure defaults for production, allow override via env vars for dev
        app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
        app.config['SESSION_COOKIE_HTTPONLY'] = os.environ.get('SESSION_COOKIE_HTTPONLY', 'True').lower() == 'true'
        app.config['SESSION_COOKIE_SAMESITE'] = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax') # Lax is often a good default
        app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=int(os.environ.get('PERMANENT_SESSION_LIFETIME', 2592000))) # Default 30 days
        logger.info(f"Session cookie settings: Secure={app.config['SESSION_COOKIE_SECURE']}, HttpOnly={app.config['SESSION_COOKIE_HTTPONLY']}, SameSite={app.config['SESSION_COOKIE_SAMESITE']}")
    except Exception as config_err:
        logger.error(f"Error setting basic Flask config: {config_err}", exc_info=True)
        # Handle error - maybe raise or use safe defaults


    # --- Swagger ---
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs" # Recommend /docs or /api/docs
    }
    template = {
        "swagger": "2.0", # Consider OpenAPI 3+ using apispec directly or flask-smorest
        "info": {
            "title": "Roleplay Bot API", # Updated Title
            "description": "API for managing roleplay sessions, NPCs, memories, and interactions via Flask and SocketIO.",
            "version": "1.0.0",
            "contact": {"email": "support@example.com"} # Replace with actual contact
        },
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "JWT token in format: Bearer <token>" # If using JWT
            },
            # Add session cookie auth definition if applicable for Swagger UI
            "cookieAuth": {
                "type": "apiKey",
                "name": "session", # Or your actual cookie name
                "in": "cookie"
            }
        },
        "security": [{"cookieAuth": []}] # Default to cookie auth if that's primary
    }
    Swagger(app, config=swagger_config, template=template)

    # --- Security ---
    # Configure CSP more securely if possible (avoid 'unsafe-inline', 'unsafe-eval')
    # This often requires changes to frontend JS/CSS.
    csp = {
        'default-src': "'self'",
        'script-src': ["'self'"], # Remove 'unsafe-inline', 'unsafe-eval' if possible
        'style-src': ["'self'"], # Remove 'unsafe-inline' if possible
        'img-src': ["'self'", "data:"], # Allow data URIs if needed for images
        'connect-src': ["'self'", "ws://*", "wss://*"], # Allow self and websockets
    }
    Talisman(app, content_security_policy=csp)
    csrf = CSRFProtect(app)

    # --- Metrics ---
    metrics = PrometheusMetrics(app)
    metrics.info('app_info', 'Application info', version='1.0.0')

    # --- CORS ---
    # Be more specific with origins in production
    cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(',')
    CORS(app, resources={r"/*": {"origins": cors_origins}}, supports_credentials=True)
    logger.info(f"CORS configured for origins: {cors_origins}")


    # --- Register Blueprints ---
    # (Ensure blueprints using async routes correctly use asyncpg)
    app.register_blueprint(new_game_bp, url_prefix='/new_game')
    app.register_blueprint(player_input_bp, url_prefix='/player_input')
    app.register_blueprint(player_input_root_bp)
    app.register_blueprint(settings_bp, url_prefix='/settings')
    app.register_blueprint(knowledge_bp, url_prefix='/knowledge')
    app.register_blueprint(story_bp, url_prefix='/story')
    app.register_blueprint(memory_bp, url_prefix='/memory')
    app.register_blueprint(rule_enforcement_bp, url_prefix='/rules')
    app.register_blueprint(admin_bp, url_prefix='/admin') # For DB seeding etc.
    app.register_blueprint(debug_bp, url_prefix='/debug')
    app.register_blueprint(universal_bp, url_prefix='/universal')
    app.register_blueprint(multiuser_bp, url_prefix='/multiuser')
    app.register_blueprint(nyx_agent_bp, url_prefix='/nyx')
    app.register_blueprint(conflict_bp, url_prefix='/conflict')
    app.register_blueprint(npc_learning_bp, url_prefix='/npc-learning')

    init_image_routes(app) # Ensure this uses asyncpg if needed
    init_chat_routes(app) # Ensure this uses asyncpg if needed

    # --- Run Async Initializations ---
    # Run the async setup tasks AFTER the main app config but before returning app
    # This uses asyncio.run, which is okay here as it's during initial setup phase.
    try:
        asyncio.run(initialize_systems(app))
    except Exception as init_err:
        logger.critical(f"Asynchronous initialization failed: {init_err}", exc_info=True)
        # Decide whether to proceed or exit based on the error
        raise RuntimeError("Failed to initialize critical systems.") from init_err


    ###########################################################################
    # ROUTES (Defined in main app - keep minimal, prefer blueprints)
    ###########################################################################

    # --- Authentication Routes ---
    @app.route("/login_page", methods=["GET"])
    def login_page():
        return render_template("login.html") # Ensure login.html exists

    @app.route("/register_page", methods=["GET"])
    def register_page():
        return render_template("register.html") # Ensure register.html exists

    @app.route("/login", methods=["POST"])
    @rate_limit(limit=5, period=60) # Example rate limit
    @validate_request({ # Ensure middleware correctly populates request.sanitized_data
        'username': {'type': 'string', 'pattern': r'^[a-zA-Z0-9_.-]{3,30}$', 'required': True}, # Added pattern
        'password': {'type': 'string', 'max_length': 100, 'required': True}
    })
    async def login(): # Make async to use asyncpg
        # Use request.sanitized_data if middleware provides it, else request.json
        data = getattr(request, 'sanitized_data', request.get_json())
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
             return jsonify({"error": "Missing username or password"}), 400

        # Use asyncpg for database access
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    "SELECT id, password_hash FROM users WHERE username=$1",
                    username
                )

            if not row:
                # Mitigate timing attacks: Perform a fake hash check
                # Use a known salt structure if possible, otherwise generate a fake hash
                fake_hash = bcrypt.hashpw(b"dummyPassword", bcrypt.gensalt()).decode('utf-8')
                bcrypt.checkpw(password.encode('utf-8'), fake_hash.encode('utf-8'))
                logger.warning(f"Login attempt failed for non-existent user: {username}")
                return jsonify({"error": "Invalid username or password"}), 401

            user_id, hashed_password = row['id'], row['password_hash']

            # Check password using bcrypt
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                session["user_id"] = user_id
                session.permanent = True # Make session permanent based on lifetime config
                logger.info(f"User {username} (ID: {user_id}) logged in successfully")
                return jsonify({"message": "Logged in", "user_id": user_id})
            else:
                logger.warning(f"Login attempt failed for user: {username} (ID: {user_id}) - Invalid password")
                return jsonify({"error": "Invalid username or password"}), 401

        except asyncpg.PostgresError as db_err:
            logger.error(f"Database error during login for user {username}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error during login"}), 500
        except Exception as e:
            logger.error(f"Unexpected error during login for user {username}: {e}", exc_info=True)
            return jsonify({"error": "Server error during login"}), 500

    @app.route("/register", methods=["POST"])
    @rate_limit(limit=3, period=300) # Stricter rate limit for registration
    @validate_request({
        'username': {'type': 'string', 'pattern': r'^[a-zA-Z0-9_.-]{3,30}$', 'required': True},
        'password': {'type': 'string', 'min_length': 8, 'max_length': 100, 'required': True},
        'email':    {'type': 'string', 'pattern': r'[^@]+@[^@]+\.[^@]+', 'max_length': 100, 'required': False} # Email optional? Add validation.
    })
    async def register(): # Make async for asyncpg
        data = getattr(request, 'sanitized_data', request.get_json())
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
        username = data.get("username")
        password = data.get("password")
        email = data.get("email") # Optional based on validation schema

        if not username or not password: # Add email check if required
             return jsonify({"error": "Missing required fields"}), 400

        # Hash password
        try:
             password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except Exception as hash_err:
             logger.error(f"Error hashing password during registration: {hash_err}", exc_info=True)
             return jsonify({"error": "Server error during registration setup"}), 500

        # Insert user using asyncpg within a transaction
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    # Check for existing username
                    existing_user = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
                    if existing_user:
                        logger.warning(f"Registration attempt failed for existing username: {username}")
                        return jsonify({"error": "Username already exists"}), 409

                    # Check for existing email if provided and required to be unique
                    if email:
                        existing_email = await conn.fetchval("SELECT id FROM users WHERE email=$1", email)
                        if existing_email:
                            logger.warning(f"Registration attempt failed for existing email: {email}")
                            return jsonify({"error": "Email already exists"}), 409

                    # Insert new user
                    # Add email to query if it's a column
                    query = """
                        INSERT INTO users (username, password_hash, email, created_at)
                        VALUES ($1, $2, $3, NOW()) RETURNING id
                    """
                    user_id = await conn.fetchval(query, username, password_hash, email)

            if user_id:
                logger.info(f"New user registered: {username} (ID: {user_id})")
                session["user_id"] = user_id
                session.permanent = True
                return jsonify({"message": "User registered successfully", "user_id": user_id}), 201 # Use 201 Created status
            else:
                # This case should ideally not happen if RETURNING id works and transaction succeeds
                logger.error(f"Registration seemed successful but no user ID returned for username: {username}")
                return jsonify({"error": "Server error during registration confirmation"}), 500

        except asyncpg.PostgresError as db_err:
            logger.error(f"Database error during registration for {username}: {db_err}", exc_info=True)
            # Check for unique constraint violation specifically if needed
            if isinstance(db_err, asyncpg.exceptions.UniqueViolationError):
                 # This might be redundant if checks above work, but good as a fallback
                 # Determine which constraint failed if possible (check db_err details)
                 return jsonify({"error": "Username or email already exists."}), 409
            return jsonify({"error": "Database error during registration"}), 500
        except Exception as e:
            logger.error(f"Unexpected error during registration for {username}: {e}", exc_info=True)
            return jsonify({"error": "Server error during registration"}), 500


    @app.route("/logout", methods=["POST"])
    def logout():
        user_id = session.get("user_id")
        session.clear()
        if user_id:
            logger.info(f"User ID {user_id} logged out.")
        else:
            logger.info("Logout request received for non-logged-in session.")
        # Could add CSRF protection here if needed for logout POST
        return jsonify({"message": "Logged out"}), 200

    @app.route("/whoami", methods=["GET"])
    def whoami():
        user_id = session.get("user_id")
        if user_id:
            # Optionally fetch username or other non-sensitive data
            return jsonify({"logged_in": True, "user_id": user_id}), 200
        return jsonify({"logged_in": False}), 200


    # --- Core Game Routes (Example - move complex logic to blueprints) ---

    @app.route("/chat") # Serves the chat page UI
    def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        # Pass necessary data to template (e.g., user_id, active conversation_id)
        user_id = session.get("user_id")
        # Fetch active conversation or list of conversations here if needed (using asyncpg if route becomes async)
        return render_template("chat.html", user_id=user_id) # Ensure chat.html exists

    # Note: /start_chat and /openai_chat POST routes were removed as the primary interaction
    # now seems to happen via SocketIO ('message' event). If you need these HTTP endpoints,
    # ensure they use asyncpg and potentially start Celery tasks instead of socketio background tasks.

    @app.route("/start_new_game", methods=["POST"])
    # @validate_request({'user_id': {'type': 'integer','required': True}}) # Validation might be redundant if using session
    async def start_new_game(): # Make async
        """Starts a new game asynchronously, potentially using Celery for heavy lifting."""
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not authenticated"}), 401

        logger.info(f"User {user_id} requested a new game.")

        try:
            # 1. Create conversation record synchronously first (or async if preferred)
            # Using asyncpg here for consistency
            async with get_db_connection_context() as conn:
                 # Use a transaction
                 async with conn.transaction():
                    # Insert conversation with 'processing' status
                    conv_row = await conn.fetchrow("""
                        INSERT INTO conversations (user_id, conversation_name, status)
                        VALUES ($1, $2, 'processing') RETURNING id
                    """, user_id, "New Game - Initializing...")
                    conversation_id = conv_row['id']
                    logger.info(f"Created new conversation record {conversation_id} for user {user_id}.")

                    # Optionally, insert default player stats immediately
                    # Assuming insert_default_player_stats_chase uses asyncpg or can be awaited
                    await insert_default_player_stats_chase(user_id, conversation_id, conn) # Pass connection


            # 2. Trigger the heavy lifting asynchronously via Celery
            # Pass necessary data to the task
            task_result = process_new_game_task.delay(user_id, {"conversation_id": conversation_id})
            logger.info(f"Dispatched new game processing task {task_result.id} for conv {conversation_id}.")

            # 3. Return immediately to the user, indicating processing has started
            return jsonify({
                "status": "processing",
                "message": "New game creation started. Please wait for initialization.",
                "conversation_id": conversation_id,
                "task_id": task_result.id # Client can potentially poll this task ID
            }), 202 # 202 Accepted status code

        except asyncpg.PostgresError as db_err:
            logger.error(f"Database error starting new game for user {user_id}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error starting game"}), 500
        except Exception as e:
            logger.error(f"Error dispatching new game task for user {user_id}: {e}", exc_info=True)
            return jsonify({"error": "Server error starting game process"}), 500


    # --- Admin/Debug Routes ---
    @app.route("/admin/nyx_direct", methods=["POST"])
    async def admin_nyx_direct(): # Make async
        """Direct access to NyxBrain for admin users only"""
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401

        # Check if user is admin
        admin_ids = app.config.get('ADMIN_USER_IDS', [])
        if session.get("user_id") not in admin_ids:
            logger.warning(f"Non-admin user {session.get('user_id')} attempted Nyx direct access.")
            return jsonify({"error": "Access denied"}), 403

        data = request.get_json()
        if not data or "user_input" not in data:
             return jsonify({"error": "Missing user_input in request"}), 400
        user_input = data.get("user_input")

        # Get NyxBrain instance (should already be initialized)
        nyx_brain = getattr(app, 'nyx_brain', None)
        if not nyx_brain:
             logger.error("NyxBrain not initialized on app context.")
             return jsonify({"error": "Nyx system not available"}), 503

        # Process directly with NyxBrain instead of agent SDK
        # No need for asyncio.run here as the route handler is async
        try:
            logger.info(f"Admin user {session.get('user_id')} executing direct Nyx command: {user_input[:50]}...")
            # Assuming process_input_with_thinking and generate_response_with_thinking are async
            processing_result = await nyx_brain.process_input_with_thinking(
                user_input=user_input,
                context={"admin_mode": True}
            )
            response_result = await nyx_brain.generate_response_with_thinking(
                user_input=user_input,
                context={"admin_mode": True}
            )

            result = {
                "brain_processing": processing_result, # Ensure these are serializable
                "brain_response": response_result, # Ensure these are serializable
                "admin_mode": True
            }
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error during admin Nyx direct call: {e}", exc_info=True)
            return jsonify({"error": "Error processing direct Nyx command", "details": str(e)}), 500


    @app.route("/nyx_response", methods=["POST"])
    async def get_nyx_response(): # Make async
        """Regular user access via nyx_agent_sdk"""
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not authenticated"}), 401

        data = request.get_json()
        if not data or "user_input" not in data or "conversation_id" not in data:
             return jsonify({"error": "Missing user_input or conversation_id"}), 400

        user_input = data.get("user_input")
        conversation_id = data.get("conversation_id")
        # TODO: Add validation to ensure user owns this conversation_id

        # Use the distilled agent SDK
        from nyx.nyx_agent_sdk import process_user_input # Ensure this is async

        try:
            logger.info(f"Processing Nyx SDK request for user={user_id}, conv={conversation_id}")
            # Call the async SDK function directly
            result = await process_user_input(
                user_id,
                conversation_id,
                user_input,
                {} # Add relevant context if needed
            )
            return jsonify(result) # Ensure result is JSON serializable
        except Exception as e:
            logger.error(f"Error processing Nyx SDK request for user={user_id}, conv={conversation_id}: {e}", exc_info=True)
            return jsonify({"error": "Error processing request via Nyx SDK", "details": str(e)}), 500


    ###########################################################################
    # HEALTH ENDPOINTS
    ###########################################################################

    @app.route("/health", methods=["GET"])
    def health_check():
        """Basic health check endpoint."""
        return jsonify({"status": "healthy", "timestamp": time.time()})

    @app.route("/readiness", methods=["GET"])
    async def readiness_check(): # Make async for async DB check
        """Readiness check endpoint for checking dependencies."""
        status = {"status": "ready", "timestamp": time.time(), "checks": {}}
        is_ready = True

        # --- DB check (Async) ---
        try:
            async with get_db_connection_context(timeout=5) as conn: # Add timeout
                # Perform a simple query
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    status["checks"]["database"] = "connected"
                else:
                    status["checks"]["database"] = "error: unexpected query result"
                    is_ready = False
        except asyncio.TimeoutError:
             status["checks"]["database"] = "error: connection timeout"
             is_ready = False
        except asyncpg.PostgresError as db_err:
            status["checks"]["database"] = f"error: {type(db_err).__name__}"
            is_ready = False
            logger.warning(f"Readiness check DB error: {db_err}")
        except Exception as e:
            status["checks"]["database"] = f"error: {str(e)}"
            is_ready = False
            logger.warning(f"Readiness check unexpected DB error: {e}", exc_info=True)


        # --- Redis Check (Sync - consider async if heavily used) ---
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        try:
            # Use short timeout for readiness check
            r = Redis(host=redis_host, port=redis_port, socket_timeout=2, socket_connect_timeout=2)
            r.ping()
            status["checks"]["redis"] = "connected"
        except Exception as e:
            status["checks"]["redis"] = f"error: {str(e)}"
            is_ready = False
            logger.warning(f"Readiness check Redis error: {e}")

        # --- Celery Check (using inspect) ---
        # This can be slow and unreliable; consider a dedicated health check task
        try:
            # Assuming celery_app is accessible, maybe attach to app context?
            # Or import directly from celery_config (might cause circular import issues)
            from celery_config import celery_app as current_celery_app
            # Add a timeout to inspect to prevent hanging
            inspector = current_celery_app.control.inspect(timeout=5)
            active_workers = inspector.ping() # Ping is usually faster than active()

            if active_workers:
                status["checks"]["celery"] = f"ping ok ({len(active_workers)} responses)"
            else:
                # Try checking stats as backup
                stats = inspector.stats()
                if stats:
                     status["checks"]["celery"] = f"stats ok ({len(stats)} workers)"
                else:
                     status["checks"]["celery"] = "error: no active workers responded to ping/stats"
                     is_ready = False
                     logger.warning("Readiness check Celery error: No workers responded.")
        except Exception as e:
            status["checks"]["celery"] = f"error: {str(e)}"
            is_ready = False
            logger.warning(f"Readiness check Celery error: {e}", exc_info=True)


        # --- Final Status ---
        if not is_ready:
            status["status"] = "not ready"
            return jsonify(status), 503 # Service Unavailable

        return jsonify(status), 200


    # --- Return the configured app ---
    return app


###############################################################################
# SOCKET.IO SETUP
###############################################################################

def create_socketio(app):
    global socketio # Assign to global variable

    # Configure SocketIO
    # Use Redis or RabbitMQ for message queue in production with multiple workers
    message_queue_url = os.getenv("SOCKETIO_MESSAGE_QUEUE", None) # e.g., redis://localhost:6379/1
    if not message_queue_url:
         logger.warning("SOCKETIO_MESSAGE_QUEUE not set. SocketIO will only work correctly with a single web worker.")

    socketio = SocketIO(
        app,
        cors_allowed_origins="*", # Restrict in production
        async_mode='eventlet',
        logger=True, # Enable SocketIO logging
        engineio_logger=True, # Enable EngineIO logging (can be verbose)
        ping_timeout=20, # Lower ping timeout
        ping_interval=10, # Lower ping interval
        message_queue=message_queue_url # Enable message queue for multi-worker setups
    )

    # --- SocketIO Event Handlers ---

    @socketio.on('connect')
    def handle_connect():
        user_id = session.get("user_id", "anonymous") # Get user_id from session if available
        logger.info(f"SocketIO: Client connected - SID: {request.sid}, User: {user_id}")
        emit('response', {'data': 'Connected successfully!'}) # Send confirmation to client

    @socketio.on('disconnect')
    def handle_disconnect():
        user_id = session.get("user_id", "anonymous")
        logger.info(f"SocketIO: Client disconnected - SID: {request.sid}, User: {user_id}")
        # Clean up user-specific resources or rooms if necessary

    @socketio.on('join')
    def handle_join(data):
        if not isinstance(data, dict):
            logger.warning(f"Invalid 'join' data received from SID {request.sid}: {data}")
            emit('error', {'error': 'Invalid join data format'})
            return

        conversation_id = data.get('conversation_id')
        user_id = session.get('user_id') # Get user from secure session

        if not user_id:
            logger.warning(f"Unauthorized attempt to join room from SID {request.sid}")
            emit('error', {'error': 'Authentication required to join room'})
            return

        if conversation_id:
            try:
                # Convert to string for room name consistency
                room_name = str(conversation_id)

                # TODO: Add authorization check: Does this user_id have access to this conversation_id?
                # Need an async DB call here
                # async def check_auth():
                #    async with get_db_connection_context() as conn:
                #        auth = await conn.fetchval("SELECT 1 FROM conversations WHERE id=$1 AND user_id=$2", conversation_id, user_id)
                #        return bool(auth)
                # is_authorized = asyncio.run(check_auth()) # Problematic with eventlet, better to do in async handler if possible
                # if not is_authorized:
                #     logger.warning(f"User {user_id} forbidden to join room {room_name} (SID: {request.sid})")
                #     emit('error', {'error': 'Forbidden'})
                #     return

                join_room(room_name)
                logger.info(f"SocketIO: Client SID {request.sid} (User: {user_id}) joined room {room_name}")

                # Send confirmation back to the specific client
                emit('joined', {'room': room_name}, room=request.sid)

                # Optionally, broadcast to room that user joined (if desired)
                # emit('user_joined', {'user_id': user_id}, room=room_name, include_self=False)

                # Initialize conversation state in background if needed (using async task)
                # socketio.start_background_task(initialize_conversation_state_async, conversation_id, user_id)

            except Exception as e:
                 logger.error(f"Error joining room {conversation_id} for SID {request.sid}: {e}", exc_info=True)
                 emit('error', {'error': 'Server error joining room'})
        else:
            logger.warning(f"Client SID {request.sid} (User: {user_id}) attempted to join room without conversation_id")
            emit('error', {'error': 'Missing conversation_id'})

    @socketio.on('message')
    async def handle_message(data): # Make handler async
        """Handles incoming chat messages from the client."""
        user_id = session.get('user_id')
        if not user_id:
            logger.warning(f"Unauthorized message received from SID {request.sid}")
            emit('error', {'error': 'Authentication required'}, room=request.sid)
            return

        if not isinstance(data, dict):
            logger.warning(f"Invalid 'message' data received from SID {request.sid} (User: {user_id}): {data}")
            emit('error', {'error': 'Invalid message data format'}, room=request.sid)
            return

        conversation_id = data.get('conversation_id')
        message_text = data.get('message')
        universal_update = data.get('universal_update') # Optional

        if not conversation_id or not message_text:
            logger.warning(f"Incomplete message data from SID {request.sid} (User: {user_id}): conv={conversation_id}, msg='{message_text}'")
            emit('error', {'error': 'Invalid message data (missing conversation_id or message)'}, room=request.sid)
            return

        # Convert conv_id for consistency
        conversation_id = int(conversation_id)
        room_name = str(conversation_id)

        # TODO: Authorization check: Does this user own this conversation? (Similar to 'join')

        # 1. Store user message asynchronously
        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """INSERT INTO messages (conversation_id, sender, content, created_at)
                       VALUES ($1, $2, $3, NOW())""",
                    conversation_id, f"user:{user_id}", message_text # Include user ID in sender field
                )
            logger.info(f"Stored user message from User {user_id} in Conv {conversation_id}")
        except Exception as db_err:
            logger.error(f"DB Error storing user message for User {user_id}, Conv {conversation_id}: {db_err}", exc_info=True)
            emit('error', {'error': 'Failed to save your message'}, room=request.sid)
            return # Stop processing if saving failed

        # 2. Start the async background task for processing the message
        # Pass socketio instance or make it globally accessible if needed inside the task
        logger.debug(f"Starting background chat task for User {user_id}, Conv {conversation_id}")
        socketio.start_background_task(
            background_chat_task, # Ensure this task is async and uses asyncpg
            conversation_id=conversation_id,
            user_input=message_text,
            user_id=user_id,
            universal_update=universal_update
        )
        # Optionally acknowledge receipt to the user immediately
        emit('message_received', {'status': 'processing'}, room=request.sid)

    # Remove handle_storybeat - Refactor this into a standard 'message' handler
    # or create a new event type like 'action' or 'command' if it's distinct.
    # The logic inside handle_storybeat was very complex and mixed sync/async DB calls.
    # Refactor its core logic into async helper functions or agents called from
    # the main 'message' handler or a dedicated event handler, ensuring all DB
    # access uses asyncpg.

    return socketio

###############################################################################
# MAIN ENTRY (Use wsgi.py for deployment)
###############################################################################

# This block is primarily for local development using `python main.py`
if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch() # Monkey patch for eventlet worker

    logging.basicConfig(
        level=logging.INFO, # Use DEBUG for more verbose local dev logging
        format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s' # Added threadName
    )

    # Load environment variables from .env file for local development
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file (if found).")


    port = int(os.getenv("PORT", 8080)) # Use 8080 as default dev port

    # Create app and socketio instances
    app = create_flask_app()
    # create_socketio needs the app instance
    socketio_instance = create_socketio(app) # This assigns to the global 'socketio'

    logger.info(f"Starting development server on http://0.0.0.0:{port}")
    # Use socketio.run() for development, which handles eventlet/gevent setup
    socketio_instance.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true", # Enable Flask debug mode via env var
        use_reloader=os.getenv("FLASK_USE_RELOADER", "True").lower() == "true" # Enable reloader for dev
    )
