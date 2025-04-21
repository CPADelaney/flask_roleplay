# main.py

import os
import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional

# quart and related imports
from quart import Quart, render_template, session, request, jsonify, redirect, Response
import socketio

from aioprometheus import Registry, Counter, render
from aioprometheus.asgi.starlette import metrics as metrics_middleware

from quart_schema import QuartSchema
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
from routes.nyx_agent_routes_sdk import nyx_agent_bp
from routes.conflict_routes import conflict_bp
from routes.npc_learning_routes import npc_learning_bp
from routes.auth import register_auth_routes, auth_bp

from nyx.core.brain.base import NyxBrain

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
    initialize_connection_pool, # Async function
    close_connection_pool, # Async function
    get_db_connection_context # Async context manager
)

from nyx.core.sync.nyx_sync_daemon import NyxSyncDaemon

# Middleware
from middleware.rate_limiting import rate_limit, ip_block_list, ip_block_middleware
from middleware.validation import validate_request

from logic.aggregator_sdk import init_singletons

logger = logging.getLogger(__name__)

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
    from quart import current_app

    logger.info(f"[BG Task {conversation_id}] Starting for user {user_id}")
    try:
        # Get aggregator context (ensure this function is async or thread-safe if it hits DB)
        # If sync and hits DB, consider running in an executor or making it async
        from logic.aggregator import get_aggregated_roleplay_context
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
            logger.info(f"[BG Task {conversation_id}] Applying universal updates...")
            try:
                from logic.universal_updater_agent import apply_universal_updates_async # Needs to be async
                async with get_db_connection_context() as conn: # Use async context manager
                    await apply_universal_updates_async(
                        user_id,
                        conversation_id,
                        universal_update,
                        conn # Pass the connection
                    )
                logger.info(f"[BG Task {conversation_id}] Applied universal updates.")
                # Refresh aggregator data post-update
                aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
                context["aggregator_data"] = aggregator_data
            except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as update_db_err:
                 logger.error(f"[BG Task {conversation_id}] DB Error applying universal updates: {update_db_err}", exc_info=True)
                 # Decide if to continue or emit error and stop
                 await current_app.socketio.emit('error', {'error': 'Failed to apply world updates.'}, room=str(conversation_id))
                 return
            except Exception as update_err:
                 logger.error(f"[BG Task {conversation_id}] Error applying universal updates: {update_err}", exc_info=True)
                 await current_app.socketio.emit('error', {'error': 'Failed to apply world updates.'}, room=str(conversation_id))
                 return

        # Process the user_input with OpenAI-enhanced Nyx agent
        # Ensure this function is async
        from nyx.nyx_agent_sdk import process_user_input_with_openai
        logger.info(f"[BG Task {conversation_id}] Processing input with Nyx agent...")
        response = await process_user_input_with_openai(user_id, conversation_id, user_input, context)
        logger.info(f"[BG Task {conversation_id}] Nyx agent processing complete.")

        if not response or not response.get("success", False):
            error_msg = response.get("error", "Unknown error from Nyx agent") if response else "Empty response from Nyx agent"
            logger.error(f"[BG Task {conversation_id}] Nyx agent failed: {error_msg}")
            await current_app.socketio.emit('error', {'error': error_msg}, room=str(conversation_id))
            return

        # Extract the message content
        message_content = response.get("message", "")
        if not message_content and "function_args" in response:
            message_content = response["function_args"].get("narrative", "")
        logger.debug(f"[BG Task {conversation_id}] Nyx response: {message_content[:100]}...")

        # Store the Nyx response in DB using asyncpg
        try:
            async with get_db_connection_context() as conn: # Use async context manager
                await conn.execute( # Use await and $n params
                    """INSERT INTO messages (conversation_id, sender, content, created_at)
                       VALUES ($1, $2, $3, NOW())""",
                    conversation_id, "Nyx", message_content
                )
            logger.info(f"[BG Task {conversation_id}] Stored Nyx response to DB.")
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"[BG Task {conversation_id}] DB Error storing Nyx response: {db_err}", exc_info=True)
            # Continue but log the error

        # Check if we should generate an image
        should_generate = response.get("generate_image", False)
        if "function_args" in response and "image_generation" in response["function_args"]:
            img_settings = response["function_args"]["image_generation"]
            should_generate = should_generate or img_settings.get("generate", False)

        # Generate image if needed (ensure generate_roleplay_image_from_gpt is async)
        if should_generate:
            logger.info(f"[BG Task {conversation_id}] Image generation triggered.")
            try:
                img_data = { # Prepare data structure
                    "narrative": message_content,
                    "image_generation": response.get("function_args", {}).get("image_generation", {
                        "generate": True, "priority": "medium", "focus": "balanced",
                        "framing": "medium_shot", "reason": "Narrative moment"
                    })
                }
                # Ensure generate_roleplay_image_from_gpt is async
                res = await generate_roleplay_image_from_gpt(img_data, user_id, conversation_id)

                if res and "image_urls" in res and res["image_urls"]:
                    image_url = res["image_urls"][0]
                    prompt_used = res.get('prompt_used', '')
                    reason = img_data["image_generation"].get("reason", "Narrative moment")
                    logger.info(f"[BG Task {conversation_id}] Image generated: {image_url}")
                    await current_app.socketio.emit('image', {
                        'image_url': image_url, 'prompt_used': prompt_used, 'reason': reason
                    }, room=str(conversation_id))
                else:
                    logger.warning(f"[BG Task {conversation_id}] Image generation task ran but produced no valid URLs. Response: {res}")
            except Exception as img_err:
                logger.error(f"[BG Task {conversation_id}] Error generating image: {img_err}", exc_info=True)

        # Stream the text tokens
        if message_content:
            logger.debug(f"[BG Task {conversation_id}] Streaming tokens...")
            chunk_size = 5
            delay = 0.01 # Small delay between chunks
            for i in range(0, len(message_content), chunk_size):
                token = message_content[i:i+chunk_size]
                await current_app.socketio.emit('new_token', {'token': token}, room=str(conversation_id))
                await asyncio.sleep(delay) # Use asyncio.sleep in async task

            await current_app.socketio.emit('done', {'full_text': message_content}, room=str(conversation_id))
            logger.info(f"[BG Task {conversation_id}] Finished streaming response.")
        else:
             logger.warning(f"[BG Task {conversation_id}] No message content to stream.")
             await current_app.socketio.emit('done', {'full_text': ''}, room=str(conversation_id)) # Still signal done

    except Exception as e:
        logger.error(f"[BG Task {conversation_id}] Critical Error: {str(e)}", exc_info=True)
        await current_app.socketio.emit('error', {'error': f"Server error processing message: {str(e)}"}, room=str(conversation_id))

async def initialize_systems(app):
    """Initialize systems required for the application AFTER app creation."""
    logger.info("Starting asynchronous system initializations...")
    from nyx.core.brain.base import NyxBrain
    from nyx.core.brain.checkpointing_agent import llm_periodic_checkpoint    
    try:
        # --- Database Schema/Seed ---
        # !! IMPORTANT !!: Schema creation/migration and seeding should ideally
        # be done via separate CLI commands (e.g., using quart-Migrate, Alembic, or your init_db_script.py)
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
        # This *should* be done here or early in create_quart_app
        if not await initialize_connection_pool():
            raise RuntimeError("Database pool initialization failed")
        else:
            logger.info("Database connection pool initialized successfully.")

        # Register pool cleanup (atexit might be unreliable for async)
        # Consider a more robust shutdown handler in production
        async def cleanup_pool_on_exit():
            logger.info("Running async pool cleanup...")
            await close_connection_pool()

        def start_background_services():
            loop = asyncio.get_event_loop()
            daemon = NyxSyncDaemon()
            loop.create_task(daemon.start())

        # Wrap the async cleanup for atexit (use with caution)
        def run_async_cleanup():
            try:
                asyncio.run(cleanup_pool_on_exit())
            except RuntimeError as e:
                # Handle cases where event loop might already be closed
                logger.warning(f"Could not run async cleanup in atexit: {e}")

        atexit.register(run_async_cleanup)
        logger.info("Registered async pool cleanup with atexit (best effort).")

        # --- Initialize Nyx Memory System ---
        # Pass DSN or ensure it's configured globally for initialize_nyx_memory_system
        await initialize_nyx_memory_system() # Ensure this uses asyncpg/DB_DSN if needed
        logger.info("Nyx memory system initialized successfully.")

        # --- Initialize Global NyxBrain Instance ---
        # Ensure NyxBrain.get_instance is async and uses asyncpg if needed
        print(">>> NyxBrain module:", NyxBrain.__module__)
        print(">>> NyxBrain class file:", NyxBrain.__dict__.get('__module__', None))
        print(">>> NyxBrain has get_instance?", hasattr(NyxBrain, "get_instance"))
        print(">>> NyxBrain dir:", dir(NyxBrain))
        print(">>> NyxBrain MRO:", NyxBrain.__mro__)

        try:
            system_user_id = 0
            system_conversation_id = 0
            
            # Check if get_instance method exists before trying to use it
            if hasattr(NyxBrain, "get_instance"):
                app.nyx_brain = await NyxBrain.get_instance(system_user_id, system_conversation_id)
                await app.nyx_brain.initialize()
                logger.info("Global NyxBrain instance initialized.")
            
                # RESTORE ONLY FROM THE DISTRIBUTED CHECKPOINT LOGIC:
                await app.nyx_brain.restore_entity_from_distributed_checkpoints()
                    
                asyncio.create_task(llm_periodic_checkpoint(app.nyx_brain))          
                
                # Register processors (ensure handlers are async)
                from nyx.nyx_agent_sdk import process_user_input, process_user_input_with_openai
                
                app.nyx_brain.response_processors = {
                    "default": background_chat_task,
                    "openai": process_user_input_with_openai,
                    "base": process_user_input
                }
                logger.info("Response processors registered with NyxBrain.")
            else:
                logger.warning("NyxBrain.get_instance method not available. Skipping NyxBrain initialization.")
                app.nyx_brain = None
        except ImportError as e:
             logger.error(f"Could not import NyxBrain: {e}. Skipping init.")
             app.nyx_brain = None
        except Exception as e:
             logger.error(f"Error initializing NyxBrain: {e}", exc_info=True)
             app.nyx_brain = None

        # MCP orchestrator (assuming async)
        try:
            app.mcp_orchestrator = MCPOrchestrator()
            await app.mcp_orchestrator.initialize() # Assuming async
            logger.info("MCP orchestrator initialized.")
        except Exception as e:
             logger.error(f"Error initializing MCP Orchestrator: {e}", exc_info=True)

        # Conflict system registration (assuming async)
        try:
            system_user_id = 1
            system_conversation_id = 1
            res = await register_enhanced_integration(system_user_id, system_conversation_id) # Assuming async
            if res.get("success"): logger.info("Conflict system registered.")
            else: logger.error(f"Failed to register conflict system: {res.get('message')}")
        except Exception as e: logger.error(f"Error registering conflict system: {e}", exc_info=True)

        # NPC learning (assuming async)
        try:
            learning_manager = NPCLearningManager(system_user_id, system_conversation_id)
            await learning_manager.initialize()
            logger.info("NPC learning system initialized.")
        except Exception as e: logger.error(f"Error initializing NPC Learning: {e}", exc_info=True)

        # Universal updater (assuming async)
        try:
            from logic.universal_updater_agent import initialize_universal_updater
            # Use the same system IDs as used for other initializations
            await initialize_universal_updater(system_user_id, system_conversation_id)
            logger.info("Universal updater initialized.")
        except Exception as e: logger.error(f"Error initializing Universal Updater: {e}", exc_info=True)

        # Admin config
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "1")
        try:
            app.config['ADMIN_USER_IDS'] = [int(uid.strip()) for uid in admin_ids_str.split(',')]
        except ValueError:
            logger.error(f"Invalid ADMIN_USER_IDS format: '{admin_ids_str}'. Defaulting to [1].")
            app.config['ADMIN_USER_IDS'] = [1]
        logger.info(f"Admin User IDs configured: {app.config['ADMIN_USER_IDS']}")

        logger.info("All asynchronous system initializations completed.")

        await init_singletons()  # Initialize aggregator_sdk singletons here
        logger.info("Aggregator SDK singletons are ready.")

        from nyx.core.brain import base as nyx_base
        print('Dir on NyxBrain:', dir(nyx_base.NyxBrain))
        print('NyxBrain.__dict__:', nyx_base.NyxBrain.__dict__.keys())
        print('Any get_instance global?', hasattr(nyx_base, "get_instance"))
        print('get_instance:', getattr(nyx_base, "get_instance", None))
    

    except Exception as e:
        logger.critical(f"Fatal error during system initialization: {str(e)}", exc_info=True)
        raise # Prevent app from starting if critical systems fail

###############################################################################
# quart APP CREATION
###############################################################################

def create_quart_app():
    app = Quart(__name__, static_folder="static", template_folder="templates")
    QuartSchema(app)
    
    # 2) Create & attach Socket.IO _before_ any @sio.event handlers
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    app.asgi_app = socketio.ASGIApp(sio, app.asgi_app)
    app.socketio = sio

    # 3) Metrics (aioprometheus)
    registry = Registry()
    http_requests = Counter(
        "http_requests_total", 
        "Total HTTP requests",
        const_labels={"service": "my‑quart‑app"}
    )
    registry.register(http_requests)

    @app.route("/metrics")
    async def metrics_endpoint():
        from aioprometheus import render
        return Response(await render(registry), mimetype="text/plain; version=0.0.4")

    # 4) CORS
    # make sure you have `pip install quart-cors`
    from quart_cors import cors
    
    def parse_cors_origins(origins_str):
        """Parse comma-separated or JSON-like origins string into a list."""
        if not origins_str or origins_str == "*":
            return "*"
        
        # Remove any quotes and spaces from the beginning and end
        cleaned = origins_str.strip('" ')
        
        # Try to handle the format "url1","url2","url3"
        if '","' in cleaned:
            return [url.strip('" ') for url in cleaned.split('","')]
        
        # Handle normal comma-separated format
        return [url.strip() for url in cleaned.split(',')]
    
    origins = parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS", ""))
    
    # If origins is still empty after parsing, use a default
    if not origins or origins == [""]:
        # For development, you might want to use localhost
        origins = ["http://localhost:3000", "https://nyx-m85p.onrender.com"]
        # Log a warning
        logger.warning(f"No valid CORS origins found in environment. Using defaults: {origins}")
    
    # Configure CORS - if using specific origins, don't use wildcard
    if origins == "*":
        # When using wildcard, cannot use credentials
        cors(app,
             allow_origin="*",
             allow_credentials=False,  # Must be False with wildcard
             allow_methods="*",
             allow_headers="*")
        logger.info("CORS configured with wildcard origin (credentials disabled)")
    else:
        # When using specific origins, can use credentials
        cors(app,
             allow_origin=origins,
             allow_credentials=True,
             allow_methods="*",
             allow_headers="*")
        logger.info(f"CORS configured with specific origins: {origins}")


    # 5) Socket.IO event handlers
    @sio.event
    async def connect(sid, environ):
        user = session.get("user_id", "anonymous")
        app.logger.info(f"connect: {sid}/{user}")
        await sio.emit("response", {"data": "Connected!"}, to=sid)

    @sio.on("join")
    async def on_join(sid, data):
        room = str(data.get("conversation_id"))
        sio.enter_room(sid, room)
        await sio.emit("joined", {"room": room}, to=sid)

    @sio.on("message")
    async def on_message(sid, data):
        await sio.emit("message_received", {"status": "processing"}, to=sid)
        # … your background task kick‑off here …

    # 6) Security headers
    @app.after_request
    async def set_security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self' ws://* wss://*"
        )
        return response

    # (Removed stray PrometheusMetrics import & metrics.info — we’re using aioprometheus now)

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
    app.before_request(ip_block_middleware)
    register_auth_routes(app)

    init_image_routes(app) # Ensure this uses asyncpg if needed
    init_chat_routes(app) # Ensure this uses asyncpg if needed

    # --- Run Async Initializations ---
    # Run the async setup tasks AFTER the main app config but before returning app
    # This uses asyncio.run, which is okay here as it's during initial setup phase.
    try:
        # Use asyncio.run() to execute the async initializer function
        # This is acceptable here during the synchronous app creation phase.
        asyncio.run(initialize_systems(app))
    except Exception as init_err:
        logger.critical(f"Application initialization failed: {init_err}", exc_info=True)
        # Exit or raise prevents the app from being returned in a broken state
        raise RuntimeError("Failed to initialize application systems.") from init_err
    logger.info("Async initializations complete. quart app creation finished.")


    ###########################################################################
    # ROUTES (Defined in main app - keep minimal, prefer blueprints)
    ###########################################################################

    # --- Authentication Routes ---
    @app.route("/login_page", methods=["GET"])
    async def login_page():
        return await render_template("login.html") # Ensure login.html exists

    @app.route("/register_page", methods=["GET"])
    async def register_page():
        return await render_template("register.html") # Ensure register.html exists

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

            if not row: # User not found
                 # Mitigate timing attacks
                 fake_hash = bcrypt.hashpw(b"dummy", bcrypt.gensalt())
                 bcrypt.checkpw(password.encode('utf-8'), fake_hash)
                 logger.warning(f"Login failed (no such user): {username}")
                 return jsonify({"error": "Invalid username or password"}), 401

            user_id, hashed_password_bytes = row['id'], row['password_hash'].encode('utf-8')

            if bcrypt.checkpw(password.encode('utf-8'), hashed_password_bytes):
                 session["user_id"] = user_id
                 session.permanent = True
                 logger.info(f"Login successful: User {user_id}")
                 return jsonify({"message": "Logged in", "user_id": user_id})
            else:
                 logger.warning(f"Login failed (bad password): User {user_id}")
                 return jsonify({"error": "Invalid username or password"}), 401

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Login DB error for {username}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error during login"}), 500
        except Exception as e:
            logger.error(f"Login unexpected error for {username}: {e}", exc_info=True)
            return jsonify({"error": "Server error during login"}), 500

    @app.route("/register", methods=["POST"])
    @rate_limit(limit=3, period=300)
    @validate_request({ # Added patterns, made email optional
        'username': {'type': 'string', 'pattern': r'^[a-zA-Z0-9_.-]{3,30}$', 'required': True},
        'password': {'type': 'string', 'min_length': 8, 'max_length': 100, 'required': True},
        'email':    {'type': 'string', 'pattern': r'[^@]+@[^@]+\.[^@]+', 'max_length': 100, 'required': False}
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
            logger.error(f"Password hashing error: {hash_err}", exc_info=True)
            return jsonify({"error": "Registration error"}), 500

        try:
            async with get_db_connection_context() as conn: # Use async context
                # Use transaction for atomic check-and-insert
                async with conn.transaction():
                    # Check existing username
                    existing_user = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
                    if existing_user:
                        return jsonify({"error": "Username already exists"}), 409

                    # Check existing email if provided and email column exists and is unique
                    if email:
                        existing_email = await conn.fetchval("SELECT id FROM users WHERE email=$1", email)
                        if existing_email:
                            return jsonify({"error": "Email already exists"}), 409

                    # Insert new user
                    user_id = await conn.fetchval( # Use await and $n params
                        """INSERT INTO users (username, password_hash, email, created_at)
                           VALUES ($1, $2, $3, NOW()) RETURNING id""",
                        username, password_hash, email
                    )

            if user_id:
                session["user_id"] = user_id
                session.permanent = True
                logger.info(f"Registration successful: User {user_id} ({username})")
                return jsonify({"message": "User registered successfully", "user_id": user_id}), 201
            else:
                logger.error(f"Registration failed: No user ID returned for {username}")
                return jsonify({"error": "Registration failed"}), 500

        except asyncpg.exceptions.UniqueViolationError as uve:
             # Catch specific unique constraint error
             logger.warning(f"Registration conflict for {username}: {uve}")
             # Determine if username or email caused it based on constraint name if possible
             if 'username' in str(uve): return jsonify({"error": "Username already exists"}), 409
             if 'email' in str(uve): return jsonify({"error": "Email already exists"}), 409
             return jsonify({"error": "Username or email already exists"}), 409
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Registration DB error for {username}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error during registration"}), 500
        except Exception as e:
            logger.error(f"Registration unexpected error for {username}: {e}", exc_info=True)
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
    async def start_new_game(): # Needs to be async
        user_id = session.get("user_id")
        if not user_id: return jsonify({"error": "Not authenticated"}), 401
        logger.info(f"User {user_id} starting new game...")

        try:
            # Create initial conversation record
            async with get_db_connection_context() as conn: # Use async context
                 async with conn.transaction():
                    conv_row = await conn.fetchrow( # Use await and $n params
                        """INSERT INTO conversations (user_id, conversation_name, status)
                        VALUES ($1, $2, 'processing') RETURNING id""",
                        user_id, "New Game - Initializing..."
                    )
                    conversation_id = conv_row['id']
                    logger.info(f"Created conversation {conversation_id} for user {user_id}")

                    # Optionally, insert default player stats immediately
                    # Assuming insert_default_player_stats_chase uses asyncpg or can be awaited
                    await insert_default_player_stats_chase(user_id, conversation_id, conn) # Pass connection


            # 2. Trigger the heavy lifting asynchronously via Celery
            # Pass necessary data to the task
            from tasks import process_new_game_task # Import task
            task_result = process_new_game_task.delay(user_id, {"conversation_id": conversation_id})
            logger.info(f"Dispatched Celery task {task_result.id} for new game {conversation_id}")

            return jsonify({
                "status": "processing",
                "message": "New game creation started.",
                "conversation_id": conversation_id,
                "task_id": task_result.id
            }), 202

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"New game DB error for user {user_id}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error starting game"}), 500
        except Exception as e:
            logger.error(f"New game dispatch error for user {user_id}: {e}", exc_info=True)
            return jsonify({"error": "Server error starting game"}), 500

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
    async def readiness_check(): # Keep async
        status = {"status": "ready", "timestamp": time.time(), "checks": {}}
        is_ready = True
        # DB Check (Async)
        try:
            # Use short timeout
            async with get_db_connection_context(timeout=5) as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1: status["checks"]["database"] = "connected"
                else: status["checks"]["database"] = "error: bad query result"; is_ready = False
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            status["checks"]["database"] = f"error: {type(db_err).__name__}"; is_ready = False
            logger.warning(f"Readiness DB check failed: {db_err}")
        except Exception as e:
            status["checks"]["database"] = f"error: {type(e).__name__}"; is_ready = False
            logger.warning(f"Readiness DB check unexpected error: {e}", exc_info=True)


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

