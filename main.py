# main.py

import os
import logging
import time
import json
import aioredis 
import asyncio
from typing import Dict, Any, Optional

# quart and related imports
from quart import Quart, render_template, session, request, jsonify, redirect, Response, current_app
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
from redis import Redis # Keep Redis sync for now, unless heavy usage requires 
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
from middleware.security import validate_input

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
from middleware.rate_limiting import rate_limit, async_ip_block_middleware # Use the async IP block
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


async def startup_worker_resources(app=None):
    """Initialize resources for each worker, with duplicate initialization protection."""
    worker_pid = os.getpid()
    logger.info(f"Worker {worker_pid}: Initializing resources (before_serving).")

    # Get reference to app
    from quart import current_app
    
    # Use the passed app if provided, otherwise use current_app
    actual_app = app if app is not None else current_app
    
    # --- 0. Initialize systems first ---
    if not getattr(actual_app, 'systems_initialized', False):
        try:
            logger.info(f"Worker {worker_pid}: Initializing application systems...")
            await initialize_systems(actual_app)
            actual_app.systems_initialized = True
            logger.info(f"Worker {worker_pid}: Application systems initialized.")
        except Exception as e:
            logger.error(f"Worker {worker_pid}: Error initializing application systems: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize application systems in worker {worker_pid}.") from e
    

    # --- 2. Initialize Nyx Memory System ---
    if not getattr(current_app, 'nyx_memory_initialized', False):
        try:
            from logic.nyx_enhancements_integration import initialize_nyx_memory_system
            logger.info(f"Worker {worker_pid}: Initializing Nyx Memory System...")
            await initialize_nyx_memory_system()
            current_app.nyx_memory_initialized = True
            logger.info(f"Worker {worker_pid}: Nyx Memory System initialized.")
        except Exception as e:
            logger.error(f"Worker {worker_pid}: Error initializing Nyx Memory System: {e}", exc_info=True)
    else:
        logger.info(f"Worker {worker_pid}: Nyx Memory System already initialized.")

    # --- 3. Initialize NyxBrain DB-dependent parts ---
    if hasattr(current_app, 'nyx_brain') and current_app.nyx_brain and not getattr(current_app, 'nyx_brain_initialized', False):
        logger.info(f"Worker {worker_pid}: Running NyxBrain DB-dependent initialization...")
        try:
            # Run checkpoints restoration now that we have DB access
            await current_app.nyx_brain.restore_entity_from_distributed_checkpoints()
            
            # Initialize any worker-specific state
            if hasattr(current_app.nyx_brain, 'initialize_worker_state'):
                await current_app.nyx_brain.initialize_worker_state(db_pool=current_app.db_pool)
                
            current_app.nyx_brain_initialized = True
            logger.info(f"Worker {worker_pid}: NyxBrain checkpoints restored.")
        except Exception as nyx_e:
            logger.error(f"Worker {worker_pid}: Error in NyxBrain DB initialization: {nyx_e}", exc_info=True)
    elif hasattr(current_app, 'nyx_brain') and current_app.nyx_brain:
        logger.info(f"Worker {worker_pid}: NyxBrain already initialized.")

    # --- 4. Register with governance systems ---
    if not getattr(current_app, 'story_director_initialized', False):
        try:
            from story_agent.story_director_agent import register_with_governance
            story_user_id = 1; story_conversation_id = 1
            logger.info(f"Worker {worker_pid}: Registering StoryDirector with governance...")
            await register_with_governance(story_user_id, story_conversation_id)
            current_app.story_director_initialized = True
            logger.info(f"Worker {worker_pid}: StoryDirector registered with governance.")
        except Exception as e:
            logger.error(f"Worker {worker_pid}: Error registering StoryDirector: {e}", exc_info=True)
    else:
        logger.info(f"Worker {worker_pid}: StoryDirector already initialized.")

    # --- 5. Signal Celery tasks that the app is ready ---
    try:
        from tasks import set_app_initialized
        set_app_initialized()
        logger.info(f"Worker {worker_pid}: Set app initialized status for Celery tasks.")
    except Exception as e:
        logger.error(f"Worker {worker_pid}: Error setting app initialized status: {e}", exc_info=True)



async def shutdown_worker_resources(app=None):
    """Clean up resources for each worker."""
    worker_pid = os.getpid()
    logger.info(f"Worker {worker_pid}: Closing resources (after_serving).")
    
    # Get reference to app
    from quart import current_app
    
    # Use the passed app if provided, otherwise use current_app
    actual_app = app if app is not None else current_app
      
    # 1. Close NyxBrain connections if necessary
    if hasattr(current_app, 'nyx_brain') and current_app.nyx_brain:
        if hasattr(current_app.nyx_brain, 'close_worker_state'):
            try:
                logger.info(f"Worker {worker_pid}: Closing NyxBrain worker state...")
                await current_app.nyx_brain.close_worker_state()
            except Exception as e:
                logger.error(f"Worker {worker_pid}: Error closing NyxBrain worker state: {e}", exc_info=True)
    
    # 2. Close Nyx Memory System if it has a shutdown method
    try:
        from logic.nyx_enhancements_integration import shutdown_nyx_memory_system
        if callable(shutdown_nyx_memory_system):
            logger.info(f"Worker {worker_pid}: Shutting down Nyx Memory System...")
            await shutdown_nyx_memory_system()
    except (ImportError, AttributeError):
        # Function may not exist, so this is not necessarily an error
        pass
    except Exception as e:
        logger.error(f"Worker {worker_pid}: Error shutting down Nyx Memory System: {e}", exc_info=True)
    
    # 3. Close the DB pool and reset initialization flags
    logger.info(f"Worker {worker_pid}: Closing database connection pool...")
    await close_connection_pool(app=current_app)
    
    # Reset initialization flags for potentially reused workers
    current_app.db_initialized = False
    current_app.nyx_memory_initialized = False
    current_app.nyx_brain_initialized = False
    current_app.story_director_initialized = False
    
    logger.info(f"Worker {worker_pid}: DB pool closed and initialization flags reset.")

async def initialize_systems(app: Quart):
    """
    Initialize non-DB-pool and non-Nyx-Memory systems.
    These are application-level singletons or configurations.
    DB_POOL and Nyx Memory System are initialized by each worker via @app.before_serving.
    """
    logger.info("Main App: Starting application systems initialization (DB Pool & Nyx Memory handled by worker lifecycle)...")
    
    # Try-except for critical module imports needed by this function
    try:
        from nyx.core.brain.base import NyxBrain
        from mcp_orchestrator import MCPOrchestrator
        # from logic.nyx_enhancements_integration import initialize_nyx_memory_system # MOVED TO before_serving
        from logic.aggregator_sdk import init_singletons
        from story_agent.story_director_agent import initialize_story_director, register_with_governance
        from nyx.nyx_agent_sdk import process_user_input, process_user_input_with_openai
        from tasks import set_app_initialized # Ensure this exists
        global background_chat_task # Make sure background_chat_task is accessible
    except ImportError as e:
        logger.critical(f"Main App: Import failed in initialize_systems: {e}", exc_info=True)
        raise RuntimeError(f"Module import failed during system initialization: {e}") from e

    logger.warning("Main App: Skipping DB schema/seed/migration checks in app startup. Run these manually.")

    # --- NyxBrain Instance ---
    if hasattr(NyxBrain, "get_instance"):
        try:
            system_user_id = 0; system_conversation_id = 0
            # NyxBrain.get_instance should NOT use the global DB_POOL from db.connection directly
            # if it's meant to be used by workers later.
            # It could take a DSN and create its own internal pool for setup if needed,
            # or its DB-dependent parts should be initialized per-worker.
            # For now, let's assume get_instance is mostly about object creation.
            app.nyx_brain = await NyxBrain.get_instance(system_user_id, system_conversation_id)
            logger.info("Main App: NyxBrain instance object created.")

            # IMPORTANT: DEFER DB-dependent parts of NyxBrain to worker initialization
            # Don't call restore_entity_from_distributed_checkpoints here!
            if app.nyx_brain:
                app.nyx_brain.response_processors = {
                    "default": background_chat_task,
                    "openai": process_user_input_with_openai,
                    "base": process_user_input
                }
                logger.info("Main App: Response processors registered.")
        except Exception as e:
            logger.error(f"Main App: Error initializing NyxBrain: {e}", exc_info=True)
            app.nyx_brain = None
    else:
        logger.warning("Main App: NyxBrain.get_instance not available.")
        app.nyx_brain = None

    # --- MCP orchestrator ---
    try:
        app.mcp_orchestrator = MCPOrchestrator()
        await app.mcp_orchestrator.initialize()
        logger.info("Main App: MCP orchestrator initialized.")
    except Exception as e:
        logger.error(f"Main App: Error initializing MCP Orchestrator: {e}", exc_info=True)
        app.mcp_orchestrator = None

    # --- Admin config ---
    admin_ids_str = os.getenv("ADMIN_USER_IDS", "1")
    try:
        app.config['ADMIN_USER_IDS'] = [int(uid.strip()) for uid in admin_ids_str.split(',')]
    except ValueError:
        logger.error(f"Main App: Invalid ADMIN_USER_IDS: '{admin_ids_str}'. Defaulting to [1].")
        app.config['ADMIN_USER_IDS'] = [1]
    logger.info(f"Main App: Admin User IDs: {app.config['ADMIN_USER_IDS']}")

    # --- Aggregator SDK ---
    try:
        await init_singletons()
        logger.info("Main App: Aggregator SDK singletons initialized.")
    except Exception as e:
        logger.error(f"Main App: Error initializing Aggregator SDK: {e}", exc_info=True)

    # --- StoryDirector ---
    try:
        # IMPORTANT: Don't initialize with DB access here - just set up the basic objects
        # The actual DB operations will happen during worker startup
        story_user_id = 1; story_conversation_id = 1
        await initialize_story_director(story_user_id, story_conversation_id)
        # DEFER this to worker startup: await register_with_governance(story_user_id, story_conversation_id)
        logger.info("Main App: StoryDirector objects initialized. Registration deferred to workers.")
    except Exception as e:
        logger.error(f"Main App: Error initializing StoryDirector: {e}", exc_info=True)

    logger.info("Main App: Attempting to initialize aioredis pools...")
    try:
        redis_url = app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
        if redis_url:
            # Create the pool object. The actual connections are typically made lazily.
            # The ping here is to test connectivity during startup.
            temp_pool_for_ping_test = None
            try:
                temp_pool_for_ping_test = await asyncio.wait_for(
                    aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2),
                    timeout=5 # Timeout for creating the pool object itself
                )
                await asyncio.wait_for(temp_pool_for_ping_test.ping(), timeout=3) # Timeout for the ping
                logger.info("Main App: aioredis ping successful during initial setup.")
                # Now assign the successfully created and pinged pool to the app
                app.aioredis_rate_limit_pool = temp_pool_for_ping_test
                app.aioredis_ip_block_pool = app.aioredis_rate_limit_pool # Share pool
                logger.info("Main App: aioredis pools configured.")
                temp_pool_for_ping_test = None # Avoid closing the one stored on app
            except Exception as e_ping:
                logger.error(f"Main App: aioredis initial ping/setup failed: {e_ping}", exc_info=True)
                app.aioredis_rate_limit_pool = None
                app.aioredis_ip_block_pool = None
            finally:
                if temp_pool_for_ping_test: # If ping failed after pool creation, close the temp one
                    await temp_pool_for_ping_test.close()
        else:
            logger.warning("Main App: REDIS_URL not configured. aioredis pools not created.")
            app.aioredis_rate_limit_pool = None
            app.aioredis_ip_block_pool = None
    except Exception as e:
        logger.error(f"Main App: General error initializing aioredis pools: {e}", exc_info=True)
        app.aioredis_rate_limit_pool = None
        app.aioredis_ip_block_pool = None

###############################################################################
# quart APP CREATION
###############################################################################



def create_quart_app():
    app = Quart(__name__, static_folder="static", template_folder="templates")
    QuartSchema(app)
    
    # Create & attach Socket.IO
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    app.asgi_app = socketio.ASGIApp(sio, app.asgi_app)
    app.socketio = sio

    # IMPORTANT: Create initialization tracking flags
    app.db_initialized = False
    app.nyx_memory_initialized = False
    app.nyx_brain_initialized = False
    app.story_director_initialized = False
    app.systems_initialized = False  # Add this flag
    
    # CORRECTLY register startup/shutdown handlers ONLY ONCE
    app.before_serving(startup_worker_resources)
    app.after_serving(shutdown_worker_resources)
    
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
        # Define allowed CDN sources
        cdn_scripts = "https://cdn.jsdelivr.net https://cdn.socket.io https://code.jquery.com"
        cdn_styles = "https://cdn.jsdelivr.net" # For Bootstrap CSS

        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' {cdn_scripts}; " # Allow 'self' and specific CDNs for scripts
            f"style-src 'self' {cdn_styles}; "   # Allow 'self' and specific CDNs for styles
            "img-src 'self' data: https://*; " # Allow images from self, data URIs, and any HTTPS source (be more specific if possible)
            "font-src 'self' https://cdn.jsdelivr.net; " # If Bootstrap uses webfonts from its CDN
            "connect-src 'self' ws://* wss://* https://nyx-m85p.onrender.com; " # Allows connections to self, all WebSocket URLs, and your Render domain explicitly.
            "frame-ancestors 'none'; " # Good practice to prevent clickjacking
            "object-src 'none'; " # Good practice, disallow <object>, <embed>, <applet>
            "base-uri 'self';" # Restricts <base> tag
            "form-action 'self';" # Restricts where forms can submit to
        )
        # Other security headers (optional but recommended)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY" # Or SAMEORIGIN
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains" # Only if site is HTTPS only

        return response

    @app.before_serving
    async def startup_db_pool():
        """Initialize DB_POOL for this worker process."""
        logger.info(f"Worker {os.getpid()}: Initializing database connection pool (before_serving).")
        # Pass 'app' if you want db.connection to store the pool on app.db_pool
        # This allows routes to access it via current_app.db_pool if needed,
        # though get_db_connection_context can also use the global DB_POOL.
        if not await initialize_connection_pool(app=app):
            logger.critical(f"Worker {os.getpid()}: Database pool initialization FAILED. This worker might not function.")
            # Consider raising an error to stop the worker if DB is absolutely essential from the start
            # For now, it logs and continues, problems will arise when DB is accessed.
            # raise RuntimeError("DB Pool failed to initialize in worker.")
        else:
            logger.info(f"Worker {os.getpid()}: Database pool initialized successfully.")
    



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ FULL VERSION OF REDIS POOL SHUTDOWN LOGIC                      +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    async def shutdown_redis_pools_app_event():
        """Gracefully closes aioredis connection pools stored on the app context."""
        logger.info("Attempting to shut down aioredis pools (app after_serving event)...")
        rate_limit_pool = getattr(app, 'aioredis_rate_limit_pool', None)
        if rate_limit_pool and hasattr(rate_limit_pool, 'close') and callable(rate_limit_pool.close):
            try:
                logger.info("Closing aioredis_rate_limit_pool...")
                await rate_limit_pool.close()
                logger.info("Rate limiter aioredis pool closed successfully.")
            except Exception as e: logger.error(f"Error closing rate_limit_pool: {e}", exc_info=True)
            app.aioredis_rate_limit_pool = None

        ip_block_pool = getattr(app, 'aioredis_ip_block_pool', None)
        if ip_block_pool and ip_block_pool is not rate_limit_pool: # Only close if different instance
            if hasattr(ip_block_pool, 'close') and callable(ip_block_pool.close):
                try:
                    logger.info("Closing aioredis_ip_block_pool...")
                    await ip_block_pool.close()
                    logger.info("IP block list aioredis pool closed successfully.")
                except Exception as e: logger.error(f"Error closing ip_block_pool: {e}", exc_info=True)
            app.aioredis_ip_block_pool = None
        elif ip_block_pool and ip_block_pool is rate_limit_pool: # Was shared
            app.aioredis_ip_block_pool = None # Just clear the reference
            logger.info("IP block list aioredis pool (shared) reference cleared.")

    app.after_serving(shutdown_redis_pools_app_event)

    logger.info("Running initial application setup (asyncio.run(initialize_systems))...")
    try:
        asyncio.run(initialize_systems(app)) # Pass the app instance
    except Exception as init_err:
        logger.critical(f"Core application system initialization failed: {init_err}", exc_info=True)
        raise RuntimeError("Failed to initialize core application systems.") from init_err
    logger.info("Core application systems initialization complete.")

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
    app.before_request(async_ip_block_middleware)
    register_auth_routes(app)

    init_image_routes(app) # Ensure this uses asyncpg if needed
    init_chat_routes(app) # Ensure this uses asyncpg if needed

    # --- Run Async Initializations ---
    # Run the async setup tasks AFTER the main app config but before returning app
    # This uses asyncio.run, which is okay here as it's during initial setup phase.
    
    # Initialize non-DB components synchronously if needed
    try:
        asyncio.run(initialize_systems(app))
        logger.info("Non-DB application systems initialized. Worker-specific initialization will happen at startup.")
    except Exception as init_err:
        logger.critical(f"Non-DB application initialization failed: {init_err}", exc_info=True)
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
    @rate_limit(limit=5, period=60) # Ensure rate_limit uses current_app for its Redis pool
    @validate_input(schema={ # Ensure validate_input uses current_app for its needs if any
        'username': {'type': 'string', 'pattern': 'username', 'required': True},
        'password': {'type': 'string', 'max_length': 100, 'required': True}
    })
    async def login():
        data = getattr(request, 'sanitized_data', None)
        if data is None: return jsonify({"error": "Invalid request data or not JSON"}), 400
        username = data.get("username"); password = data.get("password")
        if not username or not password: return jsonify({"error": "Missing username or password"}), 400

        try:
            # Use current_app if get_db_connection_context was modified to accept it
            # and if you stored the pool on app.db_pool
            async with get_db_connection_context(app=current_app) as conn:
                row = await conn.fetchrow("SELECT id, password_hash FROM users WHERE username=$1", username)

            if not row:
                # Timing attack mitigation: hash a dummy password
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, bcrypt.hashpw, b"dummy", bcrypt.gensalt())
                logger.warning(f"Login attempt for non-existent user: {username}")
                return jsonify({"error": "Invalid username or password"}), 401

            user_id, hashed_password_from_db = row['id'], row['password_hash'].encode('utf-8')

            loop = asyncio.get_running_loop()
            password_matches = await loop.run_in_executor(
                None, bcrypt.checkpw, password.encode('utf-8'), hashed_password_from_db
            )

            if password_matches:
                session["user_id"] = user_id; session.permanent = True
                logger.info(f"Login successful: User {user_id}")
                return jsonify({"message": "Logged in", "user_id": user_id})
            else:
                logger.warning(f"Login failed (bad password): User {user_id} ({username})")
                return jsonify({"error": "Invalid username or password"}), 401
        except asyncio.TimeoutError:
            logger.error(f"Login DB timeout for {username}", exc_info=True)
            return jsonify({"error": "Database operation timed out"}), 503
        except asyncpg.PostgresError as db_err:
            logger.error(f"Login DB error for {username}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error during login"}), 500
        except ConnectionError as conn_err: # From get_db_connection_context
            logger.error(f"Login DB pool error for {username}: {conn_err}", exc_info=True)
            return jsonify({"error": "Database connection issue"}), 503
        except Exception as e:
            logger.error(f"Login unexpected error for {username}: {e}", exc_info=True)
            return jsonify({"error": "Server error during login"}), 500
    
        # Now that 'data' is guaranteed to be a dict (or an error was returned)
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

    @app.route("/register", methods=["POST"]) # Needs `app` to be defined
    @rate_limit(limit=3, period=300)
    @validate_input(schema={
        'username': {'type': 'string', 'pattern': 'username', 'required': True},
        'password': {'type': 'string', 'min_length': 8, 'max_length': 100, 'required': True},
        'email':    {'type': 'string', 'pattern': 'email', 'max_length': 100, 'required': False}
    })
    async def register():
        data = getattr(request, 'sanitized_data', None)
        if data is None:
            logger.warning("/register: request.sanitized_data not found.")
            return jsonify({"error": "Invalid or missing request data"}), 400
    
        username = data.get("username")
        password = data.get("password")
        email = data.get("email")
    
        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400
    
        loop = asyncio.get_running_loop()
        try:
            password_hash_bytes = await loop.run_in_executor(
                None, bcrypt.hashpw, password.encode('utf-8'), bcrypt.gensalt()
            )
            password_hash = password_hash_bytes.decode('utf-8')
        except Exception as hash_err:
            logger.error(f"Password hashing error for user {username}: {hash_err}", exc_info=True)
            return jsonify({"error": "Registration process failed (hashing error)"}), 500
    
        try:
            async with get_db_connection_context(app=current_app) as conn:
                async with conn.transaction():
                    existing_user = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
                    if existing_user:
                        return jsonify({"error": "Username already exists"}), 409
                    if email:
                        existing_email = await conn.fetchval("SELECT id FROM users WHERE email=$1", email)
                        if existing_email:
                            return jsonify({"error": "Email already exists"}), 409
                    user_id = await conn.fetchval(
                        "INSERT INTO users (username, password_hash, email, created_at) VALUES ($1, $2, $3, NOW()) RETURNING id",
                        username, password_hash, email
                    )
            if user_id:
                session["user_id"] = user_id
                session.permanent = True
                logger.info(f"Registration successful: User {user_id} ({username}) created.")
                return jsonify({"message": "User registered successfully", "user_id": user_id}), 201
            else:
                logger.error(f"Registration failed for {username} - no user_id returned.")
                return jsonify({"error": "Registration failed (DB error)"}), 500
        except asyncpg.exceptions.UniqueViolationError as uve:
            logger.warning(f"Registration conflict for {username}: {uve}")
            if 'username' in str(uve).lower() or (uve.constraint_name and 'username' in uve.constraint_name.lower()):
                return jsonify({"error": "Username already taken"}), 409
            elif 'email' in str(uve).lower() or (uve.constraint_name and 'email' in uve.constraint_name.lower()):
                return jsonify({"error": "Email already registered"}), 409
            return jsonify({"error": "User credential already exists"}), 409
        except asyncio.TimeoutError:
            logger.error(f"DB timeout during registration for {username}.", exc_info=True)
            return jsonify({"error": "Registration timed out"}), 503
        except asyncpg.PostgresError as db_err:
            logger.error(f"DB error during registration for {username}: {db_err}", exc_info=True)
            return jsonify({"error": "Database issue during registration"}), 500
        except ConnectionError as conn_err:
            logger.error(f"DB pool error during registration for {username}: {conn_err}", exc_info=True)
            return jsonify({"error": "Database connection problem"}), 503
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
    async def readiness_check(): # All code below must be indented under this
        status = {"status": "ready", "timestamp": time.time(), "checks": {}}
        is_ready = True
        
        # DB Check
        try:
            async with get_db_connection_context(timeout=5, app=current_app) as conn:
                result = await conn.fetchval("SELECT 1")
                status["checks"]["database"] = "connected" if result == 1 else "error: bad query"
                if result != 1: is_ready = False
        except Exception as db_err:
            status["checks"]["database"] = f"error: {type(db_err).__name__}"
            is_ready = False
            logger.warning(f"Readiness DB check failed: {db_err}")
        
        # aioredis Check (CORRECTLY INDENTED)
        redis_url = current_app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
        if redis_url:
            try:
                aredis_client = await asyncio.wait_for( # This is now correctly inside async def
                    aioredis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2),
                    timeout=3
                )
                await aredis_client.ping()
                status["checks"]["aioredis"] = "connected"
                await aredis_client.close()
            except (aioredis.RedisError, ConnectionRefusedError, asyncio.TimeoutError) as aredis_err:
                status["checks"]["aioredis"] = f"error: {type(aredis_err).__name__}"
                is_ready = False
                logger.warning(f"Readiness aioredis check failed: {aredis_err}")
            except Exception as e_aredis:
                status["checks"]["aioredis"] = f"unexpected error: {type(e_aredis).__name__}"
                is_ready = False
                logger.warning(f"Readiness aioredis check unexpected error: {e_aredis}", exc_info=True)
        else:
            status["checks"]["aioredis"] = "not configured (REDIS_URL missing)"

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


    try:
        import wsgi
        wsgi.server_should_exit = True
        # Make a request to the server to trigger handle_request() one more time
        try:
            import urllib.request
            urllib.request.urlopen('http://localhost:8080/shutdown').close()
        except:
            pass  # Ignore errors here
    except:
        pass  # In case wsgi is not available

    async def shutdown_redis_pools():
        logger.info("Attempting to shut down aioredis pools...")
        if hasattr(app, 'aioredis_rate_limit_pool') and app.aioredis_rate_limit_pool:
            await app.aioredis_rate_limit_pool.close()
            # await app.aioredis_rate_limit_pool.wait_closed() # For older aioredis
            logger.info("Rate limiter aioredis pool closed.")
        # No need to close ip_block_pool if it's the same object as rate_limit_pool
        # If they were different:
        # if hasattr(app, 'aioredis_ip_block_pool') and app.aioredis_ip_block_pool and \
        #    app.aioredis_ip_block_pool is not app.aioredis_rate_limit_pool:
        #     await app.aioredis_ip_block_pool.close()
        #     logger.info("IP blocklist aioredis pool closed.")
    
    app.after_serving(shutdown_redis_pools)
    
    return app

