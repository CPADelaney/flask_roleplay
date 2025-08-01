# main.py

import os
import logging
import time
import sys
import json
from redis import asyncio as redis_async
import asyncio
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

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
from logic.stats_logic import insert_default_player_stats_chase

from nyx.core.brain.base import NyxBrain

# MCP Orchestrator
# from mcp_orchestrator import MCPOrchestrator

# NPC creation / learning
from npcs.new_npc_creation import NPCCreationHandler, RunContextWrapper
from npcs.npc_learning_adaptation import NPCLearningManager

# OpenAI and image generation
from logic.chatgpt_integration import build_message_history
from routes.ai_image_generator import init_app as init_image_routes, generate_roleplay_image_from_gpt
from routes.chatgpt_routes import init_app as init_chat_routes
from logic.gpt_image_decision import should_generate_image_for_response
# from logic.gpt_image_prompting import get_system_prompt_with_image_guidance
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

logging.basicConfig(level=logging.INFO)

# Database DSN
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.critical("DB_DSN environment variable not set!")


def ensure_int_ids(user_id=None, conversation_id=None):
    """
    Ensure user_id and conversation_id are integers.
    Returns tuple (user_id, conversation_id) with proper types.
    Raises ValueError if conversion fails.
    """
    # Handle user_id
    if user_id is not None and user_id != "anonymous":
        if isinstance(user_id, str):
            try:
                user_id = int(user_id)
            except ValueError:
                raise ValueError(f"Invalid user_id: '{user_id}' cannot be converted to integer")
        elif not isinstance(user_id, int):
            raise TypeError(f"user_id must be int or str, got {type(user_id)}")
    
    # Handle conversation_id
    if conversation_id is not None:
        if isinstance(conversation_id, str):
            try:
                conversation_id = int(conversation_id)
            except ValueError:
                raise ValueError(f"Invalid conversation_id: '{conversation_id}' cannot be converted to integer")
        elif not isinstance(conversation_id, int):
            raise TypeError(f"conversation_id must be int or str, got {type(conversation_id)}")
    
    return user_id, conversation_id

###############################################################################
# BACKGROUND TASKS (Called via SocketIO or Celery)
###############################################################################

# Ensure this task ONLY uses asyncpg for DB access
async def background_chat_task(conversation_id, user_input, user_id, universal_update=None, sio=None):
    """
    Background task for processing chat messages using Nyx agent with OpenAI integration.
    """
    # FIX: Convert string IDs to integers immediately
    try:
        if isinstance(conversation_id, str):
            conversation_id = int(conversation_id)
        if isinstance(user_id, str) and user_id != "anonymous":
            user_id = int(user_id)
    except (ValueError, TypeError) as e:
        logger.error(f"[BG Task] Invalid ID format: conversation_id={conversation_id}, user_id={user_id}, error={e}")
        if sio:
            await sio.emit('error', {'error': 'Invalid ID format'}, room=str(conversation_id))
        return
        
    from quart import current_app
    if not sio:
        logger.error(f"[BG Task {conversation_id}] No socketio instance provided")
        return

    logger.info(f"[BG Task {conversation_id}] Starting for user {user_id}")
    try:
        # Get aggregator context (ensure this function is async or thread-safe if it hits DB)
        # If sync and hits DB, consider running in an executor or making it async
        from logic.aggregator_sdk import get_aggregated_roleplay_context
        aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, "Chase") # Adjust player name if needed

        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
            "player_name": aggregator_data.get("playerName", "Chase"),
            "npc_present": aggregator_data.get("npcsPresent", []),
            "aggregator_data": aggregator_data
        }

        # Apply universal update if provided (ensure this uses its own connection)
        if universal_update:
            logger.info(f"[BG Task {conversation_id}] Applying universal updates...")
            try:
                from logic.universal_updater_agent import apply_universal_updates_async, UniversalUpdaterContext
                
                updater_context = UniversalUpdaterContext(user_id, conversation_id)
                await updater_context.initialize()
                
                # apply_universal_updates_async should use its own connection internally
                # Don't pass a connection from outside
                await apply_universal_updates_async(
                    updater_context,
                    user_id,
                    conversation_id,
                    universal_update,
                    None  # Let it get its own connection
                )
                
                logger.info(f"[BG Task {conversation_id}] Applied universal updates.")
                
                # Refresh aggregator data post-update
                aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
                context["aggregator_data"] = aggregator_data
                
            except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as update_db_err:
                logger.error(f"[BG Task {conversation_id}] DB Error applying universal updates: {update_db_err}", exc_info=True)
                await sio.emit('error', {'error': 'Failed to apply world updates.'}, room=str(conversation_id))
                return
            except Exception as update_err:
                logger.error(f"[BG Task {conversation_id}] Error applying universal updates: {update_err}", exc_info=True)
                await sio.emit('error', {'error': 'Failed to apply world updates.'}, room=str(conversation_id))
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
            await sio.emit('error', {'error': error_msg}, room=str(conversation_id))
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
                    await sio.emit('image', {
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
                await sio.emit('new_token', {'token': token}, room=str(conversation_id))
                await asyncio.sleep(delay) # Use asyncio.sleep in async task

            await sio.emit('done', {'full_text': message_content}, room=str(conversation_id))
            logger.info(f"[BG Task {conversation_id}] Finished streaming response.")
        else:
             logger.warning(f"[BG Task {conversation_id}] No message content to stream.")
             await sio.emit('done', {'full_text': ''}, room=str(conversation_id)) # Still signal done

    except Exception as e:
        logger.error(f"[BG Task {conversation_id}] Critical Error: {str(e)}", exc_info=True)
        await sio.emit('error', {'error': f"Server error processing message: {str(e)}"}, room=str(conversation_id))

app_is_ready = asyncio.Event()

async def initialize_preset_stories():
    """Load all preset stories into database on startup"""
    from story_templates.preset_story_loader import PresetStoryLoader
    
    try:
        # This will load THE_MOTH_AND_FLAME story
        await PresetStoryLoader.load_all_preset_stories()
        logger.info("Preset stories loaded successfully")
        
        # Verify they're in the database
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM PresetStories")
            logger.info(f"Total preset stories in database: {count}")
            
            # List all story IDs
            rows = await conn.fetch("SELECT story_id FROM PresetStories")
            for row in rows:
                logger.info(f"Available preset story: {row['story_id']}")
                
    except Exception as e:
        logger.error(f"Failed to load preset stories: {e}", exc_info=True)

async def initialize_systems(app: Quart):
    logger.info("Starting asynchronous system initializations...")
    # Imports for initialize_systems
    from nyx.core.brain.base import NyxBrain
    from tasks import set_app_initialized
    from logic.aggregator_sdk import init_singletons
    from story_agent.story_director_agent import initialize_story_director, register_with_governance
    from db.connection import initialize_connection_pool, close_connection_pool
    from logic.nyx_enhancements_integration import initialize_nyx_memory_system
  #  from mcp_orchestrator import MCPOrchestrator

    try:
        # --- 1. Database Connection Pool ---
        logger.info("Initializing database connection pool...")
        if not await initialize_connection_pool(app=app):
            raise RuntimeError("Database pool initialization failed critically.")
        logger.info("Database connection pool initialized successfully.")

        # --- 2. Redis Connection Pools (Centralized Here) ---
        logger.info("Initializing Redis async pools...")
        try:
            redis_url = app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
            
            if redis_url:
                # Create connection pool
                shared_redis_pool = await redis_async.from_url(
                    redis_url,
                    decode_responses=True,
                    max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", 10)),
                    socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", 5)),
                    socket_connect_timeout=int(os.getenv("REDIS_CONNECT_TIMEOUT", 5))
                )
                
                # Test connection
                await shared_redis_pool.ping()
                
                # Store pools on app (you might want to rename these attributes)
                app.redis_rate_limit_pool = shared_redis_pool
                app.redis_ip_block_pool = shared_redis_pool
                
                logger.info("Redis async pools initialized successfully.")
            else:
                logger.warning("REDIS_URL not configured.")
                app.redis_rate_limit_pool = None
                app.redis_ip_block_pool = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}", exc_info=True)
            app.redis_rate_limit_pool = None
            app.redis_ip_block_pool = None

        # --- 3. Core Application Logic (Nyx, MCP, etc.) ---
        # COMMENTED OUT NYX BRAIN INITIALIZATION
        """
        logger.info("Initializing Nyx memory system...")
        await initialize_nyx_memory_system()
        logger.info("Nyx memory system initialized successfully.")

        logger.info("Initializing global NyxBrain instance...")
        try:
            system_user_id = 0
            system_conversation_id = 0
            if hasattr(NyxBrain, "get_instance"):
                app.nyx_brain = await NyxBrain.get_instance(system_user_id, system_conversation_id)
                logger.info("Global NyxBrain instance obtained/initialized.")
                if app.nyx_brain:
                    await app.nyx_brain.restore_entity_from_distributed_checkpoints()
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
             logger.error(f"Could not import or use NyxBrain components: {e}. NyxBrain might be unavailable.", exc_info=True)
             app.nyx_brain = None
        except Exception as e:
             logger.error(f"Error initializing NyxBrain: {e}", exc_info=True)
             app.nyx_brain = None
        
        if not app.nyx_brain: # Example of a critical check
            # raise RuntimeError("NyxBrain initialization failed, which is critical.")
            logger.warning("NyxBrain initialization failed. Some features might be unavailable.")
        """
        
        # Set app.nyx_brain to None since we're not initializing it
        app.nyx_brain = None
        logger.info("NyxBrain initialization skipped (commented out)")

        # COMMENTED OUT MCP ORCHESTRATOR
        #logger.info("Initializing MCP orchestrator...")
        #try:
        #    app.mcp_orchestrator = MCPOrchestrator()
        #    await app.mcp_orchestrator.initialize()
        #    logger.info("MCP orchestrator initialized.")
        #except Exception as e:
        #     logger.error(f"Error initializing MCP Orchestrator: {e}", exc_info=True)

        # --- 4. Other System Initializations ---
        logger.info("Initializing Aggregator SDK singletons...")
        await init_singletons()
        logger.info("Aggregator SDK singletons are ready.")

        # --- 5. Configuration Settings ---
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "1")
        try:
            app.config['ADMIN_USER_IDS'] = [int(uid.strip()) for uid in admin_ids_str.split(',')]
        except ValueError:
            logger.error(f"Invalid ADMIN_USER_IDS format: '{admin_ids_str}'. Defaulting to [1].")
            app.config['ADMIN_USER_IDS'] = [1]
        logger.info(f"Admin User IDs configured: {app.config['ADMIN_USER_IDS']}")

        await initialize_preset_stories()
        logger.info("Preset stories are ready.")

        # --- 6. Final Readiness Signals ---
        set_app_initialized() # For Celery
        logger.info("Celery tasks application initialization status set to True.")

        app_is_ready.set() # For Quart app
        logger.info("Quart application is_ready event set. Application fully initialized and ready to serve.")

    except Exception as e:
        logger.critical(f"Fatal error during system initialization: {str(e)}", exc_info=True)
        # app_is_ready will NOT be set.
        # Ensure Celery's _APP_INITIALIZED reflects this failure if tasks.py uses it.
        # (set_app_initialized() should not have been called if we errored before it)
        raise # Re-raise to halt app startup if this function is awaited.
        
###############################################################################
# quart APP CREATION
###############################################################################

def create_quart_app():
    app = Quart(__name__, static_folder="static", template_folder="templates")
    QuartSchema(app)

    app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SESSION_TYPE'] = 'filesystem'
    # Optionally set session lifetime - 7 days here
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
    
    # 2) Create & attach Socket.IO _before_ any @sio.event handlers
    sio = socketio.AsyncServer(
        async_mode="asgi", 
        cors_allowed_origins="*",
        ping_timeout=20, # Reduce to more standard value
        ping_interval=10,
        max_http_buffer_size=1024*1024, # Reduce slightly to 1MB
        logger=True,
        engineio_logger=True,
        async_handlers=True, # Enable async event handlers
        always_connect=True, # Be more permissive in connections
        http_compression=True # Enable HTTP compression
    )


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
    @sio.on("storybeat")
    async def on_storybeat(sid, data):
        if not app_is_ready.is_set():
            logger.warning(f"Received 'storybeat' from sid={sid} before app is fully ready. Rejecting.")
            await sio.emit('error', {'error': 'Server is initializing, please try again in a moment.'}, to=sid)
            return
        
        sock_sess = await sio.get_session(sid)
        user_id = sock_sess.get("user_id", "anonymous")
        
        # Reject anonymous users
        if user_id == "anonymous":
            logger.warning(f"Socket session has anonymous user, cannot process authenticated request")
            await sio.emit('error', {
                'error': 'Not authenticated. Please refresh the page and log in again.',
                'requiresAuth': True
            }, to=sid)
            return
        
        # Ensure user_id is int
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)
        
        # Get and validate conversation_id
        conversation_id = data.get("conversation_id")
        if conversation_id is not None:
            try:
                conversation_id = int(conversation_id)
            except (ValueError, TypeError):
                logger.error(f"Invalid conversation_id from client: {conversation_id}")
                await sio.emit('error', {'error': 'Invalid conversation_id format'}, to=sid)
                return
        
        user_input = data.get("user_input")
        universal_update = data.get("universal_update")
        
        app.logger.info(f"Received 'storybeat' from sid={sid}, user_id={user_id}, conv_id={conversation_id}")
        
        # Basic validation
        if not all([conversation_id is not None, user_input is not None]):
            error_msg = "Invalid 'storybeat' payload: missing conversation_id or user_input."
            app.logger.error(f"{error_msg} SID: {sid}. Data: {data}")
            await sio.emit('error', {'error': error_msg}, room=str(conversation_id))
            return
        
        try:
            await sio.emit("processing", {"message": "Your request is being processed..."}, to=sid)
            
            # Start background task with proper user_id
            sio.start_background_task(
                background_chat_task,
                conversation_id,
                user_input,
                user_id,  # This will now be the actual user ID, not "anonymous"
                universal_update,
                sio
            )
            app.logger.info(f"Started background_chat_task for sid={sid}, user_id={user_id}, conv_id={conversation_id}")
        
        except Exception as e:
            app.logger.error(f"Error dispatching background_chat_task for sid={sid}: {e}", exc_info=True)
            await sio.emit('error', {'error': 'Server failed to initiate message processing.'}, room=str(conversation_id))


    @app.before_serving
    async def on_startup():
        try:
            # IMPORTANT: initialize_systems should now handle all critical async setups
            await initialize_systems(app) 
        except Exception as e:
            logger.critical(f"Application startup failed during initialize_systems: {e}", exc_info=True)
            # This will prevent Hypercorn from fully starting if init fails
            raise 

    @app.after_serving
    async def shutdown_resources():
        logger.info("Starting graceful shutdown of resources...")

        # Close aioredis pools
        if hasattr(app, 'aioredis_rate_limit_pool') and app.aioredis_rate_limit_pool:
            try:
                await app.aioredis_rate_limit_pool.close()
                # await app.aioredis_rate_limit_pool.wait_closed() # For older redis-py versions or if needed
                logger.info("aioredis pool for Rate Limiter (and IP Block) closed.")
            except Exception as e:
                logger.error(f"Error closing aioredis_rate_limit_pool: {e}", exc_info=True)
        
        # Close database pool (your existing db.connection.close_connection_pool should handle app.db_pool)
        try:
            await close_connection_pool(app=app) # Pass app if your function expects it
            logger.info("Database connection pool closed via db.connection.close_connection_pool.")
        except Exception as e:
            logger.error(f"Error closing database connection pool: {e}", exc_info=True)
        
        logger.info("Resource shutdown complete.")

    @sio.event
    async def disconnect(sid):
        # sock_sess = await sio.get_session(sid) # This might fail if session is already gone
        # user_id = sock_sess.get("user_id", "unknown") if sock_sess else "unknown"
        app.logger.warning(f"SERVER-SIDE: Socket disconnected: sid={sid}.")
        
    @sio.event
    async def client_heartbeat(sid, data):
        """Handle client heartbeat to keep connections alive."""
        try:
            # Respond with server heartbeat
            await sio.emit('server_heartbeat', {'timestamp': time.time()}, room=sid)
        except Exception as e:
            logger.error(f"Error handling heartbeat from {sid}: {e}")
        
    @sio.event
    async def connect(sid, environ, auth):
        # Get user_id from auth (passed by client)
        user_id = auth.get("user_id") if auth else None
        
        # Convert to int if it's a valid numeric string
        if user_id and str(user_id).isdigit():
            user_id = int(user_id)
        elif not user_id:
            # If not in auth, try to get from HTTP session
            # This requires parsing cookies from environ
            cookie_header = environ.get('HTTP_COOKIE', '')
            # For now, if no auth user_id, default to anonymous
            user_id = "anonymous"
            app.logger.warning(f"No user_id in auth for sid={sid}, defaulting to anonymous")
        
        # Save to socketio session
        await sio.save_session(sid, {"user_id": user_id})
        app.logger.info(f"Socket connected: sid={sid}, user_id={user_id}")
        await sio.emit("response", {"data": "Connected!", "user_id": user_id}, to=sid)

    @sio.on("join")
    async def on_join(sid, data):
        sock_sess = await sio.get_session(sid)
        user_id = sock_sess.get("user_id", "anonymous")
        
        # Validate user is authenticated
        if user_id == "anonymous":
            await sio.emit("error", {"error": "Not authenticated"}, to=sid)
            return
        
        # Ensure proper integer conversion
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)
        
        try:
            conversation_id = int(data["conversation_id"])
            room = str(conversation_id)  # Room names should be strings
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid conversation_id in join request: {data}")
            await sio.emit("error", {"error": "Invalid conversation_id"}, to=sid)
            return
        
        await sio.enter_room(sid, room)
        await sio.emit("joined", {"room": room, "user_id": user_id}, to=sid)
        app.logger.info(f"User {user_id} joined room {room}")

    @sio.on("message")
    async def on_message(sid, data):
        sock_sess = await sio.get_session(sid)
        user_id = sock_sess.get("user_id", "anonymous")
        
        # Validate authentication
        if user_id == "anonymous":
            logger.error(f"Anonymous user attempted to send message")
            await sio.emit("error", {"error": "Not authenticated"}, to=sid)
            return
        
        # Ensure user_id is int
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)
        
        # Extract and convert conversation_id
        conversation_id = data.get("conversation_id")
        if conversation_id is not None:
            try:
                conversation_id = int(conversation_id)
            except (ValueError, TypeError):
                logger.error(f"Invalid conversation_id from client: {conversation_id}")
                await sio.emit("error", {"error": "Invalid conversation_id format"}, to=sid)
                return
        else:
            await sio.emit("error", {"error": "Missing conversation_id"}, to=sid)
            return
        
        # Get the message content
        message_content = data.get("message", "").strip()
        if not message_content:
            await sio.emit("error", {"error": "Empty message"}, to=sid)
            return
        
        # Emit acknowledgment
        await sio.emit("message_received", {"status": "processing"}, to=sid)
        
        # Start background task
        try:
            sio.start_background_task(
                background_chat_task,
                conversation_id,
                message_content,
                user_id,  # Properly authenticated user_id
                None,  # universal_update
                sio
            )
            app.logger.info(f"Started background task for message from user {user_id}, conv_id={conversation_id}")
            
        except Exception as e:
            app.logger.error(f"Error starting message processing task: {e}", exc_info=True)
            await sio.emit('error', {'error': 'Failed to process message'}, room=str(conversation_id))

    # 6) Security headers
    @app.after_request
    async def set_security_headers(response):
        # Define allowed CDN sources
        cdn_scripts = "https://cdn.jsdelivr.net https://cdn.socket.io https://code.jquery.com"
        cdn_styles = "https://cdn.jsdelivr.net"  # e.g. Bootstrap CSS

        # Build a CSP that allows exactly one inline <script> block:
        csp = (
            "default-src 'self'; "
            # Allow our one inline <script> (for window.CURRENT_USER_ID) plus the CDNs:
            f"script-src 'self' 'unsafe-inline' {cdn_scripts}; "
            # Allow inline styles for your CSS plus the CDN:
            f"style-src 'self' {cdn_styles} 'unsafe-inline'; "
            "img-src 'self' data: https://*; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "connect-src 'self' ws://* wss://* https://nyx-m85p.onrender.com; "
            "frame-ancestors 'none'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )

        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    
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

    ###########################################################################
    # ROUTES (Defined in main app - keep minimal, prefer blueprints)
    ###########################################################################

    @app.route("/login_page", methods=["GET"])
    async def login_page():
        return await render_template("login.html") # Ensure login.html exists

    @app.route("/register_page", methods=["GET"])
    async def register_page():
        return await render_template("register.html") # Ensure register.html exists    
    # --- Authentication Routes ---
    # At the top of main.py, add this import:
    from db.connection import get_db_dsn

    @app.route("/socket-health")
    async def socket_health():
        active_count = len(app.socketio.manager.get_participants())
        return {"status": "healthy", "active_connections": active_count}
    
    @app.route("/login", methods=["POST"])
    @rate_limit(limit=5, period=60)
    @validate_input({
        'username': {'type': 'string', 'pattern': 'username', 'required': True},
        'password': {'type': 'string', 'max_length': 100, 'required': True}
    })
    async def login():
        data = getattr(request, 'sanitized_data', None)
        if data is None:
            try:
                data = await request.get_json()
            except Exception:
                return jsonify({"error": "Invalid JSON request body"}), 400
        
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400
    
        try:
            # Create connection with statement_cache_size=0 to fix pgbouncer issue
            dsn = get_db_dsn()
            conn = await asyncpg.connect(dsn, statement_cache_size=0)
            try:
                row = await conn.fetchrow(
                    "SELECT id, password_hash FROM users WHERE username=$1",
                    username
                )
                
                if not row:
                    # Timing attack mitigation
                    fake_hash = bcrypt.hashpw(b"dummy", bcrypt.gensalt())
                    bcrypt.checkpw(password.encode('utf-8'), fake_hash)
                    logger.warning(f"Login failed (no such user): {username}")
                    return jsonify({"error": "Invalid username or password"}), 401
                    
                user_id, hashed_password = row['id'], row['password_hash']
                
                try:
                    hashed_password_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
                    
                    if bcrypt.checkpw(password.encode('utf-8'), hashed_password_bytes):
                        session["user_id"] = user_id
                        session.permanent = True
                        logger.info(f"Login successful: User {user_id}")
                        return jsonify({"message": "Logged in", "user_id": user_id})
                    else:
                        logger.warning(f"Login failed (bad password): User {user_id}")
                        return jsonify({"error": "Invalid username or password"}), 401
                except ValueError as salt_err:
                    logger.error(f"Password hash format error for {username}: {salt_err}")
                    return jsonify({"error": "System error during authentication"}), 500
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Login error for {username}: {e}", exc_info=True)
            return jsonify({"error": "Database error during login"}), 500

    @app.before_request
    async def validate_session():
        """Ensure session data is valid"""
        if 'user_id' in session:
            user_id = session.get('user_id')
            # Clear invalid sessions
            if user_id == "anonymous" or user_id is None:
                session.clear()

    @app.route("/register", methods=["POST"])
    @rate_limit(limit=3, period=300)
    @validate_input({
        'username': {'type': 'string', 'pattern': 'username', 'required': True},
        'password': {'type': 'string', 'min_length': 8, 'max_length': 100, 'required': True},
        'email': {'type': 'string', 'pattern': 'email', 'max_length': 100, 'required': False}
    })
    async def register():
        data = getattr(request, 'sanitized_data', None)
        if data is None:
            try:
                data = await request.get_json()
            except Exception:
                return jsonify({"error": "Invalid request data"}), 400
                
        username = data.get("username")
        password = data.get("password")
        email = data.get("email")
    
        if not username or not password:
            return jsonify({"error": "Missing required fields"}), 400
    
        # Hash password
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except Exception as hash_err:
            logger.error(f"Password hashing error: {hash_err}", exc_info=True)
            return jsonify({"error": "Registration error"}), 500
    
        try:
            # Create a fresh connection with statement_cache_size=0 to fix pgbouncer issue
            dsn = get_db_dsn()
            conn = await asyncpg.connect(dsn, statement_cache_size=0)
            try:
                # Use transaction for atomic check-and-insert
                async with conn.transaction():
                    # Check existing username
                    existing_user = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
                    if existing_user:
                        return jsonify({"error": "Username already exists"}), 409
    
                    # Check if the 'email' column exists in the users table
                    has_email_column = False
                    try:
                        # Query information_schema to check if email column exists
                        email_check = await conn.fetchval("""
                            SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_name = 'users' AND column_name = 'email'
                        """)
                        has_email_column = email_check > 0
                    except Exception as schema_err:
                        logger.warning(f"Error checking for email column: {schema_err}")
                        has_email_column = False
    
                    # Insert new user based on table structure
                    user_id = None
                    if has_email_column:
                        # If email column exists, include it in the query
                        if email:
                            # Check if email already exists
                            existing_email = await conn.fetchval("SELECT id FROM users WHERE email=$1", email)
                            if existing_email:
                                return jsonify({"error": "Email already exists"}), 409
                                
                        user_id = await conn.fetchval(
                            """INSERT INTO users (username, password_hash, email, created_at)
                               VALUES ($1, $2, $3, NOW()) RETURNING id""",
                            username, password_hash, email
                        )
                    else:
                        # If email column doesn't exist, skip it
                        user_id = await conn.fetchval(
                            """INSERT INTO users (username, password_hash, created_at)
                               VALUES ($1, $2, NOW()) RETURNING id""",
                            username, password_hash
                        )
    
                if user_id:
                    session["user_id"] = user_id
                    session.permanent = True
                    logger.info(f"Registration successful: User {user_id} ({username})")
                    return jsonify({"message": "User registered successfully", "user_id": user_id}), 201
                else:
                    logger.error(f"Registration failed: No user ID returned for {username}")
                    return jsonify({"error": "Registration failed"}), 500
            finally:
                await conn.close()
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

    @app.route("/chat")
    async def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        
        user_id = session.get("user_id")
        
        # Additional validation to ensure user_id is valid
        if not user_id or user_id == "anonymous":
            session.clear()  # Clear invalid session
            return redirect("/login_page")
        
        # Await the render_template call
        return await render_template("chat.html", user_id=user_id)

    # Note: /start_chat and /openai_chat POST routes were removed as the primary interaction
    # now seems to happen via SocketIO ('message' event). If you need these HTTP endpoints,
    # ensure they use asyncpg and potentially start Celery tasks instead of socketio background tasks.

    @app.route("/start_new_game", methods=["POST"])
    async def start_new_game():
        user_id = session.get("user_id")
        if not user_id: 
            return jsonify({"error": "Not authenticated"}), 401
        
        logger.info(f"User {user_id} starting new game...")
    
        try:
            # Create initial conversation record
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    conv_row = await conn.fetchrow(
                        """INSERT INTO conversations (user_id, conversation_name, status)
                        VALUES ($1, $2, 'processing') RETURNING id""",
                        user_id, "New Game - Initializing..."
                    )
                    conversation_id = conv_row['id']
                    
                    # Use the existing connection with its transaction
                    await insert_default_player_stats_chase(user_id, conversation_id, conn)
    
            # Trigger the heavy lifting asynchronously via Celery (ONLY ONCE)
            from tasks import process_new_game_task
            logger.info(f"ABOUT TO DISPATCH TASK: user_id={user_id}, conversation_id={conversation_id}")
            task_result = process_new_game_task.delay(user_id, {"conversation_id": conversation_id})
            logger.info(f"TASK DISPATCHED: task_id={task_result.id}, status={task_result.status}")
            
            return jsonify({
                "job_id": task_result.id,
                "conversation_id": conversation_id,
                "task_status": task_result.status
            }), 202
    
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"New game DB error for user {user_id}: {db_err}", exc_info=True)
            return jsonify({"error": "Database error starting game"}), 500
        except Exception as e:
            logger.error(f"New game dispatch error for user {user_id}: {e}", exc_info=True)
            return jsonify({"error": "Server error starting game"}), 500

    # --- Admin/Debug Routes ---
    @app.route('/nyx_space/messages', methods=['GET'])
    async def get_nyx_space_messages():
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        user_id = session["user_id"]
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                "SELECT message FROM nyx_dm_messages WHERE user_id = $1 ORDER BY created_at ASC", user_id)
            messages = [row['message'] for row in rows]
        return jsonify({"messages": messages})

    @app.route('/nyx_space/messages', methods=['POST'])
    async def post_nyx_space_message():
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        user_id = session["user_id"]
        data = await request.get_json()
        msg_obj = {
            "sender": data.get("sender"),
            "content": data.get("content"),
            "timestamp": data.get("timestamp"), # or time.time()
        }
        async with get_db_connection_context() as conn:
            await conn.execute(
                "INSERT INTO nyx_dm_messages (user_id, message) VALUES ($1, $2)",
                user_id, json.dumps(msg_obj)
            )
        return jsonify({"ok": True})


    
    @app.route("/admin/nyx_direct", methods=["POST"])
    async def admin_nyx_direct():
        """
        Direct access to NyxBrain for admin users only, with full feature control.
        Accepts:
            user_input: str (required)
            context: dict (optional, will be merged with admin_mode=True)
            use_thinking: bool (optional)
            use_conditioning: bool (optional)
            use_coordination: bool (optional)
            thinking_level: int (optional)
            mode: str (optional)
            generate_response: bool (optional, default True)
        """
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
    
        # Check admin
        admin_ids = app.config.get('ADMIN_USER_IDS', [])
        if session.get("user_id") not in admin_ids:
            logger.warning(f"Non-admin user {session.get('user_id')} attempted Nyx direct access.")
            return jsonify({"error": "Access denied"}), 403
    
        data = await request.get_json(force=True, silent=True) or {}
        user_input = data.get("user_input")
        if not user_input:
            return jsonify({"error": "Missing user_input in request"}), 400
    
        # Feature toggles (accept null/None as "auto-detect")
        def to_bool(val):
            if isinstance(val, bool): return val
            if isinstance(val, str): return val.lower() in ['true', '1', 'yes', 'on']
            return None if val is None else bool(val)
        
        use_thinking = data.get("use_thinking")
        use_conditioning = data.get("use_conditioning")
        use_coordination = data.get("use_coordination")
        thinking_level = data.get("thinking_level")
        mode = data.get("mode")
        context = data.get("context") or {}
        context["admin_mode"] = True
    
        # Accept hierarchical memory (if supported)
        use_hierarchical_memory = data.get("use_hierarchical_memory", None)
    
        # Whether to run generate_response as well
        want_response = data.get("generate_response", True)
    
        nyx_brain = getattr(app, 'nyx_brain', None)
        if not nyx_brain:
            logger.error("NyxBrain not initialized on app context.")
            return jsonify({"error": "Nyx system not available"}), 503
    
        try:
            # Always run process_input with full toggle support
            processing_result = await nyx_brain.process_input(
                user_input=user_input,
                context=context,
                use_thinking=to_bool(use_thinking),
                use_conditioning=to_bool(use_conditioning),
                use_coordination=to_bool(use_coordination),
                thinking_level=int(thinking_level) if thinking_level is not None else None,
                mode=mode
            )
    
            # Optionally, run generate_response if desired
            if want_response:
                response_result = await nyx_brain.generate_response(
                    user_input=user_input,
                    context=context,
                    use_thinking=to_bool(use_thinking),
                    use_conditioning=to_bool(use_conditioning),
                    use_coordination=to_bool(use_coordination),
                    use_hierarchical_memory=to_bool(use_hierarchical_memory),
                    mode=mode
                )
            else:
                response_result = None
    
            result = {
                "processing_result": processing_result,
                "response_result": response_result,
                "admin_mode": True
            }
            return jsonify(result)
    
        except Exception as e:
            logger.error(f"Error during admin Nyx direct call: {e}", exc_info=True)
            return jsonify({"error": "Error processing direct Nyx command", "details": str(e)}), 500



    @app.route("/nyx_response", methods=["POST"])
    async def get_nyx_response():
        """Regular user access via nyx_agent_sdk"""
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not authenticated"}), 401
    
        data = await request.get_json()  # Add 'await' here
        if not data or "user_input" not in data or "conversation_id" not in data:
             return jsonify({"error": "Missing user_input or conversation_id"}), 400
    
        user_input = data.get("user_input")
        conversation_id = data.get("conversation_id")
        # TODO: Add validation to ensure user owns this conversation_id
    
        # Use the distilled agent SDK
        from nyx.nyx_agent_sdk import process_user_input
    
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
        redis_host = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        try:
            # Get the aioredis pool/client similar to how your middleware does.
            # For simplicity, let's assume you have a way to get an aioredis client instance.
            # If you stored the pool on 'current_app' during initialize_systems:
            # redis_pool = getattr(current_app, 'aioredis_rate_limit_pool', None)
            # Or create a temporary one for the check if not easily accessible:
            redis_url = current_app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
            if redis_url:
                # Use a timeout for the connection attempt in readiness
                try:
                    aredis_client = await asyncio.wait_for(
                        aioredis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2),
                        timeout=3 # Overall timeout for from_url and ping
                    )
                    await aredis_client.ping()
                    status["checks"]["aioredis"] = "connected"
                    await aredis_client.close() # Close the temporary client/pool
                except (aioredis.RedisError, ConnectionRefusedError, asyncio.TimeoutError) as aredis_err:
                    status["checks"]["aioredis"] = f"error: {type(aredis_err).__name__}"
                    is_ready = False
                    logger.warning(f"Readiness aioredis check failed: {aredis_err}")
                except Exception as e_aredis: # Catch any other exception during aioredis init/ping
                    status["checks"]["aioredis"] = f"unexpected error: {type(e_aredis).__name__}"
                    is_ready = False
                    logger.warning(f"Readiness aioredis check unexpected error: {e_aredis}", exc_info=True)
            else:
                status["checks"]["aioredis"] = "not configured (REDIS_URL missing)"
                is_ready = False # Or handle as per your requirements

        except Exception as e: # General catch for the try block
            status["checks"]["aioredis"] = f"error: {type(e).__name__}"
            is_ready = False
            logger.warning(f"Readiness check aioredis setup error: {e}", exc_info=True)

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

#    @app.before_serving
#    async def init_redis_pools():  # Remove the 'app' parameter
#        """Initialize Redis connection pools properly."""
#        try:
#            redis_url = app.config.get('REDIS_URL', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
#            if redis_url:
#                # For Rate Limiter
#                app.aioredis_rate_limit_pool = await aioredis.from_url(
#                    redis_url, 
#                    decode_responses=True,
#                    max_connections=10
#                )
#                await app.aioredis_rate_limit_pool.ping()  # Test connection
#                logger.info("aioredis pool for Rate Limiter initialized.")
#                
#                # Use the same pool for IP blocking
#                app.aioredis_ip_block_pool = app.aioredis_rate_limit_pool
#               logger.info("aioredis pool for IP Block List configured.")
#           else:
#               logger.warning("REDIS_URL not configured. Using in-memory rate limiting.")
#                app.aioredis_rate_limit_pool = None
#                app.aioredis_ip_block_pool = None
#            
#        except Exception as e:
#            logger.error(f"Failed to initialize aioredis pools: {e}", exc_info=True)
#            app.aioredis_rate_limit_pool = None
#            app.aioredis_ip_block_pool = None
#    
#    @app.after_serving
#    async def shutdown_redis_pools():
#        """Properly close Redis connections on shutdown."""
#        logger.info("Closing Redis connection pools...")
#        if hasattr(app, 'aioredis_rate_limit_pool') and app.aioredis_rate_limit_pool:
#            await app.aioredis_rate_limit_pool.close()
#            logger.info("Redis rate limiter pool closed.")
#    
#    app.after_serving(shutdown_redis_pools)
    
    return app
