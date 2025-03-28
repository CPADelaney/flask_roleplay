import os
import logging
import time
from flask import Flask, render_template, session, request, jsonify, redirect
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
import bcrypt  # Import bcrypt for password hashing
import secrets  # For generating secure keys
import atexit  # For cleanup functions
from datetime import timedelta
from flasgger import Swagger
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from prometheus_flask_exporter import PrometheusMetrics
from mcp_orchestrator import MCPOrchestrator

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
from logic.chatgpt_integration import build_message_history
from routes.ai_image_generator import init_app as init_image_routes
from routes.chatgpt_routes import init_app as init_chat_routes
from logic.gpt_image_decision import should_generate_image_for_response
from logic.gpt_image_prompting import get_system_prompt_with_image_guidance
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from logic.nyx_enhancements_integration import initialize_nyx_memory_system, enhanced_background_chat_task
from routes.nyx_agent_routes import nyx_agent_bp
from routes.conflict_routes import conflict_bp
from routes.npc_learning_routes import npc_learning_bp

# DB connection helper
from db.connection import (
    get_db_connection, 
    initialize_connection_pool, 
    close_connection_pool,
    get_db_connection_context
)

# Import rate limiting
from middleware.rate_limiting import rate_limit, ip_block_list

# Import validation
from middleware.validation import validate_request

# Additional modules
import asyncpg  # <-- ensure explicit import if not done globally

# NPC creation / learning
from npcs.new_npc_creation import NPCCreationHandler, RunContextWrapper
from npcs.npc_learning_adaptation import NPCLearningManager

# Governance & conflict integration
from nyx.integrate import get_central_governance
from logic.conflict_system.conflict_integration import register_enhanced_integration

import asyncio
from typing import Dict, Any, Optional
from redis import Redis
from celery import Celery

# Adjust as appropriate for your project’s config
from .config.settings import config
from .utils.health_check import HealthCheck

logger = logging.getLogger(__name__)

# Initialize Nyx memory system at import time
# (if you prefer this at the bottom of the file or in create_flask_app, that can also work)
asyncio.run(initialize_nyx_memory_system())  # <-- Make sure eventlet or gevent monkeypatching is done if needed

class ConnectivityManager:
    """Manages connectivity to Redis and Celery services."""
    
    def __init__(self):
        self.redis_client = Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        self.celery_app = Celery(
            'roleplay',
            broker=config.CELERY_BROKER_URL,
            backend=config.CELERY_RESULT_BACKEND
        )
        self.health_check = HealthCheck()
    
    async def check_connectivity(self) -> Dict[str, Any]:
        """Check connectivity to Redis and Celery services."""
        status = {
            'redis': False,
            'celery': False,
            'details': {}
        }
        
        try:
            # Check Redis connectivity
            redis_status = await self._check_redis()
            status['redis'] = redis_status['connected']
            status['details']['redis'] = redis_status
            
            # Check Celery connectivity
            celery_status = await self._check_celery()
            status['celery'] = celery_status['connected']
            status['details']['celery'] = celery_status
            
            # Update health check status
            await self.health_check.update_status('connectivity', status)
            
            return status
        except Exception as e:
            logger.error(f"Error checking connectivity: {e}", exc_info=True)
            return {
                'redis': False,
                'celery': False,
                'details': {'error': str(e)}
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            self.redis_client.ping()
            test_key = 'connectivity_test'
            test_value = 'test'
            self.redis_client.set(test_key, test_value)
            retrieved_value = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)
            
            if retrieved_value == test_value:
                info = self.redis_client.info()
                return {
                    'connected': True,
                    'latency': info.get('instantaneous_ops_per_sec', 'unknown'),
                    'memory_usage': info.get('used_memory_human', 'unknown')
                }
            else:
                return {
                    'connected': False,
                    'error': 'Redis test value mismatch'
                }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    async def _check_celery(self) -> Dict[str, Any]:
        """Check Celery connectivity."""
        try:
            inspector = self.celery_app.control.inspect()
            active_workers = inspector.active()
            
            if active_workers:
                return {
                    'connected': True,
                    'active_workers': len(active_workers),
                    'worker_status': 'active'
                }
            else:
                return {
                    'connected': False,
                    'error': 'No active Celery workers found'
                }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get status of Celery queues."""
        try:
            inspector = self.celery_app.control.inspect()
            reserved = inspector.reserved()
            scheduled = inspector.scheduled()
            return {
                'reserved_tasks': len(reserved) if reserved else 0,
                'scheduled_tasks': len(scheduled) if scheduled else 0,
                'queue_details': {
                    'reserved': reserved,
                    'scheduled': scheduled
                }
            }
        except Exception as e:
            logger.error(f"Error getting queue status: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def monitor_connectivity(self, interval: int = 60):
        """Monitor connectivity status periodically."""
        while True:
            status = await self.check_connectivity()
            if not status['redis'] or not status['celery']:
                logger.warning(f"Connectivity issues detected: {status['details']}")
                # Handle or notify as appropriate
            await asyncio.sleep(interval)

# Global connectivity manager
connectivity_manager = ConnectivityManager()

###############################################################################
# BACKGROUND TASKS
###############################################################################

async def background_chat_task(conversation_id, user_input, universal_update=None):
    """
    Background task for processing chat messages using Nyx agent with OpenAI integration.
    """
    try:
        # Acquire user_id from DB
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT user_id FROM conversations WHERE id = $1", 
                    conversation_id
                )
                if not row:
                    logger.error(f"No conversation found with id {conversation_id}")
                    return
                user_id = row['user_id']
        
        # Get aggregator context
        from logic.aggregator import get_aggregated_roleplay_context
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
        
        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
            "player_name": aggregator_data.get("playerName", "Chase"),
            "npc_present": aggregator_data.get("npcsPresent", []),
            "aggregator_data": aggregator_data
        }
        
        # If universal updates are provided, add them to context
        if universal_update:
            context["universal_update"] = universal_update
            async def apply_updates():
                from logic.universal_updater import apply_universal_updates_async
                dsn = os.getenv("DB_DSN")
                async with asyncpg.create_pool(dsn=dsn) as pool:
                    async with pool.acquire() as conn:
                        return await apply_universal_updates_async(
                            user_id,
                            conversation_id,
                            universal_update,
                            conn
                        )
            await apply_updates()
            
            # Refresh aggregator data post-update
            aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
            context["aggregator_data"] = aggregator_data
        
        # Process the user_input with OpenAI-enhanced Nyx agent
        from nyx.nyx_agent_sdk import process_user_input_with_openai
        response = await process_user_input_with_openai(user_id, conversation_id, user_input, context)
        
        if not response.get("success", False):
            emit('error', {'error': response.get("error", "Unknown error")}, room=conversation_id)
            return
            
        # Extract the message content
        message_content = response.get("message", "")
        if not message_content and "function_args" in response:
            # Extract from function arguments if available
            message_content = response["function_args"].get("narrative", "")
        
        # Store the Nyx response in DB
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                    (conversation_id, "Nyx", message_content)
                )
                conn.commit()
        conn.close()
        
        # Check if we should generate an image
        should_generate = response.get("generate_image", False)
        if "function_args" in response and "image_generation" in response["function_args"]:
            # Also check image_generation settings in function args
            img_settings = response["function_args"]["image_generation"]
            should_generate = should_generate or img_settings.get("generate", False)
        
        # Generate image if needed
        if should_generate:
            try:
                img_data = {
                    "narrative": message_content,
                    "image_generation": {
                        "generate": True,
                        "priority": "medium",
                        "focus": "balanced",
                        "framing": "medium_shot",
                        "reason": "Narrative moment"
                    }
                }
                if "function_args" in response and "image_generation" in response["function_args"]:
                    # Use the image generation settings from the function args if available
                    img_data["image_generation"].update(response["function_args"]["image_generation"])
                
                res = await generate_roleplay_image_from_gpt(img_data, user_id, conversation_id)
                if res and "image_urls" in res and res["image_urls"]:
                    emit('image', {
                        'image_url': res["image_urls"][0],
                        'prompt_used': res.get('prompt_used', ''),
                        'reason': img_data["image_generation"].get("reason", "Narrative moment")
                    }, room=conversation_id)
            except Exception as e:
                logger.error(f"Error generating image: {e}", exc_info=True)
        
        # Stream the text tokens
        for i in range(0, len(message_content), 3):
            token = message_content[i:i+3]
            emit('new_token', {'token': token}, room=conversation_id)
            socketio.sleep(0.05)
        
        emit('done', {'full_text': message_content}, room=conversation_id)
        
    except Exception as e:
        logger.error(f"Error in background_chat_task: {str(e)}", exc_info=True)
        emit('error', {'error': str(e)}, room=conversation_id)
###############################################################################
# FLASK APP CREATION
###############################################################################

def create_flask_app():
    """Create and configure a Flask application."""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    from config import get_config
    security_config = get_config("security")
    
    # Basic app config
    app.config['SECRET_KEY'] = security_config["secret_key"]
    app.config['SESSION_COOKIE_SECURE'] = security_config["session_cookie_secure"]
    app.config['SESSION_COOKIE_HTTPONLY'] = security_config["session_cookie_httponly"]
    app.config['SESSION_COOKIE_SAMESITE'] = security_config["session_cookie_samesite"]
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=security_config["permanent_session_lifetime"])
    
    # Initialize DB pool
    if not initialize_connection_pool():
        logger.error("Failed to initialize database connection pool")
    atexit.register(close_connection_pool)
    
    # Swagger
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
        "specs_route": "/docs"
    }
    template = {
        "swagger": "2.0",
        "info": {
            "title": "NPC Roleplay API",
            "description": "API for managing NPCs, memories, and interactions",
            "version": "1.0.0",
            "contact": {"email": "support@example.com"}
        },
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "JWT token in format: Bearer <token>"
            }
        },
        "security": [{"Bearer": []}]
    }
    Swagger(app, config=swagger_config, template=template)
    
    # Security
    Talisman(app, content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
        'style-src': ["'self'", "'unsafe-inline'"],
    })
    csrf = CSRFProtect(app)
    
    # Metrics
    metrics = PrometheusMetrics(app)
    metrics.info('app_info', 'Application info', version='1.0.0')
    
    # Register blueprints
    app.register_blueprint(new_game_bp, url_prefix='/new_game')
    app.register_blueprint(player_input_bp, url_prefix='/player_input')
    app.register_blueprint(player_input_root_bp)
    app.register_blueprint(settings_bp, url_prefix='/settings')
    app.register_blueprint(knowledge_bp, url_prefix='/knowledge')
    app.register_blueprint(story_bp, url_prefix='/story')
    app.register_blueprint(memory_bp, url_prefix='/memory')
    app.register_blueprint(rule_enforcement_bp, url_prefix='/rules')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(debug_bp, url_prefix='/debug')
    app.register_blueprint(universal_bp, url_prefix='/universal')
    app.register_blueprint(multiuser_bp, url_prefix='/multiuser')
    app.register_blueprint(nyx_agent_bp, url_prefix='/nyx')
    app.register_blueprint(conflict_bp, url_prefix='/conflict')
    app.register_blueprint(npc_learning_bp, url_prefix='/npc-learning')
    
    init_image_routes(app)
    init_chat_routes(app)


    async def initialize_openai_integration():
        """Initialize the OpenAI integration system."""
        try:
            from nyx.eternal.openai_integration import initialize
            from nyx.nyx_agent_sdk import process_user_input
            
            # Initialize the OpenAI integration with the original processor
            initialize(
                api_key=os.environ.get("OPENAI_API_KEY"),
                original_processor=process_user_input
            )
            
            logger.info("OpenAI integration system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI integration: {e}", exc_info=True)
            return False
    
    @app.before_first_request
async def initialize_systems():
    """Initialize all required systems on application startup."""
    try:
        from db.schema_and_seed import create_all_tables, seed_initial_data
        from db.schema_migrations import ensure_schema_version
        
        ensure_schema_version()
        logger.info("Database migrations completed successfully")
        
        create_all_tables()
        seed_initial_data()
        logger.info("Database tables initialized successfully")
        
        await initialize_nyx_memory_system()
        logger.info("Nyx memory system initialized successfully")
        
        # Add this line to initialize OpenAI integration
        await initialize_openai_integration()
        logger.info("OpenAI integration initialized successfully")
        
        # Add these lines to initialize the MCP orchestrator
        from mcp_orchestrator import MCPOrchestrator
        app.mcp_orchestrator = MCPOrchestrator()
        await app.mcp_orchestrator.initialize()
        logger.info("MCP orchestrator initialized successfully")

        # Initialize a global NyxBrain instance
        from nyx.core.brain.base import NyxBrain
        system_user_id = 0  # Use a system user ID
        system_conversation_id = 0  # Use a system conversation ID
        app.nyx_brain = await NyxBrain.get_instance(system_user_id, system_conversation_id)
        logger.info("Global NyxBrain instance initialized successfully")
        
        await register_conflict_system()
        logger.info("Conflict system initialized successfully")
        
        await NPCLearningManager.initialize_system()
        logger.info("NPC learning system initialized successfully")
        
        from logic.universal_updater import initialize_universal_updater
        await initialize_universal_updater()
        logger.info("Universal updater initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing systems: {str(e)}", exc_info=True)
    
    ###########################################################################
    # ROUTES
    ###########################################################################
    
    @app.route("/chat")
    def chat_page():
        if "user_id" not in session:
            return redirect("/login_page")
        return render_template("chat.html")
    
    @app.route("/start_chat", methods=["POST"])
    def start_chat():
        """Enqueues a background chat task using SocketIO for the user input."""
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
    
        data = request.get_json()
        user_input = data.get("user_input")
        conversation_id = data.get("conversation_id")
        universal_update = data.get("universal_update", {})
        
        # Store user message
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
            logger.error(f"Database error: {str(e)}", exc_info=True)
            return jsonify({"error": "Database error"}), 500
        
        # Kick off background chat
        socketio.start_background_task(background_chat_task, conversation_id, user_input, universal_update)
        return jsonify({"status": "success", "message": "Chat started"})
    
    @app.route("/login_page", methods=["GET"])
    def login_page():
        return render_template("login.html")
    
    @app.route("/login", methods=["POST"])
    @rate_limit(limit=5, period=60)
    @validate_request({
        'username': {'type': 'string', 'pattern': 'username', 'max_length': 30, 'required': True},
        'password': {'type': 'string', 'max_length': 100, 'required': True}
    })
    def login():
        data = request.sanitized_data
        username = data["username"]
        password = data["password"]
        
        try:
            conn = get_db_connection()
            user_id, hashed_password = None, None
            with conn.cursor() as cur:
                cur.execute("SELECT id, password_hash FROM users WHERE username=%s", (username,))
                row = cur.fetchone()
                if row:
                    user_id, hashed_password = row
            conn.close()
            
            if not user_id:
                # Fake check to mitigate timing attacks
                bcrypt.checkpw(password.encode('utf-8'), b'$2b$12$' + b'x' * 53)
                return jsonify({"error": "Invalid username or password"}), 401
            
            if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                return jsonify({"error": "Invalid username or password"}), 401
            
            session["user_id"] = user_id
            session.permanent = True
            logger.info(f"User {username} (ID: {user_id}) logged in successfully")
            return jsonify({"message": "Logged in", "user_id": user_id})
        
        except Exception as e:
            logger.error(f"Login error: {str(e)}", exc_info=True)
            return jsonify({"error": "Server error"}), 500
    
    @app.route("/whoami", methods=["GET"])
    def whoami():
        user_id = session.get("user_id")
        if user_id:
            return jsonify({"logged_in": True, "user_id": user_id}), 200
        return jsonify({"logged_in": False}), 200

    @app.route("/openai_chat", methods=["POST"])
    @rate_limit(limit=10, period=60)
    def openai_chat():
        """Enqueues a background chat task using OpenAI integration for the user input."""
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        
        data = request.get_json()
        user_input = data.get("user_input")
        conversation_id = data.get("conversation_id")
        universal_update = data.get("universal_update", {})
        use_standalone = data.get("use_standalone", False)
        
        if not user_input or not conversation_id:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Store user message
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
            logger.error(f"Database error: {str(e)}", exc_info=True)
            return jsonify({"error": "Database error"}), 500
        
        # Kick off background chat using OpenAI integration
        async def openai_chat_task():
            try:
                # Get aggregator data
                from logic.aggregator import get_aggregated_roleplay_context
                aggregator_data = get_aggregated_roleplay_context(session.get("user_id"), conversation_id, "Chase")
                
                context = {
                    "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
                    "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
                    "player_name": aggregator_data.get("playerName", "Chase"),
                    "npc_present": aggregator_data.get("npcsPresent", []),
                    "aggregator_data": aggregator_data,
                    "universal_update": universal_update if universal_update else None
                }
                
                # Apply universal update if provided
                if universal_update:
                    from logic.universal_updater import apply_universal_updates
                    apply_universal_updates(
                        session.get("user_id"),
                        conversation_id,
                        universal_update
                    )
                    # Refresh context
                    aggregator_data = get_aggregated_roleplay_context(session.get("user_id"), conversation_id, "Chase")
                    context["aggregator_data"] = aggregator_data
                
                # Process with OpenAI integration
                from nyx.nyx_agent_sdk import process_user_input_with_openai, process_user_input_standalone
                
                if use_standalone:
                    # Use standalone OpenAI processing
                    response = await process_user_input_standalone(
                        session.get("user_id"), 
                        conversation_id, 
                        user_input,
                        context
                    )
                else:
                    # Use enhanced OpenAI processing
                    response = await process_user_input_with_openai(
                        session.get("user_id"), 
                        conversation_id, 
                        user_input,
                        context
                    )
                
                if not response.get("success", False):
                    emit('error', {'error': response.get("error", "Unknown error")}, room=conversation_id)
                    return
                    
                # Extract the message content
                message_content = response.get("message", "")
                if not message_content and "function_args" in response:
                    # Extract from function arguments if available
                    message_content = response["function_args"].get("narrative", "")
                
                # Store the response
                conn = get_db_connection()
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                            (conversation_id, "Nyx", message_content)
                        )
                        conn.commit()
                conn.close()
                
                # Check if we should generate an image
                should_generate = response.get("generate_image", False)
                if "function_args" in response and "image_generation" in response["function_args"]:
                    # Also check image_generation settings in function args
                    img_settings = response["function_args"]["image_generation"]
                    should_generate = should_generate or img_settings.get("generate", False)
                
                # Generate image if needed
                if should_generate:
                    try:
                        img_data = {
                            "narrative": message_content,
                            "image_generation": {
                                "generate": True,
                                "priority": "medium",
                                "focus": "balanced",
                                "framing": "medium_shot",
                                "reason": "Narrative moment"
                            }
                        }
                        if "function_args" in response and "image_generation" in response["function_args"]:
                            # Use the image generation settings from the function args if available
                            img_data["image_generation"].update(response["function_args"]["image_generation"])
                        
                        res = await generate_roleplay_image_from_gpt(img_data, session.get("user_id"), conversation_id)
                        if res and "image_urls" in res and res["image_urls"]:
                            emit('image', {
                                'image_url': res["image_urls"][0],
                                'prompt_used': res.get('prompt_used', ''),
                                'reason': img_data["image_generation"].get("reason", "Narrative moment")
                            }, room=conversation_id)
                    except Exception as e:
                        logger.error(f"Error generating image: {e}", exc_info=True)
                
                # Stream the text tokens
                for i in range(0, len(message_content), 3):
                    token = message_content[i:i+3]
                    emit('new_token', {'token': token}, room=conversation_id)
                    socketio.sleep(0.05)
                
                emit('done', {'full_text': message_content}, room=conversation_id)
                
            except Exception as e:
                logger.error(f"Error in openai_chat_task: {str(e)}", exc_info=True)
                emit('error', {'error': str(e)}, room=conversation_id)
        
        socketio.start_background_task(openai_chat_task)
        return jsonify({"status": "success", "message": "OpenAI chat started"})
    
    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return jsonify({"message": "Logged out"}), 200
    
    @app.route("/register", methods=["POST"])
    @rate_limit(limit=3, period=300)
    @validate_request({
        'username': {'type': 'string', 'pattern': 'username', 'max_length': 30, 'required': True},
        'password': {'type': 'string', 'min_length': 8, 'max_length': 100, 'required': True},
        'email':    {'type': 'string', 'pattern': 'email', 'max_length': 100, 'required': True}
    })
    def register():
        data = request.sanitized_data
        username = data["username"]
        password = data["password"]
        email = data["email"]
        
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            conn = get_db_connection()
            user_id = None
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM users WHERE username=%s", (username,))
                    if cur.fetchone():
                        return jsonify({"error": "Username already exists"}), 409
                    
                    cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                    if cur.fetchone():
                        return jsonify({"error": "Email already exists"}), 409
                    
                    cur.execute(
                        """INSERT INTO users (username, password_hash, email, created_at)
                           VALUES (%s, %s, %s, NOW()) RETURNING id""",
                        (username, password_hash, email)
                    )
                    user_id = cur.fetchone()[0]
                    conn.commit()
            conn.close()
            
            logger.info(f"New user registered: {username} (ID: {user_id})")
            session["user_id"] = user_id
            session.permanent = True
            
            return jsonify({"message": "User registered successfully", "user_id": user_id})
        except Exception as e:
            logger.error(f"Registration error: {str(e)}", exc_info=True)
            return jsonify({"error": "Server error"}), 500
    
    @app.route("/start_new_game", methods=["POST"])
    @validate_request({'user_id': {'type': 'integer','required': True}})
    async def start_new_game():
        """Start a new game with Nyx integration."""
        try:
            user_id = session.get("user_id")
            if not user_id:
                return jsonify({"error": "Not authenticated"}), 401
            
            # Create conversation
            conversation_id = create_conversation_sync(user_id)  # <-- presumably a local helper
            await initialize_nyx_memory_system(user_id, conversation_id)
            
            from new_game_agent import register_with_governance, NewGameAgent
            await register_with_governance(user_id, conversation_id)
            
            agent = NewGameAgent()
            result = await agent.process_new_game(user_id, {"conversation_id": conversation_id})
            
            if not result.get("success"):
                return jsonify({
                    "error": result.get("error", "Failed to create new game"),
                    "conversation_id": conversation_id
                }), 500
            
            async_conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                await async_conn.execute(
                    """UPDATE conversations 
                       SET status='ready', conversation_name=$1
                       WHERE id=$2 AND user_id=$3""",
                    result.get("game_name","New Game"),
                    conversation_id,
                    user_id
                )
            finally:
                await async_conn.close()
            
            return jsonify({
                "success": True,
                "conversation_id": conversation_id,
                "game_name": result.get("game_name","New Game")
            })
            
        except Exception as e:
            logger.error(f"Error starting new game: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @app.before_first_request
    async def register_conflict_system():
        """Register the conflict system with Nyx on startup."""
        try:
            user_id = 1
            conversation_id = 1
            res = await register_enhanced_integration(user_id, conversation_id)
            if res["success"]:
                app.logger.info("Conflict system successfully registered with Nyx governance")
            else:
                app.logger.error(f"Failed to register conflict system: {res['message']}")
        except Exception as e:
            app.logger.error(f"Error during conflict system registration: {str(e)}", exc_info=True)
    
    ###########################################################################
    # HEALTH ENDPOINTS
    ###########################################################################
    
    @app.route("/health", methods=["GET"])
    def health_check():
        """
        Basic health check endpoint
        """
        return jsonify({"status": "healthy","timestamp": time.time()})
    
    @app.route("/readiness", methods=["GET"])
    def readiness_check():
        """
        Readiness check endpoint for checking dependencies
        """
        status = {"status": "ready", "timestamp": time.time(), "checks": {}}
        is_ready = True
        
        # DB check
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.close()
            status["checks"]["database"] = "connected"
        except Exception as e:
            status["checks"]["database"] = f"error: {str(e)}"
            is_ready = False
        
        # Redis/Celery check if desired
        if os.getenv("REDIS_URL"):
            try:
                r = Redis.from_url(os.getenv("REDIS_URL"))
                r.ping()
                status["checks"]["redis"] = "connected"
            except Exception as e:
                status["checks"]["redis"] = f"error: {str(e)}"
                is_ready = False
        
        if not is_ready:
            status["status"] = "not ready"
            return jsonify(status), 503
        return jsonify(status)
    
    return app

###############################################################################
# SOCKET.IO 
###############################################################################

socketio = None

def create_socketio(app):
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', 
                        logger=True, engineio_logger=True, ping_timeout=60)
    
    @socketio.on('connect')
    def handle_connect():
        logger.info("SocketIO: Client connected")
        emit('response', {'data': 'Connected to SocketIO server!'})
        
        # Schedule learning cycle if not yet scheduled
        if not hasattr(app, '_npc_learning_scheduler_running'):
            app._npc_learning_scheduler_running = True
            socketio.start_background_task(schedule_npc_learning_cycles)
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info("SocketIO: Client disconnected")
    
    @socketio.on('join')
    def handle_join(data):
        conversation_id = data.get('conversation_id')
        if conversation_id:
            join_room(conversation_id)
            logger.info(f"SocketIO: Client joined room {conversation_id}")
            # Initialize conversation state in background
            socketio.start_background_task(initialize_conversation_state, conversation_id)
            emit('joined', {'room': conversation_id})
        else:
            logger.warning("Client attempted to join room without conversation_id")
            emit('error', {'error': 'Missing conversation_id'})
    
    @socketio.on('message')
    def handle_message(data):
        conversation_id = data.get('conversation_id')
        message = data.get('message')
        user_id = data.get('user_id')
        
        if not conversation_id or not message:
            logger.warning("Received incomplete message data")
            emit('error', {'error': 'Invalid message data'}, room=request.sid)
            return
        
        universal_update = data.get('universal_update')
        
        # Enhanced chat with universal updates
        socketio.start_background_task(
            enhanced_background_chat_task,
            conversation_id=conversation_id,
            user_input=message,
            universal_update=universal_update,
            user_id=user_id
        )
        logger.debug(f"Started background chat task for conversation {conversation_id}")
    
    @socketio.on('storybeat')
    def handle_storybeat(data):
        """
        Similar logic as your custom story route,
        but triggered via SocketIO instead of HTTP
        """
        try:
            user_id = session.get("user_id")
            if not user_id:
                emit('error', {'error': 'Not authenticated'})
                return
            
            user_input = data.get("user_input", "").strip()
            conversation_id = data.get("conversation_id")
            player_name = data.get("player_name", "Chase")
            advance_time = data.get("advance_time", False)
            universal_update = data.get("universal_update", {})
            
            if not user_input or not conversation_id:
                emit('error', {'error': 'Missing required fields'})
                return
            
            join_room(conversation_id)
            emit('processing', {'message': 'Processing your request...'})
            
            def process_message_background():
                conn = None
                try:
                    conn = get_db_connection()
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conversation_id,))
                            row = cur.fetchone()
                            if not row:
                                socketio.emit('error', {'error': f"Conversation {conversation_id} not found"}, room=conversation_id)
                                return
                            if row[0] != user_id:
                                socketio.emit('error', {'error': f"Conversation {conversation_id} not owned by this user"}, room=conversation_id)
                                return
                            
                            # Store user input
                            cur.execute(
                                "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                                (conversation_id, "user", user_input)
                            )
                            conn.commit()
                    
                    # Possibly apply universal updates
                    if universal_update:
                        from logic.universal_updater import apply_universal_updates_async
                        async def apply_updates():
                            dsn = os.getenv("DB_DSN")
                            pool = await asyncpg.create_pool(dsn=dsn)
                            async with pool.acquire() as async_conn:
                                return await apply_universal_updates_async(user_id, conversation_id, universal_update, async_conn)
                        asyncio.run(apply_updates())  # <-- blocking call in background thread
                        
                    # Check unintroduced NPCs
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT COUNT(*) FROM NPCStats
                                WHERE user_id=%s AND conversation_id=%s AND introduced=FALSE
                            """,(user_id, conversation_id))
                            unintroduced_count = cur.fetchone()[0]
                    
                    if unintroduced_count < 2:
                        logger.info("Only %d unintroduced NPC(s) found; generating 3 more.", unintroduced_count)
                        
                        async def spawn_npcs_async():
                            npc_handler = NPCCreationHandler()
                            ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
                            return await npc_handler.spawn_multiple_npcs(ctx, 3)
                        
                        try:
                            new_npc_ids = asyncio.run(spawn_npcs_async())
                            logger.info(f"Generated new NPCs: {new_npc_ids}")
                        except Exception as e:
                            logger.error(f"Error spawning NPCs: {e}", exc_info=True)
                    
                    # Advance time if requested
                    if advance_time:
                        from logic.time_cycle import advance_time_and_update
                        new_year, new_month, new_day, new_phase = advance_time_and_update(user_id, conversation_id, increment=1)
                        logger.info(f"Advanced time to Y{new_year} M{new_month} D{new_day} {new_phase}")
                    
                    # Rebuild aggregator data
                    from logic.aggregator import get_aggregated_roleplay_context
                    aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
                    
                    # Gather extra NPC memory
                    npc_context_summary = ""
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT npc_name, memory, archetype_extras_summary
                                FROM NPCStats
                                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                            """,(user_id, conversation_id))
                            for row in cur.fetchall():
                                nm, mem_json, extras = row
                                mem_list = []
                                if mem_json:
                                    if isinstance(mem_json, list):
                                        mem_list = mem_json
                                    else:
                                        try:
                                            mem_list = json.loads(mem_json)
                                        except Exception as e:
                                            logger.warning(f"Error parsing memory JSON for NPC {nm}: {e}")
                                mem_text = " ".join(str(item) for item in mem_list) if mem_list else ""
                                npc_context_summary += f"{nm}: {mem_text} {extras or ''}\n"
                    
                    from routes.story_routes import build_aggregator_text, gather_rule_knowledge
                    rule_knowledge = gather_rule_knowledge()
                    aggregator_text = build_aggregator_text(aggregator_data, rule_knowledge)
                    if npc_context_summary:
                        aggregator_text += "\nNPC Context:\n" + npc_context_summary
                    
                    # ChatGPT response
                    from logic.chatgpt_integration import get_chatgpt_response
                    response_data = get_chatgpt_response(conversation_id, aggregator_text, user_input)
                    
                    if response_data["type"] == "function_call":
                        ai_response = response_data["function_args"].get("narrative","")
                        
                        if response_data["function_args"]:
                            # Possibly apply universal updates
                            from logic.universal_updater import apply_universal_updates_async
                            async def apply_gpt_updates():
                                dsn = os.getenv("DB_DSN")
                                pool = await asyncpg.create_pool(dsn=dsn)
                                async with pool.acquire() as async_conn:
                                    return await apply_universal_updates_async(
                                        user_id, conversation_id, response_data["function_args"], async_conn
                                    )
                            try:
                                update_result = asyncio.run(apply_gpt_updates())
                                logger.info(f"Applied universal updates from GPT: {update_result}")
                            except Exception as e:
                                logger.error(f"Error applying GPT updates: {e}", exc_info=True)
                            
                        # Possibly generate an image
                        should_gen, reason = should_generate_image_for_response(user_id, conversation_id, response_data["function_args"])
                        if should_gen:
                            logger.info(f"Generating image for scene: {reason}")
                            res = generate_roleplay_image_from_gpt(response_data["function_args"], user_id, conversation_id)
                            if res and "image_urls" in res and res["image_urls"]:
                                socketio.emit('image',{
                                    'image_url':res["image_urls"][0],
                                    'prompt_used':res.get('prompt_used',''),
                                    'reason':reason
                                },room=conversation_id)
                    else:
                        ai_response = response_data.get("response","")
                    
                    # Stream response
                    for i in range(0, len(ai_response),3):
                        token = ai_response[i:i+3]
                        socketio.emit('new_token',{'token':token},room=conversation_id)
                        socketio.sleep(0.05)
                    
                    # Save GPT response
                    try:
                        with conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "INSERT INTO messages (conversation_id,sender,content) VALUES (%s,%s,%s)",
                                    (conversation_id,"Nyx",ai_response)
                                )
                                conn.commit()
                    except Exception as db_error:
                        logger.error(f"DB error storing GPT response: {db_error}", exc_info=True)
                    
                    socketio.emit('done',{'full_text':ai_response},room=conversation_id)
                
                except Exception as e:
                    logger.error(f"Error in background storybeat processing: {e}", exc_info=True)
                    socketio.emit('error',{'error':f"Server error: {e}"},room=conversation_id)
                
                finally:
                    if conn:
                        conn.close()
            
            socketio.start_background_task(process_message_background)
        
        except Exception as e:
            logger.error(f"Unexpected error in storybeat: {e}", exc_info=True)
            emit('error',{'error':f"Unexpected server error: {e}"})
    
    return socketio

###############################################################################
# SCHEDULERS / HELPERS
###############################################################################

def schedule_npc_learning_cycles():
    """
    Example of an eventlet-based recurring job. 
    (If you want reliable scheduling, consider Celery beat or APScheduler.)
    """
    logger.info("Starting NPC learning cycle scheduler")
    try:
        while True:
            socketio.sleep(900)  # every 15 min
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, user_id 
                        FROM conversations
                        WHERE last_active > NOW() - INTERVAL '1 day'
                    """)
                    convs = cur.fetchall()
                conn.close()
                
                for (conv_id, uid) in convs:
                    try:
                        # get NPCs
                        conn = get_db_connection()
                        npc_ids = []
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT npc_id FROM NPCStats
                                WHERE user_id=%s AND conversation_id=%s
                            """,(uid, conv_id))
                            npc_ids = [row[0] for row in cur.fetchall()]
                        conn.close()
                        
                        if npc_ids:
                            async def run_learning():
                                manager = NPCLearningManager(uid, conv_id)
                                await manager.initialize()
                                await manager.run_regular_adaptation_cycle(npc_ids)
                                logger.info(f"Learning cycle for conversation {conv_id}: {len(npc_ids)} NPCs")
                            asyncio.run(run_learning())
                    
                    except Exception as e:
                        logger.error(f"Error in NPC learning cycle for {conv_id}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error in learning scheduler: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"NPC learning scheduler crashed: {e}", exc_info=True)

async def initialize_conversation_state(conversation_id):
    """
    Called in a background task to ensure conversation is 'ready'.
    """
    try:
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            row = await conn.fetchrow("SELECT user_id, status FROM conversations WHERE id=$1", conversation_id)
            if not row:
                logger.error(f"Conversation {conversation_id} not found")
                return
            
            user_id, status = row
            if status != 'ready':
                from new_game_agent import NewGameAgent
                agent = NewGameAgent()
                await agent.process_new_game(user_id, {"conversation_id": conversation_id})
                await conn.execute("""
                    UPDATE conversations
                    SET status='ready'
                    WHERE id=$1
                """, conversation_id)
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Error in initialize_conversation_state: {e}", exc_info=True)

###############################################################################
# MAIN ENTRY
###############################################################################

if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    port = int(os.getenv("PORT", 5000))
    
    app = create_flask_app()
    socketio = create_socketio(app)
    
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
