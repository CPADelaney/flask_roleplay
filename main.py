# main.py

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
from flask_seasurf import SeaSurf
from prometheus_flask_exporter import PrometheusMetrics

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

# Global socketio instance
socketio = None

from logic.chatgpt_integration import get_chatgpt_response
from db.connection import get_db_connection

from nyx.integrate import get_central_governance
from logic.conflict_system.conflict_integration import register_enhanced_integration
from npcs.npc_learning_adaptation import NPCLearningManager

import asyncio
from typing import Dict, Any, Optional
from redis import Redis
from celery import Celery
from .config.settings import config
from .utils.health_check import HealthCheck

asyncio.run(initialize_nyx_memory_system())

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
            logger.error(f"Error checking connectivity: {e}")
            return {
                'redis': False,
                'celery': False,
                'details': {'error': str(e)}
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            # Try to ping Redis
            self.redis_client.ping()
            
            # Try to set and get a test value
            test_key = 'connectivity_test'
            test_value = 'test'
            self.redis_client.set(test_key, test_value)
            retrieved_value = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)
            
            if retrieved_value == test_value:
                return {
                    'connected': True,
                    'latency': self.redis_client.info()['instantaneous_ops_per_sec'],
                    'memory_usage': self.redis_client.info()['used_memory_human']
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
            # Try to inspect Celery workers
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
            logger.error(f"Error getting queue status: {e}")
            return {
                'error': str(e)
            }
    
    async def monitor_connectivity(self, interval: int = 60):
        """Monitor connectivity status periodically."""
        while True:
            status = await self.check_connectivity()
            if not status['redis'] or not status['celery']:
                logger.warning(f"Connectivity issues detected: {status['details']}")
                # Notify administrators or take corrective action
            await asyncio.sleep(interval)

# Create global instance
connectivity_manager = ConnectivityManager()

async def background_chat_task(conversation_id, user_input, universal_update=None):
    """
    Background task for processing chat messages using SDK Nyx agent.
    """
    try:
        # Get user_id
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT user_id FROM conversations WHERE id = $1", 
                    conversation_id
                )
                
                if not row:
                    logging.error(f"No conversation found with id {conversation_id}")
                    return
                    
                user_id = row['user_id']
        
        # Get context data
        from logic.aggregator import get_aggregated_roleplay_context
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")  # Default name
        
        # Build context
        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
            "player_name": aggregator_data.get("playerName", "Chase"),
            "npc_present": aggregator_data.get("npcsPresent", []),
            "aggregator_data": aggregator_data
        }
        
        # Add universal update data
        if universal_update:
            context["universal_update"] = universal_update
            
            # Process updates with async context
            async def apply_updates():
                universal_update["user_id"] = user_id
                universal_update["conversation_id"] = conversation_id
                dsn = os.getenv("DB_DSN")
                
                async with asyncpg.create_pool(dsn=dsn) as pool:
                    async with pool.acquire() as conn:
                        return await apply_universal_updates_async(
                            user_id, 
                            conversation_id, 
                            universal_update,
                            conn
                        )
            
            # Apply updates
            await apply_updates()
            
            # Refresh context after updates
            aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
            context["aggregator_data"] = aggregator_data
        
        # Process via SDK
        from nyx.nyx_agent_sdk import process_user_input
        response = await process_user_input(user_id, conversation_id, user_input, context)
        
        # Store Nyx's response
        from db.connection import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
            (conversation_id, "Nyx", response["message"])
        )
        conn.commit()
        conn.close()
        
        # Check if we should generate an image
        should_generate = response.get("generate_image", False)
        
        if should_generate:
            try:
                # Generate image based on the response
                from routes.ai_image_generator import generate_roleplay_image_from_gpt
                
                image_result = await generate_roleplay_image_from_gpt(
                    {
                        "narrative": response["message"],
                        "image_generation": {
                            "generate": True,
                            "priority": "medium",
                            "focus": "balanced",
                            "framing": "medium_shot",
                            "reason": "Narrative moment"
                        }
                    },
                    user_id,
                    conversation_id
                )
                
                # Emit image to the client via SocketIO
                from flask_socketio import emit
                if image_result and "image_urls" in image_result and image_result["image_urls"]:
                    emit('image', {
                        'image_url': image_result["image_urls"][0],
                        'prompt_used': image_result.get('prompt_used', '')
                    }, room=conversation_id)
            except Exception as e:
                logging.error(f"Error generating image: {e}")
        
        # Emit response to client
        from flask_socketio import emit
        
        # Stream the response token by token
        for i in range(0, len(response["message"]), 3):
            token = response["message"][i:i+3]
            emit('new_token', {'token': token}, room=conversation_id)
            socketio.sleep(0.05)
            
        # Signal completion
        emit('done', {'full_text': response["message"]}, room=conversation_id)
        
    except Exception as e:
        logging.error(f"Error in background_chat_task: {str(e)}", exc_info=True)
        from flask_socketio import emit
        emit('error', {'error': str(e)}, room=conversation_id)

def create_flask_app():
    """Create and configure a Flask application."""
    app = Flask(
        __name__,
        static_folder='static',
        template_folder='templates'
    )
    
    # Load configuration from config module
    from config import get_config
    security_config = get_config("security")
    
    # Configure app
    app.config['SECRET_KEY'] = security_config["secret_key"]
    app.config['SESSION_COOKIE_SECURE'] = security_config["session_cookie_secure"]
    app.config['SESSION_COOKIE_HTTPONLY'] = security_config["session_cookie_httponly"]
    app.config['SESSION_COOKIE_SAMESITE'] = security_config["session_cookie_samesite"]
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=security_config["permanent_session_lifetime"])
    
    # Initialize database connection pool
    if not initialize_connection_pool():
        logging.error("Failed to initialize database connection pool")
    
    # Register function to close database connections on shutdown
    atexit.register(close_connection_pool)
    
    # Initialize Swagger documentation
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,  # all in
                "model_filter": lambda tag: True,  # all in
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
            "contact": {
                "email": "support@example.com"
            }
        },
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "JWT token in format: Bearer <token>"
            }
        },
        "security": [
            {
                "Bearer": []
            }
        ]
    }
    
    Swagger(app, config=swagger_config, template=template)
    
    # Initialize security middleware
    Talisman(app, content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
        'style-src': ["'self'", "'unsafe-inline'"],
    })
    csrf = SeaSurf(app)
    
    # Initialize metrics
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
    
    # Initialize routes for other modules
    init_image_routes(app)
    init_chat_routes(app)
    
    # Initialize Nyx memory system
    @app.before_first_request
    async def initialize_systems():
        """Initialize all required systems on application startup."""
        try:
            # Initialize database tables and run migrations
            from db.schema_and_seed import create_all_tables, seed_initial_data
            from db.schema_migrations import ensure_schema_version
            
            # Run migrations first
            ensure_schema_version()
            logging.info("Database migrations completed successfully")
            
            # Create tables and seed data
            create_all_tables()
            seed_initial_data()
            logging.info("Database tables initialized successfully")
            
            # Initialize database tables for Nyx memory
            await initialize_nyx_memory_system()
            logging.info("Nyx memory system initialized successfully")
            
            # Initialize conflict system
            await register_conflict_system()
            logging.info("Conflict system initialized successfully")
            
            # Initialize NPC learning system
            from npcs.npc_learning_adaptation import NPCLearningManager
            await NPCLearningManager.initialize_system()
            logging.info("NPC learning system initialized successfully")
            
            # Initialize universal updater
            from logic.universal_updater import initialize_universal_updater
            await initialize_universal_updater()
            logging.info("Universal updater initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing systems: {str(e)}", exc_info=True)
            # Don't fail startup, but log the error
    
    # HTTP routes
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
    
        # Store user message in the database
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
            logging.error(f"Database error: {str(e)}")
            return jsonify({"error": "Database error"}), 500
    
        # Start the background chat task (now using enhanced context)
        socketio.start_background_task(background_chat_task, conversation_id, user_input, universal_update)
    
        return jsonify({"status": "success", "message": "Chat started"})
    
    @app.route("/login_page", methods=["GET"])
    def login_page():
        return render_template("login.html")
    
    @app.route("/login", methods=["POST"])
    @rate_limit(limit=5, period=60)  # 5 login attempts per minute
    @validate_request({
        'username': {
            'type': 'string',
            'pattern': 'username',
            'max_length': 30,
            'required': True
        },
        'password': {
            'type': 'string',
            'max_length': 100,
            'required': True
        }
    })
    def login():
        # Get sanitized data from the request
        data = request.sanitized_data
        username = data["username"]
        password = data["password"]
        
        try:
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
                    row = cur.fetchone()
            finally:
                if conn:
                    conn.close()
            
            if not row:
                # Use constant time comparison to prevent timing attacks
                # Still do a fake check even if user doesn't exist
                bcrypt.checkpw(password.encode('utf-8'), b'$2b$12$' + b'x' * 53)
                return jsonify({"error": "Invalid username or password"}), 401
            
            user_id, hashed_password = row
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                return jsonify({"error": "Invalid username or password"}), 401
            
            # Set session with appropriate security
            session["user_id"] = user_id
            session.permanent = True  # Use permanent session with lifetime
            
            # Log successful login
            logging.info(f"User {username} (ID: {user_id}) logged in successfully")
            
            return jsonify({"message": "Logged in", "user_id": user_id})
        except Exception as e:
            logging.error(f"Login error: {str(e)}", exc_info=True)
            return jsonify({"error": "Server error"}), 500
    
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
    @rate_limit(limit=3, period=300)  # 3 register attempts per 5 minutes
    @validate_request({
        'username': {
            'type': 'string',
            'pattern': 'username',
            'max_length': 30,
            'required': True
        },
        'password': {
            'type': 'string',
            'min_length': 8,
            'max_length': 100,
            'required': True
        },
        'email': {
            'type': 'string',
            'pattern': 'email',
            'max_length': 100,
            'required': True
        }
    })
    def register():
        # Get sanitized data
        data = request.sanitized_data
        username = data["username"]
        password = data["password"]
        email = data["email"]
        
        try:
            # Generate password hash
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    # Check if username already exists
                    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                    if cur.fetchone():
                        return jsonify({"error": "Username already exists"}), 409
                    
                    # Check if email already exists
                    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                    if cur.fetchone():
                        return jsonify({"error": "Email already exists"}), 409
                    
                    # Insert new user
                    cur.execute(
                        "INSERT INTO users (username, password_hash, email, created_at) VALUES (%s, %s, %s, NOW()) RETURNING id",
                        (username, password_hash, email)
                    )
                    user_id = cur.fetchone()[0]
                    conn.commit()
            finally:
                if conn:
                    conn.close()
            
            # Log successful registration
            logging.info(f"New user registered: {username} (ID: {user_id})")
            
            # Set session
            session["user_id"] = user_id
            session.permanent = True
            
            return jsonify({"message": "User registered successfully", "user_id": user_id})
        except Exception as e:
            logging.error(f"Registration error: {str(e)}", exc_info=True)
            return jsonify({"error": "Server error"}), 500
    
    @app.route("/start_new_game", methods=["POST"])
    @validate_request({
        'user_id': {
            'type': 'integer',
            'required': True
        }
    })
    async def start_new_game():
        """Start a new game with proper Nyx integration."""
        try:
            user_id = session.get("user_id")
            if not user_id:
                return jsonify({"error": "Not authenticated"}), 401

            # Create new conversation
            conversation_id = create_conversation_sync(user_id)
            
            # Initialize Nyx memory system for this conversation
            await initialize_nyx_memory_system(user_id, conversation_id)
            
            # Register the new game agent with Nyx governance
            from new_game_agent import register_with_governance
            await register_with_governance(user_id, conversation_id)
            
            # Create and initialize the new game agent
            from new_game_agent import NewGameAgent
            agent = NewGameAgent()
            
            # Process the new game creation
            result = await agent.process_new_game(user_id, {"conversation_id": conversation_id})
            
            if not result.get("success"):
                return jsonify({
                    "error": result.get("error", "Failed to create new game"),
                    "conversation_id": conversation_id
                }), 500
                
            # Update conversation status
            conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                await conn.execute("""
                    UPDATE conversations 
                    SET status = 'ready', 
                        conversation_name = $1
                    WHERE id = $2 AND user_id = $3
                """, result.get("game_name", "New Game"), conversation_id, user_id)
            finally:
                await conn.close()
            
            return jsonify({
                "success": True,
                "conversation_id": conversation_id,
                "game_name": result.get("game_name", "New Game")
            })
            
        except Exception as e:
            logging.error(f"Error starting new game: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @app.before_first_request
    async def register_conflict_system():
        """Register the conflict system with Nyx on application startup."""
        try:
            # Register for default user (adapt as needed for your multi-user setup)
            user_id = 1
            conversation_id = 1
            
            # Register the enhanced conflict system
            result = await register_enhanced_integration(user_id, conversation_id)
            
            if result["success"]:
                app.logger.info("Conflict system successfully registered with Nyx governance")
            else:
                app.logger.error(f"Failed to register conflict system: {result['message']}")
                
        except Exception as e:
            app.logger.error(f"Error during conflict system registration: {str(e)}", exc_info=True)
    
    # Health and monitoring endpoints
    @app.route("/health", methods=["GET"])
    def health_check():
        """
        Basic health check endpoint for monitoring and load balancers.
        Returns 200 OK if the application is running.
        """
        return jsonify({
            "status": "healthy",
            "timestamp": time.time()
        })
    
    @app.route("/readiness", methods=["GET"])
    def readiness_check():
        """
        Readiness check endpoint that tests critical dependencies.
        Returns 200 OK if the application is ready to accept traffic,
        or 503 Service Unavailable if any dependencies are not available.
        """
        status = {"status": "ready", "timestamp": time.time(), "checks": {}}
        is_ready = True
        
        # Check database connectivity
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
        
        # Check Redis/Celery connectivity if applicable
        try:
            # This is a placeholder for Redis/Celery connectivity check
            # Replace with actual code if you have Redis/Celery
            import os
            if os.getenv("REDIS_URL"):
                from redis import Redis
                r = Redis.from_url(os.getenv("REDIS_URL"))
                r.ping()
                status["checks"]["redis"] = "connected"
        except ImportError:
            status["checks"]["redis"] = "not configured"
        except Exception as e:
            status["checks"]["redis"] = f"error: {str(e)}"
            is_ready = False
        
        # Update overall status
        if not is_ready:
            status["status"] = "not ready"
            return jsonify(status), 503
        
        return jsonify(status)
    
    return app

def create_socketio(app):
    """
    Create and configure the SocketIO instance
    """
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', 
                       logger=True, engineio_logger=True, ping_timeout=60)
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection and initialize game state."""
        logging.info("SocketIO: Client connected")
        emit('response', {'data': 'Connected to SocketIO server!'})
        
        # Schedule regular NPC learning cycle
        if not hasattr(app, '_npc_learning_scheduler_running'):
            app._npc_learning_scheduler_running = True
            socketio.start_background_task(schedule_npc_learning_cycles)
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logging.info("SocketIO: Client disconnected")
    
    @socketio.on('join')
    def handle_join(data):
        """Handle client joining a conversation room."""
        conversation_id = data.get('conversation_id')
        if conversation_id:
            join_room(conversation_id)
            logging.info(f"SocketIO: Client joined room {conversation_id}")
            
            # Initialize conversation state if needed
            socketio.start_background_task(initialize_conversation_state, conversation_id)
            
            emit('joined', {'room': conversation_id})
        else:
            logging.warning("Client attempted to join room without conversation_id")
            emit('error', {'error': 'Missing conversation_id'})
    
    @socketio.on('message')
    def handle_message(data):
        """Handle incoming chat messages and process them with Nyx."""
        conversation_id = data.get('conversation_id')
        message = data.get('message')
        user_id = data.get('user_id')  # Optional, may be fetched from DB based on conversation_id
        
        # Simple validation
        if not conversation_id or not message:
            logging.warning("Received incomplete message data")
            emit('error', {'error': 'Invalid message data'}, room=request.sid)
            return

        try:
            # Process universal updates if present
            universal_update = None
            if 'universal_update' in data:
                universal_update = data['universal_update']
                logging.info(f"Universal update received for conversation {conversation_id}")
            
            # Background processing using the enhanced Nyx task
            socketio.start_background_task(
                enhanced_background_chat_task,
                conversation_id=conversation_id,
                user_input=message,
                universal_update=universal_update,
                user_id=user_id
            )
            
            logging.debug(f"Started background chat task for conversation {conversation_id}")
            
        except Exception as e:
            logging.error(f"Error handling message: {str(e)}", exc_info=True)
            emit('error', {'error': 'Server error processing message'}, room=request.sid)
    
    # Move the storybeat handler here
    @socketio.on('storybeat')
    def handle_storybeat(data):
        """
        Socket.IO event handler for advanced storybeat processing.
        Incorporates the functionality from next_storybeat in story_routes.py
        but uses Socket.IO for real-time streaming of responses.
        """
        try:
            # Get user session data
            user_id = session.get("user_id")
            if not user_id:
                emit('error', {'error': 'Not authenticated'})
                return
    
            # Extract data from the message
            user_input = data.get("user_input", "").strip()
            conversation_id = data.get("conversation_id")
            player_name = data.get("player_name", "Chase")
            advance_time = data.get("advance_time", False)
            universal_update = data.get("universal_update", {})
    
            if not user_input or not conversation_id:
                emit('error', {'error': 'Missing required fields'})
                return
    
            # Join the room for this conversation
            join_room(conversation_id)
            
            # Emit an acknowledgment that processing has begun
            emit('processing', {'message': 'Processing your request...'})
            
            # Function to do the heavy processing in background
            def process_message_background():
                conn = None
                try:
                    # Create database connection
                    conn = get_db_connection()
                    
                    # Validate conversation ownership
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
                            
                            # Store user message
                            cur.execute(
                                "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                                (conversation_id, "user", user_input)
                            )
                            conn.commit()
                    
                    # Process universal updates if provided
                    if universal_update:
                        def apply_universal_updates_thread():
                            import asyncio
                            import os
                            import asyncpg
                            from logic.universal_updater import apply_universal_updates_async
                            
                            universal_update["user_id"] = user_id
                            universal_update["conversation_id"] = conversation_id
                            
                            async def run_updates():
                                dsn = os.getenv("DB_DSN")
                                async_conn = await asyncpg.connect(dsn=dsn)
                                try:
                                    return await apply_universal_updates_async(
                                        user_id, conversation_id, universal_update, async_conn
                                    )
                                finally:
                                    await async_conn.close()
                            
                            result = asyncio.run(run_updates())
                            logging.info(f"Universal update result: {result}")
                        
                        # Apply updates before generating response
                        apply_universal_updates_thread()
                    
                    # Check and spawn NPCs if needed
                    from logic.aggregator import get_aggregated_roleplay_context
                    from logic.npc_creation import spawn_multiple_npcs
                    import asyncio
                    
                    # Check NPC count
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT COUNT(*) FROM NPCStats
                                WHERE user_id=%s AND conversation_id=%s AND introduced=FALSE
                            """, (user_id, conversation_id))
                            count = cur.fetchone()[0]
                    
                    # If too few NPCs, spawn more
                    if count < 2:
                        logging.info("Only %d unintroduced NPC(s) found; generating 3 more.", count)
                        # Get aggregator data for environment context
                        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
                        env_desc = aggregator_data.get("currentRoleplay", {}).get("EnvironmentDesc", 
                                                                               "A default environment description.")
                        calendar = aggregator_data.get("calendar", {})
                        day_names = calendar.get("days", 
                                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                        
                        # Spawn NPCs using asyncio
                        async def spawn_npcs_async():
                            return await spawn_multiple_npcs(user_id, conversation_id, env_desc, day_names, count=3)
                        
                        npc_ids = asyncio.run(spawn_npcs_async())
                        logging.info(f"Generated new NPCs: {npc_ids}")
                    
                    # Advance time if requested
                    from logic.time_cycle import advance_time_and_update
                    
                    if advance_time:
                        new_year, new_month, new_day, new_phase = advance_time_and_update(
                            user_id, conversation_id, increment=1
                        )
                        logging.info(f"Advanced time to Y{new_year} M{new_month} D{new_day} {new_phase}")
                    
                    # Get the latest aggregated context (after time advance and NPC creation)
                    aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, player_name)
                    
                    # Append additional NPC context
                    npc_context_summary = ""
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT npc_name, memory, archetype_extras_summary
                                FROM NPCStats
                                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                            """, (user_id, conversation_id))
                            
                            for row in cur.fetchall():
                                npc_name, memory_json, extras_summary = row
                                mem_text = ""
                                if memory_json:
                                    # Handle the case where memory_json might already be a list
                                    if isinstance(memory_json, list):
                                        mem_list = memory_json
                                    else:
                                        try:
                                            mem_list = json.loads(memory_json)
                                        except Exception as e:
                                            logging.warning(f"Error parsing memory JSON for NPC {npc_name}: {e}")
                                            mem_list = []
                                    
                                    # Convert the memory list to a string
                                    if isinstance(mem_list, list):
                                        mem_text = " ".join(str(item) for item in mem_list)
                                    else:
                                        mem_text = str(mem_list)
                                
                                extra_text = extras_summary if extras_summary else ""
                                npc_context_summary += f"{npc_name}: {mem_text} {extra_text}\n"
                    
                    # Build the complete context
                    from routes.story_routes import build_aggregator_text, gather_rule_knowledge
                    rule_knowledge = gather_rule_knowledge()
                    aggregator_text = build_aggregator_text(aggregator_data, rule_knowledge)
                    
                    if npc_context_summary:
                        aggregator_text += "\nNPC Context:\n" + npc_context_summary
                    
                    # Get response from GPT
                    from logic.chatgpt_integration import get_chatgpt_response
                    response_data = get_chatgpt_response(conversation_id, aggregator_text, user_input)
                    
                    # Extract narrative
                    if response_data["type"] == "function_call":
                        ai_response = response_data["function_args"].get("narrative", "")
                        
                        # Process universal updates from GPT response
                        if response_data["function_args"]:
                            try:
                                import asyncio
                                import asyncpg
                                from logic.universal_updater import apply_universal_updates_async
                                
                                async def apply_updates():
                                    dsn = os.getenv("DB_DSN")
                                    conn = await asyncpg.connect(dsn=dsn, statement_cache_size=0)
                                    try:
                                        return await apply_universal_updates_async(
                                            user_id, 
                                            conversation_id, 
                                            response_data["function_args"],
                                            conn
                                        )
                                    finally:
                                        await conn.close()
                                
                                update_result = asyncio.run(apply_updates())
                                logging.info(f"Applied universal updates from GPT: {update_result}")
                            except Exception as update_error:
                                logging.error(f"Error applying universal updates from GPT: {str(update_error)}")
                                
                        should_generate, reason = should_generate_image_for_response(
                            user_id, 
                            conversation_id, 
                            response_data["function_args"]
                        )
                        
                        # Process image generation if needed
                        if should_generate:
                            logging.info(f"Generating image for scene: {reason}")
                            image_result = generate_roleplay_image_from_gpt(
                                response_data["function_args"], 
                                user_id, 
                                conversation_id
                            )
                            
                            # Emit image to the client
                            if image_result and "image_urls" in image_result and image_result["image_urls"]:
                                socketio.emit('image', {
                                    'image_url': image_result["image_urls"][0],
                                    'prompt_used': image_result.get('prompt_used', ''),
                                    'reason': reason
                                }, room=conversation_id)
                                logging.info(f"Image emitted to client: {image_result['image_urls'][0]}")
                                
                    else:
                        ai_response = response_data.get("response", "")
                    
                    # Stream the response token by token
                    for i in range(0, len(ai_response), 3):
                        token = ai_response[i:i+3]
                        socketio.emit('new_token', {'token': token}, room=conversation_id)
                        socketio.sleep(0.05)
                    
                    # Store the complete GPT response in the database
                    try:
                        with conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                                    (conversation_id, "Nyx", ai_response)
                                )
                                conn.commit()
                    except Exception as db_error:
                        logging.error(f"Database error storing GPT response: {str(db_error)}")
                    
                    # Emit the final 'done' event with the full text
                    socketio.emit('done', {'full_text': ai_response}, room=conversation_id)
                    
                except Exception as e:
                    logging.error(f"Error in background processing: {str(e)}", exc_info=True)
                    socketio.emit('error', {'error': f"Server error: {str(e)}"}, room=conversation_id)
                finally:
                    if conn:
                        conn.close()
            
            # Start the background processing task
            socketio.start_background_task(process_message_background)
                    
        except Exception as e:
            logging.error(f"Unexpected error in handle_storybeat: {str(e)}", exc_info=True)
            emit('error', {'error': f"Unexpected server error: {str(e)}"})
    
    return socketio
        
# Optional ASGI wrapper for ASGI servers
asgi_app = None

def init_asgi():
    global asgi_app
    app = create_flask_app()
    socketio = create_socketio(app)
    asgi_app = WsgiToAsgi(app)
    return asgi_app

def schedule_npc_learning_cycles():
    """Schedule regular NPC learning cycles for all active game sessions"""
    logging.info("Starting NPC learning cycle scheduler")
    try:
        while True:
            # Run every 15 minutes (900 seconds)
            socketio.sleep(900)
            
            try:
                # Get active conversations
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id 
                    FROM conversations 
                    WHERE last_active > NOW() - INTERVAL '1 day'
                """)
                conversations = cursor.fetchall()
                conn.close()
                
                for conversation_id, user_id in conversations:
                    try:
                        # Get NPCs in this conversation
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT npc_id 
                            FROM NPCStats 
                            WHERE user_id = %s AND conversation_id = %s
                        """, (user_id, conversation_id))
                        npc_ids = [row[0] for row in cursor.fetchall()]
                        conn.close()
                        
                        if npc_ids:
                            # Process learning cycle in background
                            async def process_learning_cycle():
                                try:
                                    manager = NPCLearningManager(user_id, conversation_id)
                                    await manager.initialize()
                                    result = await manager.run_regular_adaptation_cycle(npc_ids)
                                    logging.info(f"NPC learning cycle for conversation {conversation_id}: {len(npc_ids)} NPCs processed")
                                except Exception as e:
                                    logging.error(f"Error in NPC learning cycle for conversation {conversation_id}: {e}")
                            
                            # Run the learning cycle asynchronously
                            import asyncio
                            asyncio.run(process_learning_cycle())
                    
                    except Exception as e:
                        logging.error(f"Error processing NPCs for conversation {conversation_id}: {e}")
                
            except Exception as e:
                logging.error(f"Error in NPC learning scheduler: {e}")
                
    except Exception as e:
        logging.error(f"NPC learning scheduler crashed: {e}")
        
async def initialize_conversation_state(conversation_id):
    """Initialize conversation state when a client joins."""
    try:
        # Get conversation details
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            row = await conn.fetchrow("""
                SELECT user_id, status 
                FROM conversations 
                WHERE id = $1
            """, conversation_id)
            
            if not row:
                logging.error(f"Conversation {conversation_id} not found")
                return
                
            user_id, status = row
            
            # If conversation is not ready, initialize it
            if status != 'ready':
                from new_game_agent import NewGameAgent
                agent = NewGameAgent()
                await agent.process_new_game(user_id, {"conversation_id": conversation_id})
                
                # Update status
                await conn.execute("""
                    UPDATE conversations 
                    SET status = 'ready'
                    WHERE id = $1
                """, conversation_id)
                
        finally:
            await conn.close()
            
    except Exception as e:
        logging.error(f"Error initializing conversation state: {e}", exc_info=True)

if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    port = int(os.getenv("PORT", 5000))
    app = create_flask_app()
    socketio = create_socketio(app)
    socketio.run(app, host="0.0.0.0", port=port, debug=False)

async def process_universal_update_with_governance(
    user_id: str,
    conversation_id: str,
    data: dict,
    conn: asyncpg.Connection
) -> dict:
    """
    Process universal updates with governance and context integration.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        data: Update data
        conn: Database connection
        
    Returns:
        Dictionary with update results
    """
    try:
        # 1. Validate and normalize data
        data = normalize_smart_quotes_inplace(data)
        
        # 2. Check governance permissions
        if not await check_governance_permissions(user_id, "universal_update"):
            return {
                "success": False,
                "error": "Insufficient permissions for universal update"
            }
            
        # 3. Get current context
        context = await get_current_context(user_id, conversation_id)
        
        # 4. Apply updates to database
        update_result = await apply_universal_updates_async(
            user_id, conversation_id, data, conn
        )
        
        if not update_result["success"]:
            return update_result
            
        # 5. Update context with changes
        updated_context = await update_context_with_universal_updates(
            context, data, user_id, conversation_id
        )
        
        # 6. Save updated context
        await save_context(user_id, conversation_id, updated_context)
        
        # 7. Report action to governance
        await report_action_to_governance(
            user_id,
            "universal_update",
            {
                "conversation_id": conversation_id,
                "updates_applied": update_result["updates_applied"],
                "errors": update_result.get("errors")
            }
        )
        
        return {
            "success": True,
            "updates_applied": update_result["updates_applied"],
            "context_updated": True,
            "errors": update_result.get("errors")
        }
        
    except Exception as e:
        logging.error(f"Error in process_universal_update_with_governance: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def handle_story_route(
    user_id: str,
    conversation_id: str,
    route_data: dict,
    conn: asyncpg.Connection
) -> dict:
    """
    Handle story route with universal updates and context integration.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        route_data: Route data
        conn: Database connection
        
    Returns:
        Dictionary with route results
    """
    try:
        # 1. Get current context
        context = await get_current_context(user_id, conversation_id)
        
        # 2. Process route data
        route_result = await process_route_data(route_data, context)
        
        # 3. Apply universal updates if present
        if "universal_updates" in route_result:
            update_result = await process_universal_update_with_governance(
                user_id, conversation_id,
                route_result["universal_updates"],
                conn
            )
            
            if not update_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to apply universal updates: {update_result.get('error')}"
                }
                
        # 4. Update context with route changes
        if "context_updates" in route_result:
            updated_context = await update_context_with_universal_updates(
                context,
                route_result["context_updates"],
                user_id,
                conversation_id
            )
            
            # Save updated context
            await save_context(user_id, conversation_id, updated_context)
            
        # 5. Generate response
        response = await generate_route_response(
            route_result,
            context,
            user_id,
            conversation_id
        )
        
        return {
            "success": True,
            "response": response,
            "context_updated": "context_updates" in route_result,
            "universal_updates_applied": "universal_updates" in route_result
        }
        
    except Exception as e:
        logging.error(f"Error in handle_story_route: {e}")
        return {
            "success": False,
            "error": str(e)
        }
