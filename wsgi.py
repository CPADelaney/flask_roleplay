# wsgi.py

import os
import logging
from dotenv import load_dotenv
from quart import Quart, render_template, session, request, jsonify, redirect
import socketio
from quart_cors import CORS


# Load environment variables from .env file if it exists (good for Gunicorn)
load_dotenv()

# Configure logging (ensure format/level is set before app creation)
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s [%(process)d:%(threadName)s] - %(message)s' # Added process ID
)
logging.info(f"Logging level set to: {log_level_name}")

# Import the app creation functions AFTER patching and logging setup
from main import create_quart_app, create_socketio

# Create the quart app instance using the factory
logging.info("Creating quart app instance...")
app = create_quart_app()
logging.info("quart app instance created.")

# Initialize SocketIO with the app using the factory
# This should also assign the instance to the global `socketio` variable in main.py
logging.info("Creating SocketIO instance...")
sio = create_socketio(app)
logging.info("SocketIO instance created.")

# Gunicorn or other ASGI servers will import 'app' from this module.
# For local dev, main.py's __main__ block runs socketio.run()

logging.info("wsgi.py loaded successfully.")
