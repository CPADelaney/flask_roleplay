# wsgi.py

# Import eventlet first and monkey patch BEFORE other imports like socketio or flask
import eventlet
eventlet.monkey_patch()

import os
import logging
from dotenv import load_dotenv

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
from main import create_flask_app, create_socketio

# Create the Flask app instance using the factory
logging.info("Creating Flask app instance...")
app = create_flask_app()
logging.info("Flask app instance created.")

# Initialize SocketIO with the app using the factory
# This should also assign the instance to the global `socketio` variable in main.py
logging.info("Creating SocketIO instance...")
socketio_instance = create_socketio(app) # Use the return value if needed, but relies on global assignment in main
logging.info("SocketIO instance created.")

# Gunicorn or other ASGI servers will import 'app' from this module.
# For local dev, main.py's __main__ block runs socketio.run()

logging.info("wsgi.py loaded successfully.")
