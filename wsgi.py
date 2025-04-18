# wsgi.py

import os
import logging
from dotenv import load_dotenv
from quart import Quart, render_template, session, request, jsonify, redirect
import socketio
from quart_cors import cors


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
from main import create_quart_app

# Create the quart app instance using the factory
logging.info("Creating quart app instance...")
app = create_quart_app()
logging.info("quart app instance created.")

app = cors(
  app,
  allow_origin="*",           # or a list/pattern of origins
  allow_credentials=True,     # optional
  allow_methods="*",          # optional
  allow_headers="*"           # optional
)


logging.info("wsgi.py loaded successfully.")
