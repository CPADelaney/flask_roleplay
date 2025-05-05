# wsgi.py

import os
import logging
from dotenv import load_dotenv
from quart import Quart, render_template, session, request, jsonify, redirect
import socketio
from quart_cors import cors

import threading
import http.server
import socketserver

def start_dummy_server():
    """Start a minimal HTTP server on port 8080 to satisfy port scanners"""
    try:
        handler = http.server.SimpleHTTPRequestHandler
        dummy_server = socketserver.TCPServer(("", 8080), handler)
        print("Started dummy server on port 8080")
        # Set a short timeout for socket operations
        dummy_server.socket.settimeout(0.5)
        # Serve until the main app is ready
        dummy_server.serve_forever()
    except Exception as e:
        print(f"Dummy server error: {e}")

# Start the dummy server in a separate thread BEFORE app creation
dummy_server_thread = threading.Thread(target=start_dummy_server, daemon=True)
dummy_server_thread.start()

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

logging.info("wsgi.py loaded successfully.")
