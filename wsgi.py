# wsgi.py

import os
import logging
from dotenv import load_dotenv
from quart import Quart, render_template, session, request, jsonify, redirect
import socketio
from quart_cors import cors

import http.server
import threading
import socket
import time

# Global flag and server object
dummy_server = None
server_should_exit = False

def run_simple_server():
    global dummy_server, server_should_exit
    
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Application initializing...')
            
        def log_message(self, format, *args):
            return  # Silent
    
    # Create the server with socket reuse option
    dummy_server = http.server.HTTPServer(('', 8080), Handler)
    dummy_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Custom serve loop that checks the exit flag
    while not server_should_exit:
        dummy_server.handle_request()
    
    print("Dummy server shutting down")

# Start in a separate thread
server_thread = threading.Thread(target=run_simple_server)
server_thread.daemon = False
server_thread.start()

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
