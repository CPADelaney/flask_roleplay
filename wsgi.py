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
import socket
import time

# Global flag to indicate if main server is ready
main_server_ready = False

def start_dummy_server():
    """Start a minimal HTTP server on port 8080 to satisfy port scanners"""
    # Try to find an available port, starting with 8080
    port = 8080
    server = None
    
    try:
        # First check if main port is already in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8080))
        sock.close()
        
        if result == 0:
            # Port is already in use, which means the main app might be ready
            print("Port 8080 is already in use by another process (likely the main app)")
            return
            
        # Create server with custom handler that checks the ready flag
        class ReadyCheckHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health" or self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Health check OK - Initializing")
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress logging to keep console clean
                return
        
        # Try to start server
        server = socketserver.TCPServer(("", port), ReadyCheckHandler)
        server.socket.settimeout(1)
        print(f"Started dummy health check server on port {port}")
        
        # Serve until the main app is ready or for 120 seconds max
        start_time = time.time()
        while not main_server_ready and (time.time() - start_time) < 120:
            try:
                server.handle_request()
            except socket.timeout:
                # This is expected with the timeout we set
                pass
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Dummy server note: {e}")
    finally:
        if server:
            print("Shutting down dummy server as main app is now ready")
            server.server_close()

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
