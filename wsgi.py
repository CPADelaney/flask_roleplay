# wsgi.py

import os
import logging
from dotenv import load_dotenv
# Removed unused Quart imports directly here as app comes from main
# from quart import Quart, render_template, session, request, jsonify, redirect
# import socketio
# from quart_cors import cors

import http.server
import threading
import socket
import time

# Global flag and server object
dummy_server_instance = None # Renamed for clarity
server_should_exit = False
DUMMY_SERVER_PORT = int(os.environ.get("PORT", "8080"))

def run_simple_server():
    global dummy_server_instance, server_should_exit # Use the global instance variable
    
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Application initializing...')
            
        def log_message(self, format, *args):
            # Optionally log to the main logger if desired
            # logging.debug(f"Dummy server: {format % args}")
            return  # Keep it silent for now
    
    try:
        server_address = ('', DUMMY_SERVER_PORT)
        # Assign to the global dummy_server_instance
        dummy_server_instance = http.server.HTTPServer(server_address, Handler)
        # SO_REUSEADDR allows immediate reuse of the port if it's in TIME_WAIT state,
        # which is good practice but doesn't solve the "already in use by active listener" problem.
        dummy_server_instance.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Set a timeout on the server socket so handle_request() doesn't block indefinitely
        dummy_server_instance.socket.settimeout(1.0) # Check server_should_exit every 1 second
        logging.info(f"Dummy placeholder server started on port {DUMMY_SERVER_PORT}")
    except OSError as e:
        logging.error(f"Dummy placeholder server could not start on port {DUMMY_SERVER_PORT}: {e}")
        # If the dummy server can't start, it might be that Hypercorn's target port
        # is already in use by *another* external process.
        return # Exit this thread if server creation fails

    # Custom serve loop that checks the exit flag
    while not server_should_exit:
        try:
            dummy_server_instance.handle_request()
        except socket.timeout:
            # This is expected, just continue to check server_should_exit
            continue
        except Exception as e:
            # Log other unexpected errors in the loop
            logging.error(f"Error in dummy server request handling: {e}")
            break # Exit loop on other errors
    
    logging.info("Dummy placeholder server shutting down...")
    if dummy_server_instance:
        dummy_server_instance.server_close() # Properly close the server socket
    logging.info("Dummy placeholder server shut down.")

# --- Main script execution starts here ---

# Configure logging FIRST (ensure format/level is set before app creation or any other logs)
# This needs to be done before any logging.info/error etc. calls to take effect immediately.
# Load environment variables from .env file if it exists
load_dotenv()
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s [%(process)d:%(threadName)s] - %(message)s'
)
logging.info(f"Logging level set to: {log_level_name}")


# Start the dummy server in a separate thread
logging.info("Starting dummy placeholder server thread...")
server_thread = threading.Thread(target=run_simple_server, name="DummyServerThread")
# daemon=True means this thread won't prevent the main program from exiting.
# This is safer if the join() times out or has an issue.
server_thread.daemon = True
server_thread.start()

# Optional: Give the dummy server a very brief moment to actually start and bind.
# This can help avoid a race condition where the main thread signals shutdown
# before the dummy server's HTTPServer has even fully initialized.
# However, the join() later is the more critical synchronization point.
time.sleep(0.1) # Usually not strictly necessary but can be a pragmatic addition

# Import the app creation functions AFTER logging setup
from main import create_quart_app

# Create the quart app instance using the factory
logging.info("Creating quart app instance...")
app = create_quart_app()
logging.info("Quart app instance created.")

# Now, signal the dummy server to stop
logging.info("Signaling dummy placeholder server to stop...")
server_should_exit = True

# Wait for the dummy server thread to finish.
# This is CRUCIAL to ensure the port is freed before Hypercorn (launched externally)
# tries to use it.
logging.info("Waiting for dummy placeholder server thread to join...")
server_thread.join(timeout=5) # Wait up to 5 seconds for the thread to terminate

if server_thread.is_alive():
    logging.warning("Dummy placeholder server thread did not exit in time. Hypercorn might still face port issues if using the same port.")
else:
    logging.info("Dummy placeholder server thread joined successfully. Port should be free.")

logging.info("wsgi.py loaded successfully. Hypercorn should now be able to bind.")
