# Import eventlet first and monkey patch
import eventlet
eventlet.monkey_patch()

# Now import Flask and create the app
from flask import Flask
from flask_socketio import SocketIO

# Import the app creation function but don't run it yet
from main import create_flask_app, create_socketio

# Create the Flask app
app = create_flask_app()
socketio = create_socketio(app)

# Initialize SocketIO with the app
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='eventlet',
                   logger=True,
                   engineio_logger=True,
                   ping_timeout=60)  # Increased ping timeout

# For local development
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080)
