# wsgi.py

# Import eventlet first and monkey patch
import eventlet
eventlet.monkey_patch()

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the app creation functions
from main import create_flask_app, create_socketio

# Create the Flask app
app = create_flask_app()

# Initialize SocketIO with the app
# Note: We're using the create_socketio function to maintain consistency
socketio = create_socketio(app)

# For local development
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
