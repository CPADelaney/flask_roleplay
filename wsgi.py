import eventlet
eventlet.monkey_patch()  # This must be the first import before anything else

# Now import the rest
from main import app, socketio

# For local development
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080)
