# wsgi.py
import eventlet
eventlet.monkey_patch()  # Must be the very first import

from main import app, socketio

if __name__ == '__main__':
    socketio.run(app)
