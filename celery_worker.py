import eventlet
eventlet.monkey_patch()

import threading
import logging
from flask import Flask
from celery_app import celery_app

# Create a minimal dummy Flask app to serve on port 9000
dummy_app = Flask("dummy")

@dummy_app.route("/")
def dummy():
    return "Worker dummy endpoint", 200

def start_dummy_server():
    logging.info("Starting dummy server on port 9000")
    # Use eventlet's WSGI server to listen on 0.0.0.0:9000
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 9000)), dummy_app)

if __name__ == "__main__":
    # Start the dummy server in a background daemon thread
    t = threading.Thread(target=start_dummy_server, daemon=True)
    t.start()
    
    logging.info("Starting Celery worker")
    # Start the Celery worker directly in the same process
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
