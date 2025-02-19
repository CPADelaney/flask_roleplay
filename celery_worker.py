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
    import eventlet
    eventlet.spawn(start_dummy_server)
    
    logging.info("Starting Celery worker")
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
