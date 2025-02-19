import eventlet
eventlet.monkey_patch()

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
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 9000)), dummy_app)

if __name__ == "__main__":
    # Start dummy server in an eventlet green thread
    eventlet.spawn(start_dummy_server)
    # Give it a bit more time to bind (try increasing if needed)
    eventlet.sleep(3)
    
    logging.info("Starting Celery worker")
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
