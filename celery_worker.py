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
    # Use Eventlet's WSGI server to listen on 0.0.0.0:9000
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 9000)), dummy_app)

if __name__ == "__main__":
    # Spawn the dummy server using eventlet.spawn
    eventlet.spawn(start_dummy_server)
    # Pause briefly to ensure the dummy server has time to bind
    eventlet.sleep(2)
    
    logging.info("Starting Celery worker")
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
