import eventlet
eventlet.monkey_patch()
import logging
from flask import Flask
from celery_app import celery_app

dummy_app = Flask("dummy")

@dummy_app.route("/")
def dummy():
    return "Worker dummy endpoint", 200

def start_dummy_server():
    logging.info("Starting dummy server on port 9000")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 9000)), dummy_app)

if __name__ == "__main__":
    # Spawn dummy server using eventlet's green thread
    eventlet.spawn(start_dummy_server)
    # Optionally wait a moment to ensure the server starts before proceeding
    eventlet.sleep(1)
    
    logging.info("Starting Celery worker")
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
