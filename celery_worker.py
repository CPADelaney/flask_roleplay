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
    try:
        logging.info("Starting dummy server on port 9000")
        eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 9000)), dummy_app)
    except Exception as e:
        logging.exception("Dummy server failed to start: %s", e)


if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    eventlet.spawn(start_dummy_server)
    eventlet.sleep(10)  # Keep the process alive for testing
    # Remove the Celery worker startup temporarily.
