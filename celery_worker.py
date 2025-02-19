import eventlet
eventlet.monkey_patch()

import logging
from celery_app import celery_app

if __name__ == "__main__":
    logging.info("Starting Celery worker")
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
