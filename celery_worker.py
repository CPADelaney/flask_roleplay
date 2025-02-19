import eventlet
eventlet.monkey_patch()

from celery_app import celery_app

if __name__ == "__main__":
    # Start the Celery worker directly
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
