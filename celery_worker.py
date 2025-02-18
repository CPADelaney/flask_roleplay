# celery_worker.py
import eventlet
eventlet.monkey_patch()  # 1) Done early!

from celery.bin import worker
from main import celery_app  # 2) Now import your app, which may import tasks, etc.

if __name__ == "__main__":
    w = worker.worker()
    # Start Celery with eventlet concurrency:
    w.run_from_argv(["celery", "-A", "main:celery_app", "worker", "-P", "eventlet", "--loglevel=INFO"])
