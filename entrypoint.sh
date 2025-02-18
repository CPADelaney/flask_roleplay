#!/bin/sh
set -e

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server..."
    exec gunicorn --bind 0.0.0.0:8080 --timeout 600 --worker-class uvicorn.workers.UvicornWorker main:app
fi
