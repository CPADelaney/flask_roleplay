#!/bin/sh
set -e

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server..."
    # Use the Render provided PORT environment variable, fallback to 8080 if not set.
    PORT=${PORT:-8080}
    exec gunicorn --bind 0.0.0.0:$PORT --timeout 600 --worker-class uvicorn.workers.UvicornWorker main:app
fi
