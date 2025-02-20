#!/bin/sh
set -e

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server..."
    PORT=${PORT:-8080}
    # Use the UvicornWorker and reference the ASGI app
    exec gunicorn --bind 0.0.0.0:$PORT --timeout 600 --worker-class uvicorn.workers.UvicornWorker main:asgi_app
fi
