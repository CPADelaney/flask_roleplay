#!/bin/sh
set -e

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker..."
    exec celery -A tasks.celery_app worker --loglevel=info
else
    echo "Starting Web Server..."
    exec gunicorn --bind 0.0.0.0:8080 --timeout 600 --worker-class uvicorn.workers.UvicornWorker main:app
fi
