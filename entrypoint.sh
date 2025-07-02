#!/bin/sh
set -e
cd /app 
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1 -E
else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    exec uvicorn wsgi:app --host 0.0.0.0 --port ${PORT}
fi
