#!/bin/sh
set -e
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    # Use the celery_app explicitly
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1 -E
else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    exec hypercorn --bind 0.0.0.0:${PORT:-8080} wsgi:app
fi
