#!/bin/sh
set -e
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1 -E
else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    # Add the --workers=1 flag to ensure a single worker process
    exec hypercorn --workers 1 --worker-class asyncio --bind 0.0.0.0:${PORT} wsgi:app
fi
