#!/bin/sh
set -e
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    # Use just 'tasks' instead of 'tasks.celery_app'
    exec celery -A tasks worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    # Use the "eventlet" worker class with only 1 worker
    exec gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet --workers 1 --log-level info --timeout 120 wsgi:app
fi
