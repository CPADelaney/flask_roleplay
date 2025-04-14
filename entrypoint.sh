#!/bin/sh
set -e

# In entrypoint.sh
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1

else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    # Use the "eventlet" worker class with only 1 worker
    exec gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet --workers 1 --log-level info --timeout 120 wsgi:app
fi
