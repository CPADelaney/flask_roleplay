#!/bin/sh
set -e
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    # Use just 'tasks' instead of 'tasks.celery_app'
    exec celery -A tasks worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    # Use gunicorn with the eventlet worker and point to our new wsgi.py file
    exec gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet --log-level info wsgi:app
fi
