#!/bin/sh
set -e

# No changes needed for the worker part
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1 -E
else
    echo "Starting Web Server with SocketIO using Hypercorn config file..."
    # The PORT environment variable will be read by hypercorn_config.py
    # Make sure hypercorn_config.py is in /app in the container (if WORKDIR /app)
    exec hypercorn -c /app/hypercorn_config.py wsgi:app
fi
