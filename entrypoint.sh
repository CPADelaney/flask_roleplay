#!/bin/sh
set -e

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    # IMPORTANT: specify "-k eventlet" and point to your Flask app object
    exec gunicorn \
      --bind 0.0.0.0:8080 \
      --timeout 600 \
      -k eventlet \
      main:app
fi
