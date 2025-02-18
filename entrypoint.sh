#!/bin/sh
set -e

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec python celery_worker.py
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    # IMPORTANT: specify "-k eventlet" and point to your Flask app object
    exec gunicorn \
      --bind 0.0.0.0:8080 \
      --timeout 600 \
      -k eventlet \
      main:app
fi
