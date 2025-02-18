#!/bin/sh
set -e

# Force the container to use Google's public DNS
echo "nameserver 8.8.8.8" > /etc/resolv.conf

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    # Drop privileges to run as appuser
    exec su appuser -c "python celery_worker.py"
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    # Use the Railway-provided PORT (with a fallback of 8080)
    exec su appuser -c "gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 600 -k eventlet main:app"
fi
