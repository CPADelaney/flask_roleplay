#!/bin/sh
set -e

# Attempt to force the container to use Google's public DNS
echo "nameserver 8.8.8.8" > /etc/resolv.conf

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec python celery_worker.py
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    exec gunicorn \
      --bind 0.0.0.0:${PORT:-8080} \
      --timeout 600 \
      -k eventlet \
      main:app
fi
