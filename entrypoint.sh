#!/bin/sh
set -e

# Attempt to override the container's DNS settings using Cloudflare's DNS
echo "nameserver 1.1.1.1" > /etc/resolv.conf

# Debug: Print current /etc/resolv.conf contents
echo "===== /etc/resolv.conf ====="
cat /etc/resolv.conf
echo "============================="

# Debug: Use Python to resolve CloudAMQP's host
echo "Attempting to resolve duck.lmq.cloudamqp.com using Python:"
python -c "import socket; print(socket.gethostbyname('duck.lmq.cloudamqp.com'))" || echo "Python DNS lookup failed"

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec su appuser -c "python celery_worker.py"
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    exec su appuser -c "gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 600 -k eventlet main:app"
fi
