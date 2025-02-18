#!/bin/sh
set -e

# Attempt to override the container's DNS settings using Cloudflare's DNS
echo "nameserver 1.1.1.1" > /etc/resolv.conf

# Debug: Print current /etc/resolv.conf contents
echo "===== /etc/resolv.conf ====="
cat /etc/resolv.conf
echo "============================="

# Debug: Try resolving CloudAMQP's host
echo "Performing nslookup for duck.lmq.cloudamqp.com:"
nslookup duck.lmq.cloudamqp.com || echo "nslookup failed"

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec su appuser -c "python celery_worker.py"
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    # Use the Railway-provided PORT with a fallback to 8080
    exec su appuser -c "gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 600 -k eventlet main:app"
fi
