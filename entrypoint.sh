#!/bin/sh
set -e

# Override DNS settings using a public DNS (Cloudflare in this case)
echo "nameserver 1.1.1.1" > /etc/resolv.conf

# Force the broker hostname to resolve to its known IP address.
# (Replace 54.193.232.128 with the actual IP address you tested.)
echo "54.193.232.128 duck.lmq.cloudamqp.com" >> /etc/hosts

# (Optional) Print the files for debugging
echo "===== /etc/resolv.conf ====="
cat /etc/resolv.conf
echo "===== /etc/hosts ====="
cat /etc/hosts

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec su appuser -c "python celery_worker.py"
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    exec su appuser -c "gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 600 -k eventlet main:app"
fi
