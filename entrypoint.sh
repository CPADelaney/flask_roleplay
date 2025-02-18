#!/bin/sh
set -e

# Override DNS settings using Cloudflare's DNS
echo "nameserver 1.1.1.1" > /etc/resolv.conf

# Debug: Print /etc/resolv.conf contents
echo "===== /etc/resolv.conf ====="
cat /etc/resolv.conf
echo "============================="

# Debug: Test DNS resolution using Python
echo "Attempting to resolve duck.lmq.cloudamqp.com using Python:"
python -c "import socket; print(socket.gethostbyname('duck.lmq.cloudamqp.com'))" || echo "Python DNS lookup failed"

# Debug: Test TCP connectivity to duck.lmq.cloudamqp.com:5671
echo "Testing connectivity to duck.lmq.cloudamqp.com:5671:"
python <<'EOF'
import socket, sys
try:
    sock = socket.create_connection(('duck.lmq.cloudamqp.com', 5671), timeout=10)
    print('Connection successful')
    sock.close()
except Exception as e:
    print('Connection test failed:', e)
EOF

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec su appuser -c "python celery_worker.py"
else
    echo "Starting Web Server (Gunicorn + Eventlet)..."
    exec su appuser -c "gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 600 -k eventlet main:app"
fi
