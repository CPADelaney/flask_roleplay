#!/bin/sh
set -e
# In entrypoint.sh
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    exec celery -A tasks worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server with SocketIO using Gunicorn..."
    PORT=${PORT:-8080}
    # The key change is here - use the socketio instance directly
    exec gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet -w 1 'main:socketio.run(app, host="0.0.0.0", port='$PORT', allow_unsafe_werkzeug=True)'
fi
