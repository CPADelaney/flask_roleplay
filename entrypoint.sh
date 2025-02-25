#!/bin/sh
set -e
if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with concurrency=1..."
    # Use just 'tasks' instead of 'tasks.celery_app'
    exec celery -A tasks worker --loglevel=INFO --concurrency=1
else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    # Use python directly to run the app with socketio
    exec python -c "import os; from main import app, socketio; socketio.run(app, host='0.0.0.0', port=$PORT, allow_unsafe_werkzeug=True)"
fi
