#!/bin/sh
set -e
cd /app 

# Set environment variable to help detect Celery context
export SERVER_SOFTWARE="celery"

if [ "$SERVICE_TYPE" = "worker" ]; then
    echo "Starting Celery Worker with prefork pool and concurrency=1..."
    # Use prefork with concurrency=1 to avoid asyncpg issues
    # The -E flag enables events for monitoring
    exec celery -A tasks.celery_app worker --loglevel=INFO --concurrency=1 --pool=prefork -E
else
    echo "Starting Web Server with SocketIO..."
    PORT=${PORT:-8080}
    exec uvicorn wsgi:app --host 0.0.0.0 --port ${PORT}
fi
