# celery_config.py
from celery import Celery
import os

# Use RabbitMQ as the broker (or override via environment variable)
broker_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost//")

# Create the Celery app with an RPC result backend (or another backend of your choice)
celery_app = Celery('tasks', broker=broker_url, backend='rpc://')

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
