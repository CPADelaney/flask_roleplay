# celery_config.py
from celery import Celery
import os

# Use RabbitMQ as the broker.
# The default URL for RabbitMQ (with guest/guest) is: amqp://guest:guest@localhost//
# For the result backend, you can use RPC (which is free) or choose another backend.

broker_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost//")
celery_app = Celery('tasks', broker=broker_url, backend='rpc://')

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
