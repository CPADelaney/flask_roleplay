# celery_app.py
import os
from celery import Celery
import tasks

def create_celery_app():
    RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")
    celery_app = Celery("my_celery_app", broker=RABBITMQ_URL, backend="rpc://", include=["tasks"])
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        worker_log_format="%(levelname)s:%(name)s:%(message)s",
        worker_redirect_stdouts_level='INFO'
    )
    return celery_app

celery_app = create_celery_app()
