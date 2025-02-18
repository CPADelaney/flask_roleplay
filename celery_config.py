# celery_config.py
#import os
#from celery import Celery

#BROKER_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost//")
# Or if you use AMQPS, default to something like:
# BROKER_URL = os.getenv("RABBITMQ_URL", "amqps://guest:guest@localhost:5671//")

#celery_app = Celery('tasks', broker=BROKER_URL, backend='rpc://')

#celery_app.conf.update(
#    task_serializer='json',
#    accept_content=['json'],
#    result_serializer='json',
#    timezone='UTC',
#    enable_utc=True, 
#)
