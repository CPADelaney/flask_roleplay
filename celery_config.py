# celery_config.py

from celery import Celery
import os
from celery.schedules import crontab

from celery import Celery
import os
from celery.schedules import crontab

# Use REDIS as the default broker and result backend!
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create the Celery app
celery_app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks']  # Explicitly include the tasks module
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Uncomment if you want to track task start times
    # task_track_started=True,
)

# Schedule periodic tasks using Celery Beat
celery_app.conf.beat_schedule = {
    # Run NPC learning cycle every 15 minutes
    'npc-learning-cycle-every-15-mins': {
        'task': 'tasks.run_npc_learning_cycle_task',
        'schedule': crontab(minute='*/15'),
        # 'args': ()  # Add args if needed
    },
    'nyx-memory-maintenance-daily': {
        'task': 'tasks.nyx_memory_maintenance_task',
        'schedule': crontab(hour=3, minute=0),
    },
    'memory-system-maintenance-daily': {
        'task': 'tasks.memory_maintenance_task',
        'schedule': crontab(hour=4, minute=30),
        'options': {'queue': 'low'}
    },
    "sweep-and-merge-nyx-split-brains-every-5min": {
        "task": "tasks.sweep_and_merge_nyx_split_brains",
        "schedule": crontab(minute="*/5"),
    },
    'memory-embedding-consolidation-weekly': {
        'task': 'tasks.memory_embedding_consolidation_task',
        'schedule': crontab(day_of_week='sun', hour=5, minute=0),
        'options': {'queue': 'low'}
    }
    # Add other periodic tasks here as needed
}

# Optional: If using Django for Celery Beat Scheduler persistence
# celery_app.conf.beat_scheduler = 'django_celery_beat.schedulers:DatabaseScheduler'

if __name__ == '__main__':
    celery_app.start()
