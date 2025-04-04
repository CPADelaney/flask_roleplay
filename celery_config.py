from celery import Celery
import os
from celery.schedules import crontab

# Use RabbitMQ as the broker (or override via environment variable)
# Ensure robust default handling if env var isn't set
broker_url = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
result_backend_url = os.getenv("CELERY_RESULT_BACKEND", "rpc://") # Or use redis: "redis://localhost:6379/0"

# Create the Celery app with an RPC result backend (or another backend of your choice)
celery_app = Celery(
    'tasks',
    broker=broker_url,
    backend=result_backend_url,
    include=['tasks'] # Explicitly include the tasks module
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Consider adding task tracking settings if needed
    # task_track_started=True,
)

# Schedule periodic tasks using Celery Beat
celery_app.conf.beat_schedule = {
    # Run NPC learning cycle every 15 minutes
    'npc-learning-cycle-every-15-mins': {
        'task': 'tasks.run_npc_learning_cycle_task', # Task defined in tasks.py
        'schedule': crontab(minute='*/15'), # Run every 15 minutes
        # 'args': (some_arg, another_arg), # Add args if the task needs them
    },
    # Add the memory maintenance task to run daily at 3am
    'nyx-memory-maintenance-daily': {
        'task': 'tasks.nyx_memory_maintenance_task',
        'schedule': crontab(hour=3, minute=0),  # Run at 3 AM daily
    },
    # Memory system maintenance task
    'memory-system-maintenance-daily': {
        'task': 'tasks.memory_maintenance_task',
        'schedule': crontab(hour=4, minute=30),  # Run at 4:30 AM daily
        'options': {'queue': 'low'}
    },
    
    # Memory embedding consolidation
    'memory-embedding-consolidation-weekly': {
        'task': 'tasks.memory_embedding_consolidation_task',
        'schedule': crontab(day_of_week='sun', hour=5, minute=0),  # Run weekly on Sundays at 5 AM
        'options': {'queue': 'low'}
    }
    # Add other periodic tasks here
}

# Optional: If using Django for Celery Beat Scheduler persistence
# celery_app.conf.beat_scheduler = 'django_celery_beat.schedulers:DatabaseScheduler'

if __name__ == '__main__':
    celery_app.start()
